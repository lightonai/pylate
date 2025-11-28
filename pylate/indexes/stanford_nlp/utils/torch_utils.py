"""
Torch limits quantile operation to 2^24 elements, this can be problematic for colbert indexing with larger embedding dims while computing the codebooks.
"""

from typing import Optional

import torch
from torch import Tensor


def torch_quantile(
    input: Tensor,
    q: float | Tensor,
    dim: Optional[int] = None,
    keepdim: bool = False,
    *,
    interpolation: str = "linear",
    out: Optional[Tensor] = None,
) -> Tensor:
    """Batched torch.quantile replacement that avoids sorting the whole axis (works via kthvalue).
    Supports interpolation: 'linear', 'lower', 'higher', 'midpoint', 'nearest'.
    Signature matches torch.quantile (PyTorch >=2.9.0).
    Note: This uses repeated torch.kthvalue calls for unique indices.
    """
    if (
        dim is not None
    ):  # TODO: Multiple dims will work but not tested rigorously so better not use it for now.
        assert isinstance(dim, int), "Currently only a single dimension is supported :)"

    if interpolation not in {"linear", "lower", "higher", "midpoint", "nearest"}:
        raise ValueError(f"unsupported interpolation: {interpolation}")

    # Normalize q to 1-D tensor on same device
    if isinstance(q, (float, int)):
        q_t = torch.tensor([float(q)], dtype=torch.double, device=input.device)
        scalar_q = True
    else:
        q_t = torch.as_tensor(q, dtype=torch.double, device=input.device).flatten()
        scalar_q = False

    if (q_t < 0).any() or (q_t > 1).any():
        raise ValueError("q must be in [0, 1]")

    # If dim is None, flatten
    if dim is None:
        input_ = input.flatten()
        dim = 0
    else:
        input_ = input

    n = input_.size(dim)
    if n == 0:
        raise ValueError("quantile of an empty tensor is undefined")

    # positions in [0, n-1]
    pos = (n - 1) * q_t  # double
    i0 = torch.floor(pos).to(torch.long).clamp(min=0, max=n - 1)  # lower index
    i1 = torch.ceil(pos).to(torch.long).clamp(min=0, max=n - 1)  # higher index

    # Determine which indices we'll need to select
    if interpolation == "lower":
        indices_needed = i0.unique()
    elif interpolation == "higher":
        indices_needed = i1.unique()
    elif interpolation == "nearest":
        # nearest uses round(pos)
        i_nearest = torch.clamp(torch.round(pos).to(torch.long), 0, n - 1)
        indices_needed = i_nearest.unique()
    else:
        # linear or midpoint: need both i0 and i1 (but if equal, one is enough)
        indices_needed = torch.cat([i0, i1]).unique()

    # Compute kthvalue for each unique index (kthvalue's k is 1-based)
    # We'll store mapping from index -> tensor of shape input_.shape without dim
    indices_needed = indices_needed.to(torch.long)
    # Sort indices_needed for deterministic ordering (not necessary but nicer)
    indices_needed_sorted, _ = torch.sort(indices_needed)
    mapping = {}
    for idx in indices_needed_sorted.tolist():
        k = int(idx) + 1
        vals, _ = input_.kthvalue(k, dim=dim, keepdim=False)
        # vals has shape input_.shape without 'dim'
        mapping[idx] = vals

    # helper to pick a batch of index tensors and stack them in the q-order
    def gather_for(index_tensor: Tensor):
        # index_tensor shape: (m,) where m = number of q's
        # returns tensor shape (m, *rest) where rest = input_.shape without dim
        pieces = [mapping[int(x.item())] for x in index_tensor]
        stacked = torch.stack(pieces, dim=0)  # (m, *rest)
        return stacked

    # Gather required arrays
    if interpolation == "lower":
        res_stack = gather_for(i0)
    elif interpolation == "higher":
        res_stack = gather_for(i1)
    elif interpolation == "nearest":
        i_nearest = torch.clamp(torch.round(pos).to(torch.long), 0, n - 1)
        res_stack = gather_for(i_nearest)
    elif interpolation == "midpoint":
        v0 = gather_for(i0).to(dtype=torch.double)
        v1 = gather_for(i1).to(dtype=torch.double)
        res_stack = (v0 + v1) / 2.0
    else:  # linear
        v0 = gather_for(i0).to(dtype=torch.double)
        v1 = gather_for(i1).to(dtype=torch.double)
        frac = pos - i0.to(dtype=torch.double)
        # reshape frac to (m, 1, 1, ..., 1) so it broadcasts over v0/v1 shape
        expand_shape = [frac.shape[0]] + [1] * (v0.dim() - 1)
        frac = frac.view(*expand_shape)
        res_stack = v0 + (v1 - v0) * frac

    # res_stack has shape (m, *rest) where rest = input_.shape with dim removed
    # Now move the q-axis into the place of the reduced dim so shapes match torch.quantile
    D = input_.dim()
    # stacked axes: 0 -> q, 1.. -> original dims except 'dim'
    perm = list(range(1, dim + 1)) + [0] + list(range(dim + 1, D))
    res_permuted = res_stack.permute(
        perm
    )  # shape: original.shape[:dim] + (m,) + original.shape[dim+1:]

    # If q was scalar, collapse that axis
    if scalar_q:
        res_final = res_permuted.select(dim=dim, index=0)
    else:
        # If q had original multi-dim shape, we should reshape to insert q's original shape
        # q_t was flattened to (m,), so we expand to that ordering.
        res_final = res_permuted

    # If keepdim, insert a dim at 'dim' (size m or 1 depending)
    if keepdim:
        if scalar_q:
            res_final = res_final.unsqueeze(dim)
        else:
            res_final = res_permuted
            # currently shape already contains q as axis; but for keepdim True torch.quantile places q-dim at 'dim' with same size as q
            # When keepdim=True, torch.quantile keeps reduced dim; semantics for multi-q:
            # torch.quantile(..., keepdim=True) returns shape where the reduced dim has size=len(q)
            # Our res_permuted already has the axis at position 'dim' so it's correct.

    # Cast back to input dtype if we computed double for interpolation
    if res_final.dtype != input.dtype:
        # Only cast if original dtype was a floating or integral; keep whatever torch.quantile would do.
        try:
            res_final = res_final.to(dtype=input.dtype)
        except Exception:
            # fallback: leave as-is TODO: Handle edge cases
            pass

    if (not scalar_q) and dim > 0:  # reshape to match ordering
        res_final = res_final.permute(
            dim, *[i for i in range(res_final.ndim) if i != dim]
        )

    if out is not None:
        out.copy_(res_final)
        return out

    return res_final
