import os

import torch
import tqdm

from pylate.indexes.stanford_nlp.indexing.loaders import load_doclens
from pylate.indexes.stanford_nlp.utils.utils import flatten, print_message


def optimize_ivf(orig_ivf, orig_ivf_lengths, index_path, verbose: int = 3):
    if verbose > 1:
        print_message("#> Optimizing IVF to store map from centroids to list of pids..")

        print_message("#> Building the emb2pid mapping..")
    all_doclens = load_doclens(index_path, flatten=False)

    # assert self.num_embeddings == sum(flatten(all_doclens))

    all_doclens = flatten(all_doclens)
    total_num_embeddings = sum(all_doclens)

    emb2pid = torch.zeros(total_num_embeddings, dtype=torch.int)

    """
    EVENTUALLY: Use two tensors. emb2pid_offsets will have every 256th element.
    emb2pid_delta will have the delta from the corresponding offset,
    """

    offset_doclens = 0
    for pid, dlength in enumerate(all_doclens):
        emb2pid[offset_doclens : offset_doclens + dlength] = pid
        offset_doclens += dlength

    if verbose > 1:
        print_message("len(emb2pid) =", len(emb2pid))

    ivf = emb2pid[orig_ivf]
    unique_pids_per_centroid = []
    ivf_lengths = []

    offset = 0
    for length in tqdm.tqdm(orig_ivf_lengths.tolist()):
        pids = torch.unique(ivf[offset : offset + length])
        unique_pids_per_centroid.append(pids)
        ivf_lengths.append(pids.shape[0])
        offset += length
    ivf = torch.cat(unique_pids_per_centroid)
    ivf_lengths = torch.tensor(ivf_lengths)

    max_stride = ivf_lengths.max().item()
    zero = torch.zeros(1, dtype=torch.long, device=ivf_lengths.device)
    offsets = torch.cat((zero, torch.cumsum(ivf_lengths, dim=0)))
    inner_dims = ivf.size()[1:]

    if offsets[-2] + max_stride > ivf.size(0):
        padding = torch.zeros(
            max_stride, *inner_dims, dtype=ivf.dtype, device=ivf.device
        )
        ivf = torch.cat((ivf, padding))

    original_ivf_path = os.path.join(index_path, "ivf.pt")
    optimized_ivf_path = os.path.join(index_path, "ivf.pid.pt")
    torch.save((ivf, ivf_lengths), optimized_ivf_path)
    if verbose > 1:
        print_message(f"#> Saved optimized IVF to {optimized_ivf_path}")
        if os.path.exists(original_ivf_path):
            print_message(
                f'#> Original IVF at path "{original_ivf_path}" can now be removed'
            )

    return ivf, ivf_lengths
