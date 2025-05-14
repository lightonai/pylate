import os
import pathlib

from torch.utils.cpp_extension import load

from pylate.indexes.stanford_nlp.infra.config.config import ColBERTConfig
from pylate.indexes.stanford_nlp.search.strided_tensor import StridedTensor
from pylate.indexes.stanford_nlp.utils.utils import print_message


def try_load_torch_extensions(use_gpu):
    print_message(
        "Loading segmented_maxsim_cpp extension (set COLBERT_LOAD_TORCH_EXTENSION_VERBOSE=True for more info)..."
    )
    segmented_maxsim_cpp = load(
        name="segmented_maxsim_cpp",
        sources=[
            os.path.join(
                pathlib.Path(__file__).parent.resolve(), "segmented_maxsim.cpp"
            ),
        ],
        extra_cflags=["-O3"],
        verbose=os.getenv("COLBERT_LOAD_TORCH_EXTENSION_VERBOSE", "False") == "True",
    )

    return segmented_maxsim_cpp.segmented_maxsim_cpp


# TODO: The masking below might also be applicable in the kNN part
def colbert_score_reduce(scores_padded, D_mask, config: ColBERTConfig):
    D_padding = ~D_mask.view(scores_padded.size(0), scores_padded.size(1)).bool()
    scores_padded[D_padding] = -9999
    scores = scores_padded.max(1).values


    return scores.sum(-1)


# TODO: Wherever this is called, pass `config=`
def colbert_score(Q, D_padded, D_mask, config=ColBERTConfig()):
    """
    Supply sizes Q = (1 | num_docs, *, dim) and D = (num_docs, *, dim).
    If Q.size(0) is 1, the matrix will be compared with all passages.
    Otherwise, each query matrix will be compared against the *aligned* passage.

    EVENTUALLY: Consider masking with -inf for the maxsim (or enforcing a ReLU).
    """

    use_gpu = config.total_visible_gpus > 0
    if use_gpu:
        Q, D_padded, D_mask = Q.cuda(), D_padded.cuda(), D_mask.cuda()

    assert Q.dim() == 3, Q.size()
    assert D_padded.dim() == 3, D_padded.size()
    assert Q.size(0) in [1, D_padded.size(0)]

    scores = D_padded @ Q.to(dtype=D_padded.dtype).permute(0, 2, 1)

    return colbert_score_reduce(scores, D_mask, config)


_segmented_maxsim = None


def colbert_score_packed(Q, D_packed, D_lengths, config=ColBERTConfig()):
    """
    Works with a single query only.
    """

    use_gpu = config.total_visible_gpus > 0

    if use_gpu:
        Q, D_packed, D_lengths = Q.cuda(), D_packed.cuda(), D_lengths.cuda()

    Q = Q.squeeze(0)

    assert Q.dim() == 2, Q.size()
    assert D_packed.dim() == 2, D_packed.size()

    scores = D_packed @ Q.to(dtype=D_packed.dtype).T

    if use_gpu or config.interaction == "flipr":
        scores_padded, scores_mask = StridedTensor(
            scores, D_lengths, use_gpu=use_gpu
        ).as_padded_tensor()

        return colbert_score_reduce(scores_padded, scores_mask, config)
    else:
        global _segmented_maxsim
        if _segmented_maxsim is None:
            _segmented_maxsim = try_load_torch_extensions(use_gpu)

        # _ = ColBERT(config)
        return _segmented_maxsim(scores, D_lengths)
