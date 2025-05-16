// segmented_maxsim.cpp          ← rename if you like
#include <torch/extension.h>

#include <algorithm>
#include <cmath>
#include <numeric>
#include <thread>
#include <vector>

// ---------------------------------------------------------------------
// Data passed to each worker thread
// ---------------------------------------------------------------------
struct max_args_t {
    int tid;
    int nthreads;

    int ndocs;
    int ndoc_vectors;
    int nquery_vectors;

    const int64_t* lengths;      //   input
    const float*   scores;       //   input
    const int64_t* offsets;      //   input

    float* max_scores;           //   output (size = ndocs × nquery_vectors)
};

// ---------------------------------------------------------------------
// Worker ----------------------------------------------------------------
static void max_worker(max_args_t* a)
{
    const int ndocs_per_thread =
        static_cast<int>(std::ceil(static_cast<float>(a->ndocs) / a->nthreads));
    const int start = a->tid * ndocs_per_thread;
    const int end   = std::min((a->tid + 1) * ndocs_per_thread, a->ndocs);

    float*       max_scores_row = a->max_scores + start * a->nquery_vectors;
    const float* scores_ptr     = a->scores +
                                  a->offsets[start] * a->nquery_vectors;

    for (int i = start; i < end; ++i) {
        for (int j = 0; j < a->lengths[i]; ++j) {
            std::transform(max_scores_row,
                           max_scores_row + a->nquery_vectors,
                           scores_ptr,
                           max_scores_row,
                           [](float a, float b) { return std::max(a, b); });

            scores_ptr += a->nquery_vectors;
        }
        max_scores_row += a->nquery_vectors;
    }
}

// ---------------------------------------------------------------------
// Python-visible entry-point ------------------------------------------
// ---------------------------------------------------------------------
torch::Tensor segmented_maxsim(const torch::Tensor scores,
                               const torch::Tensor lengths)
{
    TORCH_CHECK(scores.dim() == 2, "scores must be 2-D (dv × qv)");
    TORCH_CHECK(lengths.dim() == 1, "lengths must be 1-D");

    const auto* lengths_a       = lengths.data_ptr<int64_t>();
    const auto* scores_a        = scores.data_ptr<float>();
    const int   ndocs           = lengths.size(0);
    const int   ndoc_vectors    = scores.size(0);
    const int   nquery_vectors  = scores.size(1);
    const int   nthreads        = at::get_num_threads();

    // output tensor: (ndocs × nquery_vectors) filled with zeros
    torch::Tensor max_scores =
        torch::zeros({ndocs, nquery_vectors}, scores.options());

    // prefix-sum offsets so that offsets[i] = Σ_{k<i} lengths[k]
    std::vector<int64_t> offsets(ndocs + 1);
    offsets[0] = 0;
    std::partial_sum(lengths_a, lengths_a + ndocs, offsets.begin() + 1);

    //------------------------------------------------------------------
    // Launch threads
    //------------------------------------------------------------------
    std::vector<max_args_t> args(nthreads);
    std::vector<std::thread> workers;
    workers.reserve(nthreads);

    for (int t = 0; t < nthreads; ++t) {
        auto& a      = args[t];
        a.tid        = t;
        a.nthreads   = nthreads;
        a.ndocs      = ndocs;
        a.ndoc_vectors   = ndoc_vectors;
        a.nquery_vectors = nquery_vectors;
        a.lengths    = lengths_a;
        a.scores     = scores_a;
        a.offsets    = offsets.data();
        a.max_scores = max_scores.data_ptr<float>();

        workers.emplace_back(max_worker, &a);
    }
    for (auto& w : workers) w.join();

    // return 1-D tensor   (ndocs,)
    return max_scores.sum(/*dim=*/1);
}

// ---------------------------------------------------------------------
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    m.def("segmented_maxsim_cpp",
          &segmented_maxsim,
          "Segmented MaxSim",
          py::call_guard<py::gil_scoped_release>());
}
