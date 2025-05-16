// segmented_lookup.cpp
#include <torch/extension.h>

#include <algorithm>
#include <chrono>
#include <numeric>
#include <queue>
#include <thread>          // ⟵ C++11 threads
#include <unordered_set>
#include <utility>

// ---------------------------------------------------------------------
// Structures ----------------------------------------------------------
// ---------------------------------------------------------------------
struct maxsim_args_t {
    int tid;
    int nthreads;

    int ncentroids;
    int nquery_vectors;
    int npids;

    int*      pids;
    float*    centroid_scores;
    int*      codes;
    int64_t*  doclens;
    int64_t*  offsets;
    bool*     idx;

    std::priority_queue<std::pair<float, int>> approx_scores;
};

// ---------------------------------------------------------------------
// Worker ----------------------------------------------------------------
static void maxsim(maxsim_args_t* a) {
    std::vector<float> per_doc_approx_scores(a->nquery_vectors, -9999.f);

    int ndocs_per_thread =
        static_cast<int>(std::ceil(static_cast<float>(a->npids) / a->nthreads));
    int start = a->tid * ndocs_per_thread;
    int end   = std::min((a->tid + 1) * ndocs_per_thread, a->npids);

    std::unordered_set<int> seen_codes;

    for (int i = start; i < end; ++i) {
        const int pid = a->pids[i];

        for (int j = 0; j < a->doclens[pid]; ++j) {
            int code = a->codes[a->offsets[pid] + j];
            TORCH_CHECK(code < a->ncentroids, "invalid centroid code");

            if (a->idx[code] && !seen_codes.count(code)) {
                for (int k = 0; k < a->nquery_vectors; ++k) {
                    per_doc_approx_scores[k] =
                        std::max(per_doc_approx_scores[k],
                                 a->centroid_scores[code * a->nquery_vectors + k]);
                }
                seen_codes.insert(code);
            }
        }

        float score = std::accumulate(per_doc_approx_scores.begin(),
                                      per_doc_approx_scores.end(), 0.f);
        std::fill(per_doc_approx_scores.begin(), per_doc_approx_scores.end(), -9999.f);

        a->approx_scores.emplace(score, pid);
        seen_codes.clear();
    }
}

// ---------------------------------------------------------------------
// Thread driver --------------------------------------------------------
// ---------------------------------------------------------------------
static std::vector<int> filter_pids_helper(int ncentroids, int nquery_vectors,
                                           int npids, int* pids,
                                           float* centroid_scores, int* codes,
                                           int64_t* doclens, int64_t* offsets,
                                           bool* idx, int nfiltered_docs) {
    const int nthreads = at::get_num_threads();

    std::vector<std::thread> threads;
    std::vector<maxsim_args_t> args(nthreads);

    for (int i = 0; i < nthreads; ++i) {
        maxsim_args_t& a = args[i];
        a.tid            = i;
        a.nthreads       = nthreads;
        a.ncentroids     = ncentroids;
        a.nquery_vectors = nquery_vectors;
        a.npids          = npids;
        a.pids           = pids;
        a.centroid_scores = centroid_scores;
        a.codes          = codes;
        a.doclens        = doclens;
        a.offsets        = offsets;
        a.idx            = idx;

        threads.emplace_back(maxsim, &a);
    }

    for (auto& t : threads) t.join();

    // gather best scores from all threads
    std::priority_queue<std::pair<float, int>> global;
    for (auto& a : args) {
        for (int k = 0; k < nfiltered_docs && !a.approx_scores.empty(); ++k) {
            global.push(a.approx_scores.top());
            a.approx_scores.pop();
        }
    }

    std::vector<int> filtered;
    for (int k = 0; k < nfiltered_docs && !global.empty(); ++k) {
        filtered.push_back(global.top().second);
        global.pop();
    }
    return filtered;
}

// ---------------------------------------------------------------------
// Python-visible entry-point ------------------------------------------
// ---------------------------------------------------------------------
torch::Tensor filter_pids(const torch::Tensor pids,
                          const torch::Tensor centroid_scores,
                          const torch::Tensor codes,
                          const torch::Tensor doclens,
                          const torch::Tensor offsets,
                          const torch::Tensor idx,
                          int nfiltered_docs) {
    const int ncentroids     = centroid_scores.size(0);
    const int nquery_vectors = centroid_scores.size(1);
    const int npids          = pids.size(0);

    auto* pids_a            = pids.data_ptr<int>();
    auto* centroid_scores_a = centroid_scores.data_ptr<float>();
    auto* codes_a           = codes.data_ptr<int>();
    auto* doclens_a         = doclens.data_ptr<int64_t>();
    auto* offsets_a         = offsets.data_ptr<int64_t>();
    auto* idx_a             = idx.data_ptr<bool>();

    // first pass
    std::vector<int> filtered =
        filter_pids_helper(ncentroids, nquery_vectors, npids, pids_a,
                           centroid_scores_a, codes_a, doclens_a, offsets_a,
                           idx_a, nfiltered_docs);

    // second pass (quarter the list as in original code)
    const int nfinal = static_cast<int>(nfiltered_docs / 4);
    std::vector<uint8_t> ones(ncentroids, 1);

    std::vector<int> final_filtered =
        filter_pids_helper(ncentroids, nquery_vectors,
                           static_cast<int>(filtered.size()), filtered.data(),
                           centroid_scores_a, codes_a, doclens_a, offsets_a,
                            reinterpret_cast<bool*>(ones.data()), nfinal);

    auto opts = torch::TensorOptions().dtype(torch::kInt32).requires_grad(false);
    return torch::from_blob(final_filtered.data(),
                            {static_cast<int64_t>(final_filtered.size())},
                            opts)
        .clone();               // copy into a new tensor, owner = PyTorch
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("filter_pids_cpp", &filter_pids,
          "Filter pids",
          py::call_guard<py::gil_scoped_release>());
}
