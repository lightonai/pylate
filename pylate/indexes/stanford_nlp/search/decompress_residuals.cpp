// decompress_residuals.cpp
#include <torch/extension.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <thread>
#include <vector>

// ---------------------------------------------------------------------
// Thread-local data ----------------------------------------------------
// ---------------------------------------------------------------------
struct decompress_args_t {
    int tid;
    int nthreads;

    int npids;
    int dim;
    int packed_dim;
    int npacked_vals_per_byte;

    const int*      pids;
    const int64_t*  lengths;
    const int64_t*  offsets;
    const float*    bucket_weights;
    const uint8_t*  reversed_bit_map;
    const uint8_t*  bucket_weight_combinations;
    const uint8_t*  binary_residuals;
    const int*      codes;
    const float*    centroids;
    const int64_t*  cumulative_lengths;

    float*          output;
};

// ---------------------------------------------------------------------
// Worker thread -------------------------------------------------------
// ---------------------------------------------------------------------
static void decompress_worker(decompress_args_t* a)
{
    const int npids_per_thread =
        static_cast<int>(std::ceil(static_cast<float>(a->npids) / a->nthreads));
    const int start = a->tid * npids_per_thread;
    const int end   = std::min((a->tid + 1) * npids_per_thread, a->npids);

    for (int i = start; i < end; ++i) {
        const int pid    = a->pids[i];
        const int64_t off = a->offsets[pid];

        for (int j = 0; j < a->lengths[pid]; ++j) {
            const int code = a->codes[off + j];

            for (int k = 0; k < a->packed_dim; ++k) {
                uint8_t x = a->binary_residuals[(off + j) * a->packed_dim + k];
                x = a->reversed_bit_map[x];

                for (int l = 0; l < a->npacked_vals_per_byte; ++l) {
                    const int out_idx  = k * a->npacked_vals_per_byte + l;
                    const int bw_idx   =
                        a->bucket_weight_combinations[x * a->npacked_vals_per_byte + l];

                    a->output[(a->cumulative_lengths[i] + j) * a->dim + out_idx] =
                        a->bucket_weights[bw_idx] +
                        a->centroids[code * a->dim + out_idx];
                }
            }
        }
    }
}

// ---------------------------------------------------------------------
// Python-visible entry point ------------------------------------------
// ---------------------------------------------------------------------
torch::Tensor decompress_residuals(const torch::Tensor pids,
                                   const torch::Tensor lengths,
                                   const torch::Tensor offsets,
                                   const torch::Tensor bucket_weights,
                                   const torch::Tensor reversed_bit_map,
                                   const torch::Tensor bucket_weight_combinations,
                                   const torch::Tensor binary_residuals,
                                   const torch::Tensor codes,
                                   const torch::Tensor centroids,
                                   int dim,
                                   int nbits)
{
    const int npacked_vals_per_byte = 8 / nbits;
    const int packed_dim            = dim / npacked_vals_per_byte;
    const int npids                 = pids.size(0);

    // raw pointers ----------------------------------------------------
    const int*     pids_a                     = pids.data_ptr<int>();
    const int64_t* lengths_a                  = lengths.data_ptr<int64_t>();
    const int64_t* offsets_a                  = offsets.data_ptr<int64_t>();
    const float*   bucket_weights_a           = bucket_weights.data_ptr<float>();
    const uint8_t* reversed_bit_map_a         = reversed_bit_map.data_ptr<uint8_t>();
    const uint8_t* bucket_weight_combinations_a =
        bucket_weight_combinations.data_ptr<uint8_t>();
    const uint8_t* binary_residuals_a         = binary_residuals.data_ptr<uint8_t>();
    const int*     codes_a                    = codes.data_ptr<int>();
    const float*   centroids_a                = centroids.data_ptr<float>();

    // ----------------------------------------------------------------
    // cumulative lengths
    // ----------------------------------------------------------------
    std::vector<int64_t> cumulative_lengths(npids + 1, 0);
    int noutputs = 0;
    for (int i = 0; i < npids; ++i) {
        noutputs += lengths_a[pids_a[i]];
        cumulative_lengths[i + 1] = cumulative_lengths[i] + lengths_a[pids_a[i]];
    }

    // output tensor ---------------------------------------------------
    auto opts   = torch::TensorOptions().dtype(torch::kFloat32).requires_grad(false);
    torch::Tensor output = torch::zeros({noutputs, dim}, opts);
    float* output_a      = output.data_ptr<float>();

    // ----------------------------------------------------------------
    // threading
    // ----------------------------------------------------------------
    const int nthreads = at::get_num_threads();
    std::vector<decompress_args_t> args(nthreads);
    std::vector<std::thread>       workers;
    workers.reserve(nthreads);

    for (int t = 0; t < nthreads; ++t) {
        auto& a = args[t];
        a.tid                     = t;
        a.nthreads                = nthreads;

        a.npids                   = npids;
        a.dim                     = dim;
        a.packed_dim              = packed_dim;
        a.npacked_vals_per_byte   = npacked_vals_per_byte;

        a.pids                    = pids_a;
        a.lengths                 = lengths_a;
        a.offsets                 = offsets_a;
        a.bucket_weights          = bucket_weights_a;
        a.reversed_bit_map        = reversed_bit_map_a;
        a.bucket_weight_combinations = bucket_weight_combinations_a;
        a.binary_residuals        = binary_residuals_a;
        a.codes                   = codes_a;
        a.centroids               = centroids_a;
        a.cumulative_lengths      = cumulative_lengths.data();
        a.output                  = output_a;

        workers.emplace_back(decompress_worker, &a);
    }
    for (auto& w : workers) w.join();

    return output;
}

// ---------------------------------------------------------------------
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    m.def("decompress_residuals_cpp",
          &decompress_residuals,
          "Decompress residuals",
          py::call_guard<py::gil_scoped_release>());
}
