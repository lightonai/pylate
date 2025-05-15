// segmented_lookup.cpp
#include <torch/extension.h>

#include <algorithm>
#include <cassert>
#include <cstring>
#include <mutex>
#include <numeric>
#include <queue>
#include <thread>
#include <vector>

// ---------------------------------------------------------------------
// Shared data passed to each worker
// ---------------------------------------------------------------------
struct lookup_args_t {
    int tid;
    std::mutex*          mtx;              // protects `queue`
    std::queue<int>*     queue;            // document indices still to process

    int64_t ndocs;
    int64_t noutputs;
    int64_t dim;

    const void*   input;                   // raw pointer to tensor data
    const int64_t* lengths;
    const int64_t* offsets;
    const int64_t* cumulative_lengths;

    void*         output;                  // raw pointer to result tensor
};

// ---------------------------------------------------------------------
// Worker thread -------------------------------------------------------
// ---------------------------------------------------------------------
template <typename T>
static void lookup_worker(lookup_args_t* a)
{
    const int64_t* lengths            = a->lengths;
    const int64_t* cumulative_lengths = a->cumulative_lengths;
    const int64_t* offsets            = a->offsets;
    const int64_t  dim                = a->dim;

    const T* input  = static_cast<const T*>(a->input);
    T*       output = static_cast<T*>(a->output);

    while (true) {
        int i;
        {   // ── critical section ─────────────────────────────────────
            std::lock_guard<std::mutex> lock(*a->mtx);
            if (a->queue->empty()) break;
            i = a->queue->front();
            a->queue->pop();
        }   // mutex released here

        std::memcpy(output + cumulative_lengths[i] * dim,
                    input  + offsets[i]            * dim,
                    lengths[i] * dim * sizeof(T));
    }
}

// ---------------------------------------------------------------------
// Implementation template --------------------------------------------
// ---------------------------------------------------------------------
template <typename T>
static torch::Tensor segmented_lookup_impl(const torch::Tensor input,
                                           const torch::Tensor pids,
                                           const torch::Tensor lengths,
                                           const torch::Tensor offsets)
{
    const int64_t* lengths_a = lengths.data_ptr<int64_t>();
    const int64_t* offsets_a = offsets.data_ptr<int64_t>();

    const int64_t ndocs     = pids.size(0);
    const int64_t noutputs  = std::accumulate(lengths_a, lengths_a + ndocs, 0LL);
    const int     nthreads  = at::get_num_threads();

    int64_t dim;
    torch::Tensor output;
    if (input.dim() == 1) {
        dim    = 1;
        output = torch::zeros({noutputs}, input.options());
    } else {
        TORCH_CHECK(input.dim() == 2, "input must be 1-D or 2-D");
        dim    = input.size(1);
        output = torch::zeros({noutputs, dim}, input.options());
    }

    // prefix-sum cumulative lengths
    std::vector<int64_t> cumulative(ndocs + 1, 0);
    std::partial_sum(lengths_a, lengths_a + ndocs, cumulative.begin() + 1);

    //------------------------------------------------------------------
    // Thread pool
    //------------------------------------------------------------------
    std::mutex              mtx;
    std::queue<int>         work_queue;
    for (int i = 0; i < ndocs; ++i) work_queue.push(i);

    std::vector<lookup_args_t> args(nthreads);
    std::vector<std::thread>   workers;
    workers.reserve(nthreads);

    for (int t = 0; t < nthreads; ++t) {
        auto& a  = args[t];
        a.tid    = t;
        a.mtx    = &mtx;
        a.queue  = &work_queue;
        a.ndocs  = ndocs;
        a.noutputs = noutputs;
        a.dim    = dim;
        a.input  = static_cast<const void*>(input.data_ptr<T>());
        a.lengths = const_cast<int64_t*>(lengths_a);
        a.offsets = const_cast<int64_t*>(offsets_a);
        a.cumulative_lengths = cumulative.data();
        a.output = static_cast<void*>(output.data_ptr<T>());

        workers.emplace_back(lookup_worker<T>, &a);
    }
    for (auto& w : workers) w.join();

    return output;
}

// ---------------------------------------------------------------------
// Public entry point --------------------------------------------------
// ---------------------------------------------------------------------
torch::Tensor segmented_lookup(const torch::Tensor input,
                               const torch::Tensor pids,
                               const torch::Tensor lengths,
                               const torch::Tensor offsets)
{
    switch (input.scalar_type()) {
        case torch::kUInt8:   return segmented_lookup_impl<uint8_t >(input, pids, lengths, offsets);
        case torch::kInt32:   return segmented_lookup_impl<int32_t >(input, pids, lengths, offsets);
        case torch::kInt64:   return segmented_lookup_impl<int64_t >(input, pids, lengths, offsets);
        case torch::kFloat32: return segmented_lookup_impl<float   >(input, pids, lengths, offsets);
        case torch::kFloat16: return segmented_lookup_impl<at::Half>(input, pids, lengths, offsets);
        default: TORCH_CHECK(false, "unsupported dtype for segmented_lookup");
    }
}

// ---------------------------------------------------------------------
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    m.def("segmented_lookup_cpp",
          &segmented_lookup,
          "Segmented lookup",
          py::call_guard<py::gil_scoped_release>());
}
