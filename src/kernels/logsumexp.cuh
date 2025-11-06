#pragma once

#include "core/types.hpp"
#include <cuda_runtime.h>

namespace swiftimpute {
namespace kernels {

// Log-sum-exp operation for numerical stability
// Computes log(sum(exp(log_values))) without overflow/underflow

// Single array reduction
// Computes single log-sum-exp value from array of log-values
void launch_logsumexp(
    const prob_t* d_log_values,
    uint32_t count,
    prob_t* d_result,
    cudaStream_t stream = 0
);

// Pairwise log-sum-exp
// result[i] = logsumexp(a[i], b[i])
void launch_logsumexp_pairwise(
    const prob_t* d_a,
    const prob_t* d_b,
    uint32_t count,
    prob_t* d_result,
    cudaStream_t stream = 0
);

// 2D array reduction
// Reduces along specified dimension (0 = rows, 1 = cols)
void launch_logsumexp_2d(
    const prob_t* d_log_values,    // [rows][cols]
    uint32_t rows,
    uint32_t cols,
    int dim,                        // 0 or 1
    prob_t* d_result,              // [cols] if dim=0, [rows] if dim=1
    cudaStream_t stream = 0
);

// Device functions (for use within other kernels)

// Warp-level log-sum-exp reduction
// All threads in warp must call with same active mask
__device__ inline prob_t warp_logsumexp(prob_t log_val, uint32_t mask = 0xffffffff) {
    // Find max within warp
    prob_t max_val = log_val;
    for (int offset = 16; offset > 0; offset >>= 1) {
        prob_t other = __shfl_down_sync(mask, max_val, offset);
        max_val = fmaxf(max_val, other);
    }
    max_val = __shfl_sync(mask, max_val, 0);
    
    // Compute exp(log_val - max_val) and sum
    prob_t exp_val = expf(log_val - max_val);
    for (int offset = 16; offset > 0; offset >>= 1) {
        exp_val += __shfl_down_sync(mask, exp_val, offset);
    }
    exp_val = __shfl_sync(mask, exp_val, 0);
    
    // Return log(sum(exp)) + max
    return logf(exp_val) + max_val;
}

// Log-sum-exp of two values (simple case)
__device__ inline prob_t logsumexp2(prob_t a, prob_t b) {
    prob_t max_val = fmaxf(a, b);
    return logf(expf(a - max_val) + expf(b - max_val)) + max_val;
}

// Log-sum-exp of three values
__device__ inline prob_t logsumexp3(prob_t a, prob_t b, prob_t c) {
    prob_t max_val = fmaxf(fmaxf(a, b), c);
    return logf(expf(a - max_val) + expf(b - max_val) + expf(c - max_val)) + max_val;
}

} // namespace kernels
} // namespace swiftimpute
