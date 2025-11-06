#include "kernels/logsumexp.cuh"
#include "core/types.hpp"
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cfloat>

namespace swiftimpute {
namespace kernels {

// Kernel for log-sum-exp reduction over array
// Uses warp-level primitives for maximum efficiency
__global__ void logsumexp_kernel(
    const prob_t* __restrict__ log_values,
    const uint32_t count,
    prob_t* __restrict__ result
) {
    // Shared memory for block-level reduction
    __shared__ prob_t shared_max[32];  // One per warp
    __shared__ prob_t shared_sum[32];
    
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    int lane = threadIdx.x & 31;
    int warp_id = threadIdx.x >> 5;
    
    // Load value (or -inf if out of bounds)
    prob_t log_val = (tid < count) ? log_values[tid] : -INFINITY;
    
    // Step 1: Find maximum within warp
    prob_t warp_max = log_val;
    for (int offset = 16; offset > 0; offset >>= 1) {
        prob_t other = __shfl_down_sync(0xffffffff, warp_max, offset);
        warp_max = fmaxf(warp_max, other);
    }
    warp_max = __shfl_sync(0xffffffff, warp_max, 0);
    
    // First thread in each warp writes to shared memory
    if (lane == 0) {
        shared_max[warp_id] = warp_max;
    }
    __syncthreads();
    
    // Step 2: First warp finds global maximum
    prob_t block_max = (threadIdx.x < ((blockDim.x + 31) / 32)) ? 
                       shared_max[threadIdx.x] : -INFINITY;
    for (int offset = 16; offset > 0; offset >>= 1) {
        prob_t other = __shfl_down_sync(0xffffffff, block_max, offset);
        block_max = fmaxf(block_max, other);
    }
    block_max = __shfl_sync(0xffffffff, block_max, 0);
    
    // Step 3: Compute exp(log_val - block_max) in each thread
    prob_t exp_val = expf(log_val - block_max);
    
    // Step 4: Sum within warp
    for (int offset = 16; offset > 0; offset >>= 1) {
        exp_val += __shfl_down_sync(0xffffffff, exp_val, offset);
    }
    
    // First thread in each warp writes sum to shared memory
    if (lane == 0) {
        shared_sum[warp_id] = exp_val;
    }
    __syncthreads();
    
    // Step 5: First warp sums all warp sums
    prob_t block_sum = (threadIdx.x < ((blockDim.x + 31) / 32)) ? 
                       shared_sum[threadIdx.x] : 0.0f;
    for (int offset = 16; offset > 0; offset >>= 1) {
        block_sum += __shfl_down_sync(0xffffffff, block_sum, offset);
    }
    
    // Step 6: First thread writes final result
    if (threadIdx.x == 0) {
        result[blockIdx.x] = logf(block_sum) + block_max;
    }
}

// Kernel for element-wise log-sum-exp of two arrays
// result[i] = logsumexp(a[i], b[i])
__global__ void logsumexp_pairwise_kernel(
    const prob_t* __restrict__ a,
    const prob_t* __restrict__ b,
    const uint32_t count,
    prob_t* __restrict__ result
) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    
    if (tid < count) {
        prob_t max_val = fmaxf(a[tid], b[tid]);
        prob_t sum_exp = expf(a[tid] - max_val) + expf(b[tid] - max_val);
        result[tid] = logf(sum_exp) + max_val;
    }
}

// Kernel for log-sum-exp over 2D array along specified dimension
// For HMM state reductions
__global__ void logsumexp_2d_kernel(
    const prob_t* __restrict__ log_values,  // [rows][cols]
    const uint32_t rows,
    const uint32_t cols,
    const int dim,                           // 0 = reduce rows, 1 = reduce cols
    prob_t* __restrict__ result
) {
    __shared__ prob_t shared_data[256];
    
    if (dim == 1) {
        // Reduce along columns (for each row)
        int row = blockIdx.x;
        if (row >= rows) return;
        
        int tid = threadIdx.x;
        const prob_t* row_data = log_values + row * cols;
        
        // Load and find max
        prob_t local_max = -INFINITY;
        for (int i = tid; i < cols; i += blockDim.x) {
            local_max = fmaxf(local_max, row_data[i]);
        }
        shared_data[tid] = local_max;
        __syncthreads();
        
        // Reduce max in shared memory
        for (int s = blockDim.x / 2; s > 0; s >>= 1) {
            if (tid < s) {
                shared_data[tid] = fmaxf(shared_data[tid], shared_data[tid + s]);
            }
            __syncthreads();
        }
        prob_t row_max = shared_data[0];
        __syncthreads();
        
        // Compute exp and sum
        prob_t local_sum = 0.0f;
        for (int i = tid; i < cols; i += blockDim.x) {
            local_sum += expf(row_data[i] - row_max);
        }
        shared_data[tid] = local_sum;
        __syncthreads();
        
        // Reduce sum in shared memory
        for (int s = blockDim.x / 2; s > 0; s >>= 1) {
            if (tid < s) {
                shared_data[tid] += shared_data[tid + s];
            }
            __syncthreads();
        }
        
        if (tid == 0) {
            result[row] = logf(shared_data[0]) + row_max;
        }
    } else {
        // Reduce along rows (for each column)
        int col = blockIdx.x;
        if (col >= cols) return;
        
        int tid = threadIdx.x;
        
        // Load and find max
        prob_t local_max = -INFINITY;
        for (int i = tid; i < rows; i += blockDim.x) {
            local_max = fmaxf(local_max, log_values[i * cols + col]);
        }
        shared_data[tid] = local_max;
        __syncthreads();
        
        // Reduce max
        for (int s = blockDim.x / 2; s > 0; s >>= 1) {
            if (tid < s) {
                shared_data[tid] = fmaxf(shared_data[tid], shared_data[tid + s]);
            }
            __syncthreads();
        }
        prob_t col_max = shared_data[0];
        __syncthreads();
        
        // Compute exp and sum
        prob_t local_sum = 0.0f;
        for (int i = tid; i < rows; i += blockDim.x) {
            local_sum += expf(log_values[i * cols + col] - col_max);
        }
        shared_data[tid] = local_sum;
        __syncthreads();
        
        // Reduce sum
        for (int s = blockDim.x / 2; s > 0; s >>= 1) {
            if (tid < s) {
                shared_data[tid] += shared_data[tid + s];
            }
            __syncthreads();
        }
        
        if (tid == 0) {
            result[col] = logf(shared_data[0]) + col_max;
        }
    }
}

// Host wrapper functions

void launch_logsumexp(
    const prob_t* d_log_values,
    uint32_t count,
    prob_t* d_result,
    cudaStream_t stream
) {
    const int block_size = 256;
    const int grid_size = (count + block_size - 1) / block_size;
    
    logsumexp_kernel<<<grid_size, block_size, 0, stream>>>(
        d_log_values,
        count,
        d_result
    );
}

void launch_logsumexp_pairwise(
    const prob_t* d_a,
    const prob_t* d_b,
    uint32_t count,
    prob_t* d_result,
    cudaStream_t stream
) {
    const int block_size = 256;
    const int grid_size = (count + block_size - 1) / block_size;
    
    logsumexp_pairwise_kernel<<<grid_size, block_size, 0, stream>>>(
        d_a,
        d_b,
        count,
        d_result
    );
}

void launch_logsumexp_2d(
    const prob_t* d_log_values,
    uint32_t rows,
    uint32_t cols,
    int dim,
    prob_t* d_result,
    cudaStream_t stream
) {
    const int block_size = 256;
    const int grid_size = (dim == 1) ? rows : cols;
    
    logsumexp_2d_kernel<<<grid_size, block_size, 0, stream>>>(
        d_log_values,
        rows,
        cols,
        dim,
        d_result
    );
}

} // namespace kernels
} // namespace swiftimpute
