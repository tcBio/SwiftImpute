#include "forward_backward.cuh"
#include "logsumexp.cuh"
#include "core/types.hpp"
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cmath>
#include <cfloat>

namespace swiftimpute {
namespace kernels {

// CUDA error checking
#define CHECK_CUDA(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            throw std::runtime_error(std::string("CUDA error: ") + cudaGetErrorString(err)); \
        } \
    } while(0)

// Log-space constants
constexpr prob_t LOG_ZERO = -999.0f;
constexpr prob_t LOG_ONE = 0.0f;

/**
 * Forward pass kernel with checkpointing
 *
 * Computes forward probabilities α(m, s) = P(observations[0:m], state=s at marker m)
 * Uses checkpointing to save memory: stores α only at checkpoint markers
 *
 * Algorithm:
 *   α(0, s) = emission(0, s)
 *   α(m, s) = emission(m, s) * Σ_i α(m-1, i) * transition(i, s)
 *
 * Memory:
 *   - emission_probs: [num_samples][num_markers][num_states]
 *   - transition_matrix: [num_markers-1][num_states][num_states]
 *   - forward_checkpoints: [num_samples][num_checkpoints][num_states]
 *   - scaling_factors: [num_samples][num_markers]
 */
__global__ void forward_pass_kernel(
    const prob_t* __restrict__ emission_probs,
    const prob_t* __restrict__ transition_matrix,
    const marker_t* __restrict__ selected_states,
    uint32_t num_markers,
    uint32_t num_states,
    uint32_t checkpoint_interval,
    prob_t* __restrict__ forward_checkpoints,
    prob_t* __restrict__ scaling_factors
) {
    // One block per sample
    uint32_t sample_idx = blockIdx.x;
    uint32_t tid = threadIdx.x;

    // Shared memory for current and previous alpha
    extern __shared__ prob_t shared_mem[];
    prob_t* alpha_curr = shared_mem;
    prob_t* alpha_prev = &shared_mem[num_states];

    // Base offset for this sample
    uint64_t emission_base = static_cast<uint64_t>(sample_idx) * num_markers * num_states;
    uint64_t checkpoint_base = static_cast<uint64_t>(sample_idx) *
                               ((num_markers + checkpoint_interval - 1) / checkpoint_interval) * num_states;

    // Initialize at marker 0
    if (tid < num_states) {
        alpha_curr[tid] = emission_probs[emission_base + tid];
    }
    __syncthreads();

    // Compute scaling factor for marker 0
    if (tid == 0) {
        prob_t max_val = alpha_curr[0];
        for (uint32_t s = 1; s < num_states; ++s) {
            max_val = fmaxf(max_val, alpha_curr[s]);
        }
        scaling_factors[sample_idx * num_markers] = max_val;

        // Apply scaling
        for (uint32_t s = 0; s < num_states; ++s) {
            alpha_curr[s] -= max_val;
        }
    }
    __syncthreads();

    // Save checkpoint at marker 0
    if (tid < num_states) {
        forward_checkpoints[checkpoint_base + tid] = alpha_curr[tid];
    }

    // Forward pass through remaining markers
    for (marker_t m = 1; m < num_markers; ++m) {
        // Swap buffers
        prob_t* temp = alpha_prev;
        alpha_prev = alpha_curr;
        alpha_curr = temp;
        __syncthreads();

        // Each thread computes one state
        if (tid < num_states) {
            uint32_t to_state = tid;

            // Use single transition matrix (state-independent transitions)
            // Compute Σ_i α(m-1, i) * transition(i, to_state)
            prob_t sum = LOG_ZERO;

            for (uint32_t from_state = 0; from_state < num_states; ++from_state) {
                prob_t trans_prob = transition_matrix[from_state * num_states + to_state];
                prob_t val = alpha_prev[from_state] + trans_prob;
                sum = logsumexp2(sum, val);
            }

            // Add emission probability
            uint64_t emission_idx = emission_base + m * num_states + to_state;
            alpha_curr[to_state] = sum + emission_probs[emission_idx];
        }
        __syncthreads();

        // Scaling to prevent underflow
        if (tid == 0) {
            prob_t max_val = alpha_curr[0];
            for (uint32_t s = 1; s < num_states; ++s) {
                max_val = fmaxf(max_val, alpha_curr[s]);
            }
            scaling_factors[sample_idx * num_markers + m] = max_val;

            // Apply scaling
            for (uint32_t s = 0; s < num_states; ++s) {
                alpha_curr[s] -= max_val;
            }
        }
        __syncthreads();

        // Save checkpoint if needed
        if (m % checkpoint_interval == 0) {
            if (tid < num_states) {
                uint32_t checkpoint_idx = m / checkpoint_interval;
                forward_checkpoints[checkpoint_base + checkpoint_idx * num_states + tid] = alpha_curr[tid];
            }
        }
    }
}

/**
 * Backward pass kernel with forward recomputation
 *
 * Computes backward probabilities β(m, s) = P(observations[m+1:M] | state=s at marker m)
 * Recomputes forward probabilities between checkpoints
 * Combines α and β to compute posteriors: P(state=s at marker m | all observations)
 *
 * Algorithm:
 *   β(M-1, s) = 1
 *   β(m, s) = Σ_j transition(s, j) * emission(m+1, j) * β(m+1, j)
 *   posterior(m, s) = α(m, s) * β(m, s) / Σ_s' α(m, s') * β(m, s')
 */
__global__ void backward_pass_kernel(
    const prob_t* __restrict__ emission_probs,
    const prob_t* __restrict__ transition_matrix,
    const marker_t* __restrict__ selected_states,
    const prob_t* __restrict__ forward_checkpoints,
    const prob_t* __restrict__ scaling_factors,
    uint32_t num_markers,
    uint32_t num_states,
    uint32_t checkpoint_interval,
    prob_t* __restrict__ posterior_probs
) {
    uint32_t sample_idx = blockIdx.x;
    uint32_t tid = threadIdx.x;

    extern __shared__ prob_t shared_mem[];
    prob_t* beta_curr = shared_mem;
    prob_t* beta_prev = &shared_mem[num_states];
    prob_t* alpha_recomp = &shared_mem[2 * num_states];

    uint64_t emission_base = static_cast<uint64_t>(sample_idx) * num_markers * num_states;
    uint64_t checkpoint_base = static_cast<uint64_t>(sample_idx) *
                               ((num_markers + checkpoint_interval - 1) / checkpoint_interval) * num_states;
    uint64_t posterior_base = emission_base;

    // Initialize β at last marker: β(M-1, s) = 1 (log space = 0)
    if (tid < num_states) {
        beta_curr[tid] = LOG_ONE;
    }
    __syncthreads();

    // Load last forward checkpoint
    uint32_t last_checkpoint = (num_markers - 1) / checkpoint_interval;
    if (tid < num_states) {
        alpha_recomp[tid] = forward_checkpoints[checkpoint_base + last_checkpoint * num_states + tid];
    }
    __syncthreads();

    // Recompute forward from last checkpoint to last marker
    for (marker_t m = last_checkpoint * checkpoint_interval + 1; m < num_markers; ++m) {
        // Similar to forward pass, but only updating alpha_recomp
        if (tid < num_states) {
            uint32_t to_state = tid;

            prob_t sum = LOG_ZERO;
            for (uint32_t from_state = 0; from_state < num_states; ++from_state) {
                prob_t trans_prob = transition_matrix[from_state * num_states + to_state];
                prob_t val = alpha_recomp[from_state] + trans_prob;
                sum = logsumexp2(sum, val);
            }

            uint64_t emission_idx = emission_base + m * num_states + to_state;
            alpha_recomp[to_state] = sum + emission_probs[emission_idx] -
                                     scaling_factors[sample_idx * num_markers + m];
        }
        __syncthreads();
    }

    // Compute posterior at last marker
    if (tid < num_states) {
        posterior_probs[posterior_base + (num_markers - 1) * num_states + tid] =
            alpha_recomp[tid] + beta_curr[tid];
    }
    __syncthreads();

    // Backward pass through all markers
    for (int m = num_markers - 2; m >= 0; --m) {
        // Swap beta buffers
        prob_t* temp = beta_prev;
        beta_prev = beta_curr;
        beta_curr = temp;
        __syncthreads();

        // Compute β(m, from_state)
        if (tid < num_states) {
            uint32_t from_state = tid;

            prob_t sum = LOG_ZERO;
            for (uint32_t to_state = 0; to_state < num_states; ++to_state) {
                prob_t trans_prob = transition_matrix[from_state * num_states + to_state];
                uint64_t emission_idx = emission_base + (m + 1) * num_states + to_state;
                prob_t val = trans_prob + emission_probs[emission_idx] + beta_prev[to_state];
                sum = logsumexp2(sum, val);
            }

            beta_curr[from_state] = sum;
        }
        __syncthreads();

        // Recompute or load forward probability at this marker
        if (m % checkpoint_interval == 0) {
            // Load from checkpoint
            if (tid < num_states) {
                uint32_t checkpoint_idx = m / checkpoint_interval;
                alpha_recomp[tid] = forward_checkpoints[checkpoint_base + checkpoint_idx * num_states + tid];
            }
        } else {
            // Recompute from previous checkpoint
            uint32_t checkpoint_m = (m / checkpoint_interval) * checkpoint_interval;

            // Load checkpoint
            if (tid < num_states) {
                uint32_t checkpoint_idx = checkpoint_m / checkpoint_interval;
                alpha_recomp[tid] = forward_checkpoints[checkpoint_base + checkpoint_idx * num_states + tid];
            }
            __syncthreads();

            // Recompute forward from checkpoint to current marker
            for (marker_t mm = checkpoint_m + 1; mm <= m; ++mm) {
                if (tid < num_states) {
                    uint32_t to_state = tid;

                    prob_t sum = LOG_ZERO;
                    for (uint32_t from_state = 0; from_state < num_states; ++from_state) {
                        prob_t trans_prob = transition_matrix[from_state * num_states + to_state];
                        prob_t val = alpha_recomp[from_state] + trans_prob;
                        sum = logsumexp2(sum, val);
                    }

                    uint64_t emission_idx = emission_base + mm * num_states + to_state;
                    alpha_recomp[to_state] = sum + emission_probs[emission_idx] -
                                            scaling_factors[sample_idx * num_markers + mm];
                }
                __syncthreads();
            }
        }
        __syncthreads();

        // Compute posterior: α(m) + β(m)
        if (tid < num_states) {
            posterior_probs[posterior_base + m * num_states + tid] =
                alpha_recomp[tid] + beta_curr[tid];
        }
        __syncthreads();
    }

    // Normalize posteriors to sum to 1 (in probability space)
    for (marker_t m = 0; m < num_markers; ++m) {
        __syncthreads();

        // Load posteriors for this marker
        if (tid < num_states) {
            alpha_recomp[tid] = posterior_probs[posterior_base + m * num_states + tid];
        }
        __syncthreads();

        // Compute normalization constant (log-sum-exp)
        if (tid == 0) {
            prob_t log_sum = LOG_ZERO;
            for (uint32_t s = 0; s < num_states; ++s) {
                log_sum = logsumexp2(log_sum, alpha_recomp[s]);
            }
            beta_curr[0] = log_sum;  // Store in shared memory
        }
        __syncthreads();

        // Normalize
        if (tid < num_states) {
            posterior_probs[posterior_base + m * num_states + tid] =
                alpha_recomp[tid] - beta_curr[0];
        }
    }
}

// Host wrapper functions

void launch_forward_pass(
    const prob_t* d_emission_probs,
    const prob_t* d_transition_matrix,
    const marker_t* d_selected_states,
    uint32_t num_samples,
    uint32_t num_markers,
    uint32_t num_states,
    uint32_t checkpoint_interval,
    prob_t* d_forward_checkpoints,
    prob_t* d_scaling_factors,
    cudaStream_t stream
) {
    // Grid: one block per sample
    dim3 grid(num_samples);

    // Block: one thread per state (up to 256 states supported)
    dim3 block(std::min(num_states, 256u));

    // Shared memory: 2 * num_states for current and previous alpha
    size_t shared_mem = 2 * num_states * sizeof(prob_t);

    forward_pass_kernel<<<grid, block, shared_mem, stream>>>(
        d_emission_probs,
        d_transition_matrix,
        d_selected_states,
        num_markers,
        num_states,
        checkpoint_interval,
        d_forward_checkpoints,
        d_scaling_factors
    );

    CHECK_CUDA(cudaGetLastError());
}

void launch_backward_pass(
    const prob_t* d_emission_probs,
    const prob_t* d_transition_matrix,
    const marker_t* d_selected_states,
    const prob_t* d_forward_checkpoints,
    const prob_t* d_scaling_factors,
    uint32_t num_samples,
    uint32_t num_markers,
    uint32_t num_states,
    uint32_t checkpoint_interval,
    prob_t* d_posterior_probs,
    cudaStream_t stream
) {
    dim3 grid(num_samples);
    dim3 block(std::min(num_states, 256u));

    // Shared memory: 2 * num_states for beta, 1 * num_states for alpha recomputation
    size_t shared_mem = 3 * num_states * sizeof(prob_t);

    backward_pass_kernel<<<grid, block, shared_mem, stream>>>(
        d_emission_probs,
        d_transition_matrix,
        d_selected_states,
        d_forward_checkpoints,
        d_scaling_factors,
        num_markers,
        num_states,
        checkpoint_interval,
        d_posterior_probs
    );

    CHECK_CUDA(cudaGetLastError());
}

LaunchConfig calculate_launch_config(
    uint32_t num_samples,
    uint32_t num_states,
    uint32_t num_markers
) {
    LaunchConfig config;
    config.grid_size = dim3(num_samples);
    config.block_size = dim3(std::min(num_states, 256u));
    config.shared_mem_bytes = 3 * num_states * sizeof(prob_t);
    return config;
}

} // namespace kernels
} // namespace swiftimpute
