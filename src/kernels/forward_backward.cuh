#pragma once

#include "core/types.hpp"
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

namespace swiftimpute {
namespace kernels {

// Kernel launch configuration
struct LaunchConfig {
    dim3 grid_size;
    dim3 block_size;
    size_t shared_mem_bytes;
    
    LaunchConfig() : shared_mem_bytes(0) {}
};

// Calculate optimal launch configuration
LaunchConfig calculate_launch_config(
    uint32_t num_samples,
    uint32_t num_states,
    uint32_t num_markers
);

// Forward pass with checkpointing
// Computes forward probabilities and stores checkpoint values
// grid: (num_samples, 1, 1) - one block per sample
// block: (min(num_states, 256), 1, 1) - threads collaborate on states
__global__ void forward_pass_kernel(
    // Inputs
    const prob_t* __restrict__ emission_probs,     // [num_samples][num_markers][num_states]
    const prob_t* __restrict__ transition_matrix,  // [num_states][num_states]
    const marker_t* __restrict__ selected_states,  // [num_markers][num_states]
    const uint32_t num_markers,
    const uint32_t num_states,
    const uint32_t checkpoint_interval,
    // Outputs
    prob_t* __restrict__ forward_checkpoints,      // [num_samples][num_checkpoints][num_states]
    prob_t* __restrict__ scaling_factors           // [num_samples][num_markers]
);

// Backward pass with forward recomputation
// Computes backward probabilities and recomputes forward between checkpoints
// grid: (num_samples, 1, 1)
// block: (min(num_states, 256), 1, 1)
__global__ void backward_pass_kernel(
    // Inputs
    const prob_t* __restrict__ emission_probs,     // [num_samples][num_markers][num_states]
    const prob_t* __restrict__ transition_matrix,  // [num_states][num_states]
    const marker_t* __restrict__ selected_states,  // [num_markers][num_states]
    const prob_t* __restrict__ forward_checkpoints,
    const prob_t* __restrict__ scaling_factors,
    const uint32_t num_markers,
    const uint32_t num_states,
    const uint32_t checkpoint_interval,
    // Outputs
    prob_t* __restrict__ posterior_probs           // [num_samples][num_markers][num_states]
);

// Compute emission probabilities from genotype likelihoods
// grid: (num_samples, (num_markers + 255) / 256, 1)
// block: (256, 1, 1)
__global__ void compute_emissions_kernel(
    // Inputs
    const GenotypeLikelihoods* __restrict__ genotype_liks,  // [num_samples][num_markers]
    const allele_t* __restrict__ reference_panel,           // [num_markers][num_haplotypes]
    const marker_t* __restrict__ selected_states,           // [num_markers][num_states]
    const uint32_t num_markers,
    const uint32_t num_states,
    const prob_t theta,                                      // Mutation rate
    // Outputs
    prob_t* __restrict__ emission_probs                      // [num_samples][num_markers][num_states]
);

// Compute transition probabilities from genetic distances
// Single block kernel - precompute transition matrix
__global__ void compute_transitions_kernel(
    // Inputs
    const double* __restrict__ genetic_distances,    // [num_markers] (cM)
    const uint32_t num_markers,
    const uint32_t num_states,
    const uint32_t ne,                                // Effective population size
    const prob_t rho_rate,                            // Recombination rate multiplier
    // Outputs
    prob_t* __restrict__ transition_matrices         // [num_markers-1][num_states][num_states]
);

// Device functions for use within kernels
#ifdef __CUDACC__

// Log-sum-exp with warp-level reduction
__device__ inline prob_t warp_logsumexp(prob_t log_val) {
    // Find max within warp for numerical stability
    prob_t max_val = log_val;
    for (int offset = 16; offset > 0; offset >>= 1) {
        max_val = fmaxf(max_val, __shfl_down_sync(0xffffffff, max_val, offset));
    }
    max_val = __shfl_sync(0xffffffff, max_val, 0);
    
    // Compute exp(log_val - max_val) and sum
    prob_t exp_val = expf(log_val - max_val);
    for (int offset = 16; offset > 0; offset >>= 1) {
        exp_val += __shfl_down_sync(0xffffffff, exp_val, offset);
    }
    exp_val = __shfl_sync(0xffffffff, exp_val, 0);
    
    // Return log(sum(exp)) + max
    return logf(exp_val) + max_val;
}

// Block-level log-sum-exp reduction
__device__ inline prob_t block_logsumexp(prob_t log_val, prob_t* shared_mem) {
    int tid = threadIdx.x;
    int lane = tid & 31;
    int warp_id = tid >> 5;
    
    // Warp-level reduction
    prob_t warp_max = log_val;
    for (int offset = 16; offset > 0; offset >>= 1) {
        warp_max = fmaxf(warp_max, __shfl_down_sync(0xffffffff, warp_max, offset));
    }
    
    // First thread in each warp writes to shared memory
    if (lane == 0) {
        shared_mem[warp_id] = warp_max;
    }
    __syncthreads();
    
    // First warp finds global max
    prob_t block_max = (tid < ((blockDim.x + 31) / 32)) ? shared_mem[tid] : -INFINITY;
    for (int offset = 16; offset > 0; offset >>= 1) {
        block_max = fmaxf(block_max, __shfl_down_sync(0xffffffff, block_max, offset));
    }
    block_max = __shfl_sync(0xffffffff, block_max, 0);
    
    // Compute exp(log_val - block_max) in each thread
    prob_t exp_val = expf(log_val - block_max);
    
    // Warp-level sum
    for (int offset = 16; offset > 0; offset >>= 1) {
        exp_val += __shfl_down_sync(0xffffffff, exp_val, offset);
    }
    
    if (lane == 0) {
        shared_mem[warp_id] = exp_val;
    }
    __syncthreads();
    
    // First warp sums all warp sums
    prob_t block_sum = (tid < ((blockDim.x + 31) / 32)) ? shared_mem[tid] : 0.0f;
    for (int offset = 16; offset > 0; offset >>= 1) {
        block_sum += __shfl_down_sync(0xffffffff, block_sum, offset);
    }
    block_sum = __shfl_sync(0xffffffff, block_sum, 0);
    
    return logf(block_sum) + block_max;
}

// Compute emission probability for diploid genotype
__device__ inline prob_t compute_emission_prob(
    const GenotypeLikelihoods& gl,
    allele_t ref_allele,
    prob_t theta
) {
    // Convert genotype likelihoods to probabilities
    prob_t p00 = powf(10.0f, gl.ll_00);
    prob_t p01 = powf(10.0f, gl.ll_01);
    prob_t p11 = powf(10.0f, gl.ll_11);
    
    // Normalize
    prob_t total = p00 + p01 + p11;
    p00 /= total;
    p01 /= total;
    p11 /= total;
    
    // Emission probability given reference allele
    if (ref_allele == 0) {
        return (1.0f - theta) * p00 + 0.5f * p01 + theta * p11;
    } else {
        return theta * p00 + 0.5f * p01 + (1.0f - theta) * p11;
    }
}

// Compute transition probability
__device__ inline prob_t compute_transition_prob(
    double genetic_distance_cM,
    uint32_t ne,
    prob_t rho_rate,
    bool is_switch
) {
    // rho = 4 * Ne * r where r is recombination rate
    double r = genetic_distance_cM / 100.0;  // Convert cM to Morgans
    double rho = rho_rate * ne * r;
    
    if (is_switch) {
        return static_cast<prob_t>(rho);
    } else {
        return static_cast<prob_t>(1.0 - rho);
    }
}

#endif // __CUDACC__

// Host wrapper functions

// Launch forward pass
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
    cudaStream_t stream = 0
);

// Launch backward pass
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
    cudaStream_t stream = 0
);

// Launch emission computation
void launch_compute_emissions(
    const GenotypeLikelihoods* d_genotype_liks,
    const allele_t* d_reference_panel,
    const marker_t* d_selected_states,
    uint32_t num_samples,
    uint32_t num_markers,
    uint32_t num_states,
    prob_t theta,
    prob_t* d_emission_probs,
    cudaStream_t stream = 0
);

// Launch transition computation
void launch_compute_transitions(
    const double* d_genetic_distances,
    uint32_t num_markers,
    uint32_t num_states,
    uint32_t ne,
    prob_t rho_rate,
    prob_t* d_transition_matrices,
    cudaStream_t stream = 0
);

// Complete forward-backward inference
class ForwardBackward {
public:
    ForwardBackward(
        uint32_t num_samples,
        uint32_t num_markers,
        uint32_t num_states,
        int device_id = 0
    );
    
    ~ForwardBackward();
    
    // Run forward-backward algorithm
    void run(
        const GenotypeLikelihoods* h_genotype_liks,
        const allele_t* h_reference_panel,
        const marker_t* h_selected_states,
        const double* h_genetic_distances,
        const HMMParameters& params,
        prob_t* h_posterior_probs
    );
    
    // Get device memory usage
    size_t get_device_memory_usage() const;
    
private:
    uint32_t num_samples_;
    uint32_t num_markers_;
    uint32_t num_states_;
    uint32_t checkpoint_interval_;
    int device_id_;
    
    // Device memory
    DevicePtr<GenotypeLikelihoods> d_genotype_liks_;
    DevicePtr<allele_t> d_reference_panel_;
    DevicePtr<marker_t> d_selected_states_;
    DevicePtr<double> d_genetic_distances_;
    DevicePtr<prob_t> d_emission_probs_;
    DevicePtr<prob_t> d_transition_matrices_;
    DevicePtr<prob_t> d_forward_checkpoints_;
    DevicePtr<prob_t> d_scaling_factors_;
    DevicePtr<prob_t> d_posterior_probs_;
    
    // CUDA stream for async operations
    cudaStream_t stream_;
    
    void allocate_device_memory();
    void free_device_memory();
    uint32_t calculate_checkpoint_interval() const;
};

} // namespace kernels
} // namespace swiftimpute
