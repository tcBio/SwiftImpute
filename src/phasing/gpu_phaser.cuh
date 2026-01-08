#pragma once

#include "../core/types.hpp"
#include "../kernels/forward_backward.cuh"
#include <cuda_runtime.h>
#include <memory>
#include <vector>

namespace swiftimpute {
namespace phasing {

/**
 * @brief GPU-accelerated phasing kernel
 *
 * Determines optimal phasing for heterozygous sites using posterior probabilities
 * Processes all samples in parallel on GPU
 *
 * For each heterozygous site, we need to determine whether the phase is 0|1 or 1|0
 * This is done by computing P(hap0=ref) vs P(hap0=alt) from the diploid HMM posteriors
 */
__global__ void phase_from_posteriors_kernel(
    // Inputs
    const prob_t* __restrict__ posterior_probs,    // [num_samples][num_markers][num_states*2]
    const GenotypeLikelihoods* __restrict__ geno_liks,  // [num_samples][num_markers]
    const allele_t* __restrict__ ref_haplotypes,   // [num_markers][num_haplotypes]
    const haplotype_t* __restrict__ selected_states,  // [num_markers][num_states]
    uint32_t num_markers,
    uint32_t num_states,
    uint32_t num_haplotypes,
    // Outputs
    allele_t* __restrict__ haplotype0,             // [num_samples][num_markers]
    allele_t* __restrict__ haplotype1              // [num_samples][num_markers]
);

/**
 * @brief Viterbi decoding kernel for maximum likelihood phasing
 *
 * Uses Viterbi algorithm to find the most likely haplotype pair assignment
 * More accurate than posterior sampling for phasing
 */
__global__ void viterbi_phase_kernel(
    // Inputs
    const prob_t* __restrict__ emission_probs,     // [num_samples][num_markers][num_states*num_states]
    const prob_t* __restrict__ transition_probs,   // [num_markers-1][num_states][num_states]
    uint32_t num_markers,
    uint32_t num_states,
    // Working memory
    prob_t* __restrict__ viterbi_probs,            // [num_samples][num_markers][num_states*num_states]
    uint32_t* __restrict__ backpointers,           // [num_samples][num_markers][num_states*num_states]
    // Outputs
    uint32_t* __restrict__ best_state_pairs        // [num_samples][num_markers] - encodes (state1, state2)
);

/**
 * @brief Convert state pair indices to phased haplotypes
 */
__global__ void decode_haplotypes_kernel(
    const uint32_t* __restrict__ best_state_pairs, // [num_samples][num_markers]
    const haplotype_t* __restrict__ selected_states,  // [num_markers][num_states]
    const allele_t* __restrict__ ref_haplotypes,   // [num_markers][num_haplotypes]
    uint32_t num_markers,
    uint32_t num_states,
    // Outputs
    allele_t* __restrict__ haplotype0,             // [num_samples][num_markers]
    allele_t* __restrict__ haplotype1              // [num_samples][num_markers]
);

/**
 * @brief Host wrapper for phasing kernels
 */
void launch_phase_from_posteriors(
    const prob_t* d_posterior_probs,
    const GenotypeLikelihoods* d_geno_liks,
    const allele_t* d_ref_haplotypes,
    const haplotype_t* d_selected_states,
    uint32_t num_samples,
    uint32_t num_markers,
    uint32_t num_states,
    uint32_t num_haplotypes,
    allele_t* d_haplotype0,
    allele_t* d_haplotype1,
    cudaStream_t stream = 0
);

void launch_viterbi_phase(
    const prob_t* d_emission_probs,
    const prob_t* d_transition_probs,
    uint32_t num_samples,
    uint32_t num_markers,
    uint32_t num_states,
    prob_t* d_viterbi_probs,
    uint32_t* d_backpointers,
    uint32_t* d_best_state_pairs,
    cudaStream_t stream = 0
);

void launch_decode_haplotypes(
    const uint32_t* d_best_state_pairs,
    const haplotype_t* d_selected_states,
    const allele_t* d_ref_haplotypes,
    uint32_t num_samples,
    uint32_t num_markers,
    uint32_t num_states,
    allele_t* d_haplotype0,
    allele_t* d_haplotype1,
    cudaStream_t stream = 0
);

/**
 * @brief GPU-accelerated phaser class
 *
 * Performs HMM-based statistical phasing using GPU acceleration.
 * All samples are processed in parallel on the GPU.
 */
class GPUPhaser {
public:
    struct Config {
        uint32_t num_states;            // HMM states per haplotype (default: 8)
        uint32_t ne;                    // Effective population size
        uint32_t num_iterations;        // MCMC iterations for iterative phasing
        bool use_viterbi;               // Use Viterbi (true) or posterior sampling (false)
        int device_id;                  // GPU device (-1 for auto)

        Config() :
            num_states(8),
            ne(10000),
            num_iterations(5),
            use_viterbi(true),
            device_id(-1) {}
    };

    GPUPhaser(
        const ReferencePanel& reference,
        const Config& config = Config()
    );

    ~GPUPhaser();

    /**
     * @brief Phase all samples in parallel on GPU
     *
     * @param targets Target data with unphased genotypes
     * @return Phased haplotypes for all samples
     */
    struct PhasedResult {
        std::vector<allele_t> haplotype0;  // [num_samples * num_markers]
        std::vector<allele_t> haplotype1;  // [num_samples * num_markers]
        std::vector<prob_t> phase_confidence;  // [num_samples * num_markers] confidence scores
    };

    PhasedResult phase(const TargetData& targets);

    /**
     * @brief Phase with progress callback
     */
    PhasedResult phase_with_progress(
        const TargetData& targets,
        std::function<void(uint32_t, uint32_t)> progress_callback
    );

    /**
     * @brief Get estimated GPU memory requirement
     */
    static size_t estimate_memory(
        uint32_t num_samples,
        uint32_t num_markers,
        uint32_t num_states
    );

    /**
     * @brief Get current device memory usage
     */
    size_t get_device_memory_usage() const;

private:
    const ReferencePanel& reference_;
    Config config_;
    int device_id_;

    // GPU memory
    cudaStream_t stream_;

    // Device buffers
    GenotypeLikelihoods* d_geno_liks_;
    allele_t* d_ref_haplotypes_;
    haplotype_t* d_selected_states_;
    double* d_genetic_distances_;
    prob_t* d_emission_probs_;
    prob_t* d_transition_probs_;
    prob_t* d_forward_checkpoints_;
    prob_t* d_scaling_factors_;
    prob_t* d_posterior_probs_;
    allele_t* d_haplotype0_;
    allele_t* d_haplotype1_;

    // For Viterbi
    prob_t* d_viterbi_probs_;
    uint32_t* d_backpointers_;
    uint32_t* d_best_state_pairs_;

    // Buffer sizes
    size_t buffer_size_samples_;
    size_t buffer_size_markers_;

    void allocate_buffers(uint32_t num_samples, uint32_t num_markers);
    void free_buffers();
    void build_pbwt_and_select_states();
};

} // namespace phasing
} // namespace swiftimpute
