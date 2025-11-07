#pragma once

#include "core/types.hpp"
#include <cuda_runtime.h>
#include <curand_kernel.h>

namespace swiftimpute {
namespace kernels {

/**
 * HaplotypeSampler: Samples phased haplotypes from posterior distribution
 *
 * Purpose:
 *   After forward-backward algorithm computes posterior probabilities,
 *   sample phased haplotypes by:
 *   1. For each marker, sample a state from posterior distribution
 *   2. Look up the reference haplotypes for that state
 *   3. Assign those alleles to the output haplotypes
 *
 * Two Modes:
 *   - Deterministic: Take argmax of posterior (for testing)
 *   - Stochastic: Sample from posterior distribution (proper sampling)
 *
 * Memory Layout:
 *   - Posteriors: [num_samples][num_markers][num_states] (log probabilities)
 *   - Selected states: [num_samples][num_markers][num_states][2]
 *   - Reference haplotypes: [num_markers][num_haplotypes]
 *   - Output: [num_samples][2][num_markers]
 *
 * Example:
 *   HaplotypeSampler sampler(10000, 1000, 8);
 *   sampler.set_reference_panel(reference_haplotypes);
 *   sampler.initialize_rng(num_samples, seed);
 *   sampler.sample(posteriors, selected_states, output, num_samples, false);
 */
class HaplotypeSampler {
public:
    /**
     * Constructor
     *
     * @param num_markers Number of markers
     * @param num_haplotypes Number of reference haplotypes
     * @param num_states Number of HMM states
     * @param device_id CUDA device ID
     */
    HaplotypeSampler(
        marker_t num_markers,
        haplotype_t num_haplotypes,
        uint32_t num_states,
        int device_id = 0
    );

    ~HaplotypeSampler();

    // Non-copyable
    HaplotypeSampler(const HaplotypeSampler&) = delete;
    HaplotypeSampler& operator=(const HaplotypeSampler&) = delete;

    /**
     * Set reference panel
     *
     * Must be called before sampling.
     *
     * @param h_reference_haplotypes Host array [num_markers][num_haplotypes]
     */
    void set_reference_panel(const allele_t* h_reference_haplotypes);

    /**
     * Initialize random number generator for stochastic sampling
     *
     * @param num_samples Number of samples
     * @param seed Random seed
     */
    void initialize_rng(uint32_t num_samples, unsigned long long seed = 12345ULL);

    /**
     * Sample haplotypes from posterior distribution
     *
     * All pointers must be device pointers.
     *
     * @param d_posteriors Device array [num_samples][num_markers][num_states] (log probs)
     * @param d_selected_states Device array [num_samples][num_markers][num_states][2]
     * @param d_output_haplotypes Device array (output) [num_samples][2][num_markers]
     * @param num_samples Number of samples
     * @param deterministic If true, use argmax; if false, sample stochastically
     */
    void sample(
        const prob_t* d_posteriors,
        const haplotype_t* d_selected_states,
        allele_t* d_output_haplotypes,
        uint32_t num_samples,
        bool deterministic = false
    );

    /**
     * Get GPU memory usage in bytes
     */
    size_t memory_usage() const;

private:
    marker_t num_markers_;
    haplotype_t num_haplotypes_;
    uint32_t num_states_;
    int device_id_;

    // GPU memory
    allele_t* d_reference_haplotypes_;
    curandState* d_rng_states_;

    // CUDA stream
    cudaStream_t stream_;

    // RNG state
    bool rng_initialized_;
    uint32_t rng_num_samples_;
};

} // namespace kernels
} // namespace swiftimpute
