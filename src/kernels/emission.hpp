#pragma once

#include "core/types.hpp"
#include <cuda_runtime.h>

namespace swiftimpute {
namespace kernels {

/**
 * EmissionComputer: Computes emission probabilities for HMM
 *
 * Purpose:
 *   For each (sample, marker, state), compute P(observed_genotype | hidden_state)
 *   using genotype likelihoods and reference haplotypes.
 *
 * Algorithm:
 *   1. Each state corresponds to a pair of reference haplotypes
 *   2. Look up the alleles for those haplotypes at the current marker
 *   3. Combine alleles into diploid genotype (0/0, 0/1, or 1/1)
 *   4. Return the genotype likelihood for that genotype
 *
 * Memory Layout:
 *   - Genotype likelihoods: [num_samples][num_markers]
 *   - Reference haplotypes: [num_markers][num_haplotypes]
 *   - Selected states: [num_samples][num_markers][num_states][2]
 *   - Emission probs (output): [num_samples][num_markers][num_states]
 *
 * Performance:
 *   - Grid: (num_samples, num_markers)
 *   - Block: (num_states)
 *   - Optional shared memory optimization for small state counts
 *
 * Example:
 *   EmissionComputer computer(10000, 1000, 8);
 *   computer.set_reference_panel(reference_haplotypes);
 *   computer.compute(genotype_liks, selected_states, emission_probs, num_samples);
 */
class EmissionComputer {
public:
    /**
     * Constructor
     *
     * @param num_markers Number of markers
     * @param num_haplotypes Number of reference haplotypes
     * @param num_states Number of states (L)
     * @param device_id CUDA device ID
     */
    EmissionComputer(
        marker_t num_markers,
        haplotype_t num_haplotypes,
        uint32_t num_states,
        int device_id = 0
    );

    ~EmissionComputer();

    // Non-copyable
    EmissionComputer(const EmissionComputer&) = delete;
    EmissionComputer& operator=(const EmissionComputer&) = delete;

    /**
     * Set reference panel (must be called before compute)
     *
     * Copies reference haplotypes to GPU memory.
     *
     * @param h_reference_haplotypes Host array [num_markers][num_haplotypes]
     */
    void set_reference_panel(const allele_t* h_reference_haplotypes);

    /**
     * Compute emission probabilities
     *
     * All pointers must be device pointers.
     *
     * @param d_genotype_liks Device array [num_samples][num_markers]
     * @param d_selected_states Device array [num_samples][num_markers][num_states][2]
     * @param d_emission_probs Device array (output) [num_samples][num_markers][num_states]
     * @param num_samples Number of target samples
     * @param use_shared_memory Use shared memory optimization (default: true)
     */
    void compute(
        const GenotypeLikelihoods* d_genotype_liks,
        const haplotype_t* d_selected_states,
        prob_t* d_emission_probs,
        uint32_t num_samples,
        bool use_shared_memory = true
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

    // CUDA stream
    cudaStream_t stream_;
};

} // namespace kernels
} // namespace swiftimpute
