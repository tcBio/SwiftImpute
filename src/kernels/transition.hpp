#pragma once

#include "core/types.hpp"
#include <cuda_runtime.h>

namespace swiftimpute {
namespace kernels {

/**
 * HMM parameters for Li-Stephens model
 */
struct HMMParameters {
    double ne;         // Effective population size (default: 10000)
    double rho_rate;   // Recombination rate per base pair (default: 1e-8)
    double theta;      // Mutation rate (default: 0.001)

    HMMParameters()
        : ne(10000.0), rho_rate(1e-8), theta(0.001) {}
};

/**
 * TransitionComputer: Computes transition probabilities for Li-Stephens HMM
 *
 * Purpose:
 *   For each marker transition (m-1 → m), compute the L×L transition matrix
 *   P(state_j at marker m | state_i at marker m-1) using the Li-Stephens model.
 *
 * Li-Stephens Model:
 *   rho = 1 - exp(-4 * ne * genetic_dist * rho_rate)
 *   P(j|i) = (1-rho) * delta(i,j) + rho / (2*ne)
 *
 *   Where:
 *   - rho: recombination probability between markers
 *   - ne: effective population size
 *   - genetic_dist: genetic distance in Morgans
 *   - delta(i,j): 1 if i==j (stay in same state), 0 otherwise
 *
 * Memory Layout:
 *   - Input: genetic_distances [num_markers] (cumulative Morgans)
 *   - Output: transition_matrices [num_markers-1][num_states][num_states]
 *
 * Performance:
 *   - Grid: (num_markers - 1) blocks (one per transition)
 *   - Block: (num_states, num_states) threads
 *   - Shared memory optimization available for small state counts
 *
 * Example:
 *   TransitionComputer computer(10000, 8);
 *   computer.set_genetic_map(genetic_distances);
 *   computer.compute();
 *   const prob_t* transitions = computer.get_transition_matrices();
 */
class TransitionComputer {
public:
    /**
     * Constructor
     *
     * @param num_markers Number of markers
     * @param num_states Number of states (L)
     * @param device_id CUDA device ID
     */
    TransitionComputer(
        marker_t num_markers,
        uint32_t num_states,
        int device_id = 0
    );

    ~TransitionComputer();

    // Non-copyable
    TransitionComputer(const TransitionComputer&) = delete;
    TransitionComputer& operator=(const TransitionComputer&) = delete;

    /**
     * Set genetic map (must be called before compute)
     *
     * @param h_genetic_distances Host array [num_markers] with cumulative genetic
     *                           distances in Morgans (or centiMorgans)
     */
    void set_genetic_map(const double* h_genetic_distances);

    /**
     * Set HMM parameters
     *
     * @param params HMM parameters (ne, rho_rate, theta)
     */
    void set_parameters(const HMMParameters& params);

    /**
     * Compute all transition matrices
     *
     * Computes (num_markers - 1) transition matrices, one for each marker-to-marker
     * transition. Results are stored in GPU memory.
     *
     * @param use_shared_memory Use shared memory optimization (default: true)
     */
    void compute(bool use_shared_memory = true);

    /**
     * Get device pointer to transition matrices
     *
     * @return Device pointer to [num_markers-1][num_states][num_states] array
     */
    const prob_t* get_transition_matrices() const;

    /**
     * Copy transition matrices to host
     *
     * @param h_transition_matrices Host array [num_markers-1][num_states][num_states]
     */
    void copy_to_host(prob_t* h_transition_matrices) const;

    /**
     * Get GPU memory usage in bytes
     */
    size_t memory_usage() const;

private:
    marker_t num_markers_;
    uint32_t num_states_;
    int device_id_;

    // HMM parameters
    HMMParameters params_;

    // GPU memory
    double* d_genetic_distances_;
    prob_t* d_transition_matrices_;

    // CUDA stream
    cudaStream_t stream_;
};

} // namespace kernels
} // namespace swiftimpute
