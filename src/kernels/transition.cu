#include "transition.hpp"
#include "core/types.hpp"
#include <cuda_runtime.h>
#include <cmath>
#include <cfloat>

namespace swiftimpute {
namespace kernels {

// CUDA error checking macro
#define CHECK_CUDA(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            throw std::runtime_error(std::string("CUDA error: ") + cudaGetErrorString(err)); \
        } \
    } while(0)

// Log-space constants
constexpr prob_t LOG_ZERO = -999.0;
constexpr prob_t LOG_ONE = 0.0;

/**
 * Device function: Log-sum-exp for numerical stability
 */
__device__ prob_t log_add_exp(prob_t log_a, prob_t log_b) {
    if (log_a < LOG_ZERO + 1.0) return log_b;
    if (log_b < LOG_ZERO + 1.0) return log_a;

    prob_t max_val = (log_a > log_b) ? log_a : log_b;
    prob_t min_val = (log_a > log_b) ? log_b : log_a;

    // log(a + b) = log(a) + log(1 + b/a) = max + log(1 + exp(min - max))
    return max_val + log10(1.0 + pow(10.0, min_val - max_val));
}

/**
 * CUDA kernel: Compute Li-Stephens transition probabilities
 *
 * For each marker transition (m-1 → m), compute the L×L transition matrix:
 * P(state_j at marker m | state_i at marker m-1)
 *
 * Li-Stephens Model:
 *   rho = 1 - exp(-4 * ne * genetic_dist * rho_rate)
 *   P(j|i) = (1-rho) * delta(i,j) + rho / (2*ne)
 *
 * Where:
 *   - rho: recombination probability
 *   - ne: effective population size (default 10000)
 *   - genetic_dist: genetic distance in Morgans
 *   - theta: mutation rate (default 0.001)
 *   - delta(i,j): 1 if i==j, 0 otherwise
 *
 * @param genetic_distances Genetic distances in Morgans [num_markers]
 * @param params HMM parameters (ne, rho_rate, theta)
 * @param transition_matrices Output [num_markers-1][num_states][num_states]
 * @param num_markers Number of markers
 * @param num_states Number of states (L)
 */
__global__ void compute_transition_probs_kernel(
    const double* genetic_distances,
    HMMParameters params,
    prob_t* transition_matrices,
    marker_t num_markers,
    uint32_t num_states
) {
    // Each block handles one marker transition
    marker_t marker_idx = blockIdx.x;

    if (marker_idx >= num_markers - 1) {
        return;
    }

    // Thread handles one (i,j) pair in the transition matrix
    uint32_t i = threadIdx.y;  // From state
    uint32_t j = threadIdx.x;  // To state

    if (i >= num_states || j >= num_states) {
        return;
    }

    // Get genetic distance for this transition
    double genetic_dist = genetic_distances[marker_idx + 1] - genetic_distances[marker_idx];

    // Ensure non-negative distance
    if (genetic_dist < 0.0) {
        genetic_dist = 0.0;
    }

    // Compute recombination probability: rho = 1 - exp(-4 * ne * dist * rho_rate)
    double exponent = -4.0 * params.ne * genetic_dist * params.rho_rate;
    double rho = 1.0 - exp(exponent);

    // Clamp rho to valid range [0, 1]
    if (rho < 0.0) rho = 0.0;
    if (rho > 1.0) rho = 1.0;

    // Li-Stephens transition probability
    prob_t transition_prob;

    if (i == j) {
        // Stay in same state: (1 - rho) + rho / (2*ne)
        double stay_prob = (1.0 - rho) + rho / (2.0 * params.ne);
        transition_prob = log10(stay_prob);
    } else {
        // Switch to different state: rho / (2*ne)
        double switch_prob = rho / (2.0 * params.ne);

        if (switch_prob < 1e-300) {
            // Avoid log10(0)
            transition_prob = LOG_ZERO;
        } else {
            transition_prob = log10(switch_prob);
        }
    }

    // Store in output matrix
    // Layout: [marker][from_state][to_state]
    uint64_t offset = static_cast<uint64_t>(marker_idx) * num_states * num_states +
                      i * num_states + j;
    transition_matrices[offset] = transition_prob;
}

/**
 * Optimized kernel with shared memory for small state counts
 */
__global__ void compute_transition_probs_kernel_shared(
    const double* genetic_distances,
    HMMParameters params,
    prob_t* transition_matrices,
    marker_t num_markers,
    uint32_t num_states
) {
    // Shared memory for transition matrix
    extern __shared__ prob_t shared_transition[];

    marker_t marker_idx = blockIdx.x;

    if (marker_idx >= num_markers - 1) {
        return;
    }

    uint32_t i = threadIdx.y;
    uint32_t j = threadIdx.x;

    if (i >= num_states || j >= num_states) {
        return;
    }

    // Compute genetic distance
    double genetic_dist = genetic_distances[marker_idx + 1] - genetic_distances[marker_idx];
    if (genetic_dist < 0.0) genetic_dist = 0.0;

    // Compute recombination probability
    double exponent = -4.0 * params.ne * genetic_dist * params.rho_rate;
    double rho = 1.0 - exp(exponent);
    if (rho < 0.0) rho = 0.0;
    if (rho > 1.0) rho = 1.0;

    // Compute transition probability
    prob_t transition_prob;

    if (i == j) {
        double stay_prob = (1.0 - rho) + rho / (2.0 * params.ne);
        transition_prob = log10(stay_prob);
    } else {
        double switch_prob = rho / (2.0 * params.ne);
        transition_prob = (switch_prob < 1e-300) ? LOG_ZERO : log10(switch_prob);
    }

    // Store in shared memory
    shared_transition[i * num_states + j] = transition_prob;
    __syncthreads();

    // Copy to global memory
    uint64_t offset = static_cast<uint64_t>(marker_idx) * num_states * num_states +
                      i * num_states + j;
    transition_matrices[offset] = shared_transition[i * num_states + j];
}

// TransitionComputer implementation

TransitionComputer::TransitionComputer(
    marker_t num_markers,
    uint32_t num_states,
    int device_id
) : num_markers_(num_markers),
    num_states_(num_states),
    device_id_(device_id),
    d_genetic_distances_(nullptr),
    d_transition_matrices_(nullptr),
    stream_(nullptr)
{
    CHECK_CUDA(cudaSetDevice(device_id_));
    CHECK_CUDA(cudaStreamCreate(&stream_));

    // Set default HMM parameters
    params_.ne = 10000.0;           // Effective population size
    params_.rho_rate = 1e-8;        // Recombination rate per base pair
    params_.theta = 0.001;          // Mutation rate

    LOG_INFO("TransitionComputer initialized: markers=" + std::to_string(num_markers) +
             ", states=" + std::to_string(num_states));
}

TransitionComputer::~TransitionComputer() {
    if (d_genetic_distances_) {
        cudaFree(d_genetic_distances_);
    }
    if (d_transition_matrices_) {
        cudaFree(d_transition_matrices_);
    }
    if (stream_) {
        cudaStreamDestroy(stream_);
    }
}

void TransitionComputer::set_genetic_map(const double* h_genetic_distances) {
    CHECK_CUDA(cudaSetDevice(device_id_));

    size_t size = num_markers_ * sizeof(double);

    // Allocate if not already allocated
    if (d_genetic_distances_ == nullptr) {
        CHECK_CUDA(cudaMalloc(&d_genetic_distances_, size));
    }

    // Copy to device
    CHECK_CUDA(cudaMemcpyAsync(d_genetic_distances_, h_genetic_distances, size,
                               cudaMemcpyHostToDevice, stream_));
    CHECK_CUDA(cudaStreamSynchronize(stream_));

    LOG_INFO("Genetic map copied to GPU");
}

void TransitionComputer::set_parameters(const HMMParameters& params) {
    params_ = params;
    LOG_INFO("HMM parameters updated: ne=" + std::to_string(params.ne) +
             ", rho_rate=" + std::to_string(params.rho_rate) + ", theta=" + std::to_string(params.theta));
}

void TransitionComputer::compute(bool use_shared_memory) {
    CHECK_CUDA(cudaSetDevice(device_id_));

    if (d_genetic_distances_ == nullptr) {
        throw std::runtime_error("Genetic map not set. Call set_genetic_map() first.");
    }

    // Allocate output if not already allocated
    if (d_transition_matrices_ == nullptr) {
        // Size: (num_markers - 1) transitions × num_states² matrix entries
        size_t size = static_cast<size_t>(num_markers_ - 1) * num_states_ * num_states_ * sizeof(prob_t);
        CHECK_CUDA(cudaMalloc(&d_transition_matrices_, size));
        LOG_INFO("Allocated " + std::to_string(size / 1024.0 / 1024.0) + " MB for transition matrices");
    }

    // Grid: one block per marker transition
    dim3 grid(num_markers_ - 1);

    // Block: num_states × num_states threads
    dim3 block(num_states_, num_states_);

    if (use_shared_memory && num_states_ <= 16) {
        // Use shared memory version for small state counts (≤16 states = 256 threads)
        size_t shared_mem_size = num_states_ * num_states_ * sizeof(prob_t);

        compute_transition_probs_kernel_shared<<<grid, block, shared_mem_size, stream_>>>(
            d_genetic_distances_,
            params_,
            d_transition_matrices_,
            num_markers_,
            num_states_
        );
    } else {
        // Use standard version
        compute_transition_probs_kernel<<<grid, block, 0, stream_>>>(
            d_genetic_distances_,
            params_,
            d_transition_matrices_,
            num_markers_,
            num_states_
        );
    }

    CHECK_CUDA(cudaGetLastError());
    CHECK_CUDA(cudaStreamSynchronize(stream_));

    LOG_INFO("Computed " + std::to_string(num_markers_ - 1) + " transition matrices");
}

const prob_t* TransitionComputer::get_transition_matrices() const {
    return d_transition_matrices_;
}

void TransitionComputer::copy_to_host(prob_t* h_transition_matrices) const {
    CHECK_CUDA(cudaSetDevice(device_id_));

    if (d_transition_matrices_ == nullptr) {
        throw std::runtime_error("Transition matrices not computed. Call compute() first.");
    }

    size_t size = static_cast<size_t>(num_markers_ - 1) * num_states_ * num_states_ * sizeof(prob_t);
    CHECK_CUDA(cudaMemcpy(h_transition_matrices, d_transition_matrices_, size,
                         cudaMemcpyDeviceToHost));
}

size_t TransitionComputer::memory_usage() const {
    size_t genetic_map_size = num_markers_ * sizeof(double);
    size_t transition_size = static_cast<size_t>(num_markers_ - 1) * num_states_ * num_states_ * sizeof(prob_t);
    return genetic_map_size + transition_size;
}

} // namespace kernels
} // namespace swiftimpute
