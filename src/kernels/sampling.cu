#include "sampling.hpp"
#include "core/types.hpp"
#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <cmath>

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

constexpr prob_t LOG_ZERO = -999.0f;

/**
 * Initialize cuRAND states for each sample
 */
__global__ void init_curand_kernel(
    curandState* rng_states,
    uint32_t num_samples,
    unsigned long long seed
) {
    uint32_t sample_idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (sample_idx < num_samples) {
        curand_init(seed, sample_idx, 0, &rng_states[sample_idx]);
    }
}

/**
 * Sample haplotypes from posterior distribution (stochastic mode)
 *
 * For each marker, samples a state from the posterior distribution,
 * then looks up the corresponding reference haplotypes.
 *
 * @param posteriors Posterior probabilities [num_samples][num_markers][num_states] (log scale)
 * @param selected_states Selected state indices [num_samples][num_markers][num_states]
 * @param reference_haplotypes Reference panel [num_markers][num_haplotypes]
 * @param output_haplotypes Output [num_samples][2][num_markers]
 * @param rng_states cuRAND states [num_samples]
 * @param num_samples Number of samples
 * @param num_markers Number of markers
 * @param num_haplotypes Number of reference haplotypes
 * @param num_states Number of HMM states
 */
__global__ void sample_haplotypes_stochastic_kernel(
    const prob_t* __restrict__ posteriors,
    const haplotype_t* __restrict__ selected_states,
    const allele_t* __restrict__ reference_haplotypes,
    allele_t* __restrict__ output_haplotypes,
    curandState* rng_states,
    uint32_t num_samples,
    marker_t num_markers,
    haplotype_t num_haplotypes,
    uint32_t num_states
) {
    // One thread per sample
    uint32_t sample_idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (sample_idx >= num_samples) {
        return;
    }

    curandState local_rng = rng_states[sample_idx];

    uint64_t posterior_base = static_cast<uint64_t>(sample_idx) * num_markers * num_states;
    uint64_t selected_base = posterior_base;
    uint64_t output_base = static_cast<uint64_t>(sample_idx) * 2 * num_markers;

    // Sample each marker
    for (marker_t m = 0; m < num_markers; ++m) {
        // Get posteriors for this marker (in log space)
        prob_t log_probs[32];  // Assuming num_states <= 32
        prob_t max_log = LOG_ZERO;

        // Find max for numerical stability
        for (uint32_t s = 0; s < num_states; ++s) {
            log_probs[s] = posteriors[posterior_base + m * num_states + s];
            max_log = fmaxf(max_log, log_probs[s]);
        }

        // Convert to probability space and compute cumulative sum
        float probs[32];
        float cumsum = 0.0f;

        for (uint32_t s = 0; s < num_states; ++s) {
            probs[s] = expf(log_probs[s] - max_log);
            cumsum += probs[s];
        }

        // Normalize
        for (uint32_t s = 0; s < num_states; ++s) {
            probs[s] /= cumsum;
        }

        // Sample state using cumulative distribution
        float r = curand_uniform(&local_rng);
        cumsum = 0.0f;
        uint32_t sampled_state = 0;

        for (uint32_t s = 0; s < num_states; ++s) {
            cumsum += probs[s];
            if (r < cumsum) {
                sampled_state = s;
                break;
            }
        }

        // Get the two haplotype indices for this state
        haplotype_t hap0_idx = selected_states[(selected_base + m * num_states + sampled_state) * 2];
        haplotype_t hap1_idx = selected_states[(selected_base + m * num_states + sampled_state) * 2 + 1];

        // Look up alleles from reference panel
        uint64_t ref_offset = static_cast<uint64_t>(m) * num_haplotypes;
        allele_t allele0 = reference_haplotypes[ref_offset + hap0_idx];
        allele_t allele1 = reference_haplotypes[ref_offset + hap1_idx];

        // Store in output
        output_haplotypes[output_base + m] = allele0;                      // First haplotype
        output_haplotypes[output_base + num_markers + m] = allele1;        // Second haplotype
    }

    // Save RNG state
    rng_states[sample_idx] = local_rng;
}

/**
 * Sample haplotypes deterministically (take most likely state)
 *
 * For testing and reproducibility, takes the argmax of posterior distribution.
 */
__global__ void sample_haplotypes_deterministic_kernel(
    const prob_t* __restrict__ posteriors,
    const haplotype_t* __restrict__ selected_states,
    const allele_t* __restrict__ reference_haplotypes,
    allele_t* __restrict__ output_haplotypes,
    uint32_t num_samples,
    marker_t num_markers,
    haplotype_t num_haplotypes,
    uint32_t num_states
) {
    uint32_t sample_idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (sample_idx >= num_samples) {
        return;
    }

    uint64_t posterior_base = static_cast<uint64_t>(sample_idx) * num_markers * num_states;
    uint64_t selected_base = posterior_base;
    uint64_t output_base = static_cast<uint64_t>(sample_idx) * 2 * num_markers;

    for (marker_t m = 0; m < num_markers; ++m) {
        // Find argmax of posteriors
        uint32_t best_state = 0;
        prob_t best_prob = posteriors[posterior_base + m * num_states];

        for (uint32_t s = 1; s < num_states; ++s) {
            prob_t prob = posteriors[posterior_base + m * num_states + s];
            if (prob > best_prob) {
                best_prob = prob;
                best_state = s;
            }
        }

        // Get haplotypes for best state
        haplotype_t hap0_idx = selected_states[(selected_base + m * num_states + best_state) * 2];
        haplotype_t hap1_idx = selected_states[(selected_base + m * num_states + best_state) * 2 + 1];

        uint64_t ref_offset = static_cast<uint64_t>(m) * num_haplotypes;
        allele_t allele0 = reference_haplotypes[ref_offset + hap0_idx];
        allele_t allele1 = reference_haplotypes[ref_offset + hap1_idx];

        output_haplotypes[output_base + m] = allele0;
        output_haplotypes[output_base + num_markers + m] = allele1;
    }
}

// HaplotypeSampler implementation

HaplotypeSampler::HaplotypeSampler(
    marker_t num_markers,
    haplotype_t num_haplotypes,
    uint32_t num_states,
    int device_id
) : num_markers_(num_markers),
    num_haplotypes_(num_haplotypes),
    num_states_(num_states),
    device_id_(device_id),
    d_reference_haplotypes_(nullptr),
    d_rng_states_(nullptr),
    stream_(nullptr),
    rng_initialized_(false)
{
    CHECK_CUDA(cudaSetDevice(device_id_));
    CHECK_CUDA(cudaStreamCreate(&stream_));

    LOG_INFO("HaplotypeSampler initialized: markers=" + std::to_string(num_markers) +
             ", haplotypes=" + std::to_string(num_haplotypes) + ", states=" + std::to_string(num_states));
}

HaplotypeSampler::~HaplotypeSampler() {
    if (d_reference_haplotypes_) {
        cudaFree(d_reference_haplotypes_);
    }
    if (d_rng_states_) {
        cudaFree(d_rng_states_);
    }
    if (stream_) {
        cudaStreamDestroy(stream_);
    }
}

void HaplotypeSampler::set_reference_panel(const allele_t* h_reference_haplotypes) {
    CHECK_CUDA(cudaSetDevice(device_id_));

    size_t size = static_cast<size_t>(num_markers_) * num_haplotypes_ * sizeof(allele_t);

    if (d_reference_haplotypes_ == nullptr) {
        CHECK_CUDA(cudaMalloc(&d_reference_haplotypes_, size));
        LOG_INFO("Allocated " + std::to_string(size / 1024.0 / 1024.0) + " MB for reference panel");
    }

    CHECK_CUDA(cudaMemcpyAsync(d_reference_haplotypes_, h_reference_haplotypes, size,
                               cudaMemcpyHostToDevice, stream_));
    CHECK_CUDA(cudaStreamSynchronize(stream_));

    LOG_INFO("Reference panel copied to GPU for sampling");
}

void HaplotypeSampler::initialize_rng(uint32_t num_samples, unsigned long long seed) {
    CHECK_CUDA(cudaSetDevice(device_id_));

    // Allocate RNG states
    if (d_rng_states_ == nullptr || rng_num_samples_ != num_samples) {
        if (d_rng_states_) {
            cudaFree(d_rng_states_);
        }

        size_t size = num_samples * sizeof(curandState);
        CHECK_CUDA(cudaMalloc(&d_rng_states_, size));
        rng_num_samples_ = num_samples;

        LOG_INFO("Allocated " + std::to_string(size / 1024.0 / 1024.0) + " MB for RNG states");
    }

    // Initialize RNG states
    int block_size = 256;
    int grid_size = (num_samples + block_size - 1) / block_size;

    init_curand_kernel<<<grid_size, block_size, 0, stream_>>>(
        d_rng_states_,
        num_samples,
        seed
    );

    CHECK_CUDA(cudaGetLastError());
    CHECK_CUDA(cudaStreamSynchronize(stream_));

    rng_initialized_ = true;
    LOG_INFO("Initialized RNG for " + std::to_string(num_samples) + " samples with seed " + std::to_string(seed));
}

void HaplotypeSampler::sample(
    const prob_t* d_posteriors,
    const haplotype_t* d_selected_states,
    allele_t* d_output_haplotypes,
    uint32_t num_samples,
    bool deterministic
) {
    CHECK_CUDA(cudaSetDevice(device_id_));

    if (d_reference_haplotypes_ == nullptr) {
        throw std::runtime_error("Reference panel not set. Call set_reference_panel() first.");
    }

    // Launch configuration
    int block_size = 256;
    int grid_size = (num_samples + block_size - 1) / block_size;

    if (deterministic) {
        // Deterministic sampling (argmax)
        sample_haplotypes_deterministic_kernel<<<grid_size, block_size, 0, stream_>>>(
            d_posteriors,
            d_selected_states,
            d_reference_haplotypes_,
            d_output_haplotypes,
            num_samples,
            num_markers_,
            num_haplotypes_,
            num_states_
        );
    } else {
        // Stochastic sampling
        if (!rng_initialized_ || rng_num_samples_ != num_samples) {
            // Initialize RNG with default seed if not already done
            initialize_rng(num_samples, 12345ULL);
        }

        sample_haplotypes_stochastic_kernel<<<grid_size, block_size, 0, stream_>>>(
            d_posteriors,
            d_selected_states,
            d_reference_haplotypes_,
            d_output_haplotypes,
            d_rng_states_,
            num_samples,
            num_markers_,
            num_haplotypes_,
            num_states_
        );
    }

    CHECK_CUDA(cudaGetLastError());
    CHECK_CUDA(cudaStreamSynchronize(stream_));
}

size_t HaplotypeSampler::memory_usage() const {
    size_t total = 0;

    if (d_reference_haplotypes_) {
        total += static_cast<size_t>(num_markers_) * num_haplotypes_ * sizeof(allele_t);
    }

    if (d_rng_states_) {
        total += rng_num_samples_ * sizeof(curandState);
    }

    return total;
}

} // namespace kernels
} // namespace swiftimpute
