#include "emission.hpp"
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
constexpr prob_t LOG_ZERO = -999.0;  // log10(0) approximation
constexpr prob_t LOG_ONE = 0.0;      // log10(1) = 0
constexpr prob_t LOG_HALF = -0.30103; // log10(0.5)

/**
 * Device function: Compute emission probability for a single state
 *
 * Given genotype likelihoods and two haplotype alleles, compute
 * P(observed genotype | haplotype pair)
 *
 * @param gl Genotype likelihoods (log10 scale)
 * @param allele0 First haplotype allele
 * @param allele1 Second haplotype allele
 * @return Emission probability (log10 scale)
 */
__device__ prob_t compute_single_emission(
    const GenotypeLikelihoods& gl,
    allele_t allele0,
    allele_t allele1
) {
    // Determine diploid genotype from haplotype pair
    // Order doesn't matter: 0|1 and 1|0 both give genotype 0/1
    int genotype_sum = allele0 + allele1;

    if (genotype_sum == 0) {
        // Both haplotypes are REF (0|0) → genotype 0/0
        return gl.ll_00;
    } else if (genotype_sum == 1) {
        // One REF, one ALT (0|1 or 1|0) → genotype 0/1
        return gl.ll_01;
    } else {
        // Both haplotypes are ALT (1|1) → genotype 1/1
        return gl.ll_11;
    }
}

/**
 * CUDA kernel: Compute emission probabilities for all samples, markers, and states
 *
 * For each (sample, marker, state) triple, compute:
 * P(observed_genotype[sample][marker] | reference_state[marker][state])
 *
 * @param genotype_liks Input genotype likelihoods [num_samples][num_markers]
 * @param reference_haplotypes Reference panel [num_markers][num_haplotypes]
 * @param selected_states Selected state indices [num_samples][num_markers][num_states]
 * @param emission_probs Output emission probabilities [num_samples][num_markers][num_states]
 * @param num_samples Number of target samples
 * @param num_markers Number of markers
 * @param num_haplotypes Number of reference haplotypes
 * @param num_states Number of states per marker (L)
 */
__global__ void compute_emission_probs_kernel(
    const GenotypeLikelihoods* genotype_liks,
    const allele_t* reference_haplotypes,
    const haplotype_t* selected_states,
    prob_t* emission_probs,
    uint32_t num_samples,
    marker_t num_markers,
    haplotype_t num_haplotypes,
    uint32_t num_states
) {
    // Thread indices
    uint32_t sample_idx = blockIdx.x;
    marker_t marker_idx = blockIdx.y;
    uint32_t state_idx = threadIdx.x;

    // Bounds check
    if (sample_idx >= num_samples || marker_idx >= num_markers || state_idx >= num_states) {
        return;
    }

    // Get genotype likelihood for this sample and marker
    const GenotypeLikelihoods& gl = genotype_liks[sample_idx * num_markers + marker_idx];

    // Get the two haplotype indices for this state
    // Each state corresponds to a pair of reference haplotypes
    uint32_t state_offset = (sample_idx * num_markers + marker_idx) * num_states + state_idx;
    haplotype_t hap0_idx = selected_states[state_offset * 2];
    haplotype_t hap1_idx = selected_states[state_offset * 2 + 1];

    // Look up alleles from reference panel
    uint64_t ref_offset = marker_idx * num_haplotypes;
    allele_t allele0 = reference_haplotypes[ref_offset + hap0_idx];
    allele_t allele1 = reference_haplotypes[ref_offset + hap1_idx];

    // Compute emission probability
    prob_t emission = compute_single_emission(gl, allele0, allele1);

    // Store result
    emission_probs[state_offset] = emission;
}

/**
 * Optimized kernel using shared memory for selected states
 * This version loads selected states into shared memory to reduce global memory accesses
 */
__global__ void compute_emission_probs_kernel_shared(
    const GenotypeLikelihoods* genotype_liks,
    const allele_t* reference_haplotypes,
    const haplotype_t* selected_states,
    prob_t* emission_probs,
    uint32_t num_samples,
    marker_t num_markers,
    haplotype_t num_haplotypes,
    uint32_t num_states
) {
    // Shared memory for selected states (2 haplotypes per state)
    extern __shared__ haplotype_t shared_states[];

    uint32_t sample_idx = blockIdx.x;
    marker_t marker_idx = blockIdx.y;
    uint32_t state_idx = threadIdx.x;

    if (sample_idx >= num_samples || marker_idx >= num_markers) {
        return;
    }

    // Cooperatively load selected states into shared memory
    uint32_t state_base = (sample_idx * num_markers + marker_idx) * num_states;
    if (state_idx < num_states) {
        shared_states[state_idx * 2] = selected_states[(state_base + state_idx) * 2];
        shared_states[state_idx * 2 + 1] = selected_states[(state_base + state_idx) * 2 + 1];
    }
    __syncthreads();

    // Compute emission for this state
    if (state_idx < num_states) {
        const GenotypeLikelihoods& gl = genotype_liks[sample_idx * num_markers + marker_idx];

        haplotype_t hap0_idx = shared_states[state_idx * 2];
        haplotype_t hap1_idx = shared_states[state_idx * 2 + 1];

        uint64_t ref_offset = marker_idx * num_haplotypes;
        allele_t allele0 = reference_haplotypes[ref_offset + hap0_idx];
        allele_t allele1 = reference_haplotypes[ref_offset + hap1_idx];

        prob_t emission = compute_single_emission(gl, allele0, allele1);

        emission_probs[state_base + state_idx] = emission;
    }
}

// EmissionComputer implementation

EmissionComputer::EmissionComputer(
    marker_t num_markers,
    haplotype_t num_haplotypes,
    uint32_t num_states,
    int device_id
) : num_markers_(num_markers),
    num_haplotypes_(num_haplotypes),
    num_states_(num_states),
    device_id_(device_id),
    d_reference_haplotypes_(nullptr),
    stream_(nullptr)
{
    CHECK_CUDA(cudaSetDevice(device_id_));
    CHECK_CUDA(cudaStreamCreate(&stream_));

    LOG_INFO("EmissionComputer initialized: markers=" + std::to_string(num_markers) +
             ", haplotypes=" + std::to_string(num_haplotypes) + ", states=" + std::to_string(num_states));
}

EmissionComputer::~EmissionComputer() {
    if (stream_) {
        cudaStreamDestroy(stream_);
    }
}

void EmissionComputer::set_reference_panel(const allele_t* h_reference_haplotypes) {
    CHECK_CUDA(cudaSetDevice(device_id_));

    size_t size = static_cast<size_t>(num_markers_) * num_haplotypes_ * sizeof(allele_t);

    // Allocate if not already allocated
    if (d_reference_haplotypes_ == nullptr) {
        CHECK_CUDA(cudaMalloc(&d_reference_haplotypes_, size));
        LOG_INFO("Allocated " + std::to_string(size / 1024.0 / 1024.0) + " MB for reference panel on GPU");
    }

    // Copy to device
    CHECK_CUDA(cudaMemcpyAsync(d_reference_haplotypes_, h_reference_haplotypes, size,
                               cudaMemcpyHostToDevice, stream_));
    CHECK_CUDA(cudaStreamSynchronize(stream_));

    LOG_INFO("Reference panel copied to GPU");
}

void EmissionComputer::compute(
    const GenotypeLikelihoods* d_genotype_liks,
    const haplotype_t* d_selected_states,
    prob_t* d_emission_probs,
    uint32_t num_samples,
    bool use_shared_memory
) {
    CHECK_CUDA(cudaSetDevice(device_id_));

    if (d_reference_haplotypes_ == nullptr) {
        throw std::runtime_error("Reference panel not set. Call set_reference_panel() first.");
    }

    // Grid dimensions: one block per (sample, marker) pair
    dim3 grid(num_samples, num_markers_);

    // Block dimensions: one thread per state
    dim3 block(num_states_);

    if (use_shared_memory && num_states_ <= 32) {
        // Use shared memory version for small state counts
        size_t shared_mem_size = num_states_ * 2 * sizeof(haplotype_t);

        compute_emission_probs_kernel_shared<<<grid, block, shared_mem_size, stream_>>>(
            d_genotype_liks,
            d_reference_haplotypes_,
            d_selected_states,
            d_emission_probs,
            num_samples,
            num_markers_,
            num_haplotypes_,
            num_states_
        );
    } else {
        // Use standard version
        compute_emission_probs_kernel<<<grid, block, 0, stream_>>>(
            d_genotype_liks,
            d_reference_haplotypes_,
            d_selected_states,
            d_emission_probs,
            num_samples,
            num_markers_,
            num_haplotypes_,
            num_states_
        );
    }

    CHECK_CUDA(cudaGetLastError());
    CHECK_CUDA(cudaStreamSynchronize(stream_));
}

size_t EmissionComputer::memory_usage() const {
    size_t reference_size = static_cast<size_t>(num_markers_) * num_haplotypes_ * sizeof(allele_t);
    return reference_size;
}

} // namespace kernels
} // namespace swiftimpute
