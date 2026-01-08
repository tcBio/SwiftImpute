#include "gpu_phaser.cuh"
#include "../pbwt/pbwt_index.hpp"
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cmath>

namespace swiftimpute {
namespace phasing {

// ============================================================================
// GPU Kernels
// ============================================================================

__global__ void phase_from_posteriors_kernel(
    const prob_t* __restrict__ posterior_probs,
    const GenotypeLikelihoods* __restrict__ geno_liks,
    const allele_t* __restrict__ ref_haplotypes,
    const haplotype_t* __restrict__ selected_states,
    uint32_t num_markers,
    uint32_t num_states,
    uint32_t num_haplotypes,
    allele_t* __restrict__ haplotype0,
    allele_t* __restrict__ haplotype1
) {
    // Each block handles one sample, threads collaborate on markers
    uint32_t sample_idx = blockIdx.x;
    uint32_t marker_offset = threadIdx.x + blockIdx.y * blockDim.x;

    if (marker_offset >= num_markers) return;

    // Get genotype likelihood for this sample/marker
    const GenotypeLikelihoods& gl = geno_liks[sample_idx * num_markers + marker_offset];

    // Check if missing data
    bool is_missing = (fabsf(gl.ll_00 - gl.ll_01) < 1e-6f &&
                       fabsf(gl.ll_01 - gl.ll_11) < 1e-6f);

    size_t out_idx = sample_idx * num_markers + marker_offset;

    if (is_missing) {
        haplotype0[out_idx] = ALLELE_MISSING;
        haplotype1[out_idx] = ALLELE_MISSING;
        return;
    }

    // Determine genotype
    bool is_hom_ref = (gl.ll_00 >= gl.ll_01 && gl.ll_00 >= gl.ll_11);
    bool is_hom_alt = (gl.ll_11 >= gl.ll_00 && gl.ll_11 >= gl.ll_01);

    if (is_hom_ref) {
        haplotype0[out_idx] = 0;
        haplotype1[out_idx] = 0;
        return;
    }

    if (is_hom_alt) {
        haplotype0[out_idx] = 1;
        haplotype1[out_idx] = 1;
        return;
    }

    // Heterozygous - use posterior probabilities to determine phase
    // Sum posterior probability that haplotype 0 carries ref allele
    prob_t prob_h0_ref = 0.0f;
    prob_t prob_h0_alt = 0.0f;

    size_t post_base = sample_idx * num_markers * num_states * num_states +
                       marker_offset * num_states * num_states;

    // For diploid HMM, state pairs are (s0, s1) where s0 is for haplotype0, s1 for haplotype1
    // Posterior prob for state pair (i, j) is at index i * num_states + j
    for (uint32_t s0 = 0; s0 < num_states; ++s0) {
        haplotype_t h0 = selected_states[marker_offset * num_states + s0];
        allele_t allele0 = ref_haplotypes[marker_offset * num_haplotypes + h0];

        for (uint32_t s1 = 0; s1 < num_states; ++s1) {
            prob_t post = posterior_probs[post_base + s0 * num_states + s1];

            if (allele0 == 0) {
                prob_h0_ref += post;
            } else {
                prob_h0_alt += post;
            }
        }
    }

    // Normalize and decide phase
    prob_t total = prob_h0_ref + prob_h0_alt;
    if (total > 0.0f) {
        prob_h0_ref /= total;
    }

    // Phase: if h0 more likely to carry ref, then 0|1, else 1|0
    if (prob_h0_ref >= 0.5f) {
        haplotype0[out_idx] = 0;
        haplotype1[out_idx] = 1;
    } else {
        haplotype0[out_idx] = 1;
        haplotype1[out_idx] = 0;
    }
}

__global__ void viterbi_phase_kernel(
    const prob_t* __restrict__ emission_probs,
    const prob_t* __restrict__ transition_probs,
    uint32_t num_markers,
    uint32_t num_states,
    prob_t* __restrict__ viterbi_probs,
    uint32_t* __restrict__ backpointers,
    uint32_t* __restrict__ best_state_pairs
) {
    // Each block handles one sample
    uint32_t sample_idx = blockIdx.x;
    uint32_t state_pair = threadIdx.x;  // Thread handles one state pair

    uint32_t num_state_pairs = num_states * num_states;
    if (state_pair >= num_state_pairs) return;

    extern __shared__ prob_t shared_mem[];
    prob_t* prev_probs = shared_mem;
    prob_t* curr_probs = shared_mem + num_state_pairs;

    size_t emit_base = sample_idx * num_markers * num_state_pairs;
    size_t vit_base = sample_idx * num_markers * num_state_pairs;
    size_t bp_base = sample_idx * num_markers * num_state_pairs;

    // Initialize first marker
    curr_probs[state_pair] = emission_probs[emit_base + state_pair];
    viterbi_probs[vit_base + state_pair] = curr_probs[state_pair];
    __syncthreads();

    // Forward pass with max
    for (uint32_t m = 1; m < num_markers; ++m) {
        // Swap buffers
        prob_t* temp = prev_probs;
        prev_probs = curr_probs;
        curr_probs = temp;

        __syncthreads();

        // Find best previous state pair
        uint32_t s0 = state_pair / num_states;
        uint32_t s1 = state_pair % num_states;

        prob_t best_prev = -INFINITY;
        uint32_t best_prev_idx = 0;

        // Check all previous state pairs (can transition independently)
        for (uint32_t prev_s0 = 0; prev_s0 < num_states; ++prev_s0) {
            prob_t trans0 = transition_probs[(m - 1) * num_states * num_states +
                                             prev_s0 * num_states + s0];

            for (uint32_t prev_s1 = 0; prev_s1 < num_states; ++prev_s1) {
                prob_t trans1 = transition_probs[(m - 1) * num_states * num_states +
                                                 prev_s1 * num_states + s1];

                uint32_t prev_pair = prev_s0 * num_states + prev_s1;
                prob_t score = prev_probs[prev_pair] + logf(trans0) + logf(trans1);

                if (score > best_prev) {
                    best_prev = score;
                    best_prev_idx = prev_pair;
                }
            }
        }

        // Add emission
        prob_t emit = emission_probs[emit_base + m * num_state_pairs + state_pair];
        curr_probs[state_pair] = best_prev + emit;

        viterbi_probs[vit_base + m * num_state_pairs + state_pair] = curr_probs[state_pair];
        backpointers[bp_base + m * num_state_pairs + state_pair] = best_prev_idx;

        __syncthreads();
    }

    // Final: find best ending state (one thread does this)
    if (state_pair == 0) {
        prob_t best_final = -INFINITY;
        uint32_t best_final_idx = 0;

        for (uint32_t sp = 0; sp < num_state_pairs; ++sp) {
            if (curr_probs[sp] > best_final) {
                best_final = curr_probs[sp];
                best_final_idx = sp;
            }
        }

        // Traceback
        uint32_t current_state = best_final_idx;
        for (int32_t m = num_markers - 1; m >= 0; --m) {
            best_state_pairs[sample_idx * num_markers + m] = current_state;
            if (m > 0) {
                current_state = backpointers[bp_base + m * num_state_pairs + current_state];
            }
        }
    }
}

__global__ void decode_haplotypes_kernel(
    const uint32_t* __restrict__ best_state_pairs,
    const haplotype_t* __restrict__ selected_states,
    const allele_t* __restrict__ ref_haplotypes,
    uint32_t num_markers,
    uint32_t num_states,
    uint32_t num_haplotypes,
    allele_t* __restrict__ haplotype0,
    allele_t* __restrict__ haplotype1
) {
    uint32_t sample_idx = blockIdx.x;
    uint32_t marker = threadIdx.x + blockIdx.y * blockDim.x;

    if (marker >= num_markers) return;

    uint32_t state_pair = best_state_pairs[sample_idx * num_markers + marker];
    uint32_t s0 = state_pair / num_states;
    uint32_t s1 = state_pair % num_states;

    // Get actual haplotype indices
    haplotype_t h0 = selected_states[marker * num_states + s0];
    haplotype_t h1 = selected_states[marker * num_states + s1];

    // Get alleles from reference panel
    size_t out_idx = sample_idx * num_markers + marker;
    haplotype0[out_idx] = ref_haplotypes[marker * num_haplotypes + h0];
    haplotype1[out_idx] = ref_haplotypes[marker * num_haplotypes + h1];
}

// ============================================================================
// Host Wrappers
// ============================================================================

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
    cudaStream_t stream
) {
    dim3 block(256);
    dim3 grid(num_samples, (num_markers + block.x - 1) / block.x);

    phase_from_posteriors_kernel<<<grid, block, 0, stream>>>(
        d_posterior_probs, d_geno_liks, d_ref_haplotypes, d_selected_states,
        num_markers, num_states, num_haplotypes,
        d_haplotype0, d_haplotype1
    );
}

void launch_viterbi_phase(
    const prob_t* d_emission_probs,
    const prob_t* d_transition_probs,
    uint32_t num_samples,
    uint32_t num_markers,
    uint32_t num_states,
    prob_t* d_viterbi_probs,
    uint32_t* d_backpointers,
    uint32_t* d_best_state_pairs,
    cudaStream_t stream
) {
    uint32_t num_state_pairs = num_states * num_states;

    // Limit block size for shared memory
    uint32_t block_size = std::min(num_state_pairs, 256u);
    size_t shared_mem = 2 * num_state_pairs * sizeof(prob_t);

    dim3 block(block_size);
    dim3 grid(num_samples);

    viterbi_phase_kernel<<<grid, block, shared_mem, stream>>>(
        d_emission_probs, d_transition_probs,
        num_markers, num_states,
        d_viterbi_probs, d_backpointers, d_best_state_pairs
    );
}

void launch_decode_haplotypes(
    const uint32_t* d_best_state_pairs,
    const haplotype_t* d_selected_states,
    const allele_t* d_ref_haplotypes,
    uint32_t num_samples,
    uint32_t num_markers,
    uint32_t num_states,
    uint32_t num_haplotypes,
    allele_t* d_haplotype0,
    allele_t* d_haplotype1,
    cudaStream_t stream
) {
    dim3 block(256);
    dim3 grid(num_samples, (num_markers + block.x - 1) / block.x);

    decode_haplotypes_kernel<<<grid, block, 0, stream>>>(
        d_best_state_pairs, d_selected_states, d_ref_haplotypes,
        num_markers, num_states, num_haplotypes,
        d_haplotype0, d_haplotype1
    );
}

// ============================================================================
// GPUPhaser Class Implementation
// ============================================================================

GPUPhaser::GPUPhaser(
    const ReferencePanel& reference,
    const Config& config
) : reference_(reference),
    config_(config),
    device_id_(-1),
    stream_(nullptr),
    d_geno_liks_(nullptr),
    d_ref_haplotypes_(nullptr),
    d_selected_states_(nullptr),
    d_genetic_distances_(nullptr),
    d_emission_probs_(nullptr),
    d_transition_probs_(nullptr),
    d_forward_checkpoints_(nullptr),
    d_scaling_factors_(nullptr),
    d_posterior_probs_(nullptr),
    d_haplotype0_(nullptr),
    d_haplotype1_(nullptr),
    d_viterbi_probs_(nullptr),
    d_backpointers_(nullptr),
    d_best_state_pairs_(nullptr),
    buffer_size_samples_(0),
    buffer_size_markers_(0)
{
    // Select GPU
    device_id_ = (config_.device_id >= 0) ? config_.device_id : select_best_device();
    cudaSetDevice(device_id_);

    // Create stream
    cudaStreamCreate(&stream_);

    LOG_INFO("GPUPhaser initialized on device " + std::to_string(device_id_) +
             " with " + std::to_string(config_.num_states) + " states");
}

GPUPhaser::~GPUPhaser() {
    free_buffers();
    if (stream_) {
        cudaStreamDestroy(stream_);
    }
}

void GPUPhaser::allocate_buffers(uint32_t num_samples, uint32_t num_markers) {
    if (num_samples == buffer_size_samples_ && num_markers == buffer_size_markers_) {
        return;  // Already allocated
    }

    free_buffers();

    uint32_t num_states = config_.num_states;
    uint32_t num_state_pairs = num_states * num_states;
    uint32_t num_haplotypes = reference_.num_haplotypes();

    // Allocate device memory
    cudaMalloc(&d_geno_liks_, num_samples * num_markers * sizeof(GenotypeLikelihoods));
    cudaMalloc(&d_ref_haplotypes_, num_markers * num_haplotypes * sizeof(allele_t));
    cudaMalloc(&d_selected_states_, num_markers * num_states * sizeof(haplotype_t));
    cudaMalloc(&d_genetic_distances_, num_markers * sizeof(double));
    cudaMalloc(&d_emission_probs_, num_samples * num_markers * num_state_pairs * sizeof(prob_t));
    cudaMalloc(&d_transition_probs_, (num_markers - 1) * num_states * num_states * sizeof(prob_t));

    // Forward-backward buffers
    uint32_t checkpoint_interval = std::max(1u, static_cast<uint32_t>(std::sqrt(num_markers)));
    uint32_t num_checkpoints = (num_markers + checkpoint_interval - 1) / checkpoint_interval;
    cudaMalloc(&d_forward_checkpoints_, num_samples * num_checkpoints * num_state_pairs * sizeof(prob_t));
    cudaMalloc(&d_scaling_factors_, num_samples * num_markers * sizeof(prob_t));
    cudaMalloc(&d_posterior_probs_, num_samples * num_markers * num_state_pairs * sizeof(prob_t));

    // Output haplotypes
    cudaMalloc(&d_haplotype0_, num_samples * num_markers * sizeof(allele_t));
    cudaMalloc(&d_haplotype1_, num_samples * num_markers * sizeof(allele_t));

    // Viterbi buffers (if using)
    if (config_.use_viterbi) {
        cudaMalloc(&d_viterbi_probs_, num_samples * num_markers * num_state_pairs * sizeof(prob_t));
        cudaMalloc(&d_backpointers_, num_samples * num_markers * num_state_pairs * sizeof(uint32_t));
        cudaMalloc(&d_best_state_pairs_, num_samples * num_markers * sizeof(uint32_t));
    }

    buffer_size_samples_ = num_samples;
    buffer_size_markers_ = num_markers;
}

void GPUPhaser::free_buffers() {
    if (d_geno_liks_) cudaFree(d_geno_liks_);
    if (d_ref_haplotypes_) cudaFree(d_ref_haplotypes_);
    if (d_selected_states_) cudaFree(d_selected_states_);
    if (d_genetic_distances_) cudaFree(d_genetic_distances_);
    if (d_emission_probs_) cudaFree(d_emission_probs_);
    if (d_transition_probs_) cudaFree(d_transition_probs_);
    if (d_forward_checkpoints_) cudaFree(d_forward_checkpoints_);
    if (d_scaling_factors_) cudaFree(d_scaling_factors_);
    if (d_posterior_probs_) cudaFree(d_posterior_probs_);
    if (d_haplotype0_) cudaFree(d_haplotype0_);
    if (d_haplotype1_) cudaFree(d_haplotype1_);
    if (d_viterbi_probs_) cudaFree(d_viterbi_probs_);
    if (d_backpointers_) cudaFree(d_backpointers_);
    if (d_best_state_pairs_) cudaFree(d_best_state_pairs_);

    d_geno_liks_ = nullptr;
    d_ref_haplotypes_ = nullptr;
    d_selected_states_ = nullptr;
    d_genetic_distances_ = nullptr;
    d_emission_probs_ = nullptr;
    d_transition_probs_ = nullptr;
    d_forward_checkpoints_ = nullptr;
    d_scaling_factors_ = nullptr;
    d_posterior_probs_ = nullptr;
    d_haplotype0_ = nullptr;
    d_haplotype1_ = nullptr;
    d_viterbi_probs_ = nullptr;
    d_backpointers_ = nullptr;
    d_best_state_pairs_ = nullptr;

    buffer_size_samples_ = 0;
    buffer_size_markers_ = 0;
}

GPUPhaser::PhasedResult GPUPhaser::phase(const TargetData& targets) {
    return phase_with_progress(targets, nullptr);
}

GPUPhaser::PhasedResult GPUPhaser::phase_with_progress(
    const TargetData& targets,
    std::function<void(uint32_t, uint32_t)> progress_callback
) {
    uint32_t num_samples = targets.num_samples();
    uint32_t num_markers = targets.num_markers();
    uint32_t num_states = config_.num_states;
    uint32_t num_haplotypes = reference_.num_haplotypes();

    LOG_INFO("GPU phasing " + std::to_string(num_samples) + " samples, " +
             std::to_string(num_markers) + " markers");

    // Allocate GPU buffers
    allocate_buffers(num_samples, num_markers);

    // Build PBWT index and select states
    auto pbwt_index = pbwt::PBWTIndex::build(
        reference_.haplotypes(),
        reference_.num_markers(),
        reference_.num_haplotypes()
    );

    // Get selected states for each marker
    std::vector<haplotype_t> selected_states(num_markers * num_states);
    for (marker_t m = 0; m < num_markers; ++m) {
        auto states = pbwt_index->select_states(m, num_states);
        for (uint32_t s = 0; s < num_states && s < states.size(); ++s) {
            selected_states[m * num_states + s] = states[s];
        }
    }

    if (progress_callback) progress_callback(1, 5);

    // Copy data to GPU
    cudaMemcpyAsync(d_geno_liks_, targets.genotype_likelihoods(),
                    num_samples * num_markers * sizeof(GenotypeLikelihoods),
                    cudaMemcpyHostToDevice, stream_);

    cudaMemcpyAsync(d_ref_haplotypes_, reference_.haplotypes(),
                    num_markers * num_haplotypes * sizeof(allele_t),
                    cudaMemcpyHostToDevice, stream_);

    cudaMemcpyAsync(d_selected_states_, selected_states.data(),
                    num_markers * num_states * sizeof(haplotype_t),
                    cudaMemcpyHostToDevice, stream_);

    if (progress_callback) progress_callback(2, 5);

    // Compute emission probabilities
    prob_t theta = 0.001f;  // Mutation rate
    kernels::launch_compute_emissions(
        d_geno_liks_,
        d_ref_haplotypes_,
        reinterpret_cast<marker_t*>(d_selected_states_),
        num_samples,
        num_markers,
        num_states,
        theta,
        d_emission_probs_,
        stream_
    );

    if (progress_callback) progress_callback(3, 5);

    // Compute transition probabilities
    std::vector<double> genetic_distances(num_markers);
    const auto& markers = reference_.markers();
    for (marker_t m = 1; m < num_markers; ++m) {
        // Simple genetic distance estimate: 1 cM per Mb
        double dist_bp = static_cast<double>(markers[m].pos - markers[m-1].pos);
        genetic_distances[m] = dist_bp / 1e6;  // cM (assuming 1cM/Mb)
    }

    cudaMemcpyAsync(d_genetic_distances_, genetic_distances.data(),
                    num_markers * sizeof(double),
                    cudaMemcpyHostToDevice, stream_);

    kernels::launch_compute_transitions(
        d_genetic_distances_,
        num_markers,
        num_states,
        config_.ne,
        4.0f,  // rho_rate = 4 * Ne * r
        d_transition_probs_,
        stream_
    );

    if (progress_callback) progress_callback(4, 5);

    // Run phasing
    if (config_.use_viterbi) {
        // Viterbi decoding for optimal phasing
        launch_viterbi_phase(
            d_emission_probs_,
            d_transition_probs_,
            num_samples,
            num_markers,
            num_states,
            d_viterbi_probs_,
            d_backpointers_,
            d_best_state_pairs_,
            stream_
        );

        launch_decode_haplotypes(
            d_best_state_pairs_,
            d_selected_states_,
            d_ref_haplotypes_,
            num_samples,
            num_markers,
            num_states,
            num_haplotypes,
            d_haplotype0_,
            d_haplotype1_,
            stream_
        );
    } else {
        // Forward-backward with posterior sampling
        uint32_t checkpoint_interval = std::max(1u, static_cast<uint32_t>(std::sqrt(num_markers)));

        kernels::launch_forward_pass(
            d_emission_probs_,
            d_transition_probs_,
            reinterpret_cast<marker_t*>(d_selected_states_),
            num_samples,
            num_markers,
            num_states,
            checkpoint_interval,
            d_forward_checkpoints_,
            d_scaling_factors_,
            stream_
        );

        kernels::launch_backward_pass(
            d_emission_probs_,
            d_transition_probs_,
            reinterpret_cast<marker_t*>(d_selected_states_),
            d_forward_checkpoints_,
            d_scaling_factors_,
            num_samples,
            num_markers,
            num_states,
            checkpoint_interval,
            d_posterior_probs_,
            stream_
        );

        launch_phase_from_posteriors(
            d_posterior_probs_,
            d_geno_liks_,
            d_ref_haplotypes_,
            d_selected_states_,
            num_samples,
            num_markers,
            num_states,
            num_haplotypes,
            d_haplotype0_,
            d_haplotype1_,
            stream_
        );
    }

    // Copy results back
    PhasedResult result;
    result.haplotype0.resize(num_samples * num_markers);
    result.haplotype1.resize(num_samples * num_markers);
    result.phase_confidence.resize(num_samples * num_markers, 1.0f);

    cudaMemcpyAsync(result.haplotype0.data(), d_haplotype0_,
                    num_samples * num_markers * sizeof(allele_t),
                    cudaMemcpyDeviceToHost, stream_);

    cudaMemcpyAsync(result.haplotype1.data(), d_haplotype1_,
                    num_samples * num_markers * sizeof(allele_t),
                    cudaMemcpyDeviceToHost, stream_);

    cudaStreamSynchronize(stream_);

    if (progress_callback) progress_callback(5, 5);

    LOG_INFO("GPU phasing complete");

    return result;
}

size_t GPUPhaser::estimate_memory(
    uint32_t num_samples,
    uint32_t num_markers,
    uint32_t num_states
) {
    uint32_t num_state_pairs = num_states * num_states;

    size_t total = 0;
    total += num_samples * num_markers * sizeof(GenotypeLikelihoods);  // geno_liks
    total += num_samples * num_markers * num_state_pairs * sizeof(prob_t);  // emission
    total += (num_markers - 1) * num_states * num_states * sizeof(prob_t);  // transition
    total += num_samples * num_markers * num_state_pairs * sizeof(prob_t);  // posterior
    total += num_samples * num_markers * 2 * sizeof(allele_t);  // haplotypes

    // Viterbi buffers
    total += num_samples * num_markers * num_state_pairs * sizeof(prob_t);  // viterbi
    total += num_samples * num_markers * num_state_pairs * sizeof(uint32_t);  // backpointers

    return total;
}

size_t GPUPhaser::get_device_memory_usage() const {
    return estimate_memory(buffer_size_samples_, buffer_size_markers_, config_.num_states);
}

} // namespace phasing
} // namespace swiftimpute
