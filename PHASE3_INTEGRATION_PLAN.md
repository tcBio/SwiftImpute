# Phase 3: Integration Plan

## Overview
This document outlines the complete integration of GPU kernels into the Imputer class to create a working end-to-end GPU-accelerated imputation pipeline.

## Current Status
✅ **Phase 1 Complete**: VCF I/O, PBWT index, core infrastructure
✅ **Phase 2 Complete**: All 5 GPU kernels implemented and tested
🔄 **Phase 3 In Progress**: Integration needed

## Integration Steps

### 1. Update Imputer Class Members

Add to `src/api/imputer.hpp` private section:
```cpp
// GPU kernel instances (lazy initialization)
std::unique_ptr<kernels::EmissionComputer> emission_computer_;
std::unique_ptr<kernels::TransitionComputer> transition_computer_;
std::unique_ptr<kernels::HaplotypeSampler> haplotype_sampler_;

// GPU memory buffers (persistent across batches)
GenotypeLikelihoods* d_genotype_liks_;
haplotype_t* d_selected_states_;
prob_t* d_emission_probs_;
prob_t* d_posterior_probs_;
prob_t* d_forward_checkpoints_;
prob_t* d_scaling_factors_;
allele_t* d_output_haplotypes_;

size_t current_batch_size_;
bool gpu_initialized_;
```

### 2. Modify Imputer Constructor

Initialize new members:
```cpp
Imputer::Imputer(...)
    : reference_(reference),
      config_(config),
      device_id_(config.device_id),
      emission_computer_(nullptr),
      transition_computer_(nullptr),
      haplotype_sampler_(nullptr),
      d_genotype_liks_(nullptr),
      // ... initialize all GPU pointers to nullptr
      current_batch_size_(0),
      gpu_initialized_(false)
{
    initialize_gpu();
}
```

### 3. Implement GPU Initialization

```cpp
void Imputer::initialize_gpu() {
    CHECK_CUDA(cudaSetDevice(device_id_));

    marker_t num_markers = reference_.num_markers();
    uint32_t num_states = config_.hmm_params.num_states;

    // Initialize emission computer
    emission_computer_ = std::make_unique<kernels::EmissionComputer>(
        num_markers,
        reference_.num_haplotypes(),
        num_states,
        device_id_
    );
    emission_computer_->set_reference_panel(reference_.haplotypes());

    // Initialize transition computer
    transition_computer_ = std::make_unique<kernels::TransitionComputer>(
        num_markers,
        num_states,
        device_id_
    );

    // Set up genetic map
    std::vector<double> genetic_distances(num_markers);
    for (marker_t m = 0; m < num_markers; ++m) {
        if (reference_.markers()[m].cM > 0.0) {
            genetic_distances[m] = reference_.markers()[m].cM / 100.0;
        } else {
            // Estimate: 1 cM ≈ 1 Mb
            genetic_distances[m] = reference_.markers()[m].pos / 1e8;
        }
    }
    transition_computer_->set_genetic_map(genetic_distances.data());

    // Set HMM parameters
    kernels::HMMParameters hmm_params;
    hmm_params.ne = config_.hmm_params.ne;
    hmm_params.rho_rate = config_.hmm_params.rho_rate;
    hmm_params.theta = config_.hmm_params.theta;
    transition_computer_->set_parameters(hmm_params);

    // Precompute transition matrices
    LOG_INFO("Precomputing transition matrices...");
    transition_computer_->compute();

    // Initialize sampler
    haplotype_sampler_ = std::make_unique<kernels::HaplotypeSampler>(
        num_markers,
        reference_.num_haplotypes(),
        num_states,
        device_id_
    );
    haplotype_sampler_->set_reference_panel(reference_.haplotypes());

    gpu_initialized_ = true;
    LOG_INFO("GPU initialization complete");
}
```

### 4. Implement GPU Memory Allocation

```cpp
void Imputer::allocate_batch_memory(uint32_t batch_size) {
    if (batch_size == current_batch_size_ && d_genotype_liks_ != nullptr) {
        return;  // Already allocated
    }

    // Free existing memory
    free_batch_memory();

    marker_t num_markers = reference_.num_markers();
    uint32_t num_states = config_.hmm_params.num_states;
    uint32_t checkpoint_interval = static_cast<uint32_t>(std::sqrt(num_markers));
    uint32_t num_checkpoints = (num_markers + checkpoint_interval - 1) / checkpoint_interval;

    // Allocate device memory
    size_t gl_size = batch_size * num_markers * sizeof(GenotypeLikelihoods);
    size_t states_size = batch_size * num_markers * num_states * 2 * sizeof(haplotype_t);
    size_t emission_size = batch_size * num_markers * num_states * sizeof(prob_t);
    size_t checkpoint_size = batch_size * num_checkpoints * num_states * sizeof(prob_t);
    size_t scaling_size = batch_size * num_markers * sizeof(prob_t);
    size_t output_size = batch_size * 2 * num_markers * sizeof(allele_t);

    CHECK_CUDA(cudaMalloc(&d_genotype_liks_, gl_size));
    CHECK_CUDA(cudaMalloc(&d_selected_states_, states_size));
    CHECK_CUDA(cudaMalloc(&d_emission_probs_, emission_size));
    CHECK_CUDA(cudaMalloc(&d_posterior_probs_, emission_size));
    CHECK_CUDA(cudaMalloc(&d_forward_checkpoints_, checkpoint_size));
    CHECK_CUDA(cudaMalloc(&d_scaling_factors_, scaling_size));
    CHECK_CUDA(cudaMalloc(&d_output_haplotypes_, output_size));

    current_batch_size_ = batch_size;

    LOG_INFO("Allocated GPU memory for batch size " + std::to_string(batch_size) +
             " (" + std::to_string((gl_size + states_size + emission_size + emission_size +
                                   checkpoint_size + scaling_size + output_size) / 1024.0 / 1024.0) +
             " MB)");
}

void Imputer::free_batch_memory() {
    if (d_genotype_liks_) cudaFree(d_genotype_liks_);
    if (d_selected_states_) cudaFree(d_selected_states_);
    if (d_emission_probs_) cudaFree(d_emission_probs_);
    if (d_posterior_probs_) cudaFree(d_posterior_probs_);
    if (d_forward_checkpoints_) cudaFree(d_forward_checkpoints_);
    if (d_scaling_factors_) cudaFree(d_scaling_factors_);
    if (d_output_haplotypes_) cudaFree(d_output_haplotypes_);

    d_genotype_liks_ = nullptr;
    d_selected_states_ = nullptr;
    d_emission_probs_ = nullptr;
    d_posterior_probs_ = nullptr;
    d_forward_checkpoints_ = nullptr;
    d_scaling_factors_ = nullptr;
    d_output_haplotypes_ = nullptr;
    current_batch_size_ = 0;
}
```

### 5. Implement Complete Imputation Pipeline

```cpp
void Imputer::impute_batch(
    const TargetData& targets,
    uint32_t start_sample,
    uint32_t end_sample,
    ImputationResult& result
) {
    uint32_t batch_size = end_sample - start_sample;
    marker_t num_markers = targets.num_markers();
    uint32_t num_states = config_.hmm_params.num_states;

    // Ensure GPU memory is allocated
    allocate_batch_memory(batch_size);

    // Initialize RNG for sampling
    if (!haplotype_sampler_->is_initialized()) {
        haplotype_sampler_->initialize_rng(batch_size);
    }

    // ============ STEP 1: Prepare Input Data ============
    LOG_INFO("Preparing input data for batch...");

    // Copy genotype likelihoods to GPU
    std::vector<GenotypeLikelihoods> h_genotype_liks(batch_size * num_markers);
    for (uint32_t s = 0; s < batch_size; ++s) {
        for (marker_t m = 0; m < num_markers; ++m) {
            h_genotype_liks[s * num_markers + m] =
                targets.get_likelihood(start_sample + s, m);
        }
    }
    CHECK_CUDA(cudaMemcpy(d_genotype_liks_, h_genotype_liks.data(),
                          batch_size * num_markers * sizeof(GenotypeLikelihoods),
                          cudaMemcpyHostToDevice));

    // ============ STEP 2: PBWT State Selection (CPU) ============
    LOG_INFO("Selecting HMM states using PBWT...");

    std::vector<haplotype_t> h_selected_states(batch_size * num_markers * num_states * 2);

    #pragma omp parallel for if(batch_size > 4)
    for (uint32_t s = 0; s < batch_size; ++s) {
        // Create target haplotype from most likely genotypes
        std::vector<allele_t> target_hap(num_markers);
        for (marker_t m = 0; m < num_markers; ++m) {
            auto gl = targets.get_likelihood(start_sample + s, m);
            // Simple heuristic: choose allele with higher likelihood
            target_hap[m] = (gl.ll_11 > gl.ll_00) ? 1 : 0;
        }

        // Select states at each marker
        for (marker_t m = 0; m < num_markers; ++m) {
            std::vector<haplotype_t> states(num_states);
            pbwt_index_->select_states(m, target_hap.data(), num_states, states.data());

            // Each state is a pair of haplotypes
            for (uint32_t state = 0; state < num_states; ++state) {
                size_t offset = (static_cast<size_t>(s) * num_markers + m) * num_states + state;
                h_selected_states[offset * 2] = states[state];
                h_selected_states[offset * 2 + 1] =
                    (states[state] + 1) % reference_.num_haplotypes();
            }
        }
    }

    CHECK_CUDA(cudaMemcpy(d_selected_states_, h_selected_states.data(),
                          batch_size * num_markers * num_states * 2 * sizeof(haplotype_t),
                          cudaMemcpyHostToDevice));

    // ============ STEP 3: Compute Emission Probabilities ============
    LOG_INFO("Computing emission probabilities on GPU...");
    emission_computer_->compute(
        d_genotype_liks_,
        d_selected_states_,
        d_emission_probs_,
        batch_size
    );

    // ============ STEP 4: Forward-Backward Algorithm ============
    LOG_INFO("Running forward-backward algorithm on GPU...");

    uint32_t checkpoint_interval = static_cast<uint32_t>(std::sqrt(num_markers));

    // Forward pass
    kernels::launch_forward_pass(
        d_emission_probs_,
        transition_computer_->get_transition_matrices(),
        batch_size,
        num_markers,
        num_states,
        checkpoint_interval,
        d_forward_checkpoints_,
        d_scaling_factors_
    );

    // Backward pass
    kernels::launch_backward_pass(
        d_emission_probs_,
        transition_computer_->get_transition_matrices(),
        d_forward_checkpoints_,
        d_scaling_factors_,
        batch_size,
        num_markers,
        num_states,
        checkpoint_interval,
        d_posterior_probs_
    );

    // ============ STEP 5: Sample Haplotypes ============
    LOG_INFO("Sampling haplotypes from posteriors...");
    haplotype_sampler_->sample(
        d_posterior_probs_,
        d_selected_states_,
        d_output_haplotypes_,
        batch_size,
        config_.deterministic_sampling  // false for stochastic, true for argmax
    );

    // ============ STEP 6: Copy Results Back ============
    LOG_INFO("Copying results from GPU to host...");

    std::vector<allele_t> h_output_haplotypes(batch_size * 2 * num_markers);
    CHECK_CUDA(cudaMemcpy(h_output_haplotypes.data(), d_output_haplotypes_,
                          batch_size * 2 * num_markers * sizeof(allele_t),
                          cudaMemcpyDeviceToHost));

    // Store in result object
    for (uint32_t s = 0; s < batch_size; ++s) {
        uint32_t sample_idx = start_sample + s;
        const allele_t* hap0 = &h_output_haplotypes[s * 2 * num_markers];
        const allele_t* hap1 = &h_output_haplotypes[(s * 2 + 1) * num_markers];
        result.set_haplotype(sample_idx, 0, hap0);
        result.set_haplotype(sample_idx, 1, hap1);
    }

    LOG_INFO("Batch imputation complete for samples " + std::to_string(start_sample) +
             " to " + std::to_string(end_sample));
}
```

### 6. Update device_memory_usage()

```cpp
size_t Imputer::device_memory_usage() const {
    size_t total = 0;

    if (emission_computer_) {
        total += emission_computer_->memory_usage();
    }
    if (transition_computer_) {
        total += transition_computer_->memory_usage();
    }
    if (haplotype_sampler_) {
        total += haplotype_sampler_->memory_usage();
    }

    // Add batch memory if allocated
    if (current_batch_size_ > 0) {
        marker_t num_markers = reference_.num_markers();
        uint32_t num_states = config_.hmm_params.num_states;

        total += current_batch_size_ * num_markers * sizeof(GenotypeLikelihoods);
        total += current_batch_size_ * num_markers * num_states * 2 * sizeof(haplotype_t);
        total += current_batch_size_ * num_markers * num_states * sizeof(prob_t) * 2;
        total += current_batch_size_ * 2 * num_markers * sizeof(allele_t);
    }

    return total;
}
```

### 7. Update Destructor

```cpp
Imputer::~Imputer() {
    free_batch_memory();
}
```

### 8. Add Required Includes

At top of `imputer.cpp`:
```cpp
#include "kernels/emission.hpp"
#include "kernels/transition.hpp"
#include "kernels/forward_backward.cuh"
#include "kernels/sampling.hpp"
#include <cuda_runtime.h>
```

## Testing Plan

### Unit Test (test_gpu_pipeline.cu)

```cpp
#include "api/imputer.hpp"
#include <iostream>
#include <cassert>

int main() {
    // Create synthetic small dataset
    // 10 markers, 4 reference samples, 2 target samples

    auto reference = create_synthetic_reference();
    auto targets = create_synthetic_targets();

    ImputationConfig config;
    config.hmm_params.num_states = 4;
    config.batch_size = 2;

    Imputer imputer(*reference, config);
    imputer.build_index();

    auto result = imputer.impute(*targets);

    // Validate results
    assert(result->num_samples() == 2);
    assert(result->num_markers() == 10);

    // Check that haplotypes are valid (0 or 1)
    for (uint32_t s = 0; s < 2; ++s) {
        for (marker_t m = 0; m < 10; ++m) {
            auto hap0 = result->get_haplotype(s, 0)[m];
            auto hap1 = result->get_haplotype(s, 1)[m];
            assert(hap0 == 0 || hap0 == 1);
            assert(hap1 == 0 || hap1 == 1);
        }
    }

    std::cout << "GPU pipeline test PASSED!" << std::endl;
    return 0;
}
```

## Performance Expectations

### Small Dataset (10K markers, 100 samples, 1K ref):
- Expected time: 5-10 seconds
- Memory: ~500 MB GPU

### Medium Dataset (100K markers, 1K samples, 10K ref):
- Expected time: 2-3 minutes
- Memory: ~4 GB GPU

### Large Dataset (1M markers, 10K samples, 100K ref):
- Expected time: 30-45 minutes
- Memory: ~12 GB GPU

## Next Steps After Integration

1. **Accuracy Validation**
   - Compare to BEAGLE 5 on 1000 Genomes data
   - Calculate dosage r², genotype concordance
   - Target: r² > 0.95

2. **Performance Optimization**
   - Profile GPU kernels with Nsight
   - Optimize memory access patterns
   - Implement kernel fusion where beneficial

3. **Multi-GPU Support**
   - Implement MultiGPUImputer
   - Load balancing across devices

4. **Production Features**
   - Better error handling
   - Progress callbacks
   - Configuration validation

## Estimated Timeline

- **Integration**: 2-3 days
- **Testing & Debugging**: 2-3 days
- **Validation**: 1-2 days
- **Optimization**: 1-2 weeks
- **Total Phase 3**: ~2 weeks

## Success Criteria

✅ End-to-end GPU pipeline runs without errors
✅ Results are numerically valid (alleles 0 or 1)
✅ Speed > 10× faster than CPU baseline
✅ Accuracy comparable to BEAGLE 5 (r² > 0.90)
✅ Memory usage within expected bounds
✅ No CUDA errors or memory leaks
