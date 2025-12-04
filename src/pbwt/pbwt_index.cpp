#include "pbwt_index.hpp"
#include <algorithm>
#include <numeric>
#include <vector>
#include <queue>

namespace swiftimpute {
namespace pbwt {

// PBWTIndex implementation

std::unique_ptr<PBWTIndex> PBWTIndex::build(
    const allele_t* reference_panel,
    marker_t num_markers,
    haplotype_t num_haplotypes
) {
    return PBWTBuilder::build(reference_panel, num_markers, num_haplotypes, true);
}

void PBWTIndex::select_states(
    marker_t m,
    const allele_t* target_sequence,
    uint32_t L,
    haplotype_t* selected_states
) const {
    std::vector<std::pair<marker_t, haplotype_t>> candidates;
    select_states_at_marker(m, target_sequence, L, candidates);

    // Sort by divergence (higher is better - longer match)
    std::sort(candidates.begin(), candidates.end(),
        [](const auto& a, const auto& b) {
            return a.first > b.first;
        });

    // Take top L
    for (uint32_t i = 0; i < L && i < candidates.size(); ++i) {
        selected_states[i] = candidates[i].second;
    }

    // Fill remaining with first haplotypes if not enough candidates
    for (uint32_t i = candidates.size(); i < L; ++i) {
        selected_states[i] = i % num_haplotypes_;
    }
}

void PBWTIndex::select_states_batch(
    marker_t start_marker,
    marker_t end_marker,
    const allele_t* target_sequence,
    uint32_t L,
    marker_t* selected_states
) const {
    for (marker_t m = start_marker; m < end_marker; ++m) {
        select_states(m, target_sequence, L, &selected_states[(m - start_marker) * L]);
    }
}

void PBWTIndex::select_states_at_marker(
    marker_t m,
    const allele_t* target_sequence,
    uint32_t L,
    std::vector<std::pair<marker_t, haplotype_t>>& candidates
) const {
    candidates.clear();

    if (m >= num_markers_) return;

    // Get target allele at this marker
    allele_t target_allele = (m < num_markers_) ? target_sequence[m] : 0;

    // Scan prefix array for matching haplotypes
    for (haplotype_t i = 0; i < num_haplotypes_; ++i) {
        haplotype_t hap = prefix_.at(m, i);
        marker_t div = divergence_.at(m, i);

        // Prioritize haplotypes that match longer
        candidates.push_back({div, hap});
    }
}

size_t PBWTIndex::memory_usage() const {
    size_t total = 0;

    // Prefix array
    total += prefix_.data.size() * sizeof(haplotype_t);

    // Divergence array
    total += divergence_.data.size() * sizeof(marker_t);

    return total;
}

// PBWTBuilder implementation

std::unique_ptr<PBWTIndex> PBWTBuilder::build(
    const allele_t* reference_panel,
    marker_t num_markers,
    haplotype_t num_haplotypes,
    bool parallel
) {
    LOG_INFO("Building PBWT index for " + std::to_string(num_markers) + " markers, " +
             std::to_string(num_haplotypes) + " haplotypes");

    auto index = std::make_unique<PBWTIndex>();
    index->num_markers_ = num_markers;
    index->num_haplotypes_ = num_haplotypes;

    // Allocate arrays
    index->prefix_.num_markers = num_markers;
    index->prefix_.num_haplotypes = num_haplotypes;
    index->prefix_.data.resize(static_cast<size_t>(num_markers) * num_haplotypes);

    index->divergence_.num_markers = num_markers;
    index->divergence_.num_haplotypes = num_haplotypes;
    index->divergence_.data.resize(static_cast<size_t>(num_markers) * num_haplotypes);

    if (parallel) {
        build_parallel(reference_panel, num_markers, num_haplotypes,
                      index->prefix_, index->divergence_);
    } else {
        // Sequential build
        std::vector<haplotype_t> prev_prefix(num_haplotypes);
        std::vector<marker_t> prev_divergence(num_haplotypes, 0);
        std::vector<haplotype_t> curr_prefix(num_haplotypes);
        std::vector<marker_t> curr_divergence(num_haplotypes);

        // Initialize first marker
        std::iota(prev_prefix.begin(), prev_prefix.end(), 0);
        std::fill(prev_divergence.begin(), prev_divergence.end(), 0);

        for (marker_t m = 0; m < num_markers; ++m) {
            build_marker(m, reference_panel, num_haplotypes,
                        prev_prefix.data(), prev_divergence.data(),
                        curr_prefix.data(), curr_divergence.data());

            // Copy to index
            for (haplotype_t h = 0; h < num_haplotypes; ++h) {
                index->prefix_.set(m, h, curr_prefix[h]);
                index->divergence_.set(m, h, curr_divergence[h]);
            }

            // Swap buffers
            std::swap(prev_prefix, curr_prefix);
            std::swap(prev_divergence, curr_divergence);
        }
    }

    LOG_INFO("PBWT index built successfully");

    return index;
}

void PBWTBuilder::build_marker(
    marker_t m,
    const allele_t* reference_panel,
    haplotype_t num_haplotypes,
    const haplotype_t* prev_prefix,
    const marker_t* prev_divergence,
    haplotype_t* curr_prefix,
    marker_t* curr_divergence
) {
    // Partition haplotypes by allele at marker m
    std::vector<haplotype_t> allele0_haps;
    std::vector<haplotype_t> allele1_haps;
    std::vector<marker_t> allele0_divs;
    std::vector<marker_t> allele1_divs;

    allele0_haps.reserve(num_haplotypes);
    allele1_haps.reserve(num_haplotypes);
    allele0_divs.reserve(num_haplotypes);
    allele1_divs.reserve(num_haplotypes);

    marker_t p = m;
    marker_t q = m;

    for (haplotype_t i = 0; i < num_haplotypes; ++i) {
        haplotype_t hap = prev_prefix[i];
        marker_t div = prev_divergence[i];

        // Get allele value at marker m for this haplotype
        allele_t allele = reference_panel[static_cast<size_t>(m) * num_haplotypes + hap];

        if (allele == 0) {
            allele0_haps.push_back(hap);
            allele0_divs.push_back(p);
            p = m;
        } else {
            allele1_haps.push_back(hap);
            allele1_divs.push_back(q);
            q = m;
        }
    }

    // Concatenate: allele0 first, then allele1
    haplotype_t idx = 0;

    for (size_t i = 0; i < allele0_haps.size(); ++i) {
        curr_prefix[idx] = allele0_haps[i];
        curr_divergence[idx] = allele0_divs[i];
        ++idx;
    }

    for (size_t i = 0; i < allele1_haps.size(); ++i) {
        curr_prefix[idx] = allele1_haps[i];
        curr_divergence[idx] = allele1_divs[i];
        ++idx;
    }
}

void PBWTBuilder::build_parallel(
    const allele_t* reference_panel,
    marker_t num_markers,
    haplotype_t num_haplotypes,
    PrefixArray& prefix,
    DivergenceArray& divergence
) {
    // For now, just call sequential version
    // TODO: Implement parallel version with thread pool

    std::vector<haplotype_t> prev_prefix(num_haplotypes);
    std::vector<marker_t> prev_divergence(num_haplotypes, 0);
    std::vector<haplotype_t> curr_prefix(num_haplotypes);
    std::vector<marker_t> curr_divergence(num_haplotypes);

    // Initialize
    std::iota(prev_prefix.begin(), prev_prefix.end(), 0);
    std::fill(prev_divergence.begin(), prev_divergence.end(), 0);

    for (marker_t m = 0; m < num_markers; ++m) {
        build_marker(m, reference_panel, num_haplotypes,
                    prev_prefix.data(), prev_divergence.data(),
                    curr_prefix.data(), curr_divergence.data());

        for (haplotype_t h = 0; h < num_haplotypes; ++h) {
            prefix.set(m, h, curr_prefix[h]);
            divergence.set(m, h, curr_divergence[h]);
        }

        std::swap(prev_prefix, curr_prefix);
        std::swap(prev_divergence, curr_divergence);

        if ((m + 1) % 1000 == 0) {
            LOG_INFO("Built PBWT for " + std::to_string(m + 1) + "/" +
                     std::to_string(num_markers) + " markers");
        }
    }
}

// StateSelector implementation

StateSelector::StateSelector(const PBWTIndex& index, uint32_t num_states)
    : index_(index), num_states_(num_states)
{}

void StateSelector::select_for_sample(
    const allele_t* target_haplotypes,
    marker_t num_markers,
    marker_t* selected_states
) const {
    // Select states for each marker
    for (marker_t m = 0; m < num_markers; ++m) {
        haplotype_t* states = reinterpret_cast<haplotype_t*>(&selected_states[m * num_states_]);
        index_.select_states(m, target_haplotypes, num_states_, states);
    }
}

void StateSelector::select_for_batch(
    const allele_t* target_haplotypes,
    uint32_t num_samples,
    marker_t num_markers,
    marker_t* selected_states
) const {
    // Process each sample sequentially
    // TODO: Parallelize with OpenMP or thread pool

    for (uint32_t s = 0; s < num_samples; ++s) {
        const allele_t* sample_haps = &target_haplotypes[s * 2 * num_markers];
        marker_t* sample_states = &selected_states[s * num_markers * num_states_];

        select_for_sample(sample_haps, num_markers, sample_states);
    }
}

void StateSelector::select_dynamic(
    const allele_t* target_haplotypes,
    marker_t num_markers,
    const uint32_t* num_states_per_marker,
    marker_t* selected_states,
    uint32_t* offsets
) const {
    // TODO: Implement dynamic state selection
    throw ImputationError("Dynamic state selection not yet implemented");
}

marker_t StateSelector::compute_divergence_score(
    marker_t m,
    haplotype_t haplotype,
    const allele_t* target_sequence
) const {
    // TODO: Implement divergence score computation
    return m;
}

// ============================================================================
// GPUStateSelector implementation
// Note: GPU kernel functions are declared in pbwt_index.hpp and implemented
// in pbwt_selector.cu
// ============================================================================

// GPUStateSelector implementation
GPUStateSelector::GPUStateSelector(
    const PBWTIndex& index,
    uint32_t num_states,
    int device_id
) : index_(index),
    num_states_(num_states),
    device_id_(device_id)
{
    allocate_device_memory();
    LOG_INFO("GPUStateSelector initialized for " + std::to_string(index.num_markers()) +
             " markers, " + std::to_string(index.num_haplotypes()) + " haplotypes, L=" +
             std::to_string(num_states));
}

GPUStateSelector::~GPUStateSelector() {
    free_device_memory();
}

void GPUStateSelector::allocate_device_memory() {
    haplotype_t* prefix_ptr = nullptr;
    marker_t* divergence_ptr = nullptr;

    gpu_state_selector_allocate(index_, device_id_, &prefix_ptr, &divergence_ptr);

    d_prefix_ = DevicePtr<haplotype_t>();
    d_divergence_ = DevicePtr<marker_t>();

    // Store raw pointers - we'll manage them ourselves
    // Note: DevicePtr doesn't support taking ownership of existing pointers,
    // so we store them directly and manage manually
    d_prefix_raw_ = prefix_ptr;
    d_divergence_raw_ = divergence_ptr;
    memory_allocated_ = true;
}

void GPUStateSelector::free_device_memory() {
    if (memory_allocated_) {
        gpu_state_selector_free(d_prefix_raw_, d_divergence_raw_);
        d_prefix_raw_ = nullptr;
        d_divergence_raw_ = nullptr;
        memory_allocated_ = false;
    }
}

void GPUStateSelector::transfer_index_to_device() {
    if (!memory_allocated_) {
        throw ImputationError("GPU memory not allocated for state selector");
    }

    LOG_INFO("Transferring PBWT index to GPU device " + std::to_string(device_id_));

    cudaStream_t stream;
    CHECK_CUDA(cudaStreamCreate(&stream));

    gpu_state_selector_transfer(index_, d_prefix_raw_, d_divergence_raw_, stream);

    CHECK_CUDA(cudaStreamDestroy(stream));

    index_on_device_ = true;
    LOG_INFO("PBWT index transfer complete");
}

void GPUStateSelector::select_on_device(
    const allele_t* d_target_haplotypes,
    uint32_t num_samples,
    marker_t num_markers,
    marker_t* d_selected_states,
    cudaStream_t stream
) {
    if (!index_on_device_) {
        throw ImputationError("PBWT index not transferred to device - call transfer_index_to_device() first");
    }

    // Cast marker_t* to haplotype_t* since they're both uint32_t
    // The kernel outputs haplotype indices into the selected_states array
    launch_select_states(
        d_prefix_raw_,
        d_divergence_raw_,
        d_target_haplotypes,
        num_samples,
        num_markers,
        index_.num_haplotypes(),
        num_states_,
        reinterpret_cast<haplotype_t*>(d_selected_states),
        stream
    );
}

void GPUStateSelector::select(
    const allele_t* h_target_haplotypes,
    uint32_t num_samples,
    marker_t num_markers,
    marker_t* h_selected_states
) {
    if (!index_on_device_) {
        transfer_index_to_device();
    }

    // Allocate device memory for target haplotypes and results
    size_t target_size = static_cast<size_t>(num_samples) * 2 * num_markers;
    size_t result_size = static_cast<size_t>(num_samples) * num_markers * num_states_;

    allele_t* d_target = nullptr;
    marker_t* d_selected = nullptr;

    CHECK_CUDA(cudaMalloc(&d_target, target_size * sizeof(allele_t)));
    CHECK_CUDA(cudaMalloc(&d_selected, result_size * sizeof(marker_t)));

    // Copy input to device
    CHECK_CUDA(cudaMemcpy(d_target, h_target_haplotypes,
                          target_size * sizeof(allele_t), cudaMemcpyHostToDevice));

    // Create stream for this operation
    cudaStream_t stream;
    CHECK_CUDA(cudaStreamCreate(&stream));

    // Run selection kernel
    select_on_device(d_target, num_samples, num_markers, d_selected, stream);

    // Synchronize
    CHECK_CUDA(cudaStreamSynchronize(stream));

    // Copy results back to host
    CHECK_CUDA(cudaMemcpy(h_selected_states, d_selected,
                          result_size * sizeof(marker_t), cudaMemcpyDeviceToHost));

    // Cleanup
    CHECK_CUDA(cudaStreamDestroy(stream));
    CHECK_CUDA(cudaFree(d_target));
    CHECK_CUDA(cudaFree(d_selected));
}

size_t GPUStateSelector::device_memory_usage() const {
    return gpu_state_selector_memory_usage(index_.num_markers(), index_.num_haplotypes());
}

} // namespace pbwt
} // namespace swiftimpute
