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

// GPUStateSelector implementation
GPUStateSelector::GPUStateSelector(
    const PBWTIndex& index,
    uint32_t num_states,
    int device_id
) : index_(index),
    num_states_(num_states),
    device_id_(device_id)
{
    // TODO: Implement GPU memory allocation
}

GPUStateSelector::~GPUStateSelector() {
    // TODO: Implement GPU memory cleanup
}

void GPUStateSelector::transfer_index_to_device() {
    // TODO: Implement index transfer to GPU
    throw ImputationError("GPU state selection not yet implemented");
}

void GPUStateSelector::select_on_device(
    const allele_t* d_target_haplotypes,
    uint32_t num_samples,
    marker_t num_markers,
    marker_t* d_selected_states,
    cudaStream_t stream
) {
    // TODO: Implement GPU state selection
    throw ImputationError("GPU state selection not yet implemented");
}

void GPUStateSelector::select(
    const allele_t* h_target_haplotypes,
    uint32_t num_samples,
    marker_t num_markers,
    marker_t* h_selected_states
) {
    // TODO: Implement host wrapper with data transfer
    throw ImputationError("GPU state selection not yet implemented");
}

size_t GPUStateSelector::device_memory_usage() const {
    // TODO: Implement memory usage calculation
    return 0;
}

void GPUStateSelector::allocate_device_memory() {
    // TODO: Implement device memory allocation
}

void GPUStateSelector::free_device_memory() {
    // TODO: Implement device memory deallocation
}

} // namespace pbwt
} // namespace swiftimpute
