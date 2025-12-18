#pragma once

#include "pbwt_index.hpp"
#include <variant>

namespace swiftimpute {
namespace pbwt {

// Compact types for smaller panels (< 65,536 markers/haplotypes)
using compact_haplotype_t = uint16_t;
using compact_marker_t = uint16_t;

constexpr uint32_t COMPACT_THRESHOLD = 65535;

/**
 * @brief Check if panel can use compact representation
 */
inline bool can_use_compact(marker_t num_markers, haplotype_t num_haplotypes) {
    return num_markers <= COMPACT_THRESHOLD && num_haplotypes <= COMPACT_THRESHOLD;
}

/**
 * @brief Compact prefix array using 16-bit indices
 */
struct CompactPrefixArray {
    std::vector<compact_haplotype_t> data;
    compact_marker_t num_markers;
    compact_haplotype_t num_haplotypes;

    CompactPrefixArray() : num_markers(0), num_haplotypes(0) {}

    compact_haplotype_t at(compact_marker_t m, uint32_t i) const {
        return data[static_cast<size_t>(m) * num_haplotypes + i];
    }

    void set(compact_marker_t m, uint32_t i, compact_haplotype_t value) {
        data[static_cast<size_t>(m) * num_haplotypes + i] = value;
    }
};

/**
 * @brief Compact divergence array using 16-bit indices
 */
struct CompactDivergenceArray {
    std::vector<compact_marker_t> data;
    compact_marker_t num_markers;
    compact_haplotype_t num_haplotypes;

    CompactDivergenceArray() : num_markers(0), num_haplotypes(0) {}

    compact_marker_t at(compact_marker_t m, uint32_t i) const {
        return data[static_cast<size_t>(m) * num_haplotypes + i];
    }

    void set(compact_marker_t m, uint32_t i, compact_marker_t value) {
        data[static_cast<size_t>(m) * num_haplotypes + i] = value;
    }
};

/**
 * @brief Compact PBWT index using 16-bit indices
 *
 * Uses half the memory of the standard PBWTIndex for panels with
 * fewer than 65,536 markers and haplotypes.
 *
 * Memory comparison:
 * - Standard: 8 bytes per (marker, haplotype) pair
 * - Compact:  4 bytes per (marker, haplotype) pair
 *
 * Example savings:
 * - 100K markers × 10K haplotypes: 8 GB → 4 GB (50% reduction)
 */
class CompactPBWTIndex {
public:
    CompactPBWTIndex() = default;

    /**
     * @brief Build compact PBWT from reference panel
     *
     * @throws ImputationError if panel exceeds compact size limits
     */
    static std::unique_ptr<CompactPBWTIndex> build(
        const allele_t* reference_panel,
        marker_t num_markers,
        haplotype_t num_haplotypes
    );

    // Accessors
    const CompactPrefixArray& prefix() const { return prefix_; }
    const CompactDivergenceArray& divergence() const { return divergence_; }

    compact_marker_t num_markers() const { return num_markers_; }
    compact_haplotype_t num_haplotypes() const { return num_haplotypes_; }

    // Full-width accessors for API compatibility
    marker_t num_markers_full() const { return static_cast<marker_t>(num_markers_); }
    haplotype_t num_haplotypes_full() const { return static_cast<haplotype_t>(num_haplotypes_); }

    /**
     * @brief Select L best matching states for target haplotype at marker m
     */
    void select_states(
        compact_marker_t m,
        const allele_t* target_sequence,
        uint32_t L,
        haplotype_t* selected_states  // Output uses full width for compatibility
    ) const;

    /**
     * @brief Memory usage in bytes
     */
    size_t memory_usage() const;

    /**
     * @brief Comparison with standard index memory
     */
    size_t memory_savings_vs_standard() const {
        // Standard uses 8 bytes per element, compact uses 4
        size_t standard_size = static_cast<size_t>(num_markers_) * num_haplotypes_ * 8;
        return standard_size - memory_usage();
    }

    /**
     * @brief Convert to standard PBWTIndex (for GPU operations)
     *
     * Some GPU kernels may require 32-bit indices for compatibility.
     */
    std::unique_ptr<PBWTIndex> to_standard() const;

private:
    friend class CompactPBWTBuilder;

    CompactPrefixArray prefix_;
    CompactDivergenceArray divergence_;
    compact_marker_t num_markers_;
    compact_haplotype_t num_haplotypes_;

    void select_states_at_marker(
        compact_marker_t m,
        const allele_t* target_sequence,
        uint32_t L,
        std::vector<std::pair<compact_marker_t, compact_haplotype_t>>& candidates
    ) const;
};

/**
 * @brief Builder for compact PBWT index
 */
class CompactPBWTBuilder {
public:
    static std::unique_ptr<CompactPBWTIndex> build(
        const allele_t* reference_panel,
        marker_t num_markers,
        haplotype_t num_haplotypes,
        bool parallel = true
    );

private:
    static void build_marker(
        compact_marker_t m,
        const allele_t* reference_panel,
        compact_haplotype_t num_haplotypes,
        const compact_haplotype_t* prev_prefix,
        const compact_marker_t* prev_divergence,
        compact_haplotype_t* curr_prefix,
        compact_marker_t* curr_divergence
    );

    static void build_parallel(
        const allele_t* reference_panel,
        compact_marker_t num_markers,
        compact_haplotype_t num_haplotypes,
        CompactPrefixArray& prefix,
        CompactDivergenceArray& divergence
    );
};

/**
 * @brief Unified PBWT index that auto-selects compact or standard representation
 *
 * Usage:
 *   auto index = AdaptivePBWTIndex::build(panel, num_markers, num_haplotypes);
 *   std::cout << "Using compact: " << index.is_compact() << std::endl;
 *   std::cout << "Memory: " << index.memory_usage() << " bytes" << std::endl;
 */
class AdaptivePBWTIndex {
public:
    /**
     * @brief Build index, automatically selecting compact or standard
     */
    static std::unique_ptr<AdaptivePBWTIndex> build(
        const allele_t* reference_panel,
        marker_t num_markers,
        haplotype_t num_haplotypes
    );

    /**
     * @brief Force compact representation (throws if panel too large)
     */
    static std::unique_ptr<AdaptivePBWTIndex> build_compact(
        const allele_t* reference_panel,
        marker_t num_markers,
        haplotype_t num_haplotypes
    );

    /**
     * @brief Force standard representation
     */
    static std::unique_ptr<AdaptivePBWTIndex> build_standard(
        const allele_t* reference_panel,
        marker_t num_markers,
        haplotype_t num_haplotypes
    );

    // Check representation type
    bool is_compact() const { return std::holds_alternative<std::unique_ptr<CompactPBWTIndex>>(index_); }
    bool is_standard() const { return std::holds_alternative<std::unique_ptr<PBWTIndex>>(index_); }

    // Accessors (return full-width types)
    marker_t num_markers() const;
    haplotype_t num_haplotypes() const;
    size_t memory_usage() const;

    /**
     * @brief Select states (works with either representation)
     */
    void select_states(
        marker_t m,
        const allele_t* target_sequence,
        uint32_t L,
        haplotype_t* selected_states
    ) const;

    /**
     * @brief Get standard index for GPU operations
     *
     * If currently compact, converts to standard (expensive).
     * Caches the conversion for subsequent calls.
     */
    const PBWTIndex& get_standard_index() const;

    /**
     * @brief Get memory savings compared to always using standard
     */
    size_t memory_savings() const;

private:
    using IndexVariant = std::variant<
        std::unique_ptr<CompactPBWTIndex>,
        std::unique_ptr<PBWTIndex>
    >;

    IndexVariant index_;
    mutable std::unique_ptr<PBWTIndex> cached_standard_;  // Lazy conversion cache
};

/**
 * @brief GPU-accelerated compact state selector
 *
 * Supports both compact and standard indices on GPU.
 * Uses 16-bit indices when possible for reduced memory bandwidth.
 */
class CompactGPUStateSelector {
public:
    CompactGPUStateSelector(
        const AdaptivePBWTIndex& index,
        uint32_t num_states,
        int device_id = 0
    );

    ~CompactGPUStateSelector();

    void transfer_index_to_device();

    void select_on_device(
        const allele_t* d_target_haplotypes,
        uint32_t num_samples,
        marker_t num_markers,
        haplotype_t* d_selected_states,
        cudaStream_t stream = 0
    );

    size_t device_memory_usage() const;
    bool is_using_compact() const { return using_compact_; }

private:
    const AdaptivePBWTIndex& index_;
    uint32_t num_states_;
    int device_id_;
    bool using_compact_;
    bool index_on_device_;

    // Device memory - compact version
    compact_haplotype_t* d_compact_prefix_ = nullptr;
    compact_marker_t* d_compact_divergence_ = nullptr;

    // Device memory - standard version
    haplotype_t* d_standard_prefix_ = nullptr;
    marker_t* d_standard_divergence_ = nullptr;

    void allocate_device_memory();
    void free_device_memory();
};

// GPU kernel declarations for compact indices
void launch_select_states_compact(
    const compact_haplotype_t* d_prefix,
    const compact_marker_t* d_divergence,
    const allele_t* d_target_haplotypes,
    uint32_t num_samples,
    uint32_t num_markers,
    uint32_t num_haplotypes,
    uint32_t num_states_L,
    haplotype_t* d_selected_states,  // Output still uses 32-bit for compatibility
    cudaStream_t stream = 0
);

/**
 * @brief Utility: estimate memory for PBWT index
 */
inline size_t estimate_pbwt_memory(marker_t num_markers, haplotype_t num_haplotypes) {
    size_t elements = static_cast<size_t>(num_markers) * num_haplotypes;
    if (can_use_compact(num_markers, num_haplotypes)) {
        return elements * 4;  // 2 arrays × 2 bytes each
    } else {
        return elements * 8;  // 2 arrays × 4 bytes each
    }
}

/**
 * @brief Utility: format memory size for display
 */
inline std::string format_memory_size(size_t bytes) {
    if (bytes >= 1024ULL * 1024 * 1024) {
        return std::to_string(bytes / (1024ULL * 1024 * 1024)) + " GB";
    } else if (bytes >= 1024ULL * 1024) {
        return std::to_string(bytes / (1024ULL * 1024)) + " MB";
    } else if (bytes >= 1024ULL) {
        return std::to_string(bytes / 1024ULL) + " KB";
    } else {
        return std::to_string(bytes) + " bytes";
    }
}

} // namespace pbwt
} // namespace swiftimpute
