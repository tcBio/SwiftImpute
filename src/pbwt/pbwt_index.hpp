#pragma once

#include "core/types.hpp"
#include <vector>
#include <memory>
#include <cuda_runtime.h>

namespace swiftimpute {
namespace pbwt {

// PBWT data structures

// Prefix array: sorted order of haplotypes by matching prefix
struct PrefixArray {
    std::vector<haplotype_t> data;    // [num_markers][num_haplotypes]
    marker_t num_markers;
    haplotype_t num_haplotypes;
    
    PrefixArray() : num_markers(0), num_haplotypes(0) {}
    
    // Access element at marker m, position i
    haplotype_t at(marker_t m, uint32_t i) const {
        return data[m * num_haplotypes + i];
    }
    
    void set(marker_t m, uint32_t i, haplotype_t value) {
        data[m * num_haplotypes + i] = value;
    }
};

// Divergence array: last marker where prefix differs
struct DivergenceArray {
    std::vector<marker_t> data;       // [num_markers][num_haplotypes]
    marker_t num_markers;
    haplotype_t num_haplotypes;
    
    DivergenceArray() : num_markers(0), num_haplotypes(0) {}
    
    marker_t at(marker_t m, uint32_t i) const {
        return data[m * num_haplotypes + i];
    }
    
    void set(marker_t m, uint32_t i, marker_t value) {
        data[m * num_haplotypes + i] = value;
    }
};

// Complete PBWT index
class PBWTIndex {
public:
    PBWTIndex() = default;

    // Build from reference panel
    static std::unique_ptr<PBWTIndex> build(
        const allele_t* reference_panel,
        marker_t num_markers,
        haplotype_t num_haplotypes
    );

    // Accessors
    const PrefixArray& prefix() const { return prefix_; }
    const DivergenceArray& divergence() const { return divergence_; }

    marker_t num_markers() const { return num_markers_; }
    haplotype_t num_haplotypes() const { return num_haplotypes_; }

    // Select L best matching states for target haplotype at marker m
    void select_states(
        marker_t m,
        const allele_t* target_sequence,
        uint32_t L,
        haplotype_t* selected_states
    ) const;

    // Batch select states for multiple markers
    void select_states_batch(
        marker_t start_marker,
        marker_t end_marker,
        const allele_t* target_sequence,
        uint32_t L,
        marker_t* selected_states    // [num_markers][L]
    ) const;

    // Memory usage
    size_t memory_usage() const;

private:
    friend class PBWTBuilder;  // Allow builder to access private members

    PrefixArray prefix_;
    DivergenceArray divergence_;
    marker_t num_markers_;
    haplotype_t num_haplotypes_;

    // Internal helper for selection
    void select_states_at_marker(
        marker_t m,
        const allele_t* target_sequence,
        uint32_t L,
        std::vector<std::pair<marker_t, haplotype_t>>& candidates
    ) const;
};

// PBWT builder - constructs index from reference panel
class PBWTBuilder {
public:
    // Build complete PBWT index
    static std::unique_ptr<PBWTIndex> build(
        const allele_t* reference_panel,
        marker_t num_markers,
        haplotype_t num_haplotypes,
        bool parallel = true
    );
    
private:
    // Build single marker's prefix and divergence arrays
    static void build_marker(
        marker_t m,
        const allele_t* reference_panel,
        haplotype_t num_haplotypes,
        const haplotype_t* prev_prefix,
        const marker_t* prev_divergence,
        haplotype_t* curr_prefix,
        marker_t* curr_divergence
    );
    
    // Parallel version using multiple threads
    static void build_parallel(
        const allele_t* reference_panel,
        marker_t num_markers,
        haplotype_t num_haplotypes,
        PrefixArray& prefix,
        DivergenceArray& divergence
    );
};

// State selector - optimized selection algorithms
class StateSelector {
public:
    StateSelector(const PBWTIndex& index, uint32_t num_states);
    
    // Select states for single target sample
    void select_for_sample(
        const allele_t* target_haplotypes,  // [2][num_markers]
        marker_t num_markers,
        marker_t* selected_states           // [num_markers][num_states]
    ) const;
    
    // Select states for batch of samples (parallel)
    void select_for_batch(
        const allele_t* target_haplotypes,  // [num_samples][2][num_markers]
        uint32_t num_samples,
        marker_t num_markers,
        marker_t* selected_states           // [num_samples][num_markers][num_states]
    ) const;
    
    // Dynamic selection with varying L
    void select_dynamic(
        const allele_t* target_haplotypes,
        marker_t num_markers,
        const uint32_t* num_states_per_marker,  // [num_markers]
        marker_t* selected_states,               // Variable length
        uint32_t* offsets                        // [num_markers+1]
    ) const;
    
private:
    const PBWTIndex& index_;
    uint32_t num_states_;
    
    // Helper: compute divergence score for state selection
    marker_t compute_divergence_score(
        marker_t m,
        haplotype_t haplotype,
        const allele_t* target_sequence
    ) const;
};

// GPU-accelerated state selection
class GPUStateSelector {
public:
    GPUStateSelector(
        const PBWTIndex& index,
        uint32_t num_states,
        int device_id = 0
    );
    
    ~GPUStateSelector();
    
    // Transfer index to GPU
    void transfer_index_to_device();
    
    // Select states on GPU
    void select_on_device(
        const allele_t* d_target_haplotypes,
        uint32_t num_samples,
        marker_t num_markers,
        marker_t* d_selected_states,
        cudaStream_t stream = 0
    );
    
    // Host wrapper with data transfer
    void select(
        const allele_t* h_target_haplotypes,
        uint32_t num_samples,
        marker_t num_markers,
        marker_t* h_selected_states
    );
    
    size_t device_memory_usage() const;

    // Check if index has been transferred to GPU
    bool is_index_on_device() const { return index_on_device_; }

private:
    const PBWTIndex& index_;
    uint32_t num_states_;
    int device_id_;

    // Device memory for PBWT index (managed via DevicePtr wrapper)
    DevicePtr<haplotype_t> d_prefix_;
    DevicePtr<marker_t> d_divergence_;

    // Raw device pointers (for direct management when DevicePtr doesn't fit)
    haplotype_t* d_prefix_raw_ = nullptr;
    marker_t* d_divergence_raw_ = nullptr;
    bool memory_allocated_ = false;
    bool index_on_device_ = false;

    void allocate_device_memory();
    void free_device_memory();
};

// PBWT statistics and utilities
struct PBWTStats {
    double avg_divergence;
    double median_divergence;
    marker_t max_divergence;
    double compression_ratio;
    
    PBWTStats() :
        avg_divergence(0.0),
        median_divergence(0.0),
        max_divergence(0),
        compression_ratio(0.0) {}
};

// Compute statistics about PBWT structure
PBWTStats compute_pbwt_stats(const PBWTIndex& index);

// Validate PBWT correctness
bool validate_pbwt_index(
    const PBWTIndex& index,
    const allele_t* reference_panel
);

// Save/load PBWT index
void save_pbwt_index(const PBWTIndex& index, const std::string& filename);
std::unique_ptr<PBWTIndex> load_pbwt_index(const std::string& filename);

// ============================================================================
// GPU state selection functions (implemented in pbwt_selector.cu)
// ============================================================================

// Launch GPU kernel for state selection
void launch_select_states(
    const haplotype_t* d_prefix,
    const marker_t* d_divergence,
    const allele_t* d_target_haplotypes,
    uint32_t num_samples,
    uint32_t num_markers,
    uint32_t num_haplotypes,
    uint32_t num_states_L,
    haplotype_t* d_selected_states,
    cudaStream_t stream = 0
);

// GPU memory management for state selector
void gpu_state_selector_allocate(
    const PBWTIndex& index,
    int device_id,
    haplotype_t** d_prefix,
    marker_t** d_divergence
);

void gpu_state_selector_transfer(
    const PBWTIndex& index,
    haplotype_t* d_prefix,
    marker_t* d_divergence,
    cudaStream_t stream = 0
);

void gpu_state_selector_free(
    haplotype_t* d_prefix,
    marker_t* d_divergence
);

size_t gpu_state_selector_memory_usage(
    marker_t num_markers,
    haplotype_t num_haplotypes
);

} // namespace pbwt
} // namespace swiftimpute
