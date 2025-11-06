#pragma once

#include "core/types.hpp"
#include "pbwt/pbwt_index.hpp"
#include "kernels/forward_backward.cuh"
#include <string>
#include <memory>
#include <vector>
#include <functional>

namespace swiftimpute {

// Configuration for imputation
struct ImputationConfig {
    // HMM parameters
    HMMParameters hmm_params;
    
    // Computational parameters
    int device_id;                  // GPU device ID (-1 for auto-select)
    uint32_t batch_size;            // Number of samples per GPU batch
    uint32_t num_threads;           // CPU threads for I/O and preprocessing
    bool use_pinned_memory;         // Use pinned memory for faster transfers
    
    // Output options
    bool output_dosages;            // Output DS field (dosage 0-2)
    bool output_probabilities;      // Output GP field (3 probabilities)
    bool output_info_score;         // Output INFO quality score
    bool deterministic;             // Deterministic mode (argmax vs sampling)
    
    // Checkpointing
    uint32_t checkpoint_interval;   // 0 = auto-calculate
    
    ImputationConfig() :
        device_id(-1),
        batch_size(100),
        num_threads(4),
        use_pinned_memory(true),
        output_dosages(true),
        output_probabilities(true),
        output_info_score(true),
        deterministic(false),
        checkpoint_interval(0) {}
};

// Reference panel data
class ReferencePanel {
public:
    ReferencePanel() = default;
    
    // Load from VCF file
    static std::unique_ptr<ReferencePanel> load_vcf(
        const std::string& filename,
        const std::string& region = ""
    );
    
    // Load from pre-processed binary format
    static std::unique_ptr<ReferencePanel> load_binary(
        const std::string& filename
    );
    
    // Save to binary format for faster loading
    void save_binary(const std::string& filename) const;
    
    // Accessors
    marker_t num_markers() const { return markers_.size(); }
    haplotype_t num_haplotypes() const { return num_haplotypes_; }
    sample_t num_samples() const { return num_haplotypes_ / 2; }
    
    const std::vector<Marker>& markers() const { return markers_; }
    const std::vector<Sample>& samples() const { return samples_; }
    
    // Get haplotype data
    const allele_t* haplotypes() const { return haplotypes_.get(); }
    allele_t get_allele(marker_t m, haplotype_t h) const;
    
    // Memory usage
    size_t memory_usage() const;
    
private:
    std::vector<Marker> markers_;
    std::vector<Sample> samples_;
    haplotype_t num_haplotypes_;
    std::unique_ptr<allele_t[]> haplotypes_;  // [num_markers][num_haplotypes]
    
    ReferencePanel(
        std::vector<Marker> markers,
        std::vector<Sample> samples,
        haplotype_t num_haplotypes,
        std::unique_ptr<allele_t[]> haplotypes
    );
};

// Target data for imputation
class TargetData {
public:
    TargetData() = default;
    
    // Load from VCF file
    static std::unique_ptr<TargetData> load_vcf(
        const std::string& filename,
        const std::string& region = ""
    );
    
    // Accessors
    marker_t num_markers() const { return markers_.size(); }
    sample_t num_samples() const { return samples_.size(); }
    
    const std::vector<Marker>& markers() const { return markers_; }
    const std::vector<Sample>& samples() const { return samples_; }
    
    // Get genotype likelihoods
    const GenotypeLikelihoods* genotype_likelihoods() const {
        return genotype_liks_.get();
    }
    
    GenotypeLikelihoods get_likelihood(sample_t s, marker_t m) const;
    
    // Memory usage
    size_t memory_usage() const;
    
private:
    std::vector<Marker> markers_;
    std::vector<Sample> samples_;
    std::unique_ptr<GenotypeLikelihoods[]> genotype_liks_;  // [num_samples][num_markers]
    
    TargetData(
        std::vector<Marker> markers,
        std::vector<Sample> samples,
        std::unique_ptr<GenotypeLikelihoods[]> genotype_liks
    );
};

// Imputation result
class ImputationResult {
public:
    ImputationResult(
        uint32_t num_samples,
        uint32_t num_markers
    );
    
    // Write to VCF file
    void write_vcf(
        const std::string& filename,
        const TargetData& targets,
        const ReferencePanel& reference,
        const ImputationConfig& config
    ) const;
    
    // Accessors
    uint32_t num_samples() const { return num_samples_; }
    uint32_t num_markers() const { return num_markers_; }
    
    // Get phased haplotypes
    const allele_t* haplotype0(sample_t s) const;
    const allele_t* haplotype1(sample_t s) const;
    
    // Get dosages (0-2)
    prob_t get_dosage(sample_t s, marker_t m) const;
    
    // Get genotype probabilities (AA, AB, BB)
    void get_probabilities(sample_t s, marker_t m, prob_t probs[3]) const;
    
    // Get INFO score
    prob_t get_info_score(marker_t m) const;
    
    // Internal: set results (called by imputer)
    void set_haplotype(sample_t s, uint32_t phase, const allele_t* haplotype);
    void set_posterior(sample_t s, marker_t m, const prob_t* posterior);
    void compute_info_scores();
    
    // Memory usage
    size_t memory_usage() const;
    
private:
    uint32_t num_samples_;
    uint32_t num_markers_;
    
    std::unique_ptr<allele_t[]> haplotypes_;      // [num_samples][2][num_markers]
    std::unique_ptr<prob_t[]> posteriors_;        // [num_samples][num_markers][2]
    std::unique_ptr<prob_t[]> info_scores_;       // [num_markers]
};

// Main imputation engine
class Imputer {
public:
    Imputer(
        const ReferencePanel& reference,
        const ImputationConfig& config = ImputationConfig()
    );
    
    ~Imputer();
    
    // Build PBWT index (can be done once and reused)
    void build_index();
    
    // Check if index is built
    bool has_index() const { return pbwt_index_ != nullptr; }
    
    // Impute target samples
    std::unique_ptr<ImputationResult> impute(const TargetData& targets);
    
    // Impute in batches with progress callback
    std::unique_ptr<ImputationResult> impute_with_progress(
        const TargetData& targets,
        std::function<void(uint32_t, uint32_t)> progress_callback
    );
    
    // Get configuration
    const ImputationConfig& config() const { return config_; }
    
    // Get device memory usage
    size_t device_memory_usage() const;
    
    // Get estimated memory requirement
    static size_t estimate_memory_requirement(
        const ReferencePanel& reference,
        uint32_t num_target_samples,
        const ImputationConfig& config
    );
    
private:
    const ReferencePanel& reference_;
    ImputationConfig config_;
    
    std::unique_ptr<pbwt::PBWTIndex> pbwt_index_;
    std::unique_ptr<pbwt::GPUStateSelector> state_selector_;
    std::unique_ptr<kernels::ForwardBackward> forward_backward_;
    
    int device_id_;
    
    void initialize_gpu();
    void validate_targets(const TargetData& targets);
    
    void impute_batch(
        const TargetData& targets,
        uint32_t start_sample,
        uint32_t end_sample,
        ImputationResult& result
    );
};

// Convenience function for simple use case
std::unique_ptr<ImputationResult> impute(
    const std::string& reference_vcf,
    const std::string& target_vcf,
    const std::string& output_vcf,
    const ImputationConfig& config = ImputationConfig()
);

// Multi-GPU imputation
class MultiGPUImputer {
public:
    MultiGPUImputer(
        const ReferencePanel& reference,
        const std::vector<int>& device_ids,
        const ImputationConfig& config = ImputationConfig()
    );
    
    ~MultiGPUImputer();
    
    // Build indices on all GPUs
    void build_indices();
    
    // Impute with automatic load balancing
    std::unique_ptr<ImputationResult> impute(const TargetData& targets);
    
    // Get number of GPUs
    uint32_t num_gpus() const { return static_cast<uint32_t>(imputers_.size()); }
    
private:
    const ReferencePanel& reference_;
    ImputationConfig config_;
    std::vector<int> device_ids_;
    std::vector<std::unique_ptr<Imputer>> imputers_;
    
    void partition_samples(
        uint32_t num_samples,
        std::vector<uint32_t>& start_indices,
        std::vector<uint32_t>& end_indices
    ) const;
};

} // namespace swiftimpute
