#pragma once

#include "core/types.hpp"
#include "imputer.hpp"
#include <string>
#include <vector>
#include <map>
#include <memory>
#include <chrono>
#include <fstream>

namespace swiftimpute {

/**
 * Checkpoint: Save and resume imputation progress
 *
 * Enables crash recovery and chromosome-by-chromosome processing with
 * automatic persistence. Designed for large-scale imputation pipelines.
 *
 * Features:
 * - Automatic checkpoint creation after each chromosome
 * - Resume from any point after crash/interruption
 * - Track per-chromosome status (pending, in_progress, completed)
 * - Store partial window progress for crash recovery
 * - Memory-efficient: only keeps minimal state in RAM
 *
 * Usage:
 *   CheckpointManager checkpoint("imputation.ckpt");
 *   checkpoint.initialize(reference, targets, config);
 *
 *   for (auto& chrom : checkpoint.pending_chromosomes()) {
 *       checkpoint.start_chromosome(chrom);
 *       // ... impute chromosome ...
 *       checkpoint.complete_chromosome(chrom, result);
 *   }
 *
 *   auto final_result = checkpoint.merge_results();
 */

enum class ChromosomeStatus {
    PENDING,        // Not yet started
    IN_PROGRESS,    // Currently processing
    COMPLETED,      // Finished successfully
    FAILED          // Failed (can retry)
};

/**
 * Progress information for a chromosome
 */
struct ChromosomeProgress {
    std::string chrom;
    ChromosomeStatus status;

    // Marker range in the full dataset
    marker_t start_marker;
    marker_t end_marker;
    marker_t num_markers;

    // Progress within chromosome
    uint32_t completed_windows;
    uint32_t total_windows;
    uint32_t completed_samples;
    uint32_t total_samples;

    // Timing
    std::chrono::system_clock::time_point start_time;
    std::chrono::system_clock::time_point end_time;
    double elapsed_seconds;

    // Result file (for completed chromosomes)
    std::string result_file;

    ChromosomeProgress() : status(ChromosomeStatus::PENDING),
        start_marker(0), end_marker(0), num_markers(0),
        completed_windows(0), total_windows(0),
        completed_samples(0), total_samples(0),
        elapsed_seconds(0.0) {}
};

/**
 * Window progress for fine-grained crash recovery
 */
struct WindowProgress {
    uint32_t window_index;
    marker_t start_marker;
    marker_t end_marker;
    uint32_t completed_samples;
    bool completed;
};

/**
 * Checkpoint file header
 */
struct CheckpointHeader {
    static constexpr uint32_t MAGIC = 0x53574654;  // "SWFT"
    static constexpr uint32_t VERSION = 1;

    uint32_t magic;
    uint32_t version;
    uint64_t created_timestamp;
    uint64_t modified_timestamp;

    // Dataset info
    uint32_t num_chromosomes;
    uint32_t num_samples;
    uint32_t total_markers;

    // Configuration hash (to detect config changes)
    uint64_t config_hash;

    CheckpointHeader() : magic(MAGIC), version(VERSION),
        created_timestamp(0), modified_timestamp(0),
        num_chromosomes(0), num_samples(0), total_markers(0),
        config_hash(0) {}

    bool is_valid() const { return magic == MAGIC && version <= VERSION; }
};

/**
 * Main checkpoint manager
 */
class CheckpointManager {
public:
    /**
     * Create or load checkpoint
     *
     * @param checkpoint_path Path to checkpoint file
     * @param output_dir Directory for per-chromosome result files
     */
    CheckpointManager(
        const std::string& checkpoint_path,
        const std::string& output_dir = ""
    );

    ~CheckpointManager();

    /**
     * Initialize checkpoint for a new imputation run
     *
     * Call this before starting imputation. If a checkpoint already
     * exists and is compatible, it will be loaded for resumption.
     *
     * @param reference Reference panel
     * @param targets Target samples
     * @param config Imputation configuration
     * @return true if resuming from existing checkpoint
     */
    bool initialize(
        const ReferencePanel& reference,
        const TargetData& targets,
        const ImputationConfig& config
    );

    /**
     * Check if resuming from existing checkpoint
     */
    bool is_resuming() const { return is_resuming_; }

    /**
     * Get list of pending chromosomes
     */
    std::vector<std::string> pending_chromosomes() const;

    /**
     * Get list of completed chromosomes
     */
    std::vector<std::string> completed_chromosomes() const;

    /**
     * Get all chromosome names in order
     */
    std::vector<std::string> all_chromosomes() const;

    /**
     * Get chromosome progress
     */
    const ChromosomeProgress* get_progress(const std::string& chrom) const;

    /**
     * Mark chromosome as started
     */
    void start_chromosome(const std::string& chrom);

    /**
     * Update window progress within chromosome
     *
     * Call this periodically for fine-grained crash recovery.
     */
    void update_window_progress(
        const std::string& chrom,
        uint32_t window_index,
        uint32_t completed_samples
    );

    /**
     * Mark chromosome as completed and save results
     *
     * @param chrom Chromosome name
     * @param result Imputation result for this chromosome
     * @param targets Target data (for VCF output)
     * @param reference Reference panel (for VCF output)
     */
    void complete_chromosome(
        const std::string& chrom,
        const ImputationResult& result,
        const TargetData& targets,
        const ReferencePanel& reference
    );

    /**
     * Mark chromosome as failed
     */
    void fail_chromosome(const std::string& chrom, const std::string& error);

    /**
     * Merge all completed chromosome results into final output
     *
     * @param output_path Path for merged VCF output
     * @param targets Full target data
     * @param reference Full reference panel
     */
    void merge_results(
        const std::string& output_path,
        const TargetData& targets,
        const ReferencePanel& reference
    );

    /**
     * Get overall progress (0.0 - 1.0)
     */
    double overall_progress() const;

    /**
     * Get elapsed time in seconds
     */
    double elapsed_time() const;

    /**
     * Get estimated time remaining in seconds
     */
    double estimated_time_remaining() const;

    /**
     * Get summary string
     */
    std::string summary() const;

    /**
     * Delete checkpoint (call after successful completion)
     */
    void cleanup();

    /**
     * Get configuration
     */
    const ImputationConfig& config() const { return config_; }

private:
    std::string checkpoint_path_;
    std::string output_dir_;

    CheckpointHeader header_;
    ImputationConfig config_;
    std::vector<std::string> sample_names_;
    std::map<std::string, ChromosomeProgress> chromosome_progress_;
    std::vector<std::string> chromosome_order_;

    bool is_resuming_;
    bool is_initialized_;
    std::chrono::system_clock::time_point session_start_;

    // File I/O
    void save_checkpoint();
    bool load_checkpoint();
    void save_chromosome_result(
        const std::string& chrom,
        const ImputationResult& result,
        const TargetData& targets,
        const ReferencePanel& reference
    );

    // Hash computation
    uint64_t compute_config_hash(const ImputationConfig& config) const;

    // Filename generation
    std::string chromosome_result_path(const std::string& chrom) const;
};

/**
 * Batch imputation runner with automatic checkpointing
 *
 * High-level interface for processing entire datasets with:
 * - Automatic chromosome-by-chromosome processing
 * - Checkpointing after each chromosome
 * - Progress reporting
 * - Memory-efficient streaming
 */
class BatchImputer {
public:
    /**
     * Create batch imputer
     *
     * @param reference_vcf Path to reference VCF
     * @param target_vcf Path to target VCF
     * @param output_vcf Path for output VCF
     * @param config Imputation configuration
     */
    BatchImputer(
        const std::string& reference_vcf,
        const std::string& target_vcf,
        const std::string& output_vcf,
        const ImputationConfig& config = ImputationConfig()
    );

    ~BatchImputer();

    /**
     * Set checkpoint path for resume support
     */
    void set_checkpoint_path(const std::string& path);

    /**
     * Set working directory for temporary files
     */
    void set_work_dir(const std::string& path);

    /**
     * Set progress callback
     */
    using ProgressCallback = std::function<void(
        const std::string& chrom,    // Current chromosome
        uint32_t chrom_index,        // 0-based chromosome index
        uint32_t total_chroms,       // Total chromosomes
        double chrom_progress,       // Progress within chromosome (0-1)
        double overall_progress      // Overall progress (0-1)
    )>;

    void set_progress_callback(ProgressCallback callback);

    /**
     * Run imputation for all chromosomes
     *
     * Automatically:
     * - Detects chromosomes from input files
     * - Processes each chromosome sequentially
     * - Checkpoints after each chromosome
     * - Resumes from checkpoint if interrupted
     * - Merges results into final output
     *
     * @return true on success
     */
    bool run();

    /**
     * Run imputation for specific chromosomes only
     */
    bool run(const std::vector<std::string>& chromosomes);

    /**
     * Get summary of results
     */
    std::string summary() const;

    /**
     * Get list of detected chromosomes
     */
    std::vector<std::string> detect_chromosomes() const;

private:
    std::string reference_vcf_;
    std::string target_vcf_;
    std::string output_vcf_;
    std::string checkpoint_path_;
    std::string work_dir_;

    ImputationConfig config_;
    ProgressCallback progress_callback_;

    std::unique_ptr<CheckpointManager> checkpoint_;

    // Per-chromosome processing
    bool process_chromosome(
        const std::string& chrom,
        uint32_t chrom_index,
        uint32_t total_chroms
    );

    // Memory-mapped binary format support
    void prepare_binary_reference(const std::string& chrom);
    void prepare_binary_targets(const std::string& chrom);
};

/**
 * Convenience function for batch imputation with checkpointing
 */
bool batch_impute(
    const std::string& reference_vcf,
    const std::string& target_vcf,
    const std::string& output_vcf,
    const std::string& checkpoint_path = "",
    const ImputationConfig& config = ImputationConfig(),
    BatchImputer::ProgressCallback progress = nullptr
);

} // namespace swiftimpute
