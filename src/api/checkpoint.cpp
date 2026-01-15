#include "checkpoint.hpp"
#include "io/vcf_reader.hpp"
#include <fstream>
#include <filesystem>
#include <sstream>
#include <iomanip>
#include <algorithm>
#include <cstring>

namespace swiftimpute {
namespace fs = std::filesystem;

// ============================================================================
// CheckpointManager Implementation
// ============================================================================

CheckpointManager::CheckpointManager(
    const std::string& checkpoint_path,
    const std::string& output_dir
) : checkpoint_path_(checkpoint_path),
    output_dir_(output_dir.empty() ? fs::path(checkpoint_path).parent_path().string() : output_dir),
    is_resuming_(false),
    is_initialized_(false) {

    // Ensure output directory exists
    if (!output_dir_.empty()) {
        fs::create_directories(output_dir_);
    }
}

CheckpointManager::~CheckpointManager() {
    // Auto-save on destruction if initialized
    if (is_initialized_) {
        try {
            save_checkpoint();
        } catch (...) {
            // Ignore errors in destructor
        }
    }
}

bool CheckpointManager::initialize(
    const ReferencePanel& reference,
    const TargetData& targets,
    const ImputationConfig& config
) {
    config_ = config;
    session_start_ = std::chrono::system_clock::now();

    // Store sample names for result merging
    sample_names_.clear();
    for (const auto& sample : targets.samples()) {
        sample_names_.push_back(sample.id);
    }

    // Detect chromosomes from reference
    chromosome_order_ = reference.get_chromosomes();

    // Try to load existing checkpoint
    if (fs::exists(checkpoint_path_)) {
        if (load_checkpoint()) {
            // Verify compatibility
            uint64_t current_hash = compute_config_hash(config);
            if (header_.config_hash == current_hash &&
                header_.num_samples == targets.num_samples() &&
                header_.num_chromosomes == chromosome_order_.size()) {

                is_resuming_ = true;
                is_initialized_ = true;

                LOG_INFO("Resuming from checkpoint: " + checkpoint_path_);
                LOG_INFO("Progress: " + std::to_string(completed_chromosomes().size()) +
                        "/" + std::to_string(chromosome_order_.size()) + " chromosomes completed");
                return true;
            } else {
                LOG_WARNING("Checkpoint incompatible with current run, starting fresh");
            }
        }
    }

    // Initialize fresh checkpoint
    header_ = CheckpointHeader();
    header_.created_timestamp = std::chrono::duration_cast<std::chrono::seconds>(
        session_start_.time_since_epoch()).count();
    header_.num_chromosomes = chromosome_order_.size();
    header_.num_samples = targets.num_samples();
    header_.total_markers = reference.num_markers();
    header_.config_hash = compute_config_hash(config);

    // Initialize chromosome progress
    chromosome_progress_.clear();
    marker_t offset = 0;

    for (const auto& chrom : chromosome_order_) {
        ChromosomeProgress progress;
        progress.chrom = chrom;
        progress.status = ChromosomeStatus::PENDING;

        // Count markers for this chromosome
        marker_t count = 0;
        for (const auto& marker : reference.markers()) {
            if (marker.chrom == chrom) {
                count++;
            }
        }

        progress.start_marker = offset;
        progress.end_marker = offset + count;
        progress.num_markers = count;
        progress.total_samples = targets.num_samples();

        chromosome_progress_[chrom] = progress;
        offset += count;
    }

    is_resuming_ = false;
    is_initialized_ = true;

    save_checkpoint();
    LOG_INFO("Created new checkpoint: " + checkpoint_path_);
    LOG_INFO("Processing " + std::to_string(chromosome_order_.size()) + " chromosomes");

    return false;
}

std::vector<std::string> CheckpointManager::pending_chromosomes() const {
    std::vector<std::string> result;
    for (const auto& chrom : chromosome_order_) {
        auto it = chromosome_progress_.find(chrom);
        if (it != chromosome_progress_.end() &&
            (it->second.status == ChromosomeStatus::PENDING ||
             it->second.status == ChromosomeStatus::FAILED)) {
            result.push_back(chrom);
        }
    }
    return result;
}

std::vector<std::string> CheckpointManager::completed_chromosomes() const {
    std::vector<std::string> result;
    for (const auto& chrom : chromosome_order_) {
        auto it = chromosome_progress_.find(chrom);
        if (it != chromosome_progress_.end() &&
            it->second.status == ChromosomeStatus::COMPLETED) {
            result.push_back(chrom);
        }
    }
    return result;
}

std::vector<std::string> CheckpointManager::all_chromosomes() const {
    return chromosome_order_;
}

const ChromosomeProgress* CheckpointManager::get_progress(const std::string& chrom) const {
    auto it = chromosome_progress_.find(chrom);
    if (it != chromosome_progress_.end()) {
        return &it->second;
    }
    return nullptr;
}

void CheckpointManager::start_chromosome(const std::string& chrom) {
    auto it = chromosome_progress_.find(chrom);
    if (it == chromosome_progress_.end()) {
        throw std::runtime_error("Unknown chromosome: " + chrom);
    }

    it->second.status = ChromosomeStatus::IN_PROGRESS;
    it->second.start_time = std::chrono::system_clock::now();
    it->second.completed_windows = 0;
    it->second.completed_samples = 0;

    // Calculate total windows
    uint32_t window_size = config_.window_size > 0 ? config_.window_size : it->second.num_markers;
    it->second.total_windows = (it->second.num_markers + window_size - 1) / window_size;

    save_checkpoint();
    LOG_INFO("Started processing chromosome " + chrom +
            " (" + std::to_string(it->second.num_markers) + " markers)");
}

void CheckpointManager::update_window_progress(
    const std::string& chrom,
    uint32_t window_index,
    uint32_t completed_samples
) {
    auto it = chromosome_progress_.find(chrom);
    if (it != chromosome_progress_.end()) {
        it->second.completed_windows = window_index + 1;
        it->second.completed_samples = completed_samples;

        // Update checkpoint periodically (not every call to avoid I/O overhead)
        if (window_index % 10 == 0 || completed_samples == it->second.total_samples) {
            header_.modified_timestamp = std::chrono::duration_cast<std::chrono::seconds>(
                std::chrono::system_clock::now().time_since_epoch()).count();
            save_checkpoint();
        }
    }
}

void CheckpointManager::complete_chromosome(
    const std::string& chrom,
    const ImputationResult& result,
    const TargetData& targets,
    const ReferencePanel& reference
) {
    auto it = chromosome_progress_.find(chrom);
    if (it == chromosome_progress_.end()) {
        throw std::runtime_error("Unknown chromosome: " + chrom);
    }

    // Save chromosome result to file
    save_chromosome_result(chrom, result, targets, reference);

    // Update progress
    it->second.status = ChromosomeStatus::COMPLETED;
    it->second.end_time = std::chrono::system_clock::now();
    it->second.elapsed_seconds = std::chrono::duration<double>(
        it->second.end_time - it->second.start_time).count();
    it->second.completed_windows = it->second.total_windows;
    it->second.completed_samples = it->second.total_samples;
    it->second.result_file = chromosome_result_path(chrom);

    save_checkpoint();
    LOG_INFO("Completed chromosome " + chrom +
            " in " + std::to_string(static_cast<int>(it->second.elapsed_seconds)) + "s");
}

void CheckpointManager::fail_chromosome(const std::string& chrom, const std::string& error) {
    auto it = chromosome_progress_.find(chrom);
    if (it != chromosome_progress_.end()) {
        it->second.status = ChromosomeStatus::FAILED;
        it->second.end_time = std::chrono::system_clock::now();
        it->second.elapsed_seconds = std::chrono::duration<double>(
            it->second.end_time - it->second.start_time).count();
        save_checkpoint();
        LOG_ERROR("Failed chromosome " + chrom + ": " + error);
    }
}

void CheckpointManager::merge_results(
    const std::string& output_path,
    const TargetData& targets,
    const ReferencePanel& reference
) {
    LOG_INFO("Merging chromosome results to: " + output_path);

    // Collect all completed chromosome result files
    std::vector<std::string> result_files;
    for (const auto& chrom : chromosome_order_) {
        auto it = chromosome_progress_.find(chrom);
        if (it != chromosome_progress_.end() &&
            it->second.status == ChromosomeStatus::COMPLETED &&
            !it->second.result_file.empty()) {
            result_files.push_back(it->second.result_file);
        }
    }

    if (result_files.empty()) {
        throw std::runtime_error("No completed chromosome results to merge");
    }

    // Open output file
    std::ofstream output(output_path);
    if (!output) {
        throw std::runtime_error("Cannot open output file: " + output_path);
    }

    bool header_written = false;

    // Concatenate VCF files
    for (const auto& result_file : result_files) {
        std::ifstream input(result_file);
        if (!input) {
            LOG_WARNING("Cannot open result file: " + result_file);
            continue;
        }

        std::string line;
        while (std::getline(input, line)) {
            // Skip header lines for subsequent files
            if (line.empty() || line[0] == '#') {
                if (!header_written) {
                    output << line << "\n";
                }
                continue;
            }
            header_written = true;
            output << line << "\n";
        }
    }

    LOG_INFO("Merged " + std::to_string(result_files.size()) + " chromosome results");
}

double CheckpointManager::overall_progress() const {
    if (chromosome_order_.empty()) return 0.0;

    double total_markers = 0;
    double completed_markers = 0;

    for (const auto& [chrom, progress] : chromosome_progress_) {
        total_markers += progress.num_markers;

        if (progress.status == ChromosomeStatus::COMPLETED) {
            completed_markers += progress.num_markers;
        } else if (progress.status == ChromosomeStatus::IN_PROGRESS) {
            // Estimate based on windows completed
            if (progress.total_windows > 0) {
                double chrom_progress = static_cast<double>(progress.completed_windows) /
                                       progress.total_windows;
                completed_markers += progress.num_markers * chrom_progress;
            }
        }
    }

    return total_markers > 0 ? completed_markers / total_markers : 0.0;
}

double CheckpointManager::elapsed_time() const {
    auto now = std::chrono::system_clock::now();
    return std::chrono::duration<double>(now - session_start_).count();
}

double CheckpointManager::estimated_time_remaining() const {
    double progress = overall_progress();
    if (progress <= 0.0) return -1.0;

    double elapsed = elapsed_time();
    return (elapsed / progress) * (1.0 - progress);
}

std::string CheckpointManager::summary() const {
    std::ostringstream ss;

    auto completed = completed_chromosomes();
    auto pending = pending_chromosomes();

    ss << "Checkpoint Summary:\n";
    ss << "  Total chromosomes: " << chromosome_order_.size() << "\n";
    ss << "  Completed: " << completed.size() << "\n";
    ss << "  Pending: " << pending.size() << "\n";
    ss << "  Progress: " << std::fixed << std::setprecision(1)
       << (overall_progress() * 100) << "%\n";

    double elapsed = elapsed_time();
    if (elapsed > 0) {
        ss << "  Elapsed time: " << static_cast<int>(elapsed / 60) << "m "
           << static_cast<int>(static_cast<int>(elapsed) % 60) << "s\n";

        double remaining = estimated_time_remaining();
        if (remaining > 0) {
            ss << "  Estimated remaining: " << static_cast<int>(remaining / 60) << "m "
               << static_cast<int>(static_cast<int>(remaining) % 60) << "s\n";
        }
    }

    return ss.str();
}

void CheckpointManager::cleanup() {
    // Delete checkpoint file
    if (fs::exists(checkpoint_path_)) {
        fs::remove(checkpoint_path_);
        LOG_INFO("Deleted checkpoint file: " + checkpoint_path_);
    }

    // Delete per-chromosome result files
    for (const auto& [chrom, progress] : chromosome_progress_) {
        if (!progress.result_file.empty() && fs::exists(progress.result_file)) {
            fs::remove(progress.result_file);
        }
    }
}

void CheckpointManager::save_checkpoint() {
    std::ofstream file(checkpoint_path_, std::ios::binary);
    if (!file) {
        throw std::runtime_error("Cannot write checkpoint file: " + checkpoint_path_);
    }

    // Update timestamp
    header_.modified_timestamp = std::chrono::duration_cast<std::chrono::seconds>(
        std::chrono::system_clock::now().time_since_epoch()).count();

    // Write header
    file.write(reinterpret_cast<const char*>(&header_), sizeof(CheckpointHeader));

    // Write chromosome order
    uint32_t num_chroms = chromosome_order_.size();
    file.write(reinterpret_cast<const char*>(&num_chroms), sizeof(num_chroms));

    for (const auto& chrom : chromosome_order_) {
        uint32_t len = chrom.size();
        file.write(reinterpret_cast<const char*>(&len), sizeof(len));
        file.write(chrom.data(), len);
    }

    // Write chromosome progress
    for (const auto& chrom : chromosome_order_) {
        const auto& progress = chromosome_progress_.at(chrom);

        uint32_t status = static_cast<uint32_t>(progress.status);
        file.write(reinterpret_cast<const char*>(&status), sizeof(status));
        file.write(reinterpret_cast<const char*>(&progress.start_marker), sizeof(progress.start_marker));
        file.write(reinterpret_cast<const char*>(&progress.end_marker), sizeof(progress.end_marker));
        file.write(reinterpret_cast<const char*>(&progress.num_markers), sizeof(progress.num_markers));
        file.write(reinterpret_cast<const char*>(&progress.completed_windows), sizeof(progress.completed_windows));
        file.write(reinterpret_cast<const char*>(&progress.total_windows), sizeof(progress.total_windows));
        file.write(reinterpret_cast<const char*>(&progress.completed_samples), sizeof(progress.completed_samples));
        file.write(reinterpret_cast<const char*>(&progress.total_samples), sizeof(progress.total_samples));
        file.write(reinterpret_cast<const char*>(&progress.elapsed_seconds), sizeof(progress.elapsed_seconds));

        // Write result file path
        uint32_t path_len = progress.result_file.size();
        file.write(reinterpret_cast<const char*>(&path_len), sizeof(path_len));
        if (path_len > 0) {
            file.write(progress.result_file.data(), path_len);
        }
    }

    // Write sample names
    uint32_t num_samples = sample_names_.size();
    file.write(reinterpret_cast<const char*>(&num_samples), sizeof(num_samples));

    for (const auto& name : sample_names_) {
        uint32_t len = name.size();
        file.write(reinterpret_cast<const char*>(&len), sizeof(len));
        file.write(name.data(), len);
    }

    file.close();
}

bool CheckpointManager::load_checkpoint() {
    std::ifstream file(checkpoint_path_, std::ios::binary);
    if (!file) {
        return false;
    }

    // Read header
    file.read(reinterpret_cast<char*>(&header_), sizeof(CheckpointHeader));
    if (!header_.is_valid()) {
        LOG_WARNING("Invalid checkpoint file format");
        return false;
    }

    // Read chromosome order
    uint32_t num_chroms;
    file.read(reinterpret_cast<char*>(&num_chroms), sizeof(num_chroms));

    chromosome_order_.clear();
    for (uint32_t i = 0; i < num_chroms; ++i) {
        uint32_t len;
        file.read(reinterpret_cast<char*>(&len), sizeof(len));
        std::string chrom(len, '\0');
        file.read(&chrom[0], len);
        chromosome_order_.push_back(chrom);
    }

    // Read chromosome progress
    chromosome_progress_.clear();
    for (const auto& chrom : chromosome_order_) {
        ChromosomeProgress progress;
        progress.chrom = chrom;

        uint32_t status;
        file.read(reinterpret_cast<char*>(&status), sizeof(status));
        progress.status = static_cast<ChromosomeStatus>(status);

        file.read(reinterpret_cast<char*>(&progress.start_marker), sizeof(progress.start_marker));
        file.read(reinterpret_cast<char*>(&progress.end_marker), sizeof(progress.end_marker));
        file.read(reinterpret_cast<char*>(&progress.num_markers), sizeof(progress.num_markers));
        file.read(reinterpret_cast<char*>(&progress.completed_windows), sizeof(progress.completed_windows));
        file.read(reinterpret_cast<char*>(&progress.total_windows), sizeof(progress.total_windows));
        file.read(reinterpret_cast<char*>(&progress.completed_samples), sizeof(progress.completed_samples));
        file.read(reinterpret_cast<char*>(&progress.total_samples), sizeof(progress.total_samples));
        file.read(reinterpret_cast<char*>(&progress.elapsed_seconds), sizeof(progress.elapsed_seconds));

        // Read result file path
        uint32_t path_len;
        file.read(reinterpret_cast<char*>(&path_len), sizeof(path_len));
        if (path_len > 0) {
            progress.result_file.resize(path_len);
            file.read(&progress.result_file[0], path_len);
        }

        chromosome_progress_[chrom] = progress;
    }

    // Read sample names
    uint32_t num_samples;
    file.read(reinterpret_cast<char*>(&num_samples), sizeof(num_samples));

    sample_names_.clear();
    for (uint32_t i = 0; i < num_samples; ++i) {
        uint32_t len;
        file.read(reinterpret_cast<char*>(&len), sizeof(len));
        std::string name(len, '\0');
        file.read(&name[0], len);
        sample_names_.push_back(name);
    }

    return true;
}

void CheckpointManager::save_chromosome_result(
    const std::string& chrom,
    const ImputationResult& result,
    const TargetData& targets,
    const ReferencePanel& reference
) {
    std::string result_path = chromosome_result_path(chrom);

    // Filter reference and targets to this chromosome
    auto chrom_reference = reference.filter_chromosome(chrom);
    auto chrom_targets = targets.filter_chromosome(chrom);

    if (chrom_reference && chrom_targets) {
        result.write_vcf(result_path, *chrom_targets, *chrom_reference, config_);
    }
}

uint64_t CheckpointManager::compute_config_hash(const ImputationConfig& config) const {
    // Simple hash of config parameters
    uint64_t hash = 0;

    hash ^= static_cast<uint64_t>(config.hmm_params.num_states) << 0;
    hash ^= static_cast<uint64_t>(config.batch_size) << 8;
    hash ^= static_cast<uint64_t>(config.window_size) << 16;
    hash ^= static_cast<uint64_t>(config.window_overlap) << 32;
    hash ^= (config.rebuild_pbwt_per_window ? 1ULL : 0ULL) << 48;
    hash ^= (config.deterministic ? 1ULL : 0ULL) << 49;

    // Include Ne in hash (as bits of float)
    uint64_t ne_bits;
    memcpy(&ne_bits, &config.hmm_params.ne, sizeof(double));
    hash ^= ne_bits;

    return hash;
}

std::string CheckpointManager::chromosome_result_path(const std::string& chrom) const {
    fs::path base(output_dir_);
    return (base / ("chrom_" + chrom + ".vcf.gz")).string();
}

// ============================================================================
// BatchImputer Implementation
// ============================================================================

BatchImputer::BatchImputer(
    const std::string& reference_vcf,
    const std::string& target_vcf,
    const std::string& output_vcf,
    const ImputationConfig& config
) : reference_vcf_(reference_vcf),
    target_vcf_(target_vcf),
    output_vcf_(output_vcf),
    config_(config) {

    // Default work directory
    work_dir_ = fs::path(output_vcf).parent_path().string();
    if (work_dir_.empty()) {
        work_dir_ = ".";
    }

    // Default checkpoint path
    checkpoint_path_ = work_dir_ + "/imputation.checkpoint";
}

BatchImputer::~BatchImputer() = default;

void BatchImputer::set_checkpoint_path(const std::string& path) {
    checkpoint_path_ = path;
}

void BatchImputer::set_work_dir(const std::string& path) {
    work_dir_ = path;
    fs::create_directories(work_dir_);
}

void BatchImputer::set_progress_callback(ProgressCallback callback) {
    progress_callback_ = callback;
}

std::vector<std::string> BatchImputer::detect_chromosomes() const {
    // Read VCF header to get contig list
    io::VCFReader reader(reference_vcf_);
    auto header = reader.read_header();

    std::vector<std::string> chromosomes;

    // If contigs are listed in header, use those
    // Otherwise, scan the file for unique chromosome names

    // For now, scan the file
    std::set<std::string> seen;
    io::VCFReader::Variant variant;

    while (reader.read_variant(variant)) {
        if (seen.find(variant.chrom) == seen.end()) {
            chromosomes.push_back(variant.chrom);
            seen.insert(variant.chrom);
        }
    }

    return chromosomes;
}

bool BatchImputer::run() {
    return run(detect_chromosomes());
}

bool BatchImputer::run(const std::vector<std::string>& chromosomes) {
    LOG_INFO("Starting batch imputation for " + std::to_string(chromosomes.size()) + " chromosomes");

    // Create work directory
    fs::create_directories(work_dir_);

    // Load reference and target headers for initialization
    LOG_INFO("Loading reference panel header...");
    auto reference = ReferencePanel::load_vcf(reference_vcf_);

    LOG_INFO("Loading target data header...");
    auto targets = TargetData::load_vcf(target_vcf_);

    // Initialize checkpoint manager
    checkpoint_ = std::make_unique<CheckpointManager>(checkpoint_path_, work_dir_);
    bool resuming = checkpoint_->initialize(*reference, *targets, config_);

    if (resuming) {
        LOG_INFO(checkpoint_->summary());
    }

    // Get list of chromosomes to process
    auto pending = checkpoint_->pending_chromosomes();
    uint32_t total_chroms = chromosomes.size();

    LOG_INFO("Processing " + std::to_string(pending.size()) + " pending chromosomes");

    // Process each chromosome
    for (uint32_t i = 0; i < pending.size(); ++i) {
        const std::string& chrom = pending[i];

        // Find chromosome index in original list
        uint32_t chrom_index = 0;
        for (uint32_t j = 0; j < chromosomes.size(); ++j) {
            if (chromosomes[j] == chrom) {
                chrom_index = j;
                break;
            }
        }

        if (!process_chromosome(chrom, chrom_index, total_chroms)) {
            LOG_ERROR("Failed to process chromosome " + chrom);
            // Continue with other chromosomes
        }
    }

    // Merge results
    if (!checkpoint_->completed_chromosomes().empty()) {
        checkpoint_->merge_results(output_vcf_, *targets, *reference);
    }

    LOG_INFO("Batch imputation complete");
    LOG_INFO(checkpoint_->summary());

    // Cleanup checkpoint if all chromosomes completed
    if (checkpoint_->pending_chromosomes().empty()) {
        checkpoint_->cleanup();
    }

    return checkpoint_->pending_chromosomes().empty();
}

bool BatchImputer::process_chromosome(
    const std::string& chrom,
    uint32_t chrom_index,
    uint32_t total_chroms
) {
    LOG_INFO("Processing chromosome " + chrom +
            " (" + std::to_string(chrom_index + 1) + "/" + std::to_string(total_chroms) + ")");

    try {
        // Mark chromosome as started
        checkpoint_->start_chromosome(chrom);

        // Load chromosome-specific data
        LOG_INFO("Loading reference data for chromosome " + chrom + "...");
        auto chrom_reference = ReferencePanel::load_vcf(reference_vcf_, chrom);

        LOG_INFO("Loading target data for chromosome " + chrom + "...");
        auto chrom_targets = TargetData::load_vcf(target_vcf_, chrom);

        if (!chrom_reference || chrom_reference->num_markers() == 0) {
            LOG_WARNING("No reference data for chromosome " + chrom + ", skipping");
            checkpoint_->fail_chromosome(chrom, "No reference data");
            return false;
        }

        if (!chrom_targets || chrom_targets->num_markers() == 0) {
            LOG_WARNING("No target data for chromosome " + chrom + ", skipping");
            checkpoint_->fail_chromosome(chrom, "No target data");
            return false;
        }

        LOG_INFO("Chromosome " + chrom + ": " +
                std::to_string(chrom_reference->num_markers()) + " markers, " +
                std::to_string(chrom_targets->num_samples()) + " samples");

        // Create imputer for this chromosome
        Imputer imputer(*chrom_reference, config_);

        // Build index
        LOG_INFO("Building PBWT index for chromosome " + chrom + "...");
        imputer.build_index();

        // Progress callback wrapper
        auto progress_wrapper = [&](uint32_t completed, uint32_t total) {
            double chrom_progress = static_cast<double>(completed) / total;
            double overall = checkpoint_->overall_progress();

            checkpoint_->update_window_progress(chrom, completed / config_.batch_size, completed);

            if (progress_callback_) {
                progress_callback_(chrom, chrom_index, total_chroms, chrom_progress, overall);
            }
        };

        // Run imputation
        LOG_INFO("Running imputation for chromosome " + chrom + "...");
        auto result = imputer.impute_with_progress(*chrom_targets, progress_wrapper);

        // Complete chromosome
        checkpoint_->complete_chromosome(chrom, *result, *chrom_targets, *chrom_reference);

        return true;

    } catch (const std::exception& e) {
        LOG_ERROR("Error processing chromosome " + chrom + ": " + e.what());
        checkpoint_->fail_chromosome(chrom, e.what());
        return false;
    }
}

std::string BatchImputer::summary() const {
    if (checkpoint_) {
        return checkpoint_->summary();
    }
    return "No checkpoint data available";
}

// ============================================================================
// Convenience Function
// ============================================================================

bool batch_impute(
    const std::string& reference_vcf,
    const std::string& target_vcf,
    const std::string& output_vcf,
    const std::string& checkpoint_path,
    const ImputationConfig& config,
    BatchImputer::ProgressCallback progress
) {
    BatchImputer imputer(reference_vcf, target_vcf, output_vcf, config);

    if (!checkpoint_path.empty()) {
        imputer.set_checkpoint_path(checkpoint_path);
    }

    if (progress) {
        imputer.set_progress_callback(progress);
    }

    return imputer.run();
}

} // namespace swiftimpute
