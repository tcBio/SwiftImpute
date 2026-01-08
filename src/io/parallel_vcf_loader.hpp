#pragma once

#include "../core/types.hpp"
#include "mmap_reader.hpp"
#include <string>
#include <vector>
#include <memory>
#include <thread>
#include <atomic>
#include <mutex>
#include <condition_variable>
#include <queue>
#include <functional>

namespace swiftimpute {
namespace io {

/**
 * @brief Configuration for parallel VCF loading
 */
struct ParallelLoadConfig {
    uint32_t num_threads;           // Number of parsing threads
    size_t chunk_size_bytes;        // Bytes per chunk (default: 64MB)
    size_t max_queue_depth;         // Max chunks in flight
    bool use_mmap;                  // Use memory-mapped I/O
    bool prefetch_next_chunk;       // Prefetch next chunk while processing current
    bool parse_genotypes;           // Parse GT field
    bool parse_dosages;             // Parse DS field
    bool parse_probabilities;       // Parse GP field

    ParallelLoadConfig() :
        num_threads(std::thread::hardware_concurrency()),
        chunk_size_bytes(64 * 1024 * 1024),  // 64 MB
        max_queue_depth(4),
        use_mmap(true),
        prefetch_next_chunk(true),
        parse_genotypes(true),
        parse_dosages(false),
        parse_probabilities(false) {}

    // Presets for different scenarios
    static ParallelLoadConfig nvme_optimized() {
        ParallelLoadConfig config;
        config.chunk_size_bytes = 128 * 1024 * 1024;  // 128 MB - NVMe can handle large reads
        config.max_queue_depth = 8;
        config.use_mmap = true;
        config.prefetch_next_chunk = true;
        return config;
    }

    static ParallelLoadConfig hdd_optimized() {
        ParallelLoadConfig config;
        config.chunk_size_bytes = 32 * 1024 * 1024;   // 32 MB - smaller for HDD
        config.max_queue_depth = 2;
        config.use_mmap = false;  // Traditional I/O better for HDD
        config.prefetch_next_chunk = true;
        return config;
    }

    static ParallelLoadConfig low_memory() {
        ParallelLoadConfig config;
        config.chunk_size_bytes = 16 * 1024 * 1024;   // 16 MB
        config.max_queue_depth = 2;
        config.num_threads = 2;
        config.use_mmap = false;
        return config;
    }
};

/**
 * @brief Progress callback for loading
 */
using LoadProgressCallback = std::function<void(size_t bytes_loaded, size_t total_bytes)>;

/**
 * @brief Statistics from parallel loading
 */
struct LoadStats {
    size_t total_bytes;
    size_t bytes_read;
    size_t num_variants;
    size_t num_samples;
    double read_time_ms;
    double parse_time_ms;
    double total_time_ms;
    double throughput_mbps;
    bool used_mmap;
    uint32_t threads_used;
};

/**
 * @brief Parsed variant data (thread-local buffer)
 */
struct ParsedVariant {
    std::string chrom;
    uint64_t pos;
    std::string id;
    std::string ref;
    std::vector<std::string> alt;
    std::vector<allele_t> genotypes;      // [num_samples * 2] for diploid
    std::vector<float> dosages;           // [num_samples]
    std::vector<float> probabilities;     // [num_samples * 3]
    bool valid;

    void clear() {
        chrom.clear();
        pos = 0;
        id.clear();
        ref.clear();
        alt.clear();
        genotypes.clear();
        dosages.clear();
        probabilities.clear();
        valid = false;
    }
};

/**
 * @brief Parallel VCF loader for high-speed loading
 *
 * Uses multiple threads to parse VCF data in parallel:
 * - I/O thread reads chunks from file (optionally with mmap)
 * - Worker threads parse VCF lines in parallel
 * - Results are merged in order
 *
 * For compressed files, uses separate decompression threads
 */
class ParallelVCFLoader {
public:
    explicit ParallelVCFLoader(const ParallelLoadConfig& config = ParallelLoadConfig());
    ~ParallelVCFLoader();

    /**
     * @brief Load reference panel from VCF file
     *
     * @param filename VCF file path
     * @param progress Progress callback
     * @return ReferencePanel with loaded data
     */
    std::unique_ptr<ReferencePanel> load_reference(
        const std::string& filename,
        LoadProgressCallback progress = nullptr
    );

    /**
     * @brief Load target data from VCF file
     */
    std::unique_ptr<TargetData> load_targets(
        const std::string& filename,
        LoadProgressCallback progress = nullptr
    );

    /**
     * @brief Load markers only (for overlap analysis)
     */
    std::vector<Marker> load_markers_only(
        const std::string& filename,
        LoadProgressCallback progress = nullptr
    );

    /**
     * @brief Get stats from last load operation
     */
    const LoadStats& get_last_stats() const { return last_stats_; }

    /**
     * @brief Estimate load time for a file
     */
    static double estimate_load_time_ms(
        const std::string& filename,
        const ParallelLoadConfig& config
    );

private:
    ParallelLoadConfig config_;
    LoadStats last_stats_;

    // Thread pool
    std::vector<std::thread> workers_;
    std::atomic<bool> stop_workers_;
    std::mutex queue_mutex_;
    std::condition_variable queue_cv_;

    // Work queue
    struct ChunkWork {
        const char* data;
        size_t size;
        size_t chunk_id;
        size_t line_offset;  // First line number in this chunk
    };
    std::queue<ChunkWork> work_queue_;

    // Results
    struct ChunkResult {
        size_t chunk_id;
        std::vector<ParsedVariant> variants;
    };
    std::mutex result_mutex_;
    std::vector<ChunkResult> results_;

    // Internal methods
    void start_workers(size_t num_samples);
    void stop_workers();
    void worker_thread(size_t num_samples);

    void parse_chunk(const char* data, size_t size, size_t num_samples,
                     std::vector<ParsedVariant>& out_variants);

    bool parse_vcf_line(const char* line, size_t len, size_t num_samples,
                        ParsedVariant& out);

    void parse_genotype_field(const char* field, size_t len,
                              allele_t& allele0, allele_t& allele1);

    void parse_dosage_field(const char* field, size_t len, float& dosage);

    void parse_probability_field(const char* field, size_t len,
                                 float& p00, float& p01, float& p11);

    // Find line boundaries in chunk (for correct parallel parsing)
    std::pair<size_t, size_t> find_line_boundaries(const char* data, size_t size);
};

/**
 * @brief High-performance batch loader
 *
 * Loads multiple VCF regions in parallel for chromosome-level processing
 */
class BatchVCFLoader {
public:
    struct RegionSpec {
        std::string filename;
        std::string chrom;
        uint64_t start;
        uint64_t end;
    };

    explicit BatchVCFLoader(const ParallelLoadConfig& config = ParallelLoadConfig());

    /**
     * @brief Load multiple regions in parallel
     */
    std::vector<std::unique_ptr<ReferencePanel>> load_regions(
        const std::vector<RegionSpec>& regions,
        LoadProgressCallback progress = nullptr
    );

private:
    ParallelLoadConfig config_;
};

/**
 * @brief Factory function to create optimal loader for system
 */
std::unique_ptr<ParallelVCFLoader> create_optimal_loader(const std::string& filename);

/**
 * @brief Quick load with auto-configuration
 */
std::unique_ptr<ReferencePanel> quick_load_reference(
    const std::string& filename,
    bool verbose = true
);

std::unique_ptr<TargetData> quick_load_targets(
    const std::string& filename,
    bool verbose = true
);

} // namespace io
} // namespace swiftimpute
