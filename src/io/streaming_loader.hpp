#pragma once

#include "../core/types.hpp"
#include "../io/vcf_reader.hpp"
#include <string>
#include <vector>
#include <memory>
#include <functional>

namespace swiftimpute {

/**
 * @brief Configuration for chunked/streaming reference loading
 */
struct ChunkedLoadConfig {
    size_t max_markers_per_chunk;       // Max markers to load at once
    size_t max_memory_bytes;            // Max memory to use for reference data
    bool use_memory_mapping;            // Memory-map large files
    bool precompute_transitions;        // Precompute transitions during load

    ChunkedLoadConfig() :
        max_markers_per_chunk(100000),  // 100K markers per chunk
        max_memory_bytes(8ULL * 1024 * 1024 * 1024),  // 8 GB default
        use_memory_mapping(true),
        precompute_transitions(true) {}
};

/**
 * @brief Statistics about the chunked loading process
 */
struct ChunkLoadStats {
    size_t total_markers;
    size_t total_haplotypes;
    size_t num_chunks;
    size_t peak_memory_bytes;
    double load_time_seconds;

    ChunkLoadStats() :
        total_markers(0),
        total_haplotypes(0),
        num_chunks(0),
        peak_memory_bytes(0),
        load_time_seconds(0) {}
};

/**
 * @brief A chunk of the reference panel for streaming processing
 */
struct ReferenceChunk {
    std::vector<Marker> markers;
    std::unique_ptr<allele_t[]> haplotypes;  // [marker][haplotype]
    haplotype_t num_haplotypes;
    size_t start_marker;                      // Global marker index
    size_t end_marker;                        // Global marker index (exclusive)

    size_t num_markers() const { return markers.size(); }

    allele_t get_allele(size_t local_marker, haplotype_t hap) const {
        return haplotypes[local_marker * num_haplotypes + hap];
    }
};

/**
 * @brief Callback for processing each chunk
 */
using ChunkProcessor = std::function<void(const ReferenceChunk&, size_t chunk_idx)>;

/**
 * @brief Streaming reference panel loader for WGS-scale datasets
 *
 * Enables processing of very large reference panels (>8M variants) that
 * don't fit in GPU memory by:
 * - Loading data in chunks
 * - Processing each chunk independently
 * - Optionally using memory mapping for large files
 */
class StreamingReferenceLoader {
public:
    /**
     * @brief Construct a streaming loader
     *
     * @param filename VCF file path
     * @param config Loading configuration
     */
    StreamingReferenceLoader(
        const std::string& filename,
        const ChunkedLoadConfig& config = ChunkedLoadConfig()
    );

    ~StreamingReferenceLoader();

    /**
     * @brief Get statistics about the file without fully loading
     */
    ChunkLoadStats scan_file();

    /**
     * @brief Load and process the reference panel in chunks
     *
     * @param processor Callback function to process each chunk
     * @return ChunkLoadStats with loading statistics
     */
    ChunkLoadStats process_chunks(ChunkProcessor processor);

    /**
     * @brief Load a specific region as a single chunk
     *
     * @param region Genomic region (chr:start-end)
     * @return The loaded chunk
     */
    std::unique_ptr<ReferenceChunk> load_region(const std::string& region);

    /**
     * @brief Estimate the number of chunks for the entire file
     */
    size_t estimate_num_chunks() const;

    /**
     * @brief Get the optimal chunk size based on available memory
     *
     * @param num_haplotypes Number of haplotypes in reference
     * @param available_memory Available memory in bytes
     * @return Recommended markers per chunk
     */
    static size_t optimal_chunk_size(
        haplotype_t num_haplotypes,
        size_t available_memory
    );

    /**
     * @brief Calculate memory required for a chunk
     *
     * @param num_markers Number of markers
     * @param num_haplotypes Number of haplotypes
     * @return Memory in bytes
     */
    static size_t memory_for_chunk(
        size_t num_markers,
        haplotype_t num_haplotypes
    );

private:
    std::string filename_;
    ChunkedLoadConfig config_;
    VCFReader reader_;
    VCFReader::Header header_;
    bool header_loaded_;

    std::unique_ptr<ReferenceChunk> load_next_chunk();
};

/**
 * @brief Iterator-style interface for streaming chunks
 */
class ChunkIterator {
public:
    ChunkIterator(StreamingReferenceLoader& loader);

    bool has_next() const;
    std::unique_ptr<ReferenceChunk> next();
    size_t current_chunk_index() const { return current_chunk_; }

private:
    StreamingReferenceLoader& loader_;
    size_t current_chunk_;
    bool has_more_;
};

/**
 * @brief Factory function to estimate memory and create optimal loader
 *
 * @param filename VCF file path
 * @param available_gpu_memory Available GPU memory in bytes
 * @param available_host_memory Available host memory in bytes
 * @return Configured streaming loader
 */
std::unique_ptr<StreamingReferenceLoader> create_optimal_loader(
    const std::string& filename,
    size_t available_gpu_memory,
    size_t available_host_memory
);

} // namespace swiftimpute
