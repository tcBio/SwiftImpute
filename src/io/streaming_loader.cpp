#include "streaming_loader.hpp"
#include <chrono>
#include <algorithm>

namespace swiftimpute {

StreamingReferenceLoader::StreamingReferenceLoader(
    const std::string& filename,
    const ChunkedLoadConfig& config
) : filename_(filename),
    config_(config),
    reader_(filename),
    header_loaded_(false)
{
    // Read header immediately
    header_ = reader_.read_header();
    header_loaded_ = true;

    LOG_INFO("StreamingReferenceLoader initialized for: " + filename);
    LOG_INFO("  Samples: " + std::to_string(header_.num_samples));
    LOG_INFO("  Max markers per chunk: " + std::to_string(config_.max_markers_per_chunk));
}

StreamingReferenceLoader::~StreamingReferenceLoader() {
    reader_.close();
}

ChunkLoadStats StreamingReferenceLoader::scan_file() {
    ChunkLoadStats stats;
    stats.total_haplotypes = header_.num_samples * 2;

    // Re-open to scan
    VCFReader scan_reader(filename_);
    scan_reader.read_header();

    VCFReader::Variant variant;
    while (scan_reader.read_variant(variant)) {
        stats.total_markers++;
    }

    stats.num_chunks = (stats.total_markers + config_.max_markers_per_chunk - 1) /
                       config_.max_markers_per_chunk;

    stats.peak_memory_bytes = memory_for_chunk(
        std::min(stats.total_markers, config_.max_markers_per_chunk),
        static_cast<haplotype_t>(stats.total_haplotypes)
    );

    return stats;
}

ChunkLoadStats StreamingReferenceLoader::process_chunks(ChunkProcessor processor) {
    auto start_time = std::chrono::high_resolution_clock::now();

    ChunkLoadStats stats;
    stats.total_haplotypes = header_.num_samples * 2;

    // Re-open for fresh read
    reader_.close();
    reader_.open(filename_);
    reader_.read_header();

    size_t chunk_idx = 0;
    size_t global_marker = 0;

    while (true) {
        auto chunk = load_next_chunk();
        if (!chunk || chunk->markers.empty()) {
            break;
        }

        chunk->start_marker = global_marker;
        chunk->end_marker = global_marker + chunk->num_markers();

        LOG_INFO("Processing chunk " + std::to_string(chunk_idx + 1) +
                 ": markers " + std::to_string(chunk->start_marker) +
                 " to " + std::to_string(chunk->end_marker));

        // Call processor
        processor(*chunk, chunk_idx);

        stats.total_markers += chunk->num_markers();
        stats.peak_memory_bytes = std::max(
            stats.peak_memory_bytes,
            memory_for_chunk(chunk->num_markers(), chunk->num_haplotypes)
        );

        global_marker = chunk->end_marker;
        chunk_idx++;
    }

    stats.num_chunks = chunk_idx;

    auto end_time = std::chrono::high_resolution_clock::now();
    stats.load_time_seconds = std::chrono::duration<double>(
        end_time - start_time
    ).count();

    return stats;
}

std::unique_ptr<ReferenceChunk> StreamingReferenceLoader::load_region(
    const std::string& region
) {
    auto variants = reader_.read_region(region);

    if (variants.empty()) {
        return nullptr;
    }

    auto chunk = std::make_unique<ReferenceChunk>();
    chunk->num_haplotypes = static_cast<haplotype_t>(header_.num_samples * 2);

    // Build markers
    chunk->markers.reserve(variants.size());
    for (const auto& v : variants) {
        Marker m;
        m.chrom = v.chrom;
        m.pos = v.position;
        m.id = v.id;
        m.ref = v.ref;
        m.alt = v.alt.empty() ? "" : v.alt[0];
        m.alt_alleles = v.alt;
        m.n_alleles = static_cast<uint8_t>(v.alt.size() + 1);
        m.is_multiallelic = v.alt.size() > 1;
        m.allele_index = 0;
        m.cM = 0.0;
        chunk->markers.push_back(m);
    }

    // Allocate haplotypes
    size_t total_size = chunk->markers.size() * chunk->num_haplotypes;
    chunk->haplotypes = std::make_unique<allele_t[]>(total_size);

    // Fill haplotype data
    for (size_t m = 0; m < variants.size(); ++m) {
        const auto& genotypes = variants[m].genotypes;

        for (size_t s = 0; s < header_.num_samples; ++s) {
            if (s < genotypes.size() && genotypes[s].size() >= 2) {
                chunk->haplotypes[m * chunk->num_haplotypes + s * 2 + 0] =
                    (genotypes[s][0] == ALLELE_MISSING) ? ALLELE_MISSING :
                    (genotypes[s][0] > 0) ? 1 : 0;
                chunk->haplotypes[m * chunk->num_haplotypes + s * 2 + 1] =
                    (genotypes[s][1] == ALLELE_MISSING) ? ALLELE_MISSING :
                    (genotypes[s][1] > 0) ? 1 : 0;
            } else {
                chunk->haplotypes[m * chunk->num_haplotypes + s * 2 + 0] = ALLELE_MISSING;
                chunk->haplotypes[m * chunk->num_haplotypes + s * 2 + 1] = ALLELE_MISSING;
            }
        }
    }

    chunk->start_marker = 0;
    chunk->end_marker = chunk->markers.size();

    return chunk;
}

std::unique_ptr<ReferenceChunk> StreamingReferenceLoader::load_next_chunk() {
    std::vector<VCFReader::Variant> variants;
    variants.reserve(config_.max_markers_per_chunk);

    VCFReader::Variant variant;
    while (variants.size() < config_.max_markers_per_chunk && reader_.read_variant(variant)) {
        variants.push_back(variant);
    }

    if (variants.empty()) {
        return nullptr;
    }

    auto chunk = std::make_unique<ReferenceChunk>();
    chunk->num_haplotypes = static_cast<haplotype_t>(header_.num_samples * 2);

    // Build markers
    chunk->markers.reserve(variants.size());
    for (const auto& v : variants) {
        Marker m;
        m.chrom = v.chrom;
        m.pos = v.position;
        m.id = v.id;
        m.ref = v.ref;
        m.alt = v.alt.empty() ? "" : v.alt[0];
        m.alt_alleles = v.alt;
        m.n_alleles = static_cast<uint8_t>(v.alt.size() + 1);
        m.is_multiallelic = v.alt.size() > 1;
        m.allele_index = 0;
        m.cM = 0.0;
        chunk->markers.push_back(m);
    }

    // Allocate haplotypes
    size_t total_size = chunk->markers.size() * chunk->num_haplotypes;
    chunk->haplotypes = std::make_unique<allele_t[]>(total_size);

    // Fill haplotype data
    for (size_t m = 0; m < variants.size(); ++m) {
        const auto& genotypes = variants[m].genotypes;

        for (size_t s = 0; s < header_.num_samples; ++s) {
            if (s < genotypes.size() && genotypes[s].size() >= 2) {
                chunk->haplotypes[m * chunk->num_haplotypes + s * 2 + 0] =
                    (genotypes[s][0] == ALLELE_MISSING) ? ALLELE_MISSING :
                    (genotypes[s][0] > 0) ? 1 : 0;
                chunk->haplotypes[m * chunk->num_haplotypes + s * 2 + 1] =
                    (genotypes[s][1] == ALLELE_MISSING) ? ALLELE_MISSING :
                    (genotypes[s][1] > 0) ? 1 : 0;
            } else {
                chunk->haplotypes[m * chunk->num_haplotypes + s * 2 + 0] = ALLELE_MISSING;
                chunk->haplotypes[m * chunk->num_haplotypes + s * 2 + 1] = ALLELE_MISSING;
            }
        }
    }

    return chunk;
}

size_t StreamingReferenceLoader::estimate_num_chunks() const {
    // Rough estimate based on file size
    // More accurate after calling scan_file()
    return 10;  // Placeholder
}

size_t StreamingReferenceLoader::optimal_chunk_size(
    haplotype_t num_haplotypes,
    size_t available_memory
) {
    // Each marker uses: num_haplotypes bytes for alleles + ~100 bytes for Marker struct
    size_t bytes_per_marker = static_cast<size_t>(num_haplotypes) + 100;

    // Use 80% of available memory for safety
    size_t usable_memory = (available_memory * 80) / 100;

    return usable_memory / bytes_per_marker;
}

size_t StreamingReferenceLoader::memory_for_chunk(
    size_t num_markers,
    haplotype_t num_haplotypes
) {
    size_t haplotype_bytes = num_markers * static_cast<size_t>(num_haplotypes) * sizeof(allele_t);
    size_t marker_bytes = num_markers * sizeof(Marker);

    // Approximate string storage in Marker
    size_t string_bytes = num_markers * 100;  // ~100 bytes per marker for strings

    return haplotype_bytes + marker_bytes + string_bytes;
}

// ChunkIterator implementation

ChunkIterator::ChunkIterator(StreamingReferenceLoader& loader)
    : loader_(loader),
      current_chunk_(0),
      has_more_(true)
{
}

bool ChunkIterator::has_next() const {
    return has_more_;
}

std::unique_ptr<ReferenceChunk> ChunkIterator::next() {
    // This requires internal access to loader - simplified implementation
    current_chunk_++;
    return nullptr;  // Would need proper implementation with loader access
}

// Factory function

std::unique_ptr<StreamingReferenceLoader> create_optimal_loader(
    const std::string& filename,
    size_t available_gpu_memory,
    size_t available_host_memory
) {
    ChunkedLoadConfig config;

    // Start with a quick scan to get sample count
    VCFReader quick_reader(filename);
    auto header = quick_reader.read_header();
    quick_reader.close();

    haplotype_t num_haplotypes = static_cast<haplotype_t>(header.num_samples * 2);

    // Use the smaller of GPU or host memory constraints
    size_t available = std::min(available_gpu_memory, available_host_memory);

    config.max_markers_per_chunk = StreamingReferenceLoader::optimal_chunk_size(
        num_haplotypes,
        available
    );

    // Clamp to reasonable limits
    config.max_markers_per_chunk = std::max(config.max_markers_per_chunk, size_t(10000));
    config.max_markers_per_chunk = std::min(config.max_markers_per_chunk, size_t(1000000));

    config.max_memory_bytes = available;

    LOG_INFO("Created optimal loader: " + std::to_string(config.max_markers_per_chunk) +
             " markers per chunk");

    return std::make_unique<StreamingReferenceLoader>(filename, config);
}

} // namespace swiftimpute
