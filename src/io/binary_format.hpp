#pragma once

#include "core/types.hpp"
#include "mmap_reader.hpp"
#include <string>
#include <vector>
#include <memory>
#include <cstdint>

namespace swiftimpute {
namespace io {

/**
 * Binary Format Specification for SwiftImpute
 *
 * SwiftImpute uses custom binary formats optimized for:
 * - Fast loading (memory-mapped access)
 * - Minimal RAM footprint (stream from disk)
 * - Cross-platform compatibility
 *
 * All integers are stored in little-endian format.
 * All floats are IEEE 754 double-precision.
 *
 * File extensions:
 * - .swref  - Binary reference panel
 * - .swpbwt - Binary PBWT index
 * - .swres  - Binary imputation results
 */

// Version constants
constexpr uint32_t BINARY_FORMAT_VERSION = 1;
constexpr uint32_t REF_PANEL_MAGIC = 0x52455753;    // "SWER" (SwiftImpute REference)
constexpr uint32_t PBWT_INDEX_MAGIC = 0x50575753;   // "SWWP" (SwiftImpute PBWT)
constexpr uint32_t RESULT_MAGIC = 0x45525753;       // "SWRE" (SwiftImpute REsult)

/**
 * Common file header for all binary formats
 */
struct BinaryFileHeader {
    uint32_t magic;             // Format identifier
    uint32_t version;           // Format version
    uint32_t header_size;       // Size of this header
    uint32_t flags;             // Format-specific flags
    uint64_t created_timestamp; // Unix timestamp
    uint64_t data_offset;       // Offset to main data section
    uint64_t data_size;         // Size of main data section
    char reserved[32];          // Reserved for future use

    BinaryFileHeader() : magic(0), version(BINARY_FORMAT_VERSION),
        header_size(sizeof(BinaryFileHeader)), flags(0),
        created_timestamp(0), data_offset(0), data_size(0) {
        std::memset(reserved, 0, sizeof(reserved));
    }

    bool is_valid(uint32_t expected_magic) const {
        return magic == expected_magic && version <= BINARY_FORMAT_VERSION;
    }
};

// =============================================================================
// Reference Panel Binary Format
// =============================================================================

/**
 * Reference panel binary format header
 */
struct RefPanelHeader {
    BinaryFileHeader base;

    // Dimensions
    uint32_t num_markers;
    uint32_t num_samples;
    uint32_t num_haplotypes;

    // Section offsets (relative to base.data_offset)
    uint64_t markers_offset;    // Marker metadata
    uint64_t markers_size;
    uint64_t samples_offset;    // Sample metadata
    uint64_t samples_size;
    uint64_t haplotypes_offset; // Haplotype data
    uint64_t haplotypes_size;

    // Optional index offsets
    uint64_t chrom_index_offset; // Chromosome index (for fast filtering)
    uint64_t chrom_index_size;

    RefPanelHeader() : num_markers(0), num_samples(0), num_haplotypes(0),
        markers_offset(0), markers_size(0),
        samples_offset(0), samples_size(0),
        haplotypes_offset(0), haplotypes_size(0),
        chrom_index_offset(0), chrom_index_size(0) {
        base.magic = REF_PANEL_MAGIC;
    }
};

/**
 * Marker record in binary format
 */
struct BinaryMarker {
    uint64_t pos;           // Physical position
    double cM;              // Genetic position
    uint16_t chrom_idx;     // Index into chromosome list
    uint16_t ref_len;       // Reference allele length
    uint16_t alt_len;       // Alt allele length
    uint16_t flags;         // Marker flags
    // Followed by: ref_allele (ref_len bytes), alt_allele (alt_len bytes)
};

/**
 * Memory-mapped reference panel reader
 *
 * Provides random access to reference panel data without loading
 * the entire file into RAM. Ideal for large reference panels.
 */
class MappedReferencePanel {
public:
    /**
     * Open a binary reference panel file
     */
    explicit MappedReferencePanel(
        const std::string& filename,
        const MMapConfig& config = MMapConfig()
    );

    ~MappedReferencePanel() = default;

    /**
     * Get dimensions
     */
    uint32_t num_markers() const { return header_.num_markers; }
    uint32_t num_samples() const { return header_.num_samples; }
    uint32_t num_haplotypes() const { return header_.num_haplotypes; }

    /**
     * Get marker at index
     */
    Marker get_marker(uint32_t idx) const;

    /**
     * Get chromosome list
     */
    std::vector<std::string> get_chromosomes() const;

    /**
     * Get marker range for chromosome
     */
    void get_chromosome_range(
        const std::string& chrom,
        uint32_t& start_idx,
        uint32_t& end_idx
    ) const;

    /**
     * Get haplotype data for a marker range
     *
     * Returns pointer to haplotype data in memory-mapped file.
     * Data is organized as [markers][haplotypes].
     */
    const allele_t* haplotypes(uint32_t start_marker, uint32_t end_marker) const;

    /**
     * Get a window of haplotype data into provided buffer
     *
     * More efficient for non-contiguous access patterns.
     */
    void copy_haplotypes(
        uint32_t start_marker,
        uint32_t end_marker,
        allele_t* buffer
    ) const;

    /**
     * Get single haplotype value
     */
    allele_t get_allele(uint32_t marker, uint32_t haplotype) const;

    /**
     * Check if file is valid
     */
    bool is_valid() const { return file_.is_open() && header_.base.is_valid(REF_PANEL_MAGIC); }

    /**
     * Get filename
     */
    const std::string& filename() const { return filename_; }

private:
    std::string filename_;
    MemoryMappedFile file_;
    RefPanelHeader header_;
    std::vector<std::string> chromosomes_;
    std::vector<std::pair<uint32_t, uint32_t>> chrom_ranges_;

    void parse_header();
    void build_chromosome_index();
};

/**
 * Write reference panel to binary format
 */
void write_reference_panel_binary(
    const std::string& filename,
    const std::vector<Marker>& markers,
    const std::vector<Sample>& samples,
    const allele_t* haplotypes,
    uint32_t num_haplotypes
);

/**
 * Read reference panel from binary format (full load)
 */
void read_reference_panel_binary(
    const std::string& filename,
    std::vector<Marker>& markers,
    std::vector<Sample>& samples,
    std::unique_ptr<allele_t[]>& haplotypes,
    uint32_t& num_haplotypes
);

// =============================================================================
// PBWT Index Binary Format
// =============================================================================

/**
 * PBWT index binary format header
 */
struct PBWTHeader {
    BinaryFileHeader base;

    // Dimensions
    uint32_t num_markers;
    uint32_t num_haplotypes;

    // Data type sizes
    uint32_t haplotype_bytes;   // sizeof(haplotype_t)
    uint32_t marker_bytes;      // sizeof(marker_t)

    // Section offsets
    uint64_t prefix_offset;     // Prefix array data
    uint64_t prefix_size;
    uint64_t divergence_offset; // Divergence array data
    uint64_t divergence_size;

    // Statistics
    double avg_divergence;
    uint32_t max_divergence;

    PBWTHeader() : num_markers(0), num_haplotypes(0),
        haplotype_bytes(sizeof(haplotype_t)), marker_bytes(sizeof(marker_t)),
        prefix_offset(0), prefix_size(0),
        divergence_offset(0), divergence_size(0),
        avg_divergence(0), max_divergence(0) {
        base.magic = PBWT_INDEX_MAGIC;
    }
};

/**
 * Memory-mapped PBWT index reader
 *
 * Provides random access to PBWT index data without loading
 * the entire index into RAM. Critical for large reference panels.
 */
class MappedPBWTIndex {
public:
    /**
     * Open a binary PBWT index file
     */
    explicit MappedPBWTIndex(
        const std::string& filename,
        const MMapConfig& config = MMapConfig()
    );

    ~MappedPBWTIndex() = default;

    /**
     * Get dimensions
     */
    uint32_t num_markers() const { return header_.num_markers; }
    uint32_t num_haplotypes() const { return header_.num_haplotypes; }

    /**
     * Get prefix value at (marker, position)
     */
    haplotype_t prefix_at(uint32_t marker, uint32_t pos) const;

    /**
     * Get divergence value at (marker, position)
     */
    marker_t divergence_at(uint32_t marker, uint32_t pos) const;

    /**
     * Get pointer to prefix array for marker range
     */
    const haplotype_t* prefix_data(uint32_t start_marker, uint32_t end_marker) const;

    /**
     * Get pointer to divergence array for marker range
     */
    const marker_t* divergence_data(uint32_t start_marker, uint32_t end_marker) const;

    /**
     * Copy prefix data for marker range into buffer
     */
    void copy_prefix(
        uint32_t start_marker,
        uint32_t end_marker,
        haplotype_t* buffer
    ) const;

    /**
     * Copy divergence data for marker range into buffer
     */
    void copy_divergence(
        uint32_t start_marker,
        uint32_t end_marker,
        marker_t* buffer
    ) const;

    /**
     * Get statistics
     */
    double avg_divergence() const { return header_.avg_divergence; }
    uint32_t max_divergence() const { return header_.max_divergence; }

    /**
     * Check if file is valid
     */
    bool is_valid() const { return file_.is_open() && header_.base.is_valid(PBWT_INDEX_MAGIC); }

    /**
     * Get filename
     */
    const std::string& filename() const { return filename_; }

    /**
     * Memory usage of this reader (not including OS page cache)
     */
    size_t memory_usage() const;

private:
    std::string filename_;
    MemoryMappedFile file_;
    PBWTHeader header_;

    void parse_header();
};

/**
 * Write PBWT index to binary format
 */
void write_pbwt_binary(
    const std::string& filename,
    uint32_t num_markers,
    uint32_t num_haplotypes,
    const haplotype_t* prefix_data,
    const marker_t* divergence_data
);

/**
 * Read PBWT index from binary format (full load)
 */
void read_pbwt_binary(
    const std::string& filename,
    uint32_t& num_markers,
    uint32_t& num_haplotypes,
    std::vector<haplotype_t>& prefix_data,
    std::vector<marker_t>& divergence_data
);

// =============================================================================
// Conversion Utilities
// =============================================================================

/**
 * Convert VCF reference panel to binary format
 *
 * This is a one-time preprocessing step that significantly speeds up
 * subsequent loads and enables memory-mapped access.
 */
void convert_vcf_to_binary_reference(
    const std::string& vcf_path,
    const std::string& binary_path,
    const std::string& region = ""
);

/**
 * Check if binary file exists and is compatible
 */
bool is_binary_reference_valid(
    const std::string& binary_path,
    const std::string& vcf_path
);

/**
 * Check if PBWT binary file exists and matches reference
 */
bool is_pbwt_binary_valid(
    const std::string& pbwt_path,
    uint32_t expected_markers,
    uint32_t expected_haplotypes
);

} // namespace io
} // namespace swiftimpute
