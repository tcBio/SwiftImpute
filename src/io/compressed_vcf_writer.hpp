#pragma once

#include "../core/types.hpp"
#include "vcf_writer.hpp"
#include <string>
#include <vector>
#include <memory>

#ifdef HAVE_HTSLIB
#include <htslib/vcf.h>
#include <htslib/hts.h>
#include <htslib/bgzf.h>
#include <htslib/tbx.h>
#endif

namespace swiftimpute {

/**
 * @brief Configuration for compressed VCF output
 */
struct CompressedVCFConfig {
    int compression_level;      // 0-9, default 6 (same as bgzip default)
    bool create_index;          // Create .tbi index file
    bool write_header;          // Write VCF header (default true)
    int threads;                // Number of compression threads (0 = auto)

    CompressedVCFConfig()
        : compression_level(6)
        , create_index(true)
        , write_header(true)
        , threads(0) {}
};

/**
 * @brief Compressed VCF writer with bgzip and tabix support
 *
 * Uses htslib for efficient block-gzipped VCF output with optional
 * tabix indexing for random access.
 *
 * Usage:
 *   CompressedVCFWriter writer("output.vcf.gz");
 *   writer.write_header(samples);
 *   writer.write_variant(...);
 *   writer.close();  // Automatically creates .tbi index
 */
class CompressedVCFWriter {
public:
    CompressedVCFWriter() = default;

    /**
     * @brief Open compressed VCF file for writing
     *
     * @param filename Output filename (should end in .vcf.gz)
     * @param config Compression configuration
     */
    explicit CompressedVCFWriter(
        const std::string& filename,
        const CompressedVCFConfig& config = CompressedVCFConfig()
    );

    ~CompressedVCFWriter();

    // Non-copyable
    CompressedVCFWriter(const CompressedVCFWriter&) = delete;
    CompressedVCFWriter& operator=(const CompressedVCFWriter&) = delete;

    // Movable
    CompressedVCFWriter(CompressedVCFWriter&& other) noexcept;
    CompressedVCFWriter& operator=(CompressedVCFWriter&& other) noexcept;

    /**
     * @brief Open file for writing
     */
    void open(
        const std::string& filename,
        const CompressedVCFConfig& config = CompressedVCFConfig()
    );

    /**
     * @brief Close file and create index
     */
    void close();

    /**
     * @brief Check if file is open
     */
    bool is_open() const;

    /**
     * @brief Write VCF header
     */
    void write_header(
        const std::vector<std::string>& sample_names,
        const std::vector<std::string>& contigs = {}
    );

    /**
     * @brief Write variant with probabilities and dosages
     */
    void write_variant(
        const std::string& chrom,
        uint64_t position,
        const std::string& id,
        const std::string& ref,
        const std::vector<std::string>& alt,
        const std::vector<std::vector<prob_t>>& genotype_probs,
        const std::vector<prob_t>& dosages = {}
    );

    /**
     * @brief Write variant with phased haplotypes
     */
    void write_phased_variant(
        const std::string& chrom,
        uint64_t position,
        const std::string& id,
        const std::string& ref,
        const std::vector<std::string>& alt,
        const std::vector<std::vector<allele_t>>& phased_genotypes
    );

    /**
     * @brief Get filename
     */
    const std::string& filename() const { return filename_; }

    /**
     * @brief Check if htslib support is available
     */
    static bool has_htslib_support();

    /**
     * @brief Get compression statistics
     */
    struct CompressionStats {
        size_t uncompressed_bytes;
        size_t compressed_bytes;
        size_t num_variants;
        double compression_ratio;
    };

    CompressionStats get_stats() const;

private:
    std::string filename_;
    CompressedVCFConfig config_;
    bool header_written_ = false;
    size_t uncompressed_bytes_ = 0;
    size_t num_variants_ = 0;

#ifdef HAVE_HTSLIB
    BGZF* bgzf_ = nullptr;

    void write_line(const std::string& line);
    void create_tabix_index();
#else
    // Fallback to uncompressed writer
    std::unique_ptr<VCFWriter> fallback_writer_;
#endif

    std::string format_genotype_probs(const std::vector<prob_t>& probs);
    std::string format_dosage(prob_t dosage);
};

/**
 * @brief Factory function to create appropriate VCF writer
 *
 * Automatically selects compressed or uncompressed based on filename
 * and htslib availability.
 *
 * @param filename Output filename
 * @param config Compression config (ignored for uncompressed)
 * @return Unique pointer to writer interface
 */
std::unique_ptr<CompressedVCFWriter> create_vcf_writer(
    const std::string& filename,
    const CompressedVCFConfig& config = CompressedVCFConfig()
);

/**
 * @brief Create tabix index for an existing VCF.gz file
 *
 * @param vcf_filename Path to bgzipped VCF file
 * @return true if index creation succeeded
 */
bool create_tabix_index(const std::string& vcf_filename);

/**
 * @brief Compress an uncompressed VCF file to bgzip format
 *
 * @param input_vcf Path to uncompressed VCF
 * @param output_vcf_gz Path to output bgzipped VCF (optional, defaults to input + .gz)
 * @param create_index Whether to create .tbi index
 * @return true if compression succeeded
 */
bool compress_vcf(
    const std::string& input_vcf,
    const std::string& output_vcf_gz = "",
    bool create_index = true
);

} // namespace swiftimpute
