#include "compressed_vcf_writer.hpp"
#include <sstream>
#include <iomanip>
#include <fstream>
#include <cstring>

namespace swiftimpute {

// ============================================================================
// Static methods
// ============================================================================

bool CompressedVCFWriter::has_htslib_support() {
#ifdef HAVE_HTSLIB
    return true;
#else
    return false;
#endif
}

// ============================================================================
// CompressedVCFWriter Implementation
// ============================================================================

CompressedVCFWriter::CompressedVCFWriter(
    const std::string& filename,
    const CompressedVCFConfig& config
) {
    open(filename, config);
}

CompressedVCFWriter::~CompressedVCFWriter() {
    close();
}

CompressedVCFWriter::CompressedVCFWriter(CompressedVCFWriter&& other) noexcept
    : filename_(std::move(other.filename_))
    , config_(other.config_)
    , header_written_(other.header_written_)
    , uncompressed_bytes_(other.uncompressed_bytes_)
    , num_variants_(other.num_variants_)
#ifdef HAVE_HTSLIB
    , bgzf_(other.bgzf_)
#else
    , fallback_writer_(std::move(other.fallback_writer_))
#endif
{
#ifdef HAVE_HTSLIB
    other.bgzf_ = nullptr;
#endif
    other.header_written_ = false;
}

CompressedVCFWriter& CompressedVCFWriter::operator=(CompressedVCFWriter&& other) noexcept {
    if (this != &other) {
        close();

        filename_ = std::move(other.filename_);
        config_ = other.config_;
        header_written_ = other.header_written_;
        uncompressed_bytes_ = other.uncompressed_bytes_;
        num_variants_ = other.num_variants_;

#ifdef HAVE_HTSLIB
        bgzf_ = other.bgzf_;
        other.bgzf_ = nullptr;
#else
        fallback_writer_ = std::move(other.fallback_writer_);
#endif
        other.header_written_ = false;
    }
    return *this;
}

void CompressedVCFWriter::open(
    const std::string& filename,
    const CompressedVCFConfig& config
) {
    close();  // Close any existing file

    filename_ = filename;
    config_ = config;
    header_written_ = false;
    uncompressed_bytes_ = 0;
    num_variants_ = 0;

    bool is_compressed = (filename.size() >= 3 &&
                          filename.substr(filename.size() - 3) == ".gz");

#ifdef HAVE_HTSLIB
    if (is_compressed) {
        // Open BGZF file for writing
        std::string mode = "w";
        mode += std::to_string(config_.compression_level);

        bgzf_ = bgzf_open(filename.c_str(), mode.c_str());
        if (!bgzf_) {
            throw std::runtime_error("Failed to open compressed VCF file: " + filename);
        }

        // Set compression threads if requested
        if (config_.threads > 0) {
            bgzf_mt(bgzf_, config_.threads, 256);
        }

        LOG_INFO("Opened compressed VCF writer: " + filename +
                 " (compression level " + std::to_string(config_.compression_level) + ")");
    } else {
        // Use uncompressed output via fallback
        fallback_writer_ = std::make_unique<VCFWriter>(filename);
    }
#else
    if (is_compressed) {
        throw std::runtime_error(
            "Compressed VCF output requires htslib. "
            "Build with -DENABLE_HTSLIB=ON or use uncompressed .vcf output."
        );
    }
    fallback_writer_ = std::make_unique<VCFWriter>(filename);
#endif
}

void CompressedVCFWriter::close() {
#ifdef HAVE_HTSLIB
    if (bgzf_) {
        bgzf_close(bgzf_);
        bgzf_ = nullptr;

        // Create tabix index if requested
        if (config_.create_index && header_written_) {
            create_tabix_index();
        }
    }

    if (fallback_writer_) {
        fallback_writer_->close();
        fallback_writer_.reset();
    }
#else
    if (fallback_writer_) {
        fallback_writer_->close();
        fallback_writer_.reset();
    }
#endif

    header_written_ = false;
}

bool CompressedVCFWriter::is_open() const {
#ifdef HAVE_HTSLIB
    return bgzf_ != nullptr || (fallback_writer_ && fallback_writer_->is_open());
#else
    return fallback_writer_ && fallback_writer_->is_open();
#endif
}

#ifdef HAVE_HTSLIB
void CompressedVCFWriter::write_line(const std::string& line) {
    if (!bgzf_) {
        throw std::runtime_error("Compressed VCF writer not open");
    }

    ssize_t written = bgzf_write(bgzf_, line.c_str(), line.size());
    if (written < 0 || static_cast<size_t>(written) != line.size()) {
        throw std::runtime_error("Failed to write to compressed VCF");
    }

    uncompressed_bytes_ += line.size();
}

void CompressedVCFWriter::create_tabix_index() {
    LOG_INFO("Creating tabix index for " + filename_);

    int ret = tbx_index_build(filename_.c_str(), 0, &tbx_conf_vcf);
    if (ret < 0) {
        LOG_WARNING("Failed to create tabix index for " + filename_);
    } else {
        LOG_INFO("Tabix index created: " + filename_ + ".tbi");
    }
}
#endif

void CompressedVCFWriter::write_header(
    const std::vector<std::string>& sample_names,
    const std::vector<std::string>& contigs
) {
#ifdef HAVE_HTSLIB
    if (fallback_writer_) {
        fallback_writer_->write_header(sample_names, contigs);
        header_written_ = true;
        return;
    }

    std::ostringstream header;

    // VCF version
    header << "##fileformat=VCFv4.2\n";

    // Source
    header << "##source=SwiftImpute\n";

    // Contigs
    for (const auto& contig : contigs) {
        header << "##contig=<ID=" << contig << ">\n";
    }

    // INFO fields
    header << "##INFO=<ID=IMP,Number=0,Type=Flag,Description=\"Imputed variant\">\n";
    header << "##INFO=<ID=INFO,Number=1,Type=Float,Description=\"Imputation INFO score\">\n";

    // FORMAT fields
    header << "##FORMAT=<ID=GT,Number=1,Type=String,Description=\"Genotype\">\n";
    header << "##FORMAT=<ID=DS,Number=1,Type=Float,Description=\"Dosage (expected ALT allele count)\">\n";
    header << "##FORMAT=<ID=GP,Number=G,Type=Float,Description=\"Genotype posterior probabilities\">\n";
    header << "##FORMAT=<ID=AP,Number=2,Type=Float,Description=\"Allelic probabilities\">\n";

    // Column header
    header << "#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT";
    for (const auto& sample : sample_names) {
        header << "\t" << sample;
    }
    header << "\n";

    write_line(header.str());
    header_written_ = true;
#else
    if (fallback_writer_) {
        fallback_writer_->write_header(sample_names, contigs);
        header_written_ = true;
    }
#endif
}

void CompressedVCFWriter::write_variant(
    const std::string& chrom,
    uint64_t position,
    const std::string& id,
    const std::string& ref,
    const std::vector<std::string>& alt,
    const std::vector<std::vector<prob_t>>& genotype_probs,
    const std::vector<prob_t>& dosages
) {
#ifdef HAVE_HTSLIB
    if (fallback_writer_) {
        fallback_writer_->write_variant(chrom, position, id, ref, alt, genotype_probs, dosages);
        num_variants_++;
        return;
    }

    if (!header_written_) {
        throw std::runtime_error("Must write header before writing variants");
    }

    std::ostringstream line;
    line << std::fixed << std::setprecision(4);

    // Fixed fields
    line << chrom << "\t"
         << position << "\t"
         << (id.empty() ? "." : id) << "\t"
         << ref << "\t";

    // ALT field
    for (size_t i = 0; i < alt.size(); ++i) {
        if (i > 0) line << ",";
        line << alt[i];
    }

    // QUAL, FILTER, INFO
    line << "\t.\tPASS\tIMP\t";

    // FORMAT
    if (dosages.empty()) {
        line << "GT:GP";
    } else {
        line << "GT:DS:GP";
    }

    // Sample genotypes
    for (size_t s = 0; s < genotype_probs.size(); ++s) {
        const auto& probs = genotype_probs[s];

        // Best genotype
        size_t best_gt = 0;
        prob_t best_prob = probs[0];
        for (size_t i = 1; i < probs.size(); ++i) {
            if (probs[i] > best_prob) {
                best_prob = probs[i];
                best_gt = i;
            }
        }

        // Convert genotype index
        std::string gt_str;
        if (alt.size() == 1) {
            switch (best_gt) {
                case 0: gt_str = "0|0"; break;
                case 1: gt_str = "0|1"; break;
                case 2: gt_str = "1|1"; break;
                default: gt_str = "./."; break;
            }
        } else {
            gt_str = std::to_string(best_gt) + "/" + std::to_string(best_gt);
        }

        line << "\t" << gt_str;

        // Dosage
        if (!dosages.empty()) {
            line << ":" << format_dosage(dosages[s]);
        }

        // Genotype probabilities
        line << ":" << format_genotype_probs(probs);
    }

    line << "\n";

    write_line(line.str());
    num_variants_++;
#else
    if (fallback_writer_) {
        fallback_writer_->write_variant(chrom, position, id, ref, alt, genotype_probs, dosages);
        num_variants_++;
    }
#endif
}

void CompressedVCFWriter::write_phased_variant(
    const std::string& chrom,
    uint64_t position,
    const std::string& id,
    const std::string& ref,
    const std::vector<std::string>& alt,
    const std::vector<std::vector<allele_t>>& phased_genotypes
) {
#ifdef HAVE_HTSLIB
    if (fallback_writer_) {
        fallback_writer_->write_phased_variant(chrom, position, id, ref, alt, phased_genotypes);
        num_variants_++;
        return;
    }

    if (!header_written_) {
        throw std::runtime_error("Must write header before writing variants");
    }

    std::ostringstream line;

    // Fixed fields
    line << chrom << "\t"
         << position << "\t"
         << (id.empty() ? "." : id) << "\t"
         << ref << "\t";

    // ALT field
    for (size_t i = 0; i < alt.size(); ++i) {
        if (i > 0) line << ",";
        line << alt[i];
    }

    // QUAL, FILTER, INFO, FORMAT
    line << "\t.\tPASS\tIMP\tGT";

    // Sample genotypes
    for (const auto& gt : phased_genotypes) {
        line << "\t";
        if (gt.size() == 2) {
            line << static_cast<int>(gt[0]) << "|" << static_cast<int>(gt[1]);
        } else {
            line << "./.";
        }
    }

    line << "\n";

    write_line(line.str());
    num_variants_++;
#else
    if (fallback_writer_) {
        fallback_writer_->write_phased_variant(chrom, position, id, ref, alt, phased_genotypes);
        num_variants_++;
    }
#endif
}

CompressedVCFWriter::CompressionStats CompressedVCFWriter::get_stats() const {
    CompressionStats stats;
    stats.uncompressed_bytes = uncompressed_bytes_;
    stats.num_variants = num_variants_;

#ifdef HAVE_HTSLIB
    if (bgzf_) {
        // Get compressed size from file (approximate)
        stats.compressed_bytes = bgzf_tell(bgzf_);
    } else {
        stats.compressed_bytes = uncompressed_bytes_;  // No compression
    }
#else
    stats.compressed_bytes = uncompressed_bytes_;
#endif

    if (stats.compressed_bytes > 0) {
        stats.compression_ratio =
            static_cast<double>(stats.uncompressed_bytes) / stats.compressed_bytes;
    } else {
        stats.compression_ratio = 1.0;
    }

    return stats;
}

std::string CompressedVCFWriter::format_genotype_probs(const std::vector<prob_t>& probs) {
    std::ostringstream oss;
    oss << std::fixed << std::setprecision(4);

    for (size_t i = 0; i < probs.size(); ++i) {
        if (i > 0) oss << ",";
        oss << probs[i];
    }

    return oss.str();
}

std::string CompressedVCFWriter::format_dosage(prob_t dosage) {
    std::ostringstream oss;
    oss << std::fixed << std::setprecision(4) << dosage;
    return oss.str();
}

// ============================================================================
// Factory and utility functions
// ============================================================================

std::unique_ptr<CompressedVCFWriter> create_vcf_writer(
    const std::string& filename,
    const CompressedVCFConfig& config
) {
    return std::make_unique<CompressedVCFWriter>(filename, config);
}

bool create_tabix_index(const std::string& vcf_filename) {
#ifdef HAVE_HTSLIB
    LOG_INFO("Creating tabix index for " + vcf_filename);

    int ret = tbx_index_build(vcf_filename.c_str(), 0, &tbx_conf_vcf);
    if (ret < 0) {
        LOG_WARNING("Failed to create tabix index");
        return false;
    }

    LOG_INFO("Tabix index created: " + vcf_filename + ".tbi");
    return true;
#else
    LOG_WARNING("Tabix indexing requires htslib support");
    return false;
#endif
}

bool compress_vcf(
    const std::string& input_vcf,
    const std::string& output_vcf_gz,
    bool create_index
) {
#ifdef HAVE_HTSLIB
    std::string output = output_vcf_gz.empty() ? input_vcf + ".gz" : output_vcf_gz;

    LOG_INFO("Compressing " + input_vcf + " to " + output);

    // Open input file
    std::ifstream input(input_vcf, std::ios::binary);
    if (!input) {
        LOG_ERROR("Failed to open input file: " + input_vcf);
        return false;
    }

    // Open output BGZF file
    BGZF* bgzf = bgzf_open(output.c_str(), "w");
    if (!bgzf) {
        LOG_ERROR("Failed to open output file: " + output);
        return false;
    }

    // Copy with compression
    char buffer[65536];
    size_t total_bytes = 0;

    while (input) {
        input.read(buffer, sizeof(buffer));
        std::streamsize bytes_read = input.gcount();

        if (bytes_read > 0) {
            ssize_t written = bgzf_write(bgzf, buffer, bytes_read);
            if (written < 0 || written != bytes_read) {
                LOG_ERROR("Failed to write to compressed file");
                bgzf_close(bgzf);
                return false;
            }
            total_bytes += bytes_read;
        }
    }

    bgzf_close(bgzf);

    LOG_INFO("Compressed " + std::to_string(total_bytes) + " bytes");

    // Create tabix index if requested
    if (create_index) {
        return create_tabix_index(output);
    }

    return true;
#else
    LOG_WARNING("VCF compression requires htslib support");
    return false;
#endif
}

} // namespace swiftimpute
