#pragma once

#include "../core/types.hpp"
#include <string>
#include <vector>
#include <fstream>
#include <sstream>
#include <algorithm>
#include <memory>
#include <stdexcept>

#ifdef HAVE_HTSLIB
#include <htslib/vcf.h>
#include <htslib/hts.h>
#include <htslib/tbx.h>
#include <htslib/kstring.h>
#endif

namespace swiftimpute {

/**
 * @brief VCF/BCF file reader for loading genomic data
 *
 * Supports:
 * - Uncompressed VCF (.vcf)
 * - Gzip-compressed VCF (.vcf.gz) - requires htslib
 * - BCF binary format (.bcf) - requires htslib
 * - Region queries with tabix index - requires htslib
 */
class VCFReader {
public:
    struct Variant {
        std::string chrom;
        uint64_t position;
        std::string ref;
        std::vector<std::string> alt;
        std::vector<std::vector<allele_t>> genotypes;  // [sample][haplotype]
        double qual;
        std::string id;
    };

    struct Header {
        std::vector<std::string> sample_names;
        std::vector<std::string> contigs;
        size_t num_samples;
    };

    VCFReader() = default;
    explicit VCFReader(const std::string& filename);
    ~VCFReader();

    // Prevent copying (owns file handles)
    VCFReader(const VCFReader&) = delete;
    VCFReader& operator=(const VCFReader&) = delete;

    // Allow moving
    VCFReader(VCFReader&& other) noexcept;
    VCFReader& operator=(VCFReader&& other) noexcept;

    // Open/close file
    void open(const std::string& filename);
    void close();
    bool is_open() const;

    // Read header information
    const Header& read_header();
    const Header& header() const { return header_; }

    // Read variants
    bool read_variant(Variant& variant);
    std::vector<Variant> read_all_variants();
    std::vector<Variant> read_region(const std::string& region);

    // Query information
    size_t num_samples() const { return header_.num_samples; }
    const std::vector<std::string>& sample_names() const { return header_.sample_names; }

    // Check if htslib support is compiled in
    static bool has_htslib_support() {
#ifdef HAVE_HTSLIB
        return true;
#else
        return false;
#endif
    }

private:
    Header header_;
    bool header_parsed_ = false;
    std::string filename_;
    bool use_htslib_ = false;

    // Plain text file handle
    std::ifstream file_;

#ifdef HAVE_HTSLIB
    // htslib handles
    htsFile* hts_file_ = nullptr;
    bcf_hdr_t* hts_header_ = nullptr;
    bcf1_t* hts_record_ = nullptr;
    tbx_t* tbx_index_ = nullptr;
    hts_itr_t* hts_iter_ = nullptr;
    kstring_t kstr_ = {0, 0, nullptr};
#endif

    // Parsing helpers for plain text
    void parse_header_line(const std::string& line);
    bool parse_variant_line(const std::string& line, Variant& variant);
    std::vector<allele_t> parse_genotype(const std::string& gt_str);

    // File type detection
    bool is_gzipped(const std::string& filename) const;
    bool is_bcf(const std::string& filename) const;
    bool requires_htslib(const std::string& filename) const;

#ifdef HAVE_HTSLIB
    // htslib parsing helpers
    void parse_hts_header();
    bool read_hts_variant(Variant& variant);
    std::vector<allele_t> parse_hts_genotype(int32_t* gt_arr, int ngt);
#endif
};

// ============================================================================
// Implementation
// ============================================================================

inline VCFReader::VCFReader(const std::string& filename) {
    open(filename);
}

inline VCFReader::~VCFReader() {
    close();
}

inline VCFReader::VCFReader(VCFReader&& other) noexcept
    : header_(std::move(other.header_))
    , header_parsed_(other.header_parsed_)
    , filename_(std::move(other.filename_))
    , use_htslib_(other.use_htslib_)
    , file_(std::move(other.file_))
#ifdef HAVE_HTSLIB
    , hts_file_(other.hts_file_)
    , hts_header_(other.hts_header_)
    , hts_record_(other.hts_record_)
    , tbx_index_(other.tbx_index_)
    , hts_iter_(other.hts_iter_)
    , kstr_(other.kstr_)
#endif
{
#ifdef HAVE_HTSLIB
    other.hts_file_ = nullptr;
    other.hts_header_ = nullptr;
    other.hts_record_ = nullptr;
    other.tbx_index_ = nullptr;
    other.hts_iter_ = nullptr;
    other.kstr_ = {0, 0, nullptr};
#endif
    other.header_parsed_ = false;
    other.use_htslib_ = false;
}

inline VCFReader& VCFReader::operator=(VCFReader&& other) noexcept {
    if (this != &other) {
        close();
        header_ = std::move(other.header_);
        header_parsed_ = other.header_parsed_;
        filename_ = std::move(other.filename_);
        use_htslib_ = other.use_htslib_;
        file_ = std::move(other.file_);
#ifdef HAVE_HTSLIB
        hts_file_ = other.hts_file_;
        hts_header_ = other.hts_header_;
        hts_record_ = other.hts_record_;
        tbx_index_ = other.tbx_index_;
        hts_iter_ = other.hts_iter_;
        kstr_ = other.kstr_;
        other.hts_file_ = nullptr;
        other.hts_header_ = nullptr;
        other.hts_record_ = nullptr;
        other.tbx_index_ = nullptr;
        other.hts_iter_ = nullptr;
        other.kstr_ = {0, 0, nullptr};
#endif
        other.header_parsed_ = false;
        other.use_htslib_ = false;
    }
    return *this;
}

inline bool VCFReader::is_gzipped(const std::string& filename) const {
    return filename.size() >= 3 &&
           filename.substr(filename.size() - 3) == ".gz";
}

inline bool VCFReader::is_bcf(const std::string& filename) const {
    return filename.size() >= 4 &&
           filename.substr(filename.size() - 4) == ".bcf";
}

inline bool VCFReader::requires_htslib(const std::string& filename) const {
    return is_gzipped(filename) || is_bcf(filename);
}

inline void VCFReader::open(const std::string& filename) {
    close();
    filename_ = filename;
    header_parsed_ = false;

    if (requires_htslib(filename)) {
#ifdef HAVE_HTSLIB
        use_htslib_ = true;

        // Open with htslib
        hts_file_ = hts_open(filename.c_str(), "r");
        if (!hts_file_) {
            throw std::runtime_error("Failed to open VCF/BCF file with htslib: " + filename);
        }

        // Read header
        hts_header_ = bcf_hdr_read(hts_file_);
        if (!hts_header_) {
            hts_close(hts_file_);
            hts_file_ = nullptr;
            throw std::runtime_error("Failed to read VCF/BCF header: " + filename);
        }

        // Allocate record
        hts_record_ = bcf_init();
        if (!hts_record_) {
            bcf_hdr_destroy(hts_header_);
            hts_close(hts_file_);
            hts_header_ = nullptr;
            hts_file_ = nullptr;
            throw std::runtime_error("Failed to allocate VCF record");
        }

        // Try to load tabix index for region queries
        if (is_gzipped(filename)) {
            tbx_index_ = tbx_index_load(filename.c_str());
            // Index is optional - don't fail if not present
        }
#else
        throw std::runtime_error(
            "Compressed VCF/BCF files require htslib support. "
            "Please use uncompressed .vcf files or rebuild with ENABLE_HTSLIB=ON. "
            "File: " + filename
        );
#endif
    } else {
        use_htslib_ = false;
        file_.open(filename);
        if (!file_.is_open()) {
            throw std::runtime_error("Failed to open VCF file: " + filename);
        }
    }
}

inline void VCFReader::close() {
#ifdef HAVE_HTSLIB
    if (kstr_.s) {
        free(kstr_.s);
        kstr_ = {0, 0, nullptr};
    }
    if (hts_iter_) {
        hts_itr_destroy(hts_iter_);
        hts_iter_ = nullptr;
    }
    if (tbx_index_) {
        tbx_destroy(tbx_index_);
        tbx_index_ = nullptr;
    }
    if (hts_record_) {
        bcf_destroy(hts_record_);
        hts_record_ = nullptr;
    }
    if (hts_header_) {
        bcf_hdr_destroy(hts_header_);
        hts_header_ = nullptr;
    }
    if (hts_file_) {
        hts_close(hts_file_);
        hts_file_ = nullptr;
    }
#endif
    if (file_.is_open()) {
        file_.close();
    }
    use_htslib_ = false;
    header_parsed_ = false;
}

inline bool VCFReader::is_open() const {
#ifdef HAVE_HTSLIB
    if (use_htslib_) {
        return hts_file_ != nullptr;
    }
#endif
    return file_.is_open();
}

#ifdef HAVE_HTSLIB
inline void VCFReader::parse_hts_header() {
    if (!hts_header_) return;

    // Get sample names
    int nsamples = bcf_hdr_nsamples(hts_header_);
    header_.sample_names.clear();
    header_.sample_names.reserve(nsamples);
    for (int i = 0; i < nsamples; ++i) {
        header_.sample_names.push_back(hts_header_->samples[i]);
    }
    header_.num_samples = nsamples;

    // Get contig names
    header_.contigs.clear();
    int nseq = hts_header_->n[BCF_DT_CTG];
    for (int i = 0; i < nseq; ++i) {
        header_.contigs.push_back(bcf_hdr_id2name(hts_header_, i));
    }
}

inline std::vector<allele_t> VCFReader::parse_hts_genotype(int32_t* gt_arr, int ngt) {
    std::vector<allele_t> gt;
    gt.reserve(ngt);

    for (int i = 0; i < ngt; ++i) {
        if (gt_arr[i] == bcf_int32_vector_end) {
            break;  // End of genotype
        }
        if (bcf_gt_is_missing(gt_arr[i])) {
            gt.push_back(ALLELE_MISSING);
        } else {
            gt.push_back(static_cast<allele_t>(bcf_gt_allele(gt_arr[i])));
        }
    }

    return gt;
}

inline bool VCFReader::read_hts_variant(Variant& variant) {
    if (!hts_file_ || !hts_header_ || !hts_record_) {
        return false;
    }

    // Read next record
    int ret;
    if (hts_iter_) {
        // Reading with iterator (region query)
        ret = tbx_itr_next(hts_file_, tbx_index_, hts_iter_, &kstr_);
        if (ret < 0) {
            return false;
        }
        // Parse the line
        vcf_parse(&kstr_, hts_header_, hts_record_);
    } else {
        // Sequential read
        ret = bcf_read(hts_file_, hts_header_, hts_record_);
        if (ret < 0) {
            return false;
        }
    }

    // Unpack the record
    bcf_unpack(hts_record_, BCF_UN_ALL);

    // Extract fields
    variant.chrom = bcf_hdr_id2name(hts_header_, hts_record_->rid);
    variant.position = hts_record_->pos + 1;  // Convert 0-based to 1-based
    variant.id = hts_record_->d.id ? hts_record_->d.id : ".";
    variant.qual = bcf_float_is_missing(hts_record_->qual) ? 0.0 : hts_record_->qual;

    // REF allele
    variant.ref = hts_record_->d.allele[0];

    // ALT alleles
    variant.alt.clear();
    for (int i = 1; i < hts_record_->n_allele; ++i) {
        variant.alt.push_back(hts_record_->d.allele[i]);
    }

    // Extract genotypes
    int32_t* gt_arr = nullptr;
    int ngt_arr = 0;
    int ngt = bcf_get_genotypes(hts_header_, hts_record_, &gt_arr, &ngt_arr);

    variant.genotypes.clear();
    if (ngt > 0 && gt_arr) {
        int nsamples = bcf_hdr_nsamples(hts_header_);
        int ploidy = ngt / nsamples;

        variant.genotypes.reserve(nsamples);
        for (int s = 0; s < nsamples; ++s) {
            variant.genotypes.push_back(
                parse_hts_genotype(gt_arr + s * ploidy, ploidy)
            );
        }
    }

    if (gt_arr) {
        free(gt_arr);
    }

    return true;
}
#endif

inline const VCFReader::Header& VCFReader::read_header() {
    if (header_parsed_) {
        return header_;
    }

#ifdef HAVE_HTSLIB
    if (use_htslib_) {
        parse_hts_header();
        header_parsed_ = true;
        return header_;
    }
#endif

    // Plain text VCF parsing
    std::string line;
    while (std::getline(file_, line)) {
        if (line.empty()) continue;

        if (line[0] == '#') {
            if (line[1] == '#') {
                // Meta-information line
                parse_header_line(line);
            } else {
                // Column header line (#CHROM POS ID...)
                std::istringstream iss(line);
                std::vector<std::string> columns;
                std::string col;

                while (iss >> col) {
                    columns.push_back(col);
                }

                // Sample names start at column 9 (0-indexed)
                if (columns.size() > 9) {
                    header_.sample_names.assign(
                        columns.begin() + 9,
                        columns.end()
                    );
                    header_.num_samples = header_.sample_names.size();
                }
                break;
            }
        } else {
            // No more header lines
            break;
        }
    }

    header_parsed_ = true;
    return header_;
}

inline void VCFReader::parse_header_line(const std::string& line) {
    // Parse ##contig=<ID=chr1,length=248956422>
    if (line.find("##contig=") == 0) {
        size_t id_start = line.find("ID=");
        if (id_start != std::string::npos) {
            id_start += 3;
            size_t id_end = line.find_first_of(",>", id_start);
            if (id_end != std::string::npos) {
                header_.contigs.push_back(line.substr(id_start, id_end - id_start));
            }
        }
    }
}

inline bool VCFReader::read_variant(Variant& variant) {
    if (!header_parsed_) {
        read_header();
    }

#ifdef HAVE_HTSLIB
    if (use_htslib_) {
        return read_hts_variant(variant);
    }
#endif

    // Plain text parsing
    std::string line;
    while (std::getline(file_, line)) {
        if (line.empty() || line[0] == '#') continue;
        return parse_variant_line(line, variant);
    }

    return false;  // EOF
}

inline bool VCFReader::parse_variant_line(const std::string& line, Variant& variant) {
    std::istringstream iss(line);
    std::vector<std::string> fields;
    std::string field;

    while (std::getline(iss, field, '\t')) {
        fields.push_back(field);
    }

    if (fields.size() < 9) {
        return false;  // Invalid line
    }

    // Parse fixed fields
    variant.chrom = fields[0];
    variant.position = std::stoul(fields[1]);
    variant.id = fields[2];
    variant.ref = fields[3];

    // Parse ALT alleles
    variant.alt.clear();
    std::istringstream alt_stream(fields[4]);
    std::string alt_allele;
    while (std::getline(alt_stream, alt_allele, ',')) {
        variant.alt.push_back(alt_allele);
    }

    // Parse QUAL
    variant.qual = (fields[5] == ".") ? 0.0 : std::stod(fields[5]);

    // Parse genotypes (assuming GT is first in FORMAT)
    variant.genotypes.clear();
    for (size_t i = 9; i < fields.size(); ++i) {
        variant.genotypes.push_back(parse_genotype(fields[i]));
    }

    return true;
}

inline std::vector<allele_t> VCFReader::parse_genotype(const std::string& gt_str) {
    std::vector<allele_t> gt;

    // Find the GT field (first field in FORMAT)
    size_t colon_pos = gt_str.find(':');
    std::string gt_field = (colon_pos != std::string::npos)
        ? gt_str.substr(0, colon_pos)
        : gt_str;

    // Parse phased (|) or unphased (/) genotypes
    size_t delim_pos = gt_field.find_first_of("|/");
    if (delim_pos == std::string::npos) {
        // Haploid or missing
        if (gt_field == ".") {
            gt.push_back(ALLELE_MISSING);
        } else {
            gt.push_back(static_cast<allele_t>(std::stoi(gt_field)));
        }
    } else {
        // Diploid
        std::string allele1 = gt_field.substr(0, delim_pos);
        std::string allele2 = gt_field.substr(delim_pos + 1);

        gt.push_back(allele1 == "." ? ALLELE_MISSING : std::stoi(allele1));
        gt.push_back(allele2 == "." ? ALLELE_MISSING : std::stoi(allele2));
    }

    return gt;
}

inline std::vector<VCFReader::Variant> VCFReader::read_all_variants() {
    std::vector<Variant> variants;
    Variant v;

    while (read_variant(v)) {
        variants.push_back(v);
    }

    return variants;
}

inline std::vector<VCFReader::Variant> VCFReader::read_region(const std::string& region) {
#ifdef HAVE_HTSLIB
    if (!use_htslib_) {
        throw std::runtime_error(
            "Region queries on uncompressed VCF files require re-opening with htslib. "
            "Please use a .vcf.gz file with tabix index for region queries."
        );
    }

    if (!tbx_index_) {
        throw std::runtime_error(
            "Region queries require a tabix index. "
            "Create one with: tabix -p vcf " + filename_
        );
    }

    // Clean up any existing iterator
    if (hts_iter_) {
        hts_itr_destroy(hts_iter_);
        hts_iter_ = nullptr;
    }

    // Create iterator for region
    hts_iter_ = tbx_itr_querys(tbx_index_, region.c_str());
    if (!hts_iter_) {
        throw std::runtime_error("Failed to query region: " + region);
    }

    // Read all variants in region
    std::vector<Variant> variants;
    Variant v;
    while (read_hts_variant(v)) {
        variants.push_back(v);
    }

    // Clean up iterator
    hts_itr_destroy(hts_iter_);
    hts_iter_ = nullptr;

    return variants;
#else
    (void)region;  // Suppress unused parameter warning
    throw std::runtime_error(
        "Region queries require htslib support. "
        "Please rebuild with ENABLE_HTSLIB=ON"
    );
#endif
}

} // namespace swiftimpute
