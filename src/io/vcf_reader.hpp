#pragma once

#include "../core/types.hpp"
#include <string>
#include <vector>
#include <fstream>
#include <sstream>
#include <algorithm>

namespace swiftimpute {

/**
 * @brief VCF/BCF file reader for loading genomic data
 *
 * Supports:
 * - Uncompressed VCF (.vcf)
 * - Gzip-compressed VCF (.vcf.gz) - requires htslib
 * - BCF binary format (.bcf) - requires htslib
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

    // Open/close file
    void open(const std::string& filename);
    void close();
    bool is_open() const { return file_.is_open(); }

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

private:
    std::ifstream file_;
    Header header_;
    bool header_parsed_ = false;
    std::string filename_;

    // Parsing helpers
    void parse_header_line(const std::string& line);
    bool parse_variant_line(const std::string& line, Variant& variant);
    std::vector<allele_t> parse_genotype(const std::string& gt_str);

    // Check if file is gzipped
    bool is_gzipped(const std::string& filename);
};

// Implementation

inline VCFReader::VCFReader(const std::string& filename) {
    open(filename);
}

inline VCFReader::~VCFReader() {
    close();
}

inline void VCFReader::open(const std::string& filename) {
    filename_ = filename;

    if (is_gzipped(filename)) {
        throw std::runtime_error(
            "Gzipped VCF files require htslib support. "
            "Please use uncompressed .vcf files or build with ENABLE_HTSLIB=ON"
        );
    }

    file_.open(filename);
    if (!file_.is_open()) {
        throw std::runtime_error("Failed to open VCF file: " + filename);
    }
}

inline void VCFReader::close() {
    if (file_.is_open()) {
        file_.close();
    }
}

inline bool VCFReader::is_gzipped(const std::string& filename) {
    return filename.size() >= 3 &&
           filename.substr(filename.size() - 3) == ".gz";
}

inline const VCFReader::Header& VCFReader::read_header() {
    if (header_parsed_) {
        return header_;
    }

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
    // TODO: Implement tabix-based region queries (requires htslib)
    throw std::runtime_error("Region queries require htslib support");
}

} // namespace swiftimpute
