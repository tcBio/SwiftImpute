#pragma once

#include "../core/types.hpp"
#include <string>
#include <vector>
#include <fstream>
#include <sstream>
#include <iomanip>

namespace swiftimpute {

/**
 * @brief VCF file writer for outputting imputed genotypes
 *
 * Writes VCF 4.2 format with dosage (DS) and genotype probability (GP) fields
 */
class VCFWriter {
public:
    VCFWriter() = default;
    explicit VCFWriter(const std::string& filename);
    ~VCFWriter();

    // Open/close file
    void open(const std::string& filename);
    void close();
    bool is_open() const { return file_.is_open(); }

    // Write header
    void write_header(
        const std::vector<std::string>& sample_names,
        const std::vector<std::string>& contigs = {}
    );

    // Write variant with imputed genotypes
    void write_variant(
        const std::string& chrom,
        uint64_t position,
        const std::string& id,
        const std::string& ref,
        const std::vector<std::string>& alt,
        const std::vector<std::vector<prob_t>>& genotype_probs,  // [sample][genotype]
        const std::vector<prob_t>& dosages = {}  // [sample] - optional dosages
    );

    // Write variant with phased haplotypes
    void write_phased_variant(
        const std::string& chrom,
        uint64_t position,
        const std::string& id,
        const std::string& ref,
        const std::vector<std::string>& alt,
        const std::vector<std::vector<allele_t>>& phased_genotypes  // [sample][haplotype]
    );

private:
    std::ofstream file_;
    std::string filename_;
    bool header_written_ = false;

    std::string format_genotype_probs(const std::vector<prob_t>& probs);
    std::string format_dosage(prob_t dosage);
};

// Implementation

inline VCFWriter::VCFWriter(const std::string& filename) {
    open(filename);
}

inline VCFWriter::~VCFWriter() {
    close();
}

inline void VCFWriter::open(const std::string& filename) {
    filename_ = filename;

    if (filename.size() >= 3 && filename.substr(filename.size() - 3) == ".gz") {
        throw std::runtime_error(
            "Gzipped VCF writing requires htslib support. "
            "Please use uncompressed .vcf files or build with ENABLE_HTSLIB=ON"
        );
    }

    file_.open(filename);
    if (!file_.is_open()) {
        throw std::runtime_error("Failed to open VCF file for writing: " + filename);
    }

    // Set precision for floating point output
    file_ << std::fixed << std::setprecision(4);
}

inline void VCFWriter::close() {
    if (file_.is_open()) {
        file_.close();
    }
}

inline void VCFWriter::write_header(
    const std::vector<std::string>& sample_names,
    const std::vector<std::string>& contigs
) {
    // VCF version
    file_ << "##fileformat=VCFv4.2\n";

    // Source
    file_ << "##source=SwiftImpute\n";

    // Contigs
    for (const auto& contig : contigs) {
        file_ << "##contig=<ID=" << contig << ">\n";
    }

    // INFO fields
    file_ << "##INFO=<ID=IMP,Number=0,Type=Flag,Description=\"Imputed variant\">\n";

    // FORMAT fields
    file_ << "##FORMAT=<ID=GT,Number=1,Type=String,Description=\"Genotype\">\n";
    file_ << "##FORMAT=<ID=DS,Number=1,Type=Float,Description=\"Dosage (expected ALT allele count)\">\n";
    file_ << "##FORMAT=<ID=GP,Number=G,Type=Float,Description=\"Genotype posterior probabilities\">\n";
    file_ << "##FORMAT=<ID=AP,Number=2,Type=Float,Description=\"Allelic probabilities\">\n";

    // Column header
    file_ << "#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT";

    for (const auto& sample : sample_names) {
        file_ << "\t" << sample;
    }
    file_ << "\n";

    header_written_ = true;
}

inline void VCFWriter::write_variant(
    const std::string& chrom,
    uint64_t position,
    const std::string& id,
    const std::string& ref,
    const std::vector<std::string>& alt,
    const std::vector<std::vector<prob_t>>& genotype_probs,
    const std::vector<prob_t>& dosages
) {
    if (!header_written_) {
        throw std::runtime_error("Must write header before writing variants");
    }

    // Fixed fields
    file_ << chrom << "\t"
          << position << "\t"
          << (id.empty() ? "." : id) << "\t"
          << ref << "\t";

    // ALT field
    for (size_t i = 0; i < alt.size(); ++i) {
        if (i > 0) file_ << ",";
        file_ << alt[i];
    }

    // QUAL, FILTER, INFO
    file_ << "\t.\tPASS\tIMP\t";

    // FORMAT
    if (dosages.empty()) {
        file_ << "GT:GP";
    } else {
        file_ << "GT:DS:GP";
    }

    // Sample genotypes
    for (size_t s = 0; s < genotype_probs.size(); ++s) {
        const auto& probs = genotype_probs[s];

        // Best genotype (highest probability)
        size_t best_gt = 0;
        prob_t best_prob = probs[0];
        for (size_t i = 1; i < probs.size(); ++i) {
            if (probs[i] > best_prob) {
                best_prob = probs[i];
                best_gt = i;
            }
        }

        // Convert genotype index to alleles (for biallelic: 0=0/0, 1=0/1, 2=1/1)
        std::string gt_str;
        if (alt.size() == 1) {
            // Biallelic
            switch (best_gt) {
                case 0: gt_str = "0|0"; break;
                case 1: gt_str = "0|1"; break;
                case 2: gt_str = "1|1"; break;
                default: gt_str = "./."; break;
            }
        } else {
            // Multiallelic - simplified to unphased
            gt_str = std::to_string(best_gt) + "/" + std::to_string(best_gt);
        }

        file_ << "\t" << gt_str;

        // Dosage
        if (!dosages.empty()) {
            file_ << ":" << format_dosage(dosages[s]);
        }

        // Genotype probabilities
        file_ << ":" << format_genotype_probs(probs);
    }

    file_ << "\n";
}

inline void VCFWriter::write_phased_variant(
    const std::string& chrom,
    uint64_t position,
    const std::string& id,
    const std::string& ref,
    const std::vector<std::string>& alt,
    const std::vector<std::vector<allele_t>>& phased_genotypes
) {
    if (!header_written_) {
        throw std::runtime_error("Must write header before writing variants");
    }

    // Fixed fields
    file_ << chrom << "\t"
          << position << "\t"
          << (id.empty() ? "." : id) << "\t"
          << ref << "\t";

    // ALT field
    for (size_t i = 0; i < alt.size(); ++i) {
        if (i > 0) file_ << ",";
        file_ << alt[i];
    }

    // QUAL, FILTER, INFO, FORMAT
    file_ << "\t.\tPASS\tIMP\tGT";

    // Sample genotypes
    for (const auto& gt : phased_genotypes) {
        file_ << "\t";
        if (gt.size() == 2) {
            file_ << static_cast<int>(gt[0]) << "|" << static_cast<int>(gt[1]);
        } else {
            file_ << "./.";
        }
    }

    file_ << "\n";
}

inline std::string VCFWriter::format_genotype_probs(const std::vector<prob_t>& probs) {
    std::ostringstream oss;
    oss << std::fixed << std::setprecision(4);

    for (size_t i = 0; i < probs.size(); ++i) {
        if (i > 0) oss << ",";
        oss << probs[i];
    }

    return oss.str();
}

inline std::string VCFWriter::format_dosage(prob_t dosage) {
    std::ostringstream oss;
    oss << std::fixed << std::setprecision(4) << dosage;
    return oss.str();
}

} // namespace swiftimpute
