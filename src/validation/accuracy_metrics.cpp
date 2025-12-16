#include "validation/accuracy_metrics.hpp"
#include <algorithm>
#include <numeric>
#include <cmath>
#include <fstream>
#include <sstream>
#include <iomanip>
#include <chrono>
#include <unordered_map>
#include <unordered_set>

// Forward declare VCFReader to avoid circular includes
namespace swiftimpute {
    class VCFReader;
}

namespace swiftimpute {
namespace validation {

// MAF bin helpers
MAFBin get_maf_bin(double maf) {
    if (maf < 0.005) return MAFBin::VERY_RARE;
    if (maf < 0.01) return MAFBin::RARE;
    if (maf < 0.05) return MAFBin::LOW_FREQ;
    return MAFBin::COMMON;
}

std::string maf_bin_name(MAFBin bin) {
    switch (bin) {
        case MAFBin::VERY_RARE: return "MAF<0.5%";
        case MAFBin::RARE: return "0.5%<=MAF<1%";
        case MAFBin::LOW_FREQ: return "1%<=MAF<5%";
        case MAFBin::COMMON: return "MAF>=5%";
        case MAFBin::ALL: return "All";
    }
    return "Unknown";
}

// VariantAccuracy implementation
VariantAccuracy::VariantAccuracy()
    : marker_idx(0), pos(0), maf(0), info_score(0), dosage_r2(0),
      genotype_concordance(0), allelic_concordance(0), n_samples(0),
      n_correct(0), n_het_correct(0), n_homalt_correct(0),
      true_hom_ref(0), true_het(0), true_hom_alt(0),
      imputed_hom_ref(0), imputed_het(0), imputed_hom_alt(0) {}

// AggregatedMetrics implementation
AggregatedMetrics::AggregatedMetrics()
    : maf_bin(MAFBin::ALL), n_variants(0), n_genotypes(0),
      mean_info_score(0), mean_dosage_r2(0), mean_concordance(0),
      mean_allelic_r2(0), weighted_dosage_r2(0), weighted_concordance(0),
      r2_q25(0), r2_median(0), r2_q75(0),
      n_variants_r2_gt_50(0), n_variants_r2_gt_80(0), n_variants_r2_gt_95(0) {}

// ValidationReport implementation
ValidationReport::ValidationReport()
    : n_samples(0), n_variants(0), n_variants_evaluated(0),
      evaluation_time_seconds(0) {}

void ValidationReport::print_summary(std::ostream& os) const {
    os << "\n";
    os << "========================================\n";
    os << "       IMPUTATION ACCURACY REPORT      \n";
    os << "========================================\n\n";

    os << "Dataset Information:\n";
    os << "  Samples evaluated:  " << n_samples << "\n";
    os << "  Variants total:     " << n_variants << "\n";
    os << "  Variants evaluated: " << n_variants_evaluated << "\n";
    os << "  Evaluation time:    " << std::fixed << std::setprecision(2)
       << evaluation_time_seconds << " seconds\n\n";

    os << "Overall Metrics:\n";
    os << "  Mean Dosage R²:     " << std::fixed << std::setprecision(4)
       << overall.mean_dosage_r2 << "\n";
    os << "  Mean Concordance:   " << std::fixed << std::setprecision(4)
       << overall.mean_concordance << "\n";
    os << "  Mean INFO Score:    " << std::fixed << std::setprecision(4)
       << overall.mean_info_score << "\n";
    os << "  Weighted R²:        " << std::fixed << std::setprecision(4)
       << overall.weighted_dosage_r2 << "\n\n";

    os << "R² Distribution:\n";
    os << "  25th percentile:    " << std::fixed << std::setprecision(4)
       << overall.r2_q25 << "\n";
    os << "  Median:             " << std::fixed << std::setprecision(4)
       << overall.r2_median << "\n";
    os << "  75th percentile:    " << std::fixed << std::setprecision(4)
       << overall.r2_q75 << "\n\n";

    os << "R² Thresholds:\n";
    os << "  Variants R² > 0.50: " << overall.n_variants_r2_gt_50
       << " (" << std::fixed << std::setprecision(1)
       << (100.0 * overall.n_variants_r2_gt_50 / std::max(1u, overall.n_variants)) << "%)\n";
    os << "  Variants R² > 0.80: " << overall.n_variants_r2_gt_80
       << " (" << std::fixed << std::setprecision(1)
       << (100.0 * overall.n_variants_r2_gt_80 / std::max(1u, overall.n_variants)) << "%)\n";
    os << "  Variants R² > 0.95: " << overall.n_variants_r2_gt_95
       << " (" << std::fixed << std::setprecision(1)
       << (100.0 * overall.n_variants_r2_gt_95 / std::max(1u, overall.n_variants)) << "%)\n\n";

    os << "MAF-Stratified Results:\n";
    os << std::string(72, '-') << "\n";
    os << std::setw(16) << "MAF Bin"
       << std::setw(10) << "N Var"
       << std::setw(12) << "Mean R²"
       << std::setw(12) << "Median R²"
       << std::setw(12) << "Concordance"
       << std::setw(10) << "INFO" << "\n";
    os << std::string(72, '-') << "\n";

    for (const auto& [bin, metrics] : by_maf) {
        os << std::setw(16) << maf_bin_name(bin)
           << std::setw(10) << metrics.n_variants
           << std::setw(12) << std::fixed << std::setprecision(4) << metrics.mean_dosage_r2
           << std::setw(12) << std::fixed << std::setprecision(4) << metrics.r2_median
           << std::setw(12) << std::fixed << std::setprecision(4) << metrics.mean_concordance
           << std::setw(10) << std::fixed << std::setprecision(4) << metrics.mean_info_score
           << "\n";
    }
    os << std::string(72, '-') << "\n";
    os << "\n";
}

void ValidationReport::write_csv(const std::string& filename) const {
    std::ofstream out(filename);
    if (!out.is_open()) {
        throw ImputationError("Failed to open output file: " + filename);
    }

    // Write header
    out << "variant_id,chrom,pos,maf,maf_bin,info_score,dosage_r2,"
        << "concordance,allelic_concordance,n_samples,n_correct\n";

    // Write per-variant data
    for (const auto& v : variant_details) {
        out << v.variant_id << ","
            << v.chrom << ","
            << v.pos << ","
            << std::fixed << std::setprecision(6) << v.maf << ","
            << maf_bin_name(get_maf_bin(v.maf)) << ","
            << std::fixed << std::setprecision(4) << v.info_score << ","
            << std::fixed << std::setprecision(4) << v.dosage_r2 << ","
            << std::fixed << std::setprecision(4) << v.genotype_concordance << ","
            << std::fixed << std::setprecision(4) << v.allelic_concordance << ","
            << v.n_samples << ","
            << v.n_correct << "\n";
    }

    out.close();
}

void ValidationReport::write_json(const std::string& filename) const {
    std::ofstream out(filename);
    if (!out.is_open()) {
        throw ImputationError("Failed to open output file: " + filename);
    }

    out << "{\n";
    out << "  \"summary\": {\n";
    out << "    \"n_samples\": " << n_samples << ",\n";
    out << "    \"n_variants\": " << n_variants << ",\n";
    out << "    \"n_variants_evaluated\": " << n_variants_evaluated << ",\n";
    out << "    \"evaluation_time_seconds\": " << evaluation_time_seconds << "\n";
    out << "  },\n";

    out << "  \"overall\": {\n";
    out << "    \"mean_dosage_r2\": " << overall.mean_dosage_r2 << ",\n";
    out << "    \"mean_concordance\": " << overall.mean_concordance << ",\n";
    out << "    \"mean_info_score\": " << overall.mean_info_score << ",\n";
    out << "    \"weighted_dosage_r2\": " << overall.weighted_dosage_r2 << ",\n";
    out << "    \"r2_q25\": " << overall.r2_q25 << ",\n";
    out << "    \"r2_median\": " << overall.r2_median << ",\n";
    out << "    \"r2_q75\": " << overall.r2_q75 << ",\n";
    out << "    \"n_variants_r2_gt_50\": " << overall.n_variants_r2_gt_50 << ",\n";
    out << "    \"n_variants_r2_gt_80\": " << overall.n_variants_r2_gt_80 << ",\n";
    out << "    \"n_variants_r2_gt_95\": " << overall.n_variants_r2_gt_95 << "\n";
    out << "  },\n";

    out << "  \"by_maf\": {\n";
    bool first = true;
    for (const auto& [bin, metrics] : by_maf) {
        if (!first) out << ",\n";
        first = false;
        out << "    \"" << maf_bin_name(bin) << "\": {\n";
        out << "      \"n_variants\": " << metrics.n_variants << ",\n";
        out << "      \"mean_dosage_r2\": " << metrics.mean_dosage_r2 << ",\n";
        out << "      \"mean_concordance\": " << metrics.mean_concordance << ",\n";
        out << "      \"r2_median\": " << metrics.r2_median << "\n";
        out << "    }";
    }
    out << "\n  }\n";
    out << "}\n";

    out.close();
}

// Helper: parse VCF genotype string to diploid dosage
namespace {

struct ParsedGenotype {
    int8_t allele1 = -1;
    int8_t allele2 = -1;
    float dosage = -1.0f;
    float gp[3] = {-1.0f, -1.0f, -1.0f};
};

ParsedGenotype parse_vcf_genotype(const std::string& gt_field, const std::string& format,
                                   const std::vector<std::string>& format_parts) {
    ParsedGenotype result;

    // Find GT index in format
    int gt_idx = -1, ds_idx = -1, gp_idx = -1;
    for (size_t i = 0; i < format_parts.size(); ++i) {
        if (format_parts[i] == "GT") gt_idx = static_cast<int>(i);
        else if (format_parts[i] == "DS") ds_idx = static_cast<int>(i);
        else if (format_parts[i] == "GP") gp_idx = static_cast<int>(i);
    }

    // Split genotype field by ':'
    std::vector<std::string> parts;
    std::istringstream iss(gt_field);
    std::string part;
    while (std::getline(iss, part, ':')) {
        parts.push_back(part);
    }

    // Parse GT
    if (gt_idx >= 0 && gt_idx < static_cast<int>(parts.size())) {
        const std::string& gt = parts[gt_idx];
        if (gt != "." && gt != "./." && gt != ".|.") {
            char sep = (gt.find('|') != std::string::npos) ? '|' : '/';
            size_t sep_pos = gt.find(sep);
            if (sep_pos != std::string::npos) {
                std::string a1 = gt.substr(0, sep_pos);
                std::string a2 = gt.substr(sep_pos + 1);
                if (a1 != "." && !a1.empty()) result.allele1 = static_cast<int8_t>(std::stoi(a1));
                if (a2 != "." && !a2.empty()) result.allele2 = static_cast<int8_t>(std::stoi(a2));
            }
        }
    }

    // Parse DS
    if (ds_idx >= 0 && ds_idx < static_cast<int>(parts.size())) {
        const std::string& ds = parts[ds_idx];
        if (ds != "." && !ds.empty()) {
            result.dosage = std::stof(ds);
        }
    }

    // Parse GP
    if (gp_idx >= 0 && gp_idx < static_cast<int>(parts.size())) {
        const std::string& gp = parts[gp_idx];
        if (gp != "." && !gp.empty()) {
            std::istringstream gp_iss(gp);
            std::string val;
            int idx = 0;
            while (std::getline(gp_iss, val, ',') && idx < 3) {
                if (!val.empty() && val != ".") {
                    result.gp[idx] = std::stof(val);
                }
                idx++;
            }
        }
    }

    return result;
}

// Simple VCF line parser
bool parse_vcf_line(const std::string& line, std::string& chrom, uint64_t& pos,
                    std::string& id, std::string& ref, std::string& alt,
                    std::string& format, std::vector<std::string>& sample_fields) {
    std::istringstream iss(line);
    std::string qual, filter, info;

    if (!(iss >> chrom >> pos >> id >> ref >> alt >> qual >> filter >> info >> format)) {
        return false;
    }

    sample_fields.clear();
    std::string field;
    while (iss >> field) {
        sample_fields.push_back(field);
    }

    return true;
}

std::vector<std::string> split_format(const std::string& format) {
    std::vector<std::string> parts;
    std::istringstream iss(format);
    std::string part;
    while (std::getline(iss, part, ':')) {
        parts.push_back(part);
    }
    return parts;
}

} // anonymous namespace

// TruthData implementation
TruthData::TruthData(
    std::vector<Marker> markers,
    std::vector<Sample> samples,
    std::unique_ptr<uint8_t[]> genotypes
) : markers_(std::move(markers)),
    samples_(std::move(samples)),
    genotypes_(std::move(genotypes)) {}

std::unique_ptr<TruthData> TruthData::load_vcf(
    const std::string& filename,
    const std::string& region
) {
    std::ifstream file(filename);
    if (!file.is_open()) {
        throw ImputationError("Failed to open VCF file: " + filename);
    }

    std::vector<Marker> markers;
    std::vector<Sample> samples;
    std::vector<std::vector<uint8_t>> genotype_data;

    std::string line;
    while (std::getline(file, line)) {
        if (line.empty()) continue;

        if (line[0] == '#') {
            if (line[1] != '#') {
                // Column header - parse sample names
                std::istringstream iss(line);
                std::string col;
                int col_idx = 0;
                while (iss >> col) {
                    if (col_idx >= 9) {
                        Sample s;
                        s.id = col;
                        s.index = static_cast<uint32_t>(samples.size());
                        samples.push_back(s);
                    }
                    col_idx++;
                }
            }
            continue;
        }

        // Parse variant line
        std::string chrom, id, ref, alt, format;
        uint64_t pos;
        std::vector<std::string> sample_fields;

        if (!parse_vcf_line(line, chrom, pos, id, ref, alt, format, sample_fields)) {
            continue;
        }

        Marker m;
        m.chrom = chrom;
        m.pos = pos;
        m.id = id;
        m.ref = ref;
        m.alt = alt;
        markers.push_back(m);

        // Parse genotypes
        auto format_parts = split_format(format);
        std::vector<uint8_t> gts(samples.size());

        for (size_t i = 0; i < samples.size() && i < sample_fields.size(); ++i) {
            auto pg = parse_vcf_genotype(sample_fields[i], format, format_parts);
            if (pg.allele1 < 0 || pg.allele2 < 0) {
                gts[i] = ALLELE_MISSING;
            } else {
                gts[i] = static_cast<uint8_t>(pg.allele1 + pg.allele2);
            }
        }
        genotype_data.push_back(std::move(gts));
    }

    // Allocate and fill genotype array [samples][markers]
    size_t n_samples = samples.size();
    size_t n_markers = markers.size();
    auto genotypes = std::make_unique<uint8_t[]>(n_samples * n_markers);

    for (size_t m = 0; m < n_markers; ++m) {
        for (size_t s = 0; s < n_samples; ++s) {
            genotypes[s * n_markers + m] = genotype_data[m][s];
        }
    }

    return std::unique_ptr<TruthData>(new TruthData(
        std::move(markers),
        std::move(samples),
        std::move(genotypes)
    ));
}

uint8_t TruthData::get_genotype(sample_t s, marker_t m) const {
    return genotypes_[s * markers_.size() + m];
}

double TruthData::get_dosage(sample_t s, marker_t m) const {
    uint8_t gt = get_genotype(s, m);
    if (gt == ALLELE_MISSING) return -1.0;
    return static_cast<double>(gt);
}

bool TruthData::is_missing(sample_t s, marker_t m) const {
    return get_genotype(s, m) == ALLELE_MISSING;
}

double TruthData::calculate_maf(marker_t m) const {
    uint32_t allele_count = 0;
    uint32_t total_alleles = 0;

    for (sample_t s = 0; s < samples_.size(); ++s) {
        uint8_t gt = get_genotype(s, m);
        if (gt != ALLELE_MISSING) {
            allele_count += gt;
            total_alleles += 2;
        }
    }

    if (total_alleles == 0) return 0.0;

    double af = static_cast<double>(allele_count) / total_alleles;
    return std::min(af, 1.0 - af);  // Return minor allele frequency
}

// ImputedData implementation
ImputedData::ImputedData(
    std::vector<Marker> markers,
    std::vector<Sample> samples,
    std::unique_ptr<uint8_t[]> genotypes,
    std::unique_ptr<float[]> dosages,
    std::unique_ptr<float[]> probabilities,
    bool has_dosages,
    bool has_probabilities
) : markers_(std::move(markers)),
    samples_(std::move(samples)),
    genotypes_(std::move(genotypes)),
    dosages_(std::move(dosages)),
    probabilities_(std::move(probabilities)),
    has_dosages_(has_dosages),
    has_probabilities_(has_probabilities) {}

std::unique_ptr<ImputedData> ImputedData::load_vcf(
    const std::string& filename,
    const std::string& region
) {
    std::ifstream file(filename);
    if (!file.is_open()) {
        throw ImputationError("Failed to open VCF file: " + filename);
    }

    std::vector<Marker> markers;
    std::vector<Sample> samples;

    // Temporary storage
    std::vector<std::vector<uint8_t>> genotype_data;
    std::vector<std::vector<float>> dosage_data;
    std::vector<std::vector<std::array<float, 3>>> prob_data;

    bool has_ds = false;
    bool has_gp = false;

    std::string line;
    while (std::getline(file, line)) {
        if (line.empty()) continue;

        if (line[0] == '#') {
            if (line[1] != '#') {
                // Column header - parse sample names
                std::istringstream iss(line);
                std::string col;
                int col_idx = 0;
                while (iss >> col) {
                    if (col_idx >= 9) {
                        Sample s;
                        s.id = col;
                        s.index = static_cast<uint32_t>(samples.size());
                        samples.push_back(s);
                    }
                    col_idx++;
                }
            }
            continue;
        }

        // Parse variant line
        std::string chrom, id, ref, alt, format;
        uint64_t pos;
        std::vector<std::string> sample_fields;

        if (!parse_vcf_line(line, chrom, pos, id, ref, alt, format, sample_fields)) {
            continue;
        }

        Marker m;
        m.chrom = chrom;
        m.pos = pos;
        m.id = id;
        m.ref = ref;
        m.alt = alt;
        markers.push_back(m);

        auto format_parts = split_format(format);
        std::vector<uint8_t> gts(samples.size());
        std::vector<float> ds(samples.size(), -1.0f);
        std::vector<std::array<float, 3>> gp(samples.size());

        for (size_t i = 0; i < samples.size() && i < sample_fields.size(); ++i) {
            auto pg = parse_vcf_genotype(sample_fields[i], format, format_parts);

            // Genotype
            if (pg.allele1 < 0 || pg.allele2 < 0) {
                gts[i] = ALLELE_MISSING;
            } else {
                gts[i] = static_cast<uint8_t>(pg.allele1 + pg.allele2);
            }

            // Dosage
            if (pg.dosage >= 0) {
                ds[i] = pg.dosage;
                has_ds = true;
            } else if (gts[i] != ALLELE_MISSING) {
                ds[i] = static_cast<float>(gts[i]);
            }

            // Genotype probabilities
            if (pg.gp[0] >= 0 && pg.gp[1] >= 0 && pg.gp[2] >= 0) {
                gp[i] = {pg.gp[0], pg.gp[1], pg.gp[2]};
                has_gp = true;
            } else {
                gp[i] = {-1.0f, -1.0f, -1.0f};
            }
        }

        genotype_data.push_back(std::move(gts));
        dosage_data.push_back(std::move(ds));
        prob_data.push_back(std::move(gp));
    }

    // Allocate and fill arrays
    size_t n_samples = samples.size();
    size_t n_markers = markers.size();

    auto genotypes = std::make_unique<uint8_t[]>(n_samples * n_markers);
    auto dosages = std::make_unique<float[]>(n_samples * n_markers);
    auto probabilities = std::make_unique<float[]>(n_samples * n_markers * 3);

    for (size_t m = 0; m < n_markers; ++m) {
        for (size_t s = 0; s < n_samples; ++s) {
            size_t idx = s * n_markers + m;
            genotypes[idx] = genotype_data[m][s];
            dosages[idx] = dosage_data[m][s];

            size_t prob_idx = idx * 3;
            probabilities[prob_idx] = prob_data[m][s][0];
            probabilities[prob_idx + 1] = prob_data[m][s][1];
            probabilities[prob_idx + 2] = prob_data[m][s][2];
        }
    }

    return std::unique_ptr<ImputedData>(new ImputedData(
        std::move(markers),
        std::move(samples),
        std::move(genotypes),
        std::move(dosages),
        std::move(probabilities),
        has_ds,
        has_gp
    ));
}

uint8_t ImputedData::get_genotype(sample_t s, marker_t m) const {
    return genotypes_[s * markers_.size() + m];
}

double ImputedData::get_dosage(sample_t s, marker_t m) const {
    return static_cast<double>(dosages_[s * markers_.size() + m]);
}

void ImputedData::get_probabilities(sample_t s, marker_t m, double probs[3]) const {
    size_t idx = (s * markers_.size() + m) * 3;
    probs[0] = static_cast<double>(probabilities_[idx]);
    probs[1] = static_cast<double>(probabilities_[idx + 1]);
    probs[2] = static_cast<double>(probabilities_[idx + 2]);
}

// AccuracyCalculator implementation
AccuracyCalculator::Config::Config()
    : compute_per_variant(true),
      compute_confusion_matrix(false),
      maf_bins({MAFBin::VERY_RARE, MAFBin::RARE, MAFBin::LOW_FREQ, MAFBin::COMMON}),
      min_maf(0.0),
      max_maf(0.5) {}

AccuracyCalculator::AccuracyCalculator(const Config& config)
    : config_(config) {}

ValidationReport AccuracyCalculator::evaluate(
    const TruthData& truth,
    const ImputedData& imputed
) {
    auto start_time = std::chrono::high_resolution_clock::now();

    ValidationReport report;
    report.n_samples = truth.num_samples();
    report.n_variants = truth.num_markers();

    // Build sample index mapping (handle mismatched sample orders)
    std::unordered_map<std::string, sample_t> truth_sample_idx;
    for (sample_t s = 0; s < truth.num_samples(); ++s) {
        truth_sample_idx[truth.samples()[s].id] = s;
    }

    std::vector<std::pair<sample_t, sample_t>> sample_mapping;  // (truth_idx, imputed_idx)
    for (sample_t s = 0; s < imputed.num_samples(); ++s) {
        auto it = truth_sample_idx.find(imputed.samples()[s].id);
        if (it != truth_sample_idx.end()) {
            sample_mapping.emplace_back(it->second, s);
        }
    }

    if (sample_mapping.empty()) {
        throw ImputationError("No matching samples between truth and imputed data");
    }

    // Build variant index mapping
    std::unordered_map<std::string, marker_t> truth_variant_idx;
    for (marker_t m = 0; m < truth.num_markers(); ++m) {
        const auto& marker = truth.markers()[m];
        std::string key = marker.chrom + ":" + std::to_string(marker.pos);
        truth_variant_idx[key] = m;
    }

    // Calculate per-variant metrics
    std::vector<VariantAccuracy> all_variants;
    std::map<MAFBin, std::vector<VariantAccuracy>> variants_by_maf;

    for (marker_t imp_m = 0; imp_m < imputed.num_markers(); ++imp_m) {
        const auto& imp_marker = imputed.markers()[imp_m];
        std::string key = imp_marker.chrom + ":" + std::to_string(imp_marker.pos);

        auto it = truth_variant_idx.find(key);
        if (it == truth_variant_idx.end()) continue;

        marker_t truth_m = it->second;

        // Calculate MAF from truth
        double maf = truth.calculate_maf(truth_m);
        if (maf < config_.min_maf || maf > config_.max_maf) continue;

        // Calculate variant accuracy
        VariantAccuracy va;
        va.marker_idx = imp_m;
        va.variant_id = imp_marker.id;
        va.chrom = imp_marker.chrom;
        va.pos = imp_marker.pos;
        va.maf = maf;

        std::vector<double> truth_dosages;
        std::vector<double> imputed_dosages;
        std::vector<std::array<double, 3>> imputed_probs;

        uint32_t n_correct = 0;
        uint32_t n_het_correct = 0;
        uint32_t n_homalt_correct = 0;
        uint32_t alleles_correct = 0;
        uint32_t total_alleles = 0;

        for (const auto& [truth_s, imp_s] : sample_mapping) {
            double truth_d = truth.get_dosage(truth_s, truth_m);
            double imp_d = imputed.get_dosage(imp_s, imp_m);

            if (truth_d < 0 || imp_d < 0) continue;  // Skip missing

            truth_dosages.push_back(truth_d);
            imputed_dosages.push_back(imp_d);

            if (imputed.has_probabilities()) {
                double probs[3];
                imputed.get_probabilities(imp_s, imp_m, probs);
                imputed_probs.push_back({probs[0], probs[1], probs[2]});
            }

            uint8_t truth_gt = truth.get_genotype(truth_s, truth_m);
            uint8_t imp_gt = imputed.get_genotype(imp_s, imp_m);

            // Count truth genotype types
            if (truth_gt == 0) va.true_hom_ref++;
            else if (truth_gt == 1) va.true_het++;
            else if (truth_gt == 2) va.true_hom_alt++;

            // Count imputed genotype types
            if (imp_gt == 0) va.imputed_hom_ref++;
            else if (imp_gt == 1) va.imputed_het++;
            else if (imp_gt == 2) va.imputed_hom_alt++;

            // Concordance
            if (truth_gt == imp_gt) {
                n_correct++;
                if (truth_gt == 1) n_het_correct++;
                if (truth_gt == 2) n_homalt_correct++;
            }

            // Allelic concordance (count matching alleles)
            int truth_a1 = (truth_gt > 0) ? 1 : 0;
            int truth_a2 = (truth_gt > 1) ? 1 : 0;
            int imp_a1 = (imp_gt > 0) ? 1 : 0;
            int imp_a2 = (imp_gt > 1) ? 1 : 0;

            if (truth_a1 == imp_a1) alleles_correct++;
            if (truth_a2 == imp_a2) alleles_correct++;
            total_alleles += 2;
        }

        va.n_samples = static_cast<uint32_t>(truth_dosages.size());
        va.n_correct = n_correct;
        va.n_het_correct = n_het_correct;
        va.n_homalt_correct = n_homalt_correct;

        if (va.n_samples >= 10) {  // Minimum samples for meaningful metrics
            va.genotype_concordance = static_cast<double>(n_correct) / va.n_samples;
            va.allelic_concordance = static_cast<double>(alleles_correct) / total_alleles;
            va.dosage_r2 = utils::r_squared(truth_dosages, imputed_dosages);

            if (!imputed_probs.empty()) {
                va.info_score = utils::info_score(imputed_dosages, imputed_probs);
            } else {
                va.info_score = va.dosage_r2;  // Approximate
            }

            all_variants.push_back(va);
            variants_by_maf[get_maf_bin(maf)].push_back(va);
        }
    }

    report.n_variants_evaluated = static_cast<uint32_t>(all_variants.size());

    // Aggregate overall metrics
    report.overall = aggregate_metrics(all_variants, MAFBin::ALL);

    // Aggregate by MAF bin
    for (const auto& bin : config_.maf_bins) {
        if (variants_by_maf.count(bin)) {
            report.by_maf[bin] = aggregate_metrics(variants_by_maf[bin], bin);
        } else {
            AggregatedMetrics empty;
            empty.maf_bin = bin;
            report.by_maf[bin] = empty;
        }
    }

    // Store variant details if requested
    if (config_.compute_per_variant) {
        report.variant_details = std::move(all_variants);
    }

    auto end_time = std::chrono::high_resolution_clock::now();
    report.evaluation_time_seconds =
        std::chrono::duration<double>(end_time - start_time).count();

    return report;
}

ValidationReport AccuracyCalculator::evaluate_subset(
    const TruthData& truth,
    const ImputedData& imputed,
    const std::vector<sample_t>& sample_indices
) {
    // TODO: Implement subset evaluation
    return evaluate(truth, imputed);
}

std::pair<double, double> AccuracyCalculator::quick_evaluate(
    const TruthData& truth,
    const ImputedData& imputed
) {
    Config quick_config;
    quick_config.compute_per_variant = false;

    AccuracyCalculator calc(quick_config);
    auto report = calc.evaluate(truth, imputed);

    return {report.overall.mean_dosage_r2, report.overall.mean_concordance};
}

AggregatedMetrics AccuracyCalculator::aggregate_metrics(
    const std::vector<VariantAccuracy>& variants,
    MAFBin bin
) {
    AggregatedMetrics agg;
    agg.maf_bin = bin;
    agg.n_variants = static_cast<uint32_t>(variants.size());

    if (variants.empty()) return agg;

    double sum_info = 0, sum_r2 = 0, sum_conc = 0;
    double weighted_r2 = 0, weighted_conc = 0;
    uint64_t total_samples = 0;
    std::vector<double> r2_values;

    for (const auto& v : variants) {
        sum_info += v.info_score;
        sum_r2 += v.dosage_r2;
        sum_conc += v.genotype_concordance;

        weighted_r2 += v.dosage_r2 * v.n_samples;
        weighted_conc += v.genotype_concordance * v.n_samples;
        total_samples += v.n_samples;

        r2_values.push_back(v.dosage_r2);

        if (v.dosage_r2 > 0.50) agg.n_variants_r2_gt_50++;
        if (v.dosage_r2 > 0.80) agg.n_variants_r2_gt_80++;
        if (v.dosage_r2 > 0.95) agg.n_variants_r2_gt_95++;
    }

    agg.n_genotypes = static_cast<uint32_t>(total_samples);
    agg.mean_info_score = sum_info / variants.size();
    agg.mean_dosage_r2 = sum_r2 / variants.size();
    agg.mean_concordance = sum_conc / variants.size();

    if (total_samples > 0) {
        agg.weighted_dosage_r2 = weighted_r2 / total_samples;
        agg.weighted_concordance = weighted_conc / total_samples;
    }

    // Calculate quantiles
    agg.r2_q25 = utils::quantile(r2_values, 0.25);
    agg.r2_median = utils::quantile(r2_values, 0.50);
    agg.r2_q75 = utils::quantile(r2_values, 0.75);

    return agg;
}

// Utility functions
namespace utils {

double pearson_correlation(
    const std::vector<double>& x,
    const std::vector<double>& y
) {
    if (x.size() != y.size() || x.size() < 2) return 0.0;

    size_t n = x.size();
    double sum_x = 0, sum_y = 0, sum_xy = 0;
    double sum_x2 = 0, sum_y2 = 0;

    for (size_t i = 0; i < n; ++i) {
        sum_x += x[i];
        sum_y += y[i];
        sum_xy += x[i] * y[i];
        sum_x2 += x[i] * x[i];
        sum_y2 += y[i] * y[i];
    }

    double numerator = n * sum_xy - sum_x * sum_y;
    double denominator = std::sqrt((n * sum_x2 - sum_x * sum_x) *
                                   (n * sum_y2 - sum_y * sum_y));

    if (denominator < 1e-10) return 0.0;

    return numerator / denominator;
}

double r_squared(
    const std::vector<double>& truth,
    const std::vector<double>& imputed
) {
    double r = pearson_correlation(truth, imputed);
    return r * r;
}

double info_score(
    const std::vector<double>& dosages,
    const std::vector<std::array<double, 3>>& probabilities
) {
    if (dosages.empty()) return 0.0;

    // Calculate allele frequency from dosages
    double p = allele_frequency(dosages);
    if (p < 1e-10 || p > 1.0 - 1e-10) return 1.0;  // Monomorphic

    size_t n = dosages.size();
    double expected_var = 2.0 * p * (1.0 - p);

    // Calculate variance of posterior mean
    double sum_var = 0.0;
    for (size_t i = 0; i < n; ++i) {
        if (probabilities[i][0] < 0) continue;  // Skip missing

        double e_g = probabilities[i][1] + 2.0 * probabilities[i][2];
        double e_g2 = probabilities[i][1] + 4.0 * probabilities[i][2];
        double var_i = e_g2 - e_g * e_g;
        sum_var += var_i;
    }

    double avg_var = sum_var / n;

    // INFO = 1 - avg_var / expected_var
    double info = 1.0 - avg_var / expected_var;
    return std::max(0.0, std::min(1.0, info));
}

double allele_frequency(const std::vector<double>& dosages) {
    if (dosages.empty()) return 0.0;

    double sum = 0.0;
    size_t count = 0;

    for (double d : dosages) {
        if (d >= 0) {
            sum += d;
            count++;
        }
    }

    if (count == 0) return 0.0;
    return sum / (2.0 * count);
}

std::pair<double, double> mean_variance(const std::vector<double>& values) {
    if (values.empty()) return {0.0, 0.0};

    double sum = std::accumulate(values.begin(), values.end(), 0.0);
    double mean = sum / values.size();

    double sq_sum = 0.0;
    for (double v : values) {
        sq_sum += (v - mean) * (v - mean);
    }

    double variance = (values.size() > 1) ? sq_sum / (values.size() - 1) : 0.0;
    return {mean, variance};
}

double quantile(std::vector<double> values, double q) {
    if (values.empty()) return 0.0;

    std::sort(values.begin(), values.end());

    double idx = q * (values.size() - 1);
    size_t lo = static_cast<size_t>(idx);
    size_t hi = std::min(lo + 1, values.size() - 1);
    double frac = idx - lo;

    return values[lo] * (1.0 - frac) + values[hi] * frac;
}

} // namespace utils
} // namespace validation
} // namespace swiftimpute
