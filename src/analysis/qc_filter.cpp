#include "qc_filter.hpp"
#include "../io/vcf_reader.hpp"
#include "../io/vcf_writer.hpp"
#include <chrono>
#include <algorithm>
#include <numeric>
#include <fstream>
#include <iomanip>
#include <sstream>
#include <cmath>

namespace swiftimpute {
namespace analysis {

// ============================================================================
// QCFilter Implementation
// ============================================================================

QCFilter::QCFilter(const QCConfig& config)
    : config_(config) {}

bool QCFilter::passes(const VariantQC& variant) const {
    if (config_.filter_by_info && variant.info_score < config_.min_info_score) {
        return false;
    }

    if (config_.filter_by_hwe && variant.hw_pvalue < config_.min_hwe_pvalue) {
        return false;
    }

    if (config_.filter_by_call_rate && variant.call_rate < config_.min_call_rate) {
        return false;
    }

    if (config_.filter_by_maf) {
        if (variant.maf < config_.min_maf || variant.maf > config_.max_maf) {
            return false;
        }
    }

    return true;
}

double QCFilter::calculate_hwe_pvalue(
    size_t n_hom_ref, size_t n_het, size_t n_hom_alt
) const {
    // Hardy-Weinberg exact test (simplified implementation)
    size_t n = n_hom_ref + n_het + n_hom_alt;
    if (n == 0) return 1.0;

    // Calculate expected frequencies
    double p = (2.0 * n_hom_ref + n_het) / (2.0 * n);
    double q = 1.0 - p;

    double exp_hom_ref = p * p * n;
    double exp_het = 2 * p * q * n;
    double exp_hom_alt = q * q * n;

    // Chi-squared test (approximation)
    double chi_sq = 0;
    if (exp_hom_ref > 0) {
        chi_sq += std::pow(n_hom_ref - exp_hom_ref, 2) / exp_hom_ref;
    }
    if (exp_het > 0) {
        chi_sq += std::pow(n_het - exp_het, 2) / exp_het;
    }
    if (exp_hom_alt > 0) {
        chi_sq += std::pow(n_hom_alt - exp_hom_alt, 2) / exp_hom_alt;
    }

    // Approximate p-value (chi-squared with 1 df)
    // Using simple approximation; for production, use proper statistics library
    double p_value = std::exp(-chi_sq / 2);

    return p_value;
}

double QCFilter::calculate_maf(const std::vector<double>& dosages) const {
    if (dosages.empty()) return 0.0;

    double sum = std::accumulate(dosages.begin(), dosages.end(), 0.0);
    double af = sum / (2.0 * dosages.size());

    return std::min(af, 1.0 - af);  // Return minor allele frequency
}

QCSummary QCFilter::compute_qc_metrics(
    const std::string& input_vcf,
    QCProgressCallback progress
) {
    variant_qc_.clear();

    VCFReader reader(input_vcf);
    auto header = reader.read_header();

    VCFReader::Variant variant;
    size_t variant_count = 0;

    while (reader.read_variant(variant)) {
        VariantQC qc;
        qc.chrom = variant.chrom;
        qc.pos = variant.position;
        qc.id = variant.id;
        qc.ref = variant.ref;
        qc.alt = variant.alt.empty() ? "" : variant.alt[0];

        // Extract dosages from genotypes
        std::vector<double> dosages;
        size_t n_hom_ref = 0, n_het = 0, n_hom_alt = 0;
        size_t n_called = 0;

        for (const auto& gt : variant.genotypes) {
            if (gt.size() >= 2 && gt[0] != ALLELE_MISSING && gt[1] != ALLELE_MISSING) {
                int dose = static_cast<int>(gt[0]) + static_cast<int>(gt[1]);
                dosages.push_back(dose);
                n_called++;

                if (dose == 0) n_hom_ref++;
                else if (dose == 1) n_het++;
                else if (dose == 2) n_hom_alt++;
            }
        }

        // Calculate metrics
        qc.call_rate = variant.genotypes.empty() ? 0.0 :
            static_cast<double>(n_called) / variant.genotypes.size();

        qc.maf = calculate_maf(dosages);

        qc.hw_pvalue = calculate_hwe_pvalue(n_hom_ref, n_het, n_hom_alt);

        // INFO score approximation from dosages
        qc.info_score = calculate_info_score_approx(dosages);

        // Apply filters
        qc.passes_info = !config_.filter_by_info ||
            qc.info_score >= config_.min_info_score;
        qc.passes_hwe = !config_.filter_by_hwe ||
            qc.hw_pvalue >= config_.min_hwe_pvalue;
        qc.passes_call_rate = !config_.filter_by_call_rate ||
            qc.call_rate >= config_.min_call_rate;
        qc.passes_maf = !config_.filter_by_maf ||
            (qc.maf >= config_.min_maf && qc.maf <= config_.max_maf);

        qc.passes_all = qc.passes_info && qc.passes_hwe &&
                        qc.passes_call_rate && qc.passes_maf;

        variant_qc_.push_back(qc);
        variant_count++;

        if (progress && variant_count % 10000 == 0) {
            progress(variant_count, variant_count);  // Total unknown
        }
    }

    return aggregate_qc(variant_qc_);
}

QCSummary QCFilter::filter_vcf(
    const std::string& input_vcf,
    const std::string& output_vcf,
    QCProgressCallback progress
) {
    // First compute QC metrics
    auto summary = compute_qc_metrics(input_vcf, progress);

    // Then filter
    VCFReader reader(input_vcf);
    auto header = reader.read_header();

    VCFWriter writer(output_vcf);

    // Get sample names and contigs for header
    std::vector<std::string> sample_names;
    for (const auto& s : header.sample_ids) {
        sample_names.push_back(s);
    }

    // Extract contigs from variants
    std::set<std::string> contigs_set;
    for (const auto& qc : variant_qc_) {
        contigs_set.insert(qc.chrom);
    }
    std::vector<std::string> contigs(contigs_set.begin(), contigs_set.end());

    writer.write_header(sample_names, contigs);

    // Re-read and filter
    reader.close();
    reader.open(input_vcf);
    reader.read_header();

    VCFReader::Variant variant;
    size_t idx = 0;
    size_t written = 0;

    while (reader.read_variant(variant)) {
        if (idx < variant_qc_.size() && variant_qc_[idx].passes_all) {
            // Convert genotypes to allele format for writer
            std::vector<std::vector<allele_t>> phased_gts;
            for (const auto& gt : variant.genotypes) {
                if (gt.size() >= 2) {
                    phased_gts.push_back({gt[0], gt[1]});
                } else {
                    phased_gts.push_back({ALLELE_MISSING, ALLELE_MISSING});
                }
            }

            writer.write_phased_variant(
                variant.chrom,
                variant.position,
                variant.id,
                variant.ref,
                variant.alt,
                phased_gts
            );
            written++;
        }

        idx++;

        if (progress && idx % 10000 == 0) {
            progress(idx, variant_qc_.size());
        }
    }

    writer.close();

    summary.variants_passing = written;
    summary.variants_filtered = summary.total_variants - written;

    return summary;
}

QCSummary QCFilter::aggregate_qc(const std::vector<VariantQC>& variants) const {
    QCSummary summary;
    summary.total_variants = variants.size();

    if (variants.empty()) return summary;

    std::vector<double> info_scores;
    std::map<validation::MAFBin, std::vector<double>> info_by_maf;

    for (const auto& v : variants) {
        info_scores.push_back(v.info_score);

        // Count passes
        if (v.passes_all) {
            summary.variants_passing++;
        }

        // Count filtered by each criterion
        if (!v.passes_info) summary.filtered_by_info++;
        if (!v.passes_hwe) summary.filtered_by_hwe++;
        if (!v.passes_call_rate) summary.filtered_by_call_rate++;
        if (!v.passes_maf) summary.filtered_by_maf++;

        // INFO tiers
        if (v.info_score >= 0.9) {
            summary.info_high++;
        } else if (v.info_score >= 0.8) {
            summary.info_good++;
        } else if (v.info_score >= 0.6) {
            summary.info_acceptable++;
        } else {
            summary.info_low++;
        }

        // MAF-stratified INFO
        validation::MAFBin bin = validation::get_maf_bin(v.maf);
        info_by_maf[bin].push_back(v.info_score);
    }

    summary.variants_filtered = summary.total_variants - summary.variants_passing;

    // Calculate INFO statistics
    if (!info_scores.empty()) {
        summary.mean_info_score = std::accumulate(info_scores.begin(),
            info_scores.end(), 0.0) / info_scores.size();

        std::sort(info_scores.begin(), info_scores.end());
        summary.median_info_score = info_scores[info_scores.size() / 2];
        summary.info_q25 = info_scores[info_scores.size() / 4];
        summary.info_q75 = info_scores[3 * info_scores.size() / 4];
    }

    // MAF-stratified means
    for (const auto& [bin, scores] : info_by_maf) {
        if (!scores.empty()) {
            summary.mean_info_by_maf[bin] =
                std::accumulate(scores.begin(), scores.end(), 0.0) / scores.size();
        }
    }

    return summary;
}

void QCFilter::export_variant_qc(const std::string& filename) const {
    std::ofstream out(filename);
    if (!out) {
        throw std::runtime_error("Failed to open file: " + filename);
    }

    // Header
    out << "CHROM\tPOS\tID\tREF\tALT\tINFO_SCORE\tMAF\tHWE_P\tCALL_RATE\tPASS\n";

    // Data
    for (const auto& qc : variant_qc_) {
        out << qc.chrom << "\t"
            << qc.pos << "\t"
            << (qc.id.empty() ? "." : qc.id) << "\t"
            << qc.ref << "\t"
            << qc.alt << "\t"
            << std::fixed << std::setprecision(4) << qc.info_score << "\t"
            << std::fixed << std::setprecision(6) << qc.maf << "\t"
            << std::scientific << std::setprecision(2) << qc.hw_pvalue << "\t"
            << std::fixed << std::setprecision(4) << qc.call_rate << "\t"
            << (qc.passes_all ? "PASS" : "FAIL") << "\n";
    }
}

void QCFilter::print_summary(const QCSummary& summary, std::ostream& os) const {
    os << "\n";
    os << "================================================================================\n";
    os << "                         QC FILTERING SUMMARY\n";
    os << "================================================================================\n\n";

    os << "Variant Counts:\n";
    os << "  Total variants:    " << std::setw(12) << summary.total_variants << "\n";
    os << "  Passing QC:        " << std::setw(12) << summary.variants_passing
       << " (" << std::fixed << std::setprecision(1)
       << (100.0 * summary.variants_passing / summary.total_variants) << "%)\n";
    os << "  Filtered:          " << std::setw(12) << summary.variants_filtered << "\n\n";

    os << "Filtered by criterion:\n";
    if (config_.filter_by_info) {
        os << "  INFO < " << config_.min_info_score << ":       "
           << std::setw(12) << summary.filtered_by_info << "\n";
    }
    if (config_.filter_by_hwe) {
        os << "  HWE p < " << std::scientific << config_.min_hwe_pvalue << ": "
           << std::setw(12) << std::fixed << summary.filtered_by_hwe << "\n";
    }
    if (config_.filter_by_call_rate) {
        os << "  Call rate < " << config_.min_call_rate << ": "
           << std::setw(12) << summary.filtered_by_call_rate << "\n";
    }
    if (config_.filter_by_maf) {
        os << "  MAF filter:        " << std::setw(12) << summary.filtered_by_maf << "\n";
    }
    os << "\n";

    os << "INFO Score Distribution:\n";
    os << "  Mean:    " << std::fixed << std::setprecision(4) << summary.mean_info_score << "\n";
    os << "  Median:  " << summary.median_info_score << "\n";
    os << "  Q25:     " << summary.info_q25 << "\n";
    os << "  Q75:     " << summary.info_q75 << "\n\n";

    os << "INFO Score Tiers:\n";
    os << "  High (>=0.9):      " << std::setw(12) << summary.info_high
       << " (" << std::fixed << std::setprecision(1)
       << (100.0 * summary.info_high / summary.total_variants) << "%)\n";
    os << "  Good (0.8-0.9):    " << std::setw(12) << summary.info_good
       << " (" << (100.0 * summary.info_good / summary.total_variants) << "%)\n";
    os << "  Acceptable (0.6-0.8):" << std::setw(10) << summary.info_acceptable
       << " (" << (100.0 * summary.info_acceptable / summary.total_variants) << "%)\n";
    os << "  Low (<0.6):        " << std::setw(12) << summary.info_low
       << " (" << (100.0 * summary.info_low / summary.total_variants) << "%)\n\n";

    // MAF-stratified INFO
    if (!summary.mean_info_by_maf.empty()) {
        os << "Mean INFO by MAF Bin:\n";
        for (const auto& [bin, mean_info] : summary.mean_info_by_maf) {
            os << "  " << std::left << std::setw(12) << validation::maf_bin_name(bin)
               << " " << std::fixed << std::setprecision(4) << mean_info << "\n";
        }
    }

    os << "================================================================================\n";
}

void QCFilter::print_info_histogram(
    const QCSummary& summary,
    std::ostream& os,
    int width
) const {
    os << "\nINFO Score Distribution:\n";
    os << std::string(width + 20, '-') << "\n";

    // Create histogram bins
    const int n_bins = 10;
    std::vector<size_t> bins(n_bins, 0);

    for (const auto& qc : variant_qc_) {
        int bin = static_cast<int>(qc.info_score * n_bins);
        if (bin >= n_bins) bin = n_bins - 1;
        if (bin < 0) bin = 0;
        bins[bin]++;
    }

    // Find max for scaling
    size_t max_count = *std::max_element(bins.begin(), bins.end());

    // Print histogram
    for (int i = n_bins - 1; i >= 0; --i) {
        double bin_start = i * 0.1;
        double bin_end = (i + 1) * 0.1;

        int bar_len = max_count > 0 ?
            static_cast<int>(width * bins[i] / max_count) : 0;

        os << std::fixed << std::setprecision(1);
        os << std::setw(4) << bin_start << "-" << std::setw(4) << bin_end << " |";
        os << std::string(bar_len, '#');
        os << " " << bins[i] << "\n";
    }

    os << std::string(width + 20, '-') << "\n";
}

// ============================================================================
// ResultFilter Implementation
// ============================================================================

ResultFilter::ResultFilter(const QCConfig& config)
    : config_(config) {}

size_t ResultFilter::filter_in_place(
    std::vector<prob_t>& info_scores,
    std::vector<prob_t>& dosages,
    size_t num_samples,
    size_t num_markers
) {
    size_t filtered = 0;

    for (size_t m = 0; m < num_markers; ++m) {
        if (info_scores[m] < config_.min_info_score) {
            // Set all dosages for this marker to missing (-1)
            for (size_t s = 0; s < num_samples; ++s) {
                dosages[s * num_markers + m] = -1.0;
            }
            filtered++;
        }
    }

    return filtered;
}

std::vector<size_t> ResultFilter::get_passing_indices(
    const std::vector<prob_t>& info_scores
) const {
    std::vector<size_t> indices;
    for (size_t i = 0; i < info_scores.size(); ++i) {
        if (info_scores[i] >= config_.min_info_score) {
            indices.push_back(i);
        }
    }
    return indices;
}

std::vector<size_t> ResultFilter::get_failing_indices(
    const std::vector<prob_t>& info_scores
) const {
    std::vector<size_t> indices;
    for (size_t i = 0; i < info_scores.size(); ++i) {
        if (info_scores[i] < config_.min_info_score) {
            indices.push_back(i);
        }
    }
    return indices;
}

// ============================================================================
// ConcordanceQC Implementation
// ============================================================================

ConcordanceQC::ConcordanceQC(const ConcordanceConfig& config)
    : config_(config) {}

std::vector<sample_t> ConcordanceQC::identify_low_quality_samples(
    const validation::TruthData& truth,
    const validation::ImputedData& imputed,
    double threshold
) {
    std::vector<sample_t> low_quality;

    size_t n_samples = std::min(truth.num_samples(), imputed.num_samples());
    size_t n_markers = std::min(truth.num_markers(), imputed.num_markers());

    for (sample_t s = 0; s < n_samples; ++s) {
        size_t correct = 0;
        size_t total = 0;

        for (marker_t m = 0; m < n_markers; ++m) {
            uint8_t truth_gt = truth.get_genotype(s, m);
            uint8_t imputed_gt = imputed.get_genotype(s, m);

            if (truth_gt != ALLELE_MISSING && imputed_gt != ALLELE_MISSING) {
                total++;
                if (truth_gt == imputed_gt) {
                    correct++;
                }
            }
        }

        if (total >= config_.min_samples_for_eval) {
            double concordance = static_cast<double>(correct) / total;
            if (concordance < threshold) {
                low_quality.push_back(s);
            }
        }
    }

    return low_quality;
}

std::map<marker_t, double> ConcordanceQC::evaluate_variant_concordance(
    const validation::TruthData& truth,
    const validation::ImputedData& imputed
) {
    std::map<marker_t, double> result;

    size_t n_samples = std::min(truth.num_samples(), imputed.num_samples());
    size_t n_markers = std::min(truth.num_markers(), imputed.num_markers());

    for (marker_t m = 0; m < n_markers; ++m) {
        size_t correct = 0;
        size_t total = 0;

        for (sample_t s = 0; s < n_samples; ++s) {
            uint8_t truth_gt = truth.get_genotype(s, m);
            uint8_t imputed_gt = imputed.get_genotype(s, m);

            if (truth_gt != ALLELE_MISSING && imputed_gt != ALLELE_MISSING) {
                total++;
                if (truth_gt == imputed_gt) {
                    correct++;
                }
            }
        }

        if (total > 0) {
            result[m] = static_cast<double>(correct) / total;
        }
    }

    return result;
}

// ============================================================================
// Utility Functions
// ============================================================================

double calculate_info_score(
    const std::vector<double>& dosages,
    const std::vector<std::array<double, 3>>& probabilities
) {
    if (dosages.empty()) return 0.0;

    size_t n = dosages.size();

    // Calculate allele frequency
    double sum_dosage = std::accumulate(dosages.begin(), dosages.end(), 0.0);
    double p = sum_dosage / (2.0 * n);

    if (p <= 0 || p >= 1) return 1.0;  // Monomorphic

    // Calculate variance of posterior
    double var_sum = 0.0;
    for (size_t i = 0; i < n; ++i) {
        if (i < probabilities.size()) {
            const auto& probs = probabilities[i];
            // Expected dosage = 0*P(0) + 1*P(1) + 2*P(2)
            double exp_d = probs[1] + 2.0 * probs[2];
            // Expected dosage squared = 0*P(0) + 1*P(1) + 4*P(2)
            double exp_d2 = probs[1] + 4.0 * probs[2];
            // Variance = E[D²] - E[D]²
            double var = exp_d2 - exp_d * exp_d;
            var_sum += var;
        }
    }

    // INFO = 1 - sum(var) / (2 * p * (1-p) * n)
    double expected_var = 2.0 * p * (1.0 - p) * n;
    double info = expected_var > 0 ? 1.0 - var_sum / expected_var : 1.0;

    return std::max(0.0, std::min(1.0, info));
}

double calculate_info_score_approx(const std::vector<double>& dosages) {
    if (dosages.empty()) return 0.0;

    size_t n = dosages.size();

    // Calculate mean and variance
    double sum = std::accumulate(dosages.begin(), dosages.end(), 0.0);
    double mean = sum / n;

    double var_sum = 0.0;
    for (double d : dosages) {
        var_sum += (d - mean) * (d - mean);
    }
    double variance = var_sum / n;

    // Allele frequency
    double p = mean / 2.0;

    if (p <= 0 || p >= 1) return 1.0;  // Monomorphic

    // Expected variance under HWE
    double expected_var = 2.0 * p * (1.0 - p);

    // Approximation: INFO ~ var(dosage) / expected_var
    double info = expected_var > 0 ? variance / expected_var : 1.0;

    return std::max(0.0, std::min(1.0, info));
}

double hwe_exact_test(size_t n_aa, size_t n_ab, size_t n_bb) {
    // Simplified HWE exact test
    size_t n = n_aa + n_ab + n_bb;
    if (n == 0) return 1.0;

    double p = (2.0 * n_aa + n_ab) / (2.0 * n);
    double q = 1.0 - p;

    // Chi-squared approximation
    double exp_aa = p * p * n;
    double exp_ab = 2 * p * q * n;
    double exp_bb = q * q * n;

    double chi_sq = 0;
    if (exp_aa > 0) chi_sq += std::pow(n_aa - exp_aa, 2) / exp_aa;
    if (exp_ab > 0) chi_sq += std::pow(n_ab - exp_ab, 2) / exp_ab;
    if (exp_bb > 0) chi_sq += std::pow(n_bb - exp_bb, 2) / exp_bb;

    // Approximate p-value
    return std::exp(-chi_sq / 2);
}

QCSummary quick_qc_check(const std::string& vcf_file, double info_threshold) {
    QCConfig config;
    config.min_info_score = info_threshold;
    config.filter_by_info = true;

    QCFilter filter(config);
    return filter.compute_qc_metrics(vcf_file);
}

size_t filter_vcf_by_info(
    const std::string& input_vcf,
    const std::string& output_vcf,
    double min_info,
    bool verbose
) {
    QCConfig config;
    config.min_info_score = min_info;
    config.filter_by_info = true;
    config.filter_by_hwe = false;
    config.filter_by_call_rate = false;
    config.filter_by_maf = false;

    QCFilter filter(config);

    QCProgressCallback progress = nullptr;
    if (verbose) {
        progress = [](size_t completed, size_t total) {
            if (completed % 50000 == 0) {
                std::cout << "\rProcessing variants: " << completed << std::flush;
            }
        };
    }

    auto summary = filter.filter_vcf(input_vcf, output_vcf, progress);

    if (verbose) {
        std::cout << "\n";
        filter.print_summary(summary, std::cout);
    }

    return summary.variants_passing;
}

} // namespace analysis
} // namespace swiftimpute
