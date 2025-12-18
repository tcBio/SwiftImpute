#pragma once

#include "../core/types.hpp"
#include "../validation/accuracy_metrics.hpp"
#include <string>
#include <vector>
#include <map>
#include <memory>
#include <functional>

namespace swiftimpute {
namespace analysis {

/**
 * @brief INFO score thresholds commonly used in imputation QC
 */
struct InfoScoreThresholds {
    static constexpr double HIGH_QUALITY = 0.9;     // High confidence
    static constexpr double GOOD_QUALITY = 0.8;     // Standard threshold
    static constexpr double ACCEPTABLE = 0.6;       // Minimum for most studies
    static constexpr double LOW_QUALITY = 0.3;      // Often filtered out

    // Beagle-equivalent thresholds (for DR² metric)
    static constexpr double BEAGLE_HIGH = 0.9;
    static constexpr double BEAGLE_DEFAULT = 0.8;
};

/**
 * @brief Quality metrics for a single variant
 */
struct VariantQC {
    std::string chrom;
    uint64_t pos;
    std::string id;
    std::string ref;
    std::string alt;

    // Quality scores
    double info_score;              // INFO/R² score (0-1)
    double dosage_r2;               // Dosage R² (if truth available)
    double hw_pvalue;               // Hardy-Weinberg p-value
    double call_rate;               // Proportion of non-missing calls
    double maf;                     // Minor allele frequency

    // Flags
    bool passes_info;               // Passes INFO threshold
    bool passes_hwe;                // Passes HWE threshold
    bool passes_call_rate;          // Passes call rate threshold
    bool passes_maf;                // Passes MAF threshold
    bool passes_all;                // Passes all filters

    VariantQC() :
        info_score(0), dosage_r2(0), hw_pvalue(1), call_rate(1), maf(0),
        passes_info(true), passes_hwe(true), passes_call_rate(true),
        passes_maf(true), passes_all(true) {}
};

/**
 * @brief QC summary statistics
 */
struct QCSummary {
    size_t total_variants;
    size_t variants_passing;
    size_t variants_filtered;

    // Per-filter counts
    size_t filtered_by_info;
    size_t filtered_by_hwe;
    size_t filtered_by_call_rate;
    size_t filtered_by_maf;

    // Score distributions
    double mean_info_score;
    double median_info_score;
    double info_q25;                // 25th percentile
    double info_q75;                // 75th percentile

    // INFO score tier breakdown
    size_t info_high;               // >= 0.9
    size_t info_good;               // 0.8 - 0.9
    size_t info_acceptable;         // 0.6 - 0.8
    size_t info_low;                // < 0.6

    // MAF-stratified INFO scores
    std::map<validation::MAFBin, double> mean_info_by_maf;

    QCSummary() :
        total_variants(0), variants_passing(0), variants_filtered(0),
        filtered_by_info(0), filtered_by_hwe(0), filtered_by_call_rate(0),
        filtered_by_maf(0), mean_info_score(0), median_info_score(0),
        info_q25(0), info_q75(0), info_high(0), info_good(0),
        info_acceptable(0), info_low(0) {}
};

/**
 * @brief Configuration for QC filtering
 */
struct QCConfig {
    // INFO score filtering
    double min_info_score;          // Minimum INFO score
    bool filter_by_info;            // Enable INFO filtering

    // Hardy-Weinberg equilibrium
    double min_hwe_pvalue;          // Minimum HWE p-value
    bool filter_by_hwe;             // Enable HWE filtering

    // Call rate filtering
    double min_call_rate;           // Minimum call rate (0-1)
    bool filter_by_call_rate;       // Enable call rate filtering

    // MAF filtering
    double min_maf;                 // Minimum MAF
    double max_maf;                 // Maximum MAF (default 0.5)
    bool filter_by_maf;             // Enable MAF filtering

    // Output options
    bool output_per_variant_qc;     // Output per-variant QC metrics
    bool output_filtered_list;      // Output list of filtered variants

    QCConfig() :
        min_info_score(0.8),
        filter_by_info(true),
        min_hwe_pvalue(1e-6),
        filter_by_hwe(false),
        min_call_rate(0.95),
        filter_by_call_rate(false),
        min_maf(0.0),
        max_maf(0.5),
        filter_by_maf(false),
        output_per_variant_qc(true),
        output_filtered_list(true) {}

    // Preset configurations
    static QCConfig strict() {
        QCConfig config;
        config.min_info_score = 0.9;
        config.min_hwe_pvalue = 1e-4;
        config.filter_by_hwe = true;
        config.min_call_rate = 0.98;
        config.filter_by_call_rate = true;
        return config;
    }

    static QCConfig standard() {
        QCConfig config;
        config.min_info_score = 0.8;
        config.filter_by_hwe = false;
        config.filter_by_call_rate = false;
        return config;
    }

    static QCConfig lenient() {
        QCConfig config;
        config.min_info_score = 0.6;
        config.filter_by_hwe = false;
        config.filter_by_call_rate = false;
        return config;
    }

    static QCConfig rare_variants() {
        QCConfig config;
        config.min_info_score = 0.3;  // Lower threshold for rare variants
        config.min_maf = 0.0;
        config.max_maf = 0.01;
        config.filter_by_maf = true;
        return config;
    }
};

/**
 * @brief Progress callback for QC operations
 */
using QCProgressCallback = std::function<void(size_t completed, size_t total)>;

/**
 * @brief QC filter for imputed variants
 */
class QCFilter {
public:
    explicit QCFilter(const QCConfig& config = QCConfig());

    /**
     * @brief Check if a single variant passes QC
     */
    bool passes(const VariantQC& variant) const;

    /**
     * @brief Apply QC to imputed VCF and generate filtered output
     *
     * @param input_vcf Input VCF file (imputed)
     * @param output_vcf Output VCF file (filtered)
     * @param progress Progress callback
     * @return QCSummary with filtering statistics
     */
    QCSummary filter_vcf(
        const std::string& input_vcf,
        const std::string& output_vcf,
        QCProgressCallback progress = nullptr
    );

    /**
     * @brief Compute QC metrics without filtering
     */
    QCSummary compute_qc_metrics(
        const std::string& input_vcf,
        QCProgressCallback progress = nullptr
    );

    /**
     * @brief Get per-variant QC from last operation
     */
    const std::vector<VariantQC>& get_variant_qc() const { return variant_qc_; }

    /**
     * @brief Export variant QC to file
     */
    void export_variant_qc(const std::string& filename) const;

    /**
     * @brief Print QC summary report
     */
    void print_summary(const QCSummary& summary, std::ostream& os) const;

    /**
     * @brief Print INFO score distribution histogram
     */
    void print_info_histogram(const QCSummary& summary, std::ostream& os,
                              int width = 50) const;

    // Configuration access
    const QCConfig& config() const { return config_; }
    void set_config(const QCConfig& config) { config_ = config; }

private:
    QCConfig config_;
    std::vector<VariantQC> variant_qc_;

    // Internal methods
    double calculate_hwe_pvalue(
        size_t n_hom_ref, size_t n_het, size_t n_hom_alt
    ) const;

    double calculate_maf(const std::vector<double>& dosages) const;

    QCSummary aggregate_qc(const std::vector<VariantQC>& variants) const;
};

/**
 * @brief Post-imputation variant filter
 *
 * Applies filters to an ImputationResult in memory
 */
class ResultFilter {
public:
    explicit ResultFilter(const QCConfig& config = QCConfig());

    /**
     * @brief Filter imputation result in place
     *
     * Sets filtered variants to missing
     */
    size_t filter_in_place(
        std::vector<prob_t>& info_scores,
        std::vector<prob_t>& dosages,
        size_t num_samples,
        size_t num_markers
    );

    /**
     * @brief Get indices of variants passing QC
     */
    std::vector<size_t> get_passing_indices(
        const std::vector<prob_t>& info_scores
    ) const;

    /**
     * @brief Get indices of variants failing QC
     */
    std::vector<size_t> get_failing_indices(
        const std::vector<prob_t>& info_scores
    ) const;

private:
    QCConfig config_;
};

/**
 * @brief Concordance-based QC (requires truth data)
 */
class ConcordanceQC {
public:
    struct ConcordanceConfig {
        double min_concordance;         // Minimum genotype concordance
        double min_dosage_r2;           // Minimum dosage R²
        size_t min_samples_for_eval;    // Minimum samples for evaluation

        ConcordanceConfig() :
            min_concordance(0.9),
            min_dosage_r2(0.8),
            min_samples_for_eval(10) {}
    };

    explicit ConcordanceQC(const ConcordanceConfig& config = ConcordanceConfig());

    /**
     * @brief Evaluate concordance and identify low-quality samples
     */
    std::vector<sample_t> identify_low_quality_samples(
        const validation::TruthData& truth,
        const validation::ImputedData& imputed,
        double threshold
    );

    /**
     * @brief Evaluate concordance per variant
     */
    std::map<marker_t, double> evaluate_variant_concordance(
        const validation::TruthData& truth,
        const validation::ImputedData& imputed
    );

private:
    ConcordanceConfig config_;
};

// ============================================================================
// Utility Functions
// ============================================================================

/**
 * @brief Calculate INFO score from dosages and probabilities
 *
 * INFO = 1 - (sum of variance) / (2 * p * (1-p) * n)
 */
double calculate_info_score(
    const std::vector<double>& dosages,
    const std::vector<std::array<double, 3>>& probabilities
);

/**
 * @brief Calculate INFO score from dosages only (approximation)
 */
double calculate_info_score_approx(const std::vector<double>& dosages);

/**
 * @brief Hardy-Weinberg exact test p-value
 */
double hwe_exact_test(size_t n_aa, size_t n_ab, size_t n_bb);

/**
 * @brief Quick QC check from VCF
 */
QCSummary quick_qc_check(
    const std::string& vcf_file,
    double info_threshold = 0.8
);

/**
 * @brief Filter VCF by INFO score
 *
 * Convenience function for simple filtering
 */
size_t filter_vcf_by_info(
    const std::string& input_vcf,
    const std::string& output_vcf,
    double min_info = 0.8,
    bool verbose = true
);

} // namespace analysis
} // namespace swiftimpute
