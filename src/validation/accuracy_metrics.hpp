#pragma once

#include "core/types.hpp"
#include <vector>
#include <string>
#include <map>
#include <memory>
#include <optional>

namespace swiftimpute {
namespace validation {

// Minor Allele Frequency bins for stratified analysis
enum class MAFBin {
    VERY_RARE,      // MAF < 0.5%
    RARE,           // 0.5% <= MAF < 1%
    LOW_FREQ,       // 1% <= MAF < 5%
    COMMON,         // 5% <= MAF < 50%
    ALL             // All variants combined
};

// Convert MAF to bin
MAFBin get_maf_bin(double maf);

// Get MAF bin name for reporting
std::string maf_bin_name(MAFBin bin);

// Per-variant accuracy metrics
struct VariantAccuracy {
    marker_t marker_idx;
    std::string variant_id;
    std::string chrom;
    uint64_t pos;

    double maf;                     // Minor allele frequency in truth data
    double info_score;              // INFO score (imputation quality)
    double dosage_r2;               // R² between imputed and true dosages
    double genotype_concordance;    // Fraction of genotypes matching exactly
    double allelic_concordance;     // Fraction of alleles matching

    uint32_t n_samples;             // Total samples evaluated
    uint32_t n_correct;             // Genotypes matching exactly
    uint32_t n_het_correct;         // Heterozygotes correctly imputed
    uint32_t n_homalt_correct;      // Homozygous alt correctly imputed

    // Confusion matrix counts
    uint32_t true_hom_ref;          // Truth = 0/0
    uint32_t true_het;              // Truth = 0/1 or 1/0
    uint32_t true_hom_alt;          // Truth = 1/1
    uint32_t imputed_hom_ref;       // Imputed = 0/0
    uint32_t imputed_het;           // Imputed = 0/1
    uint32_t imputed_hom_alt;       // Imputed = 1/1

    VariantAccuracy();
};

// Aggregated metrics for a set of variants (e.g., by MAF bin)
struct AggregatedMetrics {
    MAFBin maf_bin;
    uint32_t n_variants;
    uint32_t n_genotypes;           // Total genotypes evaluated

    // Mean metrics
    double mean_info_score;
    double mean_dosage_r2;
    double mean_concordance;
    double mean_allelic_r2;

    // Weighted metrics (by sample count)
    double weighted_dosage_r2;
    double weighted_concordance;

    // Quantiles for R²
    double r2_q25;                  // 25th percentile
    double r2_median;               // 50th percentile
    double r2_q75;                  // 75th percentile

    // Thresholds
    uint32_t n_variants_r2_gt_50;   // R² > 0.50
    uint32_t n_variants_r2_gt_80;   // R² > 0.80
    uint32_t n_variants_r2_gt_95;   // R² > 0.95

    AggregatedMetrics();
};

// Full validation report
struct ValidationReport {
    std::string reference_file;
    std::string target_file;
    std::string truth_file;
    std::string imputed_file;

    uint32_t n_samples;
    uint32_t n_variants;
    uint32_t n_variants_evaluated;  // Variants present in both truth and imputed

    // Overall metrics
    AggregatedMetrics overall;

    // MAF-stratified metrics
    std::map<MAFBin, AggregatedMetrics> by_maf;

    // Per-variant details (optional, can be large)
    std::vector<VariantAccuracy> variant_details;

    // Timing
    double evaluation_time_seconds;

    ValidationReport();

    // Output methods
    void print_summary(std::ostream& os) const;
    void write_csv(const std::string& filename) const;
    void write_json(const std::string& filename) const;
};

// Truth data container
class TruthData {
public:
    TruthData() = default;

    // Load truth genotypes from VCF
    static std::unique_ptr<TruthData> load_vcf(
        const std::string& filename,
        const std::string& region = ""
    );

    // Accessors
    marker_t num_markers() const { return markers_.size(); }
    sample_t num_samples() const { return samples_.size(); }

    const std::vector<Marker>& markers() const { return markers_; }
    const std::vector<Sample>& samples() const { return samples_; }

    // Get true genotype (0, 1, 2 for diploid dosage, or ALLELE_MISSING)
    uint8_t get_genotype(sample_t s, marker_t m) const;

    // Get true dosage (0.0, 1.0, 2.0 for hard calls)
    double get_dosage(sample_t s, marker_t m) const;

    // Check if genotype is missing
    bool is_missing(sample_t s, marker_t m) const;

    // Calculate MAF from truth data
    double calculate_maf(marker_t m) const;

private:
    std::vector<Marker> markers_;
    std::vector<Sample> samples_;
    std::unique_ptr<uint8_t[]> genotypes_;  // [num_samples][num_markers]

    TruthData(
        std::vector<Marker> markers,
        std::vector<Sample> samples,
        std::unique_ptr<uint8_t[]> genotypes
    );
};

// Imputed data wrapper (for comparison)
class ImputedData {
public:
    ImputedData() = default;

    // Load imputed genotypes/dosages from VCF
    static std::unique_ptr<ImputedData> load_vcf(
        const std::string& filename,
        const std::string& region = ""
    );

    // Accessors
    marker_t num_markers() const { return markers_.size(); }
    sample_t num_samples() const { return samples_.size(); }

    const std::vector<Marker>& markers() const { return markers_; }
    const std::vector<Sample>& samples() const { return samples_; }

    // Get imputed hard-call genotype (from GT field)
    uint8_t get_genotype(sample_t s, marker_t m) const;

    // Get imputed dosage (from DS field, or derived from GP)
    double get_dosage(sample_t s, marker_t m) const;

    // Get genotype probabilities (from GP field)
    void get_probabilities(sample_t s, marker_t m, double probs[3]) const;

    // Check if has dosage data
    bool has_dosages() const { return has_dosages_; }

    // Check if has probability data
    bool has_probabilities() const { return has_probabilities_; }

private:
    std::vector<Marker> markers_;
    std::vector<Sample> samples_;
    std::unique_ptr<uint8_t[]> genotypes_;      // [num_samples][num_markers]
    std::unique_ptr<float[]> dosages_;          // [num_samples][num_markers]
    std::unique_ptr<float[]> probabilities_;    // [num_samples][num_markers][3]
    bool has_dosages_;
    bool has_probabilities_;

    ImputedData(
        std::vector<Marker> markers,
        std::vector<Sample> samples,
        std::unique_ptr<uint8_t[]> genotypes,
        std::unique_ptr<float[]> dosages,
        std::unique_ptr<float[]> probabilities,
        bool has_dosages,
        bool has_probabilities
    );
};

// Main accuracy calculator
class AccuracyCalculator {
public:
    struct Config {
        bool compute_per_variant;       // Compute per-variant metrics (slower)
        bool compute_confusion_matrix;  // Compute confusion matrix per variant
        std::vector<MAFBin> maf_bins;   // Which MAF bins to report
        double min_maf;                 // Minimum MAF to include (default: 0)
        double max_maf;                 // Maximum MAF to include (default: 0.5)

        Config();
    };

    explicit AccuracyCalculator(const Config& config = Config());

    // Calculate accuracy metrics
    ValidationReport evaluate(
        const TruthData& truth,
        const ImputedData& imputed
    );

    // Calculate accuracy for a subset of samples
    ValidationReport evaluate_subset(
        const TruthData& truth,
        const ImputedData& imputed,
        const std::vector<sample_t>& sample_indices
    );

    // Quick summary (overall R² and concordance only)
    std::pair<double, double> quick_evaluate(
        const TruthData& truth,
        const ImputedData& imputed
    );

private:
    Config config_;

    // Internal calculation methods
    VariantAccuracy calculate_variant_accuracy(
        const TruthData& truth,
        const ImputedData& imputed,
        marker_t marker_idx,
        const std::vector<sample_t>* sample_subset = nullptr
    );

    AggregatedMetrics aggregate_metrics(
        const std::vector<VariantAccuracy>& variants,
        MAFBin bin
    );

    double calculate_dosage_r2(
        const std::vector<double>& truth_dosages,
        const std::vector<double>& imputed_dosages
    );

    double calculate_info_score(
        const std::vector<double>& imputed_dosages,
        const std::vector<std::array<double, 3>>& probabilities
    );
};

// Utility functions
namespace utils {

// Pearson correlation coefficient
double pearson_correlation(
    const std::vector<double>& x,
    const std::vector<double>& y
);

// R-squared (squared correlation)
double r_squared(
    const std::vector<double>& truth,
    const std::vector<double>& imputed
);

// INFO score calculation
// INFO = 1 - (sum of variance of posterior) / (2 * p * (1-p) * n)
// where p is estimated allele frequency
double info_score(
    const std::vector<double>& dosages,
    const std::vector<std::array<double, 3>>& probabilities
);

// Allele frequency from dosages
double allele_frequency(const std::vector<double>& dosages);

// Mean and variance
std::pair<double, double> mean_variance(const std::vector<double>& values);

// Quantile calculation
double quantile(std::vector<double> values, double q);

} // namespace utils

} // namespace validation
} // namespace swiftimpute
