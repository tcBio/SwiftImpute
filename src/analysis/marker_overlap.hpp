#pragma once

#include "../core/types.hpp"
#include <string>
#include <vector>
#include <map>
#include <memory>
#include <set>
#include <functional>

namespace swiftimpute {
namespace analysis {

/**
 * @brief Marker key for comparison (chrom:pos:ref:alt)
 */
struct MarkerKey {
    std::string chrom;
    uint64_t pos;
    std::string ref;
    std::string alt;

    bool operator<(const MarkerKey& other) const {
        if (chrom != other.chrom) return chrom < other.chrom;
        if (pos != other.pos) return pos < other.pos;
        if (ref != other.ref) return ref < other.ref;
        return alt < other.alt;
    }

    bool operator==(const MarkerKey& other) const {
        return chrom == other.chrom && pos == other.pos &&
               ref == other.ref && alt == other.alt;
    }

    std::string to_string() const {
        return chrom + ":" + std::to_string(pos) + ":" + ref + ":" + alt;
    }
};

/**
 * @brief Statistics for a single chromosome's overlap
 */
struct ChromosomeOverlapStats {
    std::string chrom;
    size_t reference_markers;       // Markers in reference panel
    size_t target_markers;          // Markers in target panel
    size_t overlapping_markers;     // Markers in both
    size_t reference_only;          // Markers only in reference
    size_t target_only;             // Markers only in target (won't be imputed)

    // Position coverage
    uint64_t ref_start_pos;         // First marker position in reference
    uint64_t ref_end_pos;           // Last marker position in reference
    uint64_t target_start_pos;      // First marker position in target
    uint64_t target_end_pos;        // Last marker position in target

    // Derived metrics
    double overlap_rate() const {
        return target_markers > 0 ?
            static_cast<double>(overlapping_markers) / target_markers : 0.0;
    }

    double coverage_rate() const {
        return reference_markers > 0 ?
            static_cast<double>(overlapping_markers) / reference_markers : 0.0;
    }

    // Marker density (markers per Mb)
    double ref_marker_density() const {
        uint64_t span = ref_end_pos - ref_start_pos;
        return span > 0 ? reference_markers * 1e6 / span : 0.0;
    }

    double target_marker_density() const {
        uint64_t span = target_end_pos - target_start_pos;
        return span > 0 ? target_markers * 1e6 / span : 0.0;
    }

    ChromosomeOverlapStats() :
        reference_markers(0), target_markers(0), overlapping_markers(0),
        reference_only(0), target_only(0),
        ref_start_pos(UINT64_MAX), ref_end_pos(0),
        target_start_pos(UINT64_MAX), target_end_pos(0) {}
};

/**
 * @brief Gap analysis for sparse marker coverage
 */
struct GapInfo {
    std::string chrom;
    uint64_t start;
    uint64_t end;
    uint64_t length;
    size_t ref_markers_in_gap;      // Reference markers available in this gap

    GapInfo(const std::string& c, uint64_t s, uint64_t e, size_t rm = 0)
        : chrom(c), start(s), end(e), length(e - s), ref_markers_in_gap(rm) {}
};

/**
 * @brief Comprehensive overlap analysis report
 */
struct OverlapReport {
    // Files analyzed
    std::string reference_file;
    std::string target_file;

    // Overall statistics
    size_t total_reference_markers;
    size_t total_target_markers;
    size_t total_overlapping_markers;
    size_t total_imputable_markers;     // Reference markers that can be imputed

    // Sample counts
    size_t reference_samples;
    size_t target_samples;

    // Per-chromosome breakdown
    std::vector<ChromosomeOverlapStats> by_chromosome;

    // Gap analysis (large gaps in target coverage)
    std::vector<GapInfo> large_gaps;    // Gaps > threshold
    size_t num_gaps;
    uint64_t mean_gap_size;
    uint64_t max_gap_size;
    uint64_t median_gap_size;

    // Allele frequency correlation (if available)
    double af_correlation;              // Pearson r between ref and target AFs
    bool has_af_data;

    // Quality warnings
    std::vector<std::string> warnings;

    // Recommendations
    std::vector<std::string> recommendations;

    // Timing
    double analysis_time_seconds;

    // Derived metrics
    double overall_overlap_rate() const {
        return total_target_markers > 0 ?
            static_cast<double>(total_overlapping_markers) / total_target_markers : 0.0;
    }

    double imputation_coverage() const {
        return total_reference_markers > 0 ?
            static_cast<double>(total_imputable_markers) / total_reference_markers : 0.0;
    }

    OverlapReport() :
        total_reference_markers(0), total_target_markers(0),
        total_overlapping_markers(0), total_imputable_markers(0),
        reference_samples(0), target_samples(0),
        num_gaps(0), mean_gap_size(0), max_gap_size(0), median_gap_size(0),
        af_correlation(0.0), has_af_data(false),
        analysis_time_seconds(0.0) {}
};

/**
 * @brief Configuration for overlap analysis
 */
struct OverlapConfig {
    // Gap detection
    uint64_t min_gap_size;              // Minimum gap to report (bp)
    uint64_t max_gap_for_warning;       // Gap size that triggers warning (bp)

    // Matching options
    bool require_allele_match;          // Require ref/alt alleles to match
    bool check_strand_flips;            // Detect potential strand flips

    // Analysis options
    bool compute_af_correlation;        // Compute allele frequency correlation
    bool per_chromosome_stats;          // Compute per-chromosome statistics
    bool detailed_gap_analysis;         // Compute detailed gap analysis

    OverlapConfig() :
        min_gap_size(100000),           // 100 kb default
        max_gap_for_warning(1000000),   // 1 Mb triggers warning
        require_allele_match(true),
        check_strand_flips(true),
        compute_af_correlation(true),
        per_chromosome_stats(true),
        detailed_gap_analysis(true) {}
};

/**
 * @brief Progress callback for long-running analysis
 */
using ProgressCallback = std::function<void(size_t completed, size_t total, const std::string& stage)>;

/**
 * @brief Marker overlap analyzer
 *
 * Analyzes the overlap between reference and target marker sets
 * to help users understand imputation coverage and identify issues.
 */
class MarkerOverlapAnalyzer {
public:
    explicit MarkerOverlapAnalyzer(const OverlapConfig& config = OverlapConfig());

    /**
     * @brief Analyze overlap from loaded data
     */
    OverlapReport analyze(
        const std::vector<Marker>& reference_markers,
        const std::vector<Marker>& target_markers,
        ProgressCallback progress = nullptr
    );

    /**
     * @brief Analyze overlap directly from VCF files
     */
    OverlapReport analyze_files(
        const std::string& reference_vcf,
        const std::string& target_vcf,
        ProgressCallback progress = nullptr
    );

    /**
     * @brief Quick overlap check (fast, minimal statistics)
     */
    std::pair<size_t, size_t> quick_overlap_count(
        const std::vector<Marker>& reference_markers,
        const std::vector<Marker>& target_markers
    );

    /**
     * @brief Generate recommendations based on analysis
     */
    std::vector<std::string> generate_recommendations(const OverlapReport& report);

    /**
     * @brief Print formatted report to stream
     */
    void print_report(const OverlapReport& report, std::ostream& os) const;

    /**
     * @brief Print ASCII visualization of coverage
     */
    void print_coverage_plot(const OverlapReport& report, std::ostream& os,
                             int width = 60) const;

    /**
     * @brief Export report to JSON
     */
    void export_json(const OverlapReport& report, const std::string& filename) const;

    /**
     * @brief Export report to CSV
     */
    void export_csv(const OverlapReport& report, const std::string& filename) const;

private:
    OverlapConfig config_;

    // Internal helper methods
    std::set<MarkerKey> build_marker_set(const std::vector<Marker>& markers);

    ChromosomeOverlapStats analyze_chromosome(
        const std::vector<Marker>& ref_markers,
        const std::vector<Marker>& target_markers,
        const std::string& chrom
    );

    std::vector<GapInfo> find_gaps(
        const std::vector<Marker>& target_markers,
        const std::vector<Marker>& ref_markers,
        const std::string& chrom
    );

    double compute_af_correlation(
        const std::vector<Marker>& ref_markers,
        const std::vector<Marker>& target_markers,
        const std::set<MarkerKey>& overlap
    );

    bool is_strand_flip(const Marker& m1, const Marker& m2) const;
    bool is_complement(char a, char b) const;
};

/**
 * @brief Factory function to create analyzer with preset configs
 */
std::unique_ptr<MarkerOverlapAnalyzer> create_overlap_analyzer(
    const std::string& preset = "default"
);

/**
 * @brief Convenience function for quick analysis
 */
OverlapReport analyze_overlap(
    const std::string& reference_vcf,
    const std::string& target_vcf,
    bool verbose = true
);

} // namespace analysis
} // namespace swiftimpute
