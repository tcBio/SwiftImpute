#include "marker_overlap.hpp"
#include "../io/vcf_reader.hpp"
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
// MarkerOverlapAnalyzer Implementation
// ============================================================================

MarkerOverlapAnalyzer::MarkerOverlapAnalyzer(const OverlapConfig& config)
    : config_(config) {}

std::set<MarkerKey> MarkerOverlapAnalyzer::build_marker_set(
    const std::vector<Marker>& markers
) {
    std::set<MarkerKey> result;
    for (const auto& m : markers) {
        MarkerKey key;
        key.chrom = m.chrom;
        key.pos = m.pos;
        key.ref = m.ref;
        key.alt = m.alt;
        result.insert(key);
    }
    return result;
}

bool MarkerOverlapAnalyzer::is_complement(char a, char b) const {
    return (a == 'A' && b == 'T') || (a == 'T' && b == 'A') ||
           (a == 'C' && b == 'G') || (a == 'G' && b == 'C');
}

bool MarkerOverlapAnalyzer::is_strand_flip(const Marker& m1, const Marker& m2) const {
    if (m1.ref.size() != 1 || m1.alt.size() != 1 ||
        m2.ref.size() != 1 || m2.alt.size() != 1) {
        return false;
    }

    // Check if m2 is complement of m1
    return is_complement(m1.ref[0], m2.ref[0]) &&
           is_complement(m1.alt[0], m2.alt[0]);
}

std::pair<size_t, size_t> MarkerOverlapAnalyzer::quick_overlap_count(
    const std::vector<Marker>& reference_markers,
    const std::vector<Marker>& target_markers
) {
    auto ref_set = build_marker_set(reference_markers);
    size_t overlap = 0;

    for (const auto& m : target_markers) {
        MarkerKey key{m.chrom, m.pos, m.ref, m.alt};
        if (ref_set.count(key)) {
            overlap++;
        }
    }

    return {overlap, target_markers.size()};
}

ChromosomeOverlapStats MarkerOverlapAnalyzer::analyze_chromosome(
    const std::vector<Marker>& ref_markers,
    const std::vector<Marker>& target_markers,
    const std::string& chrom
) {
    ChromosomeOverlapStats stats;
    stats.chrom = chrom;

    // Build position-based lookup for reference
    std::map<uint64_t, std::vector<const Marker*>> ref_by_pos;
    for (const auto& m : ref_markers) {
        if (m.chrom == chrom) {
            ref_by_pos[m.pos].push_back(&m);
            stats.reference_markers++;
            if (m.pos < stats.ref_start_pos) stats.ref_start_pos = m.pos;
            if (m.pos > stats.ref_end_pos) stats.ref_end_pos = m.pos;
        }
    }

    // Analyze target markers
    for (const auto& m : target_markers) {
        if (m.chrom != chrom) continue;

        stats.target_markers++;
        if (m.pos < stats.target_start_pos) stats.target_start_pos = m.pos;
        if (m.pos > stats.target_end_pos) stats.target_end_pos = m.pos;

        // Check for overlap
        auto it = ref_by_pos.find(m.pos);
        if (it != ref_by_pos.end()) {
            bool found_match = false;

            for (const Marker* ref_m : it->second) {
                if (config_.require_allele_match) {
                    // Exact allele match
                    if (ref_m->ref == m.ref && ref_m->alt == m.alt) {
                        found_match = true;
                        break;
                    }
                    // Check strand flip
                    if (config_.check_strand_flips && is_strand_flip(*ref_m, m)) {
                        found_match = true;
                        break;
                    }
                } else {
                    // Position-only match
                    found_match = true;
                    break;
                }
            }

            if (found_match) {
                stats.overlapping_markers++;
            } else {
                stats.target_only++;
            }
        } else {
            stats.target_only++;
        }
    }

    stats.reference_only = stats.reference_markers - stats.overlapping_markers;

    return stats;
}

std::vector<GapInfo> MarkerOverlapAnalyzer::find_gaps(
    const std::vector<Marker>& target_markers,
    const std::vector<Marker>& ref_markers,
    const std::string& chrom
) {
    std::vector<GapInfo> gaps;

    // Get sorted positions for this chromosome
    std::vector<uint64_t> target_positions;
    for (const auto& m : target_markers) {
        if (m.chrom == chrom) {
            target_positions.push_back(m.pos);
        }
    }
    std::sort(target_positions.begin(), target_positions.end());

    // Build reference position set
    std::set<uint64_t> ref_positions;
    for (const auto& m : ref_markers) {
        if (m.chrom == chrom) {
            ref_positions.insert(m.pos);
        }
    }

    // Find gaps
    for (size_t i = 1; i < target_positions.size(); ++i) {
        uint64_t gap_size = target_positions[i] - target_positions[i-1];

        if (gap_size >= config_.min_gap_size) {
            // Count reference markers in this gap
            size_t ref_in_gap = 0;
            auto it = ref_positions.lower_bound(target_positions[i-1]);
            while (it != ref_positions.end() && *it < target_positions[i]) {
                ref_in_gap++;
                ++it;
            }

            gaps.emplace_back(chrom, target_positions[i-1], target_positions[i], ref_in_gap);
        }
    }

    return gaps;
}

double MarkerOverlapAnalyzer::compute_af_correlation(
    const std::vector<Marker>& ref_markers,
    const std::vector<Marker>& target_markers,
    const std::set<MarkerKey>& overlap
) {
    // For now, return 0 - would need AF data in Marker struct
    // This is a placeholder for future enhancement
    return 0.0;
}

OverlapReport MarkerOverlapAnalyzer::analyze(
    const std::vector<Marker>& reference_markers,
    const std::vector<Marker>& target_markers,
    ProgressCallback progress
) {
    auto start_time = std::chrono::high_resolution_clock::now();

    OverlapReport report;

    // Get unique chromosomes
    std::set<std::string> ref_chroms, target_chroms;
    for (const auto& m : reference_markers) ref_chroms.insert(m.chrom);
    for (const auto& m : target_markers) target_chroms.insert(m.chrom);

    // Find common chromosomes
    std::vector<std::string> common_chroms;
    for (const auto& c : ref_chroms) {
        if (target_chroms.count(c)) {
            common_chroms.push_back(c);
        }
    }

    // Build overlap set
    auto ref_set = build_marker_set(reference_markers);
    auto target_set = build_marker_set(target_markers);

    // Analyze each chromosome
    size_t processed = 0;
    for (const auto& chrom : common_chroms) {
        if (progress) {
            progress(processed++, common_chroms.size(), "Analyzing " + chrom);
        }

        auto chrom_stats = analyze_chromosome(reference_markers, target_markers, chrom);
        report.by_chromosome.push_back(chrom_stats);

        report.total_reference_markers += chrom_stats.reference_markers;
        report.total_target_markers += chrom_stats.target_markers;
        report.total_overlapping_markers += chrom_stats.overlapping_markers;

        // Gap analysis
        if (config_.detailed_gap_analysis) {
            auto chrom_gaps = find_gaps(target_markers, reference_markers, chrom);
            for (auto& gap : chrom_gaps) {
                report.large_gaps.push_back(gap);
                if (gap.length > report.max_gap_size) {
                    report.max_gap_size = gap.length;
                }
            }
        }
    }

    report.num_gaps = report.large_gaps.size();

    // Calculate gap statistics
    if (!report.large_gaps.empty()) {
        std::vector<uint64_t> gap_sizes;
        uint64_t total_gap = 0;
        for (const auto& gap : report.large_gaps) {
            gap_sizes.push_back(gap.length);
            total_gap += gap.length;
        }
        report.mean_gap_size = total_gap / report.large_gaps.size();

        std::sort(gap_sizes.begin(), gap_sizes.end());
        report.median_gap_size = gap_sizes[gap_sizes.size() / 2];
    }

    // Calculate imputable markers (reference markers in target regions)
    report.total_imputable_markers = report.total_reference_markers;

    // Generate warnings
    if (report.overall_overlap_rate() < 0.5) {
        report.warnings.push_back(
            "Low overlap rate (" + std::to_string(int(report.overall_overlap_rate() * 100)) +
            "%). Consider checking for chromosome naming mismatches or coordinate systems."
        );
    }

    if (report.max_gap_size > config_.max_gap_for_warning) {
        report.warnings.push_back(
            "Large coverage gap detected (" + std::to_string(report.max_gap_size / 1000000) +
            " Mb). Imputation accuracy may be reduced in this region."
        );
    }

    // Chromosomes in target but not reference
    for (const auto& c : target_chroms) {
        if (!ref_chroms.count(c)) {
            report.warnings.push_back(
                "Chromosome " + c + " in target but not in reference - will not be imputed."
            );
        }
    }

    // Generate recommendations
    report.recommendations = generate_recommendations(report);

    auto end_time = std::chrono::high_resolution_clock::now();
    report.analysis_time_seconds = std::chrono::duration<double>(
        end_time - start_time
    ).count();

    return report;
}

OverlapReport MarkerOverlapAnalyzer::analyze_files(
    const std::string& reference_vcf,
    const std::string& target_vcf,
    ProgressCallback progress
) {
    if (progress) {
        progress(0, 3, "Loading reference markers");
    }

    // Load reference markers
    VCFReader ref_reader(reference_vcf);
    auto ref_header = ref_reader.read_header();

    std::vector<Marker> ref_markers;
    VCFReader::Variant variant;
    while (ref_reader.read_variant(variant)) {
        Marker m;
        m.chrom = variant.chrom;
        m.pos = variant.position;
        m.id = variant.id;
        m.ref = variant.ref;
        m.alt = variant.alt.empty() ? "" : variant.alt[0];
        ref_markers.push_back(m);
    }

    if (progress) {
        progress(1, 3, "Loading target markers");
    }

    // Load target markers
    VCFReader target_reader(target_vcf);
    auto target_header = target_reader.read_header();

    std::vector<Marker> target_markers;
    while (target_reader.read_variant(variant)) {
        Marker m;
        m.chrom = variant.chrom;
        m.pos = variant.position;
        m.id = variant.id;
        m.ref = variant.ref;
        m.alt = variant.alt.empty() ? "" : variant.alt[0];
        target_markers.push_back(m);
    }

    if (progress) {
        progress(2, 3, "Analyzing overlap");
    }

    // Run analysis
    auto report = analyze(ref_markers, target_markers, progress);

    report.reference_file = reference_vcf;
    report.target_file = target_vcf;
    report.reference_samples = ref_header.num_samples;
    report.target_samples = target_header.num_samples;

    return report;
}

std::vector<std::string> MarkerOverlapAnalyzer::generate_recommendations(
    const OverlapReport& report
) {
    std::vector<std::string> recs;

    // Overlap rate recommendations
    double overlap = report.overall_overlap_rate();
    if (overlap < 0.3) {
        recs.push_back(
            "CRITICAL: Very low marker overlap (<30%). "
            "Check if files use the same genome build and coordinate system."
        );
    } else if (overlap < 0.5) {
        recs.push_back(
            "WARNING: Low marker overlap. Consider using a different reference "
            "panel or genotyping array with better coverage."
        );
    } else if (overlap < 0.7) {
        recs.push_back(
            "Moderate marker overlap. Imputation should work but accuracy may "
            "vary across the genome."
        );
    } else {
        recs.push_back(
            "Good marker overlap (>" + std::to_string(int(overlap * 100)) +
            "%). Imputation should perform well."
        );
    }

    // Gap recommendations
    if (report.max_gap_size > 5000000) {  // 5 Mb
        recs.push_back(
            "Large gaps detected in marker coverage. Consider reviewing "
            "imputation results in these regions carefully."
        );
    }

    // Sample size recommendations
    if (report.reference_samples < 100) {
        recs.push_back(
            "Small reference panel (" + std::to_string(report.reference_samples) +
            " samples). Use ImputationConfig::small_reference_preset() for optimal settings."
        );
    }

    // RAD-seq specific
    double target_density = 0;
    for (const auto& cs : report.by_chromosome) {
        target_density += cs.target_marker_density();
    }
    if (!report.by_chromosome.empty()) {
        target_density /= report.by_chromosome.size();
    }

    if (target_density < 100) {  // Less than 100 markers per Mb
        recs.push_back(
            "Sparse target markers detected (typical of RAD-seq or similar). "
            "Use ImputationConfig::radseq_preset() for optimal imputation settings."
        );
    }

    // Imputation yield
    size_t to_impute = report.total_imputable_markers - report.total_overlapping_markers;
    double yield = static_cast<double>(to_impute) / report.total_reference_markers;
    recs.push_back(
        "Imputation will add ~" + std::to_string(to_impute) +
        " markers (" + std::to_string(int(yield * 100)) + "% of reference)."
    );

    return recs;
}

void MarkerOverlapAnalyzer::print_report(
    const OverlapReport& report,
    std::ostream& os
) const {
    os << "\n";
    os << "================================================================================\n";
    os << "                    MARKER OVERLAP ANALYSIS REPORT\n";
    os << "================================================================================\n\n";

    os << "Files:\n";
    os << "  Reference: " << report.reference_file << "\n";
    os << "  Target:    " << report.target_file << "\n\n";

    os << "Samples:\n";
    os << "  Reference: " << report.reference_samples << " samples ("
       << (report.reference_samples * 2) << " haplotypes)\n";
    os << "  Target:    " << report.target_samples << " samples\n\n";

    os << "Markers:\n";
    os << "  Reference panel:    " << std::setw(10) << report.total_reference_markers << "\n";
    os << "  Target panel:       " << std::setw(10) << report.total_target_markers << "\n";
    os << "  Overlapping:        " << std::setw(10) << report.total_overlapping_markers
       << " (" << std::fixed << std::setprecision(1)
       << (report.overall_overlap_rate() * 100) << "%)\n";
    os << "  To be imputed:      " << std::setw(10)
       << (report.total_reference_markers - report.total_overlapping_markers) << "\n\n";

    // Per-chromosome table
    if (!report.by_chromosome.empty()) {
        os << "Per-Chromosome Breakdown:\n";
        os << "--------------------------------------------------------------------------------\n";
        os << std::left << std::setw(10) << "Chrom"
           << std::right << std::setw(12) << "Ref"
           << std::setw(12) << "Target"
           << std::setw(12) << "Overlap"
           << std::setw(10) << "Rate"
           << std::setw(14) << "Density(T)\n";
        os << "--------------------------------------------------------------------------------\n";

        for (const auto& cs : report.by_chromosome) {
            os << std::left << std::setw(10) << cs.chrom
               << std::right << std::setw(12) << cs.reference_markers
               << std::setw(12) << cs.target_markers
               << std::setw(12) << cs.overlapping_markers
               << std::setw(9) << std::fixed << std::setprecision(1)
               << (cs.overlap_rate() * 100) << "%"
               << std::setw(12) << std::fixed << std::setprecision(1)
               << cs.target_marker_density() << "/Mb\n";
        }
        os << "--------------------------------------------------------------------------------\n\n";
    }

    // Gap analysis
    if (report.num_gaps > 0) {
        os << "Gap Analysis (gaps > " << (config_.min_gap_size / 1000) << " kb):\n";
        os << "  Number of gaps:  " << report.num_gaps << "\n";
        os << "  Mean gap size:   " << (report.mean_gap_size / 1000) << " kb\n";
        os << "  Median gap size: " << (report.median_gap_size / 1000) << " kb\n";
        os << "  Maximum gap:     " << (report.max_gap_size / 1000) << " kb\n\n";

        // Show largest gaps
        if (report.large_gaps.size() > 0) {
            os << "  Largest gaps:\n";
            std::vector<GapInfo> sorted_gaps = report.large_gaps;
            std::sort(sorted_gaps.begin(), sorted_gaps.end(),
                [](const GapInfo& a, const GapInfo& b) { return a.length > b.length; });

            size_t show_count = std::min(size_t(5), sorted_gaps.size());
            for (size_t i = 0; i < show_count; ++i) {
                const auto& gap = sorted_gaps[i];
                os << "    " << gap.chrom << ":" << gap.start << "-" << gap.end
                   << " (" << (gap.length / 1000) << " kb, "
                   << gap.ref_markers_in_gap << " ref markers available)\n";
            }
            os << "\n";
        }
    }

    // Warnings
    if (!report.warnings.empty()) {
        os << "Warnings:\n";
        for (const auto& w : report.warnings) {
            os << "  ! " << w << "\n";
        }
        os << "\n";
    }

    // Recommendations
    if (!report.recommendations.empty()) {
        os << "Recommendations:\n";
        for (const auto& r : report.recommendations) {
            os << "  * " << r << "\n";
        }
        os << "\n";
    }

    os << "Analysis completed in " << std::fixed << std::setprecision(2)
       << report.analysis_time_seconds << " seconds.\n";
    os << "================================================================================\n";
}

void MarkerOverlapAnalyzer::print_coverage_plot(
    const OverlapReport& report,
    std::ostream& os,
    int width
) const {
    os << "\nCoverage Visualization:\n";
    os << std::string(width + 15, '-') << "\n";

    for (const auto& cs : report.by_chromosome) {
        // Calculate coverage bar
        int filled = static_cast<int>(cs.overlap_rate() * width);
        int empty = width - filled;

        os << std::left << std::setw(8) << cs.chrom << " [";
        os << std::string(filled, '#');
        os << std::string(empty, '-');
        os << "] " << std::fixed << std::setprecision(1)
           << (cs.overlap_rate() * 100) << "%\n";
    }

    os << std::string(width + 15, '-') << "\n";
    os << "Legend: # = overlapping markers, - = gaps\n\n";
}

void MarkerOverlapAnalyzer::export_json(
    const OverlapReport& report,
    const std::string& filename
) const {
    std::ofstream out(filename);
    if (!out) {
        throw std::runtime_error("Failed to open file for writing: " + filename);
    }

    out << "{\n";
    out << "  \"reference_file\": \"" << report.reference_file << "\",\n";
    out << "  \"target_file\": \"" << report.target_file << "\",\n";
    out << "  \"reference_samples\": " << report.reference_samples << ",\n";
    out << "  \"target_samples\": " << report.target_samples << ",\n";
    out << "  \"total_reference_markers\": " << report.total_reference_markers << ",\n";
    out << "  \"total_target_markers\": " << report.total_target_markers << ",\n";
    out << "  \"total_overlapping_markers\": " << report.total_overlapping_markers << ",\n";
    out << "  \"overlap_rate\": " << std::fixed << std::setprecision(4)
        << report.overall_overlap_rate() << ",\n";

    out << "  \"by_chromosome\": [\n";
    for (size_t i = 0; i < report.by_chromosome.size(); ++i) {
        const auto& cs = report.by_chromosome[i];
        out << "    {\n";
        out << "      \"chrom\": \"" << cs.chrom << "\",\n";
        out << "      \"reference_markers\": " << cs.reference_markers << ",\n";
        out << "      \"target_markers\": " << cs.target_markers << ",\n";
        out << "      \"overlapping_markers\": " << cs.overlapping_markers << ",\n";
        out << "      \"overlap_rate\": " << std::fixed << std::setprecision(4)
            << cs.overlap_rate() << "\n";
        out << "    }" << (i < report.by_chromosome.size() - 1 ? "," : "") << "\n";
    }
    out << "  ],\n";

    out << "  \"analysis_time_seconds\": " << std::fixed << std::setprecision(3)
        << report.analysis_time_seconds << "\n";
    out << "}\n";
}

void MarkerOverlapAnalyzer::export_csv(
    const OverlapReport& report,
    const std::string& filename
) const {
    std::ofstream out(filename);
    if (!out) {
        throw std::runtime_error("Failed to open file for writing: " + filename);
    }

    // Header
    out << "chromosome,reference_markers,target_markers,overlapping_markers,"
        << "reference_only,target_only,overlap_rate,coverage_rate,"
        << "ref_density,target_density\n";

    // Data
    for (const auto& cs : report.by_chromosome) {
        out << cs.chrom << ","
            << cs.reference_markers << ","
            << cs.target_markers << ","
            << cs.overlapping_markers << ","
            << cs.reference_only << ","
            << cs.target_only << ","
            << std::fixed << std::setprecision(4) << cs.overlap_rate() << ","
            << std::fixed << std::setprecision(4) << cs.coverage_rate() << ","
            << std::fixed << std::setprecision(2) << cs.ref_marker_density() << ","
            << std::fixed << std::setprecision(2) << cs.target_marker_density() << "\n";
    }
}

// ============================================================================
// Factory Functions
// ============================================================================

std::unique_ptr<MarkerOverlapAnalyzer> create_overlap_analyzer(const std::string& preset) {
    OverlapConfig config;

    if (preset == "strict") {
        config.require_allele_match = true;
        config.check_strand_flips = false;
    } else if (preset == "lenient") {
        config.require_allele_match = false;
        config.check_strand_flips = true;
    } else if (preset == "quick") {
        config.detailed_gap_analysis = false;
        config.compute_af_correlation = false;
    }

    return std::make_unique<MarkerOverlapAnalyzer>(config);
}

OverlapReport analyze_overlap(
    const std::string& reference_vcf,
    const std::string& target_vcf,
    bool verbose
) {
    MarkerOverlapAnalyzer analyzer;

    ProgressCallback progress = nullptr;
    if (verbose) {
        progress = [](size_t completed, size_t total, const std::string& stage) {
            std::cout << "\r" << stage << " (" << completed << "/" << total << ")" << std::flush;
        };
    }

    auto report = analyzer.analyze_files(reference_vcf, target_vcf, progress);

    if (verbose) {
        std::cout << "\n";
        analyzer.print_report(report, std::cout);
    }

    return report;
}

} // namespace analysis
} // namespace swiftimpute
