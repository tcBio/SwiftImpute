#include "api/imputer.hpp"
#include "core/types.hpp"
#include "phasing/pre_phaser.hpp"
#include "phasing/gpu_phaser.cuh"
#include "analysis/marker_overlap.hpp"
#include "analysis/qc_filter.hpp"
#include "io/parallel_vcf_loader.hpp"
#include <iostream>
#include <string>
#include <chrono>
#include <iomanip>

using namespace swiftimpute;

// Command-line argument parser
struct CommandLineArgs {
    std::string reference_vcf;
    std::string target_vcf;
    std::string output_vcf;
    std::string region;
    std::string chromosome;           // Process single chromosome
    int device_id = -1;
    uint32_t num_states = 8;
    uint32_t ne = 10000;
    uint32_t batch_size = 100;
    bool deterministic = false;
    bool benchmark = false;
    bool verbose = false;
    bool per_chromosome = false;      // Process each chromosome separately
    bool skip_prephase = false;       // Skip pre-phasing (assume input is phased)

    // Configuration presets
    std::string preset;               // radseq, small-ref, biobank, low-memory, high-accuracy

    // Analysis options
    bool analyze_overlap = false;     // Run marker overlap analysis
    bool run_qc = false;              // Run QC filtering
    double min_info_score = 0.8;      // Minimum INFO score for QC
    std::string qc_output;            // QC report output file
    bool interactive = false;         // Interactive mode with UI

    // GPU acceleration options
    bool gpu_phasing = true;          // Use GPU for phasing (default: on)
    bool parallel_load = true;        // Use parallel VCF loading (default: on)
    uint32_t load_threads = 0;        // Threads for loading (0 = auto)

    bool parse(int argc, char* argv[]) {
        for (int i = 1; i < argc; i++) {
            std::string arg = argv[i];

            if (arg == "--reference" || arg == "-r") {
                if (++i < argc) reference_vcf = argv[i];
            } else if (arg == "--targets" || arg == "-t") {
                if (++i < argc) target_vcf = argv[i];
            } else if (arg == "--output" || arg == "-o") {
                if (++i < argc) output_vcf = argv[i];
            } else if (arg == "--region") {
                if (++i < argc) region = argv[i];
            } else if (arg == "--chromosome" || arg == "--chr") {
                if (++i < argc) chromosome = argv[i];
            } else if (arg == "--per-chromosome") {
                per_chromosome = true;
            } else if (arg == "--gpu" || arg == "-g") {
                if (++i < argc) device_id = std::stoi(argv[i]);
            } else if (arg == "--states" || arg == "-s") {
                if (++i < argc) num_states = std::stoul(argv[i]);
            } else if (arg == "--ne") {
                if (++i < argc) ne = std::stoul(argv[i]);
            } else if (arg == "--batch-size") {
                if (++i < argc) batch_size = std::stoul(argv[i]);
            } else if (arg == "--deterministic") {
                deterministic = true;
            } else if (arg == "--benchmark") {
                benchmark = true;
            } else if (arg == "--verbose" || arg == "-v") {
                verbose = true;
            } else if (arg == "--no-prephase" || arg == "--skip-prephase") {
                skip_prephase = true;
            } else if (arg == "--preset") {
                if (++i < argc) preset = argv[i];
            } else if (arg == "--analyze-overlap" || arg == "--overlap") {
                analyze_overlap = true;
            } else if (arg == "--qc" || arg == "--filter") {
                run_qc = true;
            } else if (arg == "--min-info") {
                if (++i < argc) min_info_score = std::stod(argv[i]);
            } else if (arg == "--qc-output") {
                if (++i < argc) qc_output = argv[i];
            } else if (arg == "--interactive" || arg == "-i") {
                interactive = true;
            } else if (arg == "--no-gpu-phasing" || arg == "--cpu-phasing") {
                gpu_phasing = false;
            } else if (arg == "--no-parallel-load") {
                parallel_load = false;
            } else if (arg == "--load-threads") {
                if (++i < argc) load_threads = std::stoul(argv[i]);
            } else if (arg == "--help" || arg == "-h") {
                return false;
            }
        }

        // For overlap-only analysis, we don't need output
        if (analyze_overlap && output_vcf.empty() && !reference_vcf.empty() && !target_vcf.empty()) {
            return true;
        }

        return !reference_vcf.empty() && !target_vcf.empty() && !output_vcf.empty();
    }

    static void print_usage(const char* program_name) {
        std::cout << "SwiftImpute - GPU-Accelerated Genomic Imputation\n\n";
        std::cout << "Usage: " << program_name << " [options]\n\n";
        std::cout << "Required arguments:\n";
        std::cout << "  -r, --reference FILE    Reference panel VCF file\n";
        std::cout << "  -t, --targets FILE      Target samples VCF file\n";
        std::cout << "  -o, --output FILE       Output VCF file\n\n";
        std::cout << "Optional arguments:\n";
        std::cout << "  --region REGION         Genomic region (chr:start-end)\n";
        std::cout << "  --chr, --chromosome CHR Process only specified chromosome\n";
        std::cout << "  --per-chromosome        Process each chromosome separately\n";
        std::cout << "  --no-prephase           Skip pre-phasing (assume input is phased)\n";
        std::cout << "  -g, --gpu ID            GPU device ID (-1 for auto-select)\n";
        std::cout << "  -s, --states N          Number of HMM states [default: 8]\n";
        std::cout << "  --ne N                  Effective population size [default: 10000]\n";
        std::cout << "  --batch-size N          Samples per GPU batch [default: 100]\n";
        std::cout << "  --deterministic         Use deterministic mode (no sampling)\n";
        std::cout << "  --benchmark             Run in benchmark mode\n";
        std::cout << "  -v, --verbose           Verbose output\n";
        std::cout << "  -h, --help              Show this help message\n\n";
        std::cout << "Configuration presets:\n";
        std::cout << "  --preset NAME           Use preset configuration:\n";
        std::cout << "                          - radseq: Optimized for RAD-seq data (sparse markers)\n";
        std::cout << "                          - small-ref: For reference panels < 1000 samples\n";
        std::cout << "                          - biobank: For large biobank-scale data\n";
        std::cout << "                          - low-memory: Minimize memory usage\n";
        std::cout << "                          - high-accuracy: Maximum accuracy (slower)\n\n";
        std::cout << "Analysis options:\n";
        std::cout << "  --overlap, --analyze-overlap\n";
        std::cout << "                          Analyze marker overlap before imputation\n";
        std::cout << "  --qc, --filter          Apply QC filtering to output\n";
        std::cout << "  --min-info SCORE        Minimum INFO score for QC [default: 0.8]\n";
        std::cout << "  --qc-output FILE        Write QC report to file\n";
        std::cout << "  -i, --interactive       Interactive mode with analysis UI\n\n";
        std::cout << "Performance options:\n";
        std::cout << "  --no-gpu-phasing        Use CPU phasing instead of GPU [default: GPU]\n";
        std::cout << "  --no-parallel-load      Disable parallel VCF loading [default: parallel]\n";
        std::cout << "  --load-threads N        Number of threads for loading [default: auto]\n\n";
        std::cout << "Examples:\n";
        std::cout << "  # Basic imputation\n";
        std::cout << "  " << program_name << " -r ref.vcf.gz -t targets.vcf.gz -o imputed.vcf.gz\n\n";
        std::cout << "  # RAD-seq data with overlap analysis\n";
        std::cout << "  " << program_name << " -r ref.vcf.gz -t radseq.vcf.gz -o imputed.vcf.gz --preset radseq --overlap\n\n";
        std::cout << "  # Imputation with QC filtering\n";
        std::cout << "  " << program_name << " -r ref.vcf.gz -t targets.vcf.gz -o imputed.vcf.gz --qc --min-info 0.9\n\n";
        std::cout << "  # Overlap analysis only (no imputation)\n";
        std::cout << "  " << program_name << " -r ref.vcf.gz -t targets.vcf.gz --overlap\n";
    }
};

// Helper function to apply preset configuration
ImputationConfig apply_preset(const std::string& preset, const CommandLineArgs& args) {
    ImputationConfig config;

    if (preset == "radseq") {
        config = ImputationConfig::radseq_preset();
        LOG_INFO("Using RAD-seq preset configuration");
    } else if (preset == "small-ref") {
        config = ImputationConfig::small_reference_preset();
        LOG_INFO("Using small reference panel preset");
    } else if (preset == "biobank") {
        config = ImputationConfig::biobank_preset();
        LOG_INFO("Using biobank-scale preset");
    } else if (preset == "low-memory") {
        config = ImputationConfig::low_memory_preset();
        LOG_INFO("Using low-memory preset");
    } else if (preset == "high-accuracy") {
        config = ImputationConfig::high_accuracy_preset();
        LOG_INFO("Using high-accuracy preset");
    } else if (!preset.empty()) {
        LOG_WARNING("Unknown preset '" + preset + "', using default configuration");
    }

    // Override with explicit command-line arguments
    config.device_id = args.device_id;
    if (args.num_states != 8) {  // Non-default value specified
        config.hmm_params.num_states = args.num_states;
    }
    if (args.ne != 10000) {  // Non-default value specified
        config.hmm_params.ne = args.ne;
    }
    if (args.batch_size != 100) {  // Non-default value specified
        config.batch_size = args.batch_size;
    }
    config.deterministic = args.deterministic;

    return config;
}

// Interactive UI for overlap analysis
void run_interactive_analysis(
    const analysis::OverlapReport& report,
    const CommandLineArgs& args
) {
    analysis::MarkerOverlapAnalyzer analyzer;

    std::cout << "\n";
    std::cout << "================================================================================\n";
    std::cout << "              SwiftImpute - Interactive Analysis Mode\n";
    std::cout << "================================================================================\n\n";

    // Print summary
    analyzer.print_report(report, std::cout);

    // Print coverage visualization
    analyzer.print_coverage_plot(report, std::cout);

    // Interactive menu
    bool running = true;
    while (running) {
        std::cout << "\n";
        std::cout << "Options:\n";
        std::cout << "  [1] Export report to JSON\n";
        std::cout << "  [2] Export report to CSV\n";
        std::cout << "  [3] Show detailed gap analysis\n";
        std::cout << "  [4] Show recommended configuration\n";
        std::cout << "  [5] Continue with imputation\n";
        std::cout << "  [q] Quit\n";
        std::cout << "\nChoice: ";

        std::string choice;
        std::getline(std::cin, choice);

        if (choice == "1") {
            std::string filename;
            std::cout << "Enter JSON filename [overlap_report.json]: ";
            std::getline(std::cin, filename);
            if (filename.empty()) filename = "overlap_report.json";

            try {
                analyzer.export_json(report, filename);
                std::cout << "Report exported to " << filename << "\n";
            } catch (const std::exception& e) {
                std::cout << "Error: " << e.what() << "\n";
            }
        }
        else if (choice == "2") {
            std::string filename;
            std::cout << "Enter CSV filename [overlap_report.csv]: ";
            std::getline(std::cin, filename);
            if (filename.empty()) filename = "overlap_report.csv";

            try {
                analyzer.export_csv(report, filename);
                std::cout << "Report exported to " << filename << "\n";
            } catch (const std::exception& e) {
                std::cout << "Error: " << e.what() << "\n";
            }
        }
        else if (choice == "3") {
            std::cout << "\nDetailed Gap Analysis:\n";
            std::cout << "----------------------\n";

            if (report.large_gaps.empty()) {
                std::cout << "No large gaps detected (threshold: 100kb)\n";
            } else {
                std::cout << std::left << std::setw(12) << "Chromosome"
                          << std::right << std::setw(15) << "Start"
                          << std::setw(15) << "End"
                          << std::setw(12) << "Size (kb)"
                          << std::setw(12) << "Ref Avail\n";
                std::cout << std::string(66, '-') << "\n";

                for (const auto& gap : report.large_gaps) {
                    std::cout << std::left << std::setw(12) << gap.chrom
                              << std::right << std::setw(15) << gap.start
                              << std::setw(15) << gap.end
                              << std::setw(12) << (gap.length / 1000)
                              << std::setw(12) << gap.ref_markers_in_gap << "\n";
                }
            }
        }
        else if (choice == "4") {
            std::cout << "\nRecommended Configuration:\n";
            std::cout << "--------------------------\n";

            // Determine recommended preset
            std::string rec_preset = "default";
            double target_density = 0;
            for (const auto& cs : report.by_chromosome) {
                target_density += cs.target_marker_density();
            }
            if (!report.by_chromosome.empty()) {
                target_density /= report.by_chromosome.size();
            }

            if (target_density < 100) {
                rec_preset = "radseq";
                std::cout << "Detected sparse target markers (RAD-seq-like)\n";
            }
            if (report.reference_samples < 500) {
                rec_preset = "small-ref";
                std::cout << "Detected small reference panel\n";
            }

            std::cout << "\nSuggested command:\n";
            std::cout << "  swiftimpute -r " << report.reference_file
                      << " -t " << report.target_file
                      << " -o imputed.vcf.gz";
            if (rec_preset != "default") {
                std::cout << " --preset " << rec_preset;
            }
            std::cout << "\n";
        }
        else if (choice == "5") {
            running = false;
        }
        else if (choice == "q" || choice == "Q") {
            std::cout << "Exiting.\n";
            exit(0);
        }
    }
}

// Run QC filtering after imputation
void run_qc_filtering(
    const std::string& imputed_vcf,
    const std::string& output_vcf,
    double min_info,
    const std::string& qc_report_file,
    bool verbose
) {
    LOG_INFO("Running QC filtering on " + imputed_vcf);

    analysis::QCConfig qc_config;
    qc_config.min_info_score = min_info;
    qc_config.filter_by_info = true;

    analysis::QCFilter filter(qc_config);

    auto progress = [verbose](size_t completed, size_t total) {
        if (verbose && completed % 50000 == 0) {
            std::cout << "\rQC progress: " << completed << " variants" << std::flush;
        }
    };

    auto summary = filter.filter_vcf(imputed_vcf, output_vcf, progress);

    if (verbose) {
        std::cout << "\n";
        filter.print_summary(summary, std::cout);
        filter.print_info_histogram(summary, std::cout);
    }

    // Export QC report if requested
    if (!qc_report_file.empty()) {
        filter.export_variant_qc(qc_report_file);
        LOG_INFO("QC report written to " + qc_report_file);
    }
}

// Helper function to process a single chromosome
void process_chromosome(
    const ReferencePanel& ref_chrom,
    const TargetData& target_chrom,
    const std::string& output_path,
    const ImputationConfig& config,
    const CommandLineArgs& args,
    const std::string& chrom_name
) {
    LOG_INFO("Processing chromosome: " + chrom_name);
    LOG_INFO("  Reference: " + std::to_string(ref_chrom.num_markers()) + " markers, " +
             std::to_string(ref_chrom.num_haplotypes()) + " haplotypes");
    LOG_INFO("  Target: " + std::to_string(target_chrom.num_markers()) + " markers, " +
             std::to_string(target_chrom.num_samples()) + " samples");

    auto start_time = std::chrono::high_resolution_clock::now();

    // Pre-phasing step (if not skipped)
    const TargetData* targets_to_use = &target_chrom;
    std::unique_ptr<TargetData> phased_targets;

    if (!args.skip_prephase) {
        // Check if phasing is needed
        auto phase_status = phasing::PrePhaser::detect_phase_status(target_chrom);

        if (!phase_status.is_fully_phased) {
            LOG_INFO("  Detected unphased genotypes (" +
                     std::to_string(100.0 * (1.0 - phase_status.phased_fraction)) +
                     "% unphased)");
            LOG_INFO("  Running pre-phasing...");

            phasing::PrePhasingConfig prephase_config;
            prephase_config.num_states = config.hmm_params.num_states;
            prephase_config.ne = config.hmm_params.ne;
            prephase_config.verbose = args.verbose;

            phasing::PrePhaser prephaser(ref_chrom, prephase_config);
            phased_targets = prephaser.phase(target_chrom);

            if (phased_targets) {
                targets_to_use = phased_targets.get();
                LOG_INFO("  Pre-phasing complete");
            } else {
                LOG_INFO("  Pre-phasing returned null - using original targets");
            }
        } else {
            LOG_INFO("  Target data is fully phased - skipping pre-phasing");
        }
    } else {
        LOG_INFO("  Pre-phasing skipped (--no-prephase)");
    }

    // Create imputer for this chromosome
    Imputer imputer(ref_chrom, config);

    // Build PBWT index
    LOG_INFO("  Building PBWT index...");
    imputer.build_index();

    // Run imputation
    LOG_INFO("  Running imputation...");
    auto result = imputer.impute_with_progress(
        *targets_to_use,
        [&chrom_name](uint32_t completed, uint32_t total) {
            if (completed % 100 == 0 || completed == total) {
                double percent = 100.0 * completed / total;
                std::cout << "\r  " << chrom_name << " Progress: "
                          << std::fixed << std::setprecision(1)
                          << percent << "% (" << completed << "/" << total << ")"
                          << std::flush;
            }
        }
    );
    std::cout << std::endl;

    // Write output
    LOG_INFO("  Writing output: " + output_path);
    result->write_vcf(output_path, target_chrom, ref_chrom, config);

    auto end_time = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(
        end_time - start_time
    ).count();

    LOG_INFO("  Chromosome " + chrom_name + " completed in " + std::to_string(duration) + " ms");
}

int main(int argc, char* argv[]) {
    try {
        // Parse command-line arguments
        CommandLineArgs args;
        if (!args.parse(argc, argv)) {
            CommandLineArgs::print_usage(argv[0]);
            return args.reference_vcf.empty() ? 0 : 1;
        }

        // Set logging level
        if (args.verbose) {
            Logger::instance().set_level(Logger::DEBUG);
        }

        LOG_INFO("SwiftImpute - GPU-Accelerated Genomic Imputation");
        LOG_INFO("================================================");

        // Run overlap analysis if requested (can run without loading full data)
        if (args.analyze_overlap) {
            LOG_INFO("Running marker overlap analysis...");
            auto start_time = std::chrono::high_resolution_clock::now();

            analysis::MarkerOverlapAnalyzer analyzer;
            auto progress = args.verbose ?
                [](size_t completed, size_t total, const std::string& stage) {
                    std::cout << "\r" << stage << " (" << completed << "/" << total << ")" << std::flush;
                } : analysis::ProgressCallback(nullptr);

            auto overlap_report = analyzer.analyze_files(
                args.reference_vcf, args.target_vcf, progress
            );

            if (args.verbose) {
                std::cout << "\n";
            }

            auto end_time = std::chrono::high_resolution_clock::now();
            auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(
                end_time - start_time
            ).count();
            LOG_INFO("Overlap analysis completed in " + std::to_string(duration) + " ms");

            // If interactive mode, show UI
            if (args.interactive) {
                run_interactive_analysis(overlap_report, args);
            } else {
                // Print summary
                analyzer.print_report(overlap_report, std::cout);
                analyzer.print_coverage_plot(overlap_report, std::cout);
            }

            // If no output specified, exit after analysis
            if (args.output_vcf.empty()) {
                LOG_INFO("Overlap analysis complete. No output file specified, exiting.");
                return 0;
            }
        }

        // Load reference panel
        LOG_INFO("Loading reference panel: " + args.reference_vcf);
        auto start_time = std::chrono::high_resolution_clock::now();

        auto reference = ReferencePanel::load_vcf(args.reference_vcf, args.region);

        auto load_time = std::chrono::high_resolution_clock::now();
        auto load_duration = std::chrono::duration_cast<std::chrono::milliseconds>(
            load_time - start_time
        ).count();

        LOG_INFO("Loaded " + std::to_string(reference->num_samples()) + " samples, " +
                 std::to_string(reference->num_markers()) + " markers in " +
                 std::to_string(load_duration) + " ms");

        // Load target data
        LOG_INFO("Loading target samples: " + args.target_vcf);
        start_time = std::chrono::high_resolution_clock::now();

        auto targets = TargetData::load_vcf(args.target_vcf, args.region);

        load_time = std::chrono::high_resolution_clock::now();
        load_duration = std::chrono::duration_cast<std::chrono::milliseconds>(
            load_time - start_time
        ).count();

        LOG_INFO("Loaded " + std::to_string(targets->num_samples()) + " samples, " +
                 std::to_string(targets->num_markers()) + " markers in " +
                 std::to_string(load_duration) + " ms");

        // Configure imputation (using presets if specified)
        ImputationConfig config = apply_preset(args.preset, args);

        // Select GPU
        if (config.device_id < 0) {
            config.device_id = select_best_device();
            LOG_INFO("Auto-selected GPU " + std::to_string(config.device_id));
        }

        DeviceInfo dev_info = get_device_info(config.device_id);
        LOG_INFO("Using GPU: " + dev_info.name);
        LOG_INFO("  Compute capability: " +
                 std::to_string(dev_info.compute_capability_major) + "." +
                 std::to_string(dev_info.compute_capability_minor));
        LOG_INFO("  Total memory: " +
                 std::to_string(dev_info.total_memory / (1024*1024*1024)) + " GB");

        // Get chromosomes in the data
        auto ref_chroms = reference->get_chromosomes();
        auto target_chroms = targets->get_chromosomes();

        LOG_INFO("Reference chromosomes: " + std::to_string(ref_chroms.size()));
        LOG_INFO("Target chromosomes: " + std::to_string(target_chroms.size()));

        // Determine processing mode
        bool multi_chrom_mode = false;
        std::vector<std::string> chroms_to_process;

        if (!args.chromosome.empty()) {
            // Single chromosome specified
            chroms_to_process.push_back(args.chromosome);
            LOG_INFO("Processing single chromosome: " + args.chromosome);
        } else if (args.per_chromosome || ref_chroms.size() > 1) {
            // Per-chromosome mode (explicit or auto-detected)
            multi_chrom_mode = true;

            // Find common chromosomes
            for (const auto& rc : ref_chroms) {
                for (const auto& tc : target_chroms) {
                    if (rc == tc) {
                        chroms_to_process.push_back(rc);
                        break;
                    }
                }
            }

            if (chroms_to_process.empty()) {
                throw ImputationError("No common chromosomes found between reference and target");
            }

            LOG_INFO("Multi-chromosome mode: processing " +
                     std::to_string(chroms_to_process.size()) + " chromosomes");
        }

        // Process based on mode
        if (multi_chrom_mode || !args.chromosome.empty()) {
            // Per-chromosome processing
            auto overall_start = std::chrono::high_resolution_clock::now();

            for (size_t i = 0; i < chroms_to_process.size(); ++i) {
                const auto& chrom = chroms_to_process[i];

                LOG_INFO("");
                LOG_INFO("========================================");
                LOG_INFO("Chromosome " + std::to_string(i + 1) + "/" +
                         std::to_string(chroms_to_process.size()) + ": " + chrom);
                LOG_INFO("========================================");

                // Filter to this chromosome
                auto ref_chrom = reference->filter_chromosome(chrom);
                auto target_chrom = targets->filter_chromosome(chrom);

                // Generate output filename for this chromosome
                std::string output_path;
                if (chroms_to_process.size() == 1) {
                    output_path = args.output_vcf;
                } else {
                    // Insert chromosome into filename: output.vcf.gz -> output.chr1.vcf.gz
                    size_t ext_pos = args.output_vcf.rfind(".vcf");
                    if (ext_pos != std::string::npos) {
                        output_path = args.output_vcf.substr(0, ext_pos) + "." +
                                      chrom + args.output_vcf.substr(ext_pos);
                    } else {
                        output_path = args.output_vcf + "." + chrom;
                    }
                }

                // Process this chromosome
                process_chromosome(*ref_chrom, *target_chrom, output_path,
                                   config, args, chrom);
            }

            auto overall_end = std::chrono::high_resolution_clock::now();
            auto overall_duration = std::chrono::duration_cast<std::chrono::milliseconds>(
                overall_end - overall_start
            ).count();

            LOG_INFO("");
            LOG_INFO("========================================");
            LOG_INFO("All chromosomes completed in " + std::to_string(overall_duration) + " ms");
            LOG_INFO("========================================");

        } else {
            // Single-pass processing (original behavior for single-chromosome data)

            // Pre-phasing step (if not skipped)
            const TargetData* targets_to_use = targets.get();
            std::unique_ptr<TargetData> phased_targets;

            if (!args.skip_prephase) {
                // Check if phasing is needed
                auto phase_status = phasing::PrePhaser::detect_phase_status(*targets);

                if (!phase_status.is_fully_phased) {
                    LOG_INFO("Detected unphased genotypes (" +
                             std::to_string(100.0 * (1.0 - phase_status.phased_fraction)) +
                             "% unphased)");
                    LOG_INFO("Running pre-phasing...");
                    start_time = std::chrono::high_resolution_clock::now();

                    phasing::PrePhasingConfig prephase_config;
                    prephase_config.num_states = config.hmm_params.num_states;
                    prephase_config.ne = config.hmm_params.ne;
                    prephase_config.verbose = args.verbose;

                    phasing::PrePhaser prephaser(*reference, prephase_config);
                    phased_targets = prephaser.phase(*targets);

                    auto prephase_time = std::chrono::high_resolution_clock::now();
                    auto prephase_duration = std::chrono::duration_cast<std::chrono::milliseconds>(
                        prephase_time - start_time
                    ).count();

                    if (phased_targets) {
                        targets_to_use = phased_targets.get();
                        LOG_INFO("Pre-phasing complete in " + std::to_string(prephase_duration) + " ms");
                    } else {
                        LOG_INFO("Pre-phasing returned null - using original targets");
                    }
                } else {
                    LOG_INFO("Target data is fully phased - skipping pre-phasing");
                }
            } else {
                LOG_INFO("Pre-phasing skipped (--no-prephase)");
            }

            LOG_INFO("Initializing imputer...");
            Imputer imputer(*reference, config);

            // Build PBWT index
            LOG_INFO("Building PBWT index...");
            start_time = std::chrono::high_resolution_clock::now();

            imputer.build_index();

            auto index_time = std::chrono::high_resolution_clock::now();
            auto index_duration = std::chrono::duration_cast<std::chrono::milliseconds>(
                index_time - start_time
            ).count();

            LOG_INFO("PBWT index built in " + std::to_string(index_duration) + " ms");

            // Run imputation
            LOG_INFO("Starting imputation...");
            start_time = std::chrono::high_resolution_clock::now();

            auto result = imputer.impute_with_progress(
                *targets_to_use,
                [](uint32_t completed, uint32_t total) {
                    if (completed % 100 == 0 || completed == total) {
                        double percent = 100.0 * completed / total;
                        std::cout << "\rProgress: " << std::fixed << std::setprecision(1)
                                  << percent << "% (" << completed << "/" << total << ")"
                                  << std::flush;
                    }
                }
            );
            std::cout << std::endl;

            auto impute_time = std::chrono::high_resolution_clock::now();
            auto impute_duration = std::chrono::duration_cast<std::chrono::milliseconds>(
                impute_time - start_time
            ).count();

            LOG_INFO("Imputation completed in " + std::to_string(impute_duration) + " ms");

            // Performance metrics
            if (args.benchmark) {
                double samples_per_sec = 1000.0 * targets->num_samples() / impute_duration;
                double markers_per_sec = 1000.0 * targets->num_markers() / impute_duration;

                LOG_INFO("Performance:");
                LOG_INFO("  Samples/sec: " + std::to_string(samples_per_sec));
                LOG_INFO("  Markers/sec: " + std::to_string(markers_per_sec));
                LOG_INFO("  GPU memory: " +
                         std::to_string(imputer.device_memory_usage() / (1024*1024)) + " MB");
            }

            // Write output
            LOG_INFO("Writing output: " + args.output_vcf);
            start_time = std::chrono::high_resolution_clock::now();

            result->write_vcf(args.output_vcf, *targets, *reference, config);

            auto write_time = std::chrono::high_resolution_clock::now();
            auto write_duration = std::chrono::duration_cast<std::chrono::milliseconds>(
                write_time - start_time
            ).count();

            LOG_INFO("Output written in " + std::to_string(write_duration) + " ms");
        }

        // Run QC filtering if requested
        if (args.run_qc && !args.output_vcf.empty()) {
            std::string qc_output = args.output_vcf;
            // Generate QC'd filename: output.vcf.gz -> output.qc.vcf.gz
            size_t ext_pos = qc_output.rfind(".vcf");
            if (ext_pos != std::string::npos) {
                qc_output = qc_output.substr(0, ext_pos) + ".qc" + qc_output.substr(ext_pos);
            } else {
                qc_output = qc_output + ".qc";
            }

            run_qc_filtering(
                args.output_vcf,
                qc_output,
                args.min_info_score,
                args.qc_output,
                args.verbose
            );

            LOG_INFO("QC-filtered output: " + qc_output);
        }

        LOG_INFO("Done!");

        return 0;

    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
}
