#include "api/imputer.hpp"
#include "core/types.hpp"
#include "phasing/pre_phaser.hpp"
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
            } else if (arg == "--help" || arg == "-h") {
                return false;
            }
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
        std::cout << "Examples:\n";
        std::cout << "  " << program_name << " -r ref.vcf.gz -t targets.vcf.gz -o imputed.vcf.gz\n";
        std::cout << "  " << program_name << " -r ref.vcf.gz -t targets.vcf.gz -o imputed.vcf.gz --chr chr22\n";
        std::cout << "  " << program_name << " -r ref.vcf.gz -t targets.vcf.gz -o imputed.vcf.gz --per-chromosome\n";
        std::cout << "  " << program_name << " -r ref.vcf.gz -t targets.vcf.gz -o imputed.vcf.gz --gpu 0 --states 16\n";
    }
};

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

        // Configure imputation
        ImputationConfig config;
        config.device_id = args.device_id;
        config.hmm_params.num_states = args.num_states;
        config.hmm_params.ne = args.ne;
        config.batch_size = args.batch_size;
        config.deterministic = args.deterministic;

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

        LOG_INFO("Done!");

        return 0;

    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
}
