#include "api/imputer.hpp"
#include "core/types.hpp"
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
    int device_id = -1;
    uint32_t num_states = 8;
    uint32_t ne = 10000;
    uint32_t batch_size = 100;
    bool deterministic = false;
    bool benchmark = false;
    bool verbose = false;
    
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
        std::cout << "  " << program_name << " -r ref.vcf.gz -t targets.vcf.gz -o imputed.vcf.gz --region chr20:1-10000000\n";
        std::cout << "  " << program_name << " -r ref.vcf.gz -t targets.vcf.gz -o imputed.vcf.gz --gpu 0 --states 16\n";
    }
};

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
        
        // Create imputer
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
            *targets,
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
        LOG_INFO("Done!");
        
        return 0;
        
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
}
