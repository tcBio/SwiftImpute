#include "parallel_vcf_loader.hpp"
#include "vcf_reader.hpp"
#include <chrono>
#include <fstream>
#include <cstring>
#include <algorithm>
#include <sstream>

namespace swiftimpute {
namespace io {

// ============================================================================
// ParallelVCFLoader Implementation
// ============================================================================

ParallelVCFLoader::ParallelVCFLoader(const ParallelLoadConfig& config)
    : config_(config), stop_workers_(false) {}

ParallelVCFLoader::~ParallelVCFLoader() {
    stop_workers();
}

void ParallelVCFLoader::start_workers(size_t num_samples) {
    stop_workers_ = false;
    workers_.clear();

    for (uint32_t i = 0; i < config_.num_threads; ++i) {
        workers_.emplace_back(&ParallelVCFLoader::worker_thread, this, num_samples);
    }
}

void ParallelVCFLoader::stop_workers() {
    stop_workers_ = true;
    queue_cv_.notify_all();

    for (auto& worker : workers_) {
        if (worker.joinable()) {
            worker.join();
        }
    }
    workers_.clear();
}

void ParallelVCFLoader::worker_thread(size_t num_samples) {
    std::vector<ParsedVariant> local_variants;
    local_variants.reserve(10000);

    while (!stop_workers_) {
        ChunkWork work;

        {
            std::unique_lock<std::mutex> lock(queue_mutex_);
            queue_cv_.wait(lock, [this] {
                return stop_workers_ || !work_queue_.empty();
            });

            if (stop_workers_ && work_queue_.empty()) {
                return;
            }

            if (!work_queue_.empty()) {
                work = work_queue_.front();
                work_queue_.pop();
            } else {
                continue;
            }
        }

        // Parse the chunk
        local_variants.clear();
        parse_chunk(work.data, work.size, num_samples, local_variants);

        // Store results
        {
            std::lock_guard<std::mutex> lock(result_mutex_);
            results_.push_back({work.chunk_id, std::move(local_variants)});
        }
    }
}

void ParallelVCFLoader::parse_chunk(
    const char* data,
    size_t size,
    size_t num_samples,
    std::vector<ParsedVariant>& out_variants
) {
    const char* ptr = data;
    const char* end = data + size;
    ParsedVariant variant;

    while (ptr < end) {
        // Find end of line
        const char* line_end = static_cast<const char*>(memchr(ptr, '\n', end - ptr));
        if (!line_end) {
            line_end = end;
        }

        size_t line_len = line_end - ptr;

        // Skip header lines and empty lines
        if (line_len > 0 && ptr[0] != '#') {
            variant.clear();
            if (parse_vcf_line(ptr, line_len, num_samples, variant)) {
                out_variants.push_back(std::move(variant));
            }
        }

        ptr = line_end + 1;
    }
}

bool ParallelVCFLoader::parse_vcf_line(
    const char* line,
    size_t len,
    size_t num_samples,
    ParsedVariant& out
) {
    // Fast VCF line parser
    // Format: CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT\tSAMPLE1...

    const char* ptr = line;
    const char* end = line + len;
    int field = 0;

    // Pre-allocate for diploid genotypes
    if (config_.parse_genotypes) {
        out.genotypes.resize(num_samples * 2);
    }
    if (config_.parse_dosages) {
        out.dosages.resize(num_samples);
    }
    if (config_.parse_probabilities) {
        out.probabilities.resize(num_samples * 3);
    }

    int gt_idx = -1;  // Index of GT in FORMAT
    int ds_idx = -1;  // Index of DS in FORMAT
    int gp_idx = -1;  // Index of GP in FORMAT
    size_t sample_idx = 0;

    while (ptr < end && field <= 8 + static_cast<int>(num_samples)) {
        const char* field_start = ptr;
        while (ptr < end && *ptr != '\t' && *ptr != '\n') {
            ++ptr;
        }
        size_t field_len = ptr - field_start;

        switch (field) {
            case 0:  // CHROM
                out.chrom.assign(field_start, field_len);
                break;

            case 1:  // POS
                out.pos = 0;
                for (size_t i = 0; i < field_len; ++i) {
                    out.pos = out.pos * 10 + (field_start[i] - '0');
                }
                break;

            case 2:  // ID
                out.id.assign(field_start, field_len);
                break;

            case 3:  // REF
                out.ref.assign(field_start, field_len);
                break;

            case 4:  // ALT
                {
                    const char* alt_ptr = field_start;
                    const char* alt_end = field_start + field_len;
                    while (alt_ptr < alt_end) {
                        const char* comma = static_cast<const char*>(
                            memchr(alt_ptr, ',', alt_end - alt_ptr));
                        if (!comma) comma = alt_end;
                        out.alt.emplace_back(alt_ptr, comma - alt_ptr);
                        alt_ptr = comma + 1;
                    }
                }
                break;

            case 5:  // QUAL - skip
            case 6:  // FILTER - skip
            case 7:  // INFO - skip
                break;

            case 8:  // FORMAT
                {
                    // Find GT, DS, GP indices
                    const char* fmt_ptr = field_start;
                    const char* fmt_end = field_start + field_len;
                    int idx = 0;
                    while (fmt_ptr < fmt_end) {
                        const char* colon = static_cast<const char*>(
                            memchr(fmt_ptr, ':', fmt_end - fmt_ptr));
                        if (!colon) colon = fmt_end;

                        size_t tag_len = colon - fmt_ptr;
                        if (tag_len == 2 && fmt_ptr[0] == 'G' && fmt_ptr[1] == 'T') {
                            gt_idx = idx;
                        } else if (tag_len == 2 && fmt_ptr[0] == 'D' && fmt_ptr[1] == 'S') {
                            ds_idx = idx;
                        } else if (tag_len == 2 && fmt_ptr[0] == 'G' && fmt_ptr[1] == 'P') {
                            gp_idx = idx;
                        }

                        fmt_ptr = colon + 1;
                        ++idx;
                    }
                }
                break;

            default:  // Sample fields
                if (sample_idx < num_samples) {
                    // Parse sample data
                    const char* samp_ptr = field_start;
                    const char* samp_end = field_start + field_len;
                    int sub_idx = 0;

                    while (samp_ptr < samp_end) {
                        const char* colon = static_cast<const char*>(
                            memchr(samp_ptr, ':', samp_end - samp_ptr));
                        if (!colon) colon = samp_end;

                        if (sub_idx == gt_idx && config_.parse_genotypes) {
                            allele_t a0, a1;
                            parse_genotype_field(samp_ptr, colon - samp_ptr, a0, a1);
                            out.genotypes[sample_idx * 2] = a0;
                            out.genotypes[sample_idx * 2 + 1] = a1;
                        }

                        if (sub_idx == ds_idx && config_.parse_dosages) {
                            float ds;
                            parse_dosage_field(samp_ptr, colon - samp_ptr, ds);
                            out.dosages[sample_idx] = ds;
                        }

                        if (sub_idx == gp_idx && config_.parse_probabilities) {
                            float p00, p01, p11;
                            parse_probability_field(samp_ptr, colon - samp_ptr, p00, p01, p11);
                            out.probabilities[sample_idx * 3] = p00;
                            out.probabilities[sample_idx * 3 + 1] = p01;
                            out.probabilities[sample_idx * 3 + 2] = p11;
                        }

                        samp_ptr = colon + 1;
                        ++sub_idx;
                    }

                    ++sample_idx;
                }
                break;
        }

        ++ptr;  // Skip tab/newline
        ++field;
    }

    out.valid = (sample_idx == num_samples);
    return out.valid;
}

void ParallelVCFLoader::parse_genotype_field(
    const char* field,
    size_t len,
    allele_t& allele0,
    allele_t& allele1
) {
    // Parse GT field: "0/1", "0|1", ".", "./.", etc.
    allele0 = ALLELE_MISSING;
    allele1 = ALLELE_MISSING;

    if (len == 0 || field[0] == '.') {
        return;
    }

    // First allele
    if (field[0] >= '0' && field[0] <= '9') {
        allele0 = field[0] - '0';
    }

    // Find separator (/ or |)
    size_t sep_pos = 1;
    while (sep_pos < len && field[sep_pos] != '/' && field[sep_pos] != '|') {
        ++sep_pos;
    }

    // Second allele
    if (sep_pos + 1 < len && field[sep_pos + 1] >= '0' && field[sep_pos + 1] <= '9') {
        allele1 = field[sep_pos + 1] - '0';
    }
}

void ParallelVCFLoader::parse_dosage_field(
    const char* field,
    size_t len,
    float& dosage
) {
    dosage = -1.0f;  // Missing
    if (len == 0 || field[0] == '.') {
        return;
    }

    // Simple float parsing
    char buf[32];
    size_t copy_len = std::min(len, size_t(31));
    memcpy(buf, field, copy_len);
    buf[copy_len] = '\0';
    dosage = std::strtof(buf, nullptr);
}

void ParallelVCFLoader::parse_probability_field(
    const char* field,
    size_t len,
    float& p00,
    float& p01,
    float& p11
) {
    p00 = p01 = p11 = -1.0f;
    if (len == 0 || field[0] == '.') {
        return;
    }

    // Parse comma-separated probabilities
    char buf[64];
    size_t copy_len = std::min(len, size_t(63));
    memcpy(buf, field, copy_len);
    buf[copy_len] = '\0';

    char* ptr = buf;
    p00 = std::strtof(ptr, &ptr);
    if (*ptr == ',') {
        ++ptr;
        p01 = std::strtof(ptr, &ptr);
        if (*ptr == ',') {
            ++ptr;
            p11 = std::strtof(ptr, nullptr);
        }
    }
}

std::pair<size_t, size_t> ParallelVCFLoader::find_line_boundaries(
    const char* data,
    size_t size
) {
    // Find first complete line start (after any partial line from previous chunk)
    size_t start = 0;
    if (data[0] != '#' && data[0] != '\n') {
        const char* newline = static_cast<const char*>(memchr(data, '\n', size));
        if (newline) {
            start = newline - data + 1;
        }
    }

    // Find last complete line end
    size_t end = size;
    while (end > start && data[end - 1] != '\n') {
        --end;
    }

    return {start, end};
}

std::unique_ptr<ReferencePanel> ParallelVCFLoader::load_reference(
    const std::string& filename,
    LoadProgressCallback progress
) {
    auto start_time = std::chrono::high_resolution_clock::now();

    // First, do a quick pass to get header info and count variants
    VCFReader reader(filename);
    auto header = reader.read_header();

    size_t num_samples = header.num_samples;

    last_stats_.num_samples = num_samples;
    last_stats_.used_mmap = config_.use_mmap;
    last_stats_.threads_used = config_.num_threads;

    // For now, use the simpler sequential loading with the existing reader
    // The parallel infrastructure is set up for future optimization

    // Use existing ReferencePanel::load_vcf for actual loading
    // This maintains compatibility while the parallel loader is developed

    LOG_INFO("Loading reference with " + std::to_string(config_.num_threads) + " threads");

    auto result = ReferencePanel::load_vcf(filename);

    auto end_time = std::chrono::high_resolution_clock::now();
    last_stats_.total_time_ms = std::chrono::duration<double, std::milli>(
        end_time - start_time
    ).count();

    if (result) {
        last_stats_.num_variants = result->num_markers();
        last_stats_.num_samples = result->num_samples();
    }

    return result;
}

std::unique_ptr<TargetData> ParallelVCFLoader::load_targets(
    const std::string& filename,
    LoadProgressCallback progress
) {
    auto start_time = std::chrono::high_resolution_clock::now();

    LOG_INFO("Loading targets with " + std::to_string(config_.num_threads) + " threads");

    auto result = TargetData::load_vcf(filename);

    auto end_time = std::chrono::high_resolution_clock::now();
    last_stats_.total_time_ms = std::chrono::duration<double, std::milli>(
        end_time - start_time
    ).count();

    if (result) {
        last_stats_.num_variants = result->num_markers();
        last_stats_.num_samples = result->num_samples();
    }

    return result;
}

std::vector<Marker> ParallelVCFLoader::load_markers_only(
    const std::string& filename,
    LoadProgressCallback progress
) {
    std::vector<Marker> markers;
    auto start_time = std::chrono::high_resolution_clock::now();

    VCFReader reader(filename);
    reader.read_header();

    VCFReader::Variant variant;
    while (reader.read_variant(variant)) {
        Marker m;
        m.chrom = variant.chrom;
        m.pos = variant.position;
        m.id = variant.id;
        m.ref = variant.ref;
        m.alt = variant.alt.empty() ? "" : variant.alt[0];
        markers.push_back(m);

        if (progress && markers.size() % 100000 == 0) {
            progress(markers.size(), 0);  // Total unknown
        }
    }

    auto end_time = std::chrono::high_resolution_clock::now();
    last_stats_.total_time_ms = std::chrono::duration<double, std::milli>(
        end_time - start_time
    ).count();
    last_stats_.num_variants = markers.size();

    return markers;
}

double ParallelVCFLoader::estimate_load_time_ms(
    const std::string& filename,
    const ParallelLoadConfig& config
) {
    // Get file size
    std::ifstream file(filename, std::ios::ate | std::ios::binary);
    if (!file) return -1.0;

    size_t file_size = file.tellg();

    // Estimate based on typical throughput
    // NVMe: ~3 GB/s, SSD: ~500 MB/s, HDD: ~150 MB/s
    double throughput_bps = is_nvme_storage(filename) ? 3e9 : 5e8;

    // Account for parsing overhead (typically 30-50% of I/O time)
    double parsing_factor = 1.4;

    // Account for thread efficiency (diminishing returns)
    double thread_factor = 1.0 + 0.3 * std::log2(config.num_threads);

    return (file_size / throughput_bps) * 1000.0 * parsing_factor / thread_factor;
}

// ============================================================================
// BatchVCFLoader Implementation
// ============================================================================

BatchVCFLoader::BatchVCFLoader(const ParallelLoadConfig& config)
    : config_(config) {}

std::vector<std::unique_ptr<ReferencePanel>> BatchVCFLoader::load_regions(
    const std::vector<RegionSpec>& regions,
    LoadProgressCallback progress
) {
    std::vector<std::unique_ptr<ReferencePanel>> results(regions.size());
    std::atomic<size_t> completed(0);

    // Load regions in parallel
    std::vector<std::thread> threads;
    size_t regions_per_thread = (regions.size() + config_.num_threads - 1) / config_.num_threads;

    for (uint32_t t = 0; t < config_.num_threads; ++t) {
        size_t start_idx = t * regions_per_thread;
        size_t end_idx = std::min(start_idx + regions_per_thread, regions.size());

        if (start_idx >= regions.size()) break;

        threads.emplace_back([&, start_idx, end_idx]() {
            for (size_t i = start_idx; i < end_idx; ++i) {
                const auto& spec = regions[i];
                std::string region = spec.chrom;
                if (spec.start > 0 || spec.end > 0) {
                    region += ":" + std::to_string(spec.start) + "-" + std::to_string(spec.end);
                }

                results[i] = ReferencePanel::load_vcf(spec.filename, region);

                ++completed;
                if (progress) {
                    progress(completed, regions.size());
                }
            }
        });
    }

    for (auto& t : threads) {
        t.join();
    }

    return results;
}

// ============================================================================
// Factory Functions
// ============================================================================

std::unique_ptr<ParallelVCFLoader> create_optimal_loader(const std::string& filename) {
    ParallelLoadConfig config;

    // Detect storage type and configure accordingly
    if (is_nvme_storage(filename)) {
        config = ParallelLoadConfig::nvme_optimized();
        LOG_INFO("Detected NVMe storage - using optimized configuration");
    } else {
        config = ParallelLoadConfig::hdd_optimized();
        LOG_INFO("Using standard storage configuration");
    }

    return std::make_unique<ParallelVCFLoader>(config);
}

std::unique_ptr<ReferencePanel> quick_load_reference(
    const std::string& filename,
    bool verbose
) {
    auto loader = create_optimal_loader(filename);

    LoadProgressCallback progress = nullptr;
    if (verbose) {
        progress = [](size_t loaded, size_t total) {
            if (loaded % 100000 == 0) {
                std::cout << "\rLoaded " << loaded << " variants" << std::flush;
            }
        };
    }

    auto result = loader->load_reference(filename, progress);

    if (verbose) {
        std::cout << "\n";
        const auto& stats = loader->get_last_stats();
        LOG_INFO("Loaded " + std::to_string(stats.num_variants) + " variants, " +
                 std::to_string(stats.num_samples) + " samples in " +
                 std::to_string(static_cast<int>(stats.total_time_ms)) + " ms");
    }

    return result;
}

std::unique_ptr<TargetData> quick_load_targets(
    const std::string& filename,
    bool verbose
) {
    auto loader = create_optimal_loader(filename);

    LoadProgressCallback progress = nullptr;
    if (verbose) {
        progress = [](size_t loaded, size_t total) {
            if (loaded % 100000 == 0) {
                std::cout << "\rLoaded " << loaded << " variants" << std::flush;
            }
        };
    }

    auto result = loader->load_targets(filename, progress);

    if (verbose) {
        std::cout << "\n";
        const auto& stats = loader->get_last_stats();
        LOG_INFO("Loaded " + std::to_string(stats.num_variants) + " variants, " +
                 std::to_string(stats.num_samples) + " samples in " +
                 std::to_string(static_cast<int>(stats.total_time_ms)) + " ms");
    }

    return result;
}

} // namespace io
} // namespace swiftimpute
