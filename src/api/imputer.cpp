#include "imputer.hpp"
#include "../io/vcf_reader.hpp"
#include "../io/vcf_writer.hpp"
#include <algorithm>
#include <cmath>
#include <iostream>

namespace swiftimpute {

// ReferencePanel implementation

ReferencePanel::ReferencePanel(
    std::vector<Marker> markers,
    std::vector<Sample> samples,
    haplotype_t num_haplotypes,
    std::unique_ptr<allele_t[]> haplotypes
) : markers_(std::move(markers)),
    samples_(std::move(samples)),
    num_haplotypes_(num_haplotypes),
    haplotypes_(std::move(haplotypes))
{}

std::unique_ptr<ReferencePanel> ReferencePanel::load_vcf(
    const std::string& filename,
    const std::string& region
) {
    LOG_INFO("Loading reference panel from: " + filename);

    VCFReader reader(filename);
    const auto& header = reader.read_header();

    LOG_INFO("Reference panel: " + std::to_string(header.num_samples) + " samples");

    // Read all variants
    std::vector<VCFReader::Variant> variants;
    if (region.empty()) {
        variants = reader.read_all_variants();
    } else {
        variants = reader.read_region(region);
    }

    reader.close();

    if (variants.empty()) {
        throw ImputationError("No variants found in reference VCF");
    }

    LOG_INFO("Read " + std::to_string(variants.size()) + " variants");

    // Build markers
    std::vector<Marker> markers;
    markers.reserve(variants.size());

    for (const auto& v : variants) {
        Marker m;
        m.chrom = v.chrom;
        m.pos = v.position;
        m.id = v.id;
        m.ref = v.ref;
        m.alt = v.alt.empty() ? "" : v.alt[0];  // Take first ALT allele
        m.cM = 0.0;  // Will be computed later if genetic map provided
        markers.push_back(m);
    }

    // Build samples
    std::vector<Sample> samples;
    samples.reserve(header.num_samples);

    for (size_t i = 0; i < header.num_samples; ++i) {
        Sample s(header.sample_names[i]);
        s.index = static_cast<uint32_t>(i);
        samples.push_back(s);
    }

    // Allocate haplotype array
    haplotype_t num_haplotypes = static_cast<haplotype_t>(header.num_samples * 2);
    size_t total_size = static_cast<size_t>(markers.size()) * num_haplotypes;
    auto haplotypes = std::make_unique<allele_t[]>(total_size);

    // Fill haplotype data [marker][haplotype]
    for (size_t m = 0; m < variants.size(); ++m) {
        const auto& variant = variants[m];

        for (size_t s = 0; s < header.num_samples; ++s) {
            if (s >= variant.genotypes.size()) {
                // Missing sample data
                haplotypes[m * num_haplotypes + s * 2 + 0] = ALLELE_MISSING;
                haplotypes[m * num_haplotypes + s * 2 + 1] = ALLELE_MISSING;
                continue;
            }

            const auto& gt = variant.genotypes[s];

            if (gt.size() >= 2) {
                // Diploid genotype
                haplotypes[m * num_haplotypes + s * 2 + 0] = gt[0];
                haplotypes[m * num_haplotypes + s * 2 + 1] = gt[1];
            } else if (gt.size() == 1) {
                // Haploid or homozygous
                haplotypes[m * num_haplotypes + s * 2 + 0] = gt[0];
                haplotypes[m * num_haplotypes + s * 2 + 1] = gt[0];
            } else {
                // Missing
                haplotypes[m * num_haplotypes + s * 2 + 0] = ALLELE_MISSING;
                haplotypes[m * num_haplotypes + s * 2 + 1] = ALLELE_MISSING;
            }
        }
    }

    LOG_INFO("Reference panel loaded: " +
             std::to_string(markers.size()) + " markers, " +
             std::to_string(num_haplotypes) + " haplotypes");

    return std::unique_ptr<ReferencePanel>(new ReferencePanel(
        std::move(markers),
        std::move(samples),
        num_haplotypes,
        std::move(haplotypes)
    ));
}

std::unique_ptr<ReferencePanel> ReferencePanel::load_binary(
    const std::string& filename
) {
    // TODO: Implement binary format loading
    throw ImputationError("Binary reference format not yet implemented");
}

void ReferencePanel::save_binary(const std::string& filename) const {
    // TODO: Implement binary format saving
    throw ImputationError("Binary reference format not yet implemented");
}

allele_t ReferencePanel::get_allele(marker_t m, haplotype_t h) const {
    if (m >= markers_.size() || h >= num_haplotypes_) {
        return ALLELE_MISSING;
    }
    return haplotypes_[m * num_haplotypes_ + h];
}

size_t ReferencePanel::memory_usage() const {
    size_t total = 0;

    // Marker data
    total += markers_.capacity() * sizeof(Marker);
    for (const auto& m : markers_) {
        total += m.chrom.capacity() + m.id.capacity() + m.ref.capacity() + m.alt.capacity();
    }

    // Sample data
    total += samples_.capacity() * sizeof(Sample);
    for (const auto& s : samples_) {
        total += s.id.capacity();
    }

    // Haplotype data
    total += static_cast<size_t>(markers_.size()) * num_haplotypes_;

    return total;
}

// TargetData implementation

TargetData::TargetData(
    std::vector<Marker> markers,
    std::vector<Sample> samples,
    std::unique_ptr<GenotypeLikelihoods[]> genotype_liks
) : markers_(std::move(markers)),
    samples_(std::move(samples)),
    genotype_liks_(std::move(genotype_liks))
{}

std::unique_ptr<TargetData> TargetData::load_vcf(
    const std::string& filename,
    const std::string& region
) {
    LOG_INFO("Loading target data from: " + filename);

    VCFReader reader(filename);
    const auto& header = reader.read_header();

    LOG_INFO("Target samples: " + std::to_string(header.num_samples));

    // Read all variants
    std::vector<VCFReader::Variant> variants;
    if (region.empty()) {
        variants = reader.read_all_variants();
    } else {
        variants = reader.read_region(region);
    }

    reader.close();

    if (variants.empty()) {
        throw ImputationError("No variants found in target VCF");
    }

    LOG_INFO("Read " + std::to_string(variants.size()) + " variants");

    // Build markers
    std::vector<Marker> markers;
    markers.reserve(variants.size());

    for (const auto& v : variants) {
        Marker m;
        m.chrom = v.chrom;
        m.pos = v.position;
        m.id = v.id;
        m.ref = v.ref;
        m.alt = v.alt.empty() ? "" : v.alt[0];
        m.cM = 0.0;
        markers.push_back(m);
    }

    // Build samples
    std::vector<Sample> samples;
    samples.reserve(header.num_samples);

    for (size_t i = 0; i < header.num_samples; ++i) {
        Sample s(header.sample_names[i]);
        s.index = static_cast<uint32_t>(i);
        samples.push_back(s);
    }

    // Allocate genotype likelihood array [sample][marker]
    size_t total_size = header.num_samples * variants.size();
    auto genotype_liks = std::make_unique<GenotypeLikelihoods[]>(total_size);

    // Convert observed genotypes to genotype likelihoods
    // For perfect data (hard calls), we use:
    //   P(D|AA) = 1.0 if GT=0/0, else 0.0
    //   P(D|AB) = 1.0 if GT=0/1, else 0.0
    //   P(D|BB) = 1.0 if GT=1/1, else 0.0
    //
    // In log10 scale:
    //   LL_AA = 0.0 if GT=0/0, else -999.0 (essentially 0 probability)
    //   LL_AB = 0.0 if GT=0/1, else -999.0
    //   LL_BB = 0.0 if GT=1/1, else -999.0

    const prob_t LOG10_ZERO = -999.0f;
    const prob_t LOG10_ONE = 0.0f;

    for (size_t m = 0; m < variants.size(); ++m) {
        const auto& variant = variants[m];

        for (size_t s = 0; s < header.num_samples; ++s) {
            size_t idx = s * variants.size() + m;

            if (s >= variant.genotypes.size()) {
                // Missing data - uniform likelihood
                genotype_liks[idx].ll_00 = LOG10_ONE;
                genotype_liks[idx].ll_01 = LOG10_ONE;
                genotype_liks[idx].ll_11 = LOG10_ONE;
                continue;
            }

            const auto& gt = variant.genotypes[s];

            if (gt.size() < 2 || gt[0] == ALLELE_MISSING || gt[1] == ALLELE_MISSING) {
                // Missing data - uniform likelihood
                genotype_liks[idx].ll_00 = LOG10_ONE;
                genotype_liks[idx].ll_01 = LOG10_ONE;
                genotype_liks[idx].ll_11 = LOG10_ONE;
            } else {
                // Hard call genotype
                allele_t a1 = gt[0];
                allele_t a2 = gt[1];

                if (a1 == 0 && a2 == 0) {
                    // 0/0
                    genotype_liks[idx].ll_00 = LOG10_ONE;
                    genotype_liks[idx].ll_01 = LOG10_ZERO;
                    genotype_liks[idx].ll_11 = LOG10_ZERO;
                } else if ((a1 == 0 && a2 == 1) || (a1 == 1 && a2 == 0)) {
                    // 0/1 or 1/0
                    genotype_liks[idx].ll_00 = LOG10_ZERO;
                    genotype_liks[idx].ll_01 = LOG10_ONE;
                    genotype_liks[idx].ll_11 = LOG10_ZERO;
                } else if (a1 == 1 && a2 == 1) {
                    // 1/1
                    genotype_liks[idx].ll_00 = LOG10_ZERO;
                    genotype_liks[idx].ll_01 = LOG10_ZERO;
                    genotype_liks[idx].ll_11 = LOG10_ONE;
                } else {
                    // Other genotype (should not happen for biallelic)
                    genotype_liks[idx].ll_00 = LOG10_ONE;
                    genotype_liks[idx].ll_01 = LOG10_ONE;
                    genotype_liks[idx].ll_11 = LOG10_ONE;
                }
            }
        }
    }

    LOG_INFO("Target data loaded: " +
             std::to_string(markers.size()) + " markers, " +
             std::to_string(samples.size()) + " samples");

    return std::unique_ptr<TargetData>(new TargetData(
        std::move(markers),
        std::move(samples),
        std::move(genotype_liks)
    ));
}

GenotypeLikelihoods TargetData::get_likelihood(sample_t s, marker_t m) const {
    if (s >= samples_.size() || m >= markers_.size()) {
        return GenotypeLikelihoods();
    }
    return genotype_liks_[s * markers_.size() + m];
}

size_t TargetData::memory_usage() const {
    size_t total = 0;

    // Marker data
    total += markers_.capacity() * sizeof(Marker);

    // Sample data
    total += samples_.capacity() * sizeof(Sample);

    // Genotype likelihood data
    total += samples_.size() * markers_.size() * sizeof(GenotypeLikelihoods);

    return total;
}

// ImputationResult implementation

ImputationResult::ImputationResult(
    uint32_t num_samples,
    uint32_t num_markers
) : num_samples_(num_samples),
    num_markers_(num_markers)
{
    // Allocate arrays
    size_t hap_size = static_cast<size_t>(num_samples) * 2 * num_markers;
    haplotypes_ = std::make_unique<allele_t[]>(hap_size);

    size_t post_size = static_cast<size_t>(num_samples) * num_markers * 2;
    posteriors_ = std::make_unique<prob_t[]>(post_size);

    info_scores_ = std::make_unique<prob_t[]>(num_markers);

    // Initialize to zero
    std::fill(haplotypes_.get(), haplotypes_.get() + hap_size, 0);
    std::fill(posteriors_.get(), posteriors_.get() + post_size, 0.0f);
    std::fill(info_scores_.get(), info_scores_.get() + num_markers, 0.0f);
}

void ImputationResult::write_vcf(
    const std::string& filename,
    const TargetData& targets,
    const ReferencePanel& reference,
    const ImputationConfig& config
) const {
    LOG_INFO("Writing results to: " + filename);

    VCFWriter writer(filename);

    // Get sample names
    std::vector<std::string> sample_names;
    for (const auto& s : targets.samples()) {
        sample_names.push_back(s.id);
    }

    // Get contigs
    std::vector<std::string> contigs;
    if (!reference.markers().empty()) {
        contigs.push_back(reference.markers()[0].chrom);
    }

    writer.write_header(sample_names, contigs);

    // Write each variant
    for (marker_t m = 0; m < num_markers_; ++m) {
        const auto& marker = reference.markers()[m];

        // Build phased genotypes
        std::vector<std::vector<allele_t>> phased_gts;
        phased_gts.reserve(num_samples_);

        for (sample_t s = 0; s < num_samples_; ++s) {
            std::vector<allele_t> gt(2);
            gt[0] = haplotypes_[s * 2 * num_markers_ + 0 * num_markers_ + m];
            gt[1] = haplotypes_[s * 2 * num_markers_ + 1 * num_markers_ + m];
            phased_gts.push_back(gt);
        }

        writer.write_phased_variant(
            marker.chrom,
            marker.pos,
            marker.id,
            marker.ref,
            {marker.alt},
            phased_gts
        );
    }

    writer.close();

    LOG_INFO("Results written successfully");
}

const allele_t* ImputationResult::haplotype0(sample_t s) const {
    if (s >= num_samples_) return nullptr;
    return &haplotypes_[s * 2 * num_markers_];
}

const allele_t* ImputationResult::haplotype1(sample_t s) const {
    if (s >= num_samples_) return nullptr;
    return &haplotypes_[s * 2 * num_markers_ + num_markers_];
}

prob_t ImputationResult::get_dosage(sample_t s, marker_t m) const {
    if (s >= num_samples_ || m >= num_markers_) return 0.0f;

    allele_t h0 = haplotypes_[s * 2 * num_markers_ + 0 * num_markers_ + m];
    allele_t h1 = haplotypes_[s * 2 * num_markers_ + 1 * num_markers_ + m];

    return static_cast<prob_t>(h0 + h1);
}

void ImputationResult::get_probabilities(sample_t s, marker_t m, prob_t probs[3]) const {
    if (s >= num_samples_ || m >= num_markers_) {
        probs[0] = probs[1] = probs[2] = 0.0f;
        return;
    }

    // Get posterior probabilities for each haplotype
    size_t idx = s * num_markers_ * 2 + m * 2;
    prob_t p_h0_1 = posteriors_[idx + 0];  // P(hap0 = 1)
    prob_t p_h1_1 = posteriors_[idx + 1];  // P(hap1 = 1)

    prob_t p_h0_0 = 1.0f - p_h0_1;  // P(hap0 = 0)
    prob_t p_h1_0 = 1.0f - p_h1_1;  // P(hap1 = 0)

    // Compute genotype probabilities
    probs[0] = p_h0_0 * p_h1_0;  // P(0/0)
    probs[1] = p_h0_0 * p_h1_1 + p_h0_1 * p_h1_0;  // P(0/1)
    probs[2] = p_h0_1 * p_h1_1;  // P(1/1)
}

prob_t ImputationResult::get_info_score(marker_t m) const {
    if (m >= num_markers_) return 0.0f;
    return info_scores_[m];
}

void ImputationResult::set_haplotype(sample_t s, uint32_t phase, const allele_t* haplotype) {
    if (s >= num_samples_ || phase >= 2) return;

    allele_t* dest = &haplotypes_[s * 2 * num_markers_ + phase * num_markers_];
    std::copy(haplotype, haplotype + num_markers_, dest);
}

void ImputationResult::set_posterior(sample_t s, marker_t m, const prob_t* posterior) {
    if (s >= num_samples_ || m >= num_markers_) return;

    size_t idx = s * num_markers_ * 2 + m * 2;
    posteriors_[idx + 0] = posterior[0];
    posteriors_[idx + 1] = posterior[1];
}

void ImputationResult::compute_info_scores() {
    // Compute INFO score for each marker
    // INFO = 1 - Var(dosage) / (2*p*(1-p))
    // where p is the ALT allele frequency

    for (marker_t m = 0; m < num_markers_; ++m) {
        double sum_dosage = 0.0;
        double sum_dosage_sq = 0.0;

        for (sample_t s = 0; s < num_samples_; ++s) {
            double dosage = get_dosage(s, m);
            sum_dosage += dosage;
            sum_dosage_sq += dosage * dosage;
        }

        double mean_dosage = sum_dosage / num_samples_;
        double p = mean_dosage / 2.0;  // Allele frequency

        if (p == 0.0 || p == 1.0) {
            info_scores_[m] = 1.0f;  // Monomorphic
            continue;
        }

        double var_dosage = (sum_dosage_sq / num_samples_) - (mean_dosage * mean_dosage);
        double expected_var = 2.0 * p * (1.0 - p);

        double info = 1.0 - (var_dosage / expected_var);
        info_scores_[m] = static_cast<prob_t>(std::max(0.0, std::min(1.0, info)));
    }
}

size_t ImputationResult::memory_usage() const {
    size_t total = 0;

    // Haplotype data
    total += static_cast<size_t>(num_samples_) * 2 * num_markers_;

    // Posterior data
    total += static_cast<size_t>(num_samples_) * num_markers_ * 2 * sizeof(prob_t);

    // INFO scores
    total += num_markers_ * sizeof(prob_t);

    return total;
}

// Imputer implementation

Imputer::Imputer(
    const ReferencePanel& reference,
    const ImputationConfig& config
) : reference_(reference),
    config_(config),
    device_id_(-1)
{
    initialize_gpu();
}

Imputer::~Imputer() {
    // Cleanup will be handled by unique_ptr destructors
}

void Imputer::initialize_gpu() {
    // Select GPU device
    if (config_.device_id < 0) {
        device_id_ = select_best_device();
    } else {
        device_id_ = config_.device_id;
    }

    LOG_INFO("Using GPU device: " + std::to_string(device_id_));

    // Set device
    CHECK_CUDA(cudaSetDevice(device_id_));

    // TODO: Initialize GPU memory pools, streams, etc.
}

void Imputer::build_index() {
    LOG_INFO("Building PBWT index...");

    // Build PBWT index from reference panel
    pbwt_index_ = pbwt::PBWTIndex::build(
        reference_.haplotypes(),
        reference_.num_markers(),
        reference_.num_haplotypes()
    );

    // Create GPU state selector (will throw if not implemented)
    try {
        state_selector_ = std::make_unique<pbwt::GPUStateSelector>(
            *pbwt_index_,
            config_.hmm_params.num_states,
            device_id_
        );
    } catch (const ImputationError& e) {
        LOG_WARNING("GPU state selector not available: " + std::string(e.what()));
    }

    LOG_INFO("PBWT index built with " + std::to_string(pbwt_index_->memory_usage() / 1024 / 1024) + " MB");
}

void Imputer::validate_targets(const TargetData& targets) {
    // Check that targets and reference have compatible markers
    if (targets.num_markers() != reference_.num_markers()) {
        throw ImputationError(
            "Target and reference marker counts don't match: " +
            std::to_string(targets.num_markers()) + " vs " +
            std::to_string(reference_.num_markers())
        );
    }

    // TODO: Check marker positions match
}

std::unique_ptr<ImputationResult> Imputer::impute(const TargetData& targets) {
    return impute_with_progress(targets, nullptr);
}

std::unique_ptr<ImputationResult> Imputer::impute_with_progress(
    const TargetData& targets,
    std::function<void(uint32_t, uint32_t)> progress_callback
) {
    LOG_INFO("Starting imputation...");

    validate_targets(targets);

    // Create result object
    auto result = std::make_unique<ImputationResult>(
        targets.num_samples(),
        targets.num_markers()
    );

    // Process in batches
    uint32_t num_samples = targets.num_samples();
    uint32_t batch_size = config_.batch_size;

    for (uint32_t start = 0; start < num_samples; start += batch_size) {
        uint32_t end = std::min(start + batch_size, num_samples);

        LOG_INFO("Processing samples " + std::to_string(start) + " to " + std::to_string(end));

        // TODO: Implement actual imputation
        // For now, just copy observed genotypes as phased haplotypes
        impute_batch(targets, start, end, *result);

        if (progress_callback) {
            progress_callback(end, num_samples);
        }
    }

    // Compute quality metrics
    result->compute_info_scores();

    LOG_INFO("Imputation complete");

    return result;
}

void Imputer::impute_batch(
    const TargetData& targets,
    uint32_t start_sample,
    uint32_t end_sample,
    ImputationResult& result
) {
    // TODO: Implement actual GPU imputation
    // For now, just copy observed genotypes as a placeholder

    for (uint32_t s = start_sample; s < end_sample; ++s) {
        std::vector<allele_t> hap0(targets.num_markers());
        std::vector<allele_t> hap1(targets.num_markers());

        for (marker_t m = 0; m < targets.num_markers(); ++m) {
            auto gl = targets.get_likelihood(s, m);

            // Convert genotype likelihood to most likely genotype
            if (gl.ll_00 > gl.ll_01 && gl.ll_00 > gl.ll_11) {
                hap0[m] = 0;
                hap1[m] = 0;
            } else if (gl.ll_01 > gl.ll_11) {
                hap0[m] = 0;
                hap1[m] = 1;
            } else {
                hap0[m] = 1;
                hap1[m] = 1;
            }
        }

        result.set_haplotype(s, 0, hap0.data());
        result.set_haplotype(s, 1, hap1.data());
    }
}

size_t Imputer::device_memory_usage() const {
    // TODO: Query actual GPU memory usage
    return 0;
}

size_t Imputer::estimate_memory_requirement(
    const ReferencePanel& reference,
    uint32_t num_target_samples,
    const ImputationConfig& config
) {
    // Rough estimate of GPU memory required

    size_t total = 0;

    // Reference haplotypes
    total += static_cast<size_t>(reference.num_markers()) * reference.num_haplotypes();

    // PBWT index (approximate)
    total += static_cast<size_t>(reference.num_markers()) * reference.num_haplotypes() * 4;

    // Per-batch memory
    uint32_t batch_size = config.batch_size;
    size_t batch_memory = static_cast<size_t>(reference.num_markers()) * batch_size *
                          config.hmm_params.num_states * sizeof(prob_t) * 2;

    total += batch_memory;

    return total;
}

// Convenience function implementation

std::unique_ptr<ImputationResult> impute(
    const std::string& reference_vcf,
    const std::string& target_vcf,
    const std::string& output_vcf,
    const ImputationConfig& config
) {
    LOG_INFO("Loading reference panel...");
    auto reference = ReferencePanel::load_vcf(reference_vcf);

    LOG_INFO("Loading target data...");
    auto targets = TargetData::load_vcf(target_vcf);

    LOG_INFO("Creating imputer...");
    Imputer imputer(*reference, config);

    LOG_INFO("Building PBWT index...");
    try {
        imputer.build_index();
    } catch (const ImputationError&) {
        LOG_WARNING("PBWT index not implemented yet, proceeding without it");
    }

    LOG_INFO("Running imputation...");
    auto result = imputer.impute(*targets);

    LOG_INFO("Writing results...");
    result->write_vcf(output_vcf, *targets, *reference, config);

    return result;
}

// MultiGPUImputer stub implementations

MultiGPUImputer::MultiGPUImputer(
    const ReferencePanel& reference,
    const std::vector<int>& device_ids,
    const ImputationConfig& config
) : reference_(reference),
    config_(config),
    device_ids_(device_ids)
{
    // Create imputer for each GPU
    for (int device_id : device_ids_) {
        ImputationConfig gpu_config = config;
        gpu_config.device_id = device_id;
        imputers_.push_back(std::make_unique<Imputer>(reference, gpu_config));
    }
}

MultiGPUImputer::~MultiGPUImputer() {
    // Cleanup handled by unique_ptr
}

void MultiGPUImputer::build_indices() {
    for (auto& imputer : imputers_) {
        try {
            imputer->build_index();
        } catch (const ImputationError&) {
            LOG_WARNING("PBWT index not implemented yet");
        }
    }
}

std::unique_ptr<ImputationResult> MultiGPUImputer::impute(const TargetData& targets) {
    // TODO: Implement multi-GPU load balancing
    // For now, just use first GPU
    return imputers_[0]->impute(targets);
}

void MultiGPUImputer::partition_samples(
    uint32_t num_samples,
    std::vector<uint32_t>& start_indices,
    std::vector<uint32_t>& end_indices
) const {
    // Simple even partitioning
    uint32_t samples_per_gpu = (num_samples + imputers_.size() - 1) / imputers_.size();

    start_indices.clear();
    end_indices.clear();

    for (size_t i = 0; i < imputers_.size(); ++i) {
        uint32_t start = i * samples_per_gpu;
        uint32_t end = std::min(start + samples_per_gpu, num_samples);

        if (start < num_samples) {
            start_indices.push_back(start);
            end_indices.push_back(end);
        }
    }
}

} // namespace swiftimpute
