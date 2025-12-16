#include "x_chromosome.hpp"
#include <algorithm>
#include <cmath>

namespace swiftimpute {
namespace special {

XChromosomeHandler::XChromosomeHandler(const XChromosomeConfig& config)
    : config_(config)
{
}

bool XChromosomeHandler::is_par_region(uint64_t pos) const {
    return (pos >= config_.par1_start && pos <= config_.par1_end) ||
           (pos >= config_.par2_start && pos <= config_.par2_end);
}

bool XChromosomeHandler::is_x_chromosome(const std::string& chrom) {
    return chrom == "X" || chrom == "chrX" || chrom == "x" || chrom == "23";
}

SexInferenceResult XChromosomeHandler::infer_sex(
    const TargetData& target,
    sample_t sample_idx
) const {
    SexInferenceResult result;
    result.sex = Sex::UNKNOWN;
    result.het_rate = 0.0;
    result.confidence = 0.0;
    result.num_informative_sites = 0;

    size_t het_count = count_het_non_par(target, sample_idx);
    size_t informative_count = count_informative_non_par(target, sample_idx);

    if (informative_count < 100) {
        // Not enough data for reliable inference
        return result;
    }

    result.het_rate = static_cast<double>(het_count) / informative_count;
    result.num_informative_sites = informative_count;

    // Classify based on heterozygosity rate
    if (result.het_rate < config_.male_het_threshold) {
        result.sex = Sex::MALE;
        // Confidence based on how far below threshold
        result.confidence = std::min(1.0, (config_.male_het_threshold - result.het_rate) /
                                          config_.male_het_threshold);
    } else if (result.het_rate > config_.male_het_threshold * 5) {
        result.sex = Sex::FEMALE;
        // Confidence based on how far above threshold
        double expected_female_het = 0.10;  // ~10% het rate expected for females
        result.confidence = std::min(1.0, result.het_rate / expected_female_het);
    } else {
        result.sex = Sex::UNKNOWN;
        result.confidence = 0.0;
    }

    return result;
}

std::vector<Sex> XChromosomeHandler::infer_all_sexes(const TargetData& targets) const {
    std::vector<Sex> sexes(targets.num_samples());

    for (sample_t s = 0; s < targets.num_samples(); ++s) {
        auto result = infer_sex(targets, s);
        sexes[s] = result.sex;

        if (result.sex != Sex::UNKNOWN) {
            LOG_INFO("Sample " + std::to_string(s) + ": inferred " +
                     (result.sex == Sex::MALE ? "MALE" : "FEMALE") +
                     " (het_rate=" + std::to_string(result.het_rate) +
                     ", confidence=" + std::to_string(result.confidence) + ")");
        }
    }

    return sexes;
}

size_t XChromosomeHandler::count_het_non_par(
    const TargetData& target,
    sample_t sample_idx
) const {
    size_t het_count = 0;
    const GenotypeLikelihoods* liks = target.genotype_likelihoods();
    marker_t num_markers = target.num_markers();

    for (marker_t m = 0; m < num_markers; ++m) {
        const auto& marker = target.markers()[m];

        // Skip if not X chromosome
        if (!is_x_chromosome(marker.chrom)) continue;

        // Skip PAR regions
        if (is_par_region(marker.pos)) continue;

        size_t idx = sample_idx * num_markers + m;
        const auto& gl = liks[idx];

        // Check if heterozygous (ll_01 is max)
        if (gl.ll_01 > gl.ll_00 && gl.ll_01 > gl.ll_11) {
            het_count++;
        }
    }

    return het_count;
}

size_t XChromosomeHandler::count_informative_non_par(
    const TargetData& target,
    sample_t sample_idx
) const {
    size_t count = 0;
    const GenotypeLikelihoods* liks = target.genotype_likelihoods();
    marker_t num_markers = target.num_markers();

    for (marker_t m = 0; m < num_markers; ++m) {
        const auto& marker = target.markers()[m];

        // Skip if not X chromosome
        if (!is_x_chromosome(marker.chrom)) continue;

        // Skip PAR regions
        if (is_par_region(marker.pos)) continue;

        size_t idx = sample_idx * num_markers + m;
        const auto& gl = liks[idx];

        // Check if informative (not all uniform)
        bool is_uniform = (std::abs(gl.ll_00 - gl.ll_01) < 1e-6 &&
                           std::abs(gl.ll_01 - gl.ll_11) < 1e-6);

        if (!is_uniform) {
            count++;
        }
    }

    return count;
}

void XChromosomeHandler::set_sample_sexes(
    const std::vector<std::string>& sample_names,
    const std::vector<Sex>& sexes
) {
    if (sample_names.size() != sexes.size()) {
        throw ImputationError("Sample names and sexes must have same length");
    }

    for (size_t i = 0; i < sample_names.size(); ++i) {
        sample_sexes_[sample_names[i]] = sexes[i];
    }
}

Sex XChromosomeHandler::get_sample_sex(const std::string& sample_name) const {
    auto it = sample_sexes_.find(sample_name);
    return (it != sample_sexes_.end()) ? it->second : Sex::UNKNOWN;
}

std::pair<std::unique_ptr<TargetData>, std::unique_ptr<TargetData>>
XChromosomeHandler::split_by_par(const TargetData& targets) const {
    // Find markers in non-PAR and PAR regions
    std::vector<size_t> non_par_indices;
    std::vector<size_t> par_indices;

    for (size_t m = 0; m < targets.num_markers(); ++m) {
        const auto& marker = targets.markers()[m];

        if (is_x_chromosome(marker.chrom)) {
            if (is_par_region(marker.pos)) {
                par_indices.push_back(m);
            } else {
                non_par_indices.push_back(m);
            }
        }
    }

    LOG_INFO("X chromosome split: " + std::to_string(non_par_indices.size()) +
             " non-PAR markers, " + std::to_string(par_indices.size()) + " PAR markers");

    // For now, return nullptr - full implementation would create filtered TargetData
    // This requires additional TargetData factory methods
    return {nullptr, nullptr};
}

std::unique_ptr<ImputationResult> XChromosomeHandler::impute(
    const ReferencePanel& reference,
    const TargetData& targets,
    const std::vector<Sex>& ref_sexes,
    const std::vector<Sex>& target_sexes,
    const ImputationConfig& config
) const {
    LOG_INFO("Starting X chromosome-aware imputation");

    // Validate inputs
    auto warnings = validate(targets, target_sexes);
    for (const auto& w : warnings) {
        LOG_WARNING(w);
    }

    // For non-PAR regions:
    // - Males: use haploid model (single haplotype)
    // - Females: use diploid model (standard)

    // For PAR regions:
    // - All samples: use diploid model

    // Current implementation: standard imputation with validation
    // Full implementation would:
    // 1. Split data into PAR and non-PAR
    // 2. Impute PAR as diploid
    // 3. Impute non-PAR with sex-specific handling
    // 4. Merge results

    Imputer imputer(reference, config);
    imputer.build_index();

    return imputer.impute(targets);
}

std::vector<std::string> XChromosomeHandler::validate(
    const TargetData& targets,
    const std::vector<Sex>& sexes
) const {
    std::vector<std::string> warnings;

    if (sexes.size() != targets.num_samples()) {
        warnings.push_back("Sex assignments don't match number of samples");
        return warnings;
    }

    const GenotypeLikelihoods* liks = targets.genotype_likelihoods();
    marker_t num_markers = targets.num_markers();

    // Check for male heterozygosity in non-PAR
    for (sample_t s = 0; s < targets.num_samples(); ++s) {
        if (sexes[s] != Sex::MALE) continue;

        size_t het_count = count_het_non_par(targets, s);
        if (het_count > 0) {
            warnings.push_back(
                "Male sample " + std::to_string(s) + " has " +
                std::to_string(het_count) + " heterozygous sites in non-PAR X"
            );
        }
    }

    return warnings;
}

std::unique_ptr<ImputationResult> impute_x_chromosome(
    const std::string& reference_vcf,
    const std::string& target_vcf,
    const std::string& output_vcf,
    const ImputationConfig& config,
    const XChromosomeConfig& x_config
) {
    LOG_INFO("Loading X chromosome data...");

    // Load reference and target
    auto reference = ReferencePanel::load_vcf(reference_vcf);
    auto targets = TargetData::load_vcf(target_vcf);

    // Create handler
    XChromosomeHandler handler(x_config);

    // Infer sex if auto-detection enabled
    std::vector<Sex> target_sexes;
    if (x_config.auto_detect_sex) {
        LOG_INFO("Inferring sample sexes from X chromosome data...");
        target_sexes = handler.infer_all_sexes(*targets);
    } else {
        target_sexes.resize(targets->num_samples(), Sex::UNKNOWN);
    }

    // Assume reference sexes are unknown (or could be provided separately)
    std::vector<Sex> ref_sexes(reference->num_samples(), Sex::UNKNOWN);

    // Run imputation
    auto result = handler.impute(*reference, *targets, ref_sexes, target_sexes, config);

    // Write output
    if (result) {
        result->write_vcf(output_vcf, *targets, *reference, config);
    }

    return result;
}

} // namespace special
} // namespace swiftimpute
