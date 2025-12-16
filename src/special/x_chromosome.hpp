#pragma once

#include "../core/types.hpp"
#include "../api/imputer.hpp"
#include <string>
#include <vector>
#include <memory>
#include <unordered_set>

namespace swiftimpute {
namespace special {

/**
 * @brief Sex determination for samples
 */
enum class Sex {
    UNKNOWN = 0,
    MALE = 1,
    FEMALE = 2
};

/**
 * @brief Configuration for X chromosome handling
 */
struct XChromosomeConfig {
    bool auto_detect_sex;           // Infer sex from X chromosome heterozygosity
    double male_het_threshold;      // Max het rate for male (default: 0.02)
    uint64_t par1_start;            // PAR1 start position (GRCh38: 10001)
    uint64_t par1_end;              // PAR1 end position (GRCh38: 2781479)
    uint64_t par2_start;            // PAR2 start position (GRCh38: 155701383)
    uint64_t par2_end;              // PAR2 end position (GRCh38: 156030895)
    bool handle_par_as_diploid;     // Treat PAR regions as diploid

    XChromosomeConfig() :
        auto_detect_sex(true),
        male_het_threshold(0.02),
        // GRCh38 X chromosome PAR coordinates
        par1_start(10001),
        par1_end(2781479),
        par2_start(155701383),
        par2_end(156030895),
        handle_par_as_diploid(true) {}

    // GRCh37 PAR coordinates
    static XChromosomeConfig grch37() {
        XChromosomeConfig config;
        config.par1_start = 60001;
        config.par1_end = 2699520;
        config.par2_start = 154931044;
        config.par2_end = 155260560;
        return config;
    }
};

/**
 * @brief Result of sex inference
 */
struct SexInferenceResult {
    Sex sex;
    double het_rate;                // Observed heterozygosity rate on X
    double confidence;              // Confidence in the inference (0-1)
    size_t num_informative_sites;   // Number of sites used for inference
};

/**
 * @brief X chromosome-aware imputation handler
 *
 * Handles the special requirements for X chromosome imputation:
 * - Males are haploid on non-PAR regions
 * - PAR regions are diploid for both sexes
 * - Sex can be inferred from heterozygosity patterns
 */
class XChromosomeHandler {
public:
    /**
     * @brief Construct X chromosome handler
     *
     * @param config X chromosome configuration
     */
    explicit XChromosomeHandler(const XChromosomeConfig& config = XChromosomeConfig());

    /**
     * @brief Check if a position is in a PAR region
     *
     * @param pos Genomic position
     * @return true if position is in PAR1 or PAR2
     */
    bool is_par_region(uint64_t pos) const;

    /**
     * @brief Infer sex from genotype data on X chromosome
     *
     * Uses heterozygosity rate to determine sex:
     * - Males should have very low het rate on non-PAR X
     * - Females should have het rate similar to autosomes
     *
     * @param target Target data for a single sample
     * @return SexInferenceResult with inferred sex and confidence
     */
    SexInferenceResult infer_sex(const TargetData& target, sample_t sample_idx) const;

    /**
     * @brief Infer sex for all samples
     *
     * @param targets Target data
     * @return Vector of sex assignments
     */
    std::vector<Sex> infer_all_sexes(const TargetData& targets) const;

    /**
     * @brief Set known sex for samples
     *
     * @param sample_names Sample names
     * @param sexes Sex assignments
     */
    void set_sample_sexes(
        const std::vector<std::string>& sample_names,
        const std::vector<Sex>& sexes
    );

    /**
     * @brief Get sex for a sample
     *
     * @param sample_name Sample name
     * @return Sex assignment (UNKNOWN if not set)
     */
    Sex get_sample_sex(const std::string& sample_name) const;

    /**
     * @brief Prepare reference panel for X chromosome imputation
     *
     * For male reference samples, converts diploid to haploid in non-PAR
     *
     * @param reference Original reference panel
     * @param sample_sexes Sex assignments for reference samples
     * @return Modified reference panel
     */
    std::unique_ptr<ReferencePanel> prepare_reference(
        const ReferencePanel& reference,
        const std::vector<Sex>& sample_sexes
    ) const;

    /**
     * @brief Split target data into PAR and non-PAR regions
     *
     * @param targets Original target data
     * @return Pair of (non-PAR data, PAR data)
     */
    std::pair<std::unique_ptr<TargetData>, std::unique_ptr<TargetData>>
    split_by_par(const TargetData& targets) const;

    /**
     * @brief Impute X chromosome with sex-aware handling
     *
     * @param reference Reference panel
     * @param targets Target data
     * @param ref_sexes Sex assignments for reference samples
     * @param target_sexes Sex assignments for target samples
     * @param config Imputation configuration
     * @return Imputation result
     */
    std::unique_ptr<ImputationResult> impute(
        const ReferencePanel& reference,
        const TargetData& targets,
        const std::vector<Sex>& ref_sexes,
        const std::vector<Sex>& target_sexes,
        const ImputationConfig& config
    ) const;

    /**
     * @brief Validate X chromosome data
     *
     * Checks for:
     * - Male heterozygosity in non-PAR regions (warning)
     * - Missing genotypes in PAR regions
     *
     * @param targets Target data
     * @param sexes Sample sex assignments
     * @return Vector of validation warnings
     */
    std::vector<std::string> validate(
        const TargetData& targets,
        const std::vector<Sex>& sexes
    ) const;

private:
    XChromosomeConfig config_;
    std::unordered_map<std::string, Sex> sample_sexes_;

    // Check if chromosome name indicates X
    static bool is_x_chromosome(const std::string& chrom);

    // Count heterozygous sites in non-PAR regions
    size_t count_het_non_par(const TargetData& target, sample_t sample_idx) const;

    // Count informative sites in non-PAR regions
    size_t count_informative_non_par(const TargetData& target, sample_t sample_idx) const;
};

/**
 * @brief Convenience function to impute X chromosome
 *
 * @param reference_vcf Reference VCF file
 * @param target_vcf Target VCF file
 * @param output_vcf Output VCF file
 * @param config Imputation configuration
 * @param x_config X chromosome configuration
 * @return Imputation result
 */
std::unique_ptr<ImputationResult> impute_x_chromosome(
    const std::string& reference_vcf,
    const std::string& target_vcf,
    const std::string& output_vcf,
    const ImputationConfig& config = ImputationConfig(),
    const XChromosomeConfig& x_config = XChromosomeConfig()
);

} // namespace special
} // namespace swiftimpute
