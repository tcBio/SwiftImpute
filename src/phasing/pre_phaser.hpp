#pragma once

#include "../core/types.hpp"
#include "../api/imputer.hpp"
#include <vector>
#include <memory>

namespace swiftimpute {
namespace phasing {

/**
 * @brief Configuration for pre-phasing
 */
struct PrePhasingConfig {
    uint32_t num_states;            // Number of HMM states for phasing (default: 8)
    double ne;                      // Effective population size
    uint32_t num_iterations;        // Number of phasing iterations (default: 5)
    bool use_reference_panel;       // Use reference panel for phasing
    bool verbose;                   // Verbose logging

    PrePhasingConfig() :
        num_states(8),
        ne(10000.0),
        num_iterations(5),
        use_reference_panel(true),
        verbose(false) {}
};

/**
 * @brief Result of phase detection
 */
struct PhaseStatus {
    bool is_fully_phased;           // All genotypes are phased
    bool is_partially_phased;       // Some genotypes are phased
    double phased_fraction;         // Fraction of genotypes that are phased
    size_t num_unphased_sites;      // Number of sites with unphased genotypes
    size_t num_unphased_samples;    // Number of samples with unphased genotypes
};

/**
 * @brief Pre-phaser for unphased genotype data
 *
 * Uses the Li-Stephens HMM model to phase unphased diploid genotypes
 * before imputation. This is important because the imputation HMM
 * operates on haplotypes rather than genotypes.
 */
class PrePhaser {
public:
    /**
     * @brief Construct a pre-phaser
     *
     * @param reference Reference panel for phasing context
     * @param config Pre-phasing configuration
     */
    PrePhaser(
        const ReferencePanel& reference,
        const PrePhasingConfig& config = PrePhasingConfig()
    );

    ~PrePhaser();

    /**
     * @brief Detect the phasing status of target data
     *
     * Scans the target data to determine how much is phased vs unphased.
     *
     * @param targets Target data to analyze
     * @return PhaseStatus indicating phasing state
     */
    static PhaseStatus detect_phase_status(const TargetData& targets);

    /**
     * @brief Phase unphased genotypes in target data
     *
     * Uses the Li-Stephens HMM to determine the most likely phasing
     * for unphased diploid genotypes.
     *
     * @param targets Target data to phase (modified in place via returned copy)
     * @return New TargetData with phased genotypes
     */
    std::unique_ptr<TargetData> phase(const TargetData& targets);

    /**
     * @brief Check if pre-phasing is needed for the target data
     *
     * @param targets Target data to check
     * @return true if any genotypes are unphased
     */
    static bool needs_phasing(const TargetData& targets);

private:
    const ReferencePanel& reference_;
    PrePhasingConfig config_;
    int device_id_;

    // Internal phasing using forward-backward
    void phase_sample(
        const GenotypeLikelihoods* sample_liks,
        marker_t num_markers,
        allele_t* haplotype0,
        allele_t* haplotype1
    );
};

/**
 * @brief Convenience function to phase target data if needed
 *
 * Checks if target data needs phasing and phases if necessary.
 * Returns the original data if already phased.
 *
 * @param targets Target data to potentially phase
 * @param reference Reference panel for phasing
 * @param config Pre-phasing configuration
 * @return TargetData (potentially phased)
 */
std::unique_ptr<TargetData> phase_if_needed(
    const TargetData& targets,
    const ReferencePanel& reference,
    const PrePhasingConfig& config = PrePhasingConfig()
);

} // namespace phasing
} // namespace swiftimpute
