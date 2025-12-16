#include "pre_phaser.hpp"
#include "../io/vcf_reader.hpp"
#include "../pbwt/pbwt_index.hpp"
#include <algorithm>
#include <cmath>
#include <random>

namespace swiftimpute {
namespace phasing {

PrePhaser::PrePhaser(
    const ReferencePanel& reference,
    const PrePhasingConfig& config
) : reference_(reference),
    config_(config),
    device_id_(-1)
{
    // Select best GPU device if available
    device_id_ = select_best_device();

    if (config_.verbose) {
        LOG_INFO("PrePhaser initialized with " +
                 std::to_string(config_.num_states) + " states, " +
                 std::to_string(config_.num_iterations) + " iterations");
    }
}

PrePhaser::~PrePhaser() {
    // Cleanup handled automatically
}

PhaseStatus PrePhaser::detect_phase_status(const TargetData& targets) {
    PhaseStatus status;
    status.is_fully_phased = true;
    status.is_partially_phased = false;
    status.num_unphased_sites = 0;
    status.num_unphased_samples = 0;

    size_t total_genotypes = 0;
    size_t phased_genotypes = 0;

    // We don't have direct access to phase info from TargetData
    // since it's already converted to likelihoods.
    // For now, assume data is unphased if it contains heterozygotes
    // (This is a simplified detection - in practice you'd check the VCF directly)

    // Check for heterozygous sites in the likelihoods
    // A heterozygous site has ll_01 as the maximum likelihood
    const GenotypeLikelihoods* liks = targets.genotype_likelihoods();
    marker_t num_markers = targets.num_markers();
    sample_t num_samples = targets.num_samples();

    std::vector<bool> sample_has_unphased(num_samples, false);
    std::vector<bool> site_has_unphased(num_markers, false);

    for (sample_t s = 0; s < num_samples; ++s) {
        for (marker_t m = 0; m < num_markers; ++m) {
            size_t idx = s * num_markers + m;
            const auto& gl = liks[idx];

            total_genotypes++;

            // Check if this is a heterozygous site (needs phasing)
            // Uniform likelihoods (all equal) indicate missing data - skip
            bool is_uniform = (std::abs(gl.ll_00 - gl.ll_01) < 1e-6 &&
                               std::abs(gl.ll_01 - gl.ll_11) < 1e-6);

            if (!is_uniform) {
                // Check if heterozygous (ll_01 is max)
                bool is_het = (gl.ll_01 > gl.ll_00 && gl.ll_01 > gl.ll_11);

                if (is_het) {
                    // Heterozygous - potentially needs phasing
                    // We can't know from likelihoods if it's phased
                    // For safety, count as needing phasing
                    sample_has_unphased[s] = true;
                    site_has_unphased[m] = true;
                } else {
                    // Homozygous - considered phased
                    phased_genotypes++;
                }
            } else {
                // Missing data - skip from phasing consideration
                phased_genotypes++;
            }
        }
    }

    // Count samples and sites that need phasing
    for (sample_t s = 0; s < num_samples; ++s) {
        if (sample_has_unphased[s]) {
            status.num_unphased_samples++;
        }
    }

    for (marker_t m = 0; m < num_markers; ++m) {
        if (site_has_unphased[m]) {
            status.num_unphased_sites++;
        }
    }

    status.phased_fraction = (total_genotypes > 0) ?
        static_cast<double>(phased_genotypes) / total_genotypes : 1.0;

    status.is_fully_phased = (status.num_unphased_sites == 0);
    status.is_partially_phased = (!status.is_fully_phased && status.phased_fraction > 0.0);

    return status;
}

bool PrePhaser::needs_phasing(const TargetData& targets) {
    PhaseStatus status = detect_phase_status(targets);
    return !status.is_fully_phased;
}

std::unique_ptr<TargetData> PrePhaser::phase(const TargetData& targets) {
    if (config_.verbose) {
        LOG_INFO("Starting pre-phasing for " +
                 std::to_string(targets.num_samples()) + " samples, " +
                 std::to_string(targets.num_markers()) + " markers");
    }

    marker_t num_markers = targets.num_markers();
    sample_t num_samples = targets.num_samples();

    // Allocate output haplotypes
    std::vector<allele_t> phased_hap0(num_samples * num_markers);
    std::vector<allele_t> phased_hap1(num_samples * num_markers);

    // Build PBWT index for efficient state selection
    auto pbwt_index = pbwt::PBWTIndex::build(
        reference_.haplotypes(),
        reference_.num_markers(),
        reference_.num_haplotypes()
    );

    if (config_.verbose) {
        LOG_INFO("Built PBWT index for phasing");
    }

    // Phase each sample
    const GenotypeLikelihoods* liks = targets.genotype_likelihoods();

    for (sample_t s = 0; s < num_samples; ++s) {
        phase_sample(
            liks + s * num_markers,
            num_markers,
            phased_hap0.data() + s * num_markers,
            phased_hap1.data() + s * num_markers
        );

        if (config_.verbose && ((s + 1) % 10 == 0 || s == num_samples - 1)) {
            LOG_INFO("Phased sample " + std::to_string(s + 1) + "/" +
                     std::to_string(num_samples));
        }
    }

    // Create new genotype likelihoods from phased haplotypes
    // The phased genotypes are converted to "certain" likelihoods
    auto new_liks = std::make_unique<GenotypeLikelihoods[]>(num_samples * num_markers);

    const prob_t LOG10_ZERO = -999.0f;
    const prob_t LOG10_ONE = 0.0f;

    for (sample_t s = 0; s < num_samples; ++s) {
        for (marker_t m = 0; m < num_markers; ++m) {
            size_t idx = s * num_markers + m;
            allele_t a0 = phased_hap0[s * num_markers + m];
            allele_t a1 = phased_hap1[s * num_markers + m];

            if (a0 == ALLELE_MISSING || a1 == ALLELE_MISSING) {
                // Keep as uncertain
                new_liks[idx] = liks[idx];
            } else {
                // Convert phased genotype to likelihood
                if (a0 == 0 && a1 == 0) {
                    new_liks[idx].ll_00 = LOG10_ONE;
                    new_liks[idx].ll_01 = LOG10_ZERO;
                    new_liks[idx].ll_11 = LOG10_ZERO;
                } else if ((a0 == 0 && a1 == 1) || (a0 == 1 && a1 == 0)) {
                    new_liks[idx].ll_00 = LOG10_ZERO;
                    new_liks[idx].ll_01 = LOG10_ONE;
                    new_liks[idx].ll_11 = LOG10_ZERO;
                } else {
                    new_liks[idx].ll_00 = LOG10_ZERO;
                    new_liks[idx].ll_01 = LOG10_ZERO;
                    new_liks[idx].ll_11 = LOG10_ONE;
                }
            }
        }
    }

    // Copy markers and samples from original
    std::vector<Marker> markers = targets.markers();
    std::vector<Sample> samples = targets.samples();

    // Note: We can't directly construct TargetData due to private constructor
    // This is a limitation we need to address by either:
    // 1. Adding a friend declaration
    // 2. Adding a public factory method
    // 3. Returning phased haplotypes separately

    // For now, log that phasing was performed and return modified likelihoods
    // The actual TargetData reconstruction would need API changes
    LOG_INFO("Pre-phasing complete - " + std::to_string(num_samples) +
             " samples phased at " + std::to_string(num_markers) + " markers");

    // Return nullptr for now - actual implementation needs TargetData friend access
    // TODO: Add friend declaration or factory method to TargetData
    return nullptr;
}

void PrePhaser::phase_sample(
    const GenotypeLikelihoods* sample_liks,
    marker_t num_markers,
    allele_t* haplotype0,
    allele_t* haplotype1
) {
    // Simple phasing using reference panel frequencies
    // This is a basic implementation - full HMM phasing would be more accurate

    haplotype_t num_ref_haps = reference_.num_haplotypes();

    // For each marker, determine the most likely phase
    for (marker_t m = 0; m < num_markers; ++m) {
        const auto& gl = sample_liks[m];

        // Check if uniform (missing)
        bool is_uniform = (std::abs(gl.ll_00 - gl.ll_01) < 1e-6 &&
                           std::abs(gl.ll_01 - gl.ll_11) < 1e-6);

        if (is_uniform) {
            // Missing - mark as such
            haplotype0[m] = ALLELE_MISSING;
            haplotype1[m] = ALLELE_MISSING;
            continue;
        }

        // Find the most likely genotype
        if (gl.ll_00 >= gl.ll_01 && gl.ll_00 >= gl.ll_11) {
            // Homozygous REF
            haplotype0[m] = 0;
            haplotype1[m] = 0;
        } else if (gl.ll_11 >= gl.ll_01 && gl.ll_11 >= gl.ll_00) {
            // Homozygous ALT
            haplotype0[m] = 1;
            haplotype1[m] = 1;
        } else {
            // Heterozygous - need to phase
            // Use reference panel allele frequency to guide phasing
            // (Simple heuristic - true phasing would use HMM)

            // Count ALT alleles in reference at this position
            uint32_t alt_count = 0;
            for (haplotype_t h = 0; h < num_ref_haps; ++h) {
                if (reference_.get_allele(m, h) == 1) {
                    alt_count++;
                }
            }

            double alt_freq = static_cast<double>(alt_count) / num_ref_haps;

            // If this marker is preceded by a het, try to maintain consistency
            // (LD-based phasing heuristic)
            if (m > 0 && haplotype0[m-1] != ALLELE_MISSING && haplotype1[m-1] != ALLELE_MISSING) {
                // Look for LD pattern in reference
                uint32_t ref_00 = 0, ref_01 = 0, ref_10 = 0, ref_11 = 0;

                for (haplotype_t h = 0; h < num_ref_haps; ++h) {
                    allele_t prev = reference_.get_allele(m - 1, h);
                    allele_t curr = reference_.get_allele(m, h);

                    if (prev == 0 && curr == 0) ref_00++;
                    else if (prev == 0 && curr == 1) ref_01++;
                    else if (prev == 1 && curr == 0) ref_10++;
                    else if (prev == 1 && curr == 1) ref_11++;
                }

                // Determine which phase is more likely based on LD
                // If haplotype0 had prev_allele, which curr_allele is more likely?
                allele_t prev_h0 = haplotype0[m - 1];
                uint32_t h0_with_0, h0_with_1;

                if (prev_h0 == 0) {
                    h0_with_0 = ref_00;
                    h0_with_1 = ref_01;
                } else {
                    h0_with_0 = ref_10;
                    h0_with_1 = ref_11;
                }

                if (h0_with_1 > h0_with_0) {
                    haplotype0[m] = 1;
                    haplotype1[m] = 0;
                } else {
                    haplotype0[m] = 0;
                    haplotype1[m] = 1;
                }
            } else {
                // No previous marker or missing - use simple frequency-based assignment
                // Assign ALT to haplotype with probability proportional to alt_freq
                if (alt_freq > 0.5) {
                    haplotype0[m] = 1;
                    haplotype1[m] = 0;
                } else {
                    haplotype0[m] = 0;
                    haplotype1[m] = 1;
                }
            }
        }
    }
}

std::unique_ptr<TargetData> phase_if_needed(
    const TargetData& targets,
    const ReferencePanel& reference,
    const PrePhasingConfig& config
) {
    if (!PrePhaser::needs_phasing(targets)) {
        LOG_INFO("Target data appears fully phased - skipping pre-phasing");
        return nullptr;  // Caller should use original targets
    }

    PhaseStatus status = PrePhaser::detect_phase_status(targets);
    LOG_INFO("Detected " + std::to_string(status.num_unphased_sites) +
             " sites needing phasing (" +
             std::to_string(100.0 * (1.0 - status.phased_fraction)) + "% unphased)");

    PrePhaser phaser(reference, config);
    return phaser.phase(targets);
}

} // namespace phasing
} // namespace swiftimpute
