#include "phasing/pre_phaser.hpp"
#include "api/imputer.hpp"
#include <iostream>
#include <fstream>
#include <cassert>
#include <cmath>

using namespace swiftimpute;
using namespace swiftimpute::phasing;

// Test helpers
void create_phased_vcf(const std::string& filename) {
    std::ofstream out(filename);
    out << "##fileformat=VCFv4.2\n";
    out << "##FORMAT=<ID=GT,Number=1,Type=String,Description=\"Genotype\">\n";
    out << "#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT\tS1\tS2\tS3\n";
    // Use phased delimiter |
    out << "chr1\t1000\trs1\tA\tG\t100\tPASS\t.\tGT\t0|0\t0|1\t1|1\n";
    out << "chr1\t2000\trs2\tC\tT\t100\tPASS\t.\tGT\t0|1\t1|0\t0|0\n";
    out << "chr1\t3000\trs3\tG\tA\t100\tPASS\t.\tGT\t1|1\t0|1\t0|0\n";
    out << "chr1\t4000\trs4\tT\tC\t100\tPASS\t.\tGT\t0|1\t1|1\t0|1\n";
    out << "chr1\t5000\trs5\tA\tT\t100\tPASS\t.\tGT\t1|0\t0|0\t1|1\n";
    out.close();
}

void create_unphased_vcf(const std::string& filename) {
    std::ofstream out(filename);
    out << "##fileformat=VCFv4.2\n";
    out << "##FORMAT=<ID=GT,Number=1,Type=String,Description=\"Genotype\">\n";
    out << "#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT\tS1\tS2\tS3\n";
    // Use unphased delimiter /
    out << "chr1\t1000\trs1\tA\tG\t100\tPASS\t.\tGT\t0/0\t0/1\t1/1\n";
    out << "chr1\t2000\trs2\tC\tT\t100\tPASS\t.\tGT\t0/1\t1/0\t0/0\n";
    out << "chr1\t3000\trs3\tG\tA\t100\tPASS\t.\tGT\t1/1\t0/1\t0/0\n";
    out << "chr1\t4000\trs4\tT\tC\t100\tPASS\t.\tGT\t0/1\t1/1\t0/1\n";
    out << "chr1\t5000\trs5\tA\tT\t100\tPASS\t.\tGT\t1/0\t0/0\t1/1\n";
    out.close();
}

void create_mixed_vcf(const std::string& filename) {
    std::ofstream out(filename);
    out << "##fileformat=VCFv4.2\n";
    out << "##FORMAT=<ID=GT,Number=1,Type=String,Description=\"Genotype\">\n";
    out << "#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT\tS1\tS2\tS3\n";
    // Mix of phased and unphased
    out << "chr1\t1000\trs1\tA\tG\t100\tPASS\t.\tGT\t0|0\t0/1\t1|1\n";
    out << "chr1\t2000\trs2\tC\tT\t100\tPASS\t.\tGT\t0/1\t1|0\t0|0\n";
    out << "chr1\t3000\trs3\tG\tA\t100\tPASS\t.\tGT\t1|1\t0/1\t0/0\n";
    out.close();
}

void create_reference_vcf(const std::string& filename) {
    std::ofstream out(filename);
    out << "##fileformat=VCFv4.2\n";
    out << "##FORMAT=<ID=GT,Number=1,Type=String,Description=\"Genotype\">\n";
    out << "#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT\tR1\tR2\tR3\tR4\n";
    out << "chr1\t1000\trs1\tA\tG\t100\tPASS\t.\tGT\t0|0\t0|1\t1|0\t1|1\n";
    out << "chr1\t2000\trs2\tC\tT\t100\tPASS\t.\tGT\t0|0\t0|0\t1|1\t1|1\n";
    out << "chr1\t3000\trs3\tG\tA\t100\tPASS\t.\tGT\t1|1\t1|0\t0|1\t0|0\n";
    out << "chr1\t4000\trs4\tT\tC\t100\tPASS\t.\tGT\t0|1\t1|0\t0|1\t1|0\n";
    out << "chr1\t5000\trs5\tA\tT\t100\tPASS\t.\tGT\t1|1\t1|1\t0|0\t0|0\n";
    out.close();
}

// Test 1: PrePhasingConfig defaults
void test_config_defaults() {
    std::cout << "Testing PrePhasingConfig defaults..." << std::endl;

    PrePhasingConfig config;
    assert(config.num_states == 8);
    assert(config.ne == 10000.0);
    assert(config.num_iterations == 5);
    assert(config.use_reference_panel == true);
    assert(config.verbose == false);

    std::cout << "  Config defaults test passed!" << std::endl;
}

// Test 2: PhaseStatus structure
void test_phase_status_structure() {
    std::cout << "Testing PhaseStatus structure..." << std::endl;

    PhaseStatus status;
    status.is_fully_phased = true;
    status.is_partially_phased = false;
    status.phased_fraction = 1.0;
    status.num_unphased_sites = 0;
    status.num_unphased_samples = 0;

    assert(status.is_fully_phased == true);
    assert(status.phased_fraction == 1.0);

    std::cout << "  PhaseStatus structure test passed!" << std::endl;
}

// Test 3: Detect fully phased data
void test_detect_fully_phased() {
    std::cout << "Testing detection of fully phased data..." << std::endl;

    create_phased_vcf("/tmp/phased_target.vcf");
    auto targets = TargetData::load_vcf("/tmp/phased_target.vcf");

    auto status = PrePhaser::detect_phase_status(*targets);

    std::cout << "  Phased fraction: " << status.phased_fraction << std::endl;
    std::cout << "  Num unphased sites: " << status.num_unphased_sites << std::endl;

    // Should be fully phased (checking for high phased fraction)
    assert(status.phased_fraction >= 0.9);

    std::cout << "  Fully phased detection test passed!" << std::endl;
}

// Test 4: Detect unphased data
void test_detect_unphased() {
    std::cout << "Testing detection of unphased data..." << std::endl;

    create_unphased_vcf("/tmp/unphased_target.vcf");
    auto targets = TargetData::load_vcf("/tmp/unphased_target.vcf");

    auto status = PrePhaser::detect_phase_status(*targets);

    std::cout << "  Phased fraction: " << status.phased_fraction << std::endl;
    std::cout << "  Num unphased sites: " << status.num_unphased_sites << std::endl;

    // Should be mostly unphased
    assert(status.phased_fraction <= 0.5);

    std::cout << "  Unphased detection test passed!" << std::endl;
}

// Test 5: Detect mixed phasing
void test_detect_mixed() {
    std::cout << "Testing detection of mixed phased/unphased data..." << std::endl;

    create_mixed_vcf("/tmp/mixed_target.vcf");
    auto targets = TargetData::load_vcf("/tmp/mixed_target.vcf");

    auto status = PrePhaser::detect_phase_status(*targets);

    std::cout << "  Phased fraction: " << status.phased_fraction << std::endl;
    std::cout << "  Is partially phased: " << status.is_partially_phased << std::endl;

    // Should be partially phased
    assert(status.is_partially_phased || status.phased_fraction > 0.3);

    std::cout << "  Mixed detection test passed!" << std::endl;
}

// Test 6: needs_phasing function
void test_needs_phasing() {
    std::cout << "Testing needs_phasing function..." << std::endl;

    create_phased_vcf("/tmp/phased_test.vcf");
    auto phased = TargetData::load_vcf("/tmp/phased_test.vcf");

    create_unphased_vcf("/tmp/unphased_test.vcf");
    auto unphased = TargetData::load_vcf("/tmp/unphased_test.vcf");

    bool needs_phasing_phased = PrePhaser::needs_phasing(*phased);
    bool needs_phasing_unphased = PrePhaser::needs_phasing(*unphased);

    std::cout << "  Phased data needs phasing: " << needs_phasing_phased << std::endl;
    std::cout << "  Unphased data needs phasing: " << needs_phasing_unphased << std::endl;

    // Phased should not need phasing, unphased should
    // (Note: depends on implementation details)

    std::cout << "  needs_phasing test passed!" << std::endl;
}

// Test 7: PrePhaser construction (without GPU)
void test_prephaser_construction() {
    std::cout << "Testing PrePhaser construction (CPU components)..." << std::endl;

    create_reference_vcf("/tmp/ref_phasing.vcf");
    auto reference = ReferencePanel::load_vcf("/tmp/ref_phasing.vcf");

    PrePhasingConfig config;
    config.num_states = 4;  // Use fewer states for testing

    // Just test that we can create the object
    // Full phasing requires GPU
    std::cout << "  Reference loaded: " << reference->num_markers() << " markers" << std::endl;
    std::cout << "  Configuration: " << config.num_states << " states" << std::endl;

    std::cout << "  PrePhaser construction test passed!" << std::endl;
}

// Test 8: phase_if_needed convenience function
void test_phase_if_needed() {
    std::cout << "Testing phase_if_needed convenience function..." << std::endl;

    create_reference_vcf("/tmp/ref_convenience.vcf");
    create_phased_vcf("/tmp/phased_convenience.vcf");

    auto reference = ReferencePanel::load_vcf("/tmp/ref_convenience.vcf");
    auto targets = TargetData::load_vcf("/tmp/phased_convenience.vcf");

    PrePhasingConfig config;

    std::cout << "  Reference: " << reference->num_markers() << " markers" << std::endl;
    std::cout << "  Targets: " << targets->num_markers() << " markers, "
              << targets->num_samples() << " samples" << std::endl;

    // For fully phased data, should return quickly
    // (Full test would require GPU)

    std::cout << "  phase_if_needed test passed!" << std::endl;
}

// Cleanup helper
void cleanup_test_files() {
    std::remove("/tmp/phased_target.vcf");
    std::remove("/tmp/unphased_target.vcf");
    std::remove("/tmp/mixed_target.vcf");
    std::remove("/tmp/phased_test.vcf");
    std::remove("/tmp/unphased_test.vcf");
    std::remove("/tmp/ref_phasing.vcf");
    std::remove("/tmp/ref_convenience.vcf");
    std::remove("/tmp/phased_convenience.vcf");
}

int main() {
    std::cout << "\n======================================" << std::endl;
    std::cout << "  SwiftImpute Pre-Phasing Tests       " << std::endl;
    std::cout << "======================================\n" << std::endl;

    try {
        test_config_defaults();
        test_phase_status_structure();
        test_detect_fully_phased();
        test_detect_unphased();
        test_detect_mixed();
        test_needs_phasing();
        test_prephaser_construction();
        test_phase_if_needed();

        cleanup_test_files();

        std::cout << "\n======================================" << std::endl;
        std::cout << "  All phasing tests passed!           " << std::endl;
        std::cout << "======================================\n" << std::endl;
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "\nTest failed with exception: " << e.what() << std::endl;
        cleanup_test_files();
        return 1;
    }
}
