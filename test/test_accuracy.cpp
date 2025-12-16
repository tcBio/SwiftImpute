#include "validation/accuracy_metrics.hpp"
#include <iostream>
#include <fstream>
#include <cassert>
#include <cmath>
#include <random>

using namespace swiftimpute;
using namespace swiftimpute::validation;

// Test helpers
bool approx_equal(double a, double b, double tol = 1e-6) {
    return std::abs(a - b) < tol;
}

void create_test_vcf(const std::string& filename,
                     const std::vector<std::string>& samples,
                     const std::vector<std::tuple<std::string, uint64_t, std::string>>& variants,
                     const std::vector<std::vector<std::string>>& genotypes) {
    std::ofstream out(filename);

    // Write header
    out << "##fileformat=VCFv4.2\n";
    out << "##FORMAT=<ID=GT,Number=1,Type=String,Description=\"Genotype\">\n";
    out << "##FORMAT=<ID=DS,Number=1,Type=Float,Description=\"Dosage\">\n";
    out << "##FORMAT=<ID=GP,Number=3,Type=Float,Description=\"Genotype Probabilities\">\n";
    out << "#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT";
    for (const auto& s : samples) {
        out << "\t" << s;
    }
    out << "\n";

    // Write variants
    for (size_t i = 0; i < variants.size(); ++i) {
        const auto& [chrom, pos, id] = variants[i];
        out << chrom << "\t" << pos << "\t" << id << "\tA\tG\t.\tPASS\t.\tGT";
        for (const auto& gt : genotypes[i]) {
            out << "\t" << gt;
        }
        out << "\n";
    }

    out.close();
}

// Test 1: Utility functions
void test_utils() {
    std::cout << "Testing utility functions..." << std::endl;

    // Test Pearson correlation
    {
        std::vector<double> x = {1, 2, 3, 4, 5};
        std::vector<double> y = {1, 2, 3, 4, 5};
        double r = utils::pearson_correlation(x, y);
        assert(approx_equal(r, 1.0));
    }

    {
        std::vector<double> x = {1, 2, 3, 4, 5};
        std::vector<double> y = {5, 4, 3, 2, 1};
        double r = utils::pearson_correlation(x, y);
        assert(approx_equal(r, -1.0));
    }

    // Test R-squared
    {
        std::vector<double> truth = {0, 1, 2, 0, 1, 2};
        std::vector<double> imputed = {0, 1, 2, 0, 1, 2};
        double r2 = utils::r_squared(truth, imputed);
        assert(approx_equal(r2, 1.0));
    }

    // Test allele frequency
    {
        std::vector<double> dosages = {0, 0, 1, 1, 2, 2};
        double af = utils::allele_frequency(dosages);
        assert(approx_equal(af, 0.5));
    }

    // Test quantile
    {
        std::vector<double> values = {1, 2, 3, 4, 5};
        assert(approx_equal(utils::quantile(values, 0.5), 3.0));
        assert(approx_equal(utils::quantile(values, 0.0), 1.0));
        assert(approx_equal(utils::quantile(values, 1.0), 5.0));
    }

    // Test mean_variance
    {
        std::vector<double> values = {2, 4, 4, 4, 5, 5, 7, 9};
        auto [mean, var] = utils::mean_variance(values);
        assert(approx_equal(mean, 5.0));
        assert(approx_equal(var, 4.571428, 1e-4));  // Sample variance
    }

    std::cout << "  Utility tests passed!" << std::endl;
}

// Test 2: MAF bin classification
void test_maf_bins() {
    std::cout << "Testing MAF bin classification..." << std::endl;

    assert(get_maf_bin(0.001) == MAFBin::VERY_RARE);
    assert(get_maf_bin(0.005) == MAFBin::RARE);
    assert(get_maf_bin(0.01) == MAFBin::LOW_FREQ);
    assert(get_maf_bin(0.05) == MAFBin::COMMON);
    assert(get_maf_bin(0.25) == MAFBin::COMMON);

    assert(maf_bin_name(MAFBin::VERY_RARE) == "MAF<0.5%");
    assert(maf_bin_name(MAFBin::COMMON) == "MAF>=5%");

    std::cout << "  MAF bin tests passed!" << std::endl;
}

// Test 3: VCF loading
void test_vcf_loading() {
    std::cout << "Testing VCF loading..." << std::endl;

    // Create test VCF files
    std::vector<std::string> samples = {"S1", "S2", "S3", "S4"};
    std::vector<std::tuple<std::string, uint64_t, std::string>> variants = {
        {"chr1", 100, "rs1"},
        {"chr1", 200, "rs2"},
        {"chr1", 300, "rs3"}
    };

    // Truth genotypes (diploid: 0/0, 0/1, 1/1)
    std::vector<std::vector<std::string>> truth_gts = {
        {"0/0", "0/1", "1/1", "0/0"},
        {"0/1", "0/1", "0/0", "1/1"},
        {"1/1", "0/0", "0/1", "0/1"}
    };

    create_test_vcf("/tmp/truth_test.vcf", samples, variants, truth_gts);

    auto truth = TruthData::load_vcf("/tmp/truth_test.vcf");

    assert(truth->num_samples() == 4);
    assert(truth->num_markers() == 3);

    // Check genotypes
    assert(truth->get_genotype(0, 0) == 0);  // S1, rs1: 0/0 = 0
    assert(truth->get_genotype(1, 0) == 1);  // S2, rs1: 0/1 = 1
    assert(truth->get_genotype(2, 0) == 2);  // S3, rs1: 1/1 = 2
    assert(truth->get_genotype(3, 1) == 2);  // S4, rs2: 1/1 = 2

    // Check MAF calculation
    double maf1 = truth->calculate_maf(0);  // rs1: 0+1+2+0 = 3 alt alleles / 8 = 0.375
    assert(approx_equal(maf1, 0.375));

    std::cout << "  VCF loading tests passed!" << std::endl;
}

// Test 4: Accuracy calculation
void test_accuracy_calculation() {
    std::cout << "Testing accuracy calculation..." << std::endl;

    std::vector<std::string> samples = {"S1", "S2", "S3", "S4", "S5", "S6", "S7", "S8", "S9", "S10"};
    std::vector<std::tuple<std::string, uint64_t, std::string>> variants = {
        {"chr1", 100, "rs1"},
        {"chr1", 200, "rs2"}
    };

    // Truth genotypes
    std::vector<std::vector<std::string>> truth_gts = {
        {"0/0", "0/0", "0/1", "0/1", "0/1", "1/1", "1/1", "0/0", "0/1", "1/1"},
        {"0/0", "0/1", "1/1", "0/0", "0/1", "1/1", "0/0", "0/1", "1/1", "0/0"}
    };

    // Perfect imputation
    std::vector<std::vector<std::string>> imputed_gts = {
        {"0/0", "0/0", "0/1", "0/1", "0/1", "1/1", "1/1", "0/0", "0/1", "1/1"},
        {"0/0", "0/1", "1/1", "0/0", "0/1", "1/1", "0/0", "0/1", "1/1", "0/0"}
    };

    create_test_vcf("/tmp/truth_acc.vcf", samples, variants, truth_gts);
    create_test_vcf("/tmp/imputed_acc.vcf", samples, variants, imputed_gts);

    auto truth = TruthData::load_vcf("/tmp/truth_acc.vcf");
    auto imputed = ImputedData::load_vcf("/tmp/imputed_acc.vcf");

    AccuracyCalculator::Config config;
    config.compute_per_variant = true;
    AccuracyCalculator calc(config);

    auto report = calc.evaluate(*truth, *imputed);

    // Perfect imputation should have R² = 1.0 and concordance = 1.0
    assert(approx_equal(report.overall.mean_dosage_r2, 1.0, 0.01));
    assert(approx_equal(report.overall.mean_concordance, 1.0, 0.01));

    std::cout << "  Accuracy calculation tests passed!" << std::endl;
}

// Test 5: Imperfect imputation
void test_imperfect_imputation() {
    std::cout << "Testing imperfect imputation metrics..." << std::endl;

    std::vector<std::string> samples = {"S1", "S2", "S3", "S4", "S5", "S6", "S7", "S8", "S9", "S10"};
    std::vector<std::tuple<std::string, uint64_t, std::string>> variants = {
        {"chr1", 100, "rs1"}
    };

    // Truth: 5 hom-ref, 3 het, 2 hom-alt
    std::vector<std::vector<std::string>> truth_gts = {
        {"0/0", "0/0", "0/0", "0/0", "0/0", "0/1", "0/1", "0/1", "1/1", "1/1"}
    };

    // Imputed: 2 errors (S6: 0/1->0/0, S9: 1/1->0/1)
    std::vector<std::vector<std::string>> imputed_gts = {
        {"0/0", "0/0", "0/0", "0/0", "0/0", "0/0", "0/1", "0/1", "0/1", "1/1"}
    };

    create_test_vcf("/tmp/truth_imp.vcf", samples, variants, truth_gts);
    create_test_vcf("/tmp/imputed_imp.vcf", samples, variants, imputed_gts);

    auto truth = TruthData::load_vcf("/tmp/truth_imp.vcf");
    auto imputed = ImputedData::load_vcf("/tmp/imputed_imp.vcf");

    AccuracyCalculator calc;
    auto report = calc.evaluate(*truth, *imputed);

    // 8/10 correct = 0.8 concordance
    assert(approx_equal(report.overall.mean_concordance, 0.8, 0.01));

    std::cout << "  Imperfect imputation tests passed!" << std::endl;
}

// Test 6: Report output
void test_report_output() {
    std::cout << "Testing report output..." << std::endl;

    ValidationReport report;
    report.n_samples = 100;
    report.n_variants = 1000;
    report.n_variants_evaluated = 950;
    report.evaluation_time_seconds = 1.5;

    report.overall.n_variants = 950;
    report.overall.mean_dosage_r2 = 0.95;
    report.overall.mean_concordance = 0.92;
    report.overall.mean_info_score = 0.88;
    report.overall.r2_q25 = 0.85;
    report.overall.r2_median = 0.96;
    report.overall.r2_q75 = 0.99;
    report.overall.n_variants_r2_gt_50 = 900;
    report.overall.n_variants_r2_gt_80 = 850;
    report.overall.n_variants_r2_gt_95 = 500;

    // Test print_summary
    std::ostringstream oss;
    report.print_summary(oss);
    std::string summary = oss.str();
    assert(summary.find("IMPUTATION ACCURACY REPORT") != std::string::npos);
    assert(summary.find("0.9500") != std::string::npos);  // R²

    // Test JSON output
    report.write_json("/tmp/report_test.json");
    std::ifstream json_file("/tmp/report_test.json");
    assert(json_file.good());

    std::cout << "  Report output tests passed!" << std::endl;
}

// Test 7: Missing data handling
void test_missing_data() {
    std::cout << "Testing missing data handling..." << std::endl;

    std::vector<std::string> samples = {"S1", "S2", "S3", "S4"};
    std::vector<std::tuple<std::string, uint64_t, std::string>> variants = {
        {"chr1", 100, "rs1"}
    };

    // Truth with missing data
    std::vector<std::vector<std::string>> truth_gts = {
        {"0/0", "./.", "0/1", "1/1"}
    };

    // Imputed (all present)
    std::vector<std::vector<std::string>> imputed_gts = {
        {"0/0", "0/0", "0/1", "1/1"}
    };

    create_test_vcf("/tmp/truth_miss.vcf", samples, variants, truth_gts);
    create_test_vcf("/tmp/imputed_miss.vcf", samples, variants, imputed_gts);

    auto truth = TruthData::load_vcf("/tmp/truth_miss.vcf");
    auto imputed = ImputedData::load_vcf("/tmp/imputed_miss.vcf");

    // Check missing detection
    assert(!truth->is_missing(0, 0));  // S1 not missing
    assert(truth->is_missing(1, 0));   // S2 is missing
    assert(!truth->is_missing(2, 0));  // S3 not missing

    std::cout << "  Missing data tests passed!" << std::endl;
}

// Test 8: Dosage and GP fields
void test_dosage_gp_fields() {
    std::cout << "Testing DS and GP field parsing..." << std::endl;

    // Create VCF with DS and GP fields
    std::ofstream out("/tmp/imputed_ds.vcf");
    out << "##fileformat=VCFv4.2\n";
    out << "##FORMAT=<ID=GT,Number=1,Type=String,Description=\"Genotype\">\n";
    out << "##FORMAT=<ID=DS,Number=1,Type=Float,Description=\"Dosage\">\n";
    out << "##FORMAT=<ID=GP,Number=3,Type=Float,Description=\"Genotype Probabilities\">\n";
    out << "#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT\tS1\tS2\n";
    out << "chr1\t100\trs1\tA\tG\t.\tPASS\t.\tGT:DS:GP\t0/0:0.1:0.9,0.1,0.0\t0/1:1.2:0.1,0.7,0.2\n";
    out.close();

    auto imputed = ImputedData::load_vcf("/tmp/imputed_ds.vcf");

    assert(imputed->has_dosages());
    assert(imputed->has_probabilities());

    // Check dosage values
    assert(approx_equal(imputed->get_dosage(0, 0), 0.1, 0.01));  // S1: DS=0.1
    assert(approx_equal(imputed->get_dosage(1, 0), 1.2, 0.01));  // S2: DS=1.2

    // Check GP values
    double probs[3];
    imputed->get_probabilities(0, 0, probs);
    assert(approx_equal(probs[0], 0.9, 0.01));  // P(0/0)
    assert(approx_equal(probs[1], 0.1, 0.01));  // P(0/1)
    assert(approx_equal(probs[2], 0.0, 0.01));  // P(1/1)

    std::cout << "  DS/GP field tests passed!" << std::endl;
}

// Main test runner
int main() {
    std::cout << "\n======================================" << std::endl;
    std::cout << "  SwiftImpute Accuracy Metrics Tests  " << std::endl;
    std::cout << "======================================\n" << std::endl;

    try {
        test_utils();
        test_maf_bins();
        test_vcf_loading();
        test_accuracy_calculation();
        test_imperfect_imputation();
        test_report_output();
        test_missing_data();
        test_dosage_gp_fields();

        std::cout << "\n======================================" << std::endl;
        std::cout << "  All tests passed successfully!      " << std::endl;
        std::cout << "======================================\n" << std::endl;
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "\nTest failed with exception: " << e.what() << std::endl;
        return 1;
    }
}
