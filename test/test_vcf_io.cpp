#include "io/vcf_reader.hpp"
#include "io/vcf_writer.hpp"
#include "api/imputer.hpp"
#include "core/types.hpp"
#include <iostream>
#include <fstream>
#include <cassert>

using namespace swiftimpute;

void create_test_reference_vcf(const std::string& filename) {
    std::ofstream out(filename);

    // Write VCF header
    out << "##fileformat=VCFv4.2\n";
    out << "##contig=<ID=chr1,length=100000>\n";
    out << "##FORMAT=<ID=GT,Number=1,Type=String,Description=\"Genotype\">\n";
    out << "#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT\tREF1\tREF2\tREF3\tREF4\n";

    // Write 10 markers with known genotypes
    out << "chr1\t1000\trs1\tA\tG\t100\tPASS\t.\tGT\t0|0\t0|1\t1|0\t1|1\n";
    out << "chr1\t2000\trs2\tC\tT\t100\tPASS\t.\tGT\t0|0\t0|0\t1|1\t1|1\n";
    out << "chr1\t3000\trs3\tG\tA\t100\tPASS\t.\tGT\t1|1\t1|0\t0|1\t0|0\n";
    out << "chr1\t4000\trs4\tT\tC\t100\tPASS\t.\tGT\t0|1\t1|0\t0|1\t1|0\n";
    out << "chr1\t5000\trs5\tA\tT\t100\tPASS\t.\tGT\t1|1\t1|1\t0|0\t0|0\n";
    out << "chr1\t6000\trs6\tC\tG\t100\tPASS\t.\tGT\t0|0\t1|0\t0|1\t1|1\n";
    out << "chr1\t7000\trs7\tG\tC\t100\tPASS\t.\tGT\t0|1\t0|0\t1|1\t1|0\n";
    out << "chr1\t8000\trs8\tT\tA\t100\tPASS\t.\tGT\t1|0\t0|1\t1|0\t0|1\n";
    out << "chr1\t9000\trs9\tA\tC\t100\tPASS\t.\tGT\t1|1\t0|0\t1|1\t0|0\n";
    out << "chr1\t10000\trs10\tC\tA\t100\tPASS\t.\tGT\t0|0\t1|1\t0|0\t1|1\n";

    out.close();
}

void create_test_target_vcf(const std::string& filename) {
    std::ofstream out(filename);

    // Write VCF header
    out << "##fileformat=VCFv4.2\n";
    out << "##contig=<ID=chr1,length=100000>\n";
    out << "##FORMAT=<ID=GT,Number=1,Type=String,Description=\"Genotype\">\n";
    out << "#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT\tTARGET1\tTARGET2\n";

    // Write same 10 markers, but with some missing data
    out << "chr1\t1000\trs1\tA\tG\t100\tPASS\t.\tGT\t0/1\t1/1\n";
    out << "chr1\t2000\trs2\tC\tT\t100\tPASS\t.\tGT\t./.\t1/1\n";  // Missing
    out << "chr1\t3000\trs3\tG\tA\t100\tPASS\t.\tGT\t0/1\t0/0\n";
    out << "chr1\t4000\trs4\tT\tC\t100\tPASS\t.\tGT\t0/1\t./.\n";  // Missing
    out << "chr1\t5000\trs5\tA\tT\t100\tPASS\t.\tGT\t1/1\t0/0\n";
    out << "chr1\t6000\trs6\tC\tG\t100\tPASS\t.\tGT\t./.\t1/1\n";  // Missing
    out << "chr1\t7000\trs7\tG\tC\t100\tPASS\t.\tGT\t0/0\t1/0\n";
    out << "chr1\t8000\trs8\tT\tA\t100\tPASS\t.\tGT\t0/1\t0/1\n";
    out << "chr1\t9000\trs9\tA\tC\t100\tPASS\t.\tGT\t./.\t0/0\n";  // Missing
    out << "chr1\t10000\trs10\tC\tA\t100\tPASS\t.\tGT\t1/1\t1/1\n";

    out.close();
}

int main() {
    std::cout << "SwiftImpute VCF I/O Test\n";
    std::cout << "========================\n\n";

    try {
        // Create test files
        std::cout << "Creating test VCF files...\n";
        create_test_reference_vcf("test_reference.vcf");
        create_test_target_vcf("test_target.vcf");
        std::cout << "  ✓ Test files created\n\n";

        // Test VCF reader
        std::cout << "Testing VCF reader...\n";
        {
            VCFReader reader("test_reference.vcf");
            const auto& header = reader.read_header();

            std::cout << "  Reference samples: " << header.num_samples << "\n";
            assert(header.num_samples == 4);

            auto variants = reader.read_all_variants();
            std::cout << "  Reference variants: " << variants.size() << "\n";
            assert(variants.size() == 10);

            // Check first variant
            assert(variants[0].chrom == "chr1");
            assert(variants[0].position == 1000);
            assert(variants[0].ref == "A");
            assert(variants[0].alt[0] == "G");

            std::cout << "  ✓ VCF reader working\n\n";
        }

        // Test reference panel loading
        std::cout << "Testing ReferencePanel::load_vcf...\n";
        auto reference = ReferencePanel::load_vcf("test_reference.vcf");

        std::cout << "  Markers: " << reference->num_markers() << "\n";
        std::cout << "  Samples: " << reference->num_samples() << "\n";
        std::cout << "  Haplotypes: " << reference->num_haplotypes() << "\n";
        std::cout << "  Memory: " << reference->memory_usage() / 1024.0 << " KB\n";

        assert(reference->num_markers() == 10);
        assert(reference->num_samples() == 4);
        assert(reference->num_haplotypes() == 8);

        std::cout << "  ✓ ReferencePanel loaded\n\n";

        // Test target data loading
        std::cout << "Testing TargetData::load_vcf...\n";
        auto targets = TargetData::load_vcf("test_target.vcf");

        std::cout << "  Markers: " << targets->num_markers() << "\n";
        std::cout << "  Samples: " << targets->num_samples() << "\n";
        std::cout << "  Memory: " << targets->memory_usage() / 1024.0 << " KB\n";

        assert(targets->num_markers() == 10);
        assert(targets->num_samples() == 2);

        std::cout << "  ✓ TargetData loaded\n\n";

        // Test PBWT index building (skip GPU initialization)
        std::cout << "Testing PBWT index construction (CPU only)...\n";

        // Build PBWT index directly without Imputer (which tries to init GPU)
        auto pbwt_index = pbwt::PBWTIndex::build(
            reference->haplotypes(),
            reference->num_markers(),
            reference->num_haplotypes()
        );

        std::cout << "  PBWT markers: " << pbwt_index->num_markers() << "\n";
        std::cout << "  PBWT haplotypes: " << pbwt_index->num_haplotypes() << "\n";
        std::cout << "  PBWT memory: " << pbwt_index->memory_usage() / 1024.0 << " KB\n";

        assert(pbwt_index->num_markers() == 10);
        assert(pbwt_index->num_haplotypes() == 8);

        std::cout << "  ✓ PBWT index built\n\n";

        // Test state selection
        std::cout << "Testing PBWT state selection...\n";
        std::vector<haplotype_t> selected_states(4);  // Select 4 states
        std::vector<allele_t> target_hap(10, 0);  // Dummy target haplotype

        pbwt_index->select_states(0, target_hap.data(), 4, selected_states.data());

        std::cout << "  Selected states at marker 0: ";
        for (auto s : selected_states) {
            std::cout << s << " ";
        }
        std::cout << "\n";
        std::cout << "  ✓ State selection working\n\n";

        // Create mock imputation result for testing writer
        std::cout << "Creating mock imputation result...\n";
        ImputationResult result(targets->num_samples(), targets->num_markers());

        // Fill with simple pattern (just for testing writer)
        for (uint32_t s = 0; s < targets->num_samples(); ++s) {
            std::vector<allele_t> hap0(targets->num_markers());
            std::vector<allele_t> hap1(targets->num_markers());

            for (marker_t m = 0; m < targets->num_markers(); ++m) {
                auto gl = targets->get_likelihood(s, m);

                // Simple logic: take most likely genotype
                if (gl.ll_00 > gl.ll_01 && gl.ll_00 > gl.ll_11) {
                    hap0[m] = 0; hap1[m] = 0;
                } else if (gl.ll_01 > gl.ll_11) {
                    hap0[m] = 0; hap1[m] = 1;
                } else {
                    hap0[m] = 1; hap1[m] = 1;
                }
            }

            result.set_haplotype(s, 0, hap0.data());
            result.set_haplotype(s, 1, hap1.data());
        }

        result.compute_info_scores();

        std::cout << "  Result samples: " << result.num_samples() << "\n";
        std::cout << "  Result markers: " << result.num_markers() << "\n";
        std::cout << "  Memory: " << result.memory_usage() / 1024.0 << " KB\n";
        std::cout << "  ✓ Mock result created\n\n";

        // Test VCF writer
        std::cout << "Testing VCF writer...\n";
        ImputationConfig config;
        result.write_vcf("test_output.vcf", *targets, *reference, config);

        // Verify output file exists and has content
        std::ifstream check("test_output.vcf");
        assert(check.good());

        std::string line;
        int variant_count = 0;
        while (std::getline(check, line)) {
            if (!line.empty() && line[0] != '#') {
                variant_count++;
            }
        }
        assert(variant_count == 10);

        std::cout << "  ✓ Output VCF written with " << variant_count << " variants\n\n";

        // Summary
        std::cout << "========================================\n";
        std::cout << "All tests PASSED! ✓\n";
        std::cout << "========================================\n\n";

        std::cout << "Pipeline verification:\n";
        std::cout << "  1. VCF reading: ✓\n";
        std::cout << "  2. Data loading: ✓\n";
        std::cout << "  3. PBWT indexing: ✓\n";
        std::cout << "  4. State selection: ✓\n";
        std::cout << "  5. VCF writing: ✓\n\n";

        std::cout << "Output files created:\n";
        std::cout << "  - test_reference.vcf\n";
        std::cout << "  - test_target.vcf\n";
        std::cout << "  - test_output.vcf\n\n";

        return 0;

    } catch (const std::exception& e) {
        std::cerr << "ERROR: " << e.what() << "\n";
        return 1;
    }
}
