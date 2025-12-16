# SwiftImpute Benchmarks

This directory contains scripts for benchmarking SwiftImpute against Beagle and validating imputation accuracy.

## Quick Start

```bash
# 1. Prepare benchmark dataset (downloads 1000 Genomes data)
./prepare_benchmark.sh --small --chr 22

# 2. Run comparison against Beagle
./run_comparison.sh --chr 22
```

## Scripts

### prepare_benchmark.sh

Downloads and prepares benchmark datasets from 1000 Genomes Phase 3.

**Options:**
- `--chr CHROM` - Chromosome to prepare (default: 22)
- `--output DIR` - Output directory (default: ./benchmark_data)
- `--small` - Prepare small test dataset (~1000 variants, ~1MB region)
- `--full` - Prepare full chromosome dataset

**Output files:**
- `reference_chr{N}.vcf.gz` - Reference panel (80% of samples)
- `truth_chr{N}.vcf.gz` - True genotypes for target samples
- `target_masked_chr{N}.vcf.gz` - Target samples with 50% masked genotypes

### run_comparison.sh

Runs SwiftImpute and Beagle on benchmark data and compares accuracy.

**Options:**
- `--data DIR` - Benchmark data directory
- `--output DIR` - Output directory
- `--chr CHROM` - Chromosome
- `--beagle PATH` - Path to beagle.jar
- `--swiftimpute PATH` - Path to swiftimpute binary
- `--skip-beagle` - Skip Beagle run
- `--skip-swift` - Skip SwiftImpute run

**Output:**
- Imputed VCF files from each tool
- `comparison_results.json` - Accuracy metrics in JSON format
- Console output with comparison table

## Metrics Calculated

| Metric | Description |
|--------|-------------|
| **R² (Dosage R²)** | Squared Pearson correlation between true and imputed dosages |
| **Concordance** | Fraction of genotypes matching exactly |
| **INFO Score** | Ratio of observed to expected variance |

### MAF Stratification

Results are stratified by minor allele frequency:
- **Very rare**: MAF < 0.5%
- **Rare**: 0.5% ≤ MAF < 1%
- **Low frequency**: 1% ≤ MAF < 5%
- **Common**: MAF ≥ 5%

## Requirements

### For prepare_benchmark.sh:
- `bcftools` - VCF manipulation
- `tabix` - VCF indexing
- `wget` or `curl` - Download data
- ~50GB disk space for full datasets

### For run_comparison.sh:
- SwiftImpute binary
- Beagle JAR file (downloaded automatically if not found)
- Java 8+
- Python 3

## Example Output

```
IMPUTATION ACCURACY COMPARISON
============================================================

Metric                        SwiftImpute        Beagle
------------------------------------------------------------
Variants evaluated              12,345          12,345
Mean R²                         0.9512          0.9534
Median R²                       0.9801          0.9823
Mean Concordance                0.9234          0.9267
% R² > 0.50                     94.2%           94.5%
% R² > 0.80                     89.1%           89.8%
% R² > 0.95                     72.3%           73.1%

MAF-Stratified R²:
------------------------------------------------------------
  MAF < 0.5%                    0.7234          0.7312
  0.5-1%                        0.8456          0.8521
  1-5%                          0.9123          0.9178
  >5%                           0.9678          0.9701
```

## Target Accuracy

For SwiftImpute to be considered a viable Beagle alternative:

| MAF Range | Target R² |
|-----------|-----------|
| MAF ≥ 5% | > 0.95 |
| 1-5% | > 0.90 |
| 0.5-1% | > 0.80 |
| < 0.5% | > 0.70 |

Overall metrics should be within 2% of Beagle performance.
