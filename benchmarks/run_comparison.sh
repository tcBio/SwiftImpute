#!/bin/bash
#
# SwiftImpute vs Beagle Comparison Script
#
# This script runs both SwiftImpute and Beagle on the benchmark dataset
# and compares their accuracy metrics.
#
# Requirements:
#   - SwiftImpute binary (swiftimpute)
#   - Beagle JAR file (beagle.jar)
#   - Java 8+
#   - Python 3 with numpy
#   - bcftools
#
# Usage:
#   ./run_comparison.sh [OPTIONS]
#
# Options:
#   --data DIR          Benchmark data directory (default: ./benchmark_data)
#   --output DIR        Output directory (default: ./comparison_results)
#   --chr CHROM         Chromosome (default: 22)
#   --beagle PATH       Path to beagle.jar
#   --swiftimpute PATH  Path to swiftimpute binary
#   --skip-beagle       Skip Beagle run (use existing results)
#   --skip-swift        Skip SwiftImpute run (use existing results)
#   --help              Show this help message

set -e

# Default parameters
DATA_DIR="./benchmark_data"
OUTPUT_DIR="./comparison_results"
CHROM="22"
BEAGLE_JAR="beagle.jar"
SWIFTIMPUTE_BIN="swiftimpute"
RUN_BEAGLE=true
RUN_SWIFT=true

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --data)
            DATA_DIR="$2"
            shift 2
            ;;
        --output)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        --chr)
            CHROM="$2"
            shift 2
            ;;
        --beagle)
            BEAGLE_JAR="$2"
            shift 2
            ;;
        --swiftimpute)
            SWIFTIMPUTE_BIN="$2"
            shift 2
            ;;
        --skip-beagle)
            RUN_BEAGLE=false
            shift
            ;;
        --skip-swift)
            RUN_SWIFT=false
            shift
            ;;
        --help)
            head -30 "$0" | tail -25
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

echo "================================================"
echo "SwiftImpute vs Beagle Comparison"
echo "================================================"
echo ""
echo "Configuration:"
echo "  Data directory: $DATA_DIR"
echo "  Output directory: $OUTPUT_DIR"
echo "  Chromosome: $CHROM"
echo ""

# Create output directory
mkdir -p "$OUTPUT_DIR"

# Input files
REF_VCF="${DATA_DIR}/reference_chr${CHROM}.vcf.gz"
TARGET_VCF="${DATA_DIR}/target_masked_chr${CHROM}.vcf.gz"
TRUTH_VCF="${DATA_DIR}/truth_chr${CHROM}.vcf.gz"

# Check input files exist
for f in "$REF_VCF" "$TARGET_VCF" "$TRUTH_VCF"; do
    if [[ ! -f "$f" ]]; then
        echo "Error: Required file not found: $f"
        echo "Run prepare_benchmark.sh first to create benchmark datasets."
        exit 1
    fi
done

# Output files
SWIFT_OUT="${OUTPUT_DIR}/swiftimpute_chr${CHROM}.vcf.gz"
BEAGLE_OUT="${OUTPUT_DIR}/beagle_chr${CHROM}"

# ============================================
# Run SwiftImpute
# ============================================
if [[ "$RUN_SWIFT" == true ]]; then
    echo ""
    echo "Running SwiftImpute..."
    echo "----------------------"

    SWIFT_START=$(date +%s.%N)

    "$SWIFTIMPUTE_BIN" \
        -r "$REF_VCF" \
        -t "$TARGET_VCF" \
        -o "$SWIFT_OUT" \
        --states 8 \
        --ne 15000 \
        --verbose

    SWIFT_END=$(date +%s.%N)
    SWIFT_TIME=$(echo "$SWIFT_END - $SWIFT_START" | bc)

    echo "SwiftImpute completed in ${SWIFT_TIME} seconds"
fi

# ============================================
# Run Beagle
# ============================================
if [[ "$RUN_BEAGLE" == true ]]; then
    echo ""
    echo "Running Beagle..."
    echo "-----------------"

    if [[ ! -f "$BEAGLE_JAR" ]]; then
        echo "Warning: Beagle JAR not found at $BEAGLE_JAR"
        echo "Downloading Beagle 5.4..."
        wget -O "$BEAGLE_JAR" "https://faculty.washington.edu/browning/beagle/beagle.28Jun21.220.jar" || {
            echo "Failed to download Beagle. Please download manually."
            RUN_BEAGLE=false
        }
    fi

    if [[ "$RUN_BEAGLE" == true ]]; then
        BEAGLE_START=$(date +%s.%N)

        java -Xmx8g -jar "$BEAGLE_JAR" \
            ref="$REF_VCF" \
            gt="$TARGET_VCF" \
            out="$BEAGLE_OUT" \
            nthreads=4 \
            ne=15000

        BEAGLE_END=$(date +%s.%N)
        BEAGLE_TIME=$(echo "$BEAGLE_END - $BEAGLE_START" | bc)

        echo "Beagle completed in ${BEAGLE_TIME} seconds"
    fi
fi

# ============================================
# Calculate Accuracy Metrics
# ============================================
echo ""
echo "Calculating accuracy metrics..."
echo "-------------------------------"

# Create Python comparison script
python3 << 'PYTHON_SCRIPT'
import gzip
import sys
import os
from collections import defaultdict
import math

def load_vcf_genotypes(vcf_path):
    """Load genotypes from VCF file as dictionary {(chrom, pos): {sample: dosage}}"""
    genotypes = {}
    samples = []

    opener = gzip.open if vcf_path.endswith('.gz') else open

    with opener(vcf_path, 'rt') as f:
        for line in f:
            if line.startswith('##'):
                continue
            if line.startswith('#CHROM'):
                fields = line.strip().split('\t')
                samples = fields[9:]
                continue

            fields = line.strip().split('\t')
            chrom = fields[0]
            pos = int(fields[1])
            format_fields = fields[8].split(':')

            gt_idx = format_fields.index('GT') if 'GT' in format_fields else 0
            ds_idx = format_fields.index('DS') if 'DS' in format_fields else -1

            site_gts = {}
            for i, sample_data in enumerate(fields[9:]):
                parts = sample_data.split(':')
                gt = parts[gt_idx] if gt_idx < len(parts) else './.'

                # Parse dosage
                if ds_idx >= 0 and ds_idx < len(parts) and parts[ds_idx] != '.':
                    dosage = float(parts[ds_idx])
                else:
                    # Calculate from GT
                    if gt in ['./.', '.|.', '.']:
                        dosage = None
                    else:
                        sep = '|' if '|' in gt else '/'
                        alleles = gt.split(sep)
                        try:
                            dosage = sum(int(a) for a in alleles if a != '.')
                        except:
                            dosage = None

                site_gts[samples[i]] = dosage

            genotypes[(chrom, pos)] = site_gts

    return genotypes, samples

def calculate_r2(truth_dosages, imputed_dosages):
    """Calculate R-squared (squared Pearson correlation)"""
    n = len(truth_dosages)
    if n < 2:
        return None

    mean_t = sum(truth_dosages) / n
    mean_i = sum(imputed_dosages) / n

    cov = sum((t - mean_t) * (i - mean_i) for t, i in zip(truth_dosages, imputed_dosages))
    var_t = sum((t - mean_t) ** 2 for t in truth_dosages)
    var_i = sum((i - mean_i) ** 2 for i in imputed_dosages)

    if var_t < 1e-10 or var_i < 1e-10:
        return None

    r = cov / math.sqrt(var_t * var_i)
    return r * r

def calculate_maf(dosages):
    """Calculate minor allele frequency from dosages"""
    valid = [d for d in dosages if d is not None]
    if not valid:
        return None
    af = sum(valid) / (2 * len(valid))
    return min(af, 1 - af)

def evaluate_imputation(truth_path, imputed_path):
    """Evaluate imputation accuracy"""
    print(f"Loading truth: {truth_path}")
    truth, samples = load_vcf_genotypes(truth_path)
    print(f"Loading imputed: {imputed_path}")
    imputed, _ = load_vcf_genotypes(imputed_path)

    # Find common sites
    common_sites = set(truth.keys()) & set(imputed.keys())
    print(f"Common sites: {len(common_sites)}")

    # Calculate per-variant metrics
    results = []
    maf_bins = {'very_rare': [], 'rare': [], 'low_freq': [], 'common': []}

    for site in common_sites:
        truth_d = []
        imputed_d = []

        for sample in samples:
            t = truth[site].get(sample)
            i = imputed[site].get(sample)

            if t is not None and i is not None:
                truth_d.append(t)
                imputed_d.append(i)

        if len(truth_d) < 10:
            continue

        r2 = calculate_r2(truth_d, imputed_d)
        concordance = sum(1 for t, i in zip(truth_d, imputed_d) if round(t) == round(i)) / len(truth_d)
        maf = calculate_maf(truth_d)

        if r2 is not None and maf is not None:
            results.append({
                'site': site,
                'r2': r2,
                'concordance': concordance,
                'maf': maf,
                'n': len(truth_d)
            })

            # Bin by MAF
            if maf < 0.005:
                maf_bins['very_rare'].append(r2)
            elif maf < 0.01:
                maf_bins['rare'].append(r2)
            elif maf < 0.05:
                maf_bins['low_freq'].append(r2)
            else:
                maf_bins['common'].append(r2)

    # Calculate summary statistics
    all_r2 = [r['r2'] for r in results]
    all_conc = [r['concordance'] for r in results]

    return {
        'n_variants': len(results),
        'mean_r2': sum(all_r2) / len(all_r2) if all_r2 else 0,
        'mean_concordance': sum(all_conc) / len(all_conc) if all_conc else 0,
        'median_r2': sorted(all_r2)[len(all_r2)//2] if all_r2 else 0,
        'r2_gt_50': sum(1 for r in all_r2 if r > 0.5) / len(all_r2) * 100 if all_r2 else 0,
        'r2_gt_80': sum(1 for r in all_r2 if r > 0.8) / len(all_r2) * 100 if all_r2 else 0,
        'r2_gt_95': sum(1 for r in all_r2 if r > 0.95) / len(all_r2) * 100 if all_r2 else 0,
        'maf_bins': {k: sum(v)/len(v) if v else 0 for k, v in maf_bins.items()}
    }

# Main comparison
CHROM = os.environ.get('CHROM', '22')
OUTPUT_DIR = os.environ.get('OUTPUT_DIR', './comparison_results')
DATA_DIR = os.environ.get('DATA_DIR', './benchmark_data')

truth_path = f"{DATA_DIR}/truth_chr{CHROM}.vcf.gz"
swift_path = f"{OUTPUT_DIR}/swiftimpute_chr{CHROM}.vcf.gz"
beagle_path = f"{OUTPUT_DIR}/beagle_chr{CHROM}.vcf.gz"

print("\n" + "=" * 60)
print("IMPUTATION ACCURACY COMPARISON")
print("=" * 60 + "\n")

results = {}

# Evaluate SwiftImpute
if os.path.exists(swift_path):
    print("\nEvaluating SwiftImpute results...")
    results['swiftimpute'] = evaluate_imputation(truth_path, swift_path)

# Evaluate Beagle
if os.path.exists(beagle_path):
    print("\nEvaluating Beagle results...")
    results['beagle'] = evaluate_imputation(truth_path, beagle_path)

# Print comparison table
print("\n" + "=" * 60)
print("RESULTS SUMMARY")
print("=" * 60)

print(f"\n{'Metric':<25} ", end='')
for tool in results.keys():
    print(f"{tool:>15}", end='')
print()
print("-" * (25 + 15 * len(results)))

metrics = [
    ('Variants evaluated', 'n_variants', '{:,.0f}'),
    ('Mean R²', 'mean_r2', '{:.4f}'),
    ('Median R²', 'median_r2', '{:.4f}'),
    ('Mean Concordance', 'mean_concordance', '{:.4f}'),
    ('% R² > 0.50', 'r2_gt_50', '{:.1f}%'),
    ('% R² > 0.80', 'r2_gt_80', '{:.1f}%'),
    ('% R² > 0.95', 'r2_gt_95', '{:.1f}%'),
]

for label, key, fmt in metrics:
    print(f"{label:<25} ", end='')
    for tool in results.keys():
        val = results[tool][key]
        print(f"{fmt.format(val):>15}", end='')
    print()

# MAF-stratified results
print("\nMAF-Stratified R²:")
print("-" * (25 + 15 * len(results)))
for maf_bin in ['very_rare', 'rare', 'low_freq', 'common']:
    label = {'very_rare': 'MAF < 0.5%', 'rare': '0.5-1%', 'low_freq': '1-5%', 'common': '>5%'}[maf_bin]
    print(f"  {label:<23} ", end='')
    for tool in results.keys():
        val = results[tool]['maf_bins'].get(maf_bin, 0)
        print(f"{val:>15.4f}", end='')
    print()

print("\n" + "=" * 60 + "\n")

# Save results to JSON
import json
with open(f"{OUTPUT_DIR}/comparison_results.json", 'w') as f:
    json.dump(results, f, indent=2)
print(f"Results saved to: {OUTPUT_DIR}/comparison_results.json")
PYTHON_SCRIPT

echo ""
echo "================================================"
echo "Comparison complete!"
echo "================================================"
echo ""
echo "Results saved to: $OUTPUT_DIR"
echo ""
