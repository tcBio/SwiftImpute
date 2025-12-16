#!/bin/bash
#
# SwiftImpute Benchmark Dataset Preparation Script
#
# This script downloads and prepares standard benchmark datasets for
# evaluating imputation accuracy against Beagle.
#
# Requirements:
#   - bcftools
#   - tabix
#   - wget or curl
#   - ~50GB disk space for full datasets
#
# Usage:
#   ./prepare_benchmark.sh [OPTIONS]
#
# Options:
#   --chr CHROM     Chromosome to prepare (default: 22)
#   --output DIR    Output directory (default: ./benchmark_data)
#   --small         Prepare small test dataset only (~1000 variants)
#   --full          Prepare full chromosome dataset
#   --help          Show this help message

set -e

# Default parameters
CHROM="22"
OUTPUT_DIR="./benchmark_data"
DATASET_SIZE="small"

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --chr)
            CHROM="$2"
            shift 2
            ;;
        --output)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        --small)
            DATASET_SIZE="small"
            shift
            ;;
        --full)
            DATASET_SIZE="full"
            shift
            ;;
        --help)
            head -25 "$0" | tail -20
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

echo "================================================"
echo "SwiftImpute Benchmark Dataset Preparation"
echo "================================================"
echo ""
echo "Configuration:"
echo "  Chromosome: $CHROM"
echo "  Output dir: $OUTPUT_DIR"
echo "  Dataset size: $DATASET_SIZE"
echo ""

# Create output directory
mkdir -p "$OUTPUT_DIR"
cd "$OUTPUT_DIR"

# URLs for 1000 Genomes Phase 3 data (GRCh38)
BASE_URL="http://ftp.1000genomes.ebi.ac.uk/vol1/ftp/data_collections/1000G_2504_high_coverage/working/20220422_3202_phased_SNV_INDEL_SV"
VCF_FILE="1kGP_high_coverage_Illumina.chr${CHROM}.filtered.SNV_INDEL_SV_phased_panel.vcf.gz"

# Download reference VCF if not present
if [[ ! -f "$VCF_FILE" ]]; then
    echo "Downloading 1000 Genomes chr${CHROM} reference panel..."
    wget -c "${BASE_URL}/${VCF_FILE}" || curl -C - -O "${BASE_URL}/${VCF_FILE}"
    wget -c "${BASE_URL}/${VCF_FILE}.tbi" || curl -C - -O "${BASE_URL}/${VCF_FILE}.tbi"
fi

# Extract region for small dataset
if [[ "$DATASET_SIZE" == "small" ]]; then
    REGION="chr${CHROM}:20000000-21000000"
    echo "Extracting small region: $REGION"

    bcftools view -r "$REGION" "$VCF_FILE" -Oz -o "chr${CHROM}_small.vcf.gz"
    tabix -p vcf "chr${CHROM}_small.vcf.gz"

    WORK_VCF="chr${CHROM}_small.vcf.gz"
else
    WORK_VCF="$VCF_FILE"
fi

echo ""
echo "Creating reference and target datasets..."

# Get sample list
bcftools query -l "$WORK_VCF" > all_samples.txt
TOTAL_SAMPLES=$(wc -l < all_samples.txt)
echo "Total samples in VCF: $TOTAL_SAMPLES"

# Split into reference (80%) and target (20%) samples
NREF=$((TOTAL_SAMPLES * 80 / 100))
NTARGET=$((TOTAL_SAMPLES - NREF))

head -n $NREF all_samples.txt > reference_samples.txt
tail -n $NTARGET all_samples.txt > target_samples.txt

echo "Reference samples: $NREF"
echo "Target samples: $NTARGET"

# Create reference panel (all variants, reference samples only)
echo "Creating reference panel..."
bcftools view -S reference_samples.txt "$WORK_VCF" -Oz -o "reference_chr${CHROM}.vcf.gz"
tabix -p vcf "reference_chr${CHROM}.vcf.gz"

# Create truth data (all variants, target samples)
echo "Creating truth dataset..."
bcftools view -S target_samples.txt "$WORK_VCF" -Oz -o "truth_chr${CHROM}.vcf.gz"
tabix -p vcf "truth_chr${CHROM}.vcf.gz"

# Create masked target data (simulate missing genotypes)
echo "Creating masked target dataset (50% missing)..."
python3 << 'PYTHON_SCRIPT'
import gzip
import random
import sys

random.seed(42)
MASK_RATE = 0.5

with gzip.open(f"truth_chr{sys.argv[1] if len(sys.argv) > 1 else '22'}.vcf.gz", 'rt') as fin:
    with gzip.open(f"target_masked_chr{sys.argv[1] if len(sys.argv) > 1 else '22'}.vcf.gz", 'wt') as fout:
        for line in fin:
            if line.startswith('#'):
                fout.write(line)
                continue

            fields = line.strip().split('\t')
            # Fields: CHROM, POS, ID, REF, ALT, QUAL, FILTER, INFO, FORMAT, samples...

            new_fields = fields[:9]
            for gt in fields[9:]:
                if random.random() < MASK_RATE:
                    # Mask this genotype
                    parts = gt.split(':')
                    parts[0] = './.'
                    new_fields.append(':'.join(parts))
                else:
                    new_fields.append(gt)

            fout.write('\t'.join(new_fields) + '\n')
PYTHON_SCRIPT

# Handle chromosome argument
if command -v python3 &> /dev/null; then
    python3 -c "
import gzip
import random

random.seed(42)
MASK_RATE = 0.5
CHROM = '$CHROM'

with gzip.open(f'truth_chr{CHROM}.vcf.gz', 'rt') as fin:
    with gzip.open(f'target_masked_chr{CHROM}.vcf.gz', 'wt') as fout:
        for line in fin:
            if line.startswith('#'):
                fout.write(line)
                continue

            fields = line.strip().split('\t')
            new_fields = fields[:9]
            for gt in fields[9:]:
                if random.random() < MASK_RATE:
                    parts = gt.split(':')
                    parts[0] = './.'
                    new_fields.append(':'.join(parts))
                else:
                    new_fields.append(gt)

            fout.write('\t'.join(new_fields) + '\n')
"
    tabix -p vcf "target_masked_chr${CHROM}.vcf.gz"
fi

# Count variants
echo ""
echo "Dataset statistics:"
echo "-------------------"
NVAR_REF=$(bcftools view -H "reference_chr${CHROM}.vcf.gz" | wc -l)
NVAR_TARGET=$(bcftools view -H "truth_chr${CHROM}.vcf.gz" | wc -l)

echo "Reference panel variants: $NVAR_REF"
echo "Target variants: $NVAR_TARGET"
echo "Reference samples: $NREF"
echo "Target samples: $NTARGET"

# Create info file
cat > "benchmark_info.txt" << EOF
SwiftImpute Benchmark Dataset
=============================

Source: 1000 Genomes Phase 3 (GRCh38)
Chromosome: $CHROM
Dataset size: $DATASET_SIZE

Files:
  reference_chr${CHROM}.vcf.gz  - Reference panel ($NREF samples, $NVAR_REF variants)
  truth_chr${CHROM}.vcf.gz      - Truth genotypes ($NTARGET samples)
  target_masked_chr${CHROM}.vcf.gz - Target with 50% masked ($NTARGET samples)

Sample lists:
  reference_samples.txt - Reference sample IDs
  target_samples.txt    - Target sample IDs

Usage:
  # Run SwiftImpute
  swiftimpute -r reference_chr${CHROM}.vcf.gz \\
              -t target_masked_chr${CHROM}.vcf.gz \\
              -o imputed_chr${CHROM}.vcf.gz

  # Run Beagle
  java -jar beagle.jar \\
       ref=reference_chr${CHROM}.vcf.gz \\
       gt=target_masked_chr${CHROM}.vcf.gz \\
       out=beagle_imputed_chr${CHROM}

  # Compare accuracy
  ./compare_results.py truth_chr${CHROM}.vcf.gz imputed_chr${CHROM}.vcf.gz

Created: $(date)
EOF

echo ""
echo "================================================"
echo "Benchmark dataset preparation complete!"
echo "================================================"
echo ""
echo "Files created in: $OUTPUT_DIR"
echo "See benchmark_info.txt for usage instructions"
echo ""
