# SwiftImpute: Beagle Alternative Implementation Plan

**Goal**: Transform SwiftImpute into a production-ready alternative to Java-based Beagle
**Priority**: Address critical gaps in accuracy validation, feature parity, and testing

---

## Phase 1: Accuracy Validation (CRITICAL - Week 1-2)

### 1.1 Create Validation Framework
- [ ] Build accuracy testing infrastructure
  - Implement concordance calculation (genotype match rate)
  - Implement R² correlation metric (dosage correlation)
  - Implement INFO score calculation per variant
  - Add MAF-stratified accuracy reporting (common vs rare variants)

**Files to create:**
- `src/validation/accuracy_metrics.hpp` - Metric calculation interfaces
- `src/validation/accuracy_metrics.cpp` - Implementation
- `test/test_accuracy.cpp` - Validation test suite

### 1.2 Benchmark Dataset Preparation
- [ ] Prepare standard benchmark datasets
  - 1000 Genomes Phase 3 subset (chr20 or chr22)
  - HapMap3 reference panel
  - Create masked test sets (10%, 20%, 50% missing)

**Files to create:**
- `benchmarks/prepare_benchmark.sh` - Dataset preparation script
- `benchmarks/README.md` - Benchmark instructions

### 1.3 Beagle Comparison
- [ ] Run parallel imputation with Beagle 5.4
- [ ] Generate comparison report (R², concordance, runtime, memory)
- [ ] Target: R² > 0.95 for common variants (MAF > 5%)

**Files to create:**
- `benchmarks/run_comparison.sh` - Comparison runner
- `benchmarks/compare_results.py` - Analysis script

---

## Phase 2: Core Feature Parity (Week 2-4)

### 2.1 Multi-allelic Variant Support
- [ ] Update VCF parser for multi-allelic sites
  - Parse ALT field with multiple alleles (A,G,T)
  - Handle GT values > 1 (0/2, 1/2, etc.)
  - Update emission probability calculation for >2 alleles

**Files to modify:**
- `src/io/vcf_reader.hpp` - Multi-allelic parsing
- `src/kernels/emission.cu` - Multi-allele emission kernel
- `src/core/types.hpp` - Allele type extension

### 2.2 Multi-chromosome Support
- [ ] Enable processing multiple chromosomes in single run
  - Add chromosome iteration in main pipeline
  - Maintain separate PBWT indices per chromosome
  - Handle chromosome boundaries in output

**Files to modify:**
- `src/api/imputer.cpp` - Multi-chromosome orchestration
- `src/main.cpp` - CLI updates for chromosome handling

### 2.3 Pre-phasing Capability
- [ ] Implement phasing for unphased input
  - Add SHAPEIT-style pre-phasing pass
  - Integrate with existing HMM infrastructure
  - Support phased/unphased input auto-detection

**Files to create:**
- `src/phasing/prephaser.hpp` - Pre-phasing interface
- `src/phasing/prephaser.cu` - GPU phasing kernel

---

## Phase 3: Scalability & Performance (Week 4-6)

### 3.1 WGS-scale Testing
- [ ] Test with 8M+ variant reference panels
  - Optimize memory for large marker counts
  - Implement streaming/chunked reference loading
  - Profile and optimize bottlenecks

**Target**: Process chr1 (10M variants) in < 2 hours

### 3.2 GPU State Selection
- [ ] Complete GPU PBWT implementation
  - Port CPU `StateSelector` to CUDA kernel
  - Implement parallel divergence ranking
  - Optimize memory access patterns

**Files to modify:**
- `src/pbwt/pbwt_selector.cu` - Complete GPU implementation

### 3.3 Multi-GPU Implementation
- [ ] Complete `MultiGPUImputer` implementation
  - Sample partitioning across GPUs
  - Load balancing based on GPU memory
  - Result aggregation

**Files to modify:**
- `src/api/imputer.cpp` - Multi-GPU orchestration

---

## Phase 4: Production Hardening (Week 6-8)

### 4.1 Comprehensive Testing
- [ ] Unit tests for all components
  - VCF I/O edge cases
  - GPU kernel correctness
  - Memory management
  - Error handling

**Files to create:**
- `test/test_vcf_io.cpp` - VCF parsing tests
- `test/test_kernels.cu` - GPU kernel tests
- `test/test_pbwt.cpp` - PBWT index tests
- `test/test_integration.cpp` - End-to-end tests

### 4.2 Error Handling & Edge Cases
- [ ] Robust error handling
  - Missing data patterns (all missing, block missing)
  - Monomorphic sites
  - Sample/marker count mismatches
  - GPU memory exhaustion recovery

### 4.3 Linux Build & CI/CD
- [ ] Complete Linux support
  - Test CMake build on Ubuntu 22.04
  - Add GitHub Actions CI pipeline
  - Create conda package recipe

**Files to create:**
- `.github/workflows/ci.yml` - CI pipeline
- `conda/meta.yaml` - Conda recipe

---

## Phase 5: Extended Features (Week 8-12)

### 5.1 X Chromosome Handling
- [ ] Implement sex-aware imputation
  - Haploid male X imputation
  - PAR region handling
  - Sex inference from data

**Files to create:**
- `src/special/x_chromosome.hpp` - X chr handling

### 5.2 Rare Variant Optimization
- [ ] Improve rare variant imputation (MAF < 1%)
  - Extended state selection for rare variants
  - Larger reference panel search radius
  - Conditional rare variant handling

### 5.3 IBD Detection (Optional)
- [ ] Add IBD segment detection
  - PBWT-based IBD finding
  - IBD-aware imputation boost

---

## Implementation Priority Order

```
CRITICAL (Blocks Production Use):
├── 1.1 Accuracy metrics framework
├── 1.2 Benchmark datasets
├── 1.3 Beagle comparison
└── 4.1 Comprehensive testing

HIGH (Feature Parity):
├── 2.1 Multi-allelic support
├── 2.2 Multi-chromosome support
├── 3.1 WGS-scale testing
└── 4.3 Linux build & CI

MEDIUM (Performance):
├── 3.2 GPU state selection
├── 3.3 Multi-GPU implementation
└── 2.3 Pre-phasing capability

LOW (Extended):
├── 5.1 X chromosome
├── 5.2 Rare variant optimization
└── 5.3 IBD detection
```

---

## Success Criteria

### Minimum Viable Alternative (MVP)
- [ ] R² > 0.95 for MAF > 5% variants on 1000G benchmark
- [ ] R² > 0.80 for MAF 1-5% variants
- [ ] Process chr22 (1M variants, 5K samples) in < 10 minutes
- [ ] Linux and Windows builds passing CI
- [ ] Multi-allelic variant support

### Full Beagle Parity
- [ ] R² within 2% of Beagle 5.4 across MAF spectrum
- [ ] WGS-scale (8M+ variants) support
- [ ] Pre-phasing for unphased input
- [ ] X chromosome support
- [ ] 10-20× speed improvement over Beagle confirmed

---

## Resource Requirements

### Hardware for Testing
- NVIDIA GPU: A100 or RTX 4090 (40+ GB preferred for WGS)
- RAM: 64 GB+ for large reference panels
- Storage: 500 GB+ for benchmark datasets

### Reference Datasets
- 1000 Genomes Phase 3 (GRCh38)
- HapMap3 sites
- gnomAD (optional, for rare variants)

---

## File Structure After Implementation

```
SwiftImpute/
├── src/
│   ├── core/           # Existing
│   ├── io/             # Extended for multi-allelic
│   ├── pbwt/           # GPU selector completed
│   ├── kernels/        # Multi-allele emission
│   ├── api/            # Multi-chr, multi-GPU
│   ├── phasing/        # NEW: Pre-phasing
│   ├── validation/     # NEW: Accuracy metrics
│   └── special/        # NEW: X chr, rare variants
├── test/               # Comprehensive test suite
├── benchmarks/         # Benchmark scripts & data
├── .github/workflows/  # CI/CD
└── conda/              # Package recipes
```

---

## Next Immediate Steps

1. **Start with Phase 1.1**: Create accuracy metrics framework
2. **Download 1000G chr22**: Prepare benchmark dataset
3. **Run baseline Beagle**: Establish comparison target
4. **Implement R² calculation**: First validation metric

---

*This plan prioritizes accuracy validation as the critical blocker for production use, followed by feature parity items that affect the broadest user base.*
