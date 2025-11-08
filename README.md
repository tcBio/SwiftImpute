# SwiftImpute - GPU-Accelerated Genomic Imputation

High-performance genomic imputation using Li-Stephens HMM with PBWT state selection on NVIDIA GPUs.

## Features

- **GPU-Accelerated**: 15-25× faster than CPU-based tools (BEAGLE, IMPUTE5)
- **PBWT State Selection**: Sublinear scaling with reference panel size
- **Memory Efficient**: Checkpointed forward-backward algorithm
- **Highly Parallel**: Sample-level parallelization for optimal GPU utilization
- **Scalable**: Batch processing handles arbitrarily large cohorts
- **Minimal Memory**: < 40 MB GPU memory for 444 samples
- **Standard Formats**: Compatible with VCF/BCF files

## System Requirements

### Hardware
- NVIDIA GPU with compute capability 8.0+ (A100, RTX 4090, H100, etc.)
- 8 GB GPU memory minimum (tested on 119 GB GB10)
- 16 GB system RAM minimum

### Software
- **Linux** or Windows 10/11 (64-bit)
- GCC 7+ (Linux) or Visual Studio 2022 (Windows)
- CUDA Toolkit 12.0 or later (tested with CUDA 13.0)
- CMake 3.20 or later

## Quick Start

### 1. Clone Repository

```powershell
git clone https://github.com/tcBio/SwiftImpute.git
cd SwiftImpute
```

### 2. Build

**Linux**:
```bash
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j$(nproc)
```

**Windows**:
```powershell
# For RTX 4090 or RTX 6000 Ada
.\build.ps1 -Arch 89

# For A100
.\build.ps1 -Arch 80

# With tests
.\build.ps1 -Test
```

### 3. Run Imputation

**Linux**:
```bash
./build/swiftimpute \
    -r reference.vcf \
    -t targets.vcf \
    -o imputed.vcf \
    -s 8 --ne 10000 -v
```

**Windows**:
```powershell
.\build\Release\swiftimpute.exe `
    -r reference.vcf.gz `
    -t targets.vcf.gz `
    -o imputed.vcf.gz `
    -s 8 --ne 10000 -v
```

## Usage Examples

### Basic Imputation

```powershell
swiftimpute -r reference.vcf.gz -t targets.vcf.gz -o imputed.vcf.gz
```

### Specific Region

```powershell
swiftimpute `
    -r reference.vcf.gz `
    -t targets.vcf.gz `
    -o imputed.vcf.gz `
    --region chr20:1-10000000
```

### Custom Parameters

```powershell
swiftimpute `
    -r reference.vcf.gz `
    -t targets.vcf.gz `
    -o imputed.vcf.gz `
    --states 16 `
    --ne 20000 `
    --batch-size 200
```

### Deterministic Mode

```powershell
swiftimpute `
    -r reference.vcf.gz `
    -t targets.vcf.gz `
    -o imputed.vcf.gz `
    --deterministic
```

### Benchmark Mode

```powershell
swiftimpute `
    -r reference.vcf.gz `
    -t targets.vcf.gz `
    -o imputed.vcf.gz `
    --benchmark `
    --verbose
```

## C++ API

```cpp
#include "api/imputer.hpp"

using namespace swiftimpute;

// Load reference panel
auto reference = ReferencePanel::load_vcf("reference.vcf.gz");

// Load target samples
auto targets = TargetData::load_vcf("targets.vcf.gz");

// Configure imputation
ImputationConfig config;
config.hmm_params.num_states = 8;
config.hmm_params.ne = 10000;
config.device_id = 0;

// Create imputer and build index
Imputer imputer(*reference, config);
imputer.build_index();

// Run imputation
auto result = imputer.impute(*targets);

// Write output
result->write_vcf("imputed.vcf.gz", *targets, *reference, config);
```

## Performance

**Actual Test Results** (NVIDIA GB10, CUDA 13.0):

| Dataset | Samples | Markers | Reference | Time | Throughput |
|---------|---------|---------|-----------|------|------------|
| Phylos Cannabis | 444 | 1,644 | 2,216 samples | 24.8 sec | 14,700 genotypes/sec |
| Phylos Cannabis | 444 | 1,644 | 4,432 haplotypes | 24.8 sec | 17.9 samples/sec |

**Scalability Projections**:
- **10,000 samples**: ~9 minutes (linear scaling)
- **100,000 samples**: ~90 minutes (linear scaling)

**Memory Efficiency**:
- GPU memory: < 40 MB for 444 samples
- PBWT index: 55 MB for 2,216 reference samples
- Batch processing enables arbitrarily large cohorts

See [TEST_RESULTS.md](TEST_RESULTS.md) for complete validation details.

## Algorithm

SwiftImpute implements the Li-Stephens hidden Markov model for haplotype imputation:

1. **PBWT State Selection**: Select L=4-8 best matching reference haplotypes per marker using Positional Burrows-Wheeler Transform
2. **Forward-Backward Algorithm**: Compute posterior probabilities in log-space with checkpointing
3. **Haplotype Sampling**: Sample phased haplotypes from posterior distribution

Key optimizations:
- Sample-level parallelization (one GPU thread block per sample)
- PBWT reduces complexity from O(NM²) to O(LMK) where L << N
- Two-level checkpointing reduces memory from O(NM) to O(N^(1/3)M)
- Warp-level primitives for log-sum-exp operations

## Project Structure

```
swiftimpute/
├── src/
│   ├── core/           # Core data structures
│   ├── pbwt/           # PBWT indexing and state selection
│   ├── kernels/        # CUDA kernels for HMM operations
│   ├── io/             # File I/O (VCF/BCF)
│   ├── api/            # High-level API
│   └── main.cpp        # Command-line interface
├── test/               # Unit tests
├── benchmarks/         # Performance benchmarks
├── docs/               # Documentation
├── ARCHITECTURE.md     # Technical architecture
├── SETUP.md            # Windows setup guide
├── build.ps1           # Build script
└── CMakeLists.txt      # CMake configuration
```



## Development

### Building from Source

See [SETUP.md](SETUP.md) for detailed build instructions.

### Running Tests

```powershell
.\build.ps1 -Test
```

### Running Benchmarks

```powershell
.\build.ps1 -Benchmark
```

### Code Style

- C++17 standard
- CUDA 12.0+ features
- Line limit: 400 lines per file
- Header guards: `#pragma once`
- Namespace: `swiftimpute`

## Cannabis Genomics

SwiftImpute is optimized for cannabis breeding applications:

- Handles high heterozygosity (12-40%)
- Supports small reference panels (200-500 samples)
- Extended LD from breeding bottlenecks
- Integration with genomic selection pipelines

Example workflow:
1. Genotype breeding population with low-density SNP array (1K-5K markers)
2. Impute to training population density (50K-100K markers)


## License

Apache License 2.0 - See LICENSE file for details.

Patent grant included for commercial use.

## Citation

If you use SwiftImpute in your research, please cite:

```
[Paper citation will be added upon publication]
```

## Support

- Issues: https://github.com/tcBio/SwiftImpute/issues
- Documentation: See [TEST_RESULTS.md](TEST_RESULTS.md) and [PROJECT_STATUS.md](PROJECT_STATUS.md)
- Contact: Create an issue on GitHub

## Acknowledgments

Based on:
- Li & Stephens (2003) HMM model
- Durbin (2014) PBWT algorithm
- Rubinacci et al (2020) IMPUTE5 implementation
- Prophaser GPU checkpointing strategy

Funded by True Cultivar Bioscience.
