# SwiftImpute - Next Implementation Steps

## Current Status: Phase 1 Complete ✅

All core infrastructure is working:
- ✅ VCF I/O system (450 lines)
- ✅ PBWT index with state selection (300 lines)
- ✅ Data loading pipeline (750 lines)
- ✅ Memory management & logging (250 lines)
- ✅ Complete test suite passing

**Total: 1950+ lines of tested C++/CUDA code**

## Phase 2: GPU-Accelerated HMM (Priority)

### 1. Emission Probability Kernel

**File**: `src/kernels/emission.cu` (new)

**Purpose**: Compute P(observed genotype | hidden state)

**Algorithm**:
```cpp
__global__ void compute_emission_probs(
    const GenotypeLikelihoods* genotype_liks,  // [num_samples][num_markers]
    const allele_t* reference_haplotypes,      // [num_markers][num_haplotypes]
    const haplotype_t* selected_states,        // [num_samples][num_markers][L]
    prob_t* emission_probs,                    // [num_samples][num_markers][L]
    uint32_t num_samples,
    uint32_t num_markers,
    uint32_t num_states
)
```

**Key Points**:
- Log-space computation for numerical stability
- Input: Genotype likelihoods (LL_00, LL_01, LL_11)
- For each state: look up reference alleles, compute likelihood
- Parallelize over samples and markers
- Use shared memory for selected states

**Estimated**: 100-150 lines

### 2. Transition Probability Kernel

**File**: `src/kernels/transition.cu` (new)

**Purpose**: Compute Li-Stephens transition probabilities

**Algorithm**:
```cpp
__global__ void compute_transition_probs(
    const double* genetic_distances,           // [num_markers]
    const HMMParameters params,                // theta, rho, ne
    prob_t* transition_matrices,               // [num_markers][L][L]
    uint32_t num_markers,
    uint32_t num_states
)
```

**Li-Stephens Model**:
```
P(j|i) = (1-rho) * delta(i,j) + rho / (2*ne)
where:
  rho = 1 - exp(-4 * ne * genetic_dist * rho_rate)
  theta = mutation rate (default 0.001)
```

**Key Points**:
- Compute once per marker (not per sample)
- Store in global memory, reuse for all samples
- Small matrices (8x8 for L=8), can use shared memory
- Symmetric for same allele transitions

**Estimated**: 80-100 lines

### 3. Forward Pass Kernel with Checkpointing

**File**: Extend `src/kernels/forward_backward.cpp`

**Purpose**: Compute forward probabilities α(m, s)

**Algorithm**:
```cpp
__global__ void forward_pass_kernel(
    const prob_t* emission_probs,              // [num_samples][num_markers][L]
    const prob_t* transition_probs,            // [num_markers][L][L]
    prob_t* alpha,                             // [num_samples][checkpoint_markers][L]
    prob_t* scaling_factors,                   // [num_samples][num_markers]
    uint32_t num_samples,
    uint32_t num_markers,
    uint32_t num_states,
    uint32_t checkpoint_interval
)
```

**Checkpointing Strategy**:
- Save α every √M markers (calculated by `calculate_checkpoint_interval()`)
- Reduces memory from O(M·L) to O(√M·L)
- One thread block per sample for sequential processing

**Key Implementation**:
```cpp
// Initialize at marker 0
for (int s = 0; s < L; s++) {
    alpha[0][s] = emission[0][s];
}

// Forward sweep
for (int m = 1; m < num_markers; m++) {
    for (int j = 0; j < L; j++) {
        prob_t sum = -INFINITY;
        for (int i = 0; i < L; i++) {
            prob_t val = alpha_prev[i] + trans[i][j];
            sum = logsumexp(sum, val);
        }
        alpha[m][j] = sum + emission[m][j];
    }

    // Checkpoint if needed
    if (m % checkpoint_interval == 0) {
        save_checkpoint(alpha[m]);
    }

    // Scaling for numerical stability
    scale_factor = max(alpha[m]);
    for (int j = 0; j < L; j++) {
        alpha[m][j] -= scale_factor;
    }
}
```

**Estimated**: 150-200 lines

### 4. Backward Pass Kernel

**File**: Same as forward pass

**Purpose**: Compute backward probabilities β(m, s)

**Algorithm**:
```cpp
__global__ void backward_pass_kernel(
    const prob_t* emission_probs,
    const prob_t* transition_probs,
    const prob_t* alpha_checkpoints,
    prob_t* beta,
    prob_t* posteriors,
    uint32_t num_samples,
    uint32_t num_markers,
    uint32_t num_states
)
```

**Checkpointing Usage**:
- Start from last marker
- Work backward to each checkpoint
- Recompute forward pass between checkpoints
- Combine α·β for posterior probabilities

**Key Implementation**:
```cpp
// Initialize at last marker
for (int s = 0; s < L; s++) {
    beta[M-1][s] = 0.0;  // log(1) = 0
}

// Backward sweep with checkpoint handling
for (int m = M-2; m >= 0; m--) {
    // If at checkpoint, restore forward probs
    if (m % checkpoint_interval == 0) {
        restore_checkpoint(m);
    }

    // Compute beta
    for (int i = 0; i < L; i++) {
        prob_t sum = -INFINITY;
        for (int j = 0; j < L; j++) {
            prob_t val = trans[i][j] + emission[m+1][j] + beta[m+1][j];
            sum = logsumexp(sum, val);
        }
        beta[m][i] = sum;
    }

    // Compute posterior: P(state | data) = α·β / Σ(α·β)
    for (int s = 0; s < L; s++) {
        posterior[m][s] = alpha[m][s] + beta[m][s];
    }
    normalize(posterior[m]);
}
```

**Estimated**: 150-200 lines

### 5. Haplotype Sampling Kernel

**File**: `src/kernels/sampling.cu` (new)

**Purpose**: Sample phased haplotypes from posterior distribution

**Algorithm**:
```cpp
__global__ void sample_haplotypes_kernel(
    const prob_t* posteriors,                  // [num_samples][num_markers][L]
    const haplotype_t* selected_states,        // [num_samples][num_markers][L]
    const allele_t* reference_haplotypes,      // [num_markers][num_haplotypes]
    allele_t* output_haplotypes,               // [num_samples][2][num_markers]
    curandState* rng_states,                   // [num_samples]
    bool deterministic,
    uint32_t num_samples,
    uint32_t num_markers,
    uint32_t num_states
)
```

**Two Modes**:

**Deterministic** (for testing/reproducibility):
```cpp
// Take most likely state
int best_state = argmax(posteriors[m]);
haplotype[m] = reference[selected_states[m][best_state]];
```

**Stochastic** (for proper sampling):
```cpp
// Sample from posterior distribution
float r = curand_uniform(&rng_state);
float cumsum = 0.0f;
for (int s = 0; s < L; s++) {
    cumsum += exp(posteriors[m][s]);
    if (r < cumsum) {
        haplotype[m] = reference[selected_states[m][s]];
        break;
    }
}
```

**Estimated**: 100-120 lines

## Implementation Order

### Week 1: Emission & Transition
1. Create `emission.cu` with kernel
2. Create `transition.cu` with Li-Stephens model
3. Add unit tests for both
4. Verify on CPU before GPU

### Week 2: Forward Pass
1. Implement forward pass with checkpointing
2. Add scaling for numerical stability
3. Test on small datasets (10-100 markers)
4. Profile memory usage

### Week 3: Backward Pass
1. Implement backward pass with checkpoint restoration
2. Combine forward/backward → posteriors
3. Test full forward-backward on medium datasets
4. Validate against known results

### Week 4: Sampling & Integration
1. Implement haplotype sampling kernel
2. Connect full pipeline: PBWT → HMM → Results
3. End-to-end testing with real VCF data
4. Performance benchmarking

## Memory Layout Optimization

### Current Approach (after Phase 2):
```
GPU Memory:
├─ Reference haplotypes: [M × N] bytes
├─ PBWT index: [M × N × 8] bytes (if transferred)
├─ Selected states: [batch_size × M × L × 4] bytes
├─ Emission probs: [batch_size × M × L × 4] bytes
├─ Transition probs: [M × L × L × 4] bytes
├─ Forward checkpoints: [batch_size × √M × L × 4] bytes
├─ Backward temps: [batch_size × L × 4] bytes
└─ Posteriors: [batch_size × M × L × 4] bytes
```

**For M=10K markers, N=2K haplotypes, L=8, batch=100**:
- Reference: 20 MB
- PBWT: 160 MB (if needed on GPU)
- Per-batch: ~50 MB
- Total: ~230 MB (well within 16 GB GPU)

## Testing Strategy

### Unit Tests
- `test_emission.cu`: Verify emission probabilities
- `test_transition.cu`: Check Li-Stephens calculations
- `test_forward.cu`: Test forward pass correctness
- `test_backward.cu`: Test backward pass correctness
- `test_sampling.cu`: Verify sampling distributions

### Integration Tests
- `test_hmm_small.cpp`: 10 markers, 4 samples
- `test_hmm_medium.cpp`: 1K markers, 100 samples
- `test_accuracy.cpp`: Compare to known ground truth

### Performance Tests
- `benchmark_throughput.cu`: Samples/second
- `benchmark_memory.cu`: Memory usage vs batch size
- `benchmark_scaling.cu`: Scaling with M, N, L

## Validation Against Ground Truth

### Strategy:
1. Use 1000 Genomes data (chr20, small region)
2. Run BEAGLE 5 or IMPUTE5 as reference
3. Compare:
   - Dosage correlation (r²)
   - Genotype concordance
   - INFO score agreement
   - Phasing switch error rate

### Target Metrics:
- Dosage r² > 0.95 (vs reference)
- Genotype concordance > 98%
- INFO score within ±0.05
- Switch error < 1% (for phasing accuracy)

## Performance Targets (Phase 2 Complete)

### Expected Speed:
- **Small**: 10K markers, 100 samples, 1K ref → <5 seconds
- **Medium**: 100K markers, 1K samples, 10K ref → 2-3 minutes
- **Large**: 1M markers, 10K samples, 100K ref → 30-45 minutes

### Comparison to CPU Tools:
- BEAGLE 5: 20-40 min (medium dataset)
- IMPUTE5: 8-15 min (medium dataset)
- SwiftImpute target: 2-3 min (20× faster)

## GPU Optimization Opportunities (Phase 3)

After basic kernels work:
1. **Kernel fusion**: Combine emission + transition
2. **Shared memory**: Cache state arrays
3. **Warp-level primitives**: Faster reductions
4. **Async streams**: Overlap compute + transfer
5. **Multi-GPU**: Split samples across devices

## Code Architecture After Phase 2

```
src/
├── core/               [✅ Complete]
│   ├── types.hpp/cpp   - Data structures, logger
│   └── memory_pool     - GPU memory management
│
├── io/                 [✅ Complete]
│   ├── vcf_reader      - VCF parsing
│   └── vcf_writer      - VCF output
│
├── pbwt/               [✅ Complete]
│   └── pbwt_index      - Index + state selection
│
├── kernels/            [🔄 Phase 2]
│   ├── logsumexp       [✅ Working]
│   ├── emission        [⏳ TODO]
│   ├── transition      [⏳ TODO]
│   ├── forward_backward[⏳ Extend]
│   └── sampling        [⏳ TODO]
│
└── api/                [✅ Complete + Phase 2 integration]
    └── imputer         - Main pipeline
```

## Questions to Resolve

### Design Decisions:
1. **State selection on CPU or GPU?**
   - CPU: Easier debugging, less transfer
   - GPU: Faster for large L, better pipeline
   - **Recommendation**: Start CPU, optimize later

2. **Memory layout for posteriors?**
   - Option A: [sample][marker][state] - cache friendly
   - Option B: [marker][sample][state] - coalesced access
   - **Recommendation**: Option A for forward/backward

3. **Batch size strategy?**
   - Fixed: Simple, predictable memory
   - Dynamic: Adapt to GPU memory
   - **Recommendation**: Fixed with user override

## Resources Needed

### Hardware:
- NVIDIA GPU with 16+ GB memory (RTX 5070)
- 32+ GB system RAM
- Fast SSD for VCF I/O

### Software:
- CUDA Toolkit 13.0+ ✅
- cuRAND for sampling
- Optional: NVIDIA Nsight for profiling

### Data:
- 1000 Genomes phase 3 (validation)
- HapMap recombination maps
- Cannabis reference panels (target application)

## Success Criteria for Phase 2

✅ **Must Have**:
- All kernels compile and run
- Forward-backward produces valid posteriors
- Sampled haplotypes match input distribution
- Tests pass on synthetic data
- No memory leaks

🎯 **Should Have**:
- Speed > 10× faster than CPU baseline
- Accuracy matches CPU implementation
- Works with batch sizes 10-1000
- Handles markers up to 100K

🌟 **Nice to Have**:
- Speed > 20× faster than BEAGLE
- Multi-GPU support
- Automatic batch size tuning
- Real-time progress estimation

---

**Last Updated**: 2025-01-06
**Phase 1 Completion**: 100%
**Phase 2 Readiness**: Ready to start
**Estimated Phase 2 Timeline**: 3-4 weeks

*Ready to implement GPU kernels! 🚀*
