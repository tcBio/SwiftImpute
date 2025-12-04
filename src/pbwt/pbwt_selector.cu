/**
 * GPU-accelerated PBWT state selection kernel
 *
 * Selects top-L haplotype states based on PBWT divergence scores.
 * This replaces the CPU bottleneck in state selection for large datasets.
 */

#include "pbwt_index.hpp"
#include "core/types.hpp"
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cub/cub.cuh>

namespace swiftimpute {
namespace pbwt {

// Constants for kernel configuration
constexpr uint32_t WARP_SIZE = 32;
constexpr uint32_t MAX_STATES_PER_BLOCK = 256;  // Max L value handled per block
constexpr uint32_t THREADS_PER_BLOCK = 256;

// Structure for sorting (divergence, haplotype) pairs
struct DivergenceHaplotypePair {
    marker_t divergence;
    haplotype_t haplotype;

    __device__ __host__ DivergenceHaplotypePair() : divergence(0), haplotype(0) {}
    __device__ __host__ DivergenceHaplotypePair(marker_t d, haplotype_t h) : divergence(d), haplotype(h) {}
};

// Comparator for descending sort (higher divergence = better match)
struct DivergenceDescending {
    __device__ __host__ bool operator()(const DivergenceHaplotypePair& a,
                                        const DivergenceHaplotypePair& b) const {
        return a.divergence > b.divergence;
    }
};

/**
 * Kernel: Select top-L states for each marker
 *
 * Grid: (num_markers, num_samples, 1)
 * Block: (THREADS_PER_BLOCK, 1, 1)
 *
 * Each block handles one (marker, sample) pair, finding top-L states.
 */
__global__ void select_states_kernel(
    const haplotype_t* __restrict__ d_prefix,        // [num_markers][num_haplotypes]
    const marker_t* __restrict__ d_divergence,       // [num_markers][num_haplotypes]
    const allele_t* __restrict__ d_target_haplotypes,// [num_samples][2][num_markers] (optional, unused for now)
    uint32_t num_markers,
    uint32_t num_haplotypes,
    uint32_t num_states_L,
    haplotype_t* __restrict__ d_selected_states      // [num_samples][num_markers][num_states_L]
) {
    // Identify which (marker, sample) this block is processing
    uint32_t marker = blockIdx.x;
    uint32_t sample = blockIdx.y;
    uint32_t tid = threadIdx.x;

    if (marker >= num_markers) return;

    // Shared memory for top-L candidates
    extern __shared__ char shared_mem[];
    DivergenceHaplotypePair* candidates = reinterpret_cast<DivergenceHaplotypePair*>(shared_mem);

    // Each thread maintains its local top-L candidates
    // We use a simple approach: each thread processes a chunk of haplotypes
    // and keeps track of its local top-L, then we merge across threads

    uint32_t haps_per_thread = (num_haplotypes + blockDim.x - 1) / blockDim.x;
    uint32_t start_hap = tid * haps_per_thread;
    uint32_t end_hap = min(start_hap + haps_per_thread, num_haplotypes);

    // Base pointers for this marker
    const haplotype_t* prefix_row = d_prefix + static_cast<size_t>(marker) * num_haplotypes;
    const marker_t* divergence_row = d_divergence + static_cast<size_t>(marker) * num_haplotypes;

    // Each thread finds its best candidates using a partial sort approach
    // For small L, we use insertion sort to maintain top-L in shared memory

    // Initialize shared memory slots for this thread's top candidates
    uint32_t local_top_count = 0;
    uint32_t local_slots_start = tid * num_states_L;

    // Process assigned haplotypes
    for (uint32_t i = start_hap; i < end_hap && local_top_count < num_states_L; ++i) {
        haplotype_t hap = prefix_row[i];
        marker_t div = divergence_row[i];

        // Insert into local top-L using insertion sort
        uint32_t insert_pos = local_top_count;
        for (uint32_t j = 0; j < local_top_count; ++j) {
            if (div > candidates[local_slots_start + j].divergence) {
                insert_pos = j;
                break;
            }
        }

        if (insert_pos < num_states_L) {
            // Shift elements down
            for (uint32_t j = min(local_top_count, num_states_L - 1); j > insert_pos; --j) {
                candidates[local_slots_start + j] = candidates[local_slots_start + j - 1];
            }
            candidates[local_slots_start + insert_pos] = DivergenceHaplotypePair(div, hap);
            local_top_count = min(local_top_count + 1, num_states_L);
        }
    }

    // Continue processing remaining haplotypes, only keeping if better than current worst
    for (uint32_t i = start_hap + num_states_L; i < end_hap; ++i) {
        haplotype_t hap = prefix_row[i];
        marker_t div = divergence_row[i];

        // Check if this is better than our worst candidate
        if (local_top_count > 0 && div > candidates[local_slots_start + local_top_count - 1].divergence) {
            // Find insertion point
            uint32_t insert_pos = local_top_count - 1;
            for (uint32_t j = 0; j < local_top_count - 1; ++j) {
                if (div > candidates[local_slots_start + j].divergence) {
                    insert_pos = j;
                    break;
                }
            }

            // Shift and insert
            for (uint32_t j = local_top_count - 1; j > insert_pos; --j) {
                candidates[local_slots_start + j] = candidates[local_slots_start + j - 1];
            }
            candidates[local_slots_start + insert_pos] = DivergenceHaplotypePair(div, hap);
        }
    }

    __syncthreads();

    // Now we need to merge all thread-local top-L lists into a global top-L
    // Use a tree reduction approach

    // For simplicity, thread 0 does the final merge
    // This is efficient for small L values (L <= 64)
    if (tid == 0) {
        // Collect all candidates from all threads
        DivergenceHaplotypePair global_top[MAX_STATES_PER_BLOCK];
        uint32_t global_count = 0;

        // Merge all thread-local lists
        for (uint32_t t = 0; t < blockDim.x; ++t) {
            uint32_t t_start = t * num_states_L;
            // Check how many valid entries this thread has
            for (uint32_t i = 0; i < num_states_L; ++i) {
                if (candidates[t_start + i].divergence == 0 && candidates[t_start + i].haplotype == 0) {
                    // Could be uninitialized, but haplotype 0 with div 0 is valid
                    // We need a better sentinel, but for now include it
                }

                DivergenceHaplotypePair cand = candidates[t_start + i];

                // Insert into global_top using insertion sort
                if (global_count < num_states_L) {
                    // Find insertion point
                    uint32_t insert_pos = global_count;
                    for (uint32_t j = 0; j < global_count; ++j) {
                        if (cand.divergence > global_top[j].divergence) {
                            insert_pos = j;
                            break;
                        }
                    }
                    // Shift and insert
                    for (uint32_t j = global_count; j > insert_pos; --j) {
                        global_top[j] = global_top[j - 1];
                    }
                    global_top[insert_pos] = cand;
                    global_count++;
                } else if (cand.divergence > global_top[num_states_L - 1].divergence) {
                    // Better than worst in global_top
                    uint32_t insert_pos = num_states_L - 1;
                    for (uint32_t j = 0; j < num_states_L - 1; ++j) {
                        if (cand.divergence > global_top[j].divergence) {
                            insert_pos = j;
                            break;
                        }
                    }
                    for (uint32_t j = num_states_L - 1; j > insert_pos; --j) {
                        global_top[j] = global_top[j - 1];
                    }
                    global_top[insert_pos] = cand;
                }
            }
        }

        // Write output
        haplotype_t* output = d_selected_states +
            (static_cast<size_t>(sample) * num_markers + marker) * num_states_L;

        for (uint32_t i = 0; i < num_states_L; ++i) {
            if (i < global_count) {
                output[i] = global_top[i].haplotype;
            } else {
                // Fill with sequential haplotypes if not enough candidates
                output[i] = i % num_haplotypes;
            }
        }
    }
}

/**
 * Optimized kernel for small L values (L <= 32) using warp-level primitives
 *
 * Each warp handles one (marker, sample) pair
 * Grid: ((num_markers * num_samples + warps_per_block - 1) / warps_per_block, 1, 1)
 */
__global__ void select_states_warp_kernel(
    const haplotype_t* __restrict__ d_prefix,
    const marker_t* __restrict__ d_divergence,
    uint32_t num_markers,
    uint32_t num_haplotypes,
    uint32_t num_states_L,
    uint32_t num_samples,
    haplotype_t* __restrict__ d_selected_states
) {
    uint32_t global_tid = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t warp_id = global_tid / WARP_SIZE;
    uint32_t lane = global_tid % WARP_SIZE;

    uint32_t total_pairs = num_markers * num_samples;
    if (warp_id >= total_pairs) return;

    uint32_t sample = warp_id / num_markers;
    uint32_t marker = warp_id % num_markers;

    // Each lane in the warp processes different haplotypes
    // and tracks its best candidate

    const haplotype_t* prefix_row = d_prefix + static_cast<size_t>(marker) * num_haplotypes;
    const marker_t* divergence_row = d_divergence + static_cast<size_t>(marker) * num_haplotypes;

    // Each lane maintains local top candidates
    marker_t local_best_div[4] = {0, 0, 0, 0};  // Track top 4 per lane for L up to 128
    haplotype_t local_best_hap[4] = {0, 0, 0, 0};
    uint32_t local_count = 0;

    // Process haplotypes in strided fashion
    for (uint32_t i = lane; i < num_haplotypes; i += WARP_SIZE) {
        haplotype_t hap = prefix_row[i];
        marker_t div = divergence_row[i];

        // Update local top-4
        for (uint32_t j = 0; j < 4; ++j) {
            if (local_count <= j || div > local_best_div[j]) {
                // Shift down
                for (uint32_t k = 3; k > j; --k) {
                    local_best_div[k] = local_best_div[k-1];
                    local_best_hap[k] = local_best_hap[k-1];
                }
                local_best_div[j] = div;
                local_best_hap[j] = hap;
                local_count = min(local_count + 1, 4u);
                break;
            }
        }
    }

    // Now use warp shuffle to collect best candidates across lanes
    // For L=8, we need to find the top 8 across all 32 lanes

    // Output pointer for this (sample, marker)
    haplotype_t* output = d_selected_states +
        (static_cast<size_t>(sample) * num_markers + marker) * num_states_L;

    // Simple approach: lane 0 collects from all lanes
    // For better performance, use warp-level reduction

    // Broadcast local_best_div[0] and local_best_hap[0] from each lane
    // Use multiple rounds to find top L

    for (uint32_t out_idx = 0; out_idx < num_states_L && out_idx < local_count; ++out_idx) {
        // Find global max across warp
        marker_t my_div = (out_idx < local_count) ? local_best_div[out_idx] : 0;
        haplotype_t my_hap = (out_idx < local_count) ? local_best_hap[out_idx] : 0;

        // Warp-level max reduction
        marker_t max_div = my_div;
        for (int offset = 16; offset > 0; offset >>= 1) {
            marker_t other_div = __shfl_down_sync(0xFFFFFFFF, max_div, offset);
            max_div = max(max_div, other_div);
        }
        max_div = __shfl_sync(0xFFFFFFFF, max_div, 0);

        // Find which lane has the max (could be multiple, take first)
        uint32_t winner_mask = __ballot_sync(0xFFFFFFFF, my_div == max_div);
        uint32_t winner_lane = __ffs(winner_mask) - 1;

        // Broadcast winner's haplotype
        haplotype_t winner_hap = __shfl_sync(0xFFFFFFFF, my_hap, winner_lane);

        if (lane == 0) {
            output[out_idx] = winner_hap;
        }

        // Mark the winner's entry as used (set divergence to 0)
        if (lane == winner_lane) {
            local_best_div[out_idx] = 0;
        }
    }

    // Fill remaining slots if needed
    if (lane == 0) {
        for (uint32_t i = local_count; i < num_states_L; ++i) {
            output[i] = i % num_haplotypes;
        }
    }
}

/**
 * High-performance kernel using bitonic sort for medium L values
 *
 * Grid: (num_samples, (num_markers + markers_per_block - 1) / markers_per_block, 1)
 * Block: (THREADS_PER_BLOCK, 1, 1)
 */
__global__ void select_states_bitonic_kernel(
    const haplotype_t* __restrict__ d_prefix,
    const marker_t* __restrict__ d_divergence,
    uint32_t num_markers,
    uint32_t num_haplotypes,
    uint32_t num_states_L,
    haplotype_t* __restrict__ d_selected_states
) {
    uint32_t sample = blockIdx.x;
    uint32_t marker_base = blockIdx.y * blockDim.x;
    uint32_t local_marker = threadIdx.x;
    uint32_t marker = marker_base + local_marker;

    if (marker >= num_markers) return;

    // Each thread processes one marker
    const haplotype_t* prefix_row = d_prefix + static_cast<size_t>(marker) * num_haplotypes;
    const marker_t* divergence_row = d_divergence + static_cast<size_t>(marker) * num_haplotypes;

    // Use local memory for top-L selection
    marker_t top_div[64];      // Support up to L=64
    haplotype_t top_hap[64];
    uint32_t top_count = 0;

    // Simple linear scan with insertion sort for top-L
    for (uint32_t i = 0; i < num_haplotypes; ++i) {
        haplotype_t hap = prefix_row[i];
        marker_t div = divergence_row[i];

        if (top_count < num_states_L) {
            // Insert maintaining sorted order
            uint32_t pos = top_count;
            for (uint32_t j = 0; j < top_count; ++j) {
                if (div > top_div[j]) {
                    pos = j;
                    break;
                }
            }
            for (uint32_t j = top_count; j > pos; --j) {
                top_div[j] = top_div[j-1];
                top_hap[j] = top_hap[j-1];
            }
            top_div[pos] = div;
            top_hap[pos] = hap;
            top_count++;
        } else if (div > top_div[num_states_L - 1]) {
            // Better than worst, insert
            uint32_t pos = num_states_L - 1;
            for (uint32_t j = 0; j < num_states_L - 1; ++j) {
                if (div > top_div[j]) {
                    pos = j;
                    break;
                }
            }
            for (uint32_t j = num_states_L - 1; j > pos; --j) {
                top_div[j] = top_div[j-1];
                top_hap[j] = top_hap[j-1];
            }
            top_div[pos] = div;
            top_hap[pos] = hap;
        }
    }

    // Write output
    haplotype_t* output = d_selected_states +
        (static_cast<size_t>(sample) * num_markers + marker) * num_states_L;

    for (uint32_t i = 0; i < num_states_L; ++i) {
        if (i < top_count) {
            output[i] = top_hap[i];
        } else {
            output[i] = i % num_haplotypes;
        }
    }
}

// ============================================================================
// Host wrapper functions
// ============================================================================

void launch_select_states(
    const haplotype_t* d_prefix,
    const marker_t* d_divergence,
    const allele_t* d_target_haplotypes,  // Can be nullptr if not using target-specific selection
    uint32_t num_samples,
    uint32_t num_markers,
    uint32_t num_haplotypes,
    uint32_t num_states_L,
    haplotype_t* d_selected_states,
    cudaStream_t stream
) {
    // Choose kernel based on L value and problem size

    if (num_states_L <= 32 && num_haplotypes <= 10000) {
        // Use warp-based kernel for small L
        uint32_t total_pairs = num_markers * num_samples;
        uint32_t warps_per_block = THREADS_PER_BLOCK / WARP_SIZE;
        uint32_t num_blocks = (total_pairs + warps_per_block - 1) / warps_per_block;

        select_states_warp_kernel<<<num_blocks, THREADS_PER_BLOCK, 0, stream>>>(
            d_prefix,
            d_divergence,
            num_markers,
            num_haplotypes,
            num_states_L,
            num_samples,
            d_selected_states
        );
    } else if (num_states_L <= 64) {
        // Use bitonic kernel for medium L - one thread per marker
        dim3 grid(num_samples, (num_markers + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK);
        dim3 block(THREADS_PER_BLOCK);

        select_states_bitonic_kernel<<<grid, block, 0, stream>>>(
            d_prefix,
            d_divergence,
            num_markers,
            num_haplotypes,
            num_states_L,
            d_selected_states
        );
    } else {
        // Use block-based kernel for large L
        dim3 grid(num_markers, num_samples);
        dim3 block(THREADS_PER_BLOCK);
        size_t shared_mem = THREADS_PER_BLOCK * num_states_L * sizeof(DivergenceHaplotypePair);

        select_states_kernel<<<grid, block, shared_mem, stream>>>(
            d_prefix,
            d_divergence,
            d_target_haplotypes,
            num_markers,
            num_haplotypes,
            num_states_L,
            d_selected_states
        );
    }

    // Check for errors
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        throw CUDAError("launch_select_states failed", static_cast<int>(err));
    }
}

// ============================================================================
// GPUStateSelector implementation
// ============================================================================

// These functions provide the implementation for GPUStateSelector class methods
// defined in pbwt_index.hpp

void gpu_state_selector_allocate(
    const PBWTIndex& index,
    int device_id,
    haplotype_t** d_prefix,
    marker_t** d_divergence
) {
    CHECK_CUDA(cudaSetDevice(device_id));

    size_t prefix_size = static_cast<size_t>(index.num_markers()) * index.num_haplotypes();
    size_t divergence_size = prefix_size;

    CHECK_CUDA(cudaMalloc(d_prefix, prefix_size * sizeof(haplotype_t)));
    CHECK_CUDA(cudaMalloc(d_divergence, divergence_size * sizeof(marker_t)));
}

void gpu_state_selector_transfer(
    const PBWTIndex& index,
    haplotype_t* d_prefix,
    marker_t* d_divergence,
    cudaStream_t stream
) {
    size_t prefix_size = static_cast<size_t>(index.num_markers()) * index.num_haplotypes();
    size_t divergence_size = prefix_size;

    CHECK_CUDA(cudaMemcpyAsync(
        d_prefix,
        index.prefix().data.data(),
        prefix_size * sizeof(haplotype_t),
        cudaMemcpyHostToDevice,
        stream
    ));

    CHECK_CUDA(cudaMemcpyAsync(
        d_divergence,
        index.divergence().data.data(),
        divergence_size * sizeof(marker_t),
        cudaMemcpyHostToDevice,
        stream
    ));

    CHECK_CUDA(cudaStreamSynchronize(stream));
}

void gpu_state_selector_free(
    haplotype_t* d_prefix,
    marker_t* d_divergence
) {
    if (d_prefix) cudaFree(d_prefix);
    if (d_divergence) cudaFree(d_divergence);
}

size_t gpu_state_selector_memory_usage(
    marker_t num_markers,
    haplotype_t num_haplotypes
) {
    size_t array_size = static_cast<size_t>(num_markers) * num_haplotypes;
    return array_size * sizeof(haplotype_t) +  // prefix
           array_size * sizeof(marker_t);       // divergence
}

} // namespace pbwt
} // namespace swiftimpute
