#include "core/memory_pool.hpp"
#include "core/types.hpp"
#include <cuda_runtime.h>
#include <stdexcept>
#include <iostream>

namespace swiftimpute {

// DevicePtr implementation

template<typename T>
void DevicePtr<T>::allocate() {
    if (size_ == 0) return;
    
    cudaError_t error = cudaMalloc(&ptr_, size_ * sizeof(T));
    if (error != cudaSuccess) {
        throw CUDAError("Failed to allocate device memory", static_cast<int>(error));
    }
}

template<typename T>
void DevicePtr<T>::deallocate() {
    if (ptr_) {
        cudaFree(ptr_);
        ptr_ = nullptr;
    }
}

template<typename T>
void DevicePtr<T>::copy_from_host(const T* host_data, size_t count) {
    if (count > size_) {
        throw ImputationError("Copy count exceeds allocated size");
    }
    
    CHECK_CUDA(cudaMemcpy(ptr_, host_data, count * sizeof(T), cudaMemcpyHostToDevice));
}

template<typename T>
void DevicePtr<T>::copy_to_host(T* host_data, size_t count) const {
    if (count > size_) {
        throw ImputationError("Copy count exceeds allocated size");
    }
    
    CHECK_CUDA(cudaMemcpy(host_data, ptr_, count * sizeof(T), cudaMemcpyDeviceToHost));
}

// Explicit template instantiations for common types
template class DevicePtr<uint8_t>;
template class DevicePtr<uint32_t>;
template class DevicePtr<float>;
template class DevicePtr<double>;
template class DevicePtr<GenotypeLikelihoods>;

// PinnedPtr implementation

template<typename T>
void PinnedPtr<T>::allocate() {
    if (size_ == 0) return;
    
    cudaError_t error = cudaMallocHost(&ptr_, size_ * sizeof(T));
    if (error != cudaSuccess) {
        throw CUDAError("Failed to allocate pinned memory", static_cast<int>(error));
    }
}

template<typename T>
void PinnedPtr<T>::deallocate() {
    if (ptr_) {
        cudaFreeHost(ptr_);
        ptr_ = nullptr;
    }
}

// Explicit template instantiations
template class PinnedPtr<uint8_t>;
template class PinnedPtr<uint32_t>;
template class PinnedPtr<float>;
template class PinnedPtr<double>;
template class PinnedPtr<GenotypeLikelihoods>;

// PackedHaplotypes implementation

PackedHaplotypes::PackedHaplotypes(marker_t num_markers, haplotype_t num_haplotypes)
    : num_markers_(num_markers), num_haplotypes_(num_haplotypes) {
    
    // 4 alleles per byte (2 bits each)
    size_t total_alleles = static_cast<size_t>(num_markers) * num_haplotypes;
    size_t num_bytes = (total_alleles + 3) / 4;  // Round up
    data_.resize(num_bytes, 0);
}

PackedHaplotypes::~PackedHaplotypes() = default;

void PackedHaplotypes::set_allele(marker_t m, haplotype_t h, allele_t value) {
    if (m >= num_markers_ || h >= num_haplotypes_) {
        throw ImputationError("Index out of bounds");
    }
    
    size_t bit_offset = get_offset(m, h);
    size_t byte_idx = bit_offset / 8;
    size_t bit_pos = bit_offset % 8;
    
    // Clear existing 2 bits
    data_[byte_idx] &= ~(0x03 << bit_pos);
    
    // Set new value (only use lower 2 bits)
    data_[byte_idx] |= ((value & 0x03) << bit_pos);
}

allele_t PackedHaplotypes::get_allele(marker_t m, haplotype_t h) const {
    if (m >= num_markers_ || h >= num_haplotypes_) {
        throw ImputationError("Index out of bounds");
    }
    
    size_t bit_offset = get_offset(m, h);
    size_t byte_idx = bit_offset / 8;
    size_t bit_pos = bit_offset % 8;
    
    return (data_[byte_idx] >> bit_pos) & 0x03;
}

size_t PackedHaplotypes::get_offset(marker_t m, haplotype_t h) const {
    // Column-major layout: all haplotypes for marker m, then marker m+1
    // Each allele is 2 bits
    return (static_cast<size_t>(m) * num_haplotypes_ + h) * 2;
}

// Logger implementation

Logger& Logger::instance() {
    static Logger instance;
    return instance;
}

void Logger::debug(const std::string& msg) {
    if (level_ <= DEBUG) {
        log(DEBUG, "[DEBUG]", msg);
    }
}

void Logger::info(const std::string& msg) {
    if (level_ <= INFO) {
        log(INFO, "[INFO]", msg);
    }
}

void Logger::warning(const std::string& msg) {
    if (level_ <= WARNING) {
        log(WARNING, "[WARNING]", msg);
    }
}

void Logger::error(const std::string& msg) {
    if (level_ <= ERROR) {
        log(ERROR, "[ERROR]", msg);
    }
}

void Logger::log(Level level, const std::string& prefix, const std::string& msg) {
    // Simple console output - can be extended with file logging, threading, etc.
    std::ostream& out = (level >= ERROR) ? std::cerr : std::cout;
    out << prefix << " " << msg << std::endl;
}

// Device information functions

DeviceInfo get_device_info(int device_id) {
    DeviceInfo info;
    info.device_id = device_id;
    
    cudaDeviceProp prop;
    CHECK_CUDA(cudaGetDeviceProperties(&prop, device_id));
    
    info.name = prop.name;
    info.total_memory = prop.totalGlobalMem;
    info.compute_capability_major = prop.major;
    info.compute_capability_minor = prop.minor;
    info.multiprocessor_count = prop.multiProcessorCount;
    info.max_threads_per_block = prop.maxThreadsPerBlock;
    info.max_threads_per_multiprocessor = prop.maxThreadsPerMultiProcessor;
    
    // Get free memory
    size_t free_mem, total_mem;
    CHECK_CUDA(cudaMemGetInfo(&free_mem, &total_mem));
    info.free_memory = free_mem;
    
    return info;
}

int select_best_device() {
    int device_count;
    CHECK_CUDA(cudaGetDeviceCount(&device_count));
    
    if (device_count == 0) {
        throw ImputationError("No CUDA-capable devices found");
    }
    
    // Select device with most free memory and highest compute capability
    int best_device = 0;
    size_t max_free_memory = 0;
    int max_compute_capability = 0;
    
    for (int i = 0; i < device_count; i++) {
        DeviceInfo info = get_device_info(i);
        int compute_cap = info.compute_capability_major * 10 + info.compute_capability_minor;
        
        // Prioritize compute capability, then free memory
        if (compute_cap > max_compute_capability ||
            (compute_cap == max_compute_capability && info.free_memory > max_free_memory)) {
            best_device = i;
            max_compute_capability = compute_cap;
            max_free_memory = info.free_memory;
        }
    }
    
    return best_device;
}

size_t get_available_gpu_memory(int device_id) {
    int current_device;
    CHECK_CUDA(cudaGetDevice(&current_device));
    
    if (device_id != current_device) {
        CHECK_CUDA(cudaSetDevice(device_id));
    }
    
    size_t free_mem, total_mem;
    CHECK_CUDA(cudaMemGetInfo(&free_mem, &total_mem));
    
    if (device_id != current_device) {
        CHECK_CUDA(cudaSetDevice(current_device));
    }
    
    return free_mem;
}

size_t get_total_gpu_memory(int device_id) {
    cudaDeviceProp prop;
    CHECK_CUDA(cudaGetDeviceProperties(&prop, device_id));
    return prop.totalGlobalMem;
}

} // namespace swiftimpute
