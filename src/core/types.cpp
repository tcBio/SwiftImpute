#include "types.hpp"
#include <cuda_runtime.h>
#include <iostream>
#include <mutex>

namespace swiftimpute {

// Logger implementation

Logger& Logger::instance() {
    static Logger logger;
    return logger;
}

void Logger::log(Level level, const std::string& prefix, const std::string& msg) {
    if (level < level_) return;

    static std::mutex log_mutex;
    std::lock_guard<std::mutex> lock(log_mutex);

    std::cout << prefix << msg << std::endl;
}

void Logger::debug(const std::string& msg) {
    log(DEBUG, "[DEBUG] ", msg);
}

void Logger::info(const std::string& msg) {
    log(INFO, "[INFO] ", msg);
}

void Logger::warning(const std::string& msg) {
    log(WARNING, "[WARNING] ", msg);
}

void Logger::error(const std::string& msg) {
    log(ERROR, "[ERROR] ", msg);
}

// Device information functions

DeviceInfo get_device_info(int device_id) {
    DeviceInfo info;
    info.device_id = device_id;

    cudaDeviceProp prop;
    cudaError_t err = cudaGetDeviceProperties(&prop, device_id);

    if (err != cudaSuccess) {
        return info;
    }

    info.name = prop.name;
    info.compute_capability_major = prop.major;
    info.compute_capability_minor = prop.minor;
    info.multiprocessor_count = prop.multiProcessorCount;
    info.max_threads_per_block = prop.maxThreadsPerBlock;
    info.max_threads_per_multiprocessor = prop.maxThreadsPerMultiProcessor;
    info.total_memory = prop.totalGlobalMem;

    size_t free_mem, total_mem;
    cudaSetDevice(device_id);
    cudaMemGetInfo(&free_mem, &total_mem);
    info.free_memory = free_mem;

    return info;
}

int select_best_device() {
    int device_count = 0;
    cudaError_t err = cudaGetDeviceCount(&device_count);

    if (err != cudaSuccess || device_count == 0) {
        throw CUDAError("No CUDA devices found", static_cast<int>(err));
    }

    // Select device with most free memory
    int best_device = 0;
    size_t max_free_memory = 0;

    for (int i = 0; i < device_count; ++i) {
        auto info = get_device_info(i);
        if (info.free_memory > max_free_memory) {
            max_free_memory = info.free_memory;
            best_device = i;
        }
    }

    return best_device;
}

size_t get_available_gpu_memory(int device_id) {
    cudaSetDevice(device_id);
    size_t free_mem, total_mem;
    cudaMemGetInfo(&free_mem, &total_mem);
    return free_mem;
}

size_t get_total_gpu_memory(int device_id) {
    cudaSetDevice(device_id);
    size_t free_mem, total_mem;
    cudaMemGetInfo(&free_mem, &total_mem);
    return total_mem;
}

// PackedHaplotypes implementation

PackedHaplotypes::PackedHaplotypes(marker_t num_markers, haplotype_t num_haplotypes)
    : num_markers_(num_markers), num_haplotypes_(num_haplotypes)
{
    // 4 alleles per byte (2 bits each)
    size_t num_bytes = (static_cast<size_t>(num_markers) * num_haplotypes + 3) / 4;
    data_.resize(num_bytes, 0);
}

PackedHaplotypes::~PackedHaplotypes() {
    // Vector handles cleanup
}

void PackedHaplotypes::set_allele(marker_t m, haplotype_t h, allele_t value) {
    size_t offset = get_offset(m, h);
    size_t byte_idx = offset / 4;
    size_t bit_idx = (offset % 4) * 2;

    // Clear the 2 bits
    data_[byte_idx] &= ~(0x03 << bit_idx);

    // Set the 2 bits
    data_[byte_idx] |= ((value & 0x03) << bit_idx);
}

allele_t PackedHaplotypes::get_allele(marker_t m, haplotype_t h) const {
    size_t offset = get_offset(m, h);
    size_t byte_idx = offset / 4;
    size_t bit_idx = (offset % 4) * 2;

    return (data_[byte_idx] >> bit_idx) & 0x03;
}

size_t PackedHaplotypes::get_offset(marker_t m, haplotype_t h) const {
    return static_cast<size_t>(m) * num_haplotypes_ + h;
}

} // namespace swiftimpute
