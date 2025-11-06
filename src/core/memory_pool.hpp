#pragma once

#include "types.hpp"

// This file provides GPU memory management utilities
// Implementation in memory_pool.cu

namespace swiftimpute {

// Memory pool for efficient allocation/deallocation
class MemoryPool {
public:
    MemoryPool(size_t initial_size_bytes, int device_id = 0);
    ~MemoryPool();
    
    // Allocate memory from pool
    void* allocate(size_t bytes);
    
    // Free memory back to pool
    void free(void* ptr);
    
    // Get total allocated memory
    size_t total_allocated() const { return total_allocated_; }
    
    // Get peak memory usage
    size_t peak_usage() const { return peak_usage_; }
    
    // Reset pool statistics
    void reset_stats();
    
private:
    struct Block {
        void* ptr;
        size_t size;
        bool in_use;
    };
    
    std::vector<Block> blocks_;
    size_t total_allocated_;
    size_t peak_usage_;
    int device_id_;
    
    void grow_pool(size_t min_size);
};

// RAII wrapper for temporary GPU memory
template<typename T>
class TempDeviceBuffer {
public:
    TempDeviceBuffer(size_t count) : count_(count) {
        CHECK_CUDA(cudaMalloc(&ptr_, count * sizeof(T)));
    }
    
    ~TempDeviceBuffer() {
        if (ptr_) {
            cudaFree(ptr_);
        }
    }
    
    // No copy
    TempDeviceBuffer(const TempDeviceBuffer&) = delete;
    TempDeviceBuffer& operator=(const TempDeviceBuffer&) = delete;
    
    // Move semantics
    TempDeviceBuffer(TempDeviceBuffer&& other) noexcept :
        ptr_(other.ptr_), count_(other.count_) {
        other.ptr_ = nullptr;
    }
    
    T* get() { return ptr_; }
    const T* get() const { return ptr_; }
    size_t size() const { return count_; }
    
private:
    T* ptr_;
    size_t count_;
};

} // namespace swiftimpute
