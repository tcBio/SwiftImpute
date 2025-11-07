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

// NOTE: Other implementations (PackedHaplotypes, Logger, device functions)
// are in types.cpp to avoid duplicate symbols

} // namespace swiftimpute
