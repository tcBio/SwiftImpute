#include "core/types.hpp"
#include "core/memory_pool.hpp"
#include <cuda_runtime.h>
#include <iostream>
#include <cassert>

using namespace swiftimpute;

// Simple test harness
struct TestResult {
    std::string name;
    bool passed;
    std::string message;
};

std::vector<TestResult> test_results;

void report_test(const std::string& name, bool passed, const std::string& msg = "") {
    test_results.push_back({name, passed, msg});
    if (passed) {
        std::cout << "[PASS] " << name << std::endl;
    } else {
        std::cout << "[FAIL] " << name;
        if (!msg.empty()) {
            std::cout << ": " << msg;
        }
        std::cout << std::endl;
    }
}

// Test device pointer allocation
void test_device_ptr_allocation() {
    try {
        DevicePtr<float> ptr(1024);
        report_test("DevicePtr allocation", ptr.get() != nullptr);
    } catch (const std::exception& e) {
        report_test("DevicePtr allocation", false, e.what());
    }
}

// Test device pointer copy
void test_device_ptr_copy() {
    try {
        const size_t count = 100;
        std::vector<float> host_data(count);
        for (size_t i = 0; i < count; i++) {
            host_data[i] = static_cast<float>(i);
        }
        
        DevicePtr<float> d_ptr(count);
        d_ptr.copy_from_host(host_data.data(), count);
        
        std::vector<float> result(count);
        d_ptr.copy_to_host(result.data(), count);
        
        bool all_match = true;
        for (size_t i = 0; i < count; i++) {
            if (result[i] != host_data[i]) {
                all_match = false;
                break;
            }
        }
        
        report_test("DevicePtr host-device copy", all_match);
    } catch (const std::exception& e) {
        report_test("DevicePtr host-device copy", false, e.what());
    }
}

// Test pinned memory allocation
void test_pinned_ptr_allocation() {
    try {
        PinnedPtr<float> ptr(1024);
        report_test("PinnedPtr allocation", ptr.get() != nullptr);
    } catch (const std::exception& e) {
        report_test("PinnedPtr allocation", false, e.what());
    }
}

// Test pinned memory access
void test_pinned_ptr_access() {
    try {
        const size_t count = 100;
        PinnedPtr<float> ptr(count);
        
        for (size_t i = 0; i < count; i++) {
            ptr[i] = static_cast<float>(i * 2);
        }
        
        bool all_correct = true;
        for (size_t i = 0; i < count; i++) {
            if (ptr[i] != static_cast<float>(i * 2)) {
                all_correct = false;
                break;
            }
        }
        
        report_test("PinnedPtr access", all_correct);
    } catch (const std::exception& e) {
        report_test("PinnedPtr access", false, e.what());
    }
}

// Test packed haplotypes
void test_packed_haplotypes() {
    try {
        const marker_t num_markers = 100;
        const haplotype_t num_haplotypes = 200;
        
        PackedHaplotypes packed(num_markers, num_haplotypes);
        
        // Set some values
        packed.set_allele(0, 0, 0);
        packed.set_allele(0, 1, 1);
        packed.set_allele(1, 0, 2);
        packed.set_allele(1, 1, 3);
        
        // Check values
        bool correct = (
            packed.get_allele(0, 0) == 0 &&
            packed.get_allele(0, 1) == 1 &&
            packed.get_allele(1, 0) == 2 &&
            packed.get_allele(1, 1) == 3
        );
        
        report_test("PackedHaplotypes set/get", correct);
    } catch (const std::exception& e) {
        report_test("PackedHaplotypes set/get", false, e.what());
    }
}

// Test packed haplotypes size
void test_packed_haplotypes_size() {
    try {
        const marker_t num_markers = 1000;
        const haplotype_t num_haplotypes = 2000;
        
        PackedHaplotypes packed(num_markers, num_haplotypes);
        
        // Total alleles = 1000 * 2000 = 2,000,000
        // 4 alleles per byte = 500,000 bytes
        size_t expected_size = (num_markers * num_haplotypes + 3) / 4;
        size_t actual_size = packed.data_size();
        
        report_test("PackedHaplotypes size", actual_size == expected_size);
    } catch (const std::exception& e) {
        report_test("PackedHaplotypes size", false, e.what());
    }
}

// Test device info retrieval
void test_device_info() {
    try {
        DeviceInfo info = get_device_info(0);
        
        bool valid = (
            !info.name.empty() &&
            info.total_memory > 0 &&
            info.compute_capability_major >= 8
        );
        
        if (valid) {
            std::cout << "  GPU: " << info.name << std::endl;
            std::cout << "  Compute: " << info.compute_capability_major << "." 
                      << info.compute_capability_minor << std::endl;
            std::cout << "  Memory: " << (info.total_memory / (1024*1024*1024)) << " GB" << std::endl;
        }
        
        report_test("Device info retrieval", valid);
    } catch (const std::exception& e) {
        report_test("Device info retrieval", false, e.what());
    }
}

// Test best device selection
void test_select_best_device() {
    try {
        int device_id = select_best_device();
        report_test("Select best device", device_id >= 0);
    } catch (const std::exception& e) {
        report_test("Select best device", false, e.what());
    }
}

// Test memory query
void test_memory_query() {
    try {
        size_t available = get_available_gpu_memory(0);
        size_t total = get_total_gpu_memory(0);
        
        bool valid = (available > 0 && total > 0 && available <= total);
        
        if (valid) {
            std::cout << "  Available: " << (available / (1024*1024)) << " MB" << std::endl;
            std::cout << "  Total: " << (total / (1024*1024)) << " MB" << std::endl;
        }
        
        report_test("Memory query", valid);
    } catch (const std::exception& e) {
        report_test("Memory query", false, e.what());
    }
}

// Test temporary buffer
void test_temp_buffer() {
    try {
        {
            TempDeviceBuffer<float> buffer(1000);
            bool valid = (buffer.get() != nullptr && buffer.size() == 1000);
            report_test("TempDeviceBuffer", valid);
        }
        // Buffer should be freed automatically
    } catch (const std::exception& e) {
        report_test("TempDeviceBuffer", false, e.what());
    }
}

int main() {
    std::cout << "SwiftImpute Memory Management Tests" << std::endl;
    std::cout << "====================================" << std::endl;
    std::cout << std::endl;
    
    // Run all tests
    test_device_ptr_allocation();
    test_device_ptr_copy();
    test_pinned_ptr_allocation();
    test_pinned_ptr_access();
    test_packed_haplotypes();
    test_packed_haplotypes_size();
    test_device_info();
    test_select_best_device();
    test_memory_query();
    test_temp_buffer();
    
    // Summary
    std::cout << std::endl;
    std::cout << "Test Summary" << std::endl;
    std::cout << "============" << std::endl;
    
    int passed = 0;
    int failed = 0;
    for (const auto& result : test_results) {
        if (result.passed) {
            passed++;
        } else {
            failed++;
        }
    }
    
    std::cout << "Passed: " << passed << "/" << test_results.size() << std::endl;
    std::cout << "Failed: " << failed << "/" << test_results.size() << std::endl;
    
    return (failed == 0) ? 0 : 1;
}
