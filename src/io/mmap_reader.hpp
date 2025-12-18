#pragma once

#include "../core/types.hpp"
#include <string>
#include <memory>
#include <cstddef>

#ifdef _WIN32
#include <windows.h>
#else
#include <sys/mman.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>
#endif

namespace swiftimpute {

/**
 * @brief Configuration for memory-mapped file I/O
 */
struct MMapConfig {
    bool use_huge_pages;        // Use huge pages if available (2MB pages)
    bool sequential_access;     // Hint for sequential access pattern
    bool populate_on_map;       // Pre-fault pages on mapping (faster first access)
    bool lock_pages;            // Lock pages in memory (prevent swapping)
    size_t read_ahead_bytes;    // Kernel read-ahead hint

    MMapConfig()
        : use_huge_pages(false)
        , sequential_access(true)
        , populate_on_map(false)
        , lock_pages(false)
        , read_ahead_bytes(4 * 1024 * 1024) {}  // 4 MB default

    // Preset for NVMe SSDs
    static MMapConfig nvme_optimized() {
        MMapConfig config;
        config.sequential_access = true;
        config.read_ahead_bytes = 8 * 1024 * 1024;  // 8 MB read-ahead for NVMe
        return config;
    }

    // Preset for spinning disks
    static MMapConfig hdd_optimized() {
        MMapConfig config;
        config.sequential_access = true;
        config.read_ahead_bytes = 2 * 1024 * 1024;
        return config;
    }

    // Preset for memory-constrained systems
    static MMapConfig low_memory() {
        MMapConfig config;
        config.sequential_access = true;
        config.populate_on_map = false;
        config.lock_pages = false;
        config.read_ahead_bytes = 1024 * 1024;
        return config;
    }
};

/**
 * @brief Statistics about memory-mapped file operations
 */
struct MMapStats {
    size_t file_size;
    size_t mapped_size;
    size_t bytes_read;
    size_t page_faults;           // Minor page faults
    double map_time_seconds;
    double total_read_seconds;

    MMapStats()
        : file_size(0)
        , mapped_size(0)
        , bytes_read(0)
        , page_faults(0)
        , map_time_seconds(0)
        , total_read_seconds(0) {}
};

/**
 * @brief Memory-mapped file reader for high-performance I/O
 *
 * Provides zero-copy access to file contents via memory mapping,
 * optimized for NVMe SSDs and sequential access patterns.
 *
 * Benefits:
 * - Zero-copy data access (no user-space buffering)
 * - Kernel handles prefetching and caching
 * - Efficient for large sequential reads
 * - Reduced system call overhead
 *
 * Usage:
 *   MemoryMappedFile file("reference.vcf", MMapConfig::nvme_optimized());
 *   const char* data = file.data();
 *   // Access data directly...
 */
class MemoryMappedFile {
public:
    MemoryMappedFile() = default;

    /**
     * @brief Open and map a file
     *
     * @param filename Path to file
     * @param config Memory mapping configuration
     */
    explicit MemoryMappedFile(
        const std::string& filename,
        const MMapConfig& config = MMapConfig()
    );

    ~MemoryMappedFile();

    // Non-copyable
    MemoryMappedFile(const MemoryMappedFile&) = delete;
    MemoryMappedFile& operator=(const MemoryMappedFile&) = delete;

    // Movable
    MemoryMappedFile(MemoryMappedFile&& other) noexcept;
    MemoryMappedFile& operator=(MemoryMappedFile&& other) noexcept;

    /**
     * @brief Open and map a file
     */
    void open(const std::string& filename, const MMapConfig& config = MMapConfig());

    /**
     * @brief Unmap and close the file
     */
    void close();

    /**
     * @brief Check if file is mapped
     */
    bool is_open() const { return data_ != nullptr; }

    /**
     * @brief Get pointer to mapped data
     */
    const char* data() const { return data_; }

    /**
     * @brief Get file size
     */
    size_t size() const { return size_; }

    /**
     * @brief Get filename
     */
    const std::string& filename() const { return filename_; }

    /**
     * @brief Advise kernel about access pattern for a region
     *
     * @param offset Start offset in file
     * @param length Length of region
     * @param sequential True for sequential, false for random access
     */
    void advise_region(size_t offset, size_t length, bool sequential);

    /**
     * @brief Pre-fetch a region into memory
     *
     * Triggers read-ahead for the specified region.
     *
     * @param offset Start offset in file
     * @param length Length of region to prefetch
     */
    void prefetch_region(size_t offset, size_t length);

    /**
     * @brief Get mapping statistics
     */
    MMapStats get_stats() const;

    /**
     * @brief Check if memory mapping is supported on this platform
     */
    static bool is_supported();

    /**
     * @brief Get the system page size
     */
    static size_t page_size();

private:
    std::string filename_;
    char* data_ = nullptr;
    size_t size_ = 0;
    MMapConfig config_;
    MMapStats stats_;

#ifdef _WIN32
    HANDLE file_handle_ = INVALID_HANDLE_VALUE;
    HANDLE mapping_handle_ = NULL;
#else
    int fd_ = -1;
#endif

    void apply_access_hints();
};

/**
 * @brief Sliding window view over a memory-mapped file
 *
 * Useful for processing large files in chunks without mapping
 * the entire file at once.
 */
class MMapWindow {
public:
    /**
     * @brief Create a sliding window view
     *
     * @param filename File to map
     * @param window_size Size of window in bytes
     * @param config Memory mapping configuration
     */
    MMapWindow(
        const std::string& filename,
        size_t window_size = 64 * 1024 * 1024,  // 64 MB default
        const MMapConfig& config = MMapConfig()
    );

    ~MMapWindow();

    /**
     * @brief Slide the window to a new position
     *
     * @param offset New start offset
     * @return true if successful
     */
    bool slide_to(size_t offset);

    /**
     * @brief Get current window data
     */
    const char* data() const { return current_data_; }

    /**
     * @brief Get current window size
     */
    size_t window_size() const { return current_size_; }

    /**
     * @brief Get current window start offset
     */
    size_t current_offset() const { return current_offset_; }

    /**
     * @brief Get total file size
     */
    size_t file_size() const { return file_size_; }

    /**
     * @brief Check if at end of file
     */
    bool at_end() const { return current_offset_ >= file_size_; }

private:
    std::string filename_;
    size_t window_size_;
    MMapConfig config_;
    size_t file_size_;
    size_t current_offset_;
    size_t current_size_;
    char* current_data_;

#ifdef _WIN32
    HANDLE file_handle_ = INVALID_HANDLE_VALUE;
    HANDLE mapping_handle_ = NULL;
#else
    int fd_ = -1;
#endif

    void map_window();
    void unmap_window();
};

/**
 * @brief Async prefetcher for predictive read-ahead
 *
 * Prefetches the next chunk while current chunk is being processed.
 */
class AsyncPrefetcher {
public:
    AsyncPrefetcher(MemoryMappedFile& file);

    /**
     * @brief Start prefetching a region asynchronously
     *
     * @param offset Start offset
     * @param length Length to prefetch
     */
    void start_prefetch(size_t offset, size_t length);

    /**
     * @brief Wait for prefetch to complete
     */
    void wait();

    /**
     * @brief Check if prefetch is complete
     */
    bool is_complete() const;

private:
    MemoryMappedFile& file_;
    size_t prefetch_offset_;
    size_t prefetch_length_;
    bool prefetch_active_;
};

/**
 * @brief Factory: detect storage type and create optimal config
 *
 * @param filename Path to file (used to detect storage type)
 * @return Optimized MMapConfig
 */
MMapConfig detect_optimal_config(const std::string& filename);

/**
 * @brief Check if a path is on an NVMe device
 *
 * @param path File or directory path
 * @return true if on NVMe storage
 */
bool is_nvme_storage(const std::string& path);

} // namespace swiftimpute
