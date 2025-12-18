#include "mmap_reader.hpp"
#include <chrono>
#include <cstring>
#include <stdexcept>

#ifdef _WIN32
// Windows implementation
#else
#include <sys/mman.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>
#include <sys/utsname.h>
#endif

namespace swiftimpute {

// ============================================================================
// Platform utilities
// ============================================================================

bool MemoryMappedFile::is_supported() {
#ifdef _WIN32
    return true;  // Windows always supports memory mapping
#else
    return true;  // POSIX systems support mmap
#endif
}

size_t MemoryMappedFile::page_size() {
#ifdef _WIN32
    SYSTEM_INFO si;
    GetSystemInfo(&si);
    return si.dwPageSize;
#else
    return static_cast<size_t>(sysconf(_SC_PAGESIZE));
#endif
}

// ============================================================================
// MemoryMappedFile Implementation
// ============================================================================

MemoryMappedFile::MemoryMappedFile(
    const std::string& filename,
    const MMapConfig& config
) {
    open(filename, config);
}

MemoryMappedFile::~MemoryMappedFile() {
    close();
}

MemoryMappedFile::MemoryMappedFile(MemoryMappedFile&& other) noexcept
    : filename_(std::move(other.filename_))
    , data_(other.data_)
    , size_(other.size_)
    , config_(other.config_)
    , stats_(other.stats_)
#ifdef _WIN32
    , file_handle_(other.file_handle_)
    , mapping_handle_(other.mapping_handle_)
#else
    , fd_(other.fd_)
#endif
{
    other.data_ = nullptr;
    other.size_ = 0;
#ifdef _WIN32
    other.file_handle_ = INVALID_HANDLE_VALUE;
    other.mapping_handle_ = NULL;
#else
    other.fd_ = -1;
#endif
}

MemoryMappedFile& MemoryMappedFile::operator=(MemoryMappedFile&& other) noexcept {
    if (this != &other) {
        close();

        filename_ = std::move(other.filename_);
        data_ = other.data_;
        size_ = other.size_;
        config_ = other.config_;
        stats_ = other.stats_;

#ifdef _WIN32
        file_handle_ = other.file_handle_;
        mapping_handle_ = other.mapping_handle_;
        other.file_handle_ = INVALID_HANDLE_VALUE;
        other.mapping_handle_ = NULL;
#else
        fd_ = other.fd_;
        other.fd_ = -1;
#endif

        other.data_ = nullptr;
        other.size_ = 0;
    }
    return *this;
}

void MemoryMappedFile::open(const std::string& filename, const MMapConfig& config) {
    close();  // Close any existing mapping

    auto start_time = std::chrono::high_resolution_clock::now();

    filename_ = filename;
    config_ = config;

#ifdef _WIN32
    // Windows implementation
    file_handle_ = CreateFileA(
        filename.c_str(),
        GENERIC_READ,
        FILE_SHARE_READ,
        NULL,
        OPEN_EXISTING,
        FILE_ATTRIBUTE_NORMAL | (config.sequential_access ? FILE_FLAG_SEQUENTIAL_SCAN : 0),
        NULL
    );

    if (file_handle_ == INVALID_HANDLE_VALUE) {
        throw std::runtime_error("Failed to open file: " + filename);
    }

    // Get file size
    LARGE_INTEGER li;
    if (!GetFileSizeEx(file_handle_, &li)) {
        CloseHandle(file_handle_);
        file_handle_ = INVALID_HANDLE_VALUE;
        throw std::runtime_error("Failed to get file size: " + filename);
    }
    size_ = static_cast<size_t>(li.QuadPart);

    // Create file mapping
    mapping_handle_ = CreateFileMappingA(
        file_handle_,
        NULL,
        PAGE_READONLY,
        0,
        0,
        NULL
    );

    if (mapping_handle_ == NULL) {
        CloseHandle(file_handle_);
        file_handle_ = INVALID_HANDLE_VALUE;
        throw std::runtime_error("Failed to create file mapping: " + filename);
    }

    // Map view
    data_ = static_cast<char*>(MapViewOfFile(
        mapping_handle_,
        FILE_MAP_READ,
        0,
        0,
        0
    ));

    if (data_ == nullptr) {
        CloseHandle(mapping_handle_);
        CloseHandle(file_handle_);
        mapping_handle_ = NULL;
        file_handle_ = INVALID_HANDLE_VALUE;
        throw std::runtime_error("Failed to map file: " + filename);
    }

#else
    // POSIX implementation
    fd_ = ::open(filename.c_str(), O_RDONLY);
    if (fd_ < 0) {
        throw std::runtime_error("Failed to open file: " + filename + " - " + strerror(errno));
    }

    // Get file size
    struct stat sb;
    if (fstat(fd_, &sb) < 0) {
        ::close(fd_);
        fd_ = -1;
        throw std::runtime_error("Failed to get file size: " + filename);
    }
    size_ = static_cast<size_t>(sb.st_size);

    if (size_ == 0) {
        ::close(fd_);
        fd_ = -1;
        throw std::runtime_error("File is empty: " + filename);
    }

    // Build mmap flags
    int flags = MAP_PRIVATE;
#ifdef MAP_POPULATE
    if (config.populate_on_map) {
        flags |= MAP_POPULATE;
    }
#endif
#ifdef MAP_HUGETLB
    if (config.use_huge_pages) {
        flags |= MAP_HUGETLB;
    }
#endif

    // Memory map the file
    void* addr = mmap(nullptr, size_, PROT_READ, flags, fd_, 0);
    if (addr == MAP_FAILED) {
        ::close(fd_);
        fd_ = -1;
        throw std::runtime_error("Failed to mmap file: " + filename + " - " + strerror(errno));
    }

    data_ = static_cast<char*>(addr);

    // Apply access hints
    apply_access_hints();

    // Lock pages if requested
    if (config.lock_pages) {
        if (mlock(data_, size_) < 0) {
            LOG_WARNING("Failed to lock pages in memory: " + std::string(strerror(errno)));
        }
    }
#endif

    auto end_time = std::chrono::high_resolution_clock::now();
    stats_.file_size = size_;
    stats_.mapped_size = size_;
    stats_.map_time_seconds = std::chrono::duration<double>(end_time - start_time).count();

    LOG_INFO("Memory-mapped file: " + filename + " (" +
             std::to_string(size_ / (1024 * 1024)) + " MB)");
}

void MemoryMappedFile::close() {
    if (data_ == nullptr) return;

#ifdef _WIN32
    UnmapViewOfFile(data_);
    if (mapping_handle_ != NULL) {
        CloseHandle(mapping_handle_);
        mapping_handle_ = NULL;
    }
    if (file_handle_ != INVALID_HANDLE_VALUE) {
        CloseHandle(file_handle_);
        file_handle_ = INVALID_HANDLE_VALUE;
    }
#else
    if (config_.lock_pages) {
        munlock(data_, size_);
    }
    munmap(data_, size_);
    if (fd_ >= 0) {
        ::close(fd_);
        fd_ = -1;
    }
#endif

    data_ = nullptr;
    size_ = 0;
}

void MemoryMappedFile::apply_access_hints() {
#ifndef _WIN32
    if (data_ == nullptr) return;

    int advice = MADV_NORMAL;
    if (config_.sequential_access) {
        advice = MADV_SEQUENTIAL;
    }

    if (madvise(data_, size_, advice) < 0) {
        LOG_WARNING("madvise failed: " + std::string(strerror(errno)));
    }

#ifdef MADV_WILLNEED
    // Hint that we will need the data soon
    if (config_.populate_on_map) {
        madvise(data_, std::min(size_, config_.read_ahead_bytes), MADV_WILLNEED);
    }
#endif
#endif
}

void MemoryMappedFile::advise_region(size_t offset, size_t length, bool sequential) {
#ifndef _WIN32
    if (data_ == nullptr || offset >= size_) return;

    // Align to page boundary
    size_t page = page_size();
    size_t aligned_offset = (offset / page) * page;
    size_t adjusted_length = std::min(length + (offset - aligned_offset), size_ - aligned_offset);

    int advice = sequential ? MADV_SEQUENTIAL : MADV_RANDOM;
    madvise(data_ + aligned_offset, adjusted_length, advice);
#endif
}

void MemoryMappedFile::prefetch_region(size_t offset, size_t length) {
#ifndef _WIN32
    if (data_ == nullptr || offset >= size_) return;

    size_t actual_length = std::min(length, size_ - offset);

#ifdef MADV_WILLNEED
    // Align to page boundary
    size_t page = page_size();
    size_t aligned_offset = (offset / page) * page;
    size_t adjusted_length = actual_length + (offset - aligned_offset);

    madvise(data_ + aligned_offset, adjusted_length, MADV_WILLNEED);
#else
    // Fallback: touch pages to trigger read-ahead
    volatile char dummy = 0;
    size_t page = page_size();
    for (size_t i = offset; i < offset + actual_length; i += page) {
        dummy += data_[i];
    }
    (void)dummy;
#endif

    stats_.bytes_read += actual_length;
#endif
}

MMapStats MemoryMappedFile::get_stats() const {
    return stats_;
}

// ============================================================================
// MMapWindow Implementation
// ============================================================================

MMapWindow::MMapWindow(
    const std::string& filename,
    size_t window_size,
    const MMapConfig& config
)
    : filename_(filename)
    , window_size_(window_size)
    , config_(config)
    , current_offset_(0)
    , current_size_(0)
    , current_data_(nullptr)
{
#ifdef _WIN32
    file_handle_ = CreateFileA(
        filename.c_str(),
        GENERIC_READ,
        FILE_SHARE_READ,
        NULL,
        OPEN_EXISTING,
        FILE_ATTRIBUTE_NORMAL,
        NULL
    );

    if (file_handle_ == INVALID_HANDLE_VALUE) {
        throw std::runtime_error("Failed to open file: " + filename);
    }

    LARGE_INTEGER li;
    GetFileSizeEx(file_handle_, &li);
    file_size_ = static_cast<size_t>(li.QuadPart);

    mapping_handle_ = CreateFileMappingA(file_handle_, NULL, PAGE_READONLY, 0, 0, NULL);
#else
    fd_ = ::open(filename.c_str(), O_RDONLY);
    if (fd_ < 0) {
        throw std::runtime_error("Failed to open file: " + filename);
    }

    struct stat sb;
    fstat(fd_, &sb);
    file_size_ = static_cast<size_t>(sb.st_size);
#endif

    // Map initial window
    map_window();
}

MMapWindow::~MMapWindow() {
    unmap_window();

#ifdef _WIN32
    if (mapping_handle_ != NULL) CloseHandle(mapping_handle_);
    if (file_handle_ != INVALID_HANDLE_VALUE) CloseHandle(file_handle_);
#else
    if (fd_ >= 0) ::close(fd_);
#endif
}

bool MMapWindow::slide_to(size_t offset) {
    if (offset >= file_size_) {
        return false;
    }

    unmap_window();
    current_offset_ = offset;
    map_window();

    return true;
}

void MMapWindow::map_window() {
    current_size_ = std::min(window_size_, file_size_ - current_offset_);

#ifdef _WIN32
    DWORD offset_high = static_cast<DWORD>(current_offset_ >> 32);
    DWORD offset_low = static_cast<DWORD>(current_offset_ & 0xFFFFFFFF);

    current_data_ = static_cast<char*>(MapViewOfFile(
        mapping_handle_,
        FILE_MAP_READ,
        offset_high,
        offset_low,
        current_size_
    ));
#else
    // Align offset to page boundary
    size_t page = MemoryMappedFile::page_size();
    size_t aligned_offset = (current_offset_ / page) * page;
    size_t adjust = current_offset_ - aligned_offset;
    size_t map_size = current_size_ + adjust;

    void* addr = mmap(nullptr, map_size, PROT_READ, MAP_PRIVATE, fd_, aligned_offset);
    if (addr == MAP_FAILED) {
        throw std::runtime_error("Failed to map window: " + std::string(strerror(errno)));
    }

    current_data_ = static_cast<char*>(addr) + adjust;

    if (config_.sequential_access) {
        madvise(addr, map_size, MADV_SEQUENTIAL);
    }
#endif
}

void MMapWindow::unmap_window() {
    if (current_data_ == nullptr) return;

#ifdef _WIN32
    UnmapViewOfFile(current_data_);
#else
    size_t page = MemoryMappedFile::page_size();
    size_t aligned_offset = (current_offset_ / page) * page;
    size_t adjust = current_offset_ - aligned_offset;
    void* aligned_addr = current_data_ - adjust;
    size_t map_size = current_size_ + adjust;

    munmap(aligned_addr, map_size);
#endif

    current_data_ = nullptr;
    current_size_ = 0;
}

// ============================================================================
// AsyncPrefetcher Implementation
// ============================================================================

AsyncPrefetcher::AsyncPrefetcher(MemoryMappedFile& file)
    : file_(file)
    , prefetch_offset_(0)
    , prefetch_length_(0)
    , prefetch_active_(false)
{
}

void AsyncPrefetcher::start_prefetch(size_t offset, size_t length) {
    prefetch_offset_ = offset;
    prefetch_length_ = length;
    prefetch_active_ = true;

    // Use async hint - kernel will prefetch in background
    file_.prefetch_region(offset, length);
}

void AsyncPrefetcher::wait() {
    // For madvise-based prefetch, nothing to wait for
    // Pages will be faulted in on access
    prefetch_active_ = false;
}

bool AsyncPrefetcher::is_complete() const {
    // madvise is asynchronous, we assume complete after call
    return !prefetch_active_;
}

// ============================================================================
// Utility Functions
// ============================================================================

bool is_nvme_storage(const std::string& path) {
#ifdef __linux__
    // On Linux, check if device is NVMe by looking at /sys/block
    struct stat sb;
    if (stat(path.c_str(), &sb) < 0) {
        return false;
    }

    // Get device major/minor
    dev_t dev = sb.st_dev;
    unsigned int major_num = major(dev);

    // NVMe devices typically have major number 259
    // This is a heuristic and may not work on all systems
    if (major_num == 259) {
        return true;
    }

    // Alternative: check /sys/block/*/device/transport
    // Would need to map device to block device name

    return false;
#else
    // On other platforms, assume NVMe for SSDs
    // Could use platform-specific APIs
    return false;
#endif
}

MMapConfig detect_optimal_config(const std::string& filename) {
    if (is_nvme_storage(filename)) {
        LOG_INFO("Detected NVMe storage for: " + filename);
        return MMapConfig::nvme_optimized();
    }

    // Check if on SSD by looking at rotational flag
#ifdef __linux__
    struct stat sb;
    if (stat(filename.c_str(), &sb) == 0) {
        // Try to read rotational flag from sysfs
        // For simplicity, use default config
    }
#endif

    return MMapConfig();  // Default config
}

} // namespace swiftimpute
