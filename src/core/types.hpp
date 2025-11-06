#pragma once

#include <cstdint>
#include <vector>
#include <string>
#include <memory>
#include <stdexcept>

namespace swiftimpute {

// Fundamental types
using marker_t = uint32_t;      // Marker index (0 to M-1)
using sample_t = uint32_t;      // Sample index (0 to N-1)
using haplotype_t = uint32_t;   // Haplotype index (0 to 2N-1)
using allele_t = uint8_t;       // Allele value (0, 1, or missing)
using state_t = uint16_t;       // HMM state index
using prob_t = float;           // Probability value

// Constants
constexpr allele_t ALLELE_MISSING = 255;
constexpr marker_t INVALID_MARKER = 0xFFFFFFFF;
constexpr sample_t INVALID_SAMPLE = 0xFFFFFFFF;

// Marker information
struct Marker {
    std::string chrom;          // Chromosome name
    uint64_t pos;               // Physical position
    std::string id;             // Variant ID (rsID)
    std::string ref;            // Reference allele
    std::string alt;            // Alternate allele
    double cM;                  // Genetic position in centiMorgans
    
    Marker() : pos(0), cM(0.0) {}
};

// Sample information
struct Sample {
    std::string id;             // Sample identifier
    uint32_t index;             // Internal index
    
    Sample() : index(0) {}
    explicit Sample(std::string id_) : id(std::move(id_)), index(0) {}
};

// Genotype likelihoods (log10 scale)
struct GenotypeLikelihoods {
    prob_t ll_00;               // P(data | AA)
    prob_t ll_01;               // P(data | AB)
    prob_t ll_11;               // P(data | BB)
    
    GenotypeLikelihoods() : ll_00(0.0f), ll_01(0.0f), ll_11(0.0f) {}
};

// Phased haplotype result
struct PhasedHaplotype {
    allele_t* hap0;             // First haplotype (M markers)
    allele_t* hap1;             // Second haplotype (M markers)
    prob_t* probabilities;      // Posterior probabilities
    
    PhasedHaplotype() : hap0(nullptr), hap1(nullptr), probabilities(nullptr) {}
};

// Imputation quality metrics
struct QualityMetrics {
    prob_t info_score;          // INFO score (0-1)
    prob_t r2_score;            // R-squared score (0-1)
    prob_t concordance;         // Concordance rate (0-1)
    uint32_t num_called;        // Number of called genotypes
    uint32_t num_missing;       // Number of missing genotypes
    
    QualityMetrics() : 
        info_score(0.0f), r2_score(0.0f), concordance(0.0f),
        num_called(0), num_missing(0) {}
};

// HMM parameters
struct HMMParameters {
    prob_t theta;               // Mutation rate (default: 0.001)
    prob_t rho_rate;            // Recombination rate multiplier (default: 4.0)
    uint32_t ne;                // Effective population size (default: 10000)
    uint32_t num_states;        // Number of HMM states (L parameter, default: 8)
    bool use_pbwt_selection;    // Use PBWT-based state selection
    
    HMMParameters() :
        theta(0.001f),
        rho_rate(4.0f),
        ne(10000),
        num_states(8),
        use_pbwt_selection(true) {}
};

// Device memory pointer wrapper
template<typename T>
class DevicePtr {
public:
    DevicePtr() : ptr_(nullptr), size_(0), owns_(false) {}
    
    explicit DevicePtr(size_t size) : size_(size), owns_(true) {
        allocate();
    }
    
    ~DevicePtr() {
        if (owns_) {
            deallocate();
        }
    }
    
    // No copy
    DevicePtr(const DevicePtr&) = delete;
    DevicePtr& operator=(const DevicePtr&) = delete;
    
    // Move semantics
    DevicePtr(DevicePtr&& other) noexcept :
        ptr_(other.ptr_), size_(other.size_), owns_(other.owns_) {
        other.ptr_ = nullptr;
        other.owns_ = false;
    }
    
    DevicePtr& operator=(DevicePtr&& other) noexcept {
        if (this != &other) {
            if (owns_) deallocate();
            ptr_ = other.ptr_;
            size_ = other.size_;
            owns_ = other.owns_;
            other.ptr_ = nullptr;
            other.owns_ = false;
        }
        return *this;
    }
    
    T* get() { return ptr_; }
    const T* get() const { return ptr_; }
    size_t size() const { return size_; }
    
    void copy_from_host(const T* host_data, size_t count);
    void copy_to_host(T* host_data, size_t count) const;
    
private:
    T* ptr_;
    size_t size_;
    bool owns_;
    
    void allocate();
    void deallocate();
};

// Pinned host memory wrapper for faster transfers
template<typename T>
class PinnedPtr {
public:
    PinnedPtr() : ptr_(nullptr), size_(0) {}
    
    explicit PinnedPtr(size_t size) : size_(size) {
        allocate();
    }
    
    ~PinnedPtr() {
        deallocate();
    }
    
    // No copy
    PinnedPtr(const PinnedPtr&) = delete;
    PinnedPtr& operator=(const PinnedPtr&) = delete;
    
    // Move semantics
    PinnedPtr(PinnedPtr&& other) noexcept :
        ptr_(other.ptr_), size_(other.size_) {
        other.ptr_ = nullptr;
    }
    
    PinnedPtr& operator=(PinnedPtr&& other) noexcept {
        if (this != &other) {
            deallocate();
            ptr_ = other.ptr_;
            size_ = other.size_;
            other.ptr_ = nullptr;
        }
        return *this;
    }
    
    T* get() { return ptr_; }
    const T* get() const { return ptr_; }
    size_t size() const { return size_; }
    
    T& operator[](size_t i) { return ptr_[i]; }
    const T& operator[](size_t i) const { return ptr_[i]; }
    
private:
    T* ptr_;
    size_t size_;
    
    void allocate();
    void deallocate();
};

// Bit-packed haplotype storage (2 bits per allele)
class PackedHaplotypes {
public:
    PackedHaplotypes(marker_t num_markers, haplotype_t num_haplotypes);
    ~PackedHaplotypes();
    
    // Set/get allele value
    void set_allele(marker_t m, haplotype_t h, allele_t value);
    allele_t get_allele(marker_t m, haplotype_t h) const;
    
    // Get raw data pointer (for GPU transfer)
    const uint8_t* data() const { return data_.data(); }
    size_t data_size() const { return data_.size(); }
    
    marker_t num_markers() const { return num_markers_; }
    haplotype_t num_haplotypes() const { return num_haplotypes_; }
    
private:
    marker_t num_markers_;
    haplotype_t num_haplotypes_;
    std::vector<uint8_t> data_;     // 4 alleles per byte
    
    size_t get_offset(marker_t m, haplotype_t h) const;
};

// Error handling
class ImputationError : public std::runtime_error {
public:
    explicit ImputationError(const std::string& msg) : std::runtime_error(msg) {}
};

class CUDAError : public ImputationError {
public:
    CUDAError(const std::string& msg, int error_code) :
        ImputationError(msg + " (CUDA error: " + std::to_string(error_code) + ")"),
        error_code_(error_code) {}
    
    int error_code() const { return error_code_; }
    
private:
    int error_code_;
};

// CUDA error checking macro
#define CHECK_CUDA(call) \
    do { \
        cudaError_t error = call; \
        if (error != cudaSuccess) { \
            throw CUDAError( \
                std::string(__FILE__) + ":" + std::to_string(__LINE__) + \
                " in " + std::string(__func__), \
                static_cast<int>(error) \
            ); \
        } \
    } while(0)

// Thread-safe logger
class Logger {
public:
    enum Level {
        DEBUG = 0,
        INFO = 1,
        WARNING = 2,
        ERROR = 3
    };
    
    static Logger& instance();
    
    void set_level(Level level) { level_ = level; }
    Level get_level() const { return level_; }
    
    void debug(const std::string& msg);
    void info(const std::string& msg);
    void warning(const std::string& msg);
    void error(const std::string& msg);
    
private:
    Logger() : level_(INFO) {}
    Level level_;
    
    void log(Level level, const std::string& prefix, const std::string& msg);
};

// Convenience macros
#define LOG_DEBUG(msg) Logger::instance().debug(msg)
#define LOG_INFO(msg) Logger::instance().info(msg)
#define LOG_WARNING(msg) Logger::instance().warning(msg)
#define LOG_ERROR(msg) Logger::instance().error(msg)

// GPU device information
struct DeviceInfo {
    int device_id;
    std::string name;
    size_t total_memory;
    size_t free_memory;
    int compute_capability_major;
    int compute_capability_minor;
    int multiprocessor_count;
    int max_threads_per_block;
    int max_threads_per_multiprocessor;
    
    DeviceInfo() :
        device_id(-1),
        total_memory(0),
        free_memory(0),
        compute_capability_major(0),
        compute_capability_minor(0),
        multiprocessor_count(0),
        max_threads_per_block(0),
        max_threads_per_multiprocessor(0) {}
};

// Get GPU device information
DeviceInfo get_device_info(int device_id = 0);

// Select best available GPU
int select_best_device();

// Memory utilities
size_t get_available_gpu_memory(int device_id = 0);
size_t get_total_gpu_memory(int device_id = 0);

} // namespace swiftimpute
