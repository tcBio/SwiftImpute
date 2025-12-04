cmake_minimum_required(VERSION 3.20)
project(SwiftImpute LANGUAGES CXX CUDA)

# C++ Standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# CUDA Standard
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# Build type default
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

# Options
option(USE_FAST_MATH "Enable fast math optimizations" OFF)
option(BUILD_TESTS "Build test suite" ON)
option(BUILD_BENCHMARKS "Build benchmarks" ON)
option(ENABLE_HTSLIB "Enable htslib for compressed VCF/BCF support" ON)

# CUDA architectures (user can override with -DCUDA_ARCHITECTURES)
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES "80;89")
endif()

message(STATUS "Building for CUDA architectures: ${CMAKE_CUDA_ARCHITECTURES}")

# CUDA flags
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --extended-lambda --expt-relaxed-constexpr")

if(USE_FAST_MATH)
    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --use_fast_math")
    message(STATUS "Fast math enabled")
endif()

# Release optimizations
set(CMAKE_CUDA_FLAGS_RELEASE "${CMAKE_CUDA_FLAGS_RELEASE} -O3 -DNDEBUG")
if(MSVC)
    set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} /O2 /DNDEBUG")
else()
    set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -O3 -DNDEBUG")
endif()

# Debug flags
set(CMAKE_CUDA_FLAGS_DEBUG "${CMAKE_CUDA_FLAGS_DEBUG} -G -g")
if(MSVC)
    set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} /Zi /Od")
else()
    set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -g -O0")
endif()

# Find CUDA
find_package(CUDAToolkit REQUIRED)

# Find htslib (for compressed VCF/BCF support)
set(HTSLIB_FOUND FALSE)
if(ENABLE_HTSLIB)
    # Try pkg-config first
    find_package(PkgConfig QUIET)
    if(PkgConfig_FOUND)
        pkg_check_modules(HTSLIB QUIET htslib)
    endif()

    if(HTSLIB_FOUND)
        message(STATUS "Found htslib via pkg-config: ${HTSLIB_LIBRARIES}")
    else()
        # Try find_library as fallback
        find_library(HTSLIB_LIBRARY NAMES hts htslib)
        find_path(HTSLIB_INCLUDE_DIR NAMES htslib/vcf.h htslib/hts.h)

        if(HTSLIB_LIBRARY AND HTSLIB_INCLUDE_DIR)
            set(HTSLIB_FOUND TRUE)
            set(HTSLIB_LIBRARIES ${HTSLIB_LIBRARY})
            set(HTSLIB_INCLUDE_DIRS ${HTSLIB_INCLUDE_DIR})
            message(STATUS "Found htslib via find_library: ${HTSLIB_LIBRARIES}")
        else()
            message(WARNING "htslib not found. Compressed VCF/BCF support will be disabled. "
                           "Install htslib-dev or build with -DENABLE_HTSLIB=OFF")
        endif()
    endif()

    if(HTSLIB_FOUND)
        # Find zlib (required by htslib)
        find_package(ZLIB QUIET)
        if(NOT ZLIB_FOUND)
            message(WARNING "zlib not found. Required by htslib.")
        endif()
    endif()
endif()

# Include directories
include_directories(
    ${CMAKE_SOURCE_DIR}/src
    ${CUDAToolkit_INCLUDE_DIRS}
)

# Add htslib include directories if found
if(HTSLIB_FOUND)
    include_directories(${HTSLIB_INCLUDE_DIRS})
    add_compile_definitions(HAVE_HTSLIB)
endif()

# Core library
set(CORE_SOURCES
    src/core/types.hpp
    src/core/types.cpp
    src/core/memory_pool.hpp
    src/core/memory_pool.cu
)

# PBWT library
set(PBWT_SOURCES
    src/pbwt/pbwt_index.hpp
    src/pbwt/pbwt_index.cpp
    src/pbwt/pbwt_selector.cu
)

# Kernel library
set(KERNEL_SOURCES
    src/kernels/forward_backward.cuh
    src/kernels/forward_backward.cpp
    src/kernels/forward_backward.cu
    src/kernels/logsumexp.cuh
    src/kernels/logsumexp.cu
    src/kernels/emission.hpp
    src/kernels/emission.cu
    src/kernels/transition.hpp
    src/kernels/transition.cu
    src/kernels/sampling.hpp
    src/kernels/sampling.cu
)

# I/O library
set(IO_SOURCES
    src/io/vcf_reader.hpp
    src/io/vcf_writer.hpp
)

# API library
set(API_SOURCES
    src/api/imputer.hpp
    src/api/imputer.cpp
)

# Create static library
add_library(swiftimpute_lib STATIC
    ${CORE_SOURCES}
    ${PBWT_SOURCES}
    ${KERNEL_SOURCES}
    ${IO_SOURCES}
    ${API_SOURCES}
)

# Link CUDA libraries
target_link_libraries(swiftimpute_lib
    CUDA::cudart
    CUDA::curand
)

# Link htslib if found
if(HTSLIB_FOUND)
    target_link_libraries(swiftimpute_lib ${HTSLIB_LIBRARIES})
    if(ZLIB_FOUND)
        target_link_libraries(swiftimpute_lib ZLIB::ZLIB)
    endif()
endif()

# Set library properties
set_target_properties(swiftimpute_lib PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_RESOLVE_DEVICE_SYMBOLS ON
    POSITION_INDEPENDENT_CODE ON
)

# Main executable
add_executable(swiftimpute
    src/main.cpp
)

target_link_libraries(swiftimpute
    swiftimpute_lib
)

# Tests (existing tests only)
if(BUILD_TESTS)
    enable_testing()

    # Memory management test (exists)
    add_executable(test_memory test/test_memory.cu)
    target_link_libraries(test_memory swiftimpute_lib)
    add_test(NAME memory_test COMMAND test_memory)

    # VCF I/O and pipeline test
    add_executable(test_vcf_io test/test_vcf_io.cpp)
    target_link_libraries(test_vcf_io swiftimpute_lib)
    add_test(NAME vcf_io_test COMMAND test_vcf_io)

    # GPU kernel tests
    add_executable(test_kernels test/test_kernels.cu)
    target_link_libraries(test_kernels swiftimpute_lib)
    add_test(NAME kernel_test COMMAND test_kernels)

    # TODO: Add when implemented
    # add_executable(test_pbwt test/test_pbwt.cpp)
    # target_link_libraries(test_pbwt swiftimpute_lib)
    # add_test(NAME pbwt_test COMMAND test_pbwt)
endif()

# Benchmarks (to be implemented)
if(BUILD_BENCHMARKS)
    # TODO: Add when implemented
    # add_executable(benchmark_forward benchmarks/benchmark_forward.cu)
    # target_link_libraries(benchmark_forward swiftimpute_lib)

    # add_executable(benchmark_pbwt benchmarks/benchmark_pbwt.cpp)
    # target_link_libraries(benchmark_pbwt swiftimpute_lib)
endif()

# Installation
install(TARGETS swiftimpute
    RUNTIME DESTINATION bin
)

install(TARGETS swiftimpute_lib
    ARCHIVE DESTINATION lib
)

install(DIRECTORY src/
    DESTINATION include/swiftimpute
    FILES_MATCHING PATTERN "*.hpp" PATTERN "*.cuh"
)

# Print configuration summary
message(STATUS "")
message(STATUS "SwiftImpute Configuration:")
message(STATUS "  Build type: ${CMAKE_BUILD_TYPE}")
message(STATUS "  C++ compiler: ${CMAKE_CXX_COMPILER}")
message(STATUS "  CUDA compiler: ${CMAKE_CUDA_COMPILER}")
message(STATUS "  CUDA architectures: ${CMAKE_CUDA_ARCHITECTURES}")
message(STATUS "  Fast math: ${USE_FAST_MATH}")
message(STATUS "  Build tests: ${BUILD_TESTS}")
message(STATUS "  Build benchmarks: ${BUILD_BENCHMARKS}")
if(ENABLE_HTSLIB)
    if(HTSLIB_FOUND)
        message(STATUS "  htslib support: ENABLED (${HTSLIB_LIBRARIES})")
    else()
        message(STATUS "  htslib support: NOT FOUND (compressed VCF disabled)")
    endif()
else()
    message(STATUS "  htslib support: DISABLED")
endif()
message(STATUS "")
