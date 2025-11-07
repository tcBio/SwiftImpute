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
set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} /O2 /DNDEBUG")

# Debug flags
set(CMAKE_CUDA_FLAGS_DEBUG "${CMAKE_CUDA_FLAGS_DEBUG} -G -g")
set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} /Zi /Od")

# Find CUDA
find_package(CUDAToolkit REQUIRED)

# Include directories
include_directories(
    ${CMAKE_SOURCE_DIR}/src
    ${CUDAToolkit_INCLUDE_DIRS}
)

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
)

# Kernel library
set(KERNEL_SOURCES
    src/kernels/forward_backward.cuh
    src/kernels/forward_backward.cpp
    src/kernels/logsumexp.cuh
    src/kernels/logsumexp.cu
    src/kernels/emission.hpp
    src/kernels/emission.cu
    src/kernels/transition.hpp
    src/kernels/transition.cu
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

    # TODO: Add when implemented
    # add_executable(test_pbwt test/test_pbwt.cpp)
    # target_link_libraries(test_pbwt swiftimpute_lib)
    # add_test(NAME pbwt_test COMMAND test_pbwt)

    # add_executable(test_kernels test/test_kernels.cu)
    # target_link_libraries(test_kernels swiftimpute_lib)
    # add_test(NAME kernel_test COMMAND test_kernels)
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
message(STATUS "")
