cmake_minimum_required(VERSION 3.22)
project(vecops)

set_property(GLOBAL PROPERTY USE_FOLDERS ON)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_VERBOSE_MAKEFILE ON)

# Address sanitizer

# ---------------------------------- Options -----------------------------------
set(VECOPS_GITHUB_ENDPOINT "https://github.com" CACHE STRING "Set endpoint address for GitHub repositories")
option(VECOPS_PRESERVE_SUBNORMALS
       "Preserve representable subnormal results in strict vector operations" OFF)
set(VECOPS_FIXED_SVE_BITS "" CACHE STRING
    "SVE vector width in bits for fixed-SVE targets; empty detects the host maximum")

include(cmake/VecopsMultiarchTargets.cmake)

# -------------------------------- Dependencies --------------------------------
include(FetchContent)

# ------------------------------- Source Config --------------------------------
add_subdirectory(src)

# -------------------------------- Test Config ---------------------------------
option(BUILD_TESTING "Build tests" ON)
if(BUILD_TESTING)
    enable_testing()
    FetchContent_Declare(
            googletest
            GIT_REPOSITORY "${VECOPS_GITHUB_ENDPOINT}/google/googletest.git"
            GIT_TAG        v1.17.0
    )
    FetchContent_MakeAvailable(googletest)
    include(GoogleTest)
    add_subdirectory(tests)
endif()

option(BUILD_BENCHMARKS "Build benchmarks" ON)
if(BUILD_BENCHMARKS)
    set(BENCHMARK_ENABLE_TESTING OFF CACHE BOOL "" FORCE)
    set(BENCHMARK_ENABLE_INSTALL OFF CACHE BOOL "" FORCE)
    set(BENCHMARK_ENABLE_GTEST_TESTS OFF CACHE BOOL "" FORCE)
    FetchContent_Declare(
            googlebenchmark
            GIT_REPOSITORY "${VECOPS_GITHUB_ENDPOINT}/google/benchmark.git"
            GIT_TAG        v1.8.3
    )
    FetchContent_MakeAvailable(googlebenchmark)
    add_subdirectory(benchmarks)
endif()
