# SPDX-FileCopyrightText: 2026 HenryRenYz and vecops contributors
# SPDX-License-Identifier: MIT

find_package(Threads REQUIRED)
add_executable(vecops-memory-bench
    MemoryBench.cpp Support.cpp Runner.cpp
    ScalarKernels.cpp VectorKernels.cpp LatencyKernels.cpp Sdma.cpp)
set(VECOPS_MEMORY_SDMA "AUTO" CACHE STRING "SDMA SDK backend: AUTO, ON, OFF")
set_property(CACHE VECOPS_MEMORY_SDMA PROPERTY STRINGS AUTO ON OFF)
if(NOT VECOPS_MEMORY_SDMA MATCHES "^(AUTO|ON|OFF)$")
    message(FATAL_ERROR "VECOPS_MEMORY_SDMA requires AUTO, ON, or OFF")
endif()
if(NOT VECOPS_MEMORY_SDMA STREQUAL "OFF")
    find_path(VECOPS_SDMA_INCLUDE_DIR mdk_sdma.h)
    if(VECOPS_SDMA_INCLUDE_DIR)
        target_include_directories(vecops-memory-bench SYSTEM PRIVATE "${VECOPS_SDMA_INCLUDE_DIR}")
        target_compile_definitions(vecops-memory-bench PRIVATE VECOPS_MEMORY_HAVE_SDMA=1)
    elseif(VECOPS_MEMORY_SDMA STREQUAL "ON")
        message(FATAL_ERROR "SDMA requires the installed sdma-dk mdk_sdma.h header")
    endif()
endif()
target_link_libraries(vecops-memory-bench PRIVATE ${CMAKE_DL_LIBS})
target_compile_features(vecops-memory-bench PRIVATE cxx_std_20)
target_include_directories(vecops-memory-bench SYSTEM PRIVATE
    "${VECOPS_SOURCE_DIR}/include" "${VECOPS_SOURCE_DIR}/include/vecops")
target_link_libraries(vecops-memory-bench PRIVATE Threads::Threads)
vecops_configure_target_arch(TARGET vecops-memory-bench ARCH Native)
set_target_properties(vecops-memory-bench PROPERTIES INTERPROCEDURAL_OPTIMIZATION FALSE)
target_compile_options(vecops-memory-bench PRIVATE
    -O3 -Wall -Wextra -Wno-ignored-attributes -fno-lto -ffp-contract=off)
execute_process(COMMAND git describe --always --dirty
    WORKING_DIRECTORY "${VECOPS_SOURCE_DIR}"
    OUTPUT_VARIABLE _MEMORY_BENCH_REVISION OUTPUT_STRIP_TRAILING_WHITESPACE
    ERROR_QUIET)
target_compile_definitions(vecops-memory-bench PRIVATE
    VECOPS_MEMORY_BENCH_REVISION="${_MEMORY_BENCH_REVISION}")
# Explicit Vec operations still vectorize; automatic loop/SLP transforms must not
# change the requested scalar instructions or the number of Vecs per iteration.
if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
    target_compile_options(vecops-memory-bench PRIVATE
        -fno-tree-vectorize -fno-tree-slp-vectorize -fno-unroll-loops
        -fno-peel-loops -fno-tree-loop-distribute-patterns)
elseif(CMAKE_CXX_COMPILER_ID MATCHES "Clang")
    target_compile_options(vecops-memory-bench PRIVATE
        -fno-vectorize -fno-slp-vectorize -fno-unroll-loops)
else()
    message(FATAL_ERROR "vecops-memory-bench requires GCC or Clang")
endif()
set(_MEMORY_BENCH_SOURCE_HASHES "")
foreach(_SOURCE MemoryBench.cpp Bench.h Support.cpp Runner.cpp Kernels.h
                ScalarKernels.cpp VectorKernels.cpp LatencyKernels.cpp Sdma.cpp CMakeLists.txt)
    file(SHA256 "${CMAKE_CURRENT_SOURCE_DIR}/${_SOURCE}" _SOURCE_HASH)
    string(APPEND _MEMORY_BENCH_SOURCE_HASHES "${_SOURCE}:${_SOURCE_HASH};")
    set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS "${_SOURCE}")
endforeach()
string(SHA256 _MEMORY_BENCH_SOURCE_SHA "${_MEMORY_BENCH_SOURCE_HASHES}")
get_target_property(_MEMORY_BENCH_FLAGS vecops-memory-bench COMPILE_OPTIONS)
string(JOIN " " _MEMORY_BENCH_FLAGS ${_MEMORY_BENCH_FLAGS})
target_compile_definitions(vecops-memory-bench PRIVATE
    VECOPS_MEMORY_BENCH_SOURCE_SHA="${_MEMORY_BENCH_SOURCE_SHA}"
    VECOPS_MEMORY_BENCH_FLAGS="${_MEMORY_BENCH_FLAGS}")
install(TARGETS vecops-memory-bench RUNTIME DESTINATION "${CMAKE_INSTALL_BINDIR}")
