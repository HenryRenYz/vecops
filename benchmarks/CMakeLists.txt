# ======================================================================
# Platform Detection & Architecture Metadata
# ======================================================================

if(CMAKE_SYSTEM_PROCESSOR MATCHES "^(x86_64|AMD64|amd64)$")
    set(VECOPS_ARCH_FAMILY "x86")
elseif(CMAKE_SYSTEM_PROCESSOR MATCHES "^(aarch64|arm64|ARM64)$")
    set(VECOPS_ARCH_FAMILY "ARM")
else()
    message(FATAL_ERROR "Unsupported architecture: ${CMAKE_SYSTEM_PROCESSOR}")
endif()

set(VECOPS_MAP_x86_Scalar    "x86-64")
set(VECOPS_MAP_x86_AVX       "corei7-avx")
set(VECOPS_MAP_x86_AVX2      "core-avx2")
set(VECOPS_MAP_x86_AVX512    "skylake-avx512")
set(VECOPS_MAP_x86_Native    "native")

set(VECOPS_MAP_ARM_Scalar    "armv8-a")
set(VECOPS_MAP_ARM_NEON      "armv8-a+simd")
set(VECOPS_MAP_ARM_SVE       "armv8-a+sve")
set(VECOPS_MAP_ARM_SVE2      "armv8-a+sve2")

# ARM Native: only enable extensions advertised by the running host.
set(VECOPS_MAP_ARM_Native "native")
if(VECOPS_ARCH_FAMILY STREQUAL "ARM" AND EXISTS "/proc/cpuinfo")
    file(READ "/proc/cpuinfo" _CPUINFO)
    string(REGEX MATCH "Features[ \t]*:.*" _FEAT_LINE "${_CPUINFO}")
    foreach(_FEAT IN ITEMS sve2p1 bf16 sme)
        string(FIND "${_FEAT_LINE}" " ${_FEAT} " _POS)
        if(NOT _POS EQUAL -1)
            string(APPEND VECOPS_MAP_ARM_Native "+${_FEAT}")
        endif()
    endforeach()
    unset(_CPUINFO)
    unset(_FEAT_LINE)
endif()

set(VECOPS_ARCH_LIST_x86 Scalar AVX AVX2 AVX512 Native)
set(VECOPS_ARCH_LIST_ARM Scalar SVE SVE2 Native)
set(VECOPS_ARCH_LIST ${VECOPS_ARCH_LIST_${VECOPS_ARCH_FAMILY}})

add_compile_definitions(VECOPS_MATH_ASSUME_VALID_INPUTS)

# ======================================================================
# add_multiarch_benchmark
# ======================================================================

function(add_multiarch_benchmark NAME)
    set(options "")
    set(oneValueArgs "")
    set(multiValueArgs FILES ARCH)
    cmake_parse_arguments(ARG "${options}" "${oneValueArgs}" "${multiValueArgs}" ${ARGN})

    if(NOT ARG_ARCH)
        set(ARG_ARCH ${VECOPS_ARCH_LIST_${VECOPS_ARCH_FAMILY}})
    endif()

    foreach(CODE ${ARG_ARCH})
        set(MARCH ${VECOPS_MAP_${VECOPS_ARCH_FAMILY}_${CODE}})
        if(NOT MARCH)
            message(STATUS "Skipping benchmark arch '${CODE}' - not defined for ${VECOPS_ARCH_FAMILY}")
            continue()
        endif()

        set(TARGET_NAME "${NAME}-${CODE}")
        add_executable(${TARGET_NAME} ${ARG_FILES})
        target_include_directories(${TARGET_NAME} PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
        target_compile_options(${TARGET_NAME} PRIVATE -march=${MARCH})
        target_compile_definitions(${TARGET_NAME} PRIVATE
            VECOPS_BENCH_ARCH_CODE="${CODE}"
            VECOPS_SOURCE_DIR="${CMAKE_SOURCE_DIR}"
        )
        if(CODE STREQUAL "Scalar")
            target_compile_definitions(${TARGET_NAME} PRIVATE
                CPU_CAPABILITY=GENERIC
                CPU_CAPABILITY_GENERIC=1
            )
        endif()
        target_link_libraries(${TARGET_NAME} PRIVATE vecops benchmark::benchmark)
    endforeach()
endfunction()

add_multiarch_benchmark(LayerNormBench
    FILES ${CMAKE_CURRENT_SOURCE_DIR}/ops/LayerNormBench.cpp
    ARCH Native Scalar
)

add_multiarch_benchmark(SoftmaxBench
    FILES ${CMAKE_CURRENT_SOURCE_DIR}/ops/SoftmaxBench.cpp
    ARCH Native Scalar
)

option(VECOPS_ENABLE_ONEDNN_BENCHMARKS
       "Build LayerNorm and Softmax benchmarks against oneDNN" OFF)
set(VECOPS_ONEDNN_SOURCE_DIR "" CACHE PATH
    "Existing oneDNN source tree; fetched from GitHub when empty")
set(VECOPS_ONEDNN_GIT_TAG "v3.13" CACHE STRING
    "oneDNN revision used by the benchmark targets")

if(VECOPS_ENABLE_ONEDNN_BENCHMARKS)
    if(VECOPS_ARCH_FAMILY STREQUAL "ARM")
        set(VECOPS_ONEDNN_BENCH_ARCH "oneDNN-SVE")
        set(VECOPS_ONEDNN_IMPL_TOKEN "sve")
    else()
        set(VECOPS_ONEDNN_BENCH_ARCH "oneDNN-Native")
        set(VECOPS_ONEDNN_IMPL_TOKEN "jit")
    endif()

    # Keep the baseline single-threaded and limit the embedded oneDNN build to
    # the primitives required by these benchmarks. ACL is deliberately
    # disabled so the implementation cannot silently fall back to that
    # backend; the benchmark also checks the selected implementation name.
    set(ONEDNN_BUILD_TESTS OFF CACHE BOOL "" FORCE)
    set(ONEDNN_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
    set(ONEDNN_BUILD_GRAPH OFF CACHE BOOL "" FORCE)
    set(ONEDNN_CPU_RUNTIME "SEQ" CACHE STRING "" FORCE)
    set(ONEDNN_GPU_RUNTIME "NONE" CACHE STRING "" FORCE)
    set(ONEDNN_LIBRARY_TYPE "STATIC" CACHE STRING "" FORCE)
    set(ONEDNN_ENABLE_WORKLOAD "INFERENCE" CACHE STRING "" FORCE)
    set(ONEDNN_ENABLE_PRIMITIVE
        "LAYER_NORMALIZATION;SOFTMAX;REORDER" CACHE STRING "" FORCE)
    set(ONEDNN_AARCH64_USE_ACL OFF CACHE BOOL "" FORCE)

    if(VECOPS_ONEDNN_SOURCE_DIR)
        get_filename_component(ONEDNN_BENCH_SOURCE_DIR
            "${VECOPS_ONEDNN_SOURCE_DIR}" ABSOLUTE)
        if(NOT EXISTS "${ONEDNN_BENCH_SOURCE_DIR}/CMakeLists.txt")
            message(FATAL_ERROR
                "Invalid oneDNN source tree: ${ONEDNN_BENCH_SOURCE_DIR}")
        endif()
        add_subdirectory(
            "${ONEDNN_BENCH_SOURCE_DIR}"
            "${CMAKE_BINARY_DIR}/_deps/vecops_onednn-build"
            EXCLUDE_FROM_ALL)
    else()
        FetchContent_Declare(
            vecops_onednn
            GIT_REPOSITORY
                "${VECOPS_GITHUB_ENDPOINT}/uxlfoundation/oneDNN.git"
            GIT_TAG "${VECOPS_ONEDNN_GIT_TAG}"
            GIT_SHALLOW TRUE
        )
        FetchContent_MakeAvailable(vecops_onednn)
    endif()

    function(add_onednn_benchmark NAME SOURCE)
        set(TARGET_NAME "${NAME}-${VECOPS_ONEDNN_BENCH_ARCH}")
        add_executable(${TARGET_NAME} "${SOURCE}")
        target_include_directories(${TARGET_NAME} PRIVATE
            ${CMAKE_CURRENT_SOURCE_DIR})
        target_compile_options(${TARGET_NAME} PRIVATE
            -march=${VECOPS_MAP_${VECOPS_ARCH_FAMILY}_Native})
        target_compile_definitions(${TARGET_NAME} PRIVATE
            VECOPS_BENCH_ARCH_CODE="${VECOPS_ONEDNN_BENCH_ARCH}"
            VECOPS_BENCH_USE_ONEDNN=1
            VECOPS_BENCH_ONEDNN_IMPL_TOKEN="${VECOPS_ONEDNN_IMPL_TOKEN}"
            VECOPS_SOURCE_DIR="${CMAKE_SOURCE_DIR}"
        )
        target_link_libraries(${TARGET_NAME} PRIVATE
            vecops benchmark::benchmark dnnl)
    endfunction()

    add_onednn_benchmark(LayerNormBench
        ${CMAKE_CURRENT_SOURCE_DIR}/ops/LayerNormBench.cpp)
    add_onednn_benchmark(SoftmaxBench
        ${CMAKE_CURRENT_SOURCE_DIR}/ops/SoftmaxBench.cpp)
endif()

add_multiarch_benchmark(ConversionLoadStoreBench
    FILES ${CMAKE_CURRENT_SOURCE_DIR}/vec/ConversionLoadStoreBench.cpp
    ARCH Native
)

add_multiarch_benchmark(ExpBench
    FILES ${CMAKE_CURRENT_SOURCE_DIR}/vec/ExpBench.cpp
    ARCH Native
)

set(INSTRUCTION_BENCH_FILES
    ${CMAKE_CURRENT_SOURCE_DIR}/inst/InstructionBench.cpp
)
if(VECOPS_ARCH_FAMILY STREQUAL "x86")
    list(APPEND INSTRUCTION_BENCH_FILES
        ${CMAKE_CURRENT_SOURCE_DIR}/inst/InstructionBenchX86.cpp)
elseif(VECOPS_ARCH_FAMILY STREQUAL "ARM")
    list(APPEND INSTRUCTION_BENCH_FILES
        ${CMAKE_CURRENT_SOURCE_DIR}/inst/InstructionBenchSVE.cpp)
endif()

if(VECOPS_ARCH_FAMILY STREQUAL "x86")
    set(INSTRUCTION_BENCH_ARCH AVX2 AVX512 Native)
else()
    set(INSTRUCTION_BENCH_ARCH SVE SVE2 Native)
endif()

add_multiarch_benchmark(InstructionBench
    FILES ${INSTRUCTION_BENCH_FILES}
    ARCH ${INSTRUCTION_BENCH_ARCH}
)

# Keep the instruction kernels easy to audit and prevent project-wide release
# or LTO settings from changing their intended O2 code shape.
foreach(CODE ${INSTRUCTION_BENCH_ARCH})
    target_compile_options(InstructionBench-${CODE} PRIVATE -O2 -fno-lto)
    set_property(TARGET InstructionBench-${CODE}
        PROPERTY INTERPROCEDURAL_OPTIMIZATION FALSE)
endforeach()
