include("${CMAKE_CURRENT_LIST_DIR}/../cmake/VecopsMultiarchTargets.cmake")

add_compile_definitions(VECOPS_MATH_ASSUME_VALID_INPUTS)

# Benchmark wrapper around the common sharding and architecture primitive.
function(vecops_add_benchmark)
    set(options FIXED_SVE)
    set(oneValueArgs NAME FOLDER)
    set(multiValueArgs FILES ARCH DEFINITIONS LIBRARIES)
    cmake_parse_arguments(ARG "${options}" "${oneValueArgs}" "${multiValueArgs}" ${ARGN})
    if(NOT ARG_NAME)
        message(FATAL_ERROR "vecops_add_benchmark requires NAME")
    endif()
    if(NOT ARG_FILES)
        message(FATAL_ERROR "vecops_add_benchmark(${ARG_NAME}) requires FILES")
    endif()

    if(NOT ARG_ARCH)
        set(ARG_ARCH ${VECOPS_ARCH_LIST_${VECOPS_ARCH_FAMILY}})
    endif()

    set(_FIXED_SVE_ARG "")
    if(ARG_FIXED_SVE)
        set(_FIXED_SVE_ARG FIXED_SVE)
    endif()
    vecops_add_multiarch_executable(
        ${_FIXED_SVE_ARG}
        NAME ${ARG_NAME}
        FILES ${ARG_FILES}
        FOLDER ${ARG_FOLDER}
        ARCH ${ARG_ARCH}
        ARCH_DEFINITION VECOPS_BENCH_ARCH_CODE
        DEFINITIONS VECOPS_SOURCE_DIR="${CMAKE_SOURCE_DIR}" ${ARG_DEFINITIONS}
        LIBRARIES vecops benchmark::benchmark ${ARG_LIBRARIES}
        INCLUDE_DIRECTORIES "${CMAKE_CURRENT_SOURCE_DIR}")
endfunction()

vecops_add_benchmark(
    NAME LayerNormBench
    FILES ${CMAKE_CURRENT_SOURCE_DIR}/ops/LayerNormBench.cpp
    ARCH Native Scalar
)

vecops_add_benchmark(
    NAME SoftmaxBench
    FILES ${CMAKE_CURRENT_SOURCE_DIR}/ops/SoftmaxBench.cpp
    ARCH Native Scalar
)

vecops_add_benchmark(
    NAME TransposeBench
    FILES
        ${CMAKE_CURRENT_SOURCE_DIR}/ops/TransposeBench.cpp
    ARCH Native SVE2 Scalar
)

if(VECOPS_ARCH_FAMILY STREQUAL "ARM")
    vecops_add_benchmark(
        NAME TransposeCompareBench
        FIXED_SVE
        FILES
            ${CMAKE_CURRENT_SOURCE_DIR}/ops/TransposeCompareBench.cpp
        ARCH Native SVE2
    )
endif()

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
        if(VECOPS_ARCH_FAMILY STREQUAL "ARM" AND
           CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
            target_compile_options(${TARGET_NAME} PRIVATE -Wno-psabi)
        endif()
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

vecops_add_benchmark(
    NAME ConversionLoadStoreBench
    FILES ${CMAKE_CURRENT_SOURCE_DIR}/vec/ConversionLoadStoreBench.cpp
    ARCH Native
)

vecops_add_benchmark(
    NAME DataAccessBench
    FILES ${CMAKE_CURRENT_SOURCE_DIR}/tensor/DataAccessBench.cpp
    ARCH Native
)

vecops_add_benchmark(
    NAME ExpBench
    FILES ${CMAKE_CURRENT_SOURCE_DIR}/vec/ExpBench.cpp
    ARCH Native
)

vecops_add_benchmark(
    NAME RsqrtBench
    FILES ${CMAKE_CURRENT_SOURCE_DIR}/vec/RsqrtBench.cpp
    ARCH Native
)

vecops_add_benchmark(
    NAME RcpBench
    FILES ${CMAKE_CURRENT_SOURCE_DIR}/vec/RcpBench.cpp
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

vecops_add_benchmark(
    NAME InstructionBench
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
