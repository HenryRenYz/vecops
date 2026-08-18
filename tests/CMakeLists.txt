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

# ── Static Code → -march Mappings ────────────────────────────
# x86
set(VECOPS_MAP_x86_Scalar    "x86-64")
set(VECOPS_MAP_x86_AVX       "corei7-avx")
set(VECOPS_MAP_x86_AVX2      "core-avx2")
set(VECOPS_MAP_x86_AVX512    "skylake-avx512")
set(VECOPS_MAP_x86_Native    "native")

# ARM (static mappings)
set(VECOPS_MAP_ARM_Scalar    "armv8-a")
set(VECOPS_MAP_ARM_NEON      "armv8-a+simd")
set(VECOPS_MAP_ARM_SVE       "armv8-a+sve")
set(VECOPS_MAP_ARM_SVE2      "armv8-a+sve2")

# ARM Native: starts from native, conditionally appends advertised extensions.
# because some compilers (e.g. BiSheng) do not auto-enable them with -march=native
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

# NativeFixedSVE uses the native ISA feature set while making SVE types sized.
# An explicit cache value is useful for testing a smaller supported VL. When it
# is omitted, ask Linux for the largest VL supported by the current machine;
# the probe runs in a child process and does not change the configure process.
if(VECOPS_ARCH_FAMILY STREQUAL "ARM")
    set(VECOPS_NATIVE_FIXED_SVE_BITS "${VECOPS_FIXED_SVE_BITS}")
    if(NOT VECOPS_NATIVE_FIXED_SVE_BITS AND NOT CMAKE_CROSSCOMPILING)
        set(_SVE_VL_PROBE "${CMAKE_BINARY_DIR}/CMakeFiles/vecops_sve_vl_probe.c")
        file(WRITE "${_SVE_VL_PROBE}" [=[
#include <stdio.h>
#include <sys/prctl.h>

#ifndef PR_SVE_SET_VL
#define PR_SVE_SET_VL 50
#endif
#ifndef PR_SVE_VL_LEN_MASK
#define PR_SVE_VL_LEN_MASK 0xffff
#endif

int main(void) {
  const int result = prctl(PR_SVE_SET_VL, 256UL, 0UL, 0UL, 0UL);
  if (result < 0) return 1;
  printf("%d", (result & PR_SVE_VL_LEN_MASK) * 8);
  return 0;
}
]=])
        try_run(
            _SVE_VL_RUN_RESULT
            _SVE_VL_COMPILE_RESULT
            "${CMAKE_BINARY_DIR}/CMakeFiles/vecops_sve_vl_probe"
            "${_SVE_VL_PROBE}"
            RUN_OUTPUT_VARIABLE _SVE_VL_OUTPUT
        )
        if(_SVE_VL_COMPILE_RESULT AND _SVE_VL_RUN_RESULT EQUAL 0)
            string(STRIP "${_SVE_VL_OUTPUT}" VECOPS_NATIVE_FIXED_SVE_BITS)
            message(STATUS
                "Detected maximum native SVE width: ${VECOPS_NATIVE_FIXED_SVE_BITS} bits")
        else()
            message(WARNING
                "Could not detect the maximum native SVE width; "
                "NativeFixedSVE tests will be skipped. Set VECOPS_FIXED_SVE_BITS explicitly to enable them.")
        endif()
    endif()

    if(VECOPS_NATIVE_FIXED_SVE_BITS)
        if(NOT VECOPS_NATIVE_FIXED_SVE_BITS MATCHES "^[0-9]+$")
            message(FATAL_ERROR "VECOPS_FIXED_SVE_BITS must be an integer number of bits")
        endif()
        math(EXPR _SVE_VL_REMAINDER "${VECOPS_NATIVE_FIXED_SVE_BITS} % 128")
        if(VECOPS_NATIVE_FIXED_SVE_BITS LESS 128 OR
           VECOPS_NATIVE_FIXED_SVE_BITS GREATER 2048 OR
           NOT _SVE_VL_REMAINDER EQUAL 0)
            message(FATAL_ERROR
                "VECOPS_FIXED_SVE_BITS must be a multiple of 128 in [128, 2048]")
        endif()
        set(VECOPS_MAP_ARM_NativeFixedSVE "${VECOPS_MAP_ARM_Native}")
    endif()
endif()

# ── Default Arch Code Lists (full sequence per family) ──────
set(VECOPS_ARCH_LIST_x86 Scalar AVX AVX2 AVX512 Native)
set(VECOPS_ARCH_LIST_ARM Scalar SVE SVE2 Native)
set(VECOPS_ARCH_LIST ${VECOPS_ARCH_LIST_${VECOPS_ARCH_FAMILY}})

# ======================================================================
# GDB Pretty-Printer Discovery
# ======================================================================

file(GLOB _GDB_SCRIPTS "${CMAKE_SOURCE_DIR}/scripts/*.py")

# ======================================================================
# add_multiarch_executable
# ======================================================================

function(add_multiarch_executable NAME)
    set(options "")
    set(oneValueArgs "")
    set(multiValueArgs FILES ARCH DEFINITIONS LABELS)
    cmake_parse_arguments(ARG "${options}" "${oneValueArgs}" "${multiValueArgs}" ${ARGN})

    if(NOT ARG_ARCH)
        set(ARG_ARCH ${VECOPS_ARCH_LIST_${VECOPS_ARCH_FAMILY}})
    endif()

    foreach(CODE ${ARG_ARCH})
        set(MARCH ${VECOPS_MAP_${VECOPS_ARCH_FAMILY}_${CODE}})
        if(NOT MARCH)
            message(STATUS "Skipping arch '${CODE}' — not defined for ${VECOPS_ARCH_FAMILY}")
            continue()
        endif()

        set(TARGET_NAME "${NAME}-${CODE}")
        add_executable(${TARGET_NAME} ${ARG_FILES})
        target_compile_options(${TARGET_NAME} PRIVATE -march=${MARCH})
        if(VECOPS_ARCH_FAMILY STREQUAL "ARM" AND CODE STREQUAL "NativeFixedSVE")
            target_compile_options(${TARGET_NAME} PRIVATE
                -msve-vector-bits=${VECOPS_NATIVE_FIXED_SVE_BITS})
        endif()
        if(ARG_DEFINITIONS)
            target_compile_definitions(${TARGET_NAME} PRIVATE ${ARG_DEFINITIONS})
        endif()
        if(CODE STREQUAL "Scalar")
            target_compile_definitions(${TARGET_NAME} PRIVATE
                CPU_CAPABILITY=GENERIC
                CPU_CAPABILITY_GENERIC=1
            )
        endif()
        target_link_libraries(${TARGET_NAME} PRIVATE vecops gtest gtest_main)
        string(JOIN "+" _GTEST_LABELS ${ARG_LABELS})
        gtest_discover_tests(${TARGET_NAME}
            PROPERTIES LABELS "${_GTEST_LABELS}")

        # ── GDB pretty-printer auto-load ───────────────────
        list(LENGTH _GDB_SCRIPTS _NUM_GDB_SCRIPTS)
        if(_NUM_GDB_SCRIPTS LESS_EQUAL 0)
            continue()
        endif()

        set(_IMPORTS "")
        foreach(_MOD ${_GDB_SCRIPTS})
            get_filename_component(_MNAME "${_MOD}" NAME_WE)
            string(APPEND _IMPORTS "try:\n    import ${_MNAME}\nexcept Exception:\n    pass\n")
        endforeach()

        set(_LOADER "${CMAKE_CURRENT_BINARY_DIR}/${TARGET_NAME}-gdb.py")
        file(WRITE "${_LOADER}"
"import sys
sys.path.insert(0, '${CMAKE_SOURCE_DIR}/scripts')

${_IMPORTS}")

        add_custom_command(TARGET ${TARGET_NAME} POST_BUILD
            COMMAND ${CMAKE_COMMAND} -E copy_if_different
                "${_LOADER}"
                "$<TARGET_FILE_DIR:${TARGET_NAME}>/${TARGET_NAME}-gdb.py"
            COMMENT "Installing GDB pretty-printer for ${TARGET_NAME}"
        )
    endforeach()
endfunction()

# ======================================================================
# Tests
# ======================================================================

include_directories(${CMAKE_CURRENT_SOURCE_DIR})

add_executable(ArrayTest ${CMAKE_CURRENT_SOURCE_DIR}/array/ArrayTest.cpp)
target_link_libraries(ArrayTest PRIVATE vecops gtest gtest_main)
gtest_discover_tests(ArrayTest)

add_executable(TensorTest ${CMAKE_CURRENT_SOURCE_DIR}/tensor/TensorTest.cpp)
target_link_libraries(TensorTest PRIVATE vecops gtest gtest_main)
gtest_discover_tests(TensorTest)

add_executable(GemmTest ${CMAKE_CURRENT_SOURCE_DIR}/array/GemmTest.cpp)
target_link_libraries(GemmTest PRIVATE vecops gtest gtest_main)
gtest_discover_tests(GemmTest)

add_executable(LayoutTest ${CMAKE_CURRENT_SOURCE_DIR}/tensor/LayoutTest.cpp)
target_link_libraries(LayoutTest PRIVATE vecops gtest gtest_main)
gtest_discover_tests(LayoutTest)

add_executable(KernelLoopTest ${CMAKE_CURRENT_SOURCE_DIR}/kernel/LoopTest.cpp)
target_link_libraries(KernelLoopTest PRIVATE vecops gtest gtest_main)
gtest_discover_tests(KernelLoopTest)

if(VECOPS_ARCH_FAMILY STREQUAL "ARM")
    add_multiarch_executable(KernelLoopSizelessTest
        FILES ${CMAKE_CURRENT_SOURCE_DIR}/kernel/LoopSizelessTest.cpp
        ARCH SVE
    )
endif()

if(VECOPS_ARCH_FAMILY STREQUAL "x86")
    set(VECOPS_TENSOR_KERNEL_ARCH Scalar AVX512 Native)
else()
    set(VECOPS_TENSOR_KERNEL_ARCH Scalar SVE SVE2 Native)
endif()

add_multiarch_executable(TensorDataAccessTest
    FILES ${CMAKE_CURRENT_SOURCE_DIR}/tensor/DataAccessTest.cpp
          ${CMAKE_CURRENT_SOURCE_DIR}/TestUtils.cpp
    ARCH ${VECOPS_TENSOR_KERNEL_ARCH}
)

set(VEC_BASE_ARCH ${VECOPS_ARCH_LIST})
if(VECOPS_ARCH_FAMILY STREQUAL "ARM")
    if(VECOPS_NATIVE_FIXED_SVE_BITS)
        list(APPEND VEC_BASE_ARCH NativeFixedSVE)
    endif()
endif()
add_multiarch_executable(VecBaseTest
    FILES ${CMAKE_CURRENT_SOURCE_DIR}/vec/BaseTest.cpp
    ARCH ${VEC_BASE_ARCH}
    LABELS vec
)
add_multiarch_executable(VecOptionsTest
    FILES ${CMAKE_CURRENT_SOURCE_DIR}/vec/OptionsTest.cpp
    ARCH ${VEC_BASE_ARCH}
    LABELS vec
)
add_multiarch_executable(VecArithmeticTest
    FILES ${CMAKE_CURRENT_SOURCE_DIR}/vec/ArithmeticTest.cpp
    ARCH ${VEC_BASE_ARCH}
    LABELS vec
)
add_multiarch_executable(VecUnaryArithmeticTest
    FILES ${CMAKE_CURRENT_SOURCE_DIR}/vec/UnaryArithmeticTest.cpp
    ARCH ${VEC_BASE_ARCH}
    LABELS vec
)
add_multiarch_executable(VecFloatingUnaryTest
    FILES ${CMAKE_CURRENT_SOURCE_DIR}/vec/FloatingUnaryTest.cpp
    ARCH ${VEC_BASE_ARCH}
    LABELS vec
)
add_multiarch_executable(VecMathTest
    FILES ${CMAKE_CURRENT_SOURCE_DIR}/vec/MathTest.cpp
    ARCH ${VEC_BASE_ARCH}
    LABELS vec
)
add_multiarch_executable(VecMemoryTest
    FILES ${CMAKE_CURRENT_SOURCE_DIR}/vec/MemoryTest.cpp
    ARCH ${VEC_BASE_ARCH}
    LABELS vec
)
add_multiarch_executable(VecPrefetchTest
    FILES ${CMAKE_CURRENT_SOURCE_DIR}/vec/PrefetchTest.cpp
    ARCH ${VEC_BASE_ARCH}
    LABELS vec
)
add_multiarch_executable(VecMaskConversionTest
    FILES ${CMAKE_CURRENT_SOURCE_DIR}/vec/MaskConversionTest.cpp
    ARCH ${VEC_BASE_ARCH}
    LABELS vec vec_conversion
)
add_multiarch_executable(VecConversionTest
    FILES ${CMAKE_CURRENT_SOURCE_DIR}/vec/ConversionTest.cpp
    ARCH ${VEC_BASE_ARCH}
    LABELS vec vec_conversion
)
add_multiarch_executable(VecConversionMemoryTest
    FILES ${CMAKE_CURRENT_SOURCE_DIR}/vec/ConversionMemoryTest.cpp
    ARCH ${VEC_BASE_ARCH}
    LABELS vec vec_conversion
)
add_multiarch_executable(VecReductionTest
    FILES ${CMAKE_CURRENT_SOURCE_DIR}/vec/ReductionTest.cpp
    ARCH ${VEC_BASE_ARCH}
    LABELS vec
)
add_multiarch_executable(VecBitTest
    FILES ${CMAKE_CURRENT_SOURCE_DIR}/vec/BitTest.cpp
    ARCH ${VEC_BASE_ARCH}
    LABELS vec
)
add_multiarch_executable(VecComparisonTest
    FILES ${CMAKE_CURRENT_SOURCE_DIR}/vec/ComparisonTest.cpp
    ARCH ${VEC_BASE_ARCH}
    LABELS vec
)
add_multiarch_executable(VecBasicTest
    FILES ${CMAKE_CURRENT_SOURCE_DIR}/vec/BasicTest.cpp
    ARCH ${VEC_BASE_ARCH}
    LABELS vec
)
add_multiarch_executable(VecShuffleTest
    FILES ${CMAKE_CURRENT_SOURCE_DIR}/vec/ShuffleTest.cpp
    ARCH ${VEC_BASE_ARCH}
    LABELS vec
)
add_multiarch_executable(VecTaglessTest
    FILES ${CMAKE_CURRENT_SOURCE_DIR}/vec/TaglessTest.cpp
    ARCH ${VEC_BASE_ARCH}
    LABELS vec
)

add_multiarch_executable(VecOverloadContractTest
    FILES ${CMAKE_CURRENT_SOURCE_DIR}/vec/OverloadContractTest.cpp
    ARCH ${VEC_BASE_ARCH}
    LABELS vec
)

foreach(_SOURCE_BYTES IN ITEMS 1 2 4 8)
    add_multiarch_executable(VecConversionMatrix${_SOURCE_BYTES}Test
        FILES ${CMAKE_CURRENT_SOURCE_DIR}/vec/ConversionMatrixTest.cpp
        ARCH ${VEC_BASE_ARCH}
        DEFINITIONS VECOPS_TEST_SOURCE_BYTES=${_SOURCE_BYTES}
        LABELS vec vec_conversion vec_exhaustive
    )
endforeach()

# ConversionMemoryMatrix is split per source element type: one heavy TU per
# type keeps parallel build wall time low while preserving full coverage.
set(_CONV_MEM_TYPES_1 vecops::int8_t vecops::uint8_t)
set(_CONV_MEM_TYPES_2 vecops::bfloat16_t vecops::float16_t vecops::int16_t vecops::uint16_t)
set(_CONV_MEM_TYPES_4 vecops::float32_t vecops::int32_t vecops::uint32_t)
set(_CONV_MEM_TYPES_8 vecops::float64_t vecops::int64_t vecops::uint64_t)
foreach(_SOURCE_BYTES IN ITEMS 1 2 4 8)
    foreach(_SOURCE_ELEM IN ITEMS ${_CONV_MEM_TYPES_${_SOURCE_BYTES}})
        string(REPLACE "vecops::" "" _ELEM_CLEAN "${_SOURCE_ELEM}")
        add_multiarch_executable(VecConversionMemory${_SOURCE_BYTES}${_ELEM_CLEAN}Test
            FILES ${CMAKE_CURRENT_SOURCE_DIR}/vec/ConversionMemoryMatrixTest.cpp
            ARCH ${VEC_BASE_ARCH}
            DEFINITIONS VECOPS_TEST_SOURCE_BYTES=${_SOURCE_BYTES}
                       VECOPS_TEST_SOURCE_ELEM=${_SOURCE_ELEM}
            LABELS vec vec_conversion vec_exhaustive
        )
    endforeach()
endforeach()

add_multiarch_executable(TensorTransformTest
    FILES ${CMAKE_CURRENT_SOURCE_DIR}/tensor/TransformTest.cpp
    ARCH ${VECOPS_TENSOR_KERNEL_ARCH}
)

add_multiarch_executable(LayerNormTest
    FILES ${CMAKE_CURRENT_SOURCE_DIR}/ops/LayerNormTest.cpp
          ${CMAKE_CURRENT_SOURCE_DIR}/ops/LayerNormDTypeFp16Test.cpp
          ${CMAKE_CURRENT_SOURCE_DIR}/ops/LayerNormDTypeFp32Test.cpp
          ${CMAKE_CURRENT_SOURCE_DIR}/ops/LayerNormDTypeFp64Test.cpp
          ${CMAKE_CURRENT_SOURCE_DIR}/ops/LayerNormDTypeBf16Test.cpp
          ${CMAKE_CURRENT_SOURCE_DIR}/TestUtils.cpp
    ARCH ${VECOPS_TENSOR_KERNEL_ARCH}
)

add_multiarch_executable(SoftmaxTest
    FILES ${CMAKE_CURRENT_SOURCE_DIR}/ops/SoftmaxTest.cpp
          ${CMAKE_CURRENT_SOURCE_DIR}/TestUtils.cpp
    ARCH ${VECOPS_TENSOR_KERNEL_ARCH}
)
