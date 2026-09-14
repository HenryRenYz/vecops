include_guard(GLOBAL)

include(CheckCXXCompilerFlag)
include("${CMAKE_CURRENT_LIST_DIR}/VecopsTargetArch.cmake")

# Architecture metadata shared by tests, benchmarks, and any future
# multi-architecture executable targets.
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

# Some compilers (including BiSheng releases) do not enable all advertised
# native extensions with the bare -march=native spelling.
set(VECOPS_MAP_ARM_Native "native")
set(VECOPS_NATIVE_HAS_SME OFF)
set(VECOPS_NATIVE_HAS_SME_FA64 OFF)
set(VECOPS_NATIVE_HAS_SME_F64F64 OFF)
set(VECOPS_NATIVE_HAS_I8MM OFF)
if(VECOPS_ARCH_FAMILY STREQUAL "ARM" AND EXISTS "/proc/cpuinfo")
    file(READ "/proc/cpuinfo" _VECOPS_CPUINFO)
    string(REGEX MATCH "Features[ \t]*:.*" _VECOPS_FEAT_LINE
        "${_VECOPS_CPUINFO}")
    # GCC does not permit feature modifiers after the special "native"
    # architecture name.  The 920 SME toolchain also resolves its custom CPU
    # to a generic core, so spell the architectural baseline explicitly before
    # appending the features reported by Linux.  SME and SVE2.1 imply an
    # Armv9-A baseline; the remaining advertised extensions are valid on the
    # Armv8-A baseline.
    if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
        if(_VECOPS_FEAT_LINE MATCHES " (sme|sve2p1) ")
            set(VECOPS_MAP_ARM_Native "armv9-a")
        else()
            set(VECOPS_MAP_ARM_Native "armv8-a")
        endif()
    endif()
    # Matrix-vector fallbacks use the ordinary-SVE dot/MMLA extensions when
    # the native CPU exposes them.  These are independent of SME_FA64:
    # i8mm enables SVE integer MMLA, while f32mm/f64mm enable FP FMMLA.
    foreach(_VECOPS_FEAT IN ITEMS sve2p1 bf16 i8mm sme)
        string(FIND "${_VECOPS_FEAT_LINE}" " ${_VECOPS_FEAT} " _VECOPS_POS)
        if(NOT _VECOPS_POS EQUAL -1)
            string(APPEND VECOPS_MAP_ARM_Native "+${_VECOPS_FEAT}")
            if(_VECOPS_FEAT STREQUAL "sme")
                set(VECOPS_NATIVE_HAS_SME ON)
            elseif(_VECOPS_FEAT STREQUAL "i8mm")
                set(VECOPS_NATIVE_HAS_I8MM ON)
            endif()
        endif()
    endforeach()
    # Linux names the SVE floating matrix HWCAPs svef32mm/svef64mm,
    # whereas compiler feature modifiers omit the "sve" prefix.
    foreach(_VECOPS_MM_FEAT IN ITEMS f32mm f64mm)
        if(_VECOPS_FEAT_LINE MATCHES " (sve)?${_VECOPS_MM_FEAT} ")
            string(APPEND VECOPS_MAP_ARM_Native "+${_VECOPS_MM_FEAT}")
        endif()
    endforeach()
    # Linux reports FEAT_SME_FA64 as "smefa64", while compiler -march
    # strings use "sme-fa64". Native SME code needs the explicit compiler
    # feature because some compilers do not infer it from -march=native.
    string(FIND "${_VECOPS_FEAT_LINE}" " smefa64 " _VECOPS_SME_FA64_POS)
    if(NOT _VECOPS_SME_FA64_POS EQUAL -1)
        set(VECOPS_NATIVE_HAS_SME_FA64 ON)
        # GCC 15 recognizes the hardware and accepts SME asm, but does not
        # accept "sme-fa64" as an -march feature modifier. Clang needs the
        # explicit spelling because -march=native can omit it.
        if(NOT CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
            string(APPEND VECOPS_MAP_ARM_Native "+sme-fa64")
        endif()
    endif()
    string(FIND "${_VECOPS_FEAT_LINE}" " smef64f64 " _VECOPS_SME_F64_POS)
    if(NOT _VECOPS_SME_F64_POS EQUAL -1)
        set(VECOPS_NATIVE_HAS_SME_F64F64 ON)
        string(APPEND VECOPS_MAP_ARM_Native "+sme-f64f64")
    endif()

    check_cxx_compiler_flag(
        "-march=${VECOPS_MAP_ARM_Native}"
        VECOPS_COMPILER_ACCEPTS_ARM_NATIVE_FEATURES)
    if(NOT VECOPS_COMPILER_ACCEPTS_ARM_NATIVE_FEATURES)
        message(STATUS
            "Compiler rejects -march=${VECOPS_MAP_ARM_Native}; "
            "falling back to -march=native")
        set(VECOPS_MAP_ARM_Native "native")
    endif()
    # This file is normally evaluated in the Vecops subdirectory, while a
    # generated kernel target is added by its parent project. Preserve the
    # detected mapping and feature facts in the CMake cache so
    # VecopsTargetArch.cmake sees the same Native contract across that
    # directory-scope boundary. Without this, BiSheng clang falls back to bare
    # -march=native and omits SME even when Linux advertises it.
    set(VECOPS_MAP_ARM_Native "${VECOPS_MAP_ARM_Native}" CACHE INTERNAL
        "Detected native ARM architecture spelling" FORCE)
    set(VECOPS_NATIVE_HAS_SME "${VECOPS_NATIVE_HAS_SME}" CACHE INTERNAL
        "Native ARM target has SME" FORCE)
    set(VECOPS_NATIVE_HAS_SME_FA64 "${VECOPS_NATIVE_HAS_SME_FA64}"
        CACHE INTERNAL "Native ARM target has SME FA64" FORCE)
    set(VECOPS_NATIVE_HAS_SME_F64F64 "${VECOPS_NATIVE_HAS_SME_F64F64}"
        CACHE INTERNAL "Native ARM target has SME F64F64" FORCE)
    set(VECOPS_NATIVE_HAS_I8MM "${VECOPS_NATIVE_HAS_I8MM}" CACHE INTERNAL
        "Native ARM target has I8MM" FORCE)
    unset(_VECOPS_CPUINFO)
    unset(_VECOPS_FEAT_LINE)
endif()

# NativeFixedSVE uses the native ISA feature set while making SVE types sized.
if(VECOPS_ARCH_FAMILY STREQUAL "ARM")
    set(VECOPS_NATIVE_FIXED_SVE_BITS "${VECOPS_FIXED_SVE_BITS}")
    if(NOT VECOPS_NATIVE_FIXED_SVE_BITS AND NOT CMAKE_CROSSCOMPILING)
        set(_VECOPS_SVE_VL_PROBE
            "${CMAKE_BINARY_DIR}/CMakeFiles/vecops_sve_vl_probe.c")
        file(WRITE "${_VECOPS_SVE_VL_PROBE}" [=[
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
            _VECOPS_SVE_VL_RUN_RESULT
            _VECOPS_SVE_VL_COMPILE_RESULT
            "${CMAKE_BINARY_DIR}/CMakeFiles/vecops_sve_vl_probe"
            "${_VECOPS_SVE_VL_PROBE}"
            RUN_OUTPUT_VARIABLE _VECOPS_SVE_VL_OUTPUT)
        if(_VECOPS_SVE_VL_COMPILE_RESULT AND
           _VECOPS_SVE_VL_RUN_RESULT EQUAL 0)
            string(STRIP "${_VECOPS_SVE_VL_OUTPUT}"
                VECOPS_NATIVE_FIXED_SVE_BITS)
            message(STATUS
                "Detected maximum native SVE width: "
                "${VECOPS_NATIVE_FIXED_SVE_BITS} bits")
        else()
            message(WARNING
                "Could not detect the maximum native SVE width; "
                "NativeFixedSVE targets will be skipped. Set "
                "VECOPS_FIXED_SVE_BITS explicitly to enable them.")
        endif()
    endif()

    if(VECOPS_NATIVE_FIXED_SVE_BITS)
        if(NOT VECOPS_NATIVE_FIXED_SVE_BITS MATCHES "^[0-9]+$")
            message(FATAL_ERROR
                "VECOPS_FIXED_SVE_BITS must be an integer number of bits")
        endif()
        math(EXPR _VECOPS_SVE_VL_REMAINDER
            "${VECOPS_NATIVE_FIXED_SVE_BITS} % 128")
        if(VECOPS_NATIVE_FIXED_SVE_BITS LESS 128 OR
           VECOPS_NATIVE_FIXED_SVE_BITS GREATER 2048 OR
           NOT _VECOPS_SVE_VL_REMAINDER EQUAL 0)
            message(FATAL_ERROR
                "VECOPS_FIXED_SVE_BITS must be a multiple of 128 in [128, 2048]")
        endif()
        set(VECOPS_MAP_ARM_NativeFixedSVE "${VECOPS_MAP_ARM_Native}")
    endif()
endif()

# Fixed streaming SVE is independent of the ordinary SVE VL.  Detect the
# native SME SVL separately, and let cross builds provide the target guarantee
# explicitly.  Probe -msve-streaming-vector-bits because support varies by
# compiler version (and GCC does not currently document the option).  The
# project feature definition remains useful on compiler-specific manual-SME
# targets when the driver flag is unavailable.
if(VECOPS_ARCH_FAMILY STREQUAL "ARM")
    set(VECOPS_NATIVE_FIXED_STREAMING_SVE_BITS
        "${VECOPS_FIXED_STREAMING_SVE_BITS}")
    if(NOT VECOPS_NATIVE_FIXED_STREAMING_SVE_BITS AND
       NOT CMAKE_CROSSCOMPILING AND VECOPS_NATIVE_HAS_SME)
        set(_VECOPS_SME_SVL_PROBE
            "${CMAKE_BINARY_DIR}/CMakeFiles/vecops_sme_svl_probe.c")
        file(WRITE "${_VECOPS_SME_SVL_PROBE}" [=[
#include <stdio.h>
#include <sys/prctl.h>

#ifndef PR_SME_SET_VL
#define PR_SME_SET_VL 63
#endif
#ifndef PR_SME_VL_LEN_MASK
#define PR_SME_VL_LEN_MASK 0xffff
#endif

int main(void) {
  const int result = prctl(PR_SME_SET_VL, 256UL, 0UL, 0UL, 0UL);
  if (result < 0) return 1;
  printf("%d", (result & PR_SME_VL_LEN_MASK) * 8);
  return 0;
}
]=])
        try_run(
            _VECOPS_SME_SVL_RUN_RESULT
            _VECOPS_SME_SVL_COMPILE_RESULT
            "${CMAKE_BINARY_DIR}/CMakeFiles/vecops_sme_svl_probe"
            "${_VECOPS_SME_SVL_PROBE}"
            RUN_OUTPUT_VARIABLE _VECOPS_SME_SVL_OUTPUT)
        if(_VECOPS_SME_SVL_COMPILE_RESULT AND
           _VECOPS_SME_SVL_RUN_RESULT EQUAL 0)
            string(STRIP "${_VECOPS_SME_SVL_OUTPUT}"
                VECOPS_NATIVE_FIXED_STREAMING_SVE_BITS)
            message(STATUS
                "Detected maximum native streaming SVE width: "
                "${VECOPS_NATIVE_FIXED_STREAMING_SVE_BITS} bits")
        else()
            message(WARNING
                "Could not detect the maximum native streaming SVE width; "
                "set VECOPS_FIXED_STREAMING_SVE_BITS explicitly to enable "
                "fixed-SVL SME metadata.")
        endif()
    endif()

    if(VECOPS_NATIVE_FIXED_STREAMING_SVE_BITS)
        if(NOT VECOPS_NATIVE_FIXED_STREAMING_SVE_BITS MATCHES "^[0-9]+$")
            message(FATAL_ERROR
                "VECOPS_FIXED_STREAMING_SVE_BITS must be an integer number "
                "of bits")
        endif()
        math(EXPR _VECOPS_SME_SVL_REMAINDER
            "${VECOPS_NATIVE_FIXED_STREAMING_SVE_BITS} % 128")
        if(VECOPS_NATIVE_FIXED_STREAMING_SVE_BITS LESS 128 OR
           VECOPS_NATIVE_FIXED_STREAMING_SVE_BITS GREATER 2048 OR
           NOT _VECOPS_SME_SVL_REMAINDER EQUAL 0)
            message(FATAL_ERROR
                "VECOPS_FIXED_STREAMING_SVE_BITS must be a multiple of 128 "
                "in [128, 2048]")
        endif()

        set(_VECOPS_SAVED_REQUIRED_FLAGS "${CMAKE_REQUIRED_FLAGS}")
        set(CMAKE_REQUIRED_FLAGS
            "${CMAKE_REQUIRED_FLAGS} -march=${VECOPS_MAP_ARM_Native}")
        check_cxx_compiler_flag(
            "-msve-streaming-vector-bits=${VECOPS_NATIVE_FIXED_STREAMING_SVE_BITS}"
            VECOPS_COMPILER_ACCEPTS_FIXED_STREAMING_SVE_BITS)
        set(CMAKE_REQUIRED_FLAGS "${_VECOPS_SAVED_REQUIRED_FLAGS}")
        if(NOT VECOPS_COMPILER_ACCEPTS_FIXED_STREAMING_SVE_BITS)
            message(STATUS
                "Compiler has no -msve-streaming-vector-bits support; "
                "using the fixed-SVL project target guarantee only")
        endif()
        set(VECOPS_MAP_ARM_NativeFixedStreamingSVE
            "${VECOPS_MAP_ARM_Native}")
    endif()
endif()

set(VECOPS_ARCH_LIST_x86 Scalar AVX AVX2 AVX512 Native)
set(VECOPS_ARCH_LIST_ARM Scalar SVE SVE2 Native)
set(VECOPS_ARCH_LIST ${VECOPS_ARCH_LIST_${VECOPS_ARCH_FAMILY}})

function(_vecops_write_if_different PATH CONTENT)
    set(_CURRENT_CONTENT "")
    if(EXISTS "${PATH}")
        file(READ "${PATH}" _CURRENT_CONTENT)
    endif()
    if(NOT "${_CURRENT_CONTENT}" STREQUAL "${CONTENT}")
        file(WRITE "${PATH}" "${CONTENT}")
    endif()
endfunction()

# A source can request parallel template instantiation with either the generic
# marker below or the legacy test marker. The original source remains a small
# registry/main translation unit; generated stubs compile the heavy shards.
#
#   // @vecops-target-shards: 4
#   // @vecops-target-shards-x86: 4
#   // @vecops-target-shards-ARM: 16
#   // @vecops-test-shards: 4   (backward compatible)
function(_vecops_expand_target_source OUT_VAR TARGET_NAME SOURCE)
    if(IS_ABSOLUTE "${SOURCE}")
        cmake_path(NORMAL_PATH SOURCE OUTPUT_VARIABLE _SOURCE)
    else()
        cmake_path(ABSOLUTE_PATH SOURCE
            BASE_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
            NORMALIZE OUTPUT_VARIABLE _SOURCE)
    endif()
    if(NOT EXISTS "${_SOURCE}")
        message(FATAL_ERROR "Target source does not exist: ${SOURCE}")
    endif()

    set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS "${_SOURCE}")
    file(READ "${_SOURCE}" _SOURCE_HEADER LIMIT 1024)
    string(REGEX MATCHALL
        "(^|\n)[ \t]*//[ \t]+@vecops-(target|test)-shards:[ \t]*[1-9][0-9]*[ \t]*(\n|$)"
        _GENERIC_SHARD_MARKERS "${_SOURCE_HEADER}")
    string(REGEX MATCHALL
        "(^|\n)[ \t]*//[ \t]+@vecops-target-shards-(x86|ARM):[ \t]*[1-9][0-9]*[ \t]*(\n|$)"
        _FAMILY_SHARD_MARKERS "${_SOURCE_HEADER}")
    list(LENGTH _GENERIC_SHARD_MARKERS _GENERIC_MARKER_COUNT)
    list(LENGTH _FAMILY_SHARD_MARKERS _FAMILY_MARKER_COUNT)
    if(_GENERIC_MARKER_COUNT GREATER 1 OR _FAMILY_MARKER_COUNT GREATER 2)
        message(FATAL_ERROR "Duplicate vecops shard markers in ${_SOURCE}")
    endif()
    if(_GENERIC_MARKER_COUNT GREATER 0 AND _FAMILY_MARKER_COUNT GREATER 0)
        message(FATAL_ERROR
            "Cannot mix generic and architecture-specific shard markers in ${_SOURCE}")
    endif()
    if(_GENERIC_MARKER_COUNT EQUAL 0 AND _FAMILY_MARKER_COUNT EQUAL 0)
        set(${OUT_VAR} "${_SOURCE}" PARENT_SCOPE)
        return()
    endif()

    if(_GENERIC_MARKER_COUNT EQUAL 1)
        list(GET _GENERIC_SHARD_MARKERS 0 _SHARD_MARKER)
    else()
        set(_SELECTED_FAMILY_MARKERS "")
        foreach(_MARKER IN LISTS _FAMILY_SHARD_MARKERS)
            if(_MARKER MATCHES
               "@vecops-target-shards-${VECOPS_ARCH_FAMILY}:")
                list(APPEND _SELECTED_FAMILY_MARKERS "${_MARKER}")
            endif()
        endforeach()
        list(LENGTH _SELECTED_FAMILY_MARKERS _SELECTED_MARKER_COUNT)
        if(NOT _SELECTED_MARKER_COUNT EQUAL 1)
            message(FATAL_ERROR
                "Expected exactly one ${VECOPS_ARCH_FAMILY} shard marker in ${_SOURCE}")
        endif()
        list(GET _SELECTED_FAMILY_MARKERS 0 _SHARD_MARKER)
    endif()
    string(REGEX REPLACE
        "^.*@vecops-(target|test)-shards(-x86|-ARM)?:[ \t]*([1-9][0-9]*).*$" "\\3"
        _SHARD_COUNT "${_SHARD_MARKER}")
    string(STRIP "${_SHARD_COUNT}" _SHARD_COUNT)
    if(_SHARD_COUNT GREATER 512)
        message(FATAL_ERROR
            "Refusing to generate ${_SHARD_COUNT} shards for ${_SOURCE}; "
            "maximum is 512")
    endif()

    file(RELATIVE_PATH _SOURCE_KEY "${VECOPS_SOURCE_DIR}" "${_SOURCE}")
    if(_SOURCE_KEY MATCHES "^\\.\\.")
        string(SHA1 _SOURCE_KEY "${_SOURCE}")
    else()
        string(REGEX REPLACE "\\.[^.]*$" "" _SOURCE_KEY "${_SOURCE_KEY}")
    endif()
    string(MAKE_C_IDENTIFIER "${TARGET_NAME}" _TARGET_KEY)
    set(_SHARD_DIR
        "${CMAKE_CURRENT_BINARY_DIR}/generated/shards/${_TARGET_KEY}/${_SOURCE_KEY}")
    file(MAKE_DIRECTORY "${_SHARD_DIR}")

    set(_EXPANDED_SOURCES "${_SOURCE}")
    set(_LEGACY_TEST_DEFINITIONS "")
    if(_SHARD_MARKER MATCHES "@vecops-test-shards")
        set(_LEGACY_TEST_DEFINITIONS
"#define VECOPS_TEST_SHARD_ACTIVE 1
#define VECOPS_TEST_SHARD_INDEX @SHARD_INDEX@
#define VECOPS_TEST_SHARD_COUNT ${_SHARD_COUNT}
")
    endif()
    math(EXPR _LAST_SHARD "${_SHARD_COUNT} - 1")
    foreach(_SHARD_INDEX RANGE 0 ${_LAST_SHARD})
        set(_STUB "${_SHARD_DIR}/shard_${_SHARD_INDEX}.cpp")
        string(REPLACE "@SHARD_INDEX@" "${_SHARD_INDEX}"
            _SHARD_LEGACY_DEFINITIONS "${_LEGACY_TEST_DEFINITIONS}")
        set(_STUB_CONTENT
"#define VECOPS_TARGET_SHARD_ACTIVE 1
#define VECOPS_TARGET_SHARD_INDEX ${_SHARD_INDEX}
#define VECOPS_TARGET_SHARD_COUNT ${_SHARD_COUNT}
${_SHARD_LEGACY_DEFINITIONS}
#include \"${_SOURCE}\"
")
        _vecops_write_if_different("${_STUB}" "${_STUB_CONTENT}")
        list(APPEND _EXPANDED_SOURCES "${_STUB}")
    endforeach()
    set(${OUT_VAR} "${_EXPANDED_SOURCES}" PARENT_SCOPE)
endfunction()

# Return one ISA-specific archive containing the selected non-header SME matmul
# leaves. The archive is shared by targets with the same -march and feature
# set. Each leaf remains a separate object, so a static linker extracts only
# the objects referenced by a consumer's header instantiations. Target ISA
# features are the compile-time availability contract: every multiarch target
# that can instantiate one of these paths is linked to the matching archive.
function(_vecops_get_sme_matmul_leaf_library
        OUT_VAR MARCH USE_F64 USE_RUNTIME_QUANT_INT8 USE_MIXED_SIGN_SKINNY)
    set(_LEAF_KEY
        "${MARCH};f64=${USE_F64};runtime_i8=${USE_RUNTIME_QUANT_INT8};mixed_sign_skinny=${USE_MIXED_SIGN_SKINNY}")
    string(SHA1 _MARCH_HASH "${_LEAF_KEY}")
    string(SUBSTRING "${_MARCH_HASH}" 0 12 _MARCH_ID)
    set(_LEAF_TARGET "vecops_sme_matmul_leaves_${_MARCH_ID}")
    if(NOT TARGET ${_LEAF_TARGET})
        set(_LEAF_SOURCES "")
        if(USE_F64)
            list(APPEND _LEAF_SOURCES
                "${VECOPS_SOURCE_DIR}/src/arch/sme/FusedSkinnyF64.cpp")
        endif()
        if(USE_RUNTIME_QUANT_INT8)
            list(APPEND _LEAF_SOURCES
                "${VECOPS_SOURCE_DIR}/src/arch/sme/RuntimeQuantInt8.cpp")
        endif()
        if(USE_MIXED_SIGN_SKINNY)
            list(APPEND _LEAF_SOURCES
                "${VECOPS_SOURCE_DIR}/src/arch/sme/MixedSignSkinnyInt8.cpp")
        endif()
        if(NOT _LEAF_SOURCES)
            message(FATAL_ERROR
                "SME matmul leaf library requested without any leaves")
        endif()
        add_library(${_LEAF_TARGET} STATIC ${_LEAF_SOURCES})
        target_compile_features(${_LEAF_TARGET} PRIVATE cxx_std_20)
        target_include_directories(${_LEAF_TARGET} PRIVATE
            "${VECOPS_SOURCE_DIR}/include"
            "${VECOPS_SOURCE_DIR}/include/vecops")
        target_compile_definitions(${_LEAF_TARGET} PRIVATE
            "$<$<CONFIG:Debug>:VECOPS_DEBUG>")
        if(VECOPS_PRESERVE_SUBNORMALS)
            target_compile_definitions(${_LEAF_TARGET} PRIVATE
                VECOPS_PRESERVE_SUBNORMALS=1)
        endif()
        target_compile_options(${_LEAF_TARGET} PRIVATE
            "-march=${MARCH}" "-Wno-ignored-attributes")
        if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
            target_compile_options(${_LEAF_TARGET} PRIVATE "-Wno-psabi")
        endif()
        target_link_libraries(${_LEAF_TARGET} PRIVATE
            vecops_optimize_for_kernels)
        set_property(TARGET ${_LEAF_TARGET} PROPERTY FOLDER
            "internal/sme_matmul")
    endif()
    set(${OUT_VAR} "${_LEAF_TARGET}" PARENT_SCOPE)
endfunction()

# Public primitive used by vecops_add_test and vecops_add_benchmark. It owns
# source sharding and all architecture-specific compile/link policy.
function(vecops_add_multiarch_executable)
    set(options FIXED_SVE FIXED_STREAMING_SVE)
    set(oneValueArgs NAME FOLDER OUT_TARGETS ARCH_DEFINITION)
    set(multiValueArgs
        FILES ARCH DEFINITIONS LIBRARIES INCLUDE_DIRECTORIES
        COMPILE_OPTIONS LINK_OPTIONS)
    cmake_parse_arguments(ARG "${options}" "${oneValueArgs}"
        "${multiValueArgs}" ${ARGN})
    if(NOT ARG_NAME)
        message(FATAL_ERROR "vecops_add_multiarch_executable requires NAME")
    endif()
    if(NOT ARG_FILES)
        message(FATAL_ERROR
            "vecops_add_multiarch_executable(${ARG_NAME}) requires FILES")
    endif()

    set(_SOURCES "")
    foreach(_FILE IN LISTS ARG_FILES)
        _vecops_expand_target_source(
            _EXPANDED_SOURCE "${ARG_NAME}" "${_FILE}")
        list(APPEND _SOURCES ${_EXPANDED_SOURCE})
    endforeach()

    if(ARG_FOLDER)
        set(_FOLDER "${ARG_FOLDER}")
    else()
        list(GET ARG_FILES 0 _PRIMARY_SOURCE)
        if(IS_ABSOLUTE "${_PRIMARY_SOURCE}")
            file(RELATIVE_PATH _PRIMARY_SOURCE
                "${CMAKE_CURRENT_SOURCE_DIR}" "${_PRIMARY_SOURCE}")
        endif()
        get_filename_component(_PRIMARY_DIR "${_PRIMARY_SOURCE}" DIRECTORY)
        get_filename_component(_PRIMARY_NAME "${_PRIMARY_SOURCE}" NAME_WE)
        if(_PRIMARY_DIR)
            set(_FOLDER "${_PRIMARY_DIR}/${_PRIMARY_NAME}")
        else()
            set(_FOLDER "${_PRIMARY_NAME}")
        endif()
    endif()

    if(ARG_ARCH)
        set(_ARCHES ${ARG_ARCH})
    else()
        set(_ARCHES NONE)
    endif()

    set(_CREATED_TARGETS "")
    foreach(_ARCH IN LISTS _ARCHES)
        if(_ARCH STREQUAL "NONE")
            set(_TARGET_NAME "${ARG_NAME}")
            set(_MARCH "")
        else()
            set(_MARCH "${VECOPS_MAP_${VECOPS_ARCH_FAMILY}_${_ARCH}}")
            if(NOT _MARCH)
                message(STATUS
                    "Skipping arch '${_ARCH}' - not defined for "
                    "${VECOPS_ARCH_FAMILY}")
                continue()
            endif()
            set(_TARGET_NAME "${ARG_NAME}-${_ARCH}")
        endif()

        add_executable(${_TARGET_NAME} ${_SOURCES})
        list(APPEND _CREATED_TARGETS ${_TARGET_NAME})
        set_property(TARGET ${_TARGET_NAME} PROPERTY FOLDER "${_FOLDER}")
        if(NOT _ARCH STREQUAL "NONE")
            vecops_configure_target_arch(
                TARGET ${_TARGET_NAME}
                ARCH ${_ARCH}
                OUT_MARCH _MARCH)
        endif()
        if(ARG_INCLUDE_DIRECTORIES)
            target_include_directories(${_TARGET_NAME} PRIVATE
                ${ARG_INCLUDE_DIRECTORIES})
        endif()
        if(ARG_LIBRARIES)
            target_link_libraries(${_TARGET_NAME} PRIVATE ${ARG_LIBRARIES})
        endif()
        if(ARG_COMPILE_OPTIONS)
            target_compile_options(${_TARGET_NAME} PRIVATE
                ${ARG_COMPILE_OPTIONS})
        endif()
        if(ARG_LINK_OPTIONS)
            target_link_options(${_TARGET_NAME} PRIVATE ${ARG_LINK_OPTIONS})
        endif()

        # ACLE currently has no portable FEAT_SME_FA64 feature-test macro.
        # Propagate the CMake target guarantee so streaming kernels can select
        # ordinary SVE gather/scatter only when Full A64 is available.
        set(_VECOPS_TARGET_HAS_SME_FA64 OFF)
        if(VECOPS_ARCH_FAMILY STREQUAL "ARM" AND
           ((_MARCH MATCHES "(^|\\+)sme-fa64($|\\+)") OR
            (VECOPS_NATIVE_HAS_SME_FA64 AND
             ((_ARCH STREQUAL "Native") OR
              (_ARCH STREQUAL "NativeFixedSVE") OR
              (_ARCH STREQUAL "NativeFixedStreamingSVE")))))
            set(_VECOPS_TARGET_HAS_SME_FA64 ON)
            target_compile_definitions(${_TARGET_NAME} PRIVATE
                VECOPS_TARGET_SME_FA64=1)
        endif()
        set(_VECOPS_TARGET_HAS_SME OFF)
        if(VECOPS_ARCH_FAMILY STREQUAL "ARM" AND
           ((_MARCH MATCHES "(^|\\+)sme($|\\+)") OR
            (VECOPS_NATIVE_HAS_SME AND
             ((_ARCH STREQUAL "Native") OR
              (_ARCH STREQUAL "NativeFixedSVE") OR
              (_ARCH STREQUAL "NativeFixedStreamingSVE")))))
            set(_VECOPS_TARGET_HAS_SME ON)
        endif()
        set(_VECOPS_TARGET_HAS_I8MM OFF)
        if(VECOPS_ARCH_FAMILY STREQUAL "ARM" AND
           ((_MARCH MATCHES "(^|\\+)i8mm($|\\+)") OR
            (VECOPS_NATIVE_HAS_I8MM AND
             ((_ARCH STREQUAL "Native") OR
              (_ARCH STREQUAL "NativeFixedSVE") OR
              (_ARCH STREQUAL "NativeFixedStreamingSVE")))))
            set(_VECOPS_TARGET_HAS_I8MM ON)
        endif()
        set(_VECOPS_USE_SME_F64_LEAF OFF)
        if(_VECOPS_TARGET_HAS_SME AND
           ((_MARCH MATCHES "(^|\\+)sme-f64f64($|\\+)") OR
            (VECOPS_NATIVE_HAS_SME_F64F64 AND
             ((_ARCH STREQUAL "Native") OR
              (_ARCH STREQUAL "NativeFixedSVE") OR
              (_ARCH STREQUAL "NativeFixedStreamingSVE")))))
            set(_VECOPS_USE_SME_F64_LEAF ON)
        endif()

        set(_VECOPS_USE_SME_RUNTIME_QUANT_INT8_LEAF OFF)
        if(_VECOPS_TARGET_HAS_SME AND
           _VECOPS_TARGET_HAS_I8MM)
            set(_VECOPS_USE_SME_RUNTIME_QUANT_INT8_LEAF ON)
        endif()

        set(_VECOPS_USE_SME_MIXED_SIGN_SKINNY_LEAF OFF)
        if(_VECOPS_TARGET_HAS_SME AND
           _VECOPS_TARGET_HAS_I8MM)
            set(_VECOPS_USE_SME_MIXED_SIGN_SKINNY_LEAF ON)
        endif()

        if(_VECOPS_TARGET_HAS_SME AND VECOPS_ENABLE_KUPL_MMA)
            target_compile_definitions(${_TARGET_NAME} PRIVATE
                VECOPS_HAS_KUPL_MMA=1)
        endif()

        if(_VECOPS_USE_SME_F64_LEAF OR
           _VECOPS_USE_SME_RUNTIME_QUANT_INT8_LEAF OR
           _VECOPS_USE_SME_MIXED_SIGN_SKINNY_LEAF)
            _vecops_get_sme_matmul_leaf_library(
                _VECOPS_SME_MATMUL_LEAF_LIBRARY "${_MARCH}"
                "${_VECOPS_USE_SME_F64_LEAF}"
                "${_VECOPS_USE_SME_RUNTIME_QUANT_INT8_LEAF}"
                "${_VECOPS_USE_SME_MIXED_SIGN_SKINNY_LEAF}")
            target_link_libraries(${_TARGET_NAME} PRIVATE
                ${_VECOPS_SME_MATMUL_LEAF_LIBRARY})
        endif()
        if(_VECOPS_TARGET_HAS_SME AND
           VECOPS_NATIVE_FIXED_STREAMING_SVE_BITS AND
           ((_ARCH STREQUAL "NativeFixedStreamingSVE") OR
            ARG_FIXED_STREAMING_SVE))
            target_compile_definitions(${_TARGET_NAME} PRIVATE
                "VECOPS_TARGET_FIXED_STREAMING_SVE_BITS=${VECOPS_NATIVE_FIXED_STREAMING_SVE_BITS}")
            if(VECOPS_COMPILER_ACCEPTS_FIXED_STREAMING_SVE_BITS)
                target_compile_options(${_TARGET_NAME} PRIVATE
                    "-msve-streaming-vector-bits=${VECOPS_NATIVE_FIXED_STREAMING_SVE_BITS}")
            endif()
        endif()
        if(VECOPS_ARCH_FAMILY STREQUAL "ARM" AND
           CMAKE_CXX_COMPILER_ID STREQUAL "GNU" AND _MARCH AND
           NOT _ARCH STREQUAL "Scalar")
            target_compile_options(${_TARGET_NAME} PRIVATE "-Wno-psabi")
        endif()
        if(VECOPS_ARCH_FAMILY STREQUAL "ARM" AND
           ((_ARCH STREQUAL "NativeFixedSVE") OR
            (ARG_FIXED_SVE AND _ARCH MATCHES "^SVE")) AND
           VECOPS_NATIVE_FIXED_SVE_BITS)
            target_compile_options(${_TARGET_NAME} PRIVATE
                "-msve-vector-bits=${VECOPS_NATIVE_FIXED_SVE_BITS}")
        endif()
        if(ARG_DEFINITIONS)
            target_compile_definitions(${_TARGET_NAME} PRIVATE
                ${ARG_DEFINITIONS})
        endif()
        if(ARG_ARCH_DEFINITION)
            target_compile_definitions(${_TARGET_NAME} PRIVATE
                "${ARG_ARCH_DEFINITION}=\"${_ARCH}\"")
        endif()
    endforeach()

    if(ARG_OUT_TARGETS)
        set(${ARG_OUT_TARGETS} "${_CREATED_TARGETS}" PARENT_SCOPE)
    endif()
endfunction()
