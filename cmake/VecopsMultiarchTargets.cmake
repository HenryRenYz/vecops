include_guard(GLOBAL)

include(CheckCXXCompilerFlag)

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
if(VECOPS_ARCH_FAMILY STREQUAL "ARM" AND EXISTS "/proc/cpuinfo")
    file(READ "/proc/cpuinfo" _VECOPS_CPUINFO)
    string(REGEX MATCH "Features[ \t]*:.*" _VECOPS_FEAT_LINE
        "${_VECOPS_CPUINFO}")
    foreach(_VECOPS_FEAT IN ITEMS sve2p1 bf16 sme)
        string(FIND "${_VECOPS_FEAT_LINE}" " ${_VECOPS_FEAT} " _VECOPS_POS)
        if(NOT _VECOPS_POS EQUAL -1)
            string(APPEND VECOPS_MAP_ARM_Native "+${_VECOPS_FEAT}")
        endif()
    endforeach()
    string(FIND "${_VECOPS_FEAT_LINE}" " smef64f64 " _VECOPS_SME_F64_POS)
    if(NOT _VECOPS_SME_F64_POS EQUAL -1)
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
        _SHARD_MARKERS "${_SOURCE_HEADER}")
    list(LENGTH _SHARD_MARKERS _MARKER_COUNT)
    if(_MARKER_COUNT GREATER 1)
        message(FATAL_ERROR "Multiple vecops shard markers in ${_SOURCE}")
    endif()
    if(_MARKER_COUNT EQUAL 0)
        set(${OUT_VAR} "${_SOURCE}" PARENT_SCOPE)
        return()
    endif()

    list(GET _SHARD_MARKERS 0 _SHARD_MARKER)
    string(REGEX REPLACE
        "^.*@vecops-(target|test)-shards:[ \t]*([1-9][0-9]*).*$" "\\2"
        _SHARD_COUNT "${_SHARD_MARKER}")
    string(STRIP "${_SHARD_COUNT}" _SHARD_COUNT)
    if(_SHARD_COUNT GREATER 512)
        message(FATAL_ERROR
            "Refusing to generate ${_SHARD_COUNT} shards for ${_SOURCE}; "
            "maximum is 512")
    endif()

    file(RELATIVE_PATH _SOURCE_KEY "${CMAKE_SOURCE_DIR}" "${_SOURCE}")
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

# Public primitive used by vecops_add_test and vecops_add_benchmark. It owns
# source sharding and all architecture-specific compile/link policy.
function(vecops_add_multiarch_executable)
    set(options FIXED_SVE)
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

        if(_MARCH)
            target_compile_options(${_TARGET_NAME} PRIVATE "-march=${_MARCH}")
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
        # SME ACLE private-ZA functions use AAPCS64 compiler-rt helpers.
        if(VECOPS_ARCH_FAMILY STREQUAL "ARM" AND
           CMAKE_CXX_COMPILER_ID MATCHES "Clang" AND
           _MARCH MATCHES "(^|\\+)sme($|\\+)")
            target_compile_options(${_TARGET_NAME} PRIVATE
                "-Wno-aarch64-sme-attributes")
            target_link_options(${_TARGET_NAME} PRIVATE
                "-rtlib=compiler-rt" "-unwindlib=libgcc")
        endif()
        if(ARG_DEFINITIONS)
            target_compile_definitions(${_TARGET_NAME} PRIVATE
                ${ARG_DEFINITIONS})
        endif()
        if(ARG_ARCH_DEFINITION)
            target_compile_definitions(${_TARGET_NAME} PRIVATE
                "${ARG_ARCH_DEFINITION}=\"${_ARCH}\"")
        endif()
        if(_ARCH STREQUAL "Scalar")
            target_compile_definitions(${_TARGET_NAME} PRIVATE
                CPU_CAPABILITY=GENERIC
                CPU_CAPABILITY_GENERIC=1)
        endif()
    endforeach()

    if(ARG_OUT_TARGETS)
        set(${ARG_OUT_TARGETS} "${_CREATED_TARGETS}" PARENT_SCOPE)
    endif()
endfunction()
