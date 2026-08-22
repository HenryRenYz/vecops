include_guard(GLOBAL)

# Platform detection and architecture metadata used by test targets.
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
    file(READ "/proc/cpuinfo" _CPUINFO)
    string(REGEX MATCH "Features[ \t]*:.*" _FEAT_LINE "${_CPUINFO}")
    foreach(_FEAT IN ITEMS sve2p1 bf16 sme)
        string(FIND "${_FEAT_LINE}" " ${_FEAT} " _POS)
        if(NOT _POS EQUAL -1)
            string(APPEND VECOPS_MAP_ARM_Native "+${_FEAT}")
        endif()
    endforeach()

    include(CheckCXXCompilerFlag)
    check_cxx_compiler_flag(
        "-march=${VECOPS_MAP_ARM_Native}"
        VECOPS_COMPILER_ACCEPTS_ARM_NATIVE_FEATURES)
    if(NOT VECOPS_COMPILER_ACCEPTS_ARM_NATIVE_FEATURES)
        message(STATUS
            "Compiler rejects -march=${VECOPS_MAP_ARM_Native}; "
            "falling back to -march=native")
        set(VECOPS_MAP_ARM_Native "native")
    endif()
    unset(_CPUINFO)
    unset(_FEAT_LINE)
endif()

# NativeFixedSVE uses the native ISA feature set while making SVE types sized.
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
            RUN_OUTPUT_VARIABLE _SVE_VL_OUTPUT)
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

set(VECOPS_ARCH_LIST_x86 Scalar AVX AVX2 AVX512 Native)
set(VECOPS_ARCH_LIST_ARM Scalar SVE SVE2 Native)
set(VECOPS_ARCH_LIST ${VECOPS_ARCH_LIST_${VECOPS_ARCH_FAMILY}})

file(GLOB _GDB_SCRIPTS "${CMAKE_SOURCE_DIR}/scripts/*.py")

function(_vecops_write_if_different PATH CONTENT)
    set(_CURRENT_CONTENT "")
    if(EXISTS "${PATH}")
        file(READ "${PATH}" _CURRENT_CONTENT)
    endif()
    if(NOT "${_CURRENT_CONTENT}" STREQUAL "${CONTENT}")
        file(WRITE "${PATH}" "${CONTENT}")
    endif()
endfunction()

function(_vecops_expand_test_source OUT_VAR SOURCE)
    if(IS_ABSOLUTE "${SOURCE}")
        cmake_path(NORMAL_PATH SOURCE OUTPUT_VARIABLE _SOURCE)
    else()
        cmake_path(ABSOLUTE_PATH SOURCE
            BASE_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
            NORMALIZE OUTPUT_VARIABLE _SOURCE)
    endif()
    if(NOT EXISTS "${_SOURCE}")
        message(FATAL_ERROR "Test source does not exist: ${SOURCE}")
    endif()

    set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS "${_SOURCE}")
    file(READ "${_SOURCE}" _SOURCE_HEADER LIMIT 1024)
    string(REGEX MATCHALL
        "(^|\n)[ \t]*//[ \t]+@vecops-test-shards:[ \t]*[1-9][0-9]*[ \t]*(\n|$)"
        _SHARD_MARKERS "${_SOURCE_HEADER}")
    list(LENGTH _SHARD_MARKERS _MARKER_COUNT)
    if(_MARKER_COUNT GREATER 1)
        message(FATAL_ERROR "Multiple @vecops-test-shards markers in ${_SOURCE}")
    endif()
    if(_MARKER_COUNT EQUAL 0)
        set(${OUT_VAR} "${_SOURCE}" PARENT_SCOPE)
        return()
    endif()

    list(GET _SHARD_MARKERS 0 _SHARD_MARKER)
    string(REGEX REPLACE
        "^.*@vecops-test-shards:[ \t]*([1-9][0-9]*).*$" "\\1"
        _SHARD_COUNT "${_SHARD_MARKER}")
    string(STRIP "${_SHARD_COUNT}" _SHARD_COUNT)
    if(_SHARD_COUNT GREATER 512)
        message(FATAL_ERROR
            "Refusing to generate ${_SHARD_COUNT} test shards for ${_SOURCE}; maximum is 512")
    endif()

    file(RELATIVE_PATH _SOURCE_KEY "${CMAKE_CURRENT_SOURCE_DIR}" "${_SOURCE}")
    if(_SOURCE_KEY MATCHES "^\\.\\.")
        string(SHA1 _SOURCE_KEY "${_SOURCE}")
    else()
        string(REGEX REPLACE "\\.[^.]*$" "" _SOURCE_KEY "${_SOURCE_KEY}")
    endif()
    set(_SHARD_DIR "${CMAKE_CURRENT_BINARY_DIR}/generated/shards/${_SOURCE_KEY}")
    file(MAKE_DIRECTORY "${_SHARD_DIR}")

    # Compile the original file once as the lightweight GoogleTest registry.
    # Generated include stubs compile only one heavy template specialization.
    set(_EXPANDED_SOURCES "${_SOURCE}")
    math(EXPR _LAST_SHARD "${_SHARD_COUNT} - 1")
    foreach(_SHARD_INDEX RANGE 0 ${_LAST_SHARD})
        set(_STUB "${_SHARD_DIR}/shard_${_SHARD_INDEX}.cpp")
        set(_STUB_CONTENT
"#define VECOPS_TEST_SHARD_ACTIVE 1
#define VECOPS_TEST_SHARD_INDEX ${_SHARD_INDEX}
#define VECOPS_TEST_SHARD_COUNT ${_SHARD_COUNT}
#include \"${_SOURCE}\"
")
        _vecops_write_if_different("${_STUB}" "${_STUB_CONTENT}")
        list(APPEND _EXPANDED_SOURCES "${_STUB}")
    endforeach()
    set(${OUT_VAR} "${_EXPANDED_SOURCES}" PARENT_SCOPE)
endfunction()

function(vecops_add_test)
    set(options "")
    set(oneValueArgs NAME FOLDER)
    set(multiValueArgs FILES ARCH DEFINITIONS LABELS LIBRARIES)
    cmake_parse_arguments(ARG "${options}" "${oneValueArgs}" "${multiValueArgs}" ${ARGN})
    if(NOT ARG_NAME)
        message(FATAL_ERROR "vecops_add_test requires NAME")
    endif()
    if(NOT ARG_FILES)
        message(FATAL_ERROR "vecops_add_test(${ARG_NAME}) requires FILES")
    endif()

    set(_SOURCES "")
    foreach(_FILE IN LISTS ARG_FILES)
        _vecops_expand_test_source(_EXPANDED_SOURCE "${_FILE}")
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

    foreach(_ARCH IN LISTS _ARCHES)
        if(_ARCH STREQUAL "NONE")
            set(_TARGET_NAME "${ARG_NAME}")
        else()
            set(_MARCH "${VECOPS_MAP_${VECOPS_ARCH_FAMILY}_${_ARCH}}")
            if(NOT _MARCH)
                message(STATUS
                    "Skipping arch '${_ARCH}' — not defined for ${VECOPS_ARCH_FAMILY}")
                continue()
            endif()
            set(_TARGET_NAME "${ARG_NAME}-${_ARCH}")
        endif()

        add_executable(${_TARGET_NAME} ${_SOURCES})
        set_property(TARGET ${_TARGET_NAME} PROPERTY FOLDER "${_FOLDER}")
        target_include_directories(${_TARGET_NAME} PRIVATE
            "${CMAKE_CURRENT_SOURCE_DIR}")
        target_link_libraries(${_TARGET_NAME} PRIVATE
            vecops gtest gtest_main ${ARG_LIBRARIES})

        if(NOT _ARCH STREQUAL "NONE")
            target_compile_options(${_TARGET_NAME} PRIVATE "-march=${_MARCH}")
        endif()
        if(VECOPS_ARCH_FAMILY STREQUAL "ARM" AND
           CMAKE_CXX_COMPILER_ID STREQUAL "GNU" AND
           NOT _ARCH STREQUAL "NONE" AND
           NOT _ARCH STREQUAL "Scalar")
            target_compile_options(${_TARGET_NAME} PRIVATE "-Wno-psabi")
        endif()
        if(VECOPS_ARCH_FAMILY STREQUAL "ARM" AND
           _ARCH STREQUAL "NativeFixedSVE")
            target_compile_options(${_TARGET_NAME} PRIVATE
                "-msve-vector-bits=${VECOPS_NATIVE_FIXED_SVE_BITS}")
        endif()
        if(ARG_DEFINITIONS)
            target_compile_definitions(${_TARGET_NAME} PRIVATE ${ARG_DEFINITIONS})
        endif()
        if(_ARCH STREQUAL "Scalar")
            target_compile_definitions(${_TARGET_NAME} PRIVATE
                CPU_CAPABILITY=GENERIC
                CPU_CAPABILITY_GENERIC=1)
        endif()

        if(ARG_LABELS)
            string(JOIN "+" _GTEST_LABELS ${ARG_LABELS})
            gtest_discover_tests(${_TARGET_NAME}
                PROPERTIES LABELS "${_GTEST_LABELS}")
        else()
            gtest_discover_tests(${_TARGET_NAME})
        endif()

        list(LENGTH _GDB_SCRIPTS _NUM_GDB_SCRIPTS)
        if(_NUM_GDB_SCRIPTS GREATER 0)
            set(_IMPORTS "")
            foreach(_MOD IN LISTS _GDB_SCRIPTS)
                get_filename_component(_MNAME "${_MOD}" NAME_WE)
                string(APPEND _IMPORTS
                    "try:\n    import ${_MNAME}\nexcept Exception:\n    pass\n")
            endforeach()
            set(_LOADER "${CMAKE_CURRENT_BINARY_DIR}/${_TARGET_NAME}-gdb.py")
            _vecops_write_if_different("${_LOADER}"
"import sys
sys.path.insert(0, '${CMAKE_SOURCE_DIR}/scripts')

${_IMPORTS}")
            add_custom_command(TARGET ${_TARGET_NAME} POST_BUILD
                COMMAND ${CMAKE_COMMAND} -E copy_if_different
                    "${_LOADER}"
                    "$<TARGET_FILE_DIR:${_TARGET_NAME}>/${_TARGET_NAME}-gdb.py"
                COMMENT "Installing GDB pretty-printer for ${_TARGET_NAME}")
        endif()
    endforeach()
endfunction()
