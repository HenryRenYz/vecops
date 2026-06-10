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

# ARM Native: starts from native, conditionally appends +bf16 / +sme
# because some compilers (e.g. BiSheng) do not auto-enable them with -march=native
set(VECOPS_MAP_ARM_Native "native")
if(VECOPS_ARCH_FAMILY STREQUAL "ARM" AND EXISTS "/proc/cpuinfo")
    file(READ "/proc/cpuinfo" _CPUINFO)
    string(REGEX MATCH "Features[ \t]*:.*" _FEAT_LINE "${_CPUINFO}")
    foreach(_FEAT IN ITEMS bf16 sme)
        string(FIND "${_FEAT_LINE}" " ${_FEAT} " _POS)
        if(NOT _POS EQUAL -1)
            string(APPEND VECOPS_MAP_ARM_Native "+${_FEAT}")
        endif()
    endforeach()
    unset(_CPUINFO)
    unset(_FEAT_LINE)
endif()

# ── Default Arch Code Lists (full sequence per family) ──────
set(VECOPS_ARCH_LIST_x86 Scalar AVX AVX2 AVX512 Native)
set(VECOPS_ARCH_LIST_ARM Scalar NEON SVE SVE2 Native)
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
    set(multiValueArgs FILES ARCH)
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
        target_link_libraries(${TARGET_NAME} PRIVATE vecops gtest gtest_main)
        gtest_discover_tests(${TARGET_NAME})

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

add_executable(ArrayTest ${CMAKE_CURRENT_SOURCE_DIR}/array/ArrayTest.cpp)
target_link_libraries(ArrayTest PRIVATE vecops gtest gtest_main)
gtest_discover_tests(ArrayTest)

add_executable(GemmTest ${CMAKE_CURRENT_SOURCE_DIR}/array/GemmTest.cpp)
target_link_libraries(GemmTest PRIVATE vecops gtest gtest_main)
gtest_discover_tests(GemmTest)

add_executable(LayoutTest ${CMAKE_CURRENT_SOURCE_DIR}/gemm/LayoutTest.cpp)
target_link_libraries(LayoutTest PRIVATE vecops gtest gtest_main)
gtest_discover_tests(LayoutTest)

add_multiarch_executable(VecBaseTest
    FILES ${CMAKE_CURRENT_SOURCE_DIR}/vec/BaseTest.cpp
    ARCH Scalar
)

set(VECOPS_VEC_TESTS BasicTest LoadStoreTest ArithTest ConversionTest ShuffleTest)
foreach(TEST ${VECOPS_VEC_TESTS})
    add_multiarch_executable(Vec${TEST}
        FILES ${CMAKE_CURRENT_SOURCE_DIR}/vec/${TEST}.cpp
        ARCH ${VECOPS_ARCH_LIST}
    )
endforeach()
