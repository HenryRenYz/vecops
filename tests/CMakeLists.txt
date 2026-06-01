
# Discover all GDB pretty-printer Python modules in scripts/
# (evaluated at configure time — re-run cmake if you add new .py files)
file(GLOB _GDB_SCRIPTS "${CMAKE_SOURCE_DIR}/scripts/*.py")

function(add_multiarch_executable NAME)
    set(options "")
    set(oneValueArgs "")
    set(multiValueArgs FILES ARCH)
    cmake_parse_arguments(ARG "${options}" "${oneValueArgs}" "${multiValueArgs}" ${ARGN})

    # Default to native architecture
    if(NOT ARG_ARCH)
        set(ARG_ARCH "native")
    endif()

    foreach(ARCH ${ARG_ARCH})
        set(TARGET_NAME "${NAME}-${ARCH}")
        add_executable(${TARGET_NAME} ${ARG_FILES})
        target_compile_options(${TARGET_NAME} PRIVATE -march=${ARCH})
        target_link_libraries(${TARGET_NAME} PRIVATE vecops gtest gtest_main)
        gtest_discover_tests(${TARGET_NAME})

        # --- GDB pretty-printer auto-load: generate <target>-gdb.py ---
        list(LENGTH _GDB_SCRIPTS _NUM_GDB_SCRIPTS)
        if(_NUM_GDB_SCRIPTS LESS_EQUAL 0)
            continue()
        endif()

        set(_IMPORTS "")
        foreach(_MOD ${_GDB_SCRIPTS})
            get_filename_component(_MNAME "${_MOD}" NAME_WE)
            string(APPEND _IMPORTS "try:\n    import ${_MNAME}\nexcept Exception:\n    pass\n")
        endforeach()

        # Write a loader script that imports every module in scripts/
        set(_LOADER "${CMAKE_CURRENT_BINARY_DIR}/${TARGET_NAME}-gdb.py")
        file(WRITE "${_LOADER}"
"import sys
sys.path.insert(0, '${CMAKE_SOURCE_DIR}/scripts')

${_IMPORTS}")

        # Copy the loader next to the executable after each build
        add_custom_command(TARGET ${TARGET_NAME} POST_BUILD
            COMMAND ${CMAKE_COMMAND} -E copy_if_different
                "${_LOADER}"
                "$<TARGET_FILE_DIR:${TARGET_NAME}>/${TARGET_NAME}-gdb.py"
            COMMENT "Installing GDB pretty-printer for ${TARGET_NAME}"
        )
    endforeach()
endfunction()


# vecops-ArrayTest
add_executable(vecops-ArrayTest ${CMAKE_CURRENT_SOURCE_DIR}/array/ArrayTest.cpp)
target_link_libraries(vecops-ArrayTest PRIVATE vecops gtest gtest_main)
gtest_discover_tests(vecops-ArrayTest)

# vecops-GemmTest
add_executable(vecops-GemmTest ${CMAKE_CURRENT_SOURCE_DIR}/array/GemmTest.cpp)
target_link_libraries(vecops-GemmTest PRIVATE vecops gtest gtest_main)
gtest_discover_tests(vecops-GemmTest)

# vecops-VecBaseTest
add_multiarch_executable(vecops-VecBaseTest
        FILES ${CMAKE_CURRENT_SOURCE_DIR}/vec/BaseTest.cpp
        ARCH x86-64
)

# vecops-VecScalarTest
add_multiarch_executable(vecops-VecScalarTest
        FILES ${CMAKE_CURRENT_SOURCE_DIR}/vec/ScalarTest.cpp
        ARCH x86-64
)

# vecops-VecBasicTest (word-level basic operations: fill, get/set, blend)
add_multiarch_executable(vecops-VecBasicTest
        FILES ${CMAKE_CURRENT_SOURCE_DIR}/vec/BasicTest.cpp
        ARCH x86-64 corei7-avx core-avx2 native
)
# vecops-VecLoadStoreTest
add_multiarch_executable(vecops-VecLoadStoreTest
        FILES ${CMAKE_CURRENT_SOURCE_DIR}/vec/LoadStoreTest.cpp
        ARCH x86-64 corei7-avx core-avx2 native
)
# vecops-VecArithTest
add_multiarch_executable(vecops-VecArithTest
        FILES ${CMAKE_CURRENT_SOURCE_DIR}/vec/ArithTest.cpp
        ARCH x86-64 corei7-avx core-avx2 native
)
# vecops-VecConversionTest
add_multiarch_executable(vecops-VecConversionTest
        FILES ${CMAKE_CURRENT_SOURCE_DIR}/vec/ConversionTest.cpp
        ARCH x86-64 corei7-avx core-avx2 native
)
# vecops-VecShuffleTest
add_multiarch_executable(vecops-VecShuffleTest
        FILES ${CMAKE_CURRENT_SOURCE_DIR}/vec/ShuffleTest.cpp
        ARCH x86-64 corei7-avx core-avx2 native
)

# vecops-VecSVETest (SVE type infrastructure test)
add_multiarch_executable(vecops-VecSVETest
        FILES ${CMAKE_CURRENT_SOURCE_DIR}/vec/SVETest.cpp
        ARCH native
)
