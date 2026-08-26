include_guard(GLOBAL)

include("${CMAKE_CURRENT_LIST_DIR}/../../cmake/VecopsMultiarchTargets.cmake")

file(GLOB _GDB_SCRIPTS "${CMAKE_SOURCE_DIR}/scripts/*.py")

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

    if(ARG_ARCH)
        set(_ARCHES ${ARG_ARCH})
    else()
        set(_ARCHES NONE)
    endif()

    vecops_add_multiarch_executable(
        NAME ${ARG_NAME}
        FILES ${ARG_FILES}
        FOLDER ${ARG_FOLDER}
        ARCH ${_ARCHES}
        DEFINITIONS ${ARG_DEFINITIONS}
        LIBRARIES vecops gtest gtest_main ${ARG_LIBRARIES}
        INCLUDE_DIRECTORIES "${CMAKE_CURRENT_SOURCE_DIR}"
        OUT_TARGETS _TARGETS)

    foreach(_TARGET_NAME IN LISTS _TARGETS)

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
