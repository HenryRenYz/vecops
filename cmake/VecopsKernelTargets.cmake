include_guard(GLOBAL)
include(GNUInstallDirs)
include("${CMAKE_CURRENT_LIST_DIR}/VecopsTargetArch.cmake")

# Add one loadable Vecops kernel module.  This deliberately contains only
# target construction policy: schema generation, manifests, caching, and
# runtime registration live above the CMake SDK boundary.
function(vecops_add_kernel_library)
    set(options NO_INSTALL NO_RUNTIME)
    set(oneValueArgs NAME OUTPUT_NAME INSTALL_DESTINATION TARGET_ARCH)
    set(multiValueArgs
        SOURCES INCLUDE_DIRECTORIES COMPILE_DEFINITIONS COMPILE_OPTIONS
        LINK_DIRECTORIES LINK_LIBRARIES LINK_OPTIONS)
    cmake_parse_arguments(ARG "${options}" "${oneValueArgs}"
        "${multiValueArgs}" ${ARGN})

    if(ARG_UNPARSED_ARGUMENTS)
        message(FATAL_ERROR
            "vecops_add_kernel_library: unknown arguments: "
            "${ARG_UNPARSED_ARGUMENTS}")
    endif()
    if(NOT ARG_NAME)
        message(FATAL_ERROR "vecops_add_kernel_library requires NAME")
    endif()
    if(NOT ARG_SOURCES)
        message(FATAL_ERROR
            "vecops_add_kernel_library(${ARG_NAME}) requires SOURCES")
    endif()

    add_library(${ARG_NAME} MODULE ${ARG_SOURCES})
    target_compile_features(${ARG_NAME} PRIVATE cxx_std_20)
    set_target_properties(${ARG_NAME} PROPERTIES
        CXX_EXTENSIONS OFF
        C_VISIBILITY_PRESET hidden
        CXX_VISIBILITY_PRESET hidden
        VISIBILITY_INLINES_HIDDEN YES)
    if(ARG_TARGET_ARCH)
        vecops_configure_target_arch(
            TARGET ${ARG_NAME}
            ARCH ${ARG_TARGET_ARCH})
    endif()

    if(NOT ARG_NO_RUNTIME)
        # The KUPL-derived provider is header-only and deliberately has no KUPL
        # or OpenMP runtime dependency. Selecting it is an explicit build
        # decision for vecops kernels, never for framework-only bridges.
        if(VECOPS_ENABLE_KUPL_MMA)
            target_compile_definitions(${ARG_NAME} PRIVATE
                VECOPS_HAS_KUPL_MMA=1)
        endif()
        if(TARGET vecops::vecops)
            target_link_libraries(${ARG_NAME} PRIVATE vecops::vecops)
        elseif(TARGET vecops)
            target_link_libraries(${ARG_NAME} PRIVATE vecops)
        else()
            message(FATAL_ERROR
                "vecops_add_kernel_library requires the Vecops core target")
        endif()
        if(TARGET vecops::optimize_for_kernels)
            target_link_libraries(${ARG_NAME} PRIVATE
                vecops::optimize_for_kernels)
        elseif(TARGET vecops_optimize_for_kernels)
            target_link_libraries(${ARG_NAME} PRIVATE
                vecops_optimize_for_kernels)
        endif()
    endif()

    if(ARG_INCLUDE_DIRECTORIES)
        target_include_directories(${ARG_NAME} PRIVATE
            ${ARG_INCLUDE_DIRECTORIES})
    endif()
    if(ARG_COMPILE_DEFINITIONS)
        target_compile_definitions(${ARG_NAME} PRIVATE
            ${ARG_COMPILE_DEFINITIONS})
    endif()
    if(ARG_COMPILE_OPTIONS)
        target_compile_options(${ARG_NAME} PRIVATE ${ARG_COMPILE_OPTIONS})
    endif()
    if(ARG_LINK_DIRECTORIES)
        target_link_directories(${ARG_NAME} PRIVATE ${ARG_LINK_DIRECTORIES})
    endif()
    if(ARG_LINK_LIBRARIES)
        target_link_libraries(${ARG_NAME} PRIVATE ${ARG_LINK_LIBRARIES})
    endif()
    if(ARG_LINK_OPTIONS)
        target_link_options(${ARG_NAME} PRIVATE ${ARG_LINK_OPTIONS})
    endif()
    if(ARG_OUTPUT_NAME)
        set_target_properties(${ARG_NAME} PROPERTIES
            OUTPUT_NAME "${ARG_OUTPUT_NAME}")
    endif()

    if(NOT ARG_NO_INSTALL)
        if(ARG_INSTALL_DESTINATION)
            set(_VECOPS_KERNEL_INSTALL_DESTINATION
                "${ARG_INSTALL_DESTINATION}")
        else()
            set(_VECOPS_KERNEL_INSTALL_DESTINATION
                "${CMAKE_INSTALL_LIBDIR}")
        endif()
        install(TARGETS ${ARG_NAME}
            LIBRARY DESTINATION "${_VECOPS_KERNEL_INSTALL_DESTINATION}"
            RUNTIME DESTINATION "${_VECOPS_KERNEL_INSTALL_DESTINATION}")
    endif()
endfunction()
