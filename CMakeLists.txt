cmake_minimum_required(VERSION 3.22)
project(Vecops VERSION 0.1.0 LANGUAGES C CXX)

include(GNUInstallDirs)
include(CMakePackageConfigHelpers)

set_property(GLOBAL PROPERTY USE_FOLDERS ON)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_VERBOSE_MAKEFILE ON)

# Keep an explicit project root.  CMAKE_SOURCE_DIR belongs to the outermost
# project and is therefore wrong when Vecops is consumed with add_subdirectory.
set(VECOPS_SOURCE_DIR "${CMAKE_CURRENT_SOURCE_DIR}")

# Address sanitizer

# ---------------------------------- Options -----------------------------------
set(VECOPS_GITHUB_ENDPOINT "https://github.com" CACHE STRING "Set endpoint address for GitHub repositories")
option(VECOPS_PRESERVE_SUBNORMALS
       "Preserve representable subnormal results in strict vector operations" OFF)
option(VECOPS_ENABLE_OPENMP
       "Use OpenMP for vecops CPU parallel execution" ON)
option(VECOPS_BUILD_COMPILER "Build the C++ kernel compiler" ON)
option(VECOPS_BUILD_PYTHON "Build the optional Python extension" OFF)
option(VECOPS_BUILD_TESTING "Build Vecops tests" OFF)
option(VECOPS_BUILD_BENCHMARKS "Build Vecops benchmarks" OFF)
option(VECOPS_ENABLE_KUPL_MMA
       "Enable KUPL MMA for eligible ARM SME matmul kernels" OFF)
set(VECOPS_KUPL_ROOT "" CACHE PATH
    "KUPL source or install root (must contain kupl_mma.h)")
set(VECOPS_KUPL_LIBRARY "" CACHE FILEPATH
    "KUPL shared library built with ENABLE_KUPL_MMA=ON")
set(VECOPS_FIXED_SVE_BITS "" CACHE STRING
    "SVE vector width in bits for fixed-SVE targets; empty detects the host maximum")
set(VECOPS_FIXED_STREAMING_SVE_BITS "" CACHE STRING
    "SME streaming vector width in bits; empty detects the native host maximum")

if(VECOPS_BUILD_COMPILER AND NOT UNIX)
    message(FATAL_ERROR
        "The initial vecops C++ compiler implementation requires a POSIX host; "
        "configure with VECOPS_BUILD_COMPILER=OFF on other platforms")
endif()

include(cmake/VecopsMultiarchTargets.cmake)
include(cmake/VecopsKernelOptimizations.cmake)

if(VECOPS_ENABLE_KUPL_MMA)
    if(NOT VECOPS_ARCH_FAMILY STREQUAL "ARM")
        message(FATAL_ERROR "VECOPS_ENABLE_KUPL_MMA is available only for ARM targets")
    endif()
    find_path(VECOPS_KUPL_INCLUDE_DIR kupl_mma.h
        HINTS
            "${VECOPS_KUPL_ROOT}"
            "${VECOPS_KUPL_ROOT}/include"
            "${VECOPS_KUPL_ROOT}/src"
        REQUIRED NO_DEFAULT_PATH)
    if(NOT VECOPS_KUPL_LIBRARY)
        find_library(VECOPS_KUPL_LIBRARY kupl
            HINTS
                "${VECOPS_KUPL_ROOT}/lib"
                "${VECOPS_KUPL_ROOT}/lib64"
                "${VECOPS_KUPL_ROOT}/build/src/src"
            REQUIRED NO_DEFAULT_PATH)
    elseif(NOT EXISTS "${VECOPS_KUPL_LIBRARY}")
        message(FATAL_ERROR
            "VECOPS_KUPL_LIBRARY does not exist: ${VECOPS_KUPL_LIBRARY}")
    endif()
    add_library(KUPL::MMA SHARED IMPORTED GLOBAL)
    set_target_properties(KUPL::MMA PROPERTIES
        IMPORTED_LOCATION "${VECOPS_KUPL_LIBRARY}"
        INTERFACE_INCLUDE_DIRECTORIES "${VECOPS_KUPL_INCLUDE_DIR}")
endif()

# -------------------------------- Dependencies --------------------------------
include(FetchContent)

if(VECOPS_ENABLE_OPENMP)
    find_package(OpenMP REQUIRED COMPONENTS CXX)
endif()

# ------------------------------- Source Config --------------------------------
add_subdirectory(src)

if(VECOPS_BUILD_PYTHON)
    set(PYBIND11_FINDPYTHON ON)
    find_package(Python COMPONENTS Interpreter Development.Module REQUIRED)
    find_package(pybind11 CONFIG REQUIRED)
    add_subdirectory(python/bindings)

    configure_file(
        cmake/VecopsPythonSdkTargets.cmake.in
        "${CMAKE_CURRENT_BINARY_DIR}/VecopsPythonSdkTargets.cmake"
        @ONLY
    )

    # A wheel is a self-contained compiler SDK. Editable builds continue to
    # reference the checkout directly, while wheel installs use this relocatable
    # package below vecops/_sdk.
    install(
        DIRECTORY include/vecops
        DESTINATION vecops/_sdk/include
        COMPONENT python
    )
    install(
        FILES "$<TARGET_FILE:vecops>"
        DESTINATION vecops/_sdk/lib
        COMPONENT python
    )
    install(
        FILES "${CMAKE_CURRENT_BINARY_DIR}/VecopsPythonSdkTargets.cmake"
        RENAME VecopsTargets.cmake
        DESTINATION vecops/_sdk/lib/cmake/Vecops
        COMPONENT python
    )
    install(
        FILES
            cmake/VecopsKernelTargets.cmake
            cmake/VecopsTargetArch.cmake
            "${CMAKE_CURRENT_BINARY_DIR}/VecopsConfig.cmake"
            "${CMAKE_CURRENT_BINARY_DIR}/VecopsConfigVersion.cmake"
        DESTINATION vecops/_sdk/lib/cmake/Vecops
        COMPONENT python
    )
endif()

# -------------------------------- Test Config ---------------------------------
if(VECOPS_BUILD_TESTING)
    enable_testing()
    # Vecops owns installation of its SDK only.  Do not let an opt-in test
    # dependency add GoogleTest/GoogleMock artifacts to `cmake --install`.
    set(INSTALL_GTEST OFF CACHE BOOL "" FORCE)
    FetchContent_Declare(
            googletest
            GIT_REPOSITORY "${VECOPS_GITHUB_ENDPOINT}/google/googletest.git"
            GIT_TAG        v1.17.0
    )
    FetchContent_MakeAvailable(googletest)
    include(GoogleTest)
    add_subdirectory(tests)
endif()

if(VECOPS_BUILD_BENCHMARKS)
    set(BENCHMARK_ENABLE_TESTING OFF CACHE BOOL "" FORCE)
    set(BENCHMARK_ENABLE_INSTALL OFF CACHE BOOL "" FORCE)
    set(BENCHMARK_ENABLE_GTEST_TESTS OFF CACHE BOOL "" FORCE)
    FetchContent_Declare(
            googlebenchmark
            GIT_REPOSITORY "${VECOPS_GITHUB_ENDPOINT}/google/benchmark.git"
            GIT_TAG        v1.8.3
    )
    FetchContent_MakeAvailable(googlebenchmark)
    add_subdirectory(benchmarks)
endif()

# ------------------------------- Installation --------------------------------
install(
    DIRECTORY include/vecops
    DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}"
)

install(
    TARGETS vecops vecops_optimize_for_kernels
    EXPORT VecopsTargets
    ARCHIVE DESTINATION "${CMAKE_INSTALL_LIBDIR}"
    LIBRARY DESTINATION "${CMAKE_INSTALL_LIBDIR}"
    RUNTIME DESTINATION "${CMAKE_INSTALL_BINDIR}"
    INCLUDES DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}"
)
if(TARGET vecops_compiler)
    install(
        TARGETS vecops_compiler
        EXPORT VecopsTargets
        ARCHIVE DESTINATION "${CMAKE_INSTALL_LIBDIR}"
        LIBRARY DESTINATION "${CMAKE_INSTALL_LIBDIR}"
        RUNTIME DESTINATION "${CMAKE_INSTALL_BINDIR}"
        INCLUDES DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}"
    )
endif()

install(
    EXPORT VecopsTargets
    FILE VecopsTargets.cmake
    NAMESPACE vecops::
    DESTINATION "${CMAKE_INSTALL_LIBDIR}/cmake/Vecops"
)
install(
    FILES
        cmake/VecopsKernelTargets.cmake
        cmake/VecopsTargetArch.cmake
    DESTINATION "${CMAKE_INSTALL_LIBDIR}/cmake/Vecops"
)

configure_package_config_file(
    cmake/VecopsConfig.cmake.in
    "${CMAKE_CURRENT_BINARY_DIR}/VecopsConfig.cmake"
    INSTALL_DESTINATION "${CMAKE_INSTALL_LIBDIR}/cmake/Vecops"
)
write_basic_package_version_file(
    "${CMAKE_CURRENT_BINARY_DIR}/VecopsConfigVersion.cmake"
    VERSION "${PROJECT_VERSION}"
    COMPATIBILITY SameMajorVersion
)
install(
    FILES
        "${CMAKE_CURRENT_BINARY_DIR}/VecopsConfig.cmake"
        "${CMAKE_CURRENT_BINARY_DIR}/VecopsConfigVersion.cmake"
    DESTINATION "${CMAKE_INSTALL_LIBDIR}/cmake/Vecops"
)

# Let a generated JIT project consume an uninstalled build tree with the same
# find_package(Vecops CONFIG) contract as an installed SDK.
export(
    EXPORT VecopsTargets
    FILE "${CMAKE_CURRENT_BINARY_DIR}/VecopsTargets.cmake"
    NAMESPACE vecops::
)
configure_file(
    cmake/VecopsKernelTargets.cmake
    "${CMAKE_CURRENT_BINARY_DIR}/VecopsKernelTargets.cmake"
    COPYONLY
)
configure_file(
    cmake/VecopsTargetArch.cmake
    "${CMAKE_CURRENT_BINARY_DIR}/VecopsTargetArch.cmake"
    COPYONLY
)
