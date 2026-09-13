
file(GLOB_RECURSE _VECOPS_SRCS
    "${CMAKE_CURRENT_SOURCE_DIR}/*.cpp"
)
# Architecture-specialized objects are compiled with the target ISA by
# VecopsMultiarchTargets.cmake.  They must not enter the generic libvecops.a.
list(FILTER _VECOPS_SRCS EXCLUDE REGEX "/arch/")
list(FILTER _VECOPS_SRCS EXCLUDE REGEX "/compiler/")
list(FILTER _VECOPS_SRCS EXCLUDE REGEX "/memory/")
list(FILTER _VECOPS_SRCS EXCLUDE REGEX "/OpenMPThreadPool\\.cpp$")

add_library(vecops ${_VECOPS_SRCS})
add_library(vecops::vecops ALIAS vecops)
target_compile_features(vecops PUBLIC cxx_std_20)
target_include_directories(vecops
    PUBLIC
        "$<BUILD_INTERFACE:${VECOPS_SOURCE_DIR}/include>"
        "$<BUILD_INTERFACE:${VECOPS_SOURCE_DIR}/include/vecops>"
        "$<INSTALL_INTERFACE:include>"
        "$<INSTALL_INTERFACE:include/vecops>"
    PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}"
)
set_target_properties(vecops PROPERTIES
    EXPORT_NAME vecops
    POSITION_INDEPENDENT_CODE ON
)

target_compile_definitions(vecops PUBLIC "$<$<CONFIG:Debug>:VECOPS_DEBUG>")
if(VECOPS_PRESERVE_SUBNORMALS)
    target_compile_definitions(vecops PUBLIC VECOPS_PRESERVE_SUBNORMALS=1)
endif()
target_compile_options(vecops PUBLIC
    "$<$<COMPILE_LANG_AND_ID:CXX,GNU,Clang,AppleClang>:-Wno-ignored-attributes>")
target_link_libraries(vecops PUBLIC ${CMAKE_DL_LIBS})
if(VECOPS_ENABLE_OPENMP)
    add_library(vecops_threadpool_openmp SHARED OpenMPThreadPool.cpp)
    add_library(vecops::threadpool_openmp ALIAS vecops_threadpool_openmp)
    target_compile_features(vecops_threadpool_openmp PUBLIC cxx_std_20)
    target_include_directories(vecops_threadpool_openmp PUBLIC
        "$<BUILD_INTERFACE:${VECOPS_SOURCE_DIR}/include>"
        "$<INSTALL_INTERFACE:include>")
    target_link_libraries(vecops_threadpool_openmp PRIVATE OpenMP::OpenMP_CXX)
    set_target_properties(vecops_threadpool_openmp PROPERTIES
        EXPORT_NAME threadpool_openmp
        POSITION_INDEPENDENT_CODE ON)
endif()

if(NOT VECOPS_BUILD_MEMORY STREQUAL "OFF")
    add_library(vecops_memory memory/MemorySystem.cpp)
    add_library(vecops::memory ALIAS vecops_memory)
    target_compile_features(vecops_memory PUBLIC cxx_std_20)
    target_include_directories(vecops_memory
        PUBLIC
            "$<BUILD_INTERFACE:${VECOPS_SOURCE_DIR}/include>"
            "$<INSTALL_INTERFACE:include>"
    )
    set_target_properties(vecops_memory PROPERTIES
        EXPORT_NAME memory
        POSITION_INDEPENDENT_CODE ON
    )
    if(VECOPS_MEMORY_HAVE_HWLOC)
        target_compile_definitions(vecops_memory PRIVATE VECOPS_MEMORY_HAVE_HWLOC=1)
        target_link_libraries(vecops_memory PUBLIC PkgConfig::HWLOC)
    endif()
endif()

if(VECOPS_BUILD_COMPILER)
    add_library(vecops_compiler compiler/Compiler.cpp compiler/KernelCompiler.cpp)
    add_library(vecops::compiler ALIAS vecops_compiler)
    target_compile_features(vecops_compiler PUBLIC cxx_std_20)
    target_include_directories(vecops_compiler PUBLIC
        "$<BUILD_INTERFACE:${VECOPS_SOURCE_DIR}/include>"
        "$<INSTALL_INTERFACE:include>"
    )
    set_target_properties(vecops_compiler PROPERTIES
        EXPORT_NAME compiler
        POSITION_INDEPENDENT_CODE ON
    )
    target_link_libraries(vecops_compiler PUBLIC vecops)
endif()
