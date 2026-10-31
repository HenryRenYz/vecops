
file(GLOB_RECURSE _VECOPS_SRCS
    "${CMAKE_CURRENT_SOURCE_DIR}/*.cpp"
)
# Architecture-specialized objects are compiled with the target ISA by
# VecopsMultiarchTargets.cmake.  They must not enter the generic libvecops.a.
list(FILTER _VECOPS_SRCS EXCLUDE REGEX "/arch/")
list(FILTER _VECOPS_SRCS EXCLUDE REGEX "/compiler/")
list(FILTER _VECOPS_SRCS EXCLUDE REGEX "/memory/")
list(FILTER _VECOPS_SRCS EXCLUDE REGEX "/comm/")
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

if(VECOPS_BUILD_COMMUNICATION)
    find_package(Threads REQUIRED)
    add_library(vecops_comm comm/Communication.cpp)
    add_library(vecops::comm ALIAS vecops_comm)
    target_compile_features(vecops_comm PUBLIC cxx_std_20)
    target_include_directories(vecops_comm PUBLIC
        "$<BUILD_INTERFACE:${VECOPS_SOURCE_DIR}/include>"
        "$<BUILD_INTERFACE:${VECOPS_SOURCE_DIR}/include/vecops>"
        "$<INSTALL_INTERFACE:include>")
    target_sources(vecops_comm PRIVATE comm/CopyDispatch.cpp)
    # All variants share one vec implementation. Runtime selection is cold;
    # the portable variant keeps prebuilt libraries safe on older CPUs.
    function(vecops_comm_copy_variant variant)
        set(copy_target vecops_comm_copy_${variant})
        add_library(${copy_target} OBJECT comm/Copy.cpp)
        target_compile_features(${copy_target} PRIVATE cxx_std_20)
        target_include_directories(${copy_target} PRIVATE
            "${VECOPS_SOURCE_DIR}/include" "${VECOPS_SOURCE_DIR}/include/vecops")
        target_compile_definitions(${copy_target} PRIVATE VECOPS_COMM_COPY_VARIANT=${variant})
        target_compile_options(${copy_target} PRIVATE ${ARGN}
            "$<$<COMPILE_LANG_AND_ID:CXX,GNU,Clang,AppleClang>:-Wno-ignored-attributes>"
            "$<$<COMPILE_LANG_AND_ID:CXX,GNU,Clang,AppleClang>:-fvisibility=hidden>"
            "$<$<COMPILE_LANG_AND_ID:CXX,GNU,Clang,AppleClang>:-fvisibility-inlines-hidden>")
        set_target_properties(${copy_target} PROPERTIES POSITION_INDEPENDENT_CODE ON)
        target_sources(vecops_comm PRIVATE $<TARGET_OBJECTS:${copy_target}>)
    endfunction()
    vecops_comm_copy_variant(portable -DCPU_CAPABILITY_GENERIC=1)
    if(CMAKE_SYSTEM_PROCESSOR MATCHES "^(x86_64|AMD64|amd64)$")
        vecops_comm_copy_variant(avx2 -mavx2)
        vecops_comm_copy_variant(avx512 -mavx512f -mavx512bw -mavx512vl -mavx512dq -mavx512cd)
        target_compile_definitions(vecops_comm PRIVATE VECOPS_COMM_COPY_AVX2=1 VECOPS_COMM_COPY_AVX512=1)
    elseif(CMAKE_SYSTEM_PROCESSOR MATCHES "^(aarch64|arm64|ARM64)$")
        vecops_comm_copy_variant(sve -march=armv8-a+sve)
        target_compile_definitions(vecops_comm PRIVATE VECOPS_COMM_COPY_SVE=1)
    endif()
    set_target_properties(vecops_comm PROPERTIES EXPORT_NAME comm POSITION_INDEPENDENT_CODE ON)
    target_link_libraries(vecops_comm PUBLIC Threads::Threads rt)
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
