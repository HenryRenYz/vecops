option(VECOPS_MATMUL_OTHERLIBS_OPENBLAS "Build the OpenBLAS adapter" ON)
option(VECOPS_MATMUL_OTHERLIBS_ONEDNN "Build the oneDNN adapter" OFF)
option(VECOPS_MATMUL_OTHERLIBS_LIBXSMM "Build the LIBXSMM adapter" OFF)
option(VECOPS_MATMUL_OTHERLIBS_ACL "Build the Arm Compute Library adapter" OFF)
option(VECOPS_MATMUL_OTHERLIBS_KUPL "Build the KUPL MMA adapter" OFF)
set(VECOPS_MATMUL_OTHERLIBS_OPENBLAS_ROOT "" CACHE PATH
    "Pinned OpenBLAS install prefix with BUILD_BFLOAT16 enabled")
set(VECOPS_MATMUL_OTHERLIBS_OPENBLAS_BUILD_TARGET "unspecified" CACHE STRING
    "OpenBLAS TARGET used to build the pinned library (recorded in CSV)")

set(VECOPS_MATMUL_OTHERLIBS_DIR "${CMAKE_CURRENT_SOURCE_DIR}")

set(_VECOPS_CASE_GROUP_NAMES
    General LLM AF3Projection AF3AttentionQKLarge AF3Triangle
    AF3AttentionShallow AF3AttentionPV)
foreach(_index RANGE 0 6)
    list(GET _VECOPS_CASE_GROUP_NAMES ${_index} _group_name)
    vecops_add_benchmark(
        NAME MatmulOtherlibsVecops${_group_name}Bench
        FILES ${VECOPS_MATMUL_OTHERLIBS_DIR}/VecopsBench.cpp
        ARCH Native
        DEFINITIONS
            VECOPS_MATMUL_OTHERLIBS_PROVIDER="vecops"
            VECOPS_CASE_GROUP=${_index}
    )
endforeach()

vecops_add_benchmark(
    NAME MatmulOtherlibsVecopsTilingProbe
    FILES ${VECOPS_MATMUL_OTHERLIBS_DIR}/VecopsTilingProbe.cpp
    ARCH Native
    DEFINITIONS VECOPS_MATMUL_OTHERLIBS_PROVIDER="vecops"
)

vecops_add_benchmark(
    NAME MatmulOtherlibsVecopsPhaseProbe
    FILES ${VECOPS_MATMUL_OTHERLIBS_DIR}/VecopsPhaseProbe.cpp
    ARCH Native
    DEFINITIONS VECOPS_MATMUL_OTHERLIBS_PROVIDER="vecops"
)

if(VECOPS_ARCH_FAMILY STREQUAL "ARM")
    vecops_add_benchmark(
        NAME MatmulOtherlibsVecopsBatchPhaseProbe
        FILES ${VECOPS_MATMUL_OTHERLIBS_DIR}/VecopsBatchPhaseProbe.cpp
        ARCH Native
        DEFINITIONS VECOPS_MATMUL_OTHERLIBS_PROVIDER="vecops"
    )
endif()

set(_VECOPS_FUSION_NAMES Bias BiasRelu BiasSilu)
set(_VECOPS_FUSION_VALUES 0 1 2)
foreach(_index RANGE 0 2)
    list(GET _VECOPS_FUSION_NAMES ${_index} _fusion_name)
    list(GET _VECOPS_FUSION_VALUES ${_index} _fusion_value)
    vecops_add_benchmark(
        NAME MatmulOtherlibsVecops${_fusion_name}Bench
        FILES ${VECOPS_MATMUL_OTHERLIBS_DIR}/VecopsFusionBench.cpp
        ARCH Native
        DEFINITIONS
            VECOPS_MATMUL_OTHERLIBS_PROVIDER="vecops"
            VECOPS_FUSION_OPERATION=${_fusion_value}
    )
endforeach()

if(VECOPS_MATMUL_OTHERLIBS_OPENBLAS)
    if(VECOPS_MATMUL_OTHERLIBS_OPENBLAS_ROOT)
        # OpenBLAS_DIR is cached by find_package.  Clear it so changing the
        # explicitly pinned root cannot silently keep linking an older build.
        unset(OpenBLAS_DIR CACHE)
        find_package(OpenBLAS CONFIG REQUIRED
            PATHS "${VECOPS_MATMUL_OTHERLIBS_OPENBLAS_ROOT}" NO_DEFAULT_PATH)
        set(_VECOPS_OPENBLAS_TARGET OpenBLAS::OpenBLAS)
    else()
        find_package(PkgConfig REQUIRED)
        pkg_check_modules(OPENBLAS REQUIRED IMPORTED_TARGET openblas)
        set(_VECOPS_OPENBLAS_TARGET PkgConfig::OPENBLAS)
    endif()
    vecops_add_benchmark(
        NAME MatmulOtherlibsOpenBLASBench
        FILES ${VECOPS_MATMUL_OTHERLIBS_DIR}/OpenBLASBench.cpp
        ARCH Native
        LIBRARIES ${_VECOPS_OPENBLAS_TARGET}
        DEFINITIONS
            VECOPS_MATMUL_OTHERLIBS_PROVIDER="OpenBLAS"
            VECOPS_OPENBLAS_BUILD_TARGET="${VECOPS_MATMUL_OTHERLIBS_OPENBLAS_BUILD_TARGET}"
    )
    if(VECOPS_ARCH_FAMILY STREQUAL "x86")
        vecops_add_benchmark(
            NAME MatmulOtherlibsVecopsSpecialCaseProbe
            FILES ${VECOPS_MATMUL_OTHERLIBS_DIR}/VecopsSpecialCaseProbe.cpp
            ARCH Native
            LIBRARIES ${_VECOPS_OPENBLAS_TARGET}
            DEFINITIONS
                VECOPS_MATMUL_OTHERLIBS_PROVIDER="diagnostic"
                VECOPS_OPENBLAS_BUILD_TARGET="${VECOPS_MATMUL_OTHERLIBS_OPENBLAS_BUILD_TARGET}"
        )
    endif()
endif()

# The remaining adapters are deliberately opt-in. Their source trees are
# supplied explicitly so a benchmark run can pin and record exact revisions.
set(VECOPS_MATMUL_OTHERLIBS_ONEDNN_SOURCE_DIR "" CACHE PATH
    "Pinned oneDNN source tree used by the matmul comparison")
set(VECOPS_MATMUL_OTHERLIBS_LIBXSMM_SOURCE_DIR "" CACHE PATH
    "Pinned LIBXSMM source tree used by the matmul comparison")
set(VECOPS_MATMUL_OTHERLIBS_ACL_SOURCE_DIR "" CACHE PATH
    "Pinned Arm Compute Library source tree used by the matmul comparison")
set(VECOPS_MATMUL_OTHERLIBS_KUPL_SOURCE_DIR "" CACHE PATH
    "Pinned KUPL source tree used by the matmul comparison")

if(VECOPS_MATMUL_OTHERLIBS_ONEDNN)
    if(NOT EXISTS "${VECOPS_MATMUL_OTHERLIBS_ONEDNN_SOURCE_DIR}/CMakeLists.txt")
        message(FATAL_ERROR "Set VECOPS_MATMUL_OTHERLIBS_ONEDNN_SOURCE_DIR")
    endif()
    set(ONEDNN_BUILD_TESTS OFF CACHE BOOL "" FORCE)
    set(ONEDNN_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
    set(ONEDNN_BUILD_GRAPH OFF CACHE BOOL "" FORCE)
    set(ONEDNN_CPU_RUNTIME "SEQ" CACHE STRING "" FORCE)
    set(ONEDNN_GPU_RUNTIME "NONE" CACHE STRING "" FORCE)
    set(ONEDNN_LIBRARY_TYPE "STATIC" CACHE STRING "" FORCE)
    set(ONEDNN_ENABLE_WORKLOAD "INFERENCE" CACHE STRING "" FORCE)
    set(ONEDNN_ENABLE_PRIMITIVE "MATMUL;REORDER;ELTWISE;BINARY" CACHE STRING "" FORCE)
    set(ONEDNN_AARCH64_USE_ACL OFF CACHE BOOL "" FORCE)
    add_subdirectory(
        "${VECOPS_MATMUL_OTHERLIBS_ONEDNN_SOURCE_DIR}"
        "${CMAKE_BINARY_DIR}/_deps/matmul-otherlibs-onednn-build"
        EXCLUDE_FROM_ALL)
    vecops_add_benchmark(
        NAME MatmulOtherlibsOneDNNBench
        FILES ${VECOPS_MATMUL_OTHERLIBS_DIR}/OneDNNBench.cpp
        ARCH Native
        LIBRARIES dnnl
        DEFINITIONS VECOPS_MATMUL_OTHERLIBS_PROVIDER="oneDNN"
    )
    vecops_add_benchmark(
        NAME MatmulOtherlibsOneDNNPhaseProbe
        FILES ${VECOPS_MATMUL_OTHERLIBS_DIR}/OneDNNPhaseProbe.cpp
        ARCH Native
        LIBRARIES dnnl
        DEFINITIONS VECOPS_MATMUL_OTHERLIBS_PROVIDER="oneDNN"
    )
endif()

if(VECOPS_MATMUL_OTHERLIBS_LIBXSMM)
    if(NOT EXISTS "${VECOPS_MATMUL_OTHERLIBS_LIBXSMM_SOURCE_DIR}/CMakeLists.txt")
        message(FATAL_ERROR "Set VECOPS_MATMUL_OTHERLIBS_LIBXSMM_SOURCE_DIR")
    endif()
    set(LIBXSMM_FORTRAN OFF CACHE BOOL "" FORCE)
    set(BUILD_TESTING OFF CACHE BOOL "" FORCE)
    add_subdirectory(
        "${VECOPS_MATMUL_OTHERLIBS_LIBXSMM_SOURCE_DIR}"
        "${CMAKE_BINARY_DIR}/_deps/matmul-otherlibs-libxsmm-build"
        EXCLUDE_FROM_ALL)
    vecops_add_benchmark(
        NAME MatmulOtherlibsLIBXSMMBench
        FILES ${VECOPS_MATMUL_OTHERLIBS_DIR}/LIBXSMMBench.cpp
        ARCH Native
        LIBRARIES xsmm
        DEFINITIONS VECOPS_MATMUL_OTHERLIBS_PROVIDER="LIBXSMM"
    )
endif()

if(VECOPS_MATMUL_OTHERLIBS_ACL)
    if(NOT VECOPS_ARCH_FAMILY STREQUAL "ARM")
        message(FATAL_ERROR "The optimized ACL comparison is ARM-only")
    endif()
    if(NOT EXISTS "${VECOPS_MATMUL_OTHERLIBS_ACL_SOURCE_DIR}/SConstruct")
        message(FATAL_ERROR "Set VECOPS_MATMUL_OTHERLIBS_ACL_SOURCE_DIR")
    endif()
    set(VECOPS_MATMUL_OTHERLIBS_ACL_LIB_DIR "" CACHE PATH
        "Directory containing a pinned ACL build")
    find_library(ACL_COMPUTE_LIBRARY arm_compute
        PATHS "${VECOPS_MATMUL_OTHERLIBS_ACL_LIB_DIR}" REQUIRED NO_DEFAULT_PATH)
    find_library(ACL_CORE_LIBRARY arm_compute_core
        PATHS "${VECOPS_MATMUL_OTHERLIBS_ACL_LIB_DIR}" NO_DEFAULT_PATH)
    set(_VECOPS_ACL_LIBRARIES ${ACL_COMPUTE_LIBRARY})
    if(ACL_CORE_LIBRARY)
        list(APPEND _VECOPS_ACL_LIBRARIES ${ACL_CORE_LIBRARY})
    endif()
    vecops_add_benchmark(
        NAME MatmulOtherlibsACLBench
        FILES ${VECOPS_MATMUL_OTHERLIBS_DIR}/ACLBench.cpp
        ARCH Native
        LIBRARIES ${_VECOPS_ACL_LIBRARIES} dl pthread
        DEFINITIONS VECOPS_MATMUL_OTHERLIBS_PROVIDER="ACL"
    )
    foreach(_target MatmulOtherlibsACLBench-Native)
        target_include_directories(${_target} PRIVATE
            "${VECOPS_MATMUL_OTHERLIBS_ACL_SOURCE_DIR}"
            "${VECOPS_MATMUL_OTHERLIBS_ACL_SOURCE_DIR}/include"
            "${VECOPS_MATMUL_OTHERLIBS_ACL_SOURCE_DIR}/src/cpu/kernels/assembly")
    endforeach()
endif()

if(VECOPS_MATMUL_OTHERLIBS_KUPL)
    if(NOT VECOPS_ARCH_FAMILY STREQUAL "ARM")
        message(FATAL_ERROR "KUPL MMA is ARM-only")
    endif()
    if(NOT EXISTS "${VECOPS_MATMUL_OTHERLIBS_KUPL_SOURCE_DIR}/src/kupl_mma.h")
        message(FATAL_ERROR "Set VECOPS_MATMUL_OTHERLIBS_KUPL_SOURCE_DIR")
    endif()
    set(VECOPS_MATMUL_OTHERLIBS_KUPL_LIB "" CACHE FILEPATH
        "Pinned libkupl.so used by the comparison")
    find_package(OpenMP REQUIRED)
    vecops_add_benchmark(
        NAME MatmulOtherlibsKUPLBench
        FILES ${VECOPS_MATMUL_OTHERLIBS_DIR}/KUPLBench.cpp
        ARCH Native
        LIBRARIES "${VECOPS_MATMUL_OTHERLIBS_KUPL_LIB}" OpenMP::OpenMP_CXX
        DEFINITIONS VECOPS_MATMUL_OTHERLIBS_PROVIDER="KUPL-MMA"
    )
    foreach(_target MatmulOtherlibsKUPLBench-Native)
        target_include_directories(${_target} PRIVATE
            "${VECOPS_MATMUL_OTHERLIBS_KUPL_SOURCE_DIR}/src")
    endforeach()
endif()
