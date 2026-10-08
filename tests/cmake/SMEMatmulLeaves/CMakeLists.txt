# SPDX-FileCopyrightText: 2026 HenryRenYz and vecops contributors
# SPDX-License-Identifier: MIT
cmake_minimum_required(VERSION 3.22)
project(SMEMatmulLeafContract LANGUAGES CXX)

if(NOT VECOPS_MODULE_DIR)
    message(FATAL_ERROR "VECOPS_MODULE_DIR is required")
endif()
include("${VECOPS_MODULE_DIR}/VecopsSMEMatmulLeaves.cmake")

# Test target metadata on any host without executing an ARM compiler probe.
set(CMAKE_SYSTEM_PROCESSOR aarch64)
add_library(vecops INTERFACE)
add_library(vecops::vecops ALIAS vecops)
add_library(kernel MODULE Kernel.cpp)
_vecops_link_sme_matmul_leaves(kernel Native
    "armv9-a+sve2+bf16+i8mm+sme+sme-fa64+sme-f64f64")
get_target_property(_LINKS kernel LINK_LIBRARIES)
list(LENGTH _LINKS _LINK_COUNT)
if(NOT _LINK_COUNT EQUAL 1 OR NOT _LINKS MATCHES "^vecops_sme_matmul_leaves_")
    message(FATAL_ERROR "Source kernel did not link the shared SME leaf archive")
endif()
get_target_property(_PIC ${_LINKS} POSITION_INDEPENDENT_CODE)
if(NOT _PIC)
    message(FATAL_ERROR "SME leaves are not position-independent")
endif()
get_target_property(_SOURCES ${_LINKS} SOURCES)
list(LENGTH _SOURCES _SOURCE_COUNT)
if(NOT _SOURCE_COUNT EQUAL 3)
    message(FATAL_ERROR "Missing F64/runtime-quant/mixed-sign leaf sources")
endif()
foreach(_SOURCE IN LISTS _SOURCES)
    if(NOT EXISTS "${_SOURCE}")
        message(FATAL_ERROR "SDK source not found: ${_SOURCE}")
    endif()
endforeach()

# Identical feature combinations share one archive across compilation units.
add_library(second_kernel MODULE Kernel.cpp)
_vecops_link_sme_matmul_leaves(second_kernel Native
    "armv9-a+sve2+bf16+i8mm+sme+sme-fa64+sme-f64f64")
get_target_property(_SECOND_LINKS second_kernel LINK_LIBRARIES)
if(NOT _SECOND_LINKS STREQUAL _LINKS)
    message(FATAL_ERROR "SME leaf archive was duplicated")
endif()

# A plain SVE target must not acquire an SME-only object dependency.
add_library(sve_kernel MODULE Kernel.cpp)
_vecops_link_sme_matmul_leaves(sve_kernel SVE "armv8-a+sve")
get_target_property(_SVE_LINKS sve_kernel LINK_LIBRARIES)
if(_SVE_LINKS)
    message(FATAL_ERROR "Plain SVE unexpectedly links SME leaves")
endif()
