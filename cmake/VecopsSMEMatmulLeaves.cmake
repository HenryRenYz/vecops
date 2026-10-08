# SPDX-FileCopyrightText: 2026 HenryRenYz and vecops contributors
# SPDX-License-Identifier: MIT
include_guard(GLOBAL)

# SME Matmul's ordinary-SVE skinny and runtime-quantized leaves are not
# header-only. Share the target-ISA archive between executable and JIT module
# consumers. Installed SDKs (including wheels) carry these three small sources
# beside this module; generic libvecops must remain ISA-independent.
function(_vecops_get_sme_matmul_leaf_library
        OUT_VAR MARCH USE_F64 USE_RUNTIME_QUANT_INT8 USE_MIXED_SIGN_SKINNY)
    set(_LEAF_KEY
        "${MARCH};f64=${USE_F64};runtime_i8=${USE_RUNTIME_QUANT_INT8};mixed_sign_skinny=${USE_MIXED_SIGN_SKINNY}")
    string(SHA1 _MARCH_HASH "${_LEAF_KEY}")
    string(SUBSTRING "${_MARCH_HASH}" 0 12 _MARCH_ID)
    set(_LEAF_TARGET "vecops_sme_matmul_leaves_${_MARCH_ID}")
    if(NOT TARGET ${_LEAF_TARGET})
        set(_LEAF_SOURCE_DIR
            "${CMAKE_CURRENT_FUNCTION_LIST_DIR}/../src/arch/sme")
        if(NOT EXISTS "${_LEAF_SOURCE_DIR}/FusedSkinnyF64.cpp")
            set(_LEAF_SOURCE_DIR "${CMAKE_CURRENT_FUNCTION_LIST_DIR}/sme")
        endif()
        set(_LEAF_SOURCES "")
        if(USE_F64)
            list(APPEND _LEAF_SOURCES
                "${_LEAF_SOURCE_DIR}/FusedSkinnyF64.cpp")
        endif()
        if(USE_RUNTIME_QUANT_INT8)
            list(APPEND _LEAF_SOURCES
                "${_LEAF_SOURCE_DIR}/RuntimeQuantInt8.cpp")
        endif()
        if(USE_MIXED_SIGN_SKINNY)
            list(APPEND _LEAF_SOURCES
                "${_LEAF_SOURCE_DIR}/MixedSignSkinnyInt8.cpp")
        endif()
        if(NOT _LEAF_SOURCES)
            message(FATAL_ERROR
                "SME matmul leaf library requested without any leaves")
        endif()
        foreach(_SOURCE IN LISTS _LEAF_SOURCES)
            if(NOT EXISTS "${_SOURCE}")
                message(FATAL_ERROR
                    "Vecops SDK is missing the SME Matmul leaf source '${_SOURCE}'")
            endif()
        endforeach()
        add_library(${_LEAF_TARGET} STATIC ${_LEAF_SOURCES})
        target_compile_features(${_LEAF_TARGET} PRIVATE cxx_std_20)
        set_target_properties(${_LEAF_TARGET} PROPERTIES
            POSITION_INDEPENDENT_CODE ON
            CXX_EXTENSIONS OFF
            FOLDER "internal/sme_matmul")
        if(USE_F64)
            # Clang 19 accepts the extension without defining its ACLE macro.
            target_compile_definitions(${_LEAF_TARGET} PRIVATE
                VECOPS_TARGET_SME_F64F64=1)
        endif()
        target_compile_options(${_LEAF_TARGET} PRIVATE
            "-march=${MARCH}" "-Wno-ignored-attributes")
        if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
            target_compile_options(${_LEAF_TARGET} PRIVATE "-Wno-psabi")
        endif()
        # The public target is the include/ABI/subnormal-policy authority for
        # source trees, build-tree packages and relocatable installed SDKs.
        if(TARGET vecops::vecops)
            target_link_libraries(${_LEAF_TARGET} PRIVATE vecops::vecops)
        elseif(TARGET vecops)
            target_link_libraries(${_LEAF_TARGET} PRIVATE vecops)
        else()
            message(FATAL_ERROR "SME Matmul leaves require the Vecops core target")
        endif()
        if(TARGET vecops::optimize_for_kernels)
            target_link_libraries(${_LEAF_TARGET} PRIVATE vecops::optimize_for_kernels)
        elseif(TARGET vecops_optimize_for_kernels)
            target_link_libraries(${_LEAF_TARGET} PRIVATE vecops_optimize_for_kernels)
        endif()
    endif()
    set(${OUT_VAR} "${_LEAF_TARGET}" PARENT_SCOPE)
endfunction()

function(_vecops_link_sme_matmul_leaves TARGET ARCH MARCH)
    if(ARCH STREQUAL "NONE")
        return()
    endif()
    if(NOT CMAKE_SYSTEM_PROCESSOR MATCHES "^(aarch64|arm64|ARM64)$")
        return()
    endif()
    set(_HAS_SME OFF)
    if((MARCH MATCHES "(^|\\+)sme($|\\+)") OR
       (ARCH MATCHES "^Native" AND VECOPS_NATIVE_HAS_SME))
        set(_HAS_SME ON)
    endif()
    if(NOT _HAS_SME)
        return()
    endif()
    set(_USE_F64 OFF)
    set(_USE_INT8 OFF)
    if((MARCH MATCHES "(^|\\+)sme-f64f64($|\\+)") OR
       (ARCH MATCHES "^Native" AND VECOPS_NATIVE_HAS_SME_F64F64))
        set(_USE_F64 ON)
    endif()
    if((MARCH MATCHES "(^|\\+)i8mm($|\\+)") OR
       (ARCH MATCHES "^Native" AND VECOPS_NATIVE_HAS_I8MM))
        set(_USE_INT8 ON)
    endif()
    if(_USE_F64 OR _USE_INT8)
        _vecops_get_sme_matmul_leaf_library(_LEAF_LIBRARY "${MARCH}"
            "${_USE_F64}" "${_USE_INT8}" "${_USE_INT8}")
        target_link_libraries(${TARGET} PRIVATE ${_LEAF_LIBRARY})
    endif()
endfunction()
