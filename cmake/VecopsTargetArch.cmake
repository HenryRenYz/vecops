include_guard(GLOBAL)

# Framework-neutral ISA policy shared by ordinary Vecops targets and generated
# kernel modules. More specialized targets may add feature-specific libraries
# after applying this baseline.
function(vecops_configure_target_arch)
    set(options)
    set(oneValueArgs TARGET ARCH OUT_MARCH)
    cmake_parse_arguments(ARG "${options}" "${oneValueArgs}" "" ${ARGN})
    if(ARG_UNPARSED_ARGUMENTS OR NOT ARG_TARGET OR NOT ARG_ARCH)
        message(FATAL_ERROR
            "vecops_configure_target_arch requires TARGET and ARCH")
    endif()
    if(NOT TARGET ${ARG_TARGET})
        message(FATAL_ERROR
            "vecops_configure_target_arch: unknown target '${ARG_TARGET}'")
    endif()

    if(CMAKE_SYSTEM_PROCESSOR MATCHES "^(x86_64|AMD64|amd64)$")
        set(_VECOPS_ARCH_FAMILY x86)
        set(_VECOPS_SUPPORTED_ARCHES Scalar AVX AVX2 AVX512 Native)
        set(_VECOPS_MARCH_Scalar x86-64)
        set(_VECOPS_MARCH_AVX corei7-avx)
        set(_VECOPS_MARCH_AVX2 core-avx2)
        set(_VECOPS_MARCH_AVX512 skylake-avx512)
        set(_VECOPS_MARCH_Native native)
    elseif(CMAKE_SYSTEM_PROCESSOR MATCHES "^(aarch64|arm64|ARM64)$")
        set(_VECOPS_ARCH_FAMILY ARM)
        set(_VECOPS_SUPPORTED_ARCHES Scalar NEON SVE SVE2 Native
            NativeFixedSVE NativeFixedStreamingSVE)
        set(_VECOPS_MARCH_Scalar armv8-a)
        set(_VECOPS_MARCH_NEON armv8-a+simd)
        set(_VECOPS_MARCH_SVE armv8-a+sve)
        set(_VECOPS_MARCH_SVE2 armv8-a+sve2)
        set(_VECOPS_MARCH_Native native)
        set(_VECOPS_MARCH_NativeFixedSVE native)
        set(_VECOPS_MARCH_NativeFixedStreamingSVE native)
    else()
        message(FATAL_ERROR
            "vecops_configure_target_arch: unsupported architecture "
            "'${CMAKE_SYSTEM_PROCESSOR}'")
    endif()

    list(FIND _VECOPS_SUPPORTED_ARCHES "${ARG_ARCH}" _VECOPS_ARCH_INDEX)
    if(_VECOPS_ARCH_INDEX EQUAL -1)
        message(FATAL_ERROR
            "Vecops target arch '${ARG_ARCH}' is not supported on "
            "${_VECOPS_ARCH_FAMILY}; choose one of ${_VECOPS_SUPPORTED_ARCHES}")
    endif()

    # A source-tree build may have performed richer native feature detection.
    # Prefer that result while keeping installed SDKs self-contained.
    if(DEFINED VECOPS_MAP_${_VECOPS_ARCH_FAMILY}_${ARG_ARCH})
        set(_VECOPS_MARCH
            "${VECOPS_MAP_${_VECOPS_ARCH_FAMILY}_${ARG_ARCH}}")
    else()
        set(_VECOPS_MARCH "${_VECOPS_MARCH_${ARG_ARCH}}")
    endif()
    if(NOT _VECOPS_MARCH)
        message(FATAL_ERROR
            "Vecops target arch '${ARG_ARCH}' has no compiler mapping")
    endif()

    target_compile_options(${ARG_TARGET} PRIVATE "-march=${_VECOPS_MARCH}")
    if(_VECOPS_ARCH_FAMILY STREQUAL "ARM" AND
       ARG_ARCH MATCHES "^Native")
        # ACLE does not expose portable macros for every optional SME feature.
        # Reapply the facts detected when the SDK itself was configured so a
        # generated kernel sees the same backend contract as ordinary targets.
        if(VECOPS_NATIVE_HAS_SME_FA64)
            target_compile_definitions(${ARG_TARGET} PRIVATE
                VECOPS_TARGET_SME_FA64=1)
        endif()
    endif()
    if(ARG_ARCH STREQUAL "Scalar")
        target_compile_definitions(${ARG_TARGET} PRIVATE
            CPU_CAPABILITY=GENERIC CPU_CAPABILITY_GENERIC=1)
    endif()
    if(_VECOPS_ARCH_FAMILY STREQUAL "ARM" AND
       CMAKE_CXX_COMPILER_ID STREQUAL "GNU" AND
       NOT ARG_ARCH STREQUAL "Scalar")
        target_compile_options(${ARG_TARGET} PRIVATE -Wno-psabi)
    endif()
    if(ARG_ARCH STREQUAL "NativeFixedSVE")
        if(VECOPS_FIXED_SVE_BITS)
            set(_VECOPS_TARGET_FIXED_SVE_BITS
                "${VECOPS_FIXED_SVE_BITS}")
        else()
            set(_VECOPS_TARGET_FIXED_SVE_BITS
                "${VECOPS_NATIVE_FIXED_SVE_BITS}")
        endif()
        if(NOT _VECOPS_TARGET_FIXED_SVE_BITS)
            message(FATAL_ERROR
                "NativeFixedSVE requires an explicit or natively detected "
                "fixed SVE width")
        endif()
        target_compile_options(${ARG_TARGET} PRIVATE
            "-msve-vector-bits=${_VECOPS_TARGET_FIXED_SVE_BITS}")
    endif()
    if(ARG_ARCH STREQUAL "NativeFixedStreamingSVE")
        if(VECOPS_FIXED_STREAMING_SVE_BITS)
            set(_VECOPS_TARGET_FIXED_STREAMING_SVE_BITS
                "${VECOPS_FIXED_STREAMING_SVE_BITS}")
        else()
            set(_VECOPS_TARGET_FIXED_STREAMING_SVE_BITS
                "${VECOPS_NATIVE_FIXED_STREAMING_SVE_BITS}")
        endif()
        if(NOT _VECOPS_TARGET_FIXED_STREAMING_SVE_BITS)
            message(FATAL_ERROR
                "NativeFixedStreamingSVE requires "
                "an explicit or natively detected fixed streaming SVE width")
        endif()
        target_compile_definitions(${ARG_TARGET} PRIVATE
            "VECOPS_TARGET_FIXED_STREAMING_SVE_BITS=${_VECOPS_TARGET_FIXED_STREAMING_SVE_BITS}")
    endif()

    if(ARG_OUT_MARCH)
        set(${ARG_OUT_MARCH} "${_VECOPS_MARCH}" PARENT_SCOPE)
    endif()
endfunction()
