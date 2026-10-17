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
       CMAKE_CXX_COMPILER_ID STREQUAL "Clang" AND
       ARG_ARCH MATCHES "^(SVE|SVE2|Native.*)$")
        _vecops_scalable_sink_compatibility(${ARG_TARGET} "${_VECOPS_MARCH}")
    endif()
    # Clang 19 accepts the optional ISA but omits its ACLE feature macro.
    if(_VECOPS_ARCH_FAMILY STREQUAL "ARM" AND
       ((_VECOPS_MARCH MATCHES "(^|\\+)sme-f64f64($|\\+)") OR
        (ARG_ARCH MATCHES "^Native" AND VECOPS_NATIVE_HAS_SME_F64F64)))
        target_compile_definitions(${ARG_TARGET} PRIVATE VECOPS_TARGET_SME_F64F64=1)
    endif()
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

# A bounded compiler probe for the actual SimplifyCFG common-store-sinking
# failure. This is deliberately not a Clang-version heuristic. The compiler
# executable and target flags identify the configure-time cache entry.
function(_vecops_scalable_sink_compatibility TARGET_NAME MARCH)
    get_property(_COMPILER_SHA GLOBAL PROPERTY VECOPS_SCALABLE_SINK_COMPILER_IDENTITY)
    if(NOT _COMPILER_SHA)
        file(SHA256 "${CMAKE_CXX_COMPILER}" _WRAPPER_SHA)
        execute_process(COMMAND "${CMAKE_CXX_COMPILER}" --version
            OUTPUT_VARIABLE _COMPILER_VERSION ERROR_VARIABLE _VERSION_STDERR
            RESULT_VARIABLE _VERSION_RESULT TIMEOUT 5)
        if(NOT "${_VERSION_RESULT}" STREQUAL "0")
            message(FATAL_ERROR "Cannot attest scalable compiler: ${_VERSION_STDERR}")
        endif()
        string(REGEX MATCH "InstalledDir: ([^\n\r]+)" _INSTALLED "${_COMPILER_VERSION}")
        set(_INSTALLED_SHA "")
        if(_INSTALLED AND EXISTS "${CMAKE_MATCH_1}/clang")
            file(SHA256 "${CMAKE_MATCH_1}/clang" _INSTALLED_SHA)
        endif()
        string(SHA256 _COMPILER_SHA "${_WRAPPER_SHA};${_COMPILER_VERSION};${_INSTALLED_SHA}")
        set_property(GLOBAL PROPERTY VECOPS_SCALABLE_SINK_COMPILER_IDENTITY "${_COMPILER_SHA}")
    endif()
    string(SHA256 _POLICY_KEY
        "${_COMPILER_SHA};${CMAKE_CXX_COMPILER_ARG1};${CMAKE_CXX_FLAGS};${MARCH};scalable-sink-v2")
    set(_CACHE_VAR "VECOPS_SCALABLE_SINK_${_POLICY_KEY}")
    if(NOT DEFINED ${_CACHE_VAR})
        set(_DIRECTORY "${CMAKE_BINARY_DIR}/CMakeFiles/scalable-sink-${_POLICY_KEY}")
        file(MAKE_DIRECTORY "${_DIRECTORY}")
        file(WRITE "${_DIRECTORY}/probe.cpp" "#include <arm_sve.h>\n#if defined(__ARM_FEATURE_SVE)\nvoid repro(bool cond, char* p) {\n if (cond) *reinterpret_cast<svint16_t*>(p) = svdup_s16(0);\n else *reinterpret_cast<svint16_t*>(p + 320) = svdup_s16(0);\n}\n#endif\n")
        separate_arguments(_FLAGS NATIVE_COMMAND "${CMAKE_CXX_FLAGS}")
        separate_arguments(_ARG1 NATIVE_COMMAND "${CMAKE_CXX_COMPILER_ARG1}")
        # An affected compiler is expected to crash here. Retain its stderr
        # attestation without leaving a preprocessed crash reproducer per build.
        set(_COMMAND "${CMAKE_CXX_COMPILER}" ${_ARG1} ${_FLAGS}
            -fno-crash-diagnostics -O3 "-march=${MARCH}" -msve-vector-bits=scalable
            -c "${_DIRECTORY}/probe.cpp" -o "${_DIRECTORY}/probe.o")
        execute_process(COMMAND ${_COMMAND} RESULT_VARIABLE _BASE_RESULT
            OUTPUT_VARIABLE _BASE_STDOUT ERROR_VARIABLE _BASE_STDERR TIMEOUT 15)
        set(_OPTION "")
        if(NOT "${_BASE_RESULT}" STREQUAL "0")
            if(_BASE_STDERR MATCHES "Invalid size request on a scalable vector")
                set(_OPTION "-mllvm=-simplifycfg-sink-common=false")
                execute_process(COMMAND ${_COMMAND} "${_OPTION}"
                    RESULT_VARIABLE _FIXED_RESULT OUTPUT_VARIABLE _FIXED_STDOUT
                    ERROR_VARIABLE _FIXED_STDERR TIMEOUT 15)
                if(NOT "${_FIXED_RESULT}" STREQUAL "0")
                    message(FATAL_ERROR "Scalable sinking compatibility probe still fails: ${_FIXED_STDERR}")
                endif()
            else()
                message(FATAL_ERROR "Scalable sinking probe failed (${_BASE_RESULT}): ${_BASE_STDERR}")
            endif()
        endif()
        file(WRITE "${_DIRECTORY}/attestation.txt"
            "compiler_sha=${_COMPILER_SHA}\nbase_exit=${_BASE_RESULT}\nflags=${_OPTION}\n${_BASE_STDERR}")
        set(${_CACHE_VAR} "${_OPTION}" CACHE INTERNAL "Attested scalable sinking policy")
    endif()
    if(${_CACHE_VAR})
        target_compile_options(${TARGET_NAME} PRIVATE "${${_CACHE_VAR}}")
    endif()
endfunction()
