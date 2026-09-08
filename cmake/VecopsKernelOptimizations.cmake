include_guard(GLOBAL)

include(CheckCXXCompilerFlag)

# Opt-in compile policy for translation units containing performance-critical
# CPU kernels.  Keep this separate from vecops so ordinary users and tests do
# not silently inherit benchmark-oriented code-layout and math-library rules.
add_library(vecops_optimize_for_kernels INTERFACE)
add_library(vecops::optimize_for_kernels ALIAS vecops_optimize_for_kernels)
set_target_properties(vecops_optimize_for_kernels PROPERTIES
    EXPORT_NAME optimize_for_kernels)

function(_vecops_add_kernel_compile_option OPTION)
    string(MAKE_C_IDENTIFIER "${OPTION}" _OPTION_ID)
    set(_SUPPORTED_VAR
        "VECOPS_KERNEL_COMPILER_SUPPORTS_${_OPTION_ID}")
    check_cxx_compiler_flag("${OPTION}" "${_SUPPORTED_VAR}")
    if(${_SUPPORTED_VAR})
        target_compile_options(vecops_optimize_for_kernels INTERFACE
            "$<$<COMPILE_LANGUAGE:CXX>:${OPTION}>")
    endif()
endfunction()

if(CMAKE_CXX_COMPILER_ID MATCHES "^(GNU|Clang|AppleClang)$")
    # The AMX kernels are sensitive to function and hot-loop placement.  Both
    # GCC and current Clang accept these GCC-compatible driver options.
    _vecops_add_kernel_compile_option("-falign-functions=64")
    _vecops_add_kernel_compile_option("-falign-loops=64")

    # CPU math kernels do not use errno for error reporting.  This permits
    # scalar libm operations such as sqrt to be lowered to inline instructions
    # without enabling the result-changing transformations from -ffast-math.
    _vecops_add_kernel_compile_option("-fno-math-errno")
endif()
