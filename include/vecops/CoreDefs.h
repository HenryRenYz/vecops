//
// Created by renyz on 2026/6/1.
//

#ifndef VECOPS_COREDEFS_H
#define VECOPS_COREDEFS_H

#if !defined(__cplusplus) || __cplusplus < 202002L
#error "VecOps requires C++20 or newer"
#endif

/**
 * Define VECOPS_PRESERVE_SUBNORMALS consistently for every translation unit
 * to make strict vector operations preserve representable subnormal results.
 * The default build permits hardware FTZ/DAZ behavior for maximum throughput.
 */

#include <csignal>

#include "vecops/platform/Features.h"

/**
 * Debug & release flags
 */
#if !defined(VECOPS_DEBUG)
#define VECOPS_RELEASE 1
#endif

/**
 * Definition for VECOPS_NOINLINE, VECOPS_INLINE, and VECOPS_ALWAYS_INLINE
 * Inline controller.
 * VECOPS_NOINLINE: function never inline.
 * VECOPS_INLINE: function always inline, except when compiling in debug mode (for easy debugging).
 * VECOPS_ALWAYS_INLINE: function always inline, used for primitives.
 */
#if defined(COMPILER_GCC) || defined(COMPILER_CLANG)
  #define VECOPS_NOINLINE __attribute__((noinline))
  #define VECOPS_ALWAYS_INLINE __attribute__((always_inline)) inline
  #define VECOPS_INLINE_LAMBDA __attribute__((always_inline))
#elif defined(COMPILER_MSVC)
#define VECOPS_NOINLINE __declspec(noinline)
  #define VECOPS_ALWAYS_INLINE __forceinline
  #define VECOPS_INLINE_LAMBDA
#else
  #define VECOPS_NOINLINE
  #define VECOPS_ALWAYS_INLINE inline
  #define VECOPS_INLINE_LAMBDA
#endif
#if defined(VECOPS_RELEASE)
#define VECOPS_INLINE VECOPS_ALWAYS_INLINE
#else
#define VECOPS_INLINE inline
#endif

/**
 * Code-placement alignment controls.
 *
 * VECOPS_FUNCTION_ALIGN: request a minimum alignment for a function entry.
 * VECOPS_LOOP_ALIGN: request a minimum alignment for one loop header.  This is
 * currently available only in Clang; unsupported compilers treat it as a
 * performance-only no-op.
 *
 * Both arguments must be power-of-two byte counts.  Function alignment may be
 * capped by the object format or linker.  These macros align generated code,
 * not objects or pointed-to storage; use standard alignas for data instead.
 */
#if defined(COMPILER_GCC) || defined(COMPILER_CLANG)
#define VECOPS_FUNCTION_ALIGN(bytes) __attribute__((aligned(bytes)))
#else
#define VECOPS_FUNCTION_ALIGN(bytes)
#endif

#if defined(COMPILER_CLANG)
#if __has_cpp_attribute(clang::code_align)
#define VECOPS_LOOP_ALIGN(bytes) [[clang::code_align(bytes)]]
#else
#define VECOPS_LOOP_ALIGN(bytes)
#endif
#else
#define VECOPS_LOOP_ALIGN(bytes)
#endif

/* Clang and GCC require opposite ordering for a GNU lambda attribute and
 * noexcept. Keep that grammar difference out of kernel call sites. */
#if defined(COMPILER_CLANG)
#define VECOPS_INLINE_LAMBDA_NOEXCEPT \
  VECOPS_INLINE_LAMBDA noexcept
#define VECOPS_INLINE_LAMBDA_NOEXCEPT_IF(...) \
  VECOPS_INLINE_LAMBDA noexcept(__VA_ARGS__)
#elif defined(COMPILER_GCC)
#define VECOPS_INLINE_LAMBDA_NOEXCEPT \
  noexcept VECOPS_INLINE_LAMBDA
#define VECOPS_INLINE_LAMBDA_NOEXCEPT_IF(...) \
  noexcept(__VA_ARGS__) VECOPS_INLINE_LAMBDA
#else
#define VECOPS_INLINE_LAMBDA_NOEXCEPT noexcept
#define VECOPS_INLINE_LAMBDA_NOEXCEPT_IF(...) noexcept(__VA_ARGS__)
#endif

/** Force-inline syntax shared by generic kernel functions and lambdas. */
#define VECOPS_KERNEL_FUNCTION(...) VECOPS_ALWAYS_INLINE __VA_ARGS__
#define VECOPS_KERNEL_LAMBDA VECOPS_INLINE_LAMBDA

/**
 * Pretty function name
 */
#if defined(COMPILER_GCC) || defined(COMPILER_CLANG)
#define VECOPS_FUNC_NAME __PRETTY_FUNCTION__
#elif defined(COMPILER_MSVC)
#define VECOPS_FUNC_NAME __FUNCSIG__
#else
#define VECOPS_FUNC_NAME __func__
#endif

/**
 * Explicit breakpoint
 */
#if defined(COMPILER_GCC)
#define VECOPS_BREAKPOINT std::raise(SIGTRAP)
#elif defined(COMPILER_CLANG)
#define VECOPS_BREAKPOINT __builtin_debugtrap()
#elif defined(COMPILER_MSVC)
#define VECOPS_BREAKPOINT __debugbreak()
#else
#define VECOPS_BREAKPOINT std::raise(SIGTRAP)
#endif

/**
 * Pure marker
 */
#if defined(COMPILER_GCC) || defined(COMPILER_CLANG)
#define VECOPS_PURE __attribute__((const))
#else
#define VECOPS_PURE
#endif

/**
 * Branch prediction hints
 * VECOPS_LIKELY:   condition is expected to be true.
 * VECOPS_UNLIKELY: condition is expected to be false.
 */
#if defined(COMPILER_GCC) || defined(COMPILER_CLANG)
#define VECOPS_LIKELY(x)   (__builtin_expect(!!(x), 1))
#define VECOPS_UNLIKELY(x) (__builtin_expect(!!(x), 0))
#else
#define VECOPS_LIKELY(x)   (x)
#define VECOPS_UNLIKELY(x) (x)
#endif

/**
 * Express a caller-side contract to the optimizer.  Unlike an assertion this
 * emits no failure path; violating the condition is undefined behavior.
 */
#if defined(COMPILER_CLANG)
#define VECOPS_ASSUME(cond) __builtin_assume(cond)
#elif defined(COMPILER_GCC)
#define VECOPS_ASSUME(cond)                              \
  do {                                                   \
    if (!(cond)) __builtin_unreachable();                \
  } while (0)
#elif defined(COMPILER_MSVC)
#define VECOPS_ASSUME(cond) __assume(cond)
#else
#define VECOPS_ASSUME(cond) ((void)0)
#endif

/**
 * Loop unroll controls.
 */
#if defined(COMPILER_GCC) || defined(COMPILER_CLANG)
#define VECOPS_UNROLL _Pragma("GCC unroll 16")
#define VECOPS_NOUNROLL _Pragma("GCC unroll 1")
#elif defined(COMPILER_MSVC)
#define VECOPS_UNROLL __pragma(loop(unroll))
#define VECOPS_NOUNROLL
#else
#define VECOPS_UNROLL
#define VECOPS_NOUNROLL
#endif

/**
 * Unreachable hint
 */
#if defined(__cplusplus) && __cplusplus >= 202302L
#define VECOPS_UNREACHABLE() std::unreachable()
#elif defined(COMPILER_GCC) || defined(COMPILER_CLANG)
#define VECOPS_UNREACHABLE() __builtin_unreachable()
#elif defined(COMPILER_MSVC)
#define VECOPS_UNREACHABLE() __assume(false)
#else
#define VECOPS_UNREACHABLE() ((void)0)
#endif

/**
 * Check for constant result
 */
#if defined(COMPILER_GCC) || defined(COMPILER_CLANG)
#define VECOPS_IS_CONST_RESULT(x) (__builtin_constant_p(x))
#else
#define VECOPS_IS_CONST_RESULT(x) (0)
#endif


#endif //VECOPS_COREDEFS_H
