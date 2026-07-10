//
// Created by renyz on 2026/6/1.
//

#ifndef VECOPS_COREDEFS_H
#define VECOPS_COREDEFS_H

/**
 * Define VECOPS_PRESERVE_SUBNORMALS consistently for every translation unit
 * to make strict vector operations preserve representable subnormal results.
 * The default build permits hardware FTZ/DAZ behavior for maximum throughput.
 */

#include <csignal>

#include "vecops/Features.h"

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
#if VECOPS_RELEASE
#define VECOPS_INLINE VECOPS_ALWAYS_INLINE
#else
#define VECOPS_INLINE inline
#endif

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
 * Unroll pragma for loop
 */
#if defined(COMPILER_GCC) || defined(COMPILER_CLANG)
#define VECOPS_UNROLL _Pragma("GCC unroll 16")
#elif defined(COMPILER_MSVC)
#define VECOPS_UNROLL __pragma(loop(unroll))
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
