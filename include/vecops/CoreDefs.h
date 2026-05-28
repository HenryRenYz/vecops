//
// Created by renyz on 2026/3/13.
//

#ifndef VECOPS_COREDEFS_H
#define VECOPS_COREDEFS_H

#include <csignal>
#include <cstdint>  // for standard int defs

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
#elif defined(COMPILER_MSVC)
  #define VECOPS_NOINLINE __declspec(noinline)
  #define VECOPS_ALWAYS_INLINE __forceinline
#else
  #define VECOPS_NOINLINE
  #define VECOPS_ALWAYS_INLINE inline
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


namespace vecops {

/**
 * Float types
 */
using bfloat16_t = __bf16;
#if defined(__arm__) || defined(__aarch64__)
using float16_t = __fp16;
#else
using float16_t = _Float16;
#endif
using float32_t = float;
using float64_t = double;
/**
 * Signed native int, having the same width as machine word.
 */
using nint_t = ptrdiff_t;
/**
 * Unsigned native int, having the same width as machine word.
 */
using nuint_t = size_t;

template <typename T>
struct TypeTraits {
  static constexpr bool is_integer = std::is_integral_v<T>;
  static constexpr bool is_signed = std::is_signed_v<T>;
  static constexpr bool is_float = std::is_floating_point_v<T>;
  static constexpr size_t bits = sizeof(T) * 8;
  static constexpr bool is_bfloat16 = false;
  static constexpr bool is_float16 = false;
};

template <>
struct TypeTraits<bfloat16_t> {
  static constexpr bool is_integer = false;
  static constexpr bool is_signed = true;
  static constexpr bool is_float = true;
  static constexpr size_t bits = 16;
  static constexpr bool is_bfloat16 = true;
  static constexpr bool is_float16 = false;
};

template <>
struct TypeTraits<float16_t> {
  static constexpr bool is_integer = false;
  static constexpr bool is_signed = true;
  static constexpr bool is_float = true;
  static constexpr size_t bits = 16;
  static constexpr bool is_bfloat16 = false;
  static constexpr bool is_float16 = true;
};

template <typename T> constexpr bool IsIntV = TypeTraits<T>::is_integer;
template <typename T> constexpr bool IsFloatV = TypeTraits<T>::is_float;
template <typename T> constexpr bool IsSignedV = TypeTraits<T>::is_signed;
template <typename T> constexpr bool IsBfloat16V = TypeTraits<T>::is_bfloat16;
template <typename T> constexpr bool IsFloat16V = TypeTraits<T>::is_float16;
template <typename T> constexpr bool IsStandardFloatV = IsFloatV<T> && !IsBfloat16V<T> && !IsFloat16V<T>;
template <typename T> constexpr size_t TypeBitsV = TypeTraits<T>::bits;


} // namespace vecops

#endif //VECOPS_COREDEFS_H
