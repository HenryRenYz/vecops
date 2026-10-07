// SPDX-FileCopyrightText: 2026 HenryRenYz and vecops contributors
// SPDX-License-Identifier: MIT
//
//

#ifndef VECOPS_VEC_DETAILS_SME_STATE_H
#define VECOPS_VEC_DETAILS_SME_STATE_H

#include <cstdint>
#include <functional>
#include <type_traits>
#include <utility>

#include "vecops/Assertion.h"
#include "vecops/CoreDefs.h"
#include "vecops/Meta.h"

#if !defined(__aarch64__) || !defined(HAS_SME)
#error "Manual SME state management requires an AArch64 SME target"
#endif

// A no_sanitize function attribute is itself an optimization boundary in the
// supported Clang and prevents ordinary callees from being folded into the
// owner. Reject instrumented translation units instead: silently accepting
// one could insert a runtime call while SM and ZA are active.
#if defined(__has_feature)
#if __has_feature(address_sanitizer) || __has_feature(thread_sanitizer) || \
    __has_feature(memory_sanitizer) || __has_feature(undefined_behavior_sanitizer)
#error "Manual SME regions cannot be compiled with sanitizer instrumentation"
#endif
#endif
#if defined(__SANITIZE_ADDRESS__) || defined(__SANITIZE_THREAD__) || \
    defined(__SANITIZE_UNDEFINED__)
#error "Manual SME regions cannot be compiled with sanitizer instrumentation"
#endif

namespace vecops::vec::details::sme {

template <typename T>
VECOPS_ALWAYS_INLINE constexpr T min_value(T a, T b) noexcept {
  return b < a ? b : a;
}

template <typename T>
VECOPS_ALWAYS_INLINE constexpr T max_value(T a, T b) noexcept {
  return a < b ? b : a;
}

template <typename T>
VECOPS_ALWAYS_INLINE constexpr T clamp_value(T value, T lo, T hi) noexcept {
  return min_value(max_value(value, lo), hi);
}

#define VECOPS_SME_ZP_CLOBBERS                                           \
  "z0", "z1", "z2", "z3", "z4", "z5", "z6", "z7",             \
  "z8", "z9", "z10", "z11", "z12", "z13", "z14", "z15",      \
  "z16", "z17", "z18", "z19", "z20", "z21", "z22", "z23",    \
  "z24", "z25", "z26", "z27", "z28", "z29", "z30", "z31",    \
  "p0", "p1", "p2", "p3", "p4", "p5", "p6", "p7",             \
  "p8", "p9", "p10", "p11", "p12", "p13", "p14", "p15"

#if defined(COMPILER_CLANG)
#define VECOPS_MANUAL_SME_OWNER_ATTRIBUTES                               \
  __attribute__((noinline, no_instrument_function))
#elif defined(COMPILER_GCC)
// GCC's shrink-wrapping may clone the state-closing inline asm into multiple
// mutually exclusive return blocks.  That is dynamically correct, but it
// obscures the one-owner/one-boundary contract and defeats the static
// disassembly gate.  Keep one conventional prologue/epilogue for this very
// small set of state-owner functions; the inlined compute body is unchanged.
#define VECOPS_MANUAL_SME_OWNER_ATTRIBUTES                               \
  __attribute__((noinline, no_instrument_function, optimize("no-shrink-wrap")))
#else
#define VECOPS_MANUAL_SME_OWNER_ATTRIBUTES VECOPS_NOINLINE
#endif

/** Read the architectural Streaming Vector Length in bytes. */
VECOPS_ALWAYS_INLINE nint_t streaming_vector_bytes() noexcept {
  nint_t bytes;
  asm volatile("rdsvl %0, #1" : "=r"(bytes));
  return bytes;
}

/**
 * Return SVL bytes while retaining every build-time architectural guarantee.
 * An architectural SVL is a multiple of 16 bytes in [16, 256].
 */
#if defined(HAS_FIXED_STREAMING_SVE_BITS)
VECOPS_ALWAYS_INLINE constexpr auto streaming_vector_bytes_value() noexcept {
  return meta::cint<FIXED_STREAMING_SVE_BITS / 8>;
}

template <typename T>
/** Return streaming lanes as typed integer metadata. */
VECOPS_ALWAYS_INLINE constexpr auto streaming_lanes_value() noexcept {
  static_assert(sizeof(T) > 0);
  return streaming_vector_bytes_value() / meta::cint<sizeof(T)>;
}
#else
VECOPS_ALWAYS_INLINE auto streaming_vector_bytes_value() noexcept {
  return meta::dyn<16, 16, 256>(streaming_vector_bytes());
}

template <typename T>
/** Return streaming lanes as typed integer metadata. */
VECOPS_ALWAYS_INLINE auto streaming_lanes_value() noexcept {
  static_assert(sizeof(T) > 0);
  return streaming_vector_bytes_value() / meta::cint<sizeof(T)>;
}
#endif

template <typename T>
/** Return the number of T elements in one streaming vector. */
VECOPS_ALWAYS_INLINE nint_t streaming_lanes() noexcept {
  return static_cast<nint_t>(streaming_lanes_value<T>());
}

VECOPS_ALWAYS_INLINE std::uint64_t read_svcr() noexcept {
  std::uint64_t value;
  asm volatile("mrs %0, svcr" : "=r"(value));
  return value;
}

/**
 * Own one destructive, non-nestable Streaming+ZA interval.
 *
 * This is a compiler-specific backend contract rather than an ACLE function
 * interface. The callback and every function below it must actually inline;
 * disassembly tests enforce that no call remains between the two boundaries.
 */
template <typename Fn>
VECOPS_MANUAL_SME_OWNER_ATTRIBUTES void with_streaming_za(Fn&& fn) noexcept {
  static_assert(std::is_void_v<std::invoke_result_t<Fn&&>>,
                "an SME region cannot return state across SMSTOP");
  static_assert(std::is_nothrow_invocable_v<Fn&&>,
                "an SME region callback must be noexcept");
#if defined(VECOPS_DEBUG)
  VECOPS_ASSERT(read_svcr() == 0,
                "manual SME region requires SM=0 and ZA=0 on entry");
#endif
  asm volatile(
      "smstart\n\t"
      "isb"
      :
      :
      : VECOPS_SME_ZP_CLOBBERS, "memory");
  std::forward<Fn>(fn)();
  asm volatile(
      "smstop\n\t"
      "isb"
      :
      :
      : VECOPS_SME_ZP_CLOBBERS, "memory");
#if defined(VECOPS_DEBUG)
  VECOPS_ASSERT(read_svcr() == 0,
                "manual SME region did not restore SM=0 and ZA=0");
#endif
}

} // namespace vecops::vec::details::sme

#undef VECOPS_MANUAL_SME_OWNER_ATTRIBUTES
#undef VECOPS_SME_ZP_CLOBBERS

#endif // VECOPS_VEC_DETAILS_SME_STATE_H
