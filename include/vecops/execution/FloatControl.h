// SPDX-FileCopyrightText: 2026 HenryRenYz and vecops contributors
// SPDX-License-Identifier: MIT

#ifndef VECOPS_EXECUTION_FLOAT_CONTROL_H
#define VECOPS_EXECUTION_FLOAT_CONTROL_H

#include <cstdlib>
#include <cstdint>

#if defined(__x86_64__) || defined(_M_X64)
#  include <xmmintrin.h>
#endif

/**
 * @file vecops/execution/FloatControl.h
 * @brief Per-thread hardware control of subnormal floating-point values.
 *
 * FPCR (AArch64) and MXCSR (x86) belong to the calling thread. Set the
 * desired mode before creating worker threads, and call
 * apply_process_flush_subnormals() at a worker entry if workers may already
 * exist. Only the flush bits are changed; rounding and exception controls
 * retain their previous values.
 */
namespace vecops::execution {

[[nodiscard]] constexpr bool flush_subnormals_supported() noexcept {
#if defined(__aarch64__) || defined(__x86_64__) || defined(_M_X64)
  return true;
#else
  return false;
#endif
}

[[nodiscard]] inline bool current_thread_flush_subnormals() noexcept {
#if defined(__aarch64__)
  std::uint64_t fpcr;
  __asm__ __volatile__("mrs %0, fpcr" : "=r"(fpcr));
  constexpr std::uint64_t mask = (1ULL << 24) | (1ULL << 19);
  return (fpcr & mask) == mask;
#elif defined(__x86_64__) || defined(_M_X64)
  constexpr unsigned mask = (1U << 15) | (1U << 6);
  return (_mm_getcsr() & mask) == mask;
#else
  return false;
#endif
}

inline void set_current_thread_flush_subnormals(bool enabled) noexcept {
#if defined(__aarch64__)
  std::uint64_t fpcr;
  __asm__ __volatile__("mrs %0, fpcr" : "=r"(fpcr));
  constexpr std::uint64_t mask = (1ULL << 24) | (1ULL << 19);
  const auto updated = enabled ? (fpcr | mask) : (fpcr & ~mask);
  if (updated != fpcr) {
    __asm__ __volatile__("msr fpcr, %0\n\tisb" : : "r"(updated) : "memory");
  }
#elif defined(__x86_64__) || defined(_M_X64)
  constexpr unsigned mask = (1U << 15) | (1U << 6);
  const unsigned mxcsr = _mm_getcsr();
  const unsigned updated = enabled ? (mxcsr | mask) : (mxcsr & ~mask);
  if (updated != mxcsr)
    _mm_setcsr(updated);
#else
  (void)enabled;
#endif
}

/**
 * Apply VECOPS_FLUSH_SUBNORMALS=1 or 0 on the calling worker thread.
 * An unset value leaves its hardware mode unchanged. Reading the process
 * environment on each task also covers pools created before configuration.
 */
inline void apply_process_flush_subnormals() noexcept {
  const char* value = std::getenv("VECOPS_FLUSH_SUBNORMALS");
  if (value != nullptr && value[1] == '\0' && (value[0] == '0' || value[0] == '1'))
    set_current_thread_flush_subnormals(value[0] == '1');
}

} // namespace vecops::execution

#endif // VECOPS_EXECUTION_FLOAT_CONTROL_H
