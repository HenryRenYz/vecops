#ifndef VECOPS_VEC_DETAILS_X86_MATH_LOG_H
#define VECOPS_VEC_DETAILS_X86_MATH_LOG_H

/**
 * @file Log.h
 * @brief x86 backend implementations for the log family (all tiers).
 *
 * Transitional implementation: x86 offers no logarithm approximation
 * instruction to ladder the tiers against, so every tier evaluates the
 * IEEE libm logarithm per lane (through a float intermediate for the
 * narrow formats). That satisfies all documented tier contracts, including
 * Strict; the vector-native table/polynomial kernels remain future work
 * for this backend. Special inputs follow the shared log contract:
 * log(+-0) = -inf, log(+inf) = +inf, log(x < 0) = NaN, NaN propagates,
 * and log(1) = +0.
 */

#include <cmath>
#include <cstring>

#include "vecops/vec/details/Dispatch.h"
#include "vecops/vec/details/x86/Types.h"

namespace vecops::vec::details {

template <LogBase Base, typename T>
VECOPS_ALWAYS_INLINE T x86_log_lane(T value) {
  const double widened = static_cast<double>(value);
  const double result = [&] {
    if constexpr (Base == LogBase::E) return std::log(widened);
    else if constexpr (Base == LogBase::Base2) return std::log2(widened);
    else return std::log10(widened);
  }();
  if constexpr (
      std::same_as<T, bfloat16_t> || std::same_as<T, float16_t>)
    return static_cast<T>(static_cast<float>(result));
  else
    return static_cast<T>(result);
}

template <LogBase Base, Accuracy A>
struct X86LogWordImpl {
  template <nint_t Index, FloatingTag Tag>
  static VECOPS_ALWAYS_INLINE NativeWordVec<Tag> call(
      LogOp<Base, A>, Tag tag, NativeWordVec<Tag> value) {
    using T = ElementOf<Tag>;
    constexpr std::size_t lanes = sizeof(value.value) / sizeof(T);
    alignas(64) T buffer[lanes];
    std::memcpy(buffer, &value.value, sizeof(value.value));
    for (std::size_t lane = 0; lane < lanes; ++lane)
      buffer[lane] = x86_log_lane<Base>(buffer[lane]);
    NativeWordVec<Tag> result;
    std::memcpy(&result.value, buffer, sizeof(result.value));
    return result;
  }

  template <nint_t Index, FloatingTag Tag, typename Policy>
  static VECOPS_ALWAYS_INLINE NativeWordVec<Tag> call(
      LogOp<Base, A> op, Tag tag, NativeWordVec<Tag> value,
      NativeWordMask<Tag> mask, NativeWordVec<Tag> inactive, Policy) {
    const auto computed = call<Index>(op, tag, value);
    return execute_word<Index, X86Backend>(
        BlendOp{}, tag, inactive, mask, computed);
  }
};

template <LogBase Base, Accuracy A>
struct NativeWordImpl<X86Backend, LogOp<Base, A>> : X86LogWordImpl<Base, A> {
};

} // namespace vecops::vec::details

#endif // VECOPS_VEC_DETAILS_X86_MATH_LOG_H
