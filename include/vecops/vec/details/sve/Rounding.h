// SPDX-FileCopyrightText: 2026 HenryRenYz and vecops contributors
// SPDX-License-Identifier: MIT
#ifndef VECOPS_VEC_DETAILS_SVE_ROUNDING_H
#define VECOPS_VEC_DETAILS_SVE_ROUNDING_H

/**
 * @file Rounding.h
 * @brief SVE rounding using the base-SVE FRINTA/I/M/N/P/X/Z family, with
 * BF16 widening through FP32.
 */

#include "vecops/vec/details/Rounding.h"
#include "vecops/vec/details/sve/Arithmetic.h"
#include "vecops/vec/details/sve/Bf16.h"

namespace vecops::vec::details {

template <typename Op, Element T, typename Policy, typename Raw>
VECOPS_ALWAYS_INLINE Raw sve_rounding_native(
    svbool_t mask, Raw value, Raw inactive, Policy) {
#define VECOPS_VEC_SVE_ROUNDING_OP(Name, Suffix)                       \
  if constexpr (std::same_as<Policy, PreserveArithmeticInactive>)     \
    return sv##Name##_##Suffix##_m(value, mask, value);                \
  else if constexpr (std::same_as<Policy, ZeroArithmeticInactive>)    \
    return sv##Name##_##Suffix##_z(mask, value);                       \
  else                                                                 \
    return svsel_##Suffix(                                             \
        mask, sv##Name##_##Suffix##_x(mask, value), inactive)

#define VECOPS_VEC_SVE_ROUNDING_SUFFIX(Suffix)                         \
  if constexpr (std::same_as<Op, FloorOp>) {                           \
    VECOPS_VEC_SVE_ROUNDING_OP(rintm, Suffix);                         \
  } else if constexpr (std::same_as<Op, CeilOp>) {                    \
    VECOPS_VEC_SVE_ROUNDING_OP(rintp, Suffix);                         \
  } else if constexpr (std::same_as<Op, TruncOp>) {                   \
    VECOPS_VEC_SVE_ROUNDING_OP(rintz, Suffix);                         \
  } else if constexpr (std::same_as<Op, RoundOp>) {                   \
    VECOPS_VEC_SVE_ROUNDING_OP(rinta, Suffix);                         \
  } else if constexpr (std::same_as<Op, RoundEvenOp>) {               \
    VECOPS_VEC_SVE_ROUNDING_OP(rintn, Suffix);                         \
  } else if constexpr (std::same_as<Op, NearbyIntOp>) {               \
    VECOPS_VEC_SVE_ROUNDING_OP(rinti, Suffix);                         \
  } else if constexpr (std::same_as<Op, RintOp>) {                    \
    VECOPS_VEC_SVE_ROUNDING_OP(rintx, Suffix);                         \
  } else {                                                             \
    static_assert(dispatch_dependent_false<Op>);                       \
  }

  if constexpr (std::same_as<T, float16_t>) {
    VECOPS_VEC_SVE_ROUNDING_SUFFIX(f16);
  } else if constexpr (std::same_as<T, float32_t>) {
    VECOPS_VEC_SVE_ROUNDING_SUFFIX(f32);
  } else if constexpr (std::same_as<T, float64_t>) {
    VECOPS_VEC_SVE_ROUNDING_SUFFIX(f64);
  } else {
    static_assert(dispatch_dependent_false<Raw>);
  }

#undef VECOPS_VEC_SVE_ROUNDING_SUFFIX
#undef VECOPS_VEC_SVE_ROUNDING_OP
}

template <typename Op>
VECOPS_ALWAYS_INLINE svbfloat16_t sve_bfloat16_rounding(
    svbfloat16_t value, svbool_t mask) {
  const auto low_mask = svunpklo_b(mask);
  const auto high_mask = svunpkhi_b(mask);
  const auto low = sve_bf16_to_f32_lo(value);
  const auto high = sve_bf16_to_f32_hi(value);
  return sve_f32_pair_to_bf16(
      sve_rounding_native<Op, float32_t>(
          low_mask, low, low, PreserveArithmeticInactive{}),
      sve_rounding_native<Op, float32_t>(
          high_mask, high, high, PreserveArithmeticInactive{}));
}

template <typename Op>
struct SVERoundingWordImpl {
  template <nint_t Index, FloatingTag Tag>
  static VECOPS_ALWAYS_INLINE NativeWordVec<Tag> call(
      Op op, Tag tag, NativeWordVec<Tag> value) {
    return call<Index>(
        op, tag, value, svptrue_b8(), value,
        PreserveArithmeticInactive{});
  }

  template <nint_t Index, FloatingTag Tag, typename Policy>
  static VECOPS_ALWAYS_INLINE NativeWordVec<Tag> call(
      Op, Tag tag, NativeWordVec<Tag> value,
      NativeWordMask<Tag> mask, NativeWordVec<Tag> inactive,
      Policy policy) {
    using T = ElementOf<Tag>;
    static_assert(Index >= 0 && Index < num_words(tag));
    if constexpr (std::same_as<T, bfloat16_t>) {
      const auto computed = sve_bfloat16_rounding<Op>(
          sve_basic_raw_word(value), mask);
      return sve_basic_wrap_word<Tag>(svreinterpret_bf16_u16(svsel_u16(
          mask, svreinterpret_u16_bf16(computed),
          svreinterpret_u16_bf16(sve_basic_raw_word(inactive)))));
    } else {
      return sve_basic_wrap_word<Tag>(sve_rounding_native<Op, T>(
          mask, sve_basic_raw_word(value),
          sve_basic_raw_word(inactive), policy));
    }
  }
};

#define VECOPS_VEC_REGISTER_SVE_ROUNDING(OpType)                       \
  template <>                                                          \
  struct NativeWordImpl<SVEBackend, OpType>                            \
      : SVERoundingWordImpl<OpType> {}

VECOPS_VEC_REGISTER_SVE_ROUNDING(FloorOp);
VECOPS_VEC_REGISTER_SVE_ROUNDING(CeilOp);
VECOPS_VEC_REGISTER_SVE_ROUNDING(TruncOp);
VECOPS_VEC_REGISTER_SVE_ROUNDING(RoundOp);
VECOPS_VEC_REGISTER_SVE_ROUNDING(RoundEvenOp);
VECOPS_VEC_REGISTER_SVE_ROUNDING(NearbyIntOp);
VECOPS_VEC_REGISTER_SVE_ROUNDING(RintOp);

#undef VECOPS_VEC_REGISTER_SVE_ROUNDING

} // namespace vecops::vec::details

#endif // VECOPS_VEC_DETAILS_SVE_ROUNDING_H
