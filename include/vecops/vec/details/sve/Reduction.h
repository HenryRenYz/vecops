// SPDX-FileCopyrightText: 2026 HenryRenYz and vecops contributors
// SPDX-License-Identifier: MIT
#ifndef VECOPS_VEC_DETAILS_SVE_REDUCTION_H
#define VECOPS_VEC_DETAILS_SVE_REDUCTION_H

/**
 * @file Reduction.h
 * @brief SVE backend implementations for reduction operations (reduce_add, reduce_max, reduce_min).
 */

#include "vecops/vec/details/sve/Arithmetic.h"

namespace vecops::vec::details {


/* **************************************************************************** */
//                       Reduction word implementations                       //
/* **************************************************************************** */

template <typename Op, typename T, typename Raw>
VECOPS_ALWAYS_INLINE T sve_reduction_raw(Raw value, svbool_t active) {
  if constexpr (std::same_as<T, bfloat16_t>) {
    const auto low_value = sve_bf16_to_f32_lo(value);
    const auto high_value = sve_bf16_to_f32_hi(value);
    const auto low_mask = svunpklo_b(active);
    const auto high_mask = svunpkhi_b(active);
    if constexpr (std::same_as<Op, ReduceAddOp>) {
      return static_cast<T>(
          svaddv_f32(low_mask, low_value) +
          svaddv_f32(high_mask, high_value));
    } else if constexpr (std::same_as<Op, ReduceMaxOp>) {
      const auto low = svmaxv_f32(low_mask, low_value);
      const auto high = svmaxv_f32(high_mask, high_value);
      return static_cast<T>(low > high ? low : high);
    } else {
      const auto low = svminv_f32(low_mask, low_value);
      const auto high = svminv_f32(high_mask, high_value);
      return static_cast<T>(low < high ? low : high);
    }
  } else if constexpr (std::same_as<T, float16_t>) {
    if constexpr (std::same_as<Op, ReduceAddOp>) return T{svaddv_f16(active, value)};
    else if constexpr (std::same_as<Op, ReduceMaxOp>) return T{svmaxv_f16(active, value)};
    else return T{svminv_f16(active, value)};
  } else if constexpr (std::same_as<T, float32_t>) {
    if constexpr (std::same_as<Op, ReduceAddOp>) return svaddv_f32(active, value);
    else if constexpr (std::same_as<Op, ReduceMaxOp>) return svmaxv_f32(active, value);
    else return svminv_f32(active, value);
  } else if constexpr (std::same_as<T, float64_t>) {
    if constexpr (std::same_as<Op, ReduceAddOp>) return svaddv_f64(active, value);
    else if constexpr (std::same_as<Op, ReduceMaxOp>) return svmaxv_f64(active, value);
    else return svminv_f64(active, value);
  } else if constexpr (std::same_as<T, int8_t>) {
    if constexpr (std::same_as<Op, ReduceAddOp>) return static_cast<T>(svaddv_s8(active, value));
    else if constexpr (std::same_as<Op, ReduceMaxOp>) return svmaxv_s8(active, value);
    else return svminv_s8(active, value);
  } else if constexpr (std::same_as<T, uint8_t>) {
    if constexpr (std::same_as<Op, ReduceAddOp>) return static_cast<T>(svaddv_u8(active, value));
    else if constexpr (std::same_as<Op, ReduceMaxOp>) return svmaxv_u8(active, value);
    else return svminv_u8(active, value);
  } else if constexpr (std::same_as<T, int16_t>) {
    if constexpr (std::same_as<Op, ReduceAddOp>) return static_cast<T>(svaddv_s16(active, value));
    else if constexpr (std::same_as<Op, ReduceMaxOp>) return svmaxv_s16(active, value);
    else return svminv_s16(active, value);
  } else if constexpr (std::same_as<T, uint16_t>) {
    if constexpr (std::same_as<Op, ReduceAddOp>) return static_cast<T>(svaddv_u16(active, value));
    else if constexpr (std::same_as<Op, ReduceMaxOp>) return svmaxv_u16(active, value);
    else return svminv_u16(active, value);
  } else if constexpr (std::same_as<T, int32_t>) {
    if constexpr (std::same_as<Op, ReduceAddOp>) return static_cast<T>(svaddv_s32(active, value));
    else if constexpr (std::same_as<Op, ReduceMaxOp>) return svmaxv_s32(active, value);
    else return svminv_s32(active, value);
  } else if constexpr (std::same_as<T, uint32_t>) {
    if constexpr (std::same_as<Op, ReduceAddOp>) return static_cast<T>(svaddv_u32(active, value));
    else if constexpr (std::same_as<Op, ReduceMaxOp>) return svmaxv_u32(active, value);
    else return svminv_u32(active, value);
  } else if constexpr (std::same_as<T, int64_t>) {
    if constexpr (std::same_as<Op, ReduceAddOp>) return static_cast<T>(svaddv_s64(active, value));
    else if constexpr (std::same_as<Op, ReduceMaxOp>) return svmaxv_s64(active, value);
    else return svminv_s64(active, value);
  } else if constexpr (std::same_as<T, uint64_t>) {
    if constexpr (std::same_as<Op, ReduceAddOp>) return static_cast<T>(svaddv_u64(active, value));
    else if constexpr (std::same_as<Op, ReduceMaxOp>) return svmaxv_u64(active, value);
    else return svminv_u64(active, value);
  } else {
    static_assert(
        dispatch_dependent_false<T>,
        "unsupported SVE reduce_add element type");
  }
}

template <typename Op>
struct SVEReductionWordImpl {
  template <nint_t Index, VectorTag Tag>
  static VECOPS_ALWAYS_INLINE ElementOf<Tag> call(
      Op, Tag tag, NativeWordVec<Tag> value) {
    using T = ElementOf<Tag>;
    const auto active = sve_prefix_predicate<T>(
        sve_valid_word_lanes<Index>(tag));
    return sve_reduction_raw<Op, T>(sve_basic_raw_word(value), active);
  }

  template <nint_t Index, VectorTag Tag>
  static VECOPS_ALWAYS_INLINE ElementOf<Tag> call(
      Op, Tag tag, NativeWordVec<Tag> value,
      NativeWordMask<Tag> mask) {
    using T = ElementOf<Tag>;
    const auto valid = sve_prefix_predicate<T>(
        sve_valid_word_lanes<Index>(tag));
    const auto active = svand_b_z(valid, mask, valid);
    return sve_reduction_raw<Op, T>(sve_basic_raw_word(value), active);
  }
};

template <>
struct NativeWordImpl<SVEBackend, ReduceAddOp>
    : SVEReductionWordImpl<ReduceAddOp> {};
template <>
struct NativeWordImpl<SVEBackend, ReduceMaxOp>
    : SVEReductionWordImpl<ReduceMaxOp> {};
template <>
struct NativeWordImpl<SVEBackend, ReduceMinOp>
    : SVEReductionWordImpl<ReduceMinOp> {};

} // namespace vecops::vec::details

#endif // VECOPS_VEC_DETAILS_SVE_REDUCTION_H
