// SPDX-FileCopyrightText: 2026 HenryRenYz and vecops contributors
// SPDX-License-Identifier: MIT
#ifndef VECOPS_VEC_DETAILS_SVE_BIT_H
#define VECOPS_VEC_DETAILS_SVE_BIT_H

/**
 * @file Bit.h
 * @brief SVE backend implementations for bitwise operations, shifts, bit
 * counts, and rotations.
 */

#include <limits>
#include <type_traits>

#include "vecops/vec/details/sve/Basic.h"

namespace vecops::vec::details {


/* **************************************************************************** */
//                   Bitwise operation word implementations                   //
/* **************************************************************************** */

#define VECOPS_VEC_SVE_INTEGER_CASES(Name, ...)                         \
  if constexpr (std::same_as<T, int8_t>) return Name##_s8_x(__VA_ARGS__); \
  else if constexpr (std::same_as<T, uint8_t>)                          \
    return Name##_u8_x(__VA_ARGS__);                                     \
  else if constexpr (std::same_as<T, int16_t>)                          \
    return Name##_s16_x(__VA_ARGS__);                                    \
  else if constexpr (std::same_as<T, uint16_t>)                         \
    return Name##_u16_x(__VA_ARGS__);                                    \
  else if constexpr (std::same_as<T, int32_t>)                          \
    return Name##_s32_x(__VA_ARGS__);                                    \
  else if constexpr (std::same_as<T, uint32_t>)                         \
    return Name##_u32_x(__VA_ARGS__);                                    \
  else if constexpr (std::same_as<T, int64_t>)                          \
    return Name##_s64_x(__VA_ARGS__);                                    \
  else if constexpr (std::same_as<T, uint64_t>)                         \
    return Name##_u64_x(__VA_ARGS__);                                    \
  else static_assert(dispatch_dependent_false<T>,                       \
                     "SVE bit operation requires an integer element type")

template <typename T, typename Raw>
VECOPS_ALWAYS_INLINE auto sve_bit_and_raw(Raw a, Raw b) {
  VECOPS_VEC_SVE_INTEGER_CASES(svand, svptrue_b8(), a, b);
}

template <typename T, typename Raw>
VECOPS_ALWAYS_INLINE auto sve_bit_or_raw(Raw a, Raw b) {
  VECOPS_VEC_SVE_INTEGER_CASES(svorr, svptrue_b8(), a, b);
}

template <typename T, typename Raw>
VECOPS_ALWAYS_INLINE auto sve_bit_xor_raw(Raw a, Raw b) {
  VECOPS_VEC_SVE_INTEGER_CASES(sveor, svptrue_b8(), a, b);
}

template <typename T, typename Raw>
VECOPS_ALWAYS_INLINE auto sve_bit_andnot_raw(Raw a, Raw b) {
  VECOPS_VEC_SVE_INTEGER_CASES(svbic, svptrue_b8(), b, a);
}

template <typename T, typename Raw>
VECOPS_ALWAYS_INLINE auto sve_bit_not_raw(Raw value) {
  VECOPS_VEC_SVE_INTEGER_CASES(svnot, svptrue_b8(), value);
}

#define VECOPS_VEC_SVE_INTEGER_MERGE_CASES(Name, ...)                   \
  if constexpr (std::same_as<T, int8_t>) return Name##_s8_m(__VA_ARGS__); \
  else if constexpr (std::same_as<T, uint8_t>)                          \
    return Name##_u8_m(__VA_ARGS__);                                     \
  else if constexpr (std::same_as<T, int16_t>)                          \
    return Name##_s16_m(__VA_ARGS__);                                    \
  else if constexpr (std::same_as<T, uint16_t>)                         \
    return Name##_u16_m(__VA_ARGS__);                                    \
  else if constexpr (std::same_as<T, int32_t>)                          \
    return Name##_s32_m(__VA_ARGS__);                                    \
  else if constexpr (std::same_as<T, uint32_t>)                         \
    return Name##_u32_m(__VA_ARGS__);                                    \
  else if constexpr (std::same_as<T, int64_t>)                          \
    return Name##_s64_m(__VA_ARGS__);                                    \
  else if constexpr (std::same_as<T, uint64_t>)                         \
    return Name##_u64_m(__VA_ARGS__);                                    \
  else static_assert(dispatch_dependent_false<T>,                       \
                     "SVE bit operation requires an integer element type")

template <typename T, typename Raw>
VECOPS_ALWAYS_INLINE auto sve_bit_and_masked_raw(
    Raw a, Raw b, svbool_t mask) {
  VECOPS_VEC_SVE_INTEGER_MERGE_CASES(svand, mask, a, b);
}

template <typename T, typename Raw>
VECOPS_ALWAYS_INLINE auto sve_bit_or_masked_raw(
    Raw a, Raw b, svbool_t mask) {
  VECOPS_VEC_SVE_INTEGER_MERGE_CASES(svorr, mask, a, b);
}

template <typename T, typename Raw>
VECOPS_ALWAYS_INLINE auto sve_bit_xor_masked_raw(
    Raw a, Raw b, svbool_t mask) {
  VECOPS_VEC_SVE_INTEGER_MERGE_CASES(sveor, mask, a, b);
}

template <typename T, typename Raw>
VECOPS_ALWAYS_INLINE auto sve_bit_not_masked_raw(
    Raw inactive, Raw value, svbool_t mask) {
  VECOPS_VEC_SVE_INTEGER_MERGE_CASES(svnot, inactive, mask, value);
}

#undef VECOPS_VEC_SVE_INTEGER_MERGE_CASES

#define VECOPS_VEC_SVE_INTEGER_ZERO_CASES(Name, ...)                    \
  if constexpr (std::same_as<T, int8_t>) return Name##_s8_z(__VA_ARGS__); \
  else if constexpr (std::same_as<T, uint8_t>) return Name##_u8_z(__VA_ARGS__); \
  else if constexpr (std::same_as<T, int16_t>) return Name##_s16_z(__VA_ARGS__); \
  else if constexpr (std::same_as<T, uint16_t>) return Name##_u16_z(__VA_ARGS__); \
  else if constexpr (std::same_as<T, int32_t>) return Name##_s32_z(__VA_ARGS__); \
  else if constexpr (std::same_as<T, uint32_t>) return Name##_u32_z(__VA_ARGS__); \
  else if constexpr (std::same_as<T, int64_t>) return Name##_s64_z(__VA_ARGS__); \
  else if constexpr (std::same_as<T, uint64_t>) return Name##_u64_z(__VA_ARGS__); \
  else static_assert(dispatch_dependent_false<T>,                       \
                     "SVE bit operation requires an integer element type")

template <typename T, typename Raw>
VECOPS_ALWAYS_INLINE auto sve_bit_and_zero_raw(Raw a, Raw b, svbool_t mask) {
  VECOPS_VEC_SVE_INTEGER_ZERO_CASES(svand, mask, a, b);
}
template <typename T, typename Raw>
VECOPS_ALWAYS_INLINE auto sve_bit_or_zero_raw(Raw a, Raw b, svbool_t mask) {
  VECOPS_VEC_SVE_INTEGER_ZERO_CASES(svorr, mask, a, b);
}
template <typename T, typename Raw>
VECOPS_ALWAYS_INLINE auto sve_bit_xor_zero_raw(Raw a, Raw b, svbool_t mask) {
  VECOPS_VEC_SVE_INTEGER_ZERO_CASES(sveor, mask, a, b);
}
template <typename T, typename Raw>
VECOPS_ALWAYS_INLINE auto sve_bit_andnot_zero_raw(Raw a, Raw b, svbool_t mask) {
  VECOPS_VEC_SVE_INTEGER_ZERO_CASES(svbic, mask, b, a);
}
template <typename T, typename Raw>
VECOPS_ALWAYS_INLINE auto sve_bit_not_zero_raw(Raw value, svbool_t mask) {
  VECOPS_VEC_SVE_INTEGER_ZERO_CASES(svnot, mask, value);
}

#undef VECOPS_VEC_SVE_INTEGER_ZERO_CASES

template <nint_t Count, typename T, typename Raw>
VECOPS_ALWAYS_INLINE auto sve_bit_shift_left_immediate_raw(
    Raw value, svbool_t mask) {
  static_assert(Count >= 0);
  if constexpr (std::same_as<T, int8_t>)
    return svlsl_n_s8_m(mask, value, Count);
  else if constexpr (std::same_as<T, uint8_t>)
    return svlsl_n_u8_m(mask, value, Count);
  else if constexpr (std::same_as<T, int16_t>)
    return svlsl_n_s16_m(mask, value, Count);
  else if constexpr (std::same_as<T, uint16_t>)
    return svlsl_n_u16_m(mask, value, Count);
  else if constexpr (std::same_as<T, int32_t>)
    return svlsl_n_s32_m(mask, value, Count);
  else if constexpr (std::same_as<T, uint32_t>)
    return svlsl_n_u32_m(mask, value, Count);
  else if constexpr (std::same_as<T, int64_t>)
    return svlsl_n_s64_m(mask, value, Count);
  else if constexpr (std::same_as<T, uint64_t>)
    return svlsl_n_u64_m(mask, value, Count);
  else static_assert(
      dispatch_dependent_false<T>,
      "SVE bit shift requires an integer element type");
}

template <nint_t Count, typename T, typename Raw>
VECOPS_ALWAYS_INLINE auto sve_bit_shift_right_immediate_raw(
    Raw value, svbool_t mask) {
  static_assert(Count >= 0);
  if constexpr (std::same_as<T, int8_t>)
    return svasr_n_s8_m(mask, value, Count);
  else if constexpr (std::same_as<T, uint8_t>)
    return svlsr_n_u8_m(mask, value, Count);
  else if constexpr (std::same_as<T, int16_t>)
    return svasr_n_s16_m(mask, value, Count);
  else if constexpr (std::same_as<T, uint16_t>)
    return svlsr_n_u16_m(mask, value, Count);
  else if constexpr (std::same_as<T, int32_t>)
    return svasr_n_s32_m(mask, value, Count);
  else if constexpr (std::same_as<T, uint32_t>)
    return svlsr_n_u32_m(mask, value, Count);
  else if constexpr (std::same_as<T, int64_t>)
    return svasr_n_s64_m(mask, value, Count);
  else if constexpr (std::same_as<T, uint64_t>)
    return svlsr_n_u64_m(mask, value, Count);
  else static_assert(
      dispatch_dependent_false<T>,
      "SVE bit shift requires an integer element type");
}

template <typename T, typename Raw>
VECOPS_ALWAYS_INLINE auto sve_bit_shift_left_variable_raw(
    Raw value, Raw counts, svbool_t mask) {
  if constexpr (std::same_as<T, int8_t>)
    return svlsl_s8_x(mask, value, svreinterpret_u8_s8(counts));
  else if constexpr (std::same_as<T, uint8_t>)
    return svlsl_u8_x(mask, value, counts);
  else if constexpr (std::same_as<T, int16_t>)
    return svlsl_s16_x(mask, value, svreinterpret_u16_s16(counts));
  else if constexpr (std::same_as<T, uint16_t>)
    return svlsl_u16_x(mask, value, counts);
  else if constexpr (std::same_as<T, int32_t>)
    return svlsl_s32_x(mask, value, svreinterpret_u32_s32(counts));
  else if constexpr (std::same_as<T, uint32_t>)
    return svlsl_u32_x(mask, value, counts);
  else if constexpr (std::same_as<T, int64_t>)
    return svlsl_s64_x(mask, value, svreinterpret_u64_s64(counts));
  else if constexpr (std::same_as<T, uint64_t>)
    return svlsl_u64_x(mask, value, counts);
}

template <typename T, typename Raw>
VECOPS_ALWAYS_INLINE auto sve_bit_shift_right_variable_raw(
    Raw value, Raw counts, svbool_t mask) {
  if constexpr (std::same_as<T, int8_t>)
    return svasr_s8_x(mask, value, svreinterpret_u8_s8(counts));
  else if constexpr (std::same_as<T, uint8_t>)
    return svlsr_u8_x(mask, value, counts);
  else if constexpr (std::same_as<T, int16_t>)
    return svasr_s16_x(mask, value, svreinterpret_u16_s16(counts));
  else if constexpr (std::same_as<T, uint16_t>)
    return svlsr_u16_x(mask, value, counts);
  else if constexpr (std::same_as<T, int32_t>)
    return svasr_s32_x(mask, value, svreinterpret_u32_s32(counts));
  else if constexpr (std::same_as<T, uint32_t>)
    return svlsr_u32_x(mask, value, counts);
  else if constexpr (std::same_as<T, int64_t>)
    return svasr_s64_x(mask, value, svreinterpret_u64_s64(counts));
  else if constexpr (std::same_as<T, uint64_t>)
    return svlsr_u64_x(mask, value, counts);
}

template <typename T, typename Raw>
VECOPS_ALWAYS_INLINE auto sve_bit_unsigned_raw(Raw value) {
  if constexpr (std::same_as<T, int8_t>) return svreinterpret_u8_s8(value);
  else if constexpr (std::same_as<T, uint8_t>)
    return svorr_n_u8_x(svptrue_b8(), value, 0);
  else if constexpr (std::same_as<T, int16_t>)
    return svreinterpret_u16_s16(value);
  else if constexpr (std::same_as<T, uint16_t>)
    return svorr_n_u16_x(svptrue_b8(), value, 0);
  else if constexpr (std::same_as<T, int32_t>)
    return svreinterpret_u32_s32(value);
  else if constexpr (std::same_as<T, uint32_t>)
    return svorr_n_u32_x(svptrue_b8(), value, 0);
  else if constexpr (std::same_as<T, int64_t>)
    return svreinterpret_u64_s64(value);
  else return svorr_n_u64_x(svptrue_b8(), value, 0);
}

template <typename T, typename Raw>
VECOPS_ALWAYS_INLINE auto sve_bit_from_unsigned_raw(Raw value) {
  if constexpr (std::same_as<T, int8_t>) return svreinterpret_s8_u8(value);
  else if constexpr (std::same_as<T, int16_t>)
    return svreinterpret_s16_u16(value);
  else if constexpr (std::same_as<T, int32_t>)
    return svreinterpret_s32_u32(value);
  else if constexpr (std::same_as<T, int64_t>)
    return svreinterpret_s64_u64(value);
  else return value;
}

template <typename T, typename Raw>
VECOPS_ALWAYS_INLINE auto sve_bit_unsigned_not(Raw value) {
  if constexpr (sizeof(T) == 1)
    return svnot_u8_x(svptrue_b8(), value);
  else if constexpr (sizeof(T) == 2)
    return svnot_u16_x(svptrue_b8(), value);
  else if constexpr (sizeof(T) == 4)
    return svnot_u32_x(svptrue_b8(), value);
  else
    return svnot_u64_x(svptrue_b8(), value);
}

template <typename T, typename Raw>
VECOPS_ALWAYS_INLINE auto sve_bit_unsigned_popcount(Raw value) {
  if constexpr (sizeof(T) == 1)
    return svcnt_u8_x(svptrue_b8(), value);
  else if constexpr (sizeof(T) == 2)
    return svcnt_u16_x(svptrue_b8(), value);
  else if constexpr (sizeof(T) == 4)
    return svcnt_u32_x(svptrue_b8(), value);
  else
    return svcnt_u64_x(svptrue_b8(), value);
}

template <typename T, typename Raw>
VECOPS_ALWAYS_INLINE auto sve_bit_unsigned_clz(Raw value) {
  if constexpr (sizeof(T) == 1)
    return svclz_u8_x(svptrue_b8(), value);
  else if constexpr (sizeof(T) == 2)
    return svclz_u16_x(svptrue_b8(), value);
  else if constexpr (sizeof(T) == 4)
    return svclz_u32_x(svptrue_b8(), value);
  else
    return svclz_u64_x(svptrue_b8(), value);
}

template <typename T, typename Raw>
VECOPS_ALWAYS_INLINE auto sve_bit_unsigned_reverse(Raw value) {
  if constexpr (sizeof(T) == 1)
    return svrbit_u8_x(svptrue_b8(), value);
  else if constexpr (sizeof(T) == 2)
    return svrbit_u16_x(svptrue_b8(), value);
  else if constexpr (sizeof(T) == 4)
    return svrbit_u32_x(svptrue_b8(), value);
  else
    return svrbit_u64_x(svptrue_b8(), value);
}

template <typename Op, typename T, typename Raw>
VECOPS_ALWAYS_INLINE auto sve_bit_count_raw(Raw value) {
  auto bits = sve_bit_unsigned_raw<T>(value);
  if constexpr (
      std::same_as<Op, CountLeadingOneOp> ||
      std::same_as<Op, CountTrailingOneOp>)
    bits = sve_bit_unsigned_not<T>(bits);
  if constexpr (std::same_as<Op, PopCountOp>)
    return sve_bit_unsigned_popcount<T>(bits);
  else if constexpr (
      std::same_as<Op, CountLeadingZeroOp> ||
      std::same_as<Op, CountLeadingOneOp>)
    return sve_bit_unsigned_clz<T>(bits);
  else
    return sve_bit_unsigned_clz<T>(sve_bit_unsigned_reverse<T>(bits));
}

template <typename T, typename Policy, typename Raw>
VECOPS_ALWAYS_INLINE auto sve_bit_unsigned_popcount_masked(
    Raw bits, Raw inactive, svbool_t mask) {
  if constexpr (sizeof(T) == 1) {
    if constexpr (std::same_as<Policy, ZeroBitLanes>)
      return svcnt_u8_z(mask, bits);
    else return svcnt_u8_m(inactive, mask, bits);
  } else if constexpr (sizeof(T) == 2) {
    if constexpr (std::same_as<Policy, ZeroBitLanes>)
      return svcnt_u16_z(mask, bits);
    else return svcnt_u16_m(inactive, mask, bits);
  } else if constexpr (sizeof(T) == 4) {
    if constexpr (std::same_as<Policy, ZeroBitLanes>)
      return svcnt_u32_z(mask, bits);
    else return svcnt_u32_m(inactive, mask, bits);
  } else {
    if constexpr (std::same_as<Policy, ZeroBitLanes>)
      return svcnt_u64_z(mask, bits);
    else return svcnt_u64_m(inactive, mask, bits);
  }
}

template <typename T, typename Policy, typename Raw>
VECOPS_ALWAYS_INLINE auto sve_bit_unsigned_clz_masked(
    Raw bits, Raw inactive, svbool_t mask) {
  if constexpr (sizeof(T) == 1) {
    if constexpr (std::same_as<Policy, ZeroBitLanes>)
      return svclz_u8_z(mask, bits);
    else return svclz_u8_m(inactive, mask, bits);
  } else if constexpr (sizeof(T) == 2) {
    if constexpr (std::same_as<Policy, ZeroBitLanes>)
      return svclz_u16_z(mask, bits);
    else return svclz_u16_m(inactive, mask, bits);
  } else if constexpr (sizeof(T) == 4) {
    if constexpr (std::same_as<Policy, ZeroBitLanes>)
      return svclz_u32_z(mask, bits);
    else return svclz_u32_m(inactive, mask, bits);
  } else {
    if constexpr (std::same_as<Policy, ZeroBitLanes>)
      return svclz_u64_z(mask, bits);
    else return svclz_u64_m(inactive, mask, bits);
  }
}

template <typename Op, typename T, typename Policy, typename Raw,
          typename InactiveRaw>
VECOPS_ALWAYS_INLINE auto sve_bit_count_masked_raw(
    Raw value, InactiveRaw inactive, svbool_t mask) {
  auto bits = sve_bit_unsigned_raw<T>(value);
  const auto inactive_bits = sve_bit_unsigned_raw<T>(inactive);
  if constexpr (
      std::same_as<Op, CountLeadingOneOp> ||
      std::same_as<Op, CountTrailingOneOp>)
    bits = sve_bit_unsigned_not<T>(bits);
  if constexpr (std::same_as<Op, PopCountOp>)
    return sve_bit_unsigned_popcount_masked<T, Policy>(
        bits, inactive_bits, mask);
  else if constexpr (
      std::same_as<Op, CountLeadingZeroOp> ||
      std::same_as<Op, CountLeadingOneOp>)
    return sve_bit_unsigned_clz_masked<T, Policy>(
        bits, inactive_bits, mask);
  else
    return sve_bit_unsigned_clz_masked<T, Policy>(
        sve_bit_unsigned_reverse<T>(bits), inactive_bits, mask);
}

template <typename T, typename Raw>
VECOPS_ALWAYS_INLINE auto sve_bit_unsigned_and_n(
    Raw value, std::make_unsigned_t<T> bits) {
  if constexpr (sizeof(T) == 1)
    return svand_n_u8_x(svptrue_b8(), value, static_cast<uint8_t>(bits));
  else if constexpr (sizeof(T) == 2)
    return svand_n_u16_x(svptrue_b8(), value, static_cast<uint16_t>(bits));
  else if constexpr (sizeof(T) == 4)
    return svand_n_u32_x(svptrue_b8(), value, static_cast<uint32_t>(bits));
  else
    return svand_n_u64_x(svptrue_b8(), value, static_cast<uint64_t>(bits));
}

template <typename T, typename Raw>
VECOPS_ALWAYS_INLINE auto sve_bit_unsigned_neg(Raw value) {
  if constexpr (sizeof(T) == 1)
    return svsub_u8_x(svptrue_b8(), svdup_n_u8(0), value);
  else if constexpr (sizeof(T) == 2)
    return svsub_u16_x(svptrue_b8(), svdup_n_u16(0), value);
  else if constexpr (sizeof(T) == 4)
    return svsub_u32_x(svptrue_b8(), svdup_n_u32(0), value);
  else
    return svsub_u64_x(svptrue_b8(), svdup_n_u64(0), value);
}

template <typename T, typename Raw>
VECOPS_ALWAYS_INLINE auto sve_bit_unsigned_shl(Raw value, Raw counts) {
  if constexpr (sizeof(T) == 1)
    return svlsl_u8_x(svptrue_b8(), value, counts);
  else if constexpr (sizeof(T) == 2)
    return svlsl_u16_x(svptrue_b8(), value, counts);
  else if constexpr (sizeof(T) == 4)
    return svlsl_u32_x(svptrue_b8(), value, counts);
  else
    return svlsl_u64_x(svptrue_b8(), value, counts);
}

template <typename T, typename Raw>
VECOPS_ALWAYS_INLINE auto sve_bit_unsigned_shr(Raw value, Raw counts) {
  if constexpr (sizeof(T) == 1)
    return svlsr_u8_x(svptrue_b8(), value, counts);
  else if constexpr (sizeof(T) == 2)
    return svlsr_u16_x(svptrue_b8(), value, counts);
  else if constexpr (sizeof(T) == 4)
    return svlsr_u32_x(svptrue_b8(), value, counts);
  else
    return svlsr_u64_x(svptrue_b8(), value, counts);
}

template <nint_t Count, typename T, typename Raw>
VECOPS_ALWAYS_INLINE auto sve_bit_unsigned_shl_n(Raw value) {
  if constexpr (sizeof(T) == 1)
    return svlsl_n_u8_x(svptrue_b8(), value, Count);
  else if constexpr (sizeof(T) == 2)
    return svlsl_n_u16_x(svptrue_b8(), value, Count);
  else if constexpr (sizeof(T) == 4)
    return svlsl_n_u32_x(svptrue_b8(), value, Count);
  else
    return svlsl_n_u64_x(svptrue_b8(), value, Count);
}

template <nint_t Count, typename T, typename Raw>
VECOPS_ALWAYS_INLINE auto sve_bit_unsigned_shr_n(Raw value) {
  if constexpr (sizeof(T) == 1)
    return svlsr_n_u8_x(svptrue_b8(), value, Count);
  else if constexpr (sizeof(T) == 2)
    return svlsr_n_u16_x(svptrue_b8(), value, Count);
  else if constexpr (sizeof(T) == 4)
    return svlsr_n_u32_x(svptrue_b8(), value, Count);
  else
    return svlsr_n_u64_x(svptrue_b8(), value, Count);
}

template <typename T, typename Raw>
VECOPS_ALWAYS_INLINE auto sve_bit_unsigned_or(Raw a, Raw b) {
  if constexpr (sizeof(T) == 1)
    return svorr_u8_x(svptrue_b8(), a, b);
  else if constexpr (sizeof(T) == 2)
    return svorr_u16_x(svptrue_b8(), a, b);
  else if constexpr (sizeof(T) == 4)
    return svorr_u32_x(svptrue_b8(), a, b);
  else
    return svorr_u64_x(svptrue_b8(), a, b);
}

template <typename Op, typename T, typename Raw>
VECOPS_ALWAYS_INLINE auto sve_bit_rotate_variable_raw(
    Raw value, Raw counts) {
  using U = std::make_unsigned_t<T>;
  constexpr U lane_mask = std::numeric_limits<U>::digits - 1;
  const auto unsigned_value = sve_bit_unsigned_raw<T>(value);
  const auto unsigned_counts = sve_bit_unsigned_raw<T>(counts);
  const auto normalized = sve_bit_unsigned_and_n<T>(
      unsigned_counts, lane_mask);
  const auto inverse = sve_bit_unsigned_and_n<T>(
      sve_bit_unsigned_neg<T>(normalized), lane_mask);
  const auto left = sve_bit_unsigned_shl<T>(
      unsigned_value,
      std::same_as<Op, RotateLeftOp> ? normalized : inverse);
  const auto right = sve_bit_unsigned_shr<T>(
      unsigned_value,
      std::same_as<Op, RotateLeftOp> ? inverse : normalized);
  return sve_bit_from_unsigned_raw<T>(sve_bit_unsigned_or<T>(left, right));
}

#if defined(HAS_SVE2) || defined(HAS_SME)
template <typename Op, nint_t Count, typename T, typename Raw>
VECOPS_ALWAYS_INLINE auto sve_bit_rotate_immediate_raw(Raw value) {
  using U = std::make_unsigned_t<T>;
  constexpr nint_t width = std::numeric_limits<U>::digits;
  constexpr nint_t remainder = Count % width;
  constexpr int rotate = static_cast<int>(
      remainder < 0 ? remainder + width : remainder);
  constexpr int inverse = static_cast<int>(width) - rotate;
  if constexpr (rotate == 0) return value;
  const auto bits = sve_bit_unsigned_raw<T>(value);
  if constexpr (std::same_as<Op, RotateLeftOp>) {
    const auto low = sve_bit_unsigned_shr_n<inverse, T>(bits);
    if constexpr (sizeof(T) == 1)
      return sve_bit_from_unsigned_raw<T>(svsli_n_u8(low, bits, rotate));
    else if constexpr (sizeof(T) == 2)
      return sve_bit_from_unsigned_raw<T>(svsli_n_u16(low, bits, rotate));
    else if constexpr (sizeof(T) == 4)
      return sve_bit_from_unsigned_raw<T>(svsli_n_u32(low, bits, rotate));
    else
      return sve_bit_from_unsigned_raw<T>(svsli_n_u64(low, bits, rotate));
  } else {
    const auto high = sve_bit_unsigned_shl_n<inverse, T>(bits);
    if constexpr (sizeof(T) == 1)
      return sve_bit_from_unsigned_raw<T>(svsri_n_u8(high, bits, rotate));
    else if constexpr (sizeof(T) == 2)
      return sve_bit_from_unsigned_raw<T>(svsri_n_u16(high, bits, rotate));
    else if constexpr (sizeof(T) == 4)
      return sve_bit_from_unsigned_raw<T>(svsri_n_u32(high, bits, rotate));
    else
      return sve_bit_from_unsigned_raw<T>(svsri_n_u64(high, bits, rotate));
  }
}
#endif

#undef VECOPS_VEC_SVE_INTEGER_CASES

#define VECOPS_VEC_DEFINE_SVE_BIT_BINARY(                               \
    OpType, Helper, MaskedHelper, ZeroHelper)                            \
  template <>                                                            \
  struct NativeWordImpl<SVEBackend, OpType> {                            \
    template <nint_t Index, IntegerTag Tag>                              \
    static VECOPS_ALWAYS_INLINE NativeWordVec<Tag> call(                 \
        OpType, Tag, NativeWordVec<Tag> a, NativeWordVec<Tag> b) {       \
      using Traits = RepresentationTraits<SVEBackend, Tag>;             \
      using T = ElementOf<Tag>;                                          \
      static_assert(Index >= 0 && Index < Traits::word_count);           \
      return sve_basic_wrap_word<Tag>(                                  \
          Helper<T>(sve_basic_raw_word(a), sve_basic_raw_word(b)));      \
    }                                                                    \
    template <nint_t Index, IntegerTag Tag, typename Policy>             \
    static VECOPS_ALWAYS_INLINE NativeWordVec<Tag> call(                 \
        OpType op, Tag tag, NativeWordVec<Tag> a, NativeWordVec<Tag> b, \
        NativeWordMask<Tag> mask, NativeWordVec<Tag> inactive, Policy) {\
      using T = ElementOf<Tag>;                                          \
      static_assert(Index >= 0 && Index < num_words(tag));               \
      if constexpr (std::same_as<Policy, PreserveBitLanes>)             \
        return sve_basic_wrap_word<Tag>(MaskedHelper<T>(                \
            sve_basic_raw_word(a), sve_basic_raw_word(b), mask));        \
      else if constexpr (std::same_as<Policy, ZeroBitLanes>)            \
        return sve_basic_wrap_word<Tag>(ZeroHelper<T>(                  \
            sve_basic_raw_word(a), sve_basic_raw_word(b), mask));        \
      else {                                                             \
        const auto computed = call<Index>(op, tag, a, b);               \
        return blend(tag, inactive, mask, computed);                   \
      }                                                                  \
    }                                                                    \
  }

VECOPS_VEC_DEFINE_SVE_BIT_BINARY(
    BitAndOp, sve_bit_and_raw, sve_bit_and_masked_raw, sve_bit_and_zero_raw);
VECOPS_VEC_DEFINE_SVE_BIT_BINARY(
    BitOrOp, sve_bit_or_raw, sve_bit_or_masked_raw, sve_bit_or_zero_raw);
VECOPS_VEC_DEFINE_SVE_BIT_BINARY(
    BitXorOp, sve_bit_xor_raw, sve_bit_xor_masked_raw, sve_bit_xor_zero_raw);

#undef VECOPS_VEC_DEFINE_SVE_BIT_BINARY

// bic computes b & ~a, so its merging form would preserve b, not the public
// contract's a. The old backend therefore used bic_x followed by one select.
template <>
struct NativeWordImpl<SVEBackend, BitAndNotOp> {
  template <nint_t Index, IntegerTag Tag>
  static VECOPS_ALWAYS_INLINE NativeWordVec<Tag> call(
      BitAndNotOp, Tag, NativeWordVec<Tag> a, NativeWordVec<Tag> b) {
    using T = ElementOf<Tag>;
    static_assert(Index >= 0 && Index < num_words(Tag{}));
    return sve_basic_wrap_word<Tag>(sve_bit_andnot_raw<T>(
        sve_basic_raw_word(a), sve_basic_raw_word(b)));
  }

  template <nint_t Index, IntegerTag Tag, typename Policy>
  static VECOPS_ALWAYS_INLINE NativeWordVec<Tag> call(
      BitAndNotOp op, Tag tag, NativeWordVec<Tag> a, NativeWordVec<Tag> b,
      NativeWordMask<Tag> mask, NativeWordVec<Tag> inactive, Policy) {
    using T = ElementOf<Tag>;
    if constexpr (std::same_as<Policy, ZeroBitLanes>) {
      return sve_basic_wrap_word<Tag>(sve_bit_andnot_zero_raw<T>(
          sve_basic_raw_word(a), sve_basic_raw_word(b), mask));
    }
    const auto computed = call<Index>(op, tag, a, b);
    return blend(tag, inactive, mask, computed);
  }
};

template <>
struct NativeWordImpl<SVEBackend, BitNotOp> {
  template <nint_t Index, IntegerTag Tag>
  static VECOPS_ALWAYS_INLINE NativeWordVec<Tag> call(
      BitNotOp, Tag, NativeWordVec<Tag> value) {
    using Traits = RepresentationTraits<SVEBackend, Tag>;
    using T = ElementOf<Tag>;
    static_assert(Index >= 0 && Index < Traits::word_count);
    return sve_basic_wrap_word<Tag>(
        sve_bit_not_raw<T>(sve_basic_raw_word(value)));
  }

  template <nint_t Index, IntegerTag Tag, typename Policy>
  static VECOPS_ALWAYS_INLINE NativeWordVec<Tag> call(
      BitNotOp, Tag tag, NativeWordVec<Tag> value,
      NativeWordMask<Tag> mask, NativeWordVec<Tag> inactive, Policy) {
    using T = ElementOf<Tag>;
    static_assert(Index >= 0 && Index < num_words(tag));
    if constexpr (std::same_as<Policy, ZeroBitLanes>)
      return sve_basic_wrap_word<Tag>(sve_bit_not_zero_raw<T>(
          sve_basic_raw_word(value), mask));
    else if constexpr (std::same_as<Policy, PreserveBitLanes>)
      return sve_basic_wrap_word<Tag>(sve_bit_not_masked_raw<T>(
          sve_basic_raw_word(value), sve_basic_raw_word(value), mask));
    else {
      return sve_basic_wrap_word<Tag>(sve_bit_not_masked_raw<T>(
          sve_basic_raw_word(inactive), sve_basic_raw_word(value), mask));
    }
  }
};

#define VECOPS_VEC_DEFINE_SVE_BIT_COUNT(OpType)                         \
  template <>                                                           \
  struct NativeWordImpl<SVEBackend, OpType> {                           \
    template <nint_t Index, IntegerTag Tag>                             \
    static VECOPS_ALWAYS_INLINE NativeWordVec<Tag> call(                \
        OpType, Tag, NativeWordVec<Tag> value) {                        \
      using Traits = RepresentationTraits<SVEBackend, Tag>;            \
      using T = ElementOf<Tag>;                                         \
      static_assert(Index >= 0 && Index < Traits::word_count);          \
      return sve_basic_wrap_word<Tag>(sve_bit_from_unsigned_raw<T>(     \
          sve_bit_count_raw<OpType, T>(sve_basic_raw_word(value))));    \
    }                                                                   \
    template <nint_t Index, IntegerTag Tag, typename Policy>            \
    static VECOPS_ALWAYS_INLINE NativeWordVec<Tag> call(                \
        OpType, Tag, NativeWordVec<Tag> value,                          \
        NativeWordMask<Tag> mask, NativeWordVec<Tag> inactive, Policy) {\
      using T = ElementOf<Tag>;                                         \
      return sve_basic_wrap_word<Tag>(sve_bit_from_unsigned_raw<T>(     \
          sve_bit_count_masked_raw<OpType, T, Policy>(                  \
              sve_basic_raw_word(value),                                \
              sve_basic_raw_word(inactive), mask)));                    \
    }                                                                   \
  }

VECOPS_VEC_DEFINE_SVE_BIT_COUNT(PopCountOp);
VECOPS_VEC_DEFINE_SVE_BIT_COUNT(CountLeadingZeroOp);
VECOPS_VEC_DEFINE_SVE_BIT_COUNT(CountLeadingOneOp);
VECOPS_VEC_DEFINE_SVE_BIT_COUNT(CountTrailingZeroOp);
VECOPS_VEC_DEFINE_SVE_BIT_COUNT(CountTrailingOneOp);

#undef VECOPS_VEC_DEFINE_SVE_BIT_COUNT

template <>
struct NativeWordImpl<SVEBackend, BitShiftLeftOp> {
  template <nint_t Index, IntegerTag Tag, nint_t Count>
  static VECOPS_ALWAYS_INLINE NativeWordVec<Tag> call(
      BitShiftLeftOp, Tag tag, NativeWordVec<Tag> value,
      meta::Const<Count>) {
    using Traits = RepresentationTraits<SVEBackend, Tag>;
    using T = ElementOf<Tag>;
    using U = std::make_unsigned_t<T>;
    constexpr nint_t width = std::numeric_limits<U>::digits;
    static_assert(Count >= 0);
    static_assert(Index >= 0 && Index < Traits::word_count);
    if constexpr (Count >= width) return fill_word(tag, T{});
    else return sve_basic_wrap_word<Tag>(
        sve_bit_shift_left_immediate_raw<Count, T>(
            sve_basic_raw_word(value), svptrue_b8()));
  }

  template <nint_t Index, IntegerTag Tag>
  static VECOPS_ALWAYS_INLINE NativeWordVec<Tag> call(
      BitShiftLeftOp, Tag tag, NativeWordVec<Tag> value, int count) {
    using Traits = RepresentationTraits<SVEBackend, Tag>;
    using T = ElementOf<Tag>;
    using U = std::make_unsigned_t<T>;
    constexpr int width = std::numeric_limits<U>::digits;
    const auto counts = fill_word(
        tag, static_cast<T>(count < width ? count : width));
    static_assert(Index >= 0 && Index < Traits::word_count);
    return sve_basic_wrap_word<Tag>(sve_bit_shift_left_variable_raw<T>(
        sve_basic_raw_word(value), sve_basic_raw_word(counts),
        svptrue_b8()));
  }

  template <nint_t Index, IntegerTag Tag>
  static VECOPS_ALWAYS_INLINE NativeWordVec<Tag> call(
      BitShiftLeftOp, Tag, NativeWordVec<Tag> value,
      NativeWordVec<Tag> counts) {
    using T = ElementOf<Tag>;
    static_assert(Index >= 0 && Index < num_words(Tag{}));
    return sve_basic_wrap_word<Tag>(sve_bit_shift_left_variable_raw<T>(
        sve_basic_raw_word(value), sve_basic_raw_word(counts),
        svptrue_b8()));
  }

  template <nint_t Index, IntegerTag Tag, nint_t Count, typename Policy>
  static VECOPS_ALWAYS_INLINE NativeWordVec<Tag> call(
      BitShiftLeftOp op, Tag tag, NativeWordVec<Tag> value,
      meta::Const<Count> count, NativeWordMask<Tag> mask,
      NativeWordVec<Tag> inactive, Policy) {
    const auto computed = call<Index>(op, tag, value, count);
    return blend(tag, inactive, mask, computed);
  }

  template <nint_t Index, IntegerTag Tag, typename Policy>
  static VECOPS_ALWAYS_INLINE NativeWordVec<Tag> call(
      BitShiftLeftOp op, Tag tag, NativeWordVec<Tag> value, int count,
      NativeWordMask<Tag> mask, NativeWordVec<Tag> inactive, Policy) {
    const auto computed = call<Index>(op, tag, value, count);
    return blend(tag, inactive, mask, computed);
  }

  template <nint_t Index, IntegerTag Tag, typename Policy>
  static VECOPS_ALWAYS_INLINE NativeWordVec<Tag> call(
      BitShiftLeftOp op, Tag tag, NativeWordVec<Tag> value,
      NativeWordVec<Tag> counts, NativeWordMask<Tag> mask,
      NativeWordVec<Tag> inactive, Policy) {
    const auto computed = call<Index>(op, tag, value, counts);
    return blend(tag, inactive, mask, computed);
  }
};

template <>
struct NativeWordImpl<SVEBackend, BitShiftRightOp> {
  template <nint_t Index, IntegerTag Tag, nint_t Count>
  static VECOPS_ALWAYS_INLINE NativeWordVec<Tag> call(
      BitShiftRightOp, Tag tag, NativeWordVec<Tag> value,
      meta::Const<Count>) {
    using Traits = RepresentationTraits<SVEBackend, Tag>;
    using T = ElementOf<Tag>;
    using U = std::make_unsigned_t<T>;
    constexpr nint_t width = std::numeric_limits<U>::digits;
    static_assert(Count >= 0);
    static_assert(Index >= 0 && Index < Traits::word_count);
    if constexpr (Count >= width) {
      if constexpr (std::is_unsigned_v<T>) return fill_word(tag, T{});
      else return sve_basic_wrap_word<Tag>(
          sve_bit_shift_right_immediate_raw<width - 1, T>(
              sve_basic_raw_word(value), svptrue_b8()));
    } else {
      return sve_basic_wrap_word<Tag>(
          sve_bit_shift_right_immediate_raw<Count, T>(
              sve_basic_raw_word(value), svptrue_b8()));
    }
  }

  template <nint_t Index, IntegerTag Tag>
  static VECOPS_ALWAYS_INLINE NativeWordVec<Tag> call(
      BitShiftRightOp, Tag tag, NativeWordVec<Tag> value, int count) {
    using Traits = RepresentationTraits<SVEBackend, Tag>;
    using T = ElementOf<Tag>;
    using U = std::make_unsigned_t<T>;
    constexpr int width = std::numeric_limits<U>::digits;
    const auto counts = fill_word(
        tag, static_cast<T>(count < width ? count : width));
    static_assert(Index >= 0 && Index < Traits::word_count);
    return sve_basic_wrap_word<Tag>(sve_bit_shift_right_variable_raw<T>(
        sve_basic_raw_word(value), sve_basic_raw_word(counts),
        svptrue_b8()));
  }

  template <nint_t Index, IntegerTag Tag>
  static VECOPS_ALWAYS_INLINE NativeWordVec<Tag> call(
      BitShiftRightOp, Tag, NativeWordVec<Tag> value,
      NativeWordVec<Tag> counts) {
    using T = ElementOf<Tag>;
    static_assert(Index >= 0 && Index < num_words(Tag{}));
    return sve_basic_wrap_word<Tag>(sve_bit_shift_right_variable_raw<T>(
        sve_basic_raw_word(value), sve_basic_raw_word(counts),
        svptrue_b8()));
  }

  template <nint_t Index, IntegerTag Tag, nint_t Count, typename Policy>
  static VECOPS_ALWAYS_INLINE NativeWordVec<Tag> call(
      BitShiftRightOp op, Tag tag, NativeWordVec<Tag> value,
      meta::Const<Count> count, NativeWordMask<Tag> mask,
      NativeWordVec<Tag> inactive, Policy) {
    const auto computed = call<Index>(op, tag, value, count);
    return blend(tag, inactive, mask, computed);
  }

  template <nint_t Index, IntegerTag Tag, typename Policy>
  static VECOPS_ALWAYS_INLINE NativeWordVec<Tag> call(
      BitShiftRightOp op, Tag tag, NativeWordVec<Tag> value, int count,
      NativeWordMask<Tag> mask, NativeWordVec<Tag> inactive, Policy) {
    const auto computed = call<Index>(op, tag, value, count);
    return blend(tag, inactive, mask, computed);
  }

  template <nint_t Index, IntegerTag Tag, typename Policy>
  static VECOPS_ALWAYS_INLINE NativeWordVec<Tag> call(
      BitShiftRightOp op, Tag tag, NativeWordVec<Tag> value,
      NativeWordVec<Tag> counts, NativeWordMask<Tag> mask,
      NativeWordVec<Tag> inactive, Policy) {
    const auto computed = call<Index>(op, tag, value, counts);
    return blend(tag, inactive, mask, computed);
  }
};

template <typename Op, nint_t Count, IntegerTag Tag>
VECOPS_ALWAYS_INLINE NativeWordVec<Tag> sve_bit_rotate_immediate_word(
    Tag tag, NativeWordVec<Tag> value) {
  using T = ElementOf<Tag>;
  using U = std::make_unsigned_t<T>;
  constexpr nint_t width = std::numeric_limits<U>::digits;
  constexpr nint_t remainder = Count % width;
  constexpr U normalized = static_cast<U>(
      remainder < 0 ? remainder + width : remainder);
  if constexpr (normalized == 0) {
    return value;
  }
#if defined(HAS_SVE2) || defined(HAS_SME)
  else {
    return sve_basic_wrap_word<Tag>(
        sve_bit_rotate_immediate_raw<Op, Count, T>(
            sve_basic_raw_word(value)));
  }
#else
  else {
    const auto counts = fill_word(tag, scalar_bit_value<T>(normalized));
    return sve_basic_wrap_word<Tag>(
        sve_bit_rotate_variable_raw<Op, T>(
            sve_basic_raw_word(value), sve_basic_raw_word(counts)));
  }
#endif
}

#define VECOPS_VEC_DEFINE_SVE_ROTATE(OpType)                            \
  template <>                                                           \
  struct NativeWordImpl<SVEBackend, OpType> {                           \
    template <nint_t Index, IntegerTag Tag, nint_t Count>               \
    static VECOPS_ALWAYS_INLINE NativeWordVec<Tag> call(                \
        OpType, Tag tag, NativeWordVec<Tag> value, meta::Const<Count>) {\
      using Traits = RepresentationTraits<SVEBackend, Tag>;            \
      static_assert(Index >= 0 && Index < Traits::word_count);          \
      return sve_bit_rotate_immediate_word<OpType, Count>(tag, value);  \
    }                                                                   \
    template <nint_t Index, IntegerTag Tag>                             \
    static VECOPS_ALWAYS_INLINE NativeWordVec<Tag> call(                \
        OpType, Tag tag, NativeWordVec<Tag> value, int count) {         \
      using Traits = RepresentationTraits<SVEBackend, Tag>;            \
      using T = ElementOf<Tag>;                                         \
      using U = std::make_unsigned_t<T>;                                \
      constexpr U lane_mask = std::numeric_limits<U>::digits - 1;      \
      const U normalized = static_cast<U>(count) & lane_mask;           \
      const auto counts = fill_word(                                    \
          tag, scalar_bit_value<T>(normalized));                        \
      static_assert(Index >= 0 && Index < Traits::word_count);          \
      return sve_basic_wrap_word<Tag>(                                  \
          sve_bit_rotate_variable_raw<OpType, T>(                       \
              sve_basic_raw_word(value), sve_basic_raw_word(counts)));  \
    }                                                                   \
    template <nint_t Index, IntegerTag Tag>                             \
    static VECOPS_ALWAYS_INLINE NativeWordVec<Tag> call(                \
        OpType, Tag, NativeWordVec<Tag> value,                          \
        NativeWordVec<Tag> counts) {                                   \
      using T = ElementOf<Tag>;                                         \
      static_assert(Index >= 0 && Index < num_words(Tag{}));            \
      return sve_basic_wrap_word<Tag>(                                  \
          sve_bit_rotate_variable_raw<OpType, T>(                       \
              sve_basic_raw_word(value), sve_basic_raw_word(counts)));  \
    }                                                                   \
    template <nint_t Index, IntegerTag Tag, nint_t Count, typename Policy>\
    static VECOPS_ALWAYS_INLINE NativeWordVec<Tag> call(                \
        OpType op, Tag tag, NativeWordVec<Tag> value,                   \
        meta::Const<Count> count, NativeWordMask<Tag> mask,             \
        NativeWordVec<Tag> inactive, Policy) {                          \
      return blend(                                                     \
          tag, inactive, mask, call<Index>(op, tag, value, count));     \
    }                                                                   \
    template <nint_t Index, IntegerTag Tag, typename Policy>            \
    static VECOPS_ALWAYS_INLINE NativeWordVec<Tag> call(                \
        OpType op, Tag tag, NativeWordVec<Tag> value, int count,        \
        NativeWordMask<Tag> mask, NativeWordVec<Tag> inactive, Policy) {\
      return blend(                                                     \
          tag, inactive, mask, call<Index>(op, tag, value, count));     \
    }                                                                   \
    template <nint_t Index, IntegerTag Tag, typename Policy>            \
    static VECOPS_ALWAYS_INLINE NativeWordVec<Tag> call(                \
        OpType op, Tag tag, NativeWordVec<Tag> value,                   \
        NativeWordVec<Tag> counts, NativeWordMask<Tag> mask,            \
        NativeWordVec<Tag> inactive, Policy) {                          \
      return blend(                                                     \
          tag, inactive, mask, call<Index>(op, tag, value, counts));    \
    }                                                                   \
  }

VECOPS_VEC_DEFINE_SVE_ROTATE(RotateLeftOp);
VECOPS_VEC_DEFINE_SVE_ROTATE(RotateRightOp);

#undef VECOPS_VEC_DEFINE_SVE_ROTATE

} // namespace vecops::vec::details

#endif // VECOPS_VEC_DETAILS_SVE_BIT_H
