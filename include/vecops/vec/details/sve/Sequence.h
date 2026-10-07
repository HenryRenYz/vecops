// SPDX-FileCopyrightText: 2026 HenryRenYz and vecops contributors
// SPDX-License-Identifier: MIT
#ifndef VECOPS_VEC_DETAILS_SVE_SEQUENCE_H
#define VECOPS_VEC_DETAILS_SVE_SEQUENCE_H

/**
 * @file Sequence.h
 * @brief SVE lane sequences built from INDEX, with one integer-to-float
 * conversion for floating element types.
 */

#include "vecops/vec/details/sve/Arithmetic.h"

namespace vecops::vec::details {

template <Element T, typename Raw>
VECOPS_ALWAYS_INLINE Raw sve_iota_small_float_madd(
    Raw index, Raw step, Raw start) {
  const auto widen_low = [](Raw value) {
    if constexpr (std::same_as<T, bfloat16_t>)
      return sve_bf16_to_f32_lo(value);
    else
      return svcvt_f32_f16_x(svptrue_b32(), value);
  };
  const auto widen_high = [](Raw value) {
    if constexpr (std::same_as<T, bfloat16_t>) {
      return sve_bf16_to_f32_hi(value);
    } else {
#if defined(__ARM_FEATURE_SVE2)
      return svcvtlt_f32_f16_x(svptrue_b32(), value);
#else
      const auto odd = svuzp2_u16(
          svreinterpret_u16_f16(value), svreinterpret_u16_f16(value));
      return svcvt_f32_f16_x(
          svptrue_b32(), svreinterpret_f16_u16(svzip1_u16(odd, odd)));
#endif
    }
  };
  const auto compute = [](svfloat32_t a, svfloat32_t b, svfloat32_t c) {
    return svmad_f32_x(svptrue_b32(), a, b, c);
  };
  const auto low = compute(widen_low(index), widen_low(step), widen_low(start));
  const auto high =
      compute(widen_high(index), widen_high(step), widen_high(start));
  if constexpr (std::same_as<T, bfloat16_t>) {
    return sve_f32_pair_to_bf16(low, high);
  } else {
    const auto packed_low = svcvt_f16_f32_z(svptrue_b32(), low);
#if defined(__ARM_FEATURE_SVE2)
    return svcvtnt_f16_f32_m(packed_low, svptrue_b32(), high);
#else
    const auto packed_high = svcvt_f16_f32_z(svptrue_b32(), high);
    return svtrn1_f16(packed_low, packed_high);
#endif
  }
}

template <std::integral T>
VECOPS_ALWAYS_INLINE T sve_iota_word_start(
    T start, T step, nint_t lane_base) {
  using U = std::make_unsigned_t<T>;
  const U start_bits = [&] {
    if constexpr (std::is_signed_v<T>) return ::vecops::bitcast<U>(start);
    else return start;
  }();
  const U step_bits = [&] {
    if constexpr (std::is_signed_v<T>) return ::vecops::bitcast<U>(step);
    else return step;
  }();
  const U result = static_cast<U>(
      start_bits + step_bits * static_cast<U>(lane_base));
  if constexpr (std::is_signed_v<T>) return ::vecops::bitcast<T>(result);
  else return result;
}

template <nint_t Index, VectorTag Tag>
  requires std::integral<ElementOf<Tag>>
VECOPS_ALWAYS_INLINE NativeWordVec<Tag> sve_integer_iota(
    Tag, ElementOf<Tag> start, ElementOf<Tag> step) {
  using T = ElementOf<Tag>;
  const T word_start = sve_iota_word_start(
      start, step, Index * native_word_size(Tag{}));
#define VECOPS_VEC_SVE_DIRECT_IOTA(Type, Suffix)                        \
  if constexpr (std::same_as<T, Type>)                                  \
    return sve_basic_wrap_word<Tag>(svindex_##Suffix(word_start, step))
  VECOPS_VEC_SVE_DIRECT_IOTA(int8_t, s8);
  else VECOPS_VEC_SVE_DIRECT_IOTA(uint8_t, u8);
  else VECOPS_VEC_SVE_DIRECT_IOTA(int16_t, s16);
  else VECOPS_VEC_SVE_DIRECT_IOTA(uint16_t, u16);
  else VECOPS_VEC_SVE_DIRECT_IOTA(int32_t, s32);
  else VECOPS_VEC_SVE_DIRECT_IOTA(uint32_t, u32);
  else VECOPS_VEC_SVE_DIRECT_IOTA(int64_t, s64);
  else VECOPS_VEC_SVE_DIRECT_IOTA(uint64_t, u64);
#undef VECOPS_VEC_SVE_DIRECT_IOTA
}

template <nint_t Index, VectorTag Tag>
VECOPS_ALWAYS_INLINE NativeWordVec<Tag> sve_iota_indices(Tag) {
  using T = ElementOf<Tag>;
  const nint_t base = Index * native_word_size(Tag{});
#define VECOPS_VEC_SVE_INTEGER_INDEX(Type, Suffix)                      \
  if constexpr (std::same_as<T, Type>)                                  \
    return sve_basic_wrap_word<Tag>(svindex_##Suffix(                   \
        static_cast<Type>(base), static_cast<Type>(1)))
  VECOPS_VEC_SVE_INTEGER_INDEX(int8_t, s8);
  else VECOPS_VEC_SVE_INTEGER_INDEX(uint8_t, u8);
  else VECOPS_VEC_SVE_INTEGER_INDEX(int16_t, s16);
  else VECOPS_VEC_SVE_INTEGER_INDEX(uint16_t, u16);
  else VECOPS_VEC_SVE_INTEGER_INDEX(int32_t, s32);
  else VECOPS_VEC_SVE_INTEGER_INDEX(uint32_t, u32);
  else VECOPS_VEC_SVE_INTEGER_INDEX(int64_t, s64);
  else VECOPS_VEC_SVE_INTEGER_INDEX(uint64_t, u64);
#undef VECOPS_VEC_SVE_INTEGER_INDEX
  else if constexpr (std::same_as<T, float16_t>) {
    return sve_basic_wrap_word<Tag>(svcvt_f16_u16_x(
        svptrue_b16(), svindex_u16(static_cast<uint16_t>(base), 1)));
  } else if constexpr (std::same_as<T, float32_t>) {
    return sve_basic_wrap_word<Tag>(svcvt_f32_u32_x(
        svptrue_b32(), svindex_u32(static_cast<uint32_t>(base), 1)));
  } else if constexpr (std::same_as<T, float64_t>) {
    return sve_basic_wrap_word<Tag>(svcvt_f64_u64_x(
        svptrue_b64(), svindex_u64(static_cast<uint64_t>(base), 1)));
  } else if constexpr (std::same_as<T, bfloat16_t>) {
    const auto index = svindex_u16(static_cast<uint16_t>(base), 1);
    const auto low = svcvt_f32_u32_x(svptrue_b32(), svunpklo_u32(index));
    const auto high = svcvt_f32_u32_x(svptrue_b32(), svunpkhi_u32(index));
    return sve_basic_wrap_word<Tag>(sve_f32_pair_to_bf16(low, high));
  } else {
    static_assert(dispatch_dependent_false<T>, "unsupported SVE iota type");
  }
}

template <>
struct NativeWordImpl<SVEBackend, IotaOp> {
  template <nint_t Index, VectorTag Tag>
  static VECOPS_ALWAYS_INLINE NativeWordVec<Tag> call(
      IotaOp, Tag tag, ElementOf<Tag> start) {
    using T = ElementOf<Tag>;
    if constexpr (std::integral<T>) {
      return sve_integer_iota<Index>(tag, start, static_cast<T>(1));
    } else {
      return NativeWordImpl<SVEBackend, AddOp>::template call<Index>(
          AddOp{}, tag,
          NativeWordImpl<SVEBackend, FillOp>::template call<Index>(
              FillOp{}, tag, start),
          sve_iota_indices<Index>(tag));
    }
  }

  template <nint_t Index, VectorTag Tag>
  static VECOPS_ALWAYS_INLINE NativeWordVec<Tag> call(
      IotaOp, Tag tag, ElementOf<Tag> start, ElementOf<Tag> step) {
    if constexpr (std::integral<ElementOf<Tag>>) {
      return sve_integer_iota<Index>(tag, start, step);
    } else {
      using T = ElementOf<Tag>;
      if constexpr (
          std::same_as<T, bfloat16_t> || std::same_as<T, float16_t>) {
        const auto index = sve_basic_raw_word(sve_iota_indices<Index>(tag));
        const auto step_word =
            NativeWordImpl<SVEBackend, FillOp>::template call<Index>(
                FillOp{}, tag, step);
        const auto start_word =
            NativeWordImpl<SVEBackend, FillOp>::template call<Index>(
                FillOp{}, tag, start);
        return sve_basic_wrap_word<Tag>(sve_iota_small_float_madd<T>(
            index, sve_basic_raw_word(step_word),
            sve_basic_raw_word(start_word)));
      }
      return NativeWordImpl<SVEBackend, FmaddOp>::template call<Index>(
          FmaddOp{}, tag, sve_iota_indices<Index>(tag),
          NativeWordImpl<SVEBackend, FillOp>::template call<Index>(
              FillOp{}, tag, step),
          NativeWordImpl<SVEBackend, FillOp>::template call<Index>(
              FillOp{}, tag, start));
    }
  }
};

} // namespace vecops::vec::details

#endif // VECOPS_VEC_DETAILS_SVE_SEQUENCE_H
