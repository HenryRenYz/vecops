#ifndef VECOPS_NVEC_DETAILS_SVE_ARITHMETIC_H
#define VECOPS_NVEC_DETAILS_SVE_ARITHMETIC_H

#include "vecops/nvec/details/Dispatch.h"

namespace vecops::nvec::details {

VECOPS_ALWAYS_INLINE svfloat32_t sve_bfloat16_to_float32_low(
    svbfloat16_t value) {
  const auto bits = svunpklo_u32(svreinterpret_u16_bf16(value));
  return svreinterpret_f32_u32(
      svlsl_n_u32_x(svptrue_b32(), bits, 16));
}

VECOPS_ALWAYS_INLINE svfloat32_t sve_bfloat16_to_float32_high(
    svbfloat16_t value) {
  const auto bits = svunpkhi_u32(svreinterpret_u16_bf16(value));
  return svreinterpret_f32_u32(
      svlsl_n_u32_x(svptrue_b32(), bits, 16));
}

VECOPS_ALWAYS_INLINE svuint32_t sve_float32_to_bfloat16_rne_bits(
    svfloat32_t value) {
  const auto predicate = svptrue_b32();
  const auto bits = svreinterpret_u32_f32(value);
  const auto lsb = svand_n_u32_x(
      predicate, svlsr_n_u32_x(predicate, bits, 16), 1);
  const auto rounded = svlsr_n_u32_x(
      predicate,
      svadd_u32_x(
          predicate, bits, svadd_n_u32_x(predicate, lsb, 0x7fffu)),
      16);
  const auto absolute = svand_n_u32_x(predicate, bits, 0x7fffffffu);
  const auto is_nan = svcmpgt_n_u32(predicate, absolute, 0x7f800000u);
  return svsel_u32(is_nan, svdup_n_u32(0x7fc0u), rounded);
}

VECOPS_ALWAYS_INLINE svbfloat16_t sve_float32_pair_to_bfloat16(
    svfloat32_t low, svfloat32_t high) {
#if defined(__ARM_FEATURE_SVE_BF16) && !defined(VECOPS_PRESERVE_SUBNORMALS)
  const auto predicate = svptrue_b16();
  return svuzp1(
      svcvt_bf16_x(predicate, low), svcvt_bf16_x(predicate, high));
#else
  const auto low_bits = svreinterpret_u16_u32(
      sve_float32_to_bfloat16_rne_bits(low));
  const auto high_bits = svreinterpret_u16_u32(
      sve_float32_to_bfloat16_rne_bits(high));
  return svreinterpret_bf16_u16(svuzp1_u16(low_bits, high_bits));
#endif
}

VECOPS_ALWAYS_INLINE svbfloat16_t sve_add_bfloat16(
    svbfloat16_t a, svbfloat16_t b) {
  const auto predicate = svptrue_b32();
  const auto low = svadd_f32_x(
      predicate,
      sve_bfloat16_to_float32_low(a),
      sve_bfloat16_to_float32_low(b));
  const auto high = svadd_f32_x(
      predicate,
      sve_bfloat16_to_float32_high(a),
      sve_bfloat16_to_float32_high(b));
  return sve_float32_pair_to_bfloat16(low, high);
}

template <>
struct NativeWordImpl<SVEBackend, AddOp> {
  template <nint_t Index, VectorTag Tag>
  static VECOPS_ALWAYS_INLINE NativeWordVec<Tag> call(
      AddOp, Tag, NativeWordVec<Tag> a, NativeWordVec<Tag> b) {
    using Traits = RepresentationTraits<SVEBackend, Tag>;
    using T = ElementOf<Tag>;
    static_assert(Index >= 0 && Index < Traits::word_count);
    const auto raw_a = sve_basic_raw_word(a);
    const auto raw_b = sve_basic_raw_word(b);
    const auto raw_result = [&]() {
      if constexpr (std::same_as<T, bfloat16_t>) {
        return sve_add_bfloat16(raw_a, raw_b);
      } else if constexpr (std::same_as<T, float16_t>) {
        return svadd_f16_x(svptrue_b16(), raw_a, raw_b);
      } else if constexpr (std::same_as<T, float32_t>) {
        return svadd_f32_x(svptrue_b32(), raw_a, raw_b);
      } else if constexpr (std::same_as<T, float64_t>) {
        return svadd_f64_x(svptrue_b64(), raw_a, raw_b);
      } else if constexpr (std::same_as<T, int8_t>) {
        return svadd_s8_x(svptrue_b8(), raw_a, raw_b);
      } else if constexpr (std::same_as<T, uint8_t>) {
        return svadd_u8_x(svptrue_b8(), raw_a, raw_b);
      } else if constexpr (std::same_as<T, int16_t>) {
        return svadd_s16_x(svptrue_b16(), raw_a, raw_b);
      } else if constexpr (std::same_as<T, uint16_t>) {
        return svadd_u16_x(svptrue_b16(), raw_a, raw_b);
      } else if constexpr (std::same_as<T, int32_t>) {
        return svadd_s32_x(svptrue_b32(), raw_a, raw_b);
      } else if constexpr (std::same_as<T, uint32_t>) {
        return svadd_u32_x(svptrue_b32(), raw_a, raw_b);
      } else if constexpr (std::same_as<T, int64_t>) {
        return svadd_s64_x(svptrue_b64(), raw_a, raw_b);
      } else if constexpr (std::same_as<T, uint64_t>) {
        return svadd_u64_x(svptrue_b64(), raw_a, raw_b);
      } else {
        static_assert(
            dispatch_dependent_false<T>,
            "SVE add has no implementation for this element type");
      }
    }();
    return sve_basic_wrap_word<Tag>(raw_result);
  }
};

} // namespace vecops::nvec::details

#endif // VECOPS_NVEC_DETAILS_SVE_ARITHMETIC_H
