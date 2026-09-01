#ifndef VECOPS_VEC_DETAILS_SVE_COMPARISON_H
#define VECOPS_VEC_DETAILS_SVE_COMPARISON_H

/**
 * @file Comparison.h
 * @brief SVE backend implementations for comparison and classification operations.
 */

#include <type_traits>

#include "vecops/vec/details/Dispatch.h"
#include "vecops/vec/details/sve/Basic.h"

namespace vecops::vec::details {

VECOPS_ALWAYS_INLINE svfloat32_t sve_comparison_bf16_low(svbfloat16_t value) {
  const auto bits = svunpklo_u32(svreinterpret_u16_bf16(value));
  return svreinterpret_f32_u32(
      svlsl_n_u32_x(svptrue_b32(), bits, 16));
}

VECOPS_ALWAYS_INLINE svfloat32_t sve_comparison_bf16_high(svbfloat16_t value) {
  const auto bits = svunpkhi_u32(svreinterpret_u16_bf16(value));
  return svreinterpret_f32_u32(
      svlsl_n_u32_x(svptrue_b32(), bits, 16));
}


/* **************************************************************************** */
//             Comparison and classification word implementations             //
/* **************************************************************************** */

template <typename Op, typename Raw>
VECOPS_ALWAYS_INLINE svbool_t sve_compare_f32(
    Op, svbool_t active, Raw a, Raw b) {
  if constexpr (std::same_as<Op, CmpEqOp>) return svcmpeq_f32(active, a, b);
  else if constexpr (std::same_as<Op, CmpNeOp>) return svcmpne_f32(active, a, b);
  else if constexpr (std::same_as<Op, CmpLtOp>) return svcmplt_f32(active, a, b);
  else if constexpr (std::same_as<Op, CmpGtOp>) return svcmpgt_f32(active, a, b);
  else if constexpr (std::same_as<Op, CmpLeOp>) return svcmple_f32(active, a, b);
  else if constexpr (std::same_as<Op, CmpGeOp>) return svcmpge_f32(active, a, b);
  else static_assert(dispatch_dependent_false<Op>, "unsupported SVE comparison");
}

template <typename Op, typename T, typename Raw>
VECOPS_ALWAYS_INLINE svbool_t sve_compare_raw(
    Op op, svbool_t active, Raw a, Raw b) {
  if constexpr (std::same_as<T, bfloat16_t>) {
    const auto low = sve_compare_f32(
        op, svunpklo_b(active),
        sve_comparison_bf16_low(a), sve_comparison_bf16_low(b));
    const auto high = sve_compare_f32(
        op, svunpkhi_b(active),
        sve_comparison_bf16_high(a), sve_comparison_bf16_high(b));
    return svuzp1_b16(low, high);
  } else if constexpr (std::same_as<T, float16_t>) {
    if constexpr (std::same_as<Op, CmpEqOp>) return svcmpeq_f16(active, a, b);
    else if constexpr (std::same_as<Op, CmpNeOp>) return svcmpne_f16(active, a, b);
    else if constexpr (std::same_as<Op, CmpLtOp>) return svcmplt_f16(active, a, b);
    else if constexpr (std::same_as<Op, CmpGtOp>) return svcmpgt_f16(active, a, b);
    else if constexpr (std::same_as<Op, CmpLeOp>) return svcmple_f16(active, a, b);
    else if constexpr (std::same_as<Op, CmpGeOp>) return svcmpge_f16(active, a, b);
    else static_assert(dispatch_dependent_false<Op>, "unsupported SVE comparison");
  } else if constexpr (std::same_as<T, float32_t>) {
    return sve_compare_f32(op, active, a, b);
#define VECOPS_VEC_SVE_COMPARE_TYPE(ElementType, Suffix)                 \
  } else if constexpr (std::same_as<T, ElementType>) {                   \
    if constexpr (std::same_as<Op, CmpEqOp>) return svcmpeq_##Suffix(active, a, b); \
    else if constexpr (std::same_as<Op, CmpNeOp>) return svcmpne_##Suffix(active, a, b); \
    else if constexpr (std::same_as<Op, CmpLtOp>) return svcmplt_##Suffix(active, a, b); \
    else if constexpr (std::same_as<Op, CmpGtOp>) return svcmpgt_##Suffix(active, a, b); \
    else if constexpr (std::same_as<Op, CmpLeOp>) return svcmple_##Suffix(active, a, b); \
    else if constexpr (std::same_as<Op, CmpGeOp>) return svcmpge_##Suffix(active, a, b); \
    else static_assert(dispatch_dependent_false<Op>, "unsupported SVE comparison")

  VECOPS_VEC_SVE_COMPARE_TYPE(float64_t, f64);
  VECOPS_VEC_SVE_COMPARE_TYPE(int8_t, s8);
  VECOPS_VEC_SVE_COMPARE_TYPE(uint8_t, u8);
  VECOPS_VEC_SVE_COMPARE_TYPE(int16_t, s16);
  VECOPS_VEC_SVE_COMPARE_TYPE(uint16_t, u16);
  VECOPS_VEC_SVE_COMPARE_TYPE(int32_t, s32);
  VECOPS_VEC_SVE_COMPARE_TYPE(uint32_t, u32);
  VECOPS_VEC_SVE_COMPARE_TYPE(int64_t, s64);
  VECOPS_VEC_SVE_COMPARE_TYPE(uint64_t, u64);
#undef VECOPS_VEC_SVE_COMPARE_TYPE
  } else {
    static_assert(
        dispatch_dependent_false<T>,
        "SVE comparison has no implementation for this element type");
  }
}

template <typename Op, typename T, typename Raw>
VECOPS_ALWAYS_INLINE svbool_t sve_classify_raw(
    Op, svbool_t active, Raw value) {
  if constexpr (std::same_as<T, bfloat16_t>) {
    const auto bits = svreinterpret_u16_bf16(value);
    const auto absolute = svand_n_u16_x(active, bits, 0x7fffu);
    if constexpr (std::same_as<Op, IsNanOp>)
      return svcmpgt_n_u16(active, absolute, 0x7f80u);
    else if constexpr (std::same_as<Op, IsPosInfOp>)
      return svcmpeq_n_u16(active, bits, 0x7f80u);
    else if constexpr (std::same_as<Op, IsNegInfOp>)
      return svcmpeq_n_u16(active, bits, 0xff80u);
    else if constexpr (std::same_as<Op, IsInfOp>)
      return svcmpeq_n_u16(active, absolute, 0x7f80u);
    else if constexpr (std::same_as<Op, IsFiniteOp>)
      return svcmplt_n_u16(active, absolute, 0x7f80u);
    else if constexpr (std::same_as<Op, IsNormalOp>)
      return svand_b_z(
          active,
          svcmpge_n_u16(active, absolute, 0x0080u),
          svcmplt_n_u16(active, absolute, 0x7f80u));
    else if constexpr (std::same_as<Op, SignBitOp>)
      return svcmpeq_n_u16(
          active, svand_n_u16_x(active, bits, 0x8000u), 0x8000u);
    else static_assert(dispatch_dependent_false<Op>, "unsupported SVE classification");
  } else if constexpr (std::same_as<T, float16_t>) {
    const auto bits = svreinterpret_u16_f16(value);
    const auto absolute = svand_n_u16_x(active, bits, 0x7fffu);
    if constexpr (std::same_as<Op, IsNanOp>)
      return svcmpgt_n_u16(active, absolute, 0x7c00u);
    else if constexpr (std::same_as<Op, IsPosInfOp>)
      return svcmpeq_n_u16(active, bits, 0x7c00u);
    else if constexpr (std::same_as<Op, IsNegInfOp>)
      return svcmpeq_n_u16(active, bits, 0xfc00u);
    else if constexpr (std::same_as<Op, IsInfOp>)
      return svcmpeq_n_u16(active, absolute, 0x7c00u);
    else if constexpr (std::same_as<Op, IsFiniteOp>)
      return svcmplt_n_u16(active, absolute, 0x7c00u);
    else if constexpr (std::same_as<Op, IsNormalOp>)
      return svand_b_z(
          active,
          svcmpge_n_u16(active, absolute, 0x0400u),
          svcmplt_n_u16(active, absolute, 0x7c00u));
    else if constexpr (std::same_as<Op, SignBitOp>)
      return svcmpeq_n_u16(
          active, svand_n_u16_x(active, bits, 0x8000u), 0x8000u);
    else static_assert(dispatch_dependent_false<Op>, "unsupported SVE classification");
  } else if constexpr (std::same_as<T, float32_t>) {
    const auto bits = svreinterpret_u32_f32(value);
    const auto absolute = svand_n_u32_x(active, bits, 0x7fffffffu);
    if constexpr (std::same_as<Op, IsNanOp>)
      return svcmpgt_n_u32(active, absolute, 0x7f800000u);
    else if constexpr (std::same_as<Op, IsPosInfOp>)
      return svcmpeq_n_u32(active, bits, 0x7f800000u);
    else if constexpr (std::same_as<Op, IsNegInfOp>)
      return svcmpeq_n_u32(active, bits, 0xff800000u);
    else if constexpr (std::same_as<Op, IsInfOp>)
      return svcmpeq_n_u32(active, absolute, 0x7f800000u);
    else if constexpr (std::same_as<Op, IsFiniteOp>)
      return svcmplt_n_u32(active, absolute, 0x7f800000u);
    else if constexpr (std::same_as<Op, IsNormalOp>)
      return svand_b_z(
          active,
          svcmpge_n_u32(active, absolute, 0x00800000u),
          svcmplt_n_u32(active, absolute, 0x7f800000u));
    else if constexpr (std::same_as<Op, SignBitOp>)
      return svcmpeq_n_u32(
          active,
          svand_n_u32_x(active, bits, 0x80000000u),
          0x80000000u);
    else static_assert(dispatch_dependent_false<Op>, "unsupported SVE classification");
  } else if constexpr (std::same_as<T, float64_t>) {
    const auto bits = svreinterpret_u64_f64(value);
    const auto absolute = svand_n_u64_x(active, bits, 0x7fffffffffffffffull);
    if constexpr (std::same_as<Op, IsNanOp>)
      return svcmpgt_n_u64(active, absolute, 0x7ff0000000000000ull);
    else if constexpr (std::same_as<Op, IsPosInfOp>)
      return svcmpeq_n_u64(active, bits, 0x7ff0000000000000ull);
    else if constexpr (std::same_as<Op, IsNegInfOp>)
      return svcmpeq_n_u64(active, bits, 0xfff0000000000000ull);
    else if constexpr (std::same_as<Op, IsInfOp>)
      return svcmpeq_n_u64(active, absolute, 0x7ff0000000000000ull);
    else if constexpr (std::same_as<Op, IsFiniteOp>)
      return svcmplt_n_u64(active, absolute, 0x7ff0000000000000ull);
    else if constexpr (std::same_as<Op, IsNormalOp>)
      return svand_b_z(
          active,
          svcmpge_n_u64(active, absolute, 0x0010000000000000ull),
          svcmplt_n_u64(active, absolute, 0x7ff0000000000000ull));
    else if constexpr (std::same_as<Op, SignBitOp>)
      return svcmpeq_n_u64(
          active,
          svand_n_u64_x(active, bits, 0x8000000000000000ull),
          0x8000000000000000ull);
    else static_assert(dispatch_dependent_false<Op>, "unsupported SVE classification");
  } else {
    static_assert(
        dispatch_dependent_false<T>,
        "SVE classification has no implementation for this element type");
  }
}

template <typename Op>
struct SVEBinaryComparisonImpl {
  template <nint_t Index, VectorTag Tag>
  static VECOPS_ALWAYS_INLINE NativeWordMask<Tag> call(
      Op op, Tag tag, NativeWordVec<Tag> a, NativeWordVec<Tag> b) {
    using Traits = RepresentationTraits<SVEBackend, Tag>;
    using T = ElementOf<Tag>;
    static_assert(Index >= 0 && Index < Traits::word_count);
    const auto active = sve_prefix_predicate<T>(sve_valid_word_lanes<Index>(tag));
    return sve_compare_raw<Op, T>(
        op, active, sve_basic_raw_word(a), sve_basic_raw_word(b));
  }

  template <nint_t Index, VectorTag Tag>
  static VECOPS_ALWAYS_INLINE NativeWordMask<Tag> call(
      Op op, Tag tag, NativeWordVec<Tag> a, NativeWordVec<Tag> b,
      NativeWordMask<Tag> mask) {
    using T = ElementOf<Tag>;
    const auto valid = sve_prefix_predicate<T>(sve_valid_word_lanes<Index>(tag));
    const auto active = svand_b_z(valid, valid, mask);
    return sve_compare_raw<Op, T>(
        op, active, sve_basic_raw_word(a), sve_basic_raw_word(b));
  }
};

template <typename Op>
struct SVEClassificationImpl {
  template <nint_t Index, VectorTag Tag>
  static VECOPS_ALWAYS_INLINE NativeWordMask<Tag> call(
      Op op, Tag tag, NativeWordVec<Tag> value) {
    using Traits = RepresentationTraits<SVEBackend, Tag>;
    using T = ElementOf<Tag>;
    static_assert(Index >= 0 && Index < Traits::word_count);
    const auto active = sve_prefix_predicate<T>(sve_valid_word_lanes<Index>(tag));
    return sve_classify_raw<Op, T>(op, active, sve_basic_raw_word(value));
  }

  template <nint_t Index, VectorTag Tag>
  static VECOPS_ALWAYS_INLINE NativeWordMask<Tag> call(
      Op op, Tag tag, NativeWordVec<Tag> value, NativeWordMask<Tag> mask) {
    using T = ElementOf<Tag>;
    const auto valid = sve_prefix_predicate<T>(sve_valid_word_lanes<Index>(tag));
    const auto active = svand_b_z(valid, valid, mask);
    return sve_classify_raw<Op, T>(op, active, sve_basic_raw_word(value));
  }
};

#define VECOPS_VEC_SVE_BINARY_COMPARISON(OpType)                         \
  template <>                                                             \
  struct NativeWordImpl<SVEBackend, OpType>                               \
      : SVEBinaryComparisonImpl<OpType> {}

VECOPS_VEC_SVE_BINARY_COMPARISON(CmpEqOp);
VECOPS_VEC_SVE_BINARY_COMPARISON(CmpNeOp);
VECOPS_VEC_SVE_BINARY_COMPARISON(CmpLtOp);
VECOPS_VEC_SVE_BINARY_COMPARISON(CmpGtOp);
VECOPS_VEC_SVE_BINARY_COMPARISON(CmpLeOp);
VECOPS_VEC_SVE_BINARY_COMPARISON(CmpGeOp);

#undef VECOPS_VEC_SVE_BINARY_COMPARISON

#define VECOPS_VEC_SVE_CLASSIFICATION(OpType)                            \
  template <>                                                             \
  struct NativeWordImpl<SVEBackend, OpType>                               \
      : SVEClassificationImpl<OpType> {}

VECOPS_VEC_SVE_CLASSIFICATION(IsNanOp);
VECOPS_VEC_SVE_CLASSIFICATION(IsPosInfOp);
VECOPS_VEC_SVE_CLASSIFICATION(IsNegInfOp);
VECOPS_VEC_SVE_CLASSIFICATION(IsInfOp);
VECOPS_VEC_SVE_CLASSIFICATION(IsFiniteOp);
VECOPS_VEC_SVE_CLASSIFICATION(IsNormalOp);
VECOPS_VEC_SVE_CLASSIFICATION(SignBitOp);

#undef VECOPS_VEC_SVE_CLASSIFICATION

} // namespace vecops::vec::details

#endif // VECOPS_VEC_DETAILS_SVE_COMPARISON_H
