#ifndef VECOPS_VEC_DETAILS_SVE_ARITHMETIC_H
#define VECOPS_VEC_DETAILS_SVE_ARITHMETIC_H

/**
 * @file Arithmetic.h
 * @brief SVE backend arithmetic operations using ARM SVE intrinsics.
 *
 * Small-float types (bfloat16, float16) are widened to float32 for
 * computation since SVE lacks native fp16/bf16 arithmetic at most width
 * configurations. Integer operations use SVE native add/sub/mul
 * instructions. Masked variants use svsel (predicated blend) to compose
 * computed results with an inactive vector according to the policy tag.
 */

#include "vecops/vec/details/Dispatch.h"

namespace vecops::vec::details {


/* **************************************************************************** */
//    bfloat16 conversion helpers                                             //
/* **************************************************************************** */

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
  // Keep UZP1 in the integer domain. The operation is bitwise-identical, but
  // avoids a bf16 vector_interleave that BiSheng 5.1 cannot select at -O2.
  const auto predicate = svptrue_b16();
  const auto low_converted = svcvt_bf16_f32_x(predicate, low);
  const auto high_converted = svcvt_bf16_f32_x(predicate, high);
  return svreinterpret_bf16_u16(svuzp1_u16(
      svreinterpret_u16_bf16(low_converted),
      svreinterpret_u16_bf16(high_converted)));
#else
  const auto low_bits = svreinterpret_u16_u32(
      sve_float32_to_bfloat16_rne_bits(low));
  const auto high_bits = svreinterpret_u16_u32(
      sve_float32_to_bfloat16_rne_bits(high));
  return svreinterpret_bf16_u16(svuzp1_u16(low_bits, high_bits));
#endif
}

/* **************************************************************************** */
//    sve_bfloat16_binary helper                                              //
/* **************************************************************************** */

template <typename Op, typename Policy>
VECOPS_ALWAYS_INLINE svbfloat16_t sve_bfloat16_binary(
    svbfloat16_t a, svbfloat16_t b, svbool_t mask,
    svbfloat16_t inactive, Policy) {
  const auto a_low = sve_bfloat16_to_float32_low(a);
  const auto a_high = sve_bfloat16_to_float32_high(a);
  const auto b_low = sve_bfloat16_to_float32_low(b);
  const auto b_high = sve_bfloat16_to_float32_high(b);
  const auto mask_low = svunpklo_b(mask);
  const auto mask_high = svunpkhi_b(mask);
#define VECOPS_VEC_SVE_BF16_BINARY(Mask, A, B)                          \
    if constexpr (std::same_as<Op, AddOp>) {                             \
      if constexpr (std::same_as<Policy, PreserveArithmeticInactive>)   \
        return svadd_f32_m(Mask, A, B);                                  \
      else if constexpr (std::same_as<Policy, ZeroArithmeticInactive>)  \
        return svadd_f32_z(Mask, A, B);                                  \
      else return svadd_f32_x(Mask, A, B);                               \
    } else if constexpr (std::same_as<Op, SubOp>) {                      \
      if constexpr (std::same_as<Policy, PreserveArithmeticInactive>)   \
        return svsub_f32_m(Mask, A, B);                                  \
      else if constexpr (std::same_as<Policy, ZeroArithmeticInactive>)  \
        return svsub_f32_z(Mask, A, B);                                  \
      else return svsub_f32_x(Mask, A, B);                               \
    } else if constexpr (std::same_as<Op, MulOp>) {                      \
      if constexpr (std::same_as<Policy, PreserveArithmeticInactive>)   \
        return svmul_f32_m(Mask, A, B);                                  \
      else if constexpr (std::same_as<Policy, ZeroArithmeticInactive>)  \
        return svmul_f32_z(Mask, A, B);                                  \
      else return svmul_f32_x(Mask, A, B);                               \
    } else if constexpr (std::same_as<Op, DivOp>) {                      \
      if constexpr (std::same_as<Policy, PreserveArithmeticInactive>)   \
        return svdiv_f32_m(Mask, A, B);                                  \
      else if constexpr (std::same_as<Policy, ZeroArithmeticInactive>)  \
        return svdiv_f32_z(Mask, A, B);                                  \
      else return svdiv_f32_x(Mask, A, B);                               \
    } else if constexpr (std::same_as<Op, MinOp>) {                      \
      if constexpr (std::same_as<Policy, PreserveArithmeticInactive>)   \
        return svmin_f32_m(Mask, A, B);                                  \
      else if constexpr (std::same_as<Policy, ZeroArithmeticInactive>)  \
        return svmin_f32_z(Mask, A, B);                                  \
      else return svmin_f32_x(Mask, A, B);                               \
    } else if constexpr (std::same_as<Op, MaxOp>) {                      \
      if constexpr (std::same_as<Policy, PreserveArithmeticInactive>)   \
        return svmax_f32_m(Mask, A, B);                                  \
      else if constexpr (std::same_as<Policy, ZeroArithmeticInactive>)  \
        return svmax_f32_z(Mask, A, B);                                  \
      else return svmax_f32_x(Mask, A, B);                               \
    } else {                                                             \
      static_assert(                                                     \
          dispatch_dependent_false<Op>,                                  \
          "unsupported SVE bfloat16 arithmetic operation");            \
    }
  const auto low = [&]() {
    VECOPS_VEC_SVE_BF16_BINARY(
        mask_low, a_low, b_low);
  }();
  const auto high = [&]() {
    VECOPS_VEC_SVE_BF16_BINARY(
        mask_high, a_high, b_high);
  }();
#undef VECOPS_VEC_SVE_BF16_BINARY
  const auto computed = sve_float32_pair_to_bfloat16(low, high);
  if constexpr (std::same_as<Policy, ZeroArithmeticInactive>) {
    return computed;
  } else {
    return svreinterpret_bf16_u16(svsel_u16(
        mask,
        svreinterpret_u16_bf16(computed),
        svreinterpret_u16_bf16(inactive)));
  }
}

/* **************************************************************************** */
//    SVEArithmeticWordImpl and registrations                                 //
/* **************************************************************************** */

template <typename Op>
struct SVEArithmeticWordImpl {
  template <nint_t Index, VectorTag Tag>
  static VECOPS_ALWAYS_INLINE NativeWordVec<Tag> call(
      Op op, Tag tag, NativeWordVec<Tag> a, NativeWordVec<Tag> b) {
    return call<Index>(
        op,
        tag,
        a,
        b,
        svptrue_b8(),
        a,
        PreserveArithmeticInactive{});
  }

  template <nint_t Index, VectorTag Tag, typename Policy>
  static VECOPS_ALWAYS_INLINE NativeWordVec<Tag> call(
      Op, Tag, NativeWordVec<Tag> a, NativeWordVec<Tag> b,
      NativeWordMask<Tag> mask, NativeWordVec<Tag> inactive, Policy policy) {
    using Traits = RepresentationTraits<SVEBackend, Tag>;
    using T = ElementOf<Tag>;
    static_assert(Index >= 0 && Index < Traits::word_count);
    const auto raw_a = sve_basic_raw_word(a);
    const auto raw_b = sve_basic_raw_word(b);
    const auto raw_inactive = sve_basic_raw_word(inactive);
    const auto raw_result = [&]() {
      if constexpr (std::same_as<T, bfloat16_t>) {
        return sve_bfloat16_binary<Op>(
            raw_a, raw_b, mask, raw_inactive, policy);
#define VECOPS_VEC_SVE_ARITHMETIC(Suffix)                               \
        if constexpr (std::same_as<Op, AddOp>) {                         \
          if constexpr (std::same_as<Policy, PreserveArithmeticInactive>) return svadd_##Suffix##_m(mask, raw_a, raw_b); \
          else if constexpr (std::same_as<Policy, ZeroArithmeticInactive>) return svadd_##Suffix##_z(mask, raw_a, raw_b); \
          else return svsel_##Suffix(mask, svadd_##Suffix##_x(mask, raw_a, raw_b), raw_inactive); \
        } else if constexpr (std::same_as<Op, SubOp>) {                  \
          if constexpr (std::same_as<Policy, PreserveArithmeticInactive>) return svsub_##Suffix##_m(mask, raw_a, raw_b); \
          else if constexpr (std::same_as<Policy, ZeroArithmeticInactive>) return svsub_##Suffix##_z(mask, raw_a, raw_b); \
          else return svsel_##Suffix(mask, svsub_##Suffix##_x(mask, raw_a, raw_b), raw_inactive); \
        } else if constexpr (std::same_as<Op, MulOp>) {                  \
          if constexpr (std::same_as<Policy, PreserveArithmeticInactive>) return svmul_##Suffix##_m(mask, raw_a, raw_b); \
          else if constexpr (std::same_as<Policy, ZeroArithmeticInactive>) return svmul_##Suffix##_z(mask, raw_a, raw_b); \
          else return svsel_##Suffix(mask, svmul_##Suffix##_x(mask, raw_a, raw_b), raw_inactive); \
        } else if constexpr (std::same_as<Op, DivOp>) {                  \
          if constexpr (std::same_as<Policy, PreserveArithmeticInactive>) return svdiv_##Suffix##_m(mask, raw_a, raw_b); \
          else if constexpr (std::same_as<Policy, ZeroArithmeticInactive>) return svdiv_##Suffix##_z(mask, raw_a, raw_b); \
          else return svsel_##Suffix(mask, svdiv_##Suffix##_x(mask, raw_a, raw_b), raw_inactive); \
        } else if constexpr (std::same_as<Op, MinOp>) {                  \
          if constexpr (std::same_as<Policy, PreserveArithmeticInactive>) return svmin_##Suffix##_m(mask, raw_a, raw_b); \
          else if constexpr (std::same_as<Policy, ZeroArithmeticInactive>) return svmin_##Suffix##_z(mask, raw_a, raw_b); \
          else return svsel_##Suffix(mask, svmin_##Suffix##_x(mask, raw_a, raw_b), raw_inactive); \
        } else if constexpr (std::same_as<Op, MaxOp>) {                  \
          if constexpr (std::same_as<Policy, PreserveArithmeticInactive>) return svmax_##Suffix##_m(mask, raw_a, raw_b); \
          else if constexpr (std::same_as<Policy, ZeroArithmeticInactive>) return svmax_##Suffix##_z(mask, raw_a, raw_b); \
          else return svsel_##Suffix(mask, svmax_##Suffix##_x(mask, raw_a, raw_b), raw_inactive); \
        } else static_assert(dispatch_dependent_false<Op>, "unsupported SVE arithmetic operation")
      } else if constexpr (std::same_as<T, float16_t>) {
        VECOPS_VEC_SVE_ARITHMETIC(f16);
      } else if constexpr (std::same_as<T, float32_t>) {
        VECOPS_VEC_SVE_ARITHMETIC(f32);
      } else if constexpr (std::same_as<T, float64_t>) {
        VECOPS_VEC_SVE_ARITHMETIC(f64);
      } else if constexpr (std::same_as<T, int8_t>) {
        VECOPS_VEC_SVE_ARITHMETIC(s8);
      } else if constexpr (std::same_as<T, uint8_t>) {
        VECOPS_VEC_SVE_ARITHMETIC(u8);
      } else if constexpr (std::same_as<T, int16_t>) {
        VECOPS_VEC_SVE_ARITHMETIC(s16);
      } else if constexpr (std::same_as<T, uint16_t>) {
        VECOPS_VEC_SVE_ARITHMETIC(u16);
      } else if constexpr (std::same_as<T, int32_t>) {
        VECOPS_VEC_SVE_ARITHMETIC(s32);
      } else if constexpr (std::same_as<T, uint32_t>) {
        VECOPS_VEC_SVE_ARITHMETIC(u32);
      } else if constexpr (std::same_as<T, int64_t>) {
        VECOPS_VEC_SVE_ARITHMETIC(s64);
      } else if constexpr (std::same_as<T, uint64_t>) {
        VECOPS_VEC_SVE_ARITHMETIC(u64);
      } else {
        static_assert(
            dispatch_dependent_false<T>,
            "unsupported SVE arithmetic element type");
      }
#undef VECOPS_VEC_SVE_ARITHMETIC
    }();
    return sve_basic_wrap_word<Tag>(raw_result);
  }
};

template <>
struct NativeWordImpl<SVEBackend, AddOp> : SVEArithmeticWordImpl<AddOp> {};
template <>
struct NativeWordImpl<SVEBackend, SubOp> : SVEArithmeticWordImpl<SubOp> {};
template <>
struct NativeWordImpl<SVEBackend, MulOp> : SVEArithmeticWordImpl<MulOp> {};
template <>
struct NativeWordImpl<SVEBackend, DivOp> : SVEArithmeticWordImpl<DivOp> {};
template <>
struct NativeWordImpl<SVEBackend, MinOp> : SVEArithmeticWordImpl<MinOp> {};
template <>
struct NativeWordImpl<SVEBackend, MaxOp> : SVEArithmeticWordImpl<MaxOp> {};

/* **************************************************************************** */
//    SVEUnaryArithmeticWordImpl and registrations                            //
/* **************************************************************************** */

// WORKAROUND(BiSheng 5.1): its AArch64 SVE instruction selector cannot lower
// the scalable-BF16 form produced after combining the equivalent u16 ACLE
// intrinsics with the surrounding BF16 reinterpret/select. Force the sign-bit
// operation to remain an integer EOR/AND until that compiler bug is fixed.
template <bool Negate>
VECOPS_ALWAYS_INLINE svuint16_t sve_bfloat16_sign_bits(
    svuint16_t bits) {
  const auto sign_mask = svdup_n_u16(Negate ? 0x8000u : 0x7fffu);
  svuint16_t result;
  if constexpr (Negate) {
    __asm__("eor %0.d, %1.d, %2.d"
            : "=w"(result)
            : "w"(bits), "w"(sign_mask));
  } else {
    __asm__("and %0.d, %1.d, %2.d"
            : "=w"(result)
            : "w"(bits), "w"(sign_mask));
  }
  return result;
}

template <typename Op>
struct SVEUnaryArithmeticWordImpl {
  template <nint_t Index, VectorTag Tag>
  static VECOPS_ALWAYS_INLINE NativeWordVec<Tag> call(
      Op op, Tag tag, NativeWordVec<Tag> value) {
    return call<Index>(
        op, tag, value, svptrue_b8(), value,
        PreserveArithmeticInactive{});
  }

  template <nint_t Index, VectorTag Tag, typename Policy>
  static VECOPS_ALWAYS_INLINE NativeWordVec<Tag> call(
      Op, Tag, NativeWordVec<Tag> value, NativeWordMask<Tag> mask,
      NativeWordVec<Tag> inactive, Policy) {
    using Traits = RepresentationTraits<SVEBackend, Tag>;
    using T = ElementOf<Tag>;
    static_assert(Index >= 0 && Index < Traits::word_count);
    const auto raw_value = sve_basic_raw_word(value);
    const auto raw_inactive = sve_basic_raw_word(inactive);
    const auto raw_result = [&]() {
      if constexpr (std::same_as<T, bfloat16_t>) {
        const auto bits = svreinterpret_u16_bf16(raw_value);
        const auto computed = sve_bfloat16_sign_bits<
            std::same_as<Op, NegOp>>(bits);
        return svreinterpret_bf16_u16(svsel_u16(
            mask, computed, svreinterpret_u16_bf16(raw_inactive)));
#define VECOPS_VEC_SVE_UNARY(Suffix)                                   \
        if constexpr (std::same_as<Op, NegOp>)                          \
          return svneg_##Suffix##_m(raw_inactive, mask, raw_value);     \
        else                                                            \
          return svabs_##Suffix##_m(raw_inactive, mask, raw_value)
      } else if constexpr (std::same_as<T, float16_t>) {
        VECOPS_VEC_SVE_UNARY(f16);
      } else if constexpr (std::same_as<T, float32_t>) {
        VECOPS_VEC_SVE_UNARY(f32);
      } else if constexpr (std::same_as<T, float64_t>) {
        VECOPS_VEC_SVE_UNARY(f64);
      } else if constexpr (std::same_as<T, int8_t>) {
        VECOPS_VEC_SVE_UNARY(s8);
      } else if constexpr (std::same_as<T, int16_t>) {
        VECOPS_VEC_SVE_UNARY(s16);
      } else if constexpr (std::same_as<T, int32_t>) {
        VECOPS_VEC_SVE_UNARY(s32);
      } else if constexpr (std::same_as<T, int64_t>) {
        VECOPS_VEC_SVE_UNARY(s64);
      } else if constexpr (std::same_as<Op, AbsOp>) {
        if constexpr (std::same_as<T, uint8_t>)
          return svsel_u8(mask, raw_value, raw_inactive);
        else if constexpr (std::same_as<T, uint16_t>)
          return svsel_u16(mask, raw_value, raw_inactive);
        else if constexpr (std::same_as<T, uint32_t>)
          return svsel_u32(mask, raw_value, raw_inactive);
        else
          return svsel_u64(mask, raw_value, raw_inactive);
      } else if constexpr (std::same_as<T, uint8_t>) {
        return svreinterpret_u8_s8(svneg_s8_m(
            svreinterpret_s8_u8(raw_inactive), mask,
            svreinterpret_s8_u8(raw_value)));
      } else if constexpr (std::same_as<T, uint16_t>) {
        return svreinterpret_u16_s16(svneg_s16_m(
            svreinterpret_s16_u16(raw_inactive), mask,
            svreinterpret_s16_u16(raw_value)));
      } else if constexpr (std::same_as<T, uint32_t>) {
        return svreinterpret_u32_s32(svneg_s32_m(
            svreinterpret_s32_u32(raw_inactive), mask,
            svreinterpret_s32_u32(raw_value)));
      } else if constexpr (std::same_as<T, uint64_t>) {
        return svreinterpret_u64_s64(svneg_s64_m(
            svreinterpret_s64_u64(raw_inactive), mask,
            svreinterpret_s64_u64(raw_value)));
      } else {
        static_assert(dispatch_dependent_false<T>);
      }
#undef VECOPS_VEC_SVE_UNARY
    }();
    return sve_basic_wrap_word<Tag>(raw_result);
  }
};

template <>
struct NativeWordImpl<SVEBackend, NegOp>
    : SVEUnaryArithmeticWordImpl<NegOp> {};

template <>
struct NativeWordImpl<SVEBackend, AbsOp>
    : SVEUnaryArithmeticWordImpl<AbsOp> {};

/* **************************************************************************** */
//    SVEFloatingUnaryWordImpl and registrations                              //
/* **************************************************************************** */

template <typename Op>
struct SVEFloatingUnaryWordImpl {
  template <nint_t Index, FloatingTag Tag>
  static VECOPS_ALWAYS_INLINE NativeWordVec<Tag> call(
      Op op, Tag tag, NativeWordVec<Tag> value) {
    return call<Index>(
        op, tag, value, svptrue_b8(), value,
        PreserveArithmeticInactive{});
  }

  template <nint_t Index, FloatingTag Tag, typename Policy>
  static VECOPS_ALWAYS_INLINE NativeWordVec<Tag> call(
      Op, Tag tag, NativeWordVec<Tag> value, NativeWordMask<Tag> mask,
      NativeWordVec<Tag> inactive, Policy) {
    using T = ElementOf<Tag>;
    static_assert(Index >= 0 && Index < num_words(tag));
    const auto raw_value = sve_basic_raw_word(value);
    const auto raw_inactive = sve_basic_raw_word(inactive);
    const auto result = [&]() {
      if constexpr (std::same_as<T, bfloat16_t>) {
        const auto low = sve_bfloat16_to_float32_low(raw_value);
        const auto high = sve_bfloat16_to_float32_high(raw_value);
        const auto inactive_low =
            sve_bfloat16_to_float32_low(raw_inactive);
        const auto inactive_high =
            sve_bfloat16_to_float32_high(raw_inactive);
        const auto mask_low = svunpklo_b(mask);
        const auto mask_high = svunpkhi_b(mask);
        const auto compute = [](
            svfloat32_t input, svfloat32_t fallback, svbool_t active) {
          if constexpr (std::same_as<Op, SqrtOp>)
            return svsqrt_f32_m(fallback, active, input);
          else if constexpr (std::same_as<Op, RcpOp>)
            return svsel_f32(active, svrecpe_f32(input), fallback);
          else if constexpr (std::same_as<Op, RsqrtOp>)
            return svsel_f32(active, svrsqrte_f32(input), fallback);
          else
            static_assert(dispatch_dependent_false<Op>);
        };
        return sve_float32_pair_to_bfloat16(
            compute(low, inactive_low, mask_low),
            compute(high, inactive_high, mask_high));
      } else {
#define VECOPS_VEC_SVE_FLOATING_UNARY(Suffix)                          \
        if constexpr (std::same_as<Op, SqrtOp>)                        \
          return svsqrt_##Suffix##_m(raw_inactive, mask, raw_value);   \
        else if constexpr (std::same_as<Op, RcpOp>)                    \
          return svsel_##Suffix(                                       \
              mask, svrecpe_##Suffix(raw_value), raw_inactive);       \
        else if constexpr (std::same_as<Op, RsqrtOp>)                  \
          return svsel_##Suffix(                                       \
              mask, svrsqrte_##Suffix(raw_value), raw_inactive);      \
        else static_assert(dispatch_dependent_false<Op>)
        if constexpr (std::same_as<T, float16_t>) {
          VECOPS_VEC_SVE_FLOATING_UNARY(f16);
        } else if constexpr (std::same_as<T, float32_t>) {
          VECOPS_VEC_SVE_FLOATING_UNARY(f32);
        } else {
          VECOPS_VEC_SVE_FLOATING_UNARY(f64);
        }
#undef VECOPS_VEC_SVE_FLOATING_UNARY
      }
    }();
    return sve_basic_wrap_word<Tag>(result);
  }
};

template <>
struct NativeWordImpl<SVEBackend, SqrtOp>
    : SVEFloatingUnaryWordImpl<SqrtOp> {};
template <>
struct NativeWordImpl<SVEBackend, RcpOp>
    : SVEFloatingUnaryWordImpl<RcpOp> {};
template <>
struct NativeWordImpl<SVEBackend, RsqrtOp>
    : SVEFloatingUnaryWordImpl<RsqrtOp> {};

/* **************************************************************************** */
//    SVEFmaWordImpl and registrations                                        //
/* **************************************************************************** */

template <typename Op>
struct SVEFmaWordImpl {
  template <nint_t Index, VectorTag Tag>
  static VECOPS_ALWAYS_INLINE NativeWordVec<Tag> call(
      Op op, Tag tag, NativeWordVec<Tag> a, NativeWordVec<Tag> b,
      NativeWordVec<Tag> c) {
    return call<Index>(
        op, tag, a, b, c, svptrue_b8(), a,
        PreserveArithmeticInactive{});
  }

  template <nint_t Index, VectorTag Tag, typename Policy>
  static VECOPS_ALWAYS_INLINE NativeWordVec<Tag> call(
      Op op, Tag tag, NativeWordVec<Tag> a, NativeWordVec<Tag> b,
      NativeWordVec<Tag> c, NativeWordMask<Tag> mask,
      NativeWordVec<Tag> inactive, Policy) {
    using T = ElementOf<Tag>;
    static_assert(Index >= 0 && Index < num_words(tag));
    if constexpr (
        std::same_as<T, float16_t> || std::same_as<T, float32_t> ||
        std::same_as<T, float64_t>) {
      const auto raw_a = sve_basic_raw_word(a);
      const auto raw_b = sve_basic_raw_word(b);
      const auto raw_c = sve_basic_raw_word(c);
      const auto raw_inactive = sve_basic_raw_word(inactive);
#define VECOPS_VEC_SVE_FMA(Suffix)                                      \
      if constexpr (std::same_as<Op, FmaddOp>) {                        \
        if constexpr (std::same_as<Policy, PreserveArithmeticInactive>) \
          return sve_basic_wrap_word<Tag>(                              \
              svmad_##Suffix##_m(mask, raw_a, raw_b, raw_c));           \
        else if constexpr (std::same_as<Policy, ZeroArithmeticInactive>)\
          return sve_basic_wrap_word<Tag>(                              \
              svmad_##Suffix##_z(mask, raw_a, raw_b, raw_c));           \
        else return sve_basic_wrap_word<Tag>(svsel_##Suffix(            \
            mask, svmad_##Suffix##_x(mask, raw_a, raw_b, raw_c),        \
            raw_inactive));                                             \
      } else if constexpr (std::same_as<Op, FmsubOp>) {                 \
        if constexpr (std::same_as<Policy, PreserveArithmeticInactive>) \
          return sve_basic_wrap_word<Tag>(                              \
              svnmsb_##Suffix##_m(mask, raw_a, raw_b, raw_c));          \
        else if constexpr (std::same_as<Policy, ZeroArithmeticInactive>)\
          return sve_basic_wrap_word<Tag>(                              \
              svnmsb_##Suffix##_z(mask, raw_a, raw_b, raw_c));          \
        else return sve_basic_wrap_word<Tag>(svsel_##Suffix(            \
            mask, svnmsb_##Suffix##_x(mask, raw_a, raw_b, raw_c),       \
            raw_inactive));                                             \
      } else if constexpr (std::same_as<Op, FnmaddOp>) {                \
        if constexpr (std::same_as<Policy, PreserveArithmeticInactive>) \
          return sve_basic_wrap_word<Tag>(                              \
              svmsb_##Suffix##_m(mask, raw_a, raw_b, raw_c));           \
        else if constexpr (std::same_as<Policy, ZeroArithmeticInactive>)\
          return sve_basic_wrap_word<Tag>(                              \
              svmsb_##Suffix##_z(mask, raw_a, raw_b, raw_c));           \
        else return sve_basic_wrap_word<Tag>(svsel_##Suffix(            \
            mask, svmsb_##Suffix##_x(mask, raw_a, raw_b, raw_c),        \
            raw_inactive));                                             \
      } else if constexpr (std::same_as<Op, FnmsubOp>) {                \
        if constexpr (std::same_as<Policy, PreserveArithmeticInactive>) \
          return sve_basic_wrap_word<Tag>(                              \
              svnmad_##Suffix##_m(mask, raw_a, raw_b, raw_c));          \
        else if constexpr (std::same_as<Policy, ZeroArithmeticInactive>)\
          return sve_basic_wrap_word<Tag>(                              \
              svnmad_##Suffix##_z(mask, raw_a, raw_b, raw_c));          \
        else return sve_basic_wrap_word<Tag>(svsel_##Suffix(            \
            mask, svnmad_##Suffix##_x(mask, raw_a, raw_b, raw_c),       \
            raw_inactive));                                             \
      }
      if constexpr (std::same_as<T, float16_t>) {
        VECOPS_VEC_SVE_FMA(f16);
      } else if constexpr (std::same_as<T, float32_t>) {
        VECOPS_VEC_SVE_FMA(f32);
      } else {
        VECOPS_VEC_SVE_FMA(f64);
      }
#undef VECOPS_VEC_SVE_FMA
    } else {
      const auto product =
          NativeWordImpl<SVEBackend, MulOp>::template call<Index>(
              MulOp{}, tag, a, b);
      const auto computed = [&]() {
        if constexpr (std::same_as<Op, FmaddOp>)
          return NativeWordImpl<SVEBackend, AddOp>::template call<Index>(
              AddOp{}, tag, product, c);
        else if constexpr (std::same_as<Op, FmsubOp>)
          return NativeWordImpl<SVEBackend, SubOp>::template call<Index>(
              SubOp{}, tag, product, c);
        else if constexpr (std::same_as<Op, FnmaddOp>)
          return NativeWordImpl<SVEBackend, SubOp>::template call<Index>(
              SubOp{}, tag, c, product);
        else if constexpr (std::same_as<Op, FnmsubOp>) {
          using T = ElementOf<Tag>;
          const T negative_zero = [] {
            if constexpr (IsFloatV<T>) return static_cast<T>(-0.0F);
            else return T{};
          }();
          const auto zero =
              NativeWordImpl<SVEBackend, FillOp>::template call<Index>(
                  FillOp{}, tag, negative_zero);
          const auto negative_product =
              NativeWordImpl<SVEBackend, SubOp>::template call<Index>(
                  SubOp{}, tag, zero, product);
          return NativeWordImpl<SVEBackend, SubOp>::template call<Index>(
              SubOp{}, tag, negative_product, c);
        } else
          static_assert(dispatch_dependent_false<Op>);
      }();
      return NativeWordImpl<SVEBackend, BlendOp>::template call<Index>(
          BlendOp{}, tag, inactive, mask, computed);
    }
  }
};

template <>
struct NativeWordImpl<SVEBackend, FmaddOp> : SVEFmaWordImpl<FmaddOp> {};
template <>
struct NativeWordImpl<SVEBackend, FmsubOp> : SVEFmaWordImpl<FmsubOp> {};
template <>
struct NativeWordImpl<SVEBackend, FnmaddOp> : SVEFmaWordImpl<FnmaddOp> {};
template <>
struct NativeWordImpl<SVEBackend, FnmsubOp> : SVEFmaWordImpl<FnmsubOp> {};

} // namespace vecops::vec::details

#endif // VECOPS_VEC_DETAILS_SVE_ARITHMETIC_H
