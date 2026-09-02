#ifndef VECOPS_VEC_DETAILS_SVE_ARITHMETIC_H
#define VECOPS_VEC_DETAILS_SVE_ARITHMETIC_H

#include <limits>

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
#include "vecops/vec/details/sve/Bf16.h"

namespace vecops::vec::details {


/* **************************************************************************** */
//    sve_bfloat16_binary helper                                              //
/* **************************************************************************** */

template <typename Op, typename Policy>
VECOPS_ALWAYS_INLINE svbfloat16_t sve_bfloat16_binary(
    svbfloat16_t a, svbfloat16_t b, svbool_t mask,
    svbfloat16_t inactive, Policy) {
  const auto a_low = sve_bf16_to_f32_lo(a);
  const auto a_high = sve_bf16_to_f32_hi(a);
  const auto b_low = sve_bf16_to_f32_lo(b);
  const auto b_high = sve_bf16_to_f32_hi(b);
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
  const auto low = [&]() VECOPS_KERNEL_LAMBDA {
    VECOPS_VEC_SVE_BF16_BINARY(
        mask_low, a_low, b_low);
  }();
  const auto high = [&]() VECOPS_KERNEL_LAMBDA {
    VECOPS_VEC_SVE_BF16_BINARY(
        mask_high, a_high, b_high);
  }();
#undef VECOPS_VEC_SVE_BF16_BINARY
  const auto computed = sve_f32_pair_to_bf16(low, high);
  if constexpr (std::same_as<Policy, ZeroArithmeticInactive>) {
    return computed;
  } else {
    return svreinterpret_bf16_u16(svsel_u16(
        mask,
        svreinterpret_u16_bf16(computed),
        svreinterpret_u16_bf16(inactive)));
  }
}

VECOPS_ALWAYS_INLINE svbfloat16_t sve_bfloat16_clamp(
    svbfloat16_t value, svbfloat16_t lower, svbfloat16_t upper) {
  const auto clamp_part = [](svfloat32_t input, svfloat32_t low,
                             svfloat32_t high) {
    const auto full = svptrue_b32();
    return svmin_f32_x(full, svmax_f32_x(full, input, low), high);
  };
  return sve_f32_pair_to_bf16(
      clamp_part(
          sve_bf16_to_f32_lo(value), sve_bf16_to_f32_lo(lower),
          sve_bf16_to_f32_lo(upper)),
      clamp_part(
          sve_bf16_to_f32_hi(value), sve_bf16_to_f32_hi(lower),
          sve_bf16_to_f32_hi(upper)));
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
    const auto raw_result = [&]() VECOPS_KERNEL_LAMBDA {
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

template <typename Op>
struct SVESaturatingArithmeticWordImpl {
  template <nint_t Index, VectorTag Tag>
    requires std::integral<ElementOf<Tag>>
  static VECOPS_ALWAYS_INLINE NativeWordVec<Tag> call(
      Op op, Tag tag, NativeWordVec<Tag> a, NativeWordVec<Tag> b) {
    return call<Index>(
        op, tag, a, b, svptrue_b8(), a,
        PreserveArithmeticInactive{});
  }

  template <nint_t Index, VectorTag Tag, typename Policy>
    requires std::integral<ElementOf<Tag>>
  static VECOPS_ALWAYS_INLINE NativeWordVec<Tag> call(
      Op, Tag, NativeWordVec<Tag> a, NativeWordVec<Tag> b,
      NativeWordMask<Tag> mask, NativeWordVec<Tag> inactive, Policy) {
    using Traits = RepresentationTraits<SVEBackend, Tag>;
    using T = ElementOf<Tag>;
    static_assert(Index >= 0 && Index < Traits::word_count);
    const auto raw_a = sve_basic_raw_word(a);
    const auto raw_b = sve_basic_raw_word(b);
    const auto raw_inactive = sve_basic_raw_word(inactive);
#if defined(HAS_SVE2)
#define VECOPS_VEC_SVE_SATURATING_ARITHMETIC(Suffix)                   \
    if constexpr (std::same_as<Op, SaturatedAddOp>) {                   \
      if constexpr (std::same_as<Policy, PreserveArithmeticInactive>)  \
        return sve_basic_wrap_word<Tag>(                               \
            svqadd_##Suffix##_m(mask, raw_a, raw_b));                  \
      else if constexpr (std::same_as<Policy, ZeroArithmeticInactive>) \
        return sve_basic_wrap_word<Tag>(                               \
            svqadd_##Suffix##_z(mask, raw_a, raw_b));                  \
      else                                                              \
        return sve_basic_wrap_word<Tag>(svsel_##Suffix(                \
            mask, svqadd_##Suffix##_x(mask, raw_a, raw_b),             \
            raw_inactive));                                             \
    } else {                                                            \
      if constexpr (std::same_as<Policy, PreserveArithmeticInactive>)  \
        return sve_basic_wrap_word<Tag>(                               \
            svqsub_##Suffix##_m(mask, raw_a, raw_b));                  \
      else if constexpr (std::same_as<Policy, ZeroArithmeticInactive>) \
        return sve_basic_wrap_word<Tag>(                               \
            svqsub_##Suffix##_z(mask, raw_a, raw_b));                  \
      else                                                              \
        return sve_basic_wrap_word<Tag>(svsel_##Suffix(                \
            mask, svqsub_##Suffix##_x(mask, raw_a, raw_b),             \
            raw_inactive));                                             \
    }
    if constexpr (std::same_as<T, int8_t>) {
      VECOPS_VEC_SVE_SATURATING_ARITHMETIC(s8);
    } else if constexpr (std::same_as<T, uint8_t>) {
      VECOPS_VEC_SVE_SATURATING_ARITHMETIC(u8);
    } else if constexpr (std::same_as<T, int16_t>) {
      VECOPS_VEC_SVE_SATURATING_ARITHMETIC(s16);
    } else if constexpr (std::same_as<T, uint16_t>) {
      VECOPS_VEC_SVE_SATURATING_ARITHMETIC(u16);
    } else if constexpr (std::same_as<T, int32_t>) {
      VECOPS_VEC_SVE_SATURATING_ARITHMETIC(s32);
    } else if constexpr (std::same_as<T, uint32_t>) {
      VECOPS_VEC_SVE_SATURATING_ARITHMETIC(u32);
    } else if constexpr (std::same_as<T, int64_t>) {
      VECOPS_VEC_SVE_SATURATING_ARITHMETIC(s64);
    } else {
      VECOPS_VEC_SVE_SATURATING_ARITHMETIC(u64);
    }
#undef VECOPS_VEC_SVE_SATURATING_ARITHMETIC
#else
#define VECOPS_VEC_SVE_SYNTHESIZE_SATURATING(Suffix, Bits)             \
    {                                                                   \
      const auto full = svptrue_b##Bits();                              \
      const auto wrapped = [&]() VECOPS_KERNEL_LAMBDA {                \
        if constexpr (std::same_as<Op, SaturatedAddOp>)                \
          return svadd_##Suffix##_x(full, raw_a, raw_b);                \
        else                                                            \
          return svsub_##Suffix##_x(full, raw_a, raw_b);                \
      }();                                                              \
      const auto saturated = [&]() VECOPS_KERNEL_LAMBDA {              \
        if constexpr (std::is_unsigned_v<T>) {                          \
          const auto overflow = [&]() VECOPS_KERNEL_LAMBDA {           \
            if constexpr (std::same_as<Op, SaturatedAddOp>)            \
              return svcmplt_##Suffix(full, wrapped, raw_a);            \
            else                                                        \
              return svcmplt_##Suffix(full, raw_a, raw_b);              \
          }();                                                          \
          const auto limit = [&]() VECOPS_KERNEL_LAMBDA {              \
            if constexpr (std::same_as<Op, SaturatedAddOp>)            \
              return svdup_n_##Suffix(std::numeric_limits<T>::max());  \
            else                                                        \
              return svdup_n_##Suffix(T{0});                            \
          }();                                                          \
          return svsel_##Suffix(overflow, limit, wrapped);              \
        } else {                                                        \
          const auto overflow_bits = [&]() VECOPS_KERNEL_LAMBDA {      \
            if constexpr (std::same_as<Op, SaturatedAddOp>)            \
              return svand_##Suffix##_x(                               \
                  full,                                                 \
                  sveor_##Suffix##_x(full, raw_a, wrapped),            \
                  sveor_##Suffix##_x(full, raw_b, wrapped));           \
            else                                                        \
              return svand_##Suffix##_x(                               \
                  full, sveor_##Suffix##_x(full, raw_a, raw_b),        \
                  sveor_##Suffix##_x(full, raw_a, wrapped));           \
          }();                                                          \
          const auto overflow =                                        \
              svcmplt_n_##Suffix(full, overflow_bits, 0);               \
          const auto negative = svcmplt_n_##Suffix(full, raw_a, 0);    \
          const auto limit = svsel_##Suffix(                            \
              negative,                                                \
              svdup_n_##Suffix(std::numeric_limits<T>::min()),         \
              svdup_n_##Suffix(std::numeric_limits<T>::max()));        \
          return svsel_##Suffix(overflow, limit, wrapped);              \
        }                                                               \
      }();                                                              \
      if constexpr (std::same_as<Policy, PreserveArithmeticInactive>)  \
        return sve_basic_wrap_word<Tag>(                               \
            svsel_##Suffix(mask, saturated, raw_a));                   \
      else if constexpr (std::same_as<Policy, ZeroArithmeticInactive>) \
        return sve_basic_wrap_word<Tag>(svsel_##Suffix(                \
            mask, saturated, svdup_n_##Suffix(T{0})));                 \
      else                                                              \
        return sve_basic_wrap_word<Tag>(                               \
            svsel_##Suffix(mask, saturated, raw_inactive));            \
    }
    if constexpr (std::same_as<T, int8_t>) {
      VECOPS_VEC_SVE_SYNTHESIZE_SATURATING(s8, 8);
    } else if constexpr (std::same_as<T, uint8_t>) {
      VECOPS_VEC_SVE_SYNTHESIZE_SATURATING(u8, 8);
    } else if constexpr (std::same_as<T, int16_t>) {
      VECOPS_VEC_SVE_SYNTHESIZE_SATURATING(s16, 16);
    } else if constexpr (std::same_as<T, uint16_t>) {
      VECOPS_VEC_SVE_SYNTHESIZE_SATURATING(u16, 16);
    } else if constexpr (std::same_as<T, int32_t>) {
      VECOPS_VEC_SVE_SYNTHESIZE_SATURATING(s32, 32);
    } else if constexpr (std::same_as<T, uint32_t>) {
      VECOPS_VEC_SVE_SYNTHESIZE_SATURATING(u32, 32);
    } else if constexpr (std::same_as<T, int64_t>) {
      VECOPS_VEC_SVE_SYNTHESIZE_SATURATING(s64, 64);
    } else {
      VECOPS_VEC_SVE_SYNTHESIZE_SATURATING(u64, 64);
    }
#undef VECOPS_VEC_SVE_SYNTHESIZE_SATURATING
#endif
  }
};

template <>
struct NativeWordImpl<SVEBackend, SaturatedAddOp>
    : SVESaturatingArithmeticWordImpl<SaturatedAddOp> {};

template <>
struct NativeWordImpl<SVEBackend, SaturatedSubOp>
    : SVESaturatingArithmeticWordImpl<SaturatedSubOp> {};

template <>
struct NativeWordImpl<SVEBackend, MulOp> : SVEArithmeticWordImpl<MulOp> {};
template <>
struct NativeWordImpl<SVEBackend, DivOp> : SVEArithmeticWordImpl<DivOp> {};
template <>
struct NativeWordImpl<SVEBackend, MinOp> : SVEArithmeticWordImpl<MinOp> {};
template <>
struct NativeWordImpl<SVEBackend, MaxOp> : SVEArithmeticWordImpl<MaxOp> {};

template <>
struct NativeWordImpl<SVEBackend, CopySignOp> {
  template <nint_t Index, FloatingTag Tag>
  static VECOPS_ALWAYS_INLINE NativeWordVec<Tag> call(
      CopySignOp, Tag, NativeWordVec<Tag> magnitude,
      NativeWordVec<Tag> sign) {
    using T = ElementOf<Tag>;
    const auto magnitude_raw = sve_basic_raw_word(magnitude);
    const auto sign_raw = sve_basic_raw_word(sign);
    if constexpr (std::same_as<T, bfloat16_t>) {
      const auto magnitude_bits = svreinterpret_u16_bf16(magnitude_raw);
      const auto sign_bits = svreinterpret_u16_bf16(sign_raw);
      const auto changed = sveor_u16_x(svptrue_b16(), magnitude_bits, sign_bits);
      const auto bits = sveor_u16_x(
          svptrue_b16(), magnitude_bits,
          svand_n_u16_x(svptrue_b16(), changed, 0x8000u));
      return sve_basic_wrap_word<Tag>(svreinterpret_bf16_u16(bits));
    } else if constexpr (std::same_as<T, float16_t>) {
      const auto magnitude_bits = svreinterpret_u16_f16(magnitude_raw);
      const auto sign_bits = svreinterpret_u16_f16(sign_raw);
      const auto changed = sveor_u16_x(svptrue_b16(), magnitude_bits, sign_bits);
      const auto bits = sveor_u16_x(
          svptrue_b16(), magnitude_bits,
          svand_n_u16_x(svptrue_b16(), changed, 0x8000u));
      return sve_basic_wrap_word<Tag>(svreinterpret_f16_u16(bits));
    } else if constexpr (std::same_as<T, float32_t>) {
      const auto magnitude_bits = svreinterpret_u32_f32(magnitude_raw);
      const auto sign_bits = svreinterpret_u32_f32(sign_raw);
      const auto changed = sveor_u32_x(svptrue_b32(), magnitude_bits, sign_bits);
      const auto bits = sveor_u32_x(
          svptrue_b32(), magnitude_bits,
          svand_n_u32_x(svptrue_b32(), changed, 0x80000000u));
      return sve_basic_wrap_word<Tag>(svreinterpret_f32_u32(bits));
    } else {
      const auto magnitude_bits = svreinterpret_u64_f64(magnitude_raw);
      const auto sign_bits = svreinterpret_u64_f64(sign_raw);
      const auto changed = sveor_u64_x(svptrue_b64(), magnitude_bits, sign_bits);
      const auto bits = sveor_u64_x(
          svptrue_b64(), magnitude_bits,
          svand_n_u64_x(
              svptrue_b64(), changed, 0x8000000000000000ull));
      return sve_basic_wrap_word<Tag>(svreinterpret_f64_u64(bits));
    }
  }

  template <nint_t Index, FloatingTag Tag, typename Policy>
  static VECOPS_ALWAYS_INLINE NativeWordVec<Tag> call(
      CopySignOp op, Tag tag, NativeWordVec<Tag> magnitude,
      NativeWordVec<Tag> sign, NativeWordMask<Tag> mask,
      NativeWordVec<Tag> inactive, Policy) {
    return blend(
        tag, inactive, mask, call<Index>(op, tag, magnitude, sign));
  }
};

/* **************************************************************************** */
//    SVEClampWordImpl                                                         //
/* **************************************************************************** */

struct SVEClampWordImpl {
  template <nint_t Index, VectorTag Tag>
  static VECOPS_ALWAYS_INLINE NativeWordVec<Tag> call(
      ClampOp, Tag tag, NativeWordVec<Tag> value,
      NativeWordVec<Tag> lower, NativeWordVec<Tag> upper) {
    using T = ElementOf<Tag>;
    static_assert(Index >= 0 && Index < num_words(tag));
    if constexpr (std::same_as<T, bfloat16_t>) {
      return sve_basic_wrap_word<Tag>(sve_bfloat16_clamp(
          sve_basic_raw_word(value), sve_basic_raw_word(lower),
          sve_basic_raw_word(upper)));
    }
    // ACLE permits these intrinsics in an ordinary non-streaming function
    // only with SVE2.1. SME enables integer CLAMP in streaming functions and
    // SME2 enables floating CLAMP there, but this vec backend is non-streaming.
#if defined(HAS_SVE2P1)
    else {
      const auto raw_value = sve_basic_raw_word(value);
      const auto raw_lower = sve_basic_raw_word(lower);
      const auto raw_upper = sve_basic_raw_word(upper);
#define VECOPS_VEC_SVE_CLAMP(Suffix)                                   \
      return sve_basic_wrap_word<Tag>(                                 \
          svclamp_##Suffix(raw_value, raw_lower, raw_upper))
      if constexpr (std::same_as<T, float16_t>) {
        VECOPS_VEC_SVE_CLAMP(f16);
      } else if constexpr (std::same_as<T, float32_t>) {
        VECOPS_VEC_SVE_CLAMP(f32);
      } else if constexpr (std::same_as<T, float64_t>) {
        VECOPS_VEC_SVE_CLAMP(f64);
      } else if constexpr (std::same_as<T, int8_t>) {
        VECOPS_VEC_SVE_CLAMP(s8);
      } else if constexpr (std::same_as<T, uint8_t>) {
        VECOPS_VEC_SVE_CLAMP(u8);
      } else if constexpr (std::same_as<T, int16_t>) {
        VECOPS_VEC_SVE_CLAMP(s16);
      } else if constexpr (std::same_as<T, uint16_t>) {
        VECOPS_VEC_SVE_CLAMP(u16);
      } else if constexpr (std::same_as<T, int32_t>) {
        VECOPS_VEC_SVE_CLAMP(s32);
      } else if constexpr (std::same_as<T, uint32_t>) {
        VECOPS_VEC_SVE_CLAMP(u32);
      } else if constexpr (std::same_as<T, int64_t>) {
        VECOPS_VEC_SVE_CLAMP(s64);
      } else if constexpr (std::same_as<T, uint64_t>) {
        VECOPS_VEC_SVE_CLAMP(u64);
      } else {
        static_assert(dispatch_dependent_false<T>);
      }
#undef VECOPS_VEC_SVE_CLAMP
    }
#endif
    const auto bounded_low =
        SVEArithmeticWordImpl<MaxOp>::template call<Index>(
            MaxOp{}, tag, value, lower);
    return SVEArithmeticWordImpl<MinOp>::template call<Index>(
        MinOp{}, tag, bounded_low, upper);
  }

  template <nint_t Index, VectorTag Tag, typename Policy>
  static VECOPS_ALWAYS_INLINE NativeWordVec<Tag> call(
      ClampOp op, Tag tag, NativeWordVec<Tag> value,
      NativeWordVec<Tag> lower, NativeWordVec<Tag> upper,
      NativeWordMask<Tag> mask, NativeWordVec<Tag> inactive,
      Policy policy) {
    using T = ElementOf<Tag>;
    if constexpr (std::same_as<T, bfloat16_t>) {
      const auto computed = call<Index>(op, tag, value, lower, upper);
      return blend(tag, inactive, mask, computed);
    }
#if defined(HAS_SVE2P1)
    else {
      const auto computed = call<Index>(op, tag, value, lower, upper);
      return blend(tag, inactive, mask, computed);
    }
#endif
    const auto bounded_low =
        SVEArithmeticWordImpl<MaxOp>::template call<Index>(
            MaxOp{}, tag, value, lower);
    return SVEArithmeticWordImpl<MinOp>::template call<Index>(
        MinOp{}, tag, bounded_low, upper, mask, inactive, policy);
  }
};

template <>
struct NativeWordImpl<SVEBackend, ClampOp> : SVEClampWordImpl {};

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
    const auto raw_result = [&]() VECOPS_KERNEL_LAMBDA {
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
    const auto result = [&]() VECOPS_KERNEL_LAMBDA {
      if constexpr (std::same_as<T, bfloat16_t>) {
        const auto low = sve_bf16_to_f32_lo(raw_value);
        const auto high = sve_bf16_to_f32_hi(raw_value);
        const auto inactive_low =
            sve_bf16_to_f32_lo(raw_inactive);
        const auto inactive_high =
            sve_bf16_to_f32_hi(raw_inactive);
        const auto mask_low = svunpklo_b(mask);
        const auto mask_high = svunpkhi_b(mask);
        const auto compute = [](
            svfloat32_t input, svfloat32_t fallback, svbool_t active)
            VECOPS_KERNEL_LAMBDA {
          if constexpr (std::same_as<Op, SqrtOp>)
            return svsqrt_f32_m(fallback, active, input);
          else
            static_assert(dispatch_dependent_false<Op>);
        };
        return sve_f32_pair_to_bf16(
            compute(low, inactive_low, mask_low),
            compute(high, inactive_high, mask_high));
      } else {
#define VECOPS_VEC_SVE_FLOATING_UNARY(Suffix)                          \
        if constexpr (std::same_as<Op, SqrtOp>)                        \
          return svsqrt_##Suffix##_m(raw_inactive, mask, raw_value);   \
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
      const auto computed =
          synthesize_fma_word<SVEBackend, Op, Index>(tag, a, b, c);
      return blend(tag, inactive, mask, computed);
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
