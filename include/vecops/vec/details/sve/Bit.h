#ifndef VECOPS_VEC_DETAILS_SVE_BIT_H
#define VECOPS_VEC_DETAILS_SVE_BIT_H

/**
 * @file Bit.h
 * @brief SVE backend implementations for bitwise operations (bit_and, bit_or, bit_xor, bit_andnot, bit_not, bit_shl, bit_shr).
 */

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

template <typename T, typename Raw>
VECOPS_ALWAYS_INLINE auto sve_bit_shift_left_raw(
    Raw value, int count, svbool_t mask) {
  if constexpr (std::same_as<T, int8_t>)
    return svlsl_n_s8_m(mask, value, static_cast<uint64_t>(count));
  else if constexpr (std::same_as<T, uint8_t>)
    return svlsl_n_u8_m(mask, value, static_cast<uint64_t>(count));
  else if constexpr (std::same_as<T, int16_t>)
    return svlsl_n_s16_m(mask, value, static_cast<uint64_t>(count));
  else if constexpr (std::same_as<T, uint16_t>)
    return svlsl_n_u16_m(mask, value, static_cast<uint64_t>(count));
  else if constexpr (std::same_as<T, int32_t>)
    return svlsl_n_s32_m(mask, value, static_cast<uint64_t>(count));
  else if constexpr (std::same_as<T, uint32_t>)
    return svlsl_n_u32_m(mask, value, static_cast<uint64_t>(count));
  else if constexpr (std::same_as<T, int64_t>)
    return svlsl_n_s64_m(mask, value, static_cast<uint64_t>(count));
  else if constexpr (std::same_as<T, uint64_t>)
    return svlsl_n_u64_m(mask, value, static_cast<uint64_t>(count));
  else static_assert(
      dispatch_dependent_false<T>,
      "SVE bit shift requires an integer element type");
}

template <typename T, typename Raw>
VECOPS_ALWAYS_INLINE auto sve_bit_shift_right_raw(
    Raw value, int count, svbool_t mask) {
  if constexpr (std::same_as<T, int8_t>)
    return svasr_n_s8_m(mask, value, static_cast<uint64_t>(count));
  else if constexpr (std::same_as<T, uint8_t>)
    return svlsr_n_u8_m(mask, value, static_cast<uint64_t>(count));
  else if constexpr (std::same_as<T, int16_t>)
    return svasr_n_s16_m(mask, value, static_cast<uint64_t>(count));
  else if constexpr (std::same_as<T, uint16_t>)
    return svlsr_n_u16_m(mask, value, static_cast<uint64_t>(count));
  else if constexpr (std::same_as<T, int32_t>)
    return svasr_n_s32_m(mask, value, static_cast<uint64_t>(count));
  else if constexpr (std::same_as<T, uint32_t>)
    return svlsr_n_u32_m(mask, value, static_cast<uint64_t>(count));
  else if constexpr (std::same_as<T, int64_t>)
    return svasr_n_s64_m(mask, value, static_cast<uint64_t>(count));
  else if constexpr (std::same_as<T, uint64_t>)
    return svlsr_n_u64_m(mask, value, static_cast<uint64_t>(count));
  else static_assert(
      dispatch_dependent_false<T>,
      "SVE bit shift requires an integer element type");
}

template <typename T, typename Raw>
VECOPS_ALWAYS_INLINE auto sve_bit_shift_left_zero_raw(
    Raw value, int count, svbool_t mask) {
  if constexpr (std::same_as<T, int8_t>) return svlsl_n_s8_z(mask, value, count);
  else if constexpr (std::same_as<T, uint8_t>) return svlsl_n_u8_z(mask, value, count);
  else if constexpr (std::same_as<T, int16_t>) return svlsl_n_s16_z(mask, value, count);
  else if constexpr (std::same_as<T, uint16_t>) return svlsl_n_u16_z(mask, value, count);
  else if constexpr (std::same_as<T, int32_t>) return svlsl_n_s32_z(mask, value, count);
  else if constexpr (std::same_as<T, uint32_t>) return svlsl_n_u32_z(mask, value, count);
  else if constexpr (std::same_as<T, int64_t>) return svlsl_n_s64_z(mask, value, count);
  else if constexpr (std::same_as<T, uint64_t>) return svlsl_n_u64_z(mask, value, count);
}

template <typename T, typename Raw>
VECOPS_ALWAYS_INLINE auto sve_bit_shift_right_zero_raw(
    Raw value, int count, svbool_t mask) {
  if constexpr (std::same_as<T, int8_t>) return svasr_n_s8_z(mask, value, count);
  else if constexpr (std::same_as<T, uint8_t>) return svlsr_n_u8_z(mask, value, count);
  else if constexpr (std::same_as<T, int16_t>) return svasr_n_s16_z(mask, value, count);
  else if constexpr (std::same_as<T, uint16_t>) return svlsr_n_u16_z(mask, value, count);
  else if constexpr (std::same_as<T, int32_t>) return svasr_n_s32_z(mask, value, count);
  else if constexpr (std::same_as<T, uint32_t>) return svlsr_n_u32_z(mask, value, count);
  else if constexpr (std::same_as<T, int64_t>) return svasr_n_s64_z(mask, value, count);
  else if constexpr (std::same_as<T, uint64_t>) return svlsr_n_u64_z(mask, value, count);
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
        return NativeWordImpl<SVEBackend, BlendOp>::template call<Index>(\
            BlendOp{}, tag, inactive, mask, computed);                   \
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
    return NativeWordImpl<SVEBackend, BlendOp>::template call<Index>(
        BlendOp{}, tag, inactive, mask, computed);
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

template <>
struct NativeWordImpl<SVEBackend, BitShiftLeftOp> {
  template <nint_t Index, IntegerTag Tag>
  static VECOPS_ALWAYS_INLINE NativeWordVec<Tag> call(
      BitShiftLeftOp, Tag, NativeWordVec<Tag> value, int count) {
    using Traits = RepresentationTraits<SVEBackend, Tag>;
    using T = ElementOf<Tag>;
    static_assert(Index >= 0 && Index < Traits::word_count);
    return sve_basic_wrap_word<Tag>(sve_bit_shift_left_raw<T>(
        sve_basic_raw_word(value), count, svptrue_b8()));
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

  template <nint_t Index, IntegerTag Tag, typename Policy>
  static VECOPS_ALWAYS_INLINE NativeWordVec<Tag> call(
      BitShiftLeftOp, Tag, NativeWordVec<Tag> value, int count,
      NativeWordMask<Tag> mask, NativeWordVec<Tag> inactive, Policy) {
    using Traits = RepresentationTraits<SVEBackend, Tag>;
    using T = ElementOf<Tag>;
    static_assert(Index >= 0 && Index < Traits::word_count);
    if constexpr (std::same_as<Policy, PreserveBitLanes>)
      return sve_basic_wrap_word<Tag>(sve_bit_shift_left_raw<T>(
          sve_basic_raw_word(value), count, mask));
    else if constexpr (std::same_as<Policy, ZeroBitLanes>)
      return sve_basic_wrap_word<Tag>(sve_bit_shift_left_zero_raw<T>(
          sve_basic_raw_word(value), count, mask));
    else {
      const auto computed = call<Index>(BitShiftLeftOp{}, Tag{}, value, count);
      return NativeWordImpl<SVEBackend, BlendOp>::template call<Index>(
          BlendOp{}, Tag{}, inactive, mask, computed);
    }
  }

  template <nint_t Index, IntegerTag Tag, typename Policy>
  static VECOPS_ALWAYS_INLINE NativeWordVec<Tag> call(
      BitShiftLeftOp op, Tag tag, NativeWordVec<Tag> value,
      NativeWordVec<Tag> counts, NativeWordMask<Tag> mask,
      NativeWordVec<Tag> inactive, Policy) {
    const auto computed = call<Index>(op, tag, value, counts);
    return NativeWordImpl<SVEBackend, BlendOp>::template call<Index>(
        BlendOp{}, tag, inactive, mask, computed);
  }
};

template <>
struct NativeWordImpl<SVEBackend, BitShiftRightOp> {
  template <nint_t Index, IntegerTag Tag>
  static VECOPS_ALWAYS_INLINE NativeWordVec<Tag> call(
      BitShiftRightOp, Tag, NativeWordVec<Tag> value, int count) {
    using Traits = RepresentationTraits<SVEBackend, Tag>;
    using T = ElementOf<Tag>;
    static_assert(Index >= 0 && Index < Traits::word_count);
    return sve_basic_wrap_word<Tag>(sve_bit_shift_right_raw<T>(
        sve_basic_raw_word(value), count, svptrue_b8()));
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

  template <nint_t Index, IntegerTag Tag, typename Policy>
  static VECOPS_ALWAYS_INLINE NativeWordVec<Tag> call(
      BitShiftRightOp, Tag, NativeWordVec<Tag> value, int count,
      NativeWordMask<Tag> mask, NativeWordVec<Tag> inactive, Policy) {
    using Traits = RepresentationTraits<SVEBackend, Tag>;
    using T = ElementOf<Tag>;
    static_assert(Index >= 0 && Index < Traits::word_count);
    if constexpr (std::same_as<Policy, PreserveBitLanes>)
      return sve_basic_wrap_word<Tag>(sve_bit_shift_right_raw<T>(
          sve_basic_raw_word(value), count, mask));
    else if constexpr (std::same_as<Policy, ZeroBitLanes>)
      return sve_basic_wrap_word<Tag>(sve_bit_shift_right_zero_raw<T>(
          sve_basic_raw_word(value), count, mask));
    else {
      const auto computed = call<Index>(BitShiftRightOp{}, Tag{}, value, count);
      return NativeWordImpl<SVEBackend, BlendOp>::template call<Index>(
          BlendOp{}, Tag{}, inactive, mask, computed);
    }
  }

  template <nint_t Index, IntegerTag Tag, typename Policy>
  static VECOPS_ALWAYS_INLINE NativeWordVec<Tag> call(
      BitShiftRightOp op, Tag tag, NativeWordVec<Tag> value,
      NativeWordVec<Tag> counts, NativeWordMask<Tag> mask,
      NativeWordVec<Tag> inactive, Policy) {
    const auto computed = call<Index>(op, tag, value, counts);
    return NativeWordImpl<SVEBackend, BlendOp>::template call<Index>(
        BlendOp{}, tag, inactive, mask, computed);
  }
};

} // namespace vecops::vec::details

#endif // VECOPS_VEC_DETAILS_SVE_BIT_H
