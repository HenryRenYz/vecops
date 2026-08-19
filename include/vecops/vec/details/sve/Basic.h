#ifndef VECOPS_VEC_DETAILS_SVE_BASIC_H
#define VECOPS_VEC_DETAILS_SVE_BASIC_H

/**
 * @file Basic.h
 * @brief SVE backend implementations for fill, mask, blend, lane access,
 * interleave, and shuffle operations.
 *
 * The sizeless SVE mode requires special handling: vector word construction
 * goes through ConstructWordsHook (tuple creation), and mask construction
 * through ConstructMaskWordsHook. Whole-vector operations that exceed the
 * hardware register count use tuple types (svfloat32x2_t, svfloat32x4_t).
 */

#include <algorithm>
#include <type_traits>
#include <utility>

#include "vecops/vec/details/Wordwise.h"

namespace vecops::vec::details {

template <typename Word>

/* **************************************************************************** */
//    Basic word wrappers — sve_basic_raw_word, sve_basic_wrap_word,         //
//    sve_basic_word_bytes                                                   //
/* **************************************************************************** */

VECOPS_ALWAYS_INLINE auto sve_basic_raw_word(Word word) {
  if constexpr (requires { word.value; }) return word.value;
  else return word;
}

template <VectorTag Tag, typename Raw>
VECOPS_ALWAYS_INLINE NativeWordVec<Tag> sve_basic_wrap_word(Raw raw) {
  if constexpr (std::same_as<NativeWordVec<Tag>, Raw>) return raw;
  else return NativeWordVec<Tag>{raw};
}

template <Element T, typename Raw>
VECOPS_ALWAYS_INLINE auto sve_basic_word_bytes(Raw raw) {
  if constexpr (std::same_as<T, bfloat16_t>)
    return svreinterpret_u8_u16(svreinterpret_u16_bf16(raw));
  else if constexpr (std::same_as<T, float16_t>) return svreinterpret_u8_f16(raw);
  else if constexpr (std::same_as<T, float32_t>) return svreinterpret_u8_f32(raw);
  else if constexpr (std::same_as<T, float64_t>) return svreinterpret_u8_f64(raw);
  else if constexpr (std::same_as<T, int8_t>) return svreinterpret_u8_s8(raw);
  else if constexpr (std::same_as<T, uint8_t>) return raw;
  else if constexpr (std::same_as<T, int16_t>) return svreinterpret_u8_s16(raw);
  else if constexpr (std::same_as<T, uint16_t>) return svreinterpret_u8_u16(raw);
  else if constexpr (std::same_as<T, int32_t>) return svreinterpret_u8_s32(raw);
  else if constexpr (std::same_as<T, uint32_t>) return svreinterpret_u8_u32(raw);
  else if constexpr (std::same_as<T, int64_t>) return svreinterpret_u8_s64(raw);
  else if constexpr (std::same_as<T, uint64_t>) return svreinterpret_u8_u64(raw);
  else static_assert(
      dispatch_dependent_false<T>,
      "SVE byte reinterpretation has no implementation for this element type");
}

/* **************************************************************************** */
//    Predicate helpers — sve_prefix_predicate, sve_single_lane_predicate,  //
//    sve_valid_word_lanes                                                   //
/* **************************************************************************** */

template <Element T>
VECOPS_ALWAYS_INLINE svbool_t sve_prefix_predicate(nint_t count) {
  const auto end = static_cast<uint64_t>(count);
  if constexpr (std::same_as<T, bfloat16_t>) {
    return svwhilelt_b16_u64(0, end);
  } else if constexpr (std::same_as<T, float16_t>) {
    return svwhilelt_b16_u64(0, end);
  } else if constexpr (std::same_as<T, float32_t>) {
    return svwhilelt_b32_u64(0, end);
  } else if constexpr (std::same_as<T, float64_t>) {
    return svwhilelt_b64_u64(0, end);
  } else if constexpr (std::same_as<T, int8_t>) {
    return svwhilelt_b8_u64(0, end);
  } else if constexpr (std::same_as<T, uint8_t>) {
    return svwhilelt_b8_u64(0, end);
  } else if constexpr (std::same_as<T, int16_t>) {
    return svwhilelt_b16_u64(0, end);
  } else if constexpr (std::same_as<T, uint16_t>) {
    return svwhilelt_b16_u64(0, end);
  } else if constexpr (std::same_as<T, int32_t>) {
    return svwhilelt_b32_u64(0, end);
  } else if constexpr (std::same_as<T, uint32_t>) {
    return svwhilelt_b32_u64(0, end);
  } else if constexpr (std::same_as<T, int64_t>) {
    return svwhilelt_b64_u64(0, end);
  } else if constexpr (std::same_as<T, uint64_t>) {
    return svwhilelt_b64_u64(0, end);
  } else {
    static_assert(
        dispatch_dependent_false<T>,
        "SVE predicate construction has no implementation for this element type");
  }
}

template <Element T>
VECOPS_ALWAYS_INLINE svbool_t sve_single_lane_predicate(nint_t lane) {
  if constexpr (std::same_as<T, bfloat16_t>) {
    return svcmpeq_n_u16(svptrue_b16(), svindex_u16(0, 1),
                         static_cast<uint16_t>(lane));
  } else if constexpr (std::same_as<T, float16_t>) {
    return svcmpeq_n_u16(svptrue_b16(), svindex_u16(0, 1),
                         static_cast<uint16_t>(lane));
  } else if constexpr (std::same_as<T, float32_t>) {
    return svcmpeq_n_u32(svptrue_b32(), svindex_u32(0, 1),
                         static_cast<uint32_t>(lane));
  } else if constexpr (std::same_as<T, float64_t>) {
    return svcmpeq_n_u64(svptrue_b64(), svindex_u64(0, 1),
                         static_cast<uint64_t>(lane));
  } else if constexpr (std::same_as<T, int8_t>) {
    return svcmpeq_n_u8(svptrue_b8(), svindex_u8(0, 1),
                        static_cast<uint8_t>(lane));
  } else if constexpr (std::same_as<T, uint8_t>) {
    return svcmpeq_n_u8(svptrue_b8(), svindex_u8(0, 1),
                        static_cast<uint8_t>(lane));
  } else if constexpr (std::same_as<T, int16_t>) {
    return svcmpeq_n_u16(svptrue_b16(), svindex_u16(0, 1),
                         static_cast<uint16_t>(lane));
  } else if constexpr (std::same_as<T, uint16_t>) {
    return svcmpeq_n_u16(svptrue_b16(), svindex_u16(0, 1),
                         static_cast<uint16_t>(lane));
  } else if constexpr (std::same_as<T, int32_t>) {
    return svcmpeq_n_u32(svptrue_b32(), svindex_u32(0, 1),
                         static_cast<uint32_t>(lane));
  } else if constexpr (std::same_as<T, uint32_t>) {
    return svcmpeq_n_u32(svptrue_b32(), svindex_u32(0, 1),
                         static_cast<uint32_t>(lane));
  } else if constexpr (std::same_as<T, int64_t>) {
    return svcmpeq_n_u64(svptrue_b64(), svindex_u64(0, 1),
                         static_cast<uint64_t>(lane));
  } else if constexpr (std::same_as<T, uint64_t>) {
    return svcmpeq_n_u64(svptrue_b64(), svindex_u64(0, 1),
                         static_cast<uint64_t>(lane));
  } else {
    static_assert(
        dispatch_dependent_false<T>,
        "SVE lane predicate has no implementation for this element type");
  }
}

template <nint_t Index, VectorTag Tag>
VECOPS_ALWAYS_INLINE nint_t sve_valid_word_lanes(Tag tag) {
  const nint_t word_lanes = native_word_size(tag);
  if constexpr (is_scalable_tag<Tag> && scale_power<Tag> >= 0) {
    // Every physical word of a non-subword scalable tuple is complete. Keep
    // this explicit so Clang does not retain per-word whilelo predicates for
    // a multi-word Tag whose runtime size is expressed through svcnt*().
    return word_lanes;
  } else {
    return std::clamp<nint_t>(
        size(tag) - Index * word_lanes, 0, word_lanes);
  }
}

/* **************************************************************************** */
//    Construction hooks — ConstructWordsHook, ConstructMaskWordsHook        //
/* **************************************************************************** */

template <>
struct ConstructWordsHook<SVEBackend> {
  template <VectorTag Tag, typename Builder>
  static VECOPS_ALWAYS_INLINE Vec<Tag> call(Tag tag, Builder& builder) {
    using Traits = RepresentationTraits<SVEBackend, Tag>;
    if constexpr (requires { sizeof(Vec<Tag>); }) {
      return construct_sized_value<Vec<Tag>>(
          tag,
          builder,
          std::make_index_sequence<
              static_cast<std::size_t>(Traits::word_count)>{});
    } else if constexpr (Traits::word_count == 1) {
      return builder.template operator()<0>(tag);
    } else if constexpr (Traits::word_count == 2) {
      return ::vecops::vec::from_words(
          tag, builder.template operator()<0>(tag),
          builder.template operator()<1>(tag));
    } else if constexpr (Traits::word_count == 4) {
      return ::vecops::vec::from_words(
          tag, builder.template operator()<0>(tag),
          builder.template operator()<1>(tag),
          builder.template operator()<2>(tag),
          builder.template operator()<3>(tag));
    } else {
      static_assert(
          dispatch_dependent_false<Tag>,
          "sizeless SVE vector tuples support only one, two, or four words");
    }
  }
};

template <>
struct ConstructMaskWordsHook<SVEBackend> {
  template <VectorTag Tag, typename Builder>
  static VECOPS_ALWAYS_INLINE Mask<Tag> call(Tag tag, Builder& builder) {
    using Traits = RepresentationTraits<SVEBackend, Tag>;
    if constexpr (requires { sizeof(Mask<Tag>); }) {
      // Fixed-length SVE masks are ordinary arrays. Keeping this path on the
      // generic word accessors avoids any SVE2.1 tuple dependency.
      return construct_sized_value<Mask<Tag>>(
          tag,
          builder,
          std::make_index_sequence<
              static_cast<std::size_t>(Traits::word_count)>{});
    } else if constexpr (Traits::word_count == 1) {
      return builder.template operator()<0>(tag);
    } else if constexpr (Traits::word_count == 2) {
      // A sizeless predicate tuple has no initial value for set_word. Tuple
      // creation is isolated in Types.h because it is an SVE2.1 ACLE operation.
      return ::vecops::vec::mask_from_words(
          tag,
          builder.template operator()<0>(tag),
          builder.template operator()<1>(tag));
    } else if constexpr (Traits::word_count == 4) {
      return ::vecops::vec::mask_from_words(
          tag,
          builder.template operator()<0>(tag),
          builder.template operator()<1>(tag),
          builder.template operator()<2>(tag),
          builder.template operator()<3>(tag));
    } else {
      static_assert(
          dispatch_dependent_false<Tag>,
          "sizeless SVE mask tuples support only one, two, or four words");
    }
  }
};

/* **************************************************************************** */
//    Fill and MaskFill implementations                                      //
/* **************************************************************************** */

template <>
struct NativeWordImpl<SVEBackend, FillOp> {
  template <nint_t Index, VectorTag Tag>
  static VECOPS_ALWAYS_INLINE NativeWordVec<Tag> call(
      FillOp, Tag, ElementOf<Tag> value) {
    using Traits = RepresentationTraits<SVEBackend, Tag>;
    using T = ElementOf<Tag>;
    static_assert(Index >= 0 && Index < Traits::word_count);
    if constexpr (std::same_as<T, bfloat16_t>) {
      return sve_basic_wrap_word<Tag>(
          svreinterpret_bf16_u16(svdup_n_u16(value.to_bits())));
    } else if constexpr (std::same_as<T, float16_t>) {
      return sve_basic_wrap_word<Tag>(
          svreinterpret_f16_u16(svdup_n_u16(value.to_bits())));
    } else if constexpr (std::same_as<T, float32_t>) {
      return sve_basic_wrap_word<Tag>(svdup_n_f32(value));
    } else if constexpr (std::same_as<T, float64_t>) {
      return sve_basic_wrap_word<Tag>(svdup_n_f64(value));
    } else if constexpr (std::same_as<T, int8_t>) {
      return sve_basic_wrap_word<Tag>(svdup_n_s8(value));
    } else if constexpr (std::same_as<T, uint8_t>) {
      return sve_basic_wrap_word<Tag>(svdup_n_u8(value));
    } else if constexpr (std::same_as<T, int16_t>) {
      return sve_basic_wrap_word<Tag>(svdup_n_s16(value));
    } else if constexpr (std::same_as<T, uint16_t>) {
      return sve_basic_wrap_word<Tag>(svdup_n_u16(value));
    } else if constexpr (std::same_as<T, int32_t>) {
      return sve_basic_wrap_word<Tag>(svdup_n_s32(value));
    } else if constexpr (std::same_as<T, uint32_t>) {
      return sve_basic_wrap_word<Tag>(svdup_n_u32(value));
    } else if constexpr (std::same_as<T, int64_t>) {
      return sve_basic_wrap_word<Tag>(svdup_n_s64(value));
    } else if constexpr (std::same_as<T, uint64_t>) {
      return sve_basic_wrap_word<Tag>(svdup_n_u64(value));
    } else {
      static_assert(
          dispatch_dependent_false<T>,
          "SVE fill has no implementation for this element type");
    }
  }
};

template <>
struct NativeWordImpl<SVEBackend, MaskFillOp> {
  template <nint_t Index, VectorTag Tag>
  static VECOPS_ALWAYS_INLINE NativeWordMask<Tag> call(
      MaskFillOp, Tag tag, bool value) {
    using Traits = RepresentationTraits<SVEBackend, Tag>;
    static_assert(Index >= 0 && Index < Traits::word_count);
    return value
        ? sve_prefix_predicate<ElementOf<Tag>>(sve_valid_word_lanes<Index>(tag))
        : svpfalse_b();
  }
};

/* **************************************************************************** */
//    MaskWhile implementations                                              //
/* **************************************************************************** */

template <>
struct NativeWordImpl<SVEBackend, MaskWhileLtOp> {
  template <nint_t Index, VectorTag Tag>
  static VECOPS_ALWAYS_INLINE NativeWordMask<Tag> call(
      MaskWhileLtOp, Tag tag, nint_t a, nint_t b) {
    using Traits = RepresentationTraits<SVEBackend, Tag>;
    static_assert(Index >= 0 && Index < Traits::word_count);
    const nint_t base = Index * native_word_size(tag);
    const nint_t count = std::clamp<nint_t>(
        b - a - base, 0, sve_valid_word_lanes<Index>(tag));
    return sve_prefix_predicate<ElementOf<Tag>>(count);
  }
};

template <>
struct NativeWordImpl<SVEBackend, MaskWhileGeOp> {
  template <nint_t Index, VectorTag Tag>
  static VECOPS_ALWAYS_INLINE NativeWordMask<Tag> call(
      MaskWhileGeOp, Tag tag, nint_t a, nint_t b) {
    using Traits = RepresentationTraits<SVEBackend, Tag>;
    static_assert(Index >= 0 && Index < Traits::word_count);
    const nint_t base = Index * native_word_size(tag);
    const nint_t valid = sve_valid_word_lanes<Index>(tag);
    const nint_t first = std::clamp<nint_t>(b - a - base, 0, valid);
    const auto active = sve_prefix_predicate<ElementOf<Tag>>(valid);
    const auto before = sve_prefix_predicate<ElementOf<Tag>>(first);
    return svbic_b_z(active, active, before);
  }
};

#define VECOPS_VEC_SVE_DEFINE_MASK_CONSTRUCTOR(OpType)                   \
  template <VectorTag Tag>                                               \
  struct NativeImpl<SVEBackend, OpType, Tag> {                           \
    template <typename... Args>                                          \
    static VECOPS_ALWAYS_INLINE Mask<Tag> call(                          \
        OpType op, Tag tag, Args... args) {                              \
      return construct_mask_words<SVEBackend>(                          \
          tag,                                                          \
          [&]<nint_t Index>(Tag) {                                      \
            return NativeWordImpl<SVEBackend, OpType>::template call<Index>( \
                op, tag, args...);                                      \
          });                                                           \
    }                                                                    \
  }

VECOPS_VEC_SVE_DEFINE_MASK_CONSTRUCTOR(MaskFillOp);
VECOPS_VEC_SVE_DEFINE_MASK_CONSTRUCTOR(MaskWhileLtOp);
VECOPS_VEC_SVE_DEFINE_MASK_CONSTRUCTOR(MaskWhileGeOp);

#undef VECOPS_VEC_SVE_DEFINE_MASK_CONSTRUCTOR

/* **************************************************************************** */
//    Blend, mask logic, lane access, bitcast                                //
/* **************************************************************************** */

template <>
struct NativeWordImpl<SVEBackend, BlendOp> {
  template <nint_t Index, VectorTag Tag>
  static VECOPS_ALWAYS_INLINE NativeWordVec<Tag> call(
      BlendOp, Tag,
      NativeWordVec<Tag> false_value,
      NativeWordMask<Tag> mask,
      NativeWordVec<Tag> true_value) {
    using Traits = RepresentationTraits<SVEBackend, Tag>;
    using T = ElementOf<Tag>;
    static_assert(Index >= 0 && Index < Traits::word_count);
    const auto v0 = sve_basic_raw_word(false_value);
    const auto v1 = sve_basic_raw_word(true_value);
    if constexpr (std::same_as<T, bfloat16_t>) {
      return sve_basic_wrap_word<Tag>(svreinterpret_bf16_u16(svsel_u16(
          mask, svreinterpret_u16_bf16(v1), svreinterpret_u16_bf16(v0))));
    } else if constexpr (std::same_as<T, float16_t>) {
      return sve_basic_wrap_word<Tag>(svsel_f16(mask, v1, v0));
    } else if constexpr (std::same_as<T, float32_t>) {
      return sve_basic_wrap_word<Tag>(svsel_f32(mask, v1, v0));
    } else if constexpr (std::same_as<T, float64_t>) {
      return sve_basic_wrap_word<Tag>(svsel_f64(mask, v1, v0));
    } else if constexpr (std::same_as<T, int8_t>) {
      return sve_basic_wrap_word<Tag>(svsel_s8(mask, v1, v0));
    } else if constexpr (std::same_as<T, uint8_t>) {
      return sve_basic_wrap_word<Tag>(svsel_u8(mask, v1, v0));
    } else if constexpr (std::same_as<T, int16_t>) {
      return sve_basic_wrap_word<Tag>(svsel_s16(mask, v1, v0));
    } else if constexpr (std::same_as<T, uint16_t>) {
      return sve_basic_wrap_word<Tag>(svsel_u16(mask, v1, v0));
    } else if constexpr (std::same_as<T, int32_t>) {
      return sve_basic_wrap_word<Tag>(svsel_s32(mask, v1, v0));
    } else if constexpr (std::same_as<T, uint32_t>) {
      return sve_basic_wrap_word<Tag>(svsel_u32(mask, v1, v0));
    } else if constexpr (std::same_as<T, int64_t>) {
      return sve_basic_wrap_word<Tag>(svsel_s64(mask, v1, v0));
    } else if constexpr (std::same_as<T, uint64_t>) {
      return sve_basic_wrap_word<Tag>(svsel_u64(mask, v1, v0));
    } else {
      static_assert(
          dispatch_dependent_false<T>,
          "SVE blend has no implementation for this element type");
    }
  }
};

#define VECOPS_VEC_SVE_DEFINE_MASK_BINARY(OpType, Expression)            \
  template <>                                                            \
  struct NativeWordImpl<SVEBackend, OpType> {                            \
    template <nint_t Index, VectorTag Tag>                               \
    static VECOPS_ALWAYS_INLINE NativeWordMask<Tag> call(                \
        OpType, Tag tag, NativeWordMask<Tag> a, NativeWordMask<Tag> b) { \
      using Traits = RepresentationTraits<SVEBackend, Tag>;              \
      static_assert(Index >= 0 && Index < Traits::word_count);           \
      const auto active = sve_prefix_predicate<ElementOf<Tag>>(          \
          sve_valid_word_lanes<Index>(tag));                             \
      return (Expression);                                               \
    }                                                                    \
  }

VECOPS_VEC_SVE_DEFINE_MASK_BINARY(MaskAndOp, svand_b_z(active, a, b));
VECOPS_VEC_SVE_DEFINE_MASK_BINARY(MaskOrOp, svorr_b_z(active, a, b));
VECOPS_VEC_SVE_DEFINE_MASK_BINARY(MaskXorOp, sveor_b_z(active, a, b));
VECOPS_VEC_SVE_DEFINE_MASK_BINARY(MaskAndNotOp, svbic_b_z(active, b, a));

#undef VECOPS_VEC_SVE_DEFINE_MASK_BINARY

template <>
struct NativeWordImpl<SVEBackend, MaskNotOp> {
  template <nint_t Index, VectorTag Tag>
  static VECOPS_ALWAYS_INLINE NativeWordMask<Tag> call(
      MaskNotOp, Tag tag, NativeWordMask<Tag> value) {
    using Traits = RepresentationTraits<SVEBackend, Tag>;
    static_assert(Index >= 0 && Index < Traits::word_count);
    const auto active = sve_prefix_predicate<ElementOf<Tag>>(
        sve_valid_word_lanes<Index>(tag));
    return svnot_b_z(active, value);
  }
};

template <>
struct NativeWordImpl<SVEBackend, GetVecLaneOp> {
  template <nint_t Index, VectorTag Tag>
  static VECOPS_ALWAYS_INLINE ElementOf<Tag> call(
      GetVecLaneOp, Tag, NativeWordVec<Tag> value, nint_t lane) {
    using Traits = RepresentationTraits<SVEBackend, Tag>;
    using T = ElementOf<Tag>;
    static_assert(Index >= 0 && Index < Traits::word_count);
    const auto predicate = sve_single_lane_predicate<T>(lane);
    const auto raw = sve_basic_raw_word(value);
    if constexpr (std::same_as<T, bfloat16_t>) {
      return bfloat16_t::from_bits(
          svlastb_u16(predicate, svreinterpret_u16_bf16(raw)));
    } else if constexpr (std::same_as<T, float16_t>) {
      return float16_t{svlastb_f16(predicate, raw)};
    } else if constexpr (std::same_as<T, float32_t>) {
      return svlastb_f32(predicate, raw);
    } else if constexpr (std::same_as<T, float64_t>) {
      return svlastb_f64(predicate, raw);
    } else if constexpr (std::same_as<T, int8_t>) {
      return svlastb_s8(predicate, raw);
    } else if constexpr (std::same_as<T, uint8_t>) {
      return svlastb_u8(predicate, raw);
    } else if constexpr (std::same_as<T, int16_t>) {
      return svlastb_s16(predicate, raw);
    } else if constexpr (std::same_as<T, uint16_t>) {
      return svlastb_u16(predicate, raw);
    } else if constexpr (std::same_as<T, int32_t>) {
      return svlastb_s32(predicate, raw);
    } else if constexpr (std::same_as<T, uint32_t>) {
      return svlastb_u32(predicate, raw);
    } else if constexpr (std::same_as<T, int64_t>) {
      return svlastb_s64(predicate, raw);
    } else if constexpr (std::same_as<T, uint64_t>) {
      return svlastb_u64(predicate, raw);
    } else {
      static_assert(
          dispatch_dependent_false<T>,
          "SVE lane get has no implementation for this element type");
    }
  }
};

template <>
struct NativeWordImpl<SVEBackend, SetVecLaneOp> {
  template <nint_t Index, VectorTag Tag>
  static VECOPS_ALWAYS_INLINE NativeWordVec<Tag> call(
      SetVecLaneOp, Tag tag, NativeWordVec<Tag> value,
      nint_t lane, ElementOf<Tag> replacement) {
    using Traits = RepresentationTraits<SVEBackend, Tag>;
    using T = ElementOf<Tag>;
    static_assert(Index >= 0 && Index < Traits::word_count);
    const auto predicate = sve_single_lane_predicate<T>(lane);
    const auto raw = sve_basic_raw_word(value);
    const auto replacement_word =
        NativeWordImpl<SVEBackend, FillOp>::template call<Index>(
            FillOp{}, tag, replacement);
    const auto replacement_raw = sve_basic_raw_word(replacement_word);
    if constexpr (std::same_as<T, bfloat16_t>) {
      return sve_basic_wrap_word<Tag>(svreinterpret_bf16_u16(svsel_u16(
          predicate,
          svreinterpret_u16_bf16(replacement_raw),
          svreinterpret_u16_bf16(raw))));
    } else if constexpr (std::same_as<T, float16_t>) {
      return sve_basic_wrap_word<Tag>(svsel_f16(predicate, replacement_raw, raw));
    } else if constexpr (std::same_as<T, float32_t>) {
      return sve_basic_wrap_word<Tag>(svsel_f32(predicate, replacement_raw, raw));
    } else if constexpr (std::same_as<T, float64_t>) {
      return sve_basic_wrap_word<Tag>(svsel_f64(predicate, replacement_raw, raw));
    } else if constexpr (std::same_as<T, int8_t>) {
      return sve_basic_wrap_word<Tag>(svsel_s8(predicate, replacement_raw, raw));
    } else if constexpr (std::same_as<T, uint8_t>) {
      return sve_basic_wrap_word<Tag>(svsel_u8(predicate, replacement_raw, raw));
    } else if constexpr (std::same_as<T, int16_t>) {
      return sve_basic_wrap_word<Tag>(svsel_s16(predicate, replacement_raw, raw));
    } else if constexpr (std::same_as<T, uint16_t>) {
      return sve_basic_wrap_word<Tag>(svsel_u16(predicate, replacement_raw, raw));
    } else if constexpr (std::same_as<T, int32_t>) {
      return sve_basic_wrap_word<Tag>(svsel_s32(predicate, replacement_raw, raw));
    } else if constexpr (std::same_as<T, uint32_t>) {
      return sve_basic_wrap_word<Tag>(svsel_u32(predicate, replacement_raw, raw));
    } else if constexpr (std::same_as<T, int64_t>) {
      return sve_basic_wrap_word<Tag>(svsel_s64(predicate, replacement_raw, raw));
    } else if constexpr (std::same_as<T, uint64_t>) {
      return sve_basic_wrap_word<Tag>(svsel_u64(predicate, replacement_raw, raw));
    } else {
      static_assert(
          dispatch_dependent_false<T>,
          "SVE lane set has no implementation for this element type");
    }
  }
};

template <>
struct NativeWordImpl<SVEBackend, GetMaskLaneOp> {
  template <nint_t Index, VectorTag Tag>
  static VECOPS_ALWAYS_INLINE bool call(
      GetMaskLaneOp, Tag tag, NativeWordMask<Tag> value, nint_t lane) {
    using Traits = RepresentationTraits<SVEBackend, Tag>;
    static_assert(Index >= 0 && Index < Traits::word_count);
    const auto active = sve_prefix_predicate<ElementOf<Tag>>(
        sve_valid_word_lanes<Index>(tag));
    const auto selected = sve_single_lane_predicate<ElementOf<Tag>>(lane);
    return svptest_any(active, svand_b_z(active, value, selected));
  }
};

template <>
struct NativeWordImpl<SVEBackend, SetMaskLaneOp> {
  template <nint_t Index, VectorTag Tag>
  static VECOPS_ALWAYS_INLINE NativeWordMask<Tag> call(
      SetMaskLaneOp, Tag tag, NativeWordMask<Tag> value,
      nint_t lane, bool replacement) {
    using Traits = RepresentationTraits<SVEBackend, Tag>;
    static_assert(Index >= 0 && Index < Traits::word_count);
    const auto active = sve_prefix_predicate<ElementOf<Tag>>(
        sve_valid_word_lanes<Index>(tag));
    const auto selected = sve_single_lane_predicate<ElementOf<Tag>>(lane);
    return replacement
        ? svorr_b_z(active, value, selected)
        : svbic_b_z(active, value, selected);
  }
};

template <>
struct NativeWordImpl<SVEBackend, BitCastOp> {
  template <nint_t Index, VectorTag ToTag, VectorTag FromTag>
  static VECOPS_ALWAYS_INLINE NativeWordVec<ToTag> call(
      BitCastOp, ToTag, FromTag, NativeWordVec<FromTag> value) {
    using ToTraits = RepresentationTraits<SVEBackend, ToTag>;
    using From = ElementOf<FromTag>;
    using To = ElementOf<ToTag>;
    static_assert(Index >= 0 && Index < ToTraits::word_count);
    const auto raw = sve_basic_raw_word(value);
    const auto bits = [&] {
      if constexpr (std::same_as<From, bfloat16_t>) {
#if defined(HAS_BF16)
        return svreinterpret_u8_bf16(raw);
#else
        return svreinterpret_u8_u16(svreinterpret_u16_bf16(raw));
#endif
      } else if constexpr (std::same_as<From, float16_t>) {
        return svreinterpret_u8_f16(raw);
      } else if constexpr (std::same_as<From, float32_t>) {
        return svreinterpret_u8_f32(raw);
      } else if constexpr (std::same_as<From, float64_t>) {
        return svreinterpret_u8_f64(raw);
      } else if constexpr (std::same_as<From, int8_t>) {
        return svreinterpret_u8_s8(raw);
      } else if constexpr (std::same_as<From, uint8_t>) {
        return raw;
      } else if constexpr (std::same_as<From, int16_t>) {
        return svreinterpret_u8_s16(raw);
      } else if constexpr (std::same_as<From, uint16_t>) {
        return svreinterpret_u8_u16(raw);
      } else if constexpr (std::same_as<From, int32_t>) {
        return svreinterpret_u8_s32(raw);
      } else if constexpr (std::same_as<From, uint32_t>) {
        return svreinterpret_u8_u32(raw);
      } else if constexpr (std::same_as<From, int64_t>) {
        return svreinterpret_u8_s64(raw);
      } else if constexpr (std::same_as<From, uint64_t>) {
        return svreinterpret_u8_u64(raw);
      } else {
        static_assert(
            dispatch_dependent_false<From>,
            "SVE bitcast has no implementation for this source element type");
      }
    }();

    if constexpr (std::same_as<To, bfloat16_t>) {
#if defined(HAS_BF16)
      return sve_basic_wrap_word<ToTag>(svreinterpret_bf16_u8(bits));
#else
      return sve_basic_wrap_word<ToTag>(svreinterpret_bf16_u16(
          svreinterpret_u16_u8(bits)));
#endif
    } else if constexpr (std::same_as<To, float16_t>) {
      return sve_basic_wrap_word<ToTag>(svreinterpret_f16_u8(bits));
    } else if constexpr (std::same_as<To, float32_t>) {
      return sve_basic_wrap_word<ToTag>(svreinterpret_f32_u8(bits));
    } else if constexpr (std::same_as<To, float64_t>) {
      return sve_basic_wrap_word<ToTag>(svreinterpret_f64_u8(bits));
    } else if constexpr (std::same_as<To, int8_t>) {
      return sve_basic_wrap_word<ToTag>(svreinterpret_s8_u8(bits));
    } else if constexpr (std::same_as<To, uint8_t>) {
      return sve_basic_wrap_word<ToTag>(bits);
    } else if constexpr (std::same_as<To, int16_t>) {
      return sve_basic_wrap_word<ToTag>(svreinterpret_s16_u8(bits));
    } else if constexpr (std::same_as<To, uint16_t>) {
      return sve_basic_wrap_word<ToTag>(svreinterpret_u16_u8(bits));
    } else if constexpr (std::same_as<To, int32_t>) {
      return sve_basic_wrap_word<ToTag>(svreinterpret_s32_u8(bits));
    } else if constexpr (std::same_as<To, uint32_t>) {
      return sve_basic_wrap_word<ToTag>(svreinterpret_u32_u8(bits));
    } else if constexpr (std::same_as<To, int64_t>) {
      return sve_basic_wrap_word<ToTag>(svreinterpret_s64_u8(bits));
    } else if constexpr (std::same_as<To, uint64_t>) {
      return sve_basic_wrap_word<ToTag>(svreinterpret_u64_u8(bits));
    } else {
      static_assert(
          dispatch_dependent_false<To>,
          "SVE bitcast has no implementation for this destination element type");
    }
  }
};

/* **************************************************************************** */
//    Lower / Upper / Concat implementations                                 //
/* **************************************************************************** */

template <VectorTag ResultTag, typename A, typename B>
VECOPS_ALWAYS_INLINE NativeWordVec<ResultTag> sve_splice_words(
    svbool_t predicate, A a, B b) {
  using T = ElementOf<ResultTag>;
  if constexpr (std::same_as<T, bfloat16_t>) {
    return sve_basic_wrap_word<ResultTag>(svreinterpret_bf16_u16(
        svsplice_u16(
            predicate,
            svreinterpret_u16_bf16(a),
            svreinterpret_u16_bf16(b))));
  } else if constexpr (std::same_as<T, float16_t>) {
    return sve_basic_wrap_word<ResultTag>(svsplice_f16(predicate, a, b));
  } else if constexpr (std::same_as<T, float32_t>) {
    return sve_basic_wrap_word<ResultTag>(svsplice_f32(predicate, a, b));
  } else if constexpr (std::same_as<T, float64_t>) {
    return sve_basic_wrap_word<ResultTag>(svsplice_f64(predicate, a, b));
  } else if constexpr (std::same_as<T, int8_t>) {
    return sve_basic_wrap_word<ResultTag>(svsplice_s8(predicate, a, b));
  } else if constexpr (std::same_as<T, uint8_t>) {
    return sve_basic_wrap_word<ResultTag>(svsplice_u8(predicate, a, b));
  } else if constexpr (std::same_as<T, int16_t>) {
    return sve_basic_wrap_word<ResultTag>(svsplice_s16(predicate, a, b));
  } else if constexpr (std::same_as<T, uint16_t>) {
    return sve_basic_wrap_word<ResultTag>(svsplice_u16(predicate, a, b));
  } else if constexpr (std::same_as<T, int32_t>) {
    return sve_basic_wrap_word<ResultTag>(svsplice_s32(predicate, a, b));
  } else if constexpr (std::same_as<T, uint32_t>) {
    return sve_basic_wrap_word<ResultTag>(svsplice_u32(predicate, a, b));
  } else if constexpr (std::same_as<T, int64_t>) {
    return sve_basic_wrap_word<ResultTag>(svsplice_s64(predicate, a, b));
  } else if constexpr (std::same_as<T, uint64_t>) {
    return sve_basic_wrap_word<ResultTag>(svsplice_u64(predicate, a, b));
  } else {
    static_assert(
        dispatch_dependent_false<T>,
        "SVE splice has no implementation for this element type");
  }
}

template <VectorTag Tag>
  requires (num_words(Tag{}) == 1 || num_words(Half<Tag>{}) == 1)
struct NativeImpl<SVEBackend, LowerOp, Tag> {
  static VECOPS_ALWAYS_INLINE Vec<Half<Tag>> call(
      LowerOp, Tag tag, Vec<Tag> value) {
    if constexpr (num_words(Tag{}) == 1) {
      return value;
    } else {
      return ::vecops::vec::get_word<0>(tag, value);
    }
  }

  static VECOPS_ALWAYS_INLINE Mask<Half<Tag>> call(
      LowerOp, Tag tag, Mask<Tag> value) {
    if constexpr (num_words(Tag{}) == 1) {
      return value;
    } else {
      return ::vecops::vec::get_word<0>(tag, value);
    }
  }
};

template <VectorTag Tag>
  requires (num_words(Tag{}) == 1 || num_words(Half<Tag>{}) == 1)
struct NativeImpl<SVEBackend, UpperOp, Tag> {
  static VECOPS_ALWAYS_INLINE Vec<Half<Tag>> call(
      UpperOp, Tag tag, Vec<Tag> value) {
    if constexpr (num_words(Tag{}) > 1 && !is_subword<Half<Tag>>) {
      return ::vecops::vec::get_word<1>(tag, value);
    }
    using T = ElementOf<Tag>;
    const auto word0 = sve_basic_raw_word(
        ::vecops::vec::get_word<0>(tag, value));
    const auto word1 = sve_basic_raw_word(
        ::vecops::vec::get_word<(num_words(Tag{}) > 1 ? 1 : 0)>(tag, value));
    const auto physical = sve_prefix_predicate<T>(native_word_size(tag));
    const auto before_upper = sve_prefix_predicate<T>(size(Half<Tag>{}));
    const auto upper = svbic_b_z(physical, physical, before_upper);
    return sve_splice_words<Half<Tag>>(upper, word0, word1);
  }

  static VECOPS_ALWAYS_INLINE Mask<Half<Tag>> call(
      UpperOp, Tag tag, Mask<Tag> value) {
    if constexpr (num_words(Tag{}) > 1) {
      const auto word0 = ::vecops::vec::get_word<0>(tag, value);
      const auto word1 = ::vecops::vec::get_word<1>(tag, value);
      const auto start_bytes = static_cast<uint64_t>(
          size(Half<Tag>{}) * static_cast<nint_t>(sizeof(ElementOf<Tag>)));
      const auto word_bytes = static_cast<uint64_t>(
          native_word_size(tag) * static_cast<nint_t>(sizeof(ElementOf<Tag>)));
      const auto physical = svwhilelt_b8_u64(0, word_bytes);
      const auto before_upper = svwhilelt_b8_u64(0, start_bytes);
      const auto upper = svbic_b_z(physical, physical, before_upper);
      const auto joined = svsplice_u8(
          upper, svdup_u8_z(word0, 1), svdup_u8_z(word1, 1));
      return svcmpne_n_u8(svptrue_b8(), joined, 0);
    } else {
      const auto half_bytes = static_cast<uint64_t>(
          size(Half<Tag>{}) * static_cast<nint_t>(sizeof(ElementOf<Tag>)));
      const auto lower_bytes = svwhilelt_b8_u64(0, half_bytes);
      const auto bytes = svdup_u8_z(value, 1);
      const auto shifted = svtbl_u8(bytes, svindex_u8(
          static_cast<uint8_t>(half_bytes), 1));
      return svcmpne_n_u8(
          svptrue_b8(),
          svsel_u8(lower_bytes, shifted, svdup_n_u8(0)),
          0);
    }
  }
};

template <VectorTag Tag>
  requires (num_words(Tag{}) == 1 || num_words(Half<Tag>{}) == 1)
struct NativeImpl<SVEBackend, ConcatOp, Tag> {
  static VECOPS_ALWAYS_INLINE Vec<Tag> call(
      ConcatOp, Tag tag,
      Vec<Half<Tag>> lower_value,
      Vec<Half<Tag>> upper_value) {
    using T = ElementOf<Tag>;
    if constexpr (num_words(Tag{}) > 1 && !is_subword<Half<Tag>>) {
      const auto lower_raw = sve_basic_raw_word(lower_value);
      const auto upper_raw = sve_basic_raw_word(upper_value);
      return construct_words<SVEBackend>(
          tag,
          [&]<nint_t Index>(Tag) -> NativeWordVec<Tag> {
            static_assert(Index == 0 || Index == 1);
            if constexpr (Index == 0)
              return sve_basic_wrap_word<Tag>(lower_raw);
            else
              return sve_basic_wrap_word<Tag>(upper_raw);
          });
    } else if constexpr (num_words(Tag{}) > 1) {
      const auto lower_raw = sve_basic_raw_word(lower_value);
      const auto upper_raw = sve_basic_raw_word(upper_value);
      const nint_t half_lanes = size(Half<Tag>{});
      const nint_t word_lanes = native_word_size(tag);
      return construct_words<SVEBackend>(
          tag,
          [&]<nint_t Index>(Tag) -> NativeWordVec<Tag> {
            static_assert(Index == 0 || Index == 1);
            if constexpr (Index == 0) {
              return sve_splice_words<Tag>(
                  sve_prefix_predicate<T>(half_lanes), lower_raw, upper_raw);
            } else {
              const auto before_tail = sve_prefix_predicate<T>(
                  word_lanes - half_lanes);
              const auto physical = sve_prefix_predicate<T>(word_lanes);
              const auto tail = svbic_b_z(physical, physical, before_tail);
              return sve_splice_words<Tag>(tail, upper_raw, upper_raw);
            }
          });
    } else {
      const auto lower_raw = sve_basic_raw_word(lower_value);
      const auto upper_raw = sve_basic_raw_word(upper_value);
      const auto lower = sve_prefix_predicate<T>(size(Half<Tag>{}));
      if constexpr (std::same_as<T, bfloat16_t>) {
#if defined(HAS_BF16)
        return sve_basic_wrap_word<Tag>(
            svsplice_bf16(lower, lower_raw, upper_raw));
#else
        return sve_basic_wrap_word<Tag>(svreinterpret_bf16_u16(
            svsplice_u16(
                lower,
                svreinterpret_u16_bf16(lower_raw),
                svreinterpret_u16_bf16(upper_raw))));
#endif
      } else if constexpr (std::same_as<T, float16_t>) {
        return sve_basic_wrap_word<Tag>(svsplice_f16(lower, lower_raw, upper_raw));
      } else if constexpr (std::same_as<T, float32_t>) {
        return sve_basic_wrap_word<Tag>(svsplice_f32(lower, lower_raw, upper_raw));
      } else if constexpr (std::same_as<T, float64_t>) {
        return sve_basic_wrap_word<Tag>(svsplice_f64(lower, lower_raw, upper_raw));
      } else if constexpr (std::same_as<T, int8_t>) {
        return sve_basic_wrap_word<Tag>(svsplice_s8(lower, lower_raw, upper_raw));
      } else if constexpr (std::same_as<T, uint8_t>) {
        return sve_basic_wrap_word<Tag>(svsplice_u8(lower, lower_raw, upper_raw));
      } else if constexpr (std::same_as<T, int16_t>) {
        return sve_basic_wrap_word<Tag>(svsplice_s16(lower, lower_raw, upper_raw));
      } else if constexpr (std::same_as<T, uint16_t>) {
        return sve_basic_wrap_word<Tag>(svsplice_u16(lower, lower_raw, upper_raw));
      } else if constexpr (std::same_as<T, int32_t>) {
        return sve_basic_wrap_word<Tag>(svsplice_s32(lower, lower_raw, upper_raw));
      } else if constexpr (std::same_as<T, uint32_t>) {
        return sve_basic_wrap_word<Tag>(svsplice_u32(lower, lower_raw, upper_raw));
      } else if constexpr (std::same_as<T, int64_t>) {
        return sve_basic_wrap_word<Tag>(svsplice_s64(lower, lower_raw, upper_raw));
      } else if constexpr (std::same_as<T, uint64_t>) {
        return sve_basic_wrap_word<Tag>(svsplice_u64(lower, lower_raw, upper_raw));
      } else {
        static_assert(
            dispatch_dependent_false<T>,
            "SVE concat has no implementation for this element type");
      }
    }
  }

  static VECOPS_ALWAYS_INLINE Mask<Tag> call(
      ConcatOp, Tag tag,
      Mask<Half<Tag>> lower_value,
      Mask<Half<Tag>> upper_value) {
    if constexpr (num_words(Tag{}) > 1) {
      const nint_t half_bytes =
          size(Half<Tag>{}) * static_cast<nint_t>(sizeof(ElementOf<Tag>));
      const nint_t word_bytes =
          native_word_size(tag) * static_cast<nint_t>(sizeof(ElementOf<Tag>));
      const auto lower = svdup_u8_z(lower_value, 1);
      const auto upper = svdup_u8_z(upper_value, 1);
      return construct_mask_words<SVEBackend>(
          tag,
          [&]<nint_t Index>(Tag) -> NativeWordMask<Tag> {
            static_assert(Index == 0 || Index == 1);
            if constexpr (Index == 0) {
              const auto prefix = svwhilelt_b8_u64(
                  0, static_cast<uint64_t>(half_bytes));
              return svcmpne_n_u8(
                  svptrue_b8(), svsplice_u8(prefix, lower, upper), 0);
            } else {
              const auto physical = svwhilelt_b8_u64(
                  0, static_cast<uint64_t>(word_bytes));
              const auto before_tail = svwhilelt_b8_u64(
                  0, static_cast<uint64_t>(word_bytes - half_bytes));
              const auto tail = svbic_b_z(physical, physical, before_tail);
              return svcmpne_n_u8(
                  svptrue_b8(), svsplice_u8(tail, upper, upper), 0);
            }
          });
    } else {
      const auto half_bytes = static_cast<uint64_t>(
          size(Half<Tag>{}) * static_cast<nint_t>(sizeof(ElementOf<Tag>)));
      const auto lower_bytes = svwhilelt_b8_u64(0, half_bytes);
      const auto lower = svdup_u8_z(lower_value, 1);
      const auto upper = svdup_u8_z(upper_value, 1);
      const auto joined = svsplice_u8(lower_bytes, lower, upper);
      return svcmpne_n_u8(svptrue_b8(), joined, 0);
    }
  }
};

/* **************************************************************************** */
//    Even / Odd / ConcatEven / ConcatOdd                                   //
/* **************************************************************************** */

#define VECOPS_VEC_SVE_DEFINE_HALF_SELECT(OpType, Intrinsic)             \
  template <VectorTag Tag>                                               \
    requires (num_words(Tag{}) == 1 || num_words(Half<Tag>{}) == 1)      \
  struct NativeImpl<SVEBackend, OpType, Tag> {                           \
    static VECOPS_ALWAYS_INLINE Vec<Half<Tag>> call(                     \
        OpType, Tag tag, Vec<Tag> value) {                               \
      using T = ElementOf<Tag>;                                          \
      const auto word0 = sve_basic_raw_word(                             \
          ::vecops::vec::get_word<0>(tag, value));                      \
      const auto word1 = sve_basic_raw_word(                             \
          ::vecops::vec::get_word<                                     \
              (num_words(Tag{}) > 1 ? 1 : 0)>(tag, value));              \
      if constexpr (std::same_as<T, bfloat16_t>) {                       \
        /* BF16 tuple operations need +bf16; the u16 form is bit-identical. */ \
        return sve_basic_wrap_word<Half<Tag>>(svreinterpret_bf16_u16(    \
            Intrinsic##_u16(                                            \
                svreinterpret_u16_bf16(word0),                           \
                svreinterpret_u16_bf16(word1))));                        \
      } else if constexpr (std::same_as<T, float16_t>) {                 \
        return sve_basic_wrap_word<Half<Tag>>(Intrinsic##_f16(word0, word1)); \
      } else if constexpr (std::same_as<T, float32_t>) {                 \
        return sve_basic_wrap_word<Half<Tag>>(Intrinsic##_f32(word0, word1)); \
      } else if constexpr (std::same_as<T, float64_t>) {                 \
        return sve_basic_wrap_word<Half<Tag>>(Intrinsic##_f64(word0, word1)); \
      } else if constexpr (std::same_as<T, int8_t>) {                    \
        return sve_basic_wrap_word<Half<Tag>>(Intrinsic##_s8(word0, word1)); \
      } else if constexpr (std::same_as<T, uint8_t>) {                   \
        return sve_basic_wrap_word<Half<Tag>>(Intrinsic##_u8(word0, word1)); \
      } else if constexpr (std::same_as<T, int16_t>) {                   \
        return sve_basic_wrap_word<Half<Tag>>(Intrinsic##_s16(word0, word1)); \
      } else if constexpr (std::same_as<T, uint16_t>) {                  \
        return sve_basic_wrap_word<Half<Tag>>(Intrinsic##_u16(word0, word1)); \
      } else if constexpr (std::same_as<T, int32_t>) {                   \
        return sve_basic_wrap_word<Half<Tag>>(Intrinsic##_s32(word0, word1)); \
      } else if constexpr (std::same_as<T, uint32_t>) {                  \
        return sve_basic_wrap_word<Half<Tag>>(Intrinsic##_u32(word0, word1)); \
      } else if constexpr (std::same_as<T, int64_t>) {                   \
        return sve_basic_wrap_word<Half<Tag>>(Intrinsic##_s64(word0, word1)); \
      } else if constexpr (std::same_as<T, uint64_t>) {                  \
        return sve_basic_wrap_word<Half<Tag>>(Intrinsic##_u64(word0, word1)); \
      } else {                                                           \
        static_assert(                                                   \
            dispatch_dependent_false<T>,                                 \
            "SVE half select has no implementation for this element type"); \
      }                                                                  \
    }                                                                    \
  }

VECOPS_VEC_SVE_DEFINE_HALF_SELECT(EvenOp, svuzp1);
VECOPS_VEC_SVE_DEFINE_HALF_SELECT(OddOp, svuzp2);

#undef VECOPS_VEC_SVE_DEFINE_HALF_SELECT

#define VECOPS_VEC_SVE_DEFINE_SAME_SHAPE(OpType, Intrinsic)             \
  template <>                                                            \
  struct NativeWordImpl<SVEBackend, OpType> {                            \
    template <nint_t Index, VectorTag Tag>                               \
    static VECOPS_ALWAYS_INLINE NativeWordVec<Tag> call(                 \
        OpType, Tag, NativeWordVec<Tag> a, NativeWordVec<Tag> b) {       \
      using Traits = RepresentationTraits<SVEBackend, Tag>;              \
      using T = ElementOf<Tag>;                                          \
      static_assert(Index >= 0 && Index < Traits::word_count);           \
      const auto raw_a = sve_basic_raw_word(a);                          \
      const auto raw_b = sve_basic_raw_word(b);                          \
      if constexpr (std::same_as<T, bfloat16_t>) {                       \
        return sve_basic_wrap_word<Tag>(svreinterpret_bf16_u16(          \
            Intrinsic##_u16(                                            \
                svreinterpret_u16_bf16(raw_a),                           \
                svreinterpret_u16_bf16(raw_b))));                        \
      } else if constexpr (std::same_as<T, float16_t>) {                 \
        return sve_basic_wrap_word<Tag>(Intrinsic##_f16(raw_a, raw_b));  \
      } else if constexpr (std::same_as<T, float32_t>) {                 \
        return sve_basic_wrap_word<Tag>(Intrinsic##_f32(raw_a, raw_b));  \
      } else if constexpr (std::same_as<T, float64_t>) {                 \
        return sve_basic_wrap_word<Tag>(Intrinsic##_f64(raw_a, raw_b));  \
      } else if constexpr (std::same_as<T, int8_t>) {                    \
        return sve_basic_wrap_word<Tag>(Intrinsic##_s8(raw_a, raw_b));   \
      } else if constexpr (std::same_as<T, uint8_t>) {                   \
        return sve_basic_wrap_word<Tag>(Intrinsic##_u8(raw_a, raw_b));   \
      } else if constexpr (std::same_as<T, int16_t>) {                   \
        return sve_basic_wrap_word<Tag>(Intrinsic##_s16(raw_a, raw_b));  \
      } else if constexpr (std::same_as<T, uint16_t>) {                  \
        return sve_basic_wrap_word<Tag>(Intrinsic##_u16(raw_a, raw_b));  \
      } else if constexpr (std::same_as<T, int32_t>) {                   \
        return sve_basic_wrap_word<Tag>(Intrinsic##_s32(raw_a, raw_b));  \
      } else if constexpr (std::same_as<T, uint32_t>) {                  \
        return sve_basic_wrap_word<Tag>(Intrinsic##_u32(raw_a, raw_b));  \
      } else if constexpr (std::same_as<T, int64_t>) {                   \
        return sve_basic_wrap_word<Tag>(Intrinsic##_s64(raw_a, raw_b));  \
      } else if constexpr (std::same_as<T, uint64_t>) {                  \
        return sve_basic_wrap_word<Tag>(Intrinsic##_u64(raw_a, raw_b));  \
      } else {                                                           \
        static_assert(                                                   \
            dispatch_dependent_false<T>,                                 \
            "SVE rearrange has no implementation for this element type"); \
      }                                                                  \
    }                                                                    \
  }

VECOPS_VEC_SVE_DEFINE_SAME_SHAPE(InterleaveEvenOp, svtrn1);
VECOPS_VEC_SVE_DEFINE_SAME_SHAPE(InterleaveOddOp, svtrn2);

#undef VECOPS_VEC_SVE_DEFINE_SAME_SHAPE

template <bool Odd, VectorTag Tag>
VECOPS_ALWAYS_INLINE NativeWordVec<Tag> sve_concat_parity_word(
    Tag tag, NativeWordVec<Tag> a, NativeWordVec<Tag> b) {
  using T = ElementOf<Tag>;
  const auto raw_a = sve_basic_raw_word(a);
  const auto raw_b = sve_basic_raw_word(b);
  const auto select = []<typename Raw>(Raw raw) {
    if constexpr (std::same_as<T, bfloat16_t>) {
      const auto bits = svreinterpret_u16_bf16(raw);
      return svreinterpret_bf16_u16(
          Odd ? svuzp2_u16(bits, bits) : svuzp1_u16(bits, bits));
    } else if constexpr (std::same_as<T, float16_t>) {
      return Odd ? svuzp2_f16(raw, raw) : svuzp1_f16(raw, raw);
    } else if constexpr (std::same_as<T, float32_t>) {
      return Odd ? svuzp2_f32(raw, raw) : svuzp1_f32(raw, raw);
    } else if constexpr (std::same_as<T, float64_t>) {
      return Odd ? svuzp2_f64(raw, raw) : svuzp1_f64(raw, raw);
    } else if constexpr (std::same_as<T, int8_t>) {
      return Odd ? svuzp2_s8(raw, raw) : svuzp1_s8(raw, raw);
    } else if constexpr (std::same_as<T, uint8_t>) {
      return Odd ? svuzp2_u8(raw, raw) : svuzp1_u8(raw, raw);
    } else if constexpr (std::same_as<T, int16_t>) {
      return Odd ? svuzp2_s16(raw, raw) : svuzp1_s16(raw, raw);
    } else if constexpr (std::same_as<T, uint16_t>) {
      return Odd ? svuzp2_u16(raw, raw) : svuzp1_u16(raw, raw);
    } else if constexpr (std::same_as<T, int32_t>) {
      return Odd ? svuzp2_s32(raw, raw) : svuzp1_s32(raw, raw);
    } else if constexpr (std::same_as<T, uint32_t>) {
      return Odd ? svuzp2_u32(raw, raw) : svuzp1_u32(raw, raw);
    } else if constexpr (std::same_as<T, int64_t>) {
      return Odd ? svuzp2_s64(raw, raw) : svuzp1_s64(raw, raw);
    } else if constexpr (std::same_as<T, uint64_t>) {
      return Odd ? svuzp2_u64(raw, raw) : svuzp1_u64(raw, raw);
    } else {
      static_assert(
          dispatch_dependent_false<T>,
          "SVE concat parity has no implementation for this element type");
    }
  };
  return sve_splice_words<Tag>(
      sve_prefix_predicate<T>(size(tag) / 2), select(raw_a), select(raw_b));
}

#define VECOPS_VEC_SVE_DEFINE_CONCAT_PARITY(OpType, OddValue)           \
  template <>                                                            \
  struct NativeWordImpl<SVEBackend, OpType> {                            \
    template <nint_t Index, VectorTag Tag>                               \
    static VECOPS_ALWAYS_INLINE NativeWordVec<Tag> call(                 \
        OpType, Tag tag, NativeWordVec<Tag> a, NativeWordVec<Tag> b) {   \
      static_assert(Index == 0);                                         \
      static_assert(num_words(Tag{}) == 1);                              \
      return sve_concat_parity_word<OddValue>(tag, a, b);                \
    }                                                                    \
  }

VECOPS_VEC_SVE_DEFINE_CONCAT_PARITY(ConcatEvenOp, false);
VECOPS_VEC_SVE_DEFINE_CONCAT_PARITY(ConcatOddOp, true);

#undef VECOPS_VEC_SVE_DEFINE_CONCAT_PARITY

/* **************************************************************************** */
//    Interleave / LocalInterleave                                           //
/* **************************************************************************** */

template <VectorTag Tag>
  requires (num_words(Tag{}) == 1 || num_words(Half<Tag>{}) == 1)
struct NativeImpl<SVEBackend, InterleaveOp, Tag> {
  static VECOPS_ALWAYS_INLINE Vec<Tag> call(
      InterleaveOp, Tag tag,
      Vec<Half<Tag>> a,
      Vec<Half<Tag>> b) {
    using T = ElementOf<Tag>;
    const auto raw_a = sve_basic_raw_word(a);
    const auto raw_b = sve_basic_raw_word(b);
    return construct_words<SVEBackend>(
        tag,
        [&]<nint_t Index>(Tag) -> NativeWordVec<Tag> {
          static_assert(Index == 0 || Index == 1);
          if constexpr (std::same_as<T, bfloat16_t>) {
            if constexpr (Index == 0) {
              return sve_basic_wrap_word<Tag>(svreinterpret_bf16_u16(
                  svzip1_u16(
                      svreinterpret_u16_bf16(raw_a),
                      svreinterpret_u16_bf16(raw_b))));
            } else {
              return sve_basic_wrap_word<Tag>(svreinterpret_bf16_u16(
                  svzip2_u16(
                      svreinterpret_u16_bf16(raw_a),
                      svreinterpret_u16_bf16(raw_b))));
            }
          } else if constexpr (std::same_as<T, float16_t>) {
            if constexpr (Index == 0) return sve_basic_wrap_word<Tag>(svzip1_f16(raw_a, raw_b));
            else return sve_basic_wrap_word<Tag>(svzip2_f16(raw_a, raw_b));
          } else if constexpr (std::same_as<T, float32_t>) {
            if constexpr (Index == 0) return sve_basic_wrap_word<Tag>(svzip1_f32(raw_a, raw_b));
            else return sve_basic_wrap_word<Tag>(svzip2_f32(raw_a, raw_b));
          } else if constexpr (std::same_as<T, float64_t>) {
            if constexpr (Index == 0) return sve_basic_wrap_word<Tag>(svzip1_f64(raw_a, raw_b));
            else return sve_basic_wrap_word<Tag>(svzip2_f64(raw_a, raw_b));
          } else if constexpr (std::same_as<T, int8_t>) {
            if constexpr (Index == 0) return sve_basic_wrap_word<Tag>(svzip1_s8(raw_a, raw_b));
            else return sve_basic_wrap_word<Tag>(svzip2_s8(raw_a, raw_b));
          } else if constexpr (std::same_as<T, uint8_t>) {
            if constexpr (Index == 0) return sve_basic_wrap_word<Tag>(svzip1_u8(raw_a, raw_b));
            else return sve_basic_wrap_word<Tag>(svzip2_u8(raw_a, raw_b));
          } else if constexpr (std::same_as<T, int16_t>) {
            if constexpr (Index == 0) return sve_basic_wrap_word<Tag>(svzip1_s16(raw_a, raw_b));
            else return sve_basic_wrap_word<Tag>(svzip2_s16(raw_a, raw_b));
          } else if constexpr (std::same_as<T, uint16_t>) {
            if constexpr (Index == 0) return sve_basic_wrap_word<Tag>(svzip1_u16(raw_a, raw_b));
            else return sve_basic_wrap_word<Tag>(svzip2_u16(raw_a, raw_b));
          } else if constexpr (std::same_as<T, int32_t>) {
            if constexpr (Index == 0) return sve_basic_wrap_word<Tag>(svzip1_s32(raw_a, raw_b));
            else return sve_basic_wrap_word<Tag>(svzip2_s32(raw_a, raw_b));
          } else if constexpr (std::same_as<T, uint32_t>) {
            if constexpr (Index == 0) return sve_basic_wrap_word<Tag>(svzip1_u32(raw_a, raw_b));
            else return sve_basic_wrap_word<Tag>(svzip2_u32(raw_a, raw_b));
          } else if constexpr (std::same_as<T, int64_t>) {
            if constexpr (Index == 0) return sve_basic_wrap_word<Tag>(svzip1_s64(raw_a, raw_b));
            else return sve_basic_wrap_word<Tag>(svzip2_s64(raw_a, raw_b));
          } else if constexpr (std::same_as<T, uint64_t>) {
            if constexpr (Index == 0) return sve_basic_wrap_word<Tag>(svzip1_u64(raw_a, raw_b));
            else return sve_basic_wrap_word<Tag>(svzip2_u64(raw_a, raw_b));
          } else {
            static_assert(
                dispatch_dependent_false<T>,
                "SVE interleave has no implementation for this element type");
          }
        });
  }
};

#define VECOPS_VEC_SVE_DEFINE_LOCAL_INTERLEAVE(OpType, Unzip)            \
  template <>                                                            \
  struct NativeWordImpl<SVEBackend, OpType> {                            \
    template <nint_t Index, VectorTag Tag>                               \
    static VECOPS_ALWAYS_INLINE NativeWordVec<Tag> call(                 \
        OpType, Tag, NativeWordVec<Tag> a, NativeWordVec<Tag> b) {       \
      using Traits = RepresentationTraits<SVEBackend, Tag>;              \
      using T = ElementOf<Tag>;                                          \
      static_assert(Index >= 0 && Index < Traits::word_count);           \
      const auto raw_a = sve_basic_raw_word(a);                          \
      const auto raw_b = sve_basic_raw_word(b);                          \
      const auto a8 = sve_basic_word_bytes<T>(raw_a);                    \
      const auto b8 = sve_basic_word_bytes<T>(raw_b);                    \
      const auto selected_a = Unzip##_u64(                               \
          svreinterpret_u64_u8(a8), svreinterpret_u64_u8(a8));          \
      const auto selected_b = Unzip##_u64(                               \
          svreinterpret_u64_u8(b8), svreinterpret_u64_u8(b8));          \
      const auto selected_a8 = svreinterpret_u8_u64(selected_a);        \
      const auto selected_b8 = svreinterpret_u8_u64(selected_b);        \
      if constexpr (std::same_as<T, bfloat16_t>) {                       \
        return sve_basic_wrap_word<Tag>(svreinterpret_bf16_u16(          \
            svzip1_u16(                                                  \
                svreinterpret_u16_u8(selected_a8),                       \
                svreinterpret_u16_u8(selected_b8))));                    \
      } else if constexpr (std::same_as<T, float16_t>) {                 \
        return sve_basic_wrap_word<Tag>(svzip1_f16(                      \
            svreinterpret_f16_u8(selected_a8), svreinterpret_f16_u8(selected_b8))); \
      } else if constexpr (std::same_as<T, float32_t>) {                 \
        return sve_basic_wrap_word<Tag>(svzip1_f32(                      \
            svreinterpret_f32_u8(selected_a8), svreinterpret_f32_u8(selected_b8))); \
      } else if constexpr (std::same_as<T, float64_t>) {                 \
        return sve_basic_wrap_word<Tag>(svzip1_f64(                      \
            svreinterpret_f64_u8(selected_a8), svreinterpret_f64_u8(selected_b8))); \
      } else if constexpr (std::same_as<T, int8_t>) {                    \
        return sve_basic_wrap_word<Tag>(svzip1_s8(                       \
            svreinterpret_s8_u8(selected_a8), svreinterpret_s8_u8(selected_b8))); \
      } else if constexpr (std::same_as<T, uint8_t>) {                   \
        return sve_basic_wrap_word<Tag>(svzip1_u8(selected_a8, selected_b8)); \
      } else if constexpr (std::same_as<T, int16_t>) {                   \
        return sve_basic_wrap_word<Tag>(svzip1_s16(                      \
            svreinterpret_s16_u8(selected_a8), svreinterpret_s16_u8(selected_b8))); \
      } else if constexpr (std::same_as<T, uint16_t>) {                  \
        return sve_basic_wrap_word<Tag>(svzip1_u16(                      \
            svreinterpret_u16_u8(selected_a8), svreinterpret_u16_u8(selected_b8))); \
      } else if constexpr (std::same_as<T, int32_t>) {                   \
        return sve_basic_wrap_word<Tag>(svzip1_s32(                      \
            svreinterpret_s32_u8(selected_a8), svreinterpret_s32_u8(selected_b8))); \
      } else if constexpr (std::same_as<T, uint32_t>) {                  \
        return sve_basic_wrap_word<Tag>(svzip1_u32(                      \
            svreinterpret_u32_u8(selected_a8), svreinterpret_u32_u8(selected_b8))); \
      } else if constexpr (std::same_as<T, int64_t>) {                   \
        return sve_basic_wrap_word<Tag>(svzip1_s64(                      \
            svreinterpret_s64_u8(selected_a8), svreinterpret_s64_u8(selected_b8))); \
      } else if constexpr (std::same_as<T, uint64_t>) {                  \
        return sve_basic_wrap_word<Tag>(svzip1_u64(                      \
            svreinterpret_u64_u8(selected_a8), svreinterpret_u64_u8(selected_b8))); \
      } else {                                                           \
        static_assert(                                                   \
            dispatch_dependent_false<T>,                                 \
            "SVE local interleave has no implementation for this element type"); \
      }                                                                  \
    }                                                                    \
  }

VECOPS_VEC_SVE_DEFINE_LOCAL_INTERLEAVE(LocalInterleaveLowerOp, svuzp1);
VECOPS_VEC_SVE_DEFINE_LOCAL_INTERLEAVE(LocalInterleaveUpperOp, svuzp2);

#undef VECOPS_VEC_SVE_DEFINE_LOCAL_INTERLEAVE

/* **************************************************************************** */
//    Shuffle                                                                 //
/* **************************************************************************** */

template <VectorTag Tag, typename Raw, typename UnsignedIndices>
VECOPS_ALWAYS_INLINE NativeWordVec<Tag> sve_table_lookup_word(
    Raw value, UnsignedIndices indices) {
  using T = ElementOf<Tag>;
  if constexpr (std::same_as<T, bfloat16_t>) {
    return sve_basic_wrap_word<Tag>(svreinterpret_bf16_u16(
        svtbl_u16(svreinterpret_u16_bf16(value), indices)));
  } else if constexpr (std::same_as<T, float16_t>) {
    return sve_basic_wrap_word<Tag>(svtbl_f16(value, indices));
  } else if constexpr (std::same_as<T, float32_t>) {
    return sve_basic_wrap_word<Tag>(svtbl_f32(value, indices));
  } else if constexpr (std::same_as<T, float64_t>) {
    return sve_basic_wrap_word<Tag>(svtbl_f64(value, indices));
  } else if constexpr (std::same_as<T, int8_t>) {
    return sve_basic_wrap_word<Tag>(svtbl_s8(value, indices));
  } else if constexpr (std::same_as<T, uint8_t>) {
    return sve_basic_wrap_word<Tag>(svtbl_u8(value, indices));
  } else if constexpr (std::same_as<T, int16_t>) {
    return sve_basic_wrap_word<Tag>(svtbl_s16(value, indices));
  } else if constexpr (std::same_as<T, uint16_t>) {
    return sve_basic_wrap_word<Tag>(svtbl_u16(value, indices));
  } else if constexpr (std::same_as<T, int32_t>) {
    return sve_basic_wrap_word<Tag>(svtbl_s32(value, indices));
  } else if constexpr (std::same_as<T, uint32_t>) {
    return sve_basic_wrap_word<Tag>(svtbl_u32(value, indices));
  } else if constexpr (std::same_as<T, int64_t>) {
    return sve_basic_wrap_word<Tag>(svtbl_s64(value, indices));
  } else if constexpr (std::same_as<T, uint64_t>) {
    return sve_basic_wrap_word<Tag>(svtbl_u64(value, indices));
  } else {
    static_assert(
        dispatch_dependent_false<T>,
        "SVE shuffle has no implementation for this element type");
  }
}

template <>
struct NativeWordImpl<SVEBackend, ShuffleOp> {
  template <nint_t Index, VectorTag Tag>
  static VECOPS_ALWAYS_INLINE NativeWordVec<Tag> call(
      ShuffleOp, Tag,
      NativeWordVec<Tag> value,
      NativeWordVec<IndexTag<Tag>> indices) {
    using Traits = RepresentationTraits<SVEBackend, Tag>;
    using T = ElementOf<Tag>;
    static_assert(Index >= 0 && Index < Traits::word_count);
    const auto raw = sve_basic_raw_word(value);
    const auto signed_indices = sve_basic_raw_word(indices);
    if constexpr (std::same_as<T, bfloat16_t>) {
      return sve_table_lookup_word<Tag>(
          raw, svreinterpret_u16_s16(signed_indices));
    } else if constexpr (std::same_as<T, float16_t>) {
      return sve_table_lookup_word<Tag>(
          raw, svreinterpret_u16_s16(signed_indices));
    } else if constexpr (std::same_as<T, float32_t>) {
      return sve_table_lookup_word<Tag>(
          raw, svreinterpret_u32_s32(signed_indices));
    } else if constexpr (std::same_as<T, float64_t>) {
      return sve_table_lookup_word<Tag>(
          raw, svreinterpret_u64_s64(signed_indices));
    } else if constexpr (std::same_as<T, int8_t>) {
      return sve_table_lookup_word<Tag>(
          raw, svreinterpret_u8_s8(signed_indices));
    } else if constexpr (std::same_as<T, uint8_t>) {
      return sve_table_lookup_word<Tag>(
          raw, svreinterpret_u8_s8(signed_indices));
    } else if constexpr (std::same_as<T, int16_t>) {
      return sve_table_lookup_word<Tag>(
          raw, svreinterpret_u16_s16(signed_indices));
    } else if constexpr (std::same_as<T, uint16_t>) {
      return sve_table_lookup_word<Tag>(
          raw, svreinterpret_u16_s16(signed_indices));
    } else if constexpr (std::same_as<T, int32_t>) {
      return sve_table_lookup_word<Tag>(
          raw, svreinterpret_u32_s32(signed_indices));
    } else if constexpr (std::same_as<T, uint32_t>) {
      return sve_table_lookup_word<Tag>(
          raw, svreinterpret_u32_s32(signed_indices));
    } else if constexpr (std::same_as<T, int64_t>) {
      return sve_table_lookup_word<Tag>(
          raw, svreinterpret_u64_s64(signed_indices));
    } else if constexpr (std::same_as<T, uint64_t>) {
      return sve_table_lookup_word<Tag>(
          raw, svreinterpret_u64_s64(signed_indices));
    } else {
      static_assert(
          dispatch_dependent_false<T>,
          "SVE shuffle index conversion has no implementation for this element type");
    }
  }
};

template <>
struct NativeWordImpl<SVEBackend, LocalShuffleOp> {
  template <nint_t Index, VectorTag Tag>
  static VECOPS_ALWAYS_INLINE NativeWordVec<Tag> call(
      LocalShuffleOp, Tag,
      NativeWordVec<Tag> value,
      NativeWordVec<IndexTag<Tag>> indices) {
    using Traits = RepresentationTraits<SVEBackend, Tag>;
    using T = ElementOf<Tag>;
    static_assert(Index >= 0 && Index < Traits::word_count);
    constexpr nint_t block_lanes = 16 / static_cast<nint_t>(sizeof(T));
    constexpr int shift = [] {
      int result = 0;
      for (nint_t lanes = block_lanes; lanes > 1; lanes >>= 1) ++result;
      return result;
    }();
    const auto raw = sve_basic_raw_word(value);
    const auto signed_indices = sve_basic_raw_word(indices);
    if constexpr (std::same_as<T, bfloat16_t>) {
      const auto active = svptrue_b16();
      auto base = svindex_u16(0, 1);
      base = svlsl_n_u16_z(
          active, svlsr_n_u16_z(active, base, shift), shift);
      return sve_table_lookup_word<Tag>(
          raw,
          svadd_u16_z(
              active, base, svreinterpret_u16_s16(signed_indices)));
    } else if constexpr (std::same_as<T, float16_t>) {
      const auto active = svptrue_b16();
      auto base = svindex_u16(0, 1);
      base = svlsl_n_u16_z(
          active, svlsr_n_u16_z(active, base, shift), shift);
      return sve_table_lookup_word<Tag>(
          raw,
          svadd_u16_z(
              active, base, svreinterpret_u16_s16(signed_indices)));
    } else if constexpr (std::same_as<T, float32_t>) {
      const auto active = svptrue_b32();
      auto base = svindex_u32(0, 1);
      base = svlsl_n_u32_z(
          active, svlsr_n_u32_z(active, base, shift), shift);
      return sve_table_lookup_word<Tag>(
          raw,
          svadd_u32_z(
              active, base, svreinterpret_u32_s32(signed_indices)));
    } else if constexpr (std::same_as<T, float64_t>) {
      const auto active = svptrue_b64();
      auto base = svindex_u64(0, 1);
      base = svlsl_n_u64_z(
          active, svlsr_n_u64_z(active, base, shift), shift);
      return sve_table_lookup_word<Tag>(
          raw,
          svadd_u64_z(
              active, base, svreinterpret_u64_s64(signed_indices)));
    } else if constexpr (std::same_as<T, int8_t>) {
      const auto active = svptrue_b8();
      auto base = svindex_u8(0, 1);
      base = svlsl_n_u8_z(
          active, svlsr_n_u8_z(active, base, shift), shift);
      return sve_table_lookup_word<Tag>(
          raw,
          svadd_u8_z(
              active, base, svreinterpret_u8_s8(signed_indices)));
    } else if constexpr (std::same_as<T, uint8_t>) {
      const auto active = svptrue_b8();
      auto base = svindex_u8(0, 1);
      base = svlsl_n_u8_z(
          active, svlsr_n_u8_z(active, base, shift), shift);
      return sve_table_lookup_word<Tag>(
          raw,
          svadd_u8_z(
              active, base, svreinterpret_u8_s8(signed_indices)));
    } else if constexpr (std::same_as<T, int16_t>) {
      const auto active = svptrue_b16();
      auto base = svindex_u16(0, 1);
      base = svlsl_n_u16_z(
          active, svlsr_n_u16_z(active, base, shift), shift);
      return sve_table_lookup_word<Tag>(
          raw,
          svadd_u16_z(
              active, base, svreinterpret_u16_s16(signed_indices)));
    } else if constexpr (std::same_as<T, uint16_t>) {
      const auto active = svptrue_b16();
      auto base = svindex_u16(0, 1);
      base = svlsl_n_u16_z(
          active, svlsr_n_u16_z(active, base, shift), shift);
      return sve_table_lookup_word<Tag>(
          raw,
          svadd_u16_z(
              active, base, svreinterpret_u16_s16(signed_indices)));
    } else if constexpr (std::same_as<T, int32_t>) {
      const auto active = svptrue_b32();
      auto base = svindex_u32(0, 1);
      base = svlsl_n_u32_z(
          active, svlsr_n_u32_z(active, base, shift), shift);
      return sve_table_lookup_word<Tag>(
          raw,
          svadd_u32_z(
              active, base, svreinterpret_u32_s32(signed_indices)));
    } else if constexpr (std::same_as<T, uint32_t>) {
      const auto active = svptrue_b32();
      auto base = svindex_u32(0, 1);
      base = svlsl_n_u32_z(
          active, svlsr_n_u32_z(active, base, shift), shift);
      return sve_table_lookup_word<Tag>(
          raw,
          svadd_u32_z(
              active, base, svreinterpret_u32_s32(signed_indices)));
    } else if constexpr (std::same_as<T, int64_t>) {
      const auto active = svptrue_b64();
      auto base = svindex_u64(0, 1);
      base = svlsl_n_u64_z(
          active, svlsr_n_u64_z(active, base, shift), shift);
      return sve_table_lookup_word<Tag>(
          raw,
          svadd_u64_z(
              active, base, svreinterpret_u64_s64(signed_indices)));
    } else if constexpr (std::same_as<T, uint64_t>) {
      const auto active = svptrue_b64();
      auto base = svindex_u64(0, 1);
      base = svlsl_n_u64_z(
          active, svlsr_n_u64_z(active, base, shift), shift);
      return sve_table_lookup_word<Tag>(
          raw,
          svadd_u64_z(
              active, base, svreinterpret_u64_s64(signed_indices)));
    } else {
      static_assert(
          dispatch_dependent_false<T>,
          "SVE local shuffle has no implementation for this element type");
    }
  }
};

} // namespace vecops::vec::details

#endif // VECOPS_VEC_DETAILS_SVE_BASIC_H
