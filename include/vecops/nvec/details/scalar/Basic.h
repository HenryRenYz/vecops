#ifndef VECOPS_NVEC_DETAILS_SCALAR_BASIC_H
#define VECOPS_NVEC_DETAILS_SCALAR_BASIC_H

#include <bit>
#include <bitset>

#include "vecops/nvec/details/Dispatch.h"

namespace vecops::nvec::details {

template <VectorTag Tag>
struct NativeImpl<ScalarBackend, FillOp, Tag> {
  static VECOPS_ALWAYS_INLINE Vec<Tag> call(
      FillOp, Tag, ElementOf<Tag> value) {
    using Traits = RepresentationTraits<ScalarBackend, Tag>;
    Vec<Tag> result{};
    if constexpr (Traits::word_count == 1) {
      for (nint_t lane = 0; lane < Traits::word_lanes; ++lane) {
        result[lane] = value;
      }
    } else {
      for (nint_t word = 0; word < Traits::word_count; ++word) {
        for (nint_t lane = 0; lane < Traits::word_lanes; ++lane) {
          result[word][lane] = value;
        }
      }
    }
    return result;
  }
};

template <VectorTag Tag>
struct NativeImpl<ScalarBackend, MaskFillOp, Tag> {
  static VECOPS_ALWAYS_INLINE Mask<Tag> call(
      MaskFillOp, Tag, bool value) {
    using Traits = RepresentationTraits<ScalarBackend, Tag>;
    Mask<Tag> result{};
    if constexpr (Traits::word_count == 1) {
      value ? result.bits.set() : result.bits.reset();
    } else {
      for (nint_t word = 0; word < Traits::word_count; ++word) {
        value ? result[word].bits.set() : result[word].bits.reset();
      }
    }
    return result;
  }
};

template <VectorTag Tag, bool Less>
VECOPS_ALWAYS_INLINE Mask<Tag> scalar_while_mask(
    nint_t a, nint_t b) {
  using Traits = RepresentationTraits<ScalarBackend, Tag>;
  Mask<Tag> result{};
  for (nint_t index = 0; index < Traits::logical_lanes; ++index) {
    const bool active = Less ? (a + index < b) : (a + index >= b);
    const nint_t word = index / Traits::word_lanes;
    const nint_t lane = index % Traits::word_lanes;
    if constexpr (Traits::word_count == 1) {
      result.bits.set(static_cast<std::size_t>(lane), active);
    } else {
      result[word].bits.set(static_cast<std::size_t>(lane), active);
    }
  }
  return result;
}

template <VectorTag Tag>
struct NativeImpl<ScalarBackend, MaskWhileLtOp, Tag> {
  static VECOPS_ALWAYS_INLINE Mask<Tag> call(
      MaskWhileLtOp, Tag, nint_t a, nint_t b) {
    return scalar_while_mask<Tag, true>(a, b);
  }
};

template <VectorTag Tag>
struct NativeImpl<ScalarBackend, MaskWhileGeOp, Tag> {
  static VECOPS_ALWAYS_INLINE Mask<Tag> call(
      MaskWhileGeOp, Tag, nint_t a, nint_t b) {
    return scalar_while_mask<Tag, false>(a, b);
  }
};

template <>
struct NativeWordImpl<ScalarBackend, BlendOp> {
  template <nint_t Index, VectorTag Tag>
  static VECOPS_ALWAYS_INLINE NativeWordVec<Tag> call(
      BlendOp, Tag,
      NativeWordVec<Tag> false_value,
      NativeWordMask<Tag> mask,
      NativeWordVec<Tag> true_value) {
    static_assert(Index >= 0 && Index < num_words(Tag{}));
    for (nint_t lane = 0; lane < native_word_size(Tag{}); ++lane) {
      if (mask.bits.test(static_cast<std::size_t>(lane))) {
        false_value[lane] = true_value[lane];
      }
    }
    return false_value;
  }
};

#define VECOPS_NVEC_DEFINE_SCALAR_MASK_BINARY(OpType, Expression)          \
  template <>                                                             \
  struct NativeWordImpl<ScalarBackend, OpType> {                          \
    template <nint_t Index, VectorTag Tag>                                \
    static VECOPS_ALWAYS_INLINE NativeWordMask<Tag> call(                 \
        OpType, Tag, NativeWordMask<Tag> a, NativeWordMask<Tag> b) {       \
      static_assert(Index >= 0 && Index < num_words(Tag{}));              \
      a.bits = (Expression);                                              \
      return a;                                                           \
    }                                                                     \
  }

VECOPS_NVEC_DEFINE_SCALAR_MASK_BINARY(MaskAndOp, a.bits & b.bits);
VECOPS_NVEC_DEFINE_SCALAR_MASK_BINARY(MaskOrOp, a.bits | b.bits);
VECOPS_NVEC_DEFINE_SCALAR_MASK_BINARY(MaskXorOp, a.bits ^ b.bits);
VECOPS_NVEC_DEFINE_SCALAR_MASK_BINARY(MaskAndNotOp, (~a.bits) & b.bits);

#undef VECOPS_NVEC_DEFINE_SCALAR_MASK_BINARY

template <>
struct NativeWordImpl<ScalarBackend, MaskNotOp> {
  template <nint_t Index, VectorTag Tag>
  static VECOPS_ALWAYS_INLINE NativeWordMask<Tag> call(
      MaskNotOp, Tag, NativeWordMask<Tag> value) {
    static_assert(Index >= 0 && Index < num_words(Tag{}));
    value.bits.flip();
    return value;
  }
};

template <>
struct NativeWordImpl<ScalarBackend, GetVecLaneOp> {
  template <nint_t Index, VectorTag Tag>
  static VECOPS_ALWAYS_INLINE ElementOf<Tag> call(
      GetVecLaneOp, Tag, NativeWordVec<Tag> value, nint_t lane) {
    static_assert(Index >= 0 && Index < num_words(Tag{}));
    return value[lane];
  }
};

template <>
struct NativeWordImpl<ScalarBackend, SetVecLaneOp> {
  template <nint_t Index, VectorTag Tag>
  static VECOPS_ALWAYS_INLINE NativeWordVec<Tag> call(
      SetVecLaneOp, Tag, NativeWordVec<Tag> value,
      nint_t lane, ElementOf<Tag> replacement) {
    static_assert(Index >= 0 && Index < num_words(Tag{}));
    value[lane] = replacement;
    return value;
  }
};

template <>
struct NativeWordImpl<ScalarBackend, GetMaskLaneOp> {
  template <nint_t Index, VectorTag Tag>
  static VECOPS_ALWAYS_INLINE bool call(
      GetMaskLaneOp, Tag, NativeWordMask<Tag> value, nint_t lane) {
    static_assert(Index >= 0 && Index < num_words(Tag{}));
    return value.bits.test(static_cast<std::size_t>(lane));
  }
};

template <>
struct NativeWordImpl<ScalarBackend, SetMaskLaneOp> {
  template <nint_t Index, VectorTag Tag>
  static VECOPS_ALWAYS_INLINE NativeWordMask<Tag> call(
      SetMaskLaneOp, Tag, NativeWordMask<Tag> value,
      nint_t lane, bool replacement) {
    static_assert(Index >= 0 && Index < num_words(Tag{}));
    value.bits.set(static_cast<std::size_t>(lane), replacement);
    return value;
  }
};

template <>
struct NativeWordImpl<ScalarBackend, BitCastOp> {
  template <nint_t Index, VectorTag ToTag, VectorTag FromTag>
  static VECOPS_ALWAYS_INLINE NativeWordVec<ToTag> call(
      BitCastOp, ToTag, FromTag, NativeWordVec<FromTag> value) {
    static_assert(Index >= 0 && Index < num_words(ToTag{}));
    static_assert(Index < num_words(FromTag{}));
    static_assert(
        sizeof(NativeWordVec<ToTag>) == sizeof(NativeWordVec<FromTag>));
    return ::vecops::bitcast<NativeWordVec<ToTag>>(value);
  }
};

template <VectorTag Tag>
struct NativeImpl<ScalarBackend, LowerOp, Tag> {
  static VECOPS_ALWAYS_INLINE Vec<Half<Tag>> call(
      LowerOp, Tag tag, Vec<Tag> value) {
    using OutTag = Half<Tag>;
    Vec<OutTag> result{};
    for (nint_t i = 0; i < size(OutTag{}); ++i) {
      result = ::vecops::nvec::set(
          OutTag{}, result, i, ::vecops::nvec::get(tag, value, i));
    }
    return result;
  }

  static VECOPS_ALWAYS_INLINE Mask<Half<Tag>> call(
      LowerOp, Tag tag, Mask<Tag> value) {
    using OutTag = Half<Tag>;
    Mask<OutTag> result{};
    for (nint_t i = 0; i < size(OutTag{}); ++i) {
      result = ::vecops::nvec::set(
          OutTag{}, result, i, ::vecops::nvec::get(tag, value, i));
    }
    return result;
  }
};

template <VectorTag Tag>
struct NativeImpl<ScalarBackend, UpperOp, Tag> {
  static VECOPS_ALWAYS_INLINE Vec<Half<Tag>> call(
      UpperOp, Tag tag, Vec<Tag> value) {
    using OutTag = Half<Tag>;
    Vec<OutTag> result{};
    for (nint_t i = 0; i < size(OutTag{}); ++i) {
      result = ::vecops::nvec::set(
          OutTag{}, result, i,
          ::vecops::nvec::get(tag, value, i + size(OutTag{})));
    }
    return result;
  }

  static VECOPS_ALWAYS_INLINE Mask<Half<Tag>> call(
      UpperOp, Tag tag, Mask<Tag> value) {
    using OutTag = Half<Tag>;
    Mask<OutTag> result{};
    for (nint_t i = 0; i < size(OutTag{}); ++i) {
      result = ::vecops::nvec::set(
          OutTag{}, result, i,
          ::vecops::nvec::get(tag, value, i + size(OutTag{})));
    }
    return result;
  }
};

template <VectorTag Tag>
struct NativeImpl<ScalarBackend, ConcatOp, Tag> {
  static VECOPS_ALWAYS_INLINE Vec<Tag> call(
      ConcatOp, Tag tag, Vec<Half<Tag>> lower_value,
      Vec<Half<Tag>> upper_value) {
    using InTag = Half<Tag>;
    Vec<Tag> result{};
    for (nint_t i = 0; i < size(InTag{}); ++i) {
      result = ::vecops::nvec::set(
          tag, result, i,
          ::vecops::nvec::get(InTag{}, lower_value, i));
      result = ::vecops::nvec::set(
          tag, result, i + size(InTag{}),
          ::vecops::nvec::get(InTag{}, upper_value, i));
    }
    return result;
  }

  static VECOPS_ALWAYS_INLINE Mask<Tag> call(
      ConcatOp, Tag tag, Mask<Half<Tag>> lower_value,
      Mask<Half<Tag>> upper_value) {
    using InTag = Half<Tag>;
    Mask<Tag> result{};
    for (nint_t i = 0; i < size(InTag{}); ++i) {
      result = ::vecops::nvec::set(
          tag, result, i,
          ::vecops::nvec::get(InTag{}, lower_value, i));
      result = ::vecops::nvec::set(
          tag, result, i + size(InTag{}),
          ::vecops::nvec::get(InTag{}, upper_value, i));
    }
    return result;
  }
};

#define VECOPS_NVEC_DEFINE_SCALAR_EVEN_ODD(OpType, Offset)                \
  template <VectorTag Tag>                                               \
  struct NativeImpl<ScalarBackend, OpType, Tag> {                        \
    static VECOPS_ALWAYS_INLINE Vec<Half<Tag>> call(                     \
        OpType, Tag tag, Vec<Tag> value) {                               \
      using OutTag = Half<Tag>;                                          \
      Vec<OutTag> result{};                                              \
      for (nint_t i = 0; i < size(OutTag{}); ++i) {                      \
        result = ::vecops::nvec::set(                                    \
            OutTag{}, result, i,                                        \
            ::vecops::nvec::get(tag, value, 2 * i + Offset));           \
      }                                                                  \
      return result;                                                     \
    }                                                                    \
  }

VECOPS_NVEC_DEFINE_SCALAR_EVEN_ODD(EvenOp, 0);
VECOPS_NVEC_DEFINE_SCALAR_EVEN_ODD(OddOp, 1);

#undef VECOPS_NVEC_DEFINE_SCALAR_EVEN_ODD

#define VECOPS_NVEC_DEFINE_SCALAR_CONCAT_SELECT(OpType, Offset)           \
  template <VectorTag Tag>                                               \
  struct NativeImpl<ScalarBackend, OpType, Tag> {                        \
    static VECOPS_ALWAYS_INLINE Vec<Tag> call(                           \
        OpType, Tag tag, Vec<Tag> a, Vec<Tag> b) {                       \
      Vec<Tag> result{};                                                 \
      const nint_t half = size(tag) / 2;                                \
      for (nint_t i = 0; i < half; ++i) {                               \
        result = ::vecops::nvec::set(                                    \
            tag, result, i,                                             \
            ::vecops::nvec::get(tag, a, 2 * i + Offset));               \
        result = ::vecops::nvec::set(                                    \
            tag, result, i + half,                                      \
            ::vecops::nvec::get(tag, b, 2 * i + Offset));               \
      }                                                                  \
      return result;                                                     \
    }                                                                    \
  }

VECOPS_NVEC_DEFINE_SCALAR_CONCAT_SELECT(ConcatEvenOp, 0);
VECOPS_NVEC_DEFINE_SCALAR_CONCAT_SELECT(ConcatOddOp, 1);

#undef VECOPS_NVEC_DEFINE_SCALAR_CONCAT_SELECT

template <VectorTag Tag>
struct NativeImpl<ScalarBackend, InterleaveOp, Tag> {
  static VECOPS_ALWAYS_INLINE Vec<Tag> call(
      InterleaveOp, Tag tag, Vec<Half<Tag>> a, Vec<Half<Tag>> b) {
    using InTag = Half<Tag>;
    Vec<Tag> result{};
    for (nint_t i = 0; i < size(InTag{}); ++i) {
      result = ::vecops::nvec::set(
          tag, result, 2 * i,
          ::vecops::nvec::get(InTag{}, a, i));
      result = ::vecops::nvec::set(
          tag, result, 2 * i + 1,
          ::vecops::nvec::get(InTag{}, b, i));
    }
    return result;
  }
};

#define VECOPS_NVEC_DEFINE_SCALAR_INTERLEAVE_PAIR(OpType, Offset)         \
  template <VectorTag Tag>                                               \
  struct NativeImpl<ScalarBackend, OpType, Tag> {                        \
    static VECOPS_ALWAYS_INLINE Vec<Tag> call(                           \
        OpType, Tag tag, Vec<Tag> a, Vec<Tag> b) {                       \
      Vec<Tag> result{};                                                 \
      for (nint_t i = 0; i < size(tag) / 2; ++i) {                      \
        result = ::vecops::nvec::set(                                    \
            tag, result, 2 * i,                                         \
            ::vecops::nvec::get(tag, a, 2 * i + Offset));               \
        result = ::vecops::nvec::set(                                    \
            tag, result, 2 * i + 1,                                     \
            ::vecops::nvec::get(tag, b, 2 * i + Offset));               \
      }                                                                  \
      return result;                                                     \
    }                                                                    \
  }

VECOPS_NVEC_DEFINE_SCALAR_INTERLEAVE_PAIR(InterleaveEvenOp, 0);
VECOPS_NVEC_DEFINE_SCALAR_INTERLEAVE_PAIR(InterleaveOddOp, 1);

#undef VECOPS_NVEC_DEFINE_SCALAR_INTERLEAVE_PAIR

#define VECOPS_NVEC_DEFINE_SCALAR_LOCAL_INTERLEAVE(OpType, Upper)         \
  template <VectorTag Tag>                                               \
  struct NativeImpl<ScalarBackend, OpType, Tag> {                        \
    static VECOPS_ALWAYS_INLINE Vec<Tag> call(                           \
        OpType, Tag tag, Vec<Tag> a, Vec<Tag> b) {                       \
      Vec<Tag> result{};                                                 \
      constexpr nint_t block_lanes =                                    \
          16 / static_cast<nint_t>(sizeof(ElementOf<Tag>));              \
      const nint_t source_half = Upper ? block_lanes / 2 : 0;           \
      for (nint_t i = 0; i < size(tag); ++i) {                          \
        const nint_t block = (i / block_lanes) * block_lanes;           \
        const nint_t within = i % block_lanes;                          \
        const nint_t source = block + source_half + within / 2;         \
        result = ::vecops::nvec::set(                                    \
            tag, result, i, (within & 1)                                \
                ? ::vecops::nvec::get(tag, b, source)                   \
                : ::vecops::nvec::get(tag, a, source));                 \
      }                                                                  \
      return result;                                                     \
    }                                                                    \
  }

VECOPS_NVEC_DEFINE_SCALAR_LOCAL_INTERLEAVE(
    LocalInterleaveLowerOp, false);
VECOPS_NVEC_DEFINE_SCALAR_LOCAL_INTERLEAVE(
    LocalInterleaveUpperOp, true);

#undef VECOPS_NVEC_DEFINE_SCALAR_LOCAL_INTERLEAVE

#define VECOPS_NVEC_DEFINE_SCALAR_SHUFFLE(OpType, Local)                 \
  template <>                                                            \
  struct NativeWordImpl<ScalarBackend, OpType> {                         \
    template <nint_t Index, VectorTag Tag>                               \
    static VECOPS_ALWAYS_INLINE NativeWordVec<Tag> call(                 \
        OpType, Tag tag, NativeWordVec<Tag> value,                       \
        NativeWordVec<IndexTag<Tag>> indices) {                          \
      using Traits = RepresentationTraits<ScalarBackend, Tag>;          \
      static_assert(Index >= 0 && Index < Traits::word_count);          \
      NativeWordVec<Tag> result = value;                                 \
      const nint_t remaining =                                          \
          size(tag) - Index * Traits::word_lanes;                        \
      const nint_t valid = remaining < Traits::word_lanes               \
          ? remaining                                                    \
          : Traits::word_lanes;                                         \
      constexpr nint_t block_lanes =                                    \
          16 / static_cast<nint_t>(sizeof(ElementOf<Tag>));              \
      for (nint_t lane = 0; lane < valid; ++lane) {                     \
        const nint_t selected = static_cast<nint_t>(indices[lane]);     \
        const nint_t limit = Local ? block_lanes : Traits::word_lanes;  \
        assert(selected >= 0 && selected < limit);                      \
        const nint_t source = Local                                     \
            ? (lane / block_lanes) * block_lanes + selected             \
            : selected;                                                  \
        result[lane] = value[source];                                    \
      }                                                                  \
      return result;                                                     \
    }                                                                    \
  }

VECOPS_NVEC_DEFINE_SCALAR_SHUFFLE(ShuffleOp, false);
VECOPS_NVEC_DEFINE_SCALAR_SHUFFLE(LocalShuffleOp, true);

#undef VECOPS_NVEC_DEFINE_SCALAR_SHUFFLE

} // namespace vecops::nvec::details

#endif // VECOPS_NVEC_DETAILS_SCALAR_BASIC_H
