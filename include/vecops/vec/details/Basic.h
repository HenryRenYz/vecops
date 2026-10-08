// SPDX-FileCopyrightText: 2026 HenryRenYz and vecops contributors
// SPDX-License-Identifier: MIT
#ifndef VECOPS_VEC_DETAILS_BASIC_H
#define VECOPS_VEC_DETAILS_BASIC_H

#include <cstring>
#include <type_traits>
#include <utility>

#include "vecops/vec/details/Elementwise.h"
#include "vecops/vec/Options.h"

namespace vecops::vec::details {

/**
 * Broadcasts one scalar into a physical word. The word twin of the public
 * fill: fill's (Tag, scalar) parameter shape cannot distinguish the two
 * representations by overload resolution, so the word form lives under its
 * own name.
 */
template <VectorTag Tag>
VECOPS_ALWAYS_INLINE NativeWordVec<Tag> fill_word(
    Tag tag, ElementOf<Tag> value) {
  return execute_word<0, CurrentBackend>(FillOp{}, tag, value);
}

/**
 * Shared sanitize-compute-blend skeleton for masked unary math word
 * implementations: inactive-lane inputs are replaced by zero before the
 * computation, then the caller's inactive value is blended back in. compute
 * receives the sanitized input word and returns the unmasked result word.
 */
template <VectorTag Tag, typename Compute>
VECOPS_ALWAYS_INLINE NativeWordVec<Tag> masked_unary_word(
    Tag tag, NativeWordMask<Tag> mask, NativeWordVec<Tag> inactive,
    NativeWordVec<Tag> value, Compute&& compute) {
  const auto safe = BlendOp{}(
      tag, fill_word(tag, ElementOf<Tag>{}), mask, value);
  return BlendOp{}(tag, inactive, mask, compute(safe));
}

template <typename Backend, VectorTag ToTag>
struct GenericImpl<Backend, ResizeBitCastOp, ToTag> {
  template <VectorTag FromTag>
    requires (is_fixed_tag_v<ToTag> == is_fixed_tag_v<FromTag>)
  static VECOPS_ALWAYS_INLINE Vec<ToTag> call(
      ResizeBitCastOp, ToTag to, FromTag from, Vec<FromTag> value) {
    using ToTraits = RepresentationTraits<Backend, ToTag>;
    using FromTraits = RepresentationTraits<Backend, FromTag>;
    if constexpr (std::same_as<Backend, SVEBackend>) {
      return construct_words<Backend>(
          to, [&]<nint_t Index>(ToTag parent) VECOPS_INLINE_LAMBDA {
        if constexpr (Index < FromTraits::word_count) {
          return execute_word<Index, Backend>(
              BitCastOp{}, parent, from,
              ::vecops::vec::get_word<Index>(from, value));
        } else {
          return execute_word<Index, Backend>(
              FillOp{}, parent, ElementOf<ToTag>{});
        }
      });
    } else {
      Vec<ToTag> result{};
      constexpr std::size_t copy_bytes =
          sizeof(result) < sizeof(value) ? sizeof(result) : sizeof(value);
      std::memcpy(&result, &value, copy_bytes);
      return result;
    }
  }
};

template <>
struct EnableElementwiseWordBatching<BlendOp> : std::true_type {};

template <>
struct EnableMaskElementwiseWordBatching<MaskAndOp> : std::true_type {};
template <>
struct EnableMaskElementwiseWordBatching<MaskOrOp> : std::true_type {};
template <>
struct EnableMaskElementwiseWordBatching<MaskXorOp> : std::true_type {};
template <>
struct EnableMaskElementwiseWordBatching<MaskAndNotOp> : std::true_type {};
template <>
struct EnableMaskElementwiseWordBatching<MaskNotOp> : std::true_type {};

template <>
struct EnableElementwiseWordBatching<InterleaveEvenOp> : std::true_type {};
template <>
struct EnableElementwiseWordBatching<InterleaveOddOp> : std::true_type {};
template <>
struct EnableElementwiseWordBatching<LocalInterleaveLowerOp>
    : std::true_type {};
template <>
struct EnableElementwiseWordBatching<LocalInterleaveUpperOp>
    : std::true_type {};

template <typename Backend, VectorTag Tag>
  requires (RepresentationTraits<Backend, Tag>::word_count > 1)
struct GenericImpl<Backend, FillOp, Tag> {
  static VECOPS_ALWAYS_INLINE Vec<Tag> call(
      FillOp op, Tag tag, ElementOf<Tag> value) {
    return construct_words<Backend>(
        tag,
        [&]<nint_t Index>(Tag parent) VECOPS_INLINE_LAMBDA {
          return execute_word<Index, Backend>(op, parent, value);
        });
  }
};

template <typename Backend, VectorTag Tag>
  requires (RepresentationTraits<Backend, Tag>::word_count > 1)
struct GenericImpl<Backend, MaskFillOp, Tag> {
  static VECOPS_ALWAYS_INLINE Mask<Tag> call(
      MaskFillOp op, Tag tag, bool value) {
    return construct_mask_words<Backend>(
        tag,
        [&]<nint_t Index>(Tag parent) VECOPS_INLINE_LAMBDA {
          return execute_word<Index, Backend>(op, parent, value);
        });
  }
};

#define VECOPS_VEC_DEFINE_GENERIC_WHILE(OpType)                           \
  template <typename Backend, VectorTag Tag>                              \
    requires (RepresentationTraits<Backend, Tag>::word_count > 1)        \
  struct GenericImpl<Backend, OpType, Tag> {                              \
    static VECOPS_ALWAYS_INLINE Mask<Tag> call(                           \
        OpType op, Tag tag, nint_t a, nint_t b) {                         \
      return construct_mask_words<Backend>(                              \
          tag,                                                           \
          [&]<nint_t Index>(Tag parent) VECOPS_INLINE_LAMBDA {           \
            return execute_word<Index, Backend>(op, parent, a, b);       \
          });                                                            \
    }                                                                    \
  }

VECOPS_VEC_DEFINE_GENERIC_WHILE(MaskWhileLtOp);
VECOPS_VEC_DEFINE_GENERIC_WHILE(MaskWhileGeOp);

#undef VECOPS_VEC_DEFINE_GENERIC_WHILE

/* **************************************************************************** */
//    Whole-Tag mask-query folds                                               //
/* **************************************************************************** */

template <typename Backend, nint_t Index, nint_t Count, VectorTag Tag>
VECOPS_ALWAYS_INLINE nint_t mask_first_words(Tag tag, Mask<Tag> value) {
  const nint_t local = execute_word<Index, Backend>(
      MaskFirstOp{}, tag, ::vecops::vec::get_word<Index>(tag, value));
  if (local >= 0) return Index * native_word_size(tag) + local;
  if constexpr (Index + 1 < Count)
    return mask_first_words<Backend, Index + 1, Count>(tag, value);
  else
    return -1;
}

template <typename Backend, nint_t Index, VectorTag Tag>
VECOPS_ALWAYS_INLINE nint_t mask_last_words(Tag tag, Mask<Tag> value) {
  const nint_t local = execute_word<Index, Backend>(
      MaskLastOp{}, tag, ::vecops::vec::get_word<Index>(tag, value));
  if (local >= 0) return Index * native_word_size(tag) + local;
  if constexpr (Index > 0)
    return mask_last_words<Backend, Index - 1>(tag, value);
  else
    return -1;
}

template <typename Backend, VectorTag Tag>
  requires (RepresentationTraits<Backend, Tag>::word_count > 1)
struct GenericImpl<Backend, MaskAllOp, Tag> {
  static VECOPS_ALWAYS_INLINE bool call(
      MaskAllOp op, Tag tag, Mask<Tag> value) {
    return [&]<std::size_t... Index>(std::index_sequence<Index...>) {
      return (execute_word<static_cast<nint_t>(Index), Backend>(
                  op, tag,
                  ::vecops::vec::get_word<static_cast<nint_t>(Index)>(
                      tag, value)) && ...);
    }(std::make_index_sequence<static_cast<std::size_t>(
        RepresentationTraits<Backend, Tag>::word_count)>{});
  }
};

template <typename Backend, VectorTag Tag>
  requires (RepresentationTraits<Backend, Tag>::word_count > 1)
struct GenericImpl<Backend, MaskAnyOp, Tag> {
  static VECOPS_ALWAYS_INLINE bool call(
      MaskAnyOp op, Tag tag, Mask<Tag> value) {
    return [&]<std::size_t... Index>(std::index_sequence<Index...>) {
      return (execute_word<static_cast<nint_t>(Index), Backend>(
                  op, tag,
                  ::vecops::vec::get_word<static_cast<nint_t>(Index)>(
                      tag, value)) || ...);
    }(std::make_index_sequence<static_cast<std::size_t>(
        RepresentationTraits<Backend, Tag>::word_count)>{});
  }
};

template <typename Backend, VectorTag Tag>
  requires (RepresentationTraits<Backend, Tag>::word_count > 1)
struct GenericImpl<Backend, MaskCountOp, Tag> {
  static VECOPS_ALWAYS_INLINE nint_t call(
      MaskCountOp op, Tag tag, Mask<Tag> value) {
    return [&]<std::size_t... Index>(std::index_sequence<Index...>) {
      return (nint_t{0} + ... + execute_word<
          static_cast<nint_t>(Index), Backend>(
              op, tag,
              ::vecops::vec::get_word<static_cast<nint_t>(Index)>(
                  tag, value)));
    }(std::make_index_sequence<static_cast<std::size_t>(
        RepresentationTraits<Backend, Tag>::word_count)>{});
  }
};

template <typename Backend, VectorTag Tag>
  requires (RepresentationTraits<Backend, Tag>::word_count > 1)
struct GenericImpl<Backend, MaskFirstOp, Tag> {
  static VECOPS_ALWAYS_INLINE nint_t call(
      MaskFirstOp, Tag tag, Mask<Tag> value) {
    return mask_first_words<
        Backend, 0, RepresentationTraits<Backend, Tag>::word_count>(
            tag, value);
  }
};

template <typename Backend, VectorTag Tag>
  requires (RepresentationTraits<Backend, Tag>::word_count > 1)
struct GenericImpl<Backend, MaskLastOp, Tag> {
  static VECOPS_ALWAYS_INLINE nint_t call(
      MaskLastOp, Tag tag, Mask<Tag> value) {
    return mask_last_words<
        Backend, RepresentationTraits<Backend, Tag>::word_count - 1>(
            tag, value);
  }
};

template <typename Backend, VectorTag ToTag>
  requires (RepresentationTraits<Backend, ToTag>::word_count > 1)
struct GenericImpl<Backend, BitCastOp, ToTag> {
  template <VectorTag FromTag>
    requires same_logical_bytes_v<ToTag, FromTag> &&
             (RepresentationTraits<Backend, ToTag>::word_count ==
              RepresentationTraits<Backend, FromTag>::word_count)
  static VECOPS_ALWAYS_INLINE Vec<ToTag> call(
      BitCastOp op, ToTag to, FromTag from, Vec<FromTag> value) {
    return construct_words<Backend>(
        to,
        [&]<nint_t Index>(ToTag parent) VECOPS_INLINE_LAMBDA {
          return execute_word<Index, Backend>(
              op, parent, from,
              ::vecops::vec::get_word<Index>(from, value));
        });
  }
};

template <typename Backend, VectorTag Tag>
  requires (RepresentationTraits<Backend, Tag>::word_count > 1) &&
           (RepresentationTraits<Backend, Tag>::word_count ==
            2 * RepresentationTraits<Backend, Half<Tag>>::word_count)
struct GenericImpl<Backend, LowerOp, Tag> {
  static VECOPS_ALWAYS_INLINE Vec<Half<Tag>> call(
      LowerOp, Tag tag, Vec<Tag> value) {
    using OutTag = Half<Tag>;
    return construct_words<Backend>(
        OutTag{},
        [&]<nint_t Index>(OutTag) VECOPS_INLINE_LAMBDA {
          return ::vecops::vec::get_word<Index>(tag, value);
        });
  }

  static VECOPS_ALWAYS_INLINE Mask<Half<Tag>> call(
      LowerOp, Tag tag, Mask<Tag> value) {
    using OutTag = Half<Tag>;
    return construct_mask_words<Backend>(
        OutTag{},
        [&]<nint_t Index>(OutTag) VECOPS_INLINE_LAMBDA {
          return ::vecops::vec::get_word<Index>(tag, value);
        });
  }
};

template <typename Backend, VectorTag Tag>
  requires (RepresentationTraits<Backend, Tag>::word_count > 1) &&
           (RepresentationTraits<Backend, Tag>::word_count ==
            2 * RepresentationTraits<Backend, Half<Tag>>::word_count)
struct GenericImpl<Backend, UpperOp, Tag> {
  static VECOPS_ALWAYS_INLINE Vec<Half<Tag>> call(
      UpperOp, Tag tag, Vec<Tag> value) {
    using OutTag = Half<Tag>;
    constexpr nint_t offset =
        RepresentationTraits<Backend, OutTag>::word_count;
    return construct_words<Backend>(
        OutTag{},
        [&]<nint_t Index>(OutTag) VECOPS_INLINE_LAMBDA {
          return ::vecops::vec::get_word<Index + offset>(tag, value);
        });
  }

  static VECOPS_ALWAYS_INLINE Mask<Half<Tag>> call(
      UpperOp, Tag tag, Mask<Tag> value) {
    using OutTag = Half<Tag>;
    constexpr nint_t offset =
        RepresentationTraits<Backend, OutTag>::word_count;
    return construct_mask_words<Backend>(
        OutTag{},
        [&]<nint_t Index>(OutTag) VECOPS_INLINE_LAMBDA {
          return ::vecops::vec::get_word<Index + offset>(tag, value);
        });
  }
};

template <typename Backend, VectorTag Tag>
  requires (RepresentationTraits<Backend, Tag>::word_count > 1) &&
           (RepresentationTraits<Backend, Tag>::word_count ==
            2 * RepresentationTraits<Backend, Half<Tag>>::word_count)
struct GenericImpl<Backend, ConcatOp, Tag> {
  static VECOPS_ALWAYS_INLINE Vec<Tag> call(
      ConcatOp, Tag tag, Vec<Half<Tag>> lower_value,
      Vec<Half<Tag>> upper_value) {
    using InTag = Half<Tag>;
    constexpr nint_t split =
        RepresentationTraits<Backend, InTag>::word_count;
    return construct_words<Backend>(
        tag,
        [&]<nint_t Index>(Tag) VECOPS_INLINE_LAMBDA {
          if constexpr (Index < split) {
            return ::vecops::vec::get_word<Index>(InTag{}, lower_value);
          } else {
            return ::vecops::vec::get_word<Index - split>(
                InTag{}, upper_value);
          }
        });
  }

  static VECOPS_ALWAYS_INLINE Mask<Tag> call(
      ConcatOp, Tag tag, Mask<Half<Tag>> lower_value,
      Mask<Half<Tag>> upper_value) {
    using InTag = Half<Tag>;
    constexpr nint_t split =
        RepresentationTraits<Backend, InTag>::word_count;
    return construct_mask_words<Backend>(
        tag,
        [&]<nint_t Index>(Tag) VECOPS_INLINE_LAMBDA {
          if constexpr (Index < split) {
            return ::vecops::vec::get_word<Index>(InTag{}, lower_value);
          } else {
            return ::vecops::vec::get_word<Index - split>(
                InTag{}, upper_value);
          }
        });
  }
};

#define VECOPS_VEC_DEFINE_RECURSIVE_SELECT(OpType)                        \
  template <typename Backend, VectorTag Tag>                              \
    requires (RepresentationTraits<Backend, Tag>::word_count > 1) &&     \
             (RepresentationTraits<Backend, Half<Tag>>::word_count > 1)  \
  struct GenericImpl<Backend, OpType, Tag> {                              \
    static VECOPS_ALWAYS_INLINE Vec<Half<Tag>> call(                     \
        OpType op, Tag tag, Vec<Tag> value) {                             \
      using Child = Half<Tag>;                                           \
      const auto lo = lower(tag, value);                    \
      const auto hi = upper(tag, value);                    \
      return execute(                                                    \
          ConcatOp{}, Child{},                                           \
          execute(op, Child{}, lo), execute(op, Child{}, hi));           \
    }                                                                    \
  }

VECOPS_VEC_DEFINE_RECURSIVE_SELECT(EvenOp);
VECOPS_VEC_DEFINE_RECURSIVE_SELECT(OddOp);

#undef VECOPS_VEC_DEFINE_RECURSIVE_SELECT

template <typename Backend, VectorTag Tag>
struct GenericImpl<Backend, ConcatEvenOp, Tag> {
  static VECOPS_ALWAYS_INLINE Vec<Tag> call(
      ConcatEvenOp, Tag tag, Vec<Tag> a, Vec<Tag> b) {
    return execute(
        ConcatOp{}, tag,
        even(tag, a), even(tag, b));
  }
};

template <typename Backend, VectorTag Tag>
struct GenericImpl<Backend, ConcatOddOp, Tag> {
  static VECOPS_ALWAYS_INLINE Vec<Tag> call(
      ConcatOddOp, Tag tag, Vec<Tag> a, Vec<Tag> b) {
    return execute(
        ConcatOp{}, tag,
        odd(tag, a), odd(tag, b));
  }
};

template <typename Backend, VectorTag Tag>
  requires (RepresentationTraits<Backend, Tag>::word_count > 1)
struct GenericImpl<Backend, InterleaveOp, Tag> {
  static VECOPS_ALWAYS_INLINE Vec<Tag> call(
      InterleaveOp op, Tag tag, Vec<Half<Tag>> a, Vec<Half<Tag>> b) {
    using Child = Half<Tag>;
    const auto lo = execute(
        op, Child{},
        lower(Child{}, a),
        lower(Child{}, b));
    const auto hi = execute(
        op, Child{},
        upper(Child{}, a),
        upper(Child{}, b));
    return concat(tag, lo, hi);
  }
};

#define VECOPS_VEC_DEFINE_GENERIC_SHUFFLE(OpType)                        \
  template <typename Backend, VectorTag Tag>                             \
    requires (RepresentationTraits<Backend, Tag>::word_count > 1)       \
  struct GenericImpl<Backend, OpType, Tag> {                             \
    static VECOPS_ALWAYS_INLINE Vec<Tag> call(                           \
        OpType op, Tag tag, Vec<Tag> value,                             \
        Vec<IndexTag<Tag>> indices) {                                    \
      return construct_words<Backend>(                                  \
          tag,                                                          \
          [&]<nint_t Index>(Tag parent) VECOPS_INLINE_LAMBDA {          \
            return execute_word<Index, Backend>(                        \
                op, parent, ::vecops::vec::get_word<Index>(tag, value), \
                ::vecops::vec::get_word<Index>(                        \
                    IndexTag<Tag>{}, indices));                          \
          });                                                           \
    }                                                                    \
  }

VECOPS_VEC_DEFINE_GENERIC_SHUFFLE(ShuffleOp);
VECOPS_VEC_DEFINE_GENERIC_SHUFFLE(LocalShuffleOp);

#undef VECOPS_VEC_DEFINE_GENERIC_SHUFFLE

template <VectorTag Tag, typename Option>
inline constexpr bool is_fill_option_for_v = [] {
  using Clean = std::remove_cvref_t<Option>;
  if constexpr (is_unmasked_option_v<Clean>) {
    return true;
  } else if constexpr (is_masked_option_v<Clean>) {
    return std::same_as<typename IsMaskedOption<Clean>::Value, Mask<Tag>>;
  } else if constexpr (is_first_option_v<Clean>) {
    return true;
  } else if constexpr (is_vector_merge_option_v<Clean>) {
    return std::same_as<
        typename IsVectorMergeOption<Clean>::Value, Vec<Tag>>;
  } else if constexpr (is_scalar_merge_option_v<Clean>) {
    return std::same_as<
        typename IsScalarMergeOption<Clean>::Value, ElementOf<Tag>>;
  } else {
    return false;
  }
}();

} // namespace vecops::vec::details

#endif // VECOPS_VEC_DETAILS_BASIC_H
