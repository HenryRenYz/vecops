#ifndef VECOPS_NVEC_DETAILS_BASIC_H
#define VECOPS_NVEC_DETAILS_BASIC_H

#include <type_traits>
#include <utility>

#include "vecops/nvec/details/Elementwise.h"

namespace vecops::nvec::details {

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
        [&]<nint_t Index>(Tag parent) {
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
        [&]<nint_t Index>(Tag parent) {
          return execute_word<Index, Backend>(op, parent, value);
        });
  }
};

#define VECOPS_NVEC_DEFINE_GENERIC_WHILE(OpType)                           \
  template <typename Backend, VectorTag Tag>                              \
    requires (RepresentationTraits<Backend, Tag>::word_count > 1)        \
  struct GenericImpl<Backend, OpType, Tag> {                              \
    static VECOPS_ALWAYS_INLINE Mask<Tag> call(                           \
        OpType op, Tag tag, nint_t a, nint_t b) {                         \
      return construct_mask_words<Backend>(                              \
          tag,                                                           \
          [&]<nint_t Index>(Tag parent) {                                \
            return execute_word<Index, Backend>(op, parent, a, b);       \
          });                                                            \
    }                                                                    \
  }

VECOPS_NVEC_DEFINE_GENERIC_WHILE(MaskWhileLtOp);
VECOPS_NVEC_DEFINE_GENERIC_WHILE(MaskWhileGeOp);

#undef VECOPS_NVEC_DEFINE_GENERIC_WHILE

template <typename Backend, VectorTag ToTag>
  requires (RepresentationTraits<Backend, ToTag>::word_count > 1)
struct GenericImpl<Backend, BitCastOp, ToTag> {
  template <VectorTag FromTag>
    requires same_logical_bytes<ToTag, FromTag> &&
             (RepresentationTraits<Backend, ToTag>::word_count ==
              RepresentationTraits<Backend, FromTag>::word_count)
  static VECOPS_ALWAYS_INLINE Vec<ToTag> call(
      BitCastOp op, ToTag to, FromTag from, Vec<FromTag> value) {
    return construct_words<Backend>(
        to,
        [&]<nint_t Index>(ToTag parent) {
          return execute_word<Index, Backend>(
              op, parent, from,
              ::vecops::nvec::get_word<Index>(from, value));
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
        [&]<nint_t Index>(OutTag) {
          return ::vecops::nvec::get_word<Index>(tag, value);
        });
  }

  static VECOPS_ALWAYS_INLINE Mask<Half<Tag>> call(
      LowerOp, Tag tag, Mask<Tag> value) {
    using OutTag = Half<Tag>;
    return construct_mask_words<Backend>(
        OutTag{},
        [&]<nint_t Index>(OutTag) {
          return ::vecops::nvec::get_word<Index>(tag, value);
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
        [&]<nint_t Index>(OutTag) {
          return ::vecops::nvec::get_word<Index + offset>(tag, value);
        });
  }

  static VECOPS_ALWAYS_INLINE Mask<Half<Tag>> call(
      UpperOp, Tag tag, Mask<Tag> value) {
    using OutTag = Half<Tag>;
    constexpr nint_t offset =
        RepresentationTraits<Backend, OutTag>::word_count;
    return construct_mask_words<Backend>(
        OutTag{},
        [&]<nint_t Index>(OutTag) {
          return ::vecops::nvec::get_word<Index + offset>(tag, value);
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
        [&]<nint_t Index>(Tag) {
          if constexpr (Index < split) {
            return ::vecops::nvec::get_word<Index>(InTag{}, lower_value);
          } else {
            return ::vecops::nvec::get_word<Index - split>(
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
        [&]<nint_t Index>(Tag) {
          if constexpr (Index < split) {
            return ::vecops::nvec::get_word<Index>(InTag{}, lower_value);
          } else {
            return ::vecops::nvec::get_word<Index - split>(
                InTag{}, upper_value);
          }
        });
  }
};

#define VECOPS_NVEC_DEFINE_RECURSIVE_SELECT(OpType)                        \
  template <typename Backend, VectorTag Tag>                              \
    requires (RepresentationTraits<Backend, Tag>::word_count > 1) &&     \
             (RepresentationTraits<Backend, Half<Tag>>::word_count > 1)  \
  struct GenericImpl<Backend, OpType, Tag> {                              \
    static VECOPS_ALWAYS_INLINE Vec<Half<Tag>> call(                     \
        OpType op, Tag tag, Vec<Tag> value) {                             \
      using Child = Half<Tag>;                                           \
      const auto lo = execute(LowerOp{}, tag, value);                    \
      const auto hi = execute(UpperOp{}, tag, value);                    \
      return execute(                                                    \
          ConcatOp{}, Child{},                                           \
          execute(op, Child{}, lo), execute(op, Child{}, hi));           \
    }                                                                    \
  }

VECOPS_NVEC_DEFINE_RECURSIVE_SELECT(EvenOp);
VECOPS_NVEC_DEFINE_RECURSIVE_SELECT(OddOp);

#undef VECOPS_NVEC_DEFINE_RECURSIVE_SELECT

template <typename Backend, VectorTag Tag>
struct GenericImpl<Backend, ConcatEvenOp, Tag> {
  static VECOPS_ALWAYS_INLINE Vec<Tag> call(
      ConcatEvenOp, Tag tag, Vec<Tag> a, Vec<Tag> b) {
    return execute(
        ConcatOp{}, tag,
        execute(EvenOp{}, tag, a), execute(EvenOp{}, tag, b));
  }
};

template <typename Backend, VectorTag Tag>
struct GenericImpl<Backend, ConcatOddOp, Tag> {
  static VECOPS_ALWAYS_INLINE Vec<Tag> call(
      ConcatOddOp, Tag tag, Vec<Tag> a, Vec<Tag> b) {
    return execute(
        ConcatOp{}, tag,
        execute(OddOp{}, tag, a), execute(OddOp{}, tag, b));
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
        execute(LowerOp{}, Child{}, a),
        execute(LowerOp{}, Child{}, b));
    const auto hi = execute(
        op, Child{},
        execute(UpperOp{}, Child{}, a),
        execute(UpperOp{}, Child{}, b));
    return execute(ConcatOp{}, tag, lo, hi);
  }
};

#define VECOPS_NVEC_DEFINE_GENERIC_SHUFFLE(OpType)                        \
  template <typename Backend, VectorTag Tag>                             \
    requires (RepresentationTraits<Backend, Tag>::word_count > 1)       \
  struct GenericImpl<Backend, OpType, Tag> {                             \
    static VECOPS_ALWAYS_INLINE Vec<Tag> call(                           \
        OpType op, Tag tag, Vec<Tag> value,                             \
        Vec<IndexTag<Tag>> indices) {                                    \
      return construct_words<Backend>(                                  \
          tag,                                                          \
          [&]<nint_t Index>(Tag parent) {                               \
            return execute_word<Index, Backend>(                        \
                op, parent, ::vecops::nvec::get_word<Index>(tag, value), \
                ::vecops::nvec::get_word<Index>(                        \
                    IndexTag<Tag>{}, indices));                          \
          });                                                           \
    }                                                                    \
  }

VECOPS_NVEC_DEFINE_GENERIC_SHUFFLE(ShuffleOp);
VECOPS_NVEC_DEFINE_GENERIC_SHUFFLE(LocalShuffleOp);

#undef VECOPS_NVEC_DEFINE_GENERIC_SHUFFLE

template <typename>
struct IsMaskedOption : std::false_type {};

template <MaskValue M>
struct IsMaskedOption<opt::Masked<M>> : std::true_type {
  using Value = M;
};

template <typename T>
inline constexpr bool is_masked_option =
    IsMaskedOption<std::remove_cvref_t<T>>::value;

template <typename T>
struct IsFirstOption : std::bool_constant<
    std::same_as<std::remove_cvref_t<T>, opt::First>> {};

template <typename T>
inline constexpr bool is_first_option = IsFirstOption<T>::value;

template <typename>
struct IsVectorMergeOption : std::false_type {};

template <VectorValue V>
struct IsVectorMergeOption<opt::VectorMerge<V>> : std::true_type {
  using Value = V;
};

template <typename T>
inline constexpr bool is_vector_merge_option =
    IsVectorMergeOption<std::remove_cvref_t<T>>::value;

template <typename>
struct IsScalarMergeOption : std::false_type {};

template <Element T>
struct IsScalarMergeOption<opt::ScalarMerge<T>> : std::true_type {
  using Value = T;
};

template <typename T>
inline constexpr bool is_scalar_merge_option =
    IsScalarMergeOption<std::remove_cvref_t<T>>::value;

template <VectorTag Tag, typename Option>
inline constexpr bool is_fill_option_for = [] {
  using Clean = std::remove_cvref_t<Option>;
  if constexpr (is_masked_option<Clean>) {
    return std::same_as<typename IsMaskedOption<Clean>::Value, Mask<Tag>>;
  } else if constexpr (is_first_option<Clean>) {
    return true;
  } else if constexpr (is_vector_merge_option<Clean>) {
    return std::same_as<
        typename IsVectorMergeOption<Clean>::Value, Vec<Tag>>;
  } else if constexpr (is_scalar_merge_option<Clean>) {
    return std::same_as<
        typename IsScalarMergeOption<Clean>::Value, ElementOf<Tag>>;
  } else {
    return false;
  }
}();

template <template <typename> typename Predicate,
          typename First, typename... Rest>
VECOPS_ALWAYS_INLINE constexpr decltype(auto) find_option(
    First&& first, Rest&&... rest) {
  if constexpr (Predicate<std::remove_cvref_t<First>>::value) {
    return std::forward<First>(first);
  } else {
    static_assert(sizeof...(Rest) > 0, "requested option is absent");
    return find_option<Predicate>(std::forward<Rest>(rest)...);
  }
}

} // namespace vecops::nvec::details

#endif // VECOPS_NVEC_DETAILS_BASIC_H
