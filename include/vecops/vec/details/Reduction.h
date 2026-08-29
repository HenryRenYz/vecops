#ifndef VECOPS_VEC_DETAILS_REDUCTION_H
#define VECOPS_VEC_DETAILS_REDUCTION_H

/**
 * @file Reduction.h
 * @brief Option dispatch (execute_reduction_options), balanced fold
 * infrastructure for multi-word reductions (fold_reduction_words), and
 * the multi-word GenericImpl fallback (ReductionGenericImpl).
 *
 * Reductions use a continuation-passing fold to avoid storing vector words
 * and predicates together — a limitation of sizeless SVE where predicates
 * cannot be stored as ordinary struct members.
 */

#include "vecops/vec/Options.h"
#include "vecops/vec/details/Request.h"

namespace vecops::vec::details {

/* **************************************************************************** */
//    Option dispatch for reductions: masked/unmasked, no population options    //
/* **************************************************************************** */

template <typename Op, VectorTag Tag, typename... Options>
VECOPS_ALWAYS_INLINE ElementOf<Tag> execute_reduction_options(
    Op op, Tag tag, Vec<Tag> value, Options&&... options) {
  static_assert(
      ReductionFilterOptions<Tag, Options...>,
      "reduction accepts exactly one opt::masked or opt::unmasked");
  if constexpr (option_count_v<IsUnmaskedOption, Options...> == 1) {
    return execute(op, tag, value);
  } else {
    const auto& mask = find_option<IsMaskedOption>(
        std::forward<Options>(options)...).value;
    return execute(op, tag, value, mask);
  }
}

/* **************************************************************************** */
//    Balanced continuation fold for multi-word reductions                      //
/* **************************************************************************** */

/** Request-driven reduction dispatch. */
template <typename Op, VectorTag Tag, Active A>
VECOPS_ALWAYS_INLINE ElementOf<Tag> execute_reduction_request(
    Op op, Tag tag, Vec<Tag> value, const ReduceRequest<Tag, A>& request) {
  if constexpr (A == Active::Unmasked) {
    return execute(op, tag, value);
  } else {
    return execute(op, tag, value, *request.mask);
  }
}

/**
 * Balanced continuation fold for reductions.
 *
 * The callback form is required for sizeless SVE: a folded vector word and
 * predicate cannot be stored together in std::pair or an ordinary struct.
 * Every combine retains the complete parent Tag. Multi-word representations
 * use complete physical words, so folding into word zero preserves all lanes.
 *
 * The fold tree splits the word range at each level into left/right halves,
 * recursively reducing each side before combining with the ReductionCombineOp
 * and passing the accumulated result forward via a callback.
 */
/**
 * Maps each reduction operation to its lane-wise combine operation.
 * reduce_add → AddOp, reduce_max → MaxOp, reduce_min → MinOp.
 */
template <typename ReduceOp>
struct ReductionCombineOp;

template <>
struct ReductionCombineOp<ReduceAddOp> { using type = AddOp; };
template <>
struct ReductionCombineOp<ReduceMaxOp> { using type = MaxOp; };
template <>
struct ReductionCombineOp<ReduceMinOp> { using type = MinOp; };

template <typename ReduceOp>
using ReductionCombineOpOf = typename ReductionCombineOp<ReduceOp>::type;

template <typename Backend, typename ReduceOp,
          nint_t Begin, nint_t Count,
          VectorTag Tag, typename Callback>
VECOPS_ALWAYS_INLINE decltype(auto) fold_reduction_words(
    ReduceOp, Tag tag, Vec<Tag> value, Callback&& callback) {
  static_assert(Count > 0);
  static_assert(Begin >= 0);
  static_assert(
      Begin + Count <= RepresentationTraits<Backend, Tag>::word_count);
  if constexpr (Count == 1) {
    return std::forward<Callback>(callback)(
        ::vecops::vec::get_word<Begin>(tag, value));
  } else {
    constexpr nint_t left_count = Count / 2;
    constexpr nint_t right_count = Count - left_count;
    return fold_reduction_words<
        Backend, ReduceOp, Begin, left_count>(
        ReduceOp{}, tag, value,
        [&](auto left) VECOPS_INLINE_LAMBDA -> decltype(auto) {
          return fold_reduction_words<
              Backend, ReduceOp, Begin + left_count, right_count>(
              ReduceOp{}, tag, value,
              [&](auto right) VECOPS_INLINE_LAMBDA -> decltype(auto) {
                auto folded = execute_word<Begin, Backend>(
                    ReductionCombineOpOf<ReduceOp>{}, tag, left, right);
                return std::forward<Callback>(callback)(folded);
              });
        });
  }
}

/** Sized backends can return one folded word directly without a lambda tree. */
template <typename Backend, typename ReduceOp,
          nint_t Begin, nint_t Count, VectorTag Tag>
VECOPS_ALWAYS_INLINE NativeWordVec<Tag> fold_sized_reduction_words(
    ReduceOp, Tag tag, const Vec<Tag>& value) {
  static_assert(Count > 0);
  if constexpr (Count == 1) {
    return ::vecops::vec::get_word<Begin>(tag, value);
  } else {
    constexpr nint_t left_count = Count / 2;
    constexpr nint_t right_count = Count - left_count;
    const auto left = fold_sized_reduction_words<
        Backend, ReduceOp, Begin, left_count>(ReduceOp{}, tag, value);
    const auto right = fold_sized_reduction_words<
        Backend, ReduceOp, Begin + left_count, right_count>(
        ReduceOp{}, tag, value);
    return execute_word<Begin, Backend>(
        ReductionCombineOpOf<ReduceOp>{}, tag, left, right);
  }
}

template <VectorTag Tag>
struct SizedMaskedReductionWord {
  NativeWordVec<Tag> value;
  NativeWordMask<Tag> mask;
};

template <typename Backend, typename ReduceOp,
          nint_t Begin, nint_t Count, VectorTag Tag>
VECOPS_ALWAYS_INLINE SizedMaskedReductionWord<Tag>
fold_sized_masked_reduction_words(
    ReduceOp, Tag tag, const Vec<Tag>& value, const Mask<Tag>& mask) {
  static_assert(Count > 0);
  if constexpr (Count == 1) {
    return {
        ::vecops::vec::get_word<Begin>(tag, value),
        ::vecops::vec::get_word<Begin>(tag, mask)};
  } else {
    constexpr nint_t left_count = Count / 2;
    constexpr nint_t right_count = Count - left_count;
    const auto left = fold_sized_masked_reduction_words<
        Backend, ReduceOp, Begin, left_count>(
        ReduceOp{}, tag, value, mask);
    const auto right = fold_sized_masked_reduction_words<
        Backend, ReduceOp, Begin + left_count, right_count>(
        ReduceOp{}, tag, value, mask);
    const auto both = execute_word<Begin, Backend>(
        MaskAndOp{}, tag, left.mask, right.mask);
    const auto either = execute_word<Begin, Backend>(
        MaskOrOp{}, tag, left.mask, right.mask);
    const auto combined = execute_word<Begin, Backend>(
        ReductionCombineOpOf<ReduceOp>{}, tag, left.value, right.value);
    const auto unique = execute_word<Begin, Backend>(
        BlendOp{}, tag, right.value, left.mask, left.value);
    return {
        execute_word<Begin, Backend>(
            BlendOp{}, tag, unique, both, combined),
        either};
  }
}

template <typename Backend, typename ReduceOp,
          nint_t Begin, nint_t Count,
          VectorTag Tag, typename Callback>
VECOPS_ALWAYS_INLINE decltype(auto) fold_masked_reduction_words(
    ReduceOp, Tag tag, Vec<Tag> value, Mask<Tag> mask,
    Callback&& callback) {
  static_assert(Count > 0);
  static_assert(Begin >= 0);
  static_assert(
      Begin + Count <= RepresentationTraits<Backend, Tag>::word_count);
  if constexpr (Count == 1) {
    return std::forward<Callback>(callback)(
        ::vecops::vec::get_word<Begin>(tag, value),
        ::vecops::vec::get_word<Begin>(tag, mask));
  } else {
    constexpr nint_t left_count = Count / 2;
    constexpr nint_t right_count = Count - left_count;
    return fold_masked_reduction_words<
        Backend, ReduceOp, Begin, left_count>(
        ReduceOp{}, tag, value, mask,
        [&](auto left, auto left_mask)
            VECOPS_INLINE_LAMBDA -> decltype(auto) {
          return fold_masked_reduction_words<
              Backend, ReduceOp, Begin + left_count, right_count>(
              ReduceOp{}, tag, value, mask,
              [&](auto right, auto right_mask)
                  VECOPS_INLINE_LAMBDA -> decltype(auto) {
                const auto both = execute_word<Begin, Backend>(
                    MaskAndOp{}, tag, left_mask, right_mask);
                const auto either = execute_word<Begin, Backend>(
                    MaskOrOp{}, tag, left_mask, right_mask);
                const auto combined = execute_word<Begin, Backend>(
                    ReductionCombineOpOf<ReduceOp>{}, tag, left, right);
                const auto unique = execute_word<Begin, Backend>(
                    BlendOp{}, tag, right, left_mask, left);
                const auto folded = execute_word<Begin, Backend>(
                    BlendOp{}, tag, unique, both, combined);
                return std::forward<Callback>(callback)(folded, either);
              });
        });
  }
}

template <typename Backend, typename ReduceOp, VectorTag Tag>
  requires (RepresentationTraits<Backend, Tag>::word_count > 1)
struct ReductionGenericImpl {
  static VECOPS_ALWAYS_INLINE ElementOf<Tag> call(
      ReduceOp op, Tag tag, Vec<Tag> value) {
    constexpr nint_t count = RepresentationTraits<Backend, Tag>::word_count;
    if constexpr (requires { sizeof(Vec<Tag>); }) {
      const auto folded = fold_sized_reduction_words<
          Backend, ReduceOp, 0, count>(op, tag, value);
      return execute_word<0, Backend>(op, tag, folded);
    } else {
      return fold_reduction_words<Backend, ReduceOp, 0, count>(
          op, tag, value, [&](auto folded) VECOPS_INLINE_LAMBDA {
            return execute_word<0, Backend>(op, tag, folded);
          });
    }
  }

  static VECOPS_ALWAYS_INLINE ElementOf<Tag> call(
      ReduceOp op, Tag tag, Vec<Tag> value, Mask<Tag> mask) {
    constexpr nint_t count = RepresentationTraits<Backend, Tag>::word_count;
    if constexpr (requires { sizeof(Vec<Tag>); sizeof(Mask<Tag>); }) {
      const auto folded = fold_sized_masked_reduction_words<
          Backend, ReduceOp, 0, count>(op, tag, value, mask);
      return execute_word<0, Backend>(
          op, tag, folded.value, folded.mask);
    } else {
      return fold_masked_reduction_words<Backend, ReduceOp, 0, count>(
          op, tag, value, mask,
          [&](auto folded, auto active) VECOPS_INLINE_LAMBDA {
            return execute_word<0, Backend>(op, tag, folded, active);
          });
    }
  }
};

#define VECOPS_VEC_DEFINE_REDUCTION_GENERIC(OpType)                    \
  template <typename Backend, VectorTag Tag>                            \
    requires (RepresentationTraits<Backend, Tag>::word_count > 1)      \
  struct GenericImpl<Backend, OpType, Tag>                              \
      : ReductionGenericImpl<Backend, OpType, Tag> {}

VECOPS_VEC_DEFINE_REDUCTION_GENERIC(ReduceAddOp);
VECOPS_VEC_DEFINE_REDUCTION_GENERIC(ReduceMaxOp);
VECOPS_VEC_DEFINE_REDUCTION_GENERIC(ReduceMinOp);

#undef VECOPS_VEC_DEFINE_REDUCTION_GENERIC

} // namespace vecops::vec::details

#endif // VECOPS_VEC_DETAILS_REDUCTION_H
