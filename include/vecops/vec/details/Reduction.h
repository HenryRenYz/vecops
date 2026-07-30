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

#include "vecops/vec/details/Options.h"

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
  if constexpr (option_count<IsUnmaskedOption, Options...> == 1) {
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
struct ReductionCombineOp<ReduceAddOp> { using Type = AddOp; };
template <>
struct ReductionCombineOp<ReduceMaxOp> { using Type = MaxOp; };
template <>
struct ReductionCombineOp<ReduceMinOp> { using Type = MinOp; };

template <typename ReduceOp>
using ReductionCombineOpOf = typename ReductionCombineOp<ReduceOp>::Type;

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
        ReduceOp{}, tag, value, [&](auto left) -> decltype(auto) {
          return fold_reduction_words<
              Backend, ReduceOp, Begin + left_count, right_count>(
              ReduceOp{}, tag, value, [&](auto right) -> decltype(auto) {
                auto folded = execute_word<Begin, Backend>(
                    ReductionCombineOpOf<ReduceOp>{}, tag, left, right);
                return std::forward<Callback>(callback)(folded);
              });
        });
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
        [&](auto left, auto left_mask) -> decltype(auto) {
          return fold_masked_reduction_words<
              Backend, ReduceOp, Begin + left_count, right_count>(
              ReduceOp{}, tag, value, mask,
              [&](auto right, auto right_mask) -> decltype(auto) {
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
    return fold_reduction_words<Backend, ReduceOp, 0, count>(
        op, tag, value, [&](auto folded) {
          return execute_word<0, Backend>(op, tag, folded);
        });
  }

  static VECOPS_ALWAYS_INLINE ElementOf<Tag> call(
      ReduceOp op, Tag tag, Vec<Tag> value, Mask<Tag> mask) {
    constexpr nint_t count = RepresentationTraits<Backend, Tag>::word_count;
    return fold_masked_reduction_words<Backend, ReduceOp, 0, count>(
        op, tag, value, mask, [&](auto folded, auto active) {
          return execute_word<0, Backend>(op, tag, folded, active);
        });
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
