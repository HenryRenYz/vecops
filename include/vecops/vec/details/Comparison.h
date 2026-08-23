#ifndef VECOPS_VEC_DETAILS_COMPARISON_H
#define VECOPS_VEC_DETAILS_COMPARISON_H

/**
 * @file Comparison.h
 * @brief Comparison operation infrastructure: options validation
 * (validate_comparison_options), option dispatch
 * (execute_comparison_options), and the multi-word GenericImpl
 * fallback (ComparisonWordBatch) for mask-producing comparisons.
 */

#include <utility>

#include "vecops/vec/details/Wordwise.h"
#include "vecops/vec/details/Request.h"

namespace vecops::vec::details {

/* **************************************************************************** */
//    Option validation and dispatch for comparison operations                  //
/* **************************************************************************** */

template <VectorTag Tag, typename Option>
inline constexpr bool is_comparison_option_for_v =
    IsUnmaskedOption<std::remove_cvref_t<Option>>::value ||
    is_masked_option_for_v<Tag, Option> ||
    is_mask_population_option_for_v<Tag, Option>;

template <VectorTag Tag, typename... Options>
consteval void validate_comparison_options() {
  static_assert(
      (is_comparison_option_for_v<Tag, Options> && ...),
      "comparison received an option with the wrong kind or mask type");
  constexpr std::size_t masked_count =
      (std::size_t{0} + ... + std::size_t{is_masked_option_v<Options>});
  constexpr std::size_t unmasked_count =
      option_count_v<IsUnmaskedOption, Options...>;
  constexpr std::size_t zero_count =
      (std::size_t{0} + ... + std::size_t{is_zero_option_v<Options>});
  constexpr std::size_t merge_count =
      (std::size_t{0} + ... + std::size_t{is_mask_merge_option_v<Options>});
  static_assert(
      masked_count + unmasked_count == 1,
      "comparison requires exactly one opt::masked or opt::unmasked");
  static_assert(
      masked_count * unmasked_count == 0,
      "opt::masked and opt::unmasked are mutually exclusive");
  static_assert(zero_count <= 1, "comparison accepts at most one opt::zero");
  static_assert(merge_count <= 1, "comparison accepts at most one mask merge");
  static_assert(
      zero_count + merge_count <= 1,
      "comparison zero and mask merge policies are mutually exclusive");
}

/**
 * Shared option resolution for the comparison dispatchers: validates the
 * option pack, computes the result unmasked or under the active mask, and
 * merges the inactive lanes when a mask-merge policy is present. The value
 * operands are captured by the caller's compute lambdas.
 */
template <VectorTag Tag, typename ComputeUnmasked, typename ComputeMasked,
          typename... Options>
VECOPS_ALWAYS_INLINE Mask<Tag> comparison_options_dispatch(
    Tag tag, ComputeUnmasked&& compute_unmasked,
    ComputeMasked&& compute_masked, Options&&... options) {
  validate_comparison_options<Tag, Options...>();
  constexpr std::size_t unmasked_count =
      option_count_v<IsUnmaskedOption, Options...>;
  constexpr std::size_t merge_count =
      (std::size_t{0} + ... + std::size_t{is_mask_merge_option_v<Options>});
  if constexpr (unmasked_count == 1) {
    return compute_unmasked();
  } else {
    const auto& active = find_option<IsMaskedOption>(
        std::forward<Options>(options)...).value;
    const auto active_result = compute_masked(active);
    if constexpr (merge_count == 1) {
      const auto& inactive = find_option<IsMaskMergeOption>(
          std::forward<Options>(options)...).value;
      return mask_or(tag, active_result, mask_andnot(tag, active, inactive));
    } else {
      return active_result;
    }
  }
}

template <typename Op, VectorTag Tag, typename... Options>
VECOPS_ALWAYS_INLINE Mask<Tag> execute_comparison_options(
    Op op, Tag tag, Vec<Tag> a, Vec<Tag> b, Options&&... options) {
  return comparison_options_dispatch(
      tag,
      [&] { return execute(op, tag, a, b); },
      [&](const Mask<Tag>& active) {
        return execute(op, tag, a, b, active);
      },
      std::forward<Options>(options)...);
}

template <typename Op, VectorTag Tag, typename... Options>
VECOPS_ALWAYS_INLINE Mask<Tag> execute_comparison_options(
    Op op, Tag tag, Vec<Tag> value, Options&&... options) {
  return comparison_options_dispatch(
      tag,
      [&] { return execute(op, tag, value); },
      [&](const Mask<Tag>& active) {
        return execute(op, tag, value, active);
      },
      std::forward<Options>(options)...);
}

/**
 * Shared request resolution for the comparison dispatchers: no first-count
 * form exists; unmasked, masked, and mask-merge outcomes are selected
 * through the caller's compute lambdas. The merge lambda is only instantiated
 * for Inactive::MergeMask requests, whose type carries the mask_merge field.
 */
template <VectorTag Tag, Active A, Inactive I, typename ComputeUnmasked,
          typename ComputeMasked, typename ComputeMerge>
VECOPS_ALWAYS_INLINE Mask<Tag> comparison_request_dispatch(
    Tag tag, ComputeUnmasked&& compute_unmasked,
    ComputeMasked&& compute_masked, ComputeMerge&& compute_merge) {
  static_assert(
      A != Active::First,
      "comparison operations have no first-count form");
  if constexpr (A == Active::Unmasked) {
    return compute_unmasked();
  } else {
    const auto active_result = compute_masked();
    if constexpr (I == Inactive::MergeMask) {
      return compute_merge(active_result);
    } else {
      return active_result;
    }
  }
}

/** Request-driven comparison dispatch (binary). */
template <typename Op, VectorTag Tag, Active A, Inactive I>
VECOPS_ALWAYS_INLINE Mask<Tag> execute_comparison_request(
    Op op, Tag tag, Vec<Tag> a, Vec<Tag> b,
    const OpRequest<Tag, A, I>& request) {
  return comparison_request_dispatch<Tag, A, I>(
      tag,
      [&] { return execute(op, tag, a, b); },
      [&] { return execute(op, tag, a, b, *request.mask); },
      [&](const Mask<Tag>& partial) {
        return mask_or(
            tag, partial,
            mask_andnot(tag, *request.mask, *request.mask_merge));
      });
}

/** Request-driven comparison dispatch (unary / value-to-scalar forms). */
template <typename Op, VectorTag Tag, Active A, Inactive I>
VECOPS_ALWAYS_INLINE Mask<Tag> execute_comparison_request(
    Op op, Tag tag, Vec<Tag> value,
    const OpRequest<Tag, A, I>& request) {
  return comparison_request_dispatch<Tag, A, I>(
      tag,
      [&] { return execute(op, tag, value); },
      [&] { return execute(op, tag, value, *request.mask); },
      [&](const Mask<Tag>& partial) {
        return mask_or(
            tag, partial,
            mask_andnot(tag, *request.mask, *request.mask_merge));
      });
}

template <typename Backend, typename Op, VectorTag Tag>
struct ComparisonWordBatch {
  template <typename... Values>
    requires ((std::same_as<std::remove_cvref_t<Values>, Vec<Tag>> ||
               std::same_as<std::remove_cvref_t<Values>, Mask<Tag>>) && ...)
  static VECOPS_ALWAYS_INLINE Mask<Tag> call(
      Op op, Tag tag, Values... values) {
    return construct_mask_words<Backend>(
        tag,
        [&]<nint_t Index>(Tag) {
          return execute_word<Index, Backend>(
              op, tag, ::vecops::vec::get_word<Index>(tag, values)...);
        });
  }
};

#define VECOPS_VEC_DEFINE_COMPARISON_BATCH(OpType)                       \
  template <typename Backend, VectorTag Tag>                              \
    requires (RepresentationTraits<Backend, Tag>::word_count > 1)        \
  struct GenericImpl<Backend, OpType, Tag>                                \
      : ComparisonWordBatch<Backend, OpType, Tag> {}

VECOPS_VEC_DEFINE_COMPARISON_BATCH(CmpEqOp);
VECOPS_VEC_DEFINE_COMPARISON_BATCH(CmpNeOp);
VECOPS_VEC_DEFINE_COMPARISON_BATCH(CmpLtOp);
VECOPS_VEC_DEFINE_COMPARISON_BATCH(CmpGtOp);
VECOPS_VEC_DEFINE_COMPARISON_BATCH(CmpLeOp);
VECOPS_VEC_DEFINE_COMPARISON_BATCH(CmpGeOp);
VECOPS_VEC_DEFINE_COMPARISON_BATCH(IsNanOp);
VECOPS_VEC_DEFINE_COMPARISON_BATCH(IsPosInfOp);
VECOPS_VEC_DEFINE_COMPARISON_BATCH(IsNegInfOp);
VECOPS_VEC_DEFINE_COMPARISON_BATCH(IsInfOp);

#undef VECOPS_VEC_DEFINE_COMPARISON_BATCH

} // namespace vecops::vec::details

#endif // VECOPS_VEC_DETAILS_COMPARISON_H
