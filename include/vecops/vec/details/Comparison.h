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

namespace vecops::vec::details {

/* **************************************************************************** */
//    Option validation and dispatch for comparison operations                  //
/* **************************************************************************** */

template <VectorTag Tag, typename Option>
inline constexpr bool is_comparison_option_for =
    IsUnmaskedOption<std::remove_cvref_t<Option>>::value ||
    is_masked_option_for<Tag, Option> ||
    is_mask_population_option_for<Tag, Option>;

template <VectorTag Tag, typename... Options>
consteval void validate_comparison_options() {
  static_assert(
      (is_comparison_option_for<Tag, Options> && ...),
      "comparison received an option with the wrong kind or mask type");
  constexpr std::size_t masked_count =
      (std::size_t{0} + ... + std::size_t{is_masked_option<Options>});
  constexpr std::size_t unmasked_count =
      option_count<IsUnmaskedOption, Options...>;
  constexpr std::size_t zero_count =
      (std::size_t{0} + ... + std::size_t{is_zero_option<Options>});
  constexpr std::size_t merge_count =
      (std::size_t{0} + ... + std::size_t{is_mask_merge_option<Options>});
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

template <typename Op, VectorTag Tag, typename... Options>
VECOPS_ALWAYS_INLINE Mask<Tag> execute_comparison_options(
    Op op, Tag tag, Vec<Tag> a, Vec<Tag> b, Options&&... options) {
  validate_comparison_options<Tag, Options...>();
  constexpr std::size_t unmasked_count =
      option_count<IsUnmaskedOption, Options...>;
  constexpr std::size_t merge_count =
      (std::size_t{0} + ... + std::size_t{is_mask_merge_option<Options>});
  if constexpr (unmasked_count == 1) {
    return execute(op, tag, a, b);
  } else {
    const auto& active = find_option<IsMaskedOption>(
        std::forward<Options>(options)...).value;
    const auto active_result = execute(op, tag, a, b, active);
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
    Op op, Tag tag, Vec<Tag> value, Options&&... options) {
  validate_comparison_options<Tag, Options...>();
  constexpr std::size_t unmasked_count =
      option_count<IsUnmaskedOption, Options...>;
  constexpr std::size_t merge_count =
      (std::size_t{0} + ... + std::size_t{is_mask_merge_option<Options>});
  if constexpr (unmasked_count == 1) {
    return execute(op, tag, value);
  } else {
    const auto& active = find_option<IsMaskedOption>(
        std::forward<Options>(options)...).value;
    const auto active_result = execute(op, tag, value, active);
    if constexpr (merge_count == 1) {
      const auto& inactive = find_option<IsMaskMergeOption>(
          std::forward<Options>(options)...).value;
      return mask_or(tag, active_result, mask_andnot(tag, active, inactive));
    } else {
      return active_result;
    }
  }
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
