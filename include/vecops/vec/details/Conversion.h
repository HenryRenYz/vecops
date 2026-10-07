// SPDX-FileCopyrightText: 2026 HenryRenYz and vecops contributors
// SPDX-License-Identifier: MIT
#ifndef VECOPS_VEC_DETAILS_CONVERSION_H
#define VECOPS_VEC_DETAILS_CONVERSION_H

/**
 * @file Conversion.h
 * @brief Conversion infrastructure: compile-time options validation
 * (valid_conversion_options), mask conversion validation, and the
 * GenericImpl fallback (ConvertOp::call) that performs lane-by-lane
 * element conversion using the scalar convert/ wrap_convert utilities.
 */

#include <algorithm>

#include "vecops/vec/details/Basic.h"
#include "vecops/vec/Options.h"
#include "vecops/util/ScalarConvert.h"

namespace vecops::vec::details {

/* **************************************************************************** */
//    Compile-time validation for conversion options and mask conversion        //
/* **************************************************************************** */

template <VectorTag ToTag, VectorTag FromTag>
consteval bool valid_mask_conversion() {
  if constexpr (is_fixed_tag_v<ToTag> && is_fixed_tag_v<FromTag>) {
    return fixed_lanes_v<ToTag> == fixed_lanes_v<FromTag>;
  } else if constexpr (is_scalable_tag_v<ToTag> && is_scalable_tag_v<FromTag>) {
    return scale_power_v<ToTag> ==
        scale_power_v<FromTag> + element_size_shift(
            sizeof(ElementOf<FromTag>), sizeof(ElementOf<ToTag>));
  } else {
    return false;
  }
}

template <VectorTag ToTag, VectorTag FromTag, typename... Options>
consteval bool valid_conversion_options() {
  using From = ElementOf<FromTag>;
  using To = ElementOf<ToTag>;
  constexpr std::size_t layout_count =
      option_count_v<IsOrderedOption, Options...> +
      option_count_v<IsUnorderedOption, Options...> +
      option_count_v<IsLaneOption, Options...>;
  constexpr std::size_t value_count =
      option_count_v<IsSaturateOption, Options...> +
      option_count_v<IsWrapOption, Options...>;
  constexpr std::size_t population_count =
      option_count_v<IsZeroOption, Options...> +
      option_count_v<IsVectorMergeOption, Options...> +
      option_count_v<IsScalarMergeOption, Options...>;
  constexpr std::size_t masked_count =
      option_count_v<IsMaskedOption, Options...>;
  constexpr std::size_t unmasked_count =
      option_count_v<IsUnmaskedOption, Options...>;
  constexpr std::size_t active_count = masked_count + unmasked_count;
  constexpr bool supported_options =
      ((is_ordered_option_v<std::remove_cvref_t<Options>> ||
        is_unordered_option_v<std::remove_cvref_t<Options>> ||
        is_lane_option_v<std::remove_cvref_t<Options>> ||
        is_saturate_option_v<std::remove_cvref_t<Options>> ||
        is_wrap_option_v<std::remove_cvref_t<Options>> ||
        is_unmasked_option_v<std::remove_cvref_t<Options>> ||
        is_masked_option_for_v<ToTag, Options> ||
        is_vector_population_option_for_v<ToTag, Options>) && ...);
  constexpr bool wraps = option_count_v<IsWrapOption, Options...> == 1;
  constexpr bool lane_layout = option_count_v<IsLaneOption, Options...> == 1;
  constexpr bool ordinary_layout = !lane_layout;
  if constexpr (!supported_options || layout_count > 1 || value_count > 1 ||
                population_count > 1 || active_count > 1) {
    return false;
  } else if constexpr (
      wraps && !(std::integral<From> && std::integral<To> &&
                 sizeof(To) < sizeof(From))) {
    return false;
  } else if constexpr (ordinary_layout) {
    return valid_mask_conversion<ToTag, FromTag>() &&
        (population_count == 0 || masked_count == 1);
  } else {
    constexpr int ratio = sizeof(From) < sizeof(To)
        ? static_cast<int>(sizeof(To) / sizeof(From))
        : static_cast<int>(sizeof(From) / sizeof(To));
    constexpr int phase = [] {
      int found = -1;
      ([]<typename Option>(int& value) {
        if constexpr (is_lane_option_v<Option>)
          value = IsLaneOption<Option>::phase;
      }.template operator()<std::remove_cvref_t<Options>>(found), ...);
      return found;
    }();
    return active_count == 0 && same_logical_bytes_v<ToTag, FromTag> &&
        (ratio == 2 || ratio == 4 || ratio == 8) &&
        (phase == 0 || (phase == 1 && ratio == 2)) &&
        (sizeof(From) > sizeof(To) || population_count == 0);
  }
}

template <typename... Options>
inline constexpr bool conversion_uses_wrap_v =
    option_count_v<IsWrapOption, Options...> == 1;

template <typename... Options>
inline constexpr bool conversion_uses_lane_v =
    option_count_v<IsLaneOption, Options...> == 1;

template <typename... Options>
inline constexpr bool conversion_is_masked_v =
    option_count_v<IsMaskedOption, Options...> == 1;

template <typename... Options>
consteval int conversion_lane_phase() {
  int found = -1;
  ([]<typename Option>(int& value) {
    if constexpr (is_lane_option_v<Option>)
      value = IsLaneOption<Option>::phase;
  }.template operator()<std::remove_cvref_t<Options>>(found), ...);
  return found;
}

template <typename Backend, VectorTag Tag, typename... Options>
VECOPS_ALWAYS_INLINE Vec<Tag> conversion_population(
    Tag tag, Options&&... options) {
  if constexpr (option_count_v<IsVectorMergeOption, Options...> == 1) {
    return find_option<IsVectorMergeOption>(
        std::forward<Options>(options)...).value;
  } else if constexpr (option_count_v<IsScalarMergeOption, Options...> == 1) {
    return execute(
        FillOp{}, tag,
        find_option<IsScalarMergeOption>(
            std::forward<Options>(options)...).value);
  } else {
    return fill(tag, ElementOf<Tag>{});
  }
}

/** Recursively selects the even lanes Levels times, halving the tag. */
template <int Levels, VectorTag Tag>
VECOPS_ALWAYS_INLINE auto conversion_select_even(
    Tag tag, Vec<Tag> value) {
  if constexpr (Levels == 0) {
    return value;
  } else {
    using HalfTag = Half<Tag>;
    return conversion_select_even<Levels - 1>(
        HalfTag{}, even(tag, value));
  }
}

/** Recursively widens @p values (tag ValuesTag) back to Tag, writing them
 *  into the even lanes and preserving the fallback's odd lanes. */
template <int Levels, VectorTag Tag, VectorTag ValuesTag>
VECOPS_ALWAYS_INLINE Vec<Tag> conversion_insert_even(
    Tag tag, Vec<ValuesTag> values, Vec<Tag> fallback) {
  if constexpr (Levels == 0) {
    static_assert(std::same_as<Tag, ValuesTag>);
    return values;
  } else {
    using HalfTag = Half<Tag>;
    const auto fallback_even = even(tag, fallback);
    const auto fallback_odd = odd(tag, fallback);
    const auto result_even =
        conversion_insert_even<Levels - 1, HalfTag, ValuesTag>(
            HalfTag{}, values, fallback_even);
    return interleave(tag, result_even, fallback_odd);
  }
}

/* **************************************************************************** */
//    GenericImpl for ConvertOp: lane-by-lane element conversion                //
/* **************************************************************************** */

template <typename Backend, VectorTag ToTag>
struct GenericImpl<Backend, ConvertOp, ToTag> {
  template <VectorTag FromTag, typename... Options>
    requires (valid_conversion_options<ToTag, FromTag, Options...>())
  static VECOPS_ALWAYS_INLINE Vec<ToTag> call(
      ConvertOp, ToTag to, FromTag from, Vec<FromTag> value,
      Options&&... options) {
    constexpr bool lane_layout = conversion_uses_lane_v<Options...>;
    constexpr bool masked = conversion_is_masked_v<Options...>;
    constexpr bool wraps = conversion_uses_wrap_v<Options...>;
    constexpr bool narrows =
        sizeof(ElementOf<FromTag>) > sizeof(ElementOf<ToTag>);
    constexpr int ratio = sizeof(ElementOf<FromTag>) < sizeof(ElementOf<ToTag>)
        ? static_cast<int>(sizeof(ElementOf<ToTag>) / sizeof(ElementOf<FromTag>))
        : static_cast<int>(sizeof(ElementOf<FromTag>) / sizeof(ElementOf<ToTag>));
    constexpr int phase = lane_layout
        ? conversion_lane_phase<Options...>() : 0;
    const auto initial = [&]() VECOPS_INLINE_LAMBDA {
      if constexpr ((lane_layout && narrows) || masked)
        return conversion_population<Backend>(
            to, std::forward<Options>(options)...);
      else
        return fill(to, ElementOf<ToTag>{});
    }();
    return construct_words<Backend>(
        to, [&]<nint_t Index>(ToTag) VECOPS_INLINE_LAMBDA {
      auto word = ::vecops::vec::get_word<Index>(to, initial);
      constexpr nint_t word_lanes =
          RepresentationTraits<Backend, ToTag>::word_lanes;
      const nint_t begin = Index * word_lanes;
      const nint_t end = std::min(begin + word_lanes, size(to));
      for (nint_t output_lane = begin; output_lane < end; ++output_lane) {
        if constexpr (lane_layout && narrows) {
          if ((output_lane - phase) % ratio != 0) continue;
        }
        const nint_t input_lane = lane_layout && !narrows
            ? ratio * output_lane + phase
            : lane_layout ? (output_lane - phase) / ratio : output_lane;
        if constexpr (lane_layout && narrows) {
          if (input_lane < 0 || input_lane >= size(from)) continue;
        }
        if constexpr (masked) {
          const auto& output_mask = find_option<IsMaskedOption>(
              std::forward<Options>(options)...).value;
          if (!get_mask_lane_at<Backend>(
                  to, output_mask, output_lane)) continue;
        }
        const auto input = get_vec_lane_at<Backend>(
            from, value, input_lane);
        const auto converted = [&]() VECOPS_INLINE_LAMBDA {
          if constexpr (wraps)
            return ::vecops::wrap_convert<ElementOf<ToTag>>(input);
          else
            return ::vecops::convert<
                ElementOf<ToTag>, ElementOf<FromTag>>(input);
        }();
        word = execute_word<Index, Backend>(
            SetVecLaneOp{}, to, word, output_lane - begin, converted);
      }
      return word;
    });
  }

  template <VectorTag FromTag>
    requires (valid_mask_conversion<ToTag, FromTag>())
  static VECOPS_ALWAYS_INLINE Mask<ToTag> call(
      ConvertOp, ToTag to, FromTag from, Mask<FromTag> value) {
    return construct_mask_words<Backend>(
        to, [&]<nint_t Index>(ToTag) VECOPS_INLINE_LAMBDA {
      auto word = execute_word<Index, Backend>(MaskFillOp{}, to, false);
      constexpr nint_t word_lanes =
          RepresentationTraits<Backend, ToTag>::word_lanes;
      const nint_t begin = Index * word_lanes;
      const nint_t end = std::min(begin + word_lanes, size(to));
      for (nint_t lane = begin; lane < end; ++lane) {
        word = execute_word<Index, Backend>(
            SetMaskLaneOp{}, to, word, lane - begin,
            get_mask_lane_at<Backend>(from, value, lane));
      }
      return word;
    });
  }
};

} // namespace vecops::vec::details

#endif // VECOPS_VEC_DETAILS_CONVERSION_H
