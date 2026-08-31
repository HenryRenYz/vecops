#ifndef VECOPS_VEC_DETAILS_SVE_CONVERSION_MEMORY_H
#define VECOPS_VEC_DETAILS_SVE_CONVERSION_MEMORY_H

/**
 * @file ConversionMemory.h
 * @brief SVE backend implementations for load_convert and store_convert operations.
 */

#if !defined(HAS_SVE)
#error "This header requires an SVE target"
#endif

#include <limits>
#include <type_traits>

#include "vecops/vec/details/sve/Conversion.h"
#include "vecops/vec/details/sve/Memory.h"

namespace vecops::vec::details {

template <typename... Options>
inline constexpr bool sve_has_oversized_contiguous_conversion_lowering_v =
    option_count_v<IsMemoryAddressingOption, Options...> == 0 &&
    option_count_v<IsUnorderedOption, Options...> == 0 &&
    option_count_v<IsWrapOption, Options...> == 0;

template <VectorTag Tag>
inline constexpr bool sve_conversion_tag_representable_v = [] {
  if constexpr (is_fixed_tag_v<Tag>) return true;
  else return scale_power_v<Tag> <= VEC_MAX_POW;
}();

/**
 * Ordered saturating SVE conversion recursively lowers the logical Tag until
 * both sides fit one physical word. It therefore does not materialize the
 * possibly oversized memory-side Rebind and must run before generic boundary
 * splitting.
 */
template <VectorTag ToTag, Element From, typename... Options>
struct HasOversizedMemoryConversionLowering<
    SVEBackend, LoadConvertOp, ToTag, From, Options...>
    : std::bool_constant<
          sve_has_oversized_contiguous_conversion_lowering_v<Options...>> {};

template <VectorTag FromTag, Element To, typename... Options>
struct HasOversizedMemoryConversionLowering<
    SVEBackend, StoreConvertOp, FromTag, To, Options...>
    : std::bool_constant<
          sve_has_oversized_contiguous_conversion_lowering_v<Options...>> {};


/* **************************************************************************** */
//                       Load and store with conversion                       //
/* **************************************************************************** */

template <Element To, Element From>
VECOPS_ALWAYS_INLINE auto sve_load_extend_integer(
    svbool_t active, const From* pointer) {
  static_assert(
      sve_is_integer_element_v<To> && sve_is_integer_element_v<From> &&
      sizeof(From) < sizeof(To));
  using Wide = SVEInteger<sizeof(To), std::is_signed_v<From>>;
  auto widened = [&]() VECOPS_INLINE_LAMBDA {
    if constexpr (std::same_as<From, int8_t> && sizeof(To) == 2)
      return svld1sb_s16(active, pointer);
    else if constexpr (std::same_as<From, int8_t> && sizeof(To) == 4)
      return svld1sb_s32(active, pointer);
    else if constexpr (std::same_as<From, int8_t> && sizeof(To) == 8)
      return svld1sb_s64(active, pointer);
    else if constexpr (std::same_as<From, uint8_t> && sizeof(To) == 2)
      return svld1ub_u16(active, pointer);
    else if constexpr (std::same_as<From, uint8_t> && sizeof(To) == 4)
      return svld1ub_u32(active, pointer);
    else if constexpr (std::same_as<From, uint8_t> && sizeof(To) == 8)
      return svld1ub_u64(active, pointer);
    else if constexpr (std::same_as<From, int16_t> && sizeof(To) == 4)
      return svld1sh_s32(active, pointer);
    else if constexpr (std::same_as<From, int16_t> && sizeof(To) == 8)
      return svld1sh_s64(active, pointer);
    else if constexpr (std::same_as<From, uint16_t> && sizeof(To) == 4)
      return svld1uh_u32(active, pointer);
    else if constexpr (std::same_as<From, uint16_t> && sizeof(To) == 8)
      return svld1uh_u64(active, pointer);
    else if constexpr (std::same_as<From, int32_t> && sizeof(To) == 8)
      return svld1sw_s64(active, pointer);
    else if constexpr (std::same_as<From, uint32_t> && sizeof(To) == 8)
      return svld1uw_u64(active, pointer);
    else
      static_assert(
          dispatch_dependent_false<To, From>,
          "unsupported SVE extending load element types");
  }();
  // Match ordered vector conversion: widening a signed integer to an
  // unsigned destination sign-extends first, then reinterprets modulo the
  // wider width. Same-width signedness conversion would clamp instead.
  return sve_reinterpret_integer<To, Wide>(widened);
}

template <Element Wide, Element From, Element Index, typename Offset,
          typename Temporality>
VECOPS_ALWAYS_INLINE auto sve_load_extend_integer_indexed_raw(
    svbool_t active, const From* pointer, Offset offsets, Temporality) {
  static_assert(
      sve_is_integer_element_v<Wide> && sve_is_integer_element_v<From> &&
      sizeof(From) < sizeof(Wide));
  constexpr bool wide_signed = std::is_signed_v<Wide>;
  if constexpr (sizeof(Index) == 4) {
#if defined(HAS_SVE2)
    if constexpr (std::same_as<Temporality, mem::NonTemporal>) {
      const auto uoffsets = svreinterpret_u32_s32(offsets);
      if constexpr (std::same_as<From, int8_t> && wide_signed)
        return svldnt1sb_gather_u32offset_s32(active, pointer, uoffsets);
      else if constexpr (std::same_as<From, int8_t>)
        return svldnt1sb_gather_u32offset_u32(active, pointer, uoffsets);
      else if constexpr (std::same_as<From, uint8_t> && wide_signed)
        return svldnt1ub_gather_u32offset_s32(active, pointer, uoffsets);
      else if constexpr (std::same_as<From, uint8_t>)
        return svldnt1ub_gather_u32offset_u32(active, pointer, uoffsets);
      else if constexpr (std::same_as<From, int16_t> && wide_signed)
        return svldnt1sh_gather_u32offset_s32(active, pointer, uoffsets);
      else if constexpr (std::same_as<From, int16_t>)
        return svldnt1sh_gather_u32offset_u32(active, pointer, uoffsets);
      else if constexpr (std::same_as<From, uint16_t> && wide_signed)
        return svldnt1uh_gather_u32offset_s32(active, pointer, uoffsets);
      else
        return svldnt1uh_gather_u32offset_u32(active, pointer, uoffsets);
    }
#endif
    if constexpr (std::same_as<From, int8_t> && wide_signed)
      return svld1sb_gather_s32offset_s32(active, pointer, offsets);
    else if constexpr (std::same_as<From, int8_t>)
      return svld1sb_gather_s32offset_u32(active, pointer, offsets);
    else if constexpr (std::same_as<From, uint8_t> && wide_signed)
      return svld1ub_gather_s32offset_s32(active, pointer, offsets);
    else if constexpr (std::same_as<From, uint8_t>)
      return svld1ub_gather_s32offset_u32(active, pointer, offsets);
    else if constexpr (std::same_as<From, int16_t> && wide_signed)
      return svld1sh_gather_s32offset_s32(active, pointer, offsets);
    else if constexpr (std::same_as<From, int16_t>)
      return svld1sh_gather_s32offset_u32(active, pointer, offsets);
    else if constexpr (std::same_as<From, uint16_t> && wide_signed)
      return svld1uh_gather_s32offset_s32(active, pointer, offsets);
    else
      return svld1uh_gather_s32offset_u32(active, pointer, offsets);
  } else {
#if defined(HAS_SVE2)
    if constexpr (std::same_as<Temporality, mem::NonTemporal>) {
      if constexpr (std::same_as<From, int8_t> && wide_signed)
        return svldnt1sb_gather_s64offset_s64(active, pointer, offsets);
      else if constexpr (std::same_as<From, int8_t>)
        return svldnt1sb_gather_s64offset_u64(active, pointer, offsets);
      else if constexpr (std::same_as<From, uint8_t> && wide_signed)
        return svldnt1ub_gather_s64offset_s64(active, pointer, offsets);
      else if constexpr (std::same_as<From, uint8_t>)
        return svldnt1ub_gather_s64offset_u64(active, pointer, offsets);
      else if constexpr (std::same_as<From, int16_t> && wide_signed)
        return svldnt1sh_gather_s64offset_s64(active, pointer, offsets);
      else if constexpr (std::same_as<From, int16_t>)
        return svldnt1sh_gather_s64offset_u64(active, pointer, offsets);
      else if constexpr (std::same_as<From, uint16_t> && wide_signed)
        return svldnt1uh_gather_s64offset_s64(active, pointer, offsets);
      else
        return svldnt1uh_gather_s64offset_u64(active, pointer, offsets);
    }
#endif
    if constexpr (std::same_as<From, int8_t> && wide_signed)
      return svld1sb_gather_s64offset_s64(active, pointer, offsets);
    else if constexpr (std::same_as<From, int8_t>)
      return svld1sb_gather_s64offset_u64(active, pointer, offsets);
    else if constexpr (std::same_as<From, uint8_t> && wide_signed)
      return svld1ub_gather_s64offset_s64(active, pointer, offsets);
    else if constexpr (std::same_as<From, uint8_t>)
      return svld1ub_gather_s64offset_u64(active, pointer, offsets);
    else if constexpr (std::same_as<From, int16_t> && wide_signed)
      return svld1sh_gather_s64offset_s64(active, pointer, offsets);
    else if constexpr (std::same_as<From, int16_t>)
      return svld1sh_gather_s64offset_u64(active, pointer, offsets);
    else if constexpr (std::same_as<From, uint16_t> && wide_signed)
      return svld1uh_gather_s64offset_s64(active, pointer, offsets);
    else
      return svld1uh_gather_s64offset_u64(active, pointer, offsets);
  }
}

template <int Scale, VectorTag ToTag, Element From, VectorTag IndexTag,
          typename Temporality>
VECOPS_ALWAYS_INLINE Vec<ToTag> sve_load_convert_indexed_integer(
    ToTag to, const From* pointer, Vec<IndexTag> indices,
    Mask<ToTag> mask, Vec<ToTag> inactive, Temporality temporality) {
  using To = ElementOf<ToTag>;
  using I = ElementOf<IndexTag>;
  using Wide = SVEInteger<sizeof(I), std::is_signed_v<From>>;
  using WideTag = Rebind<Wide, ToTag>;
  if constexpr (
      num_words(to) > 1 || num_words(IndexTag{}) > 1 ||
      num_words(WideTag{}) > 1) {
    using ToHalf = Half<ToTag>;
    const auto lower_value = sve_load_convert_indexed_integer<
        Scale, ToHalf, From, Half<IndexTag>>(
        ToHalf{}, pointer, lower(IndexTag{}, indices),
        lower(to, mask), lower(to, inactive),
        temporality);
    const auto upper_value = sve_load_convert_indexed_integer<
        Scale, ToHalf, From, Half<IndexTag>>(
        ToHalf{}, pointer, upper(IndexTag{}, indices),
        upper(to, mask), upper(to, inactive),
        temporality);
    return concat(to, lower_value, upper_value);
  } else {
    const auto wide_mask = sve_convert_mask(WideTag{}, to, mask);
    const auto active = ::vecops::vec::get_word<0>(WideTag{}, wide_mask);
    const auto raw_indices = sve_basic_raw_word(indices);
    const auto offsets = sve_scale_indexed_offsets<Scale>(active, raw_indices);
    const auto wide = sve_load_extend_integer_indexed_raw<Wide, From, I>(
        active, pointer, offsets, temporality);
    const auto converted = [&]() VECOPS_INLINE_LAMBDA {
      if constexpr (sizeof(To) == sizeof(Wide))
        return sve_reinterpret_integer<To, Wide>(wide);
      else if constexpr (
          sizeof(To) < sizeof(Wide) && std::is_signed_v<From> &&
          std::is_unsigned_v<To>)
        return sve_convert_one_word_raw<To, Wide, true>(wide);
      else
        return sve_convert_one_word_raw<To, Wide>(wide);
    }();
    const auto result = sve_basic_wrap_word<ToTag>(converted);
    return blend(to, inactive, mask, result);
  }
}

template <int Scale, VectorTag ToTag, Element From, VectorTag IndexTag,
          typename Temporality>
VECOPS_ALWAYS_INLINE Vec<ToTag> sve_load_convert_indexed_integer_unmasked(
    ToTag to, const From* pointer, Vec<IndexTag> indices,
    Temporality temporality) {
  using To = ElementOf<ToTag>;
  using I = ElementOf<IndexTag>;
  using Wide = SVEInteger<sizeof(I), std::is_signed_v<From>>;
  using WideTag = Rebind<Wide, ToTag>;
  if constexpr (
      num_words(to) > 1 || num_words(IndexTag{}) > 1 ||
      num_words(WideTag{}) > 1) {
    using ToHalf = Half<ToTag>;
    const auto lower_value = sve_load_convert_indexed_integer_unmasked<
        Scale, ToHalf, From, Half<IndexTag>>(
        ToHalf{}, pointer, lower(IndexTag{}, indices),
        temporality);
    const auto upper_value = sve_load_convert_indexed_integer_unmasked<
        Scale, ToHalf, From, Half<IndexTag>>(
        ToHalf{}, pointer, upper(IndexTag{}, indices),
        temporality);
    return concat(to, lower_value, upper_value);
  } else {
    const auto active = sve_prefix_predicate<Wide>(size(WideTag{}));
    const auto raw_indices = sve_basic_raw_word(indices);
    const auto offsets = sve_scale_indexed_offsets<Scale>(active, raw_indices);
    const auto wide = sve_load_extend_integer_indexed_raw<Wide, From, I>(
        active, pointer, offsets, temporality);
    const auto converted = [&]() VECOPS_INLINE_LAMBDA {
      if constexpr (sizeof(To) == sizeof(Wide))
        return sve_reinterpret_integer<To, Wide>(wide);
      else if constexpr (
          sizeof(To) < sizeof(Wide) && std::is_signed_v<From> &&
          std::is_unsigned_v<To>)
        return sve_convert_one_word_raw<To, Wide, true>(wide);
      else
        return sve_convert_one_word_raw<To, Wide>(wide);
    }();
    return sve_basic_wrap_word<ToTag>(converted);
  }
}

template <VectorTag ToTag, Element From>
VECOPS_ALWAYS_INLINE Vec<ToTag> sve_load_convert_word(
    ToTag to, const From* pointer, Mask<ToTag> mask,
    Vec<ToTag> inactive) {
  using To = ElementOf<ToTag>;
  using FromTag = Rebind<From, ToTag>;
  static_assert(num_words(to) == 1 && num_words(FromTag{}) == 1);
  const auto valid = sve_prefix_predicate<To>(size(to));
  const auto active = svand_b_z(valid, valid, mask);
  const auto converted = [&]() VECOPS_INLINE_LAMBDA {
    if constexpr (
        sve_is_integer_element_v<To> && sve_is_integer_element_v<From> &&
        sizeof(From) < sizeof(To)) {
      return sve_load_extend_integer<To>(active, pointer);
    } else if constexpr (
        std::same_as<From, bfloat16_t> && std::same_as<To, float32_t>) {
      const auto bits = svld1uh_u32(
          active, reinterpret_cast<const uint16_t*>(pointer));
      return svreinterpret_f32_u32(
          svlsl_n_u32_x(svptrue_b32(), bits, 16));
    } else {
      // Convert the output predicate before touching memory. This is required
      // when source and result have different predicate granularities; a full
      // source load followed by a blend could fault on an inactive address.
      const auto memory_mask = sve_convert_mask(FromTag{}, to, active);
      const auto loaded = sve_load_memory_word(
          memory_mask, pointer, mem::Temporal{});
      return sve_convert_one_word_raw<To, From>(loaded);
    }
  }();
  const auto result = sve_basic_wrap_word<ToTag>(converted);
  return blend(to, inactive, active, result);
}

template <VectorTag ToTag, Element From>
VECOPS_ALWAYS_INLINE Vec<ToTag> sve_load_convert_word_unmasked(
    ToTag to, const From* pointer) {
  using To = ElementOf<ToTag>;
  using FromTag = Rebind<From, ToTag>;
  static_assert(num_words(to) == 1 && num_words(FromTag{}) == 1);
  const auto active = sve_prefix_predicate<To>(size(to));
  const auto converted = [&]() VECOPS_INLINE_LAMBDA {
    if constexpr (
        sve_is_integer_element_v<To> && sve_is_integer_element_v<From> &&
        sizeof(From) < sizeof(To)) {
      return sve_load_extend_integer<To>(active, pointer);
    } else if constexpr (
        std::same_as<From, bfloat16_t> && std::same_as<To, float32_t>) {
      const auto bits = svld1uh_u32(
          active, reinterpret_cast<const uint16_t*>(pointer));
      return svreinterpret_f32_u32(
          svlsl_n_u32_x(svptrue_b32(), bits, 16));
    } else {
      const auto memory_active =
          sve_prefix_predicate<From>(size(FromTag{}));
      const auto loaded = sve_load_memory_word(
          memory_active, pointer, mem::Temporal{});
      return sve_convert_one_word_raw<To, From>(loaded);
    }
  }();
  return sve_basic_wrap_word<ToTag>(converted);
}

template <VectorTag ToTag, Element From>
VECOPS_ALWAYS_INLINE Vec<ToTag> sve_load_convert_ordered_saturating(
    ToTag to, const From* pointer, Mask<ToTag> mask,
    Vec<ToTag> inactive) {
  using FromTag = Rebind<From, ToTag>;
  static_assert(sve_is_conversion_element_v<ElementOf<ToTag>>);
  static_assert(sve_is_conversion_element_v<From>);
  constexpr bool OneWordPair = [] {
    if constexpr (!sve_conversion_tag_representable_v<FromTag>) return false;
    else return num_words(ToTag{}) == 1 && num_words(FromTag{}) == 1;
  }();
  if constexpr (OneWordPair) {
    return sve_load_convert_word(to, pointer, mask, inactive);
  } else if constexpr (is_scalable_tag_v<ToTag> && num_words(to) > 1) {
    using WordTag = ScalableTag<ElementOf<ToTag>, 0>;
    return construct_words<SVEBackend>(
        to, [&]<nint_t Index>(ToTag) VECOPS_INLINE_LAMBDA {
          return sve_load_convert_ordered_saturating(
              WordTag{}, pointer + Index * size(WordTag{}),
              ::vecops::vec::get_word<Index>(to, mask),
              ::vecops::vec::get_word<Index>(to, inactive));
        });
  } else {
    // Shape-changing conversion cannot use independent word batching: split
    // both complete Tags so the pointer offset remains a logical lane count.
    using ToHalf = Half<ToTag>;
    const auto lower_value = sve_load_convert_ordered_saturating(
        ToHalf{}, pointer,
        lower(to, mask), lower(to, inactive));
    const auto upper_value = sve_load_convert_ordered_saturating(
        ToHalf{}, pointer + size(ToHalf{}),
        upper(to, mask), upper(to, inactive));
    return concat(to, lower_value, upper_value);
  }
}

template <VectorTag ToTag, Element From>
VECOPS_ALWAYS_INLINE Vec<ToTag> sve_load_convert_ordered_saturating(
    ToTag to, const From* pointer) {
  using FromTag = Rebind<From, ToTag>;
  static_assert(sve_is_conversion_element_v<ElementOf<ToTag>>);
  static_assert(sve_is_conversion_element_v<From>);
  constexpr bool OneWordPair = [] {
    if constexpr (!sve_conversion_tag_representable_v<FromTag>) return false;
    else return num_words(ToTag{}) == 1 && num_words(FromTag{}) == 1;
  }();
  if constexpr (OneWordPair) {
    return sve_load_convert_word_unmasked(to, pointer);
  } else if constexpr (is_scalable_tag_v<ToTag> && num_words(to) > 1) {
    using WordTag = ScalableTag<ElementOf<ToTag>, 0>;
    return construct_words<SVEBackend>(
        to, [&]<nint_t Index>(ToTag) VECOPS_INLINE_LAMBDA {
          return sve_load_convert_ordered_saturating(
              WordTag{}, pointer + Index * size(WordTag{}));
        });
  } else {
    using ToHalf = Half<ToTag>;
    const auto lower_value = sve_load_convert_ordered_saturating(
        ToHalf{}, pointer);
    const auto upper_value = sve_load_convert_ordered_saturating(
        ToHalf{}, pointer + size(ToHalf{}));
    return concat(to, lower_value, upper_value);
  }
}

template <Element To, VectorTag FromTag>
VECOPS_ALWAYS_INLINE void sve_narrow_integer_store_word(
    To* pointer, Vec<FromTag> value, svbool_t active) {
  using From = ElementOf<FromTag>;
  static_assert(
      sve_is_integer_element_v<From> && sve_is_integer_element_v<To> &&
      sizeof(From) > sizeof(To) && num_words(FromTag{}) == 1);
  auto raw = sve_basic_raw_word(value);
  if constexpr (std::is_signed_v<From>) {
    constexpr From low = std::is_signed_v<To>
        ? static_cast<From>(std::numeric_limits<To>::lowest()) : From{0};
    constexpr From high = static_cast<From>(std::numeric_limits<To>::max());
    if constexpr (sizeof(From) == 2) {
      raw = svmin_n_s16_x(active, svmax_n_s16_x(active, raw, low), high);
      svst1b_s16(active, reinterpret_cast<int8_t*>(pointer), raw);
    } else if constexpr (sizeof(From) == 4) {
      raw = svmin_n_s32_x(active, svmax_n_s32_x(active, raw, low), high);
      if constexpr (sizeof(To) == 1)
        svst1b_s32(active, reinterpret_cast<int8_t*>(pointer), raw);
      else
        svst1h_s32(active, reinterpret_cast<int16_t*>(pointer), raw);
    } else {
      raw = svmin_n_s64_x(active, svmax_n_s64_x(active, raw, low), high);
      if constexpr (sizeof(To) == 1)
        svst1b_s64(active, reinterpret_cast<int8_t*>(pointer), raw);
      else if constexpr (sizeof(To) == 2)
        svst1h_s64(active, reinterpret_cast<int16_t*>(pointer), raw);
      else
        svst1w_s64(active, reinterpret_cast<int32_t*>(pointer), raw);
    }
  } else {
    constexpr From high = static_cast<From>(std::numeric_limits<To>::max());
    if constexpr (sizeof(From) == 2) {
      raw = svmin_n_u16_x(active, raw, high);
      svst1b_u16(active, reinterpret_cast<uint8_t*>(pointer), raw);
    } else if constexpr (sizeof(From) == 4) {
      raw = svmin_n_u32_x(active, raw, high);
      if constexpr (sizeof(To) == 1)
        svst1b_u32(active, reinterpret_cast<uint8_t*>(pointer), raw);
      else
        svst1h_u32(active, reinterpret_cast<uint16_t*>(pointer), raw);
    } else {
      raw = svmin_n_u64_x(active, raw, high);
      if constexpr (sizeof(To) == 1)
        svst1b_u64(active, reinterpret_cast<uint8_t*>(pointer), raw);
      else if constexpr (sizeof(To) == 2)
        svst1h_u64(active, reinterpret_cast<uint16_t*>(pointer), raw);
      else
        svst1w_u64(active, reinterpret_cast<uint32_t*>(pointer), raw);
    }
  }
}

template <Element To, Element From, typename Raw>
VECOPS_ALWAYS_INLINE Raw sve_clamp_narrow_integer_raw(
    svbool_t active, Raw raw) {
  static_assert(
      sve_is_integer_element_v<From> && sve_is_integer_element_v<To> &&
      sizeof(From) > sizeof(To));
  if constexpr (std::is_signed_v<From>) {
    constexpr From low = std::is_signed_v<To>
        ? static_cast<From>(std::numeric_limits<To>::lowest()) : From{0};
    constexpr From high = static_cast<From>(std::numeric_limits<To>::max());
    if constexpr (sizeof(From) == 2)
      return svmin_n_s16_x(active, svmax_n_s16_x(active, raw, low), high);
    else if constexpr (sizeof(From) == 4)
      return svmin_n_s32_x(active, svmax_n_s32_x(active, raw, low), high);
    else
      return svmin_n_s64_x(active, svmax_n_s64_x(active, raw, low), high);
  } else {
    constexpr From high = static_cast<From>(std::numeric_limits<To>::max());
    if constexpr (sizeof(From) == 2)
      return svmin_n_u16_x(active, raw, high);
    else if constexpr (sizeof(From) == 4)
      return svmin_n_u32_x(active, raw, high);
    else
      return svmin_n_u64_x(active, raw, high);
  }
}

template <int Scale, Element To, VectorTag FromTag, VectorTag IndexTag,
          typename Temporality>
VECOPS_ALWAYS_INLINE void sve_store_convert_indexed_integer(
    FromTag from, To* pointer, Vec<FromTag> value, Vec<IndexTag> indices,
    Mask<FromTag> mask, Temporality temporality) {
  using From = ElementOf<FromTag>;
  using I = ElementOf<IndexTag>;
  if constexpr (num_words(from) > 1 || num_words(IndexTag{}) > 1) {
    using FromHalf = Half<FromTag>;
    sve_store_convert_indexed_integer<Scale, To, FromHalf, Half<IndexTag>>(
        FromHalf{}, pointer, lower(from, value),
        lower(IndexTag{}, indices),
        lower(from, mask), temporality);
    sve_store_convert_indexed_integer<Scale, To, FromHalf, Half<IndexTag>>(
        FromHalf{}, pointer, upper(from, value),
        upper(IndexTag{}, indices),
        upper(from, mask), temporality);
  } else {
    const auto conversion_active =
        ::vecops::vec::get_word<0>(from, mask);
    auto memory_active = conversion_active;
    if constexpr (sizeof(From) < 4 && sizeof(I) >= 4)
      memory_active = svunpklo_b(memory_active);
    if constexpr (sizeof(From) < 2 && sizeof(I) >= 4)
      memory_active = svunpklo_b(memory_active);
    if constexpr (sizeof(From) < 8 && sizeof(I) == 8)
      memory_active = svunpklo_b(memory_active);
    const auto clamped = sve_clamp_narrow_integer_raw<To, From>(
        conversion_active, sve_basic_raw_word(value));
    const auto clamped_value = sve_basic_wrap_word<FromTag>(clamped);
    const auto raw_indices = sve_basic_raw_word(indices);
    if constexpr (sizeof(I) == 4) {
      const auto offsets = sve_scale_indexed_offsets<Scale>(
          memory_active, raw_indices);
      if constexpr (sizeof(From) <= 4) {
        sve_store_indexed_u32_bits(
            memory_active, pointer, offsets,
            sve_expand_indexed_store_u32_bits<FromTag>(clamped_value),
            temporality);
      } else {
        sve_store_indexed_u64_bits(
            memory_active, pointer, svunpklo_s64(offsets),
            sve_expand_indexed_store_u64_bits<FromTag>(clamped_value),
            temporality);
      }
    } else {
      const auto offsets = sve_scale_indexed_offsets<Scale>(
          memory_active, raw_indices);
      sve_store_indexed_u64_bits(
          memory_active, pointer, offsets,
          sve_expand_indexed_store_u64_bits<FromTag>(clamped_value),
          temporality);
    }
  }
}

template <int Scale, Element To, VectorTag FromTag, VectorTag IndexTag,
          typename Temporality>
VECOPS_ALWAYS_INLINE void sve_store_convert_indexed_integer_unmasked(
    FromTag from, To* pointer, Vec<FromTag> value, Vec<IndexTag> indices,
    Temporality temporality) {
  using From = ElementOf<FromTag>;
  using I = ElementOf<IndexTag>;
  if constexpr (num_words(from) > 1 || num_words(IndexTag{}) > 1) {
    using FromHalf = Half<FromTag>;
    sve_store_convert_indexed_integer_unmasked<
        Scale, To, FromHalf, Half<IndexTag>>(
        FromHalf{}, pointer, lower(from, value),
        lower(IndexTag{}, indices), temporality);
    sve_store_convert_indexed_integer_unmasked<
        Scale, To, FromHalf, Half<IndexTag>>(
        FromHalf{}, pointer, upper(from, value),
        upper(IndexTag{}, indices), temporality);
  } else {
    const auto conversion_active =
        sve_prefix_predicate<From>(size(from));
    // The scatter operates at the wider of the data and index granularities.
    // Construct that predicate directly: unpacking a logical subword prefix
    // would reduce its active-lane count at every widening step.
    const auto memory_active = [&]() VECOPS_INLINE_LAMBDA {
      if constexpr (sizeof(I) > sizeof(From))
        return sve_prefix_predicate<I>(size(IndexTag{}));
      else
        return conversion_active;
    }();
    const auto clamped = sve_clamp_narrow_integer_raw<To, From>(
        conversion_active, sve_basic_raw_word(value));
    const auto clamped_value = sve_basic_wrap_word<FromTag>(clamped);
    const auto raw_indices = sve_basic_raw_word(indices);
    if constexpr (sizeof(I) == 4) {
      const auto offsets = sve_scale_indexed_offsets<Scale>(
          memory_active, raw_indices);
      if constexpr (sizeof(From) <= 4) {
        sve_store_indexed_u32_bits(
            memory_active, pointer, offsets,
            sve_expand_indexed_store_u32_bits<FromTag>(clamped_value),
            temporality);
      } else {
        sve_store_indexed_u64_bits(
            memory_active, pointer, svunpklo_s64(offsets),
            sve_expand_indexed_store_u64_bits<FromTag>(clamped_value),
            temporality);
      }
    } else {
      const auto offsets = sve_scale_indexed_offsets<Scale>(
          memory_active, raw_indices);
      sve_store_indexed_u64_bits(
          memory_active, pointer, offsets,
          sve_expand_indexed_store_u64_bits<FromTag>(clamped_value),
          temporality);
    }
  }
}

#if defined(HAS_SVE2)
template <Element To, VectorTag FromTag>
VECOPS_ALWAYS_INLINE auto sve_pack_narrow_integer_x2(
    FromTag from, Vec<FromTag> value) {
  using From = ElementOf<FromTag>;
  static_assert(
      sve_is_integer_element_v<From> && sve_is_integer_element_v<To> &&
      sizeof(From) == sizeof(To) * 2 && num_words(from) == 2);
  const auto lower_value = sve_basic_raw_word(lower(from, value));
  const auto upper_value = sve_basic_raw_word(upper(from, value));
  // QXTNB/QXTNT write the bottom/top narrow half of each wide lane, so using
  // lower_value and upper_value directly would interleave the two logical words. Unzip
  // their even_value/odd_value wide lanes first; narrowing then restores concatenation.
  const auto even_value = [&]() VECOPS_INLINE_LAMBDA {
    if constexpr (sizeof(From) == 2 && std::is_signed_v<From>)
      return svuzp1_s16(lower_value, upper_value);
    else if constexpr (sizeof(From) == 2)
      return svuzp1_u16(lower_value, upper_value);
    else if constexpr (sizeof(From) == 4 && std::is_signed_v<From>)
      return svuzp1_s32(lower_value, upper_value);
    else if constexpr (sizeof(From) == 4)
      return svuzp1_u32(lower_value, upper_value);
    else if constexpr (std::is_signed_v<From>)
      return svuzp1_s64(lower_value, upper_value);
    else
      return svuzp1_u64(lower_value, upper_value);
  }();
  const auto odd_value = [&]() VECOPS_INLINE_LAMBDA {
    if constexpr (sizeof(From) == 2 && std::is_signed_v<From>)
      return svuzp2_s16(lower_value, upper_value);
    else if constexpr (sizeof(From) == 2)
      return svuzp2_u16(lower_value, upper_value);
    else if constexpr (sizeof(From) == 4 && std::is_signed_v<From>)
      return svuzp2_s32(lower_value, upper_value);
    else if constexpr (sizeof(From) == 4)
      return svuzp2_u32(lower_value, upper_value);
    else if constexpr (std::is_signed_v<From>)
      return svuzp2_s64(lower_value, upper_value);
    else
      return svuzp2_u64(lower_value, upper_value);
  }();
  if constexpr (std::is_signed_v<From> && std::is_signed_v<To>) {
    if constexpr (sizeof(From) == 2)
      return svqxtnt_s16(svqxtnb_s16(even_value), odd_value);
    else if constexpr (sizeof(From) == 4)
      return svqxtnt_s32(svqxtnb_s32(even_value), odd_value);
    else
      return svqxtnt_s64(svqxtnb_s64(even_value), odd_value);
  } else if constexpr (std::is_unsigned_v<From> && std::is_unsigned_v<To>) {
    if constexpr (sizeof(From) == 2)
      return svqxtnt_u16(svqxtnb_u16(even_value), odd_value);
    else if constexpr (sizeof(From) == 4)
      return svqxtnt_u32(svqxtnb_u32(even_value), odd_value);
    else
      return svqxtnt_u64(svqxtnb_u64(even_value), odd_value);
  } else if constexpr (std::is_signed_v<From>) {
    if constexpr (sizeof(From) == 2)
      return svqxtunt_s16(svqxtunb_s16(even_value), odd_value);
    else if constexpr (sizeof(From) == 4)
      return svqxtunt_s32(svqxtunb_s32(even_value), odd_value);
    else
      return svqxtunt_s64(svqxtunb_s64(even_value), odd_value);
  } else {
    constexpr From high = static_cast<From>(std::numeric_limits<To>::max());
    const auto lo = [&]() VECOPS_INLINE_LAMBDA {
      if constexpr (sizeof(From) == 2)
        return svmin_n_u16_x(svptrue_b16(), even_value, high);
      else if constexpr (sizeof(From) == 4)
        return svmin_n_u32_x(svptrue_b32(), even_value, high);
      else
        return svmin_n_u64_x(svptrue_b64(), even_value, high);
    }();
    const auto hi = [&]() VECOPS_INLINE_LAMBDA {
      if constexpr (sizeof(From) == 2)
        return svmin_n_u16_x(svptrue_b16(), odd_value, high);
      else if constexpr (sizeof(From) == 4)
        return svmin_n_u32_x(svptrue_b32(), odd_value, high);
      else
        return svmin_n_u64_x(svptrue_b64(), odd_value, high);
    }();
    if constexpr (sizeof(From) == 2)
      return svreinterpret_s8_u8(svqxtnt_u16(svqxtnb_u16(lo), hi));
    else if constexpr (sizeof(From) == 4)
      return svreinterpret_s16_u16(svqxtnt_u32(svqxtnb_u32(lo), hi));
    else
      return svreinterpret_s32_u32(svqxtnt_u64(svqxtnb_u64(lo), hi));
  }
}
#endif

template <Element To, VectorTag FromTag>
VECOPS_ALWAYS_INLINE void sve_store_convert_ordered_saturating(
    FromTag from, To* pointer, Vec<FromTag> value, Mask<FromTag> mask) {
  using From = ElementOf<FromTag>;
  using ToTag = Rebind<To, FromTag>;
  static_assert(sve_is_conversion_element_v<From>);
  static_assert(sve_is_conversion_element_v<To>);

  if constexpr (!sve_conversion_tag_representable_v<ToTag>) {
    using FromHalf = Half<FromTag>;
    sve_store_convert_ordered_saturating(
        FromHalf{}, pointer, lower(from, value),
        lower(from, mask));
    sve_store_convert_ordered_saturating(
        FromHalf{}, pointer + size(FromHalf{}),
        upper(from, value), upper(from, mask));
  } else {
#if defined(HAS_SVE2)
  if constexpr (
      sve_is_integer_element_v<From> && sve_is_integer_element_v<To> &&
      sizeof(From) == sizeof(To) * 2 && num_words(from) == 2 &&
      num_words(ToTag{}) == 1) {
    const auto packed = sve_pack_narrow_integer_x2<To>(from, value);
    const auto memory_mask = sve_convert_mask(ToTag{}, from, mask);
    sve_store_memory_word(
        memory_mask, pointer, packed, mem::Temporal{});
  } else
#endif
  if constexpr (
      std::same_as<From, float32_t> && std::same_as<To, bfloat16_t> &&
      num_words(from) == 2 && num_words(ToTag{}) == 1) {
    // Two F32 words exactly fill one BF16 word. Keeping this as one packed
    // conversion and one full-width store is important for LayerNorm streams.
    const auto packed = sve_f32_pair_to_bf16(
        sve_basic_raw_word(lower(from, value)),
        sve_basic_raw_word(upper(from, value)));
    const auto memory_mask = sve_convert_mask(ToTag{}, from, mask);
    sve_store_memory_word(
        memory_mask, pointer, packed, mem::Temporal{});
  } else if constexpr (num_words(from) == 1 && num_words(ToTag{}) == 1) {
    const auto valid = sve_prefix_predicate<From>(size(from));
    const auto active = svand_b_z(valid, valid, mask);
    if constexpr (
        sve_is_integer_element_v<From> && sve_is_integer_element_v<To> &&
        sizeof(From) > sizeof(To)) {
      sve_narrow_integer_store_word<To, FromTag>(pointer, value, active);
    } else {
      const auto converted = sve_convert_one_word_raw<To, From>(
          sve_basic_raw_word(value));
      const auto memory_mask = sve_convert_mask(ToTag{}, from, active);
      sve_store_memory_word(
          memory_mask, pointer, converted, mem::Temporal{});
    }
  } else {
    using FromHalf = Half<FromTag>;
    sve_store_convert_ordered_saturating(
        FromHalf{}, pointer, lower(from, value),
        lower(from, mask));
    sve_store_convert_ordered_saturating(
        FromHalf{}, pointer + size(FromHalf{}),
        upper(from, value), upper(from, mask));
  }
  }
}

template <Element To, VectorTag FromTag>
VECOPS_ALWAYS_INLINE void sve_store_convert_ordered_saturating(
    FromTag from, To* pointer, Vec<FromTag> value) {
  using From = ElementOf<FromTag>;
  using ToTag = Rebind<To, FromTag>;
  static_assert(sve_is_conversion_element_v<From>);
  static_assert(sve_is_conversion_element_v<To>);

  if constexpr (!sve_conversion_tag_representable_v<ToTag>) {
    using FromHalf = Half<FromTag>;
    sve_store_convert_ordered_saturating(
        FromHalf{}, pointer, lower(from, value));
    sve_store_convert_ordered_saturating(
        FromHalf{}, pointer + size(FromHalf{}),
        upper(from, value));
  } else {
#if defined(HAS_SVE2)
  if constexpr (
      sve_is_integer_element_v<From> && sve_is_integer_element_v<To> &&
      sizeof(From) == sizeof(To) * 2 && num_words(from) == 2 &&
      num_words(ToTag{}) == 1) {
    const auto packed = sve_pack_narrow_integer_x2<To>(from, value);
    const auto active = sve_prefix_predicate<To>(size(ToTag{}));
    sve_store_memory_word(active, pointer, packed, mem::Temporal{});
  } else
#endif
  if constexpr (
      std::same_as<From, float32_t> && std::same_as<To, bfloat16_t> &&
      num_words(from) == 2 && num_words(ToTag{}) == 1) {
    const auto packed = sve_f32_pair_to_bf16(
        sve_basic_raw_word(lower(from, value)),
        sve_basic_raw_word(upper(from, value)));
    const auto active = sve_prefix_predicate<To>(size(ToTag{}));
    sve_store_memory_word(active, pointer, packed, mem::Temporal{});
  } else if constexpr (num_words(from) == 1 && num_words(ToTag{}) == 1) {
    const auto active = sve_prefix_predicate<From>(size(from));
    if constexpr (
        sve_is_integer_element_v<From> && sve_is_integer_element_v<To> &&
        sizeof(From) > sizeof(To)) {
      sve_narrow_integer_store_word<To, FromTag>(pointer, value, active);
    } else {
      const auto converted = sve_convert_one_word_raw<To, From>(
          sve_basic_raw_word(value));
      const auto memory_active =
          sve_prefix_predicate<To>(size(ToTag{}));
      sve_store_memory_word(
          memory_active, pointer, converted, mem::Temporal{});
    }
  } else {
    using FromHalf = Half<FromTag>;
    sve_store_convert_ordered_saturating(
        FromHalf{}, pointer, lower(from, value));
    sve_store_convert_ordered_saturating(
        FromHalf{}, pointer + size(FromHalf{}),
        upper(from, value));
  }
  }
}

template <VectorTag ToTag>
struct NativeImpl<SVEBackend, LoadConvertOp, ToTag> {
  template <Element From, VectorValue Indices, int Scale,
            typename Temporality>
    requires (
        sve_is_integer_element_v<From> &&
        sve_is_integer_element_v<ElementOf<ToTag>> &&
        sizeof(From) < sizeof(ElementOf<ToTag>) && sizeof(From) < 4)
  static VECOPS_ALWAYS_INLINE Vec<ToTag> call(
      LoadConvertOp, ToTag to, const From* pointer,
      cvt::Ordered, cvt::Saturate,
      opt::Indexed<Indices, Scale> addressing, Temporality temporality) {
    using IndexTag = Rebind<ElementOf<VecToTag<Indices>>, ToTag>;
    constexpr int scale = Scale == 0 ? sizeof(From) : Scale;
    return sve_load_convert_indexed_integer_unmasked<
        scale, ToTag, From, IndexTag>(
        to, pointer, addressing.indices, temporality);
  }

  template <Element From, VectorValue Indices, int Scale,
            typename Temporality>
    requires (
        sve_is_integer_element_v<From> &&
        sve_is_integer_element_v<ElementOf<ToTag>> &&
        sizeof(From) < sizeof(ElementOf<ToTag>) && sizeof(From) < 4)
  static VECOPS_ALWAYS_INLINE Vec<ToTag> call(
      LoadConvertOp, ToTag to, const From* pointer, Mask<ToTag> mask,
      Vec<ToTag> inactive, cvt::Ordered, cvt::Saturate,
      opt::Indexed<Indices, Scale> addressing, Temporality temporality) {
    using IndexTag = Rebind<ElementOf<VecToTag<Indices>>, ToTag>;
    constexpr int scale = Scale == 0 ? sizeof(From) : Scale;
    return sve_load_convert_indexed_integer<scale, ToTag, From, IndexTag>(
        to, pointer, addressing.indices, mask, inactive, temporality);
  }

  template <Element From, typename Alignment, typename Temporality>
    requires IsMemoryAlignmentOption<Alignment>::value &&
             IsMemoryTemporalityOption<Temporality>::value
  static VECOPS_ALWAYS_INLINE Vec<ToTag> call(
      LoadConvertOp, ToTag to, const From* pointer,
      cvt::Ordered, cvt::Saturate, Alignment alignment,
      Temporality temporality) {
    return sve_load_convert_ordered_saturating(to, pointer);
  }

  template <Element From, typename Alignment, typename Temporality>
    requires IsMemoryAlignmentOption<Alignment>::value &&
             IsMemoryTemporalityOption<Temporality>::value
  static VECOPS_ALWAYS_INLINE Vec<ToTag> call(
      LoadConvertOp, ToTag to, const From* pointer,
      Mask<ToTag> mask, Vec<ToTag> inactive,
      cvt::Ordered, cvt::Saturate, Alignment alignment,
      Temporality temporality) {
    return sve_load_convert_ordered_saturating(to, pointer, mask, inactive);
  }
};

template <VectorTag FromTag>
struct NativeImpl<SVEBackend, StoreConvertOp, FromTag> {
  template <Element To, VectorValue Indices, int Scale,
            typename Temporality, typename Packing>
    requires (
        sve_is_integer_element_v<ElementOf<FromTag>> &&
        sve_is_integer_element_v<To> && sizeof(ElementOf<FromTag>) > sizeof(To))
  static VECOPS_ALWAYS_INLINE void call(
      StoreConvertOp, FromTag from, To* pointer, Vec<FromTag> value,
      cvt::Ordered, cvt::Saturate,
      opt::Indexed<Indices, Scale> addressing, Temporality temporality,
      Packing) {
    using IndexTag = Rebind<ElementOf<VecToTag<Indices>>, FromTag>;
    constexpr int scale = Scale == 0 ? sizeof(To) : Scale;
    sve_store_convert_indexed_integer_unmasked<
        scale, To, FromTag, IndexTag>(
        from, pointer, value, addressing.indices, temporality);
  }

  template <Element To, VectorValue Indices, int Scale,
            typename Temporality, typename Packing>
    requires (
        sve_is_integer_element_v<ElementOf<FromTag>> &&
        sve_is_integer_element_v<To> && sizeof(ElementOf<FromTag>) > sizeof(To))
  static VECOPS_ALWAYS_INLINE void call(
      StoreConvertOp, FromTag from, To* pointer, Vec<FromTag> value,
      Mask<FromTag> mask, cvt::Ordered, cvt::Saturate,
      opt::Indexed<Indices, Scale> addressing, Temporality temporality,
      Packing) {
    using IndexTag = Rebind<ElementOf<VecToTag<Indices>>, FromTag>;
    constexpr int scale = Scale == 0 ? sizeof(To) : Scale;
    sve_store_convert_indexed_integer<scale, To, FromTag, IndexTag>(
        from, pointer, value, addressing.indices, mask, temporality);
  }

  template <Element To, typename Alignment, typename Temporality,
            typename Packing>
    requires (
        IsMemoryAlignmentOption<Alignment>::value &&
        IsMemoryTemporalityOption<Temporality>::value &&
        (std::same_as<Packing, mem::Packed> ||
         std::same_as<Packing, mem::Split>))
  static VECOPS_ALWAYS_INLINE void call(
      StoreConvertOp, FromTag from, To* pointer, Vec<FromTag> value,
      cvt::Ordered, cvt::Saturate, Alignment alignment,
      Temporality temporality, Packing packing) {
    sve_store_convert_ordered_saturating(from, pointer, value);
  }

  template <Element To, typename Alignment, typename Temporality,
            typename Packing>
    requires (
        IsMemoryAlignmentOption<Alignment>::value &&
        IsMemoryTemporalityOption<Temporality>::value &&
        (std::same_as<Packing, mem::Packed> ||
         std::same_as<Packing, mem::Split>))
  static VECOPS_ALWAYS_INLINE void call(
      StoreConvertOp, FromTag from, To* pointer, Vec<FromTag> value,
      Mask<FromTag> mask, cvt::Ordered, cvt::Saturate,
      Alignment alignment, Temporality temporality, Packing packing) {
    sve_store_convert_ordered_saturating(from, pointer, value, mask);
  }
};

} // namespace vecops::vec::details

#endif // VECOPS_VEC_DETAILS_SVE_CONVERSION_MEMORY_H
