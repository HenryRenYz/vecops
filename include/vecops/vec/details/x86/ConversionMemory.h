#ifndef VECOPS_VEC_DETAILS_X86_CONVERSION_MEMORY_H
#define VECOPS_VEC_DETAILS_X86_CONVERSION_MEMORY_H

/**
 * @file ConversionMemory.h
 * @brief x86 backend implementations for load_convert and store_convert operations.
 */

#if !defined(ARCH_X86_FAMILY)
#error "This header requires an x86 target"
#endif

#include <limits>
#include <type_traits>

#include "vecops/vec/details/x86/Conversion.h"
#include "vecops/vec/details/x86/Memory.h"

namespace vecops::vec::details {


/* **************************************************************************** */
//                       Load and store with conversion                       //
/* **************************************************************************** */

template <VectorTag ToTag, Element From>
VECOPS_ALWAYS_INLINE Vec<ToTag> x86_load_convert_ordered_saturating(
    ToTag to, const From* pointer) {
  using FromTag = Rebind<From, ToTag>;
  static_assert(is_x86_conversion_element<ElementOf<ToTag>>);
  static_assert(is_x86_conversion_element<From>);
  const auto loaded = execute(
      LoadOp{}, FromTag{}, pointer, mem::unaligned, mem::temporal);
  return x86_convert_vec_native(to, FromTag{}, loaded);
}

template <VectorTag ToTag, Element From>
VECOPS_ALWAYS_INLINE Vec<ToTag> x86_load_convert_ordered_saturating(
    ToTag to, const From* pointer, Mask<ToTag> mask, Vec<ToTag> inactive) {
  using FromTag = Rebind<From, ToTag>;
  static_assert(is_x86_conversion_element<ElementOf<ToTag>>);
  static_assert(is_x86_conversion_element<From>);

  // The mask must be converted before the load. Loading a full source vector
  // and blending after conversion would fault on inactive addresses at a page
  // boundary and would violate the filtered-memory contract.
  const auto memory_mask = x86_convert_mask_native(FromTag{}, to, mask);
  const auto zero = execute(FillOp{}, FromTag{}, From{});
  const auto loaded = execute(
      LoadOp{}, FromTag{}, pointer, memory_mask, zero,
      mem::unaligned, mem::temporal);
  const auto converted = x86_convert_vec_native(to, FromTag{}, loaded);
  return execute(BlendOp{}, to, inactive, mask, converted);
}

#if defined(CPU_CAPABILITY_AVX512)
template <Element To, VectorTag FromTag>
VECOPS_ALWAYS_INLINE bool x86_try_narrow_integer_store_avx512(
    To* pointer, Vec<FromTag> value, Mask<FromTag> mask) {
  using From = ElementOf<FromTag>;
  using Traits = RepresentationTraits<X86Backend, FromTag>;
  if constexpr (
      std::integral<From> && std::integral<To> && sizeof(From) > sizeof(To) &&
      Traits::word_count == 1 && sizeof(typename Traits::RawVec) == 64 &&
      Traits::logical_lanes == Traits::word_lanes) {
    auto bits = value.value;

    // The AVX-512 conversion-store instructions either signed-saturate or
    // truncate. Clamp the mixed-signedness and unsigned cases first so the
    // truncating form has the public saturating semantics.
    if constexpr (std::is_unsigned_v<From>) {
      constexpr From high = static_cast<From>(std::numeric_limits<To>::max());
      if constexpr (sizeof(From) == 2)
        bits = _mm512_min_epu16(bits, _mm512_set1_epi16(static_cast<int16_t>(high)));
      else if constexpr (sizeof(From) == 4)
        bits = _mm512_min_epu32(bits, _mm512_set1_epi32(static_cast<int32_t>(high)));
      else
        bits = _mm512_min_epu64(bits, _mm512_set1_epi64(static_cast<int64_t>(high)));
    } else if constexpr (std::is_unsigned_v<To>) {
      constexpr From high = static_cast<From>(std::numeric_limits<To>::max());
      if constexpr (sizeof(From) == 2) {
        bits = _mm512_max_epi16(bits, _mm512_setzero_si512());
        bits = _mm512_min_epi16(bits, _mm512_set1_epi16(static_cast<int16_t>(high)));
      } else if constexpr (sizeof(From) == 4) {
        bits = _mm512_max_epi32(bits, _mm512_setzero_si512());
        bits = _mm512_min_epi32(bits, _mm512_set1_epi32(static_cast<int32_t>(high)));
      } else {
        bits = _mm512_max_epi64(bits, _mm512_setzero_si512());
        bits = _mm512_min_epi64(bits, _mm512_set1_epi64(static_cast<int64_t>(high)));
      }
    }

    if constexpr (sizeof(From) == 2) {
      if constexpr (std::is_signed_v<From> && std::is_signed_v<To>)
        _mm512_mask_cvtsepi16_storeu_epi8(pointer, mask.value, bits);
      else
        _mm512_mask_cvtepi16_storeu_epi8(pointer, mask.value, bits);
    } else if constexpr (sizeof(From) == 4 && sizeof(To) == 1) {
      if constexpr (std::is_signed_v<From> && std::is_signed_v<To>)
        _mm512_mask_cvtsepi32_storeu_epi8(pointer, mask.value, bits);
      else
        _mm512_mask_cvtepi32_storeu_epi8(pointer, mask.value, bits);
    } else if constexpr (sizeof(From) == 4) {
      if constexpr (std::is_signed_v<From> && std::is_signed_v<To>)
        _mm512_mask_cvtsepi32_storeu_epi16(pointer, mask.value, bits);
      else
        _mm512_mask_cvtepi32_storeu_epi16(pointer, mask.value, bits);
    } else if constexpr (sizeof(To) == 1) {
      if constexpr (std::is_signed_v<From> && std::is_signed_v<To>)
        _mm512_mask_cvtsepi64_storeu_epi8(pointer, mask.value, bits);
      else
        _mm512_mask_cvtepi64_storeu_epi8(pointer, mask.value, bits);
    } else if constexpr (sizeof(To) == 2) {
      if constexpr (std::is_signed_v<From> && std::is_signed_v<To>)
        _mm512_mask_cvtsepi64_storeu_epi16(pointer, mask.value, bits);
      else
        _mm512_mask_cvtepi64_storeu_epi16(pointer, mask.value, bits);
    } else {
      if constexpr (std::is_signed_v<From> && std::is_signed_v<To>)
        _mm512_mask_cvtsepi64_storeu_epi32(pointer, mask.value, bits);
      else
        _mm512_mask_cvtepi64_storeu_epi32(pointer, mask.value, bits);
    }
    return true;
  } else {
    return false;
  }
}
#endif

template <Element To, VectorTag FromTag>
VECOPS_ALWAYS_INLINE void x86_store_convert_ordered_saturating_packed(
    FromTag from, To* pointer, Vec<FromTag> value, Mask<FromTag> mask) {
  using ToTag = Rebind<To, FromTag>;
  static_assert(is_x86_conversion_element<To>);
  static_assert(is_x86_conversion_element<ElementOf<FromTag>>);

#if defined(CPU_CAPABILITY_AVX512)
  if (x86_try_narrow_integer_store_avx512<To, FromTag>(
          pointer, value, mask)) return;
#endif
  const auto converted = x86_convert_vec_native(ToTag{}, from, value);
  const auto memory_mask = x86_convert_mask_native(ToTag{}, from, mask);
  execute(
      StoreOp{}, ToTag{}, pointer, converted, memory_mask,
      mem::unaligned, mem::temporal);
}

template <Element To, VectorTag FromTag>
VECOPS_ALWAYS_INLINE void x86_store_convert_ordered_saturating_packed(
    FromTag from, To* pointer, Vec<FromTag> value) {
  const auto mask = execute(MaskFillOp{}, from, true);
  x86_store_convert_ordered_saturating_packed(
      from, pointer, value, mask);
}

template <Element To, VectorTag FromTag>
VECOPS_ALWAYS_INLINE void x86_store_convert_ordered_saturating_split(
    FromTag from, To* pointer, Vec<FromTag> value, Mask<FromTag> mask) {
  using ToTag = Rebind<To, FromTag>;
  using FromTraits = RepresentationTraits<X86Backend, FromTag>;
  using ToTraits = RepresentationTraits<X86Backend, ToTag>;
  constexpr bool narrowing = sizeof(To) < sizeof(ElementOf<FromTag>);
  if constexpr (
      (narrowing && ToTraits::word_count == 1) ||
      (FromTraits::word_count == 1 && ToTraits::word_count == 1)) {
    // A multiword narrowing result that fits one destination word must be
    // packed before storing. Recursing per source word would turn one wide
    // write into several partial writes and defeat write combining.
    x86_store_convert_ordered_saturating_packed(
        from, pointer, value, mask);
  } else {
    using FromHalf = Half<FromTag>;
    const auto lower_value = execute(LowerOp{}, from, value);
    const auto upper_value = execute(UpperOp{}, from, value);
    const auto lower_mask = x86_conversion_mask_lower(from, mask);
    const auto upper_mask = x86_conversion_mask_upper(from, mask);
    x86_store_convert_ordered_saturating_split(
        FromHalf{}, pointer, lower_value, lower_mask);
    x86_store_convert_ordered_saturating_split(
        FromHalf{}, pointer + size(FromHalf{}), upper_value, upper_mask);
  }
}

template <Element To, VectorTag FromTag>
VECOPS_ALWAYS_INLINE void x86_store_convert_ordered_saturating_split(
    FromTag from, To* pointer, Vec<FromTag> value) {
  const auto mask = execute(MaskFillOp{}, from, true);
  x86_store_convert_ordered_saturating_split(from, pointer, value, mask);
}

template <VectorTag ToTag>
struct NativeImpl<X86Backend, LoadConvertOp, ToTag> {
  template <Element From, typename Layout, typename ValuePolicy,
            typename Alignment, typename Temporality>
    requires IsMemoryAlignmentOption<Alignment>::value &&
             IsMemoryTemporalityOption<Temporality>::value
  static VECOPS_ALWAYS_INLINE Vec<ToTag> call(
      LoadConvertOp, ToTag to, const From* pointer,
      Layout layout, ValuePolicy value_policy,
      Alignment alignment, Temporality temporality) {
    if constexpr (std::same_as<Layout, cvt::Ordered> &&
                  std::same_as<ValuePolicy, cvt::Saturate>) {
      return x86_load_convert_ordered_saturating(to, pointer);
    } else {
      using FromTag = Rebind<From, ToTag>;
      const auto loaded = execute_load_options(
          LoadOp{}, FromTag{}, pointer, alignment, temporality);
      return execute(
          ConvertOp{}, to, FromTag{}, loaded, layout, value_policy);
    }
  }

  template <Element From, typename ValuePolicy, typename Alignment,
            typename Temporality>
    requires IsMemoryAlignmentOption<Alignment>::value &&
             IsMemoryTemporalityOption<Temporality>::value
  static VECOPS_ALWAYS_INLINE Vec<ToTag> call(
      LoadConvertOp, ToTag to, const From* pointer,
      Mask<ToTag> mask, Vec<ToTag> inactive,
      cvt::Ordered layout, ValuePolicy value_policy,
      Alignment alignment, Temporality temporality) {
    if constexpr (std::same_as<ValuePolicy, cvt::Saturate>) {
      return x86_load_convert_ordered_saturating(
          to, pointer, mask, inactive);
    } else {
      using FromTag = Rebind<From, ToTag>;
      const auto memory_mask = x86_convert_mask_native(
          FromTag{}, to, mask);
      const auto loaded = execute_load_options(
          LoadOp{}, FromTag{}, pointer, opt::masked(memory_mask),
          opt::zero, alignment, temporality);
      const auto converted = execute(
          ConvertOp{}, to, FromTag{}, loaded, layout, value_policy);
      return execute(BlendOp{}, to, inactive, mask, converted);
    }
  }

  template <Element From, typename ValuePolicy, typename Alignment,
            typename Temporality>
    requires IsMemoryAlignmentOption<Alignment>::value &&
             IsMemoryTemporalityOption<Temporality>::value
  static VECOPS_ALWAYS_INLINE Vec<ToTag> call(
      LoadConvertOp, ToTag to, const From* pointer,
      Mask<Rebind<From, ToTag>> memory_mask,
      cvt::Unordered layout, ValuePolicy value_policy,
      Alignment alignment, Temporality temporality) {
    using FromTag = Rebind<From, ToTag>;
    const auto loaded = execute_load_options(
        LoadOp{}, FromTag{}, pointer, opt::masked(memory_mask),
        opt::zero, alignment, temporality);
    return execute(
        ConvertOp{}, to, FromTag{}, loaded, layout, value_policy);
  }
};

template <VectorTag FromTag>
struct NativeImpl<X86Backend, StoreConvertOp, FromTag> {
  template <Element To, typename Layout, typename ValuePolicy,
            typename Alignment, typename Temporality, typename Packing>
    requires IsMemoryAlignmentOption<Alignment>::value &&
             IsMemoryTemporalityOption<Temporality>::value &&
             (std::same_as<Packing, mem::Packed> ||
              std::same_as<Packing, mem::Split>)
  static VECOPS_ALWAYS_INLINE void call(
      StoreConvertOp, FromTag from, To* pointer, Vec<FromTag> value,
      Layout layout, ValuePolicy value_policy, Alignment alignment,
      Temporality temporality, Packing packing) {
    if constexpr (std::same_as<Layout, cvt::Ordered> &&
                  std::same_as<ValuePolicy, cvt::Saturate>) {
      if constexpr (std::same_as<Packing, mem::Packed>)
        x86_store_convert_ordered_saturating_packed(from, pointer, value);
      else
        x86_store_convert_ordered_saturating_split(from, pointer, value);
    } else {
      using ToTag = Rebind<To, FromTag>;
      const auto converted = execute(
          ConvertOp{}, ToTag{}, from, value, layout, value_policy);
      execute_store_options(
          StoreOp{}, ToTag{}, pointer, converted, alignment, temporality);
      (void)packing;
    }
  }

  template <Element To, typename ValuePolicy, typename Alignment,
            typename Temporality, typename Packing>
    requires IsMemoryAlignmentOption<Alignment>::value &&
             IsMemoryTemporalityOption<Temporality>::value &&
             (std::same_as<Packing, mem::Packed> ||
              std::same_as<Packing, mem::Split>)
  static VECOPS_ALWAYS_INLINE void call(
      StoreConvertOp, FromTag from, To* pointer, Vec<FromTag> value,
      Mask<FromTag> mask, cvt::Ordered layout, ValuePolicy value_policy,
      Alignment alignment, Temporality temporality, Packing packing) {
    if constexpr (std::same_as<ValuePolicy, cvt::Saturate>) {
      if constexpr (std::same_as<Packing, mem::Packed>)
        x86_store_convert_ordered_saturating_packed(
            from, pointer, value, mask);
      else
        x86_store_convert_ordered_saturating_split(
            from, pointer, value, mask);
    } else {
      using ToTag = Rebind<To, FromTag>;
      const auto converted = execute(
          ConvertOp{}, ToTag{}, from, value, layout, value_policy);
      const auto memory_mask = x86_convert_mask_native(
          ToTag{}, from, mask);
      execute_store_options(
          StoreOp{}, ToTag{}, pointer, converted, opt::masked(memory_mask),
          alignment, temporality);
      (void)packing;
    }
  }

  template <Element To, typename ValuePolicy, typename Alignment,
            typename Temporality, typename Packing>
    requires IsMemoryAlignmentOption<Alignment>::value &&
             IsMemoryTemporalityOption<Temporality>::value &&
             std::same_as<Packing, mem::Packed>
  static VECOPS_ALWAYS_INLINE void call(
      StoreConvertOp, FromTag from, To* pointer, Vec<FromTag> value,
      Mask<Rebind<To, FromTag>> memory_mask, cvt::Unordered layout,
      ValuePolicy value_policy, Alignment alignment,
      Temporality temporality, Packing packing) {
    using ToTag = Rebind<To, FromTag>;
    const auto converted = execute(
        ConvertOp{}, ToTag{}, from, value, layout, value_policy);
    execute_store_options(
        StoreOp{}, ToTag{}, pointer, converted, opt::masked(memory_mask),
        alignment, temporality);
    (void)packing;
  }
};

} // namespace vecops::vec::details

#endif // VECOPS_VEC_DETAILS_X86_CONVERSION_MEMORY_H
