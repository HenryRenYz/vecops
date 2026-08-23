#ifndef VECOPS_VEC_DETAILS_X86_CONVERSION_H
#define VECOPS_VEC_DETAILS_X86_CONVERSION_H

/**
 * @file Conversion.h
 * @brief x86 backend implementations for element type conversion operations.
 */

#if !defined(ARCH_X86_FAMILY)
#error "This header requires an x86 target"
#endif

#include <type_traits>
#include <cstring>

#include "vecops/vec/details/Conversion.h"
#include "vecops/vec/details/x86/Basic.h"
#include "vecops/util/ScalarConvert.h"

namespace vecops::vec::details {


/* **************************************************************************** */
//                  Element type conversion implementations                   //
/* **************************************************************************** */

template <typename T>
inline constexpr bool is_x86_conversion_element_v =
    std::same_as<T, bfloat16_t> || std::same_as<T, float16_t> ||
    std::same_as<T, float32_t> || std::same_as<T, float64_t> ||
    std::same_as<T, int8_t> || std::same_as<T, uint8_t> ||
    std::same_as<T, int16_t> || std::same_as<T, uint16_t> ||
    std::same_as<T, int32_t> || std::same_as<T, uint32_t> ||
    std::same_as<T, int64_t> || std::same_as<T, uint64_t>;

#if defined(HAS_AVX512_BF16) && !defined(VECOPS_PRESERVE_SUBNORMALS)
template <typename Integer, typename BFloat>
VECOPS_ALWAYS_INLINE Integer x86_conversion_bfloat_bits(BFloat value) {
  static_assert(sizeof(Integer) == sizeof(BFloat));
  union {
    BFloat bfloat;
    Integer integer;
  } cast{.bfloat = value};
  return cast.integer;
}
#endif

// Conversion changes mask granularity, so ordinary word batching is invalid:
// one output word can consume a fraction of an input word or several words.
// These helpers split by the complete logical Tag without going through the
// public CPO. Unlike Basic's general rearrangement fallback, single-word mask
// halves stay entirely in registers.
template <VectorTag Tag>
VECOPS_ALWAYS_INLINE Mask<Half<Tag>> x86_conversion_mask_lower(
    Tag, Mask<Tag> value) {
  using InTraits = RepresentationTraits<X86Backend, Tag>;
  using OutTag = Half<Tag>;
  using OutTraits = RepresentationTraits<X86Backend, OutTag>;

  if constexpr (InTraits::word_count > 1) {
    return construct_mask_words<X86Backend>(
        OutTag{}, [&]<nint_t Index>(OutTag) {
          return ::vecops::vec::get_word<Index>(Tag{}, value);
        });
  } else {
    using InRaw = typename InTraits::RawMask;
    using OutRaw = typename OutTraits::RawMask;
#if defined(CPU_CAPABILITY_AVX512)
    return Mask<OutTag>{static_cast<OutRaw>(value.value)};
#else
    if constexpr (sizeof(InRaw) == sizeof(OutRaw)) {
      return Mask<OutTag>{value.value};
    } else {
      static_assert(sizeof(InRaw) == 32 && sizeof(OutRaw) == 16);
      return Mask<OutTag>{_mm256_castsi256_si128(value.value)};
    }
#endif
  }
}

template <VectorTag Tag>
VECOPS_ALWAYS_INLINE Mask<Half<Tag>> x86_conversion_mask_upper(
    Tag, Mask<Tag> value) {
  using InTraits = RepresentationTraits<X86Backend, Tag>;
  using OutTag = Half<Tag>;
  using OutTraits = RepresentationTraits<X86Backend, OutTag>;

  if constexpr (InTraits::word_count > 1) {
    constexpr nint_t offset = OutTraits::word_count;
    return construct_mask_words<X86Backend>(
        OutTag{}, [&]<nint_t Index>(OutTag) {
          return ::vecops::vec::get_word<Index + offset>(Tag{}, value);
        });
  } else {
    using InRaw = typename InTraits::RawMask;
    using OutRaw = typename OutTraits::RawMask;
    constexpr int half_lanes = static_cast<int>(OutTraits::logical_lanes);
#if defined(CPU_CAPABILITY_AVX512)
    return Mask<OutTag>{
        static_cast<OutRaw>(static_cast<uint64_t>(value.value) >> half_lanes)};
#else
    constexpr int half_bytes =
        half_lanes * static_cast<int>(sizeof(ElementOf<Tag>));
    if constexpr (sizeof(InRaw) == sizeof(OutRaw)) {
      return Mask<OutTag>{_mm_srli_si128(value.value, half_bytes)};
    } else {
      static_assert(sizeof(InRaw) == 32 && sizeof(OutRaw) == 16);
      return Mask<OutTag>{_mm256_extracti128_si256(value.value, 1)};
    }
#endif
  }
}

template <VectorTag Tag>
VECOPS_ALWAYS_INLINE Mask<Tag> x86_conversion_mask_concat(
    Tag, Mask<Half<Tag>> lower, Mask<Half<Tag>> upper) {
  using OutTraits = RepresentationTraits<X86Backend, Tag>;
  using InTag = Half<Tag>;
  using InTraits = RepresentationTraits<X86Backend, InTag>;

  if constexpr (OutTraits::word_count > 1) {
    constexpr nint_t split = InTraits::word_count;
    return construct_mask_words<X86Backend>(
        Tag{}, [&]<nint_t Index>(Tag) {
          if constexpr (Index < split) {
            return ::vecops::vec::get_word<Index>(InTag{}, lower);
          } else {
            return ::vecops::vec::get_word<Index - split>(InTag{}, upper);
          }
        });
  } else {
    using OutRaw = typename OutTraits::RawMask;
    using InRaw = typename InTraits::RawMask;
    constexpr int half_lanes = static_cast<int>(InTraits::logical_lanes);
#if defined(CPU_CAPABILITY_AVX512)
    const uint64_t bits = static_cast<uint64_t>(lower.value) |
        (static_cast<uint64_t>(upper.value) << half_lanes);
    return Mask<Tag>{static_cast<OutRaw>(bits)};
#else
    if constexpr (sizeof(OutRaw) == sizeof(InRaw)) {
      constexpr int half_bytes =
          half_lanes * static_cast<int>(sizeof(ElementOf<Tag>));
      const auto kept_lower = _mm_srli_si128(
          _mm_slli_si128(lower.value, 16 - half_bytes), 16 - half_bytes);
      return Mask<Tag>{
          _mm_or_si128(kept_lower, _mm_slli_si128(upper.value, half_bytes))};
    } else {
      static_assert(sizeof(OutRaw) == 32 && sizeof(InRaw) == 16);
      return Mask<Tag>{_mm256_inserti128_si256(
          _mm256_castsi128_si256(lower.value), upper.value, 1)};
    }
#endif
  }
}

template <VectorTag ToTag, VectorTag FromTag>
VECOPS_ALWAYS_INLINE Mask<ToTag> x86_convert_mask_single_word(
    ToTag, FromTag, Mask<FromTag> value) {
  using ToTraits = RepresentationTraits<X86Backend, ToTag>;
  using FromTraits = RepresentationTraits<X86Backend, FromTag>;
  using ToRaw = typename ToTraits::RawMask;
  using FromRaw = typename FromTraits::RawMask;
  constexpr nint_t to_bytes = sizeof(ElementOf<ToTag>);
  constexpr nint_t from_bytes = sizeof(ElementOf<FromTag>);

  static_assert(ToTraits::word_count == 1 && FromTraits::word_count == 1);

#if defined(CPU_CAPABILITY_AVX512)
  return Mask<ToTag>{static_cast<ToRaw>(value.value)};
#else
  if constexpr (to_bytes == from_bytes) {
    static_assert(sizeof(ToRaw) == sizeof(FromRaw));
    return Mask<ToTag>{value.value};
  } else if constexpr (to_bytes == 2 && from_bytes == 1) {
    if constexpr (sizeof(ToRaw) == 16)
      return Mask<ToTag>{_mm_cvtepi8_epi16(value.value)};
    else
      return Mask<ToTag>{_mm256_cvtepi8_epi16(value.value)};
  } else if constexpr (to_bytes == 4 && from_bytes == 1) {
    if constexpr (sizeof(ToRaw) == 16)
      return Mask<ToTag>{_mm_cvtepi8_epi32(value.value)};
    else
      return Mask<ToTag>{_mm256_cvtepi8_epi32(value.value)};
  } else if constexpr (to_bytes == 8 && from_bytes == 1) {
    if constexpr (sizeof(ToRaw) == 16)
      return Mask<ToTag>{_mm_cvtepi8_epi64(value.value)};
    else
      return Mask<ToTag>{_mm256_cvtepi8_epi64(value.value)};
  } else if constexpr (to_bytes == 4 && from_bytes == 2) {
    if constexpr (sizeof(ToRaw) == 16)
      return Mask<ToTag>{_mm_cvtepi16_epi32(value.value)};
    else
      return Mask<ToTag>{_mm256_cvtepi16_epi32(value.value)};
  } else if constexpr (to_bytes == 8 && from_bytes == 2) {
    if constexpr (sizeof(ToRaw) == 16)
      return Mask<ToTag>{_mm_cvtepi16_epi64(value.value)};
    else
      return Mask<ToTag>{_mm256_cvtepi16_epi64(value.value)};
  } else if constexpr (to_bytes == 8 && from_bytes == 4) {
    if constexpr (sizeof(ToRaw) == 16)
      return Mask<ToTag>{_mm_cvtepi32_epi64(value.value)};
    else
      return Mask<ToTag>{_mm256_cvtepi32_epi64(value.value)};
  } else if constexpr (to_bytes == 1 && from_bytes == 2) {
    if constexpr (sizeof(FromRaw) == 16) {
      return Mask<ToTag>{_mm_packs_epi16(value.value, _mm_setzero_si128())};
    } else {
      const auto lo = _mm_packs_epi16(
          _mm256_castsi256_si128(value.value), _mm_setzero_si128());
      const auto hi = _mm_packs_epi16(
          _mm256_extracti128_si256(value.value, 1), _mm_setzero_si128());
      return Mask<ToTag>{_mm_unpacklo_epi64(lo, hi)};
    }
  } else if constexpr (to_bytes == 2 && from_bytes == 4) {
    if constexpr (sizeof(FromRaw) == 16) {
      return Mask<ToTag>{_mm_packs_epi32(value.value, _mm_setzero_si128())};
    } else {
      const auto lo = _mm_packs_epi32(
          _mm256_castsi256_si128(value.value), _mm_setzero_si128());
      const auto hi = _mm_packs_epi32(
          _mm256_extracti128_si256(value.value, 1), _mm_setzero_si128());
      return Mask<ToTag>{_mm_unpacklo_epi64(lo, hi)};
    }
  } else if constexpr (to_bytes == 4 && from_bytes == 8) {
    if constexpr (sizeof(FromRaw) == 16) {
      const auto shifted = _mm_srli_epi64(value.value, 32);
      return Mask<ToTag>{
          _mm_shuffle_epi32(shifted, _MM_SHUFFLE(0, 0, 2, 0))};
    } else {
      const auto shifted = _mm256_srli_epi64(value.value, 32);
      const auto packed = _mm256_permutevar8x32_epi32(
          shifted, _mm256_setr_epi32(0, 2, 4, 6, 0, 0, 0, 0));
      return Mask<ToTag>{_mm256_castsi256_si128(packed)};
    }
  } else if constexpr (to_bytes == 1 && from_bytes == 4) {
    using MidTag = Rebind<int16_t, ToTag>;
    return x86_convert_mask_single_word(
        ToTag{}, MidTag{},
        x86_convert_mask_single_word(MidTag{}, FromTag{}, value));
  } else if constexpr (to_bytes == 1 && from_bytes == 8) {
    using MidTag = Rebind<int32_t, ToTag>;
    return x86_convert_mask_single_word(
        ToTag{}, MidTag{},
        x86_convert_mask_single_word(MidTag{}, FromTag{}, value));
  } else if constexpr (to_bytes == 2 && from_bytes == 8) {
    using MidTag = Rebind<int32_t, ToTag>;
    return x86_convert_mask_single_word(
        ToTag{}, MidTag{},
        x86_convert_mask_single_word(MidTag{}, FromTag{}, value));
  } else {
    static_assert(
        dispatch_dependent_false<ToTag, FromTag>,
        "unsupported x86 mask conversion element widths");
  }
#endif
}

template <VectorTag ToTag, VectorTag FromTag>
VECOPS_ALWAYS_INLINE Mask<ToTag> x86_convert_mask_native(
    ToTag to, FromTag from, Mask<FromTag> value) {
  using ToTraits = RepresentationTraits<X86Backend, ToTag>;
  using FromTraits = RepresentationTraits<X86Backend, FromTag>;
  static_assert(
      is_x86_conversion_element_v<ElementOf<ToTag>>,
      "unsupported x86 mask conversion destination element type");
  static_assert(
      is_x86_conversion_element_v<ElementOf<FromTag>>,
      "unsupported x86 mask conversion source element type");

  if constexpr (ToTraits::word_count == 1 && FromTraits::word_count == 1) {
    auto converted = x86_convert_mask_single_word(to, from, value);
    const auto valid = execute(MaskFillOp{}, to, true);
    converted.value = x86_mask_and(converted.value, valid.value);
    return converted;
  } else {
    using ToHalf = Half<ToTag>;
    using FromHalf = Half<FromTag>;
    const auto lower = x86_convert_mask_native(
        ToHalf{}, FromHalf{}, x86_conversion_mask_lower(from, value));
    const auto upper = x86_convert_mask_native(
        ToHalf{}, FromHalf{}, x86_conversion_mask_upper(from, value));
    return x86_conversion_mask_concat(to, lower, upper);
  }
}

template <bool Wrap = false, VectorTag ToTag, VectorTag FromTag>
VECOPS_ALWAYS_INLINE Vec<ToTag> x86_convert_vec_scalar_word(
    ToTag, FromTag, Vec<FromTag> value) {
  using ToTraits = RepresentationTraits<X86Backend, ToTag>;
  using FromTraits = RepresentationTraits<X86Backend, FromTag>;
  using To = ElementOf<ToTag>;
  using From = ElementOf<FromTag>;
  constexpr std::size_t from_lanes =
      sizeof(typename FromTraits::RawVec) / sizeof(From);
  constexpr std::size_t to_lanes =
      sizeof(typename ToTraits::RawVec) / sizeof(To);
  alignas(64) From input[from_lanes];
  alignas(64) To output[to_lanes]{};
  std::memcpy(input, &value.value, sizeof(value.value));
  for (nint_t lane = 0; lane < ToTraits::logical_lanes; ++lane) {
    if constexpr (Wrap) {
      output[static_cast<std::size_t>(lane)] =
          ::vecops::wrap_convert<To>(input[static_cast<std::size_t>(lane)]);
    } else {
      output[static_cast<std::size_t>(lane)] =
          ::vecops::convert<To, From>(input[static_cast<std::size_t>(lane)]);
    }
  }
  Vec<ToTag> result{};
  std::memcpy(&result.value, output, sizeof(result.value));
  return result;
}

template <VectorTag ToTag, VectorTag FromTag>
VECOPS_ALWAYS_INLINE Vec<ToTag> x86_narrow_integer_vec_native(
    ToTag to, FromTag from, Vec<FromTag> value) {
  using To = ElementOf<ToTag>;
  using From = ElementOf<FromTag>;
  using ToTraits = RepresentationTraits<X86Backend, ToTag>;
  using FromTraits = RepresentationTraits<X86Backend, FromTag>;
  static_assert(std::integral<To> && std::integral<From>);
  static_assert(sizeof(To) < sizeof(From));

  if constexpr (sizeof(From) > 2 * sizeof(To)) {
    using Mid = std::conditional_t<
        std::is_signed_v<To>,
        std::conditional_t<sizeof(From) == 8, int32_t, int16_t>,
        std::conditional_t<sizeof(From) == 8, uint32_t, uint16_t>>;
    using MidTag = Rebind<Mid, ToTag>;
    return x86_narrow_integer_vec_native(
        to, MidTag{},
        x86_narrow_integer_vec_native(MidTag{}, from, value));
  } else if constexpr (sizeof(From) == 8) {
    // SSE/AVX have no saturating 64-to-32 pack; the legacy backend also uses
    // a scalar conversion for this base case when AVX-512 narrowing is absent.
    return x86_convert_vec_scalar_word(to, from, value);
  } else if constexpr (
      FromTraits::logical_lanes > 16 / static_cast<nint_t>(sizeof(From))) {
    using ToHalf = Half<ToTag>;
    using FromHalf = Half<FromTag>;
    const auto lower = x86_narrow_integer_vec_native(
        ToHalf{}, FromHalf{}, execute(LowerOp{}, from, value));
    const auto upper = x86_narrow_integer_vec_native(
        ToHalf{}, FromHalf{}, execute(UpperOp{}, from, value));
    return execute(ConcatOp{}, to, lower, upper);
  } else {
    static_assert(sizeof(typename FromTraits::RawVec) == 16);
    static_assert(sizeof(typename ToTraits::RawVec) == 16);
    if constexpr (sizeof(From) == 2) {
      if constexpr (std::is_signed_v<From> && std::is_signed_v<To>) {
        return Vec<ToTag>{_mm_packs_epi16(value.value, value.value)};
      } else if constexpr (std::is_signed_v<From> && !std::is_signed_v<To>) {
        return Vec<ToTag>{_mm_packus_epi16(value.value, value.value)};
      } else {
        const auto limited = _mm_min_epu16(
            value.value, _mm_set1_epi16(std::numeric_limits<To>::max()));
        if constexpr (std::is_signed_v<To>)
          return Vec<ToTag>{_mm_packs_epi16(limited, limited)};
        else
          return Vec<ToTag>{_mm_packus_epi16(limited, limited)};
      }
    } else {
      if constexpr (std::is_signed_v<From> && std::is_signed_v<To>) {
        return Vec<ToTag>{_mm_packs_epi32(value.value, value.value)};
      } else if constexpr (std::is_signed_v<From> && !std::is_signed_v<To>) {
        return Vec<ToTag>{_mm_packus_epi32(value.value, value.value)};
      } else {
        const auto limited = _mm_min_epu32(
            value.value, _mm_set1_epi32(std::numeric_limits<To>::max()));
        if constexpr (std::is_signed_v<To>)
          return Vec<ToTag>{_mm_packs_epi32(limited, limited)};
        else
          return Vec<ToTag>{_mm_packus_epi32(limited, limited)};
      }
    }
  }
}

template <VectorTag ToTag, VectorTag FromTag>
VECOPS_ALWAYS_INLINE Vec<ToTag> x86_convert_vec_native(
    ToTag to, FromTag from, Vec<FromTag> value);

template <VectorTag ToTag, VectorTag FromTag>
VECOPS_ALWAYS_INLINE Vec<ToTag> x86_convert_vec_wrap_native(
    ToTag to, FromTag from, Vec<FromTag> value) {
  using ToTraits = RepresentationTraits<X86Backend, ToTag>;
  using FromTraits = RepresentationTraits<X86Backend, FromTag>;
  static_assert(
      std::integral<ElementOf<ToTag>> && std::integral<ElementOf<FromTag>> &&
      sizeof(ElementOf<ToTag>) < sizeof(ElementOf<FromTag>));
  if constexpr (ToTraits::word_count == 1 && FromTraits::word_count == 1) {
    // The x86 pack instructions saturate, so modulo narrowing cannot use
    // them without first rearranging low bytes. Keep this base case explicit
    // and scalar-correct; wider logical vectors still split in registers.
    return x86_convert_vec_scalar_word<true>(to, from, value);
  } else {
    using ToHalf = Half<ToTag>;
    using FromHalf = Half<FromTag>;
    const auto lower = x86_convert_vec_wrap_native(
        ToHalf{}, FromHalf{}, execute(LowerOp{}, from, value));
    const auto upper = x86_convert_vec_wrap_native(
        ToHalf{}, FromHalf{}, execute(UpperOp{}, from, value));
    return execute(ConcatOp{}, to, lower, upper);
  }
}

template <VectorTag ToTag, VectorTag FromTag>
VECOPS_ALWAYS_INLINE Vec<ToTag> x86_convert_vec_single_word(
    ToTag to, FromTag from, Vec<FromTag> value) {
  using ToTraits = RepresentationTraits<X86Backend, ToTag>;
  using FromTraits = RepresentationTraits<X86Backend, FromTag>;
  using ToRaw = typename ToTraits::RawVec;
  using FromRaw = typename FromTraits::RawVec;
  using To = ElementOf<ToTag>;
  using From = ElementOf<FromTag>;
  static_assert(ToTraits::word_count == 1 && FromTraits::word_count == 1);

  if constexpr (std::same_as<To, From>) {
    static_assert(std::same_as<ToRaw, FromRaw>);
    return Vec<ToTag>{value.value};
  } else if constexpr (std::same_as<To, float32_t> &&
                       std::same_as<From, int32_t>) {
    if constexpr (sizeof(ToRaw) == 16)
      return Vec<ToTag>{_mm_cvtepi32_ps(value.value)};
#if VEC_WIDTH >= 256
    else if constexpr (sizeof(ToRaw) == 32)
      return Vec<ToTag>{_mm256_cvtepi32_ps(value.value)};
#endif
#if VEC_WIDTH >= 512
    else return Vec<ToTag>{_mm512_cvtepi32_ps(value.value)};
#endif
  } else if constexpr (std::same_as<To, int32_t> &&
                       std::same_as<From, float32_t>) {
    if constexpr (sizeof(ToRaw) == 16)
      return Vec<ToTag>{_mm_cvttps_epi32(value.value)};
#if VEC_WIDTH >= 256
    else if constexpr (sizeof(ToRaw) == 32)
      return Vec<ToTag>{_mm256_cvttps_epi32(value.value)};
#endif
#if VEC_WIDTH >= 512
    else return Vec<ToTag>{_mm512_cvttps_epi32(value.value)};
#endif
  } else if constexpr (std::same_as<To, float32_t> &&
                       std::same_as<From, uint32_t>) {
#if defined(CPU_CAPABILITY_AVX512)
    if constexpr (sizeof(ToRaw) == 16)
      return Vec<ToTag>{_mm_cvtepu32_ps(value.value)};
    else if constexpr (sizeof(ToRaw) == 32)
      return Vec<ToTag>{_mm256_cvtepu32_ps(value.value)};
    else
      return Vec<ToTag>{_mm512_cvtepu32_ps(value.value)};
#else
    if constexpr (sizeof(ToRaw) == 16) {
      const auto high = _mm_castsi128_ps(
          _mm_cmplt_epi32(value.value, _mm_setzero_si128()));
      const auto cleared = _mm_and_si128(
          value.value, _mm_set1_epi32(0x7fffffff));
      return Vec<ToTag>{_mm_add_ps(
          _mm_cvtepi32_ps(cleared),
          _mm_and_ps(high, _mm_set1_ps(2147483648.0F)))};
    }
#if VEC_WIDTH >= 256
    else {
      const auto high = _mm256_castsi256_ps(_mm256_cmpgt_epi32(
          _mm256_setzero_si256(), value.value));
      const auto cleared = _mm256_and_si256(
          value.value, _mm256_set1_epi32(0x7fffffff));
      return Vec<ToTag>{_mm256_add_ps(
          _mm256_cvtepi32_ps(cleared),
          _mm256_and_ps(high, _mm256_set1_ps(2147483648.0F)))};
    }
#endif
#endif
  } else if constexpr (std::same_as<To, uint32_t> &&
                       std::same_as<From, float32_t>) {
#if defined(CPU_CAPABILITY_AVX512)
    if constexpr (sizeof(ToRaw) == 16)
      return Vec<ToTag>{_mm_cvttps_epu32(value.value)};
    else if constexpr (sizeof(ToRaw) == 32)
      return Vec<ToTag>{_mm256_cvttps_epu32(value.value)};
    else
      return Vec<ToTag>{_mm512_cvttps_epu32(value.value)};
#else
    if constexpr (sizeof(ToRaw) == 16) {
      const auto two31 = _mm_set1_ps(2147483648.0F);
      const auto high = _mm_cmpge_ps(value.value, two31);
      const auto adjusted = _mm_sub_ps(value.value, _mm_and_ps(high, two31));
      return Vec<ToTag>{_mm_add_epi32(
          _mm_cvttps_epi32(adjusted),
          _mm_and_si128(
              _mm_castps_si128(high), _mm_set1_epi32(0x80000000)))};
    }
#if VEC_WIDTH >= 256
    else {
      const auto two31 = _mm256_set1_ps(2147483648.0F);
      const auto high = _mm256_cmp_ps(value.value, two31, _CMP_GE_OS);
      const auto adjusted =
          _mm256_sub_ps(value.value, _mm256_and_ps(high, two31));
      return Vec<ToTag>{_mm256_add_epi32(
          _mm256_cvttps_epi32(adjusted),
          _mm256_and_si256(
              _mm256_castps_si256(high), _mm256_set1_epi32(0x80000000)))};
    }
#endif
#endif
  } else if constexpr (
      std::same_as<To, float64_t> &&
      (std::same_as<From, int64_t> || std::same_as<From, uint64_t>)) {
#if defined(CPU_CAPABILITY_AVX512)
#define VECOPS_VEC_X86_I64_TO_PD(Prefix)                               \
    if constexpr (sizeof(ToRaw) == 16)                                  \
      return Vec<ToTag>{_mm_cvte##Prefix##64_pd(value.value)};           \
    else if constexpr (sizeof(ToRaw) == 32)                             \
      return Vec<ToTag>{_mm256_cvte##Prefix##64_pd(value.value)};        \
    else                                                                \
      return Vec<ToTag>{_mm512_cvte##Prefix##64_pd(value.value)}
    if constexpr (std::same_as<From, int64_t>) {
      VECOPS_VEC_X86_I64_TO_PD(pi);
    } else {
      VECOPS_VEC_X86_I64_TO_PD(pu);
    }
#undef VECOPS_VEC_X86_I64_TO_PD
#else
    return x86_convert_vec_scalar_word(to, from, value);
#endif
  } else if constexpr (
      (std::same_as<To, int64_t> || std::same_as<To, uint64_t>) &&
      std::same_as<From, float64_t>) {
#if defined(CPU_CAPABILITY_AVX512)
#define VECOPS_VEC_X86_PD_TO_I64(Prefix)                               \
    if constexpr (sizeof(ToRaw) == 16)                                  \
      return Vec<ToTag>{_mm_cvttpd_##Prefix##64(value.value)};           \
    else if constexpr (sizeof(ToRaw) == 32)                             \
      return Vec<ToTag>{_mm256_cvttpd_##Prefix##64(value.value)};        \
    else                                                                \
      return Vec<ToTag>{_mm512_cvttpd_##Prefix##64(value.value)}
    if constexpr (std::same_as<To, int64_t>) {
      VECOPS_VEC_X86_PD_TO_I64(epi);
    } else {
      VECOPS_VEC_X86_PD_TO_I64(epu);
    }
#undef VECOPS_VEC_X86_PD_TO_I64
#else
    return x86_convert_vec_scalar_word(to, from, value);
#endif
  } else if constexpr (std::same_as<To, float32_t> &&
                       std::same_as<From, float64_t>) {
    if constexpr (sizeof(FromRaw) == 16)
      return Vec<ToTag>{_mm_cvtpd_ps(value.value)};
#if VEC_WIDTH >= 256
    else if constexpr (sizeof(FromRaw) == 32)
      return Vec<ToTag>{_mm256_cvtpd_ps(value.value)};
#endif
#if VEC_WIDTH >= 512
    else return Vec<ToTag>{_mm512_cvtpd_ps(value.value)};
#endif
  } else if constexpr (std::same_as<To, float64_t> &&
                       std::same_as<From, float32_t>) {
    if constexpr (sizeof(ToRaw) == 16)
      return Vec<ToTag>{_mm_cvtps_pd(value.value)};
#if VEC_WIDTH >= 256
    else if constexpr (sizeof(ToRaw) == 32)
      return Vec<ToTag>{_mm256_cvtps_pd(value.value)};
#endif
#if VEC_WIDTH >= 512
    else return Vec<ToTag>{_mm512_cvtps_pd(value.value)};
#endif
  } else if constexpr (std::same_as<To, int32_t> &&
                       std::same_as<From, float64_t>) {
    if constexpr (sizeof(FromRaw) == 16)
      return Vec<ToTag>{_mm_cvttpd_epi32(value.value)};
#if VEC_WIDTH >= 256
    else if constexpr (sizeof(FromRaw) == 32)
      return Vec<ToTag>{_mm256_cvttpd_epi32(value.value)};
#endif
#if VEC_WIDTH >= 512
    else return Vec<ToTag>{_mm512_cvttpd_epi32(value.value)};
#endif
  } else if constexpr (std::same_as<To, float64_t> &&
                       std::same_as<From, int32_t>) {
    if constexpr (sizeof(ToRaw) == 16)
      return Vec<ToTag>{_mm_cvtepi32_pd(value.value)};
#if VEC_WIDTH >= 256
    else if constexpr (sizeof(ToRaw) == 32)
      return Vec<ToTag>{_mm256_cvtepi32_pd(value.value)};
#endif
#if VEC_WIDTH >= 512
    else return Vec<ToTag>{_mm512_cvtepi32_pd(value.value)};
#endif
  } else if constexpr (std::same_as<To, float16_t> &&
                       std::same_as<From, float32_t>) {
#if defined(HAS_F16C)
    constexpr int rounding = _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC;
    if constexpr (sizeof(FromRaw) == 16)
      return Vec<ToTag>{_mm_cvtps_ph(value.value, rounding)};
#if VEC_WIDTH >= 256
    else if constexpr (sizeof(FromRaw) == 32)
      return Vec<ToTag>{_mm256_cvtps_ph(value.value, rounding)};
#endif
#if VEC_WIDTH >= 512
    else return Vec<ToTag>{_mm512_cvtps_ph(value.value, rounding)};
#endif
#else
    return x86_convert_vec_scalar_word(to, from, value);
#endif
  } else if constexpr (std::same_as<To, float32_t> &&
                       std::same_as<From, float16_t>) {
#if defined(HAS_F16C)
    if constexpr (sizeof(ToRaw) == 16)
      return Vec<ToTag>{_mm_cvtph_ps(value.value)};
#if VEC_WIDTH >= 256
    else if constexpr (sizeof(ToRaw) == 32)
      return Vec<ToTag>{_mm256_cvtph_ps(value.value)};
#endif
#if VEC_WIDTH >= 512
    else return Vec<ToTag>{_mm512_cvtph_ps(value.value)};
#endif
#else
    return x86_convert_vec_scalar_word(to, from, value);
#endif
  } else if constexpr (std::same_as<To, bfloat16_t> &&
                       std::same_as<From, float32_t>) {
    static_assert(ToTraits::logical_lanes <= 4);
    const auto bits = _mm_castps_si128(value.value);
    const auto ones = _mm_set1_epi32(1);
    const auto bias = _mm_set1_epi32(0x7fff);
    auto rounded = _mm_add_epi32(
        bits, _mm_add_epi32(_mm_and_si128(_mm_srli_epi32(bits, 16), ones), bias));
    rounded = _mm_srli_epi32(rounded, 16);
    const auto nan = _mm_set1_epi32(0x7fc0);
    rounded = _mm_blendv_epi8(
        nan, rounded,
        _mm_castps_si128(_mm_cmpord_ps(value.value, value.value)));
    return Vec<ToTag>{_mm_packus_epi32(rounded, rounded)};
  } else if constexpr (std::same_as<To, float32_t> &&
                       std::same_as<From, bfloat16_t>) {
    if constexpr (sizeof(ToRaw) == 16) {
      return Vec<ToTag>{_mm_castsi128_ps(
          _mm_slli_epi32(_mm_cvtepu16_epi32(value.value), 16))};
    }
#if VEC_WIDTH >= 256
    else if constexpr (sizeof(ToRaw) == 32) {
      return Vec<ToTag>{_mm256_castsi256_ps(
          _mm256_slli_epi32(_mm256_cvtepu16_epi32(value.value), 16))};
    }
#endif
#if VEC_WIDTH >= 512
    else {
      return Vec<ToTag>{_mm512_castsi512_ps(
          _mm512_slli_epi32(_mm512_cvtepu16_epi32(value.value), 16))};
    }
#endif
  } else if constexpr (
      std::integral<To> && std::integral<From> &&
      sizeof(To) == sizeof(From) &&
      std::is_signed_v<To> != std::is_signed_v<From> && sizeof(To) < 8) {
    if constexpr (std::is_unsigned_v<To>) {
      if constexpr (sizeof(ToRaw) == 16) {
        if constexpr (sizeof(To) == 1)
          return Vec<ToTag>{_mm_max_epi8(value.value, _mm_setzero_si128())};
        else if constexpr (sizeof(To) == 2)
          return Vec<ToTag>{_mm_max_epi16(value.value, _mm_setzero_si128())};
        else
          return Vec<ToTag>{_mm_max_epi32(value.value, _mm_setzero_si128())};
      }
#if VEC_WIDTH >= 256
      else if constexpr (sizeof(ToRaw) == 32) {
        if constexpr (sizeof(To) == 1)
          return Vec<ToTag>{_mm256_max_epi8(value.value, _mm256_setzero_si256())};
        else if constexpr (sizeof(To) == 2)
          return Vec<ToTag>{_mm256_max_epi16(value.value, _mm256_setzero_si256())};
        else
          return Vec<ToTag>{_mm256_max_epi32(value.value, _mm256_setzero_si256())};
      }
#endif
#if VEC_WIDTH >= 512
      else {
        if constexpr (sizeof(To) == 1)
          return Vec<ToTag>{_mm512_max_epi8(value.value, _mm512_setzero_si512())};
        else if constexpr (sizeof(To) == 2)
          return Vec<ToTag>{_mm512_max_epi16(value.value, _mm512_setzero_si512())};
        else
          return Vec<ToTag>{_mm512_max_epi32(value.value, _mm512_setzero_si512())};
      }
#endif
    } else {
      if constexpr (sizeof(ToRaw) == 16) {
        if constexpr (sizeof(To) == 1)
          return Vec<ToTag>{_mm_min_epu8(
              value.value, _mm_set1_epi8(std::numeric_limits<To>::max()))};
        else if constexpr (sizeof(To) == 2)
          return Vec<ToTag>{_mm_min_epu16(
              value.value, _mm_set1_epi16(std::numeric_limits<To>::max()))};
        else
          return Vec<ToTag>{_mm_min_epu32(
              value.value, _mm_set1_epi32(std::numeric_limits<To>::max()))};
      }
#if VEC_WIDTH >= 256
      else if constexpr (sizeof(ToRaw) == 32) {
        if constexpr (sizeof(To) == 1)
          return Vec<ToTag>{_mm256_min_epu8(
              value.value, _mm256_set1_epi8(std::numeric_limits<To>::max()))};
        else if constexpr (sizeof(To) == 2)
          return Vec<ToTag>{_mm256_min_epu16(
              value.value, _mm256_set1_epi16(std::numeric_limits<To>::max()))};
        else
          return Vec<ToTag>{_mm256_min_epu32(
              value.value, _mm256_set1_epi32(std::numeric_limits<To>::max()))};
      }
#endif
#if VEC_WIDTH >= 512
      else {
        if constexpr (sizeof(To) == 1)
          return Vec<ToTag>{_mm512_min_epu8(
              value.value, _mm512_set1_epi8(std::numeric_limits<To>::max()))};
        else if constexpr (sizeof(To) == 2)
          return Vec<ToTag>{_mm512_min_epu16(
              value.value, _mm512_set1_epi16(std::numeric_limits<To>::max()))};
        else
          return Vec<ToTag>{_mm512_min_epu32(
              value.value, _mm512_set1_epi32(std::numeric_limits<To>::max()))};
      }
#endif
    }
  } else if constexpr (
      std::integral<To> && std::integral<From> && sizeof(To) < sizeof(From)) {
    return x86_narrow_integer_vec_native(to, from, value);
  } else if constexpr (
      std::integral<To> && std::integral<From> && sizeof(To) > sizeof(From)) {
#define VECOPS_VEC_X86_WIDEN_CASE(FromType, ToType, Suffix)             \
    if constexpr (std::same_as<From, FromType> &&                       \
                  std::same_as<To, ToType>) {                           \
      if constexpr (sizeof(ToRaw) == 16)                                \
        return Vec<ToTag>{_mm_cvte##Suffix(value.value)};                \
      else if constexpr (sizeof(ToRaw) == 32)                           \
        return Vec<ToTag>{_mm256_cvte##Suffix(value.value)};             \
      else                                                               \
        return Vec<ToTag>{_mm512_cvte##Suffix(value.value)};             \
    }
    VECOPS_VEC_X86_WIDEN_CASE(int8_t, int16_t, pi8_epi16)
    else VECOPS_VEC_X86_WIDEN_CASE(uint8_t, int16_t, pu8_epi16)
    else VECOPS_VEC_X86_WIDEN_CASE(int8_t, uint16_t, pi8_epi16)
    else VECOPS_VEC_X86_WIDEN_CASE(uint8_t, uint16_t, pu8_epi16)
    else VECOPS_VEC_X86_WIDEN_CASE(int8_t, int32_t, pi8_epi32)
    else VECOPS_VEC_X86_WIDEN_CASE(uint8_t, int32_t, pu8_epi32)
    else VECOPS_VEC_X86_WIDEN_CASE(int8_t, uint32_t, pi8_epi32)
    else VECOPS_VEC_X86_WIDEN_CASE(uint8_t, uint32_t, pu8_epi32)
    else VECOPS_VEC_X86_WIDEN_CASE(int8_t, int64_t, pi8_epi64)
    else VECOPS_VEC_X86_WIDEN_CASE(uint8_t, int64_t, pu8_epi64)
    else VECOPS_VEC_X86_WIDEN_CASE(int8_t, uint64_t, pi8_epi64)
    else VECOPS_VEC_X86_WIDEN_CASE(uint8_t, uint64_t, pu8_epi64)
    else VECOPS_VEC_X86_WIDEN_CASE(int16_t, int32_t, pi16_epi32)
    else VECOPS_VEC_X86_WIDEN_CASE(uint16_t, int32_t, pu16_epi32)
    else VECOPS_VEC_X86_WIDEN_CASE(int16_t, uint32_t, pi16_epi32)
    else VECOPS_VEC_X86_WIDEN_CASE(uint16_t, uint32_t, pu16_epi32)
    else VECOPS_VEC_X86_WIDEN_CASE(int16_t, int64_t, pi16_epi64)
    else VECOPS_VEC_X86_WIDEN_CASE(uint16_t, int64_t, pu16_epi64)
    else VECOPS_VEC_X86_WIDEN_CASE(int16_t, uint64_t, pi16_epi64)
    else VECOPS_VEC_X86_WIDEN_CASE(uint16_t, uint64_t, pu16_epi64)
    else VECOPS_VEC_X86_WIDEN_CASE(int32_t, int64_t, pi32_epi64)
    else VECOPS_VEC_X86_WIDEN_CASE(uint32_t, int64_t, pu32_epi64)
    else VECOPS_VEC_X86_WIDEN_CASE(int32_t, uint64_t, pi32_epi64)
    else VECOPS_VEC_X86_WIDEN_CASE(uint32_t, uint64_t, pu32_epi64)
    else return x86_convert_vec_scalar_word(to, from, value);
#undef VECOPS_VEC_X86_WIDEN_CASE
  } else if constexpr (
      std::same_as<From, float32_t> && std::integral<To> && sizeof(To) < 4) {
    using MidTag = Rebind<int32_t, ToTag>;
    return x86_convert_vec_native(
        to, MidTag{}, x86_convert_vec_native(MidTag{}, from, value));
  } else if constexpr (
      std::same_as<To, float32_t> && std::integral<From> && sizeof(From) < 4) {
    using Mid = std::conditional_t<std::is_signed_v<From>, int32_t, uint32_t>;
    using MidTag = Rebind<Mid, FromTag>;
    return x86_convert_vec_native(
        to, MidTag{}, x86_convert_vec_native(MidTag{}, from, value));
  } else if constexpr (
      std::same_as<From, float64_t> && std::integral<To> && sizeof(To) < 4) {
    using MidTag = Rebind<int32_t, ToTag>;
    return x86_convert_vec_native(
        to, MidTag{}, x86_convert_vec_native(MidTag{}, from, value));
  } else if constexpr (
      std::same_as<To, float64_t> && std::integral<From> && sizeof(From) < 4) {
    using Mid = std::conditional_t<std::is_signed_v<From>, int32_t, uint32_t>;
    using MidTag = Rebind<Mid, FromTag>;
    return x86_convert_vec_native(
        to, MidTag{}, x86_convert_vec_native(MidTag{}, from, value));
  } else {
    return x86_convert_vec_scalar_word(to, from, value);
  }
}

template <VectorTag ToTag, VectorTag FromTag, typename... Options>
VECOPS_ALWAYS_INLINE Vec<ToTag> x86_convert_vec_lane_native(
    ToTag to, FromTag from, Vec<FromTag> value, Options&&... options) {
  constexpr bool narrows =
      sizeof(ElementOf<FromTag>) > sizeof(ElementOf<ToTag>);
  constexpr int ratio = narrows
      ? static_cast<int>(sizeof(ElementOf<FromTag>) / sizeof(ElementOf<ToTag>))
      : static_cast<int>(sizeof(ElementOf<ToTag>) / sizeof(ElementOf<FromTag>));
  constexpr int levels = ratio == 2 ? 1 : ratio == 4 ? 2 : 3;
  constexpr int phase = conversion_lane_phase<Options...>();

  if constexpr (!narrows) {
    if constexpr (phase == 0) {
      const auto selected = conversion_select_even<levels>(from, value);
      using SelectedTag = decltype([] {
        if constexpr (levels == 1) return Half<FromTag>{};
        else if constexpr (levels == 2) return Half<Half<FromTag>>{};
        else return Half<Half<Half<FromTag>>>{};
      }());
      return x86_convert_vec_native(to, SelectedTag{}, selected);
    } else {
      static_assert(ratio == 2 && phase == 1);
      using SelectedTag = Half<FromTag>;
      return x86_convert_vec_native(
          to, SelectedTag{}, execute(OddOp{}, from, value));
    }
  } else {
    constexpr bool wraps =
        (std::same_as<std::remove_cvref_t<Options>, cvt::Wrap> || ...);
    using CompactTag = Rebind<ElementOf<ToTag>, FromTag>;
    const auto compact = [&] {
      if constexpr (wraps)
        return x86_convert_vec_wrap_native(CompactTag{}, from, value);
      else
        return x86_convert_vec_native(CompactTag{}, from, value);
    }();
    auto fallback = conversion_population<X86Backend>(
        to, std::forward<Options>(options)...);
    if constexpr (phase == 0) {
      return conversion_insert_even<levels, ToTag, CompactTag>(
          to, compact, fallback);
    } else {
      static_assert(ratio == 2 && phase == 1);
      const auto fallback_even = execute(EvenOp{}, to, fallback);
      return execute(InterleaveOp{}, to, fallback_even, compact);
    }
  }
}

template <VectorTag ToTag, VectorTag FromTag>
VECOPS_ALWAYS_INLINE Vec<ToTag> x86_convert_vec_native(
    ToTag to, FromTag from, Vec<FromTag> value) {
  using ToTraits = RepresentationTraits<X86Backend, ToTag>;
  using FromTraits = RepresentationTraits<X86Backend, FromTag>;
  using To = ElementOf<ToTag>;
  using From = ElementOf<FromTag>;
  static_assert(is_x86_conversion_element_v<ElementOf<ToTag>>);
  static_assert(is_x86_conversion_element_v<ElementOf<FromTag>>);

  constexpr bool split_bfloat32_narrowing =
      std::same_as<ElementOf<ToTag>, bfloat16_t> &&
      std::same_as<ElementOf<FromTag>, float32_t> &&
      ToTraits::logical_lanes > 4;
  using ToWordTag = FixedTag<To, ToTraits::word_lanes>;
  using FromChunkTag = Rebind<From, ToWordTag>;
  using FromChunkTraits = RepresentationTraits<X86Backend, FromChunkTag>;
  constexpr bool directly_partitioned_output =
      ToTraits::word_count > 1 &&
      FromTraits::word_count ==
          ToTraits::word_count * FromChunkTraits::word_count;

  if constexpr (directly_partitioned_output) {
    return construct_words<X86Backend>(
        to, [&]<nint_t OutputIndex>(ToTag) VECOPS_INLINE_LAMBDA {
      const auto input = construct_words<X86Backend>(
          FromChunkTag{},
          [&]<nint_t InputIndex>(FromChunkTag) VECOPS_INLINE_LAMBDA {
            constexpr nint_t source_index =
                OutputIndex * FromChunkTraits::word_count + InputIndex;
            return ::vecops::vec::get_word<source_index>(from, value);
          });
      const auto converted = x86_convert_vec_native(
          ToWordTag{}, FromChunkTag{}, input);
      return ::vecops::vec::get_word<0>(ToWordTag{}, converted);
    });
  } else
#if defined(HAS_AVX512_BF16) && !defined(VECOPS_PRESERVE_SUBNORMALS)
  if constexpr (
      split_bfloat32_narrowing && ToTraits::logical_lanes <= 32) {
    if constexpr (ToTraits::logical_lanes <= 8) {
      return Vec<ToTag>{x86_conversion_bfloat_bits<__m128i>(
          _mm256_cvtneps_pbh(value.value))};
    } else if constexpr (ToTraits::logical_lanes <= 16) {
      return Vec<ToTag>{x86_conversion_bfloat_bits<__m256i>(
          _mm512_cvtneps_pbh(value.value))};
    } else {
      static_assert(ToTraits::logical_lanes == 32);
      const auto low = ::vecops::vec::get_word<0>(from, value).value;
      const auto high = ::vecops::vec::get_word<1>(from, value).value;
      return Vec<ToTag>{x86_conversion_bfloat_bits<__m512i>(
          _mm512_cvtne2ps_pbh(high, low))};
    }
  } else
#endif
  if constexpr (
      ToTraits::word_count == 1 && FromTraits::word_count == 1 &&
      !split_bfloat32_narrowing) {
    return x86_convert_vec_single_word(to, from, value);
  } else {
    using ToHalf = Half<ToTag>;
    using FromHalf = Half<FromTag>;
    const auto lower = x86_convert_vec_native(
        ToHalf{}, FromHalf{}, execute(LowerOp{}, from, value));
    const auto upper = x86_convert_vec_native(
        ToHalf{}, FromHalf{}, execute(UpperOp{}, from, value));
    return execute(ConcatOp{}, to, lower, upper);
  }
}

template <VectorTag ToTag>
struct NativeImpl<X86Backend, ConvertOp, ToTag> {
  template <VectorTag FromTag, typename... Options>
    requires (valid_conversion_options<ToTag, FromTag, Options...>())
  static VECOPS_ALWAYS_INLINE Vec<ToTag> call(
      ConvertOp, ToTag to, FromTag from, Vec<FromTag> value,
      Options&&... options) {
    constexpr bool lane_layout =
        (is_lane_option_v<std::remove_cvref_t<Options>> || ...);
    constexpr bool wraps =
        (std::same_as<std::remove_cvref_t<Options>, cvt::Wrap> || ...);
    constexpr bool masked =
        (IsMaskedOption<std::remove_cvref_t<Options>>::value || ...);
    const auto converted = [&] {
      if constexpr (lane_layout) {
        return x86_convert_vec_lane_native(
            to, from, value, std::forward<Options>(options)...);
      } else if constexpr (wraps) {
        return x86_convert_vec_wrap_native(to, from, value);
      } else {
        return x86_convert_vec_native(to, from, value);
      }
    }();
    if constexpr (masked) {
      const auto inactive = conversion_population<X86Backend>(
          to, std::forward<Options>(options)...);
      const Mask<ToTag>* mask = nullptr;
      ([&]<typename Option>(Option&& option) {
        if constexpr (IsMaskedOption<
                          std::remove_cvref_t<Option>>::value)
          mask = &option.value;
      }(std::forward<Options>(options)), ...);
      return execute(BlendOp{}, to, inactive, *mask, converted);
    } else {
      return converted;
    }
  }

  template <VectorTag FromTag>
    requires (valid_mask_conversion<ToTag, FromTag>())
  static VECOPS_ALWAYS_INLINE Mask<ToTag> call(
      ConvertOp, ToTag to, FromTag from, Mask<FromTag> value) {
    return x86_convert_mask_native(to, from, value);
  }
};

} // namespace vecops::vec::details

#endif // VECOPS_VEC_DETAILS_X86_CONVERSION_H
