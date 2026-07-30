#ifndef VECOPS_VEC_DETAILS_SVE_CONVERSION_H
#define VECOPS_VEC_DETAILS_SVE_CONVERSION_H

/**
 * @file Conversion.h
 * @brief SVE backend implementations for element type conversion operations.
 */

#include <cstdint>
#include <limits>
#include <type_traits>

#include "vecops/vec/details/sve/Basic.h"
#include "vecops/vec/details/sve/Bf16.h"

namespace vecops::vec::details {


/* **************************************************************************** */
//                  Element type conversion implementations                   //
/* **************************************************************************** */

template <typename T>
inline constexpr bool sve_is_signed_integer_element =
    std::same_as<T, int8_t> || std::same_as<T, int16_t> ||
    std::same_as<T, int32_t> || std::same_as<T, int64_t>;

template <typename T>
inline constexpr bool sve_is_unsigned_integer_element =
    std::same_as<T, uint8_t> || std::same_as<T, uint16_t> ||
    std::same_as<T, uint32_t> || std::same_as<T, uint64_t>;

template <typename T>
inline constexpr bool sve_is_integer_element =
    sve_is_signed_integer_element<T> || sve_is_unsigned_integer_element<T>;

template <typename T>
inline constexpr bool sve_is_conversion_element =
    std::same_as<T, bfloat16_t> || std::same_as<T, float16_t> ||
    std::same_as<T, float32_t> || std::same_as<T, float64_t> ||
    sve_is_integer_element<T>;

template <Element To, Element From, typename Raw>
VECOPS_ALWAYS_INLINE auto sve_convert_same_size(Raw value) {
  static_assert(sizeof(To) == sizeof(From));
  if constexpr (std::same_as<To, From>) {
    return value;
  } else if constexpr (std::same_as<From, int8_t> &&
                       std::same_as<To, uint8_t>) {
    return svreinterpret_u8_s8(svmax_n_s8_z(svptrue_b8(), value, 0));
  } else if constexpr (std::same_as<From, uint8_t> &&
                       std::same_as<To, int8_t>) {
    return svreinterpret_s8_u8(svmin_n_u8_z(
        svptrue_b8(), value, std::numeric_limits<int8_t>::max()));
  } else if constexpr (std::same_as<From, int16_t> &&
                       std::same_as<To, uint16_t>) {
    return svreinterpret_u16_s16(svmax_n_s16_z(svptrue_b16(), value, 0));
  } else if constexpr (std::same_as<From, uint16_t> &&
                       std::same_as<To, int16_t>) {
    return svreinterpret_s16_u16(svmin_n_u16_z(
        svptrue_b16(), value, std::numeric_limits<int16_t>::max()));
  } else if constexpr (std::same_as<From, int32_t> &&
                       std::same_as<To, uint32_t>) {
    return svreinterpret_u32_s32(svmax_n_s32_z(svptrue_b32(), value, 0));
  } else if constexpr (std::same_as<From, uint32_t> &&
                       std::same_as<To, int32_t>) {
    return svreinterpret_s32_u32(svmin_n_u32_z(
        svptrue_b32(), value, std::numeric_limits<int32_t>::max()));
  } else if constexpr (std::same_as<From, int64_t> &&
                       std::same_as<To, uint64_t>) {
    return svreinterpret_u64_s64(svmax_n_s64_z(svptrue_b64(), value, 0));
  } else if constexpr (std::same_as<From, uint64_t> &&
                       std::same_as<To, int64_t>) {
    return svreinterpret_s64_u64(svmin_n_u64_z(
        svptrue_b64(), value,
        static_cast<uint64_t>(std::numeric_limits<int64_t>::max())));
  } else if constexpr (std::same_as<From, float16_t> &&
                       std::same_as<To, int16_t>) {
    return svcvt_s16_f16_x(svptrue_b16(), value);
  } else if constexpr (std::same_as<From, float16_t> &&
                       std::same_as<To, uint16_t>) {
    return svcvt_u16_f16_x(svptrue_b16(), value);
  } else if constexpr (std::same_as<From, int16_t> &&
                       std::same_as<To, float16_t>) {
    return svcvt_f16_s16_x(svptrue_b16(), value);
  } else if constexpr (std::same_as<From, uint16_t> &&
                       std::same_as<To, float16_t>) {
    return svcvt_f16_u16_x(svptrue_b16(), value);
  } else if constexpr (std::same_as<From, float32_t> &&
                       std::same_as<To, int32_t>) {
    return svcvt_s32_f32_x(svptrue_b32(), value);
  } else if constexpr (std::same_as<From, float32_t> &&
                       std::same_as<To, uint32_t>) {
    return svcvt_u32_f32_x(svptrue_b32(), value);
  } else if constexpr (std::same_as<From, int32_t> &&
                       std::same_as<To, float32_t>) {
    return svcvt_f32_s32_x(svptrue_b32(), value);
  } else if constexpr (std::same_as<From, uint32_t> &&
                       std::same_as<To, float32_t>) {
    return svcvt_f32_u32_x(svptrue_b32(), value);
  } else if constexpr (std::same_as<From, float64_t> &&
                       std::same_as<To, int64_t>) {
    return svcvt_s64_f64_x(svptrue_b64(), value);
  } else if constexpr (std::same_as<From, float64_t> &&
                       std::same_as<To, uint64_t>) {
    return svcvt_u64_f64_x(svptrue_b64(), value);
  } else if constexpr (std::same_as<From, int64_t> &&
                       std::same_as<To, float64_t>) {
    return svcvt_f64_s64_x(svptrue_b64(), value);
  } else if constexpr (std::same_as<From, uint64_t> &&
                       std::same_as<To, float64_t>) {
    return svcvt_f64_u64_x(svptrue_b64(), value);
  } else if constexpr (std::same_as<From, bfloat16_t>) {
    const auto lo_f32 = sve_bf16_to_f32_lo(value);
    const auto hi_f32 = sve_bf16_to_f32_hi(value);
    if constexpr (std::same_as<To, float16_t>) {
      const auto lo = svcvt_f16_f32_x(svptrue_b16(), lo_f32);
      const auto hi = svcvt_f16_f32_x(svptrue_b16(), hi_f32);
      return svreinterpret_f16_u16(svuzp1_u16(
          svreinterpret_u16_f16(lo), svreinterpret_u16_f16(hi)));
    } else if constexpr (std::same_as<To, int16_t>) {
      const auto lo = svcvt_s32_f32_x(svptrue_b32(), lo_f32);
      const auto hi = svcvt_s32_f32_x(svptrue_b32(), hi_f32);
      const auto clamped_lo = svmax_n_s32_z(
          svptrue_b32(),
          svmin_n_s32_z(svptrue_b32(), lo,
                        std::numeric_limits<int16_t>::max()),
          std::numeric_limits<int16_t>::min());
      const auto clamped_hi = svmax_n_s32_z(
          svptrue_b32(),
          svmin_n_s32_z(svptrue_b32(), hi,
                        std::numeric_limits<int16_t>::max()),
          std::numeric_limits<int16_t>::min());
      return svreinterpret_s16_u16(svuzp1_u16(
          svreinterpret_u16_s32(clamped_lo),
          svreinterpret_u16_s32(clamped_hi)));
    } else if constexpr (std::same_as<To, uint16_t>) {
      const auto lo = svcvt_u32_f32_x(svptrue_b32(), lo_f32);
      const auto hi = svcvt_u32_f32_x(svptrue_b32(), hi_f32);
      const auto clamped_lo = svmin_n_u32_z(
          svptrue_b32(), lo, std::numeric_limits<uint16_t>::max());
      const auto clamped_hi = svmin_n_u32_z(
          svptrue_b32(), hi, std::numeric_limits<uint16_t>::max());
      return svuzp1_u16(
          svreinterpret_u16_u32(clamped_lo),
          svreinterpret_u16_u32(clamped_hi));
    } else {
      static_assert(dispatch_dependent_false<To, From>,
                    "unsupported same-size SVE conversion");
    }
  } else if constexpr (std::same_as<To, bfloat16_t>) {
    svfloat32_t lo_f32;
    svfloat32_t hi_f32;
    if constexpr (std::same_as<From, float16_t>) {
      lo_f32 = svcvt_f32_f16_x(svptrue_b32(), value);
#if defined(__ARM_FEATURE_SVE2)
      hi_f32 = svcvtlt_f32_f16_x(svptrue_b32(), value);
#else
      const auto odd = svuzp2_u16(
          svreinterpret_u16_f16(value), svreinterpret_u16_f16(value));
      hi_f32 = svcvt_f32_f16_x(
          svptrue_b32(), svreinterpret_f16_u16(svzip1_u16(odd, odd)));
#endif
    } else if constexpr (std::same_as<From, int16_t>) {
      lo_f32 = svcvt_f32_s32_x(svptrue_b32(), svunpklo_s32(value));
      hi_f32 = svcvt_f32_s32_x(svptrue_b32(), svunpkhi_s32(value));
    } else if constexpr (std::same_as<From, uint16_t>) {
      lo_f32 = svcvt_f32_u32_x(svptrue_b32(), svunpklo_u32(value));
      hi_f32 = svcvt_f32_u32_x(svptrue_b32(), svunpkhi_u32(value));
    } else {
      static_assert(dispatch_dependent_false<To, From>,
                    "unsupported same-size SVE conversion");
    }
#if defined(__ARM_FEATURE_SVE_BF16) && !defined(VECOPS_PRESERVE_SUBNORMALS)
    const auto lo = svcvt_bf16_f32_x(svptrue_b16(), lo_f32);
    if constexpr (std::same_as<From, float16_t>) {
      return svcvtnt_bf16_f32_x(lo, svptrue_b16(), hi_f32);
    } else {
      const auto hi = svcvt_bf16_f32_x(svptrue_b16(), hi_f32);
      return svuzp1_bf16(lo, hi);
    }
#else
    const auto lo = sve_f32_to_bf16_rne_bits(lo_f32);
    const auto hi = sve_f32_to_bf16_rne_bits(hi_f32);
    if constexpr (std::same_as<From, float16_t>) {
      const auto zero = svdup_n_u16(0);
      const auto lo_compact = svuzp1_u16(svreinterpret_u16_u32(lo), zero);
      const auto hi_compact = svuzp1_u16(svreinterpret_u16_u32(hi), zero);
      return svreinterpret_bf16_u16(svzip1_u16(lo_compact, hi_compact));
    } else {
      return svreinterpret_bf16_u16(svuzp1_u16(
          svreinterpret_u16_u32(lo), svreinterpret_u16_u32(hi)));
    }
#endif
  } else {
    static_assert(dispatch_dependent_false<To, From>,
                  "unsupported same-size SVE conversion");
  }
}

template <std::size_t Bytes, bool Signed>
using SVEInteger = std::conditional_t<
    Bytes == 1, std::conditional_t<Signed, int8_t, uint8_t>,
    std::conditional_t<
        Bytes == 2, std::conditional_t<Signed, int16_t, uint16_t>,
        std::conditional_t<
            Bytes == 4, std::conditional_t<Signed, int32_t, uint32_t>,
            std::conditional_t<Signed, int64_t, uint64_t>>>>;

template <Element From>
using SVEWidenedElement = std::conditional_t<
    std::same_as<From, float16_t> || std::same_as<From, bfloat16_t>,
    float32_t,
    std::conditional_t<
        std::same_as<From, float32_t>, float64_t,
        SVEInteger<sizeof(From) * 2, sve_is_signed_integer_element<From>>>>;

template <Element To>
using SVENarrowedElement = std::conditional_t<
    std::same_as<To, float32_t>, float32_t,
    std::conditional_t<
        std::same_as<To, float16_t> || std::same_as<To, bfloat16_t>,
        float32_t,
        SVEInteger<sizeof(To) * 2, sve_is_signed_integer_element<To>>>>;

template <Element To, Element From, bool Wrap = false, typename Raw>
VECOPS_ALWAYS_INLINE auto sve_convert_one_word_raw(Raw value);

template <Element To, Element From, typename Raw>
VECOPS_ALWAYS_INLINE auto sve_reinterpret_integer(Raw value) {
  static_assert(sve_is_integer_element<To> && sve_is_integer_element<From>);
  static_assert(sizeof(To) == sizeof(From));
  if constexpr (std::same_as<To, From> ||
                sve_is_signed_integer_element<To> ==
                    sve_is_signed_integer_element<From>) {
    return value;
  } else if constexpr (sizeof(To) == 1) {
    if constexpr (sve_is_signed_integer_element<To>)
      return svreinterpret_s8_u8(value);
    else return svreinterpret_u8_s8(value);
  } else if constexpr (sizeof(To) == 2) {
    if constexpr (sve_is_signed_integer_element<To>)
      return svreinterpret_s16_u16(value);
    else return svreinterpret_u16_s16(value);
  } else if constexpr (sizeof(To) == 4) {
    if constexpr (sve_is_signed_integer_element<To>)
      return svreinterpret_s32_u32(value);
    else return svreinterpret_u32_s32(value);
  } else {
    if constexpr (sve_is_signed_integer_element<To>)
      return svreinterpret_s64_u64(value);
    else return svreinterpret_u64_s64(value);
  }
}

template <Element To, Element From, typename Raw>
VECOPS_ALWAYS_INLINE auto sve_promote_one_step(Raw value) {
  static_assert(sizeof(To) == sizeof(From) * 2);
  using Widened = SVEWidenedElement<From>;
  auto widened = [&] {
    if constexpr (std::same_as<From, bfloat16_t>)
      return sve_bf16_to_f32_lo(value);
    else if constexpr (std::same_as<From, float16_t>)
      return svcvt_f32_f16_x(svptrue_b32(), svzip1_f16(value, value));
    else if constexpr (std::same_as<From, float32_t>)
      return svcvt_f64_f32_x(svptrue_b64(), svzip1_f32(value, value));
    else if constexpr (sve_is_signed_integer_element<From>) {
      if constexpr (sizeof(From) == 1) return svunpklo_s16(value);
      else if constexpr (sizeof(From) == 2) return svunpklo_s32(value);
      else return svunpklo_s64(value);
    } else {
      if constexpr (sizeof(From) == 1) return svunpklo_u16(value);
      else if constexpr (sizeof(From) == 2) return svunpklo_u32(value);
      else return svunpklo_u64(value);
    }
  }();
  if constexpr (sve_is_integer_element<To> &&
                sve_is_integer_element<Widened>) {
    // Widening follows scalar cast semantics. In particular, a negative
    // signed input converted to a wider unsigned type is sign-extended and
    // then reinterpreted modulo the wider width; applying the equal-width
    // signedness clamp here would incorrectly turn it into zero.
    return sve_reinterpret_integer<To, Widened>(widened);
  } else {
    return sve_convert_same_size<To, Widened>(widened);
  }
}

template <Element To, Element From, bool Wrap = false, typename Raw>
VECOPS_ALWAYS_INLINE auto sve_demote_integer_one_step(Raw value) {
  static_assert(sve_is_integer_element<To> && sve_is_integer_element<From>);
  static_assert(sizeof(To) * 2 == sizeof(From));
  if constexpr (Wrap) {
    if constexpr (sizeof(To) == 1) {
      const auto bytes = [&] {
        if constexpr (std::is_signed_v<From>)
          return svreinterpret_u8_s16(value);
        else
          return svreinterpret_u8_u16(value);
      }();
      const auto compact = svuzp1_u8(bytes, bytes);
      if constexpr (std::is_signed_v<To>)
        return svreinterpret_s8_u8(compact);
      else
        return compact;
    } else if constexpr (sizeof(To) == 2) {
      const auto halves = [&] {
        if constexpr (std::is_signed_v<From>)
          return svreinterpret_u16_s32(value);
        else
          return svreinterpret_u16_u32(value);
      }();
      const auto compact = svuzp1_u16(halves, halves);
      if constexpr (std::is_signed_v<To>)
        return svreinterpret_s16_u16(compact);
      else
        return compact;
    } else {
      const auto words = [&] {
        if constexpr (std::is_signed_v<From>)
          return svreinterpret_u32_s64(value);
        else
          return svreinterpret_u32_u64(value);
      }();
      const auto compact = svuzp1_u32(words, words);
      if constexpr (std::is_signed_v<To>)
        return svreinterpret_s32_u32(compact);
      else
        return compact;
    }
  }
#if defined(HAS_SVE2)
  // QXTNB places each narrowed value in the bottom subelement of its wide
  // lane; UZP1 compacts that single input stream into consecutive lanes.
  // QXTNT is useful when a second independent wide vector supplies the top
  // subelements, but cannot replace the compaction for this one-vector step.
  if constexpr (std::is_signed_v<From> && std::is_signed_v<To>) {
    if constexpr (sizeof(From) == 2) {
      const auto bottom = svqxtnb_s16(value);
      return svreinterpret_s8_u8(svuzp1_u8(
          svreinterpret_u8_s8(bottom), svreinterpret_u8_s8(bottom)));
    } else if constexpr (sizeof(From) == 4) {
      const auto bottom = svqxtnb_s32(value);
      return svreinterpret_s16_u16(svuzp1_u16(
          svreinterpret_u16_s16(bottom), svreinterpret_u16_s16(bottom)));
    } else {
      const auto bottom = svqxtnb_s64(value);
      return svreinterpret_s32_u32(svuzp1_u32(
          svreinterpret_u32_s32(bottom), svreinterpret_u32_s32(bottom)));
    }
  } else if constexpr (std::is_unsigned_v<From> && std::is_unsigned_v<To>) {
    if constexpr (sizeof(From) == 2) {
      const auto bottom = svqxtnb_u16(value);
      return svuzp1_u8(bottom, bottom);
    } else if constexpr (sizeof(From) == 4) {
      const auto bottom = svqxtnb_u32(value);
      return svuzp1_u16(bottom, bottom);
    } else {
      const auto bottom = svqxtnb_u64(value);
      return svuzp1_u32(bottom, bottom);
    }
  } else if constexpr (std::is_signed_v<From> && std::is_unsigned_v<To>) {
    if constexpr (sizeof(From) == 2) {
      const auto bottom = svqxtunb_s16(value);
      return svuzp1_u8(bottom, bottom);
    } else if constexpr (sizeof(From) == 4) {
      const auto bottom = svqxtunb_s32(value);
      return svuzp1_u16(bottom, bottom);
    } else {
      const auto bottom = svqxtunb_s64(value);
      return svuzp1_u32(bottom, bottom);
    }
  }
#endif
  const auto pg = sve_prefix_predicate<From>(sve_word_lanes<From>());
  auto same_sign = [&] {
    if constexpr (sve_is_signed_integer_element<To>) {
      auto signed_value = [&] {
        if constexpr (sve_is_signed_integer_element<From>) return value;
        else return sve_convert_same_size<SVEInteger<sizeof(From), true>, From>(value);
      }();
      const auto clamped = [&] {
        if constexpr (sizeof(From) == 2)
          return svmax_n_s16_z(pg, svmin_n_s16_z(
              pg, signed_value, std::numeric_limits<To>::max()),
              std::numeric_limits<To>::min());
        else if constexpr (sizeof(From) == 4)
          return svmax_n_s32_z(pg, svmin_n_s32_z(
              pg, signed_value, std::numeric_limits<To>::max()),
              std::numeric_limits<To>::min());
        else
          return svmax_n_s64_z(pg, svmin_n_s64_z(
              pg, signed_value, std::numeric_limits<To>::max()),
              std::numeric_limits<To>::min());
      }();
      if constexpr (sizeof(To) == 1)
        return svreinterpret_s8_u8(svuzp1_u8(
            svreinterpret_u8_s16(clamped), svreinterpret_u8_s16(clamped)));
      else if constexpr (sizeof(To) == 2)
        return svreinterpret_s16_u16(svuzp1_u16(
            svreinterpret_u16_s32(clamped), svreinterpret_u16_s32(clamped)));
      else
        return svreinterpret_s32_u32(svuzp1_u32(
            svreinterpret_u32_s64(clamped), svreinterpret_u32_s64(clamped)));
    } else {
      auto unsigned_value = [&] {
        if constexpr (sve_is_unsigned_integer_element<From>) return value;
        else return sve_convert_same_size<SVEInteger<sizeof(From), false>, From>(value);
      }();
      const auto clamped = [&] {
        if constexpr (sizeof(From) == 2)
          return svmin_n_u16_z(pg, unsigned_value,
                               std::numeric_limits<To>::max());
        else if constexpr (sizeof(From) == 4)
          return svmin_n_u32_z(pg, unsigned_value,
                               std::numeric_limits<To>::max());
        else
          return svmin_n_u64_z(pg, unsigned_value,
                               std::numeric_limits<To>::max());
      }();
      if constexpr (sizeof(To) == 1)
        return svuzp1_u8(
            svreinterpret_u8_u16(clamped), svreinterpret_u8_u16(clamped));
      else if constexpr (sizeof(To) == 2)
        return svuzp1_u16(
            svreinterpret_u16_u32(clamped), svreinterpret_u16_u32(clamped));
      else
        return svuzp1_u32(
            svreinterpret_u32_u64(clamped), svreinterpret_u32_u64(clamped));
    }
  }();
  return same_sign;
}

template <Element To, Element From, bool Wrap = false, typename Raw>
VECOPS_ALWAYS_INLINE auto sve_demote_one_step(Raw value) {
  static_assert(sizeof(To) * 2 == sizeof(From));
  if constexpr (sve_is_integer_element<From> && sve_is_integer_element<To>) {
    return sve_demote_integer_one_step<To, From, Wrap>(value);
  } else if constexpr (sizeof(From) == 8) {
    auto narrowed = [&] {
      if constexpr (std::same_as<To, float32_t>) {
        if constexpr (std::same_as<From, float64_t>)
          return svcvt_f32_f64_x(svptrue_b64(), value);
        else if constexpr (std::same_as<From, int64_t>)
          return svcvt_f32_s64_x(svptrue_b64(), value);
        else
          return svcvt_f32_u64_x(svptrue_b64(), value);
      } else if constexpr (std::same_as<To, int32_t>) {
        if constexpr (std::same_as<From, float64_t>)
          return svcvt_s32_f64_x(svptrue_b64(), value);
        else {
          using Mid = SVEInteger<8, sve_is_signed_integer_element<From>>;
          return sve_demote_integer_one_step<int32_t, Mid>(value);
        }
      } else if constexpr (std::same_as<To, uint32_t>) {
        if constexpr (std::same_as<From, float64_t>)
          return svcvt_u32_f64_x(svptrue_b64(), value);
        else {
          using Mid = SVEInteger<8, sve_is_signed_integer_element<From>>;
          return sve_demote_integer_one_step<uint32_t, Mid>(value);
        }
      } else {
        static_assert(dispatch_dependent_false<To, From>,
                      "unsupported one-step SVE demotion");
      }
    }();
    if constexpr (std::same_as<To, float32_t>)
      return svuzp1_f32(narrowed, narrowed);
    else if constexpr (std::same_as<To, int32_t>) {
      if constexpr (std::same_as<From, float64_t>)
        return svuzp1_s32(narrowed, narrowed);
      else return narrowed;
    } else {
      if constexpr (std::same_as<From, float64_t>)
        return svuzp1_u32(narrowed, narrowed);
      else return narrowed;
    }
  } else if constexpr (sizeof(From) == 4) {
    if constexpr (std::same_as<To, float16_t>) {
      const auto f32 = sve_convert_same_size<float32_t, From>(value);
      const auto converted = svcvt_f16_f32_x(svptrue_b16(), f32);
      return svreinterpret_f16_u16(svuzp1_u16(
          svreinterpret_u16_f16(converted),
          svreinterpret_u16_f16(converted)));
    } else if constexpr (std::same_as<To, bfloat16_t>) {
      return sve_f32_to_bf16_lo(
          sve_convert_same_size<float32_t, From>(value));
    } else {
      using Mid = SVEInteger<4, sve_is_signed_integer_element<To>>;
      return sve_demote_integer_one_step<To, Mid>(
          sve_convert_same_size<Mid, From>(value));
    }
  } else {
    static_assert(sizeof(From) == 2);
    using Mid = SVEInteger<2, sve_is_signed_integer_element<To>>;
    return sve_demote_integer_one_step<To, Mid>(
        sve_convert_same_size<Mid, From>(value));
  }
}

template <Element To, Element From, bool Wrap, typename Raw>
VECOPS_ALWAYS_INLINE auto sve_convert_one_word_raw(Raw value) {
  static_assert(
      sve_is_conversion_element<To> && sve_is_conversion_element<From>,
      "unsupported SVE conversion element type");
  if constexpr (sizeof(To) == sizeof(From)) {
    return sve_convert_same_size<To, From>(value);
  } else if constexpr (sizeof(To) == sizeof(From) * 2) {
    return sve_promote_one_step<To, From>(value);
  } else if constexpr (sizeof(To) > sizeof(From)) {
    using Mid = SVEWidenedElement<From>;
    return sve_convert_one_word_raw<To, Mid, Wrap>(
        sve_promote_one_step<Mid, From>(value));
  } else if constexpr (sizeof(To) * 2 == sizeof(From)) {
    return sve_demote_one_step<To, From, Wrap>(value);
  } else {
    using Mid = SVENarrowedElement<To>;
    return sve_convert_one_word_raw<To, Mid, Wrap>(
        sve_convert_one_word_raw<Mid, From, Wrap>(value));
  }
}

template <bool Wrap = false, VectorTag ToTag, VectorTag FromTag>
VECOPS_ALWAYS_INLINE Vec<ToTag> sve_convert_vector(
    ToTag to, FromTag from, Vec<FromTag> value) {
  if constexpr (num_words(to) == 1 && num_words(from) == 1) {
    const auto raw = sve_basic_raw_word(value);
    return sve_basic_wrap_word<ToTag>(
        sve_convert_one_word_raw<
            ElementOf<ToTag>, ElementOf<FromTag>, Wrap>(raw));
  } else {
    using ToHalf = Half<ToTag>;
    using FromHalf = Half<FromTag>;
    const auto lower = sve_convert_vector<Wrap>(
        ToHalf{}, FromHalf{}, execute(LowerOp{}, from, value));
    const auto upper = sve_convert_vector<Wrap>(
        ToHalf{}, FromHalf{}, execute(UpperOp{}, from, value));
    return execute(ConcatOp{}, to, lower, upper);
  }
}

template <VectorTag ToTag, VectorTag FromTag>
VECOPS_ALWAYS_INLINE Mask<ToTag> sve_convert_mask(
    ToTag to, FromTag from, Mask<FromTag> value) {
  if constexpr (num_words(to) == 1 && num_words(from) == 1) {
    constexpr std::size_t to_bytes = sizeof(ElementOf<ToTag>);
    constexpr std::size_t from_bytes = sizeof(ElementOf<FromTag>);
    svbool_t result = value;
    if constexpr (to_bytes > from_bytes) {
      if constexpr (to_bytes / from_bytes >= 2) result = svunpklo_b(result);
      if constexpr (to_bytes / from_bytes >= 4) result = svunpklo_b(result);
      if constexpr (to_bytes / from_bytes >= 8) result = svunpklo_b(result);
    } else if constexpr (to_bytes < from_bytes) {
      if constexpr (from_bytes == 8) result = svuzp1_b32(result, result);
      if constexpr (from_bytes >= 4 && to_bytes <= 2)
        result = svuzp1_b16(result, result);
      if constexpr (from_bytes >= 2 && to_bytes == 1)
        result = svuzp1_b8(result, result);
    }
    const auto valid = sve_prefix_predicate<ElementOf<ToTag>>(size(to));
    return svand_b_z(svptrue_b8(), result, valid);
  } else {
    // Conversion changes predicate granularity and may change the physical
    // word count. Split both complete logical Tags together; wordwise batching
    // cannot infer this correspondence from either predicate representation.
    using ToHalf = Half<ToTag>;
    using FromHalf = Half<FromTag>;
    const auto lower = sve_convert_mask(
        ToHalf{}, FromHalf{}, execute(LowerOp{}, from, value));
    const auto upper = sve_convert_mask(
        ToHalf{}, FromHalf{}, execute(UpperOp{}, from, value));
    return execute(ConcatOp{}, to, lower, upper);
  }
}

template <int Levels, VectorTag Tag>
VECOPS_ALWAYS_INLINE auto sve_conversion_select_even(
    Tag tag, Vec<Tag> value) {
  if constexpr (Levels == 0) {
    return value;
  } else {
    using HalfTag = Half<Tag>;
    return sve_conversion_select_even<Levels - 1>(
        HalfTag{}, execute(EvenOp{}, tag, value));
  }
}

template <typename... Options>
consteval int sve_conversion_lane_phase() {
  int phase = -1;
  ([]<typename Option>(int& result) {
    using Clean = std::remove_cvref_t<Option>;
    if constexpr (IsLaneOption<Clean>::value)
      result = IsLaneOption<Clean>::phase;
  }.template operator()<Options>(phase), ...);
  return phase;
}

template <VectorTag Tag, typename... Options>
VECOPS_ALWAYS_INLINE Vec<Tag> sve_conversion_population(
    Tag tag, Options&&... options) {
  if constexpr (option_count<IsVectorMergeOption, Options...> == 1) {
    return find_option<IsVectorMergeOption>(
        std::forward<Options>(options)...).value;
  } else if constexpr (option_count<IsScalarMergeOption, Options...> == 1) {
    return execute(
        FillOp{}, tag,
        find_option<IsScalarMergeOption>(
            std::forward<Options>(options)...).value);
  } else {
    return execute(FillOp{}, tag, ElementOf<Tag>{});
  }
}

template <int Levels, VectorTag Tag, VectorTag ValuesTag>
VECOPS_ALWAYS_INLINE Vec<Tag> sve_conversion_insert_even(
    Tag tag, Vec<ValuesTag> values, Vec<Tag> fallback) {
  if constexpr (Levels == 0) {
    static_assert(std::same_as<Tag, ValuesTag>);
    return values;
  } else {
    using HalfTag = Half<Tag>;
    const auto fallback_even = execute(EvenOp{}, tag, fallback);
    const auto fallback_odd = execute(OddOp{}, tag, fallback);
    const auto result_even =
        sve_conversion_insert_even<Levels - 1, HalfTag, ValuesTag>(
            HalfTag{}, values, fallback_even);
    return execute(InterleaveOp{}, tag, result_even, fallback_odd);
  }
}

template <VectorTag ToTag, VectorTag FromTag, typename... Options>
VECOPS_ALWAYS_INLINE Vec<ToTag> sve_convert_lane_native(
    ToTag to, FromTag from, Vec<FromTag> value, Options&&... options) {
  constexpr bool narrows =
      sizeof(ElementOf<FromTag>) > sizeof(ElementOf<ToTag>);
  constexpr int ratio = narrows
      ? static_cast<int>(sizeof(ElementOf<FromTag>) / sizeof(ElementOf<ToTag>))
      : static_cast<int>(sizeof(ElementOf<ToTag>) / sizeof(ElementOf<FromTag>));
  constexpr int levels = ratio == 2 ? 1 : ratio == 4 ? 2 : 3;
  constexpr int phase = sve_conversion_lane_phase<Options...>();

  if constexpr (!narrows) {
    if constexpr (phase == 0) {
      const auto selected = sve_conversion_select_even<levels>(from, value);
      using SelectedTag = decltype([] {
        if constexpr (levels == 1) return Half<FromTag>{};
        else if constexpr (levels == 2) return Half<Half<FromTag>>{};
        else return Half<Half<Half<FromTag>>>{};
      }());
      return sve_convert_vector(to, SelectedTag{}, selected);
    } else {
      static_assert(ratio == 2 && phase == 1);
      using SelectedTag = Half<FromTag>;
      return sve_convert_vector(
          to, SelectedTag{}, execute(OddOp{}, from, value));
    }
  } else {
    constexpr bool wraps = option_count<IsWrapOption, Options...> == 1;
    using CompactTag = Rebind<ElementOf<ToTag>, FromTag>;
    const auto compact = sve_convert_vector<wraps>(
        CompactTag{}, from, value);
    auto fallback = sve_conversion_population(
        to, std::forward<Options>(options)...);
    if constexpr (phase == 0) {
      return sve_conversion_insert_even<levels, ToTag, CompactTag>(
          to, compact, fallback);
    } else {
      static_assert(ratio == 2 && phase == 1);
      const auto fallback_even = execute(EvenOp{}, to, fallback);
      return execute(InterleaveOp{}, to, fallback_even, compact);
    }
  }
}

template <VectorTag ToTag>
struct NativeImpl<SVEBackend, ConvertOp, ToTag> {
  template <VectorTag FromTag, typename... Options>
    requires (valid_conversion_options<ToTag, FromTag, Options...>())
  static VECOPS_ALWAYS_INLINE Vec<ToTag> call(
      ConvertOp, ToTag to, FromTag from, Vec<FromTag> value,
      Options&&... options) {
    constexpr bool lane_layout = option_count<IsLaneOption, Options...> == 1;
    constexpr bool wraps = option_count<IsWrapOption, Options...> == 1;
    constexpr bool masked = option_count<IsMaskedOption, Options...> == 1;
    auto converted = [&] {
      if constexpr (lane_layout) {
        return sve_convert_lane_native(
            to, from, value, std::forward<Options>(options)...);
      } else {
        return sve_convert_vector<wraps>(to, from, value);
      }
    }();
    if constexpr (masked) {
      const auto inactive = sve_conversion_population(
          to, std::forward<Options>(options)...);
      const auto& mask = find_option<IsMaskedOption>(
          std::forward<Options>(options)...).value;
      return execute(BlendOp{}, to, inactive, mask, converted);
    } else {
      return converted;
    }
  }

  template <VectorTag FromTag>
    requires (valid_mask_conversion<ToTag, FromTag>())
  static VECOPS_ALWAYS_INLINE Mask<ToTag> call(
      ConvertOp, ToTag to, FromTag from, Mask<FromTag> value) {
    return sve_convert_mask(to, from, value);
  }
};

} // namespace vecops::vec::details

#endif // VECOPS_VEC_DETAILS_SVE_CONVERSION_H
