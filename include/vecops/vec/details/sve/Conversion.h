#ifndef VECOPS_VEC_DETAILS_SVE_CONVERSION_H
#define VECOPS_VEC_DETAILS_SVE_CONVERSION_H

/**
 * @file Conversion.h
 * @brief SVE backend implementations for element type conversion operations.
 */

#include <cstdint>
#include <limits>
#include <type_traits>

#include "vecops/vec/details/Conversion.h"
#include "vecops/vec/details/sve/Basic.h"
#include "vecops/vec/details/sve/Bf16.h"

namespace vecops::vec::details {


/* **************************************************************************** */
//                  Element type conversion implementations                   //
/* **************************************************************************** */

template <typename T>
inline constexpr bool sve_is_signed_integer_element_v =
    std::same_as<T, int8_t> || std::same_as<T, int16_t> ||
    std::same_as<T, int32_t> || std::same_as<T, int64_t>;

template <typename T>
inline constexpr bool sve_is_unsigned_integer_element_v =
    std::same_as<T, uint8_t> || std::same_as<T, uint16_t> ||
    std::same_as<T, uint32_t> || std::same_as<T, uint64_t>;

template <typename T>
inline constexpr bool sve_is_integer_element_v =
    sve_is_signed_integer_element_v<T> || sve_is_unsigned_integer_element_v<T>;

template <typename T>
inline constexpr bool sve_is_conversion_element_v =
    std::same_as<T, bfloat16_t> || std::same_as<T, float16_t> ||
    std::same_as<T, float32_t> || std::same_as<T, float64_t> ||
    sve_is_integer_element_v<T>;

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
        SVEInteger<sizeof(From) * 2, sve_is_signed_integer_element_v<From>>>>;

template <Element To>
using SVENarrowedElement = std::conditional_t<
    std::same_as<To, float32_t>, float32_t,
    std::conditional_t<
        std::same_as<To, float16_t> || std::same_as<To, bfloat16_t>,
        float32_t,
        SVEInteger<sizeof(To) * 2, sve_is_signed_integer_element_v<To>>>>;

template <Element To, Element From, bool Wrap = false, typename Raw>
VECOPS_ALWAYS_INLINE auto sve_convert_one_word_raw(Raw value);

template <Element To, Element From, typename Raw>
VECOPS_ALWAYS_INLINE auto sve_reinterpret_integer(Raw value) {
  static_assert(sve_is_integer_element_v<To> && sve_is_integer_element_v<From>);
  static_assert(sizeof(To) == sizeof(From));
  if constexpr (std::same_as<To, From> ||
                sve_is_signed_integer_element_v<To> ==
                    sve_is_signed_integer_element_v<From>) {
    return value;
  } else if constexpr (sizeof(To) == 1) {
    if constexpr (sve_is_signed_integer_element_v<To>)
      return svreinterpret_s8_u8(value);
    else return svreinterpret_u8_s8(value);
  } else if constexpr (sizeof(To) == 2) {
    if constexpr (sve_is_signed_integer_element_v<To>)
      return svreinterpret_s16_u16(value);
    else return svreinterpret_u16_s16(value);
  } else if constexpr (sizeof(To) == 4) {
    if constexpr (sve_is_signed_integer_element_v<To>)
      return svreinterpret_s32_u32(value);
    else return svreinterpret_u32_s32(value);
  } else {
    if constexpr (sve_is_signed_integer_element_v<To>)
      return svreinterpret_s64_u64(value);
    else return svreinterpret_u64_s64(value);
  }
}

template <Element To, Element From, typename Raw>
VECOPS_ALWAYS_INLINE auto sve_promote_one_step(Raw value) {
  static_assert(sizeof(To) == sizeof(From) * 2);
  using Widened = SVEWidenedElement<From>;
  auto widened = [&]() VECOPS_INLINE_LAMBDA {
    if constexpr (std::same_as<From, bfloat16_t>)
      return sve_bf16_to_f32_lo(value);
    else if constexpr (std::same_as<From, float16_t>)
      return svcvt_f32_f16_x(svptrue_b32(), svzip1_f16(value, value));
    else if constexpr (std::same_as<From, float32_t>)
      return svcvt_f64_f32_x(svptrue_b64(), svzip1_f32(value, value));
    else if constexpr (sve_is_signed_integer_element_v<From>) {
      if constexpr (sizeof(From) == 1) return svunpklo_s16(value);
      else if constexpr (sizeof(From) == 2) return svunpklo_s32(value);
      else return svunpklo_s64(value);
    } else {
      if constexpr (sizeof(From) == 1) return svunpklo_u16(value);
      else if constexpr (sizeof(From) == 2) return svunpklo_u32(value);
      else return svunpklo_u64(value);
    }
  }();
  if constexpr (sve_is_integer_element_v<To> &&
                sve_is_integer_element_v<Widened>) {
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
  static_assert(sve_is_integer_element_v<To> && sve_is_integer_element_v<From>);
  static_assert(sizeof(To) * 2 == sizeof(From));
  if constexpr (Wrap) {
    if constexpr (sizeof(To) == 1) {
      const auto bytes = [&]() VECOPS_INLINE_LAMBDA {
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
      const auto halves = [&]() VECOPS_INLINE_LAMBDA {
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
      const auto words = [&]() VECOPS_INLINE_LAMBDA {
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
  auto same_sign = [&]() VECOPS_INLINE_LAMBDA {
    if constexpr (sve_is_signed_integer_element_v<To>) {
      auto signed_value = [&]() VECOPS_INLINE_LAMBDA {
        if constexpr (sve_is_signed_integer_element_v<From>) return value;
        else return sve_convert_same_size<SVEInteger<sizeof(From), true>, From>(value);
      }();
      const auto clamped = [&]() VECOPS_INLINE_LAMBDA {
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
      auto unsigned_value = [&]() VECOPS_INLINE_LAMBDA {
        if constexpr (sve_is_unsigned_integer_element_v<From>) return value;
        else return sve_convert_same_size<SVEInteger<sizeof(From), false>, From>(value);
      }();
      const auto clamped = [&]() VECOPS_INLINE_LAMBDA {
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
  if constexpr (sve_is_integer_element_v<From> && sve_is_integer_element_v<To>) {
    return sve_demote_integer_one_step<To, From, Wrap>(value);
  } else if constexpr (sizeof(From) == 8) {
    auto narrowed = [&]() VECOPS_INLINE_LAMBDA {
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
          using Mid = SVEInteger<8, sve_is_signed_integer_element_v<From>>;
          return sve_demote_integer_one_step<int32_t, Mid>(value);
        }
      } else if constexpr (std::same_as<To, uint32_t>) {
        if constexpr (std::same_as<From, float64_t>)
          return svcvt_u32_f64_x(svptrue_b64(), value);
        else {
          using Mid = SVEInteger<8, sve_is_signed_integer_element_v<From>>;
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
      using Mid = SVEInteger<4, sve_is_signed_integer_element_v<To>>;
      return sve_demote_integer_one_step<To, Mid>(
          sve_convert_same_size<Mid, From>(value));
    }
  } else {
    static_assert(sizeof(From) == 2);
    using Mid = SVEInteger<2, sve_is_signed_integer_element_v<To>>;
    return sve_demote_integer_one_step<To, Mid>(
        sve_convert_same_size<Mid, From>(value));
  }
}

template <Element To, Element From, bool Wrap, typename Raw>
VECOPS_ALWAYS_INLINE auto sve_convert_one_word_raw(Raw value) {
  static_assert(
      sve_is_conversion_element_v<To> && sve_is_conversion_element_v<From>,
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

#if defined(HAS_SVE2)
template <int Phase, Element To, Element From, typename Raw>
VECOPS_ALWAYS_INLINE auto sve_convert_lane_ratio2_widen_raw(Raw value) {
  static_assert(Phase == 0 || Phase == 1);
  static_assert(sizeof(To) == sizeof(From) * 2);
  using Widened = SVEWidenedElement<From>;

  // The widening FP-to-integer FCVT forms have the same bottom/even lane
  // mapping as FP-to-FP FCVT, so phase 0 needs no intermediate FP vector.
  if constexpr (Phase == 0 && std::same_as<From, float16_t> &&
                std::same_as<To, int32_t>)
    return svcvt_s32_f16_x(svptrue_b32(), value);
  else if constexpr (Phase == 0 && std::same_as<From, float16_t> &&
                     std::same_as<To, uint32_t>)
    return svcvt_u32_f16_x(svptrue_b32(), value);
  else if constexpr (Phase == 0 && std::same_as<From, float32_t> &&
                     std::same_as<To, int64_t>)
    return svcvt_s64_f32_x(svptrue_b64(), value);
  else if constexpr (Phase == 0 && std::same_as<From, float32_t> &&
                     std::same_as<To, uint64_t>)
    return svcvt_u64_f32_x(svptrue_b64(), value);

  // FCVT/FCVTLT and SHLLB/SHLLT already implement the public lane layout:
  // phase 0 consumes source lanes 2*i and phase 1 consumes 2*i+1.
  const auto widened = [&]() VECOPS_INLINE_LAMBDA {
    if constexpr (std::same_as<From, bfloat16_t>) {
      const auto bits = svreinterpret_u16_bf16(value);
      const auto widened_bits = [&]() VECOPS_INLINE_LAMBDA {
        if constexpr (Phase == 0) return svshllb_n_u32(bits, 0);
        else return svshllt_n_u32(bits, 0);
      }();
      const auto shifted =
          svlsl_n_u32_x(svptrue_b32(), widened_bits, 16);
      return svreinterpret_f32_u32(shifted);
    } else if constexpr (std::same_as<From, float16_t>) {
      if constexpr (Phase == 0)
        return svcvt_f32_f16_x(svptrue_b32(), value);
      else
        return svcvtlt_f32_f16_x(svptrue_b32(), value);
    } else if constexpr (std::same_as<From, float32_t>) {
      if constexpr (Phase == 0)
        return svcvt_f64_f32_x(svptrue_b64(), value);
      else
        return svcvtlt_f64_f32_x(svptrue_b64(), value);
    } else if constexpr (sve_is_signed_integer_element_v<From>) {
      if constexpr (sizeof(From) == 1) {
        if constexpr (Phase == 0) return svshllb_n_s16(value, 0);
        else return svshllt_n_s16(value, 0);
      } else if constexpr (sizeof(From) == 2) {
        if constexpr (Phase == 0) return svshllb_n_s32(value, 0);
        else return svshllt_n_s32(value, 0);
      } else {
        if constexpr (Phase == 0) return svshllb_n_s64(value, 0);
        else return svshllt_n_s64(value, 0);
      }
    } else {
      static_assert(sve_is_unsigned_integer_element_v<From>);
      if constexpr (sizeof(From) == 1) {
        if constexpr (Phase == 0) return svshllb_n_u16(value, 0);
        else return svshllt_n_u16(value, 0);
      } else if constexpr (sizeof(From) == 2) {
        if constexpr (Phase == 0) return svshllb_n_u32(value, 0);
        else return svshllt_n_u32(value, 0);
      } else {
        if constexpr (Phase == 0) return svshllb_n_u64(value, 0);
        else return svshllt_n_u64(value, 0);
      }
    }
  }();

  if constexpr (sve_is_integer_element_v<To> &&
                sve_is_integer_element_v<Widened>)
    return sve_reinterpret_integer<To, Widened>(widened);
  else
    return sve_convert_same_size<To, Widened>(widened);
}

template <Element To, Element From, typename Raw>
VECOPS_ALWAYS_INLINE auto sve_lane_qxt_bottom(Raw value) {
  static_assert(sve_is_integer_element_v<To> &&
                sve_is_integer_element_v<From>);
  static_assert(sizeof(To) * 2 == sizeof(From));
  if constexpr (std::is_signed_v<From> && std::is_signed_v<To>) {
    if constexpr (sizeof(From) == 2) return svqxtnb_s16(value);
    else if constexpr (sizeof(From) == 4) return svqxtnb_s32(value);
    else return svqxtnb_s64(value);
  } else if constexpr (std::is_unsigned_v<From> &&
                       std::is_unsigned_v<To>) {
    if constexpr (sizeof(From) == 2) return svqxtnb_u16(value);
    else if constexpr (sizeof(From) == 4) return svqxtnb_u32(value);
    else return svqxtnb_u64(value);
  } else if constexpr (std::is_signed_v<From>) {
    if constexpr (sizeof(From) == 2) return svqxtunb_s16(value);
    else if constexpr (sizeof(From) == 4) return svqxtunb_s32(value);
    else return svqxtunb_s64(value);
  } else {
    constexpr From high = static_cast<From>(std::numeric_limits<To>::max());
    const auto clamped = [&]() VECOPS_INLINE_LAMBDA {
      if constexpr (sizeof(From) == 2)
        return svmin_n_u16_x(svptrue_b16(), value, high);
      else if constexpr (sizeof(From) == 4)
        return svmin_n_u32_x(svptrue_b32(), value, high);
      else
        return svmin_n_u64_x(svptrue_b64(), value, high);
    }();
    if constexpr (sizeof(From) == 2)
      return svreinterpret_s8_u8(svqxtnb_u16(clamped));
    else if constexpr (sizeof(From) == 4)
      return svreinterpret_s16_u16(svqxtnb_u32(clamped));
    else
      return svreinterpret_s32_u32(svqxtnb_u64(clamped));
  }
}

template <Element To, Element From, typename NarrowRaw, typename Raw>
VECOPS_ALWAYS_INLINE auto sve_lane_qxt_top(NarrowRaw fallback, Raw value) {
  static_assert(sve_is_integer_element_v<To> &&
                sve_is_integer_element_v<From>);
  static_assert(sizeof(To) * 2 == sizeof(From));
  if constexpr (std::is_signed_v<From> && std::is_signed_v<To>) {
    if constexpr (sizeof(From) == 2) return svqxtnt_s16(fallback, value);
    else if constexpr (sizeof(From) == 4) return svqxtnt_s32(fallback, value);
    else return svqxtnt_s64(fallback, value);
  } else if constexpr (std::is_unsigned_v<From> &&
                       std::is_unsigned_v<To>) {
    if constexpr (sizeof(From) == 2) return svqxtnt_u16(fallback, value);
    else if constexpr (sizeof(From) == 4) return svqxtnt_u32(fallback, value);
    else return svqxtnt_u64(fallback, value);
  } else if constexpr (std::is_signed_v<From>) {
    if constexpr (sizeof(From) == 2) return svqxtunt_s16(fallback, value);
    else if constexpr (sizeof(From) == 4) return svqxtunt_s32(fallback, value);
    else return svqxtunt_s64(fallback, value);
  } else {
    constexpr From high = static_cast<From>(std::numeric_limits<To>::max());
    const auto clamped = [&]() VECOPS_INLINE_LAMBDA {
      if constexpr (sizeof(From) == 2)
        return svmin_n_u16_x(svptrue_b16(), value, high);
      else if constexpr (sizeof(From) == 4)
        return svmin_n_u32_x(svptrue_b32(), value, high);
      else
        return svmin_n_u64_x(svptrue_b64(), value, high);
    }();
    if constexpr (sizeof(From) == 2)
      return svreinterpret_s8_u8(svqxtnt_u16(
          svreinterpret_u8_s8(fallback), clamped));
    else if constexpr (sizeof(From) == 4)
      return svreinterpret_s16_u16(svqxtnt_u32(
          svreinterpret_u16_s16(fallback), clamped));
    else
      return svreinterpret_s32_u32(svqxtnt_u64(
          svreinterpret_u32_s32(fallback), clamped));
  }
}

template <int Phase, Element To, Element From, typename Raw,
          typename FallbackRaw>
VECOPS_ALWAYS_INLINE auto sve_convert_lane_ratio2_narrow_raw(
    Raw value, FallbackRaw fallback) {
  static_assert(Phase == 0 || Phase == 1);
  static_assert(sizeof(To) * 2 == sizeof(From));

  if constexpr (std::same_as<To, float32_t>) {
    const auto wide = sve_convert_same_size<float64_t, From>(value);
    if constexpr (Phase == 0)
      return svcvt_f32_f64_m(fallback, svptrue_b64(), wide);
    else
      return svcvtnt_f32_f64_m(fallback, svptrue_b64(), wide);
  } else if constexpr (std::same_as<To, float16_t>) {
    const auto wide = sve_convert_same_size<float32_t, From>(value);
    if constexpr (Phase == 0)
      return svcvt_f16_f32_m(fallback, svptrue_b32(), wide);
    else
      return svcvtnt_f16_f32_m(fallback, svptrue_b32(), wide);
  } else if constexpr (std::same_as<To, bfloat16_t>) {
    const auto wide = sve_convert_same_size<float32_t, From>(value);
#if defined(__ARM_FEATURE_SVE_BF16) && !defined(VECOPS_PRESERVE_SUBNORMALS)
    if constexpr (Phase == 0)
      return svcvt_bf16_f32_m(fallback, svptrue_b32(), wide);
    else
      return svcvtnt_bf16_f32_m(fallback, svptrue_b32(), wide);
#else
    const auto compact = sve_f32_to_bf16_lo(wide);
    const auto compact_bits = svreinterpret_u16_bf16(compact);
    const auto fallback_bits = svreinterpret_u16_bf16(fallback);
    if constexpr (Phase == 0)
      return svreinterpret_bf16_u16(svzip1_u16(
          compact_bits, svuzp2_u16(fallback_bits, fallback_bits)));
    else
      return svreinterpret_bf16_u16(svzip1_u16(
          svuzp1_u16(fallback_bits, fallback_bits), compact_bits));
#endif
  } else {
    static_assert(sve_is_integer_element_v<To>);
    if constexpr (sve_is_integer_element_v<From>) {
      if constexpr (Phase == 0)
        return sve_lane_qxt_bottom<To, From>(value);
      else
        return sve_lane_qxt_top<To, From>(fallback, value);
    } else {
      using Converted = SVEInteger<sizeof(From), std::is_signed_v<To>>;
      const auto converted = sve_convert_same_size<Converted, From>(value);
      if constexpr (Phase == 0) {
        const auto bottom = sve_lane_qxt_bottom<To, Converted>(converted);
        // Clang can fold FP-to-integer plus QXTNB into a narrowing FCVT whose
        // inactive top half is a sign/zero extension, not the public lane
        // population. Select at the source granularity to retain only the
        // bottom destination lane and restore the requested fallback above it.
        const auto bottom_lanes = [] {
          if constexpr (sizeof(From) == 2) return svptrue_b16();
          else if constexpr (sizeof(From) == 4) return svptrue_b32();
          else return svptrue_b64();
        }();
        if constexpr (std::same_as<To, int8_t>)
          return svsel_s8(bottom_lanes, bottom, fallback);
        else if constexpr (std::same_as<To, uint8_t>)
          return svsel_u8(bottom_lanes, bottom, fallback);
        else if constexpr (std::same_as<To, int16_t>)
          return svsel_s16(bottom_lanes, bottom, fallback);
        else if constexpr (std::same_as<To, uint16_t>)
          return svsel_u16(bottom_lanes, bottom, fallback);
        else if constexpr (std::same_as<To, int32_t>)
          return svsel_s32(bottom_lanes, bottom, fallback);
        else
          return svsel_u32(bottom_lanes, bottom, fallback);
      } else {
        return sve_lane_qxt_top<To, Converted>(fallback, converted);
      }
    }
  }
}
#endif

template <VectorTag ToTag, VectorTag FromTag, typename... Options>
VECOPS_ALWAYS_INLINE Vec<ToTag> sve_convert_lane_native(
    ToTag to, FromTag from, Vec<FromTag> value, Options&&... options) {
  constexpr bool narrows =
      sizeof(ElementOf<FromTag>) > sizeof(ElementOf<ToTag>);
  constexpr int ratio = narrows
      ? static_cast<int>(sizeof(ElementOf<FromTag>) / sizeof(ElementOf<ToTag>))
      : static_cast<int>(sizeof(ElementOf<ToTag>) / sizeof(ElementOf<FromTag>));
  constexpr int levels = ratio == 2 ? 1 : ratio == 4 ? 2 : 3;
  constexpr int phase = conversion_lane_phase<Options...>();

#if defined(HAS_SVE2)
  constexpr bool wraps = option_count_v<IsWrapOption, Options...> == 1;
  constexpr bool nonzero_population =
      option_count_v<IsVectorMergeOption, Options...> == 1 ||
      option_count_v<IsScalarMergeOption, Options...> == 1;
  if constexpr (
      ratio == 2 && !wraps &&
      (!narrows || phase == 1 || !nonzero_population)) {
    auto fallback = [&]() VECOPS_INLINE_LAMBDA {
      if constexpr (narrows)
        return conversion_population<SVEBackend>(
            to, std::forward<Options>(options)...);
      else
        return value;
    }();
    return construct_words<SVEBackend>(
        to, [&]<nint_t Index>(auto) VECOPS_INLINE_LAMBDA {
          const auto source_word = ::vecops::vec::get_word<Index>(from, value);
          if constexpr (!narrows) {
            return sve_basic_wrap_word<ToTag>(
                sve_convert_lane_ratio2_widen_raw<
                    phase, ElementOf<ToTag>, ElementOf<FromTag>>(
                    sve_basic_raw_word(source_word)));
          } else {
            const auto fallback_word =
                ::vecops::vec::get_word<Index>(to, fallback);
            return sve_basic_wrap_word<ToTag>(
                sve_convert_lane_ratio2_narrow_raw<
                    phase, ElementOf<ToTag>, ElementOf<FromTag>>(
                    sve_basic_raw_word(source_word),
                    sve_basic_raw_word(fallback_word)));
          }
        });
  }
#endif

  if constexpr (!narrows) {
    if constexpr (phase == 0) {
      const auto selected = conversion_select_even<levels>(from, value);
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
    constexpr bool wraps_fallback =
        option_count_v<IsWrapOption, Options...> == 1;
    using CompactTag = Rebind<ElementOf<ToTag>, FromTag>;
    const auto compact = sve_convert_vector<wraps_fallback>(
        CompactTag{}, from, value);
    auto fallback = conversion_population<SVEBackend>(
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

template <bool Wrap = false, VectorTag ToTag, VectorTag FromTag>
VECOPS_ALWAYS_INLINE Vec<ToTag> sve_convert_unordered_ratio2(
    ToTag to, FromTag from, Vec<FromTag> value) {
  constexpr bool widens =
      sizeof(ElementOf<ToTag>) == sizeof(ElementOf<FromTag>) * 2;
  static_assert(
      widens ||
      sizeof(ElementOf<FromTag>) == sizeof(ElementOf<ToTag>) * 2);

  if constexpr (widens) {
    static_assert(!Wrap);
    if constexpr (num_words(from) == 1) {
      static_assert(num_words(to) == 2);
      using ToHalf = Half<ToTag>;
      const auto bottom = sve_convert_lane_native(
          ToHalf{}, from, value, cvt::lane<0>);
      const auto top = sve_convert_lane_native(
          ToHalf{}, from, value, cvt::lane<1>);
      return execute(ConcatOp{}, to, bottom, top);
    } else {
      using ToHalf = Half<ToTag>;
      using FromHalf = Half<FromTag>;
      const auto lower = sve_convert_unordered_ratio2(
          ToHalf{}, FromHalf{}, execute(LowerOp{}, from, value));
      const auto upper = sve_convert_unordered_ratio2(
          ToHalf{}, FromHalf{}, execute(UpperOp{}, from, value));
      return execute(ConcatOp{}, to, lower, upper);
    }
  } else {
    if constexpr (num_words(to) == 1) {
      static_assert(num_words(from) == 2);
      using FromHalf = Half<FromTag>;
      const auto lower = execute(LowerOp{}, from, value);
      const auto upper = execute(UpperOp{}, from, value);
      if constexpr (Wrap) {
        const auto bottom = sve_convert_lane_native(
            to, FromHalf{}, lower, cvt::lane<0>, cvt::wrap, opt::zero);
        return sve_convert_lane_native(
            to, FromHalf{}, upper, cvt::lane<1>, cvt::wrap,
            opt::merge(bottom));
      } else {
        const auto bottom = sve_convert_lane_native(
            to, FromHalf{}, lower, cvt::lane<0>, opt::zero);
        return sve_convert_lane_native(
            to, FromHalf{}, upper, cvt::lane<1>, opt::merge(bottom));
      }
    } else {
      using ToHalf = Half<ToTag>;
      using FromHalf = Half<FromTag>;
      const auto lower = sve_convert_unordered_ratio2<Wrap>(
          ToHalf{}, FromHalf{}, execute(LowerOp{}, from, value));
      const auto upper = sve_convert_unordered_ratio2<Wrap>(
          ToHalf{}, FromHalf{}, execute(UpperOp{}, from, value));
      return execute(ConcatOp{}, to, lower, upper);
    }
  }
}

template <bool Wrap = false, VectorTag ToTag, VectorTag FromTag>
VECOPS_ALWAYS_INLINE Vec<ToTag> sve_convert_unordered(
    ToTag to, FromTag from, Vec<FromTag> value) {
  using To = ElementOf<ToTag>;
  using From = ElementOf<FromTag>;
  constexpr bool same_width = sizeof(To) == sizeof(From);
  constexpr bool widens = sizeof(To) > sizeof(From);
  constexpr bool ratio2 =
      sizeof(To) == sizeof(From) * 2 || sizeof(From) == sizeof(To) * 2;

  if constexpr (same_width) {
    // Equal-width unordered conversion is ordered by contract.
    return sve_convert_vector<Wrap>(to, from, value);
  } else if constexpr (ratio2) {
    constexpr bool ratio2_word_shape = widens
        ? num_words(to) == num_words(from) * 2
        : num_words(from) == num_words(to) * 2;
    if constexpr (ratio2_word_shape) {
      return sve_convert_unordered_ratio2<Wrap>(to, from, value);
    } else {
      // An adjacent conversion without a 2:1 physical-word shape has no
      // independent top/bottom word phases. Make this edge ordered; wider
      // unordered conversions compose it and therefore stay path-independent.
      return sve_convert_vector<Wrap>(to, from, value);
    }
  } else if constexpr (widens) {
    static_assert(!Wrap);
    using Mid = SVEWidenedElement<From>;
    using MidTag = Rebind<Mid, FromTag>;
    const auto middle = sve_convert_unordered(
        MidTag{}, from, value);
    return sve_convert_unordered(to, MidTag{}, middle);
  } else {
    using Mid = SVENarrowedElement<To>;
    using MidTag = Rebind<Mid, FromTag>;
    const auto middle = sve_convert_unordered<Wrap>(
        MidTag{}, from, value);
    return sve_convert_unordered<Wrap>(to, MidTag{}, middle);
  }
}

template <VectorTag ToTag>
struct NativeImpl<SVEBackend, ConvertOp, ToTag> {
  template <VectorTag FromTag, typename... Options>
    requires (valid_conversion_options<ToTag, FromTag, Options...>())
  static VECOPS_ALWAYS_INLINE Vec<ToTag> call(
      ConvertOp, ToTag to, FromTag from, Vec<FromTag> value,
      Options&&... options) {
    constexpr bool lane_layout = option_count_v<IsLaneOption, Options...> == 1;
    constexpr bool unordered_layout =
        option_count_v<IsUnorderedOption, Options...> == 1;
    constexpr bool wraps = option_count_v<IsWrapOption, Options...> == 1;
    constexpr bool masked = option_count_v<IsMaskedOption, Options...> == 1;
    auto converted = [&]() VECOPS_INLINE_LAMBDA {
      if constexpr (lane_layout) {
        return sve_convert_lane_native(
            to, from, value, std::forward<Options>(options)...);
      } else if constexpr (unordered_layout) {
        return sve_convert_unordered<wraps>(to, from, value);
      } else {
        return sve_convert_vector<wraps>(to, from, value);
      }
    }();
    if constexpr (masked) {
      const auto inactive = conversion_population<SVEBackend>(
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
