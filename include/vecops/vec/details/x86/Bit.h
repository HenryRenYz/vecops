// SPDX-FileCopyrightText: 2026 HenryRenYz and vecops contributors
// SPDX-License-Identifier: MIT
#ifndef VECOPS_VEC_DETAILS_X86_BIT_H
#define VECOPS_VEC_DETAILS_X86_BIT_H

/**
 * @file Bit.h
 * @brief x86 backend implementations for bitwise operations, shifts, bit
 * counts, and rotations.
 */

#include <array>
#include <cstring>
#include <limits>
#include <type_traits>

#include "vecops/vec/details/x86/Basic.h"

namespace vecops::vec::details {


/* **************************************************************************** */
//                   Bitwise operation word implementations                   //
/* **************************************************************************** */

template <typename Raw>
VECOPS_ALWAYS_INLINE Raw x86_bit_set1_i8(int value) {
  if constexpr (sizeof(Raw) == 16) return _mm_set1_epi8(static_cast<char>(value));
#if VEC_WIDTH >= 256
  else if constexpr (sizeof(Raw) == 32)
    return _mm256_set1_epi8(static_cast<char>(value));
#endif
#if VEC_WIDTH >= 512
  else return _mm512_set1_epi8(static_cast<char>(value));
#endif
}

template <nint_t Count, typename T, typename Raw>
VECOPS_ALWAYS_INLINE Raw x86_bit_shift_left_immediate_raw(Raw value) {
  using U = std::make_unsigned_t<T>;
  constexpr int width = std::numeric_limits<U>::digits;
  static_assert(Count >= 0);
  if constexpr (Count >= width) {
    return x86_bit_zero_raw<Raw>();
  } else if constexpr (sizeof(T) == 1) {
    const auto shifted = [&] {
      if constexpr (sizeof(Raw) == 16)
        return _mm_slli_epi16(value, static_cast<int>(Count));
#if VEC_WIDTH >= 256
      else if constexpr (sizeof(Raw) == 32)
        return _mm256_slli_epi16(value, static_cast<int>(Count));
#endif
#if VEC_WIDTH >= 512
      else return _mm512_slli_epi16(value, static_cast<int>(Count));
#endif
    }();
    constexpr int mask = static_cast<int>(0xffu << Count);
    return x86_bit_and_raw(shifted, x86_bit_set1_i8<Raw>(mask));
  } else if constexpr (sizeof(T) == 2) {
    if constexpr (sizeof(Raw) == 16)
      return _mm_slli_epi16(value, static_cast<int>(Count));
#if VEC_WIDTH >= 256
    else if constexpr (sizeof(Raw) == 32)
      return _mm256_slli_epi16(value, static_cast<int>(Count));
#endif
#if VEC_WIDTH >= 512
    else return _mm512_slli_epi16(value, static_cast<int>(Count));
#endif
  } else if constexpr (sizeof(T) == 4) {
    if constexpr (sizeof(Raw) == 16)
      return _mm_slli_epi32(value, static_cast<int>(Count));
#if VEC_WIDTH >= 256
    else if constexpr (sizeof(Raw) == 32)
      return _mm256_slli_epi32(value, static_cast<int>(Count));
#endif
#if VEC_WIDTH >= 512
    else return _mm512_slli_epi32(value, static_cast<int>(Count));
#endif
  } else {
    if constexpr (sizeof(Raw) == 16)
      return _mm_slli_epi64(value, static_cast<int>(Count));
#if VEC_WIDTH >= 256
    else if constexpr (sizeof(Raw) == 32)
      return _mm256_slli_epi64(value, static_cast<int>(Count));
#endif
#if VEC_WIDTH >= 512
    else return _mm512_slli_epi64(value, static_cast<int>(Count));
#endif
  }
}

template <typename T, typename Raw>
VECOPS_ALWAYS_INLINE Raw x86_bit_shift_left_scalar_raw(
    Raw value, int count) {
  const auto shift = _mm_cvtsi32_si128(count);
  if constexpr (sizeof(T) == 1) {
    if (count >= 8) return x86_bit_zero_raw<Raw>();
    const auto shifted = [&] {
      if constexpr (sizeof(Raw) == 16) return _mm_sll_epi16(value, shift);
#if VEC_WIDTH >= 256
      else if constexpr (sizeof(Raw) == 32) return _mm256_sll_epi16(value, shift);
#endif
#if VEC_WIDTH >= 512
      else return _mm512_sll_epi16(value, shift);
#endif
    }();
    const auto byte_mask = x86_bit_set1_i8<Raw>(0xff << count);
    return x86_bit_and_raw(shifted, byte_mask);
  } else if constexpr (sizeof(T) == 2) {
    if constexpr (sizeof(Raw) == 16) return _mm_sll_epi16(value, shift);
#if VEC_WIDTH >= 256
    else if constexpr (sizeof(Raw) == 32) return _mm256_sll_epi16(value, shift);
#endif
#if VEC_WIDTH >= 512
    else return _mm512_sll_epi16(value, shift);
#endif
  } else if constexpr (sizeof(T) == 4) {
    if constexpr (sizeof(Raw) == 16) return _mm_sll_epi32(value, shift);
#if VEC_WIDTH >= 256
    else if constexpr (sizeof(Raw) == 32) return _mm256_sll_epi32(value, shift);
#endif
#if VEC_WIDTH >= 512
    else return _mm512_sll_epi32(value, shift);
#endif
  } else if constexpr (sizeof(Raw) == 16) return _mm_sll_epi64(value, shift);
#if VEC_WIDTH >= 256
  else if constexpr (sizeof(Raw) == 32) return _mm256_sll_epi64(value, shift);
#endif
#if VEC_WIDTH >= 512
  else return _mm512_sll_epi64(value, shift);
#endif
}

template <nint_t Count, typename T, typename Raw>
VECOPS_ALWAYS_INLINE Raw x86_bit_shift_right_logical_immediate_raw(
    Raw value) {
  using U = std::make_unsigned_t<T>;
  constexpr int width = std::numeric_limits<U>::digits;
  static_assert(Count >= 0);
  if constexpr (Count >= width) {
    return x86_bit_zero_raw<Raw>();
  } else if constexpr (sizeof(T) == 1) {
    const auto shifted = [&] {
      if constexpr (sizeof(Raw) == 16)
        return _mm_srli_epi16(value, static_cast<int>(Count));
#if VEC_WIDTH >= 256
      else if constexpr (sizeof(Raw) == 32)
        return _mm256_srli_epi16(value, static_cast<int>(Count));
#endif
#if VEC_WIDTH >= 512
      else return _mm512_srli_epi16(value, static_cast<int>(Count));
#endif
    }();
    constexpr int mask = static_cast<int>((0xffu >> Count) & 0xffu);
    return x86_bit_and_raw(shifted, x86_bit_set1_i8<Raw>(mask));
  } else if constexpr (sizeof(T) == 2) {
    if constexpr (sizeof(Raw) == 16)
      return _mm_srli_epi16(value, static_cast<int>(Count));
#if VEC_WIDTH >= 256
    else if constexpr (sizeof(Raw) == 32)
      return _mm256_srli_epi16(value, static_cast<int>(Count));
#endif
#if VEC_WIDTH >= 512
    else return _mm512_srli_epi16(value, static_cast<int>(Count));
#endif
  } else if constexpr (sizeof(T) == 4) {
    if constexpr (sizeof(Raw) == 16)
      return _mm_srli_epi32(value, static_cast<int>(Count));
#if VEC_WIDTH >= 256
    else if constexpr (sizeof(Raw) == 32)
      return _mm256_srli_epi32(value, static_cast<int>(Count));
#endif
#if VEC_WIDTH >= 512
    else return _mm512_srli_epi32(value, static_cast<int>(Count));
#endif
  } else {
    if constexpr (sizeof(Raw) == 16)
      return _mm_srli_epi64(value, static_cast<int>(Count));
#if VEC_WIDTH >= 256
    else if constexpr (sizeof(Raw) == 32)
      return _mm256_srli_epi64(value, static_cast<int>(Count));
#endif
#if VEC_WIDTH >= 512
    else return _mm512_srli_epi64(value, static_cast<int>(Count));
#endif
  }
}

template <typename T, typename Raw>
VECOPS_ALWAYS_INLINE Raw x86_bit_shift_right_logical_scalar_raw(
    Raw value, int count) {
  const auto shift = _mm_cvtsi32_si128(count);
  if constexpr (sizeof(T) == 1) {
    if (count >= 8) return x86_bit_zero_raw<Raw>();
    const auto shifted = [&] {
      if constexpr (sizeof(Raw) == 16) return _mm_srl_epi16(value, shift);
#if VEC_WIDTH >= 256
      else if constexpr (sizeof(Raw) == 32) return _mm256_srl_epi16(value, shift);
#endif
#if VEC_WIDTH >= 512
      else return _mm512_srl_epi16(value, shift);
#endif
    }();
    const auto byte_mask = x86_bit_set1_i8<Raw>((0xffu >> count) & 0xffu);
    return x86_bit_and_raw(shifted, byte_mask);
  } else if constexpr (sizeof(T) == 2) {
    if constexpr (sizeof(Raw) == 16) return _mm_srl_epi16(value, shift);
#if VEC_WIDTH >= 256
    else if constexpr (sizeof(Raw) == 32) return _mm256_srl_epi16(value, shift);
#endif
#if VEC_WIDTH >= 512
    else return _mm512_srl_epi16(value, shift);
#endif
  } else if constexpr (sizeof(T) == 4) {
    if constexpr (sizeof(Raw) == 16) return _mm_srl_epi32(value, shift);
#if VEC_WIDTH >= 256
    else if constexpr (sizeof(Raw) == 32) return _mm256_srl_epi32(value, shift);
#endif
#if VEC_WIDTH >= 512
    else return _mm512_srl_epi32(value, shift);
#endif
  } else if constexpr (sizeof(Raw) == 16) return _mm_srl_epi64(value, shift);
#if VEC_WIDTH >= 256
  else if constexpr (sizeof(Raw) == 32) return _mm256_srl_epi64(value, shift);
#endif
#if VEC_WIDTH >= 512
  else return _mm512_srl_epi64(value, shift);
#endif
}

template <typename T, typename Raw>
VECOPS_ALWAYS_INLINE Raw x86_bit_negative_mask(Raw value) {
  const auto zero = x86_bit_zero_raw<Raw>();
  if constexpr (sizeof(T) == 1) {
    if constexpr (sizeof(Raw) == 16) return _mm_cmpgt_epi8(zero, value);
#if VEC_WIDTH >= 256
    else if constexpr (sizeof(Raw) == 32) return _mm256_cmpgt_epi8(zero, value);
#endif
#if VEC_WIDTH >= 512
    else return _mm512_movm_epi8(_mm512_cmpgt_epi8_mask(zero, value));
#endif
  } else if constexpr (sizeof(T) == 2) {
    if constexpr (sizeof(Raw) == 16) return _mm_cmpgt_epi16(zero, value);
#if VEC_WIDTH >= 256
    else if constexpr (sizeof(Raw) == 32) return _mm256_cmpgt_epi16(zero, value);
#endif
#if VEC_WIDTH >= 512
    else return _mm512_movm_epi16(_mm512_cmpgt_epi16_mask(zero, value));
#endif
  } else {
    const auto sign32 = [&] {
    if constexpr (sizeof(Raw) == 16) return _mm_srai_epi32(value, 31);
#if VEC_WIDTH >= 256
    else if constexpr (sizeof(Raw) == 32) return _mm256_srai_epi32(value, 31);
#endif
#if VEC_WIDTH >= 512
    else return _mm512_srai_epi32(value, 31);
#endif
    }();
    if constexpr (sizeof(T) == 4) return sign32;
    else if constexpr (sizeof(Raw) == 16)
      return _mm_shuffle_epi32(sign32, _MM_SHUFFLE(3, 3, 1, 1));
#if VEC_WIDTH >= 256
    else if constexpr (sizeof(Raw) == 32)
      return _mm256_shuffle_epi32(sign32, _MM_SHUFFLE(3, 3, 1, 1));
#endif
#if VEC_WIDTH >= 512
    else return _mm512_shuffle_epi32(
        sign32, static_cast<_MM_PERM_ENUM>(_MM_SHUFFLE(3, 3, 1, 1)));
#endif
  }
}

template <nint_t Count, typename T, typename Raw>
VECOPS_ALWAYS_INLINE Raw x86_bit_shift_right_signed_immediate_raw(
    Raw value) {
  const auto sign = x86_bit_negative_mask<T>(value);
  const auto magnitude = x86_bit_xor_raw(value, sign);
  return x86_bit_xor_raw(
      x86_bit_shift_right_logical_immediate_raw<Count, T>(magnitude), sign);
}

template <typename T, typename Raw>
VECOPS_ALWAYS_INLINE Raw x86_bit_shift_right_signed_scalar_raw(
    Raw value, int count) {
  const auto sign = x86_bit_negative_mask<T>(value);
  const auto magnitude = x86_bit_xor_raw(value, sign);
  return x86_bit_xor_raw(
      x86_bit_shift_right_logical_scalar_raw<T>(magnitude, count), sign);
}

template <typename Op, typename T, typename Raw>
VECOPS_ALWAYS_INLINE Raw x86_bit_shift_variable_raw(
    Raw value, Raw counts) {
#if defined(HAS_AVX2)
  if constexpr (sizeof(T) == 1) {
    // AVX2 has no byte-granularity variable shift. Widen four bytes at a
    // time to dwords, shift there, then pack the low byte of every lane.
    const auto shift128 = [](__m128i bytes, __m128i byte_counts) {
      const auto shift_quad = [&]<int Offset>() {
        const auto source = _mm_srli_si128(bytes, Offset);
        const auto source_counts = _mm_srli_si128(byte_counts, Offset);
        const auto lanes =
            std::is_signed_v<T> && !std::same_as<Op, BitShiftLeftOp>
            ? _mm_cvtepi8_epi32(source)
            : _mm_cvtepu8_epi32(source);
        const auto counts32 = _mm_min_epu32(
            _mm_cvtepu8_epi32(source_counts), _mm_set1_epi32(8));
        const auto shifted = [&] {
          if constexpr (std::same_as<Op, BitShiftLeftOp>)
            return _mm_sllv_epi32(lanes, counts32);
          else if constexpr (std::is_signed_v<T>)
            return _mm_srav_epi32(lanes, counts32);
          else
            return _mm_srlv_epi32(lanes, counts32);
        }();
        return _mm_and_si128(shifted, _mm_set1_epi32(0xff));
      };
      const auto lanes0 = shift_quad.template operator()<0>();
      const auto lanes1 = shift_quad.template operator()<4>();
      const auto lanes2 = shift_quad.template operator()<8>();
      const auto lanes3 = shift_quad.template operator()<12>();
      return _mm_packus_epi16(
          _mm_packus_epi32(lanes0, lanes1),
          _mm_packus_epi32(lanes2, lanes3));
    };
    if constexpr (sizeof(Raw) == 16) {
      return shift128(value, counts);
    } else if constexpr (sizeof(Raw) == 32) {
      return _mm256_set_m128i(
          shift128(
              _mm256_extracti128_si256(value, 1),
              _mm256_extracti128_si256(counts, 1)),
          shift128(
              _mm256_castsi256_si128(value),
              _mm256_castsi256_si128(counts)));
    }
#if VEC_WIDTH >= 512
    else {
      auto result = _mm512_castsi128_si512(shift128(
          _mm512_castsi512_si128(value),
          _mm512_castsi512_si128(counts)));
      result = _mm512_inserti32x4(result, shift128(
          _mm512_extracti32x4_epi32(value, 1),
          _mm512_extracti32x4_epi32(counts, 1)), 1);
      result = _mm512_inserti32x4(result, shift128(
          _mm512_extracti32x4_epi32(value, 2),
          _mm512_extracti32x4_epi32(counts, 2)), 2);
      return _mm512_inserti32x4(result, shift128(
          _mm512_extracti32x4_epi32(value, 3),
          _mm512_extracti32x4_epi32(counts, 3)), 3);
    }
#endif
  } else if constexpr (sizeof(T) == 2) {
    const auto shift_part = []<typename Wide>(Wide lanes, Wide lane_counts) {
      if constexpr (std::same_as<Op, BitShiftLeftOp>)
        return x86_bit_shift_variable_raw<BitShiftLeftOp, std::uint32_t>(
            lanes, lane_counts);
      else if constexpr (std::is_signed_v<T>)
        return x86_bit_shift_variable_raw<BitShiftRightOp, std::int32_t>(
            lanes, lane_counts);
      else
        return x86_bit_shift_variable_raw<BitShiftRightOp, std::uint32_t>(
            lanes, lane_counts);
    };
    if constexpr (sizeof(Raw) == 16) {
      const auto low_counts = _mm_cvtepu16_epi32(counts);
      const auto high_counts =
          _mm_cvtepu16_epi32(_mm_srli_si128(counts, 8));
      const auto low = shift_part(
          std::is_signed_v<T> && !std::same_as<Op, BitShiftLeftOp>
              ? _mm_cvtepi16_epi32(value) : _mm_cvtepu16_epi32(value),
          low_counts);
      const auto high = shift_part(
          std::is_signed_v<T> && !std::same_as<Op, BitShiftLeftOp>
              ? _mm_cvtepi16_epi32(_mm_srli_si128(value, 8))
              : _mm_cvtepu16_epi32(_mm_srli_si128(value, 8)),
          high_counts);
      if constexpr (
          std::is_signed_v<T> && !std::same_as<Op, BitShiftLeftOp>) {
        return _mm_packs_epi32(low, high);
      } else {
        const auto low16 = _mm_set1_epi32(0xffff);
        return _mm_packus_epi32(
            _mm_and_si128(low, low16), _mm_and_si128(high, low16));
      }
    } else if constexpr (sizeof(Raw) == 32) {
      const auto low_value = _mm256_castsi256_si128(value);
      const auto high_value = _mm256_extracti128_si256(value, 1);
      const auto low_counts = _mm256_castsi256_si128(counts);
      const auto high_counts = _mm256_extracti128_si256(counts, 1);
      const auto low = shift_part(
          std::is_signed_v<T> && !std::same_as<Op, BitShiftLeftOp>
              ? _mm256_cvtepi16_epi32(low_value)
              : _mm256_cvtepu16_epi32(low_value),
          _mm256_cvtepu16_epi32(low_counts));
      const auto high = shift_part(
          std::is_signed_v<T> && !std::same_as<Op, BitShiftLeftOp>
              ? _mm256_cvtepi16_epi32(high_value)
              : _mm256_cvtepu16_epi32(high_value),
          _mm256_cvtepu16_epi32(high_counts));
      const auto packed = [&] {
        if constexpr (
            std::is_signed_v<T> && !std::same_as<Op, BitShiftLeftOp>) {
          return _mm256_packs_epi32(low, high);
        } else {
          const auto low16 = _mm256_set1_epi32(0xffff);
          return _mm256_packus_epi32(
              _mm256_and_si256(low, low16),
              _mm256_and_si256(high, low16));
        }
      }();
      return _mm256_permute4x64_epi64(packed, 0xd8);
    }
  } else if constexpr (sizeof(T) == 4) {
    if constexpr (sizeof(Raw) == 16) {
      if constexpr (std::same_as<Op, BitShiftLeftOp>)
        return _mm_sllv_epi32(value, counts);
      else if constexpr (std::is_signed_v<T>)
        return _mm_srav_epi32(value, counts);
      else
        return _mm_srlv_epi32(value, counts);
    } else if constexpr (sizeof(Raw) == 32) {
      if constexpr (std::same_as<Op, BitShiftLeftOp>)
        return _mm256_sllv_epi32(value, counts);
      else if constexpr (std::is_signed_v<T>)
        return _mm256_srav_epi32(value, counts);
      else
        return _mm256_srlv_epi32(value, counts);
    }
  } else if constexpr (sizeof(T) == 8) {
    if constexpr (sizeof(Raw) == 16) {
      if constexpr (std::same_as<Op, BitShiftLeftOp>)
        return _mm_sllv_epi64(value, counts);
      else if constexpr (std::is_unsigned_v<T>)
        return _mm_srlv_epi64(value, counts);
      else {
        const auto sign = x86_bit_negative_mask<T>(value);
        return x86_bit_xor_raw(
            _mm_srlv_epi64(x86_bit_xor_raw(value, sign), counts), sign);
      }
    } else if constexpr (sizeof(Raw) == 32) {
      if constexpr (std::same_as<Op, BitShiftLeftOp>)
        return _mm256_sllv_epi64(value, counts);
      else if constexpr (std::is_unsigned_v<T>)
        return _mm256_srlv_epi64(value, counts);
      else {
        const auto sign = x86_bit_negative_mask<T>(value);
        return x86_bit_xor_raw(
            _mm256_srlv_epi64(
                x86_bit_xor_raw(value, sign), counts), sign);
      }
    }
  }
#endif
#if defined(CPU_CAPABILITY_AVX512)
#if defined(HAS_AVX512BW) && defined(HAS_AVX512VL)
  if constexpr (sizeof(Raw) == 16 && sizeof(T) == 2) {
    if constexpr (std::same_as<Op, BitShiftLeftOp>)
      return _mm_sllv_epi16(value, counts);
    else if constexpr (std::is_signed_v<T>)
      return _mm_srav_epi16(value, counts);
    else
      return _mm_srlv_epi16(value, counts);
  } else if constexpr (sizeof(Raw) == 32 && sizeof(T) == 2) {
    if constexpr (std::same_as<Op, BitShiftLeftOp>)
      return _mm256_sllv_epi16(value, counts);
    else if constexpr (std::is_signed_v<T>)
      return _mm256_srav_epi16(value, counts);
    else
      return _mm256_srlv_epi16(value, counts);
  } else
#endif
  if constexpr (sizeof(Raw) == 64 && sizeof(T) == 2) {
    if constexpr (std::same_as<Op, BitShiftLeftOp>)
      return _mm512_sllv_epi16(value, counts);
    else if constexpr (std::is_signed_v<T>)
      return _mm512_srav_epi16(value, counts);
    else
      return _mm512_srlv_epi16(value, counts);
  } else if constexpr (sizeof(Raw) == 64 && sizeof(T) == 4) {
    if constexpr (std::same_as<Op, BitShiftLeftOp>)
      return _mm512_sllv_epi32(value, counts);
    else if constexpr (std::is_signed_v<T>)
      return _mm512_srav_epi32(value, counts);
    else
      return _mm512_srlv_epi32(value, counts);
  } else if constexpr (sizeof(Raw) == 64 && sizeof(T) == 8) {
    if constexpr (std::same_as<Op, BitShiftLeftOp>)
      return _mm512_sllv_epi64(value, counts);
    else if constexpr (std::is_signed_v<T>)
      return _mm512_srav_epi64(value, counts);
    else
      return _mm512_srlv_epi64(value, counts);
  }
#endif
  std::array<T, sizeof(Raw) / sizeof(T)> values{};
  std::array<T, sizeof(Raw) / sizeof(T)> lane_counts{};
  std::memcpy(values.data(), &value, sizeof(Raw));
  std::memcpy(lane_counts.data(), &counts, sizeof(Raw));
  using U = std::make_unsigned_t<T>;
  constexpr U width = std::numeric_limits<U>::digits;
  for (std::size_t lane = 0; lane < values.size(); ++lane) {
    const U lane_count = static_cast<U>(lane_counts[lane]);
    const int count = lane_count >= width
        ? static_cast<int>(width)
        : static_cast<int>(lane_count);
    if constexpr (std::same_as<Op, BitShiftLeftOp>)
      values[lane] = scalar_shift_left(values[lane], count);
    else
      values[lane] = scalar_shift_right(values[lane], count);
  }
  Raw result;
  std::memcpy(&result, values.data(), sizeof(Raw));
  return result;
}

template <typename T, typename Raw>
VECOPS_ALWAYS_INLINE Raw x86_bit_set1_lane(std::make_unsigned_t<T> value) {
  if constexpr (sizeof(T) == 1) {
    if constexpr (sizeof(Raw) == 16)
      return _mm_set1_epi8(static_cast<int8_t>(value));
#if VEC_WIDTH >= 256
    else if constexpr (sizeof(Raw) == 32)
      return _mm256_set1_epi8(static_cast<int8_t>(value));
#endif
#if VEC_WIDTH >= 512
    else return _mm512_set1_epi8(static_cast<int8_t>(value));
#endif
  } else if constexpr (sizeof(T) == 2) {
    if constexpr (sizeof(Raw) == 16)
      return _mm_set1_epi16(static_cast<int16_t>(value));
#if VEC_WIDTH >= 256
    else if constexpr (sizeof(Raw) == 32)
      return _mm256_set1_epi16(static_cast<int16_t>(value));
#endif
#if VEC_WIDTH >= 512
    else return _mm512_set1_epi16(static_cast<int16_t>(value));
#endif
  } else if constexpr (sizeof(T) == 4) {
    if constexpr (sizeof(Raw) == 16)
      return _mm_set1_epi32(static_cast<int32_t>(value));
#if VEC_WIDTH >= 256
    else if constexpr (sizeof(Raw) == 32)
      return _mm256_set1_epi32(static_cast<int32_t>(value));
#endif
#if VEC_WIDTH >= 512
    else return _mm512_set1_epi32(static_cast<int32_t>(value));
#endif
  } else {
    if constexpr (sizeof(Raw) == 16)
      return _mm_set1_epi64x(static_cast<int64_t>(value));
#if VEC_WIDTH >= 256
    else if constexpr (sizeof(Raw) == 32)
      return _mm256_set1_epi64x(static_cast<int64_t>(value));
#endif
#if VEC_WIDTH >= 512
    else return _mm512_set1_epi64(static_cast<int64_t>(value));
#endif
  }
}

template <typename T, typename Raw>
VECOPS_ALWAYS_INLINE Raw x86_bit_add_lanes(Raw a, Raw b) {
  if constexpr (sizeof(Raw) == 16) {
    if constexpr (sizeof(T) == 1) return _mm_add_epi8(a, b);
    else if constexpr (sizeof(T) == 2) return _mm_add_epi16(a, b);
    else if constexpr (sizeof(T) == 4) return _mm_add_epi32(a, b);
    else return _mm_add_epi64(a, b);
  }
#if VEC_WIDTH >= 256
  else if constexpr (sizeof(Raw) == 32) {
    if constexpr (sizeof(T) == 1) return _mm256_add_epi8(a, b);
    else if constexpr (sizeof(T) == 2) return _mm256_add_epi16(a, b);
    else if constexpr (sizeof(T) == 4) return _mm256_add_epi32(a, b);
    else return _mm256_add_epi64(a, b);
  }
#endif
#if VEC_WIDTH >= 512
  else {
    if constexpr (sizeof(T) == 1) return _mm512_add_epi8(a, b);
    else if constexpr (sizeof(T) == 2) return _mm512_add_epi16(a, b);
    else if constexpr (sizeof(T) == 4) return _mm512_add_epi32(a, b);
    else return _mm512_add_epi64(a, b);
  }
#endif
}

template <typename T, typename Raw>
VECOPS_ALWAYS_INLINE Raw x86_bit_sub_lanes(Raw a, Raw b) {
  if constexpr (sizeof(Raw) == 16) {
    if constexpr (sizeof(T) == 1) return _mm_sub_epi8(a, b);
    else if constexpr (sizeof(T) == 2) return _mm_sub_epi16(a, b);
    else if constexpr (sizeof(T) == 4) return _mm_sub_epi32(a, b);
    else return _mm_sub_epi64(a, b);
  }
#if VEC_WIDTH >= 256
  else if constexpr (sizeof(Raw) == 32) {
    if constexpr (sizeof(T) == 1) return _mm256_sub_epi8(a, b);
    else if constexpr (sizeof(T) == 2) return _mm256_sub_epi16(a, b);
    else if constexpr (sizeof(T) == 4) return _mm256_sub_epi32(a, b);
    else return _mm256_sub_epi64(a, b);
  }
#endif
#if VEC_WIDTH >= 512
  else {
    if constexpr (sizeof(T) == 1) return _mm512_sub_epi8(a, b);
    else if constexpr (sizeof(T) == 2) return _mm512_sub_epi16(a, b);
    else if constexpr (sizeof(T) == 4) return _mm512_sub_epi32(a, b);
    else return _mm512_sub_epi64(a, b);
  }
#endif
}

template <typename Raw>
VECOPS_ALWAYS_INLINE Raw x86_bit_popcount_bytes(Raw value) {
#if defined(HAS_SSSE3)
  const auto table128 = _mm_setr_epi8(
      0, 1, 1, 2, 1, 2, 2, 3, 1, 2, 2, 3, 2, 3, 3, 4);
  const auto table = [&] {
    if constexpr (sizeof(Raw) == 16) return table128;
#if VEC_WIDTH >= 256
    else if constexpr (sizeof(Raw) == 32)
      return _mm256_broadcastsi128_si256(table128);
#endif
#if VEC_WIDTH >= 512
    else return _mm512_broadcast_i32x4(table128);
#endif
  }();
  const auto low_mask = x86_bit_set1_lane<uint8_t, Raw>(0x0f);
  const auto low = x86_bit_and_raw(value, low_mask);
  const auto high = x86_bit_and_raw(
      x86_bit_shift_right_logical_immediate_raw<4, uint8_t>(value),
      low_mask);
  if constexpr (sizeof(Raw) == 16)
    return _mm_add_epi8(
        _mm_shuffle_epi8(table, low), _mm_shuffle_epi8(table, high));
#if VEC_WIDTH >= 256
  else if constexpr (sizeof(Raw) == 32)
    return _mm256_add_epi8(
        _mm256_shuffle_epi8(table, low),
        _mm256_shuffle_epi8(table, high));
#endif
#if VEC_WIDTH >= 512
  else return _mm512_add_epi8(
      _mm512_shuffle_epi8(table, low),
      _mm512_shuffle_epi8(table, high));
#endif
#else
  const auto mask55 = x86_bit_set1_lane<uint8_t, Raw>(0x55);
  const auto mask33 = x86_bit_set1_lane<uint8_t, Raw>(0x33);
  const auto mask0f = x86_bit_set1_lane<uint8_t, Raw>(0x0f);
  value = x86_bit_sub_lanes<uint8_t>(
      value,
      x86_bit_and_raw(
          x86_bit_shift_right_logical_immediate_raw<1, uint8_t>(value),
          mask55));
  value = x86_bit_add_lanes<uint8_t>(
      x86_bit_and_raw(value, mask33),
      x86_bit_and_raw(
          x86_bit_shift_right_logical_immediate_raw<2, uint8_t>(value),
          mask33));
  return x86_bit_and_raw(
      x86_bit_add_lanes<uint8_t>(
          value,
          x86_bit_shift_right_logical_immediate_raw<4, uint8_t>(value)),
      mask0f);
#endif
}

template <typename T, typename Raw>
VECOPS_ALWAYS_INLINE Raw x86_bit_popcount_raw(Raw value) {
#if VEC_WIDTH >= 512 && defined(HAS_AVX512BITALG)
  if constexpr (sizeof(Raw) == 64 && sizeof(T) == 1)
    return _mm512_popcnt_epi8(value);
  else if constexpr (sizeof(Raw) == 64 && sizeof(T) == 2)
    return _mm512_popcnt_epi16(value);
  else
#endif
#if VEC_WIDTH >= 512 && defined(HAS_AVX512VPOPCNTDQ)
  if constexpr (sizeof(Raw) == 64 && sizeof(T) == 4)
    return _mm512_popcnt_epi32(value);
  else if constexpr (sizeof(Raw) == 64 && sizeof(T) == 8)
    return _mm512_popcnt_epi64(value);
  else
#endif
  {
    auto count = x86_bit_popcount_bytes(value);
    if constexpr (sizeof(T) >= 2) {
      count = x86_bit_and_raw(
          x86_bit_add_lanes<uint16_t>(
              count,
              x86_bit_shift_right_logical_immediate_raw<8, uint16_t>(
                  count)),
          x86_bit_set1_lane<uint16_t, Raw>(0x00ff));
    }
    if constexpr (sizeof(T) >= 4) {
      count = x86_bit_and_raw(
          x86_bit_add_lanes<uint32_t>(
              count,
              x86_bit_shift_right_logical_immediate_raw<16, uint32_t>(
                  count)),
          x86_bit_set1_lane<uint32_t, Raw>(0x000000ff));
    }
    if constexpr (sizeof(T) >= 8) {
      count = x86_bit_and_raw(
          x86_bit_add_lanes<uint64_t>(
              count,
              x86_bit_shift_right_logical_immediate_raw<32, uint64_t>(
                  count)),
          x86_bit_set1_lane<uint64_t, Raw>(0xff));
    }
    return count;
  }
}

template <typename T, typename Raw>
VECOPS_ALWAYS_INLINE Raw x86_bit_countl_zero_raw(Raw value) {
#if VEC_WIDTH >= 512 && defined(HAS_AVX512CD)
  if constexpr (sizeof(Raw) == 64 && sizeof(T) == 4)
    return _mm512_lzcnt_epi32(value);
  else if constexpr (sizeof(Raw) == 64 && sizeof(T) == 8)
    return _mm512_lzcnt_epi64(value);
  else
#endif
  {
    auto spread = value;
    spread = x86_bit_or_raw(
        spread,
        x86_bit_shift_right_logical_immediate_raw<1, T>(spread));
    spread = x86_bit_or_raw(
        spread,
        x86_bit_shift_right_logical_immediate_raw<2, T>(spread));
    spread = x86_bit_or_raw(
        spread,
        x86_bit_shift_right_logical_immediate_raw<4, T>(spread));
    if constexpr (sizeof(T) >= 2)
      spread = x86_bit_or_raw(
          spread,
          x86_bit_shift_right_logical_immediate_raw<8, T>(spread));
    if constexpr (sizeof(T) >= 4)
      spread = x86_bit_or_raw(
          spread,
          x86_bit_shift_right_logical_immediate_raw<16, T>(spread));
    if constexpr (sizeof(T) >= 8)
      spread = x86_bit_or_raw(
          spread,
          x86_bit_shift_right_logical_immediate_raw<32, T>(spread));
    constexpr auto width = static_cast<std::make_unsigned_t<T>>(
        std::numeric_limits<std::make_unsigned_t<T>>::digits);
    return x86_bit_sub_lanes<T>(
        x86_bit_set1_lane<T, Raw>(width),
        x86_bit_popcount_raw<T>(spread));
  }
}

template <typename Op, typename T, typename Raw>
VECOPS_ALWAYS_INLINE Raw x86_bit_count_raw(Raw value) {
  if constexpr (std::same_as<Op, PopCountOp>) {
    return x86_bit_popcount_raw<T>(value);
  } else if constexpr (std::same_as<Op, CountLeadingZeroOp>) {
    return x86_bit_countl_zero_raw<T>(value);
  } else if constexpr (std::same_as<Op, CountLeadingOneOp>) {
    return x86_bit_countl_zero_raw<T>(
        x86_bit_xor_raw(value, x86_bit_ones_raw<Raw>()));
  } else if constexpr (
      std::same_as<Op, CountTrailingZeroOp> ||
      std::same_as<Op, CountTrailingOneOp>) {
    if constexpr (std::same_as<Op, CountTrailingOneOp>)
      value = x86_bit_xor_raw(value, x86_bit_ones_raw<Raw>());
    return x86_bit_popcount_raw<T>(x86_bit_andnot_raw(
        value,
        x86_bit_sub_lanes<T>(
            value, x86_bit_set1_lane<T, Raw>(1))));
  } else {
    static_assert(dispatch_dependent_false<Op>, "unknown bit-count op");
  }
}

template <typename Op, nint_t Count, typename T, typename Raw>
VECOPS_ALWAYS_INLINE Raw x86_bit_rotate_immediate_raw(Raw value) {
  using U = std::make_unsigned_t<T>;
  constexpr nint_t width = std::numeric_limits<U>::digits;
  constexpr nint_t remainder = Count % width;
  constexpr int rotate = static_cast<int>(
      remainder < 0 ? remainder + width : remainder);
  constexpr int inverse = (-rotate) & (static_cast<int>(width) - 1);
  if constexpr (rotate == 0) return value;
#if VEC_WIDTH >= 512 && defined(HAS_AVX512F)
  else if constexpr (sizeof(Raw) == 64 && sizeof(T) == 4) {
    if constexpr (std::same_as<Op, RotateLeftOp>)
      return _mm512_rol_epi32(value, rotate);
    else return _mm512_ror_epi32(value, rotate);
  } else if constexpr (sizeof(Raw) == 64 && sizeof(T) == 8) {
    if constexpr (std::same_as<Op, RotateLeftOp>)
      return _mm512_rol_epi64(value, rotate);
    else return _mm512_ror_epi64(value, rotate);
  }
#endif
  else if constexpr (std::same_as<Op, RotateLeftOp>) {
    return x86_bit_or_raw(
        x86_bit_shift_left_immediate_raw<rotate, T>(value),
        x86_bit_shift_right_logical_immediate_raw<inverse, T>(value));
  } else {
    return x86_bit_or_raw(
        x86_bit_shift_right_logical_immediate_raw<rotate, T>(value),
        x86_bit_shift_left_immediate_raw<inverse, T>(value));
  }
}

template <typename Op, typename T, typename Raw>
VECOPS_ALWAYS_INLINE Raw x86_bit_rotate_scalar_raw(Raw value, int count) {
  using U = std::make_unsigned_t<T>;
  constexpr unsigned width = std::numeric_limits<U>::digits;
  const int rotate = static_cast<int>(
      static_cast<unsigned>(count) & (width - 1));
  const int inverse = (-rotate) & (width - 1);
  if (rotate == 0) return value;
#if VEC_WIDTH >= 512 && defined(HAS_AVX512F)
  if constexpr (sizeof(Raw) == 64 && sizeof(T) == 4) {
    const auto counts = x86_bit_set1_lane<T, Raw>(static_cast<U>(rotate));
    if constexpr (std::same_as<Op, RotateLeftOp>)
      return _mm512_rolv_epi32(value, counts);
    else return _mm512_rorv_epi32(value, counts);
  } else if constexpr (sizeof(Raw) == 64 && sizeof(T) == 8) {
    const auto counts = x86_bit_set1_lane<T, Raw>(static_cast<U>(rotate));
    if constexpr (std::same_as<Op, RotateLeftOp>)
      return _mm512_rolv_epi64(value, counts);
    else return _mm512_rorv_epi64(value, counts);
  } else
#endif
  if constexpr (std::same_as<Op, RotateLeftOp>) {
    return x86_bit_or_raw(
        x86_bit_shift_left_scalar_raw<T>(value, rotate),
        x86_bit_shift_right_logical_scalar_raw<T>(value, inverse));
  } else {
    return x86_bit_or_raw(
        x86_bit_shift_right_logical_scalar_raw<T>(value, rotate),
        x86_bit_shift_left_scalar_raw<T>(value, inverse));
  }
}

template <typename Op, typename T, typename Raw>
VECOPS_ALWAYS_INLINE Raw x86_bit_rotate_variable_raw(
    Raw value, Raw counts) {
  using U = std::make_unsigned_t<T>;
#if VEC_WIDTH >= 512 && defined(HAS_AVX512F)
  if constexpr (sizeof(Raw) == 64 && sizeof(T) == 4) {
    if constexpr (std::same_as<Op, RotateLeftOp>)
      return _mm512_rolv_epi32(value, counts);
    else return _mm512_rorv_epi32(value, counts);
  } else if constexpr (sizeof(Raw) == 64 && sizeof(T) == 8) {
    if constexpr (std::same_as<Op, RotateLeftOp>)
      return _mm512_rolv_epi64(value, counts);
    else return _mm512_rorv_epi64(value, counts);
  } else
#endif
  if constexpr (
      sizeof(T) >= 4
#if defined(HAS_AVX2)
      || sizeof(T) <= 2
#endif
  ) {
    constexpr U lane_mask = std::numeric_limits<U>::digits - 1;
    const auto mask = x86_bit_set1_lane<T, Raw>(lane_mask);
    const auto normalized = x86_bit_and_raw(counts, mask);
    const auto inverse = x86_bit_and_raw(
        x86_bit_sub_lanes<T>(x86_bit_zero_raw<Raw>(), normalized), mask);
    const auto left = x86_bit_shift_variable_raw<BitShiftLeftOp, U>(
        value, std::same_as<Op, RotateLeftOp> ? normalized : inverse);
    const auto right = x86_bit_shift_variable_raw<BitShiftRightOp, U>(
        value, std::same_as<Op, RotateLeftOp> ? inverse : normalized);
    return x86_bit_or_raw(left, right);
  } else {
    std::array<T, sizeof(Raw) / sizeof(T)> values{};
    std::array<T, sizeof(Raw) / sizeof(T)> lane_counts{};
    std::memcpy(values.data(), &value, sizeof(Raw));
    std::memcpy(lane_counts.data(), &counts, sizeof(Raw));
    constexpr U lane_mask = std::numeric_limits<U>::digits - 1;
    for (std::size_t lane = 0; lane < values.size(); ++lane) {
      const int count = static_cast<int>(
          scalar_bit_bits(lane_counts[lane]) & lane_mask);
      values[lane] = scalar_rotate<Op>(values[lane], count);
    }
    Raw result;
    std::memcpy(&result, values.data(), sizeof(Raw));
    return result;
  }
}

#if defined(CPU_CAPABILITY_AVX512) && defined(HAS_AVX512DQ)
inline constexpr bool x86_has_native_masked_shift_v = true;
#else
inline constexpr bool x86_has_native_masked_shift_v = false;
#endif

template <typename Op, typename T, typename Policy, typename Raw, typename MaskRaw>
  requires x86_has_native_masked_shift_v
VECOPS_ALWAYS_INLINE Raw x86_bit_masked_shift_raw(
    Raw value, Raw inactive, MaskRaw mask, int count);

#if defined(CPU_CAPABILITY_AVX512) && defined(HAS_AVX512DQ)
template <typename Op, typename T, typename Policy, typename Raw, typename MaskRaw>
  requires x86_has_native_masked_shift_v
VECOPS_ALWAYS_INLINE Raw x86_bit_masked_shift_raw(
    Raw value, Raw inactive, MaskRaw mask, int count) {
  static_assert(sizeof(T) == 4 || sizeof(T) == 8);
  const auto shift = _mm_cvtsi32_si128(count);
  if constexpr (std::same_as<Op, BitShiftLeftOp>) {
    if constexpr (sizeof(T) == 4) {
      if constexpr (sizeof(Raw) == 16)
        return std::same_as<Policy, ZeroBitLanes>
            ? _mm_maskz_sll_epi32(mask, value, shift)
            : _mm_mask_sll_epi32(inactive, mask, value, shift);
      else if constexpr (sizeof(Raw) == 32)
        return std::same_as<Policy, ZeroBitLanes>
            ? _mm256_maskz_sll_epi32(mask, value, shift)
            : _mm256_mask_sll_epi32(inactive, mask, value, shift);
      else
        return std::same_as<Policy, ZeroBitLanes>
            ? _mm512_maskz_sll_epi32(mask, value, shift)
            : _mm512_mask_sll_epi32(inactive, mask, value, shift);
    } else {
      if constexpr (sizeof(Raw) == 16)
        return std::same_as<Policy, ZeroBitLanes>
            ? _mm_maskz_sll_epi64(mask, value, shift)
            : _mm_mask_sll_epi64(inactive, mask, value, shift);
      else if constexpr (sizeof(Raw) == 32)
        return std::same_as<Policy, ZeroBitLanes>
            ? _mm256_maskz_sll_epi64(mask, value, shift)
            : _mm256_mask_sll_epi64(inactive, mask, value, shift);
      else
        return std::same_as<Policy, ZeroBitLanes>
            ? _mm512_maskz_sll_epi64(mask, value, shift)
            : _mm512_mask_sll_epi64(inactive, mask, value, shift);
    }
  } else if constexpr (std::is_signed_v<T>) {
    if constexpr (sizeof(T) == 4) {
      if constexpr (sizeof(Raw) == 16)
        return std::same_as<Policy, ZeroBitLanes>
            ? _mm_maskz_sra_epi32(mask, value, shift)
            : _mm_mask_sra_epi32(inactive, mask, value, shift);
      else if constexpr (sizeof(Raw) == 32)
        return std::same_as<Policy, ZeroBitLanes>
            ? _mm256_maskz_sra_epi32(mask, value, shift)
            : _mm256_mask_sra_epi32(inactive, mask, value, shift);
      else
        return std::same_as<Policy, ZeroBitLanes>
            ? _mm512_maskz_sra_epi32(mask, value, shift)
            : _mm512_mask_sra_epi32(inactive, mask, value, shift);
    } else {
      if constexpr (sizeof(Raw) == 16)
        return std::same_as<Policy, ZeroBitLanes>
            ? _mm_maskz_sra_epi64(mask, value, shift)
            : _mm_mask_sra_epi64(inactive, mask, value, shift);
      else if constexpr (sizeof(Raw) == 32)
        return std::same_as<Policy, ZeroBitLanes>
            ? _mm256_maskz_sra_epi64(mask, value, shift)
            : _mm256_mask_sra_epi64(inactive, mask, value, shift);
      else
        return std::same_as<Policy, ZeroBitLanes>
            ? _mm512_maskz_sra_epi64(mask, value, shift)
            : _mm512_mask_sra_epi64(inactive, mask, value, shift);
    }
  } else {
    if constexpr (sizeof(T) == 4) {
      if constexpr (sizeof(Raw) == 16)
        return std::same_as<Policy, ZeroBitLanes>
            ? _mm_maskz_srl_epi32(mask, value, shift)
            : _mm_mask_srl_epi32(inactive, mask, value, shift);
      else if constexpr (sizeof(Raw) == 32)
        return std::same_as<Policy, ZeroBitLanes>
            ? _mm256_maskz_srl_epi32(mask, value, shift)
            : _mm256_mask_srl_epi32(inactive, mask, value, shift);
      else
        return std::same_as<Policy, ZeroBitLanes>
            ? _mm512_maskz_srl_epi32(mask, value, shift)
            : _mm512_mask_srl_epi32(inactive, mask, value, shift);
    } else {
      if constexpr (sizeof(Raw) == 16)
        return std::same_as<Policy, ZeroBitLanes>
            ? _mm_maskz_srl_epi64(mask, value, shift)
            : _mm_mask_srl_epi64(inactive, mask, value, shift);
      else if constexpr (sizeof(Raw) == 32)
        return std::same_as<Policy, ZeroBitLanes>
            ? _mm256_maskz_srl_epi64(mask, value, shift)
            : _mm256_mask_srl_epi64(inactive, mask, value, shift);
      else
        return std::same_as<Policy, ZeroBitLanes>
            ? _mm512_maskz_srl_epi64(mask, value, shift)
            : _mm512_mask_srl_epi64(inactive, mask, value, shift);
    }
  }
}
#endif

#define VECOPS_VEC_X86_MASK_BINARY_32_64(Name)                           \
      if constexpr (sizeof(T) == 4) {                                    \
        if constexpr (sizeof(a.value) == 16)                             \
          return NativeWordVec<Tag>{_mm_mask_##Name##_epi32(             \
              a.value, mask.value, a.value, b.value)};                   \
        else if constexpr (sizeof(a.value) == 32)                        \
          return NativeWordVec<Tag>{_mm256_mask_##Name##_epi32(          \
              a.value, mask.value, a.value, b.value)};                   \
        else                                                              \
          return NativeWordVec<Tag>{_mm512_mask_##Name##_epi32(          \
              a.value, mask.value, a.value, b.value)};                   \
      } else if constexpr (sizeof(T) == 8) {                             \
        if constexpr (sizeof(a.value) == 16)                             \
          return NativeWordVec<Tag>{_mm_mask_##Name##_epi64(             \
              a.value, mask.value, a.value, b.value)};                   \
        else if constexpr (sizeof(a.value) == 32)                        \
          return NativeWordVec<Tag>{_mm256_mask_##Name##_epi64(          \
              a.value, mask.value, a.value, b.value)};                   \
        else                                                              \
          return NativeWordVec<Tag>{_mm512_mask_##Name##_epi64(          \
              a.value, mask.value, a.value, b.value)};                   \
      }

#define VECOPS_VEC_X86_MASKZ_BINARY_32_64(Name)                          \
      if constexpr (sizeof(T) == 4) {                                    \
        if constexpr (sizeof(a.value) == 16)                             \
          return NativeWordVec<Tag>{_mm_maskz_##Name##_epi32(            \
              mask.value, a.value, b.value)};                            \
        else if constexpr (sizeof(a.value) == 32)                        \
          return NativeWordVec<Tag>{_mm256_maskz_##Name##_epi32(         \
              mask.value, a.value, b.value)};                            \
        else return NativeWordVec<Tag>{_mm512_maskz_##Name##_epi32(      \
              mask.value, a.value, b.value)};                            \
      } else if constexpr (sizeof(T) == 8) {                             \
        if constexpr (sizeof(a.value) == 16)                             \
          return NativeWordVec<Tag>{_mm_maskz_##Name##_epi64(            \
              mask.value, a.value, b.value)};                            \
        else if constexpr (sizeof(a.value) == 32)                        \
          return NativeWordVec<Tag>{_mm256_maskz_##Name##_epi64(         \
              mask.value, a.value, b.value)};                            \
        else return NativeWordVec<Tag>{_mm512_maskz_##Name##_epi64(      \
              mask.value, a.value, b.value)};                            \
      }

#define VECOPS_VEC_DEFINE_X86_BIT_BINARY(OpType, RawHelper, MaskName)    \
  template <>                                                            \
  struct NativeWordImpl<X86Backend, OpType> {                            \
    template <nint_t Index, IntegerTag Tag>                              \
    static VECOPS_ALWAYS_INLINE NativeWordVec<Tag> call(                 \
        OpType, Tag, NativeWordVec<Tag> a, NativeWordVec<Tag> b) {       \
      using Traits = RepresentationTraits<X86Backend, Tag>;             \
      static_assert(Index >= 0 && Index < Traits::word_count);           \
      return NativeWordVec<Tag>{RawHelper(a.value, b.value)};            \
    }                                                                    \
    template <nint_t Index, IntegerTag Tag, typename Policy>             \
    static VECOPS_ALWAYS_INLINE NativeWordVec<Tag> call(                 \
        OpType op, Tag tag, NativeWordVec<Tag> a, NativeWordVec<Tag> b, \
        NativeWordMask<Tag> mask, NativeWordVec<Tag> inactive,          \
        Policy) {                                                        \
      using T = ElementOf<Tag>;                                          \
      static_assert(Index >= 0 && Index < num_words(tag));               \
      if constexpr (std::same_as<Policy, PreserveBitLanes>) {            \
        VECOPS_VEC_X86_MASK_BINARY_32_64(MaskName);                     \
      } else if constexpr (std::same_as<Policy, ZeroBitLanes>) {         \
        VECOPS_VEC_X86_MASKZ_BINARY_32_64(MaskName);                    \
      }                                                                  \
      const auto computed = call<Index>(op, tag, a, b);                 \
      return NativeWordImpl<X86Backend, BlendOp>::template call<Index>( \
          BlendOp{}, tag, inactive, mask, computed);                     \
    }                                                                    \
  }

#if defined(CPU_CAPABILITY_AVX512) && defined(HAS_AVX512DQ)
VECOPS_VEC_DEFINE_X86_BIT_BINARY(BitAndOp, x86_bit_and_raw, and);
VECOPS_VEC_DEFINE_X86_BIT_BINARY(BitOrOp, x86_bit_or_raw, or);
VECOPS_VEC_DEFINE_X86_BIT_BINARY(BitXorOp, x86_bit_xor_raw, xor);
VECOPS_VEC_DEFINE_X86_BIT_BINARY(BitAndNotOp, x86_bit_andnot_raw, andnot);
#else
#undef VECOPS_VEC_X86_MASK_BINARY_32_64
#undef VECOPS_VEC_X86_MASKZ_BINARY_32_64
#define VECOPS_VEC_X86_MASK_BINARY_32_64(Name)
#define VECOPS_VEC_X86_MASKZ_BINARY_32_64(Name)
VECOPS_VEC_DEFINE_X86_BIT_BINARY(BitAndOp, x86_bit_and_raw, and);
VECOPS_VEC_DEFINE_X86_BIT_BINARY(BitOrOp, x86_bit_or_raw, or);
VECOPS_VEC_DEFINE_X86_BIT_BINARY(BitXorOp, x86_bit_xor_raw, xor);
VECOPS_VEC_DEFINE_X86_BIT_BINARY(BitAndNotOp, x86_bit_andnot_raw, andnot);
#endif

#undef VECOPS_VEC_DEFINE_X86_BIT_BINARY
#undef VECOPS_VEC_X86_MASK_BINARY_32_64
#undef VECOPS_VEC_X86_MASKZ_BINARY_32_64

template <>
struct NativeWordImpl<X86Backend, BitNotOp> {
  template <nint_t Index, IntegerTag Tag>
  static VECOPS_ALWAYS_INLINE NativeWordVec<Tag> call(
      BitNotOp, Tag, NativeWordVec<Tag> value) {
    using Traits = RepresentationTraits<X86Backend, Tag>;
    static_assert(Index >= 0 && Index < Traits::word_count);
    return NativeWordVec<Tag>{
        x86_bit_xor_raw(value.value, x86_bit_ones_raw<decltype(value.value)>())};
  }

  template <nint_t Index, IntegerTag Tag, typename Policy>
  static VECOPS_ALWAYS_INLINE NativeWordVec<Tag> call(
      BitNotOp op, Tag tag, NativeWordVec<Tag> value,
      NativeWordMask<Tag> mask, NativeWordVec<Tag> inactive, Policy) {
#if defined(CPU_CAPABILITY_AVX512) && defined(HAS_AVX512DQ)
    using T = ElementOf<Tag>;
    if constexpr (std::same_as<Policy, ZeroBitLanes> && sizeof(T) == 4) {
      const auto ones = x86_bit_ones_raw<decltype(value.value)>();
      if constexpr (sizeof(value.value) == 16)
        return NativeWordVec<Tag>{
            _mm_maskz_xor_epi32(mask.value, value.value, ones)};
      else if constexpr (sizeof(value.value) == 32)
        return NativeWordVec<Tag>{
            _mm256_maskz_xor_epi32(mask.value, value.value, ones)};
      else return NativeWordVec<Tag>{
          _mm512_maskz_xor_epi32(mask.value, value.value, ones)};
    } else if constexpr (
        std::same_as<Policy, ZeroBitLanes> && sizeof(T) == 8) {
      const auto ones = x86_bit_ones_raw<decltype(value.value)>();
      if constexpr (sizeof(value.value) == 16)
        return NativeWordVec<Tag>{
            _mm_maskz_xor_epi64(mask.value, value.value, ones)};
      else if constexpr (sizeof(value.value) == 32)
        return NativeWordVec<Tag>{
            _mm256_maskz_xor_epi64(mask.value, value.value, ones)};
      else return NativeWordVec<Tag>{
          _mm512_maskz_xor_epi64(mask.value, value.value, ones)};
    }
    if constexpr (sizeof(T) == 4) {
      if constexpr (sizeof(value.value) == 16)
        return NativeWordVec<Tag>{_mm_mask_ternarylogic_epi32(
            inactive.value, mask.value, value.value, value.value,
            ~_MM_TERNLOG_B)};
      else if constexpr (sizeof(value.value) == 32)
        return NativeWordVec<Tag>{_mm256_mask_ternarylogic_epi32(
            inactive.value, mask.value, value.value, value.value,
            ~_MM_TERNLOG_B)};
      else
        return NativeWordVec<Tag>{_mm512_mask_ternarylogic_epi32(
            inactive.value, mask.value, value.value, value.value,
            ~_MM_TERNLOG_B)};
    } else if constexpr (sizeof(T) == 8) {
      if constexpr (sizeof(value.value) == 16)
        return NativeWordVec<Tag>{_mm_mask_ternarylogic_epi64(
            inactive.value, mask.value, value.value, value.value,
            ~_MM_TERNLOG_B)};
      else if constexpr (sizeof(value.value) == 32)
        return NativeWordVec<Tag>{_mm256_mask_ternarylogic_epi64(
            inactive.value, mask.value, value.value, value.value,
            ~_MM_TERNLOG_B)};
      else
        return NativeWordVec<Tag>{_mm512_mask_ternarylogic_epi64(
            inactive.value, mask.value, value.value, value.value,
            ~_MM_TERNLOG_B)};
    }
#endif
    const auto computed = call<Index>(op, tag, value);
    return NativeWordImpl<X86Backend, BlendOp>::template call<Index>(
        BlendOp{}, tag, inactive, mask, computed);
  }
};

#define VECOPS_VEC_DEFINE_X86_BIT_COUNT(OpType)                         \
  template <>                                                           \
  struct NativeWordImpl<X86Backend, OpType> {                           \
    template <nint_t Index, IntegerTag Tag>                             \
    static VECOPS_ALWAYS_INLINE NativeWordVec<Tag> call(                \
        OpType, Tag, NativeWordVec<Tag> value) {                        \
      using Traits = RepresentationTraits<X86Backend, Tag>;            \
      using T = ElementOf<Tag>;                                         \
      static_assert(Index >= 0 && Index < Traits::word_count);          \
      return NativeWordVec<Tag>{                                        \
          x86_bit_count_raw<OpType, T>(value.value)};                   \
    }                                                                   \
    template <nint_t Index, IntegerTag Tag, typename Policy>            \
    static VECOPS_ALWAYS_INLINE NativeWordVec<Tag> call(                \
        OpType op, Tag tag, NativeWordVec<Tag> value,                   \
        NativeWordMask<Tag> mask, NativeWordVec<Tag> inactive, Policy) {\
      const auto computed = call<Index>(op, tag, value);                \
      return NativeWordImpl<X86Backend, BlendOp>::template call<Index>(\
          BlendOp{}, tag, inactive, mask, computed);                    \
    }                                                                   \
  }

VECOPS_VEC_DEFINE_X86_BIT_COUNT(PopCountOp);
VECOPS_VEC_DEFINE_X86_BIT_COUNT(CountLeadingZeroOp);
VECOPS_VEC_DEFINE_X86_BIT_COUNT(CountLeadingOneOp);
VECOPS_VEC_DEFINE_X86_BIT_COUNT(CountTrailingZeroOp);
VECOPS_VEC_DEFINE_X86_BIT_COUNT(CountTrailingOneOp);

#undef VECOPS_VEC_DEFINE_X86_BIT_COUNT

#define VECOPS_VEC_DEFINE_X86_SHIFT(                                    \
    OpType, ImmediateExpression, ScalarExpression)                      \
  template <>                                                            \
  struct NativeWordImpl<X86Backend, OpType> {                            \
    template <nint_t Index, IntegerTag Tag, nint_t Count>                \
    static VECOPS_ALWAYS_INLINE NativeWordVec<Tag> call(                 \
        OpType, Tag, NativeWordVec<Tag> value, meta::Const<Count>) {     \
      using Traits = RepresentationTraits<X86Backend, Tag>;             \
      using T = ElementOf<Tag>;                                          \
      static_assert(Count >= 0);                                         \
      static_assert(Index >= 0 && Index < Traits::word_count);           \
      return NativeWordVec<Tag>{ImmediateExpression};                    \
    }                                                                    \
    template <nint_t Index, IntegerTag Tag>                              \
    static VECOPS_ALWAYS_INLINE NativeWordVec<Tag> call(                 \
        OpType, Tag, NativeWordVec<Tag> value, int count) {              \
      using Traits = RepresentationTraits<X86Backend, Tag>;             \
      using T = ElementOf<Tag>;                                          \
      static_assert(Index >= 0 && Index < Traits::word_count);           \
      return NativeWordVec<Tag>{ScalarExpression};                       \
    }                                                                    \
    template <nint_t Index, IntegerTag Tag>                              \
    static VECOPS_ALWAYS_INLINE NativeWordVec<Tag> call(                 \
        OpType, Tag, NativeWordVec<Tag> value,                           \
        NativeWordVec<Tag> counts) {                                    \
      using T = ElementOf<Tag>;                                          \
      static_assert(Index >= 0 && Index < num_words(Tag{}));             \
      return NativeWordVec<Tag>{x86_bit_shift_variable_raw<OpType, T>(  \
          value.value, counts.value)};                                  \
    }                                                                    \
    template <nint_t Index, IntegerTag Tag, nint_t Count, typename Policy>\
    static VECOPS_ALWAYS_INLINE NativeWordVec<Tag> call(                 \
        OpType op, Tag tag, NativeWordVec<Tag> value,                   \
        meta::Const<Count> count, NativeWordMask<Tag> mask,             \
        NativeWordVec<Tag> inactive, Policy) {                           \
      const auto computed = call<Index>(op, tag, value, count);         \
      return NativeWordImpl<X86Backend, BlendOp>::template call<Index>( \
          BlendOp{}, tag, inactive, mask, computed);                     \
    }                                                                    \
    template <nint_t Index, IntegerTag Tag, typename Policy>             \
    static VECOPS_ALWAYS_INLINE NativeWordVec<Tag> call(                 \
        OpType op, Tag tag, NativeWordVec<Tag> value, int count,        \
        NativeWordMask<Tag> mask, NativeWordVec<Tag> inactive, Policy) {\
      using T = ElementOf<Tag>;                                          \
      static_assert(Index >= 0 && Index < num_words(tag));               \
      if constexpr (                                                     \
          sizeof(T) >= 4                                                 \
          && requires { x86_bit_masked_shift_raw<OpType, T, Policy>(     \
              value.value, inactive.value, mask.value, count); }) {      \
        return NativeWordVec<Tag>{x86_bit_masked_shift_raw<             \
            OpType, T, Policy>(                                         \
            value.value, inactive.value, mask.value, count)};            \
      }                                                                  \
      const auto computed = call<Index>(op, tag, value, count);         \
      return NativeWordImpl<X86Backend, BlendOp>::template call<Index>( \
          BlendOp{}, tag, inactive, mask, computed);                     \
    }                                                                    \
    template <nint_t Index, IntegerTag Tag, typename Policy>             \
    static VECOPS_ALWAYS_INLINE NativeWordVec<Tag> call(                 \
        OpType op, Tag tag, NativeWordVec<Tag> value,                   \
        NativeWordVec<Tag> counts, NativeWordMask<Tag> mask,            \
        NativeWordVec<Tag> inactive, Policy) {                           \
      const auto computed = call<Index>(op, tag, value, counts);        \
      return NativeWordImpl<X86Backend, BlendOp>::template call<Index>( \
          BlendOp{}, tag, inactive, mask, computed);                     \
    }                                                                    \
  }

VECOPS_VEC_DEFINE_X86_SHIFT(
    BitShiftLeftOp,
    (x86_bit_shift_left_immediate_raw<Count, T>(value.value)),
    x86_bit_shift_left_scalar_raw<T>(value.value, count));
VECOPS_VEC_DEFINE_X86_SHIFT(
    BitShiftRightOp,
    (std::is_signed_v<T>
         ? x86_bit_shift_right_signed_immediate_raw<Count, T>(value.value)
         : x86_bit_shift_right_logical_immediate_raw<Count, T>(value.value)),
    (std::is_signed_v<T>
         ? x86_bit_shift_right_signed_scalar_raw<T>(value.value, count)
         : x86_bit_shift_right_logical_scalar_raw<T>(value.value, count)));

#undef VECOPS_VEC_DEFINE_X86_SHIFT

#define VECOPS_VEC_DEFINE_X86_ROTATE(OpType)                            \
  template <>                                                           \
  struct NativeWordImpl<X86Backend, OpType> {                           \
    template <nint_t Index, IntegerTag Tag, nint_t Count>               \
    static VECOPS_ALWAYS_INLINE NativeWordVec<Tag> call(                \
        OpType, Tag, NativeWordVec<Tag> value, meta::Const<Count>) {    \
      using Traits = RepresentationTraits<X86Backend, Tag>;            \
      using T = ElementOf<Tag>;                                         \
      static_assert(Index >= 0 && Index < Traits::word_count);          \
      return NativeWordVec<Tag>{                                        \
          x86_bit_rotate_immediate_raw<OpType, Count, T>(value.value)}; \
    }                                                                   \
    template <nint_t Index, IntegerTag Tag>                             \
    static VECOPS_ALWAYS_INLINE NativeWordVec<Tag> call(                \
        OpType, Tag, NativeWordVec<Tag> value, int count) {             \
      using Traits = RepresentationTraits<X86Backend, Tag>;            \
      using T = ElementOf<Tag>;                                         \
      static_assert(Index >= 0 && Index < Traits::word_count);          \
      return NativeWordVec<Tag>{                                        \
          x86_bit_rotate_scalar_raw<OpType, T>(value.value, count)};    \
    }                                                                   \
    template <nint_t Index, IntegerTag Tag>                             \
    static VECOPS_ALWAYS_INLINE NativeWordVec<Tag> call(                \
        OpType, Tag, NativeWordVec<Tag> value,                          \
        NativeWordVec<Tag> counts) {                                   \
      using T = ElementOf<Tag>;                                         \
      static_assert(Index >= 0 && Index < num_words(Tag{}));            \
      return NativeWordVec<Tag>{x86_bit_rotate_variable_raw<OpType, T>(\
          value.value, counts.value)};                                  \
    }                                                                   \
    template <nint_t Index, IntegerTag Tag, nint_t Count, typename Policy>\
    static VECOPS_ALWAYS_INLINE NativeWordVec<Tag> call(                \
        OpType op, Tag tag, NativeWordVec<Tag> value,                   \
        meta::Const<Count> count, NativeWordMask<Tag> mask,             \
        NativeWordVec<Tag> inactive, Policy) {                          \
      const auto computed = call<Index>(op, tag, value, count);        \
      return NativeWordImpl<X86Backend, BlendOp>::template call<Index>(\
          BlendOp{}, tag, inactive, mask, computed);                    \
    }                                                                   \
    template <nint_t Index, IntegerTag Tag, typename Policy>            \
    static VECOPS_ALWAYS_INLINE NativeWordVec<Tag> call(                \
        OpType op, Tag tag, NativeWordVec<Tag> value, int count,        \
        NativeWordMask<Tag> mask, NativeWordVec<Tag> inactive, Policy) {\
      const auto computed = call<Index>(op, tag, value, count);        \
      return NativeWordImpl<X86Backend, BlendOp>::template call<Index>(\
          BlendOp{}, tag, inactive, mask, computed);                    \
    }                                                                   \
    template <nint_t Index, IntegerTag Tag, typename Policy>            \
    static VECOPS_ALWAYS_INLINE NativeWordVec<Tag> call(                \
        OpType op, Tag tag, NativeWordVec<Tag> value,                   \
        NativeWordVec<Tag> counts, NativeWordMask<Tag> mask,            \
        NativeWordVec<Tag> inactive, Policy) {                          \
      const auto computed = call<Index>(op, tag, value, counts);       \
      return NativeWordImpl<X86Backend, BlendOp>::template call<Index>(\
          BlendOp{}, tag, inactive, mask, computed);                    \
    }                                                                   \
  }

VECOPS_VEC_DEFINE_X86_ROTATE(RotateLeftOp);
VECOPS_VEC_DEFINE_X86_ROTATE(RotateRightOp);

#undef VECOPS_VEC_DEFINE_X86_ROTATE

} // namespace vecops::vec::details

#endif // VECOPS_VEC_DETAILS_X86_BIT_H
