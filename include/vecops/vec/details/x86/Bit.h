#ifndef VECOPS_VEC_DETAILS_X86_BIT_H
#define VECOPS_VEC_DETAILS_X86_BIT_H

/**
 * @file Bit.h
 * @brief x86 backend implementations for bitwise operations.
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

template <typename Raw>
VECOPS_ALWAYS_INLINE Raw x86_bit_shift_left_raw(Raw value, int count,
                                                 int width) {
  const auto shift = _mm_cvtsi32_si128(count);
  if (width == 8) {
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
  }
  if (width == 16) {
    if constexpr (sizeof(Raw) == 16) return _mm_sll_epi16(value, shift);
#if VEC_WIDTH >= 256
    else if constexpr (sizeof(Raw) == 32) return _mm256_sll_epi16(value, shift);
#endif
#if VEC_WIDTH >= 512
    else return _mm512_sll_epi16(value, shift);
#endif
  }
  if (width == 32) {
    if constexpr (sizeof(Raw) == 16) return _mm_sll_epi32(value, shift);
#if VEC_WIDTH >= 256
    else if constexpr (sizeof(Raw) == 32) return _mm256_sll_epi32(value, shift);
#endif
#if VEC_WIDTH >= 512
    else return _mm512_sll_epi32(value, shift);
#endif
  }
  if constexpr (sizeof(Raw) == 16) return _mm_sll_epi64(value, shift);
#if VEC_WIDTH >= 256
  else if constexpr (sizeof(Raw) == 32) return _mm256_sll_epi64(value, shift);
#endif
#if VEC_WIDTH >= 512
  else return _mm512_sll_epi64(value, shift);
#endif
}

template <typename Raw>
VECOPS_ALWAYS_INLINE Raw x86_bit_shift_right_logical_raw(
    Raw value, int count, int width) {
  const auto shift = _mm_cvtsi32_si128(count);
  if (width <= 8) {
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
  }
  if (width == 16) {
    if constexpr (sizeof(Raw) == 16) return _mm_srl_epi16(value, shift);
#if VEC_WIDTH >= 256
    else if constexpr (sizeof(Raw) == 32) return _mm256_srl_epi16(value, shift);
#endif
#if VEC_WIDTH >= 512
    else return _mm512_srl_epi16(value, shift);
#endif
  }
  if (width == 32) {
    if constexpr (sizeof(Raw) == 16) return _mm_srl_epi32(value, shift);
#if VEC_WIDTH >= 256
    else if constexpr (sizeof(Raw) == 32) return _mm256_srl_epi32(value, shift);
#endif
#if VEC_WIDTH >= 512
    else return _mm512_srl_epi32(value, shift);
#endif
  }
  if constexpr (sizeof(Raw) == 16) return _mm_srl_epi64(value, shift);
#if VEC_WIDTH >= 256
  else if constexpr (sizeof(Raw) == 32) return _mm256_srl_epi64(value, shift);
#endif
#if VEC_WIDTH >= 512
  else return _mm512_srl_epi64(value, shift);
#endif
}

template <typename Raw>
VECOPS_ALWAYS_INLINE Raw x86_bit_negative_mask(Raw value, int width) {
  const auto zero = x86_bit_zero_raw<Raw>();
  if (width == 8) {
    if constexpr (sizeof(Raw) == 16) return _mm_cmpgt_epi8(zero, value);
#if VEC_WIDTH >= 256
    else if constexpr (sizeof(Raw) == 32) return _mm256_cmpgt_epi8(zero, value);
#endif
#if VEC_WIDTH >= 512
    else return _mm512_movm_epi8(_mm512_cmpgt_epi8_mask(zero, value));
#endif
  }
  if (width == 16) {
    if constexpr (sizeof(Raw) == 16) return _mm_cmpgt_epi16(zero, value);
#if VEC_WIDTH >= 256
    else if constexpr (sizeof(Raw) == 32) return _mm256_cmpgt_epi16(zero, value);
#endif
#if VEC_WIDTH >= 512
    else return _mm512_movm_epi16(_mm512_cmpgt_epi16_mask(zero, value));
#endif
  }
  const auto sign32 = [&] {
    if constexpr (sizeof(Raw) == 16) return _mm_srai_epi32(value, 31);
#if VEC_WIDTH >= 256
    else if constexpr (sizeof(Raw) == 32) return _mm256_srai_epi32(value, 31);
#endif
#if VEC_WIDTH >= 512
    else return _mm512_srai_epi32(value, 31);
#endif
  }();
  if (width == 32) return sign32;
  if constexpr (sizeof(Raw) == 16)
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

template <typename Raw>
VECOPS_ALWAYS_INLINE Raw x86_bit_shift_right_signed_raw(
    Raw value, int count, int width) {
  const auto sign = x86_bit_negative_mask(value, width);
  if (count >= width) return sign;
  if (count == 0) return value;
  const auto logical = x86_bit_shift_right_logical_raw(value, count, width);
  if (width == 8) {
    const auto low_mask = x86_bit_set1_i8<Raw>((0xffu >> count) & 0xffu);
    return x86_bit_or_raw(logical, x86_bit_andnot_raw(low_mask, sign));
  }
  const auto fill = x86_bit_shift_left_raw(sign, width - count, width);
  return x86_bit_or_raw(logical, fill);
}

template <typename Op, typename T, typename Raw>
VECOPS_ALWAYS_INLINE Raw x86_bit_shift_variable_raw(
    Raw value, Raw counts) {
#if defined(HAS_AVX2)
  if constexpr (sizeof(T) == 4) {
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
    } else if constexpr (sizeof(Raw) == 32) {
      if constexpr (std::same_as<Op, BitShiftLeftOp>)
        return _mm256_sllv_epi64(value, counts);
      else if constexpr (std::is_unsigned_v<T>)
        return _mm256_srlv_epi64(value, counts);
    }
  }
#endif
#if defined(CPU_CAPABILITY_AVX512)
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
  for (std::size_t lane = 0; lane < values.size(); ++lane) {
    const int count = static_cast<int>(lane_counts[lane]);
    if (count < 0) continue;
    if constexpr (std::same_as<Op, BitShiftLeftOp>)
      values[lane] = scalar_shift_left(values[lane], count);
    else
      values[lane] = scalar_shift_right(values[lane], count);
  }
  Raw result;
  std::memcpy(&result, values.data(), sizeof(Raw));
  return result;
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

#define VECOPS_VEC_DEFINE_X86_SHIFT(OpType, Expression)                 \
  template <>                                                            \
  struct NativeWordImpl<X86Backend, OpType> {                            \
    template <nint_t Index, IntegerTag Tag>                              \
    static VECOPS_ALWAYS_INLINE NativeWordVec<Tag> call(                 \
        OpType, Tag, NativeWordVec<Tag> value, int count) {              \
      using Traits = RepresentationTraits<X86Backend, Tag>;             \
      using T = ElementOf<Tag>;                                          \
      constexpr int width = std::numeric_limits<                        \
          std::make_unsigned_t<T>>::digits;                              \
      static_assert(Index >= 0 && Index < Traits::word_count);           \
      if constexpr (std::same_as<OpType, BitShiftLeftOp>) {              \
        if (count >= width)                                              \
          return NativeWordVec<Tag>{                                    \
              x86_bit_zero_raw<decltype(value.value)>()};                \
      }                                                                  \
      return NativeWordVec<Tag>{Expression};                             \
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
    BitShiftLeftOp, x86_bit_shift_left_raw(value.value, count, width));
VECOPS_VEC_DEFINE_X86_SHIFT(
    BitShiftRightOp,
    (std::is_signed_v<T>
         ? x86_bit_shift_right_signed_raw(value.value, count, width)
         : (count >= width
                ? x86_bit_zero_raw<decltype(value.value)>()
                : x86_bit_shift_right_logical_raw(value.value, count, width))));

#undef VECOPS_VEC_DEFINE_X86_SHIFT

} // namespace vecops::vec::details

#endif // VECOPS_VEC_DETAILS_X86_BIT_H
