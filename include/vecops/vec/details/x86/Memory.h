#ifndef VECOPS_VEC_DETAILS_X86_MEMORY_H
#define VECOPS_VEC_DETAILS_X86_MEMORY_H

/**
 * @file Memory.h
 * @brief x86 backend implementations for load and store operations.
 */

#include <array>
#include <cstring>
#include <type_traits>

#include "vecops/vec/details/x86/Basic.h"
#include "vecops/vec/details/Options.h"

namespace vecops::vec::details {


/* **************************************************************************** */
//                    Load and store word implementations                     //
/* **************************************************************************** */

template <typename Raw, typename IntegerRaw>
VECOPS_ALWAYS_INLINE Raw x86_memory_from_integer(IntegerRaw value) {
  static_assert(sizeof(Raw) == sizeof(IntegerRaw));
  Raw result;
  std::memcpy(&result, &value, sizeof(result));
  return result;
}

template <typename IntegerRaw, typename Raw>
VECOPS_ALWAYS_INLINE IntegerRaw x86_memory_to_integer(Raw value) {
  static_assert(sizeof(Raw) == sizeof(IntegerRaw));
  IntegerRaw result;
  std::memcpy(&result, &value, sizeof(result));
  return result;
}

template <typename Raw, typename T, typename Alignment, typename Temporality>
VECOPS_ALWAYS_INLINE Raw x86_load_memory_word(
    const T* pointer, Alignment, Temporality) {
  constexpr bool aligned = std::same_as<Alignment, mem::Aligned>;
  constexpr bool non_temporal =
      std::same_as<Temporality, mem::NonTemporal>;
  if constexpr (sizeof(Raw) == 16) {
    __m128i bits;
    if constexpr (aligned && non_temporal) {
      bits = _mm_stream_load_si128(
          const_cast<__m128i*>(reinterpret_cast<const __m128i*>(pointer)));
    } else if constexpr (aligned) {
      bits = _mm_load_si128(reinterpret_cast<const __m128i*>(pointer));
    } else {
      // x86 has no unaligned non-temporal vector load.
      bits = _mm_loadu_si128(reinterpret_cast<const __m128i*>(pointer));
    }
    return x86_memory_from_integer<Raw>(bits);
  }
#if VEC_WIDTH >= 256
  else if constexpr (sizeof(Raw) == 32) {
    __m256i bits;
#if defined(HAS_AVX2)
    if constexpr (aligned && non_temporal) {
      bits = _mm256_stream_load_si256(
          const_cast<__m256i*>(reinterpret_cast<const __m256i*>(pointer)));
    } else
#endif
    if constexpr (aligned) {
      bits = _mm256_load_si256(reinterpret_cast<const __m256i*>(pointer));
    } else {
      bits = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(pointer));
    }
    return x86_memory_from_integer<Raw>(bits);
  }
#endif
#if VEC_WIDTH >= 512
  else if constexpr (sizeof(Raw) == 64) {
    __m512i bits;
    if constexpr (aligned && non_temporal) {
      bits = _mm512_stream_load_si512(
          const_cast<void*>(reinterpret_cast<const void*>(pointer)));
    } else if constexpr (aligned) {
      bits = _mm512_load_si512(reinterpret_cast<const void*>(pointer));
    } else {
      bits = _mm512_loadu_si512(reinterpret_cast<const void*>(pointer));
    }
    return x86_memory_from_integer<Raw>(bits);
  }
#endif
  else {
    static_assert(dispatch_dependent_false<Raw>, "unsupported x86 load width");
  }
}

template <typename Raw, typename T, typename Alignment, typename Temporality>
VECOPS_ALWAYS_INLINE void x86_store_memory_word(
    T* pointer, Raw value, Alignment, Temporality) {
  constexpr bool aligned = std::same_as<Alignment, mem::Aligned>;
  constexpr bool non_temporal =
      std::same_as<Temporality, mem::NonTemporal>;
  if constexpr (sizeof(Raw) == 16) {
    const auto bits = x86_memory_to_integer<__m128i>(value);
    if constexpr (aligned && non_temporal)
      _mm_stream_si128(reinterpret_cast<__m128i*>(pointer), bits);
    else if constexpr (aligned)
      _mm_store_si128(reinterpret_cast<__m128i*>(pointer), bits);
    else
      _mm_storeu_si128(reinterpret_cast<__m128i*>(pointer), bits);
  }
#if VEC_WIDTH >= 256
  else if constexpr (sizeof(Raw) == 32) {
    const auto bits = x86_memory_to_integer<__m256i>(value);
    if constexpr (aligned && non_temporal)
      _mm256_stream_si256(reinterpret_cast<__m256i*>(pointer), bits);
    else if constexpr (aligned)
      _mm256_store_si256(reinterpret_cast<__m256i*>(pointer), bits);
    else
      _mm256_storeu_si256(reinterpret_cast<__m256i*>(pointer), bits);
  }
#endif
#if VEC_WIDTH >= 512
  else if constexpr (sizeof(Raw) == 64) {
    const auto bits = x86_memory_to_integer<__m512i>(value);
    if constexpr (aligned && non_temporal)
      _mm512_stream_si512(reinterpret_cast<__m512i*>(pointer), bits);
    else if constexpr (aligned)
      _mm512_store_si512(reinterpret_cast<void*>(pointer), bits);
    else
      _mm512_storeu_si512(reinterpret_cast<void*>(pointer), bits);
  }
#endif
  else {
    static_assert(dispatch_dependent_false<Raw>, "unsupported x86 store width");
  }
}

template <Element T, typename Raw, typename MaskRaw>
VECOPS_ALWAYS_INLINE Raw x86_masked_load_memory_word(
    const T* pointer, MaskRaw mask, Raw inactive) {
#if defined(CPU_CAPABILITY_AVX512)
#define VECOPS_VEC_X86_MASK_LOAD(Suffix)                              \
    do {                                                               \
      if constexpr (sizeof(Raw) == 16)                                \
        return _mm_mask_loadu_##Suffix(inactive, mask, pointer);       \
      else if constexpr (sizeof(Raw) == 32)                           \
        return _mm256_mask_loadu_##Suffix(inactive, mask, pointer);    \
      else                                                             \
        return _mm512_mask_loadu_##Suffix(inactive, mask, pointer);    \
    } while (false)
  if constexpr (sizeof(T) == 1) VECOPS_VEC_X86_MASK_LOAD(epi8);
  else if constexpr (sizeof(T) == 2) VECOPS_VEC_X86_MASK_LOAD(epi16);
  else if constexpr (std::same_as<T, float32_t>) VECOPS_VEC_X86_MASK_LOAD(ps);
  else if constexpr (sizeof(T) == 4) VECOPS_VEC_X86_MASK_LOAD(epi32);
  else if constexpr (std::same_as<T, float64_t>) VECOPS_VEC_X86_MASK_LOAD(pd);
  else if constexpr (sizeof(T) == 8) VECOPS_VEC_X86_MASK_LOAD(epi64);
  else static_assert(dispatch_dependent_false<T>, "unsupported x86 masked load type");
#undef VECOPS_VEC_X86_MASK_LOAD
#else
  if constexpr (sizeof(T) == 4 && sizeof(Raw) == 16) {
    if constexpr (std::same_as<T, float32_t>) {
      const auto loaded = _mm_maskload_ps(pointer, mask);
      return _mm_blendv_ps(
          inactive, loaded, _mm_castsi128_ps(mask));
    } else {
#if defined(HAS_AVX2)
      const auto loaded = _mm_maskload_epi32(
          reinterpret_cast<const int*>(pointer), mask);
      return _mm_blendv_epi8(inactive, loaded, mask);
#else
      const auto loaded = _mm_maskload_ps(
          reinterpret_cast<const float*>(pointer), mask);
      return _mm_castps_si128(_mm_blendv_ps(
          _mm_castsi128_ps(inactive), loaded, _mm_castsi128_ps(mask)));
#endif
    }
  } else if constexpr (sizeof(T) == 8 && sizeof(Raw) == 16) {
    if constexpr (std::same_as<T, float64_t>) {
      const auto loaded = _mm_maskload_pd(pointer, mask);
      return _mm_blendv_pd(
          inactive, loaded, _mm_castsi128_pd(mask));
    } else {
#if defined(HAS_AVX2)
      const auto loaded = _mm_maskload_epi64(
          reinterpret_cast<const long long*>(pointer), mask);
      return _mm_blendv_epi8(inactive, loaded, mask);
#else
      const auto loaded = _mm_maskload_pd(
          reinterpret_cast<const double*>(pointer), mask);
      return _mm_castpd_si128(_mm_blendv_pd(
          _mm_castsi128_pd(inactive), loaded, _mm_castsi128_pd(mask)));
#endif
    }
  }
#if VEC_WIDTH >= 256
  else if constexpr (sizeof(T) == 4 && sizeof(Raw) == 32) {
    if constexpr (std::same_as<T, float32_t>) {
      const auto loaded = _mm256_maskload_ps(pointer, mask);
      return _mm256_blendv_ps(
          inactive, loaded, _mm256_castsi256_ps(mask));
    } else {
#if defined(HAS_AVX2)
      const auto loaded = _mm256_maskload_epi32(
          reinterpret_cast<const int*>(pointer), mask);
      return _mm256_blendv_epi8(inactive, loaded, mask);
#else
      const auto loaded = _mm256_maskload_ps(
          reinterpret_cast<const float*>(pointer), mask);
      return _mm256_castps_si256(_mm256_blendv_ps(
          _mm256_castsi256_ps(inactive), loaded,
          _mm256_castsi256_ps(mask)));
#endif
    }
  } else if constexpr (sizeof(T) == 8 && sizeof(Raw) == 32) {
    if constexpr (std::same_as<T, float64_t>) {
      const auto loaded = _mm256_maskload_pd(pointer, mask);
      return _mm256_blendv_pd(
          inactive, loaded, _mm256_castsi256_pd(mask));
    } else {
#if defined(HAS_AVX2)
      const auto loaded = _mm256_maskload_epi64(
          reinterpret_cast<const long long*>(pointer), mask);
      return _mm256_blendv_epi8(inactive, loaded, mask);
#else
      const auto loaded = _mm256_maskload_pd(
          reinterpret_cast<const double*>(pointer), mask);
      return _mm256_castpd_si256(_mm256_blendv_pd(
          _mm256_castsi256_pd(inactive), loaded,
          _mm256_castsi256_pd(mask)));
#endif
    }
  }
#endif
  else {
  constexpr nint_t lanes = sizeof(Raw) / sizeof(T);
  alignas(64) std::array<T, static_cast<std::size_t>(lanes)> values;
  alignas(64) std::array<unsigned char, sizeof(MaskRaw)> mask_bytes;
  std::memcpy(values.data(), &inactive, sizeof(Raw));
  std::memcpy(mask_bytes.data(), &mask, sizeof(MaskRaw));
  for (nint_t lane = 0; lane < lanes; ++lane) {
    if (mask_bytes[static_cast<std::size_t>(lane * sizeof(T))] != 0)
      values[static_cast<std::size_t>(lane)] = pointer[lane];
  }
  Raw result;
  std::memcpy(&result, values.data(), sizeof(result));
  return result;
  }
#endif
}

template <Element T, typename Raw, typename MaskRaw>
VECOPS_ALWAYS_INLINE void x86_masked_store_memory_word(
    T* pointer, MaskRaw mask, Raw value) {
#if defined(CPU_CAPABILITY_AVX512)
#define VECOPS_VEC_X86_MASK_STORE(Suffix)                             \
    do {                                                               \
      if constexpr (sizeof(Raw) == 16)                                \
        _mm_mask_storeu_##Suffix(pointer, mask, value);                \
      else if constexpr (sizeof(Raw) == 32)                           \
        _mm256_mask_storeu_##Suffix(pointer, mask, value);             \
      else                                                             \
        _mm512_mask_storeu_##Suffix(pointer, mask, value);             \
      return;                                                          \
    } while (false)
  if constexpr (sizeof(T) == 1) VECOPS_VEC_X86_MASK_STORE(epi8);
  else if constexpr (sizeof(T) == 2) VECOPS_VEC_X86_MASK_STORE(epi16);
  else if constexpr (std::same_as<T, float32_t>) VECOPS_VEC_X86_MASK_STORE(ps);
  else if constexpr (sizeof(T) == 4) VECOPS_VEC_X86_MASK_STORE(epi32);
  else if constexpr (std::same_as<T, float64_t>) VECOPS_VEC_X86_MASK_STORE(pd);
  else if constexpr (sizeof(T) == 8) VECOPS_VEC_X86_MASK_STORE(epi64);
  else static_assert(dispatch_dependent_false<T>, "unsupported x86 masked store type");
#undef VECOPS_VEC_X86_MASK_STORE
#else
  if constexpr (sizeof(T) == 4 && sizeof(Raw) == 16) {
    if constexpr (std::same_as<T, float32_t>)
      _mm_maskstore_ps(pointer, mask, value);
    else {
#if defined(HAS_AVX2)
      _mm_maskstore_epi32(
          reinterpret_cast<int*>(pointer), mask, value);
#else
      _mm_maskstore_ps(
          reinterpret_cast<float*>(pointer), mask,
          _mm_castsi128_ps(value));
#endif
    }
    return;
  } else if constexpr (sizeof(T) == 8 && sizeof(Raw) == 16) {
    if constexpr (std::same_as<T, float64_t>)
      _mm_maskstore_pd(pointer, mask, value);
    else {
#if defined(HAS_AVX2)
      _mm_maskstore_epi64(
          reinterpret_cast<long long*>(pointer), mask, value);
#else
      _mm_maskstore_pd(
          reinterpret_cast<double*>(pointer), mask,
          _mm_castsi128_pd(value));
#endif
    }
    return;
  }
#if VEC_WIDTH >= 256
  else if constexpr (sizeof(T) == 4 && sizeof(Raw) == 32) {
    if constexpr (std::same_as<T, float32_t>)
      _mm256_maskstore_ps(pointer, mask, value);
    else {
#if defined(HAS_AVX2)
      _mm256_maskstore_epi32(
          reinterpret_cast<int*>(pointer), mask, value);
#else
      _mm256_maskstore_ps(
          reinterpret_cast<float*>(pointer), mask,
          _mm256_castsi256_ps(value));
#endif
    }
    return;
  } else if constexpr (sizeof(T) == 8 && sizeof(Raw) == 32) {
    if constexpr (std::same_as<T, float64_t>)
      _mm256_maskstore_pd(pointer, mask, value);
    else {
#if defined(HAS_AVX2)
      _mm256_maskstore_epi64(
          reinterpret_cast<long long*>(pointer), mask, value);
#else
      _mm256_maskstore_pd(
          reinterpret_cast<double*>(pointer), mask,
          _mm256_castsi256_pd(value));
#endif
    }
    return;
  }
#endif
  else {
  constexpr nint_t lanes = sizeof(Raw) / sizeof(T);
  alignas(64) std::array<T, static_cast<std::size_t>(lanes)> values;
  alignas(64) std::array<unsigned char, sizeof(MaskRaw)> mask_bytes;
  std::memcpy(values.data(), &value, sizeof(Raw));
  std::memcpy(mask_bytes.data(), &mask, sizeof(MaskRaw));
  for (nint_t lane = 0; lane < lanes; ++lane) {
    if (mask_bytes[static_cast<std::size_t>(lane * sizeof(T))] != 0)
      pointer[lane] = values[static_cast<std::size_t>(lane)];
  }
  }
#endif
}

#if defined(HAS_AVX2)
template <VectorTag Tag, typename Gathered>
  requires (sizeof(ElementOf<Tag>) < 4)
VECOPS_ALWAYS_INLINE Vec<Tag> x86_compact_indexed_subword(Gathered gathered) {
  const auto packed16 = [&] {
    if constexpr (sizeof(Gathered) == 16) {
      return _mm_packus_epi32(gathered, _mm_setzero_si128());
    } else if constexpr (sizeof(Gathered) == 32) {
      return _mm_packus_epi32(
          _mm256_castsi256_si128(gathered),
          _mm256_extracti128_si256(gathered, 1));
    } else {
      return _mm512_cvtepi32_epi16(gathered);
    }
  }();
  if constexpr (sizeof(ElementOf<Tag>) == 2) {
    return Vec<Tag>{packed16};
  } else if constexpr (sizeof(Gathered) <= 32) {
    return Vec<Tag>{_mm_packus_epi16(packed16, _mm_setzero_si128())};
  } else {
    return Vec<Tag>{_mm_packus_epi16(
        _mm256_castsi256_si128(packed16),
        _mm256_extracti128_si256(packed16, 1))};
  }
}

template <int Scale, VectorTag Tag, typename Raw>
VECOPS_ALWAYS_INLINE Raw x86_indexed_subword_offsets(
    Raw indices, const void* pointer, Raw& shifts) {
  const int remainder = static_cast<int>(
      reinterpret_cast<std::uintptr_t>(pointer) & std::uintptr_t{3});
  const auto fill = [=](int value) {
    if constexpr (sizeof(Raw) == 16) return _mm_set1_epi32(value);
    else if constexpr (sizeof(Raw) == 32) return _mm256_set1_epi32(value);
    else return _mm512_set1_epi32(value);
  };
  const auto add = [](Raw a, Raw b) {
    if constexpr (sizeof(Raw) == 16) return _mm_add_epi32(a, b);
    else if constexpr (sizeof(Raw) == 32) return _mm256_add_epi32(a, b);
    else return _mm512_add_epi32(a, b);
  };
  const auto bit_and = [](Raw a, Raw b) {
    if constexpr (sizeof(Raw) == 16) return _mm_and_si128(a, b);
    else if constexpr (sizeof(Raw) == 32) return _mm256_and_si256(a, b);
    else return _mm512_and_si512(a, b);
  };
  const auto shift_left = [](Raw value, int bits) {
    if constexpr (sizeof(Raw) == 16) return _mm_slli_epi32(value, bits);
    else if constexpr (sizeof(Raw) == 32) return _mm256_slli_epi32(value, bits);
    else return _mm512_slli_epi32(value, bits);
  };
  constexpr int shift = Scale == 1 ? 0 : Scale == 2 ? 1 : Scale == 4 ? 2 : 3;
  const auto byte_indices = add(
      shift == 0 ? indices : shift_left(indices, shift), fill(remainder));
  if constexpr (sizeof(ElementOf<Tag>) == 1) {
    shifts = shift_left(bit_and(byte_indices, fill(3)), 3);
    return bit_and(byte_indices, fill(~3));
  } else {
    const auto at_offset_three = bit_and(
        bit_and(byte_indices, shift_left(byte_indices, 1)), fill(2));
    const auto tail = [&] {
      if constexpr (sizeof(Raw) == 16)
        return _mm_xor_si128(at_offset_three, fill(3));
      else if constexpr (sizeof(Raw) == 32)
        return _mm256_xor_si256(at_offset_three, fill(3));
      else
        return _mm512_xor_si512(at_offset_three, fill(3));
    }();
    shifts = shift_left(bit_and(byte_indices, tail), 3);
    if constexpr (sizeof(Raw) == 16)
      return _mm_andnot_si128(tail, byte_indices);
    else if constexpr (sizeof(Raw) == 32)
      return _mm256_andnot_si256(tail, byte_indices);
    else
      return _mm512_andnot_si512(tail, byte_indices);
  }
}

template <int Scale, VectorTag Tag, typename IndexWord>
  requires (sizeof(ElementOf<Tag>) < 4)
VECOPS_ALWAYS_INLINE Vec<Tag> x86_load_indexed_subword_leaf(
    Tag tag, const ElementOf<Tag>* pointer, IndexWord indices,
    Mask<Tag> mask) {
  using IndexRaw = decltype(indices.value);
  IndexRaw shifts;
  const auto offsets =
      x86_indexed_subword_offsets<Scale, Tag>(
          indices.value, pointer, shifts);
  const auto* base = reinterpret_cast<const int*>(
      reinterpret_cast<std::uintptr_t>(pointer) & ~std::uintptr_t{3});
  const auto gathered = [&] {
#if defined(CPU_CAPABILITY_AVX512)
    if constexpr (sizeof(IndexRaw) == 16)
      return _mm_mmask_i32gather_epi32(
          _mm_setzero_si128(), mask.value, offsets, base, 1);
    else if constexpr (sizeof(IndexRaw) == 32)
      return _mm256_mmask_i32gather_epi32(
          _mm256_setzero_si256(), mask.value, offsets, base, 1);
    else
      return _mm512_mask_i32gather_epi32(
          _mm512_setzero_si512(), mask.value, offsets, base, 1);
#else
    const auto active = [&] {
      if constexpr (sizeof(ElementOf<Tag>) == 1) {
        if constexpr (sizeof(IndexRaw) == 16)
          return _mm_cvtepi8_epi32(mask.value);
        else
          return _mm256_cvtepi8_epi32(mask.value);
      } else {
        if constexpr (sizeof(IndexRaw) == 16)
          return _mm_cvtepi16_epi32(mask.value);
        else
          return _mm256_cvtepi16_epi32(mask.value);
      }
    }();
    if constexpr (sizeof(IndexRaw) == 16)
      return _mm_mask_i32gather_epi32(
          _mm_setzero_si128(), base, offsets, active, 1);
    else
      return _mm256_mask_i32gather_epi32(
          _mm256_setzero_si256(), base, offsets, active, 1);
#endif
  }();
  const auto shifted = [&] {
    if constexpr (sizeof(IndexRaw) == 16)
      return _mm_srlv_epi32(gathered, shifts);
    else if constexpr (sizeof(IndexRaw) == 32)
      return _mm256_srlv_epi32(gathered, shifts);
    else
      return _mm512_srlv_epi32(gathered, shifts);
  }();
  const auto selected = [&] {
    constexpr int selection = sizeof(ElementOf<Tag>) == 1 ? 0xff : 0xffff;
    if constexpr (sizeof(IndexRaw) == 16)
      return _mm_and_si128(shifted, _mm_set1_epi32(selection));
    else if constexpr (sizeof(IndexRaw) == 32)
      return _mm256_and_si256(shifted, _mm256_set1_epi32(selection));
    else
      return _mm512_and_si512(shifted, _mm512_set1_epi32(selection));
  }();
  (void)tag;
  return x86_compact_indexed_subword<Tag>(selected);
}

template <int Scale, VectorTag Tag, VectorTag IndexTag>
  requires (sizeof(ElementOf<Tag>) >= 4)
VECOPS_ALWAYS_INLINE Vec<Tag> x86_load_indexed_leaf(
    Tag, const ElementOf<Tag>* pointer, Vec<IndexTag> indices,
    Mask<Tag> mask, Vec<Tag> inactive) {
  using T = ElementOf<Tag>;
  using I = ElementOf<IndexTag>;
  using Raw = typename RepresentationTraits<X86Backend, Tag>::RawVec;
  using IndexRaw = typename RepresentationTraits<X86Backend, IndexTag>::RawVec;
  constexpr int scale = Scale == 0 ? sizeof(T) : Scale;
  if constexpr (sizeof(I) == 4 && sizeof(T) == 4) {
#if defined(CPU_CAPABILITY_AVX512)
    if constexpr (sizeof(Raw) == 16) {
      if constexpr (std::same_as<T, float32_t>)
        return Vec<Tag>{_mm_mmask_i32gather_ps(
            inactive.value, mask.value, indices.value, pointer, scale)};
      else
        return Vec<Tag>{_mm_mmask_i32gather_epi32(
            inactive.value, mask.value, indices.value, pointer, scale)};
    } else if constexpr (sizeof(Raw) == 32) {
      if constexpr (std::same_as<T, float32_t>)
        return Vec<Tag>{_mm256_mmask_i32gather_ps(
            inactive.value, mask.value, indices.value, pointer, scale)};
      else
        return Vec<Tag>{_mm256_mmask_i32gather_epi32(
            inactive.value, mask.value, indices.value, pointer, scale)};
    } else if constexpr (std::same_as<T, float32_t>)
      return Vec<Tag>{_mm512_mask_i32gather_ps(
          inactive.value, mask.value, indices.value, pointer, scale)};
    else
      return Vec<Tag>{_mm512_mask_i32gather_epi32(
          inactive.value, mask.value, indices.value, pointer, scale)};
#else
    if constexpr (sizeof(Raw) == 16) {
      if constexpr (std::same_as<T, float32_t>)
        return Vec<Tag>{_mm_mask_i32gather_ps(
            inactive.value, pointer, indices.value,
            _mm_castsi128_ps(mask.value), scale)};
      else
        return Vec<Tag>{_mm_mask_i32gather_epi32(
            inactive.value, reinterpret_cast<const int*>(pointer),
            indices.value, mask.value, scale)};
    } else if constexpr (std::same_as<T, float32_t>)
      return Vec<Tag>{_mm256_mask_i32gather_ps(
          inactive.value, pointer, indices.value,
          _mm256_castsi256_ps(mask.value), scale)};
    else
      return Vec<Tag>{_mm256_mask_i32gather_epi32(
          inactive.value, reinterpret_cast<const int*>(pointer),
          indices.value, mask.value, scale)};
#endif
  } else if constexpr (sizeof(I) == 4) {
#if defined(CPU_CAPABILITY_AVX512)
    if constexpr (sizeof(Raw) == 16) {
      if constexpr (std::same_as<T, float64_t>)
        return Vec<Tag>{_mm_mmask_i32gather_pd(
            inactive.value, mask.value, indices.value, pointer, scale)};
      else
        return Vec<Tag>{_mm_mmask_i32gather_epi64(
            inactive.value, mask.value, indices.value, pointer, scale)};
    } else if constexpr (sizeof(Raw) == 32) {
      if constexpr (std::same_as<T, float64_t>)
        return Vec<Tag>{_mm256_mmask_i32gather_pd(
            inactive.value, mask.value, indices.value, pointer, scale)};
      else
        return Vec<Tag>{_mm256_mmask_i32gather_epi64(
            inactive.value, mask.value, indices.value, pointer, scale)};
    } else if constexpr (std::same_as<T, float64_t>)
      return Vec<Tag>{_mm512_mask_i32gather_pd(
          inactive.value, mask.value, indices.value, pointer, scale)};
    else
      return Vec<Tag>{_mm512_mask_i32gather_epi64(
          inactive.value, mask.value, indices.value, pointer, scale)};
#else
    if constexpr (sizeof(Raw) == 16) {
      if constexpr (std::same_as<T, float64_t>)
        return Vec<Tag>{_mm_mask_i32gather_pd(
            inactive.value, pointer, indices.value,
            _mm_castsi128_pd(mask.value), scale)};
      else
        return Vec<Tag>{_mm_mask_i32gather_epi64(
            inactive.value, reinterpret_cast<const long long*>(pointer),
            indices.value, mask.value, scale)};
    } else if constexpr (std::same_as<T, float64_t>)
      return Vec<Tag>{_mm256_mask_i32gather_pd(
          inactive.value, pointer, indices.value,
          _mm256_castsi256_pd(mask.value), scale)};
    else
      return Vec<Tag>{_mm256_mask_i32gather_epi64(
          inactive.value, reinterpret_cast<const long long*>(pointer),
          indices.value, mask.value, scale)};
#endif
  } else if constexpr (sizeof(T) == 8) {
#if defined(CPU_CAPABILITY_AVX512)
    if constexpr (sizeof(Raw) == 16) {
      if constexpr (std::same_as<T, float64_t>)
        return Vec<Tag>{_mm_mmask_i64gather_pd(
            inactive.value, mask.value, indices.value, pointer, scale)};
      else
        return Vec<Tag>{_mm_mmask_i64gather_epi64(
            inactive.value, mask.value, indices.value, pointer, scale)};
    } else if constexpr (sizeof(Raw) == 32) {
      if constexpr (std::same_as<T, float64_t>)
        return Vec<Tag>{_mm256_mmask_i64gather_pd(
            inactive.value, mask.value, indices.value, pointer, scale)};
      else
        return Vec<Tag>{_mm256_mmask_i64gather_epi64(
            inactive.value, mask.value, indices.value, pointer, scale)};
    } else if constexpr (std::same_as<T, float64_t>)
      return Vec<Tag>{_mm512_mask_i64gather_pd(
          inactive.value, mask.value, indices.value, pointer, scale)};
    else
      return Vec<Tag>{_mm512_mask_i64gather_epi64(
          inactive.value, mask.value, indices.value, pointer, scale)};
#else
    if constexpr (sizeof(Raw) == 16) {
      if constexpr (std::same_as<T, float64_t>)
        return Vec<Tag>{_mm_mask_i64gather_pd(
            inactive.value, pointer, indices.value,
            _mm_castsi128_pd(mask.value), scale)};
      else
        return Vec<Tag>{_mm_mask_i64gather_epi64(
            inactive.value, reinterpret_cast<const long long*>(pointer),
            indices.value, mask.value, scale)};
    } else if constexpr (std::same_as<T, float64_t>)
      return Vec<Tag>{_mm256_mask_i64gather_pd(
          inactive.value, pointer, indices.value,
          _mm256_castsi256_pd(mask.value), scale)};
    else
      return Vec<Tag>{_mm256_mask_i64gather_epi64(
          inactive.value, reinterpret_cast<const long long*>(pointer),
          indices.value, mask.value, scale)};
#endif
  } else {
    static_assert(sizeof(T) == 4 && sizeof(I) == 8);
#if defined(CPU_CAPABILITY_AVX512)
    if constexpr (sizeof(IndexRaw) == 16) {
      if constexpr (std::same_as<T, float32_t>)
        return Vec<Tag>{_mm_mmask_i64gather_ps(
            inactive.value, mask.value, indices.value, pointer, scale)};
      else
        return Vec<Tag>{_mm_mmask_i64gather_epi32(
            inactive.value, mask.value, indices.value, pointer, scale)};
    } else if constexpr (sizeof(IndexRaw) == 32) {
      if constexpr (std::same_as<T, float32_t>)
        return Vec<Tag>{_mm256_mmask_i64gather_ps(
            inactive.value, mask.value, indices.value, pointer, scale)};
      else
        return Vec<Tag>{_mm256_mmask_i64gather_epi32(
            inactive.value, mask.value, indices.value, pointer, scale)};
    } else if constexpr (std::same_as<T, float32_t>)
      return Vec<Tag>{_mm512_mask_i64gather_ps(
          inactive.value, mask.value, indices.value, pointer, scale)};
    else
      return Vec<Tag>{_mm512_mask_i64gather_epi32(
          inactive.value, mask.value, indices.value, pointer, scale)};
#else
    if constexpr (sizeof(IndexRaw) == 16) {
      if constexpr (std::same_as<T, float32_t>)
        return Vec<Tag>{_mm_mask_i64gather_ps(
            inactive.value, pointer, indices.value,
            _mm_castsi128_ps(mask.value), scale)};
      else
        return Vec<Tag>{_mm_mask_i64gather_epi32(
            inactive.value, reinterpret_cast<const int*>(pointer),
            indices.value, mask.value, scale)};
    } else if constexpr (std::same_as<T, float32_t>)
      return Vec<Tag>{_mm256_mask_i64gather_ps(
          inactive.value, pointer, indices.value,
          _mm_castsi128_ps(mask.value), scale)};
    else
      return Vec<Tag>{_mm256_mask_i64gather_epi32(
          inactive.value, reinterpret_cast<const int*>(pointer),
          indices.value, mask.value, scale)};
#endif
  }
}

template <int Scale, VectorTag Tag, VectorTag IndexTag>
VECOPS_ALWAYS_INLINE Vec<Tag> x86_load_indexed_native(
    Tag tag, const ElementOf<Tag>* pointer, Vec<IndexTag> indices,
    Mask<Tag> mask, Vec<Tag> inactive) {
  using Traits = RepresentationTraits<X86Backend, Tag>;
  using IndexTraits = RepresentationTraits<X86Backend, IndexTag>;
  using WordTag = FixedTag<ElementOf<Tag>, Traits::word_lanes>;
  using IndexChunkTag = Rebind<ElementOf<IndexTag>, WordTag>;
  using IndexChunkTraits =
      RepresentationTraits<X86Backend, IndexChunkTag>;
  constexpr bool complete_output_words =
      Traits::logical_lanes == Traits::word_count * Traits::word_lanes;
  constexpr bool directly_partitioned_output =
      Traits::word_count > 1 && complete_output_words &&
      IndexTraits::word_count ==
          Traits::word_count * IndexChunkTraits::word_count;

  if constexpr (directly_partitioned_output) {
    return construct_words<X86Backend>(
        tag, [&]<nint_t OutputIndex>(Tag) VECOPS_INLINE_LAMBDA {
      const auto index_chunk = construct_words<X86Backend>(
          IndexChunkTag{},
          [&]<nint_t InputIndex>(IndexChunkTag) VECOPS_INLINE_LAMBDA {
            constexpr nint_t source_index =
                OutputIndex * IndexChunkTraits::word_count + InputIndex;
            return ::vecops::vec::get_word<source_index>(
                IndexTag{}, indices);
          });
      const auto loaded = x86_load_indexed_native<
          Scale, WordTag, IndexChunkTag>(
          WordTag{}, pointer, index_chunk,
          ::vecops::vec::get_word<OutputIndex>(tag, mask),
          ::vecops::vec::get_word<OutputIndex>(tag, inactive));
      return ::vecops::vec::get_word<0>(WordTag{}, loaded);
    });
  } else if constexpr (num_words(tag) > 1 || num_words(IndexTag{}) > 1) {
    const auto lower = x86_load_indexed_native<
        Scale, Half<Tag>, Half<IndexTag>>(
        Half<Tag>{}, pointer, execute(LowerOp{}, IndexTag{}, indices),
        execute(LowerOp{}, tag, mask), execute(LowerOp{}, tag, inactive));
    const auto upper = x86_load_indexed_native<
        Scale, Half<Tag>, Half<IndexTag>>(
        Half<Tag>{}, pointer, execute(UpperOp{}, IndexTag{}, indices),
        execute(UpperOp{}, tag, mask), execute(UpperOp{}, tag, inactive));
    return execute(ConcatOp{}, tag, lower, upper);
  } else if constexpr (sizeof(ElementOf<Tag>) < 4) {
    const auto loaded = x86_load_indexed_subword_leaf<Scale>(
        tag, pointer, indices, mask);
    return execute(BlendOp{}, tag, inactive, mask, loaded);
  } else {
    return x86_load_indexed_leaf<Scale, Tag, IndexTag>(
        tag, pointer, indices, mask, inactive);
  }
}

template <VectorTag Tag>
struct NativeImpl<X86Backend, LoadOp, Tag> {
  template <VectorValue Indices, int Scale, typename Temporality>
    // TODO(perf): x86 has no i64-offset subword gather. Those combinations
    // deliberately fall through to GenericImpl's lane-by-lane byte addressing;
    // a future path may range-check and narrow indices to i32 first.
    requires (!(sizeof(ElementOf<Tag>) < 4 &&
                sizeof(ElementOf<VecToTagT<Indices>>) == 8))
  static VECOPS_ALWAYS_INLINE Vec<Tag> call(
      LoadOp op, Tag tag, const ElementOf<Tag>* pointer,
      opt::Indexed<Indices, Scale> addressing, Temporality temporality) {
    return call(
        op, tag, pointer, addressing, execute(MaskFillOp{}, tag, true),
        execute(FillOp{}, tag, ElementOf<Tag>{}), temporality);
  }

  template <VectorValue Indices, int Scale, typename Temporality>
    requires (!(sizeof(ElementOf<Tag>) < 4 &&
                sizeof(ElementOf<VecToTagT<Indices>>) == 8))
  static VECOPS_ALWAYS_INLINE Vec<Tag> call(
      LoadOp, Tag tag, const ElementOf<Tag>* pointer,
      opt::Indexed<Indices, Scale> addressing,
      Mask<Tag> mask, Vec<Tag> inactive, Temporality) {
    using IndexTag = Rebind<ElementOf<VecToTagT<Indices>>, Tag>;
    constexpr int scale = Scale == 0 ? sizeof(ElementOf<Tag>) : Scale;
    return x86_load_indexed_native<scale, Tag, IndexTag>(
        tag, pointer, addressing.indices, mask, inactive);
  }
};

#if defined(CPU_CAPABILITY_AVX512)
template <int Scale, VectorTag Tag, VectorTag IndexTag>
  requires (sizeof(ElementOf<Tag>) >= 4)
VECOPS_ALWAYS_INLINE void x86_store_indexed_leaf(
    Tag, ElementOf<Tag>* pointer, Vec<Tag> value, Vec<IndexTag> indices,
    Mask<Tag> mask) {
  using T = ElementOf<Tag>;
  using I = ElementOf<IndexTag>;
  using Raw = typename RepresentationTraits<X86Backend, Tag>::RawVec;
  using IndexRaw = typename RepresentationTraits<X86Backend, IndexTag>::RawVec;
  if constexpr (sizeof(I) == 4 && sizeof(T) == 4) {
    if constexpr (sizeof(Raw) == 16) {
      if constexpr (std::same_as<T, float32_t>)
        _mm_mask_i32scatter_ps(pointer, mask.value, indices.value, value.value, Scale);
      else
        _mm_mask_i32scatter_epi32(pointer, mask.value, indices.value, value.value, Scale);
    } else if constexpr (sizeof(Raw) == 32) {
      if constexpr (std::same_as<T, float32_t>)
        _mm256_mask_i32scatter_ps(pointer, mask.value, indices.value, value.value, Scale);
      else
        _mm256_mask_i32scatter_epi32(pointer, mask.value, indices.value, value.value, Scale);
    } else if constexpr (std::same_as<T, float32_t>)
      _mm512_mask_i32scatter_ps(pointer, mask.value, indices.value, value.value, Scale);
    else
      _mm512_mask_i32scatter_epi32(pointer, mask.value, indices.value, value.value, Scale);
  } else if constexpr (sizeof(I) == 4) {
    if constexpr (sizeof(Raw) == 16) {
      if constexpr (std::same_as<T, float64_t>)
        _mm_mask_i32scatter_pd(pointer, mask.value, indices.value, value.value, Scale);
      else
        _mm_mask_i32scatter_epi64(pointer, mask.value, indices.value, value.value, Scale);
    } else if constexpr (sizeof(Raw) == 32) {
      if constexpr (std::same_as<T, float64_t>)
        _mm256_mask_i32scatter_pd(pointer, mask.value, indices.value, value.value, Scale);
      else
        _mm256_mask_i32scatter_epi64(pointer, mask.value, indices.value, value.value, Scale);
    } else if constexpr (std::same_as<T, float64_t>)
      _mm512_mask_i32scatter_pd(pointer, mask.value, indices.value, value.value, Scale);
    else
      _mm512_mask_i32scatter_epi64(pointer, mask.value, indices.value, value.value, Scale);
  } else if constexpr (sizeof(T) == 8) {
    if constexpr (sizeof(Raw) == 16) {
      if constexpr (std::same_as<T, float64_t>)
        _mm_mask_i64scatter_pd(pointer, mask.value, indices.value, value.value, Scale);
      else
        _mm_mask_i64scatter_epi64(pointer, mask.value, indices.value, value.value, Scale);
    } else if constexpr (sizeof(Raw) == 32) {
      if constexpr (std::same_as<T, float64_t>)
        _mm256_mask_i64scatter_pd(pointer, mask.value, indices.value, value.value, Scale);
      else
        _mm256_mask_i64scatter_epi64(pointer, mask.value, indices.value, value.value, Scale);
    } else if constexpr (std::same_as<T, float64_t>)
      _mm512_mask_i64scatter_pd(pointer, mask.value, indices.value, value.value, Scale);
    else
      _mm512_mask_i64scatter_epi64(pointer, mask.value, indices.value, value.value, Scale);
  } else {
    static_assert(sizeof(T) == 4 && sizeof(I) == 8);
    if constexpr (sizeof(IndexRaw) == 16) {
      if constexpr (std::same_as<T, float32_t>)
        _mm_mask_i64scatter_ps(pointer, mask.value, indices.value, value.value, Scale);
      else
        _mm_mask_i64scatter_epi32(pointer, mask.value, indices.value, value.value, Scale);
    } else if constexpr (sizeof(IndexRaw) == 32) {
      if constexpr (std::same_as<T, float32_t>)
        _mm256_mask_i64scatter_ps(pointer, mask.value, indices.value, value.value, Scale);
      else
        _mm256_mask_i64scatter_epi32(pointer, mask.value, indices.value, value.value, Scale);
    } else if constexpr (std::same_as<T, float32_t>)
      _mm512_mask_i64scatter_ps(pointer, mask.value, indices.value, value.value, Scale);
    else
      _mm512_mask_i64scatter_epi32(pointer, mask.value, indices.value, value.value, Scale);
  }
}

template <int Scale, VectorTag Tag, VectorTag IndexTag>
  requires (sizeof(ElementOf<Tag>) >= 4)
VECOPS_ALWAYS_INLINE void x86_store_indexed_native(
    Tag tag, ElementOf<Tag>* pointer, Vec<Tag> value,
    Vec<IndexTag> indices, Mask<Tag> mask) {
  using Traits = RepresentationTraits<X86Backend, Tag>;
  using IndexTraits = RepresentationTraits<X86Backend, IndexTag>;
  using WordTag = FixedTag<ElementOf<Tag>, Traits::word_lanes>;
  using IndexChunkTag = Rebind<ElementOf<IndexTag>, WordTag>;
  using IndexChunkTraits =
      RepresentationTraits<X86Backend, IndexChunkTag>;
  constexpr bool complete_output_words =
      Traits::logical_lanes == Traits::word_count * Traits::word_lanes;
  constexpr bool directly_partitioned_output =
      Traits::word_count > 1 && complete_output_words &&
      IndexTraits::word_count ==
          Traits::word_count * IndexChunkTraits::word_count;

  if constexpr (directly_partitioned_output) {
    [&]<std::size_t... OutputIndex>(std::index_sequence<OutputIndex...>) {
      ([&]() VECOPS_INLINE_LAMBDA {
        constexpr nint_t output_index =
            static_cast<nint_t>(OutputIndex);
        const auto index_chunk = construct_words<X86Backend>(
            IndexChunkTag{},
            [&]<nint_t InputIndex>(IndexChunkTag) VECOPS_INLINE_LAMBDA {
              constexpr nint_t source_index =
                  output_index * IndexChunkTraits::word_count + InputIndex;
              return ::vecops::vec::get_word<source_index>(
                  IndexTag{}, indices);
            });
        x86_store_indexed_native<Scale, WordTag, IndexChunkTag>(
            WordTag{}, pointer,
            ::vecops::vec::get_word<output_index>(tag, value),
            index_chunk,
            ::vecops::vec::get_word<output_index>(tag, mask));
      }(), ...);
    }(std::make_index_sequence<
        static_cast<std::size_t>(Traits::word_count)>{});
  } else if constexpr (num_words(tag) > 1 || num_words(IndexTag{}) > 1) {
    x86_store_indexed_native<Scale, Half<Tag>, Half<IndexTag>>(
        Half<Tag>{}, pointer, execute(LowerOp{}, tag, value),
        execute(LowerOp{}, IndexTag{}, indices),
        execute(LowerOp{}, tag, mask));
    x86_store_indexed_native<Scale, Half<Tag>, Half<IndexTag>>(
        Half<Tag>{}, pointer, execute(UpperOp{}, tag, value),
        execute(UpperOp{}, IndexTag{}, indices),
        execute(UpperOp{}, tag, mask));
  } else {
    x86_store_indexed_leaf<Scale, Tag, IndexTag>(
        tag, pointer, value, indices, mask);
  }
}

template <VectorTag Tag>
  requires (sizeof(ElementOf<Tag>) >= 4)
struct NativeImpl<X86Backend, StoreOp, Tag> {
  template <VectorValue Indices, int Scale, typename Temporality>
  static VECOPS_ALWAYS_INLINE void call(
      StoreOp op, Tag tag, ElementOf<Tag>* pointer, Vec<Tag> value,
      opt::Indexed<Indices, Scale> addressing, Temporality temporality) {
    call(
        op, tag, pointer, value, addressing,
        execute(MaskFillOp{}, tag, true), temporality);
  }

  template <VectorValue Indices, int Scale, typename Temporality>
  static VECOPS_ALWAYS_INLINE void call(
      StoreOp, Tag tag, ElementOf<Tag>* pointer, Vec<Tag> value,
      opt::Indexed<Indices, Scale> addressing, Mask<Tag> mask, Temporality) {
    using IndexTag = Rebind<ElementOf<VecToTagT<Indices>>, Tag>;
    constexpr int scale = Scale == 0 ? sizeof(ElementOf<Tag>) : Scale;
    x86_store_indexed_native<scale, Tag, IndexTag>(
        tag, pointer, value, addressing.indices, mask);
  }
};
#endif
#endif

template <>
struct NativeWordImpl<X86Backend, LoadOp> {
  template <nint_t Index, VectorTag Tag,
            typename Alignment, typename Temporality>
    requires ((IsAlignedOption<Alignment>::value ||
               IsUnalignedOption<Alignment>::value) &&
              (IsTemporalOption<Temporality>::value ||
               IsNonTemporalOption<Temporality>::value))
  static VECOPS_ALWAYS_INLINE NativeWordVec<Tag> call(
      LoadOp, Tag, const ElementOf<Tag>* pointer,
      Alignment alignment, Temporality temporality) {
    using Traits = RepresentationTraits<X86Backend, Tag>;
    using T = ElementOf<Tag>;
    using Raw = typename Traits::RawVec;
    using MaskRaw = typename Traits::RawMask;
    constexpr nint_t valid = x86_valid_word_lanes<Index, Tag>();
    if constexpr (valid == Traits::word_lanes) {
      return NativeWordVec<Tag>{x86_load_memory_word<Raw>(
          pointer, alignment, temporality)};
    } else {
      const auto zero = execute_word<Index, X86Backend>(FillOp{}, Tag{}, T{});
      return NativeWordVec<Tag>{x86_masked_load_memory_word<T>(
          pointer, x86_mask_prefix<T, MaskRaw>(valid), zero.value)};
    }
  }

  template <nint_t Index, VectorTag Tag,
            typename Alignment, typename Temporality>
    requires ((IsAlignedOption<Alignment>::value ||
               IsUnalignedOption<Alignment>::value) &&
              (IsTemporalOption<Temporality>::value ||
               IsNonTemporalOption<Temporality>::value))
  static VECOPS_ALWAYS_INLINE NativeWordVec<Tag> call(
      LoadOp, Tag, const ElementOf<Tag>* pointer,
      NativeWordMask<Tag> mask, NativeWordVec<Tag> inactive,
      Alignment, Temporality) {
    using T = ElementOf<Tag>;
    return NativeWordVec<Tag>{x86_masked_load_memory_word<T>(
        pointer, mask.value, inactive.value)};
  }
};

template <>
struct NativeWordImpl<X86Backend, StoreOp> {
  template <nint_t Index, VectorTag Tag,
            typename Alignment, typename Temporality>
    requires ((IsAlignedOption<Alignment>::value ||
               IsUnalignedOption<Alignment>::value) &&
              (IsTemporalOption<Temporality>::value ||
               IsNonTemporalOption<Temporality>::value))
  static VECOPS_ALWAYS_INLINE void call(
      StoreOp, Tag, ElementOf<Tag>* pointer, NativeWordVec<Tag> value,
      Alignment alignment, Temporality temporality) {
    using Traits = RepresentationTraits<X86Backend, Tag>;
    using T = ElementOf<Tag>;
    using MaskRaw = typename Traits::RawMask;
    constexpr nint_t valid = x86_valid_word_lanes<Index, Tag>();
    if constexpr (valid == Traits::word_lanes) {
      x86_store_memory_word(pointer, value.value, alignment, temporality);
    } else {
      x86_masked_store_memory_word<T>(
          pointer, x86_mask_prefix<T, MaskRaw>(valid), value.value);
    }
  }

  template <nint_t Index, VectorTag Tag,
            typename Alignment, typename Temporality>
    requires ((IsAlignedOption<Alignment>::value ||
               IsUnalignedOption<Alignment>::value) &&
              (IsTemporalOption<Temporality>::value ||
               IsNonTemporalOption<Temporality>::value))
  static VECOPS_ALWAYS_INLINE void call(
      StoreOp, Tag, ElementOf<Tag>* pointer,
      NativeWordMask<Tag> mask, NativeWordVec<Tag> value,
      Alignment, Temporality) {
    x86_masked_store_memory_word<ElementOf<Tag>>(
        pointer, mask.value, value.value);
  }
};

} // namespace vecops::vec::details

#endif // VECOPS_VEC_DETAILS_X86_MEMORY_H
