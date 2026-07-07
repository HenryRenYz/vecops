//
// Created by renyz on 2026/3/25.
//

#ifndef VECOPS_X86_LOADSTORE_H
#define VECOPS_X86_LOADSTORE_H

#include "./x86_Types.h"
#include "./x86_Basic.h"
#include "./x86_Arithmetic.h"

//@formatter:off
namespace vecops::vec::CPU_CAPABILITY {
namespace word {

#if defined(COMPILER_GCC) || defined(COMPILER_CLANG)
#define VECOPS_X86_NO_ASAN __attribute__((no_sanitize_address))
#else
#define VECOPS_X86_NO_ASAN
#endif

/* ************************************************************************** */
//                              Consecutive Load                              //
/* ************************************************************************** */
namespace details {
#ifdef HAS_AVX512DQ
template <TLV_DECL_TAG(T)>
VECOPS_VFUNC Mask<T> restrict_mask_range(T t, Mask<T> m) {
  if constexpr (T::Bytes == 16 || T::Bytes == 32 || T::Bytes == 64) {
    return m;
  }
  constexpr nint_t N = size(t);
  constexpr uint64_t M = N == 64 ? -1 : (uint64_t(1) << N) - 1;
  if constexpr (N > 32) {
    return m.v & M;
  } else {
    return m.v & uint32_t(M);
  }
}
#else // HAS_AVX512DQ
template <TLV_DECL_TAG(T), TL_IF(T::Bytes == 1)>
VECOPS_VFUNC Mask<T> restrict_mask_range(T t, Mask<T> m) {
  return _mm_and_si128(m.v, _mm_set_epi32(0, 0, 0, 0x000000FF));
}
template <TLV_DECL_TAG(T), TL_IF(T::Bytes == 2)>
VECOPS_VFUNC Mask<T> restrict_mask_range(T t, Mask<T> m) {
  return _mm_and_si128(m.v, _mm_set_epi32(0, 0, 0, 0x0000FFFF));
}
template <TLV_DECL_TAG(T), TL_IF(T::Bytes == 4)>
VECOPS_VFUNC Mask<T> restrict_mask_range(T t, Mask<T> m) {
  return _mm_and_si128(m.v, _mm_set_epi32(0, 0, 0, (int)0xFFFFFFFF));
}
template <TLV_DECL_TAG(T), TL_IF(T::Bytes == 8)>
VECOPS_VFUNC Mask<T> restrict_mask_range(T t, Mask<T> m) {
  return _mm_move_epi64(m.v);
}
template <TLV_DECL_TAG(T), TL_IF(T::Bytes == 16)>
VECOPS_VFUNC Mask<T> restrict_mask_range(T t, Mask<T> m) {
  return m;
}
#endif // HAS_AVX512DQ

template <TLV_DECL_TAG(T)>
VECOPS_VFUNC Vec<T> i32_set1(T, int32_t x) {
  Vec<T> r;
  if constexpr (T::Bytes <= 16)
    r.v = _mm_set1_epi32(x);
  else if constexpr (T::Bytes == 32)
    r.v = _mm256_set1_epi32(x);
  else
    r.v = _mm512_set1_epi32(x);
  return r;
}

template <TLV_DECL_TAG(T)>
VECOPS_VFUNC Vec<T> i32_add(Vec<T> a, Vec<T> b) {
  if constexpr (T::Bytes <= 16)
    return _mm_add_epi32(a.v, b.v);
  else if constexpr (T::Bytes == 32)
    return _mm256_add_epi32(a.v, b.v);
  else
    return _mm512_add_epi32(a.v, b.v);
}

template <TLV_DECL_TAG(T)>
VECOPS_VFUNC Vec<T> i32_and(Vec<T> a, Vec<T> b) {
  if constexpr (T::Bytes <= 16)
    return _mm_and_si128(a.v, b.v);
  else if constexpr (T::Bytes == 32)
    return _mm256_and_si256(a.v, b.v);
  else
    return _mm512_and_si512(a.v, b.v);
}

template <TLV_DECL_TAG(T)>
VECOPS_VFUNC Vec<T> i32_andnot(Vec<T> a, Vec<T> b) {
  if constexpr (T::Bytes <= 16)
    return _mm_andnot_si128(a.v, b.v);
  else if constexpr (T::Bytes == 32)
    return _mm256_andnot_si256(a.v, b.v);
  else
    return _mm512_andnot_si512(a.v, b.v);
}

template <TLV_DECL_TAG(T)>
VECOPS_VFUNC Vec<T> i32_or(Vec<T> a, Vec<T> b) {
  if constexpr (T::Bytes <= 16)
    return _mm_or_si128(a.v, b.v);
  else if constexpr (T::Bytes == 32)
    return _mm256_or_si256(a.v, b.v);
  else
    return _mm512_or_si512(a.v, b.v);
}

template <TLV_DECL_TAG(T)>
VECOPS_VFUNC Vec<T> i32_xor(Vec<T> a, Vec<T> b) {
  if constexpr (T::Bytes <= 16)
    return _mm_xor_si128(a.v, b.v);
  else if constexpr (T::Bytes == 32)
    return _mm256_xor_si256(a.v, b.v);
  else
    return _mm512_xor_si512(a.v, b.v);
}

template <int Shift, TLV_DECL_TAG(T)>
VECOPS_VFUNC Vec<T> i32_slli(Vec<T> a) {
  if constexpr (T::Bytes <= 16)
    return _mm_slli_epi32(a.v, Shift);
  else if constexpr (T::Bytes == 32)
    return _mm256_slli_epi32(a.v, Shift);
  else
    return _mm512_slli_epi32(a.v, Shift);
}

template <TLV_DECL_TAG(T)>
VECOPS_VFUNC Vec<T> i32_srlv(Vec<T> a, Vec<T> shift) {
  if constexpr (T::Bytes <= 16)
    return _mm_srlv_epi32(a.v, shift.v);
  else if constexpr (T::Bytes == 32)
    return _mm256_srlv_epi32(a.v, shift.v);
  else
    return _mm512_srlv_epi32(a.v, shift.v);
}

template <TLV_DECL_TAG(T)>
VECOPS_VFUNC Vec<T> i32_sllv(Vec<T> a, Vec<T> shift) {
  if constexpr (T::Bytes <= 16)
    return _mm_sllv_epi32(a.v, shift.v);
  else if constexpr (T::Bytes == 32)
    return _mm256_sllv_epi32(a.v, shift.v);
  else
    return _mm512_sllv_epi32(a.v, shift.v);
}

template <TLV_DECL_TAG(T)>
VECOPS_X86_NO_ASAN VECOPS_VFUNC Vec<T> i32_gather_bytes(T t, const void* p, Vec<T> byte_offsets) {
  Vec<T> r;
  if constexpr (T::Bytes <= 16)
    r.v = _mm_i32gather_epi32(reinterpret_cast<const int*>(p), byte_offsets.v, 1);
  else if constexpr (T::Bytes == 32)
    r.v = _mm256_i32gather_epi32(reinterpret_cast<const int*>(p), byte_offsets.v, 1);
  else
    r.v = _mm512_i32gather_epi32(byte_offsets.v, p, 1);
  return r;
}

template <TLV_DECL_TAG(TI)>
VECOPS_VFUNC void subint1_offsets(TI t_i, const void* p, Vec<TI> i,
                                  Vec<TI>& i_base, Vec<TI>& i_sel) {
  auto r_32 = details::i32_set1(t_i, int32_t(nuint_t(p) & 3));
  auto i_b32 = details::i32_add<TI>(i, r_32);
  i_base = details::i32_and<TI>(i_b32, details::i32_set1(t_i, ~int32_t(3)));
  i_sel = details::i32_slli<3, TI>(details::i32_and<TI>(i_b32, details::i32_set1(t_i, 3)));
}

template <TLV_DECL_TAG(TI)>
VECOPS_VFUNC void subint2_offsets(TI t_i, const void* p, Vec<TI> i,
                                  Vec<TI>& i_base, Vec<TI>& i_sel) {
  auto r_32 = details::i32_set1(t_i, int32_t(nuint_t(p) & 3));
  auto i_b32 = details::i32_add<TI>(details::i32_slli<1, TI>(i), r_32);
  auto is_3 = details::i32_and<TI>(
      details::i32_and<TI>(i_b32, details::i32_slli<1, TI>(i_b32)),
      details::i32_set1(t_i, 2));
  auto m_tail = details::i32_xor<TI>(is_3, details::i32_set1(t_i, 3));
  i_base = details::i32_andnot<TI>(m_tail, i_b32);
  i_sel = details::i32_slli<3, TI>(details::i32_and<TI>(i_b32, m_tail));
}

template <TLV_DECL_TAG(TI)>
VECOPS_X86_NO_ASAN VECOPS_VFUNC Vec<TI> subint1_gather_i32(TI t_i, const void* p, Vec<TI> i) {
#ifdef VECOPS_X86_ENABLE_UNSAFE_GATHER
  return details::i32_gather_bytes(t_i, p, i);
#else
  Vec<TI> i_base, i_sel;
  details::subint1_offsets(t_i, p, i, i_base, i_sel);
  auto g = details::i32_gather_bytes(t_i, reinterpret_cast<const void*>(nuint_t(p) & ~nuint_t(3)), i_base);
  return details::i32_and<TI>(details::i32_srlv<TI>(g, i_sel), details::i32_set1(t_i, 0xff));
#endif
}

template <TLV_DECL_TAG(TI)>
VECOPS_X86_NO_ASAN VECOPS_VFUNC Vec<TI> subint2_gather_i32(TI t_i, const void* p, Vec<TI> i) {
#ifdef VECOPS_X86_ENABLE_UNSAFE_GATHER
  auto i_b = details::i32_slli<1, TI>(i);
  return details::i32_gather_bytes(t_i, p, i_b);
#else
  Vec<TI> i_base, i_sel;
  details::subint2_offsets(t_i, p, i, i_base, i_sel);
  auto g = details::i32_gather_bytes(t_i, reinterpret_cast<const void*>(nuint_t(p) & ~nuint_t(3)), i_base);
  return details::i32_and<TI>(details::i32_srlv<TI>(g, i_sel), details::i32_set1(t_i, 0xffff));
#endif
}

#undef VECOPS_X86_NO_ASAN
} // namespace details

template <TLV_DECL_TAG(T), TL_IF(T::Bytes == 8), TL_IF(is_any<TypeOf<T>, float64_t>)>
VECOPS_VFUNC Vec<T> loadu(T t, const TypeOf<T> * p) {
  return _mm_load_sd(p);
}
template <TLV_DECL_TAG(T), TL_IF(T::Bytes == 16), TL_IF(is_any<TypeOf<T>, float64_t>)>
VECOPS_VFUNC Vec<T> loadu(T t, const TypeOf<T> * p) {
  return _mm_loadu_pd(p);
}
template <TLV_DECL_TAG(T), TL_IF(T::Bytes == 4), TL_IF(is_any<TypeOf<T>, float32_t>)>
VECOPS_VFUNC Vec<T> loadu(T t, const TypeOf<T> * p) {
  return _mm_load_ss(p);
}
template <TLV_DECL_TAG(T), TL_IF(T::Bytes == 8), TL_IF(is_any<TypeOf<T>, float32_t>)>
VECOPS_VFUNC Vec<T> loadu(T t, const TypeOf<T> * p) {
  Tag<float64_t, 1> t1;
  return word::bitcast(t, word::loadu(t1, (const float64_t *) p));
}
template <TLV_DECL_TAG(T), TL_IF(T::Bytes == 16), TL_IF(is_any<TypeOf<T>, float32_t>)>
VECOPS_VFUNC Vec<T> loadu(T t, const TypeOf<T> * p) {
  return _mm_loadu_ps(p);
}
template <TLV_DECL_TAG(T), TL_IF(T::Bytes == 1), TL_IF(is_int<TypeOf<T>> || is_small_float<TypeOf<T>>)>
VECOPS_VFUNC Vec<T> loadu(T t, const TypeOf<T> * p) {
  return _mm_cvtsi32_si128((int32_t)((const uint8_t *) p)[0]);
}
template <TLV_DECL_TAG(T), TL_IF(T::Bytes == 2), TL_IF(is_int<TypeOf<T>> || is_small_float<TypeOf<T>>)>
VECOPS_VFUNC Vec<T> loadu(T t, const TypeOf<T> * p) {
  return _mm_cvtsi32_si128((int32_t)((const uint16_t *) p)[0]);
}
template <TLV_DECL_TAG(T), TL_IF(T::Bytes == 4), TL_IF(is_int<TypeOf<T>> || is_small_float<TypeOf<T>>)>
VECOPS_VFUNC Vec<T> loadu(T t, const TypeOf<T> * p) {
  return _mm_cvtsi32_si128(((const int32_t *) p)[0]);
}
template <TLV_DECL_TAG(T), TL_IF(T::Bytes == 8), TL_IF(is_int<TypeOf<T>> || is_small_float<TypeOf<T>>)>
__attribute__((optimize("O1"))) // Note: O2+ triggers compiler bug in GCC 13
VECOPS_VFUNC Vec<T> loadu(T t, const TypeOf<T> * p) {
  return _mm_cvtsi64_si128(((const int64_t *) p)[0]);
}
template <TLV_DECL_TAG(T), TL_IF(T::Bytes == 16), TL_IF(is_int<TypeOf<T>> || is_small_float<TypeOf<T>>)>
VECOPS_VFUNC Vec<T> loadu(T t, const TypeOf<T> * p) {
  return _mm_loadu_si128((const __m128i*) p);
}


template <TLV_DECL_TAG(T), TL_IF(T::Bytes == 16), TL_IF(is_any<TypeOf<T>, float64_t>)>
VECOPS_VFUNC Vec<T> load(T t, const TypeOf<T> * p) {
  VECOPS_ASSERT(is_aligned(T::Bytes, p), "Not aligned");
  return _mm_load_pd(p);
}
template <TLV_DECL_TAG(T), TL_IF(T::Bytes == 16), TL_IF(is_any<TypeOf<T>, float32_t>)>
VECOPS_VFUNC Vec<T> load(T t, const TypeOf<T> * p) {
  VECOPS_ASSERT(is_aligned(T::Bytes, p), "Not aligned");
  return _mm_load_ps(p);
}
template <TLV_DECL_TAG(T), TL_IF(T::Bytes == 16), TL_IF(is_int<TypeOf<T>> || is_small_float<TypeOf<T>>)>
VECOPS_VFUNC Vec<T> load(T t, const TypeOf<T> * p) {
  VECOPS_ASSERT(is_aligned(T::Bytes, p), "Not aligned");
  return _mm_load_si128((const __m128i*) p);
}
template <TLV_DECL_TAG(T), TL_IF(T::Bytes < 16)>
VECOPS_VFUNC Vec<T> load(T t, const TypeOf<T> * p) {
  VECOPS_ASSERT(is_aligned(T::Bytes, p), "Not aligned");
  return word::loadu(t, p);
}


#if VEC_WIDTH >= 256
template <TLV_DECL_TAG(T), TL_IF(T::Bytes == 32), TL_IF(is_any<TypeOf<T>, float64_t>)>
VECOPS_VFUNC Vec<T> loadu(T t, const TypeOf<T> * p) {
  return _mm256_loadu_pd(p);
}
template <TLV_DECL_TAG(T), TL_IF(T::Bytes == 32), TL_IF(is_any<TypeOf<T>, float32_t>)>
VECOPS_VFUNC Vec<T> loadu(T t, const TypeOf<T> * p) {
  return _mm256_loadu_ps(p);
}
template <TLV_DECL_TAG(T), TL_IF(T::Bytes == 32), TL_IF(is_int<TypeOf<T>> || is_small_float<TypeOf<T>>)>
VECOPS_VFUNC Vec<T> loadu(T t, const TypeOf<T> * p) {
  return _mm256_loadu_si256((const __m256i*) p);
}

template <TLV_DECL_TAG(T), TL_IF(T::Bytes == 32), TL_IF(is_any<TypeOf<T>, float64_t>)>
VECOPS_VFUNC Vec<T> load(T t, const TypeOf<T> * p) {
  VECOPS_ASSERT(is_aligned(T::Bytes, p), "Not aligned");
  return _mm256_load_pd(p);
}
template <TLV_DECL_TAG(T), TL_IF(T::Bytes == 32), TL_IF(is_any<TypeOf<T>, float32_t>)>
VECOPS_VFUNC Vec<T> load(T t, const TypeOf<T> * p) {
  VECOPS_ASSERT(is_aligned(T::Bytes, p), "Not aligned");
  return _mm256_load_ps(p);
}
template <TLV_DECL_TAG(T), TL_IF(T::Bytes == 32), TL_IF(is_int<TypeOf<T>> || is_small_float<TypeOf<T>>)>
VECOPS_VFUNC Vec<T> load(T t, const TypeOf<T> * p) {
  VECOPS_ASSERT(is_aligned(T::Bytes, p), "Not aligned");
  return _mm256_load_si256((const __m256i*) p);
}
#endif // VEC_WIDTH >= 256

#if VEC_WIDTH >= 512
template <TLV_DECL_TAG(T), TL_IF(T::Bytes == 64), TL_IF(is_any<TypeOf<T>, float64_t>)>
VECOPS_VFUNC Vec<T> loadu(T t, const TypeOf<T> * p) {
  return _mm512_loadu_pd(p);
}
template <TLV_DECL_TAG(T), TL_IF(T::Bytes == 64), TL_IF(is_any<TypeOf<T>, float32_t>)>
VECOPS_VFUNC Vec<T> loadu(T t, const TypeOf<T> * p) {
  return _mm512_loadu_ps(p);
}
template <TLV_DECL_TAG(T), TL_IF(T::Bytes == 64), TL_IF(is_int<TypeOf<T>> || is_small_float<TypeOf<T>>)>
VECOPS_VFUNC Vec<T> loadu(T t, const TypeOf<T> * p) {
  return _mm512_loadu_si512((const __m256i*) p);
}

template <TLV_DECL_TAG(T), TL_IF(T::Bytes == 64), TL_IF(is_any<TypeOf<T>, float64_t>)>
VECOPS_VFUNC Vec<T> load(T t, const TypeOf<T> * p) {
  VECOPS_ASSERT(is_aligned(T::Bytes, p), "Not aligned");
  return _mm512_load_pd(p);
}
template <TLV_DECL_TAG(T), TL_IF(T::Bytes == 64), TL_IF(is_any<TypeOf<T>, float32_t>)>
VECOPS_VFUNC Vec<T> load(T t, const TypeOf<T> * p) {
  VECOPS_ASSERT(is_aligned(T::Bytes, p), "Not aligned");
  return _mm512_load_ps(p);
}
template <TLV_DECL_TAG(T), TL_IF(T::Bytes == 64), TL_IF(is_int<TypeOf<T>> || is_small_float<TypeOf<T>>)>
VECOPS_VFUNC Vec<T> load(T t, const TypeOf<T> * p) {
  VECOPS_ASSERT(is_aligned(T::Bytes, p), "Not aligned");
  return _mm512_load_si512((const __m512i*) p);
}
#endif // VEC_WIDTH >= 512


#ifdef HAS_AVX512DQ
template <TLV_DECL_TAG(T), TL_IF(T::Bytes <= 16), TL_IF(is_any<TypeOf<T>, float32_t>)>
VECOPS_VFUNC Vec<T> loadu(T t, const TypeOf<T> * p, Mask<T> m, Vec<T> default_v) {
  return _mm_mask_loadu_ps(default_v.v, details::restrict_mask_range(t, m).v, p);
}
template <TLV_DECL_TAG(T), TL_IF(T::Bytes <= 16), TL_IF(is_any<TypeOf<T>, float64_t>)>
VECOPS_VFUNC Vec<T> loadu(T t, const TypeOf<T> * p, Mask<T> m, Vec<T> default_v) {
  return _mm_mask_loadu_pd(default_v.v, details::restrict_mask_range(t, m).v, p);
}
template <TLV_DECL_TAG(T), TL_IF(T::Bytes <= 16), TL_IF(is_any<TypeOf<T>, int8_t, uint8_t>)>
VECOPS_VFUNC Vec<T> loadu(T t, const TypeOf<T> * p, Mask<T> m, Vec<T> default_v) {
  return _mm_mask_loadu_epi8(default_v.v, details::restrict_mask_range(t, m).v, p);
}
template <TLV_DECL_TAG(T), TL_IF(T::Bytes <= 16), TL_IF(is_any<TypeOf<T>, int16_t, uint16_t, float16_t, bfloat16_t>)>
VECOPS_VFUNC Vec<T> loadu(T t, const TypeOf<T> * p, Mask<T> m, Vec<T> default_v) {
  return _mm_mask_loadu_epi16(default_v.v, details::restrict_mask_range(t, m).v, p);
}
template <TLV_DECL_TAG(T), TL_IF(T::Bytes <= 16), TL_IF(is_any<TypeOf<T>, int32_t, uint32_t>)>
VECOPS_VFUNC Vec<T> loadu(T t, const TypeOf<T> * p, Mask<T> m, Vec<T> default_v) {
  return _mm_mask_loadu_epi32(default_v.v, details::restrict_mask_range(t, m).v, p);
}
template <TLV_DECL_TAG(T), TL_IF(T::Bytes <= 16), TL_IF(is_any<TypeOf<T>, int64_t, uint64_t>)>
VECOPS_VFUNC Vec<T> loadu(T t, const TypeOf<T> * p, Mask<T> m, Vec<T> default_v) {
  return _mm_mask_loadu_epi64(default_v.v, details::restrict_mask_range(t, m).v, p);
}

template <TLV_DECL_TAG(T), TL_IF(T::Bytes == 32), TL_IF(is_any<TypeOf<T>, float32_t>)>
VECOPS_VFUNC Vec<T> loadu(T t, const TypeOf<T> * p, Mask<T> m, Vec<T> default_v) {
  return _mm256_mask_loadu_ps(default_v.v, m.v, p);
}
template <TLV_DECL_TAG(T), TL_IF(T::Bytes == 32), TL_IF(is_any<TypeOf<T>, float64_t>)>
VECOPS_VFUNC Vec<T> loadu(T t, const TypeOf<T> * p, Mask<T> m, Vec<T> default_v) {
  return _mm256_mask_loadu_pd(default_v.v, m.v, p);
}
template <TLV_DECL_TAG(T), TL_IF(T::Bytes == 32), TL_IF(is_any<TypeOf<T>, int8_t, uint8_t>)>
VECOPS_VFUNC Vec<T> loadu(T t, const TypeOf<T> * p, Mask<T> m, Vec<T> default_v) {
  return _mm256_mask_loadu_epi8(default_v.v, m.v, p);
}
template <TLV_DECL_TAG(T), TL_IF(T::Bytes == 32), TL_IF(is_any<TypeOf<T>, int16_t, uint16_t, float16_t, bfloat16_t>)>
VECOPS_VFUNC Vec<T> loadu(T t, const TypeOf<T> * p, Mask<T> m, Vec<T> default_v) {
  return _mm256_mask_loadu_epi16(default_v.v, m.v, p);
}
template <TLV_DECL_TAG(T), TL_IF(T::Bytes == 32), TL_IF(is_any<TypeOf<T>, int32_t, uint32_t>)>
VECOPS_VFUNC Vec<T> loadu(T t, const TypeOf<T> * p, Mask<T> m, Vec<T> default_v) {
  return _mm256_mask_loadu_epi32(default_v.v, m.v, p);
}
template <TLV_DECL_TAG(T), TL_IF(T::Bytes == 32), TL_IF(is_any<TypeOf<T>, int64_t, uint64_t>)>
VECOPS_VFUNC Vec<T> loadu(T t, const TypeOf<T> * p, Mask<T> m, Vec<T> default_v) {
  return _mm256_mask_loadu_epi64(default_v.v, m.v, p);
}

template <TLV_DECL_TAG(T), TL_IF(T::Bytes == 64), TL_IF(is_any<TypeOf<T>, float32_t>)>
VECOPS_VFUNC Vec<T> loadu(T t, const TypeOf<T> * p, Mask<T> m, Vec<T> default_v) {
  return _mm512_mask_loadu_ps(default_v.v, m.v, p);
}
template <TLV_DECL_TAG(T), TL_IF(T::Bytes == 64), TL_IF(is_any<TypeOf<T>, float64_t>)>
VECOPS_VFUNC Vec<T> loadu(T t, const TypeOf<T> * p, Mask<T> m, Vec<T> default_v) {
  return _mm512_mask_loadu_pd(default_v.v, m.v, p);
}
template <TLV_DECL_TAG(T), TL_IF(T::Bytes == 64), TL_IF(is_any<TypeOf<T>, int8_t, uint8_t>)>
VECOPS_VFUNC Vec<T> loadu(T t, const TypeOf<T> * p, Mask<T> m, Vec<T> default_v) {
  return _mm512_mask_loadu_epi8(default_v.v, m.v, p);
}
template <TLV_DECL_TAG(T), TL_IF(T::Bytes == 64), TL_IF(is_any<TypeOf<T>, int16_t, uint16_t, float16_t, bfloat16_t>)>
VECOPS_VFUNC Vec<T> loadu(T t, const TypeOf<T> * p, Mask<T> m, Vec<T> default_v) {
  return _mm512_mask_loadu_epi16(default_v.v, m.v, p);
}
template <TLV_DECL_TAG(T), TL_IF(T::Bytes == 64), TL_IF(is_any<TypeOf<T>, int32_t, uint32_t>)>
VECOPS_VFUNC Vec<T> loadu(T t, const TypeOf<T> * p, Mask<T> m, Vec<T> default_v) {
  return _mm512_mask_loadu_epi32(default_v.v, m.v, p);
}
template <TLV_DECL_TAG(T), TL_IF(T::Bytes == 64), TL_IF(is_any<TypeOf<T>, int64_t, uint64_t>)>
VECOPS_VFUNC Vec<T> loadu(T t, const TypeOf<T> * p, Mask<T> m, Vec<T> default_v) {
  return _mm512_mask_loadu_epi64(default_v.v, m.v, p);
}
#else // HAS_AVX512DQ
template <TLV_DECL_TAG(T), TL_IF(T::Bytes <= 16), TL_IF(is_any<TypeOf<T>, float32_t>)>
VECOPS_VFUNC Vec<T> loadu(T t, const TypeOf<T> * p, Mask<T> m, Vec<T> default_v) {
  return word::blend(default_v, m, Vec<T>{
    _mm_maskload_ps(p, details::restrict_mask_range(t, m).v)
  });
}
template <TLV_DECL_TAG(T), TL_IF(T::Bytes <= 16), TL_IF(is_any<TypeOf<T>, float64_t>)>
VECOPS_VFUNC Vec<T> loadu(T t, const TypeOf<T> * p, Mask<T> m, Vec<T> default_v) {
  return word::blend(default_v, m, Vec<T>{
      _mm_maskload_pd(p, details::restrict_mask_range(t, m).v)
  });
}
template <TLV_DECL_TAG(T), TL_IF(T::Bytes <= 16), TL_IF(is_any<TypeOf<T>, int32_t, uint32_t>)>
VECOPS_VFUNC Vec<T> loadu(T t, const TypeOf<T> * p, Mask<T> m, Vec<T> default_v) {
  #ifdef HAS_AVX2
  return word::blend(default_v, m, Vec<T>{
      _mm_maskload_epi32((const int *)p, details::restrict_mask_range(t, m).v)
  });
  #else
  Rebind<float32_t, T> t1;
  return word::bitcast(t, word::loadu(t1, (const float32_t *) p, m.v, word::bitcast(t1, default_v)));
  #endif
}
template <TLV_DECL_TAG(T), TL_IF(T::Bytes <= 16), TL_IF(is_any<TypeOf<T>, int64_t, uint64_t>)>
VECOPS_VFUNC Vec<T> loadu(T t, const TypeOf<T> * p, Mask<T> m, Vec<T> default_v) {
  #ifdef HAS_AVX2
  return word::blend(default_v, m, Vec<T>{
      _mm_maskload_epi64((const long long *)p, details::restrict_mask_range(t, m).v)
  });
  #else
  Rebind<float64_t, T> t1;
  return word::bitcast(t, word::loadu(t1, (const float64_t *) p, m.v, word::bitcast(t1, default_v)));
  #endif
}
template <TLV_DECL_TAG(T), TL_IF(T::Bytes <= 16), TL_IF(is_any<TypeOf<T>, int16_t, uint16_t, float16_t, bfloat16_t>)>
VECOPS_VFUNC Vec<T> loadu(T t, const TypeOf<T> * p, Mask<T> m, Vec<T> default_v) {
  // if p + 15 still in the same page as p
  if (((nuint_t(p) & 0xfff) + 15) <= 0xfff) {
    // excessive reads are safe
    return word::blend(default_v, m, word::loadu(t, p));
  } else {
    // Fallback to scalar implementation, TODO slow
    auto mask = details::restrict_mask_range(t, m);
    union { int16_t i[8]; __m128i m; } V{.m = default_v.v}, M{.m = mask.v};
    alignas(16) int16_t S[8];
    auto P = (const int16_t*) p;
    for (int i = 0; i < 8; ++i) S[i] = M.i[i] ? P[i] : V.i[i];
    return _mm_load_si128((const __m128i *)S);
  }
}
template <TLV_DECL_TAG(T), TL_IF(T::Bytes <= 16), TL_IF(is_any<TypeOf<T>, int8_t, uint8_t>)>
VECOPS_VFUNC Vec<T> loadu(T t, const TypeOf<T> * p, Mask<T> m, Vec<T> default_v) {
  // if p + 15 still in the same page as p
  if (((nuint_t(p) & 0xfff) + 15) <= 0xfff) {
    // excessive reads are safe
    return word::blend(default_v, m, word::loadu(t, p));
  } else {
    // Fallback to scalar implementation, TODO slow
    auto mask = details::restrict_mask_range(t, m);
    union { int8_t i[16]; __m128i m; } V{.m = default_v.v}, M{.m = mask.v};
    alignas(16) int8_t S[16];
    auto P = (const int8_t*) p;
    for (int i = 0; i < 16; ++i) S[i] = M.i[i] ? P[i] : V.i[i];
    return _mm_load_si128((const __m128i *)S);
  }
}

#if VEC_WIDTH >= 256
template <TLV_DECL_TAG(T), TL_IF(T::Bytes == 32), TL_IF(is_any<TypeOf<T>, float32_t>)>
VECOPS_VFUNC Vec<T> loadu(T t, const TypeOf<T> * p, Mask<T> m, Vec<T> default_v) {
  return word::blend(default_v, m, Vec<T>{_mm256_maskload_ps(p, m.v)});
}
template <TLV_DECL_TAG(T), TL_IF(T::Bytes == 32), TL_IF(is_any<TypeOf<T>, float64_t>)>
VECOPS_VFUNC Vec<T> loadu(T t, const TypeOf<T> * p, Mask<T> m, Vec<T> default_v) {
  return word::blend(default_v, m, Vec<T>{_mm256_maskload_pd(p, m.v)});
}
template <TLV_DECL_TAG(T), TL_IF(T::Bytes == 32), TL_IF(is_any<TypeOf<T>, int32_t, uint32_t>)>
VECOPS_VFUNC Vec<T> loadu(T t, const TypeOf<T> * p, Mask<T> m, Vec<T> default_v) {
  return word::blend(default_v, m, Vec<T>{_mm256_maskload_epi32((const int *)p, m.v)});
}
template <TLV_DECL_TAG(T), TL_IF(T::Bytes == 32), TL_IF(is_any<TypeOf<T>, int64_t, uint64_t>)>
VECOPS_VFUNC Vec<T> loadu(T t, const TypeOf<T> * p, Mask<T> m, Vec<T> default_v) {
  return word::blend(default_v, m, Vec<T>{_mm256_maskload_epi64((const long long *)p, m.v)});
}
template <TLV_DECL_TAG(T), TL_IF(T::Bytes == 32), TL_IF(is_any<TypeOf<T>, int16_t, uint16_t, float16_t, bfloat16_t>)>
VECOPS_VFUNC Vec<T> loadu(T t, const TypeOf<T> * p, Mask<T> m, Vec<T> default_v) {
  // if p + 15 still in the same page as p
  if (((nuint_t(p) & 0xfff) + 31) <= 0xfff) {
    // excessive reads are safe
    return word::blend(default_v, m, word::loadu(t, p));
  } else {
    // Fallback to scalar implementation, TODO slow
    union { int16_t i[16]; __m256i m; } V{.m = default_v.v}, M{.m = m.v};
    alignas(16) int16_t S[16];
    auto P = (const int16_t*) p;
    for (int i = 0; i < 16; ++i) S[i] = M.i[i] ? P[i] : V.i[i];
    return _mm256_load_si256((const __m256i *)S);
  }
}
template <TLV_DECL_TAG(T), TL_IF(T::Bytes == 32), TL_IF(is_any<TypeOf<T>, int8_t, uint8_t>)>
VECOPS_VFUNC Vec<T> loadu(T t, const TypeOf<T> * p, Mask<T> m, Vec<T> default_v) {
  // if p + 31 still in the same page as p
  if (((nuint_t(p) & 0xfff) + 31) <= 0xfff) {
    // excessive reads are safe
    return word::blend(default_v, m, word::loadu(t, p));
  } else {
    // Fallback to scalar implementation, TODO slow
    union { int8_t i[32]; __m256i m; } V{.m = default_v.v}, M{.m = m.v};
    alignas(32) int8_t S[32];
    auto P = (const int8_t*) p;
    for (int i = 0; i < 32; ++i) S[i] = M.i[i] ? P[i] : V.i[i];
    return _mm256_load_si256((const __m256i *)S);
  }
}
#endif // VEC_WIDTH >= 256

#if VEC_WIDTH >= 512
  #error "Unreachable"
#endif // VEC_WIDTH >= 512
#endif // HAS_AVX512DQ

template <TLV_DECL_TAG(T)>
VECOPS_VFUNC Vec<T> load(T t, const TypeOf<T> * p, Mask<T> m, Vec<T> default_v) {
  VECOPS_ASSERT(is_aligned(T::Bytes, p), "Not aligned");
  return word::loadu(t, p, m, default_v);
}

template <TLV_DECL_TAG(T)>
VECOPS_VFUNC Vec<T> loadu(T t, const TypeOf<T> * p, nint_t n, Vec<T> default_v) {
  VECOPS_ASSERT(0 <= n && n <= size(t), "%zd !in 0..%zd", n, size(t));
  auto m = word::mwhilelt(t, 0, n);
  return word::loadu(t, p, m, default_v);
}

template <TLV_DECL_TAG(T)>
VECOPS_VFUNC Vec<T> load(T t, const TypeOf<T> * p, nint_t n, Vec<T> default_v) {
  VECOPS_ASSERT(is_aligned(T::Bytes, p), "Not aligned");
  return word::loadu(t, p, n, default_v);
}


/* ************************************************************************** */
//                             Consecutive Store                              //
/* ************************************************************************** */
template <TLV_DECL_TAG(T), TL_IF(T::Bytes == 8), TL_IF(is_any<TypeOf<T>, float64_t>)>
VECOPS_VFUNC void storeu(T t, TypeOf<T> * p, Vec<T> v) {
  _mm_store_sd(p, v.v);
}
template <TLV_DECL_TAG(T), TL_IF(T::Bytes == 16), TL_IF(is_any<TypeOf<T>, float64_t>)>
VECOPS_VFUNC void storeu(T t, TypeOf<T> * p, Vec<T> v) {
  _mm_storeu_pd(p, v.v);
}
template <TLV_DECL_TAG(T), TL_IF(T::Bytes == 4), TL_IF(is_any<TypeOf<T>, float32_t>)>
VECOPS_VFUNC void storeu(T t, TypeOf<T> * p, Vec<T> v) {
  _mm_store_ss(p, v.v);
}
template <TLV_DECL_TAG(T), TL_IF(T::Bytes == 8), TL_IF(is_any<TypeOf<T>, float32_t>)>
VECOPS_VFUNC void storeu(T t, TypeOf<T> * p, Vec<T> v) {
  Tag<float64_t, 1> t1;
  word::storeu(t1, (float64_t *) p, word::bitcast(t1, v));
}
template <TLV_DECL_TAG(T), TL_IF(T::Bytes == 16), TL_IF(is_any<TypeOf<T>, float32_t>)>
VECOPS_VFUNC void storeu(T t, TypeOf<T> * p, Vec<T> v) {
  _mm_storeu_ps(p, v.v);
}
template <TLV_DECL_TAG(T), TL_IF(T::Bytes == 16), TL_IF(is_int<TypeOf<T>> || is_small_float<TypeOf<T>>)>
VECOPS_VFUNC void storeu(T t, TypeOf<T> * p, Vec<T> v) {
  _mm_storeu_si128((__m128i *) p, v.v);
}
template <TLV_DECL_TAG(T), TL_IF(T::Bytes == 8), TL_IF(is_int<TypeOf<T>> || is_small_float<TypeOf<T>>)>
VECOPS_VFUNC void storeu(T t, TypeOf<T> * p, Vec<T> v) {
  _mm_storeu_si64(p, v.v);
}
template <TLV_DECL_TAG(T), TL_IF(T::Bytes == 4), TL_IF(is_int<TypeOf<T>> || is_small_float<TypeOf<T>>)>
VECOPS_VFUNC void storeu(T t, TypeOf<T> * p, Vec<T> v) {
  _mm_storeu_si32(p, v.v);
}
template <TLV_DECL_TAG(T), TL_IF(T::Bytes == 2), TL_IF(is_int<TypeOf<T>> || is_small_float<TypeOf<T>>)>
VECOPS_VFUNC void storeu(T t, TypeOf<T> * p, Vec<T> v) {
  _mm_storeu_si16(p, v.v);
}
template <TLV_DECL_TAG(T), TL_IF(T::Bytes == 1), TL_IF(is_int<TypeOf<T>> || is_small_float<TypeOf<T>>)>
VECOPS_VFUNC void storeu(T t, TypeOf<T> * p, Vec<T> v) {
  auto r = _mm_cvtsi128_si32(v.v);
  ((int8_t *) p)[0] = (int8_t)r;
}

template <TLV_DECL_TAG(T), TL_IF(T::Bytes < 16)>
VECOPS_VFUNC void store(T t, TypeOf<T> * p, Vec<T> v) {
  VECOPS_ASSERT(is_aligned(T::Bytes, p), "Not aligned");
  word::storeu(t, p, v);
}
template <TLV_DECL_TAG(T), TL_IF(T::Bytes == 16), TL_IF(is_any<TypeOf<T>, float32_t>)>
VECOPS_VFUNC void store(T t, TypeOf<T> * p, Vec<T> v) {
  VECOPS_ASSERT(is_aligned(T::Bytes, p), "Not aligned");
  _mm_store_ps(p, v.v);
}
template <TLV_DECL_TAG(T), TL_IF(T::Bytes == 16), TL_IF(is_any<TypeOf<T>, float64_t>)>
VECOPS_VFUNC void store(T t, TypeOf<T> * p, Vec<T> v) {
  VECOPS_ASSERT(is_aligned(T::Bytes, p), "Not aligned");
  _mm_store_pd(p, v.v);
}
template <TLV_DECL_TAG(T), TL_IF(T::Bytes == 16), TL_IF(is_int<TypeOf<T>> || is_small_float<TypeOf<T>>)>
VECOPS_VFUNC void store(T t, TypeOf<T> * p, Vec<T> v) {
  VECOPS_ASSERT(is_aligned(T::Bytes, p), "Not aligned");
  _mm_store_si128((__m128i *) p, v.v);
}

#if VEC_WIDTH >= 256
template <TLV_DECL_TAG(T), TL_IF(T::Bytes == 32), TL_IF(is_any<TypeOf<T>, float64_t>)>
VECOPS_VFUNC void storeu(T t, TypeOf<T> * p, Vec<T> v) {
  _mm256_storeu_pd(p, v.v);
}
template <TLV_DECL_TAG(T), TL_IF(T::Bytes == 32), TL_IF(is_any<TypeOf<T>, float32_t>)>
VECOPS_VFUNC void storeu(T t, TypeOf<T> * p, Vec<T> v) {
  _mm256_storeu_ps(p, v.v);
}
template <TLV_DECL_TAG(T), TL_IF(T::Bytes == 32), TL_IF(is_int<TypeOf<T>> || is_small_float<TypeOf<T>>)>
VECOPS_VFUNC void storeu(T t, TypeOf<T> * p, Vec<T> v) {
  _mm256_storeu_si256((__m256i *) p, v.v);
}

template <TLV_DECL_TAG(T), TL_IF(T::Bytes == 32), TL_IF(is_any<TypeOf<T>, float32_t>)>
VECOPS_VFUNC void store(T t, TypeOf<T> * p, Vec<T> v) {
  VECOPS_ASSERT(is_aligned(T::Bytes, p), "Not aligned");
  _mm256_store_ps(p, v.v);
}
template <TLV_DECL_TAG(T), TL_IF(T::Bytes == 32), TL_IF(is_any<TypeOf<T>, float64_t>)>
VECOPS_VFUNC void store(T t, TypeOf<T> * p, Vec<T> v) {
  VECOPS_ASSERT(is_aligned(T::Bytes, p), "Not aligned");
  _mm256_store_pd(p, v.v);
}
template <TLV_DECL_TAG(T), TL_IF(T::Bytes == 32), TL_IF(is_int<TypeOf<T>> || is_small_float<TypeOf<T>>)>
VECOPS_VFUNC void store(T t, TypeOf<T> * p, Vec<T> v) {
  VECOPS_ASSERT(is_aligned(T::Bytes, p), "Not aligned");
  _mm256_store_si256((__m256i *) p, v.v);
}
#endif // VEC_WIDTH >= 256

#if VEC_WIDTH >= 512
template <TLV_DECL_TAG(T), TL_IF(T::Bytes == 64), TL_IF(is_any<TypeOf<T>, float64_t>)>
VECOPS_VFUNC void storeu(T t, TypeOf<T> * p, Vec<T> v) {
  _mm512_storeu_pd(p, v.v);
}
template <TLV_DECL_TAG(T), TL_IF(T::Bytes == 64), TL_IF(is_any<TypeOf<T>, float32_t>)>
VECOPS_VFUNC void storeu(T t, TypeOf<T> * p, Vec<T> v) {
  _mm512_storeu_ps(p, v.v);
}
template <TLV_DECL_TAG(T), TL_IF(T::Bytes == 64), TL_IF(is_int<TypeOf<T>> || is_small_float<TypeOf<T>>)>
VECOPS_VFUNC void storeu(T t, TypeOf<T> * p, Vec<T> v) {
  _mm512_storeu_si512((__m512i *) p, v.v);
}

template <TLV_DECL_TAG(T), TL_IF(T::Bytes == 64), TL_IF(is_any<TypeOf<T>, float32_t>)>
VECOPS_VFUNC void store(T t, TypeOf<T> * p, Vec<T> v) {
  VECOPS_ASSERT(is_aligned(T::Bytes, p), "Not aligned");
  _mm512_store_ps(p, v.v);
}
template <TLV_DECL_TAG(T), TL_IF(T::Bytes == 64), TL_IF(is_any<TypeOf<T>, float64_t>)>
VECOPS_VFUNC void store(T t, TypeOf<T> * p, Vec<T> v) {
  VECOPS_ASSERT(is_aligned(T::Bytes, p), "Not aligned");
  _mm512_store_pd(p, v.v);
}
template <TLV_DECL_TAG(T), TL_IF(T::Bytes == 64), TL_IF(is_int<TypeOf<T>> || is_small_float<TypeOf<T>>)>
VECOPS_VFUNC void store(T t, TypeOf<T> * p, Vec<T> v) {
  VECOPS_ASSERT(is_aligned(T::Bytes, p), "Not aligned");
  _mm512_store_si512((__m512i *) p, v.v);
}
#endif// VEC_WIDTH >= 512

#ifdef HAS_AVX512DQ
template <TLV_DECL_TAG(T), TL_IF(T::Bytes <= 16), TL_IF(is_any<TypeOf<T>, float32_t>)>
VECOPS_VFUNC void storeu(T t, TypeOf<T> * p, Mask<T> m, Vec<T> v) {
  _mm_mask_storeu_ps(p, details::restrict_mask_range(t, m).v, v.v);
}
template <TLV_DECL_TAG(T), TL_IF(T::Bytes <= 16), TL_IF(is_any<TypeOf<T>, float64_t>)>
VECOPS_VFUNC void storeu(T t, TypeOf<T> * p, Mask<T> m, Vec<T> v) {
  _mm_mask_storeu_pd(p, details::restrict_mask_range(t, m).v, v.v);
}
template <TLV_DECL_TAG(T), TL_IF(T::Bytes <= 16), TL_IF(is_any<TypeOf<T>, int8_t, uint8_t>)>
VECOPS_VFUNC void storeu(T t, TypeOf<T> * p, Mask<T> m, Vec<T> v) {
  _mm_mask_storeu_epi8(p, details::restrict_mask_range(t, m).v, v.v);
}
template <TLV_DECL_TAG(T), TL_IF(T::Bytes <= 16), TL_IF(is_any<TypeOf<T>, int16_t, uint16_t, float16_t, bfloat16_t>)>
VECOPS_VFUNC void storeu(T t, TypeOf<T> * p, Mask<T> m, Vec<T> v) {
  _mm_mask_storeu_epi16(p, details::restrict_mask_range(t, m).v, v.v);
}
template <TLV_DECL_TAG(T), TL_IF(T::Bytes <= 16), TL_IF(is_any<TypeOf<T>, int32_t, uint32_t>)>
VECOPS_VFUNC void storeu(T t, TypeOf<T> * p, Mask<T> m, Vec<T> v) {
  _mm_mask_storeu_epi32(p, details::restrict_mask_range(t, m).v, v.v);
}
template <TLV_DECL_TAG(T), TL_IF(T::Bytes <= 16), TL_IF(is_any<TypeOf<T>, int64_t, uint64_t>)>
VECOPS_VFUNC void storeu(T t, TypeOf<T> * p, Mask<T> m, Vec<T> v) {
  _mm_mask_storeu_epi64(p, details::restrict_mask_range(t, m).v, v.v);
}

template <TLV_DECL_TAG(T), TL_IF(T::Bytes == 32), TL_IF(is_any<TypeOf<T>, float32_t>)>
VECOPS_VFUNC void storeu(T t, TypeOf<T> * p, Mask<T> m, Vec<T> v) {
  _mm256_mask_storeu_ps(p, m.v, v.v);
}
template <TLV_DECL_TAG(T), TL_IF(T::Bytes == 32), TL_IF(is_any<TypeOf<T>, float64_t>)>
VECOPS_VFUNC void storeu(T t, TypeOf<T> * p, Mask<T> m, Vec<T> v) {
  _mm256_mask_storeu_pd(p, m.v, v.v);
}
template <TLV_DECL_TAG(T), TL_IF(T::Bytes == 32), TL_IF(is_any<TypeOf<T>, int8_t, uint8_t>)>
VECOPS_VFUNC void storeu(T t, TypeOf<T> * p, Mask<T> m, Vec<T> v) {
  _mm256_mask_storeu_epi8(p, m.v, v.v);
}
template <TLV_DECL_TAG(T), TL_IF(T::Bytes == 32), TL_IF(is_any<TypeOf<T>, int16_t, uint16_t, float16_t, bfloat16_t>)>
VECOPS_VFUNC void storeu(T t, TypeOf<T> * p, Mask<T> m, Vec<T> v) {
  _mm256_mask_storeu_epi16(p, m.v, v.v);
}
template <TLV_DECL_TAG(T), TL_IF(T::Bytes == 32), TL_IF(is_any<TypeOf<T>, int32_t, uint32_t>)>
VECOPS_VFUNC void storeu(T t, TypeOf<T> * p, Mask<T> m, Vec<T> v) {
  _mm256_mask_storeu_epi32(p, m.v, v.v);
}
template <TLV_DECL_TAG(T), TL_IF(T::Bytes == 32), TL_IF(is_any<TypeOf<T>, int64_t, uint64_t>)>
VECOPS_VFUNC void storeu(T t, TypeOf<T> * p, Mask<T> m, Vec<T> v) {
  _mm256_mask_storeu_epi64(p, m.v, v.v);
}

template <TLV_DECL_TAG(T), TL_IF(T::Bytes == 64), TL_IF(is_any<TypeOf<T>, float32_t>)>
VECOPS_VFUNC void storeu(T t, TypeOf<T> * p, Mask<T> m, Vec<T> v) {

  _mm512_mask_storeu_ps(p, m.v, v.v);
}
template <TLV_DECL_TAG(T), TL_IF(T::Bytes == 64), TL_IF(is_any<TypeOf<T>, float64_t>)>
VECOPS_VFUNC void storeu(T t, TypeOf<T> * p, Mask<T> m, Vec<T> v) {
  _mm512_mask_storeu_pd(p, m.v, v.v);
}
template <TLV_DECL_TAG(T), TL_IF(T::Bytes == 64), TL_IF(is_any<TypeOf<T>, int8_t, uint8_t>)>
VECOPS_VFUNC void storeu(T t, TypeOf<T> * p, Mask<T> m, Vec<T> v) {
  _mm512_mask_storeu_epi8(p, m.v, v.v);
}
template <TLV_DECL_TAG(T), TL_IF(T::Bytes == 64), TL_IF(is_any<TypeOf<T>, int16_t, uint16_t, float16_t, bfloat16_t>)>
VECOPS_VFUNC void storeu(T t, TypeOf<T> * p, Mask<T> m, Vec<T> v) {
  _mm512_mask_storeu_epi16(p, m.v, v.v);
}
template <TLV_DECL_TAG(T), TL_IF(T::Bytes == 64), TL_IF(is_any<TypeOf<T>, int32_t, uint32_t>)>
VECOPS_VFUNC void storeu(T t, TypeOf<T> * p, Mask<T> m, Vec<T> v) {
  _mm512_mask_storeu_epi32(p, m.v, v.v);
}
template <TLV_DECL_TAG(T), TL_IF(T::Bytes == 64), TL_IF(is_any<TypeOf<T>, int64_t, uint64_t>)>
VECOPS_VFUNC void storeu(T t, TypeOf<T> * p, Mask<T> m, Vec<T> v) {
  _mm512_mask_storeu_epi64(p, m.v, v.v);
}

template <TLV_DECL_TAG(T), TL_IF(T::Bytes < 16)>
VECOPS_VFUNC void store(T t, TypeOf<T> * p, Mask<T> m, Vec<T> v) {
  VECOPS_ASSERT(is_aligned(T::Bytes, p), "Not aligned");
  word::storeu(t, p, m, v);
}

template <TLV_DECL_TAG(T), TL_IF(T::Bytes == 16), TL_IF(is_any<TypeOf<T>, float32_t>)>
VECOPS_VFUNC void store(T t, TypeOf<T> * p, Mask<T> m, Vec<T> v) {
  VECOPS_ASSERT(is_aligned(T::Bytes, p), "Not aligned");
  _mm_mask_store_ps(p, m.v, v.v);
}
template <TLV_DECL_TAG(T), TL_IF(T::Bytes == 16), TL_IF(is_any<TypeOf<T>, float64_t>)>
VECOPS_VFUNC void store(T t, TypeOf<T> * p, Mask<T> m, Vec<T> v) {
  VECOPS_ASSERT(is_aligned(T::Bytes, p), "Not aligned");
  _mm_mask_store_pd(p, m.v, v.v);
}
template <TLV_DECL_TAG(T), TL_IF(sizeof(TypeOf<T>) < 4)>
VECOPS_VFUNC void store(T t, TypeOf<T> * p, Mask<T> m, Vec<T> v) {
  VECOPS_ASSERT(is_aligned(T::Bytes, p), "Not aligned");
  word::storeu(t, p, m, v);
}
template <TLV_DECL_TAG(T), TL_IF(T::Bytes == 16), TL_IF(is_any<TypeOf<T>, int32_t, uint32_t>)>
VECOPS_VFUNC void store(T t, TypeOf<T> * p, Mask<T> m, Vec<T> v) {
  VECOPS_ASSERT(is_aligned(T::Bytes, p), "Not aligned");
  _mm_mask_store_epi32(p, m.v, v.v);
}
template <TLV_DECL_TAG(T), TL_IF(T::Bytes == 16), TL_IF(is_any<TypeOf<T>, int64_t, uint64_t>)>
VECOPS_VFUNC void store(T t, TypeOf<T> * p, Mask<T> m, Vec<T> v) {
  VECOPS_ASSERT(is_aligned(T::Bytes, p), "Not aligned");
  _mm_mask_store_epi64(p, m.v, v.v);
}

template <TLV_DECL_TAG(T), TL_IF(T::Bytes == 32), TL_IF(is_any<TypeOf<T>, float32_t>)>
VECOPS_VFUNC void store(T t, TypeOf<T> * p, Mask<T> m, Vec<T> v) {
  VECOPS_ASSERT(is_aligned(T::Bytes, p), "Not aligned");
  _mm256_mask_store_ps(p, m.v, v.v);
}
template <TLV_DECL_TAG(T), TL_IF(T::Bytes == 32), TL_IF(is_any<TypeOf<T>, float64_t>)>
VECOPS_VFUNC void store(T t, TypeOf<T> * p, Mask<T> m, Vec<T> v) {
  VECOPS_ASSERT(is_aligned(T::Bytes, p), "Not aligned");
  _mm256_mask_store_pd(p, m.v, v.v);
}
template <TLV_DECL_TAG(T), TL_IF(T::Bytes == 32), TL_IF(is_any<TypeOf<T>, int32_t, uint32_t>)>
VECOPS_VFUNC void store(T t, TypeOf<T> * p, Mask<T> m, Vec<T> v) {
  VECOPS_ASSERT(is_aligned(T::Bytes, p), "Not aligned");
  _mm256_mask_store_epi32(p, m.v, v.v);
}
template <TLV_DECL_TAG(T), TL_IF(T::Bytes == 32), TL_IF(is_any<TypeOf<T>, int64_t, uint64_t>)>
VECOPS_VFUNC void store(T t, TypeOf<T> * p, Mask<T> m, Vec<T> v) {
  VECOPS_ASSERT(is_aligned(T::Bytes, p), "Not aligned");
  _mm256_mask_store_epi64(p, m.v, v.v);
}

template <TLV_DECL_TAG(T), TL_IF(T::Bytes == 64), TL_IF(is_any<TypeOf<T>, float32_t>)>
VECOPS_VFUNC void store(T t, TypeOf<T> * p, Mask<T> m, Vec<T> v) {
  VECOPS_ASSERT(is_aligned(T::Bytes, p), "Not aligned");
  _mm512_mask_store_ps(p, m.v, v.v);
}
template <TLV_DECL_TAG(T), TL_IF(T::Bytes == 64), TL_IF(is_any<TypeOf<T>, float64_t>)>
VECOPS_VFUNC void store(T t, TypeOf<T> * p, Mask<T> m, Vec<T> v) {
  VECOPS_ASSERT(is_aligned(T::Bytes, p), "Not aligned");
  _mm512_mask_store_pd(p, m.v, v.v);
}
template <TLV_DECL_TAG(T), TL_IF(T::Bytes == 64), TL_IF(is_any<TypeOf<T>, int32_t, uint32_t>)>
VECOPS_VFUNC void store(T t, TypeOf<T> * p, Mask<T> m, Vec<T> v) {
  VECOPS_ASSERT(is_aligned(T::Bytes, p), "Not aligned");
  _mm512_mask_store_epi32(p, m.v, v.v);
}
template <TLV_DECL_TAG(T), TL_IF(T::Bytes == 64), TL_IF(is_any<TypeOf<T>, int64_t, uint64_t>)>
VECOPS_VFUNC void store(T t, TypeOf<T> * p, Mask<T> m, Vec<T> v) {
  VECOPS_ASSERT(is_aligned(T::Bytes, p), "Not aligned");
  _mm512_mask_store_epi64(p, m.v, v.v);
}
#else // HAS_AVX512DQ
template <TLV_DECL_TAG(T), TL_IF(T::Bytes <= 16), TL_IF(is_any<TypeOf<T>, float32_t>)>
VECOPS_VFUNC void storeu(T t, TypeOf<T> * p, Mask<T> m, Vec<T> v) {
  _mm_maskstore_ps(p, details::restrict_mask_range(t, m).v, v.v);
}
template <TLV_DECL_TAG(T), TL_IF(T::Bytes <= 16), TL_IF(is_any<TypeOf<T>, float64_t>)>
VECOPS_VFUNC void storeu(T t, TypeOf<T> * p, Mask<T> m, Vec<T> v) {
  _mm_maskstore_pd(p, details::restrict_mask_range(t, m).v, v.v);
}
template <TLV_DECL_TAG(T), TL_IF(T::Bytes <= 16), TL_IF(is_any<TypeOf<T>, int32_t, uint32_t>)>
VECOPS_VFUNC void storeu(T t, TypeOf<T> * p, Mask<T> m, Vec<T> v) {
  #ifdef HAS_AVX2
  _mm_maskstore_epi32((int32_t *) p, details::restrict_mask_range(t, m).v, v.v);
  #else
  Rebind<float32_t, T> t1;
  word::storeu(t1, (float32_t *) p, m.v, word::bitcast(t1, v));
  #endif
}
template <TLV_DECL_TAG(T), TL_IF(T::Bytes <= 16), TL_IF(is_any<TypeOf<T>, int64_t, uint64_t>)>
VECOPS_VFUNC void storeu(T t, TypeOf<T> * p, Mask<T> m, Vec<T> v) {
  #ifdef HAS_AVX2
  _mm_maskstore_epi64((long long *) p, details::restrict_mask_range(t, m).v, v.v);
  #else
  Rebind<float64_t, T> t1;
  word::storeu(t1, (float64_t *) p, m.v, word::bitcast(t1, v));
  #endif
}
template <TLV_DECL_TAG(T), TL_IF(T::Bytes <= 16), TL_IF(is_any<TypeOf<T>, int16_t, uint16_t, float16_t, bfloat16_t, int8_t, uint8_t>)>
VECOPS_VFUNC void storeu(T t, TypeOf<T> * p, Mask<T> m, Vec<T> v) {
  // TODO non-temporal memory hint?
  _mm_maskmoveu_si128(v.v, details::restrict_mask_range(t, m).v, (char *) p);
}

#if VEC_WIDTH >= 256
template <TLV_DECL_TAG(T), TL_IF(T::Bytes == 32), TL_IF(is_any<TypeOf<T>, float32_t>)>
VECOPS_VFUNC void storeu(T t, TypeOf<T> * p, Mask<T> m, Vec<T> v) {
  _mm256_maskstore_ps(p, m.v, v.v);
}
template <TLV_DECL_TAG(T), TL_IF(T::Bytes == 32), TL_IF(is_any<TypeOf<T>, float64_t>)>
VECOPS_VFUNC void storeu(T t, TypeOf<T> * p, Mask<T> m, Vec<T> v) {
  _mm256_maskstore_pd(p, m.v, v.v);
}
template <TLV_DECL_TAG(T), TL_IF(T::Bytes == 32), TL_IF(is_any<TypeOf<T>, int32_t, uint32_t>)>
VECOPS_VFUNC void storeu(T t, TypeOf<T> * p, Mask<T> m, Vec<T> v) {
  _mm256_maskstore_epi32((int32_t *) p, m.v, v.v);
}
template <TLV_DECL_TAG(T), TL_IF(T::Bytes == 32), TL_IF(is_any<TypeOf<T>, int64_t, uint64_t>)>
VECOPS_VFUNC void storeu(T t, TypeOf<T> * p, Mask<T> m, Vec<T> v) {
  _mm256_maskstore_epi64((long long *) p, m.v, v.v);
}
template <TLV_DECL_TAG(T), TL_IF(T::Bytes == 32), TL_IF(is_any<TypeOf<T>, int16_t, uint16_t, float16_t, bfloat16_t, int8_t, uint8_t>)>
VECOPS_VFUNC void storeu(T t, TypeOf<T> * p, Mask<T> m, Vec<T> v) {
  ViewAs<Index<TypeOf<T>>, T> tm;
  auto lo = word::lower(t, v); auto lom = word::lower(tm, Vec<decltype(tm)>{m.v});
  auto hi = word::upper(t, v); auto him = word::upper(tm, Vec<decltype(tm)>{m.v});
  // TODO non-temporal memory hint?
  _mm_maskmoveu_si128(lo.v, lom.v, (char *) p);
  _mm_maskmoveu_si128(hi.v, him.v, (char *) p + 16);
}
#endif // VEC_WIDTH >= 256

#if VEC_WIDTH >= 512
  #error "Unreachable"
#endif // VEC_WIDTH >= 512

template <TLV_DECL_TAG(T)>
VECOPS_VFUNC void store(T t, TypeOf<T> * p, Mask<T> m, Vec<T> v) {
  VECOPS_ASSERT(is_aligned(T::Bytes, p), "Not aligned");
  word::storeu(t, p, m, v);
}
#endif // HAS_AVX512DQ

template <TLV_DECL_TAG(T)>
VECOPS_VFUNC void storeu(T t, TypeOf<T> * p, nint_t n, Vec<T> v) {
  VECOPS_ASSERT(0 <= n && n <= size(t), "%zd !in 0..%zd", n, size(t));
  auto m = word::mwhilelt(t, 0, n);
  word::storeu(t, p, m, v);
}

template <TLV_DECL_TAG(T)>
VECOPS_VFUNC void store(T t, TypeOf<T> * p, nint_t n, Vec<T> v) {
  VECOPS_ASSERT(0 <= n && n <= size(t), "%zd !in 0..%zd", n, size(t));
  auto m = word::mwhilelt(t, 0, n);
  word::store(t, p, m, v);
}


/* ************************************************************************** */
//                                  Gather                                    //
/* ************************************************************************** */
#ifdef HAS_AVX2
template <TLV_DECL_TAG(T), TL_IF(T::Bytes <= 16), TL_IF(is_any<TypeOf<T>, float32_t>)>
VECOPS_VFUNC Vec<T> gather(T t, const TypeOf<T>* p, Vec<Rebind<GatherScatterIndex<TypeOf<T>>, T>> i) {
  return _mm_i32gather_ps(p, i.v, (int)sizeof(TypeOf<T>));
}

template <TLV_DECL_TAG(T), TL_IF(T::Bytes <= 16), TL_IF(is_any<TypeOf<T>, float64_t>)>
VECOPS_VFUNC Vec<T> gather(T t, const TypeOf<T>* p, Vec<Rebind<GatherScatterIndex<TypeOf<T>>, T>> i) {
  return _mm_i64gather_pd(p, i.v, (int)sizeof(TypeOf<T>));
}

template <TLV_DECL_TAG(T), TL_IF(T::Bytes <= 16), TL_IF(is_any<TypeOf<T>, int32_t, uint32_t>)>
VECOPS_VFUNC Vec<T> gather(T t, const TypeOf<T>* p, Vec<Rebind<GatherScatterIndex<TypeOf<T>>, T>> i) {
  return _mm_i32gather_epi32(reinterpret_cast<const int*>(p), i.v, (int)sizeof(TypeOf<T>));
}

template <TLV_DECL_TAG(T), TL_IF(T::Bytes <= 16), TL_IF(is_any<TypeOf<T>, int64_t, uint64_t>)>
VECOPS_VFUNC Vec<T> gather(T t, const TypeOf<T>* p, Vec<Rebind<GatherScatterIndex<TypeOf<T>>, T>> i) {
  return _mm_i64gather_epi64(reinterpret_cast<const long long*>(p), i.v, (int)sizeof(TypeOf<T>));
}

#if VEC_WIDTH >= 256
template <TLV_DECL_TAG(T), TL_IF(T::Bytes == 32), TL_IF(is_any<TypeOf<T>, float32_t>)>
VECOPS_VFUNC Vec<T> gather(T t, const TypeOf<T>* p, Vec<Rebind<GatherScatterIndex<TypeOf<T>>, T>> i) {
  return _mm256_i32gather_ps(p, i.v, (int)sizeof(TypeOf<T>));
}

template <TLV_DECL_TAG(T), TL_IF(T::Bytes == 32), TL_IF(is_any<TypeOf<T>, float64_t>)>
VECOPS_VFUNC Vec<T> gather(T t, const TypeOf<T>* p, Vec<Rebind<GatherScatterIndex<TypeOf<T>>, T>> i) {
  return _mm256_i64gather_pd(p, i.v, (int)sizeof(TypeOf<T>));
}

template <TLV_DECL_TAG(T), TL_IF(T::Bytes == 32), TL_IF(is_any<TypeOf<T>, int32_t, uint32_t>)>
VECOPS_VFUNC Vec<T> gather(T t, const TypeOf<T>* p, Vec<Rebind<GatherScatterIndex<TypeOf<T>>, T>> i) {
  return _mm256_i32gather_epi32(reinterpret_cast<const int*>(p), i.v, (int)sizeof(TypeOf<T>));
}

template <TLV_DECL_TAG(T), TL_IF(T::Bytes == 32), TL_IF(is_any<TypeOf<T>, int64_t, uint64_t>)>
VECOPS_VFUNC Vec<T> gather(T t, const TypeOf<T>* p, Vec<Rebind<GatherScatterIndex<TypeOf<T>>, T>> i) {
  return _mm256_i64gather_epi64(reinterpret_cast<const long long*>(p), i.v, (int)sizeof(TypeOf<T>));
}
#endif // VEC_WIDTH >= 256

#if VEC_WIDTH >= 512
template <TLV_DECL_TAG(T), TL_IF(T::Bytes == 64), TL_IF(is_any<TypeOf<T>, float32_t>)>
VECOPS_VFUNC Vec<T> gather(T t, const TypeOf<T>* p, Vec<Rebind<GatherScatterIndex<TypeOf<T>>, T>> i) {
  return _mm512_i32gather_ps(i.v, p, (int)sizeof(TypeOf<T>));
}

template <TLV_DECL_TAG(T), TL_IF(T::Bytes == 64), TL_IF(is_any<TypeOf<T>, float64_t>)>
VECOPS_VFUNC Vec<T> gather(T t, const TypeOf<T>* p, Vec<Rebind<GatherScatterIndex<TypeOf<T>>, T>> i) {
  return _mm512_i64gather_pd(i.v, p, (int)sizeof(TypeOf<T>));
}

template <TLV_DECL_TAG(T), TL_IF(T::Bytes == 64), TL_IF(is_any<TypeOf<T>, int32_t, uint32_t>)>
VECOPS_VFUNC Vec<T> gather(T t, const TypeOf<T>* p, Vec<Rebind<GatherScatterIndex<TypeOf<T>>, T>> i) {
  return _mm512_i32gather_epi32(i.v, p, (int)sizeof(TypeOf<T>));
}

template <TLV_DECL_TAG(T), TL_IF(T::Bytes == 64), TL_IF(is_any<TypeOf<T>, int64_t, uint64_t>)>
VECOPS_VFUNC Vec<T> gather(T t, const TypeOf<T>* p, Vec<Rebind<GatherScatterIndex<TypeOf<T>>, T>> i) {
  return _mm512_i64gather_epi64(i.v, p, (int)sizeof(TypeOf<T>));
}
#endif // VEC_WIDTH >= 512

#ifdef HAS_AVX512DQ
template <TLV_DECL_TAG(T), TL_IF(T::Bytes <= 16), TL_IF(is_any<TypeOf<T>, float32_t>)>
VECOPS_VFUNC Vec<T> gather(T t, const TypeOf<T>* p, Vec<Rebind<GatherScatterIndex<TypeOf<T>>, T>> i, Mask<T> m, Vec<T> default_v) {
  return _mm_mmask_i32gather_ps(default_v.v, m.v, i.v, p, (int)sizeof(TypeOf<T>));
  }
template <TLV_DECL_TAG(T), TL_IF(T::Bytes <= 16), TL_IF(is_any<TypeOf<T>, float64_t>)>
VECOPS_VFUNC Vec<T> gather(T t, const TypeOf<T>* p, Vec<Rebind<GatherScatterIndex<TypeOf<T>>, T>> i, Mask<T> m, Vec<T> default_v) {
  return _mm_mmask_i64gather_pd(default_v.v, m.v, i.v, p, (int)sizeof(TypeOf<T>));
}
template <TLV_DECL_TAG(T), TL_IF(T::Bytes <= 16), TL_IF(is_any<TypeOf<T>, int32_t, uint32_t>)>
VECOPS_VFUNC Vec<T> gather(T t, const TypeOf<T>* p, Vec<Rebind<GatherScatterIndex<TypeOf<T>>, T>> i, Mask<T> m, Vec<T> default_v) {
  return _mm_mmask_i32gather_epi32(default_v.v, m.v, i.v, p, (int)sizeof(TypeOf<T>));
}
template <TLV_DECL_TAG(T), TL_IF(T::Bytes <= 16), TL_IF(is_any<TypeOf<T>, int64_t, uint64_t>)>
VECOPS_VFUNC Vec<T> gather(T t, const TypeOf<T>* p, Vec<Rebind<GatherScatterIndex<TypeOf<T>>, T>> i, Mask<T> m, Vec<T> default_v) {
  return _mm_mmask_i64gather_epi64(default_v.v, m.v, i.v, p, (int)sizeof(TypeOf<T>));
}

template <TLV_DECL_TAG(T), TL_IF(T::Bytes == 32), TL_IF(is_any<TypeOf<T>, float32_t>)>
VECOPS_VFUNC Vec<T> gather(T t, const TypeOf<T>* p, Vec<Rebind<GatherScatterIndex<TypeOf<T>>, T>> i, Mask<T> m, Vec<T> default_v) {
  return _mm256_mmask_i32gather_ps(default_v.v, m.v, i.v, p, (int)sizeof(TypeOf<T>));
  }
template <TLV_DECL_TAG(T), TL_IF(T::Bytes == 32), TL_IF(is_any<TypeOf<T>, float64_t>)>
VECOPS_VFUNC Vec<T> gather(T t, const TypeOf<T>* p, Vec<Rebind<GatherScatterIndex<TypeOf<T>>, T>> i, Mask<T> m, Vec<T> default_v) {
  return _mm256_mmask_i64gather_pd(default_v.v, m.v, i.v, p, (int)sizeof(TypeOf<T>));
}
template <TLV_DECL_TAG(T), TL_IF(T::Bytes == 32), TL_IF(is_any<TypeOf<T>, int32_t, uint32_t>)>
VECOPS_VFUNC Vec<T> gather(T t, const TypeOf<T>* p, Vec<Rebind<GatherScatterIndex<TypeOf<T>>, T>> i, Mask<T> m, Vec<T> default_v) {
  return _mm256_mmask_i32gather_epi32(default_v.v, m.v, i.v, p, (int)sizeof(TypeOf<T>));
}
template <TLV_DECL_TAG(T), TL_IF(T::Bytes == 32), TL_IF(is_any<TypeOf<T>, int64_t, uint64_t>)>
VECOPS_VFUNC Vec<T> gather(T t, const TypeOf<T>* p, Vec<Rebind<GatherScatterIndex<TypeOf<T>>, T>> i, Mask<T> m, Vec<T> default_v) {
  return _mm256_mmask_i64gather_epi64(default_v.v, m.v, i.v, p, (int)sizeof(TypeOf<T>));
}

template <TLV_DECL_TAG(T), TL_IF(T::Bytes == 64), TL_IF(is_any<TypeOf<T>, float32_t>)>
VECOPS_VFUNC Vec<T> gather(T t, const TypeOf<T>* p, Vec<Rebind<GatherScatterIndex<TypeOf<T>>, T>> i, Mask<T> m, Vec<T> default_v) {
  return _mm512_mask_i32gather_ps(default_v.v, m.v, i.v, p, (int)sizeof(TypeOf<T>));
  }
template <TLV_DECL_TAG(T), TL_IF(T::Bytes == 64), TL_IF(is_any<TypeOf<T>, float64_t>)>
VECOPS_VFUNC Vec<T> gather(T t, const TypeOf<T>* p, Vec<Rebind<GatherScatterIndex<TypeOf<T>>, T>> i, Mask<T> m, Vec<T> default_v) {
  return _mm512_mask_i64gather_pd(default_v.v, m.v, i.v, p, (int)sizeof(TypeOf<T>));
}
template <TLV_DECL_TAG(T), TL_IF(T::Bytes == 64), TL_IF(is_any<TypeOf<T>, int32_t, uint32_t>)>
VECOPS_VFUNC Vec<T> gather(T t, const TypeOf<T>* p, Vec<Rebind<GatherScatterIndex<TypeOf<T>>, T>> i, Mask<T> m, Vec<T> default_v) {
  return _mm512_mask_i32gather_epi32(default_v.v, m.v, i.v, p, (int)sizeof(TypeOf<T>));
}
template <TLV_DECL_TAG(T), TL_IF(T::Bytes == 64), TL_IF(is_any<TypeOf<T>, int64_t, uint64_t>)>
VECOPS_VFUNC Vec<T> gather(T t, const TypeOf<T>* p, Vec<Rebind<GatherScatterIndex<TypeOf<T>>, T>> i, Mask<T> m, Vec<T> default_v) {
  return _mm512_mask_i64gather_epi64(default_v.v, m.v, i.v, p, (int)sizeof(TypeOf<T>));
}
#else // HAS_AVX512DQ
template <TLV_DECL_TAG(T), TL_IF(T::Bytes <= 16), TL_IF(is_any<TypeOf<T>, float32_t>)>
VECOPS_VFUNC Vec<T> gather(T t, const TypeOf<T>* p, Vec<Rebind<GatherScatterIndex<TypeOf<T>>, T>> i, Mask<T> m, Vec<T> default_v) {
  return _mm_mask_i32gather_ps(default_v.v, p, i.v, _mm_castsi128_ps(m.v), (int)sizeof(TypeOf<T>));
  }
template <TLV_DECL_TAG(T), TL_IF(T::Bytes <= 16), TL_IF(is_any<TypeOf<T>, float64_t>)>
VECOPS_VFUNC Vec<T> gather(T t, const TypeOf<T>* p, Vec<Rebind<GatherScatterIndex<TypeOf<T>>, T>> i, Mask<T> m, Vec<T> default_v) {
  return _mm_mask_i64gather_pd(default_v.v, p, i.v, _mm_castsi128_pd(m.v), (int)sizeof(TypeOf<T>));
}
template <TLV_DECL_TAG(T), TL_IF(T::Bytes <= 16), TL_IF(is_any<TypeOf<T>, int32_t, uint32_t>)>
VECOPS_VFUNC Vec<T> gather(T t, const TypeOf<T>* p, Vec<Rebind<GatherScatterIndex<TypeOf<T>>, T>> i, Mask<T> m, Vec<T> default_v) {
  return _mm_mask_i32gather_epi32(default_v.v, reinterpret_cast<const int*>(p), i.v, m.v, (int)sizeof(TypeOf<T>));
}
template <TLV_DECL_TAG(T), TL_IF(T::Bytes <= 16), TL_IF(is_any<TypeOf<T>, int64_t, uint64_t>)>
VECOPS_VFUNC Vec<T> gather(T t, const TypeOf<T>* p, Vec<Rebind<GatherScatterIndex<TypeOf<T>>, T>> i, Mask<T> m, Vec<T> default_v) {
  return _mm_mask_i64gather_epi64(default_v.v, reinterpret_cast<const long long*>(p), i.v, m.v, (int)sizeof(TypeOf<T>));
}

#if VEC_WIDTH >= 256
template <TLV_DECL_TAG(T), TL_IF(T::Bytes == 32), TL_IF(is_any<TypeOf<T>, float32_t>)>
VECOPS_VFUNC Vec<T> gather(T t, const TypeOf<T>* p, Vec<Rebind<GatherScatterIndex<TypeOf<T>>, T>> i, Mask<T> m, Vec<T> default_v) {
  return _mm256_mask_i32gather_ps(default_v.v, p, i.v, _mm256_castsi256_ps(m.v), (int)sizeof(TypeOf<T>));
  }
template <TLV_DECL_TAG(T), TL_IF(T::Bytes == 32), TL_IF(is_any<TypeOf<T>, float64_t>)>
VECOPS_VFUNC Vec<T> gather(T t, const TypeOf<T>* p, Vec<Rebind<GatherScatterIndex<TypeOf<T>>, T>> i, Mask<T> m, Vec<T> default_v) {
  return _mm256_mask_i64gather_pd(default_v.v, p, i.v, _mm256_castsi256_pd(m.v), (int)sizeof(TypeOf<T>));
}
template <TLV_DECL_TAG(T), TL_IF(T::Bytes == 32), TL_IF(is_any<TypeOf<T>, int32_t, uint32_t>)>
VECOPS_VFUNC Vec<T> gather(T t, const TypeOf<T>* p, Vec<Rebind<GatherScatterIndex<TypeOf<T>>, T>> i, Mask<T> m, Vec<T> default_v) {
  return _mm256_mask_i32gather_epi32(default_v.v, reinterpret_cast<const int*>(p), i.v, m.v, (int)sizeof(TypeOf<T>));
}
template <TLV_DECL_TAG(T), TL_IF(T::Bytes == 32), TL_IF(is_any<TypeOf<T>, int64_t, uint64_t>)>
VECOPS_VFUNC Vec<T> gather(T t, const TypeOf<T>* p, Vec<Rebind<GatherScatterIndex<TypeOf<T>>, T>> i, Mask<T> m, Vec<T> default_v) {
  return _mm256_mask_i64gather_epi64(default_v.v, reinterpret_cast<const long long*>(p), i.v, m.v, (int)sizeof(TypeOf<T>));
}
#endif // VEC_WIDTH >= 256

#if VEC_WIDTH >= 512
  #error "Unreachable"
#endif // VEC_WIDTH >= 512
#endif // HAS_AVX512DQ

// ── Gather 2B types (int16/uint16/float16/bfloat16) ─────────────────────
// Gather int32 at scale=2 (element address); element is in low 16 bits.

template <TLV_DECL_TAG(T), TL_IF(is_any<TypeOf<T>, int16_t, uint16_t, float16_t, bfloat16_t>)>
VECOPS_VFUNC Vec<T> gather(T t, const TypeOf<T>* p, Vec<Rebind<GatherScatterIndex<TypeOf<T>>, T>> i) {
  Rebind<GatherScatterIndex<TypeOf<T>>, T> t_i;
  if constexpr (num_words(t_i) == 1) {
    auto g_v = details::subint2_gather_i32(t_i, p, i);
    Twice<T> t_2;
    return word::even(t_2, word::bitcast(t_2, g_v));
  } else {
    Half<T> t_h;
    auto lo = word::gather(t_h, p, word::lower(t_i, i));
    auto hi = word::gather(t_h, p, word::upper(t_i, i));
    return word::concat(t, lo, hi);
  }
}

template <TLV_DECL_TAG(T), TL_IF(is_any<TypeOf<T>, int16_t, uint16_t, float16_t, bfloat16_t>)>
VECOPS_VFUNC Vec<T> gather(T t, const TypeOf<T>* p, Vec<Rebind<GatherScatterIndex<TypeOf<T>>, T>> i, Mask<T> m, Vec<T> default_v) {
  Rebind<GatherScatterIndex<TypeOf<T>>, T> t_i;
  if constexpr (num_words(t_i) == 1) {
    auto g = word::gather(t, p, i);
    return word::blend(default_v, m, g);
  } else {
    Half<T> t_h;
    auto lo = word::gather(t_h, p, word::lower(t_i, i));
    auto hi = word::gather(t_h, p, word::upper(t_i, i));
    auto def_lo = word::lower(t, default_v);
    auto def_hi = word::upper(t, default_v);
    auto m_lo = word::lower(t, m);
    auto m_hi = word::upper(t, m);
    return word::concat(t,
        word::blend(def_lo, m_lo, lo),
        word::blend(def_hi, m_hi, hi));
  }
}

// ── Gather 1B types (int8/uint8) ────────────────────────────────────────
// Gather int32 at scale=1 (element address); element is in low 8 bits.

template <TLV_DECL_TAG(T), TL_IF(is_any<TypeOf<T>, int8_t, uint8_t>)>
VECOPS_VFUNC Vec<T> gather(T t, const TypeOf<T>* p, Vec<Rebind<GatherScatterIndex<TypeOf<T>>, T>> i) {
  Rebind<GatherScatterIndex<TypeOf<T>>, T> t_i;
  if constexpr (num_words(t_i) == 1) {
    ViewAs<int16_t, decltype(t_i)> t_i16;
    auto g_v = details::subint1_gather_i32(t_i, p, i);
    Twice<T> t_2;
    return word::even(t_2, word::bitcast(t_2, word::even(t_i16, word::bitcast(t_i16, g_v))));
  } else {
    Half<T> t_h;
    auto lo = word::gather(t_h, p, word::lower(t_i, i));
    auto hi = word::gather(t_h, p, word::upper(t_i, i));
    return word::concat(t, lo, hi);
  }
}

template <TLV_DECL_TAG(T), TL_IF(is_any<TypeOf<T>, int8_t, uint8_t>)>
VECOPS_VFUNC Vec<T> gather(T t, const TypeOf<T>* p, Vec<Rebind<GatherScatterIndex<TypeOf<T>>, T>> i, Mask<T> m, Vec<T> default_v) {
  Rebind<GatherScatterIndex<TypeOf<T>>, T> t_i;
  if constexpr (num_words(t_i) == 1) {
    auto g = word::gather(t, p, i);
    return word::blend(default_v, m, g);
  } else {
    Half<T> t_h;
    auto lo = word::gather(t_h, p, word::lower(t_i, i));
    auto hi = word::gather(t_h, p, word::upper(t_i, i));
    auto def_lo = word::lower(t, default_v);
    auto def_hi = word::upper(t, default_v);
    auto m_lo = word::lower(t, m);
    auto m_hi = word::upper(t, m);
    return word::concat(t,
        word::blend(def_lo, m_lo, lo),
        word::blend(def_hi, m_hi, hi));
  }
}

#else
template <TLV_DECL_TAG(T)>
static VECOPS_VFUNC Vec<T> _gather_scalar(T t, const TypeOf<T>* p, Vec<Rebind<GatherScatterIndex<TypeOf<T>>, T>> i) {
  alignas(T::Bytes) TypeOf<T> data[size(t)];
  VECOPS_UNROLL for (int j = 0; j < size(t); ++j) {
    data[j] = p[nint_t(word::get(i, j))];
  }
  return word::load(t, data);
}
template <TLV_DECL_TAG(T)>
static VECOPS_VFUNC Vec<T> _mask_gather_scalar(T t, const TypeOf<T>* p, Vec<Rebind<GatherScatterIndex<TypeOf<T>>, T>> i, Mask<T> m, Vec<T> default_v) {
  alignas(T::Bytes) TypeOf<T> data[size(t)];
  VECOPS_UNROLL for (int j = 0; j < size(t); ++j) {
    data[j] = word::get(t, m, j) ? p[nint_t(word::get(i, j))] : word::get(default_v, j);
  }
  return word::load(t, data);
}

template <TLV_DECL_TAG(T)>
VECOPS_VFUNC Vec<T> gather(T t, const TypeOf<T>* p, Vec<Rebind<GatherScatterIndex<TypeOf<T>>, T>> i) {
  Rebind<GatherScatterIndex<TypeOf<T>>, T> t_i;
  if constexpr (num_words(t_i) == 1) {
    return word::_gather_scalar(t, p, i);
  } else {
    Half<T> t_h;
    auto lo = word::gather(t_h, p, word::lower(t_i, i));
    auto hi = word::gather(t_h, p, word::upper(t_i, i));
    return word::concat(t, lo, hi);
  }
}
template <TLV_DECL_TAG(T)>
VECOPS_VFUNC Vec<T> gather(T t, const TypeOf<T>* p, Vec<Rebind<GatherScatterIndex<TypeOf<T>>, T>> i, Mask<T> m, Vec<T> default_v) {
  Rebind<GatherScatterIndex<TypeOf<T>>, T> t_i;
  if constexpr (num_words(t_i) == 1) {
    return word::_mask_gather_scalar(t, p, i, m, default_v);
  } else {
    Half<T> t_h;
    auto i_lo = word::lower(t_i, i);
    auto i_hi = word::upper(t_i, i);
    if constexpr (num_words(t) > 1) {
      auto lo = word::gather(t_h, p, i_lo, word::lower(t, m), word::lower(t, default_v));
      auto hi = word::gather(t_h, p, i_hi, word::upper(t, m), word::upper(t, default_v));
      return word::concat(t, lo, hi);
    } else {
      auto g = word::gather(t, p, i);
      return word::blend(default_v, m, g);
    }
  }
}
#endif // HAS_AVX2

template <TLV_DECL_TAG(T)>
VECOPS_VFUNC Vec<T> gather(T t, const TypeOf<T>* p, Vec<Rebind<GatherScatterIndex<TypeOf<T>>, T>> i, nint_t n, Vec<T> default_v) {
  VECOPS_ASSERT(0 <= n && n <= size(t), "%zd !in 0..%zd", n, size(t));
  auto m = word::mwhilelt(t, 0, n);
  return word::gather(t, p, i, m, default_v);
}


/* ************************************************************************** */
//                                  Scatter                                   //
/* ************************************************************************** */
#ifdef HAS_AVX512F
template <TLV_DECL_TAG(T), TL_IF(T::Bytes <= 16), TL_IF(is_any<TypeOf<T>, float32_t>)>
VECOPS_VFUNC void scatter(T t, TypeOf<T> * p, Vec<Rebind<GatherScatterIndex<TypeOf<T>>, T>> i, Vec<T> v) {
  _mm_i32scatter_ps(p, i.v, v.v, (int)sizeof(TypeOf<T>));
}
template <TLV_DECL_TAG(T), TL_IF(T::Bytes <= 16), TL_IF(is_any<TypeOf<T>, float64_t>)>
VECOPS_VFUNC void scatter(T t, TypeOf<T> * p, Vec<Rebind<GatherScatterIndex<TypeOf<T>>, T>> i, Vec<T> v) {
  _mm_i64scatter_pd(p, i.v, v.v, (int)sizeof(TypeOf<T>));
}
template <TLV_DECL_TAG(T), TL_IF(T::Bytes <= 16), TL_IF(is_any<TypeOf<T>, int32_t, uint32_t>)>
VECOPS_VFUNC void scatter(T t, TypeOf<T> * p, Vec<Rebind<GatherScatterIndex<TypeOf<T>>, T>> i, Vec<T> v) {
  _mm_i32scatter_epi32(p, i.v, v.v, (int)sizeof(TypeOf<T>));
}
template <TLV_DECL_TAG(T), TL_IF(T::Bytes <= 16), TL_IF(is_any<TypeOf<T>, int64_t, uint64_t>)>
VECOPS_VFUNC void scatter(T t, TypeOf<T> * p, Vec<Rebind<GatherScatterIndex<TypeOf<T>>, T>> i, Vec<T> v) {
  _mm_i64scatter_epi64(p, i.v, v.v, (int)sizeof(TypeOf<T>));
}

#if VEC_WIDTH >= 256
template <TLV_DECL_TAG(T), TL_IF(T::Bytes == 32), TL_IF(is_any<TypeOf<T>, float32_t>)>
VECOPS_VFUNC void scatter(T t, TypeOf<T> * p, Vec<Rebind<GatherScatterIndex<TypeOf<T>>, T>> i, Vec<T> v) {
  _mm256_i32scatter_ps(p, i.v, v.v, (int)sizeof(TypeOf<T>));
}
template <TLV_DECL_TAG(T), TL_IF(T::Bytes == 32), TL_IF(is_any<TypeOf<T>, float64_t>)>
VECOPS_VFUNC void scatter(T t, TypeOf<T> * p, Vec<Rebind<GatherScatterIndex<TypeOf<T>>, T>> i, Vec<T> v) {
  _mm256_i64scatter_pd(p, i.v, v.v, (int)sizeof(TypeOf<T>));
}
template <TLV_DECL_TAG(T), TL_IF(T::Bytes == 32), TL_IF(is_any<TypeOf<T>, int32_t, uint32_t>)>
VECOPS_VFUNC void scatter(T t, TypeOf<T> * p, Vec<Rebind<GatherScatterIndex<TypeOf<T>>, T>> i, Vec<T> v) {
  _mm256_i32scatter_epi32(p, i.v, v.v, (int)sizeof(TypeOf<T>));
}
template <TLV_DECL_TAG(T), TL_IF(T::Bytes == 32), TL_IF(is_any<TypeOf<T>, int64_t, uint64_t>)>
VECOPS_VFUNC void scatter(T t, TypeOf<T> * p, Vec<Rebind<GatherScatterIndex<TypeOf<T>>, T>> i, Vec<T> v) {
  _mm256_i64scatter_epi64(p, i.v, v.v, (int)sizeof(TypeOf<T>));
}
#endif

#if VEC_WIDTH >= 512
template <TLV_DECL_TAG(T), TL_IF(T::Bytes == 64), TL_IF(is_any<TypeOf<T>, float32_t>)>
VECOPS_VFUNC void scatter(T t, TypeOf<T> * p, Vec<Rebind<GatherScatterIndex<TypeOf<T>>, T>> i, Vec<T> v) {
  _mm512_i32scatter_ps(p, i.v, v.v, (int)sizeof(TypeOf<T>));
}
template <TLV_DECL_TAG(T), TL_IF(T::Bytes == 64), TL_IF(is_any<TypeOf<T>, float64_t>)>
VECOPS_VFUNC void scatter(T t, TypeOf<T> * p, Vec<Rebind<GatherScatterIndex<TypeOf<T>>, T>> i, Vec<T> v) {
  _mm512_i64scatter_pd(p, i.v, v.v, (int)sizeof(TypeOf<T>));
}
template <TLV_DECL_TAG(T), TL_IF(T::Bytes == 64), TL_IF(is_any<TypeOf<T>, int32_t, uint32_t>)>
VECOPS_VFUNC void scatter(T t, TypeOf<T> * p, Vec<Rebind<GatherScatterIndex<TypeOf<T>>, T>> i, Vec<T> v) {
  _mm512_i32scatter_epi32(p, i.v, v.v, (int)sizeof(TypeOf<T>));
}
template <TLV_DECL_TAG(T), TL_IF(T::Bytes == 64), TL_IF(is_any<TypeOf<T>, int64_t, uint64_t>)>
VECOPS_VFUNC void scatter(T t, TypeOf<T> * p, Vec<Rebind<GatherScatterIndex<TypeOf<T>>, T>> i, Vec<T> v) {
  _mm512_i64scatter_epi64(p, i.v, v.v, (int)sizeof(TypeOf<T>));
}
#endif

template <TLV_DECL_TAG(T), TL_IF(T::Bytes <= 16), TL_IF(is_any<TypeOf<T>, float32_t>)>
VECOPS_VFUNC void scatter(T t, TypeOf<T> * p, Vec<Rebind<GatherScatterIndex<TypeOf<T>>, T>> i, Mask<T> m, Vec<T> v) {
  _mm_mask_i32scatter_ps(p, m.v, i.v, v.v, (int)sizeof(TypeOf<T>));
}
template <TLV_DECL_TAG(T), TL_IF(T::Bytes <= 16), TL_IF(is_any<TypeOf<T>, float64_t>)>
VECOPS_VFUNC void scatter(T t, TypeOf<T> * p, Vec<Rebind<GatherScatterIndex<TypeOf<T>>, T>> i, Mask<T> m, Vec<T> v) {
  _mm_mask_i64scatter_pd(p, m.v, i.v, v.v, (int)sizeof(TypeOf<T>));
}
template <TLV_DECL_TAG(T), TL_IF(T::Bytes <= 16), TL_IF(is_any<TypeOf<T>, int32_t, uint32_t>)>
VECOPS_VFUNC void scatter(T t, TypeOf<T> * p, Vec<Rebind<GatherScatterIndex<TypeOf<T>>, T>> i, Mask<T> m, Vec<T> v) {
  _mm_mask_i32scatter_epi32(p, m.v, i.v, v.v, (int)sizeof(TypeOf<T>));
}
template <TLV_DECL_TAG(T), TL_IF(T::Bytes <= 16), TL_IF(is_any<TypeOf<T>, int64_t, uint64_t>)>
VECOPS_VFUNC void scatter(T t, TypeOf<T> * p, Vec<Rebind<GatherScatterIndex<TypeOf<T>>, T>> i, Mask<T> m, Vec<T> v) {
  _mm_mask_i64scatter_epi64(p, m.v, i.v, v.v, (int)sizeof(TypeOf<T>));
}

template <TLV_DECL_TAG(T), TL_IF(T::Bytes == 32), TL_IF(is_any<TypeOf<T>, float32_t>)>
VECOPS_VFUNC void scatter(T t, TypeOf<T> * p, Vec<Rebind<GatherScatterIndex<TypeOf<T>>, T>> i, Mask<T> m, Vec<T> v) {
  _mm256_mask_i32scatter_ps(p, m.v, i.v, v.v, (int)sizeof(TypeOf<T>));
}
template <TLV_DECL_TAG(T), TL_IF(T::Bytes == 32), TL_IF(is_any<TypeOf<T>, float64_t>)>
VECOPS_VFUNC void scatter(T t, TypeOf<T> * p, Vec<Rebind<GatherScatterIndex<TypeOf<T>>, T>> i, Mask<T> m, Vec<T> v) {
  _mm256_mask_i64scatter_pd(p, m.v, i.v, v.v, (int)sizeof(TypeOf<T>));
}
template <TLV_DECL_TAG(T), TL_IF(T::Bytes == 32), TL_IF(is_any<TypeOf<T>, int32_t, uint32_t>)>
VECOPS_VFUNC void scatter(T t, TypeOf<T> * p, Vec<Rebind<GatherScatterIndex<TypeOf<T>>, T>> i, Mask<T> m, Vec<T> v) {
  _mm256_mask_i32scatter_epi32(p, m.v, i.v, v.v, (int)sizeof(TypeOf<T>));
}
template <TLV_DECL_TAG(T), TL_IF(T::Bytes == 32), TL_IF(is_any<TypeOf<T>, int64_t, uint64_t>)>
VECOPS_VFUNC void scatter(T t, TypeOf<T> * p, Vec<Rebind<GatherScatterIndex<TypeOf<T>>, T>> i, Mask<T> m, Vec<T> v) {
  _mm256_mask_i64scatter_epi64(p, m.v, i.v, v.v, (int)sizeof(TypeOf<T>));
}

template <TLV_DECL_TAG(T), TL_IF(T::Bytes == 64), TL_IF(is_any<TypeOf<T>, float32_t>)>
VECOPS_VFUNC void scatter(T t, TypeOf<T> * p, Vec<Rebind<GatherScatterIndex<TypeOf<T>>, T>> i, Mask<T> m, Vec<T> v) {
  _mm512_mask_i32scatter_ps(p, m.v, i.v, v.v, (int)sizeof(TypeOf<T>));
}
template <TLV_DECL_TAG(T), TL_IF(T::Bytes == 64), TL_IF(is_any<TypeOf<T>, float64_t>)>
VECOPS_VFUNC void scatter(T t, TypeOf<T> * p, Vec<Rebind<GatherScatterIndex<TypeOf<T>>, T>> i, Mask<T> m, Vec<T> v) {
  _mm512_mask_i64scatter_pd(p, m.v, i.v, v.v, (int)sizeof(TypeOf<T>));
}
template <TLV_DECL_TAG(T), TL_IF(T::Bytes == 64), TL_IF(is_any<TypeOf<T>, int32_t, uint32_t>)>
VECOPS_VFUNC void scatter(T t, TypeOf<T> * p, Vec<Rebind<GatherScatterIndex<TypeOf<T>>, T>> i, Mask<T> m, Vec<T> v) {
  _mm512_mask_i32scatter_epi32(p, m.v, i.v, v.v, (int)sizeof(TypeOf<T>));
}
template <TLV_DECL_TAG(T), TL_IF(T::Bytes == 64), TL_IF(is_any<TypeOf<T>, int64_t, uint64_t>)>
VECOPS_VFUNC void scatter(T t, TypeOf<T> * p, Vec<Rebind<GatherScatterIndex<TypeOf<T>>, T>> i, Mask<T> m, Vec<T> v) {
  _mm512_mask_i64scatter_epi64(p, m.v, i.v, v.v, (int)sizeof(TypeOf<T>));
}

// ── Sub-word scatter: 2B ────────────────────────────────────────────────

template <TLV_DECL_TAG(T), TL_IF(is_any<TypeOf<T>, int16_t, uint16_t, float16_t, bfloat16_t>)>
VECOPS_VFUNC void scatter(T t, TypeOf<T>* p, Vec<Rebind<GatherScatterIndex<TypeOf<T>>, T>> i, Vec<T> v) {
  Rebind<GatherScatterIndex<TypeOf<T>>, T> t_i;
  if constexpr (num_words(t_i) == 1) {
    if constexpr (t_i.Bytes == 64) {
#ifdef HAS_AVX512CD
      Vec<decltype(t_i)> i_base, i_sel;
      details::subint2_offsets(t_i, p, i, i_base, i_sel);
      auto vals_32 = _mm512_cvtepu16_epi32(v.v);
      const void* p32_const = reinterpret_cast<const void*>(nuint_t(p) & ~nuint_t(3));
      void* p32 = reinterpret_cast<void*>(nuint_t(p) & ~nuint_t(3));
      auto vals = details::i32_and<decltype(t_i)>(Vec<decltype(t_i)>{vals_32}, details::i32_set1(t_i, 0xffff));
      auto conflict = _mm512_conflict_epi32(i_base.v);
      auto do_pass = [&](__mmask16 mask) -> bool {
        if (!mask) return false;
        if ((_mm512_test_epi32_mask(conflict, conflict) & mask) != 0) return true;
        auto field_mask = details::i32_sllv<decltype(t_i)>(details::i32_set1(t_i, 0xffff), i_sel);
        auto payload = details::i32_sllv<decltype(t_i)>(vals, i_sel);
        auto g = details::i32_gather_bytes(t_i, p32_const, i_base);
        auto d = details::i32_or<decltype(t_i)>(details::i32_andnot<decltype(t_i)>(field_mask, g), payload);
        _mm512_mask_i32scatter_epi32(p32, mask, i_base.v, d.v, 1);
        return false;
      };
      bool fallback = false;
      fallback |= do_pass(_mm512_cmpeq_epi32_mask(i_sel.v, _mm512_set1_epi32(0)));
      fallback |= do_pass(_mm512_cmpeq_epi32_mask(i_sel.v, _mm512_set1_epi32(8)));
      fallback |= do_pass(_mm512_cmpeq_epi32_mask(i_sel.v, _mm512_set1_epi32(16)));
      if (fallback) {
        VECOPS_UNROLL for (int j = 0; j < size(t); ++j)
          p[nint_t(word::get(i, j))] = word::get(v, j);
      }
#else
      VECOPS_UNROLL for (int j = 0; j < size(t); ++j)
        p[nint_t(word::get(i, j))] = word::get(v, j);
#endif
    } else {
      VECOPS_UNROLL for (int j = 0; j < size(t); ++j)
        p[nint_t(word::get(i, j))] = word::get(v, j);
    }
  } else {
    Half<T> t_h;
    auto i_lo = word::lower(t_i, i);
    auto i_hi = word::upper(t_i, i);
    if constexpr (num_words(t) > 1) {
      auto v_lo = word::lower(t, v);
      auto v_hi = word::upper(t, v);
      word::scatter(t_h, p, i_lo, v_lo);
      word::scatter(t_h, p, i_hi, v_hi);
    } else {
      alignas(16) TypeOf<T> v_arr[size(t)];
      word::storeu(t, v_arr, v);
      word::scatter(t_h, p, i_lo, word::loadu(t_h, v_arr));
      word::scatter(t_h, p, i_hi, word::loadu(t_h, v_arr + size(t_h)));
    }
  }
}

template <TLV_DECL_TAG(T), TL_IF(is_any<TypeOf<T>, int16_t, uint16_t, float16_t, bfloat16_t>)>
VECOPS_VFUNC void scatter(T t, TypeOf<T>* p, Vec<Rebind<GatherScatterIndex<TypeOf<T>>, T>> i, Mask<T> m, Vec<T> v) {
  Rebind<GatherScatterIndex<TypeOf<T>>, T> t_i;
  if constexpr (num_words(t_i) == 1) {
    VECOPS_UNROLL for (int j = 0; j < size(t); ++j)
      if (word::get(t, m, j)) p[nint_t(word::get(i, j))] = word::get(v, j);
  } else {
    Half<T> t_h;
    auto i_lo = word::lower(t_i, i);
    auto i_hi = word::upper(t_i, i);
    if constexpr (num_words(t) > 1) {
      auto v_lo = word::lower(t, v);
      auto v_hi = word::upper(t, v);
      auto m_lo = word::lower(t, m);
      auto m_hi = word::upper(t, m);
      word::scatter(t_h, p, i_lo, m_lo, v_lo);
      word::scatter(t_h, p, i_hi, m_hi, v_hi);
    } else {
      alignas(16) TypeOf<T> v_arr[size(t)];
      word::storeu(t, v_arr, v);
      auto v_lo = word::loadu(t_h, v_arr);
      auto v_hi = word::loadu(t_h, v_arr + size(t_h));
      auto m_lo = word::lower(t, m);
      auto m_hi = word::upper(t, m);
      word::scatter(t_h, p, i_lo, m_lo, v_lo);
      word::scatter(t_h, p, i_hi, m_hi, v_hi);
    }
  }
}

// ── Sub-word scatter: 1B ────────────────────────────────────────────────

template <TLV_DECL_TAG(T), TL_IF(is_any<TypeOf<T>, int8_t, uint8_t>)>
VECOPS_VFUNC void scatter(T t, TypeOf<T>* p, Vec<Rebind<GatherScatterIndex<TypeOf<T>>, T>> i, Vec<T> v) {
  Rebind<GatherScatterIndex<TypeOf<T>>, T> t_i;
  if constexpr (num_words(t_i) == 1) {
    if constexpr (t_i.Bytes == 64) {
#ifdef HAS_AVX512CD
      Vec<decltype(t_i)> i_base, i_sel;
      details::subint1_offsets(t_i, p, i, i_base, i_sel);
      auto vals_32 = _mm512_cvtepu8_epi32(v.v);
      const void* p32_const = reinterpret_cast<const void*>(nuint_t(p) & ~nuint_t(3));
      void* p32 = reinterpret_cast<void*>(nuint_t(p) & ~nuint_t(3));
      auto vals = details::i32_and<decltype(t_i)>(Vec<decltype(t_i)>{vals_32}, details::i32_set1(t_i, 0xff));
      auto conflict = _mm512_conflict_epi32(i_base.v);
      auto do_pass = [&](__mmask16 mask) -> bool {
        if (!mask) return false;
        if ((_mm512_test_epi32_mask(conflict, conflict) & mask) != 0) return true;
        auto field_mask = details::i32_sllv<decltype(t_i)>(details::i32_set1(t_i, 0xff), i_sel);
        auto payload = details::i32_sllv<decltype(t_i)>(vals, i_sel);
        auto g = details::i32_gather_bytes(t_i, p32_const, i_base);
        auto d = details::i32_or<decltype(t_i)>(details::i32_andnot<decltype(t_i)>(field_mask, g), payload);
        _mm512_mask_i32scatter_epi32(p32, mask, i_base.v, d.v, 1);
        return false;
      };
      bool fallback = false;
      fallback |= do_pass(_mm512_cmpeq_epi32_mask(i_sel.v, _mm512_set1_epi32(0)));
      fallback |= do_pass(_mm512_cmpeq_epi32_mask(i_sel.v, _mm512_set1_epi32(8)));
      fallback |= do_pass(_mm512_cmpeq_epi32_mask(i_sel.v, _mm512_set1_epi32(16)));
      fallback |= do_pass(_mm512_cmpeq_epi32_mask(i_sel.v, _mm512_set1_epi32(24)));
      if (fallback) {
        VECOPS_UNROLL for (int j = 0; j < size(t); ++j)
          p[nint_t(word::get(i, j))] = word::get(v, j);
      }
#else
      VECOPS_UNROLL for (int j = 0; j < size(t); ++j)
        p[nint_t(word::get(i, j))] = word::get(v, j);
#endif
    } else {
      VECOPS_UNROLL for (int j = 0; j < size(t); ++j)
        p[nint_t(word::get(i, j))] = word::get(v, j);
    }
  } else {
    Half<T> t_h;
    auto i_lo = word::lower(t_i, i);
    auto i_hi = word::upper(t_i, i);
    if constexpr (num_words(t) > 1) {
      auto v_lo = word::lower(t, v);
      auto v_hi = word::upper(t, v);
      word::scatter(t_h, p, i_lo, v_lo);
      word::scatter(t_h, p, i_hi, v_hi);
    } else {
      alignas(16) TypeOf<T> v_arr[size(t)];
      word::storeu(t, v_arr, v);
      word::scatter(t_h, p, i_lo, word::loadu(t_h, v_arr));
      word::scatter(t_h, p, i_hi, word::loadu(t_h, v_arr + size(t_h)));
    }
  }
}

template <TLV_DECL_TAG(T), TL_IF(is_any<TypeOf<T>, int8_t, uint8_t>)>
VECOPS_VFUNC void scatter(T t, TypeOf<T>* p, Vec<Rebind<GatherScatterIndex<TypeOf<T>>, T>> i, Mask<T> m, Vec<T> v) {
  Rebind<GatherScatterIndex<TypeOf<T>>, T> t_i;
  if constexpr (num_words(t_i) == 1) {
    VECOPS_UNROLL for (int j = 0; j < size(t); ++j)
      if (word::get(t, m, j)) p[nint_t(word::get(i, j))] = word::get(v, j);
  } else {
    Half<T> t_h;
    auto i_lo = word::lower(t_i, i);
    auto i_hi = word::upper(t_i, i);
    if constexpr (num_words(t) > 1) {
      auto v_lo = word::lower(t, v);
      auto v_hi = word::upper(t, v);
      auto m_lo = word::lower(t, m);
      auto m_hi = word::upper(t, m);
      word::scatter(t_h, p, i_lo, m_lo, v_lo);
      word::scatter(t_h, p, i_hi, m_hi, v_hi);
    } else {
      alignas(16) TypeOf<T> v_arr[size(t)];
      word::storeu(t, v_arr, v);
      auto v_lo = word::loadu(t_h, v_arr);
      auto v_hi = word::loadu(t_h, v_arr + size(t_h));
      auto m_lo = word::lower(t, m);
      auto m_hi = word::upper(t, m);
      word::scatter(t_h, p, i_lo, m_lo, v_lo);
      word::scatter(t_h, p, i_hi, m_hi, v_hi);
    }
  }
}
#else // HAS_AVX512F
template <TLV_DECL_TAG(T)>
static VECOPS_VFUNC void _scatter_scalar(T t, TypeOf<T>* p, Vec<Rebind<GatherScatterIndex<TypeOf<T>>, T>> i, Vec<T> v) {
  VECOPS_UNROLL for (int j = 0; j < size(t); ++j) {
    p[nint_t(word::get(i, j))] = word::get(v, j);
  }
}

template <TLV_DECL_TAG(T)>
static VECOPS_VFUNC void _mask_scatter_scalar(T t, TypeOf<T>* p, Vec<Rebind<GatherScatterIndex<TypeOf<T>>, T>> i, Mask<T> m, Vec<T> v) {
  VECOPS_UNROLL for (int j = 0; j < size(t); ++j) {
    if (word::get(t, m, j)) {
      p[nint_t(word::get(i, j))] = word::get(v, j);
    }
  }
}

template <TLV_DECL_TAG(T)>
VECOPS_VFUNC void scatter(T t, TypeOf<T> * p, Vec<Rebind<GatherScatterIndex<TypeOf<T>>, T>> i, Vec<T> v) {
  Rebind<GatherScatterIndex<TypeOf<T>>, T> t_i;
  if constexpr (num_words(t_i) == 1) {
    word::_scatter_scalar(t, p, i, v);
  } else {
    Half<T> t_h;
    auto i_lo = word::lower(t_i, i);
    auto i_hi = word::upper(t_i, i);
    if constexpr (num_words(t) > 1) {
      auto v_lo = word::lower(t, v);
      auto v_hi = word::upper(t, v);
      word::scatter(t_h, p, i_lo, v_lo);
      word::scatter(t_h, p, i_hi, v_hi);
    } else {
      alignas(16) TypeOf<T> v_arr[size(t)];
      word::storeu(t, v_arr, v);
      word::scatter(t_h, p, i_lo, word::loadu(t_h, v_arr));
      word::scatter(t_h, p, i_hi, word::loadu(t_h, v_arr + size(t_h)));
    }
  }
}
template <TLV_DECL_TAG(T)>
VECOPS_VFUNC void scatter(T t, TypeOf<T> * p, Vec<Rebind<GatherScatterIndex<TypeOf<T>>, T>> i, Mask<T> m, Vec<T> v) {
  Rebind<GatherScatterIndex<TypeOf<T>>, T> t_i;
  if constexpr (num_words(t_i) == 1) {
    word::_mask_scatter_scalar(t, p, i, m, v);
  } else {
    Half<T> t_h;
    auto i_lo = word::lower(t_i, i);
    auto i_hi = word::upper(t_i, i);
    if constexpr (num_words(t) > 1) {
      auto v_lo = word::lower(t, v);
      auto v_hi = word::upper(t, v);
      auto m_lo = word::lower(t, m);
      auto m_hi = word::upper(t, m);
      word::scatter(t_h, p, i_lo, m_lo, v_lo);
      word::scatter(t_h, p, i_hi, m_hi, v_hi);
    } else {
      alignas(16) TypeOf<T> v_arr[size(t)];
      word::storeu(t, v_arr, v);
      auto v_lo = word::loadu(t_h, v_arr);
      auto v_hi = word::loadu(t_h, v_arr + size(t_h));
      auto m_lo = word::lower(t, m);
      auto m_hi = word::upper(t, m);
      word::scatter(t_h, p, i_lo, m_lo, v_lo);
      word::scatter(t_h, p, i_hi, m_hi, v_hi);
    }
  }
}
#endif // HAS_AVX512F

template <TLV_DECL_TAG(T)>
VECOPS_VFUNC void scatter(T t, TypeOf<T> * p, Vec<Rebind<GatherScatterIndex<TypeOf<T>>, T>> i, nint_t n, Vec<T> v) {
  VECOPS_ASSERT(0 <= n && n <= size(t), "%zd !in 0..%zd", n, size(t));
  auto m = word::mwhilelt(t, 0, n);
  word::scatter(t, p, i, m, v);
}

} // namespace word
} // namespace vecops::vec::CPU_CAPABILITY
//@formatter:on

#endif //VECOPS_X86_LOADSTORE_H
