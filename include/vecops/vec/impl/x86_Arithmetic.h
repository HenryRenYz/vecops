//
// Created by renyz on 2026/3/28.
//

#ifndef VECOPS_X86_ARITHMETIC_H
#define VECOPS_X86_ARITHMETIC_H

#include <cmath>

#include "./x86_Basic.h"
#include "./x86_Bit.h"

//@formatter:on
namespace vecops::vec::CPU_CAPABILITY {
namespace word {
/* ****************************** half_cvt ********************************** */
namespace half_cvt {
#if defined(HAS_AVX512_BF16) || defined(HAS_AVX_NE_CONVERT)
VECOPS_VFUNC __m128bh cast_si128_to_bh(__m128i v) {
  union { __m128i si;__m128bh bh; } u{.si=v};
  return u.bh;
}
VECOPS_VFUNC __m128i cast_bh_to_si128(__m128bh v) {
  union { __m128i si;__m128bh bh; } u{.bh=v};
  return u.si;
}

#if VEC_WIDTH >= 256
VECOPS_VFUNC __m256bh cast_si256_to_bh(__m256i v) {
  union { __m256i si;__m256bh bh; } u{.si=v};
  return u.bh;
}
VECOPS_VFUNC __m256i cast_bh_to_si256(__m256bh v) {
  union { __m256i si;__m256bh bh; } u{.bh=v};
  return u.si;
}
#endif

#if VEC_WIDTH >= 512
VECOPS_VFUNC __m512bh cast_si512_to_bh(__m512i v) {
  union { __m512i si;__m512bh bh; } u{.si=v};
  return u.bh;
}
VECOPS_VFUNC __m512i cast_bh_to_si512(__m512bh v) {
  union { __m512i si;__m512bh bh; } u{.bh=v};
  return u.si;
}
#endif

#endif

VECOPS_VFUNC void cvt_bf16_to_two_fp32(__m128i a, __m128& lo, __m128& hi){
  #if defined(HAS_AVX512_BF16)
  auto x256 = _mm256_cvtpbh_ps(cast_si128_to_bh(a));
  lo = _mm256_castps256_ps128(x256);
  hi = _mm256_extractf128_ps(x256, 1);
  lo = _mm_cvtpbh_ps(cast_si128_to_bh(a));
  #else
  lo = _mm_castsi128_ps(_mm_slli_epi32(_mm_cvtepu16_epi32(a), 16));
  hi = _mm_castsi128_ps(_mm_slli_epi32(_mm_cvtepu16_epi32(_mm_srli_si128(a, 8)), 16));
  #endif
}
VECOPS_VFUNC __m128i cvt_two_fp32_to_bf16(__m128 lo, __m128 hi) {
  #if defined(HAS_AVX512_BF16)
  return cast_bh_to_si128(_mm_cvtne2ps_pbh(hi, lo));
  #elif defined(HAS_AVX_NE_CONVERT)
  auto t_256 = _mm256_insertf128_ps(_mm256_castps128_ps256(lo), hi, 1);
  return cast_bh_to_si128(_mm256_cvtneps_pbh(t_256));
  #else
  __m128i lo_val = _mm_castps_si128(lo), hi_val = _mm_castps_si128(hi);
  __m128i nan = _mm_set1_epi32(0x7FC0), ones = _mm_set1_epi32(0x1), bias = _mm_set1_epi32(0x7fff);
  auto t_lo = _mm_add_epi32(_mm_and_si128(_mm_srli_epi32(lo_val, 16), ones), bias);
  t_lo = _mm_srli_epi32(_mm_add_epi32(t_lo, lo_val), 16);
  t_lo = _mm_blendv_epi8(nan, t_lo, _mm_castps_si128(_mm_cmpord_ps(lo, lo)));
  auto t_hi = _mm_add_epi32(_mm_and_si128(_mm_srli_epi32(hi_val, 16), ones), bias);
  t_hi = _mm_srli_epi32(_mm_add_epi32(t_hi, hi_val), 16);
  t_hi = _mm_blendv_epi8(nan, t_hi, _mm_castps_si128(_mm_cmpord_ps(hi, hi)));
  return _mm_packus_epi32(t_lo, t_hi);
  #endif
}

#if VEC_WIDTH >= 256
VECOPS_VFUNC void cvt_bf16_to_two_fp32(__m256i a, __m256& o1, __m256& o2) {
  #if defined(HAS_AVX512_BF16)
  auto x512 = _mm512_cvtpbh_ps(cast_si256_to_bh(a));
  o1 = _mm512_castps512_ps256(x512);
  o2 = _mm512_extractf32x8_ps(x512, 1);
  #else
  __m128i lo128 = _mm256_castsi256_si128(a), hi128 = _mm256_extractf128_si256(a, 1);
  o1 = _mm256_castsi256_ps(_mm256_slli_epi32(_mm256_cvtepu16_epi32(lo128), 16));
  o2 = _mm256_castsi256_ps(_mm256_slli_epi32(_mm256_cvtepu16_epi32(hi128), 16));
  #endif
}
VECOPS_VFUNC __m256i cvt_two_fp32_to_bf16(__m256 a, __m256 b){
  #if defined(HAS_AVX512_BF16)
  return cast_bh_to_si256(_mm256_cvtne2ps_pbh(b, a));
  #elif defined(HAS_AVX_NE_CONVERT)
  __m128bh lo_bh=_mm256_cvtneps_pbh(a),hi_bh=_mm256_cvtneps_pbh(b);
  return _mm256_insertf128_si256(_mm256_castsi128_si256(cast_bh_to_si128(lo_bh)),cast_bh_to_si128(hi_bh),1);
  #else
  __m256i lo=_mm256_castps_si256(a),hi=_mm256_castps_si256(b);
  __m256i nan=_mm256_set1_epi32(0x7FC0),ones=_mm256_set1_epi32(0x1),bias=_mm256_set1_epi32(0x7fff);
  __m256i mlo=_mm256_castps_si256(_mm256_cmp_ps(a,a,_CMP_ORD_Q));
  __m256i mhi=_mm256_castps_si256(_mm256_cmp_ps(b,b,_CMP_ORD_Q));
  auto t_lo=_mm256_and_si256(_mm256_srli_epi32(lo,16),ones);t_lo=_mm256_add_epi32(t_lo,bias);
  t_lo=_mm256_add_epi32(t_lo,lo);t_lo=_mm256_srli_epi32(t_lo,16);t_lo=_mm256_blendv_epi8(nan,t_lo,mlo);
  auto t_hi=_mm256_and_si256(_mm256_srli_epi32(hi,16),ones);t_hi=_mm256_add_epi32(t_hi,bias);
  t_hi=_mm256_add_epi32(t_hi,hi);t_hi=_mm256_srli_epi32(t_hi,16);t_hi=_mm256_blendv_epi8(nan,t_hi,mhi);
  auto p=_mm256_packus_epi32(t_lo,t_hi);return _mm256_permute4x64_epi64(p,0xd8);
  #endif
}
#endif

#if VEC_WIDTH >= 512
VECOPS_VFUNC void cvt_bf16_to_two_fp32(__m512i a, __m512& o1, __m512& o2) {
  __m256i lo256 = _mm512_castsi512_si256(a), hi256 = _mm512_extracti64x4_epi64(a, 1);
  #if defined(HAS_AVX512_BF16)
  o1 = _mm512_cvtpbh_ps(cast_si256_to_bh(lo256));
  o2 = _mm512_cvtpbh_ps(cast_si256_to_bh(hi256));
  #else
  o1 = _mm512_castsi512_ps(_mm512_slli_epi32(_mm512_cvtepu16_epi32(lo256), 16));
  o2 = _mm512_castsi512_ps(_mm512_slli_epi32(_mm512_cvtepu16_epi32(hi256), 16));
  #endif
}
VECOPS_VFUNC __m512i cvt_two_fp32_to_bf16(__m512 a, __m512 b){
  #if defined(HAS_AVX512_BF16)
  return cast_bh_to_si512(_mm512_cvtne2ps_pbh(b, a));
  #else
  __m256 a_lo = _mm512_castps512_ps256(a), a_hi = _mm512_extractf32x8_ps(a, 1);
  __m256 b_lo = _mm512_castps512_ps256(b), b_hi = _mm512_extractf32x8_ps(b, 1);
  __m256i r_lo = cvt_two_fp32_to_bf16(a_lo, a_hi), r_hi = cvt_two_fp32_to_bf16(b_lo, b_hi);
  return _mm512_inserti64x4(_mm512_castsi256_si512(r_lo), r_hi, 1);
  #endif
}
#endif

#ifdef HAS_F16C
VECOPS_VFUNC void cvt_fp16_to_two_fp32(__m128i a,__m128& lo,__m128& hi){
  lo=_mm_cvtph_ps(a);hi=_mm_cvtph_ps(_mm_srli_si128(a,8));
}
VECOPS_VFUNC __m128i cvt_two_fp32_to_fp16(__m128 lo,__m128 hi){
  __m128i lo16=_mm_cvtps_ph(lo,_MM_FROUND_TO_NEAREST_INT|_MM_FROUND_NO_EXC);
  __m128i hi16=_mm_cvtps_ph(hi,_MM_FROUND_TO_NEAREST_INT|_MM_FROUND_NO_EXC);
  return _mm_or_si128(lo16,_mm_slli_si128(hi16,8));
}
#if VEC_WIDTH >= 256
VECOPS_VFUNC void cvt_fp16_to_two_fp32(__m256i a,__m256& o1,__m256& o2){
  __m128i lo128=_mm256_castsi256_si128(a),hi128=_mm256_extractf128_si256(a,1);
  o1=_mm256_cvtph_ps(lo128);o2=_mm256_cvtph_ps(hi128);
}
VECOPS_VFUNC __m256i cvt_two_fp32_to_fp16(__m256 a,__m256 b){
  __m128i lo=_mm256_cvtps_ph(a,_MM_FROUND_TO_NEAREST_INT|_MM_FROUND_NO_EXC);
  __m128i hi=_mm256_cvtps_ph(b,_MM_FROUND_TO_NEAREST_INT|_MM_FROUND_NO_EXC);
  return _mm256_insertf128_si256(_mm256_castsi128_si256(lo),hi,1);
}
#endif
#if VEC_WIDTH >= 512
VECOPS_VFUNC void cvt_fp16_to_two_fp32(__m512i a,__m512& o1,__m512& o2){
  __m256i lo256=_mm512_castsi512_si256(a),hi256=_mm512_extracti64x4_epi64(a,1);
  __m256 t1,t2;cvt_fp16_to_two_fp32(lo256,t1,t2);o1=_mm512_insertf32x8(_mm512_castps256_ps512(t1),t2,1);
  cvt_fp16_to_two_fp32(hi256,t1,t2);o2=_mm512_insertf32x8(_mm512_castps256_ps512(t1),t2,1);
}
VECOPS_VFUNC __m512i cvt_two_fp32_to_fp16(__m512 a,__m512 b){
  __m256i lo=_mm512_cvtps_ph(a,_MM_FROUND_TO_NEAREST_INT|_MM_FROUND_NO_EXC);
  __m256i hi=_mm512_cvtps_ph(b,_MM_FROUND_TO_NEAREST_INT|_MM_FROUND_NO_EXC);
  return _mm512_inserti64x4(_mm512_castsi256_si512(lo),hi,1);
}
#endif
#endif
#if !defined(HAS_F16C)
VECOPS_VFUNC void cvt_fp16_to_two_fp32(__m128i a,__m128& lo,__m128& hi){
  alignas(16) float16_t arr[8];alignas(16) float arr_f[8];
  _mm_store_si128((__m128i*)arr,a);for(int i=0;i<8;++i)arr_f[i]=float(arr[i]);
  lo=_mm_load_ps(arr_f);hi=_mm_load_ps(arr_f+4);
}
VECOPS_VFUNC __m128i cvt_two_fp32_to_fp16(__m128 lo,__m128 hi){
  alignas(16) float arr_f[8];alignas(16) float16_t arr[8];
  _mm_store_ps(arr_f,lo);_mm_store_ps(arr_f+4,hi);
  for(int i=0;i<8;++i)arr[i]=float16_t(arr_f[i]);return _mm_load_si128((__m128i*)arr);
}
#endif
} // namespace half_cvt

#define VECOPS_X86_HALF_BIN_128(FN,TYPE,FP32_FN,CVT_TO,CVT_FROM) template<TLV_DECL_VEC(V),typename T=Vec2Tag<V>,TL_IF(T::Bytes<=16),TL_IF(is_any<TypeOf<T>,TYPE>)> VECOPS_VFUNC V FN(V a,V b){__m128 alo,ahi,blo,bhi;CVT_TO(a.v,alo,ahi);CVT_TO(b.v,blo,bhi);return CVT_FROM(FP32_FN(alo,blo),FP32_FN(ahi,bhi));}
#if VEC_WIDTH>=256
#define VECOPS_X86_HALF_BIN_256(FN,TYPE,FP32_FN,CVT_TO,CVT_FROM) template<TLV_DECL_VEC(V),typename T=Vec2Tag<V>,TL_IF(T::Bytes==32),TL_IF(is_any<TypeOf<T>,TYPE>)> VECOPS_VFUNC V FN(V a,V b){__m256 alo,ahi,blo,bhi;CVT_TO(a.v,alo,ahi);CVT_TO(b.v,blo,bhi);return CVT_FROM(FP32_FN(alo,blo),FP32_FN(ahi,bhi));}
#endif
#if VEC_WIDTH>=512
#define VECOPS_X86_HALF_BIN_512(FN,TYPE,FP32_FN,CVT_TO,CVT_FROM) template<TLV_DECL_VEC(V),typename T=Vec2Tag<V>,TL_IF(T::Bytes==64),TL_IF(is_any<TypeOf<T>,TYPE>)> VECOPS_VFUNC V FN(V a,V b){__m512 alo,ahi,blo,bhi;CVT_TO(a.v,alo,ahi);CVT_TO(b.v,blo,bhi);return CVT_FROM(FP32_FN(alo,blo),FP32_FN(ahi,bhi));}
#endif

#define VECOPS_X86_HALF_UNARY_128(FN,TYPE,FP32_FN,CVT_TO,CVT_FROM) template<TLV_DECL_VEC(V),typename T=Vec2Tag<V>,TL_IF(T::Bytes<=16),TL_IF(is_any<TypeOf<T>,TYPE>)> VECOPS_VFUNC V FN(V v){__m128 lo,hi;CVT_TO(v.v,lo,hi);return CVT_FROM(FP32_FN(lo),FP32_FN(hi));}
#if VEC_WIDTH>=256
#define VECOPS_X86_HALF_UNARY_256(FN,TYPE,FP32_FN,CVT_TO,CVT_FROM) template<TLV_DECL_VEC(V),typename T=Vec2Tag<V>,TL_IF(T::Bytes==32),TL_IF(is_any<TypeOf<T>,TYPE>)> VECOPS_VFUNC V FN(V v){__m256 lo,hi;CVT_TO(v.v,lo,hi);return CVT_FROM(FP32_FN(lo),FP32_FN(hi));}
#endif
#if VEC_WIDTH>=512
#define VECOPS_X86_HALF_UNARY_512(FN,TYPE,FP32_FN,CVT_TO,CVT_FROM) template<TLV_DECL_VEC(V),typename T=Vec2Tag<V>,TL_IF(T::Bytes==64),TL_IF(is_any<TypeOf<T>,TYPE>)> VECOPS_VFUNC V FN(V v){__m512 lo,hi;CVT_TO(v.v,lo,hi);return CVT_FROM(FP32_FN(lo),FP32_FN(hi));}
#endif

#ifdef HAS_AVX512_FP16
#define VECOPS_X86_FP16_NATIVE_BIN_128(FN,FP16_FN) template<TLV_DECL_VEC(V),typename T=Vec2Tag<V>,TL_IF(T::Bytes<=16),TL_IF(is_any<TypeOf<T>,float16_t>)> VECOPS_VFUNC V FN(V a,V b){return _mm_castph_si128(FP16_FN(_mm_castsi128_ph(a.v),_mm_castsi128_ph(b.v)));}
#if VEC_WIDTH>=256
#define VECOPS_X86_FP16_NATIVE_BIN_256(FN,FP16_FN) template<TLV_DECL_VEC(V),typename T=Vec2Tag<V>,TL_IF(T::Bytes==32),TL_IF(is_any<TypeOf<T>,float16_t>)> VECOPS_VFUNC V FN(V a,V b){return _mm256_castph_si256(FP16_FN(_mm256_castsi256_ph(a.v),_mm256_castsi256_ph(b.v)));}
#endif
#if VEC_WIDTH>=512
#define VECOPS_X86_FP16_NATIVE_BIN_512(FN,FP16_FN) template<TLV_DECL_VEC(V),typename T=Vec2Tag<V>,TL_IF(T::Bytes==64),TL_IF(is_any<TypeOf<T>,float16_t>)> VECOPS_VFUNC V FN(V a,V b){return _mm512_castph_si512(FP16_FN(_mm512_castsi512_ph(a.v),_mm512_castsi512_ph(b.v)));}
#endif
#define VECOPS_X86_FP16_NATIVE_UNARY_128(FN,FP16_FN) template<TLV_DECL_VEC(V),typename T=Vec2Tag<V>,TL_IF(T::Bytes<=16),TL_IF(is_any<TypeOf<T>,float16_t>)> VECOPS_VFUNC V FN(V v){return _mm_castph_si128(FP16_FN(_mm_castsi128_ph(v.v)));}
#if VEC_WIDTH>=256
#define VECOPS_X86_FP16_NATIVE_UNARY_256(FN,FP16_FN) template<TLV_DECL_VEC(V),typename T=Vec2Tag<V>,TL_IF(T::Bytes==32),TL_IF(is_any<TypeOf<T>,float16_t>)> VECOPS_VFUNC V FN(V v){return _mm256_castph_si256(FP16_FN(_mm256_castsi256_ph(v.v)));}
#endif
#if VEC_WIDTH>=512
#define VECOPS_X86_FP16_NATIVE_UNARY_512(FN,FP16_FN) template<TLV_DECL_VEC(V),typename T=Vec2Tag<V>,TL_IF(T::Bytes==64),TL_IF(is_any<TypeOf<T>,float16_t>)> VECOPS_VFUNC V FN(V v){return _mm512_castph_si512(FP16_FN(_mm512_castsi512_ph(v.v)));}
#endif
#endif

/* ************************************************************************** */
//                                   Add                                      //
/* ************************************************************************** */
template <TLV_DECL_VEC(V), typename T = Vec2Tag<V>, TL_IF(T::Bytes <= 16), TL_IF(is_any<TypeOf<T>, float32_t>)>
VECOPS_VFUNC V add(V a, V b) {
  return _mm_add_ps(a.v, b.v);
}
template <TLV_DECL_VEC(V), typename T = Vec2Tag<V>, TL_IF(T::Bytes <= 16), TL_IF(is_any<TypeOf<T>, float64_t>)>
VECOPS_VFUNC V add(V a, V b) {
  return _mm_add_pd(a.v, b.v);
}
template <TLV_DECL_VEC(V), typename T = Vec2Tag<V>, TL_IF(T::Bytes <= 16), TL_IF(is_any<TypeOf<T>, int8_t, uint8_t>)>
VECOPS_VFUNC V add(V a, V b) {
  return _mm_add_epi8(a.v, b.v);
}
template <TLV_DECL_VEC(V), typename T = Vec2Tag<V>, TL_IF(T::Bytes <= 16), TL_IF(is_any<TypeOf<T>, int16_t, uint16_t>)>
VECOPS_VFUNC V add(V a, V b) {
  return _mm_add_epi16(a.v, b.v);
}
template <TLV_DECL_VEC(V), typename T = Vec2Tag<V>, TL_IF(T::Bytes <= 16), TL_IF(is_any<TypeOf<T>, int32_t, uint32_t>)>
VECOPS_VFUNC V add(V a, V b) {
  return _mm_add_epi32(a.v, b.v);
}
template <TLV_DECL_VEC(V), typename T = Vec2Tag<V>, TL_IF(T::Bytes <= 16), TL_IF(is_any<TypeOf<T>, int64_t, uint64_t>)>
VECOPS_VFUNC V add(V a, V b) {
  return _mm_add_epi64(a.v, b.v);
}
#ifdef HAS_AVX512_FP16
VECOPS_X86_FP16_NATIVE_BIN_128(add,_mm_add_ph)
#else
VECOPS_X86_HALF_BIN_128(add,float16_t,_mm_add_ps,half_cvt::cvt_fp16_to_two_fp32,half_cvt::cvt_two_fp32_to_fp16)
#endif
VECOPS_X86_HALF_BIN_128(add,bfloat16_t,_mm_add_ps,half_cvt::cvt_bf16_to_two_fp32,half_cvt::cvt_two_fp32_to_bf16)

#if VEC_WIDTH >= 256
template <TLV_DECL_VEC(V), typename T = Vec2Tag<V>, TL_IF(T::Bytes == 32), TL_IF(is_any<TypeOf<T>, float32_t>)>
VECOPS_VFUNC V add(V a, V b) {
  return _mm256_add_ps(a.v, b.v);
}
template <TLV_DECL_VEC(V), typename T = Vec2Tag<V>, TL_IF(T::Bytes == 32), TL_IF(is_any<TypeOf<T>, float64_t>)>
VECOPS_VFUNC V add(V a, V b) {
  return _mm256_add_pd(a.v, b.v);
}
template <TLV_DECL_VEC(V), typename T = Vec2Tag<V>, TL_IF(T::Bytes == 32), TL_IF(is_any<TypeOf<T>, int8_t, uint8_t>)>
VECOPS_VFUNC V add(V a, V b) {
  return _mm256_add_epi8(a.v, b.v);
}
template <TLV_DECL_VEC(V), typename T = Vec2Tag<V>, TL_IF(T::Bytes == 32), TL_IF(is_any<TypeOf<T>, int16_t, uint16_t>)>
VECOPS_VFUNC V add(V a, V b) {
  return _mm256_add_epi16(a.v, b.v);
}
template <TLV_DECL_VEC(V), typename T = Vec2Tag<V>, TL_IF(T::Bytes == 32), TL_IF(is_any<TypeOf<T>, int32_t, uint32_t>)>
VECOPS_VFUNC V add(V a, V b) {
  return _mm256_add_epi32(a.v, b.v);
}
template <TLV_DECL_VEC(V), typename T = Vec2Tag<V>, TL_IF(T::Bytes == 32), TL_IF(is_any<TypeOf<T>, int64_t, uint64_t>)>
VECOPS_VFUNC V add(V a, V b) {
  return _mm256_add_epi64(a.v, b.v);
}
#ifdef HAS_AVX512_FP16
VECOPS_X86_FP16_NATIVE_BIN_256(add,_mm256_add_ph)
#else
VECOPS_X86_HALF_BIN_256(add,float16_t,_mm256_add_ps,half_cvt::cvt_fp16_to_two_fp32,half_cvt::cvt_two_fp32_to_fp16)
#endif
VECOPS_X86_HALF_BIN_256(add,bfloat16_t,_mm256_add_ps,half_cvt::cvt_bf16_to_two_fp32,half_cvt::cvt_two_fp32_to_bf16)

#endif // VEC_WIDTH >= 256

#if VEC_WIDTH >= 512
template <TLV_DECL_VEC(V), typename T = Vec2Tag<V>, TL_IF(T::Bytes == 64), TL_IF(is_any<TypeOf<T>, float32_t>)>
VECOPS_VFUNC V add(V a, V b) {
  return _mm512_add_ps(a.v, b.v);
}
template <TLV_DECL_VEC(V), typename T = Vec2Tag<V>, TL_IF(T::Bytes == 64), TL_IF(is_any<TypeOf<T>, float64_t>)>
VECOPS_VFUNC V add(V a, V b) {
  return _mm512_add_pd(a.v, b.v);
}
template <TLV_DECL_VEC(V), typename T = Vec2Tag<V>, TL_IF(T::Bytes == 64), TL_IF(is_any<TypeOf<T>, int8_t, uint8_t>)>
VECOPS_VFUNC V add(V a, V b) {
  return _mm512_add_epi8(a.v, b.v);
}
template <TLV_DECL_VEC(V), typename T = Vec2Tag<V>, TL_IF(T::Bytes == 64), TL_IF(is_any<TypeOf<T>, int16_t, uint16_t>)>
VECOPS_VFUNC V add(V a, V b) {
  return _mm512_add_epi16(a.v, b.v);
}
template <TLV_DECL_VEC(V), typename T = Vec2Tag<V>, TL_IF(T::Bytes == 64), TL_IF(is_any<TypeOf<T>, int32_t, uint32_t>)>
VECOPS_VFUNC V add(V a, V b) {
  return _mm512_add_epi32(a.v, b.v);
}
template <TLV_DECL_VEC(V), typename T = Vec2Tag<V>, TL_IF(T::Bytes == 64), TL_IF(is_any<TypeOf<T>, int64_t, uint64_t>)>
VECOPS_VFUNC V add(V a, V b) {
  return _mm512_add_epi64(a.v, b.v);
}
#ifdef HAS_AVX512_FP16
VECOPS_X86_FP16_NATIVE_BIN_512(add,_mm512_add_ph)
#else
VECOPS_X86_HALF_BIN_512(add,float16_t,_mm512_add_ps,half_cvt::cvt_fp16_to_two_fp32,half_cvt::cvt_two_fp32_to_fp16)
#endif
VECOPS_X86_HALF_BIN_512(add,bfloat16_t,_mm512_add_ps,half_cvt::cvt_bf16_to_two_fp32,half_cvt::cvt_two_fp32_to_bf16)
#endif // VEC_WIDTH >= 512

#ifdef HAS_AVX512DQ
template <TLV_DECL_VEC(V), typename T = Vec2Tag<V>, TL_IF(T::Bytes <= 16), TL_IF(is_any<TypeOf<T>, float32_t>)>
VECOPS_VFUNC V add(V a, V b, Mask<T> m) {
  return _mm_mask_add_ps(a.v, m.v, a.v, b.v);
}
template <TLV_DECL_VEC(V), typename T = Vec2Tag<V>, TL_IF(T::Bytes <= 16), TL_IF(is_any<TypeOf<T>, float64_t>)>
VECOPS_VFUNC V add(V a, V b, Mask<T> m) {
  return _mm_mask_add_pd(a.v, m.v, a.v, b.v);
}
template <TLV_DECL_VEC(V), typename T = Vec2Tag<V>, TL_IF(T::Bytes <= 16), TL_IF(is_any<TypeOf<T>, int8_t, uint8_t>)>
VECOPS_VFUNC V add(V a, V b, Mask<T> m) {
  return _mm_mask_add_epi8(a.v, m.v, a.v, b.v);
}
template <TLV_DECL_VEC(V), typename T = Vec2Tag<V>, TL_IF(T::Bytes <= 16), TL_IF(is_any<TypeOf<T>, int16_t, uint16_t>)>
VECOPS_VFUNC V add(V a, V b, Mask<T> m) {
  return _mm_mask_add_epi16(a.v, m.v, a.v, b.v);
}
template <TLV_DECL_VEC(V), typename T = Vec2Tag<V>, TL_IF(T::Bytes <= 16), TL_IF(is_any<TypeOf<T>, int32_t, uint32_t>)>
VECOPS_VFUNC V add(V a, V b, Mask<T> m) {
  return _mm_mask_add_epi32(a.v, m.v, a.v, b.v);
}
template <TLV_DECL_VEC(V), typename T = Vec2Tag<V>, TL_IF(T::Bytes <= 16), TL_IF(is_any<TypeOf<T>, int64_t, uint64_t>)>
VECOPS_VFUNC V add(V a, V b, Mask<T> m) {
  return _mm_mask_add_epi64(a.v, m.v, a.v, b.v);
}

template <TLV_DECL_VEC(V), typename T = Vec2Tag<V>, TL_IF(T::Bytes == 32), TL_IF(is_any<TypeOf<T>, float32_t>)>
VECOPS_VFUNC V add(V a, V b, Mask<T> m) {
  return _mm256_mask_add_ps(a.v, m.v, a.v, b.v);
}
template <TLV_DECL_VEC(V), typename T = Vec2Tag<V>, TL_IF(T::Bytes == 32), TL_IF(is_any<TypeOf<T>, float64_t>)>
VECOPS_VFUNC V add(V a, V b, Mask<T> m) {
  return _mm256_mask_add_pd(a.v, m.v, a.v, b.v);
}
template <TLV_DECL_VEC(V), typename T = Vec2Tag<V>, TL_IF(T::Bytes == 32), TL_IF(is_any<TypeOf<T>, int8_t, uint8_t>)>
VECOPS_VFUNC V add(V a, V b, Mask<T> m) {
  return _mm256_mask_add_epi8(a.v, m.v, a.v, b.v);
}
template <TLV_DECL_VEC(V), typename T = Vec2Tag<V>, TL_IF(T::Bytes == 32), TL_IF(is_any<TypeOf<T>, int16_t, uint16_t>)>
VECOPS_VFUNC V add(V a, V b, Mask<T> m) {
  return _mm256_mask_add_epi16(a.v, m.v, a.v, b.v);
}
template <TLV_DECL_VEC(V), typename T = Vec2Tag<V>, TL_IF(T::Bytes == 32), TL_IF(is_any<TypeOf<T>, int32_t, uint32_t>)>
VECOPS_VFUNC V add(V a, V b, Mask<T> m) {
  return _mm256_mask_add_epi32(a.v, m.v, a.v, b.v);
}
template <TLV_DECL_VEC(V), typename T = Vec2Tag<V>, TL_IF(T::Bytes == 32), TL_IF(is_any<TypeOf<T>, int64_t, uint64_t>)>
VECOPS_VFUNC V add(V a, V b, Mask<T> m) {
  return _mm256_mask_add_epi64(a.v, m.v, a.v, b.v);
}

template <TLV_DECL_VEC(V), typename T = Vec2Tag<V>, TL_IF(T::Bytes == 64), TL_IF(is_any<TypeOf<T>, float32_t>)>
VECOPS_VFUNC V add(V a, V b, Mask<T> m) {
  return _mm512_mask_add_ps(a.v, m.v, a.v, b.v);
}
template <TLV_DECL_VEC(V), typename T = Vec2Tag<V>, TL_IF(T::Bytes == 64), TL_IF(is_any<TypeOf<T>, float64_t>)>
VECOPS_VFUNC V add(V a, V b, Mask<T> m) {
  return _mm512_mask_add_pd(a.v, m.v, a.v, b.v);
}
template <TLV_DECL_VEC(V), typename T = Vec2Tag<V>, TL_IF(T::Bytes == 64), TL_IF(is_any<TypeOf<T>, int8_t, uint8_t>)>
VECOPS_VFUNC V add(V a, V b, Mask<T> m) {
  return _mm512_mask_add_epi8(a.v, m.v, a.v, b.v);
}
template <TLV_DECL_VEC(V), typename T = Vec2Tag<V>, TL_IF(T::Bytes == 64), TL_IF(is_any<TypeOf<T>, int16_t, uint16_t>)>
VECOPS_VFUNC V add(V a, V b, Mask<T> m) {
  return _mm512_mask_add_epi16(a.v, m.v, a.v, b.v);
}
template <TLV_DECL_VEC(V), typename T = Vec2Tag<V>, TL_IF(T::Bytes == 64), TL_IF(is_any<TypeOf<T>, int32_t, uint32_t>)>
VECOPS_VFUNC V add(V a, V b, Mask<T> m) {
  return _mm512_mask_add_epi32(a.v, m.v, a.v, b.v);
}
template <TLV_DECL_VEC(V), typename T = Vec2Tag<V>, TL_IF(T::Bytes == 64), TL_IF(is_any<TypeOf<T>, int64_t, uint64_t>)>
VECOPS_VFUNC V add(V a, V b, Mask<T> m) {
  return _mm512_mask_add_epi64(a.v, m.v, a.v, b.v);
}
template <TLV_DECL_VEC(V), typename T = Vec2Tag<V>, TL_IF(is_any<TypeOf<T>, float16_t, bfloat16_t>)>
VECOPS_VFUNC V add(V a, V b, Mask<T> m) {
  return word::blend(a, m, word::add(a, b));
}
#else // HAS_AVX512DQ
template <TLV_DECL_VEC(V), typename T = Vec2Tag<V>>
VECOPS_VFUNC V add(V a, V b, Mask<T> m) {
  return word::blend(a, m, word::add(a, b));
}
#endif // HAS_AVX512DQ


/* ************************************************************************** */
//                                   Sub                                      //
/* ************************************************************************** */
template <TLV_DECL_VEC(V), typename T = Vec2Tag<V>, TL_IF(T::Bytes <= 16), TL_IF(is_any<TypeOf<T>, float32_t>)>
VECOPS_VFUNC V sub(V a, V b) {
  return _mm_sub_ps(a.v, b.v);
}
template <TLV_DECL_VEC(V), typename T = Vec2Tag<V>, TL_IF(T::Bytes <= 16), TL_IF(is_any<TypeOf<T>, float64_t>)>
VECOPS_VFUNC V sub(V a, V b) {
  return _mm_sub_pd(a.v, b.v);
}
template <TLV_DECL_VEC(V), typename T = Vec2Tag<V>, TL_IF(T::Bytes <= 16), TL_IF(is_any<TypeOf<T>, int8_t, uint8_t>)>
VECOPS_VFUNC V sub(V a, V b) {
  return _mm_sub_epi8(a.v, b.v);
}
template <TLV_DECL_VEC(V), typename T = Vec2Tag<V>, TL_IF(T::Bytes <= 16), TL_IF(is_any<TypeOf<T>, int16_t, uint16_t>)>
VECOPS_VFUNC V sub(V a, V b) {
  return _mm_sub_epi16(a.v, b.v);
}
template <TLV_DECL_VEC(V), typename T = Vec2Tag<V>, TL_IF(T::Bytes <= 16), TL_IF(is_any<TypeOf<T>, int32_t, uint32_t>)>
VECOPS_VFUNC V sub(V a, V b) {
  return _mm_sub_epi32(a.v, b.v);
}
template <TLV_DECL_VEC(V), typename T = Vec2Tag<V>, TL_IF(T::Bytes <= 16), TL_IF(is_any<TypeOf<T>, int64_t, uint64_t>)>
VECOPS_VFUNC V sub(V a, V b) {
  return _mm_sub_epi64(a.v, b.v);
}
#ifdef HAS_AVX512_FP16
VECOPS_X86_FP16_NATIVE_BIN_128(sub,_mm_sub_ph)
#else
VECOPS_X86_HALF_BIN_128(sub,float16_t,_mm_sub_ps,half_cvt::cvt_fp16_to_two_fp32,half_cvt::cvt_two_fp32_to_fp16)
#endif
VECOPS_X86_HALF_BIN_128(sub,bfloat16_t,_mm_sub_ps,half_cvt::cvt_bf16_to_two_fp32,half_cvt::cvt_two_fp32_to_bf16)

#if VEC_WIDTH >= 256
template <TLV_DECL_VEC(V), typename T = Vec2Tag<V>, TL_IF(T::Bytes == 32), TL_IF(is_any<TypeOf<T>, float32_t>)>
VECOPS_VFUNC V sub(V a, V b) {
  return _mm256_sub_ps(a.v, b.v);
}
template <TLV_DECL_VEC(V), typename T = Vec2Tag<V>, TL_IF(T::Bytes == 32), TL_IF(is_any<TypeOf<T>, float64_t>)>
VECOPS_VFUNC V sub(V a, V b) {
  return _mm256_sub_pd(a.v, b.v);
}
template <TLV_DECL_VEC(V), typename T = Vec2Tag<V>, TL_IF(T::Bytes == 32), TL_IF(is_any<TypeOf<T>, int8_t, uint8_t>)>
VECOPS_VFUNC V sub(V a, V b) {
  return _mm256_sub_epi8(a.v, b.v);
}
template <TLV_DECL_VEC(V), typename T = Vec2Tag<V>, TL_IF(T::Bytes == 32), TL_IF(is_any<TypeOf<T>, int16_t, uint16_t>)>
VECOPS_VFUNC V sub(V a, V b) {
  return _mm256_sub_epi16(a.v, b.v);
}
template <TLV_DECL_VEC(V), typename T = Vec2Tag<V>, TL_IF(T::Bytes == 32), TL_IF(is_any<TypeOf<T>, int32_t, uint32_t>)>
VECOPS_VFUNC V sub(V a, V b) {
  return _mm256_sub_epi32(a.v, b.v);
}
template <TLV_DECL_VEC(V), typename T = Vec2Tag<V>, TL_IF(T::Bytes == 32), TL_IF(is_any<TypeOf<T>, int64_t, uint64_t>)>
VECOPS_VFUNC V sub(V a, V b) {
  return _mm256_sub_epi64(a.v, b.v);
}
#ifdef HAS_AVX512_FP16
VECOPS_X86_FP16_NATIVE_BIN_256(sub,_mm256_sub_ph)
#else
VECOPS_X86_HALF_BIN_256(sub,float16_t,_mm256_sub_ps,half_cvt::cvt_fp16_to_two_fp32,half_cvt::cvt_two_fp32_to_fp16)
#endif
VECOPS_X86_HALF_BIN_256(sub,bfloat16_t,_mm256_sub_ps,half_cvt::cvt_bf16_to_two_fp32,half_cvt::cvt_two_fp32_to_bf16)

#endif // VEC_WIDTH >= 256

#if VEC_WIDTH >= 512
template <TLV_DECL_VEC(V), typename T = Vec2Tag<V>, TL_IF(T::Bytes == 64), TL_IF(is_any<TypeOf<T>, float32_t>)>
VECOPS_VFUNC V sub(V a, V b) {
  return _mm512_sub_ps(a.v, b.v);
}
template <TLV_DECL_VEC(V), typename T = Vec2Tag<V>, TL_IF(T::Bytes == 64), TL_IF(is_any<TypeOf<T>, float64_t>)>
VECOPS_VFUNC V sub(V a, V b) {
  return _mm512_sub_pd(a.v, b.v);
}
template <TLV_DECL_VEC(V), typename T = Vec2Tag<V>, TL_IF(T::Bytes == 64), TL_IF(is_any<TypeOf<T>, int8_t, uint8_t>)>
VECOPS_VFUNC V sub(V a, V b) {
  return _mm512_sub_epi8(a.v, b.v);
}
template <TLV_DECL_VEC(V), typename T = Vec2Tag<V>, TL_IF(T::Bytes == 64), TL_IF(is_any<TypeOf<T>, int16_t, uint16_t>)>
VECOPS_VFUNC V sub(V a, V b) {
  return _mm512_sub_epi16(a.v, b.v);
}
template <TLV_DECL_VEC(V), typename T = Vec2Tag<V>, TL_IF(T::Bytes == 64), TL_IF(is_any<TypeOf<T>, int32_t, uint32_t>)>
VECOPS_VFUNC V sub(V a, V b) {
  return _mm512_sub_epi32(a.v, b.v);
}
template <TLV_DECL_VEC(V), typename T = Vec2Tag<V>, TL_IF(T::Bytes == 64), TL_IF(is_any<TypeOf<T>, int64_t, uint64_t>)>
VECOPS_VFUNC V sub(V a, V b) {
  return _mm512_sub_epi64(a.v, b.v);
}
#ifdef HAS_AVX512_FP16
VECOPS_X86_FP16_NATIVE_BIN_512(sub,_mm512_sub_ph)
#else
VECOPS_X86_HALF_BIN_512(sub,float16_t,_mm512_sub_ps,half_cvt::cvt_fp16_to_two_fp32,half_cvt::cvt_two_fp32_to_fp16)
#endif
VECOPS_X86_HALF_BIN_512(sub,bfloat16_t,_mm512_sub_ps,half_cvt::cvt_bf16_to_two_fp32,half_cvt::cvt_two_fp32_to_bf16)
#endif // VEC_WIDTH >= 512

#ifdef HAS_AVX512DQ
template <TLV_DECL_VEC(V), typename T = Vec2Tag<V>, TL_IF(T::Bytes <= 16), TL_IF(is_any<TypeOf<T>, float32_t>)>
VECOPS_VFUNC V sub(V a, V b, Mask<T> m) {
  return _mm_mask_sub_ps(a.v, m.v, a.v, b.v);
}
template <TLV_DECL_VEC(V), typename T = Vec2Tag<V>, TL_IF(T::Bytes <= 16), TL_IF(is_any<TypeOf<T>, float64_t>)>
VECOPS_VFUNC V sub(V a, V b, Mask<T> m) {
  return _mm_mask_sub_pd(a.v, m.v, a.v, b.v);
}
template <TLV_DECL_VEC(V), typename T = Vec2Tag<V>, TL_IF(T::Bytes <= 16), TL_IF(is_any<TypeOf<T>, int8_t, uint8_t>)>
VECOPS_VFUNC V sub(V a, V b, Mask<T> m) {
  return _mm_mask_sub_epi8(a.v, m.v, a.v, b.v);
}
template <TLV_DECL_VEC(V), typename T = Vec2Tag<V>, TL_IF(T::Bytes <= 16), TL_IF(is_any<TypeOf<T>, int16_t, uint16_t>)>
VECOPS_VFUNC V sub(V a, V b, Mask<T> m) {
  return _mm_mask_sub_epi16(a.v, m.v, a.v, b.v);
}
template <TLV_DECL_VEC(V), typename T = Vec2Tag<V>, TL_IF(T::Bytes <= 16), TL_IF(is_any<TypeOf<T>, int32_t, uint32_t>)>
VECOPS_VFUNC V sub(V a, V b, Mask<T> m) {
  return _mm_mask_sub_epi32(a.v, m.v, a.v, b.v);
}
template <TLV_DECL_VEC(V), typename T = Vec2Tag<V>, TL_IF(T::Bytes <= 16), TL_IF(is_any<TypeOf<T>, int64_t, uint64_t>)>
VECOPS_VFUNC V sub(V a, V b, Mask<T> m) {
  return _mm_mask_sub_epi64(a.v, m.v, a.v, b.v);
}

template <TLV_DECL_VEC(V), typename T = Vec2Tag<V>, TL_IF(T::Bytes == 32), TL_IF(is_any<TypeOf<T>, float32_t>)>
VECOPS_VFUNC V sub(V a, V b, Mask<T> m) {
  return _mm256_mask_sub_ps(a.v, m.v, a.v, b.v);
}
template <TLV_DECL_VEC(V), typename T = Vec2Tag<V>, TL_IF(T::Bytes == 32), TL_IF(is_any<TypeOf<T>, float64_t>)>
VECOPS_VFUNC V sub(V a, V b, Mask<T> m) {
  return _mm256_mask_sub_pd(a.v, m.v, a.v, b.v);
}
template <TLV_DECL_VEC(V), typename T = Vec2Tag<V>, TL_IF(T::Bytes == 32), TL_IF(is_any<TypeOf<T>, int8_t, uint8_t>)>
VECOPS_VFUNC V sub(V a, V b, Mask<T> m) {
  return _mm256_mask_sub_epi8(a.v, m.v, a.v, b.v);
}
template <TLV_DECL_VEC(V), typename T = Vec2Tag<V>, TL_IF(T::Bytes == 32), TL_IF(is_any<TypeOf<T>, int16_t, uint16_t>)>
VECOPS_VFUNC V sub(V a, V b, Mask<T> m) {
  return _mm256_mask_sub_epi16(a.v, m.v, a.v, b.v);
}
template <TLV_DECL_VEC(V), typename T = Vec2Tag<V>, TL_IF(T::Bytes == 32), TL_IF(is_any<TypeOf<T>, int32_t, uint32_t>)>
VECOPS_VFUNC V sub(V a, V b, Mask<T> m) {
  return _mm256_mask_sub_epi32(a.v, m.v, a.v, b.v);
}
template <TLV_DECL_VEC(V), typename T = Vec2Tag<V>, TL_IF(T::Bytes == 32), TL_IF(is_any<TypeOf<T>, int64_t, uint64_t>)>
VECOPS_VFUNC V sub(V a, V b, Mask<T> m) {
  return _mm256_mask_sub_epi64(a.v, m.v, a.v, b.v);
}

template <TLV_DECL_VEC(V), typename T = Vec2Tag<V>, TL_IF(T::Bytes == 64), TL_IF(is_any<TypeOf<T>, float32_t>)>
VECOPS_VFUNC V sub(V a, V b, Mask<T> m) {
  return _mm512_mask_sub_ps(a.v, m.v, a.v, b.v);
}
template <TLV_DECL_VEC(V), typename T = Vec2Tag<V>, TL_IF(T::Bytes == 64), TL_IF(is_any<TypeOf<T>, float64_t>)>
VECOPS_VFUNC V sub(V a, V b, Mask<T> m) {
  return _mm512_mask_sub_pd(a.v, m.v, a.v, b.v);
}
template <TLV_DECL_VEC(V), typename T = Vec2Tag<V>, TL_IF(T::Bytes == 64), TL_IF(is_any<TypeOf<T>, int8_t, uint8_t>)>
VECOPS_VFUNC V sub(V a, V b, Mask<T> m) {
  return _mm512_mask_sub_epi8(a.v, m.v, a.v, b.v);
}
template <TLV_DECL_VEC(V), typename T = Vec2Tag<V>, TL_IF(T::Bytes == 64), TL_IF(is_any<TypeOf<T>, int16_t, uint16_t>)>
VECOPS_VFUNC V sub(V a, V b, Mask<T> m) {
  return _mm512_mask_sub_epi16(a.v, m.v, a.v, b.v);
}
template <TLV_DECL_VEC(V), typename T = Vec2Tag<V>, TL_IF(T::Bytes == 64), TL_IF(is_any<TypeOf<T>, int32_t, uint32_t>)>
VECOPS_VFUNC V sub(V a, V b, Mask<T> m) {
  return _mm512_mask_sub_epi32(a.v, m.v, a.v, b.v);
}
template <TLV_DECL_VEC(V), typename T = Vec2Tag<V>, TL_IF(T::Bytes == 64), TL_IF(is_any<TypeOf<T>, int64_t, uint64_t>)>
VECOPS_VFUNC V sub(V a, V b, Mask<T> m) {
  return _mm512_mask_sub_epi64(a.v, m.v, a.v, b.v);
}
template <TLV_DECL_VEC(V), typename T = Vec2Tag<V>, TL_IF(is_any<TypeOf<T>, float16_t, bfloat16_t>)>
VECOPS_VFUNC V sub(V a, V b, Mask<T> m) {
  return word::blend(a, m, word::sub(a, b));
}
#else // HAS_AVX512DQ
template <TLV_DECL_VEC(V), typename T = Vec2Tag<V>>
VECOPS_VFUNC V sub(V a, V b, Mask<T> m) {
  return word::blend(a, m, word::sub(a, b));
}
#endif // HAS_AVX512DQ


/* ************************************************************************** */
//                                   Mul                                      //
/* ************************************************************************** */
template <TLV_DECL_VEC(V), typename T = Vec2Tag<V>, TL_IF(T::Bytes <= 16), TL_IF(is_any<TypeOf<T>, float32_t>)>
VECOPS_VFUNC V mul(V a, V b) {
  return _mm_mul_ps(a.v, b.v);
}
template <TLV_DECL_VEC(V), typename T = Vec2Tag<V>, TL_IF(T::Bytes <= 16), TL_IF(is_any<TypeOf<T>, float64_t>)>
VECOPS_VFUNC V mul(V a, V b) {
  return _mm_mul_pd(a.v, b.v);
}
template <TLV_DECL_VEC(V), typename T = Vec2Tag<V>, TL_IF(T::Bytes <= 16), TL_IF(is_any<TypeOf<T>, int8_t, uint8_t>)>
VECOPS_VFUNC V mul(V a, V b) {
  auto even = _mm_mullo_epi16(a.v, b.v);
  auto odd = _mm_mullo_epi16(_mm_srli_epi16(a.v, 8), _mm_srli_epi16(b.v, 8));
  return _mm_or_si128(_mm_slli_epi16(odd, 8), _mm_and_si128(even, _mm_set1_epi16(0xFF)));
}
template <TLV_DECL_VEC(V), typename T = Vec2Tag<V>, TL_IF(T::Bytes <= 16), TL_IF(is_any<TypeOf<T>, int16_t, uint16_t>)>
VECOPS_VFUNC V mul(V a, V b) {
  return _mm_mullo_epi16(a.v, b.v);
}
template <TLV_DECL_VEC(V), typename T = Vec2Tag<V>, TL_IF(T::Bytes <= 16), TL_IF(is_any<TypeOf<T>, int32_t, uint32_t>)>
VECOPS_VFUNC V mul(V a, V b) {
  return _mm_mullo_epi32(a.v, b.v);
}
template <TLV_DECL_VEC(V), typename T = Vec2Tag<V>, TL_IF(T::Bytes <= 16), TL_IF(is_any<TypeOf<T>, int64_t, uint64_t>)>
VECOPS_VFUNC V mul(V a, V b) {
  #ifdef HAS_AVX512DQ
  return _mm_mullo_epi64(a.v, b.v);
  #else
  ViewAs<int32_t, T> t32;
  auto lo_lo = _mm_mul_epu32(a.v, b.v);
  auto a_hi = word::local_shuf<3, 3, 1, 1>(word::bitcast(t32, a));
  auto b_hi = word::local_shuf<3, 3, 1, 1>(word::bitcast(t32, b));
  auto hi_lo = _mm_mul_epu32(a_hi.v, b.v); // a_hi × b_lo
  auto lo_hi = _mm_mul_epu32(a.v, b_hi.v); // a_lo × b_hi
  auto cross = _mm_add_epi64(hi_lo, lo_hi);
  cross = _mm_slli_epi64(cross, 32);
  return _mm_add_epi64(lo_lo, cross);
  #endif
}
#ifdef HAS_AVX512_FP16
VECOPS_X86_FP16_NATIVE_BIN_128(mul,_mm_mul_ph)
#else
VECOPS_X86_HALF_BIN_128(mul,float16_t,_mm_mul_ps,half_cvt::cvt_fp16_to_two_fp32,half_cvt::cvt_two_fp32_to_fp16)
#endif
VECOPS_X86_HALF_BIN_128(mul,bfloat16_t,_mm_mul_ps,half_cvt::cvt_bf16_to_two_fp32,half_cvt::cvt_two_fp32_to_bf16)

#if VEC_WIDTH >= 256
template <TLV_DECL_VEC(V), typename T = Vec2Tag<V>, TL_IF(T::Bytes == 32), TL_IF(is_any<TypeOf<T>, float32_t>)>
VECOPS_VFUNC V mul(V a, V b) {
  return _mm256_mul_ps(a.v, b.v);
}
template <TLV_DECL_VEC(V), typename T = Vec2Tag<V>, TL_IF(T::Bytes == 32), TL_IF(is_any<TypeOf<T>, float64_t>)>
VECOPS_VFUNC V mul(V a, V b) {
  return _mm256_mul_pd(a.v, b.v);
}
template <TLV_DECL_VEC(V), typename T = Vec2Tag<V>, TL_IF(T::Bytes == 32), TL_IF(is_any<TypeOf<T>, int8_t, uint8_t>)>
VECOPS_VFUNC V mul(V a, V b) {
  auto even = _mm256_mullo_epi16(a.v, b.v);
  auto odd = _mm256_mullo_epi16(_mm256_srli_epi16(a.v, 8), _mm256_srli_epi16(b.v, 8));
  return _mm256_or_si256(_mm256_slli_epi16(odd, 8), _mm256_and_si256(even, _mm256_set1_epi16(0xFF)));
}
template <TLV_DECL_VEC(V), typename T = Vec2Tag<V>, TL_IF(T::Bytes == 32), TL_IF(is_any<TypeOf<T>, int16_t, uint16_t>)>
VECOPS_VFUNC V mul(V a, V b) {
  return _mm256_mullo_epi16(a.v, b.v);
}
template <TLV_DECL_VEC(V), typename T = Vec2Tag<V>, TL_IF(T::Bytes == 32), TL_IF(is_any<TypeOf<T>, int32_t, uint32_t>)>
VECOPS_VFUNC V mul(V a, V b) {
  return _mm256_mullo_epi32(a.v, b.v);
}
template <TLV_DECL_VEC(V), typename T = Vec2Tag<V>, TL_IF(T::Bytes == 32), TL_IF(is_any<TypeOf<T>, int64_t, uint64_t>)>
VECOPS_VFUNC V mul(V a, V b) {
  #ifdef HAS_AVX512DQ
  return _mm256_mullo_epi64(a.v, b.v);
  #else
  auto lo_lo = _mm256_mul_epu32(a.v, b.v);
  ViewAs<int32_t, T> t32;
  auto a_hi = word::bitcast(T(), word::local_shuf<3, 3, 1, 1>(word::bitcast(t32, a)));
  auto b_hi = word::bitcast(T(), word::local_shuf<3, 3, 1, 1>(word::bitcast(t32, b)));
  auto hi_lo = _mm256_mul_epu32(a_hi.v, b.v); // a_hi × b_lo
  auto lo_hi = _mm256_mul_epu32(a.v, b_hi.v); // a_lo × b_hi
  auto cross = _mm256_add_epi64(hi_lo, lo_hi);
  cross = _mm256_slli_epi64(cross, 32);
  return _mm256_add_epi64(lo_lo, cross);
  #endif
}
#ifdef HAS_AVX512_FP16
VECOPS_X86_FP16_NATIVE_BIN_256(mul,_mm256_mul_ph)
#else
VECOPS_X86_HALF_BIN_256(mul,float16_t,_mm256_mul_ps,half_cvt::cvt_fp16_to_two_fp32,half_cvt::cvt_two_fp32_to_fp16)
#endif
VECOPS_X86_HALF_BIN_256(mul,bfloat16_t,_mm256_mul_ps,half_cvt::cvt_bf16_to_two_fp32,half_cvt::cvt_two_fp32_to_bf16)

#endif // VEC_WIDTH >= 256

#if VEC_WIDTH >= 512
template <TLV_DECL_VEC(V), typename T = Vec2Tag<V>, TL_IF(T::Bytes == 64), TL_IF(is_any<TypeOf<T>, float32_t>)>
VECOPS_VFUNC V mul(V a, V b) {
  return _mm512_mul_ps(a.v, b.v);
}
template <TLV_DECL_VEC(V), typename T = Vec2Tag<V>, TL_IF(T::Bytes == 64), TL_IF(is_any<TypeOf<T>, float64_t>)>
VECOPS_VFUNC V mul(V a, V b) {
  return _mm512_mul_pd(a.v, b.v);
}
template <TLV_DECL_VEC(V), typename T = Vec2Tag<V>, TL_IF(T::Bytes == 64), TL_IF(is_any<TypeOf<T>, int8_t, uint8_t>)>
VECOPS_VFUNC V mul(V a, V b) {
  auto even = _mm512_mullo_epi16(a.v, b.v);
  auto odd = _mm512_mullo_epi16(_mm512_srli_epi16(a.v, 8), _mm512_srli_epi16(b.v, 8));
  return _mm512_or_si512(_mm512_slli_epi16(odd, 8), _mm512_and_si512(even, _mm512_set1_epi16(0xFF)));
}
template <TLV_DECL_VEC(V), typename T = Vec2Tag<V>, TL_IF(T::Bytes == 64), TL_IF(is_any<TypeOf<T>, int16_t, uint16_t>)>
VECOPS_VFUNC V mul(V a, V b) {
  return _mm512_mullo_epi16(a.v, b.v);
}
template <TLV_DECL_VEC(V), typename T = Vec2Tag<V>, TL_IF(T::Bytes == 64), TL_IF(is_any<TypeOf<T>, int32_t, uint32_t>)>
VECOPS_VFUNC V mul(V a, V b) {
  return _mm512_mullo_epi32(a.v, b.v);
}
template <TLV_DECL_VEC(V), typename T = Vec2Tag<V>, TL_IF(T::Bytes == 64), TL_IF(is_any<TypeOf<T>, int64_t, uint64_t>)>
VECOPS_VFUNC V mul(V a, V b) {
  return _mm512_mullo_epi64(a.v, b.v);
}
#ifdef HAS_AVX512_FP16
VECOPS_X86_FP16_NATIVE_BIN_512(mul,_mm512_mul_ph)
#else
VECOPS_X86_HALF_BIN_512(mul,float16_t,_mm512_mul_ps,half_cvt::cvt_fp16_to_two_fp32,half_cvt::cvt_two_fp32_to_fp16)
#endif
VECOPS_X86_HALF_BIN_512(mul,bfloat16_t,_mm512_mul_ps,half_cvt::cvt_bf16_to_two_fp32,half_cvt::cvt_two_fp32_to_bf16)
#endif // VEC_WIDTH >= 512

#ifdef HAS_AVX512DQ
template <TLV_DECL_VEC(V), typename T = Vec2Tag<V>, TL_IF(T::Bytes <= 16), TL_IF(is_any<TypeOf<T>, float32_t>)>
VECOPS_VFUNC V mul(V a, V b, Mask<T> m) {
  return _mm_mask_mul_ps(a.v, m.v, a.v, b.v);
}
template <TLV_DECL_VEC(V), typename T = Vec2Tag<V>, TL_IF(T::Bytes <= 16), TL_IF(is_any<TypeOf<T>, float64_t>)>
VECOPS_VFUNC V mul(V a, V b, Mask<T> m) {
  return _mm_mask_mul_pd(a.v, m.v, a.v, b.v);
}
template <TLV_DECL_VEC(V), typename T = Vec2Tag<V>, TL_IF(is_any<TypeOf<T>, int8_t, uint8_t>)>
VECOPS_VFUNC V mul(V a, V b, Mask<T> m) {
  return word::blend(a, m, word::mul(a, b));
}
template <TLV_DECL_VEC(V), typename T = Vec2Tag<V>, TL_IF(T::Bytes <= 16), TL_IF(is_any<TypeOf<T>, int16_t, uint16_t>)>
VECOPS_VFUNC V mul(V a, V b, Mask<T> m) {
  return _mm_mask_mullo_epi16(a.v, m.v, a.v, b.v);
}
template <TLV_DECL_VEC(V), typename T = Vec2Tag<V>, TL_IF(T::Bytes <= 16), TL_IF(is_any<TypeOf<T>, int32_t, uint32_t>)>
VECOPS_VFUNC V mul(V a, V b, Mask<T> m) {
  return _mm_mask_mullo_epi32(a.v, m.v, a.v, b.v);
}
template <TLV_DECL_VEC(V), typename T = Vec2Tag<V>, TL_IF(T::Bytes <= 16), TL_IF(is_any<TypeOf<T>, int64_t, uint64_t>)>
VECOPS_VFUNC V mul(V a, V b, Mask<T> m) {
  return _mm_mask_mullo_epi64(a.v, m.v, a.v, b.v);
}

template <TLV_DECL_VEC(V), typename T = Vec2Tag<V>, TL_IF(T::Bytes == 32), TL_IF(is_any<TypeOf<T>, float32_t>)>
VECOPS_VFUNC V mul(V a, V b, Mask<T> m) {
  return _mm256_mask_mul_ps(a.v, m.v, a.v, b.v);
}
template <TLV_DECL_VEC(V), typename T = Vec2Tag<V>, TL_IF(T::Bytes == 32), TL_IF(is_any<TypeOf<T>, float64_t>)>
VECOPS_VFUNC V mul(V a, V b, Mask<T> m) {
  return _mm256_mask_mul_pd(a.v, m.v, a.v, b.v);
}
template <TLV_DECL_VEC(V), typename T = Vec2Tag<V>, TL_IF(T::Bytes == 32), TL_IF(is_any<TypeOf<T>, int16_t, uint16_t>)>
VECOPS_VFUNC V mul(V a, V b, Mask<T> m) {
  return _mm256_mask_mullo_epi16(a.v, m.v, a.v, b.v);
}
template <TLV_DECL_VEC(V), typename T = Vec2Tag<V>, TL_IF(T::Bytes == 32), TL_IF(is_any<TypeOf<T>, int32_t, uint32_t>)>
VECOPS_VFUNC V mul(V a, V b, Mask<T> m) {
  return _mm256_mask_mullo_epi32(a.v, m.v, a.v, b.v);
}
template <TLV_DECL_VEC(V), typename T = Vec2Tag<V>, TL_IF(T::Bytes == 32), TL_IF(is_any<TypeOf<T>, int64_t, uint64_t>)>
VECOPS_VFUNC V mul(V a, V b, Mask<T> m) {
  return _mm256_mask_mullo_epi64(a.v, m.v, a.v, b.v);
}

template <TLV_DECL_VEC(V), typename T = Vec2Tag<V>, TL_IF(T::Bytes == 64), TL_IF(is_any<TypeOf<T>, float32_t>)>
VECOPS_VFUNC V mul(V a, V b, Mask<T> m) {
  return _mm512_mask_mul_ps(a.v, m.v, a.v, b.v);
}
template <TLV_DECL_VEC(V), typename T = Vec2Tag<V>, TL_IF(T::Bytes == 64), TL_IF(is_any<TypeOf<T>, float64_t>)>
VECOPS_VFUNC V mul(V a, V b, Mask<T> m) {
  return _mm512_mask_mul_pd(a.v, m.v, a.v, b.v);
}
template <TLV_DECL_VEC(V), typename T = Vec2Tag<V>, TL_IF(T::Bytes == 64), TL_IF(is_any<TypeOf<T>, int16_t, uint16_t>)>
VECOPS_VFUNC V mul(V a, V b, Mask<T> m) {
  return _mm512_mask_mullo_epi16(a.v, m.v, a.v, b.v);
}
template <TLV_DECL_VEC(V), typename T = Vec2Tag<V>, TL_IF(T::Bytes == 64), TL_IF(is_any<TypeOf<T>, int32_t, uint32_t>)>
VECOPS_VFUNC V mul(V a, V b, Mask<T> m) {
  return _mm512_mask_mullo_epi32(a.v, m.v, a.v, b.v);
}
template <TLV_DECL_VEC(V), typename T = Vec2Tag<V>, TL_IF(T::Bytes == 64), TL_IF(is_any<TypeOf<T>, int64_t, uint64_t>)>
VECOPS_VFUNC V mul(V a, V b, Mask<T> m) {
  return _mm512_mask_mullo_epi64(a.v, m.v, a.v, b.v);
}
template <TLV_DECL_VEC(V), typename T = Vec2Tag<V>, TL_IF(is_any<TypeOf<T>, float16_t, bfloat16_t>)>
VECOPS_VFUNC V mul(V a, V b, Mask<T> m) {
  return word::blend(a, m, word::mul(a, b));
}
#else // HAS_AVX512DQ
template <TLV_DECL_VEC(V), typename T = Vec2Tag<V>>
VECOPS_VFUNC V mul(V a, V b, Mask<T> m) {
  return word::blend(a, m, word::mul(a, b));
}
#endif // HAS_AVX512DQ


/* ************************************************************************** */
//                                   FMA                                      //
/* ************************************************************************** */
template <TLV_DECL_VEC(V), typename T = Vec2Tag<V>>
VECOPS_VFUNC V fmadd(V a, V b, V c) {
  using E = TypeOf<T>;
  #ifdef HAS_FMA
  if constexpr (std::is_same_v<E, float32_t>) {
    if constexpr (T::Bytes <= 16) return _mm_fmadd_ps(a.v, b.v, c.v);
    #if VEC_WIDTH >= 256
    else if constexpr (T::Bytes == 32) return _mm256_fmadd_ps(a.v, b.v, c.v);
    #endif
    #if VEC_WIDTH >= 512
    else if constexpr (T::Bytes == 64) return _mm512_fmadd_ps(a.v, b.v, c.v);
    #endif
  } else if constexpr (std::is_same_v<E, float64_t>) {
    if constexpr (T::Bytes <= 16) return _mm_fmadd_pd(a.v, b.v, c.v);
    #if VEC_WIDTH >= 256
    else if constexpr (T::Bytes == 32) return _mm256_fmadd_pd(a.v, b.v, c.v);
    #endif
    #if VEC_WIDTH >= 512
    else if constexpr (T::Bytes == 64) return _mm512_fmadd_pd(a.v, b.v, c.v);
    #endif
  }
  #endif
  #ifdef HAS_AVX512_FP16
  if constexpr (std::is_same_v<E, float16_t>) {
    if constexpr (T::Bytes <= 16) return _mm_castph_si128(_mm_fmadd_ph(_mm_castsi128_ph(a.v), _mm_castsi128_ph(b.v), _mm_castsi128_ph(c.v)));
    #if VEC_WIDTH >= 256
    else if constexpr (T::Bytes == 32) return _mm256_castph_si256(_mm256_fmadd_ph(_mm256_castsi256_ph(a.v), _mm256_castsi256_ph(b.v), _mm256_castsi256_ph(c.v)));
    #endif
    #if VEC_WIDTH >= 512
    else if constexpr (T::Bytes == 64) return _mm512_castph_si512(_mm512_fmadd_ph(_mm512_castsi512_ph(a.v), _mm512_castsi512_ph(b.v), _mm512_castsi512_ph(c.v)));
    #endif
  }
  #endif
  return word::add(word::mul(a, b), c);
}

template <TLV_DECL_VEC(V), typename T = Vec2Tag<V>>
VECOPS_VFUNC V fmadd(V a, V b, V c, Mask<T> m) {
  return word::blend(a, m, word::fmadd(a, b, c));
}

template <TLV_DECL_VEC(V), typename T = Vec2Tag<V>>
VECOPS_VFUNC V fmsub(V a, V b, V c) {
  using E = TypeOf<T>;
  #ifdef HAS_FMA
  if constexpr (std::is_same_v<E, float32_t>) {
    if constexpr (T::Bytes <= 16) return _mm_fmsub_ps(a.v, b.v, c.v);
    #if VEC_WIDTH >= 256
    else if constexpr (T::Bytes == 32) return _mm256_fmsub_ps(a.v, b.v, c.v);
    #endif
    #if VEC_WIDTH >= 512
    else if constexpr (T::Bytes == 64) return _mm512_fmsub_ps(a.v, b.v, c.v);
    #endif
  } else if constexpr (std::is_same_v<E, float64_t>) {
    if constexpr (T::Bytes <= 16) return _mm_fmsub_pd(a.v, b.v, c.v);
    #if VEC_WIDTH >= 256
    else if constexpr (T::Bytes == 32) return _mm256_fmsub_pd(a.v, b.v, c.v);
    #endif
    #if VEC_WIDTH >= 512
    else if constexpr (T::Bytes == 64) return _mm512_fmsub_pd(a.v, b.v, c.v);
    #endif
  }
  #endif
  #ifdef HAS_AVX512_FP16
  if constexpr (std::is_same_v<E, float16_t>) {
    if constexpr (T::Bytes <= 16) return _mm_castph_si128(_mm_fmsub_ph(_mm_castsi128_ph(a.v), _mm_castsi128_ph(b.v), _mm_castsi128_ph(c.v)));
    #if VEC_WIDTH >= 256
    else if constexpr (T::Bytes == 32) return _mm256_castph_si256(_mm256_fmsub_ph(_mm256_castsi256_ph(a.v), _mm256_castsi256_ph(b.v), _mm256_castsi256_ph(c.v)));
    #endif
    #if VEC_WIDTH >= 512
    else if constexpr (T::Bytes == 64) return _mm512_castph_si512(_mm512_fmsub_ph(_mm512_castsi512_ph(a.v), _mm512_castsi512_ph(b.v), _mm512_castsi512_ph(c.v)));
    #endif
  }
  #endif
  return word::sub(word::mul(a, b), c);
}

template <TLV_DECL_VEC(V), typename T = Vec2Tag<V>>
VECOPS_VFUNC V fmsub(V a, V b, V c, Mask<T> m) {
  return word::blend(a, m, word::fmsub(a, b, c));
}

template <TLV_DECL_VEC(V), typename T = Vec2Tag<V>>
VECOPS_VFUNC V fnmadd(V a, V b, V c) {
  using E = TypeOf<T>;
  #ifdef HAS_FMA
  if constexpr (std::is_same_v<E, float32_t>) {
    if constexpr (T::Bytes <= 16) return _mm_fnmadd_ps(a.v, b.v, c.v);
    #if VEC_WIDTH >= 256
    else if constexpr (T::Bytes == 32) return _mm256_fnmadd_ps(a.v, b.v, c.v);
    #endif
    #if VEC_WIDTH >= 512
    else if constexpr (T::Bytes == 64) return _mm512_fnmadd_ps(a.v, b.v, c.v);
    #endif
  } else if constexpr (std::is_same_v<E, float64_t>) {
    if constexpr (T::Bytes <= 16) return _mm_fnmadd_pd(a.v, b.v, c.v);
    #if VEC_WIDTH >= 256
    else if constexpr (T::Bytes == 32) return _mm256_fnmadd_pd(a.v, b.v, c.v);
    #endif
    #if VEC_WIDTH >= 512
    else if constexpr (T::Bytes == 64) return _mm512_fnmadd_pd(a.v, b.v, c.v);
    #endif
  }
  #endif
  #ifdef HAS_AVX512_FP16
  if constexpr (std::is_same_v<E, float16_t>) {
    if constexpr (T::Bytes <= 16) return _mm_castph_si128(_mm_fnmadd_ph(_mm_castsi128_ph(a.v), _mm_castsi128_ph(b.v), _mm_castsi128_ph(c.v)));
    #if VEC_WIDTH >= 256
    else if constexpr (T::Bytes == 32) return _mm256_castph_si256(_mm256_fnmadd_ph(_mm256_castsi256_ph(a.v), _mm256_castsi256_ph(b.v), _mm256_castsi256_ph(c.v)));
    #endif
    #if VEC_WIDTH >= 512
    else if constexpr (T::Bytes == 64) return _mm512_castph_si512(_mm512_fnmadd_ph(_mm512_castsi512_ph(a.v), _mm512_castsi512_ph(b.v), _mm512_castsi512_ph(c.v)));
    #endif
  }
  #endif
  return word::sub(c, word::mul(a, b));
}

template <TLV_DECL_VEC(V), typename T = Vec2Tag<V>>
VECOPS_VFUNC V fnmadd(V a, V b, V c, Mask<T> m) {
  return word::blend(a, m, word::fnmadd(a, b, c));
}

template <TLV_DECL_VEC(V), typename T = Vec2Tag<V>>
VECOPS_VFUNC V fnmsub(V a, V b, V c) {
  using E = TypeOf<T>;
  #ifdef HAS_FMA
  if constexpr (std::is_same_v<E, float32_t>) {
    if constexpr (T::Bytes <= 16) return _mm_fnmsub_ps(a.v, b.v, c.v);
    #if VEC_WIDTH >= 256
    else if constexpr (T::Bytes == 32) return _mm256_fnmsub_ps(a.v, b.v, c.v);
    #endif
    #if VEC_WIDTH >= 512
    else if constexpr (T::Bytes == 64) return _mm512_fnmsub_ps(a.v, b.v, c.v);
    #endif
  } else if constexpr (std::is_same_v<E, float64_t>) {
    if constexpr (T::Bytes <= 16) return _mm_fnmsub_pd(a.v, b.v, c.v);
    #if VEC_WIDTH >= 256
    else if constexpr (T::Bytes == 32) return _mm256_fnmsub_pd(a.v, b.v, c.v);
    #endif
    #if VEC_WIDTH >= 512
    else if constexpr (T::Bytes == 64) return _mm512_fnmsub_pd(a.v, b.v, c.v);
    #endif
  }
  #endif
  #ifdef HAS_AVX512_FP16
  if constexpr (std::is_same_v<E, float16_t>) {
    if constexpr (T::Bytes <= 16) return _mm_castph_si128(_mm_fnmsub_ph(_mm_castsi128_ph(a.v), _mm_castsi128_ph(b.v), _mm_castsi128_ph(c.v)));
    #if VEC_WIDTH >= 256
    else if constexpr (T::Bytes == 32) return _mm256_castph_si256(_mm256_fnmsub_ph(_mm256_castsi256_ph(a.v), _mm256_castsi256_ph(b.v), _mm256_castsi256_ph(c.v)));
    #endif
    #if VEC_WIDTH >= 512
    else if constexpr (T::Bytes == 64) return _mm512_castph_si512(_mm512_fnmsub_ph(_mm512_castsi512_ph(a.v), _mm512_castsi512_ph(b.v), _mm512_castsi512_ph(c.v)));
    #endif
  }
  #endif
  return word::sub(word::sub(a, a), word::add(word::mul(a, b), c));
}

template <TLV_DECL_VEC(V), typename T = Vec2Tag<V>>
VECOPS_VFUNC V fnmsub(V a, V b, V c, Mask<T> m) {
  return word::blend(a, m, word::fnmsub(a, b, c));
}


/* ************************************************************************** */
//                                   Div                                      //
/* ************************************************************************** */
template <TLV_DECL_VEC(V), typename T = Vec2Tag<V>, TL_IF(T::Bytes <= 16), TL_IF(is_any<TypeOf<T>, float32_t>)>
VECOPS_VFUNC V div(V a, V b) {
  return _mm_div_ps(a.v, b.v);
}
template <TLV_DECL_VEC(V), typename T = Vec2Tag<V>, TL_IF(T::Bytes <= 16), TL_IF(is_any<TypeOf<T>, float64_t>)>
VECOPS_VFUNC V div(V a, V b) {
  return _mm_div_pd(a.v, b.v);
}
#ifdef HAS_AVX512_FP16
VECOPS_X86_FP16_NATIVE_BIN_128(div,_mm_div_ph)
#else
VECOPS_X86_HALF_BIN_128(div,float16_t,_mm_div_ps,half_cvt::cvt_fp16_to_two_fp32,half_cvt::cvt_two_fp32_to_fp16)
#endif
VECOPS_X86_HALF_BIN_128(div,bfloat16_t,_mm_div_ps,half_cvt::cvt_bf16_to_two_fp32,half_cvt::cvt_two_fp32_to_bf16)

#if VEC_WIDTH >= 256
template <TLV_DECL_VEC(V), typename T = Vec2Tag<V>, TL_IF(T::Bytes == 32), TL_IF(is_any<TypeOf<T>, float32_t>)>
VECOPS_VFUNC V div(V a, V b) {
  return _mm256_div_ps(a.v, b.v);
}
template <TLV_DECL_VEC(V), typename T = Vec2Tag<V>, TL_IF(T::Bytes == 32), TL_IF(is_any<TypeOf<T>, float64_t>)>
VECOPS_VFUNC V div(V a, V b) {
  return _mm256_div_pd(a.v, b.v);
}
#ifdef HAS_AVX512_FP16
VECOPS_X86_FP16_NATIVE_BIN_256(div,_mm256_div_ph)
#else
VECOPS_X86_HALF_BIN_256(div,float16_t,_mm256_div_ps,half_cvt::cvt_fp16_to_two_fp32,half_cvt::cvt_two_fp32_to_fp16)
#endif
VECOPS_X86_HALF_BIN_256(div,bfloat16_t,_mm256_div_ps,half_cvt::cvt_bf16_to_two_fp32,half_cvt::cvt_two_fp32_to_bf16)

#endif // VEC_WIDTH >= 256

#if VEC_WIDTH >= 512
template <TLV_DECL_VEC(V), typename T = Vec2Tag<V>, TL_IF(T::Bytes == 64), TL_IF(is_any<TypeOf<T>, float32_t>)>
VECOPS_VFUNC V div(V a, V b) {
  return _mm512_div_ps(a.v, b.v);
}
template <TLV_DECL_VEC(V), typename T = Vec2Tag<V>, TL_IF(T::Bytes == 64), TL_IF(is_any<TypeOf<T>, float64_t>)>
VECOPS_VFUNC V div(V a, V b) {
  return _mm512_div_pd(a.v, b.v);
}
#ifdef HAS_AVX512_FP16
VECOPS_X86_FP16_NATIVE_BIN_512(div,_mm512_div_ph)
#else
VECOPS_X86_HALF_BIN_512(div,float16_t,_mm512_div_ps,half_cvt::cvt_fp16_to_two_fp32,half_cvt::cvt_two_fp32_to_fp16)
#endif
VECOPS_X86_HALF_BIN_512(div,bfloat16_t,_mm512_div_ps,half_cvt::cvt_bf16_to_two_fp32,half_cvt::cvt_two_fp32_to_bf16)
#endif // VEC_WIDTH >= 512

#ifdef HAS_AVX512DQ
template <TLV_DECL_VEC(V), typename T = Vec2Tag<V>, TL_IF(T::Bytes <= 16), TL_IF(is_any<TypeOf<T>, float32_t>)>
VECOPS_VFUNC V div(V a, V b, Mask<T> m) {
  return _mm_mask_div_ps(a.v, m.v, a.v, b.v);
}
template <TLV_DECL_VEC(V), typename T = Vec2Tag<V>, TL_IF(T::Bytes <= 16), TL_IF(is_any<TypeOf<T>, float64_t>)>
VECOPS_VFUNC V div(V a, V b, Mask<T> m) {
  return _mm_mask_div_pd(a.v, m.v, a.v, b.v);
}

template <TLV_DECL_VEC(V), typename T = Vec2Tag<V>, TL_IF(T::Bytes == 32), TL_IF(is_any<TypeOf<T>, float32_t>)>
VECOPS_VFUNC V div(V a, V b, Mask<T> m) {
  return _mm256_mask_div_ps(a.v, m.v, a.v, b.v);
}
template <TLV_DECL_VEC(V), typename T = Vec2Tag<V>, TL_IF(T::Bytes == 32), TL_IF(is_any<TypeOf<T>, float64_t>)>
VECOPS_VFUNC V div(V a, V b, Mask<T> m) {
  return _mm256_mask_div_pd(a.v, m.v, a.v, b.v);
}

template <TLV_DECL_VEC(V), typename T = Vec2Tag<V>, TL_IF(T::Bytes == 64), TL_IF(is_any<TypeOf<T>, float32_t>)>
VECOPS_VFUNC V div(V a, V b, Mask<T> m) {
  return _mm512_mask_div_ps(a.v, m.v, a.v, b.v);
}
template <TLV_DECL_VEC(V), typename T = Vec2Tag<V>, TL_IF(T::Bytes == 64), TL_IF(is_any<TypeOf<T>, float64_t>)>
VECOPS_VFUNC V div(V a, V b, Mask<T> m) {
  return _mm512_mask_div_pd(a.v, m.v, a.v, b.v);
}
template <TLV_DECL_VEC(V), typename T = Vec2Tag<V>, TL_IF(is_any<TypeOf<T>, float16_t, bfloat16_t>)>
VECOPS_VFUNC V div(V a, V b, Mask<T> m) {
  return word::blend(a, m, word::div(a, b));
}
#else // HAS_AVX512DQ
template <TLV_DECL_VEC(V), typename T = Vec2Tag<V>>
VECOPS_VFUNC V div(V a, V b, Mask<T> m) {
  return word::blend(a, m, word::div(a, b));
}
#endif // HAS_AVX512DQ


/* ************************************************************************** */
//                                   Min                                      //
/* ************************************************************************** */
template <TLV_DECL_VEC(V), typename T = Vec2Tag<V>, TL_IF(T::Bytes <= 16), TL_IF(is_any<TypeOf<T>, float32_t>)>
VECOPS_VFUNC V min(V a, V b) {
  return _mm_min_ps(a.v, b.v);
}
template <TLV_DECL_VEC(V), typename T = Vec2Tag<V>, TL_IF(T::Bytes <= 16), TL_IF(is_any<TypeOf<T>, float64_t>)>
VECOPS_VFUNC V min(V a, V b) {
  return _mm_min_pd(a.v, b.v);
}
template <TLV_DECL_VEC(V), typename T = Vec2Tag<V>, TL_IF(T::Bytes <= 16), TL_IF(is_any<TypeOf<T>, int8_t>)>
VECOPS_VFUNC V min(V a, V b) {
  return _mm_min_epi8(a.v, b.v);
}
template <TLV_DECL_VEC(V), typename T = Vec2Tag<V>, TL_IF(T::Bytes <= 16), TL_IF(is_any<TypeOf<T>, uint8_t>)>
VECOPS_VFUNC V min(V a, V b) {
  return _mm_min_epu8(a.v, b.v);
}
template <TLV_DECL_VEC(V), typename T = Vec2Tag<V>, TL_IF(T::Bytes <= 16), TL_IF(is_any<TypeOf<T>, int16_t>)>
VECOPS_VFUNC V min(V a, V b) {
  return _mm_min_epi16(a.v, b.v);
}
template <TLV_DECL_VEC(V), typename T = Vec2Tag<V>, TL_IF(T::Bytes <= 16), TL_IF(is_any<TypeOf<T>, uint16_t>)>
VECOPS_VFUNC V min(V a, V b) {
  return _mm_min_epu16(a.v, b.v);
}
template <TLV_DECL_VEC(V), typename T = Vec2Tag<V>, TL_IF(T::Bytes <= 16), TL_IF(is_any<TypeOf<T>, int32_t>)>
VECOPS_VFUNC V min(V a, V b) {
  return _mm_min_epi32(a.v, b.v);
}
template <TLV_DECL_VEC(V), typename T = Vec2Tag<V>, TL_IF(T::Bytes <= 16), TL_IF(is_any<TypeOf<T>, uint32_t>)>
VECOPS_VFUNC V min(V a, V b) {
  return _mm_min_epu32(a.v, b.v);
}
template <TLV_DECL_VEC(V), typename T = Vec2Tag<V>, TL_IF(T::Bytes <= 16), TL_IF(is_any<TypeOf<T>, int64_t>)>
VECOPS_VFUNC V min(V a, V b) {
  #ifdef HAS_AVX512F
  return _mm_min_epi64(a.v, b.v);
  #else
  return _mm_blendv_epi8(b.v, a.v, _mm_cmpgt_epi64(b.v, a.v));
  #endif
}
template <TLV_DECL_VEC(V), typename T = Vec2Tag<V>, TL_IF(T::Bytes <= 16), TL_IF(is_any<TypeOf<T>, uint64_t>)>
VECOPS_VFUNC V min(V a, V b) {
  #ifdef HAS_AVX512F
  return _mm_min_epu64(a.v, b.v);
  #else
  static const __m128i flip = _mm_set1_epi64x((int64_t)0x8000000000000000LL);
  auto a_flip = _mm_xor_si128(a.v, flip);
  auto b_flip = _mm_xor_si128(b.v, flip);
  return _mm_blendv_epi8(b.v, a.v, _mm_cmpgt_epi64(b_flip, a_flip));
  #endif
}
#ifdef HAS_AVX512_FP16
VECOPS_X86_FP16_NATIVE_BIN_128(min,_mm_min_ph)
#else
VECOPS_X86_HALF_BIN_128(min,float16_t,_mm_min_ps,half_cvt::cvt_fp16_to_two_fp32,half_cvt::cvt_two_fp32_to_fp16)
#endif
VECOPS_X86_HALF_BIN_128(min,bfloat16_t,_mm_min_ps,half_cvt::cvt_bf16_to_two_fp32,half_cvt::cvt_two_fp32_to_bf16)

#if VEC_WIDTH >= 256
template <TLV_DECL_VEC(V), typename T = Vec2Tag<V>, TL_IF(T::Bytes == 32), TL_IF(is_any<TypeOf<T>, float32_t>)>
VECOPS_VFUNC V min(V a, V b) {
  return _mm256_min_ps(a.v, b.v);
}
template <TLV_DECL_VEC(V), typename T = Vec2Tag<V>, TL_IF(T::Bytes == 32), TL_IF(is_any<TypeOf<T>, float64_t>)>
VECOPS_VFUNC V min(V a, V b) {
  return _mm256_min_pd(a.v, b.v);
}
template <TLV_DECL_VEC(V), typename T = Vec2Tag<V>, TL_IF(T::Bytes == 32), TL_IF(is_any<TypeOf<T>, int8_t>)>
VECOPS_VFUNC V min(V a, V b) {
  return _mm256_min_epi8(a.v, b.v);
}
template <TLV_DECL_VEC(V), typename T = Vec2Tag<V>, TL_IF(T::Bytes == 32), TL_IF(is_any<TypeOf<T>, uint8_t>)>
VECOPS_VFUNC V min(V a, V b) {
  return _mm256_min_epu8(a.v, b.v);
}
template <TLV_DECL_VEC(V), typename T = Vec2Tag<V>, TL_IF(T::Bytes == 32), TL_IF(is_any<TypeOf<T>, int16_t>)>
VECOPS_VFUNC V min(V a, V b) {
  return _mm256_min_epi16(a.v, b.v);
}
template <TLV_DECL_VEC(V), typename T = Vec2Tag<V>, TL_IF(T::Bytes == 32), TL_IF(is_any<TypeOf<T>, uint16_t>)>
VECOPS_VFUNC V min(V a, V b) {
  return _mm256_min_epu16(a.v, b.v);
}
template <TLV_DECL_VEC(V), typename T = Vec2Tag<V>, TL_IF(T::Bytes == 32), TL_IF(is_any<TypeOf<T>, int32_t>)>
VECOPS_VFUNC V min(V a, V b) {
  return _mm256_min_epi32(a.v, b.v);
}
template <TLV_DECL_VEC(V), typename T = Vec2Tag<V>, TL_IF(T::Bytes == 32), TL_IF(is_any<TypeOf<T>, uint32_t>)>
VECOPS_VFUNC V min(V a, V b) {
  return _mm256_min_epu32(a.v, b.v);
}
template <TLV_DECL_VEC(V), typename T = Vec2Tag<V>, TL_IF(T::Bytes == 32), TL_IF(is_any<TypeOf<T>, int64_t>)>
VECOPS_VFUNC V min(V a, V b) {
  #ifdef HAS_AVX512F
  return _mm256_min_epi64(a.v, b.v);
  #else
  return _mm256_blendv_epi8(b.v, a.v, _mm256_cmpgt_epi64(b.v, a.v));
  #endif
}
template <TLV_DECL_VEC(V), typename T = Vec2Tag<V>, TL_IF(T::Bytes == 32), TL_IF(is_any<TypeOf<T>, uint64_t>)>
VECOPS_VFUNC V min(V a, V b) {
  #ifdef HAS_AVX512F
  return _mm256_min_epu64(a.v, b.v);
  #else
  static const __m256i flip = _mm256_set1_epi64x((int64_t)0x8000000000000000LL);
  auto a_flip = _mm256_xor_si256(a.v, flip);
  auto b_flip = _mm256_xor_si256(b.v, flip);
  return _mm256_blendv_epi8(b.v, a.v, _mm256_cmpgt_epi64(b_flip, a_flip));
  #endif
}
#ifdef HAS_AVX512_FP16
VECOPS_X86_FP16_NATIVE_BIN_256(min,_mm256_min_ph)
#else
VECOPS_X86_HALF_BIN_256(min,float16_t,_mm256_min_ps,half_cvt::cvt_fp16_to_two_fp32,half_cvt::cvt_two_fp32_to_fp16)
#endif
VECOPS_X86_HALF_BIN_256(min,bfloat16_t,_mm256_min_ps,half_cvt::cvt_bf16_to_two_fp32,half_cvt::cvt_two_fp32_to_bf16)

#endif // VEC_WIDTH >= 256

#if VEC_WIDTH >= 512
template <TLV_DECL_VEC(V), typename T = Vec2Tag<V>, TL_IF(T::Bytes == 64), TL_IF(is_any<TypeOf<T>, float32_t>)>
VECOPS_VFUNC V min(V a, V b) {
  return _mm512_min_ps(a.v, b.v);
}
template <TLV_DECL_VEC(V), typename T = Vec2Tag<V>, TL_IF(T::Bytes == 64), TL_IF(is_any<TypeOf<T>, float64_t>)>
VECOPS_VFUNC V min(V a, V b) {
  return _mm512_min_pd(a.v, b.v);
}
template <TLV_DECL_VEC(V), typename T = Vec2Tag<V>, TL_IF(T::Bytes == 64), TL_IF(is_any<TypeOf<T>, int8_t>)>
VECOPS_VFUNC V min(V a, V b) {
  return _mm512_min_epi8(a.v, b.v);
}
template <TLV_DECL_VEC(V), typename T = Vec2Tag<V>, TL_IF(T::Bytes == 64), TL_IF(is_any<TypeOf<T>, uint8_t>)>
VECOPS_VFUNC V min(V a, V b) {
  return _mm512_min_epu8(a.v, b.v);
}
template <TLV_DECL_VEC(V), typename T = Vec2Tag<V>, TL_IF(T::Bytes == 64), TL_IF(is_any<TypeOf<T>, int16_t>)>
VECOPS_VFUNC V min(V a, V b) {
  return _mm512_min_epi16(a.v, b.v);
}
template <TLV_DECL_VEC(V), typename T = Vec2Tag<V>, TL_IF(T::Bytes == 64), TL_IF(is_any<TypeOf<T>, uint16_t>)>
VECOPS_VFUNC V min(V a, V b) {
  return _mm512_min_epu16(a.v, b.v);
}
template <TLV_DECL_VEC(V), typename T = Vec2Tag<V>, TL_IF(T::Bytes == 64), TL_IF(is_any<TypeOf<T>, int32_t>)>
VECOPS_VFUNC V min(V a, V b) {
  return _mm512_min_epi32(a.v, b.v);
}
template <TLV_DECL_VEC(V), typename T = Vec2Tag<V>, TL_IF(T::Bytes == 64), TL_IF(is_any<TypeOf<T>, uint32_t>)>
VECOPS_VFUNC V min(V a, V b) {
  return _mm512_min_epu32(a.v, b.v);
}
template <TLV_DECL_VEC(V), typename T = Vec2Tag<V>, TL_IF(T::Bytes == 64), TL_IF(is_any<TypeOf<T>, int64_t>)>
VECOPS_VFUNC V min(V a, V b) {
  return _mm512_min_epi64(a.v, b.v);
}
template <TLV_DECL_VEC(V), typename T = Vec2Tag<V>, TL_IF(T::Bytes == 64), TL_IF(is_any<TypeOf<T>, uint64_t>)>
VECOPS_VFUNC V min(V a, V b) {
  return _mm512_min_epu64(a.v, b.v);
}
#ifdef HAS_AVX512_FP16
VECOPS_X86_FP16_NATIVE_BIN_512(min,_mm512_min_ph)
#else
VECOPS_X86_HALF_BIN_512(min,float16_t,_mm512_min_ps,half_cvt::cvt_fp16_to_two_fp32,half_cvt::cvt_two_fp32_to_fp16)
#endif
VECOPS_X86_HALF_BIN_512(min,bfloat16_t,_mm512_min_ps,half_cvt::cvt_bf16_to_two_fp32,half_cvt::cvt_two_fp32_to_bf16)
#endif // VEC_WIDTH >= 512

#ifdef HAS_AVX512DQ
template <TLV_DECL_VEC(V), typename T = Vec2Tag<V>, TL_IF(T::Bytes <= 16), TL_IF(is_any<TypeOf<T>, float32_t>)>
VECOPS_VFUNC V min(V a, V b, Mask<T> m) {
  return _mm_mask_min_ps(a.v, m.v, a.v, b.v);
}
template <TLV_DECL_VEC(V), typename T = Vec2Tag<V>, TL_IF(T::Bytes <= 16), TL_IF(is_any<TypeOf<T>, float64_t>)>
VECOPS_VFUNC V min(V a, V b, Mask<T> m) {
  return _mm_mask_min_pd(a.v, m.v, a.v, b.v);
}
template <TLV_DECL_VEC(V), typename T = Vec2Tag<V>, TL_IF(T::Bytes <= 16), TL_IF(is_any<TypeOf<T>, int8_t>)>
VECOPS_VFUNC V min(V a, V b, Mask<T> m) {
  return _mm_mask_min_epi8(a.v, m.v, a.v, b.v);
}
template <TLV_DECL_VEC(V), typename T = Vec2Tag<V>, TL_IF(T::Bytes <= 16), TL_IF(is_any<TypeOf<T>, uint8_t>)>
VECOPS_VFUNC V min(V a, V b, Mask<T> m) {
  return _mm_mask_min_epu8(a.v, m.v, a.v, b.v);
}
template <TLV_DECL_VEC(V), typename T = Vec2Tag<V>, TL_IF(T::Bytes <= 16), TL_IF(is_any<TypeOf<T>, int16_t>)>
VECOPS_VFUNC V min(V a, V b, Mask<T> m) {
  return _mm_mask_min_epi16(a.v, m.v, a.v, b.v);
}
template <TLV_DECL_VEC(V), typename T = Vec2Tag<V>, TL_IF(T::Bytes <= 16), TL_IF(is_any<TypeOf<T>, uint16_t>)>
VECOPS_VFUNC V min(V a, V b, Mask<T> m) {
  return _mm_mask_min_epu16(a.v, m.v, a.v, b.v);
}
template <TLV_DECL_VEC(V), typename T = Vec2Tag<V>, TL_IF(T::Bytes <= 16), TL_IF(is_any<TypeOf<T>, int32_t>)>
VECOPS_VFUNC V min(V a, V b, Mask<T> m) {
  return _mm_mask_min_epi32(a.v, m.v, a.v, b.v);
}
template <TLV_DECL_VEC(V), typename T = Vec2Tag<V>, TL_IF(T::Bytes <= 16), TL_IF(is_any<TypeOf<T>, uint32_t>)>
VECOPS_VFUNC V min(V a, V b, Mask<T> m) {
  return _mm_mask_min_epu32(a.v, m.v, a.v, b.v);
}
template <TLV_DECL_VEC(V), typename T = Vec2Tag<V>, TL_IF(T::Bytes <= 16), TL_IF(is_any<TypeOf<T>, int64_t>)>
VECOPS_VFUNC V min(V a, V b, Mask<T> m) {
  return _mm_mask_min_epi64(a.v, m.v, a.v, b.v);
}
template <TLV_DECL_VEC(V), typename T = Vec2Tag<V>, TL_IF(T::Bytes <= 16), TL_IF(is_any<TypeOf<T>, uint64_t>)>
VECOPS_VFUNC V min(V a, V b, Mask<T> m) {
  return _mm_mask_min_epu64(a.v, m.v, a.v, b.v);
}

template <TLV_DECL_VEC(V), typename T = Vec2Tag<V>, TL_IF(T::Bytes == 32), TL_IF(is_any<TypeOf<T>, float32_t>)>
VECOPS_VFUNC V min(V a, V b, Mask<T> m) {
  return _mm256_mask_min_ps(a.v, m.v, a.v, b.v);
}
template <TLV_DECL_VEC(V), typename T = Vec2Tag<V>, TL_IF(T::Bytes == 32), TL_IF(is_any<TypeOf<T>, float64_t>)>
VECOPS_VFUNC V min(V a, V b, Mask<T> m) {
  return _mm256_mask_min_pd(a.v, m.v, a.v, b.v);
}
template <TLV_DECL_VEC(V), typename T = Vec2Tag<V>, TL_IF(T::Bytes == 32), TL_IF(is_any<TypeOf<T>, int8_t>)>
VECOPS_VFUNC V min(V a, V b, Mask<T> m) {
  return _mm256_mask_min_epi8(a.v, m.v, a.v, b.v);
}
template <TLV_DECL_VEC(V), typename T = Vec2Tag<V>, TL_IF(T::Bytes == 32), TL_IF(is_any<TypeOf<T>, uint8_t>)>
VECOPS_VFUNC V min(V a, V b, Mask<T> m) {
  return _mm256_mask_min_epu8(a.v, m.v, a.v, b.v);
}
template <TLV_DECL_VEC(V), typename T = Vec2Tag<V>, TL_IF(T::Bytes == 32), TL_IF(is_any<TypeOf<T>, int16_t>)>
VECOPS_VFUNC V min(V a, V b, Mask<T> m) {
  return _mm256_mask_min_epi16(a.v, m.v, a.v, b.v);
}
template <TLV_DECL_VEC(V), typename T = Vec2Tag<V>, TL_IF(T::Bytes == 32), TL_IF(is_any<TypeOf<T>, uint16_t>)>
VECOPS_VFUNC V min(V a, V b, Mask<T> m) {
  return _mm256_mask_min_epu16(a.v, m.v, a.v, b.v);
}
template <TLV_DECL_VEC(V), typename T = Vec2Tag<V>, TL_IF(T::Bytes == 32), TL_IF(is_any<TypeOf<T>, int32_t>)>
VECOPS_VFUNC V min(V a, V b, Mask<T> m) {
  return _mm256_mask_min_epi32(a.v, m.v, a.v, b.v);
}
template <TLV_DECL_VEC(V), typename T = Vec2Tag<V>, TL_IF(T::Bytes == 32), TL_IF(is_any<TypeOf<T>, uint32_t>)>
VECOPS_VFUNC V min(V a, V b, Mask<T> m) {
  return _mm256_mask_min_epu32(a.v, m.v, a.v, b.v);
}
template <TLV_DECL_VEC(V), typename T = Vec2Tag<V>, TL_IF(T::Bytes == 32), TL_IF(is_any<TypeOf<T>, int64_t>)>
VECOPS_VFUNC V min(V a, V b, Mask<T> m) {
  return _mm256_mask_min_epi64(a.v, m.v, a.v, b.v);
}
template <TLV_DECL_VEC(V), typename T = Vec2Tag<V>, TL_IF(T::Bytes == 32), TL_IF(is_any<TypeOf<T>, uint64_t>)>
VECOPS_VFUNC V min(V a, V b, Mask<T> m) {
  return _mm256_mask_min_epu64(a.v, m.v, a.v, b.v);
}

template <TLV_DECL_VEC(V), typename T = Vec2Tag<V>, TL_IF(T::Bytes == 64), TL_IF(is_any<TypeOf<T>, float32_t>)>
VECOPS_VFUNC V min(V a, V b, Mask<T> m) {
  return _mm512_mask_min_ps(a.v, m.v, a.v, b.v);
}
template <TLV_DECL_VEC(V), typename T = Vec2Tag<V>, TL_IF(T::Bytes == 64), TL_IF(is_any<TypeOf<T>, float64_t>)>
VECOPS_VFUNC V min(V a, V b, Mask<T> m) {
  return _mm512_mask_min_pd(a.v, m.v, a.v, b.v);
}
template <TLV_DECL_VEC(V), typename T = Vec2Tag<V>, TL_IF(T::Bytes == 64), TL_IF(is_any<TypeOf<T>, int8_t>)>
VECOPS_VFUNC V min(V a, V b, Mask<T> m) {
  return _mm512_mask_min_epi8(a.v, m.v, a.v, b.v);
}
template <TLV_DECL_VEC(V), typename T = Vec2Tag<V>, TL_IF(T::Bytes == 64), TL_IF(is_any<TypeOf<T>, uint8_t>)>
VECOPS_VFUNC V min(V a, V b, Mask<T> m) {
  return _mm512_mask_min_epu8(a.v, m.v, a.v, b.v);
}
template <TLV_DECL_VEC(V), typename T = Vec2Tag<V>, TL_IF(T::Bytes == 64), TL_IF(is_any<TypeOf<T>, int16_t>)>
VECOPS_VFUNC V min(V a, V b, Mask<T> m) {
  return _mm512_mask_min_epi16(a.v, m.v, a.v, b.v);
}
template <TLV_DECL_VEC(V), typename T = Vec2Tag<V>, TL_IF(T::Bytes == 64), TL_IF(is_any<TypeOf<T>, uint16_t>)>
VECOPS_VFUNC V min(V a, V b, Mask<T> m) {
  return _mm512_mask_min_epu16(a.v, m.v, a.v, b.v);
}
template <TLV_DECL_VEC(V), typename T = Vec2Tag<V>, TL_IF(T::Bytes == 64), TL_IF(is_any<TypeOf<T>, int32_t>)>
VECOPS_VFUNC V min(V a, V b, Mask<T> m) {
  return _mm512_mask_min_epi32(a.v, m.v, a.v, b.v);
}
template <TLV_DECL_VEC(V), typename T = Vec2Tag<V>, TL_IF(T::Bytes == 64), TL_IF(is_any<TypeOf<T>, uint32_t>)>
VECOPS_VFUNC V min(V a, V b, Mask<T> m) {
  return _mm512_mask_min_epu32(a.v, m.v, a.v, b.v);
}
template <TLV_DECL_VEC(V), typename T = Vec2Tag<V>, TL_IF(T::Bytes == 64), TL_IF(is_any<TypeOf<T>, int64_t>)>
VECOPS_VFUNC V min(V a, V b, Mask<T> m) {
  return _mm512_mask_min_epi64(a.v, m.v, a.v, b.v);
}
template <TLV_DECL_VEC(V), typename T = Vec2Tag<V>, TL_IF(T::Bytes == 64), TL_IF(is_any<TypeOf<T>, uint64_t>)>
VECOPS_VFUNC V min(V a, V b, Mask<T> m) {
  return _mm512_mask_min_epu64(a.v, m.v, a.v, b.v);
}
template <TLV_DECL_VEC(V), typename T = Vec2Tag<V>, TL_IF(is_any<TypeOf<T>, float16_t, bfloat16_t>)>
VECOPS_VFUNC V min(V a, V b, Mask<T> m) {
  return word::blend(a, m, word::min(a, b));
}
#else // HAS_AVX512DQ
template <TLV_DECL_VEC(V), typename T = Vec2Tag<V>>
VECOPS_VFUNC V min(V a, V b, Mask<T> m) {
  return word::blend(a, m, word::min(a, b));
}
#endif // HAS_AVX512DQ


/* ************************************************************************** */
//                                   Max                                      //
/* ************************************************************************** */
template <TLV_DECL_VEC(V), typename T = Vec2Tag<V>, TL_IF(T::Bytes <= 16), TL_IF(is_any<TypeOf<T>, float32_t>)>
VECOPS_VFUNC V max(V a, V b) {
  return _mm_max_ps(a.v, b.v);
}
template <TLV_DECL_VEC(V), typename T = Vec2Tag<V>, TL_IF(T::Bytes <= 16), TL_IF(is_any<TypeOf<T>, float64_t>)>
VECOPS_VFUNC V max(V a, V b) {
  return _mm_max_pd(a.v, b.v);
}
template <TLV_DECL_VEC(V), typename T = Vec2Tag<V>, TL_IF(T::Bytes <= 16), TL_IF(is_any<TypeOf<T>, int8_t>)>
VECOPS_VFUNC V max(V a, V b) {
  return _mm_max_epi8(a.v, b.v);
}
template <TLV_DECL_VEC(V), typename T = Vec2Tag<V>, TL_IF(T::Bytes <= 16), TL_IF(is_any<TypeOf<T>, uint8_t>)>
VECOPS_VFUNC V max(V a, V b) {
  return _mm_max_epu8(a.v, b.v);
}
template <TLV_DECL_VEC(V), typename T = Vec2Tag<V>, TL_IF(T::Bytes <= 16), TL_IF(is_any<TypeOf<T>, int16_t>)>
VECOPS_VFUNC V max(V a, V b) {
  return _mm_max_epi16(a.v, b.v);
}
template <TLV_DECL_VEC(V), typename T = Vec2Tag<V>, TL_IF(T::Bytes <= 16), TL_IF(is_any<TypeOf<T>, uint16_t>)>
VECOPS_VFUNC V max(V a, V b) {
  return _mm_max_epu16(a.v, b.v);
}
template <TLV_DECL_VEC(V), typename T = Vec2Tag<V>, TL_IF(T::Bytes <= 16), TL_IF(is_any<TypeOf<T>, int32_t>)>
VECOPS_VFUNC V max(V a, V b) {
  return _mm_max_epi32(a.v, b.v);
}
template <TLV_DECL_VEC(V), typename T = Vec2Tag<V>, TL_IF(T::Bytes <= 16), TL_IF(is_any<TypeOf<T>, uint32_t>)>
VECOPS_VFUNC V max(V a, V b) {
  return _mm_max_epu32(a.v, b.v);
}
template <TLV_DECL_VEC(V), typename T = Vec2Tag<V>, TL_IF(T::Bytes <= 16), TL_IF(is_any<TypeOf<T>, int64_t>)>
VECOPS_VFUNC V max(V a, V b) {
  #ifdef HAS_AVX512F
  return _mm_max_epi64(a.v, b.v);
  #else
  return _mm_blendv_epi8(b.v, a.v, _mm_cmpgt_epi64(a.v, b.v));
  #endif
}
template <TLV_DECL_VEC(V), typename T = Vec2Tag<V>, TL_IF(T::Bytes <= 16), TL_IF(is_any<TypeOf<T>, uint64_t>)>
VECOPS_VFUNC V max(V a, V b) {
  #ifdef HAS_AVX512F
  return _mm_max_epu64(a.v, b.v);
  #else
  static const __m128i sign_bit = _mm_set1_epi64x((int64_t)0x8000000000000000LL);
  auto a_flip = _mm_xor_si128(a.v, sign_bit);
  auto b_flip = _mm_xor_si128(b.v, sign_bit);
  return _mm_blendv_epi8(b.v, a.v, _mm_cmpgt_epi64(a_flip, b_flip));
  #endif
}
#ifdef HAS_AVX512_FP16
VECOPS_X86_FP16_NATIVE_BIN_128(max,_mm_max_ph)
#else
VECOPS_X86_HALF_BIN_128(max,float16_t,_mm_max_ps,half_cvt::cvt_fp16_to_two_fp32,half_cvt::cvt_two_fp32_to_fp16)
#endif
VECOPS_X86_HALF_BIN_128(max,bfloat16_t,_mm_max_ps,half_cvt::cvt_bf16_to_two_fp32,half_cvt::cvt_two_fp32_to_bf16)

#if VEC_WIDTH >= 256
template <TLV_DECL_VEC(V), typename T = Vec2Tag<V>, TL_IF(T::Bytes == 32), TL_IF(is_any<TypeOf<T>, float32_t>)>
VECOPS_VFUNC V max(V a, V b) {
  return _mm256_max_ps(a.v, b.v);
}
template <TLV_DECL_VEC(V), typename T = Vec2Tag<V>, TL_IF(T::Bytes == 32), TL_IF(is_any<TypeOf<T>, float64_t>)>
VECOPS_VFUNC V max(V a, V b) {
  return _mm256_max_pd(a.v, b.v);
}
template <TLV_DECL_VEC(V), typename T = Vec2Tag<V>, TL_IF(T::Bytes == 32), TL_IF(is_any<TypeOf<T>, int8_t>)>
VECOPS_VFUNC V max(V a, V b) {
  return _mm256_max_epi8(a.v, b.v);
}
template <TLV_DECL_VEC(V), typename T = Vec2Tag<V>, TL_IF(T::Bytes == 32), TL_IF(is_any<TypeOf<T>, uint8_t>)>
VECOPS_VFUNC V max(V a, V b) {
  return _mm256_max_epu8(a.v, b.v);
}
template <TLV_DECL_VEC(V), typename T = Vec2Tag<V>, TL_IF(T::Bytes == 32), TL_IF(is_any<TypeOf<T>, int16_t>)>
VECOPS_VFUNC V max(V a, V b) {
  return _mm256_max_epi16(a.v, b.v);
}
template <TLV_DECL_VEC(V), typename T = Vec2Tag<V>, TL_IF(T::Bytes == 32), TL_IF(is_any<TypeOf<T>, uint16_t>)>
VECOPS_VFUNC V max(V a, V b) {
  return _mm256_max_epu16(a.v, b.v);
}
template <TLV_DECL_VEC(V), typename T = Vec2Tag<V>, TL_IF(T::Bytes == 32), TL_IF(is_any<TypeOf<T>, int32_t>)>
VECOPS_VFUNC V max(V a, V b) {
  return _mm256_max_epi32(a.v, b.v);
}
template <TLV_DECL_VEC(V), typename T = Vec2Tag<V>, TL_IF(T::Bytes == 32), TL_IF(is_any<TypeOf<T>, uint32_t>)>
VECOPS_VFUNC V max(V a, V b) {
  return _mm256_max_epu32(a.v, b.v);
}
template <TLV_DECL_VEC(V), typename T = Vec2Tag<V>, TL_IF(T::Bytes == 32), TL_IF(is_any<TypeOf<T>, int64_t>)>
VECOPS_VFUNC V max(V a, V b) {
  #ifdef HAS_AVX512F
  return _mm256_max_epi64(a.v, b.v);
  #else
  return _mm256_blendv_epi8(b.v, a.v, _mm256_cmpgt_epi64(a.v, b.v));
  #endif
}
template <TLV_DECL_VEC(V), typename T = Vec2Tag<V>, TL_IF(T::Bytes == 32), TL_IF(is_any<TypeOf<T>, uint64_t>)>
VECOPS_VFUNC V max(V a, V b) {
  #ifdef HAS_AVX512F
  return _mm256_max_epu64(a.v, b.v);
  #else
  static const __m256i flip = _mm256_set1_epi64x((int64_t)0x8000000000000000LL);
  auto a_flip = _mm256_xor_si256(a.v, flip);
  auto b_flip = _mm256_xor_si256(b.v, flip);
  return _mm256_blendv_epi8(b.v, a.v, _mm256_cmpgt_epi64(a_flip, b_flip));
  #endif
}
#ifdef HAS_AVX512_FP16
VECOPS_X86_FP16_NATIVE_BIN_256(max,_mm256_max_ph)
#else
VECOPS_X86_HALF_BIN_256(max,float16_t,_mm256_max_ps,half_cvt::cvt_fp16_to_two_fp32,half_cvt::cvt_two_fp32_to_fp16)
#endif
VECOPS_X86_HALF_BIN_256(max,bfloat16_t,_mm256_max_ps,half_cvt::cvt_bf16_to_two_fp32,half_cvt::cvt_two_fp32_to_bf16)

#endif // VEC_WIDTH >= 256

#if VEC_WIDTH >= 512
template <TLV_DECL_VEC(V), typename T = Vec2Tag<V>, TL_IF(T::Bytes == 64), TL_IF(is_any<TypeOf<T>, float32_t>)>
VECOPS_VFUNC V max(V a, V b) {
  return _mm512_max_ps(a.v, b.v);
}
template <TLV_DECL_VEC(V), typename T = Vec2Tag<V>, TL_IF(T::Bytes == 64), TL_IF(is_any<TypeOf<T>, float64_t>)>
VECOPS_VFUNC V max(V a, V b) {
  return _mm512_max_pd(a.v, b.v);
}
template <TLV_DECL_VEC(V), typename T = Vec2Tag<V>, TL_IF(T::Bytes == 64), TL_IF(is_any<TypeOf<T>, int8_t>)>
VECOPS_VFUNC V max(V a, V b) {
  return _mm512_max_epi8(a.v, b.v);
}
template <TLV_DECL_VEC(V), typename T = Vec2Tag<V>, TL_IF(T::Bytes == 64), TL_IF(is_any<TypeOf<T>, uint8_t>)>
VECOPS_VFUNC V max(V a, V b) {
  return _mm512_max_epu8(a.v, b.v);
}
template <TLV_DECL_VEC(V), typename T = Vec2Tag<V>, TL_IF(T::Bytes == 64), TL_IF(is_any<TypeOf<T>, int16_t>)>
VECOPS_VFUNC V max(V a, V b) {
  return _mm512_max_epi16(a.v, b.v);
}
template <TLV_DECL_VEC(V), typename T = Vec2Tag<V>, TL_IF(T::Bytes == 64), TL_IF(is_any<TypeOf<T>, uint16_t>)>
VECOPS_VFUNC V max(V a, V b) {
  return _mm512_max_epu16(a.v, b.v);
}
template <TLV_DECL_VEC(V), typename T = Vec2Tag<V>, TL_IF(T::Bytes == 64), TL_IF(is_any<TypeOf<T>, int32_t>)>
VECOPS_VFUNC V max(V a, V b) {
  return _mm512_max_epi32(a.v, b.v);
}
template <TLV_DECL_VEC(V), typename T = Vec2Tag<V>, TL_IF(T::Bytes == 64), TL_IF(is_any<TypeOf<T>, uint32_t>)>
VECOPS_VFUNC V max(V a, V b) {
  return _mm512_max_epu32(a.v, b.v);
}
template <TLV_DECL_VEC(V), typename T = Vec2Tag<V>, TL_IF(T::Bytes == 64), TL_IF(is_any<TypeOf<T>, int64_t>)>
VECOPS_VFUNC V max(V a, V b) {
  return _mm512_max_epi64(a.v, b.v);
}
template <TLV_DECL_VEC(V), typename T = Vec2Tag<V>, TL_IF(T::Bytes == 64), TL_IF(is_any<TypeOf<T>, uint64_t>)>
VECOPS_VFUNC V max(V a, V b) {
  return _mm512_max_epu64(a.v, b.v);
}
#ifdef HAS_AVX512_FP16
VECOPS_X86_FP16_NATIVE_BIN_512(max,_mm512_max_ph)
#else
VECOPS_X86_HALF_BIN_512(max,float16_t,_mm512_max_ps,half_cvt::cvt_fp16_to_two_fp32,half_cvt::cvt_two_fp32_to_fp16)
#endif
VECOPS_X86_HALF_BIN_512(max,bfloat16_t,_mm512_max_ps,half_cvt::cvt_bf16_to_two_fp32,half_cvt::cvt_two_fp32_to_bf16)
#endif // VEC_WIDTH >= 512

#ifdef HAS_AVX512DQ
template <TLV_DECL_VEC(V), typename T = Vec2Tag<V>, TL_IF(T::Bytes <= 16), TL_IF(is_any<TypeOf<T>, float32_t>)>
VECOPS_VFUNC V max(V a, V b, Mask<T> m) {
  return _mm_mask_max_ps(a.v, m.v, a.v, b.v);
}
template <TLV_DECL_VEC(V), typename T = Vec2Tag<V>, TL_IF(T::Bytes <= 16), TL_IF(is_any<TypeOf<T>, float64_t>)>
VECOPS_VFUNC V max(V a, V b, Mask<T> m) {
  return _mm_mask_max_pd(a.v, m.v, a.v, b.v);
}
template <TLV_DECL_VEC(V), typename T = Vec2Tag<V>, TL_IF(T::Bytes <= 16), TL_IF(is_any<TypeOf<T>, int8_t>)>
VECOPS_VFUNC V max(V a, V b, Mask<T> m) {
  return _mm_mask_max_epi8(a.v, m.v, a.v, b.v);
}
template <TLV_DECL_VEC(V), typename T = Vec2Tag<V>, TL_IF(T::Bytes <= 16), TL_IF(is_any<TypeOf<T>, uint8_t>)>
VECOPS_VFUNC V max(V a, V b, Mask<T> m) {
  return _mm_mask_max_epu8(a.v, m.v, a.v, b.v);
}
template <TLV_DECL_VEC(V), typename T = Vec2Tag<V>, TL_IF(T::Bytes <= 16), TL_IF(is_any<TypeOf<T>, int16_t>)>
VECOPS_VFUNC V max(V a, V b, Mask<T> m) {
  return _mm_mask_max_epi16(a.v, m.v, a.v, b.v);
}
template <TLV_DECL_VEC(V), typename T = Vec2Tag<V>, TL_IF(T::Bytes <= 16), TL_IF(is_any<TypeOf<T>, uint16_t>)>
VECOPS_VFUNC V max(V a, V b, Mask<T> m) {
  return _mm_mask_max_epu16(a.v, m.v, a.v, b.v);
}
template <TLV_DECL_VEC(V), typename T = Vec2Tag<V>, TL_IF(T::Bytes <= 16), TL_IF(is_any<TypeOf<T>, int32_t>)>
VECOPS_VFUNC V max(V a, V b, Mask<T> m) {
  return _mm_mask_max_epi32(a.v, m.v, a.v, b.v);
}
template <TLV_DECL_VEC(V), typename T = Vec2Tag<V>, TL_IF(T::Bytes <= 16), TL_IF(is_any<TypeOf<T>, uint32_t>)>
VECOPS_VFUNC V max(V a, V b, Mask<T> m) {
  return _mm_mask_max_epu32(a.v, m.v, a.v, b.v);
}
template <TLV_DECL_VEC(V), typename T = Vec2Tag<V>, TL_IF(T::Bytes <= 16), TL_IF(is_any<TypeOf<T>, int64_t>)>
VECOPS_VFUNC V max(V a, V b, Mask<T> m) {
  return _mm_mask_max_epi64(a.v, m.v, a.v, b.v);
}
template <TLV_DECL_VEC(V), typename T = Vec2Tag<V>, TL_IF(T::Bytes <= 16), TL_IF(is_any<TypeOf<T>, uint64_t>)>
VECOPS_VFUNC V max(V a, V b, Mask<T> m) {
  return _mm_mask_max_epu64(a.v, m.v, a.v, b.v);
}

template <TLV_DECL_VEC(V), typename T = Vec2Tag<V>, TL_IF(T::Bytes == 32), TL_IF(is_any<TypeOf<T>, float32_t>)>
VECOPS_VFUNC V max(V a, V b, Mask<T> m) {
  return _mm256_mask_max_ps(a.v, m.v, a.v, b.v);
}
template <TLV_DECL_VEC(V), typename T = Vec2Tag<V>, TL_IF(T::Bytes == 32), TL_IF(is_any<TypeOf<T>, float64_t>)>
VECOPS_VFUNC V max(V a, V b, Mask<T> m) {
  return _mm256_mask_max_pd(a.v, m.v, a.v, b.v);
}
template <TLV_DECL_VEC(V), typename T = Vec2Tag<V>, TL_IF(T::Bytes == 32), TL_IF(is_any<TypeOf<T>, int8_t>)>
VECOPS_VFUNC V max(V a, V b, Mask<T> m) {
  return _mm256_mask_max_epi8(a.v, m.v, a.v, b.v);
}
template <TLV_DECL_VEC(V), typename T = Vec2Tag<V>, TL_IF(T::Bytes == 32), TL_IF(is_any<TypeOf<T>, uint8_t>)>
VECOPS_VFUNC V max(V a, V b, Mask<T> m) {
  return _mm256_mask_max_epu8(a.v, m.v, a.v, b.v);
}
template <TLV_DECL_VEC(V), typename T = Vec2Tag<V>, TL_IF(T::Bytes == 32), TL_IF(is_any<TypeOf<T>, int16_t>)>
VECOPS_VFUNC V max(V a, V b, Mask<T> m) {
  return _mm256_mask_max_epi16(a.v, m.v, a.v, b.v);
}
template <TLV_DECL_VEC(V), typename T = Vec2Tag<V>, TL_IF(T::Bytes == 32), TL_IF(is_any<TypeOf<T>, uint16_t>)>
VECOPS_VFUNC V max(V a, V b, Mask<T> m) {
  return _mm256_mask_max_epu16(a.v, m.v, a.v, b.v);
}
template <TLV_DECL_VEC(V), typename T = Vec2Tag<V>, TL_IF(T::Bytes == 32), TL_IF(is_any<TypeOf<T>, int32_t>)>
VECOPS_VFUNC V max(V a, V b, Mask<T> m) {
  return _mm256_mask_max_epi32(a.v, m.v, a.v, b.v);
}
template <TLV_DECL_VEC(V), typename T = Vec2Tag<V>, TL_IF(T::Bytes == 32), TL_IF(is_any<TypeOf<T>, uint32_t>)>
VECOPS_VFUNC V max(V a, V b, Mask<T> m) {
  return _mm256_mask_max_epu32(a.v, m.v, a.v, b.v);
}
template <TLV_DECL_VEC(V), typename T = Vec2Tag<V>, TL_IF(T::Bytes == 32), TL_IF(is_any<TypeOf<T>, int64_t>)>
VECOPS_VFUNC V max(V a, V b, Mask<T> m) {
  return _mm256_mask_max_epi64(a.v, m.v, a.v, b.v);
}
template <TLV_DECL_VEC(V), typename T = Vec2Tag<V>, TL_IF(T::Bytes == 32), TL_IF(is_any<TypeOf<T>, uint64_t>)>
VECOPS_VFUNC V max(V a, V b, Mask<T> m) {
  return _mm256_mask_max_epu64(a.v, m.v, a.v, b.v);
}

template <TLV_DECL_VEC(V), typename T = Vec2Tag<V>, TL_IF(T::Bytes == 64), TL_IF(is_any<TypeOf<T>, float32_t>)>
VECOPS_VFUNC V max(V a, V b, Mask<T> m) {
  return _mm512_mask_max_ps(a.v, m.v, a.v, b.v);
}
template <TLV_DECL_VEC(V), typename T = Vec2Tag<V>, TL_IF(T::Bytes == 64), TL_IF(is_any<TypeOf<T>, float64_t>)>
VECOPS_VFUNC V max(V a, V b, Mask<T> m) {
  return _mm512_mask_max_pd(a.v, m.v, a.v, b.v);
}
template <TLV_DECL_VEC(V), typename T = Vec2Tag<V>, TL_IF(T::Bytes == 64), TL_IF(is_any<TypeOf<T>, int8_t>)>
VECOPS_VFUNC V max(V a, V b, Mask<T> m) {
  return _mm512_mask_max_epi8(a.v, m.v, a.v, b.v);
}
template <TLV_DECL_VEC(V), typename T = Vec2Tag<V>, TL_IF(T::Bytes == 64), TL_IF(is_any<TypeOf<T>, uint8_t>)>
VECOPS_VFUNC V max(V a, V b, Mask<T> m) {
  return _mm512_mask_max_epu8(a.v, m.v, a.v, b.v);
}
template <TLV_DECL_VEC(V), typename T = Vec2Tag<V>, TL_IF(T::Bytes == 64), TL_IF(is_any<TypeOf<T>, int16_t>)>
VECOPS_VFUNC V max(V a, V b, Mask<T> m) {
  return _mm512_mask_max_epi16(a.v, m.v, a.v, b.v);
}
template <TLV_DECL_VEC(V), typename T = Vec2Tag<V>, TL_IF(T::Bytes == 64), TL_IF(is_any<TypeOf<T>, uint16_t>)>
VECOPS_VFUNC V max(V a, V b, Mask<T> m) {
  return _mm512_mask_max_epu16(a.v, m.v, a.v, b.v);
}
template <TLV_DECL_VEC(V), typename T = Vec2Tag<V>, TL_IF(T::Bytes == 64), TL_IF(is_any<TypeOf<T>, int32_t>)>
VECOPS_VFUNC V max(V a, V b, Mask<T> m) {
  return _mm512_mask_max_epi32(a.v, m.v, a.v, b.v);
}
template <TLV_DECL_VEC(V), typename T = Vec2Tag<V>, TL_IF(T::Bytes == 64), TL_IF(is_any<TypeOf<T>, uint32_t>)>
VECOPS_VFUNC V max(V a, V b, Mask<T> m) {
  return _mm512_mask_max_epu32(a.v, m.v, a.v, b.v);
}
template <TLV_DECL_VEC(V), typename T = Vec2Tag<V>, TL_IF(T::Bytes == 64), TL_IF(is_any<TypeOf<T>, int64_t>)>
VECOPS_VFUNC V max(V a, V b, Mask<T> m) {
  return _mm512_mask_max_epi64(a.v, m.v, a.v, b.v);
}
template <TLV_DECL_VEC(V), typename T = Vec2Tag<V>, TL_IF(T::Bytes == 64), TL_IF(is_any<TypeOf<T>, uint64_t>)>
VECOPS_VFUNC V max(V a, V b, Mask<T> m) {
  return _mm512_mask_max_epu64(a.v, m.v, a.v, b.v);
}
template <TLV_DECL_VEC(V), typename T = Vec2Tag<V>, TL_IF(is_any<TypeOf<T>, float16_t, bfloat16_t>)>
VECOPS_VFUNC V max(V a, V b, Mask<T> m) {
  return word::blend(a, m, word::max(a, b));
}
#else // HAS_AVX512DQ
template <TLV_DECL_VEC(V), typename T = Vec2Tag<V>>
VECOPS_VFUNC V max(V a, V b, Mask<T> m) {
  return word::blend(a, m, word::max(a, b));
}
#endif // HAS_AVX512DQ

/* ************************************************************************** */
//                             Rcp / Reciprocal                               //
/* ************************************************************************** */
// Note: AVX512 uses rcp14 which gives higher accuracy than rcp used not in AVX512
template <TLV_DECL_VEC(V), typename T = Vec2Tag<V>, TL_IF(T::Bytes <= 16), TL_IF(is_any<TypeOf<T>, float32_t>)>
VECOPS_VFUNC V rcp(V v) {
  #ifdef HAS_AVX512F
  return _mm_rcp14_ps(v.v);
  #else
  return _mm_rcp_ps(v.v);
  #endif
}
template <TLV_DECL_VEC(V), typename T = Vec2Tag<V>, TL_IF(T::Bytes <= 16), TL_IF(is_any<TypeOf<T>, float64_t>)>
VECOPS_VFUNC V rcp(V v) {
  #ifdef HAS_AVX512F
  return _mm_rcp14_pd(v.v);
  #else
  return word::div(word::fill(T(), 1), v);
  #endif
}
#ifdef HAS_AVX512_FP16
VECOPS_X86_FP16_NATIVE_UNARY_128(rcp,_mm_rcp_ph)
#else
VECOPS_X86_HALF_UNARY_128(rcp,float16_t,_mm_rcp_ps,half_cvt::cvt_fp16_to_two_fp32,half_cvt::cvt_two_fp32_to_fp16)
#endif
VECOPS_X86_HALF_UNARY_128(rcp,bfloat16_t,_mm_rcp_ps,half_cvt::cvt_bf16_to_two_fp32,half_cvt::cvt_two_fp32_to_bf16)

#if VEC_WIDTH >= 256
template <TLV_DECL_VEC(V), typename T = Vec2Tag<V>, TL_IF(T::Bytes == 32), TL_IF(is_any<TypeOf<T>, float32_t>)>
VECOPS_VFUNC V rcp(V v) {
  #ifdef HAS_AVX512F
  return _mm256_rcp14_ps(v.v);
  #else
  return _mm256_rcp_ps(v.v);
  #endif
}
template <TLV_DECL_VEC(V), typename T = Vec2Tag<V>, TL_IF(T::Bytes == 32), TL_IF(is_any<TypeOf<T>, float64_t>)>
VECOPS_VFUNC V rcp(V v) {
  #ifdef HAS_AVX512F
  return _mm256_rcp14_pd(v.v);
  #else
  return word::div(word::fill(T(), static_cast<TypeOf<T>>(1)), v);
  #endif
}
#ifdef HAS_AVX512_FP16
VECOPS_X86_FP16_NATIVE_UNARY_256(rcp,_mm256_rcp_ph)
#else
VECOPS_X86_HALF_UNARY_256(rcp,float16_t,_mm256_rcp_ps,half_cvt::cvt_fp16_to_two_fp32,half_cvt::cvt_two_fp32_to_fp16)
#endif
VECOPS_X86_HALF_UNARY_256(rcp,bfloat16_t,_mm256_rcp_ps,half_cvt::cvt_bf16_to_two_fp32,half_cvt::cvt_two_fp32_to_bf16)
#endif // VEC_WIDTH >= 256

#if VEC_WIDTH >= 512
template <TLV_DECL_VEC(V), typename T = Vec2Tag<V>, TL_IF(T::Bytes == 64), TL_IF(is_any<TypeOf<T>, float32_t>)>
VECOPS_VFUNC V rcp(V v) {
  return _mm512_rcp14_ps(v.v);
}
template <TLV_DECL_VEC(V), typename T = Vec2Tag<V>, TL_IF(T::Bytes == 64), TL_IF(is_any<TypeOf<T>, float64_t>)>
VECOPS_VFUNC V rcp(V v) {
  return _mm512_rcp14_pd(v.v);
}
#ifdef HAS_AVX512_FP16
VECOPS_X86_FP16_NATIVE_UNARY_512(rcp,_mm512_rcp_ph)
#else
VECOPS_X86_HALF_UNARY_512(rcp,float16_t,_mm512_rcp14_ps,half_cvt::cvt_fp16_to_two_fp32,half_cvt::cvt_two_fp32_to_fp16)
#endif
VECOPS_X86_HALF_UNARY_512(rcp,bfloat16_t,_mm512_rcp14_ps,half_cvt::cvt_bf16_to_two_fp32,half_cvt::cvt_two_fp32_to_bf16)
#endif // VEC_WIDTH >= 512

#ifdef HAS_AVX512DQ
template <TLV_DECL_VEC(V), typename T = Vec2Tag<V>, TL_IF(T::Bytes <= 16), TL_IF(is_any<TypeOf<T>, float32_t>)>
VECOPS_VFUNC V rcp(V v, Mask<T> m, V default_v) {
  return _mm_mask_rcp14_ps(default_v.v, m.v, v.v);
}
template <TLV_DECL_VEC(V), typename T = Vec2Tag<V>, TL_IF(T::Bytes <= 16), TL_IF(is_any<TypeOf<T>, float64_t>)>
VECOPS_VFUNC V rcp(V v, Mask<T> m, V default_v) {
  return _mm_mask_rcp14_pd(default_v.v, m.v, v.v);
}

template <TLV_DECL_VEC(V), typename T = Vec2Tag<V>, TL_IF(T::Bytes == 32), TL_IF(is_any<TypeOf<T>, float32_t>)>
VECOPS_VFUNC V rcp(V v, Mask<T> m, V default_v) {
  return _mm256_mask_rcp14_ps(default_v.v, m.v, v.v);
}
template <TLV_DECL_VEC(V), typename T = Vec2Tag<V>, TL_IF(T::Bytes == 32), TL_IF(is_any<TypeOf<T>, float64_t>)>
VECOPS_VFUNC V rcp(V v, Mask<T> m, V default_v) {
  return _mm256_mask_rcp14_pd(default_v.v, m.v, v.v);
}

template <TLV_DECL_VEC(V), typename T = Vec2Tag<V>, TL_IF(T::Bytes == 64), TL_IF(is_any<TypeOf<T>, float32_t>)>
VECOPS_VFUNC V rcp(V v, Mask<T> m, V default_v) {
  return _mm512_mask_rcp14_ps(default_v.v, m.v, v.v);
}
template <TLV_DECL_VEC(V), typename T = Vec2Tag<V>, TL_IF(T::Bytes == 64), TL_IF(is_any<TypeOf<T>, float64_t>)>
VECOPS_VFUNC V rcp(V v, Mask<T> m, V default_v) {
  return _mm512_mask_rcp14_pd(default_v.v, m.v, v.v);
}
template <TLV_DECL_VEC(V), typename T = Vec2Tag<V>, TL_IF(is_any<TypeOf<T>, float16_t, bfloat16_t>)>
VECOPS_VFUNC V rcp(V v, Mask<T> m, V default_v) {
  return word::blend(default_v, m, word::rcp(v));
}
#else // HAS_AVX512DQ
template <TLV_DECL_VEC(V), typename T = Vec2Tag<V>>
VECOPS_VFUNC V rcp(V v, Mask<T> m, V default_v) {
  return word::blend(default_v, m, word::rcp(v));
}
#endif // HAS_AVX512DQ


/* ************************************************************************** */
//                                   Sqrt                                     //
/* ************************************************************************** */
template <TLV_DECL_VEC(V), typename T = Vec2Tag<V>, TL_IF(T::Bytes <= 16), TL_IF(is_any<TypeOf<T>, float32_t>)>
VECOPS_VFUNC V sqrt(V v) {
  return _mm_sqrt_ps(v.v);
}
template <TLV_DECL_VEC(V), typename T = Vec2Tag<V>, TL_IF(T::Bytes <= 16), TL_IF(is_any<TypeOf<T>, float64_t>)>
VECOPS_VFUNC V sqrt(V v) {
  return _mm_sqrt_pd(v.v);
}
#ifdef HAS_AVX512_FP16
VECOPS_X86_FP16_NATIVE_UNARY_128(sqrt,_mm_sqrt_ph)
#else
VECOPS_X86_HALF_UNARY_128(sqrt,float16_t,_mm_sqrt_ps,half_cvt::cvt_fp16_to_two_fp32,half_cvt::cvt_two_fp32_to_fp16)
#endif
VECOPS_X86_HALF_UNARY_128(sqrt,bfloat16_t,_mm_sqrt_ps,half_cvt::cvt_bf16_to_two_fp32,half_cvt::cvt_two_fp32_to_bf16)

#if VEC_WIDTH >= 256
template <TLV_DECL_VEC(V), typename T = Vec2Tag<V>, TL_IF(T::Bytes == 32), TL_IF(is_any<TypeOf<T>, float32_t>)>
VECOPS_VFUNC V sqrt(V v) {
  return _mm256_sqrt_ps(v.v);
}
template <TLV_DECL_VEC(V), typename T = Vec2Tag<V>, TL_IF(T::Bytes == 32), TL_IF(is_any<TypeOf<T>, float64_t>)>
VECOPS_VFUNC V sqrt(V v) {
  return _mm256_sqrt_pd(v.v);
}
#ifdef HAS_AVX512_FP16
VECOPS_X86_FP16_NATIVE_UNARY_256(sqrt,_mm256_sqrt_ph)
#else
VECOPS_X86_HALF_UNARY_256(sqrt,float16_t,_mm256_sqrt_ps,half_cvt::cvt_fp16_to_two_fp32,half_cvt::cvt_two_fp32_to_fp16)
#endif
VECOPS_X86_HALF_UNARY_256(sqrt,bfloat16_t,_mm256_sqrt_ps,half_cvt::cvt_bf16_to_two_fp32,half_cvt::cvt_two_fp32_to_bf16)

#endif // VEC_WIDTH >= 256

#if VEC_WIDTH >= 512
template <TLV_DECL_VEC(V), typename T = Vec2Tag<V>, TL_IF(T::Bytes == 64), TL_IF(is_any<TypeOf<T>, float32_t>)>
VECOPS_VFUNC V sqrt(V v) {
  return _mm512_sqrt_ps(v.v);
}
template <TLV_DECL_VEC(V), typename T = Vec2Tag<V>, TL_IF(T::Bytes == 64), TL_IF(is_any<TypeOf<T>, float64_t>)>
VECOPS_VFUNC V sqrt(V v) {
  return _mm512_sqrt_pd(v.v);
}
#ifdef HAS_AVX512_FP16
VECOPS_X86_FP16_NATIVE_UNARY_512(sqrt,_mm512_sqrt_ph)
#else
VECOPS_X86_HALF_UNARY_512(sqrt,float16_t,_mm512_sqrt_ps,half_cvt::cvt_fp16_to_two_fp32,half_cvt::cvt_two_fp32_to_fp16)
#endif
VECOPS_X86_HALF_UNARY_512(sqrt,bfloat16_t,_mm512_sqrt_ps,half_cvt::cvt_bf16_to_two_fp32,half_cvt::cvt_two_fp32_to_bf16)
#endif // VEC_WIDTH >= 512

#ifdef HAS_AVX512DQ
template <TLV_DECL_VEC(V), typename T = Vec2Tag<V>, TL_IF(T::Bytes <= 16), TL_IF(is_any<TypeOf<T>, float32_t>)>
VECOPS_VFUNC V sqrt(V v, Mask<T> m, V default_v) {
  return _mm_mask_sqrt_ps(default_v.v, m.v, v.v);
}
template <TLV_DECL_VEC(V), typename T = Vec2Tag<V>, TL_IF(T::Bytes <= 16), TL_IF(is_any<TypeOf<T>, float64_t>)>
VECOPS_VFUNC V sqrt(V v, Mask<T> m, V default_v) {
  return _mm_mask_sqrt_pd(default_v.v, m.v, v.v);
}

template <TLV_DECL_VEC(V), typename T = Vec2Tag<V>, TL_IF(T::Bytes == 32), TL_IF(is_any<TypeOf<T>, float32_t>)>
VECOPS_VFUNC V sqrt(V v, Mask<T> m, V default_v) {
  return _mm256_mask_sqrt_ps(default_v.v, m.v, v.v);
}
template <TLV_DECL_VEC(V), typename T = Vec2Tag<V>, TL_IF(T::Bytes == 32), TL_IF(is_any<TypeOf<T>, float64_t>)>
VECOPS_VFUNC V sqrt(V v, Mask<T> m, V default_v) {
  return _mm256_mask_sqrt_pd(default_v.v, m.v, v.v);
}

template <TLV_DECL_VEC(V), typename T = Vec2Tag<V>, TL_IF(T::Bytes == 64), TL_IF(is_any<TypeOf<T>, float32_t>)>
VECOPS_VFUNC V sqrt(V v, Mask<T> m, V default_v) {
  return _mm512_mask_sqrt_ps(default_v.v, m.v, v.v);
}
template <TLV_DECL_VEC(V), typename T = Vec2Tag<V>, TL_IF(T::Bytes == 64), TL_IF(is_any<TypeOf<T>, float64_t>)>
VECOPS_VFUNC V sqrt(V v, Mask<T> m, V default_v) {
  return _mm512_mask_sqrt_pd(default_v.v, m.v, v.v);
}
template <TLV_DECL_VEC(V), typename T = Vec2Tag<V>, TL_IF(is_any<TypeOf<T>, float16_t, bfloat16_t>)>
VECOPS_VFUNC V sqrt(V v, Mask<T> m, V default_v) {
  return word::blend(default_v, m, word::sqrt(v));
}
#else // HAS_AVX512DQ
template <TLV_DECL_VEC(V), typename T = Vec2Tag<V>>
VECOPS_VFUNC V sqrt(V v, Mask<T> m, V default_v) {
  return word::blend(default_v, m, word::sqrt(v));
}
#endif // HAS_AVX512DQ


/* ************************************************************************** */
//                        Rsqrt / Reciprocal of Sqrt                          //
/* ************************************************************************** */
// Note: AVX512 uses rsqrt14 which gives higher accuracy than rsqrt used not in AVX512
template <TLV_DECL_VEC(V), typename T = Vec2Tag<V>, TL_IF(T::Bytes <= 16), TL_IF(is_any<TypeOf<T>, float32_t>)>
VECOPS_VFUNC V rsqrt(V v) {
  #ifdef HAS_AVX512F
  return _mm_rsqrt14_ps(v.v);
  #else
  return _mm_rsqrt_ps(v.v);
  #endif
}
template <TLV_DECL_VEC(V), typename T = Vec2Tag<V>, TL_IF(T::Bytes <= 16), TL_IF(is_any<TypeOf<T>, float64_t>)>
VECOPS_VFUNC V rsqrt(V v) {
  #ifdef HAS_AVX512F
  return _mm_rsqrt14_pd(v.v);
  #else
  return word::div(word::fill(T(), 1), word::sqrt(v));
  #endif
}
#ifdef HAS_AVX512_FP16
VECOPS_X86_FP16_NATIVE_UNARY_128(rsqrt,_mm_rsqrt_ph)
#else
VECOPS_X86_HALF_UNARY_128(rsqrt,float16_t,_mm_rsqrt_ps,half_cvt::cvt_fp16_to_two_fp32,half_cvt::cvt_two_fp32_to_fp16)
#endif
VECOPS_X86_HALF_UNARY_128(rsqrt,bfloat16_t,_mm_rsqrt_ps,half_cvt::cvt_bf16_to_two_fp32,half_cvt::cvt_two_fp32_to_bf16)

#if VEC_WIDTH >= 256
template <TLV_DECL_VEC(V), typename T = Vec2Tag<V>, TL_IF(T::Bytes == 32), TL_IF(is_any<TypeOf<T>, float32_t>)>
VECOPS_VFUNC V rsqrt(V v) {
  #ifdef HAS_AVX512F
  return _mm256_rsqrt14_ps(v.v);
  #else
  return _mm256_rsqrt_ps(v.v);
  #endif
}
template <TLV_DECL_VEC(V), typename T = Vec2Tag<V>, TL_IF(T::Bytes == 32), TL_IF(is_any<TypeOf<T>, float64_t>)>
VECOPS_VFUNC V rsqrt(V v) {
  #ifdef HAS_AVX512F
  return _mm256_rsqrt14_pd(v.v);
  #else
  return word::div(word::fill(T(), static_cast<TypeOf<T>>(1)), word::sqrt(v));
  #endif
}
#ifdef HAS_AVX512_FP16
VECOPS_X86_FP16_NATIVE_UNARY_256(rsqrt,_mm256_rsqrt_ph)
#else
VECOPS_X86_HALF_UNARY_256(rsqrt,float16_t,_mm256_rsqrt_ps,half_cvt::cvt_fp16_to_two_fp32,half_cvt::cvt_two_fp32_to_fp16)
#endif
VECOPS_X86_HALF_UNARY_256(rsqrt,bfloat16_t,_mm256_rsqrt_ps,half_cvt::cvt_bf16_to_two_fp32,half_cvt::cvt_two_fp32_to_bf16)
#endif // VEC_WIDTH >= 256

#if VEC_WIDTH >= 512
template <TLV_DECL_VEC(V), typename T = Vec2Tag<V>, TL_IF(T::Bytes == 64), TL_IF(is_any<TypeOf<T>, float32_t>)>
VECOPS_VFUNC V rsqrt(V v) {
  return _mm512_rsqrt14_ps(v.v);
}
template <TLV_DECL_VEC(V), typename T = Vec2Tag<V>, TL_IF(T::Bytes == 64), TL_IF(is_any<TypeOf<T>, float64_t>)>
VECOPS_VFUNC V rsqrt(V v) {
  return _mm512_rsqrt14_pd(v.v);
}
#ifdef HAS_AVX512_FP16
VECOPS_X86_FP16_NATIVE_UNARY_512(rsqrt,_mm512_rsqrt_ph)
#else
VECOPS_X86_HALF_UNARY_512(rsqrt,float16_t,_mm512_rsqrt14_ps,half_cvt::cvt_fp16_to_two_fp32,half_cvt::cvt_two_fp32_to_fp16)
#endif
VECOPS_X86_HALF_UNARY_512(rsqrt,bfloat16_t,_mm512_rsqrt14_ps,half_cvt::cvt_bf16_to_two_fp32,half_cvt::cvt_two_fp32_to_bf16)
#endif // VEC_WIDTH >= 512

#ifdef HAS_AVX512DQ
template <TLV_DECL_VEC(V), typename T = Vec2Tag<V>, TL_IF(T::Bytes <= 16), TL_IF(is_any<TypeOf<T>, float32_t>)>
VECOPS_VFUNC V rsqrt(V v, Mask<T> m, V default_v) {
  return _mm_mask_rsqrt14_ps(default_v.v, m.v, v.v);
}
template <TLV_DECL_VEC(V), typename T = Vec2Tag<V>, TL_IF(T::Bytes <= 16), TL_IF(is_any<TypeOf<T>, float64_t>)>
VECOPS_VFUNC V rsqrt(V v, Mask<T> m, V default_v) {
  return _mm_mask_rsqrt14_pd(default_v.v, m.v, v.v);
}

template <TLV_DECL_VEC(V), typename T = Vec2Tag<V>, TL_IF(T::Bytes == 32), TL_IF(is_any<TypeOf<T>, float32_t>)>
VECOPS_VFUNC V rsqrt(V v, Mask<T> m, V default_v) {
  return _mm256_mask_rsqrt14_ps(default_v.v, m.v, v.v);
}
template <TLV_DECL_VEC(V), typename T = Vec2Tag<V>, TL_IF(T::Bytes == 32), TL_IF(is_any<TypeOf<T>, float64_t>)>
VECOPS_VFUNC V rsqrt(V v, Mask<T> m, V default_v) {
  return _mm256_mask_rsqrt14_pd(default_v.v, m.v, v.v);
}

template <TLV_DECL_VEC(V), typename T = Vec2Tag<V>, TL_IF(T::Bytes == 64), TL_IF(is_any<TypeOf<T>, float32_t>)>
VECOPS_VFUNC V rsqrt(V v, Mask<T> m, V default_v) {
  return _mm512_mask_rsqrt14_ps(default_v.v, m.v, v.v);
}
template <TLV_DECL_VEC(V), typename T = Vec2Tag<V>, TL_IF(T::Bytes == 64), TL_IF(is_any<TypeOf<T>, float64_t>)>
VECOPS_VFUNC V rsqrt(V v, Mask<T> m, V default_v) {
  return _mm512_mask_rsqrt14_pd(default_v.v, m.v, v.v);
}
template <TLV_DECL_VEC(V), typename T = Vec2Tag<V>, TL_IF(is_any<TypeOf<T>, float16_t, bfloat16_t>)>
VECOPS_VFUNC V rsqrt(V v, Mask<T> m, V default_v) {
  return word::blend(default_v, m, word::rsqrt(v));
}
#else // HAS_AVX512DQ
template <TLV_DECL_VEC(V), typename T = Vec2Tag<V>>
VECOPS_VFUNC V rsqrt(V v, Mask<T> m, V default_v) {
  return word::blend(default_v, m, word::rsqrt(v));
}
#endif // HAS_AVX512DQ


/* ************************************************************************** */
//                                  Negate                                    //
/* ************************************************************************** */
// Also defined for unsigned ints
template <TLV_DECL_VEC(V), typename T = Vec2Tag<V>>
VECOPS_VFUNC V neg(V v) {
  return word::sub(word::zeros(T()), v);
}
template <TLV_DECL_VEC(V), typename T = Vec2Tag<V>>
VECOPS_VFUNC V neg(V v, Mask<T> m, V default_v) {
  return word::blend(default_v, m, word::sub(word::zeros(T()), v));
}


/* ************************************************************************** */
//                              Absolute Value                                //
/* ************************************************************************** */
template <TLV_DECL_VEC(V), typename T = Vec2Tag<V>, TL_IF(T::Bytes <= 16), TL_IF(is_any<TypeOf<T>, float32_t>)>
VECOPS_VFUNC V abs(V v) {
  return _mm_and_ps(_mm_castsi128_ps(_mm_set1_epi32(0x7FFFFFFF)), v.v);
}
template <TLV_DECL_VEC(V), typename T = Vec2Tag<V>, TL_IF(T::Bytes <= 16), TL_IF(is_any<TypeOf<T>, float64_t>)>
VECOPS_VFUNC V abs(V v) {
  return _mm_and_pd(_mm_castsi128_pd(_mm_set1_epi64x(0x7FFFFFFFFFFFFFFFLL)), v.v);
}
template <TLV_DECL_VEC(V), typename T = Vec2Tag<V>, TL_IF(is_int<TypeOf<T>> && std::is_unsigned_v<TypeOf<T>>)>
VECOPS_VFUNC V abs(V v) {
  return v;
}
template <TLV_DECL_VEC(V), typename T = Vec2Tag<V>, TL_IF(T::Bytes <= 16), TL_IF(is_any<TypeOf<T>, int8_t>)>
VECOPS_VFUNC V abs(V v) {
  return _mm_abs_epi8(v.v);
}
template <TLV_DECL_VEC(V), typename T = Vec2Tag<V>, TL_IF(T::Bytes <= 16), TL_IF(is_any<TypeOf<T>, int16_t>)>
VECOPS_VFUNC V abs(V v) {
  return _mm_abs_epi16(v.v);
}
template <TLV_DECL_VEC(V), typename T = Vec2Tag<V>, TL_IF(T::Bytes <= 16), TL_IF(is_any<TypeOf<T>, int32_t>)>
VECOPS_VFUNC V abs(V v) {
  return _mm_abs_epi32(v.v);
}
template <TLV_DECL_VEC(V), typename T = Vec2Tag<V>, TL_IF(T::Bytes <= 16), TL_IF(is_any<TypeOf<T>, int64_t>)>
VECOPS_VFUNC V abs(V v) {
  #ifdef HAS_AVX512DQ
  return _mm_abs_epi64(v.v);
  #else
  ViewAs<int32_t, T> t32; T t64;
  auto high32 = word::local_shuf<3, 3, 1, 1>(word::bitcast(t32, v));
  auto sign_mask = word::bit_shr<31>(high32);
  auto xored = word::bit_xor(word::bitcast(t32, v), sign_mask);
  return word::sub(word::bitcast(t64, xored), word::bitcast(t64, sign_mask));
  #endif
}

template<TLV_DECL_VEC(V),typename T=Vec2Tag<V>,TL_IF(T::Bytes<=16),TL_IF(is_any<TypeOf<T>,float16_t,bfloat16_t>)>VECOPS_VFUNC V abs(V v){return _mm_and_si128(v.v,_mm_set1_epi16(0x7FFF));}

#if VEC_WIDTH >= 256
template <TLV_DECL_VEC(V), typename T = Vec2Tag<V>, TL_IF(T::Bytes == 32), TL_IF(is_any<TypeOf<T>, float32_t>)>
VECOPS_VFUNC V abs(V v) {
  return _mm256_and_ps(_mm256_castsi256_ps(_mm256_set1_epi32(0x7FFFFFFF)), v.v);
}
template <TLV_DECL_VEC(V), typename T = Vec2Tag<V>, TL_IF(T::Bytes == 32), TL_IF(is_any<TypeOf<T>, float64_t>)>
VECOPS_VFUNC V abs(V v) {
  return _mm256_and_pd(_mm256_castsi256_pd(_mm256_set1_epi64x(0x7FFFFFFFFFFFFFFFLL)), v.v);
}
template <TLV_DECL_VEC(V), typename T = Vec2Tag<V>, TL_IF(T::Bytes == 32), TL_IF(is_any<TypeOf<T>, int8_t>)>
VECOPS_VFUNC V abs(V v) {
  return _mm256_abs_epi8(v.v);
}
template <TLV_DECL_VEC(V), typename T = Vec2Tag<V>, TL_IF(T::Bytes == 32), TL_IF(is_any<TypeOf<T>, int16_t>)>
VECOPS_VFUNC V abs(V v) {
  return _mm256_abs_epi16(v.v);
}
template <TLV_DECL_VEC(V), typename T = Vec2Tag<V>, TL_IF(T::Bytes == 32), TL_IF(is_any<TypeOf<T>, int32_t>)>
VECOPS_VFUNC V abs(V v) {
  return _mm256_abs_epi32(v.v);
}
template <TLV_DECL_VEC(V), typename T = Vec2Tag<V>, TL_IF(T::Bytes == 32), TL_IF(is_any<TypeOf<T>, int64_t>)>
VECOPS_VFUNC V abs(V v) {
  #ifdef HAS_AVX512DQ
  return _mm256_abs_epi64(v.v);
  #else
  ViewAs<int32_t, T> t32;
  auto high32 = word::local_shuf<3, 3, 1, 1>(word::bitcast(t32, v));
  auto sign_mask = word::bitcast(T(), word::bit_shr<31>(high32));
  auto xored = word::bit_xor(v, sign_mask);
  return word::sub(xored, sign_mask);
  #endif
}
template<TLV_DECL_VEC(V),typename T=Vec2Tag<V>,TL_IF(T::Bytes==32),TL_IF(is_any<TypeOf<T>,float16_t,bfloat16_t>)>VECOPS_VFUNC V abs(V v){return _mm256_and_si256(v.v,_mm256_set1_epi16(0x7FFF));}
#endif // VEC_WIDTH >= 256

#if VEC_WIDTH >= 512
template <TLV_DECL_VEC(V), typename T = Vec2Tag<V>, TL_IF(T::Bytes == 64), TL_IF(is_any<TypeOf<T>, float32_t>)>
VECOPS_VFUNC V abs(V v) {
  return _mm512_and_ps(_mm512_castsi512_ps(_mm512_set1_epi32(0x7FFFFFFF)), v.v);
}
template <TLV_DECL_VEC(V), typename T = Vec2Tag<V>, TL_IF(T::Bytes == 64), TL_IF(is_any<TypeOf<T>, float64_t>)>
VECOPS_VFUNC V abs(V v) {
  return _mm512_and_pd(_mm512_castsi512_pd(_mm512_set1_epi64(0x7FFFFFFFFFFFFFFFLL)), v.v);
}
template <TLV_DECL_VEC(V), typename T = Vec2Tag<V>, TL_IF(T::Bytes == 64), TL_IF(is_any<TypeOf<T>, int8_t>)>
VECOPS_VFUNC V abs(V v) {
  return _mm512_abs_epi8(v.v);
}
template <TLV_DECL_VEC(V), typename T = Vec2Tag<V>, TL_IF(T::Bytes == 64), TL_IF(is_any<TypeOf<T>, int16_t>)>
VECOPS_VFUNC V abs(V v) {
  return _mm512_abs_epi16(v.v);
}
template <TLV_DECL_VEC(V), typename T = Vec2Tag<V>, TL_IF(T::Bytes == 64), TL_IF(is_any<TypeOf<T>, int32_t>)>
VECOPS_VFUNC V abs(V v) {
  return _mm512_abs_epi32(v.v);
}
template <TLV_DECL_VEC(V), typename T = Vec2Tag<V>, TL_IF(T::Bytes == 64), TL_IF(is_any<TypeOf<T>, int64_t>)>
VECOPS_VFUNC V abs(V v) {
  return _mm512_abs_epi64(v.v);
}
template <TLV_DECL_VEC(V), typename T = Vec2Tag<V>, TL_IF(T::Bytes == 64), TL_IF(is_any<TypeOf<T>, float16_t, bfloat16_t>)>
VECOPS_VFUNC V abs(V v) { return _mm512_and_si512(v.v, _mm512_set1_epi16(0x7FFF)); }
#endif // VEC_WIDTH >= 512

#ifdef HAS_AVX512DQ
template <TLV_DECL_VEC(V), typename T = Vec2Tag<V>, TL_IF(is_int<TypeOf<T>> && std::is_unsigned_v<TypeOf<T>>)>
VECOPS_VFUNC V abs(V v, Mask<T> m, V default_v) {
  return word::blend(default_v, m, v);
}
template <TLV_DECL_VEC(V), typename T = Vec2Tag<V>, TL_IF(T::Bytes <= 16), TL_IF(is_any<TypeOf<T>, float32_t>)>
VECOPS_VFUNC V abs(V v, Mask<T> m, V default_v) {
  return _mm_mask_and_ps(default_v.v, m.v, _mm_castsi128_ps(_mm_set1_epi32(0x7FFFFFFF)), v.v);
}
template <TLV_DECL_VEC(V), typename T = Vec2Tag<V>, TL_IF(T::Bytes <= 16), TL_IF(is_any<TypeOf<T>, float64_t>)>
VECOPS_VFUNC V abs(V v, Mask<T> m, V default_v) {
  return _mm_mask_and_pd(default_v.v, m.v, _mm_castsi128_pd(_mm_set1_epi64x(0x7FFFFFFFFFFFFFFFLL)), v.v);
}
template <TLV_DECL_VEC(V), typename T = Vec2Tag<V>, TL_IF(T::Bytes <= 16), TL_IF(is_any<TypeOf<T>, int8_t>)>
VECOPS_VFUNC V abs(V v, Mask<T> m, V default_v) {
  return _mm_mask_abs_epi8(default_v.v, m.v, v.v);
}
template <TLV_DECL_VEC(V), typename T = Vec2Tag<V>, TL_IF(T::Bytes <= 16), TL_IF(is_any<TypeOf<T>, int16_t>)>
VECOPS_VFUNC V abs(V v, Mask<T> m, V default_v) {
  return _mm_mask_abs_epi16(default_v.v, m.v, v.v);
}
template <TLV_DECL_VEC(V), typename T = Vec2Tag<V>, TL_IF(T::Bytes <= 16), TL_IF(is_any<TypeOf<T>, int32_t>)>
VECOPS_VFUNC V abs(V v, Mask<T> m, V default_v) {
  return _mm_mask_abs_epi32(default_v.v, m.v, v.v);
}
template <TLV_DECL_VEC(V), typename T = Vec2Tag<V>, TL_IF(T::Bytes <= 16), TL_IF(is_any<TypeOf<T>, int64_t>)>
VECOPS_VFUNC V abs(V v, Mask<T> m, V default_v) {
  return _mm_mask_abs_epi64(default_v.v, m.v, v.v);
}

template <TLV_DECL_VEC(V), typename T = Vec2Tag<V>, TL_IF(T::Bytes == 32), TL_IF(is_any<TypeOf<T>, float32_t>)>
VECOPS_VFUNC V abs(V v, Mask<T> m, V default_v) {
  return _mm256_mask_and_ps(default_v.v, m.v, _mm256_castsi256_ps(_mm256_set1_epi32(0x7FFFFFFF)), v.v);
}
template <TLV_DECL_VEC(V), typename T = Vec2Tag<V>, TL_IF(T::Bytes == 32), TL_IF(is_any<TypeOf<T>, float64_t>)>
VECOPS_VFUNC V abs(V v, Mask<T> m, V default_v) {
  return _mm256_mask_and_pd(default_v.v, m.v, _mm256_castsi256_pd(_mm256_set1_epi64x(0x7FFFFFFFFFFFFFFFLL)), v.v);
}
template <TLV_DECL_VEC(V), typename T = Vec2Tag<V>, TL_IF(T::Bytes == 32), TL_IF(is_any<TypeOf<T>, int8_t>)>
VECOPS_VFUNC V abs(V v, Mask<T> m, V default_v) {
  return _mm256_mask_abs_epi8(default_v.v, m.v, v.v);
}
template <TLV_DECL_VEC(V), typename T = Vec2Tag<V>, TL_IF(T::Bytes == 32), TL_IF(is_any<TypeOf<T>, int16_t>)>
VECOPS_VFUNC V abs(V v, Mask<T> m, V default_v) {
  return _mm256_mask_abs_epi16(default_v.v, m.v, v.v);
}
template <TLV_DECL_VEC(V), typename T = Vec2Tag<V>, TL_IF(T::Bytes == 32), TL_IF(is_any<TypeOf<T>, int32_t>)>
VECOPS_VFUNC V abs(V v, Mask<T> m, V default_v) {
  return _mm256_mask_abs_epi32(default_v.v, m.v, v.v);
}
template <TLV_DECL_VEC(V), typename T = Vec2Tag<V>, TL_IF(T::Bytes == 32), TL_IF(is_any<TypeOf<T>, int64_t>)>
VECOPS_VFUNC V abs(V v, Mask<T> m, V default_v) {
  return _mm256_mask_abs_epi64(default_v.v, m.v, v.v);
}

template <TLV_DECL_VEC(V), typename T = Vec2Tag<V>, TL_IF(T::Bytes == 64), TL_IF(is_any<TypeOf<T>, float32_t>)>
VECOPS_VFUNC V abs(V v, Mask<T> m, V default_v) {
  return _mm512_mask_and_ps(default_v.v, m.v, _mm512_castsi512_ps(_mm512_set1_epi32(0x7FFFFFFF)), v.v);
}
template <TLV_DECL_VEC(V), typename T = Vec2Tag<V>, TL_IF(T::Bytes == 64), TL_IF(is_any<TypeOf<T>, float64_t>)>
VECOPS_VFUNC V abs(V v, Mask<T> m, V default_v) {
  return _mm512_mask_and_pd(default_v.v, m.v, _mm512_castsi512_pd(_mm512_set1_epi64(0x7FFFFFFFFFFFFFFFLL)), v.v);
}
template <TLV_DECL_VEC(V), typename T = Vec2Tag<V>, TL_IF(T::Bytes == 64), TL_IF(is_any<TypeOf<T>, int8_t>)>
VECOPS_VFUNC V abs(V v, Mask<T> m, V default_v) {
  return _mm512_mask_abs_epi8(default_v.v, m.v, v.v);
}
template <TLV_DECL_VEC(V), typename T = Vec2Tag<V>, TL_IF(T::Bytes == 64), TL_IF(is_any<TypeOf<T>, int16_t>)>
VECOPS_VFUNC V abs(V v, Mask<T> m, V default_v) {
  return _mm512_mask_abs_epi16(default_v.v, m.v, v.v);
}
template <TLV_DECL_VEC(V), typename T = Vec2Tag<V>, TL_IF(T::Bytes == 64), TL_IF(is_any<TypeOf<T>, int32_t>)>
VECOPS_VFUNC V abs(V v, Mask<T> m, V default_v) {
  return _mm512_mask_abs_epi32(default_v.v, m.v, v.v);
}
template <TLV_DECL_VEC(V), typename T = Vec2Tag<V>, TL_IF(T::Bytes == 64), TL_IF(is_any<TypeOf<T>, int64_t>)>
VECOPS_VFUNC V abs(V v, Mask<T> m, V default_v) {
  return _mm512_mask_abs_epi64(default_v.v, m.v, v.v);
}
template <TLV_DECL_VEC(V), typename T = Vec2Tag<V>, TL_IF(is_any<TypeOf<T>, float16_t, bfloat16_t>)>
VECOPS_VFUNC V abs(V v, Mask<T> m, V default_v) {
  return word::blend(default_v, m, word::abs(v));
}
#else // HAS_AVX512DQ
template <TLV_DECL_VEC(V), typename T = Vec2Tag<V>>
VECOPS_VFUNC V abs(V v, Mask<T> m, V default_v) {
  return word::blend(default_v, m, word::abs(v));
}
#endif // HAS_AVX512DQ


/* ************************************************************************** */
//                              Compare Equals                                //
/* ************************************************************************** */
#ifdef HAS_AVX512DQ
template <TLV_DECL_VEC(V), typename T = Vec2Tag<V>, TL_IF(T::Bytes <= 16), TL_IF(is_any<TypeOf<T>, float32_t>)>
VECOPS_VFUNC Mask<T> cmpeq(V a, V b) {
  return _mm_cmp_ps_mask(a.v, b.v, _CMP_EQ_OQ);
}
template <TLV_DECL_VEC(V), typename T = Vec2Tag<V>, TL_IF(T::Bytes <= 16), TL_IF(is_any<TypeOf<T>, float64_t>)>
VECOPS_VFUNC Mask<T> cmpeq(V a, V b) {
  return _mm_cmp_pd_mask(a.v, b.v, _CMP_EQ_OQ);
}
template <TLV_DECL_VEC(V), typename T = Vec2Tag<V>, TL_IF(T::Bytes <= 16), TL_IF(is_any<TypeOf<T>, int8_t>)>
VECOPS_VFUNC Mask<T> cmpeq(V a, V b) {
  return _mm_cmp_epi8_mask(a.v, b.v, _MM_CMPINT_EQ);
}
template <TLV_DECL_VEC(V), typename T = Vec2Tag<V>, TL_IF(T::Bytes <= 16), TL_IF(is_any<TypeOf<T>, uint8_t>)>
VECOPS_VFUNC Mask<T> cmpeq(V a, V b) {
  return _mm_cmp_epu8_mask(a.v, b.v, _MM_CMPINT_EQ);
}
template <TLV_DECL_VEC(V), typename T = Vec2Tag<V>, TL_IF(T::Bytes <= 16), TL_IF(is_any<TypeOf<T>, int16_t>)>
VECOPS_VFUNC Mask<T> cmpeq(V a, V b) {
  return _mm_cmp_epi16_mask(a.v, b.v, _MM_CMPINT_EQ);
}
template <TLV_DECL_VEC(V), typename T = Vec2Tag<V>, TL_IF(T::Bytes <= 16), TL_IF(is_any<TypeOf<T>, uint16_t>)>
VECOPS_VFUNC Mask<T> cmpeq(V a, V b) {
  return _mm_cmp_epu16_mask(a.v, b.v, _MM_CMPINT_EQ);
}
template <TLV_DECL_VEC(V), typename T = Vec2Tag<V>, TL_IF(T::Bytes <= 16), TL_IF(is_any<TypeOf<T>, int32_t>)>
VECOPS_VFUNC Mask<T> cmpeq(V a, V b) {
  return _mm_cmp_epi32_mask(a.v, b.v, _MM_CMPINT_EQ);
}
template <TLV_DECL_VEC(V), typename T = Vec2Tag<V>, TL_IF(T::Bytes <= 16), TL_IF(is_any<TypeOf<T>, uint32_t>)>
VECOPS_VFUNC Mask<T> cmpeq(V a, V b) {
  return _mm_cmp_epu32_mask(a.v, b.v, _MM_CMPINT_EQ);
}
template <TLV_DECL_VEC(V), typename T = Vec2Tag<V>, TL_IF(T::Bytes <= 16), TL_IF(is_any<TypeOf<T>, int64_t>)>
VECOPS_VFUNC Mask<T> cmpeq(V a, V b) {
  return _mm_cmp_epi64_mask(a.v, b.v, _MM_CMPINT_EQ);
}
template <TLV_DECL_VEC(V), typename T = Vec2Tag<V>, TL_IF(T::Bytes <= 16), TL_IF(is_any<TypeOf<T>, uint64_t>)>
VECOPS_VFUNC Mask<T> cmpeq(V a, V b) {
  return _mm_cmp_epu64_mask(a.v, b.v, _MM_CMPINT_EQ);
}

template <TLV_DECL_VEC(V), typename T = Vec2Tag<V>, TL_IF(T::Bytes == 32), TL_IF(is_any<TypeOf<T>, float32_t>)>
VECOPS_VFUNC Mask<T> cmpeq(V a, V b) {
  return _mm256_cmp_ps_mask(a.v, b.v, _CMP_EQ_OQ);
}
template <TLV_DECL_VEC(V), typename T = Vec2Tag<V>, TL_IF(T::Bytes == 32), TL_IF(is_any<TypeOf<T>, float64_t>)>
VECOPS_VFUNC Mask<T> cmpeq(V a, V b) {
  return _mm256_cmp_pd_mask(a.v, b.v, _CMP_EQ_OQ);
}
template <TLV_DECL_VEC(V), typename T = Vec2Tag<V>, TL_IF(T::Bytes == 32), TL_IF(is_any<TypeOf<T>, int8_t>)>
VECOPS_VFUNC Mask<T> cmpeq(V a, V b) {
  return _mm256_cmp_epi8_mask(a.v, b.v, _MM_CMPINT_EQ);
}
template <TLV_DECL_VEC(V), typename T = Vec2Tag<V>, TL_IF(T::Bytes == 32), TL_IF(is_any<TypeOf<T>, uint8_t>)>
VECOPS_VFUNC Mask<T> cmpeq(V a, V b) {
  return _mm256_cmp_epu8_mask(a.v, b.v, _MM_CMPINT_EQ);
}
template <TLV_DECL_VEC(V), typename T = Vec2Tag<V>, TL_IF(T::Bytes == 32), TL_IF(is_any<TypeOf<T>, int16_t>)>
VECOPS_VFUNC Mask<T> cmpeq(V a, V b) {
  return _mm256_cmp_epi16_mask(a.v, b.v, _MM_CMPINT_EQ);
}
template <TLV_DECL_VEC(V), typename T = Vec2Tag<V>, TL_IF(T::Bytes == 32), TL_IF(is_any<TypeOf<T>, uint16_t>)>
VECOPS_VFUNC Mask<T> cmpeq(V a, V b) {
  return _mm256_cmp_epu16_mask(a.v, b.v, _MM_CMPINT_EQ);
}
template <TLV_DECL_VEC(V), typename T = Vec2Tag<V>, TL_IF(T::Bytes == 32), TL_IF(is_any<TypeOf<T>, int32_t>)>
VECOPS_VFUNC Mask<T> cmpeq(V a, V b) {
  return _mm256_cmp_epi32_mask(a.v, b.v, _MM_CMPINT_EQ);
}
template <TLV_DECL_VEC(V), typename T = Vec2Tag<V>, TL_IF(T::Bytes == 32), TL_IF(is_any<TypeOf<T>, uint32_t>)>
VECOPS_VFUNC Mask<T> cmpeq(V a, V b) {
  return _mm256_cmp_epu32_mask(a.v, b.v, _MM_CMPINT_EQ);
}
template <TLV_DECL_VEC(V), typename T = Vec2Tag<V>, TL_IF(T::Bytes == 32), TL_IF(is_any<TypeOf<T>, int64_t>)>
VECOPS_VFUNC Mask<T> cmpeq(V a, V b) {
  return _mm256_cmp_epi64_mask(a.v, b.v, _MM_CMPINT_EQ);
}
template <TLV_DECL_VEC(V), typename T = Vec2Tag<V>, TL_IF(T::Bytes == 32), TL_IF(is_any<TypeOf<T>, uint64_t>)>
VECOPS_VFUNC Mask<T> cmpeq(V a, V b) {
  return _mm256_cmp_epu64_mask(a.v, b.v, _MM_CMPINT_EQ);
}

template <TLV_DECL_VEC(V), typename T = Vec2Tag<V>, TL_IF(T::Bytes == 64), TL_IF(is_any<TypeOf<T>, float32_t>)>
VECOPS_VFUNC Mask<T> cmpeq(V a, V b) {
  return _mm512_cmp_ps_mask(a.v, b.v, _CMP_EQ_OQ);
}
template <TLV_DECL_VEC(V), typename T = Vec2Tag<V>, TL_IF(T::Bytes == 64), TL_IF(is_any<TypeOf<T>, float64_t>)>
VECOPS_VFUNC Mask<T> cmpeq(V a, V b) {
  return _mm512_cmp_pd_mask(a.v, b.v, _CMP_EQ_OQ);
}
template <TLV_DECL_VEC(V), typename T = Vec2Tag<V>, TL_IF(T::Bytes == 64), TL_IF(is_any<TypeOf<T>, int8_t>)>
VECOPS_VFUNC Mask<T> cmpeq(V a, V b) {
  return _mm512_cmp_epi8_mask(a.v, b.v, _MM_CMPINT_EQ);
}
template <TLV_DECL_VEC(V), typename T = Vec2Tag<V>, TL_IF(T::Bytes == 64), TL_IF(is_any<TypeOf<T>, uint8_t>)>
VECOPS_VFUNC Mask<T> cmpeq(V a, V b) {
  return _mm512_cmp_epu8_mask(a.v, b.v, _MM_CMPINT_EQ);
}
template <TLV_DECL_VEC(V), typename T = Vec2Tag<V>, TL_IF(T::Bytes == 64), TL_IF(is_any<TypeOf<T>, int16_t>)>
VECOPS_VFUNC Mask<T> cmpeq(V a, V b) {
  return _mm512_cmp_epi16_mask(a.v, b.v, _MM_CMPINT_EQ);
}
template <TLV_DECL_VEC(V), typename T = Vec2Tag<V>, TL_IF(T::Bytes == 64), TL_IF(is_any<TypeOf<T>, uint16_t>)>
VECOPS_VFUNC Mask<T> cmpeq(V a, V b) {
  return _mm512_cmp_epu16_mask(a.v, b.v, _MM_CMPINT_EQ);
}
template <TLV_DECL_VEC(V), typename T = Vec2Tag<V>, TL_IF(T::Bytes == 64), TL_IF(is_any<TypeOf<T>, int32_t>)>
VECOPS_VFUNC Mask<T> cmpeq(V a, V b) {
  return _mm512_cmp_epi32_mask(a.v, b.v, _MM_CMPINT_EQ);
}
template <TLV_DECL_VEC(V), typename T = Vec2Tag<V>, TL_IF(T::Bytes == 64), TL_IF(is_any<TypeOf<T>, uint32_t>)>
VECOPS_VFUNC Mask<T> cmpeq(V a, V b) {
  return _mm512_cmp_epu32_mask(a.v, b.v, _MM_CMPINT_EQ);
}
template <TLV_DECL_VEC(V), typename T = Vec2Tag<V>, TL_IF(T::Bytes == 64), TL_IF(is_any<TypeOf<T>, int64_t>)>
VECOPS_VFUNC Mask<T> cmpeq(V a, V b) {
  return _mm512_cmp_epi64_mask(a.v, b.v, _MM_CMPINT_EQ);
}
template <TLV_DECL_VEC(V), typename T = Vec2Tag<V>, TL_IF(T::Bytes == 64), TL_IF(is_any<TypeOf<T>, uint64_t>)>
VECOPS_VFUNC Mask<T> cmpeq(V a, V b) {
  return _mm512_cmp_epu64_mask(a.v, b.v, _MM_CMPINT_EQ);
}
#else // HAS_AVX512DQ
template <TLV_DECL_VEC(V), typename T = Vec2Tag<V>, TL_IF(T::Bytes <= 16), TL_IF(is_any<TypeOf<T>, float32_t>)>
VECOPS_VFUNC Mask<T> cmpeq(V a, V b) {
  return _mm_castps_si128(_mm_cmpeq_ps(a.v, b.v));
}
template <TLV_DECL_VEC(V), typename T = Vec2Tag<V>, TL_IF(T::Bytes <= 16), TL_IF(is_any<TypeOf<T>, float64_t>)>
VECOPS_VFUNC Mask<T> cmpeq(V a, V b) {
  return _mm_castpd_si128(_mm_cmpeq_pd(a.v, b.v));
}
template <TLV_DECL_VEC(V), typename T = Vec2Tag<V>, TL_IF(T::Bytes <= 16), TL_IF(is_any<TypeOf<T>, int8_t, uint8_t>)>
VECOPS_VFUNC Mask<T> cmpeq(V a, V b) {
  return _mm_cmpeq_epi8(a.v, b.v);
}
template <TLV_DECL_VEC(V), typename T = Vec2Tag<V>, TL_IF(T::Bytes <= 16), TL_IF(is_any<TypeOf<T>, int16_t, uint16_t>)>
VECOPS_VFUNC Mask<T> cmpeq(V a, V b) {
  return _mm_cmpeq_epi16(a.v, b.v);
}
template <TLV_DECL_VEC(V), typename T = Vec2Tag<V>, TL_IF(T::Bytes <= 16), TL_IF(is_any<TypeOf<T>, int32_t, uint32_t>)>
VECOPS_VFUNC Mask<T> cmpeq(V a, V b) {
  return _mm_cmpeq_epi32(a.v, b.v);
}
template <TLV_DECL_VEC(V), typename T = Vec2Tag<V>, TL_IF(T::Bytes <= 16), TL_IF(is_any<TypeOf<T>, int64_t, uint64_t>)>
VECOPS_VFUNC Mask<T> cmpeq(V a, V b) {
  return _mm_cmpeq_epi64(a.v, b.v);
}

#if VEC_WIDTH >= 256
template <TLV_DECL_VEC(V), typename T = Vec2Tag<V>, TL_IF(T::Bytes == 32), TL_IF(is_any<TypeOf<T>, float32_t>)>
VECOPS_VFUNC Mask<T> cmpeq(V a, V b) {
  return _mm256_castps_si256(_mm256_cmp_ps(a.v, b.v, _CMP_EQ_OQ));
}
template <TLV_DECL_VEC(V), typename T = Vec2Tag<V>, TL_IF(T::Bytes == 32), TL_IF(is_any<TypeOf<T>, float64_t>)>
VECOPS_VFUNC Mask<T> cmpeq(V a, V b) {
  return _mm256_castpd_si256(_mm256_cmp_pd(a.v, b.v, _CMP_EQ_OQ));
}
template <TLV_DECL_VEC(V), typename T = Vec2Tag<V>, TL_IF(T::Bytes == 32), TL_IF(is_any<TypeOf<T>, int8_t, uint8_t>)>
VECOPS_VFUNC Mask<T> cmpeq(V a, V b) {
  return _mm256_cmpeq_epi8(a.v, b.v);
}
template <TLV_DECL_VEC(V), typename T = Vec2Tag<V>, TL_IF(T::Bytes == 32), TL_IF(is_any<TypeOf<T>, int16_t, uint16_t>)>
VECOPS_VFUNC Mask<T> cmpeq(V a, V b) {
  return _mm256_cmpeq_epi16(a.v, b.v);
}
template <TLV_DECL_VEC(V), typename T = Vec2Tag<V>, TL_IF(T::Bytes == 32), TL_IF(is_any<TypeOf<T>, int32_t, uint32_t>)>
VECOPS_VFUNC Mask<T> cmpeq(V a, V b) {
  return _mm256_cmpeq_epi32(a.v, b.v);
}
template <TLV_DECL_VEC(V), typename T = Vec2Tag<V>, TL_IF(T::Bytes == 32), TL_IF(is_any<TypeOf<T>, int64_t, uint64_t>)>
VECOPS_VFUNC Mask<T> cmpeq(V a, V b) {
  return _mm256_cmpeq_epi64(a.v, b.v);
}
#endif // VEC_WIDTH >= 256

#if VEC_WIDTH >= 512
  #error "Unreachable"
#endif // VEC_WIDTH >= 512

template <TLV_DECL_VEC(V), typename T = Vec2Tag<V>, TL_IF(T::Bytes <= 16), TL_IF(is_any<TypeOf<T>, float16_t>)>
VECOPS_VFUNC Mask<T> cmpeq(V a, V b) {
  __m128 a_lo, a_hi, b_lo, b_hi;
  half_cvt::cvt_fp16_to_two_fp32(a.v, a_lo, a_hi);
  half_cvt::cvt_fp16_to_two_fp32(b.v, b_lo, b_hi);
  return _mm_packs_epi32(
      _mm_castps_si128(_mm_cmpeq_ps(a_lo, b_lo)),
      _mm_castps_si128(_mm_cmpeq_ps(a_hi, b_hi)));
}
template <TLV_DECL_VEC(V), typename T = Vec2Tag<V>, TL_IF(T::Bytes <= 16), TL_IF(is_any<TypeOf<T>, bfloat16_t>)>
VECOPS_VFUNC Mask<T> cmpeq(V a, V b) {
  __m128 a_lo, a_hi, b_lo, b_hi;
  half_cvt::cvt_bf16_to_two_fp32(a.v, a_lo, a_hi);
  half_cvt::cvt_bf16_to_two_fp32(b.v, b_lo, b_hi);
  return _mm_packs_epi32(
      _mm_castps_si128(_mm_cmpeq_ps(a_lo, b_lo)),
      _mm_castps_si128(_mm_cmpeq_ps(a_hi, b_hi)));
}
#if VEC_WIDTH >= 256
template <TLV_DECL_VEC(V), typename T = Vec2Tag<V>, TL_IF(T::Bytes == 32), TL_IF(is_any<TypeOf<T>, float16_t>)>
VECOPS_VFUNC Mask<T> cmpeq(V a, V b) {
  __m256 a_lo, a_hi, b_lo, b_hi;
  half_cvt::cvt_fp16_to_two_fp32(a.v, a_lo, a_hi);
  half_cvt::cvt_fp16_to_two_fp32(b.v, b_lo, b_hi);
  return _mm256_permute4x64_epi64(_mm256_packs_epi32(
      _mm256_castps_si256(_mm256_cmp_ps(a_lo, b_lo, _CMP_EQ_OQ)),
      _mm256_castps_si256(_mm256_cmp_ps(a_hi, b_hi, _CMP_EQ_OQ))), _MM_SHUFFLE(3, 1, 2, 0));
}
template <TLV_DECL_VEC(V), typename T = Vec2Tag<V>, TL_IF(T::Bytes == 32), TL_IF(is_any<TypeOf<T>, bfloat16_t>)>
VECOPS_VFUNC Mask<T> cmpeq(V a, V b) {
  __m256 a_lo, a_hi, b_lo, b_hi;
  half_cvt::cvt_bf16_to_two_fp32(a.v, a_lo, a_hi);
  half_cvt::cvt_bf16_to_two_fp32(b.v, b_lo, b_hi);
  return _mm256_permute4x64_epi64(_mm256_packs_epi32(
      _mm256_castps_si256(_mm256_cmp_ps(a_lo, b_lo, _CMP_EQ_OQ)),
      _mm256_castps_si256(_mm256_cmp_ps(a_hi, b_hi, _CMP_EQ_OQ))), _MM_SHUFFLE(3, 1, 2, 0));
}
#endif
#endif // HAS_AVX512DQ

#ifdef HAS_AVX512DQ
template <TLV_DECL_VEC(V), typename T = Vec2Tag<V>, TL_IF(T::Bytes <= 16), TL_IF(is_any<TypeOf<T>, float32_t>)>
VECOPS_VFUNC Mask<T> cmpeq(V a, V b, Mask<T> m) {
  return _mm_mask_cmp_ps_mask(m.v, a.v, b.v, _CMP_EQ_OQ);
}
template <TLV_DECL_VEC(V), typename T = Vec2Tag<V>, TL_IF(T::Bytes <= 16), TL_IF(is_any<TypeOf<T>, float64_t>)>
VECOPS_VFUNC Mask<T> cmpeq(V a, V b, Mask<T> m) {
  return _mm_mask_cmp_pd_mask(m.v, a.v, b.v, _CMP_EQ_OQ);
}
template <TLV_DECL_VEC(V), typename T = Vec2Tag<V>, TL_IF(T::Bytes <= 16), TL_IF(is_any<TypeOf<T>, int8_t>)>
VECOPS_VFUNC Mask<T> cmpeq(V a, V b, Mask<T> m) {
  return _mm_mask_cmp_epi8_mask(m.v, a.v, b.v, _MM_CMPINT_EQ);
}
template <TLV_DECL_VEC(V), typename T = Vec2Tag<V>, TL_IF(T::Bytes <= 16), TL_IF(is_any<TypeOf<T>, uint8_t>)>
VECOPS_VFUNC Mask<T> cmpeq(V a, V b, Mask<T> m) {
  return _mm_mask_cmp_epu8_mask(m.v, a.v, b.v, _MM_CMPINT_EQ);
}
template <TLV_DECL_VEC(V), typename T = Vec2Tag<V>, TL_IF(T::Bytes <= 16), TL_IF(is_any<TypeOf<T>, int16_t>)>
VECOPS_VFUNC Mask<T> cmpeq(V a, V b, Mask<T> m) {
  return _mm_mask_cmp_epi16_mask(m.v, a.v, b.v, _MM_CMPINT_EQ);
}
template <TLV_DECL_VEC(V), typename T = Vec2Tag<V>, TL_IF(T::Bytes <= 16), TL_IF(is_any<TypeOf<T>, uint16_t>)>
VECOPS_VFUNC Mask<T> cmpeq(V a, V b, Mask<T> m) {
  return _mm_mask_cmp_epu16_mask(m.v, a.v, b.v, _MM_CMPINT_EQ);
}
template <TLV_DECL_VEC(V), typename T = Vec2Tag<V>, TL_IF(T::Bytes <= 16), TL_IF(is_any<TypeOf<T>, int32_t>)>
VECOPS_VFUNC Mask<T> cmpeq(V a, V b, Mask<T> m) {
  return _mm_mask_cmp_epi32_mask(m.v, a.v, b.v, _MM_CMPINT_EQ);
}
template <TLV_DECL_VEC(V), typename T = Vec2Tag<V>, TL_IF(T::Bytes <= 16), TL_IF(is_any<TypeOf<T>, uint32_t>)>
VECOPS_VFUNC Mask<T> cmpeq(V a, V b, Mask<T> m) {
  return _mm_mask_cmp_epu32_mask(m.v, a.v, b.v, _MM_CMPINT_EQ);
}
template <TLV_DECL_VEC(V), typename T = Vec2Tag<V>, TL_IF(T::Bytes <= 16), TL_IF(is_any<TypeOf<T>, int64_t>)>
VECOPS_VFUNC Mask<T> cmpeq(V a, V b, Mask<T> m) {
  return _mm_mask_cmp_epi64_mask(m.v, a.v, b.v, _MM_CMPINT_EQ);
}
template <TLV_DECL_VEC(V), typename T = Vec2Tag<V>, TL_IF(T::Bytes <= 16), TL_IF(is_any<TypeOf<T>, uint64_t>)>
VECOPS_VFUNC Mask<T> cmpeq(V a, V b, Mask<T> m) {
  return _mm_mask_cmp_epu64_mask(m.v, a.v, b.v, _MM_CMPINT_EQ);
}

template <TLV_DECL_VEC(V), typename T = Vec2Tag<V>, TL_IF(T::Bytes == 32), TL_IF(is_any<TypeOf<T>, float32_t>)>
VECOPS_VFUNC Mask<T> cmpeq(V a, V b, Mask<T> m) {
  return _mm256_mask_cmp_ps_mask(m.v, a.v, b.v, _CMP_EQ_OQ);
}
template <TLV_DECL_VEC(V), typename T = Vec2Tag<V>, TL_IF(T::Bytes == 32), TL_IF(is_any<TypeOf<T>, float64_t>)>
VECOPS_VFUNC Mask<T> cmpeq(V a, V b, Mask<T> m) {
  return _mm256_mask_cmp_pd_mask(m.v, a.v, b.v, _CMP_EQ_OQ);
}
template <TLV_DECL_VEC(V), typename T = Vec2Tag<V>, TL_IF(T::Bytes == 32), TL_IF(is_any<TypeOf<T>, int8_t>)>
VECOPS_VFUNC Mask<T> cmpeq(V a, V b, Mask<T> m) {
  return _mm256_mask_cmp_epi8_mask(m.v, a.v, b.v, _MM_CMPINT_EQ);
}
template <TLV_DECL_VEC(V), typename T = Vec2Tag<V>, TL_IF(T::Bytes == 32), TL_IF(is_any<TypeOf<T>, uint8_t>)>
VECOPS_VFUNC Mask<T> cmpeq(V a, V b, Mask<T> m) {
  return _mm256_mask_cmp_epu8_mask(m.v, a.v, b.v, _MM_CMPINT_EQ);
}
template <TLV_DECL_VEC(V), typename T = Vec2Tag<V>, TL_IF(T::Bytes == 32), TL_IF(is_any<TypeOf<T>, int16_t>)>
VECOPS_VFUNC Mask<T> cmpeq(V a, V b, Mask<T> m) {
  return _mm256_mask_cmp_epi16_mask(m.v, a.v, b.v, _MM_CMPINT_EQ);
}
template <TLV_DECL_VEC(V), typename T = Vec2Tag<V>, TL_IF(T::Bytes == 32), TL_IF(is_any<TypeOf<T>, uint16_t>)>
VECOPS_VFUNC Mask<T> cmpeq(V a, V b, Mask<T> m) {
  return _mm256_mask_cmp_epu16_mask(m.v, a.v, b.v, _MM_CMPINT_EQ);
}
template <TLV_DECL_VEC(V), typename T = Vec2Tag<V>, TL_IF(T::Bytes == 32), TL_IF(is_any<TypeOf<T>, int32_t>)>
VECOPS_VFUNC Mask<T> cmpeq(V a, V b, Mask<T> m) {
  return _mm256_mask_cmp_epi32_mask(m.v, a.v, b.v, _MM_CMPINT_EQ);
}
template <TLV_DECL_VEC(V), typename T = Vec2Tag<V>, TL_IF(T::Bytes == 32), TL_IF(is_any<TypeOf<T>, uint32_t>)>
VECOPS_VFUNC Mask<T> cmpeq(V a, V b, Mask<T> m) {
  return _mm256_mask_cmp_epu32_mask(m.v, a.v, b.v, _MM_CMPINT_EQ);
}
template <TLV_DECL_VEC(V), typename T = Vec2Tag<V>, TL_IF(T::Bytes == 32), TL_IF(is_any<TypeOf<T>, int64_t>)>
VECOPS_VFUNC Mask<T> cmpeq(V a, V b, Mask<T> m) {
  return _mm256_mask_cmp_epi64_mask(m.v, a.v, b.v, _MM_CMPINT_EQ);
}
template <TLV_DECL_VEC(V), typename T = Vec2Tag<V>, TL_IF(T::Bytes == 32), TL_IF(is_any<TypeOf<T>, uint64_t>)>
VECOPS_VFUNC Mask<T> cmpeq(V a, V b, Mask<T> m) {
  return _mm256_mask_cmp_epu64_mask(m.v, a.v, b.v, _MM_CMPINT_EQ);
}

template <TLV_DECL_VEC(V), typename T = Vec2Tag<V>, TL_IF(T::Bytes == 64), TL_IF(is_any<TypeOf<T>, float32_t>)>
VECOPS_VFUNC Mask<T> cmpeq(V a, V b, Mask<T> m) {
  return _mm512_mask_cmp_ps_mask(m.v, a.v, b.v, _CMP_EQ_OQ);
}
template <TLV_DECL_VEC(V), typename T = Vec2Tag<V>, TL_IF(T::Bytes == 64), TL_IF(is_any<TypeOf<T>, float64_t>)>
VECOPS_VFUNC Mask<T> cmpeq(V a, V b, Mask<T> m) {
  return _mm512_mask_cmp_pd_mask(m.v, a.v, b.v, _CMP_EQ_OQ);
}
template <TLV_DECL_VEC(V), typename T = Vec2Tag<V>, TL_IF(T::Bytes == 64), TL_IF(is_any<TypeOf<T>, int8_t>)>
VECOPS_VFUNC Mask<T> cmpeq(V a, V b, Mask<T> m) {
  return _mm512_mask_cmp_epi8_mask(m.v, a.v, b.v, _MM_CMPINT_EQ);
}
template <TLV_DECL_VEC(V), typename T = Vec2Tag<V>, TL_IF(T::Bytes == 64), TL_IF(is_any<TypeOf<T>, uint8_t>)>
VECOPS_VFUNC Mask<T> cmpeq(V a, V b, Mask<T> m) {
  return _mm512_mask_cmp_epu8_mask(m.v, a.v, b.v, _MM_CMPINT_EQ);
}
template <TLV_DECL_VEC(V), typename T = Vec2Tag<V>, TL_IF(T::Bytes == 64), TL_IF(is_any<TypeOf<T>, int16_t>)>
VECOPS_VFUNC Mask<T> cmpeq(V a, V b, Mask<T> m) {
  return _mm512_mask_cmp_epi16_mask(m.v, a.v, b.v, _MM_CMPINT_EQ);
}
template <TLV_DECL_VEC(V), typename T = Vec2Tag<V>, TL_IF(T::Bytes == 64), TL_IF(is_any<TypeOf<T>, uint16_t>)>
VECOPS_VFUNC Mask<T> cmpeq(V a, V b, Mask<T> m) {
  return _mm512_mask_cmp_epu16_mask(m.v, a.v, b.v, _MM_CMPINT_EQ);
}
template <TLV_DECL_VEC(V), typename T = Vec2Tag<V>, TL_IF(T::Bytes == 64), TL_IF(is_any<TypeOf<T>, int32_t>)>
VECOPS_VFUNC Mask<T> cmpeq(V a, V b, Mask<T> m) {
  return _mm512_mask_cmp_epi32_mask(m.v, a.v, b.v, _MM_CMPINT_EQ);
}
template <TLV_DECL_VEC(V), typename T = Vec2Tag<V>, TL_IF(T::Bytes == 64), TL_IF(is_any<TypeOf<T>, uint32_t>)>
VECOPS_VFUNC Mask<T> cmpeq(V a, V b, Mask<T> m) {
  return _mm512_mask_cmp_epu32_mask(m.v, a.v, b.v, _MM_CMPINT_EQ);
}
template <TLV_DECL_VEC(V), typename T = Vec2Tag<V>, TL_IF(T::Bytes == 64), TL_IF(is_any<TypeOf<T>, int64_t>)>
VECOPS_VFUNC Mask<T> cmpeq(V a, V b, Mask<T> m) {
  return _mm512_mask_cmp_epi64_mask(m.v, a.v, b.v, _MM_CMPINT_EQ);
}
template <TLV_DECL_VEC(V), typename T = Vec2Tag<V>, TL_IF(T::Bytes == 64), TL_IF(is_any<TypeOf<T>, uint64_t>)>
VECOPS_VFUNC Mask<T> cmpeq(V a, V b, Mask<T> m) {
  return _mm512_mask_cmp_epu64_mask(m.v, a.v, b.v, _MM_CMPINT_EQ);
}
#ifdef HAS_AVX512_FP16
template <TLV_DECL_VEC(V), typename T = Vec2Tag<V>, TL_IF(T::Bytes <= 16), TL_IF(is_any<TypeOf<T>, float16_t>)>
VECOPS_VFUNC Mask<T> cmpeq(V a, V b) {
  return _mm_cmp_ph_mask(_mm_castsi128_ph(a.v), _mm_castsi128_ph(b.v), _CMP_EQ_OQ);
}
#endif
#ifndef HAS_AVX512_FP16
template <TLV_DECL_VEC(V), typename T = Vec2Tag<V>, TL_IF(T::Bytes <= 16), TL_IF(is_any<TypeOf<T>, float16_t>)>
VECOPS_VFUNC Mask<T> cmpeq(V a, V b) {
  __m128 a_lo, a_hi, b_lo, b_hi;
  half_cvt::cvt_fp16_to_two_fp32(a.v, a_lo, a_hi);
  half_cvt::cvt_fp16_to_two_fp32(b.v, b_lo, b_hi);
  __mmask8 lo = _mm_cmp_ps_mask(a_lo, b_lo, _CMP_EQ_OQ);
  __mmask8 hi = _mm_cmp_ps_mask(a_hi, b_hi, _CMP_EQ_OQ);
  return (hi << 4) | lo;
}
#endif
template <TLV_DECL_VEC(V), typename T = Vec2Tag<V>, TL_IF(T::Bytes <= 16), TL_IF(is_any<TypeOf<T>, bfloat16_t>)>
VECOPS_VFUNC Mask<T> cmpeq(V a, V b) {
  __m128 a_lo, a_hi, b_lo, b_hi;
  half_cvt::cvt_bf16_to_two_fp32(a.v, a_lo, a_hi);
  half_cvt::cvt_bf16_to_two_fp32(b.v, b_lo, b_hi);
  __mmask8 lo = _mm_cmp_ps_mask(a_lo, b_lo, _CMP_EQ_OQ);
  __mmask8 hi = _mm_cmp_ps_mask(a_hi, b_hi, _CMP_EQ_OQ);
  return (hi << 4) | lo;
}
#if VEC_WIDTH >= 256
#ifdef HAS_AVX512_FP16
template <TLV_DECL_VEC(V), typename T = Vec2Tag<V>, TL_IF(T::Bytes == 32), TL_IF(is_any<TypeOf<T>, float16_t>)>
VECOPS_VFUNC Mask<T> cmpeq(V a, V b) {
  return _mm256_cmp_ph_mask(_mm256_castsi256_ph(a.v), _mm256_castsi256_ph(b.v), _CMP_EQ_OQ);
}
#else
template <TLV_DECL_VEC(V), typename T = Vec2Tag<V>, TL_IF(T::Bytes == 32), TL_IF(is_any<TypeOf<T>, float16_t>)>
VECOPS_VFUNC Mask<T> cmpeq(V a, V b) {
  __m256 a_lo, a_hi, b_lo, b_hi;
  half_cvt::cvt_fp16_to_two_fp32(a.v, a_lo, a_hi);
  half_cvt::cvt_fp16_to_two_fp32(b.v, b_lo, b_hi);
  __mmask16 lo = _mm256_cmp_ps_mask(a_lo, b_lo, _CMP_EQ_OQ);
  __mmask16 hi = _mm256_cmp_ps_mask(a_hi, b_hi, _CMP_EQ_OQ);
  return (hi << 8) | lo;
}
#endif
template <TLV_DECL_VEC(V), typename T = Vec2Tag<V>, TL_IF(T::Bytes == 32), TL_IF(is_any<TypeOf<T>, bfloat16_t>)>
VECOPS_VFUNC Mask<T> cmpeq(V a, V b) {
  __m256 a_lo, a_hi, b_lo, b_hi;
  half_cvt::cvt_bf16_to_two_fp32(a.v, a_lo, a_hi);
  half_cvt::cvt_bf16_to_two_fp32(b.v, b_lo, b_hi);
  __mmask16 lo = _mm256_cmp_ps_mask(a_lo, b_lo, _CMP_EQ_OQ);
  __mmask16 hi = _mm256_cmp_ps_mask(a_hi, b_hi, _CMP_EQ_OQ);
  return (hi << 8) | lo;
}
#endif
#if VEC_WIDTH >= 512
#ifdef HAS_AVX512_FP16
template <TLV_DECL_VEC(V), typename T = Vec2Tag<V>, TL_IF(T::Bytes == 64), TL_IF(is_any<TypeOf<T>, float16_t>)>
VECOPS_VFUNC Mask<T> cmpeq(V a, V b) {
  return _mm512_cmp_ph_mask(_mm512_castsi512_ph(a.v), _mm512_castsi512_ph(b.v), _CMP_EQ_OQ);
}
#else
template <TLV_DECL_VEC(V), typename T = Vec2Tag<V>, TL_IF(T::Bytes == 64), TL_IF(is_any<TypeOf<T>, float16_t>)>
VECOPS_VFUNC Mask<T> cmpeq(V a, V b) {
  __m256i a_lo256 = _mm512_castsi512_si256(a.v);
  __m256i a_hi256 = _mm512_extracti64x4_epi64(a.v, 1);
  __m256i b_lo256 = _mm512_castsi512_si256(b.v);
  __m256i b_hi256 = _mm512_extracti64x4_epi64(b.v, 1);
  __m256 a_lo0, a_lo1, a_hi0, a_hi1, b_lo0, b_lo1, b_hi0, b_hi1;
  half_cvt::cvt_fp16_to_two_fp32(a_lo256, a_lo0, a_lo1);
  half_cvt::cvt_fp16_to_two_fp32(a_hi256, a_hi0, a_hi1);
  half_cvt::cvt_fp16_to_two_fp32(b_lo256, b_lo0, b_lo1);
  half_cvt::cvt_fp16_to_two_fp32(b_hi256, b_hi0, b_hi1);
  __mmask16 m0 = _mm256_cmp_ps_mask(a_lo0, b_lo0, _CMP_EQ_OQ);
  __mmask16 m1 = _mm256_cmp_ps_mask(a_lo1, b_lo1, _CMP_EQ_OQ);
  __mmask16 m2 = _mm256_cmp_ps_mask(a_hi0, b_hi0, _CMP_EQ_OQ);
  __mmask16 m3 = _mm256_cmp_ps_mask(a_hi1, b_hi1, _CMP_EQ_OQ);
  return __mmask32(m0) | (__mmask32(m1) << 8) | (__mmask32(m2) << 16) | (__mmask32(m3) << 24);
}
#endif
template <TLV_DECL_VEC(V), typename T = Vec2Tag<V>, TL_IF(T::Bytes == 64), TL_IF(is_any<TypeOf<T>, bfloat16_t>)>
VECOPS_VFUNC Mask<T> cmpeq(V a, V b) {
  __m256i a_lo256 = _mm512_castsi512_si256(a.v);
  __m256i a_hi256 = _mm512_extracti64x4_epi64(a.v, 1);
  __m256i b_lo256 = _mm512_castsi512_si256(b.v);
  __m256i b_hi256 = _mm512_extracti64x4_epi64(b.v, 1);
  __m256 a_lo0, a_lo1, a_hi0, a_hi1, b_lo0, b_lo1, b_hi0, b_hi1;
  half_cvt::cvt_bf16_to_two_fp32(a_lo256, a_lo0, a_lo1);
  half_cvt::cvt_bf16_to_two_fp32(a_hi256, a_hi0, a_hi1);
  half_cvt::cvt_bf16_to_two_fp32(b_lo256, b_lo0, b_lo1);
  half_cvt::cvt_bf16_to_two_fp32(b_hi256, b_hi0, b_hi1);
  __mmask16 m0 = _mm256_cmp_ps_mask(a_lo0, b_lo0, _CMP_EQ_OQ);
  __mmask16 m1 = _mm256_cmp_ps_mask(a_lo1, b_lo1, _CMP_EQ_OQ);
  __mmask16 m2 = _mm256_cmp_ps_mask(a_hi0, b_hi0, _CMP_EQ_OQ);
  __mmask16 m3 = _mm256_cmp_ps_mask(a_hi1, b_hi1, _CMP_EQ_OQ);
  return __mmask32(m0) | (__mmask32(m1) << 8) | (__mmask32(m2) << 16) | (__mmask32(m3) << 24);
}
#endif
#ifdef HAS_AVX512_FP16
template <TLV_DECL_VEC(V), typename T = Vec2Tag<V>, TL_IF(T::Bytes <= 16), TL_IF(is_any<TypeOf<T>, float16_t>)>
VECOPS_VFUNC Mask<T> cmpeq(V a, V b, Mask<T> m) {
  return _mm_mask_cmp_ph_mask(m.v, _mm_castsi128_ph(a.v), _mm_castsi128_ph(b.v), _CMP_EQ_OQ);
}
#endif
#ifndef HAS_AVX512_FP16
template <TLV_DECL_VEC(V), typename T = Vec2Tag<V>, TL_IF(T::Bytes <= 16), TL_IF(is_any<TypeOf<T>, float16_t>)>
VECOPS_VFUNC Mask<T> cmpeq(V a, V b, Mask<T> m) {
  __m128 a_lo, a_hi, b_lo, b_hi;
  half_cvt::cvt_fp16_to_two_fp32(a.v, a_lo, a_hi);
  half_cvt::cvt_fp16_to_two_fp32(b.v, b_lo, b_hi);
  __mmask8 lo = _mm_mask_cmp_ps_mask(m.v, a_lo, b_lo, _CMP_EQ_OQ);
  __mmask8 hi = _mm_mask_cmp_ps_mask(m.v, a_hi, b_hi, _CMP_EQ_OQ);
  return (hi << 4) | lo;
}
#endif
template <TLV_DECL_VEC(V), typename T = Vec2Tag<V>, TL_IF(T::Bytes <= 16), TL_IF(is_any<TypeOf<T>, bfloat16_t>)>
VECOPS_VFUNC Mask<T> cmpeq(V a, V b, Mask<T> m) {
  __m128 a_lo, a_hi, b_lo, b_hi;
  half_cvt::cvt_bf16_to_two_fp32(a.v, a_lo, a_hi);
  half_cvt::cvt_bf16_to_two_fp32(b.v, b_lo, b_hi);
  __mmask8 lo = _mm_mask_cmp_ps_mask(m.v, a_lo, b_lo, _CMP_EQ_OQ);
  __mmask8 hi = _mm_mask_cmp_ps_mask(m.v, a_hi, b_hi, _CMP_EQ_OQ);
  return (hi << 4) | lo;
}
#if VEC_WIDTH >= 256
#ifdef HAS_AVX512_FP16
template <TLV_DECL_VEC(V), typename T = Vec2Tag<V>, TL_IF(T::Bytes == 32), TL_IF(is_any<TypeOf<T>, float16_t>)>
VECOPS_VFUNC Mask<T> cmpeq(V a, V b, Mask<T> m) {
  return _mm256_mask_cmp_ph_mask(m.v, _mm256_castsi256_ph(a.v), _mm256_castsi256_ph(b.v), _CMP_EQ_OQ);
}
#else
template <TLV_DECL_VEC(V), typename T = Vec2Tag<V>, TL_IF(T::Bytes == 32), TL_IF(is_any<TypeOf<T>, float16_t>)>
VECOPS_VFUNC Mask<T> cmpeq(V a, V b, Mask<T> m) {
  __m256 a_lo, a_hi, b_lo, b_hi;
  half_cvt::cvt_fp16_to_two_fp32(a.v, a_lo, a_hi);
  half_cvt::cvt_fp16_to_two_fp32(b.v, b_lo, b_hi);
  __mmask16 lo = _mm256_mask_cmp_ps_mask(m.v, a_lo, b_lo, _CMP_EQ_OQ);
  __mmask16 hi = _mm256_mask_cmp_ps_mask(m.v, a_hi, b_hi, _CMP_EQ_OQ);
  return (hi << 8) | lo;
}
#endif
template <TLV_DECL_VEC(V), typename T = Vec2Tag<V>, TL_IF(T::Bytes == 32), TL_IF(is_any<TypeOf<T>, bfloat16_t>)>
VECOPS_VFUNC Mask<T> cmpeq(V a, V b, Mask<T> m) {
  __m256 a_lo, a_hi, b_lo, b_hi;
  half_cvt::cvt_bf16_to_two_fp32(a.v, a_lo, a_hi);
  half_cvt::cvt_bf16_to_two_fp32(b.v, b_lo, b_hi);
  __mmask16 lo = _mm256_mask_cmp_ps_mask(m.v, a_lo, b_lo, _CMP_EQ_OQ);
  __mmask16 hi = _mm256_mask_cmp_ps_mask(m.v, a_hi, b_hi, _CMP_EQ_OQ);
  return (hi << 8) | lo;
}
#endif
#if VEC_WIDTH >= 512
#ifdef HAS_AVX512_FP16
template <TLV_DECL_VEC(V), typename T = Vec2Tag<V>, TL_IF(T::Bytes == 64), TL_IF(is_any<TypeOf<T>, float16_t>)>
VECOPS_VFUNC Mask<T> cmpeq(V a, V b, Mask<T> m) {
  return _mm512_mask_cmp_ph_mask(m.v, _mm512_castsi512_ph(a.v), _mm512_castsi512_ph(b.v), _CMP_EQ_OQ);
}
#else
template <TLV_DECL_VEC(V), typename T = Vec2Tag<V>, TL_IF(T::Bytes == 64), TL_IF(is_any<TypeOf<T>, float16_t>)>
VECOPS_VFUNC Mask<T> cmpeq(V a, V b, Mask<T> m) {
  __m256i a_lo256 = _mm512_castsi512_si256(a.v);
  __m256i a_hi256 = _mm512_extracti64x4_epi64(a.v, 1);
  __m256i b_lo256 = _mm512_castsi512_si256(b.v);
  __m256i b_hi256 = _mm512_extracti64x4_epi64(b.v, 1);
  __mmask16 lo_m = __mmask16(m.v & 0xFFFF);
  __mmask16 hi_m = __mmask16(m.v >> 16);
  __m256 a_lo0, a_lo1, a_hi0, a_hi1, b_lo0, b_lo1, b_hi0, b_hi1;
  half_cvt::cvt_fp16_to_two_fp32(a_lo256, a_lo0, a_lo1);
  half_cvt::cvt_fp16_to_two_fp32(a_hi256, a_hi0, a_hi1);
  half_cvt::cvt_fp16_to_two_fp32(b_lo256, b_lo0, b_lo1);
  half_cvt::cvt_fp16_to_two_fp32(b_hi256, b_hi0, b_hi1);
  __mmask16 m0 = _mm256_mask_cmp_ps_mask(lo_m, a_lo0, b_lo0, _CMP_EQ_OQ);
  __mmask16 m1 = _mm256_mask_cmp_ps_mask(lo_m, a_lo1, b_lo1, _CMP_EQ_OQ);
  __mmask16 m2 = _mm256_mask_cmp_ps_mask(hi_m, a_hi0, b_hi0, _CMP_EQ_OQ);
  __mmask16 m3 = _mm256_mask_cmp_ps_mask(hi_m, a_hi1, b_hi1, _CMP_EQ_OQ);
  return __mmask32(m0) | (__mmask32(m1) << 8) | (__mmask32(m2) << 16) | (__mmask32(m3) << 24);
}
#endif
template <TLV_DECL_VEC(V), typename T = Vec2Tag<V>, TL_IF(T::Bytes == 64), TL_IF(is_any<TypeOf<T>, bfloat16_t>)>
VECOPS_VFUNC Mask<T> cmpeq(V a, V b, Mask<T> m) {
  __m256i a_lo256 = _mm512_castsi512_si256(a.v);
  __m256i a_hi256 = _mm512_extracti64x4_epi64(a.v, 1);
  __m256i b_lo256 = _mm512_castsi512_si256(b.v);
  __m256i b_hi256 = _mm512_extracti64x4_epi64(b.v, 1);
  __mmask16 lo_m = __mmask16(m.v & 0xFFFF);
  __mmask16 hi_m = __mmask16(m.v >> 16);
  __m256 a_lo0, a_lo1, a_hi0, a_hi1, b_lo0, b_lo1, b_hi0, b_hi1;
  half_cvt::cvt_bf16_to_two_fp32(a_lo256, a_lo0, a_lo1);
  half_cvt::cvt_bf16_to_two_fp32(a_hi256, a_hi0, a_hi1);
  half_cvt::cvt_bf16_to_two_fp32(b_lo256, b_lo0, b_lo1);
  half_cvt::cvt_bf16_to_two_fp32(b_hi256, b_hi0, b_hi1);
  __mmask16 m0 = _mm256_mask_cmp_ps_mask(lo_m, a_lo0, b_lo0, _CMP_EQ_OQ);
  __mmask16 m1 = _mm256_mask_cmp_ps_mask(lo_m, a_lo1, b_lo1, _CMP_EQ_OQ);
  __mmask16 m2 = _mm256_mask_cmp_ps_mask(hi_m, a_hi0, b_hi0, _CMP_EQ_OQ);
  __mmask16 m3 = _mm256_mask_cmp_ps_mask(hi_m, a_hi1, b_hi1, _CMP_EQ_OQ);
  return __mmask32(m0) | (__mmask32(m1) << 8) | (__mmask32(m2) << 16) | (__mmask32(m3) << 24);
}
#endif
#else // HAS_AVX512DQ
template <TLV_DECL_VEC(V), typename T = Vec2Tag<V>>
VECOPS_VFUNC Mask<T> cmpeq(V a, V b, Mask<T> m) {
  using Vi = Vec<ViewAs<Index<TypeOf<T>>, T>>;
  return word::bit_and(Vi{m.v}, Vi{word::cmpeq(a, b).v}).v;
}
#endif // HAS_AVX512DQ


/* ************************************************************************** */
//                            Compare Not Equals                              //
/* ************************************************************************** */
#ifdef HAS_AVX512DQ
template <TLV_DECL_VEC(V), typename T = Vec2Tag<V>, TL_IF(T::Bytes <= 16), TL_IF(is_any<TypeOf<T>, float32_t>)>
VECOPS_VFUNC Mask<T> cmpne(V a, V b) {
  return _mm_cmp_ps_mask(a.v, b.v, _CMP_NEQ_OQ);
}
template <TLV_DECL_VEC(V), typename T = Vec2Tag<V>, TL_IF(T::Bytes <= 16), TL_IF(is_any<TypeOf<T>, float64_t>)>
VECOPS_VFUNC Mask<T> cmpne(V a, V b) {
  return _mm_cmp_pd_mask(a.v, b.v, _CMP_NEQ_OQ);
}
template <TLV_DECL_VEC(V), typename T = Vec2Tag<V>, TL_IF(T::Bytes <= 16), TL_IF(is_any<TypeOf<T>, int8_t>)>
VECOPS_VFUNC Mask<T> cmpne(V a, V b) {
  return _mm_cmp_epi8_mask(a.v, b.v, _MM_CMPINT_NE);
}
template <TLV_DECL_VEC(V), typename T = Vec2Tag<V>, TL_IF(T::Bytes <= 16), TL_IF(is_any<TypeOf<T>, uint8_t>)>
VECOPS_VFUNC Mask<T> cmpne(V a, V b) {
  return _mm_cmp_epu8_mask(a.v, b.v, _MM_CMPINT_NE);
}
template <TLV_DECL_VEC(V), typename T = Vec2Tag<V>, TL_IF(T::Bytes <= 16), TL_IF(is_any<TypeOf<T>, int16_t>)>
VECOPS_VFUNC Mask<T> cmpne(V a, V b) {
  return _mm_cmp_epi16_mask(a.v, b.v, _MM_CMPINT_NE);
}
template <TLV_DECL_VEC(V), typename T = Vec2Tag<V>, TL_IF(T::Bytes <= 16), TL_IF(is_any<TypeOf<T>, uint16_t>)>
VECOPS_VFUNC Mask<T> cmpne(V a, V b) {
  return _mm_cmp_epu16_mask(a.v, b.v, _MM_CMPINT_NE);
}
template <TLV_DECL_VEC(V), typename T = Vec2Tag<V>, TL_IF(T::Bytes <= 16), TL_IF(is_any<TypeOf<T>, int32_t>)>
VECOPS_VFUNC Mask<T> cmpne(V a, V b) {
  return _mm_cmp_epi32_mask(a.v, b.v, _MM_CMPINT_NE);
}
template <TLV_DECL_VEC(V), typename T = Vec2Tag<V>, TL_IF(T::Bytes <= 16), TL_IF(is_any<TypeOf<T>, uint32_t>)>
VECOPS_VFUNC Mask<T> cmpne(V a, V b) {
  return _mm_cmp_epu32_mask(a.v, b.v, _MM_CMPINT_NE);
}
template <TLV_DECL_VEC(V), typename T = Vec2Tag<V>, TL_IF(T::Bytes <= 16), TL_IF(is_any<TypeOf<T>, int64_t>)>
VECOPS_VFUNC Mask<T> cmpne(V a, V b) {
  return _mm_cmp_epi64_mask(a.v, b.v, _MM_CMPINT_NE);
}
template <TLV_DECL_VEC(V), typename T = Vec2Tag<V>, TL_IF(T::Bytes <= 16), TL_IF(is_any<TypeOf<T>, uint64_t>)>
VECOPS_VFUNC Mask<T> cmpne(V a, V b) {
  return _mm_cmp_epu64_mask(a.v, b.v, _MM_CMPINT_NE);
}

template <TLV_DECL_VEC(V), typename T = Vec2Tag<V>, TL_IF(T::Bytes == 32), TL_IF(is_any<TypeOf<T>, float32_t>)>
VECOPS_VFUNC Mask<T> cmpne(V a, V b) {
  return _mm256_cmp_ps_mask(a.v, b.v, _CMP_NEQ_OQ);
}
template <TLV_DECL_VEC(V), typename T = Vec2Tag<V>, TL_IF(T::Bytes == 32), TL_IF(is_any<TypeOf<T>, float64_t>)>
VECOPS_VFUNC Mask<T> cmpne(V a, V b) {
  return _mm256_cmp_pd_mask(a.v, b.v, _CMP_NEQ_OQ);
}
template <TLV_DECL_VEC(V), typename T = Vec2Tag<V>, TL_IF(T::Bytes == 32), TL_IF(is_any<TypeOf<T>, int8_t>)>
VECOPS_VFUNC Mask<T> cmpne(V a, V b) {
  return _mm256_cmp_epi8_mask(a.v, b.v, _MM_CMPINT_NE);
}
template <TLV_DECL_VEC(V), typename T = Vec2Tag<V>, TL_IF(T::Bytes == 32), TL_IF(is_any<TypeOf<T>, uint8_t>)>
VECOPS_VFUNC Mask<T> cmpne(V a, V b) {
  return _mm256_cmp_epu8_mask(a.v, b.v, _MM_CMPINT_NE);
}
template <TLV_DECL_VEC(V), typename T = Vec2Tag<V>, TL_IF(T::Bytes == 32), TL_IF(is_any<TypeOf<T>, int16_t>)>
VECOPS_VFUNC Mask<T> cmpne(V a, V b) {
  return _mm256_cmp_epi16_mask(a.v, b.v, _MM_CMPINT_NE);
}
template <TLV_DECL_VEC(V), typename T = Vec2Tag<V>, TL_IF(T::Bytes == 32), TL_IF(is_any<TypeOf<T>, uint16_t>)>
VECOPS_VFUNC Mask<T> cmpne(V a, V b) {
  return _mm256_cmp_epu16_mask(a.v, b.v, _MM_CMPINT_NE);
}
template <TLV_DECL_VEC(V), typename T = Vec2Tag<V>, TL_IF(T::Bytes == 32), TL_IF(is_any<TypeOf<T>, int32_t>)>
VECOPS_VFUNC Mask<T> cmpne(V a, V b) {
  return _mm256_cmp_epi32_mask(a.v, b.v, _MM_CMPINT_NE);
}
template <TLV_DECL_VEC(V), typename T = Vec2Tag<V>, TL_IF(T::Bytes == 32), TL_IF(is_any<TypeOf<T>, uint32_t>)>
VECOPS_VFUNC Mask<T> cmpne(V a, V b) {
  return _mm256_cmp_epu32_mask(a.v, b.v, _MM_CMPINT_NE);
}
template <TLV_DECL_VEC(V), typename T = Vec2Tag<V>, TL_IF(T::Bytes == 32), TL_IF(is_any<TypeOf<T>, int64_t>)>
VECOPS_VFUNC Mask<T> cmpne(V a, V b) {
  return _mm256_cmp_epi64_mask(a.v, b.v, _MM_CMPINT_NE);
}
template <TLV_DECL_VEC(V), typename T = Vec2Tag<V>, TL_IF(T::Bytes == 32), TL_IF(is_any<TypeOf<T>, uint64_t>)>
VECOPS_VFUNC Mask<T> cmpne(V a, V b) {
  return _mm256_cmp_epu64_mask(a.v, b.v, _MM_CMPINT_NE);
}

template <TLV_DECL_VEC(V), typename T = Vec2Tag<V>, TL_IF(T::Bytes == 64), TL_IF(is_any<TypeOf<T>, float32_t>)>
VECOPS_VFUNC Mask<T> cmpne(V a, V b) {
  return _mm512_cmp_ps_mask(a.v, b.v, _CMP_NEQ_OQ);
}
template <TLV_DECL_VEC(V), typename T = Vec2Tag<V>, TL_IF(T::Bytes == 64), TL_IF(is_any<TypeOf<T>, float64_t>)>
VECOPS_VFUNC Mask<T> cmpne(V a, V b) {
  return _mm512_cmp_pd_mask(a.v, b.v, _CMP_NEQ_OQ);
}
template <TLV_DECL_VEC(V), typename T = Vec2Tag<V>, TL_IF(T::Bytes == 64), TL_IF(is_any<TypeOf<T>, int8_t>)>
VECOPS_VFUNC Mask<T> cmpne(V a, V b) {
  return _mm512_cmp_epi8_mask(a.v, b.v, _MM_CMPINT_NE);
}
template <TLV_DECL_VEC(V), typename T = Vec2Tag<V>, TL_IF(T::Bytes == 64), TL_IF(is_any<TypeOf<T>, uint8_t>)>
VECOPS_VFUNC Mask<T> cmpne(V a, V b) {
  return _mm512_cmp_epu8_mask(a.v, b.v, _MM_CMPINT_NE);
}
template <TLV_DECL_VEC(V), typename T = Vec2Tag<V>, TL_IF(T::Bytes == 64), TL_IF(is_any<TypeOf<T>, int16_t>)>
VECOPS_VFUNC Mask<T> cmpne(V a, V b) {
  return _mm512_cmp_epi16_mask(a.v, b.v, _MM_CMPINT_NE);
}
template <TLV_DECL_VEC(V), typename T = Vec2Tag<V>, TL_IF(T::Bytes == 64), TL_IF(is_any<TypeOf<T>, uint16_t>)>
VECOPS_VFUNC Mask<T> cmpne(V a, V b) {
  return _mm512_cmp_epu16_mask(a.v, b.v, _MM_CMPINT_NE);
}
template <TLV_DECL_VEC(V), typename T = Vec2Tag<V>, TL_IF(T::Bytes == 64), TL_IF(is_any<TypeOf<T>, int32_t>)>
VECOPS_VFUNC Mask<T> cmpne(V a, V b) {
  return _mm512_cmp_epi32_mask(a.v, b.v, _MM_CMPINT_NE);
}
template <TLV_DECL_VEC(V), typename T = Vec2Tag<V>, TL_IF(T::Bytes == 64), TL_IF(is_any<TypeOf<T>, uint32_t>)>
VECOPS_VFUNC Mask<T> cmpne(V a, V b) {
  return _mm512_cmp_epu32_mask(a.v, b.v, _MM_CMPINT_NE);
}
template <TLV_DECL_VEC(V), typename T = Vec2Tag<V>, TL_IF(T::Bytes == 64), TL_IF(is_any<TypeOf<T>, int64_t>)>
VECOPS_VFUNC Mask<T> cmpne(V a, V b) {
  return _mm512_cmp_epi64_mask(a.v, b.v, _MM_CMPINT_NE);
}
template <TLV_DECL_VEC(V), typename T = Vec2Tag<V>, TL_IF(T::Bytes == 64), TL_IF(is_any<TypeOf<T>, uint64_t>)>
VECOPS_VFUNC Mask<T> cmpne(V a, V b) {
  return _mm512_cmp_epu64_mask(a.v, b.v, _MM_CMPINT_NE);
}
#else // HAS_AVX512DQ
template <TLV_DECL_VEC(V), typename T = Vec2Tag<V>, TL_IF(T::Bytes <= 16), TL_IF(is_any<TypeOf<T>, float32_t>)>
VECOPS_VFUNC Mask<T> cmpne(V a, V b) {
  return _mm_castps_si128(_mm_cmpneq_ps(a.v, b.v));
}
template <TLV_DECL_VEC(V), typename T = Vec2Tag<V>, TL_IF(T::Bytes <= 16), TL_IF(is_any<TypeOf<T>, float64_t>)>
VECOPS_VFUNC Mask<T> cmpne(V a, V b) {
  return _mm_castpd_si128(_mm_cmpneq_pd(a.v, b.v));
}
template <TLV_DECL_VEC(V), typename T = Vec2Tag<V>, TL_IF(T::Bytes <= 32), TL_IF(is_int<TypeOf<T>>)>
VECOPS_VFUNC Mask<T> cmpne(V a, V b) {
  return word::bit_not(V{word::cmpeq(a, b).v}).v; // TODO dabian
}

#if VEC_WIDTH >= 256
template <TLV_DECL_VEC(V), typename T = Vec2Tag<V>, TL_IF(T::Bytes == 32), TL_IF(is_any<TypeOf<T>, float32_t>)>
VECOPS_VFUNC Mask<T> cmpne(V a, V b) {
  return _mm256_castps_si256(_mm256_cmp_ps(a.v, b.v, _CMP_NEQ_OQ));
}
template <TLV_DECL_VEC(V), typename T = Vec2Tag<V>, TL_IF(T::Bytes == 32), TL_IF(is_any<TypeOf<T>, float64_t>)>
VECOPS_VFUNC Mask<T> cmpne(V a, V b) {
  return _mm256_castpd_si256(_mm256_cmp_pd(a.v, b.v, _CMP_NEQ_OQ));
}
#endif // VEC_WIDTH >= 256

#if VEC_WIDTH >= 512
  #error "Unreachable"
#endif // VEC_WIDTH >= 512

template <TLV_DECL_VEC(V), typename T = Vec2Tag<V>, TL_IF(T::Bytes <= 16), TL_IF(is_any<TypeOf<T>, float16_t>)>
VECOPS_VFUNC Mask<T> cmpne(V a, V b) {
  __m128 a_lo, a_hi, b_lo, b_hi;
  half_cvt::cvt_fp16_to_two_fp32(a.v, a_lo, a_hi);
  half_cvt::cvt_fp16_to_two_fp32(b.v, b_lo, b_hi);
  return _mm_packs_epi32(
      _mm_castps_si128(_mm_cmpneq_ps(a_lo, b_lo)),
      _mm_castps_si128(_mm_cmpneq_ps(a_hi, b_hi)));
}
template <TLV_DECL_VEC(V), typename T = Vec2Tag<V>, TL_IF(T::Bytes <= 16), TL_IF(is_any<TypeOf<T>, bfloat16_t>)>
VECOPS_VFUNC Mask<T> cmpne(V a, V b) {
  __m128 a_lo, a_hi, b_lo, b_hi;
  half_cvt::cvt_bf16_to_two_fp32(a.v, a_lo, a_hi);
  half_cvt::cvt_bf16_to_two_fp32(b.v, b_lo, b_hi);
  return _mm_packs_epi32(
      _mm_castps_si128(_mm_cmpneq_ps(a_lo, b_lo)),
      _mm_castps_si128(_mm_cmpneq_ps(a_hi, b_hi)));
}
#if VEC_WIDTH >= 256
template <TLV_DECL_VEC(V), typename T = Vec2Tag<V>, TL_IF(T::Bytes == 32), TL_IF(is_any<TypeOf<T>, float16_t>)>
VECOPS_VFUNC Mask<T> cmpne(V a, V b) {
  __m256 a_lo, a_hi, b_lo, b_hi;
  half_cvt::cvt_fp16_to_two_fp32(a.v, a_lo, a_hi);
  half_cvt::cvt_fp16_to_two_fp32(b.v, b_lo, b_hi);
  return _mm256_packs_epi32(
      _mm256_castps_si256(_mm256_cmp_ps(a_lo, b_lo, _CMP_NEQ_OQ)),
      _mm256_castps_si256(_mm256_cmp_ps(a_hi, b_hi, _CMP_NEQ_OQ)));
}
template <TLV_DECL_VEC(V), typename T = Vec2Tag<V>, TL_IF(T::Bytes == 32), TL_IF(is_any<TypeOf<T>, bfloat16_t>)>
VECOPS_VFUNC Mask<T> cmpne(V a, V b) {
  __m256 a_lo, a_hi, b_lo, b_hi;
  half_cvt::cvt_bf16_to_two_fp32(a.v, a_lo, a_hi);
  half_cvt::cvt_bf16_to_two_fp32(b.v, b_lo, b_hi);
  return _mm256_packs_epi32(
      _mm256_castps_si256(_mm256_cmp_ps(a_lo, b_lo, _CMP_NEQ_OQ)),
      _mm256_castps_si256(_mm256_cmp_ps(a_hi, b_hi, _CMP_NEQ_OQ)));
}
#endif
#endif // HAS_AVX512DQ

#ifdef HAS_AVX512DQ
template <TLV_DECL_VEC(V), typename T = Vec2Tag<V>, TL_IF(T::Bytes <= 16), TL_IF(is_any<TypeOf<T>, float32_t>)>
VECOPS_VFUNC Mask<T> cmpne(V a, V b, Mask<T> m) {
  return _mm_mask_cmp_ps_mask(m.v, a.v, b.v, _CMP_NEQ_OQ);
}
template <TLV_DECL_VEC(V), typename T = Vec2Tag<V>, TL_IF(T::Bytes <= 16), TL_IF(is_any<TypeOf<T>, float64_t>)>
VECOPS_VFUNC Mask<T> cmpne(V a, V b, Mask<T> m) {
  return _mm_mask_cmp_pd_mask(m.v, a.v, b.v, _CMP_NEQ_OQ);
}
template <TLV_DECL_VEC(V), typename T = Vec2Tag<V>, TL_IF(T::Bytes <= 16), TL_IF(is_any<TypeOf<T>, int8_t>)>
VECOPS_VFUNC Mask<T> cmpne(V a, V b, Mask<T> m) {
  return _mm_mask_cmp_epi8_mask(m.v, a.v, b.v, _MM_CMPINT_NE);
}
template <TLV_DECL_VEC(V), typename T = Vec2Tag<V>, TL_IF(T::Bytes <= 16), TL_IF(is_any<TypeOf<T>, uint8_t>)>
VECOPS_VFUNC Mask<T> cmpne(V a, V b, Mask<T> m) {
  return _mm_mask_cmp_epu8_mask(m.v, a.v, b.v, _MM_CMPINT_NE);
}
template <TLV_DECL_VEC(V), typename T = Vec2Tag<V>, TL_IF(T::Bytes <= 16), TL_IF(is_any<TypeOf<T>, int16_t>)>
VECOPS_VFUNC Mask<T> cmpne(V a, V b, Mask<T> m) {
  return _mm_mask_cmp_epi16_mask(m.v, a.v, b.v, _MM_CMPINT_NE);
}
template <TLV_DECL_VEC(V), typename T = Vec2Tag<V>, TL_IF(T::Bytes <= 16), TL_IF(is_any<TypeOf<T>, uint16_t>)>
VECOPS_VFUNC Mask<T> cmpne(V a, V b, Mask<T> m) {
  return _mm_mask_cmp_epu16_mask(m.v, a.v, b.v, _MM_CMPINT_NE);
}
template <TLV_DECL_VEC(V), typename T = Vec2Tag<V>, TL_IF(T::Bytes <= 16), TL_IF(is_any<TypeOf<T>, int32_t>)>
VECOPS_VFUNC Mask<T> cmpne(V a, V b, Mask<T> m) {
  return _mm_mask_cmp_epi32_mask(m.v, a.v, b.v, _MM_CMPINT_NE);
}
template <TLV_DECL_VEC(V), typename T = Vec2Tag<V>, TL_IF(T::Bytes <= 16), TL_IF(is_any<TypeOf<T>, uint32_t>)>
VECOPS_VFUNC Mask<T> cmpne(V a, V b, Mask<T> m) {
  return _mm_mask_cmp_epu32_mask(m.v, a.v, b.v, _MM_CMPINT_NE);
}
template <TLV_DECL_VEC(V), typename T = Vec2Tag<V>, TL_IF(T::Bytes <= 16), TL_IF(is_any<TypeOf<T>, int64_t>)>
VECOPS_VFUNC Mask<T> cmpne(V a, V b, Mask<T> m) {
  return _mm_mask_cmp_epi64_mask(m.v, a.v, b.v, _MM_CMPINT_NE);
}
template <TLV_DECL_VEC(V), typename T = Vec2Tag<V>, TL_IF(T::Bytes <= 16), TL_IF(is_any<TypeOf<T>, uint64_t>)>
VECOPS_VFUNC Mask<T> cmpne(V a, V b, Mask<T> m) {
  return _mm_mask_cmp_epu64_mask(m.v, a.v, b.v, _MM_CMPINT_NE);
}

template <TLV_DECL_VEC(V), typename T = Vec2Tag<V>, TL_IF(T::Bytes == 32), TL_IF(is_any<TypeOf<T>, float32_t>)>
VECOPS_VFUNC Mask<T> cmpne(V a, V b, Mask<T> m) {
  return _mm256_mask_cmp_ps_mask(m.v, a.v, b.v, _CMP_NEQ_OQ);
}
template <TLV_DECL_VEC(V), typename T = Vec2Tag<V>, TL_IF(T::Bytes == 32), TL_IF(is_any<TypeOf<T>, float64_t>)>
VECOPS_VFUNC Mask<T> cmpne(V a, V b, Mask<T> m) {
  return _mm256_mask_cmp_pd_mask(m.v, a.v, b.v, _CMP_NEQ_OQ);
}
template <TLV_DECL_VEC(V), typename T = Vec2Tag<V>, TL_IF(T::Bytes == 32), TL_IF(is_any<TypeOf<T>, int8_t>)>
VECOPS_VFUNC Mask<T> cmpne(V a, V b, Mask<T> m) {
  return _mm256_mask_cmp_epi8_mask(m.v, a.v, b.v, _MM_CMPINT_NE);
}
template <TLV_DECL_VEC(V), typename T = Vec2Tag<V>, TL_IF(T::Bytes == 32), TL_IF(is_any<TypeOf<T>, uint8_t>)>
VECOPS_VFUNC Mask<T> cmpne(V a, V b, Mask<T> m) {
  return _mm256_mask_cmp_epu8_mask(m.v, a.v, b.v, _MM_CMPINT_NE);
}
template <TLV_DECL_VEC(V), typename T = Vec2Tag<V>, TL_IF(T::Bytes == 32), TL_IF(is_any<TypeOf<T>, int16_t>)>
VECOPS_VFUNC Mask<T> cmpne(V a, V b, Mask<T> m) {
  return _mm256_mask_cmp_epi16_mask(m.v, a.v, b.v, _MM_CMPINT_NE);
}
template <TLV_DECL_VEC(V), typename T = Vec2Tag<V>, TL_IF(T::Bytes == 32), TL_IF(is_any<TypeOf<T>, uint16_t>)>
VECOPS_VFUNC Mask<T> cmpne(V a, V b, Mask<T> m) {
  return _mm256_mask_cmp_epu16_mask(m.v, a.v, b.v, _MM_CMPINT_NE);
}
template <TLV_DECL_VEC(V), typename T = Vec2Tag<V>, TL_IF(T::Bytes == 32), TL_IF(is_any<TypeOf<T>, int32_t>)>
VECOPS_VFUNC Mask<T> cmpne(V a, V b, Mask<T> m) {
  return _mm256_mask_cmp_epi32_mask(m.v, a.v, b.v, _MM_CMPINT_NE);
}
template <TLV_DECL_VEC(V), typename T = Vec2Tag<V>, TL_IF(T::Bytes == 32), TL_IF(is_any<TypeOf<T>, uint32_t>)>
VECOPS_VFUNC Mask<T> cmpne(V a, V b, Mask<T> m) {
  return _mm256_mask_cmp_epu32_mask(m.v, a.v, b.v, _MM_CMPINT_NE);
}
template <TLV_DECL_VEC(V), typename T = Vec2Tag<V>, TL_IF(T::Bytes == 32), TL_IF(is_any<TypeOf<T>, int64_t>)>
VECOPS_VFUNC Mask<T> cmpne(V a, V b, Mask<T> m) {
  return _mm256_mask_cmp_epi64_mask(m.v, a.v, b.v, _MM_CMPINT_NE);
}
template <TLV_DECL_VEC(V), typename T = Vec2Tag<V>, TL_IF(T::Bytes == 32), TL_IF(is_any<TypeOf<T>, uint64_t>)>
VECOPS_VFUNC Mask<T> cmpne(V a, V b, Mask<T> m) {
  return _mm256_mask_cmp_epu64_mask(m.v, a.v, b.v, _MM_CMPINT_NE);
}

template <TLV_DECL_VEC(V), typename T = Vec2Tag<V>, TL_IF(T::Bytes == 64), TL_IF(is_any<TypeOf<T>, float32_t>)>
VECOPS_VFUNC Mask<T> cmpne(V a, V b, Mask<T> m) {
  return _mm512_mask_cmp_ps_mask(m.v, a.v, b.v, _CMP_NEQ_OQ);
}
template <TLV_DECL_VEC(V), typename T = Vec2Tag<V>, TL_IF(T::Bytes == 64), TL_IF(is_any<TypeOf<T>, float64_t>)>
VECOPS_VFUNC Mask<T> cmpne(V a, V b, Mask<T> m) {
  return _mm512_mask_cmp_pd_mask(m.v, a.v, b.v, _CMP_NEQ_OQ);
}
template <TLV_DECL_VEC(V), typename T = Vec2Tag<V>, TL_IF(T::Bytes == 64), TL_IF(is_any<TypeOf<T>, int8_t>)>
VECOPS_VFUNC Mask<T> cmpne(V a, V b, Mask<T> m) {
  return _mm512_mask_cmp_epi8_mask(m.v, a.v, b.v, _MM_CMPINT_NE);
}
template <TLV_DECL_VEC(V), typename T = Vec2Tag<V>, TL_IF(T::Bytes == 64), TL_IF(is_any<TypeOf<T>, uint8_t>)>
VECOPS_VFUNC Mask<T> cmpne(V a, V b, Mask<T> m) {
  return _mm512_mask_cmp_epu8_mask(m.v, a.v, b.v, _MM_CMPINT_NE);
}
template <TLV_DECL_VEC(V), typename T = Vec2Tag<V>, TL_IF(T::Bytes == 64), TL_IF(is_any<TypeOf<T>, int16_t>)>
VECOPS_VFUNC Mask<T> cmpne(V a, V b, Mask<T> m) {
  return _mm512_mask_cmp_epi16_mask(m.v, a.v, b.v, _MM_CMPINT_NE);
}
template <TLV_DECL_VEC(V), typename T = Vec2Tag<V>, TL_IF(T::Bytes == 64), TL_IF(is_any<TypeOf<T>, uint16_t>)>
VECOPS_VFUNC Mask<T> cmpne(V a, V b, Mask<T> m) {
  return _mm512_mask_cmp_epu16_mask(m.v, a.v, b.v, _MM_CMPINT_NE);
}
template <TLV_DECL_VEC(V), typename T = Vec2Tag<V>, TL_IF(T::Bytes == 64), TL_IF(is_any<TypeOf<T>, int32_t>)>
VECOPS_VFUNC Mask<T> cmpne(V a, V b, Mask<T> m) {
  return _mm512_mask_cmp_epi32_mask(m.v, a.v, b.v, _MM_CMPINT_NE);
}
template <TLV_DECL_VEC(V), typename T = Vec2Tag<V>, TL_IF(T::Bytes == 64), TL_IF(is_any<TypeOf<T>, uint32_t>)>
VECOPS_VFUNC Mask<T> cmpne(V a, V b, Mask<T> m) {
  return _mm512_mask_cmp_epu32_mask(m.v, a.v, b.v, _MM_CMPINT_NE);
}
template <TLV_DECL_VEC(V), typename T = Vec2Tag<V>, TL_IF(T::Bytes == 64), TL_IF(is_any<TypeOf<T>, int64_t>)>
VECOPS_VFUNC Mask<T> cmpne(V a, V b, Mask<T> m) {
  return _mm512_mask_cmp_epi64_mask(m.v, a.v, b.v, _MM_CMPINT_NE);
}
template <TLV_DECL_VEC(V), typename T = Vec2Tag<V>, TL_IF(T::Bytes == 64), TL_IF(is_any<TypeOf<T>, uint64_t>)>
VECOPS_VFUNC Mask<T> cmpne(V a, V b, Mask<T> m) {
  return _mm512_mask_cmp_epu64_mask(m.v, a.v, b.v, _MM_CMPINT_NE);
}
#ifdef HAS_AVX512_FP16
template <TLV_DECL_VEC(V), typename T = Vec2Tag<V>, TL_IF(T::Bytes <= 16), TL_IF(is_any<TypeOf<T>, float16_t>)>
VECOPS_VFUNC Mask<T> cmpne(V a, V b) {
  return _mm_cmp_ph_mask(_mm_castsi128_ph(a.v), _mm_castsi128_ph(b.v), _CMP_NEQ_OQ);
}
#endif
#ifndef HAS_AVX512_FP16
template <TLV_DECL_VEC(V), typename T = Vec2Tag<V>, TL_IF(T::Bytes <= 16), TL_IF(is_any<TypeOf<T>, float16_t>)>
VECOPS_VFUNC Mask<T> cmpne(V a, V b) {
  __m128 a_lo, a_hi, b_lo, b_hi;
  half_cvt::cvt_fp16_to_two_fp32(a.v, a_lo, a_hi);
  half_cvt::cvt_fp16_to_two_fp32(b.v, b_lo, b_hi);
  __mmask8 lo = _mm_cmp_ps_mask(a_lo, b_lo, _CMP_NEQ_OQ);
  __mmask8 hi = _mm_cmp_ps_mask(a_hi, b_hi, _CMP_NEQ_OQ);
  return (hi << 4) | lo;
}
#endif
template <TLV_DECL_VEC(V), typename T = Vec2Tag<V>, TL_IF(T::Bytes <= 16), TL_IF(is_any<TypeOf<T>, bfloat16_t>)>
VECOPS_VFUNC Mask<T> cmpne(V a, V b) {
  __m128 a_lo, a_hi, b_lo, b_hi;
  half_cvt::cvt_bf16_to_two_fp32(a.v, a_lo, a_hi);
  half_cvt::cvt_bf16_to_two_fp32(b.v, b_lo, b_hi);
  __mmask8 lo = _mm_cmp_ps_mask(a_lo, b_lo, _CMP_NEQ_OQ);
  __mmask8 hi = _mm_cmp_ps_mask(a_hi, b_hi, _CMP_NEQ_OQ);
  return (hi << 4) | lo;
}
#if VEC_WIDTH >= 256
#ifdef HAS_AVX512_FP16
template <TLV_DECL_VEC(V), typename T = Vec2Tag<V>, TL_IF(T::Bytes == 32), TL_IF(is_any<TypeOf<T>, float16_t>)>
VECOPS_VFUNC Mask<T> cmpne(V a, V b) {
  return _mm256_cmp_ph_mask(_mm256_castsi256_ph(a.v), _mm256_castsi256_ph(b.v), _CMP_NEQ_OQ);
}
#else
template <TLV_DECL_VEC(V), typename T = Vec2Tag<V>, TL_IF(T::Bytes == 32), TL_IF(is_any<TypeOf<T>, float16_t>)>
VECOPS_VFUNC Mask<T> cmpne(V a, V b) {
  __m256 a_lo, a_hi, b_lo, b_hi;
  half_cvt::cvt_fp16_to_two_fp32(a.v, a_lo, a_hi);
  half_cvt::cvt_fp16_to_two_fp32(b.v, b_lo, b_hi);
  __mmask16 lo = _mm256_cmp_ps_mask(a_lo, b_lo, _CMP_NEQ_OQ);
  __mmask16 hi = _mm256_cmp_ps_mask(a_hi, b_hi, _CMP_NEQ_OQ);
  return (hi << 8) | lo;
}
#endif
template <TLV_DECL_VEC(V), typename T = Vec2Tag<V>, TL_IF(T::Bytes == 32), TL_IF(is_any<TypeOf<T>, bfloat16_t>)>
VECOPS_VFUNC Mask<T> cmpne(V a, V b) {
  __m256 a_lo, a_hi, b_lo, b_hi;
  half_cvt::cvt_bf16_to_two_fp32(a.v, a_lo, a_hi);
  half_cvt::cvt_bf16_to_two_fp32(b.v, b_lo, b_hi);
  __mmask16 lo = _mm256_cmp_ps_mask(a_lo, b_lo, _CMP_NEQ_OQ);
  __mmask16 hi = _mm256_cmp_ps_mask(a_hi, b_hi, _CMP_NEQ_OQ);
  return (hi << 8) | lo;
}
#endif
#if VEC_WIDTH >= 512
#ifdef HAS_AVX512_FP16
template <TLV_DECL_VEC(V), typename T = Vec2Tag<V>, TL_IF(T::Bytes == 64), TL_IF(is_any<TypeOf<T>, float16_t>)>
VECOPS_VFUNC Mask<T> cmpne(V a, V b) {
  return _mm512_cmp_ph_mask(_mm512_castsi512_ph(a.v), _mm512_castsi512_ph(b.v), _CMP_NEQ_OQ);
}
#else
template <TLV_DECL_VEC(V), typename T = Vec2Tag<V>, TL_IF(T::Bytes == 64), TL_IF(is_any<TypeOf<T>, float16_t>)>
VECOPS_VFUNC Mask<T> cmpne(V a, V b) {
  __m256i a_lo256 = _mm512_castsi512_si256(a.v);
  __m256i a_hi256 = _mm512_extracti64x4_epi64(a.v, 1);
  __m256i b_lo256 = _mm512_castsi512_si256(b.v);
  __m256i b_hi256 = _mm512_extracti64x4_epi64(b.v, 1);
  __m256 a_lo0, a_lo1, a_hi0, a_hi1, b_lo0, b_lo1, b_hi0, b_hi1;
  half_cvt::cvt_fp16_to_two_fp32(a_lo256, a_lo0, a_lo1);
  half_cvt::cvt_fp16_to_two_fp32(a_hi256, a_hi0, a_hi1);
  half_cvt::cvt_fp16_to_two_fp32(b_lo256, b_lo0, b_lo1);
  half_cvt::cvt_fp16_to_two_fp32(b_hi256, b_hi0, b_hi1);
  __mmask16 m0 = _mm256_cmp_ps_mask(a_lo0, b_lo0, _CMP_NEQ_OQ);
  __mmask16 m1 = _mm256_cmp_ps_mask(a_lo1, b_lo1, _CMP_NEQ_OQ);
  __mmask16 m2 = _mm256_cmp_ps_mask(a_hi0, b_hi0, _CMP_NEQ_OQ);
  __mmask16 m3 = _mm256_cmp_ps_mask(a_hi1, b_hi1, _CMP_NEQ_OQ);
  return __mmask32(m0) | (__mmask32(m1) << 8) | (__mmask32(m2) << 16) | (__mmask32(m3) << 24);
}
#endif
template <TLV_DECL_VEC(V), typename T = Vec2Tag<V>, TL_IF(T::Bytes == 64), TL_IF(is_any<TypeOf<T>, bfloat16_t>)>
VECOPS_VFUNC Mask<T> cmpne(V a, V b) {
  __m256i a_lo256 = _mm512_castsi512_si256(a.v);
  __m256i a_hi256 = _mm512_extracti64x4_epi64(a.v, 1);
  __m256i b_lo256 = _mm512_castsi512_si256(b.v);
  __m256i b_hi256 = _mm512_extracti64x4_epi64(b.v, 1);
  __m256 a_lo0, a_lo1, a_hi0, a_hi1, b_lo0, b_lo1, b_hi0, b_hi1;
  half_cvt::cvt_bf16_to_two_fp32(a_lo256, a_lo0, a_lo1);
  half_cvt::cvt_bf16_to_two_fp32(a_hi256, a_hi0, a_hi1);
  half_cvt::cvt_bf16_to_two_fp32(b_lo256, b_lo0, b_lo1);
  half_cvt::cvt_bf16_to_two_fp32(b_hi256, b_hi0, b_hi1);
  __mmask16 m0 = _mm256_cmp_ps_mask(a_lo0, b_lo0, _CMP_NEQ_OQ);
  __mmask16 m1 = _mm256_cmp_ps_mask(a_lo1, b_lo1, _CMP_NEQ_OQ);
  __mmask16 m2 = _mm256_cmp_ps_mask(a_hi0, b_hi0, _CMP_NEQ_OQ);
  __mmask16 m3 = _mm256_cmp_ps_mask(a_hi1, b_hi1, _CMP_NEQ_OQ);
  return __mmask32(m0) | (__mmask32(m1) << 8) | (__mmask32(m2) << 16) | (__mmask32(m3) << 24);
}
#endif
#ifdef HAS_AVX512_FP16
template <TLV_DECL_VEC(V), typename T = Vec2Tag<V>, TL_IF(T::Bytes <= 16), TL_IF(is_any<TypeOf<T>, float16_t>)>
VECOPS_VFUNC Mask<T> cmpne(V a, V b, Mask<T> m) {
  return _mm_mask_cmp_ph_mask(m.v, _mm_castsi128_ph(a.v), _mm_castsi128_ph(b.v), _CMP_NEQ_OQ);
}
#endif
#ifndef HAS_AVX512_FP16
template <TLV_DECL_VEC(V), typename T = Vec2Tag<V>, TL_IF(T::Bytes <= 16), TL_IF(is_any<TypeOf<T>, float16_t>)>
VECOPS_VFUNC Mask<T> cmpne(V a, V b, Mask<T> m) {
  __m128 a_lo, a_hi, b_lo, b_hi;
  half_cvt::cvt_fp16_to_two_fp32(a.v, a_lo, a_hi);
  half_cvt::cvt_fp16_to_two_fp32(b.v, b_lo, b_hi);
  __mmask8 lo = _mm_mask_cmp_ps_mask(m.v, a_lo, b_lo, _CMP_NEQ_OQ);
  __mmask8 hi = _mm_mask_cmp_ps_mask(m.v, a_hi, b_hi, _CMP_NEQ_OQ);
  return (hi << 4) | lo;
}
#endif
template <TLV_DECL_VEC(V), typename T = Vec2Tag<V>, TL_IF(T::Bytes <= 16), TL_IF(is_any<TypeOf<T>, bfloat16_t>)>
VECOPS_VFUNC Mask<T> cmpne(V a, V b, Mask<T> m) {
  __m128 a_lo, a_hi, b_lo, b_hi;
  half_cvt::cvt_bf16_to_two_fp32(a.v, a_lo, a_hi);
  half_cvt::cvt_bf16_to_two_fp32(b.v, b_lo, b_hi);
  __mmask8 lo = _mm_mask_cmp_ps_mask(m.v, a_lo, b_lo, _CMP_NEQ_OQ);
  __mmask8 hi = _mm_mask_cmp_ps_mask(m.v, a_hi, b_hi, _CMP_NEQ_OQ);
  return (hi << 4) | lo;
}
#if VEC_WIDTH >= 256
#ifdef HAS_AVX512_FP16
template <TLV_DECL_VEC(V), typename T = Vec2Tag<V>, TL_IF(T::Bytes == 32), TL_IF(is_any<TypeOf<T>, float16_t>)>
VECOPS_VFUNC Mask<T> cmpne(V a, V b, Mask<T> m) {
  return _mm256_mask_cmp_ph_mask(m.v, _mm256_castsi256_ph(a.v), _mm256_castsi256_ph(b.v), _CMP_NEQ_OQ);
}
#else
template <TLV_DECL_VEC(V), typename T = Vec2Tag<V>, TL_IF(T::Bytes == 32), TL_IF(is_any<TypeOf<T>, float16_t>)>
VECOPS_VFUNC Mask<T> cmpne(V a, V b, Mask<T> m) {
  __m256 a_lo, a_hi, b_lo, b_hi;
  half_cvt::cvt_fp16_to_two_fp32(a.v, a_lo, a_hi);
  half_cvt::cvt_fp16_to_two_fp32(b.v, b_lo, b_hi);
  __mmask16 lo = _mm256_mask_cmp_ps_mask(m.v, a_lo, b_lo, _CMP_NEQ_OQ);
  __mmask16 hi = _mm256_mask_cmp_ps_mask(m.v, a_hi, b_hi, _CMP_NEQ_OQ);
  return (hi << 8) | lo;
}
#endif
template <TLV_DECL_VEC(V), typename T = Vec2Tag<V>, TL_IF(T::Bytes == 32), TL_IF(is_any<TypeOf<T>, bfloat16_t>)>
VECOPS_VFUNC Mask<T> cmpne(V a, V b, Mask<T> m) {
  __m256 a_lo, a_hi, b_lo, b_hi;
  half_cvt::cvt_bf16_to_two_fp32(a.v, a_lo, a_hi);
  half_cvt::cvt_bf16_to_two_fp32(b.v, b_lo, b_hi);
  __mmask16 lo = _mm256_mask_cmp_ps_mask(m.v, a_lo, b_lo, _CMP_NEQ_OQ);
  __mmask16 hi = _mm256_mask_cmp_ps_mask(m.v, a_hi, b_hi, _CMP_NEQ_OQ);
  return (hi << 8) | lo;
}
#endif
#if VEC_WIDTH >= 512
#ifdef HAS_AVX512_FP16
template <TLV_DECL_VEC(V), typename T = Vec2Tag<V>, TL_IF(T::Bytes == 64), TL_IF(is_any<TypeOf<T>, float16_t>)>
VECOPS_VFUNC Mask<T> cmpne(V a, V b, Mask<T> m) {
  return _mm512_mask_cmp_ph_mask(m.v, _mm512_castsi512_ph(a.v), _mm512_castsi512_ph(b.v), _CMP_NEQ_OQ);
}
#else
template <TLV_DECL_VEC(V), typename T = Vec2Tag<V>, TL_IF(T::Bytes == 64), TL_IF(is_any<TypeOf<T>, float16_t>)>
VECOPS_VFUNC Mask<T> cmpne(V a, V b, Mask<T> m) {
  __m256i a_lo256 = _mm512_castsi512_si256(a.v);
  __m256i a_hi256 = _mm512_extracti64x4_epi64(a.v, 1);
  __m256i b_lo256 = _mm512_castsi512_si256(b.v);
  __m256i b_hi256 = _mm512_extracti64x4_epi64(b.v, 1);
  __mmask16 lo_m = __mmask16(m.v & 0xFFFF);
  __mmask16 hi_m = __mmask16(m.v >> 16);
  __m256 a_lo0, a_lo1, a_hi0, a_hi1, b_lo0, b_lo1, b_hi0, b_hi1;
  half_cvt::cvt_fp16_to_two_fp32(a_lo256, a_lo0, a_lo1);
  half_cvt::cvt_fp16_to_two_fp32(a_hi256, a_hi0, a_hi1);
  half_cvt::cvt_fp16_to_two_fp32(b_lo256, b_lo0, b_lo1);
  half_cvt::cvt_fp16_to_two_fp32(b_hi256, b_hi0, b_hi1);
  __mmask16 m0 = _mm256_mask_cmp_ps_mask(lo_m, a_lo0, b_lo0, _CMP_NEQ_OQ);
  __mmask16 m1 = _mm256_mask_cmp_ps_mask(lo_m, a_lo1, b_lo1, _CMP_NEQ_OQ);
  __mmask16 m2 = _mm256_mask_cmp_ps_mask(hi_m, a_hi0, b_hi0, _CMP_NEQ_OQ);
  __mmask16 m3 = _mm256_mask_cmp_ps_mask(hi_m, a_hi1, b_hi1, _CMP_NEQ_OQ);
  return __mmask32(m0) | (__mmask32(m1) << 8) | (__mmask32(m2) << 16) | (__mmask32(m3) << 24);
}
#endif
template <TLV_DECL_VEC(V), typename T = Vec2Tag<V>, TL_IF(T::Bytes == 64), TL_IF(is_any<TypeOf<T>, bfloat16_t>)>
VECOPS_VFUNC Mask<T> cmpne(V a, V b, Mask<T> m) {
  __m256i a_lo256 = _mm512_castsi512_si256(a.v);
  __m256i a_hi256 = _mm512_extracti64x4_epi64(a.v, 1);
  __m256i b_lo256 = _mm512_castsi512_si256(b.v);
  __m256i b_hi256 = _mm512_extracti64x4_epi64(b.v, 1);
  __mmask16 lo_m = __mmask16(m.v & 0xFFFF);
  __mmask16 hi_m = __mmask16(m.v >> 16);
  __m256 a_lo0, a_lo1, a_hi0, a_hi1, b_lo0, b_lo1, b_hi0, b_hi1;
  half_cvt::cvt_bf16_to_two_fp32(a_lo256, a_lo0, a_lo1);
  half_cvt::cvt_bf16_to_two_fp32(a_hi256, a_hi0, a_hi1);
  half_cvt::cvt_bf16_to_two_fp32(b_lo256, b_lo0, b_lo1);
  half_cvt::cvt_bf16_to_two_fp32(b_hi256, b_hi0, b_hi1);
  __mmask16 m0 = _mm256_mask_cmp_ps_mask(lo_m, a_lo0, b_lo0, _CMP_NEQ_OQ);
  __mmask16 m1 = _mm256_mask_cmp_ps_mask(lo_m, a_lo1, b_lo1, _CMP_NEQ_OQ);
  __mmask16 m2 = _mm256_mask_cmp_ps_mask(hi_m, a_hi0, b_hi0, _CMP_NEQ_OQ);
  __mmask16 m3 = _mm256_mask_cmp_ps_mask(hi_m, a_hi1, b_hi1, _CMP_NEQ_OQ);
  return __mmask32(m0) | (__mmask32(m1) << 8) | (__mmask32(m2) << 16) | (__mmask32(m3) << 24);
}
#endif
#else // HAS_AVX512DQ
template <TLV_DECL_VEC(V), typename T = Vec2Tag<V>>
VECOPS_VFUNC Mask<T> cmpne(V a, V b, Mask<T> m) {
  using Vi = Vec<ViewAs<Index<TypeOf<T>>, T>>;
  return word::bit_and(Vi{m.v}, Vi{word::cmpne(a, b).v}).v;
}
#endif // HAS_AVX512DQ


/* ************************************************************************** */
//                            Compare Less Than                               //
/* ************************************************************************** */
#ifdef HAS_AVX512DQ
template <TLV_DECL_VEC(V), typename T = Vec2Tag<V>, TL_IF(T::Bytes <= 16), TL_IF(is_any<TypeOf<T>, float32_t>)>
VECOPS_VFUNC Mask<T> cmplt(V a, V b) {
  return _mm_cmp_ps_mask(a.v, b.v, _CMP_LT_OQ);
}
template <TLV_DECL_VEC(V), typename T = Vec2Tag<V>, TL_IF(T::Bytes <= 16), TL_IF(is_any<TypeOf<T>, float64_t>)>
VECOPS_VFUNC Mask<T> cmplt(V a, V b) {
  return _mm_cmp_pd_mask(a.v, b.v, _CMP_LT_OQ);
}
template <TLV_DECL_VEC(V), typename T = Vec2Tag<V>, TL_IF(T::Bytes <= 16), TL_IF(is_any<TypeOf<T>, int8_t>)>
VECOPS_VFUNC Mask<T> cmplt(V a, V b) {
  return _mm_cmp_epi8_mask(a.v, b.v, _MM_CMPINT_LT);
}
template <TLV_DECL_VEC(V), typename T = Vec2Tag<V>, TL_IF(T::Bytes <= 16), TL_IF(is_any<TypeOf<T>, uint8_t>)>
VECOPS_VFUNC Mask<T> cmplt(V a, V b) {
  return _mm_cmp_epu8_mask(a.v, b.v, _MM_CMPINT_LT);
}
template <TLV_DECL_VEC(V), typename T = Vec2Tag<V>, TL_IF(T::Bytes <= 16), TL_IF(is_any<TypeOf<T>, int16_t>)>
VECOPS_VFUNC Mask<T> cmplt(V a, V b) {
  return _mm_cmp_epi16_mask(a.v, b.v, _MM_CMPINT_LT);
}
template <TLV_DECL_VEC(V), typename T = Vec2Tag<V>, TL_IF(T::Bytes <= 16), TL_IF(is_any<TypeOf<T>, uint16_t>)>
VECOPS_VFUNC Mask<T> cmplt(V a, V b) {
  return _mm_cmp_epu16_mask(a.v, b.v, _MM_CMPINT_LT);
}
template <TLV_DECL_VEC(V), typename T = Vec2Tag<V>, TL_IF(T::Bytes <= 16), TL_IF(is_any<TypeOf<T>, int32_t>)>
VECOPS_VFUNC Mask<T> cmplt(V a, V b) {
  return _mm_cmp_epi32_mask(a.v, b.v, _MM_CMPINT_LT);
}
template <TLV_DECL_VEC(V), typename T = Vec2Tag<V>, TL_IF(T::Bytes <= 16), TL_IF(is_any<TypeOf<T>, uint32_t>)>
VECOPS_VFUNC Mask<T> cmplt(V a, V b) {
  return _mm_cmp_epu32_mask(a.v, b.v, _MM_CMPINT_LT);
}
template <TLV_DECL_VEC(V), typename T = Vec2Tag<V>, TL_IF(T::Bytes <= 16), TL_IF(is_any<TypeOf<T>, int64_t>)>
VECOPS_VFUNC Mask<T> cmplt(V a, V b) {
  return _mm_cmp_epi64_mask(a.v, b.v, _MM_CMPINT_LT);
}
template <TLV_DECL_VEC(V), typename T = Vec2Tag<V>, TL_IF(T::Bytes <= 16), TL_IF(is_any<TypeOf<T>, uint64_t>)>
VECOPS_VFUNC Mask<T> cmplt(V a, V b) {
  return _mm_cmp_epu64_mask(a.v, b.v, _MM_CMPINT_LT);
}

template <TLV_DECL_VEC(V), typename T = Vec2Tag<V>, TL_IF(T::Bytes == 32), TL_IF(is_any<TypeOf<T>, float32_t>)>
VECOPS_VFUNC Mask<T> cmplt(V a, V b) {
  return _mm256_cmp_ps_mask(a.v, b.v, _CMP_LT_OQ);
}
template <TLV_DECL_VEC(V), typename T = Vec2Tag<V>, TL_IF(T::Bytes == 32), TL_IF(is_any<TypeOf<T>, float64_t>)>
VECOPS_VFUNC Mask<T> cmplt(V a, V b) {
  return _mm256_cmp_pd_mask(a.v, b.v, _CMP_LT_OQ);
}
template <TLV_DECL_VEC(V), typename T = Vec2Tag<V>, TL_IF(T::Bytes == 32), TL_IF(is_any<TypeOf<T>, int8_t>)>
VECOPS_VFUNC Mask<T> cmplt(V a, V b) {
  return _mm256_cmp_epi8_mask(a.v, b.v, _MM_CMPINT_LT);
}
template <TLV_DECL_VEC(V), typename T = Vec2Tag<V>, TL_IF(T::Bytes == 32), TL_IF(is_any<TypeOf<T>, uint8_t>)>
VECOPS_VFUNC Mask<T> cmplt(V a, V b) {
  return _mm256_cmp_epu8_mask(a.v, b.v, _MM_CMPINT_LT);
}
template <TLV_DECL_VEC(V), typename T = Vec2Tag<V>, TL_IF(T::Bytes == 32), TL_IF(is_any<TypeOf<T>, int16_t>)>
VECOPS_VFUNC Mask<T> cmplt(V a, V b) {
  return _mm256_cmp_epi16_mask(a.v, b.v, _MM_CMPINT_LT);
}
template <TLV_DECL_VEC(V), typename T = Vec2Tag<V>, TL_IF(T::Bytes == 32), TL_IF(is_any<TypeOf<T>, uint16_t>)>
VECOPS_VFUNC Mask<T> cmplt(V a, V b) {
  return _mm256_cmp_epu16_mask(a.v, b.v, _MM_CMPINT_LT);
}
template <TLV_DECL_VEC(V), typename T = Vec2Tag<V>, TL_IF(T::Bytes == 32), TL_IF(is_any<TypeOf<T>, int32_t>)>
VECOPS_VFUNC Mask<T> cmplt(V a, V b) {
  return _mm256_cmp_epi32_mask(a.v, b.v, _MM_CMPINT_LT);
}
template <TLV_DECL_VEC(V), typename T = Vec2Tag<V>, TL_IF(T::Bytes == 32), TL_IF(is_any<TypeOf<T>, uint32_t>)>
VECOPS_VFUNC Mask<T> cmplt(V a, V b) {
  return _mm256_cmp_epu32_mask(a.v, b.v, _MM_CMPINT_LT);
}
template <TLV_DECL_VEC(V), typename T = Vec2Tag<V>, TL_IF(T::Bytes == 32), TL_IF(is_any<TypeOf<T>, int64_t>)>
VECOPS_VFUNC Mask<T> cmplt(V a, V b) {
  return _mm256_cmp_epi64_mask(a.v, b.v, _MM_CMPINT_LT);
}
template <TLV_DECL_VEC(V), typename T = Vec2Tag<V>, TL_IF(T::Bytes == 32), TL_IF(is_any<TypeOf<T>, uint64_t>)>
VECOPS_VFUNC Mask<T> cmplt(V a, V b) {
  return _mm256_cmp_epu64_mask(a.v, b.v, _MM_CMPINT_LT);
}

template <TLV_DECL_VEC(V), typename T = Vec2Tag<V>, TL_IF(T::Bytes == 64), TL_IF(is_any<TypeOf<T>, float32_t>)>
VECOPS_VFUNC Mask<T> cmplt(V a, V b) {
  return _mm512_cmp_ps_mask(a.v, b.v, _CMP_LT_OQ);
}
template <TLV_DECL_VEC(V), typename T = Vec2Tag<V>, TL_IF(T::Bytes == 64), TL_IF(is_any<TypeOf<T>, float64_t>)>
VECOPS_VFUNC Mask<T> cmplt(V a, V b) {
  return _mm512_cmp_pd_mask(a.v, b.v, _CMP_LT_OQ);
}
template <TLV_DECL_VEC(V), typename T = Vec2Tag<V>, TL_IF(T::Bytes == 64), TL_IF(is_any<TypeOf<T>, int8_t>)>
VECOPS_VFUNC Mask<T> cmplt(V a, V b) {
  return _mm512_cmp_epi8_mask(a.v, b.v, _MM_CMPINT_LT);
}
template <TLV_DECL_VEC(V), typename T = Vec2Tag<V>, TL_IF(T::Bytes == 64), TL_IF(is_any<TypeOf<T>, uint8_t>)>
VECOPS_VFUNC Mask<T> cmplt(V a, V b) {
  return _mm512_cmp_epu8_mask(a.v, b.v, _MM_CMPINT_LT);
}
template <TLV_DECL_VEC(V), typename T = Vec2Tag<V>, TL_IF(T::Bytes == 64), TL_IF(is_any<TypeOf<T>, int16_t>)>
VECOPS_VFUNC Mask<T> cmplt(V a, V b) {
  return _mm512_cmp_epi16_mask(a.v, b.v, _MM_CMPINT_LT);
}
template <TLV_DECL_VEC(V), typename T = Vec2Tag<V>, TL_IF(T::Bytes == 64), TL_IF(is_any<TypeOf<T>, uint16_t>)>
VECOPS_VFUNC Mask<T> cmplt(V a, V b) {
  return _mm512_cmp_epu16_mask(a.v, b.v, _MM_CMPINT_LT);
}
template <TLV_DECL_VEC(V), typename T = Vec2Tag<V>, TL_IF(T::Bytes == 64), TL_IF(is_any<TypeOf<T>, int32_t>)>
VECOPS_VFUNC Mask<T> cmplt(V a, V b) {
  return _mm512_cmp_epi32_mask(a.v, b.v, _MM_CMPINT_LT);
}
template <TLV_DECL_VEC(V), typename T = Vec2Tag<V>, TL_IF(T::Bytes == 64), TL_IF(is_any<TypeOf<T>, uint32_t>)>
VECOPS_VFUNC Mask<T> cmplt(V a, V b) {
  return _mm512_cmp_epu32_mask(a.v, b.v, _MM_CMPINT_LT);
}
template <TLV_DECL_VEC(V), typename T = Vec2Tag<V>, TL_IF(T::Bytes == 64), TL_IF(is_any<TypeOf<T>, int64_t>)>
VECOPS_VFUNC Mask<T> cmplt(V a, V b) {
  return _mm512_cmp_epi64_mask(a.v, b.v, _MM_CMPINT_LT);
}
template <TLV_DECL_VEC(V), typename T = Vec2Tag<V>, TL_IF(T::Bytes == 64), TL_IF(is_any<TypeOf<T>, uint64_t>)>
VECOPS_VFUNC Mask<T> cmplt(V a, V b) {
  return _mm512_cmp_epu64_mask(a.v, b.v, _MM_CMPINT_LT);
}
#else // HAS_AVX512DQ
template <TLV_DECL_VEC(V), typename T = Vec2Tag<V>, TL_IF(T::Bytes <= 16), TL_IF(is_any<TypeOf<T>, float32_t>)>
VECOPS_VFUNC Mask<T> cmplt(V a, V b) {
  return _mm_castps_si128(_mm_cmplt_ps(a.v, b.v));
}
template <TLV_DECL_VEC(V), typename T = Vec2Tag<V>, TL_IF(T::Bytes <= 16), TL_IF(is_any<TypeOf<T>, float64_t>)>
VECOPS_VFUNC Mask<T> cmplt(V a, V b) {
  return _mm_castpd_si128(_mm_cmplt_pd(a.v, b.v));
}
template <TLV_DECL_VEC(V), typename T = Vec2Tag<V>, TL_IF(T::Bytes <= 16), TL_IF(is_any<TypeOf<T>, int8_t>)>
VECOPS_VFUNC Mask<T> cmplt(V a, V b) {
  return _mm_cmpgt_epi8(b.v, a.v);
}
template <TLV_DECL_VEC(V), typename T = Vec2Tag<V>, TL_IF(T::Bytes <= 16), TL_IF(is_any<TypeOf<T>, int16_t>)>
VECOPS_VFUNC Mask<T> cmplt(V a, V b) {
  return _mm_cmpgt_epi16(b.v, a.v);
}
template <TLV_DECL_VEC(V), typename T = Vec2Tag<V>, TL_IF(T::Bytes <= 16), TL_IF(is_any<TypeOf<T>, int32_t>)>
VECOPS_VFUNC Mask<T> cmplt(V a, V b) {
  return _mm_cmpgt_epi32(b.v, a.v);
}
template <TLV_DECL_VEC(V), typename T = Vec2Tag<V>, TL_IF(T::Bytes <= 16), TL_IF(is_any<TypeOf<T>, int64_t>)>
VECOPS_VFUNC Mask<T> cmplt(V a, V b) {
  return _mm_cmpgt_epi64(b.v, a.v);
}
#if VEC_WIDTH >= 256
template <TLV_DECL_VEC(V), typename T = Vec2Tag<V>, TL_IF(T::Bytes == 32), TL_IF(is_any<TypeOf<T>, float32_t>)>
VECOPS_VFUNC Mask<T> cmplt(V a, V b) {
  return _mm256_castps_si256(_mm256_cmp_ps(a.v, b.v, _CMP_LT_OQ));
}
template <TLV_DECL_VEC(V), typename T = Vec2Tag<V>, TL_IF(T::Bytes == 32), TL_IF(is_any<TypeOf<T>, float64_t>)>
VECOPS_VFUNC Mask<T> cmplt(V a, V b) {
  return _mm256_castpd_si256(_mm256_cmp_pd(a.v, b.v, _CMP_LT_OQ));
}
template <TLV_DECL_VEC(V), typename T = Vec2Tag<V>, TL_IF(T::Bytes == 32), TL_IF(is_any<TypeOf<T>, int8_t>)>
VECOPS_VFUNC Mask<T> cmplt(V a, V b) {
  return _mm256_cmpgt_epi8(b.v, a.v);
}
template <TLV_DECL_VEC(V), typename T = Vec2Tag<V>, TL_IF(T::Bytes == 32), TL_IF(is_any<TypeOf<T>, int16_t>)>
VECOPS_VFUNC Mask<T> cmplt(V a, V b) {
  return _mm256_cmpgt_epi16(b.v, a.v);
}
template <TLV_DECL_VEC(V), typename T = Vec2Tag<V>, TL_IF(T::Bytes == 32), TL_IF(is_any<TypeOf<T>, int32_t>)>
VECOPS_VFUNC Mask<T> cmplt(V a, V b) {
  return _mm256_cmpgt_epi32(b.v, a.v);
}
template <TLV_DECL_VEC(V), typename T = Vec2Tag<V>, TL_IF(T::Bytes == 32), TL_IF(is_any<TypeOf<T>, int64_t>)>
VECOPS_VFUNC Mask<T> cmplt(V a, V b) {
  return _mm256_cmpgt_epi64(b.v, a.v);
}
#endif // VEC_WIDTH >= 256

template <TLV_DECL_VEC(V), typename T = Vec2Tag<V>, TL_IF(is_int<TypeOf<T>> && std::is_unsigned_v<TypeOf<T>>)>
VECOPS_VFUNC Mask<T> cmplt(V a, V b) {
  using Es = std::make_signed_t<TypeOf<T>>;
  constexpr Es val = 1uLL << (sizeof(Es) * CHAR_BIT - 1);
  ViewAs<Es, T> ts;
  const auto bits = word::fill(ts, val);
  return word::cmplt(
      word::bit_xor(word::bitcast(ts, a), bits),
      word::bit_xor(word::bitcast(ts, b), bits)
  );
}

#if VEC_WIDTH >= 512
  #error "Unreachable"
#endif // VEC_WIDTH >= 512

template <TLV_DECL_VEC(V), typename T = Vec2Tag<V>, TL_IF(T::Bytes <= 16), TL_IF(is_any<TypeOf<T>, float16_t>)>
VECOPS_VFUNC Mask<T> cmplt(V a, V b) {
  __m128 a_lo, a_hi, b_lo, b_hi;
  half_cvt::cvt_fp16_to_two_fp32(a.v, a_lo, a_hi);
  half_cvt::cvt_fp16_to_two_fp32(b.v, b_lo, b_hi);
  return _mm_packs_epi32(
      _mm_castps_si128(_mm_cmplt_ps(a_lo, b_lo)),
      _mm_castps_si128(_mm_cmplt_ps(a_hi, b_hi)));
}
template <TLV_DECL_VEC(V), typename T = Vec2Tag<V>, TL_IF(T::Bytes <= 16), TL_IF(is_any<TypeOf<T>, bfloat16_t>)>
VECOPS_VFUNC Mask<T> cmplt(V a, V b) {
  __m128 a_lo, a_hi, b_lo, b_hi;
  half_cvt::cvt_bf16_to_two_fp32(a.v, a_lo, a_hi);
  half_cvt::cvt_bf16_to_two_fp32(b.v, b_lo, b_hi);
  return _mm_packs_epi32(
      _mm_castps_si128(_mm_cmplt_ps(a_lo, b_lo)),
      _mm_castps_si128(_mm_cmplt_ps(a_hi, b_hi)));
}
#if VEC_WIDTH >= 256
template <TLV_DECL_VEC(V), typename T = Vec2Tag<V>, TL_IF(T::Bytes == 32), TL_IF(is_any<TypeOf<T>, float16_t>)>
VECOPS_VFUNC Mask<T> cmplt(V a, V b) {
  __m256 a_lo, a_hi, b_lo, b_hi;
  half_cvt::cvt_fp16_to_two_fp32(a.v, a_lo, a_hi);
  half_cvt::cvt_fp16_to_two_fp32(b.v, b_lo, b_hi);
  return _mm256_packs_epi32(
      _mm256_castps_si256(_mm256_cmp_ps(a_lo, b_lo, _CMP_LT_OQ)),
      _mm256_castps_si256(_mm256_cmp_ps(a_hi, b_hi, _CMP_LT_OQ)));
}
template <TLV_DECL_VEC(V), typename T = Vec2Tag<V>, TL_IF(T::Bytes == 32), TL_IF(is_any<TypeOf<T>, bfloat16_t>)>
VECOPS_VFUNC Mask<T> cmplt(V a, V b) {
  __m256 a_lo, a_hi, b_lo, b_hi;
  half_cvt::cvt_bf16_to_two_fp32(a.v, a_lo, a_hi);
  half_cvt::cvt_bf16_to_two_fp32(b.v, b_lo, b_hi);
  return _mm256_packs_epi32(
      _mm256_castps_si256(_mm256_cmp_ps(a_lo, b_lo, _CMP_LT_OQ)),
      _mm256_castps_si256(_mm256_cmp_ps(a_hi, b_hi, _CMP_LT_OQ)));
}
#endif
#endif // HAS_AVX512DQ

#ifdef HAS_AVX512DQ
template <TLV_DECL_VEC(V), typename T = Vec2Tag<V>, TL_IF(T::Bytes <= 16), TL_IF(is_any<TypeOf<T>, float32_t>)>
VECOPS_VFUNC Mask<T> cmplt(V a, V b, Mask<T> m) {
  return _mm_mask_cmp_ps_mask(m.v, a.v, b.v, _CMP_LT_OQ);
}
template <TLV_DECL_VEC(V), typename T = Vec2Tag<V>, TL_IF(T::Bytes <= 16), TL_IF(is_any<TypeOf<T>, float64_t>)>
VECOPS_VFUNC Mask<T> cmplt(V a, V b, Mask<T> m) {
  return _mm_mask_cmp_pd_mask(m.v, a.v, b.v, _CMP_LT_OQ);
}
template <TLV_DECL_VEC(V), typename T = Vec2Tag<V>, TL_IF(T::Bytes <= 16), TL_IF(is_any<TypeOf<T>, int8_t>)>
VECOPS_VFUNC Mask<T> cmplt(V a, V b, Mask<T> m) {
  return _mm_mask_cmp_epi8_mask(m.v, a.v, b.v, _MM_CMPINT_LT);
}
template <TLV_DECL_VEC(V), typename T = Vec2Tag<V>, TL_IF(T::Bytes <= 16), TL_IF(is_any<TypeOf<T>, uint8_t>)>
VECOPS_VFUNC Mask<T> cmplt(V a, V b, Mask<T> m) {
  return _mm_mask_cmp_epu8_mask(m.v, a.v, b.v, _MM_CMPINT_LT);
}
template <TLV_DECL_VEC(V), typename T = Vec2Tag<V>, TL_IF(T::Bytes <= 16), TL_IF(is_any<TypeOf<T>, int16_t>)>
VECOPS_VFUNC Mask<T> cmplt(V a, V b, Mask<T> m) {
  return _mm_mask_cmp_epi16_mask(m.v, a.v, b.v, _MM_CMPINT_LT);
}
template <TLV_DECL_VEC(V), typename T = Vec2Tag<V>, TL_IF(T::Bytes <= 16), TL_IF(is_any<TypeOf<T>, uint16_t>)>
VECOPS_VFUNC Mask<T> cmplt(V a, V b, Mask<T> m) {
  return _mm_mask_cmp_epu16_mask(m.v, a.v, b.v, _MM_CMPINT_LT);
}
template <TLV_DECL_VEC(V), typename T = Vec2Tag<V>, TL_IF(T::Bytes <= 16), TL_IF(is_any<TypeOf<T>, int32_t>)>
VECOPS_VFUNC Mask<T> cmplt(V a, V b, Mask<T> m) {
  return _mm_mask_cmp_epi32_mask(m.v, a.v, b.v, _MM_CMPINT_LT);
}
template <TLV_DECL_VEC(V), typename T = Vec2Tag<V>, TL_IF(T::Bytes <= 16), TL_IF(is_any<TypeOf<T>, uint32_t>)>
VECOPS_VFUNC Mask<T> cmplt(V a, V b, Mask<T> m) {
  return _mm_mask_cmp_epu32_mask(m.v, a.v, b.v, _MM_CMPINT_LT);
}
template <TLV_DECL_VEC(V), typename T = Vec2Tag<V>, TL_IF(T::Bytes <= 16), TL_IF(is_any<TypeOf<T>, int64_t>)>
VECOPS_VFUNC Mask<T> cmplt(V a, V b, Mask<T> m) {
  return _mm_mask_cmp_epi64_mask(m.v, a.v, b.v, _MM_CMPINT_LT);
}
template <TLV_DECL_VEC(V), typename T = Vec2Tag<V>, TL_IF(T::Bytes <= 16), TL_IF(is_any<TypeOf<T>, uint64_t>)>
VECOPS_VFUNC Mask<T> cmplt(V a, V b, Mask<T> m) {
  return _mm_mask_cmp_epu64_mask(m.v, a.v, b.v, _MM_CMPINT_LT);
}

template <TLV_DECL_VEC(V), typename T = Vec2Tag<V>, TL_IF(T::Bytes == 32), TL_IF(is_any<TypeOf<T>, float32_t>)>
VECOPS_VFUNC Mask<T> cmplt(V a, V b, Mask<T> m) {
  return _mm256_mask_cmp_ps_mask(m.v, a.v, b.v, _CMP_LT_OQ);
}
template <TLV_DECL_VEC(V), typename T = Vec2Tag<V>, TL_IF(T::Bytes == 32), TL_IF(is_any<TypeOf<T>, float64_t>)>
VECOPS_VFUNC Mask<T> cmplt(V a, V b, Mask<T> m) {
  return _mm256_mask_cmp_pd_mask(m.v, a.v, b.v, _CMP_LT_OQ);
}
template <TLV_DECL_VEC(V), typename T = Vec2Tag<V>, TL_IF(T::Bytes == 32), TL_IF(is_any<TypeOf<T>, int8_t>)>
VECOPS_VFUNC Mask<T> cmplt(V a, V b, Mask<T> m) {
  return _mm256_mask_cmp_epi8_mask(m.v, a.v, b.v, _MM_CMPINT_LT);
}
template <TLV_DECL_VEC(V), typename T = Vec2Tag<V>, TL_IF(T::Bytes == 32), TL_IF(is_any<TypeOf<T>, uint8_t>)>
VECOPS_VFUNC Mask<T> cmplt(V a, V b, Mask<T> m) {
  return _mm256_mask_cmp_epu8_mask(m.v, a.v, b.v, _MM_CMPINT_LT);
}
template <TLV_DECL_VEC(V), typename T = Vec2Tag<V>, TL_IF(T::Bytes == 32), TL_IF(is_any<TypeOf<T>, int16_t>)>
VECOPS_VFUNC Mask<T> cmplt(V a, V b, Mask<T> m) {
  return _mm256_mask_cmp_epi16_mask(m.v, a.v, b.v, _MM_CMPINT_LT);
}
template <TLV_DECL_VEC(V), typename T = Vec2Tag<V>, TL_IF(T::Bytes == 32), TL_IF(is_any<TypeOf<T>, uint16_t>)>
VECOPS_VFUNC Mask<T> cmplt(V a, V b, Mask<T> m) {
  return _mm256_mask_cmp_epu16_mask(m.v, a.v, b.v, _MM_CMPINT_LT);
}
template <TLV_DECL_VEC(V), typename T = Vec2Tag<V>, TL_IF(T::Bytes == 32), TL_IF(is_any<TypeOf<T>, int32_t>)>
VECOPS_VFUNC Mask<T> cmplt(V a, V b, Mask<T> m) {
  return _mm256_mask_cmp_epi32_mask(m.v, a.v, b.v, _MM_CMPINT_LT);
}
template <TLV_DECL_VEC(V), typename T = Vec2Tag<V>, TL_IF(T::Bytes == 32), TL_IF(is_any<TypeOf<T>, uint32_t>)>
VECOPS_VFUNC Mask<T> cmplt(V a, V b, Mask<T> m) {
  return _mm256_mask_cmp_epu32_mask(m.v, a.v, b.v, _MM_CMPINT_LT);
}
template <TLV_DECL_VEC(V), typename T = Vec2Tag<V>, TL_IF(T::Bytes == 32), TL_IF(is_any<TypeOf<T>, int64_t>)>
VECOPS_VFUNC Mask<T> cmplt(V a, V b, Mask<T> m) {
  return _mm256_mask_cmp_epi64_mask(m.v, a.v, b.v, _MM_CMPINT_LT);
}
template <TLV_DECL_VEC(V), typename T = Vec2Tag<V>, TL_IF(T::Bytes == 32), TL_IF(is_any<TypeOf<T>, uint64_t>)>
VECOPS_VFUNC Mask<T> cmplt(V a, V b, Mask<T> m) {
  return _mm256_mask_cmp_epu64_mask(m.v, a.v, b.v, _MM_CMPINT_LT);
}

template <TLV_DECL_VEC(V), typename T = Vec2Tag<V>, TL_IF(T::Bytes == 64), TL_IF(is_any<TypeOf<T>, float32_t>)>
VECOPS_VFUNC Mask<T> cmplt(V a, V b, Mask<T> m) {
  return _mm512_mask_cmp_ps_mask(m.v, a.v, b.v, _CMP_LT_OQ);
}
template <TLV_DECL_VEC(V), typename T = Vec2Tag<V>, TL_IF(T::Bytes == 64), TL_IF(is_any<TypeOf<T>, float64_t>)>
VECOPS_VFUNC Mask<T> cmplt(V a, V b, Mask<T> m) {
  return _mm512_mask_cmp_pd_mask(m.v, a.v, b.v, _CMP_LT_OQ);
}
template <TLV_DECL_VEC(V), typename T = Vec2Tag<V>, TL_IF(T::Bytes == 64), TL_IF(is_any<TypeOf<T>, int8_t>)>
VECOPS_VFUNC Mask<T> cmplt(V a, V b, Mask<T> m) {
  return _mm512_mask_cmp_epi8_mask(m.v, a.v, b.v, _MM_CMPINT_LT);
}
template <TLV_DECL_VEC(V), typename T = Vec2Tag<V>, TL_IF(T::Bytes == 64), TL_IF(is_any<TypeOf<T>, uint8_t>)>
VECOPS_VFUNC Mask<T> cmplt(V a, V b, Mask<T> m) {
  return _mm512_mask_cmp_epu8_mask(m.v, a.v, b.v, _MM_CMPINT_LT);
}
template <TLV_DECL_VEC(V), typename T = Vec2Tag<V>, TL_IF(T::Bytes == 64), TL_IF(is_any<TypeOf<T>, int16_t>)>
VECOPS_VFUNC Mask<T> cmplt(V a, V b, Mask<T> m) {
  return _mm512_mask_cmp_epi16_mask(m.v, a.v, b.v, _MM_CMPINT_LT);
}
template <TLV_DECL_VEC(V), typename T = Vec2Tag<V>, TL_IF(T::Bytes == 64), TL_IF(is_any<TypeOf<T>, uint16_t>)>
VECOPS_VFUNC Mask<T> cmplt(V a, V b, Mask<T> m) {
  return _mm512_mask_cmp_epu16_mask(m.v, a.v, b.v, _MM_CMPINT_LT);
}
template <TLV_DECL_VEC(V), typename T = Vec2Tag<V>, TL_IF(T::Bytes == 64), TL_IF(is_any<TypeOf<T>, int32_t>)>
VECOPS_VFUNC Mask<T> cmplt(V a, V b, Mask<T> m) {
  return _mm512_mask_cmp_epi32_mask(m.v, a.v, b.v, _MM_CMPINT_LT);
}
template <TLV_DECL_VEC(V), typename T = Vec2Tag<V>, TL_IF(T::Bytes == 64), TL_IF(is_any<TypeOf<T>, uint32_t>)>
VECOPS_VFUNC Mask<T> cmplt(V a, V b, Mask<T> m) {
  return _mm512_mask_cmp_epu32_mask(m.v, a.v, b.v, _MM_CMPINT_LT);
}
template <TLV_DECL_VEC(V), typename T = Vec2Tag<V>, TL_IF(T::Bytes == 64), TL_IF(is_any<TypeOf<T>, int64_t>)>
VECOPS_VFUNC Mask<T> cmplt(V a, V b, Mask<T> m) {
  return _mm512_mask_cmp_epi64_mask(m.v, a.v, b.v, _MM_CMPINT_LT);
}
template <TLV_DECL_VEC(V), typename T = Vec2Tag<V>, TL_IF(T::Bytes == 64), TL_IF(is_any<TypeOf<T>, uint64_t>)>
VECOPS_VFUNC Mask<T> cmplt(V a, V b, Mask<T> m) {
  return _mm512_mask_cmp_epu64_mask(m.v, a.v, b.v, _MM_CMPINT_LT);
}
#ifdef HAS_AVX512_FP16
template <TLV_DECL_VEC(V), typename T = Vec2Tag<V>, TL_IF(T::Bytes <= 16), TL_IF(is_any<TypeOf<T>, float16_t>)>
VECOPS_VFUNC Mask<T> cmplt(V a, V b) {
  return _mm_cmp_ph_mask(_mm_castsi128_ph(a.v), _mm_castsi128_ph(b.v), _CMP_LT_OQ);
}
#endif
#ifndef HAS_AVX512_FP16
template <TLV_DECL_VEC(V), typename T = Vec2Tag<V>, TL_IF(T::Bytes <= 16), TL_IF(is_any<TypeOf<T>, float16_t>)>
VECOPS_VFUNC Mask<T> cmplt(V a, V b) {
  __m128 a_lo, a_hi, b_lo, b_hi;
  half_cvt::cvt_fp16_to_two_fp32(a.v, a_lo, a_hi);
  half_cvt::cvt_fp16_to_two_fp32(b.v, b_lo, b_hi);
  __mmask8 lo = _mm_cmp_ps_mask(a_lo, b_lo, _CMP_LT_OQ);
  __mmask8 hi = _mm_cmp_ps_mask(a_hi, b_hi, _CMP_LT_OQ);
  return (hi << 4) | lo;
}
#endif
template <TLV_DECL_VEC(V), typename T = Vec2Tag<V>, TL_IF(T::Bytes <= 16), TL_IF(is_any<TypeOf<T>, bfloat16_t>)>
VECOPS_VFUNC Mask<T> cmplt(V a, V b) {
  __m128 a_lo, a_hi, b_lo, b_hi;
  half_cvt::cvt_bf16_to_two_fp32(a.v, a_lo, a_hi);
  half_cvt::cvt_bf16_to_two_fp32(b.v, b_lo, b_hi);
  __mmask8 lo = _mm_cmp_ps_mask(a_lo, b_lo, _CMP_LT_OQ);
  __mmask8 hi = _mm_cmp_ps_mask(a_hi, b_hi, _CMP_LT_OQ);
  return (hi << 4) | lo;
}
#if VEC_WIDTH >= 256
#ifdef HAS_AVX512_FP16
template <TLV_DECL_VEC(V), typename T = Vec2Tag<V>, TL_IF(T::Bytes == 32), TL_IF(is_any<TypeOf<T>, float16_t>)>
VECOPS_VFUNC Mask<T> cmplt(V a, V b) {
  return _mm256_cmp_ph_mask(_mm256_castsi256_ph(a.v), _mm256_castsi256_ph(b.v), _CMP_LT_OQ);
}
#else
template <TLV_DECL_VEC(V), typename T = Vec2Tag<V>, TL_IF(T::Bytes == 32), TL_IF(is_any<TypeOf<T>, float16_t>)>
VECOPS_VFUNC Mask<T> cmplt(V a, V b) {
  __m256 a_lo, a_hi, b_lo, b_hi;
  half_cvt::cvt_fp16_to_two_fp32(a.v, a_lo, a_hi);
  half_cvt::cvt_fp16_to_two_fp32(b.v, b_lo, b_hi);
  __mmask16 lo = _mm256_cmp_ps_mask(a_lo, b_lo, _CMP_LT_OQ);
  __mmask16 hi = _mm256_cmp_ps_mask(a_hi, b_hi, _CMP_LT_OQ);
  return (hi << 8) | lo;
}
#endif
template <TLV_DECL_VEC(V), typename T = Vec2Tag<V>, TL_IF(T::Bytes == 32), TL_IF(is_any<TypeOf<T>, bfloat16_t>)>
VECOPS_VFUNC Mask<T> cmplt(V a, V b) {
  __m256 a_lo, a_hi, b_lo, b_hi;
  half_cvt::cvt_bf16_to_two_fp32(a.v, a_lo, a_hi);
  half_cvt::cvt_bf16_to_two_fp32(b.v, b_lo, b_hi);
  __mmask16 lo = _mm256_cmp_ps_mask(a_lo, b_lo, _CMP_LT_OQ);
  __mmask16 hi = _mm256_cmp_ps_mask(a_hi, b_hi, _CMP_LT_OQ);
  return (hi << 8) | lo;
}
#endif
#if VEC_WIDTH >= 512
#ifdef HAS_AVX512_FP16
template <TLV_DECL_VEC(V), typename T = Vec2Tag<V>, TL_IF(T::Bytes == 64), TL_IF(is_any<TypeOf<T>, float16_t>)>
VECOPS_VFUNC Mask<T> cmplt(V a, V b) {
  return _mm512_cmp_ph_mask(_mm512_castsi512_ph(a.v), _mm512_castsi512_ph(b.v), _CMP_LT_OQ);
}
#else
template <TLV_DECL_VEC(V), typename T = Vec2Tag<V>, TL_IF(T::Bytes == 64), TL_IF(is_any<TypeOf<T>, float16_t>)>
VECOPS_VFUNC Mask<T> cmplt(V a, V b) {
  __m256i a_lo256 = _mm512_castsi512_si256(a.v);
  __m256i a_hi256 = _mm512_extracti64x4_epi64(a.v, 1);
  __m256i b_lo256 = _mm512_castsi512_si256(b.v);
  __m256i b_hi256 = _mm512_extracti64x4_epi64(b.v, 1);
  __m256 a_lo0, a_lo1, a_hi0, a_hi1, b_lo0, b_lo1, b_hi0, b_hi1;
  half_cvt::cvt_fp16_to_two_fp32(a_lo256, a_lo0, a_lo1);
  half_cvt::cvt_fp16_to_two_fp32(a_hi256, a_hi0, a_hi1);
  half_cvt::cvt_fp16_to_two_fp32(b_lo256, b_lo0, b_lo1);
  half_cvt::cvt_fp16_to_two_fp32(b_hi256, b_hi0, b_hi1);
  __mmask16 m0 = _mm256_cmp_ps_mask(a_lo0, b_lo0, _CMP_LT_OQ);
  __mmask16 m1 = _mm256_cmp_ps_mask(a_lo1, b_lo1, _CMP_LT_OQ);
  __mmask16 m2 = _mm256_cmp_ps_mask(a_hi0, b_hi0, _CMP_LT_OQ);
  __mmask16 m3 = _mm256_cmp_ps_mask(a_hi1, b_hi1, _CMP_LT_OQ);
  return __mmask32(m0) | (__mmask32(m1) << 8) | (__mmask32(m2) << 16) | (__mmask32(m3) << 24);
}
#endif
template <TLV_DECL_VEC(V), typename T = Vec2Tag<V>, TL_IF(T::Bytes == 64), TL_IF(is_any<TypeOf<T>, bfloat16_t>)>
VECOPS_VFUNC Mask<T> cmplt(V a, V b) {
  __m256i a_lo256 = _mm512_castsi512_si256(a.v);
  __m256i a_hi256 = _mm512_extracti64x4_epi64(a.v, 1);
  __m256i b_lo256 = _mm512_castsi512_si256(b.v);
  __m256i b_hi256 = _mm512_extracti64x4_epi64(b.v, 1);
  __m256 a_lo0, a_lo1, a_hi0, a_hi1, b_lo0, b_lo1, b_hi0, b_hi1;
  half_cvt::cvt_bf16_to_two_fp32(a_lo256, a_lo0, a_lo1);
  half_cvt::cvt_bf16_to_two_fp32(a_hi256, a_hi0, a_hi1);
  half_cvt::cvt_bf16_to_two_fp32(b_lo256, b_lo0, b_lo1);
  half_cvt::cvt_bf16_to_two_fp32(b_hi256, b_hi0, b_hi1);
  __mmask16 m0 = _mm256_cmp_ps_mask(a_lo0, b_lo0, _CMP_LT_OQ);
  __mmask16 m1 = _mm256_cmp_ps_mask(a_lo1, b_lo1, _CMP_LT_OQ);
  __mmask16 m2 = _mm256_cmp_ps_mask(a_hi0, b_hi0, _CMP_LT_OQ);
  __mmask16 m3 = _mm256_cmp_ps_mask(a_hi1, b_hi1, _CMP_LT_OQ);
  return __mmask32(m0) | (__mmask32(m1) << 8) | (__mmask32(m2) << 16) | (__mmask32(m3) << 24);
}
#endif
#ifdef HAS_AVX512_FP16
template <TLV_DECL_VEC(V), typename T = Vec2Tag<V>, TL_IF(T::Bytes <= 16), TL_IF(is_any<TypeOf<T>, float16_t>)>
VECOPS_VFUNC Mask<T> cmplt(V a, V b, Mask<T> m) {
  return _mm_mask_cmp_ph_mask(m.v, _mm_castsi128_ph(a.v), _mm_castsi128_ph(b.v), _CMP_LT_OQ);
}
#endif
#ifndef HAS_AVX512_FP16
template <TLV_DECL_VEC(V), typename T = Vec2Tag<V>, TL_IF(T::Bytes <= 16), TL_IF(is_any<TypeOf<T>, float16_t>)>
VECOPS_VFUNC Mask<T> cmplt(V a, V b, Mask<T> m) {
  __m128 a_lo, a_hi, b_lo, b_hi;
  half_cvt::cvt_fp16_to_two_fp32(a.v, a_lo, a_hi);
  half_cvt::cvt_fp16_to_two_fp32(b.v, b_lo, b_hi);
  __mmask8 lo = _mm_mask_cmp_ps_mask(m.v, a_lo, b_lo, _CMP_LT_OQ);
  __mmask8 hi = _mm_mask_cmp_ps_mask(m.v, a_hi, b_hi, _CMP_LT_OQ);
  return (hi << 4) | lo;
}
#endif
template <TLV_DECL_VEC(V), typename T = Vec2Tag<V>, TL_IF(T::Bytes <= 16), TL_IF(is_any<TypeOf<T>, bfloat16_t>)>
VECOPS_VFUNC Mask<T> cmplt(V a, V b, Mask<T> m) {
  __m128 a_lo, a_hi, b_lo, b_hi;
  half_cvt::cvt_bf16_to_two_fp32(a.v, a_lo, a_hi);
  half_cvt::cvt_bf16_to_two_fp32(b.v, b_lo, b_hi);
  __mmask8 lo = _mm_mask_cmp_ps_mask(m.v, a_lo, b_lo, _CMP_LT_OQ);
  __mmask8 hi = _mm_mask_cmp_ps_mask(m.v, a_hi, b_hi, _CMP_LT_OQ);
  return (hi << 4) | lo;
}
#if VEC_WIDTH >= 256
#ifdef HAS_AVX512_FP16
template <TLV_DECL_VEC(V), typename T = Vec2Tag<V>, TL_IF(T::Bytes == 32), TL_IF(is_any<TypeOf<T>, float16_t>)>
VECOPS_VFUNC Mask<T> cmplt(V a, V b, Mask<T> m) {
  return _mm256_mask_cmp_ph_mask(m.v, _mm256_castsi256_ph(a.v), _mm256_castsi256_ph(b.v), _CMP_LT_OQ);
}
#else
template <TLV_DECL_VEC(V), typename T = Vec2Tag<V>, TL_IF(T::Bytes == 32), TL_IF(is_any<TypeOf<T>, float16_t>)>
VECOPS_VFUNC Mask<T> cmplt(V a, V b, Mask<T> m) {
  __m256 a_lo, a_hi, b_lo, b_hi;
  half_cvt::cvt_fp16_to_two_fp32(a.v, a_lo, a_hi);
  half_cvt::cvt_fp16_to_two_fp32(b.v, b_lo, b_hi);
  __mmask16 lo = _mm256_mask_cmp_ps_mask(m.v, a_lo, b_lo, _CMP_LT_OQ);
  __mmask16 hi = _mm256_mask_cmp_ps_mask(m.v, a_hi, b_hi, _CMP_LT_OQ);
  return (hi << 8) | lo;
}
#endif
template <TLV_DECL_VEC(V), typename T = Vec2Tag<V>, TL_IF(T::Bytes == 32), TL_IF(is_any<TypeOf<T>, bfloat16_t>)>
VECOPS_VFUNC Mask<T> cmplt(V a, V b, Mask<T> m) {
  __m256 a_lo, a_hi, b_lo, b_hi;
  half_cvt::cvt_bf16_to_two_fp32(a.v, a_lo, a_hi);
  half_cvt::cvt_bf16_to_two_fp32(b.v, b_lo, b_hi);
  __mmask16 lo = _mm256_mask_cmp_ps_mask(m.v, a_lo, b_lo, _CMP_LT_OQ);
  __mmask16 hi = _mm256_mask_cmp_ps_mask(m.v, a_hi, b_hi, _CMP_LT_OQ);
  return (hi << 8) | lo;
}
#endif
#if VEC_WIDTH >= 512
#ifdef HAS_AVX512_FP16
template <TLV_DECL_VEC(V), typename T = Vec2Tag<V>, TL_IF(T::Bytes == 64), TL_IF(is_any<TypeOf<T>, float16_t>)>
VECOPS_VFUNC Mask<T> cmplt(V a, V b, Mask<T> m) {
  return _mm512_mask_cmp_ph_mask(m.v, _mm512_castsi512_ph(a.v), _mm512_castsi512_ph(b.v), _CMP_LT_OQ);
}
#else
template <TLV_DECL_VEC(V), typename T = Vec2Tag<V>, TL_IF(T::Bytes == 64), TL_IF(is_any<TypeOf<T>, float16_t>)>
VECOPS_VFUNC Mask<T> cmplt(V a, V b, Mask<T> m) {
  __m256i a_lo256 = _mm512_castsi512_si256(a.v);
  __m256i a_hi256 = _mm512_extracti64x4_epi64(a.v, 1);
  __m256i b_lo256 = _mm512_castsi512_si256(b.v);
  __m256i b_hi256 = _mm512_extracti64x4_epi64(b.v, 1);
  __mmask16 lo_m = __mmask16(m.v & 0xFFFF);
  __mmask16 hi_m = __mmask16(m.v >> 16);
  __m256 a_lo0, a_lo1, a_hi0, a_hi1, b_lo0, b_lo1, b_hi0, b_hi1;
  half_cvt::cvt_fp16_to_two_fp32(a_lo256, a_lo0, a_lo1);
  half_cvt::cvt_fp16_to_two_fp32(a_hi256, a_hi0, a_hi1);
  half_cvt::cvt_fp16_to_two_fp32(b_lo256, b_lo0, b_lo1);
  half_cvt::cvt_fp16_to_two_fp32(b_hi256, b_hi0, b_hi1);
  __mmask16 m0 = _mm256_mask_cmp_ps_mask(lo_m, a_lo0, b_lo0, _CMP_LT_OQ);
  __mmask16 m1 = _mm256_mask_cmp_ps_mask(lo_m, a_lo1, b_lo1, _CMP_LT_OQ);
  __mmask16 m2 = _mm256_mask_cmp_ps_mask(hi_m, a_hi0, b_hi0, _CMP_LT_OQ);
  __mmask16 m3 = _mm256_mask_cmp_ps_mask(hi_m, a_hi1, b_hi1, _CMP_LT_OQ);
  return __mmask32(m0) | (__mmask32(m1) << 8) | (__mmask32(m2) << 16) | (__mmask32(m3) << 24);
}
#endif
template <TLV_DECL_VEC(V), typename T = Vec2Tag<V>, TL_IF(T::Bytes == 64), TL_IF(is_any<TypeOf<T>, bfloat16_t>)>
VECOPS_VFUNC Mask<T> cmplt(V a, V b, Mask<T> m) {
  __m256i a_lo256 = _mm512_castsi512_si256(a.v);
  __m256i a_hi256 = _mm512_extracti64x4_epi64(a.v, 1);
  __m256i b_lo256 = _mm512_castsi512_si256(b.v);
  __m256i b_hi256 = _mm512_extracti64x4_epi64(b.v, 1);
  __mmask16 lo_m = __mmask16(m.v & 0xFFFF);
  __mmask16 hi_m = __mmask16(m.v >> 16);
  __m256 a_lo0, a_lo1, a_hi0, a_hi1, b_lo0, b_lo1, b_hi0, b_hi1;
  half_cvt::cvt_bf16_to_two_fp32(a_lo256, a_lo0, a_lo1);
  half_cvt::cvt_bf16_to_two_fp32(a_hi256, a_hi0, a_hi1);
  half_cvt::cvt_bf16_to_two_fp32(b_lo256, b_lo0, b_lo1);
  half_cvt::cvt_bf16_to_two_fp32(b_hi256, b_hi0, b_hi1);
  __mmask16 m0 = _mm256_mask_cmp_ps_mask(lo_m, a_lo0, b_lo0, _CMP_LT_OQ);
  __mmask16 m1 = _mm256_mask_cmp_ps_mask(lo_m, a_lo1, b_lo1, _CMP_LT_OQ);
  __mmask16 m2 = _mm256_mask_cmp_ps_mask(hi_m, a_hi0, b_hi0, _CMP_LT_OQ);
  __mmask16 m3 = _mm256_mask_cmp_ps_mask(hi_m, a_hi1, b_hi1, _CMP_LT_OQ);
  return __mmask32(m0) | (__mmask32(m1) << 8) | (__mmask32(m2) << 16) | (__mmask32(m3) << 24);
}
#endif
#else // HAS_AVX512DQ
template <TLV_DECL_VEC(V), typename T = Vec2Tag<V>>
VECOPS_VFUNC Mask<T> cmplt(V a, V b, Mask<T> m) {
  using Vi = Vec<ViewAs<Index<TypeOf<T>>, T>>;
  return word::bit_and(Vi{m.v}, Vi{word::cmplt(a, b).v}).v;
}
#endif // HAS_AVX512DQ


/* ************************************************************************** */
//                          Compare Greater Than                              //
/* ************************************************************************** */
#ifdef HAS_AVX512DQ
template <TLV_DECL_VEC(V), typename T = Vec2Tag<V>, TL_IF(T::Bytes <= 16), TL_IF(is_any<TypeOf<T>, float32_t>)>
VECOPS_VFUNC Mask<T> cmpgt(V a, V b) {
  return _mm_cmp_ps_mask(a.v, b.v, _CMP_GT_OQ);
}
template <TLV_DECL_VEC(V), typename T = Vec2Tag<V>, TL_IF(T::Bytes <= 16), TL_IF(is_any<TypeOf<T>, float64_t>)>
VECOPS_VFUNC Mask<T> cmpgt(V a, V b) {
  return _mm_cmp_pd_mask(a.v, b.v, _CMP_GT_OQ);
}
template <TLV_DECL_VEC(V), typename T = Vec2Tag<V>, TL_IF(T::Bytes <= 16), TL_IF(is_any<TypeOf<T>, int8_t>)>
VECOPS_VFUNC Mask<T> cmpgt(V a, V b) {
  return _mm_cmp_epi8_mask(a.v, b.v, _MM_CMPINT_NLE);
}
template <TLV_DECL_VEC(V), typename T = Vec2Tag<V>, TL_IF(T::Bytes <= 16), TL_IF(is_any<TypeOf<T>, uint8_t>)>
VECOPS_VFUNC Mask<T> cmpgt(V a, V b) {
  return _mm_cmp_epu8_mask(a.v, b.v, _MM_CMPINT_NLE);
}
template <TLV_DECL_VEC(V), typename T = Vec2Tag<V>, TL_IF(T::Bytes <= 16), TL_IF(is_any<TypeOf<T>, int16_t>)>
VECOPS_VFUNC Mask<T> cmpgt(V a, V b) {
  return _mm_cmp_epi16_mask(a.v, b.v, _MM_CMPINT_NLE);
}
template <TLV_DECL_VEC(V), typename T = Vec2Tag<V>, TL_IF(T::Bytes <= 16), TL_IF(is_any<TypeOf<T>, uint16_t>)>
VECOPS_VFUNC Mask<T> cmpgt(V a, V b) {
  return _mm_cmp_epu16_mask(a.v, b.v, _MM_CMPINT_NLE);
}
template <TLV_DECL_VEC(V), typename T = Vec2Tag<V>, TL_IF(T::Bytes <= 16), TL_IF(is_any<TypeOf<T>, int32_t>)>
VECOPS_VFUNC Mask<T> cmpgt(V a, V b) {
  return _mm_cmp_epi32_mask(a.v, b.v, _MM_CMPINT_NLE);
}
template <TLV_DECL_VEC(V), typename T = Vec2Tag<V>, TL_IF(T::Bytes <= 16), TL_IF(is_any<TypeOf<T>, uint32_t>)>
VECOPS_VFUNC Mask<T> cmpgt(V a, V b) {
  return _mm_cmp_epu32_mask(a.v, b.v, _MM_CMPINT_NLE);
}
template <TLV_DECL_VEC(V), typename T = Vec2Tag<V>, TL_IF(T::Bytes <= 16), TL_IF(is_any<TypeOf<T>, int64_t>)>
VECOPS_VFUNC Mask<T> cmpgt(V a, V b) {
  return _mm_cmp_epi64_mask(a.v, b.v, _MM_CMPINT_NLE);
}
template <TLV_DECL_VEC(V), typename T = Vec2Tag<V>, TL_IF(T::Bytes <= 16), TL_IF(is_any<TypeOf<T>, uint64_t>)>
VECOPS_VFUNC Mask<T> cmpgt(V a, V b) {
  return _mm_cmp_epu64_mask(a.v, b.v, _MM_CMPINT_NLE);
}

template <TLV_DECL_VEC(V), typename T = Vec2Tag<V>, TL_IF(T::Bytes == 32), TL_IF(is_any<TypeOf<T>, float32_t>)>
VECOPS_VFUNC Mask<T> cmpgt(V a, V b) {
  return _mm256_cmp_ps_mask(a.v, b.v, _CMP_GT_OQ);
}
template <TLV_DECL_VEC(V), typename T = Vec2Tag<V>, TL_IF(T::Bytes == 32), TL_IF(is_any<TypeOf<T>, float64_t>)>
VECOPS_VFUNC Mask<T> cmpgt(V a, V b) {
  return _mm256_cmp_pd_mask(a.v, b.v, _CMP_GT_OQ);
}
template <TLV_DECL_VEC(V), typename T = Vec2Tag<V>, TL_IF(T::Bytes == 32), TL_IF(is_any<TypeOf<T>, int8_t>)>
VECOPS_VFUNC Mask<T> cmpgt(V a, V b) {
  return _mm256_cmp_epi8_mask(a.v, b.v, _MM_CMPINT_NLE);
}
template <TLV_DECL_VEC(V), typename T = Vec2Tag<V>, TL_IF(T::Bytes == 32), TL_IF(is_any<TypeOf<T>, uint8_t>)>
VECOPS_VFUNC Mask<T> cmpgt(V a, V b) {
  return _mm256_cmp_epu8_mask(a.v, b.v, _MM_CMPINT_NLE);
}
template <TLV_DECL_VEC(V), typename T = Vec2Tag<V>, TL_IF(T::Bytes == 32), TL_IF(is_any<TypeOf<T>, int16_t>)>
VECOPS_VFUNC Mask<T> cmpgt(V a, V b) {
  return _mm256_cmp_epi16_mask(a.v, b.v, _MM_CMPINT_NLE);
}
template <TLV_DECL_VEC(V), typename T = Vec2Tag<V>, TL_IF(T::Bytes == 32), TL_IF(is_any<TypeOf<T>, uint16_t>)>
VECOPS_VFUNC Mask<T> cmpgt(V a, V b) {
  return _mm256_cmp_epu16_mask(a.v, b.v, _MM_CMPINT_NLE);
}
template <TLV_DECL_VEC(V), typename T = Vec2Tag<V>, TL_IF(T::Bytes == 32), TL_IF(is_any<TypeOf<T>, int32_t>)>
VECOPS_VFUNC Mask<T> cmpgt(V a, V b) {
  return _mm256_cmp_epi32_mask(a.v, b.v, _MM_CMPINT_NLE);
}
template <TLV_DECL_VEC(V), typename T = Vec2Tag<V>, TL_IF(T::Bytes == 32), TL_IF(is_any<TypeOf<T>, uint32_t>)>
VECOPS_VFUNC Mask<T> cmpgt(V a, V b) {
  return _mm256_cmp_epu32_mask(a.v, b.v, _MM_CMPINT_NLE);
}
template <TLV_DECL_VEC(V), typename T = Vec2Tag<V>, TL_IF(T::Bytes == 32), TL_IF(is_any<TypeOf<T>, int64_t>)>
VECOPS_VFUNC Mask<T> cmpgt(V a, V b) {
  return _mm256_cmp_epi64_mask(a.v, b.v, _MM_CMPINT_NLE);
}
template <TLV_DECL_VEC(V), typename T = Vec2Tag<V>, TL_IF(T::Bytes == 32), TL_IF(is_any<TypeOf<T>, uint64_t>)>
VECOPS_VFUNC Mask<T> cmpgt(V a, V b) {
  return _mm256_cmp_epu64_mask(a.v, b.v, _MM_CMPINT_NLE);
}

template <TLV_DECL_VEC(V), typename T = Vec2Tag<V>, TL_IF(T::Bytes == 64), TL_IF(is_any<TypeOf<T>, float32_t>)>
VECOPS_VFUNC Mask<T> cmpgt(V a, V b) {
  return _mm512_cmp_ps_mask(a.v, b.v, _CMP_GT_OQ);
}
template <TLV_DECL_VEC(V), typename T = Vec2Tag<V>, TL_IF(T::Bytes == 64), TL_IF(is_any<TypeOf<T>, float64_t>)>
VECOPS_VFUNC Mask<T> cmpgt(V a, V b) {
  return _mm512_cmp_pd_mask(a.v, b.v, _CMP_GT_OQ);
}
template <TLV_DECL_VEC(V), typename T = Vec2Tag<V>, TL_IF(T::Bytes == 64), TL_IF(is_any<TypeOf<T>, int8_t>)>
VECOPS_VFUNC Mask<T> cmpgt(V a, V b) {
  return _mm512_cmp_epi8_mask(a.v, b.v, _MM_CMPINT_NLE);
}
template <TLV_DECL_VEC(V), typename T = Vec2Tag<V>, TL_IF(T::Bytes == 64), TL_IF(is_any<TypeOf<T>, uint8_t>)>
VECOPS_VFUNC Mask<T> cmpgt(V a, V b) {
  return _mm512_cmp_epu8_mask(a.v, b.v, _MM_CMPINT_NLE);
}
template <TLV_DECL_VEC(V), typename T = Vec2Tag<V>, TL_IF(T::Bytes == 64), TL_IF(is_any<TypeOf<T>, int16_t>)>
VECOPS_VFUNC Mask<T> cmpgt(V a, V b) {
  return _mm512_cmp_epi16_mask(a.v, b.v, _MM_CMPINT_NLE);
}
template <TLV_DECL_VEC(V), typename T = Vec2Tag<V>, TL_IF(T::Bytes == 64), TL_IF(is_any<TypeOf<T>, uint16_t>)>
VECOPS_VFUNC Mask<T> cmpgt(V a, V b) {
  return _mm512_cmp_epu16_mask(a.v, b.v, _MM_CMPINT_NLE);
}
template <TLV_DECL_VEC(V), typename T = Vec2Tag<V>, TL_IF(T::Bytes == 64), TL_IF(is_any<TypeOf<T>, int32_t>)>
VECOPS_VFUNC Mask<T> cmpgt(V a, V b) {
  return _mm512_cmp_epi32_mask(a.v, b.v, _MM_CMPINT_NLE);
}
template <TLV_DECL_VEC(V), typename T = Vec2Tag<V>, TL_IF(T::Bytes == 64), TL_IF(is_any<TypeOf<T>, uint32_t>)>
VECOPS_VFUNC Mask<T> cmpgt(V a, V b) {
  return _mm512_cmp_epu32_mask(a.v, b.v, _MM_CMPINT_NLE);
}
template <TLV_DECL_VEC(V), typename T = Vec2Tag<V>, TL_IF(T::Bytes == 64), TL_IF(is_any<TypeOf<T>, int64_t>)>
VECOPS_VFUNC Mask<T> cmpgt(V a, V b) {
  return _mm512_cmp_epi64_mask(a.v, b.v, _MM_CMPINT_NLE);
}
template <TLV_DECL_VEC(V), typename T = Vec2Tag<V>, TL_IF(T::Bytes == 64), TL_IF(is_any<TypeOf<T>, uint64_t>)>
VECOPS_VFUNC Mask<T> cmpgt(V a, V b) {
  return _mm512_cmp_epu64_mask(a.v, b.v, _MM_CMPINT_NLE);
}
#else // HAS_AVX512DQ
template <TLV_DECL_VEC(V), typename T = Vec2Tag<V>, TL_IF(T::Bytes <= 16), TL_IF(is_any<TypeOf<T>, float32_t>)>
VECOPS_VFUNC Mask<T> cmpgt(V a, V b) {
  return _mm_castps_si128(_mm_cmpgt_ps(a.v, b.v));
}
template <TLV_DECL_VEC(V), typename T = Vec2Tag<V>, TL_IF(T::Bytes <= 16), TL_IF(is_any<TypeOf<T>, float64_t>)>
VECOPS_VFUNC Mask<T> cmpgt(V a, V b) {
  return _mm_castpd_si128(_mm_cmpgt_pd(a.v, b.v));
}
template <TLV_DECL_VEC(V), typename T = Vec2Tag<V>, TL_IF(T::Bytes <= 16), TL_IF(is_any<TypeOf<T>, int8_t>)>
VECOPS_VFUNC Mask<T> cmpgt(V a, V b) {
  return _mm_cmpgt_epi8(a.v, b.v);
}
template <TLV_DECL_VEC(V), typename T = Vec2Tag<V>, TL_IF(T::Bytes <= 16), TL_IF(is_any<TypeOf<T>, int16_t>)>
VECOPS_VFUNC Mask<T> cmpgt(V a, V b) {
  return _mm_cmpgt_epi16(a.v, b.v);
}
template <TLV_DECL_VEC(V), typename T = Vec2Tag<V>, TL_IF(T::Bytes <= 16), TL_IF(is_any<TypeOf<T>, int32_t>)>
VECOPS_VFUNC Mask<T> cmpgt(V a, V b) {
  return _mm_cmpgt_epi32(a.v, b.v);
}
template <TLV_DECL_VEC(V), typename T = Vec2Tag<V>, TL_IF(T::Bytes <= 16), TL_IF(is_any<TypeOf<T>, int64_t>)>
VECOPS_VFUNC Mask<T> cmpgt(V a, V b) {
  return _mm_cmpgt_epi64(a.v, b.v);
}
#if VEC_WIDTH >= 256
template <TLV_DECL_VEC(V), typename T = Vec2Tag<V>, TL_IF(T::Bytes == 32), TL_IF(is_any<TypeOf<T>, float32_t>)>
VECOPS_VFUNC Mask<T> cmpgt(V a, V b) {
  return _mm256_castps_si256(_mm256_cmp_ps(a.v, b.v, _CMP_GT_OQ));
}
template <TLV_DECL_VEC(V), typename T = Vec2Tag<V>, TL_IF(T::Bytes == 32), TL_IF(is_any<TypeOf<T>, float64_t>)>
VECOPS_VFUNC Mask<T> cmpgt(V a, V b) {
  return _mm256_castpd_si256(_mm256_cmp_pd(a.v, b.v, _CMP_GT_OQ));
}
template <TLV_DECL_VEC(V), typename T = Vec2Tag<V>, TL_IF(T::Bytes == 32), TL_IF(is_any<TypeOf<T>, int8_t>)>
VECOPS_VFUNC Mask<T> cmpgt(V a, V b) {
  return _mm256_cmpgt_epi8(a.v, b.v);
}
template <TLV_DECL_VEC(V), typename T = Vec2Tag<V>, TL_IF(T::Bytes == 32), TL_IF(is_any<TypeOf<T>, int16_t>)>
VECOPS_VFUNC Mask<T> cmpgt(V a, V b) {
  return _mm256_cmpgt_epi16(a.v, b.v);
}
template <TLV_DECL_VEC(V), typename T = Vec2Tag<V>, TL_IF(T::Bytes == 32), TL_IF(is_any<TypeOf<T>, int32_t>)>
VECOPS_VFUNC Mask<T> cmpgt(V a, V b) {
  return _mm256_cmpgt_epi32(a.v, b.v);
}
template <TLV_DECL_VEC(V), typename T = Vec2Tag<V>, TL_IF(T::Bytes == 32), TL_IF(is_any<TypeOf<T>, int64_t>)>
VECOPS_VFUNC Mask<T> cmpgt(V a, V b) {
  return _mm256_cmpgt_epi64(a.v, b.v);
}
#endif // VEC_WIDTH >= 256

template <TLV_DECL_VEC(V), typename T = Vec2Tag<V>, TL_IF(is_int<TypeOf<T>> && std::is_unsigned_v<TypeOf<T>>)>
VECOPS_VFUNC Mask<T> cmpgt(V a, V b) {
  using Es = std::make_signed_t<TypeOf<T>>;
  constexpr Es val = 1uLL << (sizeof(Es) * CHAR_BIT - 1);
  ViewAs<Es, T> ts;
  const auto bits = word::fill(ts, val);
  return word::cmpgt(
      word::bit_xor(word::bitcast(ts, a), bits),
      word::bit_xor(word::bitcast(ts, b), bits)
  );
}

#if VEC_WIDTH >= 512
  #error "Unreachable"
#endif // VEC_WIDTH >= 512

template <TLV_DECL_VEC(V), typename T = Vec2Tag<V>, TL_IF(T::Bytes <= 16), TL_IF(is_any<TypeOf<T>, float16_t>)>
VECOPS_VFUNC Mask<T> cmpgt(V a, V b) {
  __m128 a_lo, a_hi, b_lo, b_hi;
  half_cvt::cvt_fp16_to_two_fp32(a.v, a_lo, a_hi);
  half_cvt::cvt_fp16_to_two_fp32(b.v, b_lo, b_hi);
  return _mm_packs_epi32(
      _mm_castps_si128(_mm_cmpgt_ps(a_lo, b_lo)),
      _mm_castps_si128(_mm_cmpgt_ps(a_hi, b_hi)));
}
template <TLV_DECL_VEC(V), typename T = Vec2Tag<V>, TL_IF(T::Bytes <= 16), TL_IF(is_any<TypeOf<T>, bfloat16_t>)>
VECOPS_VFUNC Mask<T> cmpgt(V a, V b) {
  __m128 a_lo, a_hi, b_lo, b_hi;
  half_cvt::cvt_bf16_to_two_fp32(a.v, a_lo, a_hi);
  half_cvt::cvt_bf16_to_two_fp32(b.v, b_lo, b_hi);
  return _mm_packs_epi32(
      _mm_castps_si128(_mm_cmpgt_ps(a_lo, b_lo)),
      _mm_castps_si128(_mm_cmpgt_ps(a_hi, b_hi)));
}
#if VEC_WIDTH >= 256
template <TLV_DECL_VEC(V), typename T = Vec2Tag<V>, TL_IF(T::Bytes == 32), TL_IF(is_any<TypeOf<T>, float16_t>)>
VECOPS_VFUNC Mask<T> cmpgt(V a, V b) {
  __m256 a_lo, a_hi, b_lo, b_hi;
  half_cvt::cvt_fp16_to_two_fp32(a.v, a_lo, a_hi);
  half_cvt::cvt_fp16_to_two_fp32(b.v, b_lo, b_hi);
  return _mm256_packs_epi32(
      _mm256_castps_si256(_mm256_cmp_ps(a_lo, b_lo, _CMP_GT_OQ)),
      _mm256_castps_si256(_mm256_cmp_ps(a_hi, b_hi, _CMP_GT_OQ)));
}
template <TLV_DECL_VEC(V), typename T = Vec2Tag<V>, TL_IF(T::Bytes == 32), TL_IF(is_any<TypeOf<T>, bfloat16_t>)>
VECOPS_VFUNC Mask<T> cmpgt(V a, V b) {
  __m256 a_lo, a_hi, b_lo, b_hi;
  half_cvt::cvt_bf16_to_two_fp32(a.v, a_lo, a_hi);
  half_cvt::cvt_bf16_to_two_fp32(b.v, b_lo, b_hi);
  return _mm256_packs_epi32(
      _mm256_castps_si256(_mm256_cmp_ps(a_lo, b_lo, _CMP_GT_OQ)),
      _mm256_castps_si256(_mm256_cmp_ps(a_hi, b_hi, _CMP_GT_OQ)));
}
#endif
#endif // HAS_AVX512DQ

#ifdef HAS_AVX512DQ
template <TLV_DECL_VEC(V), typename T = Vec2Tag<V>, TL_IF(T::Bytes <= 16), TL_IF(is_any<TypeOf<T>, float32_t>)>
VECOPS_VFUNC Mask<T> cmpgt(V a, V b, Mask<T> m) {
  return _mm_mask_cmp_ps_mask(m.v, a.v, b.v, _CMP_GT_OQ);
}
template <TLV_DECL_VEC(V), typename T = Vec2Tag<V>, TL_IF(T::Bytes <= 16), TL_IF(is_any<TypeOf<T>, float64_t>)>
VECOPS_VFUNC Mask<T> cmpgt(V a, V b, Mask<T> m) {
  return _mm_mask_cmp_pd_mask(m.v, a.v, b.v, _CMP_GT_OQ);
}
template <TLV_DECL_VEC(V), typename T = Vec2Tag<V>, TL_IF(T::Bytes <= 16), TL_IF(is_any<TypeOf<T>, int8_t>)>
VECOPS_VFUNC Mask<T> cmpgt(V a, V b, Mask<T> m) {
  return _mm_mask_cmp_epi8_mask(m.v, a.v, b.v, _MM_CMPINT_NLE);
}
template <TLV_DECL_VEC(V), typename T = Vec2Tag<V>, TL_IF(T::Bytes <= 16), TL_IF(is_any<TypeOf<T>, uint8_t>)>
VECOPS_VFUNC Mask<T> cmpgt(V a, V b, Mask<T> m) {
  return _mm_mask_cmp_epu8_mask(m.v, a.v, b.v, _MM_CMPINT_NLE);
}
template <TLV_DECL_VEC(V), typename T = Vec2Tag<V>, TL_IF(T::Bytes <= 16), TL_IF(is_any<TypeOf<T>, int16_t>)>
VECOPS_VFUNC Mask<T> cmpgt(V a, V b, Mask<T> m) {
  return _mm_mask_cmp_epi16_mask(m.v, a.v, b.v, _MM_CMPINT_NLE);
}
template <TLV_DECL_VEC(V), typename T = Vec2Tag<V>, TL_IF(T::Bytes <= 16), TL_IF(is_any<TypeOf<T>, uint16_t>)>
VECOPS_VFUNC Mask<T> cmpgt(V a, V b, Mask<T> m) {
  return _mm_mask_cmp_epu16_mask(m.v, a.v, b.v, _MM_CMPINT_NLE);
}
template <TLV_DECL_VEC(V), typename T = Vec2Tag<V>, TL_IF(T::Bytes <= 16), TL_IF(is_any<TypeOf<T>, int32_t>)>
VECOPS_VFUNC Mask<T> cmpgt(V a, V b, Mask<T> m) {
  return _mm_mask_cmp_epi32_mask(m.v, a.v, b.v, _MM_CMPINT_NLE);
}
template <TLV_DECL_VEC(V), typename T = Vec2Tag<V>, TL_IF(T::Bytes <= 16), TL_IF(is_any<TypeOf<T>, uint32_t>)>
VECOPS_VFUNC Mask<T> cmpgt(V a, V b, Mask<T> m) {
  return _mm_mask_cmp_epu32_mask(m.v, a.v, b.v, _MM_CMPINT_NLE);
}
template <TLV_DECL_VEC(V), typename T = Vec2Tag<V>, TL_IF(T::Bytes <= 16), TL_IF(is_any<TypeOf<T>, int64_t>)>
VECOPS_VFUNC Mask<T> cmpgt(V a, V b, Mask<T> m) {
  return _mm_mask_cmp_epi64_mask(m.v, a.v, b.v, _MM_CMPINT_NLE);
}
template <TLV_DECL_VEC(V), typename T = Vec2Tag<V>, TL_IF(T::Bytes <= 16), TL_IF(is_any<TypeOf<T>, uint64_t>)>
VECOPS_VFUNC Mask<T> cmpgt(V a, V b, Mask<T> m) {
  return _mm_mask_cmp_epu64_mask(m.v, a.v, b.v, _MM_CMPINT_NLE);
}

template <TLV_DECL_VEC(V), typename T = Vec2Tag<V>, TL_IF(T::Bytes == 32), TL_IF(is_any<TypeOf<T>, float32_t>)>
VECOPS_VFUNC Mask<T> cmpgt(V a, V b, Mask<T> m) {
  return _mm256_mask_cmp_ps_mask(m.v, a.v, b.v, _CMP_GT_OQ);
}
template <TLV_DECL_VEC(V), typename T = Vec2Tag<V>, TL_IF(T::Bytes == 32), TL_IF(is_any<TypeOf<T>, float64_t>)>
VECOPS_VFUNC Mask<T> cmpgt(V a, V b, Mask<T> m) {
  return _mm256_mask_cmp_pd_mask(m.v, a.v, b.v, _CMP_GT_OQ);
}
template <TLV_DECL_VEC(V), typename T = Vec2Tag<V>, TL_IF(T::Bytes == 32), TL_IF(is_any<TypeOf<T>, int8_t>)>
VECOPS_VFUNC Mask<T> cmpgt(V a, V b, Mask<T> m) {
  return _mm256_mask_cmp_epi8_mask(m.v, a.v, b.v, _MM_CMPINT_NLE);
}
template <TLV_DECL_VEC(V), typename T = Vec2Tag<V>, TL_IF(T::Bytes == 32), TL_IF(is_any<TypeOf<T>, uint8_t>)>
VECOPS_VFUNC Mask<T> cmpgt(V a, V b, Mask<T> m) {
  return _mm256_mask_cmp_epu8_mask(m.v, a.v, b.v, _MM_CMPINT_NLE);
}
template <TLV_DECL_VEC(V), typename T = Vec2Tag<V>, TL_IF(T::Bytes == 32), TL_IF(is_any<TypeOf<T>, int16_t>)>
VECOPS_VFUNC Mask<T> cmpgt(V a, V b, Mask<T> m) {
  return _mm256_mask_cmp_epi16_mask(m.v, a.v, b.v, _MM_CMPINT_NLE);
}
template <TLV_DECL_VEC(V), typename T = Vec2Tag<V>, TL_IF(T::Bytes == 32), TL_IF(is_any<TypeOf<T>, uint16_t>)>
VECOPS_VFUNC Mask<T> cmpgt(V a, V b, Mask<T> m) {
  return _mm256_mask_cmp_epu16_mask(m.v, a.v, b.v, _MM_CMPINT_NLE);
}
template <TLV_DECL_VEC(V), typename T = Vec2Tag<V>, TL_IF(T::Bytes == 32), TL_IF(is_any<TypeOf<T>, int32_t>)>
VECOPS_VFUNC Mask<T> cmpgt(V a, V b, Mask<T> m) {
  return _mm256_mask_cmp_epi32_mask(m.v, a.v, b.v, _MM_CMPINT_NLE);
}
template <TLV_DECL_VEC(V), typename T = Vec2Tag<V>, TL_IF(T::Bytes == 32), TL_IF(is_any<TypeOf<T>, uint32_t>)>
VECOPS_VFUNC Mask<T> cmpgt(V a, V b, Mask<T> m) {
  return _mm256_mask_cmp_epu32_mask(m.v, a.v, b.v, _MM_CMPINT_NLE);
}
template <TLV_DECL_VEC(V), typename T = Vec2Tag<V>, TL_IF(T::Bytes == 32), TL_IF(is_any<TypeOf<T>, int64_t>)>
VECOPS_VFUNC Mask<T> cmpgt(V a, V b, Mask<T> m) {
  return _mm256_mask_cmp_epi64_mask(m.v, a.v, b.v, _MM_CMPINT_NLE);
}
template <TLV_DECL_VEC(V), typename T = Vec2Tag<V>, TL_IF(T::Bytes == 32), TL_IF(is_any<TypeOf<T>, uint64_t>)>
VECOPS_VFUNC Mask<T> cmpgt(V a, V b, Mask<T> m) {
  return _mm256_mask_cmp_epu64_mask(m.v, a.v, b.v, _MM_CMPINT_NLE);
}

template <TLV_DECL_VEC(V), typename T = Vec2Tag<V>, TL_IF(T::Bytes == 64), TL_IF(is_any<TypeOf<T>, float32_t>)>
VECOPS_VFUNC Mask<T> cmpgt(V a, V b, Mask<T> m) {
  return _mm512_mask_cmp_ps_mask(m.v, a.v, b.v, _CMP_GT_OQ);
}
template <TLV_DECL_VEC(V), typename T = Vec2Tag<V>, TL_IF(T::Bytes == 64), TL_IF(is_any<TypeOf<T>, float64_t>)>
VECOPS_VFUNC Mask<T> cmpgt(V a, V b, Mask<T> m) {
  return _mm512_mask_cmp_pd_mask(m.v, a.v, b.v, _CMP_GT_OQ);
}
template <TLV_DECL_VEC(V), typename T = Vec2Tag<V>, TL_IF(T::Bytes == 64), TL_IF(is_any<TypeOf<T>, int8_t>)>
VECOPS_VFUNC Mask<T> cmpgt(V a, V b, Mask<T> m) {
  return _mm512_mask_cmp_epi8_mask(m.v, a.v, b.v, _MM_CMPINT_NLE);
}
template <TLV_DECL_VEC(V), typename T = Vec2Tag<V>, TL_IF(T::Bytes == 64), TL_IF(is_any<TypeOf<T>, uint8_t>)>
VECOPS_VFUNC Mask<T> cmpgt(V a, V b, Mask<T> m) {
  return _mm512_mask_cmp_epu8_mask(m.v, a.v, b.v, _MM_CMPINT_NLE);
}
template <TLV_DECL_VEC(V), typename T = Vec2Tag<V>, TL_IF(T::Bytes == 64), TL_IF(is_any<TypeOf<T>, int16_t>)>
VECOPS_VFUNC Mask<T> cmpgt(V a, V b, Mask<T> m) {
  return _mm512_mask_cmp_epi16_mask(m.v, a.v, b.v, _MM_CMPINT_NLE);
}
template <TLV_DECL_VEC(V), typename T = Vec2Tag<V>, TL_IF(T::Bytes == 64), TL_IF(is_any<TypeOf<T>, uint16_t>)>
VECOPS_VFUNC Mask<T> cmpgt(V a, V b, Mask<T> m) {
  return _mm512_mask_cmp_epu16_mask(m.v, a.v, b.v, _MM_CMPINT_NLE);
}
template <TLV_DECL_VEC(V), typename T = Vec2Tag<V>, TL_IF(T::Bytes == 64), TL_IF(is_any<TypeOf<T>, int32_t>)>
VECOPS_VFUNC Mask<T> cmpgt(V a, V b, Mask<T> m) {
  return _mm512_mask_cmp_epi32_mask(m.v, a.v, b.v, _MM_CMPINT_NLE);
}
template <TLV_DECL_VEC(V), typename T = Vec2Tag<V>, TL_IF(T::Bytes == 64), TL_IF(is_any<TypeOf<T>, uint32_t>)>
VECOPS_VFUNC Mask<T> cmpgt(V a, V b, Mask<T> m) {
  return _mm512_mask_cmp_epu32_mask(m.v, a.v, b.v, _MM_CMPINT_NLE);
}
template <TLV_DECL_VEC(V), typename T = Vec2Tag<V>, TL_IF(T::Bytes == 64), TL_IF(is_any<TypeOf<T>, int64_t>)>
VECOPS_VFUNC Mask<T> cmpgt(V a, V b, Mask<T> m) {
  return _mm512_mask_cmp_epi64_mask(m.v, a.v, b.v, _MM_CMPINT_NLE);
}
template <TLV_DECL_VEC(V), typename T = Vec2Tag<V>, TL_IF(T::Bytes == 64), TL_IF(is_any<TypeOf<T>, uint64_t>)>
VECOPS_VFUNC Mask<T> cmpgt(V a, V b, Mask<T> m) {
  return _mm512_mask_cmp_epu64_mask(m.v, a.v, b.v, _MM_CMPINT_NLE);
}
#ifdef HAS_AVX512_FP16
template <TLV_DECL_VEC(V), typename T = Vec2Tag<V>, TL_IF(T::Bytes <= 16), TL_IF(is_any<TypeOf<T>, float16_t>)>
VECOPS_VFUNC Mask<T> cmpgt(V a, V b) {
  return _mm_cmp_ph_mask(_mm_castsi128_ph(a.v), _mm_castsi128_ph(b.v), _CMP_GT_OQ);
}
#endif
#ifndef HAS_AVX512_FP16
template <TLV_DECL_VEC(V), typename T = Vec2Tag<V>, TL_IF(T::Bytes <= 16), TL_IF(is_any<TypeOf<T>, float16_t>)>
VECOPS_VFUNC Mask<T> cmpgt(V a, V b) {
  __m128 a_lo, a_hi, b_lo, b_hi;
  half_cvt::cvt_fp16_to_two_fp32(a.v, a_lo, a_hi);
  half_cvt::cvt_fp16_to_two_fp32(b.v, b_lo, b_hi);
  __mmask8 lo = _mm_cmp_ps_mask(a_lo, b_lo, _CMP_GT_OQ);
  __mmask8 hi = _mm_cmp_ps_mask(a_hi, b_hi, _CMP_GT_OQ);
  return (hi << 4) | lo;
}
#endif
template <TLV_DECL_VEC(V), typename T = Vec2Tag<V>, TL_IF(T::Bytes <= 16), TL_IF(is_any<TypeOf<T>, bfloat16_t>)>
VECOPS_VFUNC Mask<T> cmpgt(V a, V b) {
  __m128 a_lo, a_hi, b_lo, b_hi;
  half_cvt::cvt_bf16_to_two_fp32(a.v, a_lo, a_hi);
  half_cvt::cvt_bf16_to_two_fp32(b.v, b_lo, b_hi);
  __mmask8 lo = _mm_cmp_ps_mask(a_lo, b_lo, _CMP_GT_OQ);
  __mmask8 hi = _mm_cmp_ps_mask(a_hi, b_hi, _CMP_GT_OQ);
  return (hi << 4) | lo;
}
#if VEC_WIDTH >= 256
#ifdef HAS_AVX512_FP16
template <TLV_DECL_VEC(V), typename T = Vec2Tag<V>, TL_IF(T::Bytes == 32), TL_IF(is_any<TypeOf<T>, float16_t>)>
VECOPS_VFUNC Mask<T> cmpgt(V a, V b) {
  return _mm256_cmp_ph_mask(_mm256_castsi256_ph(a.v), _mm256_castsi256_ph(b.v), _CMP_GT_OQ);
}
#else
template <TLV_DECL_VEC(V), typename T = Vec2Tag<V>, TL_IF(T::Bytes == 32), TL_IF(is_any<TypeOf<T>, float16_t>)>
VECOPS_VFUNC Mask<T> cmpgt(V a, V b) {
  __m256 a_lo, a_hi, b_lo, b_hi;
  half_cvt::cvt_fp16_to_two_fp32(a.v, a_lo, a_hi);
  half_cvt::cvt_fp16_to_two_fp32(b.v, b_lo, b_hi);
  __mmask16 lo = _mm256_cmp_ps_mask(a_lo, b_lo, _CMP_GT_OQ);
  __mmask16 hi = _mm256_cmp_ps_mask(a_hi, b_hi, _CMP_GT_OQ);
  return (hi << 8) | lo;
}
#endif
template <TLV_DECL_VEC(V), typename T = Vec2Tag<V>, TL_IF(T::Bytes == 32), TL_IF(is_any<TypeOf<T>, bfloat16_t>)>
VECOPS_VFUNC Mask<T> cmpgt(V a, V b) {
  __m256 a_lo, a_hi, b_lo, b_hi;
  half_cvt::cvt_bf16_to_two_fp32(a.v, a_lo, a_hi);
  half_cvt::cvt_bf16_to_two_fp32(b.v, b_lo, b_hi);
  __mmask16 lo = _mm256_cmp_ps_mask(a_lo, b_lo, _CMP_GT_OQ);
  __mmask16 hi = _mm256_cmp_ps_mask(a_hi, b_hi, _CMP_GT_OQ);
  return (hi << 8) | lo;
}
#endif
#if VEC_WIDTH >= 512
#ifdef HAS_AVX512_FP16
template <TLV_DECL_VEC(V), typename T = Vec2Tag<V>, TL_IF(T::Bytes == 64), TL_IF(is_any<TypeOf<T>, float16_t>)>
VECOPS_VFUNC Mask<T> cmpgt(V a, V b) {
  return _mm512_cmp_ph_mask(_mm512_castsi512_ph(a.v), _mm512_castsi512_ph(b.v), _CMP_GT_OQ);
}
#else
template <TLV_DECL_VEC(V), typename T = Vec2Tag<V>, TL_IF(T::Bytes == 64), TL_IF(is_any<TypeOf<T>, float16_t>)>
VECOPS_VFUNC Mask<T> cmpgt(V a, V b) {
  __m256i a_lo256 = _mm512_castsi512_si256(a.v);
  __m256i a_hi256 = _mm512_extracti64x4_epi64(a.v, 1);
  __m256i b_lo256 = _mm512_castsi512_si256(b.v);
  __m256i b_hi256 = _mm512_extracti64x4_epi64(b.v, 1);
  __m256 a_lo0, a_lo1, a_hi0, a_hi1, b_lo0, b_lo1, b_hi0, b_hi1;
  half_cvt::cvt_fp16_to_two_fp32(a_lo256, a_lo0, a_lo1);
  half_cvt::cvt_fp16_to_two_fp32(a_hi256, a_hi0, a_hi1);
  half_cvt::cvt_fp16_to_two_fp32(b_lo256, b_lo0, b_lo1);
  half_cvt::cvt_fp16_to_two_fp32(b_hi256, b_hi0, b_hi1);
  __mmask16 m0 = _mm256_cmp_ps_mask(a_lo0, b_lo0, _CMP_GT_OQ);
  __mmask16 m1 = _mm256_cmp_ps_mask(a_lo1, b_lo1, _CMP_GT_OQ);
  __mmask16 m2 = _mm256_cmp_ps_mask(a_hi0, b_hi0, _CMP_GT_OQ);
  __mmask16 m3 = _mm256_cmp_ps_mask(a_hi1, b_hi1, _CMP_GT_OQ);
  return __mmask32(m0) | (__mmask32(m1) << 8) | (__mmask32(m2) << 16) | (__mmask32(m3) << 24);
}
#endif
template <TLV_DECL_VEC(V), typename T = Vec2Tag<V>, TL_IF(T::Bytes == 64), TL_IF(is_any<TypeOf<T>, bfloat16_t>)>
VECOPS_VFUNC Mask<T> cmpgt(V a, V b) {
  __m256i a_lo256 = _mm512_castsi512_si256(a.v);
  __m256i a_hi256 = _mm512_extracti64x4_epi64(a.v, 1);
  __m256i b_lo256 = _mm512_castsi512_si256(b.v);
  __m256i b_hi256 = _mm512_extracti64x4_epi64(b.v, 1);
  __m256 a_lo0, a_lo1, a_hi0, a_hi1, b_lo0, b_lo1, b_hi0, b_hi1;
  half_cvt::cvt_bf16_to_two_fp32(a_lo256, a_lo0, a_lo1);
  half_cvt::cvt_bf16_to_two_fp32(a_hi256, a_hi0, a_hi1);
  half_cvt::cvt_bf16_to_two_fp32(b_lo256, b_lo0, b_lo1);
  half_cvt::cvt_bf16_to_two_fp32(b_hi256, b_hi0, b_hi1);
  __mmask16 m0 = _mm256_cmp_ps_mask(a_lo0, b_lo0, _CMP_GT_OQ);
  __mmask16 m1 = _mm256_cmp_ps_mask(a_lo1, b_lo1, _CMP_GT_OQ);
  __mmask16 m2 = _mm256_cmp_ps_mask(a_hi0, b_hi0, _CMP_GT_OQ);
  __mmask16 m3 = _mm256_cmp_ps_mask(a_hi1, b_hi1, _CMP_GT_OQ);
  return __mmask32(m0) | (__mmask32(m1) << 8) | (__mmask32(m2) << 16) | (__mmask32(m3) << 24);
}
#endif
#ifdef HAS_AVX512_FP16
template <TLV_DECL_VEC(V), typename T = Vec2Tag<V>, TL_IF(T::Bytes <= 16), TL_IF(is_any<TypeOf<T>, float16_t>)>
VECOPS_VFUNC Mask<T> cmpgt(V a, V b, Mask<T> m) {
  return _mm_mask_cmp_ph_mask(m.v, _mm_castsi128_ph(a.v), _mm_castsi128_ph(b.v), _CMP_GT_OQ);
}
#endif
#ifndef HAS_AVX512_FP16
template <TLV_DECL_VEC(V), typename T = Vec2Tag<V>, TL_IF(T::Bytes <= 16), TL_IF(is_any<TypeOf<T>, float16_t>)>
VECOPS_VFUNC Mask<T> cmpgt(V a, V b, Mask<T> m) {
  __m128 a_lo, a_hi, b_lo, b_hi;
  half_cvt::cvt_fp16_to_two_fp32(a.v, a_lo, a_hi);
  half_cvt::cvt_fp16_to_two_fp32(b.v, b_lo, b_hi);
  __mmask8 lo = _mm_mask_cmp_ps_mask(m.v, a_lo, b_lo, _CMP_GT_OQ);
  __mmask8 hi = _mm_mask_cmp_ps_mask(m.v, a_hi, b_hi, _CMP_GT_OQ);
  return (hi << 4) | lo;
}
#endif
template <TLV_DECL_VEC(V), typename T = Vec2Tag<V>, TL_IF(T::Bytes <= 16), TL_IF(is_any<TypeOf<T>, bfloat16_t>)>
VECOPS_VFUNC Mask<T> cmpgt(V a, V b, Mask<T> m) {
  __m128 a_lo, a_hi, b_lo, b_hi;
  half_cvt::cvt_bf16_to_two_fp32(a.v, a_lo, a_hi);
  half_cvt::cvt_bf16_to_two_fp32(b.v, b_lo, b_hi);
  __mmask8 lo = _mm_mask_cmp_ps_mask(m.v, a_lo, b_lo, _CMP_GT_OQ);
  __mmask8 hi = _mm_mask_cmp_ps_mask(m.v, a_hi, b_hi, _CMP_GT_OQ);
  return (hi << 4) | lo;
}
#if VEC_WIDTH >= 256
#ifdef HAS_AVX512_FP16
template <TLV_DECL_VEC(V), typename T = Vec2Tag<V>, TL_IF(T::Bytes == 32), TL_IF(is_any<TypeOf<T>, float16_t>)>
VECOPS_VFUNC Mask<T> cmpgt(V a, V b, Mask<T> m) {
  return _mm256_mask_cmp_ph_mask(m.v, _mm256_castsi256_ph(a.v), _mm256_castsi256_ph(b.v), _CMP_GT_OQ);
}
#else
template <TLV_DECL_VEC(V), typename T = Vec2Tag<V>, TL_IF(T::Bytes == 32), TL_IF(is_any<TypeOf<T>, float16_t>)>
VECOPS_VFUNC Mask<T> cmpgt(V a, V b, Mask<T> m) {
  __m256 a_lo, a_hi, b_lo, b_hi;
  half_cvt::cvt_fp16_to_two_fp32(a.v, a_lo, a_hi);
  half_cvt::cvt_fp16_to_two_fp32(b.v, b_lo, b_hi);
  __mmask16 lo = _mm256_mask_cmp_ps_mask(m.v, a_lo, b_lo, _CMP_GT_OQ);
  __mmask16 hi = _mm256_mask_cmp_ps_mask(m.v, a_hi, b_hi, _CMP_GT_OQ);
  return (hi << 8) | lo;
}
#endif
template <TLV_DECL_VEC(V), typename T = Vec2Tag<V>, TL_IF(T::Bytes == 32), TL_IF(is_any<TypeOf<T>, bfloat16_t>)>
VECOPS_VFUNC Mask<T> cmpgt(V a, V b, Mask<T> m) {
  __m256 a_lo, a_hi, b_lo, b_hi;
  half_cvt::cvt_bf16_to_two_fp32(a.v, a_lo, a_hi);
  half_cvt::cvt_bf16_to_two_fp32(b.v, b_lo, b_hi);
  __mmask16 lo = _mm256_mask_cmp_ps_mask(m.v, a_lo, b_lo, _CMP_GT_OQ);
  __mmask16 hi = _mm256_mask_cmp_ps_mask(m.v, a_hi, b_hi, _CMP_GT_OQ);
  return (hi << 8) | lo;
}
#endif
#if VEC_WIDTH >= 512
#ifdef HAS_AVX512_FP16
template <TLV_DECL_VEC(V), typename T = Vec2Tag<V>, TL_IF(T::Bytes == 64), TL_IF(is_any<TypeOf<T>, float16_t>)>
VECOPS_VFUNC Mask<T> cmpgt(V a, V b, Mask<T> m) {
  return _mm512_mask_cmp_ph_mask(m.v, _mm512_castsi512_ph(a.v), _mm512_castsi512_ph(b.v), _CMP_GT_OQ);
}
#else
template <TLV_DECL_VEC(V), typename T = Vec2Tag<V>, TL_IF(T::Bytes == 64), TL_IF(is_any<TypeOf<T>, float16_t>)>
VECOPS_VFUNC Mask<T> cmpgt(V a, V b, Mask<T> m) {
  __m256i a_lo256 = _mm512_castsi512_si256(a.v);
  __m256i a_hi256 = _mm512_extracti64x4_epi64(a.v, 1);
  __m256i b_lo256 = _mm512_castsi512_si256(b.v);
  __m256i b_hi256 = _mm512_extracti64x4_epi64(b.v, 1);
  __mmask16 lo_m = __mmask16(m.v & 0xFFFF);
  __mmask16 hi_m = __mmask16(m.v >> 16);
  __m256 a_lo0, a_lo1, a_hi0, a_hi1, b_lo0, b_lo1, b_hi0, b_hi1;
  half_cvt::cvt_fp16_to_two_fp32(a_lo256, a_lo0, a_lo1);
  half_cvt::cvt_fp16_to_two_fp32(a_hi256, a_hi0, a_hi1);
  half_cvt::cvt_fp16_to_two_fp32(b_lo256, b_lo0, b_lo1);
  half_cvt::cvt_fp16_to_two_fp32(b_hi256, b_hi0, b_hi1);
  __mmask16 m0 = _mm256_mask_cmp_ps_mask(lo_m, a_lo0, b_lo0, _CMP_GT_OQ);
  __mmask16 m1 = _mm256_mask_cmp_ps_mask(lo_m, a_lo1, b_lo1, _CMP_GT_OQ);
  __mmask16 m2 = _mm256_mask_cmp_ps_mask(hi_m, a_hi0, b_hi0, _CMP_GT_OQ);
  __mmask16 m3 = _mm256_mask_cmp_ps_mask(hi_m, a_hi1, b_hi1, _CMP_GT_OQ);
  return __mmask32(m0) | (__mmask32(m1) << 8) | (__mmask32(m2) << 16) | (__mmask32(m3) << 24);
}
#endif
template <TLV_DECL_VEC(V), typename T = Vec2Tag<V>, TL_IF(T::Bytes == 64), TL_IF(is_any<TypeOf<T>, bfloat16_t>)>
VECOPS_VFUNC Mask<T> cmpgt(V a, V b, Mask<T> m) {
  __m256i a_lo256 = _mm512_castsi512_si256(a.v);
  __m256i a_hi256 = _mm512_extracti64x4_epi64(a.v, 1);
  __m256i b_lo256 = _mm512_castsi512_si256(b.v);
  __m256i b_hi256 = _mm512_extracti64x4_epi64(b.v, 1);
  __mmask16 lo_m = __mmask16(m.v & 0xFFFF);
  __mmask16 hi_m = __mmask16(m.v >> 16);
  __m256 a_lo0, a_lo1, a_hi0, a_hi1, b_lo0, b_lo1, b_hi0, b_hi1;
  half_cvt::cvt_bf16_to_two_fp32(a_lo256, a_lo0, a_lo1);
  half_cvt::cvt_bf16_to_two_fp32(a_hi256, a_hi0, a_hi1);
  half_cvt::cvt_bf16_to_two_fp32(b_lo256, b_lo0, b_lo1);
  half_cvt::cvt_bf16_to_two_fp32(b_hi256, b_hi0, b_hi1);
  __mmask16 m0 = _mm256_mask_cmp_ps_mask(lo_m, a_lo0, b_lo0, _CMP_GT_OQ);
  __mmask16 m1 = _mm256_mask_cmp_ps_mask(lo_m, a_lo1, b_lo1, _CMP_GT_OQ);
  __mmask16 m2 = _mm256_mask_cmp_ps_mask(hi_m, a_hi0, b_hi0, _CMP_GT_OQ);
  __mmask16 m3 = _mm256_mask_cmp_ps_mask(hi_m, a_hi1, b_hi1, _CMP_GT_OQ);
  return __mmask32(m0) | (__mmask32(m1) << 8) | (__mmask32(m2) << 16) | (__mmask32(m3) << 24);
}
#endif
#else // HAS_AVX512DQ
template <TLV_DECL_VEC(V), typename T = Vec2Tag<V>>
VECOPS_VFUNC Mask<T> cmpgt(V a, V b, Mask<T> m) {
  using Vi = Vec<ViewAs<Index<TypeOf<T>>, T>>;
  return word::bit_and(Vi{m.v}, Vi{word::cmpgt(a, b).v}).v;
}
#endif // HAS_AVX512DQ


/* ************************************************************************** */
//                       Compare Less Than or Equals                          //
/* ************************************************************************** */
#ifdef HAS_AVX512DQ
template <TLV_DECL_VEC(V), typename T = Vec2Tag<V>, TL_IF(T::Bytes <= 16), TL_IF(is_any<TypeOf<T>, float32_t>)>
VECOPS_VFUNC Mask<T> cmple(V a, V b) {
  return _mm_cmp_ps_mask(a.v, b.v, _CMP_LE_OQ);
}
template <TLV_DECL_VEC(V), typename T = Vec2Tag<V>, TL_IF(T::Bytes <= 16), TL_IF(is_any<TypeOf<T>, float64_t>)>
VECOPS_VFUNC Mask<T> cmple(V a, V b) {
  return _mm_cmp_pd_mask(a.v, b.v, _CMP_LE_OQ);
}
template <TLV_DECL_VEC(V), typename T = Vec2Tag<V>, TL_IF(T::Bytes <= 16), TL_IF(is_any<TypeOf<T>, int8_t>)>
VECOPS_VFUNC Mask<T> cmple(V a, V b) {
  return _mm_cmp_epi8_mask(a.v, b.v, _MM_CMPINT_LE);
}
template <TLV_DECL_VEC(V), typename T = Vec2Tag<V>, TL_IF(T::Bytes <= 16), TL_IF(is_any<TypeOf<T>, uint8_t>)>
VECOPS_VFUNC Mask<T> cmple(V a, V b) {
  return _mm_cmp_epu8_mask(a.v, b.v, _MM_CMPINT_LE);
}
template <TLV_DECL_VEC(V), typename T = Vec2Tag<V>, TL_IF(T::Bytes <= 16), TL_IF(is_any<TypeOf<T>, int16_t>)>
VECOPS_VFUNC Mask<T> cmple(V a, V b) {
  return _mm_cmp_epi16_mask(a.v, b.v, _MM_CMPINT_LE);
}
template <TLV_DECL_VEC(V), typename T = Vec2Tag<V>, TL_IF(T::Bytes <= 16), TL_IF(is_any<TypeOf<T>, uint16_t>)>
VECOPS_VFUNC Mask<T> cmple(V a, V b) {
  return _mm_cmp_epu16_mask(a.v, b.v, _MM_CMPINT_LE);
}
template <TLV_DECL_VEC(V), typename T = Vec2Tag<V>, TL_IF(T::Bytes <= 16), TL_IF(is_any<TypeOf<T>, int32_t>)>
VECOPS_VFUNC Mask<T> cmple(V a, V b) {
  return _mm_cmp_epi32_mask(a.v, b.v, _MM_CMPINT_LE);
}
template <TLV_DECL_VEC(V), typename T = Vec2Tag<V>, TL_IF(T::Bytes <= 16), TL_IF(is_any<TypeOf<T>, uint32_t>)>
VECOPS_VFUNC Mask<T> cmple(V a, V b) {
  return _mm_cmp_epu32_mask(a.v, b.v, _MM_CMPINT_LE);
}
template <TLV_DECL_VEC(V), typename T = Vec2Tag<V>, TL_IF(T::Bytes <= 16), TL_IF(is_any<TypeOf<T>, int64_t>)>
VECOPS_VFUNC Mask<T> cmple(V a, V b) {
  return _mm_cmp_epi64_mask(a.v, b.v, _MM_CMPINT_LE);
}
template <TLV_DECL_VEC(V), typename T = Vec2Tag<V>, TL_IF(T::Bytes <= 16), TL_IF(is_any<TypeOf<T>, uint64_t>)>
VECOPS_VFUNC Mask<T> cmple(V a, V b) {
  return _mm_cmp_epu64_mask(a.v, b.v, _MM_CMPINT_LE);
}

template <TLV_DECL_VEC(V), typename T = Vec2Tag<V>, TL_IF(T::Bytes == 32), TL_IF(is_any<TypeOf<T>, float32_t>)>
VECOPS_VFUNC Mask<T> cmple(V a, V b) {
  return _mm256_cmp_ps_mask(a.v, b.v, _CMP_LE_OQ);
}
template <TLV_DECL_VEC(V), typename T = Vec2Tag<V>, TL_IF(T::Bytes == 32), TL_IF(is_any<TypeOf<T>, float64_t>)>
VECOPS_VFUNC Mask<T> cmple(V a, V b) {
  return _mm256_cmp_pd_mask(a.v, b.v, _CMP_LE_OQ);
}
template <TLV_DECL_VEC(V), typename T = Vec2Tag<V>, TL_IF(T::Bytes == 32), TL_IF(is_any<TypeOf<T>, int8_t>)>
VECOPS_VFUNC Mask<T> cmple(V a, V b) {
  return _mm256_cmp_epi8_mask(a.v, b.v, _MM_CMPINT_LE);
}
template <TLV_DECL_VEC(V), typename T = Vec2Tag<V>, TL_IF(T::Bytes == 32), TL_IF(is_any<TypeOf<T>, uint8_t>)>
VECOPS_VFUNC Mask<T> cmple(V a, V b) {
  return _mm256_cmp_epu8_mask(a.v, b.v, _MM_CMPINT_LE);
}
template <TLV_DECL_VEC(V), typename T = Vec2Tag<V>, TL_IF(T::Bytes == 32), TL_IF(is_any<TypeOf<T>, int16_t>)>
VECOPS_VFUNC Mask<T> cmple(V a, V b) {
  return _mm256_cmp_epi16_mask(a.v, b.v, _MM_CMPINT_LE);
}
template <TLV_DECL_VEC(V), typename T = Vec2Tag<V>, TL_IF(T::Bytes == 32), TL_IF(is_any<TypeOf<T>, uint16_t>)>
VECOPS_VFUNC Mask<T> cmple(V a, V b) {
  return _mm256_cmp_epu16_mask(a.v, b.v, _MM_CMPINT_LE);
}
template <TLV_DECL_VEC(V), typename T = Vec2Tag<V>, TL_IF(T::Bytes == 32), TL_IF(is_any<TypeOf<T>, int32_t>)>
VECOPS_VFUNC Mask<T> cmple(V a, V b) {
  return _mm256_cmp_epi32_mask(a.v, b.v, _MM_CMPINT_LE);
}
template <TLV_DECL_VEC(V), typename T = Vec2Tag<V>, TL_IF(T::Bytes == 32), TL_IF(is_any<TypeOf<T>, uint32_t>)>
VECOPS_VFUNC Mask<T> cmple(V a, V b) {
  return _mm256_cmp_epu32_mask(a.v, b.v, _MM_CMPINT_LE);
}
template <TLV_DECL_VEC(V), typename T = Vec2Tag<V>, TL_IF(T::Bytes == 32), TL_IF(is_any<TypeOf<T>, int64_t>)>
VECOPS_VFUNC Mask<T> cmple(V a, V b) {
  return _mm256_cmp_epi64_mask(a.v, b.v, _MM_CMPINT_LE);
}
template <TLV_DECL_VEC(V), typename T = Vec2Tag<V>, TL_IF(T::Bytes == 32), TL_IF(is_any<TypeOf<T>, uint64_t>)>
VECOPS_VFUNC Mask<T> cmple(V a, V b) {
  return _mm256_cmp_epu64_mask(a.v, b.v, _MM_CMPINT_LE);
}

template <TLV_DECL_VEC(V), typename T = Vec2Tag<V>, TL_IF(T::Bytes == 64), TL_IF(is_any<TypeOf<T>, float32_t>)>
VECOPS_VFUNC Mask<T> cmple(V a, V b) {
  return _mm512_cmp_ps_mask(a.v, b.v, _CMP_LE_OQ);
}
template <TLV_DECL_VEC(V), typename T = Vec2Tag<V>, TL_IF(T::Bytes == 64), TL_IF(is_any<TypeOf<T>, float64_t>)>
VECOPS_VFUNC Mask<T> cmple(V a, V b) {
  return _mm512_cmp_pd_mask(a.v, b.v, _CMP_LE_OQ);
}
template <TLV_DECL_VEC(V), typename T = Vec2Tag<V>, TL_IF(T::Bytes == 64), TL_IF(is_any<TypeOf<T>, int8_t>)>
VECOPS_VFUNC Mask<T> cmple(V a, V b) {
  return _mm512_cmp_epi8_mask(a.v, b.v, _MM_CMPINT_LE);
}
template <TLV_DECL_VEC(V), typename T = Vec2Tag<V>, TL_IF(T::Bytes == 64), TL_IF(is_any<TypeOf<T>, uint8_t>)>
VECOPS_VFUNC Mask<T> cmple(V a, V b) {
  return _mm512_cmp_epu8_mask(a.v, b.v, _MM_CMPINT_LE);
}
template <TLV_DECL_VEC(V), typename T = Vec2Tag<V>, TL_IF(T::Bytes == 64), TL_IF(is_any<TypeOf<T>, int16_t>)>
VECOPS_VFUNC Mask<T> cmple(V a, V b) {
  return _mm512_cmp_epi16_mask(a.v, b.v, _MM_CMPINT_LE);
}
template <TLV_DECL_VEC(V), typename T = Vec2Tag<V>, TL_IF(T::Bytes == 64), TL_IF(is_any<TypeOf<T>, uint16_t>)>
VECOPS_VFUNC Mask<T> cmple(V a, V b) {
  return _mm512_cmp_epu16_mask(a.v, b.v, _MM_CMPINT_LE);
}
template <TLV_DECL_VEC(V), typename T = Vec2Tag<V>, TL_IF(T::Bytes == 64), TL_IF(is_any<TypeOf<T>, int32_t>)>
VECOPS_VFUNC Mask<T> cmple(V a, V b) {
  return _mm512_cmp_epi32_mask(a.v, b.v, _MM_CMPINT_LE);
}
template <TLV_DECL_VEC(V), typename T = Vec2Tag<V>, TL_IF(T::Bytes == 64), TL_IF(is_any<TypeOf<T>, uint32_t>)>
VECOPS_VFUNC Mask<T> cmple(V a, V b) {
  return _mm512_cmp_epu32_mask(a.v, b.v, _MM_CMPINT_LE);
}
template <TLV_DECL_VEC(V), typename T = Vec2Tag<V>, TL_IF(T::Bytes == 64), TL_IF(is_any<TypeOf<T>, int64_t>)>
VECOPS_VFUNC Mask<T> cmple(V a, V b) {
  return _mm512_cmp_epi64_mask(a.v, b.v, _MM_CMPINT_LE);
}
template <TLV_DECL_VEC(V), typename T = Vec2Tag<V>, TL_IF(T::Bytes == 64), TL_IF(is_any<TypeOf<T>, uint64_t>)>
VECOPS_VFUNC Mask<T> cmple(V a, V b) {
  return _mm512_cmp_epu64_mask(a.v, b.v, _MM_CMPINT_LE);
}
#else // HAS_AVX512DQ
template <TLV_DECL_VEC(V), typename T = Vec2Tag<V>, TL_IF(T::Bytes <= 16), TL_IF(is_any<TypeOf<T>, float32_t>)>
VECOPS_VFUNC Mask<T> cmple(V a, V b) {
  return _mm_castps_si128(_mm_cmple_ps(a.v, b.v));
}
template <TLV_DECL_VEC(V), typename T = Vec2Tag<V>, TL_IF(T::Bytes <= 16), TL_IF(is_any<TypeOf<T>, float64_t>)>
VECOPS_VFUNC Mask<T> cmple(V a, V b) {
  return _mm_castpd_si128(_mm_cmple_pd(a.v, b.v));
}
template <TLV_DECL_VEC(V), typename T = Vec2Tag<V>, TL_IF(is_int<TypeOf<T>>)>
VECOPS_VFUNC Mask<T> cmple(V a, V b) {
  return word::bit_not(V{word::cmpgt(a, b).v}).v; // TODO dabian
}

#if VEC_WIDTH >= 256
template <TLV_DECL_VEC(V), typename T = Vec2Tag<V>, TL_IF(T::Bytes == 32), TL_IF(is_any<TypeOf<T>, float32_t>)>
VECOPS_VFUNC Mask<T> cmple(V a, V b) {
  return _mm256_castps_si256(_mm256_cmp_ps(a.v, b.v, _CMP_LE_OQ));
}
template <TLV_DECL_VEC(V), typename T = Vec2Tag<V>, TL_IF(T::Bytes == 32), TL_IF(is_any<TypeOf<T>, float64_t>)>
VECOPS_VFUNC Mask<T> cmple(V a, V b) {
  return _mm256_castpd_si256(_mm256_cmp_pd(a.v, b.v, _CMP_LE_OQ));
}
#endif // VEC_WIDTH >= 256

#if VEC_WIDTH >= 512
  #error "Unreachable"
#endif // VEC_WIDTH >= 512

template <TLV_DECL_VEC(V), typename T = Vec2Tag<V>, TL_IF(T::Bytes <= 16), TL_IF(is_any<TypeOf<T>, float16_t>)>
VECOPS_VFUNC Mask<T> cmple(V a, V b) {
  __m128 a_lo, a_hi, b_lo, b_hi;
  half_cvt::cvt_fp16_to_two_fp32(a.v, a_lo, a_hi);
  half_cvt::cvt_fp16_to_two_fp32(b.v, b_lo, b_hi);
  return _mm_packs_epi32(
      _mm_castps_si128(_mm_cmple_ps(a_lo, b_lo)),
      _mm_castps_si128(_mm_cmple_ps(a_hi, b_hi)));
}
template <TLV_DECL_VEC(V), typename T = Vec2Tag<V>, TL_IF(T::Bytes <= 16), TL_IF(is_any<TypeOf<T>, bfloat16_t>)>
VECOPS_VFUNC Mask<T> cmple(V a, V b) {
  __m128 a_lo, a_hi, b_lo, b_hi;
  half_cvt::cvt_bf16_to_two_fp32(a.v, a_lo, a_hi);
  half_cvt::cvt_bf16_to_two_fp32(b.v, b_lo, b_hi);
  return _mm_packs_epi32(
      _mm_castps_si128(_mm_cmple_ps(a_lo, b_lo)),
      _mm_castps_si128(_mm_cmple_ps(a_hi, b_hi)));
}
#if VEC_WIDTH >= 256
template <TLV_DECL_VEC(V), typename T = Vec2Tag<V>, TL_IF(T::Bytes == 32), TL_IF(is_any<TypeOf<T>, float16_t>)>
VECOPS_VFUNC Mask<T> cmple(V a, V b) {
  __m256 a_lo, a_hi, b_lo, b_hi;
  half_cvt::cvt_fp16_to_two_fp32(a.v, a_lo, a_hi);
  half_cvt::cvt_fp16_to_two_fp32(b.v, b_lo, b_hi);
  return _mm256_packs_epi32(
      _mm256_castps_si256(_mm256_cmp_ps(a_lo, b_lo, _CMP_LE_OQ)),
      _mm256_castps_si256(_mm256_cmp_ps(a_hi, b_hi, _CMP_LE_OQ)));
}
template <TLV_DECL_VEC(V), typename T = Vec2Tag<V>, TL_IF(T::Bytes == 32), TL_IF(is_any<TypeOf<T>, bfloat16_t>)>
VECOPS_VFUNC Mask<T> cmple(V a, V b) {
  __m256 a_lo, a_hi, b_lo, b_hi;
  half_cvt::cvt_bf16_to_two_fp32(a.v, a_lo, a_hi);
  half_cvt::cvt_bf16_to_two_fp32(b.v, b_lo, b_hi);
  return _mm256_packs_epi32(
      _mm256_castps_si256(_mm256_cmp_ps(a_lo, b_lo, _CMP_LE_OQ)),
      _mm256_castps_si256(_mm256_cmp_ps(a_hi, b_hi, _CMP_LE_OQ)));
}
#endif
#endif // HAS_AVX512DQ

#ifdef HAS_AVX512DQ
template <TLV_DECL_VEC(V), typename T = Vec2Tag<V>, TL_IF(T::Bytes <= 16), TL_IF(is_any<TypeOf<T>, float32_t>)>
VECOPS_VFUNC Mask<T> cmple(V a, V b, Mask<T> m) {
  return _mm_mask_cmp_ps_mask(m.v, a.v, b.v, _CMP_LE_OQ);
}
template <TLV_DECL_VEC(V), typename T = Vec2Tag<V>, TL_IF(T::Bytes <= 16), TL_IF(is_any<TypeOf<T>, float64_t>)>
VECOPS_VFUNC Mask<T> cmple(V a, V b, Mask<T> m) {
  return _mm_mask_cmp_pd_mask(m.v, a.v, b.v, _CMP_LE_OQ);
}
template <TLV_DECL_VEC(V), typename T = Vec2Tag<V>, TL_IF(T::Bytes <= 16), TL_IF(is_any<TypeOf<T>, int8_t>)>
VECOPS_VFUNC Mask<T> cmple(V a, V b, Mask<T> m) {
  return _mm_mask_cmp_epi8_mask(m.v, a.v, b.v, _MM_CMPINT_LE);
}
template <TLV_DECL_VEC(V), typename T = Vec2Tag<V>, TL_IF(T::Bytes <= 16), TL_IF(is_any<TypeOf<T>, uint8_t>)>
VECOPS_VFUNC Mask<T> cmple(V a, V b, Mask<T> m) {
  return _mm_mask_cmp_epu8_mask(m.v, a.v, b.v, _MM_CMPINT_LE);
}
template <TLV_DECL_VEC(V), typename T = Vec2Tag<V>, TL_IF(T::Bytes <= 16), TL_IF(is_any<TypeOf<T>, int16_t>)>
VECOPS_VFUNC Mask<T> cmple(V a, V b, Mask<T> m) {
  return _mm_mask_cmp_epi16_mask(m.v, a.v, b.v, _MM_CMPINT_LE);
}
template <TLV_DECL_VEC(V), typename T = Vec2Tag<V>, TL_IF(T::Bytes <= 16), TL_IF(is_any<TypeOf<T>, uint16_t>)>
VECOPS_VFUNC Mask<T> cmple(V a, V b, Mask<T> m) {
  return _mm_mask_cmp_epu16_mask(m.v, a.v, b.v, _MM_CMPINT_LE);
}
template <TLV_DECL_VEC(V), typename T = Vec2Tag<V>, TL_IF(T::Bytes <= 16), TL_IF(is_any<TypeOf<T>, int32_t>)>
VECOPS_VFUNC Mask<T> cmple(V a, V b, Mask<T> m) {
  return _mm_mask_cmp_epi32_mask(m.v, a.v, b.v, _MM_CMPINT_LE);
}
template <TLV_DECL_VEC(V), typename T = Vec2Tag<V>, TL_IF(T::Bytes <= 16), TL_IF(is_any<TypeOf<T>, uint32_t>)>
VECOPS_VFUNC Mask<T> cmple(V a, V b, Mask<T> m) {
  return _mm_mask_cmp_epu32_mask(m.v, a.v, b.v, _MM_CMPINT_LE);
}
template <TLV_DECL_VEC(V), typename T = Vec2Tag<V>, TL_IF(T::Bytes <= 16), TL_IF(is_any<TypeOf<T>, int64_t>)>
VECOPS_VFUNC Mask<T> cmple(V a, V b, Mask<T> m) {
  return _mm_mask_cmp_epi64_mask(m.v, a.v, b.v, _MM_CMPINT_LE);
}
template <TLV_DECL_VEC(V), typename T = Vec2Tag<V>, TL_IF(T::Bytes <= 16), TL_IF(is_any<TypeOf<T>, uint64_t>)>
VECOPS_VFUNC Mask<T> cmple(V a, V b, Mask<T> m) {
  return _mm_mask_cmp_epu64_mask(m.v, a.v, b.v, _MM_CMPINT_LE);
}

template <TLV_DECL_VEC(V), typename T = Vec2Tag<V>, TL_IF(T::Bytes == 32), TL_IF(is_any<TypeOf<T>, float32_t>)>
VECOPS_VFUNC Mask<T> cmple(V a, V b, Mask<T> m) {
  return _mm256_mask_cmp_ps_mask(m.v, a.v, b.v, _CMP_LE_OQ);
}
template <TLV_DECL_VEC(V), typename T = Vec2Tag<V>, TL_IF(T::Bytes == 32), TL_IF(is_any<TypeOf<T>, float64_t>)>
VECOPS_VFUNC Mask<T> cmple(V a, V b, Mask<T> m) {
  return _mm256_mask_cmp_pd_mask(m.v, a.v, b.v, _CMP_LE_OQ);
}
template <TLV_DECL_VEC(V), typename T = Vec2Tag<V>, TL_IF(T::Bytes == 32), TL_IF(is_any<TypeOf<T>, int8_t>)>
VECOPS_VFUNC Mask<T> cmple(V a, V b, Mask<T> m) {
  return _mm256_mask_cmp_epi8_mask(m.v, a.v, b.v, _MM_CMPINT_LE);
}
template <TLV_DECL_VEC(V), typename T = Vec2Tag<V>, TL_IF(T::Bytes == 32), TL_IF(is_any<TypeOf<T>, uint8_t>)>
VECOPS_VFUNC Mask<T> cmple(V a, V b, Mask<T> m) {
  return _mm256_mask_cmp_epu8_mask(m.v, a.v, b.v, _MM_CMPINT_LE);
}
template <TLV_DECL_VEC(V), typename T = Vec2Tag<V>, TL_IF(T::Bytes == 32), TL_IF(is_any<TypeOf<T>, int16_t>)>
VECOPS_VFUNC Mask<T> cmple(V a, V b, Mask<T> m) {
  return _mm256_mask_cmp_epi16_mask(m.v, a.v, b.v, _MM_CMPINT_LE);
}
template <TLV_DECL_VEC(V), typename T = Vec2Tag<V>, TL_IF(T::Bytes == 32), TL_IF(is_any<TypeOf<T>, uint16_t>)>
VECOPS_VFUNC Mask<T> cmple(V a, V b, Mask<T> m) {
  return _mm256_mask_cmp_epu16_mask(m.v, a.v, b.v, _MM_CMPINT_LE);
}
template <TLV_DECL_VEC(V), typename T = Vec2Tag<V>, TL_IF(T::Bytes == 32), TL_IF(is_any<TypeOf<T>, int32_t>)>
VECOPS_VFUNC Mask<T> cmple(V a, V b, Mask<T> m) {
  return _mm256_mask_cmp_epi32_mask(m.v, a.v, b.v, _MM_CMPINT_LE);
}
template <TLV_DECL_VEC(V), typename T = Vec2Tag<V>, TL_IF(T::Bytes == 32), TL_IF(is_any<TypeOf<T>, uint32_t>)>
VECOPS_VFUNC Mask<T> cmple(V a, V b, Mask<T> m) {
  return _mm256_mask_cmp_epu32_mask(m.v, a.v, b.v, _MM_CMPINT_LE);
}
template <TLV_DECL_VEC(V), typename T = Vec2Tag<V>, TL_IF(T::Bytes == 32), TL_IF(is_any<TypeOf<T>, int64_t>)>
VECOPS_VFUNC Mask<T> cmple(V a, V b, Mask<T> m) {
  return _mm256_mask_cmp_epi64_mask(m.v, a.v, b.v, _MM_CMPINT_LE);
}
template <TLV_DECL_VEC(V), typename T = Vec2Tag<V>, TL_IF(T::Bytes == 32), TL_IF(is_any<TypeOf<T>, uint64_t>)>
VECOPS_VFUNC Mask<T> cmple(V a, V b, Mask<T> m) {
  return _mm256_mask_cmp_epu64_mask(m.v, a.v, b.v, _MM_CMPINT_LE);
}

template <TLV_DECL_VEC(V), typename T = Vec2Tag<V>, TL_IF(T::Bytes == 64), TL_IF(is_any<TypeOf<T>, float32_t>)>
VECOPS_VFUNC Mask<T> cmple(V a, V b, Mask<T> m) {
  return _mm512_mask_cmp_ps_mask(m.v, a.v, b.v, _CMP_LE_OQ);
}
template <TLV_DECL_VEC(V), typename T = Vec2Tag<V>, TL_IF(T::Bytes == 64), TL_IF(is_any<TypeOf<T>, float64_t>)>
VECOPS_VFUNC Mask<T> cmple(V a, V b, Mask<T> m) {
  return _mm512_mask_cmp_pd_mask(m.v, a.v, b.v, _CMP_LE_OQ);
}
template <TLV_DECL_VEC(V), typename T = Vec2Tag<V>, TL_IF(T::Bytes == 64), TL_IF(is_any<TypeOf<T>, int8_t>)>
VECOPS_VFUNC Mask<T> cmple(V a, V b, Mask<T> m) {
  return _mm512_mask_cmp_epi8_mask(m.v, a.v, b.v, _MM_CMPINT_LE);
}
template <TLV_DECL_VEC(V), typename T = Vec2Tag<V>, TL_IF(T::Bytes == 64), TL_IF(is_any<TypeOf<T>, uint8_t>)>
VECOPS_VFUNC Mask<T> cmple(V a, V b, Mask<T> m) {
  return _mm512_mask_cmp_epu8_mask(m.v, a.v, b.v, _MM_CMPINT_LE);
}
template <TLV_DECL_VEC(V), typename T = Vec2Tag<V>, TL_IF(T::Bytes == 64), TL_IF(is_any<TypeOf<T>, int16_t>)>
VECOPS_VFUNC Mask<T> cmple(V a, V b, Mask<T> m) {
  return _mm512_mask_cmp_epi16_mask(m.v, a.v, b.v, _MM_CMPINT_LE);
}
template <TLV_DECL_VEC(V), typename T = Vec2Tag<V>, TL_IF(T::Bytes == 64), TL_IF(is_any<TypeOf<T>, uint16_t>)>
VECOPS_VFUNC Mask<T> cmple(V a, V b, Mask<T> m) {
  return _mm512_mask_cmp_epu16_mask(m.v, a.v, b.v, _MM_CMPINT_LE);
}
template <TLV_DECL_VEC(V), typename T = Vec2Tag<V>, TL_IF(T::Bytes == 64), TL_IF(is_any<TypeOf<T>, int32_t>)>
VECOPS_VFUNC Mask<T> cmple(V a, V b, Mask<T> m) {
  return _mm512_mask_cmp_epi32_mask(m.v, a.v, b.v, _MM_CMPINT_LE);
}
template <TLV_DECL_VEC(V), typename T = Vec2Tag<V>, TL_IF(T::Bytes == 64), TL_IF(is_any<TypeOf<T>, uint32_t>)>
VECOPS_VFUNC Mask<T> cmple(V a, V b, Mask<T> m) {
  return _mm512_mask_cmp_epu32_mask(m.v, a.v, b.v, _MM_CMPINT_LE);
}
template <TLV_DECL_VEC(V), typename T = Vec2Tag<V>, TL_IF(T::Bytes == 64), TL_IF(is_any<TypeOf<T>, int64_t>)>
VECOPS_VFUNC Mask<T> cmple(V a, V b, Mask<T> m) {
  return _mm512_mask_cmp_epi64_mask(m.v, a.v, b.v, _MM_CMPINT_LE);
}
template <TLV_DECL_VEC(V), typename T = Vec2Tag<V>, TL_IF(T::Bytes == 64), TL_IF(is_any<TypeOf<T>, uint64_t>)>
VECOPS_VFUNC Mask<T> cmple(V a, V b, Mask<T> m) {
  return _mm512_mask_cmp_epu64_mask(m.v, a.v, b.v, _MM_CMPINT_LE);
}
#ifdef HAS_AVX512_FP16
template <TLV_DECL_VEC(V), typename T = Vec2Tag<V>, TL_IF(T::Bytes <= 16), TL_IF(is_any<TypeOf<T>, float16_t>)>
VECOPS_VFUNC Mask<T> cmple(V a, V b) {
  return _mm_cmp_ph_mask(_mm_castsi128_ph(a.v), _mm_castsi128_ph(b.v), _CMP_LE_OQ);
}
#endif
#ifndef HAS_AVX512_FP16
template <TLV_DECL_VEC(V), typename T = Vec2Tag<V>, TL_IF(T::Bytes <= 16), TL_IF(is_any<TypeOf<T>, float16_t>)>
VECOPS_VFUNC Mask<T> cmple(V a, V b) {
  __m128 a_lo, a_hi, b_lo, b_hi;
  half_cvt::cvt_fp16_to_two_fp32(a.v, a_lo, a_hi);
  half_cvt::cvt_fp16_to_two_fp32(b.v, b_lo, b_hi);
  __mmask8 lo = _mm_cmp_ps_mask(a_lo, b_lo, _CMP_LE_OQ);
  __mmask8 hi = _mm_cmp_ps_mask(a_hi, b_hi, _CMP_LE_OQ);
  return (hi << 4) | lo;
}
#endif
template <TLV_DECL_VEC(V), typename T = Vec2Tag<V>, TL_IF(T::Bytes <= 16), TL_IF(is_any<TypeOf<T>, bfloat16_t>)>
VECOPS_VFUNC Mask<T> cmple(V a, V b) {
  __m128 a_lo, a_hi, b_lo, b_hi;
  half_cvt::cvt_bf16_to_two_fp32(a.v, a_lo, a_hi);
  half_cvt::cvt_bf16_to_two_fp32(b.v, b_lo, b_hi);
  __mmask8 lo = _mm_cmp_ps_mask(a_lo, b_lo, _CMP_LE_OQ);
  __mmask8 hi = _mm_cmp_ps_mask(a_hi, b_hi, _CMP_LE_OQ);
  return (hi << 4) | lo;
}
#if VEC_WIDTH >= 256
#ifdef HAS_AVX512_FP16
template <TLV_DECL_VEC(V), typename T = Vec2Tag<V>, TL_IF(T::Bytes == 32), TL_IF(is_any<TypeOf<T>, float16_t>)>
VECOPS_VFUNC Mask<T> cmple(V a, V b) {
  return _mm256_cmp_ph_mask(_mm256_castsi256_ph(a.v), _mm256_castsi256_ph(b.v), _CMP_LE_OQ);
}
#else
template <TLV_DECL_VEC(V), typename T = Vec2Tag<V>, TL_IF(T::Bytes == 32), TL_IF(is_any<TypeOf<T>, float16_t>)>
VECOPS_VFUNC Mask<T> cmple(V a, V b) {
  __m256 a_lo, a_hi, b_lo, b_hi;
  half_cvt::cvt_fp16_to_two_fp32(a.v, a_lo, a_hi);
  half_cvt::cvt_fp16_to_two_fp32(b.v, b_lo, b_hi);
  __mmask16 lo = _mm256_cmp_ps_mask(a_lo, b_lo, _CMP_LE_OQ);
  __mmask16 hi = _mm256_cmp_ps_mask(a_hi, b_hi, _CMP_LE_OQ);
  return (hi << 8) | lo;
}
#endif
template <TLV_DECL_VEC(V), typename T = Vec2Tag<V>, TL_IF(T::Bytes == 32), TL_IF(is_any<TypeOf<T>, bfloat16_t>)>
VECOPS_VFUNC Mask<T> cmple(V a, V b) {
  __m256 a_lo, a_hi, b_lo, b_hi;
  half_cvt::cvt_bf16_to_two_fp32(a.v, a_lo, a_hi);
  half_cvt::cvt_bf16_to_two_fp32(b.v, b_lo, b_hi);
  __mmask16 lo = _mm256_cmp_ps_mask(a_lo, b_lo, _CMP_LE_OQ);
  __mmask16 hi = _mm256_cmp_ps_mask(a_hi, b_hi, _CMP_LE_OQ);
  return (hi << 8) | lo;
}
#endif
#if VEC_WIDTH >= 512
#ifdef HAS_AVX512_FP16
template <TLV_DECL_VEC(V), typename T = Vec2Tag<V>, TL_IF(T::Bytes == 64), TL_IF(is_any<TypeOf<T>, float16_t>)>
VECOPS_VFUNC Mask<T> cmple(V a, V b) {
  return _mm512_cmp_ph_mask(_mm512_castsi512_ph(a.v), _mm512_castsi512_ph(b.v), _CMP_LE_OQ);
}
#else
template <TLV_DECL_VEC(V), typename T = Vec2Tag<V>, TL_IF(T::Bytes == 64), TL_IF(is_any<TypeOf<T>, float16_t>)>
VECOPS_VFUNC Mask<T> cmple(V a, V b) {
  __m256i a_lo256 = _mm512_castsi512_si256(a.v);
  __m256i a_hi256 = _mm512_extracti64x4_epi64(a.v, 1);
  __m256i b_lo256 = _mm512_castsi512_si256(b.v);
  __m256i b_hi256 = _mm512_extracti64x4_epi64(b.v, 1);
  __m256 a_lo0, a_lo1, a_hi0, a_hi1, b_lo0, b_lo1, b_hi0, b_hi1;
  half_cvt::cvt_fp16_to_two_fp32(a_lo256, a_lo0, a_lo1);
  half_cvt::cvt_fp16_to_two_fp32(a_hi256, a_hi0, a_hi1);
  half_cvt::cvt_fp16_to_two_fp32(b_lo256, b_lo0, b_lo1);
  half_cvt::cvt_fp16_to_two_fp32(b_hi256, b_hi0, b_hi1);
  __mmask16 m0 = _mm256_cmp_ps_mask(a_lo0, b_lo0, _CMP_LE_OQ);
  __mmask16 m1 = _mm256_cmp_ps_mask(a_lo1, b_lo1, _CMP_LE_OQ);
  __mmask16 m2 = _mm256_cmp_ps_mask(a_hi0, b_hi0, _CMP_LE_OQ);
  __mmask16 m3 = _mm256_cmp_ps_mask(a_hi1, b_hi1, _CMP_LE_OQ);
  return __mmask32(m0) | (__mmask32(m1) << 8) | (__mmask32(m2) << 16) | (__mmask32(m3) << 24);
}
#endif
template <TLV_DECL_VEC(V), typename T = Vec2Tag<V>, TL_IF(T::Bytes == 64), TL_IF(is_any<TypeOf<T>, bfloat16_t>)>
VECOPS_VFUNC Mask<T> cmple(V a, V b) {
  __m256i a_lo256 = _mm512_castsi512_si256(a.v);
  __m256i a_hi256 = _mm512_extracti64x4_epi64(a.v, 1);
  __m256i b_lo256 = _mm512_castsi512_si256(b.v);
  __m256i b_hi256 = _mm512_extracti64x4_epi64(b.v, 1);
  __m256 a_lo0, a_lo1, a_hi0, a_hi1, b_lo0, b_lo1, b_hi0, b_hi1;
  half_cvt::cvt_bf16_to_two_fp32(a_lo256, a_lo0, a_lo1);
  half_cvt::cvt_bf16_to_two_fp32(a_hi256, a_hi0, a_hi1);
  half_cvt::cvt_bf16_to_two_fp32(b_lo256, b_lo0, b_lo1);
  half_cvt::cvt_bf16_to_two_fp32(b_hi256, b_hi0, b_hi1);
  __mmask16 m0 = _mm256_cmp_ps_mask(a_lo0, b_lo0, _CMP_LE_OQ);
  __mmask16 m1 = _mm256_cmp_ps_mask(a_lo1, b_lo1, _CMP_LE_OQ);
  __mmask16 m2 = _mm256_cmp_ps_mask(a_hi0, b_hi0, _CMP_LE_OQ);
  __mmask16 m3 = _mm256_cmp_ps_mask(a_hi1, b_hi1, _CMP_LE_OQ);
  return __mmask32(m0) | (__mmask32(m1) << 8) | (__mmask32(m2) << 16) | (__mmask32(m3) << 24);
}
#endif
#ifdef HAS_AVX512_FP16
template <TLV_DECL_VEC(V), typename T = Vec2Tag<V>, TL_IF(T::Bytes <= 16), TL_IF(is_any<TypeOf<T>, float16_t>)>
VECOPS_VFUNC Mask<T> cmple(V a, V b, Mask<T> m) {
  return _mm_mask_cmp_ph_mask(m.v, _mm_castsi128_ph(a.v), _mm_castsi128_ph(b.v), _CMP_LE_OQ);
}
#endif
#ifndef HAS_AVX512_FP16
template <TLV_DECL_VEC(V), typename T = Vec2Tag<V>, TL_IF(T::Bytes <= 16), TL_IF(is_any<TypeOf<T>, float16_t>)>
VECOPS_VFUNC Mask<T> cmple(V a, V b, Mask<T> m) {
  __m128 a_lo, a_hi, b_lo, b_hi;
  half_cvt::cvt_fp16_to_two_fp32(a.v, a_lo, a_hi);
  half_cvt::cvt_fp16_to_two_fp32(b.v, b_lo, b_hi);
  __mmask8 lo = _mm_mask_cmp_ps_mask(m.v, a_lo, b_lo, _CMP_LE_OQ);
  __mmask8 hi = _mm_mask_cmp_ps_mask(m.v, a_hi, b_hi, _CMP_LE_OQ);
  return (hi << 4) | lo;
}
#endif
template <TLV_DECL_VEC(V), typename T = Vec2Tag<V>, TL_IF(T::Bytes <= 16), TL_IF(is_any<TypeOf<T>, bfloat16_t>)>
VECOPS_VFUNC Mask<T> cmple(V a, V b, Mask<T> m) {
  __m128 a_lo, a_hi, b_lo, b_hi;
  half_cvt::cvt_bf16_to_two_fp32(a.v, a_lo, a_hi);
  half_cvt::cvt_bf16_to_two_fp32(b.v, b_lo, b_hi);
  __mmask8 lo = _mm_mask_cmp_ps_mask(m.v, a_lo, b_lo, _CMP_LE_OQ);
  __mmask8 hi = _mm_mask_cmp_ps_mask(m.v, a_hi, b_hi, _CMP_LE_OQ);
  return (hi << 4) | lo;
}
#if VEC_WIDTH >= 256
#ifdef HAS_AVX512_FP16
template <TLV_DECL_VEC(V), typename T = Vec2Tag<V>, TL_IF(T::Bytes == 32), TL_IF(is_any<TypeOf<T>, float16_t>)>
VECOPS_VFUNC Mask<T> cmple(V a, V b, Mask<T> m) {
  return _mm256_mask_cmp_ph_mask(m.v, _mm256_castsi256_ph(a.v), _mm256_castsi256_ph(b.v), _CMP_LE_OQ);
}
#else
template <TLV_DECL_VEC(V), typename T = Vec2Tag<V>, TL_IF(T::Bytes == 32), TL_IF(is_any<TypeOf<T>, float16_t>)>
VECOPS_VFUNC Mask<T> cmple(V a, V b, Mask<T> m) {
  __m256 a_lo, a_hi, b_lo, b_hi;
  half_cvt::cvt_fp16_to_two_fp32(a.v, a_lo, a_hi);
  half_cvt::cvt_fp16_to_two_fp32(b.v, b_lo, b_hi);
  __mmask16 lo = _mm256_mask_cmp_ps_mask(m.v, a_lo, b_lo, _CMP_LE_OQ);
  __mmask16 hi = _mm256_mask_cmp_ps_mask(m.v, a_hi, b_hi, _CMP_LE_OQ);
  return (hi << 8) | lo;
}
#endif
template <TLV_DECL_VEC(V), typename T = Vec2Tag<V>, TL_IF(T::Bytes == 32), TL_IF(is_any<TypeOf<T>, bfloat16_t>)>
VECOPS_VFUNC Mask<T> cmple(V a, V b, Mask<T> m) {
  __m256 a_lo, a_hi, b_lo, b_hi;
  half_cvt::cvt_bf16_to_two_fp32(a.v, a_lo, a_hi);
  half_cvt::cvt_bf16_to_two_fp32(b.v, b_lo, b_hi);
  __mmask16 lo = _mm256_mask_cmp_ps_mask(m.v, a_lo, b_lo, _CMP_LE_OQ);
  __mmask16 hi = _mm256_mask_cmp_ps_mask(m.v, a_hi, b_hi, _CMP_LE_OQ);
  return (hi << 8) | lo;
}
#endif
#if VEC_WIDTH >= 512
#ifdef HAS_AVX512_FP16
template <TLV_DECL_VEC(V), typename T = Vec2Tag<V>, TL_IF(T::Bytes == 64), TL_IF(is_any<TypeOf<T>, float16_t>)>
VECOPS_VFUNC Mask<T> cmple(V a, V b, Mask<T> m) {
  return _mm512_mask_cmp_ph_mask(m.v, _mm512_castsi512_ph(a.v), _mm512_castsi512_ph(b.v), _CMP_LE_OQ);
}
#else
template <TLV_DECL_VEC(V), typename T = Vec2Tag<V>, TL_IF(T::Bytes == 64), TL_IF(is_any<TypeOf<T>, float16_t>)>
VECOPS_VFUNC Mask<T> cmple(V a, V b, Mask<T> m) {
  __m256i a_lo256 = _mm512_castsi512_si256(a.v);
  __m256i a_hi256 = _mm512_extracti64x4_epi64(a.v, 1);
  __m256i b_lo256 = _mm512_castsi512_si256(b.v);
  __m256i b_hi256 = _mm512_extracti64x4_epi64(b.v, 1);
  __mmask16 lo_m = __mmask16(m.v & 0xFFFF);
  __mmask16 hi_m = __mmask16(m.v >> 16);
  __m256 a_lo0, a_lo1, a_hi0, a_hi1, b_lo0, b_lo1, b_hi0, b_hi1;
  half_cvt::cvt_fp16_to_two_fp32(a_lo256, a_lo0, a_lo1);
  half_cvt::cvt_fp16_to_two_fp32(a_hi256, a_hi0, a_hi1);
  half_cvt::cvt_fp16_to_two_fp32(b_lo256, b_lo0, b_lo1);
  half_cvt::cvt_fp16_to_two_fp32(b_hi256, b_hi0, b_hi1);
  __mmask16 m0 = _mm256_mask_cmp_ps_mask(lo_m, a_lo0, b_lo0, _CMP_LE_OQ);
  __mmask16 m1 = _mm256_mask_cmp_ps_mask(lo_m, a_lo1, b_lo1, _CMP_LE_OQ);
  __mmask16 m2 = _mm256_mask_cmp_ps_mask(hi_m, a_hi0, b_hi0, _CMP_LE_OQ);
  __mmask16 m3 = _mm256_mask_cmp_ps_mask(hi_m, a_hi1, b_hi1, _CMP_LE_OQ);
  return __mmask32(m0) | (__mmask32(m1) << 8) | (__mmask32(m2) << 16) | (__mmask32(m3) << 24);
}
#endif
template <TLV_DECL_VEC(V), typename T = Vec2Tag<V>, TL_IF(T::Bytes == 64), TL_IF(is_any<TypeOf<T>, bfloat16_t>)>
VECOPS_VFUNC Mask<T> cmple(V a, V b, Mask<T> m) {
  __m256i a_lo256 = _mm512_castsi512_si256(a.v);
  __m256i a_hi256 = _mm512_extracti64x4_epi64(a.v, 1);
  __m256i b_lo256 = _mm512_castsi512_si256(b.v);
  __m256i b_hi256 = _mm512_extracti64x4_epi64(b.v, 1);
  __mmask16 lo_m = __mmask16(m.v & 0xFFFF);
  __mmask16 hi_m = __mmask16(m.v >> 16);
  __m256 a_lo0, a_lo1, a_hi0, a_hi1, b_lo0, b_lo1, b_hi0, b_hi1;
  half_cvt::cvt_bf16_to_two_fp32(a_lo256, a_lo0, a_lo1);
  half_cvt::cvt_bf16_to_two_fp32(a_hi256, a_hi0, a_hi1);
  half_cvt::cvt_bf16_to_two_fp32(b_lo256, b_lo0, b_lo1);
  half_cvt::cvt_bf16_to_two_fp32(b_hi256, b_hi0, b_hi1);
  __mmask16 m0 = _mm256_mask_cmp_ps_mask(lo_m, a_lo0, b_lo0, _CMP_LE_OQ);
  __mmask16 m1 = _mm256_mask_cmp_ps_mask(lo_m, a_lo1, b_lo1, _CMP_LE_OQ);
  __mmask16 m2 = _mm256_mask_cmp_ps_mask(hi_m, a_hi0, b_hi0, _CMP_LE_OQ);
  __mmask16 m3 = _mm256_mask_cmp_ps_mask(hi_m, a_hi1, b_hi1, _CMP_LE_OQ);
  return __mmask32(m0) | (__mmask32(m1) << 8) | (__mmask32(m2) << 16) | (__mmask32(m3) << 24);
}
#endif
#else // HAS_AVX512DQ
template <TLV_DECL_VEC(V), typename T = Vec2Tag<V>>
VECOPS_VFUNC Mask<T> cmple(V a, V b, Mask<T> m) {
  using Vi = Vec<ViewAs<Index<TypeOf<T>>, T>>;
  return word::bit_and(Vi{m.v}, Vi{word::cmple(a, b).v}).v;
}
#endif // HAS_AVX512DQ


/* ************************************************************************** */
//                      Compare Greater Than or Equals                        //
/* ************************************************************************** */
#ifdef HAS_AVX512DQ
template <TLV_DECL_VEC(V), typename T = Vec2Tag<V>, TL_IF(T::Bytes <= 16), TL_IF(is_any<TypeOf<T>, float32_t>)>
VECOPS_VFUNC Mask<T> cmpge(V a, V b) {
  return _mm_cmp_ps_mask(a.v, b.v, _CMP_GE_OQ);
}
template <TLV_DECL_VEC(V), typename T = Vec2Tag<V>, TL_IF(T::Bytes <= 16), TL_IF(is_any<TypeOf<T>, float64_t>)>
VECOPS_VFUNC Mask<T> cmpge(V a, V b) {
  return _mm_cmp_pd_mask(a.v, b.v, _CMP_GE_OQ);
}
template <TLV_DECL_VEC(V), typename T = Vec2Tag<V>, TL_IF(T::Bytes <= 16), TL_IF(is_any<TypeOf<T>, int8_t>)>
VECOPS_VFUNC Mask<T> cmpge(V a, V b) {
  return _mm_cmp_epi8_mask(a.v, b.v, _MM_CMPINT_NLT);
}
template <TLV_DECL_VEC(V), typename T = Vec2Tag<V>, TL_IF(T::Bytes <= 16), TL_IF(is_any<TypeOf<T>, uint8_t>)>
VECOPS_VFUNC Mask<T> cmpge(V a, V b) {
  return _mm_cmp_epu8_mask(a.v, b.v, _MM_CMPINT_NLT);
}
template <TLV_DECL_VEC(V), typename T = Vec2Tag<V>, TL_IF(T::Bytes <= 16), TL_IF(is_any<TypeOf<T>, int16_t>)>
VECOPS_VFUNC Mask<T> cmpge(V a, V b) {
  return _mm_cmp_epi16_mask(a.v, b.v, _MM_CMPINT_NLT);
}
template <TLV_DECL_VEC(V), typename T = Vec2Tag<V>, TL_IF(T::Bytes <= 16), TL_IF(is_any<TypeOf<T>, uint16_t>)>
VECOPS_VFUNC Mask<T> cmpge(V a, V b) {
  return _mm_cmp_epu16_mask(a.v, b.v, _MM_CMPINT_NLT);
}
template <TLV_DECL_VEC(V), typename T = Vec2Tag<V>, TL_IF(T::Bytes <= 16), TL_IF(is_any<TypeOf<T>, int32_t>)>
VECOPS_VFUNC Mask<T> cmpge(V a, V b) {
  return _mm_cmp_epi32_mask(a.v, b.v, _MM_CMPINT_NLT);
}
template <TLV_DECL_VEC(V), typename T = Vec2Tag<V>, TL_IF(T::Bytes <= 16), TL_IF(is_any<TypeOf<T>, uint32_t>)>
VECOPS_VFUNC Mask<T> cmpge(V a, V b) {
  return _mm_cmp_epu32_mask(a.v, b.v, _MM_CMPINT_NLT);
}
template <TLV_DECL_VEC(V), typename T = Vec2Tag<V>, TL_IF(T::Bytes <= 16), TL_IF(is_any<TypeOf<T>, int64_t>)>
VECOPS_VFUNC Mask<T> cmpge(V a, V b) {
  return _mm_cmp_epi64_mask(a.v, b.v, _MM_CMPINT_NLT);
}
template <TLV_DECL_VEC(V), typename T = Vec2Tag<V>, TL_IF(T::Bytes <= 16), TL_IF(is_any<TypeOf<T>, uint64_t>)>
VECOPS_VFUNC Mask<T> cmpge(V a, V b) {
  return _mm_cmp_epu64_mask(a.v, b.v, _MM_CMPINT_NLT);
}

template <TLV_DECL_VEC(V), typename T = Vec2Tag<V>, TL_IF(T::Bytes == 32), TL_IF(is_any<TypeOf<T>, float32_t>)>
VECOPS_VFUNC Mask<T> cmpge(V a, V b) {
  return _mm256_cmp_ps_mask(a.v, b.v, _CMP_GE_OQ);
}
template <TLV_DECL_VEC(V), typename T = Vec2Tag<V>, TL_IF(T::Bytes == 32), TL_IF(is_any<TypeOf<T>, float64_t>)>
VECOPS_VFUNC Mask<T> cmpge(V a, V b) {
  return _mm256_cmp_pd_mask(a.v, b.v, _CMP_GE_OQ);
}
template <TLV_DECL_VEC(V), typename T = Vec2Tag<V>, TL_IF(T::Bytes == 32), TL_IF(is_any<TypeOf<T>, int8_t>)>
VECOPS_VFUNC Mask<T> cmpge(V a, V b) {
  return _mm256_cmp_epi8_mask(a.v, b.v, _MM_CMPINT_NLT);
}
template <TLV_DECL_VEC(V), typename T = Vec2Tag<V>, TL_IF(T::Bytes == 32), TL_IF(is_any<TypeOf<T>, uint8_t>)>
VECOPS_VFUNC Mask<T> cmpge(V a, V b) {
  return _mm256_cmp_epu8_mask(a.v, b.v, _MM_CMPINT_NLT);
}
template <TLV_DECL_VEC(V), typename T = Vec2Tag<V>, TL_IF(T::Bytes == 32), TL_IF(is_any<TypeOf<T>, int16_t>)>
VECOPS_VFUNC Mask<T> cmpge(V a, V b) {
  return _mm256_cmp_epi16_mask(a.v, b.v, _MM_CMPINT_NLT);
}
template <TLV_DECL_VEC(V), typename T = Vec2Tag<V>, TL_IF(T::Bytes == 32), TL_IF(is_any<TypeOf<T>, uint16_t>)>
VECOPS_VFUNC Mask<T> cmpge(V a, V b) {
  return _mm256_cmp_epu16_mask(a.v, b.v, _MM_CMPINT_NLT);
}
template <TLV_DECL_VEC(V), typename T = Vec2Tag<V>, TL_IF(T::Bytes == 32), TL_IF(is_any<TypeOf<T>, int32_t>)>
VECOPS_VFUNC Mask<T> cmpge(V a, V b) {
  return _mm256_cmp_epi32_mask(a.v, b.v, _MM_CMPINT_NLT);
}
template <TLV_DECL_VEC(V), typename T = Vec2Tag<V>, TL_IF(T::Bytes == 32), TL_IF(is_any<TypeOf<T>, uint32_t>)>
VECOPS_VFUNC Mask<T> cmpge(V a, V b) {
  return _mm256_cmp_epu32_mask(a.v, b.v, _MM_CMPINT_NLT);
}
template <TLV_DECL_VEC(V), typename T = Vec2Tag<V>, TL_IF(T::Bytes == 32), TL_IF(is_any<TypeOf<T>, int64_t>)>
VECOPS_VFUNC Mask<T> cmpge(V a, V b) {
  return _mm256_cmp_epi64_mask(a.v, b.v, _MM_CMPINT_NLT);
}
template <TLV_DECL_VEC(V), typename T = Vec2Tag<V>, TL_IF(T::Bytes == 32), TL_IF(is_any<TypeOf<T>, uint64_t>)>
VECOPS_VFUNC Mask<T> cmpge(V a, V b) {
  return _mm256_cmp_epu64_mask(a.v, b.v, _MM_CMPINT_NLT);
}

template <TLV_DECL_VEC(V), typename T = Vec2Tag<V>, TL_IF(T::Bytes == 64), TL_IF(is_any<TypeOf<T>, float32_t>)>
VECOPS_VFUNC Mask<T> cmpge(V a, V b) {
  return _mm512_cmp_ps_mask(a.v, b.v, _CMP_GE_OQ);
}
template <TLV_DECL_VEC(V), typename T = Vec2Tag<V>, TL_IF(T::Bytes == 64), TL_IF(is_any<TypeOf<T>, float64_t>)>
VECOPS_VFUNC Mask<T> cmpge(V a, V b) {
  return _mm512_cmp_pd_mask(a.v, b.v, _CMP_GE_OQ);
}
template <TLV_DECL_VEC(V), typename T = Vec2Tag<V>, TL_IF(T::Bytes == 64), TL_IF(is_any<TypeOf<T>, int8_t>)>
VECOPS_VFUNC Mask<T> cmpge(V a, V b) {
  return _mm512_cmp_epi8_mask(a.v, b.v, _MM_CMPINT_NLT);
}
template <TLV_DECL_VEC(V), typename T = Vec2Tag<V>, TL_IF(T::Bytes == 64), TL_IF(is_any<TypeOf<T>, uint8_t>)>
VECOPS_VFUNC Mask<T> cmpge(V a, V b) {
  return _mm512_cmp_epu8_mask(a.v, b.v, _MM_CMPINT_NLT);
}
template <TLV_DECL_VEC(V), typename T = Vec2Tag<V>, TL_IF(T::Bytes == 64), TL_IF(is_any<TypeOf<T>, int16_t>)>
VECOPS_VFUNC Mask<T> cmpge(V a, V b) {
  return _mm512_cmp_epi16_mask(a.v, b.v, _MM_CMPINT_NLT);
}
template <TLV_DECL_VEC(V), typename T = Vec2Tag<V>, TL_IF(T::Bytes == 64), TL_IF(is_any<TypeOf<T>, uint16_t>)>
VECOPS_VFUNC Mask<T> cmpge(V a, V b) {
  return _mm512_cmp_epu16_mask(a.v, b.v, _MM_CMPINT_NLT);
}
template <TLV_DECL_VEC(V), typename T = Vec2Tag<V>, TL_IF(T::Bytes == 64), TL_IF(is_any<TypeOf<T>, int32_t>)>
VECOPS_VFUNC Mask<T> cmpge(V a, V b) {
  return _mm512_cmp_epi32_mask(a.v, b.v, _MM_CMPINT_NLT);
}
template <TLV_DECL_VEC(V), typename T = Vec2Tag<V>, TL_IF(T::Bytes == 64), TL_IF(is_any<TypeOf<T>, uint32_t>)>
VECOPS_VFUNC Mask<T> cmpge(V a, V b) {
  return _mm512_cmp_epu32_mask(a.v, b.v, _MM_CMPINT_NLT);
}
template <TLV_DECL_VEC(V), typename T = Vec2Tag<V>, TL_IF(T::Bytes == 64), TL_IF(is_any<TypeOf<T>, int64_t>)>
VECOPS_VFUNC Mask<T> cmpge(V a, V b) {
  return _mm512_cmp_epi64_mask(a.v, b.v, _MM_CMPINT_NLT);
}
template <TLV_DECL_VEC(V), typename T = Vec2Tag<V>, TL_IF(T::Bytes == 64), TL_IF(is_any<TypeOf<T>, uint64_t>)>
VECOPS_VFUNC Mask<T> cmpge(V a, V b) {
  return _mm512_cmp_epu64_mask(a.v, b.v, _MM_CMPINT_NLT);
}
#else // HAS_AVX512DQ
template <TLV_DECL_VEC(V), typename T = Vec2Tag<V>, TL_IF(T::Bytes <= 16), TL_IF(is_any<TypeOf<T>, float32_t>)>
VECOPS_VFUNC Mask<T> cmpge(V a, V b) {
  return _mm_castps_si128(_mm_cmpge_ps(a.v, b.v));
}
template <TLV_DECL_VEC(V), typename T = Vec2Tag<V>, TL_IF(T::Bytes <= 16), TL_IF(is_any<TypeOf<T>, float64_t>)>
VECOPS_VFUNC Mask<T> cmpge(V a, V b) {
  return _mm_castpd_si128(_mm_cmpge_pd(a.v, b.v));
}
template <TLV_DECL_VEC(V), typename T = Vec2Tag<V>, TL_IF(is_int<TypeOf<T>>)>
VECOPS_VFUNC Mask<T> cmpge(V a, V b) {
  return word::bit_not(V{word::cmplt(a, b).v}).v; // TODO dabian
}

#if VEC_WIDTH >= 256
template <TLV_DECL_VEC(V), typename T = Vec2Tag<V>, TL_IF(T::Bytes == 32), TL_IF(is_any<TypeOf<T>, float32_t>)>
VECOPS_VFUNC Mask<T> cmpge(V a, V b) {
  return _mm256_castps_si256(_mm256_cmp_ps(a.v, b.v, _CMP_GE_OQ));
}
template <TLV_DECL_VEC(V), typename T = Vec2Tag<V>, TL_IF(T::Bytes == 32), TL_IF(is_any<TypeOf<T>, float64_t>)>
VECOPS_VFUNC Mask<T> cmpge(V a, V b) {
  return _mm256_castpd_si256(_mm256_cmp_pd(a.v, b.v, _CMP_GE_OQ));
}
#endif // VEC_WIDTH >= 256

#if VEC_WIDTH >= 512
  #error "Unreachable"
#endif // VEC_WIDTH >= 512

template <TLV_DECL_VEC(V), typename T = Vec2Tag<V>, TL_IF(T::Bytes <= 16), TL_IF(is_any<TypeOf<T>, float16_t>)>
VECOPS_VFUNC Mask<T> cmpge(V a, V b) {
  __m128 a_lo, a_hi, b_lo, b_hi;
  half_cvt::cvt_fp16_to_two_fp32(a.v, a_lo, a_hi);
  half_cvt::cvt_fp16_to_two_fp32(b.v, b_lo, b_hi);
  return _mm_packs_epi32(
      _mm_castps_si128(_mm_cmpge_ps(a_lo, b_lo)),
      _mm_castps_si128(_mm_cmpge_ps(a_hi, b_hi)));
}
template <TLV_DECL_VEC(V), typename T = Vec2Tag<V>, TL_IF(T::Bytes <= 16), TL_IF(is_any<TypeOf<T>, bfloat16_t>)>
VECOPS_VFUNC Mask<T> cmpge(V a, V b) {
  __m128 a_lo, a_hi, b_lo, b_hi;
  half_cvt::cvt_bf16_to_two_fp32(a.v, a_lo, a_hi);
  half_cvt::cvt_bf16_to_two_fp32(b.v, b_lo, b_hi);
  return _mm_packs_epi32(
      _mm_castps_si128(_mm_cmpge_ps(a_lo, b_lo)),
      _mm_castps_si128(_mm_cmpge_ps(a_hi, b_hi)));
}
#if VEC_WIDTH >= 256
template <TLV_DECL_VEC(V), typename T = Vec2Tag<V>, TL_IF(T::Bytes == 32), TL_IF(is_any<TypeOf<T>, float16_t>)>
VECOPS_VFUNC Mask<T> cmpge(V a, V b) {
  __m256 a_lo, a_hi, b_lo, b_hi;
  half_cvt::cvt_fp16_to_two_fp32(a.v, a_lo, a_hi);
  half_cvt::cvt_fp16_to_two_fp32(b.v, b_lo, b_hi);
  return _mm256_packs_epi32(
      _mm256_castps_si256(_mm256_cmp_ps(a_lo, b_lo, _CMP_GE_OQ)),
      _mm256_castps_si256(_mm256_cmp_ps(a_hi, b_hi, _CMP_GE_OQ)));
}
template <TLV_DECL_VEC(V), typename T = Vec2Tag<V>, TL_IF(T::Bytes == 32), TL_IF(is_any<TypeOf<T>, bfloat16_t>)>
VECOPS_VFUNC Mask<T> cmpge(V a, V b) {
  __m256 a_lo, a_hi, b_lo, b_hi;
  half_cvt::cvt_bf16_to_two_fp32(a.v, a_lo, a_hi);
  half_cvt::cvt_bf16_to_two_fp32(b.v, b_lo, b_hi);
  return _mm256_packs_epi32(
      _mm256_castps_si256(_mm256_cmp_ps(a_lo, b_lo, _CMP_GE_OQ)),
      _mm256_castps_si256(_mm256_cmp_ps(a_hi, b_hi, _CMP_GE_OQ)));
}
#endif
#endif // HAS_AVX512DQ

#ifdef HAS_AVX512DQ
template <TLV_DECL_VEC(V), typename T = Vec2Tag<V>, TL_IF(T::Bytes <= 16), TL_IF(is_any<TypeOf<T>, float32_t>)>
VECOPS_VFUNC Mask<T> cmpge(V a, V b, Mask<T> m) {
  return _mm_mask_cmp_ps_mask(m.v, a.v, b.v, _CMP_GE_OQ);
}
template <TLV_DECL_VEC(V), typename T = Vec2Tag<V>, TL_IF(T::Bytes <= 16), TL_IF(is_any<TypeOf<T>, float64_t>)>
VECOPS_VFUNC Mask<T> cmpge(V a, V b, Mask<T> m) {
  return _mm_mask_cmp_pd_mask(m.v, a.v, b.v, _CMP_GE_OQ);
}
template <TLV_DECL_VEC(V), typename T = Vec2Tag<V>, TL_IF(T::Bytes <= 16), TL_IF(is_any<TypeOf<T>, int8_t>)>
VECOPS_VFUNC Mask<T> cmpge(V a, V b, Mask<T> m) {
  return _mm_mask_cmp_epi8_mask(m.v, a.v, b.v, _MM_CMPINT_NLT);
}
template <TLV_DECL_VEC(V), typename T = Vec2Tag<V>, TL_IF(T::Bytes <= 16), TL_IF(is_any<TypeOf<T>, uint8_t>)>
VECOPS_VFUNC Mask<T> cmpge(V a, V b, Mask<T> m) {
  return _mm_mask_cmp_epu8_mask(m.v, a.v, b.v, _MM_CMPINT_NLT);
}
template <TLV_DECL_VEC(V), typename T = Vec2Tag<V>, TL_IF(T::Bytes <= 16), TL_IF(is_any<TypeOf<T>, int16_t>)>
VECOPS_VFUNC Mask<T> cmpge(V a, V b, Mask<T> m) {
  return _mm_mask_cmp_epi16_mask(m.v, a.v, b.v, _MM_CMPINT_NLT);
}
template <TLV_DECL_VEC(V), typename T = Vec2Tag<V>, TL_IF(T::Bytes <= 16), TL_IF(is_any<TypeOf<T>, uint16_t>)>
VECOPS_VFUNC Mask<T> cmpge(V a, V b, Mask<T> m) {
  return _mm_mask_cmp_epu16_mask(m.v, a.v, b.v, _MM_CMPINT_NLT);
}
template <TLV_DECL_VEC(V), typename T = Vec2Tag<V>, TL_IF(T::Bytes <= 16), TL_IF(is_any<TypeOf<T>, int32_t>)>
VECOPS_VFUNC Mask<T> cmpge(V a, V b, Mask<T> m) {
  return _mm_mask_cmp_epi32_mask(m.v, a.v, b.v, _MM_CMPINT_NLT);
}
template <TLV_DECL_VEC(V), typename T = Vec2Tag<V>, TL_IF(T::Bytes <= 16), TL_IF(is_any<TypeOf<T>, uint32_t>)>
VECOPS_VFUNC Mask<T> cmpge(V a, V b, Mask<T> m) {
  return _mm_mask_cmp_epu32_mask(m.v, a.v, b.v, _MM_CMPINT_NLT);
}
template <TLV_DECL_VEC(V), typename T = Vec2Tag<V>, TL_IF(T::Bytes <= 16), TL_IF(is_any<TypeOf<T>, int64_t>)>
VECOPS_VFUNC Mask<T> cmpge(V a, V b, Mask<T> m) {
  return _mm_mask_cmp_epi64_mask(m.v, a.v, b.v, _MM_CMPINT_NLT);
}
template <TLV_DECL_VEC(V), typename T = Vec2Tag<V>, TL_IF(T::Bytes <= 16), TL_IF(is_any<TypeOf<T>, uint64_t>)>
VECOPS_VFUNC Mask<T> cmpge(V a, V b, Mask<T> m) {
  return _mm_mask_cmp_epu64_mask(m.v, a.v, b.v, _MM_CMPINT_NLT);
}

template <TLV_DECL_VEC(V), typename T = Vec2Tag<V>, TL_IF(T::Bytes == 32), TL_IF(is_any<TypeOf<T>, float32_t>)>
VECOPS_VFUNC Mask<T> cmpge(V a, V b, Mask<T> m) {
  return _mm256_mask_cmp_ps_mask(m.v, a.v, b.v, _CMP_GE_OQ);
}
template <TLV_DECL_VEC(V), typename T = Vec2Tag<V>, TL_IF(T::Bytes == 32), TL_IF(is_any<TypeOf<T>, float64_t>)>
VECOPS_VFUNC Mask<T> cmpge(V a, V b, Mask<T> m) {
  return _mm256_mask_cmp_pd_mask(m.v, a.v, b.v, _CMP_GE_OQ);
}
template <TLV_DECL_VEC(V), typename T = Vec2Tag<V>, TL_IF(T::Bytes == 32), TL_IF(is_any<TypeOf<T>, int8_t>)>
VECOPS_VFUNC Mask<T> cmpge(V a, V b, Mask<T> m) {
  return _mm256_mask_cmp_epi8_mask(m.v, a.v, b.v, _MM_CMPINT_NLT);
}
template <TLV_DECL_VEC(V), typename T = Vec2Tag<V>, TL_IF(T::Bytes == 32), TL_IF(is_any<TypeOf<T>, uint8_t>)>
VECOPS_VFUNC Mask<T> cmpge(V a, V b, Mask<T> m) {
  return _mm256_mask_cmp_epu8_mask(m.v, a.v, b.v, _MM_CMPINT_NLT);
}
template <TLV_DECL_VEC(V), typename T = Vec2Tag<V>, TL_IF(T::Bytes == 32), TL_IF(is_any<TypeOf<T>, int16_t>)>
VECOPS_VFUNC Mask<T> cmpge(V a, V b, Mask<T> m) {
  return _mm256_mask_cmp_epi16_mask(m.v, a.v, b.v, _MM_CMPINT_NLT);
}
template <TLV_DECL_VEC(V), typename T = Vec2Tag<V>, TL_IF(T::Bytes == 32), TL_IF(is_any<TypeOf<T>, uint16_t>)>
VECOPS_VFUNC Mask<T> cmpge(V a, V b, Mask<T> m) {
  return _mm256_mask_cmp_epu16_mask(m.v, a.v, b.v, _MM_CMPINT_NLT);
}
template <TLV_DECL_VEC(V), typename T = Vec2Tag<V>, TL_IF(T::Bytes == 32), TL_IF(is_any<TypeOf<T>, int32_t>)>
VECOPS_VFUNC Mask<T> cmpge(V a, V b, Mask<T> m) {
  return _mm256_mask_cmp_epi32_mask(m.v, a.v, b.v, _MM_CMPINT_NLT);
}
template <TLV_DECL_VEC(V), typename T = Vec2Tag<V>, TL_IF(T::Bytes == 32), TL_IF(is_any<TypeOf<T>, uint32_t>)>
VECOPS_VFUNC Mask<T> cmpge(V a, V b, Mask<T> m) {
  return _mm256_mask_cmp_epu32_mask(m.v, a.v, b.v, _MM_CMPINT_NLT);
}
template <TLV_DECL_VEC(V), typename T = Vec2Tag<V>, TL_IF(T::Bytes == 32), TL_IF(is_any<TypeOf<T>, int64_t>)>
VECOPS_VFUNC Mask<T> cmpge(V a, V b, Mask<T> m) {
  return _mm256_mask_cmp_epi64_mask(m.v, a.v, b.v, _MM_CMPINT_NLT);
}
template <TLV_DECL_VEC(V), typename T = Vec2Tag<V>, TL_IF(T::Bytes == 32), TL_IF(is_any<TypeOf<T>, uint64_t>)>
VECOPS_VFUNC Mask<T> cmpge(V a, V b, Mask<T> m) {
  return _mm256_mask_cmp_epu64_mask(m.v, a.v, b.v, _MM_CMPINT_NLT);
}

template <TLV_DECL_VEC(V), typename T = Vec2Tag<V>, TL_IF(T::Bytes == 64), TL_IF(is_any<TypeOf<T>, float32_t>)>
VECOPS_VFUNC Mask<T> cmpge(V a, V b, Mask<T> m) {
  return _mm512_mask_cmp_ps_mask(m.v, a.v, b.v, _CMP_GE_OQ);
}
template <TLV_DECL_VEC(V), typename T = Vec2Tag<V>, TL_IF(T::Bytes == 64), TL_IF(is_any<TypeOf<T>, float64_t>)>
VECOPS_VFUNC Mask<T> cmpge(V a, V b, Mask<T> m) {
  return _mm512_mask_cmp_pd_mask(m.v, a.v, b.v, _CMP_GE_OQ);
}
template <TLV_DECL_VEC(V), typename T = Vec2Tag<V>, TL_IF(T::Bytes == 64), TL_IF(is_any<TypeOf<T>, int8_t>)>
VECOPS_VFUNC Mask<T> cmpge(V a, V b, Mask<T> m) {
  return _mm512_mask_cmp_epi8_mask(m.v, a.v, b.v, _MM_CMPINT_NLT);
}
template <TLV_DECL_VEC(V), typename T = Vec2Tag<V>, TL_IF(T::Bytes == 64), TL_IF(is_any<TypeOf<T>, uint8_t>)>
VECOPS_VFUNC Mask<T> cmpge(V a, V b, Mask<T> m) {
  return _mm512_mask_cmp_epu8_mask(m.v, a.v, b.v, _MM_CMPINT_NLT);
}
template <TLV_DECL_VEC(V), typename T = Vec2Tag<V>, TL_IF(T::Bytes == 64), TL_IF(is_any<TypeOf<T>, int16_t>)>
VECOPS_VFUNC Mask<T> cmpge(V a, V b, Mask<T> m) {
  return _mm512_mask_cmp_epi16_mask(m.v, a.v, b.v, _MM_CMPINT_NLT);
}
template <TLV_DECL_VEC(V), typename T = Vec2Tag<V>, TL_IF(T::Bytes == 64), TL_IF(is_any<TypeOf<T>, uint16_t>)>
VECOPS_VFUNC Mask<T> cmpge(V a, V b, Mask<T> m) {
  return _mm512_mask_cmp_epu16_mask(m.v, a.v, b.v, _MM_CMPINT_NLT);
}
template <TLV_DECL_VEC(V), typename T = Vec2Tag<V>, TL_IF(T::Bytes == 64), TL_IF(is_any<TypeOf<T>, int32_t>)>
VECOPS_VFUNC Mask<T> cmpge(V a, V b, Mask<T> m) {
  return _mm512_mask_cmp_epi32_mask(m.v, a.v, b.v, _MM_CMPINT_NLT);
}
template <TLV_DECL_VEC(V), typename T = Vec2Tag<V>, TL_IF(T::Bytes == 64), TL_IF(is_any<TypeOf<T>, uint32_t>)>
VECOPS_VFUNC Mask<T> cmpge(V a, V b, Mask<T> m) {
  return _mm512_mask_cmp_epu32_mask(m.v, a.v, b.v, _MM_CMPINT_NLT);
}
template <TLV_DECL_VEC(V), typename T = Vec2Tag<V>, TL_IF(T::Bytes == 64), TL_IF(is_any<TypeOf<T>, int64_t>)>
VECOPS_VFUNC Mask<T> cmpge(V a, V b, Mask<T> m) {
  return _mm512_mask_cmp_epi64_mask(m.v, a.v, b.v, _MM_CMPINT_NLT);
}
template <TLV_DECL_VEC(V), typename T = Vec2Tag<V>, TL_IF(T::Bytes == 64), TL_IF(is_any<TypeOf<T>, uint64_t>)>
VECOPS_VFUNC Mask<T> cmpge(V a, V b, Mask<T> m) {
  return _mm512_mask_cmp_epu64_mask(m.v, a.v, b.v, _MM_CMPINT_NLT);
}
#ifdef HAS_AVX512_FP16
template <TLV_DECL_VEC(V), typename T = Vec2Tag<V>, TL_IF(T::Bytes <= 16), TL_IF(is_any<TypeOf<T>, float16_t>)>
VECOPS_VFUNC Mask<T> cmpge(V a, V b) {
  return _mm_cmp_ph_mask(_mm_castsi128_ph(a.v), _mm_castsi128_ph(b.v), _CMP_GE_OQ);
}
#endif
#ifndef HAS_AVX512_FP16
template <TLV_DECL_VEC(V), typename T = Vec2Tag<V>, TL_IF(T::Bytes <= 16), TL_IF(is_any<TypeOf<T>, float16_t>)>
VECOPS_VFUNC Mask<T> cmpge(V a, V b) {
  __m128 a_lo, a_hi, b_lo, b_hi;
  half_cvt::cvt_fp16_to_two_fp32(a.v, a_lo, a_hi);
  half_cvt::cvt_fp16_to_two_fp32(b.v, b_lo, b_hi);
  __mmask8 lo = _mm_cmp_ps_mask(a_lo, b_lo, _CMP_GE_OQ);
  __mmask8 hi = _mm_cmp_ps_mask(a_hi, b_hi, _CMP_GE_OQ);
  return (hi << 4) | lo;
}
#endif
template <TLV_DECL_VEC(V), typename T = Vec2Tag<V>, TL_IF(T::Bytes <= 16), TL_IF(is_any<TypeOf<T>, bfloat16_t>)>
VECOPS_VFUNC Mask<T> cmpge(V a, V b) {
  __m128 a_lo, a_hi, b_lo, b_hi;
  half_cvt::cvt_bf16_to_two_fp32(a.v, a_lo, a_hi);
  half_cvt::cvt_bf16_to_two_fp32(b.v, b_lo, b_hi);
  __mmask8 lo = _mm_cmp_ps_mask(a_lo, b_lo, _CMP_GE_OQ);
  __mmask8 hi = _mm_cmp_ps_mask(a_hi, b_hi, _CMP_GE_OQ);
  return (hi << 4) | lo;
}
#if VEC_WIDTH >= 256
#ifdef HAS_AVX512_FP16
template <TLV_DECL_VEC(V), typename T = Vec2Tag<V>, TL_IF(T::Bytes == 32), TL_IF(is_any<TypeOf<T>, float16_t>)>
VECOPS_VFUNC Mask<T> cmpge(V a, V b) {
  return _mm256_cmp_ph_mask(_mm256_castsi256_ph(a.v), _mm256_castsi256_ph(b.v), _CMP_GE_OQ);
}
#else
template <TLV_DECL_VEC(V), typename T = Vec2Tag<V>, TL_IF(T::Bytes == 32), TL_IF(is_any<TypeOf<T>, float16_t>)>
VECOPS_VFUNC Mask<T> cmpge(V a, V b) {
  __m256 a_lo, a_hi, b_lo, b_hi;
  half_cvt::cvt_fp16_to_two_fp32(a.v, a_lo, a_hi);
  half_cvt::cvt_fp16_to_two_fp32(b.v, b_lo, b_hi);
  __mmask16 lo = _mm256_cmp_ps_mask(a_lo, b_lo, _CMP_GE_OQ);
  __mmask16 hi = _mm256_cmp_ps_mask(a_hi, b_hi, _CMP_GE_OQ);
  return (hi << 8) | lo;
}
#endif
template <TLV_DECL_VEC(V), typename T = Vec2Tag<V>, TL_IF(T::Bytes == 32), TL_IF(is_any<TypeOf<T>, bfloat16_t>)>
VECOPS_VFUNC Mask<T> cmpge(V a, V b) {
  __m256 a_lo, a_hi, b_lo, b_hi;
  half_cvt::cvt_bf16_to_two_fp32(a.v, a_lo, a_hi);
  half_cvt::cvt_bf16_to_two_fp32(b.v, b_lo, b_hi);
  __mmask16 lo = _mm256_cmp_ps_mask(a_lo, b_lo, _CMP_GE_OQ);
  __mmask16 hi = _mm256_cmp_ps_mask(a_hi, b_hi, _CMP_GE_OQ);
  return (hi << 8) | lo;
}
#endif
#if VEC_WIDTH >= 512
#ifdef HAS_AVX512_FP16
template <TLV_DECL_VEC(V), typename T = Vec2Tag<V>, TL_IF(T::Bytes == 64), TL_IF(is_any<TypeOf<T>, float16_t>)>
VECOPS_VFUNC Mask<T> cmpge(V a, V b) {
  return _mm512_cmp_ph_mask(_mm512_castsi512_ph(a.v), _mm512_castsi512_ph(b.v), _CMP_GE_OQ);
}
#else
template <TLV_DECL_VEC(V), typename T = Vec2Tag<V>, TL_IF(T::Bytes == 64), TL_IF(is_any<TypeOf<T>, float16_t>)>
VECOPS_VFUNC Mask<T> cmpge(V a, V b) {
  __m256i a_lo256 = _mm512_castsi512_si256(a.v);
  __m256i a_hi256 = _mm512_extracti64x4_epi64(a.v, 1);
  __m256i b_lo256 = _mm512_castsi512_si256(b.v);
  __m256i b_hi256 = _mm512_extracti64x4_epi64(b.v, 1);
  __m256 a_lo0, a_lo1, a_hi0, a_hi1, b_lo0, b_lo1, b_hi0, b_hi1;
  half_cvt::cvt_fp16_to_two_fp32(a_lo256, a_lo0, a_lo1);
  half_cvt::cvt_fp16_to_two_fp32(a_hi256, a_hi0, a_hi1);
  half_cvt::cvt_fp16_to_two_fp32(b_lo256, b_lo0, b_lo1);
  half_cvt::cvt_fp16_to_two_fp32(b_hi256, b_hi0, b_hi1);
  __mmask16 m0 = _mm256_cmp_ps_mask(a_lo0, b_lo0, _CMP_GE_OQ);
  __mmask16 m1 = _mm256_cmp_ps_mask(a_lo1, b_lo1, _CMP_GE_OQ);
  __mmask16 m2 = _mm256_cmp_ps_mask(a_hi0, b_hi0, _CMP_GE_OQ);
  __mmask16 m3 = _mm256_cmp_ps_mask(a_hi1, b_hi1, _CMP_GE_OQ);
  return __mmask32(m0) | (__mmask32(m1) << 8) | (__mmask32(m2) << 16) | (__mmask32(m3) << 24);
}
#endif
template <TLV_DECL_VEC(V), typename T = Vec2Tag<V>, TL_IF(T::Bytes == 64), TL_IF(is_any<TypeOf<T>, bfloat16_t>)>
VECOPS_VFUNC Mask<T> cmpge(V a, V b) {
  __m256i a_lo256 = _mm512_castsi512_si256(a.v);
  __m256i a_hi256 = _mm512_extracti64x4_epi64(a.v, 1);
  __m256i b_lo256 = _mm512_castsi512_si256(b.v);
  __m256i b_hi256 = _mm512_extracti64x4_epi64(b.v, 1);
  __m256 a_lo0, a_lo1, a_hi0, a_hi1, b_lo0, b_lo1, b_hi0, b_hi1;
  half_cvt::cvt_bf16_to_two_fp32(a_lo256, a_lo0, a_lo1);
  half_cvt::cvt_bf16_to_two_fp32(a_hi256, a_hi0, a_hi1);
  half_cvt::cvt_bf16_to_two_fp32(b_lo256, b_lo0, b_lo1);
  half_cvt::cvt_bf16_to_two_fp32(b_hi256, b_hi0, b_hi1);
  __mmask16 m0 = _mm256_cmp_ps_mask(a_lo0, b_lo0, _CMP_GE_OQ);
  __mmask16 m1 = _mm256_cmp_ps_mask(a_lo1, b_lo1, _CMP_GE_OQ);
  __mmask16 m2 = _mm256_cmp_ps_mask(a_hi0, b_hi0, _CMP_GE_OQ);
  __mmask16 m3 = _mm256_cmp_ps_mask(a_hi1, b_hi1, _CMP_GE_OQ);
  return __mmask32(m0) | (__mmask32(m1) << 8) | (__mmask32(m2) << 16) | (__mmask32(m3) << 24);
}
#endif
#ifdef HAS_AVX512_FP16
template <TLV_DECL_VEC(V), typename T = Vec2Tag<V>, TL_IF(T::Bytes <= 16), TL_IF(is_any<TypeOf<T>, float16_t>)>
VECOPS_VFUNC Mask<T> cmpge(V a, V b, Mask<T> m) {
  return _mm_mask_cmp_ph_mask(m.v, _mm_castsi128_ph(a.v), _mm_castsi128_ph(b.v), _CMP_GE_OQ);
}
#endif
#ifndef HAS_AVX512_FP16
template <TLV_DECL_VEC(V), typename T = Vec2Tag<V>, TL_IF(T::Bytes <= 16), TL_IF(is_any<TypeOf<T>, float16_t>)>
VECOPS_VFUNC Mask<T> cmpge(V a, V b, Mask<T> m) {
  __m128 a_lo, a_hi, b_lo, b_hi;
  half_cvt::cvt_fp16_to_two_fp32(a.v, a_lo, a_hi);
  half_cvt::cvt_fp16_to_two_fp32(b.v, b_lo, b_hi);
  __mmask8 lo = _mm_mask_cmp_ps_mask(m.v, a_lo, b_lo, _CMP_GE_OQ);
  __mmask8 hi = _mm_mask_cmp_ps_mask(m.v, a_hi, b_hi, _CMP_GE_OQ);
  return (hi << 4) | lo;
}
#endif
template <TLV_DECL_VEC(V), typename T = Vec2Tag<V>, TL_IF(T::Bytes <= 16), TL_IF(is_any<TypeOf<T>, bfloat16_t>)>
VECOPS_VFUNC Mask<T> cmpge(V a, V b, Mask<T> m) {
  __m128 a_lo, a_hi, b_lo, b_hi;
  half_cvt::cvt_bf16_to_two_fp32(a.v, a_lo, a_hi);
  half_cvt::cvt_bf16_to_two_fp32(b.v, b_lo, b_hi);
  __mmask8 lo = _mm_mask_cmp_ps_mask(m.v, a_lo, b_lo, _CMP_GE_OQ);
  __mmask8 hi = _mm_mask_cmp_ps_mask(m.v, a_hi, b_hi, _CMP_GE_OQ);
  return (hi << 4) | lo;
}
#if VEC_WIDTH >= 256
#ifdef HAS_AVX512_FP16
template <TLV_DECL_VEC(V), typename T = Vec2Tag<V>, TL_IF(T::Bytes == 32), TL_IF(is_any<TypeOf<T>, float16_t>)>
VECOPS_VFUNC Mask<T> cmpge(V a, V b, Mask<T> m) {
  return _mm256_mask_cmp_ph_mask(m.v, _mm256_castsi256_ph(a.v), _mm256_castsi256_ph(b.v), _CMP_GE_OQ);
}
#else
template <TLV_DECL_VEC(V), typename T = Vec2Tag<V>, TL_IF(T::Bytes == 32), TL_IF(is_any<TypeOf<T>, float16_t>)>
VECOPS_VFUNC Mask<T> cmpge(V a, V b, Mask<T> m) {
  __m256 a_lo, a_hi, b_lo, b_hi;
  half_cvt::cvt_fp16_to_two_fp32(a.v, a_lo, a_hi);
  half_cvt::cvt_fp16_to_two_fp32(b.v, b_lo, b_hi);
  __mmask16 lo = _mm256_mask_cmp_ps_mask(m.v, a_lo, b_lo, _CMP_GE_OQ);
  __mmask16 hi = _mm256_mask_cmp_ps_mask(m.v, a_hi, b_hi, _CMP_GE_OQ);
  return (hi << 8) | lo;
}
#endif
template <TLV_DECL_VEC(V), typename T = Vec2Tag<V>, TL_IF(T::Bytes == 32), TL_IF(is_any<TypeOf<T>, bfloat16_t>)>
VECOPS_VFUNC Mask<T> cmpge(V a, V b, Mask<T> m) {
  __m256 a_lo, a_hi, b_lo, b_hi;
  half_cvt::cvt_bf16_to_two_fp32(a.v, a_lo, a_hi);
  half_cvt::cvt_bf16_to_two_fp32(b.v, b_lo, b_hi);
  __mmask16 lo = _mm256_mask_cmp_ps_mask(m.v, a_lo, b_lo, _CMP_GE_OQ);
  __mmask16 hi = _mm256_mask_cmp_ps_mask(m.v, a_hi, b_hi, _CMP_GE_OQ);
  return (hi << 8) | lo;
}
#endif
#if VEC_WIDTH >= 512
#ifdef HAS_AVX512_FP16
template <TLV_DECL_VEC(V), typename T = Vec2Tag<V>, TL_IF(T::Bytes == 64), TL_IF(is_any<TypeOf<T>, float16_t>)>
VECOPS_VFUNC Mask<T> cmpge(V a, V b, Mask<T> m) {
  return _mm512_mask_cmp_ph_mask(m.v, _mm512_castsi512_ph(a.v), _mm512_castsi512_ph(b.v), _CMP_GE_OQ);
}
#else
template <TLV_DECL_VEC(V), typename T = Vec2Tag<V>, TL_IF(T::Bytes == 64), TL_IF(is_any<TypeOf<T>, float16_t>)>
VECOPS_VFUNC Mask<T> cmpge(V a, V b, Mask<T> m) {
  __m256i a_lo256 = _mm512_castsi512_si256(a.v);
  __m256i a_hi256 = _mm512_extracti64x4_epi64(a.v, 1);
  __m256i b_lo256 = _mm512_castsi512_si256(b.v);
  __m256i b_hi256 = _mm512_extracti64x4_epi64(b.v, 1);
  __mmask16 lo_m = __mmask16(m.v & 0xFFFF);
  __mmask16 hi_m = __mmask16(m.v >> 16);
  __m256 a_lo0, a_lo1, a_hi0, a_hi1, b_lo0, b_lo1, b_hi0, b_hi1;
  half_cvt::cvt_fp16_to_two_fp32(a_lo256, a_lo0, a_lo1);
  half_cvt::cvt_fp16_to_two_fp32(a_hi256, a_hi0, a_hi1);
  half_cvt::cvt_fp16_to_two_fp32(b_lo256, b_lo0, b_lo1);
  half_cvt::cvt_fp16_to_two_fp32(b_hi256, b_hi0, b_hi1);
  __mmask16 m0 = _mm256_mask_cmp_ps_mask(lo_m, a_lo0, b_lo0, _CMP_GE_OQ);
  __mmask16 m1 = _mm256_mask_cmp_ps_mask(lo_m, a_lo1, b_lo1, _CMP_GE_OQ);
  __mmask16 m2 = _mm256_mask_cmp_ps_mask(hi_m, a_hi0, b_hi0, _CMP_GE_OQ);
  __mmask16 m3 = _mm256_mask_cmp_ps_mask(hi_m, a_hi1, b_hi1, _CMP_GE_OQ);
  return __mmask32(m0) | (__mmask32(m1) << 8) | (__mmask32(m2) << 16) | (__mmask32(m3) << 24);
}
#endif
template <TLV_DECL_VEC(V), typename T = Vec2Tag<V>, TL_IF(T::Bytes == 64), TL_IF(is_any<TypeOf<T>, bfloat16_t>)>
VECOPS_VFUNC Mask<T> cmpge(V a, V b, Mask<T> m) {
  __m256i a_lo256 = _mm512_castsi512_si256(a.v);
  __m256i a_hi256 = _mm512_extracti64x4_epi64(a.v, 1);
  __m256i b_lo256 = _mm512_castsi512_si256(b.v);
  __m256i b_hi256 = _mm512_extracti64x4_epi64(b.v, 1);
  __mmask16 lo_m = __mmask16(m.v & 0xFFFF);
  __mmask16 hi_m = __mmask16(m.v >> 16);
  __m256 a_lo0, a_lo1, a_hi0, a_hi1, b_lo0, b_lo1, b_hi0, b_hi1;
  half_cvt::cvt_bf16_to_two_fp32(a_lo256, a_lo0, a_lo1);
  half_cvt::cvt_bf16_to_two_fp32(a_hi256, a_hi0, a_hi1);
  half_cvt::cvt_bf16_to_two_fp32(b_lo256, b_lo0, b_lo1);
  half_cvt::cvt_bf16_to_two_fp32(b_hi256, b_hi0, b_hi1);
  __mmask16 m0 = _mm256_mask_cmp_ps_mask(lo_m, a_lo0, b_lo0, _CMP_GE_OQ);
  __mmask16 m1 = _mm256_mask_cmp_ps_mask(lo_m, a_lo1, b_lo1, _CMP_GE_OQ);
  __mmask16 m2 = _mm256_mask_cmp_ps_mask(hi_m, a_hi0, b_hi0, _CMP_GE_OQ);
  __mmask16 m3 = _mm256_mask_cmp_ps_mask(hi_m, a_hi1, b_hi1, _CMP_GE_OQ);
  return __mmask32(m0) | (__mmask32(m1) << 8) | (__mmask32(m2) << 16) | (__mmask32(m3) << 24);
}
#endif
#else // HAS_AVX512DQ
template <TLV_DECL_VEC(V), typename T = Vec2Tag<V>>
VECOPS_VFUNC Mask<T> cmpge(V a, V b, Mask<T> m) {
  using Vi = Vec<ViewAs<Index<TypeOf<T>>, T>>;
  return word::bit_and(Vi{m.v}, Vi{word::cmpge(a, b).v}).v;
}
#endif // HAS_AVX512DQ


/* ************************************************************************** */
//                                  Is NaN                                    //
/* ************************************************************************** */
#ifdef HAS_AVX512DQ
template <TLV_DECL_VEC(V), typename T = Vec2Tag<V>, TL_IF(T::Bytes <= 16), TL_IF(is_any<TypeOf<T>, float32_t>)>
VECOPS_VFUNC Mask<T> isnan(V v) {
  return _mm_cmp_ps_mask(v.v, v.v, _CMP_UNORD_Q);
}
template <TLV_DECL_VEC(V), typename T = Vec2Tag<V>, TL_IF(T::Bytes <= 16), TL_IF(is_any<TypeOf<T>, float64_t>)>
VECOPS_VFUNC Mask<T> isnan(V v) {
  return _mm_cmp_pd_mask(v.v, v.v, _CMP_UNORD_Q);
}

template <TLV_DECL_VEC(V), typename T = Vec2Tag<V>, TL_IF(T::Bytes == 32), TL_IF(is_any<TypeOf<T>, float32_t>)>
VECOPS_VFUNC Mask<T> isnan(V v) {
  return _mm256_cmp_ps_mask(v.v, v.v, _CMP_UNORD_Q);
}
template <TLV_DECL_VEC(V), typename T = Vec2Tag<V>, TL_IF(T::Bytes == 32), TL_IF(is_any<TypeOf<T>, float64_t>)>
VECOPS_VFUNC Mask<T> isnan(V v) {
  return _mm256_cmp_pd_mask(v.v, v.v, _CMP_UNORD_Q);
}

template <TLV_DECL_VEC(V), typename T = Vec2Tag<V>, TL_IF(T::Bytes == 64), TL_IF(is_any<TypeOf<T>, float32_t>)>
VECOPS_VFUNC Mask<T> isnan(V v) {
  return _mm512_cmp_ps_mask(v.v, v.v, _CMP_UNORD_Q);
}
template <TLV_DECL_VEC(V), typename T = Vec2Tag<V>, TL_IF(T::Bytes == 64), TL_IF(is_any<TypeOf<T>, float64_t>)>
VECOPS_VFUNC Mask<T> isnan(V v) {
  return _mm512_cmp_pd_mask(v.v, v.v, _CMP_UNORD_Q);
}
#ifdef HAS_AVX512FP16
template <TLV_DECL_VEC(V), typename T = Vec2Tag<V>, TL_IF(T::Bytes <= 16), TL_IF(is_any<TypeOf<T>, float16_t>)>
VECOPS_VFUNC Mask<T> isnan(V v) {
  return _mm_cmp_ph_mask(_mm_castsi128_ph(v.v), _mm_castsi128_ph(v.v), _CMP_UNORD_Q);
}
#if VEC_WIDTH >= 256
template <TLV_DECL_VEC(V), typename T = Vec2Tag<V>, TL_IF(T::Bytes == 32), TL_IF(is_any<TypeOf<T>, float16_t>)>
VECOPS_VFUNC Mask<T> isnan(V v) {
  return _mm256_cmp_ph_mask(_mm256_castsi256_ph(v.v), _mm256_castsi256_ph(v.v), _CMP_UNORD_Q);
}
#endif
#if VEC_WIDTH >= 512
template <TLV_DECL_VEC(V), typename T = Vec2Tag<V>, TL_IF(T::Bytes == 64), TL_IF(is_any<TypeOf<T>, float16_t>)>
VECOPS_VFUNC Mask<T> isnan(V v) {
  return _mm512_cmp_ph_mask(_mm512_castsi512_ph(v.v), _mm512_castsi512_ph(v.v), _CMP_UNORD_Q);
}
#endif
#endif // HAS_AVX512FP16
template <TLV_DECL_VEC(V), typename T = Vec2Tag<V>, TL_IF(T::Bytes <= 16), TL_IF(is_any<TypeOf<T>, bfloat16_t>)>
VECOPS_VFUNC Mask<T> isnan(V v) {
  __m128 a_lo, a_hi;
  half_cvt::cvt_bf16_to_two_fp32(v.v, a_lo, a_hi);
  auto lo = _mm_cmp_ps_mask(a_lo, a_lo, _CMP_UNORD_Q);
  auto hi = _mm_cmp_ps_mask(a_hi, a_hi, _CMP_UNORD_Q);
  return lo | (hi << 4);
}
#if VEC_WIDTH >= 256
template <TLV_DECL_VEC(V), typename T = Vec2Tag<V>, TL_IF(T::Bytes == 32), TL_IF(is_any<TypeOf<T>, bfloat16_t>)>
VECOPS_VFUNC Mask<T> isnan(V v) {
  __m256 a_lo, a_hi;
  half_cvt::cvt_bf16_to_two_fp32(v.v, a_lo, a_hi);
  auto lo = _mm256_cmp_ps_mask(a_lo, a_lo, _CMP_UNORD_Q);
  auto hi = _mm256_cmp_ps_mask(a_hi, a_hi, _CMP_UNORD_Q);
  return lo | (uint16_t(hi) << 8);
}
#endif
#if VEC_WIDTH >= 512
template <TLV_DECL_VEC(V), typename T = Vec2Tag<V>, TL_IF(T::Bytes == 64), TL_IF(is_any<TypeOf<T>, bfloat16_t>)>
VECOPS_VFUNC Mask<T> isnan(V v) {
  __m512 a_lo, a_hi;
  half_cvt::cvt_bf16_to_two_fp32(v.v, a_lo, a_hi);
  auto lo = _mm512_cmp_ps_mask(a_lo, a_lo, _CMP_UNORD_Q);
  auto hi = _mm512_cmp_ps_mask(a_hi, a_hi, _CMP_UNORD_Q);
  return lo | (uint32_t(hi) << 16);
}
#endif
#ifndef HAS_AVX512FP16
template <TLV_DECL_VEC(V), typename T = Vec2Tag<V>, TL_IF(T::Bytes <= 16), TL_IF(is_any<TypeOf<T>, float16_t>)>
VECOPS_VFUNC Mask<T> isnan(V v) {
  __m128 a_lo, a_hi;
  half_cvt::cvt_fp16_to_two_fp32(v.v, a_lo, a_hi);
  auto lo = _mm_cmp_ps_mask(a_lo, a_lo, _CMP_UNORD_Q);
  auto hi = _mm_cmp_ps_mask(a_hi, a_hi, _CMP_UNORD_Q);
  return lo | (hi << 4);
}
#if VEC_WIDTH >= 256
template <TLV_DECL_VEC(V), typename T = Vec2Tag<V>, TL_IF(T::Bytes == 32), TL_IF(is_any<TypeOf<T>, float16_t>)>
VECOPS_VFUNC Mask<T> isnan(V v) {
  __m256 a_lo, a_hi;
  half_cvt::cvt_fp16_to_two_fp32(v.v, a_lo, a_hi);
  auto lo = _mm256_cmp_ps_mask(a_lo, a_lo, _CMP_UNORD_Q);
  auto hi = _mm256_cmp_ps_mask(a_hi, a_hi, _CMP_UNORD_Q);
  return lo | (uint16_t(hi) << 8);
}
#endif
#if VEC_WIDTH >= 512
template <TLV_DECL_VEC(V), typename T = Vec2Tag<V>, TL_IF(T::Bytes == 64), TL_IF(is_any<TypeOf<T>, float16_t>)>
VECOPS_VFUNC Mask<T> isnan(V v) {
  __m512 a_lo, a_hi;
  half_cvt::cvt_fp16_to_two_fp32(v.v, a_lo, a_hi);
  auto lo = _mm512_cmp_ps_mask(a_lo, a_lo, _CMP_UNORD_Q);
  auto hi = _mm512_cmp_ps_mask(a_hi, a_hi, _CMP_UNORD_Q);
  return lo | (uint32_t(hi) << 16);
}
#endif
#endif // !HAS_AVX512FP16
#else // HAS_AVX512DQ
template <TLV_DECL_VEC(V), typename T = Vec2Tag<V>, TL_IF(T::Bytes <= 16), TL_IF(is_any<TypeOf<T>, float32_t>)>
VECOPS_VFUNC Mask<T> isnan(V v) {
  return _mm_castps_si128(_mm_cmpunord_ps(v.v, v.v));
}
template <TLV_DECL_VEC(V), typename T = Vec2Tag<V>, TL_IF(T::Bytes <= 16), TL_IF(is_any<TypeOf<T>, float64_t>)>
VECOPS_VFUNC Mask<T> isnan(V v) {
  return _mm_castpd_si128(_mm_cmpunord_pd(v.v, v.v));
}

#if VEC_WIDTH >= 256
template <TLV_DECL_VEC(V), typename T = Vec2Tag<V>, TL_IF(T::Bytes == 32), TL_IF(is_any<TypeOf<T>, float32_t>)>
VECOPS_VFUNC Mask<T> isnan(V v) {
  return _mm256_castps_si256(_mm256_cmp_ps(v.v, v.v, _CMP_UNORD_Q));
}
template <TLV_DECL_VEC(V), typename T = Vec2Tag<V>, TL_IF(T::Bytes == 32), TL_IF(is_any<TypeOf<T>, float64_t>)>
VECOPS_VFUNC Mask<T> isnan(V v) {
  return _mm256_castpd_si256(_mm256_cmp_pd(v.v, v.v, _CMP_UNORD_Q));
}
#endif // VEC_WIDTH >= 256

template <TLV_DECL_VEC(V), typename T = Vec2Tag<V>, TL_IF(T::Bytes <= 16), TL_IF(is_any<TypeOf<T>, float16_t>)>
VECOPS_VFUNC Mask<T> isnan(V v) {
  __m128 a_lo, a_hi;
  half_cvt::cvt_fp16_to_two_fp32(v.v, a_lo, a_hi);
  return _mm_packs_epi32(
      _mm_castps_si128(_mm_cmpunord_ps(a_lo, a_lo)),
      _mm_castps_si128(_mm_cmpunord_ps(a_hi, a_hi)));
}
template <TLV_DECL_VEC(V), typename T = Vec2Tag<V>, TL_IF(T::Bytes <= 16), TL_IF(is_any<TypeOf<T>, bfloat16_t>)>
VECOPS_VFUNC Mask<T> isnan(V v) {
  __m128 a_lo, a_hi;
  half_cvt::cvt_bf16_to_two_fp32(v.v, a_lo, a_hi);
  return _mm_packs_epi32(
      _mm_castps_si128(_mm_cmpunord_ps(a_lo, a_lo)),
      _mm_castps_si128(_mm_cmpunord_ps(a_hi, a_hi)));
}
#if VEC_WIDTH >= 256
template <TLV_DECL_VEC(V), typename T = Vec2Tag<V>, TL_IF(T::Bytes == 32), TL_IF(is_any<TypeOf<T>, float16_t>)>
VECOPS_VFUNC Mask<T> isnan(V v) {
  __m256 a_lo, a_hi;
  half_cvt::cvt_fp16_to_two_fp32(v.v, a_lo, a_hi);
  __m256i packed = _mm256_packs_epi32(
      _mm256_castps_si256(_mm256_cmp_ps(a_lo, a_lo, _CMP_UNORD_Q)),
      _mm256_castps_si256(_mm256_cmp_ps(a_hi, a_hi, _CMP_UNORD_Q)));
  return _mm256_permute4x64_epi64(packed, _MM_SHUFFLE(3, 1, 2, 0));
}
template <TLV_DECL_VEC(V), typename T = Vec2Tag<V>, TL_IF(T::Bytes == 32), TL_IF(is_any<TypeOf<T>, bfloat16_t>)>
VECOPS_VFUNC Mask<T> isnan(V v) {
  __m256 a_lo, a_hi;
  half_cvt::cvt_bf16_to_two_fp32(v.v, a_lo, a_hi);
  __m256i packed = _mm256_packs_epi32(
      _mm256_castps_si256(_mm256_cmp_ps(a_lo, a_lo, _CMP_UNORD_Q)),
      _mm256_castps_si256(_mm256_cmp_ps(a_hi, a_hi, _CMP_UNORD_Q)));
  return _mm256_permute4x64_epi64(packed, _MM_SHUFFLE(3, 1, 2, 0));
}
#endif // VEC_WIDTH >= 256
#endif // HAS_AVX512DQ

#ifdef HAS_AVX512DQ
template <TLV_DECL_VEC(V), typename T = Vec2Tag<V>, TL_IF(T::Bytes <= 16), TL_IF(is_any<TypeOf<T>, float32_t>)>
VECOPS_VFUNC Mask<T> isnan(V v, Mask<T> m) {
  return _mm_mask_cmp_ps_mask(m.v, v.v, v.v, _CMP_UNORD_Q);
}
template <TLV_DECL_VEC(V), typename T = Vec2Tag<V>, TL_IF(T::Bytes <= 16), TL_IF(is_any<TypeOf<T>, float64_t>)>
VECOPS_VFUNC Mask<T> isnan(V v, Mask<T> m) {
  return _mm_mask_cmp_pd_mask(m.v, v.v, v.v, _CMP_UNORD_Q);
}

template <TLV_DECL_VEC(V), typename T = Vec2Tag<V>, TL_IF(T::Bytes == 32), TL_IF(is_any<TypeOf<T>, float32_t>)>
VECOPS_VFUNC Mask<T> isnan(V v, Mask<T> m) {
  return _mm256_mask_cmp_ps_mask(m.v, v.v, v.v, _CMP_UNORD_Q);
}
template <TLV_DECL_VEC(V), typename T = Vec2Tag<V>, TL_IF(T::Bytes == 32), TL_IF(is_any<TypeOf<T>, float64_t>)>
VECOPS_VFUNC Mask<T> isnan(V v, Mask<T> m) {
  return _mm256_mask_cmp_pd_mask(m.v, v.v, v.v, _CMP_UNORD_Q);
}

template <TLV_DECL_VEC(V), typename T = Vec2Tag<V>, TL_IF(T::Bytes == 64), TL_IF(is_any<TypeOf<T>, float32_t>)>
VECOPS_VFUNC Mask<T> isnan(V v, Mask<T> m) {
  return _mm512_mask_cmp_ps_mask(m.v, v.v, v.v, _CMP_UNORD_Q);
}
template <TLV_DECL_VEC(V), typename T = Vec2Tag<V>, TL_IF(T::Bytes == 64), TL_IF(is_any<TypeOf<T>, float64_t>)>
VECOPS_VFUNC Mask<T> isnan(V v, Mask<T> m) {
  return _mm512_mask_cmp_pd_mask(m.v, v.v, v.v, _CMP_UNORD_Q);
}
#ifdef HAS_AVX512FP16
template <TLV_DECL_VEC(V), typename T = Vec2Tag<V>, TL_IF(T::Bytes <= 16), TL_IF(is_any<TypeOf<T>, float16_t>)>
VECOPS_VFUNC Mask<T> isnan(V v, Mask<T> m) {
  return _mm_mask_cmp_ph_mask(m.v, _mm_castsi128_ph(v.v), _mm_castsi128_ph(v.v), _CMP_UNORD_Q);
}
#if VEC_WIDTH >= 256
template <TLV_DECL_VEC(V), typename T = Vec2Tag<V>, TL_IF(T::Bytes == 32), TL_IF(is_any<TypeOf<T>, float16_t>)>
VECOPS_VFUNC Mask<T> isnan(V v, Mask<T> m) {
  return _mm256_mask_cmp_ph_mask(m.v, _mm256_castsi256_ph(v.v), _mm256_castsi256_ph(v.v), _CMP_UNORD_Q);
}
#endif
#if VEC_WIDTH >= 512
template <TLV_DECL_VEC(V), typename T = Vec2Tag<V>, TL_IF(T::Bytes == 64), TL_IF(is_any<TypeOf<T>, float16_t>)>
VECOPS_VFUNC Mask<T> isnan(V v, Mask<T> m) {
  return _mm512_mask_cmp_ph_mask(m.v, _mm512_castsi512_ph(v.v), _mm512_castsi512_ph(v.v), _CMP_UNORD_Q);
}
#endif
#endif // HAS_AVX512FP16
template <TLV_DECL_VEC(V), typename T = Vec2Tag<V>, TL_IF(is_any<TypeOf<T>, float16_t, bfloat16_t>)>
VECOPS_VFUNC Mask<T> isnan(V v, Mask<T> m) {
  return word::bit_and(m, word::isnan(v));
}
#else // HAS_AVX512DQ
template <TLV_DECL_VEC(V), typename T = Vec2Tag<V>>
VECOPS_VFUNC Mask<T> isnan(V v, Mask<T> m) {
  return word::bit_and(m, word::isnan(v));
}
#endif // HAS_AVX512DQ


/* ************************************************************************** */
//                                Is Infinity                                 //
/* ************************************************************************** */
template <TLV_DECL_VEC(V), typename T = Vec2Tag<V>, TL_IF(is_float<TypeOf<T>> || is_small_float<TypeOf<T>>)>
VECOPS_VFUNC Mask<T> isposinf(V v) {
  return word::cmpeq(v, word::fill(T(), TypeOf<T>(INFINITY)));
}

template <TLV_DECL_VEC(V), typename T = Vec2Tag<V>, TL_IF(is_float<TypeOf<T>> || is_small_float<TypeOf<T>>)>
VECOPS_VFUNC Mask<T> isposinf(V v, Mask<T> m) {
  return word::cmpeq(v, word::fill(T(), TypeOf<T>(INFINITY)), m);
}

template <TLV_DECL_VEC(V), typename T = Vec2Tag<V>, TL_IF(is_float<TypeOf<T>> || is_small_float<TypeOf<T>>)>
VECOPS_VFUNC Mask<T> isneginf(V v) {
  return word::cmpeq(v, word::fill(T(), TypeOf<T>(-INFINITY)));
}

template <TLV_DECL_VEC(V), typename T = Vec2Tag<V>, TL_IF(is_float<TypeOf<T>> || is_small_float<TypeOf<T>>)>
VECOPS_VFUNC Mask<T> isneginf(V v, Mask<T> m) {
  return word::cmpeq(v, word::fill(T(), TypeOf<T>(-INFINITY)), m);
}

template <TLV_DECL_VEC(V), typename T = Vec2Tag<V>, TL_IF(is_float<TypeOf<T>> || is_small_float<TypeOf<T>>)>
VECOPS_VFUNC Mask<T> isinf(V v) {
  return word::isposinf(word::abs(v));
}

template <TLV_DECL_VEC(V), typename T = Vec2Tag<V>, TL_IF(is_float<TypeOf<T>> || is_small_float<TypeOf<T>>)>
VECOPS_VFUNC Mask<T> isinf(V v, Mask<T> m) {
  return word::isposinf(word::abs(v), m);
}

} // namespace word
} // namespace vecops::vec::CPU_CAPABILITY
//@formatter:on

#endif //VECOPS_X86_ARITHMETIC_H
