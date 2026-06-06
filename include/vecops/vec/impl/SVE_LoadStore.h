//
// SVE_LoadStore.h — SVE real implementations for load/store/gather/scatter
//

#ifndef VECOPS_SVE_LOADSTORE_H
#define VECOPS_SVE_LOADSTORE_H

#include <arm_sve.h>
#include <cstring>

#include "CoreTypes.h"
#include "../VecBase.h"
#include "SVE_Basic.h"

//@formatter:off
namespace vecops::vec::CPU_CAPABILITY {
namespace word {

/* ================================================================ */
//                      loadu  (unaligned consecutive load)         //
/* ================================================================ */
template <TLV_DECL_TAG(T)>
VECOPS_VFUNC Vec<T> loadu(T t, const TypeOf<T>* p) {
  using E = TypeOf<T>;
  auto pg = word::make_mask(t);
  if constexpr (std::is_same_v<E, float32_t>)      return svld1_f32(pg, p);
  else if constexpr (std::is_same_v<E, float64_t>) return svld1_f64(pg, p);
  else if constexpr (std::is_same_v<E, float16_t>) return svld1_f16(pg, (const __fp16 *)p);
#if defined(__ARM_FEATURE_BF16)
  else if constexpr (std::is_same_v<E, bfloat16_t>) return svld1_bf16(pg, (const __bf16 *)p);
#endif
  else if constexpr (std::is_same_v<E, uint8_t>)   return svld1_u8(pg, p);
  else if constexpr (std::is_same_v<E, int8_t>)    return svld1_s8(pg, p);
  else if constexpr (std::is_same_v<E, uint16_t>)  return svld1_u16(pg, p);
  else if constexpr (std::is_same_v<E, int16_t>)   return svld1_s16(pg, p);
  else if constexpr (std::is_same_v<E, uint32_t>)  return svld1_u32(pg, p);
  else if constexpr (std::is_same_v<E, int32_t>)   return svld1_s32(pg, p);
  else if constexpr (std::is_same_v<E, uint64_t>)  return svld1_u64(pg, p);
  else return svld1_s64(pg, p);
}

/* ================================================================ */
//                      load   (aligned consecutive load)           //
/* ================================================================ */
template <TLV_DECL_TAG(T)>
VECOPS_VFUNC Vec<T> load(T t, const TypeOf<T>* p) {
  return word::loadu(t, p);
}

/* ================================================================ */
//                  masked loadu  (mask + default)                   //
/* ================================================================ */
template <TLV_DECL_TAG(T)>
VECOPS_VFUNC Vec<T> loadu(T t, const TypeOf<T>* p, Mask<T> m, Vec<T> default_v) {
  using E = TypeOf<T>;
  Vec<T> loaded;
  if constexpr (std::is_same_v<E, float32_t>)      loaded = svld1_f32(m, p);
  else if constexpr (std::is_same_v<E, float64_t>) loaded = svld1_f64(m, p);
  else if constexpr (std::is_same_v<E, float16_t>) loaded = svld1_f16(m, (const __fp16 *)p);
#if defined(__ARM_FEATURE_BF16)
  else if constexpr (std::is_same_v<E, bfloat16_t>) loaded = svld1_bf16(m, (const __bf16 *)p);
#endif
  else if constexpr (std::is_same_v<E, uint8_t>)   loaded = svld1_u8(m, p);
  else if constexpr (std::is_same_v<E, int8_t>)    loaded = svld1_s8(m, p);
  else if constexpr (std::is_same_v<E, uint16_t>)  loaded = svld1_u16(m, p);
  else if constexpr (std::is_same_v<E, int16_t>)   loaded = svld1_s16(m, p);
  else if constexpr (std::is_same_v<E, uint32_t>)  loaded = svld1_u32(m, p);
  else if constexpr (std::is_same_v<E, int32_t>)   loaded = svld1_s32(m, p);
  else if constexpr (std::is_same_v<E, uint64_t>)  loaded = svld1_u64(m, p);
  else loaded = svld1_s64(m, p);
  return word::blend(default_v, m, loaded);
}

/* ================================================================ */
//                  masked load  (mask + default)                    //
/* ================================================================ */
template <TLV_DECL_TAG(T)>
VECOPS_VFUNC Vec<T> load(T t, const TypeOf<T>* p, Mask<T> m, Vec<T> default_v) {
  return word::loadu(t, p, m, default_v);
}

/* ================================================================ */
//              partial count loadu  (n elements)                    //
/* ================================================================ */
template <TLV_DECL_TAG(T)>
VECOPS_VFUNC Vec<T> loadu(T t, const TypeOf<T>* p, nint_t n, Vec<T> default_v) {
  VECOPS_ASSERT(0 <= n && n <= word_size(t), "");
  auto m = word::mwhilelt(t, 0, n);
  return word::loadu(t, p, m, default_v);
}

/* ================================================================ */
//              partial count load  (n elements)                     //
/* ================================================================ */
template <TLV_DECL_TAG(T)>
VECOPS_VFUNC Vec<T> load(T t, const TypeOf<T>* p, nint_t n, Vec<T> default_v) {
  VECOPS_ASSERT(0 <= n && n <= word_size(t), "");
  auto m = word::mwhilelt(t, 0, n);
  return word::load(t, p, m, default_v);
}

/* ================================================================ */
//                      storeu  (unaligned store)                    //
/* ================================================================ */
template <TLV_DECL_TAG(T)>
VECOPS_VFUNC void storeu(T t, TypeOf<T>* p, Vec<T> v) {
  using E = TypeOf<T>;
  auto pg = word::make_mask(t);
  if constexpr (std::is_same_v<E, float32_t>)      svst1_f32(pg, p, v);
  else if constexpr (std::is_same_v<E, float64_t>) svst1_f64(pg, p, v);
  else if constexpr (std::is_same_v<E, float16_t>) svst1_f16(pg, (__fp16 *)p, v);
#if defined(__ARM_FEATURE_BF16)
  else if constexpr (std::is_same_v<E, bfloat16_t>) svst1_bf16(pg, (__bf16 *)p, v);
#endif
  else if constexpr (std::is_same_v<E, uint8_t>)   svst1_u8(pg, p, v);
  else if constexpr (std::is_same_v<E, int8_t>)    svst1_s8(pg, p, v);
  else if constexpr (std::is_same_v<E, uint16_t>)  svst1_u16(pg, p, v);
  else if constexpr (std::is_same_v<E, int16_t>)   svst1_s16(pg, p, v);
  else if constexpr (std::is_same_v<E, uint32_t>)  svst1_u32(pg, p, v);
  else if constexpr (std::is_same_v<E, int32_t>)   svst1_s32(pg, p, v);
  else if constexpr (std::is_same_v<E, uint64_t>)  svst1_u64(pg, p, v);
  else svst1_s64(pg, p, v);
}

/* ================================================================ */
//                      store   (aligned store)                      //
/* ================================================================ */
template <TLV_DECL_TAG(T)>
VECOPS_VFUNC void store(T t, TypeOf<T>* p, Vec<T> v) {
  word::storeu(t, p, v);
}

/* ================================================================ */
//                    masked storeu  (selective store)               //
/* ================================================================ */
template <TLV_DECL_TAG(T)>
VECOPS_VFUNC void storeu(T t, TypeOf<T>* p, Mask<T> m, Vec<T> v) {
  using E = TypeOf<T>;
  if constexpr (std::is_same_v<E, float32_t>)      svst1_f32(m, p, v);
  else if constexpr (std::is_same_v<E, float64_t>) svst1_f64(m, p, v);
  else if constexpr (std::is_same_v<E, float16_t>) svst1_f16(m, (__fp16 *)p, v);
#if defined(__ARM_FEATURE_BF16)
  else if constexpr (std::is_same_v<E, bfloat16_t>) svst1_bf16(m, (__bf16 *)p, v);
#endif
  else if constexpr (std::is_same_v<E, uint8_t>)   svst1_u8(m, p, v);
  else if constexpr (std::is_same_v<E, int8_t>)    svst1_s8(m, p, v);
  else if constexpr (std::is_same_v<E, uint16_t>)  svst1_u16(m, p, v);
  else if constexpr (std::is_same_v<E, int16_t>)   svst1_s16(m, p, v);
  else if constexpr (std::is_same_v<E, uint32_t>)  svst1_u32(m, p, v);
  else if constexpr (std::is_same_v<E, int32_t>)   svst1_s32(m, p, v);
  else if constexpr (std::is_same_v<E, uint64_t>)  svst1_u64(m, p, v);
  else svst1_s64(m, p, v);
}

/* ================================================================ */
//                    masked store  (selective store)                //
/* ================================================================ */
template <TLV_DECL_TAG(T)>
VECOPS_VFUNC void store(T t, TypeOf<T>* p, Mask<T> m, Vec<T> v) {
  word::storeu(t, p, m, v);
}

/* ================================================================ */
//              partial count storeu  (n elements)                   //
/* ================================================================ */
template <TLV_DECL_TAG(T)>
VECOPS_VFUNC void storeu(T t, TypeOf<T>* p, nint_t n, Vec<T> v) {
  VECOPS_ASSERT(0 <= n && n <= word_size(t), "");
  auto m = word::mwhilelt(t, 0, n);
  word::storeu(t, p, m, v);
}

/* ================================================================ */
//              partial count store  (n elements)                    //
/* ================================================================ */
template <TLV_DECL_TAG(T)>
VECOPS_VFUNC void store(T t, TypeOf<T>* p, nint_t n, Vec<T> v) {
  VECOPS_ASSERT(0 <= n && n <= word_size(t), "");
  auto m = word::mwhilelt(t, 0, n);
  word::store(t, p, m, v);
}

/* ================================================================ */
//                            Gather                                 //
/* ================================================================ */
/* 32-bit elements: float32, int32, uint32 */
template <TLV_DECL_TAG(T), TL_IF(is_any<TypeOf<T>, float32_t>)>
VECOPS_VFUNC Vec<T> gather(T t, const TypeOf<T>* p, Vec<Rebind<Index<TypeOf<T>>, T>> i) {
  auto pg = details::ptrue<float32_t>();
  auto byte_offsets = svlsl_n_s32_x(pg, i, 2);
  return svld1_gather_s32offset_f32(pg, p, byte_offsets);
}

template <TLV_DECL_TAG(T), TL_IF(is_any<TypeOf<T>, int32_t>)>
VECOPS_VFUNC Vec<T> gather(T t, const TypeOf<T>* p, Vec<Rebind<Index<TypeOf<T>>, T>> i) {
  auto pg = details::ptrue<int32_t>();
  auto byte_offsets = svlsl_n_s32_x(pg, i, 2);
  return svld1_gather_s32offset_s32(pg, p, byte_offsets);
}

template <TLV_DECL_TAG(T), TL_IF(is_any<TypeOf<T>, uint32_t>)>
VECOPS_VFUNC Vec<T> gather(T t, const TypeOf<T>* p, Vec<Rebind<Index<TypeOf<T>>, T>> i) {
  auto pg32 = details::ptrue<int32_t>();
  auto byte_offsets = svlsl_n_s32_x(pg32, i, 2);
  auto pg = details::ptrue<uint32_t>();
  auto ui = svreinterpret_u32_s32(byte_offsets);
  return svld1_gather_u32offset_u32(pg, p, ui);
}

/* 64-bit elements: float64, int64, uint64 */
template <TLV_DECL_TAG(T), TL_IF(is_any<TypeOf<T>, float64_t>)>
VECOPS_VFUNC Vec<T> gather(T t, const TypeOf<T>* p, Vec<Rebind<Index<TypeOf<T>>, T>> i) {
  auto pg = details::ptrue<float64_t>();
  auto byte_offsets = svlsl_n_s64_x(pg, i, 3);
  return svld1_gather_s64offset_f64(pg, p, byte_offsets);
}

template <TLV_DECL_TAG(T), TL_IF(is_any<TypeOf<T>, int64_t>)>
VECOPS_VFUNC Vec<T> gather(T t, const TypeOf<T>* p, Vec<Rebind<Index<TypeOf<T>>, T>> i) {
  auto pg = details::ptrue<int64_t>();
  auto byte_offsets = svlsl_n_s64_x(pg, i, 3);
  return svld1_gather_s64offset_s64(pg, p, byte_offsets);
}

template <TLV_DECL_TAG(T), TL_IF(is_any<TypeOf<T>, uint64_t>)>
VECOPS_VFUNC Vec<T> gather(T t, const TypeOf<T>* p, Vec<Rebind<Index<TypeOf<T>>, T>> i) {
  auto pg64 = details::ptrue<int64_t>();
  auto byte_offsets = svlsl_n_s64_x(pg64, i, 3);
  auto pg = details::ptrue<uint64_t>();
  auto ui = svreinterpret_u64_s64(byte_offsets);
  return svld1_gather_u64offset_u64(pg, p, ui);
}

/* masked gather */
template <TLV_DECL_TAG(T)>
VECOPS_VFUNC Vec<T> gather(T t, const TypeOf<T>* p, Vec<Rebind<Index<TypeOf<T>>, T>> i, Mask<T> m, Vec<T> default_v) {
  return word::blend(default_v, m, word::gather(t, p, i));
}

/* partial count gather */
template <TLV_DECL_TAG(T)>
VECOPS_VFUNC Vec<T> gather(T t, TypeOf<T>* p, Vec<Rebind<Index<TypeOf<T>>, T>> i, nint_t n, Vec<T> default_v) {
  VECOPS_ASSERT(0 <= n && n <= word_size(t), "");
  auto m = word::mwhilelt(t, 0, n);
  return word::gather(t, p, i, m, default_v);
}

/* ================================================================ */
//                            Scatter                                //
/* ================================================================ */
/* 32-bit elements: float32, int32, uint32 */
template <TLV_DECL_TAG(T), TL_IF(is_any<TypeOf<T>, float32_t>)>
VECOPS_VFUNC void scatter(T t, TypeOf<T>* p, Vec<Rebind<Index<TypeOf<T>>, T>> i, Vec<T> v) {
  auto pg = details::ptrue<float32_t>();
  auto byte_offsets = svlsl_n_s32_x(pg, i, 2);
  svst1_scatter_s32offset_f32(pg, p, byte_offsets, v);
}

template <TLV_DECL_TAG(T), TL_IF(is_any<TypeOf<T>, int32_t>)>
VECOPS_VFUNC void scatter(T t, TypeOf<T>* p, Vec<Rebind<Index<TypeOf<T>>, T>> i, Vec<T> v) {
  auto pg = details::ptrue<int32_t>();
  auto byte_offsets = svlsl_n_s32_x(pg, i, 2);
  svst1_scatter_s32offset_s32(pg, p, byte_offsets, v);
}

template <TLV_DECL_TAG(T), TL_IF(is_any<TypeOf<T>, uint32_t>)>
VECOPS_VFUNC void scatter(T t, TypeOf<T>* p, Vec<Rebind<Index<TypeOf<T>>, T>> i, Vec<T> v) {
  auto pg32 = details::ptrue<int32_t>();
  auto byte_offsets = svlsl_n_s32_x(pg32, i, 2);
  auto pg = details::ptrue<uint32_t>();
  auto ui = svreinterpret_u32_s32(byte_offsets);
  svst1_scatter_u32offset_u32(pg, p, ui, v);
}

/* 64-bit elements: float64, int64, uint64 */
template <TLV_DECL_TAG(T), TL_IF(is_any<TypeOf<T>, float64_t>)>
VECOPS_VFUNC void scatter(T t, TypeOf<T>* p, Vec<Rebind<Index<TypeOf<T>>, T>> i, Vec<T> v) {
  auto pg = details::ptrue<float64_t>();
  auto byte_offsets = svlsl_n_s64_x(pg, i, 3);
  svst1_scatter_s64offset_f64(pg, p, byte_offsets, v);
}

template <TLV_DECL_TAG(T), TL_IF(is_any<TypeOf<T>, int64_t>)>
VECOPS_VFUNC void scatter(T t, TypeOf<T>* p, Vec<Rebind<Index<TypeOf<T>>, T>> i, Vec<T> v) {
  auto pg = details::ptrue<int64_t>();
  auto byte_offsets = svlsl_n_s64_x(pg, i, 3);
  svst1_scatter_s64offset_s64(pg, p, byte_offsets, v);
}

template <TLV_DECL_TAG(T), TL_IF(is_any<TypeOf<T>, uint64_t>)>
VECOPS_VFUNC void scatter(T t, TypeOf<T>* p, Vec<Rebind<Index<TypeOf<T>>, T>> i, Vec<T> v) {
  auto pg64 = details::ptrue<int64_t>();
  auto byte_offsets = svlsl_n_s64_x(pg64, i, 3);
  auto pg = details::ptrue<uint64_t>();
  auto ui = svreinterpret_u64_s64(byte_offsets);
  svst1_scatter_u64offset_u64(pg, p, ui, v);
}

/* masked scatter */
template <TLV_DECL_TAG(T)>
VECOPS_VFUNC void scatter(T t, TypeOf<T>* p, Vec<Rebind<Index<TypeOf<T>>, T>> i, Mask<T> m, Vec<T> v) {
  using E = TypeOf<T>;
  if constexpr (std::is_same_v<E, float32_t>) {
    auto pg = details::ptrue<float32_t>();
    auto byte_offsets = svlsl_n_s32_x(pg, i, 2);
    svst1_scatter_s32offset_f32(m, p, byte_offsets, v);
  } else if constexpr (std::is_same_v<E, float64_t>) {
    auto pg = details::ptrue<float64_t>();
    auto byte_offsets = svlsl_n_s64_x(pg, i, 3);
    svst1_scatter_s64offset_f64(m, p, byte_offsets, v);
  } else if constexpr (std::is_same_v<E, int32_t>) {
    auto pg = details::ptrue<int32_t>();
    auto byte_offsets = svlsl_n_s32_x(pg, i, 2);
    svst1_scatter_s32offset_s32(m, p, byte_offsets, v);
  } else if constexpr (std::is_same_v<E, uint32_t>) {
    auto pg32 = details::ptrue<int32_t>();
    auto byte_offsets = svlsl_n_s32_x(pg32, i, 2);
    auto ui = svreinterpret_u32_s32(byte_offsets);
    svst1_scatter_u32offset_u32(m, p, ui, v);
  } else if constexpr (std::is_same_v<E, int64_t>) {
    auto pg = details::ptrue<int64_t>();
    auto byte_offsets = svlsl_n_s64_x(pg, i, 3);
    svst1_scatter_s64offset_s64(m, p, byte_offsets, v);
  } else if constexpr (std::is_same_v<E, uint64_t>) {
    auto pg64 = details::ptrue<int64_t>();
    auto byte_offsets = svlsl_n_s64_x(pg64, i, 3);
    auto ui = svreinterpret_u64_s64(byte_offsets);
    svst1_scatter_u64offset_u64(m, p, ui, v);
  }
}

/* partial count scatter */
template <TLV_DECL_TAG(T)>
VECOPS_VFUNC void scatter(T t, TypeOf<T>* p, Vec<Rebind<Index<TypeOf<T>>, T>> i, nint_t n, Vec<T> v) {
  VECOPS_ASSERT(0 <= n && n <= word_size(t), "");
  auto m = word::mwhilelt(t, 0, n);
  word::scatter(t, p, i, m, v);
}

}  // namespace word
}  // namespace vecops::vec::CPU_CAPABILITY
//@formatter:on

#endif // VECOPS_SVE_LOADSTORE_H
