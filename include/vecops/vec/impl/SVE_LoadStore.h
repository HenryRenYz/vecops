//
// SVE_LoadStore.h — SVE real implementations for load/store/gather/scatter
//

#ifndef VECOPS_SVE_LOADSTORE_H
#define VECOPS_SVE_LOADSTORE_H

#include <arm_sve.h>
#include <cstring>
#include <limits>

#include "CoreTypes.h"
#include "../VecBase.h"
#include "SVE_Basic.h"
#include "SVE_Conversions.h"
#include "SVE_MaskConversions.h"

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
#else
  else if constexpr (std::is_same_v<E, bfloat16_t>) return svreinterpret_bf16_u16(svld1_u16(pg, (const uint16_t *)p));
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
//                  masked loadu  (zeroing mask)                     //
/* ================================================================ */
template <TLV_DECL_TAG(T)>
VECOPS_VFUNC Vec<T> loadu(T t, const TypeOf<T>* p, Mask<T> m) {
  using E = TypeOf<T>;
  if constexpr (std::is_same_v<E, float32_t>)      return svld1_f32(m, p);
  else if constexpr (std::is_same_v<E, float64_t>) return svld1_f64(m, p);
  else if constexpr (std::is_same_v<E, float16_t>) return svld1_f16(m, (const __fp16 *)p);
#if defined(__ARM_FEATURE_BF16)
  else if constexpr (std::is_same_v<E, bfloat16_t>) return svld1_bf16(m, (const __bf16 *)p);
#else
  else if constexpr (std::is_same_v<E, bfloat16_t>) return svreinterpret_bf16_u16(svld1_u16(m, (const uint16_t *)p));
#endif
  else if constexpr (std::is_same_v<E, uint8_t>)   return svld1_u8(m, p);
  else if constexpr (std::is_same_v<E, int8_t>)    return svld1_s8(m, p);
  else if constexpr (std::is_same_v<E, uint16_t>)  return svld1_u16(m, p);
  else if constexpr (std::is_same_v<E, int16_t>)   return svld1_s16(m, p);
  else if constexpr (std::is_same_v<E, uint32_t>)  return svld1_u32(m, p);
  else if constexpr (std::is_same_v<E, int32_t>)   return svld1_s32(m, p);
  else if constexpr (std::is_same_v<E, uint64_t>)  return svld1_u64(m, p);
  else return svld1_s64(m, p);
}

template <TLV_DECL_TAG(T)>
VECOPS_VFUNC Vec<T> loadu(T t, const TypeOf<T>* p, Mask<T> m, Vec<T> default_v) {
  return word::blend(default_v, m, word::loadu(t, p, m));
}

/* ================================================================ */
//                  masked load  (mask + default)                    //
/* ================================================================ */
template <TLV_DECL_TAG(T)>
VECOPS_VFUNC Vec<T> load(T t, const TypeOf<T>* p, Mask<T> m, Vec<T> default_v) {
  return word::loadu(t, p, m, default_v);
}

template <TLV_DECL_TAG(T)>
VECOPS_VFUNC Vec<T> load(T t, const TypeOf<T>* p, Mask<T> m) {
  return word::loadu(t, p, m);
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

template <TLV_DECL_TAG(T)>
VECOPS_VFUNC Vec<T> loadu(T t, const TypeOf<T>* p, nint_t n) {
  VECOPS_ASSERT(0 <= n && n <= word_size(t), "");
  return word::loadu(t, p, word::mwhilelt(t, 0, n));
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


template <TLV_DECL_TAG(T)>
VECOPS_VFUNC Vec<T> load(T t, const TypeOf<T>* p, nint_t n) {
  VECOPS_ASSERT(0 <= n && n <= word_size(t), "");
  return word::load(t, p, word::mwhilelt(t, 0, n));
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
#else
  else if constexpr (std::is_same_v<E, bfloat16_t>) svst1_u16(pg, (uint16_t *)p, svreinterpret_u16_bf16(v));
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
#else
  else if constexpr (std::is_same_v<E, bfloat16_t>) svst1_u16(m, (uint16_t *)p, svreinterpret_u16_bf16(v));
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

namespace gather_scatter_native {

template <int Shift>
VECOPS_VFUNC svint32_t byte_offsets(svbool_t pg, svint32_t i) {
  if constexpr (Shift == 0) return i;
  else return svlsl_n_s32_x(pg, i, Shift);
}

template <int Shift>
VECOPS_VFUNC svint64_t byte_offsets(svbool_t pg, svint64_t i) {
  if constexpr (Shift == 0) return i;
  else return svlsl_n_s64_x(pg, i, Shift);
}

template <TLV_DECL_TAG(T), TLV_DECL_TAG(Ti)>
VECOPS_VFUNC Vec<T> gather_byte_leaf(T t, const TypeOf<T>* p, Ti ti, Vec<Ti> i, Mask<T> m) {
  using E = TypeOf<T>;
  static_assert(std::is_same_v<TypeOf<Ti>, int32_t>);
  static_assert(num_words(Ti{}) == 1);
  auto pg = word::promote(ti, t, m);
  if constexpr (std::is_same_v<E, int8_t>) {
    return word::demote(t, svld1sb_gather_s32offset_s32(pg, p, i));
  } else {
    return word::demote(t, svld1ub_gather_s32offset_u32(pg, p, i));
  }
}

template <TLV_DECL_TAG(T), TLV_DECL_TAG(Ti)>
VECOPS_VFUNC Vec<T> gather_half_leaf(T t, const TypeOf<T>* p, Ti ti, Vec<Ti> i, Mask<T> m) {
  using E = TypeOf<T>;
  static_assert(std::is_same_v<TypeOf<Ti>, int32_t>);
  static_assert(num_words(Ti{}) == 1);
  auto pg = word::promote(ti, t, m);
  auto off = byte_offsets<1>(pg, i);
  if constexpr (std::is_same_v<E, int16_t>) {
    return word::demote(t, svld1sh_gather_s32offset_s32(pg, p, off));
  } else if constexpr (std::is_same_v<E, uint16_t>) {
    return word::demote(t, svld1uh_gather_s32offset_u32(pg, p, off));
  } else {
    using Tu16 = Rebind<uint16_t, T>;
    constexpr Tu16 tu16;
    auto u16 = word::demote(tu16, svld1uh_gather_s32offset_u32(pg, (const uint16_t*)p, off));
    return word::bitcast(t, u16);
  }
}

template <TLV_DECL_TAG(T), TLV_DECL_TAG(Ti)>
VECOPS_VFUNC Vec<T> gather_subword(T t, const TypeOf<T>* p, Ti ti, Vec<Ti> i, Mask<T> m) {
  using E = TypeOf<T>;
  if constexpr (num_words(Ti{}) > 1) {
    Half<T> th;
    Half<Ti> tih;
    auto lo = gather_subword(th, p, tih, word::lower(ti, i), word::lower(t, m));
    auto hi = gather_subword(th, p, tih, word::upper(ti, i), word::upper(t, m));
    return word::concat(t, lo, hi);
  } else if constexpr (sizeof(E) == 1) {
    return gather_byte_leaf(t, p, ti, i, m);
  } else {
    return gather_half_leaf(t, p, ti, i, m);
  }
}

template <TLV_DECL_TAG(T), TLV_DECL_TAG(Ti)>
VECOPS_VFUNC void scatter_byte_leaf(T t, TypeOf<T>* p, Ti ti, Vec<Ti> i, Mask<T> m, Vec<T> v) {
  using E = TypeOf<T>;
  static_assert(std::is_same_v<TypeOf<Ti>, int32_t>);
  static_assert(num_words(Ti{}) == 1);
  auto pg = word::promote(ti, t, m);
  if constexpr (std::is_same_v<E, int8_t>) {
    svst1b_scatter_s32offset_s32(pg, p, i, word::promote(ti, v));
  } else {
    using Tu32 = Rebind<uint32_t, T>;
    constexpr Tu32 tu32;
    svst1b_scatter_s32offset_u32(pg, p, i, word::promote(tu32, v));
  }
}

template <TLV_DECL_TAG(T), TLV_DECL_TAG(Ti)>
VECOPS_VFUNC void scatter_half_leaf(T t, TypeOf<T>* p, Ti ti, Vec<Ti> i, Mask<T> m, Vec<T> v) {
  using E = TypeOf<T>;
  static_assert(std::is_same_v<TypeOf<Ti>, int32_t>);
  static_assert(num_words(Ti{}) == 1);
  auto pg = word::promote(ti, t, m);
  auto off = byte_offsets<1>(pg, i);
  if constexpr (std::is_same_v<E, int16_t>) {
    svst1h_scatter_s32offset_s32(pg, p, off, word::promote(ti, v));
  } else if constexpr (std::is_same_v<E, uint16_t>) {
    using Tu32 = Rebind<uint32_t, T>;
    constexpr Tu32 tu32;
    svst1h_scatter_s32offset_u32(pg, p, off, word::promote(tu32, v));
  } else {
    using Tu16 = Rebind<uint16_t, T>;
    using Tu32 = Rebind<uint32_t, T>;
    constexpr Tu16 tu16;
    constexpr Tu32 tu32;
    auto u16 = word::bitcast(tu16, v);
    svst1h_scatter_s32offset_u32(pg, (uint16_t*)p, off, word::promote(tu32, u16));
  }
}

template <TLV_DECL_TAG(T), TLV_DECL_TAG(Ti)>
VECOPS_VFUNC void scatter_subword(T t, TypeOf<T>* p, Ti ti, Vec<Ti> i, Mask<T> m, Vec<T> v) {
  using E = TypeOf<T>;
  if constexpr (num_words(Ti{}) > 1) {
    Half<T> th;
    Half<Ti> tih;
    scatter_subword(th, p, tih, word::lower(ti, i), word::lower(t, m), word::lower(t, v));
    scatter_subword(th, p, tih, word::upper(ti, i), word::upper(t, m), word::upper(t, v));
  } else if constexpr (sizeof(E) == 1) {
    scatter_byte_leaf(t, p, ti, i, m, v);
  } else {
    scatter_half_leaf(t, p, ti, i, m, v);
  }
}

template <TLV_DECL_TAG(T)>
VECOPS_VFUNC Vec<T> gather_raw(T t, const TypeOf<T>* p, Vec<Rebind<GatherScatterIndex<TypeOf<T>>, T>> i, Mask<T> m) {
  using E = TypeOf<T>;
  using Ti = Rebind<GatherScatterIndex<E>, T>;
  constexpr Ti ti;
  if constexpr (sizeof(E) == 1) {
    return gather_subword(t, p, ti, i, m);
  } else if constexpr (sizeof(E) == 2) {
    return gather_subword(t, p, ti, i, m);
  } else if constexpr (std::is_same_v<E, float32_t>) {
    auto byte_offsets = gather_scatter_native::byte_offsets<2>(m, i);
    return svld1_gather_s32offset_f32(m, p, byte_offsets);
  } else if constexpr (std::is_same_v<E, int32_t>) {
    auto byte_offsets = gather_scatter_native::byte_offsets<2>(m, i);
    return svld1_gather_s32offset_s32(m, p, byte_offsets);
  } else if constexpr (std::is_same_v<E, uint32_t>) {
    auto byte_offsets = gather_scatter_native::byte_offsets<2>(m, i);
    return svld1_gather_s32offset_u32(m, p, byte_offsets);
  } else if constexpr (std::is_same_v<E, float64_t>) {
    auto byte_offsets = gather_scatter_native::byte_offsets<3>(m, i);
    return svld1_gather_s64offset_f64(m, p, byte_offsets);
  } else if constexpr (std::is_same_v<E, int64_t>) {
    auto byte_offsets = gather_scatter_native::byte_offsets<3>(m, i);
    return svld1_gather_s64offset_s64(m, p, byte_offsets);
  } else {
    auto byte_offsets = gather_scatter_native::byte_offsets<3>(m, i);
    return svld1_gather_s64offset_u64(m, p, byte_offsets);
  }
}

}  // namespace gather_scatter_native

/* ================================================================ */
//                            Gather                                 //
/* ================================================================ */
template <TLV_DECL_TAG(T)>
VECOPS_VFUNC Vec<T> gather(T t, const TypeOf<T>* p, Vec<Rebind<GatherScatterIndex<TypeOf<T>>, T>> i, Mask<T> m, Vec<T> default_v);

template <TLV_DECL_TAG(T)>
VECOPS_VFUNC Vec<T> gather(T t, const TypeOf<T>* p, Vec<Rebind<GatherScatterIndex<TypeOf<T>>, T>> i) {
  return gather_scatter_native::gather_raw(t, p, i, word::make_mask(t));
}

/* masked gather */
template <TLV_DECL_TAG(T)>
VECOPS_VFUNC Vec<T> gather(T t, const TypeOf<T>* p, Vec<Rebind<GatherScatterIndex<TypeOf<T>>, T>> i, Mask<T> m, Vec<T> default_v) {
  auto loaded = gather_scatter_native::gather_raw(t, p, i, m);
  return word::blend(default_v, m, loaded);
}

/* partial count gather */
template <TLV_DECL_TAG(T)>
VECOPS_VFUNC Vec<T> gather(T t, const TypeOf<T>* p, Vec<Rebind<GatherScatterIndex<TypeOf<T>>, T>> i, nint_t n, Vec<T> default_v) {
  VECOPS_ASSERT(0 <= n && n <= size(t), "");
  auto m = word::mwhilelt(t, 0, n);
  return word::gather(t, p, i, m, default_v);
}

/* ================================================================ */
//                            Scatter                                //
/* ================================================================ */
template <TLV_DECL_TAG(T)>
VECOPS_VFUNC void scatter(T t, TypeOf<T>* p, Vec<Rebind<GatherScatterIndex<TypeOf<T>>, T>> i, Mask<T> m, Vec<T> v);

template <TLV_DECL_TAG(T)>
VECOPS_VFUNC void scatter(T t, TypeOf<T>* p, Vec<Rebind<GatherScatterIndex<TypeOf<T>>, T>> i, Vec<T> v) {
  word::scatter(t, p, i, word::make_mask(t), v);
}

/* masked scatter */
template <TLV_DECL_TAG(T)>
VECOPS_VFUNC void scatter(T t, TypeOf<T>* p, Vec<Rebind<GatherScatterIndex<TypeOf<T>>, T>> i, Mask<T> m, Vec<T> v) {
  using E = TypeOf<T>;
  using Ti = Rebind<GatherScatterIndex<E>, T>;
  constexpr Ti ti;
  if constexpr (sizeof(E) == 1) {
    gather_scatter_native::scatter_subword(t, p, ti, i, m, v);
  } else if constexpr (sizeof(E) == 2) {
    gather_scatter_native::scatter_subword(t, p, ti, i, m, v);
  } else if constexpr (std::is_same_v<E, float32_t>) {
    auto byte_offsets = gather_scatter_native::byte_offsets<2>(m, i);
    svst1_scatter_s32offset_f32(m, p, byte_offsets, v);
  } else if constexpr (std::is_same_v<E, int32_t>) {
    auto byte_offsets = gather_scatter_native::byte_offsets<2>(m, i);
    svst1_scatter_s32offset_s32(m, p, byte_offsets, v);
  } else if constexpr (std::is_same_v<E, uint32_t>) {
    auto byte_offsets = gather_scatter_native::byte_offsets<2>(m, i);
    svst1_scatter_s32offset_u32(m, p, byte_offsets, v);
  } else if constexpr (std::is_same_v<E, float64_t>) {
    auto byte_offsets = gather_scatter_native::byte_offsets<3>(m, i);
    svst1_scatter_s64offset_f64(m, p, byte_offsets, v);
  } else if constexpr (std::is_same_v<E, int64_t>) {
    auto byte_offsets = gather_scatter_native::byte_offsets<3>(m, i);
    svst1_scatter_s64offset_s64(m, p, byte_offsets, v);
  } else {
    auto byte_offsets = gather_scatter_native::byte_offsets<3>(m, i);
    svst1_scatter_s64offset_u64(m, p, byte_offsets, v);
  }
}

/* partial count scatter */
template <TLV_DECL_TAG(T)>
VECOPS_VFUNC void scatter(T t, TypeOf<T>* p, Vec<Rebind<GatherScatterIndex<TypeOf<T>>, T>> i, nint_t n, Vec<T> v) {
  VECOPS_ASSERT(0 <= n && n <= size(t), "");
  auto m = word::mwhilelt(t, 0, n);
  word::scatter(t, p, i, m, v);
}

/* ================================================================ */
//                 Fused conversion load & store                    //
/* ================================================================ */

namespace conversion_load_store_detail {

template <TLV_DECL_TAG(To), TLV_DECL_TAG(Ti)>
VECOPS_VFUNC Mask<To> xconvert_mask(To to, Ti ti, Mask<Ti> m) {
  using ToMask = Rebind<SignedIntegerOfSize<sizeof(TypeOf<To>)>, To>;
  using TiMask = Rebind<SignedIntegerOfSize<sizeof(TypeOf<Ti>)>, Ti>;
  constexpr ToMask to_mask{};
  constexpr TiMask ti_mask{};
  if constexpr (sizeof(TypeOf<Ti>) < sizeof(TypeOf<To>)) {
    return word::promote(to_mask, ti_mask, m);
  } else if constexpr (sizeof(TypeOf<Ti>) > sizeof(TypeOf<To>)) {
    return word::demote(to_mask, ti_mask, m);
  } else {
    return word::convert(to_mask, ti_mask, m);
  }
}

template <TLV_DECL_TAG(T)>
VECOPS_VFUNC Vec<T> zeros(T t) {
  if constexpr (is_word_vec(t)) return word::zeros(t);
  return vecops::vec::details::vmap(
      t, [](auto tt) VECOPS_INLINE_LAMBDA { return word::zeros(tt); });
}

template <TLV_DECL_TAG(T)>
VECOPS_VFUNC Vec<T> loadu(T t, const TypeOf<T>* p) {
  if constexpr (is_word_vec(t)) return word::loadu(t, p);
  return vecops::vec::details::vmap(
      t,
      [](auto tt, const TypeOf<T>* pp) VECOPS_INLINE_LAMBDA {
        return word::loadu(tt, pp);
      },
      vecops::vec::details::StepPointer(t, p));
}

template <TLV_DECL_TAG(T)>
VECOPS_VFUNC Vec<T> loadu(
    T t, const TypeOf<T>* p, Mask<T> m, Vec<T> default_v) {
  if constexpr (is_word_vec(t)) return word::loadu(t, p, m, default_v);
  return vecops::vec::details::vmap(
      t,
      [](auto tt, const TypeOf<T>* pp, auto mm,
         auto dd) VECOPS_INLINE_LAMBDA {
        return word::loadu(tt, pp, mm, dd);
      },
      vecops::vec::details::StepPointer(t, p),
      vecops::vec::details::ShardMask(t, m),
      vecops::vec::details::ShardVec(t, default_v));
}

template <TLV_DECL_TAG(T)>
VECOPS_VFUNC Vec<T> loadu(T t, const TypeOf<T>* p, Mask<T> m) {
  if constexpr (is_word_vec(t)) return word::loadu(t, p, m);
  return vecops::vec::details::vmap(
      t,
      [](auto tt, const TypeOf<T>* pp, auto mm) VECOPS_INLINE_LAMBDA {
        return word::loadu(tt, pp, mm);
      },
      vecops::vec::details::StepPointer(t, p),
      vecops::vec::details::ShardMask(t, m));
}

template <TLV_DECL_TAG(T)>
VECOPS_VFUNC void storeu(T t, TypeOf<T>* p, Vec<T> v) {
  if constexpr (is_word_vec(t)) {
    word::storeu(t, p, v);
    return;
  }
  vecops::vec::details::vmap(
      t,
      [](auto tt, TypeOf<T>* pp, auto vv) VECOPS_INLINE_LAMBDA {
        word::storeu(tt, pp, vv);
      },
      vecops::vec::details::StepPointer(t, p),
      vecops::vec::details::ShardVec(t, v));
}

template <TLV_DECL_TAG(T)>
VECOPS_VFUNC void storeu(T t, TypeOf<T>* p, Mask<T> m, Vec<T> v) {
  if constexpr (is_word_vec(t)) {
    word::storeu(t, p, m, v);
    return;
  }
  vecops::vec::details::vmap(
      t,
      [](auto tt, TypeOf<T>* pp, auto mm,
         auto vv) VECOPS_INLINE_LAMBDA {
        word::storeu(tt, pp, mm, vv);
      },
      vecops::vec::details::StepPointer(t, p),
      vecops::vec::details::ShardMask(t, m),
      vecops::vec::details::ShardVec(t, v));
}

} // namespace conversion_load_store_detail

template <TLV_DECL_TAG(T), typename Ei>
  requires (is_element_type<Ei> && sizeof(Ei) < sizeof(TypeOf<T>))
VECOPS_VFUNC Vec<T> promote_loadu(T t, const Ei* p) {
  if constexpr (!is_word_vec(t)) {
    constexpr Half<T> th{};
    return word::concat(
        t, word::promote_loadu(th, p),
        word::promote_loadu(th, p + size(th)));
  } else {

  using Eo = TypeOf<T>;
  auto pg = word::make_mask(t);

  if constexpr (std::is_same_v<Ei, bfloat16_t> && std::is_same_v<Eo, float32_t>) {
    auto bits = svld1uh_u32(pg, reinterpret_cast<const uint16_t*>(p));
    return svreinterpret_f32_u32(svlsl_n_u32_x(pg, bits, 16));
  } else if constexpr (IsIntV<Ei> && IsIntV<Eo>) {
    if constexpr (std::is_signed_v<Ei>) {
      if constexpr (sizeof(Ei) == 1 && sizeof(Eo) == 2) {
        auto v = svld1sb_s16(pg, reinterpret_cast<const int8_t*>(p));
        if constexpr (std::is_signed_v<Eo>) return v;
        else return svreinterpret_u16_s16(v);
      } else if constexpr (sizeof(Ei) == 1 && sizeof(Eo) == 4) {
        auto v = svld1sb_s32(pg, reinterpret_cast<const int8_t*>(p));
        if constexpr (std::is_signed_v<Eo>) return v;
        else return svreinterpret_u32_s32(v);
      } else if constexpr (sizeof(Ei) == 1 && sizeof(Eo) == 8) {
        auto v = svld1sb_s64(pg, reinterpret_cast<const int8_t*>(p));
        if constexpr (std::is_signed_v<Eo>) return v;
        else return svreinterpret_u64_s64(v);
      } else if constexpr (sizeof(Ei) == 2 && sizeof(Eo) == 4) {
        auto v = svld1sh_s32(pg, reinterpret_cast<const int16_t*>(p));
        if constexpr (std::is_signed_v<Eo>) return v;
        else return svreinterpret_u32_s32(v);
      } else if constexpr (sizeof(Ei) == 2 && sizeof(Eo) == 8) {
        auto v = svld1sh_s64(pg, reinterpret_cast<const int16_t*>(p));
        if constexpr (std::is_signed_v<Eo>) return v;
        else return svreinterpret_u64_s64(v);
      } else {
        auto v = svld1sw_s64(pg, reinterpret_cast<const int32_t*>(p));
        if constexpr (std::is_signed_v<Eo>) return v;
        else return svreinterpret_u64_s64(v);
      }
    } else {
      if constexpr (sizeof(Ei) == 1 && sizeof(Eo) == 2) {
        auto v = svld1ub_u16(pg, reinterpret_cast<const uint8_t*>(p));
        if constexpr (std::is_unsigned_v<Eo>) return v;
        else return svreinterpret_s16_u16(v);
      } else if constexpr (sizeof(Ei) == 1 && sizeof(Eo) == 4) {
        auto v = svld1ub_u32(pg, reinterpret_cast<const uint8_t*>(p));
        if constexpr (std::is_unsigned_v<Eo>) return v;
        else return svreinterpret_s32_u32(v);
      } else if constexpr (sizeof(Ei) == 1 && sizeof(Eo) == 8) {
        auto v = svld1ub_u64(pg, reinterpret_cast<const uint8_t*>(p));
        if constexpr (std::is_unsigned_v<Eo>) return v;
        else return svreinterpret_s64_u64(v);
      } else if constexpr (sizeof(Ei) == 2 && sizeof(Eo) == 4) {
        auto v = svld1uh_u32(pg, reinterpret_cast<const uint16_t*>(p));
        if constexpr (std::is_unsigned_v<Eo>) return v;
        else return svreinterpret_s32_u32(v);
      } else if constexpr (sizeof(Ei) == 2 && sizeof(Eo) == 8) {
        auto v = svld1uh_u64(pg, reinterpret_cast<const uint16_t*>(p));
        if constexpr (std::is_unsigned_v<Eo>) return v;
        else return svreinterpret_s64_u64(v);
      } else {
        auto v = svld1uw_u64(pg, reinterpret_cast<const uint32_t*>(p));
        if constexpr (std::is_unsigned_v<Eo>) return v;
        else return svreinterpret_s64_u64(v);
      }
    }
  } else {
    Rebind<Ei, T> ti;
    return word::promote(t, word::loadu(ti, p));
  }
  }
}

template <TLV_DECL_TAG(T), typename Ei>
  requires (is_element_type<Ei> && sizeof(Ei) < sizeof(TypeOf<T>))
VECOPS_VFUNC Vec<T> promote_loadu(
    T t, const Ei* p, Mask<T> m) {
  if constexpr (!is_word_vec(t)) {
    constexpr Half<T> th{};
    return word::concat(
        t, word::promote_loadu(th, p, word::lower(t, m)),
        word::promote_loadu(
            th, p + size(th), word::upper(t, m)));
  } else {

  using Eo = TypeOf<T>;
  if constexpr (std::is_same_v<Ei, bfloat16_t> && std::is_same_v<Eo, float32_t>) {
    auto bits = svld1uh_u32(m, reinterpret_cast<const uint16_t*>(p));
    // svld1uh zeroes every inactive lane. Shift under ptrue so those zeroes
    // stay defined without requiring an additional predicated merge.
    return svreinterpret_f32_u32(
        svlsl_n_u32_x(svptrue_b32(), bits, 16));
  } else if constexpr (IsIntV<Ei> && IsIntV<Eo>) {
    Vec<T> loaded;
    if constexpr (std::is_signed_v<Ei>) {
      if constexpr (sizeof(Ei) == 1 && sizeof(Eo) == 2) {
        auto v = svld1sb_s16(m, reinterpret_cast<const int8_t*>(p));
        if constexpr (std::is_signed_v<Eo>) loaded = v;
        else loaded = svreinterpret_u16_s16(v);
      } else if constexpr (sizeof(Ei) == 1 && sizeof(Eo) == 4) {
        auto v = svld1sb_s32(m, reinterpret_cast<const int8_t*>(p));
        if constexpr (std::is_signed_v<Eo>) loaded = v;
        else loaded = svreinterpret_u32_s32(v);
      } else if constexpr (sizeof(Ei) == 1 && sizeof(Eo) == 8) {
        auto v = svld1sb_s64(m, reinterpret_cast<const int8_t*>(p));
        if constexpr (std::is_signed_v<Eo>) loaded = v;
        else loaded = svreinterpret_u64_s64(v);
      } else if constexpr (sizeof(Ei) == 2 && sizeof(Eo) == 4) {
        auto v = svld1sh_s32(m, reinterpret_cast<const int16_t*>(p));
        if constexpr (std::is_signed_v<Eo>) loaded = v;
        else loaded = svreinterpret_u32_s32(v);
      } else if constexpr (sizeof(Ei) == 2 && sizeof(Eo) == 8) {
        auto v = svld1sh_s64(m, reinterpret_cast<const int16_t*>(p));
        if constexpr (std::is_signed_v<Eo>) loaded = v;
        else loaded = svreinterpret_u64_s64(v);
      } else {
        auto v = svld1sw_s64(m, reinterpret_cast<const int32_t*>(p));
        if constexpr (std::is_signed_v<Eo>) loaded = v;
        else loaded = svreinterpret_u64_s64(v);
      }
    } else {
      if constexpr (sizeof(Ei) == 1 && sizeof(Eo) == 2) {
        auto v = svld1ub_u16(m, reinterpret_cast<const uint8_t*>(p));
        if constexpr (std::is_unsigned_v<Eo>) loaded = v;
        else loaded = svreinterpret_s16_u16(v);
      } else if constexpr (sizeof(Ei) == 1 && sizeof(Eo) == 4) {
        auto v = svld1ub_u32(m, reinterpret_cast<const uint8_t*>(p));
        if constexpr (std::is_unsigned_v<Eo>) loaded = v;
        else loaded = svreinterpret_s32_u32(v);
      } else if constexpr (sizeof(Ei) == 1 && sizeof(Eo) == 8) {
        auto v = svld1ub_u64(m, reinterpret_cast<const uint8_t*>(p));
        if constexpr (std::is_unsigned_v<Eo>) loaded = v;
        else loaded = svreinterpret_s64_u64(v);
      } else if constexpr (sizeof(Ei) == 2 && sizeof(Eo) == 4) {
        auto v = svld1uh_u32(m, reinterpret_cast<const uint16_t*>(p));
        if constexpr (std::is_unsigned_v<Eo>) loaded = v;
        else loaded = svreinterpret_s32_u32(v);
      } else if constexpr (sizeof(Ei) == 2 && sizeof(Eo) == 8) {
        auto v = svld1uh_u64(m, reinterpret_cast<const uint16_t*>(p));
        if constexpr (std::is_unsigned_v<Eo>) loaded = v;
        else loaded = svreinterpret_s64_u64(v);
      } else {
        auto v = svld1uw_u64(m, reinterpret_cast<const uint32_t*>(p));
        if constexpr (std::is_unsigned_v<Eo>) loaded = v;
        else loaded = svreinterpret_s64_u64(v);
      }
    }
    return loaded;
  } else {
    Rebind<Ei, T> ti;
    auto mi = conversion_load_store_detail::xconvert_mask(ti, t, m);
    auto loaded = conversion_load_store_detail::loadu(ti, p, mi);
    return word::promote(t, loaded);
  }
  }
}

template <TLV_DECL_TAG(T), typename Ei>
  requires (is_element_type<Ei> && sizeof(Ei) < sizeof(TypeOf<T>))
VECOPS_VFUNC Vec<T> promote_loadu(
    T t, const Ei* p, Mask<T> m, Vec<T> default_v) {
  if constexpr (!is_word_vec(t)) {
    constexpr Half<T> th{};
    return word::concat(
        t,
        word::promote_loadu(
            th, p, word::lower(t, m), word::lower(t, default_v)),
        word::promote_loadu(
            th, p + size(th), word::upper(t, m),
            word::upper(t, default_v)));
  } else {
    return word::blend(default_v, m, word::promote_loadu(t, p, m));
  }
}

template <TLV_DECL_TAG(T), typename Ei>
  requires (is_element_type<Ei> && sizeof(Ei) > sizeof(TypeOf<T>))
VECOPS_VFUNC Vec<T> demote_loadu(T t, const Ei* p) {
  Rebind<Ei, T> ti;
  return word::demote(
      t, conversion_load_store_detail::loadu(ti, p));
}

template <TLV_DECL_TAG(T), typename Ei>
  requires (is_element_type<Ei> && sizeof(Ei) > sizeof(TypeOf<T>))
VECOPS_VFUNC Vec<T> demote_loadu(
    T t, const Ei* p, Mask<T> m, Vec<T> default_v) {
  Rebind<Ei, T> ti;
  auto mi = conversion_load_store_detail::xconvert_mask(ti, t, m);
  auto loaded = conversion_load_store_detail::loadu(
      ti, p, mi, conversion_load_store_detail::zeros(ti));
  return word::blend(default_v, m, word::demote(t, loaded));
}

template <TLV_DECL_TAG(T), typename Ei>
  requires (is_element_type<Ei> && sizeof(Ei) > sizeof(TypeOf<T>))
VECOPS_VFUNC Vec<T> demote_loadu(T t, const Ei* p, Mask<T> m) {
  Rebind<Ei, T> ti;
  auto mi = conversion_load_store_detail::xconvert_mask(ti, t, m);
  return word::demote(
      t, conversion_load_store_detail::loadu(ti, p, mi));
}

template <TLV_DECL_TAG(T), typename Ei>
  requires (is_element_type<Ei> && sizeof(Ei) == sizeof(TypeOf<T>))
VECOPS_VFUNC Vec<T> convert_loadu(T t, const Ei* p) {
  Rebind<Ei, T> ti;
  return word::convert(
      t, conversion_load_store_detail::loadu(ti, p));
}

template <TLV_DECL_TAG(T), typename Ei>
  requires (is_element_type<Ei> && sizeof(Ei) == sizeof(TypeOf<T>))
VECOPS_VFUNC Vec<T> convert_loadu(
    T t, const Ei* p, Mask<T> m, Vec<T> default_v) {
  Rebind<Ei, T> ti;
  auto mi = conversion_load_store_detail::xconvert_mask(ti, t, m);
  auto loaded = conversion_load_store_detail::loadu(
      ti, p, mi, conversion_load_store_detail::zeros(ti));
  return word::blend(default_v, m, word::convert(t, loaded));
}

template <TLV_DECL_TAG(T), typename Ei>
  requires (is_element_type<Ei> && sizeof(Ei) == sizeof(TypeOf<T>))
VECOPS_VFUNC Vec<T> convert_loadu(T t, const Ei* p, Mask<T> m) {
  Rebind<Ei, T> ti;
  auto mi = conversion_load_store_detail::xconvert_mask(ti, t, m);
  return word::convert(
      t, conversion_load_store_detail::loadu(ti, p, mi));
}

template <TLV_DECL_TAG(Ti), typename Eo>
  requires (is_element_type<Eo> && sizeof(TypeOf<Ti>) < sizeof(Eo))
VECOPS_VFUNC void promote_storeu(Ti ti, Eo* p, Vec<Ti> v) {
  Rebind<Eo, Ti> to;
  conversion_load_store_detail::storeu(to, p, word::promote(to, v));
}

template <TLV_DECL_TAG(Ti), typename Eo>
  requires (is_element_type<Eo> && sizeof(TypeOf<Ti>) < sizeof(Eo))
VECOPS_VFUNC void promote_storeu(
    Ti ti, Eo* p, Mask<Ti> m, Vec<Ti> v) {
  Rebind<Eo, Ti> to;
  auto mo = conversion_load_store_detail::xconvert_mask(to, ti, m);
  conversion_load_store_detail::storeu(
      to, p, mo, word::promote(to, v));
}

template <TLV_DECL_TAG(Ti), typename Eo>
  requires (IsIntV<TypeOf<Ti>> && IsIntV<Eo> &&
            sizeof(TypeOf<Ti>) > sizeof(Eo))
VECOPS_VFUNC void demote_integer_storeu_sve(
    Ti ti, Eo* p, Mask<Ti> m, Vec<Ti> v) {
  using Ei = TypeOf<Ti>;

  if constexpr (std::is_signed_v<Ei>) {
    constexpr Ei lo = std::is_signed_v<Eo>
        ? static_cast<Ei>(std::numeric_limits<Eo>::lowest()) : Ei{0};
    constexpr Ei hi = static_cast<Ei>(std::numeric_limits<Eo>::max());
    if constexpr (sizeof(Ei) == 2) {
      auto clamped = svmin_n_s16_x(m, svmax_n_s16_x(m, v, lo), hi);
      svst1b_s16(m, reinterpret_cast<int8_t*>(p), clamped);
    } else if constexpr (sizeof(Ei) == 4) {
      auto clamped = svmin_n_s32_x(m, svmax_n_s32_x(m, v, lo), hi);
      if constexpr (sizeof(Eo) == 1) {
        svst1b_s32(m, reinterpret_cast<int8_t*>(p), clamped);
      } else {
        svst1h_s32(m, reinterpret_cast<int16_t*>(p), clamped);
      }
    } else {
      auto clamped = svmin_n_s64_x(m, svmax_n_s64_x(m, v, lo), hi);
      if constexpr (sizeof(Eo) == 1) {
        svst1b_s64(m, reinterpret_cast<int8_t*>(p), clamped);
      } else if constexpr (sizeof(Eo) == 2) {
        svst1h_s64(m, reinterpret_cast<int16_t*>(p), clamped);
      } else {
        svst1w_s64(m, reinterpret_cast<int32_t*>(p), clamped);
      }
    }
  } else {
    constexpr Ei hi = static_cast<Ei>(std::numeric_limits<Eo>::max());
    if constexpr (sizeof(Ei) == 2) {
      auto clamped = svmin_n_u16_x(m, v, hi);
      svst1b_u16(m, reinterpret_cast<uint8_t*>(p), clamped);
    } else if constexpr (sizeof(Ei) == 4) {
      auto clamped = svmin_n_u32_x(m, v, hi);
      if constexpr (sizeof(Eo) == 1) {
        svst1b_u32(m, reinterpret_cast<uint8_t*>(p), clamped);
      } else {
        svst1h_u32(m, reinterpret_cast<uint16_t*>(p), clamped);
      }
    } else {
      auto clamped = svmin_n_u64_x(m, v, hi);
      if constexpr (sizeof(Eo) == 1) {
        svst1b_u64(m, reinterpret_cast<uint8_t*>(p), clamped);
      } else if constexpr (sizeof(Eo) == 2) {
        svst1h_u64(m, reinterpret_cast<uint16_t*>(p), clamped);
      } else {
        svst1w_u64(m, reinterpret_cast<uint32_t*>(p), clamped);
      }
    }
  }
}

template <TLV_DECL_TAG(Ti), typename Eo>
  requires (is_element_type<Eo> && sizeof(TypeOf<Ti>) > sizeof(Eo))
VECOPS_VFUNC void demote_storeu(Ti ti, Eo* p, Vec<Ti> v) {
  using Ei = TypeOf<Ti>;
#if defined(__ARM_FEATURE_SVE_BF16) && !defined(VECOPS_PRESERVE_SUBNORMALS)
  if constexpr (std::is_same_v<Ei, float32_t> && std::is_same_v<Eo, bfloat16_t>) {
    if constexpr (is_word_vec(ti)) {
      auto pg = word::make_mask(ti);
      auto bf16 = svcvt_bf16_f32_x(pg, v);
      svst1h_u32(
          pg, reinterpret_cast<uint16_t*>(p),
          svreinterpret_u32_bf16(bf16));
    } else if constexpr (num_words(ti) == 2) {
      constexpr Rebind<Eo, Ti> to{};
      static_assert(is_word_vec(to));
      auto packed = details::f32x2_to_bf16(
          word::lower(ti, v), word::upper(ti, v));
      svst1_bf16(
          word::make_mask(to), reinterpret_cast<__bf16*>(p), packed);
    } else {
      constexpr Rebind<Eo, Ti> to{};
      constexpr auto to_word = word_tag(to);
      static_assert(num_words(ti) == 4 && num_words(to) == 2);
      auto packed_lo = details::f32x2_to_bf16(
          get_word<0>(ti, v), get_word<1>(ti, v));
      auto packed_hi = details::f32x2_to_bf16(
          get_word<2>(ti, v), get_word<3>(ti, v));
      const auto pg = word::make_mask(to_word);
      svst1_bf16(
          pg, reinterpret_cast<__bf16*>(p), packed_lo);
      svst1_bf16(
          pg, reinterpret_cast<__bf16*>(p + word_size(to_word)),
          packed_hi);
    }
  } else
#endif
  if constexpr (IsIntV<Ei> && IsIntV<Eo> && is_word_vec(ti)) {
    demote_integer_storeu_sve(ti, p, word::make_mask(ti), v);
  } else
  {
    Rebind<Eo, Ti> to;
    conversion_load_store_detail::storeu(to, p, word::demote(to, v));
  }
}

template <TLV_DECL_TAG(Ti), typename Eo>
  requires (is_element_type<Eo> && sizeof(TypeOf<Ti>) > sizeof(Eo))
VECOPS_VFUNC void demote_storeu(
    Ti ti, Eo* p, Mask<Ti> m, Vec<Ti> v) {
  using Ei = TypeOf<Ti>;
#if defined(__ARM_FEATURE_SVE_BF16) && !defined(VECOPS_PRESERVE_SUBNORMALS)
  if constexpr (std::is_same_v<Ei, float32_t> && std::is_same_v<Eo, bfloat16_t>) {
    if constexpr (is_word_vec(ti)) {
      auto bf16 = svcvt_bf16_f32_x(m, v);
      svst1h_u32(
          m, reinterpret_cast<uint16_t*>(p),
          svreinterpret_u32_bf16(bf16));
    } else if constexpr (num_words(ti) == 2) {
      constexpr Rebind<Eo, Ti> to{};
      static_assert(is_word_vec(to));
      auto packed = details::f32x2_to_bf16(
          word::lower(ti, v), word::upper(ti, v));
      auto mo = conversion_load_store_detail::xconvert_mask(to, ti, m);
      svst1_bf16(mo, reinterpret_cast<__bf16*>(p), packed);
    } else {
      constexpr Half<Ti> th{};
      constexpr Rebind<Eo, Half<Ti>> to_half{};
      static_assert(num_words(ti) == 4 && is_word_vec(to_half));
      auto packed_lo = details::f32x2_to_bf16(
          get_word<0>(ti, v), get_word<1>(ti, v));
      auto packed_hi = details::f32x2_to_bf16(
          get_word<2>(ti, v), get_word<3>(ti, v));
      const auto mo_lo = conversion_load_store_detail::xconvert_mask(
          to_half, th, word::lower(ti, m));
      const auto mo_hi = conversion_load_store_detail::xconvert_mask(
          to_half, th, word::upper(ti, m));
      svst1_bf16(
          mo_lo, reinterpret_cast<__bf16*>(p), packed_lo);
      svst1_bf16(
          mo_hi,
          reinterpret_cast<__bf16*>(p + word_size(to_half)),
          packed_hi);
    }
  } else
#endif
  if constexpr (IsIntV<Ei> && IsIntV<Eo> && is_word_vec(ti)) {
    demote_integer_storeu_sve(ti, p, m, v);
  } else
  {
    Rebind<Eo, Ti> to;
    auto mo = conversion_load_store_detail::xconvert_mask(to, ti, m);
    conversion_load_store_detail::storeu(
        to, p, mo, word::demote(to, v));
  }
}

template <TLV_DECL_TAG(Ti), typename Eo>
  requires (is_element_type<Eo> && sizeof(TypeOf<Ti>) == sizeof(Eo))
VECOPS_VFUNC void convert_storeu(Ti ti, Eo* p, Vec<Ti> v) {
  Rebind<Eo, Ti> to;
  conversion_load_store_detail::storeu(to, p, word::convert(to, v));
}

template <TLV_DECL_TAG(Ti), typename Eo>
  requires (is_element_type<Eo> && sizeof(TypeOf<Ti>) == sizeof(Eo))
VECOPS_VFUNC void convert_storeu(
    Ti ti, Eo* p, Mask<Ti> m, Vec<Ti> v) {
  Rebind<Eo, Ti> to;
  auto mo = conversion_load_store_detail::xconvert_mask(to, ti, m);
  conversion_load_store_detail::storeu(
      to, p, mo, word::convert(to, v));
}

}  // namespace word
}  // namespace vecops::vec::CPU_CAPABILITY
//@formatter:on

#endif // VECOPS_SVE_LOADSTORE_H
