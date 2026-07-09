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
#else
  else if constexpr (std::is_same_v<E, bfloat16_t>) loaded = svreinterpret_bf16_u16(svld1_u16(m, (const uint16_t *)p));
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

}  // namespace word
}  // namespace vecops::vec::CPU_CAPABILITY
//@formatter:on

#endif // VECOPS_SVE_LOADSTORE_H
