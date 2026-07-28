//
// Created by renyz on 2026/5/19.
//

#ifndef VECOPS_SVE_TYPES_H
#define VECOPS_SVE_TYPES_H

#include "vecops/CoreTypes.h"
#include "../VecBase.h"

#ifndef ARCH_ARM_FAMILY
#error "Not ARM platform"
#endif

#if !defined(HAS_SVE)
#error "SVE not available"
#endif

#include <arm_sve.h>

// Allow svget2/svget4/svset2/svset4 on predicate tuples without requiring
// global -march=...+sve2p1. These intrinsics are pure unpack/repack of
// multi-register tuples and generate no sve2p1 hardware instructions.
#if !defined(__ARM_FEATURE_SVE2p1)
#define VECOPS_SVE2P1_TARGET __attribute__((target("sve2p1"))) inline
#else
#define VECOPS_SVE2P1_TARGET VECOPS_VFUNC
#endif

namespace vecops::vec {
namespace SVE {

template <typename T>
struct RegType {
  static_assert(sizeof(T) == -1, "Unsupported SVE element type");
};

template <> struct RegType<bfloat16_t> { using Type = svbfloat16_t; };
template <> struct RegType<float16_t>   { using Type = svfloat16_t; };
template <> struct RegType<float32_t>   { using Type = svfloat32_t; };
template <> struct RegType<float64_t>   { using Type = svfloat64_t; };
template <> struct RegType<int8_t>      { using Type = svint8_t; };
template <> struct RegType<uint8_t>     { using Type = svuint8_t; };
template <> struct RegType<int16_t>     { using Type = svint16_t; };
template <> struct RegType<uint16_t>    { using Type = svuint16_t; };
template <> struct RegType<int32_t>     { using Type = svint32_t; };
template <> struct RegType<uint32_t>    { using Type = svuint32_t; };
template <> struct RegType<int64_t>     { using Type = svint64_t; };
template <> struct RegType<uint64_t>    { using Type = svuint64_t; };

template <typename T>
VECOPS_VFUNC nint_t word_count() {
  if constexpr (sizeof(T) == 1)      return (nint_t)svcntb();
  else if constexpr (sizeof(T) == 2) return (nint_t)svcnth();
  else if constexpr (sizeof(T) == 4) return (nint_t)svcntw();
  else if constexpr (sizeof(T) == 8) return (nint_t)svcntd();
  else static_assert(sizeof(T) == -1, "Unsupported element size for SVE");
}

template <typename T>
static constexpr nint_t max_word_count = []{
#if SVE_VECTOR_BITS >= 128
  return SVE_VECTOR_BITS / 8 / sizeof(T);
#else
  return MAX_VEC_WIDTH / 8 / sizeof(T);
#endif
}();

template <typename T, int POW2>
struct MultiRegType;

template <typename T>
struct MultiRegType<T, 0> {
  using Type = typename RegType<T>::Type;
};

template <> struct MultiRegType<bfloat16_t, 1> { using Type = svbfloat16x2_t; };
template <> struct MultiRegType<float16_t, 1>   { using Type = svfloat16x2_t; };
template <> struct MultiRegType<float32_t, 1>   { using Type = svfloat32x2_t; };
template <> struct MultiRegType<float64_t, 1>   { using Type = svfloat64x2_t; };
template <> struct MultiRegType<int8_t, 1>      { using Type = svint8x2_t; };
template <> struct MultiRegType<uint8_t, 1>     { using Type = svuint8x2_t; };
template <> struct MultiRegType<int16_t, 1>     { using Type = svint16x2_t; };
template <> struct MultiRegType<uint16_t, 1>    { using Type = svuint16x2_t; };
template <> struct MultiRegType<int32_t, 1>     { using Type = svint32x2_t; };
template <> struct MultiRegType<uint32_t, 1>    { using Type = svuint32x2_t; };
template <> struct MultiRegType<int64_t, 1>     { using Type = svint64x2_t; };
template <> struct MultiRegType<uint64_t, 1>    { using Type = svuint64x2_t; };

template <> struct MultiRegType<bfloat16_t, 2> { using Type = svbfloat16x4_t; };
template <> struct MultiRegType<float16_t, 2>   { using Type = svfloat16x4_t; };
template <> struct MultiRegType<float32_t, 2>   { using Type = svfloat32x4_t; };
template <> struct MultiRegType<float64_t, 2>   { using Type = svfloat64x4_t; };
template <> struct MultiRegType<int8_t, 2>      { using Type = svint8x4_t; };
template <> struct MultiRegType<uint8_t, 2>     { using Type = svuint8x4_t; };
template <> struct MultiRegType<int16_t, 2>     { using Type = svint16x4_t; };
template <> struct MultiRegType<uint16_t, 2>    { using Type = svuint16x4_t; };
template <> struct MultiRegType<int32_t, 2>     { using Type = svint32x4_t; };
template <> struct MultiRegType<uint32_t, 2>    { using Type = svuint32x4_t; };
template <> struct MultiRegType<int64_t, 2>     { using Type = svint64x4_t; };
template <> struct MultiRegType<uint64_t, 2>    { using Type = svuint64x4_t; };

template <int POW2>
struct MultiPredType;

template <> struct MultiPredType<0> { using Type = svbool_t; };
template <> struct MultiPredType<1> { using Type = svboolx2_t; };
template <> struct MultiPredType<2> { using Type = svboolx4_t; };

} // namespace SVE

// =========================================================================
// VecDefs: POW2 = 0  --  single-word scalable SVE vector
// =========================================================================
template <typename T>
struct VecDefs<T, -1, 0, void> : public BaseVecDefs<T, -1, 0> {
  using TagType = Tag<T, -1, 0>;

  static constexpr nint_t num_words = 1;

  static nint_t word_size() { return SVE::word_count<T>(); }
  static constexpr nint_t max_word_size = SVE::max_word_count<T>;

  static nint_t size() { return word_size(); }
  static constexpr nint_t max_size = max_word_size;

  static constexpr bool is_scalable   = true;
  static constexpr bool is_default_impl = false;
  static constexpr bool is_word_vec   = true;

  using VecType  = typename SVE::RegType<T>::Type;
  using MaskType = svbool_t;
  using WordDefs = VecDefs;

  template <nint_t Index>
  VECOPS_VFUNC VECOPS_PURE
  static VecType get(VecType v) {
    static_assert(Index == 0, "Static index out of range for single-word SVE vector");
    return v;
  }

  VECOPS_VFUNC VECOPS_PURE
  static VecType get(VecType v, nint_t index) {
    VECOPS_ASSERT(index == 0, "%lld !in 0..1", index);
    return v;
  }

  template <nint_t Index>
  VECOPS_VFUNC VECOPS_PURE
  static VecType set(VecType v, VecType u) {
    static_assert(Index == 0, "Static index out of range for single-word SVE vector");
    return u;
  }

  VECOPS_VFUNC VECOPS_PURE
  static VecType set(VecType v, nint_t index, VecType u) {
    VECOPS_ASSERT(index == 0, "%lld !in 0..1", index);
    return u;
  }

  template <nint_t Index>
  VECOPS_VFUNC VECOPS_PURE
  static MaskType get_mask(MaskType m) {
    static_assert(Index == 0, "Static index out of range for single-word SVE vector");
    return m;
  }

  VECOPS_VFUNC VECOPS_PURE
  static MaskType get_mask(MaskType m, nint_t index) {
    VECOPS_ASSERT(index == 0, "%lld !in 0..1", index);
    return m;
  }

  template <nint_t Index>
  VECOPS_VFUNC VECOPS_PURE
  static MaskType set_mask(MaskType m, MaskType u) {
    static_assert(Index == 0, "Static index out of range for single-word SVE vector");
    return u;
  }

  VECOPS_VFUNC VECOPS_PURE
  static MaskType set_mask(MaskType m, nint_t index, MaskType u) {
    VECOPS_ASSERT(index == 0, "%lld !in 0..1", index);
    return u;
  }
};

// =========================================================================
// VecDefs: POW2 = 1  --  2-word scalable SVE vector  (sv*x2_t)
// =========================================================================
template <typename T>
struct VecDefs<T, -1, 1, void> : public BaseVecDefs<T, -1, 1> {
  using TagType  = Tag<T, -1, 1>;
  using WordDefs = VecDefs<T, -1, 0>;
  using WordVec  = typename WordDefs::VecType;
  using WordMask = typename WordDefs::MaskType;

  static constexpr nint_t num_words = 2;

  static nint_t word_size() { return WordDefs::word_size(); }
  static constexpr nint_t max_word_size = WordDefs::max_word_size;

  static nint_t size() { return word_size() * num_words; }
  static constexpr nint_t max_size = max_word_size * num_words;

  static constexpr bool is_scalable   = true;
  static constexpr bool is_default_impl = false;
  static constexpr bool is_word_vec   = false;

  using VecType  = typename SVE::MultiRegType<T, 1>::Type;
  using MaskType = typename SVE::MultiPredType<1>::Type;

  template <nint_t Index>
  VECOPS_VFUNC VECOPS_PURE
  static WordVec get(VecType v) {
    static_assert(0 <= Index && Index < num_words, "Static index out of range");
#if !defined(__ARM_FEATURE_BF16)
    if constexpr (std::is_same_v<T, bfloat16_t>)
      return svreinterpret_bf16_u16(svget2(svreinterpret_u16_bf16_x2(v), (uint64_t)Index));
    else
#endif
    return svget2(v, (uint64_t)Index);
  }

  VECOPS_VFUNC VECOPS_PURE
  static WordVec get(VecType v, nint_t index) {
    VECOPS_ASSERT(0 <= index && index < num_words, "%lld !in 0..%lld", index, num_words);
    if constexpr (std::is_same_v<T, bfloat16_t>) {
      const auto bits = svreinterpret_u16_bf16_x2(v);
      svuint16_t selected;
      if (index == 0) selected = svget2(bits, 0);
      else            selected = svget2(bits, 1);
      return svreinterpret_bf16_u16(selected);
    } else
    if (index == 0) return svget2(v, 0);
    else            return svget2(v, 1);
  }

  template <nint_t Index>
  VECOPS_VFUNC VECOPS_PURE
  static VecType set(VecType v, WordVec u) {
    static_assert(0 <= Index && Index < num_words, "Static index out of range");
#if !defined(__ARM_FEATURE_BF16)
    if constexpr (std::is_same_v<T, bfloat16_t>)
      return svreinterpret_bf16_u16_x2(
        svset2(svreinterpret_u16_bf16_x2(v), (uint64_t)Index, svreinterpret_u16_bf16(u)));
    else
#endif
    return svset2(v, (uint64_t)Index, u);
  }

  VECOPS_VFUNC VECOPS_PURE
  static VecType set(VecType v, nint_t index, WordVec u) {
    VECOPS_ASSERT(0 <= index && index < num_words, "%lld !in 0..%lld", index, num_words);
    if constexpr (std::is_same_v<T, bfloat16_t>) {
      auto bits = svreinterpret_u16_bf16_x2(v);
      const auto word = svreinterpret_u16_bf16(u);
      if (index == 0) bits = svset2(bits, 0, word);
      else            bits = svset2(bits, 1, word);
      return svreinterpret_bf16_u16_x2(bits);
    } else
    if (index == 0) return svset2(v, 0, u);
    else            return svset2(v, 1, u);
  }

  template <nint_t Index>
  VECOPS_SVE2P1_TARGET VECOPS_PURE
  static WordMask get_mask(MaskType m) {
    static_assert(0 <= Index && Index < num_words, "Static index out of range");
    return svget2(m, (uint64_t)Index);
  }

  VECOPS_SVE2P1_TARGET VECOPS_PURE
  static WordMask get_mask(MaskType m, nint_t index) {
    VECOPS_ASSERT(0 <= index && index < num_words, "%lld !in 0..%lld", index, num_words);
    if (index == 0) return svget2(m, 0);
    else            return svget2(m, 1);
  }

  template <nint_t Index>
  VECOPS_SVE2P1_TARGET VECOPS_PURE
  static MaskType set_mask(MaskType m, WordMask u) {
    static_assert(0 <= Index && Index < num_words, "Static index out of range");
    return svset2(m, (uint64_t)Index, u);
  }

  VECOPS_SVE2P1_TARGET VECOPS_PURE
  static MaskType set_mask(MaskType m, nint_t index, WordMask u) {
    VECOPS_ASSERT(0 <= index && index < num_words, "%lld !in 0..%lld", index, num_words);
    if (index == 0) return svset2(m, 0, u);
    else            return svset2(m, 1, u);
  }
};

// =========================================================================
// VecDefs: POW2 = 2  --  4-word scalable SVE vector  (sv*x4_t)
// =========================================================================
template <typename T>
struct VecDefs<T, -1, 2, void> : public BaseVecDefs<T, -1, 2> {
  using TagType  = Tag<T, -1, 2>;
  using WordDefs = VecDefs<T, -1, 0>;
  using WordVec  = typename WordDefs::VecType;
  using WordMask = typename WordDefs::MaskType;

  static constexpr nint_t num_words = 4;

  static nint_t word_size() { return WordDefs::word_size(); }
  static constexpr nint_t max_word_size = WordDefs::max_word_size;

  static nint_t size() { return word_size() * num_words; }
  static constexpr nint_t max_size = max_word_size * num_words;

  static constexpr bool is_scalable   = true;
  static constexpr bool is_default_impl = false;
  static constexpr bool is_word_vec   = false;

  using VecType  = typename SVE::MultiRegType<T, 2>::Type;
  using MaskType = typename SVE::MultiPredType<2>::Type;

  template <nint_t Index>
  VECOPS_VFUNC VECOPS_PURE
  static WordVec get(VecType v) {
    static_assert(0 <= Index && Index < num_words, "Static index out of range");
#if !defined(__ARM_FEATURE_BF16)
    if constexpr (std::is_same_v<T, bfloat16_t>)
      return svreinterpret_bf16_u16(svget4(svreinterpret_u16_bf16_x4(v), (uint64_t)Index));
    else
#endif
    return svget4(v, (uint64_t)Index);
  }

  VECOPS_VFUNC VECOPS_PURE
  static WordVec get(VecType v, nint_t index) {
    VECOPS_ASSERT(0 <= index && index < num_words, "%lld !in 0..%lld", index, num_words);
    if constexpr (std::is_same_v<T, bfloat16_t>) {
      const auto bits = svreinterpret_u16_bf16_x4(v);
      svuint16_t selected;
      if (index == 0)      selected = svget4(bits, 0);
      else if (index == 1) selected = svget4(bits, 1);
      else if (index == 2) selected = svget4(bits, 2);
      else                 selected = svget4(bits, 3);
      return svreinterpret_bf16_u16(selected);
    } else
    if (index == 0)      return svget4(v, 0);
    else if (index == 1) return svget4(v, 1);
    else if (index == 2) return svget4(v, 2);
    else                 return svget4(v, 3);
  }

  template <nint_t Index>
  VECOPS_VFUNC VECOPS_PURE
  static VecType set(VecType v, WordVec u) {
    static_assert(0 <= Index && Index < num_words, "Static index out of range");
#if !defined(__ARM_FEATURE_BF16)
    if constexpr (std::is_same_v<T, bfloat16_t>)
      return svreinterpret_bf16_u16_x4(
        svset4(svreinterpret_u16_bf16_x4(v), (uint64_t)Index, svreinterpret_u16_bf16(u)));
    else
#endif
    return svset4(v, (uint64_t)Index, u);
  }

  VECOPS_VFUNC VECOPS_PURE
  static VecType set(VecType v, nint_t index, WordVec u) {
    VECOPS_ASSERT(0 <= index && index < num_words, "%lld !in 0..%lld", index, num_words);
    if constexpr (std::is_same_v<T, bfloat16_t>) {
      auto bits = svreinterpret_u16_bf16_x4(v);
      const auto word = svreinterpret_u16_bf16(u);
      if (index == 0)      bits = svset4(bits, 0, word);
      else if (index == 1) bits = svset4(bits, 1, word);
      else if (index == 2) bits = svset4(bits, 2, word);
      else                 bits = svset4(bits, 3, word);
      return svreinterpret_bf16_u16_x4(bits);
    } else
    if (index == 0)      return svset4(v, 0, u);
    else if (index == 1) return svset4(v, 1, u);
    else if (index == 2) return svset4(v, 2, u);
    else                 return svset4(v, 3, u);
  }

  template <nint_t Index>
  VECOPS_SVE2P1_TARGET VECOPS_PURE
  static WordMask get_mask(MaskType m) {
    static_assert(0 <= Index && Index < num_words, "Static index out of range");
    return svget4(m, (uint64_t)Index);
  }

  VECOPS_SVE2P1_TARGET VECOPS_PURE
  static WordMask get_mask(MaskType m, nint_t index) {
    VECOPS_ASSERT(0 <= index && index < num_words, "%lld !in 0..%lld", index, num_words);
    if (index == 0)      return svget4(m, 0);
    else if (index == 1) return svget4(m, 1);
    else if (index == 2) return svget4(m, 2);
    else                 return svget4(m, 3);
  }

  template <nint_t Index>
  VECOPS_SVE2P1_TARGET VECOPS_PURE
  static MaskType set_mask(MaskType m, WordMask u) {
    static_assert(0 <= Index && Index < num_words, "Static index out of range");
    return svset4(m, (uint64_t)Index, u);
  }

  VECOPS_SVE2P1_TARGET VECOPS_PURE
  static MaskType set_mask(MaskType m, nint_t index, WordMask u) {
    VECOPS_ASSERT(0 <= index && index < num_words, "%lld !in 0..%lld", index, num_words);
    if (index == 0)      return svset4(m, 0, u);
    else if (index == 1) return svset4(m, 1, u);
    else if (index == 2) return svset4(m, 2, u);
    else                 return svset4(m, 3, u);
  }
};

// =========================================================================
// VecDefs: POW2 < 0  --  fractional single-register SVE vector
// =========================================================================
template <typename T, int POW2>
struct VecDefs<T, -1, POW2, std::enable_if_t<(POW2 < 0)>> : public BaseVecDefs<T, -1, POW2> {
  using TagType  = Tag<T, -1, POW2>;
  using WordDefs = VecDefs<T, -1, 0>;
  using WordVec  = typename WordDefs::VecType;
  using WordMask = typename WordDefs::MaskType;

  static constexpr nint_t num_words = 1;

  static nint_t word_size() { return WordDefs::word_size(); }
  static constexpr nint_t max_word_size = WordDefs::max_word_size;

  static nint_t size() { return WordDefs::word_size() >> (-POW2); }
  static constexpr nint_t max_size = []{
    constexpr nint_t raw = WordDefs::max_word_size >> (-POW2);
    return raw > 0 ? raw : 1;
  }();

  static constexpr bool is_scalable   = true;
  static constexpr bool is_default_impl = false;
  static constexpr bool is_word_vec   = true;

  using VecType  = WordVec;
  using MaskType = WordMask;

  template <nint_t Index>
  VECOPS_VFUNC VECOPS_PURE
  static VecType get(VecType v) {
    static_assert(Index == 0, "Static index out of range for fractional SVE vector");
    return v;
  }

  VECOPS_VFUNC VECOPS_PURE
  static VecType get(VecType v, nint_t index) {
    VECOPS_ASSERT(index == 0, "%lld !in 0..1", index);
    return v;
  }

  template <nint_t Index>
  VECOPS_VFUNC VECOPS_PURE
  static VecType set(VecType v, VecType u) {
    static_assert(Index == 0, "Static index out of range for fractional SVE vector");
    return u;
  }

  VECOPS_VFUNC VECOPS_PURE
  static VecType set(VecType v, nint_t index, VecType u) {
    VECOPS_ASSERT(index == 0, "%lld !in 0..1", index);
    return u;
  }

  template <nint_t Index>
  VECOPS_VFUNC VECOPS_PURE
  static MaskType get_mask(MaskType m) {
    static_assert(Index == 0, "Static index out of range for fractional SVE vector");
    return m;
  }

  VECOPS_VFUNC VECOPS_PURE
  static MaskType get_mask(MaskType m, nint_t index) {
    VECOPS_ASSERT(index == 0, "%lld !in 0..1", index);
    return m;
  }

  template <nint_t Index>
  VECOPS_VFUNC VECOPS_PURE
  static MaskType set_mask(MaskType m, MaskType u) {
    static_assert(Index == 0, "Static index out of range for fractional SVE vector");
    return u;
  }

  VECOPS_VFUNC VECOPS_PURE
  static MaskType set_mask(MaskType m, nint_t index, MaskType u) {
    VECOPS_ASSERT(index == 0, "%lld !in 0..1", index);
    return u;
  }
};

// =========================================================================
// Reject fixed-size Tag  (N > 0)
// =========================================================================
template <typename T, nint_t N, int POW2>
struct VecDefs<T, N, POW2, std::enable_if_t<(N > 0)>> {
  static_assert(sizeof(T) == -1,
      "Fixed-size Tag (N>0) is not supported on SVE. Use ScalableTag<T>.");
};

// =========================================================================
// Reject POW2 > 2  (more than 4 SVE registers)
// =========================================================================
template <typename T, int POW2>
struct VecDefs<T, -1, POW2, std::enable_if_t<(POW2 > 2)>> : public BaseVecDefs<T, -1, POW2> {
  static_assert(sizeof(T) == 0, "Only POW2 <= 2 is supported for SVE");
};

// =========================================================================
// Vec2TagDefs  --  map SVE intrinsic types back to Tag
// =========================================================================
#define TL_SVE_VEC2TAG(dtype, pow2, raw_type) \
template <> struct Vec2TagDefs<raw_type> { using Type = Tag<dtype##_t, -1, pow2>; }

#define TL_SVE_VEC2TAG_ALL(dtype, prefix) \
TL_SVE_VEC2TAG(dtype, 0, sv##prefix##_t); \
TL_SVE_VEC2TAG(dtype, 1, sv##prefix##x2_t); \
TL_SVE_VEC2TAG(dtype, 2, sv##prefix##x4_t)

TL_SVE_VEC2TAG_ALL(bfloat16, bfloat16);
TL_SVE_VEC2TAG_ALL(float16,   float16);
TL_SVE_VEC2TAG_ALL(float32,   float32);
TL_SVE_VEC2TAG_ALL(float64,   float64);
TL_SVE_VEC2TAG_ALL(int8,      int8);
TL_SVE_VEC2TAG_ALL(uint8,     uint8);
TL_SVE_VEC2TAG_ALL(int16,     int16);
TL_SVE_VEC2TAG_ALL(uint16,    uint16);
TL_SVE_VEC2TAG_ALL(int32,     int32);
TL_SVE_VEC2TAG_ALL(uint32,    uint32);
TL_SVE_VEC2TAG_ALL(int64,     int64);
TL_SVE_VEC2TAG_ALL(uint64,    uint64);

#undef TL_SVE_VEC2TAG_ALL
#undef TL_SVE_VEC2TAG

// =========================================================================
// IsMask  --  identify SVE predicate types as masks
// =========================================================================
template <> struct IsMask<svbool_t>   : public std::true_type {};
template <> struct IsMask<svboolx2_t> : public std::true_type {};
template <> struct IsMask<svboolx4_t> : public std::true_type {};

} // namespace vecops::vec

#endif // VECOPS_SVE_TYPES_H
