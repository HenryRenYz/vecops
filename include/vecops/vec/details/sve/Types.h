#ifndef VECOPS_VEC_DETAILS_SVE_TYPES_H
#define VECOPS_VEC_DETAILS_SVE_TYPES_H

/**
 * @file Types.h
 * @brief SVE backend representation types, covering both sizeless (VLA)
 * and vector-length-specific (VLS) compilation modes.
 *
 * In sizeless mode, SVEScalableVector<T, P> wraps sv<T>_t for single-word
 * Tags and sv<T>x2_t / sv<T>x4_t for multi-word Tags. Masks use svbool_t,
 * svboolx2_t, or svboolx4_t. VLA FixedTag blocks up to the architectural
 * minimum 16 bytes occupy one sizeless register with an exact lane prefix.
 * Larger fixed blocks require VLS. In VLS mode SVEFixedVector<T> is sized.
 */

#if !defined(HAS_SVE)
#error "This header requires an SVE target"
#endif

#include <arm_sve.h>

#include <cstdint>
#include <type_traits>

#include "vecops/vec/details/Representation.h"
#include "vecops/vec/details/Storage.h"

namespace vecops::vec::details {

template <typename>
inline constexpr bool dependent_false = false;


/* **************************************************************************** */
//                          SVE representation types                          //
/* **************************************************************************** */

template <Element T>
struct SVERawVector;

template <> struct SVERawVector<bfloat16_t> { using type = svbfloat16_t; };
template <> struct SVERawVector<float16_t> { using type = svfloat16_t; };
template <> struct SVERawVector<float32_t> { using type = svfloat32_t; };
template <> struct SVERawVector<float64_t> { using type = svfloat64_t; };
template <> struct SVERawVector<int8_t> { using type = svint8_t; };
template <> struct SVERawVector<uint8_t> { using type = svuint8_t; };
template <> struct SVERawVector<int16_t> { using type = svint16_t; };
template <> struct SVERawVector<uint16_t> { using type = svuint16_t; };
template <> struct SVERawVector<int32_t> { using type = svint32_t; };
template <> struct SVERawVector<uint32_t> { using type = svuint32_t; };
template <> struct SVERawVector<int64_t> { using type = svint64_t; };
template <> struct SVERawVector<uint64_t> { using type = svuint64_t; };

template <Element T, int ScalePower>
struct SVEScalableVector;

template <Element T>
struct SVEScalableVector<T, 0> : SVERawVector<T> {};

#define VECOPS_VEC_DEFINE_SVE_TUPLES(ElementType, Name)                    \
  template <> struct SVEScalableVector<ElementType, 1> {                   \
    using type = sv##Name##x2_t;                                           \
  };                                                                        \
  template <> struct SVEScalableVector<ElementType, 2> {                   \
    using type = sv##Name##x4_t;                                           \
  }

VECOPS_VEC_DEFINE_SVE_TUPLES(bfloat16_t, bfloat16);
VECOPS_VEC_DEFINE_SVE_TUPLES(float16_t, float16);
VECOPS_VEC_DEFINE_SVE_TUPLES(float32_t, float32);
VECOPS_VEC_DEFINE_SVE_TUPLES(float64_t, float64);
VECOPS_VEC_DEFINE_SVE_TUPLES(int8_t, int8);
VECOPS_VEC_DEFINE_SVE_TUPLES(uint8_t, uint8);
VECOPS_VEC_DEFINE_SVE_TUPLES(int16_t, int16);
VECOPS_VEC_DEFINE_SVE_TUPLES(uint16_t, uint16);
VECOPS_VEC_DEFINE_SVE_TUPLES(int32_t, int32);
VECOPS_VEC_DEFINE_SVE_TUPLES(uint32_t, uint32);
VECOPS_VEC_DEFINE_SVE_TUPLES(int64_t, int64);
VECOPS_VEC_DEFINE_SVE_TUPLES(uint64_t, uint64);

#undef VECOPS_VEC_DEFINE_SVE_TUPLES

template <int ScalePower>
struct SVEScalableMask;

#if !defined(HAS_FIXED_SVE_BITS)

#if !defined(HAS_SVE_PREDICATE_TUPLES)
#error "VLA multiword SVE masks require predicate-tuple support; use GCC 13 or newer, Clang, or compile with -msve-vector-bits=<bits> to select the fixed-SVE array representation"
#endif

template <> struct SVEScalableMask<0> { using type = svbool_t; };
template <> struct SVEScalableMask<1> { using type = svboolx2_t; };
template <> struct SVEScalableMask<2> { using type = svboolx4_t; };

// Clang 19's ACLE predicate-tuple aliases require SVE2.1, but tuple grouping
// itself is a target-independent extract/insert, not a new ISA instruction.
// A target(sve2p1) helper cannot inline into a base-SVE owner and would violate
// the manual SME no-call boundary. Keep this compiler-only bridge local to
// the VLA representation layer: these are the exact LLVM intrinsics used by
// Clang's ACLE lowering, with constant indices in minimum predicate elements
// (scaled by vscale, not by a captured physical VL). No LLVM runtime is linked.
// https://llvm.org/docs/LangRef.html#llvm-vector-extract-intrinsic
// https://llvm.org/docs/LangRef.html#llvm-vector-insert-intrinsic
// GCC and actual SVE2.1 targets retain the ordinary ACLE entry points below.
#if defined(COMPILER_CLANG) && !defined(HAS_SVE2P1)
namespace predicate_tuple_bridge {
extern "C" svbool_t vecops_predicate_extract2(svboolx2_t, std::uint64_t)
    noexcept asm("llvm.vector.extract.nxv16i1.nxv32i1");
extern "C" svbool_t vecops_predicate_extract4(svboolx4_t, std::uint64_t)
    noexcept asm("llvm.vector.extract.nxv16i1.nxv64i1");
extern "C" svboolx2_t vecops_predicate_insert2(
    svboolx2_t, svbool_t, std::uint64_t)
    noexcept asm("llvm.vector.insert.nxv32i1.nxv16i1");
extern "C" svboolx4_t vecops_predicate_insert4(
    svboolx4_t, svbool_t, std::uint64_t)
    noexcept asm("llvm.vector.insert.nxv64i1.nxv16i1");
} // namespace predicate_tuple_bridge
#endif

template <nint_t Index>
VECOPS_ALWAYS_INLINE svbool_t get_word(svboolx2_t value) {
  static_assert(Index >= 0 && Index < 2);
#if defined(COMPILER_CLANG) && !defined(HAS_SVE2P1)
  return predicate_tuple_bridge::vecops_predicate_extract2(value, 16 * Index);
#else
  return svget2(value, static_cast<uint64_t>(Index));
#endif
}

template <nint_t Index>
VECOPS_ALWAYS_INLINE svboolx2_t set_word(
    svboolx2_t value, svbool_t word) {
  static_assert(Index >= 0 && Index < 2);
#if defined(COMPILER_CLANG) && !defined(HAS_SVE2P1)
  return predicate_tuple_bridge::vecops_predicate_insert2(value, word, 16 * Index);
#else
  return svset2(value, static_cast<uint64_t>(Index), word);
#endif
}

template <nint_t Index>
VECOPS_ALWAYS_INLINE svbool_t get_word(svboolx4_t value) {
  static_assert(Index >= 0 && Index < 4);
#if defined(COMPILER_CLANG) && !defined(HAS_SVE2P1)
  return predicate_tuple_bridge::vecops_predicate_extract4(value, 16 * Index);
#else
  return svget4(value, static_cast<uint64_t>(Index));
#endif
}

template <nint_t Index>
VECOPS_ALWAYS_INLINE svboolx4_t set_word(
    svboolx4_t value, svbool_t word) {
  static_assert(Index >= 0 && Index < 4);
#if defined(COMPILER_CLANG) && !defined(HAS_SVE2P1)
  return predicate_tuple_bridge::vecops_predicate_insert4(value, word, 16 * Index);
#else
  return svset4(value, static_cast<uint64_t>(Index), word);
#endif
}

VECOPS_ALWAYS_INLINE svboolx2_t make_word_group(
    svbool_t word0, svbool_t word1) {
#if defined(COMPILER_CLANG) && !defined(HAS_SVE2P1)
  svboolx2_t result{};
  result = predicate_tuple_bridge::vecops_predicate_insert2(result, word0, 0);
  return predicate_tuple_bridge::vecops_predicate_insert2(result, word1, 16);
#else
  return svcreate2(word0, word1);
#endif
}

VECOPS_ALWAYS_INLINE svboolx4_t make_word_group(
    svbool_t word0, svbool_t word1, svbool_t word2, svbool_t word3) {
#if defined(COMPILER_CLANG) && !defined(HAS_SVE2P1)
  svboolx4_t result{};
  result = predicate_tuple_bridge::vecops_predicate_insert4(result, word0, 0);
  result = predicate_tuple_bridge::vecops_predicate_insert4(result, word1, 16);
  result = predicate_tuple_bridge::vecops_predicate_insert4(result, word2, 32);
  return predicate_tuple_bridge::vecops_predicate_insert4(result, word3, 48);
#else
  return svcreate4(word0, word1, word2, word3);
#endif
}

#endif // !defined(HAS_FIXED_SVE_BITS)

#define VECOPS_VEC_DEFINE_SVE_TUPLE_ACCESS(Name)                         \
  template <nint_t Index>                                                \
  VECOPS_ALWAYS_INLINE sv##Name##_t get_word(sv##Name##x2_t value) {     \
    static_assert(Index >= 0 && Index < 2);                              \
    return svget2(value, static_cast<uint64_t>(Index));                  \
  }                                                                      \
  template <nint_t Index>                                                \
  VECOPS_ALWAYS_INLINE sv##Name##x2_t set_word(                          \
      sv##Name##x2_t value, sv##Name##_t word) {                         \
    static_assert(Index >= 0 && Index < 2);                              \
    return svset2(value, static_cast<uint64_t>(Index), word);            \
  }                                                                      \
  template <nint_t Index>                                                \
  VECOPS_ALWAYS_INLINE sv##Name##_t get_word(sv##Name##x4_t value) {     \
    static_assert(Index >= 0 && Index < 4);                              \
    return svget4(value, static_cast<uint64_t>(Index));                  \
  }                                                                      \
  template <nint_t Index>                                                \
  VECOPS_ALWAYS_INLINE sv##Name##x4_t set_word(                          \
      sv##Name##x4_t value, sv##Name##_t word) {                         \
    static_assert(Index >= 0 && Index < 4);                              \
    return svset4(value, static_cast<uint64_t>(Index), word);            \
  }

VECOPS_VEC_DEFINE_SVE_TUPLE_ACCESS(float16)
VECOPS_VEC_DEFINE_SVE_TUPLE_ACCESS(float32)
VECOPS_VEC_DEFINE_SVE_TUPLE_ACCESS(float64)
VECOPS_VEC_DEFINE_SVE_TUPLE_ACCESS(int8)
VECOPS_VEC_DEFINE_SVE_TUPLE_ACCESS(uint8)
VECOPS_VEC_DEFINE_SVE_TUPLE_ACCESS(int16)
VECOPS_VEC_DEFINE_SVE_TUPLE_ACCESS(uint16)
VECOPS_VEC_DEFINE_SVE_TUPLE_ACCESS(int32)
VECOPS_VEC_DEFINE_SVE_TUPLE_ACCESS(uint32)
VECOPS_VEC_DEFINE_SVE_TUPLE_ACCESS(int64)
VECOPS_VEC_DEFINE_SVE_TUPLE_ACCESS(uint64)

#undef VECOPS_VEC_DEFINE_SVE_TUPLE_ACCESS

#define VECOPS_VEC_DEFINE_SVE_WORD_GROUP(Name)                         \
  VECOPS_ALWAYS_INLINE sv##Name##x2_t make_word_group(                 \
      sv##Name##_t word0, sv##Name##_t word1) {                        \
    return svcreate2(word0, word1);                                    \
  }                                                                    \
  VECOPS_ALWAYS_INLINE sv##Name##x4_t make_word_group(                 \
      sv##Name##_t word0, sv##Name##_t word1,                          \
      sv##Name##_t word2, sv##Name##_t word3) {                        \
    return svcreate4(word0, word1, word2, word3);                      \
  }

VECOPS_VEC_DEFINE_SVE_WORD_GROUP(float16)
VECOPS_VEC_DEFINE_SVE_WORD_GROUP(float32)
VECOPS_VEC_DEFINE_SVE_WORD_GROUP(float64)
VECOPS_VEC_DEFINE_SVE_WORD_GROUP(int8)
VECOPS_VEC_DEFINE_SVE_WORD_GROUP(uint8)
VECOPS_VEC_DEFINE_SVE_WORD_GROUP(int16)
VECOPS_VEC_DEFINE_SVE_WORD_GROUP(uint16)
VECOPS_VEC_DEFINE_SVE_WORD_GROUP(int32)
VECOPS_VEC_DEFINE_SVE_WORD_GROUP(uint32)
VECOPS_VEC_DEFINE_SVE_WORD_GROUP(int64)
VECOPS_VEC_DEFINE_SVE_WORD_GROUP(uint64)

#undef VECOPS_VEC_DEFINE_SVE_WORD_GROUP

template <nint_t Index>
VECOPS_ALWAYS_INLINE svbfloat16_t get_word(svbfloat16x2_t value) {
  static_assert(Index >= 0 && Index < 2);
#if defined(HAS_BF16)
  return svget2(value, static_cast<uint64_t>(Index));
#else
  return svreinterpret_bf16_u16(svget2(
      svreinterpret_u16_bf16_x2(value), static_cast<uint64_t>(Index)));
#endif
}

template <nint_t Index>
VECOPS_ALWAYS_INLINE svbfloat16x2_t set_word(
    svbfloat16x2_t value, svbfloat16_t word) {
  static_assert(Index >= 0 && Index < 2);
#if defined(HAS_BF16)
  return svset2(value, static_cast<uint64_t>(Index), word);
#else
  return svreinterpret_bf16_u16_x2(svset2(
      svreinterpret_u16_bf16_x2(value),
      static_cast<uint64_t>(Index),
      svreinterpret_u16_bf16(word)));
#endif
}

template <nint_t Index>
VECOPS_ALWAYS_INLINE svbfloat16_t get_word(svbfloat16x4_t value) {
  static_assert(Index >= 0 && Index < 4);
#if defined(HAS_BF16)
  return svget4(value, static_cast<uint64_t>(Index));
#else
  return svreinterpret_bf16_u16(svget4(
      svreinterpret_u16_bf16_x4(value), static_cast<uint64_t>(Index)));
#endif
}

template <nint_t Index>
VECOPS_ALWAYS_INLINE svbfloat16x4_t set_word(
    svbfloat16x4_t value, svbfloat16_t word) {
  static_assert(Index >= 0 && Index < 4);
#if defined(HAS_BF16)
  return svset4(value, static_cast<uint64_t>(Index), word);
#else
  return svreinterpret_bf16_u16_x4(svset4(
      svreinterpret_u16_bf16_x4(value),
      static_cast<uint64_t>(Index),
      svreinterpret_u16_bf16(word)));
#endif
}

VECOPS_ALWAYS_INLINE svbfloat16x2_t make_word_group(
    svbfloat16_t word0, svbfloat16_t word1) {
#if defined(HAS_BF16)
  return svcreate2(word0, word1);
#else
  return svreinterpret_bf16_u16_x2(svcreate2(
      svreinterpret_u16_bf16(word0),
      svreinterpret_u16_bf16(word1)));
#endif
}

VECOPS_ALWAYS_INLINE svbfloat16x4_t make_word_group(
    svbfloat16_t word0, svbfloat16_t word1,
    svbfloat16_t word2, svbfloat16_t word3) {
#if defined(HAS_BF16)
  return svcreate4(word0, word1, word2, word3);
#else
  return svreinterpret_bf16_u16_x4(svcreate4(
      svreinterpret_u16_bf16(word0),
      svreinterpret_u16_bf16(word1),
      svreinterpret_u16_bf16(word2),
      svreinterpret_u16_bf16(word3)));
#endif
}

template <Element T>
VECOPS_ALWAYS_INLINE nint_t sve_word_lanes() {
#if !defined(HAS_FIXED_SVE_BITS)
#  if defined(VECOPS_KERNEL_SVE_BYTES) && defined(VECOPS_KERNEL_SME_BYTES) && \
      VECOPS_KERNEL_SVE_BYTES > 0 && \
      (VECOPS_KERNEL_SME_BYTES == 0 || VECOPS_KERNEL_SVE_BYTES == VECOPS_KERNEL_SME_BYTES)
  // A JIT adapter validates the captured lengths before entry. If both
  // execution domains have the same width, retaining LLVM's sizeless type
  // still permits compile-time sizes without introducing vector arrays.
  return VECOPS_KERNEL_SVE_BYTES / static_cast<nint_t>(sizeof(T));
#  else
  // LLVM models svcntb() as readnone. It can otherwise be CSE'd/hoisted
  // across our manual SMSTART owner, producing VL instead of SVL when the
  // two are different. A volatile architectural read observes the current
  // execution domain and cannot move across the streaming boundaries.
  nint_t bytes;
  asm volatile("cntb %0" : "=r"(bytes) : : "memory");
  return bytes / static_cast<nint_t>(sizeof(T));
#  endif
#else
  if constexpr (sizeof(T) == 1) return static_cast<nint_t>(svcntb());
  if constexpr (sizeof(T) == 2) return static_cast<nint_t>(svcnth());
  if constexpr (sizeof(T) == 4) return static_cast<nint_t>(svcntw());
  return static_cast<nint_t>(svcntd());
#endif
}

#if !defined(HAS_FIXED_SVE_BITS)

template <Element T, int ScalePower>
struct RepresentationTraits<
    SVEBackend,
    VectorDescriptor<T, ScalableExtent<ScalePower>>> {
  static_assert(
      ScalePower <= 2,
      "Scalable SVE supports at most four sizeless words; use FixedTag in a VLS build for larger vectors");
  static constexpr nint_t minimum_word_lanes = 16 / sizeof(T);
  static_assert(
      ScalePower >= 0 || (minimum_word_lanes >> (-ScalePower)) > 0,
      "ScalableTag subword may contain no lanes on a conforming SVE target");

  using Element = T;
  using WordVec = typename SVERawVector<T>::type;
  using WordMask = svbool_t;
  using VecType = typename SVEScalableVector<T, (ScalePower > 0 ? ScalePower : 0)>::type;
  using MaskType = typename SVEScalableMask<(ScalePower > 0 ? ScalePower : 0)>::type;

  static constexpr nint_t word_count =
      ScalePower > 0 ? (nint_t{1} << ScalePower) : 1;
  static constexpr bool is_runtime_size_v = true;
  static constexpr bool is_subword_v = ScalePower < 0;

  static VECOPS_ALWAYS_INLINE nint_t word_lanes() {
    return sve_word_lanes<T>();
  }

  static VECOPS_ALWAYS_INLINE nint_t logical_lanes() {
    if constexpr (ScalePower >= 0) {
      return word_lanes() << ScalePower;
    } else {
      return word_lanes() >> (-ScalePower);
    }
  }
};

#endif

#if defined(HAS_FIXED_SVE_BITS)

template <Element T>
struct SVEFixedRawVector;

#define VECOPS_VEC_DEFINE_FIXED_SVE(ElementType, Name)                    \
  template <> struct SVEFixedRawVector<ElementType> {                      \
    using type = sv##Name##_t                                              \
        __attribute__((arm_sve_vector_bits(FIXED_SVE_BITS)));              \
  }

VECOPS_VEC_DEFINE_FIXED_SVE(bfloat16_t, bfloat16);
VECOPS_VEC_DEFINE_FIXED_SVE(float16_t, float16);
VECOPS_VEC_DEFINE_FIXED_SVE(float32_t, float32);
VECOPS_VEC_DEFINE_FIXED_SVE(float64_t, float64);
VECOPS_VEC_DEFINE_FIXED_SVE(int8_t, int8);
VECOPS_VEC_DEFINE_FIXED_SVE(uint8_t, uint8);
VECOPS_VEC_DEFINE_FIXED_SVE(int16_t, int16);
VECOPS_VEC_DEFINE_FIXED_SVE(uint16_t, uint16);
VECOPS_VEC_DEFINE_FIXED_SVE(int32_t, int32);
VECOPS_VEC_DEFINE_FIXED_SVE(uint32_t, uint32);
VECOPS_VEC_DEFINE_FIXED_SVE(int64_t, int64);
VECOPS_VEC_DEFINE_FIXED_SVE(uint64_t, uint64);

#undef VECOPS_VEC_DEFINE_FIXED_SVE

/** Sized VLS word that retains its element type for Vec-to-Tag inference. */
template <Element T>
struct SVEFixedVector {
  using RawType = typename SVEFixedRawVector<T>::type;
  static constexpr nint_t physical_lanes =
      FIXED_SVE_BITS / 8 / static_cast<nint_t>(sizeof(T));

  RawType value;

  constexpr SVEFixedVector() = default;
  constexpr SVEFixedVector(RawType raw) : value(raw) {}
};

using SVEFixedMask =
    svbool_t __attribute__((arm_sve_vector_bits(FIXED_SVE_BITS)));

template <Element T, int ScalePower>
struct RepresentationTraits<
    SVEBackend,
    VectorDescriptor<T, ScalableExtent<ScalePower>>> {
  static constexpr nint_t word_lanes =
      FIXED_SVE_BITS / 8 / static_cast<nint_t>(sizeof(T));
  static_assert(
      ScalePower >= 0 || (word_lanes >> (-ScalePower)) > 0,
      "ScalableTag subword contains no lanes at the fixed SVE width");

  using Element = T;
  using WordVec = SVEFixedVector<T>;
  using WordMask = SVEFixedMask;
  static constexpr nint_t logical_lanes = ScalePower >= 0
      ? (word_lanes << ScalePower)
      : (word_lanes >> (-ScalePower));
  static constexpr nint_t word_count =
      ScalePower > 0 ? (nint_t{1} << ScalePower) : 1;
  static constexpr bool is_runtime_size_v = false;
  static constexpr bool is_subword_v = ScalePower < 0;

  // Sized SVE words can be ordinary array members. Besides lifting the
  // architectural x2/x4 tuple limit, this keeps multiword predicates usable
  // without requiring SVE2.1 tuple operations or non-inline target shims.
  using VecType = SingleOrArray<WordVec, word_count>;
  using MaskType = SingleOrArray<WordMask, word_count>;
};

template <Element T, nint_t N>
struct RepresentationTraits<
    SVEBackend,
    VectorDescriptor<T, FixedExtent<N>>> {
  using Element = T;
  static constexpr nint_t word_lanes =
      FIXED_SVE_BITS / 8 / static_cast<nint_t>(sizeof(T));
  static constexpr nint_t logical_lanes = N;
  static constexpr nint_t word_count =
      (logical_lanes + word_lanes - 1) / word_lanes;
  static constexpr bool is_runtime_size_v = false;
  static constexpr bool is_subword_v = logical_lanes < word_lanes;

  using WordVec = SVEFixedVector<T>;
  using WordMask = SVEFixedMask;
  using VecType = SingleOrArray<WordVec, word_count>;
  using MaskType = SingleOrArray<WordMask, word_count>;
};

#else

template <Element T, nint_t N>
struct RepresentationTraits<
    SVEBackend,
    VectorDescriptor<T, FixedExtent<N>>> {
  static_assert(
      N * sizeof(T) <= 16,
      "VLA SVE FixedTag supports at most 16 bytes; larger blocks require -msve-vector-bits=<N> or ScalableTag");
  using Element = T;
  using WordVec = typename SVERawVector<T>::type;
  using WordMask = svbool_t;
  using VecType = WordVec;
  using MaskType = WordMask;
  static constexpr nint_t logical_lanes = N;
  static constexpr nint_t word_count = 1;
  static constexpr bool is_runtime_size_v = false;
  static constexpr bool is_subword_v = true;
  static VECOPS_ALWAYS_INLINE nint_t word_lanes() {
    return sve_word_lanes<T>();
  }
};

#endif

#define VECOPS_VEC_REGISTER_SVE_VECTOR(ElementType, Name)                 \
  template <> struct IsVectorRepresentation<sv##Name##_t>                 \
      : std::true_type {};                                                 \
  template <> struct IsVectorRepresentation<sv##Name##x2_t>               \
      : std::true_type {};                                                 \
  template <> struct IsVectorRepresentation<sv##Name##x4_t>               \
      : std::true_type {};                                                 \
  template <> struct InferredTagTraits<sv##Name##_t> {                    \
    using type = ScalableTag<ElementType, 0>;                              \
  };                                                                        \
  template <> struct InferredTagTraits<sv##Name##x2_t> {                  \
    using type = ScalableTag<ElementType, 1>;                              \
  };                                                                        \
  template <> struct InferredTagTraits<sv##Name##x4_t> {                  \
    using type = ScalableTag<ElementType, 2>;                              \
  }

VECOPS_VEC_REGISTER_SVE_VECTOR(bfloat16_t, bfloat16);
VECOPS_VEC_REGISTER_SVE_VECTOR(float16_t, float16);
VECOPS_VEC_REGISTER_SVE_VECTOR(float32_t, float32);
VECOPS_VEC_REGISTER_SVE_VECTOR(float64_t, float64);
VECOPS_VEC_REGISTER_SVE_VECTOR(int8_t, int8);
VECOPS_VEC_REGISTER_SVE_VECTOR(uint8_t, uint8);
VECOPS_VEC_REGISTER_SVE_VECTOR(int16_t, int16);
VECOPS_VEC_REGISTER_SVE_VECTOR(uint16_t, uint16);
VECOPS_VEC_REGISTER_SVE_VECTOR(int32_t, int32);
VECOPS_VEC_REGISTER_SVE_VECTOR(uint32_t, uint32);
VECOPS_VEC_REGISTER_SVE_VECTOR(int64_t, int64);
VECOPS_VEC_REGISTER_SVE_VECTOR(uint64_t, uint64);

#undef VECOPS_VEC_REGISTER_SVE_VECTOR

template <> struct IsMaskRepresentation<svbool_t> : std::true_type {};

#if !defined(HAS_FIXED_SVE_BITS)
template <> struct IsMaskRepresentation<svboolx2_t> : std::true_type {};
template <> struct IsMaskRepresentation<svboolx4_t> : std::true_type {};
#endif

#if defined(HAS_FIXED_SVE_BITS)

template <Element T>
struct IsVectorRepresentation<SVEFixedVector<T>> : std::true_type {};

template <Element T>
struct InferredTagTraits<SVEFixedVector<T>> {
  using type = FixedTag<T, SVEFixedVector<T>::physical_lanes>;
};

template <Element T, nint_t Count>
struct IsVectorRepresentation<
    WordArray<SVEFixedVector<T>, Count>> : std::true_type {};

template <Element T, nint_t Count>
struct InferredTagTraits<
    WordArray<SVEFixedVector<T>, Count>> {
  using type = FixedTag<
      T, Count * FIXED_SVE_BITS / 8 / static_cast<nint_t>(sizeof(T))>;
};

template <> struct IsMaskRepresentation<SVEFixedMask> : std::true_type {};

template <nint_t Count>
struct IsMaskRepresentation<WordArray<SVEFixedMask, Count>> : std::true_type {};

#endif

} // namespace vecops::vec::details

#endif // VECOPS_VEC_DETAILS_SVE_TYPES_H
