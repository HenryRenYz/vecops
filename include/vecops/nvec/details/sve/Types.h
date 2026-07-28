#ifndef VECOPS_NVEC_DETAILS_SVE_TYPES_H
#define VECOPS_NVEC_DETAILS_SVE_TYPES_H

#if !defined(HAS_SVE)
#error "This header requires an SVE target"
#endif

#include <arm_sve.h>

#include <type_traits>

#include "vecops/nvec/details/Representation.h"
#include "vecops/nvec/details/Storage.h"

namespace vecops::nvec::details {

template <typename>
inline constexpr bool dependent_false = false;

template <Element T>
struct SVERawVector;

template <> struct SVERawVector<bfloat16_t> { using Type = svbfloat16_t; };
template <> struct SVERawVector<float16_t> { using Type = svfloat16_t; };
template <> struct SVERawVector<float32_t> { using Type = svfloat32_t; };
template <> struct SVERawVector<float64_t> { using Type = svfloat64_t; };
template <> struct SVERawVector<int8_t> { using Type = svint8_t; };
template <> struct SVERawVector<uint8_t> { using Type = svuint8_t; };
template <> struct SVERawVector<int16_t> { using Type = svint16_t; };
template <> struct SVERawVector<uint16_t> { using Type = svuint16_t; };
template <> struct SVERawVector<int32_t> { using Type = svint32_t; };
template <> struct SVERawVector<uint32_t> { using Type = svuint32_t; };
template <> struct SVERawVector<int64_t> { using Type = svint64_t; };
template <> struct SVERawVector<uint64_t> { using Type = svuint64_t; };

template <Element T, int ScalePower>
struct SVEScalableVector;

template <Element T>
struct SVEScalableVector<T, 0> : SVERawVector<T> {};

#define VECOPS_NVEC_DEFINE_SVE_TUPLES(ElementType, Name)                    \
  template <> struct SVEScalableVector<ElementType, 1> {                   \
    using Type = sv##Name##x2_t;                                           \
  };                                                                        \
  template <> struct SVEScalableVector<ElementType, 2> {                   \
    using Type = sv##Name##x4_t;                                           \
  }

VECOPS_NVEC_DEFINE_SVE_TUPLES(bfloat16_t, bfloat16);
VECOPS_NVEC_DEFINE_SVE_TUPLES(float16_t, float16);
VECOPS_NVEC_DEFINE_SVE_TUPLES(float32_t, float32);
VECOPS_NVEC_DEFINE_SVE_TUPLES(float64_t, float64);
VECOPS_NVEC_DEFINE_SVE_TUPLES(int8_t, int8);
VECOPS_NVEC_DEFINE_SVE_TUPLES(uint8_t, uint8);
VECOPS_NVEC_DEFINE_SVE_TUPLES(int16_t, int16);
VECOPS_NVEC_DEFINE_SVE_TUPLES(uint16_t, uint16);
VECOPS_NVEC_DEFINE_SVE_TUPLES(int32_t, int32);
VECOPS_NVEC_DEFINE_SVE_TUPLES(uint32_t, uint32);
VECOPS_NVEC_DEFINE_SVE_TUPLES(int64_t, int64);
VECOPS_NVEC_DEFINE_SVE_TUPLES(uint64_t, uint64);

#undef VECOPS_NVEC_DEFINE_SVE_TUPLES

template <int ScalePower>
struct SVEScalableMask;

template <> struct SVEScalableMask<0> { using Type = svbool_t; };
template <> struct SVEScalableMask<1> { using Type = svboolx2_t; };
template <> struct SVEScalableMask<2> { using Type = svboolx4_t; };

// Predicate-tuple access is an SVE2.1 ACLE operation even though the tuple is
// also used in an SVE-only build. The target boundary is therefore confined
// to these VLA accessors. VLS masks use sized WordArray storage below and do
// not need this attribute or its resulting out-of-line call.
#if defined(HAS_SVE2P1)
#define VECOPS_NVEC_SVE_PREDICATE_ACCESS VECOPS_ALWAYS_INLINE
#else
#define VECOPS_NVEC_SVE_PREDICATE_ACCESS \
    __attribute__((target("sve2p1"))) inline
#endif

template <nint_t Index>
VECOPS_NVEC_SVE_PREDICATE_ACCESS svbool_t get_word(svboolx2_t value) {
  static_assert(Index >= 0 && Index < 2);
  return svget2(value, static_cast<uint64_t>(Index));
}

template <nint_t Index>
VECOPS_NVEC_SVE_PREDICATE_ACCESS svboolx2_t set_word(
    svboolx2_t value, svbool_t word) {
  static_assert(Index >= 0 && Index < 2);
  return svset2(value, static_cast<uint64_t>(Index), word);
}

template <nint_t Index>
VECOPS_NVEC_SVE_PREDICATE_ACCESS svbool_t get_word(svboolx4_t value) {
  static_assert(Index >= 0 && Index < 4);
  return svget4(value, static_cast<uint64_t>(Index));
}

template <nint_t Index>
VECOPS_NVEC_SVE_PREDICATE_ACCESS svboolx4_t set_word(
    svboolx4_t value, svbool_t word) {
  static_assert(Index >= 0 && Index < 4);
  return svset4(value, static_cast<uint64_t>(Index), word);
}

VECOPS_NVEC_SVE_PREDICATE_ACCESS svboolx2_t create_mask_tuple(
    svbool_t word0, svbool_t word1) {
  return svcreate2(word0, word1);
}

VECOPS_NVEC_SVE_PREDICATE_ACCESS svboolx4_t create_mask_tuple(
    svbool_t word0, svbool_t word1, svbool_t word2, svbool_t word3) {
  return svcreate4(word0, word1, word2, word3);
}

#undef VECOPS_NVEC_SVE_PREDICATE_ACCESS

#define VECOPS_NVEC_DEFINE_SVE_TUPLE_ACCESS(Name)                         \
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

VECOPS_NVEC_DEFINE_SVE_TUPLE_ACCESS(float16)
VECOPS_NVEC_DEFINE_SVE_TUPLE_ACCESS(float32)
VECOPS_NVEC_DEFINE_SVE_TUPLE_ACCESS(float64)
VECOPS_NVEC_DEFINE_SVE_TUPLE_ACCESS(int8)
VECOPS_NVEC_DEFINE_SVE_TUPLE_ACCESS(uint8)
VECOPS_NVEC_DEFINE_SVE_TUPLE_ACCESS(int16)
VECOPS_NVEC_DEFINE_SVE_TUPLE_ACCESS(uint16)
VECOPS_NVEC_DEFINE_SVE_TUPLE_ACCESS(int32)
VECOPS_NVEC_DEFINE_SVE_TUPLE_ACCESS(uint32)
VECOPS_NVEC_DEFINE_SVE_TUPLE_ACCESS(int64)
VECOPS_NVEC_DEFINE_SVE_TUPLE_ACCESS(uint64)

#undef VECOPS_NVEC_DEFINE_SVE_TUPLE_ACCESS

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

template <Element T>
VECOPS_ALWAYS_INLINE nint_t sve_word_lanes() {
  if constexpr (sizeof(T) == 1) return static_cast<nint_t>(svcntb());
  if constexpr (sizeof(T) == 2) return static_cast<nint_t>(svcnth());
  if constexpr (sizeof(T) == 4) return static_cast<nint_t>(svcntw());
  return static_cast<nint_t>(svcntd());
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
  using WordVec = typename SVERawVector<T>::Type;
  using WordMask = svbool_t;
  using VecType = typename SVEScalableVector<T, (ScalePower > 0 ? ScalePower : 0)>::Type;
  using MaskType = typename SVEScalableMask<(ScalePower > 0 ? ScalePower : 0)>::Type;

  static constexpr nint_t word_count =
      ScalePower > 0 ? (nint_t{1} << ScalePower) : 1;
  static constexpr bool is_runtime_size = true;
  static constexpr bool is_subword = ScalePower < 0;

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

#define VECOPS_NVEC_DEFINE_FIXED_SVE(ElementType, Name)                    \
  template <> struct SVEFixedRawVector<ElementType> {                      \
    using Type = sv##Name##_t                                              \
        __attribute__((arm_sve_vector_bits(FIXED_SVE_BITS)));              \
  }

VECOPS_NVEC_DEFINE_FIXED_SVE(bfloat16_t, bfloat16);
VECOPS_NVEC_DEFINE_FIXED_SVE(float16_t, float16);
VECOPS_NVEC_DEFINE_FIXED_SVE(float32_t, float32);
VECOPS_NVEC_DEFINE_FIXED_SVE(float64_t, float64);
VECOPS_NVEC_DEFINE_FIXED_SVE(int8_t, int8);
VECOPS_NVEC_DEFINE_FIXED_SVE(uint8_t, uint8);
VECOPS_NVEC_DEFINE_FIXED_SVE(int16_t, int16);
VECOPS_NVEC_DEFINE_FIXED_SVE(uint16_t, uint16);
VECOPS_NVEC_DEFINE_FIXED_SVE(int32_t, int32);
VECOPS_NVEC_DEFINE_FIXED_SVE(uint32_t, uint32);
VECOPS_NVEC_DEFINE_FIXED_SVE(int64_t, int64);
VECOPS_NVEC_DEFINE_FIXED_SVE(uint64_t, uint64);

#undef VECOPS_NVEC_DEFINE_FIXED_SVE

/** Sized VLS word that retains its element type for Vec-to-Tag inference. */
template <Element T>
struct SVEFixedVector {
  using RawType = typename SVEFixedRawVector<T>::Type;
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
  static constexpr bool is_runtime_size = false;
  static constexpr bool is_subword = ScalePower < 0;

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
  static constexpr bool is_runtime_size = false;
  static constexpr bool is_subword = logical_lanes < word_lanes;

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
      dependent_false<T>,
      "FixedTag on SVE requires -msve-vector-bits=<N>; use ScalableTag in a vector-length-agnostic build");
};

#endif

#define VECOPS_NVEC_REGISTER_SVE_VECTOR(ElementType, Name)                 \
  template <> struct IsVectorRepresentation<sv##Name##_t>                 \
      : std::true_type {};                                                 \
  template <> struct IsVectorRepresentation<sv##Name##x2_t>               \
      : std::true_type {};                                                 \
  template <> struct IsVectorRepresentation<sv##Name##x4_t>               \
      : std::true_type {};                                                 \
  template <> struct InferredTagTraits<sv##Name##_t> {                    \
    using Type = ScalableTag<ElementType, 0>;                              \
  };                                                                        \
  template <> struct InferredTagTraits<sv##Name##x2_t> {                  \
    using Type = ScalableTag<ElementType, 1>;                              \
  };                                                                        \
  template <> struct InferredTagTraits<sv##Name##x4_t> {                  \
    using Type = ScalableTag<ElementType, 2>;                              \
  }

VECOPS_NVEC_REGISTER_SVE_VECTOR(bfloat16_t, bfloat16);
VECOPS_NVEC_REGISTER_SVE_VECTOR(float16_t, float16);
VECOPS_NVEC_REGISTER_SVE_VECTOR(float32_t, float32);
VECOPS_NVEC_REGISTER_SVE_VECTOR(float64_t, float64);
VECOPS_NVEC_REGISTER_SVE_VECTOR(int8_t, int8);
VECOPS_NVEC_REGISTER_SVE_VECTOR(uint8_t, uint8);
VECOPS_NVEC_REGISTER_SVE_VECTOR(int16_t, int16);
VECOPS_NVEC_REGISTER_SVE_VECTOR(uint16_t, uint16);
VECOPS_NVEC_REGISTER_SVE_VECTOR(int32_t, int32);
VECOPS_NVEC_REGISTER_SVE_VECTOR(uint32_t, uint32);
VECOPS_NVEC_REGISTER_SVE_VECTOR(int64_t, int64);
VECOPS_NVEC_REGISTER_SVE_VECTOR(uint64_t, uint64);

#undef VECOPS_NVEC_REGISTER_SVE_VECTOR

template <> struct IsMaskRepresentation<svbool_t> : std::true_type {};
template <> struct IsMaskRepresentation<svboolx2_t> : std::true_type {};
template <> struct IsMaskRepresentation<svboolx4_t> : std::true_type {};

#if defined(HAS_FIXED_SVE_BITS)

template <Element T>
struct IsVectorRepresentation<SVEFixedVector<T>> : std::true_type {};

template <Element T>
struct InferredTagTraits<SVEFixedVector<T>> {
  using Type = FixedTag<T, SVEFixedVector<T>::physical_lanes>;
};

template <Element T, nint_t Count>
struct IsVectorRepresentation<
    WordArray<SVEFixedVector<T>, Count>> : std::true_type {};

template <Element T, nint_t Count>
struct InferredTagTraits<
    WordArray<SVEFixedVector<T>, Count>> {
  using Type = FixedTag<
      T, Count * FIXED_SVE_BITS / 8 / static_cast<nint_t>(sizeof(T))>;
};

template <> struct IsMaskRepresentation<SVEFixedMask> : std::true_type {};

template <nint_t Count>
struct IsMaskRepresentation<WordArray<SVEFixedMask, Count>> : std::true_type {};

#endif

} // namespace vecops::nvec::details

#endif // VECOPS_NVEC_DETAILS_SVE_TYPES_H
