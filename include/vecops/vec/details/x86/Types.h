#ifndef VECOPS_VEC_DETAILS_X86_TYPES_H
#define VECOPS_VEC_DETAILS_X86_TYPES_H

/**
 * @file Types.h
 * @brief x86 backend representation types: X86Vector (wrapping __m128,
 * __m256, __m512), X86Mask (wrapping __mmask* or __m128i), and the
 * RepresentationTraits connecting Tags to these types.
 */

#if !defined(ARCH_X86_FAMILY)
#error "This header requires an x86 target"
#endif

#include <immintrin.h>

#include <type_traits>
#include <utility>

#include "vecops/vec/details/Representation.h"
#include "vecops/vec/details/Storage.h"

namespace vecops::vec::details {


/* **************************************************************************** */
//                          x86 representation types                          //
/* **************************************************************************** */

template <Element T, nint_t Lanes, typename Raw>
struct X86Vector {
  using ElementType = T;
  static constexpr nint_t physical_lanes = Lanes;

  Raw value;

  constexpr X86Vector() = default;
  constexpr X86Vector(Raw raw) : value(raw) {}
};

template <nint_t ElementBytes, nint_t Lanes, typename Raw>
struct X86Mask {
  static constexpr nint_t element_bytes = ElementBytes;
  static constexpr nint_t physical_lanes = Lanes;

  Raw value;

  constexpr X86Mask() = default;
  constexpr X86Mask(Raw raw) : value(raw) {}
};

template <Element T, nint_t Bytes>
struct X86RawVector;

template <Element T>
struct X86RawVector<T, 16> {
  using Type = std::conditional_t<
      std::same_as<T, float32_t>, __m128,
      std::conditional_t<std::same_as<T, float64_t>, __m128d, __m128i>>;
};

template <Element T>
struct X86RawVector<T, 32> {
  using Type = std::conditional_t<
      std::same_as<T, float32_t>, __m256,
      std::conditional_t<std::same_as<T, float64_t>, __m256d, __m256i>>;
};

template <Element T>
struct X86RawVector<T, 64> {
  using Type = std::conditional_t<
      std::same_as<T, float32_t>, __m512,
      std::conditional_t<std::same_as<T, float64_t>, __m512d, __m512i>>;
};

template <nint_t Lanes>
struct X86PredicateRaw {
  using Type = std::conditional_t<
      (Lanes <= 8), __mmask8,
      std::conditional_t<
          (Lanes <= 16), __mmask16,
          std::conditional_t<(Lanes <= 32), __mmask32, __mmask64>>>;
};

template <nint_t Bytes>
struct X86VectorMaskRaw;

template <>
struct X86VectorMaskRaw<16> { using Type = __m128i; };
template <>
struct X86VectorMaskRaw<32> { using Type = __m256i; };
template <>
struct X86VectorMaskRaw<64> { using Type = __m512i; };

inline constexpr nint_t x86_native_bytes = VEC_WIDTH / 8;
static_assert(
    x86_native_bytes == 16 || x86_native_bytes == 32 || x86_native_bytes == 64,
    "Unsupported x86 native vector width");

consteval nint_t select_x86_word_bytes(nint_t logical_bytes) {
  if (logical_bytes <= 16) return 16;
  if (logical_bytes <= 32 && x86_native_bytes >= 32) return 32;
  if (logical_bytes <= 64 && x86_native_bytes >= 64) return 64;
  return x86_native_bytes;
}

template <VectorTag Tag>
struct X86Layout;

template <Element T, nint_t N>
struct X86Layout<VectorDescriptor<T, FixedExtent<N>>> {
  static constexpr nint_t logical_lanes = N;
  static constexpr nint_t logical_bytes =
      logical_lanes * static_cast<nint_t>(sizeof(T));
  static constexpr nint_t word_bytes = select_x86_word_bytes(logical_bytes);
  static constexpr nint_t word_lanes =
      word_bytes / static_cast<nint_t>(sizeof(T));
  static constexpr nint_t word_count =
      (logical_lanes + word_lanes - 1) / word_lanes;
  static constexpr bool is_runtime_size = false;
  static constexpr bool is_subword = logical_lanes < word_lanes;
};

template <Element T, int ScalePower>
struct X86Layout<VectorDescriptor<T, ScalableExtent<ScalePower>>> {
  static constexpr nint_t requested_bytes = [] {
    if constexpr (ScalePower >= 0) {
      return x86_native_bytes << ScalePower;
    } else {
      constexpr nint_t bytes = x86_native_bytes >> (-ScalePower);
      static_assert(
          bytes >= static_cast<nint_t>(sizeof(T)),
          "ScalableTag subword contains no lanes");
      return bytes;
    }
  }();
  static constexpr nint_t logical_lanes =
      requested_bytes / static_cast<nint_t>(sizeof(T));
  static constexpr nint_t word_bytes =
      ScalePower >= 0 ? x86_native_bytes : select_x86_word_bytes(requested_bytes);
  static constexpr nint_t word_lanes =
      word_bytes / static_cast<nint_t>(sizeof(T));
  static constexpr nint_t word_count =
      ScalePower > 0 ? (nint_t{1} << ScalePower) : 1;
  static constexpr bool is_runtime_size = false;
  static constexpr bool is_subword = requested_bytes < word_bytes;
};

template <VectorTag Tag>
struct RepresentationTraits<X86Backend, Tag> {
  using Element = ElementOf<Tag>;
  using Layout = X86Layout<Tag>;

  static constexpr nint_t logical_lanes = Layout::logical_lanes;
  static constexpr nint_t word_lanes = Layout::word_lanes;
  static constexpr nint_t word_count = Layout::word_count;
  static constexpr nint_t word_bytes = Layout::word_bytes;
  static constexpr bool is_runtime_size = Layout::is_runtime_size;
  static constexpr bool is_subword = Layout::is_subword;

  using RawVec = typename X86RawVector<Element, word_bytes>::Type;
  using WordVec = X86Vector<Element, word_lanes, RawVec>;

#if defined(CPU_CAPABILITY_AVX512)
  using RawMask = typename X86PredicateRaw<word_lanes>::Type;
#else
  using RawMask = typename X86VectorMaskRaw<word_bytes>::Type;
#endif
  using WordMask = X86Mask<sizeof(Element), word_lanes, RawMask>;
  using VecType = SingleOrArray<WordVec, word_count>;
  using MaskType = SingleOrArray<WordMask, word_count>;
};

template <Element T, nint_t N, typename Raw>
struct IsVectorRepresentation<X86Vector<T, N, Raw>> : std::true_type {};

template <nint_t ElementBytes, nint_t N, typename Raw>
struct IsMaskRepresentation<X86Mask<ElementBytes, N, Raw>> : std::true_type {};

template <Element T, nint_t N, typename Raw>
struct InferredTagTraits<X86Vector<T, N, Raw>> {
  using Type = FixedTag<T, N>;
};

template <Element T, nint_t N, typename Raw, nint_t Count>
struct InferredTagTraits<WordArray<X86Vector<T, N, Raw>, Count>> {
  using Type = FixedTag<T, N * Count>;
};

} // namespace vecops::vec::details

#endif // VECOPS_VEC_DETAILS_X86_TYPES_H
