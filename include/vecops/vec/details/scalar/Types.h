#ifndef VECOPS_VEC_DETAILS_SCALAR_TYPES_H
#define VECOPS_VEC_DETAILS_SCALAR_TYPES_H

/**
 * @file Types.h
 * @brief Scalar backend representation types: ScalarVector (16-byte-aligned
 * array), ScalarMask (std::bitset), ScalarLayout (fixed and scalable extent
 * layout), and RepresentationTraits<ScalarBackend, Tag>.
 *
 * The scalar backend uses a 16-byte "word" to emulate SIMD registers.
 * Multi-word Tags produce WordArray<ScalarVector<T, lanes>, count>.
 */

#include <cstddef>
#include <type_traits>

#include "vecops/vec/details/Representation.h"
#include "vecops/vec/details/Storage.h"

namespace vecops::vec::details {

inline constexpr nint_t scalar_word_bytes = 16;

template <Element T>
inline constexpr nint_t scalar_word_lanes =
    scalar_word_bytes / static_cast<nint_t>(sizeof(T));

template <VectorTag Tag>
struct ScalarLayout;

template <Element T, nint_t N>
struct ScalarLayout<VectorDescriptor<T, FixedExtent<N>>> {
  static constexpr nint_t logical_lanes = N;
  static constexpr nint_t word_lanes = scalar_word_lanes<T>;
  static constexpr nint_t word_count =
      (logical_lanes + word_lanes - 1) / word_lanes;
  static constexpr bool is_runtime_size_v = false;
  static constexpr bool is_subword_v = logical_lanes < word_lanes;
};

template <Element T, int ScalePower>
struct ScalarLayout<VectorDescriptor<T, ScalableExtent<ScalePower>>> {
  static constexpr nint_t word_lanes = scalar_word_lanes<T>;
  static constexpr nint_t logical_lanes = [] {
    if constexpr (ScalePower >= 0) {
      return word_lanes << ScalePower;
    } else {
      constexpr nint_t lanes = word_lanes >> (-ScalePower);
      static_assert(lanes > 0, "ScalableTag subword contains no lanes");
      return lanes;
    }
  }();
  static constexpr nint_t word_count =
      ScalePower > 0 ? (nint_t{1} << ScalePower) : 1;
  static constexpr bool is_runtime_size_v = false;
  static constexpr bool is_subword_v = ScalePower < 0;
};

template <VectorTag Tag>
struct RepresentationTraits<ScalarBackend, Tag> {
  using Element = ElementOf<Tag>;
  using Layout = ScalarLayout<Tag>;

  static constexpr nint_t logical_lanes = Layout::logical_lanes;
  static constexpr nint_t word_lanes = Layout::word_lanes;
  static constexpr nint_t word_count = Layout::word_count;
  static constexpr bool is_runtime_size_v = Layout::is_runtime_size_v;
  static constexpr bool is_subword_v = Layout::is_subword_v;

  using WordVec = ScalarVector<Element, word_lanes, scalar_word_bytes>;
  using WordMask = ScalarMask<sizeof(Element), word_lanes>;
  using VecType = SingleOrArray<WordVec, word_count>;
  using MaskType = SingleOrArray<WordMask, word_count>;
};

template <typename T, nint_t N, nint_t Alignment>
struct IsVectorRepresentation<ScalarVector<T, N, Alignment>> : std::true_type {};

template <nint_t ElementBytes, nint_t N>
struct IsMaskRepresentation<ScalarMask<ElementBytes, N>> : std::true_type {};

template <typename Word, nint_t Count>
struct IsVectorRepresentation<WordArray<Word, Count>>
    : IsVectorRepresentation<Word> {};

template <typename Word, nint_t Count>
struct IsMaskRepresentation<WordArray<Word, Count>>
    : IsMaskRepresentation<Word> {};

template <Element T, nint_t N, nint_t Alignment>
struct InferredTagTraits<ScalarVector<T, N, Alignment>> {
  using type = FixedTag<T, N>;
};

template <Element T, nint_t N, nint_t Alignment, nint_t Count>
struct InferredTagTraits<WordArray<ScalarVector<T, N, Alignment>, Count>> {
  using type = FixedTag<T, N * Count>;
};

} // namespace vecops::vec::details

#endif // VECOPS_VEC_DETAILS_SCALAR_TYPES_H
