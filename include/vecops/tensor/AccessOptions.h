#ifndef VECOPS_TENSOR_ACCESS_OPTIONS_H
#define VECOPS_TENSOR_ACCESS_OPTIONS_H

#include <array>
#include <cstddef>
#include <type_traits>

#include "vecops/CoreTypes.h"
#include "vecops/vec/Options.h"

/**
 * @file AccessOptions.h
 * @brief Coordinates, axes, lane mappings, and operand facts.
 *
 * These types form the vocabulary shared by coordinate DataAccess and its
 * cursors. They describe logical tensor operations; they do not prescribe a
 * particular machine instruction. For example, `vec::strided(2)` means every
 * lane advances by two elements on the selected logical tensor axis. Depending
 * on the Layout stride, DataAccess may lower that request to a contiguous load,
 * a strided load, or gather/scatter.
 *
 * ## Coordinates and axes
 *
 * @code
 * auto p = tensor::coord(row, 0);       // Coord<2>
 * x.load(tag, p);                       // vector axis defaults to rank - 1
 * x.load(tag, p, tensor::axis<0>);      // lanes advance along rows
 * @endcode
 *
 * `Axis`, `TraversalAxis`, and `VectorAxis` are compile-time tags. The latter
 * two are semantic aliases used by `ProjectCursor` to make it difficult to
 * accidentally swap the axis being advanced with the axis being loaded.
 *
 * ## Pitfalls
 *
 * - Coordinates are signed `nint_t` values and are not bounds-checked by the
 *   type. DataAccess performs debug assertions for active lanes.
 * - `whole_block` requires an affine footprint; indexed accesses must use
 *   `first_line` or explicit points.
 * - `BaseAlignment<N>` is an externally asserted fact about the Tensor's base
 *   pointer. It is checked when the Spec is built in assertion-enabled builds;
 *   it does not align a pointer or allocate storage.
 */
namespace vecops::tensor {

/** Fixed-rank logical coordinate used by Tensor DataAccess. */
template <std::size_t Rank>
using Coord = std::array<nint_t, Rank>;

/**
 * @brief Construct a coordinate while deducing its rank.
 * @return `Coord<sizeof...(Indices)>` containing the converted indices.
 */
template <typename... Indices>
VECOPS_ALWAYS_INLINE constexpr auto coord(Indices... indices) {
  static_assert((std::is_convertible_v<Indices, nint_t> && ...));
  return Coord<sizeof...(Indices)>{static_cast<nint_t>(indices)...};
}

/** @brief Compile-time tensor axis tag. */
template <int Dim>
struct Axis {
  static constexpr int value = Dim;
};

/** @brief Axis tag value used as a normal function argument. */
template <int Dim>
inline constexpr Axis<Dim> axis{};

/** @brief Semantic axis tag for the dimension advanced by ProjectCursor. */
template <int Dim>
using TraversalAxis = Axis<Dim>;

/** @brief Semantic axis tag for the dimension represented by vector lanes. */
template <int Dim>
using VectorAxis = Axis<Dim>;

template <int Dim>
inline constexpr TraversalAxis<Dim> traversal_axis{};

template <int Dim>
inline constexpr VectorAxis<Dim> vector_axis{};

/**
 * @brief Map coordinates of a sliced view back to the original Tensor rank.
 *
 * `fixed` stores coordinates of dimensions removed by slicing, and
 * `local_to_original[d]` identifies the original dimension represented by
 * local dimension `d`. DataAccess uses this map only for TransformContext;
 * physical addressing continues to use the sliced Tensor's Layout.
 */
template <std::size_t OriginalRank, std::size_t LocalRank>
struct CoordinateProjection {
  Coord<OriginalRank> fixed{};
  std::array<int, LocalRank> local_to_original{};

  VECOPS_ALWAYS_INLINE constexpr Coord<OriginalRank> project(
      const Coord<LocalRank>& local) const {
    auto result = fixed;
    for (std::size_t d = 0; d < LocalRank; ++d) {
      result[static_cast<std::size_t>(local_to_original[d])] = local[d];
    }
    return result;
  }

  template <int Dim>
  VECOPS_ALWAYS_INLINE constexpr auto sliced(nint_t index) const {
    static_assert(0 <= Dim && Dim < static_cast<int>(LocalRank));
    CoordinateProjection<OriginalRank, LocalRank - 1> result{};
    result.fixed = fixed;
    result.fixed[static_cast<std::size_t>(local_to_original[Dim])] = index;
    for (std::size_t src = 0, dst = 0; src < LocalRank; ++src) {
      if (src == static_cast<std::size_t>(Dim)) continue;
      result.local_to_original[dst++] = local_to_original[src];
    }
    return result;
  }

  /** Map a transposed local view back to the same original coordinates. */
  template <int I, int J>
  VECOPS_ALWAYS_INLINE constexpr auto transposed() const {
    static_assert(0 <= I && I < static_cast<int>(LocalRank));
    static_assert(0 <= J && J < static_cast<int>(LocalRank));
    auto result = *this;
    const int temporary = result.local_to_original[I];
    result.local_to_original[I] = result.local_to_original[J];
    result.local_to_original[J] = temporary;
    return result;
  }
};

/** @brief Create an identity coordinate projection for an unsliced rank. */
template <std::size_t Rank>
VECOPS_ALWAYS_INLINE consteval auto identity_projection() {
  CoordinateProjection<Rank, Rank> result{};
  for (std::size_t d = 0; d < Rank; ++d) {
    result.local_to_original[d] = static_cast<int>(d);
  }
  return result;
}

/** @brief Lane `i` addresses logical axis offset `i`. */
struct ContiguousLaneMapping {
  VECOPS_ALWAYS_INLINE constexpr nint_t offset(nint_t lane) const {
    return lane;
  }
};

/** @brief Lane `i` addresses logical axis offset `i * stride`. */
struct AffineLaneMapping {
  nint_t stride = 1;

  VECOPS_ALWAYS_INLINE constexpr nint_t offset(nint_t lane) const {
    return lane * stride;
  }
};


/**
 * @brief User-asserted alignment of an operand's base pointer in bytes.
 *
 * This is a Spec fact, not a memory-operation policy. Pass
 * `assume_aligned<N>` to `input()` or `output()`; kernels remain responsible
 * for selecting aligned versus unaligned memory options.
 */
template <nint_t Alignment>
struct BaseAlignment {
  static_assert(
      Alignment > 0 && (Alignment & (Alignment - 1)) == 0,
      "base alignment must be a positive power of two");
  static constexpr nint_t value = Alignment;
};

/** @brief Value object for declaring a base-pointer alignment fact. */
template <nint_t Alignment>
inline constexpr BaseAlignment<Alignment> assume_aligned{};

} // namespace vecops::tensor

#endif // VECOPS_TENSOR_ACCESS_OPTIONS_H
