//
// Copyright (c) vecops contributors.
//

/**
 * @file vecops/matmul/details/packing/amx/Format.h
 * @brief Block-format contract for AMX matmul packing (x86 AMX / VNNI).
 *
 * This header is the single authority for the AMX packed-block geometry: the
 * `Packing<Element, Side>` constants, and the `packed_layout()` /
 * `is_packed_layout()` pair that produce and statically recognize the packed
 * output layout.  The public umbrella `vecops/matmul/Packing.h` re-exposes
 * both through `matmul::packed_layout` / `matmul::is_packed_layout`; every
 * packer and consumer of AMX-packed data must take its constants from here.
 *
 * ## Block geometry and its hardware origin
 *
 * - An AMX TMUL tile is exactly 16 rows x 64 bytes (Intel AMX ISA), so one
 *   packed block covers `Panel = 16` spatial rows and
 *   `KTile = 64 / sizeof(Element)` K values: one block feeds one tile.
 * - AMX dot products consume K in 32-bit groups, VNNI style (four bytes per
 *   group for the integer tile instructions, two bf16/fp16 values per group
 *   for TDPBF16PS), so `KPack = 4 / sizeof(Element)` K values make up one
 *   32-bit group inside the packed stream.
 *
 * ## Packed memory layout — A side (4-D)
 *
 * Each 16-row spatial panel stores, per KTile-wide K block, the 16 source
 * rows contiguously (a row-major 16 x KTile mini-matrix, zero-padded on
 * tail panels / tail K tiles):
 *
 * @code
 * dim order: [ panel ][ k-tile ][ row (16) ][ k (KTile) ]  (compact)
 *
 * source A (M x K, K contiguous)      packed A, block (p, t):
 *                                     offset  0 ..  KTile-1 : row p*16+0, k = t*KTile ..
 *                                     offset KTile ..        : row p*16+1, same K range
 *                                     ...
 *                                     offset 15*KTile ..     : row p*16+15, same K range
 * @endcode
 *
 * ## Packed memory layout — B side (5-D)
 *
 * Each KPack-wide K group stores, for all 16 rows, one 32-bit group per row
 * (K values of one row packed together, rows stacked), so a loaded 16 x 16
 * dword transpose directly produces the panel:
 *
 * @code
 * dim order: [ panel ][ k-tile ][ k-group (KTile/KPack) ][ row (16) ][ k (KPack) ]
 *
 * packed B, group (p, t, g) — int8 example, KPack = 4:
 *   row p*16+0 : B[0][k0] B[0][k1] B[0][k2] B[0][k3]   <- one 32-bit group
 *   row p*16+1 : B[1][k0] B[1][k1] B[1][k2] B[1][k3]
 *   ...
 *   row p*16+15: ...
 *   where k0..k3 = t*KTile + g*KPack .. +KPack-1
 * @endcode
 *
 * ## Contrast with the SME format
 *
 * - AMX blocks are compile-time constants (16 rows x 64 bytes) because AMX
 *   tiles are fixed-function; SME blocks scale with the streaming vector
 *   length (see vecops/matmul/details/packing/sme/Format.h).
 * - AMX gives A and B different layouts (A keeps rows contiguous, B
 *   interleaves K into 32-bit groups); SME packs both operands with one
 *   shared layout.
 */

#ifndef VECOPS_MATMUL_DETAILS_AMX_PACKING_H
#define VECOPS_MATMUL_DETAILS_AMX_PACKING_H

#include <concepts>
#include <limits>
#include <type_traits>

#include "vecops/matmul/Atom.h"
#include "vecops/tensor/Layout.h"

namespace vecops::matmul::details::amx {

/// Empty tag identifying the AMX packed format.  It selects the AMX
/// `Backend` specializations (packing/Backend.h) through
/// `Packing::FormatType` and pairs the format with an implementation tag.
struct Format {};

/// Whether T is one of the four AMX TMUL element types: bf16/fp16 (tile dot
/// products for 16-bit floats) and s8/u8 (integer tile dot products).  There
/// is no fp32 AMX TMUL, hence no fp32 packing on this format.
template <typename T>
inline constexpr bool supported_element_v =
    std::same_as<T, bfloat16_t> || std::same_as<T, float16_t> ||
    std::same_as<T, int8_t> || std::same_as<T, uint8_t>;

/**
 * @brief Compile-time block geometry and layout rules for one AMX operand.
 *
 * See the file header for the memory-layout contract and its hardware
 * origin (16 x 64-byte AMX tiles, 32-bit VNNI K groups).
 *
 * @tparam ElementT  Packed element type; must satisfy supported_element_v.
 * @tparam Side      Operand::A or Operand::B; selects the A (4-D) or
 *                   B (5-D) packed layout and the packing vector axis.
 */
template <typename ElementT, Operand Side>
struct Packing {
  using Element = ElementT;
  using FormatType = Format;
  static_assert(supported_element_v<Element>);

  /// Input axis whose stride-1 runs the access-path packers vectorize
  /// along (0 = spatial, 1 = K).  A packs walk K runs; the B access path
  /// walks spatial columns and interleaves.  The B *direct* path is the
  /// exception: it loads K-contiguous rows and transposes instead
  /// (see packing/amx/Pack.h).
  static constexpr int VectorAxis = Side == Operand::A ? 1 : 0;
  /// Spatial rows per packed block: one AMX tile row count.
  static constexpr nint_t Panel = 16;
  /// K values per packed block: the AMX tile row width, 64 bytes.
  static constexpr nint_t KTile = 64 / sizeof(Element);
  /// K values per 32-bit VNNI group inside the packed stream.
  static constexpr nint_t KPack = 4 / sizeof(Element);
  static_assert(KTile > 0 && KPack > 0 && KTile % KPack == 0);

private:
  /// Guard against nint_t overflow in the padded extents produced below.
  VECOPS_INLINE static constexpr void validate_extents(
      nint_t spatial, nint_t k) {
    VECOPS_ASSERT(spatial >= 0 && k >= 0,
                  "matrix packing extents must be non-negative");
    const nint_t spatial_groups = ceil_div(spatial, Panel);
    const nint_t k_groups = ceil_div(k, KTile);
    constexpr nint_t Limit = std::numeric_limits<nint_t>::max();
    VECOPS_ASSERT(spatial_groups <= Limit / Panel &&
                      k_groups <= Limit / KTile,
                  "packed matrix extent overflows nint_t");
    const nint_t padded_spatial = spatial_groups * Panel;
    const nint_t padded_k = k_groups * KTile;
    VECOPS_ASSERT(padded_spatial == 0 ||
                      padded_k <= Limit / padded_spatial,
                  "packed matrix element count overflows nint_t");
  }

public:
  /**
   * @brief Compute the layout of the packed output for a rank-two input.
   *
   * The result is compact row-major over the block dims documented in the
   * file header: A is 4-D `[panel, k-tile, row(16), k(KTile)]`; B is 5-D
   * `[panel, k-tile, k-group(KTile/KPack), row(16), k(KPack)]`.  Tail
   * panels and tail K tiles are rounded up (ceil_div), so the packed
   * output is larger than the input; tail lanes are zero-padded by the
   * packers.
   *
   * @param input  Layout of the rank-two source (spatial x K).
   * @return       Layout of the packed output to allocate.
   */
  template <tensor::LayoutLike InputLayout>
  VECOPS_INLINE static constexpr auto packed_layout(
      const InputLayout& input) {
    static_assert(InputLayout::Ndim == 2,
                  "matrix packing accepts a rank-two input layout");
    const auto spatial = tensor::size<0>(input);
    const auto k = tensor::size<1>(input);
    validate_extents(static_cast<nint_t>(spatial),
                     static_cast<nint_t>(k));
    if constexpr (Side == Operand::A) {
      // dims: [ spatial panel ][ k tile ][ row within panel ][ k within tile ]
      return tensor::make_layout(tensor::make_shape(
          ceil_div(spatial, meta::cint<Panel>),
          ceil_div(k, meta::cint<KTile>),
          meta::cint<Panel>, meta::cint<KTile>));
    } else {
      // dims: [ spatial panel ][ k tile ][ k group within tile ]
      //       [ row within panel ][ k within group ]
      return tensor::make_layout(tensor::make_shape(
          ceil_div(spatial, meta::cint<Panel>),
          ceil_div(k, meta::cint<KTile>),
          meta::cint<KTile / KPack>,
          meta::cint<Panel>, meta::cint<KPack>));
    }
  }

  /**
   * @brief Statically recognize a layout produced by (or equivalent to)
   *        packed_layout().
   *
   * A layout "is packed" when its rank matches and every *inner* size and
   * stride is exactly the compile-time compact constant of the format
   * (e.g. A strides `Panel*KTile, KTile, 1` on dims 1..3; B strides
   * `Panel*KTile, Panel*KPack, KPack, 1` on dims 1..4).  The two outermost
   * sizes (panel / k-tile counts) may stay dynamic, and the outermost
   * stride is deliberately unchecked — it is the total element count and
   * carries no format information.
   */
  template <tensor::LayoutLike Layout>
  static consteval bool is_packed_layout() {
    if constexpr (Side == Operand::A) {
      if constexpr (Layout::Ndim != 4) return false;
      else return
          std::same_as<tensor::size_type_t<2, Layout>, meta::Const<Panel>> &&
          std::same_as<tensor::size_type_t<3, Layout>, meta::Const<KTile>> &&
          std::same_as<tensor::stride_type_t<1, Layout>,
                       meta::Const<Panel * KTile>> &&
          std::same_as<tensor::stride_type_t<2, Layout>,
                       meta::Const<KTile>> &&
          std::same_as<tensor::stride_type_t<3, Layout>, meta::Const<1>>;
    } else {
      if constexpr (Layout::Ndim != 5) return false;
      else return
          std::same_as<tensor::size_type_t<2, Layout>,
                       meta::Const<KTile / KPack>> &&
          std::same_as<tensor::size_type_t<3, Layout>, meta::Const<Panel>> &&
          std::same_as<tensor::size_type_t<4, Layout>, meta::Const<KPack>> &&
          std::same_as<tensor::stride_type_t<1, Layout>,
                       meta::Const<Panel * KTile>> &&
          std::same_as<tensor::stride_type_t<2, Layout>,
                       meta::Const<Panel * KPack>> &&
          std::same_as<tensor::stride_type_t<3, Layout>,
                       meta::Const<KPack>> &&
          std::same_as<tensor::stride_type_t<4, Layout>, meta::Const<1>>;
    }
  }
};

} // namespace vecops::matmul::details::amx

#endif // VECOPS_MATMUL_DETAILS_AMX_PACKING_H
