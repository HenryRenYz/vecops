//
// Copyright (c) vecops contributors.
//

/**
 * @file vecops/matmul/details/packing/sme/Format.h
 * @brief Block-format contract for SME matmul packing (Arm SME outer
 *        products).
 *
 * This header is the single authority for the SME packed-block geometry: the
 * `Packing<Element, Side>` constants, and the `packed_layout()` /
 * `is_packed_layout()` pair that produce and statically recognize the packed
 * output layout.  The public umbrella `vecops/matmul/Packing.h` re-exposes
 * both through `matmul::packed_layout` / `matmul::is_packed_layout`; every
 * packer and consumer of SME-packed data must take its constants from here.
 *
 * ## Block geometry and its hardware origin
 *
 * - Arm SME outer products (FMOPA/SMOPA) consume K in 32-bit groups for
 *   element types up to 32 bits: one fp32 per group, two fp16/bf16, four
 *   bytes — hence `KPack = 4 / sizeof(Element)`.  The F64 variant consumes
 *   one element per step, hence KPack = 1 there.
 * - The spatial panel is `2 x` the FP32 lane count of the streaming vector
 *   length (SVL): the packer uses one *pair* of ZA tiles as its transpose
 *   working set (see packing/sme/Pack.h), and one ZA tile holds SVL/4
 *   fp32-sized rows.  For fp64 the panel is 2 x the fp64 lane count, i.e.
 *   the same two-tile rule measured in fp64 rows.
 * - Unlike AMX, the panel is a *runtime* value on variable-SVL targets (it
 *   is only a compile-time constant when the streaming width is fixed),
 *   and it is the same for A and B: both operands share one packed layout.
 *
 * ## Packed memory layout (4-D, identical for A and B)
 *
 * @code
 * dim order: [ panel ][ k-group ][ row (panel) ][ k (KPack) ]  (compact)
 *
 * source (spatial x K, K contiguous)   packed, group (p, g) — fp16, KPack = 2:
 *   row p*panel+0 : in[0][g*2] in[0][g*2+1]   <- one 32-bit group
 *   row p*panel+1 : in[1][g*2] in[1][g*2+1]
 *   ...
 * whole group = panel * KPack contiguous elements; groups follow in K
 * order, panels in spatial order; tail rows / tail K groups zero-padded.
 * @endcode
 *
 * ## Contrast with the AMX format (packing/amx/Format.h)
 *
 * - AMX blocks are fixed 16 rows x 64 bytes (AMX tile dimensions); SME
 *   blocks scale with SVL.
 * - AMX packs A and B differently (A: contiguous rows; B: dword groups
 *   inside 16-row panels); SME interleaves both operands the same way,
 *   with a panel-sized row dimension instead of a constant 16.
 * - SME `is_packed_layout()` matches the panel dim against the runtime
 *   panel type (`decltype(Packing::panel())`), not a `Const` value.
 */

#ifndef VECOPS_MATMUL_DETAILS_SME_PACKING_H
#define VECOPS_MATMUL_DETAILS_SME_PACKING_H

#include <concepts>
#include <limits>

#include "vecops/matmul/Atom.h"
#include "vecops/tensor/Layout.h"
#include "vecops/vec/details/sme/State.h"

namespace vecops::matmul::details::sme {

/// Empty tag identifying the SME packed format.  It selects the SME
/// `Backend` specializations (packing/sme/Backend.h) through
/// `Packing::FormatType` and pairs the format with an implementation tag.
struct Format {};

/// Whether T is one of the six SME matmul element types: fp64/fp32/bf16/fp16
/// (FMOPA) and s8/u8 (SMOPA).
template <typename T>
inline constexpr bool supported_element_v =
    std::same_as<T, float32_t> || std::same_as<T, bfloat16_t> ||
    std::same_as<T, float16_t> || std::same_as<T, int8_t> ||
    std::same_as<T, uint8_t> || std::same_as<T, float64_t>;

/// Panel height shared by every element type up to 32 bits: two ZA tiles
/// worth of rows measured in fp32 lanes (`2 * SVL/4`).  fp64 uses its own
/// panel (see Packing::panel()).
VECOPS_INLINE auto panel_lanes() {
  return meta::cint<2> *
      vec::details::sme::streaming_lanes_value<float32_t>();
}

/**
 * @brief Compile-time block geometry and layout rules for one SME operand.
 *
 * See the file header for the memory-layout contract and its hardware
 * origin (ZA tile pairs, 32-bit outer-product K groups).  A and B share the
 * same layout; only the packing *direction* through ZA differs by operand.
 *
 * @tparam ElementT  Packed element type; must satisfy supported_element_v.
 * @tparam Side      Operand::A or Operand::B (affects no layout constant
 *                   here; kept for symmetry with the AMX Packing).
 */
template <typename ElementT, Operand Side>
struct Packing {
  using Element = ElementT;
  using FormatType = Format;
  static_assert(supported_element_v<Element>);

  /// Input axis whose runs the packers vectorize along (0 = spatial).
  /// Both operands traverse spatially: the generic path loads spatial
  /// columns (pack_interleaved_panels), and the native SME kernel ingests
  /// K-contiguous rows and turns them into spatial columns through the ZA
  /// transpose.
  static constexpr int VectorAxis = 0;
  /// K values per 32-bit outer-product group (fp64: one element per step).
  static constexpr nint_t KPack = sizeof(Element) > 4 ? 1 : 4 / sizeof(Element);
  static_assert(KPack > 0);

private:
  /// Guard against nint_t overflow in the padded extents produced below.
  VECOPS_INLINE static constexpr void validate_extents(
      nint_t spatial, nint_t k, nint_t panel) {
    VECOPS_ASSERT(spatial >= 0 && k >= 0,
                  "matrix packing extents must be non-negative");
    VECOPS_ASSERT(panel > 0,
                  "matrix packing blocks must be positive");
    const nint_t spatial_groups = ceil_div(spatial, panel);
    const nint_t k_groups = ceil_div(k, KPack);
    constexpr nint_t Limit = std::numeric_limits<nint_t>::max();
    VECOPS_ASSERT(spatial_groups <= Limit / panel &&
                      k_groups <= Limit / KPack,
                  "packed matrix extent overflows nint_t");
    const nint_t padded_spatial = spatial_groups * panel;
    const nint_t padded_k = k_groups * KPack;
    VECOPS_ASSERT(padded_spatial == 0 ||
                      padded_k <= Limit / padded_spatial,
                  "packed matrix element count overflows nint_t");
  }

public:
  /// Panel height in rows: two ZA tiles measured in this element's rows.
  /// Up to 32 bits that is `2 * SVL/4` fp32-sized rows; fp64 instead fills
  /// two tiles of fp64 rows (`2 * SVL/8`), keeping the two-tile rule.
  VECOPS_INLINE static auto panel() {
    if constexpr (sizeof(Element) == 8)
      return meta::cint<2> *
          vec::details::sme::streaming_lanes_value<float64_t>();
    else
      return panel_lanes();
  }

  /**
   * @brief Compute the layout of the packed output for a rank-two input.
   *
   * The result is compact row-major over the 4-D block documented in the
   * file header: `[panel, k-group, row(panel), k(KPack)]`, identical for
   * the A and B operands.  Tail panels and tail K groups are rounded up
   * (ceil_div), so the packed output is larger than the input; tail lanes
   * are zero-padded by the packers.
   *
   * @param input  Layout of the rank-two source (spatial x K).
   * @return       Layout of the packed output to allocate.
   */
  template <tensor::LayoutLike InputLayout>
  VECOPS_INLINE static auto packed_layout(const InputLayout& input) {
    static_assert(InputLayout::Ndim == 2,
                  "matrix packing accepts a rank-two input layout");
    using Spatial = tensor::size_type_t<0, InputLayout>;
    using K = tensor::size_type_t<1, InputLayout>;
    const Spatial spatial{tensor::size<0>(input)};
    const K k{tensor::size<1>(input)};
    const auto panel = Packing::panel();
    validate_extents(static_cast<nint_t>(spatial),
                     static_cast<nint_t>(k),
                     static_cast<nint_t>(panel));
    // dims: [ spatial panel ][ k group ][ row within panel ][ k within group ]
    return tensor::make_layout(tensor::make_shape(
        ceil_div(spatial, panel),
        ceil_div(k, meta::cint<KPack>),
        panel, meta::cint<KPack>));
  }

  /**
   * @brief Statically recognize a layout produced by (or equivalent to)
   *        packed_layout().
   *
   * A layout "is packed" when it is rank 4 with the compact inner
   * structure: innermost size `Const<KPack>` with strides
   * `KPack, 1` on dims 2..3, and — unlike the AMX format — the panel size
   * on dim 2 must match the *panel type* `decltype(Packing::panel())`,
   * which folds to a constant only when the streaming width is fixed at
   * compile time.  The two outermost sizes and the outermost stride are
   * unconstrained (dynamic panel/group counts; dim-0 stride is just the
   * total element count).
   */
  template <tensor::LayoutLike Layout>
  static consteval bool is_packed_layout() {
    if constexpr (Layout::Ndim != 4) return false;
    else return
        std::same_as<tensor::size_type_t<2, Layout>,
                     decltype(Packing::panel())> &&
        meta::range_within_v<
            tensor::size_type_t<3, Layout>, KPack, KPack> &&
        meta::range_within_v<
            tensor::stride_type_t<2, Layout>, KPack, KPack> &&
        meta::range_within_v<
            tensor::stride_type_t<3, Layout>, 1, 1>;
  }
};

} // namespace vecops::matmul::details::sme

#endif // VECOPS_MATMUL_DETAILS_SME_PACKING_H
