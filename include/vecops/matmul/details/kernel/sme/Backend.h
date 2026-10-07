// SPDX-FileCopyrightText: 2026 HenryRenYz and vecops contributors
// SPDX-License-Identifier: MIT
//
//

#ifndef VECOPS_MATMUL_DETAILS_SME_BACKEND_H
#define VECOPS_MATMUL_DETAILS_SME_BACKEND_H

/**
 * @file vecops/matmul/details/kernel/sme/Backend.h
 * @brief SME matmul backend: ZA outer-product tile kernels plus a ladder of
 *        ordinary-SVE fast-path leaves.
 *
 * ## Dispatch structure (DispatchOwner)
 *
 * `Backend<SME>::run` first resolves a `DispatchOwner` at compile time
 * (`select_automatic_dispatch_owner`). The owners form two classes:
 *
 * - `General` -- the catch-all ZA microkernel. It enters a Streaming+ZA
 *   region and traverses the problem with the Tile2D layer over ZA tiles
 *   (`microkernel` below), consuming either packed or direct operands.
 * - the others -- ordinary-SVE leaves that never touch ZA: mixed-sign
 *   int8 skinny (row/column), raw skinny (row/column), fused
 *   (transform-carrying) skinny (row/column), the fused runtime-quant INT8
 *   GEMV, and the packed-input dot kernels (primary/tiny shapes). They are
 *   selectable only when the scope does not already own StreamingZA: their
 *   bodies are compiled against the ordinary SVE vector length.
 *
 * When the Meta contracts cannot decide a leaf's applicability
 * (`Applicability::runtime`), run() consults the measured thresholds in
 * RuntimeDispatch.h with the concrete m/n/k (see the applicability probes
 * next to `select_automatic_dispatch_owner`).
 *
 * The leaves map onto the `kernel_family` tags (General, SmallVector,
 * RuntimeQuantInt8, PackedDot) via `dispatch_owner_in_family_v`; a
 * `required` family selection that resolves to a different owner fails a
 * static_assert instead of silently re-routing.
 *
 * ## StreamingZARegion lifecycle
 *
 * All ZA state is owned by `run`/`run_configured`: one
 * `scope.with_resources(StreamingZARegion{})` wraps the entire Tile2D
 * traversal, so a whole problem -- not one tile -- forms a single
 * SMSTART..SMSTOP interval. Inside the region:
 *
 * - operands that are already packed are forwarded untouched: packed access
 *   is statically direct and untransformed and consumed through raw
 *   pointers, so the region's resource set cannot change how they load;
 * - direct (unpacked) operands and the output are *rebound* to the region's
 *   active resource set, so their memory ops run with streaming-appropriate
 *   resources; the output session is explicitly committed before the region
 *   closes (`active_c_output.commit()`).
 *
 * ## FastPacked plan
 *
 * `dispatch_plan` picks a `KernelPlan<FastPacked, PrefetchLargeWorkingSet>`
 * from the K-group count: BF16 problems whose N contract is at most 128 and
 * whose K contract is in [32, 64) use a tuned 16-group crossover, while
 * wider/longer BF16 and the other atoms retain 32 groups; an unconstrained K
 * also selects the fast plan. Above that crossover
 * the microkernel switches from per-group operand loads
 * (`load_operand`) to the direct packed-pointer loop
 * (`compute_packed_groups`), optionally with L2 look-ahead prefetching for
 * large BF16 working sets (`large_packed_prefetch_v`).
 */

#include <cstdint>
#include <limits>
#include <type_traits>
#include <utility>

#include "vecops/Assertion.h"
#include "vecops/execution/details/arm/Resources.h"
#include "vecops/matmul/Packing.h"
#include "vecops/matmul/details/kernel/RuntimeDispatch.h"
#include "vecops/matmul/details/kernel/sme/Atoms.h"
#include "vecops/kernel/Tile2D.h"
#include "vecops/matmul/details/kernel/TileScheduler.h"
#include "vecops/matmul/details/kernel/sme/RuntimeQuantInt8.h"
#include "vecops/matmul/details/packing/generic/Pack.h"
#include "vecops/tensor/DataAccess.h"
#include "vecops/util/Math.h"
#include "vecops/vec/details/sme/ZA.h"

namespace vecops::kernel::matmul_details::sme {

namespace generic = matmul_pack_details::generic;
namespace tile = ::vecops::kernel::loop;

/// Detects the zero-value C-input transform (`zeros_transform`): a C input
/// behind it needs no ZA seeding at all (the tile stays zeroed).
template <typename T>
struct IsZeroTransform : std::false_type {};

template <typename Out, typename In>
struct IsZeroTransform<tensor::ZeroVecTransform<Out, In>> : std::true_type {};

// Access-object introspection: spec and layout aliases shared by the
// candidate predicates and the operand loaders below.
template <typename Access>
using SpecOf = std::remove_cvref_t<decltype(
    std::declval<const std::remove_cvref_t<Access>&>().spec())>;

template <typename Access>
using InputLayoutOf = typename SpecOf<Access>::InputLayout;

template <typename Access>
using OutputLayoutOf = typename SpecOf<Access>::OutputLayout;

/// Whether an access object carries the atom's packed layout for one side.
template <::vecops::matmul::Atom Atom, ::vecops::matmul::Operand Side, typename Access>
inline constexpr bool is_packed_access_v =
    ::vecops::matmul::is_packed_layout<Atom, Side, InputLayoutOf<Access>>();

/// Raw-pointer, rank-2 input whose innermost (K) stride is compile-time 1 --
/// the shape the direct fast loads below are written against.
template <typename Access>
inline constexpr bool direct_row_major_input_v =
  generic::RawDirectAccess<Access> && Access::Rank == 2 &&
  meta::range_within_v<
      tensor::stride_type_t<1, InputLayoutOf<Access>>, 1, 1>;

template <typename Access>
inline constexpr bool row_contiguous_input_v =
  Access::Rank == 2 && meta::range_within_v<
      tensor::stride_type_t<1, InputLayoutOf<Access>>, 1, 1>;

/// Output counterpart of direct_row_major_input_v.
template <typename Access>
inline constexpr bool direct_row_major_output_v =
  generic::RawDirectAccess<Access> && Access::Rank == 2 &&
  meta::range_within_v<
      tensor::stride_type_t<1, OutputLayoutOf<Access>>, 1, 1>;

template <typename Access>
inline constexpr bool row_contiguous_output_v =
  Access::Rank == 2 && meta::range_within_v<
      tensor::stride_type_t<1, OutputLayoutOf<Access>>, 1, 1>;

/// Rank-two input whose leading (spatial) axis is contiguous: a
/// transposed operand as fed to orientation-swapped problems.
template <typename Access>
inline constexpr bool column_contiguous_input_v =
    Access::Rank == 2 && meta::range_within_v<
        tensor::stride_type_t<0, InputLayoutOf<Access>>, 1, 1>;

/// Rank-two output whose leading (spatial) axis is contiguous (transposed
/// C); the store shape produced by orientation-swapped problems.
template <typename Access>
inline constexpr bool column_contiguous_output_v =
  Access::Rank == 2 && meta::range_within_v<
      tensor::stride_type_t<0, OutputLayoutOf<Access>>, 1, 1>;

/**
 * SME's backend-local automatic transform/store policy.  BiSheng 5.1 on the
 * tested 64-byte target currently favors narrow stores; explicit transform
 * wrappers can still force a wide transform or split-transform coalescing.
 */
template <typename Access>
inline constexpr tensor::TransformStoreMode transform_store_mode_v = [] {
  constexpr auto requested = tensor::transform_store_mode_v<typename Access::Transform>;
  if constexpr (requested == tensor::TransformStoreMode::automatic)
    return tensor::TransformStoreMode::narrow;
  else
    return requested;
}();

template <typename Access>
inline constexpr bool transform_store_groups_tiles_v =
  transform_store_mode_v<Access> == tensor::TransformStoreMode::wide_transform ||
  transform_store_mode_v<Access> == tensor::TransformStoreMode::coalesced;

/// Raw-pointer, rank-two, column-contiguous output: eligible for the
/// direct vertical ZA store path (write/read_column below).
template <typename Access>
inline constexpr bool direct_column_major_output_v =
    generic::RawDirectAccess<Access> && column_contiguous_output_v<Access>;

/**
 * @brief Tile2D kernel provider: search-space limits and per-family score.
 *
 * `power()` scores one tile family (A x B atomic blocks) for the Tile2D
 * catalog search. The two-stage formula both rewards throughput (A*B atomic
 * outer products per case) and prefers square-ish shapes (the `imbalance`
 * penalty); the second tier (ZA64/expanded-catalog families) starts at a
 * 300 baseline so it scores below every compact-catalog family and only
 * ExactCover -- which detects their presence directly -- ever picks them.
 */
template <bool ExpandedCatalog>
struct KernelProvider {
  static constexpr bool four_regions_exact_constraints = true;
  static constexpr tile::Tile2DExactGridMode exact_grid_mode = ExpandedCatalog
      ? tile::Tile2DExactGridMode::exact
      : tile::Tile2DExactGridMode::runtime;
  static constexpr nint_t exact_meta_block_limit = ExpandedCatalog
      ? std::numeric_limits<nint_t>::max() : nint_t{8};

  /**
   * @brief Score of the A x B family, or -1 when the family is excluded.
   *
   * Tier 1 (A*B <= 4): `100*A*B + 4*(A+B) - 8*|A-B|` -- throughput dominates
   * (100 per atomic product), with a small shape-balance term.
   * Tier 2 (expanded catalog, A*B <= 8): `300 + 4*A*B + (A+B) - 2*|A-B|` --
   * see the class comment for why it must stay below tier 1.
   */
  template <int A, int B, tile::Tile2DMaskMode, tile::Tile2DMaskMode>
  static consteval int power() {
    if constexpr (A * B <= 4) {
      constexpr int imbalance = A > B ? A - B : B - A;
      return 100 * A * B + 4 * (A + B) - 8 * imbalance;
    } else if constexpr (ExpandedCatalog && A * B <= 8) {
      // Only ZA64 paths expose these families to Tile2D. Their lower score
      // keeps the compact catalog family as the default for other policies;
      // ExactCover detects their presence directly from the catalog.
      constexpr int imbalance = A > B ? A - B : B - A;
      return 300 + 4 * A * B + (A + B) - 2 * imbalance;
    } else {
      return -1;
    }
  }
};

/// Tile2D generated catalog over KernelProvider; the expanded variant also
/// searches the larger (up to 8 meta blocks) ZA64 families.
template <bool ExpandedCatalog>
using Catalog = tile::Tile2DGeneratedCatalog<
    KernelProvider<ExpandedCatalog>,
    tile::Tile2DSearchSpace<4, 4, ExpandedCatalog ? 8 : 4>>;

/**
 * @brief Whether to search the expanded ZA64 catalog for this problem.
 *
 * True only for the F64 atom with *both* operands unpacked: the fp64 ZA
 * tiles are SVL/8-sized, so the compact (4x4-block) catalog wastes most of
 * each tile and the wider meta blocks pay for themselves. Packed F64
 * inputs keep the compact catalog.
 */
template <::vecops::matmul::Atom Atom, typename A, typename B>
inline constexpr bool use_expanded_catalog_v = [] {
#if defined(HAS_SME_F64F64)
  return std::same_as<Atom, ::vecops::matmul::SME_F64F64> &&
      !(is_packed_access_v<Atom, ::vecops::matmul::Operand::A, A> &&
        is_packed_access_v<Atom, ::vecops::matmul::Operand::B, B>);
#else
  return false;
#endif
}();

/// Compile-time microkernel plan handed from `dispatch_plan` to `run_case`:
/// `value` selects the fast packed-pointer loop (see the file header,
/// FastPacked plan), `prefetch_large_working_set` enables look-ahead L2
/// prefetching for large packed working sets, and `share_tile_body` limits
/// out-of-line sharing to the shallow-K specialization that otherwise causes
/// pathological compiler expansion.
template <bool FastPacked, bool PrefetchLargeWorkingSet,
          bool ShareTileBody = false>
struct KernelPlan : std::bool_constant<FastPacked> {
  static constexpr bool prefetch_large_working_set =
      PrefetchLargeWorkingSet;
  static constexpr bool share_tile_body = ShareTileBody;
};

/**
 * @brief Whether a packed BF16 problem's working set is large enough that
 *        the packed-pointer loop should prefetch.
 *
 * Heuristic: estimate the bytes touched per K element
 * (`M*sizeof(TA) + N*sizeof(TB)`, packed blocks are consumed row-wise) and
 * require the total footprint to reach ~2 MiB (MinPackedBytes) -- the size
 * beyond which plain demand loads no longer keep L2 warm on their own.
 * Computed from compile-time lower bounds, so it costs nothing at run time.
 */
template <::vecops::matmul::Atom Atom,
          meta::ValueType M, meta::ValueType N, meta::ValueType K>
inline constexpr bool large_packed_prefetch_v = [] {
#if !defined(VECOPS_DISABLE_SME_LARGE_PACKED_PREFETCH)
  using MV = std::remove_cvref_t<M>;
  using NV = std::remove_cvref_t<N>;
  using KV = std::remove_cvref_t<K>;
  if constexpr (std::same_as<Atom, ::vecops::matmul::SME_BF16F32> &&
                meta::lower_bound_at_least_v<MV, 1> &&
                meta::lower_bound_at_least_v<NV, 1> &&
                meta::lower_bound_at_least_v<KV, 1>) {
    constexpr nint_t LogicalM = meta::lower_bound_v<MV>;
    constexpr nint_t LogicalN = meta::lower_bound_v<NV>;
    constexpr nint_t LogicalK = meta::lower_bound_v<KV>;
    constexpr uint64_t MinPackedBytes = uint64_t{2} * 1024 * 1024;
    constexpr uint64_t BytesPerK =
        static_cast<uint64_t>(LogicalM) * sizeof(typename Atom::TA) +
        static_cast<uint64_t>(LogicalN) * sizeof(typename Atom::TB);
    constexpr uint64_t MinBytesPerK =
        (MinPackedBytes + static_cast<uint64_t>(LogicalK) - 1) /
        static_cast<uint64_t>(LogicalK);
    return BytesPerK >= MinBytesPerK;
  }
#endif
  return false;
}();

/// Whether Meta bounds can already bound the number of M_R/N_R tile blocks
/// on at least one axis -- the precondition for constraint pruning below.
template <::vecops::matmul::Atom Atom, meta::ValueType M, meta::ValueType N>
inline constexpr bool has_bounded_tile_axis_v =
    tile::tile2d_details::has_max_block_count_v<
        M, decltype(Atom::M_R)> ||
    tile::tile2d_details::has_max_block_count_v<
        N, decltype(Atom::N_R)>;

/**
 * @brief Whether the Automatic policy should resolve to the statically
 *        constraint-pruned FourRegions plan instead of the runtime
 *        ExactCover search.
 *
 * Requires a provably bounded tile axis, then keeps the proven winners on
 * the compact runtime path (see the tuning comment inside).
 */
template <::vecops::matmul::Atom Atom,
          meta::ValueType M, meta::ValueType N, meta::ValueType K,
          typename A, typename B>
inline constexpr bool prefer_constraint_pruning_v = [] {
  if constexpr (!has_bounded_tile_axis_v<Atom, M, N>) {
    return false;
  } else if constexpr (
      is_packed_access_v<Atom, ::vecops::matmul::Operand::A, A> &&
      is_packed_access_v<Atom, ::vecops::matmul::Operand::B, B>) {
    constexpr bool FixedSquare =
        meta::is_singleton_v<M> &&
        meta::is_singleton_v<N> &&
        meta::singleton_value_v<M> ==
            meta::singleton_value_v<N>;
    constexpr nint_t KP = ::vecops::matmul::packing_t<
        Atom, ::vecops::matmul::Operand::A>::KPack;
    constexpr bool ShortK = meta::has_upper_bound_v<K> &&
        meta::upper_bound_v<K> < 32 * KP;
    // Very short packed kernels are dominated by entry/code-layout effects,
    // while fixed square packed kernels currently receive better register
    // allocation from the compact runtime ExactCover loop. Preserve those proven
    // winners; constrained raw, rectangular, and long-K packed problems use
    // the statically pruned plan.
    return !FixedSquare && !ShortK;
  } else {
    return true;
  }
}();

/// log2 of a KPack value (4 -> 2, 2 -> 1, else 0); used to shrink a scalar
/// tag so its vector holds whole K groups per lane.
consteval int log2_kpack(nint_t kpack) {
  return kpack == 4 ? 2 : (kpack == 2 ? 1 : 0);
}

/// Rows (== columns) of one ZA accumulator tile for this atom:
/// SVL / sizeof(TAcc), i.e. the atom's M_R.
template <::vecops::matmul::Atom Atom>
VECOPS_ALWAYS_INLINE nint_t accumulator_lanes() {
  return static_cast<nint_t>(Atom::M_R);
}

/**
 * @brief Row-invariant facts of one direct operand, computed once per
 *        microkernel.
 *
 * `row_bytes` is the byte distance between consecutive rows; whether it
 * fits an int32 decides `gather_offsets_fit`: the gather fast path in
 * `load_operand` issues one strided u32 gather per column block with the
 * row stride as the (signed 32-bit) gather offset scale, so a stride
 * outside int32 range would silently corrupt addressing -- this pre-check
 * rejects that case up front instead.
 */
template <::vecops::matmul::Atom Atom, ::vecops::matmul::Operand Side, typename Source>
struct OperandInvariants {
  nint_t row_bytes = 0;
  bool gather_offsets_fit = false;

  VECOPS_ALWAYS_INLINE explicit OperandInvariants(const Source& source) {
    using T = typename ::vecops::matmul::packing_t<Atom, Side>::Element;
    if constexpr (direct_row_major_input_v<Source> && sizeof(T) <= 4) {
      row_bytes = static_cast<nint_t>(tensor::stride<0>(
          source.spec().input_layout())) *
          static_cast<nint_t>(sizeof(T));
      gather_offsets_fit =
          row_bytes >= std::numeric_limits<int32_t>::min() &&
          row_bytes <= std::numeric_limits<int32_t>::max();
    }
  }
};

/// Reinterpret a K-grouped scalar-lane vector as a native (full-lane)
/// vector of the same element type.
template <typename T, vec::VectorValue V>
VECOPS_ALWAYS_INLINE auto as_native(V value) {
  using Tag = vec::ScalableTag<T, 0>;
  return static_cast<vec::Vec<Tag>>(value);
}

/**
 * @brief Load one operand vector (one K group of one spatial block).
 *
 * Packed inputs index the 4-D packed layout directly (panel, k-group, row,
 * k) and load one native vector at the block's row slice. Direct inputs
 * try the whole-group fast path first and fall back to per-k masked loads:
 *
 * - fast path (direct row-major, element <= 32 bits, whole K group within
 *   the logical K, gather offsets in range): load the column block as one
 *   strided `uint32` gather -- one row-strided instruction per whole K
 *   group instead of KPack per-element loads -- then bitcast back to the
 *   narrow element vector. This is the counterpart of the int32-range
 *   pre-check in `OperandInvariants`.
 * - fallback: per-k `source.load` along the spatial axis, masked and
 *   zero-filled for tail lanes/groups, then interleaved into the packed
 *   K-group lane order (`interleave_pair`/`interleave_quad` from the
 *   generic packing layer -- the same lane order the packers produce).
 *
 * @tparam Block  Spatial block index; the loaded rows are
 *                `origin + Block*lanes .. +lanes`.
 * @tparam FullSpatial / FullK  Compile-time guarantees that the full
 *                spatial block / whole K group is active; they select the
 *                unmasked, unchecked load forms.
 */
template <::vecops::matmul::Atom Atom, ::vecops::matmul::Operand Side, int Block,
          bool FullSpatial, bool FullK, typename Source>
VECOPS_ALWAYS_INLINE auto load_operand(
    const Source& source,
    const OperandInvariants<Atom, Side, Source>& invariants,
    nint_t origin, nint_t kg,
    nint_t logical_spatial, nint_t logical_k) {
  using Packing = ::vecops::matmul::packing_t<Atom, Side>;
  using T = typename Packing::Element;
  constexpr nint_t KP = Packing::KPack;
  using ScalarTag = vec::ScalableTag<T, -log2_kpack(KP)>;
  using NativeTag = vec::ScalableTag<T, 0>;
  const nint_t lanes = accumulator_lanes<Atom>();
  const nint_t spatial = origin + static_cast<nint_t>(Block) * lanes;
  if constexpr (is_packed_access_v<Atom, Side, Source>) {
    static_assert(generic::RawDirectAccess<Source>,
                  "packed SME input must be direct and untransformed");
    const auto& layout = source.spec().input_layout();
    const nint_t panel = static_cast<nint_t>(Packing::panel());
    const nint_t offset = tensor::offset_at(
        layout, spatial / panel, kg, spatial % panel, 0);
    return vec::load(
        NativeTag{}, reinterpret_cast<const T*>(source.raw_data()) + offset);
  } else {
    static_assert(Source::Rank == 2, "unpacked SME input must be rank two");
    const nint_t active = FullSpatial
        ? lanes
        : vecops::clamp(
              logical_spatial - spatial, nint_t{0}, lanes);
    if constexpr (direct_row_major_input_v<Source> && sizeof(T) <= 4) {
      const auto strides = source.raw_strides();
      const nint_t k = kg * KP;
      if ((FullK || k + KP <= logical_k) &&
          invariants.gather_offsets_fit) {
        // Whole-group gather fast path: one strided uint32 load fetches the
        // entire spatial column block (row_bytes apart) in a single
        // instruction; each fetched 32-bit word is KPack narrow elements of
        // one row, so bitcasting back yields the interleaved K-group lane
        // order directly.
        using WordTag = vec::ScalableTag<uint32_t, 0>;
        using Resources = typename std::remove_cvref_t<
            decltype(source.policy())>::ActiveResources;
        const auto* base = reinterpret_cast<const uint32_t*>(
            source.raw_data() + spatial * strides[0] + k);
        const auto words = [&]() VECOPS_INLINE_LAMBDA {
          if constexpr (FullSpatial) {
            return vec::load(
                WordTag{}, base,
                vec::strided(invariants.row_bytes, vec::scale<1>),
                vec::resources<Resources>);
          } else {
            return vec::load(
                WordTag{}, base, vec::opt::first(active),
                vec::strided(invariants.row_bytes, vec::scale<1>),
                vec::resources<Resources>);
          }
        }();
        return vec::bitcast(NativeTag{}, WordTag{}, words);
      }
    }
    auto load_k = [&](nint_t ki) VECOPS_INLINE_LAMBDA {
      const nint_t k = kg * KP + ki;
      if constexpr (FullK && FullSpatial) {
        return source.load(
            ScalarTag{}, tensor::coord(spatial, k), tensor::axis<0>);
      } else if constexpr (FullK) {
        return source.load(
            ScalarTag{}, tensor::coord(spatial, k), tensor::axis<0>,
            vec::opt::first(active), vec::opt::zero);
      } else if (k < logical_k) {
        if constexpr (FullSpatial) {
          return source.load(
              ScalarTag{}, tensor::coord(spatial, k), tensor::axis<0>);
        } else {
          return source.load(
              ScalarTag{}, tensor::coord(spatial, k), tensor::axis<0>,
              vec::opt::first(active), vec::opt::zero);
        }
      }
      return vec::zeros(ScalarTag{});
    };
    if constexpr (KP == 1) {
      return as_native<T>(load_k(0));
    } else if constexpr (KP == 2) {
      return as_native<T>(generic::interleave_pair<ScalarTag>(
          load_k(0), load_k(1)));
    } else {
      static_assert(KP == 4);
      return as_native<T>(generic::interleave_quad<ScalarTag>(
          load_k(0), load_k(1), load_k(2), load_k(3)));
    }
  }
}

/**
 * @brief One fully-predicated SME outer product into ZA tile `Tile`.
 *
 * Thin wrapper over the ZA intrinsic layer (vec/details/sme/ZA.h): selects
 * the FMOPA/BFMOPA/SMOPA/UMOPA/... instruction from the atom's types and
 * issues it with all-true predicates. The tile number is relative to the
 * accumulator width (validated in ZA.h).
 */
template <::vecops::matmul::Atom Atom, int Tile, typename VA, typename VB>
VECOPS_ALWAYS_INLINE void mopa(VA a, VB b) {
  using TA = typename Atom::TA;
  using TB = typename Atom::TB;
  using TAcc = typename Atom::TAcc;
  using ATag = vec::ScalableTag<TA, 0>;
  using BTag = vec::ScalableTag<TB, 0>;
  const auto pga = vec::mtrue(ATag{});
  const auto pgb = vec::mtrue(BTag{});
  vec::details::sme::mopa<Tile, TAcc, TA, TB>(
      pga, pgb, static_cast<vec::Vec<ATag>>(a),
      static_cast<vec::Vec<BTag>>(b));
}

/**
 * @brief Base pointer of one spatial block's packed rows at k-group 0.
 *
 * The fast packed loop (`compute_packed_groups`) walks K groups by stepping
 * this pointer with the packed group stride, avoiding per-group offset
 * recomputation. Same 4-D indexing as the packed branch of `load_operand`.
 */
template <::vecops::matmul::Atom Atom, ::vecops::matmul::Operand Side, int Block, typename Source>
VECOPS_ALWAYS_INLINE auto packed_block_pointer(
    const Source& source, nint_t origin) {
  static_assert(is_packed_access_v<Atom, Side, Source>);
  using Packing = ::vecops::matmul::packing_t<Atom, Side>;
  using T = typename Packing::Element;
  const nint_t lanes = accumulator_lanes<Atom>();
  const nint_t spatial = origin + static_cast<nint_t>(Block) * lanes;
  const nint_t panel = static_cast<nint_t>(Packing::panel());
  const nint_t offset = tensor::offset_at(
      source.spec().input_layout(), spatial / panel, 0,
      spatial % panel, 0);
  return reinterpret_cast<const T*>(source.raw_data()) + offset;
}

#if !defined(VECOPS_DISABLE_SME_SVE_SKINNY)
/// Widen-only helper for the skinny kernels: identity when tags already
/// match, otherwise a plain convert (never narrowing).
template <typename ToTag, typename FromTag, vec::VectorValue V>
VECOPS_ALWAYS_INLINE auto skinny_convert(ToTag to, FromTag from, V value) {
  if constexpr (std::same_as<ToTag, FromTag>) return value;
  else return vec::convert(to, from, value);
}

// TODO: Generalize after the FP16
// FMLAL path has passed all performance and numerical gates.
/// fp16 widening multiply-add for the skinny kernels: prefers the SVE
/// FMLALB/FMLALT pair (widening fma on half-lane halves, no separate
/// convert) when the architecture has it, otherwise converts then fma.
template <typename InputTag>
VECOPS_ALWAYS_INLINE auto skinny_widening_fmadd(
    InputTag, vec::Vec<vec::Rebind<float32_t, InputTag>> acc,
    vec::Vec<InputTag> a, vec::Vec<InputTag> b) {
  using AccTag = vec::Rebind<float32_t, InputTag>;
#if defined(CPU_CAPABILITY_SVE) && \
    defined(__ARM_FEATURE_FP16_VECTOR_ARITHMETIC)
  const auto raw_a = vec::details::sve_basic_raw_word(a);
  const auto raw_b = vec::details::sve_basic_raw_word(b);
  return vec::details::construct_words<vec::details::SVEBackend>(
      AccTag{}, [&]<nint_t Index>(AccTag) {
        static_assert(Index == 0 || Index == 1);
        const auto raw_acc = vec::details::sve_basic_raw_word(
            vec::get_word<Index>(AccTag{}, acc));
        if constexpr (Index == 0) {
          return vec::details::sve_basic_wrap_word<AccTag>(
              svmlalb_f32(raw_acc, raw_a, raw_b));
        } else {
          return vec::details::sve_basic_wrap_word<AccTag>(
              svmlalt_f32(raw_acc, raw_a, raw_b));
        }
      });
#else
  return vec::fmadd(
      AccTag{}, vec::convert(AccTag{}, InputTag{}, a),
      vec::convert(AccTag{}, InputTag{}, b), acc);
#endif
}

/**
 * @brief Ordinary-SVE skinny block: `Block` outputs of a GEMV-like product.
 *
 * Computes `Block` dot products (one output per accumulating vector) over
 * the shared operand -- B's row 0 when `VaryRows` (M==1, outputs are
 * columns), A's row 0 otherwise (N==1, outputs are rows) -- and reduces each
 * vector to one scalar stored through the output access. Loads are
 * masked/zero-filled per K chunk, so any K tail is handled without a
 * special-cased loop.
 *
 * @tparam VaryRows  Orientation: true = the varying index is M (outputs
 *                   stride along the output's leading axis), false = N.
 * @tparam Block     Outputs computed by this instantiation (1..8).
 */
template <bool VaryRows, int Block,
          typename A, typename B, typename COutput>
VECOPS_ALWAYS_INLINE void sve_skinny_block(
    const A& a, const B& b, COutput& c_output,
    nint_t output_origin, nint_t logical_k) {
  using TA = typename A::ComputeType;
  using TB = typename B::ComputeType;
  using Acc = typename COutput::ComputeType;
  static_assert(std::same_as<TA, TB>);
  using InputTag = vec::ScalableTag<TA, 0>;
  using AccTag = std::conditional_t<
      std::same_as<TA, bfloat16_t>,
      vec::ViewAs<float32_t, InputTag>,
      std::conditional_t<
          std::is_integral_v<TA>, vec::ViewAs<int32_t, InputTag>,
          vec::Rebind<Acc, InputTag>>>;
  auto sum0 = vec::zeros(AccTag{});
  auto sum1 = vec::zeros(AccTag{});
  auto sum2 = vec::zeros(AccTag{});
  auto sum3 = vec::zeros(AccTag{});
  auto sum4 = vec::zeros(AccTag{});
  auto sum5 = vec::zeros(AccTag{});
  auto sum6 = vec::zeros(AccTag{});
  auto sum7 = vec::zeros(AccTag{});

  const auto a_strides = a.raw_strides();
  const auto b_strides = b.raw_strides();
  const auto* a_data = reinterpret_cast<const TA*>(a.raw_data());
  const auto* b_data = reinterpret_cast<const TB*>(b.raw_data());
  nint_t kk = 0;
  nint_t remaining = logical_k;
  while (remaining > 0) {
    const nint_t active = vecops::min(vec::size(InputTag{}), remaining);
    const auto load_row = [&](const auto* data, nint_t row, nint_t stride) {
      return vec::load(
          InputTag{}, data + row * stride + kk,
          vec::opt::first(active), vec::opt::zero);
    };
    const auto madd = [&](auto lhs, auto rhs, auto sum) {
      if constexpr (std::same_as<TA, bfloat16_t>) {
        return vec::widening_dot(AccTag{}, lhs, rhs, sum);
      } else if constexpr (std::same_as<TA, float16_t>) {
        return skinny_widening_fmadd(InputTag{}, sum, lhs, rhs);
      } else if constexpr (std::is_integral_v<TA>) {
        return vec::widening_dot(AccTag{}, lhs, rhs, sum);
      } else {
        return vec::fmadd(
            skinny_convert(AccTag{}, InputTag{}, lhs),
            skinny_convert(AccTag{}, InputTag{}, rhs), sum);
      }
    };
    if constexpr (VaryRows) {
      const auto shared = load_row(b_data, 0, b_strides[0]);
      sum0 = madd(
          load_row(a_data, output_origin, a_strides[0]), shared, sum0);
      if constexpr (Block >= 2)
        sum1 = madd(
            load_row(a_data, output_origin + 1, a_strides[0]), shared, sum1);
      if constexpr (Block >= 3)
        sum2 = madd(
            load_row(a_data, output_origin + 2, a_strides[0]), shared, sum2);
      if constexpr (Block >= 4)
        sum3 = madd(
            load_row(a_data, output_origin + 3, a_strides[0]), shared, sum3);
      if constexpr (Block >= 5)
        sum4 = madd(
            load_row(a_data, output_origin + 4, a_strides[0]), shared, sum4);
      if constexpr (Block >= 6)
        sum5 = madd(
            load_row(a_data, output_origin + 5, a_strides[0]), shared, sum5);
      if constexpr (Block >= 7)
        sum6 = madd(
            load_row(a_data, output_origin + 6, a_strides[0]), shared, sum6);
      if constexpr (Block >= 8)
        sum7 = madd(
            load_row(a_data, output_origin + 7, a_strides[0]), shared, sum7);
    } else {
      const auto shared = load_row(a_data, 0, a_strides[0]);
      sum0 = madd(
          shared, load_row(b_data, output_origin, b_strides[0]), sum0);
      if constexpr (Block >= 2)
        sum1 = madd(
            shared, load_row(b_data, output_origin + 1, b_strides[0]), sum1);
      if constexpr (Block >= 3)
        sum2 = madd(
            shared, load_row(b_data, output_origin + 2, b_strides[0]), sum2);
      if constexpr (Block >= 4)
        sum3 = madd(
            shared, load_row(b_data, output_origin + 3, b_strides[0]), sum3);
      if constexpr (Block >= 5)
        sum4 = madd(
            shared, load_row(b_data, output_origin + 4, b_strides[0]), sum4);
      if constexpr (Block >= 6)
        sum5 = madd(
            shared, load_row(b_data, output_origin + 5, b_strides[0]), sum5);
      if constexpr (Block >= 7)
        sum6 = madd(
            shared, load_row(b_data, output_origin + 6, b_strides[0]), sum6);
      if constexpr (Block >= 8)
        sum7 = madd(
            shared, load_row(b_data, output_origin + 7, b_strides[0]), sum7);
    }
    kk += active;
    remaining -= active;
  }

  auto* output = reinterpret_cast<Acc*>(c_output.raw_data());
  const nint_t output_stride = static_cast<nint_t>(
      tensor::stride<0>(c_output.spec().output_layout()));
  const auto store = [&](nint_t logical, Acc value) {
    output[VaryRows ? logical * output_stride : logical] = value;
  };
  store(output_origin, vec::reduce_add(AccTag{}, sum0));
  if constexpr (Block >= 2)
    store(output_origin + 1, vec::reduce_add(AccTag{}, sum1));
  if constexpr (Block >= 3)
    store(output_origin + 2, vec::reduce_add(AccTag{}, sum2));
  if constexpr (Block >= 4)
    store(output_origin + 3, vec::reduce_add(AccTag{}, sum3));
  if constexpr (Block >= 5)
    store(output_origin + 4, vec::reduce_add(AccTag{}, sum4));
  if constexpr (Block >= 6)
    store(output_origin + 5, vec::reduce_add(AccTag{}, sum5));
  if constexpr (Block >= 7)
    store(output_origin + 6, vec::reduce_add(AccTag{}, sum6));
  if constexpr (Block >= 8)
    store(output_origin + 7, vec::reduce_add(AccTag{}, sum7));
}

/**
 * @brief Outer loop over a skinny problem: pick Block sizes left to right.
 *
 * bf16 and integer types get an 8-wide tier (their widening dot products
 * make the wide block profitable); everything else starts at 4. The 3/2/1
 * tail is switch-dispatched so no masked partial block is needed.
 */
template <bool VaryRows, typename A, typename B, typename COutput>
inline VECOPS_NOINLINE VECOPS_FUNCTION_ALIGN(64) void sve_skinny_matmul(
    const A& a, const B& b, COutput& c_output,
    nint_t outputs, nint_t logical_k) {
  using TA = typename A::ComputeType;
  nint_t output = 0;
  if constexpr (std::same_as<TA, bfloat16_t> || std::is_integral_v<TA>) {
    VECOPS_LOOP_ALIGN(64) for (; output + 8 <= outputs; output += 8)
      sve_skinny_block<VaryRows, 8>(a, b, c_output, output, logical_k);
  }
  VECOPS_LOOP_ALIGN(64) for (; output + 4 <= outputs; output += 4)
    sve_skinny_block<VaryRows, 4>(a, b, c_output, output, logical_k);
  switch (outputs - output) {
    case 3:
      sve_skinny_block<VaryRows, 3>(a, b, c_output, output, logical_k);
      break;
    case 2:
      sve_skinny_block<VaryRows, 2>(a, b, c_output, output, logical_k);
      break;
    case 1:
      sve_skinny_block<VaryRows, 1>(a, b, c_output, output, logical_k);
      break;
    default:
      break;
  }
}

/// Candidate for the raw skinny leaf: direct row-major inputs and output,
/// zero-value C input, same-type operands, and a float or (i8/u8 -> i32)
/// accumulator. No output transform (that is the fused variant below).
template <::vecops::matmul::Atom Atom,
          typename A, typename B, typename CInput, typename COutput>
inline constexpr bool sve_skinny_candidate_v =
    direct_row_major_input_v<A> && direct_row_major_input_v<B> &&
    direct_row_major_output_v<COutput> &&
    IsZeroTransform<typename CInput::Transform>::value &&
    std::same_as<typename A::ComputeType, typename Atom::TA> &&
    std::same_as<typename B::ComputeType, typename Atom::TB> &&
    std::same_as<typename COutput::ComputeType, typename Atom::TAcc> &&
    std::same_as<typename Atom::TA, typename Atom::TB> &&
    (std::is_floating_point_v<typename Atom::TAcc> ||
     (std::is_integral_v<typename Atom::TA> &&
      std::same_as<typename Atom::TAcc, int32_t>));

#if defined(__ARM_FEATURE_SVE_MATMUL_INT8)
/// Mixed-sign int8 GEMV leaves (s8 x u8 / u8 x s8, SUMOPA-style algebra),
/// implemented out of line in the backend TU.
bool try_raw_mixed_sign_skinny_s8u8(
    nint_t logical_m, nint_t logical_n, nint_t logical_k,
    const int8_t* a, nint_t a_stride,
    const uint8_t* b, nint_t b_stride,
    int32_t* output, nint_t output_stride);

bool try_raw_mixed_sign_skinny_u8s8(
    nint_t logical_m, nint_t logical_n, nint_t logical_k,
    const uint8_t* a, nint_t a_stride,
    const int8_t* b, nint_t b_stride,
    int32_t* output, nint_t output_stride);

/// Candidate for the mixed-sign skinny leaves: like sve_skinny_candidate_v
/// but with deliberately *different* operand signedness (the same-sign
/// cases stay on the regular skinny path).
template <::vecops::matmul::Atom Atom,
          typename A, typename B, typename CInput, typename COutput>
inline constexpr bool mixed_sign_sve_skinny_candidate_v =
    direct_row_major_input_v<A> && direct_row_major_input_v<B> &&
    direct_row_major_output_v<COutput> &&
    IsZeroTransform<typename CInput::Transform>::value &&
    std::same_as<typename A::ComputeType, typename Atom::TA> &&
    std::same_as<typename B::ComputeType, typename Atom::TB> &&
    std::same_as<typename COutput::ComputeType, int32_t> &&
    ((std::same_as<typename Atom::TA, int8_t> &&
      std::same_as<typename Atom::TB, uint8_t>) ||
     (std::same_as<typename Atom::TA, uint8_t> &&
      std::same_as<typename Atom::TB, int8_t>)) &&
    std::same_as<typename Atom::TAcc, int32_t>;

/// Run-time gate + forwarder for the mixed-sign skinny leaves (raw pointer
/// variants live in the backend TU; section attribute keeps them out of the
/// hot kernel text).
template <::vecops::matmul::Atom Atom,
          typename A, typename B, typename CInput, typename COutput>
VECOPS_NOINLINE VECOPS_FUNCTION_ALIGN(64)
#if defined(COMPILER_CLANG)
__attribute__((preserve_most))
#endif
__attribute__((section(".vecops_kernel_text"))) bool
try_mixed_sign_sve_skinny(
    nint_t logical_m, nint_t logical_n, nint_t logical_k,
    const A& a, const B& b, const CInput&, COutput& c_output) {
  if constexpr (!mixed_sign_sve_skinny_candidate_v<
                    Atom, A, B, CInput, COutput>) {
    return false;
  } else {
    const auto a_strides = a.raw_strides();
    const auto b_strides = b.raw_strides();
    const auto output_strides = c_output.raw_strides();
    if constexpr (
        std::same_as<typename Atom::TA, int8_t> &&
        std::same_as<typename Atom::TB, uint8_t>) {
      return try_raw_mixed_sign_skinny_s8u8(
          logical_m, logical_n, logical_k,
          reinterpret_cast<const int8_t*>(a.raw_data()), a_strides[0],
          reinterpret_cast<const uint8_t*>(b.raw_data()), b_strides[0],
          reinterpret_cast<int32_t*>(c_output.raw_data()), output_strides[0]);
    } else {
      return try_raw_mixed_sign_skinny_u8s8(
          logical_m, logical_n, logical_k,
          reinterpret_cast<const uint8_t*>(a.raw_data()), a_strides[0],
          reinterpret_cast<const int8_t*>(b.raw_data()), b_strides[0],
          reinterpret_cast<int32_t*>(c_output.raw_data()), output_strides[0]);
    }
  }
}
#endif

#if !defined(VECOPS_DISABLE_SME_FUSED_SKINNY)
#if defined(HAS_SME_F64F64)
/// fp64 fused-skinny compute cores (out of line): the ZA-less fp64 dot
/// products for the column- and row-varying orientations.
void sve_skinny_fused_compute_f64_row(
    const float64_t* a_data, nint_t a_stride,
    const float64_t* b_data, nint_t b_stride,
    float64_t* values, nint_t outputs, nint_t logical_k);

void sve_skinny_fused_compute_f64_col(
    const float64_t* a_data, nint_t a_stride,
    const float64_t* b_data, nint_t b_stride,
    float64_t* values, nint_t outputs, nint_t logical_k);
#endif

/**
 * @brief Compute core of the fused skinny kernels over raw pointers.
 *
 * Same dot-product structure as `sve_skinny_block`, but decoupled from the
 * tensor access layer: results land in a scalar `values` scratch array so
 * the epilogue (`sve_skinny_fused_matmul`) can write them through the
 * output access with its transform attached.
 *
 * @tparam VaryRows  Orientation, as in sve_skinny_block.
 * @tparam Block     Outputs per instantiation (1..8).
 */
template <bool VaryRows, int Block, typename T, typename Acc>
VECOPS_ALWAYS_INLINE void sve_skinny_fused_compute_block(
    const T* a_data, nint_t a_stride,
    const T* b_data, nint_t b_stride,
    Acc* values, nint_t output_origin, nint_t logical_k) {
  static_assert(1 <= Block && Block <= 8);
  using InputTag = vec::ScalableTag<T, 0>;
  using AccTag = std::conditional_t<
      std::same_as<T, bfloat16_t>,
      vec::ViewAs<float32_t, InputTag>,
      std::conditional_t<
          std::is_integral_v<T>, vec::ViewAs<int32_t, InputTag>,
          vec::Rebind<Acc, InputTag>>>;
  auto sum0 = vec::zeros(AccTag{});
  auto sum1 = vec::zeros(AccTag{});
  auto sum2 = vec::zeros(AccTag{});
  auto sum3 = vec::zeros(AccTag{});
  auto sum4 = vec::zeros(AccTag{});
  auto sum5 = vec::zeros(AccTag{});
  auto sum6 = vec::zeros(AccTag{});
  auto sum7 = vec::zeros(AccTag{});

  nint_t kk = 0;
  nint_t remaining = logical_k;
  while (remaining > 0) {
    const nint_t active = vecops::min(vec::size(InputTag{}), remaining);
    const auto load_row = [&](
        const T* data, nint_t row, nint_t stride) VECOPS_INLINE_LAMBDA {
      return vec::load(
          InputTag{}, data + row * stride + kk,
          vec::opt::first(active), vec::opt::zero);
    };
    const auto madd = [&](auto lhs, auto rhs, auto sum)
        VECOPS_INLINE_LAMBDA {
      if constexpr (std::same_as<T, bfloat16_t>) {
        return vec::widening_dot(AccTag{}, lhs, rhs, sum);
      } else if constexpr (std::same_as<T, float16_t>) {
        return skinny_widening_fmadd(InputTag{}, sum, lhs, rhs);
      } else if constexpr (std::is_integral_v<T>) {
        return vec::widening_dot(AccTag{}, lhs, rhs, sum);
      } else {
        return vec::fmadd(
            skinny_convert(AccTag{}, InputTag{}, lhs),
            skinny_convert(AccTag{}, InputTag{}, rhs), sum);
      }
    };
    if constexpr (VaryRows) {
      const auto shared = load_row(b_data, 0, b_stride);
      sum0 = madd(load_row(a_data, output_origin, a_stride), shared, sum0);
      if constexpr (Block >= 2)
        sum1 = madd(
            load_row(a_data, output_origin + 1, a_stride), shared, sum1);
      if constexpr (Block >= 3)
        sum2 = madd(
            load_row(a_data, output_origin + 2, a_stride), shared, sum2);
      if constexpr (Block >= 4)
        sum3 = madd(
            load_row(a_data, output_origin + 3, a_stride), shared, sum3);
      if constexpr (Block >= 5)
        sum4 = madd(
            load_row(a_data, output_origin + 4, a_stride), shared, sum4);
      if constexpr (Block >= 6)
        sum5 = madd(
            load_row(a_data, output_origin + 5, a_stride), shared, sum5);
      if constexpr (Block >= 7)
        sum6 = madd(
            load_row(a_data, output_origin + 6, a_stride), shared, sum6);
      if constexpr (Block >= 8)
        sum7 = madd(
            load_row(a_data, output_origin + 7, a_stride), shared, sum7);
    } else {
      const auto shared = load_row(a_data, 0, a_stride);
      sum0 = madd(shared, load_row(b_data, output_origin, b_stride), sum0);
      if constexpr (Block >= 2)
        sum1 = madd(
            shared, load_row(b_data, output_origin + 1, b_stride), sum1);
      if constexpr (Block >= 3)
        sum2 = madd(
            shared, load_row(b_data, output_origin + 2, b_stride), sum2);
      if constexpr (Block >= 4)
        sum3 = madd(
            shared, load_row(b_data, output_origin + 3, b_stride), sum3);
      if constexpr (Block >= 5)
        sum4 = madd(
            shared, load_row(b_data, output_origin + 4, b_stride), sum4);
      if constexpr (Block >= 6)
        sum5 = madd(
            shared, load_row(b_data, output_origin + 5, b_stride), sum5);
      if constexpr (Block >= 7)
        sum6 = madd(
            shared, load_row(b_data, output_origin + 6, b_stride), sum6);
      if constexpr (Block >= 8)
        sum7 = madd(
            shared, load_row(b_data, output_origin + 7, b_stride), sum7);
    }
    kk += active;
    remaining -= active;
  }

  values[output_origin] = vec::reduce_add(AccTag{}, sum0);
  if constexpr (Block >= 2)
    values[output_origin + 1] = vec::reduce_add(AccTag{}, sum1);
  if constexpr (Block >= 3)
    values[output_origin + 2] = vec::reduce_add(AccTag{}, sum2);
  if constexpr (Block >= 4)
    values[output_origin + 3] = vec::reduce_add(AccTag{}, sum3);
  if constexpr (Block >= 5)
    values[output_origin + 4] = vec::reduce_add(AccTag{}, sum4);
  if constexpr (Block >= 6)
    values[output_origin + 5] = vec::reduce_add(AccTag{}, sum5);
  if constexpr (Block >= 7)
    values[output_origin + 6] = vec::reduce_add(AccTag{}, sum6);
  if constexpr (Block >= 8)
    values[output_origin + 7] = vec::reduce_add(AccTag{}, sum7);
}

/// Block-size ladder over the fused compute core (8/4 tiers + switch tail),
/// mirroring sve_skinny_matmul.
template <bool VaryRows, typename T, typename Acc>
VECOPS_NOINLINE VECOPS_FUNCTION_ALIGN(64) void sve_skinny_fused_compute(
    const T* a_data, nint_t a_stride,
    const T* b_data, nint_t b_stride,
    Acc* values, nint_t outputs, nint_t logical_k) {
  nint_t output = 0;
  if constexpr (std::same_as<T, bfloat16_t> || std::is_integral_v<T>) {
    VECOPS_LOOP_ALIGN(64) for (; output + 8 <= outputs; output += 8) {
      sve_skinny_fused_compute_block<VaryRows, 8>(
          a_data, a_stride, b_data, b_stride,
          values, output, logical_k);
    }
  }
  VECOPS_LOOP_ALIGN(64) for (; output + 4 <= outputs; output += 4) {
    sve_skinny_fused_compute_block<VaryRows, 4>(
        a_data, a_stride, b_data, b_stride,
        values, output, logical_k);
  }
  switch (outputs - output) {
    case 3:
      sve_skinny_fused_compute_block<VaryRows, 3>(
          a_data, a_stride, b_data, b_stride,
          values, output, logical_k);
      break;
    case 2:
      sve_skinny_fused_compute_block<VaryRows, 2>(
          a_data, a_stride, b_data, b_stride,
          values, output, logical_k);
      break;
    case 1:
      sve_skinny_fused_compute_block<VaryRows, 1>(
          a_data, a_stride, b_data, b_stride,
          values, output, logical_k);
      break;
    default:
      break;
  }
}

/**
 * @brief Fused (transform-carrying) skinny matmul: compute then epilogue.
 *
 * Runs the raw-pointer compute core into a stack `values` array, then
 * stores through `c_output`'s access so the output transform -- the reason
 * this leaf exists, e.g. a dequantize -- is applied by the tensor layer
 * instead of forcing the ZA kernel.
 */
template <bool VaryRows, typename A, typename B,
          typename CInput, typename COutput>
VECOPS_ALWAYS_INLINE void sve_skinny_fused_matmul(
    const A& a, const B& b, const CInput& c_input, COutput& c_output,
    nint_t outputs, nint_t logical_k) {
  using T = typename A::ComputeType;
  using Acc = typename COutput::ComputeType;
  static_assert(std::same_as<T, typename B::ComputeType>);
  constexpr nint_t MaxOutputs =
      std::same_as<T, float16_t> ? 16 : 64;
  alignas(64) Acc values[MaxOutputs];
  const auto a_strides = a.raw_strides();
  const auto b_strides = b.raw_strides();
  sve_skinny_fused_compute<VaryRows>(
      reinterpret_cast<const T*>(a.raw_data()), a_strides[0],
      reinterpret_cast<const T*>(b.raw_data()), b_strides[0],
      values, outputs, logical_k);

  using OutputTag = vec::ScalableTag<Acc, 0>;
  if constexpr (VaryRows) {
    for (nint_t output = 0; output < outputs; ++output) {
      auto value = vec::fill(OutputTag{}, values[output]);
      if constexpr (!IsZeroTransform<typename CInput::Transform>::value) {
        value = vec::add(
            OutputTag{}, value,
            c_input.load(
                OutputTag{}, tensor::coord(output, nint_t{0}),
                tensor::axis<1>, vec::opt::first(1), vec::opt::zero));
      }
      c_output.store(
          OutputTag{}, tensor::coord(output, nint_t{0}), tensor::axis<1>,
          value, vec::opt::first(1));
    }
  } else {
    const nint_t lanes = vec::size(OutputTag{});
    for (nint_t output = 0; output < outputs; output += lanes) {
      const nint_t active = vecops::min(lanes, outputs - output);
      auto value = vec::load(
          OutputTag{}, values + output,
          vec::opt::first(active), vec::opt::zero);
      if constexpr (!IsZeroTransform<typename CInput::Transform>::value) {
        value = vec::add(
            OutputTag{}, value,
            c_input.load(
                OutputTag{}, tensor::coord(nint_t{0}, output),
                tensor::axis<1>, vec::opt::first(active), vec::opt::zero));
      }
      c_output.store(
          OutputTag{}, tensor::coord(nint_t{0}, output), tensor::axis<1>,
          value, vec::opt::first(active));
    }
  }
}

/**
 * @brief Lane-local (non-elementwise) fused skinny wrapper.
 *
 * Keep coordinate-aware epilogues out of the already layout-sensitive fused
 * dispatch bodies. Their transform type remains open-ended, so the wrapper
 * stays in the header, but one noinline COMDAT per actual transform is enough.
 */
template <bool VaryRows, typename A, typename B,
          typename CInput, typename COutput>
VECOPS_NOINLINE VECOPS_FUNCTION_ALIGN(64) void
sve_skinny_fused_lane_local_matmul(
    const A& a, const B& b, const CInput& c_input, COutput& c_output,
    nint_t outputs, nint_t logical_k) {
  static_assert(!COutput::Transform::is_elementwise);
  static_assert(COutput::Transform::is_lane_local);
  sve_skinny_fused_matmul<VaryRows>(
      a, b, c_input, c_output, outputs, logical_k);
}

#if defined(HAS_SME_F64F64)
/// fp64 entry of the fused skinny family: delegates to the out-of-line
/// compute cores (which the header cannot inline cheaply) and reuses the
/// same stack-values epilogue.
template <bool VaryRows, typename A, typename B,
          typename CInput, typename COutput>
VECOPS_NOINLINE VECOPS_FUNCTION_ALIGN(64) void
sve_skinny_fused_matmul_f64_external(
    const A& a, const B& b, const CInput& c_input, COutput& c_output,
    nint_t outputs, nint_t logical_k) {
  static_assert(std::same_as<typename A::ComputeType, float64_t>);
  static_assert(std::same_as<typename B::ComputeType, float64_t>);
  static_assert(std::same_as<typename COutput::ComputeType, float64_t>);
  alignas(64) float64_t values[64];
  const auto a_strides = a.raw_strides();
  const auto b_strides = b.raw_strides();
  if constexpr (VaryRows) {
    sve_skinny_fused_compute_f64_col(
        reinterpret_cast<const float64_t*>(a.raw_data()), a_strides[0],
        reinterpret_cast<const float64_t*>(b.raw_data()), b_strides[0],
        values, outputs, logical_k);
  } else {
    sve_skinny_fused_compute_f64_row(
        reinterpret_cast<const float64_t*>(a.raw_data()), a_strides[0],
        reinterpret_cast<const float64_t*>(b.raw_data()), b_strides[0],
        values, outputs, logical_k);
  }

  using OutputTag = vec::ScalableTag<float64_t, 0>;
  if constexpr (VaryRows) {
    for (nint_t output = 0; output < outputs; ++output) {
      auto value = vec::fill(OutputTag{}, values[output]);
      if constexpr (!IsZeroTransform<typename CInput::Transform>::value) {
        value = vec::add(
            OutputTag{}, value,
            c_input.load(
                OutputTag{}, tensor::coord(output, nint_t{0}),
                tensor::axis<1>, vec::opt::first(1), vec::opt::zero));
      }
      c_output.store(
          OutputTag{}, tensor::coord(output, nint_t{0}), tensor::axis<1>,
          value, vec::opt::first(1));
    }
  } else {
    const nint_t lanes = vec::size(OutputTag{});
    for (nint_t output = 0; output < outputs; output += lanes) {
      const nint_t active = vecops::min(lanes, outputs - output);
      auto value = vec::load(
          OutputTag{}, values + output,
          vec::opt::first(active), vec::opt::zero);
      if constexpr (!IsZeroTransform<typename CInput::Transform>::value) {
        value = vec::add(
            OutputTag{}, value,
            c_input.load(
                OutputTag{}, tensor::coord(nint_t{0}, output),
                tensor::axis<1>, vec::opt::first(active), vec::opt::zero));
      }
      c_output.store(
          OutputTag{}, tensor::coord(nint_t{0}, output), tensor::axis<1>,
          value, vec::opt::first(active));
    }
  }
}
#endif

/// Whether the C prologue/epilogue transforms are applicable to the fused
/// skinny wrapper: lane-local transforms see each complete active vector;
/// anything coordinate-aware is out of scope.
template <::vecops::matmul::Atom Atom,
          typename A, typename B, typename CInput, typename COutput>
inline constexpr bool sve_skinny_fused_output_supported_v =
    CInput::Transform::is_lane_local && COutput::Transform::is_lane_local;

/**
 * @brief Candidate for the fused skinny leaf.
 *
 * Direct row-major A/B with either a non-zero C prologue or a transformed C
 * output, and the same type constraints as the raw skinny candidate.  This
 * includes split-K middle/last phases whose C input is the running
 * accumulator.  See the in-place TODO below for the FP64 gating.
 */
template <::vecops::matmul::Atom Atom,
          typename A, typename B, typename CInput, typename COutput>
inline constexpr bool sve_skinny_fused_candidate_v =
    direct_row_major_input_v<A> && direct_row_major_input_v<B> &&
    CInput::Rank == 2 && COutput::Rank == 2 &&
    (!IsZeroTransform<typename CInput::Transform>::value ||
     !direct_row_major_output_v<COutput>) &&
    sve_skinny_fused_output_supported_v<Atom, A, B, CInput, COutput> &&
    std::same_as<typename A::ComputeType, typename Atom::TA> &&
    std::same_as<typename B::ComputeType, typename Atom::TB> &&
    std::same_as<typename COutput::ComputeType, typename Atom::TAcc> &&
    std::same_as<typename Atom::TA, typename Atom::TB> &&
#if defined(HAS_SME_F64F64)
    (sizeof(typename Atom::TAcc) <= sizeof(float32_t) ||
     std::same_as<typename Atom::TAcc, float64_t>) &&
#else
    // TODO: Re-evaluate FP64 after its fused wrappers can live in a separate
    // cold TU. The leaf itself is faster, but adding it to the current header
    // perturbs unchanged FP64 fusion paths beyond the production regression
    // gate.
    sizeof(typename Atom::TAcc) <= sizeof(float32_t) &&
#endif
    (std::is_floating_point_v<typename Atom::TAcc> ||
     (std::is_integral_v<typename Atom::TA> &&
      std::same_as<typename Atom::TAcc, int32_t>));
#endif
#endif

#if defined(CPU_CAPABILITY_SVE)
/**
 * @brief Per-atom operations for the ordinary-SVE packed-dot kernels.
 *
 * One specialization per supported atom, each providing the SVE ACLE types
 * and the small vector helpers the dot loop needs. Two layout contracts
 * matter for every specialization (see `packed_ab_mmla` for their use):
 *
 * - `zip_groups(a, b)` interleaves two adjacent K-group vectors into the
 *   MMLA segment layout: after the zip, each 128-bit segment of the vector
 *   holds two spatial rows x two K groups (KPack elements each).
 * - `broadcast_segment<S>(v)` (svdupq_lane) replicates the S-th 128-bit
 *   segment, i.e. one row pair, across the whole vector.
 * - `store_row_pair` de-interleaves the MMLA accumulator into the two
 *   result rows (commented per specialization).
 */
template <typename Atom>
struct PackedDotTraits;

#if defined(__ARM_FEATURE_SVE_BF16)
/// bf16 x bf16 -> fp32 (svbfmmla_f32) traits.
template <>
struct PackedDotTraits<::vecops::matmul::SME_BF16F32> {
  using Element = bfloat16_t;
  using Acc = float32_t;
  using InputVec = svbfloat16_t;
  using AccVec = svfloat32_t;

  static VECOPS_ALWAYS_INLINE InputVec load(const Element* pointer) {
    return svreinterpret_bf16_u16(svld1_u16(
        svptrue_b16(), reinterpret_cast<const uint16_t*>(pointer)));
  }

  static VECOPS_ALWAYS_INLINE InputVec zero_input() {
    return svreinterpret_bf16_u16(svdup_u16(0));
  }

  static VECOPS_ALWAYS_INLINE InputVec zip_groups(
      InputVec group0, InputVec group1) {
    return svreinterpret_bf16_u32(svzip1_u32(
        svreinterpret_u32_bf16(group0),
        svreinterpret_u32_bf16(group1)));
  }

  template <int Segment>
  static VECOPS_ALWAYS_INLINE InputVec broadcast_segment(InputVec value) {
    return svdupq_lane_bf16(value, Segment);
  }

  static VECOPS_ALWAYS_INLINE AccVec zero_acc() { return svdup_f32(0); }

  static VECOPS_ALWAYS_INLINE AccVec mmla(
      AccVec acc, InputVec a, InputVec b) {
    return svbfmmla_f32(acc, a, b);
  }

  static VECOPS_ALWAYS_INLINE void store_contiguous(
      AccVec value, Acc* output, nint_t count) {
    svst1_f32(
        svwhilelt_b32(uint64_t{0}, static_cast<uint64_t>(count)),
        output, value);
  }

  static VECOPS_ALWAYS_INLINE void store_row_pair(
      AccVec value, Acc* row0, Acc* row1, nint_t columns) {
    // MMLA accumulator lane layout: svbfmmla places each 2x2 product block
    // with row 0 in the low 64 bits and row 1 in the high 64 bits of every
    // 128-bit segment. Reading the vector as 64-bit lanes, row 0 therefore
    // occupies the even and row 1 the odd 64-bit lanes, so a 64-bit
    // uzp1/uzp2 pair extracts the two rows as contiguous column vectors.
    const auto bits = svreinterpret_u64_f32(value);
    const auto first = svreinterpret_f32_u64(svuzp1_u64(bits, bits));
    const auto second = svreinterpret_f32_u64(svuzp2_u64(bits, bits));
    const auto pg = svwhilelt_b32(
        uint64_t{0}, static_cast<uint64_t>(columns));
    svst1_f32(pg, row0, first);
    svst1_f32(pg, row1, second);
  }
};
#endif

#if defined(__ARM_FEATURE_SVE_MATMUL_INT8)
/// int8 x int8 -> int32 (svmmla_s32) traits.
template <>
struct PackedDotTraits<::vecops::matmul::SME_I8I32<int8_t, int8_t>> {
  using Element = int8_t;
  using Acc = int32_t;
  using InputVec = svint8_t;
  using AccVec = svint32_t;

  static VECOPS_ALWAYS_INLINE InputVec load(const Element* pointer) {
    return svld1_s8(svptrue_b8(), pointer);
  }

  static VECOPS_ALWAYS_INLINE InputVec zero_input() {
    return svdup_s8(0);
  }

  static VECOPS_ALWAYS_INLINE InputVec zip_groups(
      InputVec group0, InputVec group1) {
    return svreinterpret_s8_u32(svzip1_u32(
        svreinterpret_u32_s8(group0), svreinterpret_u32_s8(group1)));
  }

  template <int Segment>
  static VECOPS_ALWAYS_INLINE InputVec broadcast_segment(InputVec value) {
    return svdupq_lane_s8(value, Segment);
  }

  static VECOPS_ALWAYS_INLINE AccVec zero_acc() { return svdup_s32(0); }

  static VECOPS_ALWAYS_INLINE AccVec mmla(
      AccVec acc, InputVec a, InputVec b) {
    return svmmla_s32(acc, a, b);
  }

  static VECOPS_ALWAYS_INLINE void store_contiguous(
      AccVec value, Acc* output, nint_t count) {
    svst1_s32(
        svwhilelt_b32(uint64_t{0}, static_cast<uint64_t>(count)),
        output, value);
  }

  static VECOPS_ALWAYS_INLINE void store_row_pair(
      AccVec value, Acc* row0, Acc* row1, nint_t columns) {
    // Same 64-bit row de-interleave as the BF16 traits: svmmla also
    // delivers each 2x2 block with row 0 in even and row 1 in odd 64-bit
    // lanes of every 128-bit segment.
    const auto bits = svreinterpret_u64_s32(value);
    const auto first = svreinterpret_s32_u64(svuzp1_u64(bits, bits));
    const auto second = svreinterpret_s32_u64(svuzp2_u64(bits, bits));
    const auto pg = svwhilelt_b32(
        uint64_t{0}, static_cast<uint64_t>(columns));
    svst1_s32(pg, row0, first);
    svst1_s32(pg, row1, second);
  }
};

/// uint8 x uint8 -> uint32 (svmmla_u32) traits; the u32 accumulator is
/// reinterpreted to int32 only at the memory boundary.
template <>
struct PackedDotTraits<::vecops::matmul::SME_I8I32<uint8_t, uint8_t>> {
  using Element = uint8_t;
  using Acc = int32_t;
  using InputVec = svuint8_t;
  using AccVec = svuint32_t;

  static VECOPS_ALWAYS_INLINE InputVec load(const Element* pointer) {
    return svld1_u8(svptrue_b8(), pointer);
  }

  static VECOPS_ALWAYS_INLINE InputVec zero_input() {
    return svdup_u8(0);
  }

  static VECOPS_ALWAYS_INLINE InputVec zip_groups(
      InputVec group0, InputVec group1) {
    return svreinterpret_u8_u32(svzip1_u32(
        svreinterpret_u32_u8(group0), svreinterpret_u32_u8(group1)));
  }

  template <int Segment>
  static VECOPS_ALWAYS_INLINE InputVec broadcast_segment(InputVec value) {
    return svdupq_lane_u8(value, Segment);
  }

  static VECOPS_ALWAYS_INLINE AccVec zero_acc() { return svdup_u32(0); }

  static VECOPS_ALWAYS_INLINE AccVec mmla(
      AccVec acc, InputVec a, InputVec b) {
    return svmmla_u32(acc, a, b);
  }

  static VECOPS_ALWAYS_INLINE void store_contiguous(
      AccVec value, Acc* output, nint_t count) {
    svst1_s32(
        svwhilelt_b32(uint64_t{0}, static_cast<uint64_t>(count)),
        output, svreinterpret_s32_u32(value));
  }

  static VECOPS_ALWAYS_INLINE void store_row_pair(
      AccVec value, Acc* row0, Acc* row1, nint_t columns) {
    // Same 64-bit row de-interleave as the other traits (svmmla 2x2 block
    // layout: row 0 even, row 1 odd 64-bit lanes per segment).
    const auto bits = svreinterpret_u64_u32(value);
    const auto first = svreinterpret_s32_u64(svuzp1_u64(bits, bits));
    const auto second = svreinterpret_s32_u64(svuzp2_u64(bits, bits));
    const auto pg = svwhilelt_b32(
        uint64_t{0}, static_cast<uint64_t>(columns));
    svst1_s32(pg, row0, first);
    svst1_s32(pg, row1, second);
  }
};
#endif

/// Atoms with a PackedDotTraits specialization (guarded by the SVE
/// feature macros that provide the underlying MMLA instructions).
template <::vecops::matmul::Atom Atom>
inline constexpr bool packed_dot_supported_atom_v =
#if defined(__ARM_FEATURE_SVE_BF16)
    std::same_as<Atom, ::vecops::matmul::SME_BF16F32> ||
#endif
#if defined(__ARM_FEATURE_SVE_MATMUL_INT8)
    std::same_as<Atom, ::vecops::matmul::SME_I8I32<int8_t, int8_t>> ||
    std::same_as<Atom, ::vecops::matmul::SME_I8I32<uint8_t, uint8_t>> ||
#endif
    false;

// TODO: Keep this ordinary-SVE MMLA operation SME-backend-local until the
// segment layout, mixed-sign behavior, and VL/SVL contract have another user.
// It intentionally consumes the existing MOPA PackedAB ABI by joining two
// adjacent 32-bit K groups; no second public packing format is introduced.
/**
 * @brief Ordinary-SVE MMLA microkernel over MOPA-packed A and B operands.
 *
 * Walks K in pairs of groups. Per pair, `zip_groups` joins the two adjacent
 * K-group vectors of each operand into the MMLA segment layout -- each
 * 128-bit segment then holds two spatial rows x two K groups -- and
 * `broadcast_segment` pins one operand to a chosen row pair (segment) while
 * the other walks segments, matching how svmmla/svbfmmla consume their
 * operands. Shape-specific schedules:
 *
 * - m == 8 (n == 2): one accumulator. A walks its four row-pair segments
 *   (8 rows), B is pinned to segment 0 (its only two columns); the result
 *   segments are already contiguous row-major, so one `store_contiguous`
 *   writes all m*n values.
 * - m == 4: two accumulators, A pinned per accumulator (segments 0 and 1 =
 *   rows 0..1 / 2..3), B walking its segments (all columns); each
 *   accumulator is de-interleaved into one row pair.
 * - m == 2 (fall-through): one accumulator, A pinned to segment 0 -- its
 *   only row pair -- B walking; a single store_row_pair finishes.
 *
 * An odd final group is zero-filled (`zero_input`) so the pairing loop
 * always reads two groups.
 *
 * @tparam Atom  Must satisfy packed_dot_supported_atom_v.
 */
template <::vecops::matmul::Atom Atom>
VECOPS_NOINLINE VECOPS_FUNCTION_ALIGN(64) void packed_ab_mmla(
    const typename Atom::TA* packed_a, nint_t a_group_stride,
    const typename Atom::TB* packed_b, nint_t b_group_stride,
    typename Atom::TAcc* output,
    nint_t logical_m, nint_t logical_n, nint_t logical_k) {
  using Op = PackedDotTraits<Atom>;
  constexpr nint_t KPack =
      ::vecops::matmul::packing_t<Atom, ::vecops::matmul::Operand::A>::KPack;
  const nint_t groups = ceil_div(logical_k, KPack);

  if (logical_m == 8) {
    auto acc = Op::zero_acc();
    VECOPS_LOOP_ALIGN(64) for (
        nint_t group = 0; group < groups; group += 2) {
      const auto a0 = Op::load(packed_a + group * a_group_stride);
      const auto b0 = Op::load(packed_b + group * b_group_stride);
      const auto a1 = group + 1 < groups
          ? Op::load(packed_a + (group + 1) * a_group_stride)
          : Op::zero_input();
      const auto b1 = group + 1 < groups
          ? Op::load(packed_b + (group + 1) * b_group_stride)
          : Op::zero_input();
    const auto a = Op::zip_groups(a0, a1);
    const auto b = Op::zip_groups(b0, b1);
      // A walks its row-pair segments; B pinned to segment 0 (columns 0,1).
      acc = Op::mmla(
          acc, a, Op::template broadcast_segment<0>(b));
  }
  Op::store_contiguous(acc, output, logical_m * logical_n);
  return;
}

  auto acc0 = Op::zero_acc();
  auto acc1 = Op::zero_acc();
  VECOPS_LOOP_ALIGN(64) for (
      nint_t group = 0; group < groups; group += 2) {
    const auto a0 = Op::load(packed_a + group * a_group_stride);
    const auto b0 = Op::load(packed_b + group * b_group_stride);
    const auto a1 = group + 1 < groups
        ? Op::load(packed_a + (group + 1) * a_group_stride)
        : Op::zero_input();
    const auto b1 = group + 1 < groups
        ? Op::load(packed_b + (group + 1) * b_group_stride)
        : Op::zero_input();
    const auto a = Op::zip_groups(a0, a1);
    const auto b = Op::zip_groups(b0, b1);
    // A pinned to one row pair per accumulator (segment 0 = rows 0..1,
    // segment 1 = rows 2..3); B walks its column segments. For m == 2 only
    // segment 0 of A exists, so acc1 stays unused.
    acc0 = Op::mmla(
        acc0, Op::template broadcast_segment<0>(a), b);
    if (logical_m == 4) {
      acc1 = Op::mmla(
          acc1, Op::template broadcast_segment<1>(a), b);
    }
  }
  Op::store_row_pair(
      acc0, output, output + logical_n, logical_n);
  if (logical_m == 4) {
    Op::store_row_pair(
        acc1, output + 2 * logical_n,
        output + 3 * logical_n, logical_n);
  }
}

/// Candidate for both packed-dot leaves.  Packed A/B remain raw-direct; C
/// may carry any lane-local prologue/epilogue.  The original zero/direct case
/// writes from MMLA straight to C, while split-K and transformed phases use a
/// tiny stack accumulator followed by the ordinary tensor access layer.
template <::vecops::matmul::Atom Atom,
          typename A, typename B, typename CInput, typename COutput>
inline constexpr bool packed_dot_candidate_v =
    packed_dot_supported_atom_v<Atom> &&
    is_packed_access_v<Atom, ::vecops::matmul::Operand::A, A> &&
    is_packed_access_v<Atom, ::vecops::matmul::Operand::B, B> &&
    generic::RawDirectAccess<A> && generic::RawDirectAccess<B> &&
    CInput::Rank == 2 && COutput::Rank == 2 &&
    CInput::Transform::is_lane_local && COutput::Transform::is_lane_local &&
    std::same_as<typename CInput::ComputeType, typename Atom::TAcc> &&
    std::same_as<typename COutput::ComputeType, typename Atom::TAcc>;

template <::vecops::matmul::Atom Atom,
          typename CInput, typename COutput>
VECOPS_ALWAYS_INLINE void packed_dot_epilogue(
    const typename Atom::TAcc* values,
    nint_t logical_m, nint_t logical_n,
    const CInput& c_input, COutput& c_output) {
  using Acc = typename Atom::TAcc;
  using Tag = vec::ScalableTag<Acc, 0>;
  const nint_t lanes = vec::size(Tag{});
  for (nint_t row = 0; row < logical_m; ++row) {
    for (nint_t column = 0; column < logical_n; column += lanes) {
      const nint_t active = vecops::min(lanes, logical_n - column);
      auto value = vec::load(
          Tag{}, values + row * logical_n + column,
          vec::opt::first(active), vec::opt::zero);
      if constexpr (!IsZeroTransform<typename CInput::Transform>::value) {
        value = vec::add(
            Tag{}, value,
            c_input.load(
                Tag{}, tensor::coord(row, column), tensor::axis<1>,
                vec::opt::first(active), vec::opt::zero));
      }
      c_output.store(
          Tag{}, tensor::coord(row, column), tensor::axis<1>, value,
          vec::opt::first(active));
    }
  }
}

/**
 * @brief Primary packed-dot leaf (2x8 / 8x2 / 4x4 shapes).
 *
 * Returns false (caller falls through to the ZA kernel) when the runtime
 * shape, K bound, panel/vector-length geometry, or output stride does not
 * match what `packed_ab_mmla` was tuned for.
 */
template <::vecops::matmul::Atom Atom,
          typename A, typename B, typename CInput, typename COutput>
VECOPS_ALWAYS_INLINE bool try_packed_dot(
    nint_t logical_m, nint_t logical_n, nint_t logical_k,
    const A& a, const B& b, const CInput& c_input, COutput& c_output) {
  if constexpr (!packed_dot_candidate_v<
                    Atom, A, B, CInput, COutput>) {
    return false;
  } else {
    const bool elongated =
        (logical_m == 2 && logical_n == 8) ||
        (logical_m == 8 && logical_n == 2);
    const bool square = logical_m == 4 && logical_n == 4;
    if ((!elongated && !square) || logical_k < 0) return false;

    // K dispatch bounds (tuning, not architecture): the dot leaf covers K
    // up to 256 (+1) for the square shape, 1024 (+1) for BF16, 512 (+1)
    // for the byte atoms; longer K amortizes better on the ZA kernel.
    const nint_t max_k = square
        ? 257
        : (std::same_as<Atom, ::vecops::matmul::SME_BF16F32> ? 1025 : 513);
    if (logical_k > max_k) return false;

    const nint_t a_panel = static_cast<nint_t>(
        ::vecops::matmul::packing_t<Atom, ::vecops::matmul::Operand::A>::panel());
    const nint_t b_panel = static_cast<nint_t>(
        ::vecops::matmul::packing_t<Atom, ::vecops::matmul::Operand::B>::panel());
    const nint_t ordinary_vl_bytes =
        vec::size(vec::ScalableTag<uint8_t, 0>{});
    // Panel/VL geometry: the packed panel is SVL/2 rows, so
    // ordinary_vl_bytes == 2*panel pins ordinary VL == streaming SVL -- the
    // one geometry the fixed vector loads were written for.
    if (a_panel != b_panel || ordinary_vl_bytes != 2 * a_panel) {
      return false;
    }
    constexpr bool DirectZeroOutput =
        IsZeroTransform<typename CInput::Transform>::value &&
        direct_row_major_output_v<COutput>;
    if constexpr (DirectZeroOutput) {
      // Preserve the exact X24 store sequence on the original fast path.
      if (static_cast<nint_t>(tensor::stride<0>(
              c_output.spec().output_layout())) != logical_n) return false;
    }

    alignas(64) typename Atom::TAcc values[64];
    auto* output = [&] {
      if constexpr (DirectZeroOutput) {
        return reinterpret_cast<typename Atom::TAcc*>(c_output.raw_data());
      } else {
        return values;
      }
    }();

    packed_ab_mmla<Atom>(
        reinterpret_cast<const typename Atom::TA*>(a.raw_data()),
        static_cast<nint_t>(tensor::stride<1>(
            a.spec().input_layout())),
        reinterpret_cast<const typename Atom::TB*>(b.raw_data()),
        static_cast<nint_t>(tensor::stride<1>(
            b.spec().input_layout())),
        output,
        logical_m, logical_n, logical_k);
    if constexpr (!DirectZeroOutput)
      packed_dot_epilogue<Atom>(
          values, logical_m, logical_n, c_input, c_output);
    return true;
  }
}

/**
 * @brief Tiny packed-dot leaf (2x2 / 2x4 / 4x2 shapes).
 *
 * Same gate structure and geometry checks as `try_packed_dot`; only the
 * served shapes and their K bound differ (see the tuning comment above).
 */
template <::vecops::matmul::Atom Atom,
          typename A, typename B, typename CInput, typename COutput>
VECOPS_ALWAYS_INLINE bool try_packed_dot_tiny(
    nint_t logical_m, nint_t logical_n, nint_t logical_k,
    const A& a, const B& b, const CInput& c_input, COutput& c_output) {
  if constexpr (!packed_dot_candidate_v<
                    Atom, A, B, CInput, COutput>) {
    return false;
  } else {
    const bool short_wide =
        logical_m == 2 && (logical_n == 2 || logical_n == 4);
    const bool tall_narrow = logical_m == 4 && logical_n == 2;
    if ((!short_wide && !tall_narrow) || logical_k < 0) return false;

    // K dispatch bounds, as in try_packed_dot (tuning-derived).
    const nint_t max_k = tall_narrow
        ? 257
        : (std::same_as<Atom, ::vecops::matmul::SME_BF16F32> ? 1025 : 513);
    if (logical_k > max_k) return false;

    const nint_t a_panel = static_cast<nint_t>(
        ::vecops::matmul::packing_t<Atom, ::vecops::matmul::Operand::A>::panel());
    const nint_t b_panel = static_cast<nint_t>(
        ::vecops::matmul::packing_t<Atom, ::vecops::matmul::Operand::B>::panel());
    const nint_t ordinary_vl_bytes =
        vec::size(vec::ScalableTag<uint8_t, 0>{});
    // Panel/VL geometry: ordinary VL == 2*panel == SVL, as above.
    if (a_panel != b_panel || ordinary_vl_bytes != 2 * a_panel) {
      return false;
    }
    constexpr bool DirectZeroOutput =
        IsZeroTransform<typename CInput::Transform>::value &&
        direct_row_major_output_v<COutput>;
    if constexpr (DirectZeroOutput) {
      if (static_cast<nint_t>(tensor::stride<0>(
              c_output.spec().output_layout())) != logical_n) return false;
    }

    alignas(64) typename Atom::TAcc values[64];
    auto* output = [&] {
      if constexpr (DirectZeroOutput) {
        return reinterpret_cast<typename Atom::TAcc*>(c_output.raw_data());
      } else {
        return values;
      }
    }();

    packed_ab_mmla<Atom>(
        reinterpret_cast<const typename Atom::TA*>(a.raw_data()),
        static_cast<nint_t>(tensor::stride<1>(
            a.spec().input_layout())),
        reinterpret_cast<const typename Atom::TB*>(b.raw_data()),
        static_cast<nint_t>(tensor::stride<1>(
            b.spec().input_layout())),
        output,
        logical_m, logical_n, logical_k);
    if constexpr (!DirectZeroOutput)
      packed_dot_epilogue<Atom>(
          values, logical_m, logical_n, c_input, c_output);
    return true;
  }
}
#endif

/**
 * @brief Which concrete kernel family serves one SME leaf.
 *
 * `General` is the ZA outer-product microkernel (the fallback); the other
 * nine are the ordinary-SVE, ZA-less leaves (see the file header). Row vs
 * Column suffixes on the skinny owners record the orientation: which of
 * M/N is the single extent.
 */
enum class DispatchOwner {
  General,
  MixedSignSkinnyRow,
  MixedSignSkinnyColumn,
  RawSkinnyRow,
  RawSkinnyColumn,
  FusedSkinnyRow,
  FusedSkinnyColumn,
  RuntimeQuantINT8,
  PackedDotPrimary,
  PackedDotTiny,
};

/// Whether Meta bounds pin one axis extent to exactly `Value`.
template <meta::ValueType E, nint_t Value>
inline constexpr bool extent_is_v =
    meta::range_within_v<std::remove_cvref_t<E>, Value, Value>;

/// Whether `Value` is outside the extent's admitted range entirely
/// (constant extents compare by value; Dynamic extents by conform()).
template <meta::ValueType E, nint_t Value>
inline constexpr bool extent_excludes_v = [] {
  using EV = std::remove_cvref_t<E>;
  return !EV::conforms(Value);
}();

/// Runtime-rules shape class for this Atom's skinny leaves (see
/// RuntimeDispatch.h): fp16's block ladder tops out at 16 outputs, the
/// other types at 64.
template <::vecops::matmul::Atom Atom>
inline constexpr SmallVectorShape small_vector_shape_v =
    std::same_as<Atom, ::vecops::matmul::SME_F16F32>
    ? SmallVectorShape::sme_f16
    : SmallVectorShape::sme_other;

/// Tri-state verdict on the skinny (SmallVector) leaves for the shape
/// contracts M/N/K: always when one axis is provably 1 and the other
/// provably within the block ladder, never when the runtime rules reject
/// a singleton problem or neither axis can be 1, runtime otherwise.
template <::vecops::matmul::Atom Atom,
          meta::ValueType M, meta::ValueType N, meta::ValueType K>
consteval ::vecops::matmul::details::Applicability
small_vector_shape_applicability() {
  using Applicability = ::vecops::matmul::details::Applicability;
  using MV = std::remove_cvref_t<M>;
  using NV = std::remove_cvref_t<N>;
  using KV = std::remove_cvref_t<K>;
  constexpr nint_t MaxOutputs =
      std::same_as<Atom, ::vecops::matmul::SME_F16F32> ? 16 : 64;
  if constexpr (meta::lower_bound_at_least_v<KV, 0> &&
                ((extent_is_v<MV, 1> &&
                  meta::range_within_v<NV, 0, MaxOutputs>) ||
                 (extent_is_v<NV, 1> &&
                  meta::range_within_v<MV, 0, MaxOutputs>))) {
    return Applicability::always;
  } else if constexpr (meta::is_singleton_v<MV> &&
                       meta::is_singleton_v<NV> &&
                       meta::is_singleton_v<KV>) {
    constexpr bool Applicable =
        runtime_dispatch_rules::sme_small_vector_profitable(
            small_vector_shape_v<Atom>,
            meta::singleton_value_v<MV>, meta::singleton_value_v<NV>,
            meta::singleton_value_v<KV>);
    return Applicable ? Applicability::always : Applicability::never;
  } else if constexpr (extent_excludes_v<MV, 1> &&
                       extent_excludes_v<NV, 1>) {
    return Applicability::never;
  } else {
    return Applicability::runtime;
  }
}

/// Tri-state verdict for the fused runtime-quant leaf: only a contract
/// family with no valid problem at all is `never`; strides and the packed
/// panel geometry are run-time facts, so everything else stays `runtime`
/// (the leaf re-checks them itself before running).
template <meta::ValueType M, meta::ValueType N, meta::ValueType K>
consteval ::vecops::matmul::details::Applicability
runtime_quant_shape_applicability() {
  using Applicability = ::vecops::matmul::details::Applicability;
  using MV = std::remove_cvref_t<M>;
  using NV = std::remove_cvref_t<N>;
  using KV = std::remove_cvref_t<K>;
  if constexpr ((meta::has_upper_bound_v<MV> &&
                 meta::upper_bound_v<MV> < 1) ||
                (meta::has_upper_bound_v<NV> &&
                 meta::upper_bound_v<NV> < 1) ||
                (meta::has_upper_bound_v<KV> &&
                 meta::upper_bound_v<KV> < 0)) {
    return Applicability::never;
  } else if constexpr (meta::lower_bound_at_least_v<MV, 1> &&
                       meta::lower_bound_at_least_v<NV, 1> &&
                       meta::lower_bound_at_least_v<KV, 0>) {
    // Strides and the runtime packing panel can still reject the leaf.
    return Applicability::runtime;
  } else {
    return Applicability::runtime;
  }
}

/// Tri-state verdict for the packed-dot leaves: `never` when the contracts
/// exclude every served (M, N) pair or K exceeds every leaf's bound,
/// `always` for singleton problems inside them, `runtime` otherwise (the
/// sparse five-shape table is one shared probe per instantiation).
template <::vecops::matmul::Atom Atom,
          meta::ValueType M, meta::ValueType N, meta::ValueType K>
consteval ::vecops::matmul::details::Applicability
packed_dot_shape_applicability() {
  using Applicability = ::vecops::matmul::details::Applicability;
  using MV = std::remove_cvref_t<M>;
  using NV = std::remove_cvref_t<N>;
  using KV = std::remove_cvref_t<K>;
  constexpr bool MayContainSupportedPair =
      (!extent_excludes_v<MV, 2> &&
       (!extent_excludes_v<NV, 2> ||
        !extent_excludes_v<NV, 4> ||
        !extent_excludes_v<NV, 8>)) ||
      (!extent_excludes_v<MV, 4> &&
       (!extent_excludes_v<NV, 2> ||
        !extent_excludes_v<NV, 4>)) ||
      (!extent_excludes_v<MV, 8> &&
       !extent_excludes_v<NV, 2>);
  constexpr nint_t GlobalMaxK =
      std::same_as<Atom, ::vecops::matmul::SME_BF16F32> ? 1025 : 513;
  if constexpr (!MayContainSupportedPair ||
                (meta::has_upper_bound_v<KV> &&
                 meta::upper_bound_v<KV> < 0) ||
                (meta::has_lower_bound_v<KV> &&
                 meta::lower_bound_v<KV> > GlobalMaxK)) {
    return Applicability::never;
  } else if constexpr (meta::is_singleton_v<MV> &&
                meta::is_singleton_v<NV> &&
                meta::is_singleton_v<KV>) {
    constexpr nint_t MValue = meta::singleton_value_v<MV>;
    constexpr nint_t NValue = meta::singleton_value_v<NV>;
    constexpr nint_t KValue = meta::singleton_value_v<KV>;
    constexpr bool PrimaryShape =
        (MValue == 2 && NValue == 8) ||
        (MValue == 8 && NValue == 2) ||
        (MValue == 4 && NValue == 4);
    constexpr bool TinyShape =
        (MValue == 2 && (NValue == 2 || NValue == 4)) ||
        (MValue == 4 && NValue == 2);
    constexpr nint_t PrimaryMaxK = MValue == 4 && NValue == 4
        ? 257
        : (std::same_as<Atom, ::vecops::matmul::SME_BF16F32>
               ? 1025
               : 513);
    constexpr nint_t TinyMaxK = MValue == 4 && NValue == 2
        ? 257
        : (std::same_as<Atom, ::vecops::matmul::SME_BF16F32>
               ? 1025
               : 513);
    constexpr bool Supported = KValue >= 0 &&
        ((PrimaryShape && KValue <= PrimaryMaxK) ||
         (TinyShape && KValue <= TinyMaxK));
    return Supported ? Applicability::always : Applicability::never;
  } else {
    // The five admitted (M,N) pairs are sparse. Keeping one shared runtime
    // probe avoids cloning the shape tree into every dynamic instantiation.
    return Applicability::runtime;
  }
}

/**
 * @brief Compile-time selection of the dispatch owner for `Automatic`
 *        policies.
 *
 * Each candidate leaf is gated by its compile-time candidate predicate plus
 * Meta shape bounds; the first match wins, checked most specific
 * (mixed-sign, then raw/fused skinny, runtime-quant, packed dot) before
 * the `General` fallback. The whole chain is consteval, so `run` compiles
 * straight-line code for the selected owner. Note the enclosing guard in
 * the caller: the non-General owners are only considered when the scope
 * does not already own StreamingZA.
 */
template <::vecops::matmul::Atom Atom,
          meta::ValueType M, meta::ValueType N, meta::ValueType K,
          typename A, typename B, typename CInput, typename COutput,
          typename Scope>
consteval DispatchOwner select_automatic_dispatch_owner() {
  using MV = std::remove_cvref_t<M>;
  using NV = std::remove_cvref_t<N>;
  using KV = std::remove_cvref_t<K>;
  if constexpr (!execution::has_resource_v<
                    execution::details::arm::StreamingZA, Scope>) {
#if defined(__ARM_FEATURE_SVE_MATMUL_INT8)
    if constexpr (mixed_sign_sve_skinny_candidate_v<
                      Atom, A, B, CInput, COutput> &&
                  meta::lower_bound_at_least_v<KV, 0> &&
                  extent_is_v<MV, 1> &&
                  meta::range_within_v<NV, 0, 64>) {
      return DispatchOwner::MixedSignSkinnyRow;
    } else if constexpr (mixed_sign_sve_skinny_candidate_v<
                             Atom, A, B, CInput, COutput> &&
                         meta::lower_bound_at_least_v<KV, 0> &&
                         extent_is_v<NV, 1> &&
                         meta::range_within_v<MV, 0, 64>) {
      return DispatchOwner::MixedSignSkinnyColumn;
    }
#endif
#if !defined(VECOPS_DISABLE_SME_SVE_SKINNY)
    if constexpr (sve_skinny_candidate_v<
                      Atom, A, B, CInput, COutput>) {
      constexpr nint_t MaxOutputs =
          std::same_as<Atom, ::vecops::matmul::SME_F16F32> ? 16 : 64;
      if constexpr (extent_is_v<MV, 1> &&
                    meta::range_within_v<NV, 0, MaxOutputs> &&
                    meta::lower_bound_at_least_v<KV, 0>) {
        return DispatchOwner::RawSkinnyRow;
      } else if constexpr (extent_is_v<NV, 1> &&
                           meta::range_within_v<MV, 0, MaxOutputs> &&
                           meta::lower_bound_at_least_v<KV, 0>) {
        return DispatchOwner::RawSkinnyColumn;
      }
    }
#endif
#if !defined(VECOPS_DISABLE_SME_SVE_SKINNY) && \
    !defined(VECOPS_DISABLE_SME_FUSED_SKINNY)
    if constexpr (sve_skinny_fused_candidate_v<
                      Atom, A, B, CInput, COutput>) {
      constexpr nint_t MaxOutputs =
          std::same_as<Atom, ::vecops::matmul::SME_F16F32> ? 16 : 64;
      if constexpr (extent_is_v<MV, 1> &&
                    meta::range_within_v<NV, 0, MaxOutputs> &&
                    meta::lower_bound_at_least_v<KV, 0>) {
        return DispatchOwner::FusedSkinnyRow;
      } else if constexpr (extent_is_v<NV, 1> &&
                           meta::range_within_v<MV, 0, MaxOutputs> &&
                           meta::lower_bound_at_least_v<KV, 0>) {
        return DispatchOwner::FusedSkinnyColumn;
      }
    }
#endif
#if defined(__ARM_FEATURE_SVE_MATMUL_INT8)
    if constexpr (fused_runtime_quant_int8_candidate_v<
                      Atom, A, B, CInput, COutput> &&
                  meta::lower_bound_at_least_v<MV, 1> &&
                  meta::lower_bound_at_least_v<NV, 1> &&
                  meta::lower_bound_at_least_v<KV, 0>) {
      return DispatchOwner::RuntimeQuantINT8;
    }
#endif
#if defined(CPU_CAPABILITY_SVE)
    if constexpr (packed_dot_candidate_v<
                      Atom, A, B, CInput, COutput>) {
      constexpr bool Elongated =
          (extent_is_v<MV, 2> && extent_is_v<NV, 8>) ||
          (extent_is_v<MV, 8> && extent_is_v<NV, 2>);
      constexpr bool Square =
          extent_is_v<MV, 4> && extent_is_v<NV, 4>;
      constexpr nint_t PrimaryMaxK = Square
          ? 257
          : (std::same_as<Atom, ::vecops::matmul::SME_BF16F32> ? 1025 : 513);
      if constexpr ((Elongated || Square) &&
                    meta::range_within_v<KV, 0, PrimaryMaxK>) {
        return DispatchOwner::PackedDotPrimary;
      }
      constexpr bool ShortWide =
          extent_is_v<MV, 2> &&
          (extent_is_v<NV, 2> || extent_is_v<NV, 4>);
      constexpr bool TallNarrow =
          extent_is_v<MV, 4> && extent_is_v<NV, 2>;
      constexpr nint_t TinyMaxK = TallNarrow
          ? 257
          : (std::same_as<Atom, ::vecops::matmul::SME_BF16F32> ? 1025 : 513);
      if constexpr ((ShortWide || TallNarrow) &&
                    meta::range_within_v<KV, 0, TinyMaxK>) {
        return DispatchOwner::PackedDotTiny;
      }
    }
#endif
  }
  return DispatchOwner::General;
}

/// Maps a dispatch owner to its selectable `kernel_family` tag (General,
/// SmallVector, RuntimeQuantInt8, PackedDot). Used to validate a family
/// selection against the owner the automatic policy would pick.
template <typename Family, DispatchOwner Owner>
inline constexpr bool dispatch_owner_in_family_v =
    (std::same_as<Family, ::vecops::matmul::kernel_family::General> &&
     Owner == DispatchOwner::General) ||
    (std::same_as<Family, ::vecops::matmul::kernel_family::SmallVector> &&
     (Owner == DispatchOwner::MixedSignSkinnyRow ||
      Owner == DispatchOwner::MixedSignSkinnyColumn ||
      Owner == DispatchOwner::RawSkinnyRow ||
      Owner == DispatchOwner::RawSkinnyColumn ||
      Owner == DispatchOwner::FusedSkinnyRow ||
      Owner == DispatchOwner::FusedSkinnyColumn)) ||
    (std::same_as<
         Family, ::vecops::matmul::kernel_family::RuntimeQuantInt8> &&
     Owner == DispatchOwner::RuntimeQuantINT8) ||
    (std::same_as<Family, ::vecops::matmul::kernel_family::PackedDot> &&
     (Owner == DispatchOwner::PackedDotPrimary ||
      Owner == DispatchOwner::PackedDotTiny));

/// Tri-state support verdict for a public family on this leaf's static
/// facts: General/WholeProblem always apply; every special family needs
/// its candidate predicate, a scope that does not already own
/// StreamingZA, and its shape probe. run() uses it to honour Prefer/
/// Require requests the automatic owner does not already satisfy.
template <typename Family,
          ::vecops::matmul::Atom Atom,
          meta::ValueType M, meta::ValueType N, meta::ValueType K,
          typename A, typename B, typename CInput, typename COutput,
          typename Scope>
consteval ::vecops::matmul::details::Applicability
family_applicability() {
  using Applicability = ::vecops::matmul::details::Applicability;
  constexpr bool MixedSignCandidate =
#if defined(__ARM_FEATURE_SVE_MATMUL_INT8)
      mixed_sign_sve_skinny_candidate_v<Atom, A, B, CInput, COutput>;
#else
      false;
#endif
  constexpr bool RawCandidate =
#if !defined(VECOPS_DISABLE_SME_SVE_SKINNY)
      sve_skinny_candidate_v<Atom, A, B, CInput, COutput>;
#else
      false;
#endif
  constexpr bool FusedCandidate =
#if !defined(VECOPS_DISABLE_SME_SVE_SKINNY) && \
    !defined(VECOPS_DISABLE_SME_FUSED_SKINNY)
      sve_skinny_fused_candidate_v<Atom, A, B, CInput, COutput>;
#else
      false;
#endif
  constexpr bool RuntimeQuantCandidate =
#if defined(__ARM_FEATURE_SVE_MATMUL_INT8)
      fused_runtime_quant_int8_candidate_v<
          Atom, A, B, CInput, COutput>;
#else
      false;
#endif
  constexpr bool PackedDotCandidate =
#if defined(CPU_CAPABILITY_SVE)
      packed_dot_candidate_v<Atom, A, B, CInput, COutput>;
#else
      false;
#endif
  if constexpr (std::same_as<
                    Family, ::vecops::matmul::kernel_family::General> ||
                std::same_as<
                    Family, ::vecops::matmul::kernel_family::WholeProblem>) {
    return Applicability::always;
  } else if constexpr (!execution::has_resource_v<
                           execution::details::arm::StreamingZA, Scope> &&
                       std::same_as<
                           Family,
                           ::vecops::matmul::kernel_family::SmallVector> &&
                       (MixedSignCandidate || RawCandidate ||
                        FusedCandidate)) {
    return small_vector_shape_applicability<Atom, M, N, K>();
  } else if constexpr (!execution::has_resource_v<
                           execution::details::arm::StreamingZA, Scope> &&
                       std::same_as<
                           Family,
                           ::vecops::matmul::kernel_family::RuntimeQuantInt8> &&
                       RuntimeQuantCandidate) {
    return runtime_quant_shape_applicability<M, N, K>();
  } else if constexpr (!execution::has_resource_v<
                           execution::details::arm::StreamingZA, Scope> &&
                       std::same_as<
                           Family,
                           ::vecops::matmul::kernel_family::PackedDot> &&
                       PackedDotCandidate) {
    return packed_dot_shape_applicability<Atom, M, N, K>();
  } else {
    return Applicability::never;
  }
}

/// Whether the runtime-dispatch tier may route into CandidateFamily
/// under this request: WholeProblem always permits it, General never
/// (the ZA path is already the owner), Require permits only the exact
/// required family, and Prefer permits everything so it can fall back to
/// the complete automatic route.
template <typename FamilyDispatch, typename CandidateFamily>
inline constexpr bool permits_runtime_family_v = [] {
  using Requested = typename FamilyDispatch::Family;
  if constexpr (std::same_as<
                    Requested,
                    ::vecops::matmul::kernel_family::WholeProblem>) {
    return true;
  } else if constexpr (std::same_as<
                           Requested,
                           ::vecops::matmul::kernel_family::General>) {
    return false;
  } else if constexpr (FamilyDispatch::required) {
    return std::same_as<Requested, CandidateFamily>;
  } else {
    return true;
  }
}();

/**
 * @brief Final owner selection honoring a `FamilyDispatch` selection.
 *
 * `WholeProblem` (the automatic planner) keeps the automatic owner;
 * `General` pins the ZA kernel outright; any other selected family must
 * agree with the automatic owner (`dispatch_owner_in_family_v`) or be
 * applicable through its shape probe (`family_applicability`) -- a
 * `required` mismatch fails a static_assert, while a prefer/automatic
 * mismatch silently runs the automatic owner instead (the preferred leaf
 * only wins when the shapes also match its gates).
 */
template <typename FamilyDispatch,
          ::vecops::matmul::Atom Atom,
          meta::ValueType M, meta::ValueType N, meta::ValueType K,
          typename A, typename B, typename CInput, typename COutput,
          typename Scope>
consteval DispatchOwner select_dispatch_owner() {
  constexpr auto AutomaticOwner = select_automatic_dispatch_owner<
      Atom, M, N, K, A, B, CInput, COutput, Scope>();
  using Family = typename FamilyDispatch::Family;
  if constexpr (std::same_as<
                    Family, ::vecops::matmul::kernel_family::WholeProblem>) {
    return AutomaticOwner;
  } else if constexpr (std::same_as<
                           Family,
                           ::vecops::matmul::kernel_family::General>) {
    return DispatchOwner::General;
  } else {
    constexpr bool Applicable = dispatch_owner_in_family_v<
        Family, AutomaticOwner> ||
        family_applicability<
            Family, Atom, M, N, K, A, B, CInput, COutput, Scope>() !=
            ::vecops::matmul::details::Applicability::never;
    static_assert(
        !FamilyDispatch::required || Applicable,
        "required matmul kernel family is not applicable to this SME leaf");
    return AutomaticOwner;
  }
}

/**
 * @brief FastPacked compute loop: direct packed-pointer outer products.
 *
 * Walks K groups stepping raw pointers into the packed 4-D layout (no
 * per-group offset arithmetic), issuing one `mopa` per output tile. Three
 * schedules by block shape, all with the same tile-numbering rule as
 * `compute_group` (tile = MI*NN + NI):
 *
 * - `NN == 1`: B's single block loaded once per group; each of the NM A
 *   blocks gets its own ZA tile.
 * - `NM == 1`: symmetric, A shared, NN B blocks in tiles 0..NN-1.
 * - `NM == 2 && NN == 2`: full 2x2 tile grid from four pointers; this is
 *   the shape the optional look-ahead prefetch below is tuned for.
 *
 * @tparam PrefetchLargeWorkingSet  Emit the L2 look-ahead loop (see
 *         `large_packed_prefetch_v`); only compiled for the 2x2 schedule.
 */
template <::vecops::matmul::Atom Atom, int NM, int NN,
          bool PrefetchLargeWorkingSet>
VECOPS_ALWAYS_INLINE void compute_packed_pointer_groups(
    const typename Atom::TA* a0, const typename Atom::TA* a1,
    const typename Atom::TA* a2, const typename Atom::TA* a3,
    const typename Atom::TB* b0, const typename Atom::TB* b1,
    const typename Atom::TB* b2, const typename Atom::TB* b3,
    nint_t a_step, nint_t b_step, nint_t groups) {
  using TA = typename Atom::TA;
  using TB = typename Atom::TB;
  using ATag = vec::ScalableTag<TA, 0>;
  using BTag = vec::ScalableTag<TB, 0>;

  if constexpr (NN == 1) {
    for (nint_t kg = 0; kg < groups; ++kg) {
      const auto bv = vec::load(BTag{}, b0 + kg * b_step);
      mopa<Atom, 0>(vec::load(ATag{}, a0 + kg * a_step), bv);
      if constexpr (NM >= 2)
        mopa<Atom, 1>(vec::load(ATag{}, a1 + kg * a_step), bv);
      if constexpr (NM >= 3)
        mopa<Atom, 2>(vec::load(ATag{}, a2 + kg * a_step), bv);
      if constexpr (NM >= 4)
        mopa<Atom, 3>(vec::load(ATag{}, a3 + kg * a_step), bv);
    }
  } else if constexpr (NM == 1) {
    for (nint_t kg = 0; kg < groups; ++kg) {
      const auto av = vec::load(ATag{}, a0 + kg * a_step);
      mopa<Atom, 0>(av, vec::load(BTag{}, b0 + kg * b_step));
      mopa<Atom, 1>(av, vec::load(BTag{}, b1 + kg * b_step));
      if constexpr (NN >= 3)
        mopa<Atom, 2>(av, vec::load(BTag{}, b2 + kg * b_step));
      if constexpr (NN >= 4)
        mopa<Atom, 3>(av, vec::load(BTag{}, b3 + kg * b_step));
    }
  } else {
    static_assert(NM == 2 && NN == 2);
#if !defined(VECOPS_DISABLE_SME_LARGE_PACKED_PREFETCH)
      constexpr nint_t PrefetchDistance = 48;
      nint_t kg = 0;
      if constexpr (PrefetchLargeWorkingSet) {
        if (groups > PrefetchDistance) {
          // Look-ahead prologue: the first (groups - PrefetchDistance)
          // iterations prefetch the K groups that are PrefetchDistance
          // ahead, so the steady state finds its loads in L2. The final
          // PrefetchDistance iterations (the drain loop below) stop
          // prefetching -- there is nothing left ahead to fetch.
          const nint_t prefetch_groups = groups - PrefetchDistance;
          VECOPS_LOOP_ALIGN(64) for (; kg < prefetch_groups; ++kg) {
            vec::prefetch(
                ATag{}, a0 + (kg + PrefetchDistance) * a_step,
                vec::mem::prefetch_l2, vec::mem::prefetch_keep,
                vec::mem::prefetch_read);
            vec::prefetch(
                ATag{}, a1 + (kg + PrefetchDistance) * a_step,
                vec::mem::prefetch_l2, vec::mem::prefetch_keep,
                vec::mem::prefetch_read);
            vec::prefetch(
                BTag{}, b0 + (kg + PrefetchDistance) * b_step,
                vec::mem::prefetch_l2, vec::mem::prefetch_keep,
                vec::mem::prefetch_read);
            vec::prefetch(
                BTag{}, b1 + (kg + PrefetchDistance) * b_step,
                vec::mem::prefetch_l2, vec::mem::prefetch_keep,
                vec::mem::prefetch_read);
            const auto av0 = vec::load(ATag{}, a0 + kg * a_step);
            const auto av1 = vec::load(ATag{}, a1 + kg * a_step);
            const auto bv0 = vec::load(BTag{}, b0 + kg * b_step);
            const auto bv1 = vec::load(BTag{}, b1 + kg * b_step);
            mopa<Atom, 0>(av0, bv0);
            mopa<Atom, 1>(av0, bv1);
            mopa<Atom, 2>(av1, bv0);
            mopa<Atom, 3>(av1, bv1);
          }
        }
      }
      VECOPS_LOOP_ALIGN(64) for (; kg < groups; ++kg) {
#else
      VECOPS_LOOP_ALIGN(64) for (nint_t kg = 0; kg < groups; ++kg) {
#endif
        const auto av0 = vec::load(ATag{}, a0 + kg * a_step);
        const auto av1 = vec::load(ATag{}, a1 + kg * a_step);
        const auto bv0 = vec::load(BTag{}, b0 + kg * b_step);
        const auto bv1 = vec::load(BTag{}, b1 + kg * b_step);
        mopa<Atom, 0>(av0, bv0);
        mopa<Atom, 1>(av0, bv1);
        mopa<Atom, 2>(av1, bv0);
        mopa<Atom, 3>(av1, bv1);
      }
  }
}

/**
 * @brief Meta/layout adapter for the packed-pointer compute loop.
 *
 * Pointer discovery remains inline because it benefits from the packed
 * layout's compile-time strides.  The K loop also remains inline for
 * established long-K plans; the enclosing tile body controls selective
 * out-of-line sharing for the additional shallow-K plan.
 */
template <::vecops::matmul::Atom Atom, int NM, int NN,
          bool PrefetchLargeWorkingSet,
          typename A, typename B>
VECOPS_ALWAYS_INLINE void compute_packed_groups(
    const A& a, const B& b, nint_t m, nint_t n, nint_t logical_k) {
  static_assert(is_packed_access_v<Atom, ::vecops::matmul::Operand::A, A>);
  static_assert(is_packed_access_v<Atom, ::vecops::matmul::Operand::B, B>);
  constexpr nint_t KP = ::vecops::matmul::packing_t<
      Atom, ::vecops::matmul::Operand::A>::KPack;
  const nint_t groups = ceil_div(logical_k, KP);
  const nint_t a_step = static_cast<nint_t>(tensor::stride<1>(
      a.spec().input_layout()));
  const nint_t b_step = static_cast<nint_t>(tensor::stride<1>(
      b.spec().input_layout()));

  const auto* a0 = packed_block_pointer<
      Atom, ::vecops::matmul::Operand::A, 0>(a, m);
  const auto* b0 = packed_block_pointer<
      Atom, ::vecops::matmul::Operand::B, 0>(b, n);
  auto* a1 = a0;
  auto* a2 = a0;
  auto* a3 = a0;
  auto* b1 = b0;
  auto* b2 = b0;
  auto* b3 = b0;
  if constexpr (NM >= 2)
    a1 = packed_block_pointer<
        Atom, ::vecops::matmul::Operand::A, 1>(a, m);
  if constexpr (NM >= 3)
    a2 = packed_block_pointer<
        Atom, ::vecops::matmul::Operand::A, 2>(a, m);
  if constexpr (NM >= 4)
    a3 = packed_block_pointer<
        Atom, ::vecops::matmul::Operand::A, 3>(a, m);
  if constexpr (NN >= 2)
    b1 = packed_block_pointer<
        Atom, ::vecops::matmul::Operand::B, 1>(b, n);
  if constexpr (NN >= 3)
    b2 = packed_block_pointer<
        Atom, ::vecops::matmul::Operand::B, 2>(b, n);
  if constexpr (NN >= 4)
    b3 = packed_block_pointer<
        Atom, ::vecops::matmul::Operand::B, 3>(b, n);
  compute_packed_pointer_groups<Atom, NM, NN, PrefetchLargeWorkingSet>(
      a0, a1, a2, a3, b0, b1, b2, b3, a_step, b_step, groups);
}

/// Horizontal ZA slice write/read through the ZA move intrinsics (used by
/// the non-direct C init/store fallbacks).
template <int Tile, typename T, vec::VectorValue V, typename Mask>
VECOPS_ALWAYS_INLINE void write_row(
    uint32_t row, Mask pg, V value) noexcept {
  using Tag = vec::ScalableTag<T, 0>;
  vec::details::sme::write_hor<Tile>(
      row, static_cast<vec::Mask<Tag>>(pg),
      static_cast<vec::Vec<Tag>>(value));
}

template <int Tile, typename T, typename Mask>
VECOPS_ALWAYS_INLINE auto read_row(
    uint32_t row, Mask pg) noexcept {
  using Tag = vec::ScalableTag<T, 0>;
  return vec::details::sme::read_hor<Tile>(
      Tag{}, row, static_cast<vec::Mask<Tag>>(pg));
}

/// Vertical ZA slice write/read through the ZA move intrinsics: the
/// column-major twins of write_row/read_row, serving direct
/// column-contiguous (transposed) C epilogues.
template <int Tile, typename T, vec::VectorValue V, typename Mask>
VECOPS_ALWAYS_INLINE void write_column(
    uint32_t column, Mask pg, V value) noexcept {
  using Tag = vec::ScalableTag<T, 0>;
  vec::details::sme::write_ver<Tile>(
      column, static_cast<vec::Mask<Tag>>(pg),
      static_cast<vec::Vec<Tag>>(value));
}

template <int Tile, typename T, typename Mask>
VECOPS_ALWAYS_INLINE auto read_column(
    uint32_t column, Mask pg) noexcept {
  using Tag = vec::ScalableTag<T, 0>;
  return vec::details::sme::read_ver<Tile>(
      Tag{}, column, static_cast<vec::Mask<Tag>>(pg));
}

/**
 * @brief Compile-time extent of one spatial block within its ZA tile.
 *
 * Full blocks return the tile extent `M_R` unchanged (keeping every
 * compile-time guarantee it carries). Partial blocks clamp
 * `active - Block*lanes` into [0, lanes]; when the tile extent has an
 * upper bound the result is spelled as a `Dynamic<1, Lo, Hi>` that
 * preserves the tile's bounds -- `NonEmpty` raises the lower bound to 1 for
 * blocks whose existence is already guaranteed.
 */
template <::vecops::matmul::Atom Atom, bool Full, bool NonEmpty, int Block>
VECOPS_ALWAYS_INLINE auto tile_active_extent(nint_t active) {
  using Tile = std::remove_cvref_t<decltype(Atom::M_R)>;
  if constexpr (Full) {
    return Atom::M_R;
  } else {
    const nint_t lanes = static_cast<nint_t>(Atom::M_R);
    const nint_t value = vecops::clamp(
        active - static_cast<nint_t>(Block) * lanes,
        nint_t{0}, lanes);
    if constexpr (meta::has_upper_bound_v<Tile>) {
      constexpr nint_t Lo = NonEmpty ? 1 : 0;
      constexpr nint_t Hi = meta::upper_bound_v<Tile>;
      // Construct directly so BiSheng/Clang cannot outline the tiny dyn()
      // factory inside an active SME interval.
      return meta::Dynamic<1, Lo, Hi>{value};
    } else {
      return meta::Any{value};
    }
  }
}

/**
 * @brief Seed one ZA output tile with the C input before accumulation.
 *
 * A zero-valued C input needs nothing (ZA was just zeroed). Otherwise the
 * goal is to place the C block into the tile, and how depends on the C
 * layout, taking the cheapest instruction that covers the active rows:
 *
 * - broadcast row (`stride<0> == Const<0>`) + fp: one `mopa` of an
 *   all-ones vector with the C row -- an outer product that *multiplies by
 *   1*, adding `value` into every selected row of the zeroed tile.
 * - broadcast row + 32-bit integer: one `addha` -- SME's horizontal
 *   accumulate-add adds the vector into every selected row; a single row
 *   (M==1) is cheaper as one direct `load_hor` into row 0.
 * - non-broadcast direct row-major: per-row ZA memory loads (`load_hor`
 *   of the 32/64-bit container width).
 * - column-contiguous (transposed) input: the column twin of the last
 *   case -- per-column vector loads through the access layer plus
 *   `write_column`.
 * - anything else: per-row vector loads through the access layer plus
 *   `write_row`.
 *
 * @tparam FullM/FullN  Whether the whole spatial block is active (selects
 *                      all-true row/column predicates).
 */
template <int Tile, bool FullM, bool FullN,
          meta::ValueType ActiveM, meta::ValueType ActiveN,
          typename CInput>
VECOPS_ALWAYS_INLINE void initialize_c_tile(
    const CInput& input, nint_t m, nint_t n,
    ActiveM active_m, ActiveN active_n) {
  using T = typename CInput::ComputeType;
  if constexpr (IsZeroTransform<typename CInput::Transform>::value) {
    return;
  }
  using Tag = vec::ScalableTag<T, 0>;
  const nint_t active_m_value = static_cast<nint_t>(active_m);
  const nint_t active_n_value = static_cast<nint_t>(active_n);
  const auto pg = [&]() VECOPS_INLINE_LAMBDA {
    if constexpr (FullN) return vec::mtrue(Tag{});
    else return vec::mwhilelt(Tag{}, nint_t{0}, active_n_value);
  }();
  if constexpr (direct_row_major_input_v<CInput>) {
    const auto strides = input.raw_strides();
    const auto* base = reinterpret_cast<const T*>(input.raw_data()) + m * strides[0] + n;
    if constexpr (
        std::is_floating_point_v<T> &&
        meta::range_within_v<
            tensor::stride_type_t<0, InputLayoutOf<CInput>>, 0, 0>) {
      // ZA pre-seed trick (fp): outer product with all-ones multiplies the
      // broadcast C row by 1 into every active row -- see the function
      // comment; there is no fp horizontal-add into ZA.
      const auto value = vec::load(
          Tag{}, base, vec::opt::masked(pg), vec::opt::zero);
      const auto pg_rows = [&]() VECOPS_INLINE_LAMBDA {
        if constexpr (FullM) return vec::mtrue(Tag{});
        else return vec::mwhilelt(Tag{}, nint_t{0}, active_m_value);
      }();
      vec::details::sme::mopa<Tile, T, T, T>(
          pg_rows, pg, vec::fill(Tag{}, T{1}), value);
    } else if constexpr (
        (std::same_as<T, int32_t> || std::same_as<T, uint32_t>) &&
        meta::range_within_v<
            tensor::stride_type_t<0, InputLayoutOf<CInput>>, 0, 0>) {
      if constexpr (meta::is_singleton_v<ActiveM> &&
                    meta::singleton_value_v<ActiveM> == 1) {
        using U = std::conditional_t<std::same_as<T, int32_t>, uint32_t, T>;
        using BitsTag = vec::ScalableTag<U, 0>;
        vec::details::sme::load_hor<Tile>(
            0, static_cast<vec::Mask<BitsTag>>(pg),
            reinterpret_cast<const U*>(base));
      } else {
        // ZA pre-seed trick (integer): ADDHA adds the broadcast row into
        // every active row of the tile in one instruction.
        const auto value = vec::load(
            Tag{}, base, vec::opt::masked(pg), vec::opt::zero);
        const auto pg_rows = [&]() VECOPS_INLINE_LAMBDA {
          if constexpr (FullM) return vec::mtrue(Tag{});
          else return vec::mwhilelt(Tag{}, nint_t{0}, active_m_value);
        }();
        vec::details::sme::addha<Tile, T>(pg_rows, pg, value);
      }
    } else {
      for (nint_t row = 0; row < active_m_value; ++row) {
        using U = std::conditional_t<sizeof(T) == 8, uint64_t, uint32_t>;
        using BitsTag = vec::ScalableTag<U, 0>;
        vec::details::sme::load_hor<Tile>(
            static_cast<uint32_t>(row),
            static_cast<vec::Mask<BitsTag>>(pg),
            reinterpret_cast<const U*>(base + row * strides[0]));
      }
    }
  } else if constexpr (column_contiguous_input_v<CInput>) {
    const auto pg_rows = [&]() VECOPS_INLINE_LAMBDA {
      if constexpr (FullM) return vec::mtrue(Tag{});
      else return vec::mwhilelt(Tag{}, nint_t{0}, active_m_value);
    }();
    for (nint_t column = 0; column < active_n_value; ++column) {
      const auto value = input.load(
          Tag{}, tensor::coord(m, n + column), tensor::axis<0>,
          vec::opt::first(active_m_value), vec::opt::zero);
      write_column<Tile, T>(
          static_cast<uint32_t>(column), pg_rows, value);
    }
  } else {
    for (nint_t row = 0; row < active_m_value; ++row) {
      const auto value = input.load(
          Tag{}, tensor::coord(m + row, n), tensor::axis<1>,
          vec::opt::first(active_n_value), vec::opt::zero);
      write_row<Tile, T>(static_cast<uint32_t>(row), pg, value);
    }
  }
}

/**
 * @brief Write one ZA output tile back through the C output access.
 *
 * Direct row-major outputs use per-row ZA memory stores (`store_hor` of
 * the 32/64-bit container width); direct column-major (transposed)
 * outputs use the vertical twin (`store_ver`, per column); a
 * column-contiguous non-direct output reads each column back into a
 * vector and stores through the access layer along axis<0>; anything
 * else reads each row back into a vector and stores through the access
 * layer (applying the output transform, if any).
 */
template <int Tile, bool FullM, bool FullN,
          meta::ValueType ActiveM, meta::ValueType ActiveN,
          typename COutput>
VECOPS_ALWAYS_INLINE void store_c_tile(
    COutput& output, nint_t m, nint_t n,
    ActiveM active_m, ActiveN active_n) {
  using T = typename COutput::ComputeType;
  using Tag = vec::ScalableTag<T, 0>;
  const nint_t active_m_value = static_cast<nint_t>(active_m);
  const nint_t active_n_value = static_cast<nint_t>(active_n);
  const auto pg = [&]() VECOPS_INLINE_LAMBDA {
    if constexpr (FullN) return vec::mtrue(Tag{});
    else return vec::mwhilelt(Tag{}, nint_t{0}, active_n_value);
  }();
  if constexpr (direct_row_major_output_v<COutput>) {
    const auto strides = output.raw_strides();
    auto* base = reinterpret_cast<T*>(output.raw_data()) + m * strides[0] + n;
    for (nint_t row = 0; row < active_m_value; ++row) {
      using U = std::conditional_t<sizeof(T) == 8, uint64_t, uint32_t>;
      using BitsTag = vec::ScalableTag<U, 0>;
      vec::details::sme::store_hor<Tile>(
          static_cast<uint32_t>(row),
          static_cast<vec::Mask<BitsTag>>(pg),
          reinterpret_cast<U*>(base + row * strides[0]));
    }
  } else if constexpr (direct_column_major_output_v<COutput>) {
    const auto strides = output.raw_strides();
    auto* base = reinterpret_cast<T*>(output.raw_data()) +
        m * strides[0] + n * strides[1];
    const auto pg_rows = [&]() VECOPS_INLINE_LAMBDA {
      if constexpr (FullM) return vec::mtrue(Tag{});
      else return vec::mwhilelt(Tag{}, nint_t{0}, active_m_value);
    }();
    for (nint_t column = 0; column < active_n_value; ++column) {
      using U = std::conditional_t<sizeof(T) == 8, uint64_t, uint32_t>;
      using BitsTag = vec::ScalableTag<U, 0>;
      vec::details::sme::store_ver<Tile>(
          static_cast<uint32_t>(column),
          static_cast<vec::Mask<BitsTag>>(pg_rows),
          reinterpret_cast<U*>(base + column * strides[1]));
    }
  } else if constexpr (column_contiguous_output_v<COutput>) {
    const auto pg_rows = [&]() VECOPS_INLINE_LAMBDA {
      if constexpr (FullM) return vec::mtrue(Tag{});
      else return vec::mwhilelt(Tag{}, nint_t{0}, active_m_value);
    }();
    for (nint_t column = 0; column < active_n_value; ++column) {
      const auto value = static_cast<vec::Vec<Tag>>(
          read_column<Tile, T>(static_cast<uint32_t>(column), pg_rows));
      output.store(
          Tag{}, tensor::coord(m, n + column), tensor::axis<0>, value,
          vec::opt::first(active_m_value));
    }
  } else {
    for (nint_t row = 0; row < active_m_value; ++row) {
      const auto value = static_cast<vec::Vec<Tag>>(
          read_row<Tile, T>(static_cast<uint32_t>(row), pg));
      output.store(
          Tag{}, tensor::coord(m + row, n), tensor::axis<1>, value,
          vec::opt::first(active_n_value));
    }
  }
}

template <int LeftTile, int RightTile, bool FullM, bool FullLeftN, bool FullRightN, meta::ValueType ActiveM,
          meta::ValueType LeftActiveN, meta::ValueType RightActiveN, typename CInput>
VECOPS_ALWAYS_INLINE void initialize_c_tile_row_pair(const CInput& input, nint_t m, nint_t n, ActiveM active_m,
                                                     LeftActiveN left_active_n, RightActiveN right_active_n) {
  using T = typename CInput::ComputeType;
  if constexpr (IsZeroTransform<typename CInput::Transform>::value)
    return;
  using Tag = vec::ScalableTag<T, 0>;
  using WideTag = vec::Twice<Tag>;
  static_assert(tensor::preferred_memory_access_power_v<CInput> > 0);
  const nint_t active_m_value = static_cast<nint_t>(active_m);
  const nint_t left_n = static_cast<nint_t>(left_active_n);
  const nint_t right_n = static_cast<nint_t>(right_active_n);
  const nint_t active_n = left_n + right_n;
  const auto left_pg = FullLeftN ? vec::mtrue(Tag{}) : vec::mwhilelt(Tag{}, nint_t{0}, left_n);
  const auto right_pg = FullRightN ? vec::mtrue(Tag{}) : vec::mwhilelt(Tag{}, nint_t{0}, right_n);
  for (nint_t row = 0; row < active_m_value; ++row) {
    const auto value = [&]() VECOPS_INLINE_LAMBDA {
      if constexpr (FullLeftN && FullRightN) {
        return input.load(WideTag{}, tensor::coord(m + row, n), tensor::axis<1>);
      } else {
        return input.load(WideTag{}, tensor::coord(m + row, n), tensor::axis<1>, vec::opt::first(active_n),
                          vec::opt::zero);
      }
    }();
    write_row<LeftTile, T>(static_cast<uint32_t>(row), left_pg, vec::lower(WideTag{}, value));
    write_row<RightTile, T>(static_cast<uint32_t>(row), right_pg, vec::upper(WideTag{}, value));
  }
}

template <int TopTile, int BottomTile, bool FullTopM, bool FullBottomM, bool FullN, meta::ValueType TopActiveM,
          meta::ValueType BottomActiveM, meta::ValueType ActiveN, typename CInput>
VECOPS_ALWAYS_INLINE void initialize_c_tile_column_pair(const CInput& input, nint_t m, nint_t n,
                                                        TopActiveM top_active_m, BottomActiveM bottom_active_m,
                                                        ActiveN active_n) {
  using T = typename CInput::ComputeType;
  if constexpr (IsZeroTransform<typename CInput::Transform>::value)
    return;
  using Tag = vec::ScalableTag<T, 0>;
  using WideTag = vec::Twice<Tag>;
  static_assert(tensor::preferred_memory_access_power_v<CInput> > 0);
  const nint_t top_m = static_cast<nint_t>(top_active_m);
  const nint_t bottom_m = static_cast<nint_t>(bottom_active_m);
  const nint_t active_m = top_m + bottom_m;
  const nint_t active_n_value = static_cast<nint_t>(active_n);
  const auto top_pg = FullTopM ? vec::mtrue(Tag{}) : vec::mwhilelt(Tag{}, nint_t{0}, top_m);
  const auto bottom_pg = FullBottomM ? vec::mtrue(Tag{}) : vec::mwhilelt(Tag{}, nint_t{0}, bottom_m);
  for (nint_t column = 0; column < active_n_value; ++column) {
    const auto value = [&]() VECOPS_INLINE_LAMBDA {
      if constexpr (FullTopM && FullBottomM) {
        return input.load(WideTag{}, tensor::coord(m, n + column), tensor::axis<0>);
      } else {
        return input.load(WideTag{}, tensor::coord(m, n + column), tensor::axis<0>, vec::opt::first(active_m),
                          vec::opt::zero);
      }
    }();
    write_column<TopTile, T>(static_cast<uint32_t>(column), top_pg, vec::lower(WideTag{}, value));
    write_column<BottomTile, T>(static_cast<uint32_t>(column), bottom_pg, vec::upper(WideTag{}, value));
  }
}

template <int LeftTile, int RightTile, bool FullM, bool FullLeftN, bool FullRightN, meta::ValueType ActiveM,
          meta::ValueType LeftActiveN, meta::ValueType RightActiveN, typename COutput>
VECOPS_ALWAYS_INLINE void store_c_tile_row_pair(COutput& output, nint_t m, nint_t n, ActiveM active_m,
                                                LeftActiveN left_active_n, RightActiveN right_active_n) {
  using T = typename COutput::ComputeType;
  using Tag = vec::ScalableTag<T, 0>;
  using WideTag = vec::Twice<Tag>;
  static_assert(tensor::preferred_memory_access_power_v<COutput> > 0);
  const nint_t active_m_value = static_cast<nint_t>(active_m);
  const nint_t left_n = static_cast<nint_t>(left_active_n);
  const nint_t right_n = static_cast<nint_t>(right_active_n);
  const nint_t active_n = left_n + right_n;
  const auto left_pg = FullLeftN ? vec::mtrue(Tag{}) : vec::mwhilelt(Tag{}, nint_t{0}, left_n);
  const auto right_pg = FullRightN ? vec::mtrue(Tag{}) : vec::mwhilelt(Tag{}, nint_t{0}, right_n);
  for (nint_t row = 0; row < active_m_value; ++row) {
    const auto left = static_cast<vec::Vec<Tag>>(read_row<LeftTile, T>(static_cast<uint32_t>(row), left_pg));
    const auto right = static_cast<vec::Vec<Tag>>(read_row<RightTile, T>(static_cast<uint32_t>(row), right_pg));
    if constexpr (transform_store_mode_v<COutput> == tensor::TransformStoreMode::coalesced) {
      static_assert(requires {
        output.template store_transform_pair_coalesced<FullLeftN, FullRightN>(
          Tag{}, tensor::coord(m + row, n), tensor::coord(m + row, n + vec::size(Tag{})), tensor::axis<1>, left, right,
          left_active_n, right_active_n);
      });
      output.template store_transform_pair_coalesced<FullLeftN, FullRightN>(
        Tag{}, tensor::coord(m + row, n), tensor::coord(m + row, n + vec::size(Tag{})), tensor::axis<1>, left, right,
        left_active_n, right_active_n);
    } else {
      const auto value = vec::concat(WideTag{}, left, right);
      if constexpr (FullLeftN && FullRightN) {
        output.store(WideTag{}, tensor::coord(m + row, n), tensor::axis<1>, value);
      } else {
        output.store(WideTag{}, tensor::coord(m + row, n), tensor::axis<1>, value, vec::opt::first(active_n));
      }
    }
  }
}

template <int TopTile, int BottomTile, bool FullTopM, bool FullBottomM, bool FullN, meta::ValueType TopActiveM,
          meta::ValueType BottomActiveM, meta::ValueType ActiveN, typename COutput>
VECOPS_ALWAYS_INLINE void store_c_tile_column_pair(COutput& output, nint_t m, nint_t n, TopActiveM top_active_m,
                                                   BottomActiveM bottom_active_m, ActiveN active_n) {
  using T = typename COutput::ComputeType;
  using Tag = vec::ScalableTag<T, 0>;
  using WideTag = vec::Twice<Tag>;
  static_assert(tensor::preferred_memory_access_power_v<COutput> > 0);
  const nint_t top_m = static_cast<nint_t>(top_active_m);
  const nint_t bottom_m = static_cast<nint_t>(bottom_active_m);
  const nint_t active_m = top_m + bottom_m;
  const nint_t active_n_value = static_cast<nint_t>(active_n);
  const auto top_pg = FullTopM ? vec::mtrue(Tag{}) : vec::mwhilelt(Tag{}, nint_t{0}, top_m);
  const auto bottom_pg = FullBottomM ? vec::mtrue(Tag{}) : vec::mwhilelt(Tag{}, nint_t{0}, bottom_m);
  for (nint_t column = 0; column < active_n_value; ++column) {
    const auto top = static_cast<vec::Vec<Tag>>(read_column<TopTile, T>(static_cast<uint32_t>(column), top_pg));
    const auto bottom =
      static_cast<vec::Vec<Tag>>(read_column<BottomTile, T>(static_cast<uint32_t>(column), bottom_pg));
    if constexpr (transform_store_mode_v<COutput> == tensor::TransformStoreMode::coalesced) {
      static_assert(requires {
        output.template store_transform_pair_coalesced<FullTopM, FullBottomM>(
          Tag{}, tensor::coord(m, n + column), tensor::coord(m + vec::size(Tag{}), n + column), tensor::axis<0>, top,
          bottom, top_active_m, bottom_active_m);
      });
      output.template store_transform_pair_coalesced<FullTopM, FullBottomM>(
        Tag{}, tensor::coord(m, n + column), tensor::coord(m + vec::size(Tag{}), n + column), tensor::axis<0>, top,
        bottom, top_active_m, bottom_active_m);
    } else {
      const auto value = vec::concat(WideTag{}, top, bottom);
      if constexpr (FullTopM && FullBottomM) {
        output.store(WideTag{}, tensor::coord(m, n + column), tensor::axis<0>, value);
      } else {
        output.store(WideTag{}, tensor::coord(m, n + column), tensor::axis<0>, value, vec::opt::first(active_m));
      }
    }
  }
}

template <int Tile, bool FullM, bool FullN, meta::ValueType ActiveM, meta::ValueType ActiveN, typename CInput,
          typename AccInput, typename Route>
VECOPS_ALWAYS_INLINE void initialize_c_tile_routed(const CInput& c_input, const AccInput& acc_input, const Route& route,
                                                   nint_t m, nint_t n, ActiveM active_m, ActiveN active_n) {
  static_assert(std::same_as<
      typename CInput::ComputeType, typename AccInput::ComputeType>);
  if constexpr (matmul_details::static_accumulator_route_v<Route>) {
    if constexpr (Route::use_accumulator_input)
      initialize_c_tile<Tile, FullM, FullN>(
          acc_input, m, n, active_m, active_n);
    else
      initialize_c_tile<Tile, FullM, FullN>(
          c_input, m, n, active_m, active_n);
  } else {
    if (route.use_accumulator_input)
      initialize_c_tile<Tile, FullM, FullN>(
          acc_input, m, n, active_m, active_n);
    else
      initialize_c_tile<Tile, FullM, FullN>(
          c_input, m, n, active_m, active_n);
  }
}

template <int Tile, bool FullM, bool FullN,
          meta::ValueType ActiveM, meta::ValueType ActiveN,
          typename COutput, typename AccOutput, typename Route>
VECOPS_ALWAYS_INLINE void store_c_tile_routed(
    COutput& c_output, AccOutput& acc_output, const Route& route,
    nint_t m, nint_t n, ActiveM active_m, ActiveN active_n) {
  static_assert(std::same_as<
      typename COutput::ComputeType, typename AccOutput::ComputeType>);
  if constexpr (matmul_details::static_accumulator_route_v<Route>) {
    if constexpr (Route::write_accumulator_output)
      store_c_tile<Tile, FullM, FullN>(
          acc_output, m, n, active_m, active_n);
    else
      store_c_tile<Tile, FullM, FullN>(
          c_output, m, n, active_m, active_n);
  } else {
    if (route.write_accumulator_output)
      store_c_tile<Tile, FullM, FullN>(acc_output, m, n, active_m, active_n);
    else
      store_c_tile<Tile, FullM, FullN>(c_output, m, n, active_m, active_n);
  }
}

template <int LeftTile, int RightTile, bool FullM, bool FullLeftN, bool FullRightN, meta::ValueType ActiveM,
          meta::ValueType LeftActiveN, meta::ValueType RightActiveN, typename CInput, typename AccInput, typename Route>
VECOPS_ALWAYS_INLINE void initialize_c_tile_row_pair_routed(const CInput& c_input, const AccInput& acc_input,
                                                            const Route& route, nint_t m, nint_t n, ActiveM active_m,
                                                            LeftActiveN left_active_n, RightActiveN right_active_n,
                                                            nint_t lanes) {
  auto initialize = [&]<typename Input>(const Input& input) VECOPS_INLINE_LAMBDA {
    if constexpr (tensor::preferred_memory_access_power_v<Input> > 0) {
      initialize_c_tile_row_pair<LeftTile, RightTile, FullM, FullLeftN, FullRightN>(input, m, n, active_m,
                                                                                    left_active_n, right_active_n);
    } else {
      initialize_c_tile<LeftTile, FullM, FullLeftN>(input, m, n, active_m, left_active_n);
      initialize_c_tile<RightTile, FullM, FullRightN>(input, m, n + lanes, active_m, right_active_n);
    }
  };
  if constexpr (matmul_details::static_accumulator_route_v<Route>) {
    if constexpr (Route::use_accumulator_input)
      initialize(acc_input);
    else
      initialize(c_input);
  } else {
    if (route.use_accumulator_input)
      initialize(acc_input);
    else
      initialize(c_input);
  }
}

template <int TopTile, int BottomTile, bool FullTopM, bool FullBottomM, bool FullN, meta::ValueType TopActiveM,
          meta::ValueType BottomActiveM, meta::ValueType ActiveN, typename CInput, typename AccInput, typename Route>
VECOPS_ALWAYS_INLINE void initialize_c_tile_column_pair_routed(const CInput& c_input, const AccInput& acc_input,
                                                               const Route& route, nint_t m, nint_t n,
                                                               TopActiveM top_active_m, BottomActiveM bottom_active_m,
                                                               ActiveN active_n, nint_t lanes) {
  auto initialize = [&]<typename Input>(const Input& input) VECOPS_INLINE_LAMBDA {
    if constexpr (tensor::preferred_memory_access_power_v<Input> > 0) {
      initialize_c_tile_column_pair<TopTile, BottomTile, FullTopM, FullBottomM, FullN>(input, m, n, top_active_m,
                                                                                       bottom_active_m, active_n);
    } else {
      initialize_c_tile<TopTile, FullTopM, FullN>(input, m, n, top_active_m, active_n);
      initialize_c_tile<BottomTile, FullBottomM, FullN>(input, m + lanes, n, bottom_active_m, active_n);
    }
  };
  if constexpr (matmul_details::static_accumulator_route_v<Route>) {
    if constexpr (Route::use_accumulator_input)
      initialize(acc_input);
    else
      initialize(c_input);
  } else {
    if (route.use_accumulator_input)
      initialize(acc_input);
    else
      initialize(c_input);
  }
}

template <int LeftTile, int RightTile, bool FullM, bool FullLeftN, bool FullRightN, meta::ValueType ActiveM,
          meta::ValueType LeftActiveN, meta::ValueType RightActiveN, typename COutput, typename AccOutput,
          typename Route>
VECOPS_ALWAYS_INLINE void store_c_tile_row_pair_routed(COutput& c_output, AccOutput& acc_output, const Route& route,
                                                       nint_t m, nint_t n, ActiveM active_m, LeftActiveN left_active_n,
                                                       RightActiveN right_active_n, nint_t lanes) {
  auto store = [&]<typename Output>(Output& output) VECOPS_INLINE_LAMBDA {
    if constexpr (tensor::preferred_memory_access_power_v<Output> > 0 && transform_store_groups_tiles_v<Output>) {
      store_c_tile_row_pair<LeftTile, RightTile, FullM, FullLeftN, FullRightN>(output, m, n, active_m, left_active_n,
                                                                               right_active_n);
    } else {
      store_c_tile<LeftTile, FullM, FullLeftN>(output, m, n, active_m, left_active_n);
      store_c_tile<RightTile, FullM, FullRightN>(output, m, n + lanes, active_m, right_active_n);
    }
  };
  if constexpr (matmul_details::static_accumulator_route_v<Route>) {
    if constexpr (Route::write_accumulator_output)
      store(acc_output);
    else
      store(c_output);
  } else {
    if (route.write_accumulator_output)
      store(acc_output);
    else
      store(c_output);
  }
}

template <int TopTile, int BottomTile, bool FullTopM, bool FullBottomM, bool FullN, meta::ValueType TopActiveM,
          meta::ValueType BottomActiveM, meta::ValueType ActiveN, typename COutput, typename AccOutput, typename Route>
VECOPS_ALWAYS_INLINE void store_c_tile_column_pair_routed(COutput& c_output, AccOutput& acc_output, const Route& route,
                                                          nint_t m, nint_t n, TopActiveM top_active_m,
                                                          BottomActiveM bottom_active_m, ActiveN active_n,
                                                          nint_t lanes) {
  auto store = [&]<typename Output>(Output& output) VECOPS_INLINE_LAMBDA {
    if constexpr (tensor::preferred_memory_access_power_v<Output> > 0 && transform_store_groups_tiles_v<Output>) {
      store_c_tile_column_pair<TopTile, BottomTile, FullTopM, FullBottomM, FullN>(output, m, n, top_active_m,
                                                                                  bottom_active_m, active_n);
    } else {
      store_c_tile<TopTile, FullTopM, FullN>(output, m, n, top_active_m, active_n);
      store_c_tile<BottomTile, FullBottomM, FullN>(output, m + lanes, n, bottom_active_m, active_n);
    }
  };
  if constexpr (matmul_details::static_accumulator_route_v<Route>) {
    if constexpr (Route::write_accumulator_output)
      store(acc_output);
    else
      store(c_output);
  } else {
    if (route.write_accumulator_output)
      store(acc_output);
    else
      store(c_output);
  }
}

/**
 * @brief Generic per-K-group compute: NM x NN outer products into ZA.
 *
 * Loads one operand vector per spatial block (via `load_operand`) and
 * issues one `mopa` per output tile. The five schedule branches all obey
 * the same tile-numbering rule -- tile = MI*NN + NI for spatial blocks
 * (MI, NI) -- which matches how `microkernel` maps linear tile I to output
 * position (I/NN, I%NN):
 *
 * - `NN == 1` / `NM == 1`: degenerate one-dimensional schedules (tiles
 *   0..NM-1 / 0..NN-1).
 * - `NM == 2 && NN == 2`: the plain 2x2 grid, tiles 0..3.
 * - `NN == 2` (NM > 2): tiles assigned per M block as `2*MI` and `2*MI+1`
 *   -- consecutive tile numbers *row-interleave* across M blocks.
 * - `NM == 2` (NN > 2): tiles assigned per N block as `NI` and `NN+NI` --
 *   consecutive tile numbers *column-interleave* across N blocks.
 *
 * `ExactBlocks` proves only the blocks before the last one are full. The
 * final block on either axis still needs its tail predicate.
 *
 * The `Full*` template flags reflect a dual existence proof: a block is
 * provably full either by the case's `ExactBlocks` promise (all blocks
 * except the last per axis exist fully) or by position within the logical
 * extent (`load_operand`'s runtime clamp still guards the rest).
 */
template <::vecops::matmul::Atom Atom, int NM, int NN,
          bool FullM, bool FullN, bool ExactBlocks, bool FullK,
          typename A, typename B>
VECOPS_ALWAYS_INLINE void compute_group(
    const A& a, const OperandInvariants<Atom, ::vecops::matmul::Operand::A, A>& a_invariants,
    const B& b, const OperandInvariants<Atom, ::vecops::matmul::Operand::B, B>& b_invariants,
    nint_t m, nint_t n, nint_t kg,
    nint_t logical_m, nint_t logical_n, nint_t logical_k) {
  if constexpr (NN == 1) {
    const auto bv = load_operand<
        Atom, ::vecops::matmul::Operand::B, 0, FullN, FullK>(
        b, b_invariants, n, kg, logical_n, logical_k);
    auto one_m = [&]<int MI>() VECOPS_INLINE_LAMBDA {
      const auto av = load_operand<
          Atom, ::vecops::matmul::Operand::A, MI,
          FullM || (ExactBlocks && MI + 1 < NM), FullK>(
          a, a_invariants, m, kg, logical_m, logical_k);
      mopa<Atom, MI>(av, bv);
    };
    [&]<std::size_t... MI>(std::index_sequence<MI...>) VECOPS_INLINE_LAMBDA {
      (one_m.template operator()<static_cast<int>(MI)>(), ...);
    }(std::make_index_sequence<NM>{});
  } else if constexpr (NM == 1) {
    const auto av = load_operand<
        Atom, ::vecops::matmul::Operand::A, 0, FullM, FullK>(
        a, a_invariants, m, kg, logical_m, logical_k);
    auto one_n = [&]<int NI>() VECOPS_INLINE_LAMBDA {
      const auto bv = load_operand<
          Atom, ::vecops::matmul::Operand::B, NI,
          FullN || (ExactBlocks && NI + 1 < NN), FullK>(
          b, b_invariants, n, kg, logical_n, logical_k);
      mopa<Atom, NI>(av, bv);
    };
    [&]<std::size_t... NI>(std::index_sequence<NI...>) VECOPS_INLINE_LAMBDA {
      (one_n.template operator()<static_cast<int>(NI)>(), ...);
    }(std::make_index_sequence<NN>{});
  } else if constexpr (NM == 2 && NN == 2) {
    const auto a0 = load_operand<
        Atom, ::vecops::matmul::Operand::A, 0, FullM, FullK>(
        a, a_invariants, m, kg, logical_m, logical_k);
    const auto a1 = load_operand<
        Atom, ::vecops::matmul::Operand::A, 1,
        FullM, FullK>(
        a, a_invariants, m, kg, logical_m, logical_k);
    const auto b0 = load_operand<
        Atom, ::vecops::matmul::Operand::B, 0, FullN, FullK>(
        b, b_invariants, n, kg, logical_n, logical_k);
    const auto b1 = load_operand<
        Atom, ::vecops::matmul::Operand::B, 1,
        FullN, FullK>(
        b, b_invariants, n, kg, logical_n, logical_k);
    mopa<Atom, 0>(a0, b0);
    mopa<Atom, 1>(a0, b1);
    mopa<Atom, 2>(a1, b0);
    mopa<Atom, 3>(a1, b1);
  } else if constexpr (NN == 2) {
    const auto b0 = load_operand<
        Atom, ::vecops::matmul::Operand::B, 0, FullN, FullK>(
        b, b_invariants, n, kg, logical_n, logical_k);
    const auto b1 = load_operand<
        Atom, ::vecops::matmul::Operand::B, 1,
        FullN, FullK>(
        b, b_invariants, n, kg, logical_n, logical_k);
    auto one_m = [&]<int MI>() VECOPS_INLINE_LAMBDA {
      const auto av = load_operand<
          Atom, ::vecops::matmul::Operand::A, MI,
          FullM || (ExactBlocks && MI + 1 < NM), FullK>(
          a, a_invariants, m, kg, logical_m, logical_k);
      // Tile numbers 2*MI, 2*MI+1: row-interleaved across M blocks
      // (tile = MI*NN + NI with NN == 2).
      mopa<Atom, 2 * MI>(av, b0);
      mopa<Atom, 2 * MI + 1>(av, b1);
    };
    [&]<std::size_t... MI>(std::index_sequence<MI...>) VECOPS_INLINE_LAMBDA {
      (one_m.template operator()<static_cast<int>(MI)>(), ...);
    }(std::make_index_sequence<NM>{});
  } else {
    static_assert(NM == 2 && NN <= 4);
    const auto a0 = load_operand<
        Atom, ::vecops::matmul::Operand::A, 0, FullM, FullK>(
        a, a_invariants, m, kg, logical_m, logical_k);
    const auto a1 = load_operand<
        Atom, ::vecops::matmul::Operand::A, 1,
        FullM, FullK>(
        a, a_invariants, m, kg, logical_m, logical_k);
    auto one_n = [&]<int NI>() VECOPS_INLINE_LAMBDA {
      const auto bv = load_operand<
          Atom, ::vecops::matmul::Operand::B, NI,
          FullN || (ExactBlocks && NI + 1 < NN), FullK>(
          b, b_invariants, n, kg, logical_n, logical_k);
      // Tile numbers NI and NN+NI: column-interleaved across the two A
      // blocks (tile = MI*NN + NI with NM == 2).
      mopa<Atom, NI>(a0, bv);
      mopa<Atom, NN + NI>(a1, bv);
    };
    [&]<std::size_t... NI>(std::index_sequence<NI...>) VECOPS_INLINE_LAMBDA {
      (one_n.template operator()<static_cast<int>(NI)>(), ...);
    }(std::make_index_sequence<NN>{});
  }
}

/**
 * @brief One Tile2D case: NM x NN ZA output tiles over one (m, n) block.
 *
 * Structure: zero ZA, seed every output tile with the C input
 * (`initialize_c_tile`), accumulate all K groups (fast packed-pointer loop
 * or generic `compute_group`), store every tile back (`store_c_tile`).
 * Linear tile `I` is output block (I/NN, I%NN) at
 * `(m + (I/NN)*lanes, n + (I%NN)*lanes)`.
 *
 * The compute selection:
 * - `FastPacked && ExactBlocks`: packed inputs with all blocks guaranteed --
 *   straight into `compute_packed_groups`.
 * - `FastPacked` without `ExactBlocks`: same fast loop only when the
 *   runtime extents prove every logical block exists
 *   (`all_logical_blocks_exist`); otherwise the generic path with its
 *   per-block masking.
 * - everything else (direct inputs, short-K packed, > 4 outputs):
 *   `compute_generic`.
 *
 * The body is normally inlined.  `microkernel` below moves only the shallow-K
 * specialization out of line: that additional plan otherwise clones the
 * complete init/compute/store body into every scheduler, Meta, and epilogue
 * variant, while established long-K paths measurably benefit from inlining.
 *
 * @tparam Plan  KernelPlan selecting FastPacked and prefetch flags.
 */
template <::vecops::matmul::Atom Atom, typename Plan, int NM, int NN,
          bool FullM, bool FullN, bool ExactBlocks,
          typename A, typename B,
          typename CInput, typename COutput,
          typename AccInput, typename AccOutput, typename Route>
VECOPS_ALWAYS_INLINE void microkernel_body_routed(
    const A& a, const B& b,
    const CInput& c_input, COutput& c_output,
    const AccInput& acc_input, AccOutput& acc_output, Route route,
    nint_t logical_m, nint_t logical_n, nint_t logical_k,
    nint_t m, nint_t n, nint_t active_m, nint_t active_n) {
  constexpr int Outputs = NM * NN;
  constexpr bool FastPacked = Plan::value;
  constexpr bool PrefetchLargeWorkingSet =
    Plan::prefetch_large_working_set && std::same_as<Atom, ::vecops::matmul::SME_BF16F32> && NM == 2 && NN == 2;

  // VLA SVE input/prologue transforms still pay for P1 tuple splitting, so
  // keep that side narrow. The output pair machinery can use either one P1
  // transform or two independent P0 calls followed by one coalesced store,
  // but the backend default remains narrow until either is profitable. An
  // explicit transform-store override still takes priority.
  constexpr bool EnableWideCInputGrouping = false;
  const nint_t lanes = static_cast<nint_t>(Atom::M_R);
  vec::details::sme::zero_za();
  const OperandInvariants<Atom, ::vecops::matmul::Operand::A, A> a_invariants(a);
  const OperandInvariants<Atom, ::vecops::matmul::Operand::B, B> b_invariants(b);

  auto initialize_tile = [&]<std::size_t I>() VECOPS_INLINE_LAMBDA {
    initialize_c_tile_routed<static_cast<int>(I), FullM || (ExactBlocks && I / NN + 1 < NM),
                             FullN || (ExactBlocks && I % NN + 1 < NN)>(
      c_input, acc_input, route, m + static_cast<nint_t>(I / NN) * lanes, n + static_cast<nint_t>(I % NN) * lanes,
      tile_active_extent < Atom, FullM || (ExactBlocks && I / NN + 1 < NM), FullM || ExactBlocks,
      static_cast<int>(I / NN) > (active_m), tile_active_extent < Atom, FullN || (ExactBlocks && I % NN + 1 < NN),
      FullN || ExactBlocks, static_cast<int>(I % NN) > (active_n));
  };
  constexpr bool GroupCInputRows =
    EnableWideCInputGrouping && NN >= 2 && row_contiguous_input_v<CInput> && row_contiguous_input_v<AccInput> &&
    ((!IsZeroTransform<typename CInput::Transform>::value && tensor::preferred_memory_access_power_v<CInput> > 0) ||
     (!IsZeroTransform<typename AccInput::Transform>::value && tensor::preferred_memory_access_power_v<AccInput> > 0));
  constexpr bool GroupCInputColumns =
    EnableWideCInputGrouping && NM >= 2 && column_contiguous_input_v<CInput> && column_contiguous_input_v<AccInput> &&
    ((!IsZeroTransform<typename CInput::Transform>::value && tensor::preferred_memory_access_power_v<CInput> > 0) ||
     (!IsZeroTransform<typename AccInput::Transform>::value && tensor::preferred_memory_access_power_v<AccInput> > 0));
  if constexpr (GroupCInputColumns) {
    auto initialize_column_pair = [&]<std::size_t Pair, std::size_t Column>() VECOPS_INLINE_LAMBDA {
      constexpr std::size_t TopRow = 2 * Pair;
      constexpr std::size_t BottomRow = TopRow + 1;
      constexpr std::size_t Top = TopRow * NN + Column;
      constexpr std::size_t Bottom = BottomRow * NN + Column;
      initialize_c_tile_column_pair_routed<
        static_cast<int>(Top), static_cast<int>(Bottom), FullM || (ExactBlocks && TopRow + 1 < NM),
        FullM || (ExactBlocks && BottomRow + 1 < NM), FullN || (ExactBlocks && Column + 1 < NN)>(
        c_input, acc_input, route, m + static_cast<nint_t>(TopRow) * lanes, n + static_cast<nint_t>(Column) * lanes,
        tile_active_extent < Atom, FullM || (ExactBlocks && TopRow + 1 < NM), FullM || ExactBlocks,
        static_cast<int>(TopRow) > (active_m), tile_active_extent < Atom, FullM || (ExactBlocks && BottomRow + 1 < NM),
        FullM || ExactBlocks, static_cast<int>(BottomRow) > (active_m), tile_active_extent < Atom,
        FullN || (ExactBlocks && Column + 1 < NN), FullN || ExactBlocks, static_cast<int>(Column) > (active_n), lanes);
    };
    auto initialize_pair_columns = [&]<std::size_t Pair>() VECOPS_INLINE_LAMBDA {
      [&]<std::size_t... Column>(std::index_sequence<Column...>) VECOPS_INLINE_LAMBDA {
        (initialize_column_pair.template operator()<Pair, Column>(), ...);
      }(std::make_index_sequence<NN>{});
    };
    [&]<std::size_t... Pair>(std::index_sequence<Pair...>) VECOPS_INLINE_LAMBDA {
      (initialize_pair_columns.template operator()<Pair>(), ...);
    }(std::make_index_sequence<NM / 2>{});
    if constexpr (NM % 2 != 0) {
      [&]<std::size_t... Column>(std::index_sequence<Column...>) VECOPS_INLINE_LAMBDA {
        (initialize_tile.template operator()<(NM - 1) * NN + Column>(), ...);
      }(std::make_index_sequence<NN>{});
    }
  } else if constexpr (GroupCInputRows) {
    auto initialize_row_pair = [&]<std::size_t Row, std::size_t Pair>() VECOPS_INLINE_LAMBDA {
      constexpr std::size_t LeftColumn = 2 * Pair;
      constexpr std::size_t RightColumn = LeftColumn + 1;
      constexpr std::size_t Left = Row * NN + LeftColumn;
      constexpr std::size_t Right = Left + 1;
      initialize_c_tile_row_pair_routed<
        static_cast<int>(Left), static_cast<int>(Right), FullM || (ExactBlocks && Row + 1 < NM),
        FullN || (ExactBlocks && LeftColumn + 1 < NN), FullN || (ExactBlocks && RightColumn + 1 < NN)>(
        c_input, acc_input, route, m + static_cast<nint_t>(Row) * lanes, n + static_cast<nint_t>(LeftColumn) * lanes,
        tile_active_extent < Atom, FullM || (ExactBlocks && Row + 1 < NM), FullM || ExactBlocks,
        static_cast<int>(Row) > (active_m), tile_active_extent < Atom, FullN || (ExactBlocks && LeftColumn + 1 < NN),
        FullN || ExactBlocks, static_cast<int>(LeftColumn) > (active_n), tile_active_extent < Atom,
        FullN || (ExactBlocks && RightColumn + 1 < NN), FullN || ExactBlocks,
        static_cast<int>(RightColumn) > (active_n), lanes);
    };
    auto initialize_row_pairs = [&]<std::size_t Row>() VECOPS_INLINE_LAMBDA {
      [&]<std::size_t... Pair>(std::index_sequence<Pair...>) VECOPS_INLINE_LAMBDA {
        (initialize_row_pair.template operator()<Row, Pair>(), ...);
      }(std::make_index_sequence<NN / 2>{});
    };
    [&]<std::size_t... Row>(std::index_sequence<Row...>)
      VECOPS_INLINE_LAMBDA { (initialize_row_pairs.template operator()<Row>(), ...); }(std::make_index_sequence<NM>{});
    if constexpr (NN % 2 != 0) {
      [&]<std::size_t... Row>(std::index_sequence<Row...>) VECOPS_INLINE_LAMBDA {
        (initialize_tile.template operator()<Row * NN + NN - 1>(), ...);
      }(std::make_index_sequence<NM>{});
    }
  } else {
    [&]<std::size_t... I>(std::index_sequence<I...>)
      VECOPS_INLINE_LAMBDA { (initialize_tile.template operator()<I>(), ...); }(std::make_index_sequence<Outputs>{});
  }
  auto compute_generic = [&]() VECOPS_INLINE_LAMBDA {
    constexpr nint_t KP = ::vecops::matmul::packing_t<Atom, ::vecops::matmul::Operand::A>::KPack;
    // K-group dispatch: KP==1 and the 16-bit atoms split the K walk into a
    // provably-full body (FullK=true unlocks the unchecked whole-group
    // loads in load_operand, including the u32 gather) plus one masked
    // tail group when logical_k % KP != 0. The 4x-byte atom instead runs
    // every group through the guarded partial instantiation
    // (FullK=false), relying on load_operand's per-group k < logical_k
    // masking rather than a split loop.
    if constexpr (KP == 1 || std::same_as<Atom, ::vecops::matmul::SME_BF16F32> ||
                  std::same_as<Atom, ::vecops::matmul::SME_F16F32>) {
      const nint_t full_groups = logical_k / KP;
      for (nint_t kg = 0; kg < full_groups; ++kg) {
        compute_group<
            Atom, NM, NN, FullM, FullN, ExactBlocks, true>(
            a, a_invariants, b, b_invariants,
            m, n, kg, logical_m, logical_n, logical_k);
      }
      if constexpr (KP > 1) {
        if (full_groups * KP < logical_k) {
          compute_group<
              Atom, NM, NN, FullM, FullN, ExactBlocks, false>(
              a, a_invariants, b, b_invariants,
              m, n, full_groups,
              logical_m, logical_n, logical_k);
        }
      }
    } else {
      const nint_t groups = ceil_div(logical_k, KP);
      for (nint_t kg = 0; kg < groups; ++kg) {
        compute_group<
            Atom, NM, NN, FullM, FullN, ExactBlocks, false>(
            a, a_invariants, b, b_invariants,
            m, n, kg,
            logical_m, logical_n, logical_k);
      }
    }
  };

  if constexpr (FastPacked && ExactBlocks && Outputs <= 4) {
    static_assert(is_packed_access_v<Atom, ::vecops::matmul::Operand::A, A>);
    static_assert(is_packed_access_v<Atom, ::vecops::matmul::Operand::B, B>);
    compute_packed_groups<Atom, NM, NN, PrefetchLargeWorkingSet>(
        a, b, m, n, logical_k);
  } else if constexpr (FastPacked && Outputs <= 4) {
    static_assert(is_packed_access_v<Atom, ::vecops::matmul::Operand::A, A>);
    static_assert(is_packed_access_v<Atom, ::vecops::matmul::Operand::B, B>);
    const bool all_logical_blocks_exist =
        active_m > static_cast<nint_t>(NM - 1) * lanes &&
        active_n > static_cast<nint_t>(NN - 1) * lanes;
    if (all_logical_blocks_exist) {
      compute_packed_groups<Atom, NM, NN, PrefetchLargeWorkingSet>(
          a, b, m, n, logical_k);
    } else {
      compute_generic();
    }
  } else {
    compute_generic();
  }

  auto store_tile = [&]<std::size_t I>() VECOPS_INLINE_LAMBDA {
    store_c_tile_routed<static_cast<int>(I), FullM || (ExactBlocks && I / NN + 1 < NM),
                        FullN || (ExactBlocks && I % NN + 1 < NN)>(
      c_output, acc_output, route, m + static_cast<nint_t>(I / NN) * lanes, n + static_cast<nint_t>(I % NN) * lanes,
      tile_active_extent < Atom, FullM || (ExactBlocks && I / NN + 1 < NM), FullM || ExactBlocks,
      static_cast<int>(I / NN) > (active_m), tile_active_extent < Atom, FullN || (ExactBlocks && I % NN + 1 < NN),
      FullN || ExactBlocks, static_cast<int>(I % NN) > (active_n));
  };
  constexpr bool GroupCOutputRows =
    NN >= 2 && row_contiguous_output_v<COutput> && row_contiguous_output_v<AccOutput> &&
    ((tensor::preferred_memory_access_power_v<COutput> > 0 && transform_store_groups_tiles_v<COutput>) ||
     (tensor::preferred_memory_access_power_v<AccOutput> > 0 && transform_store_groups_tiles_v<AccOutput>));
  constexpr bool GroupCOutputColumns =
    NM >= 2 && column_contiguous_output_v<COutput> && column_contiguous_output_v<AccOutput> &&
    ((tensor::preferred_memory_access_power_v<COutput> > 0 && transform_store_groups_tiles_v<COutput>) ||
     (tensor::preferred_memory_access_power_v<AccOutput> > 0 && transform_store_groups_tiles_v<AccOutput>));
  if constexpr (GroupCOutputColumns) {
    auto store_column_pair = [&]<std::size_t Pair, std::size_t Column>() VECOPS_INLINE_LAMBDA {
      constexpr std::size_t TopRow = 2 * Pair;
      constexpr std::size_t BottomRow = TopRow + 1;
      constexpr std::size_t Top = TopRow * NN + Column;
      constexpr std::size_t Bottom = BottomRow * NN + Column;
      store_c_tile_column_pair_routed<
        static_cast<int>(Top), static_cast<int>(Bottom), FullM || (ExactBlocks && TopRow + 1 < NM),
        FullM || (ExactBlocks && BottomRow + 1 < NM), FullN || (ExactBlocks && Column + 1 < NN)>(
        c_output, acc_output, route, m + static_cast<nint_t>(TopRow) * lanes, n + static_cast<nint_t>(Column) * lanes,
        tile_active_extent < Atom, FullM || (ExactBlocks && TopRow + 1 < NM), FullM || ExactBlocks,
        static_cast<int>(TopRow) > (active_m), tile_active_extent < Atom, FullM || (ExactBlocks && BottomRow + 1 < NM),
        FullM || ExactBlocks, static_cast<int>(BottomRow) > (active_m), tile_active_extent < Atom,
        FullN || (ExactBlocks && Column + 1 < NN), FullN || ExactBlocks, static_cast<int>(Column) > (active_n), lanes);
    };
    auto store_pair_columns = [&]<std::size_t Pair>() VECOPS_INLINE_LAMBDA {
      [&]<std::size_t... Column>(std::index_sequence<Column...>) VECOPS_INLINE_LAMBDA {
        (store_column_pair.template operator()<Pair, Column>(), ...);
      }(std::make_index_sequence<NN>{});
    };
    [&]<std::size_t... Pair>(std::index_sequence<Pair...>) VECOPS_INLINE_LAMBDA {
      (store_pair_columns.template operator()<Pair>(), ...);
    }(std::make_index_sequence<NM / 2>{});
    if constexpr (NM % 2 != 0) {
      [&]<std::size_t... Column>(std::index_sequence<Column...>) VECOPS_INLINE_LAMBDA {
        (store_tile.template operator()<(NM - 1) * NN + Column>(), ...);
      }(std::make_index_sequence<NN>{});
    }
  } else if constexpr (GroupCOutputRows) {
    auto store_row_pair = [&]<std::size_t Row, std::size_t Pair>() VECOPS_INLINE_LAMBDA {
      constexpr std::size_t LeftColumn = 2 * Pair;
      constexpr std::size_t RightColumn = LeftColumn + 1;
      constexpr std::size_t Left = Row * NN + LeftColumn;
      constexpr std::size_t Right = Left + 1;
      store_c_tile_row_pair_routed<
        static_cast<int>(Left), static_cast<int>(Right), FullM || (ExactBlocks && Row + 1 < NM),
        FullN || (ExactBlocks && LeftColumn + 1 < NN), FullN || (ExactBlocks && RightColumn + 1 < NN)>(
        c_output, acc_output, route, m + static_cast<nint_t>(Row) * lanes, n + static_cast<nint_t>(LeftColumn) * lanes,
        tile_active_extent < Atom, FullM || (ExactBlocks && Row + 1 < NM), FullM || ExactBlocks,
        static_cast<int>(Row) > (active_m), tile_active_extent < Atom, FullN || (ExactBlocks && LeftColumn + 1 < NN),
        FullN || ExactBlocks, static_cast<int>(LeftColumn) > (active_n), tile_active_extent < Atom,
        FullN || (ExactBlocks && RightColumn + 1 < NN), FullN || ExactBlocks,
        static_cast<int>(RightColumn) > (active_n), lanes);
    };
    auto store_row_pairs = [&]<std::size_t Row>() VECOPS_INLINE_LAMBDA {
      [&]<std::size_t... Pair>(std::index_sequence<Pair...>) VECOPS_INLINE_LAMBDA {
        (store_row_pair.template operator()<Row, Pair>(), ...);
      }(std::make_index_sequence<NN / 2>{});
    };
    [&]<std::size_t... Row>(std::index_sequence<Row...>)
      VECOPS_INLINE_LAMBDA { (store_row_pairs.template operator()<Row>(), ...); }(std::make_index_sequence<NM>{});
    if constexpr (NN % 2 != 0) {
      [&]<std::size_t... Row>(std::index_sequence<Row...>) VECOPS_INLINE_LAMBDA {
        (store_tile.template operator()<Row * NN + NN - 1>(), ...);
      }(std::make_index_sequence<NM>{});
    }
  } else {
    [&]<std::size_t... I>(std::index_sequence<I...>)
      VECOPS_INLINE_LAMBDA { (store_tile.template operator()<I>(), ...); }(std::make_index_sequence<Outputs>{});
  }
}

template <::vecops::matmul::Atom Atom, typename Plan, int NM, int NN, bool FullM, bool FullN, bool ExactBlocks,
          typename A, typename B, typename CInput, typename COutput>
VECOPS_ALWAYS_INLINE void microkernel_body(const A& a, const B& b, const CInput& c_input, COutput& c_output,
                                           nint_t logical_m, nint_t logical_n, nint_t logical_k, nint_t m, nint_t n,
                                           nint_t active_m, nint_t active_n) {
  microkernel_body_routed<
      Atom, Plan, NM, NN, FullM, FullN, ExactBlocks>(
      a, b, c_input, c_output, c_input, c_output,
      matmul_details::UnsplitAccumulatorRoute{},
      logical_m, logical_n, logical_k,
      m, n, active_m, active_n);
}

template <::vecops::matmul::Atom Atom, typename Plan, int NM, int NN,
          bool FullM, bool FullN, bool ExactBlocks,
          typename A, typename B, typename CInput, typename COutput>
VECOPS_NOINLINE void microkernel_shared(
    const A& a, const B& b, const CInput& c_input, COutput& c_output,
    nint_t logical_m, nint_t logical_n, nint_t logical_k,
    nint_t m, nint_t n, nint_t active_m, nint_t active_n) {
  microkernel_body<
      Atom, Plan, NM, NN, FullM, FullN, ExactBlocks>(
      a, b, c_input, c_output, logical_m, logical_n, logical_k,
      m, n, active_m, active_n);
}

template <::vecops::matmul::Atom Atom, typename Plan, int NM, int NN,
          bool FullM, bool FullN, bool ExactBlocks,
          typename A, typename B, typename CInput, typename COutput>
VECOPS_ALWAYS_INLINE void microkernel(
    const A& a, const B& b, const CInput& c_input, COutput& c_output,
    nint_t logical_m, nint_t logical_n, nint_t logical_k,
    nint_t m, nint_t n, nint_t active_m, nint_t active_n) {
  if constexpr (Plan::share_tile_body) {
    microkernel_shared<
        Atom, Plan, NM, NN, FullM, FullN, ExactBlocks>(
        a, b, c_input, c_output, logical_m, logical_n, logical_k,
        m, n, active_m, active_n);
  } else {
    microkernel_body<
        Atom, Plan, NM, NN, FullM, FullN, ExactBlocks>(
        a, b, c_input, c_output, logical_m, logical_n, logical_k,
        m, n, active_m, active_n);
  }
}

/** One ZA body shared by first/middle/last split-K phases. */
template <::vecops::matmul::Atom Atom, typename Plan, int NM, int NN,
          bool FullM, bool FullN, bool ExactBlocks,
          typename A, typename B,
          typename CInput, typename COutput,
          typename AccInput, typename AccOutput, typename Route>
VECOPS_ALWAYS_INLINE void microkernel_phased(
    const A& a, const B& b,
    const CInput& c_input, COutput& c_output,
    const AccInput& acc_input, AccOutput& acc_output, Route route,
    nint_t logical_m, nint_t logical_n, nint_t logical_k,
    nint_t m, nint_t n, nint_t active_m, nint_t active_n) {
  static_assert(!matmul_details::static_accumulator_route_v<Route>);
  microkernel_body_routed<
      Atom, Plan, NM, NN, FullM, FullN, ExactBlocks>(
      a, b, c_input, c_output, acc_input, acc_output, route,
      logical_m, logical_n, logical_k,
      m, n, active_m, active_n);
}

} // namespace vecops::kernel::matmul_details::sme

namespace vecops::kernel::matmul_details {

/**
 * @brief SME matmul backend: the matmul_details::Backend contract for the
 *        `SME` implementation tag.
 *
 * Owns the whole-problem dispatch described in the file header: the
 * compile-time owner selection plus the runtime applicability tier in
 * `run`, one Streaming+ZA region around the Tile2D traversal for the
 * General path, and the FastPacked plan resolution in `dispatch_plan`.
 * Scratch is always zero -- everything the ZA microkernel needs lives in
 * the architectural ZA storage.
 */
template <>
struct Backend<matmul_implementation::SME> {
  using ResourceRequirements = execution::details::ResourceSet<>;
  /// Tile2D catalog: expanded (ZA64, up to 8 meta blocks) for unpacked F64
  /// problems, compact otherwise.
  template <::vecops::matmul::Atom Atom, typename,
            meta::ValueType, meta::ValueType, meta::ValueType,
            typename A, typename B, typename, typename>
  using Catalog = sme::Catalog<sme::use_expanded_catalog_v<Atom, A, B>>;
  template <::vecops::matmul::Atom Atom, typename,
            meta::ValueType, meta::ValueType, meta::ValueType,
            typename A, typename B, typename, typename>
  using NMajorCatalog = sme::Catalog<sme::use_expanded_catalog_v<Atom, A, B>>;
  static constexpr int ProblemRank = 2;

  static nint_t scratch_bytes() { return 0; }

  /// Resolution of an `Automatic` Tile2D policy: ExactCover over the
  /// expanded catalog for unpacked F64; otherwise FourRegions when Meta
  /// bounds favor static constraint pruning (see
  /// `prefer_constraint_pruning_v`), ExactCover otherwise.
  template <::vecops::matmul::Atom Atom, typename Policy,
            meta::ValueType M, meta::ValueType N, meta::ValueType K,
            typename A, typename B, typename, typename>
  using EffectivePolicy = std::conditional_t<
      std::same_as<Policy, matmul_policy::Automatic>,
      std::conditional_t<
          sme::use_expanded_catalog_v<Atom, A, B>,
          kernel::loop::tile2d_policy::ExactCover,
          std::conditional_t<
              sme::prefer_constraint_pruning_v<Atom, M, N, K, A, B>,
              kernel::loop::tile2d_policy::FourRegions,
              kernel::loop::tile2d_policy::ExactCover>>,
      Policy>;

  /**
   * @brief Whole-problem entry: resolve the dispatch owner at compile time
   *        and run it, with a runtime tier for `runtime` applicability.
   *
   * The nine no-ZA leaves run directly when statically selected (their
   * gates re-checked at run time where shapes are only bounded; a
   * `required` family that still rejects raises a check failure). Before
   * that, each family whose applicability probe returned `runtime` is
   * probed with the concrete m/n/k (RuntimeDispatch.h); this serves both
   * explicit family requests the static owner does not satisfy and
   * Automatic dispatch whose shapes the types alone cannot decide.
   * Everything else -- including every owner selected while the scope
   * already owns StreamingZA -- falls through to the General path: one
   * `StreamingZARegion` around the whole tile traversal. Packed operands
   * are forwarded as-is; direct operands and the output are rebound to the
   * region's active resource set (streaming-appropriate memory ops), and
   * the output session is committed inside the region so a transformed
   * output never defers its write-back past SMSTOP.
   */
  template <::vecops::matmul::Atom Atom, typename Policy,
            bool = false,
            typename FamilyDispatch =
                ::vecops::matmul::details::AutomaticFamilyDispatch,
            execution::ExecutionScope Scope,
            meta::ValueType M, meta::ValueType N, meta::ValueType K,
            typename A, typename B, typename CInput, typename COutput>
  VECOPS_ALWAYS_INLINE static void run(
      Scope& scope, M m, N n, K k,
      const A& a, const B& b, const CInput& c_input, COutput& c_output,
      void* scratch) {
    static_assert(std::same_as<typename Atom::KernelKind, ::vecops::matmul::SMEKernelKind>);
    constexpr auto Owner = sme::select_dispatch_owner<
        FamilyDispatch, Atom, M, N, K, A, B, CInput, COutput, Scope>();
    using Applicability = ::vecops::matmul::details::Applicability;
    using RequestedFamily = typename FamilyDispatch::Family;
    constexpr bool NMajor =
        matmul_details::n_major_family_dispatch_v<FamilyDispatch>;
    constexpr bool AutomaticNeedsRuntimeProbe =
        std::same_as<RequestedFamily,
                     ::vecops::matmul::kernel_family::WholeProblem> &&
        Owner == sme::DispatchOwner::General;
    constexpr bool RuntimeSmallVector = [] {
      if constexpr (!sme::permits_runtime_family_v<
                        FamilyDispatch,
                        ::vecops::matmul::kernel_family::SmallVector>) {
        return false;
      } else {
        constexpr bool Requested = std::same_as<
            RequestedFamily,
            ::vecops::matmul::kernel_family::SmallVector>;
        return (Requested || AutomaticNeedsRuntimeProbe) &&
            sme::family_applicability<
                ::vecops::matmul::kernel_family::SmallVector,
                Atom, M, N, K, A, B, CInput, COutput, Scope>() ==
                Applicability::runtime;
      }
    }();
    if constexpr (RuntimeSmallVector) {
      const nint_t logical_m = static_cast<nint_t>(m);
      const nint_t logical_n = static_cast<nint_t>(n);
      const nint_t logical_k = static_cast<nint_t>(k);
      if (runtime_sme_small_vector_profitable(
              sme::small_vector_shape_v<Atom>,
              logical_m, logical_n, logical_k)) {
#if defined(__ARM_FEATURE_SVE_MATMUL_INT8)
        if constexpr (sme::mixed_sign_sve_skinny_candidate_v<
                          Atom, A, B, CInput, COutput>) {
          const bool handled = sme::try_mixed_sign_sve_skinny<Atom>(
              logical_m, logical_n, logical_k,
              a, b, c_input, c_output);
          VECOPS_ASSERT(handled, "runtime mixed-sign skinny plan rejected");
          return;
        }
#endif
#if !defined(VECOPS_DISABLE_SME_SVE_SKINNY)
        if constexpr (sme::sve_skinny_candidate_v<
                          Atom, A, B, CInput, COutput>) {
          if (logical_m == 1)
            sme::sve_skinny_matmul<false>(
                a, b, c_output, logical_n, logical_k);
          else
            sme::sve_skinny_matmul<true>(
                a, b, c_output, logical_m, logical_k);
          return;
        }
#endif
#if !defined(VECOPS_DISABLE_SME_SVE_SKINNY) && \
    !defined(VECOPS_DISABLE_SME_FUSED_SKINNY)
        if constexpr (sme::sve_skinny_fused_candidate_v<
                          Atom, A, B, CInput, COutput>) {
          const bool vary_rows = logical_m != 1;
#if defined(HAS_SME_F64F64)
          if constexpr (std::same_as<
                            Atom, ::vecops::matmul::SME_F64F64>) {
            if (vary_rows)
              sme::sve_skinny_fused_matmul_f64_external<true>(
                  a, b, c_input, c_output, logical_m, logical_k);
            else
              sme::sve_skinny_fused_matmul_f64_external<false>(
                  a, b, c_input, c_output, logical_n, logical_k);
          } else
#endif
          if constexpr (COutput::Transform::is_elementwise) {
            if (vary_rows)
              sme::sve_skinny_fused_matmul<true>(
                  a, b, c_input, c_output, logical_m, logical_k);
            else
              sme::sve_skinny_fused_matmul<false>(
                  a, b, c_input, c_output, logical_n, logical_k);
          } else {
            if (vary_rows)
              sme::sve_skinny_fused_lane_local_matmul<true>(
                  a, b, c_input, c_output, logical_m, logical_k);
            else
              sme::sve_skinny_fused_lane_local_matmul<false>(
                  a, b, c_input, c_output, logical_n, logical_k);
          }
          return;
        }
#endif
      }
      if constexpr (FamilyDispatch::required &&
                    std::same_as<
                        RequestedFamily,
                        ::vecops::matmul::kernel_family::SmallVector>) {
        VECOPS_CHECK(false, "required SmallVector family rejected at runtime");
      }
    }

#if defined(__ARM_FEATURE_SVE_MATMUL_INT8)
    constexpr bool RuntimeQuant = [] {
      constexpr bool Requested = std::same_as<
          RequestedFamily,
          ::vecops::matmul::kernel_family::RuntimeQuantInt8>;
      return (Requested || AutomaticNeedsRuntimeProbe) &&
          sme::permits_runtime_family_v<
              FamilyDispatch,
              ::vecops::matmul::kernel_family::RuntimeQuantInt8> &&
          sme::family_applicability<
              ::vecops::matmul::kernel_family::RuntimeQuantInt8,
              Atom, M, N, K, A, B, CInput, COutput, Scope>() ==
              Applicability::runtime;
    }();
    if constexpr (RuntimeQuant) {
      const bool handled =
          sme::try_fused_runtime_quant_int8_packed_b_gemv<Atom>(
              static_cast<nint_t>(m), static_cast<nint_t>(n),
              static_cast<nint_t>(k), a, b, c_input, c_output);
      if (handled) return;
      if constexpr (FamilyDispatch::required &&
                    std::same_as<
                        RequestedFamily,
                        ::vecops::matmul::kernel_family::RuntimeQuantInt8>) {
        VECOPS_CHECK(false,
                     "required RuntimeQuantInt8 family rejected at runtime");
      }
    }
#endif

#if defined(CPU_CAPABILITY_SVE)
    constexpr bool RuntimePackedDot = [] {
      constexpr bool Requested = std::same_as<
          RequestedFamily,
          ::vecops::matmul::kernel_family::PackedDot>;
      return (Requested || AutomaticNeedsRuntimeProbe) &&
          sme::permits_runtime_family_v<
              FamilyDispatch,
              ::vecops::matmul::kernel_family::PackedDot> &&
          sme::family_applicability<
              ::vecops::matmul::kernel_family::PackedDot,
              Atom, M, N, K, A, B, CInput, COutput, Scope>() ==
              Applicability::runtime;
    }();
    if constexpr (RuntimePackedDot) {
      const nint_t logical_m = static_cast<nint_t>(m);
      const nint_t logical_n = static_cast<nint_t>(n);
      const nint_t logical_k = static_cast<nint_t>(k);
      const bool handled = sme::try_packed_dot<Atom>(
          logical_m, logical_n, logical_k,
          a, b, c_input, c_output) ||
          sme::try_packed_dot_tiny<Atom>(
              logical_m, logical_n, logical_k,
              a, b, c_input, c_output);
      if (handled) return;
      if constexpr (FamilyDispatch::required &&
                    std::same_as<
                        RequestedFamily,
                        ::vecops::matmul::kernel_family::PackedDot>) {
        VECOPS_CHECK(false, "required PackedDot family rejected at runtime");
      }
    }
#endif
#if defined(__ARM_FEATURE_SVE_MATMUL_INT8)
    if constexpr (Owner == sme::DispatchOwner::MixedSignSkinnyRow ||
                  Owner == sme::DispatchOwner::MixedSignSkinnyColumn) {
      const bool handled = sme::try_mixed_sign_sve_skinny<Atom>(
          static_cast<nint_t>(m), static_cast<nint_t>(n),
          static_cast<nint_t>(k), a, b, c_input, c_output);
      VECOPS_ASSERT(handled, "compile-time mixed-sign skinny plan rejected");
      return;
    }
#endif
#if !defined(VECOPS_DISABLE_SME_SVE_SKINNY)
    if constexpr (Owner == sme::DispatchOwner::RawSkinnyRow) {
      sme::sve_skinny_matmul<false>(
          a, b, c_output, static_cast<nint_t>(n), static_cast<nint_t>(k));
      return;
    } else if constexpr (Owner == sme::DispatchOwner::RawSkinnyColumn) {
      sme::sve_skinny_matmul<true>(
          a, b, c_output, static_cast<nint_t>(m), static_cast<nint_t>(k));
      return;
    }
#endif
#if !defined(VECOPS_DISABLE_SME_SVE_SKINNY) && \
    !defined(VECOPS_DISABLE_SME_FUSED_SKINNY)
    if constexpr (Owner == sme::DispatchOwner::FusedSkinnyRow) {
#if defined(HAS_SME_F64F64)
      if constexpr (std::same_as<Atom, ::vecops::matmul::SME_F64F64>) {
        sme::sve_skinny_fused_matmul_f64_external<false>(
            a, b, c_input, c_output,
            static_cast<nint_t>(n), static_cast<nint_t>(k));
      } else
#endif
      {
        if constexpr (COutput::Transform::is_elementwise) {
          sme::sve_skinny_fused_matmul<false>(
              a, b, c_input, c_output,
              static_cast<nint_t>(n), static_cast<nint_t>(k));
        } else {
          sme::sve_skinny_fused_lane_local_matmul<false>(
              a, b, c_input, c_output,
              static_cast<nint_t>(n), static_cast<nint_t>(k));
        }
      }
      return;
    } else if constexpr (Owner == sme::DispatchOwner::FusedSkinnyColumn) {
#if defined(HAS_SME_F64F64)
      if constexpr (std::same_as<Atom, ::vecops::matmul::SME_F64F64>) {
        sme::sve_skinny_fused_matmul_f64_external<true>(
            a, b, c_input, c_output,
            static_cast<nint_t>(m), static_cast<nint_t>(k));
      } else
#endif
      {
        if constexpr (COutput::Transform::is_elementwise) {
          sme::sve_skinny_fused_matmul<true>(
              a, b, c_input, c_output,
              static_cast<nint_t>(m), static_cast<nint_t>(k));
        } else {
          sme::sve_skinny_fused_lane_local_matmul<true>(
              a, b, c_input, c_output,
              static_cast<nint_t>(m), static_cast<nint_t>(k));
        }
      }
      return;
    }
#endif
#if defined(__ARM_FEATURE_SVE_MATMUL_INT8)
    if constexpr (Owner == sme::DispatchOwner::RuntimeQuantINT8) {
      const bool handled =
          sme::try_fused_runtime_quant_int8_packed_b_gemv<Atom>(
              static_cast<nint_t>(m), static_cast<nint_t>(n),
              static_cast<nint_t>(k), a, b, c_input, c_output);
      if constexpr (FamilyDispatch::required)
        VECOPS_CHECK(handled, "required runtime-quant INT8 family rejected");
      if (handled) {
        return;
      }
    }
#endif
#if defined(CPU_CAPABILITY_SVE)
    if constexpr (Owner == sme::DispatchOwner::PackedDotPrimary) {
      const bool handled = sme::try_packed_dot<Atom>(
              static_cast<nint_t>(m), static_cast<nint_t>(n),
              static_cast<nint_t>(k), a, b, c_input, c_output);
      if constexpr (FamilyDispatch::required)
        VECOPS_CHECK(handled, "required PackedDot family rejected");
      if (handled) {
        return;
      }
    } else if constexpr (Owner == sme::DispatchOwner::PackedDotTiny) {
      const bool handled = sme::try_packed_dot_tiny<Atom>(
              static_cast<nint_t>(m), static_cast<nint_t>(n),
              static_cast<nint_t>(k), a, b, c_input, c_output);
      if constexpr (FamilyDispatch::required)
        VECOPS_CHECK(handled, "required PackedDot family rejected");
      if (handled) {
        return;
      }
    }
#endif
    scope.with_resources(
        execution::details::arm::StreamingZARegion{},
        [&](auto& active) VECOPS_INLINE_LAMBDA_NOEXCEPT {
          // Packed operands skip the rebind: packed access is statically
          // direct and untransformed and consumed through raw pointers, so
          // the region's resource set cannot influence their loads. Direct
          // operands (and both C sides, which may carry transforms) are
          // re-expressed under the active resource set so their gathers
          // and stores issue streaming-appropriate instructions.
          if constexpr (
              sme::is_packed_access_v<Atom, ::vecops::matmul::Operand::A, A> &&
              sme::is_packed_access_v<Atom, ::vecops::matmul::Operand::B, B>) {
            if constexpr (NMajor)
              matmul_details::run_tiles_n_major<Backend, Atom, Policy>(
                  m, n, k, a, b, c_input, c_output, scratch);
            else
              matmul_details::run_tiles<Backend, Atom, Policy>(
                  m, n, k, a, b, c_input, c_output, scratch);
          } else {
            using Resources = typename std::remove_cvref_t<
                decltype(active)>::ActiveResources;
            auto active_a = tensor::rebind_active_resources<Resources>(a);
            auto active_b = tensor::rebind_active_resources<Resources>(b);
            auto active_c_input =
                tensor::rebind_active_resources<Resources>(c_input);
            auto active_c_output =
                tensor::rebind_active_resources<Resources>(c_output);
            if constexpr (NMajor)
              matmul_details::run_tiles_n_major<Backend, Atom, Policy>(
                  m, n, k, active_a, active_b,
                  active_c_input, active_c_output, scratch);
            else
              matmul_details::run_tiles<Backend, Atom, Policy>(
                  m, n, k, active_a, active_b,
                  active_c_input, active_c_output, scratch);
            // Commit while still inside the region: a materialized
            // (transformed) output session must not carry deferred state
            // across SMSTOP.
            active_c_output.commit();
          }
        });
  }

  template <::vecops::matmul::Atom Atom, typename Policy,
            bool AllowTailSplit = false,
            typename FamilyDispatch =
                ::vecops::matmul::details::AutomaticFamilyDispatch,
            execution::ExecutionScope Scope,
            meta::ValueType M, meta::ValueType N, meta::ValueType K,
            typename A, typename B, typename CInput, typename COutput>
  VECOPS_ALWAYS_INLINE static void run_n_major(
      Scope& scope, M m, N n, K k,
      const A& a, const B& b, const CInput& c_input, COutput& c_output,
      void* scratch) {
    using Dispatch = matmul_details::NMajorFamilyDispatch<FamilyDispatch>;
    run<Atom, Policy, AllowTailSplit, Dispatch>(
        scope, m, n, k, a, b, c_input, c_output, scratch);
  }

  /** Generic-Tiler leaf entry: skip all whole-problem SME selectors. */
  template <::vecops::matmul::Atom Atom, typename Policy,
            execution::ExecutionScope Scope,
            meta::ValueType M, meta::ValueType N, meta::ValueType K,
            typename A, typename B, typename CInput, typename COutput>
  VECOPS_ALWAYS_INLINE static void run_configured(
      Scope& scope, M m, N n, K k,
      const A& a, const B& b, const CInput& c_input, COutput& c_output,
      void* scratch) {
    static_assert(std::same_as<typename Atom::KernelKind,
                               ::vecops::matmul::SMEKernelKind>);
    scope.with_resources(
        execution::details::arm::StreamingZARegion{},
        [&](auto& active) VECOPS_INLINE_LAMBDA_NOEXCEPT {
          // Packed operands skip the rebind: packed access is statically
          // direct and untransformed and consumed through raw pointers, so
          // the region's resource set cannot influence their loads. Direct
          // operands (and both C sides, which may carry transforms) are
          // re-expressed under the active resource set so their gathers
          // and stores issue streaming-appropriate instructions.
          if constexpr (
              sme::is_packed_access_v<Atom, ::vecops::matmul::Operand::A, A> &&
              sme::is_packed_access_v<Atom, ::vecops::matmul::Operand::B, B>) {
            matmul_details::run_tiles<Backend, Atom, Policy>(
                m, n, k, a, b, c_input, c_output, scratch);
          } else {
            using Resources = typename std::remove_cvref_t<
                decltype(active)>::ActiveResources;
            auto active_a = tensor::rebind_active_resources<Resources>(a);
            auto active_b = tensor::rebind_active_resources<Resources>(b);
            auto active_c_input =
                tensor::rebind_active_resources<Resources>(c_input);
            auto active_c_output =
                tensor::rebind_active_resources<Resources>(c_output);
            matmul_details::run_tiles<Backend, Atom, Policy>(
                m, n, k, active_a, active_b,
                active_c_input, active_c_output, scratch);
            // Commit while still inside the region: a materialized
            // (transformed) output session must not carry deferred state
            // across SMSTOP.
            active_c_output.commit();
          }
        });
  }

  template <::vecops::matmul::Atom Atom, typename Policy,
            execution::ExecutionScope Scope,
            meta::ValueType M, meta::ValueType N, meta::ValueType K,
            typename A, typename B, typename CInput, typename COutput>
  VECOPS_ALWAYS_INLINE static void run_configured_n_major(
      Scope& scope, M m, N n, K k,
      const A& a, const B& b, const CInput& c_input, COutput& c_output,
      void* scratch) {
    scope.with_resources(
        execution::details::arm::StreamingZARegion{},
        [&](auto& active) VECOPS_INLINE_LAMBDA_NOEXCEPT {
          if constexpr (
              sme::is_packed_access_v<Atom, ::vecops::matmul::Operand::A, A> &&
              sme::is_packed_access_v<Atom, ::vecops::matmul::Operand::B, B>) {
            matmul_details::run_tiles_n_major<Backend, Atom, Policy>(
                m, n, k, a, b, c_input, c_output, scratch);
          } else {
            using Resources = typename std::remove_cvref_t<
                decltype(active)>::ActiveResources;
            auto active_a = tensor::rebind_active_resources<Resources>(a);
            auto active_b = tensor::rebind_active_resources<Resources>(b);
            auto active_c_input =
                tensor::rebind_active_resources<Resources>(c_input);
            auto active_c_output =
                tensor::rebind_active_resources<Resources>(c_output);
            matmul_details::run_tiles_n_major<Backend, Atom, Policy>(
                m, n, k, active_a, active_b,
                active_c_input, active_c_output, scratch);
            active_c_output.commit();
          }
        });
  }

  /** Generic-Tiler split-K entry with one runtime C route type. */
  template <::vecops::matmul::Atom Atom, typename Policy,
            execution::ExecutionScope Scope,
            meta::ValueType M, meta::ValueType N, meta::ValueType K,
            typename A, typename B,
            typename CInput, typename COutput,
            typename AccInput, typename AccOutput, typename Route>
  VECOPS_ALWAYS_INLINE static void run_phased(
      Scope& scope, M m, N n, K k,
      const A& a, const B& b,
      const CInput& c_input, COutput& c_output,
      const AccInput& acc_input, AccOutput& acc_output,
      Route route, void* scratch) {
    static_assert(std::same_as<typename Atom::KernelKind,
                               ::vecops::matmul::SMEKernelKind>);
    scope.with_resources(
        execution::details::arm::StreamingZARegion{},
        [&](auto& active) VECOPS_INLINE_LAMBDA_NOEXCEPT {
          if constexpr (
              sme::is_packed_access_v<Atom, ::vecops::matmul::Operand::A, A> &&
              sme::is_packed_access_v<Atom, ::vecops::matmul::Operand::B, B>) {
            run_tiles_phased_shared<Atom, Policy>(
                m, n, k, a, b, c_input, c_output,
                acc_input, acc_output, route, scratch);
          } else {
            using Resources = typename std::remove_cvref_t<
                decltype(active)>::ActiveResources;
            auto active_a = tensor::rebind_active_resources<Resources>(a);
            auto active_b = tensor::rebind_active_resources<Resources>(b);
            auto active_c_input =
                tensor::rebind_active_resources<Resources>(c_input);
            auto active_c_output =
                tensor::rebind_active_resources<Resources>(c_output);
            auto active_acc_input =
                tensor::rebind_active_resources<Resources>(acc_input);
            auto active_acc_output =
                tensor::rebind_active_resources<Resources>(acc_output);
            run_tiles_phased_shared<Atom, Policy>(
                m, n, k, active_a, active_b,
                active_c_input, active_c_output,
                active_acc_input, active_acc_output, route, scratch);
            active_c_output.commit();
            active_acc_output.commit();
          }
        });
  }

  template <::vecops::matmul::Atom Atom, typename Policy,
            execution::ExecutionScope Scope,
            meta::ValueType M, meta::ValueType N, meta::ValueType K,
            typename A, typename B, typename CInput, typename COutput,
            typename AccInput, typename AccOutput, typename Route>
  VECOPS_ALWAYS_INLINE static void run_phased_n_major(
      Scope& scope, M m, N n, K k, const A& a, const B& b,
      const CInput& c_input, COutput& c_output,
      const AccInput& acc_input, AccOutput& acc_output,
      Route route, void* scratch) {
    scope.with_resources(
        execution::details::arm::StreamingZARegion{},
        [&](auto& active) VECOPS_INLINE_LAMBDA_NOEXCEPT {
          if constexpr (
              sme::is_packed_access_v<Atom, ::vecops::matmul::Operand::A, A> &&
              sme::is_packed_access_v<Atom, ::vecops::matmul::Operand::B, B>) {
            run_tiles_phased_shared_n_major<Atom, Policy>(
                m, n, k, a, b, c_input, c_output,
                acc_input, acc_output, route, scratch);
          } else {
            using Resources = typename std::remove_cvref_t<
                decltype(active)>::ActiveResources;
            auto active_a = tensor::rebind_active_resources<Resources>(a);
            auto active_b = tensor::rebind_active_resources<Resources>(b);
            auto active_c_input =
                tensor::rebind_active_resources<Resources>(c_input);
            auto active_c_output =
                tensor::rebind_active_resources<Resources>(c_output);
            auto active_acc_input =
                tensor::rebind_active_resources<Resources>(acc_input);
            auto active_acc_output =
                tensor::rebind_active_resources<Resources>(acc_output);
            run_tiles_phased_shared_n_major<Atom, Policy>(
                m, n, k, active_a, active_b,
                active_c_input, active_c_output,
                active_acc_input, active_acc_output, route, scratch);
            active_c_output.commit();
            active_acc_output.commit();
          }
        });
  }

  /**
   * @brief Resolve the compile-time kernel plan for one traversal.
   *
   * Unpacked operands always plan the generic path. Packed operands decide
   * FastPacked from the K-group count (`ceil_div(k, KPack)`, kept as a
   * Meta value so provenance survives):
   *
 * - BF16 with N <= 128 and 32 <= K < 64 uses 16 groups, calibrated by the
 *   A08 shallow-K batch probe without changing established long-K paths;
   * - wider BF16 and other atoms retain the conservative 32-group crossover;
   * - a constant/bounded group count is compared with that atom threshold;
   * - unconstrained: fast (see the tuning comment in place).
   */
  template <::vecops::matmul::Atom Atom, typename A, typename B,
            meta::ValueType M, meta::ValueType N, meta::ValueType K,
            typename Fn>
  VECOPS_ALWAYS_INLINE static void dispatch_plan(M, N, K k, Fn&& fn) {
    using MV = std::remove_cvref_t<M>;
    using NV = std::remove_cvref_t<N>;
    using KV = std::remove_cvref_t<K>;
    constexpr bool PrefetchLargeWorkingSet =
        sme::is_packed_access_v<Atom, ::vecops::matmul::Operand::A, A> &&
        sme::is_packed_access_v<Atom, ::vecops::matmul::Operand::B, B> &&
        sme::large_packed_prefetch_v<Atom, MV, NV, KV>;
    constexpr bool ShallowBF16 = std::same_as<
        Atom, ::vecops::matmul::SME_BF16F32> &&
        meta::upper_bound_at_most_v<NV, 128> &&
        meta::lower_bound_at_least_v<KV, 32> &&
        meta::upper_bound_at_most_v<KV, 63>;
    auto invoke = [&]<bool FastPacked>() VECOPS_INLINE_LAMBDA_NOEXCEPT {
      std::forward<Fn>(fn).template operator()<
          sme::KernelPlan<
              FastPacked, PrefetchLargeWorkingSet,
              FastPacked && ShallowBF16>>();
    };
    if constexpr (
        sme::is_packed_access_v<Atom, ::vecops::matmul::Operand::A, A> &&
        sme::is_packed_access_v<Atom, ::vecops::matmul::Operand::B, B>) {
      constexpr nint_t FastPackedKGroups = ShallowBF16 ? 16 : 32;
      constexpr nint_t KP = ::vecops::matmul::packing_t<
          Atom, ::vecops::matmul::Operand::A>::KPack;
      const auto k_groups = ceil_div(k, meta::cint<KP>);
      if constexpr (meta::is_singleton_v<decltype(k_groups)>) {
        invoke.template operator()<
            meta::singleton_value_v<decltype(k_groups)> >=
                FastPackedKGroups>();
      } else if constexpr (
          meta::has_upper_bound_v<decltype(k_groups)> &&
          meta::upper_bound_v<decltype(k_groups)> < FastPackedKGroups) {
        invoke.template operator()<false>();
      } else if constexpr (
          meta::has_lower_bound_v<decltype(k_groups)> &&
          meta::lower_bound_v<decltype(k_groups)> >= FastPackedKGroups) {
        invoke.template operator()<true>();
      } else {
        // A single direct packed-pointer loop is faster overall for an
        // unconstrained K and avoids cloning the complete tile traversal.
        invoke.template operator()<true>();
      }
    } else {
      invoke.template operator()<false>();
    }
  }

  template <::vecops::matmul::Atom Atom, typename A, typename B,
            meta::ValueType M, meta::ValueType N, meta::ValueType K,
            typename Fn>
  VECOPS_ALWAYS_INLINE static void dispatch_plan_n_major(
      M m, N n, K k, Fn&& fn) {
    dispatch_plan<Atom, A, B>(m, n, k, std::forward<Fn>(fn));
  }

  /// One Tile2D Case instantiation: forward to the ZA microkernel with the
  /// case's block shape and mask/exactness guarantees as template flags.
  template <::vecops::matmul::Atom Atom, typename Case, typename Plan,
            typename A, typename B, typename CInput, typename COutput>
  VECOPS_ALWAYS_INLINE static void run_case(
      const A& a, const B& b, const CInput& c_input, COutput& c_output,
      nint_t logical_m, nint_t logical_n, nint_t logical_k,
      nint_t m, nint_t n, nint_t active_m, nint_t active_n,
      void*) {
    sme::microkernel<
        Atom, Plan, Case::a, Case::b,
        Case::m_mask == kernel::loop::Tile2DMaskMode::unmasked,
        Case::n_mask == kernel::loop::Tile2DMaskMode::unmasked,
        Case::exact_blocks>(
            a, b, c_input, c_output,
            logical_m, logical_n, logical_k,
            m, n, active_m, active_n);
  }

  /**
   * One outlined split-K traversal per MC x NC x KC block.  The dynamic route
   * keeps first/middle/last in one function body, while outlining here rather
   * than at every ZA microkernel call lets the scheduler inline its selected
   * cases and avoids a call/return for every register tile.
   */
  template <::vecops::matmul::Atom Atom, typename Policy,
            meta::ValueType M, meta::ValueType N, meta::ValueType K,
            typename A, typename B,
            typename CInput, typename COutput,
            typename AccInput, typename AccOutput, typename Route>
  VECOPS_NOINLINE static void run_tiles_phased_shared(
      M m, N n, K k,
      const A& a, const B& b,
      const CInput& c_input, COutput& c_output,
      const AccInput& acc_input, AccOutput& acc_output,
      Route route, void* scratch) {
    static_assert(!matmul_details::static_accumulator_route_v<Route>);
    matmul_details::run_tiles_phased<Backend, Atom, Policy>(
        m, n, k, a, b, c_input, c_output,
        acc_input, acc_output, route, scratch);
  }

  template <::vecops::matmul::Atom Atom, typename Policy,
            meta::ValueType M, meta::ValueType N, meta::ValueType K,
            typename A, typename B, typename CInput, typename COutput,
            typename AccInput, typename AccOutput, typename Route>
  VECOPS_NOINLINE static void run_tiles_phased_shared_n_major(
      M m, N n, K k, const A& a, const B& b,
      const CInput& c_input, COutput& c_output,
      const AccInput& acc_input, AccOutput& acc_output,
      Route route, void* scratch) {
    static_assert(!matmul_details::static_accumulator_route_v<Route>);
    matmul_details::run_tiles_phased_n_major<Backend, Atom, Policy>(
        m, n, k, a, b, c_input, c_output,
        acc_input, acc_output, route, scratch);
  }

  template <::vecops::matmul::Atom Atom, typename Case, typename Plan,
            typename A, typename B,
            typename CInput, typename COutput,
            typename AccInput, typename AccOutput, typename Route>
  VECOPS_ALWAYS_INLINE static void run_case_phased(
      const A& a, const B& b,
      const CInput& c_input, COutput& c_output,
      const AccInput& acc_input, AccOutput& acc_output, Route route,
      nint_t logical_m, nint_t logical_n, nint_t logical_k,
      nint_t m, nint_t n, nint_t active_m, nint_t active_n,
      void*) {
    sme::microkernel_phased<
        Atom, Plan, Case::a, Case::b,
        Case::m_mask == kernel::loop::Tile2DMaskMode::unmasked,
        Case::n_mask == kernel::loop::Tile2DMaskMode::unmasked,
        Case::exact_blocks>(
            a, b, c_input, c_output, acc_input, acc_output, route,
            logical_m, logical_n, logical_k,
            m, n, active_m, active_n);
  }
};

} // namespace vecops::kernel::matmul_details

#endif // VECOPS_MATMUL_DETAILS_SME_BACKEND_H
