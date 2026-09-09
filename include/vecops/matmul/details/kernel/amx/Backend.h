//
// Copyright (c) vecops contributors.
//

#ifndef VECOPS_MATMUL_DETAILS_AMX_BACKEND_H
#define VECOPS_MATMUL_DETAILS_AMX_BACKEND_H

/**
 * @file vecops/matmul/details/kernel/amx/Backend.h
 * @brief Intel AMX (TMUL) matmul backend: tile microkernels plus AVX-512
 *        small-problem leaves, behind Backend<matmul_implementation::AMX>.
 *
 * Design intent — dispatch paths. run() picks one of four DispatchOwner
 * values from the Atom, the Meta extents, and the access shapes:
 *
 *  - General: the TMUL tile path. with_configuration() installs one
 *    TILECFG image for the whole traversal, then TileScheduler drives
 *    the Catalog microkernels below.
 *  - SmallVector: plain AVX-512 leaves (AVX512BF16 dot / VNNI dpbusd)
 *    for tiny problems where the tile-configuration and -load overheads
 *    would dominate; these do not need the Tiles resource at all.
 *  - FusedSmallBF16: like SmallVector but computed into a small values
 *    buffer so the elementwise C-input/C-output transforms stay fused
 *    in the epilogue (C input need not be zero).
 *  - PackedABTailSplit: a packed-A/packed-B BF16 shape with a short M
 *    tail; run() splits it into one exact 16-row bulk region plus one
 *    tail region, each under its own shortened TILECFG image.
 *
 * Design intent — two-tier owner selection. The *shape facts* decide in
 * two tiers. The compile-time tier (select_automatic_dispatch_owner /
 * select_dispatch_owner) fires a special owner only when the Meta
 * contracts prove every admitted shape is eligible. The run-time tier
 * covers what the types cannot decide: each leaf has an applicability
 * probe returning Applicability {always, never, runtime}; on `runtime`,
 * run() consults the measured thresholds in RuntimeDispatch.h
 * (runtime_amx_small_vector_profitable, ...) with the concrete m/n/k.
 * Family requests (Prefer/Require, SmallVector/ResidualSplit) ride on
 * the same probes: prefer falls back to the automatic route when the
 * family cannot serve the shape, require raises a check failure.
 *
 * Design intent — tile register map. The eight AMX tile registers are
 * assigned statically per Case (NM x NN output blocks):
 *
 *   t0 .. t(NM*NN-1)              C accumulator tiles (block row-major)
 *   t(NM*NN) .. t(NM*NN+NM-1)     A operand tiles
 *   t(NM*NN+NM) .. t(+NM+NN-1)    B operand tiles
 *
 * The budget NM*NN + NM + NN <= 8 is exactly what KernelProvider::power()
 * enforces; compute_tiles_impl()/compute_tile() spell the same mapping
 * into dot() indices, and Configuration::set_horizontal_rows() shortens
 * the C/A rows of the horizontal (1xN) families.
 *
 * Design intent — scratch layout. scratch_bytes() = 8 KiB + 63 sizes the
 * microkernel staging area, carved up by microkernel() as
 *
 *   [ a_buffers: NM x 1024 B ]  one 16-row x 64 B register image each
 *   [ b_buffers: NN x 1024 B ]
 *   [ c_buffers: Outputs x (16 x 16 TAcc elements) ]
 *
 * 8 KiB covers the largest Catalog family for 4-byte accumulators
 * (2x2: 4x1024 + 4x1024); the extra 63 bytes pad the caller's
 * allocation out to the 64 B alignment the tileload paths rely on.
 * Tail loads that cannot read straight from memory (transformed or
 * strided inputs) pack into these buffers first.
 */

#include <array>
#include <bit>
#include <cstddef>
#include <cstdint>
#include <type_traits>
#include <utility>

#include <immintrin.h>

#include "vecops/Assertion.h"
#include "vecops/execution/details/x86/Resources.h"
#include "vecops/matmul/Packing.h"
#include "vecops/matmul/details/kernel/RuntimeDispatch.h"
#include "vecops/matmul/details/kernel/amx/Atoms.h"
#include "vecops/kernel/Tile2D.h"
#include "vecops/kernel/details/transpose/generic/Transpose2D.h"
#include "vecops/matmul/details/kernel/TileScheduler.h"
#include "vecops/matmul/details/packing/amx/Pack.h"
#include "vecops/matmul/details/packing/generic/Pack.h"
#include "vecops/tensor/DataAccess.h"
#include "vecops/util/Math.h"
#include "vecops/vec/Vec.h"
#include "vecops/vec/details/amx/AMX.h"

namespace vecops::kernel::matmul_details::amx {

namespace generic = matmul_pack_details::generic;
namespace tile = kernel::loop;
namespace amx_intrinsics = vec::details::amx;

/// True when a C-input transform is the zero initializer, i.e. the
/// microkernel may skip loading C entirely and start from zero tiles.
template <typename T>
struct IsZeroTransform : std::false_type {};

template <typename Out, typename In>
struct IsZeroTransform<tensor::ZeroVecTransform<Out, In>> : std::true_type {};

// ---- Access-shape vocabulary shared by the candidate predicates and the
// tile loaders: spec/layout extraction, packed-panel detection, and the
// "rank two with unit-stride last axis" fast-path checks.

template <typename Access>
using SpecOf = std::remove_cvref_t<decltype(
    std::declval<const std::remove_cvref_t<Access>&>().spec())>;

template <typename Access>
using InputLayoutOf = typename SpecOf<Access>::InputLayout;

template <typename Access>
using OutputLayoutOf = typename SpecOf<Access>::OutputLayout;

/// True when the access already holds an AMX-packed panel for Side.
template <::vecops::matmul::Atom Atom, ::vecops::matmul::Operand Side, typename Access>
inline constexpr bool is_packed_access_v =
    ::vecops::matmul::is_packed_layout<Atom, Side, InputLayoutOf<Access>>();

/// Rank-two raw access whose last axis is contiguous: eligible for direct
/// tileload / direct pack without a DataAccess gather.
template <typename Access>
inline constexpr bool direct_row_major_input_v =
  generic::RawDirectAccess<Access> && Access::Rank == 2 &&
  std::same_as<tensor::stride_type_t<1, InputLayoutOf<Access>>, meta::Const<1>>;

/// Rank-two logical row whose final axis is physically contiguous, including
/// converted and transformed DataAccess objects.
template <typename Access>
inline constexpr bool row_contiguous_input_v =
  Access::Rank == 2 && std::same_as<tensor::stride_type_t<1, InputLayoutOf<Access>>,
                                    meta::Const<1>>;

/// Output-side twin of direct_row_major_input_v.
template <typename Access>
inline constexpr bool direct_row_major_output_v =
  generic::RawDirectAccess<Access> && Access::Rank == 2 &&
  std::same_as<tensor::stride_type_t<1, OutputLayoutOf<Access>>, meta::Const<1>>;

template <typename Access>
inline constexpr bool row_contiguous_output_v =
  Access::Rank == 2 && std::same_as<tensor::stride_type_t<1, OutputLayoutOf<Access>>, meta::Const<1>>;

/// Rank-two output whose leading (spatial) axis is contiguous, i.e. a
/// transposed C -- the store shape produced by orientation-swapped
/// problems; served by the column-block store path in store_c_tile.
template <typename Access>
inline constexpr bool column_contiguous_output_v =
  Access::Rank == 2 && std::same_as<tensor::stride_type_t<0, OutputLayoutOf<Access>>, meta::Const<1>>;

/** AMX defaults to one wide transform; an explicit wrapper takes priority. */
template <typename Access>
inline constexpr tensor::TransformStoreMode transform_store_mode_v = [] {
  constexpr auto requested = tensor::transform_store_mode_v<typename Access::Transform>;
  if constexpr (requested == tensor::TransformStoreMode::automatic)
    return tensor::TransformStoreMode::wide_transform;
  else
    return requested;
}();

template <typename Access>
inline constexpr bool transform_store_groups_tiles_v =
  transform_store_mode_v<Access> == tensor::TransformStoreMode::wide_transform ||
  transform_store_mode_v<Access> == tensor::TransformStoreMode::coalesced;

template <typename Access>
using TransformOf = typename std::remove_cvref_t<Access>::Transform;

#if defined(__AVX512BF16__)
inline constexpr bool SmallVectorBF16Available = true;
#else
inline constexpr bool SmallVectorBF16Available = false;
#endif
#if defined(__AVX512VNNI__)
inline constexpr bool SmallVectorI8Available = true;
#else
inline constexpr bool SmallVectorI8Available = false;
#endif

/// Static preconditions of the plain AVX-512 small-problem leaf: direct
/// row-major A/B/C, zero C input, and an AVX-512 dot instruction for the
/// Atom's types. Whether it actually wins over AMX is decided separately
/// by small_vector_guaranteed() / the runtime rules.
template <::vecops::matmul::Atom Atom, typename A, typename B,
          typename CInput, typename COutput>
inline constexpr bool small_vector_candidate_v =
#if defined(VECOPS_DISABLE_AMX_SMALL_VECTOR)
    false;
#else
    direct_row_major_input_v<A> && direct_row_major_input_v<B> &&
    direct_row_major_output_v<COutput> &&
    IsZeroTransform<TransformOf<CInput>>::value &&
    ((SmallVectorBF16Available &&
      std::same_as<Atom, ::vecops::matmul::AMX_BF16F32>) ||
     (SmallVectorI8Available &&
      (std::same_as<typename Atom::TA, int8_t> ||
       std::same_as<typename Atom::TA, uint8_t>) &&
      (std::same_as<typename Atom::TB, int8_t> ||
       std::same_as<typename Atom::TB, uint8_t>)));
#endif

/// Accumulating form used by split-K middle/last phases: direct native
/// accumulator input/output with no transforms.  The vector dot core adds the
/// prior value after reduction, so it remains ZA/TILECFG-free.
template <::vecops::matmul::Atom Atom, typename A, typename B,
          typename CInput, typename COutput>
inline constexpr bool small_vector_accumulate_candidate_v =
#if defined(VECOPS_DISABLE_AMX_SMALL_VECTOR)
    false;
#else
    direct_row_major_input_v<A> && direct_row_major_input_v<B> &&
    direct_row_major_input_v<CInput> && direct_row_major_output_v<COutput> &&
    std::same_as<TransformOf<CInput>, tensor::NoTransform> &&
    std::same_as<TransformOf<COutput>, tensor::NoTransform> &&
    std::same_as<typename CInput::MemoryElement, typename Atom::TAcc> &&
    std::same_as<typename COutput::MemoryElement, typename Atom::TAcc> &&
    ((SmallVectorBF16Available &&
      std::same_as<Atom, ::vecops::matmul::AMX_BF16F32>) ||
     (SmallVectorI8Available &&
      (std::same_as<typename Atom::TA, int8_t> ||
       std::same_as<typename Atom::TA, uint8_t>) &&
      (std::same_as<typename Atom::TB, int8_t> ||
       std::same_as<typename Atom::TB, uint8_t>)));
#endif

/// Static preconditions of the fused BF16 small leaf: like
/// small_vector_candidate_v but the C transforms only need to be
/// elementwise (fused into the epilogue) and A may arrive packed.
template <::vecops::matmul::Atom Atom, typename A, typename B,
          typename CInput, typename COutput>
inline constexpr bool fused_small_bf16_candidate_v =
#if defined(VECOPS_DISABLE_AMX_SMALL_VECTOR) || \
    defined(VECOPS_DISABLE_AMX_FUSED_SMALL_BF16)
    false;
#else
    SmallVectorBF16Available &&
    std::same_as<Atom, ::vecops::matmul::AMX_BF16F32> &&
    generic::RawDirectAccess<A> &&
    (direct_row_major_input_v<A> ||
     is_packed_access_v<Atom, ::vecops::matmul::Operand::A, A>) &&
    direct_row_major_input_v<B> &&
    CInput::Rank == 2 && COutput::Rank == 2 &&
    TransformOf<CInput>::is_elementwise &&
    TransformOf<COutput>::is_elementwise
#if !defined(VECOPS_DISABLE_AMX_FUSED_SMALL_DEDUP)
    // A direct zero-C/output type that can use the plain vector leaf already
    // routes every M*N<=16 problem there.  The fused predicate tests exactly
    // that area later, so instantiating its second compute/store leaf for the
    // overlapping type is unreachable and only multiplies code.
    && !small_vector_candidate_v<Atom, A, B, CInput, COutput>
#endif
    ;
#endif

/// Static preconditions of the residual-split traversal.  Operand packing,
/// dtypes, and C prologue/epilogue remain properties of the ordinary tile
/// access layer, so raw/packed inputs, transforms, and split-K phases all use
/// the same region splitter.
/// The concrete shapes it fires for are pinned in
/// select_automatic_dispatch_owner() and residual_split_shape_applicability().
template <::vecops::matmul::Atom Atom, typename A, typename B,
          typename CInput, typename COutput>
inline constexpr bool packed_ab_tail_split_candidate_v =
    CInput::Rank == 2 && COutput::Rank == 2;

/// Which implementation owns this leaf in run(). One value is chosen
/// statically when the Meta contracts suffice, otherwise at run time
/// from the same shape rules; see the file header for what each path
/// does.
enum class DispatchOwner {
  General,
  SmallVector,
  FusedSmallBF16,
  PackedABTailSplit,
};

// ---- Meta-extent probes used by the owner selection below. All of them
// are conservative: unknown bounds answer false, never "probably fine".

/// Extent is statically pinned to exactly Value.
template <meta::ValueType E, nint_t Value>
inline constexpr bool extent_is_v =
    meta::range_within_v<std::remove_cvref_t<E>, Value, Value>;

/// Every (m, n) admitted by the contracts has m*n <= Limit; false when
/// either upper bound is unknown (assume the worst, not the best).
template <meta::ValueType M, meta::ValueType N, nint_t Limit>
inline constexpr bool max_area_at_most_v = [] {
  using MV = std::remove_cvref_t<M>;
  using NV = std::remove_cvref_t<N>;
  if constexpr (meta::has_upper_bound_v<MV> &&
                meta::has_upper_bound_v<NV> &&
                meta::upper_bound_v<MV> > 0 &&
                meta::upper_bound_v<NV> > 0) {
    return meta::upper_bound_v<MV> <=
        Limit / meta::upper_bound_v<NV>;
  } else {
    return false;
  }
}();

/// E % Alignment is statically known to be 0 or 1 (aligned or singleton
/// extents only; anything wider is conservatively rejected).
template <meta::ValueType E, nint_t Alignment>
inline constexpr bool remainder_at_most_one_v = [] {
  using EV = std::remove_cvref_t<E>;
  if constexpr (EV::aligns(Alignment)) {
    return true;
  } else if constexpr (meta::is_bounded_v<EV> &&
                       meta::lower_bound_v<EV> == meta::upper_bound_v<EV>) {
    return meta::lower_bound_v<EV> % Alignment <= 1;
  } else {
    return false;
  }
}();

/// Value is outside E's admitted range entirely (constant extents compare
/// by value; Dynamic extents by their conform() contract).
template <meta::ValueType E, nint_t Value>
inline constexpr bool extent_excludes_v = [] {
  using EV = std::remove_cvref_t<E>;
  if constexpr (EV::is_const) return EV::value != Value;
  else return !EV::conforms(Value);
}();

/// Decide, purely from the Meta contracts, that every problem the caller
/// can pose is better served by the AVX-512 small-problem leaf than by
/// the AMX tile path (TILECFG/LDTILECFG plus tile-load latency dominate
/// once the whole result fits a few vector accumulators).
///
/// The shape classes considered, per Atom family:
///  - tiny blocks whose whole M*N area fits one/few vector accumulators
///    (area <= 16 / 32 / 64 with a matching K ceiling);
///  - Skinny: rank-one problems (1 x <=64) routed as GEMV/GEMh;
///  - LargeM1/LargeN1 + LargeSkinny: long rank-one strips (1 x 128..4096)
///    with a long, alignment-friendly K, where one row of A or B stays
///    resident and TMUL setup never pays off.
///
/// All numeric cutoffs are routing constants chosen by measurement, not
/// architectural limits of either instruction set — moving them only
/// changes which owner wins, never correctness.
template <::vecops::matmul::Atom Atom,
          meta::ValueType M, meta::ValueType N, meta::ValueType K>
consteval bool small_vector_guaranteed() {
  using MV = std::remove_cvref_t<M>;
  using NV = std::remove_cvref_t<N>;
  using KV = std::remove_cvref_t<K>;
  if constexpr (!meta::lower_bound_at_least_v<MV, 1> ||
                !meta::lower_bound_at_least_v<NV, 1> ||
                !meta::lower_bound_at_least_v<KV, 1>) {
    return false;
  } else {
    constexpr bool Skinny =
        (extent_is_v<MV, 1> && meta::upper_bound_at_most_v<NV, 64>) ||
        (extent_is_v<NV, 1> && meta::upper_bound_at_most_v<MV, 64>);
    constexpr bool LargeM1 = extent_is_v<MV, 1> &&
        meta::lower_bound_at_least_v<NV, 128> &&
        meta::upper_bound_at_most_v<NV, 4096>;
    constexpr bool LargeN1 = extent_is_v<NV, 1> &&
        meta::lower_bound_at_least_v<MV, 128> &&
        meta::upper_bound_at_most_v<MV, 4096>;
    if constexpr (std::same_as<Atom, ::vecops::matmul::AMX_BF16F32>) {
      constexpr bool LargeSkinny =
          meta::lower_bound_at_least_v<KV, 32> &&
          remainder_at_most_one_v<KV, 32> &&
          (LargeM1 ||
           (LargeN1 && extent_excludes_v<KV, 256>));
      return max_area_at_most_v<MV, NV, 16> || Skinny ||
          (max_area_at_most_v<MV, NV, 32> &&
           meta::upper_bound_at_most_v<KV, 256>) ||
          (max_area_at_most_v<MV, NV, 64> &&
           meta::upper_bound_at_most_v<KV, 64>) ||
          LargeSkinny;
    } else if constexpr (std::same_as<
                             typename Atom::TA, typename Atom::TB>) {
      constexpr bool LargeSkinny =
          meta::lower_bound_at_least_v<KV, 64> &&
          remainder_at_most_one_v<KV, 64> &&
          (LargeM1 || LargeN1);
      return max_area_at_most_v<MV, NV, 32> || Skinny || LargeSkinny;
    } else {
      constexpr bool LargeSkinny =
          meta::lower_bound_at_least_v<KV, 64> &&
          remainder_at_most_one_v<KV, 64> &&
          (LargeM1 ||
           (LargeN1 && extent_excludes_v<KV, 64>));
      return max_area_at_most_v<MV, NV, 32> || Skinny ||
          (max_area_at_most_v<MV, NV, 64> &&
           meta::upper_bound_at_most_v<KV, 128>) ||
          LargeSkinny;
    }
  }
}

/// Runtime-rules shape class for this Atom's plain vector leaf (see
/// RuntimeDispatch.h): the measured thresholds differ per instruction
/// mix, so the same m/n/k can be profitable for bf16 but not for i8.
template <::vecops::matmul::Atom Atom>
inline constexpr SmallVectorShape small_vector_shape_v = [] {
  if constexpr (std::same_as<Atom, ::vecops::matmul::AMX_BF16F32>)
    return SmallVectorShape::amx_bf16;
  else if constexpr (std::same_as<typename Atom::TA, typename Atom::TB>)
    return SmallVectorShape::amx_i8_same_sign;
  else
    return SmallVectorShape::amx_i8_mixed_sign;
}();

/// Tri-state verdict on the plain vector leaf for the shape contracts
/// M/N/K: always (small_vector_guaranteed proved it), never (a singleton
/// problem the runtime rules reject), or runtime (decide with the
/// concrete m/n/k in run()).
template <::vecops::matmul::Atom Atom,
          meta::ValueType M, meta::ValueType N, meta::ValueType K>
consteval ::vecops::matmul::details::Applicability
small_vector_shape_applicability() {
  using MV = std::remove_cvref_t<M>;
  using NV = std::remove_cvref_t<N>;
  using KV = std::remove_cvref_t<K>;
  if constexpr (small_vector_guaranteed<Atom, MV, NV, KV>()) {
    return ::vecops::matmul::details::Applicability::always;
  } else if constexpr (meta::is_singleton_v<MV> &&
                       meta::is_singleton_v<NV> &&
                       meta::is_singleton_v<KV>) {
    constexpr bool Applicable =
        runtime_dispatch_rules::amx_small_vector_profitable(
            small_vector_shape_v<Atom>,
            meta::singleton_value_v<MV>, meta::singleton_value_v<NV>,
            meta::singleton_value_v<KV>);
    return Applicable
        ? ::vecops::matmul::details::Applicability::always
        : ::vecops::matmul::details::Applicability::never;
  } else {
    return ::vecops::matmul::details::Applicability::runtime;
  }
}

/// Tri-state verdict for area-bounded leaves (fused small BF16): always
/// when the contracts cap M*N at Limit, never for singleton problems
/// above it, runtime otherwise.
template <meta::ValueType M, meta::ValueType N, meta::ValueType K,
          nint_t Limit>
consteval ::vecops::matmul::details::Applicability
small_area_applicability() {
  using MV = std::remove_cvref_t<M>;
  using NV = std::remove_cvref_t<N>;
  using KV = std::remove_cvref_t<K>;
  if constexpr (meta::lower_bound_at_least_v<MV, 1> &&
                meta::lower_bound_at_least_v<NV, 1> &&
                meta::lower_bound_at_least_v<KV, 1> &&
                max_area_at_most_v<MV, NV, Limit>) {
    return ::vecops::matmul::details::Applicability::always;
  } else if constexpr (meta::is_singleton_v<MV> &&
                       meta::is_singleton_v<NV> &&
                       meta::is_singleton_v<KV>) {
    constexpr bool Applicable =
        runtime_dispatch_rules::area_at_most(
            meta::singleton_value_v<MV>, meta::singleton_value_v<NV>, Limit) &&
        meta::singleton_value_v<KV> > 0;
    return Applicable
        ? ::vecops::matmul::details::Applicability::always
        : ::vecops::matmul::details::Applicability::never;
  } else {
    return ::vecops::matmul::details::Applicability::runtime;
  }
}

/// Tri-state verdict on whether the contracts admit any solvable problem
/// at all (positive M/N, non-negative K): never when an upper bound rules
/// out every valid shape, always when the lower bounds already guarantee
/// validity, runtime when only the actual extents know.
template <meta::ValueType M, meta::ValueType N, meta::ValueType K>
consteval ::vecops::matmul::details::Applicability
valid_problem_applicability() {
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
    return Applicability::always;
  } else {
    return Applicability::runtime;
  }
}

/// Support gate for area-bounded leaves: like small_area_applicability
/// but composed with problem validity — an invalid contract family is
/// never "supported" even when its area fits (so a required family
/// rejects early instead of running a degenerate problem).
template <meta::ValueType M, meta::ValueType N, meta::ValueType K,
          nint_t Limit>
consteval ::vecops::matmul::details::Applicability
bounded_area_support_applicability() {
  using Applicability = ::vecops::matmul::details::Applicability;
  constexpr auto Valid = valid_problem_applicability<M, N, K>();
  if constexpr (Valid == Applicability::never) {
    return Applicability::never;
  } else if constexpr (meta::lower_bound_at_least_v<M, 1> &&
                       meta::lower_bound_at_least_v<N, 1> &&
                       max_area_at_most_v<M, N, Limit>) {
    return Valid;
  } else if constexpr (meta::is_singleton_v<M> &&
                       meta::is_singleton_v<N>) {
    constexpr bool Fits = runtime_dispatch_rules::area_at_most(
        meta::singleton_value_v<M>, meta::singleton_value_v<N>, Limit);
    return Fits ? Valid : Applicability::never;
  } else {
    return Applicability::runtime;
  }
}

/// Support gate for the residual split: either spatial axis may contain a
/// non-empty residual beyond an aligned 16-element bulk region.
template <meta::ValueType M, meta::ValueType N, meta::ValueType K>
consteval ::vecops::matmul::details::Applicability
residual_split_support_applicability() {
  using Applicability = ::vecops::matmul::details::Applicability;
  using MV = std::remove_cvref_t<M>;
  using NV = std::remove_cvref_t<N>;
  constexpr auto Valid = valid_problem_applicability<M, N, K>();
  constexpr bool MHasKnownTail = [] {
    if constexpr (meta::is_singleton_v<MV>)
      return meta::singleton_value_v<MV> > 16 &&
          meta::singleton_value_v<MV> % 16 != 0;
    else
      return false;
  }();
  constexpr bool NHasKnownTail = [] {
    if constexpr (meta::is_singleton_v<NV>)
      return meta::singleton_value_v<NV> > 16 &&
          meta::singleton_value_v<NV> % 16 != 0;
    else
      return false;
  }();
  if constexpr (Valid == Applicability::never ||
                ((meta::has_upper_bound_v<MV> &&
                  meta::upper_bound_v<MV> <= 16) &&
                 (meta::has_upper_bound_v<NV> &&
                  meta::upper_bound_v<NV> <= 16))) {
    return Applicability::never;
  } else if constexpr (MHasKnownTail || NHasKnownTail) {
    return Valid;
  } else {
    return Applicability::runtime;
  }
}

/// Shape verdict for the residual split's measured shapes: singleton
/// problems consult the runtime rules; contracts that exclude every
/// pinned (M, N, K) combination are never; the rest decide at run time.
template <meta::ValueType M, meta::ValueType N, meta::ValueType K>
consteval ::vecops::matmul::details::Applicability
residual_split_shape_applicability() {
  using MV = std::remove_cvref_t<M>;
  using NV = std::remove_cvref_t<N>;
  using KV = std::remove_cvref_t<K>;
  if constexpr (meta::is_singleton_v<MV> &&
                meta::is_singleton_v<NV> &&
                meta::is_singleton_v<KV>) {
    constexpr bool Applicable =
        runtime_dispatch_rules::amx_residual_split_profitable(
            meta::singleton_value_v<MV>, meta::singleton_value_v<NV>,
            meta::singleton_value_v<KV>);
    return Applicable
        ? ::vecops::matmul::details::Applicability::always
        : ::vecops::matmul::details::Applicability::never;
  } else if constexpr (extent_excludes_v<KV, 1024>) {
    return ::vecops::matmul::details::Applicability::never;
  } else {
    return ::vecops::matmul::details::Applicability::runtime;
  }
}

/// Compile-time owner selection (the automatic half; family requests are
/// resolved on top of it in select_dispatch_owner). Order matters:
/// SmallVector first, then the fused BF16 leaf, then the residual split
/// — each with its own static shape gate.
template <::vecops::matmul::Atom Atom, bool AllowTailSplit,
          meta::ValueType M, meta::ValueType N, meta::ValueType K,
          typename A, typename B, typename CInput, typename COutput>
consteval DispatchOwner select_automatic_dispatch_owner() {
  using MV = std::remove_cvref_t<M>;
  using NV = std::remove_cvref_t<N>;
  using KV = std::remove_cvref_t<K>;
  if constexpr ((small_vector_candidate_v<
                     Atom, A, B, CInput, COutput> ||
                 small_vector_accumulate_candidate_v<
                     Atom, A, B, CInput, COutput>) &&
                small_vector_guaranteed<Atom, MV, NV, KV>()) {
    return DispatchOwner::SmallVector;
  } else if constexpr (fused_small_bf16_candidate_v<
                           Atom, A, B, CInput, COutput> &&
                       meta::lower_bound_at_least_v<MV, 1> &&
                       meta::lower_bound_at_least_v<NV, 1> &&
                       meta::lower_bound_at_least_v<KV, 1> &&
                       max_area_at_most_v<MV, NV, 16>) {
    return DispatchOwner::FusedSmallBF16;
  } else if constexpr (
      AllowTailSplit &&
      std::same_as<Atom, ::vecops::matmul::AMX_BF16F32> &&
      is_packed_access_v<Atom, ::vecops::matmul::Operand::A, A> &&
      is_packed_access_v<Atom, ::vecops::matmul::Operand::B, B> &&
      packed_ab_tail_split_candidate_v<
          Atom, A, B, CInput, COutput> &&
      (((extent_is_v<MV, 17> || extent_is_v<MV, 18> ||
         extent_is_v<MV, 20>) &&
        (extent_is_v<NV, 33> || extent_is_v<NV, 47> ||
         extent_is_v<NV, 48>)) ||
       ((extent_is_v<NV, 17> || extent_is_v<NV, 18> ||
         extent_is_v<NV, 20>) &&
        (extent_is_v<MV, 33> || extent_is_v<MV, 47> ||
         extent_is_v<MV, 48>))) &&
      extent_is_v<KV, 1024>) {
    // One aligned bulk region plus a short residual on whichever spatial
    // axis wastes fewer tile lanes.  The measured catalog and its exact
    // transpose share this owner.
    return DispatchOwner::PackedABTailSplit;
  } else {
    // Unconstrained Dynamic extents intentionally own only the general AMX
    // implementation. A bounded/aligned Dynamic type may select a special
    // owner only when every value admitted by its Meta contract is eligible.
    return DispatchOwner::General;
  }
}

/// Membership table: which DispatchOwner values belong to which public
/// kernel_family. Used to validate requested (as opposed to automatic)
/// family dispatch.
template <typename Family, DispatchOwner Owner>
inline constexpr bool dispatch_owner_in_family_v =
    (std::same_as<Family, ::vecops::matmul::kernel_family::General> &&
     Owner == DispatchOwner::General) ||
    (std::same_as<Family, ::vecops::matmul::kernel_family::SmallVector> &&
     (Owner == DispatchOwner::SmallVector ||
      Owner == DispatchOwner::FusedSmallBF16)) ||
    (std::same_as<Family, ::vecops::matmul::kernel_family::ResidualSplit> &&
     Owner == DispatchOwner::PackedABTailSplit);

/// Tri-state support verdict for a public family on this leaf's static
/// facts: General/WholeProblem always apply; SmallVector applies when a
/// raw or fused vector candidate matches and the shape probes admit it;
/// ResidualSplit applies when tail splitting is allowed, the packed
/// candidate matches, and either spatial axis has a residual. Used by run() to
/// honour Prefer/Require requests that the automatic owner does not
/// already satisfy.
template <typename Family,
          ::vecops::matmul::Atom Atom, bool AllowTailSplit,
          meta::ValueType M, meta::ValueType N, meta::ValueType K,
          typename A, typename B, typename CInput, typename COutput>
consteval ::vecops::matmul::details::Applicability
family_applicability() {
  using Applicability = ::vecops::matmul::details::Applicability;
  if constexpr (std::same_as<
                    Family, ::vecops::matmul::kernel_family::General> ||
                std::same_as<
                    Family, ::vecops::matmul::kernel_family::WholeProblem>) {
    return Applicability::always;
  } else if constexpr (std::same_as<
                           Family,
                           ::vecops::matmul::kernel_family::SmallVector>) {
    constexpr auto Raw = (small_vector_candidate_v<
        Atom, A, B, CInput, COutput> ||
        small_vector_accumulate_candidate_v<
            Atom, A, B, CInput, COutput>)
        ? valid_problem_applicability<M, N, K>()
        : Applicability::never;
    constexpr auto Fused = fused_small_bf16_candidate_v<
        Atom, A, B, CInput, COutput>
        ? bounded_area_support_applicability<M, N, K, 16>()
        : Applicability::never;
    return Raw || Fused;
  } else if constexpr (std::same_as<
                           Family,
                           ::vecops::matmul::kernel_family::ResidualSplit>) {
    if constexpr (AllowTailSplit && packed_ab_tail_split_candidate_v<
                      Atom, A, B, CInput, COutput>)
      return residual_split_support_applicability<M, N, K>();
    else
      return Applicability::never;
  } else {
    return Applicability::never;
  }
}

/// Whether the runtime-dispatch tier may route into CandidateFamily
/// under this request: WholeProblem always permits it, General never
/// (the tile path is already the owner), Require permits only the exact
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
    // Prefer falls back to the complete automatic route.
    return true;
  }
}();

/// Resolve a family request over the automatic owner: WholeProblem takes
/// the automatic pick verbatim, General forces the tile path, and any
/// other family may only confirm the automatic owner (asserting when a
/// required family is not applicable; run() additionally serves runtime-
/// decidable requests through the tiered probes above).
template <typename FamilyDispatch,
          ::vecops::matmul::Atom Atom, bool AllowTailSplit,
          meta::ValueType M, meta::ValueType N, meta::ValueType K,
          typename A, typename B, typename CInput, typename COutput>
consteval DispatchOwner select_dispatch_owner() {
  constexpr auto AutomaticOwner = select_automatic_dispatch_owner<
      Atom, AllowTailSplit, M, N, K, A, B, CInput, COutput>();
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
            Family, Atom, AllowTailSplit, M, N, K,
            A, B, CInput, COutput>() !=
            ::vecops::matmul::details::Applicability::never;
    static_assert(
        !FamilyDispatch::required || Applicable,
        "required matmul kernel family is not applicable to this AMX leaf");
    // Prefer retains the proven automatic owner when the requested family is
    // unavailable.  When applicable, AutomaticOwner already names that leaf.
    return AutomaticOwner;
  }
}

/**
 * Dense vector-dot leaf shared by every AMX-side SmallVector dtype.
 *
 * All ISA selection, signedness compensation, and grouped accumulation live
 * in vec::widening_dot. Keeping the matrix traversal here makes one function
 * body usable by every runtime shape admitted by the family selector.
 */
template <bool Accumulate = false, typename TA, typename TB, typename Acc>
inline VECOPS_NOINLINE VECOPS_FUNCTION_ALIGN(64)
#if defined(COMPILER_GCC)
__attribute__((noclone))
#endif
void
small_vector_dense_matmul(
    const TA* a, nint_t a_stride,
    const TB* b, nint_t b_stride,
    Acc* c, nint_t c_stride,
    nint_t m, nint_t n, nint_t k,
    const Acc* prior = nullptr, nint_t prior_stride = 0) {
  using ATag = vec::ScalableTag<TA, 0>;
  using BTag = vec::ScalableTag<TB, 0>;
  using AccTag = std::conditional_t<
      std::same_as<TA, Acc>, ATag, vec::ViewAs<Acc, ATag>>;
  for (nint_t i = 0; i < m; ++i) {
    nint_t j = 0;
    for (; j + 4 <= n; j += 4) {
      auto accumulator0 = vec::zeros(AccTag{});
      auto accumulator1 = accumulator0;
      auto accumulator2 = accumulator0;
      auto accumulator3 = accumulator0;
      nint_t kk = 0;
      constexpr nint_t VectorK = vec::size(ATag{});
      for (; kk + VectorK <= k; kk += VectorK) {
        const auto av = vec::load(ATag{}, a + i * a_stride + kk);
        const auto bv0 = vec::load(BTag{}, b + (j + 0) * b_stride + kk);
        const auto bv1 = vec::load(BTag{}, b + (j + 1) * b_stride + kk);
        const auto bv2 = vec::load(BTag{}, b + (j + 2) * b_stride + kk);
        const auto bv3 = vec::load(BTag{}, b + (j + 3) * b_stride + kk);
        accumulator0 = vec::widening_dot(
            AccTag{}, ATag{}, BTag{}, av, bv0, accumulator0);
        accumulator1 = vec::widening_dot(
            AccTag{}, ATag{}, BTag{}, av, bv1, accumulator1);
        accumulator2 = vec::widening_dot(
            AccTag{}, ATag{}, BTag{}, av, bv2, accumulator2);
        accumulator3 = vec::widening_dot(
            AccTag{}, ATag{}, BTag{}, av, bv3, accumulator3);
      }
      if (kk < k) {
        const nint_t active = k - kk;
        const auto av = vec::load(
            ATag{}, a + i * a_stride + kk,
            vec::opt::first(active), vec::opt::zero);
        const auto bv0 = vec::load(
            BTag{}, b + (j + 0) * b_stride + kk,
            vec::opt::first(active), vec::opt::zero);
        const auto bv1 = vec::load(
            BTag{}, b + (j + 1) * b_stride + kk,
            vec::opt::first(active), vec::opt::zero);
        const auto bv2 = vec::load(
            BTag{}, b + (j + 2) * b_stride + kk,
            vec::opt::first(active), vec::opt::zero);
        const auto bv3 = vec::load(
            BTag{}, b + (j + 3) * b_stride + kk,
            vec::opt::first(active), vec::opt::zero);
        accumulator0 = vec::widening_dot(
            AccTag{}, ATag{}, BTag{}, av, bv0, accumulator0);
        accumulator1 = vec::widening_dot(
            AccTag{}, ATag{}, BTag{}, av, bv1, accumulator1);
        accumulator2 = vec::widening_dot(
            AccTag{}, ATag{}, BTag{}, av, bv2, accumulator2);
        accumulator3 = vec::widening_dot(
            AccTag{}, ATag{}, BTag{}, av, bv3, accumulator3);
      }
      auto store = [&](nint_t column, Acc value) VECOPS_INLINE_LAMBDA {
        if constexpr (Accumulate)
          value += prior[i * prior_stride + column];
        c[i * c_stride + column] = value;
      };
      store(j + 0, vec::reduce_add(AccTag{}, accumulator0));
      store(j + 1, vec::reduce_add(AccTag{}, accumulator1));
      store(j + 2, vec::reduce_add(AccTag{}, accumulator2));
      store(j + 3, vec::reduce_add(AccTag{}, accumulator3));
    }
    for (; j < n; ++j) {
      auto accumulator0 = vec::zeros(AccTag{});
      auto accumulator1 = accumulator0;
      auto accumulator2 = accumulator0;
      auto accumulator3 = accumulator0;
      nint_t kk = 0;
      constexpr nint_t VectorK = vec::size(ATag{});
      constexpr nint_t UnrolledK = 4 * VectorK;
      for (; kk + UnrolledK <= k; kk += UnrolledK) {
        const auto av0 = vec::load(ATag{}, a + i * a_stride + kk);
        const auto bv0 = vec::load(BTag{}, b + j * b_stride + kk);
        const auto av1 = vec::load(ATag{}, a + i * a_stride + kk + VectorK);
        const auto bv1 = vec::load(BTag{}, b + j * b_stride + kk + VectorK);
        const auto av2 = vec::load(
            ATag{}, a + i * a_stride + kk + 2 * VectorK);
        const auto bv2 = vec::load(
            BTag{}, b + j * b_stride + kk + 2 * VectorK);
        const auto av3 = vec::load(
            ATag{}, a + i * a_stride + kk + 3 * VectorK);
        const auto bv3 = vec::load(
            BTag{}, b + j * b_stride + kk + 3 * VectorK);
        accumulator0 = vec::widening_dot(
            AccTag{}, ATag{}, BTag{}, av0, bv0, accumulator0);
        accumulator1 = vec::widening_dot(
            AccTag{}, ATag{}, BTag{}, av1, bv1, accumulator1);
        accumulator2 = vec::widening_dot(
            AccTag{}, ATag{}, BTag{}, av2, bv2, accumulator2);
        accumulator3 = vec::widening_dot(
            AccTag{}, ATag{}, BTag{}, av3, bv3, accumulator3);
      }
      for (; kk + VectorK <= k; kk += VectorK) {
        const auto av = vec::load(ATag{}, a + i * a_stride + kk);
        const auto bv = vec::load(BTag{}, b + j * b_stride + kk);
        accumulator0 = vec::widening_dot(
            AccTag{}, ATag{}, BTag{}, av, bv, accumulator0);
      }
      if (kk < k) {
        const nint_t active = k - kk;
        const auto av = vec::load(
            ATag{}, a + i * a_stride + kk,
            vec::opt::first(active), vec::opt::zero);
        const auto bv = vec::load(
            BTag{}, b + j * b_stride + kk,
            vec::opt::first(active), vec::opt::zero);
        accumulator0 = vec::widening_dot(
            AccTag{}, ATag{}, BTag{}, av, bv, accumulator0);
      }
      const auto accumulator01 = vec::add(
          AccTag{}, accumulator0, accumulator1);
      const auto accumulator23 = vec::add(
          AccTag{}, accumulator2, accumulator3);
      const auto accumulator = vec::add(
          AccTag{}, accumulator01, accumulator23);
      Acc value = vec::reduce_add(AccTag{}, accumulator);
      if constexpr (Accumulate) value += prior[i * prior_stride + j];
      c[i * c_stride + j] = value;
    }
  }
}

/**
 * Tall-skinny twin of small_vector_dense_matmul for N == 1.
 *
 * The ordinary traversal groups four adjacent N outputs and shares one A
 * vector.  Applying the exact transposed schedule here groups four M outputs
 * and shares one B vector; this avoids loading the same B row once per output
 * row on GEMV-column shapes.  The scalar remainder reuses the dense kernel so
 * both paths keep identical tail and signedness behavior.
 */
template <bool Accumulate = false, typename TA, typename TB, typename Acc>
inline VECOPS_NOINLINE VECOPS_FUNCTION_ALIGN(64)
#if defined(COMPILER_GCC)
__attribute__((noclone))
#endif
void
small_vector_tall_matmul(
    const TA* a, nint_t a_stride,
    const TB* b, nint_t b_stride,
    Acc* c, nint_t c_stride,
    nint_t m, nint_t k,
    const Acc* prior = nullptr, nint_t prior_stride = 0) {
  using ATag = vec::ScalableTag<TA, 0>;
  using BTag = vec::ScalableTag<TB, 0>;
  using AccTag = std::conditional_t<
      std::same_as<TA, Acc>, ATag, vec::ViewAs<Acc, ATag>>;
  constexpr nint_t VectorK = vec::size(ATag{});
  nint_t i = 0;
  for (; i + 4 <= m; i += 4) {
    auto accumulator0 = vec::zeros(AccTag{});
    auto accumulator1 = accumulator0;
    auto accumulator2 = accumulator0;
    auto accumulator3 = accumulator0;
    nint_t kk = 0;
    for (; kk + VectorK <= k; kk += VectorK) {
      const auto bv = vec::load(BTag{}, b + kk);
      const auto av0 = vec::load(ATag{}, a + (i + 0) * a_stride + kk);
      const auto av1 = vec::load(ATag{}, a + (i + 1) * a_stride + kk);
      const auto av2 = vec::load(ATag{}, a + (i + 2) * a_stride + kk);
      const auto av3 = vec::load(ATag{}, a + (i + 3) * a_stride + kk);
      accumulator0 = vec::widening_dot(
          AccTag{}, ATag{}, BTag{}, av0, bv, accumulator0);
      accumulator1 = vec::widening_dot(
          AccTag{}, ATag{}, BTag{}, av1, bv, accumulator1);
      accumulator2 = vec::widening_dot(
          AccTag{}, ATag{}, BTag{}, av2, bv, accumulator2);
      accumulator3 = vec::widening_dot(
          AccTag{}, ATag{}, BTag{}, av3, bv, accumulator3);
    }
    if (kk < k) {
      const nint_t active = k - kk;
      const auto bv = vec::load(
          BTag{}, b + kk, vec::opt::first(active), vec::opt::zero);
      const auto av0 = vec::load(
          ATag{}, a + (i + 0) * a_stride + kk,
          vec::opt::first(active), vec::opt::zero);
      const auto av1 = vec::load(
          ATag{}, a + (i + 1) * a_stride + kk,
          vec::opt::first(active), vec::opt::zero);
      const auto av2 = vec::load(
          ATag{}, a + (i + 2) * a_stride + kk,
          vec::opt::first(active), vec::opt::zero);
      const auto av3 = vec::load(
          ATag{}, a + (i + 3) * a_stride + kk,
          vec::opt::first(active), vec::opt::zero);
      accumulator0 = vec::widening_dot(
          AccTag{}, ATag{}, BTag{}, av0, bv, accumulator0);
      accumulator1 = vec::widening_dot(
          AccTag{}, ATag{}, BTag{}, av1, bv, accumulator1);
      accumulator2 = vec::widening_dot(
          AccTag{}, ATag{}, BTag{}, av2, bv, accumulator2);
      accumulator3 = vec::widening_dot(
          AccTag{}, ATag{}, BTag{}, av3, bv, accumulator3);
    }
    auto store = [&](nint_t row, Acc value) VECOPS_INLINE_LAMBDA {
      if constexpr (Accumulate) value += prior[row * prior_stride];
      c[row * c_stride] = value;
    };
    store(i + 0, vec::reduce_add(AccTag{}, accumulator0));
    store(i + 1, vec::reduce_add(AccTag{}, accumulator1));
    store(i + 2, vec::reduce_add(AccTag{}, accumulator2));
    store(i + 3, vec::reduce_add(AccTag{}, accumulator3));
  }
  if (i < m) {
    small_vector_dense_matmul<Accumulate>(
        a + i * a_stride, a_stride, b, b_stride,
        c + i * c_stride, c_stride, m - i, nint_t{1}, k,
        prior == nullptr ? nullptr : prior + i * prior_stride,
        prior_stride);
  }
}

/// Fused twin of small_vector_dense_matmul for BF16: computes into a
/// row-major values buffer (max 16x16, the fused leaf's whole problem)
/// so the caller can apply the C transforms afterwards. PackedA switches
/// the A addressing to the AMX packed-panel layout; masked loads handle
/// the K remainder, so arbitrary k is accepted.
template <bool PackedA>
inline VECOPS_NOINLINE VECOPS_FUNCTION_ALIGN(64)
#if defined(COMPILER_GCC)
__attribute__((noclone))
#endif
void small_bf16_fused_compute(
    const bfloat16_t* a, nint_t a_stride,
    const bfloat16_t* b, nint_t b_stride,
    float32_t* values, nint_t m, nint_t n, nint_t k) {
  using InputTag = vec::ScalableTag<bfloat16_t, 0>;
  using AccTag = vec::ViewAs<float32_t, InputTag>;
  const nint_t k_tile = vec::size(InputTag{});
  const nint_t k_tiles = ceil_div(k, k_tile);
  for (nint_t i = 0; i < m; ++i) {
    for (nint_t j = 0; j < n; ++j) {
      auto accumulator = vec::zeros(AccTag{});
      nint_t kk = 0;
      for (; kk + k_tile <= k; kk += k_tile) {
        const auto* a_pointer = [&] {
          if constexpr (PackedA) {
            // Packed panel layout: [panel][k-tile][16 rows x k_tile
            // elems]; row i lives at (panel * k_tiles + kk/k_tile) *
            // (16 * k_tile) + (i % 16) * k_tile.
            const nint_t panel = i / 16;
            const nint_t lane = i % 16;
            return a +
                (panel * k_tiles + kk / k_tile) * 16 * k_tile +
                lane * k_tile;
          } else {
            return a + i * a_stride + kk;
          }
        }();
        const auto av = vec::load(InputTag{}, a_pointer);
        const auto bv = vec::load(InputTag{}, b + j * b_stride + kk);
        accumulator = vec::widening_dot(
            AccTag{}, InputTag{}, InputTag{}, av, bv, accumulator);
      }
      if (kk < k) {
        const nint_t active = k - kk;
        const auto* a_pointer = [&] {
          if constexpr (PackedA) {
            const nint_t panel = i / 16;
            const nint_t lane = i % 16;
            return a +
                (panel * k_tiles + kk / k_tile) * 16 * k_tile +
                lane * k_tile;
          } else {
            return a + i * a_stride + kk;
          }
        }();
        const auto av = vec::load(
            InputTag{}, a_pointer,
            vec::opt::first(active), vec::opt::zero);
        const auto bv = vec::load(
            InputTag{}, b + j * b_stride + kk,
            vec::opt::first(active), vec::opt::zero);
        accumulator = vec::widening_dot(
            AccTag{}, InputTag{}, InputTag{}, av, bv, accumulator);
      }
      values[i * n + j] = vec::reduce_add(AccTag{}, accumulator);
    }
  }
}

/// Dispatch the SmallVector owner to the dense vector leaf; strides come
/// straight from the (already direct) accesses and the element types
/// from the Atom.
template <::vecops::matmul::Atom Atom, typename A, typename B, typename COutput>
VECOPS_ALWAYS_INLINE void run_small_vector(
    const A& a, const B& b, COutput& c_output,
    nint_t m, nint_t n, nint_t k) {
  const auto a_strides = a.raw_strides();
  const auto b_strides = b.raw_strides();
  const auto c_strides = c_output.raw_strides();
  const auto* a_data = reinterpret_cast<const typename Atom::TA*>(a.raw_data());
  const auto* b_data = reinterpret_cast<const typename Atom::TB*>(b.raw_data());
  auto* c_data = reinterpret_cast<typename Atom::TAcc*>(c_output.raw_data());
  if (n == 1 && m >= 4) {
    small_vector_tall_matmul(
        a_data, a_strides[0], b_data, b_strides[0],
        c_data, c_strides[0], m, k);
  } else {
    small_vector_dense_matmul(
        a_data, a_strides[0], b_data, b_strides[0],
        c_data, c_strides[0], m, n, k);
  }
}

/// Split-K accumulating entry for the native direct accumulator form.
template <::vecops::matmul::Atom Atom, typename A, typename B,
          typename CInput, typename COutput>
VECOPS_ALWAYS_INLINE void run_small_vector_accumulating(
    const A& a, const B& b, const CInput& c_input, COutput& c_output,
    nint_t m, nint_t n, nint_t k) {
  static_assert(small_vector_accumulate_candidate_v<
      Atom, A, B, CInput, COutput>);
  const auto a_strides = a.raw_strides();
  const auto b_strides = b.raw_strides();
  const auto ci_strides = c_input.raw_strides();
  const auto co_strides = c_output.raw_strides();
  const auto* a_data = reinterpret_cast<const typename Atom::TA*>(a.raw_data());
  const auto* b_data = reinterpret_cast<const typename Atom::TB*>(b.raw_data());
  const auto* prior = reinterpret_cast<const typename Atom::TAcc*>(
      c_input.raw_data());
  auto* output = reinterpret_cast<typename Atom::TAcc*>(c_output.raw_data());
  if (n == 1 && m >= 4) {
    small_vector_tall_matmul<true>(
        a_data, a_strides[0], b_data, b_strides[0],
        output, co_strides[0], m, k, prior, ci_strides[0]);
  } else {
    small_vector_dense_matmul<true>(
        a_data, a_strides[0], b_data, b_strides[0],
        output, co_strides[0], m, n, k, prior, ci_strides[0]);
  }
}

/// Run the FusedSmallBF16 owner: compute into the 16-element values
/// buffer, then add the (elementwise-transformed) C input and store
/// through the output transform — all through the vec layer so the
/// transforms stay fused instead of forcing a zero C input.
template <::vecops::matmul::Atom Atom, typename A, typename B,
          typename CInput, typename COutput>
VECOPS_ALWAYS_INLINE void run_fused_small_bf16(
    const A& a, const B& b,
    const CInput& c_input, COutput& c_output,
    nint_t m, nint_t n, nint_t k) {
  static_assert(fused_small_bf16_candidate_v<
      Atom, A, B, CInput, COutput>);
  alignas(64) float32_t values[16];
  constexpr bool PackedA =
      is_packed_access_v<Atom, ::vecops::matmul::Operand::A, A>;
  const auto a_strides = a.raw_strides();
  const auto b_strides = b.raw_strides();
  small_bf16_fused_compute<PackedA>(
      reinterpret_cast<const bfloat16_t*>(a.raw_data()),
      PackedA ? nint_t{0} : a_strides[0],
      reinterpret_cast<const bfloat16_t*>(b.raw_data()), b_strides[0],
      values, m, n, k);
  using Tag = vec::ScalableTag<float32_t, 0>;
  for (nint_t i = 0; i < m; ++i) {
    auto value = vec::load(
        Tag{}, values + i * n, vec::opt::first(n), vec::opt::zero);
    if constexpr (!IsZeroTransform<TransformOf<CInput>>::value) {
      const auto prior = c_input.load(
          Tag{}, tensor::coord(i, 0), tensor::axis<1>,
          vec::opt::first(n), vec::opt::zero);
      value = vec::add(Tag{}, value, prior);
    }
    c_output.store(
        Tag{}, tensor::coord(i, 0), tensor::axis<1>, value,
        vec::opt::first(n));
  }
}

/** One TILECFG image spans a traversal; decode may shorten its active rows. */
struct Configuration : execution::details::x86::TileConfiguration {
  Configuration() {
    for (int i = 0; i < 8; ++i) {
      column_bytes[i] = 64;
      rows[i] = 16;
    }
  }

  void set_horizontal_rows(nint_t active_m, nint_t output_tiles) {
    VECOPS_ASSERT(active_m > 0 && active_m <= 16,
                  "AMX horizontal row count must be in [1, 16]");
    VECOPS_ASSERT(output_tiles >= 1 && output_tiles <= 3,
                  "AMX horizontal output tile count must be in [1, 3]");
    const auto tile_rows = static_cast<std::uint8_t>(active_m);
    // Horizontal cases map C0..C(N-1),A0 to the first N+1 tile registers.
    // B0..B(N-1) remain sixteen rows.
    for (nint_t i = 0; i <= output_tiles; ++i) rows[i] = tile_rows;
  }

};

/// Catalog input for Tile2D. Families are scored by tile count: area
/// (A*B) dominates, register pressure (A+B) breaks ties; a family that
/// does not fit the eight tile registers (A*B + A + B > 8) scores -1
/// and is excluded. exact_grid_mode::unmasked declares that AMX tail
/// handling never needs per-lane masking.
enum class KernelAspect { balanced, tall, wide };

template <KernelAspect Aspect = KernelAspect::balanced>
struct KernelProvider {
  static constexpr tile::Tile2DExactGridMode exact_grid_mode =
      tile::Tile2DExactGridMode::unmasked;

  template <int A, int B, tile::Tile2DMaskMode, tile::Tile2DMaskMode>
  static consteval int power() {
    if constexpr (A * B + A + B <= 8) {
      constexpr int AspectBonus = [] {
        if constexpr (Aspect == KernelAspect::tall && A == 3 && B == 1)
          return 128;
        if constexpr (Aspect == KernelAspect::wide && A == 1 && B == 3)
          return 128;
        return 0;
      }();
      return 100 * A * B + 4 * (A + B) + AspectBonus;
    } else {
      return -1;
    }
  }
};

/// Compile-time kernel plan: KGuaranteed (K is a whole multiple of the
/// Atom K step, so every tile load can skip tail clamping) and StreamB
/// (packed B is large enough that non-temporal loads pay off).
template <bool KGuaranteed, bool StreamB>
struct KernelPlan : std::bool_constant<KGuaranteed> {
  static constexpr bool stream_b = StreamB;
};

template <KernelAspect Aspect = KernelAspect::balanced>
using Catalog = tile::Tile2DGeneratedCatalog<
    KernelProvider<Aspect>, tile::Tile2DSearchSpace<3, 3, 4>>;

/** Select a register-block aspect only when Meta bounds prove a persistent
 * 4:1 spatial skew and enough K work to amortize the lower output-tile count.
 * Unbounded dynamic and shallow-K shapes keep the balanced 2x2 family; no
 * runtime catalog fan-out is introduced. */
template <meta::ValueType M, meta::ValueType N, meta::ValueType K>
inline constexpr KernelAspect kernel_aspect_v = [] {
  using MV = std::remove_cvref_t<M>;
  using NV = std::remove_cvref_t<N>;
  using KV = std::remove_cvref_t<K>;
  if constexpr (!meta::has_lower_bound_v<KV> ||
                meta::lower_bound_v<KV> < 256)
    return KernelAspect::balanced;
  if constexpr (meta::has_lower_bound_v<MV> &&
                meta::has_upper_bound_v<NV>) {
    constexpr nint_t MinM = meta::lower_bound_v<MV>;
    constexpr nint_t MaxN = meta::upper_bound_v<NV>;
    if constexpr (MinM > 0 && MaxN > 0 && MinM / 4 >= MaxN)
      return KernelAspect::tall;
  }
  if constexpr (meta::has_lower_bound_v<NV> &&
                meta::has_upper_bound_v<MV>) {
    constexpr nint_t MinN = meta::lower_bound_v<NV>;
    constexpr nint_t MaxM = meta::upper_bound_v<MV>;
    if constexpr (MinN > 0 && MaxM > 0 && MinN / 4 >= MaxM)
      return KernelAspect::wide;
  }
  return KernelAspect::balanced;
}();

/// Vector power for a "16 lanes of T" register relative to the native
/// vector width: negative when 16*T is narrower than native (fractional
/// power of two), positive when wider. Used to pick the ScalableTag whose
/// lane count matches a 16-row panel pack.
template <typename T>
consteval int sixteen_lane_power() {
  nint_t bytes = 16 * static_cast<nint_t>(sizeof(T));
  nint_t native = VEC_WIDTH / 8;
  int power = 0;
  while (bytes < native) {
    bytes *= 2;
    --power;
  }
  while (bytes > native) {
    native *= 2;
    ++power;
  }
  return power;
}

/// Resolve the source pointer for one A tile: packed panels hand back a
/// pointer straight into the panel (block indices divide cleanly by the
/// panel/block sizes), everything else packs through DataAccess into the
/// caller's staging buffer.
template <::vecops::matmul::Atom Atom, bool SpatialGuaranteed, bool KGuaranteed,
          typename Source>
VECOPS_ALWAYS_INLINE const typename Atom::TA* prepare_a(
    const Source& source, nint_t m, nint_t k, nint_t logical_m,
    nint_t logical_k, typename Atom::TA* buffer) {
  using T = typename Atom::TA;
  constexpr nint_t KR = decltype(Atom::K_R)::value;
  if constexpr (is_packed_access_v<Atom, ::vecops::matmul::Operand::A, Source>) {
    static_assert(generic::RawDirectAccess<Source>,
                  "packed AMX A must be direct and untransformed");
    const auto& layout = source.spec().input_layout();
    // Packed-A rank four: [16-row panel][K block][row][K pack]; advance
    // only the first two coordinates, the tileload consumes the rest.
    const nint_t offset = tensor::offset_at(layout, m / 16, k / KR, 0, 0);
    return reinterpret_cast<const T*>(source.raw_data()) + offset;
  } else {
    static_assert(Source::Rank == 2, "unpacked AMX A must be rank two");
    using Tag = vec::ScalableTag<T, 0>;
    matmul_pack_details::amx::pack_a_tile<
        KR, Tag, SpatialGuaranteed, KGuaranteed>(
        source, buffer, m, k, logical_m, logical_k);
    return buffer;
  }
}

/// Resolve the source pointer for one B tile; same three-tier structure
/// as prepare_a (packed panel / direct row-major pack / generic pack),
/// with a fast full-panel pack whenever both extents are guaranteed.
template <::vecops::matmul::Atom Atom, bool SpatialGuaranteed, bool KGuaranteed,
          typename Source>
VECOPS_ALWAYS_INLINE const typename Atom::TB* prepare_b(
    const Source& source, nint_t n, nint_t k, nint_t logical_n,
    nint_t logical_k, typename Atom::TB* buffer) {
  using T = typename Atom::TB;
  using Packing = ::vecops::matmul::packing_t<Atom, ::vecops::matmul::Operand::B>;
  constexpr nint_t KR = decltype(Atom::K_R)::value;
  constexpr nint_t KP = Packing::KPack;
  if constexpr (is_packed_access_v<Atom, ::vecops::matmul::Operand::B, Source>) {
    static_assert(generic::RawDirectAccess<Source>,
                  "packed AMX B must be direct and untransformed");
    const auto& layout = source.spec().input_layout();
    const nint_t offset = tensor::offset_at(
        layout, n / 16, k / KR, 0, 0, 0);
    return reinterpret_cast<const T*>(source.raw_data()) + offset;
  } else {
    static_assert(Source::Rank == 2, "unpacked AMX B must be rank two");
    if constexpr (direct_row_major_input_v<Source>) {
      const auto strides = source.raw_strides();
      // The tile traversal only calls prepare_b for n < logical_n and the K
      // loop only calls it for k < logical_k, so these tails need an upper
      // bound only. Avoid an extra lower-bound comparison in the hot loop.
      const nint_t active_n = SpatialGuaranteed
          ? nint_t{16}
          : vecops::min(logical_n - n, nint_t{16});
      const nint_t active_k = KGuaranteed
          ? KR
          : vecops::min(logical_k - k, KR);
      if constexpr (SpatialGuaranteed && KGuaranteed) {
        const auto* pointer = reinterpret_cast<const T*>(source.raw_data()) +
            n * strides[0] + k;
        matmul_pack_details::amx::pack_b_full_panel_direct<KP>(
            pointer, strides[0], buffer);
        return buffer;
      } else if constexpr (KGuaranteed) {
        if (active_n == 16) {
          const auto* pointer = reinterpret_cast<const T*>(source.raw_data()) +
              n * strides[0] + k;
          matmul_pack_details::amx::pack_b_full_panel_direct<KP>(
              pointer, strides[0], buffer);
          return buffer;
        }
      }
      // load_b_tile has already rejected an inactive N tile, and the enclosing
      // K loop only calls prepare_b for k < logical_k.
      const auto* pointer = reinterpret_cast<const T*>(source.raw_data()) +
          n * strides[0] + k;
      matmul_pack_details::amx::pack_b_partial_panel_direct<KP>(
          pointer, strides[0], buffer, active_n, active_k);
      return buffer;
    }
    using Tag = vec::ScalableTag<T, sixteen_lane_power<T>()>;
    matmul_pack_details::amx::pack_b_tile<
        KP, KR, Tag, SpatialGuaranteed, KGuaranteed>(
            source, buffer, n, k, logical_n, logical_k);
    return buffer;
  }
}

/// Per-sub-tile active extent for a Case with multiple 16-row blocks:
/// Offset is the sub-tile's start within the Case (block index * 16), so
/// the clamp of (active - Offset) into [0, 16] yields this block's own
/// share of the Case-wide active row/column count. Full collapses to
/// the constant 16; NonEmpty strengthens the lower bound to 1 for blocks
/// an exact traversal proves exist. ReferenceClamp only changes the
/// clamp helper's ABI (see microkernel).
template <bool Full, bool NonEmpty, nint_t Offset,
          bool ReferenceClamp = false>
VECOPS_ALWAYS_INLINE auto tile_active_extent(nint_t active) {
  if constexpr (Full) {
    return meta::cint<16>;
  } else {
    if constexpr (ReferenceClamp) {
      const nint_t value = vecops::clamp_reference(
          active - Offset, nint_t{0}, nint_t{16});
      if constexpr (NonEmpty) return meta::dyn<1, 1, 16>(value);
      else return meta::dyn<1, 0, 16>(value);
    } else {
      const nint_t value = vecops::clamp(
          active - Offset, nint_t{0}, nint_t{16});
      if constexpr (NonEmpty) return meta::dyn<1, 1, 16>(value);
      else return meta::dyn<1, 0, 16>(value);
    }
  }
}

/// Extent type is statically the full 16.
template <typename Extent>
inline constexpr bool full_tile_extent_v = [] {
  using E = std::remove_cvref_t<Extent>;
  if constexpr (E::is_const) return E::value == 16;
  else return false;
}();

/// Extent type is statically nonzero, so a zero check can be skipped.
template <typename Extent>
inline constexpr bool nonempty_tile_extent_v =
    meta::has_lower_bound_v<std::remove_cvref_t<Extent>> &&
    meta::lower_bound_v<std::remove_cvref_t<Extent>> > 0;

/// Initialize one C accumulator tile from the C input: zero tiles for a
/// zero transform, a direct tileload when the memory layout allows one,
/// otherwise a staged pack through a 16x16 scratch row buffer.
template <int Tile, meta::ValueType ActiveM, meta::ValueType ActiveN,
          typename CInput>
VECOPS_ALWAYS_INLINE void initialize_c_tile(
    const CInput& input, nint_t m, nint_t n,
    ActiveM active_m, ActiveN active_n,
    typename CInput::ComputeType* buffer) {
  using T = typename CInput::ComputeType;
  using Transform = typename CInput::Transform;
  constexpr bool FullM = full_tile_extent_v<ActiveM>;
  constexpr bool FullN = full_tile_extent_v<ActiveN>;
  constexpr bool NonEmpty =
      nonempty_tile_extent_v<ActiveM> && nonempty_tile_extent_v<ActiveN>;
  const nint_t active_m_value = static_cast<nint_t>(active_m);
  const nint_t active_n_value = static_cast<nint_t>(active_n);
  // A resident family may be wider than a boundary region.  Such completely
  // inactive accumulator tiles still participate in the AMX dot instruction,
  // so initialize them to zero without asking DataAccess for an out-of-range
  // base coordinate.
  // Keep the existing zero-prologue instruction schedule unchanged. A
  // statically nonempty nonzero prologue does not need the runtime check.
  if constexpr (IsZeroTransform<Transform>::value || !NonEmpty) {
    if (active_m_value == 0 || active_n_value == 0) {
      amx_intrinsics::zero<Tile>();
      return;
    }
  }
  if constexpr (IsZeroTransform<Transform>::value) {
    amx_intrinsics::zero<Tile>();
  } else {
    if constexpr (direct_row_major_input_v<CInput>) {
      if constexpr (FullM && FullN) {
        const auto strides = input.raw_strides();
        const auto* pointer = reinterpret_cast<const T*>(input.raw_data()) +
            m * strides[0] + n;
        amx_intrinsics::load<Tile>(
            pointer, strides[0] * static_cast<nint_t>(sizeof(T)));
        return;
      } else if ((FullM || active_m_value == 16) &&
                 (FullN || active_n_value == 16)) {
        const auto strides = input.raw_strides();
        const auto* pointer = reinterpret_cast<const T*>(input.raw_data()) +
            m * strides[0] + n;
        amx_intrinsics::load<Tile>(
            pointer, strides[0] * static_cast<nint_t>(sizeof(T)));
        return;
      }
    }
    using Tag = vec::ScalableTag<T, 0>;
    auto load_row = [&](nint_t row) VECOPS_INLINE_LAMBDA {
      if constexpr (FullN) {
        return input.load(
            Tag{}, tensor::coord(m + row, n), tensor::axis<1>);
      } else {
        return input.load(
            Tag{}, tensor::coord(m + row, n), tensor::axis<1>,
            vec::opt::first(active_n_value), vec::opt::zero);
      }
    };
    if constexpr (FullM) {
      VECOPS_UNROLL
      for (nint_t row = 0; row < 16; ++row) {
        vec::store(Tag{}, buffer + row * 16, load_row(row));
      }
    } else {
      for (nint_t row = 0; row < active_m_value; ++row) {
        vec::store(Tag{}, buffer + row * 16, load_row(row));
      }
      const auto zero = vec::zeros(Tag{});
      for (nint_t row = active_m_value; row < 16; ++row) {
        vec::store(Tag{}, buffer + row * 16, zero);
      }
    }
    amx_intrinsics::load<Tile>(buffer, 16 * sizeof(T));
  }
}

/**
 * Initialize two horizontally-adjacent C tiles through one wide DataAccess
 * load.  The wider root tag makes a narrow C memory type occupy a complete
 * physical vector word; its active predicate covers both tiles and is split
 * only after the prologue has run.
 */
template <int LeftTile, int RightTile, meta::ValueType ActiveM, meta::ValueType LeftActiveN,
          meta::ValueType RightActiveN, typename CInput>
VECOPS_ALWAYS_INLINE void initialize_c_tile_row_pair(const CInput& input, nint_t m, nint_t n, ActiveM active_m,
                                                     LeftActiveN left_active_n, RightActiveN right_active_n,
                                                     typename CInput::ComputeType* left_buffer,
                                                     typename CInput::ComputeType* right_buffer) {
  using T = typename CInput::ComputeType;
  using Transform = typename CInput::Transform;
  using Tag = vec::ScalableTag<T, 0>;
  using WideTag = vec::Twice<Tag>;
  static_assert(tensor::preferred_memory_access_power_v<CInput> > 0);
  static_assert(vec::size(Tag{}) == 16);
  const nint_t active_m_value = static_cast<nint_t>(active_m);
  const nint_t left_n = static_cast<nint_t>(left_active_n);
  const nint_t right_n = static_cast<nint_t>(right_active_n);
  const nint_t active_n = left_n + right_n;
  if (active_m_value == 0 || active_n == 0) {
    amx_intrinsics::zero<LeftTile>();
    amx_intrinsics::zero<RightTile>();
    return;
  }
  if constexpr (IsZeroTransform<Transform>::value) {
    amx_intrinsics::zero<LeftTile>();
    amx_intrinsics::zero<RightTile>();
    return;
  }

  auto load_row = [&](nint_t row) VECOPS_INLINE_LAMBDA {
    if constexpr (full_tile_extent_v<LeftActiveN> && full_tile_extent_v<RightActiveN>) {
      return input.load(WideTag{}, tensor::coord(m + row, n), tensor::axis<1>);
    } else {
      return input.load(WideTag{}, tensor::coord(m + row, n), tensor::axis<1>, vec::opt::first(active_n),
                        vec::opt::zero);
    }
  };
  for (nint_t row = 0; row < active_m_value; ++row) {
    const auto value = load_row(row);
    vec::store(Tag{}, left_buffer + row * 16, vec::lower(WideTag{}, value));
    vec::store(Tag{}, right_buffer + row * 16, vec::upper(WideTag{}, value));
  }
  const auto zero = vec::zeros(Tag{});
  for (nint_t row = active_m_value; row < 16; ++row) {
    vec::store(Tag{}, left_buffer + row * 16, zero);
    vec::store(Tag{}, right_buffer + row * 16, zero);
  }
  amx_intrinsics::load<LeftTile>(left_buffer, 16 * sizeof(T));
  amx_intrinsics::load<RightTile>(right_buffer, 16 * sizeof(T));
}

/// Load one A operand tile: direct tileload when the block provably fits
/// in memory, inactive-tile zeroing when DirectInactiveZero allows it,
/// otherwise a staged tail pack (which itself requires k < logical_k,
/// hence the fallback ordering).
template <::vecops::matmul::Atom Atom, int Tile, bool SpatialGuaranteed, bool NonEmpty,
          bool DirectInactiveZero, bool KGuaranteed, typename Source>
VECOPS_ALWAYS_INLINE void load_a_tile(
    const Source& source, nint_t m, nint_t k,
    nint_t logical_m, nint_t logical_k,
    typename Atom::TA* buffer) {
  using T = typename Atom::TA;
  constexpr nint_t KR = decltype(Atom::K_R)::value;
  if constexpr (!NonEmpty && DirectInactiveZero) {
    if (m >= logical_m) {
      amx_intrinsics::zero<Tile>();
      return;
    }
  }
  if constexpr (direct_row_major_input_v<Source>) {
    if constexpr (SpatialGuaranteed && KGuaranteed) {
      const auto strides = source.raw_strides();
      const auto* pointer = reinterpret_cast<const T*>(source.raw_data()) +
          m * strides[0] + k;
      amx_intrinsics::load<Tile>(
          pointer, strides[0] * static_cast<nint_t>(sizeof(T)));
      return;
    } else if ((SpatialGuaranteed || m + 16 <= logical_m) &&
               (KGuaranteed || k + KR <= logical_k)) {
      const auto strides = source.raw_strides();
      const auto* pointer = reinterpret_cast<const T*>(source.raw_data()) +
          m * strides[0] + k;
      amx_intrinsics::load<Tile>(
          pointer, strides[0] * static_cast<nint_t>(sizeof(T)));
      return;
    }
    if constexpr (!DirectInactiveZero) {
      // Keep the raw-A / packed-B tail pack in a separate AVX-512 function.
      // GCC 13 otherwise misallocates tile state when this vector pack is
      // inlined into exact 1xN AMX families.
      const auto strides = source.raw_strides();
      const auto* pointer = reinterpret_cast<const T*>(source.raw_data()) +
          m * strides[0] + k;
      // load_a_tile has already rejected an inactive M tile, and the
      // enclosing K loop only calls it for k < logical_k.
      const nint_t active_m = vecops::min(
          logical_m - m, nint_t{16});
      const nint_t active_k = KGuaranteed
          ? KR
          : vecops::min(logical_k - k, KR);
      matmul_pack_details::amx::pack_a_partial_tile_direct(
          pointer, strides[0], buffer, active_m, active_k);
      amx_intrinsics::load<Tile>(buffer, 64);
      return;
    }
  }
  amx_intrinsics::load<Tile>(
      prepare_a<Atom, SpatialGuaranteed, KGuaranteed>(
          source, m, k, logical_m, logical_k, buffer),
      64);
}

/// Load one B operand tile: reject inactive N tiles with a zeroed
/// register, then load via prepare_b (non-temporal when StreamB and B
/// is already packed — the stream hint is only safe for panel memory
/// that will not be re-read soon).
template <::vecops::matmul::Atom Atom, int Tile, bool SpatialGuaranteed, bool NonEmpty,
          bool KGuaranteed, bool StreamB, typename Source>
VECOPS_ALWAYS_INLINE void load_b_tile(
    const Source& source, nint_t n, nint_t k,
    nint_t logical_n, nint_t logical_k,
    typename Atom::TB* buffer) {
  if constexpr (!NonEmpty) {
    if (n >= logical_n) {
      amx_intrinsics::zero<Tile>();
      return;
    }
  }
  const auto* pointer = prepare_b<Atom, SpatialGuaranteed, KGuaranteed>(
      source, n, k, logical_n, logical_k, buffer);
  if constexpr (is_packed_access_v<Atom, ::vecops::matmul::Operand::B, Source>) {
    if constexpr (StreamB) amx_intrinsics::stream_load<Tile>(pointer, 64);
    else amx_intrinsics::load<Tile>(pointer, 64);
  } else {
    amx_intrinsics::load<Tile>(pointer, 64);
  }
}

// Tile-register mapping for one dot step (see the file header for the
// full map): output (i, j) accumulates in register I, reading A tile
// Outputs + I/NN (block row) and B tile Outputs + NM + I%NN (block
// column). Outputs = NM*NN keeps the C block set contiguous in the low
// registers, A and B stacked after it.
template <::vecops::matmul::Atom Atom, int NM, int NN, std::size_t... I>
VECOPS_ALWAYS_INLINE void compute_tiles_impl(std::index_sequence<I...>) {
  constexpr int Outputs = NM * NN;
  (amx_intrinsics::dot<
       typename Atom::TAcc, typename Atom::TA, typename Atom::TB,
       static_cast<int>(I), Outputs + static_cast<int>(I / NN),
       Outputs + NM + static_cast<int>(I % NN)>(), ...);
}

/// Issue all NM*NN dots of one K step at once (the default schedule).
template <::vecops::matmul::Atom Atom, int NM, int NN>
VECOPS_ALWAYS_INLINE void compute_tiles() {
  compute_tiles_impl<Atom, NM, NN>(std::make_index_sequence<NM * NN>{});
}

/// Issue the single dot for output block (Row, Column) — used by the
/// interleaved load/compute schedule in multiply_k_tile.
template <::vecops::matmul::Atom Atom, int NM, int NN, int Row, int Column>
VECOPS_ALWAYS_INLINE void compute_tile() {
  constexpr int Outputs = NM * NN;
  amx_intrinsics::dot<
      typename Atom::TAcc, typename Atom::TA, typename Atom::TB,
      Row * NN + Column, Outputs + Row, Outputs + NM + Column>();
}

/// Load one Case's A/B tile sets and run the K step. Staging buffers are
/// addressed in elements: each tile image is 1024 bytes (16 rows x 64 B).
template <::vecops::matmul::Atom Atom, typename Case, bool KGuaranteed,
          bool StreamB,
          bool DirectInactiveZeroA, typename A, typename B>
VECOPS_ALWAYS_INLINE void multiply_k_tile(
    const A& a, const B& b,
    nint_t logical_m, nint_t logical_n, nint_t logical_k,
    nint_t m, nint_t n, nint_t k,
    typename Atom::TA* a_buffers, typename Atom::TB* b_buffers) {
  constexpr int NM = Case::a;
  constexpr int NN = Case::b;
  constexpr int Outputs = NM * NN;
  constexpr bool FullM =
      Case::m_mask == tile::Tile2DMaskMode::unmasked;
  constexpr bool FullN =
      Case::n_mask == tile::Tile2DMaskMode::unmasked;
  constexpr bool NonEmptyM = FullM || Case::exact_blocks;
  constexpr bool NonEmptyN = FullN || Case::exact_blocks;
  auto load_a = [&]<std::size_t I>() VECOPS_INLINE_LAMBDA {
    load_a_tile<
        Atom, Outputs + static_cast<int>(I),
        FullM || (Case::exact_blocks && I + 1 < NM), NonEmptyM,
        DirectInactiveZeroA, KGuaranteed>(
        a, m + static_cast<nint_t>(I) * 16, k,
        logical_m, logical_k,
        a_buffers + I * 1024 / sizeof(typename Atom::TA));
  };
  auto load_b = [&]<std::size_t I>() VECOPS_INLINE_LAMBDA {
    load_b_tile<
        Atom, Outputs + NM + static_cast<int>(I),
        FullN || (Case::exact_blocks && I + 1 < NN), NonEmptyN,
        KGuaranteed, StreamB>(
        b, n + static_cast<nint_t>(I) * 16, k,
        logical_n, logical_k,
        b_buffers + I * 1024 / sizeof(typename Atom::TB));
  };
  if constexpr (
      NN == 1 &&
      is_packed_access_v<Atom, ::vecops::matmul::Operand::A, A> &&
      is_packed_access_v<Atom, ::vecops::matmul::Operand::B, B>) {
    // The vertical packed strip has one B tile resident for every output.
    // Interleave each new A load with its dependent dot; the general raw/tail
    // paths retain the compiler's original all-loads-first schedule.
    load_b.template operator()<0>();
    [&]<std::size_t... I>(std::index_sequence<I...>) VECOPS_INLINE_LAMBDA {
      ((load_a.template operator()<I>(),
        compute_tile<Atom, NM, NN, static_cast<int>(I), 0>()), ...);
    }(std::make_index_sequence<NM>{});
  } else {
    [&]<std::size_t... I>(std::index_sequence<I...>) VECOPS_INLINE_LAMBDA {
      (load_a.template operator()<I>(), ...);
    }(std::make_index_sequence<NM>{});
    [&]<std::size_t... I>(std::index_sequence<I...>) VECOPS_INLINE_LAMBDA {
      (load_b.template operator()<I>(), ...);
    }(std::make_index_sequence<NN>{});
    compute_tiles<Atom, NM, NN>();
  }
}

/// Linear-cursor K loop for caller/online prepared A and B.  Packed-format
/// recognition proves that one K block is exactly one 16-row AMX tile image,
/// so advancing both cursors by Panel*KR elements is equivalent to repeatedly
/// evaluating offset_at(..., k / KR, ...).  Keeping the two block bases and
/// panel strides live removes the divide/shift and layout-address rebuild from
/// every K step while preserving the same register family and dot schedule.
template <::vecops::matmul::Atom Atom, typename Case, bool StreamB,
          typename A, typename B>
VECOPS_ALWAYS_INLINE void multiply_packed_k_loop(
    const A& a, const B& b, nint_t logical_k, nint_t m, nint_t n) {
  static_assert(is_packed_access_v<
      Atom, ::vecops::matmul::Operand::A, A>);
  static_assert(is_packed_access_v<
      Atom, ::vecops::matmul::Operand::B, B>);
  constexpr int NM = Case::a;
  constexpr int NN = Case::b;
  constexpr int Outputs = NM * NN;
  constexpr nint_t Panel = 16;
  constexpr nint_t KR = decltype(Atom::K_R)::value;
  constexpr nint_t KBlockElements = Panel * KR;
  const auto a_strides = a.raw_strides();
  const auto b_strides = b.raw_strides();
  const nint_t a_panel_stride = a_strides[0];
  const nint_t b_panel_stride = b_strides[0];
  const auto* a_cursor = reinterpret_cast<const typename Atom::TA*>(
      a.raw_data()) + (m / Panel) * a_panel_stride;
  const auto* b_cursor = reinterpret_cast<const typename Atom::TB*>(
      b.raw_data()) + (n / Panel) * b_panel_stride;

  auto run_step = [&]<nint_t Offset>() VECOPS_INLINE_LAMBDA {
    auto load_a = [&]<std::size_t I>() VECOPS_INLINE_LAMBDA {
      const auto* pointer =
          a_cursor + Offset + static_cast<nint_t>(I) * a_panel_stride;
      amx_intrinsics::load<Outputs + static_cast<int>(I)>(pointer, 64);
    };
    auto load_b = [&]<std::size_t I>() VECOPS_INLINE_LAMBDA {
      const auto* pointer =
          b_cursor + Offset + static_cast<nint_t>(I) * b_panel_stride;
      if constexpr (StreamB) {
        amx_intrinsics::stream_load<
            Outputs + NM + static_cast<int>(I)>(pointer, 64);
      } else {
        amx_intrinsics::load<
            Outputs + NM + static_cast<int>(I)>(pointer, 64);
      }
    };
    if constexpr (NN == 1) {
      load_b.template operator()<0>();
      [&]<std::size_t... I>(std::index_sequence<I...>)
          VECOPS_INLINE_LAMBDA {
        ((load_a.template operator()<I>(),
          compute_tile<Atom, NM, NN, static_cast<int>(I), 0>()), ...);
      }(std::make_index_sequence<NM>{});
    } else {
      [&]<std::size_t... I>(std::index_sequence<I...>)
          VECOPS_INLINE_LAMBDA {
        (load_a.template operator()<I>(), ...);
      }(std::make_index_sequence<NM>{});
      [&]<std::size_t... I>(std::index_sequence<I...>)
          VECOPS_INLINE_LAMBDA {
        (load_b.template operator()<I>(), ...);
      }(std::make_index_sequence<NN>{});
      compute_tiles<Atom, NM, NN>();
    }
  };

  for (nint_t k = 0; k < logical_k; k += KR) {
    run_step.template operator()<0>();
    a_cursor += KBlockElements;
    b_cursor += KBlockElements;
  }
}

/// Store one 8-column half of a staged C tile into a column-contiguous
/// (transposed) output: transpose the half tile in registers with the
/// generic 8x8 butterfly network, then store each column as one vector
/// along axis<0>. With M <= 8 only the top half exists and the column
/// vector is zero-padded; wider tiles concatenate the top and bottom
/// column halves into a full-width store.
template <typename T>

VECOPS_ALWAYS_INLINE auto read_c_tile_column_block(T* buffer, nint_t active_m, nint_t column_offset) {
  static_assert(sizeof(T) == 4, "AMX accumulators must have 32-bit lanes");
  using HalfTag = vec::FixedTag<T, 8>;
  using FullTag = vec::ScalableTag<T, 0>;
  constexpr std::size_t Half = 8;
  std::array<vec::Vec<HalfTag>, Half> top_rows{};
  VECOPS_UNROLL
  for (std::size_t row = 0; row < Half; ++row) {
    top_rows[row] = vec::load(
        HalfTag{}, buffer + row * 16 + column_offset);
  }
  const auto top_columns = kernel::transpose2d_details::generic::transpose_square<HalfTag>(top_rows);

  std::array<vec::Vec<HalfTag>, Half> bottom_rows{};

  if (active_m > 8) {
    VECOPS_UNROLL
    for (std::size_t row = 0; row < Half; ++row) {
      bottom_rows[row] = vec::load(HalfTag{}, buffer + (row + Half) * 16 + column_offset);
    }

  } else {
    VECOPS_UNROLL
    for (std::size_t row = 0; row < Half; ++row)
      bottom_rows[row] = vec::zeros(HalfTag{});
  }
  const auto bottom_columns = kernel::transpose2d_details::generic::transpose_square<HalfTag>(bottom_rows);

  std::array<vec::Vec<FullTag>, Half> columns{};
  VECOPS_UNROLL
  for (std::size_t column = 0; column < Half; ++column) {
    columns[column] = vec::concat(FullTag{}, top_columns[column], bottom_columns[column]);
  }
  return columns;
}

template <typename COutput>
VECOPS_NOINLINE void store_c_tile_column_block(COutput& output, nint_t m, nint_t n, nint_t active_m, nint_t active_n,
                                               typename COutput::ComputeType* buffer, nint_t column_offset) {
  using T = typename COutput::ComputeType;
  using FullTag = vec::ScalableTag<T, 0>;
  const auto values = read_c_tile_column_block(buffer, active_m, column_offset);
  const nint_t columns = vecops::min(active_n - column_offset, nint_t{8});
  for (nint_t column = 0; column < columns; ++column) {
    output.store(FullTag{}, tensor::coord(m, n + column_offset + column), tensor::axis<0>,
                 values[static_cast<std::size_t>(column)], vec::opt::first(active_m));
  }
}

/// Store one C accumulator tile: direct tilestore when the layout allows
/// it, otherwise a staged store through the 16x16 scratch row buffer
/// with masked tail rows/columns (column-contiguous transposed outputs
/// store through store_c_tile_column_block instead of the row loop).
template <int Tile, meta::ValueType ActiveM, meta::ValueType ActiveN,
          typename COutput>
VECOPS_ALWAYS_INLINE void store_c_tile(
    COutput& output, nint_t m, nint_t n,
    ActiveM active_m, ActiveN active_n,
    typename COutput::ComputeType* buffer) {
  using T = typename COutput::ComputeType;
  constexpr bool FullM = full_tile_extent_v<ActiveM>;
  constexpr bool FullN = full_tile_extent_v<ActiveN>;
  constexpr bool NonEmpty =
      nonempty_tile_extent_v<ActiveM> && nonempty_tile_extent_v<ActiveN>;
  const nint_t active_m_value = static_cast<nint_t>(active_m);
  const nint_t active_n_value = static_cast<nint_t>(active_n);
  // Do not form a DataAccess base coordinate for an empty resident sub-tile.
  if constexpr (!NonEmpty) {
    if (active_m_value == 0 || active_n_value == 0) return;
  }
  if constexpr (direct_row_major_output_v<COutput>) {
    if constexpr (FullM && FullN) {
      const auto strides = output.raw_strides();
      auto* pointer = reinterpret_cast<T*>(output.raw_data()) +
          m * strides[0] + n;
      amx_intrinsics::store<Tile>(
          pointer, strides[0] * static_cast<nint_t>(sizeof(T)));
      return;
    } else if ((FullM || active_m_value == 16) &&
               (FullN || active_n_value == 16)) {
      const auto strides = output.raw_strides();
      auto* pointer = reinterpret_cast<T*>(output.raw_data()) +
          m * strides[0] + n;
      amx_intrinsics::store<Tile>(
          pointer, strides[0] * static_cast<nint_t>(sizeof(T)));
      return;
    }
  }
  amx_intrinsics::store<Tile>(buffer, 16 * sizeof(T));
  if constexpr (column_contiguous_output_v<COutput>) {
    store_c_tile_column_block(
        output, m, n, active_m_value, active_n_value, buffer, 0);
    if (active_n_value > 8) {
      store_c_tile_column_block(
          output, m, n, active_m_value, active_n_value, buffer, 8);
    }
    return;
  }
  using Tag = vec::ScalableTag<T, 0>;
  auto store_row = [&](nint_t row) VECOPS_INLINE_LAMBDA {
    const auto value = vec::load(Tag{}, buffer + row * 16);
    if constexpr (FullN) {
      output.store(
          Tag{}, tensor::coord(m + row, n), tensor::axis<1>, value);
    } else {
      output.store(
          Tag{}, tensor::coord(m + row, n), tensor::axis<1>, value,
          vec::opt::first(active_n_value));
    }
  };
  if constexpr (FullM) {
    for (nint_t row = 0; row < 16; ++row) store_row(row);
  } else {
    for (nint_t row = 0; row < active_m_value; ++row) store_row(row);
  }
}

/**
 * Store two horizontally-adjacent accumulator tiles through one wide
 * DataAccess epilogue invocation.  The combined first-active mask remains in
 * the wide domain, so a boundary pair never falls back to two narrow
 * transforms merely because its second tile is partial.
 */
template <int LeftTile, int RightTile, meta::ValueType ActiveM, meta::ValueType LeftActiveN,
          meta::ValueType RightActiveN, typename COutput>
VECOPS_ALWAYS_INLINE void store_c_tile_row_pair(COutput& output, nint_t m, nint_t n, ActiveM active_m,
                                                LeftActiveN left_active_n, RightActiveN right_active_n,
                                                typename COutput::ComputeType* left_buffer,
                                                typename COutput::ComputeType* right_buffer) {
  using T = typename COutput::ComputeType;
  using Tag = vec::ScalableTag<T, 0>;
  using WideTag = vec::Twice<Tag>;
  static_assert(tensor::preferred_memory_access_power_v<COutput> > 0);
  static_assert(vec::size(Tag{}) == 16);
  const nint_t active_m_value = static_cast<nint_t>(active_m);
  const nint_t left_n = static_cast<nint_t>(left_active_n);
  const nint_t right_n = static_cast<nint_t>(right_active_n);
  const nint_t active_n = left_n + right_n;
  if (active_m_value == 0 || active_n == 0)
    return;

  amx_intrinsics::store<LeftTile>(left_buffer, 16 * sizeof(T));
  amx_intrinsics::store<RightTile>(right_buffer, 16 * sizeof(T));
  for (nint_t row = 0; row < active_m_value; ++row) {
    const auto left = vec::load(Tag{}, left_buffer + row * 16);
    const auto right = vec::load(Tag{}, right_buffer + row * 16);
    if constexpr (transform_store_mode_v<COutput> == tensor::TransformStoreMode::coalesced) {
      static_assert(requires {
        output
          .template store_transform_pair_coalesced<full_tile_extent_v<LeftActiveN>, full_tile_extent_v<RightActiveN>>(
            Tag{}, tensor::coord(m + row, n), tensor::coord(m + row, n + 16), tensor::axis<1>, left, right,
            left_active_n, right_active_n);
      });
      output.template store_transform_pair_coalesced<full_tile_extent_v<LeftActiveN>, full_tile_extent_v<RightActiveN>>(
        Tag{}, tensor::coord(m + row, n), tensor::coord(m + row, n + 16), tensor::axis<1>, left, right, left_active_n,
        right_active_n);
    } else {
      const auto value = vec::concat(WideTag{}, left, right);
      if constexpr (full_tile_extent_v<LeftActiveN> && full_tile_extent_v<RightActiveN>) {
        output.store(WideTag{}, tensor::coord(m + row, n), tensor::axis<1>, value);
      } else {
        output.store(WideTag{}, tensor::coord(m + row, n), tensor::axis<1>, value, vec::opt::first(active_n));
      }
    }
  }
}

template <int TopTile, int BottomTile, meta::ValueType TopActiveM, meta::ValueType BottomActiveM,
          meta::ValueType ActiveN, typename COutput>
VECOPS_ALWAYS_INLINE void store_c_tile_column_pair(COutput& output, nint_t m, nint_t n, TopActiveM top_active_m,
                                                   BottomActiveM bottom_active_m, ActiveN active_n,
                                                   typename COutput::ComputeType* top_buffer,
                                                   typename COutput::ComputeType* bottom_buffer) {
  using T = typename COutput::ComputeType;
  using Tag = vec::ScalableTag<T, 0>;
  using WideTag = vec::Twice<Tag>;
  static_assert(tensor::preferred_memory_access_power_v<COutput> > 0);
  const nint_t top_m = static_cast<nint_t>(top_active_m);
  const nint_t bottom_m = static_cast<nint_t>(bottom_active_m);
  const nint_t active_m = top_m + bottom_m;
  const nint_t active_n_value = static_cast<nint_t>(active_n);
  if (active_m == 0 || active_n_value == 0)
    return;

  amx_intrinsics::store<TopTile>(top_buffer, 16 * sizeof(T));
  amx_intrinsics::store<BottomTile>(bottom_buffer, 16 * sizeof(T));
  for (nint_t column_offset = 0; column_offset < active_n_value; column_offset += 8) {
    const auto top = read_c_tile_column_block(top_buffer, top_m, column_offset);
    const auto bottom = read_c_tile_column_block(bottom_buffer, bottom_m, column_offset);
    const nint_t columns = vecops::min(active_n_value - column_offset, nint_t{8});
    for (nint_t column = 0; column < columns; ++column) {
      const auto top_value = top[static_cast<std::size_t>(column)];
      const auto bottom_value = bottom[static_cast<std::size_t>(column)];
      if constexpr (transform_store_mode_v<COutput> == tensor::TransformStoreMode::coalesced) {
        static_assert(requires {
          output
            .template store_transform_pair_coalesced<full_tile_extent_v<TopActiveM>, full_tile_extent_v<BottomActiveM>>(
              Tag{}, tensor::coord(m, n + column_offset + column), tensor::coord(m + 16, n + column_offset + column),
              tensor::axis<0>, top_value, bottom_value, top_active_m, bottom_active_m);
        });
        output
          .template store_transform_pair_coalesced<full_tile_extent_v<TopActiveM>, full_tile_extent_v<BottomActiveM>>(
            Tag{}, tensor::coord(m, n + column_offset + column), tensor::coord(m + 16, n + column_offset + column),
            tensor::axis<0>, top_value, bottom_value, top_active_m, bottom_active_m);
      } else {
        const auto value = vec::concat(WideTag{}, top_value, bottom_value);
        if constexpr (full_tile_extent_v<TopActiveM> && full_tile_extent_v<BottomActiveM>) {
          output.store(WideTag{}, tensor::coord(m, n + column_offset + column), tensor::axis<0>, value);
        } else {
          output.store(WideTag{}, tensor::coord(m, n + column_offset + column), tensor::axis<0>, value,
                       vec::opt::first(active_m));
        }
      }
    }
  }
}

/** Select the semantic C input or the running split-K accumulator. */
template <int Tile, meta::ValueType ActiveM, meta::ValueType ActiveN, typename CInput, typename AccInput,
          typename Route>
VECOPS_ALWAYS_INLINE void initialize_c_tile_routed(const CInput& c_input, const AccInput& acc_input, const Route& route,
                                                   nint_t m, nint_t n, ActiveM active_m, ActiveN active_n,
                                                   typename CInput::ComputeType* buffer) {
  static_assert(std::same_as<
      typename CInput::ComputeType, typename AccInput::ComputeType>);
  if constexpr (matmul_details::static_accumulator_route_v<Route>) {
    if constexpr (Route::use_accumulator_input)
      initialize_c_tile<Tile>(
          acc_input, m, n, active_m, active_n, buffer);
    else
      initialize_c_tile<Tile>(
          c_input, m, n, active_m, active_n, buffer);
  } else {
    if (route.use_accumulator_input)
      initialize_c_tile<Tile>(
          acc_input, m, n, active_m, active_n, buffer);
    else
      initialize_c_tile<Tile>(
          c_input, m, n, active_m, active_n, buffer);
  }
}

/** Select the running split-K accumulator or the semantic C output. */
template <int Tile, meta::ValueType ActiveM, meta::ValueType ActiveN,
          typename COutput, typename AccOutput, typename Route>
VECOPS_ALWAYS_INLINE void store_c_tile_routed(
    COutput& c_output, AccOutput& acc_output, const Route& route,
    nint_t m, nint_t n, ActiveM active_m, ActiveN active_n,
    typename COutput::ComputeType* buffer) {
  static_assert(std::same_as<
      typename COutput::ComputeType, typename AccOutput::ComputeType>);
  if constexpr (matmul_details::static_accumulator_route_v<Route>) {
    if constexpr (Route::write_accumulator_output)
      store_c_tile<Tile>(
          acc_output, m, n, active_m, active_n, buffer);
    else
      store_c_tile<Tile>(
          c_output, m, n, active_m, active_n, buffer);
  } else {
    if (route.write_accumulator_output)
      store_c_tile<Tile>(
          acc_output, m, n, active_m, active_n, buffer);
    else
      store_c_tile<Tile>(
          c_output, m, n, active_m, active_n, buffer);
  }
}

template <int LeftTile, int RightTile, meta::ValueType ActiveM, meta::ValueType LeftActiveN,
          meta::ValueType RightActiveN, typename CInput, typename AccInput, typename Route>
VECOPS_ALWAYS_INLINE void initialize_c_tile_row_pair_routed(const CInput& c_input, const AccInput& acc_input,
                                                            const Route& route, nint_t m, nint_t n, ActiveM active_m,
                                                            LeftActiveN left_active_n, RightActiveN right_active_n,
                                                            typename CInput::ComputeType* left_buffer,
                                                            typename CInput::ComputeType* right_buffer) {
  static_assert(std::same_as<typename CInput::ComputeType, typename AccInput::ComputeType>);
  auto initialize_semantic = [&]() VECOPS_INLINE_LAMBDA {
    if constexpr (tensor::preferred_memory_access_power_v<CInput> > 0) {
      initialize_c_tile_row_pair<LeftTile, RightTile>(c_input, m, n, active_m, left_active_n, right_active_n,
                                                      left_buffer, right_buffer);
    } else {
      initialize_c_tile<LeftTile>(c_input, m, n, active_m, left_active_n, left_buffer);
      initialize_c_tile<RightTile>(c_input, m, n + 16, active_m, right_active_n, right_buffer);
    }
  };
  auto initialize_accumulator = [&]() VECOPS_INLINE_LAMBDA {
    if constexpr (tensor::preferred_memory_access_power_v<AccInput> > 0) {
      initialize_c_tile_row_pair<LeftTile, RightTile>(acc_input, m, n, active_m, left_active_n, right_active_n,
                                                      left_buffer, right_buffer);
    } else {
      initialize_c_tile<LeftTile>(acc_input, m, n, active_m, left_active_n, left_buffer);
      initialize_c_tile<RightTile>(acc_input, m, n + 16, active_m, right_active_n, right_buffer);
    }
  };
  if constexpr (matmul_details::static_accumulator_route_v<Route>) {
    if constexpr (Route::use_accumulator_input)
      initialize_accumulator();
    else
      initialize_semantic();
  } else {
    if (route.use_accumulator_input)
      initialize_accumulator();
    else
      initialize_semantic();
  }
}

template <int LeftTile, int RightTile, meta::ValueType ActiveM, meta::ValueType LeftActiveN,
          meta::ValueType RightActiveN, typename COutput, typename AccOutput, typename Route>
VECOPS_ALWAYS_INLINE void
store_c_tile_row_pair_routed(COutput& c_output, AccOutput& acc_output, const Route& route, nint_t m, nint_t n,
                             ActiveM active_m, LeftActiveN left_active_n, RightActiveN right_active_n,
                             typename COutput::ComputeType* left_buffer, typename COutput::ComputeType* right_buffer) {
  static_assert(std::same_as<typename COutput::ComputeType, typename AccOutput::ComputeType>);
  auto store_semantic = [&]() VECOPS_INLINE_LAMBDA {
    if constexpr (tensor::preferred_memory_access_power_v<COutput> > 0 && transform_store_groups_tiles_v<COutput>) {
      store_c_tile_row_pair<LeftTile, RightTile>(c_output, m, n, active_m, left_active_n, right_active_n, left_buffer,
                                                 right_buffer);
    } else {
      store_c_tile<LeftTile>(c_output, m, n, active_m, left_active_n, left_buffer);
      store_c_tile<RightTile>(c_output, m, n + 16, active_m, right_active_n, right_buffer);
    }
  };
  auto store_accumulator = [&]() VECOPS_INLINE_LAMBDA {
    if constexpr (tensor::preferred_memory_access_power_v<AccOutput> > 0 && transform_store_groups_tiles_v<AccOutput>) {
      store_c_tile_row_pair<LeftTile, RightTile>(acc_output, m, n, active_m, left_active_n, right_active_n, left_buffer,
                                                 right_buffer);
    } else {
      store_c_tile<LeftTile>(acc_output, m, n, active_m, left_active_n, left_buffer);
      store_c_tile<RightTile>(acc_output, m, n + 16, active_m, right_active_n, right_buffer);
    }
  };
  if constexpr (matmul_details::static_accumulator_route_v<Route>) {
    if constexpr (Route::write_accumulator_output)
      store_accumulator();
    else
      store_semantic();
  } else {
    if (route.write_accumulator_output)
      store_accumulator();
    else
      store_semantic();
  }
}

template <int TopTile, int BottomTile, meta::ValueType TopActiveM, meta::ValueType BottomActiveM,
          meta::ValueType ActiveN, typename COutput, typename AccOutput, typename Route>
VECOPS_ALWAYS_INLINE void store_c_tile_column_pair_routed(COutput& c_output, AccOutput& acc_output, const Route& route,
                                                          nint_t m, nint_t n, TopActiveM top_active_m,
                                                          BottomActiveM bottom_active_m, ActiveN active_n,
                                                          typename COutput::ComputeType* top_buffer,
                                                          typename COutput::ComputeType* bottom_buffer) {
  static_assert(std::same_as<typename COutput::ComputeType, typename AccOutput::ComputeType>);
  auto store_semantic = [&]() VECOPS_INLINE_LAMBDA {
    if constexpr (tensor::preferred_memory_access_power_v<COutput> > 0 && transform_store_groups_tiles_v<COutput>) {
      store_c_tile_column_pair<TopTile, BottomTile>(c_output, m, n, top_active_m, bottom_active_m, active_n, top_buffer,
                                                    bottom_buffer);
    } else {
      store_c_tile<TopTile>(c_output, m, n, top_active_m, active_n, top_buffer);
      store_c_tile<BottomTile>(c_output, m + 16, n, bottom_active_m, active_n, bottom_buffer);
    }
  };
  auto store_accumulator = [&]() VECOPS_INLINE_LAMBDA {
    if constexpr (tensor::preferred_memory_access_power_v<AccOutput> > 0 && transform_store_groups_tiles_v<AccOutput>) {
      store_c_tile_column_pair<TopTile, BottomTile>(acc_output, m, n, top_active_m, bottom_active_m, active_n,
                                                    top_buffer, bottom_buffer);
    } else {
      store_c_tile<TopTile>(acc_output, m, n, top_active_m, active_n, top_buffer);
      store_c_tile<BottomTile>(acc_output, m + 16, n, bottom_active_m, active_n, bottom_buffer);
    }
  };
  if constexpr (matmul_details::static_accumulator_route_v<Route>) {
    if constexpr (Route::write_accumulator_output)
      store_accumulator();
    else
      store_semantic();
  } else {
    if (route.write_accumulator_output)
      store_accumulator();
    else
      store_semantic();
  }
}

/// The AMX microkernel for one Case: initialize the C tiles, loop the K
/// axis in Atom K steps, store the C tiles. Tail handling is resolved
/// per sub-tile through the *Guaranteed / NonEmpty template flags.
template <::vecops::matmul::Atom Atom, typename Case, typename Plan, typename A, typename B, typename CInput,
          typename COutput, typename AccInput, typename AccOutput, typename Route>
VECOPS_ALWAYS_INLINE void microkernel_routed(const A& a, const B& b, const CInput& c_input, COutput& c_output,
                                             const AccInput& acc_input, AccOutput& acc_output, Route route,
                                             nint_t logical_m, nint_t logical_n, nint_t logical_k, nint_t m, nint_t n,
                                             nint_t active_m, nint_t active_n, void* scratch) {
  constexpr int NM = Case::a;
  constexpr int NN = Case::b;
  constexpr int Outputs = NM * NN;
  constexpr nint_t KR = decltype(Atom::K_R)::value;
  constexpr bool FullM =
      Case::m_mask == tile::Tile2DMaskMode::unmasked;
  constexpr bool FullN =
      Case::n_mask == tile::Tile2DMaskMode::unmasked;
  // GCC 13 lowers the active-extent clamp differently depending on whether
  // the result follows the reference-returning clamp ABI. Multi-column AMX
  // families are faster with that form, while single-column/GEMV families
  // prefer the ordinary value form. Both are force-inlined project utilities.
  constexpr bool ReferenceActiveClamp = NN > 1;
  // GCC miscompiles the early AMX tile-zero control flow for the raw-A /
  // packed-B specialization. Keep its inactive A tile on the scratch path;
  // every other covered input combination safely uses the faster tile zero.
  constexpr bool DirectInactiveZeroA =
      is_packed_access_v<Atom, ::vecops::matmul::Operand::A, A> ||
      !is_packed_access_v<Atom, ::vecops::matmul::Operand::B, B>;
  auto* bytes = static_cast<std::byte*>(scratch);
  auto* a_buffers = reinterpret_cast<typename Atom::TA*>(bytes);
  auto* b_buffers = reinterpret_cast<typename Atom::TB*>(bytes + NM * 1024);
  auto* c_buffers = reinterpret_cast<typename Atom::TAcc*>(bytes + (NM + NN) * 1024);

  auto initialize_tile = [&]<std::size_t I>() VECOPS_INLINE_LAMBDA {
    initialize_c_tile_routed<static_cast<int>(I)>(
      c_input, acc_input, route, m + static_cast<nint_t>(I / NN) * 16, n + static_cast<nint_t>(I % NN) * 16,
      tile_active_extent < FullM || (Case::exact_blocks && I / NN + 1 < NM), FullM || Case::exact_blocks,
      static_cast<nint_t>(I / NN) * 16, ReferenceActiveClamp > (active_m),
      tile_active_extent < FullN || (Case::exact_blocks && I % NN + 1 < NN), FullN || Case::exact_blocks,
      static_cast<nint_t>(I % NN) * 16, ReferenceActiveClamp > (active_n), c_buffers + I * 256);
  };
  constexpr bool GroupCInputRows =
    NN >= 2 && row_contiguous_input_v<CInput> && row_contiguous_input_v<AccInput> &&
    ((!IsZeroTransform<typename CInput::Transform>::value && tensor::preferred_memory_access_power_v<CInput> > 0) ||
     (!IsZeroTransform<typename AccInput::Transform>::value && tensor::preferred_memory_access_power_v<AccInput> > 0));
  if constexpr (GroupCInputRows) {
    auto initialize_row_pair = [&]<std::size_t Row>() VECOPS_INLINE_LAMBDA {
      constexpr std::size_t Left = Row * NN;
      constexpr std::size_t Right = Left + 1;
      initialize_c_tile_row_pair_routed<static_cast<int>(Left), static_cast<int>(Right)>(
        c_input, acc_input, route, m + static_cast<nint_t>(Row) * 16, n,
        tile_active_extent < FullM || (Case::exact_blocks && Row + 1 < NM), FullM || Case::exact_blocks,
        static_cast<nint_t>(Row) * 16, ReferenceActiveClamp > (active_m),
        tile_active_extent < FullN || (Case::exact_blocks && 1 < NN), FullN || Case::exact_blocks, 0,
        ReferenceActiveClamp > (active_n), tile_active_extent < FullN || (Case::exact_blocks && 2 < NN),
        FullN || Case::exact_blocks, 16, ReferenceActiveClamp > (active_n), c_buffers + Left * 256,
        c_buffers + Right * 256);
      if constexpr (NN == 3)
        initialize_tile.template operator()<Left + 2>();
    };
    [&]<std::size_t... Row>(std::index_sequence<Row...>)
      VECOPS_INLINE_LAMBDA { (initialize_row_pair.template operator()<Row>(), ...); }(std::make_index_sequence<NM>{});
  } else {
    [&]<std::size_t... I>(std::index_sequence<I...>)
      VECOPS_INLINE_LAMBDA { (initialize_tile.template operator()<I>(), ...); }(std::make_index_sequence<Outputs>{});
  }
  constexpr bool SplitKForAccess =
    (!is_packed_access_v<Atom, ::vecops::matmul::Operand::A, A> && !direct_row_major_input_v<A>) ||
    (!is_packed_access_v<Atom, ::vecops::matmul::Operand::B, B> && !direct_row_major_input_v<B>);
  if constexpr (Plan::value && is_packed_access_v<Atom, ::vecops::matmul::Operand::A, A> &&
                is_packed_access_v<Atom, ::vecops::matmul::Operand::B, B>) {
    multiply_packed_k_loop<Atom, Case, Plan::stream_b>(
        a, b, logical_k, m, n);
  } else if constexpr (Plan::value) {
    for (nint_t k = 0; k < logical_k; k += KR) {
      multiply_k_tile<
          Atom, Case, true, Plan::stream_b, DirectInactiveZeroA>(
          a, b, logical_m, logical_n, logical_k, m, n, k,
          a_buffers, b_buffers);
    }
  } else if constexpr (!SplitKForAccess) {
    // Direct inputs already select an unmasked tileload/full-panel pack at
    // runtime, while packed panels are padded in K. Splitting either kind
    // would clone the kernel body merely to remove a predictable condition.
    for (nint_t k = 0; k < logical_k; k += KR) {
      multiply_k_tile<
          Atom, Case, false, Plan::stream_b, DirectInactiveZeroA>(
          a, b, logical_m, logical_n, logical_k, m, n, k,
          a_buffers, b_buffers);
    }
  } else {
    const nint_t full_k = logical_k / KR * KR;
    for (nint_t k = 0; k < full_k; k += KR) {
      multiply_k_tile<
          Atom, Case, true, Plan::stream_b, DirectInactiveZeroA>(
          a, b, logical_m, logical_n, logical_k, m, n, k,
          a_buffers, b_buffers);
    }
    if (full_k < logical_k) {
      multiply_k_tile<
          Atom, Case, false, Plan::stream_b, DirectInactiveZeroA>(
          a, b, logical_m, logical_n, logical_k, m, n, full_k,
          a_buffers, b_buffers);
    }
  }

  auto store_tile = [&]<std::size_t I>() VECOPS_INLINE_LAMBDA {
    store_c_tile_routed<static_cast<int>(I)>(
      c_output, acc_output, route, m + static_cast<nint_t>(I / NN) * 16, n + static_cast<nint_t>(I % NN) * 16,
      tile_active_extent < FullM || (Case::exact_blocks && I / NN + 1 < NM), FullM || Case::exact_blocks,
      static_cast<nint_t>(I / NN) * 16, ReferenceActiveClamp > (active_m),
      tile_active_extent < FullN || (Case::exact_blocks && I % NN + 1 < NN), FullN || Case::exact_blocks,
      static_cast<nint_t>(I % NN) * 16, ReferenceActiveClamp > (active_n), c_buffers + I * 256);
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
    auto store_column_pair = [&]<std::size_t Column>() VECOPS_INLINE_LAMBDA {
      constexpr std::size_t Top = Column;
      constexpr std::size_t Bottom = NN + Column;
      store_c_tile_column_pair_routed<static_cast<int>(Top), static_cast<int>(Bottom)>(
        c_output, acc_output, route, m, n + static_cast<nint_t>(Column) * 16,
        tile_active_extent < FullM || (Case::exact_blocks && 1 < NM), FullM || Case::exact_blocks, 0,
        ReferenceActiveClamp > (active_m), tile_active_extent < FullM || (Case::exact_blocks && 2 < NM),
        FullM || Case::exact_blocks, 16, ReferenceActiveClamp > (active_m),
        tile_active_extent < FullN || (Case::exact_blocks && Column + 1 < NN), FullN || Case::exact_blocks,
        static_cast<nint_t>(Column) * 16, ReferenceActiveClamp > (active_n), c_buffers + Top * 256,
        c_buffers + Bottom * 256);
      if constexpr (NM == 3)
        store_tile.template operator()<2 * NN + Column>();
    };
    [&]<std::size_t... Column>(std::index_sequence<Column...>)
      VECOPS_INLINE_LAMBDA { (store_column_pair.template operator()<Column>(), ...); }(std::make_index_sequence<NN>{});
  } else if constexpr (GroupCOutputRows) {
    auto store_row_pair = [&]<std::size_t Row>() VECOPS_INLINE_LAMBDA {
      constexpr std::size_t Left = Row * NN;
      constexpr std::size_t Right = Left + 1;
      store_c_tile_row_pair_routed<static_cast<int>(Left), static_cast<int>(Right)>(
        c_output, acc_output, route, m + static_cast<nint_t>(Row) * 16, n,
        tile_active_extent < FullM || (Case::exact_blocks && Row + 1 < NM), FullM || Case::exact_blocks,
        static_cast<nint_t>(Row) * 16, ReferenceActiveClamp > (active_m),
        tile_active_extent < FullN || (Case::exact_blocks && 1 < NN), FullN || Case::exact_blocks, 0,
        ReferenceActiveClamp > (active_n), tile_active_extent < FullN || (Case::exact_blocks && 2 < NN),
        FullN || Case::exact_blocks, 16, ReferenceActiveClamp > (active_n), c_buffers + Left * 256,
        c_buffers + Right * 256);
      if constexpr (NN == 3)
        store_tile.template operator()<Left + 2>();
    };
    [&]<std::size_t... Row>(std::index_sequence<Row...>)
      VECOPS_INLINE_LAMBDA { (store_row_pair.template operator()<Row>(), ...); }(std::make_index_sequence<NM>{});
  } else {
    [&]<std::size_t... I>(std::index_sequence<I...>)
      VECOPS_INLINE_LAMBDA { (store_tile.template operator()<I>(), ...); }(std::make_index_sequence<Outputs>{});
  }
}

/** Static unsplit adapter: semantic C input is written to semantic output. */
template <::vecops::matmul::Atom Atom, typename Case, typename Plan,
          typename A, typename B, typename CInput, typename COutput>
VECOPS_ALWAYS_INLINE void microkernel(
    const A& a, const B& b, const CInput& c_input, COutput& c_output,
    nint_t logical_m, nint_t logical_n, nint_t logical_k,
    nint_t m, nint_t n, nint_t active_m, nint_t active_n,
    void* scratch) {
  microkernel_routed<Atom, Case, Plan>(
      a, b, c_input, c_output, c_input, c_output,
      matmul_details::UnsplitAccumulatorRoute{},
      logical_m, logical_n, logical_k,
      m, n, active_m, active_n, scratch);
}

/// Out-of-line wrapper that pins a microkernel to an exact-blocks Case,
/// so the TileScheduler's exact traversal gets its own call target and
/// the common path stays inlinable.
template <::vecops::matmul::Atom Atom, typename Case, typename Plan,
          typename A, typename B, typename CInput, typename COutput>
VECOPS_NOINLINE void exact_microkernel(
    const A& a, const B& b, const CInput& c_input, COutput& c_output,
    nint_t logical_m, nint_t logical_n, nint_t logical_k,
    nint_t m, nint_t n, nint_t active_m, nint_t active_n,
    void* scratch) {
  static_assert(Case::exact_blocks);
  microkernel<Atom, Case, Plan>(
      a, b, c_input, c_output,
      logical_m, logical_n, logical_k,
      m, n, active_m, active_n, scratch);
}

/** Shared first/middle/last split-K wrapper; prevent GCC IPA phase clones. */
template <::vecops::matmul::Atom Atom, typename Case, typename Plan,
          typename A, typename B,
          typename CInput, typename COutput,
          typename AccInput, typename AccOutput, typename Route>
VECOPS_NOINLINE
#if defined(COMPILER_GCC)
__attribute__((noclone))
#endif
void exact_microkernel_phased(
    const A& a, const B& b,
    const CInput& c_input, COutput& c_output,
    const AccInput& acc_input, AccOutput& acc_output, Route route,
    nint_t logical_m, nint_t logical_n, nint_t logical_k,
    nint_t m, nint_t n, nint_t active_m, nint_t active_n,
    void* scratch) {
  static_assert(Case::exact_blocks);
  static_assert(!matmul_details::static_accumulator_route_v<Route>);
  microkernel_routed<Atom, Case, Plan>(
      a, b, c_input, c_output, acc_input, acc_output, route,
      logical_m, logical_n, logical_k,
      m, n, active_m, active_n, scratch);
}

} // namespace vecops::kernel::matmul_details::amx

namespace vecops::kernel::matmul_details {

/// AMX implementation of the kernel Backend contract: the resource set
/// (one x86 Tiles image), the Tile2D catalog, and the run entry points
/// that fold the DispatchOwner selection of this header into the
/// configured tile traversal.
template <>
struct Backend<matmul_implementation::AMX> {
  using ResourceRequirements = execution::details::ResourceSet<
      execution::details::x86::Tiles>;
  template <::vecops::matmul::Atom Atom, typename Policy,
            meta::ValueType M, meta::ValueType N, meta::ValueType K,
            typename A, typename B, typename CInput, typename COutput>
  using Catalog = amx::Catalog<amx::kernel_aspect_v<M, N, K>>;
  // N-major already changes the A/B reuse trade-off. Keep its finite leaf
  // set balanced until vertical/horizontal families have independently
  // measured N-major crossover rules.
  template <::vecops::matmul::Atom, typename,
            meta::ValueType, meta::ValueType, meta::ValueType,
            typename, typename, typename, typename>
  using NMajorCatalog = amx::Catalog<>;
  static constexpr int ProblemRank = 2;

  /// Scratch size for one microkernel staging area (see the file header
  /// for the layout); includes 63 bytes of caller-side alignment pad.
  static nint_t scratch_bytes() { return 8 * 1024 + 63; }

  template <::vecops::matmul::Atom Atom, typename Policy,
            meta::ValueType M, meta::ValueType N, meta::ValueType K,
            typename A, typename B, typename CInput, typename COutput>
  using EffectivePolicy = std::conditional_t<
      std::same_as<Policy, matmul_policy::Automatic>,
      std::conditional_t<
          amx::direct_row_major_input_v<CInput> &&
              !amx::IsZeroTransform<typename CInput::Transform>::value,
          kernel::loop::tile2d_policy::ExactCoverRuntimeUnmasked,
          kernel::loop::tile2d_policy::ExactCover>,
      Policy>;

  template <::vecops::matmul::Atom Atom, typename Policy, bool PackedB,
            execution::ExecutionScope Scope,
            meta::ValueType M, meta::ValueType N,
            typename Fn>
  VECOPS_ALWAYS_INLINE static decltype(auto) with_configuration(
      Scope& scope, M m, N n, Fn&& fn) {
    static_assert(std::same_as<typename Atom::KernelKind, ::vecops::matmul::AMXKernelKind>);
    static_assert(execution::has_resource_v<
        execution::details::x86::Tiles, Scope>);
    amx::Configuration configuration;
    using MV = std::remove_cvref_t<M>;
    using NV = std::remove_cvref_t<N>;
    if constexpr (std::same_as<Policy, matmul_policy::Automatic> && PackedB &&
                  meta::range_within_v<MV, 1, 16> &&
                  meta::lower_bound_at_least_v<NV, 1> &&
                  meta::has_upper_bound_v<NV>) {
      constexpr nint_t MinNBlocks =
          ceil_div(meta::lower_bound_v<NV>, nint_t{16});
      constexpr nint_t MaxNBlocks =
          ceil_div(meta::upper_bound_v<NV>, nint_t{16});
      if constexpr (MinNBlocks == 2 && MaxNBlocks == 2) {
          // The exact 1x2 family shares the same compact register mapping as
          // 1x3.  Shortening its C/A rows removes inactive-row TMUL work for
          // the common N=32 shared-weight case without changing traversal.
        configuration.set_horizontal_rows(static_cast<nint_t>(m), 2);
      } else if constexpr (MinNBlocks == MaxNBlocks &&
                           MinNBlocks % 3 == 0) {
        configuration.set_horizontal_rows(static_cast<nint_t>(m), 3);
      }
    }
    return scope.with_configuration(
        configuration, std::forward<Fn>(fn));
  }

  template <::vecops::matmul::Atom Atom, typename Policy,
            execution::ExecutionScope Scope,
            meta::ValueType M, meta::ValueType N, meta::ValueType K,
            typename A, typename B, typename CInput, typename COutput>
  VECOPS_ALWAYS_INLINE static void run_configured(
      Scope&, M m, N n, K k,
      const A& a, const B& b, const CInput& c_input, COutput& c_output,
      void* scratch) {
    static_assert(std::same_as<typename Atom::KernelKind, ::vecops::matmul::AMXKernelKind>);
    static_assert(execution::has_resource_v<
        execution::details::x86::Tiles, Scope>);
    static_assert(std::same_as<
        typename std::remove_cvref_t<Scope>::ActiveConfiguration,
        amx::Configuration>);
    VECOPS_ASSERT(scratch != nullptr, "AMX matmul scratch is null");
    matmul_details::run_tiles<Backend, Atom, Policy>(
        m, n, k, a, b, c_input, c_output, scratch);
  }

  template <::vecops::matmul::Atom Atom, typename Policy,
            execution::ExecutionScope Scope,
            meta::ValueType TraversalM, meta::ValueType TraversalN,
            meta::ValueType K,
            typename A, typename B, typename CInput, typename COutput>
  VECOPS_ALWAYS_INLINE static void run_configured_region(
      Scope&, TraversalM traversal_m, TraversalN traversal_n, K k,
      nint_t logical_m, nint_t logical_n,
      nint_t origin_m, nint_t origin_n,
      const A& a, const B& b,
      const CInput& c_input, COutput& c_output,
      void* scratch) {
    static_assert(std::same_as<typename Atom::KernelKind, ::vecops::matmul::AMXKernelKind>);
    static_assert(execution::has_resource_v<
        execution::details::x86::Tiles, Scope>);
    static_assert(std::same_as<
        typename std::remove_cvref_t<Scope>::ActiveConfiguration,
        amx::Configuration>);
    VECOPS_ASSERT(scratch != nullptr, "AMX matmul scratch is null");
    matmul_details::run_tiles_region<Backend, Atom, Policy>(
        traversal_m, traversal_n, k, logical_m, logical_n,
        origin_m, origin_n, a, b, c_input, c_output, scratch);
  }

  /** Configured split-K leaf with runtime C endpoint routing. */
  template <::vecops::matmul::Atom Atom, typename Policy,
            execution::ExecutionScope Scope,
            meta::ValueType M, meta::ValueType N, meta::ValueType K,
            typename A, typename B,
            typename CInput, typename COutput,
            typename AccInput, typename AccOutput, typename Route>
  VECOPS_ALWAYS_INLINE static void run_configured_phased(
      Scope&, M m, N n, K k,
      const A& a, const B& b,
      const CInput& c_input, COutput& c_output,
      const AccInput& acc_input, AccOutput& acc_output,
      Route route, void* scratch) {
    static_assert(std::same_as<
        typename Atom::KernelKind, ::vecops::matmul::AMXKernelKind>);
    static_assert(execution::has_resource_v<
        execution::details::x86::Tiles, Scope>);
    static_assert(std::same_as<
        typename std::remove_cvref_t<Scope>::ActiveConfiguration,
        amx::Configuration>);
    VECOPS_ASSERT(scratch != nullptr, "AMX matmul scratch is null");
    matmul_details::run_tiles_phased<Backend, Atom, Policy>(
        m, n, k, a, b, c_input, c_output,
        acc_input, acc_output, route, scratch);
  }

  /**
   * General split-K entry. Whole-problem special owners remain on the static
   * unsplit API; a real cache-K loop shares this configured AMX traversal.
   */
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
    constexpr bool PackedB = amx::is_packed_access_v<
        Atom, ::vecops::matmul::Operand::B, B>;
    constexpr bool HasActiveConfiguration = [] {
      if constexpr (requires {
                      typename std::remove_cvref_t<Scope>::ActiveConfiguration;
                    }) {
        return std::same_as<
            typename std::remove_cvref_t<Scope>::ActiveConfiguration,
            amx::Configuration>;
      } else {
        return false;
      }
    }();
    if constexpr (HasActiveConfiguration) {
      run_configured_phased<Atom, Policy>(
          scope, m, n, k, a, b, c_input, c_output,
          acc_input, acc_output, route, scratch);
    } else {
      with_configuration<Atom, Policy, PackedB>(
          scope, m, n, [&](auto& configured)
              VECOPS_INLINE_LAMBDA_NOEXCEPT {
            run_configured_phased<Atom, Policy>(
                configured, m, n, k, a, b, c_input, c_output,
                acc_input, acc_output, route, scratch);
          });
    }
  }

  template <::vecops::matmul::Atom Atom, typename Policy,
            execution::ExecutionScope Scope,
            meta::ValueType M, meta::ValueType N, meta::ValueType K,
            typename A, typename B, typename CInput, typename COutput>
  VECOPS_ALWAYS_INLINE static void run_configured_n_major(
      Scope&, M m, N n, K k, const A& a, const B& b,
      const CInput& c_input, COutput& c_output, void* scratch) {
    static_assert(execution::has_resource_v<
        execution::details::x86::Tiles, Scope>);
    VECOPS_ASSERT(scratch != nullptr, "AMX matmul scratch is null");
    matmul_details::run_tiles_n_major<Backend, Atom, Policy>(
        m, n, k, a, b, c_input, c_output, scratch);
  }

  template <::vecops::matmul::Atom Atom, typename Policy,
            execution::ExecutionScope Scope,
            meta::ValueType TraversalM, meta::ValueType TraversalN,
            meta::ValueType K,
            typename A, typename B, typename CInput, typename COutput>
  VECOPS_ALWAYS_INLINE static void run_configured_region_n_major(
      Scope&, TraversalM traversal_m, TraversalN traversal_n, K k,
      nint_t logical_m, nint_t logical_n,
      nint_t origin_m, nint_t origin_n,
      const A& a, const B& b, const CInput& c_input, COutput& c_output,
      void* scratch) {
    static_assert(execution::has_resource_v<
        execution::details::x86::Tiles, Scope>);
    VECOPS_ASSERT(scratch != nullptr, "AMX matmul scratch is null");
    matmul_details::run_tiles_region_n_major<Backend, Atom, Policy>(
        traversal_m, traversal_n, k, logical_m, logical_n,
        origin_m, origin_n, a, b, c_input, c_output, scratch);
  }

  template <::vecops::matmul::Atom Atom, typename Policy,
            execution::ExecutionScope Scope,
            meta::ValueType M, meta::ValueType N, meta::ValueType K,
            typename A, typename B, typename CInput, typename COutput,
            typename AccInput, typename AccOutput, typename Route>
  VECOPS_ALWAYS_INLINE static void run_configured_phased_n_major(
      Scope&, M m, N n, K k, const A& a, const B& b,
      const CInput& c_input, COutput& c_output,
      const AccInput& acc_input, AccOutput& acc_output,
      Route route, void* scratch) {
    static_assert(execution::has_resource_v<
        execution::details::x86::Tiles, Scope>);
    VECOPS_ASSERT(scratch != nullptr, "AMX matmul scratch is null");
    matmul_details::run_tiles_phased_n_major<Backend, Atom, Policy>(
        m, n, k, a, b, c_input, c_output,
        acc_input, acc_output, route, scratch);
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
    constexpr bool PackedB = amx::is_packed_access_v<
        Atom, ::vecops::matmul::Operand::B, B>;
    constexpr bool HasActiveConfiguration = [] {
      if constexpr (requires {
                      typename std::remove_cvref_t<Scope>::ActiveConfiguration;
                    })
        return std::same_as<
            typename std::remove_cvref_t<Scope>::ActiveConfiguration,
            amx::Configuration>;
      else
        return false;
    }();
    if constexpr (HasActiveConfiguration) {
      run_configured_phased_n_major<Atom, Policy>(
          scope, m, n, k, a, b, c_input, c_output,
          acc_input, acc_output, route, scratch);
    } else {
      with_configuration<Atom, Policy, PackedB>(
          scope, m, n, [&](auto& configured) VECOPS_INLINE_LAMBDA_NOEXCEPT {
            run_configured_phased_n_major<Atom, Policy>(
                configured, m, n, k, a, b, c_input, c_output,
                acc_input, acc_output, route, scratch);
          });
    }
  }

  /// Entry point, resolving the leaf in four tiers (first match wins):
  ///
  /// 1. Explicit family requests the automatic owner does not already
  ///    satisfy: if the family's applicability probe (and, for `runtime`
  ///    verdicts, the concrete extents) supports the shape, run that
  ///    family's leaf; a `required` request that fails here reports a
  ///    check failure instead of silently falling back.
  /// 2. The statically selected DispatchOwner (SmallVector /
  ///    FusedSmallBF16 / PackedABTailSplit / General).
  /// 3. The runtime tier: leaves whose applicability is `runtime`
  ///    consult the measured thresholds in RuntimeDispatch.h with the
  ///    actual m/n/k (raw small, fused small, residual split).
  /// 4. General: the configured TMUL tile traversal.
  template <::vecops::matmul::Atom Atom, typename Policy,
            bool AllowTailSplit = false,
            typename FamilyDispatch =
                ::vecops::matmul::details::AutomaticFamilyDispatch,
            execution::ExecutionScope Scope,
            meta::ValueType M, meta::ValueType N, meta::ValueType K,
            typename A, typename B, typename CInput, typename COutput>
  VECOPS_ALWAYS_INLINE static void run(
      Scope& scope, M m, N n, K k,
      const A& a, const B& b, const CInput& c_input, COutput& c_output,
      void* scratch) {
    using Applicability = ::vecops::matmul::details::Applicability;
    using RequestedFamily = typename FamilyDispatch::Family;
    constexpr bool NMajor =
        matmul_details::n_major_family_dispatch_v<FamilyDispatch>;
    constexpr auto Owner = amx::select_dispatch_owner<
        FamilyDispatch, Atom, AllowTailSplit,
        M, N, K, A, B, CInput, COutput>();
    const nint_t logical_m = static_cast<nint_t>(m);
    const nint_t logical_n = static_cast<nint_t>(n);
    const nint_t logical_k = static_cast<nint_t>(k);
    auto run_residual_split = [&]() VECOPS_INLINE_LAMBDA {
      constexpr bool PackedAB =
          amx::is_packed_access_v<
              Atom, ::vecops::matmul::Operand::A, A> &&
          amx::is_packed_access_v<
              Atom, ::vecops::matmul::Operand::B, B>;
      constexpr bool PackedB = amx::is_packed_access_v<
          Atom, ::vecops::matmul::Operand::B, B>;
      constexpr bool AlreadyConfigured = [] {
        if constexpr (requires {
                        typename std::remove_cvref_t<Scope>::ActiveConfiguration;
                      }) {
          return std::same_as<
              typename std::remove_cvref_t<Scope>::ActiveConfiguration,
              amx::Configuration>;
        } else {
          return false;
        }
      }();
      const nint_t residual_m = logical_m % 16;
      const nint_t residual_n = logical_n % 16;
      const bool split_m = residual_m != 0 && logical_m > 16 &&
          (residual_n == 0 || logical_n <= 16 || residual_m <= residual_n);
      if (split_m) {
        const meta::Any bulk_m{logical_m - residual_m};
        const meta::Any tail_m{residual_m};
        auto run_bulk = [&](auto& configured) VECOPS_INLINE_LAMBDA_NOEXCEPT {
          if constexpr (NMajor)
            run_configured_region_n_major<Atom, Policy>(
                configured, bulk_m, n, k, logical_m, logical_n, 0, 0,
                a, b, c_input, c_output, scratch);
          else
            run_configured_region<Atom, Policy>(
                configured, bulk_m, n, k, logical_m, logical_n, 0, 0,
                a, b, c_input, c_output, scratch);
        };
        auto run_tail = [&](auto& configured) VECOPS_INLINE_LAMBDA_NOEXCEPT {
          if constexpr (NMajor)
            run_configured_region_n_major<Atom, Policy>(
                configured, tail_m, n, k, logical_m, logical_n,
                logical_m - residual_m, 0,
                a, b, c_input, c_output, scratch);
          else
            run_configured_region<Atom, Policy>(
                configured, tail_m, n, k, logical_m, logical_n,
                logical_m - residual_m, 0,
                a, b, c_input, c_output, scratch);
        };
        if constexpr (PackedAB) {
          with_configuration<Atom, Policy, true>(scope, bulk_m, n, run_bulk);
          with_configuration<Atom, Policy, true>(scope, tail_m, n, run_tail);
        } else if constexpr (AlreadyConfigured) {
          run_bulk(scope);
          run_tail(scope);
        } else {
          with_configuration<Atom, Policy, PackedB>(
              scope, m, n,
              [&](auto& configured) VECOPS_INLINE_LAMBDA_NOEXCEPT {
                run_bulk(configured);
                run_tail(configured);
              });
        }
      } else {
        VECOPS_ASSERT(residual_n != 0 && logical_n > 16,
                      "AMX residual split requires a spatial tail");
        const meta::Any bulk_n{logical_n - residual_n};
        const meta::Any tail_n{residual_n};
        auto run_bulk = [&](auto& configured) VECOPS_INLINE_LAMBDA_NOEXCEPT {
          if constexpr (NMajor)
            run_configured_region_n_major<Atom, Policy>(
                configured, m, bulk_n, k, logical_m, logical_n, 0, 0,
                a, b, c_input, c_output, scratch);
          else
            run_configured_region<Atom, Policy>(
                configured, m, bulk_n, k, logical_m, logical_n, 0, 0,
                a, b, c_input, c_output, scratch);
        };
        auto run_tail = [&](auto& configured) VECOPS_INLINE_LAMBDA_NOEXCEPT {
          if constexpr (NMajor)
            run_configured_region_n_major<Atom, Policy>(
                configured, m, tail_n, k, logical_m, logical_n,
                0, logical_n - residual_n,
                a, b, c_input, c_output, scratch);
          else
            run_configured_region<Atom, Policy>(
                configured, m, tail_n, k, logical_m, logical_n,
                0, logical_n - residual_n,
                a, b, c_input, c_output, scratch);
        };
        if constexpr (PackedAB) {
          with_configuration<Atom, Policy, true>(scope, m, bulk_n, run_bulk);
          with_configuration<Atom, Policy, true>(scope, m, tail_n, run_tail);
        } else if constexpr (AlreadyConfigured) {
          run_bulk(scope);
          run_tail(scope);
        } else {
          with_configuration<Atom, Policy, PackedB>(
              scope, m, n,
              [&](auto& configured) VECOPS_INLINE_LAMBDA_NOEXCEPT {
                run_bulk(configured);
                run_tail(configured);
              });
        }
      }
    };
    constexpr bool RequestsSmallVector = std::same_as<
        RequestedFamily,
        ::vecops::matmul::kernel_family::SmallVector>;
    constexpr bool AutomaticOwnerIsSmallVector =
        Owner == amx::DispatchOwner::SmallVector ||
        Owner == amx::DispatchOwner::FusedSmallBF16;
    if constexpr (RequestsSmallVector && !AutomaticOwnerIsSmallVector) {
      constexpr auto Support = amx::family_applicability<
          ::vecops::matmul::kernel_family::SmallVector,
          Atom, AllowTailSplit, M, N, K,
          A, B, CInput, COutput>();
      const bool supported = [&]() VECOPS_INLINE_LAMBDA {
        if constexpr (Support == Applicability::never) {
          return false;
        } else if constexpr (Support == Applicability::always) {
          return true;
        } else {
          return logical_m > 0 && logical_n > 0 && logical_k >= 0 &&
              (!amx::fused_small_bf16_candidate_v<
                   Atom, A, B, CInput, COutput> ||
               runtime_dispatch_rules::area_at_most(
                   logical_m, logical_n, 16));
        }
      }();
      if (supported) {
        if constexpr (amx::small_vector_candidate_v<
                          Atom, A, B, CInput, COutput>) {
          amx::run_small_vector<Atom>(
              a, b, c_output, logical_m, logical_n, logical_k);
        } else if constexpr (amx::small_vector_accumulate_candidate_v<
                                 Atom, A, B, CInput, COutput>) {
          amx::run_small_vector_accumulating<Atom>(
              a, b, c_input, c_output,
              logical_m, logical_n, logical_k);
        } else if constexpr (amx::fused_small_bf16_candidate_v<
                                 Atom, A, B, CInput, COutput>) {
          amx::run_fused_small_bf16<Atom>(
              a, b, c_input, c_output,
              logical_m, logical_n, logical_k);
        }
        return;
      }
      if constexpr (FamilyDispatch::required) {
        VECOPS_CHECK(false,
                     "required SmallVector family rejected at runtime");
      }
    }
    constexpr bool RequestsResidualSplit = std::same_as<
        RequestedFamily,
        ::vecops::matmul::kernel_family::ResidualSplit>;
    if constexpr (RequestsResidualSplit &&
                  Owner != amx::DispatchOwner::PackedABTailSplit) {
      constexpr auto Support = amx::family_applicability<
          ::vecops::matmul::kernel_family::ResidualSplit,
          Atom, AllowTailSplit, M, N, K,
          A, B, CInput, COutput>();
      const bool supported = [&]() VECOPS_INLINE_LAMBDA {
        if constexpr (Support == Applicability::never) return false;
        else if constexpr (Support == Applicability::always) return true;
        else return logical_m > 0 && logical_n > 0 && logical_k >= 0 &&
            ((logical_m > 16 && logical_m % 16 != 0) ||
             (logical_n > 16 && logical_n % 16 != 0));
      }();
      if (supported) {
        run_residual_split();
        return;
      }
      if constexpr (FamilyDispatch::required) {
        VECOPS_CHECK(false,
                     "required ResidualSplit family rejected at runtime");
      }
    }
    if constexpr (Owner == amx::DispatchOwner::SmallVector) {
      if constexpr (amx::small_vector_candidate_v<
                        Atom, A, B, CInput, COutput>) {
        amx::run_small_vector<Atom>(
            a, b, c_output, logical_m, logical_n, logical_k);
      } else {
        static_assert(amx::small_vector_accumulate_candidate_v<
            Atom, A, B, CInput, COutput>);
        amx::run_small_vector_accumulating<Atom>(
            a, b, c_input, c_output,
            logical_m, logical_n, logical_k);
      }
      return;
    } else if constexpr (Owner == amx::DispatchOwner::FusedSmallBF16) {
      amx::run_fused_small_bf16<Atom>(
          a, b, c_input, c_output,
          logical_m, logical_n, logical_k);
      return;
    } else if constexpr (Owner == amx::DispatchOwner::PackedABTailSplit) {
      run_residual_split();
      return;
    }

    // Tier 3 gates: each leaf enters the runtime tier only when its
    // applicability verdict is `runtime` and the family request permits
    // routing there (permits_runtime_family_v keeps `require` from being
    // silently served by a different family's leaf).
    constexpr bool RuntimeRawSmall = [] {
      if constexpr (!(amx::small_vector_candidate_v<
                          Atom, A, B, CInput, COutput> ||
                      amx::small_vector_accumulate_candidate_v<
                          Atom, A, B, CInput, COutput>) ||
                    !amx::permits_runtime_family_v<
                        FamilyDispatch,
                        ::vecops::matmul::kernel_family::SmallVector>) {
        return false;
      } else {
        return amx::small_vector_shape_applicability<Atom, M, N, K>() ==
            Applicability::runtime;
      }
    }();
    constexpr bool RuntimeFusedSmall = [] {
      if constexpr (!amx::fused_small_bf16_candidate_v<
                        Atom, A, B, CInput, COutput> ||
                    !amx::permits_runtime_family_v<
                        FamilyDispatch,
                        ::vecops::matmul::kernel_family::SmallVector>) {
        return false;
      } else {
        return amx::small_area_applicability<M, N, K, 16>() ==
            Applicability::runtime;
      }
    }();
    constexpr bool RuntimeResidualSplit = [] {
      if constexpr (!AllowTailSplit ||
                    !std::same_as<
                        Atom, ::vecops::matmul::AMX_BF16F32> ||
                    !amx::is_packed_access_v<
                        Atom, ::vecops::matmul::Operand::A, A> ||
                    !amx::is_packed_access_v<
                        Atom, ::vecops::matmul::Operand::B, B> ||
                    !amx::packed_ab_tail_split_candidate_v<
                        Atom, A, B, CInput, COutput> ||
                    !amx::permits_runtime_family_v<
                        FamilyDispatch,
                        ::vecops::matmul::kernel_family::ResidualSplit>) {
        return false;
      } else {
        return amx::residual_split_shape_applicability<M, N, K>() ==
            Applicability::runtime;
      }
    }();

    if constexpr (RuntimeRawSmall) {
      if (runtime_amx_small_vector_profitable(
              amx::small_vector_shape_v<Atom>,
              logical_m, logical_n, logical_k)) {
        if constexpr (amx::small_vector_candidate_v<
                          Atom, A, B, CInput, COutput>) {
          amx::run_small_vector<Atom>(
              a, b, c_output, logical_m, logical_n, logical_k);
        } else {
          amx::run_small_vector_accumulating<Atom>(
              a, b, c_input, c_output,
              logical_m, logical_n, logical_k);
        }
        return;
      }
    } else if constexpr (RuntimeFusedSmall) {
      if (runtime_small_area_profitable(
              logical_m, logical_n, logical_k, 16)) {
        amx::run_fused_small_bf16<Atom>(
            a, b, c_input, c_output,
            logical_m, logical_n, logical_k);
        return;
      }
    }
    if constexpr (RuntimeResidualSplit) {
      if (runtime_amx_residual_split_profitable(
              logical_m, logical_n, logical_k)) {
        run_residual_split();
        return;
      }
    }

    // Nothing special served this shape and a specific family was
    // required: report the rejection rather than silently running the
    // general tile path.
    if constexpr (FamilyDispatch::required &&
                  !std::same_as<
                      RequestedFamily,
                      ::vecops::matmul::kernel_family::General> &&
                  !std::same_as<
                      RequestedFamily,
                      ::vecops::matmul::kernel_family::WholeProblem>) {
      VECOPS_CHECK(false, "required AMX matmul family rejected at runtime");
    }

    constexpr bool PackedB =
        amx::is_packed_access_v<Atom, ::vecops::matmul::Operand::B, B>;
    constexpr bool RuntimePreparedNMajor = [] {
      using MV = std::remove_cvref_t<M>;
      using NV = std::remove_cvref_t<N>;
      using KV = std::remove_cvref_t<K>;
      if constexpr (
          !matmul_details::automatic_traversal_family_dispatch_v<
              FamilyDispatch> ||
          !std::same_as<Atom, ::vecops::matmul::AMX_BF16F32> ||
          !amx::is_packed_access_v<
              Atom, ::vecops::matmul::Operand::A, A> ||
          !amx::is_packed_access_v<
              Atom, ::vecops::matmul::Operand::B, B> ||
          !NV::is_const || !KV::is_const || MV::is_const) {
        return false;
      } else {
        // Restrict the dual traversal to the measured L2-resident-K class.
        // Long K needs real KC blocking (L07), while constant M was already
        // decided by automatic_physical_n_major_v before reaching this tier.
        return KV::value >= 256 && KV::value <= 4096 &&
            NV::value >= 4 * 16;
      }
    }();
    const bool runtime_prepared_n_major = [&] {
      if constexpr (!RuntimePreparedNMajor) {
        return false;
      } else {
        constexpr nint_t MaxAElements = (1024 * 1024) /
            static_cast<nint_t>(sizeof(typename Atom::TA));
        return logical_m > 0 && logical_n > 0 && logical_k > 0 &&
            logical_m <= logical_n / 4 &&
            logical_k <= MaxAElements &&
            logical_m <= MaxAElements / logical_k;
      }
    }();
    constexpr bool HasActiveConfiguration = [] {
      if constexpr (requires {
                      typename std::remove_cvref_t<Scope>::ActiveConfiguration;
                    }) {
        return std::same_as<
            typename std::remove_cvref_t<Scope>::ActiveConfiguration,
            amx::Configuration>;
      } else {
        return false;
      }
    }();
    auto run_general = [&](auto& configured)
        VECOPS_INLINE_LAMBDA_NOEXCEPT {
      if constexpr (NMajor) {
        run_configured_n_major<Atom, Policy>(
            configured, m, n, k, a, b,
            c_input, c_output, scratch);
      } else if constexpr (RuntimePreparedNMajor) {
        if (runtime_prepared_n_major) {
          run_configured_n_major<Atom, Policy>(
              configured, m, n, k, a, b,
              c_input, c_output, scratch);
        } else {
          run_configured<Atom, Policy>(
              configured, m, n, k, a, b,
              c_input, c_output, scratch);
        }
      } else {
        run_configured<Atom, Policy>(
            configured, m, n, k, a, b,
            c_input, c_output, scratch);
      }
    };
    if constexpr (HasActiveConfiguration) {
      // Batch/flatten planners may own one compatible TILECFG around many
      // leaves.  Family selection above still runs for every leaf; only the
      // General fallback reuses the active image here.
      run_general(scope);
    } else {
      with_configuration<Atom, Policy, PackedB>(
          scope, m, n, [&](auto& configured)
              VECOPS_INLINE_LAMBDA_NOEXCEPT {
            run_general(configured);
          });
    }
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

  /// Resolve the compile-time kernel plan for a leaf: KGuaranteed from
  /// the K contract's alignment, StreamB from the packed-B working-set
  /// estimate (a 48 KiB L1 budget — packed B larger than that pays for
  /// non-temporal loads).
  template <::vecops::matmul::Atom Atom, typename A, typename B,
            meta::ValueType M, meta::ValueType N, meta::ValueType K,
            typename Fn>
  VECOPS_ALWAYS_INLINE static void dispatch_plan(M, N, K, Fn&& fn) {
    constexpr nint_t KR = decltype(Atom::K_R)::value;
    using NV = std::remove_cvref_t<N>;
    using KV = std::remove_cvref_t<K>;
    constexpr bool StreamB = [] {
      if constexpr (
          amx::is_packed_access_v<Atom, ::vecops::matmul::Operand::B, B> &&
          meta::lower_bound_at_least_v<NV, 0> &&
          meta::lower_bound_at_least_v<KV, 1>) {
        constexpr nint_t LogicalN = meta::lower_bound_v<NV>;
        constexpr nint_t LogicalK = meta::lower_bound_v<KV>;
        constexpr nint_t L1Elements =
            (48 * 1024) / sizeof(typename Atom::TB);
        return LogicalK >= L1Elements ||
            LogicalN > (L1Elements - 1) / LogicalK;
      } else {
        return false;
      }
    }();
    using Plan = amx::KernelPlan<KV::aligns(KR), StreamB>;
    std::forward<Fn>(fn).template operator()<Plan>();
  }

  template <::vecops::matmul::Atom Atom, typename A, typename B,
            meta::ValueType M, meta::ValueType N, meta::ValueType K,
            typename Fn>
  VECOPS_ALWAYS_INLINE static void dispatch_plan_n_major(M, N, K, Fn&& fn) {
    constexpr nint_t KR = decltype(Atom::K_R)::value;
    using KV = std::remove_cvref_t<K>;
    // N-major immediately reuses one B tile across M. Non-temporal packed-B
    // loads would evict exactly that hot panel, so disable StreamB here.
    using Plan = amx::KernelPlan<KV::aligns(KR), false>;
    std::forward<Fn>(fn).template operator()<Plan>();
  }

  template <::vecops::matmul::Atom Atom, typename Case, typename Plan,
            typename A, typename B, typename CInput, typename COutput>
  VECOPS_ALWAYS_INLINE static void run_case(
      const A& a, const B& b, const CInput& c_input, COutput& c_output,
      nint_t logical_m, nint_t logical_n, nint_t logical_k,
      nint_t m, nint_t n, nint_t active_m, nint_t active_n,
      void* scratch) {
    if constexpr (Case::exact_blocks) {
      amx::exact_microkernel<Atom, Case, Plan>(
          a, b, c_input, c_output,
          logical_m, logical_n, logical_k,
          m, n, active_m, active_n, scratch);
    } else {
      amx::microkernel<Atom, Case, Plan>(
          a, b, c_input, c_output,
          logical_m, logical_n, logical_k,
          m, n, active_m, active_n, scratch);
    }
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
      void* scratch) {
    if constexpr (Case::exact_blocks) {
      amx::exact_microkernel_phased<Atom, Case, Plan>(
          a, b, c_input, c_output, acc_input, acc_output, route,
          logical_m, logical_n, logical_k,
          m, n, active_m, active_n, scratch);
    } else {
      amx::microkernel_routed<Atom, Case, Plan>(
          a, b, c_input, c_output, acc_input, acc_output, route,
          logical_m, logical_n, logical_k,
          m, n, active_m, active_n, scratch);
    }
  }
};

} // namespace vecops::kernel::matmul_details

#endif // VECOPS_MATMUL_DETAILS_AMX_BACKEND_H
