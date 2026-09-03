//
// Copyright (c) vecops contributors.
//

#ifndef VECOPS_MATMUL_DETAILS_LOOP_NEST_H
#define VECOPS_MATMUL_DETAILS_LOOP_NEST_H

#include <algorithm>
#include <type_traits>
#include <utility>

#include "vecops/Assertion.h"
#include "vecops/Meta.h"
#include "vecops/matmul/Config.h"

/**
 * @file vecops/matmul/details/tiled/LoopNest.h
 * @brief The MC/NC/KC cache-blocking loop nest of the generic tiled matmul.
 *
 * `LoopNest<Order, Tiling>::run_phased` recursively expands the three
 * cache-blocking loops in the configured traversal order via the
 * `run_depth<Depth>` template: each recursion level handles one axis
 * (chosen by the order) and `Depth == 3` is the leaf, so the whole nest is
 * unrolled into per-axis code at compile time with no loop-order dispatch at
 * run time.
 *
 * The K axis is special: it is the only axis whose blocks accumulate into
 * the same output (split-K), so the K recursion additionally tracks a
 * `KernelPhase` -- first/middle/last K block -- and threads it down to the
 * leaf. M and N blocks are independent of each other and need no phase; the
 * phase type they pass through unchanged. The leaf callback receives one
 * `BlockContext` (block origins + per-axis Meta extents) plus the phase.
 *
 * Whether a given axis emits an actual loop at all is decided per axis by
 * `generates_loop_v` from the `CacheLoopMode` and Meta bounds.
 */

namespace vecops::matmul::details {

/// Per-leaf block description: origin of the block on each logical axis and
/// the block extents as Meta values (compile-time knowledge survives).
template <typename M, typename N, typename K>
struct BlockContext {
  nint_t m_origin = 0;
  nint_t n_origin = 0;
  nint_t k_origin = 0;
  [[no_unique_address]] M m;
  [[no_unique_address]] N n;
  [[no_unique_address]] K k;
};

/// Which K-block of the split-K accumulation this leaf performs: `first_k`
/// initializes the accumulator, `last_k` writes it back to the output.
/// Both true means a single unsplit K block (C -> C directly).
template <bool FirstK, bool LastK>
struct KernelPhase {
  static constexpr bool first_k = FirstK;
  static constexpr bool last_k = LastK;
};

/// Phase carried while no K loop has been generated yet; collapses to
/// KernelPhase<true, true> at the leaf (a single K block is trivially both
/// first and last).
struct UnselectedKPhase {};

/**
 * @brief Whether one axis emits a real cache-blocking loop.
 *
 * `disabled`/`enabled` are literal. The `automatic` branch uses Meta interval
 * reasoning, and its comparison direction is easy to read backwards: a loop
 * can only be *provably useless* when even the extent's upper bound fits
 * within the tile's lower bound -- `upper_bound(extent) > lower_bound(tile)`
 * therefore emits the loop whenever more than one iteration is not ruled
 * out, i.e. it is conservative in emitting. When either bound is unknown the
 * loop is emitted (default to the general case).
 */
template <typename Extent, typename Tile, CacheLoopMode Mode>
inline constexpr bool generates_loop_v = [] {
  if constexpr (Mode == CacheLoopMode::disabled) {
    return false;
  } else if constexpr (Mode == CacheLoopMode::enabled) {
    return true;
  } else {
    using E = std::remove_cvref_t<Extent>;
    using T = std::remove_cvref_t<Tile>;
    if constexpr (meta::has_upper_bound_v<E> &&
                  meta::has_lower_bound_v<T>) {
      return meta::upper_bound_v<E> > meta::lower_bound_v<T>;
    } else {
      return true;
    }
  }
}();

/// Replace one axis of a BlockContext with a new origin/extent, keeping the
/// other two axes (and their compile-time types) intact.
template <Axis Target, typename Context, typename Active>
VECOPS_ALWAYS_INLINE auto replace_axis(
    const Context& context, nint_t origin, Active active) {
  if constexpr (Target == Axis::M) {
    return BlockContext<Active, decltype(context.n), decltype(context.k)>{
        origin, context.n_origin, context.k_origin,
        active, context.n, context.k};
  } else if constexpr (Target == Axis::N) {
    return BlockContext<decltype(context.m), Active, decltype(context.k)>{
        context.m_origin, origin, context.k_origin,
        context.m, active, context.k};
  } else {
    return BlockContext<decltype(context.m), decltype(context.n), Active>{
        context.m_origin, context.n_origin, origin,
        context.m, context.n, active};
  }
}

/// Axis-to-value accessors: extent from the BlockContext, tile and cache
/// loop mode from the CacheTiling, and the axis a recursion depth handles.
template <Axis Target, typename Context>
VECOPS_ALWAYS_INLINE decltype(auto) axis_extent(const Context& context) {
  if constexpr (Target == Axis::M) return (context.m);
  else if constexpr (Target == Axis::N) return (context.n);
  else return (context.k);
}

template <Axis Target, typename Tiling>
VECOPS_ALWAYS_INLINE decltype(auto) axis_tile(const Tiling& tiling) {
  if constexpr (Target == Axis::M) return (tiling.mc);
  else if constexpr (Target == Axis::N) return (tiling.nc);
  else return (tiling.kc);
}

template <Axis Target, typename Tiling>
inline constexpr CacheLoopMode axis_mode_v = [] {
  if constexpr (Target == Axis::M) return Tiling::m_mode;
  else if constexpr (Target == Axis::N) return Tiling::n_mode;
  else return Tiling::k_mode;
}();

template <int Depth, typename Order>
inline constexpr Axis order_axis_v = [] {
  static_assert(0 <= Depth && Depth < 3);
  if constexpr (Depth == 0) return Order::first;
  else if constexpr (Depth == 1) return Order::second;
  else return Order::third;
}();

/// The cache-blocking loop nest for one traversal order and tiling.
/// See the file header for the recursive expansion and K-phase contract.
template <typename Order, typename Tiling>
struct LoopNest {
  static_assert(LoopOrderType<Order>);

  /// Whether the loop for `Target` is a real loop for the given problem
  /// extents (binds generates_loop_v to this nest's tiling modes).
  template <Axis Target, typename M, typename N, typename K>
  static constexpr bool generates_loop = [] {
    using Extent = std::conditional_t<
        Target == Axis::M, M,
        std::conditional_t<Target == Axis::N, N, K>>;
    using Tile = std::conditional_t<
        Target == Axis::M, typename Tiling::MTile,
        std::conditional_t<
            Target == Axis::N, typename Tiling::NTile,
            typename Tiling::KTile>>;
    return generates_loop_v<
        Extent, Tile, axis_mode_v<Target, Tiling>>;
  }();

  /// Whole-problem entry for callers that do not care about K phases: the
  /// leaf's (block, phase) pair is adapted back to (block, first, last).
  template <typename M, typename N, typename K, typename Fn>
  VECOPS_ALWAYS_INLINE static void run(
      M m, N n, K k, const Tiling& tiling, Fn&& fn) {
    auto adapter = [&](const auto& block, auto phase)
        VECOPS_INLINE_LAMBDA {
      fn(block, decltype(phase)::first_k, decltype(phase)::last_k);
    };
    run_phased(m, n, k, tiling, adapter);
  }

  /// Phase-carrying entry: drives `run_depth` from the outermost axis.
  /// Zero extents return without invoking the leaf (an empty product).
  template <typename M, typename N, typename K, typename Fn>
  VECOPS_ALWAYS_INLINE static void run_phased(
      M m, N n, K k, const Tiling& tiling, Fn&& fn) {
    static_assert(meta::ValueType<M> && meta::ValueType<N> &&
                  meta::ValueType<K>);
    const nint_t logical_m = static_cast<nint_t>(m);
    const nint_t logical_n = static_cast<nint_t>(n);
    const nint_t logical_k = static_cast<nint_t>(k);
    VECOPS_ASSERT(logical_m >= 0 && logical_n >= 0 && logical_k >= 0,
                  "matmul extents must be non-negative");
    if (logical_m == 0 || logical_n == 0 || logical_k == 0) return;
    BlockContext<M, N, K> context{0, 0, 0, m, n, k};
    auto&& fn_ref = fn;
    run_depth<0, UnselectedKPhase>(
        context, tiling, logical_k, fn_ref);
  }

private:
  /// One recursion level per axis. Depth == 3 is the leaf, where an
  /// unselected phase (no K loop was ever generated) converges to
  /// KernelPhase<true, true>: the K axis, if elided entirely, is a single
  /// block by construction. Otherwise this level emits the loop for
  /// `order_axis_v<Depth, Order>` and recurses with narrowed blocks.
  template <int Depth, typename Phase, typename Context, typename Fn>
  VECOPS_ALWAYS_INLINE static void run_depth(
      const Context& context, const Tiling& tiling,
      nint_t logical_k, Fn& fn) {
    if constexpr (Depth == 3) {
      using EffectivePhase = std::conditional_t<
          std::same_as<Phase, UnselectedKPhase>,
          KernelPhase<true, true>, Phase>;
      fn(context, EffectivePhase{});
    } else {
      constexpr Axis Target = order_axis_v<Depth, Order>;
      using Extent = std::remove_cvref_t<decltype(axis_extent<Target>(
          context))>;
      using Tile = std::remove_cvref_t<decltype(axis_tile<Target>(tiling))>;
      constexpr CacheLoopMode Mode = axis_mode_v<Target, Tiling>;
      if constexpr (!generates_loop_v<Extent, Tile, Mode>) {
        // Loop provably (or by request) unnecessary: the whole axis is one
        // block; recurse with the context unchanged.
        run_depth<Depth + 1, Phase>(context, tiling, logical_k, fn);
      } else {
        const nint_t extent = static_cast<nint_t>(
            axis_extent<Target>(context));
        const nint_t tile = static_cast<nint_t>(axis_tile<Target>(tiling));
        VECOPS_ASSERT(tile > 0, "matmul cache tile must be positive");
        if constexpr (Target == Axis::K) {
          // K is the only axis needing a phase split: it is the only axis
          // whose blocks accumulate into the same output. Blocks are split
          // into first / middle(s) / last rather than a uniform loop so the
          // first block carries a compile-time full-tile extent (Any{tile})
          // and only the last block is dynamically sized.
          if (extent <= tile) {
            const meta::Any active{extent};
            auto block = replace_axis<Target>(context, 0, active);
            run_depth<Depth + 1, KernelPhase<true, true>>(
                block, tiling, logical_k, fn);
          } else {
            auto first = replace_axis<Target>(
                context, 0, meta::Any{tile});
            run_depth<Depth + 1, KernelPhase<true, false>>(
                first, tiling, logical_k, fn);
            // Align the last block's origin down to a tile boundary
            // (((extent-1)/tile)*tile) so the tail is strictly smaller than
            // one tile and never degenerates to a full tile plus a sliver.
            const nint_t last_origin = ((extent - 1) / tile) * tile;
            for (nint_t origin = tile; origin < last_origin;
                 origin += tile) {
              auto middle = replace_axis<Target>(
                  context, origin, meta::Any{tile});
              run_depth<Depth + 1, KernelPhase<false, false>>(
                  middle, tiling, logical_k, fn);
            }
            auto last = replace_axis<Target>(
                context, last_origin,
                meta::Any{extent - last_origin});
            run_depth<Depth + 1, KernelPhase<false, true>>(
                last, tiling, logical_k, fn);
          }
        } else {
          // M/N blocks are independent: uniform stepping with an inline
          // min(tile, remainder) tail, no phase.
          for (nint_t origin = 0; origin < extent; origin += tile) {
            const meta::Any active{std::min(tile, extent - origin)};
            auto block = replace_axis<Target>(context, origin, active);
            run_depth<Depth + 1, Phase>(
                block, tiling, logical_k, fn);
          }
        }
      }
    }
  }
};

} // namespace vecops::matmul::details

#endif // VECOPS_MATMUL_DETAILS_LOOP_NEST_H
