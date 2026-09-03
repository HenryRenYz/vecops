//
// Copyright (c) vecops contributors.
//

#ifndef VECOPS_MATMUL_CONFIG_H
#define VECOPS_MATMUL_CONFIG_H

#include <concepts>
#include <type_traits>

#include "vecops/Meta.h"
#include "vecops/kernel/Tile2D.h"
#include "vecops/matmul/Atom.h"
#include "vecops/matmul/Family.h"
#include "vecops/matmul/details/kernel/Types.h"

/**
 * @file vecops/matmul/Config.h
 * @brief Tuning-parameter vocabulary for matmul: axes, loop order, cache
 *        tiling, packing policy, and accumulator buffering.
 *
 * This header collects the user-facing knobs with which a matmul
 * configuration is shaped. Everything here is a plain type or enum spelled
 * as template arguments; none of it executes code on its own. The knobs are
 * consumed through `GenericTiledTuning`, which the public entry point
 * `ops::MatmulConfig` accepts as its tuning argument.
 *
 * The vocabulary deliberately splits along two axes of control:
 *
 * - **What the generic cache-tiled tiler does**: `CacheTiling`,
 *   `LoopOrder`/`loop_order::*`, `PackingPolicy` (per operand), and
 *   `AccBufferMode` all parameterize the generic MC/NC/KC loop nest.
 * - **How each knob's `automatic` default resolves**: at compile time from
 *   surrounding context (loop order, Meta bounds) or at run time from
 *   platform cache information (see `Tiling.h`).
 *
 * ## Key components
 *
 * | Component          | Purpose                                              |
 * |--------------------|------------------------------------------------------|
 * | `Axis`             | M/N/K of the logical product `C[M,N]=A[M,K]*B[N,K]^T`|
 * | `CacheLoopMode`    | Whether a cache-blocking loop is emitted per axis    |
 * | `PackingMode`      | Whether an operand is repacked into workspace        |
 * | `PackingExtent`    | K range covered by one packed copy (per cache block vs whole K) |
 * | `AccBufferMode`    | Where accumulators live (workspace vs output tensor) |
 * | `PackingPolicy`    | Bundles `PackingMode` + `PackingExtent` for one operand |
 * | `LoopOrder` + `loop_order::*` | The M/N/K traversal order               |
 * | `AutomaticCacheTiling`, `CacheTiling` | Runtime-resolved vs explicit MC/NC/KC |
 * | `GenericTiledTuning` | The aggregate passed to `ops::MatmulConfig`       |
 * | `LoopOrderType`    | Concept satisfied by every loop-order spelling       |
 *
 * ## Usage overview
 *
 * @code
 * #include "vecops/matmul/Config.h"
 *
 * using namespace vecops::matmul;
 *
 * using Tuning = GenericTiledTuning<
 *     // MC/NC fixed at compile time, KC a runtime multiple of 64.
 *     CacheTiling<meta::Const<64>, meta::Const<256>, meta::Dynamic<64>>,
 *     loop_order::NKM,
 *     PackingPolicy<PackingMode::never>,   // A: never repack
 *     PackingPolicy<>,                     // B: automatic
 *     AccBufferMode::automatic>;
 *
 * // ops::MatmulConfig<matmul::SME_F32F32, ..., Tuning> config;
 * @endcode
 *
 * ## Pitfalls
 *
 * - These are **generic-tiled-family-local** parameters: setting any of them
 *   never disables the whole-problem architecture kernels (see the warning
 *   on `GenericTiledTuning`).
 * - `CacheTiling` dimensions are Meta `ValueType`s. `Const<N>` fixes a tile
 *   at compile time; `Dynamic<A>` keeps the value at runtime while promising
 *   alignment `A`. Mixing them up only degrades (never breaks) the loop nest.
 * - `AutomaticCacheTiling` is resolved at **run time** from the platform
 *   cache hierarchy (`matmul::DefaultTilingPolicy` in `Tiling.h`), not from
 *   the problem shape.
 * - A `LoopOrder` must contain M, N, and K exactly once (static_assert).
 */

namespace vecops::matmul {

/**
 * @brief One axis of the logical product `C[M,N] = A[M,K] * B[N,K]^T`.
 *
 * Axis names always refer to the logical matrices, not to any packed or
 * transposed physical layout the backends may use internally.
 */
enum class Axis { M, N, K };

/**
 * @brief Controls whether the cache-blocking loop for one axis is emitted.
 *
 * The generic tiler wraps the kernel in an M/N/K loop nest driven by the
 * cache tiles. This mode decides, per axis, whether that axis gets a real
 * loop or is processed as a single block.
 *
 * - `automatic`: emit the loop only when the compile-time bounds prove it
 *   can iterate more than once (the axis extent's upper bound exceeds the
 *   tile's lower bound); otherwise fall through to a single block. This is
 *   the default and never regresses a provably-single-iteration axis.
 * - `enabled`: always emit the loop, even for a provably single block.
 * - `disabled`: never emit the loop for this axis; the whole extent is
 *   handed to the kernel as one block regardless of the cache tile.
 */
enum class CacheLoopMode {
  automatic,
  enabled,
  disabled,
};

/**
 * @brief Controls whether an operand is repacked into the packed input
 *        format demanded by the atom's matrix instruction.
 *
 * - `automatic` (default): pack only when the operand's innermost (K)
 *   stride is not known to be 1 at compile time — i.e. the input is not
 *   K-contiguous and packing is what makes it hardware-friendly. Already
 *   packed inputs are never repacked, under any mode.
 * - `always`: always pack the operand into execution workspace before the
 *   kernel, even when the original layout could be consumed directly.
 * - `never`: consume the operand in its given layout; never allocate a
 *   packed copy.
 */
enum class PackingMode {
  automatic,
  always,
  never,
};

/**
 * @brief K range covered by one packed copy of an operand.
 *
 * - `cache_k`: pack per cache-K block. The packed copy lives only for the
 *   current KC iteration and is rebuilt for the next K block, so the
 *   workspace stays small but packing work repeats per K step.
 * - `full_k`: pack the operand's full K extent once and reuse the packed
 *   copy across all K iterations (hoisted out of the K loop).
 * - `automatic` (default): `cache_k` when the operand's spatial axis
 *   (M for A, N for B) is traversed before K in the loop order — the
 *   operand would otherwise be re-read many times — and `full_k`
 *   otherwise.
 */
enum class PackingExtent {
  automatic,
  cache_k,
  full_k,
};

/**
 * @brief Controls where the FP accumulator tiles are buffered.
 *
 * - `workspace`: accumulate into dedicated execution workspace, then write
 *   the output once. Always applicable.
 * - `output`: reuse the output tensor's own storage as accumulator storage.
 *   Requires the output element to be at least as large as the atom's
 *   `TAcc`, and the output data alignment and strides must be able to
 *   represent `TAcc` elements (asserted).
 * - `automatic` (default): reuse the output only when it is a perfect
 *   stand-in — output element type equals `TAcc` and no output transform is
 *   attached; otherwise fall back to workspace.
 */
enum class AccBufferMode {
  automatic,
  workspace,
  output,
};

/**
 * @brief Packing decision for one operand (A and B each carry their own).
 *
 * Bundles the "should we pack" switch (`PackingMode`) with the "how much K
 * does one packed copy cover" switch (`PackingExtent`) so a single type
 * parameterizes both.
 *
 * @code
 * using namespace vecops::matmul;
 * // Pack B once for the whole K range, never pack A.
 * using ATuning = PackingPolicy<PackingMode::never>;
 * using BTuning = PackingPolicy<PackingMode::always, PackingExtent::full_k>;
 * @endcode
 *
 * @tparam Mode    Whether/when the operand is repacked. Default `automatic`.
 * @tparam Extent  K range covered by one packed copy. Default `automatic`.
 */
template <PackingMode Mode = PackingMode::automatic,
          PackingExtent Extent = PackingExtent::automatic>
struct PackingPolicy {
  static constexpr PackingMode mode = Mode;
  static constexpr PackingExtent extent = Extent;
};

/**
 * @brief The M/N/K traversal order of the generic cache-tiled loop nest.
 *
 * `first` is the outermost axis and `third` the innermost. The order
 * interacts with `PackingExtent` (operands whose spatial axis precedes K
 * default to per-K-block packing) and with how many times each operand is
 * re-read from memory.
 *
 * Use the ready-made aliases in `loop_order` instead of spelling
 * `LoopOrder<...>` directly.
 *
 * @code
 * using namespace vecops::matmul;
 * static_assert(LoopOrderType<loop_order::NKM>);
 * @endcode
 *
 * @tparam First   Outermost axis.
 * @tparam Second  Middle axis.
 * @tparam Third   Innermost axis.
 *
 * @note The three axes must be M, N, and K exactly once each
 *       (static_assert).
 */
template <Axis First, Axis Second, Axis Third>
struct LoopOrder {
  static_assert(First != Second && First != Third && Second != Third,
                "matmul loop order must contain M, N, and K exactly once");
  static constexpr Axis first = First;
  static constexpr Axis second = Second;
  static constexpr Axis third = Third;
};

/**
 * @brief Ready-made `LoopOrder` aliases plus the automatic marker.
 *
 * Each alias name is the three traversal axes in outermost-to-innermost
 * order: `loop_order::NKM` traverses N outermost, then K, then M innermost.
 */
namespace loop_order {

/// Placeholder telling the tiler to use the backend's default traversal
/// (resolved by `resolved_loop_order_t` in `Tiling.h`).
struct Automatic {};
/// Traverse M, then N, with K innermost.
using MNK = LoopOrder<Axis::M, Axis::N, Axis::K>;
/// Traverse M, then K, with N innermost.
using MKN = LoopOrder<Axis::M, Axis::K, Axis::N>;
/// Traverse N, then M, with K innermost.
using NMK = LoopOrder<Axis::N, Axis::M, Axis::K>;
/// Traverse N, then K, with M innermost.
using NKM = LoopOrder<Axis::N, Axis::K, Axis::M>;
/// Traverse K, then M, with N innermost.
using KMN = LoopOrder<Axis::K, Axis::M, Axis::N>;
/// Traverse K, then N, with M innermost.
using KNM = LoopOrder<Axis::K, Axis::N, Axis::M>;

} // namespace loop_order

/**
 * @brief Placeholder selecting runtime, cache-hierarchy-driven cache tiling.
 *
 * Passing this (the default) as `GenericTiledTuning::CacheTiling` makes the
 * tiler resolve MC/NC/KC at run time via `DefaultTilingPolicy<Atom>` in
 * `Tiling.h`, based on the platform's L1/L2 sizes and the atom's native
 * extents. Any explicit `CacheTiling` instead fixes the tiles statically.
 */
struct AutomaticCacheTiling {};

/**
 * @brief Explicit MC/NC/KC cache tile sizes, one per axis of the logical
 *        product.
 *
 * Each dimension is a Meta `ValueType`, which selects where the tile size
 * lives:
 *
 * - `meta::Const<N>` — fixed at compile time; the loop nest can specialize
 *   on it and the dimension costs no storage.
 * - `meta::Dynamic<A>` (including `meta::Any`) — a runtime value that
 *   additionally promises divisibility by `A`; the tiler keeps the
 *   alignment information for the kernel.
 *
 * @code
 * using namespace vecops::matmul;
 * // MC/NC compile-time constants; KC runtime, multiple of 64.
 * CacheTiling<meta::Const<128>, meta::Const<256>, meta::Dynamic<64>> tiling;
 * @endcode
 *
 * @tparam MC     M-axis tile size ValueType.
 * @tparam NC     N-axis tile size ValueType.
 * @tparam KC     K-axis tile size ValueType.
 * @tparam MMode  Cache-loop emission for the M axis (default `automatic`).
 * @tparam NMode  Cache-loop emission for the N axis (default `automatic`).
 * @tparam KMode  Cache-loop emission for the K axis (default `automatic`).
 *
 * @note The runtime values are stored in `[[no_unique_address]]` members, so
 *       an all-`Const` tiling is an empty type and costs nothing inside the
 *       config object.
 * @note Tile sizes must stay positive; the loop nest asserts this at run
 *       time. For hardware-friendly traversal prefer tiles that are
 *       multiples of the atom's native extents, as produced by
 *       `AutomaticCacheTilingFor` in `Tiling.h`.
 */
template <meta::ValueType MC, meta::ValueType NC, meta::ValueType KC,
          CacheLoopMode MMode = CacheLoopMode::automatic,
          CacheLoopMode NMode = CacheLoopMode::automatic,
          CacheLoopMode KMode = CacheLoopMode::automatic>
struct CacheTiling {
  using MTile = MC;
  using NTile = NC;
  using KTile = KC;
  static constexpr CacheLoopMode m_mode = MMode;
  static constexpr CacheLoopMode n_mode = NMode;
  static constexpr CacheLoopMode k_mode = KMode;

  // [[no_unique_address]]: Const dimensions are empty Value types, so an
  // all-compile-time tiling stores nothing; only runtime dimensions occupy
  // the config object.
  [[no_unique_address]] MC mc;
  [[no_unique_address]] NC nc;
  [[no_unique_address]] KC kc;

  constexpr CacheTiling() : mc{}, nc{}, kc{} {}
  constexpr CacheTiling(MC m, NC n, KC k)
      : mc(m), nc(n), kc(k) {}
};

/**
 * @brief Parameters understood only by the generic cache-tiled family.
 *
 * Keeping these knobs in a family-scoped object prevents an explicit MC/NC/KC,
 * loop order, packing choice, or accumulator policy from implicitly disabling
 * whole-problem architecture kernels.
 *
 * @warning These parameters configure the generic tiler only. They are not
 *          read by the whole-problem architecture kernels, and — by design
 *          — setting any of them never changes which kernel family the
 *          planner selects. Do not promote them to whole-config knobs.
 *
 * @code
 * using namespace vecops::matmul;
 * GenericTiledTuning<AutomaticCacheTiling, loop_order::Automatic> tuning;
 * @endcode
 *
 * @tparam CacheTilingT  Explicit `CacheTiling` or `AutomaticCacheTiling`
 *                       (default: automatic).
 * @tparam LoopOrderT    A `LoopOrderType` or `loop_order::Automatic`
 *                       (default: automatic).
 * @tparam APackingT     `PackingPolicy` for operand A (default: automatic).
 * @tparam BPackingT     `PackingPolicy` for operand B (default: automatic).
 * @tparam AccMode       Accumulator buffering choice (default: automatic).
 */
template <
    typename CacheTilingT = AutomaticCacheTiling,
    typename LoopOrderT = loop_order::Automatic,
    typename APackingT = PackingPolicy<>,
    typename BPackingT = PackingPolicy<>,
    AccBufferMode AccMode = AccBufferMode::automatic>
struct GenericTiledTuning {
  using CacheTiling = CacheTilingT;
  using LoopOrder = LoopOrderT;
  using APacking = APackingT;
  using BPacking = BPackingT;
  static constexpr AccBufferMode acc_buffer_mode = AccMode;

  [[no_unique_address]] CacheTilingT cache_tiling{};
};

/// Concept satisfied by every loop-order spelling: exposes `first`,
/// `second`, and `third` axis members. `loop_order::Automatic` deliberately
/// does not satisfy it.
template <typename T>
concept LoopOrderType = requires {
  { T::first } -> std::convertible_to<Axis>;
  { T::second } -> std::convertible_to<Axis>;
  { T::third } -> std::convertible_to<Axis>;
};

} // namespace vecops::matmul

#endif // VECOPS_MATMUL_CONFIG_H
