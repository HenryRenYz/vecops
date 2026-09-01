//
// Copyright (c) vecops contributors.
//

#ifndef VECOPS_MATMUL_CONFIG_H
#define VECOPS_MATMUL_CONFIG_H

#include <concepts>
#include <string_view>
#include <type_traits>

#include "vecops/Meta.h"
#include "vecops/kernel/Tile2D.h"
#include "vecops/matmul/Atom.h"
#include "vecops/matmul/details/Types.h"
#include "vecops/platform/CacheInfo.h"

namespace vecops::matmul {

enum class Axis { M, N, K };

enum class CacheLoopMode {
  automatic,
  enabled,
  disabled,
};

enum class PackingMode {
  automatic,
  always,
  never,
};

enum class PackingExtent {
  automatic,
  cache_k,
  full_k,
};

enum class AccBufferMode {
  automatic,
  workspace,
  output,
};

namespace kernel_family {

/** Existing whole-problem planner, including architecture-specialized leaves. */
struct WholeProblem {};

/** General MC/NC/KC cache tiler feeding the backend Tile2D scheduler. */
struct GenericTiled {};

template <typename T>
concept Family = std::same_as<T, WholeProblem> ||
    std::same_as<T, GenericTiled>;

template <Family FamilyT>
struct Info;

template <>
struct Info<WholeProblem> {
  static constexpr std::string_view name = "whole_problem";
  static constexpr bool composite = true;
};

template <>
struct Info<GenericTiled> {
  static constexpr std::string_view name = "generic_tiled";
  static constexpr bool composite = false;
};

} // namespace kernel_family

namespace family_selection {

enum class Mode {
  automatic,
  prefer,
  require,
};

/** Let the library choose among every family made available by the backend. */
struct Automatic {
  static constexpr Mode mode = Mode::automatic;
};

/** Prefer one family, while allowing a future planner to fall back. */
template <typename FamilyT>
struct Prefer {
  static_assert(kernel_family::Family<FamilyT>,
                "unknown matmul kernel family");
  using Family = FamilyT;
  static constexpr Mode mode = Mode::prefer;
};

/** Require one family; an inapplicable family is a configuration error. */
template <typename FamilyT>
struct Require {
  static_assert(kernel_family::Family<FamilyT>,
                "unknown matmul kernel family");
  using Family = FamilyT;
  static constexpr Mode mode = Mode::require;
};

} // namespace family_selection

template <PackingMode Mode = PackingMode::automatic,
          PackingExtent Extent = PackingExtent::automatic>
struct PackingPolicy {
  static constexpr PackingMode mode = Mode;
  static constexpr PackingExtent extent = Extent;
};

template <Axis First, Axis Second, Axis Third>
struct LoopOrder {
  static_assert(First != Second && First != Third && Second != Third,
                "matmul loop order must contain M, N, and K exactly once");
  static constexpr Axis first = First;
  static constexpr Axis second = Second;
  static constexpr Axis third = Third;
};

namespace loop_order {

struct Automatic {};
using MNK = LoopOrder<Axis::M, Axis::N, Axis::K>;
using MKN = LoopOrder<Axis::M, Axis::K, Axis::N>;
using NMK = LoopOrder<Axis::N, Axis::M, Axis::K>;
using NKM = LoopOrder<Axis::N, Axis::K, Axis::M>;
using KMN = LoopOrder<Axis::K, Axis::M, Axis::N>;
using KNM = LoopOrder<Axis::K, Axis::N, Axis::M>;

} // namespace loop_order

struct AutomaticCacheTiling {};

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

  [[no_unique_address]] MC mc;
  [[no_unique_address]] NC nc;
  [[no_unique_address]] KC kc;

  constexpr CacheTiling() : mc{}, nc{}, kc{} {}
  constexpr CacheTiling(MC m, NC n, KC k)
      : mc(m), nc(n), kc(k) {}
};

/**
 * Parameters understood only by the generic cache-tiled family.
 *
 * Keeping these knobs in a family-scoped object prevents an explicit MC/NC/KC,
 * loop order, packing choice, or accumulator policy from implicitly disabling
 * whole-problem architecture kernels.
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

template <typename T>
concept LoopOrderType = requires {
  { T::first } -> std::convertible_to<Axis>;
  { T::second } -> std::convertible_to<Axis>;
  { T::third } -> std::convertible_to<Axis>;
};

} // namespace vecops::matmul

namespace vecops::ops {

template <
    ::vecops::matmul::Atom AtomT,
    typename FamilySelectionT =
        ::vecops::matmul::family_selection::Automatic,
    typename SchedulerPolicyT = kernel::matmul_policy::Automatic,
    typename GenericTiledTuningT = ::vecops::matmul::GenericTiledTuning<>,
    typename CacheInfoProviderT = platform::SystemCacheInfoProvider>
struct MatmulConfig {
  using Atom = AtomT;
  using FamilySelection = FamilySelectionT;
  using SchedulerPolicy = SchedulerPolicyT;
  using GenericTuning = GenericTiledTuningT;
  using CacheInfoProvider = CacheInfoProviderT;

  [[no_unique_address]] GenericTiledTuningT generic_tiled{};
  [[no_unique_address]] CacheInfoProviderT cache_info_provider{};
};

template <::vecops::matmul::Atom AtomT,
          typename SchedulerPolicyT = kernel::matmul_policy::Automatic>
using MatmulSchedulerConfig = MatmulConfig<
    AtomT, ::vecops::matmul::family_selection::Automatic,
    SchedulerPolicyT>;

} // namespace vecops::ops

#endif // VECOPS_MATMUL_CONFIG_H
