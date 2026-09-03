//
// Copyright (c) vecops contributors.
//

#ifndef VECOPS_KERNEL_TILE2D_H
#define VECOPS_KERNEL_TILE2D_H

#include <concepts>
#include <limits>
#include <type_traits>
#include <utility>

#include "vecops/Assertion.h"
#include "vecops/CoreDefs.h"
#include "vecops/Meta.h"

/**
 * @file include/vecops/kernel/Tile2D.h
 * @brief Compile-time kernel selection and two-dimensional tiled traversal.
 *
 * `tile2d` covers `[0, m) x [0, n)` exactly once.  A kernel family has a
 * compile-time shape `(a, b)` measured in base tiles `(tm, tn)` and four mask
 * variants.  The callback receives a case tag followed by the element offset
 * and active element extent:
 *
 * @code
 * fn(Tile2DKernelCase<Family, MMask, NMask>{}, m, n, active_m, active_n);
 * @endcode
 *
 * `unmasked` is a strict promise that active extent equals kernel capacity.
 * `masked` is a capability and may also be used with a full active extent.
 *
 * Extents and base tile sizes accept `meta::Const`, `meta::Dynamic`, or raw
 * integers.  Raw integers are normalized to `meta::Any`.  When metadata proves
 * an extent divisible by the selected kernel capacity, the corresponding tail
 * path is removed with `if constexpr`.
 *
 * ## Key components
 *
 * | Component                | Purpose                                          |
 * |--------------------------|--------------------------------------------------|
 * | Tile2DMaskMode           | Per-axis mask promise/capability tag             |
 * | tile2d_policy::*         | Traversal-region strategies                      |
 * | Tile2DKernelFamily       | Manual family descriptor: shape + four powers    |
 * | Tile2DKernelCatalog      | Explicit family list (must be downward-closed)   |
 * | Tile2DGeneratedCatalog   | Families enumerated from a Provider + bounds     |
 * | Tile2DSearchSpace        | Enumeration bounds for generated catalogs        |
 * | Tile2DKernelCase         | Case tag passed to the callback                  |
 * | tile2d / tile2d_generate | Public traversal entry points                    |
 *
 * ## Usage overview
 *
 * @code
 * #include "vecops/kernel/Tile2D.h"
 * using namespace vecops::kernel::loop;
 *
 * // Downward-closed set: (2,2) requires (1,2), (2,1); those require (1,1).
 * using Catalog = Tile2DKernelCatalog<
 *     Tile2DKernelFamily<2, 2, 4>,   // 2x2 unmasked bulk
 *     Tile2DKernelFamily<1, 2, 3>,   // bottom edge: 1 x 2
 *     Tile2DKernelFamily<2, 1, 3>,   // right edge: 2 x 1
 *     Tile2DKernelFamily<1, 1, 1>>;  // doubly-masked corner
 *
 * tile2d(cint<128>, cint<128>, 16, 16, Catalog{},
 *        [](auto kernel_case, nint_t m, nint_t n,
 *           nint_t active_m, nint_t active_n) {
 *          using Case = decltype(kernel_case);
 *          if constexpr (Case::m_mask == Tile2DMaskMode::unmasked &&
 *                        Case::n_mask == Tile2DMaskMode::unmasked) {
 *            // fast unmasked micro-kernel covering active_m x active_n
 *          } else {
 *            // masked path; active extents may still equal full capacity
 *          }
 *        });
 * @endcode
 *
 * ## Pitfalls
 *
 * - The catalog must be a complete downward-closed set containing the 1x1
 *   family: every family (a, b) with a > 1 requires (a-1, b) to be listed,
 *   and every family with b > 1 requires (a, b-1). `tile2d` enforces this
 *   with a static_assert before traversing.
 * - Family selection maximizes `power / denominator` ratios (cross
 *   multiplication), not raw `power` values; an edge family competes against
 *   the bulk family's footprint, not in isolation.
 * - Extents must be non-negative and base tile sizes positive; singleton
 *   violations are compile-time errors and runtime values are checked with
 *   `VECOPS_ASSERT`.
 * - Kernel capacity is `a * tm` by `b * tn`; `tm`/`tn` must not overflow
 *   `nint_t` when multiplied by the largest family extents (asserted).
 * - A masked invocation may still carry a full active extent; only the
 *   unmasked tag promises full capacity.
 */

namespace vecops::kernel::loop {

using namespace ::vecops::meta;

/**
 * @brief Mask variant of one tile axis: promise versus capability.
 *
 * `unmasked` is a strict promise that the active extent on this axis equals
 * the kernel capacity, so the callback must not mask. `masked` is a
 * capability: the callback must honor the active extent, which may still be
 * equal to the full capacity.
 *
 * @note A family participates in selection only when all four
 *       `(MMask, NMask)` combinations report a non-negative power.
 */
enum class Tile2DMaskMode {
  unmasked,
  masked,
};

/**
 * @brief How a generated catalog may pre-commit the exact-cover bulk grid.
 *
 * `runtime` (default) keeps the exact-cover block loop runtime-driven.
 * `exact` and `unmasked` take effect only when metadata proves both extents
 * divide the bulk capacity exactly: `exact` then traverses a compile-time
 * grid whose invocations carry `Tile2DExactKernelCase` (doubly masked,
 * `exact_blocks = true`), while `unmasked` emits plain unmasked
 * `Tile2DKernelCase` invocations over the same grid.
 *
 * @note The option is probed from the Provider of a `Tile2DGeneratedCatalog`
 *       and defaults to `runtime` when the Provider does not define it.
 */
enum class Tile2DExactGridMode {
  runtime,
  exact,
  unmasked,
};

namespace tile2d_policy {

/** Classic row-major traversal with independently selected edge families. */
struct RowMajor {};

/**
 * Grouped BULK, lower, right, and corner regions. Catalogs may opt into an
 * exact logical-block traversal for compile-time axis constraints, pruning
 * unreachable families instead of forcing a shape-independent bulk family.
 */
struct FourRegions {};

/** Use one doubly-masked family for every invocation. */
struct Uniform {};

/** Use one unmasked bulk family and share one family across all boundaries. */
struct BulkTail {};

/**
 * Cover runtime logical-block remainders with exact families from the catalog.
 *
 * The catalog is the sole source of available shapes. No family selected by
 * this policy contains a completely inactive base tile, and metadata
 * constraints prune runtime choices whenever possible.
 */
struct ExactCover {};

} // namespace tile2d_policy

/**
 * Convenience kernel-family descriptor.
 *
 * Powers are relative positive throughput scores.  A custom family may expose
 * the same `a`, `b`, and `power<MM, MN>()` interface and additionally attach
 * the actual kernel type or other compile-time metadata.
 *
 * @tparam A        Family height in base tiles (positive).
 * @tparam B        Family width in base tiles (positive).
 * @tparam PowerUU  Throughput score for the unmasked/unmasked variant.
 * @tparam PowerUM  Score for unmasked rows with masked columns.
 * @tparam PowerMU  Score for masked rows with unmasked columns.
 * @tparam PowerMM  Score for the doubly-masked variant.
 */
template <
    int A, int B,
    int PowerUU,
    int PowerUM = PowerUU,
    int PowerMU = PowerUU,
    int PowerMM = PowerUU>
struct Tile2DKernelFamily {
  static_assert(A > 0 && B > 0, "tile2d kernel shape must be positive");
  static constexpr int a = A;
  static constexpr int b = B;

  template <Tile2DMaskMode MMask, Tile2DMaskMode NMask>
  static consteval int power() {
    if constexpr (MMask == Tile2DMaskMode::unmasked &&
                  NMask == Tile2DMaskMode::unmasked) {
      return PowerUU;
    } else if constexpr (MMask == Tile2DMaskMode::unmasked) {
      return PowerUM;
    } else if constexpr (NMask == Tile2DMaskMode::unmasked) {
      return PowerMU;
    } else {
      return PowerMM;
    }
  }
};

/**
 * @brief Explicit list of kernel families.
 *
 * The set must be downward-closed and contain the 1x1 family: family (a, b)
 * with a > 1 requires (a-1, b) to be listed, and b > 1 requires (a, b-1).
 * `tile2d` verifies this with a static_assert before traversing, because the
 * edge and corner selectors shrink any family by one tile in either axis.
 */
template <typename... Families>
struct Tile2DKernelCatalog {};

/**
 * @brief Enumeration bounds for a Tile2DGeneratedCatalog.
 *
 * The generated catalog enumerates every shape (a, b) with
 * 1 <= a <= max_a, 1 <= b <= max_b, and a * b <= max_tiles; the Provider's
 * power() decides which of those shapes actually exist.
 *
 * @tparam MaxA      Largest family height in base tiles.
 * @tparam MaxB      Largest family width in base tiles.
 * @tparam MaxTiles  Cap on base tiles per family (defaults to MaxA * MaxB).
 */
template <int MaxA, int MaxB, int MaxTiles = MaxA * MaxB>
struct Tile2DSearchSpace {
  static_assert(MaxA > 0 && MaxB > 0 && MaxTiles > 0,
                "tile2d search bounds must be positive");
  static constexpr int max_a = MaxA;
  static constexpr int max_b = MaxB;
  static constexpr int max_tiles = MaxTiles;
};

/**
 * @brief Catalog whose families are enumerated from a Provider.
 *
 * A generated catalog queries:
 *
 * @code
 * Provider::template power<A, B, MMask, NMask>()
 * @endcode
 *
 * A negative result means the case does not exist.  A shape participates only
 * when all four mask variants exist.
 *
 * @tparam Provider     Static power source as described above; may also
 *                      define the optional options probed by CatalogOptions
 *                      (exact_grid_mode, exact_meta_block_limit,
 *                      four_regions_exact_constraints).
 * @tparam SearchSpace  Tile2DSearchSpace bounding the enumerated shapes.
 */
template <typename Provider, typename SearchSpace>
struct Tile2DGeneratedCatalog {};

/**
 * @brief Case tag passed as the first callback argument of `tile2d`.
 *
 * It reports which family and mask variant was selected for the current
 * invocation: the kernel capacity is `a * tm` by `b * tn` elements, and
 * `m_mask`/`n_mask` say whether the `active_m`/`active_n` extents must be
 * honored by masking. `exact_blocks` is false here: offsets are arbitrary
 * element positions produced by the region policies.
 *
 * @see Tile2DExactKernelCase
 */
template <typename Family, Tile2DMaskMode MMask, Tile2DMaskMode NMask>
struct Tile2DKernelCase {
  using family_type = Family;
  static constexpr int a = Family::a;
  static constexpr int b = Family::b;
  static constexpr Tile2DMaskMode m_mask = MMask;
  static constexpr Tile2DMaskMode n_mask = NMask;
  static constexpr bool exact_blocks = false;
};

/**
 * @brief Case tag for invocations from an exact-logical-block traversal.
 *
 * The ExactCover policy (and exact-grid mode) move over whole base-tile
 * blocks, so the offset passed to the callback is always a multiple of the
 * base tile size. The case is always doubly masked — an edge block may still
 * be partially active — and family selection guarantees that no emitted
 * family contains a completely inactive base tile.
 */
template <typename Family>
struct Tile2DExactKernelCase
    : Tile2DKernelCase<
          Family, Tile2DMaskMode::masked, Tile2DMaskMode::masked> {
  static constexpr bool exact_blocks = true;
};

namespace tile2d_details {

// True when an extent provably needs no tail loop at the given block step.
// A zero extent is trivially tail-free; a singleton extent needs the step to
// divide it; a range extent with a singleton step needs the extent type to
// prove alignment (Dynamic::aligns). Mixed runtime/runtime combinations are
// conservatively false.
template <meta::ValueType Extent, meta::ValueType Step>
inline constexpr bool has_no_tail_v = [] {
  using E = std::remove_cvref_t<Extent>;
  using S = std::remove_cvref_t<Step>;
  if constexpr (meta::is_singleton_v<E>) {
    if constexpr (meta::singleton_value_v<E> == 0) {
      return true;
    } else if constexpr (meta::is_singleton_v<S>) {
      constexpr nint_t step = meta::singleton_value_v<S>;
      return step > 0 && meta::singleton_value_v<E> % step == 0;
    } else {
      return false;
    }
  } else if constexpr (meta::is_singleton_v<S>) {
    constexpr nint_t step = meta::singleton_value_v<S>;
    return step > 0 && E::aligns(step);
  } else {
    return false;
  }
}();

// The block count is a compile-time constant only when both extent and tile
// are singletons; fixed_block_count_n is their ceil division.
template <meta::ValueType Extent, meta::ValueType Tile>
inline constexpr bool fixed_block_count_v =
    meta::is_singleton_v<Extent> && meta::is_singleton_v<Tile> &&
    meta::singleton_value_v<Tile> > 0;

template <meta::ValueType Extent, meta::ValueType Tile>
  requires fixed_block_count_v<Extent, Tile>
inline constexpr nint_t fixed_block_count_n =
    meta::singleton_value_v<Extent> / meta::singleton_value_v<Tile> +
    (meta::singleton_value_v<Extent> % meta::singleton_value_v<Tile> != 0);

// Provable from metadata alone: an extent upper bound no larger than a tile
// lower bound means at most one (possibly partial) block for any runtime
// values, so the traversal needs no block loop.
template <meta::ValueType Extent, meta::ValueType Tile>
inline constexpr bool at_most_one_block_v = [] {
  using E = std::remove_cvref_t<Extent>;
  using T = std::remove_cvref_t<Tile>;
  if constexpr (meta::has_upper_bound_v<E> &&
                meta::has_lower_bound_v<T> &&
                meta::lower_bound_v<T> > 0) {
    return meta::upper_bound_v<E> <= meta::lower_bound_v<T>;
  } else {
    return false;
  }
}();

// Upper bound on the number of blocks provable from extent/tile metadata.
// max_block_count_n falls back to numeric_limits max to encode "no bound
// known"; callers must gate on has_max_block_count_v before comparing.
template <meta::ValueType Extent, meta::ValueType Tile>
inline constexpr bool has_max_block_count_v = [] {
  using E = std::remove_cvref_t<Extent>;
  using T = std::remove_cvref_t<Tile>;
  return meta::has_upper_bound_v<E> &&
      meta::has_lower_bound_v<T> && meta::lower_bound_v<T> > 0;
}();

template <meta::ValueType Extent, meta::ValueType Tile>
inline constexpr nint_t max_block_count_n = [] {
  using E = std::remove_cvref_t<Extent>;
  using T = std::remove_cvref_t<Tile>;
  if constexpr (has_max_block_count_v<E, T>) {
    constexpr nint_t extent = meta::upper_bound_v<E>;
    constexpr nint_t tile = meta::lower_bound_v<T>;
    return extent <= 0 ? 0 : extent / tile + (extent % tile != 0);
  } else {
    return std::numeric_limits<nint_t>::max();
  }
}();

// Kernel capacity in elements: family extent (in base tiles) times base tile
// size, tracked as a meta value so divisibility stays a compile-time fact.
template <int Factor, meta::ValueType Tile>
VECOPS_ALWAYS_INLINE constexpr auto capacity(Tile tile) {
  return tile * cint<Factor>;
}

template <meta::ValueType Tile, int Factor>
using Capacity = decltype(capacity<Factor>(std::declval<Tile>()));

template <meta::ValueType Value>
consteval nint_t effective_lower_bound() {
  using V = std::remove_cvref_t<Value>;
  if constexpr (::vecops::meta::has_lower_bound_v<V>) {
    return ::vecops::meta::lower_bound_v<V>;
  } else {
    // tile2d validates tile sizes as positive, so one is a safe semantic bound.
    return 1;
  }
}

// Upper bound on `extent % step`, or a negative value when no bound can be
// proven. Zero means the division is exact, so no remainder exists at all.
// When both sides are singletons the remainder is computed exactly;
// otherwise the bound is the tighter of the extent upper bound and
// `step_upper - 1` (the largest remainder any positive step can leave),
// with missing bounds ignored.
template <meta::ValueType Extent, meta::ValueType Step>
consteval nint_t remainder_upper_bound() {
  using E = std::remove_cvref_t<Extent>;
  using S = std::remove_cvref_t<Step>;
  if constexpr (has_no_tail_v<E, S>) {
    return 0;
  } else if constexpr (meta::is_singleton_v<E> && meta::is_singleton_v<S>) {
    return meta::singleton_value_v<E> % meta::singleton_value_v<S>;
  } else {
    nint_t upper = -1;
    if constexpr (::vecops::meta::has_upper_bound_v<E>) {
      upper = ::vecops::meta::upper_bound_v<E>;
    }
    if constexpr (::vecops::meta::has_upper_bound_v<S>) {
      constexpr nint_t step_upper = ::vecops::meta::upper_bound_v<S>;
      if constexpr (step_upper > 0) {
        const nint_t from_step = step_upper - 1;
        upper = upper < 0 || from_step < upper ? from_step : upper;
      }
    }
    return upper;
  }
}

// Can a candidate family cover the main family's remainder region? Yes when
// the remainder is provably empty (0), when both capacities scale with the
// same positive runtime tile value (the candidate then never falls short of
// the main capacity for a factor >= the main factor), or when the candidate's
// guaranteed capacity lower bound reaches the remainder upper bound.
template <meta::ValueType CandidateTile, int CandidateFactor,
          meta::ValueType MainTile, int MainFactor,
          meta::ValueType Extent>
consteval bool covers_main_remainder() {
  using CandidateCapacity = Capacity<CandidateTile, CandidateFactor>;
  using MainCapacity = Capacity<MainTile, MainFactor>;
  constexpr nint_t remainder_upper =
      remainder_upper_bound<Extent, MainCapacity>();
  if constexpr (remainder_upper == 0) {
    return true;
  } else if constexpr (
      std::same_as<std::remove_cvref_t<CandidateTile>,
                   std::remove_cvref_t<MainTile>> &&
      CandidateFactor >= MainFactor) {
    // Both capacities depend on the same positive runtime tile value.
    return true;
  } else if constexpr (remainder_upper >= 0) {
    return effective_lower_bound<CandidateCapacity>() >= remainder_upper;
  } else {
    return false;
  }
}

// A family participates in selection only with a positive shape and all four
// mask variants present (non-negative power).
template <typename Family>
consteval bool complete_family() {
  return Family::a > 0 && Family::b > 0 &&
      Family::template power<Tile2DMaskMode::unmasked,
                             Tile2DMaskMode::unmasked>() >= 0 &&
      Family::template power<Tile2DMaskMode::unmasked,
                             Tile2DMaskMode::masked>() >= 0 &&
      Family::template power<Tile2DMaskMode::masked,
                             Tile2DMaskMode::unmasked>() >= 0 &&
      Family::template power<Tile2DMaskMode::masked,
                             Tile2DMaskMode::masked>() >= 0;
}

template <typename Provider, int A, int B>
struct GeneratedFamily {
  static constexpr int a = A;
  static constexpr int b = B;

  template <Tile2DMaskMode MMask, Tile2DMaskMode NMask>
  static consteval int power() {
    return Provider::template power<A, B, MMask, NMask>();
  }
};

template <int I, int End, typename Fn>
consteval void static_for(Fn& fn) {
  if constexpr (I <= End) {
    fn.template operator()<I>();
    static_for<I + 1, End>(fn);
  }
}

template <typename... Families, typename Fn>
consteval void for_each_family(Tile2DKernelCatalog<Families...>, Fn& fn) {
  (fn.template operator()<Families>(), ...);
}

// Enumerate the search rectangle of a generated catalog, skipping shapes
// over the tile budget.
template <typename Provider, typename Search, typename Fn>
consteval void for_each_family(
    Tile2DGeneratedCatalog<Provider, Search>, Fn& fn) {
  auto visit_a = [&]<int A>() consteval {
    auto visit_b = [&]<int B>() consteval {
      if constexpr (A * B <= Search::max_tiles) {
        fn.template operator()<GeneratedFamily<Provider, A, B>>();
      }
    };
    static_for<1, Search::max_b>(visit_b);
  };
  static_for<1, Search::max_a>(visit_a);
}

template <typename Catalog, int A, int B>
consteval bool has_family() {
  bool found = false;
  auto visit = [&]<typename Family>() consteval {
    if constexpr (complete_family<Family>() &&
                  Family::a == A && Family::b == B) {
      found = true;
    }
  };
  for_each_family(Catalog{}, visit);
  return found;
}

enum class TileAxis { m, n };

template <TileAxis Axis, typename Family>
inline constexpr int family_extent_v =
    Axis == TileAxis::m ? Family::a : Family::b;

template <TileAxis Axis, typename Catalog, typename Predicate>
consteval int max_family_extent(Predicate predicate) {
  int result = 0;
  auto visit = [&]<typename Family>() consteval {
    if constexpr (complete_family<Family>() &&
                  predicate.template operator()<Family>()) {
      constexpr int extent = family_extent_v<Axis, Family>;
      result = result < extent ? extent : result;
    }
  };
  for_each_family(Catalog{}, visit);
  return result;
}

template <typename Catalog>
consteval int max_family_a() {
  return max_family_extent<TileAxis::m, Catalog>(
      []<typename>() consteval { return true; });
}

template <typename Catalog>
consteval int max_family_b() {
  return max_family_extent<TileAxis::n, Catalog>(
      []<typename>() consteval { return true; });
}

template <typename Catalog, int A>
consteval int max_family_b_for_a() {
  return max_family_extent<TileAxis::n, Catalog>(
      []<typename Family>() consteval { return Family::a == A; });
}

// Widest family usable as an exact bulk row. Exact rows emit whole rows of
// one family, so a row height is preferred only when some family with b >= 2
// exists at that height (b == 1 would tile the row one base tile at a time).
// If no multi-column family exists at all, fall back to the tallest family
// so the grid degenerates gracefully instead of failing.
template <typename Catalog>
consteval int max_exact_row_a() {
  constexpr int result = max_family_extent<TileAxis::m, Catalog>(
      []<typename Family>() consteval { return Family::b >= 2; });
  return result == 0 ? max_family_a<Catalog>() : result;
}

// Options read from the catalog. A plain list catalog keeps the defaults; a
// generated catalog forwards the Provider's optional static members when
// present (probed with `requires`).
template <typename Catalog>
struct CatalogOptions {
  static constexpr Tile2DExactGridMode exact_grid_mode =
      Tile2DExactGridMode::runtime;
  static constexpr nint_t exact_meta_block_limit =
      std::numeric_limits<nint_t>::max();
  static constexpr bool four_regions_exact_constraints = false;
};

template <typename Provider, typename Search>
struct CatalogOptions<Tile2DGeneratedCatalog<Provider, Search>> {
  static constexpr Tile2DExactGridMode exact_grid_mode = [] {
    if constexpr (requires { Provider::exact_grid_mode; })
      return Provider::exact_grid_mode;
    else
      return Tile2DExactGridMode::runtime;
  }();
  static constexpr nint_t exact_meta_block_limit = [] {
    if constexpr (requires { Provider::exact_meta_block_limit; })
      return Provider::exact_meta_block_limit;
    else
      return std::numeric_limits<nint_t>::max();
  }();
  static constexpr bool four_regions_exact_constraints = [] {
    if constexpr (requires { Provider::four_regions_exact_constraints; })
      return Provider::four_regions_exact_constraints;
    else
      return false;
  }();
};

template <typename Catalog>
inline constexpr auto exact_grid_mode_v =
    CatalogOptions<Catalog>::exact_grid_mode;
template <typename Catalog>
inline constexpr auto exact_meta_block_limit_v =
    CatalogOptions<Catalog>::exact_meta_block_limit;
template <typename Catalog>
inline constexpr auto four_regions_exact_constraints_v =
    CatalogOptions<Catalog>::four_regions_exact_constraints;

// Catalog invariant: the family set is downward-closed (every (a, b) has
// (a-1, b) and (a, b-1) when applicable) and contains (1,1). Edge and corner
// selection relies on shrinking any family by one tile in either axis.
template <typename Catalog>
consteval bool valid_catalog() {
  bool valid = true;
  bool has_any = false;
  auto visit = [&]<typename Family>() consteval {
    if constexpr (complete_family<Family>()) {
      has_any = true;
      if constexpr (Family::a > 1) {
        valid = valid && has_family<Catalog, Family::a - 1, Family::b>();
      }
      if constexpr (Family::b > 1) {
        valid = valid && has_family<Catalog, Family::a, Family::b - 1>();
      }
    }
  };
  for_each_family(Catalog{}, visit);
  return valid && has_any && has_family<Catalog, 1, 1>();
}

// Selection result; power < 0 means "no candidate".
struct Choice {
  int a = 0;
  int b = 0;
  int power = -1;
  int denominator = 1;
};

// Compare power/denominator ratios with cross multiplication instead of
// division; wide intermediates keep plausible scores from overflowing.
constexpr bool better_choice(
    int power, int denominator, const Choice& best) {
  if (power < 0 || denominator <= 0) return false;
  if (best.power < 0) return true;
  using Wide = long long;
  return static_cast<Wide>(power) * best.denominator >
         static_cast<Wide>(best.power) * denominator;
}

// Pick the family with the best power-per-denominator ratio among complete
// families satisfying the predicate.
template <Tile2DMaskMode MMask, Tile2DMaskMode NMask,
          typename Catalog, typename Predicate, typename Denominator>
consteval Choice select_family(Predicate predicate, Denominator denominator) {
  Choice best{};
  auto visit = [&]<typename Family>() consteval {
    if constexpr (complete_family<Family>() && predicate.template operator()<Family>()) {
      constexpr int power = Family::template power<MMask, NMask>();
      constexpr int denom = denominator.template operator()<Family>();
      if (better_choice(power, denom, best)) {
        best = Choice{Family::a, Family::b, power, denom};
      }
    }
  };
  for_each_family(Catalog{}, visit);
  return best;
}

// Linear search through an explicit catalog. The final `return Head` arm is
// unreachable — the static_assert rejects a missing family before it — and
// exists only so the function returns a value on every path.
template <int A, int B, typename Head, typename... Tail>
consteval auto find_explicit_family(
    Tile2DKernelCatalog<Head, Tail...>) {
  if constexpr (Head::a == A && Head::b == B) {
    return std::type_identity<Head>{};
  } else {
    static_assert(sizeof...(Tail) > 0,
                  "selected tile2d family is missing from catalog");
    if constexpr (sizeof...(Tail) > 0)
      return find_explicit_family<A, B>(Tile2DKernelCatalog<Tail...>{});
    else
      return std::type_identity<Head>{};
  }
}

template <typename Catalog, int A, int B>
struct FamilyFor;

template <typename... Families, int A, int B>
struct FamilyFor<Tile2DKernelCatalog<Families...>, A, B> {
  using type = typename decltype(find_explicit_family<A, B>(
      Tile2DKernelCatalog<Families...>{}))::type;
};

template <typename Provider, typename Search, int A, int B>
struct FamilyFor<Tile2DGeneratedCatalog<Provider, Search>, A, B> {
  using type = GeneratedFamily<Provider, A, B>;
};

template <typename Catalog, int A, int B>
using family_for_t = typename FamilyFor<Catalog, A, B>::type;

// Best unmasked/unmasked family (any shape) — the bulk tile — and best
// doubly-masked family (any shape), used by the Uniform and BulkTail
// policies as their single fallback family.
template <typename Catalog>
consteval Choice select_bulk() {
  return select_family<
      Tile2DMaskMode::unmasked, Tile2DMaskMode::unmasked, Catalog>(
      []<typename Family>() consteval { return true; },
      []<typename Family>() consteval { return 1; });
}

template <typename Catalog>
consteval Choice select_single() {
  return select_family<
      Tile2DMaskMode::masked, Tile2DMaskMode::masked, Catalog>(
      []<typename Family>() consteval { return true; },
      []<typename Family>() consteval { return 1; });
}

template <TileAxis Axis>
inline constexpr TileAxis other_axis_v =
    Axis == TileAxis::m ? TileAxis::n : TileAxis::m;

// Row-major edge family: it must exactly match the bulk width on the other
// axis so tile emission stays column-contiguous along the bulk rows, and its
// extent along the masked axis must provably cover the bulk remainder.
// Maximizes power per masked-axis element (denominator = that extent).
template <TileAxis Axis, typename Catalog, typename Bulk,
          meta::ValueType Extent, meta::ValueType Tile>
consteval Choice select_natural_edge() {
  constexpr auto MMask = Axis == TileAxis::m
      ? Tile2DMaskMode::masked : Tile2DMaskMode::unmasked;
  constexpr auto NMask = Axis == TileAxis::n
      ? Tile2DMaskMode::masked : Tile2DMaskMode::unmasked;
  return select_family<
      MMask, NMask, Catalog>(
      []<typename Family>() consteval {
        return family_extent_v<other_axis_v<Axis>, Family> ==
                   family_extent_v<other_axis_v<Axis>, Bulk> &&
            covers_main_remainder<
                Tile, family_extent_v<Axis, Family>,
                Tile, family_extent_v<Axis, Bulk>, Extent>();
      },
      []<typename Family>() consteval {
        return family_extent_v<Axis, Family>;
      });
}

// Corner family: must provably cover both bulk remainders; maximizes power
// per element (denominator = a * b).
template <typename Catalog, typename Bulk,
          meta::ValueType ExtentM, meta::ValueType ExtentN,
          meta::ValueType TileM, meta::ValueType TileN>
consteval Choice select_corner() {
  return select_family<
      Tile2DMaskMode::masked, Tile2DMaskMode::masked, Catalog>(
      []<typename Family>() consteval {
        return covers_main_remainder<TileM, Family::a,
                                     TileM, Bulk::a, ExtentM>() &&
            covers_main_remainder<TileN, Family::b,
                                  TileN, Bulk::b, ExtentN>();
      },
      []<typename Family>() consteval { return Family::a * Family::b; });
}

// Four-regions edge family: unlike the row-major variant, its width on the
// other axis only has to divide the bulk width (the region is emitted as one
// band along the whole bulk extent), and its extent along the masked axis
// must still provably cover the bulk remainder.
template <TileAxis Axis, typename Catalog, typename Bulk,
          meta::ValueType Extent, meta::ValueType Tile>
consteval Choice select_region_edge() {
  constexpr auto MMask = Axis == TileAxis::m
      ? Tile2DMaskMode::masked : Tile2DMaskMode::unmasked;
  constexpr auto NMask = Axis == TileAxis::n
      ? Tile2DMaskMode::masked : Tile2DMaskMode::unmasked;
  return select_family<
      MMask, NMask, Catalog>(
      []<typename Family>() consteval {
        return family_extent_v<other_axis_v<Axis>, Bulk> %
                   family_extent_v<other_axis_v<Axis>, Family> == 0 &&
            covers_main_remainder<
                Tile, family_extent_v<Axis, Family>,
                Tile, family_extent_v<Axis, Bulk>, Extent>();
      },
      []<typename Family>() consteval {
        return family_extent_v<Axis, Family>;
      });
}

template <typename Family, Tile2DMaskMode MMask, Tile2DMaskMode NMask,
          typename Fn>
VECOPS_ALWAYS_INLINE void invoke(
    Fn& fn, nint_t m, nint_t n, nint_t active_m, nint_t active_n) {
  fn(Tile2DKernelCase<Family, MMask, NMask>{},
     m, n, active_m, active_n);
}

// Unmasked grid over [m_begin, m_end) x [n_begin, n_end); the caller
// guarantees that both ranges divide the tile capacity exactly.
template <typename Family, typename Fn>
VECOPS_ALWAYS_INLINE void emit_full_grid(
    nint_t m_begin, nint_t m_end,
    nint_t n_begin, nint_t n_end,
    nint_t cap_m, nint_t cap_n,
    Fn& fn) {
  for (nint_t m = m_begin; m < m_end; m += cap_m) {
    for (nint_t n = n_begin; n < n_end; n += cap_n) {
      invoke<Family, Tile2DMaskMode::unmasked,
             Tile2DMaskMode::unmasked>(
          fn, m, n, cap_m, cap_n);
    }
  }
}

// Decide whether metadata constraints (upper extent bounds, or fixed block
// counts under the catalog's meta block limit) allow the constrained exact
// traversal, which prunes runtime dispatch with `if constexpr`. Unbounded
// extents never qualify: the compile-time enumeration would not terminate.
template <typename Catalog,
          meta::ValueType ExtentM, meta::ValueType ExtentN,
          meta::ValueType TileM, meta::ValueType TileN>
inline constexpr bool meta_exact_candidate_v = [] {
  constexpr bool HasConstraint =
      has_max_block_count_v<ExtentM, TileM> ||
      has_max_block_count_v<ExtentN, TileN>;
  if constexpr (!HasConstraint) {
    return false;
  } else {
    constexpr nint_t Limit = exact_meta_block_limit_v<Catalog>;
    if constexpr (fixed_block_count_v<ExtentM, TileM>) {
      if constexpr (fixed_block_count_n<ExtentM, TileM> > Limit) return false;
    }
    if constexpr (fixed_block_count_v<ExtentN, TileN>) {
      if constexpr (fixed_block_count_n<ExtentN, TileN> > Limit) return false;
    }
    return true;
  }
}();

// Runtime state shared by the exact-grid emitters: extents, base tile
// sizes, and the callback. invoke_exact translates a logical block
// coordinate (block_m, block_n) into an element offset and a possibly
// partial active extent, then dispatches through Tile2DExactKernelCase.
template <typename Catalog,
          meta::ValueType ExtentM, meta::ValueType ExtentN,
          meta::ValueType TileM, meta::ValueType TileN, typename Fn>
struct ExactContext {
  using catalog_type = Catalog;
  using extent_m_type = std::remove_cvref_t<ExtentM>;
  using extent_n_type = std::remove_cvref_t<ExtentN>;
  using tile_m_type = std::remove_cvref_t<TileM>;
  using tile_n_type = std::remove_cvref_t<TileN>;

  nint_t m_extent;
  nint_t n_extent;
  nint_t tm;
  nint_t tn;
  Fn& fn;

  template <int A, int B>
  VECOPS_ALWAYS_INLINE void invoke_exact(
      nint_t block_m, nint_t block_n) {
    using Family = family_for_t<Catalog, A, B>;
    const nint_t m = block_m * tm;
    const nint_t n = block_n * tn;
    const nint_t cap_m = static_cast<nint_t>(A) * tm;
    const nint_t cap_n = static_cast<nint_t>(B) * tn;
    const nint_t remaining_m = m_extent - m;
    const nint_t remaining_n = n_extent - n;
    fn(Tile2DExactKernelCase<Family>{}, m, n,
       remaining_m < cap_m ? remaining_m : cap_m,
       remaining_n < cap_n ? remaining_n : cap_n);
  }
};

template <typename Catalog,
          meta::ValueType M, meta::ValueType N,
          meta::ValueType TM, meta::ValueType TN, typename Fn>
VECOPS_ALWAYS_INLINE auto make_exact_context(
    M m, N n, TM tm, TN tn, Fn& fn) {
  return ExactContext<Catalog, M, N, TM, TN, Fn>{
      static_cast<nint_t>(m), static_cast<nint_t>(n),
      static_cast<nint_t>(tm), static_cast<nint_t>(tn), fn};
}

// Under Constrained = false the extent/tile types are erased to Any, so the
// emit_* helpers skip the compile-time pruning instead of asserting on it.
template <bool Constrained, typename Context>
using context_extent_m_t = std::conditional_t<
    Constrained, typename Context::extent_m_type, meta::Any>;
template <bool Constrained, typename Context>
using context_extent_n_t = std::conditional_t<
    Constrained, typename Context::extent_n_type, meta::Any>;
template <bool Constrained, typename Context>
using context_tile_m_t = std::conditional_t<
    Constrained, typename Context::tile_m_type, meta::Any>;
template <bool Constrained, typename Context>
using context_tile_n_t = std::conditional_t<
    Constrained, typename Context::tile_n_type, meta::Any>;

// Find a family matching the exact remainder block count by trying
// B, B-1, ..., 0 in turn; the metadata bound prunes heights that cannot
// occur at this extent before the runtime `remaining` comparison.
template <int A, int B, bool Constrained, typename Context>
VECOPS_ALWAYS_INLINE void dispatch_exact_remainder(
    Context& ctx, nint_t block_m, nint_t block_n, nint_t remaining) {
  if constexpr (B > 0) {
    using ExtentN = context_extent_n_t<Constrained, Context>;
    using TileN = context_tile_n_t<Constrained, Context>;
    if constexpr (!has_max_block_count_v<ExtentN, TileN> ||
                  max_block_count_n<ExtentN, TileN> >= B) {
      if (remaining == B) {
        ctx.template invoke_exact<A, B>(block_m, block_n);
        return;
      }
    }
    dispatch_exact_remainder<A, B - 1, Constrained>(
        ctx, block_m, block_n, remaining);
  }
}

// Emit one row of height A: as many BMax-wide tiles as fit, then the
// remainder via dispatch_exact_remainder. A compile-time block count unrolls
// the loop and sizes the remainder tile statically; a runtime count keeps a
// plain loop.
template <int A, int BMax, bool Constrained, typename Context>
VECOPS_ALWAYS_INLINE void emit_exact_row(
    Context& ctx, nint_t n_blocks, nint_t block_m) {
  using ExtentN = context_extent_n_t<Constrained, Context>;
  using TileN = context_tile_n_t<Constrained, Context>;
  nint_t block_n = 0;
  if constexpr (fixed_block_count_v<ExtentN, TileN>) {
    constexpr nint_t NBlocks = fixed_block_count_n<ExtentN, TileN>;
    if constexpr (NBlocks >= BMax) {
      VECOPS_NOUNROLL
      for (; block_n + BMax <= NBlocks; block_n += BMax)
        ctx.template invoke_exact<A, BMax>(block_m, block_n);
    }
    constexpr nint_t Remainder = NBlocks % BMax;
    if constexpr (Remainder > 0)
      ctx.template invoke_exact<A, static_cast<int>(Remainder)>(
          block_m, block_n);
  } else {
    if constexpr (!has_max_block_count_v<ExtentN, TileN> ||
                  max_block_count_n<ExtentN, TileN> >= BMax) {
      for (; block_n + BMax <= n_blocks; block_n += BMax)
        ctx.template invoke_exact<A, BMax>(block_m, block_n);
    }
    dispatch_exact_remainder<A, BMax - 1, Constrained>(
        ctx, block_m, block_n, n_blocks - block_n);
  }
}

// Cover a compile-time-known MBlocks with rows of decreasing height: as many
// A-tall rows as fit, then recurse on the leftover rows with A - 1.
template <int A, nint_t RemainingM, bool Constrained, typename Context>
VECOPS_ALWAYS_INLINE void emit_fixed_exact_rows(
    Context& ctx, nint_t n_blocks, nint_t& block_m) {
  if constexpr (A > 0 && RemainingM > 0) {
    using Catalog = typename Context::catalog_type;
    constexpr int BMax = max_family_b_for_a<Catalog, A>();
    if constexpr (BMax >= 2 && RemainingM >= A) {
      constexpr nint_t Count = RemainingM / A;
      VECOPS_NOUNROLL
      for (nint_t i = 0; i < Count; ++i, block_m += A)
        emit_exact_row<A, BMax, Constrained>(ctx, n_blocks, block_m);
      emit_fixed_exact_rows<
          A - 1, RemainingM % A, Constrained>(ctx, n_blocks, block_m);
    } else {
      emit_fixed_exact_rows<A - 1, RemainingM, Constrained>(
          ctx, n_blocks, block_m);
    }
  }
}

// Same as emit_fixed_exact_rows for a runtime m_blocks; the metadata bound
// prunes row heights that cannot occur at this extent.
template <int A, bool Constrained, typename Context>
VECOPS_ALWAYS_INLINE void emit_runtime_exact_rows(
    Context& ctx, nint_t m_blocks, nint_t n_blocks, nint_t& block_m) {
  if constexpr (A > 0) {
    using Catalog = typename Context::catalog_type;
    using ExtentM = context_extent_m_t<Constrained, Context>;
    using TileM = context_tile_m_t<Constrained, Context>;
    constexpr int BMax = max_family_b_for_a<Catalog, A>();
    if constexpr (
        BMax >= 2 &&
        (!has_max_block_count_v<ExtentM, TileM> ||
         max_block_count_n<ExtentM, TileM> >= A)) {
      for (; block_m + A <= m_blocks; block_m += A)
        emit_exact_row<A, BMax, Constrained>(ctx, n_blocks, block_m);
    }
    emit_runtime_exact_rows<A - 1, Constrained>(
        ctx, m_blocks, n_blocks, block_m);
  }
}

// Cover the rows of a single n-block column with the tallest available
// (A, 1) families: as many A-tall blocks as fit, then recurse with A - 1.
// Fixed variant for a compile-time-known row count.
template <int A, nint_t RemainingM, typename Context>
VECOPS_ALWAYS_INLINE void emit_fixed_exact_column(
    Context& ctx, nint_t& block_m) {
  if constexpr (A > 0 && RemainingM > 0) {
    using Catalog = typename Context::catalog_type;
    if constexpr (has_family<Catalog, A, 1>() && RemainingM >= A) {
      constexpr nint_t Count = RemainingM / A;
      VECOPS_NOUNROLL
      for (nint_t i = 0; i < Count; ++i, block_m += A)
        ctx.template invoke_exact<A, 1>(block_m, 0);
      emit_fixed_exact_column<A - 1, RemainingM % A>(ctx, block_m);
    } else {
      emit_fixed_exact_column<A - 1, RemainingM>(ctx, block_m);
    }
  }
}

// Column variant for a runtime m_blocks; the metadata bound prunes heights
// that cannot occur at this extent.
template <int A, bool Constrained, typename Context>
VECOPS_ALWAYS_INLINE void emit_runtime_exact_column(
    Context& ctx, nint_t m_blocks, nint_t& block_m) {
  if constexpr (A > 0) {
    using Catalog = typename Context::catalog_type;
    using ExtentM = context_extent_m_t<Constrained, Context>;
    using TileM = context_tile_m_t<Constrained, Context>;
    if constexpr (
        has_family<Catalog, A, 1>() &&
        (!has_max_block_count_v<ExtentM, TileM> ||
         max_block_count_n<ExtentM, TileM> >= A)) {
      for (; block_m + A <= m_blocks; block_m += A)
        ctx.template invoke_exact<A, 1>(block_m, 0);
    }
    emit_runtime_exact_column<A - 1, Constrained>(
        ctx, m_blocks, block_m);
  }
}

// Compile-time grid strides A and B over runtime extents; every invocation
// goes through invoke_exact (Tile2DExactKernelCase) and clips the active
// extents at the plane boundary.
template <int A, int B, typename Context>
VECOPS_ALWAYS_INLINE void run_context_exact_grid(Context& ctx) {
  VECOPS_NOUNROLL
  for (nint_t bm = 0; bm * ctx.tm < ctx.m_extent; bm += A) {
    VECOPS_NOUNROLL
    for (nint_t bn = 0; bn * ctx.tn < ctx.n_extent; bn += B)
      ctx.template invoke_exact<A, B>(bm, bn);
  }
}

// Exact-cover driver. The four branches, most constrained first:
// 1. both block counts compile-time — an unmasked full grid when divisibility
//    and the catalog's grid mode allow it, otherwise fixed rows (or a single
//    fixed column when there is only one n-block);
// 2. n provably has at most one block — column mode over runtime rows;
// 3. only m fixed — fixed rows over runtime n-blocks;
// 4. fully runtime — column mode when n_blocks == 1 at run time, else rows.
template <bool Constrained, typename Context>
VECOPS_ALWAYS_INLINE void run_context_exact_cover(Context& ctx) {
  using Catalog = typename Context::catalog_type;
  using ExtentM = context_extent_m_t<Constrained, Context>;
  using ExtentN = context_extent_n_t<Constrained, Context>;
  using TileM = context_tile_m_t<Constrained, Context>;
  using TileN = context_tile_n_t<Constrained, Context>;
  if (ctx.m_extent == 0 || ctx.n_extent == 0) return;
  const nint_t m_blocks =
      ctx.m_extent / ctx.tm + (ctx.m_extent % ctx.tm != 0);
  const nint_t n_blocks =
      ctx.n_extent / ctx.tn + (ctx.n_extent % ctx.tn != 0);

  if constexpr (fixed_block_count_v<ExtentM, TileM> &&
                fixed_block_count_v<ExtentN, TileN>) {
    constexpr nint_t MBlocks = fixed_block_count_n<ExtentM, TileM>;
    constexpr nint_t NBlocks = fixed_block_count_n<ExtentN, TileN>;
    constexpr int BulkA = max_exact_row_a<Catalog>();
    constexpr int BulkB = max_family_b_for_a<Catalog, BulkA>();
    if constexpr (MBlocks == 0 || NBlocks == 0) {
      return;
    } else if constexpr (
        exact_grid_mode_v<Catalog> == Tile2DExactGridMode::unmasked &&
        has_no_tail_v<ExtentM, Capacity<TileM, BulkA>> &&
        has_no_tail_v<ExtentN, Capacity<TileN, BulkB>>) {
      using Bulk = family_for_t<Catalog, BulkA, BulkB>;
      emit_full_grid<Bulk>(
          0, ctx.m_extent, 0, ctx.n_extent,
          BulkA * ctx.tm, BulkB * ctx.tn, ctx.fn);
    } else {
      nint_t block_m = 0;
      if constexpr (NBlocks == 1) {
        emit_fixed_exact_column<
            max_family_a<Catalog>(), MBlocks>(ctx, block_m);
      } else {
        emit_fixed_exact_rows<
            BulkA, MBlocks, true>(ctx, n_blocks, block_m);
      }
    }
  } else if constexpr (at_most_one_block_v<ExtentN, TileN>) {
    nint_t block_m = 0;
    emit_runtime_exact_column<max_family_a<Catalog>(), true>(
        ctx, m_blocks, block_m);
  } else if constexpr (fixed_block_count_v<ExtentM, TileM>) {
    constexpr nint_t MBlocks = fixed_block_count_n<ExtentM, TileM>;
    nint_t block_m = 0;
    emit_fixed_exact_rows<
        max_exact_row_a<Catalog>(), MBlocks, true>(
            ctx, n_blocks, block_m);
  } else {
    nint_t block_m = 0;
    if (n_blocks == 1) {
      emit_runtime_exact_column<max_family_a<Catalog>(), Constrained>(
          ctx, m_blocks, block_m);
    } else {
      emit_runtime_exact_rows<max_exact_row_a<Catalog>(), Constrained>(
          ctx, m_blocks, n_blocks, block_m);
    }
  }
}

// Reorder a (major, minor) invocation back to (m, n) so a single emitter
// serves both loop nestings.
enum class GridOrder { row_major, column_major };

template <typename Family, GridOrder Order,
          Tile2DMaskMode MajorMask, Tile2DMaskMode MinorMask, typename Fn>
VECOPS_ALWAYS_INLINE void invoke_ordered(
    Fn& fn, nint_t major, nint_t minor,
    nint_t active_major, nint_t active_minor) {
  if constexpr (Order == GridOrder::row_major) {
    invoke<Family, MajorMask, MinorMask>(
        fn, major, minor, active_major, active_minor);
  } else {
    invoke<Family, MinorMask, MajorMask>(
        fn, minor, major, active_minor, active_major);
  }
}

// Four-quadrant masked grid over an arbitrary rectangle: unmasked interior
// strips, then a masked minor-axis tail row, a masked major-axis tail
// column, and the doubly-masked corner. GridOrder selects which axis is the
// loop's major axis so the caller can steer prefetch locality.
template <typename Family, GridOrder Order, typename Fn>
VECOPS_ALWAYS_INLINE void emit_adaptive_grid(
    nint_t m_begin, nint_t m_extent,
    nint_t n_begin, nint_t n_extent,
    nint_t cap_m, nint_t cap_n,
    Fn& fn) {
  const nint_t major_begin =
      Order == GridOrder::row_major ? m_begin : n_begin;
  const nint_t minor_begin =
      Order == GridOrder::row_major ? n_begin : m_begin;
  const nint_t major_end = major_begin +
      (Order == GridOrder::row_major ? m_extent : n_extent);
  const nint_t minor_end = minor_begin +
      (Order == GridOrder::row_major ? n_extent : m_extent);
  const nint_t major_cap =
      Order == GridOrder::row_major ? cap_m : cap_n;
  const nint_t minor_cap =
      Order == GridOrder::row_major ? cap_n : cap_m;

  nint_t major = major_begin;
  for (; major + major_cap <= major_end; major += major_cap) {
    nint_t minor = minor_begin;
    for (; minor + minor_cap <= minor_end; minor += minor_cap) {
      invoke_ordered<Family, Order,
                     Tile2DMaskMode::unmasked,
                     Tile2DMaskMode::unmasked>(
          fn, major, minor, major_cap, minor_cap);
    }
    if (minor < minor_end) {
      invoke_ordered<Family, Order,
                     Tile2DMaskMode::unmasked,
                     Tile2DMaskMode::masked>(
          fn, major, minor, major_cap, minor_end - minor);
    }
  }
  if (major < major_end) {
    nint_t minor = minor_begin;
    for (; minor + minor_cap <= minor_end; minor += minor_cap) {
      invoke_ordered<Family, Order,
                     Tile2DMaskMode::masked,
                     Tile2DMaskMode::unmasked>(
          fn, major, minor, major_end - major, minor_cap);
    }
    if (minor < minor_end) {
      invoke_ordered<Family, Order,
                     Tile2DMaskMode::masked,
                     Tile2DMaskMode::masked>(
          fn, major, minor, major_end - major, minor_end - minor);
    }
  }
}

// Doubly-masked row emission over one tile row; NoTailN drops the tail
// handling when n-divisibility is already proven.
template <typename Family, bool NoTailN, typename Fn>
VECOPS_ALWAYS_INLINE void emit_masked_row(
    nint_t m, nint_t active_m,
    nint_t n_extent, nint_t cap_n, Fn& fn) {
  nint_t n = 0;
  if constexpr (NoTailN) {
    for (; n < n_extent; n += cap_n) {
      invoke<Family, Tile2DMaskMode::masked,
             Tile2DMaskMode::masked>(fn, m, n, active_m, cap_n);
    }
  } else {
    for (; n + cap_n <= n_extent; n += cap_n) {
      invoke<Family, Tile2DMaskMode::masked,
             Tile2DMaskMode::masked>(fn, m, n, active_m, cap_n);
    }
    if (n < n_extent) {
      invoke<Family, Tile2DMaskMode::masked,
             Tile2DMaskMode::masked>(
          fn, m, n, active_m, n_extent - n);
    }
  }
}

// Doubly-masked grid emission for the Uniform policy; NoTail flags drop the
// corresponding tail handling when divisibility is already proven.
template <typename Family, bool NoTailM, bool NoTailN, typename Fn>
VECOPS_ALWAYS_INLINE void emit_all_masked(
    nint_t m_extent, nint_t n_extent,
    nint_t cap_m, nint_t cap_n, Fn& fn) {
  nint_t m = 0;
  if constexpr (NoTailM) {
    for (; m < m_extent; m += cap_m)
      emit_masked_row<Family, NoTailN>(
          m, cap_m, n_extent, cap_n, fn);
  } else {
    for (; m + cap_m <= m_extent; m += cap_m)
      emit_masked_row<Family, NoTailN>(
          m, cap_m, n_extent, cap_n, fn);
    if (m < m_extent)
      emit_masked_row<Family, false>(
          m, m_extent - m, n_extent, cap_n, fn);
  }
}

template <typename Policy, typename Catalog,
          meta::ValueType M, meta::ValueType N,
          meta::ValueType TM, meta::ValueType TN, typename Fn>
VECOPS_ALWAYS_INLINE void run(
    const M& m_value, const N& n_value,
    const TM& tm, const TN& tn,
    Fn& fn) {
  constexpr Choice bulk_choice = select_bulk<Catalog>();
  static_assert(bulk_choice.power >= 0, "tile2d catalog has no bulk family");
  using Bulk = family_for_t<Catalog, bulk_choice.a, bulk_choice.b>;
  using BulkCapM = Capacity<TM, Bulk::a>;
  using BulkCapN = Capacity<TN, Bulk::b>;

  constexpr bool no_tail_m = has_no_tail_v<M, BulkCapM>;
  constexpr bool no_tail_n = has_no_tail_v<N, BulkCapN>;
  const nint_t m_extent = static_cast<nint_t>(m_value);
  const nint_t n_extent = static_cast<nint_t>(n_value);

  // Policy dispatch. Uniform branches off first; the remaining three
  // policies share the unmasked bulk grid machinery below.
  if constexpr (std::same_as<Policy, tile2d_policy::Uniform>) {
    // Uniform: one doubly-masked family covers the whole extent.
    constexpr Choice single_choice = select_single<Catalog>();
    using Single = family_for_t<Catalog, single_choice.a, single_choice.b>;
    using SingleCapM = Capacity<TM, Single::a>;
    using SingleCapN = Capacity<TN, Single::b>;
    constexpr bool single_no_tail_m = has_no_tail_v<M, SingleCapM>;
    constexpr bool single_no_tail_n = has_no_tail_v<N, SingleCapN>;
    const nint_t cap_m = static_cast<nint_t>(capacity<Single::a>(tm));
    const nint_t cap_n = static_cast<nint_t>(capacity<Single::b>(tn));
    emit_all_masked<Single, single_no_tail_m, single_no_tail_n>(
        m_extent, n_extent, cap_m, cap_n, fn);
  } else {

  // RowMajor / BulkTail / FourRegions all start from the same unmasked bulk
  // grid; they differ only in how they cover the remainder frame.
  const nint_t bulk_cap_m = static_cast<nint_t>(capacity<Bulk::a>(tm));
  const nint_t bulk_cap_n = static_cast<nint_t>(capacity<Bulk::b>(tn));

  // RowMajor: dedicated right/bottom/corner families whose extent on the
  // other axis matches the bulk family, so the classic row-major traversal
  // order is preserved across the boundary.
  if constexpr (std::same_as<Policy, tile2d_policy::RowMajor>) {
    constexpr Choice right_choice = no_tail_n
        ? bulk_choice
        : select_natural_edge<TileAxis::n, Catalog, Bulk, N, TN>();
    constexpr Choice bottom_choice = no_tail_m
        ? bulk_choice
        : select_natural_edge<TileAxis::m, Catalog, Bulk, M, TM>();
    constexpr Choice corner_choice = no_tail_m || no_tail_n
        ? bulk_choice
        : select_corner<Catalog, Bulk, M, N, TM, TN>();
    static_assert(right_choice.power >= 0 && bottom_choice.power >= 0 &&
                  corner_choice.power >= 0,
                  "tile2d catalog cannot cover a natural-order boundary");
    using Right = family_for_t<Catalog, right_choice.a, right_choice.b>;
    using Bottom = family_for_t<Catalog, bottom_choice.a, bottom_choice.b>;
    using Corner = family_for_t<Catalog, corner_choice.a, corner_choice.b>;
    const nint_t right_cap_m =
        static_cast<nint_t>(capacity<Right::a>(tm));
    const nint_t right_cap_n =
        static_cast<nint_t>(capacity<Right::b>(tn));
    const nint_t bottom_cap_m =
        static_cast<nint_t>(capacity<Bottom::a>(tm));
    const nint_t bottom_cap_n =
        static_cast<nint_t>(capacity<Bottom::b>(tn));
    const nint_t corner_cap_m =
        static_cast<nint_t>(capacity<Corner::a>(tm));
    const nint_t corner_cap_n =
        static_cast<nint_t>(capacity<Corner::b>(tn));
    (void)right_cap_m;
    (void)bottom_cap_n;
    (void)corner_cap_m;
    (void)corner_cap_n;

    nint_t m = 0;
    auto emit_full_row = [&](nint_t row) VECOPS_INLINE_LAMBDA {
      nint_t n = 0;
      if constexpr (no_tail_n) {
        for (; n < n_extent; n += bulk_cap_n) {
          invoke<Bulk, Tile2DMaskMode::unmasked,
                 Tile2DMaskMode::unmasked>(
              fn, row, n, bulk_cap_m, bulk_cap_n);
        }
      } else {
        for (; n + bulk_cap_n <= n_extent; n += bulk_cap_n) {
          invoke<Bulk, Tile2DMaskMode::unmasked,
                 Tile2DMaskMode::unmasked>(
              fn, row, n, bulk_cap_m, bulk_cap_n);
        }
        if (n < n_extent) {
          invoke<Right, Tile2DMaskMode::unmasked,
                 Tile2DMaskMode::masked>(
              fn, row, n, bulk_cap_m, n_extent - n);
        }
      }
    };

    if constexpr (no_tail_m) {
      for (; m < m_extent; m += bulk_cap_m) emit_full_row(m);
    } else {
      for (; m + bulk_cap_m <= m_extent; m += bulk_cap_m) emit_full_row(m);
      if (m < m_extent) {
        nint_t n = 0;
        if constexpr (no_tail_n) {
          for (; n < n_extent; n += bulk_cap_n) {
            invoke<Bottom, Tile2DMaskMode::masked,
                   Tile2DMaskMode::unmasked>(
                fn, m, n, m_extent - m, bulk_cap_n);
          }
        } else {
          for (; n + bulk_cap_n <= n_extent; n += bulk_cap_n) {
            invoke<Bottom, Tile2DMaskMode::masked,
                   Tile2DMaskMode::unmasked>(
                fn, m, n, m_extent - m, bulk_cap_n);
          }
          if (n < n_extent) {
            invoke<Corner, Tile2DMaskMode::masked,
                   Tile2DMaskMode::masked>(
                fn, m, n, m_extent - m, n_extent - n);
          }
        }
      }
    }
  } else {

  // Round the extent down to whole bulk tiles; the policies below cover the
  // remainder frame [m_bulk, m_extent) x [n_bulk, n_extent).
  nint_t m_bulk = m_extent;
  nint_t n_bulk = n_extent;
  if constexpr (!no_tail_m) m_bulk -= m_bulk % bulk_cap_m;
  if constexpr (!no_tail_n) n_bulk -= n_bulk % bulk_cap_n;
  emit_full_grid<Bulk>(0, m_bulk, 0, n_bulk,
                       bulk_cap_m, bulk_cap_n, fn);

  // BulkTail: one adaptive doubly-masked family sweeps the remainder frame
  // band by band — bottom band, right band, then the corner rectangle.
  if constexpr (std::same_as<Policy, tile2d_policy::BulkTail>) {
    constexpr Choice tail_choice = select_single<Catalog>();
    using Tail = family_for_t<Catalog, tail_choice.a, tail_choice.b>;
    const nint_t tail_cap_m =
        static_cast<nint_t>(capacity<Tail::a>(tm));
    const nint_t tail_cap_n =
        static_cast<nint_t>(capacity<Tail::b>(tn));
    if constexpr (!no_tail_m) {
      if (m_bulk < m_extent) {
        emit_adaptive_grid<Tail, GridOrder::row_major>(
            m_bulk, m_extent - m_bulk, 0, n_bulk,
            tail_cap_m, tail_cap_n, fn);
      }
    }
    if constexpr (!no_tail_n) {
      if (n_bulk < n_extent) {
        emit_adaptive_grid<Tail, GridOrder::column_major>(
            0, m_bulk, n_bulk, n_extent - n_bulk,
            tail_cap_m, tail_cap_n, fn);
      }
    }
    if constexpr (!no_tail_m && !no_tail_n) {
      if (m_bulk < m_extent && n_bulk < n_extent) {
        emit_adaptive_grid<Tail, GridOrder::row_major>(
            m_bulk, m_extent - m_bulk,
            n_bulk, n_extent - n_bulk,
            tail_cap_m, tail_cap_n, fn);
      }
    }
  } else {

  // FourRegions: grouped lower/right/corner bands over the bulk-aligned
  // frame. Unlike RowMajor, a band family only has to divide (not match)
  // the bulk width on the other axis, so an entire band is one family
  // invocation wide.
  static_assert(
      std::same_as<Policy, tile2d_policy::RowMajor> ||
      std::same_as<Policy, tile2d_policy::FourRegions> ||
      std::same_as<Policy, tile2d_policy::Uniform> ||
      std::same_as<Policy, tile2d_policy::BulkTail>,
      "unsupported tile2d policy");
  constexpr Choice lower_choice = no_tail_m
      ? bulk_choice
      : select_region_edge<TileAxis::m, Catalog, Bulk, M, TM>();
  constexpr Choice right_choice = no_tail_n
      ? bulk_choice
      : select_region_edge<TileAxis::n, Catalog, Bulk, N, TN>();
  constexpr Choice corner_choice = no_tail_m || no_tail_n
      ? bulk_choice
      : select_corner<Catalog, Bulk, M, N, TM, TN>();
  static_assert(lower_choice.power >= 0 && right_choice.power >= 0 &&
                corner_choice.power >= 0,
                "tile2d catalog cannot cover a four-region boundary");
  using Lower = family_for_t<Catalog, lower_choice.a, lower_choice.b>;
  using Right = family_for_t<Catalog, right_choice.a, right_choice.b>;
  using Corner = family_for_t<Catalog, corner_choice.a, corner_choice.b>;
  const nint_t lower_cap_n =
      static_cast<nint_t>(capacity<Lower::b>(tn));
  const nint_t right_cap_m =
      static_cast<nint_t>(capacity<Right::a>(tm));

  if constexpr (!no_tail_m) {
    if (m_bulk < m_extent) {
      const nint_t active_m = m_extent - m_bulk;
      for (nint_t n = 0; n < n_bulk; n += lower_cap_n) {
        invoke<Lower, Tile2DMaskMode::masked,
               Tile2DMaskMode::unmasked>(
            fn, m_bulk, n, active_m, lower_cap_n);
      }
    }
  }
  if constexpr (!no_tail_n) {
    if (n_bulk < n_extent) {
      const nint_t active_n = n_extent - n_bulk;
      for (nint_t m = 0; m < m_bulk; m += right_cap_m) {
        invoke<Right, Tile2DMaskMode::unmasked,
               Tile2DMaskMode::masked>(
            fn, m, n_bulk, right_cap_m, active_n);
      }
    }
  }
  if constexpr (!no_tail_m && !no_tail_n) {
    if (m_bulk < m_extent && n_bulk < n_extent) {
      invoke<Corner, Tile2DMaskMode::masked,
             Tile2DMaskMode::masked>(
          fn, m_bulk, n_bulk,
          m_extent - m_bulk, n_extent - n_bulk);
    }
  }
  }
  }
  }
}

template <meta::ValueType Value>
consteval void validate_nonnegative_extent() {
  if constexpr (meta::is_singleton_v<Value>) {
    static_assert(meta::singleton_value_v<Value> >= 0,
                  "tile2d extents must be non-negative");
  }
}

template <meta::ValueType Value>
consteval void validate_positive_tile() {
  if constexpr (meta::is_singleton_v<Value>) {
    static_assert(meta::singleton_value_v<Value> > 0,
                  "tile2d base tile sizes must be positive");
  }
}

} // namespace tile2d_details

/**
 * @brief Cover a two-dimensional extent exactly once using a kernel catalog.
 *
 * The `[0, m) x [0, n)` rectangle is partitioned into kernel invocations
 * according to `Policy`; see the file header for the callback shape and the
 * policy descriptions. Kernel families are selected at compile time from
 * `Catalog`, and tail regions are removed with `if constexpr` whenever the
 * extent metadata proves divisibility.
 *
 * @code
 * using Catalog = Tile2DKernelCatalog<
 *     Tile2DKernelFamily<2, 2, 4>, Tile2DKernelFamily<1, 2, 3>,
 *     Tile2DKernelFamily<2, 1, 3>, Tile2DKernelFamily<1, 1, 1>>;
 * tile2d(cint<64>, cint<64>, 16, 16, Catalog{},
 *        [](auto k, nint_t m, nint_t n, nint_t am, nint_t an) {
 *          // dispatch on decltype(k)::m_mask / decltype(k)::n_mask
 *        });
 * @endcode
 *
 * @tparam Policy  Traversal strategy; defaults to tile2d_policy::RowMajor.
 * @tparam M       Row extent type (meta ValueInput).
 * @tparam N       Column extent type (meta ValueInput).
 * @tparam TM      Base tile height type (meta ValueInput).
 * @tparam TN      Base tile width type (meta ValueInput).
 * @tparam Catalog Explicit or generated kernel catalog.
 * @tparam Fn      Callback type, invoked as
 *                 `fn(case_tag, m, n, active_m, active_n)`.
 *
 * @param m   Row extent in elements (non-negative).
 * @param n   Column extent in elements (non-negative).
 * @param tm  Base tile height in elements (positive).
 * @param tn  Base tile width in elements (positive).
 * @param fn  Callback invoked once per emitted kernel tile.
 *
 * @note The catalog must be a complete downward-closed set containing the
 *       1x1 family (static_assert).
 * @note Kernel capacities `a * tm` and `b * tn` must not overflow `nint_t`
 *       (VECOPS_ASSERT against the largest selected family extents).
 * @note A masked invocation may still carry a full active extent; only the
 *       unmasked tag promises full capacity.
 */
template <typename Policy = tile2d_policy::RowMajor,
          typename M, typename N, typename TM, typename TN,
          typename Catalog, typename Fn>
  requires (meta::ValueInput<M> && meta::ValueInput<N> &&
            meta::ValueInput<TM> && meta::ValueInput<TN>)
VECOPS_ALWAYS_INLINE void tile2d(
    M m, N n, TM tm, TN tn, Catalog, Fn&& fn) {
  using MV = meta::to_value_t<M>;
  using NV = meta::to_value_t<N>;
  using TMV = meta::to_value_t<TM>;
  using TNV = meta::to_value_t<TN>;
  static_assert(tile2d_details::valid_catalog<Catalog>(),
                "tile2d catalog must be a complete downward-closed set "
                "containing the 1x1 family");
  tile2d_details::validate_nonnegative_extent<MV>();
  tile2d_details::validate_nonnegative_extent<NV>();
  tile2d_details::validate_positive_tile<TMV>();
  tile2d_details::validate_positive_tile<TNV>();

  auto m_value = meta::to_value(m);
  auto n_value = meta::to_value(n);
  auto tm_value = meta::to_value(tm);
  auto tn_value = meta::to_value(tn);
  const nint_t m_int = static_cast<nint_t>(m_value);
  const nint_t n_int = static_cast<nint_t>(n_value);
  const nint_t tm_int = static_cast<nint_t>(tm_value);
  const nint_t tn_int = static_cast<nint_t>(tn_value);
  VECOPS_ASSERT(m_int >= 0 && n_int >= 0,
                "tile2d extents must be non-negative");
  VECOPS_ASSERT(tm_int > 0 && tn_int > 0,
                "tile2d base tile sizes must be positive");

  constexpr auto bulk = tile2d_details::select_bulk<Catalog>();
  VECOPS_ASSERT(
      tm_int <= std::numeric_limits<nint_t>::max() / bulk.a &&
      tn_int <= std::numeric_limits<nint_t>::max() / bulk.b,
      "tile2d kernel capacity overflows nint_t");

  auto&& fn_ref = fn;
  if constexpr (std::same_as<Policy, tile2d_policy::ExactCover>) {
    constexpr int MaxA = tile2d_details::max_family_a<Catalog>();
    constexpr int MaxB = tile2d_details::max_family_b<Catalog>();
    constexpr int BulkA = tile2d_details::max_exact_row_a<Catalog>();
    constexpr int BulkB =
        tile2d_details::max_family_b_for_a<Catalog, BulkA>();
    VECOPS_ASSERT(
        tm_int <= std::numeric_limits<nint_t>::max() / MaxA &&
        tn_int <= std::numeric_limits<nint_t>::max() / MaxB,
        "ExactCover kernel capacity overflows nint_t");
    auto exact = tile2d_details::make_exact_context<Catalog>(
        m_value, n_value, tm_value, tn_value, fn_ref);
    if constexpr (tile2d_details::meta_exact_candidate_v<
                      Catalog, MV, NV, TMV, TNV>) {
      tile2d_details::run_context_exact_cover<true>(exact);
    } else if constexpr (
        tile2d_details::has_no_tail_v<
            MV, tile2d_details::Capacity<TMV, BulkA>> &&
        tile2d_details::has_no_tail_v<
            NV, tile2d_details::Capacity<TNV, BulkB>> &&
        tile2d_details::exact_grid_mode_v<Catalog> ==
            Tile2DExactGridMode::unmasked) {
      using Bulk = tile2d_details::family_for_t<Catalog, BulkA, BulkB>;
      tile2d_details::emit_full_grid<Bulk>(
          0, m_int, 0, n_int,
          BulkA * tm_int, BulkB * tn_int, fn_ref);
    } else if constexpr (
        tile2d_details::has_no_tail_v<
            MV, tile2d_details::Capacity<TMV, BulkA>> &&
        tile2d_details::has_no_tail_v<
            NV, tile2d_details::Capacity<TNV, BulkB>> &&
        tile2d_details::exact_grid_mode_v<Catalog> ==
            Tile2DExactGridMode::exact) {
      tile2d_details::run_context_exact_grid<BulkA, BulkB>(exact);
    } else {
      tile2d_details::run_context_exact_cover<false>(exact);
    }
  } else if constexpr (
      std::same_as<Policy, tile2d_policy::FourRegions> &&
      tile2d_details::four_regions_exact_constraints_v<Catalog> &&
      (tile2d_details::has_max_block_count_v<MV, TMV> ||
       tile2d_details::has_max_block_count_v<NV, TNV>)) {
    auto exact = tile2d_details::make_exact_context<Catalog>(
        m_value, n_value, tm_value, tn_value, fn_ref);
    tile2d_details::run_context_exact_cover<true>(exact);
  } else {
    tile2d_details::run<Policy, Catalog>(
        m_value, n_value, tm_value, tn_value, fn_ref);
  }
}

/**
 * @brief Convenience entry point for a generated catalog.
 *
 * Provider is used only as a type-level source of
 * `power<A, B, MMask, NMask>()`; no runtime provider dispatch or fallback is
 * introduced. The catalog enumerates the shapes bounded by `SearchSpace`
 * once at compile time, so `SearchSpace` bounds directly control
 * instantiation cost.
 *
 * @tparam SearchSpace  Tile2DSearchSpace bounding the enumerated shapes.
 * @tparam Policy       Traversal strategy; defaults to RowMajor.
 * @tparam Provider     Static power source (type-only use).
 *
 * @param provider  Provider tag value; only its type participates.
 */
template <typename SearchSpace,
          typename Policy = tile2d_policy::RowMajor,
          typename M, typename N, typename TM, typename TN,
          typename Provider, typename Fn>
  requires (meta::ValueInput<M> && meta::ValueInput<N> &&
            meta::ValueInput<TM> && meta::ValueInput<TN>)
VECOPS_ALWAYS_INLINE void tile2d_generate(
    M m, N n, TM tm, TN tn, Provider, Fn&& fn) {
  using Catalog = Tile2DGeneratedCatalog<
      std::remove_cvref_t<Provider>, SearchSpace>;
  tile2d<Policy>(m, n, tm, tn, Catalog{}, std::forward<Fn>(fn));
}

} // namespace vecops::kernel::loop

#endif // VECOPS_KERNEL_TILE2D_H
