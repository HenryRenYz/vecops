//
// Copyright (c) vecops contributors.
//

#ifndef VECOPS_KERNEL_TILE2D_H
#define VECOPS_KERNEL_TILE2D_H

#include <algorithm>
#include <concepts>
#include <limits>
#include <type_traits>
#include <utility>

#include "vecops/Assertion.h"
#include "vecops/CoreDefs.h"
#include "vecops/Meta.h"

/**
 * @file Tile2D.h
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
 */

namespace vecops::kernel::loop {

using namespace ::vecops::meta;

enum class Tile2DMaskMode {
  unmasked,
  masked,
};

namespace tile2d_policy {

/** Classic row-major traversal with independently selected edge families. */
struct Natural {};

/** Grouped BULK, lower, right, and corner regions. */
struct FourRegions {};

/** Use one doubly-masked family for every invocation. */
struct SingleKernel {};

/** Use one unmasked bulk family and one doubly-masked boundary family. */
struct BulkAndTail {};

} // namespace tile2d_policy

/**
 * Convenience kernel-family descriptor.
 *
 * Powers are relative positive throughput scores.  A custom family may expose
 * the same `a`, `b`, and `power<MM, MN>()` interface and additionally attach
 * the actual kernel type or other compile-time metadata.
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

template <typename... Families>
struct Tile2DKernelCatalog {};

template <int MaxA, int MaxB, int MaxTiles = MaxA * MaxB>
struct Tile2DSearchSpace {
  static_assert(MaxA > 0 && MaxB > 0 && MaxTiles > 0,
                "tile2d search bounds must be positive");
  static constexpr int max_a = MaxA;
  static constexpr int max_b = MaxB;
  static constexpr int max_tiles = MaxTiles;
};

/**
 * A generated catalog queries:
 *
 * @code
 * Provider::template power<A, B, MMask, NMask>()
 * @endcode
 *
 * A negative result means the case does not exist.  A shape participates only
 * when all four mask variants exist.
 */
template <typename Provider, typename SearchSpace>
struct Tile2DGeneratedCatalog {};

template <typename Family, Tile2DMaskMode MMask, Tile2DMaskMode NMask>
struct Tile2DKernelCase {
  using family_type = Family;
  static constexpr int a = Family::a;
  static constexpr int b = Family::b;
  static constexpr Tile2DMaskMode m_mask = MMask;
  static constexpr Tile2DMaskMode n_mask = NMask;
};

namespace tile2d_details {

template <typename T>
using TileValue = ::vecops::meta::to_value_t<std::remove_cvref_t<T>>;

template <typename T>
VECOPS_ALWAYS_INLINE constexpr TileValue<T> to_tile_value(T&& value) {
  return TileValue<T>{static_cast<nint_t>(value)};
}

template <typename T>
struct FixedValue {
  static constexpr bool known = false;
  static constexpr nint_t value = 0;
};

template <nint_t N>
struct FixedValue<Const<N>> {
  static constexpr bool known = true;
  static constexpr nint_t value = N;
};

template <nint_t A, nint_t V>
struct FixedValue<Dynamic<A, V, V>> {
  static constexpr bool known = true;
  static constexpr nint_t value = V;
};

template <typename T>
inline constexpr bool fixed_value_v =
    FixedValue<std::remove_cvref_t<T>>::known;

template <typename T>
inline constexpr nint_t fixed_value_n =
    FixedValue<std::remove_cvref_t<T>>::value;

template <typename Extent, typename Step>
struct HasNoTail : std::false_type {};

template <typename Extent, typename Step>
  requires (fixed_value_v<Extent> && fixed_value_v<Step>)
struct HasNoTail<Extent, Step>
    : std::bool_constant<
          (fixed_value_n<Step> > 0) &&
          (fixed_value_n<Extent> % fixed_value_n<Step> == 0)> {};

template <nint_t A, nint_t Lo, nint_t Hi, typename Step>
  requires (!fixed_value_v<Dynamic<A, Lo, Hi>> && fixed_value_v<Step>)
struct HasNoTail<Dynamic<A, Lo, Hi>, Step>
    : std::bool_constant<
          (fixed_value_n<Step> > 0) &&
          Dynamic<A, Lo, Hi>::aligns(fixed_value_n<Step>)> {};

template <typename Step>
struct HasNoTail<Const<0>, Step> : std::true_type {};

template <typename Extent, typename Step>
inline constexpr bool has_no_tail_v =
    HasNoTail<std::remove_cvref_t<Extent>,
              std::remove_cvref_t<Step>>::value;

template <typename Tile, int Factor>
using Capacity = decltype(
    std::declval<std::remove_cvref_t<Tile>>() * cint<Factor>);

template <typename Value>
consteval nint_t effective_lower_bound() {
  using V = std::remove_cvref_t<Value>;
  if constexpr (fixed_value_v<V>) {
    return fixed_value_n<V>;
  } else if constexpr (::vecops::meta::has_lower_bound_v<V>) {
    return ::vecops::meta::lower_bound_v<V>;
  } else {
    // tile2d validates tile sizes as positive, so one is a safe semantic bound.
    return 1;
  }
}

template <typename Extent, typename Step>
consteval nint_t remainder_upper_bound() {
  using E = std::remove_cvref_t<Extent>;
  using S = std::remove_cvref_t<Step>;
  if constexpr (has_no_tail_v<E, S>) {
    return 0;
  } else if constexpr (fixed_value_v<E> && fixed_value_v<S>) {
    return fixed_value_n<E> % fixed_value_n<S>;
  } else {
    nint_t upper = -1;
    if constexpr (::vecops::meta::has_upper_bound_v<E>) {
      upper = ::vecops::meta::upper_bound_v<E>;
    }
    if constexpr (::vecops::meta::has_upper_bound_v<S>) {
      constexpr nint_t step_upper = ::vecops::meta::upper_bound_v<S>;
      if constexpr (step_upper > 0) {
        const nint_t from_step = step_upper - 1;
        upper = upper < 0 ? from_step : std::min(upper, from_step);
      }
    }
    return upper;
  }
}

template <typename CandidateTile, int CandidateFactor,
          typename MainTile, int MainFactor, typename Extent>
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

struct Choice {
  int a = 0;
  int b = 0;
  int power = -1;
  int denominator = 1;
};

constexpr bool better_choice(
    int power, int denominator, const Choice& best) {
  if (power < 0 || denominator <= 0) return false;
  if (best.power < 0) return true;
  using Wide = long long;
  return static_cast<Wide>(power) * best.denominator >
         static_cast<Wide>(best.power) * denominator;
}

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

template <typename Catalog, int A, int B>
struct FindExplicitFamily;

template <bool Matches, typename Head, typename TailCatalog, int A, int B>
struct FindExplicitFamilyStep;

template <typename Head, typename TailCatalog, int A, int B>
struct FindExplicitFamilyStep<true, Head, TailCatalog, A, B> {
  using type = Head;
};

template <typename Head, typename TailCatalog, int A, int B>
struct FindExplicitFamilyStep<false, Head, TailCatalog, A, B> {
  using type = typename FindExplicitFamily<TailCatalog, A, B>::type;
};

template <int A, int B, typename Head, typename... Tail>
struct FindExplicitFamily<Tile2DKernelCatalog<Head, Tail...>, A, B> {
  using type = typename FindExplicitFamilyStep<
      Head::a == A && Head::b == B,
      Head, Tile2DKernelCatalog<Tail...>, A, B>::type;
};

template <int A, int B, typename Last>
struct FindExplicitFamily<Tile2DKernelCatalog<Last>, A, B> {
  static_assert(Last::a == A && Last::b == B,
                "selected tile2d family is missing from catalog");
  using type = Last;
};

template <typename Catalog, int A, int B>
struct FamilyFor;

template <typename... Families, int A, int B>
struct FamilyFor<Tile2DKernelCatalog<Families...>, A, B> {
  using type = typename FindExplicitFamily<
      Tile2DKernelCatalog<Families...>, A, B>::type;
};

template <typename Provider, typename Search, int A, int B>
struct FamilyFor<Tile2DGeneratedCatalog<Provider, Search>, A, B> {
  using type = GeneratedFamily<Provider, A, B>;
};

template <typename Catalog, int A, int B>
using family_for_t = typename FamilyFor<Catalog, A, B>::type;

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

template <typename Catalog, typename Bulk, typename ExtentN, typename TileN>
consteval Choice select_natural_right() {
  return select_family<
      Tile2DMaskMode::unmasked, Tile2DMaskMode::masked, Catalog>(
      []<typename Family>() consteval {
        return Family::a == Bulk::a &&
            covers_main_remainder<TileN, Family::b,
                                  TileN, Bulk::b, ExtentN>();
      },
      []<typename Family>() consteval { return Family::b; });
}

template <typename Catalog, typename Bulk, typename ExtentM, typename TileM>
consteval Choice select_natural_bottom() {
  return select_family<
      Tile2DMaskMode::masked, Tile2DMaskMode::unmasked, Catalog>(
      []<typename Family>() consteval {
        return Family::b == Bulk::b &&
            covers_main_remainder<TileM, Family::a,
                                  TileM, Bulk::a, ExtentM>();
      },
      []<typename Family>() consteval { return Family::a; });
}

template <typename Catalog, typename Bulk,
          typename ExtentM, typename ExtentN, typename TileM, typename TileN>
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

template <typename Catalog, typename Bulk, typename ExtentM, typename TileM>
consteval Choice select_lower_region() {
  return select_family<
      Tile2DMaskMode::masked, Tile2DMaskMode::unmasked, Catalog>(
      []<typename Family>() consteval {
        return covers_main_remainder<TileM, Family::a,
                                     TileM, Bulk::a, ExtentM>();
      },
      []<typename Family>() consteval { return Family::a; });
}

template <typename Catalog, typename Bulk, typename ExtentN, typename TileN>
consteval Choice select_right_region() {
  return select_family<
      Tile2DMaskMode::unmasked, Tile2DMaskMode::masked, Catalog>(
      []<typename Family>() consteval {
        return covers_main_remainder<TileN, Family::b,
                                     TileN, Bulk::b, ExtentN>();
      },
      []<typename Family>() consteval { return Family::b; });
}

template <typename Family, Tile2DMaskMode MMask, Tile2DMaskMode NMask,
          typename Fn>
VECOPS_ALWAYS_INLINE void invoke(
    Fn& fn, nint_t m, nint_t n, nint_t active_m, nint_t active_n) {
  fn(Tile2DKernelCase<Family, MMask, NMask>{},
     m, n, active_m, active_n);
}

template <typename TileM, typename TileN, typename Family>
VECOPS_ALWAYS_INLINE auto capacities(const TileM& tm, const TileN& tn) {
  return std::pair<nint_t, nint_t>{
      static_cast<nint_t>(tm) * Family::a,
      static_cast<nint_t>(tn) * Family::b};
}

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

enum class GridOrder { row_major, column_major };

template <typename Family, GridOrder Order, typename Fn>
VECOPS_ALWAYS_INLINE void emit_adaptive_grid(
    nint_t m_begin, nint_t m_extent,
    nint_t n_begin, nint_t n_extent,
    nint_t cap_m, nint_t cap_n,
    Fn& fn) {
  const nint_t m_end = m_begin + m_extent;
  const nint_t n_end = n_begin + n_extent;

  if constexpr (Order == GridOrder::row_major) {
    nint_t m = m_begin;
    for (; m + cap_m <= m_end; m += cap_m) {
      nint_t n = n_begin;
      for (; n + cap_n <= n_end; n += cap_n) {
        invoke<Family, Tile2DMaskMode::unmasked,
               Tile2DMaskMode::unmasked>(fn, m, n, cap_m, cap_n);
      }
      if (n < n_end) {
        invoke<Family, Tile2DMaskMode::unmasked,
               Tile2DMaskMode::masked>(
            fn, m, n, cap_m, n_end - n);
      }
    }
    if (m < m_end) {
      nint_t n = n_begin;
      for (; n + cap_n <= n_end; n += cap_n) {
        invoke<Family, Tile2DMaskMode::masked,
               Tile2DMaskMode::unmasked>(
            fn, m, n, m_end - m, cap_n);
      }
      if (n < n_end) {
        invoke<Family, Tile2DMaskMode::masked,
               Tile2DMaskMode::masked>(
            fn, m, n, m_end - m, n_end - n);
      }
    }
  } else {
    nint_t n = n_begin;
    for (; n + cap_n <= n_end; n += cap_n) {
      nint_t m = m_begin;
      for (; m + cap_m <= m_end; m += cap_m) {
        invoke<Family, Tile2DMaskMode::unmasked,
               Tile2DMaskMode::unmasked>(fn, m, n, cap_m, cap_n);
      }
      if (m < m_end) {
        invoke<Family, Tile2DMaskMode::masked,
               Tile2DMaskMode::unmasked>(
            fn, m, n, m_end - m, cap_n);
      }
    }
    if (n < n_end) {
      nint_t m = m_begin;
      for (; m + cap_m <= m_end; m += cap_m) {
        invoke<Family, Tile2DMaskMode::unmasked,
               Tile2DMaskMode::masked>(
            fn, m, n, cap_m, n_end - n);
      }
      if (m < m_end) {
        invoke<Family, Tile2DMaskMode::masked,
               Tile2DMaskMode::masked>(
            fn, m, n, m_end - m, n_end - n);
      }
    }
  }
}

template <typename Family, bool NoTailM, bool NoTailN, typename Fn>
VECOPS_ALWAYS_INLINE void emit_all_masked(
    nint_t m_extent, nint_t n_extent,
    nint_t cap_m, nint_t cap_n,
    Fn& fn) {
  nint_t m = 0;
  if constexpr (NoTailM) {
    for (; m < m_extent; m += cap_m) {
      nint_t n = 0;
      if constexpr (NoTailN) {
        for (; n < n_extent; n += cap_n) {
          invoke<Family, Tile2DMaskMode::masked,
                 Tile2DMaskMode::masked>(fn, m, n, cap_m, cap_n);
        }
      } else {
        for (; n + cap_n <= n_extent; n += cap_n) {
          invoke<Family, Tile2DMaskMode::masked,
                 Tile2DMaskMode::masked>(fn, m, n, cap_m, cap_n);
        }
        if (n < n_extent) {
          invoke<Family, Tile2DMaskMode::masked,
                 Tile2DMaskMode::masked>(
              fn, m, n, cap_m, n_extent - n);
        }
      }
    }
  } else {
    for (; m + cap_m <= m_extent; m += cap_m) {
      nint_t n = 0;
      if constexpr (NoTailN) {
        for (; n < n_extent; n += cap_n) {
          invoke<Family, Tile2DMaskMode::masked,
                 Tile2DMaskMode::masked>(fn, m, n, cap_m, cap_n);
        }
      } else {
        for (; n + cap_n <= n_extent; n += cap_n) {
          invoke<Family, Tile2DMaskMode::masked,
                 Tile2DMaskMode::masked>(fn, m, n, cap_m, cap_n);
        }
        if (n < n_extent) {
          invoke<Family, Tile2DMaskMode::masked,
                 Tile2DMaskMode::masked>(
              fn, m, n, cap_m, n_extent - n);
        }
      }
    }
    if (m < m_extent) {
      nint_t n = 0;
      for (; n + cap_n <= n_extent; n += cap_n) {
        invoke<Family, Tile2DMaskMode::masked,
               Tile2DMaskMode::masked>(
            fn, m, n, m_extent - m, cap_n);
      }
      if (n < n_extent) {
        invoke<Family, Tile2DMaskMode::masked,
               Tile2DMaskMode::masked>(
            fn, m, n, m_extent - m, n_extent - n);
      }
    }
  }
}

template <typename Policy, typename Catalog,
          typename M, typename N, typename TM, typename TN, typename Fn>
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

  if constexpr (std::same_as<Policy, tile2d_policy::SingleKernel>) {
    constexpr Choice single_choice = select_single<Catalog>();
    using Single = family_for_t<Catalog, single_choice.a, single_choice.b>;
    using SingleCapM = Capacity<TM, Single::a>;
    using SingleCapN = Capacity<TN, Single::b>;
    constexpr bool single_no_tail_m = has_no_tail_v<M, SingleCapM>;
    constexpr bool single_no_tail_n = has_no_tail_v<N, SingleCapN>;
    const auto [cap_m, cap_n] = capacities<TM, TN, Single>(tm, tn);
    emit_all_masked<Single, single_no_tail_m, single_no_tail_n>(
        m_extent, n_extent, cap_m, cap_n, fn);
  } else {

  const auto [bulk_cap_m, bulk_cap_n] = capacities<TM, TN, Bulk>(tm, tn);

  if constexpr (std::same_as<Policy, tile2d_policy::Natural>) {
    constexpr Choice right_choice = no_tail_n
        ? bulk_choice
        : select_natural_right<Catalog, Bulk, N, TN>();
    constexpr Choice bottom_choice = no_tail_m
        ? bulk_choice
        : select_natural_bottom<Catalog, Bulk, M, TM>();
    constexpr Choice corner_choice = no_tail_m || no_tail_n
        ? bulk_choice
        : select_corner<Catalog, Bulk, M, N, TM, TN>();
    static_assert(right_choice.power >= 0 && bottom_choice.power >= 0 &&
                  corner_choice.power >= 0,
                  "tile2d catalog cannot cover a natural-order boundary");
    using Right = family_for_t<Catalog, right_choice.a, right_choice.b>;
    using Bottom = family_for_t<Catalog, bottom_choice.a, bottom_choice.b>;
    using Corner = family_for_t<Catalog, corner_choice.a, corner_choice.b>;
    const auto [right_cap_m, right_cap_n] = capacities<TM, TN, Right>(tm, tn);
    const auto [bottom_cap_m, bottom_cap_n] = capacities<TM, TN, Bottom>(tm, tn);
    const auto [corner_cap_m, corner_cap_n] = capacities<TM, TN, Corner>(tm, tn);
    (void)right_cap_m;
    (void)bottom_cap_n;
    (void)corner_cap_m;
    (void)corner_cap_n;

    nint_t m = 0;
    auto emit_full_row = [&](nint_t row) {
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

  nint_t m_bulk = m_extent;
  nint_t n_bulk = n_extent;
  if constexpr (!no_tail_m) m_bulk -= m_bulk % bulk_cap_m;
  if constexpr (!no_tail_n) n_bulk -= n_bulk % bulk_cap_n;
  emit_full_grid<Bulk>(0, m_bulk, 0, n_bulk,
                       bulk_cap_m, bulk_cap_n, fn);

  if constexpr (std::same_as<Policy, tile2d_policy::BulkAndTail>) {
    constexpr Choice tail_choice = select_single<Catalog>();
    using Tail = family_for_t<Catalog, tail_choice.a, tail_choice.b>;
    const auto [tail_cap_m, tail_cap_n] = capacities<TM, TN, Tail>(tm, tn);
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

  static_assert(
      std::same_as<Policy, tile2d_policy::Natural> ||
      std::same_as<Policy, tile2d_policy::FourRegions> ||
      std::same_as<Policy, tile2d_policy::SingleKernel> ||
      std::same_as<Policy, tile2d_policy::BulkAndTail>,
      "unsupported tile2d policy");
  constexpr Choice lower_choice = no_tail_m
      ? bulk_choice
      : select_lower_region<Catalog, Bulk, M, TM>();
  constexpr Choice right_choice = no_tail_n
      ? bulk_choice
      : select_right_region<Catalog, Bulk, N, TN>();
  constexpr Choice corner_choice = no_tail_m || no_tail_n
      ? bulk_choice
      : select_corner<Catalog, Bulk, M, N, TM, TN>();
  static_assert(lower_choice.power >= 0 && right_choice.power >= 0 &&
                corner_choice.power >= 0,
                "tile2d catalog cannot cover a four-region boundary");
  using Lower = family_for_t<Catalog, lower_choice.a, lower_choice.b>;
  using Right = family_for_t<Catalog, right_choice.a, right_choice.b>;
  using Corner = family_for_t<Catalog, corner_choice.a, corner_choice.b>;
  const auto [lower_cap_m, lower_cap_n] = capacities<TM, TN, Lower>(tm, tn);
  const auto [right_cap_m, right_cap_n] = capacities<TM, TN, Right>(tm, tn);
  const auto [corner_cap_m, corner_cap_n] = capacities<TM, TN, Corner>(tm, tn);

  if constexpr (!no_tail_m) {
    if (m_bulk < m_extent) {
      emit_adaptive_grid<Lower, GridOrder::row_major>(
          m_bulk, m_extent - m_bulk, 0, n_bulk,
          lower_cap_m, lower_cap_n, fn);
    }
  }
  if constexpr (!no_tail_n) {
    if (n_bulk < n_extent) {
      emit_adaptive_grid<Right, GridOrder::column_major>(
          0, m_bulk, n_bulk, n_extent - n_bulk,
          right_cap_m, right_cap_n, fn);
    }
  }
  if constexpr (!no_tail_m && !no_tail_n) {
    if (m_bulk < m_extent && n_bulk < n_extent) {
      emit_adaptive_grid<Corner, GridOrder::row_major>(
          m_bulk, m_extent - m_bulk,
          n_bulk, n_extent - n_bulk,
          corner_cap_m, corner_cap_n, fn);
    }
  }
  }
  }
  }
}

template <typename Value>
consteval void validate_nonnegative_extent() {
  if constexpr (fixed_value_v<Value>) {
    static_assert(fixed_value_n<Value> >= 0,
                  "tile2d extents must be non-negative");
  }
}

template <typename Value>
consteval void validate_positive_tile() {
  if constexpr (fixed_value_v<Value>) {
    static_assert(fixed_value_n<Value> > 0,
                  "tile2d base tile sizes must be positive");
  }
}

} // namespace tile2d_details

/**
 * Cover a two-dimensional extent using an explicit or generated kernel
 * catalog.  The default policy preserves classic row-major traversal order.
 */
template <typename Policy = tile2d_policy::Natural,
          typename M, typename N, typename TM, typename TN,
          typename Catalog, typename Fn>
VECOPS_ALWAYS_INLINE void tile2d(
    M m, N n, TM tm, TN tn, Catalog, Fn&& fn) {
  using MV = tile2d_details::TileValue<M>;
  using NV = tile2d_details::TileValue<N>;
  using TMV = tile2d_details::TileValue<TM>;
  using TNV = tile2d_details::TileValue<TN>;
  static_assert(tile2d_details::valid_catalog<Catalog>(),
                "tile2d catalog must be a complete downward-closed set "
                "containing the 1x1 family");
  tile2d_details::validate_nonnegative_extent<MV>();
  tile2d_details::validate_nonnegative_extent<NV>();
  tile2d_details::validate_positive_tile<TMV>();
  tile2d_details::validate_positive_tile<TNV>();

  auto m_value = tile2d_details::to_tile_value(m);
  auto n_value = tile2d_details::to_tile_value(n);
  auto tm_value = tile2d_details::to_tile_value(tm);
  auto tn_value = tile2d_details::to_tile_value(tn);
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
  tile2d_details::run<Policy, Catalog>(
      m_value, n_value, tm_value, tn_value, fn_ref);
}

/**
 * Convenience entry point for a generated catalog.  Provider is used only as
 * a type-level source of `power<A, B, MMask, NMask>()`; no runtime provider
 * dispatch or fallback is introduced.
 */
template <typename SearchSpace,
          typename Policy = tile2d_policy::Natural,
          typename M, typename N, typename TM, typename TN,
          typename Provider, typename Fn>
VECOPS_ALWAYS_INLINE void tile2d_generate(
    M m, N n, TM tm, TN tn, Provider, Fn&& fn) {
  using Catalog = Tile2DGeneratedCatalog<
      std::remove_cvref_t<Provider>, SearchSpace>;
  tile2d<Policy>(m, n, tm, tn, Catalog{}, std::forward<Fn>(fn));
}

} // namespace vecops::kernel::loop

#endif // VECOPS_KERNEL_TILE2D_H
