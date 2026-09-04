#include <algorithm>
#include <array>
#include <cstddef>
#include <limits>
#include <tuple>
#include <type_traits>
#include <vector>

#include <gtest/gtest.h>

#include "vecops/kernel/Tile2D.h"

namespace {

using vecops::nint_t;
using namespace vecops::meta;
namespace hop = vecops::kernel::loop;

static_assert(is_singleton_v<Const<7>>);
static_assert(is_singleton_v<Dynamic<1, 7, 7>>);
static_assert(singleton_value_v<Dynamic<1, 7, 7>> == 7);
static_assert(!is_singleton_v<Dynamic<1, 7, 8>>);

using F11 = hop::Tile2DKernelFamily<1, 1, 1, 1, 1, 4>;
using F12 = hop::Tile2DKernelFamily<1, 2, 2, 3, 8, 4>;
using F21 = hop::Tile2DKernelFamily<2, 1, 2, 8, 3, 4>;
using F22 = hop::Tile2DKernelFamily<2, 2, 10, 10, 10, 4>;
using Catalog = hop::Tile2DKernelCatalog<F11, F12, F21, F22>;

struct Area4Provider {
  static constexpr bool four_regions_exact_constraints = true;
  static constexpr nint_t exact_meta_block_limit = 8;

  template <int A, int B, hop::Tile2DMaskMode, hop::Tile2DMaskMode>
  static consteval int power() {
    if constexpr (A * B <= 4) {
      constexpr int imbalance = A > B ? A - B : B - A;
      return 100 * A * B + 4 * (A + B) - 8 * imbalance;
    } else {
      return -1;
    }
  }
};

using Area4Catalog = hop::Tile2DGeneratedCatalog<
    Area4Provider, hop::Tile2DSearchSpace<4, 4, 4>>;

struct Area4Max3Provider : Area4Provider {
  static constexpr hop::Tile2DExactGridMode exact_grid_mode =
      hop::Tile2DExactGridMode::unmasked;
  static constexpr nint_t exact_meta_block_limit =
      std::numeric_limits<nint_t>::max();
};

using Area4Max3Catalog = hop::Tile2DGeneratedCatalog<
    Area4Max3Provider, hop::Tile2DSearchSpace<3, 3, 4>>;

struct Area8Provider {
  template <int A, int B, hop::Tile2DMaskMode, hop::Tile2DMaskMode>
  static consteval int power() {
    return A * B <= 8 ? 100 * A * B + A + B : -1;
  }
};

using Area8Catalog = hop::Tile2DGeneratedCatalog<
    Area8Provider, hop::Tile2DSearchSpace<4, 4, 8>>;

struct IrregularProvider {
  template <int A, int B, hop::Tile2DMaskMode, hop::Tile2DMaskMode>
  static consteval int power() {
    return A * B <= 10 ? 100 * A * B + A + B : -1;
  }
};

using IrregularCatalog = hop::Tile2DGeneratedCatalog<
    IrregularProvider, hop::Tile2DSearchSpace<5, 5, 10>>;
using ExactCover = hop::tile2d_policy::ExactCover;
using ExactCoverRuntimeUnmasked =
    hop::tile2d_policy::ExactCoverRuntimeUnmasked;

static_assert(hop::tile2d_details::meta_exact_candidate_v<
              Area4Catalog, Const<35>, Const<53>, Const<16>, Const<16>>);
static_assert(!hop::tile2d_details::meta_exact_candidate_v<
              Area4Catalog, Const<256>, Const<256>, Const<16>, Const<16>>);

struct Visit {
  int a;
  int b;
  hop::Tile2DMaskMode m_mask;
  hop::Tile2DMaskMode n_mask;
  nint_t m;
  nint_t n;
  nint_t active_m;
  nint_t active_n;

  auto operator<=>(const Visit&) const = default;
};

template <typename Policy, typename M, typename N, typename TM, typename TN,
          typename CatalogT = Catalog>
std::vector<Visit> run_and_check_cover(
    M m, N n, TM tm, TN tn, CatalogT catalog = {}) {
  const nint_t m_int = static_cast<nint_t>(m);
  const nint_t n_int = static_cast<nint_t>(n);
  const nint_t tm_int = static_cast<nint_t>(tm);
  const nint_t tn_int = static_cast<nint_t>(tn);
  std::vector<int> cover(
      static_cast<std::size_t>(m_int * n_int), 0);
  std::vector<Visit> visits;

  hop::tile2d<Policy>(m, n, tm, tn, catalog,
      [&]<typename Case>(Case, nint_t off_m, nint_t off_n,
                         nint_t active_m, nint_t active_n) {
        const nint_t capacity_m = Case::a * tm_int;
        const nint_t capacity_n = Case::b * tn_int;
        EXPECT_GT(active_m, 0);
        EXPECT_GT(active_n, 0);
        EXPECT_LE(active_m, capacity_m);
        EXPECT_LE(active_n, capacity_n);
        EXPECT_GE(off_m, 0);
        EXPECT_GE(off_n, 0);
        EXPECT_LE(off_m + active_m, m_int);
        EXPECT_LE(off_n + active_n, n_int);
        if constexpr (Case::m_mask == hop::Tile2DMaskMode::unmasked) {
          EXPECT_EQ(active_m, capacity_m);
        }
        if constexpr (Case::n_mask == hop::Tile2DMaskMode::unmasked) {
          EXPECT_EQ(active_n, capacity_n);
        }
        visits.push_back(Visit{
            Case::a, Case::b, Case::m_mask, Case::n_mask,
            off_m, off_n, active_m, active_n});
        for (nint_t i = off_m; i < off_m + active_m; ++i) {
          for (nint_t j = off_n; j < off_n + active_n; ++j) {
            ++cover[static_cast<std::size_t>(i * n_int + j)];
          }
        }
      });

  EXPECT_TRUE(std::all_of(
      cover.begin(), cover.end(), [](int count) { return count == 1; }));
  return visits;
}

template <typename Policy>
void check_exhaustive_dynamic_cover() {
  for (nint_t tm = 1; tm <= 3; ++tm) {
    for (nint_t tn = 1; tn <= 3; ++tn) {
      for (nint_t m = 0; m <= 13; ++m) {
        for (nint_t n = 0; n <= 13; ++n) {
          run_and_check_cover<Policy>(Any{m}, Any{n}, Any{tm}, Any{tn});
        }
      }
    }
  }
}

TEST(Tile2DTest, RowMajorExhaustivelyCoversDynamicSmallShapes) {
  check_exhaustive_dynamic_cover<hop::tile2d_policy::RowMajor>();
}

TEST(Tile2DTest, FourRegionsExhaustivelyCoversDynamicSmallShapes) {
  check_exhaustive_dynamic_cover<hop::tile2d_policy::FourRegions>();
}

TEST(Tile2DTest, UniformExhaustivelyCoversDynamicSmallShapes) {
  check_exhaustive_dynamic_cover<hop::tile2d_policy::Uniform>();
}

TEST(Tile2DTest, BulkTailExhaustivelyCoversDynamicSmallShapes) {
  check_exhaustive_dynamic_cover<hop::tile2d_policy::BulkTail>();
}

TEST(Tile2DTest, ExactCoverArea4ExhaustivelyCoversDynamicSmallShapes) {
  for (nint_t tm = 1; tm <= 3; ++tm) {
    for (nint_t tn = 1; tn <= 3; ++tn) {
      for (nint_t m = 0; m <= 13; ++m) {
        for (nint_t n = 0; n <= 13; ++n) {
          run_and_check_cover<ExactCover>(
              Any{m}, Any{n}, Any{tm}, Any{tn}, Area4Catalog{});
        }
      }
    }
  }
}

TEST(Tile2DTest, ExactCoverArea4Max3ExhaustivelyCoversDynamicSmallShapes) {
  for (nint_t tm = 1; tm <= 3; ++tm) {
    for (nint_t tn = 1; tn <= 3; ++tn) {
      for (nint_t m = 0; m <= 13; ++m) {
        for (nint_t n = 0; n <= 13; ++n) {
          const auto visits = run_and_check_cover<
              ExactCover>(
              Any{m}, Any{n}, Any{tm}, Any{tn}, Area4Max3Catalog{});
          for (const auto& visit : visits) {
            EXPECT_LE(visit.a, 3);
            EXPECT_LE(visit.b, 3);
            EXPECT_LE(visit.a * visit.b, 4);
          }
        }
      }
    }
  }
}

TEST(Tile2DTest, ExactCoverRuntimeUnmaskedPromotesOnlyAlignedGrid) {
  const auto aligned = run_and_check_cover<ExactCoverRuntimeUnmasked>(
      Any{8}, Any{8}, cint<1>, cint<1>, Area4Max3Catalog{});
  ASSERT_EQ(aligned.size(), 16u);
  EXPECT_TRUE(std::all_of(aligned.begin(), aligned.end(), [](const Visit& v) {
    return v.m_mask == hop::Tile2DMaskMode::unmasked &&
        v.n_mask == hop::Tile2DMaskMode::unmasked;
  }));
  const auto ragged = run_and_check_cover<ExactCoverRuntimeUnmasked>(
      Any{7}, Any{5}, cint<1>, cint<1>, Area4Max3Catalog{});
  EXPECT_TRUE(std::any_of(ragged.begin(), ragged.end(), [](const Visit& v) {
    return v.m_mask == hop::Tile2DMaskMode::masked ||
        v.n_mask == hop::Tile2DMaskMode::masked;
  }));
}

TEST(Tile2DTest, ExactCoverArea8ExhaustivelyCoversDynamicSmallShapes) {
  for (nint_t tm = 1; tm <= 3; ++tm) {
    for (nint_t tn = 1; tn <= 3; ++tn) {
      for (nint_t m = 0; m <= 13; ++m) {
        for (nint_t n = 0; n <= 13; ++n) {
          const auto visits =
              run_and_check_cover<ExactCover>(
                  Any{m}, Any{n}, Any{tm}, Any{tn}, Area8Catalog{});
          for (const auto& visit : visits) {
            EXPECT_LE(visit.a * visit.b, 8);
          }
        }
      }
    }
  }
}

TEST(Tile2DTest, ExactCoverUsesArbitraryCatalogBounds) {
  bool used_five_by_two = false;
  for (nint_t tm = 1; tm <= 3; ++tm) {
    for (nint_t tn = 1; tn <= 3; ++tn) {
      for (nint_t m = 0; m <= 13; ++m) {
        for (nint_t n = 0; n <= 13; ++n) {
          const auto visits = run_and_check_cover<ExactCover>(
              Any{m}, Any{n}, Any{tm}, Any{tn}, IrregularCatalog{});
          for (const auto& visit : visits) {
            EXPECT_LE(visit.a, 5);
            EXPECT_LE(visit.b, 5);
            EXPECT_LE(visit.a * visit.b, 10);
            used_five_by_two |= visit.a == 5 && visit.b == 2;
          }
        }
      }
    }
  }
  EXPECT_TRUE(used_five_by_two);
}

struct UnmaskedTwoByTwoOnly {
  template <typename Case>
    requires (Case::a == 2 && Case::b == 2 && !Case::exact_blocks &&
              Case::m_mask == hop::Tile2DMaskMode::unmasked &&
              Case::n_mask == hop::Tile2DMaskMode::unmasked)
  void operator()(Case, nint_t, nint_t, nint_t, nint_t) const {}

  template <typename Case>
  void operator()(Case, nint_t, nint_t, nint_t, nint_t) const = delete;
};

struct ExactFourByTwoOnly {
  template <typename Case>
    requires (Case::a == 4 && Case::b == 2 && Case::exact_blocks)
  void operator()(Case, nint_t, nint_t, nint_t, nint_t) const {}

  template <typename Case>
  void operator()(Case, nint_t, nint_t, nint_t, nint_t) const = delete;
};

TEST(Tile2DTest, ExactCoverArea4Max3PrunesFixedAndAlignedFamilies) {
  hop::tile2d<ExactCover>(
      cint<64>, cint<96>, cint<16>, cint<16>, Area4Max3Catalog{},
      UnmaskedTwoByTwoOnly{});
  hop::tile2d<ExactCover>(
      dyn<32>(64), dyn<32>(96), cint<16>, cint<16>, Area4Max3Catalog{},
      UnmaskedTwoByTwoOnly{});
}

TEST(Tile2DTest, ExactCoverArea4Max3FixedRaggedExtentsStayMasked) {
  const auto visits = run_and_check_cover<
      ExactCover>(
          cint<35>, cint<53>, cint<16>, cint<16>, Area4Max3Catalog{});
  ASSERT_EQ(visits.size(), 4u);
  EXPECT_TRUE(std::all_of(visits.begin(), visits.end(), [](const Visit& v) {
    return v.m_mask == hop::Tile2DMaskMode::masked &&
        v.n_mask == hop::Tile2DMaskMode::masked;
  }));
}

TEST(Tile2DTest, ExactCoverArea8PrunesFixedFamilies) {
  hop::tile2d<ExactCover>(
      cint<8>, cint<12>, cint<2>, cint<3>, Area8Catalog{},
      ExactFourByTwoOnly{});
}

TEST(Tile2DTest, ExactCoverArea4UsesOnlyExistingLogicalBlocks) {
  hop::tile2d<ExactCover>(
      Any{5}, Any{7}, Any{2}, Any{3}, Area4Catalog{},
      []<typename Case>(Case, auto, auto, auto, auto) {
        static_assert(Case::exact_blocks);
        static_assert(Case::m_mask == hop::Tile2DMaskMode::masked);
        static_assert(Case::n_mask == hop::Tile2DMaskMode::masked);
      });
  const auto broad =
      run_and_check_cover<ExactCover>(
          Any{5}, Any{7}, Any{2}, Any{3}, Area4Catalog{});
  ASSERT_EQ(broad.size(), 3u);
  EXPECT_EQ(std::tie(broad[0].a, broad[0].b), (std::tuple{2, 2}));
  EXPECT_EQ(std::tie(broad[1].a, broad[1].b), (std::tuple{2, 1}));
  EXPECT_EQ(std::tie(broad[2].a, broad[2].b), (std::tuple{1, 3}));

  const auto narrow =
      run_and_check_cover<ExactCover>(
          Any{11}, Any{2}, Any{2}, Any{3}, Area4Catalog{});
  ASSERT_EQ(narrow.size(), 2u);
  EXPECT_EQ(std::tie(narrow[0].a, narrow[0].b), (std::tuple{4, 1}));
  EXPECT_EQ(std::tie(narrow[1].a, narrow[1].b), (std::tuple{2, 1}));
}

struct OneLogicalRowOnly {
  template <typename Case>
    requires (Case::a == 1)
  void operator()(Case, nint_t, nint_t, nint_t, nint_t) const {
    static_assert(Case::exact_blocks);
  }

  template <typename Case>
    requires (Case::a != 1)
  void operator()(Case, nint_t, nint_t, nint_t, nint_t) const = delete;
};

struct OneLogicalColumnOnly {
  template <typename Case>
    requires (Case::b == 1)
  void operator()(Case, nint_t, nint_t, nint_t, nint_t) const {
    static_assert(Case::exact_blocks);
  }

  template <typename Case>
    requires (Case::b != 1)
  void operator()(Case, nint_t, nint_t, nint_t, nint_t) const = delete;
};

struct AtMostTwoRowsOnly {
  template <typename Case>
    requires (Case::a <= 2 && Case::b == 1)
  void operator()(Case, nint_t, nint_t, nint_t, nint_t) const {
    static_assert(Case::exact_blocks);
  }

  template <typename Case>
    requires (Case::a > 2 || Case::b != 1)
  void operator()(Case, nint_t, nint_t, nint_t, nint_t) const = delete;
};

struct AtMostThreeColumnsOnly {
  template <typename Case>
    requires (Case::a == 1 && Case::b <= 3)
  void operator()(Case, nint_t, nint_t, nint_t, nint_t) const {
    static_assert(Case::exact_blocks);
  }

  template <typename Case>
    requires (Case::a != 1 || Case::b > 3)
  void operator()(Case, nint_t, nint_t, nint_t, nint_t) const = delete;
};

TEST(Tile2DTest, FourRegionsUsesFixedArea4RowPlan) {
  const auto visits = run_and_check_cover<hop::tile2d_policy::FourRegions>(
      cint<16>, cint<1152>, cint<16>, cint<16>, Area4Catalog{});
  ASSERT_EQ(visits.size(), 18u);
  EXPECT_TRUE(std::all_of(visits.begin(), visits.end(), [](const Visit& v) {
    return v.a == 1 && v.b == 4 &&
        v.m_mask == hop::Tile2DMaskMode::masked &&
        v.n_mask == hop::Tile2DMaskMode::masked;
  }));
}

TEST(Tile2DTest, FourRegionsUsesFixedArea4ColumnPlan) {
  const auto visits = run_and_check_cover<hop::tile2d_policy::FourRegions>(
      cint<64>, cint<16>, cint<16>, cint<16>, Area4Catalog{});
  ASSERT_EQ(visits.size(), 1u);
  EXPECT_EQ(std::tie(visits[0].a, visits[0].b), (std::tuple{4, 1}));
  EXPECT_TRUE(visits[0].m_mask == hop::Tile2DMaskMode::masked);
  EXPECT_TRUE(visits[0].n_mask == hop::Tile2DMaskMode::masked);
}

TEST(Tile2DTest, FourRegionsPrunesFamiliesFromOneConstrainedAxis) {
  hop::tile2d<hop::tile2d_policy::FourRegions>(
      Dynamic<1, 0, 16>{16}, Any{1152}, cint<16>, cint<16>,
      Area4Catalog{}, OneLogicalRowOnly{});
  hop::tile2d<hop::tile2d_policy::FourRegions>(
      Any{1152}, Dynamic<1, 0, 16>{16}, cint<16>, cint<16>,
      Area4Catalog{}, OneLogicalColumnOnly{});
  hop::tile2d<hop::tile2d_policy::FourRegions>(
      Dynamic<1, 0, 32>{31}, cint<16>, cint<16>, cint<16>,
      Area4Catalog{}, AtMostTwoRowsOnly{});
  hop::tile2d<hop::tile2d_policy::FourRegions>(
      cint<16>, Dynamic<1, 0, 48>{47}, cint<16>, cint<16>,
      Area4Catalog{}, AtMostThreeColumnsOnly{});
}

TEST(Tile2DTest, ExactCoverArea4PrunesFamiliesFromMetaConstraints) {
  hop::tile2d<ExactCover>(
      Dynamic<1, 0, 16>{16}, Any{1152}, cint<16>, cint<16>,
      Area4Catalog{}, OneLogicalRowOnly{});
  hop::tile2d<ExactCover>(
      Any{1152}, Dynamic<1, 0, 16>{16}, cint<16>, cint<16>,
      Area4Catalog{}, OneLogicalColumnOnly{});
}

TEST(Tile2DTest, FourRegionsFixedArea4PrunesRuntimeRemainderFamilies) {
  const auto visits = run_and_check_cover<hop::tile2d_policy::FourRegions>(
      cint<10>, cint<14>, cint<2>, cint<3>, Area4Catalog{});
  ASSERT_EQ(visits.size(), 8u);
  EXPECT_EQ(std::tie(visits[0].a, visits[0].b), (std::tuple{2, 2}));
  EXPECT_EQ(std::tie(visits[2].a, visits[2].b), (std::tuple{2, 1}));
  EXPECT_EQ(std::tie(visits[6].a, visits[6].b), (std::tuple{1, 4}));
  EXPECT_EQ(std::tie(visits[7].a, visits[7].b), (std::tuple{1, 1}));
}

TEST(Tile2DTest, Area4ScorePrefersSquareBulk) {
  constexpr auto uu = hop::Tile2DMaskMode::unmasked;
  static_assert(
      Area4Provider::power<2, 2, uu, uu>() >
      Area4Provider::power<1, 4, uu, uu>());
  static_assert(
      Area4Provider::power<2, 2, uu, uu>() >
      Area4Provider::power<4, 1, uu, uu>());

  const auto visits = run_and_check_cover<hop::tile2d_policy::FourRegions>(
      Any{24}, Any{36}, Any{2}, Any{3}, Area4Catalog{});
  ASSERT_FALSE(visits.empty());
  EXPECT_EQ(std::tie(visits.front().a, visits.front().b),
            (std::tuple{2, 2}));
}

TEST(Tile2DTest, RowMajorPreservesClassicInterleavedOrder) {
  const auto visits = run_and_check_cover<hop::tile2d_policy::RowMajor>(
      cint<10>, cint<14>, cint<2>, cint<3>);
  const std::vector<std::pair<nint_t, nint_t>> actual = [&] {
    std::vector<std::pair<nint_t, nint_t>> result;
    for (const auto& visit : visits) result.emplace_back(visit.m, visit.n);
    return result;
  }();
  EXPECT_EQ(actual, (std::vector<std::pair<nint_t, nint_t>>{
      {0, 0}, {0, 6}, {0, 12},
      {4, 0}, {4, 6}, {4, 12},
      {8, 0}, {8, 6}, {8, 12}}));

  EXPECT_EQ(visits[2].a, 2);
  EXPECT_EQ(visits[2].b, 1);
  EXPECT_EQ(visits[6].a, 1);
  EXPECT_EQ(visits[6].b, 2);
  EXPECT_EQ(visits[8].a, 1);
  EXPECT_EQ(visits[8].b, 1);
}

TEST(Tile2DTest, FourRegionsGroupsBulkLowerRightAndCorner) {
  const auto visits = run_and_check_cover<hop::tile2d_policy::FourRegions>(
      cint<10>, cint<14>, cint<2>, cint<3>);
  ASSERT_EQ(visits.size(), 9u);
  EXPECT_EQ(std::tie(visits[0].m, visits[0].n), (std::tuple{nint_t{0}, nint_t{0}}));
  EXPECT_EQ(std::tie(visits[3].m, visits[3].n), (std::tuple{nint_t{4}, nint_t{6}}));
  // Lower region follows the four bulk calls and uses the selected 1x2 family.
  EXPECT_EQ(std::tie(visits[4].a, visits[4].b), (std::tuple{1, 2}));
  EXPECT_EQ(visits[4].m_mask, hop::Tile2DMaskMode::masked);
  EXPECT_EQ(visits[4].n_mask, hop::Tile2DMaskMode::unmasked);
  EXPECT_EQ(std::tie(visits[4].m, visits[4].n), (std::tuple{nint_t{8}, nint_t{0}}));
  // Right region is column-major and uses 2x1.
  EXPECT_EQ(std::tie(visits[6].a, visits[6].b), (std::tuple{2, 1}));
  EXPECT_EQ(visits[6].m_mask, hop::Tile2DMaskMode::unmasked);
  EXPECT_EQ(visits[6].n_mask, hop::Tile2DMaskMode::masked);
  EXPECT_EQ(std::tie(visits[6].m, visits[6].n), (std::tuple{nint_t{0}, nint_t{12}}));
  EXPECT_EQ(std::tie(visits[7].m, visits[7].n), (std::tuple{nint_t{4}, nint_t{12}}));
  EXPECT_EQ(std::tie(visits[8].a, visits[8].b), (std::tuple{1, 1}));
  EXPECT_EQ(visits[8].m_mask, hop::Tile2DMaskMode::masked);
  EXPECT_EQ(visits[8].n_mask, hop::Tile2DMaskMode::masked);
  EXPECT_EQ(std::tie(visits[8].m, visits[8].n), (std::tuple{nint_t{8}, nint_t{12}}));
}

struct FourRegionsOnlyExpectedCases {
  using UU = hop::Tile2DKernelCase<
      F22, hop::Tile2DMaskMode::unmasked,
      hop::Tile2DMaskMode::unmasked>;
  using MU = hop::Tile2DKernelCase<
      F22, hop::Tile2DMaskMode::masked,
      hop::Tile2DMaskMode::unmasked>;
  using UM = hop::Tile2DKernelCase<
      F22, hop::Tile2DMaskMode::unmasked,
      hop::Tile2DMaskMode::masked>;
  using MM = hop::Tile2DKernelCase<
      F22, hop::Tile2DMaskMode::masked,
      hop::Tile2DMaskMode::masked>;

  void operator()(UU, nint_t, nint_t, nint_t, nint_t) const {}
  void operator()(MU, nint_t, nint_t, nint_t, nint_t) const {}
  void operator()(UM, nint_t, nint_t, nint_t, nint_t) const {}
  void operator()(MM, nint_t, nint_t, nint_t, nint_t) const {}

  template <typename Case>
  void operator()(Case, nint_t, nint_t, nint_t, nint_t) const = delete;
};

TEST(Tile2DTest, FourRegionsInstantiatesOneCasePerRegion) {
  hop::tile2d<hop::tile2d_policy::FourRegions>(
      Any{13}, Any{17}, Any{2}, Any{3}, Catalog{},
      FourRegionsOnlyExpectedCases{});
}

TEST(Tile2DTest, ZeroExtentDoesNotInvokeCallback) {
  int calls = 0;
  hop::tile2d(cint<0>, cint<17>, cint<2>, cint<3>, Catalog{},
              [&](auto, auto, auto, auto, auto) { ++calls; });
  hop::tile2d(cint<17>, cint<0>, cint<2>, cint<3>, Catalog{},
              [&](auto, auto, auto, auto, auto) { ++calls; });
  EXPECT_EQ(calls, 0);
}

#if defined(VECOPS_DEBUG)
TEST(Tile2DDeathTest, RejectsNegativeRuntimeExtent) {
  EXPECT_DEATH(
      hop::tile2d(Any{-1}, Any{8}, Any{1}, Any{1}, Catalog{},
                  [](auto, auto, auto, auto, auto) {}),
      "extents must be non-negative");
}

TEST(Tile2DDeathTest, RejectsNonPositiveRuntimeTile) {
  EXPECT_DEATH(
      hop::tile2d(Any{8}, Any{8}, Any{0}, Any{1}, Catalog{},
                  [](auto, auto, auto, auto, auto) {}),
      "base tile sizes must be positive");
}
#endif

TEST(Tile2DTest, ConstDivisibilityInstantiatesOnlyUnmaskedCases) {
  int calls = 0;
  hop::tile2d(cint<8>, cint<12>, cint<2>, cint<3>, Catalog{},
      [&]<typename Case>(Case, auto, auto, auto, auto) {
        static_assert(Case::m_mask == hop::Tile2DMaskMode::unmasked);
        static_assert(Case::n_mask == hop::Tile2DMaskMode::unmasked);
        ++calls;
      });
  EXPECT_EQ(calls, 4);
}

TEST(Tile2DTest, DynamicAlignmentPrunesBothTailAxes) {
  int calls = 0;
  hop::tile2d(dyn<8>(16), dyn<16>(32), cint<2>, cint<4>, Catalog{},
      [&]<typename Case>(Case, auto, auto, auto, auto) {
        static_assert(Case::m_mask == hop::Tile2DMaskMode::unmasked);
        static_assert(Case::n_mask == hop::Tile2DMaskMode::unmasked);
        ++calls;
      });
  EXPECT_EQ(calls, 16);
}

TEST(Tile2DTest, DynamicAlignmentPrunesOnlyProvenAxis) {
  bool saw_m_mask = false;
  hop::tile2d(Any{10}, dyn<16>(32), cint<2>, cint<4>, Catalog{},
      [&]<typename Case>(Case, auto, auto, auto, auto) {
        static_assert(Case::n_mask == hop::Tile2DMaskMode::unmasked);
        if constexpr (Case::m_mask == hop::Tile2DMaskMode::masked) {
          saw_m_mask = true;
        }
      });
  EXPECT_TRUE(saw_m_mask);
}

TEST(Tile2DTest, FixedDynamicTileValueParticipatesInDivisibilityProof) {
  int calls = 0;
  hop::tile2d(dyn<8>(16), dyn<16>(32),
              dyn<2, 2, 2>(2), dyn<4, 4, 4>(4), Catalog{},
      [&]<typename Case>(Case, auto, auto, auto, auto) {
        static_assert(Case::m_mask == hop::Tile2DMaskMode::unmasked);
        static_assert(Case::n_mask == hop::Tile2DMaskMode::unmasked);
        ++calls;
      });
  EXPECT_EQ(calls, 16);
}

struct GeneratedProvider {
  template <int A, int B, hop::Tile2DMaskMode MMask,
            hop::Tile2DMaskMode NMask>
  static consteval int power() {
    if constexpr (A > 2 || B > 2) {
      return -1;
    } else if constexpr (A == 2 && B == 2) {
      return MMask == hop::Tile2DMaskMode::unmasked &&
                     NMask == hop::Tile2DMaskMode::unmasked
          ? 10 : 4;
    } else if constexpr (A == 2 && B == 1) {
      return NMask == hop::Tile2DMaskMode::masked ? 8 : 2;
    } else if constexpr (A == 1 && B == 2) {
      return MMask == hop::Tile2DMaskMode::masked ? 8 : 2;
    } else {
      return MMask == hop::Tile2DMaskMode::masked &&
                     NMask == hop::Tile2DMaskMode::masked
          ? 4 : 1;
    }
  }
};

using GeneratedCatalog = hop::Tile2DGeneratedCatalog<
    GeneratedProvider, hop::Tile2DSearchSpace<4, 4, 4>>;

TEST(Tile2DTest, GeneratedCatalogUsesSameSchedulingCore) {
  const auto visits =
      run_and_check_cover<hop::tile2d_policy::RowMajor>(
          cint<10>, cint<14>, cint<2>, cint<3>, GeneratedCatalog{});
  ASSERT_EQ(visits.size(), 9u);
  EXPECT_EQ(std::tie(visits[0].a, visits[0].b), (std::tuple{2, 2}));
  EXPECT_EQ(std::tie(visits[2].a, visits[2].b), (std::tuple{2, 1}));
  EXPECT_EQ(std::tie(visits[6].a, visits[6].b), (std::tuple{1, 2}));
  EXPECT_EQ(std::tie(visits[8].a, visits[8].b), (std::tuple{1, 1}));
}

TEST(Tile2DTest, GeneratedConvenienceEntryPointNeedsNoExplicitCatalog) {
  int calls = 0;
  hop::tile2d_generate<hop::Tile2DSearchSpace<4, 4, 4>>(
      cint<8>, cint<12>, cint<2>, cint<3>, GeneratedProvider{},
      [&]<typename Case>(Case, auto, auto, auto, auto) {
        static_assert(Case::m_mask == hop::Tile2DMaskMode::unmasked);
        static_assert(Case::n_mask == hop::Tile2DMaskMode::unmasked);
        ++calls;
      });
  EXPECT_EQ(calls, 4);
}

} // namespace

// Kept externally visible for optimized assembly inspection.  Both Dynamic
// alignments prove divisibility by the selected 2x2 family's capacities
// (2 rows and 32 columns), so no masked callback is even instantiated.
extern "C" VECOPS_NOINLINE nint_t tile2d_aligned_scheduler_probe(
    nint_t M, nint_t N) {
  nint_t checksum = 0;
  hop::tile2d(dyn<4>(M), dyn<32>(N), cint<1>, cint<16>, Catalog{},
      [&]<typename Case>(Case, nint_t m, nint_t n,
                         nint_t active_m, nint_t active_n) {
        static_assert(Case::m_mask == hop::Tile2DMaskMode::unmasked);
        static_assert(Case::n_mask == hop::Tile2DMaskMode::unmasked);
        checksum += m + n + active_m + active_n;
      });
  return checksum;
}
