//
// Copyright (c) vecops contributors.
//

#include <gtest/gtest.h>

#include <array>
#include <vector>

#include "vecops/matmul/details/tiled/LoopNest.h"
#include "vecops/matmul/details/planning/FamilySelector.h"
#include "vecops/matmul/details/tiled/PolicyTraits.h"
#include "vecops/matmul/Tiling.h"
#include "vecops/ops/Matmul.h"
#include "vecops/platform/CacheInfo.h"

namespace {

using namespace vecops;

struct AutomaticFamilyConfig {
  using FamilySelection = matmul::family_selection::Automatic;
};

struct PreferredGenericFamilyConfig {
  using FamilySelection = matmul::family_selection::Prefer<
      matmul::kernel_family::GenericTiled>;
};

struct RequiredGenericFamilyConfig {
  using FamilySelection = matmul::family_selection::Require<
      matmul::kernel_family::GenericTiled>;
};

struct RequiredRuntimeQuantFamilyConfig {
  using FamilySelection = matmul::family_selection::Require<
      matmul::kernel_family::RuntimeQuantInt8>;
};

static_assert(std::same_as<
              matmul::details::selected_family_t<AutomaticFamilyConfig>,
              matmul::kernel_family::WholeProblem>);
static_assert(std::same_as<
              matmul::details::selected_family_t<PreferredGenericFamilyConfig>,
              matmul::kernel_family::GenericTiled>);
static_assert(std::same_as<
              matmul::details::selected_family_t<RequiredGenericFamilyConfig>,
              matmul::kernel_family::GenericTiled>);
static_assert(std::same_as<
              matmul::details::selected_family_t<
                  RequiredRuntimeQuantFamilyConfig>,
              matmul::kernel_family::RuntimeQuantInt8>);
static_assert(
    matmul::details::family_selection_mode_v<PreferredGenericFamilyConfig> ==
    matmul::family_selection::Mode::prefer);
static_assert(
    matmul::details::family_selection_mode_v<RequiredGenericFamilyConfig> ==
    matmul::family_selection::Mode::require);
static_assert(
    matmul::kernel_family::Info<
        matmul::kernel_family::WholeProblem>::name == "whole_problem");
static_assert(
    matmul::kernel_family::Info<
        matmul::kernel_family::GenericTiled>::name == "generic_tiled");
static_assert(matmul::kernel_family::Info<
                  matmul::kernel_family::General>::name == "general");
static_assert(matmul::kernel_family::Info<
                  matmul::kernel_family::SmallVector>::name ==
              "small_vector");
static_assert(matmul::kernel_family::Info<
                  matmul::kernel_family::RuntimeQuantInt8>::name ==
              "runtime_quant_int8");
static_assert(matmul::kernel_family::Info<
                  matmul::kernel_family::PackedMMLA>::name == "packed_mmla");
static_assert(matmul::kernel_family::Info<
                  matmul::kernel_family::PackedTail>::name == "packed_tail");

#if defined(ARCH_X86_FAMILY)
using ExplicitGenericTuning = matmul::GenericTiledTuning<
    matmul::CacheTiling<meta::Const<16>, meta::Const<16>, meta::Const<32>>,
    matmul::loop_order::KMN>;
using TunedAutomaticConfig = ops::MatmulConfig<
    matmul::AMX_BF16F32, matmul::family_selection::Automatic,
    kernel::matmul_policy::Automatic, ExplicitGenericTuning>;
static_assert(std::same_as<
              matmul::details::selected_family_t<TunedAutomaticConfig>,
              matmul::kernel_family::WholeProblem>);
#endif

using UnitTiling = matmul::CacheTiling<
    meta::Const<1>, meta::Const<1>, meta::Const<1>>;

#if defined(HAS_SME)
using SMEAutomaticTiling = matmul::AutomaticCacheTilingFor<
    matmul::SME_F32F32>;
static_assert(meta::ValueType<typename SMEAutomaticTiling::MTile>);
static_assert(meta::ValueType<typename SMEAutomaticTiling::NTile>);
static_assert(meta::ValueType<typename SMEAutomaticTiling::KTile>);
#endif

template <typename Order>
std::vector<int> traversal_trace() {
  std::vector<int> result;
  matmul::details::LoopNest<Order, UnitTiling>::run(
      meta::cint<2>, meta::cint<2>, meta::cint<2>, UnitTiling{},
      [&](const auto& block, bool, bool) {
        result.push_back(static_cast<int>(
            block.m_origin * 100 + block.n_origin * 10 + block.k_origin));
      });
  return result;
}

template <typename Order, std::size_t N>
void expect_order(const std::array<int, N>& expected) {
  EXPECT_EQ(traversal_trace<Order>(),
            std::vector<int>(expected.begin(), expected.end()));
}

TEST(MatmulLoopNestTest, SupportsEveryCacheLoopPermutation) {
  expect_order<matmul::loop_order::MNK>(
      std::array{0, 1, 10, 11, 100, 101, 110, 111});
  expect_order<matmul::loop_order::MKN>(
      std::array{0, 10, 1, 11, 100, 110, 101, 111});
  expect_order<matmul::loop_order::NMK>(
      std::array{0, 1, 100, 101, 10, 11, 110, 111});
  expect_order<matmul::loop_order::NKM>(
      std::array{0, 100, 1, 101, 10, 110, 11, 111});
  expect_order<matmul::loop_order::KMN>(
      std::array{0, 10, 100, 110, 1, 11, 101, 111});
  expect_order<matmul::loop_order::KNM>(
      std::array{0, 100, 10, 110, 1, 101, 11, 111});
}

using AutomaticTiling = matmul::CacheTiling<
    meta::Const<8>, meta::Const<8>, meta::Const<8>>;
using EnabledMTiling = matmul::CacheTiling<
    meta::Const<8>, meta::Const<8>, meta::Const<8>,
    matmul::CacheLoopMode::enabled>;
using DisabledKTiling = matmul::CacheTiling<
    meta::Const<1>, meta::Const<1>, meta::Const<1>,
    matmul::CacheLoopMode::automatic,
    matmul::CacheLoopMode::automatic,
    matmul::CacheLoopMode::disabled>;

using AutomaticNest = matmul::details::LoopNest<
    matmul::loop_order::MNK, AutomaticTiling>;
using EnabledMNest = matmul::details::LoopNest<
    matmul::loop_order::MNK, EnabledMTiling>;

static_assert(!AutomaticNest::template generates_loop<
              matmul::Axis::M, meta::Const<4>, meta::Const<4>,
              meta::Const<4>>);
static_assert(AutomaticNest::template generates_loop<
              matmul::Axis::M, meta::Any, meta::Const<4>, meta::Const<4>>);
static_assert(EnabledMNest::template generates_loop<
              matmul::Axis::M, meta::Const<4>, meta::Const<4>,
              meta::Const<4>>);

using AlignedTiling = matmul::CacheTiling<
    meta::Dynamic<16, 16>, meta::Dynamic<16, 16>,
    meta::Dynamic<32, 32>>;
static_assert(matmul::details::LoopNest<
              matmul::loop_order::MNK, AlignedTiling>::template generates_loop<
                  matmul::Axis::K, meta::Any, meta::Any, meta::Any>);
static_assert(!matmul::details::LoopNest<
              matmul::loop_order::MNK, AlignedTiling>::template generates_loop<
                  matmul::Axis::K, meta::Const<16>, meta::Const<16>,
                  meta::Const<16>>);

using AlwaysAutomatic = matmul::PackingPolicy<
    matmul::PackingMode::always, matmul::PackingExtent::automatic>;
static_assert(matmul::details::resolved_packing_extent_v<
                  AlwaysAutomatic, matmul::Operand::A,
                  matmul::loop_order::MNK> ==
              matmul::PackingExtent::cache_k);
static_assert(matmul::details::resolved_packing_extent_v<
                  AlwaysAutomatic, matmul::Operand::A,
                  matmul::loop_order::KMN> ==
              matmul::PackingExtent::full_k);
static_assert(matmul::details::resolved_packing_extent_v<
                  AlwaysAutomatic, matmul::Operand::B,
                  matmul::loop_order::NKM> ==
              matmul::PackingExtent::cache_k);

TEST(MatmulLoopNestTest, DisabledAxisDoesNotGenerateBlocks) {
  int calls = 0;
  matmul::details::LoopNest<
      matmul::loop_order::MNK, DisabledKTiling>::run(
          meta::cint<1>, meta::cint<1>, meta::cint<5>, DisabledKTiling{},
          [&](const auto& block, bool first_k, bool last_k) {
            ++calls;
            EXPECT_EQ(static_cast<nint_t>(block.k), 5);
            EXPECT_TRUE(first_k);
            EXPECT_TRUE(last_k);
          });
  EXPECT_EQ(calls, 1);
}

TEST(MatmulLoopNestTest, ReportsKBlockPhases) {
  using Tiling = matmul::CacheTiling<
      meta::Const<8>, meta::Const<8>, meta::Const<2>>;
  std::vector<std::pair<bool, bool>> phases;
  std::vector<nint_t> active;
  matmul::details::LoopNest<matmul::loop_order::MNK, Tiling>::run(
      meta::cint<1>, meta::cint<1>, meta::cint<5>, Tiling{},
      [&](const auto& block, bool first_k, bool last_k) {
        phases.emplace_back(first_k, last_k);
        active.push_back(static_cast<nint_t>(block.k));
      });
  EXPECT_EQ(phases, (std::vector<std::pair<bool, bool>>{
                        {true, false}, {false, false}, {false, true}}));
  EXPECT_EQ(active, (std::vector<nint_t>{2, 2, 1}));
}

TEST(MatmulLoopNestTest, ExposesKPhasesAsCompileTimeTypes) {
  using Tiling = matmul::CacheTiling<
      meta::Const<8>, meta::Const<8>, meta::Const<2>>;
  std::vector<int> phases;
  matmul::details::LoopNest<matmul::loop_order::KMN, Tiling>::run_phased(
      meta::cint<1>, meta::cint<1>, meta::cint<5>, Tiling{},
      [&](const auto&, auto phase) {
        if constexpr (decltype(phase)::first_k &&
                      !decltype(phase)::last_k) {
          phases.push_back(0);
        } else if constexpr (!decltype(phase)::first_k &&
                             !decltype(phase)::last_k) {
          phases.push_back(1);
        } else if constexpr (!decltype(phase)::first_k &&
                             decltype(phase)::last_k) {
          phases.push_back(2);
        } else {
          phases.push_back(3);
        }
      });
  EXPECT_EQ(phases, (std::vector<int>{0, 1, 2}));
}

TEST(CacheInfoTest, ReportsUsableHierarchy) {
  const auto info = platform::detect_cache_info();
  EXPECT_GT(info.line_bytes, 0);
  EXPECT_GE(info.l1d_bytes, info.line_bytes);
  EXPECT_GE(info.l2_bytes, info.l1d_bytes);
  EXPECT_GE(info.l3_bytes, info.l2_bytes);
}

struct FakeCacheInfoProvider {
  platform::CacheInfo value;
  const platform::CacheInfo& operator()() const { return value; }
};

#if defined(HAS_AMX_TILE)
TEST(CacheInfoTest, ConfigAcceptsInjectedProviderAndFallbackFlag) {
  using Config = ops::MatmulConfig<
      matmul::AMX_BF16F32, matmul::family_selection::Automatic,
      kernel::matmul_policy::Automatic, matmul::GenericTiledTuning<>,
      FakeCacheInfoProvider>;
  Config config{
      .cache_info_provider = FakeCacheInfoProvider{platform::CacheInfo{
          .line_bytes = 64,
          .l1d_bytes = 16 * 1024,
          .l2_bytes = 256 * 1024,
          .l3_bytes = 2 * 1024 * 1024,
          .fallback = true}}};
  const auto tiling = matmul::resolve_cache_tiling(config);
  EXPECT_TRUE(config.cache_info_provider().fallback);
  EXPECT_EQ(static_cast<nint_t>(tiling.kc) % 32, 0);
}
#endif

#if defined(HAS_AMX_TILE)
TEST(CacheInfoTest, AutomaticTilingUsesDocumentedBudgets) {
  platform::CacheInfo cache{
      .line_bytes = 64,
      .l1d_bytes = 48 * 1024,
      .l2_bytes = 2 * 1024 * 1024,
      .l3_bytes = 32 * 1024 * 1024,
      .fallback = false};
  const auto tiling = matmul::DefaultTilingPolicy<
      matmul::AMX_BF16F32>::select(cache);
  EXPECT_GT(static_cast<nint_t>(tiling.mc), 0);
  EXPECT_GT(static_cast<nint_t>(tiling.nc), 0);
  EXPECT_GT(static_cast<nint_t>(tiling.kc), 0);
  EXPECT_EQ(static_cast<nint_t>(tiling.mc) % 16, 0);
  EXPECT_EQ(static_cast<nint_t>(tiling.nc) % 16, 0);
  EXPECT_EQ(static_cast<nint_t>(tiling.kc) % 32, 0);
}
#endif

} // namespace
