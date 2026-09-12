//
// Copyright (c) vecops contributors.
//

#include <gtest/gtest.h>

#include <array>
#include <map>
#include <set>
#include <vector>

#include "vecops/matmul/details/tiled/LoopNest.h"
#include "vecops/matmul/details/planning/FamilySelector.h"
#include "vecops/matmul/details/kernel/RuntimeDispatch.h"
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
                  matmul::kernel_family::PackedDot>::name == "packed_dot");
static_assert(matmul::kernel_family::Info<
                  matmul::kernel_family::ResidualSplit>::name == "residual_split");

using RuntimeShape = kernel::matmul_details::SmallVectorShape;
using namespace kernel::matmul_details::runtime_dispatch_rules;

static_assert(amx_small_vector_profitable(
    RuntimeShape::amx_bf16, 1, 16, 65));
static_assert(!amx_small_vector_profitable(
    RuntimeShape::amx_bf16, 19, 21, 65));
static_assert(amx_small_vector_profitable(
    RuntimeShape::amx_i8_mixed_sign, 8, 8, 128));
static_assert(sme_small_vector_profitable(
    RuntimeShape::sme_f16, 1, 16, 17));
static_assert(!sme_small_vector_profitable(
    RuntimeShape::sme_f16, 1, 17, 17));
static_assert(amx_residual_split_profitable(17, 33, 1024));
static_assert(amx_residual_split_profitable(33, 17, 1024));
static_assert(!amx_residual_split_profitable(19, 33, 1024));

TEST(MatmulRuntimeDispatchTest, SharedRuntimeRulesMatchConstexprRules) {
  using namespace kernel::matmul_details;
  for (const auto shape : {
           RuntimeShape::amx_bf16,
           RuntimeShape::amx_i8_same_sign,
           RuntimeShape::amx_i8_mixed_sign}) {
    for (const nint_t m : {nint_t{1}, nint_t{8}, nint_t{19}, nint_t{128}})
      for (const nint_t n : {nint_t{1}, nint_t{16}, nint_t{64}, nint_t{129}})
        for (const nint_t k : {nint_t{1}, nint_t{64}, nint_t{65}, nint_t{256}})
          EXPECT_EQ(
              runtime_amx_small_vector_profitable(shape, m, n, k),
              amx_small_vector_profitable(shape, m, n, k));
  }
  for (const nint_t m : {nint_t{17}, nint_t{19}, nint_t{20}})
    for (const nint_t n : {nint_t{33}, nint_t{47}, nint_t{49}})
      EXPECT_EQ(
          kernel::matmul_details::runtime_amx_residual_split_profitable(
              m, n, 1024),
          amx_residual_split_profitable(m, n, 1024));
}

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

struct CustomPackingTuning {
  using APacking = matmul::packing_policy::RequireOutside;
  using BPacking = matmul::packing_policy::InsideOnly;
};

struct UnknownSitePackingPolicy {
  static constexpr auto allowed_sites =
      static_cast<matmul::PackingSite>(1u << 7);
  static constexpr auto requirement =
      matmul::PackingRequirement::profitable;
  static constexpr auto prepared_input =
      matmul::PreparedInputRequirement::optional;
};

struct InvalidPackingTuning {
  using APacking = UnknownSitePackingPolicy;
  using BPacking = matmul::packing_policy::Any;
};

#if defined(ARCH_X86_FAMILY)
using PackingConfigTestAtom = matmul::AMX_BF16F32;
#elif defined(HAS_SME)
using PackingConfigTestAtom = matmul::SME_F32F32;
#endif

#if defined(ARCH_X86_FAMILY) || defined(HAS_SME)
using CustomPackingConfig = ops::MatmulConfigWithPacking<
    PackingConfigTestAtom, CustomPackingTuning,
    matmul::family_selection::Automatic,
    kernel::matmul_policy::Automatic,
    matmul::GenericTiledTuning<>,
    platform::SystemCacheInfoProvider,
    false>;
using LegacySixParameterConfig = ops::MatmulConfig<
    PackingConfigTestAtom,
    matmul::family_selection::Automatic,
    kernel::matmul_policy::Automatic,
    matmul::GenericTiledTuning<>,
    platform::SystemCacheInfoProvider,
    false>;
#endif

static_assert(matmul::allows_packing_site(
    matmul::packing_policy::Any::allowed_sites,
    matmul::PackingSite::inside_reuse_loop));
static_assert(matmul::allows_packing_site(
    matmul::packing_policy::Any::allowed_sites,
    matmul::PackingSite::outside_reuse_loop));
static_assert(!matmul::allows_packing_site(
    matmul::packing_policy::Any::allowed_sites,
    matmul::PackingSite::none));
static_assert(!matmul::allows_packing_site(
    matmul::packing_policy::InsideOnly::allowed_sites,
    matmul::PackingSite::outside_reuse_loop));
static_assert(matmul::allows_packing_site(
    matmul::packing_policy::OutsideOnly::allowed_sites,
    matmul::PackingSite::outside_reuse_loop));
static_assert(
    matmul::packing_policy::Disabled::allowed_sites ==
    matmul::PackingSite::none);
static_assert(
    matmul::packing_policy::RequireInside::requirement ==
    matmul::PackingRequirement::required);
static_assert(
    matmul::packing_policy::RequireOutside::requirement ==
    matmul::PackingRequirement::required);
static_assert(
    matmul::packing_policy::CallerPreparedOnly::prepared_input ==
    matmul::PreparedInputRequirement::required);
static_assert(matmul::OperandPackingPolicyType<
              matmul::packing_policy::Any>);
static_assert(matmul::OperandPackingPolicyType<
              matmul::packing_policy::InsideOnly>);
static_assert(matmul::OperandPackingPolicyType<
              matmul::packing_policy::OutsideOnly>);
static_assert(matmul::OperandPackingPolicyType<
              matmul::packing_policy::Disabled>);
static_assert(matmul::OperandPackingPolicyType<
              matmul::packing_policy::RequireInside>);
static_assert(matmul::OperandPackingPolicyType<
              matmul::packing_policy::RequireOutside>);
static_assert(matmul::OperandPackingPolicyType<
              matmul::packing_policy::CallerPreparedOnly>);
static_assert(matmul::MatmulPackingTuningType<CustomPackingTuning>);
static_assert(!matmul::OperandPackingPolicyType<UnknownSitePackingPolicy>);
static_assert(!matmul::MatmulPackingTuningType<InvalidPackingTuning>);
static_assert(std::same_as<
    matmul::details::config_packing_tuning_t<AutomaticFamilyConfig>,
    matmul::MatmulPackingTuning<>>);
#if defined(ARCH_X86_FAMILY) || defined(HAS_SME)
static_assert(std::same_as<
    typename ops::MatmulConfig<PackingConfigTestAtom>::PackingTuning,
    matmul::MatmulPackingTuning<>>);
static_assert(std::same_as<
    typename CustomPackingConfig::PackingTuning,
    CustomPackingTuning>);
static_assert(!CustomPackingConfig::enable_swap_ab);
static_assert(!LegacySixParameterConfig::enable_swap_ab);
static_assert(std::same_as<
    typename LegacySixParameterConfig::PackingTuning,
    matmul::MatmulPackingTuning<>>);
static_assert(std::is_empty_v<
              ops::MatmulConfig<PackingConfigTestAtom>>);
#endif
static_assert(matmul::details::spatial_traversal_order_v<
                  matmul::loop_order::MNK> ==
              kernel::loop::Tile2DTraversalOrder::m_major);
static_assert(matmul::details::spatial_traversal_order_v<
                  matmul::loop_order::MKN> ==
              kernel::loop::Tile2DTraversalOrder::m_major);
static_assert(matmul::details::spatial_traversal_order_v<
                  matmul::loop_order::NMK> ==
              kernel::loop::Tile2DTraversalOrder::n_major);
static_assert(matmul::details::spatial_traversal_order_v<
                  matmul::loop_order::NKM> ==
              kernel::loop::Tile2DTraversalOrder::n_major);
static_assert(matmul::details::spatial_traversal_order_v<
                  matmul::loop_order::KMN> ==
              kernel::loop::Tile2DTraversalOrder::m_major);
static_assert(matmul::details::spatial_traversal_order_v<
                  matmul::loop_order::KNM> ==
              kernel::loop::Tile2DTraversalOrder::n_major);
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
static_assert(matmul::details::has_panel_lifetime_site_v<
              matmul::Operand::A, matmul::loop_order::MKN>);
static_assert(matmul::details::has_panel_lifetime_site_v<
              matmul::Operand::A, matmul::loop_order::KMN>);
static_assert(matmul::details::has_panel_lifetime_site_v<
              matmul::Operand::B, matmul::loop_order::NKM>);
static_assert(matmul::details::has_panel_lifetime_site_v<
              matmul::Operand::B, matmul::loop_order::KNM>);
static_assert(!matmul::details::has_panel_lifetime_site_v<
              matmul::Operand::A, matmul::loop_order::NKM>);
using LegacyCachePacking = matmul::PackingPolicy<
    matmul::PackingMode::automatic, matmul::PackingExtent::cache_k>;
using ForcedOutsideWithoutPanel = matmul::details::ResolvedInternalPackingPolicy<
    LegacyCachePacking, matmul::packing_policy::OutsideOnly,
    matmul::Operand::B, matmul::loop_order::MNK>;
using ForcedInsideFromFull = matmul::details::ResolvedInternalPackingPolicy<
    matmul::PackingPolicy<matmul::PackingMode::always,
                          matmul::PackingExtent::full_k>,
    matmul::packing_policy::InsideOnly,
    matmul::Operand::A, matmul::loop_order::MNK>;
using RequiredOutsidePolicy = matmul::details::ResolvedInternalPackingPolicy<
    matmul::PackingPolicy<>, matmul::packing_policy::RequireOutside,
    matmul::Operand::B, matmul::loop_order::NKM>;
using DisabledInternalPolicy = matmul::details::ResolvedInternalPackingPolicy<
    matmul::PackingPolicy<matmul::PackingMode::always>,
    matmul::packing_policy::Disabled,
    matmul::Operand::B, matmul::loop_order::NKM>;
static_assert(ForcedOutsideWithoutPanel::extent ==
              matmul::PackingExtent::full_k);
static_assert(ForcedInsideFromFull::extent ==
              matmul::PackingExtent::cache_k);
static_assert(RequiredOutsidePolicy::mode == matmul::PackingMode::always);
static_assert(DisabledInternalPolicy::mode == matmul::PackingMode::never);

template <typename Order>
constexpr bool output_accumulator_keeps_logical_origins() {
  return matmul::details::accumulator_block_origin<
             true, matmul::Axis::M, Order>(17) == 17 &&
         matmul::details::accumulator_block_origin<
             true, matmul::Axis::N, Order>(19) == 19;
}

static_assert(output_accumulator_keeps_logical_origins<
              matmul::loop_order::MNK>());
static_assert(output_accumulator_keeps_logical_origins<
              matmul::loop_order::MKN>());
static_assert(output_accumulator_keeps_logical_origins<
              matmul::loop_order::NMK>());
static_assert(output_accumulator_keeps_logical_origins<
              matmul::loop_order::NKM>());
static_assert(output_accumulator_keeps_logical_origins<
              matmul::loop_order::KMN>());
static_assert(output_accumulator_keeps_logical_origins<
              matmul::loop_order::KNM>());

static_assert(matmul::details::accumulator_block_origin<
              false, matmul::Axis::M, matmul::loop_order::MKN>(17) == 0);
static_assert(matmul::details::accumulator_block_origin<
              false, matmul::Axis::N, matmul::loop_order::MKN>(19) == 19);
static_assert(matmul::details::accumulator_block_origin<
              false, matmul::Axis::M, matmul::loop_order::NKM>(17) == 17);
static_assert(matmul::details::accumulator_block_origin<
              false, matmul::Axis::N, matmul::loop_order::NKM>(19) == 0);

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

struct HookedLoopState {
  int generation;
};

struct MiddleAxisLifetimeHook {
  int* hook_calls;
  int* active_lifetimes;

  template <int Depth, matmul::Axis Target, typename Context,
            typename Phase, typename State, typename Continue>
  void operator()(const Context&, const Phase&, const State& state,
                  Continue&& continuation) const {
    if constexpr (Depth == 1) {
      static_assert(Target == matmul::Axis::K);
      ++*hook_calls;
      ++*active_lifetimes;
      continuation(HookedLoopState{*hook_calls});
      --*active_lifetimes;
    } else {
      continuation(state);
    }
  }
};

TEST(MatmulLoopNestTest, StatefulHookOwnsMiddleToInnerLoopLifetime) {
  using Tiling = matmul::CacheTiling<
      meta::Const<2>, meta::Const<2>, meta::Const<2>>;
  int hook_calls = 0;
  int active_lifetimes = 0;
  int leaves = 0;
  MiddleAxisLifetimeHook hook{&hook_calls, &active_lifetimes};
  matmul::details::LoopNest<matmul::loop_order::NKM, Tiling>::
      run_phased_with_state(
          meta::cint<4>, meta::cint<6>, meta::cint<4>, Tiling{}, 0, hook,
          [&](const auto&, const auto&, const HookedLoopState& state) {
            ++leaves;
            EXPECT_EQ(active_lifetimes, 1);
            EXPECT_GT(state.generation, 0);
          });
  // N x K establishes six lifetimes; each encloses both M blocks.
  EXPECT_EQ(hook_calls, 6);
  EXPECT_EQ(leaves, 12);
  EXPECT_EQ(active_lifetimes, 0);
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

TEST(MatmulLoopNestTest, PreservesSingletonExtentMetadata) {
  using Tiling = matmul::CacheTiling<
      meta::Const<2>, meta::Const<2>, meta::Const<2>,
      matmul::CacheLoopMode::enabled,
      matmul::CacheLoopMode::enabled,
      matmul::CacheLoopMode::enabled>;
  using Singleton = meta::Dynamic<1, 1, 1>;
  int calls = 0;
  matmul::details::LoopNest<matmul::loop_order::MNK, Tiling>::run_phased(
      Singleton{1}, meta::cint<1>, meta::cint<1>, Tiling{},
      [&](const auto& block, auto phase) {
        ++calls;
        static_assert(std::same_as<
                      std::remove_cvref_t<decltype(block.m)>, Singleton>);
        static_assert(std::same_as<
                      std::remove_cvref_t<decltype(block.n)>, meta::Const<1>>);
        static_assert(std::same_as<
                      std::remove_cvref_t<decltype(block.k)>, meta::Const<1>>);
        static_assert(matmul::details::static_kernel_phase_v<
                      decltype(phase)>);
      });
  EXPECT_EQ(calls, 1);
}

TEST(MatmulLoopNestTest, PreservesBoundedKBlockMetadataWithoutTypeSplitting) {
  using Tiling = matmul::CacheTiling<
      meta::Const<8>, meta::Const<8>, meta::Const<2>>;
  using KBlock = meta::Dynamic<1, 0, 2>;
  int full_blocks = 0;
  int tail_blocks = 0;
  matmul::details::LoopNest<matmul::loop_order::MNK, Tiling>::run_phased(
      meta::cint<1>, meta::cint<1>, meta::Any{5}, Tiling{},
      [&](const auto& block, auto) {
        using K = std::remove_cvref_t<decltype(block.k)>;
        if constexpr (std::same_as<K, KBlock>) {
          if (static_cast<nint_t>(block.k) == 2)
            ++full_blocks;
          else if (static_cast<nint_t>(block.k) == 1)
            ++tail_blocks;
        }
      });
  EXPECT_EQ(full_blocks, 2);
  EXPECT_EQ(tail_blocks, 1);
}

TEST(MatmulLoopNestTest, StatefulTraversalPreservesSpatialBlockMetadata) {
  using Tiling = matmul::CacheTiling<
      meta::Const<2>, meta::Const<8>, meta::Const<8>>;
  using MBlock = meta::Dynamic<1, 1, 2>;
  int full_blocks = 0;
  int tail_blocks = 0;
  auto pass_through = []<int, matmul::Axis>(
                          const auto&, const auto&, const auto& state,
                          auto&& continuation) {
    continuation(state);
  };
  matmul::details::LoopNest<matmul::loop_order::MNK, Tiling>::
      run_phased_with_state(
          meta::Any{5}, meta::cint<1>, meta::cint<1>, Tiling{}, 0,
          pass_through,
          [&](const auto& block, auto, int) {
            using M = std::remove_cvref_t<decltype(block.m)>;
            if constexpr (std::same_as<M, MBlock>) {
              if (static_cast<nint_t>(block.m) == 2)
                ++full_blocks;
              else if (static_cast<nint_t>(block.m) == 1)
                ++tail_blocks;
            }
          });
  EXPECT_EQ(full_blocks, 2);
  EXPECT_EQ(tail_blocks, 1);
}

TEST(MatmulLoopNestTest, ZeroKStillEmitsSemanticOutputPhase) {
  using Tiling = matmul::CacheTiling<
      meta::Const<8>, meta::Const<8>, meta::Const<2>>;
  int calls = 0;
  matmul::details::LoopNest<matmul::loop_order::NKM, Tiling>::run_phased(
      meta::cint<3>, meta::cint<5>, meta::cint<0>, Tiling{},
      [&](const auto& block, auto phase) {
        ++calls;
        EXPECT_EQ(static_cast<nint_t>(block.k), 0);
        static_assert(matmul::details::static_kernel_phase_v<
                      decltype(phase)>);
        EXPECT_TRUE(matmul::details::phase_first_k(phase));
        EXPECT_TRUE(matmul::details::phase_last_k(phase));
      });
  EXPECT_EQ(calls, 1);
}

TEST(MatmulLoopNestTest, SharesOneRuntimeTypeAcrossSplitKPhases) {
  using Tiling = matmul::CacheTiling<
      meta::Const<8>, meta::Const<8>, meta::Const<2>>;
  std::vector<int> phases;
  matmul::details::LoopNest<matmul::loop_order::KMN, Tiling>::run_phased(
      meta::cint<1>, meta::cint<1>, meta::cint<5>, Tiling{},
      [&](const auto&, auto phase) {
        if constexpr (matmul::details::static_kernel_phase_v<
                          decltype(phase)>) {
          static_assert(decltype(phase)::first_k &&
                        decltype(phase)::last_k);
          phases.push_back(3);
        } else {
          static_assert(std::same_as<
                        decltype(phase),
                        matmul::details::DynamicKernelPhase>);
          if (phase.first_k && !phase.last_k) {
            phases.push_back(0);
          } else if (!phase.first_k && !phase.last_k) {
            phases.push_back(1);
          } else if (!phase.first_k && phase.last_k) {
            phases.push_back(2);
          } else {
            phases.push_back(3);
          }
        }
      });
  EXPECT_EQ(phases, (std::vector<int>{0, 1, 2}));
}

struct KPhaseRecord {
  nint_t origin;
  nint_t extent;
  bool first;
  bool last;
  bool is_static;

  auto operator<=>(const KPhaseRecord&) const = default;
};

template <typename Order>
void expect_k_phase_boundaries_for_order(nint_t logical_k) {
  using Tiling = matmul::CacheTiling<
      meta::Const<2>, meta::Const<2>, meta::Const<2>,
      matmul::CacheLoopMode::enabled,
      matmul::CacheLoopMode::enabled,
      matmul::CacheLoopMode::enabled>;
  std::map<std::pair<nint_t, nint_t>, std::vector<KPhaseRecord>> records;
  matmul::details::LoopNest<Order, Tiling>::run_phased(
      meta::Any{3}, meta::Any{3}, meta::Any{logical_k}, Tiling{},
      [&](const auto& block, auto phase) {
        records[{block.m_origin, block.n_origin}].push_back(KPhaseRecord{
            block.k_origin, static_cast<nint_t>(block.k),
            matmul::details::phase_first_k(phase),
            matmul::details::phase_last_k(phase),
            matmul::details::static_kernel_phase_v<decltype(phase)>});
      });

  std::vector<KPhaseRecord> expected;
  if (logical_k <= 2) {
    expected.push_back({0, logical_k, true, true, true});
  } else {
    for (nint_t origin = 0; origin < logical_k; origin += 2) {
      expected.push_back({
          origin, std::min<nint_t>(2, logical_k - origin),
          origin == 0, origin + 2 >= logical_k, false});
    }
  }
  ASSERT_EQ(records.size(), 4u);
  for (nint_t m_origin : {nint_t{0}, nint_t{2}}) {
    for (nint_t n_origin : {nint_t{0}, nint_t{2}}) {
      const auto key = std::pair{m_origin, n_origin};
      EXPECT_EQ(records.at(key), expected)
          << "m_origin=" << m_origin << " n_origin=" << n_origin
          << " logical_k=" << logical_k;
    }
  }
}

template <typename Order>
void expect_zero_output_extent_skips_order() {
  using Tiling = matmul::CacheTiling<
      meta::Const<2>, meta::Const<2>, meta::Const<2>,
      matmul::CacheLoopMode::enabled,
      matmul::CacheLoopMode::enabled,
      matmul::CacheLoopMode::enabled>;
  for (const auto [m, n] : {
           std::pair<nint_t, nint_t>{0, 3},
           std::pair<nint_t, nint_t>{3, 0},
           std::pair<nint_t, nint_t>{0, 0}}) {
    int calls = 0;
    matmul::details::LoopNest<Order, Tiling>::run_phased(
        meta::Any{m}, meta::Any{n}, meta::Any{5}, Tiling{},
        [&](const auto&, auto) { ++calls; });
    EXPECT_EQ(calls, 0) << "m=" << m << " n=" << n;
  }
}

TEST(MatmulLoopNestTest, KPhaseBoundariesAreCorrectForEveryLoopOrder) {
  for (nint_t logical_k : {nint_t{0}, nint_t{1}, nint_t{2}, nint_t{3},
                           nint_t{4}, nint_t{5}, nint_t{6}}) {
    expect_k_phase_boundaries_for_order<matmul::loop_order::MNK>(logical_k);
    expect_k_phase_boundaries_for_order<matmul::loop_order::MKN>(logical_k);
    expect_k_phase_boundaries_for_order<matmul::loop_order::NMK>(logical_k);
    expect_k_phase_boundaries_for_order<matmul::loop_order::NKM>(logical_k);
    expect_k_phase_boundaries_for_order<matmul::loop_order::KMN>(logical_k);
    expect_k_phase_boundaries_for_order<matmul::loop_order::KNM>(logical_k);
  }
}

TEST(MatmulLoopNestTest, ZeroOutputExtentSkipsEveryLoopOrder) {
  expect_zero_output_extent_skips_order<matmul::loop_order::MNK>();
  expect_zero_output_extent_skips_order<matmul::loop_order::MKN>();
  expect_zero_output_extent_skips_order<matmul::loop_order::NMK>();
  expect_zero_output_extent_skips_order<matmul::loop_order::NKM>();
  expect_zero_output_extent_skips_order<matmul::loop_order::KMN>();
  expect_zero_output_extent_skips_order<matmul::loop_order::KNM>();
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

template <matmul::Operand Side, typename Order>
struct ExhaustivePanelLifetimeHook {
  int* open_count;
  int* active_count;
  std::set<std::pair<nint_t, nint_t>>* dependency_blocks;

  template <int Depth, matmul::Axis, typename Context, typename Phase,
            typename State, typename Continue>
  void operator()(const Context& block, const Phase&, const State& state,
                  Continue&& continuation) const {
    if constexpr (
        Depth == 1 &&
        matmul::details::has_panel_lifetime_site_v<Side, Order>) {
      ++*open_count;
      ++*active_count;
      if constexpr (Side == matmul::Operand::A)
        dependency_blocks->emplace(block.m_origin, block.k_origin);
      else
        dependency_blocks->emplace(block.n_origin, block.k_origin);
      continuation(state + 1);
      --*active_count;
    } else {
      continuation(state);
    }
  }
};

template <matmul::Operand Side, typename Order>
void expect_exhaustive_panel_lifetime_hook() {
  using Tiling = matmul::CacheTiling<
      meta::Const<2>, meta::Const<3>, meta::Const<4>,
      matmul::CacheLoopMode::enabled,
      matmul::CacheLoopMode::enabled,
      matmul::CacheLoopMode::enabled>;
  constexpr bool HasPanelSite =
      matmul::details::has_panel_lifetime_site_v<Side, Order>;
  int open_count = 0;
  int active_count = 0;
  int leaves = 0;
  std::set<std::pair<nint_t, nint_t>> dependency_blocks;
  ExhaustivePanelLifetimeHook<Side, Order> hook{
      &open_count, &active_count, &dependency_blocks};
  matmul::details::LoopNest<Order, Tiling>::run_phased_with_state(
      meta::Any{5}, meta::Any{10}, meta::Any{17}, Tiling{}, 0, hook,
      [&](const auto&, auto, int state) {
        ++leaves;
        EXPECT_EQ(active_count, HasPanelSite ? 1 : 0);
        EXPECT_EQ(state, HasPanelSite ? 1 : 0);
      });

  EXPECT_EQ(leaves, 3 * 4 * 5);
  const int expected_opens = HasPanelSite
      ? (Side == matmul::Operand::A ? 3 * 5 : 4 * 5)
      : 0;
  EXPECT_EQ(open_count, expected_opens);
  EXPECT_EQ(static_cast<int>(dependency_blocks.size()), expected_opens);
  EXPECT_EQ(active_count, 0);
}

TEST(MatmulLoopNestTest, PanelLifetimeHookIsSymmetricForEveryLoopOrder) {
#define VECOPS_TEST_PANEL_LIFETIME_FOR_ORDER(Order)                    \
  expect_exhaustive_panel_lifetime_hook<matmul::Operand::A, Order>(); \
  expect_exhaustive_panel_lifetime_hook<matmul::Operand::B, Order>()
  VECOPS_TEST_PANEL_LIFETIME_FOR_ORDER(matmul::loop_order::MNK);
  VECOPS_TEST_PANEL_LIFETIME_FOR_ORDER(matmul::loop_order::MKN);
  VECOPS_TEST_PANEL_LIFETIME_FOR_ORDER(matmul::loop_order::NMK);
  VECOPS_TEST_PANEL_LIFETIME_FOR_ORDER(matmul::loop_order::NKM);
  VECOPS_TEST_PANEL_LIFETIME_FOR_ORDER(matmul::loop_order::KMN);
  VECOPS_TEST_PANEL_LIFETIME_FOR_ORDER(matmul::loop_order::KNM);
#undef VECOPS_TEST_PANEL_LIFETIME_FOR_ORDER
}

} // namespace
