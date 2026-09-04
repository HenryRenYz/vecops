//
// Copyright (c) vecops contributors.
//

#include "vecops/platform/Features.h"

#if defined(ARCH_X86_FAMILY) || defined(HAS_SME)

#include <gtest/gtest.h>

#include <algorithm>
#include <vector>

#include "vecops/ops/Matmul.h"
#include "vecops/ops/MatmulPack.h"

#include "MatmulTestArch.h"

namespace {

using namespace vecops;
using namespace vecops::meta;
using namespace vecops::tensor;

#if defined(ARCH_X86_FAMILY)
using PanelTestAtom = matmul::AMX_BF16F32;
#else
using PanelTestAtom = matmul::SME_F32F32;
#endif

using AlwaysAutomatic = matmul::PackingPolicy<
    matmul::PackingMode::always, matmul::PackingExtent::automatic>;
using PanelTiles = matmul::CacheTiling<
    meta::Const<16>, meta::Const<16>, meta::Const<32>>;

template <typename Order, typename APlacement, typename BPlacement>
void check_policy_execution() {
  using GenericTuning = matmul::GenericTiledTuning<
      PanelTiles, Order, AlwaysAutomatic, AlwaysAutomatic,
      matmul::AccBufferMode::workspace>;
  using PackingTuning = matmul::MatmulPackingTuning<APlacement, BPlacement>;
  using Config = ops::MatmulConfigWithPacking<
      PanelTestAtom, PackingTuning,
      matmul::family_selection::Require<matmul::kernel_family::GenericTiled>,
      kernel::matmul_policy::Automatic, GenericTuning,
      platform::SystemCacheInfoProvider, false>;

  constexpr nint_t M = 33, N = 35, K = 65;
  using TA = typename PanelTestAtom::TA;
  using TB = typename PanelTestAtom::TB;
  using Acc = typename PanelTestAtom::TAcc;
  std::vector<TA> a(M * K);
  std::vector<TB> b(N * K);
  std::vector<Acc> c(M * N, Acc{});
  for (nint_t i = 0; i < M * K; ++i)
    a[i] = TA(static_cast<float>(i % 13 - 6) / 13.0f);
  for (nint_t i = 0; i < N * K; ++i)
    b[i] = TB(static_cast<float>(i % 11 - 5) / 11.0f);

  auto at = make_tensor(a.data(), make_layout(make_shape(cint<M>, cint<K>)));
  auto bt = make_tensor(b.data(), make_layout(make_shape(cint<N>, cint<K>)));
  auto ct = make_tensor(c.data(), make_layout(make_shape(cint<M>, cint<N>)));
  auto operation = ops::matmul(Config{.generic_tiled = GenericTuning{
      .cache_tiling = PanelTiles{cint<16>, cint<16>, cint<32>}}});
  const nint_t required = operation.required_workspace(
      cint<M>, cint<N>, cint<K>, at, bt, ct);
  kernel::Workspace storage(required);
  auto workspace = storage.view();
  operation(workspace, cint<M>, cint<N>, cint<K>, at, bt, ct);
  EXPECT_EQ(workspace.used(), 0);
  EXPECT_LE(workspace.high_watermark(), required + vec::DEFAULT_ALIGNMENT);
  for (nint_t row = 0; row < M; ++row) {
    for (nint_t column = 0; column < N; ++column) {
      Acc expected{};
      for (nint_t kk = 0; kk < K; ++kk)
        expected += static_cast<Acc>(a[row * K + kk]) *
            static_cast<Acc>(b[column * K + kk]);
      EXPECT_NEAR(static_cast<double>(c[row * N + column]),
                  static_cast<double>(expected), 4.0e-3)
          << "row=" << row << " column=" << column;
    }
  }
}

TEST(MatmulPanelLifetimeTest, OutsidePanelsAreSymmetric) {
  ASSERT_TRUE(test::matmul::MatmulTestArchTraits::enable());
  check_policy_execution<matmul::loop_order::MKN,
      matmul::packing_policy::RequireOutside,
      matmul::packing_policy::Disabled>();
  check_policy_execution<matmul::loop_order::KMN,
      matmul::packing_policy::RequireOutside,
      matmul::packing_policy::Disabled>();
  check_policy_execution<matmul::loop_order::NKM,
      matmul::packing_policy::Disabled,
      matmul::packing_policy::RequireOutside>();
  check_policy_execution<matmul::loop_order::KNM,
      matmul::packing_policy::Disabled,
      matmul::packing_policy::RequireOutside>();
}

TEST(MatmulPanelLifetimeTest, InsideAndDisabledPoliciesRemainCorrect) {
  ASSERT_TRUE(test::matmul::MatmulTestArchTraits::enable());
  check_policy_execution<matmul::loop_order::MNK,
      matmul::packing_policy::InsideOnly,
      matmul::packing_policy::InsideOnly>();
  check_policy_execution<matmul::loop_order::NMK,
      matmul::packing_policy::Disabled,
      matmul::packing_policy::Disabled>();
}

TEST(MatmulPanelLifetimeTest, CallerPreparedOnlyConsumesPersistentPackedB) {
  ASSERT_TRUE(test::matmul::MatmulTestArchTraits::enable());
  constexpr nint_t M = 17;
#if defined(ARCH_X86_FAMILY)
  constexpr nint_t N = 19;
  constexpr auto PreparedNCache = cint<16>;
  using PreparedTiles = matmul::CacheTiling<
      meta::Const<16>, meta::Const<16>, meta::Const<32>>;
  using MisalignedTiles = matmul::CacheTiling<
      meta::Const<16>, meta::Const<8>, meta::Const<32>>;
#else
  constexpr nint_t N = 35;
  constexpr auto PreparedNCache = cint<32>;
  using PreparedTiles = matmul::CacheTiling<
      meta::Const<16>, meta::Const<32>, meta::Const<32>>;
  using MisalignedTiles = matmul::CacheTiling<
      meta::Const<16>, meta::Const<16>, meta::Const<32>>;
#endif
  constexpr nint_t K = 65;
  using TA = typename PanelTestAtom::TA;
  using TB = typename PanelTestAtom::TB;
  using Acc = typename PanelTestAtom::TAcc;
  using Packed = typename matmul::packing_t<
      PanelTestAtom, matmul::Operand::B>::Element;
  std::vector<TA> a(M * K);
  std::vector<TB> b(N * K);
  std::vector<Acc> c(M * N, Acc{});
  for (nint_t i = 0; i < M * K; ++i)
    a[i] = TA(static_cast<float>(i % 13 - 6) / 13.0f);
  for (nint_t i = 0; i < N * K; ++i)
    b[i] = TB(static_cast<float>(i % 11 - 5) / 11.0f);
  auto at = make_tensor(a.data(), make_layout(make_shape(cint<M>, cint<K>)));
  auto bt = make_tensor(b.data(), make_layout(make_shape(cint<N>, cint<K>)));
  auto ct = make_tensor(c.data(), make_layout(make_shape(cint<M>, cint<N>)));

  const auto packed_layout = matmul::packed_layout<
      PanelTestAtom, matmul::Operand::B>(bt.layout());
  const nint_t packed_bytes = numel(packed_layout) *
      static_cast<nint_t>(sizeof(Packed));
  kernel::Workspace packed_storage(packed_bytes + 63);
  auto packed_workspace = packed_storage.view();
  auto* packed_data = static_cast<Packed*>(
      packed_workspace.allocate(packed_bytes, 64));
  auto packed_bt = make_tensor(packed_data, packed_layout);
  auto packer = ops::matmul_pack(ops::MatmulPackConfig<
      PanelTestAtom, matmul::Operand::B>{});
  kernel::Workspace pack_scratch(packer.required_workspace(bt, packed_bt));
  auto pack_scratch_view = pack_scratch.view();
  packer(pack_scratch_view, bt, packed_bt);

  auto packed_b_input = tensor::input<Packed>(packed_bt);
#if defined(ARCH_X86_FAMILY)
  constexpr nint_t MisalignedOrigin = 8, MisalignedExtent = 3;
#else
  constexpr nint_t MisalignedOrigin = 16, MisalignedExtent = 3;
#endif
  // A packed tile loader owns a complete hardware tile. A sub-panel base
  // would make that load cross into the next K group and is rejected in
  // release builds rather than silently returning wrong columns.
  EXPECT_ANY_THROW((matmul::details::narrow_input<
      PanelTestAtom, matmul::Operand::B>(
          packed_b_input, MisalignedOrigin, Any{MisalignedExtent},
          0, Any{K})));

  using GenericTuning = matmul::GenericTiledTuning<
      PreparedTiles, matmul::loop_order::NKM,
      AlwaysAutomatic, AlwaysAutomatic, matmul::AccBufferMode::workspace>;
  using PreparedTuning = matmul::MatmulPackingTuning<
      matmul::packing_policy::Disabled,
      matmul::packing_policy::CallerPreparedOnly>;
  using DisabledTuning = matmul::MatmulPackingTuning<
      matmul::packing_policy::Disabled,
      matmul::packing_policy::Disabled>;
  using Family = matmul::family_selection::Require<
      matmul::kernel_family::GenericTiled>;
  using PreparedConfig = ops::MatmulConfigWithPacking<
      PanelTestAtom, PreparedTuning, Family, kernel::matmul_policy::Automatic,
      GenericTuning, platform::SystemCacheInfoProvider, false>;
  using DisabledConfig = ops::MatmulConfigWithPacking<
      PanelTestAtom, DisabledTuning, Family, kernel::matmul_policy::Automatic,
      GenericTuning, platform::SystemCacheInfoProvider, false>;
  using MisalignedTuning = matmul::GenericTiledTuning<
      MisalignedTiles, matmul::loop_order::NKM,
      AlwaysAutomatic, AlwaysAutomatic, matmul::AccBufferMode::workspace>;
  using MisalignedConfig = ops::MatmulConfigWithPacking<
      PanelTestAtom, PreparedTuning, Family,
      kernel::matmul_policy::Automatic, MisalignedTuning,
      platform::SystemCacheInfoProvider, false>;
  const PreparedConfig prepared_config{.generic_tiled = GenericTuning{
      .cache_tiling = PreparedTiles{
          cint<16>, PreparedNCache, cint<32>}}};
  const DisabledConfig disabled_config{.generic_tiled = GenericTuning{
      .cache_tiling = PreparedTiles{
          cint<16>, PreparedNCache, cint<32>}}};

  // A whole/prepared packed view cannot be rebased at a half-panel cache
  // origin. The tiler must reject that configuration before touching C.
  std::fill(c.begin(), c.end(), Acc{7});
  auto misaligned_operation = ops::matmul(MisalignedConfig{});
  const nint_t misaligned_required = misaligned_operation.required_workspace(
      cint<M>, cint<N>, cint<K>, at, packed_bt, ct);
  kernel::Workspace misaligned_storage(misaligned_required);
  auto misaligned_workspace = misaligned_storage.view();
  EXPECT_ANY_THROW(misaligned_operation(
      misaligned_workspace, cint<M>, cint<N>, cint<K>, at, packed_bt, ct));
  for (Acc value : c) EXPECT_EQ(value, Acc{7});
  std::fill(c.begin(), c.end(), Acc{});

  auto operation = ops::matmul(prepared_config);
  const nint_t required = operation.required_workspace(
      cint<M>, cint<N>, cint<K>, at, packed_bt, ct);
  EXPECT_EQ(required, ops::matmul(disabled_config).required_workspace(
      cint<M>, cint<N>, cint<K>, at, packed_bt, ct));
  kernel::Workspace storage(required);
  auto workspace = storage.view();
  operation(workspace, cint<M>, cint<N>, cint<K>, at, packed_bt, ct);
  EXPECT_EQ(workspace.used(), 0);
  EXPECT_LE(workspace.high_watermark(), required + vec::DEFAULT_ALIGNMENT);
  for (nint_t row = 0; row < M; ++row) {
    for (nint_t column = 0; column < N; ++column) {
      Acc expected{};
      for (nint_t kk = 0; kk < K; ++kk)
        expected += static_cast<Acc>(a[row * K + kk]) *
            static_cast<Acc>(b[column * K + kk]);
      EXPECT_NEAR(static_cast<double>(c[row * N + column]),
                  static_cast<double>(expected), 4.0e-3)
          << "row=" << row << " column=" << column;
    }
  }

  // The top-level contract is family-independent. Automatic/WholeProblem
  // must consume the same caller-owned packed B without silently repacking.
  using WholePreparedConfig = ops::MatmulConfigWithPacking<
      PanelTestAtom, PreparedTuning,
      matmul::family_selection::Automatic,
      kernel::matmul_policy::Automatic, GenericTuning,
      platform::SystemCacheInfoProvider, false>;
  std::fill(c.begin(), c.end(), Acc{});
  auto whole_operation = ops::matmul(WholePreparedConfig{});
  const nint_t whole_required = whole_operation.required_workspace(
      cint<M>, cint<N>, cint<K>, at, packed_bt, ct);
  kernel::Workspace whole_storage(whole_required);
  auto whole_workspace = whole_storage.view();
  whole_operation(
      whole_workspace, cint<M>, cint<N>, cint<K>, at, packed_bt, ct);
  EXPECT_EQ(whole_workspace.used(), 0);
  EXPECT_LE(whole_workspace.high_watermark(),
            whole_required + vec::DEFAULT_ALIGNMENT);
  for (nint_t row = 0; row < M; ++row) {
    for (nint_t column = 0; column < N; ++column) {
      Acc expected{};
      for (nint_t kk = 0; kk < K; ++kk)
        expected += static_cast<Acc>(a[row * K + kk]) *
            static_cast<Acc>(b[column * K + kk]);
      EXPECT_NEAR(static_cast<double>(c[row * N + column]),
                  static_cast<double>(expected), 4.0e-3)
          << "WholeProblem row=" << row << " column=" << column;
    }
  }
}

} // namespace

#endif
