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

#if defined(ARCH_X86_FAMILY)
using ExplicitNMajorTuning = matmul::GenericTiledTuning<
    matmul::AutomaticCacheTiling, matmul::loop_order::NKM>;
using ExplicitMMajorTuning = matmul::GenericTiledTuning<
    matmul::AutomaticCacheTiling, matmul::loop_order::MKN>;
using ExplicitNMajorConfig = ops::MatmulConfig<
    PanelTestAtom, matmul::family_selection::Automatic,
    kernel::matmul_policy::Automatic, ExplicitNMajorTuning>;
using ExplicitMMajorConfig = ops::MatmulConfig<
    PanelTestAtom, matmul::family_selection::Automatic,
    kernel::matmul_policy::Automatic, ExplicitMMajorTuning>;
static_assert(matmul::details::explicit_logical_n_major_v<
              ExplicitNMajorConfig>);
static_assert(!matmul::details::explicit_logical_n_major_v<
              ExplicitMMajorConfig>);
static_assert(matmul::details::automatic_physical_n_major_v<
              ops::MatmulConfig<PanelTestAtom>, Const<64>, Const<1024>,
              Const<4096>>);
static_assert(!matmul::details::automatic_physical_n_major_v<
              ops::MatmulConfig<PanelTestAtom>, Any, Any, Any>);
#endif

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

#if defined(ARCH_X86_FAMILY)
TEST(MatmulPanelLifetimeTest, PreparedBWithUnboundedMStaticallyPacksA) {
  ASSERT_TRUE(test::matmul::MatmulTestArchTraits::enable());
  // Both spatial axes are ragged relative to a 16-row AMX tile, so this also
  // covers the runtime N-major tail mapping selected after the static pack.
  constexpr nint_t M = 17, N = 257, K = 512;
  using Atom = matmul::AMX_BF16F32;
  std::vector<bfloat16_t> a(M * K, bfloat16_t{0.25f});
  std::vector<bfloat16_t> b(N * K, bfloat16_t{0.5f});
  std::vector<float32_t> c(M * N, 0.0f);
  auto at = make_tensor(
      a.data(), make_layout(make_shape(Any{M}, cint<K>)));
  auto bt = make_tensor(
      b.data(), make_layout(make_shape(cint<N>, cint<K>)));
  auto ct = make_tensor(
      c.data(), make_layout(make_shape(Any{M}, cint<N>)));

  const auto b_layout = matmul::packed_layout<
      Atom, matmul::Operand::B>(bt.layout());
  const nint_t b_bytes = numel(b_layout) *
      static_cast<nint_t>(sizeof(bfloat16_t));
  kernel::Workspace b_storage(b_bytes + vec::DEFAULT_ALIGNMENT);
  auto b_workspace = b_storage.view();
  auto* b_data = static_cast<bfloat16_t*>(
      b_workspace.allocate(b_bytes, vec::DEFAULT_ALIGNMENT));
  auto packed_b = make_tensor(b_data, b_layout);
  auto pack_b = ops::matmul_pack(
      ops::MatmulPackConfig<Atom, matmul::Operand::B>{});
  kernel::Workspace pack_scratch(
      pack_b.required_workspace(bt, packed_b));
  auto pack_scratch_view = pack_scratch.view();
  pack_b(pack_scratch_view, bt, packed_b);

  auto operation = ops::matmul(ops::MatmulConfig<Atom>{});
  const auto a_layout = matmul::packed_layout<
      Atom, matmul::Operand::A>(at.layout());
  const nint_t a_bytes = numel(a_layout) *
      static_cast<nint_t>(sizeof(bfloat16_t));
  const nint_t required = operation.required_workspace(
      Any{M}, cint<N>, cint<K>, at, packed_b, ct);
  // This is the regression contract: caller-prepared B makes native A
  // packing monotone in N reuse, even though M itself is unbounded Dynamic.
  EXPECT_GT(required, a_bytes);
  kernel::Workspace storage(required);
  auto workspace = storage.view();
  operation(workspace, Any{M}, cint<N>, cint<K>, at, packed_b, ct);
  EXPECT_EQ(workspace.used(), 0);
  EXPECT_LE(workspace.high_watermark(),
            required + vec::DEFAULT_ALIGNMENT);
  const float expected = static_cast<float>(K) * 0.25f * 0.5f;
  for (float value : c) EXPECT_NEAR(value, expected, 4.0e-3f);
}

TEST(MatmulPanelLifetimeTest, WholeProblemBoundsLargeBPanelLifetime) {
  ASSERT_TRUE(test::matmul::MatmulTestArchTraits::enable());
  constexpr nint_t M = 32, N = 32768, K = 256;
  using Atom = matmul::AMX_BF16F32;
  std::vector<bfloat16_t> a(M * K, bfloat16_t{0.25f});
  std::vector<bfloat16_t> b(N * K, bfloat16_t{0.5f});
  std::vector<float32_t> c(M * N, 1.0f);
  auto at = make_tensor(
      a.data(), make_layout(make_shape(Any{M}, Any{K})));
  auto bt = make_tensor(
      b.data(), make_layout(make_shape(Any{N}, Any{K})));
  auto ct = make_tensor(
      c.data(), make_layout(make_shape(Any{M}, Any{N})));
  auto operation = ops::matmul(ops::MatmulConfig<Atom>{});
  const nint_t product_workspace = operation.required_workspace(
      Any{M}, Any{N}, Any{K}, at, bt, ct);
  // The complete B is 16 MiB. The automatic WholeProblem route retains only
  // one 64-row full-K panel plus microkernel scratch.
  EXPECT_GT(product_workspace, 32 * 1024);
  EXPECT_LT(product_workspace, 1024 * 1024);
  kernel::Workspace owner(product_workspace);
  auto workspace = owner.view();
  operation(workspace, Any{M}, Any{N}, Any{K}, at, bt, ct);
  for (nint_t index : {nint_t{0}, N - 1, (M / 2) * N + N / 2,
                       M * N - 1})
    EXPECT_NEAR(c[index], 32.0f, 1.0e-3f) << "index=" << index;

  std::fill(c.begin(), c.end(), 1.0f);
  auto c_input = input<float32_t>(ct);
  auto c_output = output<float32_t>(ct);
  const nint_t add_workspace = operation.required_workspace(
      Any{M}, Any{N}, Any{K}, at, bt, c_input, c_output);
  kernel::Workspace add_owner(add_workspace);
  auto add_view = add_owner.view();
  operation(add_view, Any{M}, Any{N}, Any{K},
            at, bt, c_input, c_output);
  for (nint_t index : {nint_t{0}, N - 1, (M / 2) * N + N / 2,
                       M * N - 1})
    EXPECT_NEAR(c[index], 33.0f, 1.0e-3f) << "index=" << index;
}

TEST(MatmulPanelLifetimeTest, WholeProblemScalesBPanelWithSpatialReuse) {
  ASSERT_TRUE(test::matmul::MatmulTestArchTraits::enable());
  constexpr nint_t M = 512, N = 4096, K = 256;
  using Atom = matmul::AMX_BF16F32;
  std::vector<bfloat16_t> a(M * K, bfloat16_t{0.25f});
  std::vector<bfloat16_t> b(N * K, bfloat16_t{0.5f});
  std::vector<float32_t> c(M * N, 0.0f);
  auto at = make_tensor(
      a.data(), make_layout(make_shape(Any{M}, Any{K})));
  auto bt = make_tensor(
      b.data(), make_layout(make_shape(Any{N}, Any{K})));
  auto ct = make_tensor(
      c.data(), make_layout(make_shape(Any{M}, Any{N})));
  auto operation = ops::matmul(ops::MatmulConfig<Atom>{});
  const nint_t required = operation.required_workspace(
      Any{M}, Any{N}, Any{K}, at, bt, ct);
  // M/4 selects a 128-column panel: smaller than complete B packing while
  // retaining four hardware-tile columns per packed lifetime.
  EXPECT_GT(required, 64 * 1024);
  EXPECT_LT(required, 192 * 1024);
  kernel::Workspace owner(required);
  auto workspace = owner.view();
  operation(workspace, Any{M}, Any{N}, Any{K}, at, bt, ct);
  for (nint_t index : {nint_t{0}, N - 1, (M / 2) * N + N / 2,
                       M * N - 1})
    EXPECT_NEAR(c[index], 32.0f, 1.0e-3f) << "index=" << index;
}
#endif

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
