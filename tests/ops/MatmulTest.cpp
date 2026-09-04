// @vecops-target-shards-x86: 5
// @vecops-target-shards-ARM: 5

#include "vecops/platform/Features.h"
#include "MatmulTestArch.h"

#if defined(ARCH_X86_FAMILY)

#if defined(VECOPS_TARGET_SHARD_ACTIVE)

#include <gtest/gtest.h>

#include <algorithm>
#include <bit>
#include <cmath>
#include <cstdint>
#include <type_traits>
#include <vector>

#include "vecops/ops/Matmul.h"
#include "vecops/ops/MatmulPack.h"

#include "MatmulConversionTestCommon.h"

namespace {

static_assert(VECOPS_TARGET_SHARD_COUNT == 5);

using namespace vecops;
using namespace vecops::meta;
using namespace vecops::tensor;


template <typename T>
T value(nint_t index, int modulus) {
  const int x = static_cast<int>(index % modulus) - modulus / 2;
  if constexpr (std::is_integral_v<T>) return static_cast<T>(x);
  else return T(static_cast<float>(x) / static_cast<float>(modulus));
}

template <typename Atom, typename FamilySelection,
          typename M, typename N, typename K>
void check_raw_family(
    M m_extent, N n_extent, K k_extent,
    bool expect_auto_packing = false,
    bool expect_no_auto_packing = false) {
  using TA = typename Atom::TA;
  using TB = typename Atom::TB;
  using Acc = typename Atom::TAcc;
  const nint_t m = static_cast<nint_t>(m_extent);
  const nint_t n = static_cast<nint_t>(n_extent);
  const nint_t k = static_cast<nint_t>(k_extent);
  std::vector<TA> a(static_cast<std::size_t>(m * k));
  std::vector<TB> b(static_cast<std::size_t>(n * k));
  std::vector<Acc> c(static_cast<std::size_t>(m * n), Acc{});
  for (nint_t i = 0; i < m * k; ++i) a[i] = value<TA>(i, 13);
  for (nint_t i = 0; i < n * k; ++i) b[i] = value<TB>(i, 11);
  auto at = make_tensor(
      a.data(), make_layout(make_shape(m_extent, k_extent)));
  auto bt = make_tensor(
      b.data(), make_layout(make_shape(n_extent, k_extent)));
  auto ct = make_tensor(
      c.data(), make_layout(make_shape(m_extent, n_extent)));
  auto operation = ops::matmul(
      ops::MatmulConfig<Atom, FamilySelection>{});
  const nint_t required = operation.required_workspace(
      m_extent, n_extent, k_extent, at, bt, ct);
  if (expect_auto_packing) {
    const nint_t scratch = kernel::matmul_implementation::scratch_bytes<
        kernel::matmul_implementation::AMX>();
    EXPECT_GT(required, scratch);
  } else if (expect_no_auto_packing) {
    const nint_t scratch = kernel::matmul_implementation::scratch_bytes<
        kernel::matmul_implementation::AMX>();
    EXPECT_EQ(required, scratch);
  }
  kernel::Workspace storage(required);
  auto workspace = storage.view();
  operation(workspace, m_extent, n_extent, k_extent, at, bt, ct);
  for (nint_t i = 0; i < m; ++i) {
    for (nint_t j = 0; j < n; ++j) {
      Acc expected{};
      for (nint_t kk = 0; kk < k; ++kk) {
        expected += static_cast<Acc>(a[i * k + kk]) *
                    static_cast<Acc>(b[j * k + kk]);
      }
      if constexpr (std::is_floating_point_v<Acc>) {
        EXPECT_NEAR(c[i * n + j], expected, 2.0e-4f)
            << "m=" << i << " n=" << j;
      } else {
        EXPECT_EQ(c[i * n + j], expected)
            << "m=" << i << " n=" << j;
      }
    }
  }
}

template <typename Atom, typename M, typename N, typename K>
void check_raw(
    M m_extent, N n_extent, K k_extent,
    bool expect_auto_packing = false,
    bool expect_no_auto_packing = false) {
  check_raw_family<
      Atom, ::vecops::matmul::family_selection::Automatic>(
          m_extent, n_extent, k_extent,
          expect_auto_packing, expect_no_auto_packing);
}

template <typename MemoryA, typename MemoryB>
void check_online_pack_decision(
    nint_t m, nint_t n, nint_t k, bool pack_a, bool pack_b) {
  using Atom = ::vecops::matmul::AMX_BF16F32;
  std::vector<MemoryA> a(static_cast<std::size_t>(m * k));
  std::vector<MemoryB> b(static_cast<std::size_t>(n * k));
  std::vector<float32_t> c(static_cast<std::size_t>(m * n));
  auto at = make_tensor(
      a.data(), make_layout(make_shape(Any{m}, Any{k})));
  auto bt = make_tensor(
      b.data(), make_layout(make_shape(Any{n}, Any{k})));
  auto ct = make_tensor(
      c.data(), make_layout(make_shape(Any{m}, Any{n})));
  auto invocation = test::matmul::make_test_matmul_invocation(
      ops::MatmulConfig<Atom>{}, Any{m}, Any{n}, Any{k}, at, bt, ct);
  EXPECT_EQ(invocation.online_packs_a(), pack_a);
  EXPECT_EQ(invocation.online_packs_b(), pack_b);
}

template <typename MemoryA, typename MemoryB,
          meta::ValueInput M, meta::ValueInput N, meta::ValueInput K>
void check_bounded_online_pack_decision(
    M m_extent, N n_extent, K k_extent, bool pack_a, bool pack_b) {
  using Atom = ::vecops::matmul::AMX_BF16F32;
  const nint_t m = static_cast<nint_t>(m_extent);
  const nint_t n = static_cast<nint_t>(n_extent);
  const nint_t k = static_cast<nint_t>(k_extent);
  std::vector<MemoryA> a(static_cast<std::size_t>(m * k));
  std::vector<MemoryB> b(static_cast<std::size_t>(n * k));
  std::vector<float32_t> c(static_cast<std::size_t>(m * n));
  auto at = make_tensor(
      a.data(), make_layout(make_shape(m_extent, k_extent)));
  auto bt = make_tensor(
      b.data(), make_layout(make_shape(n_extent, k_extent)));
  auto ct = make_tensor(
      c.data(), make_layout(make_shape(m_extent, n_extent)));
  auto invocation = test::matmul::make_test_matmul_invocation(
      ops::MatmulConfig<Atom>{}, m_extent, n_extent, k_extent, at, bt, ct);
  EXPECT_EQ(invocation.online_packs_a(), pack_a);
  EXPECT_EQ(invocation.online_packs_b(), pack_b);
}

#if VECOPS_TARGET_SHARD_INDEX == 0
TEST(MatmulTest, WholeProblemHonorsTopLevelPackingPlacement) {
  ASSERT_TRUE(vecops::test::matmul::MatmulTestArchTraits::enable());
  using Atom = ::vecops::matmul::AMX_BF16F32;
  using RequiredB = ::vecops::matmul::MatmulPackingTuning<
      ::vecops::matmul::packing_policy::Disabled,
      ::vecops::matmul::packing_policy::RequireOutside>;
  using DisabledBoth = ::vecops::matmul::MatmulPackingTuning<
      ::vecops::matmul::packing_policy::Disabled,
      ::vecops::matmul::packing_policy::Disabled>;
  using RequiredConfig = ops::MatmulConfigWithPacking<
      Atom, RequiredB, ::vecops::matmul::family_selection::Automatic,
      kernel::matmul_policy::Automatic,
      ::vecops::matmul::GenericTiledTuning<>,
      platform::SystemCacheInfoProvider, false>;
  using DisabledConfig = ops::MatmulConfigWithPacking<
      Atom, DisabledBoth, ::vecops::matmul::family_selection::Automatic,
      kernel::matmul_policy::Automatic,
      ::vecops::matmul::GenericTiledTuning<>,
      platform::SystemCacheInfoProvider, false>;

  constexpr nint_t M = 16, N = 16, K = 32;
  std::vector<bfloat16_t> a(M * K), b(N * K);
  std::vector<float32_t> c(M * N);
  for (nint_t i = 0; i < M * K; ++i) a[i] = value<bfloat16_t>(i, 13);
  for (nint_t i = 0; i < N * K; ++i) b[i] = value<bfloat16_t>(i, 11);
  auto at = make_tensor(a.data(), make_layout(make_shape(cint<M>, cint<K>)));
  auto bt = make_tensor(b.data(), make_layout(make_shape(cint<N>, cint<K>)));
  auto ct = make_tensor(c.data(), make_layout(make_shape(cint<M>, cint<N>)));

  auto required = test::matmul::make_test_matmul_invocation(
      RequiredConfig{}, cint<M>, cint<N>, cint<K>, at, bt, ct);
  EXPECT_FALSE(required.online_packs_a());
  EXPECT_TRUE(required.online_packs_b());
  kernel::Workspace required_storage(required.required_workspace());
  auto required_workspace = required_storage.view();
  required(required_workspace);
  for (nint_t i = 0; i < M; ++i) {
    for (nint_t j = 0; j < N; ++j) {
      float32_t expected{};
      for (nint_t kk = 0; kk < K; ++kk)
        expected += static_cast<float32_t>(a[i * K + kk]) *
            static_cast<float32_t>(b[j * K + kk]);
      EXPECT_NEAR(c[i * N + j], expected, 2.0e-4f);
    }
  }

  constexpr nint_t LargeM = 128, LargeN = 64, LargeK = 128;
  std::vector<bfloat16_t> large_a(LargeM * LargeK);
  std::vector<bfloat16_t> large_b(LargeN * LargeK);
  std::vector<float32_t> large_c(LargeM * LargeN);
  auto large_at = make_tensor(large_a.data(), make_layout(
      make_shape(cint<LargeM>, cint<LargeK>)));
  auto large_bt = make_tensor(large_b.data(), make_layout(
      make_shape(cint<LargeN>, cint<LargeK>)));
  auto large_ct = make_tensor(large_c.data(), make_layout(
      make_shape(cint<LargeM>, cint<LargeN>)));
  auto disabled = test::matmul::make_test_matmul_invocation(
      DisabledConfig{}, cint<LargeM>, cint<LargeN>, cint<LargeK>,
      large_at, large_bt, large_ct);
  // The legacy model chooses online B packing for this shape; the top-level
  // disabled policy must gate that choice before it reaches execution.
  EXPECT_FALSE(disabled.online_packs_a());
  EXPECT_FALSE(disabled.online_packs_b());
}
#endif

VECOPS_NOINLINE void check_dynamic_inner_stride(nint_t inner_stride) {
  using Atom = ::vecops::matmul::AMX_BF16F32;
  using T = typename Atom::TA;
  using Acc = typename Atom::TAcc;
  constexpr nint_t M = 19;
  constexpr nint_t N = 21;
  constexpr nint_t K = 35;
  const nint_t a_row_stride = K * inner_stride + 3;
  const nint_t b_row_stride = K * inner_stride + 5;
  std::vector<T> a(static_cast<std::size_t>(M * a_row_stride));
  std::vector<T> b(static_cast<std::size_t>(N * b_row_stride));
  std::vector<Acc> c(static_cast<std::size_t>(M * N), Acc{});
  for (nint_t i = 0; i < M; ++i) {
    for (nint_t k = 0; k < K; ++k) {
      a[static_cast<std::size_t>(i * a_row_stride + k * inner_stride)] =
          value<T>(i * K + k, 13);
    }
  }
  for (nint_t j = 0; j < N; ++j) {
    for (nint_t k = 0; k < K; ++k) {
      b[static_cast<std::size_t>(j * b_row_stride + k * inner_stride)] =
          value<T>(j * K + k, 11);
    }
  }
  auto al = make_layout(
      make_shape(cint<M>, cint<K>),
      make_strides(Any{a_row_stride}, Any{inner_stride}));
  auto bl = make_layout(
      make_shape(cint<N>, cint<K>),
      make_strides(Any{b_row_stride}, Any{inner_stride}));
  static_assert(!std::same_as<stride_type_t<1, decltype(al)>, Const<1>>);
  static_assert(!std::same_as<stride_type_t<1, decltype(bl)>, Const<1>>);
  auto at = make_tensor(a.data(), al);
  auto bt = make_tensor(b.data(), bl);
  auto ct = make_tensor(
      c.data(), make_layout(make_shape(cint<M>, cint<N>)));
  auto operation = ops::matmul(ops::MatmulConfig<Atom>{});
  const nint_t required = operation.required_workspace(
      cint<M>, cint<N>, cint<K>, at, bt, ct);
  EXPECT_EQ(
      required,
      kernel::matmul_implementation::scratch_bytes<
          kernel::matmul_implementation::AMX>());
  kernel::Workspace owner(required);
  auto workspace = owner.view();
  operation(workspace, cint<M>, cint<N>, cint<K>, at, bt, ct);
  for (nint_t i = 0; i < M; ++i) {
    for (nint_t j = 0; j < N; ++j) {
      Acc expected{};
      for (nint_t k = 0; k < K; ++k) {
        expected += static_cast<Acc>(a[static_cast<std::size_t>(
                        i * a_row_stride + k * inner_stride)]) *
            static_cast<Acc>(b[static_cast<std::size_t>(
                j * b_row_stride + k * inner_stride)]);
      }
      EXPECT_NEAR(c[static_cast<std::size_t>(i * N + j)], expected, 2.0e-4f)
          << "inner_stride=" << inner_stride << " m=" << i << " n=" << j;
    }
  }
}

template <::vecops::matmul::Operand Side>
void check_mixed_packing(
    nint_t m = 37, nint_t n = 51, nint_t k = 79) {
  using Atom = ::vecops::matmul::AMX_BF16F32;
  std::vector<bfloat16_t> a(m * k), b(n * k);
  std::vector<float32_t> c(m * n);
  for (nint_t i = 0; i < m * k; ++i) a[i] = value<bfloat16_t>(i, 13);
  for (nint_t i = 0; i < n * k; ++i) b[i] = value<bfloat16_t>(i, 11);
  auto al = make_layout(make_shape(Any{m}, Any{k}));
  auto bl = make_layout(make_shape(Any{n}, Any{k}));
  auto at = make_tensor(a.data(), al);
  auto bt = make_tensor(b.data(), bl);
  const auto& source_layout = Side == ::vecops::matmul::Operand::A ? al : bl;
  auto packed_layout = ::vecops::matmul::packed_layout<Atom, Side>(source_layout);
  kernel::Workspace packed_owner(
      numel(packed_layout) * static_cast<nint_t>(sizeof(bfloat16_t)) + 64);
  auto packed_workspace = packed_owner.view();
  auto* packed = static_cast<bfloat16_t*>(packed_workspace.allocate(
      numel(packed_layout) * static_cast<nint_t>(sizeof(bfloat16_t)), 64));
  auto packed_tensor = make_tensor(packed, packed_layout);
  ExecutionSession pack_execution{};
  if constexpr (Side == ::vecops::matmul::Operand::A) {
    ::vecops::matmul::details::run_matmul_pack<Atom, Side>(pack_execution, at, packed_tensor);
  } else {
    ::vecops::matmul::details::run_matmul_pack<Atom, Side>(pack_execution, bt, packed_tensor);
  }
  auto ct = make_tensor(
      c.data(), make_layout(make_shape(Any{m}, Any{n})));
  auto operation = [&] {
    if constexpr (Side == ::vecops::matmul::Operand::A)
      return test::matmul::make_test_matmul_invocation(ops::MatmulConfig<Atom>{}, m, n, k, packed_tensor, bt, ct);
    else
      return test::matmul::make_test_matmul_invocation(ops::MatmulConfig<Atom>{}, m, n, k, at, packed_tensor, ct);
  }();
  kernel::Workspace owner(operation.required_workspace());
  auto workspace = owner.view();
  operation(workspace);
  for (nint_t i = 0; i < m; ++i) {
    for (nint_t j = 0; j < n; ++j) {
      float expected = 0;
      for (nint_t kk = 0; kk < k; ++kk)
        expected += static_cast<float>(a[i * k + kk]) *
                    static_cast<float>(b[j * k + kk]);
      if (std::abs(c[i * n + j] - expected) > 2.0e-4f) {
        ADD_FAILURE()
            << "packed side="
            << (Side == ::vecops::matmul::Operand::A ? "A" : "B")
            << " m=" << i << " n=" << j
            << " actual=" << c[i * n + j]
            << " expected=" << expected;
        return;
      }
    }
  }
}

template <meta::ValueType KExtent,
          typename FamilySelection =
              ::vecops::matmul::family_selection::Automatic>
void check_both_packed(
    KExtent k_extent, nint_t m = 37, nint_t n = 16) {
  using Atom = ::vecops::matmul::AMX_BF16F32;
  using T = typename Atom::TA;
  using Acc = typename Atom::TAcc;
  const nint_t k = static_cast<nint_t>(k_extent);
  std::vector<T> a(static_cast<std::size_t>(m * k));
  std::vector<T> b(static_cast<std::size_t>(n * k));
  std::vector<Acc> c(static_cast<std::size_t>(m * n), Acc{});
  for (nint_t i = 0; i < m * k; ++i) a[i] = value<T>(i, 13);
  for (nint_t i = 0; i < n * k; ++i) b[i] = value<T>(i, 11);
  auto al = make_layout(make_shape(Any{m}, k_extent));
  auto bl = make_layout(make_shape(Any{n}, k_extent));
  auto at = make_tensor(a.data(), al);
  auto bt = make_tensor(b.data(), bl);
  auto apl = ::vecops::matmul::packed_layout<Atom, ::vecops::matmul::Operand::A>(al);
  auto bpl = ::vecops::matmul::packed_layout<Atom, ::vecops::matmul::Operand::B>(bl);
  static_assert(::vecops::matmul::is_packed_layout<
                Atom, ::vecops::matmul::Operand::A, decltype(apl)>());
  static_assert(::vecops::matmul::is_packed_layout<
                Atom, ::vecops::matmul::Operand::B, decltype(bpl)>());
  kernel::Workspace packed_owner(
      (numel(apl) + numel(bpl)) * static_cast<nint_t>(sizeof(T)) + 128);
  auto packed_workspace = packed_owner.view();
  auto* ap = static_cast<T*>(packed_workspace.allocate(
      numel(apl) * static_cast<nint_t>(sizeof(T)), 64));
  auto* bp = static_cast<T*>(packed_workspace.allocate(
      numel(bpl) * static_cast<nint_t>(sizeof(T)), 64));
  auto apt = make_tensor(ap, apl);
  auto bpt = make_tensor(bp, bpl);
  ExecutionSession pack_execution{};
  ::vecops::matmul::details::run_matmul_pack<Atom, ::vecops::matmul::Operand::A>(pack_execution, at, apt);
  ::vecops::matmul::details::run_matmul_pack<Atom, ::vecops::matmul::Operand::B>(pack_execution, bt, bpt);
  auto ct = make_tensor(
      c.data(), make_layout(make_shape(Any{m}, Any{n})));
  auto operation = test::matmul::make_test_matmul_invocation(
      ops::MatmulConfig<Atom, FamilySelection>{},
      Any{m}, Any{n}, k_extent, apt, bpt, ct);
  EXPECT_EQ(
      operation.required_workspace(),
      kernel::matmul_implementation::scratch_bytes<
          kernel::matmul_implementation::AMX>());
  kernel::Workspace owner(operation.required_workspace());
  auto workspace = owner.view();
  operation(workspace);
  for (nint_t i = 0; i < m; ++i) {
    for (nint_t j = 0; j < n; ++j) {
      Acc expected{};
      for (nint_t kk = 0; kk < k; ++kk) {
        expected += static_cast<Acc>(a[static_cast<std::size_t>(i * k + kk)]) *
            static_cast<Acc>(b[static_cast<std::size_t>(j * k + kk)]);
      }
      EXPECT_NEAR(c[static_cast<std::size_t>(i * n + j)], expected, 3.0e-4f)
          << "k=" << k << " m=" << i << " n=" << j;
    }
  }
}

template <typename KExtent>
void check_transformed_inputs(KExtent k_extent) {
  using Atom = ::vecops::matmul::AMX_BF16F32;
  constexpr nint_t M = 19, N = 21;
  const nint_t k = static_cast<nint_t>(k_extent);
  std::vector<bfloat16_t> a(M * k), b(N * k);
  std::vector<float32_t> c(M * N);
  for (nint_t i = 0; i < M * k; ++i) a[i] = value<bfloat16_t>(i, 13);
  for (nint_t i = 0; i < N * k; ++i) b[i] = value<bfloat16_t>(i, 11);
  auto at = make_tensor(
      a.data(), make_layout(make_shape(cint<M>, k_extent)));
  auto bt = make_tensor(
      b.data(), make_layout(make_shape(cint<N>, k_extent)));
  auto ct = make_tensor(
      c.data(), make_layout(make_shape(cint<M>, cint<N>)));
  auto identity = make_elementwise_vec_transform<bfloat16_t, bfloat16_t>(
      [](auto, auto x) VECOPS_KERNEL_LAMBDA { return x; });
  auto operation = test::matmul::make_test_matmul_invocation(ops::MatmulConfig<Atom>{},
      cint<M>, cint<N>, k_extent,
      input<bfloat16_t>(at, identity), input<bfloat16_t>(bt, identity), ct);
  kernel::Workspace owner(operation.required_workspace());
  auto workspace = owner.view();
  operation(workspace);
  for (nint_t i = 0; i < M; ++i) {
    for (nint_t j = 0; j < N; ++j) {
      float expected = 0;
      for (nint_t kk = 0; kk < k; ++kk)
        expected += static_cast<float>(a[i * k + kk]) *
                    static_cast<float>(b[j * k + kk]);
      EXPECT_NEAR(c[i * N + j], expected, 3.0e-4f);
    }
  }
}

#if VECOPS_TARGET_SHARD_INDEX == 0

TEST(MatmulTest, AllResidentMicrokernelShapesAndKTails) {
  ASSERT_TRUE(vecops::test::matmul::MatmulTestArchTraits::enable());
  check_raw<::vecops::matmul::AMX_BF16F32>(16, 16, 35);
  check_raw<::vecops::matmul::AMX_BF16F32>(16, 32, 67);
  check_raw<::vecops::matmul::AMX_BF16F32>(16, 48, 79);
  check_raw<::vecops::matmul::AMX_BF16F32>(1, 48, 79);
  check_raw<::vecops::matmul::AMX_BF16F32>(32, 16, 33);
  check_raw<::vecops::matmul::AMX_BF16F32>(48, 16, 65);
  check_raw<::vecops::matmul::AMX_BF16F32>(32, 32, 97);
  check_raw<::vecops::matmul::AMX_BF16F32>(16, 64, 67);
  check_raw<::vecops::matmul::AMX_BF16F32>(16, 80, 67);
  check_raw<::vecops::matmul::AMX_BF16F32>(16, 96, 67);
  check_raw<::vecops::matmul::AMX_BF16F32>(64, 16, 67);
  check_raw<::vecops::matmul::AMX_BF16F32>(80, 16, 67);
  check_raw<::vecops::matmul::AMX_BF16F32>(96, 16, 67);
  check_raw<::vecops::matmul::AMX_BF16F32>(37, 51, 79);
  check_raw<::vecops::matmul::AMX_BF16F32>(1, 128, 65);
  check_raw<::vecops::matmul::AMX_BF16F32>(512, 1, 256);
}

TEST(MatmulTest, CompileTimeExtentsAndFullKPlan) {
  ASSERT_TRUE(vecops::test::matmul::MatmulTestArchTraits::enable());
  check_raw<::vecops::matmul::AMX_BF16F32>(cint<0>, cint<16>, cint<32>);
  check_raw<::vecops::matmul::AMX_BF16F32>(cint<16>, cint<0>, cint<32>);
  check_raw<::vecops::matmul::AMX_BF16F32>(cint<16>, cint<32>, cint<67>);
  check_raw<::vecops::matmul::AMX_BF16F32>(cint<32>, cint<16>, cint<67>);
  check_raw<::vecops::matmul::AMX_BF16F32>(cint<16>, cint<48>, cint<79>);
  check_raw<::vecops::matmul::AMX_BF16F32>(cint<48>, cint<16>, cint<79>);
  check_raw<::vecops::matmul::AMX_BF16F32>(cint<32>, cint<32>, cint<128>);
  check_raw<::vecops::matmul::AMX_BF16F32>(cint<35>, cint<53>, cint<257>);
  check_raw<::vecops::matmul::AMX_I8I32<int8_t, uint8_t>>(
      cint<35>, cint<53>, cint<257>);
  check_raw<::vecops::matmul::AMX_I8I32<uint8_t, uint8_t>>(
      cint<35>, cint<53>, cint<257>);
  check_raw<::vecops::matmul::AMX_BF16F32>(
      dyn<32, 32, 64>(64), dyn<32, 32, 64>(32),
      dyn<32, 64, 128>(96));
  check_raw<::vecops::matmul::AMX_BF16F32>(
      dyn<32, 32, 64>(32), dyn<16, 32, 64>(48), dyn<32, 64, 128>(96));
  // Bounded runtime extents may select a special owner only when the full
  // admissible Meta range satisfies that owner's crossover.
  check_raw<::vecops::matmul::AMX_BF16F32>(
      dyn<1, 1, 1>(1), dyn<1, 1, 16>(8), dyn<32, 32, 256>(64));
}

TEST(MatmulTest, LargeNativeInputsUseAutoPacking) {
  ASSERT_TRUE(vecops::test::matmul::MatmulTestArchTraits::enable());
  check_raw<::vecops::matmul::AMX_BF16F32>(cint<128>, cint<64>, cint<64>, true);
  check_raw<::vecops::matmul::AMX_I8I32<int8_t, uint8_t>>(
      cint<128>, cint<64>, cint<128>, true);
}

TEST(MatmulTest, DynamicPackingBoundsPruneRuntimeCartesianVariants) {
  ASSERT_TRUE(vecops::test::matmul::MatmulTestArchTraits::enable());
  // Unconstrained native AMX retains only the irreducible raw-vs-B choice.
  check_raw<::vecops::matmul::AMX_BF16F32>(128, 64, 64, true);
  check_raw<::vecops::matmul::AMX_I8I32<int8_t, uint8_t>>(
      64, 64, 128, true);
  check_online_pack_decision<bfloat16_t, bfloat16_t>(
      128, 64, 64, false, true);

  // Lower bounds that pass every AMX reuse/work/footprint gate select one
  // packed variant; finite tiny upper bounds select raw, while unbounded Any
  // retains only the raw-vs-B choice checked above.
  check_bounded_online_pack_decision<bfloat16_t, bfloat16_t>(
      Dynamic<1, 128, 256>{128}, Const<64>{}, Const<64>{}, false, true);
  check_bounded_online_pack_decision<bfloat16_t, bfloat16_t>(
      Dynamic<1, 1, 64>{64}, Const<64>{}, Const<128>{}, false, false);
  check_bounded_online_pack_decision<bfloat16_t, bfloat16_t>(
      Dynamic<1, 128, 128>{128}, Const<256>{}, Const<64>{}, true, true);

  // Conversion/quantization-eliding packs default to their one packed path
  // unless a bounded upper corner is still below the cost-model crossover.
  check_online_pack_decision<float32_t, bfloat16_t>(16, 128, 128, true, false);
  check_online_pack_decision<bfloat16_t, float32_t>(64, 16, 128, false, true);
  check_bounded_online_pack_decision<float32_t, bfloat16_t>(
      Dynamic<1, 1, 2>{2}, Dynamic<1, 1, 4>{4},
      Dynamic<1, 1, 32>{32}, false, false);
}

TEST(MatmulTest, RequiredFamilyRuntimeRejectionPropagatesAsException) {
  ASSERT_TRUE(vecops::test::matmul::MatmulTestArchTraits::enable());
  using Atom = ::vecops::matmul::AMX_BF16F32;
  using Config = ops::MatmulConfig<
      Atom,
      ::vecops::matmul::family_selection::Require<
          ::vecops::matmul::kernel_family::ResidualSplit>>;
  constexpr nint_t M = 128, N = 128, K = 128;
  std::vector<bfloat16_t> a(M * K), b(N * K);
  std::vector<float32_t> c(M * N);
  auto at = make_tensor(
      a.data(), make_layout(make_shape(Any{M}, Any{K})));
  auto bt = make_tensor(
      b.data(), make_layout(make_shape(Any{N}, Any{K})));
  auto ct = make_tensor(
      c.data(), make_layout(make_shape(Any{M}, Any{N})));
  auto operation = ops::matmul(Config{});
  kernel::Workspace storage(operation.required_workspace(
      Any{M}, Any{N}, Any{K}, at, bt, ct));
  auto workspace = storage.view();
  EXPECT_ANY_THROW(operation(
      workspace, Any{M}, Any{N}, Any{K}, at, bt, ct));
}

TEST(MatmulTest, SharedADynamicBatchAmortizesOnlyAPacking) {
  ASSERT_TRUE(vecops::test::matmul::MatmulTestArchTraits::enable());
  using Atom = ::vecops::matmul::AMX_BF16F32;
  constexpr nint_t Batch = 16, M = 1, N = 8, K = 2048;
  std::vector<float32_t> a(M * K, 0.25f);
  std::vector<bfloat16_t> b(Batch * N * K, bfloat16_t{0.5f});
  std::vector<float32_t> c(Batch * M * N);
  auto at = make_tensor(
      a.data(), make_layout(
          make_shape(Any{Batch}, Any{M}, Any{K}),
          make_strides(cint<0>, Any{K}, cint<1>)));
  auto bt = make_tensor(
      b.data(), make_layout(make_shape(Any{Batch}, Any{N}, Any{K})));
  auto ct = make_tensor(
      c.data(), make_layout(make_shape(Any{Batch}, Any{M}, Any{N})));
  auto invocation = test::matmul::make_test_matmul_invocation(
      ops::MatmulConfig<Atom>{}, Any{M}, Any{N}, Any{K}, at, bt, ct);
  EXPECT_TRUE(invocation.online_packs_a());
  EXPECT_FALSE(invocation.online_packs_b());
  kernel::Workspace storage(invocation.required_workspace());
  auto workspace = storage.view();
  invocation(workspace);
  for (float32_t value : c)
    EXPECT_NEAR(value, 256.0f, 1.0e-3f);
}

TEST(MatmulTest, GenericTilerOutputAccumulatorUsesLogicalBlockOrigins) {
  ASSERT_TRUE(vecops::test::matmul::MatmulTestArchTraits::enable());
  using Atom = ::vecops::matmul::AMX_BF16F32;
  using Tiles = ::vecops::matmul::CacheTiling<
      meta::Const<16>, meta::Const<16>, meta::Const<32>>;
  using Never = ::vecops::matmul::PackingPolicy<
      ::vecops::matmul::PackingMode::never>;
  using Tuning = ::vecops::matmul::GenericTiledTuning<
      Tiles, ::vecops::matmul::loop_order::NKM, Never, Never,
      ::vecops::matmul::AccBufferMode::automatic>;
  using Config = ops::MatmulConfig<
      Atom,
      ::vecops::matmul::family_selection::Require<
          ::vecops::matmul::kernel_family::GenericTiled>,
      kernel::matmul_policy::Automatic, Tuning>;
  constexpr nint_t M = 33, N = 35, K = 65;
  std::vector<bfloat16_t> a(M * K), b(N * K);
  std::vector<float32_t> c_input(M * N), c_output(M * N, -9.0f);
  for (nint_t i = 0; i < M * K; ++i) a[i] = value<bfloat16_t>(i, 13);
  for (nint_t i = 0; i < N * K; ++i) b[i] = value<bfloat16_t>(i, 11);
  for (nint_t i = 0; i < M * N; ++i) c_input[i] = value<float32_t>(i, 17);
  auto at = make_tensor(a.data(), make_layout(make_shape(cint<M>, cint<K>)));
  auto bt = make_tensor(b.data(), make_layout(make_shape(cint<N>, cint<K>)));
  auto cit = make_tensor(
      c_input.data(), make_layout(make_shape(cint<M>, cint<N>)));
  auto cot = make_tensor(
      c_output.data(), make_layout(make_shape(cint<M>, cint<N>)));
  std::vector<float32_t> in_place = c_input;
  auto in_place_t = make_tensor(
      in_place.data(), make_layout(make_shape(cint<M>, cint<N>)));
  auto operation = ops::matmul(Config{.generic_tiled = Tuning{
      .cache_tiling = Tiles{cint<16>, cint<16>, cint<32>}}});
  kernel::Workspace storage(operation.required_workspace(
      cint<M>, cint<N>, cint<K>, at, bt, cit, cot));
  auto workspace = storage.view();
  operation(workspace, cint<M>, cint<N>, cint<K>, at, bt, cit, cot);
  for (nint_t i = 0; i < M; ++i) {
    for (nint_t j = 0; j < N; ++j) {
      float32_t expected = c_input[i * N + j];
      for (nint_t kk = 0; kk < K; ++kk) {
        expected += static_cast<float32_t>(a[i * K + kk]) *
            static_cast<float32_t>(b[j * K + kk]);
      }
      EXPECT_NEAR(c_output[i * N + j], expected, 3.0e-3f)
          << "m=" << i << " n=" << j;
    }
  }
  workspace.reset();
  operation(workspace, cint<M>, cint<N>, cint<K>,
            at, bt, in_place_t, in_place_t);
  for (nint_t i = 0; i < M; ++i) {
    for (nint_t j = 0; j < N; ++j) {
      float32_t expected = c_input[i * N + j];
      for (nint_t kk = 0; kk < K; ++kk) {
        expected += static_cast<float32_t>(a[i * K + kk]) *
            static_cast<float32_t>(b[j * K + kk]);
      }
      EXPECT_NEAR(in_place[i * N + j], expected, 3.0e-3f)
          << "in-place m=" << i << " n=" << j;
    }
  }
}

TEST(MatmulTest, GenericTilerZeroKMaterializesSemanticCInput) {
  ASSERT_TRUE(vecops::test::matmul::MatmulTestArchTraits::enable());
  using Atom = ::vecops::matmul::AMX_BF16F32;
  using Tiles = ::vecops::matmul::CacheTiling<
      meta::Const<16>, meta::Const<16>, meta::Const<32>>;
  using Never = ::vecops::matmul::PackingPolicy<
      ::vecops::matmul::PackingMode::never>;
  using Tuning = ::vecops::matmul::GenericTiledTuning<
      Tiles, ::vecops::matmul::loop_order::NKM, Never, Never,
      ::vecops::matmul::AccBufferMode::automatic>;
  using Config = ops::MatmulConfig<
      Atom,
      ::vecops::matmul::family_selection::Require<
          ::vecops::matmul::kernel_family::GenericTiled>,
      kernel::matmul_policy::Automatic, Tuning>;
  constexpr nint_t M = 17, N = 19, K = 0;
  std::vector<bfloat16_t> a, b;
  std::vector<float32_t> c_input(M * N), accumulated(M * N, -3.0f);
  std::vector<float32_t> product(M * N, 7.0f);
  for (nint_t i = 0; i < M * N; ++i) c_input[i] = value<float32_t>(i, 13);
  auto at = make_tensor(a.data(), make_layout(make_shape(cint<M>, cint<K>)));
  auto bt = make_tensor(b.data(), make_layout(make_shape(cint<N>, cint<K>)));
  auto cit = make_tensor(
      c_input.data(), make_layout(make_shape(cint<M>, cint<N>)));
  auto acct = make_tensor(
      accumulated.data(), make_layout(make_shape(cint<M>, cint<N>)));
  auto productt = make_tensor(
      product.data(), make_layout(make_shape(cint<M>, cint<N>)));
  auto operation = ops::matmul(Config{.generic_tiled = Tuning{
      .cache_tiling = Tiles{cint<16>, cint<16>, cint<32>}}});
  const nint_t accumulate_bytes = operation.required_workspace(
      cint<M>, cint<N>, cint<K>, at, bt, cit, acct);
  const nint_t product_bytes = operation.required_workspace(
      cint<M>, cint<N>, cint<K>, at, bt, productt);
  kernel::Workspace storage(std::max(accumulate_bytes, product_bytes));
  auto workspace = storage.view();
  operation(workspace, cint<M>, cint<N>, cint<K>, at, bt, cit, acct);
  workspace.reset();
  operation(workspace, cint<M>, cint<N>, cint<K>, at, bt, productt);
  for (nint_t i = 0; i < M * N; ++i) {
    EXPECT_EQ(accumulated[i], c_input[i]) << "index=" << i;
    EXPECT_EQ(product[i], 0.0f) << "index=" << i;
  }
}

TEST(MatmulTest, ConfigOnlyGenericTilerKBlocks) {
  ASSERT_TRUE(vecops::test::matmul::MatmulTestArchTraits::enable());
  using Atom = ::vecops::matmul::AMX_BF16F32;
  using Tiles = ::vecops::matmul::CacheTiling<
      meta::Const<32>, meta::Const<32>, meta::Const<32>>;
  using GenericTuning = ::vecops::matmul::GenericTiledTuning<
      Tiles, ::vecops::matmul::loop_order::MNK,
      ::vecops::matmul::PackingPolicy<
          ::vecops::matmul::PackingMode::never>,
      ::vecops::matmul::PackingPolicy<
          ::vecops::matmul::PackingMode::never>,
      ::vecops::matmul::AccBufferMode::workspace>;
  using Config = ops::MatmulConfig<
      Atom,
      ::vecops::matmul::family_selection::Require<
          ::vecops::matmul::kernel_family::GenericTiled>,
      kernel::matmul_policy::Automatic,
      GenericTuning>;
  constexpr nint_t M = 33;
  constexpr nint_t N = 35;
  constexpr nint_t K = 65;
  std::vector<bfloat16_t> a(M * K);
  std::vector<bfloat16_t> b(N * K);
  std::vector<float32_t> c(M * N);
  for (nint_t i = 0; i < M * K; ++i) a[i] = value<bfloat16_t>(i, 3);
  for (nint_t i = 0; i < N * K; ++i) b[i] = value<bfloat16_t>(i, 11);
  auto at = make_tensor(a.data(), make_layout(make_shape(cint<M>, cint<K>)));
  auto bt = make_tensor(b.data(), make_layout(make_shape(cint<N>, cint<K>)));
  auto ct = make_tensor(c.data(), make_layout(make_shape(cint<M>, cint<N>)));
  Config config{
      .generic_tiled = GenericTuning{
          .cache_tiling = Tiles{
              meta::cint<32>, meta::cint<32>, meta::cint<32>}}};
  auto operation = ops::matmul(config);
  kernel::Workspace storage(
      operation.required_workspace(cint<M>, cint<N>, cint<K>, at, bt, ct));
  auto workspace = storage.view();
  operation(workspace, cint<M>, cint<N>, cint<K>, at, bt, ct);
  std::vector<bfloat16_t> a_second = a;
  std::vector<bfloat16_t> b_second = b;
  std::vector<float32_t> c_second(M * N);
  auto at_second = make_tensor(
      a_second.data(), make_layout(make_shape(cint<M>, cint<K>)));
  auto bt_second = make_tensor(
      b_second.data(), make_layout(make_shape(cint<N>, cint<K>)));
  auto ct_second = make_tensor(
      c_second.data(), make_layout(make_shape(cint<M>, cint<N>)));
  workspace.reset();
  operation(workspace, cint<M>, cint<N>, cint<K>,
            at_second, bt_second, ct_second);
  for (nint_t i = 0; i < M; ++i) {
    for (nint_t j = 0; j < N; ++j) {
      float32_t expected = 0;
      for (nint_t kk = 0; kk < K; ++kk) {
        expected += static_cast<float32_t>(a[i * K + kk]) *
            static_cast<float32_t>(b[j * K + kk]);
      }
      EXPECT_NEAR(c[i * N + j], expected, 3.0e-3f);
      EXPECT_NEAR(c_second[i * N + j], expected, 3.0e-3f);
    }
  }
}

TEST(MatmulTest, GenericTilerSplitKReusesSmallVectorPhases) {
  ASSERT_TRUE(vecops::test::matmul::MatmulTestArchTraits::enable());
  using Atom = ::vecops::matmul::AMX_BF16F32;
  using Tiles = ::vecops::matmul::CacheTiling<
      meta::Const<1>, meta::Const<16>, meta::Const<32>>;
  using Never = ::vecops::matmul::PackingPolicy<
      ::vecops::matmul::PackingMode::never>;
  using Tuning = ::vecops::matmul::GenericTiledTuning<
      Tiles, ::vecops::matmul::loop_order::MNK, Never, Never,
      ::vecops::matmul::AccBufferMode::workspace>;
  using Config = ops::MatmulConfig<
      Atom,
      ::vecops::matmul::family_selection::Require<
          ::vecops::matmul::kernel_family::GenericTiled>,
      kernel::matmul_policy::Automatic, Tuning>;
  constexpr nint_t M = 1, N = 16, K = 65;
  std::vector<bfloat16_t> a(M * K), b(N * K);
  std::vector<float32_t> c(M * N);
  for (nint_t i = 0; i < M * K; ++i) a[i] = value<bfloat16_t>(i, 13);
  for (nint_t i = 0; i < N * K; ++i) b[i] = value<bfloat16_t>(i, 11);
  auto at = make_tensor(a.data(), make_layout(make_shape(cint<M>, cint<K>)));
  auto bt = make_tensor(b.data(), make_layout(make_shape(cint<N>, cint<K>)));
  auto ct = make_tensor(c.data(), make_layout(make_shape(cint<M>, cint<N>)));
  Config config{.generic_tiled = Tuning{
      .cache_tiling = Tiles{cint<M>, cint<N>, cint<32>}}};
  auto operation = ops::matmul(config);
  kernel::Workspace storage(operation.required_workspace(
      cint<M>, cint<N>, cint<K>, at, bt, ct));
  auto workspace = storage.view();
  operation(workspace, cint<M>, cint<N>, cint<K>, at, bt, ct);
  for (nint_t column = 0; column < N; ++column) {
    float32_t expected = 0;
    for (nint_t kk = 0; kk < K; ++kk)
      expected += static_cast<float32_t>(a[kk]) *
          static_cast<float32_t>(b[column * K + kk]);
    EXPECT_NEAR(c[column], expected, 3.0e-4f);
  }
}

TEST(MatmulTest, ForcedOutputAccumulatorReuseSupportsLargerSlots) {
  ASSERT_TRUE(vecops::test::matmul::MatmulTestArchTraits::enable());
  using Atom = ::vecops::matmul::AMX_BF16F32;
  using Tiles = ::vecops::matmul::CacheTiling<
      meta::Const<16>, meta::Const<16>, meta::Const<32>>;
  using GenericTuning = ::vecops::matmul::GenericTiledTuning<
      Tiles, ::vecops::matmul::loop_order::MNK,
      ::vecops::matmul::PackingPolicy<
          ::vecops::matmul::PackingMode::never>,
      ::vecops::matmul::PackingPolicy<
          ::vecops::matmul::PackingMode::never>,
      ::vecops::matmul::AccBufferMode::output>;
  using Config = ops::MatmulConfig<
      Atom,
      ::vecops::matmul::family_selection::Require<
          ::vecops::matmul::kernel_family::GenericTiled>,
      kernel::matmul_policy::Automatic,
      GenericTuning>;
  constexpr nint_t M = 17;
  constexpr nint_t N = 19;
  constexpr nint_t K = 65;
  std::vector<bfloat16_t> a(M * K);
  std::vector<bfloat16_t> b(N * K);
  std::vector<double> c(M * N);
  for (nint_t i = 0; i < M * K; ++i) a[i] = value<bfloat16_t>(i, 5);
  for (nint_t i = 0; i < N * K; ++i) b[i] = value<bfloat16_t>(i, 13);
  auto at = make_tensor(a.data(), make_layout(make_shape(cint<M>, cint<K>)));
  auto bt = make_tensor(b.data(), make_layout(make_shape(cint<N>, cint<K>)));
  auto ct = make_tensor(c.data(), make_layout(make_shape(cint<M>, cint<N>)));
  auto operation = ops::matmul(Config{
      .generic_tiled = GenericTuning{
          .cache_tiling = Tiles{
              meta::cint<16>, meta::cint<16>, meta::cint<32>}}});
  kernel::Workspace storage(
      operation.required_workspace(cint<M>, cint<N>, cint<K>, at, bt, ct));
  auto workspace = storage.view();
  operation(workspace, cint<M>, cint<N>, cint<K>, at, bt, ct);
  for (nint_t i = 0; i < M; ++i) {
    for (nint_t j = 0; j < N; ++j) {
      float expected = 0;
      for (nint_t kk = 0; kk < K; ++kk) {
        expected += static_cast<float>(a[i * K + kk]) *
            static_cast<float>(b[j * K + kk]);
      }
      EXPECT_NEAR(c[i * N + j], expected, 3.0e-3);
    }
  }
  std::vector<unsigned char> raw_output(
      static_cast<std::size_t>(M * N * sizeof(double) + 1));
  auto* misaligned = reinterpret_cast<double*>(raw_output.data() + 1);
  auto misaligned_ct = make_tensor(
      misaligned, make_layout(make_shape(cint<M>, cint<N>)));
  workspace.reset();
  EXPECT_ANY_THROW(operation(
      workspace, cint<M>, cint<N>, cint<K>, at, bt, misaligned_ct));
}

template <bool PackA, bool PackB>
void check_generic_full_k_packing() {
  SCOPED_TRACE(::testing::Message() << "PackA=" << PackA
                                    << " PackB=" << PackB);
  using Atom = ::vecops::matmul::AMX_BF16F32;
  using Tiles = ::vecops::matmul::CacheTiling<
      meta::Const<16>, meta::Const<16>, meta::Const<32>>;
  using APolicy = ::vecops::matmul::PackingPolicy<
      PackA ? ::vecops::matmul::PackingMode::always
            : ::vecops::matmul::PackingMode::never,
      ::vecops::matmul::PackingExtent::full_k>;
  using BPolicy = ::vecops::matmul::PackingPolicy<
      PackB ? ::vecops::matmul::PackingMode::always
            : ::vecops::matmul::PackingMode::never,
      ::vecops::matmul::PackingExtent::full_k>;
  using GenericTuning = ::vecops::matmul::GenericTiledTuning<
      Tiles, ::vecops::matmul::loop_order::MNK, APolicy, BPolicy,
      ::vecops::matmul::AccBufferMode::workspace>;
  using Config = ops::MatmulConfig<
      Atom,
      ::vecops::matmul::family_selection::Require<
          ::vecops::matmul::kernel_family::GenericTiled>,
      kernel::matmul_policy::Automatic, GenericTuning>;
  constexpr nint_t M = 17;
  constexpr nint_t N = 19;
  constexpr nint_t K = 65;
  std::vector<bfloat16_t> a(M * K);
  std::vector<bfloat16_t> b(N * K);
  std::vector<float> c(M * N);
  for (nint_t i = 0; i < M * K; ++i) a[i] = value<bfloat16_t>(i, 17);
  for (nint_t i = 0; i < N * K; ++i) b[i] = value<bfloat16_t>(i, 23);
  auto at = make_tensor(a.data(), make_layout(make_shape(cint<M>, cint<K>)));
  auto bt = make_tensor(b.data(), make_layout(make_shape(cint<N>, cint<K>)));
  auto ct = make_tensor(c.data(), make_layout(make_shape(cint<M>, cint<N>)));
  auto operation = ops::matmul(Config{
      .generic_tiled = GenericTuning{
          .cache_tiling = Tiles{
              meta::cint<16>, meta::cint<16>, meta::cint<32>}}});
  kernel::Workspace storage(
      operation.required_workspace(cint<M>, cint<N>, cint<K>, at, bt, ct));
  auto workspace = storage.view();
  operation(workspace, cint<M>, cint<N>, cint<K>, at, bt, ct);
  for (nint_t i = 0; i < M; ++i) {
    for (nint_t j = 0; j < N; ++j) {
      float expected = 0;
      for (nint_t kk = 0; kk < K; ++kk) {
        expected += static_cast<float>(a[i * K + kk]) *
            static_cast<float>(b[j * K + kk]);
      }
      EXPECT_NEAR(c[i * N + j], expected, 3.0e-3f);
    }
  }
}

TEST(MatmulTest, GenericTilerPacksOperandsIndependentlyAtFullK) {
  ASSERT_TRUE(vecops::test::matmul::MatmulTestArchTraits::enable());
  check_generic_full_k_packing<true, false>();
  check_generic_full_k_packing<false, true>();
  check_generic_full_k_packing<true, true>();
}

template <typename Order>
void check_generic_tiler_loop_order() {
  using Atom = ::vecops::matmul::AMX_BF16F32;
  using Tiles = ::vecops::matmul::CacheTiling<
      meta::Const<16>, meta::Const<16>, meta::Const<32>>;
  using Never = ::vecops::matmul::PackingPolicy<
      ::vecops::matmul::PackingMode::never>;
  using Tuning = ::vecops::matmul::GenericTiledTuning<
      Tiles, Order, Never, Never,
      ::vecops::matmul::AccBufferMode::workspace>;
  using Config = ops::MatmulConfig<
      Atom,
      ::vecops::matmul::family_selection::Require<
          ::vecops::matmul::kernel_family::GenericTiled>,
      kernel::matmul_policy::Automatic, Tuning>;
  constexpr nint_t M = 17, N = 19, K = 65;
  std::vector<bfloat16_t> a(M * K), b(N * K);
  std::vector<float32_t> c(M * N, -9.0f);
  for (nint_t i = 0; i < M * K; ++i) a[i] = value<bfloat16_t>(i, 13);
  for (nint_t i = 0; i < N * K; ++i) b[i] = value<bfloat16_t>(i, 11);
  auto at = make_tensor(a.data(), make_layout(make_shape(cint<M>, cint<K>)));
  auto bt = make_tensor(b.data(), make_layout(make_shape(cint<N>, cint<K>)));
  auto ct = make_tensor(c.data(), make_layout(make_shape(cint<M>, cint<N>)));
  auto operation = ops::matmul(Config{.generic_tiled = Tuning{
      .cache_tiling = Tiles{cint<16>, cint<16>, cint<32>}}});
  kernel::Workspace storage(operation.required_workspace(
      cint<M>, cint<N>, cint<K>, at, bt, ct));
  auto workspace = storage.view();
  operation(workspace, cint<M>, cint<N>, cint<K>, at, bt, ct);
  for (nint_t i = 0; i < M; ++i) {
    for (nint_t j = 0; j < N; ++j) {
      float32_t expected = 0;
      for (nint_t kk = 0; kk < K; ++kk)
        expected += static_cast<float32_t>(a[i * K + kk]) *
                    static_cast<float32_t>(b[j * K + kk]);
      EXPECT_NEAR(c[i * N + j], expected, 3.0e-3f)
          << "order=" << static_cast<int>(Order::first)
          << static_cast<int>(Order::second)
          << static_cast<int>(Order::third)
          << " m=" << i << " n=" << j;
    }
  }
}

TEST(MatmulTest, GenericTilerEveryLoopOrderIsCorrect) {
  ASSERT_TRUE(vecops::test::matmul::MatmulTestArchTraits::enable());
  check_generic_tiler_loop_order<::vecops::matmul::loop_order::MNK>();
  check_generic_tiler_loop_order<::vecops::matmul::loop_order::MKN>();
  check_generic_tiler_loop_order<::vecops::matmul::loop_order::NMK>();
  check_generic_tiler_loop_order<::vecops::matmul::loop_order::NKM>();
  check_generic_tiler_loop_order<::vecops::matmul::loop_order::KMN>();
  check_generic_tiler_loop_order<::vecops::matmul::loop_order::KNM>();
}

TEST(MatmulTest, TransformedInputsUseFullAndTailKPlans) {
  ASSERT_TRUE(vecops::test::matmul::MatmulTestArchTraits::enable());
  check_transformed_inputs(cint<64>);
  check_transformed_inputs(cint<65>);
}

TEST(MatmulTest, DynamicInnerStrideUsesGenericLoad) {
  ASSERT_TRUE(vecops::test::matmul::MatmulTestArchTraits::enable());
  check_dynamic_inner_stride(1);
  check_dynamic_inner_stride(2);
}

TEST(MatmulTest, BothPackedInputsUseFullTailAndRuntimeKPlans) {
  ASSERT_TRUE(vecops::test::matmul::MatmulTestArchTraits::enable());
  check_both_packed(cint<64>);
  check_both_packed(cint<65>);
  check_both_packed(Any{64});
  check_both_packed(Any{65});
  check_both_packed(Any{1024}, 17, 47);
  check_both_packed(Any{1024}, 20, 33);
  // Exercise the vertical packed-B N=1 traversal.
  check_both_packed(Any{65}, 32, 1);
}

#elif VECOPS_TARGET_SHARD_INDEX == 1

#if defined(HAS_AMX_FP16)
TEST(MatmulTest, NativeFP16UsesFullAndTailKPlans) {
  ASSERT_TRUE(vecops::test::matmul::MatmulTestArchTraits::enable());
  check_raw<::vecops::matmul::AMX_F16F32>(19, 21, 32);
  check_raw<::vecops::matmul::AMX_F16F32>(19, 21, 35);
}
#endif

TEST(MatmulTest, EndToEndMemoryComputeOutputConversions) {
  ASSERT_TRUE(vecops::test::matmul::MatmulTestArchTraits::enable());
  test::matmul::check_conversion<
      ::vecops::matmul::AMX_BF16F32, float32_t, float32_t, float32_t>(19, 21, 35);
  test::matmul::check_conversion<
      ::vecops::matmul::AMX_BF16F32, bfloat16_t, bfloat16_t, bfloat16_t>(19, 21, 35);
  test::matmul::check_conversion<
      ::vecops::matmul::AMX_BF16F32, bfloat16_t, bfloat16_t, bfloat16_t>(2, 2, 35);
#if defined(HAS_AMX_FP16)
  test::matmul::check_conversion<
      ::vecops::matmul::AMX_F16F32, float32_t, float32_t, float32_t>(19, 21, 35);
  test::matmul::check_conversion<
      ::vecops::matmul::AMX_F16F32, float16_t, float16_t, float16_t>(19, 21, 35);
#endif
}

TEST(MatmulTest, LargeConversionInputsUseAutoPacking) {
  ASSERT_TRUE(vecops::test::matmul::MatmulTestArchTraits::enable());
  test::matmul::check_conversion<
      ::vecops::matmul::AMX_BF16F32, float32_t, float32_t, float32_t>(
          cint<64>, cint<64>, cint<64>, true);
}

#elif VECOPS_TARGET_SHARD_INDEX == 2

TEST(MatmulTest, AllInt8SignednessCombinations) {
  ASSERT_TRUE(vecops::test::matmul::MatmulTestArchTraits::enable());
  check_raw<::vecops::matmul::AMX_I8I32<int8_t, int8_t>>(19, 21, 69);
  check_raw<::vecops::matmul::AMX_I8I32<int8_t, uint8_t>>(19, 21, 69);
  check_raw<::vecops::matmul::AMX_I8I32<uint8_t, int8_t>>(19, 21, 69);
  check_raw<::vecops::matmul::AMX_I8I32<uint8_t, uint8_t>>(19, 21, 69);
  // Large raw GEMV/GEMM crossover: exercise both orientations and a one-byte
  // K tail for every signedness combination.
  check_raw<::vecops::matmul::AMX_I8I32<int8_t, int8_t>>(1, 128, 65);
  check_raw<::vecops::matmul::AMX_I8I32<int8_t, uint8_t>>(128, 1, 65);
  check_raw<::vecops::matmul::AMX_I8I32<uint8_t, int8_t>>(1, 128, 65);
  check_raw<::vecops::matmul::AMX_I8I32<uint8_t, uint8_t>>(128, 1, 65);
  // The two mixed-signedness cases have a native VPDPBUSD mapping and use the
  // AVX-512 small-matrix crossover, including a non-multiple-of-64 K tail.
  check_raw<::vecops::matmul::AMX_I8I32<int8_t, uint8_t>>(1, 16, 129);
  check_raw<::vecops::matmul::AMX_I8I32<uint8_t, int8_t>>(16, 1, 129);
  check_raw<::vecops::matmul::AMX_I8I32<int8_t, int8_t>>(1, 16, 129);
  check_raw<::vecops::matmul::AMX_I8I32<uint8_t, uint8_t>>(16, 1, 129);
}

TEST(MatmulTest, SameSignSmallVectorPreservesWrapAndTail) {
  ASSERT_TRUE(vecops::test::matmul::MatmulTestArchTraits::enable());
  constexpr nint_t K = 262145;
  auto check = [=]<typename T>() {
    using Atom = ::vecops::matmul::AMX_I8I32<T, T>;
    const T input = [] {
      if constexpr (std::same_as<T, int8_t>) return int8_t{127};
      else return uint8_t{255};
    }();
    std::vector<T> a(static_cast<std::size_t>(K), input);
    std::vector<T> b(static_cast<std::size_t>(K), input);
    int32_t c{};
    auto at = make_tensor(
        a.data(), make_layout(make_shape(cint<1>, Any{K})));
    auto bt = make_tensor(
        b.data(), make_layout(make_shape(cint<1>, Any{K})));
    auto ct = make_tensor(
        &c, make_layout(make_shape(cint<1>, cint<1>)));
    auto operation = test::matmul::make_test_matmul_invocation(ops::MatmulConfig<Atom>{}, 1, 1, K, at, bt, ct);
    kernel::Workspace owner(operation.required_workspace());
    auto workspace = owner.view();
    operation(workspace);
    const uint32_t product = static_cast<uint32_t>(
        static_cast<int32_t>(input) * static_cast<int32_t>(input));
    const int32_t expected = std::bit_cast<int32_t>(
        product * static_cast<uint32_t>(K));
    EXPECT_EQ(c, expected) << "signed=" << std::same_as<T, int8_t>;
  };
  check.template operator()<int8_t>();
  check.template operator()<uint8_t>();
}

TEST(MatmulTest, ExplicitKernelFamiliesSelectSmallVectorOrGeneral) {
  ASSERT_TRUE(vecops::test::matmul::MatmulTestArchTraits::enable());
  using Atom = ::vecops::matmul::AMX_BF16F32;
  check_raw_family<
      Atom,
      ::vecops::matmul::family_selection::Require<
          ::vecops::matmul::kernel_family::SmallVector>>(
              cint<1>, cint<16>, cint<65>);
  // Runtime extents retain the same family when their actual shape is within
  // the shared profitability rule.
  check_raw_family<
      Atom,
      ::vecops::matmul::family_selection::Require<
          ::vecops::matmul::kernel_family::SmallVector>>(
              Any{1}, Any{16}, Any{65});
  check_raw_family<
      Atom,
      ::vecops::matmul::family_selection::Require<
          ::vecops::matmul::kernel_family::General>>(
              cint<1>, cint<16>, cint<65>);
  check_raw_family<
      Atom,
      ::vecops::matmul::family_selection::Prefer<
          ::vecops::matmul::kernel_family::SmallVector>>(
      cint<19>, cint<21>, cint<65>);
#if defined(HAS_AMX_INT8)
  using RequireSmall = ::vecops::matmul::family_selection::Require<
      ::vecops::matmul::kernel_family::SmallVector>;
  check_raw_family<::vecops::matmul::AMX_I8I32<int8_t, int8_t>, RequireSmall>(
      Any{1}, Any{16}, Any{129});
  check_raw_family<::vecops::matmul::AMX_I8I32<int8_t, uint8_t>, RequireSmall>(
      Any{16}, Any{1}, Any{129});
  check_raw_family<::vecops::matmul::AMX_I8I32<uint8_t, int8_t>, RequireSmall>(
      cint<1>, cint<16>, cint<129>);
  check_raw_family<::vecops::matmul::AMX_I8I32<uint8_t, uint8_t>, RequireSmall>(
      cint<16>, cint<1>, cint<129>);
#endif
}

TEST(MatmulTest, ResidualSplitCoversProfitableAndGeneralizedShapes) {
  using RequireResidual = ::vecops::matmul::family_selection::Require<
      ::vecops::matmul::kernel_family::ResidualSplit>;
  check_both_packed<Any, RequireResidual>(Any{1024}, 17, 33);
  check_both_packed<Any, RequireResidual>(Any{1024}, 33, 17);
  check_both_packed<Any, RequireResidual>(Any{257}, 35, 53);
  check_both_packed<Any, RequireResidual>(Any{257}, 53, 35);
  check_raw_family<::vecops::matmul::AMX_BF16F32, RequireResidual>(
      Any{35}, Any{53}, Any{257});
#if defined(HAS_AMX_INT8)
  check_raw_family<
      ::vecops::matmul::AMX_I8I32<int8_t, int8_t>, RequireResidual>(
          Any{53}, Any{35}, Any{257});
#endif
}

TEST(MatmulTest, OperandAMayBePrepacked) {
  ASSERT_TRUE(vecops::test::matmul::MatmulTestArchTraits::enable());
  check_mixed_packing<::vecops::matmul::Operand::A>();
  check_mixed_packing<::vecops::matmul::Operand::A>(1, 16, 65);
}

TEST(MatmulTest, OperandBMayBePrepacked) {
  ASSERT_TRUE(vecops::test::matmul::MatmulTestArchTraits::enable());
  check_mixed_packing<::vecops::matmul::Operand::B>();
}

#elif VECOPS_TARGET_SHARD_INDEX == 3

TEST(MatmulTest, CPrologueAndEpilogueUseDataAccess) {
  ASSERT_TRUE(vecops::test::matmul::MatmulTestArchTraits::enable());
  constexpr nint_t M = 19, N = 23, K = 35;
  std::vector<bfloat16_t> a(M * K), b(N * K);
  std::vector<float32_t> c(M * N, 1.25f);
  for (nint_t i = 0; i < M * K; ++i) a[i] = value<bfloat16_t>(i, 13);
  for (nint_t i = 0; i < N * K; ++i) b[i] = value<bfloat16_t>(i, 11);
  auto at = make_tensor(a.data(), make_layout(make_shape(Any{M}, Any{K})));
  auto bt = make_tensor(b.data(), make_layout(make_shape(Any{N}, Any{K})));
  auto ct = make_tensor(c.data(), make_layout(make_shape(Any{M}, Any{N})));
  auto epilogue = make_elementwise_vec_transform<float32_t, float32_t>(
      [](auto tag, auto x) VECOPS_KERNEL_LAMBDA {
        return vec::mul(x, vec::fill(tag, 2.0f));
      });
  auto operation = test::matmul::make_test_matmul_invocation(
      ops::MatmulConfig<::vecops::matmul::AMX_BF16F32>{},
      M, N, K, at, bt, input<float32_t>(ct), output<float32_t>(ct, epilogue));
  kernel::Workspace owner(operation.required_workspace());
  auto workspace = owner.view();
  operation(workspace);
  for (nint_t i = 0; i < M; ++i) {
    for (nint_t j = 0; j < N; ++j) {
      float expected = 1.25f;
      for (nint_t kk = 0; kk < K; ++kk)
        expected += static_cast<float>(a[i * K + kk]) *
                    static_cast<float>(b[j * K + kk]);
      EXPECT_NEAR(c[i * N + j], 2.0f * expected, 3.0e-4f);
    }
  }
}

TEST(MatmulTest, SplitKGemmAddRoutesTransformedCThroughWorkspace) {
  ASSERT_TRUE(vecops::test::matmul::MatmulTestArchTraits::enable());
  using Atom = ::vecops::matmul::AMX_BF16F32;
  using Tiles = ::vecops::matmul::CacheTiling<
      meta::Const<16>, meta::Const<16>, meta::Const<32>>;
  using Never = ::vecops::matmul::PackingPolicy<
      ::vecops::matmul::PackingMode::never>;
  using Tuning = ::vecops::matmul::GenericTiledTuning<
      Tiles, ::vecops::matmul::loop_order::NKM, Never, Never,
      ::vecops::matmul::AccBufferMode::workspace>;
  using Config = ops::MatmulConfig<
      Atom,
      ::vecops::matmul::family_selection::Require<
          ::vecops::matmul::kernel_family::GenericTiled>,
      kernel::matmul_policy::Automatic, Tuning>;
  constexpr nint_t M = 19, N = 23, K = 65;
  std::vector<bfloat16_t> a(M * K), b(N * K);
  std::vector<float32_t> initial(M * N), separate(M * N, -99.0f);
  for (nint_t i = 0; i < M * K; ++i) a[i] = value<bfloat16_t>(i, 13);
  for (nint_t i = 0; i < N * K; ++i) b[i] = value<bfloat16_t>(i, 11);
  for (nint_t i = 0; i < M * N; ++i)
    initial[i] = value<float32_t>(i, 17);
  auto at = make_tensor(a.data(), make_layout(make_shape(cint<M>, cint<K>)));
  auto bt = make_tensor(b.data(), make_layout(make_shape(cint<N>, cint<K>)));
  auto input_t = make_tensor(
      initial.data(), make_layout(make_shape(cint<M>, cint<N>)));
  auto output_t = make_tensor(
      separate.data(), make_layout(make_shape(cint<M>, cint<N>)));
  auto identity = make_elementwise_vec_transform<float32_t, float32_t>(
      [](auto, auto x) VECOPS_KERNEL_LAMBDA { return x; });
  auto scale = make_elementwise_vec_transform<float32_t, float32_t>(
      [](auto tag, auto x) VECOPS_KERNEL_LAMBDA {
        return vec::mul(x, vec::fill(tag, 2.0f));
      });
  auto operation = ops::matmul(Config{.generic_tiled = Tuning{
      .cache_tiling = Tiles{cint<16>, cint<16>, cint<32>}}});
  auto c_input = input<float32_t>(input_t, identity);
  auto c_output = output<float32_t>(output_t, scale);
  kernel::Workspace owner(operation.required_workspace(
      cint<M>, cint<N>, cint<K>, at, bt, c_input, c_output));
  auto workspace = owner.view();
  operation(workspace, cint<M>, cint<N>, cint<K>,
            at, bt, c_input, c_output);
  for (nint_t i = 0; i < M; ++i) {
    for (nint_t j = 0; j < N; ++j) {
      float expected = initial[i * N + j];
      for (nint_t kk = 0; kk < K; ++kk)
        expected += static_cast<float>(a[i * K + kk]) *
                    static_cast<float>(b[j * K + kk]);
      EXPECT_NEAR(separate[i * N + j], 2.0f * expected, 3.0e-3f)
          << "m=" << i << " n=" << j;
    }
  }
}

TEST(MatmulTest, FullCDataAccessUsesCompileTimeExtents) {
  ASSERT_TRUE(vecops::test::matmul::MatmulTestArchTraits::enable());
  constexpr nint_t M = 32, N = 32, K = 64;
  std::vector<bfloat16_t> a(M * K), b(N * K);
  std::vector<float32_t> c(M * N, 1.25f);
  for (nint_t i = 0; i < M * K; ++i) a[i] = value<bfloat16_t>(i, 13);
  for (nint_t i = 0; i < N * K; ++i) b[i] = value<bfloat16_t>(i, 11);
  auto at = make_tensor(
      a.data(), make_layout(make_shape(cint<M>, cint<K>)));
  auto bt = make_tensor(
      b.data(), make_layout(make_shape(cint<N>, cint<K>)));
  auto ct = make_tensor(
      c.data(), make_layout(make_shape(cint<M>, cint<N>)));
  auto identity = make_elementwise_vec_transform<float32_t, float32_t>(
      [](auto, auto x) VECOPS_KERNEL_LAMBDA { return x; });
  auto scale = make_elementwise_vec_transform<float32_t, float32_t>(
      [](auto tag, auto x) VECOPS_KERNEL_LAMBDA {
        return vec::mul(x, vec::fill(tag, 2.0f));
      });
  auto operation = test::matmul::make_test_matmul_invocation(
      ops::MatmulConfig<::vecops::matmul::AMX_BF16F32>{},
      cint<M>, cint<N>, cint<K>, at, bt,
      input<float32_t>(ct, identity), output<float32_t>(ct, scale));
  kernel::Workspace owner(operation.required_workspace());
  auto workspace = owner.view();
  operation(workspace);
  for (nint_t i = 0; i < M; ++i) {
    for (nint_t j = 0; j < N; ++j) {
      float expected = 1.25f;
      for (nint_t kk = 0; kk < K; ++kk)
        expected += static_cast<float>(a[i * K + kk]) *
                    static_cast<float>(b[j * K + kk]);
      EXPECT_NEAR(c[i * N + j], 2.0f * expected, 3.0e-4f);
    }
  }
}

TEST(MatmulTest, BroadcastBiasAndReluAreFusedThroughDataAccess) {
  ASSERT_TRUE(vecops::test::matmul::MatmulTestArchTraits::enable());
  test::matmul::check_bias_relu<::vecops::matmul::AMX_BF16F32>(19, 23, 65);
  test::matmul::check_bias_relu<::vecops::matmul::AMX_BF16F32>(1, 16, 65);
}

TEST(MatmulTest, AsymmetricU8S8UsesColumnSumCorrection) {
  ASSERT_TRUE(vecops::test::matmul::MatmulTestArchTraits::enable());
#if defined(HAS_AMX_INT8)
  test::matmul::check_asymmetric_quantized<
      ::vecops::matmul::AMX_I8I32<uint8_t, int8_t>>(19, 23, 65);
  test::matmul::check_asymmetric_quantized<
      ::vecops::matmul::AMX_I8I32<uint8_t, int8_t>>(
          cint<64>, cint<128>, cint<128>, true);
#endif
}

TEST(MatmulTest, CompensatedPackedBFeedsAsymmetricCPrologue) {
  ASSERT_TRUE(vecops::test::matmul::MatmulTestArchTraits::enable());
#if defined(HAS_AMX_INT8)
  using Atom = ::vecops::matmul::AMX_I8I32<uint8_t, int8_t>;
  constexpr nint_t M = 19, N = 23, K = 65;
  constexpr int32_t ZeroPointA = 3;
  std::vector<uint8_t> a(M * K);
  std::vector<int8_t> b(N * K);
  std::vector<int32_t> compensation(N);
  std::vector<int32_t> c(M * N);
  for (nint_t i = 0; i < M * K; ++i)
    a[static_cast<std::size_t>(i)] =
        static_cast<uint8_t>((i * 7) % 17 + ZeroPointA);
  for (nint_t i = 0; i < N * K; ++i)
    b[static_cast<std::size_t>(i)] =
        static_cast<int8_t>((i * 5) % 31 - 15);
  auto a_tensor = make_tensor(
      a.data(), make_layout(make_shape(cint<M>, cint<K>)));
  auto b_layout = make_layout(make_shape(cint<N>, cint<K>));
  auto b_tensor = make_tensor(b.data(), b_layout);
  auto packed_layout = ::vecops::matmul::packed_layout<
      Atom, ::vecops::matmul::Operand::B>(b_layout);
  kernel::Workspace packed_owner(
      numel(packed_layout) * static_cast<nint_t>(sizeof(int8_t)) + 64);
  auto packed_workspace = packed_owner.view();
  auto* packed = static_cast<int8_t*>(packed_workspace.allocate(
      numel(packed_layout) * static_cast<nint_t>(sizeof(int8_t)), 64));
  auto packed_tensor = make_tensor(packed, packed_layout);
  auto compensation_vector = make_tensor(
      compensation.data(), make_layout(make_shape(cint<N>)));
  ExecutionSession pack_execution{};
  ::vecops::matmul::details::run_matmul_pack_b_compensated<Atom>(
      pack_execution, b_tensor, packed_tensor,
      compensation_vector, ZeroPointA);

  auto correction_tensor = make_tensor(
      compensation.data(),
      make_layout(
          make_shape(cint<M>, cint<N>),
          make_strides(cint<0>, cint<1>)));
  auto c_tensor = make_tensor(
      c.data(), make_layout(make_shape(cint<M>, cint<N>)));
  auto operation = test::matmul::make_test_matmul_invocation(ops::MatmulConfig<Atom>{},
      cint<M>, cint<N>, cint<K>, a_tensor, packed_tensor,
      tensor::input<int32_t>(correction_tensor), c_tensor);
  kernel::Workspace owner(operation.required_workspace());
  auto workspace = owner.view();
  operation(workspace);
  for (nint_t row = 0; row < M; ++row) {
    for (nint_t col = 0; col < N; ++col) {
      int32_t expected = 0;
      for (nint_t kk = 0; kk < K; ++kk) {
        expected +=
            (static_cast<int32_t>(a[static_cast<std::size_t>(row * K + kk)]) -
             ZeroPointA) *
            static_cast<int32_t>(b[static_cast<std::size_t>(col * K + kk)]);
      }
      EXPECT_EQ(c[static_cast<std::size_t>(row * N + col)], expected)
          << "m=" << row << " n=" << col;
    }
  }
#endif
}

#elif VECOPS_TARGET_SHARD_INDEX == 4

TEST(MatmulTest, DynamicAndConstExtentPairs) {
  ASSERT_TRUE(vecops::test::matmul::MatmulTestArchTraits::enable());
  test::matmul::check_native_extent_pair<::vecops::matmul::AMX_BF16F32, 16, 16, 35>();
  test::matmul::check_native_extent_pair<::vecops::matmul::AMX_BF16F32, 16, 32, 67>();
  test::matmul::check_native_extent_pair<::vecops::matmul::AMX_BF16F32, 37, 51, 79>();
  test::matmul::check_native_extent_pair<::vecops::matmul::AMX_BF16F32, 1, 128, 65>();
  test::matmul::check_native_extent_pair<::vecops::matmul::AMX_BF16F32, 64, 64, 64>();
  test::matmul::check_native_extent_pair<::vecops::matmul::AMX_BF16F32, 35, 53, 257>();
#if defined(HAS_AMX_FP16)
  test::matmul::check_native_extent_pair<::vecops::matmul::AMX_F16F32, 19, 21, 65>();
#endif
#if defined(HAS_AMX_INT8)
  test::matmul::check_native_extent_pair<
      ::vecops::matmul::AMX_I8I32<int8_t, uint8_t>, 35, 53, 257>();
  test::matmul::check_native_extent_pair<
      ::vecops::matmul::AMX_I8I32<uint8_t, int8_t>, 64, 64, 128>();
#endif
}

#endif

} // namespace

#endif // VECOPS_TARGET_SHARD_ACTIVE

#else

#if defined(VECOPS_TARGET_SHARD_ACTIVE)

#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <type_traits>
#include <vector>

#include "vecops/ops/Matmul.h"
#include "vecops/ops/MatmulPack.h"

#include "MatmulConversionTestCommon.h"

namespace {

static_assert(VECOPS_TARGET_SHARD_COUNT == 5);

using namespace vecops;
using namespace vecops::meta;
using namespace vecops::tensor;

using SMEF32Tile2x2M = decltype(cint<2> * ::vecops::matmul::SME_F32F32::M_R);

#if defined(HAS_FIXED_STREAMING_SVE_BITS)
static_assert(std::same_as<
    std::remove_cv_t<decltype(
        vec::details::sme::streaming_vector_bytes_value())>,
    Const<FIXED_STREAMING_SVE_BITS / 8>>);
static_assert(std::same_as<
    std::remove_cv_t<decltype(::vecops::matmul::SME_F32F32::M_R)>,
    Const<FIXED_STREAMING_SVE_BITS / 8 / sizeof(float32_t)>>);
static_assert(std::same_as<
        std::remove_cv_t<SMEF32Tile2x2M>,
    Const<2 * FIXED_STREAMING_SVE_BITS / 8 / sizeof(float32_t)>>);
#else
static_assert(std::same_as<
    std::remove_cv_t<decltype(
        vec::details::sme::streaming_vector_bytes_value())>,
    Dynamic<16, 16, 256>>);
static_assert(std::same_as<
    std::remove_cv_t<decltype(::vecops::matmul::SME_F32F32::M_R)>,
    Dynamic<4, 4, 64>>);
static_assert(std::same_as<
        std::remove_cv_t<SMEF32Tile2x2M>,
    Dynamic<8, 8, 128>>);
#endif
static_assert(std::same_as<
    std::remove_cv_t<decltype(::vecops::matmul::SME_F32F32::K_R)>, Const<1>>);
static_assert(kernel::matmul_details::sme::has_bounded_tile_axis_v<
              ::vecops::matmul::SME_F32F32, Const<16>, Any>);
static_assert(kernel::matmul_details::sme::has_bounded_tile_axis_v<
              ::vecops::matmul::SME_F32F32, Dynamic<1, 0, 32>, Any>);
static_assert(!kernel::matmul_details::sme::has_bounded_tile_axis_v<
              ::vecops::matmul::SME_F32F32, Any, Any>);

template <typename T>
T value(nint_t index, int modulus) {
  if constexpr (std::same_as<T, uint8_t>) {
    return static_cast<T>(index % 7);
  } else {
    const int x = static_cast<int>(index % modulus) - modulus / 2;
    if constexpr (std::is_integral_v<T>) return static_cast<T>(x);
    else return T(static_cast<float>(x) / static_cast<float>(modulus));
  }
}

template <typename Atom,
          typename TilePolicy = kernel::matmul_policy::Automatic,
          typename FamilySelection =
              ::vecops::matmul::family_selection::Automatic,
          typename M, typename N, typename K>
void check_raw(M m_value, N n_value, K k_value) {
  const nint_t m = static_cast<nint_t>(m_value);
  const nint_t n = static_cast<nint_t>(n_value);
  const nint_t k = static_cast<nint_t>(k_value);
  using TA = typename Atom::TA;
  using TB = typename Atom::TB;
  using Acc = typename Atom::TAcc;
  std::vector<TA> a(static_cast<std::size_t>(m * k));
  std::vector<TB> b(static_cast<std::size_t>(n * k));
  std::vector<Acc> c(static_cast<std::size_t>(m * n), Acc{});
  for (nint_t i = 0; i < m * k; ++i) a[i] = value<TA>(i, 13);
  for (nint_t i = 0; i < n * k; ++i) b[i] = value<TB>(i, 11);
  auto at = make_tensor(
      a.data(), make_layout(make_shape(m_value, k_value)));
  auto bt = make_tensor(
      b.data(), make_layout(make_shape(n_value, k_value)));
  auto ct = make_tensor(
      c.data(), make_layout(make_shape(m_value, n_value)));
  using Config = ops::MatmulConfig<Atom, FamilySelection, TilePolicy>;
  auto operation = test::matmul::make_test_matmul_invocation(
      Config{},
      m_value, n_value, k_value, at, bt, ct);
  using Operation = decltype(operation);
  static_assert(std::same_as<
      typename Operation::MExtentType, meta::to_value_t<M>>);
  static_assert(std::same_as<
      typename Operation::NExtentType, meta::to_value_t<N>>);
  static_assert(std::same_as<
      typename Operation::KExtentType, meta::to_value_t<K>>);
  kernel::Workspace operation_storage(operation.required_workspace());
  auto operation_workspace = operation_storage.view();
  ExecutionSession execution{operation_workspace};
  operation(execution);
  for (nint_t i = 0; i < m; ++i) {
    for (nint_t j = 0; j < n; ++j) {
      Acc expected{};
      for (nint_t kk = 0; kk < k; ++kk) {
        expected += static_cast<Acc>(a[i * k + kk]) *
                    static_cast<Acc>(b[j * k + kk]);
      }
      if constexpr (std::is_floating_point_v<Acc>) {
        EXPECT_NEAR(c[i * n + j], expected, 3.0e-4f)
            << "m=" << i << " n=" << j;
      } else {
        EXPECT_EQ(c[i * n + j], expected)
            << "m=" << i << " n=" << j;
      }
    }
  }
}

VECOPS_NOINLINE void check_dynamic_inner_stride(nint_t inner_stride) {
  using Atom = ::vecops::matmul::SME_BF16F32;
  using T = typename Atom::TA;
  using Acc = typename Atom::TAcc;
  constexpr nint_t m = 19;
  constexpr nint_t n = 21;
  constexpr nint_t k = 23;
  const nint_t a_row_stride = k * inner_stride + 3;
  const nint_t b_row_stride = k * inner_stride + 5;
  std::vector<T> a(static_cast<std::size_t>(m * a_row_stride));
  std::vector<T> b(static_cast<std::size_t>(n * b_row_stride));
  std::vector<Acc> c(static_cast<std::size_t>(m * n), Acc{});
  for (nint_t i = 0; i < m; ++i) {
    for (nint_t kk = 0; kk < k; ++kk) {
      a[i * a_row_stride + kk * inner_stride] = value<T>(i * k + kk, 13);
    }
  }
  for (nint_t j = 0; j < n; ++j) {
    for (nint_t kk = 0; kk < k; ++kk) {
      b[j * b_row_stride + kk * inner_stride] = value<T>(j * k + kk, 11);
    }
  }
  auto al = make_layout(
      make_shape(m, k), make_strides(a_row_stride, inner_stride));
  auto bl = make_layout(
      make_shape(n, k), make_strides(b_row_stride, inner_stride));
  static_assert(!std::same_as<stride_type_t<1, decltype(al)>, Const<1>>);
  static_assert(!std::same_as<stride_type_t<1, decltype(bl)>, Const<1>>);
  auto at = make_tensor(a.data(), al);
  auto bt = make_tensor(b.data(), bl);
  auto ct = make_tensor(c.data(), make_layout(make_shape(m, n)));
  auto operation = test::matmul::make_test_matmul_invocation(ops::MatmulConfig<Atom>{}, m, n, k, at, bt, ct);
  kernel::Workspace operation_owner(operation.required_workspace());
  auto operation_workspace = operation_owner.view();
  ExecutionSession execution{operation_workspace};
  operation(execution);
  for (nint_t i = 0; i < m; ++i) {
    for (nint_t j = 0; j < n; ++j) {
      Acc expected{};
      for (nint_t kk = 0; kk < k; ++kk) {
        expected += static_cast<Acc>(
            a[i * a_row_stride + kk * inner_stride]) *
            static_cast<Acc>(b[j * b_row_stride + kk * inner_stride]);
      }
      EXPECT_NEAR(c[i * n + j], expected, 3.0e-4f)
          << "inner_stride=" << inner_stride << " m=" << i << " n=" << j;
    }
  }
}

template <::vecops::matmul::Operand Side>
void check_mixed_packing() {
  using Atom = ::vecops::matmul::SME_F32F32;
  const nint_t lanes = vec::details::sme::streaming_lanes<float32_t>();
  const nint_t m = 2 * lanes + 3;
  const nint_t n = 3 * lanes + 5;
  const nint_t k = 19;
  std::vector<float32_t> a(m * k), b(n * k), c(m * n);
  for (nint_t i = 0; i < m * k; ++i) a[i] = value<float32_t>(i, 13);
  for (nint_t i = 0; i < n * k; ++i) b[i] = value<float32_t>(i, 11);
  auto al = make_layout(make_shape(Any{m}, Any{k}));
  auto bl = make_layout(make_shape(Any{n}, Any{k}));
  auto at = make_tensor(a.data(), al);
  auto bt = make_tensor(b.data(), bl);
  const auto& source_layout = Side == ::vecops::matmul::Operand::A ? al : bl;
  auto packed_layout = ::vecops::matmul::packed_layout<Atom, Side>(source_layout);
  kernel::Workspace packed_owner(
      numel(packed_layout) * static_cast<nint_t>(sizeof(float32_t)) + 64);
  auto packed_workspace = packed_owner.view();
  auto* packed = static_cast<float32_t*>(packed_workspace.allocate(
      numel(packed_layout) * static_cast<nint_t>(sizeof(float32_t)), 64));
  auto packed_tensor = make_tensor(packed, packed_layout);
  ExecutionSession execution{};
  if constexpr (Side == ::vecops::matmul::Operand::A)
    ::vecops::matmul::details::run_matmul_pack<Atom, Side>(execution, at, packed_tensor);
  else
    ::vecops::matmul::details::run_matmul_pack<Atom, Side>(execution, bt, packed_tensor);
  auto ct = make_tensor(c.data(), make_layout(make_shape(Any{m}, Any{n})));
  auto operation = [&] {
    if constexpr (Side == ::vecops::matmul::Operand::A)
      return test::matmul::make_test_matmul_invocation(ops::MatmulConfig<Atom>{}, m, n, k, packed_tensor, bt, ct);
    else
      return test::matmul::make_test_matmul_invocation(ops::MatmulConfig<Atom>{}, m, n, k, at, packed_tensor, ct);
  }();
  kernel::Workspace operation_owner(operation.required_workspace());
  auto operation_workspace = operation_owner.view();
  ExecutionSession operation_execution{operation_workspace};
  operation(operation_execution);
  for (nint_t i = 0; i < m; ++i) {
    for (nint_t j = 0; j < n; ++j) {
      float expected = 0;
      for (nint_t kk = 0; kk < k; ++kk)
        expected += a[i * k + kk] * b[j * k + kk];
      EXPECT_NEAR(c[i * n + j], expected, 3.0e-4f);
    }
  }
}

template <meta::ValueType KExtent>
void check_fast_packed_path(KExtent k_value) {
  using Atom = ::vecops::matmul::SME_BF16F32;
  using T = typename Atom::TA;
  const nint_t lanes = vec::details::sme::streaming_lanes<float32_t>();
  const nint_t m = 2 * lanes + 3;
  const nint_t n = 2 * lanes + 5;
  const nint_t k = static_cast<nint_t>(k_value);
  std::vector<T> a(m * k), b(n * k);
  std::vector<float32_t> c(m * n);
  for (nint_t i = 0; i < m * k; ++i) a[i] = value<T>(i, 13);
  for (nint_t i = 0; i < n * k; ++i) b[i] = value<T>(i, 11);
  auto al = make_layout(make_shape(Any{m}, k_value));
  auto bl = make_layout(make_shape(Any{n}, k_value));
  auto at = make_tensor(a.data(), al);
  auto bt = make_tensor(b.data(), bl);
  auto apl = ::vecops::matmul::packed_layout<Atom, ::vecops::matmul::Operand::A>(al);
  auto bpl = ::vecops::matmul::packed_layout<Atom, ::vecops::matmul::Operand::B>(bl);
  kernel::Workspace packed_owner(
      (numel(apl) + numel(bpl)) *
          static_cast<nint_t>(sizeof(T)) +
      128);
  auto packed_workspace = packed_owner.view();
  auto* ap = static_cast<T*>(packed_workspace.allocate(
      numel(apl) * static_cast<nint_t>(sizeof(T)), 64));
  auto* bp = static_cast<T*>(packed_workspace.allocate(
      numel(bpl) * static_cast<nint_t>(sizeof(T)), 64));
  auto apt = make_tensor(ap, apl);
  auto bpt = make_tensor(bp, bpl);
  ExecutionSession execution{};
  ::vecops::matmul::details::run_matmul_pack<Atom, ::vecops::matmul::Operand::A>(execution, at, apt);
  ::vecops::matmul::details::run_matmul_pack<Atom, ::vecops::matmul::Operand::B>(execution, bt, bpt);
  auto ct = make_tensor(c.data(), make_layout(make_shape(Any{m}, Any{n})));
  auto operation = test::matmul::make_test_matmul_invocation(ops::MatmulConfig<Atom>{}, m, n, k_value, apt, bpt, ct);
  operation(execution);
  for (nint_t i = 0; i < m; ++i) {
    for (nint_t j = 0; j < n; ++j) {
      float expected = 0;
      for (nint_t kk = 0; kk < k; ++kk)
        expected += static_cast<float>(a[i * k + kk]) *
                    static_cast<float>(b[j * k + kk]);
      EXPECT_NEAR(c[i * n + j], expected, 5.0e-4f)
          << "m=" << i << " n=" << j;
    }
  }
}

template <typename Atom,
          meta::ValueType MExtent, meta::ValueType NExtent,
          meta::ValueType KExtent>
void check_required_packed_dot(
    MExtent m_value, NExtent n_value, KExtent k_value) {
  using T = typename Atom::TA;
  using Acc = typename Atom::TAcc;
  const nint_t m = static_cast<nint_t>(m_value);
  const nint_t n = static_cast<nint_t>(n_value);
  const nint_t k = static_cast<nint_t>(k_value);
  std::vector<T> a(static_cast<std::size_t>(m * k));
  std::vector<T> b(static_cast<std::size_t>(n * k));
  std::vector<Acc> c(static_cast<std::size_t>(m * n), Acc{});
  for (nint_t i = 0; i < m * k; ++i) a[i] = value<T>(i, 13);
  for (nint_t i = 0; i < n * k; ++i) b[i] = value<T>(i, 11);
  auto al = make_layout(make_shape(m_value, k_value));
  auto bl = make_layout(make_shape(n_value, k_value));
  auto at = make_tensor(a.data(), al);
  auto bt = make_tensor(b.data(), bl);
  auto apl = ::vecops::matmul::packed_layout<
      Atom, ::vecops::matmul::Operand::A>(al);
  auto bpl = ::vecops::matmul::packed_layout<
      Atom, ::vecops::matmul::Operand::B>(bl);
  kernel::Workspace packed_owner(
      (numel(apl) + numel(bpl)) * static_cast<nint_t>(sizeof(T)) + 128);
  auto packed_workspace = packed_owner.view();
  auto* ap = static_cast<T*>(packed_workspace.allocate(
      numel(apl) * static_cast<nint_t>(sizeof(T)), 64));
  auto* bp = static_cast<T*>(packed_workspace.allocate(
      numel(bpl) * static_cast<nint_t>(sizeof(T)), 64));
  auto apt = make_tensor(ap, apl);
  auto bpt = make_tensor(bp, bpl);
  ExecutionSession execution{};
  ::vecops::matmul::details::run_matmul_pack<
      Atom, ::vecops::matmul::Operand::A>(execution, at, apt);
  ::vecops::matmul::details::run_matmul_pack<
      Atom, ::vecops::matmul::Operand::B>(execution, bt, bpt);
  auto ct = make_tensor(c.data(), make_layout(make_shape(m_value, n_value)));
  using Config = ops::MatmulConfig<
      Atom,
      ::vecops::matmul::family_selection::Require<
          ::vecops::matmul::kernel_family::PackedDot>>;
  auto operation = test::matmul::make_test_matmul_invocation(
      Config{}, m_value, n_value, k_value, apt, bpt, ct);
  operation(execution);
  for (nint_t i = 0; i < m; ++i) {
    for (nint_t j = 0; j < n; ++j) {
      Acc expected{};
      for (nint_t kk = 0; kk < k; ++kk)
        expected += static_cast<Acc>(a[i * k + kk]) *
                    static_cast<Acc>(b[j * k + kk]);
      if constexpr (std::is_floating_point_v<Acc>) {
        EXPECT_NEAR(c[i * n + j], expected, 5.0e-4f);
      } else {
        EXPECT_EQ(c[i * n + j], expected);
      }
    }
  }
}

#if VECOPS_TARGET_SHARD_INDEX == 0

TEST(MatmulTest, AllZA32MicrokernelShapesAndKTails) {
  const nint_t l = vec::details::sme::streaming_lanes<float32_t>();
  check_raw<::vecops::matmul::SME_F32F32>(l, l, 19);
  check_raw<::vecops::matmul::SME_F32F32>(l, 2 * l, 19);
  check_raw<::vecops::matmul::SME_F32F32>(l, 3 * l, 19);
  check_raw<::vecops::matmul::SME_F32F32>(l, 4 * l, 19);
  check_raw<::vecops::matmul::SME_F32F32>(2 * l, l, 19);
  check_raw<::vecops::matmul::SME_F32F32>(3 * l, l, 19);
  check_raw<::vecops::matmul::SME_F32F32>(4 * l, l, 19);
  check_raw<::vecops::matmul::SME_F32F32>(2 * l, 2 * l, 19);
  check_raw<::vecops::matmul::SME_F32F32>(2 * l + 3, 3 * l + 5, 19);
}

TEST(MatmulTest, GenericTilerRunsCacheAndKBlocksOnSME) {
  using Atom = ::vecops::matmul::SME_F32F32;
  using TileAxis = std::remove_cv_t<decltype(Atom::M_R)>;
  using Tiles = ::vecops::matmul::CacheTiling<
      TileAxis, TileAxis, meta::Const<4>>;
  using GenericTuning = ::vecops::matmul::GenericTiledTuning<
      Tiles, ::vecops::matmul::loop_order::MKN,
      ::vecops::matmul::PackingPolicy<
          ::vecops::matmul::PackingMode::never>,
      ::vecops::matmul::PackingPolicy<
          ::vecops::matmul::PackingMode::never>,
      ::vecops::matmul::AccBufferMode::workspace>;
  using Config = ops::MatmulConfig<
      Atom,
      ::vecops::matmul::family_selection::Require<
          ::vecops::matmul::kernel_family::GenericTiled>,
      kernel::matmul_policy::Automatic,
      GenericTuning>;
  const nint_t tile = vec::details::sme::streaming_lanes<float32_t>();
  const nint_t m = tile + 3;
  const nint_t n = tile + 5;
  constexpr nint_t k = 9;
  std::vector<float32_t> a(m * k);
  std::vector<float32_t> b(n * k);
  std::vector<float32_t> c(m * n);
  for (nint_t i = 0; i < m * k; ++i) a[i] = value<float32_t>(i, 17);
  for (nint_t i = 0; i < n * k; ++i) b[i] = value<float32_t>(i, 23);
  auto at = make_tensor(
      a.data(), make_layout(make_shape(Any{m}, cint<k>)));
  auto bt = make_tensor(
      b.data(), make_layout(make_shape(Any{n}, cint<k>)));
  auto ct = make_tensor(
      c.data(), make_layout(make_shape(Any{m}, Any{n})));
  auto operation = ops::matmul(Config{
      .generic_tiled = GenericTuning{
          .cache_tiling = Tiles{
              TileAxis{tile}, TileAxis{tile}, meta::cint<4>}}});
  kernel::Workspace storage(
      operation.required_workspace(Any{m}, Any{n}, cint<k>, at, bt, ct));
  auto workspace = storage.view();
  operation(workspace, Any{m}, Any{n}, cint<k>, at, bt, ct);
  for (nint_t i = 0; i < m; ++i) {
    for (nint_t j = 0; j < n; ++j) {
      float32_t expected = 0;
      for (nint_t kk = 0; kk < k; ++kk)
        expected += a[i * k + kk] * b[j * k + kk];
      EXPECT_NEAR(c[i * n + j], expected, 5.0e-4f);
    }
  }
}

TEST(MatmulTest, GenericTilerOutputAccumulatorUsesLogicalOriginsOnSME) {
  using Atom = ::vecops::matmul::SME_F32F32;
  using TileAxis = std::remove_cv_t<decltype(Atom::M_R)>;
  using Tiles = ::vecops::matmul::CacheTiling<
      TileAxis, TileAxis, meta::Const<4>>;
  using Never = ::vecops::matmul::PackingPolicy<
      ::vecops::matmul::PackingMode::never>;
  using Tuning = ::vecops::matmul::GenericTiledTuning<
      Tiles, ::vecops::matmul::loop_order::MKN, Never, Never,
      ::vecops::matmul::AccBufferMode::automatic>;
  using Config = ops::MatmulConfig<
      Atom,
      ::vecops::matmul::family_selection::Require<
          ::vecops::matmul::kernel_family::GenericTiled>,
      kernel::matmul_policy::Automatic, Tuning>;
  const nint_t tile = vec::details::sme::streaming_lanes<float32_t>();
  const nint_t m = tile + 3, n = tile + 5;
  constexpr nint_t k = 9;
  std::vector<float32_t> a(m * k), b(n * k), c_input(m * n);
  std::vector<float32_t> c_output(m * n, -9.0f);
  for (nint_t i = 0; i < m * k; ++i) a[i] = value<float32_t>(i, 17);
  for (nint_t i = 0; i < n * k; ++i) b[i] = value<float32_t>(i, 23);
  for (nint_t i = 0; i < m * n; ++i) c_input[i] = value<float32_t>(i, 13);
  auto at = make_tensor(a.data(), make_layout(make_shape(Any{m}, cint<k>)));
  auto bt = make_tensor(b.data(), make_layout(make_shape(Any{n}, cint<k>)));
  auto cit = make_tensor(
      c_input.data(), make_layout(make_shape(Any{m}, Any{n})));
  auto cot = make_tensor(
      c_output.data(), make_layout(make_shape(Any{m}, Any{n})));
  auto operation = ops::matmul(Config{.generic_tiled = Tuning{
      .cache_tiling = Tiles{TileAxis{tile}, TileAxis{tile}, meta::cint<4>}}});
  kernel::Workspace storage(operation.required_workspace(
      Any{m}, Any{n}, cint<k>, at, bt, cit, cot));
  auto workspace = storage.view();
  operation(workspace, Any{m}, Any{n}, cint<k>, at, bt, cit, cot);
  for (nint_t i = 0; i < m; ++i) {
    for (nint_t j = 0; j < n; ++j) {
      float32_t expected = c_input[i * n + j];
      for (nint_t kk = 0; kk < k; ++kk)
        expected += a[i * k + kk] * b[j * k + kk];
      EXPECT_NEAR(c_output[i * n + j], expected, 5.0e-4f)
          << "m=" << i << " n=" << j;
    }
  }
}

TEST(MatmulTest, GenericTilerSplitKCommitsTransformedOutputOnceOnSME) {
  using Atom = ::vecops::matmul::SME_F32F32;
  using TileAxis = std::remove_cv_t<decltype(Atom::M_R)>;
  using Tiles = ::vecops::matmul::CacheTiling<
      TileAxis, TileAxis, meta::Const<4>>;
  using Never = ::vecops::matmul::PackingPolicy<
      ::vecops::matmul::PackingMode::never>;
  using Tuning = ::vecops::matmul::GenericTiledTuning<
      Tiles, ::vecops::matmul::loop_order::MKN, Never, Never,
      ::vecops::matmul::AccBufferMode::workspace>;
  using Config = ops::MatmulConfig<
      Atom,
      ::vecops::matmul::family_selection::Require<
          ::vecops::matmul::kernel_family::GenericTiled>,
      kernel::matmul_policy::Automatic, Tuning>;
  const nint_t tile = vec::details::sme::streaming_lanes<float32_t>();
  const nint_t m = tile + 3, n = tile + 5;
  constexpr nint_t k = 9;
  std::vector<float32_t> a(m * k), b(n * k), c_input(m * n);
  std::vector<float32_t> c_output(m * n, -9.0f);
  for (nint_t i = 0; i < m * k; ++i) a[i] = value<float32_t>(i, 17);
  for (nint_t i = 0; i < n * k; ++i) b[i] = value<float32_t>(i, 23);
  for (nint_t i = 0; i < m * n; ++i) c_input[i] = value<float32_t>(i, 13);
  auto at = make_tensor(a.data(), make_layout(make_shape(Any{m}, cint<k>)));
  auto bt = make_tensor(b.data(), make_layout(make_shape(Any{n}, cint<k>)));
  auto cit = make_tensor(
      c_input.data(), make_layout(make_shape(Any{m}, Any{n})));
  auto cot = make_tensor(
      c_output.data(), make_layout(make_shape(Any{m}, Any{n})));
  auto input_scale = make_elementwise_vec_transform<float32_t, float32_t>(
      [](auto tag, auto x) VECOPS_KERNEL_LAMBDA {
        return vec::mul(x, vec::fill(tag, 3.0f));
      });
  auto output_scale = make_elementwise_vec_transform<float32_t, float32_t>(
      [](auto tag, auto x) VECOPS_KERNEL_LAMBDA {
        return vec::mul(x, vec::fill(tag, 2.0f));
      });
  auto c_input_spec = input<float32_t>(cit, input_scale);
  auto c_output_spec = output<float32_t>(cot, output_scale);
  auto operation = ops::matmul(Config{.generic_tiled = Tuning{
      .cache_tiling = Tiles{TileAxis{tile}, TileAxis{tile}, meta::cint<4>}}});
  kernel::Workspace storage(operation.required_workspace(
      Any{m}, Any{n}, cint<k>, at, bt, c_input_spec, c_output_spec));
  auto workspace = storage.view();
  operation(
      workspace, Any{m}, Any{n}, cint<k>, at, bt,
      c_input_spec, c_output_spec);
  for (nint_t i = 0; i < m; ++i) {
    for (nint_t j = 0; j < n; ++j) {
      float32_t expected = 3.0f * c_input[i * n + j];
      for (nint_t kk = 0; kk < k; ++kk)
        expected += a[i * k + kk] * b[j * k + kk];
      EXPECT_NEAR(c_output[i * n + j], 2.0f * expected, 5.0e-4f)
          << "m=" << i << " n=" << j;
    }
  }
}

TEST(MatmulTest, GenericTilerZeroKMaterializesSemanticCInputOnSME) {
  using Atom = ::vecops::matmul::SME_F32F32;
  using TileAxis = std::remove_cv_t<decltype(Atom::M_R)>;
  using Tiles = ::vecops::matmul::CacheTiling<
      TileAxis, TileAxis, meta::Const<4>>;
  using Never = ::vecops::matmul::PackingPolicy<
      ::vecops::matmul::PackingMode::never>;
  using Tuning = ::vecops::matmul::GenericTiledTuning<
      Tiles, ::vecops::matmul::loop_order::MKN, Never, Never,
      ::vecops::matmul::AccBufferMode::automatic>;
  using Config = ops::MatmulConfig<
      Atom,
      ::vecops::matmul::family_selection::Require<
          ::vecops::matmul::kernel_family::GenericTiled>,
      kernel::matmul_policy::Automatic, Tuning>;
  const nint_t tile = vec::details::sme::streaming_lanes<float32_t>();
  const nint_t m = tile + 1, n = tile + 1;
  std::vector<float32_t> a, b, c_input(m * n), c_output(m * n, -7.0f);
  for (nint_t i = 0; i < m * n; ++i) c_input[i] = value<float32_t>(i, 13);
  auto at = make_tensor(a.data(), make_layout(make_shape(Any{m}, cint<0>)));
  auto bt = make_tensor(b.data(), make_layout(make_shape(Any{n}, cint<0>)));
  auto cit = make_tensor(
      c_input.data(), make_layout(make_shape(Any{m}, Any{n})));
  auto cot = make_tensor(
      c_output.data(), make_layout(make_shape(Any{m}, Any{n})));
  auto operation = ops::matmul(Config{.generic_tiled = Tuning{
      .cache_tiling = Tiles{TileAxis{tile}, TileAxis{tile}, meta::cint<4>}}});
  kernel::Workspace storage(operation.required_workspace(
      Any{m}, Any{n}, cint<0>, at, bt, cit, cot));
  auto workspace = storage.view();
  operation(workspace, Any{m}, Any{n}, cint<0>, at, bt, cit, cot);
  for (nint_t i = 0; i < m * n; ++i)
    EXPECT_EQ(c_output[i], c_input[i]) << "index=" << i;
}

TEST(MatmulTest, FixedExtentsReachFourRegionsAsMetaConstants) {
  check_raw<
      ::vecops::matmul::SME_F32F32, kernel::loop::tile2d_policy::FourRegions>(
          cint<19>, cint<53>, cint<17>);
}

TEST(MatmulTest, SingleAxisConstraintsReachFourRegions) {
  check_raw<
      ::vecops::matmul::SME_F32F32, kernel::loop::tile2d_policy::FourRegions>(
          Dynamic<1, 0, 16>{16}, Any{53}, cint<17>);
  check_raw<
      ::vecops::matmul::SME_F32F32, kernel::loop::tile2d_policy::FourRegions>(
          Any{53}, Dynamic<1, 0, 16>{16}, cint<17>);
}

TEST(MatmulTest, LargePackedPrefetchUsesFootprintGate) {
  using kernel::matmul_details::sme::large_packed_prefetch_v;
  static_assert(!large_packed_prefetch_v<
      ::vecops::matmul::SME_BF16F32, Const<33>, Const<35>, Const<513>>);
  static_assert(!large_packed_prefetch_v<
      ::vecops::matmul::SME_BF16F32, Const<128>, Const<1024>, Const<768>>);
#if !defined(VECOPS_DISABLE_SME_LARGE_PACKED_PREFETCH)
  static_assert(large_packed_prefetch_v<
      ::vecops::matmul::SME_BF16F32, Const<256>, Const<256>, Const<4096>>);
  static_assert(large_packed_prefetch_v<
      ::vecops::matmul::SME_BF16F32, Const<3136>, Const<64>, Const<576>>);
  static_assert(large_packed_prefetch_v<
      ::vecops::matmul::SME_BF16F32,
      Dynamic<16, 256, 512>, Dynamic<16, 256, 512>,
      Dynamic<64, 4096, 8192>>);
#endif
  static_assert(!large_packed_prefetch_v<
      ::vecops::matmul::SME_F16F32, Const<128>, Const<4096>, Const<4096>>);
}

#elif VECOPS_TARGET_SHARD_INDEX == 1

TEST(MatmulTest, FloatingInputTypes) {
  check_raw<::vecops::matmul::SME_BF16F32>(19, 21, 23);
  check_raw<::vecops::matmul::SME_F16F32>(19, 21, 23);
#if defined(HAS_SME_F64F64)
  check_raw<::vecops::matmul::SME_F64F64>(11, 13, 17);
#endif
}

TEST(MatmulTest, EndToEndMemoryComputeOutputConversions) {
  test::matmul::check_conversion<
      ::vecops::matmul::SME_BF16F32, float32_t, float32_t, float32_t>(19, 21, 23);
  test::matmul::check_conversion<
      ::vecops::matmul::SME_F32F32,
      vecops::bfloat16_t, vecops::bfloat16_t, vecops::bfloat16_t>(19, 21, 17);
  test::matmul::check_conversion<
      ::vecops::matmul::SME_F32F32,
      vecops::float16_t, vecops::float16_t, vecops::float16_t>(19, 21, 17);
#if defined(HAS_SME_F64F64)
  test::matmul::check_conversion<
      ::vecops::matmul::SME_F64F64, float32_t, float32_t, float32_t>(11, 13, 17);
#endif
}

TEST(MatmulTest, LargeConversionsUseOnTheFlyPacking) {
  test::matmul::check_conversion<
      ::vecops::matmul::SME_BF16F32, float32_t, float32_t, float32_t>(
          cint<64>, cint<64>, cint<64>, true);
  test::matmul::check_conversion<
      ::vecops::matmul::SME_F32F32,
      vecops::bfloat16_t, vecops::bfloat16_t, vecops::bfloat16_t>(
          cint<64>, cint<64>, cint<64>, true);
  test::matmul::check_conversion<
      ::vecops::matmul::SME_F32F32,
      vecops::float16_t, vecops::float16_t, vecops::float16_t>(
          cint<64>, cint<64>, cint<64>, true);
}

TEST(MatmulTest, LargeRawProblemUsesOnTheFlyPackingWorkspace) {
  using Atom = ::vecops::matmul::SME_BF16F32;
  constexpr nint_t M = 64, N = 64, K = 64;
  std::vector<typename Atom::TA> a(M * K), b(N * K);
  std::vector<typename Atom::TAcc> c(M * N);
  auto at = make_tensor(
      a.data(), make_layout(make_shape(cint<M>, cint<K>)));
  auto bt = make_tensor(
      b.data(), make_layout(make_shape(cint<N>, cint<K>)));
  auto ct = make_tensor(
      c.data(), make_layout(make_shape(cint<M>, cint<N>)));
  auto operation = test::matmul::make_test_matmul_invocation(ops::MatmulConfig<Atom>{},
      cint<M>, cint<N>, cint<K>, at, bt, ct);
  EXPECT_GT(operation.required_workspace(), 0);
  auto dynamic_operation = test::matmul::make_test_matmul_invocation(ops::MatmulConfig<Atom>{}, M, N, K, at, bt, ct);
  // Unconstrained SME extents resolve to the single default-packed path.
  EXPECT_GT(dynamic_operation.required_workspace(), 0);
  check_raw<Atom>(cint<M>, cint<N>, cint<K>);
  check_raw<Atom>(M, N, K);
}

TEST(MatmulTest, DynamicInnerStrideAlwaysUsesGenericLoad) {
  check_dynamic_inner_stride(1);
  check_dynamic_inner_stride(2);
}

#elif VECOPS_TARGET_SHARD_INDEX == 2

TEST(MatmulTest, AllInt8SignednessCombinations) {
  check_raw<::vecops::matmul::SME_I8I32<int8_t, int8_t>>(19, 21, 23);
  check_raw<::vecops::matmul::SME_I8I32<int8_t, uint8_t>>(19, 21, 23);
  check_raw<::vecops::matmul::SME_I8I32<uint8_t, int8_t>>(19, 21, 23);
  check_raw<::vecops::matmul::SME_I8I32<uint8_t, uint8_t>>(19, 21, 23);
}

TEST(MatmulTest, ExplicitKernelFamiliesSelectSmallVectorOrGeneral) {
  using Atom = ::vecops::matmul::SME_F32F32;
  check_raw<
      Atom, kernel::matmul_policy::Automatic,
      ::vecops::matmul::family_selection::Require<
          ::vecops::matmul::kernel_family::SmallVector>>(
              cint<1>, cint<16>, cint<17>);
  check_raw<
      Atom, kernel::matmul_policy::Automatic,
      ::vecops::matmul::family_selection::Require<
          ::vecops::matmul::kernel_family::SmallVector>>(
              Any{1}, Any{16}, Any{17});
  check_raw<
      Atom, kernel::matmul_policy::Automatic,
      ::vecops::matmul::family_selection::Require<
          ::vecops::matmul::kernel_family::General>>(
              cint<1>, cint<16>, cint<17>);
  check_raw<
      Atom, kernel::matmul_policy::Automatic,
      ::vecops::matmul::family_selection::Prefer<
          ::vecops::matmul::kernel_family::SmallVector>>(
      cint<19>, cint<21>, cint<17>);

  using RequireSmall = ::vecops::matmul::family_selection::Require<
      ::vecops::matmul::kernel_family::SmallVector>;
  check_raw<::vecops::matmul::SME_BF16F32,
            kernel::matmul_policy::Automatic, RequireSmall>(
      Any{1}, Any{8}, Any{65});
  check_raw<::vecops::matmul::SME_F16F32,
            kernel::matmul_policy::Automatic, RequireSmall>(
      cint<16>, cint<1>, cint<65>);
  check_raw<::vecops::matmul::SME_I8I32<int8_t, int8_t>,
            kernel::matmul_policy::Automatic, RequireSmall>(
      Any{1}, Any{64}, Any{257});
  check_raw<::vecops::matmul::SME_I8I32<uint8_t, uint8_t>,
            kernel::matmul_policy::Automatic, RequireSmall>(
      cint<64>, cint<1>, cint<257>);
  check_raw<::vecops::matmul::SME_I8I32<int8_t, uint8_t>,
            kernel::matmul_policy::Automatic, RequireSmall>(
      Any{1}, Any{64}, Any{257});
  check_raw<::vecops::matmul::SME_I8I32<uint8_t, int8_t>,
            kernel::matmul_policy::Automatic, RequireSmall>(
      cint<64>, cint<1>, cint<257>);
#if defined(HAS_SME_F64F64)
  check_raw<::vecops::matmul::SME_F64F64,
            kernel::matmul_policy::Automatic, RequireSmall>(
      Any{1}, Any{64}, Any{257});
#endif
}

TEST(MatmulTest, PackedDotCoversPrimaryTinyStaticAndDynamicShapes) {
  check_required_packed_dot<::vecops::matmul::SME_BF16F32>(
      Any{2}, Any{8}, Any{1025});
  check_required_packed_dot<::vecops::matmul::SME_BF16F32>(
      cint<4>, cint<2>, cint<257>);
  check_required_packed_dot<
      ::vecops::matmul::SME_I8I32<int8_t, int8_t>>(
          Any{2}, Any{4}, Any{513});
  check_required_packed_dot<
      ::vecops::matmul::SME_I8I32<uint8_t, uint8_t>>(
          cint<4>, cint<4>, cint<257>);
}

TEST(MatmulTest, GenericTilerSplitKReusesPackedDotPhases) {
  using Atom = ::vecops::matmul::SME_BF16F32;
  using T = typename Atom::TA;
  using Tiles = ::vecops::matmul::CacheTiling<
      meta::Const<2>, meta::Const<8>, meta::Const<64>>;
  using Always = ::vecops::matmul::PackingPolicy<
      ::vecops::matmul::PackingMode::always,
      ::vecops::matmul::PackingExtent::cache_k>;
  using Tuning = ::vecops::matmul::GenericTiledTuning<
      Tiles, ::vecops::matmul::loop_order::MNK, Always, Always,
      ::vecops::matmul::AccBufferMode::workspace>;
  using Config = ops::MatmulConfig<
      Atom,
      ::vecops::matmul::family_selection::Require<
          ::vecops::matmul::kernel_family::GenericTiled>,
      kernel::matmul_policy::Automatic, Tuning>;
  constexpr nint_t M = 2, N = 8, K = 129;
  std::vector<T> a(M * K), b(N * K);
  std::vector<float32_t> c(M * N);
  for (nint_t i = 0; i < M * K; ++i) a[i] = value<T>(i, 13);
  for (nint_t i = 0; i < N * K; ++i) b[i] = value<T>(i, 11);
  auto at = make_tensor(a.data(), make_layout(make_shape(cint<M>, cint<K>)));
  auto bt = make_tensor(b.data(), make_layout(make_shape(cint<N>, cint<K>)));
  auto ct = make_tensor(c.data(), make_layout(make_shape(cint<M>, cint<N>)));
  Config config{.generic_tiled = Tuning{
      .cache_tiling = Tiles{cint<M>, cint<N>, cint<64>}}};
  auto operation = ops::matmul(config);
  kernel::Workspace storage(operation.required_workspace(
      cint<M>, cint<N>, cint<K>, at, bt, ct));
  auto workspace = storage.view();
  operation(workspace, cint<M>, cint<N>, cint<K>, at, bt, ct);
  for (nint_t row = 0; row < M; ++row) {
    for (nint_t column = 0; column < N; ++column) {
      float32_t expected = 0;
      for (nint_t kk = 0; kk < K; ++kk)
        expected += static_cast<float32_t>(a[row * K + kk]) *
            static_cast<float32_t>(b[column * K + kk]);
      EXPECT_NEAR(c[row * N + column], expected, 5.0e-4f);
    }
  }
}

TEST(MatmulTest, EitherOperandMayBePrepacked) {
  check_mixed_packing<::vecops::matmul::Operand::A>();
  check_mixed_packing<::vecops::matmul::Operand::B>();
}

TEST(MatmulTest, BothPackedLargeKUsesFastPath) {
  check_fast_packed_path(Dynamic<2, 64, 128>{74});
}

TEST(MatmulTest, BothPackedBF16K32RemainsCorrect) {
  check_fast_packed_path(cint<32>);
}

TEST(MatmulTest, PackedKBoundsEliminateFastPathRuntimeChoice) {
  check_fast_packed_path(Dynamic<2, 2, 30>{30});
}

TEST(MatmulTest, UnboundedPackedKUsesFastPlan) {
  check_fast_packed_path(Any{30});
  check_fast_packed_path(Any{74});
}

#elif VECOPS_TARGET_SHARD_INDEX == 3

TEST(MatmulTest, CPrologueAndEpilogueUseDataAccess) {
  constexpr nint_t M = 19, N = 23, K = 17;
  std::vector<float32_t> a(M * K), b(N * K), c(M * N, 1.25f);
  for (nint_t i = 0; i < M * K; ++i) a[i] = value<float32_t>(i, 13);
  for (nint_t i = 0; i < N * K; ++i) b[i] = value<float32_t>(i, 11);
  auto at = make_tensor(a.data(), make_layout(make_shape(Any{M}, Any{K})));
  auto bt = make_tensor(b.data(), make_layout(make_shape(Any{N}, Any{K})));
  auto ct = make_tensor(c.data(), make_layout(make_shape(Any{M}, Any{N})));
  auto epilogue = make_elementwise_vec_transform<float32_t, float32_t>(
      [](auto tag, auto x) VECOPS_KERNEL_LAMBDA {
        return vec::mul(x, vec::fill(tag, 2.0f));
      });
  auto operation = test::matmul::make_test_matmul_invocation(
      ops::MatmulConfig<::vecops::matmul::SME_F32F32>{},
      M, N, K, at, bt, input<float32_t>(ct), output<float32_t>(ct, epilogue));
  kernel::Workspace operation_owner(operation.required_workspace());
  auto operation_workspace = operation_owner.view();
  ExecutionSession execution{operation_workspace};
  operation(execution);
  for (nint_t i = 0; i < M; ++i) {
    for (nint_t j = 0; j < N; ++j) {
      float expected = 1.25f;
      for (nint_t kk = 0; kk < K; ++kk)
        expected += a[i * K + kk] * b[j * K + kk];
      EXPECT_NEAR(c[i * N + j], 2.0f * expected, 4.0e-4f);
    }
  }
}

TEST(MatmulTest, BroadcastBiasAndReluAreFusedThroughDataAccess) {
  test::matmul::check_bias_relu<::vecops::matmul::SME_BF16F32>(19, 23, 17);
  test::matmul::check_bias_relu<::vecops::matmul::SME_F32F32>(19, 23, 9);
}

TEST(MatmulTest, AsymmetricU8S8UsesColumnSumCorrection) {
  test::matmul::check_asymmetric_quantized<
      ::vecops::matmul::SME_I8I32<uint8_t, int8_t>>(19, 23, 33);
}

TEST(MatmulTest, LargeQuantizedTransformsUseOnTheFlyPacking) {
  test::matmul::check_asymmetric_quantized<
      ::vecops::matmul::SME_I8I32<uint8_t, int8_t>>(
          cint<16>, cint<128>, cint<128>, true);
}

#elif VECOPS_TARGET_SHARD_INDEX == 4

TEST(MatmulTest, DynamicAndConstExtentPairs) {
  test::matmul::check_native_extent_pair<::vecops::matmul::SME_F32F32, 16, 16, 19>();
  test::matmul::check_native_extent_pair<::vecops::matmul::SME_F32F32, 19, 53, 17>();
  test::matmul::check_native_extent_pair<::vecops::matmul::SME_BF16F32, 19, 21, 23>();
  test::matmul::check_native_extent_pair<::vecops::matmul::SME_F16F32, 19, 21, 23>();
#if defined(HAS_SME_F64F64)
  test::matmul::check_native_extent_pair<::vecops::matmul::SME_F64F64, 11, 13, 17>();
#endif
  test::matmul::check_native_extent_pair<
      ::vecops::matmul::SME_I8I32<int8_t, int8_t>, 19, 21, 23>();
  test::matmul::check_native_extent_pair<
      ::vecops::matmul::SME_I8I32<uint8_t, int8_t>, 16, 128, 128>();
}

#endif

} // namespace

#endif // VECOPS_TARGET_SHARD_ACTIVE

#endif
