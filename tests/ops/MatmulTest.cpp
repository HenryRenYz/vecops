// @vecops-target-shards-x86: 5
// @vecops-target-shards-ARM: 5

#include "vecops/Features.h"
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

template <typename Atom, typename M, typename N, typename K>
void check_raw(
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
  auto operation = ops::make_matmul<Atom>(
      m_extent, n_extent, k_extent, at, bt, ct);
  using Operation = decltype(operation);
  static_assert(std::same_as<
      typename Operation::MExtentType, meta::to_value_t<M>>);
  static_assert(std::same_as<
      typename Operation::NExtentType, meta::to_value_t<N>>);
  static_assert(std::same_as<
      typename Operation::KExtentType, meta::to_value_t<K>>);
  if (expect_auto_packing) {
    const nint_t scratch = kernel::matmul_implementation::scratch_bytes<
        kernel::matmul_implementation::AMX>();
    EXPECT_GT(operation.required_workspace(), scratch);
  } else if (expect_no_auto_packing) {
    const nint_t scratch = kernel::matmul_implementation::scratch_bytes<
        kernel::matmul_implementation::AMX>();
    EXPECT_EQ(operation.required_workspace(), scratch);
  }
  kernel::Workspace storage(operation.required_workspace());
  auto workspace = storage.view();
  operation(workspace);
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

VECOPS_NOINLINE void check_dynamic_inner_stride(nint_t inner_stride) {
  using Atom = gemm::AMX_BF16F32;
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
  auto operation = ops::make_matmul<Atom>(
      cint<M>, cint<N>, cint<K>, at, bt, ct);
  EXPECT_EQ(
      operation.required_workspace(),
      kernel::matmul_implementation::scratch_bytes<
          kernel::matmul_implementation::AMX>());
  kernel::Workspace owner(operation.required_workspace());
  auto workspace = owner.view();
  operation(workspace);
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

template <gemm::Operand Side>
void check_mixed_packing(
    nint_t m = 37, nint_t n = 51, nint_t k = 79) {
  using Atom = gemm::AMX_BF16F32;
  std::vector<bfloat16_t> a(m * k), b(n * k);
  std::vector<float32_t> c(m * n);
  for (nint_t i = 0; i < m * k; ++i) a[i] = value<bfloat16_t>(i, 13);
  for (nint_t i = 0; i < n * k; ++i) b[i] = value<bfloat16_t>(i, 11);
  auto al = make_layout(make_shape(Any{m}, Any{k}));
  auto bl = make_layout(make_shape(Any{n}, Any{k}));
  auto at = make_tensor(a.data(), al);
  auto bt = make_tensor(b.data(), bl);
  const auto& source_layout = Side == gemm::Operand::A ? al : bl;
  auto packed_layout = ops::matmul_packed_layout<Atom, Side>(source_layout);
  kernel::Workspace packed_owner(
      numel(packed_layout) * static_cast<nint_t>(sizeof(bfloat16_t)) + 64);
  auto packed_workspace = packed_owner.view();
  auto* packed = static_cast<bfloat16_t*>(packed_workspace.allocate(
      numel(packed_layout) * static_cast<nint_t>(sizeof(bfloat16_t)), 64));
  auto packed_tensor = make_tensor(packed, packed_layout);
  ExecutionSession pack_execution{};
  if constexpr (Side == gemm::Operand::A) {
    ops::matmul_pack<Atom, Side>(pack_execution, at, packed_tensor);
  } else {
    ops::matmul_pack<Atom, Side>(pack_execution, bt, packed_tensor);
  }
  auto ct = make_tensor(
      c.data(), make_layout(make_shape(Any{m}, Any{n})));
  auto operation = [&] {
    if constexpr (Side == gemm::Operand::A)
      return ops::make_matmul<Atom>(m, n, k, packed_tensor, bt, ct);
    else
      return ops::make_matmul<Atom>(m, n, k, at, packed_tensor, ct);
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
            << (Side == gemm::Operand::A ? "A" : "B")
            << " m=" << i << " n=" << j
            << " actual=" << c[i * n + j]
            << " expected=" << expected;
        return;
      }
    }
  }
}

template <meta::ValueType KExtent>
void check_both_packed(
    KExtent k_extent, nint_t m = 37, nint_t n = 16) {
  using Atom = gemm::AMX_BF16F32;
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
  auto apl = ops::matmul_packed_layout<Atom, gemm::Operand::A>(al);
  auto bpl = ops::matmul_packed_layout<Atom, gemm::Operand::B>(bl);
  static_assert(gemm::is_packed_layout<
                Atom, gemm::Operand::A, decltype(apl)>());
  static_assert(gemm::is_packed_layout<
                Atom, gemm::Operand::B, decltype(bpl)>());
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
  ops::matmul_pack<Atom, gemm::Operand::A>(pack_execution, at, apt);
  ops::matmul_pack<Atom, gemm::Operand::B>(pack_execution, bt, bpt);
  auto ct = make_tensor(
      c.data(), make_layout(make_shape(Any{m}, Any{n})));
  auto operation = ops::make_matmul<Atom>(
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
  using Atom = gemm::AMX_BF16F32;
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
  auto operation = ops::make_matmul<Atom>(
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
  check_raw<gemm::AMX_BF16F32>(16, 16, 35);
  check_raw<gemm::AMX_BF16F32>(16, 32, 67);
  check_raw<gemm::AMX_BF16F32>(16, 48, 79);
  check_raw<gemm::AMX_BF16F32>(1, 48, 79);
  check_raw<gemm::AMX_BF16F32>(32, 16, 33);
  check_raw<gemm::AMX_BF16F32>(48, 16, 65);
  check_raw<gemm::AMX_BF16F32>(32, 32, 97);
  check_raw<gemm::AMX_BF16F32>(16, 64, 67);
  check_raw<gemm::AMX_BF16F32>(16, 80, 67);
  check_raw<gemm::AMX_BF16F32>(16, 96, 67);
  check_raw<gemm::AMX_BF16F32>(64, 16, 67);
  check_raw<gemm::AMX_BF16F32>(80, 16, 67);
  check_raw<gemm::AMX_BF16F32>(96, 16, 67);
  check_raw<gemm::AMX_BF16F32>(37, 51, 79);
  check_raw<gemm::AMX_BF16F32>(1, 128, 65);
  check_raw<gemm::AMX_BF16F32>(512, 1, 256);
}

TEST(MatmulTest, CompileTimeExtentsAndFullKPlan) {
  ASSERT_TRUE(vecops::test::matmul::MatmulTestArchTraits::enable());
  check_raw<gemm::AMX_BF16F32>(cint<0>, cint<16>, cint<32>);
  check_raw<gemm::AMX_BF16F32>(cint<16>, cint<0>, cint<32>);
  check_raw<gemm::AMX_BF16F32>(cint<16>, cint<32>, cint<67>);
  check_raw<gemm::AMX_BF16F32>(cint<32>, cint<16>, cint<67>);
  check_raw<gemm::AMX_BF16F32>(cint<16>, cint<48>, cint<79>);
  check_raw<gemm::AMX_BF16F32>(cint<48>, cint<16>, cint<79>);
  check_raw<gemm::AMX_BF16F32>(cint<32>, cint<32>, cint<128>);
  check_raw<gemm::AMX_BF16F32>(cint<35>, cint<53>, cint<257>);
  check_raw<gemm::AMX_I8I32<int8_t, uint8_t>>(
      cint<35>, cint<53>, cint<257>);
  check_raw<gemm::AMX_I8I32<uint8_t, uint8_t>>(
      cint<35>, cint<53>, cint<257>);
  check_raw<gemm::AMX_BF16F32>(
      dyn<32, 32, 64>(64), dyn<32, 32, 64>(32),
      dyn<32, 64, 128>(96));
  check_raw<gemm::AMX_BF16F32>(
      dyn<32, 32, 64>(32), dyn<16, 32, 64>(48), dyn<32, 64, 128>(96));
  // Bounded runtime extents may select a special owner only when the full
  // admissible Meta range satisfies that owner's crossover.
  check_raw<gemm::AMX_BF16F32>(
      dyn<1, 1, 1>(1), dyn<1, 1, 16>(8), dyn<32, 32, 256>(64));
}

TEST(MatmulTest, LargeNativeInputsUseAutoPacking) {
  ASSERT_TRUE(vecops::test::matmul::MatmulTestArchTraits::enable());
  check_raw<gemm::AMX_BF16F32>(cint<64>, cint<64>, cint<64>, true);
  check_raw<gemm::AMX_I8I32<int8_t, uint8_t>>(
      cint<64>, cint<64>, cint<128>, true);
}

TEST(MatmulTest, DynamicExtentsDoNotSelectAutoPacking) {
  ASSERT_TRUE(vecops::test::matmul::MatmulTestArchTraits::enable());
  check_raw<gemm::AMX_BF16F32>(64, 64, 64, false, true);
  check_raw<gemm::AMX_I8I32<int8_t, uint8_t>>(
      64, 64, 128, false, true);
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
}

#elif VECOPS_TARGET_SHARD_INDEX == 1

#if defined(HAS_AMX_FP16)
TEST(MatmulTest, NativeFP16UsesFullAndTailKPlans) {
  ASSERT_TRUE(vecops::test::matmul::MatmulTestArchTraits::enable());
  check_raw<gemm::AMX_F16F32>(19, 21, 32);
  check_raw<gemm::AMX_F16F32>(19, 21, 35);
}
#endif

TEST(MatmulTest, EndToEndMemoryComputeOutputConversions) {
  ASSERT_TRUE(vecops::test::matmul::MatmulTestArchTraits::enable());
  test::matmul::check_conversion<
      gemm::AMX_BF16F32, float32_t, float32_t, float32_t>(19, 21, 35);
  test::matmul::check_conversion<
      gemm::AMX_BF16F32, bfloat16_t, bfloat16_t, bfloat16_t>(19, 21, 35);
  test::matmul::check_conversion<
      gemm::AMX_BF16F32, bfloat16_t, bfloat16_t, bfloat16_t>(2, 2, 35);
#if defined(HAS_AMX_FP16)
  test::matmul::check_conversion<
      gemm::AMX_F16F32, float32_t, float32_t, float32_t>(19, 21, 35);
  test::matmul::check_conversion<
      gemm::AMX_F16F32, float16_t, float16_t, float16_t>(19, 21, 35);
#endif
}

TEST(MatmulTest, LargeConversionInputsUseAutoPacking) {
  ASSERT_TRUE(vecops::test::matmul::MatmulTestArchTraits::enable());
  test::matmul::check_conversion<
      gemm::AMX_BF16F32, float32_t, float32_t, float32_t>(
          cint<64>, cint<64>, cint<64>, true);
}

#elif VECOPS_TARGET_SHARD_INDEX == 2

TEST(MatmulTest, AllInt8SignednessCombinations) {
  ASSERT_TRUE(vecops::test::matmul::MatmulTestArchTraits::enable());
  check_raw<gemm::AMX_I8I32<int8_t, int8_t>>(19, 21, 69);
  check_raw<gemm::AMX_I8I32<int8_t, uint8_t>>(19, 21, 69);
  check_raw<gemm::AMX_I8I32<uint8_t, int8_t>>(19, 21, 69);
  check_raw<gemm::AMX_I8I32<uint8_t, uint8_t>>(19, 21, 69);
  // Large raw GEMV/GEMM crossover: exercise both orientations and a one-byte
  // K tail for every signedness combination.
  check_raw<gemm::AMX_I8I32<int8_t, int8_t>>(1, 128, 65);
  check_raw<gemm::AMX_I8I32<int8_t, uint8_t>>(128, 1, 65);
  check_raw<gemm::AMX_I8I32<uint8_t, int8_t>>(1, 128, 65);
  check_raw<gemm::AMX_I8I32<uint8_t, uint8_t>>(128, 1, 65);
  // The two mixed-signedness cases have a native VPDPBUSD mapping and use the
  // AVX-512 small-matrix crossover, including a non-multiple-of-64 K tail.
  check_raw<gemm::AMX_I8I32<int8_t, uint8_t>>(1, 16, 129);
  check_raw<gemm::AMX_I8I32<uint8_t, int8_t>>(16, 1, 129);
  check_raw<gemm::AMX_I8I32<int8_t, int8_t>>(1, 16, 129);
  check_raw<gemm::AMX_I8I32<uint8_t, uint8_t>>(16, 1, 129);
}

TEST(MatmulTest, SameSignSmallVectorPreservesWrapAndTail) {
  ASSERT_TRUE(vecops::test::matmul::MatmulTestArchTraits::enable());
  constexpr nint_t K = 262145;
  auto check = [=]<typename T>() {
    using Atom = gemm::AMX_I8I32<T, T>;
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
    auto operation = ops::make_matmul<Atom>(1, 1, K, at, bt, ct);
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

TEST(MatmulTest, OperandAMayBePrepacked) {
  ASSERT_TRUE(vecops::test::matmul::MatmulTestArchTraits::enable());
  check_mixed_packing<gemm::Operand::A>();
  check_mixed_packing<gemm::Operand::A>(1, 16, 65);
}

TEST(MatmulTest, OperandBMayBePrepacked) {
  ASSERT_TRUE(vecops::test::matmul::MatmulTestArchTraits::enable());
  check_mixed_packing<gemm::Operand::B>();
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
  auto operation = ops::make_matmul_accumulate<gemm::AMX_BF16F32>(
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
  auto operation = ops::make_matmul_accumulate<gemm::AMX_BF16F32>(
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
  test::matmul::check_bias_relu<gemm::AMX_BF16F32>(19, 23, 65);
  test::matmul::check_bias_relu<gemm::AMX_BF16F32>(1, 16, 65);
}

TEST(MatmulTest, AsymmetricU8S8UsesColumnSumCorrection) {
  ASSERT_TRUE(vecops::test::matmul::MatmulTestArchTraits::enable());
#if defined(HAS_AMX_INT8)
  test::matmul::check_asymmetric_quantized<
      gemm::AMX_I8I32<uint8_t, int8_t>>(19, 23, 65);
  test::matmul::check_asymmetric_quantized<
      gemm::AMX_I8I32<uint8_t, int8_t>>(
          cint<64>, cint<128>, cint<128>, true);
#endif
}

TEST(MatmulTest, CompensatedPackedBFeedsAsymmetricCPrologue) {
  ASSERT_TRUE(vecops::test::matmul::MatmulTestArchTraits::enable());
#if defined(HAS_AMX_INT8)
  using Atom = gemm::AMX_I8I32<uint8_t, int8_t>;
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
  auto packed_layout = ops::matmul_packed_layout<
      Atom, gemm::Operand::B>(b_layout);
  kernel::Workspace packed_owner(
      numel(packed_layout) * static_cast<nint_t>(sizeof(int8_t)) + 64);
  auto packed_workspace = packed_owner.view();
  auto* packed = static_cast<int8_t*>(packed_workspace.allocate(
      numel(packed_layout) * static_cast<nint_t>(sizeof(int8_t)), 64));
  auto packed_tensor = make_tensor(packed, packed_layout);
  auto compensation_vector = make_tensor(
      compensation.data(), make_layout(make_shape(cint<N>)));
  ExecutionSession pack_execution{};
  ops::matmul_pack_b_compensated<Atom>(
      pack_execution, b_tensor, packed_tensor,
      compensation_vector, ZeroPointA);

  auto correction_tensor = make_tensor(
      compensation.data(),
      make_layout(
          make_shape(cint<M>, cint<N>),
          make_strides(cint<0>, cint<1>)));
  auto c_tensor = make_tensor(
      c.data(), make_layout(make_shape(cint<M>, cint<N>)));
  auto operation = ops::make_matmul_accumulate<Atom>(
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
  test::matmul::check_native_extent_pair<gemm::AMX_BF16F32, 16, 16, 35>();
  test::matmul::check_native_extent_pair<gemm::AMX_BF16F32, 16, 32, 67>();
  test::matmul::check_native_extent_pair<gemm::AMX_BF16F32, 37, 51, 79>();
  test::matmul::check_native_extent_pair<gemm::AMX_BF16F32, 1, 128, 65>();
  test::matmul::check_native_extent_pair<gemm::AMX_BF16F32, 64, 64, 64>();
  test::matmul::check_native_extent_pair<gemm::AMX_BF16F32, 35, 53, 257>();
#if defined(HAS_AMX_FP16)
  test::matmul::check_native_extent_pair<gemm::AMX_F16F32, 19, 21, 65>();
#endif
#if defined(HAS_AMX_INT8)
  test::matmul::check_native_extent_pair<
      gemm::AMX_I8I32<int8_t, uint8_t>, 35, 53, 257>();
  test::matmul::check_native_extent_pair<
      gemm::AMX_I8I32<uint8_t, int8_t>, 64, 64, 128>();
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

using SMEF32Tile2x2M = decltype(cint<2> * gemm::SME_F32F32::M_R);

#if defined(HAS_FIXED_STREAMING_SVE_BITS)
static_assert(std::same_as<
    std::remove_cv_t<decltype(
        vec::details::sme::streaming_vector_bytes_value())>,
    Const<FIXED_STREAMING_SVE_BITS / 8>>);
static_assert(std::same_as<
    std::remove_cv_t<decltype(gemm::SME_F32F32::M_R)>,
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
    std::remove_cv_t<decltype(gemm::SME_F32F32::M_R)>,
    Dynamic<4, 4, 64>>);
static_assert(std::same_as<
        std::remove_cv_t<SMEF32Tile2x2M>,
    Dynamic<8, 8, 128>>);
#endif
static_assert(std::same_as<
    std::remove_cv_t<decltype(gemm::SME_F32F32::K_R)>, Const<1>>);
static_assert(kernel::matmul_details::sme::has_bounded_tile_axis_v<
              gemm::SME_F32F32, Const<16>, Any>);
static_assert(kernel::matmul_details::sme::has_bounded_tile_axis_v<
              gemm::SME_F32F32, Dynamic<1, 0, 32>, Any>);
static_assert(!kernel::matmul_details::sme::has_bounded_tile_axis_v<
              gemm::SME_F32F32, Any, Any>);

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
  auto operation = ops::make_matmul<Atom, TilePolicy>(
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
  using Atom = gemm::SME_BF16F32;
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
  auto operation = ops::make_matmul<Atom>(m, n, k, at, bt, ct);
  ExecutionSession execution{};
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

template <gemm::Operand Side>
void check_mixed_packing() {
  using Atom = gemm::SME_F32F32;
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
  const auto& source_layout = Side == gemm::Operand::A ? al : bl;
  auto packed_layout = ops::matmul_packed_layout<Atom, Side>(source_layout);
  kernel::Workspace packed_owner(
      numel(packed_layout) * static_cast<nint_t>(sizeof(float32_t)) + 64);
  auto packed_workspace = packed_owner.view();
  auto* packed = static_cast<float32_t*>(packed_workspace.allocate(
      numel(packed_layout) * static_cast<nint_t>(sizeof(float32_t)), 64));
  auto packed_tensor = make_tensor(packed, packed_layout);
  ExecutionSession execution{};
  if constexpr (Side == gemm::Operand::A)
    ops::matmul_pack<Atom, Side>(execution, at, packed_tensor);
  else
    ops::matmul_pack<Atom, Side>(execution, bt, packed_tensor);
  auto ct = make_tensor(c.data(), make_layout(make_shape(Any{m}, Any{n})));
  auto operation = [&] {
    if constexpr (Side == gemm::Operand::A)
      return ops::make_matmul<Atom>(m, n, k, packed_tensor, bt, ct);
    else
      return ops::make_matmul<Atom>(m, n, k, at, packed_tensor, ct);
  }();
  operation(execution);
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
  using Atom = gemm::SME_BF16F32;
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
  auto apl = ops::matmul_packed_layout<Atom, gemm::Operand::A>(al);
  auto bpl = ops::matmul_packed_layout<Atom, gemm::Operand::B>(bl);
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
  ops::matmul_pack<Atom, gemm::Operand::A>(execution, at, apt);
  ops::matmul_pack<Atom, gemm::Operand::B>(execution, bt, bpt);
  auto ct = make_tensor(c.data(), make_layout(make_shape(Any{m}, Any{n})));
  auto operation = ops::make_matmul<Atom>(m, n, k_value, apt, bpt, ct);
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

#if VECOPS_TARGET_SHARD_INDEX == 0

TEST(MatmulTest, AllZA32MicrokernelShapesAndKTails) {
  const nint_t l = vec::details::sme::streaming_lanes<float32_t>();
  check_raw<gemm::SME_F32F32>(l, l, 19);
  check_raw<gemm::SME_F32F32>(l, 2 * l, 19);
  check_raw<gemm::SME_F32F32>(l, 3 * l, 19);
  check_raw<gemm::SME_F32F32>(l, 4 * l, 19);
  check_raw<gemm::SME_F32F32>(2 * l, l, 19);
  check_raw<gemm::SME_F32F32>(3 * l, l, 19);
  check_raw<gemm::SME_F32F32>(4 * l, l, 19);
  check_raw<gemm::SME_F32F32>(2 * l, 2 * l, 19);
  check_raw<gemm::SME_F32F32>(2 * l + 3, 3 * l + 5, 19);
}

TEST(MatmulTest, FixedExtentsReachFourRegionsAsMetaConstants) {
  check_raw<
      gemm::SME_F32F32, kernel::loop::tile2d_policy::FourRegions>(
          cint<19>, cint<53>, cint<17>);
}

TEST(MatmulTest, SingleAxisConstraintsReachFourRegions) {
  check_raw<
      gemm::SME_F32F32, kernel::loop::tile2d_policy::FourRegions>(
          Dynamic<1, 0, 16>{16}, Any{53}, cint<17>);
  check_raw<
      gemm::SME_F32F32, kernel::loop::tile2d_policy::FourRegions>(
          Any{53}, Dynamic<1, 0, 16>{16}, cint<17>);
}

TEST(MatmulTest, LargePackedPrefetchUsesFootprintGate) {
  using kernel::matmul_details::sme::large_packed_prefetch_v;
  static_assert(!large_packed_prefetch_v<
      gemm::SME_BF16F32, Const<33>, Const<35>, Const<513>>);
  static_assert(!large_packed_prefetch_v<
      gemm::SME_BF16F32, Const<128>, Const<1024>, Const<768>>);
#if !defined(VECOPS_DISABLE_SME_LARGE_PACKED_PREFETCH)
  static_assert(large_packed_prefetch_v<
      gemm::SME_BF16F32, Const<256>, Const<256>, Const<4096>>);
  static_assert(large_packed_prefetch_v<
      gemm::SME_BF16F32, Const<3136>, Const<64>, Const<576>>);
  static_assert(large_packed_prefetch_v<
      gemm::SME_BF16F32,
      Dynamic<16, 256, 512>, Dynamic<16, 256, 512>,
      Dynamic<64, 4096, 8192>>);
#endif
  static_assert(!large_packed_prefetch_v<
      gemm::SME_F16F32, Const<128>, Const<4096>, Const<4096>>);
}

#elif VECOPS_TARGET_SHARD_INDEX == 1

TEST(MatmulTest, FloatingInputTypes) {
  check_raw<gemm::SME_BF16F32>(19, 21, 23);
  check_raw<gemm::SME_F16F32>(19, 21, 23);
#if defined(HAS_SME_F64F64)
  check_raw<gemm::SME_F64F64>(11, 13, 17);
#endif
}

TEST(MatmulTest, EndToEndMemoryComputeOutputConversions) {
  test::matmul::check_conversion<
      gemm::SME_BF16F32, float32_t, float32_t, float32_t>(19, 21, 23);
  test::matmul::check_conversion<
      gemm::SME_F32F32,
      vecops::bfloat16_t, vecops::bfloat16_t, vecops::bfloat16_t>(19, 21, 17);
  test::matmul::check_conversion<
      gemm::SME_F32F32,
      vecops::float16_t, vecops::float16_t, vecops::float16_t>(19, 21, 17);
#if defined(HAS_SME_F64F64)
  test::matmul::check_conversion<
      gemm::SME_F64F64, float32_t, float32_t, float32_t>(11, 13, 17);
#endif
}

TEST(MatmulTest, LargeConversionsUseOnTheFlyPacking) {
  test::matmul::check_conversion<
      gemm::SME_BF16F32, float32_t, float32_t, float32_t>(
          cint<32>, cint<32>, cint<64>, true);
  test::matmul::check_conversion<
      gemm::SME_F32F32,
      vecops::bfloat16_t, vecops::bfloat16_t, vecops::bfloat16_t>(
          cint<32>, cint<32>, cint<64>, true);
  test::matmul::check_conversion<
      gemm::SME_F32F32,
      vecops::float16_t, vecops::float16_t, vecops::float16_t>(
          cint<32>, cint<32>, cint<64>, true);
}

TEST(MatmulTest, LargeRawProblemUsesOnTheFlyPackingWorkspace) {
  using Atom = gemm::SME_BF16F32;
  constexpr nint_t M = 32, N = 32, K = 64;
  std::vector<typename Atom::TA> a(M * K), b(N * K);
  std::vector<typename Atom::TAcc> c(M * N);
  auto at = make_tensor(
      a.data(), make_layout(make_shape(cint<M>, cint<K>)));
  auto bt = make_tensor(
      b.data(), make_layout(make_shape(cint<N>, cint<K>)));
  auto ct = make_tensor(
      c.data(), make_layout(make_shape(cint<M>, cint<N>)));
  auto operation = ops::make_matmul<Atom>(
      cint<M>, cint<N>, cint<K>, at, bt, ct);
  EXPECT_GT(operation.required_workspace(), 0);
  auto dynamic_operation = ops::make_matmul<Atom>(M, N, K, at, bt, ct);
  EXPECT_EQ(dynamic_operation.required_workspace(), 0);
  check_raw<Atom>(cint<M>, cint<N>, cint<K>);
  check_raw<Atom>(M, N, K);
}

TEST(MatmulTest, DynamicInnerStrideAlwaysUsesGenericLoad) {
  check_dynamic_inner_stride(1);
  check_dynamic_inner_stride(2);
}

#elif VECOPS_TARGET_SHARD_INDEX == 2

TEST(MatmulTest, AllInt8SignednessCombinations) {
  check_raw<gemm::SME_I8I32<int8_t, int8_t>>(19, 21, 23);
  check_raw<gemm::SME_I8I32<int8_t, uint8_t>>(19, 21, 23);
  check_raw<gemm::SME_I8I32<uint8_t, int8_t>>(19, 21, 23);
  check_raw<gemm::SME_I8I32<uint8_t, uint8_t>>(19, 21, 23);
}

TEST(MatmulTest, EitherOperandMayBePrepacked) {
  check_mixed_packing<gemm::Operand::A>();
  check_mixed_packing<gemm::Operand::B>();
}

TEST(MatmulTest, BothPackedLargeKUsesFastPath) {
  check_fast_packed_path(Dynamic<2, 64, 128>{74});
}

TEST(MatmulTest, PackedKBoundsEliminateFastPathRuntimeChoice) {
  check_fast_packed_path(Dynamic<2, 2, 62>{30});
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
  auto operation = ops::make_matmul_accumulate<gemm::SME_F32F32>(
      M, N, K, at, bt, input<float32_t>(ct), output<float32_t>(ct, epilogue));
  ExecutionSession execution{};
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
  test::matmul::check_bias_relu<gemm::SME_BF16F32>(19, 23, 17);
  test::matmul::check_bias_relu<gemm::SME_F32F32>(19, 23, 9);
}

TEST(MatmulTest, AsymmetricU8S8UsesColumnSumCorrection) {
  test::matmul::check_asymmetric_quantized<
      gemm::SME_I8I32<uint8_t, int8_t>>(19, 23, 33);
}

TEST(MatmulTest, LargeQuantizedTransformsUseOnTheFlyPacking) {
  test::matmul::check_asymmetric_quantized<
      gemm::SME_I8I32<uint8_t, int8_t>>(
          cint<16>, cint<128>, cint<128>, true);
}

#elif VECOPS_TARGET_SHARD_INDEX == 4

TEST(MatmulTest, DynamicAndConstExtentPairs) {
  test::matmul::check_native_extent_pair<gemm::SME_F32F32, 16, 16, 19>();
  test::matmul::check_native_extent_pair<gemm::SME_F32F32, 19, 53, 17>();
  test::matmul::check_native_extent_pair<gemm::SME_BF16F32, 19, 21, 23>();
  test::matmul::check_native_extent_pair<gemm::SME_F16F32, 19, 21, 23>();
#if defined(HAS_SME_F64F64)
  test::matmul::check_native_extent_pair<gemm::SME_F64F64, 11, 13, 17>();
#endif
  test::matmul::check_native_extent_pair<
      gemm::SME_I8I32<int8_t, int8_t>, 19, 21, 23>();
  test::matmul::check_native_extent_pair<
      gemm::SME_I8I32<uint8_t, int8_t>, 16, 128, 128>();
}

#endif

} // namespace

#endif // VECOPS_TARGET_SHARD_ACTIVE

#endif
