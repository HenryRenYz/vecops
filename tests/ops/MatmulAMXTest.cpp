#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <sys/syscall.h>
#include <type_traits>
#include <unistd.h>
#include <vector>

#include "vecops/ops/Matmul.h"
#include "vecops/ops/MatmulPack.h"

namespace {

using namespace vecops;
using namespace vecops::meta;
using namespace vecops::tensor;

bool enable_amx() {
  constexpr long ArchRequestXcompPerm = 0x1023;
  constexpr long XfeatureTileData = 18;
  return syscall(
      SYS_arch_prctl, ArchRequestXcompPerm, XfeatureTileData) == 0;
}

template <typename T>
T value(nint_t index, int modulus) {
  const int x = static_cast<int>(index % modulus) - modulus / 2;
  if constexpr (std::is_integral_v<T>) return static_cast<T>(x);
  else return T(static_cast<float>(x) / static_cast<float>(modulus));
}

template <typename Atom, typename M, typename N, typename K>
void check_raw(M m_extent, N n_extent, K k_extent) {
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

template <gemm::Operand Side>
void check_mixed_packing() {
  using Atom = gemm::AMX_BF16F32;
  constexpr nint_t M = 37;
  constexpr nint_t N = 51;
  constexpr nint_t K = 79;
  std::vector<bfloat16_t> a(M * K), b(N * K);
  std::vector<float32_t> c(M * N);
  for (nint_t i = 0; i < M * K; ++i) a[i] = value<bfloat16_t>(i, 13);
  for (nint_t i = 0; i < N * K; ++i) b[i] = value<bfloat16_t>(i, 11);
  auto al = make_layout(make_shape(Any{M}, Any{K}));
  auto bl = make_layout(make_shape(Any{N}, Any{K}));
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
      c.data(), make_layout(make_shape(Any{M}, Any{N})));
  auto operation = [&] {
    if constexpr (Side == gemm::Operand::A)
      return ops::make_matmul<Atom>(M, N, K, packed_tensor, bt, ct);
    else
      return ops::make_matmul<Atom>(M, N, K, at, packed_tensor, ct);
  }();
  kernel::Workspace owner(operation.required_workspace());
  auto workspace = owner.view();
  operation(workspace);
  for (nint_t i = 0; i < M; ++i) {
    for (nint_t j = 0; j < N; ++j) {
      float expected = 0;
      for (nint_t kk = 0; kk < K; ++kk)
        expected += static_cast<float>(a[i * K + kk]) *
                    static_cast<float>(b[j * K + kk]);
      if (std::abs(c[i * N + j] - expected) > 2.0e-4f) {
        ADD_FAILURE()
            << "packed side="
            << (Side == gemm::Operand::A ? "A" : "B")
            << " m=" << i << " n=" << j
            << " actual=" << c[i * N + j]
            << " expected=" << expected;
        return;
      }
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

TEST(MatmulAMXTest, AllResidentMicrokernelShapesAndKTails) {
  ASSERT_TRUE(enable_amx());
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
}

TEST(MatmulAMXTest, CompileTimeExtentsAndFullKPlan) {
  ASSERT_TRUE(enable_amx());
  check_raw<gemm::AMX_BF16F32>(cint<32>, cint<32>, cint<128>);
  check_raw<gemm::AMX_BF16F32>(cint<35>, cint<53>, cint<257>);
  check_raw<gemm::AMX_BF16F32>(
      dyn<32, 32, 64>(32), dyn<16, 32, 64>(48), dyn<32, 64, 128>(96));
}

TEST(MatmulAMXTest, TransformedInputsUseFullAndTailKPlans) {
  ASSERT_TRUE(enable_amx());
  check_transformed_inputs(cint<64>);
  check_transformed_inputs(cint<65>);
}

TEST(MatmulAMXTest, AllInt8SignednessCombinations) {
  ASSERT_TRUE(enable_amx());
  check_raw<gemm::AMX_I8I32<int8_t, int8_t>>(19, 21, 69);
  check_raw<gemm::AMX_I8I32<int8_t, uint8_t>>(19, 21, 69);
  check_raw<gemm::AMX_I8I32<uint8_t, int8_t>>(19, 21, 69);
  check_raw<gemm::AMX_I8I32<uint8_t, uint8_t>>(19, 21, 69);
}

TEST(MatmulAMXTest, OperandAMayBePrepacked) {
  ASSERT_TRUE(enable_amx());
  check_mixed_packing<gemm::Operand::A>();
}

TEST(MatmulAMXTest, OperandBMayBePrepacked) {
  ASSERT_TRUE(enable_amx());
  check_mixed_packing<gemm::Operand::B>();
}

TEST(MatmulAMXTest, CPrologueAndEpilogueUseDataAccess) {
  ASSERT_TRUE(enable_amx());
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

TEST(MatmulAMXTest, FullCDataAccessUsesCompileTimeExtents) {
  ASSERT_TRUE(enable_amx());
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

} // namespace
