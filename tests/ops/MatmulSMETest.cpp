#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <type_traits>
#include <vector>

#include "vecops/ops/Matmul.h"
#include "vecops/ops/MatmulPack.h"

namespace {

using namespace vecops;
using namespace vecops::meta;
using namespace vecops::tensor;

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

template <typename Atom>
void check_raw(nint_t m, nint_t n, nint_t k) {
  using TA = typename Atom::TA;
  using TB = typename Atom::TB;
  using Acc = typename Atom::TAcc;
  std::vector<TA> a(static_cast<std::size_t>(m * k));
  std::vector<TB> b(static_cast<std::size_t>(n * k));
  std::vector<Acc> c(static_cast<std::size_t>(m * n), Acc{});
  for (nint_t i = 0; i < m * k; ++i) a[i] = value<TA>(i, 13);
  for (nint_t i = 0; i < n * k; ++i) b[i] = value<TB>(i, 11);
  auto at = make_tensor(
      a.data(), make_layout(make_shape(Any{m}, Any{k})));
  auto bt = make_tensor(
      b.data(), make_layout(make_shape(Any{n}, Any{k})));
  auto ct = make_tensor(
      c.data(), make_layout(make_shape(Any{m}, Any{n})));
  auto operation = ops::make_matmul<Atom>(m, n, k, at, bt, ct);
  ExecutionSession execution{};
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

template <gemm::Operand Side>
void check_mixed_packing() {
  using Atom = gemm::SME_F32F32;
  const nint_t lanes = static_cast<nint_t>(svcntsw());
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

void check_fast_packed_path() {
  using Atom = gemm::SME_BF16F32;
  using T = typename Atom::TA;
  const nint_t lanes = static_cast<nint_t>(svcntsw());
  const nint_t m = 2 * lanes + 3;
  const nint_t n = 2 * lanes + 5;
  const nint_t k = 73;
  std::vector<T> a(m * k), b(n * k);
  std::vector<float32_t> c(m * n);
  for (nint_t i = 0; i < m * k; ++i) a[i] = value<T>(i, 13);
  for (nint_t i = 0; i < n * k; ++i) b[i] = value<T>(i, 11);
  auto al = make_layout(make_shape(Any{m}, Any{k}));
  auto bl = make_layout(make_shape(Any{n}, Any{k}));
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
  auto operation = ops::make_matmul<Atom>(m, n, k, apt, bpt, ct);
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

TEST(MatmulSMETest, AllZA32MicrokernelShapesAndKTails) {
  const nint_t l = static_cast<nint_t>(svcntsw());
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

TEST(MatmulSMETest, FloatingInputTypes) {
  check_raw<gemm::SME_BF16F32>(19, 21, 23);
  check_raw<gemm::SME_F16F32>(19, 21, 23);
#if defined(HAS_SME_F64F64)
  check_raw<gemm::SME_F64F64>(11, 13, 17);
#endif
}

TEST(MatmulSMETest, AllInt8SignednessCombinations) {
  check_raw<gemm::SME_I8I32<int8_t, int8_t>>(19, 21, 23);
  check_raw<gemm::SME_I8I32<int8_t, uint8_t>>(19, 21, 23);
  check_raw<gemm::SME_I8I32<uint8_t, int8_t>>(19, 21, 23);
  check_raw<gemm::SME_I8I32<uint8_t, uint8_t>>(19, 21, 23);
}

TEST(MatmulSMETest, EitherOperandMayBePrepacked) {
  check_mixed_packing<gemm::Operand::A>();
  check_mixed_packing<gemm::Operand::B>();
}

TEST(MatmulSMETest, BothPackedLargeKUsesFastPath) {
  check_fast_packed_path();
}

TEST(MatmulSMETest, CPrologueAndEpilogueUseDataAccess) {
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

} // namespace
