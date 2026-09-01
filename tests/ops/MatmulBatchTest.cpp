// @vecops-target-shards-x86: 7
// @vecops-target-shards-ARM: 9

#include "vecops/Features.h"
#include "MatmulTestArch.h"

#if defined(ARCH_X86_FAMILY)

#include <gtest/gtest.h>


#include <cstdint>
#include <type_traits>
#include <vector>

#include "vecops/ops/Matmul.h"
#include "vecops/ops/MatmulPack.h"

#include "MatmulConversionTestCommon.h"

using namespace vecops;
using namespace vecops::meta;
using namespace vecops::tensor;

template <typename Atom, bool BroadcastB, nint_t K>
void run_native_batch_case();

#if defined(VECOPS_TARGET_SHARD_ACTIVE)

static_assert(VECOPS_TARGET_SHARD_COUNT == 7);

template <typename Atom, bool BroadcastB, nint_t K>
void run_native_batch_case() {
  if constexpr (BroadcastB) {
    test::matmul::check_batched_native_extent_pair<
        Atom, BroadcastB, 8, 1, 256, K>();
    test::matmul::check_batched_native<Atom, BroadcastB>(
        cint<8>, cint<1>, cint<256>, cint<K>, true);
    test::matmul::check_batched_packed_b<Atom>(8, 1, 256, K);
  } else {
    test::matmul::check_batched_native_extent_pair<
        Atom, BroadcastB, 3, 7, 9, K>();
    test::matmul::check_batched_native<Atom, BroadcastB>(3, 7, 9, K);
  }
}

#if VECOPS_TARGET_SHARD_INDEX == 0
template void run_native_batch_case<gemm::AMX_BF16F32, false, 33>();
#elif VECOPS_TARGET_SHARD_INDEX == 1
template void run_native_batch_case<gemm::AMX_BF16F32, true, 256>();
#elif VECOPS_TARGET_SHARD_INDEX == 2
#if defined(HAS_AMX_FP16)
template void run_native_batch_case<gemm::AMX_F16F32, false, 33>();
template void run_native_batch_case<gemm::AMX_F16F32, true, 256>();
#endif
#elif VECOPS_TARGET_SHARD_INDEX == 3
template void run_native_batch_case<
    gemm::AMX_I8I32<int8_t, int8_t>, false, 65>();
template void run_native_batch_case<
    gemm::AMX_I8I32<int8_t, int8_t>, true, 256>();
#elif VECOPS_TARGET_SHARD_INDEX == 4
template void run_native_batch_case<
    gemm::AMX_I8I32<int8_t, uint8_t>, false, 65>();
template void run_native_batch_case<
    gemm::AMX_I8I32<int8_t, uint8_t>, true, 256>();
#elif VECOPS_TARGET_SHARD_INDEX == 5
template void run_native_batch_case<
    gemm::AMX_I8I32<uint8_t, int8_t>, false, 65>();
template void run_native_batch_case<
    gemm::AMX_I8I32<uint8_t, int8_t>, true, 256>();
#else
template void run_native_batch_case<
    gemm::AMX_I8I32<uint8_t, uint8_t>, false, 65>();
template void run_native_batch_case<
    gemm::AMX_I8I32<uint8_t, uint8_t>, true, 256>();
#endif

#else


template <typename Atom, bool PackedA, bool Bias = false>
void check_shared_a_batch_columns() {
  using TA = typename Atom::TA;
  using TB = typename Atom::TB;
  using Acc = typename Atom::TAcc;
  constexpr nint_t Batch = 8;
  constexpr nint_t M = 1;
  constexpr nint_t N = 8;
  constexpr nint_t K = 65;
  std::vector<TA> a(M * K);
  std::vector<TB> b(Batch * N * K);
  std::vector<Acc> c(Batch * M * N);
  std::vector<Acc> bias(N);
  for (nint_t i = 0; i < M * K; ++i)
    a[i] = test::matmul::conversion_value<TA>(i, 3);
  for (nint_t i = 0; i < Batch * N * K; ++i)
    b[i] = test::matmul::conversion_value<TB>(i, 11);
  for (nint_t i = 0; i < N; ++i)
    bias[i] = test::matmul::conversion_value<Acc>(i, 5);

  auto at = make_tensor(
      a.data(), make_layout(
                    make_shape(Any{Batch}, Any{M}, Any{K}),
                    make_strides(cint<0>, Any{K}, cint<1>)));
  auto bt = make_tensor(
      b.data(), make_layout(make_shape(Any{Batch}, Any{N}, Any{K})));
  auto ct = make_tensor(
      c.data(), make_layout(make_shape(Any{Batch}, Any{M}, Any{N})));

  auto run = [&](const auto& a_input) {
    auto operation = [&] {
      if constexpr (Bias) {
        auto bias_tensor = make_tensor(
            bias.data(), make_layout(
                make_shape(Any{Batch}, Any{M}, Any{N}),
                make_strides(cint<0>, cint<0>, cint<1>)));
        return ops::make_matmul_accumulate<Atom>(
            M, N, K, a_input, bt,
            input<Acc>(bias_tensor), output<Acc>(ct));
      } else {
        return ops::make_matmul<Atom>(M, N, K, a_input, bt, ct);
      }
    }();
    if constexpr (Bias) {
      EXPECT_GT(operation.required_workspace(),
                kernel::matmul_implementation::scratch_bytes<
                    kernel::matmul_implementation::AMX>());
    }
    kernel::Workspace storage(operation.required_workspace());
    auto workspace = storage.view();
    operation(workspace);
  };
  if constexpr (PackedA) {
    auto a2 = make_tensor(
        a.data(), make_layout(make_shape(Any{M}, Any{K})));
    const auto packed_layout =
        ops::matmul_packed_layout<Atom, gemm::Operand::A>(a2.layout());
    std::vector<TA> packed_storage(
        static_cast<std::size_t>(numel(packed_layout) + 64));
    auto* packed_data = reinterpret_cast<TA*>(
        (reinterpret_cast<std::uintptr_t>(packed_storage.data()) + 63u) &
        ~std::uintptr_t{63u});
    auto packed_a = make_tensor(packed_data, packed_layout);
    ExecutionSession execution{};
    ops::matmul_pack<Atom, gemm::Operand::A>(
        execution, input<TA>(a2), packed_a);
    run(input<TA>(packed_a));
  } else {
    run(input<TA>(at));
  }

  for (nint_t batch = 0; batch < Batch; ++batch) {
    for (nint_t n = 0; n < N; ++n) {
      Acc expected = Bias ? bias[n] : Acc{};
      for (nint_t k = 0; k < K; ++k) {
        expected += static_cast<Acc>(a[k]) *
            static_cast<Acc>(b[(batch * N + n) * K + k]);
      }
      if constexpr (std::is_floating_point_v<Acc>) {
        EXPECT_NEAR(c[batch * N + n], expected, 2.0e-4f)
            << "packed_a=" << PackedA << " bias=" << Bias
            << " batch=" << batch << " n=" << n;
      } else {
        EXPECT_EQ(c[batch * N + n], expected)
            << "packed_a=" << PackedA << " bias=" << Bias
            << " batch=" << batch << " n=" << n;
      }
    }
  }
}

TEST(MatmulBatchTest, TraversesBatchAndExposesTilePolicy) {
  ASSERT_TRUE(vecops::test::matmul::MatmulTestArchTraits::enable());
  using Atom = gemm::AMX_BF16F32;
  using Policy = kernel::loop::tile2d_policy::RowMajor;
  constexpr nint_t Batch = 3;
  constexpr nint_t M = 19;
  constexpr nint_t N = 21;
  constexpr nint_t K = 35;
  std::vector<bfloat16_t> a(Batch * M * K);
  std::vector<bfloat16_t> b(Batch * N * K);
  std::vector<float32_t> c(Batch * M * N);
  for (nint_t i = 0; i < Batch * M * K; ++i)
    a[i] = bfloat16_t(static_cast<float>(i % 13 - 6) / 13.0f);
  for (nint_t i = 0; i < Batch * N * K; ++i)
    b[i] = bfloat16_t(static_cast<float>(i % 11 - 5) / 11.0f);

  auto at = make_tensor(
      a.data(), make_layout(make_shape(Any{Batch}, Any{M}, Any{K})));
  auto bt = make_tensor(
      b.data(), make_layout(make_shape(Any{Batch}, Any{N}, Any{K})));
  auto ct = make_tensor(
      c.data(), make_layout(make_shape(Any{Batch}, Any{M}, Any{N})));
  auto operation = ops::make_matmul<Atom, Policy>(M, N, K, at, bt, ct);
  kernel::Workspace storage(operation.required_workspace());
  auto workspace = storage.view();
  operation(workspace);

  for (nint_t batch = 0; batch < Batch; ++batch) {
    for (nint_t m = 0; m < M; ++m) {
      for (nint_t n = 0; n < N; ++n) {
        float expected = 0;
        for (nint_t k = 0; k < K; ++k) {
          expected += static_cast<float>(a[(batch * M + m) * K + k]) *
              static_cast<float>(b[(batch * N + n) * K + k]);
        }
        EXPECT_NEAR(c[(batch * M + m) * N + n], expected, 2.0e-4f)
            << "batch=" << batch << " m=" << m << " n=" << n;
      }
    }
  }
}

TEST(MatmulBatchTest, AllNativeAtomsAndSharedWeights) {
  ASSERT_TRUE(vecops::test::matmul::MatmulTestArchTraits::enable());
  run_native_batch_case<gemm::AMX_BF16F32, false, 33>();
  run_native_batch_case<gemm::AMX_BF16F32, true, 256>();
#if defined(HAS_AMX_FP16)
  run_native_batch_case<gemm::AMX_F16F32, false, 33>();
  run_native_batch_case<gemm::AMX_F16F32, true, 256>();
#endif
#if defined(HAS_AMX_INT8)
  run_native_batch_case<
      gemm::AMX_I8I32<int8_t, int8_t>, false, 65>();
  run_native_batch_case<
      gemm::AMX_I8I32<int8_t, int8_t>, true, 256>();
  run_native_batch_case<
      gemm::AMX_I8I32<int8_t, uint8_t>, false, 65>();
  run_native_batch_case<
      gemm::AMX_I8I32<int8_t, uint8_t>, true, 256>();
  run_native_batch_case<
      gemm::AMX_I8I32<uint8_t, int8_t>, false, 65>();
  run_native_batch_case<
      gemm::AMX_I8I32<uint8_t, int8_t>, true, 256>();
  run_native_batch_case<
      gemm::AMX_I8I32<uint8_t, uint8_t>, false, 65>();
  run_native_batch_case<
      gemm::AMX_I8I32<uint8_t, uint8_t>, true, 256>();
#endif
}

TEST(MatmulBatchTest, ConfiguredBatchHandlesZeroAndShortenedRows) {
  ASSERT_TRUE(vecops::test::matmul::MatmulTestArchTraits::enable());
  using Atom = gemm::AMX_BF16F32;
  test::matmul::check_batched_native<Atom, true>(0, 1, 48, 64);
  test::matmul::check_batched_packed_b<Atom>(8, 1, 32, 64);
  test::matmul::check_batched_native<Atom, true>(
      cint<8>, cint<1>, cint<47>, cint<65>, true);
  test::matmul::check_batched_packed_b<Atom>(8, 1, 47, 65);
  // Exercise the batch-row flattening path beyond decode's M=1 case.  The
  // first product fills one 16-row AMX tile; the second reaches the deliberate
  // 64-row gate and therefore spans four row tiles in one configured problem.
  test::matmul::check_batched_native<Atom, true>(
      cint<4>, cint<4>, cint<127>, cint<129>, true);
  test::matmul::check_batched_packed_b<Atom>(4, 16, 48, 64);
  // N=48 selects the shortened 1x3 TILECFG for M=1.  Follow it with a full
  // 16-row operation to ensure configuration values remain operation-scoped.
  test::matmul::check_batched_packed_b<Atom>(8, 1, 48, 64);
  test::matmul::check_batched_packed_b<Atom>(8, 16, 48, 64);
}

TEST(MatmulBatchTest, SharedAFlattensBatchColumns) {
  ASSERT_TRUE(vecops::test::matmul::MatmulTestArchTraits::enable());
  check_shared_a_batch_columns<gemm::AMX_BF16F32, false>();
  check_shared_a_batch_columns<gemm::AMX_BF16F32, true>();
  check_shared_a_batch_columns<gemm::AMX_BF16F32, false, true>();
  check_shared_a_batch_columns<gemm::AMX_BF16F32, true, true>();
  check_shared_a_batch_columns<
      gemm::AMX_I8I32<int8_t, int8_t>, false, true>();
  check_shared_a_batch_columns<
      gemm::AMX_I8I32<int8_t, uint8_t>, false, true>();
  check_shared_a_batch_columns<
      gemm::AMX_I8I32<uint8_t, int8_t>, false, true>();
  check_shared_a_batch_columns<
      gemm::AMX_I8I32<uint8_t, uint8_t>, false, true>();
}

#endif

#else

#include <gtest/gtest.h>

#include <cstdint>
#include <type_traits>
#include <vector>

#include "vecops/ops/Matmul.h"

#include "MatmulConversionTestCommon.h"

using namespace vecops;
using namespace vecops::meta;
using namespace vecops::tensor;

template <typename Atom, bool BroadcastB, nint_t K>
void run_native_batch_case();

template <typename Atom>
void run_quantized_shared_b_case();

#if defined(VECOPS_TARGET_SHARD_ACTIVE)

static_assert(VECOPS_TARGET_SHARD_COUNT == 9);

template <typename Atom, bool BroadcastB, nint_t K>
void run_native_batch_case() {
  if constexpr (BroadcastB) {
    test::matmul::check_batched_native_extent_pair<
        Atom, BroadcastB, 8, 1, 256, K>();
    test::matmul::check_batched_native<Atom, BroadcastB>(
        cint<8>, cint<1>, cint<256>, cint<K>, true);
    test::matmul::check_batched_packed_b<Atom>(8, 1, 256, K);
  } else {
    test::matmul::check_batched_native_extent_pair<
        Atom, BroadcastB, 3, 7, 9, K>();
    test::matmul::check_batched_native<Atom, BroadcastB>(3, 7, 9, K);
  }
}

template <typename Atom>
void run_quantized_shared_b_case() {
  test::matmul::check_batched_quantized_shared_b<Atom>(
      cint<4>, cint<1>, cint<64>, cint<128>);
}

#if VECOPS_TARGET_SHARD_INDEX == 0
template void run_native_batch_case<gemm::SME_F32F32, false, 9>();
template void run_native_batch_case<gemm::SME_F32F32, true, 256>();
#elif VECOPS_TARGET_SHARD_INDEX == 1
template void run_native_batch_case<gemm::SME_BF16F32, false, 17>();
#elif VECOPS_TARGET_SHARD_INDEX == 2
template void run_native_batch_case<gemm::SME_BF16F32, true, 256>();
#elif VECOPS_TARGET_SHARD_INDEX == 3
template void run_native_batch_case<gemm::SME_F16F32, false, 17>();
template void run_native_batch_case<gemm::SME_F16F32, true, 256>();
#elif VECOPS_TARGET_SHARD_INDEX == 4
#if defined(HAS_SME_F64F64)
template void run_native_batch_case<gemm::SME_F64F64, false, 9>();
template void run_native_batch_case<gemm::SME_F64F64, true, 256>();
#endif
#elif VECOPS_TARGET_SHARD_INDEX == 5
template void run_native_batch_case<
    gemm::SME_I8I32<int8_t, int8_t>, false, 33>();
template void run_native_batch_case<
    gemm::SME_I8I32<int8_t, int8_t>, true, 256>();
#elif VECOPS_TARGET_SHARD_INDEX == 6
template void run_native_batch_case<
    gemm::SME_I8I32<int8_t, uint8_t>, false, 33>();
template void run_native_batch_case<
    gemm::SME_I8I32<int8_t, uint8_t>, true, 256>();
template void run_quantized_shared_b_case<
    gemm::SME_I8I32<int8_t, uint8_t>>();
#elif VECOPS_TARGET_SHARD_INDEX == 7
template void run_native_batch_case<
    gemm::SME_I8I32<uint8_t, int8_t>, false, 33>();
template void run_native_batch_case<
    gemm::SME_I8I32<uint8_t, int8_t>, true, 256>();
#else
template void run_native_batch_case<
    gemm::SME_I8I32<uint8_t, uint8_t>, false, 33>();
template void run_native_batch_case<
    gemm::SME_I8I32<uint8_t, uint8_t>, true, 256>();
#endif

#else

template <typename Atom, bool PackedA, bool Bias = false,
          nint_t Batch = 8, nint_t M = 1,
          nint_t N = 8, nint_t K = 65>
void check_shared_a_batch_columns() {
  using TA = typename Atom::TA;
  using TB = typename Atom::TB;
  using Acc = typename Atom::TAcc;
  std::vector<TA> a(M * K);
  std::vector<TB> b(Batch * N * K);
  std::vector<Acc> c(Batch * M * N);
  std::vector<Acc> bias(N);
  for (nint_t i = 0; i < M * K; ++i)
    a[i] = test::matmul::conversion_value<TA>(i, 3);
  for (nint_t i = 0; i < Batch * N * K; ++i)
    b[i] = test::matmul::conversion_value<TB>(i, 11);
  for (nint_t i = 0; i < N; ++i)
    bias[i] = test::matmul::conversion_value<Acc>(i, 5);

  auto at = make_tensor(
      a.data(), make_layout(
                    make_shape(Any{Batch}, Any{M}, Any{K}),
                    make_strides(cint<0>, Any{K}, cint<1>)));
  auto bt = make_tensor(
      b.data(), make_layout(make_shape(Any{Batch}, Any{N}, Any{K})));
  auto ct = make_tensor(
      c.data(), make_layout(make_shape(Any{Batch}, Any{M}, Any{N})));

  auto run = [&](const auto& a_input) {
    auto operation = [&] {
      if constexpr (Bias) {
        auto bias_tensor = make_tensor(
            bias.data(), make_layout(
                make_shape(Any{Batch}, Any{M}, Any{N}),
                make_strides(cint<0>, cint<0>, cint<1>)));
        return ops::make_matmul_accumulate<Atom>(
            M, N, K, a_input, bt,
            input<Acc>(bias_tensor), output<Acc>(ct));
      } else {
        return ops::make_matmul<Atom>(M, N, K, a_input, bt, ct);
      }
    }();
    const nint_t expected_workspace = Bias
        ? Batch * N * static_cast<nint_t>(sizeof(Acc)) + 63
        : 0;
    EXPECT_EQ(operation.required_workspace(), expected_workspace);
    kernel::Workspace storage(operation.required_workspace());
    auto workspace = storage.view();
    EXPECT_EQ(workspace.used(), 0);
    operation(workspace);
    EXPECT_EQ(workspace.used(), 0);
    // The flattened periodic-bias path must be reusable with the same
    // execution workspace.  In particular, its temporary dense bias may not
    // leak bump allocations across calls.
    operation(workspace);
    EXPECT_EQ(workspace.used(), 0);
  };
  if constexpr (PackedA) {
    auto a2 = make_tensor(
        a.data(), make_layout(make_shape(Any{M}, Any{K})));
    const auto packed_layout =
        ops::matmul_packed_layout<Atom, gemm::Operand::A>(a2.layout());
    kernel::Workspace packed_owner(
        numel(packed_layout) * static_cast<nint_t>(sizeof(TA)) + 64);
    auto packed_workspace = packed_owner.view();
    auto* packed_data = static_cast<TA*>(packed_workspace.allocate(
        numel(packed_layout) * static_cast<nint_t>(sizeof(TA)), 64));
    auto packed_a = make_tensor(packed_data, packed_layout);
    ExecutionSession execution{};
    ops::matmul_pack<Atom, gemm::Operand::A>(
        execution, input<TA>(a2), packed_a);
    run(input<TA>(packed_a));
  } else {
    run(input<TA>(at));
  }

  for (nint_t batch = 0; batch < Batch; ++batch) {
    for (nint_t m = 0; m < M; ++m) {
      for (nint_t n = 0; n < N; ++n) {
        Acc expected = Bias ? bias[n] : Acc{};
        for (nint_t k = 0; k < K; ++k) {
          expected += static_cast<Acc>(a[m * K + k]) *
              static_cast<Acc>(b[(batch * N + n) * K + k]);
        }
        const nint_t offset = (batch * M + m) * N + n;
        if constexpr (std::is_floating_point_v<Acc>) {
          EXPECT_NEAR(c[offset], expected, 3.0e-4)
              << "packed_a=" << PackedA << " bias=" << Bias
              << " batch=" << batch << " m=" << m << " n=" << n;
        } else {
          EXPECT_EQ(c[offset], expected)
              << "packed_a=" << PackedA << " bias=" << Bias
              << " batch=" << batch << " m=" << m << " n=" << n;
        }
      }
    }
  }
}

TEST(MatmulBatchTest, TraversesBatchAndExposesTilePolicy) {
  using Atom = gemm::SME_F32F32;
  using Policy = kernel::loop::tile2d_policy::RowMajor;
  constexpr nint_t Batch = 3;
  constexpr nint_t M = 19;
  constexpr nint_t N = 21;
  constexpr nint_t K = 17;
  std::vector<float32_t> a(Batch * M * K);
  std::vector<float32_t> b(Batch * N * K);
  std::vector<float32_t> c(Batch * M * N);
  for (nint_t i = 0; i < Batch * M * K; ++i)
    a[i] = static_cast<float>(i % 13 - 6) / 13.0f;
  for (nint_t i = 0; i < Batch * N * K; ++i)
    b[i] = static_cast<float>(i % 11 - 5) / 11.0f;

  auto at = make_tensor(
      a.data(), make_layout(make_shape(Any{Batch}, Any{M}, Any{K})));
  auto bt = make_tensor(
      b.data(), make_layout(make_shape(Any{Batch}, Any{N}, Any{K})));
  auto ct = make_tensor(
      c.data(), make_layout(make_shape(Any{Batch}, Any{M}, Any{N})));
  auto operation = ops::make_matmul<Atom, Policy>(M, N, K, at, bt, ct);
  ExecutionSession execution{};
  operation(execution);

  for (nint_t batch = 0; batch < Batch; ++batch) {
    for (nint_t m = 0; m < M; ++m) {
      for (nint_t n = 0; n < N; ++n) {
        float expected = 0;
        for (nint_t k = 0; k < K; ++k) {
          expected += a[(batch * M + m) * K + k] *
              b[(batch * N + n) * K + k];
        }
        EXPECT_NEAR(c[(batch * M + m) * N + n], expected, 3.0e-4f)
            << "batch=" << batch << " m=" << m << " n=" << n;
      }
    }
  }
}

TEST(MatmulBatchTest, AllNativeAtomsAndSharedWeights) {
  run_native_batch_case<gemm::SME_F32F32, false, 9>();
  run_native_batch_case<gemm::SME_F32F32, true, 256>();
  run_native_batch_case<gemm::SME_BF16F32, false, 17>();
  run_native_batch_case<gemm::SME_BF16F32, true, 256>();
  run_native_batch_case<gemm::SME_F16F32, false, 17>();
  run_native_batch_case<gemm::SME_F16F32, true, 256>();
#if defined(HAS_SME_F64F64)
  run_native_batch_case<gemm::SME_F64F64, false, 9>();
  run_native_batch_case<gemm::SME_F64F64, true, 256>();
#endif
  run_native_batch_case<
      gemm::SME_I8I32<int8_t, int8_t>, false, 33>();
  run_native_batch_case<
      gemm::SME_I8I32<int8_t, int8_t>, true, 256>();
  run_native_batch_case<
      gemm::SME_I8I32<int8_t, uint8_t>, false, 33>();
  run_native_batch_case<
      gemm::SME_I8I32<int8_t, uint8_t>, true, 256>();
  run_native_batch_case<
      gemm::SME_I8I32<uint8_t, int8_t>, false, 33>();
  run_native_batch_case<
      gemm::SME_I8I32<uint8_t, int8_t>, true, 256>();
  run_native_batch_case<
      gemm::SME_I8I32<uint8_t, uint8_t>, false, 33>();
  run_native_batch_case<
      gemm::SME_I8I32<uint8_t, uint8_t>, true, 256>();
}

TEST(MatmulBatchTest, SmallQuantizedSharedBUsesLowWorkFullPacking) {
  run_quantized_shared_b_case<
      gemm::SME_I8I32<int8_t, uint8_t>>();
}

TEST(MatmulBatchTest, SharedOperandsFlattenIntoOneProblem) {
  using Atom = gemm::SME_BF16F32;
  test::matmul::check_batched_native<Atom, true>(
      cint<4>, cint<4>, cint<127>, cint<129>, true);
  test::matmul::check_batched_packed_b<Atom>(4, 16, 48, 64);
  check_shared_a_batch_columns<Atom, false>();
  check_shared_a_batch_columns<Atom, true>();
  check_shared_a_batch_columns<Atom, false, true>();
  check_shared_a_batch_columns<Atom, true, true>();
}

TEST(MatmulBatchTest, SharedOperandFlattenBoundariesAndFallbacks) {
  using Atom = gemm::SME_BF16F32;
  // shared-B: M and flat-M upper boundaries, followed by both fallback sides.
  test::matmul::check_batched_native<Atom, true>(
      cint<4>, cint<16>, cint<48>, cint<64>, true);
  test::matmul::check_batched_native<Atom, true>(
      cint<4>, cint<17>, cint<48>, cint<64>, true);
  test::matmul::check_batched_native<Atom, true>(
      cint<8>, cint<8>, cint<47>, cint<65>, true);
  test::matmul::check_batched_native<Atom, true>(
      cint<8>, cint<9>, cint<47>, cint<65>, true);
  test::matmul::check_batched_native<Atom, true>(0, 1, 16, 65, false);
  test::matmul::check_batched_packed_b<Atom>(0, 1, 16, 65);

  // shared-A: batch/N boundary plus batch<4, flat-N>64 and M>1 fallbacks.
  check_shared_a_batch_columns<Atom, false, false, 4, 1, 16, 65>();
  check_shared_a_batch_columns<Atom, false, false, 4, 1, 17, 65>();
  check_shared_a_batch_columns<Atom, false, false, 3, 1, 16, 65>();
  check_shared_a_batch_columns<Atom, false, false, 4, 2, 8, 65>();
}

#endif

#endif
