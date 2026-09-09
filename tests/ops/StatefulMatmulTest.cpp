#include <gtest/gtest.h>

#include <vector>

#include "MatmulTestArch.h"
#include "vecops/execution/WorkspaceContext.h"
#include "vecops/execution/WorkspacePlan.h"
#include "vecops/ops/Matmul.h"

namespace {

using namespace vecops;
using namespace vecops::tensor;

#if defined(ARCH_X86_FAMILY)
using TestAtom = matmul::AMX_BF16F32;
#elif defined(HAS_SME)
using TestAtom = matmul::SME_BF16F32;
#endif

#if defined(ARCH_X86_FAMILY) || defined(HAS_SME)

TEST(StatefulMatmulTest, ConstructionBindsScratchAndRepeatedRunSkipsPlanning) {
  ASSERT_TRUE(test::matmul::MatmulTestArchTraits::enable());
  constexpr nint_t M = 16;
  constexpr nint_t N = 32;
  constexpr nint_t K = 64;
  std::vector<vecops::bfloat16_t> a(M * K);
  std::vector<vecops::bfloat16_t> b(N * K);
  std::vector<vecops::float32_t> c(M * N);
  for (nint_t index = 0; index < M * K; ++index)
    a[index] = vecops::bfloat16_t(float(index % 13 - 6) / 13.0f);
  for (nint_t index = 0; index < N * K; ++index)
    b[index] = vecops::bfloat16_t(float(index % 11 - 5) / 11.0f);
  auto at = make_tensor(a.data(), make_shape(meta::cint<M>, meta::cint<K>));
  auto bt = make_tensor(b.data(), make_shape(meta::cint<N>, meta::cint<K>));
  auto ct = make_tensor(c.data(), make_shape(meta::cint<M>, meta::cint<N>));

  auto unbound = ops::matmul(ops::MatmulConfig<TestAtom>{});
  const nint_t required = unbound.required_workspace(meta::cint<M>, meta::cint<N>, meta::cint<K>, at, bt, ct);
  execution::WorkspaceContext tracing{execution::trace_workspace, "stateful_matmul", {1, 1}};
  {
    auto phase = tracing.serial_scope("product");
    auto operation = unbound.prepare(tracing, meta::cint<M>, meta::cint<N>, meta::cint<K>, at, bt, ct);
    EXPECT_EQ(operation.workspace_bytes(), required);
    operation();
  }
  const auto first = c;
  auto placement = execution::place_workspace(tracing.finish_trace());
  kernel::Workspace arena(placement.fast_bytes);
  auto arena_view = arena.view();
  void* arena_base = arena_view.allocate(placement.fast_bytes, vec::DEFAULT_ALIGNMENT);
  execution::BoundWorkspacePlan bound{placement, arena_base, placement.fast_bytes, nullptr, 0};
  execution::WorkspaceContext replay{"stateful_matmul", bound};
  std::fill(c.begin(), c.end(), 0.0f);
  {
    auto phase = replay.serial_scope("product");
    auto operation = unbound.prepare(replay, meta::cint<M>, meta::cint<N>, meta::cint<K>, at, bt, ct);
    EXPECT_EQ(operation.workspace_bytes(), required);
    operation();
  }
  EXPECT_EQ(c, first);

  for (nint_t row = 0; row < M; ++row) {
    for (nint_t column = 0; column < N; ++column) {
      float expected = 0;
      for (nint_t reduction = 0; reduction < K; ++reduction) {
        expected += static_cast<float>(a[row * K + reduction]) * static_cast<float>(b[column * K + reduction]);
      }
      EXPECT_NEAR(c[row * N + column], expected, 2.0e-4f);
    }
  }
}

TEST(StatefulMatmulTest, RunsInsideExistingResourceScopeWithPrivateWorkspace) {
  ASSERT_TRUE(test::matmul::MatmulTestArchTraits::enable());
  constexpr nint_t M = 16;
  constexpr nint_t N = 16;
  constexpr nint_t K = 32;
  std::vector<vecops::bfloat16_t> a(M * K, vecops::bfloat16_t{1});
  std::vector<vecops::bfloat16_t> b(N * K, vecops::bfloat16_t{1});
  std::vector<vecops::float32_t> c(M * N);
  auto at = make_tensor(a.data(), make_shape(meta::cint<M>, meta::cint<K>));
  auto bt = make_tensor(b.data(), make_shape(meta::cint<N>, meta::cint<K>));
  auto ct = make_tensor(c.data(), make_shape(meta::cint<M>, meta::cint<N>));
  auto unbound = ops::matmul(ops::MatmulConfig<TestAtom>{});
  const nint_t required = unbound.required_workspace(meta::cint<M>, meta::cint<N>, meta::cint<K>, at, bt, ct);
  kernel::Workspace parent_storage(required + 256);
  auto parent = parent_storage.view();
  auto operation = unbound.prepare(parent, meta::cint<M>, meta::cint<N>, meta::cint<K>, at, bt, ct);

  kernel::Workspace unrelated_storage(256);
  auto unrelated = unrelated_storage.view();
  ExecutionSession execution{unrelated};
  execution.with_region(operation, [&](auto& scope) VECOPS_INLINE_LAMBDA_NOEXCEPT { operation(scope); });
  EXPECT_FLOAT_EQ(c.front(), float(K));
  EXPECT_FLOAT_EQ(c.back(), float(K));
}

#endif

} // namespace
