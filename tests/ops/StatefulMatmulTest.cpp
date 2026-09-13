#include <gtest/gtest.h>

#include <cmath>
#include <cstring>
#include <memory>
#include <vector>

#include "MatmulTestArch.h"
#include "vecops/execution/WorkspaceContext.h"
#include "vecops/execution/WorkspacePlan.h"
#include "vecops/ops/Matmul.h"
#include "vecops/ops/MatmulPack.h"

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

TEST(StatefulMatmulTest, UnboundPatternsRebindPerLaneWithoutPublicScratchHandling) {
  ASSERT_TRUE(test::matmul::MatmulTestArchTraits::enable());
  constexpr nint_t P = 2;
  constexpr nint_t M = 16;
  constexpr nint_t N = 16;
  constexpr nint_t K = 32;
  std::vector<vecops::bfloat16_t> a(P * M * K);
  std::vector<vecops::bfloat16_t> b(P * N * K);
  std::vector<vecops::float32_t> c(P * M * N, -1.0f);
  for (nint_t lane = 0; lane < P; ++lane) {
    std::fill_n(a.data() + lane * M * K, M * K, vecops::bfloat16_t{float(lane + 1)});
    std::fill_n(b.data() + lane * N * K, N * K, vecops::bfloat16_t{float(2 * lane + 1)});
  }

  const auto shape_a = make_shape(meta::cint<M>, meta::cint<K>);
  const auto shape_b = make_shape(meta::cint<N>, meta::cint<K>);
  const auto shape_c = make_shape(meta::cint<M>, meta::cint<N>);
  auto sample_a = make_tensor(a.data(), shape_a);
  auto sample_b = make_tensor(b.data(), shape_b);
  auto sample_c = make_tensor(c.data(), shape_c);

  execution::WorkspaceContext workspace{execution::trace_workspace, "pattern_matmul"};
  {
    auto phase = workspace.serial_scope("product");
    auto operation =
      ops::matmul(ops::MatmulConfig<TestAtom>{})
        .template prepare<P>(workspace, "product_scratch", meta::cint<M>, meta::cint<N>, meta::cint<K>,
                             tensor::unbind(sample_a), tensor::unbind(sample_b), tensor::unbind(sample_c));

    workspace.parallel_lanes<P>([&](execution::TaskContext<P> task) {
      const nint_t lane = task.lane_id();
      operation(task, make_tensor(a.data() + lane * M * K, shape_a), make_tensor(b.data() + lane * N * K, shape_b),
                make_tensor(c.data() + lane * M * N, shape_c));
    });
  }

  auto plan = workspace.finish_trace();
  ASSERT_EQ(plan.allocations.size(), 1u);
  EXPECT_EQ(plan.allocations.front().request.domain, execution::WorkspaceDomain::WorkerLocal);
  EXPECT_EQ(plan.allocations.front().request.replicas, P);
  for (nint_t lane = 0; lane < P; ++lane) {
    const float expected = float(K) * float(lane + 1) * float(2 * lane + 1);
    for (nint_t index = 0; index < M * N; ++index)
      EXPECT_FLOAT_EQ(c[lane * M * N + index], expected);
  }
}

TEST(StatefulMatmulTest, ExplicitAccumulatorPatternsPreserveRuntimeOperands) {
  ASSERT_TRUE(test::matmul::MatmulTestArchTraits::enable());
  constexpr nint_t M = 16;
  constexpr nint_t N = 16;
  constexpr nint_t K = 32;
  std::vector<vecops::bfloat16_t> a(M * K, vecops::bfloat16_t{2});
  std::vector<vecops::bfloat16_t> b(N * K, vecops::bfloat16_t{3});
  std::vector<vecops::float32_t> c_input(M * N, 5.0f);
  std::vector<vecops::float32_t> c_output(M * N, -1.0f);
  auto at = make_tensor(a.data(), make_shape(meta::cint<M>, meta::cint<K>));
  auto bt = make_tensor(b.data(), make_shape(meta::cint<N>, meta::cint<K>));
  auto cit = make_tensor(c_input.data(), make_shape(meta::cint<M>, meta::cint<N>));
  auto cot = make_tensor(c_output.data(), make_shape(meta::cint<M>, meta::cint<N>));

  execution::WorkspaceContext workspace{"pattern_matmul_explicit"};
  auto operation =
    ops::matmul(ops::MatmulConfig<TestAtom>{})
      .template prepare<1>(workspace, "accumulate_scratch", meta::cint<M>, meta::cint<N>, meta::cint<K>,
                           tensor::unbind(at), tensor::unbind(bt), tensor::unbind(cit), tensor::unbind(cot));
  operation(execution::TaskContext<1>{0}, at, bt, cit, cot);

  for (float value : c_output)
    EXPECT_FLOAT_EQ(value, 5.0f + float(K * 2 * 3));
}

TEST(StatefulMatmulTest, IndependentNamedPatternsRemainLiveAndUseDistinctPlacement) {
  ASSERT_TRUE(test::matmul::MatmulTestArchTraits::enable());
  constexpr nint_t M = 16;
  constexpr nint_t N = 16;
  constexpr nint_t K = 32;
  std::vector<vecops::bfloat16_t> a(M * K);
  std::vector<vecops::bfloat16_t> b(N * K);
  std::vector<vecops::float32_t> c(M * N);
  auto at = make_tensor(a.data(), make_shape(meta::cint<M>, meta::cint<K>));
  auto bt = make_tensor(b.data(), make_shape(meta::cint<N>, meta::cint<K>));
  auto ct = make_tensor(c.data(), make_shape(meta::cint<M>, meta::cint<N>));
  auto operation_factory = ops::matmul(ops::MatmulConfig<TestAtom>{});

  execution::WorkspaceContext workspace{execution::trace_workspace, "named_pattern_matmul"};
  auto first = operation_factory.template prepare<2>(workspace, "first", meta::cint<M>, meta::cint<N>, meta::cint<K>,
                                                     tensor::unbind(at), tensor::unbind(bt), tensor::unbind(ct));
  auto second = operation_factory.template prepare<2>(workspace, "second", meta::cint<M>, meta::cint<N>, meta::cint<K>,
                                                      tensor::unbind(at), tensor::unbind(bt), tensor::unbind(ct));
  (void)first;
  (void)second;
  EXPECT_THROW((operation_factory.template prepare<2>(workspace, "first", meta::cint<M>, meta::cint<N>, meta::cint<K>,
                                                      tensor::unbind(at), tensor::unbind(bt), tensor::unbind(ct))),
               std::runtime_error);
  auto plan = workspace.finish_trace();
  ASSERT_EQ(plan.allocations.size(), 2u);
  EXPECT_NE(plan.allocations[0].site, plan.allocations[1].site);
  auto placement = execution::place_workspace(plan);
  ASSERT_EQ(placement.entries.size(), 2u);
  const nint_t first_bytes =
      placement.entries[0].replica_stride * placement.entries[0].replicas;
  const nint_t second_bytes =
      placement.entries[1].replica_stride * placement.entries[1].replicas;
  // Zero-scratch architecture leaves still retain distinct logical sites,
  // but both empty physical ranges canonically start at offset zero.
  if (first_bytes != 0 && second_bytes != 0)
    EXPECT_NE(placement.entries[0].offset, placement.entries[1].offset);
  const nint_t expected_peak = placement.entries[0].replica_stride * placement.entries[0].replicas +
                               placement.entries[1].replica_stride * placement.entries[1].replicas;
  EXPECT_EQ(placement.fast_bytes, expected_peak);
}

TEST(StatefulMatmulTest, DynamicCapacityRunsFullAndShortenedTailExtents) {
  ASSERT_TRUE(test::matmul::MatmulTestArchTraits::enable());
  constexpr nint_t MaxM = 32;
  constexpr nint_t TailM = 17;
  constexpr nint_t N = 16;
  constexpr nint_t K = 32;
  using MExtent = meta::Dynamic<1, 1, MaxM>;
  std::vector<vecops::bfloat16_t> a(MaxM * K, vecops::bfloat16_t{2});
  std::vector<vecops::bfloat16_t> b(N * K, vecops::bfloat16_t{3});
  std::vector<vecops::float32_t> full(MaxM * N, -1.0f);
  std::vector<vecops::float32_t> tail(MaxM * N, -1.0f);
  auto make_a = [&](nint_t m) {
    return make_tensor(a.data(),
                       make_layout(make_shape(MExtent{m}, meta::cint<K>), make_strides(meta::cint<K>, meta::cint<1>)));
  };
  auto bt = make_tensor(b.data(), make_shape(meta::cint<N>, meta::cint<K>));
  auto make_c = [&](vecops::float32_t* data, nint_t m) {
    return make_tensor(data,
                       make_layout(make_shape(MExtent{m}, meta::cint<N>), make_strides(meta::cint<N>, meta::cint<1>)));
  };

  auto capacity_a = make_a(MaxM);
  auto capacity_c = make_c(full.data(), MaxM);
  execution::WorkspaceContext workspace{"dynamic_pattern_matmul"};
  auto operation = ops::matmul(ops::MatmulConfig<TestAtom>{})
                     .template prepare<1>(workspace, "product", MExtent{MaxM}, meta::cint<N>, meta::cint<K>,
                                          tensor::unbind(capacity_a), tensor::unbind(bt), tensor::unbind(capacity_c));

  // Exercise the preferred explicit-active-extent form on the full bucket.
  operation(execution::TaskContext<1>{0}, MExtent{MaxM}, meta::cint<N>, meta::cint<K>, capacity_a, bt, capacity_c);
  // The convenience form derives the shortened M from the rebound layouts.
  operation(execution::TaskContext<1>{0}, make_a(TailM), bt, make_c(tail.data(), TailM));

  constexpr float Expected = float(K * 2 * 3);
  for (float value : full)
    EXPECT_FLOAT_EQ(value, Expected);
  for (nint_t index = 0; index < TailM * N; ++index)
    EXPECT_FLOAT_EQ(tail[index], Expected);
  for (nint_t index = TailM * N; index < MaxM * N; ++index)
    EXPECT_FLOAT_EQ(tail[index], -1.0f);
}

TEST(StatefulMatmulTest, StatefulOutputPatternUsesRuntimeTransformState) {
  ASSERT_TRUE(test::matmul::MatmulTestArchTraits::enable());
  constexpr nint_t M = 16;
  constexpr nint_t N = 16;
  constexpr nint_t K = 32;
  std::vector<vecops::bfloat16_t> a(M * K, vecops::bfloat16_t{1});
  std::vector<vecops::bfloat16_t> b(N * K, vecops::bfloat16_t{1});
  std::vector<vecops::float32_t> c(M * N, -1.0f);
  float planning_scale = 2.0f;
  float runtime_scale = 3.0f;
  auto at = make_tensor(a.data(), make_shape(meta::cint<M>, meta::cint<K>));
  auto bt = make_tensor(b.data(), make_shape(meta::cint<N>, meta::cint<K>));
  auto ct = make_tensor(c.data(), make_shape(meta::cint<M>, meta::cint<N>));
  auto make_scale_transform = [](float* scale) {
    return tensor::make_elementwise_vec_transform<float, float>(
      [scale](auto tag, auto value) VECOPS_KERNEL_LAMBDA { return vec::mul(tag, value, vec::fill(tag, *scale)); });
  };
  auto planning_output = tensor::output<float>(ct, make_scale_transform(&planning_scale));
  auto runtime_output = tensor::output<float>(ct, make_scale_transform(&runtime_scale));

  execution::WorkspaceContext workspace{"stateful_pattern_matmul"};
  auto operation = ops::matmul(ops::MatmulConfig<TestAtom>{})
                     .template prepare<1>(workspace, "product", meta::cint<M>, meta::cint<N>, meta::cint<K>,
                                          tensor::unbind(at), tensor::unbind(bt), tensor::unbind(planning_output));
  operation(execution::TaskContext<1>{0}, at, bt, runtime_output);

  for (float value : c)
    EXPECT_FLOAT_EQ(value, float(K) * runtime_scale);
}

TEST(StatefulMatmulTest, BatchedPlanningFlattensUnboundOperandsStructurally) {
  ASSERT_TRUE(test::matmul::MatmulTestArchTraits::enable());
  constexpr nint_t Batch = 8;
  constexpr nint_t M = 1;
  constexpr nint_t N = 8;
  constexpr nint_t K = 65;
  std::vector<vecops::bfloat16_t> a(M * K, vecops::bfloat16_t{1});
  std::vector<vecops::bfloat16_t> b(Batch * N * K, vecops::bfloat16_t{1});
  std::vector<vecops::float32_t> c(Batch * M * N, -1.0f);
  auto at = make_tensor(a.data(), make_layout(make_shape(meta::cint<Batch>, meta::cint<M>, meta::cint<K>),
                                              make_strides(meta::cint<0>, meta::cint<K>, meta::cint<1>)));
  auto bt = make_tensor(b.data(), make_shape(meta::cint<Batch>, meta::cint<N>, meta::cint<K>));
  auto ct = make_tensor(c.data(), make_shape(meta::cint<Batch>, meta::cint<M>, meta::cint<N>));

  execution::WorkspaceContext workspace{"batched_pattern_matmul"};
  auto operation = ops::matmul(ops::MatmulConfig<TestAtom>{})
                     .template prepare<1>(workspace, "product", meta::cint<M>, meta::cint<N>, meta::cint<K>,
                                          tensor::unbind(at), tensor::unbind(bt), tensor::unbind(ct));
  operation(execution::TaskContext<1>{0}, at, bt, ct);

  for (float value : c)
    EXPECT_FLOAT_EQ(value, float(K));
}

template <nint_t Parallelism>
void run_packed_a_strided_head_pattern_test() {
  constexpr nint_t M = 5;
  constexpr nint_t N = 8;
  constexpr nint_t K = 5;
  constexpr nint_t ParentChannels = 16;
  std::vector<vecops::bfloat16_t> logical_a(M * K);
  kernel::Workspace parent_b_storage(M * ParentChannels * nint_t{sizeof(vecops::bfloat16_t)} + 64);
  auto parent_b_storage_view = parent_b_storage.view();
  auto* parent_b = static_cast<vecops::bfloat16_t*>(
    parent_b_storage_view.allocate(M * ParentChannels * nint_t{sizeof(vecops::bfloat16_t)}, 64));
  std::fill_n(parent_b, M * ParentChannels + 32,
              vecops::bfloat16_t{std::nanf("")});
  for (nint_t index = 0; index < M * K; ++index)
    logical_a[index] = vecops::bfloat16_t{float(index % 7 + 1) / 13.0f};
  for (nint_t index = 0; index < M * ParentChannels; ++index)
    parent_b[index] = vecops::bfloat16_t{float(index % 11 - 5) / 17.0f};

  auto logical_a_tensor = make_tensor(logical_a.data(), make_shape(meta::cint<M>, meta::cint<K>));
  const auto packed_layout = matmul::packed_layout<TestAtom, matmul::Operand::A>(logical_a_tensor.layout());
  const nint_t packed_bytes = numel(packed_layout) * nint_t{sizeof(vecops::bfloat16_t)};
  kernel::Workspace packed_storage(packed_bytes + 64);
  auto packed_storage_view = packed_storage.view();
  auto* packed_data = static_cast<vecops::bfloat16_t*>(packed_storage_view.allocate(packed_bytes, 64));
  std::memset(packed_data, 0xff, static_cast<std::size_t>(packed_bytes));
  auto packed_a = make_tensor(packed_data, packed_layout);
  auto packer = ops::matmul_pack(ops::MatmulPackConfig<TestAtom, matmul::Operand::A>{});
  kernel::Workspace pack_scratch_storage(packer.required_workspace(logical_a_tensor, packed_a));
  auto pack_workspace = pack_scratch_storage.view();
  packer(pack_workspace, logical_a_tensor, packed_a);

  auto parent_b_tensor = make_tensor(parent_b, make_layout(make_shape(meta::cint<M>, meta::cint<ParentChannels>),
                                                           make_strides(meta::cint<ParentChannels>, meta::cint<1>)));
  auto b = transpose<0, 1>(parent_b_tensor(reserve, range(meta::cint<0>, meta::cint<N>)));
  std::vector<vecops::bfloat16_t> legacy_parent(M * ParentChannels, vecops::bfloat16_t{std::nanf("")});
  auto legacy_parent_tensor =
    make_tensor(legacy_parent.data(), make_layout(make_shape(meta::cint<M>, meta::cint<ParentChannels>),
                                                  make_strides(meta::cint<ParentChannels>, meta::cint<1>)));
  auto legacy_c =
    make_tensor(legacy_parent_tensor.data(), make_layout(make_shape(meta::cint<M>, meta::cint<N>),
                                                         make_strides(meta::cint<ParentChannels>, meta::cint<1>)));
  auto matmul_op = ops::matmul(ops::MatmulConfig<TestAtom>{});
  const nint_t legacy_bytes =
    matmul_op.required_workspace(meta::cint<M>, meta::cint<N>, meta::cint<K>, packed_a, b, legacy_c);
  const nint_t pattern_bytes = matmul_op.required_workspace(
    meta::cint<M>, meta::cint<N>, meta::cint<K>, tensor::unbind(packed_a), tensor::unbind(b),
    tensor::unbind(legacy_c));
  EXPECT_EQ(pattern_bytes, legacy_bytes);
  kernel::Workspace legacy_storage(legacy_bytes);
  auto legacy_root = legacy_storage.view();
  void* legacy_data = legacy_root.allocate(legacy_bytes);
  std::memset(legacy_data, 0xff, static_cast<std::size_t>(legacy_bytes));
  kernel::WorkspaceView legacy_workspace{legacy_data, legacy_bytes};
  ExecutionSession legacy_execution{legacy_workspace};
  for (nint_t head = 0; head < ParentChannels / N; ++head) {
    auto head_b = make_tensor(parent_b_tensor.data() + head * N, b.layout());
    auto head_c = make_tensor(legacy_parent_tensor.data() + head * N, legacy_c.layout());
    legacy_workspace.reset();
    matmul_op(legacy_execution, meta::cint<M>, meta::cint<N>, meta::cint<K>,
              tensor::input<typename TestAtom::TA>(packed_a),
              tensor::input<typename TestAtom::TB>(head_b), head_c);
  }

  constexpr nint_t ArenaBytes = 1 << 20;
  auto arena = std::make_unique<std::byte[]>(ArenaBytes);
  std::memset(arena.get(), 0xff, ArenaBytes);
  execution::WorkspaceContext workspace{"packed_a_strided_head", arena.get(), ArenaBytes};
  auto prepared_workers =
    workspace.worker_tensor<vecops::bfloat16_t, Parallelism>("weighted", legacy_parent_tensor.layout());
  auto pattern_c = prepared_workers(0, reserve, range(meta::cint<0>, meta::cint<N>));
  auto operation =
    matmul_op.template prepare<Parallelism>(workspace, "product", meta::cint<M>, meta::cint<N>, meta::cint<K>,
                                            tensor::unbind(packed_a), tensor::unbind(b), tensor::unbind(pattern_c));
  workspace.parallel_lanes<Parallelism>([&](execution::TaskContext<Parallelism> task) {
    auto parent_c = task.local(prepared_workers);
    for (nint_t head = 0; head < ParentChannels / N; ++head) {
      auto head_b = make_tensor(parent_b_tensor.data() + head * N, b.layout());
      auto head_c = make_tensor(parent_c.data() + head * N, legacy_c.layout());
      operation(task, packed_a, head_b, head_c);
    }
    for (nint_t row = 0; row < M; ++row) {
      for (nint_t column = 0; column < ParentChannels; ++column) {
        const auto actual = static_cast<float>(parent_c(row, column));
        EXPECT_TRUE(std::isfinite(actual));
        EXPECT_EQ(parent_c(row, column), legacy_parent_tensor(row, column));
      }
    }
  });
}

TEST(StatefulMatmulTest, PackedAStridedHeadMatchesLegacyAtOneAndFourLanes) {
  ASSERT_TRUE(test::matmul::MatmulTestArchTraits::enable());
  run_packed_a_strided_head_pattern_test<1>();
  run_packed_a_strided_head_pattern_test<4>();
}

template <nint_t Parallelism>
void run_packed_a_strided_head_replay_cache_test() {
  constexpr nint_t M = 5;
  constexpr nint_t N = 8;
  constexpr nint_t K = 5;
  constexpr nint_t ParentChannels = 16;
  std::vector<vecops::bfloat16_t> logical_a(M * K);
  std::vector<vecops::bfloat16_t> parent_b(M * ParentChannels);
  for (nint_t index = 0; index < M * K; ++index)
    logical_a[index] = vecops::bfloat16_t{float(index % 7 + 1) / 13.0f};
  for (nint_t index = 0; index < M * ParentChannels; ++index)
    parent_b[index] = vecops::bfloat16_t{float(index % 11 - 5) / 17.0f};

  auto logical_a_tensor = make_tensor(logical_a.data(), make_shape(meta::cint<M>, meta::cint<K>));
  const auto packed_layout = matmul::packed_layout<TestAtom, matmul::Operand::A>(logical_a_tensor.layout());
  const nint_t packed_bytes = numel(packed_layout) * nint_t{sizeof(vecops::bfloat16_t)};
  kernel::Workspace packed_storage(packed_bytes + 64);
  auto packed_storage_view = packed_storage.view();
  auto* packed_data = static_cast<vecops::bfloat16_t*>(packed_storage_view.allocate(packed_bytes, 64));
  auto packed_a = make_tensor(packed_data, packed_layout);
  auto packer = ops::matmul_pack(ops::MatmulPackConfig<TestAtom, matmul::Operand::A>{});
  kernel::Workspace pack_scratch_storage(packer.required_workspace(logical_a_tensor, packed_a));
  auto pack_workspace = pack_scratch_storage.view();
  packer(pack_workspace, logical_a_tensor, packed_a);

  auto parent_b_tensor =
    make_tensor(parent_b.data(), make_layout(make_shape(meta::cint<M>, meta::cint<ParentChannels>),
                                             make_strides(meta::cint<ParentChannels>, meta::cint<1>)));
  auto b = transpose<0, 1>(parent_b_tensor(reserve, range(meta::cint<0>, meta::cint<N>)));
  auto matmul_op = ops::matmul(ops::MatmulConfig<TestAtom>{});
  execution::WorkspaceReplayCache cache{2};

  for (int invocation = 0; invocation < 2; ++invocation) {
    cache.invoke(
      "packed_a_strided_replay", [](auto&) {},
      [&](execution::WorkspaceContext& workspace) {
        auto phase = workspace.serial_scope("attention");
        auto output_workers = workspace.worker_tensor<vecops::bfloat16_t, Parallelism>(
          "weighted", make_layout(make_shape(meta::cint<M>, meta::cint<ParentChannels>),
                                  make_strides(meta::cint<ParentChannels>, meta::cint<1>)));
        auto pattern_c = output_workers(0, reserve, range(meta::cint<0>, meta::cint<N>));
        auto operation = matmul_op.template prepare<Parallelism>(workspace, "product", meta::cint<M>, meta::cint<N>,
                                                                 meta::cint<K>, tensor::unbind(packed_a),
                                                                 tensor::unbind(b), tensor::unbind(pattern_c));
        workspace.parallel_lanes<Parallelism>([&](execution::TaskContext<Parallelism> task) {
          auto parent_c = task.local(output_workers);
          for (nint_t head = 0; head < ParentChannels / N; ++head) {
            auto head_b = make_tensor(parent_b_tensor.data() + head * N, b.layout());
            auto head_c = make_tensor(parent_c.data() + head * N, pattern_c.layout());
            operation(task, packed_a, head_b, head_c);
          }
          for (nint_t row = 0; row < M; ++row) {
            for (nint_t column = 0; column < ParentChannels; ++column)
              EXPECT_TRUE(std::isfinite(static_cast<float>(parent_c(row, column))))
                << "invocation=" << invocation << " lane=" << task.lane_id() << " row=" << row << " column=" << column;
          }
        });
      });
  }
}

TEST(StatefulMatmulTest, PackedAStridedHeadIsFiniteOnTraceAndReplay) {
  ASSERT_TRUE(test::matmul::MatmulTestArchTraits::enable());
  run_packed_a_strided_head_replay_cache_test<1>();
  run_packed_a_strided_head_replay_cache_test<4>();
}

template <nint_t Parallelism>
void run_workspace_packed_a_strided_head_replay_cache_test() {
  constexpr nint_t M = 5;
  constexpr nint_t N = 8;
  constexpr nint_t K = 5;
  constexpr nint_t ParentChannels = 16;
  std::vector<vecops::bfloat16_t> parent_b(M * ParentChannels);
  for (nint_t index = 0; index < M * ParentChannels; ++index)
    parent_b[index] = vecops::bfloat16_t{float(index % 11 - 5) / 17.0f};
  auto parent_b_tensor =
    make_tensor(parent_b.data(), make_layout(make_shape(meta::cint<M>, meta::cint<ParentChannels>),
                                             make_strides(meta::cint<ParentChannels>, meta::cint<1>)));
  auto b = transpose<0, 1>(parent_b_tensor(reserve, range(meta::cint<0>, meta::cint<N>)));
  const auto logical_a_layout = make_layout(make_shape(meta::cint<M>, meta::cint<K>));
  const auto packed_layout = matmul::packed_layout<TestAtom, matmul::Operand::A>(logical_a_layout);
  const nint_t packed_elements = numel(packed_layout);
  auto packer = ops::matmul_pack(ops::MatmulPackConfig<TestAtom, matmul::Operand::A>{});
  auto matmul_op = ops::matmul(ops::MatmulConfig<TestAtom>{});
  using RequireGeneral = matmul::family_selection::Require<matmul::kernel_family::General>;
  auto general_matmul_op = ops::matmul(ops::MatmulConfig<TestAtom, RequireGeneral>{});
  std::vector<vecops::bfloat16_t> projection_weight(ParentChannels * ParentChannels);
  for (nint_t index = 0; index < ParentChannels * ParentChannels; ++index)
    projection_weight[index] = vecops::bfloat16_t{float(index % 13 - 6) / 19.0f};
  auto projection_weight_tensor =
    make_tensor(projection_weight.data(), make_shape(meta::cint<ParentChannels>, meta::cint<ParentChannels>));
  const auto projection_packed_layout =
    matmul::packed_layout<TestAtom, matmul::Operand::B>(projection_weight_tensor.layout());
  const nint_t projection_packed_bytes = numel(projection_packed_layout) * nint_t{sizeof(vecops::bfloat16_t)};
  kernel::Workspace projection_packed_storage(projection_packed_bytes + 64);
  auto projection_packed_storage_view = projection_packed_storage.view();
  auto* projection_packed_data =
    static_cast<vecops::bfloat16_t*>(projection_packed_storage_view.allocate(projection_packed_bytes, 64));
  auto projection_packed = make_tensor(projection_packed_data, projection_packed_layout);
  auto projection_packer = ops::matmul_pack(ops::MatmulPackConfig<TestAtom, matmul::Operand::B>{});
  kernel::Workspace projection_pack_scratch(
    projection_packer.required_workspace(projection_weight_tensor, projection_packed));
  auto projection_pack_view = projection_pack_scratch.view();
  projection_packer(projection_pack_view, projection_weight_tensor, projection_packed);
  execution::WorkspaceReplayCache cache{2};
  std::vector<vecops::bfloat16_t> trace_values(Parallelism * M * ParentChannels);

  for (int invocation = 0; invocation < 2; ++invocation) {
    cache.invoke(
      "workspace_packed_a_strided_replay", [](auto&) {},
      [&](execution::WorkspaceContext& workspace) {
        auto global = workspace.serial_scope("global");
        auto logical_a = workspace.tensor<vecops::bfloat16_t>("logical_a", logical_a_layout);
        auto packed_storage = workspace.tensor<vecops::bfloat16_t>("packed_a", make_shape(meta::Any{packed_elements}));
        auto packed_a = make_tensor(packed_storage.data(), packed_layout);
        for (nint_t index = 0; index < M * K; ++index)
          logical_a.data()[index] = vecops::bfloat16_t{float(index % 7 + 1) / 13.0f};
        std::fill_n(packed_storage.data(), packed_elements, vecops::bfloat16_t{std::nanf("")});

        {
          auto closed = workspace.serial_scope("closed_before_pack");
          auto poison = workspace.worker_tensor<std::uint64_t, Parallelism>("poison", make_shape(meta::cint<257>));
          for (nint_t index = 0; index < poison.size(0) * poison.size(1); ++index)
            poison.data()[index] = UINT64_C(0xffffffffffffffff);
        }

        auto pack_phase = workspace.serial_scope("pack");
        auto prepared_pack = packer.template prepare<Parallelism>(workspace, "scratch", tensor::unbind(logical_a),
                                                                  tensor::unbind(packed_a));
        workspace.parallel_for<Parallelism>(
          nint_t{0}, nint_t{1}, meta::cint<1>,
          [&](execution::TaskContext<Parallelism> task, const auto&) { prepared_pack(task, logical_a, packed_a); });
        pack_phase.close();

        auto attention = workspace.serial_scope("attention");
        auto unrelated_a =
          workspace.worker_tensor<std::uint64_t, Parallelism>("unrelated_a", make_shape(meta::cint<113>));
        auto normalized_workers = workspace.worker_tensor<vecops::bfloat16_t, Parallelism>(
          "normalized", make_shape(meta::cint<M>, meta::cint<ParentChannels>));
        auto value_workers = workspace.worker_tensor<vecops::bfloat16_t, Parallelism>(
          "value", make_shape(meta::cint<M>, meta::cint<ParentChannels>));
        auto output_workers = workspace.worker_tensor<vecops::bfloat16_t, Parallelism>(
          "weighted", make_layout(make_shape(meta::cint<M>, meta::cint<ParentChannels>),
                                  make_strides(meta::cint<ParentChannels>, meta::cint<1>)));
        auto unrelated_b =
          workspace.worker_tensor<std::uint64_t, Parallelism>("unrelated_b", make_shape(meta::cint<197>));
        auto pattern_c = output_workers(0, reserve, range(meta::cint<0>, meta::cint<N>));
        auto value_operation = matmul_op.template prepare<Parallelism>(
          workspace, "value_projection", meta::cint<M>, meta::cint<ParentChannels>, meta::cint<ParentChannels>,
          tensor::unbind(normalized_workers(0, reserve, reserve)), tensor::unbind(projection_packed),
          tensor::unbind(value_workers(0, reserve, reserve)));
        auto operation = general_matmul_op.template prepare<Parallelism>(
          workspace, "weighted_head", meta::cint<M>, meta::cint<N>, meta::cint<K>, tensor::unbind(packed_a),
          tensor::unbind(b), tensor::unbind(pattern_c));
        std::fill_n(value_workers.data(), Parallelism * value_workers.stride(0),
                    vecops::bfloat16_t{std::nanf("")});
        workspace.parallel_lanes<Parallelism>([&](execution::TaskContext<Parallelism> task) {
          auto parent_c = task.local(output_workers);
          auto normalized = task.local(normalized_workers);
          auto value = task.local(value_workers);
          auto poison_a = task.local(unrelated_a);
          auto poison_b = task.local(unrelated_b);
          for (nint_t index = 0; index < M * ParentChannels; ++index)
            normalized.data()[index] = vecops::bfloat16_t{float(index % 17 - 8) / 23.0f};
          std::fill_n(parent_c.data(), M * ParentChannels, vecops::bfloat16_t{std::nanf("")});
          std::fill_n(poison_a.data(), poison_a.size(0), UINT64_C(0xffffffffffffffff));
          std::fill_n(poison_b.data(), poison_b.size(0), UINT64_C(0xffffffffffffffff));
          value_operation(task, normalized, projection_packed, value);
          for (nint_t row = 0; row < M; ++row) {
            for (nint_t column = 0; column < ParentChannels; ++column) {
              EXPECT_TRUE(std::isfinite(static_cast<float>(value(row, column))))
                << "stage=value invocation=" << invocation << " lane=" << task.lane_id() << " row=" << row
                << " column=" << column;
              const auto value_index = (task.lane_id() * M + row) * ParentChannels + column;
              if (invocation == 0) {
                trace_values[value_index] = value(row, column);
              } else {
                EXPECT_EQ(static_cast<float>(value(row, column)), static_cast<float>(trace_values[value_index]))
                  << "stage=value-trace-replay lane=" << task.lane_id() << " row=" << row << " column=" << column;
              }
            }
          }
          for (nint_t head = 0; head < ParentChannels / N; ++head) {
            auto head_b = make_tensor(value.data() + head * N, b.layout());
            auto head_c = make_tensor(parent_c.data() + head * N, pattern_c.layout());
            operation(task, packed_a, head_b, head_c);
          }
          for (nint_t row = 0; row < M; ++row) {
            for (nint_t column = 0; column < ParentChannels; ++column)
              EXPECT_TRUE(std::isfinite(static_cast<float>(parent_c(row, column))))
                << "invocation=" << invocation << " lane=" << task.lane_id() << " row=" << row << " column=" << column;
          }
        });
      });
  }
}

TEST(StatefulMatmulTest, WorkspacePackedAStridedHeadIsFiniteOnTraceAndReplay) {
  ASSERT_TRUE(test::matmul::MatmulTestArchTraits::enable());
  run_workspace_packed_a_strided_head_replay_cache_test<1>();
  run_workspace_packed_a_strided_head_replay_cache_test<4>();
}

TEST(StatefulMatmulTest, StridedBTailPackingIgnoresPoisonedPadding) {
  ASSERT_TRUE(test::matmul::MatmulTestArchTraits::enable());
  constexpr nint_t N = 8;
  constexpr nint_t K = 5;
  constexpr nint_t RowStride = 16;
  std::vector<vecops::bfloat16_t> source_storage(K * RowStride + 16,
                                                 vecops::bfloat16_t{std::nanf("")});
  for (nint_t k = 0; k < K; ++k)
    for (nint_t n = 0; n < N; ++n)
      source_storage[k * RowStride + n] = vecops::bfloat16_t{float(k + n + 1) / 17.0f};
  auto source = make_tensor(source_storage.data(),
                            make_layout(make_shape(meta::cint<N>, meta::cint<K>),
                                        make_strides(meta::cint<1>, meta::cint<RowStride>)));
  const auto packed_layout = matmul::packed_layout<TestAtom, matmul::Operand::B>(source.layout());
  std::vector<vecops::bfloat16_t> packed_storage(numel(packed_layout),
                                                 vecops::bfloat16_t{std::nanf("")});
  auto packed = make_tensor(packed_storage.data(), packed_layout);
  auto packer = ops::matmul_pack(ops::MatmulPackConfig<TestAtom, matmul::Operand::B>{});
  kernel::Workspace scratch(packer.required_workspace(source, packed));
  auto scratch_view = scratch.view();
  packer(scratch_view, source, packed);
  for (nint_t index = 0; index < numel(packed_layout); ++index)
    EXPECT_TRUE(std::isfinite(static_cast<float>(packed_storage[index]))) << "packed index=" << index;
}

TEST(StatefulMatmulTest, StridedATailPackingIgnoresPoisonedPadding) {
  ASSERT_TRUE(test::matmul::MatmulTestArchTraits::enable());
  constexpr nint_t M = 8;
  constexpr nint_t K = 5;
  constexpr nint_t ColumnStride = 16;
  std::vector<vecops::bfloat16_t> source_storage(K * ColumnStride + 16,
                                                 vecops::bfloat16_t{std::nanf("")});
  for (nint_t k = 0; k < K; ++k)
    for (nint_t m = 0; m < M; ++m)
      source_storage[k * ColumnStride + m] = vecops::bfloat16_t{float(k + m + 1) / 17.0f};
  auto source = make_tensor(source_storage.data(),
                            make_layout(make_shape(meta::cint<M>, meta::cint<K>),
                                        make_strides(meta::cint<1>, meta::cint<ColumnStride>)));
  const auto packed_layout = matmul::packed_layout<TestAtom, matmul::Operand::A>(source.layout());
  std::vector<vecops::bfloat16_t> packed_storage(numel(packed_layout),
                                                 vecops::bfloat16_t{std::nanf("")});
  auto packed = make_tensor(packed_storage.data(), packed_layout);
  auto packer = ops::matmul_pack(ops::MatmulPackConfig<TestAtom, matmul::Operand::A>{});
  kernel::Workspace scratch(packer.required_workspace(source, packed));
  auto scratch_view = scratch.view();
  packer(scratch_view, source, packed);
  for (nint_t index = 0; index < numel(packed_layout); ++index)
    EXPECT_TRUE(std::isfinite(static_cast<float>(packed_storage[index]))) << "packed index=" << index;
}

#endif

} // namespace
