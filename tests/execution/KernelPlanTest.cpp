// SPDX-FileCopyrightText: 2026 HenryRenYz and vecops contributors
// SPDX-License-Identifier: MIT
#include <gtest/gtest.h>
#include <array>
#include <atomic>
#include "vecops/execution/KernelPlan.h"
#include "vecops/ops/LayerNorm.h"

namespace {
using namespace vecops;
using namespace vecops::execution;
using namespace vecops::tensor;

struct CountingPool {
  const VecopsThreadPoolV1* underlying = vecops_native_thread_pool_v1(4);
  unsigned submissions = 0;
  static uint32_t capacity(void*) {
    return 4;
  }
  static uint32_t nested(void*) {
    return 0;
  }
  static int32_t tasks(void* opaque, uint32_t count, void* context, VecopsParallelTaskFn body, VecopsError* error) {
    const auto* pool = static_cast<CountingPool*>(opaque)->underlying;
    return pool->parallel_for(pool->context, count, context, body, error);
  }
  static int32_t run(void* opaque, uint32_t workers, void* context, VecopsTeamBodyFn body, VecopsError* error) {
    auto& self = *static_cast<CountingPool*>(opaque);
    ++self.submissions;
    return self.underlying->team_executor->run(self.underlying->context, workers, context, body, error);
  }
  VecopsTeamExecutorV1 executor{sizeof(executor), 1, run};
  VecopsThreadPoolV1 abi{sizeof(abi), 1, 1, 123, this, capacity, nested, tasks, nullptr, nullptr, 0, &executor};
};

std::atomic<int> prepared{0};
struct PreparedScale {
  float value;
  float operator()(float input) const {
    return input * value;
  }
};

TEST(KernelPlanTest, RebindsInputsScalarsAndPreservesConstMetaWithOneSubmission) {
  CountingPool pool;
  std::array<float, 17> x{}, y{}, out{};
  KernelBuilder builder(4);
  auto input = builder.input(make_tensor(x.data(), make_shape(meta::cint<17>)));
  auto output = builder.input(make_tensor(out.data(), make_shape(meta::cint<17>)));
  auto scale = builder.scalar<float>();
  auto temp = builder.tensor<float>("temp", make_shape(meta::cint<17>));
  const int before = prepared.load();
  auto op = builder.prepare(kernel_bind(kernel_constant(2.f)), [](float scale) {
    ++prepared;
    return PreparedScale{scale};
  });
  builder.grid<4>("projection", meta::cint<0>, meta::cint<17>, meta::cint<4>, kernel_bind(input, temp, op),
                  [](auto, const auto& item, auto input, auto temp, const auto& op) {
                    static_assert(decltype(input.template size<0>())::is_const);
                    static_assert(decltype(item.chunk())::value == 4);
                    for (auto i = item.begin(); i < item.end(); ++i)
                      temp(i) = op(input(i));
                  });
  // Only two lanes compute; all four resident workers must reach completion.
  builder.grid<2>("epilogue", meta::cint<0>, meta::cint<17>, meta::cint<9>, kernel_bind(temp, output, scale),
                  [](auto, const auto& item, auto temp, auto output, float scale) {
                    for (auto i = item.begin(); i < item.end(); ++i)
                      output(i) = temp(i) + scale;
                  });
  auto plan = builder.finish();
  KernelInstance instance(plan, &pool.abi);
  EXPECT_EQ(prepared.load(), before + 1);
  instance.set(input, x.data());
  instance.set(output, out.data());
  for (int round = 0; round < 5; ++round) {
    auto& source = round % 2 ? x : y;
    for (int i = 0; i < 17; ++i)
      source[i] = round + i;
    instance.set(input, source.data());
    instance.set(scale, float(round));
    instance.replay();
    for (int i = 0; i < 17; ++i)
      EXPECT_EQ(out[i], 2 * source[i] + round);
  }
  EXPECT_EQ(prepared.load(), before + 1);
  EXPECT_EQ(pool.submissions, 5u);
  EXPECT_EQ(instance.replay_count(), 5u);
  EXPECT_THROW(builder.finish(), std::logic_error);
}

TEST(KernelPlanTest, SequenceSharesOneTeamAcrossKernelBoundaries) {
  CountingPool pool;
  std::array<int, 4> values{};
  KernelBuilder first(4), second(4);
  auto a = first.input(make_tensor(values.data(), make_shape(meta::cint<4>)));
  auto b = second.input(make_tensor(values.data(), make_shape(meta::cint<4>)));
  first.grid<4>("write", meta::cint<0>, meta::cint<4>, meta::cint<1>, kernel_bind(a),
                [](auto, const auto& item, auto a) { a(item.begin()) = int(item.begin() + 1); });
  second.single("consume", kernel_bind(b), [](auto b) {
    int total = 0;
    for (int i = 0; i < 4; ++i)
      total += b(i);
    b(0) = total;
  });
  auto x = std::make_shared<KernelInstance>(first.finish(), &pool.abi);
  auto y = std::make_shared<KernelInstance>(second.finish(), &pool.abi);
  x->set(a, values.data());
  y->set(b, values.data());
  KernelSequence sequence;
  sequence.append(x);
  sequence.append(y);
  sequence.replay();
  EXPECT_EQ(values[0], 10);
  EXPECT_EQ(pool.submissions, 1u);
  EXPECT_EQ(x->replay_count(), 1u);
  EXPECT_EQ(y->replay_count(), 1u);
}

TEST(KernelPlanTest, RejectsForeignBindingsAndUnboundRuntimeInputs) {
  CountingPool pool;
  KernelBuilder first(4), second(4);
  auto a = first.scalar<float>();
  auto b = second.scalar<float>();
  EXPECT_THROW(first.single("foreign", kernel_bind(b), [](float) {}), std::invalid_argument);
  EXPECT_THROW(first.prepare(kernel_bind(a), [](float x) { return x; }), std::invalid_argument);
  first.single("local", kernel_bind(a), [](float) {});
  KernelInstance instance(first.finish(), &pool.abi);
  EXPECT_THROW(instance.replay(), std::invalid_argument);
  EXPECT_THROW(instance.set(b, 1.f), std::invalid_argument);
  instance.set(a, 1.f);
  instance.replay();
}

TEST(KernelPlanTest, WorkerExceptionCancelsBarriersAndInstanceCanReplayAgain) {
  CountingPool pool;
  KernelBuilder builder(4);
  auto fail = builder.scalar<bool>();
  builder.grid<4>("throw", meta::cint<0>, meta::cint<4>, meta::cint<1>, kernel_bind(fail),
                  [](auto task, const auto&, bool fail) {
                    if (fail && task.lane_id() == 2)
                      throw std::runtime_error("planned failure");
                  });
  builder.single("after", kernel_bind(), [] {});
  KernelInstance instance(builder.finish(), &pool.abi);
  instance.set(fail, true);
  EXPECT_THROW(instance.replay(), std::runtime_error);
  instance.set(fail, false);
  EXPECT_NO_THROW(instance.replay());
}

TEST(KernelPlanTest, ScratchAndPreparedStateArePrivateToEachInstance) {
  CountingPool pool;
  std::array<int, 4> a{}, b{};
  KernelBuilder builder(4);
  auto output = builder.input(make_tensor(a.data(), make_shape(meta::cint<4>)));
  auto scratch = builder.tensor<int>("scratch", make_shape(meta::cint<4>));
  auto delta = builder.scalar<int>();
  builder.grid<4>("write", meta::cint<0>, meta::cint<4>, meta::cint<1>, kernel_bind(scratch, output, delta),
                  [](auto, const auto& item, auto scratch, auto out, int delta) {
                    scratch(item.begin()) = delta;
                    out(item.begin()) = scratch(item.begin());
                  });
  auto plan = builder.finish();
  KernelInstance x(plan, &pool.abi), y(plan, &pool.abi);
  x.set(output, a.data());
  x.set(delta, 5);
  y.set(output, b.data());
  y.set(delta, 9);
  x.replay();
  y.replay();
  x.replay();
  EXPECT_EQ(a, (std::array<int, 4>{5, 5, 5, 5}));
  EXPECT_EQ(b, (std::array<int, 4>{9, 9, 9, 9}));
}

TEST(KernelPlanTest, WorkerRegionHasOnePrologueAndEpiloguePerActiveLane) {
  CountingPool pool;
  std::array<int, 4> counts{};
  KernelBuilder builder(4);
  auto output = builder.input(make_tensor(counts.data(), make_shape(meta::cint<4>)));
  builder.worker_grid<4>("worker-region", meta::cint<0>, meta::cint<5>, meta::cint<1>, kernel_bind(output),
                         [](auto& worker, auto counts) {
                           int programs = 0;
                           worker.programs([&](auto, const auto&) { ++programs; });
                           counts(worker.lane_id()) = programs;
                         });
  KernelInstance instance(builder.finish(), &pool.abi);
  instance.set(output, counts.data());
  instance.replay();
  EXPECT_EQ(counts, (std::array<int, 4>{2, 1, 1, 1}));
}

TEST(KernelPlanTest, ExistingLayerNormPreparesOnceAndUsesPersistentLaneScratch) {
  CountingPool pool;
  std::array<float, 32> x{}, out{};
  for (int i = 0; i < 32; ++i)
    x[i] = float((i * 7) % 13);
  KernelBuilder builder(4);
  auto input =
    builder.input(make_tensor(static_cast<const float*>(x.data()), make_shape(meta::cint<4>, meta::cint<8>)));
  auto output = builder.input(make_tensor(out.data(), make_shape(meta::cint<4>, meta::cint<8>)));
  auto input_pattern = tensor::make_unbound_tensor<const float>(make_shape(meta::cint<1>, meta::cint<8>));
  auto output_pattern = tensor::make_unbound_tensor<float>(make_shape(meta::cint<1>, meta::cint<8>));
  auto norm = builder.prepare_operation<4>("norm", ops::layer_norm(ops::LayerNormConfig<float>{1e-5f}),
                                           tensor::input<float>(input_pattern), tensor::output<float>(output_pattern));
  builder.grid<4>("normalize", meta::cint<0>, meta::cint<4>, meta::cint<1>, kernel_bind(input, output, norm),
                  [](auto task, const auto& item, auto input, auto output, const auto& norm) {
                    norm(task, input(range_n(item.begin(), meta::cint<1>), reserve),
                         output(range_n(item.begin(), meta::cint<1>), reserve));
                  });
  KernelInstance instance(builder.finish(), &pool.abi);
  instance.set(input, x.data());
  instance.set(output, out.data());
  for (int round = 0; round < 3; ++round) {
    instance.replay();
    for (int row = 0; row < 4; ++row) {
      float mean = 0, square = 0;
      for (int col = 0; col < 8; ++col) {
        float v = x[row * 8 + col];
        mean += v;
        square += v * v;
      }
      mean /= 8;
      square /= 8;
      for (int col = 0; col < 8; ++col)
        EXPECT_NEAR(out[row * 8 + col], (x[row * 8 + col] - mean) / std::sqrt(square - mean * mean + 1e-5f), 1e-4f);
    }
  }
}
TEST(KernelPlanTest, IndirectGridRecomputesDataWorkWithoutRebuildingTopology) {
  CountingPool pool;
  std::array<int, 9> output{};
  KernelBuilder builder(4);
  auto out = builder.input(make_tensor(output.data(), make_shape(meta::cint<9>)));
  auto live = builder.scalar<int>();
  auto count = builder.tensor<nint_t>("live_count", make_shape(meta::cint<1>));
  builder.single("compact", kernel_bind(count, live), [](auto count, int live) { count(0) = live; });
  builder.indirect_grid<4>("active", meta::cint<9>, count, kernel_bind(out),
                           [](auto, const auto& item, auto out) { out(item.begin()) += 1; });
  KernelInstance instance(builder.finish(), &pool.abi);
  instance.set(out, output.data());
  for (int value : {9, 2, 0, 5}) {
    instance.set(live, value);
    instance.replay();
  }
  for (int i = 0; i < 9; ++i)
    EXPECT_EQ(output[i], 1 + (i < 2) + (i < 5));
  EXPECT_EQ(pool.submissions, 4u);
  instance.set(live, 10);
  EXPECT_THROW(instance.replay(), std::runtime_error);
  instance.set(live, 1);
  EXPECT_NO_THROW(instance.replay());
}

TEST(KernelPlanTest, LogicalOnlyProviderUsesSynchronousGridsWithoutResidentBarriers) {
  CountingPool pool;
  pool.abi.team_executor = nullptr;
  std::array<int, 4> output{};
  KernelBuilder builder(4);
  auto out = builder.input(make_tensor(output.data(), make_shape(meta::cint<4>)));
  builder.grid<4>("write", meta::cint<0>, meta::cint<4>, meta::cint<1>, kernel_bind(out),
                  [](auto, const auto& item, auto out) { out(item.begin()) = int(item.begin()); });
  builder.single("sum", kernel_bind(out), [](auto out) { out(0) = out(1) + out(2) + out(3); });
  KernelInstance instance(builder.finish(), &pool.abi);
  instance.set(out, output.data());
  instance.replay();
  EXPECT_EQ(output[0], 6);
  EXPECT_EQ(pool.submissions, 0u);
}

class SharedTestArena final : public WorkspaceArenaProvider {
public:
  struct Storage {
    alignas(64) std::array<std::byte, 8192> bytes;
  };
  std::shared_ptr<Storage> storage = std::make_shared<Storage>();
  unsigned allocations = 0;
  nint_t capacity(WorkspaceTier) const noexcept override {
    return 8192;
  }
  bool shares_arenas_between_plans() const noexcept override {
    return true;
  }
  WorkspaceArena allocate(WorkspaceTier, nint_t bytes, nint_t) override {
    if (!bytes)
      return {};
    ++allocations;
    return {storage->bytes.data(), 8192, storage};
  }
};
TEST(KernelPlanTest, SharedProviderReceivesOneArenaAndDistinctBuffersDoNotOverlap) {
  CountingPool pool;
  KernelBuilder builder(4);
  std::array<int, 4> output{};
  auto out = builder.input(make_tensor(output.data(), make_shape(meta::cint<4>)));
  auto first = builder.tensor<int>("first", make_shape(meta::cint<4>));
  auto second = builder.tensor<int>("second", make_shape(meta::cint<4>));
  builder.single("write", kernel_bind(first, second), [](auto a, auto b) {
    if (a.data() == b.data())
      throw std::runtime_error("overlapping scratch");
    for (int i = 0; i < 4; ++i) {
      a(i) = i;
      b(i) = 10 + i;
    }
  });
  builder.grid<4>(
    "read", meta::cint<0>, meta::cint<4>, meta::cint<1>, kernel_bind(first, second, out),
    [](auto, const auto& item, auto a, auto b, auto out) { out(item.begin()) = a(item.begin()) + b(item.begin()); });
  auto provider = std::make_shared<SharedTestArena>();
  KernelInstance instance(builder.finish(), &pool.abi, provider);
  instance.set(out, output.data());
  instance.replay();
  EXPECT_EQ(provider->allocations, 1u);
  EXPECT_EQ(output, (std::array<int, 4>{10, 12, 14, 16}));
}

class FailingFastArena final : public WorkspaceArenaProvider {
  HeapWorkspaceArenaProvider heap;

public:
  unsigned slow_allocations = 0;
  nint_t capacity(WorkspaceTier) const noexcept override {
    return 8192;
  }
  WorkspaceArena allocate(WorkspaceTier tier, nint_t bytes, nint_t alignment) override {
    if (tier == WorkspaceTier::Fast && bytes)
      throw WorkspaceArenaUnavailable("test unavailable fast tier");
    if (tier == WorkspaceTier::Slow && bytes)
      ++slow_allocations;
    return heap.allocate(tier, bytes, alignment);
  }
};
TEST(KernelPlanTest, PreferredFastStorageFallsBackButRequiredFastFails) {
  CountingPool pool;
  auto provider = std::make_shared<FailingFastArena>();
  KernelBuilder builder(4);
  auto scratch = builder.tensor<int>("preferred", make_shape(meta::cint<4>));
  builder.single("write", kernel_bind(scratch), [](auto scratch) { scratch(0) = 17; });
  KernelInstance instance(builder.finish(), &pool.abi, provider);
  EXPECT_NO_THROW(instance.replay());
  EXPECT_EQ(provider->slow_allocations, 1u);
  KernelBuilder required(4);
  required.tensor<int>("required", make_shape(meta::cint<4>), {.placement = WorkspacePlacementPolicy::FastRequired});
  auto plan = required.finish();
  EXPECT_THROW((KernelInstance{plan, &pool.abi, provider}), WorkspaceArenaUnavailable);
}

TEST(KernelPlanTest, SymbolicLayoutViewsPreserveSlotOffsetsAndConstExtents) {
  CountingPool pool;
  std::array<int, 12> input{}, output{};
  for (int i = 0; i < 12; ++i)
    input[i] = i + 1;
  KernelBuilder builder(4);
  auto source = builder.input(make_tensor(input.data(), make_shape(meta::cint<12>)));
  auto out = builder.input(make_tensor(output.data(), make_shape(meta::cint<12>)));
  auto shifted = source(range(meta::cint<2>, meta::cint<8>));
  auto view = tensor::layout_view(shifted, make_shape(meta::cint<2>, meta::cint<3>));
  auto transposed = tensor::transpose<0, 1>(view);
  builder.grid<4>("copy", meta::cint<0>, meta::cint<3>, meta::cint<1>, kernel_bind(transposed, out),
                  [](auto, const auto& item, auto transposed, auto out) {
                    static_assert(decltype(transposed.template size<0>())::value == 3);
                    out(item.begin()) = transposed(item.begin(), 1);
                  });
  KernelInstance instance(builder.finish(), &pool.abi);
  instance.set(source, input.data());
  instance.set(out, output.data());
  instance.replay();
  EXPECT_EQ(output[0], 6);
  EXPECT_EQ(output[1], 7);
  EXPECT_EQ(output[2], 8);
  std::array<int, 12> replacement{};
  replacement.fill(19);
  instance.set(source, replacement.data());
  instance.replay();
  EXPECT_EQ(output[0], 19);
  EXPECT_EQ(output[1], 19);
  EXPECT_EQ(output[2], 19);
}

} // namespace
