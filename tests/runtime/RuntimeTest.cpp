// SPDX-FileCopyrightText: 2026 HenryRenYz and vecops contributors
// SPDX-License-Identifier: MIT
/** @file RuntimeTest.cpp @brief Declarative binding, compilation, caching, and invocation tests. */

#include "vecops/compiler/Compiler.h"
#include "vecops/execution/WorkspaceContext.h"
#include "vecops/runtime/Runtime.h"
#include "WorkspaceContextFixture.h"

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <concepts>
#include <cstdlib>
#include <filesystem>
#include <future>
#include <iostream>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

#ifndef VECOPS_TEST_KERNEL
#  error "VECOPS_TEST_KERNEL must identify TestKernel.cpp"
#endif
#ifndef VECOPS_SOURCE_ROOT
#  error "VECOPS_SOURCE_ROOT must identify the Vecops source root"
#endif
#ifndef VECOPS_WORKSPACE_CONTEXT_FIXTURE
#  error "VECOPS_WORKSPACE_CONTEXT_FIXTURE must identify the ABI fixture"
#endif

using namespace vecops::runtime;
namespace fs = std::filesystem;

namespace {

static_assert(std::same_as<dtype_to_type_t<DType::Bool>, bool>);
static_assert(std::same_as<dtype_to_type_t<DType::Int8>, vecops::int8_t>);
static_assert(std::same_as<dtype_to_type_t<DType::UInt8>, vecops::uint8_t>);
static_assert(std::same_as<dtype_to_type_t<DType::Int16>, vecops::int16_t>);
static_assert(std::same_as<dtype_to_type_t<DType::UInt16>, vecops::uint16_t>);
static_assert(std::same_as<dtype_to_type_t<DType::Int32>, vecops::int32_t>);
static_assert(std::same_as<dtype_to_type_t<DType::UInt32>, vecops::uint32_t>);
static_assert(std::same_as<dtype_to_type_t<DType::Int64>, vecops::int64_t>);
static_assert(std::same_as<dtype_to_type_t<DType::UInt64>, vecops::uint64_t>);
static_assert(std::same_as<dtype_to_type_t<DType::Float16>, vecops::float16_t>);
static_assert(std::same_as<dtype_to_type_t<DType::BFloat16>, vecops::bfloat16_t>);
static_assert(std::same_as<dtype_to_type_t<DType::Float32>, vecops::float32_t>);
static_assert(std::same_as<dtype_to_type_t<DType::Float64>, vecops::float64_t>);

[[noreturn]] void fail(const std::string& message) {
  std::cerr << "RuntimeTest failure: " << message << '\n';
  std::exit(1);
}

void require(bool condition, const std::string& message) {
  if (!condition)
    fail(message);
}

TensorView matrix(float* data, std::int64_t rows, std::int64_t columns, std::int64_t row_stride, bool output) {
  return TensorView{
    .data = data,
    .dtype = DType::Float32,
    .sizes = {rows, columns},
    .strides = {row_stride, 1},
    .flags = output ? VECOPS_TENSOR_WRITE : VECOPS_TENSOR_READ,
  };
}

KernelDef kernel_definition() {
  return KernelDef("test::generated",
                   {
                     TensorDef{"input", {"B", "D"}, {"stride_D", 1}, TensorDTypeDef("IOType")},
                     TensorDef{"output",
                               {"B", "D"},
                               {DimensionDef::dynamic(1, "D", 1048576), 1},
                               TensorDTypeDef("IOType"),
                               false,
                               TensorAccess::Output},
                     ValueDef::typed<double>("factor", 2.0),
                     ValueDef::typed<bool>("add_one", false),
                   },
                   {
                     {"B", SpecializationType::ConstInt},
                     {"ComputeType", SpecializationType::DType},
                     {"D", SpecializationType::ConstInt},
                     {"IOType", SpecializationType::DType},
                     {"stride_D", SpecializationType::ConstInt},
                   });
}

KernelCall kernel_call(float* input, float* output, std::int64_t row_stride = 4) {
  return KernelCall(make_arguments(matrix(input, 2, 4, 4, false), matrix(output, 2, 4, row_stride, true), 3.0, true),
                    {{"ComputeType", DType::Float32}});
}

KernelDef named_dynamic_kernel_definition() {
  const auto batch = DimensionDef::named_dynamic("B", 1, 0, 1024);
  return KernelDef(
    "test::named-dynamic-kernel",
    {
      TensorDef{"input", {batch, "D"}, {"D", 1}, TensorDTypeDef(DType::Float32)},
      TensorDef{"output", {batch, "D"}, {"D", 1}, TensorDTypeDef(DType::Float32), false, TensorAccess::Output},
    },
    {{"D", SpecializationType::ConstInt}});
}

KernelDef any_metadata_kernel_definition() {
  return KernelDef(
    "test::any-metadata-kernel",
    {
      TensorDef{"input", {DimensionDef::any(), DimensionDef::any()}, {DimensionDef::any(), 1},
                TensorDTypeDef(DType::Float32)},
      TensorDef{"output", {DimensionDef::any(), DimensionDef::any()}, {DimensionDef::any(), 1},
                TensorDTypeDef(DType::Float32), false, TensorAccess::Output},
    });
}

void test_binding(const KernelDef& definition, float* input, float* output) {
  auto call = kernel_call(input, output);
  auto bound = bind_kernel_call(definition, call);
  require(bound.ok(), bound.status().message());
  require(std::get<std::int64_t>(bound.value().values.at("B")) == 2, "infer B");
  require(std::get<std::int64_t>(bound.value().values.at("D")) == 4, "infer D");
  require(std::get<std::int64_t>(bound.value().values.at("stride_D")) == 4, "infer stride_D");
  require(std::get<DType>(bound.value().values.at("IOType")) == DType::Float32, "infer IOType");

  auto conflict = KernelCall(call.arguments(), {{"ComputeType", DType::Float32}, {"D", std::int64_t{8}}});
  auto conflict_result = bind_kernel_call(definition, conflict);
  require(!conflict_result.ok() && conflict_result.status().message().find("observed 4") != std::string::npos,
          "explicit symbol conflict");

  auto bad_stride = bind_kernel_call(definition, kernel_call(input, output, 2));
  require(!bad_stride.ok() && bad_stride.status().message().find("Dynamic") != std::string::npos,
          "symbolic Dynamic lower bound");

  auto defaults = KernelCall(make_arguments(matrix(input, 2, 4, 4, false), matrix(output, 2, 4, 4, true)),
                             {{"ComputeType", DType::Float32}});
  auto with_defaults = bind_kernel_call(definition, defaults);
  require(with_defaults.ok() && with_defaults.value().arguments.size() == 4, "append scalar defaults");
  require(std::get<Scalar>(with_defaults.value().arguments[2]).dtype == DType::Float64, "factor default");
  require(std::get<Scalar>(with_defaults.value().arguments[3]).dtype == DType::Bool, "boolean default");

  auto bad_definition = KernelDef("bad", {TensorDef{"x", {DimensionDef::dynamic("IOType", 0, 8)}, {1}}},
                                  {{"IOType", SpecializationType::DType}});
  require(!bad_definition.validate().ok(), "Dynamic symbols must be ConstInt");

  auto metadata_only = bind_kernel_call(definition, kernel_call(nullptr, nullptr));
  require(metadata_only.ok(), "binding must not require tensor storage");

  auto optional_output =
    KernelDef("bad-output", {TensorDef{"out", {4}, {1}, TensorDTypeDef(DType::Float32), true, TensorAccess::Output}});
  require(!optional_output.validate().ok(), "writable tensors cannot be optional");
}

void test_named_dynamic_binding(float* input, float* output) {
  const auto batch = DimensionDef::named_dynamic("B", 1, 0, 1024);
  const KernelDef definition(
    "test::named-dynamic",
    {
      TensorDef{"input", {batch, 4}, {4, 1}, TensorDTypeDef(DType::Float32)},
      TensorDef{"output", {batch, 4}, {4, 1}, TensorDTypeDef(DType::Float32), false, TensorAccess::Output},
    });
  require(definition.validate().ok(), "named Dynamic definition validation");

  auto first = bind_kernel_call(
    definition, KernelCall(make_arguments(matrix(input, 2, 4, 4, false), matrix(output, 2, 4, 4, true))));
  require(first.ok(), first.status().message());
  require(!first.value().values.contains("B"), "named Dynamic must not become a specialization value");

  auto second = bind_kernel_call(
    definition, KernelCall(make_arguments(matrix(input, 3, 4, 4, false), matrix(output, 3, 4, 4, true))));
  require(second.ok(), second.status().message());
  require(first.value().specialization_key == second.value().specialization_key,
          "named Dynamic runtime values must not change the specialization key");

  auto mismatch = bind_kernel_call(
    definition, KernelCall(make_arguments(matrix(input, 2, 4, 4, false), matrix(output, 3, 4, 4, true))));
  require(!mismatch.ok() && mismatch.status().message().find("runtime dimension 'B'") != std::string::npos,
          "named Dynamic occurrences must agree");

  const KernelDef conflicting("test::conflicting-dynamic",
                              {
                                TensorDef{"x", {DimensionDef::named_dynamic("B", 1, 0, 8)}, {1}},
                                TensorDef{"y", {DimensionDef::named_dynamic("B", 2, 0, 8)}, {1}},
                              });
  require(!conflicting.validate().ok(), "named Dynamic declarations must be consistent");
}

class WrongProvider final : public ExecutableProvider {
public:
  explicit WrongProvider(std::shared_ptr<Executable> executable)
    : executable_(std::move(executable)) {
  }

  Result<std::shared_ptr<Executable>> resolve(const BoundKernelRecipe&) override {
    return executable_;
  }

private:
  std::shared_ptr<Executable> executable_;
};

class WorkspaceFixtureRecipe final : public KernelRecipe {
public:
  [[nodiscard]] std::string_view id() const override {
    return "workspace-context-fixture";
  }

  [[nodiscard]] Status match(const KernelCall& call) const override {
    if (!call.arguments().values().empty())
      return Status(StatusCode::NotApplicable, "workspace fixture accepts no arguments");
    return Status::success();
  }

  [[nodiscard]] Result<BoundKernelRecipe> bind(const KernelCall& call) const override {
    auto status = match(call);
    if (!status.ok())
      return status;
    return BoundKernelRecipe{
      .recipe_id = "workspace-context-fixture",
      .specialization_key = "workspace-context-v1",
      .artifact_key = "workspace-context-fixture-v1",
    };
  }
};

class WorkspaceFixtureProvider final : public ExecutableProvider {
public:
  explicit WorkspaceFixtureProvider(std::shared_ptr<Executable> executable)
    : executable_(std::move(executable)) {
  }

  Result<std::shared_ptr<Executable>> resolve(const BoundKernelRecipe&) override {
    return executable_;
  }

private:
  std::shared_ptr<Executable> executable_;
};

struct RuntimeArenaProvider {
  RuntimeArenaProvider() {
    static std::atomic<std::uint64_t> next_identity{1};
    abi = {sizeof(VecopsWorkspaceArenaProvider),
           0,
           next_identity.fetch_add(1),
           this,
           capacity,
           allocate,
           release_arena,
           retain,
           release_context,
           0};
  }

  static std::uint64_t capacity(void*, std::uint32_t) {
    return 4096;
  }

  static std::int32_t allocate(void* context, std::uint32_t tier, std::uint64_t bytes, std::uint64_t alignment,
                               VecopsWorkspaceArena* result, VecopsError*) {
    if (bytes == 0) {
      *result = {sizeof(VecopsWorkspaceArena), 0, nullptr, 0, nullptr};
      return VECOPS_STATUS_OK;
    }
    auto& self = *static_cast<RuntimeArenaProvider*>(context);
    const auto logical_tier = tier == VECOPS_WORKSPACE_TIER_FAST ? vecops::execution::WorkspaceTier::Fast
                                                                 : vecops::execution::WorkspaceTier::Slow;
    auto* owner = new vecops::execution::WorkspaceArena(
      self.heap.allocate(logical_tier, static_cast<vecops::nint_t>(bytes), static_cast<vecops::nint_t>(alignment)));
    *result = {sizeof(VecopsWorkspaceArena), 0, owner->data, static_cast<std::uint64_t>(owner->capacity), owner};
    self.allocations.fetch_add(1);
    return VECOPS_STATUS_OK;
  }

  static void release_arena(void*, void* owner) {
    delete static_cast<vecops::execution::WorkspaceArena*>(owner);
  }

  static void retain(void* context) {
    static_cast<RuntimeArenaProvider*>(context)->references.fetch_add(1);
  }

  static void release_context(void* context) {
    auto* self = static_cast<RuntimeArenaProvider*>(context);
    if (self->references.fetch_sub(1) == 1)
      delete self;
  }

  std::atomic<unsigned> references{1};
  std::atomic<unsigned> allocations{0};
  vecops::execution::HeapWorkspaceArenaProvider heap;
  VecopsWorkspaceArenaProvider abi{};
};

void test_kernel_call_workspace_and_context_forwarding() {
  auto executable = Executable::load(VECOPS_WORKSPACE_CONTEXT_FIXTURE);
  require(executable.ok(), executable.status().message());

  auto recipe = std::make_shared<WorkspaceFixtureRecipe>();
  auto provider = std::make_shared<WorkspaceFixtureProvider>(executable.value());
  Operator operation(OperatorSchema("test::workspace-context", {}), {std::move(recipe)},
                     std::make_shared<OrderedDispatchPolicy>(), std::move(provider));

  alignas(64) std::array<std::byte, vecops::test::WorkspaceContextFixtureBytes> workspace{};
  vecops::test::WorkspaceContextObservation observation{};
  auto* const stream = reinterpret_cast<void*>(std::uintptr_t{0x1234});
  const VecopsExecutionContext context{
    .struct_size = sizeof(VecopsExecutionContext),
    .requested_threads = 7,
    .stream = stream,
    .user_data = &observation,
    .flags = UINT64_C(0xabc),
  };
  const KernelCall call(ArgumentMetadata{});

  const auto status = operation.invoke(call, &context, workspace.data(), workspace.size());
  require(status.ok(), status.message());
  require(observation.workspace_query_calls == 1, "workspace query must receive the forwarded context");
  require(observation.run_calls == 1, "workspace fixture must execute exactly once");
  require(observation.query_context == &context && observation.run_context == &context,
          "query and run must receive the original context pointer");
  require(observation.run_workspace == workspace.data() && observation.run_workspace_size == workspace.size(),
          "run must receive the original workspace allocation");
  require(observation.requested_threads == context.requested_threads && observation.stream == stream &&
            observation.context_flags == context.flags,
          "execution-context fields must survive Operator dispatch");
  require(workspace.front() == std::byte{0x5a}, "fixture must be able to write the forwarded workspace");

  vecops::test::WorkspaceContextObservation legacy_observation{};
  const VecopsExecutionContext legacy_context{
    .struct_size = sizeof(VecopsExecutionContext),
    .requested_threads = 1,
    .stream = nullptr,
    .user_data = &legacy_observation,
    .flags = 0,
  };
  const auto legacy = operation.invoke(call, &legacy_context);
  require(!legacy.ok() && legacy_observation.workspace_query_calls == 1 && legacy_observation.run_calls == 0,
          "legacy KernelCall overload must retain null-workspace behavior");
}

} // namespace

int main(int argc, char** argv) {
  static_assert(dtype_bit(static_cast<DType>(100)) == 0);
  if (argc != 2)
    fail("usage: RuntimeTest <work-directory>");
  const auto unique = std::to_string(std::chrono::steady_clock::now().time_since_epoch().count());
  const fs::path work_directory = fs::path(argv[1]) / unique;

  auto definition = kernel_definition();
  require(definition.validate().ok(), "kernel definition validation");
  std::vector<float> input{1, 2, 3, 4, 5, 6, 7, 8};
  std::vector<float> output(8);
  auto call = kernel_call(input.data(), output.data());
  test_binding(definition, input.data(), output.data());
  test_named_dynamic_binding(input.data(), output.data());
  test_kernel_call_workspace_and_context_forwarding();

  vecops::compiler::KernelCompilerConfig compiler_config;
  compiler_config.sdk = {VECOPS_SOURCE_ROOT, vecops::compiler::SdkLayout::SourceTree};
  compiler_config.toolchain.c_compiler = fs::path("/usr/bin/gcc");
  compiler_config.toolchain.cxx_compiler = fs::path("/usr/bin/g++");
  compiler_config.toolchain.parallel_jobs = 2;
  compiler_config.work_directory = work_directory / "builds";
  compiler_config.target_arch = "Scalar";
  vecops::compiler::Compiler equivalent_compiler(compiler_config);
  vecops::compiler::Compiler other_equivalent_compiler(compiler_config);
  require(equivalent_compiler.batch_key() == other_equivalent_compiler.batch_key(),
          "equivalent compiler instances must share a batch key");
  auto different_config = compiler_config;
  different_config.work_directory = work_directory / "other-builds";
  require(equivalent_compiler.batch_key() != vecops::compiler::Compiler(different_config).batch_key(),
          "different compiler work roots must split batches");
  auto compiler = std::make_shared<vecops::compiler::Compiler>(std::move(compiler_config));

  auto executable = compiler->compile_kernel(VECOPS_TEST_KERNEL, definition, call);
  require(executable.ok(), executable.status().message());
  require(executable.value()->operator_name() == definition.name(), "generated operator name");
  auto bound = bind_kernel_call(definition, call);
  require(executable.value()->specialization_key() == bound.value().specialization_key + ";$Parallelism=i:1",
          "generated specialization");
  auto bytes = executable.value()->workspace_size(call.arguments());
  require(bytes.ok() && bytes.value() == 0, "kernel-owned workspace contract");
  require(executable.value()->invoke(call.arguments()).ok(), "direct executable invocation");
  require(output == std::vector<float>({4, 7, 10, 13, 16, 19, 22, 25}), "direct result");

  auto* arena_provider = new RuntimeArenaProvider();
  const VecopsExecutionContext arena_context{
    .struct_size = sizeof(VecopsExecutionContext),
    .requested_threads = 0,
    .stream = nullptr,
    .user_data = nullptr,
    .flags = 0,
    .workspace_provider = &arena_provider->abi,
  };
  std::fill(output.begin(), output.end(), 0);
  require(executable.value()->invoke(call.arguments(), nullptr, 0, &arena_context).ok(),
          "source kernel arena-provider invocation");
  require(executable.value()->invoke(call.arguments(), nullptr, 0, &arena_context).ok(),
          "source kernel arena-provider replay");
  require(arena_provider->allocations.load() == 1, "arena provider must allocate once and replay from its cache");
  RuntimeArenaProvider::release_context(arena_provider);

  auto dynamic_definition = named_dynamic_kernel_definition();
  std::vector<float> dynamic_input(12);
  std::vector<float> dynamic_output(12);
  for (std::size_t index = 0; index < dynamic_input.size(); ++index)
    dynamic_input[index] = static_cast<float>(index);
  auto dynamic_compile_call = KernelCall(
    make_arguments(matrix(dynamic_input.data(), 2, 4, 4, false), matrix(dynamic_output.data(), 2, 4, 4, true)));
  const auto dynamic_kernel = fs::path(VECOPS_TEST_KERNEL).parent_path() / "NamedDynamicKernel.cpp";
  auto dynamic_executable = compiler->compile_kernel(dynamic_kernel, dynamic_definition, dynamic_compile_call);
  require(dynamic_executable.ok(), dynamic_executable.status().message());
  auto changed_batch_call = KernelCall(
    make_arguments(matrix(dynamic_input.data(), 3, 4, 4, false), matrix(dynamic_output.data(), 3, 4, 4, true)));
  require(dynamic_executable.value()->invoke(changed_batch_call.arguments()).ok(),
          "generated adapter must read Dynamic values from each call");
  for (std::size_t index = 0; index < dynamic_output.size(); ++index)
    require(dynamic_output[index] == dynamic_input[index] + 1.0f, "named Dynamic changed-batch result");
  auto mismatched_batch_call = KernelCall(
    make_arguments(matrix(dynamic_input.data(), 2, 4, 4, false), matrix(dynamic_output.data(), 3, 4, 4, true)));
  require(!dynamic_executable.value()->invoke(mismatched_batch_call.arguments()).ok(),
          "generated adapter must enforce named Dynamic relations");

  auto any_definition = any_metadata_kernel_definition();
  std::vector<float> any_input(21, -101.0f);
  std::vector<float> any_output(21, -202.0f);
  for (std::int64_t row = 0; row < 2; ++row)
    for (std::int64_t column = 0; column < 4; ++column)
      any_input[static_cast<std::size_t>(row * 5 + column)] = static_cast<float>(row * 10 + column);
  auto any_compile_call = KernelCall(
    make_arguments(matrix(any_input.data(), 2, 4, 5, false), matrix(any_output.data(), 2, 4, 5, true)));
  const auto any_kernel = fs::path(VECOPS_TEST_KERNEL).parent_path() / "AnyMetadataKernel.cpp";
  auto any_executable = compiler->compile_kernel(any_kernel, any_definition, any_compile_call);
  require(any_executable.ok(), any_executable.status().message());

  std::fill(any_output.begin(), any_output.end(), -202.0f);
  for (std::int64_t row = 0; row < 3; ++row)
    for (std::int64_t column = 0; column < 3; ++column)
      any_input[static_cast<std::size_t>(row * 7 + column)] = static_cast<float>(row * 10 + column);
  auto changed_any_call = KernelCall(
    make_arguments(matrix(any_input.data(), 3, 3, 7, false), matrix(any_output.data(), 3, 3, 7, true)));
  require(any_executable.value()->invoke(changed_any_call.arguments()).ok(),
          "generated adapter must read unconstrained Any metadata from each call");
  for (std::int64_t row = 0; row < 3; ++row) {
    for (std::int64_t column = 0; column < 3; ++column)
      require(any_output[static_cast<std::size_t>(row * 7 + column)] ==
                any_input[static_cast<std::size_t>(row * 7 + column)] + 1.0f,
              "unconstrained Any changed-layout result");
    for (std::int64_t column = 3; column < 7; ++column)
      require(any_output[static_cast<std::size_t>(row * 7 + column)] == -202.0f,
              "unconstrained Any changed-layout padding");
  }

  std::vector<float> wider_input(16);
  std::vector<float> wider_output(16);
  auto other_shape = KernelCall(
    make_arguments(matrix(wider_input.data(), 2, 8, 8, false), matrix(wider_output.data(), 2, 8, 8, true), 3.0, true),
    {{"ComputeType", DType::Float32}});
  require(!executable.value()->can_invoke(other_shape.arguments()).ok(), "specialization descriptor guard");

  auto shared_definition = std::make_shared<KernelDef>(definition);
  auto recipe = std::make_shared<SourceKernelRecipe>("test-kernel-v1", VECOPS_TEST_KERNEL, shared_definition);
  int builds = 0;
  ArtifactProviderConfig read_write;
  read_write.mode = ArtifactCacheMode::ReadWrite;
  read_write.cache_directory = work_directory / "cache";
  read_write.namespace_key = "scalar-test-toolchain-v1";
  read_write.build = [&](const BoundKernelRecipe&) -> Result<std::shared_ptr<Executable>> {
    ++builds;
    return executable.value();
  };
  auto provider = std::make_shared<ArtifactExecutableProvider>(std::move(read_write));
  Operator op(definition, {recipe}, std::make_shared<OrderedDispatchPolicy>(), provider);
  std::fill(output.begin(), output.end(), 0);
  require(op(call).ok(), "Operator read-write invocation");
  require(builds == 1, "read-write provider build count");

  ArtifactProviderConfig cache_only;
  cache_only.mode = ArtifactCacheMode::CacheOnly;
  cache_only.cache_directory = work_directory / "cache";
  cache_only.namespace_key = "scalar-test-toolchain-v1";
  cache_only.build = [&](const BoundKernelRecipe&) -> Result<std::shared_ptr<Executable>> {
    ++builds;
    return Status(StatusCode::InternalError, "cache-only called its builder");
  };
  auto cache_provider = std::make_shared<ArtifactExecutableProvider>(std::move(cache_only));
  Operator cached_op(definition, {recipe}, std::make_shared<OrderedDispatchPolicy>(), cache_provider);
  std::fill(output.begin(), output.end(), 0);
  require(cached_op(call).ok(), "cache-only Operator invocation");
  require(builds == 1, "cache-only must not call builder");
  require(output == std::vector<float>({4, 7, 10, 13, 16, 19, 22, 25}), "cache-only result");

  auto wrong_call = other_shape;
  Operator wrong_provider_op(definition, {recipe}, std::make_shared<OrderedDispatchPolicy>(),
                             std::make_shared<WrongProvider>(executable.value()));
  auto wrong = wrong_provider_op.resolve(wrong_call);
  require(!wrong.ok() && wrong.status().code() == StatusCode::AbiMismatch, "provider specialization guard");

  auto first_bound = recipe->bind(call);
  require(first_bound.ok(), "bind provider concurrency request");
  auto second_bound = first_bound.value();
  second_bound.artifact_key += ";independent-key";
  std::mutex build_mutex;
  std::condition_variable build_ready;
  int concurrent_builds = 0;
  bool concurrency_timeout = false;
  ArtifactProviderConfig concurrent_config;
  concurrent_config.mode = ArtifactCacheMode::CompileOnly;
  concurrent_config.build = [&](const BoundKernelRecipe&) -> Result<std::shared_ptr<Executable>> {
    std::unique_lock lock(build_mutex);
    ++concurrent_builds;
    build_ready.notify_all();
    if (!build_ready.wait_for(lock, std::chrono::seconds(2), [&] { return concurrent_builds == 2; }))
      concurrency_timeout = true;
    return Status(StatusCode::NotFound, "intentional concurrent-build probe");
  };
  ArtifactExecutableProvider concurrent_provider(std::move(concurrent_config));
  auto first_future = std::async(std::launch::async, [&] { return concurrent_provider.resolve(first_bound.value()); });
  auto second_future = std::async(std::launch::async, [&] { return concurrent_provider.resolve(second_bound); });
  require(!first_future.get().ok() && !second_future.get().ok(), "concurrent provider probe results");
  require(concurrent_builds == 2 && !concurrency_timeout,
          "different provider keys must enter the build callback concurrently");

  std::atomic<int> coalesced_builds{0};
  ArtifactProviderConfig coalesced_config;
  coalesced_config.mode = ArtifactCacheMode::CompileOnly;
  coalesced_config.build = [&](const BoundKernelRecipe&) -> Result<std::shared_ptr<Executable>> {
    ++coalesced_builds;
    std::this_thread::sleep_for(std::chrono::milliseconds(50));
    return executable.value();
  };
  ArtifactExecutableProvider coalesced_provider(std::move(coalesced_config));
  auto coalesced_first =
    std::async(std::launch::async, [&] { return coalesced_provider.resolve(first_bound.value()); });
  auto coalesced_second =
    std::async(std::launch::async, [&] { return coalesced_provider.resolve(first_bound.value()); });
  require(coalesced_first.get().ok() && coalesced_second.get().ok(), "coalesced provider probe results");
  require(coalesced_builds.load() == 1, "identical provider keys must share one in-flight build");

  ArtifactExecutableProvider batch_provider({.mode = ArtifactCacheMode::CompileOnly});
  auto batch_miss = batch_provider.lookup(first_bound.value());
  require(!batch_miss.ok() && batch_miss.status().code() == StatusCode::NotFound,
          "batch lookup must not compile a miss");
  auto batch_adopted = batch_provider.adopt(first_bound.value(), executable.value());
  require(batch_adopted.ok(), "batch provider adopt");
  require(batch_provider.lookup(first_bound.value()).ok(), "batch provider lookup after adopt");

  const auto absent_cache = work_directory / "must-not-be-created";
  ArtifactExecutableProvider absent_provider(
    {.mode = ArtifactCacheMode::CacheOnly, .cache_directory = absent_cache, .namespace_key = "missing"});
  auto bound_recipe = recipe->bind(call);
  require(bound_recipe.ok() && !absent_provider.resolve(bound_recipe.value()).ok(), "cache-only miss");
  require(!fs::exists(absent_cache), "cache-only miss must not create directories");

  std::cout << "RuntimeTest PASS\n";
  return 0;
}
