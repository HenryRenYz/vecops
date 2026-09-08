/** @file RuntimeTest.cpp @brief Declarative binding, compilation, caching, and invocation tests. */

#include "vecops/compiler/Compiler.h"
#include "vecops/runtime/Runtime.h"

#include <algorithm>
#include <chrono>
#include <concepts>
#include <cstdlib>
#include <filesystem>
#include <iostream>
#include <memory>
#include <string>
#include <vector>

#ifndef VECOPS_TEST_KERNEL
#  error "VECOPS_TEST_KERNEL must identify TestKernel.cpp"
#endif
#ifndef VECOPS_SOURCE_ROOT
#  error "VECOPS_SOURCE_ROOT must identify the Vecops source root"
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

  vecops::compiler::KernelCompilerConfig compiler_config;
  compiler_config.sdk = {VECOPS_SOURCE_ROOT, vecops::compiler::SdkLayout::SourceTree};
  compiler_config.toolchain.c_compiler = fs::path("/usr/bin/gcc");
  compiler_config.toolchain.cxx_compiler = fs::path("/usr/bin/g++");
  compiler_config.toolchain.parallel_jobs = 2;
  compiler_config.work_directory = work_directory / "builds";
  compiler_config.target_arch = "Scalar";
  auto compiler = std::make_shared<vecops::compiler::Compiler>(std::move(compiler_config));

  auto executable = compiler->compile_kernel(VECOPS_TEST_KERNEL, definition, call);
  require(executable.ok(), executable.status().message());
  require(executable.value()->operator_name() == definition.name(), "generated operator name");
  auto bound = bind_kernel_call(definition, call);
  require(executable.value()->specialization_key() == bound.value().specialization_key, "generated specialization");
  auto bytes = executable.value()->workspace_size(call.arguments());
  require(bytes.ok() && bytes.value() == 0, "kernel-owned workspace contract");
  require(executable.value()->invoke(call.arguments()).ok(), "direct executable invocation");
  require(output == std::vector<float>({4, 7, 10, 13, 16, 19, 22, 25}), "direct result");

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

  const auto absent_cache = work_directory / "must-not-be-created";
  ArtifactExecutableProvider absent_provider(
    {.mode = ArtifactCacheMode::CacheOnly, .cache_directory = absent_cache, .namespace_key = "missing"});
  auto bound_recipe = recipe->bind(call);
  require(bound_recipe.ok() && !absent_provider.resolve(bound_recipe.value()).ok(), "cache-only miss");
  require(!fs::exists(absent_cache), "cache-only miss must not create directories");

  std::cout << "RuntimeTest PASS\n";
  return 0;
}
