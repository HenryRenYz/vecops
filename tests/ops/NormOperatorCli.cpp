// SPDX-FileCopyrightText: 2026 HenryRenYz and vecops contributors
// SPDX-License-Identifier: MIT
/** @file NormOperatorCli.cpp @brief New-API correctness and latency probe. */

#include "vecops/compiler/Compiler.h"
#include "vecops/runtime/Provider.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <limits>
#include <memory>
#include <numeric>
#include <optional>
#include <random>
#include <stdexcept>
#include <string>
#include <string_view>
#include <type_traits>
#include <vector>

#ifndef VECOPS_NORM_SDK_SOURCE_DIR
#  error "VECOPS_NORM_SDK_SOURCE_DIR must name the Vecops source tree"
#endif
#ifndef VECOPS_NORM_KERNEL_SOURCE
#  error "VECOPS_NORM_KERNEL_SOURCE must name NormOperatorKernel.cpp"
#endif

namespace {

namespace fs = std::filesystem;
namespace rt = vecops::runtime;
using Clock = std::chrono::steady_clock;
using vecops::nint_t;

enum class OperatorKind { LayerNorm = 1, Softmax = 2 };

struct Options {
  OperatorKind op = OperatorKind::LayerNorm;
  rt::DType input_dtype = rt::DType::Float32;
  rt::DType output_dtype = rt::DType::Float32;
  rt::DType compute_dtype = rt::DType::Float32;
  rt::ArtifactCacheMode cache_mode = rt::ArtifactCacheMode::ReadWrite;
  std::vector<nint_t> shape{32, 1024};
  std::size_t warmup = 5;
  std::size_t iterations = 50;
  fs::path build_work_dir = ".vecops-build";
  fs::path cache_dir = ".vecops-cache";
  fs::path cmake_program = "cmake";
  std::optional<fs::path> c_compiler;
  std::optional<fs::path> cxx_compiler;
  std::string target_arch = "Native";
  std::uint64_t seed = 20260906;
  double epsilon = 1e-5;
};

struct AccuracyResult {
  bool passed = true;
  std::size_t mismatches = 0;
  std::size_t worst_index = 0;
  double max_abs_error = 0;
  double max_rel_error = 0;
  double expected_at_worst = 0;
  double actual_at_worst = 0;
};

struct RunResult {
  double first_ms = 0;
  double mean_ms = 0;
  double stddev_ms = 0;
  double atol = 0;
  double rtol = 0;
  AccuracyResult accuracy;
};

[[noreturn]] void usage_error(const std::string& message) {
  throw std::invalid_argument(message + "\nUse --help to see valid options.");
}

std::string_view op_name(OperatorKind op) {
  return op == OperatorKind::LayerNorm ? "layernorm" : "softmax";
}

std::string_view dtype_name(rt::DType dtype) {
  switch (dtype) {
  case rt::DType::Float16:
    return "float16";
  case rt::DType::BFloat16:
    return "bfloat16";
  case rt::DType::Float32:
    return "float32";
  case rt::DType::Float64:
    return "float64";
  default:
    return "unsupported";
  }
}

std::string_view cache_mode_name(rt::ArtifactCacheMode mode) {
  switch (mode) {
  case rt::ArtifactCacheMode::CacheOnly:
    return "cache-only";
  case rt::ArtifactCacheMode::ReadWrite:
    return "read-write";
  case rt::ArtifactCacheMode::CompileOnly:
    return "compile-only";
  }
  return "unknown";
}

rt::DType parse_dtype(std::string_view text) {
  if (text == "float16" || text == "fp16" || text == "f16")
    return rt::DType::Float16;
  if (text == "bfloat16" || text == "bf16")
    return rt::DType::BFloat16;
  if (text == "float32" || text == "fp32" || text == "f32")
    return rt::DType::Float32;
  if (text == "float64" || text == "fp64" || text == "f64")
    return rt::DType::Float64;
  usage_error("unsupported dtype: " + std::string(text));
}

std::size_t parse_count(std::string_view text, std::string_view name, bool allow_zero) {
  std::size_t consumed = 0;
  unsigned long long value = 0;
  try {
    value = std::stoull(std::string(text), &consumed);
  } catch (...) {
    usage_error("invalid value for --" + std::string(name));
  }
  if (consumed != text.size() || (!allow_zero && value == 0) || value > std::numeric_limits<std::size_t>::max())
    usage_error("invalid value for --" + std::string(name));
  return static_cast<std::size_t>(value);
}

std::vector<nint_t> parse_shape(std::string_view text) {
  std::vector<nint_t> shape;
  for (std::size_t begin = 0; begin <= text.size();) {
    const auto comma = text.find(',', begin);
    const auto end = comma == std::string_view::npos ? text.size() : comma;
    const auto value = parse_count(text.substr(begin, end - begin), "shape", false);
    if (value > static_cast<std::size_t>(std::numeric_limits<nint_t>::max()))
      usage_error("shape dimension is too large");
    shape.push_back(static_cast<nint_t>(value));
    if (comma == std::string_view::npos)
      break;
    begin = comma + 1;
  }
  if (shape.empty())
    usage_error("--shape must not be empty");
  return shape;
}

void print_help(const char* argv0) {
  std::cout << "Usage: " << argv0 << " [options]\n\n"
            << "  --op layernorm|softmax\n"
            << "  --input-dtype float16|bfloat16|float32|float64\n"
            << "  --output-dtype TYPE\n"
            << "  --accumulation-dtype float32|float64\n"
            << "  --shape D0,D1,...,DN\n"
            << "  --warmup N --iterations N\n"
            << "  --cache-mode cache-only|read-write|compile-only\n"
            << "  --build-work-dir PATH --cache-dir PATH\n"
            << "  --cmake PATH --c-compiler PATH --cxx-compiler PATH\n"
            << "  --target-arch ARCH --epsilon VALUE --seed N\n";
}

Options parse_options(int argc, char** argv) {
  Options options;
  for (int index = 1; index < argc; ++index) {
    std::string_view argument(argv[index]);
    if (argument == "--help" || argument == "-h") {
      print_help(argv[0]);
      std::exit(0);
    }
    if (!argument.starts_with("--"))
      usage_error("unexpected positional argument");
    const auto equal = argument.find('=');
    const auto name = argument.substr(2, equal == std::string_view::npos ? equal : equal - 2);
    std::string_view value;
    if (equal != std::string_view::npos)
      value = argument.substr(equal + 1);
    else if (++index < argc)
      value = argv[index];
    else
      usage_error("missing value for --" + std::string(name));

    if (name == "op") {
      if (value == "layernorm" || value == "layer_norm")
        options.op = OperatorKind::LayerNorm;
      else if (value == "softmax")
        options.op = OperatorKind::Softmax;
      else
        usage_error("unsupported operator");
    } else if (name == "input-dtype")
      options.input_dtype = parse_dtype(value);
    else if (name == "output-dtype")
      options.output_dtype = parse_dtype(value);
    else if (name == "accumulation-dtype" || name == "accum-dtype")
      options.compute_dtype = parse_dtype(value);
    else if (name == "shape")
      options.shape = parse_shape(value);
    else if (name == "warmup")
      options.warmup = parse_count(value, name, true);
    else if (name == "iterations")
      options.iterations = parse_count(value, name, false);
    else if (name == "build-work-dir")
      options.build_work_dir = value;
    else if (name == "cache-dir")
      options.cache_dir = value;
    else if (name == "cmake")
      options.cmake_program = value;
    else if (name == "c-compiler")
      options.c_compiler = fs::path(value);
    else if (name == "cxx-compiler")
      options.cxx_compiler = fs::path(value);
    else if (name == "target-arch")
      options.target_arch = value;
    else if (name == "seed")
      options.seed = parse_count(value, name, true);
    else if (name == "epsilon") {
      std::size_t consumed = 0;
      try {
        options.epsilon = std::stod(std::string(value), &consumed);
      } catch (...) {
        usage_error("invalid --epsilon");
      }
      if (consumed != value.size() || !std::isfinite(options.epsilon) || options.epsilon <= 0)
        usage_error("--epsilon must be finite and positive");
    } else if (name == "cache-mode") {
      if (value == "cache-only")
        options.cache_mode = rt::ArtifactCacheMode::CacheOnly;
      else if (value == "read-write")
        options.cache_mode = rt::ArtifactCacheMode::ReadWrite;
      else if (value == "compile-only")
        options.cache_mode = rt::ArtifactCacheMode::CompileOnly;
      else
        usage_error("unsupported --cache-mode");
    } else
      usage_error("unknown option: --" + std::string(name));
  }
  if (options.compute_dtype != rt::DType::Float32 && options.compute_dtype != rt::DType::Float64)
    usage_error("--accumulation-dtype must be float32 or float64");
  return options;
}

std::size_t checked_numel(const std::vector<nint_t>& shape) {
  std::size_t count = 1;
  for (auto dimension : shape) {
    if (count > static_cast<std::size_t>(std::numeric_limits<nint_t>::max()) / dimension)
      usage_error("shape element count exceeds vecops nint_t range");
    count *= static_cast<std::size_t>(dimension);
  }
  return count;
}

template <typename T>
std::vector<T> convert(const std::vector<double>& source) {
  std::vector<T> result(source.size());
  std::transform(source.begin(), source.end(), result.begin(), [](double value) { return static_cast<T>(value); });
  return result;
}

rt::TensorView tensor_view(void* data, rt::DType dtype, std::size_t rows, std::size_t columns, bool output) {
  return {.data = data,
          .dtype = dtype,
          .sizes = {static_cast<std::int64_t>(rows), static_cast<std::int64_t>(columns)},
          .strides = {static_cast<std::int64_t>(columns), 1},
          .flags = output ? VECOPS_TENSOR_WRITE : VECOPS_TENSOR_READ};
}

rt::TensorView vector_view(void* data, rt::DType dtype, std::size_t count) {
  return {.data = data,
          .dtype = dtype,
          .sizes = {static_cast<std::int64_t>(count)},
          .strides = {1},
          .flags = VECOPS_TENSOR_READ};
}

rt::KernelDef kernel_definition() {
  return rt::KernelDef(
    "vecops::norm",
    {
      rt::TensorDef{"input", {"B", "D"}, {"D", 1}, rt::TensorDTypeDef("InputType")},
      rt::TensorDef{"scale", {"D"}, {1}, rt::TensorDTypeDef("InputType"), true},
      rt::TensorDef{"bias", {"D"}, {1}, rt::TensorDTypeDef("InputType"), true},
      rt::TensorDef{"output", {"B", "D"}, {"D", 1}, rt::TensorDTypeDef("OutputType"), false, rt::TensorAccess::Output},
      rt::ValueDef::typed<double>("eps", 1e-5),
    },
    {{"B", rt::SpecializationType::ConstInt},
     {"ComputeType", rt::SpecializationType::DType},
     {"D", rt::SpecializationType::ConstInt},
     {"InputType", rt::SpecializationType::DType},
     {"Op", rt::SpecializationType::ConstInt},
     {"OutputType", rt::SpecializationType::DType}});
}

std::shared_ptr<rt::Operator> make_operator(const Options& options, rt::KernelDef definition) {
  vecops::compiler::KernelCompilerConfig config;
  config.sdk = {VECOPS_NORM_SDK_SOURCE_DIR, vecops::compiler::SdkLayout::SourceTree};
  config.toolchain.cmake_program = options.cmake_program;
  config.toolchain.c_compiler = options.c_compiler;
  config.toolchain.cxx_compiler = options.cxx_compiler;
  config.toolchain.build_type = "Release";
  config.work_directory = options.build_work_dir;
  config.target_arch = options.target_arch;
  auto compiler = std::make_shared<vecops::compiler::Compiler>(std::move(config));
  auto shared_definition = std::make_shared<rt::KernelDef>(definition);
  auto recipe =
    std::make_shared<rt::SourceKernelRecipe>("norm-kernel-v2", VECOPS_NORM_KERNEL_SOURCE, shared_definition);
  rt::ArtifactProviderConfig provider_config;
  provider_config.mode = options.cache_mode;
  provider_config.cache_directory = options.cache_dir;
  provider_config.namespace_key = "norm-cli-v2;arch=" + options.target_arch;
  provider_config.build = [compiler](const rt::BoundKernelRecipe& bound) { return compiler->compile_kernel(bound); };
  auto provider = std::make_shared<rt::ArtifactExecutableProvider>(std::move(provider_config));
  return std::make_shared<rt::Operator>(std::move(definition),
                                        std::vector<std::shared_ptr<const rt::KernelRecipe>>{recipe},
                                        std::make_shared<rt::OrderedDispatchPolicy>(), std::move(provider));
}

AccuracyResult compare(const std::vector<double>& expected, const std::vector<double>& actual, double atol,
                       double rtol) {
  AccuracyResult result;
  for (std::size_t index = 0; index < expected.size(); ++index) {
    const double abs_error = std::abs(actual[index] - expected[index]);
    const double rel_error = abs_error / std::max(std::abs(expected[index]), 1e-30);
    if (abs_error > result.max_abs_error) {
      result.max_abs_error = abs_error;
      result.worst_index = index;
      result.expected_at_worst = expected[index];
      result.actual_at_worst = actual[index];
    }
    result.max_rel_error = std::max(result.max_rel_error, rel_error);
    if (!std::isfinite(actual[index]) || abs_error > atol + rtol * std::abs(expected[index])) {
      result.passed = false;
      ++result.mismatches;
    }
  }
  return result;
}

template <typename Output, typename Compute>
std::pair<double, double> tolerances(OperatorKind op) {
  if constexpr (std::is_same_v<Output, vecops::float16_t>)
    return op == OperatorKind::LayerNorm ? std::pair{3e-2, 3e-2} : std::pair{3e-3, 3e-3};
  if constexpr (std::is_same_v<Output, vecops::bfloat16_t>)
    return op == OperatorKind::LayerNorm ? std::pair{7e-2, 7e-2} : std::pair{1e-2, 1e-2};
  if constexpr (std::is_same_v<Output, vecops::float32_t> || std::is_same_v<Compute, vecops::float32_t>)
    return op == OperatorKind::LayerNorm ? std::pair{2e-4, 2e-4} : std::pair{5e-5, 5e-5};
  return op == OperatorKind::LayerNorm ? std::pair{2e-10, 2e-10} : std::pair{2e-12, 2e-12};
}

template <typename Input, typename Output, typename Compute>
RunResult run_typed(const Options& options, std::size_t rows, std::size_t columns) {
  const auto count = rows * columns;
  std::mt19937_64 random(options.seed);
  std::uniform_real_distribution<double> input_distribution(-3, 3);
  std::uniform_real_distribution<double> scale_distribution(.6, 1.4);
  std::uniform_real_distribution<double> bias_distribution(-.3, .3);
  std::vector<double> raw_input(count), raw_scale(columns), raw_bias(columns);
  std::generate(raw_input.begin(), raw_input.end(), [&] { return input_distribution(random); });
  std::generate(raw_scale.begin(), raw_scale.end(), [&] { return scale_distribution(random); });
  std::generate(raw_bias.begin(), raw_bias.end(), [&] { return bias_distribution(random); });
  auto input = convert<Input>(raw_input);
  auto scale = convert<Input>(raw_scale);
  auto bias = convert<Input>(raw_bias);
  std::vector<Output> output(count);

  const auto input_view = tensor_view(input.data(), options.input_dtype, rows, columns, false);
  const auto output_view = tensor_view(output.data(), options.output_dtype, rows, columns, true);
  rt::ArgumentMetadata arguments;
  if (options.op == OperatorKind::LayerNorm)
    arguments =
      rt::make_arguments(input_view, vector_view(scale.data(), options.input_dtype, columns),
                         vector_view(bias.data(), options.input_dtype, columns), output_view, options.epsilon);
  else
    arguments = rt::make_arguments(input_view, std::monostate{}, std::monostate{}, output_view, options.epsilon);
  rt::KernelCall call(std::move(arguments),
                      {{"ComputeType", options.compute_dtype}, {"Op", static_cast<std::int64_t>(options.op)}});
  auto operation = make_operator(options, kernel_definition());

  RunResult result;
  auto begin = Clock::now();
  auto status = (*operation)(call);
  result.first_ms = std::chrono::duration<double, std::milli>(Clock::now() - begin).count();
  if (!status.ok())
    throw std::runtime_error(status.message());
  for (std::size_t index = 0; index < options.warmup; ++index) {
    status = (*operation)(call);
    if (!status.ok())
      throw std::runtime_error(status.message());
  }
  std::vector<double> durations;
  for (std::size_t index = 0; index < options.iterations; ++index) {
    begin = Clock::now();
    status = (*operation)(call);
    if (!status.ok())
      throw std::runtime_error(status.message());
    durations.push_back(std::chrono::duration<double, std::milli>(Clock::now() - begin).count());
  }
  result.mean_ms = std::accumulate(durations.begin(), durations.end(), 0.0) / durations.size();
  for (double duration : durations)
    result.stddev_ms += std::pow(duration - result.mean_ms, 2);
  result.stddev_ms = std::sqrt(result.stddev_ms / durations.size());

  std::vector<double> reference(count);
  for (std::size_t row = 0; row < rows; ++row) {
    if (options.op == OperatorKind::Softmax) {
      double maximum = -std::numeric_limits<double>::infinity();
      for (std::size_t column = 0; column < columns; ++column)
        maximum = std::max(maximum, static_cast<double>(input[row * columns + column]));
      double sum = 0;
      for (std::size_t column = 0; column < columns; ++column) {
        auto& value = reference[row * columns + column];
        value = std::exp(static_cast<double>(input[row * columns + column]) - maximum);
        sum += value;
      }
      for (std::size_t column = 0; column < columns; ++column)
        reference[row * columns + column] /= sum;
    } else {
      double sum = 0, sum_squared = 0;
      for (std::size_t column = 0; column < columns; ++column) {
        const double value = static_cast<double>(input[row * columns + column]);
        sum += value;
        sum_squared += value * value;
      }
      const double mean = sum / columns;
      const double variance = std::max(sum_squared / columns - mean * mean, 0.0);
      const double inverse_stddev = 1 / std::sqrt(variance + options.epsilon);
      for (std::size_t column = 0; column < columns; ++column)
        reference[row * columns + column] = (static_cast<double>(input[row * columns + column]) - mean) *
                                              inverse_stddev * static_cast<double>(scale[column]) +
                                            static_cast<double>(bias[column]);
    }
  }
  std::vector<double> actual(count);
  std::transform(output.begin(), output.end(), actual.begin(), [](Output value) { return static_cast<double>(value); });
  const auto [atol, rtol] = tolerances<Output, Compute>(options.op);
  result.atol = atol;
  result.rtol = rtol;
  result.accuracy = compare(reference, actual, atol, rtol);
  return result;
}

template <typename Input, typename Compute>
RunResult dispatch_output(const Options& options, std::size_t rows, std::size_t columns) {
  switch (options.output_dtype) {
  case rt::DType::Float16:
    return run_typed<Input, vecops::float16_t, Compute>(options, rows, columns);
  case rt::DType::BFloat16:
    return run_typed<Input, vecops::bfloat16_t, Compute>(options, rows, columns);
  case rt::DType::Float32:
    return run_typed<Input, vecops::float32_t, Compute>(options, rows, columns);
  case rt::DType::Float64:
    return run_typed<Input, vecops::float64_t, Compute>(options, rows, columns);
  default:
    throw std::logic_error("unreachable output dtype");
  }
}

template <typename Compute>
RunResult dispatch_input(const Options& options, std::size_t rows, std::size_t columns) {
  switch (options.input_dtype) {
  case rt::DType::Float16:
    return dispatch_output<vecops::float16_t, Compute>(options, rows, columns);
  case rt::DType::BFloat16:
    return dispatch_output<vecops::bfloat16_t, Compute>(options, rows, columns);
  case rt::DType::Float32:
    return dispatch_output<vecops::float32_t, Compute>(options, rows, columns);
  case rt::DType::Float64:
    return dispatch_output<vecops::float64_t, Compute>(options, rows, columns);
  default:
    throw std::logic_error("unreachable input dtype");
  }
}

RunResult run(const Options& options, std::size_t rows, std::size_t columns) {
  return options.compute_dtype == rt::DType::Float64 ? dispatch_input<vecops::float64_t>(options, rows, columns)
                                                     : dispatch_input<vecops::float32_t>(options, rows, columns);
}

std::string shape_string(const std::vector<nint_t>& shape) {
  std::string result;
  for (auto dimension : shape)
    result += (result.empty() ? "" : ",") + std::to_string(dimension);
  return result;
}

} // namespace

int main(int argc, char** argv) {
  try {
    const auto options = parse_options(argc, argv);
    const auto count = checked_numel(options.shape);
    const auto columns = static_cast<std::size_t>(options.shape.back());
    const auto rows = count / columns;
    std::cout << "vecops normalization new-API probe\n"
              << "  operator: " << op_name(options.op) << "\n"
              << "  dtype: " << dtype_name(options.input_dtype) << " -> " << dtype_name(options.output_dtype)
              << ", compute=" << dtype_name(options.compute_dtype) << "\n"
              << "  shape: [" << shape_string(options.shape) << "] (flattened to " << rows << 'x' << columns << ")\n"
              << "  cache mode: " << cache_mode_name(options.cache_mode) << "\n";
    const auto result = run(options, rows, columns);
    std::cout << std::fixed << std::setprecision(6) << "  first call: " << result.first_ms << " ms\n"
              << "  measured: " << result.mean_ms << " +/- " << result.stddev_ms << " ms\n"
              << "  max abs/rel error: " << result.accuracy.max_abs_error << " / " << result.accuracy.max_rel_error
              << "\n"
              << "  mismatches: " << result.accuracy.mismatches << " / " << count << "\n"
              << (result.accuracy.passed ? "PASS\n" : "FAIL\n");
    return result.accuracy.passed ? 0 : 1;
  } catch (const std::exception& error) {
    std::cerr << "error: " << error.what() << "\nFAIL\n";
    return 2;
  }
}
