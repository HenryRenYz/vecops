#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <sstream>
#include <string>
#include <type_traits>
#include <vector>

#include <benchmark/benchmark.h>

#include "BenchmarkUtils.h"
#include "vecops/gemm/DataAccess.h"
#include "vecops/gemm/Tensor.h"
#include "vecops/gemm/Workspace.h"
#include "vecops/ops/LayerNorm.h"

using namespace vecops;
using namespace vecops::gemm;
using namespace vecops::ops;

namespace {

#ifndef VECOPS_BENCH_ARCH_CODE
#define VECOPS_BENCH_ARCH_CODE "unknown"
#endif

#ifndef VECOPS_SOURCE_DIR
#define VECOPS_SOURCE_DIR "."
#endif

#define VECOPS_STRINGIZE_IMPL(x) #x
#define VECOPS_STRINGIZE(x) VECOPS_STRINGIZE_IMPL(x)

struct LayerNormCase {
  const char* group;
  std::array<nint_t, 4> shape;
  int rank;
};

constexpr LayerNormCase kCases[] = {
    {"small", {64, 1, 1, 1}, 1},
    {"small", {8, 128, 1, 1}, 2},
    {"small", {4, 8, 256, 1}, 3},
    {"small", {2, 4, 8, 128}, 4},
    {"medium", {64, 768, 1, 1}, 2},
    {"medium", {32, 128, 1024, 1}, 3},
    {"medium", {8, 16, 32, 768}, 4},
    {"large", {4096, 1024, 1, 1}, 2},
    {"large", {16, 64, 4096, 1}, 3},
    {"tail", {7, 1, 1, 1}, 1},
    {"tail", {64, 513, 1, 1}, 2},
    {"tail", {16, 32, 1000, 1}, 3},
};

template <typename T>
const char* dtype_name() {
  if constexpr (std::is_same_v<T, vecops::float16_t>) return "fp16";
  if constexpr (std::is_same_v<T, float32_t>) return "fp32";
  if constexpr (std::is_same_v<T, float64_t>) return "fp64";
  if constexpr (std::is_same_v<T, vecops::bfloat16_t>) return "bf16";
  return "unknown";
}

template <typename T>
double tolerance() {
  if constexpr (std::is_same_v<T, vecops::float16_t>) return 2e-2;
  if constexpr (std::is_same_v<T, vecops::bfloat16_t>) return 4e-2;
  if constexpr (std::is_same_v<T, float64_t>) return 2e-8;
  return 2e-5;
}

template <typename T>
using layernorm_compute_type_t =
    std::conditional_t<std::is_same_v<T, float64_t>, float64_t, float32_t>;

std::string shape_name(const LayerNormCase& c) {
  std::ostringstream os;
  for (int i = 0; i < c.rank; ++i) {
    if (i != 0) os << "x";
    os << c.shape[static_cast<size_t>(i)];
  }
  return os.str();
}

nint_t total_elements(const LayerNormCase& c) {
  nint_t total = 1;
  for (int i = 0; i < c.rank; ++i) total *= c.shape[static_cast<size_t>(i)];
  return total;
}

nint_t normalized_size(const LayerNormCase& c) {
  return c.shape[static_cast<size_t>(c.rank - 1)];
}

nint_t row_count(const LayerNormCase& c) {
  return total_elements(c) / normalized_size(c);
}

template <typename T>
void fill_inputs(std::vector<T>& x, std::vector<T>& scale, std::vector<T>& bias) {
  for (size_t i = 0; i < x.size(); ++i) {
    const float v = float(int(i % 37) - 18) * 0.07f + float(i % 11) * 0.013f;
    x[i] = static_cast<T>(v);
  }
  for (size_t i = 0; i < scale.size(); ++i) {
    scale[i] = static_cast<T>(0.75f + 0.001f * float(i % 257));
    bias[i] = static_cast<T>(-0.2f + 0.0007f * float(i % 263));
  }
}

template <typename T>
bool verify_output(
    const std::vector<T>& x,
    const std::vector<T>& scale,
    const std::vector<T>& bias,
    const std::vector<T>& out,
    nint_t rows,
    nint_t n) {
  const double eps = 1e-5;
  const double tol = tolerance<T>();
  for (nint_t row = 0; row < rows; ++row) {
    const nint_t base = row * n;
    double sum = 0.0;
    double sum_sq = 0.0;
    for (nint_t col = 0; col < n; ++col) {
      const double v = static_cast<double>(x[static_cast<size_t>(base + col)]);
      sum += v;
      sum_sq += v * v;
    }
    const double mean = sum / double(n);
    const double var = std::max(sum_sq / double(n) - mean * mean, 0.0);
    const double rstd = 1.0 / std::sqrt(var + eps);
    for (nint_t col = 0; col < n; ++col) {
      const size_t idx = static_cast<size_t>(base + col);
      const double expected =
          (static_cast<double>(x[idx]) - mean) * rstd *
          static_cast<double>(scale[static_cast<size_t>(col)]) +
          static_cast<double>(bias[static_cast<size_t>(col)]);
      const double actual = static_cast<double>(out[idx]);
      const double scale_ref = std::max({1.0, std::abs(expected), std::abs(actual)});
      if (std::abs(expected - actual) > tol * scale_ref) return false;
    }
  }
  return true;
}

template <int Rank, typename T>
void run_case(benchmark::State& state, const LayerNormCase& c) {
  const nint_t total = total_elements(c);
  const nint_t n = normalized_size(c);
  const nint_t rows = row_count(c);

  std::vector<T> x(static_cast<size_t>(total));
  std::vector<T> scale(static_cast<size_t>(n));
  std::vector<T> bias(static_cast<size_t>(n));
  std::vector<T> out(static_cast<size_t>(total), T{});
  fill_inputs(x, scale, bias);

  using ComputeT = layernorm_compute_type_t<T>;
  constexpr ComputeT eps = ComputeT(1e-5);
  auto s_t = make_tensor<1>(scale.data(), {n});
  auto b_t = make_tensor<1>(bias.data(), {n});
  auto op = layer_norm(LayerNormConfig<ComputeT>{.eps = eps});
  auto s_spec = input<ComputeT>(s_t);
  auto b_spec = input<ComputeT>(b_t);

  if constexpr (Rank == 1) {
    auto x_t = make_tensor<1>(x.data(), {c.shape[0]});
    auto y_t = make_tensor<1>(out.data(), {c.shape[0]});
    auto x_spec = input<ComputeT>(x_t);
    auto y_spec = output<ComputeT>(y_t);
    Workspace workspace(op.required_workspace(x_spec, s_spec, b_spec, y_spec));
    auto view = workspace.view();
    op(view, x_spec, s_spec, b_spec, y_spec);
    if (!verify_output(x, scale, bias, out, rows, n)) {
      state.SkipWithError("LayerNorm output verification failed");
      return;
    }
    for (auto _ : state) {
      benchmark::DoNotOptimize(x.data());
      op(view, x_spec, s_spec, b_spec, y_spec);
      benchmark::ClobberMemory();
    }
    state.counters["workspace_bytes"] = benchmark::Counter(double(workspace.requested_capacity()));
  } else if constexpr (Rank == 2) {
    auto x_t = make_tensor<2>(x.data(), {c.shape[0], c.shape[1]});
    auto y_t = make_tensor<2>(out.data(), {c.shape[0], c.shape[1]});
    auto x_spec = input<ComputeT>(x_t);
    auto y_spec = output<ComputeT>(y_t);
    Workspace workspace(op.required_workspace(x_spec, s_spec, b_spec, y_spec));
    auto view = workspace.view();
    op(view, x_spec, s_spec, b_spec, y_spec);
    if (!verify_output(x, scale, bias, out, rows, n)) {
      state.SkipWithError("LayerNorm output verification failed");
      return;
    }
    for (auto _ : state) {
      benchmark::DoNotOptimize(x.data());
      op(view, x_spec, s_spec, b_spec, y_spec);
      benchmark::ClobberMemory();
    }
    state.counters["workspace_bytes"] = benchmark::Counter(double(workspace.requested_capacity()));
  } else if constexpr (Rank == 3) {
    auto x_t = make_tensor<3>(x.data(), {c.shape[0], c.shape[1], c.shape[2]});
    auto y_t = make_tensor<3>(out.data(), {c.shape[0], c.shape[1], c.shape[2]});
    auto x_spec = input<ComputeT>(x_t);
    auto y_spec = output<ComputeT>(y_t);
    Workspace workspace(op.required_workspace(x_spec, s_spec, b_spec, y_spec));
    auto view = workspace.view();
    op(view, x_spec, s_spec, b_spec, y_spec);
    if (!verify_output(x, scale, bias, out, rows, n)) {
      state.SkipWithError("LayerNorm output verification failed");
      return;
    }
    for (auto _ : state) {
      benchmark::DoNotOptimize(x.data());
      op(view, x_spec, s_spec, b_spec, y_spec);
      benchmark::ClobberMemory();
    }
    state.counters["workspace_bytes"] = benchmark::Counter(double(workspace.requested_capacity()));
  } else {
    auto x_t = make_tensor<4>(x.data(), {c.shape[0], c.shape[1], c.shape[2], c.shape[3]});
    auto y_t = make_tensor<4>(out.data(), {c.shape[0], c.shape[1], c.shape[2], c.shape[3]});
    auto x_spec = input<ComputeT>(x_t);
    auto y_spec = output<ComputeT>(y_t);
    Workspace workspace(op.required_workspace(x_spec, s_spec, b_spec, y_spec));
    auto view = workspace.view();
    op(view, x_spec, s_spec, b_spec, y_spec);
    if (!verify_output(x, scale, bias, out, rows, n)) {
      state.SkipWithError("LayerNorm output verification failed");
      return;
    }
    for (auto _ : state) {
      benchmark::DoNotOptimize(x.data());
      op(view, x_spec, s_spec, b_spec, y_spec);
      benchmark::ClobberMemory();
    }
    state.counters["workspace_bytes"] = benchmark::Counter(double(workspace.requested_capacity()));
  }

  const double bytes_per_iter = double(total * sizeof(T) * 2 + n * sizeof(T) * 2);
  state.SetItemsProcessed(state.iterations() * total);
  state.SetBytesProcessed(state.iterations() * static_cast<int64_t>(bytes_per_iter));
  state.counters["rank"] = benchmark::Counter(double(c.rank));
  state.counters["rows"] = benchmark::Counter(double(rows));
  state.counters["norm"] = benchmark::Counter(double(n));
  state.counters["elements"] = benchmark::Counter(double(total));
}

template <typename T>
void bench_layernorm(benchmark::State& state, LayerNormCase c) {
  if (c.rank == 1) return run_case<1, T>(state, c);
  if (c.rank == 2) return run_case<2, T>(state, c);
  if (c.rank == 3) return run_case<3, T>(state, c);
  return run_case<4, T>(state, c);
}

template <typename T>
void register_dtype() {
  for (const auto& c : kCases) {
    const std::string name =
        "LayerNorm/" + std::string(c.group) +
        "/rank:" + std::to_string(c.rank) +
        "/shape:" + shape_name(c) +
        "/dtype:" + dtype_name<T>() +
        "/arch:" + VECOPS_BENCH_ARCH_CODE;
    benchmark::RegisterBenchmark(name.c_str(), &bench_layernorm<T>, c)
        ->Unit(benchmark::kMicrosecond)
        ->MinTime(0.02)
        ->Repetitions(3)
        ->ReportAggregatesOnly(true);
  }
}

void register_layernorm_benchmarks() {
  register_dtype<float32_t>();
  register_dtype<float64_t>();
  register_dtype<vecops::float16_t>();
#if defined(HAS_BFLOAT16) || defined(ARCH_X86_FAMILY)
  register_dtype<vecops::bfloat16_t>();
#endif
}

} // namespace

int main(int argc, char** argv) {
  register_layernorm_benchmarks();

  const auto default_csv = vecops::bench::default_result_path(
      VECOPS_SOURCE_DIR, "layernorm", VECOPS_BENCH_ARCH_CODE, "csv");
  auto injected = vecops::bench::default_google_benchmark_output_args(
      argc, argv, default_csv, "csv");

  std::vector<char*> args;
  args.reserve(static_cast<size_t>(argc) + injected.size());
  for (int i = 0; i < argc; ++i) args.push_back(argv[i]);
  for (auto& arg : injected) args.push_back(arg.data());
  int bench_argc = static_cast<int>(args.size());

  benchmark::Initialize(&bench_argc, args.data());
  if (benchmark::ReportUnrecognizedArguments(bench_argc, args.data())) return 1;
  benchmark::RunSpecifiedBenchmarks();
  benchmark::Shutdown();
  vecops::bench::print_default_output_path(argc, argv, default_csv);
  return 0;
}
