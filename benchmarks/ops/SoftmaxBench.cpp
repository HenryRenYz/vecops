#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdlib>
#include <limits>
#include <new>
#include <sstream>
#include <string>
#include <type_traits>
#include <vector>

#include <benchmark/benchmark.h>

#include "BenchmarkUtils.h"
#include "vecops/tensor/DataAccess.h"
#include "vecops/tensor/Tensor.h"
#include "vecops/kernel/Workspace.h"
#include "vecops/ops/Softmax.h"

#ifdef VECOPS_BENCH_USE_ONEDNN
#include <oneapi/dnnl/dnnl.h>
#include <oneapi/dnnl/dnnl.hpp>
#endif

using namespace vecops;
using namespace vecops::meta;
using namespace vecops::tensor;
using namespace vecops::ops;
using vecops::kernel::Workspace;

namespace {

template <typename T, std::size_t Alignment = 64>
struct AlignedAllocator {
  using value_type = T;

  constexpr AlignedAllocator() noexcept = default;

  template <typename U>
  constexpr AlignedAllocator(
      const AlignedAllocator<U, Alignment>&) noexcept {}

  [[nodiscard]] T* allocate(std::size_t count) {
    const std::size_t bytes = count * sizeof(T);
    const std::size_t aligned_bytes =
        (bytes + Alignment - 1) / Alignment * Alignment;
    void* pointer = std::aligned_alloc(Alignment, aligned_bytes);
    if (pointer == nullptr) throw std::bad_alloc{};
    return static_cast<T*>(pointer);
  }

  void deallocate(T* pointer, std::size_t) noexcept {
    std::free(pointer);
  }

  template <typename U>
  struct rebind {
    using other = AlignedAllocator<U, Alignment>;
  };
};

template <typename T, typename U, std::size_t Alignment>
constexpr bool operator==(
    const AlignedAllocator<T, Alignment>&,
    const AlignedAllocator<U, Alignment>&) noexcept {
  return true;
}

template <typename T>
using AlignedVector = std::vector<T, AlignedAllocator<T>>;

#ifndef VECOPS_BENCH_ARCH_CODE
#define VECOPS_BENCH_ARCH_CODE "unknown"
#endif

#ifndef VECOPS_SOURCE_DIR
#define VECOPS_SOURCE_DIR "."
#endif

struct SoftmaxCase {
  const char* group;
  std::array<nint_t, 4> shape;
  int rank;
};

constexpr SoftmaxCase kCases[] = {
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
    // Decode shapes are [batch, heads, keys].  The prefill shape is
    // [batch, heads, queries, keys].  Softmax is over the final dimension.
    {"llm_decode", {1, 64, 128, 1}, 3},
    {"llm_decode", {1, 64, 640, 1}, 3},
    {"llm_decode", {1, 128, 1152, 1}, 3},
    {"llm_decode", {1, 64, 2048, 1}, 3},
    {"llm_decode", {1, 128, 8320, 1}, 3},
    {"llm_prefill", {1, 64, 128, 2048}, 4},
    {"llm_vocab", {1, 154880, 1, 1}, 2},
    {"llm_stress", {1, 64, 262144, 1}, 3},
};

template <typename T>
const char* dtype_name() {
  if constexpr (std::is_same_v<T, vecops::float16_t>) return "fp16";
  if constexpr (std::is_same_v<T, vecops::float32_t>) return "fp32";
  if constexpr (std::is_same_v<T, vecops::float64_t>) return "fp64";
  if constexpr (std::is_same_v<T, vecops::bfloat16_t>) return "bf16";
  return "unknown";
}

template <vec::Accuracy Mode>
const char* mode_name() {
#ifdef VECOPS_BENCH_USE_ONEDNN
  return "oneDNN-accurate";
#else
  if constexpr (Mode == vec::Accuracy::Strict) return "strict";
  if constexpr (Mode == vec::Accuracy::Fast) return "fast";
  return "estimate";
#endif
}

template <vec::Accuracy Mode, typename T>
double tolerance() {
#ifdef VECOPS_BENCH_USE_ONEDNN
  if constexpr (std::is_same_v<T, vecops::bfloat16_t>) return 8e-3;
  if constexpr (std::is_same_v<T, vecops::float16_t>) return 2e-3;
  return 3e-5;
#else
  if constexpr (Mode == vec::Accuracy::Estimate) {
    if constexpr (std::is_same_v<T, vecops::bfloat16_t>) return 2e-2;
    if constexpr (std::is_same_v<T, vecops::float16_t>) return 1e-2;
    return 8e-3;
  }
  if constexpr (std::is_same_v<T, vecops::bfloat16_t>) return 8e-3;
  if constexpr (std::is_same_v<T, vecops::float16_t>) return 2e-3;
  if constexpr (std::is_same_v<T, vecops::float64_t>) return 2e-12;
  return 3e-5;
#endif
}

template <typename T>
using softmax_compute_type_t =
    std::conditional_t<
        std::is_same_v<T, vecops::float64_t>,
        vecops::float64_t,
        vecops::float32_t>;

std::string shape_name(const SoftmaxCase& c) {
  std::ostringstream os;
  for (int i = 0; i < c.rank; ++i) {
    if (i != 0) os << "x";
    os << c.shape[static_cast<size_t>(i)];
  }
  return os.str();
}

nint_t total_elements(const SoftmaxCase& c) {
  nint_t total = 1;
  for (int i = 0; i < c.rank; ++i) total *= c.shape[static_cast<size_t>(i)];
  return total;
}

nint_t normalized_size(const SoftmaxCase& c) {
  return c.shape[static_cast<size_t>(c.rank - 1)];
}

nint_t row_count(const SoftmaxCase& c) {
  return total_elements(c) / normalized_size(c);
}

#ifdef VECOPS_BENCH_USE_ONEDNN
dnnl::memory::dims onednn_dims(const SoftmaxCase& c) {
  dnnl::memory::dims dims;
  dims.reserve(static_cast<size_t>(c.rank));
  for (int i = 0; i < c.rank; ++i) {
    dims.push_back(c.shape[static_cast<size_t>(i)]);
  }
  return dims;
}

dnnl::memory::desc onednn_plain_desc(
    const dnnl::memory::dims& dims,
    dnnl::memory::data_type dtype) {
  dnnl::memory::dims strides(dims.size());
  dnnl::memory::dim stride = 1;
  for (size_t i = dims.size(); i-- > 0;) {
    strides[i] = stride;
    stride *= dims[i];
  }
  return dnnl::memory::desc(dims, dtype, strides);
}

template <typename T>
constexpr dnnl::memory::data_type onednn_dtype() {
  if constexpr (std::is_same_v<T, vecops::float32_t>) {
    return dnnl::memory::data_type::f32;
  } else if constexpr (std::is_same_v<T, vecops::float16_t>) {
    return dnnl::memory::data_type::f16;
  } else {
    static_assert(std::is_same_v<T, vecops::bfloat16_t>);
    return dnnl::memory::data_type::bf16;
  }
}
#endif

template <typename T>
void fill_input(AlignedVector<T>& x) {
  for (size_t i = 0; i < x.size(); ++i) {
    const float value = float(int(i % 37) - 18) * 0.17f + float(i % 11) * 0.013f;
    x[i] = static_cast<T>(value);
  }
}

template <vec::Accuracy Mode, typename T>
bool verify_output(
    const AlignedVector<T>& x,
    const AlignedVector<T>& out,
    nint_t rows,
    nint_t n) {
  const double tol = tolerance<Mode, T>();
  for (nint_t row = 0; row < rows; ++row) {
    const nint_t base = row * n;
    double max_value = -std::numeric_limits<double>::infinity();
    for (nint_t col = 0; col < n; ++col) {
      max_value = std::max(
          max_value,
          static_cast<double>(x[static_cast<size_t>(base + col)]));
    }
    double sum = 0.0;
    for (nint_t col = 0; col < n; ++col) {
      sum += std::exp(
          static_cast<double>(x[static_cast<size_t>(base + col)]) - max_value);
    }
    double actual_sum = 0.0;
    for (nint_t col = 0; col < n; ++col) {
      const size_t idx = static_cast<size_t>(base + col);
      const double expected = std::exp(static_cast<double>(x[idx]) - max_value) / sum;
      const double actual = static_cast<double>(out[idx]);
      if (std::abs(expected - actual) > tol) return false;
      actual_sum += actual;
    }
    if (std::abs(actual_sum - 1.0) > tol * std::max<double>(1.0, n / 16.0)) return false;
  }
  return true;
}

template <int Rank, vec::Accuracy Mode, typename T>
void run_case(benchmark::State& state, const SoftmaxCase& c) {
  const nint_t total = total_elements(c);
  const nint_t n = normalized_size(c);
  const nint_t rows = row_count(c);
  AlignedVector<T> x(static_cast<size_t>(total));
  AlignedVector<T> out(static_cast<size_t>(total), T{});
  fill_input(x);

#ifdef VECOPS_BENCH_USE_ONEDNN
  static_assert(
      std::is_same_v<T, vecops::float32_t> ||
      std::is_same_v<T, vecops::float16_t> ||
      std::is_same_v<T, vecops::bfloat16_t>);
  const dnnl::engine engine(dnnl::engine::kind::cpu, 0);
  const dnnl::stream stream(engine);
  const auto data_md = onednn_plain_desc(onednn_dims(c), onednn_dtype<T>());
  dnnl::primitive_attr attr;
  attr.set_scratchpad_mode(dnnl::scratchpad_mode::user);
  const dnnl::softmax_forward::primitive_desc pd(
      engine,
      dnnl::prop_kind::forward_inference,
      dnnl::algorithm::softmax_accurate,
      data_md,
      data_md,
      c.rank - 1,
      attr);
  const std::string impl = pd.impl_info_str();
  const std::string required_impl = VECOPS_BENCH_ONEDNN_IMPL_TOKEN;
  if (!required_impl.empty() &&
      impl.find(required_impl) == std::string::npos) {
    const std::string error =
        "oneDNN Softmax implementation '" + impl +
        "' does not contain required token '" + required_impl + "'";
    state.SkipWithError(error.c_str());
    return;
  }

  const dnnl::softmax_forward primitive(pd);
  const dnnl::memory src_mem(data_md, engine, x.data());
  const dnnl::memory dst_mem(data_md, engine, out.data());
  std::vector<dnnl_exec_arg_t> exec_args = {
      {DNNL_ARG_SRC, src_mem.get()},
      {DNNL_ARG_DST, dst_mem.get()},
  };
  const size_t scratchpad_bytes = pd.scratchpad_desc().get_size();
  dnnl::memory scratchpad_mem;
  if (scratchpad_bytes != 0) {
    scratchpad_mem = dnnl::memory(pd.scratchpad_desc(), engine);
    exec_args.push_back({DNNL_ARG_SCRATCHPAD, scratchpad_mem.get()});
  }
  const auto invoke = [&] {
    return dnnl_primitive_execute(
        primitive.get(),
        stream.get(),
        static_cast<int>(exec_args.size()),
        exec_args.data());
  };

  if (invoke() != dnnl_success) {
    state.SkipWithError("oneDNN Softmax execution failed");
    return;
  }
  if (!verify_output<Mode>(x, out, rows, n)) {
    state.SkipWithError("oneDNN Softmax output verification failed");
    return;
  }
  for (auto _ : state) {
    benchmark::DoNotOptimize(x.data());
    benchmark::DoNotOptimize(invoke());
    benchmark::ClobberMemory();
  }
  state.SetLabel(impl);
  state.counters["workspace_bytes"] =
      benchmark::Counter(double(scratchpad_bytes));
#else
  using ComputeT = softmax_compute_type_t<T>;
  using Config = SoftmaxConfig<ComputeT, vec::ScalableTag<ComputeT, 0>, Mode>;
  Config config{};
#ifdef VECOPS_BENCH_DISABLE_ONLINE
  config.allow_online = false;
#endif
  if (std::getenv("VECOPS_BENCH_DISABLE_ONLINE") != nullptr) {
    config.allow_online = false;
  }
  auto op = softmax(config);

  if constexpr (Rank == 1) {
    auto x_t = make_tensor<1>(x.data(), {c.shape[0]});
    auto y_t = make_tensor<1>(out.data(), {c.shape[0]});
    auto x_spec = input<ComputeT>(x_t);
    auto y_spec = output<ComputeT>(y_t);
    Workspace workspace(op.required_workspace(x_spec, y_spec));
    auto view = workspace.view();
    op(view, x_spec, y_spec);
    if (!verify_output<Mode>(x, out, rows, n)) {
      state.SkipWithError("Softmax output verification failed");
      return;
    }
    for (auto _ : state) {
      benchmark::DoNotOptimize(x.data());
      op(view, x_spec, y_spec);
      benchmark::ClobberMemory();
    }
    state.counters["workspace_bytes"] = benchmark::Counter(double(workspace.requested_capacity()));
  } else if constexpr (Rank == 2) {
    auto x_t = make_tensor<2>(x.data(), {c.shape[0], c.shape[1]});
    auto y_t = make_tensor<2>(out.data(), {c.shape[0], c.shape[1]});
    auto x_spec = input<ComputeT>(x_t);
    auto y_spec = output<ComputeT>(y_t);
    Workspace workspace(op.required_workspace(x_spec, y_spec));
    auto view = workspace.view();
    op(view, x_spec, y_spec);
    if (!verify_output<Mode>(x, out, rows, n)) {
      state.SkipWithError("Softmax output verification failed");
      return;
    }
    for (auto _ : state) {
      benchmark::DoNotOptimize(x.data());
      op(view, x_spec, y_spec);
      benchmark::ClobberMemory();
    }
    state.counters["workspace_bytes"] = benchmark::Counter(double(workspace.requested_capacity()));
  } else if constexpr (Rank == 3) {
    auto x_t = make_tensor<3>(x.data(), {c.shape[0], c.shape[1], c.shape[2]});
    auto y_t = make_tensor<3>(out.data(), {c.shape[0], c.shape[1], c.shape[2]});
    auto x_spec = input<ComputeT>(x_t);
    auto y_spec = output<ComputeT>(y_t);
    Workspace workspace(op.required_workspace(x_spec, y_spec));
    auto view = workspace.view();
    op(view, x_spec, y_spec);
    if (!verify_output<Mode>(x, out, rows, n)) {
      state.SkipWithError("Softmax output verification failed");
      return;
    }
    for (auto _ : state) {
      benchmark::DoNotOptimize(x.data());
      op(view, x_spec, y_spec);
      benchmark::ClobberMemory();
    }
    state.counters["workspace_bytes"] = benchmark::Counter(double(workspace.requested_capacity()));
  } else {
    auto x_t = make_tensor<4>(x.data(), {c.shape[0], c.shape[1], c.shape[2], c.shape[3]});
    auto y_t = make_tensor<4>(out.data(), {c.shape[0], c.shape[1], c.shape[2], c.shape[3]});
    auto x_spec = input<ComputeT>(x_t);
    auto y_spec = output<ComputeT>(y_t);
    Workspace workspace(op.required_workspace(x_spec, y_spec));
    auto view = workspace.view();
    op(view, x_spec, y_spec);
    if (!verify_output<Mode>(x, out, rows, n)) {
      state.SkipWithError("Softmax output verification failed");
      return;
    }
    for (auto _ : state) {
      benchmark::DoNotOptimize(x.data());
      op(view, x_spec, y_spec);
      benchmark::ClobberMemory();
    }
    state.counters["workspace_bytes"] = benchmark::Counter(double(workspace.requested_capacity()));
  }
#endif

  const double bytes_per_iter = double(total * sizeof(T) * 2);
  state.SetItemsProcessed(state.iterations() * total);
  state.SetBytesProcessed(state.iterations() * static_cast<int64_t>(bytes_per_iter));
  state.counters["rank"] = benchmark::Counter(double(c.rank));
  state.counters["rows"] = benchmark::Counter(double(rows));
  state.counters["norm"] = benchmark::Counter(double(n));
  state.counters["elements"] = benchmark::Counter(double(total));
}

template <vec::Accuracy Mode, typename T>
void bench_softmax(benchmark::State& state, SoftmaxCase c) {
  if (c.rank == 1) return run_case<1, Mode, T>(state, c);
  if (c.rank == 2) return run_case<2, Mode, T>(state, c);
  if (c.rank == 3) return run_case<3, Mode, T>(state, c);
  return run_case<4, Mode, T>(state, c);
}

template <vec::Accuracy Mode, typename T>
void register_mode_dtype() {
  for (const auto& c : kCases) {
    const std::string name =
        "Softmax/" + std::string(c.group) +
        "/rank:" + std::to_string(c.rank) +
        "/shape:" + shape_name(c) +
        "/dtype:" + dtype_name<T>() +
        "/mode:" + mode_name<Mode>() +
        "/arch:" + VECOPS_BENCH_ARCH_CODE;
    benchmark::RegisterBenchmark(name.c_str(), &bench_softmax<Mode, T>, c)
        ->Unit(benchmark::kMicrosecond)
        ->MinTime(0.02)
        ->Repetitions(3)
        ->ReportAggregatesOnly(true);
  }
}

template <typename T>
void register_dtype() {
#ifdef VECOPS_BENCH_USE_ONEDNN
  register_mode_dtype<vec::Accuracy::Estimate, T>();
#else
  register_mode_dtype<vec::Accuracy::Strict, T>();
  register_mode_dtype<vec::Accuracy::Fast, T>();
  register_mode_dtype<vec::Accuracy::Estimate, T>();
#endif
}

void register_softmax_benchmarks() {
#ifdef VECOPS_BENCH_USE_ONEDNN
  register_dtype<vecops::float32_t>();
  register_dtype<vecops::float16_t>();
  register_dtype<vecops::bfloat16_t>();
#else
  register_dtype<vecops::float32_t>();
  register_dtype<vecops::float64_t>();
  register_dtype<vecops::float16_t>();
#if defined(HAS_BFLOAT16) || defined(ARCH_X86_FAMILY)
  register_dtype<vecops::bfloat16_t>();
#endif
#endif
}

} // namespace

int main(int argc, char** argv) {
  register_softmax_benchmarks();

  const auto default_csv = vecops::bench::default_result_path(
      VECOPS_SOURCE_DIR, "softmax", VECOPS_BENCH_ARCH_CODE, "csv");
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
