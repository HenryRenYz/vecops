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

struct SmallGroup { static constexpr const char* name = "small"; };
struct MediumGroup { static constexpr const char* name = "medium"; };
struct LargeGroup { static constexpr const char* name = "large"; };
struct TailGroup { static constexpr const char* name = "tail"; };
struct DecodeGroup { static constexpr const char* name = "llm_decode"; };
struct PrefillGroup { static constexpr const char* name = "llm_prefill"; };
struct VocabGroup { static constexpr const char* name = "llm_vocab"; };
struct StressGroup { static constexpr const char* name = "llm_stress"; };

template <typename Group, nint_t... Dimensions>
struct SoftmaxCase {
  static_assert(sizeof...(Dimensions) > 0);
  using GroupType = Group;
  static constexpr int rank = sizeof...(Dimensions);
  static constexpr std::array<nint_t, rank> shape{Dimensions...};
  static constexpr nint_t elements = (Dimensions * ...);
  static constexpr nint_t normalized = shape.back();
  static constexpr nint_t rows = elements / normalized;

  template <bool CompileTimeShape>
  static constexpr auto tensor_shape() {
    if constexpr (CompileTimeShape) {
      return make_shape(cint<Dimensions>...);
    } else {
      return make_shape(Any{Dimensions}...);
    }
  }
};

template <typename... Cases>
struct CaseList {};

// Every case is registered with both Const and Any shape metadata so the
// benchmark exposes the cost of losing compile-time dispatch information.
#ifdef VECOPS_BENCH_QUICK_FP16
using SoftmaxCases = CaseList<SoftmaxCase<LargeGroup, 16, 64, 4096>>;
#else
using SoftmaxCases = CaseList<
    SoftmaxCase<SmallGroup, 64>,
    SoftmaxCase<SmallGroup, 8, 128>,
    SoftmaxCase<SmallGroup, 4, 8, 256>,
    SoftmaxCase<SmallGroup, 2, 4, 8, 128>,
    SoftmaxCase<MediumGroup, 64, 768>,
    SoftmaxCase<MediumGroup, 32, 128, 1024>,
    SoftmaxCase<MediumGroup, 8, 16, 32, 768>,
    SoftmaxCase<LargeGroup, 4096, 1024>,
    SoftmaxCase<LargeGroup, 16, 64, 4096>,
    SoftmaxCase<TailGroup, 7>,
    SoftmaxCase<TailGroup, 64, 513>,
    SoftmaxCase<TailGroup, 16, 32, 1000>,
    // Decode is [batch, heads, keys], prefill is
    // [batch, heads, queries, keys]. Softmax uses the final dimension.
    SoftmaxCase<DecodeGroup, 1, 64, 128>,
    SoftmaxCase<DecodeGroup, 1, 64, 640>,
    SoftmaxCase<DecodeGroup, 1, 128, 1152>,
    SoftmaxCase<DecodeGroup, 1, 64, 2048>,
    SoftmaxCase<DecodeGroup, 1, 128, 8320>,
    SoftmaxCase<PrefillGroup, 1, 64, 128, 2048>,
    SoftmaxCase<VocabGroup, 1, 154880>,
    SoftmaxCase<StressGroup, 1, 64, 262144>>;
#endif

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

template <typename Case>
std::string shape_name() {
  std::ostringstream os;
  for (int i = 0; i < Case::rank; ++i) {
    if (i != 0) os << "x";
    os << Case::shape[static_cast<size_t>(i)];
  }
  return os.str();
}

struct RuntimeSoftmaxCase {
  std::array<nint_t, 4> shape{};
  int rank{};
  nint_t elements{};
  nint_t normalized{};
  nint_t rows{};
};

template <typename Case>
constexpr RuntimeSoftmaxCase runtime_case() {
  RuntimeSoftmaxCase result{};
  for (int i = 0; i < Case::rank; ++i) {
    result.shape[static_cast<size_t>(i)] =
        Case::shape[static_cast<size_t>(i)];
  }
  result.rank = Case::rank;
  result.elements = Case::elements;
  result.normalized = Case::normalized;
  result.rows = Case::rows;
  return result;
}

#ifdef VECOPS_BENCH_USE_ONEDNN
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

template <vec::Accuracy Mode, typename T>
void run_onednn_case(
    benchmark::State& state, const RuntimeSoftmaxCase& benchmark_case) {
  const nint_t total = benchmark_case.elements;
  const nint_t n = benchmark_case.normalized;
  const nint_t rows = benchmark_case.rows;
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
  dnnl::memory::dims dims;
  dims.reserve(static_cast<size_t>(benchmark_case.rank));
  for (int i = 0; i < benchmark_case.rank; ++i) {
    dims.push_back(benchmark_case.shape[static_cast<size_t>(i)]);
  }
  const auto data_md = onednn_plain_desc(dims, onednn_dtype<T>());
  dnnl::primitive_attr attr;
  attr.set_scratchpad_mode(dnnl::scratchpad_mode::user);
  const dnnl::softmax_forward::primitive_desc pd(
      engine,
      dnnl::prop_kind::forward_inference,
      dnnl::algorithm::softmax_accurate,
      data_md,
      data_md,
      benchmark_case.rank - 1,
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
  (void)state;
  (void)benchmark_case;
#endif

  const double bytes_per_iter = double(total * sizeof(T) * 2);
  state.SetItemsProcessed(state.iterations() * total);
  state.SetBytesProcessed(
      state.iterations() * static_cast<int64_t>(bytes_per_iter));
  state.counters["rank"] = benchmark::Counter(double(benchmark_case.rank));
  state.counters["rows"] = benchmark::Counter(double(rows));
  state.counters["norm"] = benchmark::Counter(double(n));
  state.counters["elements"] = benchmark::Counter(double(total));
}

#ifndef VECOPS_BENCH_USE_ONEDNN
template <vec::Accuracy Mode, typename T, bool AllowOnline, typename Shape>
void run_vecops_case(
    benchmark::State& state, Shape shape, nint_t total, int rank,
    nint_t rows, nint_t n) {
  AlignedVector<T> x(static_cast<size_t>(total));
  AlignedVector<T> out(static_cast<size_t>(total), T{});
  fill_input(x);
  using ComputeT = softmax_compute_type_t<T>;
  using Config = SoftmaxConfig<ComputeT, Mode, AllowOnline>;
  auto op = softmax(Config{});
  auto x_t = make_tensor(x.data(), shape);
  auto y_t = make_tensor(out.data(), shape);
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
  state.counters["workspace_bytes"] =
      benchmark::Counter(double(workspace.requested_capacity()));

  const double bytes_per_iter = double(total * sizeof(T) * 2);
  state.SetItemsProcessed(state.iterations() * total);
  state.SetBytesProcessed(
      state.iterations() * static_cast<int64_t>(bytes_per_iter));
  state.counters["rank"] = benchmark::Counter(double(rank));
  state.counters["rows"] = benchmark::Counter(double(rows));
  state.counters["norm"] = benchmark::Counter(double(n));
  state.counters["elements"] = benchmark::Counter(double(total));
}

template <typename Case, vec::Accuracy Mode, typename T, bool AllowOnline>
void run_compile_time_case(benchmark::State& state) {
  run_vecops_case<Mode, T, AllowOnline>(
      state, Case::template tensor_shape<true>(), Case::elements,
      Case::rank, Case::rows, Case::normalized);
}

template <int Rank, vec::Accuracy Mode, typename T, bool AllowOnline>
void run_runtime_rank_case(
    benchmark::State& state, const RuntimeSoftmaxCase& benchmark_case) {
  const auto& d = benchmark_case.shape;
  if constexpr (Rank == 1) {
    run_vecops_case<Mode, T, AllowOnline>(
        state, make_shape(Any{d[0]}), benchmark_case.elements, Rank,
        benchmark_case.rows, benchmark_case.normalized);
  } else if constexpr (Rank == 2) {
    run_vecops_case<Mode, T, AllowOnline>(
        state, make_shape(Any{d[0]}, Any{d[1]}), benchmark_case.elements,
        Rank, benchmark_case.rows, benchmark_case.normalized);
  } else if constexpr (Rank == 3) {
    run_vecops_case<Mode, T, AllowOnline>(
        state, make_shape(Any{d[0]}, Any{d[1]}, Any{d[2]}),
        benchmark_case.elements, Rank, benchmark_case.rows,
        benchmark_case.normalized);
  } else {
    static_assert(Rank == 4);
    run_vecops_case<Mode, T, AllowOnline>(
        state,
        make_shape(Any{d[0]}, Any{d[1]}, Any{d[2]}, Any{d[3]}),
        benchmark_case.elements, Rank, benchmark_case.rows,
        benchmark_case.normalized);
  }
}

template <typename Case, vec::Accuracy Mode, typename T>
void bench_compile_time_softmax(benchmark::State& state) {
  const bool disable_online =
      std::getenv("VECOPS_BENCH_DISABLE_ONLINE") != nullptr;
#ifdef VECOPS_BENCH_DISABLE_ONLINE
  constexpr bool DefaultAllowOnline = false;
#else
  constexpr bool DefaultAllowOnline = true;
#endif
  if (disable_online) {
    return run_compile_time_case<Case, Mode, T, false>(state);
  }
  return run_compile_time_case<Case, Mode, T, DefaultAllowOnline>(state);
}

template <vec::Accuracy Mode, typename T, bool AllowOnline>
void run_runtime_case(
    benchmark::State& state, const RuntimeSoftmaxCase& benchmark_case) {
  switch (benchmark_case.rank) {
    case 1:
      return run_runtime_rank_case<1, Mode, T, AllowOnline>(
          state, benchmark_case);
    case 2:
      return run_runtime_rank_case<2, Mode, T, AllowOnline>(
          state, benchmark_case);
    case 3:
      return run_runtime_rank_case<3, Mode, T, AllowOnline>(
          state, benchmark_case);
    default:
      return run_runtime_rank_case<4, Mode, T, AllowOnline>(
          state, benchmark_case);
  }
}

template <vec::Accuracy Mode, typename T>
void bench_runtime_softmax(
    benchmark::State& state, RuntimeSoftmaxCase benchmark_case) {
  const bool disable_online =
      std::getenv("VECOPS_BENCH_DISABLE_ONLINE") != nullptr;
#ifdef VECOPS_BENCH_DISABLE_ONLINE
  constexpr bool DefaultAllowOnline = false;
#else
  constexpr bool DefaultAllowOnline = true;
#endif
  if (disable_online) {
    return run_runtime_case<Mode, T, false>(state, benchmark_case);
  }
  return run_runtime_case<Mode, T, DefaultAllowOnline>(
      state, benchmark_case);
}
#endif

template <typename Case, vec::Accuracy Mode, typename T>
std::string benchmark_name(const char* shape_metadata) {
  return
      "Softmax/" + std::string(Case::GroupType::name) +
      "/rank:" + std::to_string(Case::rank) +
      "/shape:" + shape_name<Case>() +
      "/shape_meta:" + shape_metadata +
      "/dtype:" + dtype_name<T>() +
      "/mode:" + mode_name<Mode>() +
      "/arch:" + VECOPS_BENCH_ARCH_CODE;
}

template <typename Case, vec::Accuracy Mode, typename T>
void register_case() {
#ifdef VECOPS_BENCH_USE_ONEDNN
  const std::string name =
      benchmark_name<Case, Mode, T>("runtime");
  benchmark::RegisterBenchmark(
      name.c_str(), &run_onednn_case<Mode, T>, runtime_case<Case>())
      ->Unit(benchmark::kMicrosecond)
      ->MinTime(0.02)
      ->Repetitions(3)
      ->ReportAggregatesOnly(true);
#else
  const std::string compile_name =
      benchmark_name<Case, Mode, T>("compile_time");
  benchmark::RegisterBenchmark(
      compile_name.c_str(), &bench_compile_time_softmax<Case, Mode, T>)
      ->Unit(benchmark::kMicrosecond)
      ->MinTime(0.02)
      ->Repetitions(3)
      ->ReportAggregatesOnly(true);

  const std::string runtime_name =
      benchmark_name<Case, Mode, T>("runtime");
  benchmark::RegisterBenchmark(
      runtime_name.c_str(), &bench_runtime_softmax<Mode, T>,
      runtime_case<Case>())
      ->Unit(benchmark::kMicrosecond)
      ->MinTime(0.02)
      ->Repetitions(3)
      ->ReportAggregatesOnly(true);
#endif
}

template <vec::Accuracy Mode, typename T, typename... Cases>
void register_cases(CaseList<Cases...>) {
  (register_case<Cases, Mode, T>(), ...);
}

template <vec::Accuracy Mode, typename T>
void register_mode_dtype() {
  register_cases<Mode, T>(SoftmaxCases{});
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
#ifdef VECOPS_BENCH_QUICK_FP16
  register_mode_dtype<vec::Accuracy::Strict, vecops::float16_t>();
#elif defined(VECOPS_BENCH_USE_ONEDNN)
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
