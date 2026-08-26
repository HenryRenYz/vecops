#pragma once

#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <new>
#include <sstream>
#include <string>
#include <type_traits>
#include <vector>

#include <benchmark/benchmark.h>

#include "vecops/ops/Transpose.h"

using namespace vecops;
using namespace vecops::meta;
using namespace vecops::tensor;

namespace vecops::bench::transpose {

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

struct SmallGroup { static constexpr const char* name = "small"; };
struct MediumGroup { static constexpr const char* name = "medium"; };
struct LargeGroup { static constexpr const char* name = "large"; };
struct TailGroup { static constexpr const char* name = "tail"; };
struct DecodeGroup { static constexpr const char* name = "llm_decode"; };
struct QKVLayoutGroup { static constexpr const char* name = "llm_qkv_layout"; };
struct QKGroup { static constexpr const char* name = "llm_qk"; };
struct ConversionFullGroup {
  static constexpr const char* name = "conversion_full";
};
struct ConversionTailGroup {
  static constexpr const char* name = "conversion_tail";
};
struct ConversionMediumGroup {
  static constexpr const char* name = "conversion_medium";
};
struct ConversionLargeGroup {
  static constexpr const char* name = "conversion_large";
};

template <typename Group, nint_t M, nint_t N>
struct TransposeCase {
  using GroupType = Group;
  static constexpr nint_t rows = M;
  static constexpr nint_t columns = N;
  static constexpr nint_t elements = M * N;

  static consteval nint_t alignment(nint_t extent, nint_t maximum) {
    nint_t result = 1;
    while (result * 2 <= maximum && extent % (result * 2) == 0) {
      result *= 2;
    }
    return result;
  }

  static constexpr nint_t row_alignment = alignment(M, 16);
  static constexpr nint_t column_alignment = alignment(N, 32);

  template <bool ConstShape>
  static constexpr auto input_shape() {
    if constexpr (ConstShape) {
      return make_shape(cint<M>, cint<N>);
    } else {
      return make_shape(Any{M}, Any{N});
    }
  }

  template <bool ConstShape>
  static constexpr auto output_shape() {
    if constexpr (ConstShape) {
      return make_shape(cint<N>, cint<M>);
    } else {
      return make_shape(Any{N}, Any{M});
    }
  }
};

template <typename... Cases>
struct CaseList {};

template <typename InputT, typename OutputT, typename ComputeT>
struct TypePair {
  using Input = InputT;
  using Output = OutputT;
  using Compute = ComputeT;
};

template <typename... Pairs>
struct TypePairList {};

struct RuntimeTransposeCase {
  nint_t rows;
  nint_t columns;
  nint_t elements;
};

template <typename Case>
constexpr RuntimeTransposeCase runtime_case() {
  return {Case::rows, Case::columns, Case::elements};
}

// The LLM cases model the two logical transpose planes visible in
// mainstream decoder attention implementations:
//
//   * Q/K/V layout conversion: [sequence, heads] -> [heads, sequence]
//   * QK input conversion:      [sequence, head_dim] -> [head_dim, sequence]
//
// Head counts cover Qwen2.5 (28), Llama 3.1 (32/64), and DeepSeek-V3 (128).
// Head dimensions cover the common 128, DeepSeek-V3's 128+64 QK split, and
// Gemma 2's 256. Sequence lengths span decode, short batches, and 2K/4K/8K
// prefill. Generic cases add both orientations, square matrices, full tiles,
// tails, and cache-sized working sets.
using TransposeCases = CaseList<
    TransposeCase<SmallGroup, 16, 16>,
    TransposeCase<TailGroup, 3, 5>,
    TransposeCase<TailGroup, 17, 19>,
    TransposeCase<TailGroup, 63, 129>,
    TransposeCase<MediumGroup, 32, 64>,
    TransposeCase<MediumGroup, 256, 768>,
    TransposeCase<MediumGroup, 768, 256>,
    TransposeCase<LargeGroup, 1024, 1024>,
    TransposeCase<LargeGroup, 4096, 4096>,
    TransposeCase<DecodeGroup, 1, 32>,
    TransposeCase<DecodeGroup, 128, 128>,
    TransposeCase<QKVLayoutGroup, 2048, 28>,
    TransposeCase<QKVLayoutGroup, 4096, 64>,
    TransposeCase<QKVLayoutGroup, 8192, 128>,
    TransposeCase<QKGroup, 2048, 128>,
    TransposeCase<QKGroup, 4096, 192>,
    TransposeCase<QKGroup, 4096, 256>>;

using TransposeCompareCases = CaseList<
    TransposeCase<TailGroup, 63, 129>,
    TransposeCase<DecodeGroup, 128, 128>,
    TransposeCase<LargeGroup, 1024, 1024>,
    TransposeCase<LargeGroup, 4096, 4096>>;

// Cross-dtype cases cover every off-diagonal source/destination combination
// among representative 1/2/4/8-byte types. The diagonal combinations are
// already exercised across every TransposeCase by register_same_dtype().
using ConversionCases = CaseList<
    TransposeCase<ConversionFullGroup, 16, 16>,
    TransposeCase<ConversionTailGroup, 63, 129>,
    TransposeCase<ConversionMediumGroup, 256, 256>,
    TransposeCase<ConversionLargeGroup, 1024, 1024>>;

using Int8ConversionPairs = TypePairList<
    TypePair<int8_t, vecops::float16_t, vecops::float32_t>,
    TypePair<int8_t, vecops::float32_t, vecops::float32_t>,
    TypePair<int8_t, vecops::float64_t, vecops::float64_t>>;
using Fp16ConversionPairs = TypePairList<
    TypePair<vecops::float16_t, int8_t, vecops::float32_t>,
    TypePair<vecops::float16_t, vecops::float32_t, vecops::float32_t>,
    TypePair<vecops::float16_t, vecops::float64_t, vecops::float64_t>>;
using Fp32ConversionPairs = TypePairList<
    TypePair<vecops::float32_t, int8_t, vecops::float32_t>,
    TypePair<vecops::float32_t, vecops::float16_t, vecops::float32_t>,
    TypePair<vecops::float32_t, vecops::float64_t, vecops::float64_t>>;
using Fp64ConversionPairs = TypePairList<
    TypePair<vecops::float64_t, int8_t, vecops::float64_t>,
    TypePair<vecops::float64_t, vecops::float16_t, vecops::float64_t>,
    TypePair<vecops::float64_t, vecops::float32_t, vecops::float64_t>>;

template <typename T>
const char* dtype_name() {
  if constexpr (std::is_same_v<T, int8_t>) return "int8";
  if constexpr (std::is_same_v<T, vecops::float16_t>) return "fp16";
  if constexpr (std::is_same_v<T, vecops::bfloat16_t>) return "bf16";
  if constexpr (std::is_same_v<T, vecops::float32_t>) return "fp32";
  if constexpr (std::is_same_v<T, vecops::float64_t>) return "fp64";
  return "unknown";
}

template <typename Case>
std::string shape_name() {
  std::ostringstream os;
  os << Case::rows << "x" << Case::columns;
  return os.str();
}

template <typename InputT>
void fill_input(AlignedVector<InputT>& input) {
  for (std::size_t i = 0; i < input.size(); ++i) {
    input[i] = static_cast<InputT>(static_cast<int>(i % 251) - 125);
  }
}

template <typename InputT, typename OutputT, typename ComputeT>
bool verify_output(
    const AlignedVector<InputT>& input,
    const AlignedVector<OutputT>& output,
    nint_t rows, nint_t columns) {
  for (nint_t row = 0; row < rows; ++row) {
    for (nint_t column = 0; column < columns; ++column) {
      const auto input_index = static_cast<std::size_t>(
          row * columns + column);
      const auto output_index = static_cast<std::size_t>(
          column * rows + row);
      const auto expected = static_cast<OutputT>(
          static_cast<ComputeT>(input[input_index]));
      if (output[output_index] != expected) return false;
    }
  }
  return true;
}

template <typename Implementation>
const char* implementation_name() {
  if constexpr (std::is_same_v<
                    Implementation,
                    kernel::transpose2d_implementation::SME>) {
    return "sme";
  }
  return "vector";
}

template <typename InputT, typename OutputT, typename ComputeT,
          typename InputShape, typename OutputShape>
void run_case(
    benchmark::State& state, InputShape input_shape, OutputShape output_shape,
    nint_t rows, nint_t columns, nint_t elements) {
  AlignedVector<InputT> input(static_cast<std::size_t>(elements));
  AlignedVector<OutputT> output(static_cast<std::size_t>(elements), OutputT{});
  fill_input(input);

  auto input_tensor = make_tensor(
      input.data(), make_layout(input_shape));
  auto output_tensor = make_tensor(
      output.data(), make_layout(output_shape));
  auto operation = ops::make_transpose<ComputeT>(input_tensor, output_tensor);
  ExecutionSession execution{};

  execution.with_region(
      operation, [&](auto& region) VECOPS_INLINE_LAMBDA {
        operation(region);
        if (!verify_output<InputT, OutputT, ComputeT>(
                input, output, rows, columns)) {
          state.SkipWithError("Transpose output verification failed");
          return;
        }
        for (auto _ : state) {
          benchmark::DoNotOptimize(input.data());
          operation(region);
          benchmark::ClobberMemory();
        }
      });

  const auto bytes_per_iteration = static_cast<int64_t>(
      elements * static_cast<nint_t>(sizeof(InputT) + sizeof(OutputT)));
  state.SetItemsProcessed(state.iterations() * elements);
  state.SetBytesProcessed(state.iterations() * bytes_per_iteration);
  state.counters["m"] = benchmark::Counter(double(rows));
  state.counters["n"] = benchmark::Counter(double(columns));
  state.counters["elements"] = benchmark::Counter(double(elements));
  state.counters["working_set_bytes"] = benchmark::Counter(
      double(bytes_per_iteration));
  state.SetLabel(implementation_name<typename decltype(operation)::Implementation>());
}

template <typename Case, typename InputT, typename OutputT, typename ComputeT>
void run_const_case(benchmark::State& state) {
  run_case<InputT, OutputT, ComputeT>(
      state, Case::template input_shape<true>(),
      Case::template output_shape<true>(), Case::rows, Case::columns,
      Case::elements);
}

template <typename InputT, typename OutputT, typename ComputeT>
void run_dynamic_case(
    benchmark::State& state, RuntimeTransposeCase benchmark_case) {
  run_case<InputT, OutputT, ComputeT>(
      state,
      make_shape(Any{benchmark_case.rows}, Any{benchmark_case.columns}),
      make_shape(Any{benchmark_case.columns}, Any{benchmark_case.rows}),
      benchmark_case.rows, benchmark_case.columns, benchmark_case.elements);
}

template <typename InputT, typename OutputT, typename ComputeT,
          nint_t RowAlignment, nint_t ColumnAlignment>
void run_aligned_dynamic_case(
    benchmark::State& state, RuntimeTransposeCase benchmark_case) {
  run_case<InputT, OutputT, ComputeT>(
      state,
      make_shape(
          Dynamic<RowAlignment>{benchmark_case.rows},
          Dynamic<ColumnAlignment>{benchmark_case.columns}),
      make_shape(
          Dynamic<ColumnAlignment>{benchmark_case.columns},
          Dynamic<RowAlignment>{benchmark_case.rows}),
      benchmark_case.rows, benchmark_case.columns, benchmark_case.elements);
}

template <typename Case>
std::string aligned_dynamic_name() {
  return
      "Dynamic<" + std::to_string(Case::row_alignment) + "x" +
      std::to_string(Case::column_alignment) + ">";
}

template <typename Case, typename InputT, typename OutputT, typename ComputeT>
std::string benchmark_name(const char* shape_metadata) {
  return
      "Transpose/" + std::string(Case::GroupType::name) +
      "/shape:" + shape_name<Case>() +
      "/shape_meta:" + shape_metadata +
      "/src_dtype:" + dtype_name<InputT>() +
      "/dst_dtype:" + dtype_name<OutputT>() +
      "/compute_dtype:" + dtype_name<ComputeT>() +
      "/arch:" + VECOPS_BENCH_ARCH_CODE;
}

template <typename Case, typename InputT, typename OutputT, typename ComputeT>
void register_case() {
  const std::string const_name =
      benchmark_name<Case, InputT, OutputT, ComputeT>("Const");
  benchmark::RegisterBenchmark(
      const_name.c_str(),
      &run_const_case<Case, InputT, OutputT, ComputeT>)
      ->Unit(benchmark::kMicrosecond)
      ->MinTime(0.02)
      ->Repetitions(3)
      ->ReportAggregatesOnly(true);

  const std::string aligned_name = benchmark_name<
      Case, InputT, OutputT, ComputeT>(aligned_dynamic_name<Case>().c_str());
  benchmark::RegisterBenchmark(
      aligned_name.c_str(),
      &run_aligned_dynamic_case<
          InputT, OutputT, ComputeT,
          Case::row_alignment, Case::column_alignment>,
      runtime_case<Case>())
      ->Unit(benchmark::kMicrosecond)
      ->MinTime(0.02)
      ->Repetitions(3)
      ->ReportAggregatesOnly(true);

  const std::string dynamic_name =
      benchmark_name<Case, InputT, OutputT, ComputeT>("Any");
  benchmark::RegisterBenchmark(
      dynamic_name.c_str(),
      &run_dynamic_case<InputT, OutputT, ComputeT>, runtime_case<Case>())
      ->Unit(benchmark::kMicrosecond)
      ->MinTime(0.02)
      ->Repetitions(3)
      ->ReportAggregatesOnly(true);
}

template <typename T, typename... Cases>
void register_same_dtype_cases(CaseList<Cases...>) {
  (register_case<Cases, T, T, T>(), ...);
}

template <typename T>
void register_same_dtype() {
  register_same_dtype_cases<T>(TransposeCases{});
}

template <typename T>
void register_compare_same_dtype() {
  register_same_dtype_cases<T>(TransposeCompareCases{});
}

template <typename Case, typename... Pairs>
void register_conversion_pairs(TypePairList<Pairs...>) {
  (register_case<
       Case, typename Pairs::Input, typename Pairs::Output,
       typename Pairs::Compute>(), ...);
}

template <typename... Cases, typename... Pairs>
void register_conversion_cases(
    CaseList<Cases...>, TypePairList<Pairs...> pairs) {
  (register_conversion_pairs<Cases>(pairs), ...);
}

void register_int8_benchmarks();
void register_fp16_benchmarks();
void register_bf16_benchmarks();
void register_fp32_benchmarks();
void register_fp64_benchmarks();
void register_int8_conversion_benchmarks();
void register_fp16_conversion_benchmarks();
void register_fp32_conversion_benchmarks();
void register_fp64_conversion_benchmarks();
void register_conversion_benchmarks();
void register_compare_int8_benchmarks();
void register_compare_fp16_benchmarks();
void register_compare_bf16_benchmarks();
void register_compare_fp32_benchmarks();
void register_compare_fp64_benchmarks();

} // namespace vecops::bench::transpose
