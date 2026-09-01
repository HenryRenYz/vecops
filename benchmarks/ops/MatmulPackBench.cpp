// @vecops-target-shards-x86: 8
// @vecops-target-shards-ARM: 10

#include "vecops/Features.h"

#if defined(ARCH_X86_FAMILY)

#include "MatmulPackBenchCommon.h"

#include "vecops/matmul/Atom.h"

namespace vecops::bench::matmul_pack {
template <int Shard>
void register_matmul_pack_shard();
}

#if defined(VECOPS_TARGET_SHARD_ACTIVE)

namespace vecops::bench::matmul_pack {

template <>
struct AtomName<::vecops::matmul::AMX_BF16F32> {
  static constexpr const char* value = "AMX_BF16F32";
};

template <>
struct AtomName<::vecops::matmul::AMX_F16F32> {
  static constexpr const char* value = "AMX_F16F32";
};

using AMXI8S8S8 = ::vecops::matmul::AMX_I8I32<int8_t, int8_t>;
using AMXI8S8U8 = ::vecops::matmul::AMX_I8I32<int8_t, uint8_t>;
using AMXI8U8S8 = ::vecops::matmul::AMX_I8I32<uint8_t, int8_t>;
using AMXI8U8U8 = ::vecops::matmul::AMX_I8I32<uint8_t, uint8_t>;

template <>
struct AtomName<AMXI8S8S8> {
  static constexpr const char* value = "AMX_I8I32_s8s8";
};
template <>
struct AtomName<AMXI8S8U8> {
  static constexpr const char* value = "AMX_I8I32_s8u8";
};
template <>
struct AtomName<AMXI8U8S8> {
  static constexpr const char* value = "AMX_I8I32_u8s8";
};
template <>
struct AtomName<AMXI8U8U8> {
  static constexpr const char* value = "AMX_I8I32_u8u8";
};

enum class FusedPipeline {
  FP32ToBF16,
  FP32ToFP16,
  BF16Transform,
  FP32ToS8,
  FP32ToU8,
};

template <FusedPipeline Pipeline>
struct FusedPipelineTraits;

template <>
struct FusedPipelineTraits<FusedPipeline::FP32ToBF16> {
  using Atom = ::vecops::matmul::AMX_BF16F32;
  using Memory = float32_t;
  using T = bfloat16_t;
  static constexpr const char* name = "fp32_to_bf16";
  static constexpr bool transform = false;
};

template <>
struct FusedPipelineTraits<FusedPipeline::FP32ToFP16> {
  using Atom = ::vecops::matmul::AMX_F16F32;
  using Memory = float32_t;
  using T = float16_t;
  static constexpr const char* name = "fp32_to_fp16";
  static constexpr bool transform = false;
};

template <>
struct FusedPipelineTraits<FusedPipeline::BF16Transform> {
  using Atom = ::vecops::matmul::AMX_BF16F32;
  using Memory = bfloat16_t;
  using T = bfloat16_t;
  static constexpr const char* name = "bf16_transform";
  static constexpr bool transform = true;
};

template <>
struct FusedPipelineTraits<FusedPipeline::FP32ToS8> {
  using Atom = AMXI8S8S8;
  using Memory = float32_t;
  using T = int8_t;
  static constexpr const char* name = "fp32_to_s8_quantize";
  static constexpr bool transform = true;
};

template <>
struct FusedPipelineTraits<FusedPipeline::FP32ToU8> {
  using Atom = AMXI8U8U8;
  using Memory = float32_t;
  using T = uint8_t;
  static constexpr const char* name = "fp32_to_u8_quantize";
  static constexpr bool transform = true;
};

template <typename Atom, ::vecops::matmul::Operand Side,
          typename InputSpec, typename OutputSpec>
VECOPS_ALWAYS_INLINE void run_forced_vector_pack(
    ExecutionSession& execution,
    const InputSpec& input, const OutputSpec& output) {
  using Implementation = kernel::matmul_pack_implementation::Vector;
  struct Requirement {
    using ResourceRequirements =
        kernel::matmul_pack_implementation::resource_requirements_t<
            Atom, Side, Implementation>;
  } requirement;
  execution.with_resources(
      requirement,
      [&](auto& active) VECOPS_INLINE_LAMBDA {
        using Packing = ::vecops::matmul::packing_t<Atom, Side>;
        using InputPolicy = tensor::InputAccessPolicy<
            Packing::VectorAxis, 1, tensor::AccessPlan::direct>;
        using OutputPolicy = tensor::OutputAccessPolicy<
            OutputSpec::OutputTensor::Ndim - 1,
            tensor::AccessPlan::direct>;
        kernel::with_operands(
            active,
            tensor::operand(input, InputPolicy{}),
            tensor::operand(output, OutputPolicy{}),
            [&](auto& source, auto& destination)
                VECOPS_INLINE_LAMBDA {
              kernel::matmul_pack_bound<Atom, Side>(
                  active, source, destination, Implementation{});
              destination.commit();
            });
      });
}

template <FusedPipeline Pipeline, ::vecops::matmul::Operand Side,
          nint_t Spatial, nint_t K, bool ConstShape>
void run_fused_case(benchmark::State& state) {
  using Traits = FusedPipelineTraits<Pipeline>;
  using Atom = typename Traits::Atom;
  using Memory = typename Traits::Memory;
  using T = typename Traits::T;
  std::vector<Memory> input(static_cast<std::size_t>(Spatial * K));
  fill_input(input);
  const auto shape = [&] {
    if constexpr (ConstShape) return make_shape(cint<Spatial>, cint<K>);
    else return make_shape(Any{Spatial}, Any{K});
  }();
  auto input_layout = make_layout(shape);
  auto input_tensor = make_tensor(input.data(), input_layout);
  const auto input_spec = [&] {
    if constexpr (
        Pipeline == FusedPipeline::FP32ToS8 ||
        Pipeline == FusedPipeline::FP32ToU8) {
      auto transform = make_elementwise_vec_transform<float32_t, float32_t>(
          []<vec::VectorTag Tag>(Tag tag, vec::Vec<Tag> value)
              VECOPS_KERNEL_LAMBDA {
            return vec::mul(tag, value, vec::fill(tag, float32_t{4}));
          });
      return tensor::input<T>(input_tensor, transform);
    } else if constexpr (Traits::transform) {
      auto transform = make_elementwise_vec_transform<T, T>(
          []<vec::VectorTag Tag>(Tag tag, vec::Vec<Tag> value)
              VECOPS_KERNEL_LAMBDA {
            return vec::add(tag, value, vec::fill(tag, T{1}));
          });
      return tensor::input<T>(input_tensor, transform);
    } else {
      return tensor::input<T>(input_tensor);
    }
  }();
  auto output_layout = ops::matmul_packed_layout<
      Atom, Side>(input_layout);
  const nint_t output_elements = numel(output_layout);
  const nint_t output_bytes = output_elements * nint_t{sizeof(T)};
  kernel::Workspace storage(output_bytes + 64);
  auto view = storage.view();
  auto* output = static_cast<T*>(view.allocate(output_bytes, 64));
  auto output_spec = tensor::output<T>(make_tensor(output, output_layout));
  ExecutionSession execution{};
  run_forced_vector_pack<Atom, Side>(execution, input_spec, output_spec);

  for (auto _ : state) {
    benchmark::DoNotOptimize(input.data());
    run_forced_vector_pack<Atom, Side>(execution, input_spec, output_spec);
    benchmark::DoNotOptimize(output);
    benchmark::ClobberMemory();
  }
  const int64_t input_bytes =
      static_cast<int64_t>(input.size() * sizeof(Memory));
  state.SetItemsProcessed(state.iterations() * Spatial * K);
  state.SetBytesProcessed(
      state.iterations() * (input_bytes + output_bytes));
}

template <FusedPipeline Pipeline, ::vecops::matmul::Operand Side,
          nint_t Spatial, nint_t K, bool ConstShape>
void register_fused_shape_meta() {
  using Traits = FusedPipelineTraits<Pipeline>;
  const char* operand = Side == ::vecops::matmul::Operand::A ? "A" : "B";
  const char* shape_meta = ConstShape ? "Const" : "Dynamic";
  const std::string name =
      "MatmulPack/fused/pipeline:" + std::string(Traits::name) +
      "/operand:" + operand +
      "/shape:" + std::to_string(Spatial) + "x" + std::to_string(K) +
      "/extent:" + shape_meta +
      "/implementation:Vector/arch:" + VECOPS_BENCH_ARCH_CODE;
  auto* registered = benchmark::RegisterBenchmark(
      name.c_str(),
      &run_fused_case<Pipeline, Side, Spatial, K, ConstShape>)
      ->Unit(benchmark::kMicrosecond);
  ::vecops::bench::configure_registered_benchmark(
      registered, 0.02, 3)->ReportAggregatesOnly(true);
}

template <FusedPipeline Pipeline, ::vecops::matmul::Operand Side,
          nint_t Spatial, nint_t K>
void register_fused_case() {
  register_fused_shape_meta<Pipeline, Side, Spatial, K, true>();
  register_fused_shape_meta<Pipeline, Side, Spatial, K, false>();
}

enum class CompensationMode {
  PackOnly,
  SeparateSum,
  Fused,
};

enum class CompensationInput {
  TransformedFP32,
  DirectS8,
};

template <CompensationMode Mode>
constexpr const char* compensation_mode_name() {
  if constexpr (Mode == CompensationMode::PackOnly) return "pack_only";
  if constexpr (Mode == CompensationMode::SeparateSum)
    return "pack_then_separate_sum";
  return "fused_pack_and_sum";
}

template <CompensationMode Mode, CompensationInput Input,
          nint_t N, nint_t K, bool ConstShape>
void run_compensation_case(benchmark::State& state) {
  using Atom = ::vecops::matmul::AMX_I8I32<uint8_t, int8_t>;
  using Memory = std::conditional_t<
      Input == CompensationInput::DirectS8, int8_t, float32_t>;
  constexpr int32_t ZeroPointA = 3;
  std::vector<Memory> input(static_cast<std::size_t>(N * K));
  std::vector<int8_t> reference(static_cast<std::size_t>(N * K));
  for (nint_t i = 0; i < N * K; ++i) {
    const auto quantized = static_cast<int8_t>((i * 5) % 63 - 31);
    if constexpr (Input == CompensationInput::DirectS8) {
      input[static_cast<std::size_t>(i)] = quantized;
    } else {
      input[static_cast<std::size_t>(i)] =
          static_cast<float32_t>(quantized) * 0.25f;
    }
    reference[static_cast<std::size_t>(i)] = quantized;
  }
  const auto shape = [&] {
    if constexpr (ConstShape) return make_shape(cint<N>, cint<K>);
    else return make_shape(Any{N}, Any{K});
  }();
  auto input_layout = make_layout(shape);
  auto input_tensor = make_tensor(input.data(), input_layout);
  const auto input_spec = [&] {
    if constexpr (Input == CompensationInput::DirectS8) {
      return tensor::input<int8_t>(input_tensor);
    } else {
      auto transform = make_elementwise_vec_transform<float32_t, float32_t>(
          []<vec::VectorTag Tag>(Tag tag, vec::Vec<Tag> value)
              VECOPS_KERNEL_LAMBDA {
            return vec::mul(tag, value, vec::fill(tag, float32_t{4}));
          });
      return tensor::input<int8_t>(input_tensor, transform);
    }
  }();
  auto output_layout = ops::matmul_packed_layout<
      Atom, ::vecops::matmul::Operand::B>(input_layout);
  const nint_t output_bytes =
      numel(output_layout) * static_cast<nint_t>(sizeof(int8_t));
  kernel::Workspace storage(output_bytes + 64);
  auto view = storage.view();
  auto* output = static_cast<int8_t*>(view.allocate(output_bytes, 64));
  auto output_tensor = make_tensor(output, output_layout);
  std::vector<int32_t> compensation(static_cast<std::size_t>(N));
  auto compensation_tensor = make_tensor(
      compensation.data(), make_layout(make_shape(Any{N})));
  ExecutionSession execution{};

  auto run_separate_sum = [&] VECOPS_INLINE_LAMBDA {
    ops::matmul_pack_details::run_matmul_pack<Atom, ::vecops::matmul::Operand::B>(
        execution, input_spec, output_tensor);
    for (nint_t row = 0; row < N; ++row) {
      int32_t sum = 0;
      for (nint_t kk = 0; kk < K; ++kk) {
        const auto value = input[static_cast<std::size_t>(row * K + kk)];
        if constexpr (Input == CompensationInput::DirectS8)
          sum += static_cast<int8_t>(value);
        else
          sum += static_cast<int8_t>(value * 4.0f);
      }
      compensation[static_cast<std::size_t>(row)] = -ZeroPointA * sum;
    }
  };
  auto run_once = [&] VECOPS_INLINE_LAMBDA {
    if constexpr (Mode == CompensationMode::PackOnly) {
      ops::matmul_pack_details::run_matmul_pack<Atom, ::vecops::matmul::Operand::B>(
          execution, input_spec, output_tensor);
    } else if constexpr (Mode == CompensationMode::SeparateSum) {
      run_separate_sum();
    } else {
      ops::matmul_pack_details::run_matmul_pack_b_compensated<Atom>(
          execution, input_spec, output_tensor,
          compensation_tensor, ZeroPointA);
    }
  };
  run_once();
  if constexpr (Mode != CompensationMode::PackOnly) {
    for (nint_t row = 0; row < N; ++row) {
      int32_t sum = 0;
      for (nint_t kk = 0; kk < K; ++kk)
        sum += reference[static_cast<std::size_t>(row * K + kk)];
      if (compensation[static_cast<std::size_t>(row)] != -ZeroPointA * sum) {
        state.SkipWithError("column compensation verification failed");
        return;
      }
    }
  }

  for (auto _ : state) {
    benchmark::DoNotOptimize(input.data());
    run_once();
    benchmark::DoNotOptimize(output);
    if constexpr (Mode != CompensationMode::PackOnly)
      benchmark::DoNotOptimize(compensation.data());
    benchmark::ClobberMemory();
  }
  const int64_t input_bytes =
      static_cast<int64_t>(input.size() * sizeof(Memory));
  const int64_t sidecar_bytes = Mode == CompensationMode::PackOnly
      ? 0
      : static_cast<int64_t>(N * sizeof(int32_t));
  const int64_t input_passes =
      Mode == CompensationMode::SeparateSum ? 2 : 1;
  state.SetItemsProcessed(state.iterations() * N * K);
  state.SetBytesProcessed(
      state.iterations() *
      (input_passes * input_bytes + output_bytes + sidecar_bytes));
  state.counters["sidecar_bytes"] =
      benchmark::Counter(double(sidecar_bytes));
}

template <CompensationMode Mode, CompensationInput Input,
          nint_t N, nint_t K, bool ConstShape>
void register_compensation_shape_meta() {
  const char* shape_meta = ConstShape ? "Const" : "Dynamic";
  const char* pipeline = Input == CompensationInput::DirectS8
      ? "direct_s8"
      : "fp32_to_s8_quantize";
  const std::string name =
      "MatmulPack/compensation/mode:" +
      std::string(compensation_mode_name<Mode>()) +
      "/pipeline:" + pipeline + "/operand:B/shape:" +
      std::to_string(N) + "x" + std::to_string(K) +
      "/extent:" + shape_meta +
      "/implementation:Vector/arch:" + VECOPS_BENCH_ARCH_CODE;
  auto* registered = benchmark::RegisterBenchmark(
      name.c_str(), &run_compensation_case<Mode, Input, N, K, ConstShape>)
      ->Unit(benchmark::kMicrosecond);
  ::vecops::bench::configure_registered_benchmark(
      registered, 0.02, 3)->ReportAggregatesOnly(true);
}

template <CompensationMode Mode, CompensationInput Input,
          nint_t N, nint_t K>
void register_compensation_case() {
  register_compensation_shape_meta<Mode, Input, N, K, true>();
  register_compensation_shape_meta<Mode, Input, N, K, false>();
}

template <CompensationInput Input, nint_t N, nint_t K>
void register_all_compensation_modes() {
  register_compensation_case<CompensationMode::PackOnly, Input, N, K>();
  register_compensation_case<CompensationMode::SeparateSum, Input, N, K>();
  register_compensation_case<CompensationMode::Fused, Input, N, K>();
}

template <int Shard>
void register_matmul_pack_shard() {
  static_assert(0 <= Shard && Shard < 8);
  if constexpr (Shard == 0) {
    register_cases<::vecops::matmul::AMX_BF16F32>(RepresentativeCases{});
  } else if constexpr (Shard == 1) {
    register_cases<::vecops::matmul::AMX_F16F32>(SecondaryCases{});
  } else if constexpr (Shard == 2) {
    register_cases<AMXI8S8S8>(SecondaryCases{});
  } else if constexpr (Shard == 3) {
    register_cases<AMXI8S8U8>(SecondaryCases{});
  } else if constexpr (Shard == 4) {
    register_cases<AMXI8U8S8>(SecondaryCases{});
  } else if constexpr (Shard == 5) {
    register_cases<AMXI8U8U8>(SecondaryCases{});
  } else if constexpr (Shard == 6) {
    register_fused_case<
        FusedPipeline::FP32ToBF16, ::vecops::matmul::Operand::A, 128, 3584>();
    register_fused_case<
        FusedPipeline::FP32ToBF16, ::vecops::matmul::Operand::B, 4608, 3584>();
    register_fused_case<
        FusedPipeline::FP32ToFP16, ::vecops::matmul::Operand::A, 128, 3584>();
    register_fused_case<
        FusedPipeline::FP32ToFP16, ::vecops::matmul::Operand::B, 513, 1000>();
    register_fused_case<
        FusedPipeline::BF16Transform, ::vecops::matmul::Operand::A, 513, 1000>();
    register_fused_case<
        FusedPipeline::BF16Transform, ::vecops::matmul::Operand::B, 513, 1000>();
    register_fused_case<FusedPipeline::FP32ToS8, ::vecops::matmul::Operand::A, 64, 256>();
    register_fused_case<FusedPipeline::FP32ToS8, ::vecops::matmul::Operand::B, 64, 256>();
    register_fused_case<
        FusedPipeline::FP32ToU8, ::vecops::matmul::Operand::A, 513, 1000>();
    register_fused_case<
        FusedPipeline::FP32ToU8, ::vecops::matmul::Operand::B, 513, 1000>();
  } else {
    register_all_compensation_modes<
        CompensationInput::TransformedFP32, 513, 1000>();
    register_all_compensation_modes<
        CompensationInput::TransformedFP32, 4608, 3584>();
    register_all_compensation_modes<CompensationInput::DirectS8, 16, 64>();
    register_all_compensation_modes<CompensationInput::DirectS8, 16, 128>();
    register_all_compensation_modes<CompensationInput::DirectS8, 32, 64>();
    register_all_compensation_modes<CompensationInput::DirectS8, 16, 256>();
    register_all_compensation_modes<CompensationInput::DirectS8, 32, 128>();
    register_all_compensation_modes<CompensationInput::DirectS8, 64, 64>();
    register_all_compensation_modes<CompensationInput::DirectS8, 19, 67>();
    register_all_compensation_modes<CompensationInput::DirectS8, 64, 256>();
  }
}

} // namespace vecops::bench::matmul_pack

template void vecops::bench::matmul_pack::register_matmul_pack_shard<
    VECOPS_TARGET_SHARD_INDEX>();

#else

int main(int argc, char** argv) {
  []<std::size_t... I>(std::index_sequence<I...>) {
    (vecops::bench::matmul_pack::register_matmul_pack_shard<
         static_cast<int>(I)>(), ...);
  }(std::make_index_sequence<8>{});
  return vecops::bench::matmul_pack::run_benchmarks(
      argc, argv, "matmul_pack");
}

#endif

#else

#include "MatmulPackBenchCommon.h"

#include "vecops/matmul/Atom.h"

namespace vecops::bench::matmul_pack {
template <int Shard>
void register_matmul_pack_shard();
}

#if defined(VECOPS_TARGET_SHARD_ACTIVE)

namespace vecops::bench::matmul_pack {

template <>
struct AtomName<::vecops::matmul::SME_BF16F32> {
  static constexpr const char* value = "SME_BF16F32";
};

template <>
struct AtomName<::vecops::matmul::SME_F16F32> {
  static constexpr const char* value = "SME_F16F32";
};

template <>
struct AtomName<::vecops::matmul::SME_F32F32> {
  static constexpr const char* value = "SME_F32F32";
};

#if defined(HAS_SME_F64F64)
template <>
struct AtomName<::vecops::matmul::SME_F64F64> {
  static constexpr const char* value = "SME_F64F64";
};
#endif

using SMEI8S8S8 = ::vecops::matmul::SME_I8I32<int8_t, int8_t>;
using SMEI8S8U8 = ::vecops::matmul::SME_I8I32<int8_t, uint8_t>;
using SMEI8U8S8 = ::vecops::matmul::SME_I8I32<uint8_t, int8_t>;
using SMEI8U8U8 = ::vecops::matmul::SME_I8I32<uint8_t, uint8_t>;

template <>
struct AtomName<SMEI8S8S8> {
  static constexpr const char* value = "SME_I8I32_s8s8";
};
template <>
struct AtomName<SMEI8S8U8> {
  static constexpr const char* value = "SME_I8I32_s8u8";
};
template <>
struct AtomName<SMEI8U8S8> {
  static constexpr const char* value = "SME_I8I32_u8s8";
};
template <>
struct AtomName<SMEI8U8U8> {
  static constexpr const char* value = "SME_I8I32_u8u8";
};

enum class FusedPipeline {
  FP16ToFP32,
  BF16ToFP32,
  FP32ToBF16,
  FP32ToFP16,
  FP32ToS8,
  FP32ToU8,
#if defined(HAS_SME_F64F64)
  FP32ToFP64,
#endif
  FP32Transform,
  BF16Transform,
};

template <FusedPipeline Pipeline>
struct FusedPipelineTraits;

template <>
struct FusedPipelineTraits<FusedPipeline::FP16ToFP32> {
  using Atom = ::vecops::matmul::SME_F32F32;
  using Memory = float16_t;
  using T = float32_t;
  static constexpr const char* name = "fp16_to_fp32";
  static constexpr bool transform = false;
};

template <>
struct FusedPipelineTraits<FusedPipeline::BF16ToFP32> {
  using Atom = ::vecops::matmul::SME_F32F32;
  using Memory = bfloat16_t;
  using T = float32_t;
  static constexpr const char* name = "bf16_to_fp32";
  static constexpr bool transform = false;
};

template <>
struct FusedPipelineTraits<FusedPipeline::FP32ToBF16> {
  using Atom = ::vecops::matmul::SME_BF16F32;
  using Memory = float32_t;
  using T = bfloat16_t;
  static constexpr const char* name = "fp32_to_bf16";
  static constexpr bool transform = false;
};

template <>
struct FusedPipelineTraits<FusedPipeline::FP32ToFP16> {
  using Atom = ::vecops::matmul::SME_F16F32;
  using Memory = float32_t;
  using T = float16_t;
  static constexpr const char* name = "fp32_to_fp16";
  static constexpr bool transform = false;
};

template <>
struct FusedPipelineTraits<FusedPipeline::FP32ToS8> {
  using Atom = SMEI8S8S8;
  using Memory = float32_t;
  using T = int8_t;
  static constexpr const char* name = "fp32_to_s8";
  static constexpr bool transform = false;
};

template <>
struct FusedPipelineTraits<FusedPipeline::FP32ToU8> {
  using Atom = SMEI8U8U8;
  using Memory = float32_t;
  using T = uint8_t;
  static constexpr const char* name = "fp32_to_u8";
  static constexpr bool transform = false;
};

#if defined(HAS_SME_F64F64)
template <>
struct FusedPipelineTraits<FusedPipeline::FP32ToFP64> {
  using Atom = ::vecops::matmul::SME_F64F64;
  using Memory = float32_t;
  using T = float64_t;
  static constexpr const char* name = "fp32_to_fp64";
  static constexpr bool transform = false;
};
#endif

template <>
struct FusedPipelineTraits<FusedPipeline::FP32Transform> {
  using Atom = ::vecops::matmul::SME_F32F32;
  using Memory = float32_t;
  using T = float32_t;
  static constexpr const char* name = "fp32_transform";
  static constexpr bool transform = true;
};

template <>
struct FusedPipelineTraits<FusedPipeline::BF16Transform> {
  using Atom = ::vecops::matmul::SME_BF16F32;
  using Memory = bfloat16_t;
  using T = bfloat16_t;
  static constexpr const char* name = "bf16_transform";
  static constexpr bool transform = true;
};

template <typename Implementation>
const char* implementation_name() {
  if constexpr (std::same_as<
                    Implementation,
                    kernel::matmul_pack_implementation::Vector>) {
    return "SVE";
  } else if constexpr (std::same_as<
                           Implementation,
                           kernel::matmul_pack_implementation::SMEPostprocess>) {
    return "SME_post";
  } else if constexpr (std::same_as<
                           Implementation,
                           kernel::matmul_pack_implementation::SMEStagedTransform>) {
    return "SME_staged";
  } else if constexpr (std::same_as<
                           Implementation,
                           kernel::matmul_pack_implementation::SMEStagedFP16ToFP32>) {
    return "SME_staged_conversion";
  } else if constexpr (std::same_as<
                           Implementation,
                           kernel::matmul_pack_implementation::SMEFP32ToFP64>) {
    return "SME_fp32_to_fp64";
  } else {
    static_assert(
        !std::same_as<Implementation, Implementation>,
        "unknown matmul packing benchmark implementation");
    return "unknown";
  }
}

template <typename Atom, ::vecops::matmul::Operand Side, typename Implementation,
          typename InputSpec, typename OutputSpec>
VECOPS_ALWAYS_INLINE void run_forced_pack(
    ExecutionSession& execution,
    const InputSpec& input, const OutputSpec& output) {
  struct Requirement {
    using ResourceRequirements =
        kernel::matmul_pack_implementation::resource_requirements_t<
            Atom, Side, Implementation>;
  } requirement;
  execution.with_resources(
      requirement,
      [&](auto& active) VECOPS_INLINE_LAMBDA {
        using Packing = ::vecops::matmul::packing_t<Atom, Side>;
        using InputPolicy = tensor::InputAccessPolicy<
            Packing::VectorAxis, 1, tensor::AccessPlan::direct>;
        using OutputPolicy = tensor::OutputAccessPolicy<
            OutputSpec::OutputTensor::Ndim - 1,
            tensor::AccessPlan::direct>;
        kernel::with_operands(
            active,
            tensor::operand(input, InputPolicy{}),
            tensor::operand(output, OutputPolicy{}),
            [&](auto& source, auto& destination)
                VECOPS_INLINE_LAMBDA {
              kernel::matmul_pack_bound<Atom, Side>(
                  active, source, destination, Implementation{});
              destination.commit();
            });
      });
}

template <FusedPipeline Pipeline, ::vecops::matmul::Operand Side,
          nint_t Spatial, nint_t K, bool ConstShape,
          typename Implementation>
void run_fused_case(benchmark::State& state) {
  using Traits = FusedPipelineTraits<Pipeline>;
  using Atom = typename Traits::Atom;
  using Memory = typename Traits::Memory;
  using T = typename Traits::T;
  std::vector<Memory> input(static_cast<std::size_t>(Spatial * K));
  fill_input(input);
  const auto shape = [&] {
    if constexpr (ConstShape) return make_shape(cint<Spatial>, cint<K>);
    else return make_shape(Any{Spatial}, Any{K});
  }();
  auto input_layout = make_layout(shape);
  auto input_tensor = make_tensor(input.data(), input_layout);
  const auto input_spec = [&] {
    if constexpr (
        Pipeline == FusedPipeline::FP32ToS8 ||
        Pipeline == FusedPipeline::FP32ToU8) {
      auto transform = make_elementwise_vec_transform<float32_t, float32_t>(
          []<vec::VectorTag Tag>(Tag tag, vec::Vec<Tag> value)
              VECOPS_KERNEL_LAMBDA {
            return vec::mul(tag, value, vec::fill(tag, 4.0f));
          });
      return tensor::input<T>(input_tensor, transform);
    } else if constexpr (Traits::transform) {
      auto transform = make_elementwise_vec_transform<T, T>(
          []<vec::VectorTag Tag>(Tag tag, vec::Vec<Tag> value)
              VECOPS_KERNEL_LAMBDA {
            return vec::add(tag, value, vec::fill(tag, T{1}));
          });
      return tensor::input<T>(input_tensor, transform);
    } else {
      return tensor::input<T>(input_tensor);
    }
  }();
  auto output_layout = ops::matmul_packed_layout<Atom, Side>(input_layout);
  const nint_t output_elements = numel(output_layout);
  const nint_t output_bytes = output_elements * nint_t{sizeof(T)};
  kernel::Workspace storage(output_bytes + 64);
  auto view = storage.view();
  auto* output = static_cast<T*>(view.allocate(output_bytes, 64));
  auto output_spec = tensor::output<T>(
      make_tensor(output, output_layout));
  ExecutionSession execution{};
  run_forced_pack<Atom, Side, Implementation>(
      execution, input_spec, output_spec);

  for (auto _ : state) {
    benchmark::DoNotOptimize(input.data());
    run_forced_pack<Atom, Side, Implementation>(
        execution, input_spec, output_spec);
    benchmark::DoNotOptimize(output);
    benchmark::ClobberMemory();
  }
  const int64_t input_bytes =
      static_cast<int64_t>(input.size() * sizeof(Memory));
  state.SetItemsProcessed(state.iterations() * Spatial * K);
  state.SetBytesProcessed(
      state.iterations() * (input_bytes + output_bytes));
}

template <FusedPipeline Pipeline, ::vecops::matmul::Operand Side,
          nint_t Spatial, nint_t K, bool ConstShape,
          typename Implementation>
void register_fused_implementation() {
  using Traits = FusedPipelineTraits<Pipeline>;
  const char* operand = Side == ::vecops::matmul::Operand::A ? "A" : "B";
  const char* shape_meta = ConstShape ? "Const" : "Dynamic";
  const std::string name =
      "MatmulPack/fused/pipeline:" + std::string(Traits::name) +
      "/operand:" + operand +
      "/shape:" + std::to_string(Spatial) + "x" + std::to_string(K) +
      "/extent:" + shape_meta +
      "/implementation:" + implementation_name<Implementation>() +
      "/arch:" + VECOPS_BENCH_ARCH_CODE;
  auto* registered = benchmark::RegisterBenchmark(
      name.c_str(),
      &run_fused_case<
          Pipeline, Side, Spatial, K, ConstShape, Implementation>)
      ->Unit(benchmark::kMicrosecond);
  ::vecops::bench::configure_registered_benchmark(
      registered, 0.02, 3)->ReportAggregatesOnly(true);
}

template <FusedPipeline Pipeline, ::vecops::matmul::Operand Side,
          nint_t Spatial, nint_t K, bool ConstShape>
void register_fused_shape_meta() {
  register_fused_implementation<
      Pipeline, Side, Spatial, K, ConstShape,
      kernel::matmul_pack_implementation::Vector>();
#if defined(HAS_SME_F64F64)
  if constexpr (Pipeline == FusedPipeline::FP32ToFP64) {
    register_fused_implementation<
        Pipeline, Side, Spatial, K, ConstShape,
        kernel::matmul_pack_implementation::SMEFP32ToFP64>();
  } else
#endif
  if constexpr (FusedPipelineTraits<Pipeline>::transform) {
    register_fused_implementation<
        Pipeline, Side, Spatial, K, ConstShape,
        kernel::matmul_pack_implementation::SMEStagedTransform>();
  } else if constexpr (Pipeline == FusedPipeline::FP16ToFP32) {
    register_fused_implementation<
        Pipeline, Side, Spatial, K, ConstShape,
        kernel::matmul_pack_implementation::SMEStagedFP16ToFP32>();
  } else {
    register_fused_implementation<
        Pipeline, Side, Spatial, K, ConstShape,
        kernel::matmul_pack_implementation::SMEPostprocess>();
  }
}

template <FusedPipeline Pipeline, ::vecops::matmul::Operand Side,
          nint_t Spatial, nint_t K>
void register_fused_case() {
  register_fused_shape_meta<Pipeline, Side, Spatial, K, true>();
  register_fused_shape_meta<Pipeline, Side, Spatial, K, false>();
}

template <int Shard>
void register_matmul_pack_shard() {
  static_assert(0 <= Shard && Shard < 10);
  if constexpr (Shard == 0) {
    register_cases<::vecops::matmul::SME_BF16F32>(RepresentativeCases{});
  } else if constexpr (Shard == 1) {
    register_cases<::vecops::matmul::SME_F16F32>(SecondaryCases{});
  } else if constexpr (Shard == 2) {
    register_cases<::vecops::matmul::SME_F32F32>(SecondaryCases{});
  } else if constexpr (Shard == 3) {
    register_cases<SMEI8S8S8>(SecondaryCases{});
  } else if constexpr (Shard == 4) {
    register_cases<SMEI8S8U8>(SecondaryCases{});
  } else if constexpr (Shard == 5) {
    register_cases<SMEI8U8S8>(SecondaryCases{});
  } else if constexpr (Shard == 6) {
    register_cases<SMEI8U8U8>(SecondaryCases{});
  } else if constexpr (Shard == 7) {
#if defined(HAS_SME_F64F64)
    register_cases<::vecops::matmul::SME_F64F64>(SecondaryCases{});
    register_fused_case<
        FusedPipeline::FP32ToFP64, ::vecops::matmul::Operand::A, 35, 17>();
    register_fused_case<
        FusedPipeline::FP32ToFP64, ::vecops::matmul::Operand::B, 35, 17>();
    register_fused_case<
        FusedPipeline::FP32ToFP64, ::vecops::matmul::Operand::A, 64, 256>();
    register_fused_case<
        FusedPipeline::FP32ToFP64, ::vecops::matmul::Operand::B, 256, 256>();
#endif
  } else if constexpr (Shard == 8) {
    register_fused_case<
        FusedPipeline::FP16ToFP32, ::vecops::matmul::Operand::A, 128, 3584>();
    register_fused_case<
        FusedPipeline::FP16ToFP32, ::vecops::matmul::Operand::B, 513, 1000>();
    register_fused_case<
        FusedPipeline::BF16ToFP32, ::vecops::matmul::Operand::A, 128, 3584>();
    register_fused_case<
        FusedPipeline::BF16ToFP32, ::vecops::matmul::Operand::B, 513, 1000>();
    register_fused_case<
        FusedPipeline::FP32ToBF16, ::vecops::matmul::Operand::A, 128, 3584>();
    register_fused_case<
        FusedPipeline::FP32ToBF16, ::vecops::matmul::Operand::B, 4608, 3584>();
    register_fused_case<
        FusedPipeline::FP32ToFP16, ::vecops::matmul::Operand::A, 128, 3584>();
    register_fused_case<
        FusedPipeline::FP32ToFP16, ::vecops::matmul::Operand::B, 513, 1000>();
  } else {
    register_fused_case<FusedPipeline::FP32ToS8, ::vecops::matmul::Operand::A, 35, 67>();
    register_fused_case<FusedPipeline::FP32ToS8, ::vecops::matmul::Operand::B, 35, 67>();
    register_fused_case<FusedPipeline::FP32ToU8, ::vecops::matmul::Operand::A, 35, 67>();
    register_fused_case<FusedPipeline::FP32ToU8, ::vecops::matmul::Operand::B, 35, 67>();
    register_fused_case<
        FusedPipeline::FP32Transform, ::vecops::matmul::Operand::A, 513, 1000>();
    register_fused_case<
        FusedPipeline::FP32Transform, ::vecops::matmul::Operand::B, 513, 1000>();
    register_fused_case<
        FusedPipeline::BF16Transform, ::vecops::matmul::Operand::A, 513, 1000>();
    register_fused_case<
        FusedPipeline::BF16Transform, ::vecops::matmul::Operand::B, 513, 1000>();
  }
}

} // namespace vecops::bench::matmul_pack

template void vecops::bench::matmul_pack::register_matmul_pack_shard<
    VECOPS_TARGET_SHARD_INDEX>();

#else

int main(int argc, char** argv) {
  []<std::size_t... I>(std::index_sequence<I...>) {
    (vecops::bench::matmul_pack::register_matmul_pack_shard<
         static_cast<int>(I)>(), ...);
  }(std::make_index_sequence<10>{});
  return vecops::bench::matmul_pack::run_benchmarks(
      argc, argv, "matmul_pack");
}

#endif

#endif
