#include "MatmulPackBenchCommon.h"

#include "vecops/gemm/Atoms.h"

namespace vecops::bench::matmul_pack {

template <>
struct AtomName<gemm::SME_BF16F32> {
  static constexpr const char* value = "SME_BF16F32";
};

template <>
struct AtomName<gemm::SME_F16F32> {
  static constexpr const char* value = "SME_F16F32";
};

template <>
struct AtomName<gemm::SME_F32F32> {
  static constexpr const char* value = "SME_F32F32";
};

using SMEI8 = gemm::SME_I8I32<int8_t, uint8_t>;

template <>
struct AtomName<SMEI8> {
  static constexpr const char* value = "SME_I8I32";
};

enum class FusedPipeline {
  FP16ToFP32,
  FP32ToBF16,
  FP32Transform,
  BF16Transform,
};

template <FusedPipeline Pipeline>
struct FusedPipelineTraits;

template <>
struct FusedPipelineTraits<FusedPipeline::FP16ToFP32> {
  using Atom = gemm::SME_F32F32;
  using Memory = float16_t;
  using T = float32_t;
  static constexpr const char* name = "fp16_to_fp32";
  static constexpr bool transform = false;
};

template <>
struct FusedPipelineTraits<FusedPipeline::FP32ToBF16> {
  using Atom = gemm::SME_BF16F32;
  using Memory = float32_t;
  using T = bfloat16_t;
  static constexpr const char* name = "fp32_to_bf16";
  static constexpr bool transform = false;
};

template <>
struct FusedPipelineTraits<FusedPipeline::FP32Transform> {
  using Atom = gemm::SME_F32F32;
  using Memory = float32_t;
  using T = float32_t;
  static constexpr const char* name = "fp32_transform";
  static constexpr bool transform = true;
};

template <>
struct FusedPipelineTraits<FusedPipeline::BF16Transform> {
  using Atom = gemm::SME_BF16F32;
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
  } else {
    static_assert(
        !std::same_as<Implementation, Implementation>,
        "unknown matmul packing benchmark implementation");
    return "unknown";
  }
}

template <typename Atom, gemm::Operand Side, typename Implementation,
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
        using Packing = gemm::packing_t<Atom, Side>;
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

template <FusedPipeline Pipeline, gemm::Operand Side,
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
    if constexpr (Traits::transform) {
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

template <FusedPipeline Pipeline, gemm::Operand Side,
          nint_t Spatial, nint_t K, bool ConstShape,
          typename Implementation>
void register_fused_implementation() {
  using Traits = FusedPipelineTraits<Pipeline>;
  const char* operand = Side == gemm::Operand::A ? "A" : "B";
  const char* shape_meta = ConstShape ? "Const" : "Dyn";
  const std::string name =
      "MatmulPack/fused/pipeline:" + std::string(Traits::name) +
      "/operand:" + operand +
      "/shape:" + std::to_string(Spatial) + "x" + std::to_string(K) +
      "/shape_meta:" + shape_meta +
      "/implementation:" + implementation_name<Implementation>() +
      "/arch:" + VECOPS_BENCH_ARCH_CODE;
  benchmark::RegisterBenchmark(
      name.c_str(),
      &run_fused_case<
          Pipeline, Side, Spatial, K, ConstShape, Implementation>)
      ->Unit(benchmark::kMicrosecond)
      ->MinTime(0.02)
      ->Repetitions(3)
      ->ReportAggregatesOnly(true);
}

template <FusedPipeline Pipeline, gemm::Operand Side,
          nint_t Spatial, nint_t K, bool ConstShape>
void register_fused_shape_meta() {
  register_fused_implementation<
      Pipeline, Side, Spatial, K, ConstShape,
      kernel::matmul_pack_implementation::Vector>();
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

template <FusedPipeline Pipeline, gemm::Operand Side,
          nint_t Spatial, nint_t K>
void register_fused_case() {
  register_fused_shape_meta<Pipeline, Side, Spatial, K, true>();
  register_fused_shape_meta<Pipeline, Side, Spatial, K, false>();
}

} // namespace vecops::bench::matmul_pack

int main(int argc, char** argv) {
  using namespace vecops::bench::matmul_pack;
  register_cases<vecops::gemm::SME_BF16F32>(RepresentativeCases{});
  register_cases<vecops::gemm::SME_F16F32>(DTypeProbeCases{});
  register_cases<vecops::gemm::SME_F32F32>(DTypeProbeCases{});
  register_cases<SMEI8>(DTypeProbeCases{});
  register_fused_case<
      FusedPipeline::FP16ToFP32, vecops::gemm::Operand::A, 128, 3584>();
  register_fused_case<
      FusedPipeline::FP16ToFP32, vecops::gemm::Operand::B, 513, 1000>();
  register_fused_case<
      FusedPipeline::FP32ToBF16, vecops::gemm::Operand::A, 128, 3584>();
  register_fused_case<
      FusedPipeline::FP32ToBF16, vecops::gemm::Operand::B, 4608, 3584>();
  register_fused_case<
      FusedPipeline::FP32Transform, vecops::gemm::Operand::A, 513, 1000>();
  register_fused_case<
      FusedPipeline::FP32Transform, vecops::gemm::Operand::B, 513, 1000>();
  register_fused_case<
      FusedPipeline::BF16Transform, vecops::gemm::Operand::A, 513, 1000>();
  register_fused_case<
      FusedPipeline::BF16Transform, vecops::gemm::Operand::B, 513, 1000>();
  return run_benchmarks(argc, argv, "matmul_pack_sme");
}
