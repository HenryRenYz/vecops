//
// Copyright (c) vecops contributors.
//

#ifndef VECOPS_BENCHMARKS_OPS_MATMUL_SCENARIO_BENCH_COMMON_H
#define VECOPS_BENCHMARKS_OPS_MATMUL_SCENARIO_BENCH_COMMON_H

#include "MatmulBenchCommon.h"

#include "vecops/matmul/Quantization.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <limits>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

namespace vecops::bench::matmul {

template <typename Config,
          meta::ValueInput M, meta::ValueInput N, meta::ValueInput K,
          tensor::InputOperand A, tensor::InputOperand B,
          tensor::OutputOperand C>
VECOPS_INLINE auto make_benchmark_matmul_invocation(
    const Config& config, M&& m, N&& n, K&& k,
    A&& a, B&& b, C&& c) {
  using Atom = typename Config::Atom;
  auto c_output = tensor::as_output_spec<typename Atom::TAcc>(
      std::forward<C>(c));
  using Memory = typename decltype(c_output)::MemoryElement;
  auto c_input = tensor::input<typename Atom::TAcc>(
      c_output.tensor(),
      tensor::zeros_transform<typename Atom::TAcc, Memory>);
  return ::vecops::matmul::details::make_matmul_invocation(
      config, std::forward<M>(m), std::forward<N>(n),
      std::forward<K>(k), std::forward<A>(a), std::forward<B>(b),
      std::move(c_input), std::move(c_output));
}

template <typename Config,
          meta::ValueInput M, meta::ValueInput N, meta::ValueInput K,
          tensor::InputOperand A, tensor::InputOperand B,
          tensor::InputOperand CInput, tensor::OutputOperand COutput>
VECOPS_INLINE auto make_benchmark_matmul_invocation(
    const Config& config, M&& m, N&& n, K&& k,
    A&& a, B&& b, CInput&& c_input, COutput&& c_output) {
  return ::vecops::matmul::details::make_matmul_invocation(
      config, std::forward<M>(m), std::forward<N>(n),
      std::forward<K>(k), std::forward<A>(a), std::forward<B>(b),
      std::forward<CInput>(c_input), std::forward<COutput>(c_output));
}

#if !defined(VECOPS_SCENARIO_CATALOG_CASE_SHARDS)
#define VECOPS_SCENARIO_CATALOG_CASE_SHARDS 1
#endif

inline constexpr int ScenarioCatalogCaseShardCount =
    VECOPS_SCENARIO_CATALOG_CASE_SHARDS;

/** Input-side work performed by DataAccess before the hardware atom. */
enum class InputPipeline {
  Convert,
  Quantize4,
  AsymmetricQuantize4Zp3,
  AsymmetricQuantize4Zp3Zp5,
  PrequantizedAsymmetricZp3,
  RuntimePerRowAsymmetricQuantize,
};

/** C prologue/epilogue combinations used by real fused matmuls. */
enum class OutputPipeline {
  Convert,
  Accumulate,
  Relu,
  Sigmoid,
  Silu,
  Scale,
  DynamicScale,
  PerColumnScale,
  Clamp,
  AccumulateReluScale,
  Dequantize,
  ReluRequantize,
  RequantizeU8Zp7,
  Bias,
  BiasRelu,
  BiasClamp,
  BiasSilu,
  BiasPerColumnDequantize,
  BiasPerColumnRequantizeU8,
  RuntimePerRowColumnDequantize,
  AsymmetricDequantize,
};

inline constexpr float32_t ScenarioDynamicScale = 0.75f;
inline constexpr float32_t ScenarioClampLow = -0.25f;
inline constexpr float32_t ScenarioClampHigh = 0.25f;

template <typename Acc = float32_t>
struct ScenarioOutputParameters {
  const Acc* dynamic_scale = nullptr;
  const Acc* column_scales = nullptr;
  const float32_t* quant_column_scales = nullptr;
  const int32_t* output_zero_point = nullptr;
  const float32_t* row_dequant_scales = nullptr;
  const float32_t* weight_column_scales = nullptr;
  nint_t row_scale_batch_stride = 0;
};

struct ScenarioInputParameters {
  const float32_t* row_quant_multipliers = nullptr;
  const int32_t* a_zero_point = nullptr;
  nint_t row_scale_batch_stride = 0;
};

template <InputMode Mode>
inline constexpr bool scenario_packs_a_v =
    Mode == InputMode::PackedA || Mode == InputMode::PackedAB ||
    Mode == InputMode::OnlinePackedA || Mode == InputMode::OnlinePackedAB;

template <InputMode Mode>
inline constexpr bool scenario_packs_b_v =
    Mode == InputMode::PackedB || Mode == InputMode::PackedAB ||
    Mode == InputMode::OnlinePackedB || Mode == InputMode::OnlinePackedAB;

template <InputMode Mode>
inline constexpr bool scenario_online_pack_a_v =
    Mode == InputMode::OnlinePackedA || Mode == InputMode::OnlinePackedAB;

template <InputMode Mode>
inline constexpr bool scenario_online_pack_b_v =
    Mode == InputMode::OnlinePackedB || Mode == InputMode::OnlinePackedAB;

template <InputPipeline Pipeline>
constexpr const char* input_pipeline_name() {
  if constexpr (Pipeline == InputPipeline::Convert) return "convert";
  if constexpr (Pipeline == InputPipeline::Quantize4) return "quantize_x4";
  if constexpr (Pipeline == InputPipeline::AsymmetricQuantize4Zp3)
    return "asymmetric_quantize_x4_zp3";
  if constexpr (Pipeline == InputPipeline::AsymmetricQuantize4Zp3Zp5)
    return "asymmetric_quantize_x4_zp3_zp5";
  if constexpr (Pipeline == InputPipeline::PrequantizedAsymmetricZp3)
    return "prequantized_asymmetric_zp3";
  return "runtime_per_row_asymmetric_quantize";
}

template <InputPipeline PipelineA, InputPipeline PipelineB>
std::string input_pipeline_label() {
  if constexpr (PipelineA == PipelineB) {
    return "/input_pipeline:" + std::string(input_pipeline_name<PipelineA>());
  } else {
    return "/input_pipeline_a:" +
        std::string(input_pipeline_name<PipelineA>()) +
        "/input_pipeline_b:" +
        std::string(input_pipeline_name<PipelineB>());
  }
}

template <typename MemoryCInput, typename MemoryCOutput>
std::string c_input_type_label() {
  if constexpr (std::same_as<MemoryCInput, MemoryCOutput>) {
    return {};
  } else {
    return "/c_input:" + std::string(dtype_name<MemoryCInput>());
  }
}

template <OutputPipeline Pipeline>
constexpr const char* output_pipeline_name() {
  if constexpr (Pipeline == OutputPipeline::Convert) return "convert";
  if constexpr (Pipeline == OutputPipeline::Accumulate) return "accumulate";
  if constexpr (Pipeline == OutputPipeline::Relu) return "relu";
  if constexpr (Pipeline == OutputPipeline::Sigmoid) return "sigmoid";
  if constexpr (Pipeline == OutputPipeline::Silu) return "silu";
  if constexpr (Pipeline == OutputPipeline::Scale) return "scale_0p5";
  if constexpr (Pipeline == OutputPipeline::DynamicScale)
    return "dynamic_scale_0p75";
  if constexpr (Pipeline == OutputPipeline::PerColumnScale)
    return "per_column_scale";
  if constexpr (Pipeline == OutputPipeline::Clamp)
    return "clamp_m0p25_0p25";
  if constexpr (Pipeline == OutputPipeline::AccumulateReluScale)
    return "accumulate_relu_scale";
  if constexpr (Pipeline == OutputPipeline::Dequantize)
    return "dequantize_0p125";
  if constexpr (Pipeline == OutputPipeline::ReluRequantize)
    return "relu_requantize_0p125";
  if constexpr (Pipeline == OutputPipeline::RequantizeU8Zp7)
    return "requantize_u8_0p125_zp7";
  if constexpr (Pipeline == OutputPipeline::Bias) return "bias";
  if constexpr (Pipeline == OutputPipeline::BiasRelu) return "bias_relu";
  if constexpr (Pipeline == OutputPipeline::BiasClamp) return "bias_clamp";
  if constexpr (Pipeline == OutputPipeline::BiasSilu) return "bias_silu";
  if constexpr (Pipeline == OutputPipeline::BiasPerColumnDequantize)
    return "bias_per_column_dequantize";
  if constexpr (Pipeline == OutputPipeline::BiasPerColumnRequantizeU8)
    return "bias_per_column_requantize_u8";
  if constexpr (Pipeline == OutputPipeline::RuntimePerRowColumnDequantize)
    return "runtime_per_row_column_dequantize";
  return "asymmetric_dequantize_0p0625";
}

template <OutputPipeline Pipeline>
inline constexpr bool uses_c_prologue_v =
    Pipeline == OutputPipeline::Accumulate ||
    Pipeline == OutputPipeline::AccumulateReluScale ||
    Pipeline == OutputPipeline::Bias ||
    Pipeline == OutputPipeline::BiasRelu ||
    Pipeline == OutputPipeline::BiasClamp ||
    Pipeline == OutputPipeline::BiasSilu ||
    Pipeline == OutputPipeline::BiasPerColumnDequantize ||
    Pipeline == OutputPipeline::BiasPerColumnRequantizeU8;

template <OutputPipeline Pipeline>
inline constexpr bool uses_bias_prologue_v =
  Pipeline == OutputPipeline::Bias ||
    Pipeline == OutputPipeline::BiasRelu ||
    Pipeline == OutputPipeline::BiasClamp ||
    Pipeline == OutputPipeline::BiasSilu ||
    Pipeline == OutputPipeline::BiasPerColumnDequantize ||
    Pipeline == OutputPipeline::BiasPerColumnRequantizeU8;

template <OutputPipeline Pipeline>
inline constexpr bool uses_dynamic_scale_v =
    Pipeline == OutputPipeline::DynamicScale;

template <OutputPipeline Pipeline>
inline constexpr bool uses_per_column_scale_v =
    Pipeline == OutputPipeline::PerColumnScale ||
    Pipeline == OutputPipeline::BiasPerColumnDequantize ||
    Pipeline == OutputPipeline::BiasPerColumnRequantizeU8 ||
    Pipeline == OutputPipeline::RuntimePerRowColumnDequantize;

template <OutputPipeline Pipeline>
inline constexpr bool uses_quant_column_scale_v =
    Pipeline == OutputPipeline::BiasPerColumnDequantize ||
    Pipeline == OutputPipeline::BiasPerColumnRequantizeU8 ||
    Pipeline == OutputPipeline::RuntimePerRowColumnDequantize;

template <OutputPipeline Pipeline>
inline constexpr bool uses_runtime_output_zero_point_v =
    Pipeline == OutputPipeline::BiasPerColumnRequantizeU8;

template <InputPipeline Pipeline>
inline constexpr bool uses_asymmetric_correction_v =
    Pipeline == InputPipeline::AsymmetricQuantize4Zp3 ||
    Pipeline == InputPipeline::AsymmetricQuantize4Zp3Zp5 ||
    Pipeline == InputPipeline::PrequantizedAsymmetricZp3 ||
    Pipeline == InputPipeline::RuntimePerRowAsymmetricQuantize;

template <InputPipeline Pipeline>
inline constexpr bool uses_dual_asymmetric_correction_v =
    Pipeline == InputPipeline::AsymmetricQuantize4Zp3Zp5;

template <InputPipeline Pipeline>
inline constexpr bool uses_single_asymmetric_correction_v =
    uses_asymmetric_correction_v<Pipeline> &&
    !uses_dual_asymmetric_correction_v<Pipeline>;

template <InputPipeline Pipeline>
inline constexpr bool uses_runtime_per_row_quantization_v =
    Pipeline == InputPipeline::RuntimePerRowAsymmetricQuantize;

template <InputPipeline Pipeline>
inline constexpr bool timed_asymmetric_compensation_v =
#if defined(ARCH_X86_FAMILY)
    uses_single_asymmetric_correction_v<Pipeline>;
#else
    false;
#endif

template <InputPipeline PipelineA, InputPipeline PipelineB>
inline constexpr bool supported_asymmetric_pipeline_pair_v =
    (!uses_asymmetric_correction_v<PipelineA> &&
     !uses_asymmetric_correction_v<PipelineB>) ||
    PipelineA == PipelineB ||
    (uses_single_asymmetric_correction_v<PipelineA> &&
     !uses_asymmetric_correction_v<PipelineB>);

template <::vecops::matmul::Operand Side, typename Compute, InputPipeline Pipeline>
constexpr Compute scenario_input_zero_point() {
  if constexpr (Pipeline == InputPipeline::AsymmetricQuantize4Zp3Zp5) {
    static_assert(std::is_unsigned_v<Compute>);
    return static_cast<Compute>(Side == ::vecops::matmul::Operand::A ? 3 : 5);
  } else if constexpr (
      Pipeline == InputPipeline::AsymmetricQuantize4Zp3 &&
      Side == ::vecops::matmul::Operand::A && std::is_unsigned_v<Compute>) {
    return static_cast<Compute>(3);
  } else {
    return Compute{};
  }
}

template <::vecops::matmul::Operand Side, typename Compute, typename Memory,
          InputPipeline Pipeline>
Compute reference_input(
    Memory value, nint_t row_scale_index = 0,
    const ScenarioInputParameters& parameters = {}) {
  if constexpr (
      Pipeline == InputPipeline::RuntimePerRowAsymmetricQuantize) {
    static_assert(Side == ::vecops::matmul::Operand::A);
    static_assert(std::is_unsigned_v<Compute>);
    return static_cast<Compute>(
        static_cast<float32_t>(value) *
            parameters.row_quant_multipliers[row_scale_index] +
        static_cast<float32_t>(*parameters.a_zero_point));
  } else if constexpr (Pipeline == InputPipeline::Quantize4 ||
                Pipeline == InputPipeline::AsymmetricQuantize4Zp3 ||
                Pipeline == InputPipeline::AsymmetricQuantize4Zp3Zp5) {
    static_assert(std::is_integral_v<Compute>);
    constexpr float32_t zero_point = static_cast<float32_t>(
        scenario_input_zero_point<Side, Compute, Pipeline>());
    return static_cast<Compute>(
        static_cast<float32_t>(value) * 4.0f + zero_point);
  } else {
    return static_cast<Compute>(value);
  }
}

inline float32_t scenario_column_scale(nint_t column) {
  return 0.5f + static_cast<float32_t>(column % 7) * 0.0625f;
}

inline float32_t scenario_quant_column_scale(nint_t column) {
  return 0.015625f + static_cast<float32_t>(column % 7) * 0.00390625f;
}

template <typename Memory, typename Acc, OutputPipeline Pipeline>
Memory reference_output(
    Acc value, nint_t column = 0,
    const ScenarioOutputParameters<Acc>& parameters = {},
    nint_t row_scale_index = 0) {
  if constexpr (Pipeline == OutputPipeline::Relu ||
                Pipeline == OutputPipeline::AccumulateReluScale ||
                Pipeline == OutputPipeline::ReluRequantize ||
                Pipeline == OutputPipeline::BiasRelu) {
    value = std::max(value, Acc{});
  }
  if constexpr (Pipeline == OutputPipeline::Sigmoid) {
    const double x = static_cast<double>(value);
    return static_cast<Memory>(1.0 / (1.0 + std::exp(-x)));
  } else if constexpr (
      Pipeline == OutputPipeline::Silu ||
      Pipeline == OutputPipeline::BiasSilu) {
    const double x = static_cast<double>(value);
    return static_cast<Memory>(x / (1.0 + std::exp(-x)));
  } else if constexpr (Pipeline == OutputPipeline::Scale ||
                       Pipeline == OutputPipeline::AccumulateReluScale) {
    return static_cast<Memory>(static_cast<double>(value) * 0.5);
  } else if constexpr (Pipeline == OutputPipeline::DynamicScale) {
    return static_cast<Memory>(
        static_cast<double>(value) * *parameters.dynamic_scale);
  } else if constexpr (Pipeline == OutputPipeline::PerColumnScale) {
    return static_cast<Memory>(
        static_cast<double>(value) * parameters.column_scales[column]);
  } else if constexpr (
      Pipeline == OutputPipeline::Clamp ||
      Pipeline == OutputPipeline::BiasClamp) {
    return static_cast<Memory>(std::clamp(
        value, static_cast<Acc>(ScenarioClampLow),
        static_cast<Acc>(ScenarioClampHigh)));
  } else if constexpr (Pipeline == OutputPipeline::Dequantize ||
                       Pipeline == OutputPipeline::ReluRequantize) {
    const double scaled = static_cast<double>(value) * 0.125;
    if constexpr (
        Pipeline == OutputPipeline::ReluRequantize &&
        std::is_integral_v<Memory>) {
      const double low = static_cast<double>(
          std::numeric_limits<Memory>::lowest());
      const double high = static_cast<double>(
          std::numeric_limits<Memory>::max());
      return static_cast<Memory>(std::clamp(scaled, low, high));
    } else {
      return static_cast<Memory>(scaled);
    }
  } else if constexpr (Pipeline == OutputPipeline::AsymmetricDequantize) {
    return static_cast<Memory>(static_cast<double>(value) * 0.0625);
  } else if constexpr (Pipeline == OutputPipeline::RequantizeU8Zp7) {
    static_assert(std::is_integral_v<Memory>);
    const double scaled = static_cast<double>(value) * 0.125 + 7.0;
    return static_cast<Memory>(std::clamp(
        scaled,
        static_cast<double>(std::numeric_limits<Memory>::lowest()),
        static_cast<double>(std::numeric_limits<Memory>::max())));
  } else if constexpr (
      Pipeline == OutputPipeline::BiasPerColumnDequantize) {
    return static_cast<Memory>(
        static_cast<double>(value) *
        parameters.quant_column_scales[column]);
  } else if constexpr (
      Pipeline == OutputPipeline::BiasPerColumnRequantizeU8) {
    static_assert(std::is_integral_v<Memory>);
    const double scaled = static_cast<double>(value) *
        parameters.quant_column_scales[column] +
        static_cast<double>(*parameters.output_zero_point);
    return static_cast<Memory>(std::clamp(
        scaled,
        static_cast<double>(std::numeric_limits<Memory>::lowest()),
        static_cast<double>(std::numeric_limits<Memory>::max())));
  } else if constexpr (
      Pipeline == OutputPipeline::RuntimePerRowColumnDequantize) {
    return static_cast<Memory>(
        static_cast<double>(value) *
        parameters.row_dequant_scales[row_scale_index] *
        parameters.weight_column_scales[column]);
  } else {
    return static_cast<Memory>(value);
  }
}

template <::vecops::matmul::Operand Side, typename Compute, InputPipeline Pipeline,
          typename Tensor>
auto make_scenario_input(
    const Tensor& tensor,
    const ScenarioInputParameters& parameters = {}) {
  if constexpr (
      Pipeline == InputPipeline::RuntimePerRowAsymmetricQuantize) {
    static_assert(Side == ::vecops::matmul::Operand::A);
    static_assert(std::is_unsigned_v<Compute>);
    auto quantize = ::vecops::matmul::
        make_runtime_per_row_asymmetric_quantize_transform({
            parameters.row_quant_multipliers,
            parameters.a_zero_point,
            parameters.row_scale_batch_stride});
    return tensor::input<Compute>(tensor, quantize);
  } else if constexpr (Pipeline == InputPipeline::Quantize4 ||
                Pipeline == InputPipeline::AsymmetricQuantize4Zp3 ||
                Pipeline == InputPipeline::AsymmetricQuantize4Zp3Zp5) {
    static_assert(std::is_integral_v<Compute>);
    constexpr float32_t ZeroPoint = static_cast<float32_t>(
        scenario_input_zero_point<Side, Compute, Pipeline>());
    auto quantize = tensor::make_elementwise_vec_transform<
        float32_t, float32_t>(
        [](auto tag, auto value) VECOPS_KERNEL_LAMBDA {
          auto result = vec::mul(tag, value, vec::fill(tag, 4.0f));
          if constexpr (ZeroPoint != 0.0f) {
            result = vec::add(tag, result, vec::fill(tag, ZeroPoint));
          }
          return result;
        });
    return tensor::input<Compute>(tensor, quantize);
  } else {
    return tensor::input<Compute>(tensor);
  }
}

template <typename Acc, OutputPipeline Pipeline, typename Tensor>
auto make_scenario_output(
    const Tensor& tensor,
    const ScenarioOutputParameters<Acc>& parameters = {}) {
  if constexpr (Pipeline == OutputPipeline::Relu ||
                Pipeline == OutputPipeline::BiasRelu) {
    auto transform = tensor::make_elementwise_vec_transform<Acc, Acc>(
        [](auto tag, auto value) VECOPS_KERNEL_LAMBDA {
          return vec::max(tag, value, vec::zeros(tag));
        });
    return tensor::output<Acc>(tensor, transform);
  } else if constexpr (Pipeline == OutputPipeline::Sigmoid) {
    static_assert(std::is_floating_point_v<Acc>);
    auto transform = tensor::make_elementwise_vec_transform<Acc, Acc>(
        [](auto tag, auto value) VECOPS_KERNEL_LAMBDA {
          const auto one = vec::fill(tag, Acc{1});
          return vec::div(
              tag, one,
              vec::add(tag, one, vec::exp(tag, vec::neg(tag, value))));
        });
    return tensor::output<Acc>(tensor, transform);
  } else if constexpr (
      Pipeline == OutputPipeline::Silu ||
      Pipeline == OutputPipeline::BiasSilu) {
    static_assert(std::is_floating_point_v<Acc>);
    auto transform = tensor::make_elementwise_vec_transform<Acc, Acc>(
        [](auto tag, auto value) VECOPS_KERNEL_LAMBDA {
          const auto one = vec::fill(tag, Acc{1});
          const auto sigmoid = vec::div(
              tag, one,
              vec::add(tag, one, vec::exp(tag, vec::neg(tag, value))));
          return vec::mul(tag, value, sigmoid);
        });
    return tensor::output<Acc>(tensor, transform);
  } else if constexpr (Pipeline == OutputPipeline::Scale) {
    auto transform = tensor::make_elementwise_vec_transform<Acc, Acc>(
        [](auto tag, auto value) VECOPS_KERNEL_LAMBDA {
          return vec::mul(tag, value, vec::fill(tag, Acc{0.5}));
        });
    return tensor::output<Acc>(tensor, transform);
  } else if constexpr (Pipeline == OutputPipeline::DynamicScale) {
    static_assert(std::is_floating_point_v<Acc>);
    auto transform = tensor::make_elementwise_vec_transform<Acc, Acc>(
        [scale = parameters.dynamic_scale](
            auto tag, auto value) VECOPS_KERNEL_LAMBDA {
          return vec::mul(tag, value, vec::fill(tag, *scale));
        });
    return tensor::output<Acc>(tensor, transform);
  } else if constexpr (Pipeline == OutputPipeline::PerColumnScale) {
    static_assert(std::is_floating_point_v<Acc>);
    auto transform = tensor::make_lane_local_vec_transform<Acc, Acc>(
        [scales = parameters.column_scales](
            auto tag, auto value,
            const auto& context) VECOPS_KERNEL_LAMBDA {
          const auto coordinate = context.lane_coord(0);
          const nint_t column = coordinate[coordinate.size() - 1];
          return vec::mul(tag, value, vec::load(tag, scales + column));
        });
    return tensor::output<Acc>(tensor, transform);
  } else if constexpr (
      Pipeline == OutputPipeline::BiasPerColumnDequantize ||
      Pipeline == OutputPipeline::BiasPerColumnRequantizeU8) {
    static_assert(std::same_as<Acc, int32_t>);
    auto transform = tensor::make_lane_local_vec_transform<
        float32_t, float32_t>(
        [scales = parameters.quant_column_scales,
         zero_point = parameters.output_zero_point](
            auto tag, auto value,
            const auto& context) VECOPS_KERNEL_LAMBDA {
          const auto coordinate = context.lane_coord(0);
          const nint_t column = coordinate[coordinate.size() - 1];
          auto result = vec::mul(
              tag, value, vec::load(tag, scales + column));
          if constexpr (
              Pipeline == OutputPipeline::BiasPerColumnRequantizeU8) {
            result = vec::add(
                tag, result,
                vec::fill(tag, static_cast<float32_t>(*zero_point)));
          }
          return result;
        });
    return tensor::output<Acc>(tensor, transform);
  } else if constexpr (
      Pipeline == OutputPipeline::RuntimePerRowColumnDequantize) {
    static_assert(std::same_as<Acc, int32_t>);
    auto transform = ::vecops::matmul::
        make_runtime_per_row_column_dequantize_transform({
            parameters.row_dequant_scales,
            parameters.weight_column_scales,
            parameters.row_scale_batch_stride});
    return tensor::output<Acc>(tensor, transform);
  } else if constexpr (
      Pipeline == OutputPipeline::Clamp ||
      Pipeline == OutputPipeline::BiasClamp) {
    auto transform = tensor::make_elementwise_vec_transform<Acc, Acc>(
        [](auto tag, auto value) VECOPS_KERNEL_LAMBDA {
          return vec::min(
              tag,
              vec::max(tag, value, vec::fill(tag, Acc{ScenarioClampLow})),
              vec::fill(tag, Acc{ScenarioClampHigh}));
        });
    return tensor::output<Acc>(tensor, transform);
  } else if constexpr (Pipeline == OutputPipeline::AccumulateReluScale) {
    auto transform = tensor::make_elementwise_vec_transform<Acc, Acc>(
        [](auto tag, auto value) VECOPS_KERNEL_LAMBDA {
          const auto relu = vec::max(tag, value, vec::zeros(tag));
          return vec::mul(tag, relu, vec::fill(tag, Acc{0.5}));
        });
    return tensor::output<Acc>(tensor, transform);
  } else if constexpr (Pipeline == OutputPipeline::Dequantize) {
    static_assert(std::same_as<Acc, int32_t>);
    auto transform = tensor::make_elementwise_vec_transform<
        float32_t, float32_t>(
        [](auto tag, auto value) VECOPS_KERNEL_LAMBDA {
          return vec::mul(tag, value, vec::fill(tag, 0.125f));
        });
    return tensor::output<Acc>(tensor, transform);
  } else if constexpr (Pipeline == OutputPipeline::ReluRequantize) {
    static_assert(std::same_as<Acc, int32_t>);
    auto transform = tensor::make_elementwise_vec_transform<
        float32_t, float32_t>(
        [](auto tag, auto value) VECOPS_KERNEL_LAMBDA {
          const auto relu = vec::max(tag, value, vec::zeros(tag));
          return vec::mul(tag, relu, vec::fill(tag, 0.125f));
        });
    return tensor::output<Acc>(tensor, transform);
  } else if constexpr (Pipeline == OutputPipeline::RequantizeU8Zp7) {
    static_assert(std::same_as<Acc, int32_t>);
    auto transform = tensor::make_elementwise_vec_transform<
        float32_t, float32_t>(
        [](auto tag, auto value) VECOPS_KERNEL_LAMBDA {
          return vec::add(
              tag, vec::mul(tag, value, vec::fill(tag, 0.125f)),
              vec::fill(tag, 7.0f));
        });
    return tensor::output<Acc>(tensor, transform);
  } else if constexpr (Pipeline == OutputPipeline::AsymmetricDequantize) {
    static_assert(std::same_as<Acc, int32_t>);
    auto transform = tensor::make_elementwise_vec_transform<
        float32_t, float32_t>(
        [](auto tag, auto value) VECOPS_KERNEL_LAMBDA {
          return vec::mul(tag, value, vec::fill(tag, 0.0625f));
        });
    return tensor::output<Acc>(tensor, transform);
  } else {
    return tensor::output<Acc>(tensor);
  }
}

template <typename T, typename Compute, InputPipeline Pipeline>
void fill_scenario_input(std::vector<T>& values, int seed) {
  for (std::size_t i = 0; i < values.size(); ++i) {
    if constexpr (Pipeline == InputPipeline::Quantize4 ||
                  Pipeline == InputPipeline::AsymmetricQuantize4Zp3 ||
                  Pipeline == InputPipeline::AsymmetricQuantize4Zp3Zp5) {
      const int low = std::is_unsigned_v<Compute> ? 0 : -3;
      const int span = std::is_unsigned_v<Compute> ? 7 : 7;
      const int q = low + static_cast<int>((i * 5 + seed) % span);
      values[i] = static_cast<T>(static_cast<float>(q) * 0.25f);
    } else if constexpr (std::is_integral_v<T>) {
      const int low = std::is_unsigned_v<T> ? 0 : -3;
      values[i] = static_cast<T>(
          low + static_cast<int>((i * 5 + seed) % 7));
    } else {
      const int centered = static_cast<int>((i * 7 + seed) % 17) - 8;
      values[i] = static_cast<T>(static_cast<float>(centered) / 16.0f);
    }
  }
}

template <typename T>
bool scenario_values_equal(T expected, T actual) {
  if constexpr (std::is_integral_v<T>) {
    return expected == actual;
  } else {
    const double e = static_cast<double>(expected);
    const double a = static_cast<double>(actual);
    const double tolerance = sizeof(T) <= 2 ? 2.0e-2
                            : sizeof(T) == 4 ? 8.0e-4
                                             : 2.0e-10;
    return std::abs(a - e) <= tolerance * std::max({1.0, std::abs(a), std::abs(e)});
  }
}

inline std::vector<MatmulCase> make_scenario_cases(nint_t k_step) {
  return {
      {"scenario", "tail", 19, 23, 8 * k_step + 1,
       mode_bit(InputMode::Raw)},
      {"scenario", "decode", 1, 1024, 1024,
       mode_bit(InputMode::Raw)},
      {"scenario", "batch", 64, 256, 256,
       mode_bit(InputMode::Raw)},
      {"scenario", "wide", 32, 1024, 256,
       mode_bit(InputMode::Raw)},
  };
}

inline std::vector<MatmulCase> make_fusion_scenario_cases(nint_t k_step) {
  return {
      {"fusion", "tail", 19, 23, 8 * k_step + 1,
       mode_bit(InputMode::Raw)},
      {"fusion", "tiny_area", 2, 4, 2 * k_step + 1,
       mode_bit(InputMode::Raw)},
      {"fusion", "skinny_long_k_tail", 1, 8, 4097,
       mode_bit(InputMode::Raw)},
      {"fusion", "skinny_long_k_tail_col", 8, 1, 4097,
       mode_bit(InputMode::Raw)},
      {"fusion", "gemv_tail", 1, 257, 4 * k_step + 1,
       mode_bit(InputMode::Raw)},
      {"fusion", "small", 8, 128, 128,
       mode_bit(InputMode::Raw)},
      {"fusion", "medium", 64, 256, 256,
       mode_bit(InputMode::Raw)},
      {"fusion", "wide", 32, 1024, 256,
       mode_bit(InputMode::Raw)},
  };
}

struct BatchedMatmulCase {
  const char* name;
  nint_t batch;
  nint_t m;
  nint_t n;
  nint_t k;
  bool broadcast_b;
  bool broadcast_a = false;
};

inline std::vector<BatchedMatmulCase> make_batched_scenario_cases(
    nint_t k_step) {
  return {
      {"independent_tail", 4, 19, 23, 8 * k_step + 1, false},
      {"shared_weight_tail", 8, 1, 47, 2 * k_step + 1, true},
      {"shared_weight_tiny", 8, 1, 32, 64, true},
      {"shared_weight_small", 8, 1, 64, 128, true},
      {"shared_weight_small_long_k", 8, 1, 64, 256, true},
      {"shared_weight_wide_short_k", 8, 1, 128, 64, true},
      {"shared_weight_decode", 8, 1, 256, 256, true},
      {"shared_weight_mlp_projection", 8, 1, 1024, 4096, true},
      {"shared_weight_qwen_o_proj", 8, 1, 3584, 3584, true},
      {"shared_weight_qwen_qkv", 8, 1, 4608, 3584, true},
      {"shared_activation_scalar", 16, 1, 1, 64, false, true},
      {"shared_activation_tail", 8, 1, 8, 2 * k_step + 1, false, true},
      {"shared_activation_boundary", 4, 1, 16, 64, false, true},
  };
}

inline std::vector<BatchedMatmulCase> make_batched_fusion_scenario_cases(
    nint_t k_step) {
  return {
      {"fusion_independent_tail", 4, 19, 23, 8 * k_step + 1, false},
      {"fusion_shared_small", 8, 1, 128, 128, true},
      {"fusion_shared_decode", 8, 1, 256, 256, true},
      {"fusion_shared_prefill", 4, 16, 512, 512, true},
  };
}

template <nint_t KStep>
struct ScenarioCaseCatalog {
  static constexpr nint_t k_step = KStep;
  operator std::vector<MatmulCase>() const {
    return make_scenario_cases(KStep);
  }
};

template <nint_t KStep>
struct FusionScenarioCaseCatalog {
  static constexpr nint_t k_step = KStep;
  operator std::vector<MatmulCase>() const {
    return make_fusion_scenario_cases(KStep);
  }
};

template <nint_t KStep>
struct BatchedScenarioCaseCatalog {
  static constexpr nint_t k_step = KStep;
  operator std::vector<BatchedMatmulCase>() const {
    return make_batched_scenario_cases(KStep);
  }
};

template <nint_t KStep>
struct BatchedFusionScenarioCaseCatalog {
  static constexpr nint_t k_step = KStep;
  operator std::vector<BatchedMatmulCase>() const {
    return make_batched_fusion_scenario_cases(KStep);
  }
};

#define scenario_cases(KStep) \
  ::vecops::bench::matmul::ScenarioCaseCatalog<(KStep)>{}
#define fusion_scenario_cases(KStep) \
  ::vecops::bench::matmul::FusionScenarioCaseCatalog<(KStep)>{}
#define batched_scenario_cases(KStep) \
  ::vecops::bench::matmul::BatchedScenarioCaseCatalog<(KStep)>{}
#define batched_fusion_scenario_cases(KStep) \
  ::vecops::bench::matmul::BatchedFusionScenarioCaseCatalog<(KStep)>{}

template <typename Atom, typename MemoryA, typename MemoryB, typename MemoryC,
          InputPipeline InPipeline, OutputPipeline OutPipeline,
          bool BroadcastB, bool BroadcastA, InputMode Mode,
          InputPipeline InPipelineB, typename MemoryCInput,
          typename BatchExtent, typename MExtent,
          typename NExtent, typename KExtent>
void run_batched_scenario_with_extents(
    benchmark::State& state, const BatchedMatmulCase& test_case,
    BatchExtent batch_extent, MExtent m_extent,
    NExtent n_extent, KExtent k_extent,
    nint_t calls_per_pack = 1) {
  static_assert(!scenario_packs_a_v<Mode>,
                "batched scenario currently packs only shared B");
  static_assert(!(BroadcastA && BroadcastB));
  static_assert(!scenario_packs_b_v<Mode> || BroadcastB,
                "a packed batch operand must be broadcast");
  using TA = typename Atom::TA;
  using TB = typename Atom::TB;
  using Acc = typename Atom::TAcc;
  const nint_t batch = test_case.batch;
  const nint_t m = test_case.m;
  const nint_t n = test_case.n;
  const nint_t k = test_case.k;
  const nint_t a_batches = BroadcastA ? 1 : batch;
  const nint_t b_batches = BroadcastB ? 1 : batch;
  constexpr bool UsesAsymmetricCorrection =
      uses_asymmetric_correction_v<InPipeline> ||
      uses_asymmetric_correction_v<InPipelineB>;
  static_assert(
      supported_asymmetric_pipeline_pair_v<InPipeline, InPipelineB>,
      "only an asymmetric A pipeline with a direct B pipeline is supported");
  if (calls_per_pack <= 0) {
    state.SkipWithError("calls_per_pack must be positive");
    return;
  }

  std::vector<MemoryA> a(static_cast<std::size_t>(a_batches * m * k));
  std::vector<MemoryB> b(static_cast<std::size_t>(b_batches * n * k));
  std::vector<MemoryC> c(static_cast<std::size_t>(batch * m * n));
  constexpr bool UsesBias = uses_bias_prologue_v<OutPipeline>;
  std::vector<MemoryCInput> c_input(static_cast<std::size_t>(
      UsesBias ? n : batch * m * n));
  constexpr bool DualZeroPoint =
      uses_dual_asymmetric_correction_v<InPipeline>;
  std::vector<Acc> asymmetric_correction(static_cast<std::size_t>(
      DualZeroPoint ? batch * m * n : b_batches * n));
  Acc dynamic_scale = static_cast<Acc>(ScenarioDynamicScale);
  std::vector<Acc> column_scales(
      static_cast<std::size_t>(n + 64));
  std::vector<float32_t> quant_column_scales(
      static_cast<std::size_t>(n + 64));
  const nint_t row_scale_count = a_batches * m;
  std::vector<float32_t> row_quant_multipliers(
      static_cast<std::size_t>(row_scale_count));
  std::vector<float32_t> row_dequant_scales(
      static_cast<std::size_t>(row_scale_count));
  std::vector<float32_t> weight_column_scales(
      static_cast<std::size_t>(n + 64));
  for (nint_t col = 0; col < n + 64; ++col) {
    column_scales[static_cast<std::size_t>(col)] =
        static_cast<Acc>(scenario_column_scale(col));
    quant_column_scales[static_cast<std::size_t>(col)] =
        scenario_quant_column_scale(col);
    weight_column_scales[static_cast<std::size_t>(col)] =
        0.03125f + static_cast<float32_t>(col % 5) * 0.0078125f;
  }
  for (nint_t row = 0; row < row_scale_count; ++row) {
    const float32_t multiplier =
        static_cast<float32_t>(4 << (row % 3));
    row_quant_multipliers[static_cast<std::size_t>(row)] = multiplier;
    row_dequant_scales[static_cast<std::size_t>(row)] = 1.0f / multiplier;
  }
  int32_t output_zero_point = 7;
  int32_t input_zero_point = 7;
  const nint_t row_scale_batch_stride = BroadcastA ? 0 : m;
  const ScenarioInputParameters input_parameters{
      row_quant_multipliers.data(), &input_zero_point,
      row_scale_batch_stride};
  const ScenarioOutputParameters<Acc> output_parameters{
      &dynamic_scale, column_scales.data(), quant_column_scales.data(),
      &output_zero_point, row_dequant_scales.data(),
      weight_column_scales.data(), row_scale_batch_stride};
  fill_scenario_input<MemoryA, TA, InPipeline>(a, 3);
  fill_scenario_input<MemoryB, TB, InPipelineB>(b, 11);
  fill_scenario_input<MemoryCInput, Acc, InputPipeline::Convert>(c_input, 5);
  if constexpr (uses_runtime_per_row_quantization_v<InPipeline>) {
    static_assert(std::same_as<MemoryA, float32_t>);
    for (nint_t bi = 0; bi < a_batches; ++bi) {
      for (nint_t row = 0; row < m; ++row) {
        const nint_t scale_index = bi * m + row;
        const float32_t scale =
            row_dequant_scales[static_cast<std::size_t>(scale_index)];
        for (nint_t kk = 0; kk < k; ++kk) {
          const int logical =
              static_cast<int>((bi * m * k + row * k + kk) % 7) - 3;
          a[static_cast<std::size_t>((bi * m + row) * k + kk)] =
              static_cast<float32_t>(logical) * scale;
        }
      }
    }
  }
  if constexpr (UsesAsymmetricCorrection) {
    static_assert(std::same_as<TA, uint8_t>);
    if constexpr (DualZeroPoint) {
      static_assert(std::same_as<TB, uint8_t>);
      constexpr Acc ZeroPointA = 3;
      constexpr Acc ZeroPointB = 5;
      for (nint_t bi = 0; bi < batch; ++bi) {
        const nint_t b_batch = BroadcastB ? 0 : bi;
        for (nint_t row = 0; row < m; ++row) {
          Acc sum_a{};
          for (nint_t kk = 0; kk < k; ++kk) {
            sum_a += static_cast<Acc>(reference_input<
                ::vecops::matmul::Operand::A, TA, MemoryA, InPipeline>(
                a[static_cast<std::size_t>(
                    (((BroadcastA ? 0 : bi) * m + row) * k + kk))]));
          }
          for (nint_t col = 0; col < n; ++col) {
            Acc sum_b{};
            for (nint_t kk = 0; kk < k; ++kk) {
              sum_b += static_cast<Acc>(reference_input<
                  ::vecops::matmul::Operand::B, TB, MemoryB, InPipelineB>(
                  b[static_cast<std::size_t>(
                      (b_batch * n + col) * k + kk)]));
            }
            asymmetric_correction[static_cast<std::size_t>(
                (bi * m + row) * n + col)] =
                -ZeroPointA * sum_b - ZeroPointB * sum_a +
                static_cast<Acc>(k) * ZeroPointA * ZeroPointB;
          }
        }
      }
    } else {
      static_assert(std::same_as<TB, int8_t>);
      const Acc zero_point_a =
          uses_runtime_per_row_quantization_v<InPipeline>
          ? static_cast<Acc>(input_zero_point)
          : Acc{3};
      for (nint_t bi = 0; bi < b_batches; ++bi) {
        for (nint_t col = 0; col < n; ++col) {
          Acc sum{};
          for (nint_t kk = 0; kk < k; ++kk) {
            sum += static_cast<Acc>(reference_input<
                ::vecops::matmul::Operand::B, TB, MemoryB, InPipelineB>(
                b[static_cast<std::size_t>((bi * n + col) * k + kk)]));
          }
          asymmetric_correction[static_cast<std::size_t>(bi * n + col)] =
              -zero_point_a * sum;
        }
      }
    }
  }

  const auto a_tensor = [&] {
    if constexpr (BroadcastA) {
      return tensor::make_tensor(
          a.data(), tensor::make_layout(
                        tensor::make_shape(batch_extent, m_extent, k_extent),
                        tensor::make_strides(cint<0>, k_extent, cint<1>)));
    } else {
      return tensor::make_tensor(
          a.data(), tensor::make_layout(tensor::make_shape(
                        batch_extent, m_extent, k_extent)));
    }
  }();
  const auto b_tensor = [&] {
    if constexpr (BroadcastB) {
      return tensor::make_tensor(
          b.data(),
          tensor::make_layout(
              tensor::make_shape(batch_extent, n_extent, k_extent),
              tensor::make_strides(cint<0>, k_extent, cint<1>)));
    } else {
      return tensor::make_tensor(
          b.data(), tensor::make_layout(tensor::make_shape(
                        batch_extent, n_extent, k_extent)));
    }
  }();
  const auto b_matrix_tensor = tensor::make_tensor(
      b.data(), tensor::make_layout(
                    tensor::make_shape(n_extent, k_extent)));
  const auto c_tensor = tensor::make_tensor(
      c.data(), tensor::make_layout(tensor::make_shape(
                    batch_extent, m_extent, n_extent)));
  const auto c_input_layout = [&] {
    if constexpr (UsesBias) {
      return tensor::make_layout(
          tensor::make_shape(batch_extent, m_extent, n_extent),
          tensor::make_strides(cint<0>, cint<0>, cint<1>));
    } else {
      return tensor::make_layout(tensor::make_shape(
          batch_extent, m_extent, n_extent));
    }
  }();
  const auto c_input_tensor =
      tensor::make_tensor(c_input.data(), c_input_layout);
  const auto a_operand = make_scenario_input<
      ::vecops::matmul::Operand::A, TA, InPipeline>(a_tensor, input_parameters);
  const auto b_operand = make_scenario_input<
      ::vecops::matmul::Operand::B, TB, InPipelineB>(b_tensor);
  const auto b_matrix_operand =
      make_scenario_input<::vecops::matmul::Operand::B, TB, InPipelineB>(b_matrix_tensor);
  const auto c_output = make_scenario_output<Acc, OutPipeline>(
      c_tensor, output_parameters);
  const auto packed_b_layout = ::vecops::matmul::packed_layout<
      Atom, ::vecops::matmul::Operand::B>(b_matrix_tensor.layout());
  const nint_t packed_b_elements = tensor::numel(packed_b_layout);
  const nint_t packed_b_bytes = scenario_packs_b_v<Mode>
      ? packed_b_elements * static_cast<nint_t>(sizeof(TB))
      : 0;
  kernel::Workspace packed_b_storage(packed_b_bytes + 64);
  auto packed_b_workspace = packed_b_storage.view();
  auto* packed_b = scenario_packs_b_v<Mode>
      ? static_cast<TB*>(packed_b_workspace.allocate(packed_b_bytes, 64))
      : static_cast<TB*>(nullptr);
  const auto packed_b_tensor = tensor::make_tensor(
      packed_b, packed_b_layout);
  const auto packed_b_compensation_tensor = tensor::make_tensor(
      asymmetric_correction.data(),
      tensor::make_layout(tensor::make_shape(n_extent)));
  auto pack_b_once = [&](auto& execution) VECOPS_INLINE_LAMBDA {
    if constexpr (scenario_packs_b_v<Mode>) {
      if constexpr (uses_single_asymmetric_correction_v<InPipeline>) {
#if defined(ARCH_X86_FAMILY)
        ::vecops::matmul::details::run_matmul_pack_b_compensated<Atom>(
            execution, b_matrix_operand, packed_b_tensor,
            packed_b_compensation_tensor,
            uses_runtime_per_row_quantization_v<InPipeline>
                ? input_zero_point
                : int32_t{3});
#else
        ::vecops::matmul::details::run_matmul_pack<Atom, ::vecops::matmul::Operand::B>(
            execution, b_matrix_operand, packed_b_tensor);
#endif
      } else {
        ::vecops::matmul::details::run_matmul_pack<Atom, ::vecops::matmul::Operand::B>(
            execution, b_matrix_operand, packed_b_tensor);
      }
    }
  };
  ExecutionSession initial_pack_execution{};
  pack_b_once(initial_pack_execution);

  auto make_operation = [&](const auto& selected_b) {
    if constexpr (UsesAsymmetricCorrection) {
      static_assert(
          OutPipeline == OutputPipeline::AsymmetricDequantize ||
          OutPipeline == OutputPipeline::RuntimePerRowColumnDequantize);
      const auto correction_layout = [&] {
        if constexpr (DualZeroPoint) {
          return tensor::make_layout(tensor::make_shape(
              batch_extent, m_extent, n_extent));
        } else if constexpr (BroadcastB) {
          return tensor::make_layout(
              tensor::make_shape(batch_extent, m_extent, n_extent),
              tensor::make_strides(cint<0>, cint<0>, cint<1>));
        } else {
          return tensor::make_layout(
              tensor::make_shape(batch_extent, m_extent, n_extent),
              tensor::make_strides(n_extent, cint<0>, cint<1>));
        }
      }();
      const auto correction_tensor = tensor::make_tensor(
          asymmetric_correction.data(), correction_layout);
      return make_benchmark_matmul_invocation(ops::MatmulConfig<Atom>{},
          m_extent, n_extent, k_extent, a_operand, selected_b,
          tensor::input<Acc>(correction_tensor), c_output);
    } else if constexpr (uses_c_prologue_v<OutPipeline>) {
      return make_benchmark_matmul_invocation(ops::MatmulConfig<Atom>{},
          m_extent, n_extent, k_extent, a_operand, selected_b,
          tensor::input<Acc>(c_input_tensor), c_output);
    } else {
      return make_benchmark_matmul_invocation(ops::MatmulConfig<Atom>{},
          m_extent, n_extent, k_extent, a_operand, selected_b, c_output);
    }
  };
  const auto operation = [&] {
    if constexpr (scenario_packs_b_v<Mode>)
      return make_operation(packed_b_tensor);
    else
      return make_operation(b_operand);
  }();
  kernel::Workspace storage(operation.required_workspace());
  auto workspace = storage.view();
  ExecutionSession execution{workspace};
  operation(execution);

  const std::array<std::array<nint_t, 3>, 6> samples{{
      {{0, 0, 0}},
      {{batch - 1, m - 1, n - 1}},
      {{batch / 2, m / 2, n / 2}},
      {{0, m - 1, 0}},
      {{batch - 1, 0, n - 1}},
      {{batch / 2, m - 1, n / 2}},
  }};
  for (const auto& sample : samples) {
    const nint_t bi = sample[0];
    const nint_t row = sample[1];
    const nint_t col = sample[2];
    const nint_t a_batch = BroadcastA ? 0 : bi;
    const nint_t b_batch = BroadcastB ? 0 : bi;
    const std::size_t output_index = static_cast<std::size_t>(
        (bi * m + row) * n + col);
    Acc expected{};
    if constexpr (UsesAsymmetricCorrection) {
      if constexpr (DualZeroPoint) {
        expected = asymmetric_correction[output_index];
      } else {
        expected = asymmetric_correction[static_cast<std::size_t>(
            b_batch * n + col)];
      }
    } else if constexpr (UsesBias) {
      expected = static_cast<Acc>(c_input[static_cast<std::size_t>(col)]);
    } else if constexpr (uses_c_prologue_v<OutPipeline>) {
      expected = static_cast<Acc>(c_input[output_index]);
    }
    for (nint_t kk = 0; kk < k; ++kk) {
      const nint_t a_scale_index = a_batch * m + row;
      const TA av = reference_input<
          ::vecops::matmul::Operand::A, TA, MemoryA, InPipeline>(
          a[static_cast<std::size_t>((a_batch * m + row) * k + kk)],
          a_scale_index, input_parameters);
      const TB bv = reference_input<
          ::vecops::matmul::Operand::B, TB, MemoryB, InPipelineB>(
          b[static_cast<std::size_t>((b_batch * n + col) * k + kk)]);
      expected += static_cast<Acc>(av) * static_cast<Acc>(bv);
    }
    const MemoryC reference =
        reference_output<MemoryC, Acc, OutPipeline>(
            expected, col, output_parameters, a_batch * m + row);
    if (!scenario_values_equal(reference, c[output_index])) {
      state.SkipWithError("batched matmul scenario verification failed");
      return;
    }
  }

  for (auto _ : state) {
    benchmark::DoNotOptimize(a.data());
    benchmark::DoNotOptimize(b.data());
    if constexpr (scenario_online_pack_b_v<Mode>) pack_b_once(execution);
    for (nint_t call = 0; call < calls_per_pack; ++call) {
      operation(execution);
    }
    benchmark::DoNotOptimize(c.data());
    benchmark::ClobberMemory();
  }

  const int64_t bmnk = static_cast<int64_t>(batch) * m * n * k;
  const int64_t raw_b_bytes =
      static_cast<int64_t>(b_batches) * n * k * sizeof(MemoryB);
  const int64_t selected_b_bytes = scenario_packs_b_v<Mode>
      ? static_cast<int64_t>(packed_b_bytes)
      : raw_b_bytes;
  const int64_t input_parameter_bytes =
      uses_runtime_per_row_quantization_v<InPipeline>
      ? static_cast<int64_t>(row_scale_count) * sizeof(float32_t) +
            sizeof(int32_t)
      : 0;
  const int64_t epilogue_parameter_bytes =
      OutPipeline == OutputPipeline::RuntimePerRowColumnDequantize
      ? static_cast<int64_t>(row_scale_count + n) * sizeof(float32_t)
      : uses_dynamic_scale_v<OutPipeline>
          ? static_cast<int64_t>(sizeof(Acc))
          : uses_per_column_scale_v<OutPipeline>
              ? static_cast<int64_t>(n) *
                    (uses_quant_column_scale_v<OutPipeline>
                         ? sizeof(float32_t)
                         : sizeof(Acc)) +
                    (uses_runtime_output_zero_point_v<OutPipeline>
                         ? sizeof(int32_t)
                         : 0)
              : 0;
  const int64_t matmul_bytes =
      static_cast<int64_t>(a_batches) * m * k * sizeof(MemoryA) +
      selected_b_bytes +
      static_cast<int64_t>(batch) * m * n * sizeof(MemoryC) +
      (UsesAsymmetricCorrection
           ? static_cast<int64_t>(asymmetric_correction.size()) * sizeof(Acc)
           : UsesBias
               ? static_cast<int64_t>(n) * sizeof(MemoryCInput)
               : uses_c_prologue_v<OutPipeline>
                   ? static_cast<int64_t>(batch) * m * n *
                         sizeof(MemoryCInput)
                   : 0) +
      input_parameter_bytes + epilogue_parameter_bytes;
  int64_t bytes = static_cast<int64_t>(calls_per_pack) * matmul_bytes;
  int64_t lifecycle_pack_bytes = 0;
  if constexpr (scenario_online_pack_b_v<Mode>) {
    lifecycle_pack_bytes = raw_b_bytes + packed_b_bytes +
        (timed_asymmetric_compensation_v<InPipeline>
             ? static_cast<int64_t>(n) * sizeof(Acc) +
                   (uses_runtime_per_row_quantization_v<InPipeline>
                        ? sizeof(int32_t)
                        : 0)
             : 0);
    bytes += lifecycle_pack_bytes;
  }
  state.SetItemsProcessed(
      state.iterations() * static_cast<int64_t>(calls_per_pack) * bmnk);
  state.SetBytesProcessed(state.iterations() * bytes);
  state.counters["FLOP/s"] = benchmark::Counter(
      static_cast<double>(2 * bmnk * calls_per_pack),
      benchmark::Counter::kIsIterationInvariantRate);
  state.counters["batch"] = benchmark::Counter(double(batch));
  state.counters["calls_per_pack"] =
      benchmark::Counter(double(calls_per_pack));
  state.counters["lifecycle_pack_bytes"] =
      benchmark::Counter(double(lifecycle_pack_bytes));
  state.counters["workspace_bytes"] =
      benchmark::Counter(double(operation.required_workspace()));
  state.counters["packed_elements"] = benchmark::Counter(double(
      scenario_packs_b_v<Mode> ? packed_b_elements : 0));
  state.counters["packing_included"] = benchmark::Counter(double(
      scenario_online_pack_b_v<Mode>));
  state.counters["compensation_included"] = benchmark::Counter(double(
      scenario_online_pack_b_v<Mode> &&
      timed_asymmetric_compensation_v<InPipeline>));
  state.counters["input_parameter_bytes"] =
      benchmark::Counter(double(input_parameter_bytes));
  state.counters["runtime_input_zero_point"] = benchmark::Counter(double(
      uses_runtime_per_row_quantization_v<InPipeline>
          ? input_zero_point
          : 0));
  state.counters["epilogue_parameter_bytes"] =
      benchmark::Counter(double(epilogue_parameter_bytes));
}

template <typename Atom, typename MemoryA, typename MemoryB, typename MemoryC,
          InputPipeline InPipeline, OutputPipeline OutPipeline,
          bool BroadcastB, bool BroadcastA = false,
          InputMode Mode = InputMode::Raw,
          InputPipeline InPipelineB = InPipeline,
          typename MemoryCInput = MemoryC>
void run_batched_scenario(
    benchmark::State& state, const BatchedMatmulCase& test_case,
    nint_t calls_per_pack = 1) {
  run_batched_scenario_with_extents<
      Atom, MemoryA, MemoryB, MemoryC, InPipeline, OutPipeline,
      BroadcastB, BroadcastA, Mode, InPipelineB, MemoryCInput>(
          state, test_case, Any{test_case.batch}, Any{test_case.m},
          Any{test_case.n}, Any{test_case.k}, calls_per_pack);
}

template <typename Atom, typename MemoryA, typename MemoryB, typename MemoryC,
          InputPipeline InPipeline, OutputPipeline OutPipeline,
          bool BroadcastB, bool BroadcastA, InputMode Mode,
          InputPipeline InPipelineB, typename MemoryCInput,
          ExtentMode Extents,
          nint_t Batch, nint_t M, nint_t N, nint_t K>
void run_batched_scenario_extent(
    benchmark::State& state, const BatchedMatmulCase& test_case) {
  if constexpr (Extents == ExtentMode::Dynamic) {
    run_batched_scenario_with_extents<
        Atom, MemoryA, MemoryB, MemoryC, InPipeline, OutPipeline,
        BroadcastB, BroadcastA, Mode, InPipelineB, MemoryCInput>(
            state, test_case, Any{test_case.batch}, Any{test_case.m},
            Any{test_case.n}, Any{test_case.k});
  } else {
    run_batched_scenario_with_extents<
        Atom, MemoryA, MemoryB, MemoryC, InPipeline, OutPipeline,
        BroadcastB, BroadcastA, Mode, InPipelineB, MemoryCInput>(
            state, test_case, cint<Batch>, cint<M>, cint<N>, cint<K>);
  }
}

template <typename Atom, typename MemoryA, typename MemoryB, typename MemoryC,
          InputPipeline InPipeline, OutputPipeline OutPipeline,
          bool BroadcastB, bool BroadcastA, InputMode Mode,
          InputPipeline InPipelineB, typename MemoryCInput,
          ExtentMode Extents,
          nint_t Batch, nint_t M, nint_t N, nint_t K>
void register_batched_scenario_extent(
    const char* suite, const char* case_name) {
  using TA = typename Atom::TA;
  using TB = typename Atom::TB;
  using Acc = typename Atom::TAcc;
  const BatchedMatmulCase test_case{
      case_name, Batch, M, N, K, BroadcastB, BroadcastA};
  std::string name = std::string(suite) + "/real_batch/" + case_name +
      "/batch:" + std::to_string(Batch) +
      "/shape:" + std::to_string(M) + "x" + std::to_string(N) + "x" +
          std::to_string(K) +
      "/b_mode:" + (BroadcastB ? "shared" : "independent") +
      (BroadcastA ? "/a_mode:shared" : "");
  if constexpr (Mode != InputMode::Raw) {
    name += "/input_mode:" + std::string(input_mode_name<Mode>());
  }
  name +=
      "/memory:" + std::string(dtype_name<MemoryA>()) + "x" +
          dtype_name<MemoryB>() + "_to_" + dtype_name<MemoryC>() +
      "/compute:" + dtype_name<TA>() + "x" + dtype_name<TB>() +
          "_acc_" + dtype_name<Acc>() +
      input_pipeline_label<InPipeline, InPipelineB>() +
      c_input_type_label<MemoryCInput, MemoryC>() +
      "/output_pipeline:" + output_pipeline_name<OutPipeline>() +
      "/atom:" + atom_name<Atom>() +
      "/extent:" + extent_mode_name<Extents>() +
      "/arch:" + VECOPS_BENCH_ARCH_CODE;
  auto* registered = benchmark::RegisterBenchmark(
      name.c_str(), [test_case](benchmark::State& state) {
        run_batched_scenario_extent<
            Atom, MemoryA, MemoryB, MemoryC, InPipeline, OutPipeline,
            BroadcastB, BroadcastA, Mode, InPipelineB, MemoryCInput,
            Extents, Batch, M, N, K>(state, test_case);
      });
  registered->Unit(benchmark::kMicrosecond);
  ::vecops::bench::configure_registered_benchmark(
      registered, 0.02, 3)->ReportAggregatesOnly(true);
}

template <typename Atom, typename MemoryA, typename MemoryB, typename MemoryC,
          InputPipeline InPipeline, OutputPipeline OutPipeline,
          bool BroadcastB, bool BroadcastA, InputMode Mode,
          InputPipeline InPipelineB, typename MemoryCInput,
          nint_t Batch, nint_t M, nint_t N, nint_t K,
          int CaseShard = 0>
void register_batched_scenario_extent_pair(
    const char* suite, const char* case_name) {
#if defined(VECOPS_TARGET_SHARD_ACTIVE)
  static_assert(CaseShard >= 0);
  if constexpr ((VECOPS_TARGET_SHARD_INDEX %
                 ScenarioCatalogCaseShardCount) ==
                (CaseShard % ScenarioCatalogCaseShardCount)) {
    if constexpr (((VECOPS_TARGET_SHARD_INDEX /
                    ScenarioCatalogCaseShardCount) % 2) == 0) {
      register_batched_scenario_extent<
          Atom, MemoryA, MemoryB, MemoryC, InPipeline, OutPipeline,
          BroadcastB, BroadcastA, Mode, InPipelineB, MemoryCInput,
          ExtentMode::Dynamic, Batch, M, N, K>(suite, case_name);
    } else {
      register_batched_scenario_extent<
          Atom, MemoryA, MemoryB, MemoryC, InPipeline, OutPipeline,
          BroadcastB, BroadcastA, Mode, InPipelineB, MemoryCInput,
          ExtentMode::Const, Batch, M, N, K>(suite, case_name);
    }
  }
#else
  register_batched_scenario_extent<
      Atom, MemoryA, MemoryB, MemoryC, InPipeline, OutPipeline,
      BroadcastB, BroadcastA, Mode, InPipelineB, MemoryCInput,
      ExtentMode::Dynamic, Batch, M, N, K>(suite, case_name);
  register_batched_scenario_extent<
      Atom, MemoryA, MemoryB, MemoryC, InPipeline, OutPipeline,
      BroadcastB, BroadcastA, Mode, InPipelineB, MemoryCInput,
      ExtentMode::Const, Batch, M, N, K>(suite, case_name);
#endif
}

template <typename Atom, typename MemoryA, typename MemoryB, typename MemoryC,
          InputPipeline InPipeline, OutputPipeline OutPipeline,
          InputMode Mode, InputPipeline InPipelineB, typename MemoryCInput,
          typename MExtent, typename NExtent, typename KExtent>
void run_scenario_with_extents(
    benchmark::State& state, const MatmulCase& test_case,
    MExtent m_extent, NExtent n_extent, KExtent k_extent);

template <typename Atom, typename MemoryA, typename MemoryB, typename MemoryC,
          InputPipeline InPipeline, OutputPipeline OutPipeline,
          InputMode Mode = InputMode::Raw,
          InputPipeline InPipelineB = InPipeline,
          typename MemoryCInput = MemoryC>
void run_scenario(benchmark::State& state, const MatmulCase& test_case) {
  run_scenario_with_extents<
      Atom, MemoryA, MemoryB, MemoryC, InPipeline, OutPipeline, Mode,
      InPipelineB, MemoryCInput>(
          state, test_case, Any{test_case.m}, Any{test_case.n},
          Any{test_case.k});
}

template <typename Atom, typename MemoryA, typename MemoryB, typename MemoryC,
          InputPipeline InPipeline, OutputPipeline OutPipeline,
          InputMode Mode, InputPipeline InPipelineB, typename MemoryCInput,
          ExtentMode Extents, nint_t M, nint_t N, nint_t K>
void run_scenario_extent(
    benchmark::State& state, const MatmulCase& test_case) {
  if constexpr (Extents == ExtentMode::Dynamic) {
    run_scenario_with_extents<
        Atom, MemoryA, MemoryB, MemoryC, InPipeline, OutPipeline, Mode,
        InPipelineB, MemoryCInput>(
            state, test_case, Any{test_case.m}, Any{test_case.n},
            Any{test_case.k});
  } else {
    run_scenario_with_extents<
        Atom, MemoryA, MemoryB, MemoryC, InPipeline, OutPipeline, Mode,
        InPipelineB, MemoryCInput>(
            state, test_case, cint<M>, cint<N>, cint<K>);
  }
}

template <typename Atom, typename MemoryA, typename MemoryB, typename MemoryC,
          InputPipeline InPipeline, OutputPipeline OutPipeline,
          InputMode Mode, InputPipeline InPipelineB, typename MemoryCInput,
          ExtentMode Extents, nint_t M, nint_t N, nint_t K>
void register_scenario_extent(
    const char* suite, const char* case_name) {
  using TA = typename Atom::TA;
  using TB = typename Atom::TB;
  using Acc = typename Atom::TAcc;
  constexpr uint32_t ModeMask = mode_bit(Mode);
  const MatmulCase test_case{"scenario", case_name, M, N, K, ModeMask};
  std::string name = std::string(suite) + "/" + case_name +
      "/shape:" + std::to_string(M) + "x" + std::to_string(N) + "x" +
          std::to_string(K);
  if constexpr (Mode != InputMode::Raw) {
    name += "/input_mode:" + std::string(input_mode_name<Mode>());
  }
  name +=
      "/memory:" + std::string(dtype_name<MemoryA>()) + "x" +
          dtype_name<MemoryB>() + "_to_" + dtype_name<MemoryC>() +
      "/compute:" + dtype_name<TA>() + "x" + dtype_name<TB>() +
          "_acc_" + dtype_name<Acc>() +
      input_pipeline_label<InPipeline, InPipelineB>() +
      c_input_type_label<MemoryCInput, MemoryC>() +
      "/output_pipeline:" + output_pipeline_name<OutPipeline>() +
      "/atom:" + atom_name<Atom>() +
      "/extent:" + extent_mode_name<Extents>() +
      "/arch:" + VECOPS_BENCH_ARCH_CODE;
  auto* registered = benchmark::RegisterBenchmark(
      name.c_str(), [test_case](benchmark::State& state) {
        run_scenario_extent<
            Atom, MemoryA, MemoryB, MemoryC, InPipeline, OutPipeline, Mode,
            InPipelineB, MemoryCInput, Extents, M, N, K>(state, test_case);
      });
  registered->Unit(benchmark::kMicrosecond);
  ::vecops::bench::configure_registered_benchmark(
      registered, 0.02, 3)->ReportAggregatesOnly(true);
}

template <typename Atom, typename MemoryA, typename MemoryB, typename MemoryC,
          InputPipeline InPipeline, OutputPipeline OutPipeline,
          InputMode Mode, InputPipeline InPipelineB, typename MemoryCInput,
          nint_t M, nint_t N, nint_t K, int CaseShard = 0>
void register_scenario_extent_pair(
    const char* suite, const char* case_name) {
#if defined(VECOPS_TARGET_SHARD_ACTIVE)
  static_assert(CaseShard >= 0);
  if constexpr ((VECOPS_TARGET_SHARD_INDEX %
                 ScenarioCatalogCaseShardCount) ==
                (CaseShard % ScenarioCatalogCaseShardCount)) {
    if constexpr (((VECOPS_TARGET_SHARD_INDEX /
                    ScenarioCatalogCaseShardCount) % 2) == 0) {
      register_scenario_extent<
          Atom, MemoryA, MemoryB, MemoryC, InPipeline, OutPipeline, Mode,
          InPipelineB, MemoryCInput, ExtentMode::Dynamic, M, N, K>(
              suite, case_name);
    } else {
      register_scenario_extent<
          Atom, MemoryA, MemoryB, MemoryC, InPipeline, OutPipeline, Mode,
          InPipelineB, MemoryCInput, ExtentMode::Const, M, N, K>(
              suite, case_name);
    }
  }
#else
  register_scenario_extent<
      Atom, MemoryA, MemoryB, MemoryC, InPipeline, OutPipeline, Mode,
      InPipelineB, MemoryCInput, ExtentMode::Dynamic, M, N, K>(
          suite, case_name);
  register_scenario_extent<
      Atom, MemoryA, MemoryB, MemoryC, InPipeline, OutPipeline, Mode,
      InPipelineB, MemoryCInput, ExtentMode::Const, M, N, K>(
          suite, case_name);
#endif
}

template <typename Atom, typename MemoryA, typename MemoryB, typename MemoryC,
          nint_t KStep,
          InputPipeline InPipeline = InputPipeline::Convert,
          OutputPipeline OutPipeline = OutputPipeline::Convert,
          InputMode Mode = InputMode::Raw,
          InputPipeline InPipelineB = InPipeline,
          typename MemoryCInput = MemoryC, int CaseBase = 0>
void register_scenario_extent_catalog(const char* suite = "MatmulScenario") {
  register_scenario_extent_pair<
      Atom, MemoryA, MemoryB, MemoryC, InPipeline, OutPipeline, Mode,
      InPipelineB, MemoryCInput, 19, 23, 8 * KStep + 1,
      CaseBase + 0>(suite, "tail");
  register_scenario_extent_pair<
      Atom, MemoryA, MemoryB, MemoryC, InPipeline, OutPipeline, Mode,
      InPipelineB, MemoryCInput, 1, 1024, 1024,
      CaseBase + 1>(suite, "decode");
  register_scenario_extent_pair<
      Atom, MemoryA, MemoryB, MemoryC, InPipeline, OutPipeline, Mode,
      InPipelineB, MemoryCInput, 64, 256, 256,
      CaseBase + 2>(suite, "batch");
  register_scenario_extent_pair<
      Atom, MemoryA, MemoryB, MemoryC, InPipeline, OutPipeline, Mode,
      InPipelineB, MemoryCInput, 32, 1024, 256,
      CaseBase + 3>(suite, "wide");
}

template <typename Atom, typename MemoryA, typename MemoryB, typename MemoryC,
          InputPipeline InPipeline = InputPipeline::Convert,
          OutputPipeline OutPipeline = OutputPipeline::Convert,
          bool EnableBroadcastA = true,
          InputPipeline InPipelineB = InPipeline,
          typename MemoryCInput = MemoryC>
void register_batched_scenario(
    const std::vector<BatchedMatmulCase>& cases) {
  using TA = typename Atom::TA;
  using TB = typename Atom::TB;
  using Acc = typename Atom::TAcc;
  for (const auto& test_case : cases) {
    const std::string name =
        "MatmulScenario/real_batch/" + std::string(test_case.name) +
        "/batch:" + std::to_string(test_case.batch) +
        "/shape:" + std::to_string(test_case.m) + "x" +
            std::to_string(test_case.n) + "x" +
            std::to_string(test_case.k) +
        "/b_mode:" + (test_case.broadcast_b ? "shared" : "independent") +
        (test_case.broadcast_a ? "/a_mode:shared" : "") +
        "/memory:" + dtype_name<MemoryA>() + "x" +
            dtype_name<MemoryB>() + "_to_" + dtype_name<MemoryC>() +
        "/compute:" + dtype_name<TA>() + "x" + dtype_name<TB>() +
            "_acc_" + dtype_name<Acc>() +
        input_pipeline_label<InPipeline, InPipelineB>() +
        c_input_type_label<MemoryCInput, MemoryC>() +
        "/output_pipeline:" + output_pipeline_name<OutPipeline>() +
        "/atom:" + atom_name<Atom>() +
        "/extent:Dynamic" +
        "/arch:" + VECOPS_BENCH_ARCH_CODE;
    auto configure = [](auto* registered) {
      registered->Unit(benchmark::kMicrosecond);
      ::vecops::bench::configure_registered_benchmark(
          registered, 0.02, 3)->ReportAggregatesOnly(true);
    };
    if constexpr (EnableBroadcastA) {
      if (test_case.broadcast_a) {
        configure(benchmark::RegisterBenchmark(
            name.c_str(), [test_case](benchmark::State& state) {
              run_batched_scenario<
                  Atom, MemoryA, MemoryB, MemoryC,
                  InPipeline, OutPipeline, false, true, InputMode::Raw,
                  InPipelineB, MemoryCInput>(state, test_case);
            }));
        continue;
      }
    }
    if (test_case.broadcast_b) {
      configure(benchmark::RegisterBenchmark(
          name.c_str(), [test_case](benchmark::State& state) {
            run_batched_scenario<
                Atom, MemoryA, MemoryB, MemoryC,
                InPipeline, OutPipeline, true, false, InputMode::Raw,
                InPipelineB, MemoryCInput>(state, test_case);
          }));
    } else {
      configure(benchmark::RegisterBenchmark(
          name.c_str(), [test_case](benchmark::State& state) {
            run_batched_scenario<
                Atom, MemoryA, MemoryB, MemoryC,
                InPipeline, OutPipeline, false, false, InputMode::Raw,
                InPipelineB, MemoryCInput>(state, test_case);
          }));
    }
  }
}

template <typename Atom, typename MemoryA, typename MemoryB, typename MemoryC,
          InputPipeline InPipeline = InputPipeline::Convert,
          OutputPipeline OutPipeline = OutputPipeline::Convert,
          bool EnableBroadcastA = true,
          InputPipeline InPipelineB = InPipeline,
          typename MemoryCInput = MemoryC, nint_t KStep>
void register_batched_scenario(BatchedScenarioCaseCatalog<KStep>) {
#define VECOPS_REGISTER_BATCH_PAIR(NAME, B, M, N, K, SHARED_B, SHARED_A, CASE_INDEX) \
  register_batched_scenario_extent_pair< \
      Atom, MemoryA, MemoryB, MemoryC, InPipeline, OutPipeline, \
      SHARED_B, (EnableBroadcastA && SHARED_A), InputMode::Raw, \
      InPipelineB, MemoryCInput, B, M, N, K, CASE_INDEX>("MatmulScenario", NAME)
  VECOPS_REGISTER_BATCH_PAIR(
      "independent_tail", 4, 19, 23, 8 * KStep + 1, false, false, 4);
  VECOPS_REGISTER_BATCH_PAIR(
      "shared_weight_tail", 8, 1, 47, 2 * KStep + 1, true, false, 5);
  VECOPS_REGISTER_BATCH_PAIR(
      "shared_weight_tiny", 8, 1, 32, 64, true, false, 6);
  VECOPS_REGISTER_BATCH_PAIR(
      "shared_weight_small", 8, 1, 64, 128, true, false, 7);
  VECOPS_REGISTER_BATCH_PAIR(
      "shared_weight_small_long_k", 8, 1, 64, 256, true, false, 8);
  VECOPS_REGISTER_BATCH_PAIR(
      "shared_weight_wide_short_k", 8, 1, 128, 64, true, false, 9);
  VECOPS_REGISTER_BATCH_PAIR(
      "shared_weight_decode", 8, 1, 256, 256, true, false, 10);
  VECOPS_REGISTER_BATCH_PAIR(
      "shared_weight_mlp_projection", 8, 1, 1024, 4096, true, false, 11);
  VECOPS_REGISTER_BATCH_PAIR(
      "shared_weight_qwen_o_proj", 8, 1, 3584, 3584, true, false, 12);
  VECOPS_REGISTER_BATCH_PAIR(
      "shared_weight_qwen_qkv", 8, 1, 4608, 3584, true, false, 13);
  VECOPS_REGISTER_BATCH_PAIR(
      "shared_activation_scalar", 16, 1, 1, 64, false, true, 14);
  VECOPS_REGISTER_BATCH_PAIR(
      "shared_activation_tail", 8, 1, 8, 2 * KStep + 1, false, true, 15);
  VECOPS_REGISTER_BATCH_PAIR(
      "shared_activation_boundary", 4, 1, 16, 64, false, true, 16);
#undef VECOPS_REGISTER_BATCH_PAIR
}

template <typename Atom, typename MemoryA, typename MemoryB, typename MemoryC,
          InputPipeline InPipeline = InputPipeline::Convert,
          OutputPipeline OutPipeline = OutputPipeline::Convert,
          bool EnableBroadcastA = true,
          InputPipeline InPipelineB = InPipeline,
          typename MemoryCInput = MemoryC, nint_t KStep>
void register_batched_scenario(BatchedFusionScenarioCaseCatalog<KStep>) {
#define VECOPS_REGISTER_FUSION_BATCH_PAIR(NAME, B, M, N, K, SHARED_B, CASE_INDEX) \
  register_batched_scenario_extent_pair< \
      Atom, MemoryA, MemoryB, MemoryC, InPipeline, OutPipeline, \
      SHARED_B, false, InputMode::Raw, InPipelineB, MemoryCInput, \
      B, M, N, K, CASE_INDEX>("MatmulFusion", NAME)
  VECOPS_REGISTER_FUSION_BATCH_PAIR(
      "fusion_independent_tail", 4, 19, 23, 8 * KStep + 1, false, 8);
  VECOPS_REGISTER_FUSION_BATCH_PAIR(
      "fusion_shared_small", 8, 1, 128, 128, true, 9);
  VECOPS_REGISTER_FUSION_BATCH_PAIR(
      "fusion_shared_decode", 8, 1, 256, 256, true, 10);
  VECOPS_REGISTER_FUSION_BATCH_PAIR(
      "fusion_shared_prefill", 4, 16, 512, 512, true, 11);
#undef VECOPS_REGISTER_FUSION_BATCH_PAIR
}

template <typename Atom, typename MemoryA, typename MemoryB, typename MemoryC,
          InputPipeline InPipeline, OutputPipeline OutPipeline,
          InputMode Mode,
          InputPipeline InPipelineB = InPipeline,
          typename MemoryCInput = MemoryC>
void register_batched_packed_b_scenario(
    const std::vector<BatchedMatmulCase>& cases) {
  static_assert(
      Mode == InputMode::Raw || Mode == InputMode::PackedB ||
      Mode == InputMode::OnlinePackedB);
  using TA = typename Atom::TA;
  using TB = typename Atom::TB;
  using Acc = typename Atom::TAcc;
  for (const auto& test_case : cases) {
    if (!test_case.broadcast_b) continue;
    const std::string name =
        "MatmulPackedScenario/real_batch/" + std::string(test_case.name) +
        "/batch:" + std::to_string(test_case.batch) +
        "/shape:" + std::to_string(test_case.m) + "x" +
            std::to_string(test_case.n) + "x" +
            std::to_string(test_case.k) +
        "/input_mode:" + input_mode_name<Mode>() +
        "/memory:" + dtype_name<MemoryA>() + "x" +
            dtype_name<MemoryB>() + "_to_" + dtype_name<MemoryC>() +
        "/compute:" + dtype_name<TA>() + "x" + dtype_name<TB>() +
            "_acc_" + dtype_name<Acc>() +
        input_pipeline_label<InPipeline, InPipelineB>() +
        c_input_type_label<MemoryCInput, MemoryC>() +
        "/output_pipeline:" + output_pipeline_name<OutPipeline>() +
        "/atom:" + atom_name<Atom>() +
        "/extent:Dynamic" +
        "/arch:" + VECOPS_BENCH_ARCH_CODE;
    auto* registered = benchmark::RegisterBenchmark(
        name.c_str(), [test_case](benchmark::State& state) {
          run_batched_scenario<
              Atom, MemoryA, MemoryB, MemoryC,
              InPipeline, OutPipeline, true, false, Mode,
              InPipelineB, MemoryCInput>(state, test_case);
        });
    registered->Unit(benchmark::kMicrosecond);
    ::vecops::bench::configure_registered_benchmark(
        registered, 0.02, 3)->ReportAggregatesOnly(true);
  }
}

template <typename Atom, typename MemoryA, typename MemoryB, typename MemoryC,
          InputPipeline InPipeline, OutputPipeline OutPipeline,
          InputMode Mode, InputPipeline InPipelineB = InPipeline,
          typename MemoryCInput = MemoryC, int CaseBase = 0,
          nint_t KStep>
void register_batched_packed_b_scenario(
    BatchedScenarioCaseCatalog<KStep>) {
#define VECOPS_REGISTER_PACKED_BATCH_PAIR(NAME, B, M, N, K, CASE_OFFSET) \
  register_batched_scenario_extent_pair< \
      Atom, MemoryA, MemoryB, MemoryC, InPipeline, OutPipeline, \
      true, false, Mode, InPipelineB, MemoryCInput, B, M, N, K, \
      CaseBase + CASE_OFFSET>( \
          "MatmulPacked", NAME)
  VECOPS_REGISTER_PACKED_BATCH_PAIR(
      "shared_weight_tail", 8, 1, 47, 2 * KStep + 1, 0);
  VECOPS_REGISTER_PACKED_BATCH_PAIR(
      "shared_weight_tiny", 8, 1, 32, 64, 1);
  VECOPS_REGISTER_PACKED_BATCH_PAIR(
      "shared_weight_small", 8, 1, 64, 128, 2);
  VECOPS_REGISTER_PACKED_BATCH_PAIR(
      "shared_weight_small_long_k", 8, 1, 64, 256, 3);
  VECOPS_REGISTER_PACKED_BATCH_PAIR(
      "shared_weight_wide_short_k", 8, 1, 128, 64, 4);
  VECOPS_REGISTER_PACKED_BATCH_PAIR(
      "shared_weight_decode", 8, 1, 256, 256, 5);
  VECOPS_REGISTER_PACKED_BATCH_PAIR(
      "shared_weight_mlp_projection", 8, 1, 1024, 4096, 6);
  VECOPS_REGISTER_PACKED_BATCH_PAIR(
      "shared_weight_qwen_o_proj", 8, 1, 3584, 3584, 7);
  VECOPS_REGISTER_PACKED_BATCH_PAIR(
      "shared_weight_qwen_qkv", 8, 1, 4608, 3584, 8);
#undef VECOPS_REGISTER_PACKED_BATCH_PAIR
}

template <typename Atom, typename MemoryA, typename MemoryB, typename MemoryC,
          InputPipeline InPipeline = InputPipeline::Convert,
          OutputPipeline OutPipeline = OutputPipeline::Convert,
          InputPipeline InPipelineB = InPipeline,
          typename MemoryCInput = MemoryC>
void register_all_batched_packed_b_modes(
    const std::vector<BatchedMatmulCase>& cases) {
  register_batched_packed_b_scenario<
      Atom, MemoryA, MemoryB, MemoryC, InPipeline, OutPipeline,
      InputMode::Raw, InPipelineB, MemoryCInput>(cases);
  register_batched_packed_b_scenario<
      Atom, MemoryA, MemoryB, MemoryC, InPipeline, OutPipeline,
      InputMode::PackedB, InPipelineB, MemoryCInput>(cases);
  register_batched_packed_b_scenario<
      Atom, MemoryA, MemoryB, MemoryC, InPipeline, OutPipeline,
      InputMode::OnlinePackedB, InPipelineB, MemoryCInput>(cases);
}

template <typename Atom, typename MemoryA, typename MemoryB, typename MemoryC,
          InputPipeline InPipeline = InputPipeline::Convert,
          OutputPipeline OutPipeline = OutputPipeline::Convert,
          InputPipeline InPipelineB = InPipeline,
          typename MemoryCInput = MemoryC, nint_t KStep>
void register_all_batched_packed_b_modes(
    BatchedScenarioCaseCatalog<KStep> cases) {
  register_batched_packed_b_scenario<
      Atom, MemoryA, MemoryB, MemoryC, InPipeline, OutPipeline,
      InputMode::Raw, InPipelineB, MemoryCInput, 0>(cases);
  register_batched_packed_b_scenario<
      Atom, MemoryA, MemoryB, MemoryC, InPipeline, OutPipeline,
      InputMode::PackedB, InPipelineB, MemoryCInput, 9>(cases);
  register_batched_packed_b_scenario<
      Atom, MemoryA, MemoryB, MemoryC, InPipeline, OutPipeline,
      InputMode::OnlinePackedB, InPipelineB, MemoryCInput, 18>(cases);
}

struct WeightReuseCase {
  const char* name;
  nint_t batch;
  nint_t m;
  nint_t n;
  nint_t k;
  nint_t calls_per_pack;
};

inline std::vector<WeightReuseCase> make_weight_reuse_cases() {
  std::vector<WeightReuseCase> cases;
  constexpr std::array<nint_t, 5> BatchSizes{1, 2, 4, 8, 16};
  constexpr std::array<nint_t, 5> ReuseCounts{1, 2, 4, 8, 16};
  for (const nint_t batch : BatchSizes) {
    for (const nint_t reuse : ReuseCounts) {
      cases.push_back(
          {"decode", batch, 1, 256, 256, reuse});
    }
  }
  constexpr std::array<nint_t, 3> SmallReuseCounts{1, 4, 16};
  for (const nint_t batch : BatchSizes) {
    for (const nint_t reuse : SmallReuseCounts) {
      cases.push_back(
          {"small", batch, 1, 64, 128, reuse});
    }
  }
  for (const nint_t reuse : std::array<nint_t, 4>{1, 2, 4, 8}) {
    cases.push_back(
        {"mlp_projection", 8, 1, 1024, 4096, reuse});
  }
  for (const nint_t reuse : std::array<nint_t, 3>{1, 2, 4}) {
    cases.push_back(
        {"qwen_o_proj", 8, 1, 3584, 3584, reuse});
  }
  return cases;
}

struct WeightReuseCaseCatalog {
  operator std::vector<WeightReuseCase>() const {
    return make_weight_reuse_cases();
  }
};

#define weight_reuse_cases() \
  ::vecops::bench::matmul::WeightReuseCaseCatalog{}

template <typename Atom, typename MemoryA, typename MemoryB, typename MemoryC,
          InputPipeline InPipeline, OutputPipeline OutPipeline,
          InputMode Mode,
          InputPipeline InPipelineB = InPipeline,
          typename MemoryCInput = MemoryC>
void register_batched_weight_reuse_mode(
    const std::vector<WeightReuseCase>& cases) {
  static_assert(
      Mode == InputMode::Raw || Mode == InputMode::OnlinePackedB);
  using TA = typename Atom::TA;
  using TB = typename Atom::TB;
  using Acc = typename Atom::TAcc;
  for (const auto& test_case : cases) {
    const std::string strategy = Mode == InputMode::Raw
        ? "raw"
        : "pack_once_then_reuse";
    const std::string name =
        "MatmulWeightReuse/" + std::string(test_case.name) +
        "/batch:" + std::to_string(test_case.batch) +
        "/shape:" + std::to_string(test_case.m) + "x" +
            std::to_string(test_case.n) + "x" +
            std::to_string(test_case.k) +
        "/calls_per_pack:" + std::to_string(test_case.calls_per_pack) +
        "/strategy:" + strategy +
        "/memory:" + dtype_name<MemoryA>() + "x" +
            dtype_name<MemoryB>() + "_to_" + dtype_name<MemoryC>() +
        "/compute:" + dtype_name<TA>() + "x" + dtype_name<TB>() +
            "_acc_" + dtype_name<Acc>() +
        input_pipeline_label<InPipeline, InPipelineB>() +
        c_input_type_label<MemoryCInput, MemoryC>() +
        "/output_pipeline:" + output_pipeline_name<OutPipeline>() +
        "/atom:" + atom_name<Atom>() +
        "/extent:Dynamic" +
        "/arch:" + VECOPS_BENCH_ARCH_CODE;
    auto* registered = benchmark::RegisterBenchmark(
        name.c_str(), [test_case](benchmark::State& state) {
          const BatchedMatmulCase batched_case{
              test_case.name, test_case.batch, test_case.m,
              test_case.n, test_case.k, true};
          run_batched_scenario<
              Atom, MemoryA, MemoryB, MemoryC,
              InPipeline, OutPipeline, true, false, Mode,
              InPipelineB, MemoryCInput>(
                  state, batched_case, test_case.calls_per_pack);
        });
    registered->Unit(benchmark::kMicrosecond);
    ::vecops::bench::configure_registered_benchmark(
        registered, 0.01, 3)->ReportAggregatesOnly(true);
  }
}

template <typename Atom, typename MemoryA, typename MemoryB, typename MemoryC,
          InputPipeline InPipeline = InputPipeline::Convert,
          OutputPipeline OutPipeline = OutputPipeline::Convert,
          InputPipeline InPipelineB = InPipeline,
          typename MemoryCInput = MemoryC>
void register_all_batched_weight_reuse_modes(
    const std::vector<WeightReuseCase>& cases) {
  register_batched_weight_reuse_mode<
      Atom, MemoryA, MemoryB, MemoryC, InPipeline, OutPipeline,
      InputMode::Raw, InPipelineB, MemoryCInput>(cases);
  register_batched_weight_reuse_mode<
      Atom, MemoryA, MemoryB, MemoryC, InPipeline, OutPipeline,
      InputMode::OnlinePackedB, InPipelineB, MemoryCInput>(cases);
}

template <typename Atom, typename MemoryA, typename MemoryB, typename MemoryC,
          InputPipeline InPipeline, OutputPipeline OutPipeline,
          InputMode Mode, InputPipeline InPipelineB, typename MemoryCInput,
          ExtentMode Extents,
          nint_t Batch, nint_t M, nint_t N, nint_t K, nint_t CallsPerPack>
void register_weight_reuse_extent(const char* case_name) {
  static_assert(
      Mode == InputMode::Raw || Mode == InputMode::OnlinePackedB);
  using TA = typename Atom::TA;
  using TB = typename Atom::TB;
  using Acc = typename Atom::TAcc;
  const BatchedMatmulCase test_case{
      case_name, Batch, M, N, K, true, false};
  const std::string strategy = Mode == InputMode::Raw
      ? "raw"
      : "pack_once_then_reuse";
  const std::string name =
      "MatmulWeightReuse/" + std::string(case_name) +
      "/batch:" + std::to_string(Batch) +
      "/shape:" + std::to_string(M) + "x" + std::to_string(N) + "x" +
          std::to_string(K) +
      "/calls_per_pack:" + std::to_string(CallsPerPack) +
      "/strategy:" + strategy +
      "/memory:" + dtype_name<MemoryA>() + "x" + dtype_name<MemoryB>() +
          "_to_" + dtype_name<MemoryC>() +
      "/compute:" + dtype_name<TA>() + "x" + dtype_name<TB>() +
          "_acc_" + dtype_name<Acc>() +
      input_pipeline_label<InPipeline, InPipelineB>() +
      c_input_type_label<MemoryCInput, MemoryC>() +
      "/output_pipeline:" + output_pipeline_name<OutPipeline>() +
      "/atom:" + atom_name<Atom>() +
      "/extent:" + extent_mode_name<Extents>() +
      "/arch:" + VECOPS_BENCH_ARCH_CODE;
  auto* registered = benchmark::RegisterBenchmark(
      name.c_str(), [test_case](benchmark::State& state) {
        if constexpr (Extents == ExtentMode::Dynamic) {
          run_batched_scenario_with_extents<
              Atom, MemoryA, MemoryB, MemoryC, InPipeline, OutPipeline,
              true, false, Mode, InPipelineB, MemoryCInput>(
                  state, test_case, Any{Batch}, Any{M}, Any{N}, Any{K},
                  CallsPerPack);
        } else {
          run_batched_scenario_with_extents<
              Atom, MemoryA, MemoryB, MemoryC, InPipeline, OutPipeline,
              true, false, Mode, InPipelineB, MemoryCInput>(
                  state, test_case, cint<Batch>, cint<M>, cint<N>, cint<K>,
                  CallsPerPack);
        }
      });
  registered->Unit(benchmark::kMicrosecond);
  ::vecops::bench::configure_registered_benchmark(
      registered, 0.01, 3)->ReportAggregatesOnly(true);
}

template <typename Atom, typename MemoryA, typename MemoryB, typename MemoryC,
          InputPipeline InPipeline, OutputPipeline OutPipeline,
          InputMode Mode, InputPipeline InPipelineB, typename MemoryCInput,
          nint_t Batch, nint_t M, nint_t N, nint_t K, nint_t CallsPerPack>
void register_weight_reuse_extent_pair(const char* case_name) {
#if defined(VECOPS_TARGET_SHARD_ACTIVE)
  if constexpr ((VECOPS_TARGET_SHARD_INDEX % 2) == 0) {
    register_weight_reuse_extent<
        Atom, MemoryA, MemoryB, MemoryC, InPipeline, OutPipeline, Mode,
        InPipelineB, MemoryCInput, ExtentMode::Dynamic,
        Batch, M, N, K, CallsPerPack>(case_name);
  } else {
    register_weight_reuse_extent<
        Atom, MemoryA, MemoryB, MemoryC, InPipeline, OutPipeline, Mode,
        InPipelineB, MemoryCInput, ExtentMode::Const,
        Batch, M, N, K, CallsPerPack>(case_name);
  }
#else
  register_weight_reuse_extent<
      Atom, MemoryA, MemoryB, MemoryC, InPipeline, OutPipeline, Mode,
      InPipelineB, MemoryCInput, ExtentMode::Dynamic,
      Batch, M, N, K, CallsPerPack>(case_name);
  register_weight_reuse_extent<
      Atom, MemoryA, MemoryB, MemoryC, InPipeline, OutPipeline, Mode,
      InPipelineB, MemoryCInput, ExtentMode::Const,
      Batch, M, N, K, CallsPerPack>(case_name);
#endif
}

template <typename Atom, typename MemoryA, typename MemoryB, typename MemoryC,
          InputPipeline InPipeline, OutputPipeline OutPipeline,
          InputMode Mode, InputPipeline InPipelineB, typename MemoryCInput,
          nint_t Batch, nint_t M, nint_t N, nint_t K,
          nint_t... Calls>
void register_weight_reuse_counts(
    const char* case_name, std::integer_sequence<nint_t, Calls...>) {
  (register_weight_reuse_extent_pair<
       Atom, MemoryA, MemoryB, MemoryC, InPipeline, OutPipeline, Mode,
       InPipelineB, MemoryCInput, Batch, M, N, K, Calls>(case_name), ...);
}

template <typename Atom, typename MemoryA, typename MemoryB, typename MemoryC,
          InputPipeline InPipeline, OutputPipeline OutPipeline,
          InputMode Mode, InputPipeline InPipelineB, typename MemoryCInput>
void register_weight_reuse_catalog_mode() {
#define VECOPS_REGISTER_REUSE_BATCH(BATCH) \
  register_weight_reuse_counts< \
      Atom, MemoryA, MemoryB, MemoryC, InPipeline, OutPipeline, Mode, \
      InPipelineB, MemoryCInput, BATCH, 1, 256, 256>( \
          "decode", std::integer_sequence<nint_t, 1, 2, 4, 8, 16>{}); \
  register_weight_reuse_counts< \
      Atom, MemoryA, MemoryB, MemoryC, InPipeline, OutPipeline, Mode, \
      InPipelineB, MemoryCInput, BATCH, 1, 64, 128>( \
          "small", std::integer_sequence<nint_t, 1, 4, 16>{})
  VECOPS_REGISTER_REUSE_BATCH(1);
  VECOPS_REGISTER_REUSE_BATCH(2);
  VECOPS_REGISTER_REUSE_BATCH(4);
  VECOPS_REGISTER_REUSE_BATCH(8);
  VECOPS_REGISTER_REUSE_BATCH(16);
#undef VECOPS_REGISTER_REUSE_BATCH
  register_weight_reuse_counts<
      Atom, MemoryA, MemoryB, MemoryC, InPipeline, OutPipeline, Mode,
      InPipelineB, MemoryCInput, 8, 1, 1024, 4096>(
          "mlp_projection", std::integer_sequence<nint_t, 1, 2, 4, 8>{});
  register_weight_reuse_counts<
      Atom, MemoryA, MemoryB, MemoryC, InPipeline, OutPipeline, Mode,
      InPipelineB, MemoryCInput, 8, 1, 3584, 3584>(
          "qwen_o_proj", std::integer_sequence<nint_t, 1, 2, 4>{});
}

template <typename Atom, typename MemoryA, typename MemoryB, typename MemoryC,
          InputPipeline InPipeline = InputPipeline::Convert,
          OutputPipeline OutPipeline = OutputPipeline::Convert,
          InputPipeline InPipelineB = InPipeline,
          typename MemoryCInput = MemoryC>
void register_all_batched_weight_reuse_modes(WeightReuseCaseCatalog) {
  register_weight_reuse_catalog_mode<
      Atom, MemoryA, MemoryB, MemoryC, InPipeline, OutPipeline,
      InputMode::Raw, InPipelineB, MemoryCInput>();
  register_weight_reuse_catalog_mode<
      Atom, MemoryA, MemoryB, MemoryC, InPipeline, OutPipeline,
      InputMode::OnlinePackedB, InPipelineB, MemoryCInput>();
}

template <typename Atom, typename MemoryA, typename MemoryB, typename MemoryC,
          InputPipeline InPipeline, OutputPipeline OutPipeline,
          InputMode Mode, InputPipeline InPipelineB,
          typename MemoryCInput,
          typename MExtent, typename NExtent, typename KExtent>
void run_scenario_with_extents(
    benchmark::State& state, const MatmulCase& test_case,
    MExtent m_extent, NExtent n_extent, KExtent k_extent) {
  using TA = typename Atom::TA;
  using TB = typename Atom::TB;
  using Acc = typename Atom::TAcc;
  const nint_t m = test_case.m;
  const nint_t n = test_case.n;
  const nint_t k = test_case.k;
  constexpr bool UsesAsymmetricCorrection =
      uses_asymmetric_correction_v<InPipeline> ||
      uses_asymmetric_correction_v<InPipelineB>;
  static_assert(
      supported_asymmetric_pipeline_pair_v<InPipeline, InPipelineB>,
      "only an asymmetric A pipeline with a direct B pipeline is supported");

  std::vector<MemoryA> a(static_cast<std::size_t>(m * k));
  std::vector<MemoryB> b(static_cast<std::size_t>(n * k));
  std::vector<MemoryC> c(static_cast<std::size_t>(m * n));
  constexpr bool UsesBias = uses_bias_prologue_v<OutPipeline>;
  std::vector<MemoryCInput> c_input(static_cast<std::size_t>(
      UsesBias ? n : m * n));
  constexpr bool DualZeroPoint =
      uses_dual_asymmetric_correction_v<InPipeline>;
  std::vector<Acc> asymmetric_correction(static_cast<std::size_t>(
      DualZeroPoint ? m * n : n));
  Acc dynamic_scale = static_cast<Acc>(ScenarioDynamicScale);
  std::vector<Acc> column_scales(
      static_cast<std::size_t>(n + 64));
  std::vector<float32_t> quant_column_scales(
      static_cast<std::size_t>(n + 64));
  std::vector<float32_t> row_quant_multipliers(
      static_cast<std::size_t>(m));
  std::vector<float32_t> row_dequant_scales(
      static_cast<std::size_t>(m));
  std::vector<float32_t> weight_column_scales(
      static_cast<std::size_t>(n + 64));
  for (nint_t col = 0; col < n + 64; ++col) {
    column_scales[static_cast<std::size_t>(col)] =
        static_cast<Acc>(scenario_column_scale(col));
    quant_column_scales[static_cast<std::size_t>(col)] =
        scenario_quant_column_scale(col);
    weight_column_scales[static_cast<std::size_t>(col)] =
        0.03125f + static_cast<float32_t>(col % 5) * 0.0078125f;
  }
  for (nint_t row = 0; row < m; ++row) {
    const float32_t multiplier =
        static_cast<float32_t>(4 << (row % 3));
    row_quant_multipliers[static_cast<std::size_t>(row)] = multiplier;
    row_dequant_scales[static_cast<std::size_t>(row)] = 1.0f / multiplier;
  }
  int32_t output_zero_point = 7;
  int32_t input_zero_point = 7;
  const ScenarioInputParameters input_parameters{
      row_quant_multipliers.data(), &input_zero_point, 0};
  const ScenarioOutputParameters<Acc> output_parameters{
      &dynamic_scale, column_scales.data(), quant_column_scales.data(),
      &output_zero_point, row_dequant_scales.data(),
      weight_column_scales.data(), 0};
  fill_scenario_input<MemoryA, TA, InPipeline>(a, 3);
  fill_scenario_input<MemoryB, TB, InPipelineB>(b, 11);
  fill_scenario_input<MemoryCInput, Acc, InputPipeline::Convert>(c_input, 5);
  if constexpr (uses_runtime_per_row_quantization_v<InPipeline>) {
    static_assert(std::same_as<MemoryA, float32_t>);
    for (nint_t row = 0; row < m; ++row) {
      const float32_t scale =
          row_dequant_scales[static_cast<std::size_t>(row)];
      for (nint_t kk = 0; kk < k; ++kk) {
        const int logical = static_cast<int>((row * k + kk) % 7) - 3;
        a[static_cast<std::size_t>(row * k + kk)] =
            static_cast<float32_t>(logical) * scale;
      }
    }
  }
  if constexpr (UsesAsymmetricCorrection) {
    static_assert(std::same_as<TA, uint8_t>);
    if constexpr (DualZeroPoint) {
      static_assert(std::same_as<TB, uint8_t>);
      constexpr Acc ZeroPointA = 3;
      constexpr Acc ZeroPointB = 5;
      for (nint_t row = 0; row < m; ++row) {
        Acc sum_a{};
        for (nint_t kk = 0; kk < k; ++kk) {
          sum_a += static_cast<Acc>(reference_input<
              ::vecops::matmul::Operand::A, TA, MemoryA, InPipeline>(
              a[static_cast<std::size_t>(row * k + kk)]));
        }
        for (nint_t col = 0; col < n; ++col) {
          Acc sum_b{};
          for (nint_t kk = 0; kk < k; ++kk) {
            sum_b += static_cast<Acc>(reference_input<
                ::vecops::matmul::Operand::B, TB, MemoryB, InPipelineB>(
                b[static_cast<std::size_t>(col * k + kk)]));
          }
          asymmetric_correction[static_cast<std::size_t>(row * n + col)] =
              -ZeroPointA * sum_b - ZeroPointB * sum_a +
              static_cast<Acc>(k) * ZeroPointA * ZeroPointB;
        }
      }
    } else {
      static_assert(std::same_as<TB, int8_t>);
      const Acc zero_point_a =
          uses_runtime_per_row_quantization_v<InPipeline>
          ? static_cast<Acc>(input_zero_point)
          : Acc{3};
      for (nint_t col = 0; col < n; ++col) {
        Acc sum{};
        for (nint_t kk = 0; kk < k; ++kk) {
          sum += static_cast<Acc>(reference_input<
              ::vecops::matmul::Operand::B, TB, MemoryB, InPipelineB>(
              b[static_cast<std::size_t>(col * k + kk)]));
        }
        asymmetric_correction[static_cast<std::size_t>(col)] =
            -zero_point_a * sum;
      }
    }
  }

  const auto a_tensor = tensor::make_tensor(
      a.data(), tensor::make_layout(tensor::make_shape(m_extent, k_extent)));
  const auto b_tensor = tensor::make_tensor(
      b.data(), tensor::make_layout(tensor::make_shape(n_extent, k_extent)));
  const auto c_tensor = tensor::make_tensor(
      c.data(), tensor::make_layout(tensor::make_shape(m_extent, n_extent)));
  const auto c_input_layout = [&] {
    if constexpr (UsesBias) {
      return tensor::make_layout(
          tensor::make_shape(m_extent, n_extent),
          tensor::make_strides(cint<0>, cint<1>));
    } else {
      return tensor::make_layout(tensor::make_shape(m_extent, n_extent));
    }
  }();
  const auto c_input_tensor =
      tensor::make_tensor(c_input.data(), c_input_layout);
  const auto a_operand = make_scenario_input<
      ::vecops::matmul::Operand::A, TA, InPipeline>(a_tensor, input_parameters);
  const auto b_operand = make_scenario_input<
      ::vecops::matmul::Operand::B, TB, InPipelineB>(b_tensor);
  const auto c_output = make_scenario_output<Acc, OutPipeline>(
      c_tensor, output_parameters);

  const auto packed_a_layout = ::vecops::matmul::packed_layout<
      Atom, ::vecops::matmul::Operand::A>(a_tensor.layout());
  const auto packed_b_layout = ::vecops::matmul::packed_layout<
      Atom, ::vecops::matmul::Operand::B>(b_tensor.layout());
  const nint_t packed_a_elements = tensor::numel(packed_a_layout);
  const nint_t packed_b_elements = tensor::numel(packed_b_layout);
  const nint_t packed_a_bytes = scenario_packs_a_v<Mode>
      ? packed_a_elements * static_cast<nint_t>(sizeof(TA))
      : 0;
  const nint_t packed_b_bytes = scenario_packs_b_v<Mode>
      ? packed_b_elements * static_cast<nint_t>(sizeof(TB))
      : 0;
  kernel::Workspace packed_a_storage(packed_a_bytes + 64);
  kernel::Workspace packed_b_storage(packed_b_bytes + 64);
  auto packed_a_workspace = packed_a_storage.view();
  auto packed_b_workspace = packed_b_storage.view();
  auto* packed_a = scenario_packs_a_v<Mode>
      ? static_cast<TA*>(packed_a_workspace.allocate(packed_a_bytes, 64))
      : static_cast<TA*>(nullptr);
  auto* packed_b = scenario_packs_b_v<Mode>
      ? static_cast<TB*>(packed_b_workspace.allocate(packed_b_bytes, 64))
      : static_cast<TB*>(nullptr);
  const auto packed_a_tensor = tensor::make_tensor(
      packed_a, packed_a_layout);
  const auto packed_b_tensor = tensor::make_tensor(
      packed_b, packed_b_layout);
  const auto packed_b_compensation_tensor = tensor::make_tensor(
      asymmetric_correction.data(),
      tensor::make_layout(tensor::make_shape(n_extent)));

  auto pack_a_once = [&](auto& execution) VECOPS_INLINE_LAMBDA {
    if constexpr (scenario_packs_a_v<Mode>) {
      ::vecops::matmul::details::run_matmul_pack<Atom, ::vecops::matmul::Operand::A>(
          execution, a_operand, packed_a_tensor);
    }
  };
  auto pack_b_once = [&](auto& execution) VECOPS_INLINE_LAMBDA {
    if constexpr (scenario_packs_b_v<Mode>) {
      if constexpr (uses_single_asymmetric_correction_v<InPipeline>) {
#if defined(ARCH_X86_FAMILY)
        ::vecops::matmul::details::run_matmul_pack_b_compensated<Atom>(
            execution, b_operand, packed_b_tensor,
            packed_b_compensation_tensor,
            uses_runtime_per_row_quantization_v<InPipeline>
                ? input_zero_point
                : int32_t{3});
#else
        ::vecops::matmul::details::run_matmul_pack<Atom, ::vecops::matmul::Operand::B>(
            execution, b_operand, packed_b_tensor);
#endif
      } else {
        ::vecops::matmul::details::run_matmul_pack<Atom, ::vecops::matmul::Operand::B>(
            execution, b_operand, packed_b_tensor);
      }
    }
  };
  ExecutionSession initial_pack_execution{};
  pack_a_once(initial_pack_execution);
  pack_b_once(initial_pack_execution);

  auto make_operation = [&](const auto& selected_a, const auto& selected_b) {
    if constexpr (UsesAsymmetricCorrection) {
      static_assert(
          OutPipeline == OutputPipeline::AsymmetricDequantize ||
          OutPipeline == OutputPipeline::RuntimePerRowColumnDequantize);
      const auto correction_tensor = tensor::make_tensor(
          asymmetric_correction.data(),
          [&] {
            if constexpr (DualZeroPoint) {
              return tensor::make_layout(
                  tensor::make_shape(m_extent, n_extent));
            } else {
              return tensor::make_layout(
                  tensor::make_shape(m_extent, n_extent),
                  tensor::make_strides(cint<0>, cint<1>));
            }
          }());
      return make_benchmark_matmul_invocation(ops::MatmulConfig<Atom>{},
          m_extent, n_extent, k_extent, selected_a, selected_b,
          tensor::input<Acc>(correction_tensor), c_output);
    } else if constexpr (uses_c_prologue_v<OutPipeline>) {
      return make_benchmark_matmul_invocation(ops::MatmulConfig<Atom>{},
          m_extent, n_extent, k_extent, selected_a, selected_b,
          tensor::input<Acc>(c_input_tensor), c_output);
    } else {
      return make_benchmark_matmul_invocation(ops::MatmulConfig<Atom>{},
          m_extent, n_extent, k_extent, selected_a, selected_b, c_output);
    }
  };
  const auto operation = [&] {
    if constexpr (scenario_packs_a_v<Mode> && scenario_packs_b_v<Mode>) {
      return make_operation(packed_a_tensor, packed_b_tensor);
    } else if constexpr (scenario_packs_a_v<Mode>) {
      return make_operation(packed_a_tensor, b_operand);
    } else if constexpr (scenario_packs_b_v<Mode>) {
      return make_operation(a_operand, packed_b_tensor);
    } else {
      return make_operation(a_operand, b_operand);
    }
  }();
  kernel::Workspace storage(operation.required_workspace());
  auto workspace = storage.view();
  ExecutionSession execution{workspace};
  operation(execution);

  const std::array<std::array<nint_t, 2>, 5> samples{{
      {{0, 0}}, {{m - 1, n - 1}}, {{m / 2, n / 2}},
      {{0, n - 1}}, {{m - 1, 0}},
  }};
  for (const auto& sample : samples) {
    const nint_t row = sample[0];
    const nint_t col = sample[1];
    const std::size_t output_index = static_cast<std::size_t>(row * n + col);
    Acc expected{};
    if constexpr (UsesAsymmetricCorrection) {
      expected = asymmetric_correction[static_cast<std::size_t>(
          DualZeroPoint ? output_index : col)];
    } else if constexpr (UsesBias) {
      expected = static_cast<Acc>(c_input[static_cast<std::size_t>(col)]);
    } else if constexpr (uses_c_prologue_v<OutPipeline>) {
      expected = static_cast<Acc>(c_input[output_index]);
    }
    for (nint_t kk = 0; kk < k; ++kk) {
      const TA av = reference_input<
          ::vecops::matmul::Operand::A, TA, MemoryA, InPipeline>(
          a[static_cast<std::size_t>(row * k + kk)], row,
          input_parameters);
      const TB bv = reference_input<
          ::vecops::matmul::Operand::B, TB, MemoryB, InPipelineB>(
          b[static_cast<std::size_t>(col * k + kk)]);
      expected += static_cast<Acc>(av) * static_cast<Acc>(bv);
    }
    const MemoryC reference =
        reference_output<MemoryC, Acc, OutPipeline>(
            expected, col, output_parameters, row);
    if (!scenario_values_equal(reference, c[output_index])) {
      state.SkipWithError("matmul scenario result verification failed");
      return;
    }
  }

  for (auto _ : state) {
    benchmark::DoNotOptimize(a.data());
    benchmark::DoNotOptimize(b.data());
    if constexpr (scenario_online_pack_a_v<Mode>) pack_a_once(execution);
    if constexpr (scenario_online_pack_b_v<Mode>) pack_b_once(execution);
    operation(execution);
    benchmark::DoNotOptimize(c.data());
    benchmark::ClobberMemory();
  }

  const int64_t mnk = static_cast<int64_t>(m) * n * k;
  const int64_t selected_a_bytes = scenario_packs_a_v<Mode>
      ? static_cast<int64_t>(packed_a_bytes)
      : static_cast<int64_t>(m) * k * sizeof(MemoryA);
  const int64_t selected_b_bytes = scenario_packs_b_v<Mode>
      ? static_cast<int64_t>(packed_b_bytes)
      : static_cast<int64_t>(n) * k * sizeof(MemoryB);
  const int64_t input_parameter_bytes =
      uses_runtime_per_row_quantization_v<InPipeline>
      ? static_cast<int64_t>(m) * sizeof(float32_t) + sizeof(int32_t)
      : 0;
  const int64_t epilogue_parameter_bytes =
      OutPipeline == OutputPipeline::RuntimePerRowColumnDequantize
      ? static_cast<int64_t>(m + n) * sizeof(float32_t)
      : uses_dynamic_scale_v<OutPipeline>
          ? static_cast<int64_t>(sizeof(Acc))
          : uses_per_column_scale_v<OutPipeline>
              ? static_cast<int64_t>(n) *
                    (uses_quant_column_scale_v<OutPipeline>
                         ? sizeof(float32_t)
                         : sizeof(Acc)) +
                    (uses_runtime_output_zero_point_v<OutPipeline>
                         ? sizeof(int32_t)
                         : 0)
              : 0;
  int64_t bytes =
      selected_a_bytes + selected_b_bytes +
      static_cast<int64_t>(m) * n * sizeof(MemoryC) +
      (UsesAsymmetricCorrection
           ? static_cast<int64_t>(asymmetric_correction.size()) * sizeof(Acc)
           : UsesBias
               ? static_cast<int64_t>(n) * sizeof(MemoryCInput)
               : uses_c_prologue_v<OutPipeline>
                   ? static_cast<int64_t>(m) * n * sizeof(MemoryCInput)
                   : 0) +
      input_parameter_bytes + epilogue_parameter_bytes;
  if constexpr (scenario_online_pack_a_v<Mode>) {
    bytes += static_cast<int64_t>(m) * k * sizeof(MemoryA) +
        packed_a_bytes;
  }
  if constexpr (scenario_online_pack_b_v<Mode>) {
    bytes += static_cast<int64_t>(n) * k * sizeof(MemoryB) +
        packed_b_bytes +
        (timed_asymmetric_compensation_v<InPipeline>
             ? static_cast<int64_t>(n) * sizeof(Acc) +
                   (uses_runtime_per_row_quantization_v<InPipeline>
                        ? sizeof(int32_t)
                        : 0)
             : 0);
  }
  state.SetItemsProcessed(state.iterations() * mnk);
  state.SetBytesProcessed(state.iterations() * bytes);
  state.counters["FLOP/s"] = benchmark::Counter(
      static_cast<double>(2 * mnk),
      benchmark::Counter::kIsIterationInvariantRate);
  state.counters["workspace_bytes"] =
      benchmark::Counter(double(operation.required_workspace()));
  state.counters["packed_elements"] = benchmark::Counter(double(
      (scenario_packs_a_v<Mode> ? packed_a_elements : 0) +
      (scenario_packs_b_v<Mode> ? packed_b_elements : 0)));
  state.counters["packing_included"] = benchmark::Counter(double(
      scenario_online_pack_a_v<Mode> || scenario_online_pack_b_v<Mode>));
  state.counters["compensation_included"] = benchmark::Counter(double(
      scenario_online_pack_b_v<Mode> &&
      timed_asymmetric_compensation_v<InPipeline>));
  state.counters["input_parameter_bytes"] =
      benchmark::Counter(double(input_parameter_bytes));
  state.counters["runtime_input_zero_point"] = benchmark::Counter(double(
      uses_runtime_per_row_quantization_v<InPipeline>
          ? input_zero_point
          : 0));
  state.counters["epilogue_parameter_bytes"] =
      benchmark::Counter(double(epilogue_parameter_bytes));
}

template <typename Atom, typename MemoryA, typename MemoryB, typename MemoryC,
          InputPipeline InPipeline = InputPipeline::Convert,
          OutputPipeline OutPipeline = OutputPipeline::Convert,
          InputPipeline InPipelineB = InPipeline,
          typename MemoryCInput = MemoryC>
void register_scenario(const std::vector<MatmulCase>& cases) {
  for (const auto& test_case : cases) {
    using TA = typename Atom::TA;
    using TB = typename Atom::TB;
    using Acc = typename Atom::TAcc;
    const std::string name =
        "MatmulScenario/" + std::string(test_case.name) +
        "/shape:" + std::to_string(test_case.m) + "x" +
            std::to_string(test_case.n) + "x" +
            std::to_string(test_case.k) +
        "/memory:" + dtype_name<MemoryA>() + "x" +
            dtype_name<MemoryB>() + "_to_" + dtype_name<MemoryC>() +
        "/compute:" + dtype_name<TA>() + "x" + dtype_name<TB>() +
            "_acc_" + dtype_name<Acc>() +
        input_pipeline_label<InPipeline, InPipelineB>() +
        c_input_type_label<MemoryCInput, MemoryC>() +
        "/output_pipeline:" + output_pipeline_name<OutPipeline>() +
        "/atom:" + atom_name<Atom>() +
        "/extent:Dynamic" +
        "/arch:" + VECOPS_BENCH_ARCH_CODE;
    auto* registered = benchmark::RegisterBenchmark(
        name.c_str(),
        [test_case](benchmark::State& state) {
          run_scenario<
              Atom, MemoryA, MemoryB, MemoryC,
              InPipeline, OutPipeline, InputMode::Raw,
              InPipelineB, MemoryCInput>(state, test_case);
        });
    registered->Unit(benchmark::kMicrosecond);
    ::vecops::bench::configure_registered_benchmark(
        registered, 0.02, 3)->ReportAggregatesOnly(true);
  }
}

template <typename Atom, typename MemoryA, typename MemoryB, typename MemoryC,
          InputPipeline InPipeline = InputPipeline::Convert,
          OutputPipeline OutPipeline = OutputPipeline::Convert,
          InputPipeline InPipelineB = InPipeline,
          typename MemoryCInput = MemoryC,
          nint_t KStep>
void register_scenario(ScenarioCaseCatalog<KStep>) {
  register_scenario_extent_catalog<
      Atom, MemoryA, MemoryB, MemoryC, KStep, InPipeline, OutPipeline,
      InputMode::Raw, InPipelineB, MemoryCInput>();
}

template <typename Atom, typename MemoryA, typename MemoryB, typename MemoryC,
          InputPipeline InPipeline = InputPipeline::Convert,
          OutputPipeline OutPipeline = OutputPipeline::Convert,
          InputPipeline InPipelineB = InPipeline,
          typename MemoryCInput = MemoryC,
          nint_t KStep>
void register_scenario(FusionScenarioCaseCatalog<KStep>) {
#define VECOPS_REGISTER_FUSION_PAIR(NAME, M, N, K, CASE_INDEX) \
  register_scenario_extent_pair< \
      Atom, MemoryA, MemoryB, MemoryC, InPipeline, OutPipeline, \
      InputMode::Raw, InPipelineB, MemoryCInput, M, N, K, CASE_INDEX>( \
          "MatmulFusion", NAME)
  VECOPS_REGISTER_FUSION_PAIR("tail", 19, 23, 8 * KStep + 1, 0);
  VECOPS_REGISTER_FUSION_PAIR("tiny_area", 2, 4, 2 * KStep + 1, 1);
  VECOPS_REGISTER_FUSION_PAIR("skinny_long_k_tail", 1, 8, 4097, 2);
  VECOPS_REGISTER_FUSION_PAIR("skinny_long_k_tail_col", 8, 1, 4097, 3);
  VECOPS_REGISTER_FUSION_PAIR("gemv_tail", 1, 257, 4 * KStep + 1, 4);
  VECOPS_REGISTER_FUSION_PAIR("small", 8, 128, 128, 5);
  VECOPS_REGISTER_FUSION_PAIR("medium", 64, 256, 256, 6);
  VECOPS_REGISTER_FUSION_PAIR("wide", 32, 1024, 256, 7);
#undef VECOPS_REGISTER_FUSION_PAIR
}

template <typename Atom, typename MemoryA, typename MemoryB, typename MemoryC,
          InputPipeline InPipeline, OutputPipeline OutPipeline,
          InputMode Mode,
          InputPipeline InPipelineB = InPipeline,
          typename MemoryCInput = MemoryC>
void register_packed_scenario(const std::vector<MatmulCase>& cases) {
  using TA = typename Atom::TA;
  using TB = typename Atom::TB;
  using Acc = typename Atom::TAcc;
  for (const auto& test_case : cases) {
    const std::string name =
        "MatmulPackedScenario/" + std::string(test_case.name) +
        "/shape:" + std::to_string(test_case.m) + "x" +
            std::to_string(test_case.n) + "x" +
            std::to_string(test_case.k) +
        "/input_mode:" + input_mode_name<Mode>() +
        "/memory:" + dtype_name<MemoryA>() + "x" +
            dtype_name<MemoryB>() + "_to_" + dtype_name<MemoryC>() +
        "/compute:" + dtype_name<TA>() + "x" + dtype_name<TB>() +
            "_acc_" + dtype_name<Acc>() +
        input_pipeline_label<InPipeline, InPipelineB>() +
        c_input_type_label<MemoryCInput, MemoryC>() +
        "/output_pipeline:" + output_pipeline_name<OutPipeline>() +
        "/atom:" + atom_name<Atom>() +
        "/extent:Dynamic" +
        "/arch:" + VECOPS_BENCH_ARCH_CODE;
    auto* registered = benchmark::RegisterBenchmark(
        name.c_str(), [test_case](benchmark::State& state) {
          run_scenario<
              Atom, MemoryA, MemoryB, MemoryC,
              InPipeline, OutPipeline, Mode,
              InPipelineB, MemoryCInput>(state, test_case);
        });
    registered->Unit(benchmark::kMicrosecond);
    ::vecops::bench::configure_registered_benchmark(
        registered, 0.02, 3)->ReportAggregatesOnly(true);
  }
}

template <typename Atom, typename MemoryA, typename MemoryB, typename MemoryC,
          InputPipeline InPipeline, OutputPipeline OutPipeline,
          InputMode Mode, InputPipeline InPipelineB = InPipeline,
          typename MemoryCInput = MemoryC, int CaseBase = 0,
          nint_t KStep>
void register_packed_scenario(ScenarioCaseCatalog<KStep>) {
  register_scenario_extent_catalog<
      Atom, MemoryA, MemoryB, MemoryC, KStep, InPipeline, OutPipeline, Mode,
      InPipelineB, MemoryCInput, CaseBase>("MatmulPacked");
}

template <typename Atom, typename MemoryA, typename MemoryB, typename MemoryC,
          InputPipeline InPipeline = InputPipeline::Convert,
          OutputPipeline OutPipeline = OutputPipeline::Convert,
          InputPipeline InPipelineB = InPipeline,
          typename MemoryCInput = MemoryC>
void register_all_packing_modes(const std::vector<MatmulCase>& cases) {
  register_packed_scenario<
      Atom, MemoryA, MemoryB, MemoryC, InPipeline, OutPipeline,
      InputMode::Raw, InPipelineB, MemoryCInput>(cases);
  register_packed_scenario<
      Atom, MemoryA, MemoryB, MemoryC, InPipeline, OutPipeline,
      InputMode::PackedA, InPipelineB, MemoryCInput>(cases);
  register_packed_scenario<
      Atom, MemoryA, MemoryB, MemoryC, InPipeline, OutPipeline,
      InputMode::PackedB, InPipelineB, MemoryCInput>(cases);
  register_packed_scenario<
      Atom, MemoryA, MemoryB, MemoryC, InPipeline, OutPipeline,
      InputMode::PackedAB, InPipelineB, MemoryCInput>(cases);
  register_packed_scenario<
      Atom, MemoryA, MemoryB, MemoryC, InPipeline, OutPipeline,
      InputMode::OnlinePackedA, InPipelineB, MemoryCInput>(cases);
  register_packed_scenario<
      Atom, MemoryA, MemoryB, MemoryC, InPipeline, OutPipeline,
      InputMode::OnlinePackedB, InPipelineB, MemoryCInput>(cases);
  register_packed_scenario<
      Atom, MemoryA, MemoryB, MemoryC, InPipeline, OutPipeline,
      InputMode::OnlinePackedAB, InPipelineB, MemoryCInput>(cases);
}

template <typename Atom, typename MemoryA, typename MemoryB, typename MemoryC,
          InputPipeline InPipeline = InputPipeline::Convert,
          OutputPipeline OutPipeline = OutputPipeline::Convert,
          InputPipeline InPipelineB = InPipeline,
          typename MemoryCInput = MemoryC, nint_t KStep>
void register_all_packing_modes(ScenarioCaseCatalog<KStep> cases) {
  register_packed_scenario<
      Atom, MemoryA, MemoryB, MemoryC, InPipeline, OutPipeline,
      InputMode::Raw, InPipelineB, MemoryCInput, 0>(cases);
  register_packed_scenario<
      Atom, MemoryA, MemoryB, MemoryC, InPipeline, OutPipeline,
      InputMode::PackedA, InPipelineB, MemoryCInput, 4>(cases);
  register_packed_scenario<
      Atom, MemoryA, MemoryB, MemoryC, InPipeline, OutPipeline,
      InputMode::PackedB, InPipelineB, MemoryCInput, 8>(cases);
  register_packed_scenario<
      Atom, MemoryA, MemoryB, MemoryC, InPipeline, OutPipeline,
      InputMode::PackedAB, InPipelineB, MemoryCInput, 12>(cases);
  register_packed_scenario<
      Atom, MemoryA, MemoryB, MemoryC, InPipeline, OutPipeline,
      InputMode::OnlinePackedA, InPipelineB, MemoryCInput, 16>(cases);
  register_packed_scenario<
      Atom, MemoryA, MemoryB, MemoryC, InPipeline, OutPipeline,
      InputMode::OnlinePackedB, InPipelineB, MemoryCInput, 20>(cases);
  register_packed_scenario<
      Atom, MemoryA, MemoryB, MemoryC, InPipeline, OutPipeline,
      InputMode::OnlinePackedAB, InPipelineB, MemoryCInput, 24>(cases);
}

} // namespace vecops::bench::matmul

#endif // VECOPS_BENCHMARKS_OPS_MATMUL_SCENARIO_BENCH_COMMON_H
