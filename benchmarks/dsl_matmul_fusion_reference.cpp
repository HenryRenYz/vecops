// SPDX-FileCopyrightText: 2026 HenryRenYz and vecops contributors
// SPDX-License-Identifier: MIT
// Paired native baseline for MatmulFusionBench and MatmulOrientationBench.
// Pipeline semantics match MatmulScenarioBenchCommon.h. Unlike a rewritten
// compute loop, these DataAccess operands preserve public Matmul's transform,
// packing, orientation, runtime-quantization and skinny-family fast paths.
#include "vecops/Kernel.h"
#include "vecops/ops/Matmul.h"
#include "vecops/matmul/Quantization.h"
#include "vecops/kernel/Workspace.h"
#include "vecops/execution/ExecutionSession.h"
#include "dsl_matmul_atom.h"

namespace fusion_benchmark {
enum class Input {
  Convert,
  Quantize4,
  AsymmetricQuantize4Zp3,
  AsymmetricQuantize4Zp3Zp5,
  PrequantizedAsymmetricZp3,
  RuntimePerRowAsymmetricQuantize
};
enum class Output {
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
  AsymmetricDequantize
};

template <bool Broadcast, typename Tensor, typename M, typename N>
VECOPS_ALWAYS_INLINE auto prior(const Tensor& tensor, M m, N n) {
  if constexpr (Broadcast) {
    if constexpr (Tensor::Ndim == 3)
      return vecops::tensor::make_tensor(
        tensor.data(), vecops::tensor::make_layout(vecops::tensor::make_shape(vecops::tensor::size<0>(tensor), m, n),
                                                   vecops::tensor::make_strides(vecops::tensor::stride<0>(tensor),
                                                                                vecops::meta::cint<0>,
                                                                                vecops::tensor::stride<2>(tensor))));
    else
      return vecops::tensor::make_tensor(
        tensor.data(), vecops::tensor::make_layout(
                         vecops::tensor::make_shape(m, n),
                         vecops::tensor::make_strides(vecops::meta::cint<0>, vecops::tensor::stride<1>(tensor))));
  } else
    return tensor;
}

template <typename Tensor>
VECOPS_ALWAYS_INLINE vecops::nint_t row_batch_stride(const Tensor& tensor) {
  if constexpr (Tensor::Ndim == 3)
    return vecops::tensor::stride<0>(tensor) == 0 ? 0 : vecops::nint_t(vecops::tensor::size<1>(tensor));
  else
    return 0;
}

template <bool SideA, typename Native, Input Pipeline, typename Tensor, typename Row, typename Zero>
auto input(const Tensor& tensor, const Row& row, const Zero& zero, vecops::nint_t row_stride) {
  if constexpr (Pipeline == Input::RuntimePerRowAsymmetricQuantize) {
    static_assert(SideA);
    auto transform =
      vecops::matmul::make_runtime_per_row_asymmetric_quantize_transform({row.data(), zero.data(), row_stride});
    return vecops::tensor::input<Native>(tensor, transform);
  } else if constexpr (Pipeline == Input::Quantize4 || Pipeline == Input::AsymmetricQuantize4Zp3 ||
                       Pipeline == Input::AsymmetricQuantize4Zp3Zp5) {
    constexpr float zp = Pipeline == Input::AsymmetricQuantize4Zp3Zp5 ? (SideA ? 3.0f : 5.0f)
                         : Pipeline == Input::AsymmetricQuantize4Zp3 && SideA && std::is_unsigned_v<Native> ? 3.0f
                                                                                                            : 0.0f;
    auto transform =
      vecops::tensor::make_elementwise_vec_transform<float, float>([](auto tag, auto value) VECOPS_KERNEL_LAMBDA {
        auto result = vecops::vec::mul(tag, value, vecops::vec::fill(tag, 4.0f));
        if constexpr (zp != 0)
          result = vecops::vec::add(tag, result, vecops::vec::fill(tag, zp));
        return result;
      });
    return vecops::tensor::input<Native>(tensor, transform);
  } else {
    return vecops::tensor::input<Native>(tensor);
  }
}

template <typename Acc, Output Pipeline, typename Tensor, typename Scale, typename Column, typename QuantColumn,
          typename Row, typename Weight, typename Zero>
auto output(const Tensor& tensor, const Scale& scale, const Column& column, const QuantColumn& quant_column,
            const Row& row, const Weight& weight, const Zero& zero, vecops::nint_t row_stride) {
  if constexpr (Pipeline == Output::Convert || Pipeline == Output::Accumulate || Pipeline == Output::Bias) {
    return vecops::tensor::output<Acc>(tensor);
  } else if constexpr (Pipeline == Output::RuntimePerRowColumnDequantize) {
    auto transform =
      vecops::matmul::make_runtime_per_row_column_dequantize_transform({row.data(), weight.data(), row_stride});
    return vecops::tensor::output<Acc>(tensor, transform);
  } else if constexpr (Pipeline == Output::PerColumnScale) {
    auto transform = vecops::tensor::make_lane_local_vec_transform<Acc, Acc>(
      [scales = column.data()](auto tag, auto value, const auto& context) VECOPS_KERNEL_LAMBDA {
        const auto coordinate = context.lane_coord(0);
        const auto index = coordinate[coordinate.size() - 1];
        // Col-GEMV and transposed/swap stores may vectorize original rows.
        // Auxiliary column values then broadcast rather than advance per lane.
        const auto scale = context.vector_axis == static_cast<int>(coordinate.size()) - 1
                             ? vecops::vec::load(tag, scales + index)
                             : vecops::vec::fill(tag, scales[index]);
        return vecops::vec::mul(tag, value, scale);
      });
    return vecops::tensor::output<Acc>(tensor, transform);
  } else if constexpr (Pipeline == Output::BiasPerColumnDequantize || Pipeline == Output::BiasPerColumnRequantizeU8) {
    auto transform = vecops::tensor::make_lane_local_vec_transform<float, float>(
      [scales = quant_column.data(), zp = zero.data()](auto tag, auto value, const auto& context) VECOPS_KERNEL_LAMBDA {
        const auto coordinate = context.lane_coord(0);
        const auto index = coordinate[coordinate.size() - 1];
        const auto scale = context.vector_axis == static_cast<int>(coordinate.size()) - 1
                             ? vecops::vec::load(tag, scales + index)
                             : vecops::vec::fill(tag, scales[index]);
        auto result = vecops::vec::mul(tag, value, scale);
        if constexpr (Pipeline == Output::BiasPerColumnRequantizeU8)
          result = vecops::vec::add(tag, result, vecops::vec::fill(tag, static_cast<float>(*zp)));
        return result;
      });
    return vecops::tensor::output<Acc>(tensor, transform);
  } else {
    constexpr bool quantized = Pipeline == Output::Dequantize || Pipeline == Output::ReluRequantize ||
                               Pipeline == Output::RequantizeU8Zp7 || Pipeline == Output::AsymmetricDequantize;
    using Compute = std::conditional_t<quantized, float, Acc>;
    auto transform = vecops::tensor::make_elementwise_vec_transform<Compute, Compute>(
      [runtime_scale = scale.data()](auto tag, auto value) VECOPS_KERNEL_LAMBDA {
        if constexpr (Pipeline == Output::Relu || Pipeline == Output::BiasRelu ||
                      Pipeline == Output::AccumulateReluScale || Pipeline == Output::ReluRequantize)
          value = vecops::vec::max(tag, value, vecops::vec::zeros(tag));
        if constexpr (Pipeline == Output::Sigmoid || Pipeline == Output::Silu || Pipeline == Output::BiasSilu) {
          const auto one = vecops::vec::fill(tag, Compute{1});
          const auto sigmoid =
            vecops::vec::div(tag, one, vecops::vec::add(tag, one, vecops::vec::exp(tag, vecops::vec::neg(tag, value))));
          if constexpr (Pipeline == Output::Sigmoid)
            return sigmoid;
          else
            return vecops::vec::mul(tag, value, sigmoid);
        } else if constexpr (Pipeline == Output::Scale || Pipeline == Output::AccumulateReluScale) {
          return vecops::vec::mul(tag, value, vecops::vec::fill(tag, static_cast<Compute>(0.5)));
        } else if constexpr (Pipeline == Output::DynamicScale) {
          return vecops::vec::mul(tag, value, vecops::vec::fill(tag, *runtime_scale));
        } else if constexpr (Pipeline == Output::Clamp || Pipeline == Output::BiasClamp) {
          return vecops::vec::min(tag,
                                  vecops::vec::max(tag, value, vecops::vec::fill(tag, static_cast<Compute>(-0.25))),
                                  vecops::vec::fill(tag, static_cast<Compute>(0.25)));
        } else if constexpr (Pipeline == Output::Dequantize || Pipeline == Output::ReluRequantize ||
                             Pipeline == Output::RequantizeU8Zp7) {
          auto result = vecops::vec::mul(tag, value, vecops::vec::fill(tag, 0.125f));
          if constexpr (Pipeline == Output::RequantizeU8Zp7)
            result = vecops::vec::add(tag, result, vecops::vec::fill(tag, 7.0f));
          return result;
        } else if constexpr (Pipeline == Output::AsymmetricDequantize) {
          return vecops::vec::mul(tag, value, vecops::vec::fill(tag, 0.0625f));
        } else {
          return value;
        }
      });
    return vecops::tensor::output<Acc>(tensor, transform);
  }
}
} // namespace fusion_benchmark

void __kernel__(vecops::tensor::TensorLike auto A, vecops::tensor::TensorLike auto B, vecops::tensor::TensorLike auto C,
                vecops::tensor::TensorLike auto D, vecops::tensor::TensorLike auto S,
                vecops::tensor::TensorLike auto SC, vecops::tensor::TensorLike auto SQ,
                vecops::tensor::TensorLike auto RQ, vecops::tensor::TensorLike auto RD,
                vecops::tensor::TensorLike auto W, vecops::tensor::TensorLike auto Z, std::int64_t repeat) {
  using namespace vecops;
  using Atom = BenchmarkAtom<spec::NativeA, spec::NativeB>;
  using Acc = typename Atom::TAcc;
  constexpr auto input_a = static_cast<fusion_benchmark::Input>(decltype(spec::InputPipelineA)::value);
  constexpr auto input_b = static_cast<fusion_benchmark::Input>(decltype(spec::InputPipelineB)::value);
  constexpr auto output = static_cast<fusion_benchmark::Output>(decltype(spec::OutputPipeline)::value);
  constexpr int rank = decltype(A)::Ndim;
  const auto row_stride = fusion_benchmark::row_batch_stride(A);
  auto a = fusion_benchmark::input<true, typename Atom::TA, input_a>(A, RQ, Z, row_stride);
  auto b = fusion_benchmark::input<false, typename Atom::TB, input_b>(tensor::transpose<rank - 2, rank - 1>(B), RQ, Z,
                                                                      row_stride);
  auto out = fusion_benchmark::output<Acc, output>(C, S, SC, SQ, RD, W, Z, row_stride);
  using Config = ops::MatmulConfig<Atom, matmul::family_selection::Automatic, kernel::matmul_policy::Automatic,
                                   matmul::GenericTiledTuning<>, platform::SystemCacheInfoProvider, spec::SwapAB>;
  auto operation = ops::matmul(Config{});
  const auto m = tensor::size<rank - 2>(A);
  const auto n = tensor::size<rank - 1>(B);
  const auto k = tensor::size<rank - 1>(A);
  if constexpr (spec::HasPrior) {
    auto prior_tensor = fusion_benchmark::prior<spec::BroadcastPrior>(D, m, n);
    auto prior = tensor::input<Acc>(prior_tensor);
    kernel::Workspace storage(operation.required_workspace(m, n, k, a, b, prior, out));
    auto workspace = storage.view();
    execution::ExecutionSession execution{workspace};
    for (std::int64_t iteration = 0; iteration < repeat; ++iteration) {
      operation(execution, m, n, k, a, b, prior, out);
      asm volatile("" ::: "memory");
    }
  } else {
    kernel::Workspace storage(operation.required_workspace(m, n, k, a, b, out));
    auto workspace = storage.view();
    execution::ExecutionSession execution{workspace};
    for (std::int64_t iteration = 0; iteration < repeat; ++iteration) {
      operation(execution, m, n, k, a, b, out);
      asm volatile("" ::: "memory");
    }
  }
}
