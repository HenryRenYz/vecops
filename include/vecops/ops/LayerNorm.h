#ifndef VECOPS_OPS_LAYERNORM_H
#define VECOPS_OPS_LAYERNORM_H

#include <algorithm>
#include <atomic>
#include <cmath>
#include <type_traits>

#include "vecops/Assertion.h"
#include "vecops/CoreTypes.h"
#include "vecops/kernel/Loop.h"
#include "vecops/kernel/Workspace.h"
#include "vecops/tensor/DataAccess.h"
#include "vecops/vec/Vec.h"

namespace vecops::ops {

template <typename ComputeT = float32_t,
          typename VecTag = vec::ScalableTag<ComputeT, 0>>
struct LayerNormConfig {
  using ComputeType = ComputeT;
  using Tag = VecTag;
  ComputeT eps = ComputeT(1e-5f);
};

namespace details {

template <typename InLayout, typename ScaleLayout, typename BiasLayout,
          typename OutLayout>
VECOPS_INLINE void validate_layernorm_layouts(
    const InLayout& in, const ScaleLayout& scale, const BiasLayout& bias,
    const OutLayout& out) {
  static_assert(InLayout::Ndim >= 1);
  static_assert(OutLayout::Ndim == InLayout::Ndim);
  static_assert(ScaleLayout::Ndim == 1);
  static_assert(BiasLayout::Ndim == 1);
  for (int d = 0; d < InLayout::Ndim; ++d) {
    VECOPS_ASSERT(in.shape()[d] == out.shape()[d],
                  "LayerNorm input/output shape mismatch");
  }
  const nint_t n = in.shape()[InLayout::Ndim - 1];
  VECOPS_ASSERT(n > 0, "LayerNorm normalized dimension must be non-empty");
  VECOPS_ASSERT(scale.shape()[0] == n && bias.shape()[0] == n,
                "LayerNorm scale/bias size mismatch");
}

template <typename Compute, typename Tensor, typename Transform,
          typename Projection, typename... Facts>
VECOPS_INLINE auto remove_first_dimension(
    const tensor::InputSpec<
        Compute, Tensor, Transform, Projection, Facts...>& spec) {
  auto sliced_tensor = tensor::make_tensor(
      spec.tensor().data(), tensor::remove<0>(spec.input_layout()));
  auto projection = spec.projection().template sliced<0>(0);
  return tensor::InputSpec<Compute, decltype(sliced_tensor), Transform,
                           decltype(projection)>{
      sliced_tensor, spec.transform(), projection};
}

template <typename Compute, typename Tensor, typename Transform,
          typename Projection, typename... Facts>
VECOPS_INLINE auto remove_first_dimension(
    const tensor::OutputSpec<
        Compute, Tensor, Transform, Projection, Facts...>& spec) {
  auto sliced_tensor = tensor::make_tensor(
      spec.tensor().data(), tensor::remove<0>(spec.output_layout()));
  auto projection = spec.projection().template sliced<0>(0);
  return tensor::OutputSpec<Compute, decltype(sliced_tensor), Transform,
                            decltype(projection)>{
      sliced_tensor, spec.transform(), projection};
}

template <int Count, typename Spec>
VECOPS_INLINE auto row_spec(const Spec& spec) {
  if constexpr (Count == 0) return spec;
  else return row_spec<Count - 1>(remove_first_dimension(spec));
}

} // namespace details

template <typename Config = LayerNormConfig<>>
class LayerNorm {
public:
  using ComputeType = typename Config::ComputeType;
  using Tag = typename Config::Tag;

  const Config config;

  VECOPS_INLINE constexpr explicit LayerNorm(Config cfg = {}) : config(cfg) {}

  template <typename InSpec, typename ScaleSpec, typename BiasSpec,
            typename OutSpec>
  VECOPS_INLINE nint_t required_workspace(
      const InSpec& in, const ScaleSpec& scale, const BiasSpec& bias,
      const OutSpec& out) const {
    static_assert(tensor::is_input_spec_v<InSpec>);
    static_assert(tensor::is_input_spec_v<ScaleSpec>);
    static_assert(tensor::is_input_spec_v<BiasSpec>);
    static_assert(tensor::is_output_spec_v<OutSpec>);
    details::validate_layernorm_layouts(
        in.input_layout(), scale.input_layout(), bias.input_layout(),
        out.output_layout());
    constexpr int PrefixRank = InSpec::InputTensor::Ndim - 1;
    const auto in_row = details::row_spec<PrefixRank>(in);
    const auto out_row = details::row_spec<PrefixRank>(out);
    using InPolicy = tensor::InputAccessPolicy<0, 2>;
    using ParamPolicy = tensor::InputAccessPolicy<0, 1>;
    using OutPolicy = tensor::OutputAccessPolicy<0>;
    return kernel::details::workspace_round_up(
        tensor::required_workspace(in_row, InPolicy{}) +
            tensor::required_workspace(scale, ParamPolicy{}) +
            tensor::required_workspace(bias, ParamPolicy{}) +
            tensor::required_workspace(out_row, OutPolicy{}),
        vec::DEFAULT_ALIGNMENT);
  }

  template <typename InSpec, typename ScaleSpec, typename BiasSpec,
            typename OutSpec>
  VECOPS_INLINE void operator()(
      kernel::WorkspaceView& workspace, const InSpec& in,
      const ScaleSpec& scale, const BiasSpec& bias,
      const OutSpec& out) const {
    static_assert(tensor::is_input_spec_v<InSpec>);
    static_assert(tensor::is_input_spec_v<ScaleSpec>);
    static_assert(tensor::is_input_spec_v<BiasSpec>);
    static_assert(tensor::is_output_spec_v<OutSpec>);
    details::validate_layernorm_layouts(
        in.input_layout(), scale.input_layout(), bias.input_layout(),
        out.output_layout());
    constexpr int PrefixRank = InSpec::InputTensor::Ndim - 1;

#if defined(CPU_CAPABILITY_SVE) && defined(__ARM_FEATURE_SVE2) && \
    !defined(VECOPS_DISABLE_SVE_16BIT_LAYERNORM_FAST_PATH)
    using InElement = std::remove_const_t<typename InSpec::MemoryElement>;
    using ScaleElement =
        std::remove_const_t<typename ScaleSpec::MemoryElement>;
    using BiasElement =
        std::remove_const_t<typename BiasSpec::MemoryElement>;
    using OutElement = typename OutSpec::MemoryElement;
#if defined(__ARM_FEATURE_SVE_BF16)
    constexpr bool SupportedElement =
        IsFloat16V<InElement> || IsBfloat16V<InElement>;
#else
    constexpr bool SupportedElement = IsFloat16V<InElement>;
#endif
    constexpr bool UseSVE16 =
        std::same_as<ComputeType, float32_t> &&
        std::same_as<Tag, vec::ScalableTag<float32_t, 0>> &&
        SupportedElement && std::same_as<ScaleElement, InElement> &&
        std::same_as<BiasElement, InElement> &&
        std::same_as<OutElement, InElement> &&
        std::same_as<typename InSpec::TransformType, tensor::NoTransform> &&
        std::same_as<typename ScaleSpec::TransformType, tensor::NoTransform> &&
        std::same_as<typename BiasSpec::TransformType, tensor::NoTransform> &&
        std::same_as<typename OutSpec::TransformType, tensor::NoTransform> &&
        tensor::is_ct_last_contiguous<typename InSpec::InputLayout, 1>::value &&
        tensor::is_ct_last_contiguous<
            typename ScaleSpec::InputLayout, 1>::value &&
        tensor::is_ct_last_contiguous<
            typename BiasSpec::InputLayout, 1>::value &&
        tensor::is_ct_last_contiguous<
            typename OutSpec::OutputLayout, 1>::value;
    if constexpr (UseSVE16) {
      kernel::loop::for_each_dims<PrefixRank>(
          [this, &workspace, &scale, &bias](
              const auto& in_row, const auto& out_row) {
            run_row_sve_16bit(
                workspace, in_row, scale, bias, out_row);
          },
          in, out);
      return;
    }
#endif

    // Parameters are shared by every row. Bind them once outside the row loop
    // so short/decode kernels do not repeatedly construct equivalent sessions.
    // They are single-pass rank-1 inputs, therefore direct is also the resolved
    // automatic plan for every layout.
    using ParamPolicy = tensor::InputAccessPolicy<
        0, 1, tensor::AccessPlan::direct>;
    kernel::with_operands(
        workspace, tensor::operand(scale, ParamPolicy{}),
        tensor::operand(bias, ParamPolicy{}),
        [this, &workspace, &in, &out](auto& gamma, auto& beta) {
          // Resolve the only plan that can differ for a row before entering
          // the row loop. A rank-1 output has no alternate unit-stride commit
          // axis, so its automatic plan is always direct.
          if (in.input_layout().strides()[PrefixRank] == 1) {
            kernel::loop::for_each_dims<PrefixRank>(
                [this, &workspace, &gamma, &beta](const auto& in_row,
                                                  const auto& out_row) {
                  run_row<tensor::AccessPlan::direct>(
                      workspace, in_row, gamma, beta, out_row);
                },
                in, out);
          } else {
            kernel::loop::for_each_dims<PrefixRank>(
                [this, &workspace, &gamma, &beta](const auto& in_row,
                                                  const auto& out_row) {
                  run_row<tensor::AccessPlan::automatic>(
                      workspace, in_row, gamma, beta, out_row);
                },
                in, out);
          }
        });
  }

  template <typename InSpec, typename ScaleSpec, typename BiasSpec,
            typename OutSpec>
  VECOPS_INLINE void operator()(
      const InSpec& in, const ScaleSpec& scale, const BiasSpec& bias,
      const OutSpec& out) const {
    kernel::Workspace storage(required_workspace(in, scale, bias, out));
    auto workspace = storage.view();
    (*this)(workspace, in, scale, bias, out);
  }

private:
#if defined(CPU_CAPABILITY_SVE) && defined(__ARM_FEATURE_SVE2) && \
    !defined(VECOPS_DISABLE_SVE_16BIT_LAYERNORM_FAST_PATH)
  template <typename InSpec, typename ScaleSpec, typename BiasSpec,
            typename OutSpec>
  VECOPS_INLINE void run_row_sve_16bit(
      kernel::WorkspaceView& workspace, const InSpec& in,
      const ScaleSpec& scale, const BiasSpec& bias,
      const OutSpec& out) const {
    using Element = std::remove_const_t<typename InSpec::MemoryElement>;
    using ConversionOrder = std::conditional_t<
        IsFloat16V<Element>, vec::cvt::Unordered, vec::cvt::Ordered>;
    // Contiguous whole-block prefetch now lowers without a runtime cache-line
    // loop, but it is still slower for this stream on current SVE targets.
    // Keep it disabled until a row-working-set policy shows a measured win.
    using Prefetch = tensor::PrefetchPolicy<
        false, 4, tensor::PrefetchFootprint::whole_block>;
    using XPolicy = tensor::InputAccessPolicy<
        0, 2, tensor::AccessPlan::direct, ConversionOrder,
        vec::cvt::Saturate, tensor::DefaultMemoryPolicy, true, Prefetch>;
    using ParamPolicy = tensor::InputAccessPolicy<
        0, 1, tensor::AccessPlan::direct, ConversionOrder,
        vec::cvt::Saturate, tensor::DefaultMemoryPolicy, true, Prefetch>;
    using YPolicy = tensor::OutputAccessPolicy<
        0, tensor::AccessPlan::direct, ConversionOrder,
        vec::cvt::Saturate, tensor::DefaultMemoryPolicy, true>;
    using F32PairTag = vec::ScalableTag<float32_t, 1>;
    using F32QuadTag = vec::ScalableTag<float32_t, 2>;

    const nint_t n = in.input_layout().shape()[0];
    kernel::with_operands(
        workspace, tensor::operand(in, XPolicy{}),
        tensor::operand(scale, ParamPolicy{}),
        tensor::operand(bias, ParamPolicy{}),
        tensor::operand(out, YPolicy{}),
        [this, n](auto& x, auto& gamma, auto& beta, auto& y) {
          F32QuadTag full_tag{};
          F32PairTag tail_tag{};
          const nint_t full_lanes = vec::size(full_tag);
          const nint_t tail_lanes = vec::size(tail_tag);
          auto sum = vec::zeros(full_tag);
          auto sum_sq = vec::zeros(full_tag);
          float32_t tail_sum = 0.0f;
          float32_t tail_sum_sq = 0.0f;
          nint_t col = 0;
          for (; col + full_lanes <= n; col += full_lanes) {
            if constexpr (Prefetch::enabled) {
              if (col + (Prefetch::ahead_blocks + 1) * full_lanes <= n) {
                x.prefetch(
                    full_tag,
                    tensor::coord(
                        col + Prefetch::ahead_blocks * full_lanes),
                    vec::opt::unmasked);
              }
            }
            auto value = x.load(
                full_tag, tensor::coord(col), vec::opt::unmasked);
            sum = vec::add(sum, value);
            sum_sq = vec::fmadd(value, value, sum_sq);
          }
          while (col < n) {
            const nint_t active = n - col;
//            const nint_t active = std::min(n - col, tail_lanes);
            auto value = x.load(
                tail_tag, tensor::coord(col), vec::opt::first(active));
            tail_sum += vec::reduce_add(tail_tag, value);
            tail_sum_sq +=
                vec::reduce_add(tail_tag, vec::mul(value, value));
            col += tail_lanes;
          }

          const float32_t sum_value =
              vec::reduce_add(full_tag, sum) + tail_sum;
          const float32_t sum_sq_value =
              vec::reduce_add(full_tag, sum_sq) + tail_sum_sq;
          const float32_t inv_n = 1.0f / static_cast<float32_t>(n);
          const float32_t mean = sum_value * inv_n;
          const float32_t variance =
              std::max(sum_sq_value * inv_n - mean * mean, 0.0f);
          const float32_t rstd =
              1.0f / std::sqrt(variance + config.eps);
          const float32_t shift = -mean * rstd;
          const auto full_rstd = vec::fill(full_tag, rstd);
          const auto full_shift = vec::fill(full_tag, shift);
          const auto tail_rstd = vec::fill(tail_tag, rstd);
          const auto tail_shift = vec::fill(tail_tag, shift);

          col = 0;
          for (; col + full_lanes <= n; col += full_lanes) {
            if constexpr (Prefetch::enabled) {
              if (col + (Prefetch::ahead_blocks + 1) * full_lanes <= n) {
                const auto future = tensor::coord(
                    col + Prefetch::ahead_blocks * full_lanes);
                x.prefetch(full_tag, future, vec::opt::unmasked);
                gamma.prefetch(full_tag, future, vec::opt::unmasked);
                beta.prefetch(full_tag, future, vec::opt::unmasked);
              }
            }
            const auto position = tensor::coord(col);
            auto value = x.load(
                full_tag, position, vec::opt::unmasked);
            auto scale_value = gamma.load(
                full_tag, position, vec::opt::unmasked);
            auto bias_value = beta.load(
                full_tag, position, vec::opt::unmasked);
            value = vec::fmadd(value, full_rstd, full_shift);
            y.store(
                full_tag, position,
                vec::fmadd(value, scale_value, bias_value),
                vec::opt::unmasked);
          }
          while (col < n) {
            const nint_t active = n - col;
//            const nint_t active = std::min(n - col, tail_lanes);
            const auto position = tensor::coord(col);
            auto value = x.load(
                tail_tag, position, vec::opt::first(active));
            auto scale_value = gamma.load(
                tail_tag, position, vec::opt::first(active));
            auto bias_value = beta.load(
                tail_tag, position, vec::opt::first(active));
            value = vec::fmadd(value, tail_rstd, tail_shift);
            y.store(
                tail_tag, position,
                vec::fmadd(value, scale_value, bias_value),
                vec::opt::first(active));
            col += tail_lanes;
          }
          y.commit();
        });
  }
#endif

  template <tensor::AccessPlan InPlan, typename InSpec, typename ScaleAccess,
            typename BiasAccess, typename OutSpec>
  VECOPS_INLINE void run_row(
      kernel::WorkspaceView& workspace, const InSpec& in,
      ScaleAccess& gamma, BiasAccess& beta, const OutSpec& out) const {
    static_assert(InSpec::InputTensor::Ndim == 1);
    using InPolicy = tensor::InputAccessPolicy<0, 2, InPlan>;
    using OutPolicy = tensor::OutputAccessPolicy<
        0, tensor::AccessPlan::direct>;
    auto compute = [this, &in](
        auto& x, auto& gamma, auto& beta, auto& y) VECOPS_INLINE_LAMBDA {
          using FullTag = vec::Twice<vec::Twice<Tag>>;
          FullTag full_tag{};
          Tag tag{};
          const nint_t n = in.input_layout().shape()[0];
          const nint_t full_lanes = vec::size(full_tag);
          const nint_t lanes = vec::size(tag);

          nint_t col = 0;
          auto sum = vec::zeros(full_tag);
          auto sum_sq = vec::zeros(full_tag);
          auto tail_sum = vec::zeros(tag);
          auto tail_sum_sq = vec::zeros(tag);
          for (; col + full_lanes <= n; col += full_lanes) {
            auto value = x.load(
                full_tag, tensor::coord(col), vec::opt::unmasked);
            sum = vec::add(sum, value);
            sum_sq = vec::fmadd(value, value, sum_sq);
          }
          while (col < n) {
            const nint_t active = n - col;
            auto value = x.load(
                tag, tensor::coord(col), vec::opt::first(active));
            tail_sum = vec::add(tail_sum, value);
            tail_sum_sq = vec::fmadd(value, value, tail_sum_sq);
            col += lanes;
          }
          using HalfTag = vec::Twice<Tag>;
          HalfTag half_tag{};
          const auto tail_half = vec::concat(
              half_tag, tail_sum, vec::zeros(tag));
          const auto tail_sq_half = vec::concat(
              half_tag, tail_sum_sq, vec::zeros(tag));
          sum = vec::add(
              sum,
              vec::concat(full_tag, tail_half, vec::zeros(half_tag)));
          sum_sq = vec::add(
              sum_sq,
              vec::concat(
                  full_tag, tail_sq_half, vec::zeros(half_tag)));
          const ComputeType sum_value = vec::reduce_add(full_tag, sum);
          const ComputeType sum_sq_value =
              vec::reduce_add(full_tag, sum_sq);
          const ComputeType inv_n =
              ComputeType(1) / static_cast<ComputeType>(n);
          const ComputeType mean = sum_value * inv_n;
          const ComputeType variance =
              std::max(sum_sq_value * inv_n - mean * mean, ComputeType(0));
          const ComputeType rstd =
              ComputeType(1) / std::sqrt(variance + config.eps);

          const auto tail_mean = vec::fill(tag, mean);
          const auto tail_rstd = vec::fill(tag, rstd);
          col = 0;
          // Keep sized multi-word aggregates non-const: GCC's SRA pass
          // disqualifies read-only aggregate declarations after init.
          auto full_mean = vec::fill(full_tag, mean);
          auto full_rstd = vec::fill(full_tag, rstd);
          for (; col + full_lanes <= n; col += full_lanes) {
            write_block(
                x, gamma, beta, y, full_tag, col,
                full_mean, full_rstd, vec::opt::unmasked);
          }
          while (col < n) {
            const nint_t active = n - col;
            write_block(
                x, gamma, beta, y, tag, col,
                tail_mean, tail_rstd, vec::opt::first(active));
            col += lanes;
          }
          y.commit();
        };
    if constexpr (InPlan == tensor::AccessPlan::direct) {
      auto x = tensor::bind(in, InPolicy{}, workspace);
      auto y = tensor::bind(out, OutPolicy{}, workspace);
      compute(x, gamma, beta, y);
    } else {
      kernel::with_operands(
          workspace, tensor::operand(in, InPolicy{}),
          tensor::operand(out, OutPolicy{}),
          [&](auto& x, auto& y) { compute(x, gamma, beta, y); });
    }
  }

  template <typename X, typename Gamma, typename Beta, typename Y,
            vec::VectorTag VTag, typename Active>
  VECOPS_ALWAYS_INLINE static void write_block(
      X& x, Gamma& gamma, Beta& beta, Y& y, VTag tag, nint_t col,
      vec::Vec<VTag> mean, vec::Vec<VTag> rstd, Active active) {
    auto xv = x.load(tag, tensor::coord(col), active);
    auto gamma_v = gamma.load(tag, tensor::coord(col), active);
    auto beta_v = beta.load(tag, tensor::coord(col), active);
    if constexpr (
        vec::num_words(VTag{}) > 1 && requires { sizeof(vec::Vec<VTag>); }) {
      // Preserve the three independent load streams before starting their
      // arithmetic. GCC otherwise shortens aggregate lifetimes word by word,
      // reducing memory-level parallelism in the four-word loop.
      std::atomic_signal_fence(std::memory_order_acquire);
    }
    auto normalized = vec::mul(vec::sub(xv, mean), rstd);
    y.store(tag, tensor::coord(col),
            vec::fmadd(normalized, gamma_v, beta_v), active);
  }
};

template <typename Config = LayerNormConfig<>>
VECOPS_INLINE constexpr auto layer_norm(Config config = {}) {
  return LayerNorm<Config>{config};
}

} // namespace vecops::ops

#endif // VECOPS_OPS_LAYERNORM_H
