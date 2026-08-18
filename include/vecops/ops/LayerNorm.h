#ifndef VECOPS_OPS_LAYERNORM_H
#define VECOPS_OPS_LAYERNORM_H

#include <algorithm>
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
              const auto& in_row, const auto& out_row)
              VECOPS_INLINE_LAMBDA {
            bind_sve_16bit_row(
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
                                                  const auto& out_row)
                    VECOPS_INLINE_LAMBDA {
                  bind_row<tensor::AccessPlan::direct>(
                      workspace, in_row, gamma, beta, out_row);
                },
                in, out);
          } else {
            kernel::loop::for_each_dims<PrefixRank>(
                [this, &workspace, &gamma, &beta](const auto& in_row,
                                                  const auto& out_row)
                    VECOPS_INLINE_LAMBDA {
                  bind_row<tensor::AccessPlan::automatic>(
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
  VECOPS_ALWAYS_INLINE void bind_sve_16bit_row(
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
    using F32Tag = vec::ScalableTag<float32_t, 0>;

    // Every fast-path operand is compile-time direct. Binding each one here
    // avoids routing the row kernel through std::invoke; BiSheng otherwise
    // outlines that wrapper despite the callback's always_inline attribute.
    auto x = tensor::bind(in, XPolicy{}, workspace);
    auto gamma = tensor::bind(scale, ParamPolicy{}, workspace);
    auto beta = tensor::bind(bias, ParamPolicy{}, workspace);
    auto y = tensor::bind(out, YPolicy{}, workspace);
    run_row<
        F32Tag, 4, 2, true,
        kernel::loop::TailCarryPolicy::reuse_prefix, Prefetch>(
        in, x, gamma, beta, y);
  }
#endif

  template <tensor::AccessPlan InPlan, typename InSpec, typename ScaleAccess,
            typename BiasAccess, typename OutSpec>
  VECOPS_ALWAYS_INLINE void bind_row(
      kernel::WorkspaceView& workspace, const InSpec& in,
      ScaleAccess& gamma, BiasAccess& beta, const OutSpec& out) const {
    static_assert(InSpec::InputTensor::Ndim == 1);
    using InPolicy = tensor::InputAccessPolicy<0, 2, InPlan>;
    using OutPolicy = tensor::OutputAccessPolicy<
        0, tensor::AccessPlan::direct>;
    if constexpr (InPlan == tensor::AccessPlan::direct) {
      auto x = tensor::bind(in, InPolicy{}, workspace);
      auto y = tensor::bind(out, OutPolicy{}, workspace);
      run_row<
          Tag, 4, 1, false,
          kernel::loop::TailCarryPolicy::independent, tensor::NoPrefetch>(
          in, x, gamma, beta, y);
    } else {
      kernel::with_operands(
          workspace, tensor::operand(in, InPolicy{}),
          tensor::operand(out, OutPolicy{}),
          [this, &in, &gamma, &beta](auto& x, auto& y)
              VECOPS_INLINE_LAMBDA {
            run_row<
                Tag, 4, 1, false,
                kernel::loop::TailCarryPolicy::independent,
                tensor::NoPrefetch>(in, x, gamma, beta, y);
          });
    }
  }

  template <vec::VectorTag BaseTag, int FullFactor, int TailFactor,
            bool FusedShift, kernel::loop::TailCarryPolicy TailPolicy,
            typename Prefetch,
            typename InSpec,
            typename X, typename Gamma, typename Beta, typename Y>
  VECOPS_ALWAYS_INLINE void run_row(
      const InSpec& in, X& x, Gamma& gamma, Beta& beta, Y& y) const {
    static_assert(InSpec::InputTensor::Ndim == 1);

    BaseTag base_tag{};
    const nint_t n = in.input_layout().shape()[0];

    ComputeType sum_value{};
    ComputeType sum_sq_value{};
    auto accumulate_block = [&, n](
                                auto block_tag, nint_t col, auto active,
                                auto& sum, auto& sum_sq)
        VECOPS_INLINE_LAMBDA {
      using Active = std::remove_cvref_t<decltype(active)>;
      if constexpr (
          Prefetch::enabled && std::same_as<Active, vec::opt::Unmasked>) {
        const nint_t full_lanes = vec::size(block_tag);
        if (col + (Prefetch::ahead_blocks + 1) * full_lanes <= n) {
          x.prefetch(
              block_tag,
              tensor::coord(col + Prefetch::ahead_blocks * full_lanes),
              vec::opt::unmasked);
        }
      }
      auto value = x.load(block_tag, tensor::coord(col), active);
      sum = vec::add(sum, value);
      sum_sq = vec::fmadd(value, value, sum_sq);
    };
    kernel::loop::fold<FullFactor, TailFactor, TailPolicy>(
        base_tag, n, accumulate_block,
        kernel::loop::reduce_add(sum_value),
        kernel::loop::reduce_add(sum_sq_value));

    const ComputeType inv_n =
        ComputeType(1) / static_cast<ComputeType>(n);
    const ComputeType mean = sum_value * inv_n;
    const ComputeType variance =
        std::max(sum_sq_value * inv_n - mean * mean, ComputeType(0));
    const ComputeType rstd =
        ComputeType(1) / std::sqrt(variance + config.eps);
    const ComputeType center = FusedShift ? -mean * rstd : mean;

    auto write_block = [&, n](
                           auto block_tag, nint_t col, auto active,
                           const auto& center_v, const auto& rstd_v)
        VECOPS_INLINE_LAMBDA {
      using Active = std::remove_cvref_t<decltype(active)>;
      if constexpr (
          Prefetch::enabled && std::same_as<Active, vec::opt::Unmasked>) {
        const nint_t full_lanes = vec::size(block_tag);
        if (col + (Prefetch::ahead_blocks + 1) * full_lanes <= n) {
          const auto future = tensor::coord(
              col + Prefetch::ahead_blocks * full_lanes);
          x.prefetch(block_tag, future, vec::opt::unmasked);
          gamma.prefetch(block_tag, future, vec::opt::unmasked);
          beta.prefetch(block_tag, future, vec::opt::unmasked);
        }
      }
      const auto position = tensor::coord(col);
      auto value = x.load(block_tag, position, active);
      auto scale_value = gamma.load(block_tag, position, active);
      auto bias_value = beta.load(block_tag, position, active);
      if constexpr (FusedShift) {
        value = vec::fmadd(value, rstd_v, center_v);
      } else {
        value = vec::mul(vec::sub(value, center_v), rstd_v);
      }
      y.store(
          block_tag, position,
          vec::fmadd(value, scale_value, bias_value), active);
    };
    kernel::loop::fold<FullFactor, TailFactor>(
        base_tag, n, write_block,
        kernel::loop::invariant(center),
        kernel::loop::invariant(rstd));
    y.commit();
  }
};

template <typename Config = LayerNormConfig<>>
VECOPS_INLINE constexpr auto layer_norm(Config config = {}) {
  return LayerNorm<Config>{config};
}

} // namespace vecops::ops

#endif // VECOPS_OPS_LAYERNORM_H
