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
      if constexpr (IsBfloat16V<InElement>) {
        nint_t element_count = 1;
        for (int d = 0; d <= PrefixRank; ++d) {
          element_count *= in.input_layout().shape()[d];
        }
        if (element_count > 64 * 1024) {
          kernel::loop::for_each_dims<PrefixRank>(
              [this, &workspace, &scale, &bias](
                  const auto& in_row, const auto& out_row) {
                run_row_sve_16bit<true>(
                    workspace, in_row, scale, bias, out_row);
              },
              in, out);
        } else {
          kernel::loop::for_each_dims<PrefixRank>(
              [this, &workspace, &scale, &bias](
                  const auto& in_row, const auto& out_row) {
                run_row_sve_16bit<false>(
                    workspace, in_row, scale, bias, out_row);
              },
              in, out);
        }
      } else {
        kernel::loop::for_each_dims<PrefixRank>(
            [this, &workspace, &scale, &bias](
                const auto& in_row, const auto& out_row) {
              run_row_sve_16bit<false>(
                  workspace, in_row, scale, bias, out_row);
            },
            in, out);
      }
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
  template <bool PrefetchBf16, typename InSpec, typename ScaleSpec,
            typename BiasSpec, typename OutSpec>
  VECOPS_INLINE void run_row_sve_16bit(
      kernel::WorkspaceView& workspace, const InSpec& in,
      const ScaleSpec& scale, const BiasSpec& bias,
      const OutSpec& out) const {
    using Element = std::remove_const_t<typename InSpec::MemoryElement>;
    using ConversionOrder = std::conditional_t<
        IsFloat16V<Element>, vec::cvt::Unordered, vec::cvt::Ordered>;
    using Prefetch = tensor::PrefetchPolicy<
        IsBfloat16V<Element> && PrefetchBf16, 4,
        tensor::PrefetchFootprint::whole_block>;
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
          auto x_scan = x.scan(
              full_tag, tensor::coord(0), tensor::axis<0>, n);
          auto sum = vec::zeros(full_tag);
          auto sum_sq = vec::zeros(full_tag);
          float32_t tail_sum = 0.0f;
          float32_t tail_sum_sq = 0.0f;
          while (x_scan.has_full()) {
            auto value = x_scan.load_full();
            sum = vec::add(sum, value);
            sum_sq = vec::fmadd(value, value, sum_sq);
            x_scan.advance_full();
          }
          while (!x_scan.empty()) {
            auto value = x_scan.load_tail(tail_tag);
            tail_sum += vec::reduce_add(tail_tag, value);
            tail_sum_sq +=
                vec::reduce_add(tail_tag, vec::mul(value, value));
            x_scan.advance_tail(tail_tag);
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

          auto xv = x.scan(full_tag, tensor::coord(0), tensor::axis<0>, n);
          auto gv = gamma.scan(
              full_tag, tensor::coord(0), tensor::axis<0>, n);
          auto bv = beta.scan(
              full_tag, tensor::coord(0), tensor::axis<0>, n);
          auto yv = y.scan(full_tag, tensor::coord(0), tensor::axis<0>, n);
          while (xv.has_full()) {
            auto value = xv.load_full();
            auto scale_value = gv.load_full();
            auto bias_value = bv.load_full();
            value = vec::fmadd(value, full_rstd, full_shift);
            yv.store_full(vec::fmadd(value, scale_value, bias_value));
            xv.advance_full();
            gv.advance_full();
            bv.advance_full();
            yv.advance_full();
          }
          while (!xv.empty()) {
            auto value = xv.load_tail(tail_tag);
            auto scale_value = gv.load_tail(tail_tag);
            auto bias_value = bv.load_tail(tail_tag);
            value = vec::fmadd(value, tail_rstd, tail_shift);
            yv.store_tail(
                tail_tag, vec::fmadd(value, scale_value, bias_value));
            xv.advance_tail(tail_tag);
            gv.advance_tail(tail_tag);
            bv.advance_tail(tail_tag);
            yv.advance_tail(tail_tag);
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
    kernel::with_operands(
        workspace,
        tensor::operand(in, InPolicy{}),
        tensor::operand(out, OutPolicy{}),
        [this, &in, &gamma, &beta](auto& x, auto& y) {
          Tag tag{};
          const nint_t n = in.input_layout().shape()[0];
          const nint_t lanes = vec::size(tag);
          auto sum0 = vec::zeros(tag);
          auto sum1 = vec::zeros(tag);
          auto sum2 = vec::zeros(tag);
          auto sum3 = vec::zeros(tag);
          auto sum_sq0 = vec::zeros(tag);
          auto sum_sq1 = vec::zeros(tag);
          auto sum_sq2 = vec::zeros(tag);
          auto sum_sq3 = vec::zeros(tag);
          nint_t col = 0;
          for (; col + 4 * lanes <= n; col += 4 * lanes) {
            auto x0 = x.load(tag, tensor::coord(col), vec::opt::unmasked);
            auto x1 = x.load(
                tag, tensor::coord(col + lanes), vec::opt::unmasked);
            auto x2 = x.load(
                tag, tensor::coord(col + 2 * lanes), vec::opt::unmasked);
            auto x3 = x.load(
                tag, tensor::coord(col + 3 * lanes), vec::opt::unmasked);
            sum0 = vec::add(sum0, x0);
            sum1 = vec::add(sum1, x1);
            sum2 = vec::add(sum2, x2);
            sum3 = vec::add(sum3, x3);
            sum_sq0 = vec::fmadd(x0, x0, sum_sq0);
            sum_sq1 = vec::fmadd(x1, x1, sum_sq1);
            sum_sq2 = vec::fmadd(x2, x2, sum_sq2);
            sum_sq3 = vec::fmadd(x3, x3, sum_sq3);
          }
          if (col + lanes <= n) {
            auto value = x.load(tag, tensor::coord(col), vec::opt::unmasked);
            sum0 = vec::add(sum0, value);
            sum_sq0 = vec::fmadd(value, value, sum_sq0);
            col += lanes;
          }
          if (col + lanes <= n) {
            auto value = x.load(tag, tensor::coord(col), vec::opt::unmasked);
            sum1 = vec::add(sum1, value);
            sum_sq1 = vec::fmadd(value, value, sum_sq1);
            col += lanes;
          }
          if (col + lanes <= n) {
            auto value = x.load(tag, tensor::coord(col), vec::opt::unmasked);
            sum2 = vec::add(sum2, value);
            sum_sq2 = vec::fmadd(value, value, sum_sq2);
            col += lanes;
          }
          if (col < n) {
            auto value = x.load(
                tag, tensor::coord(col), vec::opt::first(n - col));
            sum0 = vec::add(sum0, value);
            sum_sq0 = vec::fmadd(value, value, sum_sq0);
          }
          sum0 = vec::add(sum0, sum1);
          sum0 = vec::add(sum0, sum2);
          sum0 = vec::add(sum0, sum3);
          sum_sq0 = vec::add(sum_sq0, sum_sq1);
          sum_sq0 = vec::add(sum_sq0, sum_sq2);
          sum_sq0 = vec::add(sum_sq0, sum_sq3);
          const ComputeType sum = vec::reduce_add(tag, sum0);
          const ComputeType sum_sq = vec::reduce_add(tag, sum_sq0);
          const ComputeType inv_n =
              ComputeType(1) / static_cast<ComputeType>(n);
          const ComputeType mean = sum * inv_n;
          const ComputeType variance =
              std::max(sum_sq * inv_n - mean * mean, ComputeType(0));
          const ComputeType rstd =
              ComputeType(1) / std::sqrt(variance + config.eps);
          const auto mean_v = vec::fill(tag, mean);
          const auto rstd_v = vec::fill(tag, rstd);

          col = 0;
          for (; col + 4 * lanes <= n; col += 4 * lanes) {
            auto x0 = x.load(tag, tensor::coord(col), vec::opt::unmasked);
            auto x1 = x.load(
                tag, tensor::coord(col + lanes), vec::opt::unmasked);
            auto x2 = x.load(
                tag, tensor::coord(col + 2 * lanes), vec::opt::unmasked);
            auto x3 = x.load(
                tag, tensor::coord(col + 3 * lanes), vec::opt::unmasked);
            auto g0 = gamma.load(
                tag, tensor::coord(col), vec::opt::unmasked);
            auto g1 = gamma.load(
                tag, tensor::coord(col + lanes), vec::opt::unmasked);
            auto g2 = gamma.load(
                tag, tensor::coord(col + 2 * lanes), vec::opt::unmasked);
            auto g3 = gamma.load(
                tag, tensor::coord(col + 3 * lanes), vec::opt::unmasked);
            auto b0 = beta.load(tag, tensor::coord(col), vec::opt::unmasked);
            auto b1 = beta.load(
                tag, tensor::coord(col + lanes), vec::opt::unmasked);
            auto b2 = beta.load(
                tag, tensor::coord(col + 2 * lanes), vec::opt::unmasked);
            auto b3 = beta.load(
                tag, tensor::coord(col + 3 * lanes), vec::opt::unmasked);
            auto n0 = vec::mul(vec::sub(x0, mean_v), rstd_v);
            auto n1 = vec::mul(vec::sub(x1, mean_v), rstd_v);
            auto n2 = vec::mul(vec::sub(x2, mean_v), rstd_v);
            auto n3 = vec::mul(vec::sub(x3, mean_v), rstd_v);
            y.store(tag, tensor::coord(col), vec::fmadd(n0, g0, b0),
                    vec::opt::unmasked);
            y.store(tag, tensor::coord(col + lanes), vec::fmadd(n1, g1, b1),
                    vec::opt::unmasked);
            y.store(tag, tensor::coord(col + 2 * lanes),
                    vec::fmadd(n2, g2, b2), vec::opt::unmasked);
            y.store(tag, tensor::coord(col + 3 * lanes),
                    vec::fmadd(n3, g3, b3), vec::opt::unmasked);
          }
          for (; col + lanes <= n; col += lanes) {
            write_block(x, gamma, beta, y, tag, col, mean_v, rstd_v,
                        vec::opt::unmasked);
          }
          if (col < n) {
            write_block(x, gamma, beta, y, tag, col, mean_v, rstd_v,
                        vec::opt::first(n - col));
          }
          y.commit();
        });
  }

  template <typename X, typename Gamma, typename Beta, typename Y,
            vec::VectorTag VTag, typename Active>
  VECOPS_ALWAYS_INLINE static void write_block(
      X& x, Gamma& gamma, Beta& beta, Y& y, VTag tag, nint_t col,
      vec::Vec<VTag> mean, vec::Vec<VTag> rstd, Active active) {
    auto xv = x.load(tag, tensor::coord(col), active);
    auto gamma_v = gamma.load(tag, tensor::coord(col), active);
    auto beta_v = beta.load(tag, tensor::coord(col), active);
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
