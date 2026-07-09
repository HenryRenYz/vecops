//
// Created by renyz on 2026/7/9.
//

#ifndef VECOPS_OPS_LAYERNORM_H
#define VECOPS_OPS_LAYERNORM_H

#include "vecops/Assertion.h"
#include "vecops/CoreTypes.h"
#include "vecops/gemm/DataAccess.h"
#include "vecops/gemm/HOP.h"
#include "vecops/gemm/Tensor.h"
#include "vecops/gemm/Workspace.h"
#include "vecops/vec/Vec.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <tuple>
#include <type_traits>
#include <utility>

namespace vecops::ops {

template <
    typename ComputeT = float32_t,
    typename VecTag = vec::ScalableTag<ComputeT, 0>>
struct LayerNormConfig {
  using ComputeType = ComputeT;
  using Tag = VecTag;

  ComputeT eps = ComputeT(1e-5f);
};

namespace details {

template <typename Config>
using layernorm_compute_t = typename std::remove_cvref_t<Config>::ComputeType;

template <typename Config>
using layernorm_tag_t = typename std::remove_cvref_t<Config>::Tag;

// TODO sooner or later I will take this down
template <typename Tag>
VECOPS_INLINE auto vector_step_value(Tag tag) {
  using TagT = std::remove_cvref_t<Tag>;
  if constexpr (TagT::is_runtime_size) {
    return gemm::Any{vec::size(tag)};
  } else {
    return gemm::Const<vec::size(TagT{})>{};
  }
}

template <typename InLayout, typename ScaleLayout, typename BiasLayout, typename OutLayout>
VECOPS_INLINE void validate_layernorm_layouts(
    const InLayout& in,
    const ScaleLayout& scale,
    const BiasLayout& bias,
    const OutLayout& out) {
  static_assert(InLayout::Ndim >= 1, "LayerNorm input rank must be at least 1");
  static_assert(OutLayout::Ndim == InLayout::Ndim, "LayerNorm output rank must match input rank");
  static_assert(ScaleLayout::Ndim == 1, "LayerNorm scale rank must be 1");
  static_assert(BiasLayout::Ndim == 1, "LayerNorm bias rank must be 1");

  constexpr int rank = InLayout::Ndim;
  for (int d = 0; d < rank; ++d) {
    VECOPS_ASSERT(
        in.shape()[d] == out.shape()[d],
        "LayerNorm input/output shape mismatch at dim %d", d);
  }
  const nint_t normalized_size = in.shape()[rank - 1];
  VECOPS_ASSERT(normalized_size > 0, "LayerNorm normalized dimension must be non-empty");
  VECOPS_ASSERT(
      scale.shape()[0] == normalized_size,
      "LayerNorm scale size mismatch: %td != %td", scale.shape()[0], normalized_size);
  VECOPS_ASSERT(
      bias.shape()[0] == normalized_size,
      "LayerNorm bias size mismatch: %td != %td", bias.shape()[0], normalized_size);
}

} // namespace details

template <typename Config = LayerNormConfig<>>
struct LayerNorm {
  using ComputeType = details::layernorm_compute_t<Config>;
  using Tag = details::layernorm_tag_t<Config>;

public:
  const Config config;

  VECOPS_INLINE constexpr explicit LayerNorm(Config cfg = {}) : config(cfg) {}

  template <
      typename InSpec,
      typename ScaleSpec,
      typename BiasSpec,
      typename OutSpec,
      std::enable_if_t<
          gemm::is_input_spec_v<InSpec> &&
          gemm::is_input_spec_v<ScaleSpec> &&
          gemm::is_input_spec_v<BiasSpec> &&
          gemm::is_output_spec_v<OutSpec>,
          bool> = true>
  VECOPS_INLINE nint_t required_workspace(
      const InSpec& in,
      const ScaleSpec& scale,
      const BiasSpec& bias,
      const OutSpec& out) const {
    static_assert(gemm::is_input_spec_v<InSpec>, "LayerNorm input must be an InputSpec");
    static_assert(gemm::is_input_spec_v<ScaleSpec>, "LayerNorm scale must be an InputSpec");
    static_assert(gemm::is_input_spec_v<BiasSpec>, "LayerNorm bias must be an InputSpec");
    static_assert(gemm::is_output_spec_v<OutSpec>, "LayerNorm output must be an OutputSpec");
    details::validate_layernorm_layouts(
        in.input_layout(),
        scale.input_layout(),
        bias.input_layout(),
        out.output_layout());
    return required_workspace_impl(in, scale, bias, out);
  }

  template <
      typename InLayout,
      typename ScaleLayout,
      typename BiasLayout,
      typename OutLayout,
      std::enable_if_t<
          gemm::is_layout<std::remove_cvref_t<InLayout>> &&
          gemm::is_layout<std::remove_cvref_t<ScaleLayout>> &&
          gemm::is_layout<std::remove_cvref_t<BiasLayout>> &&
          gemm::is_layout<std::remove_cvref_t<OutLayout>>,
          bool> = true>
  VECOPS_INLINE nint_t required_workspace(
      const InLayout& in,
      const ScaleLayout& scale,
      const BiasLayout& bias,
      const OutLayout& out) const {
    return required_workspace(
        gemm::InputSpec<ComputeType, ComputeType, std::remove_cvref_t<InLayout>>(in),
        gemm::InputSpec<ComputeType, ComputeType, std::remove_cvref_t<ScaleLayout>>(scale),
        gemm::InputSpec<ComputeType, ComputeType, std::remove_cvref_t<BiasLayout>>(bias),
        gemm::OutputSpec<ComputeType, ComputeType, std::remove_cvref_t<OutLayout>>(out));
  }

  template <
      typename InSpec,
      typename ScaleSpec,
      typename BiasSpec,
      typename OutSpec>
  VECOPS_INLINE void operator()(
      gemm::WorkspaceView& workspace,
      const InSpec& in,
      const ScaleSpec& scale,
      const BiasSpec& bias,
      const OutSpec& out) const {
    static_assert(gemm::is_input_spec_v<InSpec>, "LayerNorm input must be an InputSpec");
    static_assert(gemm::is_input_spec_v<ScaleSpec>, "LayerNorm scale must be an InputSpec");
    static_assert(gemm::is_input_spec_v<BiasSpec>, "LayerNorm bias must be an InputSpec");
    static_assert(gemm::is_output_spec_v<OutSpec>, "LayerNorm output must be an OutputSpec");

    const auto& in_layout = in.input_layout();
    const auto& scale_layout = scale.input_layout();
    const auto& bias_layout = bias.input_layout();
    const auto& out_layout = out.output_layout();
    details::validate_layernorm_layouts(in_layout, scale_layout, bias_layout, out_layout);

    auto mark = workspace.mark();
    {
      auto gamma = scale.bind(workspace);
      auto beta = bias.bind(workspace);

      constexpr int prefix_rank = std::remove_cvref_t<decltype(in_layout)>::Ndim - 1;
      gemm::hop::for_each_dims<prefix_rank>(
          [this, &workspace, &gamma, &beta](const auto& in_row, const auto& out_row) VECOPS_ALWAYS_INLINE_LAMBDA {
            this->run_row(workspace, in_row, gamma, beta, out_row);
          }, in, out
      );
    }
    workspace.rewind(mark);
  }

  template <
      typename InSpec,
      typename ScaleSpec,
      typename BiasSpec,
      typename OutSpec>
  VECOPS_INLINE void operator()(
      const InSpec& in,
      const ScaleSpec& scale,
      const BiasSpec& bias,
      const OutSpec& out) const {
    gemm::Workspace workspace(required_workspace(in, scale, bias, out));
    auto view = workspace.view();
    (*this)(view, in, scale, bias, out);
  }

private:
  template <
      typename InSpec,
      typename ScaleAccessor,
      typename BiasAccessor,
      typename OutSpec>
  VECOPS_INLINE void run_row(
      gemm::WorkspaceView& workspace,
      const InSpec& in,
      const ScaleAccessor& gamma,
      const BiasAccessor& beta,
      const OutSpec& out) const {
    const auto & in_layout = in.input_layout();
    using InLayout = std::remove_cvref_t<decltype(in_layout)>;
    static_assert(InLayout::Ndim == 1, "LayerNorm row spec must be rank 1");

    Tag t;
    using VecT = vec::Vec<Tag>;
    auto mark = workspace.mark();
    {
      auto x = in.bind(workspace);
      auto y = out.bind(workspace);

      const auto normalized_count = gemm::size<0>(in_layout);
      const auto step = details::vector_step_value(t);

      const auto [v_mean, v_var] = gemm::hop::scan(
          std::make_pair(vec::zeros(t), vec::zeros(t)), normalized_count, step,
          [&](std::pair<VecT, VecT> acc, nint_t col, auto&& count) VECOPS_ALWAYS_INLINE_LAMBDA {
            auto xv = x(t, count, col);
            return std::make_pair(vec::add(acc.first, xv), vec::fmadd(xv, xv, acc.second));
          }
      );

      const auto inv_n = ComputeType(1) / static_cast<ComputeType>(normalized_count);
      const auto sum = vec::reduce_add(t, v_mean);
      const auto sum_sq = vec::reduce_add(t, v_var);
      const auto mean = sum * inv_n;
      const auto variance = std::max(sum_sq * inv_n - mean * mean, ComputeType(0));
      const auto rstd = ComputeType(1) / std::sqrt(variance + config.eps);

      const auto mean_v = vec::fill(t, mean);
      const auto rstd_v = vec::fill(t, rstd);

      gemm::hop::map(normalized_count, step, [&](nint_t col, auto&& count) VECOPS_ALWAYS_INLINE_LAMBDA {
        auto xv = x(t, count, col);
        auto gamma_v = gamma(t, count, col);
        auto beta_v = beta(t, count, col);
        auto centered = vec::sub(xv, mean_v);
        auto normalized = vec::mul(centered, rstd_v);
        auto affine = vec::fmadd(normalized, gamma_v, beta_v);
        y(t, affine, count, col);
      });
    }
    workspace.rewind(mark);
  }

  template <
      typename InSpec,
      typename ScaleSpec,
      typename BiasSpec,
      typename OutSpec>
  VECOPS_INLINE nint_t required_workspace_impl(
      const InSpec& in,
      const ScaleSpec& scale,
      const BiasSpec& bias,
      const OutSpec& out) const {
    using InLayout = typename std::remove_cvref_t<InSpec>::InputLayout;
    constexpr int prefix_rank = InLayout::Ndim - 1;
    const auto in_row = row_input_spec<prefix_rank>(in);
    const auto out_row = row_output_spec<prefix_rank>(out);
    const nint_t row_workspace = gemm::required_workspace(in_row, out_row);
    return gemm::details::workspace_round_up(
        gemm::required_workspace(scale, bias) + row_workspace,
        vec::DEFAULT_ALIGNMENT);
  }

  template <int PrefixRank, typename Layout>
  VECOPS_INLINE static auto row_layout(const Layout& layout) {
    if constexpr (PrefixRank == 0) {
      return layout;
    } else {
      return row_layout<PrefixRank - 1>(gemm::remove<0>(layout));
    }
  }

  template <int PrefixRank, typename Spec>
  VECOPS_INLINE static auto row_input_spec(const Spec& spec) {
    using SpecT = std::remove_cvref_t<Spec>;
    auto layout = row_layout<PrefixRank>(spec.input_layout());
    using RowLayout = std::remove_cvref_t<decltype(layout)>;
    return gemm::InputSpec<
        typename SpecT::OutputElement,
        typename SpecT::InputTensor::ElementType,
        RowLayout,
        typename SpecT::Transform>(layout, spec.transform());
  }

  template <int PrefixRank, typename Spec>
  VECOPS_INLINE static auto row_output_spec(const Spec& spec) {
    using SpecT = std::remove_cvref_t<Spec>;
    auto layout = row_layout<PrefixRank>(spec.output_layout());
    using RowLayout = std::remove_cvref_t<decltype(layout)>;
    using TOut = typename SpecT::OutputTensor::ElementType;
    return gemm::OutputSpec<
        typename SpecT::InputElement,
        TOut,
        RowLayout,
        typename SpecT::Transform>(layout, spec.transform());
  }
};

template <typename Config = LayerNormConfig<>>
VECOPS_INLINE constexpr auto layer_norm(Config config = {}) {
  return LayerNorm<Config>{config};
}

} // namespace vecops::ops

#endif // VECOPS_OPS_LAYERNORM_H
