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

template <typename T>
struct IsInputSpec : std::false_type {};

template <typename... Args>
struct IsInputSpec<gemm::InputSpec<Args...>> : std::true_type {};

template <typename T>
static constexpr bool is_input_spec_v = IsInputSpec<std::remove_cvref_t<T>>::value;

template <typename T>
struct IsOutputSpec : std::false_type {};

template <typename... Args>
struct IsOutputSpec<gemm::OutputSpec<Args...>> : std::true_type {};

template <typename T>
static constexpr bool is_output_spec_v = IsOutputSpec<std::remove_cvref_t<T>>::value;

template <typename Tag>
VECOPS_ALWAYS_INLINE vec::TypeOf<Tag> horizontal_sum(Tag tag, vec::Vec<Tag> v) {
  vec::TypeOf<Tag> acc{};
  for (nint_t i = 0; i < vec::size(tag); ++i) {
    acc += vec::get(tag, v, i);
  }
  return acc;
}

template <typename InLayout, typename ScaleLayout, typename BiasLayout, typename OutLayout>
void validate_layernorm_layouts(
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

  const Config config;

  constexpr explicit LayerNorm(Config cfg = {}) : config(cfg) {}

  template <
      typename InSpec,
      typename ScaleSpec,
      typename BiasSpec,
      typename OutSpec>
  nint_t required_workspace(
      const InSpec& in,
      const ScaleSpec& scale,
      const BiasSpec& bias,
      const OutSpec& out) const {
    static_assert(details::is_input_spec_v<InSpec>, "LayerNorm input must be an InputSpec");
    static_assert(details::is_input_spec_v<ScaleSpec>, "LayerNorm scale must be an InputSpec");
    static_assert(details::is_input_spec_v<BiasSpec>, "LayerNorm bias must be an InputSpec");
    static_assert(details::is_output_spec_v<OutSpec>, "LayerNorm output must be an OutputSpec");
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
  nint_t required_workspace(
      const InLayout& in,
      const ScaleLayout& scale,
      const BiasLayout& bias,
      const OutLayout& out) const {
    details::validate_layernorm_layouts(in, scale, bias, out);
    return gemm::required_workspace(
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
  void operator()(
      gemm::WorkspaceView& workspace,
      const InSpec& in,
      const ScaleSpec& scale,
      const BiasSpec& bias,
      const OutSpec& out) const {
    static_assert(details::is_input_spec_v<InSpec>, "LayerNorm input must be an InputSpec");
    static_assert(details::is_input_spec_v<ScaleSpec>, "LayerNorm scale must be an InputSpec");
    static_assert(details::is_input_spec_v<BiasSpec>, "LayerNorm bias must be an InputSpec");
    static_assert(details::is_output_spec_v<OutSpec>, "LayerNorm output must be an OutputSpec");

    const auto& in_layout = in.input_layout();
    const auto& scale_layout = scale.input_layout();
    const auto& bias_layout = bias.input_layout();
    const auto& out_layout = out.output_layout();
    details::validate_layernorm_layouts(in_layout, scale_layout, bias_layout, out_layout);

    auto mark = workspace.mark();
    {
      auto gamma = scale.bind(workspace);
      auto beta = bias.bind(workspace);
      run_bound(workspace, in, out, gamma, beta);
    }
    workspace.rewind(mark);
  }

  template <
      typename InSpec,
      typename ScaleSpec,
      typename BiasSpec,
      typename OutSpec>
  void operator()(
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
      typename OutSpec,
      typename ScaleAccessor,
      typename BiasAccessor>
  void run_bound(
      gemm::WorkspaceView& workspace,
      const InSpec& in,
      const OutSpec& out,
      const ScaleAccessor& gamma,
      const BiasAccessor& beta) const {
    using InLayout = typename std::remove_cvref_t<InSpec>::InputLayout;
    constexpr int rank = InLayout::Ndim;
    constexpr int prefix_rank = rank - 1;

    gemm::hop::for_each_dims<prefix_rank>(
        [&](const auto& in_row, const auto& out_row) {
          run_row(workspace, in_row, gamma, beta, out_row);
        },
        in,
        out);
  }

  template <
      typename InSpec,
      typename ScaleAccessor,
      typename BiasAccessor,
      typename OutSpec>
  void run_row(
      gemm::WorkspaceView& workspace,
      const InSpec& in,
      const ScaleAccessor& gamma,
      const BiasAccessor& beta,
      const OutSpec& out) const {
    auto mark = workspace.mark();
    {
      auto x = in.bind(workspace);
      auto y = out.bind(workspace);
      run_row_bound(in.input_layout(), x, gamma, beta, y);
    }
    workspace.rewind(mark);
  }

  template <
      typename InLayout,
      typename InAccessor,
      typename ScaleAccessor,
      typename BiasAccessor,
      typename OutAccessor>
  void run_row_bound(
      const InLayout& in_layout,
      const InAccessor& x,
      const ScaleAccessor& gamma,
      const BiasAccessor& beta,
      const OutAccessor& y) const {
    static_assert(InLayout::Ndim == 1, "LayerNorm row spec must be rank 1");
    Tag tag;
    const nint_t normalized_size = in_layout.shape()[0];
    auto sum_v = vec::zeros(tag);
    auto sum_sq_v = vec::zeros(tag);

    for (nint_t col = 0; col < normalized_size; col += vec::size(tag)) {
      const nint_t count = std::min(vec::size(tag), normalized_size - col);
      auto xv = x(tag, gemm::Any{count}, col);
      sum_v = vec::add(sum_v, xv);
      sum_sq_v = vec::fmadd(xv, xv, sum_sq_v);
    }

    const ComputeType inv_n = ComputeType(1) / static_cast<ComputeType>(normalized_size);
    const ComputeType sum = details::horizontal_sum(tag, sum_v);
    const ComputeType sum_sq = details::horizontal_sum(tag, sum_sq_v);
    const ComputeType mean = sum * inv_n;
    const ComputeType variance =
        std::max(sum_sq * inv_n - mean * mean, ComputeType(0));
    const ComputeType rstd = ComputeType(1) / std::sqrt(variance + config.eps);

    const auto mean_v = vec::fill(tag, mean);
    const auto rstd_v = vec::fill(tag, rstd);
    for (nint_t col = 0; col < normalized_size; col += vec::size(tag)) {
      const nint_t count = std::min(vec::size(tag), normalized_size - col);
      auto xv = x(tag, gemm::Any{count}, col);
      auto gamma_v = gamma(tag, gemm::Any{count}, col);
      auto beta_v = beta(tag, gemm::Any{count}, col);
      auto centered = vec::sub(xv, mean_v);
      auto normalized = vec::mul(centered, rstd_v);
      auto affine = vec::fmadd(normalized, gamma_v, beta_v);
      y(tag, affine, gemm::Any{count}, col);
    }
  }

  template <
      typename InSpec,
      typename ScaleSpec,
      typename BiasSpec,
      typename OutSpec>
  nint_t required_workspace_impl(
      const InSpec& in,
      const ScaleSpec& scale,
      const BiasSpec& bias,
      const OutSpec& out) const {
    using InLayout = typename std::remove_cvref_t<InSpec>::InputLayout;
    constexpr int prefix_rank = InLayout::Ndim - 1;
    const auto in_row = first_row_spec<prefix_rank>(in);
    const auto out_row = first_row_spec<prefix_rank>(out);
    const nint_t row_workspace = gemm::required_workspace(in_row, out_row);
    return gemm::details::workspace_round_up(
        gemm::required_workspace(scale, bias) + row_workspace,
        vec::DEFAULT_ALIGNMENT);
  }

  template <int PrefixRank, typename Spec>
  static auto first_row_spec(const Spec& spec) {
    if constexpr (PrefixRank == 0) {
      return spec;
    } else {
      return first_row_spec<PrefixRank - 1>(
          gemm::hop::details::slice_at_actual_dim<0>(spec, 0));
    }
  }
};

template <typename Config = LayerNormConfig<>>
constexpr auto layer_norm(Config config = {}) {
  return LayerNorm<Config>{config};
}

} // namespace vecops::ops

#endif // VECOPS_OPS_LAYERNORM_H
