//
// Created by renyz on 2026/7/10.
//

#ifndef VECOPS_OPS_SOFTMAX_H
#define VECOPS_OPS_SOFTMAX_H

#include "vecops/Assertion.h"
#include "vecops/CoreTypes.h"
#include "vecops/gemm/DataAccess.h"
#include "vecops/gemm/HOP.h"
#include "vecops/gemm/Tensor.h"
#include "vecops/gemm/Workspace.h"
#include "vecops/vec/Vec.h"

#include <limits>
#include <type_traits>
#include <utility>

namespace vecops::ops {

enum class SoftmaxExpMode {
  Strict,
  Fast,
  Estimate,
};

template <
    typename ComputeT = float32_t,
    typename VecTag = vec::ScalableTag<ComputeT, 0>,
    SoftmaxExpMode ExpMode = SoftmaxExpMode::Strict>
struct SoftmaxConfig {
  using ComputeType = ComputeT;
  using Tag = VecTag;

  static constexpr SoftmaxExpMode exp_mode = ExpMode;
};

namespace details {

template <typename Config>
using softmax_compute_t = typename std::remove_cvref_t<Config>::ComputeType;

template <typename Config>
using softmax_tag_t = typename std::remove_cvref_t<Config>::Tag;

template <typename Tag>
VECOPS_INLINE auto softmax_vector_step_value(Tag tag) {
  using TagT = std::remove_cvref_t<Tag>;
  if constexpr (TagT::is_runtime_size) {
    return gemm::Any{vec::size(tag)};
  } else {
    return gemm::Const<vec::size(TagT{})>{};
  }
}

template <typename InLayout, typename OutLayout>
VECOPS_INLINE void validate_softmax_layouts(
    const InLayout& in,
    const OutLayout& out) {
  static_assert(InLayout::Ndim >= 1, "Softmax input rank must be at least 1");
  static_assert(
      OutLayout::Ndim == InLayout::Ndim,
      "Softmax output rank must match input rank");

  constexpr int rank = InLayout::Ndim;
  for (int d = 0; d < rank; ++d) {
    VECOPS_ASSERT(
        in.shape()[d] == out.shape()[d],
        "Softmax input/output shape mismatch at dim %d", d);
  }
  VECOPS_ASSERT(
      in.shape()[rank - 1] > 0,
      "Softmax normalized dimension must be non-empty");
}

} // namespace details

template <typename Config = SoftmaxConfig<>>
struct Softmax {
  using ConfigType = std::remove_cvref_t<Config>;
  using ComputeType = details::softmax_compute_t<Config>;
  using Tag = details::softmax_tag_t<Config>;

  static constexpr SoftmaxExpMode ExpMode = ConfigType::exp_mode;

public:
  const Config config;

  VECOPS_INLINE constexpr explicit Softmax(Config cfg = {}) : config(cfg) {}

  template <
      typename InSpec,
      typename OutSpec,
      std::enable_if_t<
          gemm::is_input_spec_v<InSpec> && gemm::is_output_spec_v<OutSpec>,
          bool> = true>
  VECOPS_INLINE nint_t required_workspace(
      const InSpec& in,
      const OutSpec& out) const {
    static_assert(gemm::is_input_spec_v<InSpec>, "Softmax input must be an InputSpec");
    static_assert(gemm::is_output_spec_v<OutSpec>, "Softmax output must be an OutputSpec");
    details::validate_softmax_layouts(in.input_layout(), out.output_layout());
    return required_workspace_impl(in, out);
  }

  template <
      typename InLayout,
      typename OutLayout,
      std::enable_if_t<
          gemm::is_layout<std::remove_cvref_t<InLayout>> &&
          gemm::is_layout<std::remove_cvref_t<OutLayout>>,
          bool> = true>
  VECOPS_INLINE nint_t required_workspace(
      const InLayout& in,
      const OutLayout& out) const {
    return required_workspace(
        gemm::InputSpec<ComputeType, ComputeType, std::remove_cvref_t<InLayout>>(in),
        gemm::OutputSpec<ComputeType, ComputeType, std::remove_cvref_t<OutLayout>>(out));
  }

  template <typename InSpec, typename OutSpec>
  void operator()(
      gemm::WorkspaceView& workspace,
      const InSpec& in,
      const OutSpec& out) const {
    static_assert(gemm::is_input_spec_v<InSpec>, "Softmax input must be an InputSpec");
    static_assert(gemm::is_output_spec_v<OutSpec>, "Softmax output must be an OutputSpec");

    const auto& in_layout = in.input_layout();
    const auto& out_layout = out.output_layout();
    details::validate_softmax_layouts(in_layout, out_layout);

    auto mark = workspace.mark();
    {
      constexpr int prefix_rank = std::remove_cvref_t<decltype(in_layout)>::Ndim - 1;
      gemm::hop::for_each_dims<prefix_rank>(
          [this, &workspace](const auto& in_row, const auto& out_row) VECOPS_INLINE_LAMBDA {
            this->run_row(workspace, in_row, out_row);
          },
          in,
          out);
    }
    workspace.rewind(mark);
  }

  template <typename InSpec, typename OutSpec>
  VECOPS_INLINE void operator()(const InSpec& in, const OutSpec& out) const {
    gemm::Workspace workspace(required_workspace(in, out));
    auto view = workspace.view();
    (*this)(view, in, out);
  }

private:
  template <typename V, typename M>
  VECOPS_INLINE static V apply_exp(V v, M mask, V default_v) {
    if constexpr (ExpMode == SoftmaxExpMode::Strict) {
      return vec::exp_neg(v, mask, default_v);
    } else if constexpr (ExpMode == SoftmaxExpMode::Fast) {
      return vec::exp_neg_fast(v, mask, default_v);
    } else {
      static_assert(ExpMode == SoftmaxExpMode::Estimate, "Unsupported Softmax exp mode");
      return vec::exp_neg_est(v, mask, default_v);
    }
  }

  template <typename InSpec, typename OutSpec>
  VECOPS_INLINE void run_row(
      gemm::WorkspaceView& workspace,
      const InSpec& in,
      const OutSpec& out) const {
    const auto& in_layout = in.input_layout();
    using InLayout = std::remove_cvref_t<decltype(in_layout)>;
    static_assert(InLayout::Ndim == 1, "Softmax row spec must be rank 1");

    Tag t;
    using VecT = vec::Vec<Tag>;
    auto mark = workspace.mark();
    {
      auto x = in.bind(workspace);
      auto y = out.bind(workspace);

      const auto normalized_count = gemm::size<0>(in_layout);
      const nint_t normalized_n = normalized_count;
      ComputeType* exp_cache = workspace.allocate<ComputeType>(normalized_n);
      const auto step = details::softmax_vector_step_value(t);
      const auto negative_infinity =
          static_cast<ComputeType>(-std::numeric_limits<double>::infinity());

      const auto v_max = gemm::hop::scan<1>(
          vec::fill(t, negative_infinity),
          normalized_count,
          step,
          [&](VecT acc, nint_t col, auto&& count) VECOPS_INLINE_LAMBDA {
            auto xv = x(t, count, col);
            auto mask = vec::mwhilelt(t, 0, static_cast<nint_t>(count));
            return vec::max(acc, xv, mask);
          });
      const auto max_value = vec::reduce_max(t, v_max);
      const auto max_v = vec::fill(t, max_value);

      const auto v_sum = gemm::hop::scan<1>(
          vec::zeros(t),
          normalized_count,
          step,
          [&](VecT acc, nint_t col, auto&& count) VECOPS_INLINE_LAMBDA {
            auto xv = x(t, count, col);
            auto mask = vec::mwhilelt(t, 0, static_cast<nint_t>(count));
            auto shifted = vec::sub(xv, max_v);
            auto exp_v = apply_exp(shifted, mask, vec::zeros(t));
            vec::store(t, exp_cache + col, exp_v, vec::opt::masked(mask));
            return vec::add(acc, exp_v);
          });
      const auto inv_sum = ComputeType(1) / vec::reduce_add(t, v_sum);
      const auto inv_sum_v = vec::fill(t, inv_sum);

      // Loop unrolling 2
      gemm::hop::map<2>(
          normalized_count,
          step,
          [&](nint_t col, auto&& count) VECOPS_INLINE_LAMBDA {
            auto mask = vec::mwhilelt(t, 0, static_cast<nint_t>(count));
            auto exp_v = vec::load(
                t, exp_cache + col, vec::opt::masked(mask));
            y(t, vec::mul(exp_v, inv_sum_v), count, col);
          });
    }
    workspace.rewind(mark);
  }

  template <typename InSpec, typename OutSpec>
  VECOPS_INLINE nint_t required_workspace_impl(
      const InSpec& in,
      const OutSpec& out) const {
    using InLayout = typename std::remove_cvref_t<InSpec>::InputLayout;
    constexpr int prefix_rank = InLayout::Ndim - 1;
    const auto in_row = row_input_spec<prefix_rank>(in);
    const auto out_row = row_output_spec<prefix_rank>(out);
    const nint_t row_workspace = gemm::required_workspace(in_row, out_row);
    const nint_t normalized_count = gemm::size<0>(in_row.input_layout());
    const nint_t exp_cache_bytes =
        normalized_count * static_cast<nint_t>(sizeof(ComputeType));
    return gemm::details::workspace_round_up(
        row_workspace + exp_cache_bytes,
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

template <typename Config = SoftmaxConfig<>>
VECOPS_INLINE constexpr auto softmax(Config config = {}) {
  return Softmax<Config>{config};
}

} // namespace vecops::ops

#endif // VECOPS_OPS_SOFTMAX_H
