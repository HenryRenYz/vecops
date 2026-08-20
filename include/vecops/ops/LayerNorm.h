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

// Recursively drop leading dimensions until only the normalized row remains.
// `slice_view` preserves stride meta types and composes the projection.
template <int Count, typename Spec>
VECOPS_INLINE auto row_spec(const Spec& spec) {
  if constexpr (Count == 0) return spec;
  else return row_spec<Count - 1>(tensor::slice_view<0>(spec, 0));
}

/**
 * @brief Per-row access policies and loop tuning for one storage family.
 *
 * A recipe keeps every policy-level difference between row paths in one
 * place: the conversion order shared by all four operands, permutation
 * safety, the unroll and fusion parameters `run_row` consumes, and where
 * parameter sessions are constructed. `PerRowParams` records a measured
 * compiler preference rather than an algorithmic need: BiSheng schedules
 * the 16-bit row kernel better with sessions constructed inside the row
 * frame, while GCC keeps the generic path fastest with one hoisted
 * construction before the loop.
 */
template <typename ConversionOrder, int Full, int Tail, bool Fused,
          kernel::loop::TailCarryPolicy Carry, bool PermutationSafe,
          bool PerRowParams>
struct RowRecipe {
  static constexpr int FullFactor = Full;
  static constexpr int TailFactor = Tail;
  static constexpr bool FusedShift = Fused;
  static constexpr kernel::loop::TailCarryPolicy TailPolicy = Carry;
  static constexpr bool BindParamsPerRow = PerRowParams;

  template <tensor::AccessPlan Plan, int ReadPasses>
  using InputPolicy = tensor::InputAccessPolicy<
      0, ReadPasses, Plan, ConversionOrder, vec::cvt::Saturate,
      tensor::DefaultMemoryPolicy, PermutationSafe>;
  using OutputPolicy = tensor::OutputAccessPolicy<
      0, tensor::AccessPlan::direct, ConversionOrder, vec::cvt::Saturate,
      tensor::DefaultMemoryPolicy, PermutationSafe>;
  using ParamPolicy = tensor::InputAccessPolicy<
      0, 1, tensor::AccessPlan::direct, ConversionOrder, vec::cvt::Saturate,
      tensor::DefaultMemoryPolicy, PermutationSafe>;
};

/** @brief Ordered-conversion baseline with hoisted parameter sessions. */
using GenericRowRecipe = RowRecipe<
    vec::cvt::Ordered, 4, 1, false,
    kernel::loop::TailCarryPolicy::independent, false, false>;

#if defined(CPU_CAPABILITY_SVE) && defined(__ARM_FEATURE_SVE2) && \
    !defined(VECOPS_DISABLE_SVE_16BIT_LAYERNORM_FAST_PATH)
/**
 * @brief Half-precision specialization: one order-free saturating conversion
 * for the whole row region, fused shift, and a wider tail block. Layouts
 * whose stride is not provably unit keep this recipe and simply lower to
 * strided (gather/scatter) access.
 */
template <typename Element>
using SVE16RowRecipe = RowRecipe<
    std::conditional_t<
        IsFloat16V<Element>, vec::cvt::Unordered, vec::cvt::Ordered>,
    4, 2, true, kernel::loop::TailCarryPolicy::reuse_prefix, true, true>;
#endif

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
    // Storage-element and transform eligibility only: a layout whose final
    // stride is not provably unit keeps the same recipe and lowers to
    // strided access inside the row.
    constexpr bool UseSVE16 =
        std::same_as<ComputeType, float32_t> &&
        std::same_as<Tag, vec::ScalableTag<float32_t, 0>> &&
        SupportedElement && std::same_as<ScaleElement, InElement> &&
        std::same_as<BiasElement, InElement> &&
        std::same_as<OutElement, InElement> &&
        std::same_as<typename InSpec::TransformType, tensor::NoTransform> &&
        std::same_as<typename ScaleSpec::TransformType, tensor::NoTransform> &&
        std::same_as<typename BiasSpec::TransformType, tensor::NoTransform> &&
        std::same_as<typename OutSpec::TransformType, tensor::NoTransform>;
    using RowRecipe = std::conditional_t<
        UseSVE16, details::SVE16RowRecipe<InElement>,
        details::GenericRowRecipe>;
#else
    using RowRecipe = details::GenericRowRecipe;
#endif

    // The row pipeline lives in its own frame; see the attribute rationale
    // on `run_rows`. The call happens once per operator() invocation,
    // never per row.
    run_rows<RowRecipe>(workspace, in, scale, bias, out);
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
  // The row stride is decided from the layout's meta type: anything not
  // provably unit is treated as strided. Parameters are rank-1 inputs whose
  // direct-forced plan always resolves without the operand-scoping
  // machinery, so they bind with plain `tensor::bind`; the recipe decides
  // whether that happens once before the loop or inside each row frame.
  //
  // The frame itself encodes a measured per-compiler preference: GCC folds
  // the whole row nest into the caller when inlineable, and the hot loops
  // inherit the caller's layout, costing 2-6% on large prefill shapes;
  // BiSheng instead compiles the outlined nest 25% slower on rank-3 medium
  // fp16. x86/GCC keeps the outline, SVE keeps full inlining.
  template <typename Recipe, typename InSpec, typename ScaleSpec,
            typename BiasSpec, typename OutSpec>
#if defined(ARCH_X86_FAMILY)
  VECOPS_NOINLINE
#else
  VECOPS_ALWAYS_INLINE
#endif
  void run_rows(
      kernel::WorkspaceView& workspace, const InSpec& in,
      const ScaleSpec& scale, const BiasSpec& bias,
      const OutSpec& out) const {
    constexpr int PrefixRank = InSpec::InputTensor::Ndim - 1;
    using RowStrideMeta = typename tensor::details::MetaElement<
        PrefixRank, typename InSpec::InputLayout::Strides>::type;
    constexpr bool RowUnitStride =
        tensor::details::is_definitely_one_meta_v<RowStrideMeta>;
    if constexpr (RowUnitStride && Recipe::BindParamsPerRow) {
      kernel::loop::for_each_dims<PrefixRank>(
          [this, &workspace, &scale, &bias](const auto& in_row,
                                            const auto& out_row)
              VECOPS_INLINE_LAMBDA {
            auto gamma = tensor::bind(
                scale, typename Recipe::ParamPolicy{}, workspace);
            auto beta = tensor::bind(
                bias, typename Recipe::ParamPolicy{}, workspace);
            bind_row<tensor::AccessPlan::direct, Recipe>(
                workspace, in_row, gamma, beta, out_row);
          },
          in, out);
    } else {
      auto gamma = tensor::bind(
          scale, typename Recipe::ParamPolicy{}, workspace);
      auto beta = tensor::bind(
          bias, typename Recipe::ParamPolicy{}, workspace);
      if constexpr (RowUnitStride) {
        kernel::loop::for_each_dims<PrefixRank>(
            [this, &workspace, &gamma, &beta](const auto& in_row,
                                              const auto& out_row)
                VECOPS_INLINE_LAMBDA {
              bind_row<tensor::AccessPlan::direct, Recipe>(
                  workspace, in_row, gamma, beta, out_row);
            },
            in, out);
      } else {
        kernel::loop::for_each_dims<PrefixRank>(
            [this, &workspace, &gamma, &beta](const auto& in_row,
                                              const auto& out_row)
                VECOPS_INLINE_LAMBDA {
              bind_row<tensor::AccessPlan::automatic, Recipe>(
                  workspace, in_row, gamma, beta, out_row);
            },
            in, out);
      }
    }
  }

  template <tensor::AccessPlan InPlan, typename Recipe, typename InSpec,
            typename ScaleAccess, typename BiasAccess, typename OutSpec>
  VECOPS_ALWAYS_INLINE void bind_row(
      kernel::WorkspaceView& workspace, const InSpec& in,
      ScaleAccess& gamma, BiasAccess& beta, const OutSpec& out) const {
    static_assert(InSpec::InputTensor::Ndim == 1);
    using InPolicy = typename Recipe::template InputPolicy<InPlan, 2>;
    using OutPolicy = typename Recipe::OutputPolicy;
    if constexpr (InPlan == tensor::AccessPlan::direct) {
      auto x = tensor::bind(in, InPolicy{}, workspace);
      auto y = tensor::bind(out, OutPolicy{}, workspace);
      run_row<Recipe>(in, x, gamma, beta, y);
    } else {
      kernel::with_operands(
          workspace, tensor::operand(in, InPolicy{}),
          tensor::operand(out, OutPolicy{}),
          [this, &in, &gamma, &beta](auto& x, auto& y)
              VECOPS_INLINE_LAMBDA {
            run_row<Recipe>(in, x, gamma, beta, y);
          });
    }
  }

  template <typename Recipe,
            typename InSpec,
            typename X, typename Gamma, typename Beta, typename Y>
  VECOPS_ALWAYS_INLINE void run_row(
      const InSpec& in, X& x, Gamma& gamma, Beta& beta, Y& y) const {
    static_assert(InSpec::InputTensor::Ndim == 1);

    Tag base_tag{};
    const nint_t n = in.input_layout().shape()[0];

    ComputeType sum_value{};
    ComputeType sum_sq_value{};
    auto accumulate_block = [&, n](
                                auto block_tag, nint_t col, auto active,
                                auto& sum, auto& sum_sq)
        VECOPS_INLINE_LAMBDA {
      auto value = x.load(block_tag, tensor::coord(col), active);
      sum = vec::add(sum, value);
      sum_sq = vec::fmadd(value, value, sum_sq);
    };
    kernel::loop::fold<Recipe::FullFactor, Recipe::TailFactor,
                       Recipe::TailPolicy>(
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
    const ComputeType center = Recipe::FusedShift ? -mean * rstd : mean;

    auto write_block = [&, n](
                           auto block_tag, nint_t col, auto active,
                           const auto& center_v, const auto& rstd_v)
        VECOPS_INLINE_LAMBDA {
      const auto position = tensor::coord(col);
      auto value = x.load(block_tag, position, active);
      auto scale_value = gamma.load(block_tag, position, active);
      auto bias_value = beta.load(block_tag, position, active);
      if constexpr (Recipe::FusedShift) {
        value = vec::fmadd(value, rstd_v, center_v);
      } else {
        value = vec::mul(vec::sub(value, center_v), rstd_v);
      }
      y.store(
          block_tag, position,
          vec::fmadd(value, scale_value, bias_value), active);
    };
    kernel::loop::fold<Recipe::FullFactor, Recipe::TailFactor>(
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
