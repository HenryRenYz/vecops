#ifndef VECOPS_OPS_LAYERNORM_H
#define VECOPS_OPS_LAYERNORM_H

#include <algorithm>
#include <cmath>
#include <type_traits>
#include <utility>

#include "vecops/Assertion.h"
#include "vecops/CoreTypes.h"
#include "vecops/execution/ExecutionSession.h"
#include "vecops/kernel/Loop.h"
#include "vecops/kernel/Workspace.h"
#include "vecops/tensor/DataAccess.h"
#include "vecops/tensor/OptionalOperand.h"
#include "vecops/vec/Vec.h"

/**
 * @file LayerNorm.h
 * @brief Last-dimension LayerNorm with optional affine parameters.
 *
 * Every prefix coordinate defines one independently normalized row. Input,
 * gamma, beta, and output use DataAccess so dtype conversion and transforms
 * remain composable. The operator accepts an ExecutionSession/active scope,
 * caller-owned workspace, or a self-allocating convenience call.
 */

namespace vecops::ops {

template <typename ComputeT = float32_t>
/**
 * @brief LayerNorm compute configuration.
 * @tparam ComputeT Accumulation, mean, variance, and output compute type.
 */
struct LayerNormConfig {
  using ComputeType = ComputeT;
  ComputeT eps = ComputeT(1e-5f);
};

namespace details {

template <typename InLayout, typename OutLayout>
VECOPS_INLINE nint_t validate_layernorm_io_layouts(
    const InLayout& in, const OutLayout& out) {
  static_assert(InLayout::Ndim >= 1);
  static_assert(OutLayout::Ndim == InLayout::Ndim);
  for (int d = 0; d < InLayout::Ndim; ++d) {
    VECOPS_ASSERT(in.shape()[d] == out.shape()[d],
                  "LayerNorm input/output shape mismatch");
  }
  const nint_t n = in.shape()[InLayout::Ndim - 1];
  VECOPS_ASSERT(n > 0, "LayerNorm normalized dimension must be non-empty");
  return n;
}

template <typename ParamSpec>
  requires tensor::is_input_spec_v<ParamSpec>
VECOPS_INLINE void validate_layernorm_param(
    nint_t n, const ParamSpec& param) {
  static_assert(ParamSpec::InputTensor::Ndim == 1);
  VECOPS_ASSERT(param.input_layout().shape()[0] == n,
                "LayerNorm parameter size mismatch");
}

VECOPS_INLINE constexpr void validate_layernorm_param(
    nint_t, tensor::nullopt_t) {}

template <typename Operand>
inline constexpr bool is_input_operand_v =
    tensor::InputOperand<Operand>;

template <typename Operand>
inline constexpr bool is_output_operand_v = tensor::OutputOperand<Operand>;

template <typename Param>
inline constexpr bool is_layernorm_param_operand_v = [] {
  using P = std::remove_cvref_t<Param>;
  if constexpr (tensor::is_nullopt_v<P>) return true;
  else if constexpr (tensor::is_input_spec_v<P>) {
    return P::InputTensor::Ndim == 1;
  } else if constexpr (tensor::is_tensor<P>) {
    return P::Ndim == 1;
  } else return false;
}();

template <typename Param>
inline constexpr bool is_normalized_layernorm_param_v =
    tensor::is_input_spec_v<Param> || tensor::is_nullopt_v<Param>;

template <typename Compute, typename Param>
  requires (!tensor::is_nullopt_v<Param>)
VECOPS_INLINE auto as_layernorm_param(Param&& param) {
  return tensor::as_input_spec<Compute>(std::forward<Param>(param));
}

template <typename Compute>
VECOPS_INLINE constexpr tensor::nullopt_t as_layernorm_param(
    tensor::nullopt_t) {
  return tensor::nullopt;
}

template <typename ParamSpec, typename Policy>
  requires tensor::is_input_spec_v<ParamSpec>
VECOPS_INLINE auto bind_layernorm_param(
    const ParamSpec& spec, Policy policy,
    kernel::WorkspaceView& workspace) {
  return tensor::bind(spec, policy, workspace);
}

template <typename Policy>
VECOPS_INLINE constexpr tensor::nullopt_t bind_layernorm_param(
    tensor::nullopt_t, Policy, kernel::WorkspaceView&) {
  return tensor::nullopt;
}

/** Invoke an unordered region with only operands that physically exist. */
template <typename X, typename Gamma, typename Beta, typename Y, typename Fn>
VECOPS_ALWAYS_INLINE decltype(auto) with_layernorm_accesses(
    X& x, Gamma& gamma, Beta& beta, Y& y, Fn&& fn) {
  constexpr bool HasGamma = !tensor::is_nullopt_v<Gamma>;
  constexpr bool HasBeta = !tensor::is_nullopt_v<Beta>;
  if constexpr (HasGamma && HasBeta) {
    return tensor::with_unordered_access(
        x, gamma, beta, y, std::forward<Fn>(fn));
  } else if constexpr (HasGamma) {
    return tensor::with_unordered_access(
        x, gamma, y,
        [&](auto ux, auto ugamma, auto uy) -> decltype(auto) {
          return std::forward<Fn>(fn)(
              ux, ugamma, tensor::nullopt, uy);
        });
  } else if constexpr (HasBeta) {
    return tensor::with_unordered_access(
        x, beta, y,
        [&](auto ux, auto ubeta, auto uy) -> decltype(auto) {
          return std::forward<Fn>(fn)(
              ux, tensor::nullopt, ubeta, uy);
        });
  } else {
    return tensor::with_unordered_access(
        x, y,
        [&](auto ux, auto uy) -> decltype(auto) {
          return std::forward<Fn>(fn)(
              ux, tensor::nullopt, tensor::nullopt, uy);
        });
  }
}

/** @brief Per-row loop tuning. */
template <int Full, int Tail, bool Fused,
          kernel::loop::TailCarryPolicy Carry>
struct RowRecipe {
  static constexpr int FullFactor = Full;
  static constexpr int TailFactor = Tail;
  static constexpr bool FusedShift = Fused;
  static constexpr kernel::loop::TailCarryPolicy TailPolicy = Carry;
};

using GenericRowRecipe = RowRecipe<
    4, 1, false, kernel::loop::TailCarryPolicy::independent>;

#if defined(CPU_CAPABILITY_SVE) && defined(__ARM_FEATURE_SVE2) && \
    !defined(VECOPS_DISABLE_SVE_16BIT_LAYERNORM_FAST_PATH)
using SVE16RowRecipe = RowRecipe<
    4, 2, true, kernel::loop::TailCarryPolicy::reuse_prefix>;
#endif

} // namespace details

/**
 * @brief Normalize the final dimension and optionally apply gamma and beta.
 * @tparam Config `LayerNormConfig`-compatible type.
 *
 * Input/output shapes must match. Gamma and beta, when present, are rank-one
 * operands whose length equals the final dimension. Variance is computed as
 * `E[x^2] - E[x]^2`, clamped to zero before applying `eps`.
 */
template <typename Config = LayerNormConfig<>>
class LayerNorm {
  using XPolicy = tensor::InputAccessPolicy<
      0, 2, tensor::AccessPlan::automatic_deferred>;
  using ParamPolicy = tensor::InputAccessPolicy<
      0, 1, tensor::AccessPlan::direct>;
  using YPolicy = tensor::OutputAccessPolicy<
      0, tensor::AccessPlan::direct>;

public:
  using ComputeType = typename Config::ComputeType;
  using Tag = vec::ScalableTag<ComputeType, 0>;
  /** Compile-time hardware-mode contract consumed by ExecutionSession. */
  using ResourceRequirements =
      typename execution::details::CurrentBackend::DefaultRequirements;

  const Config config;

  VECOPS_INLINE constexpr explicit LayerNorm(Config cfg = {}) : config(cfg) {}

  template <typename InOperand, typename ScaleOperand, typename BiasOperand,
            typename OutOperand>
    requires (details::is_input_operand_v<InOperand> &&
              details::is_layernorm_param_operand_v<ScaleOperand> &&
              details::is_layernorm_param_operand_v<BiasOperand> &&
              details::is_output_operand_v<OutOperand>)
  /**
   * @brief Return exact DataAccess workspace bytes for four operands.
   * @param in Input Tensor/Spec.
   * @param scale Optional gamma Tensor/Spec or `tensor::nullopt`.
   * @param bias Optional beta Tensor/Spec or `tensor::nullopt`.
   * @param out Output Tensor/Spec with input shape.
   */
  VECOPS_INLINE nint_t required_workspace(
      const InOperand& in, const ScaleOperand& scale,
      const BiasOperand& bias, const OutOperand& out) const {
    if constexpr (
        tensor::is_input_spec_v<InOperand> &&
        details::is_normalized_layernorm_param_v<ScaleOperand> &&
        details::is_normalized_layernorm_param_v<BiasOperand> &&
        tensor::is_output_spec_v<OutOperand>) {
      return required_workspace_specs(in, scale, bias, out);
    } else {
      auto in_spec = tensor::as_input_spec<ComputeType>(in);
      auto scale_spec = details::as_layernorm_param<ComputeType>(scale);
      auto bias_spec = details::as_layernorm_param<ComputeType>(bias);
      auto out_spec = tensor::as_output_spec<ComputeType>(out);
      return required_workspace_specs(
          in_spec, scale_spec, bias_spec, out_spec);
    }
  }

  template <typename InOperand, typename OutOperand>
    requires (details::is_input_operand_v<InOperand> &&
              details::is_output_operand_v<OutOperand>)
  /** Workspace query for LayerNorm without affine parameters. */
  VECOPS_INLINE nint_t required_workspace(
      const InOperand& in, const OutOperand& out) const {
    return required_workspace(
        in, tensor::nullopt, tensor::nullopt, out);
  }

  template <typename InOperand, typename ScaleOperand, typename OutOperand>
    requires (details::is_input_operand_v<InOperand> &&
              details::is_layernorm_param_operand_v<ScaleOperand> &&
              !tensor::is_nullopt_v<ScaleOperand> &&
              details::is_output_operand_v<OutOperand>)
  /** Workspace query for LayerNorm with gamma and no beta. */
  VECOPS_INLINE nint_t required_workspace(
      const InOperand& in, const ScaleOperand& scale,
      const OutOperand& out) const {
    return required_workspace(in, scale, tensor::nullopt, out);
  }

  template <typename InOperand, typename ScaleOperand, typename BiasOperand,
            typename OutOperand>
    requires (details::is_input_operand_v<InOperand> &&
              details::is_layernorm_param_operand_v<ScaleOperand> &&
              details::is_layernorm_param_operand_v<BiasOperand> &&
              details::is_output_operand_v<OutOperand>)
  /**
   * @brief Execute with caller-owned workspace and optional affine operands.
   * @param workspace Scratch storage at least `required_workspace()` bytes.
   */
  VECOPS_INLINE void operator()(
      kernel::WorkspaceView& workspace, const InOperand& in,
      const ScaleOperand& scale, const BiasOperand& bias,
      const OutOperand& out) const {
    if constexpr (
        tensor::is_input_spec_v<InOperand> &&
        details::is_normalized_layernorm_param_v<ScaleOperand> &&
        details::is_normalized_layernorm_param_v<BiasOperand> &&
        tensor::is_output_spec_v<OutOperand>) {
      execute_specs(workspace, in, scale, bias, out);
    } else {
      auto in_spec = tensor::as_input_spec<ComputeType>(in);
      auto scale_spec = details::as_layernorm_param<ComputeType>(scale);
      auto bias_spec = details::as_layernorm_param<ComputeType>(bias);
      auto out_spec = tensor::as_output_spec<ComputeType>(out);
      execute_specs(workspace, in_spec, scale_spec, bias_spec, out_spec);
    }
  }

  template <typename InOperand, typename OutOperand>
    requires (details::is_input_operand_v<InOperand> &&
              details::is_output_operand_v<OutOperand>)
  /** Caller-workspace overload without affine parameters. */
  VECOPS_INLINE void operator()(
      kernel::WorkspaceView& workspace, const InOperand& in,
      const OutOperand& out) const {
    (*this)(workspace, in, tensor::nullopt, tensor::nullopt, out);
  }

  template <typename InOperand, typename ScaleOperand, typename OutOperand>
    requires (details::is_input_operand_v<InOperand> &&
              details::is_layernorm_param_operand_v<ScaleOperand> &&
              !tensor::is_nullopt_v<ScaleOperand> &&
              details::is_output_operand_v<OutOperand>)
  /** Caller-workspace overload with gamma and no beta. */
  VECOPS_INLINE void operator()(
      kernel::WorkspaceView& workspace, const InOperand& in,
      const ScaleOperand& scale, const OutOperand& out) const {
    (*this)(workspace, in, scale, tensor::nullopt, out);
  }

  template <execution::ExecutionScope Scope,
            typename InOperand, typename ScaleOperand, typename BiasOperand,
            typename OutOperand>
    requires (details::is_input_operand_v<InOperand> &&
              details::is_layernorm_param_operand_v<ScaleOperand> &&
              details::is_layernorm_param_operand_v<BiasOperand> &&
              details::is_output_operand_v<OutOperand>)
  /**
   * @brief Execute through an ExecutionSession or active enclosing scope.
   * @param scope Scope providing resources and worker-local workspace.
   *
   * Nested `with_resources` is compile-time-elided when an outer operator region
   * already satisfies LayerNorm's requirements.
   */
  VECOPS_INLINE void operator()(
      Scope& scope, const InOperand& in, const ScaleOperand& scale,
      const BiasOperand& bias, const OutOperand& out) const {
    scope.with_resources(
        *this, [&](auto& active) VECOPS_INLINE_LAMBDA {
          (*this)(active.workspace_view(), in, scale, bias, out);
        });
  }

  template <execution::ExecutionScope Scope,
            typename InOperand, typename OutOperand>
    requires (details::is_input_operand_v<InOperand> &&
              details::is_output_operand_v<OutOperand>)
  /** Execution-scope overload without affine parameters. */
  VECOPS_INLINE void operator()(
      Scope& scope, const InOperand& in, const OutOperand& out) const {
    (*this)(scope, in, tensor::nullopt, tensor::nullopt, out);
  }

  template <execution::ExecutionScope Scope,
            typename InOperand, typename ScaleOperand, typename OutOperand>
    requires (details::is_input_operand_v<InOperand> &&
              details::is_layernorm_param_operand_v<ScaleOperand> &&
              !tensor::is_nullopt_v<ScaleOperand> &&
              details::is_output_operand_v<OutOperand>)
  /** Execution-scope overload with gamma and no beta. */
  VECOPS_INLINE void operator()(
      Scope& scope, const InOperand& in, const ScaleOperand& scale,
      const OutOperand& out) const {
    (*this)(scope, in, scale, tensor::nullopt, out);
  }

  template <typename InOperand, typename ScaleOperand, typename BiasOperand,
            typename OutOperand>
    requires (details::is_input_operand_v<InOperand> &&
              details::is_layernorm_param_operand_v<ScaleOperand> &&
              details::is_layernorm_param_operand_v<BiasOperand> &&
              details::is_output_operand_v<OutOperand>)
  /** Allocate exact workspace, create an ExecutionSession, and execute once. */
  VECOPS_INLINE void operator()(
      const InOperand& in, const ScaleOperand& scale,
      const BiasOperand& bias, const OutOperand& out) const {
    kernel::Workspace storage(required_workspace(in, scale, bias, out));
    auto workspace = storage.view();
    ExecutionSession execution{workspace};
    (*this)(execution, in, scale, bias, out);
  }

  template <typename InOperand, typename OutOperand>
    requires (details::is_input_operand_v<InOperand> &&
              details::is_output_operand_v<OutOperand>)
  VECOPS_INLINE void operator()(
      const InOperand& in, const OutOperand& out) const {
    (*this)(in, tensor::nullopt, tensor::nullopt, out);
  }

  template <typename InOperand, typename ScaleOperand, typename OutOperand>
    requires (details::is_input_operand_v<InOperand> &&
              details::is_layernorm_param_operand_v<ScaleOperand> &&
              !tensor::is_nullopt_v<ScaleOperand> &&
              details::is_output_operand_v<OutOperand>)
  VECOPS_INLINE void operator()(
      const InOperand& in, const ScaleOperand& scale,
      const OutOperand& out) const {
    (*this)(in, scale, tensor::nullopt, out);
  }

  /**
   * @brief Execute one already-bound row without replanning or rebinding.
   * @param x Bound rank-one input.
   * @param gamma Optional bound rank-one scale.
   * @param beta Optional bound rank-one bias.
   * @param y Bound rank-one committable output.
   */
  template <typename X, typename Gamma, typename Beta, typename Y>
    requires (tensor::BoundInputAccess<X> &&
              (tensor::BoundInputAccess<Gamma> ||
               tensor::is_nullopt_v<Gamma>) &&
              (tensor::BoundInputAccess<Beta> ||
               tensor::is_nullopt_v<Beta>) &&
              tensor::CommittableBoundOutputAccess<Y>)
  VECOPS_ALWAYS_INLINE void run_bound(
      X& x, Gamma& gamma, Beta& beta, Y& y) const {
    static_assert(X::Rank == 1 && Y::Rank == 1);
    if constexpr (!tensor::is_nullopt_v<Gamma>) {
      static_assert(Gamma::Rank == 1);
    }
    if constexpr (!tensor::is_nullopt_v<Beta>) {
      static_assert(Beta::Rank == 1);
    }
    const nint_t n = tensor::logical_layout(x).shape()[0];
#if defined(CPU_CAPABILITY_SVE) && defined(__ARM_FEATURE_SVE2) && \
    !defined(VECOPS_DISABLE_SVE_16BIT_LAYERNORM_FAST_PATH)
    using InElement = std::remove_const_t<typename X::MemoryElement>;
    constexpr bool UseSVE16 =
        std::same_as<ComputeType, float32_t> &&
        (IsFloat16V<InElement> || IsBfloat16V<InElement>);
    using Recipe = std::conditional_t<
        UseSVE16, details::SVE16RowRecipe,
        details::GenericRowRecipe>;
#else
    using Recipe = details::GenericRowRecipe;
#endif
    run_row<Recipe>(n, x, gamma, beta, y);
  }

private:
  template <typename InSpec, typename ScaleSpec, typename BiasSpec,
            typename OutSpec>
  VECOPS_INLINE nint_t required_workspace_specs(
      const InSpec& in, const ScaleSpec& scale, const BiasSpec& bias,
      const OutSpec& out) const {
    const nint_t n = details::validate_layernorm_io_layouts(
        in.input_layout(), out.output_layout());
    details::validate_layernorm_param(n, scale);
    details::validate_layernorm_param(n, bias);
    return tensor::required_workspace(
        tensor::take_trailing<1>(in), XPolicy{});
  }

  template <typename InSpec, typename ScaleSpec, typename BiasSpec,
            typename OutSpec>
  VECOPS_INLINE void execute_specs(
      kernel::WorkspaceView& workspace, const InSpec& in,
      const ScaleSpec& scale, const BiasSpec& bias,
      const OutSpec& out) const {
    const nint_t n = details::validate_layernorm_io_layouts(
        in.input_layout(), out.output_layout());
    details::validate_layernorm_param(n, scale);
    details::validate_layernorm_param(n, bias);
#if defined(CPU_CAPABILITY_SVE) && defined(__ARM_FEATURE_SVE2) && \
    !defined(VECOPS_DISABLE_SVE_16BIT_LAYERNORM_FAST_PATH)
    using InElement = std::remove_const_t<typename InSpec::MemoryElement>;
    constexpr bool UseSVE16 =
        std::same_as<ComputeType, float32_t> &&
        (IsFloat16V<InElement> || IsBfloat16V<InElement>);
    using Recipe = std::conditional_t<
        UseSVE16, details::SVE16RowRecipe,
        details::GenericRowRecipe>;
#else
    using Recipe = details::GenericRowRecipe;
#endif
    run_rows<Recipe>(workspace, in, scale, bias, out);
  }

  // COMPILER TUNING: GCC/Clang x86 generate a smaller and faster outer loop
  // when the row nest is outlined, while SVE must inline it so scalable-vector
  // loop state and DataAccess projections optimize together. Re-check both code
  // size and row-loop assembly before making this annotation uniform.
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
    auto gamma = details::bind_layernorm_param(
        scale, ParamPolicy{}, workspace);
    auto beta = details::bind_layernorm_param(
        bias, ParamPolicy{}, workspace);
    kernel::loop::for_each_dims<PrefixRank>(
        [this, &workspace, &gamma, &beta](const auto& in_row,
                                          const auto& out_row)
            VECOPS_INLINE_LAMBDA {
          bind_row<Recipe>(workspace, in_row, gamma, beta, out_row);
        },
        in, out);
  }

  template <typename Recipe, typename InSpec, typename ScaleAccess,
            typename BiasAccess, typename OutSpec>
  VECOPS_ALWAYS_INLINE void bind_row(
      kernel::WorkspaceView& workspace, const InSpec& in,
      ScaleAccess& gamma, BiasAccess& beta, const OutSpec& out) const {
    static_assert(InSpec::InputTensor::Ndim == 1);
    kernel::with_operands(
        workspace, tensor::operand(in, XPolicy{}),
        tensor::operand(out, YPolicy{}),
        [this, &in, &gamma, &beta](auto& x, auto& y)
            VECOPS_INLINE_LAMBDA {
          run_row<Recipe>(in.input_layout().shape()[0], x, gamma, beta, y);
        });
  }

  template <typename Recipe, typename X, typename Gamma,
            typename Beta, typename Y>
  VECOPS_ALWAYS_INLINE void run_row(
      nint_t n, X& x, Gamma& gamma, Beta& beta, Y& y) const {
    Tag base_tag{};

    ComputeType sum_value{};
    ComputeType sum_sq_value{};
    tensor::with_unordered_access(x, [&](auto ux) VECOPS_INLINE_LAMBDA {
      kernel::loop::fold<Recipe::FullFactor, Recipe::TailFactor, Recipe::TailPolicy>(
          base_tag, n,
          [&, n](auto block_tag, nint_t col, auto active,
                 auto& sum, auto& sum_sq) VECOPS_INLINE_LAMBDA {
            auto value = ux.load(
                block_tag, tensor::coord(col), active,
                tensor::materialize::populate);
            sum = vec::add(sum, value);
            sum_sq = vec::fmadd(value, value, sum_sq);
          },
          kernel::loop::reduce_add(sum_value),
          kernel::loop::reduce_add(sum_sq_value));
    });

    const ComputeType inv_n =
        ComputeType(1) / static_cast<ComputeType>(n);
    const ComputeType mean = sum_value * inv_n;
    const ComputeType variance =
        std::max(sum_sq_value * inv_n - mean * mean, ComputeType(0));
    const ComputeType rstd =
        ComputeType(1) / std::sqrt(variance + config.eps);
    const ComputeType center = Recipe::FusedShift ? -mean * rstd : mean;

    constexpr bool HasGamma = !tensor::is_nullopt_v<Gamma>;
    constexpr bool HasBeta = !tensor::is_nullopt_v<Beta>;
    auto write =
        [&](auto ux, auto ugamma, auto ubeta, auto uy) VECOPS_INLINE_LAMBDA {
          kernel::loop::fold<Recipe::FullFactor, Recipe::TailFactor>(
              base_tag, n,
              [&, n](auto block_tag, nint_t col, auto active,
                     const auto& center_v,
                     const auto& rstd_v) VECOPS_INLINE_LAMBDA {
                const auto position = tensor::coord(col);
                auto value = ux.load(block_tag, position, active);
                if constexpr (Recipe::FusedShift) {
                  value = vec::fmadd(value, rstd_v, center_v);
                } else {
                  value = vec::mul(vec::sub(value, center_v), rstd_v);
                }
                if constexpr (HasGamma && HasBeta) {
                  const auto scale_value =
                      ugamma.load(block_tag, position, active);
                  const auto bias_value =
                      ubeta.load(block_tag, position, active);
                  value = vec::fmadd(value, scale_value, bias_value);
                } else if constexpr (HasGamma) {
                  const auto scale_value =
                      ugamma.load(block_tag, position, active);
                  value = vec::mul(value, scale_value);
                } else if constexpr (HasBeta) {
                  const auto bias_value =
                      ubeta.load(block_tag, position, active);
                  value = vec::add(value, bias_value);
                }
                uy.store(block_tag, position, value, active);
              },
              kernel::loop::invariant(center),
              kernel::loop::invariant(rstd));
          uy.commit();
        };
    if constexpr (HasGamma && HasBeta) {
      tensor::with_unordered_access(
          x, gamma, beta, y, std::move(write));
    } else {
      details::with_layernorm_accesses(
          x, gamma, beta, y, std::move(write));
    }
  }
};

template <typename Config = LayerNormConfig<>>
/** Construct a LayerNorm operator from its configuration value. */
VECOPS_INLINE constexpr auto layer_norm(Config config = {}) {
  return LayerNorm<Config>{config};
}

} // namespace vecops::ops

#endif // VECOPS_OPS_LAYERNORM_H
