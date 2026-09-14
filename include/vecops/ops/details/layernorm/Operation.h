#ifndef VECOPS_OPS_DETAILS_LAYERNORM_OPERATION_H
#define VECOPS_OPS_DETAILS_LAYERNORM_OPERATION_H

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <string_view>
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
 * @file vecops/ops/details/layernorm/Operation.h
 * @brief Last-dimension LayerNorm implementation with optional affine
 *        parameters.
 *
 * Every prefix coordinate defines one independently normalized row. Input,
 * gamma, beta, and output use DataAccess so dtype conversion and transforms
 * remain composable. The operator accepts an ExecutionSession/active scope,
 * caller-owned workspace, or a self-allocating convenience call.
 *
 * ## Row algorithm
 *
 * One row is processed in two sweeps:
 *
 * 1. **Reduction sweep**: a single `kernel::loop::fold` with two accumulators
 *    (sum, sum-of-squares) computes both moments in one pass over x.
 * 2. **Write sweep**: a fold over x re-loads the values, applies
 *    `(x - mean) * rstd` (fused into one FMA when the recipe has
 *    `FusedShift`), multiplies by gamma / adds beta when present, and stores.
 *
 * Variance comes from the single-pass `E[x^2] - E[x]^2` estimator, clamped
 * at zero because floating-point rounding can push it slightly negative.
 *
 * ## Recipes
 *
 * `RowRecipe` tunes the fold unrolling, the tail blocking, the shift
 * fusion, and the tail carry policy. `SVE16RowRecipe` is the SVE2 16-bit
 * fast path for fp16/bf16 inputs computed in fp32 (see its comment); it can
 * be disabled with `VECOPS_DISABLE_SVE_16BIT_LAYERNORM_FAST_PATH`.
 */

namespace vecops::ops {

template <typename ComputeT = float32_t>
/**
 * @brief LayerNorm compute configuration.
 * @tparam ComputeT Accumulation, mean, variance, and output compute type.
 */
struct LayerNormConfig {
  using ComputeType = ComputeT;
  /// Variance stabilizer added before the reciprocal-sqrt; guards rows
  /// with zero variance (constant rows) from dividing by zero.
  ComputeT eps = ComputeT(1e-5f);
};

namespace layernorm_details {

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

template <tensor::InputSpecLike ParamSpec>
VECOPS_INLINE void validate_layernorm_param(
    nint_t n, const ParamSpec& param) {
  static_assert(ParamSpec::InputTensor::Ndim == 1);
  VECOPS_ASSERT(param.input_layout().shape()[0] == n,
                "LayerNorm parameter size mismatch");
}

VECOPS_INLINE constexpr void validate_layernorm_param(
    nint_t, tensor::nullopt_t) {}

template <typename Param>
inline constexpr bool is_layernorm_param_operand_v = [] {
  using P = std::remove_cvref_t<Param>;
  if constexpr (tensor::is_nullopt_v<P>) return true;
  else if constexpr (tensor::is_input_spec_v<P>) {
    return P::InputTensor::Ndim == 1;
  } else if constexpr (tensor::is_tensor_v<P>) {
    return P::Ndim == 1;
  } else return false;
}();

template <typename Param>
concept LayerNormParamOperand = is_layernorm_param_operand_v<Param>;

template <typename Param>
/** True when the parameter is already a rank-1 input Spec (or nullopt), so
 *  the public entry points can skip Tensor-to-Spec normalization for it. */
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

/** Storage-less affine operand accepted by the prepared LayerNorm surface. */
template <typename Param>
inline constexpr bool is_unbound_layernorm_param_pattern_v = [] {
  using P = std::remove_cvref_t<Param>;
  return tensor::is_nullopt_v<P> || tensor::is_unbound_input_spec_v<P>;
}();

template <typename Param>
concept UnboundLayerNormParamPattern =
    is_unbound_layernorm_param_pattern_v<Param>;

/** Execution-time affine operand accepted by the prepared LayerNorm surface. */
template <typename Param>
inline constexpr bool is_bound_layernorm_param_operand_v = [] {
  using P = std::remove_cvref_t<Param>;
  return tensor::is_nullopt_v<P> ||
      (LayerNormParamOperand<P> && tensor::is_bound_tensor_view_v<P>);
}();

template <typename Param>
concept BoundLayerNormParamOperand =
    is_bound_layernorm_param_operand_v<Param>;

template <tensor::UnboundInputSpecLike Pattern,
          tensor::BoundTensorView Actual>
VECOPS_INLINE auto rebind_layernorm_param(
    const Pattern& pattern, Actual&& actual) {
  return tensor::rebind(pattern, std::forward<Actual>(actual));
}

VECOPS_INLINE constexpr tensor::nullopt_t rebind_layernorm_param(
    tensor::nullopt_t, tensor::nullopt_t) {
  return tensor::nullopt;
}

template <tensor::InputSpecLike ParamSpec, typename Policy>
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

/** @brief Per-row loop tuning.
 *  @tparam Full  Fold unroll factor for both row sweeps.
 *  @tparam Tail  Vector blocking of the (unmasked) tail block.
 *  @tparam Fused When true, the write sweep evaluates `(x - mean) * rstd`
 *                as `fma(x, rstd, -mean * rstd)` — one rounding instead of
 *                two, at the cost of a per-recipe center meaning (see
 *                `run_row`).
 *  @tparam Carry Tail carry policy of the reduction fold. */
template <int Full, int Tail, bool Fused,
          kernel::loop::TailCarryPolicy Carry>
struct RowRecipe {
  static constexpr int FullFactor = Full;
  static constexpr int TailFactor = Tail;
  static constexpr bool FusedShift = Fused;
  static constexpr kernel::loop::TailCarryPolicy TailPolicy = Carry;
};

/// Default recipe for all inputs except the SVE2 16-bit fast path.
using GenericRowRecipe = RowRecipe<
    4, 1, false, kernel::loop::TailCarryPolicy::independent>;

#if defined(CPU_CAPABILITY_SVE) && defined(__ARM_FEATURE_SVE2) && \
    !defined(VECOPS_DISABLE_SVE_16BIT_LAYERNORM_FAST_PATH)
/// SVE2 fast path for fp16/bf16 rows computed in fp32: the fused shift
/// shortens the conversion-dominated dependency chain, and reusing the
/// full-width carry in the tail avoids materializing a separate narrow
/// tail accumulator.
using SVE16RowRecipe = RowRecipe<
    4, 2, true, kernel::loop::TailCarryPolicy::reuse_prefix>;
#endif

} // namespace layernorm_details

/**
 * @brief Normalize the final dimension and optionally apply gamma and beta.
 * @tparam Config `LayerNormConfig`-compatible type.
 *
 * Input/output shapes must match. Gamma and beta, when present, are rank-one
 * operands whose length equals the final dimension. Variance is computed as
 * `E[x^2] - E[x]^2`, clamped to zero before applying `eps`.
 *
 * @warning The single-pass variance estimator loses precision through
 *          catastrophic cancellation when `E[x]^2` dominates `E[x^2]`
 *          (large mean, tiny variance). Switch to a two-pass estimator
 *          before relying on results for such data distributions.
 */
template <typename Config = LayerNormConfig<>>
class LayerNorm {
  // Row-level access planning, vector axis 0 = normalized dimension.
  // x needs two passes (moments sweep, then write sweep) with deferred
  // materialization; gamma/beta/y are direct unit-stride rank-1 accesses.
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
      typename execution::details::current_backend_t::DefaultRequirements;

  const Config config;

  VECOPS_INLINE constexpr explicit LayerNorm(Config cfg = {}) : config(cfg) {}

  /**
   * LayerNorm whose operand contracts and private lane scratch were prepared
   * outside the parallel region, while execution-time addresses remain free.
   *
   * The stored Specs are storage-less patterns. Each call validates and
   * rebinds them to the current tile, then recreates the legacy scratch cursor
   * and resource session internally over the calling lane's private tensor.
   */
  template <nint_t Parallelism, typename InPattern,
            typename ScalePattern, typename BiasPattern,
            typename OutPattern, typename WorkerScratch>
  class PatternPrepared {
  public:
    static_assert(Parallelism > 0,
                  "prepared LayerNorm parallelism must be positive");
    using ResourceRequirements = typename LayerNorm::ResourceRequirements;

    VECOPS_INLINE PatternPrepared(
        Config config, InPattern in, ScalePattern scale,
        BiasPattern bias, OutPattern out,
        WorkerScratch worker_scratch, nint_t workspace_bytes)
      : config_(std::move(config))
      , in_(std::move(in))
      , scale_(std::move(scale))
      , bias_(std::move(bias))
      , out_(std::move(out))
      , worker_scratch_(std::move(worker_scratch))
      , workspace_bytes_(workspace_bytes) {
    }

    /** Execute an affine LayerNorm on the scratch replica owned by `task`. */
    template <tensor::InputOperand In,
              layernorm_details::BoundLayerNormParamOperand Scale,
              layernorm_details::BoundLayerNormParamOperand Bias,
              tensor::OutputOperand Out>
      requires (tensor::is_bound_tensor_view_v<In> &&
                tensor::is_bound_tensor_view_v<Out> &&
                (tensor::is_nullopt_v<ScalePattern> ==
                 tensor::is_nullopt_v<Scale>) &&
                (tensor::is_nullopt_v<BiasPattern> ==
                 tensor::is_nullopt_v<Bias>))
    VECOPS_INLINE void operator()(
        execution::TaskContext<Parallelism> task,
        In&& in, Scale&& scale, Bias&& bias, Out&& out) const {
      run(task,
          tensor::rebind(in_, std::forward<In>(in)),
          layernorm_details::rebind_layernorm_param(
              scale_, std::forward<Scale>(scale)),
          layernorm_details::rebind_layernorm_param(
              bias_, std::forward<Bias>(bias)),
          tensor::rebind(out_, std::forward<Out>(out)));
    }

    /** Execute a prepared LayerNorm without affine parameters. */
    template <tensor::InputOperand In, tensor::OutputOperand Out>
      requires (tensor::is_nullopt_v<ScalePattern> &&
                tensor::is_nullopt_v<BiasPattern> &&
                tensor::is_bound_tensor_view_v<In> &&
                tensor::is_bound_tensor_view_v<Out>)
    VECOPS_INLINE void operator()(
        execution::TaskContext<Parallelism> task,
        In&& in, Out&& out) const {
      (*this)(task, std::forward<In>(in), tensor::nullopt,
              tensor::nullopt, std::forward<Out>(out));
    }

    /** Execute a prepared gamma-only LayerNorm. */
    template <tensor::InputOperand In,
              layernorm_details::BoundLayerNormParamOperand Scale,
              tensor::OutputOperand Out>
      requires (!tensor::is_nullopt_v<ScalePattern> &&
                tensor::is_nullopt_v<BiasPattern> &&
                !tensor::is_nullopt_v<Scale> &&
                tensor::is_bound_tensor_view_v<In> &&
                tensor::is_bound_tensor_view_v<Out>)
    VECOPS_INLINE void operator()(
        execution::TaskContext<Parallelism> task,
        In&& in, Scale&& scale, Out&& out) const {
      (*this)(task, std::forward<In>(in), std::forward<Scale>(scale),
              tensor::nullopt, std::forward<Out>(out));
    }

  private:
    template <typename In, typename Scale, typename Bias, typename Out>
    VECOPS_INLINE void run(
        execution::TaskContext<Parallelism> task,
        In&& in, Scale&& scale, Bias&& bias, Out&& out) const {
      auto scratch = task.local(worker_scratch_);
      kernel::WorkspaceView workspace{scratch.data(), workspace_bytes_};
      ExecutionSession execution{workspace};
      LayerNorm{config_}(
          execution, std::forward<In>(in), std::forward<Scale>(scale),
          std::forward<Bias>(bias), std::forward<Out>(out));
    }

    Config config_;
    InPattern in_;
    ScalePattern scale_;
    BiasPattern bias_;
    OutPattern out_;
    WorkerScratch worker_scratch_;
    nint_t workspace_bytes_ = 0;
  };

  /**
   * Prepare affine LayerNorm from storage-less input/output Specs.
   *
   * `site_name` is mandatory so independent operations in one lexical
   * workspace scope receive distinct planner sites. Scratch is represented as
   * one typed byte tensor per logical lane; byte arithmetic is private to the
   * operator migration layer.
   */
  template <nint_t Parallelism, typename WorkspaceAuthority,
            tensor::UnboundInputSpecLike InPattern,
            layernorm_details::UnboundLayerNormParamPattern ScalePattern,
            layernorm_details::UnboundLayerNormParamPattern BiasPattern,
            tensor::UnboundOutputSpecLike OutPattern>
    requires requires(WorkspaceAuthority& authority, nint_t bytes) {
      authority.template worker_tensor<std::byte, Parallelism>(
          std::string_view{}, tensor::make_shape(meta::Any{bytes}));
    }
  VECOPS_INLINE auto prepare(
      WorkspaceAuthority& parent, std::string_view site_name,
      InPattern&& in, ScalePattern&& scale,
      BiasPattern&& bias, OutPattern&& out) const {
    using In = std::remove_cvref_t<InPattern>;
    using Out = std::remove_cvref_t<OutPattern>;
    static_assert(std::same_as<typename In::ComputeType, ComputeType>,
                  "LayerNorm input pattern compute type must match Config");
    static_assert(std::same_as<typename Out::ComputeType, ComputeType>,
                  "LayerNorm output pattern compute type must match Config");

    auto in_pattern = std::forward<InPattern>(in);
    auto scale_pattern = std::forward<ScalePattern>(scale);
    auto bias_pattern = std::forward<BiasPattern>(bias);
    auto out_pattern = std::forward<OutPattern>(out);
    if constexpr (!tensor::is_nullopt_v<decltype(scale_pattern)>) {
      static_assert(
          std::same_as<typename decltype(scale_pattern)::ComputeType,
                       ComputeType>,
          "LayerNorm scale pattern compute type must match Config");
    }
    if constexpr (!tensor::is_nullopt_v<decltype(bias_pattern)>) {
      static_assert(
          std::same_as<typename decltype(bias_pattern)::ComputeType,
                       ComputeType>,
          "LayerNorm bias pattern compute type must match Config");
    }

    const nint_t bytes = required_workspace_specs(
        in_pattern, scale_pattern, bias_pattern, out_pattern);
    auto scratch = parent.template worker_tensor<std::byte, Parallelism>(
        site_name, tensor::make_shape(meta::Any{bytes}));
    return PatternPrepared<
        Parallelism, decltype(in_pattern), decltype(scale_pattern),
        decltype(bias_pattern), decltype(out_pattern), decltype(scratch)>{
          config, std::move(in_pattern), std::move(scale_pattern),
          std::move(bias_pattern), std::move(out_pattern),
          std::move(scratch), bytes};
  }

  /** Prepare LayerNorm without affine parameters. */
  template <nint_t Parallelism, typename WorkspaceAuthority,
            tensor::UnboundInputSpecLike InPattern,
            tensor::UnboundOutputSpecLike OutPattern>
  VECOPS_INLINE auto prepare(
      WorkspaceAuthority& parent, std::string_view site_name,
      InPattern&& in, OutPattern&& out) const {
    return this->template prepare<Parallelism>(
        parent, site_name, std::forward<InPattern>(in), tensor::nullopt,
        tensor::nullopt, std::forward<OutPattern>(out));
  }

  /** Prepare LayerNorm with gamma and no beta. */
  template <nint_t Parallelism, typename WorkspaceAuthority,
            tensor::UnboundInputSpecLike InPattern,
            tensor::UnboundInputSpecLike ScalePattern,
            tensor::UnboundOutputSpecLike OutPattern>
  VECOPS_INLINE auto prepare(
      WorkspaceAuthority& parent, std::string_view site_name,
      InPattern&& in, ScalePattern&& scale, OutPattern&& out) const {
    return this->template prepare<Parallelism>(
        parent, site_name, std::forward<InPattern>(in),
        std::forward<ScalePattern>(scale), tensor::nullopt,
        std::forward<OutPattern>(out));
  }

  template <tensor::InputOperand InOperand,
            layernorm_details::LayerNormParamOperand ScaleOperand,
            layernorm_details::LayerNormParamOperand BiasOperand,
            tensor::OutputOperand OutOperand>
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
        layernorm_details::is_normalized_layernorm_param_v<ScaleOperand> &&
        layernorm_details::is_normalized_layernorm_param_v<BiasOperand> &&
        tensor::is_output_spec_v<OutOperand>) {
      return required_workspace_specs(in, scale, bias, out);
    } else {
      auto in_spec = tensor::as_input_spec<ComputeType>(in);
      auto scale_spec = layernorm_details::as_layernorm_param<ComputeType>(scale);
      auto bias_spec = layernorm_details::as_layernorm_param<ComputeType>(bias);
      auto out_spec = tensor::as_output_spec<ComputeType>(out);
      return required_workspace_specs(
          in_spec, scale_spec, bias_spec, out_spec);
    }
  }

  template <tensor::InputOperand InOperand,
            tensor::OutputOperand OutOperand>
  /** Workspace query for LayerNorm without affine parameters. */
  VECOPS_INLINE nint_t required_workspace(
      const InOperand& in, const OutOperand& out) const {
    return required_workspace(
        in, tensor::nullopt, tensor::nullopt, out);
  }

  template <tensor::InputOperand InOperand,
            layernorm_details::LayerNormParamOperand ScaleOperand,
            tensor::OutputOperand OutOperand>
    requires (!tensor::is_nullopt_v<ScaleOperand>)
  /** Workspace query for LayerNorm with gamma and no beta. */
  VECOPS_INLINE nint_t required_workspace(
      const InOperand& in, const ScaleOperand& scale,
      const OutOperand& out) const {
    return required_workspace(in, scale, tensor::nullopt, out);
  }

  template <tensor::InputOperand InOperand,
            layernorm_details::LayerNormParamOperand ScaleOperand,
            layernorm_details::LayerNormParamOperand BiasOperand,
            tensor::OutputOperand OutOperand>
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
        layernorm_details::is_normalized_layernorm_param_v<ScaleOperand> &&
        layernorm_details::is_normalized_layernorm_param_v<BiasOperand> &&
        tensor::is_output_spec_v<OutOperand>) {
      execute_specs(workspace, in, scale, bias, out);
    } else {
      auto in_spec = tensor::as_input_spec<ComputeType>(in);
      auto scale_spec = layernorm_details::as_layernorm_param<ComputeType>(scale);
      auto bias_spec = layernorm_details::as_layernorm_param<ComputeType>(bias);
      auto out_spec = tensor::as_output_spec<ComputeType>(out);
      execute_specs(workspace, in_spec, scale_spec, bias_spec, out_spec);
    }
  }

  template <tensor::InputOperand InOperand,
            tensor::OutputOperand OutOperand>
  /** Caller-workspace overload without affine parameters. */
  VECOPS_INLINE void operator()(
      kernel::WorkspaceView& workspace, const InOperand& in,
      const OutOperand& out) const {
    (*this)(workspace, in, tensor::nullopt, tensor::nullopt, out);
  }

  template <tensor::InputOperand InOperand,
            layernorm_details::LayerNormParamOperand ScaleOperand,
            tensor::OutputOperand OutOperand>
    requires (!tensor::is_nullopt_v<ScaleOperand>)
  /** Caller-workspace overload with gamma and no beta. */
  VECOPS_INLINE void operator()(
      kernel::WorkspaceView& workspace, const InOperand& in,
      const ScaleOperand& scale, const OutOperand& out) const {
    (*this)(workspace, in, scale, tensor::nullopt, out);
  }

  template <execution::ExecutionScope Scope,
            tensor::InputOperand InOperand,
            layernorm_details::LayerNormParamOperand ScaleOperand,
            layernorm_details::LayerNormParamOperand BiasOperand,
            tensor::OutputOperand OutOperand>
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
            tensor::InputOperand InOperand,
            tensor::OutputOperand OutOperand>
  /** Execution-scope overload without affine parameters. */
  VECOPS_INLINE void operator()(
      Scope& scope, const InOperand& in, const OutOperand& out) const {
    (*this)(scope, in, tensor::nullopt, tensor::nullopt, out);
  }

  template <execution::ExecutionScope Scope,
            tensor::InputOperand InOperand,
            layernorm_details::LayerNormParamOperand ScaleOperand,
            tensor::OutputOperand OutOperand>
    requires (!tensor::is_nullopt_v<ScaleOperand>)
  /** Execution-scope overload with gamma and no beta. */
  VECOPS_INLINE void operator()(
      Scope& scope, const InOperand& in, const ScaleOperand& scale,
      const OutOperand& out) const {
    (*this)(scope, in, scale, tensor::nullopt, out);
  }

  template <tensor::InputOperand InOperand,
            layernorm_details::LayerNormParamOperand ScaleOperand,
            layernorm_details::LayerNormParamOperand BiasOperand,
            tensor::OutputOperand OutOperand>
  /** Allocate exact workspace, create an ExecutionSession, and execute once. */
  VECOPS_INLINE void operator()(
      const InOperand& in, const ScaleOperand& scale,
      const BiasOperand& bias, const OutOperand& out) const {
    kernel::Workspace storage(required_workspace(in, scale, bias, out));
    auto workspace = storage.view();
    ExecutionSession execution{workspace};
    (*this)(execution, in, scale, bias, out);
  }

  template <tensor::InputOperand InOperand,
            tensor::OutputOperand OutOperand>
  VECOPS_INLINE void operator()(
      const InOperand& in, const OutOperand& out) const {
    (*this)(in, tensor::nullopt, tensor::nullopt, out);
  }

  template <tensor::InputOperand InOperand,
            layernorm_details::LayerNormParamOperand ScaleOperand,
            tensor::OutputOperand OutOperand>
    requires (!tensor::is_nullopt_v<ScaleOperand>)
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
  template <tensor::BoundInputAccess X,
            typename Gamma, typename Beta,
            tensor::CommittableBoundOutputAccess Y>
    requires ((tensor::BoundInputAccess<Gamma> ||
               tensor::is_nullopt_v<Gamma>) &&
              (tensor::BoundInputAccess<Beta> ||
               tensor::is_nullopt_v<Beta>))
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
    // Recipe selection for already-bound rows. Keep in sync with the
    // identical #if block in execute_specs().
#if defined(CPU_CAPABILITY_SVE) && defined(__ARM_FEATURE_SVE2) && \
    !defined(VECOPS_DISABLE_SVE_16BIT_LAYERNORM_FAST_PATH)
    using InElement = std::remove_const_t<typename X::MemoryElement>;
    constexpr bool UseSVE16 =
        std::same_as<ComputeType, float32_t> &&
        (is_float16_v<InElement> || is_bfloat16_v<InElement>);
    using Recipe = std::conditional_t<
        UseSVE16, layernorm_details::SVE16RowRecipe,
        layernorm_details::GenericRowRecipe>;
#else
    using Recipe = layernorm_details::GenericRowRecipe;
#endif
    run_row<Recipe>(n, x, gamma, beta, y);
  }

private:
  template <typename InSpec, typename ScaleSpec, typename BiasSpec,
            typename OutSpec>
  VECOPS_INLINE nint_t required_workspace_specs(
      const InSpec& in, const ScaleSpec& scale, const BiasSpec& bias,
      const OutSpec& out) const {
    const nint_t n = layernorm_details::validate_layernorm_io_layouts(
        in.input_layout(), out.output_layout());
    layernorm_details::validate_layernorm_param(n, scale);
    layernorm_details::validate_layernorm_param(n, bias);
    return tensor::required_workspace(
        tensor::take_trailing<1>(in), XPolicy{});
  }

  template <typename InSpec, typename ScaleSpec, typename BiasSpec,
            typename OutSpec>
  VECOPS_INLINE void execute_specs(
      kernel::WorkspaceView& workspace, const InSpec& in,
      const ScaleSpec& scale, const BiasSpec& bias,
      const OutSpec& out) const {
    const nint_t n = layernorm_details::validate_layernorm_io_layouts(
        in.input_layout(), out.output_layout());
    layernorm_details::validate_layernorm_param(n, scale);
    layernorm_details::validate_layernorm_param(n, bias);
    // Recipe selection for Spec-driven rows. Keep in sync with the
    // identical #if block in run_bound().
#if defined(CPU_CAPABILITY_SVE) && defined(__ARM_FEATURE_SVE2) && \
    !defined(VECOPS_DISABLE_SVE_16BIT_LAYERNORM_FAST_PATH)
    using InElement = std::remove_const_t<typename InSpec::MemoryElement>;
    constexpr bool UseSVE16 =
        std::same_as<ComputeType, float32_t> &&
        (is_float16_v<InElement> || is_bfloat16_v<InElement>);
    using Recipe = std::conditional_t<
        UseSVE16, layernorm_details::SVE16RowRecipe,
        layernorm_details::GenericRowRecipe>;
#else
    using Recipe = layernorm_details::GenericRowRecipe;
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
    auto gamma = layernorm_details::bind_layernorm_param(
        scale, ParamPolicy{}, workspace);
    auto beta = layernorm_details::bind_layernorm_param(
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

  /**
   * @brief Core row kernel: single-pass moments, then a transforming write
   *        sweep.
   *
   * The reduction is one `fold` with two accumulators (sum and sum-of-
   * squares), wrapped in `with_unordered_access(x, ...)` because only the
   * moment sweep needs unordered (lane-agnostic) loads; the write sweep
   * re-combines x with gamma/beta/y in one combined unordered region.
   */
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
    // Single-pass variance can round to slightly negative; clamp before
    // the sqrt argument would go NaN on constant rows (where eps alone
    // must carry the denominator).
    const ComputeType variance =
        std::max(sum_sq_value * inv_n - mean * mean, ComputeType(0));
    const ComputeType rstd =
        ComputeType(1) / std::sqrt(variance + config.eps);
    // `center` is dual-purpose: with FusedShift it is the FMA bias
    // -mean*rstd (so `fma(x, rstd, center)` = (x-mean)*rstd); without it,
    // it is the plain mean for the subtract form in the write sweep.
    // Both branches below rely on exactly this meaning.
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
                // FusedShift form: fma(x, rstd, -mean*rstd) equals
                // (x-mean)*rstd mathematically, with one fewer op and one
                // rounding — results are NOT bit-identical to the
                // subtract form used by GenericRowRecipe.
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
    // When both affine parameters exist, enter the combined unordered
    // region directly with all four accesses (no lambda re-packing);
    // otherwise the helper re-inserts the missing nullopt slots so the
    // write lambda keeps its uniform four-argument signature.
    if constexpr (HasGamma && HasBeta) {
      tensor::with_unordered_access(
          x, gamma, beta, y, std::move(write));
    } else {
      layernorm_details::with_layernorm_accesses(
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

#endif // VECOPS_OPS_DETAILS_LAYERNORM_OPERATION_H
