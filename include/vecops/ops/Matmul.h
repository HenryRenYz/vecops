//
// Copyright (c) vecops contributors.
//

#ifndef VECOPS_OPS_MATMUL_H
#define VECOPS_OPS_MATMUL_H

/**
 * @file Matmul.h
 * @brief Public entry point for reusable Config-only Matmul operators.
 *
 * `ops::Matmul` computes the semantic product
 * `C[M,N] = A[M,K] * B[N,K]^T` -- note the **transposed B**: both operands
 * are addressed through `[spatial, K]` layouts, and the explicit transpose
 * in the formula is the only place B is "rotated"; nothing is physically
 * transposed at the memory level.
 *
 * ## Usage overview
 *
 * @code
 * #include "vecops/ops/Matmul.h"
 *
 * using vecops::ops::Matmul;
 * using vecops::ops::MatmulConfig;
 *
 * MatmulConfig<AMX_BF16F32> config;
 * auto op = vecops::ops::matmul(config);
 *
 * nint_t bytes = op.required_workspace(m, n, k, a, b, c);
 * kernel::Workspace workspace(bytes);
 * kernel::WorkspaceView view = workspace.reserve(bytes);
 * op(view, m, n, k, a, b, c);          // or op(execution_scope, ...)
 * @endcode
 *
 * ## Pitfalls
 *
 * - B is consumed as `[N, K]` (B-transposed semantics, see above).
 * - m/n/k are Meta `ValueInput`s: each may be a `Const<N>`, a
 *   `Dynamic<...>`, or a plain runtime integer.
 * - A zero K is an empty reduction, not a no-op: the explicit-C form copies
 *   `CInput` through the output conversion/transform, while the single-C
 *   form materializes zero. Zero M or N has no output elements.
 * - The workspace returned by `required_workspace` must stay alive (and
 *   unmodified) across the matching `operator()` call.
 */

#include <string_view>
#include <utility>

#include "vecops/execution/ExecutionSession.h"
#include "vecops/matmul/Config.h"
#include "vecops/matmul/details/planning/FamilyPlan.h"
#include "vecops/platform/CacheInfo.h"

namespace vecops::ops {

/**
 * Compile-time configuration for `ops::Matmul`.
 *
 * @tparam AtomT               The hardware atom selecting the kernel family
 *                             (e.g. `AMX_BF16F32`, `SME_I8I32<...>`).
 * @tparam FamilySelectionT    Which kernel family serves the product:
 *                             `family_selection::Automatic` / `Prefer<F>` /
 *                             `Require<F>` (see matmul/Family.h).
 * @tparam SchedulerPolicyT    Tile2D traversal policy override.
 * @tparam GenericTiledTuningT Cache/packing knobs of the generic cache-tiled
 *                             family plus an optional explicit spatial
 *                             traversal preference shared by WholeProblem
 *                             (see matmul/Config.h).
 * @tparam CacheInfoProviderT  Where cache sizes come from for automatic
 *                             cache tiling.
 * @tparam EnableSwapAB Allow an architecture policy to replace the problem
 * with its transposed identity `C^T = B*A^T`. The default is `true`; `false`
 * is a hard opt-out and leaves operand roles and M/N unchanged.
 *
 * Orientation selection runs before architecture-family online-packing
 * planning. Consequently, packing costs and packed operand roles are computed
 * for the selected orientation, with no runtime branch. Explicit family or
 * scheduler selections currently retain their requested orientation.
 */
template <
    ::vecops::matmul::Atom AtomT,
    typename FamilySelectionT =
        ::vecops::matmul::family_selection::Automatic,
    typename SchedulerPolicyT = kernel::matmul_policy::Automatic,
    typename GenericTiledTuningT = ::vecops::matmul::GenericTiledTuning<>,
    typename CacheInfoProviderT = platform::SystemCacheInfoProvider,
    bool EnableSwapAB = true>
struct MatmulConfig {
  using Atom = AtomT;
  using FamilySelection = FamilySelectionT;
  using SchedulerPolicy = SchedulerPolicyT;
  using GenericTuning = GenericTiledTuningT;
  using CacheInfoProvider = CacheInfoProviderT;
  using PackingTuning = ::vecops::matmul::MatmulPackingTuning<>;
  static constexpr bool enable_swap_ab = EnableSwapAB;

  // no_unique_address: both members are stateless in the default
  // configuration, so a MatmulConfig object stays empty.
  [[no_unique_address]] GenericTiledTuningT generic_tiled{};
  [[no_unique_address]] CacheInfoProviderT cache_info_provider{};
};

/**
 * Matmul configuration with an explicit top-level A/B packing placement.
 * Kept as a separate opt-in type so the historical six-parameter
 * `MatmulConfig` specialization identity, mangling, and hot-code layout remain
 * unchanged for default users. Orientation selection exchanges the logical
 * A/B policies together with the operands.
 */
template <
    ::vecops::matmul::Atom AtomT,
    ::vecops::matmul::MatmulPackingTuningType PackingTuningT,
    typename FamilySelectionT =
        ::vecops::matmul::family_selection::Automatic,
    typename SchedulerPolicyT = kernel::matmul_policy::Automatic,
    typename GenericTiledTuningT = ::vecops::matmul::GenericTiledTuning<>,
    typename CacheInfoProviderT = platform::SystemCacheInfoProvider,
    bool EnableSwapAB = true>
struct MatmulConfigWithPacking {
  using Atom = AtomT;
  using FamilySelection = FamilySelectionT;
  using SchedulerPolicy = SchedulerPolicyT;
  using GenericTuning = GenericTiledTuningT;
  using CacheInfoProvider = CacheInfoProviderT;
  using PackingTuning = PackingTuningT;
  static constexpr bool enable_swap_ab = EnableSwapAB;

  [[no_unique_address]] GenericTiledTuningT generic_tiled{};
  [[no_unique_address]] CacheInfoProviderT cache_info_provider{};
};

/// Convenience alias: a MatmulConfig that only overrides the scheduler
/// policy (family stays automatic).
template <::vecops::matmul::Atom AtomT,
          typename SchedulerPolicyT = kernel::matmul_policy::Automatic>
using MatmulSchedulerConfig = MatmulConfig<
    AtomT, ::vecops::matmul::family_selection::Automatic,
    SchedulerPolicyT>;

/**
 * Reusable semantic matrix-multiply operator.
 *
 * The object stores configuration only. Family selection is resolved without
 * allowing family-local tuning parameters to change the selected family.
 * `Config::enable_swap_ab` controls compile-time problem transposition. When
 * enabled, an architecture policy may evaluate
 * `C^T = B*A^T` by exchanging the two internal `[spatial,K]` operands and
 * transposing the C specs. Disabling it guarantees that the supplied M/N and
 * operand orientation reach the selected family unchanged.
 * The single-C overload computes `C = A*B^T` through a zero-valued accumulator
 * input. The explicit-C overload computes
 * `COutput = CInput + A*B^T`; CInput and COutput may be different operands.
 */
template <typename Config>
class Matmul {
public:
  using Atom = typename Config::Atom;
  using Implementation =
      ::vecops::matmul::details::SelectedImplementation<Atom>;
  using ResourceRequirements =
      kernel::matmul_implementation::resource_requirements_t<Implementation>;
  using KernelFamily =
      ::vecops::matmul::details::selected_family_t<Config>;
  using Plan = ::vecops::matmul::details::SelectedFamilyPlan<Config>;

  /// Stable name of the selected kernel family (diagnostics/tests).
  static constexpr std::string_view kernel_family_name() {
    return ::vecops::matmul::kernel_family::Info<KernelFamily>::name;
  }

  /// The stored configuration (by value; config objects are empty or tiny).
  const Config config;

  VECOPS_INLINE constexpr explicit Matmul(Config cfg)
      : config(std::move(cfg)) {}

  /// Workspace bytes for the single-C form; the C operand is adapted into
  /// the explicit-C form through a zero-valued accumulator input.
  template <meta::ValueInput M, meta::ValueInput N, meta::ValueInput K,
            tensor::InputOperand A, tensor::InputOperand B,
            tensor::OutputOperand C>
  VECOPS_INLINE nint_t required_workspace(
      M&& m, N&& n, K&& k, A&& a, B&& b, C&& c) const {
    // Adapt C = A*B^T to COut = CIn + A*B^T by dressing the output as a
    // zero-valued C input; the plan then sees the unified accumulate form.
    auto c_output = tensor::as_output_spec<typename Atom::TAcc>(
        std::forward<C>(c));
    using Memory = typename decltype(c_output)::MemoryElement;
    auto c_input = tensor::input<typename Atom::TAcc>(
        c_output.tensor(),
        tensor::zeros_transform<typename Atom::TAcc, Memory>);
    return Plan::required_workspace(
        config, std::forward<M>(m), std::forward<N>(n),
        std::forward<K>(k), std::forward<A>(a), std::forward<B>(b),
        std::move(c_input), std::move(c_output));
  }

  /// Workspace bytes for the explicit accumulate form
  /// `COutput = CInput + A*B^T`.
  template <meta::ValueInput M, meta::ValueInput N, meta::ValueInput K,
            tensor::InputOperand A, tensor::InputOperand B,
            tensor::InputOperand CInput, tensor::OutputOperand COutput>
  VECOPS_INLINE nint_t required_workspace(
      M&& m, N&& n, K&& k, A&& a, B&& b,
      CInput&& c_input, COutput&& c_output) const {
    return Plan::required_workspace(
        config, std::forward<M>(m), std::forward<N>(n),
        std::forward<K>(k), std::forward<A>(a), std::forward<B>(b),
        std::forward<CInput>(c_input), std::forward<COutput>(c_output));
  }

  /// Run `C = A*B^T` on an execution scope (see required_workspace for the
  /// C-to-accumulator adaptation).
  template <execution::ExecutionScope Scope,
            meta::ValueInput M, meta::ValueInput N, meta::ValueInput K,
            tensor::InputOperand A, tensor::InputOperand B,
            tensor::OutputOperand C>
  VECOPS_INLINE void operator()(
      Scope& scope, M&& m, N&& n, K&& k,
      A&& a, B&& b, C&& c) const {
    // Same zeros-transform adaptation as required_workspace above.
    auto c_output = tensor::as_output_spec<typename Atom::TAcc>(
        std::forward<C>(c));
    using Memory = typename decltype(c_output)::MemoryElement;
    auto c_input = tensor::input<typename Atom::TAcc>(
        c_output.tensor(),
        tensor::zeros_transform<typename Atom::TAcc, Memory>);
    Plan::run(
        scope, config, std::forward<M>(m), std::forward<N>(n),
        std::forward<K>(k), std::forward<A>(a), std::forward<B>(b),
        std::move(c_input), std::move(c_output));
  }

  /// Run `COutput = CInput + A*B^T` on an execution scope.
  template <execution::ExecutionScope Scope,
            meta::ValueInput M, meta::ValueInput N, meta::ValueInput K,
            tensor::InputOperand A, tensor::InputOperand B,
            tensor::InputOperand CInput, tensor::OutputOperand COutput>
  VECOPS_INLINE void operator()(
      Scope& scope, M&& m, N&& n, K&& k,
      A&& a, B&& b, CInput&& c_input, COutput&& c_output) const {
    Plan::run(
        scope, config, std::forward<M>(m), std::forward<N>(n),
        std::forward<K>(k), std::forward<A>(a), std::forward<B>(b),
        std::forward<CInput>(c_input), std::forward<COutput>(c_output));
  }

  /// Single-C form on a raw workspace: wraps it in a temporary
  /// ExecutionSession and forwards.
  template <meta::ValueInput M, meta::ValueInput N, meta::ValueInput K,
            tensor::InputOperand A, tensor::InputOperand B,
            tensor::OutputOperand C>
  VECOPS_INLINE void operator()(
      kernel::WorkspaceView& workspace, M&& m, N&& n, K&& k,
      A&& a, B&& b, C&& c) const {
    ExecutionSession execution{workspace};
    (*this)(execution, std::forward<M>(m), std::forward<N>(n),
            std::forward<K>(k), std::forward<A>(a), std::forward<B>(b),
            std::forward<C>(c));
  }

  /// Explicit-C form on a raw workspace: wraps it in a temporary
  /// ExecutionSession and forwards.
  template <meta::ValueInput M, meta::ValueInput N, meta::ValueInput K,
            tensor::InputOperand A, tensor::InputOperand B,
            tensor::InputOperand CInput, tensor::OutputOperand COutput>
  VECOPS_INLINE void operator()(
      kernel::WorkspaceView& workspace, M&& m, N&& n, K&& k,
      A&& a, B&& b, CInput&& c_input, COutput&& c_output) const {
    ExecutionSession execution{workspace};
    (*this)(execution, std::forward<M>(m), std::forward<N>(n),
            std::forward<K>(k), std::forward<A>(a), std::forward<B>(b),
            std::forward<CInput>(c_input),
            std::forward<COutput>(c_output));
  }

};

/// Factory: wrap a MatmulConfig into a Matmul operator object.
template <typename Config>
VECOPS_INLINE constexpr auto matmul(Config config) {
  return Matmul<Config>{std::move(config)};
}

} // namespace vecops::ops

#endif // VECOPS_OPS_MATMUL_H
