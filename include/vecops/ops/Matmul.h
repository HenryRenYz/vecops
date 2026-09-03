//
// Copyright (c) vecops contributors.
//

#ifndef VECOPS_OPS_MATMUL_H
#define VECOPS_OPS_MATMUL_H

/**
 * @file vecops/ops/Matmul.h
 * @brief Public entry point for reusable Config-only Matmul operators.
 *
 * This header exposes `Matmul<Config>` — the user-facing matrix-multiply
 * operator — plus the `MatmulConfig` / `MatmulSchedulerConfig` bundles that
 * parameterize it and the `matmul()` factory. The operator object stores
 * configuration only; all shape and operand information is passed per call,
 * so one configured object can serve any number of problem shapes.
 *
 * The computed product is always `C = A * B^T`: A is `[M,K]`, B is `[N,K]`
 * (i.e. B is accessed transposed, dot products run along each operand's
 * last dimension), and C is `[M,N]` in the accumulator element type. The
 * two output forms are:
 *
 * - **Single-C overload**: `C` is written as a pure product, discarding any
 *   previous contents.
 * - **Explicit-C overload**: `COutput = CInput + A * B^T`, where CInput and
 *   COutput are separate operands and may alias or differ.
 *
 * Execution goes through the planning layer selected by the Config: an
 * architecture-specialized whole-problem kernel family (AMX/SME) or the
 * generic cache-tiled fallback, chosen per problem without letting
 * family-local tuning parameters change the selected family.
 *
 * ## Usage
 *
 * @code
 * #include "vecops/ops/Matmul.h"
 * using namespace vecops;
 *
 * // Configure once: one Atom plus policy defaults.
 * auto operation = ops::matmul(ops::MatmulConfig<matmul::AMX_BF16F32>{});
 *
 * // Per problem: query the workspace requirement, then run.
 * nint_t required = operation.required_workspace(m, n, k, a, b, c);
 * kernel::Workspace storage(required);
 * auto workspace = storage.view();
 * operation(workspace, m, n, k, a, b, c);            // C = A * B^T
 * operation(workspace, m, n, k, a, b, c_in, c_out);  // C_out = C_in + A*B^T
 * @endcode
 *
 * ## Pitfalls
 *
 * - B is consumed **transposed**: pass B as `[N,K]`, never `[K,N]`. The
 *   kernel reduces along the last dimension of both operands.
 * - The workspace pointer handed to the `WorkspaceView&` overloads must
 *   stay valid and unmodified until the call returns; the workspace is
 *   scratch memory only and never carries results between calls.
 * - `m`, `n`, `k` are `meta::ValueInput`s: static (compile-time) extents
 *   must round-trip consistently with the tensor operands' layouts — a
 *   static extent is trusted, not re-derived from the tensors.
 * - The single-C overload does not read C, but C must still be a valid,
 *   writable output operand of the accumulator type.
 */

#include <string_view>
#include <utility>

#include "vecops/execution/ExecutionSession.h"
#include "vecops/matmul/Config.h"
#include "vecops/matmul/details/planning/FamilyPlan.h"
#include "vecops/platform/CacheInfo.h"

namespace vecops::ops {

/**
 * @brief Configuration bundle for the `Matmul` operator.
 *
 * Groups the five orthogonal policy axes of a matmul into one copyable
 * value type. The bundle is pure configuration: none of the members carry
 * per-problem state, and family-local tuning knobs live inside
 * `generic_tiled` so that setting them cannot silently change which kernel
 * family is selected.
 *
 * @tparam AtomT              Hardware atom identifying the instruction
 *                            family and operand types (e.g.
 *                            `matmul::AMX_BF16F32`).
 * @tparam FamilySelectionT   How the kernel family is chosen: automatic,
 *                            `family_selection::Prefer<F>`, or `Require<F>`.
 * @tparam SchedulerPolicyT   Tile-scheduler policy; `Automatic` keeps the
 *                            architecture's benchmark-tuned traversal.
 * @tparam GenericTiledTuningT Tuning knobs understood only by the generic
 *                            cache-tiled fallback family (MC/NC/KC, loop
 *                            order, packing, accumulator buffering).
 * @tparam CacheInfoProviderT Callable returning the process-wide
 *                            `platform::CacheInfo` used for cache tiling
 *                            decisions; swappable for deterministic tests.
 */
template <
    ::vecops::matmul::Atom AtomT,
    typename FamilySelectionT =
        ::vecops::matmul::family_selection::Automatic,
    typename SchedulerPolicyT = kernel::matmul_policy::Automatic,
    typename GenericTiledTuningT = ::vecops::matmul::GenericTiledTuning<>,
    typename CacheInfoProviderT = platform::SystemCacheInfoProvider>
struct MatmulConfig {
  using Atom = AtomT;
  using FamilySelection = FamilySelectionT;
  using SchedulerPolicy = SchedulerPolicyT;
  using GenericTuning = GenericTiledTuningT;
  using CacheInfoProvider = CacheInfoProviderT;

  [[no_unique_address]] GenericTiledTuningT generic_tiled{};
  [[no_unique_address]] CacheInfoProviderT cache_info_provider{};
};

/**
 * @brief Convenience alias: `MatmulConfig` with only the scheduler policy
 * exposed.
 *
 * Use this when the atom and the family selection stay automatic but the
 * tile scheduler traversal must be pinned (e.g. forcing a specific
 * `kernel::loop::tile2d_policy` policy type).
 *
 * @tparam AtomT           Hardware atom, as in `MatmulConfig`.
 * @tparam SchedulerPolicyT Tile-scheduler policy (default: benchmark-tuned
 *                          automatic traversal).
 */
template <::vecops::matmul::Atom AtomT,
          typename SchedulerPolicyT = kernel::matmul_policy::Automatic>
using MatmulSchedulerConfig = MatmulConfig<
    AtomT, ::vecops::matmul::family_selection::Automatic,
    SchedulerPolicyT>;

/**
 * @brief Reusable semantic matrix-multiply operator computing
 * `C = A * B^T` (or the accumulating form `COutput = CInput + A*B^T`).
 *
 * The object stores configuration only. Family selection is resolved without
 * allowing family-local tuning parameters to change the selected family.
 * The single-C overload computes `C = A*B^T` through a zero-valued accumulator
 * input. The explicit-C overload computes
 * `COutput = CInput + A*B^T`; CInput and COutput may be different operands.
 *
 * Both operand-B layouts share the `[N,K]` shape — the transpose is applied
 * by the kernel's traversal, not by materializing a transposed copy.
 *
 * @code
 * auto mm = ops::matmul(ops::MatmulConfig<matmul::AMX_BF16F32>{});
 * mm(scope, m, n, k, a, b, c);            // C = A * B^T
 * mm(scope, m, n, k, a, b, c_in, c_out);  // C_out = C_in + A * B^T
 * @endcode
 *
 * @tparam Config A `MatmulConfig` (or equivalent) bundle.
 *
 * @note The `WorkspaceView&` overloads construct a temporary
 * `ExecutionSession` around the workspace and forward to the scope-based
 * overloads; use the scope forms directly when issuing several operations
 * inside one resource-activation interval.
 */
template <typename Config>
class Matmul {
public:
  /** @brief Hardware atom from the config; fixes TA/TB/TC/TAcc and extents. */
  using Atom = typename Config::Atom;
  /** @brief Architecture backend selected from the atom's `KernelKind`. */
  using Implementation =
      ::vecops::matmul::details::SelectedImplementation<Atom>;
  /** @brief Execution resources (e.g. AMX tile / SME ZA state) required. */
  using ResourceRequirements =
      kernel::matmul_implementation::resource_requirements_t<Implementation>;
  /** @brief Kernel family chosen by the config's selection policy. */
  using KernelFamily =
      ::vecops::matmul::details::selected_family_t<Config>;
  /** @brief Concrete plan type that implements `required_workspace`/`run`. */
  using Plan = ::vecops::matmul::details::SelectedFamilyPlan<Config>;

  /**
   * @brief Stable name of the selected kernel family (for diagnostics).
   * @return A static string view naming the family, e.g. `"AMX"` or
   *         `"GenericTiled"`.
   */
  static constexpr std::string_view kernel_family_name() {
    return ::vecops::matmul::kernel_family::Info<KernelFamily>::name;
  }

  /** @brief The stored configuration bundle. */
  const Config config;

  /**
   * @brief Construct an operator from a configuration bundle.
   * @param cfg Configuration to store (moved in).
   */
  VECOPS_INLINE constexpr explicit Matmul(Config cfg)
      : config(std::move(cfg)) {}

  /**
   * @brief Workspace requirement for the pure-product form
   * `C = A * B^T`.
   *
   * The single output C is wrapped as an explicit input/output pair whose
   * input side always reads zeros (see the inline note below), so the
   * reported size matches what the corresponding `operator()` overload
   * will consume.
   *
   * @tparam M,N,K Value inputs carrying the problem extents.
   * @tparam A,B   Input operands (`[M,K]` and `[N,K]` respectively).
   * @tparam C     Output operand (`[M,N]`, accumulator-typed).
   * @param m,n,k  Problem extents; static values may enable specialized
   *               planning.
   * @param a,b,c  The operands themselves.
   * @return Workspace size in bytes for this problem shape.
   */
  template <meta::ValueInput M, meta::ValueInput N, meta::ValueInput K,
            tensor::InputOperand A, tensor::InputOperand B,
            tensor::OutputOperand C>
  VECOPS_INLINE nint_t required_workspace(
      M&& m, N&& n, K&& k, A&& a, B&& b, C&& c) const {
    auto c_output = tensor::as_output_spec<typename Atom::TAcc>(
        std::forward<C>(c));
    using Memory = typename decltype(c_output)::MemoryElement;
    // Adaptation trick: reuse the unified accumulate path for the pure
    // product. The output C is additionally presented as an input operand
    // over the same layout, but wrapped in `zeros_transform` so every read
    // yields zero regardless of memory contents. The kernel therefore
    // computes COut = 0 + A*B^T without requiring C to be pre-zeroed and
    // without a separate non-accumulating code path.
    auto c_input = tensor::input<typename Atom::TAcc>(
        c_output.tensor(),
        tensor::zeros_transform<typename Atom::TAcc, Memory>);
    return Plan::required_workspace(
        config, std::forward<M>(m), std::forward<N>(n),
        std::forward<K>(k), std::forward<A>(a), std::forward<B>(b),
        std::move(c_input), std::move(c_output));
  }

  /**
   * @brief Workspace requirement for the accumulating form
   * `COutput = CInput + A * B^T`.
   *
   * @tparam M,N,K     Value inputs carrying the problem extents.
   * @tparam A,B       Input operands (`[M,K]` and `[N,K]`).
   * @tparam CInput    Input operand supplying the initial accumulator
   *                   values (`[M,N]`, accumulator-typed).
   * @tparam COutput   Output operand receiving the result; may be the same
   *                   tensor as CInput or a distinct one.
   * @param m,n,k      Problem extents.
   * @param a,b        The A and B operands.
   * @param c_input    Initial accumulator values to add on top of.
   * @param c_output   Destination of the accumulated result.
   * @return Workspace size in bytes for this problem shape.
   */
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

  /**
   * @brief Run `C = A * B^T` inside an execution scope.
   *
   * @tparam Scope  An `execution::ExecutionScope` whose backend matches the
   *                selected implementation; it must already have the
   *                required execution resources activated.
   * @tparam M,N,K  Value inputs carrying the problem extents.
   * @tparam A,B    Input operands (`[M,K]` and `[N,K]`).
   * @tparam C      Output operand (`[M,N]`, accumulator-typed); prior
   *                contents are not read.
   * @param scope   Active execution scope to run under.
   * @param m,n,k   Problem extents.
   * @param a,b,c   The operands themselves.
   */
  template <execution::ExecutionScope Scope,
            meta::ValueInput M, meta::ValueInput N, meta::ValueInput K,
            tensor::InputOperand A, tensor::InputOperand B,
            tensor::OutputOperand C>
  VECOPS_INLINE void operator()(
      Scope& scope, M&& m, N&& n, K&& k,
      A&& a, B&& b, C&& c) const {
    auto c_output = tensor::as_output_spec<typename Atom::TAcc>(
        std::forward<C>(c));
    using Memory = typename decltype(c_output)::MemoryElement;
    // Same zeros_transform adaptation as in required_workspace above: the
    // output is mirrored as an always-zero input so the plan's single
    // accumulate path (COut = CIn + A*B^T) computes the pure product
    // without reading or pre-zeroing C's memory.
    auto c_input = tensor::input<typename Atom::TAcc>(
        c_output.tensor(),
        tensor::zeros_transform<typename Atom::TAcc, Memory>);
    Plan::run(
        scope, config, std::forward<M>(m), std::forward<N>(n),
        std::forward<K>(k), std::forward<A>(a), std::forward<B>(b),
        std::move(c_input), std::move(c_output));
  }

  /**
   * @brief Run `COutput = CInput + A * B^T` inside an execution scope.
   *
   * @tparam Scope    An `execution::ExecutionScope` with the required
   *                  execution resources already activated.
   * @tparam M,N,K    Value inputs carrying the problem extents.
   * @tparam A,B      Input operands (`[M,K]` and `[N,K]`).
   * @tparam CInput   Input operand supplying initial accumulator values.
   * @tparam COutput  Output operand receiving the result; may differ from
   *                  CInput.
   * @param scope     Active execution scope to run under.
   * @param m,n,k     Problem extents.
   * @param a,b       The A and B operands.
   * @param c_input   Initial accumulator values.
   * @param c_output  Destination of the accumulated result.
   */
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

  /**
   * @brief Run `C = A * B^T` against a raw workspace view.
   *
   * Convenience overload: wraps `workspace` in a temporary
   * `ExecutionSession` (activating the required resources for the duration
   * of the call) and forwards to the scope-based overload.
   *
   * @tparam M,N,K Value inputs carrying the problem extents.
   * @tparam A,B   Input operands (`[M,K]` and `[N,K]`).
   * @tparam C     Output operand (`[M,N]`, accumulator-typed).
   * @param workspace View of storage with at least
   *                  `required_workspace(...)` bytes for this problem.
   * @param m,n,k  Problem extents.
   * @param a,b,c  The operands themselves.
   */
  template <meta::ValueInput M, meta::ValueInput N, meta::ValueInput K,
            tensor::InputOperand A, tensor::InputOperand B,
            tensor::OutputOperand C>
  VECOPS_INLINE void operator()(
      kernel::WorkspaceView& workspace, M&& m, N&& n, K&& k,
      A&& a, B&& b, C&& c) const {
    // Temporary session per call: resource activation is scoped to this
    // single operation. Prefer the Scope& overloads to amortize activation
    // across several matmuls sharing one interval.
    ExecutionSession execution{workspace};
    (*this)(execution, std::forward<M>(m), std::forward<N>(n),
            std::forward<K>(k), std::forward<A>(a), std::forward<B>(b),
            std::forward<C>(c));
  }

  /**
   * @brief Run `COutput = CInput + A * B^T` against a raw workspace view.
   *
   * As above, this wraps `workspace` in a temporary `ExecutionSession` and
   * forwards to the scope-based accumulating overload.
   *
   * @tparam M,N,K    Value inputs carrying the problem extents.
   * @tparam A,B      Input operands (`[M,K]` and `[N,K]`).
   * @tparam CInput   Input operand supplying initial accumulator values.
   * @tparam COutput  Output operand receiving the result.
   * @param workspace View of storage with at least
   *                  `required_workspace(...)` bytes for this problem.
   * @param m,n,k     Problem extents.
   * @param a,b       The A and B operands.
   * @param c_input   Initial accumulator values.
   * @param c_output  Destination of the accumulated result.
   */
  template <meta::ValueInput M, meta::ValueInput N, meta::ValueInput K,
            tensor::InputOperand A, tensor::InputOperand B,
            tensor::InputOperand CInput, tensor::OutputOperand COutput>
  VECOPS_INLINE void operator()(
      kernel::WorkspaceView& workspace, M&& m, N&& n, K&& k,
      A&& a, B&& b, CInput&& c_input, COutput&& c_output) const {
    // Temporary session per call; see the single-C overload above.
    ExecutionSession execution{workspace};
    (*this)(execution, std::forward<M>(m), std::forward<N>(n),
            std::forward<K>(k), std::forward<A>(a), std::forward<B>(b),
            std::forward<CInput>(c_input),
            std::forward<COutput>(c_output));
  }

};

/**
 * @brief Factory: build a `Matmul` operator from a configuration bundle.
 * @tparam Config The configuration bundle type.
 * @param config  Configuration to store inside the operator.
 * @return A `Matmul<Config>` value ready for `required_workspace`/`operator()`.
 */
template <typename Config>
VECOPS_INLINE constexpr auto matmul(Config config) {
  return Matmul<Config>{std::move(config)};
}

} // namespace vecops::ops

#endif // VECOPS_OPS_MATMUL_H
