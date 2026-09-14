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

#include <cstddef>
#include <string_view>
#include <type_traits>
#include <utility>

#include "vecops/execution/ExecutionSession.h"
#include "vecops/execution/Parallel.h"
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
template <::vecops::matmul::Atom AtomT, typename FamilySelectionT = ::vecops::matmul::family_selection::Automatic,
          typename SchedulerPolicyT = kernel::matmul_policy::Automatic,
          typename GenericTiledTuningT = ::vecops::matmul::GenericTiledTuning<>,
          typename CacheInfoProviderT = platform::SystemCacheInfoProvider, bool EnableSwapAB = true>
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
template <::vecops::matmul::Atom AtomT, ::vecops::matmul::MatmulPackingTuningType PackingTuningT,
          typename FamilySelectionT = ::vecops::matmul::family_selection::Automatic,
          typename SchedulerPolicyT = kernel::matmul_policy::Automatic,
          typename GenericTiledTuningT = ::vecops::matmul::GenericTiledTuning<>,
          typename CacheInfoProviderT = platform::SystemCacheInfoProvider, bool EnableSwapAB = true>
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
template <::vecops::matmul::Atom AtomT, typename SchedulerPolicyT = kernel::matmul_policy::Automatic>
using MatmulSchedulerConfig = MatmulConfig<AtomT, ::vecops::matmul::family_selection::Automatic, SchedulerPolicyT>;

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
  using Implementation = ::vecops::matmul::details::SelectedImplementation<Atom>;
  using ResourceRequirements = kernel::matmul_implementation::resource_requirements_t<Implementation>;
  using KernelFamily = ::vecops::matmul::details::selected_family_t<Config>;
  using Plan = ::vecops::matmul::details::SelectedFamilyPlan<Config>;

  /// Stable name of the selected kernel family (diagnostics/tests).
  static constexpr std::string_view kernel_family_name() {
    return ::vecops::matmul::kernel_family::Info<KernelFamily>::name;
  }

  /// The stored configuration (by value; config objects are empty or tiny).
  const Config config;

  VECOPS_INLINE constexpr explicit Matmul(Config cfg)
    : config(std::move(cfg)) {
  }

  /**
   * A reusable architecture-family plan whose operands carry layout and
   * access semantics but no execution-time addresses.
   *
   * One scratch tensor is allocated per logical lane when this object is
   * constructed.  Calls bind the stored patterns to the current operands and
   * create the legacy byte cursor and resource session internally.  Thus a
   * kernel author neither sizes nor resets scratch and cannot accidentally
   * share it between lanes.
   *
   * This first migration step still rebuilds the lightweight bound invocation
   * on every call.  A later Plan/Call split can retain the address-independent
   * decisions without changing this public surface.
   */
  template <nint_t Parallelism, typename M, typename N, typename K, typename APattern, typename BPattern,
            typename CInputPattern, typename COutputPattern, typename WorkerScratch>
  class PatternPrepared {
  public:
    static_assert(Parallelism > 0, "prepared Matmul parallelism must be positive");
    using ResourceRequirements = typename Matmul::ResourceRequirements;

    VECOPS_INLINE PatternPrepared(Config config, M m, N n, K k, APattern a, BPattern b, CInputPattern c_input,
                                  COutputPattern c_output, WorkerScratch worker_scratch, nint_t workspace_bytes)
      : config_(std::move(config))
      , m_(std::move(m))
      , n_(std::move(n))
      , k_(std::move(k))
      , a_(std::move(a))
      , b_(std::move(b))
      , c_input_(std::move(c_input))
      , c_output_(std::move(c_output))
      , worker_scratch_(std::move(worker_scratch))
      , workspace_bytes_(workspace_bytes) {
    }

    /**
     * Execute the explicit-C form with active extents on this lane's scratch.
     * Planning extents remain capacity bounds; they are never reused as the
     * logical extent of a shortened dynamic tile.
     */
    template <meta::ValueInput ActiveM, meta::ValueInput ActiveN, meta::ValueInput ActiveK, tensor::InputOperand A,
              tensor::InputOperand B, tensor::InputOperand CInput, tensor::OutputOperand COutput>
      requires(tensor::is_bound_tensor_view_v<A> && tensor::is_bound_tensor_view_v<B> &&
               tensor::is_bound_tensor_view_v<CInput> && tensor::is_bound_tensor_view_v<COutput>)
    VECOPS_INLINE void operator()(execution::TaskContext<Parallelism> task, ActiveM&& active_m, ActiveN&& active_n,
                                  ActiveK&& active_k, A&& a, B&& b, CInput&& c_input, COutput&& c_output) const {
      run(task, meta::to_value(std::forward<ActiveM>(active_m)), meta::to_value(std::forward<ActiveN>(active_n)),
          meta::to_value(std::forward<ActiveK>(active_k)), tensor::rebind(a_, std::forward<A>(a)),
          tensor::rebind(b_, std::forward<B>(b)), tensor::rebind(c_input_, std::forward<CInput>(c_input)),
          tensor::rebind(c_output_, std::forward<COutput>(c_output)));
    }

    /** Infer M/N from C and retain the prepared logical K (packed layouts do not encode a tail K). */
    template <tensor::InputOperand A, tensor::InputOperand B, tensor::InputOperand CInput,
              tensor::OutputOperand COutput>
      requires(tensor::is_bound_tensor_view_v<A> && tensor::is_bound_tensor_view_v<B> &&
               tensor::is_bound_tensor_view_v<CInput> && tensor::is_bound_tensor_view_v<COutput>)
    VECOPS_INLINE void operator()(execution::TaskContext<Parallelism> task, A&& a, B&& b, CInput&& c_input,
                                  COutput&& c_output) const {
      auto active_a = tensor::rebind(a_, std::forward<A>(a));
      auto active_b = tensor::rebind(b_, std::forward<B>(b));
      auto active_c_input = tensor::rebind(c_input_, std::forward<CInput>(c_input));
      auto active_c_output = tensor::rebind(c_output_, std::forward<COutput>(c_output));
      constexpr int Rank = decltype(active_c_output)::OutputTensor::Ndim;
      run(task, tensor::size<Rank - 2>(active_c_output.output_layout()),
          tensor::size<Rank - 1>(active_c_output.output_layout()),
          k_, std::move(active_a),
          std::move(active_b), std::move(active_c_input), std::move(active_c_output));
    }

    /** Execute the single-C form with explicit active extents. */
    template <meta::ValueInput ActiveM, meta::ValueInput ActiveN, meta::ValueInput ActiveK, tensor::InputOperand A,
              tensor::InputOperand B, tensor::OutputOperand C>
      requires(tensor::is_bound_tensor_view_v<A> && tensor::is_bound_tensor_view_v<B> &&
               tensor::is_bound_tensor_view_v<C>)
    VECOPS_INLINE void operator()(execution::TaskContext<Parallelism> task, ActiveM&& active_m, ActiveN&& active_n,
                                  ActiveK&& active_k, A&& a, B&& b, C&& c) const {
      auto active_c_output = tensor::rebind(c_output_, std::forward<C>(c));
      using Memory = typename decltype(active_c_output)::MemoryElement;
      auto active_c_input = tensor::input<typename Atom::TAcc>(active_c_output.tensor(),
                                                               tensor::zeros_transform<typename Atom::TAcc, Memory>);
      run(task, meta::to_value(std::forward<ActiveM>(active_m)), meta::to_value(std::forward<ActiveN>(active_n)),
          meta::to_value(std::forward<ActiveK>(active_k)), tensor::rebind(a_, std::forward<A>(a)),
          tensor::rebind(b_, std::forward<B>(b)), tensor::rebind(c_input_, std::move(active_c_input)),
          std::move(active_c_output));
    }

    /** Infer M/N from C and retain the prepared logical K (packed layouts do not encode a tail K). */
    template <tensor::InputOperand A, tensor::InputOperand B, tensor::OutputOperand C>
      requires(tensor::is_bound_tensor_view_v<A> && tensor::is_bound_tensor_view_v<B> &&
               tensor::is_bound_tensor_view_v<C>)
    VECOPS_INLINE void operator()(execution::TaskContext<Parallelism> task, A&& a, B&& b, C&& c) const {
      auto active_a = tensor::rebind(a_, std::forward<A>(a));
      auto active_b = tensor::rebind(b_, std::forward<B>(b));
      auto active_c_output = tensor::rebind(c_output_, std::forward<C>(c));
      using Memory = typename decltype(active_c_output)::MemoryElement;
      auto active_c_input = tensor::input<typename Atom::TAcc>(active_c_output.tensor(),
                                                               tensor::zeros_transform<typename Atom::TAcc, Memory>);
      constexpr int Rank = decltype(active_c_output)::OutputTensor::Ndim;
      run(task, tensor::size<Rank - 2>(active_c_output.output_layout()),
          tensor::size<Rank - 1>(active_c_output.output_layout()),
          k_, std::move(active_a),
          std::move(active_b), tensor::rebind(c_input_, std::move(active_c_input)), std::move(active_c_output));
    }

  private:
    template <meta::ValueInput ActiveM, meta::ValueInput ActiveN, meta::ValueInput ActiveK, typename A, typename B,
              typename CInput, typename COutput>
    VECOPS_INLINE void run(execution::TaskContext<Parallelism> task, ActiveM&& active_m, ActiveN&& active_n,
                           ActiveK&& active_k, A&& a, B&& b, CInput&& c_input, COutput&& c_output) const {
      auto active_m_value = meta::to_value(std::forward<ActiveM>(active_m));
      auto active_n_value = meta::to_value(std::forward<ActiveN>(active_n));
      auto active_k_value = meta::to_value(std::forward<ActiveK>(active_k));
      auto active_a = std::forward<A>(a);
      auto active_b = std::forward<B>(b);
      auto active_c_input = std::forward<CInput>(c_input);
      auto active_c_output = std::forward<COutput>(c_output);
      auto invocation = Plan::prepare(config_, active_m_value, active_n_value, active_k_value,
                                      active_a, active_b, active_c_input, active_c_output);
      const nint_t active_workspace_bytes = invocation.required_workspace();
      VECOPS_CHECK(active_workspace_bytes <= workspace_bytes_,
                   "active Matmul scratch exceeds its planning-pattern capacity");
      auto scratch = task.local(worker_scratch_);
      kernel::WorkspaceView workspace{scratch.data(), workspace_bytes_};
      ExecutionSession execution{workspace};
      invocation(execution);
    }

    Config config_;
    M m_;
    N n_;
    K k_;
    APattern a_;
    BPattern b_;
    CInputPattern c_input_;
    COutputPattern c_output_;
    WorkerScratch worker_scratch_;
    nint_t workspace_bytes_ = 0;
  };

  /**
   * Prepare a single-C Matmul from storage-less operand patterns.
   *
   * `site_name` distinguishes simultaneously-live operations in one lexical
   * scope. The compile-time lane count controls both code specialization and
   * the number of private scratch replicas registered with WorkspaceContext.
   */
  template <nint_t Parallelism, typename WorkspaceAuthority, meta::ValueInput M, meta::ValueInput N, meta::ValueInput K,
            tensor::UnboundTensorView A, tensor::UnboundTensorView B, tensor::UnboundTensorView C>
    requires requires(WorkspaceAuthority& authority, nint_t bytes) {
      authority.template worker_tensor<std::byte, Parallelism>(std::string_view{},
                                                               tensor::make_shape(meta::Any{bytes}));
    }
  VECOPS_INLINE auto prepare(WorkspaceAuthority& parent, std::string_view site_name, M&& m, N&& n, K&& k, A&& a, B&& b,
                             C&& c) const {
    static_assert(::vecops::matmul::kernel_family::ArchitectureFamily<KernelFamily>,
                  "pattern-prepared Matmul currently requires an architecture kernel family");
    auto m_value = meta::to_value(std::forward<M>(m));
    auto n_value = meta::to_value(std::forward<N>(n));
    auto k_value = meta::to_value(std::forward<K>(k));
    auto a_pattern = tensor::as_input_spec<typename Atom::TA>(std::forward<A>(a));
    auto b_pattern = tensor::as_input_spec<typename Atom::TB>(std::forward<B>(b));
    auto c_output_pattern = tensor::as_output_spec<typename Atom::TAcc>(std::forward<C>(c));
    using Memory = typename decltype(c_output_pattern)::MemoryElement;
    auto c_input_pattern = tensor::input<typename Atom::TAcc>(c_output_pattern.tensor(),
                                                              tensor::zeros_transform<typename Atom::TAcc, Memory>);
    auto planning_invocation =
      Plan::prepare(config, m_value, n_value, k_value, a_pattern, b_pattern, c_input_pattern, c_output_pattern);
    const nint_t bytes = planning_invocation.required_workspace();
    auto scratch =
      parent.template worker_tensor<std::byte, Parallelism>(site_name, tensor::make_shape(meta::Any{bytes}));
    return PatternPrepared<Parallelism, decltype(m_value), decltype(n_value), decltype(k_value), decltype(a_pattern),
                           decltype(b_pattern), decltype(c_input_pattern), decltype(c_output_pattern),
                           decltype(scratch)>{config,
                                              m_value,
                                              n_value,
                                              k_value,
                                              std::move(a_pattern),
                                              std::move(b_pattern),
                                              std::move(c_input_pattern),
                                              std::move(c_output_pattern),
                                              std::move(scratch),
                                              bytes};
  }

  /** Prepare an explicit-C Matmul from storage-less operand patterns. */
  template <nint_t Parallelism, typename WorkspaceAuthority, meta::ValueInput M, meta::ValueInput N, meta::ValueInput K,
            tensor::UnboundTensorView A, tensor::UnboundTensorView B, tensor::UnboundTensorView CInput,
            tensor::UnboundTensorView COutput>
    requires requires(WorkspaceAuthority& authority, nint_t bytes) {
      authority.template worker_tensor<std::byte, Parallelism>(std::string_view{},
                                                               tensor::make_shape(meta::Any{bytes}));
    }
  VECOPS_INLINE auto prepare(WorkspaceAuthority& parent, std::string_view site_name, M&& m, N&& n, K&& k, A&& a, B&& b,
                             CInput&& c_input, COutput&& c_output) const {
    static_assert(::vecops::matmul::kernel_family::ArchitectureFamily<KernelFamily>,
                  "pattern-prepared Matmul currently requires an architecture kernel family");
    auto m_value = meta::to_value(std::forward<M>(m));
    auto n_value = meta::to_value(std::forward<N>(n));
    auto k_value = meta::to_value(std::forward<K>(k));
    auto a_pattern = tensor::as_input_spec<typename Atom::TA>(std::forward<A>(a));
    auto b_pattern = tensor::as_input_spec<typename Atom::TB>(std::forward<B>(b));
    auto c_input_pattern = tensor::as_input_spec<typename Atom::TAcc>(std::forward<CInput>(c_input));
    auto c_output_pattern = tensor::as_output_spec<typename Atom::TAcc>(std::forward<COutput>(c_output));
    auto planning_invocation =
      Plan::prepare(config, m_value, n_value, k_value, a_pattern, b_pattern, c_input_pattern, c_output_pattern);
    const nint_t bytes = planning_invocation.required_workspace();
    auto scratch =
      parent.template worker_tensor<std::byte, Parallelism>(site_name, tensor::make_shape(meta::Any{bytes}));
    return PatternPrepared<Parallelism, decltype(m_value), decltype(n_value), decltype(k_value), decltype(a_pattern),
                           decltype(b_pattern), decltype(c_input_pattern), decltype(c_output_pattern),
                           decltype(scratch)>{config,
                                              m_value,
                                              n_value,
                                              k_value,
                                              std::move(a_pattern),
                                              std::move(b_pattern),
                                              std::move(c_input_pattern),
                                              std::move(c_output_pattern),
                                              std::move(scratch),
                                              bytes};
  }

  /** Fully bound state retained by this public Matmul implementation. */
  template <typename Invocation>
  class Prepared {
  public:
    using ResourceRequirements = typename Invocation::ResourceRequirements;

    VECOPS_INLINE Prepared(Invocation invocation, void* workspace, nint_t workspace_bytes)
      : invocation_(std::move(invocation))
      , workspace_(workspace)
      , workspace_bytes_(workspace_bytes) {
    }

    [[nodiscard]] VECOPS_INLINE nint_t workspace_bytes() const {
      return workspace_bytes_;
    }

    template <execution::ExecutionScope Scope>
    VECOPS_INLINE void operator()(Scope& scope) const {
      kernel::WorkspaceView workspace{workspace_, workspace_bytes_};
      execution::with_workspace(scope, workspace, [&](auto& rebound) VECOPS_INLINE_LAMBDA { invocation_(rebound); });
    }

    VECOPS_INLINE void operator()() const {
      kernel::WorkspaceView workspace{workspace_, workspace_bytes_};
      ExecutionSession execution{workspace};
      invocation_(execution);
    }

  private:
    Invocation invocation_;
    void* workspace_ = nullptr;
    nint_t workspace_bytes_ = 0;
  };

  /**
   * @brief Prepare and fully bind the single-C form into parent workspace.
   *
   * This stateful form is intended for repeated calls with identical operand
   * addresses and metadata.  It allocates exactly one sub-slot from `parent`;
   * execution itself performs no owning allocation or decision planning.
   */
  template <meta::ValueInput M, meta::ValueInput N, meta::ValueInput K, tensor::InputOperand A, tensor::InputOperand B,
            tensor::OutputOperand C>
  VECOPS_INLINE auto prepare(kernel::WorkspaceView& parent, M&& m, N&& n, K&& k, A&& a, B&& b, C&& c) const {
    static_assert(::vecops::matmul::kernel_family::ArchitectureFamily<KernelFamily>,
                  "prepared Matmul currently requires an architecture kernel family");
    auto c_output = tensor::as_output_spec<typename Atom::TAcc>(std::forward<C>(c));
    using Memory = typename decltype(c_output)::MemoryElement;
    auto c_input =
      tensor::input<typename Atom::TAcc>(c_output.tensor(), tensor::zeros_transform<typename Atom::TAcc, Memory>);
    auto invocation = Plan::prepare(config, std::forward<M>(m), std::forward<N>(n), std::forward<K>(k),
                                    std::forward<A>(a), std::forward<B>(b), std::move(c_input), std::move(c_output));
    const nint_t bytes = invocation.required_workspace();
    void* workspace = parent.allocate(bytes);
    return Prepared<decltype(invocation)>{std::move(invocation), workspace, bytes};
  }

  /**
   * Prepare the single-C form through the kernel-call workspace authority.
   * Enclosing operator scopes distinguish separate instances, so callers do
   * not calculate offsets or pass workspace byte counts.
   */
  template <typename WorkspaceAuthority, meta::ValueInput M, meta::ValueInput N, meta::ValueInput K,
            tensor::InputOperand A, tensor::InputOperand B, tensor::OutputOperand C>
    requires requires(WorkspaceAuthority& authority, nint_t bytes) { authority.bind(std::string_view{}, bytes); }
  VECOPS_INLINE auto prepare(WorkspaceAuthority& parent, M&& m, N&& n, K&& k, A&& a, B&& b, C&& c) const {
    static_assert(::vecops::matmul::kernel_family::ArchitectureFamily<KernelFamily>,
                  "prepared Matmul currently requires an architecture kernel family");
    auto c_output = tensor::as_output_spec<typename Atom::TAcc>(std::forward<C>(c));
    using Memory = typename decltype(c_output)::MemoryElement;
    auto c_input =
      tensor::input<typename Atom::TAcc>(c_output.tensor(), tensor::zeros_transform<typename Atom::TAcc, Memory>);
    auto invocation = Plan::prepare(config, std::forward<M>(m), std::forward<N>(n), std::forward<K>(k),
                                    std::forward<A>(a), std::forward<B>(b), std::move(c_input), std::move(c_output));
    const nint_t bytes = invocation.required_workspace();
    const auto slot = parent.bind("matmul_scratch", bytes);
    return Prepared<decltype(invocation)>{std::move(invocation), slot.replica(), bytes};
  }

  /** Prepare and fully bind the explicit accumulate form. */
  template <meta::ValueInput M, meta::ValueInput N, meta::ValueInput K, tensor::InputOperand A, tensor::InputOperand B,
            tensor::InputOperand CInput, tensor::OutputOperand COutput>
  VECOPS_INLINE auto prepare(kernel::WorkspaceView& parent, M&& m, N&& n, K&& k, A&& a, B&& b, CInput&& c_input,
                             COutput&& c_output) const {
    static_assert(::vecops::matmul::kernel_family::ArchitectureFamily<KernelFamily>,
                  "prepared Matmul currently requires an architecture kernel family");
    auto invocation =
      Plan::prepare(config, std::forward<M>(m), std::forward<N>(n), std::forward<K>(k), std::forward<A>(a),
                    std::forward<B>(b), std::forward<CInput>(c_input), std::forward<COutput>(c_output));
    const nint_t bytes = invocation.required_workspace();
    void* workspace = parent.allocate(bytes);
    return Prepared<decltype(invocation)>{std::move(invocation), workspace, bytes};
  }

  /** Prepare and bind the explicit accumulate form through WorkspaceContext. */
  template <typename WorkspaceAuthority, meta::ValueInput M, meta::ValueInput N, meta::ValueInput K,
            tensor::InputOperand A, tensor::InputOperand B, tensor::InputOperand CInput, tensor::OutputOperand COutput>
    requires requires(WorkspaceAuthority& authority, nint_t bytes) { authority.bind(std::string_view{}, bytes); }
  VECOPS_INLINE auto prepare(WorkspaceAuthority& parent, M&& m, N&& n, K&& k, A&& a, B&& b, CInput&& c_input,
                             COutput&& c_output) const {
    static_assert(::vecops::matmul::kernel_family::ArchitectureFamily<KernelFamily>,
                  "prepared Matmul currently requires an architecture kernel family");
    auto invocation =
      Plan::prepare(config, std::forward<M>(m), std::forward<N>(n), std::forward<K>(k), std::forward<A>(a),
                    std::forward<B>(b), std::forward<CInput>(c_input), std::forward<COutput>(c_output));
    const nint_t bytes = invocation.required_workspace();
    const auto slot = parent.bind("matmul_scratch", bytes);
    return Prepared<decltype(invocation)>{std::move(invocation), slot.replica(), bytes};
  }

  /// Workspace bytes for the single-C form; the C operand is adapted into
  /// the explicit-C form through a zero-valued accumulator input.
  template <meta::ValueInput M, meta::ValueInput N, meta::ValueInput K, tensor::InputOperand A, tensor::InputOperand B,
            tensor::OutputOperand C>
  VECOPS_INLINE nint_t required_workspace(M&& m, N&& n, K&& k, A&& a, B&& b, C&& c) const {
    // Adapt C = A*B^T to COut = CIn + A*B^T by dressing the output as a
    // zero-valued C input; the plan then sees the unified accumulate form.
    auto c_output = tensor::as_output_spec<typename Atom::TAcc>(std::forward<C>(c));
    using Memory = typename decltype(c_output)::MemoryElement;
    auto c_input =
      tensor::input<typename Atom::TAcc>(c_output.tensor(), tensor::zeros_transform<typename Atom::TAcc, Memory>);
    return Plan::required_workspace(config, std::forward<M>(m), std::forward<N>(n), std::forward<K>(k),
                                    std::forward<A>(a), std::forward<B>(b), std::move(c_input), std::move(c_output));
  }

  /// Workspace bytes for the explicit accumulate form
  /// `COutput = CInput + A*B^T`.
  template <meta::ValueInput M, meta::ValueInput N, meta::ValueInput K, tensor::InputOperand A, tensor::InputOperand B,
            tensor::InputOperand CInput, tensor::OutputOperand COutput>
  VECOPS_INLINE nint_t required_workspace(M&& m, N&& n, K&& k, A&& a, B&& b, CInput&& c_input,
                                          COutput&& c_output) const {
    return Plan::required_workspace(config, std::forward<M>(m), std::forward<N>(n), std::forward<K>(k),
                                    std::forward<A>(a), std::forward<B>(b), std::forward<CInput>(c_input),
                                    std::forward<COutput>(c_output));
  }

  /// Run `C = A*B^T` on an execution scope (see required_workspace for the
  /// C-to-accumulator adaptation).
  template <execution::ExecutionScope Scope, meta::ValueInput M, meta::ValueInput N, meta::ValueInput K,
            tensor::InputOperand A, tensor::InputOperand B, tensor::OutputOperand C>
  VECOPS_INLINE void operator()(Scope& scope, M&& m, N&& n, K&& k, A&& a, B&& b, C&& c) const {
    // Same zeros-transform adaptation as required_workspace above.
    auto c_output = tensor::as_output_spec<typename Atom::TAcc>(std::forward<C>(c));
    using Memory = typename decltype(c_output)::MemoryElement;
    auto c_input =
      tensor::input<typename Atom::TAcc>(c_output.tensor(), tensor::zeros_transform<typename Atom::TAcc, Memory>);
    Plan::run(scope, config, std::forward<M>(m), std::forward<N>(n), std::forward<K>(k), std::forward<A>(a),
              std::forward<B>(b), std::move(c_input), std::move(c_output));
  }

  /// Run `COutput = CInput + A*B^T` on an execution scope.
  template <execution::ExecutionScope Scope, meta::ValueInput M, meta::ValueInput N, meta::ValueInput K,
            tensor::InputOperand A, tensor::InputOperand B, tensor::InputOperand CInput, tensor::OutputOperand COutput>
  VECOPS_INLINE void operator()(Scope& scope, M&& m, N&& n, K&& k, A&& a, B&& b, CInput&& c_input,
                                COutput&& c_output) const {
    Plan::run(scope, config, std::forward<M>(m), std::forward<N>(n), std::forward<K>(k), std::forward<A>(a),
              std::forward<B>(b), std::forward<CInput>(c_input), std::forward<COutput>(c_output));
  }

  /// Single-C form on a raw workspace: wraps it in a temporary
  /// ExecutionSession and forwards.
  template <meta::ValueInput M, meta::ValueInput N, meta::ValueInput K, tensor::InputOperand A, tensor::InputOperand B,
            tensor::OutputOperand C>
  VECOPS_INLINE void operator()(kernel::WorkspaceView& workspace, M&& m, N&& n, K&& k, A&& a, B&& b, C&& c) const {
    ExecutionSession execution{workspace};
    (*this)(execution, std::forward<M>(m), std::forward<N>(n), std::forward<K>(k), std::forward<A>(a),
            std::forward<B>(b), std::forward<C>(c));
  }

  /// Explicit-C form on a raw workspace: wraps it in a temporary
  /// ExecutionSession and forwards.
  template <meta::ValueInput M, meta::ValueInput N, meta::ValueInput K, tensor::InputOperand A, tensor::InputOperand B,
            tensor::InputOperand CInput, tensor::OutputOperand COutput>
  VECOPS_INLINE void operator()(kernel::WorkspaceView& workspace, M&& m, N&& n, K&& k, A&& a, B&& b, CInput&& c_input,
                                COutput&& c_output) const {
    ExecutionSession execution{workspace};
    (*this)(execution, std::forward<M>(m), std::forward<N>(n), std::forward<K>(k), std::forward<A>(a),
            std::forward<B>(b), std::forward<CInput>(c_input), std::forward<COutput>(c_output));
  }
};

/// Factory: wrap a MatmulConfig into a Matmul operator object.
template <typename Config>
VECOPS_INLINE constexpr auto matmul(Config config) {
  return Matmul<Config>{std::move(config)};
}

/** Factory for a fully bound stateful single-C Matmul. */
template <typename Config, meta::ValueInput M, meta::ValueInput N, meta::ValueInput K, tensor::InputOperand A,
          tensor::InputOperand B, tensor::OutputOperand C>
VECOPS_INLINE auto matmul(kernel::WorkspaceView& workspace, Config config, M&& m, N&& n, K&& k, A&& a, B&& b, C&& c) {
  return Matmul<Config>{std::move(config)}.prepare(workspace, std::forward<M>(m), std::forward<N>(n),
                                                   std::forward<K>(k), std::forward<A>(a), std::forward<B>(b),
                                                   std::forward<C>(c));
}

} // namespace vecops::ops

#endif // VECOPS_OPS_MATMUL_H
