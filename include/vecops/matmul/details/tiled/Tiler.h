// SPDX-FileCopyrightText: 2026 HenryRenYz and vecops contributors
// SPDX-License-Identifier: MIT
//
//

#ifndef VECOPS_MATMUL_DETAILS_TILER_H
#define VECOPS_MATMUL_DETAILS_TILER_H

#include <type_traits>
#include <utility>

#include "vecops/matmul/Tiling.h"
#include "vecops/matmul/details/tiled/Accumulator.h"
#include "vecops/matmul/details/planning/FamilySelector.h"
#include "vecops/matmul/details/kernel/Kernel.h"
#include "vecops/matmul/details/tiled/LoopNest.h"
#include "vecops/matmul/details/tiled/OperandController.h"
#include "vecops/matmul/details/tiled/PolicyTraits.h"
#include "vecops/matmul/details/tiled/ProblemMapper.h"
#include "vecops/matmul/details/tiled/Scheduler.h"
#include "vecops/tensor/DataAccess.h"

/**
 * @file vecops/matmul/details/tiled/Tiler.h
 * @brief Generic cache-tiled matmul driver: MC x NC x KC blocking with K
 *        phases, operand packing, and split-K accumulation.
 *
 * `run_tiled_rank2` drives the whole portable tiled family for one rank-two
 * problem:
 *
 * 1. Resolve the cache tiling (explicit or from the platform cache
 *    hierarchy via `Tiling.h`) and allocate the backend's microkernel
 *    scratch.
 * 2. Decide split-K (a K loop that actually iterates) and place the
 *    split-K accumulator -- reuse of the output storage or a workspace
 *    tensor sized by the loop order (see `uses_output_accumulator_v` and
 *    the acc_m/acc_n inference below).
 * 3. Hoist explicitly/mandatorily full-K packed operands, otherwise place
 *    bounded panels at the compile-time-selected loop depth, then run the
 *    `LoopNest` phase-split traversal. Each leaf narrows all
 *    operands to the block (packed operands re-based per panel/K-group) and
 *    dispatches by K phase: single-block leaves go C -> C; first K block
 *    writes the accumulator; last adds into the real output; middle blocks
 *    accumulate in place.
 * 4. One workspace rewind at the end releases scratch, accumulator, and any
 *    hoisted packed copy together -- every allocation above sits between the
 *    single mark/rewind pair.
 *
 * `tiled_workspace_bytes` mirrors the allocation decisions of
 * `run_tiled_rank2` for callers that must pre-size a workspace; the two
 * must be kept in sync (see its comment).
 */

namespace vecops::matmul::details {

/** Operand views carried through the loop nest, with logical coordinates
 * corresponding to view origin zero. A panel-lifetime packing hook replaces
 * one view and advances its bases before entering the reuse loop. */
template <tensor::InputSpecLike ASpec, tensor::InputSpecLike BSpec>
struct TiledOperandState {
  [[no_unique_address]] ASpec a;
  [[no_unique_address]] BSpec b;
  nint_t a_spatial_origin = 0;
  nint_t a_k_origin = 0;
  nint_t b_spatial_origin = 0;
  nint_t b_k_origin = 0;
};

template <typename A, typename B>
VECOPS_ALWAYS_INLINE auto make_tiled_operand_state(
    const A& a, const B& b,
    nint_t a_spatial_origin = 0, nint_t a_k_origin = 0,
    nint_t b_spatial_origin = 0, nint_t b_k_origin = 0) {
  return TiledOperandState<std::remove_cvref_t<A>, std::remove_cvref_t<B>>{
      a, b, a_spatial_origin, a_k_origin,
      b_spatial_origin, b_k_origin};
}

/// Narrow a C input spec to the block's rows/columns.
template <tensor::InputSpecLike Spec,
          meta::ValueType M, meta::ValueType N>
VECOPS_INLINE auto narrow_c_input(
    const Spec& spec, nint_t m_origin, M m,
    nint_t n_origin, N n) {
  auto rows = tensor::narrow_view<0>(spec, m_origin, m);
  return tensor::narrow_view<1>(rows, n_origin, n);
}

/// Narrow a C output spec to the block's rows/columns.
template <tensor::OutputSpecLike Spec,
          meta::ValueType M, meta::ValueType N>
VECOPS_INLINE auto narrow_c_output(
    const Spec& spec, nint_t m_origin, M m,
    nint_t n_origin, N n) {
  auto rows = tensor::narrow_view<0>(spec, m_origin, m);
  return tensor::narrow_view<1>(rows, n_origin, n);
}

/** Clamp an extent to a tile while preserving constants, alignment, and
 * finite bounds. The identical-type branch avoids the generic-scalar / Meta
 * overload ambiguity without erasing either operand's metadata. */
template <meta::ValueType Extent, meta::ValueType Tile>
VECOPS_ALWAYS_INLINE constexpr auto bounded_extent(
    Extent extent, Tile tile) {
  if constexpr (std::same_as<
                    std::remove_cvref_t<Extent>,
                    std::remove_cvref_t<Tile>>) {
    // Avoid the generic-scalar / Meta overload ambiguity for identical Value
    // types while retaining that exact type (including Dynamic constraints).
    return tile < extent ? tile : extent;
  } else {
    return vecops::min(extent, tile);
  }
}

/** Split-K accumulator extent for one spatial axis. An axis outside K only
 * needs one tile stripe; an axis inside K must keep its full logical extent.
 * Value-aware min preserves constants, alignment, and finite bounds. */
template <Axis Target, typename Order,
          meta::ValueType Extent, meta::ValueType Tile>
VECOPS_ALWAYS_INLINE constexpr auto accumulator_axis_extent(
    Extent extent, Tile tile) {
  if constexpr (!axis_precedes_k_v<Target, Order>) {
    return extent;
  } else {
    return bounded_extent(extent, tile);
  }
}

/** Establish a bounded packed panel after its M/K or N/K dependencies have
 * been selected and immediately before the operand's innermost reuse loop. */
template <typename AtomT, typename AController, typename BController,
          bool PackAOutsidePanel, bool PackBOutsidePanel,
          execution::ExecutionScope Scope>
struct PanelLifetimePackingHook {
  Scope* scope;

  template <int Depth, Axis, typename Context, typename Phase,
            typename State, typename Continue>
  VECOPS_ALWAYS_INLINE void operator()(
      const Context& block, const Phase&, const State& state,
      Continue&& continuation) const {
    // A panel becomes fully determined by M and K; with N innermost it can
    // be packed here once and reused by every N block.
    if constexpr (Depth == 1 && PackAOutsidePanel) {
      VECOPS_ASSERT(
          block.m_origin >= state.a_spatial_origin &&
              block.k_origin >= state.a_k_origin,
          "matmul A panel origin precedes its carried view");
      auto panel = narrow_input<AtomT, Operand::A>(
          state.a, block.m_origin - state.a_spatial_origin, block.m,
          block.k_origin - state.a_k_origin, block.k);
      AController::with_panel(
          *scope, panel, [&](const auto& active_a) VECOPS_INLINE_LAMBDA {
        auto next = make_tiled_operand_state(
            active_a, state.b, block.m_origin, block.k_origin,
            state.b_spatial_origin, state.b_k_origin);
        std::forward<Continue>(continuation)(next);
      });
    } else if constexpr (Depth == 1 && PackBOutsidePanel) {
      // Symmetric case: B depends on N/K and is reused by innermost M.
      VECOPS_ASSERT(
          block.n_origin >= state.b_spatial_origin &&
              block.k_origin >= state.b_k_origin,
          "matmul B panel origin precedes its carried view");
      auto panel = narrow_input<AtomT, Operand::B>(
          state.b, block.n_origin - state.b_spatial_origin, block.n,
          block.k_origin - state.b_k_origin, block.k);
      BController::with_panel(
          *scope, panel, [&](const auto& active_b) VECOPS_INLINE_LAMBDA {
        auto next = make_tiled_operand_state(
            state.a, active_b, state.a_spatial_origin, state.a_k_origin,
            block.n_origin, block.k_origin);
        std::forward<Continue>(continuation)(next);
      });
    } else {
      std::forward<Continue>(continuation)(state);
    }
  }
};

/**
 * @brief Execute one rank-two problem through the generic tiled family.
 *
 * Workspace layout between the single mark/rewind pair: microkernel scratch
 * (if the backend wants any), the split-K accumulator (only when splitting
 * and not reusing the output), and the statically selected packed operand
 * copies.
 */
template <typename Config, typename Implementation,
          execution::ExecutionScope Scope,
          meta::ValueType M, meta::ValueType N, meta::ValueType K,
          tensor::InputSpecLike ASpec, tensor::InputSpecLike BSpec,
          tensor::InputSpecLike CInputSpec,
          tensor::OutputSpecLike COutputSpec>
VECOPS_INLINE void run_tiled_rank2(
    Scope& scope, const Config& config, M m, N n, K k,
    const ASpec& a, const BSpec& b,
    const CInputSpec& c_input, const COutputSpec& c_output) {
  using AtomT = typename Config::Atom;
  using Order = resolved_loop_order_t<Config>;
  constexpr auto SpatialTraversal = spatial_traversal_order_v<Order>;
  using Tuning = typename Config::GenericTuning;
  using PackingTuning = config_packing_tuning_t<Config>;
  using APlacement = typename PackingTuning::APacking;
  using BPlacement = typename PackingTuning::BPacking;
  using APacking = ResolvedInternalPackingPolicy<
      typename Tuning::APacking, APlacement, Operand::A, Order>;
  using BPacking = ResolvedInternalPackingPolicy<
      typename Tuning::BPacking, BPlacement, Operand::B, Order>;
  using AController = OperandController<
      Operand::A, APacking, Order, AtomT>;
  using BController = OperandController<
      Operand::B, BPacking, Order, AtomT>;
  static_assert((ASpec::InputTensor::Ndim == 2 ||
                 ::vecops::matmul::is_packed_layout<
                     AtomT, Operand::A, typename ASpec::InputLayout>()) &&
                (BSpec::InputTensor::Ndim == 2 ||
                 ::vecops::matmul::is_packed_layout<
                     AtomT, Operand::B, typename BSpec::InputLayout>()) &&
                CInputSpec::InputTensor::Ndim == 2 &&
                COutputSpec::OutputTensor::Ndim == 2,
                "generic matmul Tiler currently consumes rank-two leaves");
  static_assert(
      APlacement::prepared_input != PreparedInputRequirement::required ||
          is_packed_spec_v<AtomT, Operand::A, ASpec>,
      "matmul A packing policy requires caller-prepared packed input");
  static_assert(
      BPlacement::prepared_input != PreparedInputRequirement::required ||
          is_packed_spec_v<AtomT, Operand::B, BSpec>,
      "matmul B packing policy requires caller-prepared packed input");

  constexpr bool PackAOutsidePanel = uses_panel_lifetime_packing_v<
      APacking, APlacement, Operand::A, Order> &&
      should_pack_v<APacking, AtomT, Operand::A, ASpec>;
  constexpr bool PackBOutsidePanel = uses_panel_lifetime_packing_v<
      BPacking, BPlacement, Operand::B, Order> &&
      should_pack_v<BPacking, AtomT, Operand::B, BSpec>;
  static_assert(!(PackAOutsidePanel && PackBOutsidePanel),
                "only the innermost reuse axis owns a panel-lifetime hook");
  constexpr bool AllowAInside = allows_inside_packing_v<APlacement>;
  constexpr bool AllowBInside = allows_inside_packing_v<BPlacement>;
  constexpr bool UseAWhole = uses_whole_operand_packing_v<
      typename Tuning::APacking, APlacement,
      APacking, Operand::A, Order>;
  constexpr bool UseBWhole = uses_whole_operand_packing_v<
      typename Tuning::BPacking, BPlacement,
      BPacking, Operand::B, Order>;

  const auto problem = ProblemMapper::map(m, n, k);
  using ProblemM = std::remove_cvref_t<decltype(problem.m)>;
  using ProblemN = std::remove_cvref_t<decltype(problem.n)>;
  using ProblemK = std::remove_cvref_t<decltype(problem.k)>;

  auto tiling = resolve_cache_tiling(config);
  using ResolvedTiling = std::remove_cvref_t<decltype(tiling)>;
  constexpr nint_t KR = std::remove_cvref_t<decltype(AtomT::K_R)>::value;
  VECOPS_ASSERT(static_cast<nint_t>(tiling.kc) % KR == 0,
                "KC must be a multiple of the Atom K step");
  // A caller-prepared or whole-operation packed view is rebased directly at
  // cache-block origins. Validate every origin-generating tile before any C
  // write, so an invalid tuning fails atomically rather than after block 0.
  auto validate_global_packed_tiling = [&]<Operand Side, bool Packed,
      typename Spatial, typename SpatialTile>(
          Spatial spatial, SpatialTile spatial_tile) {
    if constexpr (Packed) {
      constexpr Axis SpatialAxis = Side == Operand::A ? Axis::M : Axis::N;
      constexpr bool GeneratesSpatialLoop = LoopNest<
          Order, ResolvedTiling>::template generates_loop<
              SpatialAxis, ProblemM, ProblemN, ProblemK>;
      constexpr bool GeneratesKLoop = LoopNest<
          Order, ResolvedTiling>::template generates_loop<
              Axis::K, ProblemM, ProblemN, ProblemK>;
      VECOPS_CHECK(static_cast<nint_t>(spatial_tile) > 0 &&
                       static_cast<nint_t>(tiling.kc) > 0,
                   "packed matmul cache tiles must be positive");
      if constexpr (GeneratesSpatialLoop) {
        if (static_cast<nint_t>(spatial) >
            static_cast<nint_t>(spatial_tile)) {
          VECOPS_CHECK(
              static_cast<nint_t>(spatial_tile) %
                      (packed_spatial_panel<AtomT, Side>()) ==
                  0,
              "packed matmul cache tile has a misaligned spatial origin");
        }
      }
      if constexpr (GeneratesKLoop) {
        if (static_cast<nint_t>(problem.k) >
            static_cast<nint_t>(tiling.kc)) {
          VECOPS_CHECK(
              static_cast<nint_t>(tiling.kc) %
                      (packed_k_step_v<AtomT, Side>) ==
                  0,
              "packed matmul cache tile has a misaligned K origin");
        }
      }
    }
  };
  validate_global_packed_tiling.template operator()<
      Operand::A,
      is_packed_spec_v<AtomT, Operand::A, ASpec> ||
          (UseAWhole &&
           should_pack_v<APacking, AtomT, Operand::A, ASpec>)>(
          problem.m, tiling.mc);
  validate_global_packed_tiling.template operator()<
      Operand::B,
      is_packed_spec_v<AtomT, Operand::B, BSpec> ||
          (UseBWhole &&
           should_pack_v<BPacking, AtomT, Operand::B, BSpec>)>(
          problem.n, tiling.nc);
  auto& workspace = scope.workspace_view();
  const auto operation_mark = workspace.mark();
  void* scratch = nullptr;
  const nint_t scratch_bytes =
      kernel::matmul_implementation::scratch_bytes<Implementation>();
  if (scratch_bytes != 0) scratch = workspace.allocate(scratch_bytes, 64);

  const nint_t logical_k = static_cast<nint_t>(problem.k);
  // Split-K accumulator shape inference:
  //
  // - split_k needs a K loop that actually iterates (GeneratesKLoop) *and* a
  //   K extent beyond one KC block; otherwise every leaf is single-phase and
  //   no accumulator exists at all.
  // - acc_m/acc_n: when the M (or N) axis precedes K in the traversal order,
  //   one M-block's entire K accumulation completes before the next M block
  //   starts (the axis is consumed block-by-block and each block's partials
  //   are used up), so the accumulator only needs one tile stripe:
  //   min(logical, tile). When K precedes the axis, every block of that axis
  //   is revisited once per K block, so the whole axis's partial sums must
  //   stay resident simultaneously: full logical extent.
  constexpr bool GeneratesKLoop = LoopNest<
      Order, decltype(tiling)>::template generates_loop<
          Axis::K, ProblemM, ProblemN, ProblemK>;
  const bool split_k = GeneratesKLoop &&
      logical_k > static_cast<nint_t>(tiling.kc);
  using Acc = typename AtomT::TAcc;
  constexpr bool OutputAcc = uses_output_accumulator_v<Config, COutputSpec>;
  const auto acc_m = accumulator_axis_extent<Axis::M, Order>(
      problem.m, tiling.mc);
  const auto acc_n = accumulator_axis_extent<Axis::N, Order>(
      problem.n, tiling.nc);
  // Three accumulator sources: (1) OutputAcc reuses the output tensor's own
  // storage (full logical extent, no allocation); (2) split_k without
  // output reuse allocates the workspace tensor sized above; (3) no split
  // needs nothing -- single-phase leaves write C directly.
  Acc* acc_data = nullptr;
  if (split_k && !OutputAcc) {
    acc_data = static_cast<Acc*>(workspace.allocate(
        static_cast<nint_t>(acc_m) * static_cast<nint_t>(acc_n) *
            static_cast<nint_t>(sizeof(Acc)),
        64));
  }
  auto acc_tensor = [&]() {
    if constexpr (OutputAcc) {
      return output_acc_storage<Acc>(c_output, problem.m, problem.n);
    } else {
      return tensor::make_tensor(
          acc_data, tensor::make_layout(
              tensor::make_shape(acc_m, acc_n)));
    }
  }();
  auto acc_input = tensor::input<Acc>(acc_tensor);
  auto acc_output = tensor::output<Acc>(acc_tensor);

  auto run_loop = [&](const auto& whole_a, const auto& whole_b) {
    const auto initial_state = make_tiled_operand_state(whole_a, whole_b);
    auto run_leaf = [&](const auto& block, auto phase, const auto& operands) {
        VECOPS_ASSERT(
            block.m_origin >= operands.a_spatial_origin &&
                block.k_origin >= operands.a_k_origin &&
                block.n_origin >= operands.b_spatial_origin &&
                block.k_origin >= operands.b_k_origin,
            "matmul block origin precedes its carried operand view");
        auto a_block = narrow_input<AtomT, Operand::A>(
            operands.a, block.m_origin - operands.a_spatial_origin, block.m,
            block.k_origin - operands.a_k_origin, block.k);
        auto b_block = narrow_input<AtomT, Operand::B>(
            operands.b, block.n_origin - operands.b_spatial_origin, block.n,
            block.k_origin - operands.b_k_origin, block.k);
        auto c_input_block = narrow_c_input(
            c_input, block.m_origin, block.m,
            block.n_origin, block.n);
        auto c_output_block = narrow_c_output(
            c_output, block.m_origin, block.m,
            block.n_origin, block.n);
        AController::template with_panel<AllowAInside>(
            scope, a_block, [&](const auto& active_a) {
          BController::template with_panel<AllowBInside>(
              scope, b_block, [&](const auto& active_b) {
            if constexpr (static_kernel_phase_v<decltype(phase)>) {
              static_assert(decltype(phase)::first_k &&
                            decltype(phase)::last_k,
                            "only an unsplit K block has a static phase");
              // Single K block: no accumulator round trip, C in -> C out.
              Scheduler<
                  AtomT, typename Config::SchedulerPolicy, Implementation,
                  SpatialTraversal>::run(
                      scope, block.m, block.n, block.k,
                      active_a, active_b,
                      c_input_block, c_output_block, scratch);
              return;
            }
            // Output-backed accumulators are a full logical M x N view and
            // therefore always use the real block coordinates.  Only a
            // dedicated workspace accumulator may be stripe-shaped: an axis
            // preceding K is then rebased to zero while a whole-axis
            // dimension retains the logical block origin.
            const nint_t acc_m_origin =
                accumulator_block_origin<OutputAcc, Axis::M, Order>(
                    block.m_origin);
            const nint_t acc_n_origin =
                accumulator_block_origin<OutputAcc, Axis::N, Order>(
                    block.n_origin);
            auto acc_input_block = narrow_c_input(
                acc_input, acc_m_origin, block.m,
                acc_n_origin, block.n);
            auto acc_output_block = narrow_c_output(
                acc_output, acc_m_origin, block.m,
                acc_n_origin, block.n);
            // A real K loop uses one runtime route type for every phase. Both
            // C endpoint pairs cross the backend boundary, so first/middle/
            // last share the same A/B compute instantiation.
            const kernel::matmul_details::DynamicAccumulatorRoute route{
                !phase.first_k, !phase.last_k};
            Scheduler<
                AtomT, typename Config::SchedulerPolicy,
                Implementation, SpatialTraversal>::run_phased(
                    scope, block.m, block.n, block.k,
                    active_a, active_b,
                    c_input_block, c_output_block,
                    acc_input_block, acc_output_block,
                    route, scratch);
          });
        });
      };
    if constexpr (!PackAOutsidePanel && !PackBOutsidePanel) {
      // Preserve the historical loop recursion for raw/prepared/default
      // paths. The stateful hook is instantiated only when it creates a new
      // panel at this call site.
      LoopNest<Order, decltype(tiling)>::run_phased(
          problem.m, problem.n, problem.k, tiling,
          [&](const auto& block, auto phase) {
            run_leaf(block, phase, initial_state);
          });
    } else {
      PanelLifetimePackingHook<
          AtomT, AController, BController,
          PackAOutsidePanel, PackBOutsidePanel, Scope> packing_hook{&scope};
      LoopNest<Order, decltype(tiling)>::run_phased_with_state(
          problem.m, problem.n, problem.k,
          tiling, initial_state, packing_hook, run_leaf);
    }
  };
  // Whole-operand packing is a compile-time representation decision. This
  // keeps the call site to one raw/packed shape; optional automatic cases use
  // an allowed bounded panel rather than branching on runtime cache sizes.
  AController::template with_whole_operand<UseAWhole>(
      scope, a, [&](const auto& whole_a) {
        BController::template with_whole_operand<UseBWhole>(
            scope, b, [&](const auto& whole_b) {
              run_loop(whole_a, whole_b);
            });
      });
  // A single rewind releases everything allocated since operation_mark:
  // scratch, the split-K accumulator, and any full-K packed copies.
  workspace.rewind(operation_mark);
}

/**
 * @brief Workspace bytes `run_tiled_rank2` may allocate for this problem.
 *
 * Mirrors the allocation logic of `run_tiled_rank2` term by term
 * (microkernel scratch, split-K accumulator, packed operand copies at their
 * resolved extent) -- the two functions must be kept in sync: any allocation
 * added to the run path needs a matching term here, or pre-sized workspaces
 * will overflow at run time. Each packed-copy term adds the same `+ 63`
 * slack the run path relies on (workspace.allocate rounds up to 64-byte
 * alignment, so one term may consume up to 63 bytes more than its payload).
 */
template <typename Config, typename Implementation,
          meta::ValueType M, meta::ValueType N, meta::ValueType K,
          tensor::InputSpecLike ASpec, tensor::InputSpecLike BSpec,
          tensor::OutputSpecLike COutputSpec>
VECOPS_INLINE nint_t tiled_workspace_bytes(
    const Config& config, M m, N n, K k,
    const ASpec& a, const BSpec& b, const COutputSpec&) {
  using AtomT = typename Config::Atom;
  auto tiling = resolve_cache_tiling(config);
  const auto problem = ProblemMapper::map(m, n, k);
  using ProblemM = std::remove_cvref_t<decltype(problem.m)>;
  using ProblemN = std::remove_cvref_t<decltype(problem.n)>;
  using ProblemK = std::remove_cvref_t<decltype(problem.k)>;
  nint_t bytes = kernel::matmul_implementation::scratch_bytes<Implementation>();
  const nint_t logical_k = static_cast<nint_t>(problem.k);
  using Order = resolved_loop_order_t<Config>;
  using Tuning = typename Config::GenericTuning;
  using PackingTuning = config_packing_tuning_t<Config>;
  using APlacement = typename PackingTuning::APacking;
  using BPlacement = typename PackingTuning::BPacking;
  using APacking = ResolvedInternalPackingPolicy<
      typename Tuning::APacking, APlacement, Operand::A, Order>;
  using BPacking = ResolvedInternalPackingPolicy<
      typename Tuning::BPacking, BPlacement, Operand::B, Order>;
  static_assert(
      APlacement::prepared_input != PreparedInputRequirement::required ||
          is_packed_spec_v<AtomT, Operand::A, ASpec>,
      "matmul A packing policy requires caller-prepared packed input");
  static_assert(
      BPlacement::prepared_input != PreparedInputRequirement::required ||
          is_packed_spec_v<AtomT, Operand::B, BSpec>,
      "matmul B packing policy requires caller-prepared packed input");
  constexpr bool GeneratesKLoop = LoopNest<
      Order, decltype(tiling)>::template generates_loop<
          Axis::K, ProblemM, ProblemN, ProblemK>;
  if (GeneratesKLoop && logical_k > static_cast<nint_t>(tiling.kc) &&
      !uses_output_accumulator_v<Config, COutputSpec>) {
    // Same stripe-vs-whole-axis accumulator inference as run_tiled_rank2.
    const auto acc_m = accumulator_axis_extent<Axis::M, Order>(
        problem.m, tiling.mc);
    const auto acc_n = accumulator_axis_extent<Axis::N, Order>(
        problem.n, tiling.nc);
    // +63: 64-byte allocation-alignment slack (workspace.allocate rounds
    // the payload up to a 64-byte boundary).
    bytes += static_cast<nint_t>(acc_m) * static_cast<nint_t>(acc_n) *
        static_cast<nint_t>(sizeof(typename AtomT::TAcc)) + 63;
  }
  const auto panel_m = bounded_extent(problem.m, tiling.mc);
  const auto panel_n = bounded_extent(problem.n, tiling.nc);
  const auto panel_k = bounded_extent(problem.k, tiling.kc);
  if constexpr ((allows_inside_packing_v<APlacement> ||
                 allows_outside_packing_v<APlacement>) &&
                should_pack_v<APacking, AtomT, Operand::A, ASpec>) {
    // Packing footprint follows the resolved extent: whole operand for
    // full_k, one MC x KC panel for cache_k (see OperandController).
    constexpr bool Whole = uses_whole_operand_packing_v<
        typename Tuning::APacking, APlacement,
        APacking, Operand::A, Order>;
    constexpr bool Panel = !Whole && (allows_inside_packing_v<APlacement> ||
        uses_panel_lifetime_packing_v<
            APacking, APlacement, Operand::A, Order>);
    auto add_packing_bytes = [&](auto packing_m, auto packing_k) {
      auto layout = packed_layout<AtomT, Operand::A>(tensor::make_layout(
          tensor::make_shape(packing_m, packing_k)));
      bytes += tensor::numel(layout) *
          static_cast<nint_t>(sizeof(typename AtomT::TA)) + 63;
    };
    if constexpr (Whole) {
      add_packing_bytes(problem.m, problem.k);
    } else if constexpr (Panel) {
      add_packing_bytes(panel_m, panel_k);
    }
  }
  if constexpr ((allows_inside_packing_v<BPlacement> ||
                 allows_outside_packing_v<BPlacement>) &&
                should_pack_v<BPacking, AtomT, Operand::B, BSpec>) {
    // Packing footprint follows the resolved extent: whole operand for
    // full_k, one NC x KC panel for cache_k.
    constexpr bool Whole = uses_whole_operand_packing_v<
        typename Tuning::BPacking, BPlacement,
        BPacking, Operand::B, Order>;
    constexpr bool Panel = !Whole && (allows_inside_packing_v<BPlacement> ||
        uses_panel_lifetime_packing_v<
            BPacking, BPlacement, Operand::B, Order>);
    auto add_packing_bytes = [&](auto packing_n, auto packing_k) {
      auto layout = packed_layout<AtomT, Operand::B>(tensor::make_layout(
          tensor::make_shape(packing_n, packing_k)));
      bytes += tensor::numel(layout) *
          static_cast<nint_t>(sizeof(typename AtomT::TB)) + 63;
    };
    if constexpr (Whole) {
      add_packing_bytes(problem.n, problem.k);
    } else if constexpr (Panel) {
      add_packing_bytes(panel_n, panel_k);
    }
  }
  return bytes;
}

} // namespace vecops::matmul::details

#endif // VECOPS_MATMUL_DETAILS_TILER_H
