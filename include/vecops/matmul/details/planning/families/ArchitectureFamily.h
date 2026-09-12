//
// Copyright (c) vecops contributors.
//

#ifndef VECOPS_MATMUL_DETAILS_FAMILIES_ARCHITECTURE_FAMILY_H
#define VECOPS_MATMUL_DETAILS_FAMILIES_ARCHITECTURE_FAMILY_H

/**
 * @file vecops/matmul/details/planning/families/ArchitectureFamily.h
 * @brief The architecture-family invocation: one matmul call's complete
 *        static plan, runtime gates, and execution paths.
 *
 * An ArchitectureFamilyInvocation owns the operand specs of one
 * `C[M,N] = C_in + A[M,K] * B[N,K]^T` call (optionally batched by leading
 * dimensions) and answers two kinds of questions:
 *
 * - Statically, which transformations are legal and profitable for these
 *   metadata types.  The constexpr variables on the class decide, per
 *   instantiation, whether online packing applies (AutoPackOperand plus the
 *   Meta-bound cost model), which batch-dimension flatten shapes are candidates
 *   (BatchRows/BatchColumns*Candidate), and whether packing and multiply
 *   can fuse into one streaming region (SingleStreamingAutoPackRegion).
 * - Dynamically, whether a batch flatten candidate matches the concrete
 *   strides.  use_auto_packing_for() is normally folded from Const/Dynamic
 *   bounds.  Only unconstrained native-AMX B keeps one raw-vs-packed runtime
 *   decision; A never joins it into the former four-way Cartesian fan-out.
 *
 * Execution paths, in the order execute() tries them:
 *
 * 1. Packed-A flattened batch rows: a raw batched problem whose B was
 *    prepacked (explicitly, or online by a compile-time decision) may pack
 *    the flattened [batch*M, K] A once and run a single packed problem.
 * 2. Online packing: A and B are costed independently at compile time.  Exact
 *    metadata uses the concrete cost model; bounded Dynamic metadata uses its
 *    safe corner; unconstrained native AMX may choose raw vs packed-B, while
 *    SME and conversion-eliding packs default packed. Caller-prepared AMX B
 *    makes native A packing monotone in N reuse, so a sufficient N/K lower
 *    corner fixes A to the packed path even when M is unbounded. A shared
 *    rank-three A or B is packed once while an independent side reuses one
 *    leaf-sized staging buffer across batches.
 * 3. Bounded WholeProblem B panel: large rank-two native-B products may pack
 *    one full-K N panel and consume it across M before advancing N. This fills
 *    the lifetime gap between whole-B packing and per-microkernel packing.
 * 4. Batch-columns flatten: a shared A with M == 1 collapses to
 *    [1, batch*N].
 * 5. Batch-rows flatten: a shared B collapses to one [batch*M, K] product.
 * 6. The ordinary traversal: loop the leading batch dimensions and run one
 *    leaf problem per item (under with_matmul_configuration for AMX).
 *
 * Three sibling planners collaborate as friends and read the private member
 * specs directly, each owning one concern: ArchitectureBatchPlanner
 * (rank-three analysis and rank-two views), ArchitecturePackingPlanner
 * (packed layouts, byte accounting, streaming ownership), and
 * ArchitectureWorkspacePlanner (the required_workspace total).
 * make_matmul_invocation() is the construction entry used by
 * planning/FamilyPlan.h.
 */

#include <algorithm>
#include <limits>
#include <type_traits>
#include <utility>

#include "vecops/Assertion.h"
#include "vecops/execution/ExecutionSession.h"
#include "vecops/execution/details/arm/Resources.h"
#include "vecops/kernel/Loop.h"
#include "vecops/matmul/Config.h"
#include "vecops/matmul/Packing.h"
#include "vecops/matmul/details/planning/FamilySelector.h"
#include "vecops/matmul/details/kernel/Kernel.h"
#include "vecops/matmul/details/planning/Implementation.h"
#include "vecops/matmul/details/planning/orientation/Backend.h"
#include "vecops/matmul/details/packing/Plan.h"
#include "vecops/matmul/details/planning/families/ArchitectureBatchPlanner.h"
#include "vecops/matmul/details/planning/families/ArchitecturePackingPlanner.h"
#include "vecops/matmul/details/planning/families/ArchitectureWorkspacePlanner.h"
#include "vecops/tensor/DataAccess.h"

namespace vecops::matmul::details {

/** Carry opt-in packing policy through the existing dispatch type.
 *
 * Keeping the policy off InputSpec is important: tensor access and packing
 * APIs deliberately recognize their exact spec types.  The default Any/Any
 * case also keeps the historical FamilyDispatch type exactly unchanged, so
 * existing call sites retain the same template identity and generated code.
 */
template <typename Dispatch,
          ::vecops::matmul::OperandPackingPolicyType APacking,
          ::vecops::matmul::OperandPackingPolicyType BPacking>
struct PackingPolicyFamilyDispatch : Dispatch {
  using APackingPlacement = APacking;
  using BPackingPlacement = BPacking;
};

template <typename Dispatch, typename = void>
struct DispatchPackingPlacement {
  using A = ::vecops::matmul::packing_policy::Any;
  using B = ::vecops::matmul::packing_policy::Any;
};

template <typename Dispatch>
struct DispatchPackingPlacement<
    Dispatch,
    std::void_t<typename Dispatch::APackingPlacement,
                typename Dispatch::BPackingPlacement>> {
  using A = typename Dispatch::APackingPlacement;
  using B = typename Dispatch::BPackingPlacement;
};

template <typename Dispatch>
using dispatch_a_packing_placement_t =
    typename DispatchPackingPlacement<Dispatch>::A;

template <typename Dispatch>
using dispatch_b_packing_placement_t =
    typename DispatchPackingPlacement<Dispatch>::B;

template <typename Dispatch, typename APacking, typename BPacking>
using attach_packing_policy_t = std::conditional_t<
    std::same_as<APacking, ::vecops::matmul::packing_policy::Any> &&
        std::same_as<BPacking, ::vecops::matmul::packing_policy::Any>,
    Dispatch, PackingPolicyFamilyDispatch<Dispatch, APacking, BPacking>>;

/** Whether an explicitly configured logical loop order visits N before M.
 * WholeProblem has no cache-K loop, so this relative spatial order is the
 * only part of GenericTuning::LoopOrder that reaches its leaf traversal.
 * Automatic deliberately stays false here: backend automatic selection is a
 * separate profitability decision and must not silently inherit the generic
 * tiler's architecture default. */
template <typename Config>
inline constexpr bool explicit_logical_n_major_v = [] {
  using Order = typename Config::GenericTuning::LoopOrder;
  if constexpr (::vecops::matmul::LoopOrderType<Order>) {
    if constexpr (Order::first == ::vecops::matmul::Axis::N) return true;
    if constexpr (Order::first == ::vecops::matmul::Axis::M) return false;
    return Order::second == ::vecops::matmul::Axis::N;
  } else {
    return false;
  }
}();

template <typename Config, meta::ValueType M, meta::ValueType N,
          meta::ValueType K>
inline constexpr bool automatic_physical_n_major_v = [] {
  using MV = std::remove_cvref_t<M>;
  using NV = std::remove_cvref_t<N>;
  using KV = std::remove_cvref_t<K>;
  if constexpr (!std::same_as<
                    SelectedImplementation<typename Config::Atom>,
                    kernel::matmul_implementation::AMX>) {
    return false;
  } else if constexpr (!meta::has_upper_bound_v<MV> ||
                !meta::has_lower_bound_v<NV> ||
                !meta::has_upper_bound_v<KV>) {
    return false;
  } else {
    constexpr nint_t MaxM = meta::upper_bound_v<MV>;
    constexpr nint_t MinN = meta::lower_bound_v<NV>;
    constexpr nint_t MaxK = meta::upper_bound_v<KV>;
    constexpr nint_t MaxAElements = (1024 * 1024) /
        static_cast<nint_t>(sizeof(typename Config::Atom::TA));
    return MaxM > 0 && MinN > 0 && MaxK >= 256 &&
        MaxK <= MaxAElements && MaxM <= MinN / 4 &&
        MaxM <= MaxAElements / MaxK;
  }
}();

template <typename Config, bool Swapped, typename Dispatch,
          meta::ValueType PhysicalM, meta::ValueType PhysicalN,
          meta::ValueType PhysicalK>
using attach_spatial_traversal_t = std::conditional_t<
    ::vecops::matmul::LoopOrderType<
        typename Config::GenericTuning::LoopOrder>,
    std::conditional_t<
        explicit_logical_n_major_v<Config> != Swapped,
        kernel::matmul_details::NMajorFamilyDispatch<Dispatch>, Dispatch>,
    std::conditional_t<
        automatic_physical_n_major_v<
            Config, PhysicalM, PhysicalN, PhysicalK>,
        kernel::matmul_details::NMajorFamilyDispatch<Dispatch>,
        kernel::matmul_details::AutomaticTraversalFamilyDispatch<Dispatch>>>;

template <::vecops::matmul::Atom Atom,
          typename TilePolicy,
          typename FamilyDispatch,
          bool SwapsAB,
          meta::ValueType MExtent,
          meta::ValueType NExtent,
          meta::ValueType KExtent,
          typename ASpec, typename BSpec,
          typename CInputSpec, typename COutputSpec>
class ArchitectureFamilyInvocation {
public:
  using APackingPlacement = dispatch_a_packing_placement_t<FamilyDispatch>;
  using BPackingPlacement = dispatch_b_packing_placement_t<FamilyDispatch>;
  using AtomType = Atom;
  using TilePolicyType = TilePolicy;
  using MExtentType = MExtent;
  using NExtentType = NExtent;
  using KExtentType = KExtent;
  using APackingPolicy = APackingPlacement;
  using BPackingPolicy = BPackingPlacement;
  using Implementation = SelectedImplementation<Atom>;
  using ResourceRequirements =
      kernel::matmul_implementation::resource_requirements_t<Implementation>;
  static constexpr int ProblemRank =
      kernel::matmul_implementation::problem_rank_v<Implementation>;
  static constexpr int Rank = COutputSpec::OutputTensor::Ndim;
  static_assert(
      !FamilyDispatch::required ||
          std::same_as<typename FamilyDispatch::Family,
                       ::vecops::matmul::kernel_family::General> ||
          Rank == ProblemRank,
      "required specialized matmul family supports one leaf problem only");
  using EffectiveFamilyDispatch = std::conditional_t<
      Rank == ProblemRank ||
          std::same_as<typename FamilyDispatch::Family,
                       ::vecops::matmul::kernel_family::General>,
      FamilyDispatch,
      attach_packing_policy_t<
          ::vecops::matmul::details::AutomaticFamilyDispatch,
          APackingPlacement, BPackingPlacement>>;
  static constexpr bool swaps_ab = SwapsAB;

  VECOPS_INLINE ArchitectureFamilyInvocation(
      MExtent m, NExtent n, KExtent k,
      ASpec a, BSpec b, CInputSpec c_input, COutputSpec c_output)
      : m_(m), n_(n), k_(k),
        a_(std::move(a)), b_(std::move(b)),
        c_input_(std::move(c_input)), c_output_(std::move(c_output)) {
    validate();
    initialize_batch_rows_pack_a();
  }

  VECOPS_INLINE nint_t required_workspace() const {
    return WorkspacePlanner::required(*this);
  }

  /// Runtime-visible decisions used by diagnostics/tests.  They report the
  /// actual per-side plan after evaluating dynamic extents and batch reuse.
  VECOPS_INLINE bool online_packs_a() const {
    return selected_auto_pack<::vecops::matmul::Operand::A>();
  }
  VECOPS_INLINE bool online_packs_b() const {
    return selected_auto_pack<::vecops::matmul::Operand::B>();
  }

  VECOPS_INLINE bool whole_b_panel_enabled() const {
    return use_whole_b_panel();
  }

  VECOPS_INLINE nint_t whole_b_panel_bytes() const {
    if constexpr (!WholeBPanelCandidate) {
      return 0;
    } else {
      const auto layout = whole_b_panel_layout();
      return tensor::numel(layout) * static_cast<nint_t>(
          sizeof(typename Atom::TB)) + 63;
    }
  }

  template <execution::ExecutionScope Scope>
  VECOPS_INLINE void operator()(Scope& scope) const {
    scope.with_resources(
        *this, [&](auto& active) VECOPS_INLINE_LAMBDA {
          execute(active);
        });
  }

  VECOPS_INLINE void operator()(kernel::WorkspaceView& workspace) const {
    ExecutionSession execution{workspace};
    (*this)(execution);
  }

private:
  using BatchPlanner =
      ArchitectureBatchPlanner<ArchitectureFamilyInvocation>;
  friend BatchPlanner;
  using WorkspacePlanner = ArchitectureWorkspacePlanner<
      ArchitectureFamilyInvocation>;
  friend WorkspacePlanner;
  using PackingPlanner = ArchitecturePackingPlanner<
      ArchitectureFamilyInvocation>;
  friend PackingPlanner;
  template <::vecops::matmul::Operand Side>
  using PackingPlacement = std::conditional_t<
      Side == ::vecops::matmul::Operand::A,
      APackingPlacement, BPackingPlacement>;

  template <::vecops::matmul::Operand Side>
  static constexpr bool AllowsOutsidePacking = allows_packing_site(
      PackingPlacement<Side>::allowed_sites,
      ::vecops::matmul::PackingSite::outside_reuse_loop);

  template <::vecops::matmul::Operand Side>
  static constexpr bool RequiresOutsidePacking =
      AllowsOutsidePacking<Side> &&
      PackingPlacement<Side>::requirement ==
          ::vecops::matmul::PackingRequirement::required;
  template <::vecops::matmul::Operand Side, typename Spec>
  static constexpr bool AutoPackOperand = [] {
    using Layout = typename Spec::InputLayout;
    using Element = typename ::vecops::matmul::packing_t<Atom, Side>::Element;
    constexpr int Rank = COutputSpec::OutputTensor::Ndim;
    if constexpr (
        (Rank != 2 && Rank != 3) ||
        ::vecops::matmul::is_packed_layout<Atom, Side, Layout>()) {
      return false;
    } else {
      using Memory = typename Spec::MemoryElement;
      using Transform = typename Spec::TransformType;
      if constexpr (std::same_as<
                        Implementation, kernel::matmul_implementation::AMX>) {
        // AMX raw B needs a 16x16 AVX-512 transpose for every resident M
        // family.  For reused rank-two operands, packing once avoids repeating
        // that transpose and also makes the K loop a direct 1 KiB panel walk.
        // Rank-three operands use one leaf-sized staging buffer.  Broadcast
        // operands are packed once; independent operands may still profit
        // when packing also performs conversion or a transform, and are then
        // repacked for each batch item.
        constexpr bool ReusableRank = Rank == 2 || Rank == 3;
        constexpr bool NativeMemory = std::same_as<Memory, Element>;
        constexpr bool SupportedConversion =
            std::same_as<Element, bfloat16_t> &&
            std::same_as<Memory, float32_t>;
        constexpr bool SupportedQuantization = [] {
          if constexpr (std::same_as<Transform, tensor::NoTransform>) {
            return false;
          } else {
            return std::same_as<Memory, float32_t> &&
                (std::same_as<Element, int8_t> ||
                 std::same_as<Element, uint8_t>) &&
                Transform::is_elementwise &&
                Transform::permutation_equivariant &&
                std::same_as<typename Transform::TIn, float32_t> &&
                std::same_as<typename Transform::TOut, float32_t>;
          }
        }();
        using KStride = tensor::stride_type_t<Rank - 1, Layout>;
        return ReusableRank &&
            std::same_as<typename Spec::ComputeType, Element> &&
            (std::same_as<Transform, tensor::NoTransform> ||
             SupportedQuantization) &&
            (NativeMemory || SupportedConversion || SupportedQuantization) &&
            meta::is_singleton_v<KStride> &&
            meta::singleton_value_v<KStride> == 1;
      } else if constexpr (!std::same_as<
                               Implementation,
                               kernel::matmul_implementation::SME>) {
        return false;
      }
      constexpr bool NativeMemory = std::same_as<Memory, Element>;
      constexpr bool SupportedConversion =
          (std::same_as<Element, bfloat16_t> &&
           std::same_as<Memory, float32_t>) ||
          (std::same_as<Element, float32_t> &&
           (std::same_as<Memory, bfloat16_t> ||
            std::same_as<Memory, float16_t>)) ||
          (std::same_as<Element, float64_t> &&
           std::same_as<Memory, float32_t>);
      constexpr bool SupportedQuantization = [] {
        if constexpr (std::same_as<Transform, tensor::NoTransform>) {
          return false;
        } else {
          return std::same_as<Memory, float32_t> &&
              (std::same_as<Element, int8_t> ||
               std::same_as<Element, uint8_t>) &&
              Transform::is_elementwise &&
              Transform::permutation_equivariant &&
              std::same_as<typename Transform::TIn, float32_t> &&
              std::same_as<typename Transform::TOut, float32_t>;
        }
      }();
      using KStride = tensor::stride_type_t<Rank - 1, Layout>;
      return sizeof(Element) <= 8 && Spec::InputTensor::Ndim == Rank &&
          std::same_as<typename Spec::ComputeType, Element> &&
          (std::same_as<Transform, tensor::NoTransform> ||
           SupportedQuantization) &&
          (NativeMemory || SupportedConversion || SupportedQuantization) &&
          meta::is_singleton_v<KStride> &&
          meta::singleton_value_v<KStride> == 1;
    }
  }();

  static constexpr bool PackedAInput = ::vecops::matmul::is_packed_layout<
      Atom, ::vecops::matmul::Operand::A, typename ASpec::InputLayout>();
  static constexpr bool PackedBInput = ::vecops::matmul::is_packed_layout<
      Atom, ::vecops::matmul::Operand::B, typename BSpec::InputLayout>();

  static constexpr bool Rank3SharedB = [] {
    if constexpr (COutputSpec::OutputTensor::Ndim != 3) {
      return false;
    } else if constexpr (PackedBInput) {
      return true;
    } else {
      using BatchStride = tensor::stride_type_t<
          0, typename BSpec::InputLayout>;
      return meta::is_singleton_v<BatchStride> &&
          meta::singleton_value_v<BatchStride> == 0;
    }
  }();

  static constexpr bool Rank3SharedA = [] {
    if constexpr (COutputSpec::OutputTensor::Ndim != 3) {
      return false;
    } else if constexpr (PackedAInput) {
      // Packed operands carry no batch dimension and are broadcast by the
      // rank-three facade.
      return true;
    } else if constexpr (ASpec::InputTensor::Ndim != 3) {
      return false;
    } else {
      using BatchStride = tensor::stride_type_t<
          0, typename ASpec::InputLayout>;
      return meta::is_singleton_v<BatchStride> &&
          meta::singleton_value_v<BatchStride> == 0;
    }
  }();

  // A dense rank-three [batch, M, K] activation with one shared B is
  // mathematically one rank-two [batch*M, K] product.  Collapsing the two
  // leading A/C dimensions lets one matrix tile use rows from several batches,
  // instead of executing a mostly inactive M<=16 tile once per batch.
  // Coordinate-aware transforms cannot be rebound because the flattened row
  // no longer has a one-axis CoordinateProjection back to {batch, M}; pure
  // elementwise transforms are coordinate-free and remain safe.
  static constexpr bool BatchRowsFlattenCandidate =
      (std::same_as<Implementation, kernel::matmul_implementation::AMX>
       || std::same_as<Implementation, kernel::matmul_implementation::SME>
       ) &&
      Rank3SharedB && !PackedAInput &&
      ASpec::TransformType::is_elementwise &&
      CInputSpec::TransformType::is_elementwise &&
      COutputSpec::TransformType::is_elementwise;

  // The symmetric shared-A case can combine batch columns only when M=1:
  // C[batch, 0, N] and a rank-two C[0, batch*N] then have identical physical
  // order.  For M>1 the two orders differ and would need a scatter/transpose.
  static constexpr bool BatchColumnsFlattenCandidate =
      (std::same_as<Implementation, kernel::matmul_implementation::AMX>
       || std::same_as<Implementation, kernel::matmul_implementation::SME>
       ) &&
      Rank3SharedA && !PackedBInput &&
      BSpec::InputTensor::Ndim == 3 &&
      ASpec::TransformType::is_elementwise &&
      BSpec::TransformType::is_elementwise &&
      CInputSpec::TransformType::is_elementwise &&
      COutputSpec::TransformType::is_elementwise;

  // A shared per-N accumulator bias has physical strides {0,0,1}.  Flattening
  // batch columns needs a non-affine `flat_column % N` address, which Layout
  // deliberately cannot encode.  For the native/no-transform form, repeat at
  // most 64 accumulator values into workspace and feed the same dense rank-2
  // access type as an ordinary flattened C input.
  static constexpr bool BatchColumnsPeriodicCInputCandidate =
      BatchColumnsFlattenCandidate &&
      std::same_as<
          typename CInputSpec::MemoryElement, typename Atom::TAcc> &&
      std::same_as<
          typename CInputSpec::TransformType, tensor::NoTransform>;

  VECOPS_INLINE bool batch_rows_flatten_enabled() const {
    return BatchPlanner::rows_flatten_enabled(*this);
  }

  VECOPS_INLINE bool batch_columns_flatten_enabled() const {
    return BatchPlanner::columns_flatten_enabled(*this);
  }

  VECOPS_INLINE bool batch_columns_periodic_c_input_enabled() const {
    return BatchPlanner::periodic_c_input_enabled(*this);
  }

  VECOPS_INLINE nint_t batch_columns_periodic_c_input_bytes() const {
    return BatchPlanner::periodic_c_input_bytes(*this);
  }

  template <typename Spec,
            meta::ValueType Rows, meta::ValueType Columns,
            meta::ValueType RowStride>
  VECOPS_INLINE auto flatten_batch_rows_input(
      const Spec& spec, Rows flat_rows, Columns columns,
      RowStride row_stride) const {
    return BatchPlanner::flatten_input(
        spec, flat_rows, columns, row_stride);
  }

  template <typename Spec,
            meta::ValueType Rows, meta::ValueType Columns,
            meta::ValueType RowStride>
  VECOPS_INLINE auto flatten_batch_rows_output(
      const Spec& spec, Rows flat_rows, Columns columns,
      RowStride row_stride) const {
    return BatchPlanner::flatten_output(
        spec, flat_rows, columns, row_stride);
  }

  VECOPS_INLINE auto shared_b_leaf() const {
    return BatchPlanner::shared_b_leaf(*this);
  }

  VECOPS_INLINE auto shared_a_leaf() const {
    return BatchPlanner::shared_a_leaf(*this);
  }

  // Both rank-two and rank-three calls cost A/B independently.  A rank-three
  // broadcast changes that side's amortization but never creates a runtime
  // Cartesian choice with the opposite operand.
  static constexpr bool RankAllowsAutoPack = Rank == 2 || Rank == 3;
  static constexpr bool RequestedFamilyNeedsRawOperands =
      std::same_as<typename EffectiveFamilyDispatch::Family,
                   ::vecops::matmul::kernel_family::SmallVector> ||
      std::same_as<typename EffectiveFamilyDispatch::Family,
                   ::vecops::matmul::kernel_family::RuntimeQuantInt8>;
  static constexpr bool AutoPackA = RankAllowsAutoPack &&
      !RequestedFamilyNeedsRawOperands &&
      AllowsOutsidePacking<::vecops::matmul::Operand::A> &&
      AutoPackOperand<::vecops::matmul::Operand::A, ASpec>;
  static constexpr bool AutoPackB = RankAllowsAutoPack &&
      !RequestedFamilyNeedsRawOperands &&
      AllowsOutsidePacking<::vecops::matmul::Operand::B> &&
      AutoPackOperand<::vecops::matmul::Operand::B, BSpec>;
  static constexpr nint_t WholeBPanelMinN = 64;
  static constexpr nint_t WholeBPanelMaxN = 256;
  static constexpr nint_t WholeBPanelMaxBytes = 1024 * 1024;
  static constexpr nint_t WholeBPanelMinOperandBytes = 16 * 1024 * 1024;
  static constexpr nint_t WholeBPanelMinSpatialOperandBytes = 2 * 1024 * 1024;
  static constexpr bool WholeBPanelMetaMayRun = [] {
    using MV = std::remove_cvref_t<MExtent>;
    using NV = std::remove_cvref_t<NExtent>;
    using KV = std::remove_cvref_t<KExtent>;
    constexpr nint_t ElementBytes = sizeof(typename Atom::TB);
    constexpr nint_t MaxK = WholeBPanelMaxBytes /
        (WholeBPanelMinN * ElementBytes);
    if constexpr (meta::has_upper_bound_v<MV> &&
                  meta::upper_bound_v<MV> < 32)
      return false;
    if constexpr (meta::has_upper_bound_v<NV> &&
                  meta::upper_bound_v<NV> <= WholeBPanelMinN)
      return false;
    if constexpr (meta::has_lower_bound_v<KV> &&
                  meta::lower_bound_v<KV> > MaxK)
      return false;
    if constexpr (meta::has_upper_bound_v<NV> &&
                  meta::has_upper_bound_v<KV>) {
      constexpr nint_t N = meta::upper_bound_v<NV>;
      constexpr nint_t K = meta::upper_bound_v<KV>;
      if constexpr (N <= 0 || K <= 0) return false;
      constexpr nint_t MinNForLargeB =
          1 + (WholeBPanelMinOperandBytes - 1) / (K * ElementBytes);
      constexpr bool BMayBeLarge = N >= MinNForLargeB;
      constexpr nint_t MinNForSpatialReuse =
          1 + (WholeBPanelMinSpatialOperandBytes - 1) /
                  (K * ElementBytes);
      constexpr bool SpatialReuseMayBeLarge =
          (!meta::has_upper_bound_v<MV> ||
           meta::upper_bound_v<MV> >= 512) && N >= MinNForSpatialReuse;
      if constexpr (!BMayBeLarge && !SpatialReuseMayBeLarge) return false;
    }
    return true;
  }();
  static constexpr bool WholeBPanelCandidate =
      Rank == 2 && WholeBPanelMetaMayRun &&
      std::same_as<Implementation, kernel::matmul_implementation::AMX> &&
      std::same_as<typename EffectiveFamilyDispatch::Family,
                   ::vecops::matmul::kernel_family::WholeProblem> &&
      (kernel::matmul_details::n_major_family_dispatch_v<
           EffectiveFamilyDispatch> ||
       kernel::matmul_details::automatic_traversal_family_dispatch_v<
           EffectiveFamilyDispatch>) &&
      AllowsOutsidePacking<::vecops::matmul::Operand::B> &&
      !PackedBInput && AutoPackB &&
      std::same_as<typename Atom::TA, bfloat16_t> &&
      std::same_as<typename Atom::TB, bfloat16_t> &&
      std::same_as<typename Atom::TAcc, float32_t> &&
      std::same_as<typename BSpec::MemoryElement, bfloat16_t> &&
      std::same_as<typename BSpec::TransformType, tensor::NoTransform> &&
      meta::is_singleton_v<tensor::stride_type_t<
          Rank - 1, typename BSpec::InputLayout>> &&
      meta::singleton_value_v<tensor::stride_type_t<
          Rank - 1, typename BSpec::InputLayout>> == 1;
  static constexpr bool Rank3CompletesPackedPair = Rank3SharedB &&
      ((AutoPackA && PackedBInput) || (AutoPackB && PackedAInput));
  // AMX avoids cloning a raw-B flatten leaf when conversion-aware auto-pack
  // already owns the flattened operation; native B keeps its large-weight
  // fallback.  SME must retain the direct leaf for every B type: tiny/tail
  // shapes reject online packing, and without it they would still pay one
  // StreamingZA/MOPA traversal per batch even though flattening is legal.
  static constexpr bool NativeBInput =
      std::same_as<typename BSpec::MemoryElement, typename Atom::TB> &&
      std::same_as<typename BSpec::TransformType, tensor::NoTransform>;
  static constexpr bool DirectBatchRowsFlattenCandidate =
      BatchRowsFlattenCandidate &&
      (std::same_as<Implementation, kernel::matmul_implementation::SME> ||
       !AutoPackB || PackedBInput || NativeBInput);

  static constexpr bool NativeAInput =
      std::same_as<typename ASpec::MemoryElement, typename Atom::TA> &&
      std::same_as<typename ASpec::TransformType, tensor::NoTransform>;
  static constexpr bool SMERank3SkinnyF64APack =
      std::same_as<Implementation, kernel::matmul_implementation::SME> &&
      Rank3SharedB && AutoPackA &&
      std::same_as<typename Atom::TA, float64_t> &&
      std::same_as<typename ASpec::MemoryElement, float32_t>;
  // If SME's existing cost model has already chosen to pack both operands,
  // pack the complete dense [batch*M,K] A once and execute one packed problem.
  // The old path reused a panel-sized A buffer and therefore still launched
  // one mostly empty ZA problem per batch item.
  static constexpr bool SMEBatchRowsFullPackCandidate =
      std::same_as<Implementation, kernel::matmul_implementation::SME> &&
      BatchRowsFlattenCandidate && AutoPackA && AutoPackB;
  static constexpr bool BatchRowsPackACandidate =
#if defined(VECOPS_DISABLE_AMX_BATCH_ROWS_PACK_A)
      false;
#else
      std::same_as<Implementation, kernel::matmul_implementation::AMX> &&
      AllowsOutsidePacking<::vecops::matmul::Operand::A> &&
      BatchRowsFlattenCandidate && NativeAInput &&
      (std::same_as<typename Atom::TA, bfloat16_t> ||
       std::is_integral_v<typename Atom::TA>);
#endif

  template <::vecops::matmul::Operand Side, typename Spec>
  static constexpr bool StreamingCompatibleAutoPack = [] {
    if constexpr (!AutoPackOperand<Side, Spec>) {
      return true;
    } else {
      using Memory = typename Spec::MemoryElement;
      using Element = typename ::vecops::matmul::packing_t<Atom, Side>::Element;
      using Transform = typename Spec::TransformType;
      constexpr bool StreamingQuantization = [] {
        if constexpr (std::same_as<Transform, tensor::NoTransform>) {
          return false;
        } else {
          return std::same_as<Memory, float32_t> &&
              (std::same_as<Element, int8_t> ||
               std::same_as<Element, uint8_t>) &&
              Transform::is_elementwise &&
              Transform::permutation_equivariant &&
              std::same_as<typename Transform::TIn, float32_t> &&
              std::same_as<typename Transform::TOut, float32_t>;
        }
      }();
      // Native, FP32->BF16, and the fused FP32->I8/U8 postprocess pack execute
      // entirely in Streaming+ZA mode.  BF16/FP16->FP32 still finish with
      // ordinary-SVE postprocessing and retain separate SME intervals.
      return std::same_as<Memory, Element> ||
          (std::same_as<Memory, float32_t> &&
           std::same_as<Element, bfloat16_t>) ||
          StreamingQuantization;
    }
  }();

  static constexpr bool SingleStreamingAutoPackCandidate =
#if defined(HAS_SME)
      // Rank-two BF16/F32 showed 4-14% wins.  Other rank-two atoms stay compact,
      // but the distinct shared-B specialization crosses up to 17 SME region
      // boundaries per call and can amortize one inlined traversal over all
      // batches.
      Rank3SharedA || Rank3SharedB ||
      std::same_as<Atom, ::vecops::matmul::SME_BF16F32> ||
      std::same_as<Atom, ::vecops::matmul::SME_F32F32>;
#else
      false;
#endif

  template <bool PackA, bool PackB>
  static constexpr bool SingleStreamingAutoPackRegion =
      std::same_as<Implementation, kernel::matmul_implementation::SME> &&
      SingleStreamingAutoPackCandidate &&
      // Keep mixed raw/packed operations on the compact existing path.  A
      // rank-three shared operand is already a distinct specialization and
      // amortizes its one remaining pack over the full batch, so completing a
      // packed pair there also merits one region.
      ((PackA && PackB) ||
       (Rank == 3 && (Rank3SharedA || Rank3SharedB) &&
        (PackA || PackedAInput) && (PackB || PackedBInput))) &&
      StreamingCompatibleAutoPack<::vecops::matmul::Operand::A, ASpec> &&
      StreamingCompatibleAutoPack<::vecops::matmul::Operand::B, BSpec>;

  template <::vecops::matmul::Operand Side, typename Spec>
  static constexpr bool AutoPackWidensBF16 =
      AutoPackOperand<Side, Spec> &&
      std::same_as<typename Spec::MemoryElement, bfloat16_t> &&
      std::same_as<typename ::vecops::matmul::packing_t<Atom, Side>::Element, float32_t>;

  template <::vecops::matmul::Operand Side, typename Spec>
  static constexpr bool AutoPackQuantizes = [] {
    using Transform = typename Spec::TransformType;
    using Element = typename ::vecops::matmul::packing_t<Atom, Side>::Element;
    if constexpr (!AutoPackOperand<Side, Spec> ||
                  std::same_as<Transform, tensor::NoTransform>) {
      return false;
    } else {
      return std::same_as<typename Spec::MemoryElement, float32_t> &&
          (std::same_as<Element, int8_t> ||
           std::same_as<Element, uint8_t>);
    }
  }();

  template <::vecops::matmul::Operand Side, typename Spec>
  static constexpr bool AutoPackElidesInputWork = [] {
    if constexpr (!AutoPackOperand<Side, Spec>) {
      return false;
    } else {
      using Element = typename ::vecops::matmul::packing_t<Atom, Side>::Element;
      return !std::same_as<typename Spec::MemoryElement, Element> ||
          !std::same_as<typename Spec::TransformType, tensor::NoTransform>;
    }
  }();

  struct NoBatchRowsPackAState {};
  using BatchRowsPackAState = std::conditional_t<
      BatchRowsPackACandidate, bool, NoBatchRowsPackAState>;

  VECOPS_INLINE void initialize_batch_rows_pack_a() {
    if constexpr (BatchRowsPackACandidate)
      batch_rows_pack_a_ = use_batch_rows_pack_a();
  }

  VECOPS_INLINE bool batch_rows_pack_a_enabled() const {
    if constexpr (BatchRowsPackACandidate) return batch_rows_pack_a_;
    else return false;
  }

  VECOPS_INLINE bool use_batch_rows_pack_a() const {
    if constexpr (!BatchRowsPackACandidate) {
      return false;
    } else {
      if (!batch_rows_flatten_enabled()) return false;
      // A raw shared-B operation can only consume packed A through the branch
      // that also packed B.  Explicit packed-B inputs enter directly.
      if constexpr (!PackedBInput) {
        if constexpr (!MayAutoPackB) return false;
        if (!online_packs_b()) return false;
      }
      const nint_t batch = static_cast<nint_t>(tensor::size<0>(
          c_output_.output_layout()));
      const nint_t m = static_cast<nint_t>(m_);
      const nint_t n = static_cast<nint_t>(n_);
      const nint_t k = static_cast<nint_t>(k_);
      const nint_t flat_m = batch * m;
      // The standalone grid supported N=64, but full fused Scenario ABBA made
      // that boundary output-pipeline sensitive (one +3.2% regression).
      // N>=128 retained uniform wins, including the N=128/K=64 boundary.
      if (flat_m > 8 || n < 128 || k < 64) return false;
      return true;
    }
  }

  template <::vecops::matmul::Operand Side>
  VECOPS_INLINE static constexpr bool use_auto_packing_for(
      nint_t m, nint_t n, nint_t k,
      nint_t effective_m, nint_t effective_n,
      bool opposite_side_will_pack = false) {
    constexpr bool CanPack = Side == ::vecops::matmul::Operand::A
        ? AutoPackA : AutoPackB;
    if constexpr (!CanPack) {
      return false;
    } else {
      if (m <= 0 || n <= 0 || k <= 0 ||
          effective_m <= 0 || effective_n <= 0)
        return false;
      if constexpr (std::same_as<
                        Implementation, kernel::matmul_implementation::AMX>) {
        // Cost the two operands independently.  Packing A is a contiguous-panel
        // optimization and needs wide N reuse.  Packing B additionally removes
        // AMX's repeated 16x16 raw-B transpose, so its native threshold follows
        // M reuse instead.  A conversion/transform performed by packing is useful
        // at smaller reuse and is admitted to the generic work test below.
        if constexpr (Side == ::vecops::matmul::Operand::A) {
          constexpr bool Elides = AutoPackElidesInputWork<
              ::vecops::matmul::Operand::A, ASpec>;
          // A-only native packing was previously reachable only when B was
          // already packed.  Keep that invariant: padding a skinny raw A can
          // disable the vector leaf without removing raw-B transpose work.
          if constexpr (!Elides && !PackedBInput)
            if (!opposite_side_will_pack) return false;
          if (effective_n < (Elides ? 16 : 256)) return false;
        } else {
          constexpr bool Elides = AutoPackElidesInputWork<
              ::vecops::matmul::Operand::B, BSpec>;
          if constexpr (Rank3SharedB) {
            // Eight decode rows already reuse each packed B panel enough to
            // amortize one copy; the generic work threshold below filters the
            // tiny and short-K shared-weight cases.  Native B packing ceases
            // to pay once the packed weight no longer fits a private L2: the
            // 8 MiB MLP and 25--33 MiB Qwen weights regressed by 7--12% from
            // rereading and rewriting the full matrix.  Conversion-aware
            // packing has a different gate because it also removes repeated
            // conversion/transform work.
            if (effective_m < (Elides ? 1 : 8) || n < 16) return false;
            using Memory = typename BSpec::MemoryElement;
            using Element =
                typename ::vecops::matmul::packing_t<Atom, ::vecops::matmul::Operand::B>::Element;
            if constexpr (std::same_as<Memory, Element>) {
              constexpr nint_t MaxPackedElements =
                  (2 * 1024 * 1024) / static_cast<nint_t>(sizeof(Element));
              if (n > MaxPackedElements / k) return false;
            }
          } else {
            // Native B packing crosses over only once at least eight AMX
            // row tiles reuse every packed panel.  At M=64 the copy made
            // 64x64x{128,256} slower; M>=128 retained the packed-panel win.
            if (m < (Elides ? 1 : 128)) return false;
            if (n < 16) return false;
            if constexpr (!Elides) {
              // At the marginal M=128 reuse point, streaming a very large B
              // through a complete online copy loses to the raw per-panel
              // transpose.  Keep the measured 128x4096x4096 win (32 MiB B),
              // but reject the 128x11008x4096 case (86 MiB B).  M>=256 has
              // twice the reuse and is intentionally left uncapped.
              constexpr nint_t MaxMarginalBBytes = 64 * 1024 * 1024;
              using PackedElement = typename ::vecops::matmul::packing_t<
                  Atom, ::vecops::matmul::Operand::B>::Element;
              constexpr nint_t MaxMarginalBElements =
                  MaxMarginalBBytes / static_cast<nint_t>(
                      sizeof(PackedElement));
              if (m < 256 && k > 0 &&
                  n > MaxMarginalBElements / k)
                return false;
            }
          }
        }
      }
      // The generic BF16->FP32 pack doubles the packed operand footprint.
      // Do not widen a side unless the opposite output dimension reuses each
      // packed row at least four times.  In particular, widening a 1024x1024
      // B for M=1 decode measured slower than direct conversion, while batch
      // shapes amortize it well.  FP16->FP32 has a dedicated staged pack and
      // remains profitable at M=1, so it intentionally has no such guard.
      constexpr nint_t WidenReuseThreshold = 4;
      if constexpr (Side == ::vecops::matmul::Operand::A &&
                    AutoPackWidensBF16<
                        ::vecops::matmul::Operand::A, ASpec>)
        if (effective_n < WidenReuseThreshold) return false;
      if constexpr (Side == ::vecops::matmul::Operand::B &&
                    AutoPackWidensBF16<
                        ::vecops::matmul::Operand::B, BSpec>)
        if (effective_m < WidenReuseThreshold) return false;
      if constexpr (
          Side == ::vecops::matmul::Operand::A &&
          Rank3CompletesPackedPair && AutoPackA &&
          AutoPackQuantizes<::vecops::matmul::Operand::A, ASpec>) {
        // A is different for every batch, so shared B does not amortize its
        // FP32->I8/U8 transform.  The 64x128 boundary regressed, while either
        // twice the N reuse or twice the K work remained profitable.
        if (effective_n < 128 && k < 256) return false;
      }
      // Packing pays for itself once enough output dot products reuse it.
      // Require both aggregate work and spatial reuse; a 1x1 product with a
      // very long K has high work but cannot amortize copying either operand.
      // Express both tests with divisions to avoid overflowing M*N*K.
      constexpr bool PackElidesInputWork = Side == ::vecops::matmul::Operand::A
          ? AutoPackElidesInputWork<::vecops::matmul::Operand::A, ASpec>
          : AutoPackElidesInputWork<::vecops::matmul::Operand::B, BSpec>;
      constexpr bool AMXBPackElidesInputWork =
          std::same_as<
              Implementation, kernel::matmul_implementation::AMX> &&
          Side == ::vecops::matmul::Operand::B &&
          AutoPackElidesInputWork<::vecops::matmul::Operand::B, BSpec>;
      constexpr bool AMXRank3BOnly =
          std::same_as<
              Implementation, kernel::matmul_implementation::AMX> &&
          Side == ::vecops::matmul::Operand::B && Rank3SharedB;
      constexpr bool SMERank3QuantizedPair =
#if defined(VECOPS_DISABLE_SME_RANK3_QUANT_FULL_PACKING)
          false;
#else
          std::same_as<
              Implementation, kernel::matmul_implementation::SME> &&
          Rank3SharedB && PackElidesInputWork;
#endif
      constexpr bool LowWorkTransformPair =
          AMXBPackElidesInputWork || SMERank3QuantizedPair;
      constexpr bool SMEF64ConversionPair =
          std::same_as<
              Implementation, kernel::matmul_implementation::SME> &&
          AutoPackA && AutoPackB &&
          std::same_as<typename Atom::TA, float64_t> &&
          std::same_as<typename Atom::TB, float64_t> &&
          std::same_as<typename ASpec::MemoryElement, float32_t> &&
          std::same_as<typename BSpec::MemoryElement, float32_t>;
      constexpr nint_t ReuseThreshold =
          LowWorkTransformPair ? 64 : 128;
      constexpr nint_t WorkThreshold =
          LowWorkTransformPair
          ? 8 * 1024
          // A rank-three shared B is copied once for the complete batch.
          // Native lifecycle measurements break even around 16 Ki dots;
          // retaining the rank-two 128 Ki threshold leaves the tail/tiny/
          // small shared catalog 38--83% slower than explicit online pack.
          : AMXRank3BOnly
              ? 16 * 1024
          : SMEF64ConversionPair && !Rank3SharedB
              ? 256 * 1024
          : 128 * 1024;
      const nint_t reuse = Side == ::vecops::matmul::Operand::A
          ? effective_n : effective_m;
      if (reuse < 1 + (ReuseThreshold - 1) /
                       (Side == ::vecops::matmul::Operand::A
                            ? effective_m : effective_n))
        return false;
      nint_t remaining = 1 + (WorkThreshold - 1) / effective_m;
      remaining = 1 + (remaining - 1) / effective_n;
      return k >= remaining;
    }
  }

  using MV = std::remove_cvref_t<MExtent>;
  using NV = std::remove_cvref_t<NExtent>;
  using KV = std::remove_cvref_t<KExtent>;

  template <bool Lower, typename Value>
  static constexpr bool has_extent_bound_v = Lower
      ? meta::has_lower_bound_v<Value>
      : meta::has_upper_bound_v<Value>;

  template <bool Lower, typename Value>
  static constexpr nint_t extent_bound_v = Lower
      ? meta::lower_bound_v<Value>
      : meta::upper_bound_v<Value>;

  template <bool Lower>
  static constexpr bool HasProblemBounds = [] {
    if constexpr (!has_extent_bound_v<Lower, MV> ||
                  !has_extent_bound_v<Lower, NV> ||
                  !has_extent_bound_v<Lower, KV>) {
      return false;
    } else if constexpr (Rank == 3) {
      using Batch = tensor::size_type_t<
          0, typename COutputSpec::OutputLayout>;
      return has_extent_bound_v<Lower, Batch>;
    } else {
      return true;
    }
  }();

  static constexpr bool SingletonProblem = [] {
    if constexpr (!meta::is_singleton_v<MV> ||
                  !meta::is_singleton_v<NV> ||
                  !meta::is_singleton_v<KV>) {
      return false;
    } else if constexpr (Rank == 3) {
      using Batch = tensor::size_type_t<
          0, typename COutputSpec::OutputLayout>;
      return meta::is_singleton_v<Batch>;
    } else {
      return true;
    }
  }();

  static constexpr nint_t saturating_extent_product(
      nint_t lhs, nint_t rhs) {
    constexpr nint_t Limit = std::numeric_limits<nint_t>::max();
    if (lhs <= 0 || rhs <= 0) return lhs * rhs;
    return rhs > Limit / lhs ? Limit : lhs * rhs;
  }

  VECOPS_INLINE bool use_whole_b_panel() const {
    if constexpr (!WholeBPanelCandidate) {
      return false;
    } else {
      const nint_t m = static_cast<nint_t>(m_);
      const nint_t n = static_cast<nint_t>(n_);
      const nint_t k = static_cast<nint_t>(k_);
      const nint_t panel_n = whole_b_panel_n();
      if (m < 32 || n <= panel_n || k <= 0)
        return false;
      constexpr nint_t PanelElements = WholeBPanelMaxBytes /
          static_cast<nint_t>(sizeof(typename Atom::TB));
      if (k > PanelElements / panel_n) return false;
      // A large B operand benefits even at modest M (LLM projections). A
      // large two-dimensional spatial traversal can instead amortize a
      // bounded panel despite a smaller complete B (square/near-square
      // attention and triangle products).
      constexpr nint_t ElementBytes = sizeof(typename Atom::TB);
      const nint_t min_n = 1 +
          (WholeBPanelMinOperandBytes - 1) / (k * ElementBytes);
      const nint_t min_spatial_n = 1 +
          (WholeBPanelMinSpatialOperandBytes - 1) / (k * ElementBytes);
      const bool large_b = n >= min_n;
      const bool large_spatial_reuse =
          m >= 512 && n >= std::max<nint_t>(512, min_spatial_n) && k >= 256;
      return large_b || large_spatial_reuse;
    }
  }

  VECOPS_INLINE nint_t whole_b_panel_n() const {
    // One packed panel is reused by every M tile. Scale its N span with that
    // reuse depth, in tile multiples, while keeping small-M LLM projections
    // at the empirically robust 64-column lifetime.
    const nint_t m = static_cast<nint_t>(m_);
    const nint_t reuse_scaled = std::max<nint_t>(
        WholeBPanelMinN, (m / 4) & ~nint_t{15});
    return std::min<nint_t>(WholeBPanelMaxN, reuse_scaled);
  }

  VECOPS_INLINE auto whole_b_panel_layout() const {
    static_assert(WholeBPanelCandidate);
    const meta::Any panel_n{std::min<nint_t>(
        whole_b_panel_n(), static_cast<nint_t>(n_))};
    const auto raw_layout = tensor::make_layout(
        tensor::make_shape(panel_n, k_));
    return ::vecops::matmul::packed_layout<
        Atom, ::vecops::matmul::Operand::B>(raw_layout);
  }

  template <bool Lower>
  static constexpr nint_t batch_bound() {
    if constexpr (Rank == 3) {
      using Batch = tensor::size_type_t<
          0, typename COutputSpec::OutputLayout>;
      return extent_bound_v<Lower, Batch>;
    } else {
      return 1;
    }
  }

  template <bool Lower>
  static constexpr nint_t effective_m_bound() {
    constexpr nint_t M = extent_bound_v<Lower, MV>;
    if constexpr (Rank3SharedB)
      return saturating_extent_product(M, batch_bound<Lower>());
    else
      return M;
  }

  template <bool Lower>
  static constexpr nint_t effective_n_bound() {
    constexpr nint_t N = extent_bound_v<Lower, NV>;
    if constexpr (Rank3SharedA)
      return saturating_extent_product(N, batch_bound<Lower>());
    else
      return N;
  }

  /** Evaluate the existing profitability model at one compile-time corner. */
  template <::vecops::matmul::Operand Side, bool Lower>
  static constexpr bool auto_pack_at_bound() {
    static_assert(HasProblemBounds<Lower>);
    constexpr nint_t M = extent_bound_v<Lower, MV>;
    constexpr nint_t N = extent_bound_v<Lower, NV>;
    constexpr nint_t K = extent_bound_v<Lower, KV>;
    constexpr nint_t EffectiveM = effective_m_bound<Lower>();
    constexpr nint_t EffectiveN = effective_n_bound<Lower>();
    constexpr bool PackB = [=] {
      if constexpr (Side == ::vecops::matmul::Operand::A)
        return use_auto_packing_for<::vecops::matmul::Operand::B>(
            M, N, K, EffectiveM, EffectiveN);
      else
        return false;
    }();
    return use_auto_packing_for<Side>(
        M, N, K, EffectiveM, EffectiveN, PackB);
  }

  /** Whether every native-AMX B in the Meta range stays inside its footprint cap. */
  static constexpr bool NativeAMXBRangeProvesPacked = [] {
    if constexpr (!AutoPackB ||
                  !std::same_as<
                      Implementation, kernel::matmul_implementation::AMX> ||
                  AutoPackElidesInputWork<
                      ::vecops::matmul::Operand::B, BSpec> ||
                  !HasProblemBounds<true>) {
      return false;
    } else if constexpr (!auto_pack_at_bound<
                             ::vecops::matmul::Operand::B, true>()) {
      return false;
    } else if constexpr (Rank3SharedB) {
      if constexpr (!meta::has_upper_bound_v<NV> ||
                    !meta::has_upper_bound_v<KV>) {
        return false;
      } else {
        using Element = typename ::vecops::matmul::packing_t<
            Atom, ::vecops::matmul::Operand::B>::Element;
        constexpr nint_t MaxElements =
            (2 * 1024 * 1024) / static_cast<nint_t>(sizeof(Element));
        constexpr nint_t N = meta::upper_bound_v<NV>;
        constexpr nint_t K = meta::upper_bound_v<KV>;
        return K > 0 && N <= MaxElements / K;
      }
    } else if constexpr (meta::lower_bound_v<MV> >= 256) {
      // The marginal-footprint cap is disabled once M has twice the base
      // reuse.  All remaining native-B gates are monotone above this corner.
      return true;
    } else if constexpr (!meta::has_upper_bound_v<NV> ||
                         !meta::has_upper_bound_v<KV>) {
      return false;
    } else {
      using Element = typename ::vecops::matmul::packing_t<
          Atom, ::vecops::matmul::Operand::B>::Element;
      constexpr nint_t MaxElements =
          (64 * 1024 * 1024) / static_cast<nint_t>(sizeof(Element));
      constexpr nint_t N = meta::upper_bound_v<NV>;
      constexpr nint_t K = meta::upper_bound_v<KV>;
      return K > 0 && N <= MaxElements / K;
    }
  }();

  /** A native AMX A-pack is monotone once B is caller-prepared: its copy
   * cost is amortized only by N reuse and there is no B-footprint gate left
   * to prove.  Evaluate the known N/K lower corner with the most conservative
   * nonempty M=1, so an unbounded token axis still selects one static packed
   * path when every possible nonempty call is profitable. */
  static constexpr bool NativeAMXPreparedBRangeProvesPackedA = [] {
    if constexpr (!AutoPackA || !PackedBInput ||
                  !std::same_as<
                      Implementation, kernel::matmul_implementation::AMX> ||
                  !meta::has_lower_bound_v<NV> ||
                  !meta::has_lower_bound_v<KV>) {
      return false;
    } else {
      constexpr nint_t MinN = meta::lower_bound_v<NV>;
      constexpr nint_t MinK = meta::lower_bound_v<KV>;
      if constexpr (MinN <= 0 || MinK <= 0) return false;
      else return use_auto_packing_for<
          ::vecops::matmul::Operand::A>(
              1, MinN, MinK, 1, MinN, true);
    }
  }();

  /**
   * Resolve online packing to exactly one compile-time path.
   *
   * Exact Const or singleton-Dynamic metadata uses the full cost model.  For
   * native AMX inputs, packing is selected only when lower bounds already
   * prove it profitable: its crossover depends strongly on M reuse and the B
   * footprint, so defaulting an unbounded token count to packing regresses
   * decode and rank-expand shapes.  SME and conversion/quantization packs
   * have a much lower and monotone crossover; they default to packing unless
   * finite upper bounds prove even the largest admissible problem too small.
   * No runtime raw/pack Cartesian variants are retained.
   */
  template <::vecops::matmul::Operand Side>
  static constexpr bool CompileTimeAutoPack = [] {
    constexpr bool CanPack = Side == ::vecops::matmul::Operand::A
        ? AutoPackA : AutoPackB;
    if constexpr (!CanPack) {
      return false;
    } else if constexpr (RequiresOutsidePacking<Side>) {
      return true;
    } else if constexpr (
        Side == ::vecops::matmul::Operand::A &&
        NativeAMXPreparedBRangeProvesPackedA) {
      return true;
    } else if constexpr (SingletonProblem) {
      return auto_pack_at_bound<Side, true>();
    } else {
      constexpr bool ElidesInputWork = Side == ::vecops::matmul::Operand::A
          ? AutoPackElidesInputWork<::vecops::matmul::Operand::A, ASpec>
          : AutoPackElidesInputWork<::vecops::matmul::Operand::B, BSpec>;
      constexpr bool DefaultToPacked =
          std::same_as<
              Implementation, kernel::matmul_implementation::SME> ||
          ElidesInputWork;
      if constexpr (DefaultToPacked) {
        // The model is monotone for SME and work-eliding packs.  Evaluating
        // its upper corner therefore safely rejects a bounded tiny range;
        // missing upper bounds retain the single packed path.
        if constexpr (HasProblemBounds<false>)
          return auto_pack_at_bound<Side, false>();
        else
          return true;
      } else {
        // Native AMX online packing has non-monotone footprint gates.  Require
        // the range's lower corner to pass and its complete B range to stay
        // below the appropriate footprint cap.
        if constexpr (Side == ::vecops::matmul::Operand::B) {
          return NativeAMXBRangeProvesPacked;
        } else if constexpr (HasProblemBounds<true> &&
                             NativeAMXBRangeProvesPacked) {
          return use_auto_packing_for<::vecops::matmul::Operand::A>(
              extent_bound_v<true, MV>, extent_bound_v<true, NV>,
              extent_bound_v<true, KV>, effective_m_bound<true>(),
              effective_n_bound<true>(), true);
        } else {
          return false;
        }
      }
    }
  }();

  /** Native-AMX B ranges below these monotone minimum gates are always raw. */
  static constexpr bool NativeAMXBRangeProvesRaw = [] {
    if constexpr (!AutoPackB ||
                  !std::same_as<
                      Implementation, kernel::matmul_implementation::AMX> ||
                  AutoPackElidesInputWork<
                      ::vecops::matmul::Operand::B, BSpec> ||
                  !HasProblemBounds<false>) {
      return false;
    } else {
      constexpr nint_t M = extent_bound_v<false, MV>;
      constexpr nint_t N = extent_bound_v<false, NV>;
      constexpr nint_t K = extent_bound_v<false, KV>;
      constexpr nint_t EffectiveM = effective_m_bound<false>();
      constexpr nint_t EffectiveN = effective_n_bound<false>();
      if constexpr (M <= 0 || N <= 0 || K <= 0 ||
                    EffectiveM <= 0 || EffectiveN <= 0) {
        return true;
      } else if constexpr (N < 16) {
        return true;
      } else if constexpr (Rank3SharedB) {
        if constexpr (EffectiveM < 8) return true;
        constexpr nint_t WorkThreshold = 16 * 1024;
        constexpr nint_t RemainingM =
            1 + (WorkThreshold - 1) / EffectiveM;
        constexpr nint_t RemainingN =
            1 + (RemainingM - 1) / EffectiveN;
        return K < RemainingN;
      } else {
        if constexpr (M < 128) return true;
        constexpr nint_t ReuseThreshold = 128;
        if constexpr (EffectiveM <
                      1 + (ReuseThreshold - 1) / EffectiveN)
          return true;
        constexpr nint_t WorkThreshold = 128 * 1024;
        constexpr nint_t RemainingM =
            1 + (WorkThreshold - 1) / EffectiveM;
        constexpr nint_t RemainingN =
            1 + (RemainingM - 1) / EffectiveN;
        return K < RemainingN;
      }
    }
  }();

  static constexpr bool StaticAutoPackA =
      CompileTimeAutoPack<::vecops::matmul::Operand::A>;
  static constexpr bool StaticAutoPackB =
      CompileTimeAutoPack<::vecops::matmul::Operand::B>;
  // Unbounded native AMX B is the one irreducibly ambiguous case: packing can
  // halve a large reused product but regress decode/rank-expand by 50%+.  Keep
  // one raw-vs-B runtime bit, never the old A/B Cartesian four-way fan-out.
  static constexpr bool RuntimeAutoPackB = AutoPackB &&
      std::same_as<Implementation, kernel::matmul_implementation::AMX> &&
      !AutoPackElidesInputWork<
          ::vecops::matmul::Operand::B, BSpec> &&
      !SingletonProblem && !StaticAutoPackB && !NativeAMXBRangeProvesRaw;
  static constexpr bool MayAutoPackA = StaticAutoPackA;
  static constexpr bool MayAutoPackB = StaticAutoPackB || RuntimeAutoPackB;
  static constexpr bool CompileTimeAutoPacking =
      StaticAutoPackA || StaticAutoPackB;

  template <::vecops::matmul::Operand Side>
  VECOPS_INLINE bool selected_auto_pack() const {
    if constexpr (Side == ::vecops::matmul::Operand::A) {
      return StaticAutoPackA;
    } else if constexpr (StaticAutoPackB) {
      return true;
    } else if constexpr (!RuntimeAutoPackB) {
      return false;
    } else {
      const nint_t m = static_cast<nint_t>(m_);
      const nint_t n = static_cast<nint_t>(n_);
      const nint_t k = static_cast<nint_t>(k_);
      const nint_t batch = [&] {
        if constexpr (Rank == 3)
          return static_cast<nint_t>(tensor::size<0>(
              c_output_.output_layout()));
        else
          return nint_t{1};
      }();
      const nint_t effective_m = Rank3SharedB
          ? saturating_extent_product(m, batch) : m;
      const nint_t effective_n = Rank3SharedA
          ? saturating_extent_product(n, batch) : n;
      return use_auto_packing_for<::vecops::matmul::Operand::B>(
          m, n, k, effective_m, effective_n);
    }
  }

  template <::vecops::matmul::Operand Side, typename Spec>
  VECOPS_INLINE auto auto_packed_layout(const Spec& spec) const {
    return PackingPlanner::template packed_layout<Side>(spec);
  }

  template <::vecops::matmul::Operand Side, typename Spec>
  VECOPS_INLINE nint_t auto_packed_bytes(const Spec& spec) const {
    return PackingPlanner::template packed_bytes<Side>(spec);
  }

  VECOPS_INLINE auto batch_rows_packed_a_layout() const {
    return PackingPlanner::batch_rows_packed_a_layout(*this);
  }

  VECOPS_INLINE nint_t batch_rows_packed_a_bytes() const {
    return PackingPlanner::batch_rows_packed_a_bytes(*this);
  }

  template <typename PackImplementation,
            execution::ExecutionScope Scope,
            tensor::InputOperand Input, tensor::OutputOperand Output>
  VECOPS_ALWAYS_INLINE void matmul_pack_forced(
      Scope& scope, const Input& input, const Output& output) const {
    using Packing = ::vecops::matmul::packing_t<Atom, ::vecops::matmul::Operand::A>;
    using Element = typename Packing::Element;
    auto input_spec = tensor::as_input_spec<Element>(input);
    auto output_spec = tensor::as_output_spec<Element>(output);
    using InputPolicy = tensor::InputAccessPolicy<
        Packing::VectorAxis, 1, tensor::AccessPlan::direct>;
    using OutputPolicy = tensor::OutputAccessPolicy<
        std::remove_cvref_t<decltype(output_spec)>::OutputTensor::Ndim - 1,
        tensor::AccessPlan::direct>;
    kernel::with_operands(
        scope,
        tensor::operand(input_spec, InputPolicy{}),
        tensor::operand(output_spec, OutputPolicy{}),
        [&](auto& source, auto& destination) VECOPS_INLINE_LAMBDA {
          kernel::matmul_pack_bound<Atom, ::vecops::matmul::Operand::A>(
              scope, source, destination, PackImplementation{});
          destination.commit();
        });
  }

  template <execution::ExecutionScope Scope,
            tensor::InputOperand Input, tensor::OutputOperand Output>
  VECOPS_ALWAYS_INLINE void matmul_pack_batched_a(
      Scope& scope, const Input& input, const Output& output) const {
    if constexpr (SMERank3SkinnyF64APack) {
      const nint_t panel = static_cast<nint_t>(
          ::vecops::matmul::packing_t<Atom, ::vecops::matmul::Operand::A>::panel());
      const auto input_spec = tensor::as_input_spec<
          typename Atom::TA>(input);
      constexpr int Rank = decltype(input_spec)::InputTensor::Ndim;
      if (static_cast<nint_t>(tensor::size<Rank - 2>(
              input_spec.input_layout())) < panel) {
        matmul_pack_forced<
            kernel::matmul_pack_implementation::SMEFP32ToFP64Single>(
                scope, input, output);
        return;
      }
    }
    run_matmul_pack<Atom, ::vecops::matmul::Operand::A>(
        scope, input, output);
  }

  template <bool PackA, bool PackB, execution::ExecutionScope Scope>
  VECOPS_ALWAYS_INLINE void execute_auto_packed_batched_in_scope(
      Scope& scope) const {
    static_assert(COutputSpec::OutputTensor::Ndim == 3);
    static_assert(PackA || PackB);
    static_assert(!PackA || AutoPackA);
    static_assert(!PackB || AutoPackB);
    auto& workspace = scope.workspace_view();
    const auto mark = workspace.mark();
    void* scratch = nullptr;
    if constexpr (std::same_as<
                      Implementation, kernel::matmul_implementation::AMX>) {
      scratch = workspace.allocate(
          kernel::matmul_implementation::scratch_bytes<Implementation>(), 64);
    }

    if constexpr (PackA && PackB) {
      const auto b_layout = auto_packed_layout<::vecops::matmul::Operand::B>(b_);
      using TB = typename Atom::TB;
      auto* b_data = static_cast<TB*>(workspace.allocate(
          tensor::numel(b_layout) * static_cast<nint_t>(sizeof(TB)), 64));
      auto b_tensor = tensor::make_tensor(b_data, b_layout);
      if constexpr (SMEBatchRowsFullPackCandidate) {
        if (VECOPS_LIKELY(batch_rows_flatten_enabled())) {
          const auto b = tensor::slice_view<0>(b_, 0);
          run_matmul_pack<Atom, ::vecops::matmul::Operand::B>(scope, b, b_tensor);
          const auto a_layout = batch_rows_packed_a_layout();
          using TA = typename Atom::TA;
          auto* a_data = static_cast<TA*>(workspace.allocate(
              tensor::numel(a_layout) * static_cast<nint_t>(sizeof(TA)), 64));
          auto a_tensor = tensor::make_tensor(a_data, a_layout);
          const auto batch = tensor::size<0>(
              c_output_.output_layout());
          const auto flat_m = batch * m_;
          const auto a = flatten_batch_rows_input(a_, flat_m, k_, k_);
          matmul_pack_batched_a(scope, a, a_tensor);
          execute_flattened_batch_rows_operands(
              scope, tensor::input<TA>(a_tensor),
              tensor::input<TB>(b_tensor), scratch);
          workspace.rewind(mark);
          return;
        }
      }
      const bool reuse_b = static_cast<nint_t>(tensor::stride<0>(
          b_.input_layout())) == 0;
      bool b_ready = false;
      const auto a_layout = auto_packed_layout<::vecops::matmul::Operand::A>(a_);
      using TA = typename Atom::TA;
      auto* a_data = static_cast<TA*>(workspace.allocate(
          tensor::numel(a_layout) * static_cast<nint_t>(sizeof(TA)), 64));
      auto a_tensor = tensor::make_tensor(a_data, a_layout);
      const bool reuse_a = static_cast<nint_t>(tensor::stride<0>(
          a_.input_layout())) == 0;
      bool a_ready = false;
      kernel::loop::for_each_dims<1>(
          [&](const auto& a, const auto& b,
              const auto& c_input, const auto& c_output)
              VECOPS_KERNEL_LAMBDA {
            if (!reuse_a || !a_ready) {
              matmul_pack_batched_a(scope, a, a_tensor);
              a_ready = true;
            }
            if (!reuse_b || !b_ready) {
              run_matmul_pack<Atom, ::vecops::matmul::Operand::B>(
                  scope, b, b_tensor);
              b_ready = true;
            }
            execute_problem(
                scope, tensor::input<TA>(a_tensor),
                tensor::input<TB>(b_tensor), c_input, c_output, scratch);
          },
          a_, b_, c_input_, c_output_);
    } else if constexpr (PackA) {
      const auto layout = auto_packed_layout<::vecops::matmul::Operand::A>(a_);
      using TA = typename Atom::TA;
      auto* data = static_cast<TA*>(workspace.allocate(
          tensor::numel(layout) * static_cast<nint_t>(sizeof(TA)), 64));
      auto packed_tensor = tensor::make_tensor(data, layout);
      const bool reuse = static_cast<nint_t>(tensor::stride<0>(
          a_.input_layout())) == 0;
      bool ready = false;
      const auto b_loop = batch_loop_operand<
          Atom, ::vecops::matmul::Operand::B>(b_);
      kernel::loop::for_each_dims<1>(
          [&](const auto& a, const auto& b,
              const auto& c_input, const auto& c_output)
              VECOPS_KERNEL_LAMBDA {
            if (!reuse || !ready) {
              matmul_pack_batched_a(scope, a, packed_tensor);
              ready = true;
            }
            execute_problem(
                scope, tensor::input<TA>(packed_tensor),
                batch_loop_leaf(b),
                c_input, c_output, scratch);
          },
          a_, b_loop, c_input_, c_output_);
    } else {
      const auto layout = auto_packed_layout<::vecops::matmul::Operand::B>(b_);
      using TB = typename Atom::TB;
      auto* data = static_cast<TB*>(workspace.allocate(
          tensor::numel(layout) * static_cast<nint_t>(sizeof(TB)), 64));
      auto packed_tensor = tensor::make_tensor(data, layout);
      const bool reuse = static_cast<nint_t>(tensor::stride<0>(
          b_.input_layout())) == 0;
      bool ready = false;
      if constexpr (BatchRowsFlattenCandidate) {
        if (VECOPS_LIKELY(batch_rows_flatten_enabled())) {
          const auto b = tensor::slice_view<0>(b_, 0);
          run_matmul_pack<Atom, ::vecops::matmul::Operand::B>(
              scope, b, packed_tensor);
          if constexpr (BatchRowsPackACandidate) {
            if (VECOPS_UNLIKELY(batch_rows_pack_a_enabled())) {
              const auto a_layout = batch_rows_packed_a_layout();
              using TA = typename Atom::TA;
              auto* a_data = static_cast<TA*>(workspace.allocate(
                  tensor::numel(a_layout) *
                      static_cast<nint_t>(sizeof(TA)),
                  64));
              auto a_tensor = tensor::make_tensor(a_data, a_layout);
              const auto batch = tensor::size<0>(
                  c_output_.output_layout());
              const auto flat_m = batch * m_;
              const auto a = flatten_batch_rows_input(
                  a_, flat_m, k_, k_);
              run_matmul_pack<Atom, ::vecops::matmul::Operand::A>(scope, a, a_tensor);
              execute_flattened_batch_rows_operands(
                  scope, tensor::input<TA>(a_tensor),
                  tensor::input<TB>(packed_tensor), scratch);
              workspace.rewind(mark);
              return;
            }
          }
          execute_flattened_batch_rows(
              scope, tensor::input<TB>(packed_tensor), scratch);
          workspace.rewind(mark);
          return;
        }
      }
      const auto a_loop = batch_loop_operand<
          Atom, ::vecops::matmul::Operand::A>(a_);
      auto run_loop = [&]<bool Configured>(auto& active_scope)
          VECOPS_KERNEL_LAMBDA {
        kernel::loop::for_each_dims<1>(
            [&](const auto& a, const auto& b,
                const auto& c_input, const auto& c_output)
                VECOPS_KERNEL_LAMBDA {
              if (!reuse || !ready) {
                run_matmul_pack<Atom, ::vecops::matmul::Operand::B>(
                    active_scope, b, packed_tensor);
                ready = true;
              }
              execute_problem<Configured>(
                  active_scope, batch_loop_leaf(a),
                  tensor::input<TB>(packed_tensor),
                  c_input, c_output, scratch);
            },
            a_loop, b_, c_input_, c_output_);
      };
      if constexpr (std::same_as<
                        Implementation,
                        kernel::matmul_implementation::AMX>) {
        kernel::with_matmul_configuration<Atom, TilePolicy, true>(
            scope, m_, n_, Implementation{},
            [&](auto& configured) VECOPS_INLINE_LAMBDA_NOEXCEPT {
              run_loop.template operator()<false>(configured);
            });
      } else {
        run_loop.template operator()<false>(scope);
      }
    }
    workspace.rewind(mark);
  }

  template <bool PackA = AutoPackA, bool PackB = AutoPackB,
            execution::ExecutionScope Scope>
  VECOPS_ALWAYS_INLINE void execute_auto_packed_in_scope(Scope& scope) const {
    static_assert(PackA || PackB);
    static_assert(!PackA || AutoPackA);
    static_assert(!PackB || AutoPackB);
    if constexpr (COutputSpec::OutputTensor::Ndim == 3) {
      execute_auto_packed_batched_in_scope<PackA, PackB>(scope);
    } else {
      auto& workspace = scope.workspace_view();
      const auto mark = workspace.mark();
      void* scratch = nullptr;
      if constexpr (std::same_as<
                        Implementation, kernel::matmul_implementation::AMX>) {
        scratch = workspace.allocate(
            kernel::matmul_implementation::scratch_bytes<Implementation>(),
            64);
      }

      if constexpr (PackA && PackB) {
        const auto a_layout = ::vecops::matmul::packed_layout<Atom, ::vecops::matmul::Operand::A>(
            a_.input_layout());
        const auto b_layout = ::vecops::matmul::packed_layout<Atom, ::vecops::matmul::Operand::B>(
            b_.input_layout());
        using TA = typename Atom::TA;
        using TB = typename Atom::TB;
        const nint_t a_bytes =
            tensor::numel(a_layout) * static_cast<nint_t>(sizeof(TA));
        const nint_t b_bytes =
            tensor::numel(b_layout) * static_cast<nint_t>(sizeof(TB));
        auto* a_data = static_cast<TA*>(workspace.allocate(a_bytes, 64));
        auto* b_data = static_cast<TB*>(workspace.allocate(b_bytes, 64));
        auto a_tensor = tensor::make_tensor(a_data, a_layout);
        auto b_tensor = tensor::make_tensor(b_data, b_layout);
        run_matmul_pack<Atom, ::vecops::matmul::Operand::A>(scope, a_, a_tensor);
        run_matmul_pack<Atom, ::vecops::matmul::Operand::B>(scope, b_, b_tensor);
        execute_problem(
            scope, tensor::input<TA>(a_tensor), tensor::input<TB>(b_tensor),
            c_input_, c_output_, scratch);
      } else if constexpr (PackA) {
        const auto layout = ::vecops::matmul::packed_layout<Atom, ::vecops::matmul::Operand::A>(
            a_.input_layout());
        using TA = typename Atom::TA;
        const nint_t bytes =
            tensor::numel(layout) * static_cast<nint_t>(sizeof(TA));
        auto* data = static_cast<TA*>(workspace.allocate(bytes, 64));
        auto packed_tensor = tensor::make_tensor(data, layout);
        run_matmul_pack<Atom, ::vecops::matmul::Operand::A>(scope, a_, packed_tensor);
        execute_problem(
            scope, tensor::input<TA>(packed_tensor), b_,
            c_input_, c_output_, scratch);
      } else {
        const auto layout = ::vecops::matmul::packed_layout<Atom, ::vecops::matmul::Operand::B>(
            b_.input_layout());
        using TB = typename Atom::TB;
        const nint_t bytes =
            tensor::numel(layout) * static_cast<nint_t>(sizeof(TB));
        auto* data = static_cast<TB*>(workspace.allocate(bytes, 64));
        auto packed_tensor = tensor::make_tensor(data, layout);
        run_matmul_pack<Atom, ::vecops::matmul::Operand::B>(scope, b_, packed_tensor);
        execute_problem(
            scope, a_, tensor::input<TB>(packed_tensor),
            c_input_, c_output_, scratch);
      }
      workspace.rewind(mark);
    }
  }

  template <bool PackA = AutoPackA, bool PackB = AutoPackB,
            execution::ExecutionScope Scope>
  VECOPS_NOINLINE void execute_auto_packed(Scope& scope) const {
    PackingPlanner::template execute<PackA, PackB>(scope, *this);
  }

  VECOPS_INLINE void validate() const {
    constexpr int Rank = COutputSpec::OutputTensor::Ndim;
    static_assert(ProblemRank >= 2);
    static_assert(Rank >= ProblemRank,
                  "matmul rank is smaller than the backend problem rank");
    static_assert(CInputSpec::InputTensor::Ndim == Rank,
                  "matmul C input/output ranks must match");
    static_assert(
        APackingPlacement::prepared_input !=
                ::vecops::matmul::PreparedInputRequirement::required ||
            PackedAInput,
        "matmul A packing policy requires caller-prepared packed input");
    static_assert(
        BPackingPlacement::prepared_input !=
                ::vecops::matmul::PreparedInputRequirement::required ||
            PackedBInput,
        "matmul B packing policy requires caller-prepared packed input");
    static_assert(
        APackingPlacement::requirement !=
                ::vecops::matmul::PackingRequirement::required ||
            AllowsOutsidePacking<::vecops::matmul::Operand::A> ||
            PackedAInput,
        "architecture matmul cannot force an inside-only A pack; select "
        "GenericTiled or allow outside packing");
    static_assert(
        BPackingPlacement::requirement !=
                ::vecops::matmul::PackingRequirement::required ||
            AllowsOutsidePacking<::vecops::matmul::Operand::B> ||
            PackedBInput,
        "architecture matmul cannot force an inside-only B pack; select "
        "GenericTiled or allow outside packing");
    static_assert(
        !RequiresOutsidePacking<::vecops::matmul::Operand::A> ||
            PackedAInput || AutoPackA,
        "required outside A packing is unsupported for this operand/family");
    static_assert(
        !RequiresOutsidePacking<::vecops::matmul::Operand::B> ||
            PackedBInput || AutoPackB,
        "required outside B packing is unsupported for this operand/family");
    const nint_t m = static_cast<nint_t>(m_);
    const nint_t n = static_cast<nint_t>(n_);
    const nint_t k = static_cast<nint_t>(k_);
    VECOPS_ASSERT(m >= 0 && n >= 0 && k >= 0,
                  "matmul extents must be non-negative");
    validate_input<
        Atom, ::vecops::matmul::Operand::A, Rank>(
            a_, c_output_.output_layout(), m, k);
    validate_input<
        Atom, ::vecops::matmul::Operand::B, Rank>(
            b_, c_output_.output_layout(), n, k);
    for (int d = 0; d < Rank; ++d) {
      VECOPS_ASSERT(
          c_input_.input_layout().shape()[d] ==
              c_output_.output_layout().shape()[d],
          "matmul C input/output shape mismatch");
    }
    VECOPS_ASSERT(
        static_cast<nint_t>(tensor::size<Rank - 2>(
            c_output_.output_layout())) == m &&
        static_cast<nint_t>(tensor::size<Rank - 1>(
            c_output_.output_layout())) == n,
        "matmul C shape mismatch");
  }

  template <bool Configured = false,
            execution::ExecutionScope Scope,
            meta::ValueType M, meta::ValueType N, meta::ValueType K,
            typename ALeaf, typename BLeaf,
            typename CInputLeaf, typename COutputLeaf>
  VECOPS_KERNEL_FUNCTION(void execute_problem_extents(
      Scope& scope, M m, N n, K k,
      const ALeaf& a, const BLeaf& b,
      const CInputLeaf& c_input, const COutputLeaf& c_output,
      void* scratch) const) {
    using APolicy = tensor::InputAccessPolicy<
        ALeaf::InputTensor::Ndim - 1, 1, tensor::AccessPlan::direct>;
    using BPolicy = tensor::InputAccessPolicy<
        BLeaf::InputTensor::Ndim - 1, 1, tensor::AccessPlan::direct>;
    using CInputPolicy = tensor::InputAccessPolicy<
        CInputLeaf::InputTensor::Ndim - 1, 1, tensor::AccessPlan::direct>;
    using COutputPolicy = tensor::OutputAccessPolicy<
        COutputLeaf::OutputTensor::Ndim - 1, tensor::AccessPlan::direct>;
    kernel::with_operands(
        scope,
        tensor::operand(a, APolicy{}),
        tensor::operand(b, BPolicy{}),
        tensor::operand(c_input, CInputPolicy{}),
        tensor::operand(c_output, COutputPolicy{}),
        [&](auto& a_access, auto& b_access,
            auto& c_input_access, auto& c_output_access)
            VECOPS_KERNEL_LAMBDA {
          if constexpr (Configured) {
            kernel::matmul_bound_configured<Atom, TilePolicy>(
                scope, m, n, k, a_access, b_access,
                c_input_access, c_output_access, scratch, Implementation{});
          } else {
            kernel::matmul_bound<
                Atom, TilePolicy, true,
                EffectiveFamilyDispatch>(
                scope, m, n, k, a_access, b_access,
                c_input_access, c_output_access, scratch, Implementation{});
          }
          c_output_access.commit();
        });
  }

  template <bool Configured = false,
            execution::ExecutionScope Scope,
            typename ALeaf, typename BLeaf,
            typename CInputLeaf, typename COutputLeaf>
  VECOPS_KERNEL_FUNCTION(void execute_problem(
      Scope& scope, const ALeaf& a, const BLeaf& b,
      const CInputLeaf& c_input, const COutputLeaf& c_output,
      void* scratch) const) {
    execute_problem_extents<Configured>(
        scope, m_, n_, k_, a, b, c_input, c_output, scratch);
  }

  template <execution::ExecutionScope Scope,
            typename ALeaf, typename BLeaf>
  VECOPS_ALWAYS_INLINE void execute_flattened_batch_rows_operands(
      Scope& scope, const ALeaf& a, const BLeaf& b, void* scratch) const {
    static_assert(BatchRowsFlattenCandidate);
    const auto batch = tensor::size<0>(c_output_.output_layout());
    const auto flat_m = batch * m_;
    const auto ci_stride = tensor::stride<1>(
        c_input_.input_layout());
    const auto c_input = flatten_batch_rows_input(
        c_input_, flat_m, n_, ci_stride);
    const auto c_output = flatten_batch_rows_output(
        c_output_, flat_m, n_, n_);
    constexpr bool FlatPackedB = ::vecops::matmul::is_packed_layout<
        Atom, ::vecops::matmul::Operand::B, typename BLeaf::InputLayout>();
    if constexpr (std::same_as<
                      Implementation, kernel::matmul_implementation::AMX>) {
      kernel::with_matmul_configuration<Atom, TilePolicy, FlatPackedB>(
          scope, flat_m, n_, Implementation{},
          [&](auto& configured) VECOPS_INLINE_LAMBDA_NOEXCEPT {
            execute_problem_extents<false>(
                configured, flat_m, n_, k_, a, b,
                c_input, c_output, scratch);
          });
    } else {
      execute_problem_extents(
          scope, flat_m, n_, k_, a, b,
          c_input, c_output, scratch);
    }
  }

  template <execution::ExecutionScope Scope, typename BLeaf>
  VECOPS_ALWAYS_INLINE void execute_flattened_batch_rows(
      Scope& scope, const BLeaf& b, void* scratch) const {
    static_assert(BatchRowsFlattenCandidate);
    const auto batch = tensor::size<0>(c_output_.output_layout());
    const auto flat_m = batch * m_;
    const auto a = flatten_batch_rows_input(a_, flat_m, k_, k_);
    execute_flattened_batch_rows_operands(scope, a, b, scratch);
  }

  template <execution::ExecutionScope Scope>
  VECOPS_ALWAYS_INLINE void execute_packed_a_flattened_batch_rows(
      Scope& scope) const {
    static_assert(BatchRowsPackACandidate && PackedBInput);
    auto& workspace = scope.workspace_view();
    const auto mark = workspace.mark();
    void* scratch = workspace.allocate(
        kernel::matmul_implementation::scratch_bytes<Implementation>(), 64);
    const auto a_layout = batch_rows_packed_a_layout();
    using TA = typename Atom::TA;
    auto* a_data = static_cast<TA*>(workspace.allocate(
        tensor::numel(a_layout) * static_cast<nint_t>(sizeof(TA)), 64));
    auto a_tensor = tensor::make_tensor(a_data, a_layout);
    const auto batch = tensor::size<0>(c_output_.output_layout());
    const auto flat_m = batch * m_;
    const auto a = flatten_batch_rows_input(a_, flat_m, k_, k_);
    run_matmul_pack<Atom, ::vecops::matmul::Operand::A>(scope, a, a_tensor);
    const auto b = shared_b_leaf();
    execute_flattened_batch_rows_operands(
        scope, tensor::input<TA>(a_tensor), b, scratch);
    workspace.rewind(mark);
  }

  template <execution::ExecutionScope Scope, typename ALeaf>
  VECOPS_ALWAYS_INLINE void execute_flattened_batch_columns(
      Scope& scope, const ALeaf& a, void* scratch) const {
    static_assert(BatchColumnsFlattenCandidate);
    auto& workspace = scope.workspace_view();
    const auto workspace_mark = workspace.mark();
    const auto batch = tensor::size<0>(c_output_.output_layout());
    const auto flat_n = batch * n_;
    const auto b_row_stride = tensor::stride<1>(b_.input_layout());
    const auto b = flatten_batch_rows_input(
        b_, flat_n, k_, b_row_stride);
    const auto c_output = flatten_batch_rows_output(
        c_output_, meta::cint<1>, flat_n, flat_n);
    auto run = [&](const auto& c_input) VECOPS_INLINE_LAMBDA {
      if constexpr (std::same_as<
                        Implementation, kernel::matmul_implementation::AMX>) {
        kernel::with_matmul_configuration<Atom, TilePolicy, false>(
            scope, m_, flat_n, Implementation{},
            [&](auto& configured) VECOPS_INLINE_LAMBDA_NOEXCEPT {
              execute_problem_extents<false>(
                  configured, m_, flat_n, k_, a, b,
                  c_input, c_output, scratch);
            });
      } else {
        execute_problem_extents(
            scope, m_, flat_n, k_, a, b,
            c_input, c_output, scratch);
      }
    };
    if constexpr (BatchColumnsPeriodicCInputCandidate) {
      using Acc = typename Atom::TAcc;
      auto* flat_data = reinterpret_cast<Acc*>(c_input_.tensor().data());
      if (VECOPS_UNLIKELY(batch_columns_periodic_c_input_enabled())) {
        auto* dense = static_cast<Acc*>(workspace.allocate(
            static_cast<nint_t>(flat_n) *
                static_cast<nint_t>(sizeof(Acc)), 64));
        for (nint_t batch_index = 0;
             batch_index < static_cast<nint_t>(batch); ++batch_index)
          for (nint_t column = 0;
               column < static_cast<nint_t>(n_); ++column)
            dense[batch_index * static_cast<nint_t>(n_) + column] =
                flat_data[column];
        flat_data = dense;
      }
      // Build one direct rank-2 type for both dense and materialized input so
      // the compiler emits a single fused Matmul leaf per output pipeline.
      const auto flat_tensor = tensor::make_tensor(
          flat_data, tensor::make_layout(
              tensor::make_shape(meta::cint<1>, flat_n),
              tensor::make_strides(flat_n, meta::cint<1>)));
      run(tensor::input<Acc>(flat_tensor));
    } else {
      run(flatten_batch_rows_input(
          c_input_, meta::cint<1>, flat_n, flat_n));
    }
    workspace.rewind(workspace_mark);
  }

  /// Execute the ordinary un-packed branch, owning AMX scratch for exactly
  /// the duration of the traversal.  SME needs no software scratch.
  template <execution::ExecutionScope Scope, typename Fn>
  VECOPS_ALWAYS_INLINE static void run_without_auto_packing(
      Scope& scope, Fn&& fn) {
    if constexpr (std::same_as<
                      Implementation, kernel::matmul_implementation::AMX>) {
      auto& workspace = scope.workspace_view();
      const auto mark = workspace.mark();
      std::forward<Fn>(fn)(workspace.allocate(
          kernel::matmul_implementation::scratch_bytes<Implementation>(), 64));
      workspace.rewind(mark);
    } else {
      std::forward<Fn>(fn)(nullptr);
    }
  }

  /** Pack one bounded full-K B panel and consume it across the complete M
   * traversal before advancing N. This is the WholeProblem counterpart of
   * GenericTiled's outside-reuse-loop lifetime, without split-K accumulator
   * traffic or a second family instantiation at the public call site. */
  template <execution::ExecutionScope Scope>
  VECOPS_NOINLINE void execute_whole_b_panel(Scope& scope) const {
    static_assert(WholeBPanelCandidate);
    auto& workspace = scope.workspace_view();
    const auto mark = workspace.mark();
    void* scratch = workspace.allocate(
        kernel::matmul_implementation::scratch_bytes<Implementation>(), 64);
    const auto max_layout = whole_b_panel_layout();
    using TB = typename Atom::TB;
    auto* packed_data = static_cast<TB*>(workspace.allocate(
        tensor::numel(max_layout) * static_cast<nint_t>(sizeof(TB)), 64));

    const nint_t panel_n = whole_b_panel_n();
    const meta::Any configured_n{
        std::min<nint_t>(panel_n, static_cast<nint_t>(n_))};
    kernel::with_matmul_configuration<Atom, TilePolicy, true>(
        scope, m_, configured_n, Implementation{},
        [&](auto& configured) VECOPS_INLINE_LAMBDA_NOEXCEPT {
          const nint_t logical_n = static_cast<nint_t>(n_);
          for (nint_t origin = 0; origin < logical_n;
               origin += panel_n) {
            const meta::Any active_n{
                std::min<nint_t>(panel_n, logical_n - origin)};
            auto b_panel = tensor::narrow_view<0>(b_, origin, active_n);
            const auto panel_layout = ::vecops::matmul::packed_layout<
                Atom, ::vecops::matmul::Operand::B>(
                    b_panel.input_layout());
            auto packed_tensor = tensor::make_tensor(
                packed_data, panel_layout);
            run_matmul_pack<Atom, ::vecops::matmul::Operand::B>(
                configured, b_panel, packed_tensor);
            auto c_input_panel = tensor::narrow_view<1>(
                c_input_, origin, active_n);
            auto c_output_panel = tensor::narrow_view<1>(
                c_output_, origin, active_n);
            execute_problem_extents<true>(
                configured, m_, active_n, k_, a_,
                tensor::input<TB>(packed_tensor),
                c_input_panel, c_output_panel, scratch);
          }
        });
    workspace.rewind(mark);
  }

  template <execution::ExecutionScope Scope>
  VECOPS_KERNEL_FUNCTION(void execute(Scope& scope) const) {
    validate();
    if constexpr (WholeBPanelCandidate) {
      if (VECOPS_UNLIKELY(whole_b_panel_enabled())) {
        execute_whole_b_panel(scope);
        return;
      }
    }
    if constexpr (BatchRowsPackACandidate && PackedBInput) {
      if (VECOPS_UNLIKELY(batch_rows_pack_a_enabled())) {
        execute_packed_a_flattened_batch_rows(scope);
        return;
      }
    }
    constexpr int PrefixRank = Rank - ProblemRank;
    const auto a_loop = batch_loop_operand<
        Atom, ::vecops::matmul::Operand::A>(a_);
    const auto b_loop = batch_loop_operand<
        Atom, ::vecops::matmul::Operand::B>(b_);
    auto run_loop = [&]<bool Configured>(
                        auto& active_scope, void* scratch)
        VECOPS_KERNEL_LAMBDA {
      kernel::loop::for_each_dims<PrefixRank>(
          [this, &active_scope, scratch](
              const auto& a, const auto& b,
              const auto& c_input, const auto& c_output)
              VECOPS_KERNEL_LAMBDA {
            execute_problem<Configured>(
                active_scope, batch_loop_leaf(a),
                batch_loop_leaf(b),
                c_input, c_output, scratch);
          },
          a_loop, b_loop, c_input_, c_output_);
    };
    auto run = [&](void* scratch) VECOPS_KERNEL_LAMBDA {
      if constexpr (PrefixRank > 0) {
        for (int d = 0; d < PrefixRank; ++d)
          if (c_output_.output_layout().shape()[d] == 0) return;
      }
      if constexpr (BatchColumnsFlattenCandidate) {
        if (VECOPS_LIKELY(batch_columns_flatten_enabled())) {
          const auto a = shared_a_leaf();
          execute_flattened_batch_columns(scope, a, scratch);
          return;
        }
      }
      if constexpr (DirectBatchRowsFlattenCandidate) {
        if (VECOPS_LIKELY(batch_rows_flatten_enabled())) {
          const auto b = shared_b_leaf();
          execute_flattened_batch_rows(scope, b, scratch);
          return;
        }
      }
      if constexpr (
          std::same_as<
              Implementation, kernel::matmul_implementation::AMX> &&
          PrefixRank > 0) {
        kernel::with_matmul_configuration<
            Atom, TilePolicy, PackedBInput>(
                scope, m_, n_, Implementation{},
                [&](auto& configured) VECOPS_INLINE_LAMBDA_NOEXCEPT {
                  run_loop.template operator()<false>(configured, scratch);
                });
      } else {
        run_loop.template operator()<false>(scope, scratch);
      }
    };
    if constexpr (StaticAutoPackA && StaticAutoPackB) {
      execute_auto_packed<true, true>(scope);
    } else if constexpr (StaticAutoPackA) {
      execute_auto_packed<true, false>(scope);
    } else if constexpr (StaticAutoPackB) {
      execute_auto_packed<false, true>(scope);
    } else if constexpr (RuntimeAutoPackB) {
      if (VECOPS_LIKELY(online_packs_b()))
        execute_auto_packed<false, true>(scope);
      else
        run_without_auto_packing(scope, run);
    } else if constexpr (std::same_as<
                             Implementation,
                             kernel::matmul_implementation::AMX>) {
      auto& workspace = scope.workspace_view();
      const auto mark = workspace.mark();
      run(workspace.allocate(
          kernel::matmul_implementation::scratch_bytes<Implementation>(), 64));
      workspace.rewind(mark);
    } else {
      run(nullptr);
    }
  }

  MExtent m_;
  NExtent n_;
  KExtent k_;
  ASpec a_;
  BSpec b_;
  CInputSpec c_input_;
  COutputSpec c_output_;
  [[no_unique_address]] BatchRowsPackAState batch_rows_pack_a_{};
};

/** Build COutput = CInput + A*B^T with explicit prologue and epilogue operands. */
template <typename Config,
          meta::ValueInput M,
          meta::ValueInput N,
          meta::ValueInput K,
          tensor::InputOperand A, tensor::InputOperand B,
          tensor::InputOperand CInput, tensor::OutputOperand COutput>
VECOPS_INLINE auto make_matmul_invocation(
    const Config&,
    M&& m, N&& n, K&& k,
    A&& a, B&& b, CInput&& c_input, COutput&& c_output) {
  using Atom = typename Config::Atom;
  using TilePolicy = typename Config::SchedulerPolicy;
  using Family = ::vecops::matmul::details::selected_family_t<Config>;
  static_assert(::vecops::matmul::kernel_family::ArchitectureFamily<Family>,
                "make_matmul_invocation requires an architecture family");
  using FamilyDispatch = ::vecops::matmul::details::FamilyDispatch<
      Family, ::vecops::matmul::details::family_selection_mode_v<Config>>;
  using PackingTuning = config_packing_tuning_t<Config>;
  using LogicalAPacking = typename PackingTuning::APacking;
  using LogicalBPacking = typename PackingTuning::BPacking;
  auto m_value = meta::to_value(std::forward<M>(m));
  auto n_value = meta::to_value(std::forward<N>(n));
  auto k_value = meta::to_value(std::forward<K>(k));
  auto a_spec = tensor::as_input_spec<typename Atom::TA>(std::forward<A>(a));
  auto b_spec = tensor::as_input_spec<typename Atom::TB>(std::forward<B>(b));
  auto c_input_spec = tensor::as_input_spec<typename Atom::TAcc>(
      std::forward<CInput>(c_input));
  auto c_output_spec = tensor::as_output_spec<typename Atom::TAcc>(
      std::forward<COutput>(c_output));
  constexpr bool SwapAB = orientation::swap_ab_v<
      Config, decltype(m_value), decltype(n_value), decltype(k_value),
      decltype(a_spec), decltype(b_spec),
      decltype(c_input_spec), decltype(c_output_spec)>;
  if constexpr (SwapAB) {
    using SwappedAtom = typename Atom::SwappedAtom;
    using PackingFamilyDispatch = attach_packing_policy_t<
        FamilyDispatch, LogicalBPacking, LogicalAPacking>;
    using PhysicalFamilyDispatch = attach_spatial_traversal_t<
        Config, true, PackingFamilyDispatch,
        decltype(n_value), decltype(m_value), decltype(k_value)>;
    auto transposed_c_input = tensor::transpose_view<0, 1>(c_input_spec);
    auto transposed_c_output = tensor::transpose_view<0, 1>(c_output_spec);
    return ArchitectureFamilyInvocation<
        SwappedAtom, TilePolicy, PhysicalFamilyDispatch, true,
        decltype(n_value), decltype(m_value), decltype(k_value),
        decltype(b_spec), decltype(a_spec),
        decltype(transposed_c_input), decltype(transposed_c_output)>{
            n_value, m_value, k_value,
            std::move(b_spec), std::move(a_spec),
            std::move(transposed_c_input), std::move(transposed_c_output)};
  } else {
    using PackingFamilyDispatch = attach_packing_policy_t<
        FamilyDispatch, LogicalAPacking, LogicalBPacking>;
    using PhysicalFamilyDispatch = attach_spatial_traversal_t<
        Config, false, PackingFamilyDispatch,
        decltype(m_value), decltype(n_value), decltype(k_value)>;
    return ArchitectureFamilyInvocation<
        Atom, TilePolicy, PhysicalFamilyDispatch, false,
        decltype(m_value), decltype(n_value), decltype(k_value),
        decltype(a_spec), decltype(b_spec),
        decltype(c_input_spec), decltype(c_output_spec)>{
            m_value, n_value, k_value,
            std::move(a_spec), std::move(b_spec),
            std::move(c_input_spec), std::move(c_output_spec)};
  }
}

} // namespace vecops::matmul::details

#endif // VECOPS_MATMUL_DETAILS_FAMILIES_ARCHITECTURE_FAMILY_H
