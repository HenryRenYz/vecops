//
// Copyright (c) vecops contributors.
//

#ifndef VECOPS_MATMUL_DETAILS_FAMILIES_WHOLE_PROBLEM_H
#define VECOPS_MATMUL_DETAILS_FAMILIES_WHOLE_PROBLEM_H

#include <limits>
#include <type_traits>
#include <utility>

#include "vecops/Assertion.h"
#include "vecops/execution/ExecutionSession.h"
#include "vecops/execution/details/arm/Resources.h"
#include "vecops/kernel/Loop.h"
#include "vecops/matmul/Config.h"
#include "vecops/matmul/Packing.h"
#include "vecops/matmul/details/FamilySelector.h"
#include "vecops/matmul/details/Kernel.h"
#include "vecops/matmul/details/OperationCommon.h"
#include "vecops/matmul/details/PackOperation.h"
#include "vecops/tensor/DataAccess.h"

namespace vecops::ops {

template <::vecops::matmul::Atom Atom,
          typename TilePolicy,
          typename FamilyDispatch,
          meta::ValueType MExtent,
          meta::ValueType NExtent,
          meta::ValueType KExtent,
          typename ASpec, typename BSpec,
          typename CInputSpec, typename COutputSpec>
class WholeProblemPlan {
public:
  using MExtentType = MExtent;
  using NExtentType = NExtent;
  using KExtentType = KExtent;
  using Implementation = matmul_details::SelectedImplementation<Atom>;
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
      ::vecops::matmul::details::AutomaticFamilyDispatch>;

  VECOPS_INLINE WholeProblemPlan(
      MExtent m, NExtent n, KExtent k,
      ASpec a, BSpec b, CInputSpec c_input, COutputSpec c_output)
      : m_(m), n_(n), k_(k),
        a_(std::move(a)), b_(std::move(b)),
        c_input_(std::move(c_input)), c_output_(std::move(c_output)) {
    validate();
    initialize_batch_rows_pack_a();
  }

  VECOPS_INLINE nint_t required_workspace() const {
    nint_t bytes =
        kernel::matmul_implementation::scratch_bytes<Implementation>();
    if constexpr (CompileTimeAutoPacking) {
      if constexpr (SMEBatchRowsFullPackCandidate) {
        bytes += batch_rows_flatten_enabled()
            ? batch_rows_packed_a_bytes()
            : auto_packed_bytes<::vecops::matmul::Operand::A>(a_);
      } else {
        bytes += auto_packed_bytes<::vecops::matmul::Operand::A>(a_);
      }
      bytes += auto_packed_bytes<::vecops::matmul::Operand::B>(b_);
    }
    if (batch_rows_pack_a_enabled())
      bytes += this->batch_rows_packed_a_bytes();
    if (batch_columns_periodic_c_input_enabled())
      bytes += batch_columns_periodic_c_input_bytes();
    return bytes;
  }

  template <execution::ExecutionScope Scope>
  VECOPS_INLINE void operator()(Scope& scope) const {
    scope.with_resources(
        *this, [&](auto& active) VECOPS_INLINE_LAMBDA_NOEXCEPT {
          execute(active);
        });
  }

  VECOPS_INLINE void operator()(kernel::WorkspaceView& workspace) const {
    ExecutionSession execution{workspace};
    (*this)(execution);
  }

private:
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
        // A shared rank-three B is logically one rank-two matrix reused by
        // every batch item.  Packing it once can therefore remove the raw-B
        // transpose from every batch traversal.  Keep rank-three A disabled:
        // it is normally independent and would have to be repacked per item.
        constexpr bool ReusableRank = Rank == 2 ||
            (Rank == 3 && Side == ::vecops::matmul::Operand::B);
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
        return ReusableRank &&
            std::same_as<typename Spec::ComputeType, Element> &&
            (std::same_as<Transform, tensor::NoTransform> ||
             SupportedQuantization) &&
            (NativeMemory || SupportedConversion || SupportedQuantization) &&
            std::same_as<
                tensor::stride_type_t<Rank - 1, Layout>, meta::Const<1>>;
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
      return sizeof(Element) <= 8 && Spec::InputTensor::Ndim == Rank &&
          std::same_as<typename Spec::ComputeType, Element> &&
          (std::same_as<Transform, tensor::NoTransform> ||
           SupportedQuantization) &&
          (NativeMemory || SupportedConversion || SupportedQuantization) &&
          std::same_as<
              tensor::stride_type_t<Rank - 1, Layout>, meta::Const<1>>;
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
      return std::same_as<
          tensor::stride_type_t<0, typename BSpec::InputLayout>,
          meta::Const<0>>;
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
      return std::same_as<
          tensor::stride_type_t<0, typename ASpec::InputLayout>,
          meta::Const<0>>;
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
    if constexpr (!BatchRowsFlattenCandidate) {
      return false;
    } else {
      const auto& a_layout = a_.input_layout();
      const auto& ci_layout = c_input_.input_layout();
      const auto& co_layout = c_output_.output_layout();
      const nint_t batch = static_cast<nint_t>(
          tensor::size_value<0>(co_layout));
      const nint_t m = static_cast<nint_t>(m_);
      const nint_t n = static_cast<nint_t>(n_);
      const nint_t k = static_cast<nint_t>(k_);
      if (batch <= 1 || m <= 0 || n <= 0 || k <= 0 || m > 16)
        return false;
      if (batch > 64 / m) return false;
      const nint_t flat_m = batch * m;
      // This first implementation intentionally targets products that need no
      // cache/K tiling.  Larger flattened M values should be handled by the
      // future parallel macro-block path rather than cloned here.
      if (flat_m > 64) return false;

      const bool dense_a =
          static_cast<nint_t>(tensor::stride_value<2>(a_layout)) == 1 &&
          static_cast<nint_t>(tensor::stride_value<1>(a_layout)) == k &&
          static_cast<nint_t>(tensor::stride_value<0>(a_layout)) == m * k;
      const bool dense_co =
          static_cast<nint_t>(tensor::stride_value<2>(co_layout)) == 1 &&
          static_cast<nint_t>(tensor::stride_value<1>(co_layout)) == n &&
          static_cast<nint_t>(tensor::stride_value<0>(co_layout)) == m * n;
      const bool dense_ci =
          static_cast<nint_t>(tensor::stride_value<2>(ci_layout)) == 1 &&
          static_cast<nint_t>(tensor::stride_value<1>(ci_layout)) == n &&
          static_cast<nint_t>(tensor::stride_value<0>(ci_layout)) == m * n;
      const bool broadcast_ci =
          static_cast<nint_t>(tensor::stride_value<2>(ci_layout)) == 1 &&
          static_cast<nint_t>(tensor::stride_value<1>(ci_layout)) == 0 &&
          static_cast<nint_t>(tensor::stride_value<0>(ci_layout)) == 0;
      return dense_a && dense_co && (dense_ci || broadcast_ci);
    }
  }

  VECOPS_INLINE bool batch_columns_flatten_enabled() const {
    if constexpr (!BatchColumnsFlattenCandidate) {
      return false;
    } else {
      const auto& a_layout = a_.input_layout();
      const auto& b_layout = b_.input_layout();
      const auto& ci_layout = c_input_.input_layout();
      const auto& co_layout = c_output_.output_layout();
      const nint_t batch = static_cast<nint_t>(
          tensor::size_value<0>(co_layout));
      const nint_t m = static_cast<nint_t>(m_);
      const nint_t n = static_cast<nint_t>(n_);
      const nint_t k = static_cast<nint_t>(k_);
      if (batch < 4 || m != 1 || n <= 0 || k <= 0) return false;
      if (n > 64 / batch) return false;

      const bool shared_a = [&] {
        if constexpr (PackedAInput) {
          return true;
        } else {
          return static_cast<nint_t>(
                     tensor::stride_value<2>(a_layout)) == 1 &&
              static_cast<nint_t>(
                  tensor::stride_value<1>(a_layout)) == k &&
              static_cast<nint_t>(
                  tensor::stride_value<0>(a_layout)) == 0;
        }
      }();
      const nint_t b_row_stride = static_cast<nint_t>(
          tensor::stride_value<1>(b_layout));
      const bool mergeable_b =
          static_cast<nint_t>(tensor::stride_value<2>(b_layout)) == 1 &&
          b_row_stride >= k &&
          static_cast<nint_t>(tensor::stride_value<0>(b_layout)) ==
              n * b_row_stride;
      const bool dense_co =
          static_cast<nint_t>(tensor::stride_value<2>(co_layout)) == 1 &&
          static_cast<nint_t>(tensor::stride_value<0>(co_layout)) == n;
      // A shared per-N bias has strides {0,0,1}; flattening would require a
      // periodic modulo-N mapping and is therefore deliberately rejected.
      const bool dense_ci =
          static_cast<nint_t>(tensor::stride_value<2>(ci_layout)) == 1 &&
          static_cast<nint_t>(tensor::stride_value<0>(ci_layout)) == n;
      const bool periodic_ci =
          BatchColumnsPeriodicCInputCandidate &&
          static_cast<nint_t>(tensor::stride_value<2>(ci_layout)) == 1 &&
          static_cast<nint_t>(tensor::stride_value<1>(ci_layout)) == 0 &&
          static_cast<nint_t>(tensor::stride_value<0>(ci_layout)) == 0;
      return shared_a && mergeable_b && dense_co &&
          (dense_ci || periodic_ci);
    }
  }

  VECOPS_INLINE bool batch_columns_periodic_c_input_enabled() const {
    if constexpr (!BatchColumnsPeriodicCInputCandidate) {
      return false;
    } else {
      if (!batch_columns_flatten_enabled()) return false;
      const auto& layout = c_input_.input_layout();
      return static_cast<nint_t>(tensor::stride_value<2>(layout)) == 1 &&
          static_cast<nint_t>(tensor::stride_value<1>(layout)) == 0 &&
          static_cast<nint_t>(tensor::stride_value<0>(layout)) == 0;
    }
  }

  VECOPS_INLINE nint_t batch_columns_periodic_c_input_bytes() const {
    if constexpr (!BatchColumnsPeriodicCInputCandidate) {
      return 0;
    } else {
      const nint_t batch = static_cast<nint_t>(tensor::size_value<0>(
          c_output_.output_layout()));
      return batch * static_cast<nint_t>(n_) *
          static_cast<nint_t>(sizeof(typename Atom::TAcc)) + 63;
    }
  }

  template <typename Spec,
            meta::ValueType Rows, meta::ValueType Columns,
            meta::ValueType RowStride>
  VECOPS_INLINE auto flatten_batch_rows_input(
      const Spec& spec, Rows flat_rows, Columns columns,
      RowStride row_stride) const {
    auto tensor = tensor::make_tensor(
        spec.tensor().data(), tensor::make_layout(
            tensor::make_shape(flat_rows, columns),
            tensor::make_strides(row_stride, meta::cint<1>)));
    return tensor::InputSpec<
        typename Spec::ComputeType, decltype(tensor),
        typename Spec::TransformType>{tensor, spec.transform()};
  }

  template <typename Spec,
            meta::ValueType Rows, meta::ValueType Columns,
            meta::ValueType RowStride>
  VECOPS_INLINE auto flatten_batch_rows_output(
      const Spec& spec, Rows flat_rows, Columns columns,
      RowStride row_stride) const {
    auto tensor = tensor::make_tensor(
        spec.tensor().data(), tensor::make_layout(
            tensor::make_shape(flat_rows, columns),
            tensor::make_strides(row_stride, meta::cint<1>)));
    return tensor::OutputSpec<
        typename Spec::ComputeType, decltype(tensor),
        typename Spec::TransformType>{tensor, spec.transform()};
  }

  VECOPS_INLINE auto shared_b_leaf() const {
    if constexpr (PackedBInput) return b_;
    else return tensor::slice_view<0>(b_, 0);
  }

  VECOPS_INLINE auto shared_a_leaf() const {
    if constexpr (PackedAInput) return a_;
    else return tensor::slice_view<0>(a_, 0);
  }

  // Keep the ordinary rank-three traversal byte-for-byte compact.  Enabling
  // the auto-pack branch for a runtime-unknown batch stride adds a large typed
  // fallback to every tiny independent-batch operation.  A statically
  // broadcast B has unambiguous reuse and receives a separate specialization.
  static constexpr bool RankAllowsAutoPack =
      COutputSpec::OutputTensor::Ndim == 2 || Rank3SharedB;
  static constexpr bool AutoPackA = RankAllowsAutoPack &&
      AutoPackOperand<::vecops::matmul::Operand::A, ASpec>;
  static constexpr bool AutoPackB = RankAllowsAutoPack &&
      AutoPackOperand<::vecops::matmul::Operand::B, BSpec>;
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
      Rank3SharedB || std::same_as<Atom, ::vecops::matmul::SME_BF16F32> ||
      std::same_as<Atom, ::vecops::matmul::SME_F32F32>;
#else
      false;
#endif

  static constexpr bool SingleStreamingAutoPackRegion =
      std::same_as<Implementation, kernel::matmul_implementation::SME> &&
      SingleStreamingAutoPackCandidate &&
      // Keep mixed raw/packed operations on the compact existing path.  A
      // rank-three shared operand is already a distinct specialization and
      // amortizes its one remaining pack over the full batch, so completing a
      // packed pair there also merits one region.
      ((AutoPackA && AutoPackB) || Rank3CompletesPackedPair) &&
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
      if constexpr (!PackedBInput && !CompileTimeAutoPacking) return false;
      const nint_t batch = static_cast<nint_t>(tensor::size_value<0>(
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

  VECOPS_INLINE static constexpr bool use_auto_packing_for(
      nint_t m, nint_t n, nint_t k, nint_t effective_m) {
    if constexpr (!AutoPackA && !AutoPackB) {
      return false;
    } else {
      if (m <= 0 || n <= 0 || k <= 0) return false;
      if constexpr (std::same_as<
                        Implementation, kernel::matmul_implementation::AMX>) {
        // Online AMX packing is most valuable for B, whose raw path repeats a
        // 16x16 transpose for each resident M family.  Requiring at least four
        // M tiles avoids paying a full matrix copy for decode, tails, and the
        // 33x35 long-K probe.  When just A is raw, a wide prepacked-B product
        // can amortize making A's tile panels contiguous; the symmetric
        // B-only path retains the M-reuse requirement.
        if constexpr (AutoPackA && AutoPackB) {
          // Native decode/tail products do not reuse a copied B often enough,
          // but conversion/quantization changes the trade-off completely:
          // packing B also removes the strided transform from the resident
          // kernel.  Enhanced scenario measurements show 4--20x wins for
          // transformed small-M shapes, so retain only the N-panel guard for
          // those types and let the work/reuse tests below decide.
          if constexpr (!AutoPackElidesInputWork<
                            ::vecops::matmul::Operand::B, BSpec>) {
            if (m < 64) return false;
          }
          if (n < 16) return false;
        } else if constexpr (AutoPackA) {
          if (n < 256) return false;
        } else {
          static_assert(AutoPackB);
          if constexpr (Rank3SharedB) {
            // Eight decode rows already reuse each packed B panel enough to
            // amortize one copy; the generic work threshold below filters the
            // tiny and short-K shared-weight cases.  Native B packing ceases
            // to pay once the packed weight no longer fits a private L2: the
            // 8 MiB MLP and 25--33 MiB Qwen weights regressed by 7--12% from
            // rereading and rewriting the full matrix.  Conversion-aware
            // packing has a different gate because it also removes repeated
            // conversion/transform work.
            if (effective_m < 8 || n < 16) return false;
            using Memory = typename BSpec::MemoryElement;
            using Element =
                typename ::vecops::matmul::packing_t<Atom, ::vecops::matmul::Operand::B>::Element;
            if constexpr (std::same_as<Memory, Element>) {
              constexpr nint_t MaxPackedElements =
                  (2 * 1024 * 1024) / static_cast<nint_t>(sizeof(Element));
              if (n > MaxPackedElements / k) return false;
            }
          } else {
            // If A is already packed but B still performs conversion or
            // quantization, packing B also removes that transform from the
            // resident kernel. Mirror the two-raw exception above; native
            // and prequantized B retain the M-reuse guard.
            if constexpr (!AutoPackElidesInputWork<
                              ::vecops::matmul::Operand::B, BSpec>) {
              if (m < 64) return false;
            }
            if (n < 16) return false;
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
      if constexpr (AutoPackWidensBF16<::vecops::matmul::Operand::A, ASpec>)
        if (n < WidenReuseThreshold) return false;
      if constexpr (AutoPackWidensBF16<::vecops::matmul::Operand::B, BSpec>)
        if (effective_m < WidenReuseThreshold) return false;
      if constexpr (
          Rank3CompletesPackedPair && AutoPackA &&
          AutoPackQuantizes<::vecops::matmul::Operand::A, ASpec>) {
        // A is different for every batch, so shared B does not amortize its
        // FP32->I8/U8 transform.  The 64x128 boundary regressed, while either
        // twice the N reuse or twice the K work remained profitable.
        if (n < 128 && k < 256) return false;
      }
      // Packing pays for itself once enough output dot products reuse it.
      // Require both aggregate work and spatial reuse; a 1x1 product with a
      // very long K has high work but cannot amortize copying either operand.
      // Express both tests with divisions to avoid overflowing M*N*K.
      constexpr bool AMXBPackElidesInputWork =
          std::same_as<
              Implementation, kernel::matmul_implementation::AMX> &&
          AutoPackB &&
          AutoPackElidesInputWork<::vecops::matmul::Operand::B, BSpec>;
      constexpr bool AMXRank3BOnly =
          std::same_as<
              Implementation, kernel::matmul_implementation::AMX> &&
          Rank3SharedB && !AutoPackA && AutoPackB;
      constexpr bool SMERank3QuantizedPair =
#if defined(VECOPS_DISABLE_SME_RANK3_QUANT_FULL_PACKING)
          false;
#else
          std::same_as<
              Implementation, kernel::matmul_implementation::SME> &&
          Rank3SharedB && AutoPackA && AutoPackB &&
          AutoPackQuantizes<::vecops::matmul::Operand::A, ASpec> &&
          AutoPackQuantizes<::vecops::matmul::Operand::B, BSpec>;
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
          : (AutoPackA && AutoPackB) || Rank3CompletesPackedPair
              ? 64 * 1024
              : 128 * 1024;
      if (effective_m < 1 + (ReuseThreshold - 1) / n) return false;
      nint_t remaining = 1 + (WorkThreshold - 1) / effective_m;
      remaining = 1 + (remaining - 1) / n;
      return k >= remaining;
    }
  }

  static constexpr bool CompileTimeAutoPacking = [] {
    using MV = std::remove_cvref_t<MExtent>;
    using NV = std::remove_cvref_t<NExtent>;
    using KV = std::remove_cvref_t<KExtent>;
    if constexpr ((AutoPackA || AutoPackB) &&
                  COutputSpec::OutputTensor::Ndim == 2 &&
                  MV::is_const && NV::is_const && KV::is_const) {
      constexpr nint_t MValue =
          meta::singleton_value_v<MV>;
      constexpr nint_t NValue =
          meta::singleton_value_v<NV>;
      constexpr nint_t KValue =
          meta::singleton_value_v<KV>;
      return use_auto_packing_for(
          MValue, NValue, KValue, MValue);
    } else if constexpr ((AutoPackA || AutoPackB) && Rank3SharedB &&
                         COutputSpec::OutputTensor::Ndim == 3 &&
                         MV::is_const && NV::is_const && KV::is_const) {
      using Batch = tensor::size_type_t<
          0, typename COutputSpec::OutputLayout>;
      if constexpr (Batch::is_const) {
        constexpr nint_t MValue =
            meta::singleton_value_v<MV>;
        constexpr nint_t NValue =
            meta::singleton_value_v<NV>;
        constexpr nint_t KValue =
            meta::singleton_value_v<KV>;
        constexpr nint_t BatchValue =
            meta::singleton_value_v<Batch>;
        if constexpr (MValue <= 0 || BatchValue <= 0) {
          return false;
        } else {
          constexpr nint_t Limit = std::numeric_limits<nint_t>::max();
          constexpr nint_t EffectiveM = BatchValue > Limit / MValue
              ? Limit
              : MValue * BatchValue;
          return use_auto_packing_for(
              MValue, NValue, KValue, EffectiveM);
        }
      } else {
        return false;
      }
    } else {
      return false;
    }
  }();

  template <::vecops::matmul::Operand Side, typename Spec>
  VECOPS_INLINE auto auto_packed_layout(const Spec& spec) const {
    constexpr int Rank = Spec::InputTensor::Ndim;
    if constexpr (Rank == 2) {
      return matmul_packed_layout<Atom, Side>(spec.input_layout());
    } else {
      static_assert(Rank == 3);
      const auto& layout = spec.input_layout();
      return matmul_packed_layout<Atom, Side>(tensor::make_layout(
          tensor::make_shape(
              tensor::size_value<Rank - 2>(layout),
              tensor::size_value<Rank - 1>(layout))));
    }
  }

  template <::vecops::matmul::Operand Side, typename Spec>
  VECOPS_INLINE nint_t auto_packed_bytes(const Spec& spec) const {
    if constexpr (AutoPackOperand<Side, Spec>) {
      using Element = typename ::vecops::matmul::packing_t<Atom, Side>::Element;
      const auto layout = auto_packed_layout<Side>(spec);
      // WorkspaceView may need up to 63 bytes to establish 64B alignment.
      return tensor::numel(layout) * static_cast<nint_t>(sizeof(Element)) + 63;
    } else {
      return 0;
    }
  }

  VECOPS_INLINE auto batch_rows_packed_a_layout() const {
    static_assert(
        BatchRowsPackACandidate || SMEBatchRowsFullPackCandidate);
    const auto batch = tensor::size_value<0>(c_output_.output_layout());
    const auto flat_m = batch * m_;
    const auto flat_layout = tensor::make_layout(tensor::make_shape(
        flat_m, k_));
    return matmul_packed_layout<Atom, ::vecops::matmul::Operand::A>(flat_layout);
  }

  VECOPS_INLINE nint_t batch_rows_packed_a_bytes() const {
    if constexpr (
        !BatchRowsPackACandidate && !SMEBatchRowsFullPackCandidate) {
      return 0;
    } else {
      const auto layout = batch_rows_packed_a_layout();
      return tensor::numel(layout) *
          static_cast<nint_t>(sizeof(typename Atom::TA)) + 63;
    }
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
      if (static_cast<nint_t>(tensor::size_value<Rank - 2>(
              input_spec.input_layout())) < panel) {
        matmul_pack_forced<
            kernel::matmul_pack_implementation::SMEFP32ToFP64Single>(
                scope, input, output);
        return;
      }
    }
    matmul_pack_details::run_matmul_pack<Atom, ::vecops::matmul::Operand::A>(
        scope, input, output);
  }

  template <execution::ExecutionScope Scope>
  VECOPS_ALWAYS_INLINE void execute_auto_packed_batched_in_scope(
      Scope& scope) const {
    static_assert(COutputSpec::OutputTensor::Ndim == 3);
    static_assert(AutoPackA || AutoPackB);
    auto& workspace = scope.workspace_view();
    const auto mark = workspace.mark();
    void* scratch = nullptr;
    if constexpr (std::same_as<
                      Implementation, kernel::matmul_implementation::AMX>) {
      scratch = workspace.allocate(
          kernel::matmul_implementation::scratch_bytes<Implementation>(), 64);
    }

    if constexpr (AutoPackA && AutoPackB) {
      const auto b_layout = auto_packed_layout<::vecops::matmul::Operand::B>(b_);
      using TB = typename Atom::TB;
      auto* b_data = static_cast<TB*>(workspace.allocate(
          tensor::numel(b_layout) * static_cast<nint_t>(sizeof(TB)), 64));
      auto b_tensor = tensor::make_tensor(b_data, b_layout);
      if constexpr (SMEBatchRowsFullPackCandidate) {
        if (VECOPS_LIKELY(batch_rows_flatten_enabled())) {
          const auto b = tensor::slice_view<0>(b_, 0);
          matmul_pack_details::run_matmul_pack<Atom, ::vecops::matmul::Operand::B>(scope, b, b_tensor);
          const auto a_layout = batch_rows_packed_a_layout();
          using TA = typename Atom::TA;
          auto* a_data = static_cast<TA*>(workspace.allocate(
              tensor::numel(a_layout) * static_cast<nint_t>(sizeof(TA)), 64));
          auto a_tensor = tensor::make_tensor(a_data, a_layout);
          const auto batch = tensor::size_value<0>(
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
      const bool reuse_b = static_cast<nint_t>(tensor::stride_value<0>(
          b_.input_layout())) == 0;
      bool b_ready = false;
      const auto a_layout = auto_packed_layout<::vecops::matmul::Operand::A>(a_);
      using TA = typename Atom::TA;
      auto* a_data = static_cast<TA*>(workspace.allocate(
          tensor::numel(a_layout) * static_cast<nint_t>(sizeof(TA)), 64));
      auto a_tensor = tensor::make_tensor(a_data, a_layout);
      const bool reuse_a = static_cast<nint_t>(tensor::stride_value<0>(
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
              matmul_pack_details::run_matmul_pack<Atom, ::vecops::matmul::Operand::B>(
                  scope, b, b_tensor);
              b_ready = true;
            }
            execute_problem(
                scope, tensor::input<TA>(a_tensor),
                tensor::input<TB>(b_tensor), c_input, c_output, scratch);
          },
          a_, b_, c_input_, c_output_);
    } else if constexpr (AutoPackA) {
      const auto layout = auto_packed_layout<::vecops::matmul::Operand::A>(a_);
      using TA = typename Atom::TA;
      auto* data = static_cast<TA*>(workspace.allocate(
          tensor::numel(layout) * static_cast<nint_t>(sizeof(TA)), 64));
      auto packed_tensor = tensor::make_tensor(data, layout);
      const bool reuse = static_cast<nint_t>(tensor::stride_value<0>(
          a_.input_layout())) == 0;
      bool ready = false;
      const auto b_loop = matmul_details::batch_loop_operand<
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
                matmul_details::batch_loop_leaf(b),
                c_input, c_output, scratch);
          },
          a_, b_loop, c_input_, c_output_);
    } else {
      const auto layout = auto_packed_layout<::vecops::matmul::Operand::B>(b_);
      using TB = typename Atom::TB;
      auto* data = static_cast<TB*>(workspace.allocate(
          tensor::numel(layout) * static_cast<nint_t>(sizeof(TB)), 64));
      auto packed_tensor = tensor::make_tensor(data, layout);
      const bool reuse = static_cast<nint_t>(tensor::stride_value<0>(
          b_.input_layout())) == 0;
      bool ready = false;
      if constexpr (BatchRowsFlattenCandidate) {
        if (VECOPS_LIKELY(batch_rows_flatten_enabled())) {
          const auto b = tensor::slice_view<0>(b_, 0);
          matmul_pack_details::run_matmul_pack<Atom, ::vecops::matmul::Operand::B>(
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
              const auto batch = tensor::size_value<0>(
                  c_output_.output_layout());
              const auto flat_m = batch * m_;
              const auto a = flatten_batch_rows_input(
                  a_, flat_m, k_, k_);
              matmul_pack_details::run_matmul_pack<Atom, ::vecops::matmul::Operand::A>(scope, a, a_tensor);
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
      const auto a_loop = matmul_details::batch_loop_operand<
          Atom, ::vecops::matmul::Operand::A>(a_);
      auto run_loop = [&]<bool Configured>(auto& active_scope)
          VECOPS_KERNEL_LAMBDA {
        kernel::loop::for_each_dims<1>(
            [&](const auto& a, const auto& b,
                const auto& c_input, const auto& c_output)
                VECOPS_KERNEL_LAMBDA {
              if (!reuse || !ready) {
                matmul_pack_details::run_matmul_pack<Atom, ::vecops::matmul::Operand::B>(
                    active_scope, b, packed_tensor);
                ready = true;
              }
              execute_problem<Configured>(
                  active_scope, matmul_details::batch_loop_leaf(a),
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
              run_loop.template operator()<true>(configured);
            });
      } else {
        run_loop.template operator()<false>(scope);
      }
    }
    workspace.rewind(mark);
  }

  template <execution::ExecutionScope Scope>
  VECOPS_ALWAYS_INLINE void execute_auto_packed_in_scope(Scope& scope) const {
    static_assert(AutoPackA || AutoPackB);
    if constexpr (COutputSpec::OutputTensor::Ndim == 3) {
      execute_auto_packed_batched_in_scope(scope);
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

      if constexpr (AutoPackA && AutoPackB) {
        const auto a_layout = matmul_packed_layout<Atom, ::vecops::matmul::Operand::A>(
            a_.input_layout());
        const auto b_layout = matmul_packed_layout<Atom, ::vecops::matmul::Operand::B>(
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
        matmul_pack_details::run_matmul_pack<Atom, ::vecops::matmul::Operand::A>(scope, a_, a_tensor);
        matmul_pack_details::run_matmul_pack<Atom, ::vecops::matmul::Operand::B>(scope, b_, b_tensor);
        execute_problem(
            scope, tensor::input<TA>(a_tensor), tensor::input<TB>(b_tensor),
            c_input_, c_output_, scratch);
      } else if constexpr (AutoPackA) {
        const auto layout = matmul_packed_layout<Atom, ::vecops::matmul::Operand::A>(
            a_.input_layout());
        using TA = typename Atom::TA;
        const nint_t bytes =
            tensor::numel(layout) * static_cast<nint_t>(sizeof(TA));
        auto* data = static_cast<TA*>(workspace.allocate(bytes, 64));
        auto packed_tensor = tensor::make_tensor(data, layout);
        matmul_pack_details::run_matmul_pack<Atom, ::vecops::matmul::Operand::A>(scope, a_, packed_tensor);
        execute_problem(
            scope, tensor::input<TA>(packed_tensor), b_,
            c_input_, c_output_, scratch);
      } else {
        const auto layout = matmul_packed_layout<Atom, ::vecops::matmul::Operand::B>(
            b_.input_layout());
        using TB = typename Atom::TB;
        const nint_t bytes =
            tensor::numel(layout) * static_cast<nint_t>(sizeof(TB));
        auto* data = static_cast<TB*>(workspace.allocate(bytes, 64));
        auto packed_tensor = tensor::make_tensor(data, layout);
        matmul_pack_details::run_matmul_pack<Atom, ::vecops::matmul::Operand::B>(scope, b_, packed_tensor);
        execute_problem(
            scope, a_, tensor::input<TB>(packed_tensor),
            c_input_, c_output_, scratch);
      }
      workspace.rewind(mark);
    }
  }

  template <execution::ExecutionScope Scope>
  VECOPS_NOINLINE void execute_auto_packed(Scope& scope) const {
    if constexpr (SingleStreamingAutoPackRegion) {
      scope.with_resources(
          execution::details::arm::StreamingZARegion{},
          [&](auto& active) VECOPS_INLINE_LAMBDA_NOEXCEPT {
            execute_auto_packed_in_scope(active);
          });
    } else {
      execute_auto_packed_in_scope(scope);
    }
  }

  VECOPS_INLINE void validate() const {
    constexpr int Rank = COutputSpec::OutputTensor::Ndim;
    static_assert(ProblemRank >= 2);
    static_assert(Rank >= ProblemRank,
                  "matmul rank is smaller than the backend problem rank");
    static_assert(CInputSpec::InputTensor::Ndim == Rank,
                  "matmul C input/output ranks must match");
    const nint_t m = static_cast<nint_t>(m_);
    const nint_t n = static_cast<nint_t>(n_);
    const nint_t k = static_cast<nint_t>(k_);
    VECOPS_ASSERT(m >= 0 && n >= 0 && k >= 0,
                  "matmul extents must be non-negative");
    matmul_details::validate_input<
        Atom, ::vecops::matmul::Operand::A, Rank>(
            a_, c_output_.output_layout(), m, k);
    matmul_details::validate_input<
        Atom, ::vecops::matmul::Operand::B, Rank>(
            b_, c_output_.output_layout(), n, k);
    for (int d = 0; d < Rank; ++d) {
      VECOPS_ASSERT(
          c_input_.input_layout().shape()[d] ==
              c_output_.output_layout().shape()[d],
          "matmul C input/output shape mismatch");
    }
    VECOPS_ASSERT(
        static_cast<nint_t>(tensor::size_value<Rank - 2>(
            c_output_.output_layout())) == m &&
        static_cast<nint_t>(tensor::size_value<Rank - 1>(
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
                Atom, TilePolicy, PackedAInput && PackedBInput,
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
    const auto batch = tensor::size_value<0>(c_output_.output_layout());
    const auto flat_m = batch * m_;
    const auto ci_stride = tensor::stride_value<1>(
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
            execute_problem_extents<true>(
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
    const auto batch = tensor::size_value<0>(c_output_.output_layout());
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
    const auto batch = tensor::size_value<0>(c_output_.output_layout());
    const auto flat_m = batch * m_;
    const auto a = flatten_batch_rows_input(a_, flat_m, k_, k_);
    matmul_pack_details::run_matmul_pack<Atom, ::vecops::matmul::Operand::A>(scope, a, a_tensor);
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
    const auto batch = tensor::size_value<0>(c_output_.output_layout());
    const auto flat_n = batch * n_;
    const auto b_row_stride = tensor::stride_value<1>(b_.input_layout());
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
              execute_problem_extents<true>(
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

  template <execution::ExecutionScope Scope>
  VECOPS_KERNEL_FUNCTION(void execute(Scope& scope) const) {
    validate();
    if constexpr (BatchRowsPackACandidate && PackedBInput) {
      if (VECOPS_UNLIKELY(batch_rows_pack_a_enabled())) {
        execute_packed_a_flattened_batch_rows(scope);
        return;
      }
    }
    constexpr int Rank = COutputSpec::OutputTensor::Ndim;
    constexpr int PrefixRank = Rank - ProblemRank;
    const auto a_loop = matmul_details::batch_loop_operand<
        Atom, ::vecops::matmul::Operand::A>(a_);
    const auto b_loop = matmul_details::batch_loop_operand<
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
                active_scope, matmul_details::batch_loop_leaf(a),
                matmul_details::batch_loop_leaf(b),
                c_input, c_output, scratch);
          },
          a_loop, b_loop, c_input_, c_output_);
    };
    auto run = [&](void* scratch) VECOPS_KERNEL_LAMBDA {
      if constexpr (PrefixRank > 0) {
        // Preserve the zero-batch behavior: no leaf means no TILECFG load.
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
                  run_loop.template operator()<true>(configured, scratch);
                });
      } else {
        run_loop.template operator()<false>(scope, scratch);
      }
    };
    if constexpr (AutoPackA || AutoPackB) {
      if constexpr (CompileTimeAutoPacking) {
        execute_auto_packed(scope);
      } else {
        if constexpr (std::same_as<
                          Implementation,
                          kernel::matmul_implementation::AMX>) {
          auto& workspace = scope.workspace_view();
          const auto mark = workspace.mark();
          run(workspace.allocate(
              kernel::matmul_implementation::scratch_bytes<Implementation>(),
              64));
          workspace.rewind(mark);
        } else {
          run(nullptr);
        }
      }
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

namespace matmul_details {

/** Prepare C = A*B^T with a hardware-zero accumulator prologue. */
template <typename Config,
          matmul_details::Extent M,
          matmul_details::Extent N,
          matmul_details::Extent K,
          tensor::InputOperand A, tensor::InputOperand B,
          tensor::OutputOperand C>
VECOPS_INLINE auto prepare_matmul(
    const Config&,
    M&& m, N&& n, K&& k, A&& a, B&& b, C&& c) {
  using Atom = typename Config::Atom;
  using TilePolicy = typename Config::SchedulerPolicy;
  using Family = ::vecops::matmul::details::selected_family_t<Config>;
  static_assert(::vecops::matmul::kernel_family::WholeProblemFamily<Family>,
                "prepare_matmul requires a whole-problem kernel family");
  using FamilyDispatch = ::vecops::matmul::details::FamilyDispatch<
      Family, ::vecops::matmul::details::family_selection_mode_v<Config>>;
  auto m_value = matmul_details::extent_value(std::forward<M>(m));
  auto n_value = matmul_details::extent_value(std::forward<N>(n));
  auto k_value = matmul_details::extent_value(std::forward<K>(k));
  auto a_spec = tensor::as_input_spec<typename Atom::TA>(std::forward<A>(a));
  auto b_spec = tensor::as_input_spec<typename Atom::TB>(std::forward<B>(b));
  auto c_output = tensor::as_output_spec<typename Atom::TAcc>(
      std::forward<C>(c));
  using Memory = typename decltype(c_output)::MemoryElement;
  auto c_input = tensor::input<typename Atom::TAcc>(
      c_output.tensor(),
      tensor::zeros_transform<typename Atom::TAcc, Memory>);
  return WholeProblemPlan<
      Atom, TilePolicy, FamilyDispatch,
      decltype(m_value), decltype(n_value), decltype(k_value),
      decltype(a_spec), decltype(b_spec),
      decltype(c_input), decltype(c_output)>{
          m_value, n_value, k_value,
          std::move(a_spec), std::move(b_spec),
          std::move(c_input), std::move(c_output)};
}

/** Prepare C = C-prologue + A*B^T with explicit C input and output. */
template <typename Config,
          matmul_details::Extent M,
          matmul_details::Extent N,
          matmul_details::Extent K,
          tensor::InputOperand A, tensor::InputOperand B,
          tensor::InputOperand CInput, tensor::OutputOperand COutput>
VECOPS_INLINE auto prepare_matmul_accumulate(
    const Config&,
    M&& m, N&& n, K&& k,
    A&& a, B&& b, CInput&& c_input, COutput&& c_output) {
  using Atom = typename Config::Atom;
  using TilePolicy = typename Config::SchedulerPolicy;
  using Family = ::vecops::matmul::details::selected_family_t<Config>;
  static_assert(::vecops::matmul::kernel_family::WholeProblemFamily<Family>,
                "prepare_matmul_accumulate requires a whole-problem family");
  using FamilyDispatch = ::vecops::matmul::details::FamilyDispatch<
      Family, ::vecops::matmul::details::family_selection_mode_v<Config>>;
  auto m_value = matmul_details::extent_value(std::forward<M>(m));
  auto n_value = matmul_details::extent_value(std::forward<N>(n));
  auto k_value = matmul_details::extent_value(std::forward<K>(k));
  auto a_spec = tensor::as_input_spec<typename Atom::TA>(std::forward<A>(a));
  auto b_spec = tensor::as_input_spec<typename Atom::TB>(std::forward<B>(b));
  auto c_input_spec = tensor::as_input_spec<typename Atom::TAcc>(
      std::forward<CInput>(c_input));
  auto c_output_spec = tensor::as_output_spec<typename Atom::TAcc>(
      std::forward<COutput>(c_output));
  return WholeProblemPlan<
      Atom, TilePolicy, FamilyDispatch,
      decltype(m_value), decltype(n_value), decltype(k_value),
      decltype(a_spec), decltype(b_spec),
      decltype(c_input_spec), decltype(c_output_spec)>{
          m_value, n_value, k_value,
          std::move(a_spec), std::move(b_spec),
          std::move(c_input_spec), std::move(c_output_spec)};
}

} // namespace matmul_details


} // namespace vecops::ops

#endif // VECOPS_MATMUL_DETAILS_FAMILIES_WHOLE_PROBLEM_H
