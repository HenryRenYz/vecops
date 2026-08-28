//
// Copyright (c) vecops contributors.
//

#ifndef VECOPS_OPS_MATMUL_H
#define VECOPS_OPS_MATMUL_H

#include <type_traits>
#include <utility>

#include "vecops/Assertion.h"
#include "vecops/execution/ExecutionSession.h"
#include "vecops/gemm/Packing.h"
#include "vecops/kernel/Loop.h"
#include "vecops/kernel/Matmul.h"
#include "vecops/ops/MatmulPack.h"
#include "vecops/tensor/DataAccess.h"

namespace vecops::ops {

namespace matmul_details {

template <typename T>
concept Extent = meta::ValueType<std::remove_cvref_t<T>> ||
    is_int_v<std::remove_cvref_t<T>>;

template <Extent T>
VECOPS_INLINE constexpr auto extent_value(T&& value) {
  using V = meta::to_value_t<std::remove_cvref_t<T>>;
  if constexpr (meta::ValueType<std::remove_cvref_t<T>>) {
    return std::forward<T>(value);
  } else {
    return V{static_cast<nint_t>(value)};
  }
}

template <typename KernelKind>
struct SelectImplementation;

#if defined(HAS_AMX_TILE)
template <>
struct SelectImplementation<gemm::AMXKernelKind> {
  using type = kernel::matmul_implementation::AMX;
};
#endif

#if defined(HAS_SME)
template <>
struct SelectImplementation<gemm::SMEKernelKind> {
  using type = kernel::matmul_implementation::SME;
};
#endif

template <gemm::Atom Atom>
using SelectedImplementation =
    typename SelectImplementation<typename Atom::KernelKind>::type;

template <gemm::Atom Atom, gemm::Operand Side,
          int LogicalRank, typename Spec, typename CLayout>
VECOPS_INLINE void validate_input(
    const Spec& spec, const CLayout& c_layout,
    nint_t spatial, nint_t k) {
  using Layout = typename Spec::InputLayout;
  if constexpr (gemm::is_packed_layout<Atom, Side, Layout>()) {
    static_assert(
        LogicalRank == 2,
        "current packed matmul formats do not carry batch dimensions");
    using Packing = gemm::packing_t<Atom, Side>;
    static_assert(
        std::same_as<typename Spec::MemoryElement,
                     typename Packing::Element> &&
        std::same_as<typename Spec::TransformType, tensor::NoTransform>,
        "packed matmul operands must have their native dtype and no transform");
    const nint_t panel = [&] {
      if constexpr (requires { Packing::Panel; }) {
        return static_cast<nint_t>(Packing::Panel);
      } else {
        return static_cast<nint_t>(Packing::panel());
      }
    }();
    VECOPS_ASSERT(
        spec.input_layout().shape()[0] * panel >= spatial,
        "packed matmul spatial extent is too small");
    const nint_t padded_k = [&] {
      if constexpr (requires { Packing::KTile; }) {
        return spec.input_layout().shape()[1] * Packing::KTile;
      } else {
        return spec.input_layout().shape()[1] * Packing::KPack;
      }
    }();
    VECOPS_ASSERT(padded_k >= k, "packed matmul K extent is too small");
  } else {
    static_assert(Spec::InputTensor::Ndim == LogicalRank,
                  "unpacked matmul operands must have the same rank as C");
    static_assert(LogicalRank >= 2);
    for (int d = 0; d < LogicalRank - 2; ++d) {
      VECOPS_ASSERT(
          spec.input_layout().shape()[d] == c_layout.shape()[d],
          "matmul batch extent mismatch");
    }
    VECOPS_ASSERT(
        spec.input_layout().shape()[LogicalRank - 2] == spatial &&
        spec.input_layout().shape()[LogicalRank - 1] == k,
        "unpacked matmul operand shape mismatch");
  }
}

} // namespace matmul_details

/**
 * Prepared accelerator matrix multiplication.
 *
 * The logical operation is C[M,N] = C-prologue + A[M,K] * B[N,K]^T,
 * followed by COutput's epilogue. Equal raw leading dimensions are traversed
 * independently before entering the backend problem rank. A and B may
 * independently be raw or in the Atom's packed format for an unbatched
 * problem. Raw operands are converted, transformed, padded, and packed one
 * hardware K step at a time.
 */
template <gemm::Atom Atom,
          typename TilePolicy,
          meta::ValueType MExtent,
          meta::ValueType NExtent,
          meta::ValueType KExtent,
          typename ASpec, typename BSpec,
          typename CInputSpec, typename COutputSpec>
class Matmul {
public:
  using MExtentType = MExtent;
  using NExtentType = NExtent;
  using KExtentType = KExtent;
  using Implementation = matmul_details::SelectedImplementation<Atom>;
  using ResourceRequirements =
      kernel::matmul_implementation::resource_requirements_t<Implementation>;
  static constexpr int ProblemRank =
      kernel::matmul_implementation::problem_rank_v<Implementation>;

  VECOPS_INLINE Matmul(
      MExtent m, NExtent n, KExtent k,
      ASpec a, BSpec b, CInputSpec c_input, COutputSpec c_output)
      : m_(m), n_(n), k_(k),
        a_(std::move(a)), b_(std::move(b)),
        c_input_(std::move(c_input)), c_output_(std::move(c_output)) {
    validate();
    initialize_auto_packing();
  }

  VECOPS_INLINE nint_t required_workspace() const {
    nint_t bytes =
        kernel::matmul_implementation::scratch_bytes<Implementation>();
    if (auto_packing_enabled()) {
      bytes += auto_packed_bytes<gemm::Operand::A>(a_);
      bytes += auto_packed_bytes<gemm::Operand::B>(b_);
    }
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
  template <gemm::Operand Side, typename Spec>
  static constexpr bool AutoPackOperand = [] {
    using Layout = typename Spec::InputLayout;
    using Element = typename gemm::packing_t<Atom, Side>::Element;
    if constexpr (
        !std::same_as<Implementation, kernel::matmul_implementation::SME> ||
        COutputSpec::OutputTensor::Ndim != 2 ||
        gemm::is_packed_layout<Atom, Side, Layout>()) {
      return false;
    } else {
      return sizeof(Element) <= 4 && Spec::InputTensor::Ndim == 2 &&
          std::same_as<typename Spec::MemoryElement, Element> &&
          std::same_as<typename Spec::ComputeType, Element> &&
          std::same_as<typename Spec::TransformType, tensor::NoTransform> &&
          std::same_as<
              tensor::stride_type_t<1, Layout>, meta::Const<1>>;
    }
  }();

  static constexpr bool AutoPackA =
      AutoPackOperand<gemm::Operand::A, ASpec>;
  static constexpr bool AutoPackB =
      AutoPackOperand<gemm::Operand::B, BSpec>;

  struct NoAutoPackingState {};
  using AutoPackingState = std::conditional_t<
      AutoPackA || AutoPackB, bool, NoAutoPackingState>;

  VECOPS_INLINE void initialize_auto_packing() {
    if constexpr (AutoPackA || AutoPackB)
      auto_pack_ = use_auto_packing();
  }

  VECOPS_INLINE bool auto_packing_enabled() const {
    if constexpr (AutoPackA || AutoPackB) return auto_pack_;
    else return false;
  }

  VECOPS_INLINE bool use_auto_packing() const {
    if constexpr (!AutoPackA && !AutoPackB) {
      return false;
    } else {
      const nint_t m = static_cast<nint_t>(m_);
      const nint_t n = static_cast<nint_t>(n_);
      const nint_t k = static_cast<nint_t>(k_);
      if (m <= 0 || n <= 0 || k <= 0) return false;
      // Packing pays for itself once enough output dot products reuse it.
      // Require both aggregate work and spatial reuse; a 1x1 product with a
      // very long K has high work but cannot amortize copying either operand.
      // Express both tests with divisions to avoid overflowing M*N*K.
      constexpr nint_t ReuseThreshold = 128;
      constexpr nint_t WorkThreshold = AutoPackA && AutoPackB
          ? 64 * 1024
          : 128 * 1024;
      if (m < 1 + (ReuseThreshold - 1) / n) return false;
      nint_t remaining = 1 + (WorkThreshold - 1) / m;
      remaining = 1 + (remaining - 1) / n;
      return k >= remaining;
    }
  }

  template <gemm::Operand Side, typename Spec>
  VECOPS_INLINE nint_t auto_packed_bytes(const Spec& spec) const {
    if constexpr (AutoPackOperand<Side, Spec>) {
      using Element = typename gemm::packing_t<Atom, Side>::Element;
      const auto layout = matmul_packed_layout<Atom, Side>(
          spec.input_layout());
      // WorkspaceView may need up to 63 bytes to establish 64B alignment.
      return tensor::numel(layout) * static_cast<nint_t>(sizeof(Element)) + 63;
    } else {
      return 0;
    }
  }

  template <execution::ExecutionScope Scope>
  VECOPS_NOINLINE void execute_auto_packed(Scope& scope) const {
    static_assert(AutoPackA || AutoPackB);
    auto& workspace = scope.workspace_view();
    const auto mark = workspace.mark();

    if constexpr (AutoPackA && AutoPackB) {
      const auto a_layout = matmul_packed_layout<Atom, gemm::Operand::A>(
          a_.input_layout());
      const auto b_layout = matmul_packed_layout<Atom, gemm::Operand::B>(
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
      matmul_pack<Atom, gemm::Operand::A>(scope, a_, a_tensor);
      matmul_pack<Atom, gemm::Operand::B>(scope, b_, b_tensor);
      execute_problem(
          scope, tensor::input<TA>(a_tensor), tensor::input<TB>(b_tensor),
          c_input_, c_output_, nullptr);
    } else if constexpr (AutoPackA) {
      const auto layout = matmul_packed_layout<Atom, gemm::Operand::A>(
          a_.input_layout());
      using TA = typename Atom::TA;
      const nint_t bytes =
          tensor::numel(layout) * static_cast<nint_t>(sizeof(TA));
      auto* data = static_cast<TA*>(workspace.allocate(bytes, 64));
      auto packed_tensor = tensor::make_tensor(data, layout);
      matmul_pack<Atom, gemm::Operand::A>(scope, a_, packed_tensor);
      execute_problem(
          scope, tensor::input<TA>(packed_tensor), b_,
          c_input_, c_output_, nullptr);
    } else {
      const auto layout = matmul_packed_layout<Atom, gemm::Operand::B>(
          b_.input_layout());
      using TB = typename Atom::TB;
      const nint_t bytes =
          tensor::numel(layout) * static_cast<nint_t>(sizeof(TB));
      auto* data = static_cast<TB*>(workspace.allocate(bytes, 64));
      auto packed_tensor = tensor::make_tensor(data, layout);
      matmul_pack<Atom, gemm::Operand::B>(scope, b_, packed_tensor);
      execute_problem(
          scope, a_, tensor::input<TB>(packed_tensor),
          c_input_, c_output_, nullptr);
    }
    workspace.rewind(mark);
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
        Atom, gemm::Operand::A, Rank>(
            a_, c_output_.output_layout(), m, k);
    matmul_details::validate_input<
        Atom, gemm::Operand::B, Rank>(
            b_, c_output_.output_layout(), n, k);
    for (int d = 0; d < Rank; ++d) {
      VECOPS_ASSERT(
          c_input_.input_layout().shape()[d] ==
              c_output_.output_layout().shape()[d],
          "matmul C input/output shape mismatch");
    }
    VECOPS_ASSERT(
        c_output_.output_layout().shape()[Rank - 2] == m &&
        c_output_.output_layout().shape()[Rank - 1] == n,
        "matmul C shape mismatch");
  }

  template <execution::ExecutionScope Scope,
            typename ALeaf, typename BLeaf,
            typename CInputLeaf, typename COutputLeaf>
  VECOPS_KERNEL_FUNCTION(void execute_problem(
      Scope& scope, const ALeaf& a, const BLeaf& b,
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
          kernel::matmul_bound<Atom, TilePolicy>(
              scope, m_, n_, k_, a_access, b_access,
              c_input_access, c_output_access, scratch, Implementation{});
          c_output_access.commit();
        });
  }

  template <execution::ExecutionScope Scope>
  VECOPS_KERNEL_FUNCTION(void execute(Scope& scope) const) {
    validate();
    constexpr int Rank = COutputSpec::OutputTensor::Ndim;
    constexpr int PrefixRank = Rank - ProblemRank;
    auto run = [&](void* scratch) VECOPS_KERNEL_LAMBDA {
      kernel::loop::for_each_dims<PrefixRank>(
          [this, &scope, scratch](const auto& a, const auto& b,
                                  const auto& c_input,
                                  const auto& c_output)
              VECOPS_KERNEL_LAMBDA {
            execute_problem(scope, a, b, c_input, c_output, scratch);
          },
          a_, b_, c_input_, c_output_);
    };
    if constexpr (std::same_as<
                      Implementation, kernel::matmul_implementation::AMX>) {
      auto& workspace = scope.workspace_view();
      const auto mark = workspace.mark();
      run(workspace.allocate(required_workspace(), 64));
      workspace.rewind(mark);
    } else if constexpr (AutoPackA || AutoPackB) {
      if (VECOPS_UNLIKELY(auto_packing_enabled())) {
        execute_auto_packed(scope);
      } else {
        run(nullptr);
      }
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
  [[no_unique_address]] AutoPackingState auto_pack_{};
};

/** Build C = A*B^T with a hardware-zero accumulator prologue. */
template <gemm::Atom Atom,
          typename TilePolicy = kernel::matmul_policy::Automatic,
          matmul_details::Extent M,
          matmul_details::Extent N,
          matmul_details::Extent K,
          tensor::InputOperand A, tensor::InputOperand B,
          tensor::OutputOperand C>
VECOPS_INLINE auto make_matmul(
    M&& m, N&& n, K&& k, A&& a, B&& b, C&& c) {
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
  return Matmul<
      Atom, TilePolicy,
      decltype(m_value), decltype(n_value), decltype(k_value),
      decltype(a_spec), decltype(b_spec),
      decltype(c_input), decltype(c_output)>{
          m_value, n_value, k_value,
          std::move(a_spec), std::move(b_spec),
          std::move(c_input), std::move(c_output)};
}

/** Build C = C-prologue + A*B^T with an explicit C input and output. */
template <gemm::Atom Atom,
          typename TilePolicy = kernel::matmul_policy::Automatic,
          matmul_details::Extent M,
          matmul_details::Extent N,
          matmul_details::Extent K,
          tensor::InputOperand A, tensor::InputOperand B,
          tensor::InputOperand CInput, tensor::OutputOperand COutput>
VECOPS_INLINE auto make_matmul_accumulate(
    M&& m, N&& n, K&& k,
    A&& a, B&& b, CInput&& c_input, COutput&& c_output) {
  auto m_value = matmul_details::extent_value(std::forward<M>(m));
  auto n_value = matmul_details::extent_value(std::forward<N>(n));
  auto k_value = matmul_details::extent_value(std::forward<K>(k));
  auto a_spec = tensor::as_input_spec<typename Atom::TA>(std::forward<A>(a));
  auto b_spec = tensor::as_input_spec<typename Atom::TB>(std::forward<B>(b));
  auto c_input_spec = tensor::as_input_spec<typename Atom::TAcc>(
      std::forward<CInput>(c_input));
  auto c_output_spec = tensor::as_output_spec<typename Atom::TAcc>(
      std::forward<COutput>(c_output));
  return Matmul<
      Atom, TilePolicy,
      decltype(m_value), decltype(n_value), decltype(k_value),
      decltype(a_spec), decltype(b_spec),
      decltype(c_input_spec), decltype(c_output_spec)>{
          m_value, n_value, k_value,
          std::move(a_spec), std::move(b_spec),
          std::move(c_input_spec), std::move(c_output_spec)};
}

template <gemm::Atom Atom,
          typename TilePolicy = kernel::matmul_policy::Automatic,
          execution::ExecutionScope Scope,
          matmul_details::Extent M,
          matmul_details::Extent N,
          matmul_details::Extent K,
          tensor::InputOperand A, tensor::InputOperand B,
          tensor::OutputOperand C>
VECOPS_INLINE void matmul(
    Scope& scope, M&& m, N&& n, K&& k,
    A&& a, B&& b, C&& c) {
  auto operation = make_matmul<Atom, TilePolicy>(
      std::forward<M>(m), std::forward<N>(n), std::forward<K>(k),
      std::forward<A>(a), std::forward<B>(b),
      std::forward<C>(c));
  operation(scope);
}

template <gemm::Atom Atom,
          typename TilePolicy = kernel::matmul_policy::Automatic,
          matmul_details::Extent M,
          matmul_details::Extent N,
          matmul_details::Extent K,
          tensor::InputOperand A, tensor::InputOperand B,
          tensor::OutputOperand C>
VECOPS_INLINE void matmul(
    kernel::WorkspaceView& workspace,
    M&& m, N&& n, K&& k,
    A&& a, B&& b, C&& c) {
  ExecutionSession execution{workspace};
  matmul<Atom, TilePolicy>(
      execution,
      std::forward<M>(m), std::forward<N>(n), std::forward<K>(k),
      std::forward<A>(a), std::forward<B>(b), std::forward<C>(c));
}

} // namespace vecops::ops

#endif // VECOPS_OPS_MATMUL_H
