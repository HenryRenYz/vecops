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
#include "vecops/kernel/Matmul.h"
#include "vecops/tensor/DataAccess.h"

namespace vecops::ops {

namespace matmul_details {

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

template <gemm::Atom Atom, gemm::Operand Side, typename Spec>
VECOPS_INLINE void validate_input(
    const Spec& spec, nint_t spatial, nint_t k) {
  using Layout = typename Spec::InputLayout;
  if constexpr (gemm::is_packed_layout<Atom, Side, Layout>()) {
    using Packing = gemm::packing_t<Atom, Side>;
    static_assert(
        std::same_as<typename Spec::MemoryElement,
                     typename Packing::Element> &&
        std::same_as<typename Spec::TransformType, tensor::NoTransform>,
        "packed matmul operands must have their native dtype and no transform");
    const nint_t panel = [&] {
      if constexpr (requires { Packing::Panel; }) return Packing::Panel;
      else return Packing::panel();
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
    static_assert(Spec::InputTensor::Ndim == 2,
                  "unpacked matmul operands must be rank two");
    VECOPS_ASSERT(
        spec.input_layout().shape()[0] == spatial &&
        spec.input_layout().shape()[1] == k,
        "unpacked matmul operand shape mismatch");
  }
}

} // namespace matmul_details

/**
 * Prepared accelerator matrix multiplication.
 *
 * The logical operation is C[M,N] = C-prologue + A[M,K] * B[N,K]^T,
 * followed by COutput's epilogue. A and B may independently be raw or in the
 * Atom's packed format. Raw operands are converted, transformed, padded, and
 * packed one hardware K step at a time.
 */
template <gemm::Atom Atom,
          typename ASpec, typename BSpec,
          typename CInputSpec, typename COutputSpec>
class Matmul {
public:
  using Implementation = matmul_details::SelectedImplementation<Atom>;
  using ResourceRequirements =
      kernel::matmul_implementation::resource_requirements_t<Implementation>;

  VECOPS_INLINE Matmul(
      nint_t m, nint_t n, nint_t k,
      ASpec a, BSpec b, CInputSpec c_input, COutputSpec c_output)
      : m_(m), n_(n), k_(k),
        a_(std::move(a)), b_(std::move(b)),
        c_input_(std::move(c_input)), c_output_(std::move(c_output)) {
    validate();
  }

  VECOPS_INLINE nint_t required_workspace() const {
    return kernel::matmul_implementation::scratch_bytes<Implementation>();
  }

  template <execution::ExecutionScope Scope>
  VECOPS_INLINE void operator()(Scope& scope) const {
    scope.with_resources(
        *this, [&](auto& active) VECOPS_INLINE_LAMBDA { execute(active); });
  }

  VECOPS_INLINE void operator()(kernel::WorkspaceView& workspace) const {
    ExecutionSession execution{workspace};
    (*this)(execution);
  }

private:
  VECOPS_INLINE void validate() const {
    VECOPS_ASSERT(m_ >= 0 && n_ >= 0 && k_ >= 0,
                  "matmul extents must be non-negative");
    matmul_details::validate_input<Atom, gemm::Operand::A>(a_, m_, k_);
    matmul_details::validate_input<Atom, gemm::Operand::B>(b_, n_, k_);
    static_assert(CInputSpec::InputTensor::Ndim == 2);
    static_assert(COutputSpec::OutputTensor::Ndim == 2);
    VECOPS_ASSERT(
        c_input_.input_layout().shape()[0] == m_ &&
        c_input_.input_layout().shape()[1] == n_ &&
        c_output_.output_layout().shape()[0] == m_ &&
        c_output_.output_layout().shape()[1] == n_,
        "matmul C shape mismatch");
  }

  template <execution::ExecutionScope Scope>
  VECOPS_ALWAYS_INLINE void execute(Scope& scope) const {
    validate();
    using APolicy = tensor::InputAccessPolicy<
        ASpec::InputTensor::Ndim - 1, 1, tensor::AccessPlan::direct>;
    using BPolicy = tensor::InputAccessPolicy<
        BSpec::InputTensor::Ndim - 1, 1, tensor::AccessPlan::direct>;
    using CInputPolicy = tensor::InputAccessPolicy<
        1, 1, tensor::AccessPlan::direct>;
    using COutputPolicy = tensor::OutputAccessPolicy<
        1, tensor::AccessPlan::direct>;
    auto run = [&](void* scratch) VECOPS_INLINE_LAMBDA {
      kernel::with_operands(
          scope,
          tensor::operand(a_, APolicy{}),
          tensor::operand(b_, BPolicy{}),
          tensor::operand(c_input_, CInputPolicy{}),
          tensor::operand(c_output_, COutputPolicy{}),
          [&](auto& a, auto& b, auto& c_input, auto& c_output)
              VECOPS_INLINE_LAMBDA {
            kernel::matmul_bound<Atom>(
                scope, m_, n_, k_, a, b, c_input, c_output,
                scratch, Implementation{});
            c_output.commit();
          });
    };
    if constexpr (std::same_as<
                      Implementation, kernel::matmul_implementation::AMX>) {
      auto& workspace = scope.workspace_view();
      const auto mark = workspace.mark();
      run(workspace.allocate(required_workspace(), 64));
      workspace.rewind(mark);
    } else {
      run(nullptr);
    }
  }

  nint_t m_;
  nint_t n_;
  nint_t k_;
  ASpec a_;
  BSpec b_;
  CInputSpec c_input_;
  COutputSpec c_output_;
};

/** Build C = A*B^T with a hardware-zero accumulator prologue. */
template <gemm::Atom Atom,
          tensor::InputOperand A, tensor::InputOperand B,
          tensor::OutputOperand C>
VECOPS_INLINE auto make_matmul(
    nint_t m, nint_t n, nint_t k, A&& a, B&& b, C&& c) {
  auto a_spec = tensor::as_input_spec<typename Atom::TA>(std::forward<A>(a));
  auto b_spec = tensor::as_input_spec<typename Atom::TB>(std::forward<B>(b));
  auto c_output = tensor::as_output_spec<typename Atom::TAcc>(
      std::forward<C>(c));
  using Memory = typename decltype(c_output)::MemoryElement;
  auto c_input = tensor::input<typename Atom::TAcc>(
      c_output.tensor(),
      tensor::zeros_transform<typename Atom::TAcc, Memory>);
  return Matmul<
      Atom, decltype(a_spec), decltype(b_spec),
      decltype(c_input), decltype(c_output)>{
          m, n, k, std::move(a_spec), std::move(b_spec),
          std::move(c_input), std::move(c_output)};
}

/** Build C = C-prologue + A*B^T with an explicit C input and output. */
template <gemm::Atom Atom,
          tensor::InputOperand A, tensor::InputOperand B,
          tensor::InputOperand CInput, tensor::OutputOperand COutput>
VECOPS_INLINE auto make_matmul_accumulate(
    nint_t m, nint_t n, nint_t k,
    A&& a, B&& b, CInput&& c_input, COutput&& c_output) {
  auto a_spec = tensor::as_input_spec<typename Atom::TA>(std::forward<A>(a));
  auto b_spec = tensor::as_input_spec<typename Atom::TB>(std::forward<B>(b));
  auto c_input_spec = tensor::as_input_spec<typename Atom::TAcc>(
      std::forward<CInput>(c_input));
  auto c_output_spec = tensor::as_output_spec<typename Atom::TAcc>(
      std::forward<COutput>(c_output));
  return Matmul<
      Atom, decltype(a_spec), decltype(b_spec),
      decltype(c_input_spec), decltype(c_output_spec)>{
          m, n, k, std::move(a_spec), std::move(b_spec),
          std::move(c_input_spec), std::move(c_output_spec)};
}

template <gemm::Atom Atom, execution::ExecutionScope Scope,
          tensor::InputOperand A, tensor::InputOperand B,
          tensor::OutputOperand C>
VECOPS_INLINE void matmul(
    Scope& scope, nint_t m, nint_t n, nint_t k,
    A&& a, B&& b, C&& c) {
  auto operation = make_matmul<Atom>(
      m, n, k, std::forward<A>(a), std::forward<B>(b),
      std::forward<C>(c));
  operation(scope);
}

template <gemm::Atom Atom,
          tensor::InputOperand A, tensor::InputOperand B,
          tensor::OutputOperand C>
VECOPS_INLINE void matmul(
    kernel::WorkspaceView& workspace,
    nint_t m, nint_t n, nint_t k,
    A&& a, B&& b, C&& c) {
  ExecutionSession execution{workspace};
  matmul<Atom>(execution, m, n, k,
               std::forward<A>(a), std::forward<B>(b),
               std::forward<C>(c));
}

} // namespace vecops::ops

#endif // VECOPS_OPS_MATMUL_H
