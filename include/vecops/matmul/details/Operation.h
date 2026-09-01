//
// Copyright (c) vecops contributors.
//

#ifndef VECOPS_MATMUL_DETAILS_OPERATION_H
#define VECOPS_MATMUL_DETAILS_OPERATION_H

#include <utility>

#include "vecops/execution/ExecutionSession.h"
#include "vecops/matmul/Config.h"
#include "vecops/matmul/details/FamilyPlan.h"
#include "vecops/matmul/details/OperationCommon.h"

namespace vecops::ops {

/**
 * Reusable semantic matrix-multiply operator.
 *
 * The object stores configuration only.  Family selection is resolved without
 * allowing family-local tuning parameters to change the selected family.
 */
template <typename Config>
class Matmul {
public:
  using Atom = typename Config::Atom;
  using Implementation = matmul_details::SelectedImplementation<Atom>;
  using ResourceRequirements =
      kernel::matmul_implementation::resource_requirements_t<Implementation>;
  using KernelFamily =
      ::vecops::matmul::details::selected_family_t<Config>;
  using Plan = matmul_details::SelectedFamilyPlan<Config>;

  static constexpr std::string_view kernel_family_name() {
    return ::vecops::matmul::kernel_family::Info<KernelFamily>::name;
  }

  const Config config;

  VECOPS_INLINE constexpr explicit Matmul(Config cfg)
      : config(std::move(cfg)) {}

  template <matmul_details::Extent M, matmul_details::Extent N,
            matmul_details::Extent K,
            tensor::InputOperand A, tensor::InputOperand B,
            tensor::OutputOperand C>
  VECOPS_INLINE nint_t required_workspace(
      M&& m, N&& n, K&& k, A&& a, B&& b, C&& c) const {
    return Plan::required_workspace(
        config, std::forward<M>(m), std::forward<N>(n),
        std::forward<K>(k), std::forward<A>(a), std::forward<B>(b),
        std::forward<C>(c));
  }

  template <matmul_details::Extent M, matmul_details::Extent N,
            matmul_details::Extent K,
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

  template <execution::ExecutionScope Scope,
            matmul_details::Extent M, matmul_details::Extent N,
            matmul_details::Extent K,
            tensor::InputOperand A, tensor::InputOperand B,
            tensor::OutputOperand C>
  VECOPS_INLINE void operator()(
      Scope& scope, M&& m, N&& n, K&& k,
      A&& a, B&& b, C&& c) const {
    Plan::run(
        scope, config, std::forward<M>(m), std::forward<N>(n),
        std::forward<K>(k), std::forward<A>(a), std::forward<B>(b),
        std::forward<C>(c));
  }

  template <execution::ExecutionScope Scope,
            matmul_details::Extent M, matmul_details::Extent N,
            matmul_details::Extent K,
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

  template <matmul_details::Extent M, matmul_details::Extent N,
            matmul_details::Extent K,
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

  template <matmul_details::Extent M, matmul_details::Extent N,
            matmul_details::Extent K,
            tensor::InputOperand A, tensor::InputOperand B,
            tensor::InputOperand CInput, tensor::OutputOperand COutput>
  VECOPS_INLINE void operator()(
      kernel::WorkspaceView& workspace, M&& m, N&& n, K&& k,
      A&& a, B&& b, CInput&& c_input, COutput&& c_output) const {
    ExecutionSession execution{workspace};
    (*this)(execution, std::forward<M>(m), std::forward<N>(n),
            std::forward<K>(k), std::forward<A>(a), std::forward<B>(b),
            std::forward<CInput>(c_input), std::forward<COutput>(c_output));
  }
};

template <typename Config>
VECOPS_INLINE constexpr auto matmul(Config config) {
  return Matmul<Config>{std::move(config)};
}

} // namespace vecops::ops

#endif // VECOPS_MATMUL_DETAILS_OPERATION_H
