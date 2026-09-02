//
// Copyright (c) vecops contributors.
//

#ifndef VECOPS_OPS_MATMUL_H
#define VECOPS_OPS_MATMUL_H

/**
 * @file Matmul.h
 * @brief Public entry point for reusable Config-only Matmul operators.
 */

#include <string_view>
#include <utility>

#include "vecops/execution/ExecutionSession.h"
#include "vecops/matmul/Config.h"
#include "vecops/matmul/details/planning/FamilyPlan.h"
#include "vecops/platform/CacheInfo.h"

namespace vecops::ops {

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

  static constexpr std::string_view kernel_family_name() {
    return ::vecops::matmul::kernel_family::Info<KernelFamily>::name;
  }

  const Config config;

  VECOPS_INLINE constexpr explicit Matmul(Config cfg)
      : config(std::move(cfg)) {}

  template <meta::ValueInput M, meta::ValueInput N, meta::ValueInput K,
            tensor::InputOperand A, tensor::InputOperand B,
            tensor::OutputOperand C>
  VECOPS_INLINE nint_t required_workspace(
      M&& m, N&& n, K&& k, A&& a, B&& b, C&& c) const {
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
    auto c_input = tensor::input<typename Atom::TAcc>(
        c_output.tensor(),
        tensor::zeros_transform<typename Atom::TAcc, Memory>);
    Plan::run(
        scope, config, std::forward<M>(m), std::forward<N>(n),
        std::forward<K>(k), std::forward<A>(a), std::forward<B>(b),
        std::move(c_input), std::move(c_output));
  }

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

template <typename Config>
VECOPS_INLINE constexpr auto matmul(Config config) {
  return Matmul<Config>{std::move(config)};
}

} // namespace vecops::ops

#endif // VECOPS_OPS_MATMUL_H
