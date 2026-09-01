//
// Copyright (c) vecops contributors.
//

#ifndef VECOPS_MATMUL_DETAILS_FAMILY_PLAN_H
#define VECOPS_MATMUL_DETAILS_FAMILY_PLAN_H

#include <utility>

#include "vecops/matmul/details/FamilySelector.h"
#include "vecops/matmul/details/OperationCommon.h"
#include "vecops/matmul/details/families/GenericTiled.h"
#include "vecops/matmul/details/families/WholeProblem.h"

namespace vecops::ops::matmul_details {

template <typename Family, typename Config>
struct FamilyPlan;

/** Shared adapter for whole-problem families and their selected backend leaf. */
template <typename Family, typename Config>
struct WholeProblemFamilyPlan {
  template <Extent M, Extent N, Extent K,
            tensor::InputOperand A, tensor::InputOperand B,
            tensor::OutputOperand C>
  VECOPS_INLINE static nint_t required_workspace(
      const Config& config, M&& m, N&& n, K&& k,
      A&& a, B&& b, C&& c) {
    auto plan = prepare_matmul(
        config, std::forward<M>(m), std::forward<N>(n),
        std::forward<K>(k), std::forward<A>(a), std::forward<B>(b),
        std::forward<C>(c));
    return plan.required_workspace();
  }

  template <Extent M, Extent N, Extent K,
            tensor::InputOperand A, tensor::InputOperand B,
            tensor::InputOperand CInput, tensor::OutputOperand COutput>
  VECOPS_INLINE static nint_t required_workspace(
      const Config& config, M&& m, N&& n, K&& k,
      A&& a, B&& b, CInput&& c_input, COutput&& c_output) {
    auto plan = prepare_matmul_accumulate(
        config, std::forward<M>(m), std::forward<N>(n),
        std::forward<K>(k), std::forward<A>(a), std::forward<B>(b),
        std::forward<CInput>(c_input), std::forward<COutput>(c_output));
    return plan.required_workspace();
  }

  template <execution::ExecutionScope Scope,
            Extent M, Extent N, Extent K,
            tensor::InputOperand A, tensor::InputOperand B,
            tensor::OutputOperand C>
  VECOPS_INLINE static void run(
      Scope& scope, const Config& config, M&& m, N&& n, K&& k,
      A&& a, B&& b, C&& c) {
    auto plan = prepare_matmul(
        config, std::forward<M>(m), std::forward<N>(n),
        std::forward<K>(k), std::forward<A>(a), std::forward<B>(b),
        std::forward<C>(c));
    plan(scope);
  }

  template <execution::ExecutionScope Scope,
            Extent M, Extent N, Extent K,
            tensor::InputOperand A, tensor::InputOperand B,
            tensor::InputOperand CInput, tensor::OutputOperand COutput>
  VECOPS_INLINE static void run(
      Scope& scope, const Config& config, M&& m, N&& n, K&& k,
      A&& a, B&& b, CInput&& c_input, COutput&& c_output) {
    auto plan = prepare_matmul_accumulate(
        config, std::forward<M>(m), std::forward<N>(n),
        std::forward<K>(k), std::forward<A>(a), std::forward<B>(b),
        std::forward<CInput>(c_input), std::forward<COutput>(c_output));
    plan(scope);
  }
};

template <typename Family, typename Config>
  requires ::vecops::matmul::kernel_family::WholeProblemFamily<Family>
struct FamilyPlan<Family, Config> : WholeProblemFamilyPlan<Family, Config> {};

/** Adapter for the explicitly selected generic cache-tiled family. */
template <typename Config>
struct FamilyPlan<::vecops::matmul::kernel_family::GenericTiled, Config> {
  using Atom = typename Config::Atom;
  using Implementation = SelectedImplementation<Atom>;
  struct ResourceToken {
    using ResourceRequirements =
        kernel::matmul_implementation::resource_requirements_t<Implementation>;
  };

  template <Extent M, Extent N, Extent K,
            tensor::InputOperand A, tensor::InputOperand B,
            tensor::OutputOperand C>
  VECOPS_INLINE static nint_t required_workspace(
      const Config& config, M&& m, N&& n, K&& k,
      A&& a, B&& b, C&& c) {
    auto a_spec = tensor::as_input_spec<typename Atom::TA>(
        std::forward<A>(a));
    auto b_spec = tensor::as_input_spec<typename Atom::TB>(
        std::forward<B>(b));
    auto c_spec = tensor::as_output_spec<typename Atom::TAcc>(
        std::forward<C>(c));
    return generic_tiler_workspace_bytes<Config, Implementation>(
        config, extent_value(std::forward<M>(m)),
        extent_value(std::forward<N>(n)), extent_value(std::forward<K>(k)),
        a_spec, b_spec, c_spec);
  }

  template <Extent M, Extent N, Extent K,
            tensor::InputOperand A, tensor::InputOperand B,
            tensor::InputOperand CInput, tensor::OutputOperand COutput>
  VECOPS_INLINE static nint_t required_workspace(
      const Config& config, M&& m, N&& n, K&& k,
      A&& a, B&& b, CInput&&, COutput&& c_output) {
    return required_workspace(
        config, std::forward<M>(m), std::forward<N>(n),
        std::forward<K>(k), std::forward<A>(a), std::forward<B>(b),
        std::forward<COutput>(c_output));
  }

  template <execution::ExecutionScope Scope,
            Extent M, Extent N, Extent K,
            tensor::InputOperand A, tensor::InputOperand B,
            tensor::OutputOperand C>
  VECOPS_INLINE static void run(
      Scope& scope, const Config& config, M&& m, N&& n, K&& k,
      A&& a, B&& b, C&& c) {
    auto m_value = extent_value(std::forward<M>(m));
    auto n_value = extent_value(std::forward<N>(n));
    auto k_value = extent_value(std::forward<K>(k));
    auto a_spec = tensor::as_input_spec<typename Atom::TA>(
        std::forward<A>(a));
    auto b_spec = tensor::as_input_spec<typename Atom::TB>(
        std::forward<B>(b));
    auto c_output = tensor::as_output_spec<typename Atom::TAcc>(
        std::forward<C>(c));
    using Memory = typename decltype(c_output)::MemoryElement;
    auto c_input = tensor::input<typename Atom::TAcc>(
        c_output.tensor(),
        tensor::zeros_transform<typename Atom::TAcc, Memory>);
    scope.with_resources(
        ResourceToken{}, [&](auto& active) VECOPS_INLINE_LAMBDA_NOEXCEPT {
          run_generic_tiler<Config, Implementation>(
              active, config, m_value, n_value, k_value,
              a_spec, b_spec, c_input, c_output);
        });
  }

  template <execution::ExecutionScope Scope,
            Extent M, Extent N, Extent K,
            tensor::InputOperand A, tensor::InputOperand B,
            tensor::InputOperand CInput, tensor::OutputOperand COutput>
  VECOPS_INLINE static void run(
      Scope& scope, const Config& config, M&& m, N&& n, K&& k,
      A&& a, B&& b, CInput&& c_input, COutput&& c_output) {
    auto m_value = extent_value(std::forward<M>(m));
    auto n_value = extent_value(std::forward<N>(n));
    auto k_value = extent_value(std::forward<K>(k));
    auto a_spec = tensor::as_input_spec<typename Atom::TA>(
        std::forward<A>(a));
    auto b_spec = tensor::as_input_spec<typename Atom::TB>(
        std::forward<B>(b));
    auto c_input_spec = tensor::as_input_spec<typename Atom::TAcc>(
        std::forward<CInput>(c_input));
    auto c_output_spec = tensor::as_output_spec<typename Atom::TAcc>(
        std::forward<COutput>(c_output));
    scope.with_resources(
        ResourceToken{}, [&](auto& active) VECOPS_INLINE_LAMBDA_NOEXCEPT {
          run_generic_tiler<Config, Implementation>(
              active, config, m_value, n_value, k_value,
              a_spec, b_spec, c_input_spec, c_output_spec);
        });
  }
};

template <typename Config>
using SelectedFamilyPlan = FamilyPlan<
    ::vecops::matmul::details::selected_family_t<Config>, Config>;

} // namespace vecops::ops::matmul_details

#endif // VECOPS_MATMUL_DETAILS_FAMILY_PLAN_H
