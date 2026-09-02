//
// Copyright (c) vecops contributors.
//

#ifndef VECOPS_MATMUL_DETAILS_FAMILY_PLAN_H
#define VECOPS_MATMUL_DETAILS_FAMILY_PLAN_H

#include <utility>

#include "vecops/matmul/details/planning/FamilySelector.h"
#include "vecops/matmul/details/planning/Implementation.h"
#include "vecops/matmul/details/planning/families/GenericTiled.h"
#include "vecops/matmul/details/planning/families/ArchitectureFamily.h"

namespace vecops::matmul::details {

template <typename Family, typename Config>
struct FamilyPlan;

/** Shared adapter for architecture families and their selected backend leaf. */
template <typename Family, typename Config>
struct ArchitectureFamilyPlan {
  template <meta::ValueInput M, meta::ValueInput N, meta::ValueInput K,
            tensor::InputOperand A, tensor::InputOperand B,
            tensor::InputOperand CInput, tensor::OutputOperand COutput>
  VECOPS_INLINE static nint_t required_workspace(
      const Config& config, M&& m, N&& n, K&& k,
      A&& a, B&& b, CInput&& c_input, COutput&& c_output) {
    auto plan = make_matmul_invocation(
        config, std::forward<M>(m), std::forward<N>(n),
        std::forward<K>(k), std::forward<A>(a), std::forward<B>(b),
        std::forward<CInput>(c_input), std::forward<COutput>(c_output));
    return plan.required_workspace();
  }

  template <execution::ExecutionScope Scope,
            meta::ValueInput M, meta::ValueInput N, meta::ValueInput K,
            tensor::InputOperand A, tensor::InputOperand B,
            tensor::InputOperand CInput, tensor::OutputOperand COutput>
  VECOPS_INLINE static void run(
      Scope& scope, const Config& config, M&& m, N&& n, K&& k,
      A&& a, B&& b, CInput&& c_input, COutput&& c_output) {
    auto plan = make_matmul_invocation(
        config, std::forward<M>(m), std::forward<N>(n),
        std::forward<K>(k), std::forward<A>(a), std::forward<B>(b),
        std::forward<CInput>(c_input), std::forward<COutput>(c_output));
    plan(scope);
  }
};

template <typename Family, typename Config>
  requires ::vecops::matmul::kernel_family::ArchitectureFamily<Family>
struct FamilyPlan<Family, Config> : ArchitectureFamilyPlan<Family, Config> {};

/** Adapter for the explicitly selected generic cache-tiled family. */
template <typename Config>
struct FamilyPlan<::vecops::matmul::kernel_family::GenericTiled, Config> {
  using Atom = typename Config::Atom;
  using Implementation = SelectedImplementation<Atom>;
  struct ResourceToken {
    using ResourceRequirements =
        kernel::matmul_implementation::resource_requirements_t<Implementation>;
  };

  template <meta::ValueInput M, meta::ValueInput N, meta::ValueInput K,
            tensor::InputOperand A, tensor::InputOperand B,
            tensor::InputOperand CInput, tensor::OutputOperand COutput>
  VECOPS_INLINE static nint_t required_workspace(
      const Config& config, M&& m, N&& n, K&& k,
      A&& a, B&& b, CInput&&, COutput&& c_output) {
    auto a_spec = tensor::as_input_spec<typename Atom::TA>(
        std::forward<A>(a));
    auto b_spec = tensor::as_input_spec<typename Atom::TB>(
        std::forward<B>(b));
    auto c_output_spec = tensor::as_output_spec<typename Atom::TAcc>(
        std::forward<COutput>(c_output));
    return generic_tiler_workspace_bytes<Config, Implementation>(
        config, meta::to_value(std::forward<M>(m)),
        meta::to_value(std::forward<N>(n)), meta::to_value(std::forward<K>(k)),
        a_spec, b_spec, c_output_spec);
  }

  template <execution::ExecutionScope Scope,
            meta::ValueInput M, meta::ValueInput N, meta::ValueInput K,
            tensor::InputOperand A, tensor::InputOperand B,
            tensor::InputOperand CInput, tensor::OutputOperand COutput>
  VECOPS_INLINE static void run(
      Scope& scope, const Config& config, M&& m, N&& n, K&& k,
      A&& a, B&& b, CInput&& c_input, COutput&& c_output) {
    auto m_value = meta::to_value(std::forward<M>(m));
    auto n_value = meta::to_value(std::forward<N>(n));
    auto k_value = meta::to_value(std::forward<K>(k));
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

} // namespace vecops::matmul::details

#endif // VECOPS_MATMUL_DETAILS_FAMILY_PLAN_H
