//
// Copyright (c) vecops contributors.
//

#ifndef VECOPS_MATMUL_DETAILS_FAMILIES_GENERIC_TILED_H
#define VECOPS_MATMUL_DETAILS_FAMILIES_GENERIC_TILED_H

#include "vecops/Assertion.h"
#include "vecops/kernel/Loop.h"
#include "vecops/matmul/Config.h"
#include "vecops/matmul/Packing.h"
#include "vecops/matmul/details/planning/Implementation.h"
#include "vecops/matmul/details/tiled/Tiler.h"
#include "vecops/tensor/DataAccess.h"

namespace vecops::matmul::details {

template <typename Spec>
struct BroadcastPackedOperand {
  const Spec* spec;
};

template <::vecops::matmul::Atom Atom, ::vecops::matmul::Operand Side, typename Spec>
VECOPS_INLINE auto batch_loop_operand(const Spec& spec) {
  if constexpr (::vecops::matmul::is_packed_layout<
                    Atom, Side, typename Spec::InputLayout>()) {
    return BroadcastPackedOperand<Spec>{&spec};
  } else {
    return spec;
  }
}

template <typename Spec>
VECOPS_INLINE const Spec& batch_loop_leaf(
    const BroadcastPackedOperand<Spec>& operand) {
  return *operand.spec;
}

template <typename Leaf>
VECOPS_INLINE const Leaf& batch_loop_leaf(const Leaf& leaf) {
  return leaf;
}

template <::vecops::matmul::Atom Atom, ::vecops::matmul::Operand Side,
          int LogicalRank, typename Spec, typename CLayout>
VECOPS_INLINE void validate_input(
    const Spec& spec, const CLayout& c_layout,
    nint_t spatial, nint_t k) {
  using Layout = typename Spec::InputLayout;
  if constexpr (::vecops::matmul::is_packed_layout<Atom, Side, Layout>()) {
    static_assert(LogicalRank >= 2);
    using Packing = ::vecops::matmul::packing_t<Atom, Side>;
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
        static_cast<nint_t>(tensor::size_value<0>(
            spec.input_layout())) * panel >= spatial,
        "packed matmul spatial extent is too small");
    const nint_t padded_k = [&] {
      if constexpr (requires { Packing::KTile; }) {
        return static_cast<nint_t>(tensor::size_value<1>(
            spec.input_layout())) * Packing::KTile;
      } else {
        return static_cast<nint_t>(tensor::size_value<1>(
            spec.input_layout())) * Packing::KPack;
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
        static_cast<nint_t>(tensor::size_value<LogicalRank - 2>(
            spec.input_layout())) == spatial &&
        static_cast<nint_t>(tensor::size_value<LogicalRank - 1>(
            spec.input_layout())) == k,
        "unpacked matmul operand shape mismatch");
  }
}

template <typename Config, typename Implementation,
          meta::ValueType M, meta::ValueType N, meta::ValueType K,
          tensor::InputSpecLike ASpec, tensor::InputSpecLike BSpec,
          tensor::OutputSpecLike COutputSpec>
VECOPS_INLINE nint_t generic_tiler_workspace_bytes(
    const Config& config, M m, N n, K k,
    const ASpec& a, const BSpec& b, const COutputSpec& c_output) {
  static_assert(COutputSpec::OutputTensor::Ndim >= 2,
                "matmul output must have at least two dimensions");
  auto a_leaf = tensor::take_trailing<2>(a);
  auto b_leaf = tensor::take_trailing<2>(b);
  auto c_leaf = tensor::take_trailing<2>(c_output);
  return ::vecops::matmul::details::tiled_workspace_bytes<
      Config, Implementation>(
          config, m, n, k, a_leaf, b_leaf, c_leaf);
}

template <typename Config, typename Implementation,
          execution::ExecutionScope Scope,
          meta::ValueType M, meta::ValueType N, meta::ValueType K,
          tensor::InputSpecLike ASpec, tensor::InputSpecLike BSpec,
          tensor::InputSpecLike CInputSpec,
          tensor::OutputSpecLike COutputSpec>
VECOPS_INLINE void run_generic_tiler(
    Scope& scope, const Config& config, M m, N n, K k,
    const ASpec& a, const BSpec& b,
    const CInputSpec& c_input, const COutputSpec& c_output) {
  using Atom = typename Config::Atom;
  constexpr int Rank = COutputSpec::OutputTensor::Ndim;
  constexpr int PrefixRank = Rank - 2;
  static_assert(Rank >= 2, "matmul output must have at least two dimensions");
  validate_input<Atom, ::vecops::matmul::Operand::A, Rank>(
      a, c_output.output_layout(), static_cast<nint_t>(m),
      static_cast<nint_t>(k));
  validate_input<Atom, ::vecops::matmul::Operand::B, Rank>(
      b, c_output.output_layout(), static_cast<nint_t>(n),
      static_cast<nint_t>(k));
  for (int d = 0; d < Rank; ++d) {
    VECOPS_ASSERT(
        c_input.input_layout().shape()[d] ==
            c_output.output_layout().shape()[d],
        "matmul C input and output shapes must match");
  }
  VECOPS_ASSERT(
      static_cast<nint_t>(tensor::size_value<Rank - 2>(
          c_output.output_layout())) == static_cast<nint_t>(m) &&
          static_cast<nint_t>(tensor::size_value<Rank - 1>(
              c_output.output_layout())) == static_cast<nint_t>(n),
      "matmul output shape does not match M and N");

  if constexpr (PrefixRank == 0) {
    ::vecops::matmul::details::run_tiled_rank2<Config, Implementation>(
        scope, config, m, n, k, a, b, c_input, c_output);
  } else {
    for (int d = 0; d < PrefixRank; ++d) {
      if (c_output.output_layout().shape()[d] == 0) return;
    }
    const auto a_loop = batch_loop_operand<
        Atom, ::vecops::matmul::Operand::A>(a);
    const auto b_loop = batch_loop_operand<
        Atom, ::vecops::matmul::Operand::B>(b);
    kernel::loop::for_each_dims<PrefixRank>(
        [&](const auto& a_leaf, const auto& b_leaf,
            const auto& c_input_leaf, const auto& c_output_leaf)
            VECOPS_INLINE_LAMBDA_NOEXCEPT {
          ::vecops::matmul::details::run_tiled_rank2<
              Config, Implementation>(
                  scope, config, m, n, k,
                  batch_loop_leaf(a_leaf), batch_loop_leaf(b_leaf),
                  c_input_leaf, c_output_leaf);
        },
        a_loop, b_loop, c_input, c_output);
  }
}

} // namespace vecops::matmul::details

#endif // VECOPS_MATMUL_DETAILS_FAMILIES_GENERIC_TILED_H
