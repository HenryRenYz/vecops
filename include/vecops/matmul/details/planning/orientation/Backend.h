//
// Copyright (c) vecops contributors.
//

#ifndef VECOPS_MATMUL_DETAILS_ORIENTATION_BACKEND_H
#define VECOPS_MATMUL_DETAILS_ORIENTATION_BACKEND_H

#include <type_traits>

#include "vecops/matmul/Config.h"
#include "vecops/matmul/Packing.h"
#include "vecops/tensor/DataAccess.h"

namespace vecops::matmul::details::orientation {

template <typename KernelKind>
struct Backend {
  template <typename, typename, typename, typename, typename,
            typename, typename, typename>
  static consteval bool swap_ab() {
    return false;
  }
};

template <typename Spec>
inline constexpr bool swappable_rank2_input_v =
    Spec::InputTensor::Ndim == 2 && Spec::TransformType::is_lane_local;

// A/B specs only exchange roles and retain their [spatial,K] coordinates.
// C specs exchange axes; keep them coordinate-independent until C DataAccess
// can prove coordinate-aware vector traversal under a transposed projection.
template <typename Spec>
inline constexpr bool swappable_rank2_c_input_v =
    Spec::InputTensor::Ndim == 2 && Spec::TransformType::is_elementwise;

template <typename Spec>
inline constexpr bool swappable_rank2_c_output_v =
    Spec::OutputTensor::Ndim == 2 && Spec::TransformType::is_elementwise;

template <typename Layout>
inline constexpr bool row_contiguous_v = std::same_as<
    tensor::stride_type_t<Layout::Ndim - 1, Layout>, meta::Const<1>>;

template <typename Layout>
inline constexpr bool column_contiguous_v = Layout::Ndim == 2 && std::same_as<
    tensor::stride_type_t<0, Layout>, meta::Const<1>>;

template <meta::ValueType LHS, meta::ValueType RHS>
inline constexpr bool provably_at_least_v = [] {
  using L = std::remove_cvref_t<LHS>;
  using R = std::remove_cvref_t<RHS>;
  if constexpr (meta::has_lower_bound_v<L> && meta::has_upper_bound_v<R>)
    return meta::lower_bound_v<L> >= meta::upper_bound_v<R>;
  else
    return false;
}();

template <meta::ValueType Value, nint_t Minimum>
inline constexpr bool provably_at_least_value_v =
    meta::has_lower_bound_v<std::remove_cvref_t<Value>> &&
    meta::lower_bound_v<std::remove_cvref_t<Value>> >= Minimum;

template <meta::ValueType LHS, meta::ValueType RHS, nint_t RHSScale>
  requires (RHSScale > 0)
inline constexpr bool provably_scaled_at_least_v = [] {
  using L = std::remove_cvref_t<LHS>;
  using R = std::remove_cvref_t<RHS>;
  if constexpr (meta::has_lower_bound_v<L> && meta::has_upper_bound_v<R>) {
    constexpr nint_t Upper = meta::upper_bound_v<R>;
    constexpr nint_t ScaledUpper =
        Upper / RHSScale + static_cast<nint_t>(Upper % RHSScale != 0);
    return meta::lower_bound_v<L> >= ScaledUpper;
  } else {
    return false;
  }
}();

template <typename Config>
inline constexpr bool config_enables_swap_ab_v = [] {
  if constexpr (requires { Config::enable_swap_ab; })
    return Config::enable_swap_ab;
  else
    return false;
}();

template <typename Config, meta::ValueType M, meta::ValueType N,
          meta::ValueType K, typename ASpec, typename BSpec,
          typename CInputSpec, typename COutputSpec>
inline constexpr bool swap_ab_v = [] {
  using Atom = typename Config::Atom;
  if constexpr (!config_enables_swap_ab_v<Config> ||
                !requires { typename Atom::SwappedAtom; } ||
                !std::same_as<typename Config::FamilySelection,
                              family_selection::Automatic> ||
                !std::same_as<typename Config::SchedulerPolicy,
                              kernel::matmul_policy::Automatic>) {
    return false;
  } else {
    return Backend<typename Atom::KernelKind>::template swap_ab<
        Atom, M, N, K, ASpec, BSpec, CInputSpec, COutputSpec>();
  }
}();

} // namespace vecops::matmul::details::orientation

#if defined(ARCH_X86_FAMILY)
#include "vecops/matmul/details/planning/orientation/amx/Backend.h"
#endif

#if defined(HAS_SME)
#include "vecops/matmul/details/planning/orientation/sme/Backend.h"
#endif

#endif // VECOPS_MATMUL_DETAILS_ORIENTATION_BACKEND_H
