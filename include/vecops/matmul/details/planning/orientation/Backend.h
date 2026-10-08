// SPDX-FileCopyrightText: 2026 HenryRenYz and vecops contributors
// SPDX-License-Identifier: MIT
//
//

#ifndef VECOPS_MATMUL_DETAILS_ORIENTATION_BACKEND_H
#define VECOPS_MATMUL_DETAILS_ORIENTATION_BACKEND_H

/**
 * @file vecops/matmul/details/planning/orientation/Backend.h
 * @brief Compile-time A/B orientation (problem-transposition) planning.
 *
 * Every matmul problem has a free algebraic identity: C = A*B^T equals
 * C^T = B*A^T. Evaluating the transposed form exchanges the two
 * `[spatial, K]` operands and transposes the C specs, which can turn an
 * awkward layout (e.g. a column-contiguous C that the backend must store
 * through strided accesses) into the backend's fast path.
 *
 * This layer decides **at compile time** whether to apply that exchange:
 * `swap_ab_v` guards on the user's `MatmulConfig::enable_swap_ab` opt-out,
 * the atom's `SwappedAtom` alias, and fully-automatic family/scheduler
 * selection, then asks the KernelKind-specific `Backend` specialization
 * (amx/ or sme/Backend.h) for its layout-and-shape verdict. The decision
 * happens before any packing or family planning, so downstream packing
 * costs are computed for the selected orientation with no runtime branch.
 */

#include <type_traits>

#include "vecops/matmul/Config.h"
#include "vecops/matmul/Packing.h"
#include "vecops/tensor/DataAccess.h"

namespace vecops::matmul::details::orientation {

/// Primary template: the generic fallback never swaps. Each architecture
/// KernelKind specializes this with its own layout/shape rules (see
/// amx/Backend.h and sme/Backend.h).
template <typename KernelKind>
struct Backend {
  /// Whether to evaluate the problem as C^T = B*A^T for this leaf's
  /// operand specs and Meta extents.
  template <typename, typename, typename, typename, typename,
            typename, typename, typename>
  static consteval bool swap_ab() {
    return false;
  }
};

/// An A/B operand spec that survives a role exchange: rank two with a
/// lane-local transform. Swapping only exchanges the roles -- each operand
/// keeps its own [spatial, K] coordinates.
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

/// Last axis (the K axis of [spatial, K]) has compile-time unit stride.
template <typename Layout>
inline constexpr bool row_contiguous_v = [] {
  using Stride = tensor::stride_type_t<Layout::Ndim - 1, Layout>;
  return meta::is_singleton_v<Stride> &&
      meta::singleton_value_v<Stride> == 1;
}();

/// Rank-two layout whose leading (spatial) axis has unit stride, i.e. a
/// transposed operand.
template <typename Layout>
inline constexpr bool column_contiguous_v = [] {
  if constexpr (Layout::Ndim != 2) {
    return false;
  } else {
    using Stride = tensor::stride_type_t<0, Layout>;
    return meta::is_singleton_v<Stride> &&
        meta::singleton_value_v<Stride> == 1;
  }
}();

/// LHS provably >= RHS: compares LHS's lower bound against RHS's upper
/// bound, so an unknown bound answers false (conservative).
template <meta::ValueType LHS, meta::ValueType RHS>
inline constexpr bool provably_at_least_v = [] {
  using L = std::remove_cvref_t<LHS>;
  using R = std::remove_cvref_t<RHS>;
  if constexpr (meta::has_lower_bound_v<L> && meta::has_upper_bound_v<R>)
    return meta::lower_bound_v<L> >= meta::upper_bound_v<R>;
  else
    return false;
}();

/// Value provably >= the constant Minimum (unknown bounds answer false).
template <meta::ValueType Value, nint_t Minimum>
inline constexpr bool provably_at_least_value_v =
    meta::has_lower_bound_v<std::remove_cvref_t<Value>> &&
    meta::lower_bound_v<std::remove_cvref_t<Value>> >= Minimum;

/// LHS provably >= ceil(RHS / RHSScale): the scaled comparison used by the
/// "not disproportionately smaller" shape rules (bound-based, so the
/// ceiling is computed from RHS's upper bound).
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

/// Whether the Config opted into orientation planning (configs predating
/// the flag are treated as opted out).
template <typename Config>
inline constexpr bool config_enables_swap_ab_v = [] {
  if constexpr (requires { Config::enable_swap_ab; })
    return Config::enable_swap_ab;
  else
    return false;
}();

/// The combined swap verdict for one leaf. Besides the user opt-out, the
/// atom must provide a SwappedAtom (mixed-signedness atoms transpose
/// their type slots), and the family/scheduler selections must be
/// Automatic: explicit selections keep their requested orientation, so
/// their semantics cannot be silently rotated.
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
