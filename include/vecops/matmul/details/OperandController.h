//
// Copyright (c) vecops contributors.
//

#ifndef VECOPS_MATMUL_DETAILS_OPERAND_CONTROLLER_H
#define VECOPS_MATMUL_DETAILS_OPERAND_CONTROLLER_H

#include <type_traits>
#include <utility>

#include "vecops/matmul/Packing.h"
#include "vecops/matmul/details/PackOperation.h"
#include "vecops/matmul/details/PolicyTraits.h"
#include "vecops/tensor/DataAccess.h"

namespace vecops::matmul::details {

template <typename AtomT, Operand Side, typename Spec>
inline constexpr bool is_packed_spec_v = is_packed_layout<
    AtomT, Side, typename Spec::InputLayout>();

template <typename AtomT, Operand Side, tensor::InputSpecLike Spec,
          meta::ValueType Spatial, meta::ValueType Reduction>
VECOPS_INLINE auto narrow_input(
    const Spec& spec, nint_t spatial_origin, Spatial spatial,
    nint_t k_origin, Reduction reduction) {
  if constexpr (is_packed_spec_v<AtomT, Side, Spec>) {
    using Packing = packing_t<AtomT, Side>;
    const nint_t panel = [&] {
      if constexpr (requires { Packing::Panel; }) return Packing::Panel;
      else return static_cast<nint_t>(Packing::panel());
    }();
    constexpr nint_t KStep = [] {
      if constexpr (requires { Packing::KTile; }) return Packing::KTile;
      else return Packing::KPack;
    }();
    VECOPS_ASSERT(spatial_origin % panel == 0,
                  "packed matmul spatial block is misaligned");
    VECOPS_ASSERT(k_origin % KStep == 0,
                  "packed matmul K block is misaligned");
    static_assert(std::same_as<
        typename Spec::TransformType, tensor::NoTransform>,
        "packed matmul panels must be direct and untransformed");
    const auto& layout = spec.input_layout();
    const nint_t offset = (spatial_origin / panel) * layout.strides()[0] +
        (k_origin / KStep) * layout.strides()[1];
    auto panel_view = tensor::make_tensor(
        spec.tensor().data() + offset, layout);
    return tensor::input<typename Packing::Element>(panel_view);
  } else {
    auto spatial_view = tensor::narrow_view<0>(
        spec, spatial_origin, spatial);
    return tensor::narrow_view<1>(
        spatial_view, k_origin, reduction);
  }
}

template <typename Policy, typename AtomT, Operand Side, typename Spec>
inline constexpr bool should_pack_v = [] {
  if constexpr (is_packed_spec_v<AtomT, Side, Spec>) {
    return false;
  } else if constexpr (Policy::mode == PackingMode::always) {
    return true;
  } else if constexpr (Policy::mode == PackingMode::never) {
    return false;
  } else {
    constexpr int Rank = Spec::InputTensor::Ndim;
    return !std::same_as<
        tensor::stride_type_t<Rank - 1, typename Spec::InputLayout>,
        meta::Const<1>>;
  }
}();

template <typename Policy, typename AtomT, Operand Side,
          execution::ExecutionScope Scope,
          tensor::InputSpecLike Spec, typename Fn>
VECOPS_ALWAYS_INLINE decltype(auto) with_operand_panel(
    Scope& scope, const Spec& spec, Fn&& fn) {
  if constexpr (!should_pack_v<Policy, AtomT, Side, Spec>) {
    return std::forward<Fn>(fn)(spec);
  } else {
    auto& workspace = scope.workspace_view();
    const auto mark = workspace.mark();
    const auto layout = packed_layout<AtomT, Side>(spec.input_layout());
    using Element = typename packing_t<AtomT, Side>::Element;
    const nint_t bytes = tensor::numel(layout) *
        static_cast<nint_t>(sizeof(Element));
    auto* data = static_cast<Element*>(workspace.allocate(bytes, 64));
    auto packed = tensor::make_tensor(data, layout);
    ops::matmul_pack_details::run_matmul_pack<AtomT, Side>(
        scope, spec, packed);
    auto packed_spec = tensor::input<Element>(packed);
    if constexpr (std::is_void_v<decltype(std::forward<Fn>(fn)(packed_spec))>) {
      std::forward<Fn>(fn)(packed_spec);
      workspace.rewind(mark);
    } else {
      auto result = std::forward<Fn>(fn)(packed_spec);
      workspace.rewind(mark);
      return result;
    }
  }
}

template <typename Policy, typename Order, typename AtomT, Operand Side,
          execution::ExecutionScope Scope,
          tensor::InputSpecLike Spec, typename Fn>
VECOPS_ALWAYS_INLINE decltype(auto) with_full_k_operand(
    Scope& scope, const Spec& spec, bool allow_full_k, Fn&& fn) {
  if constexpr (full_k_packing_v<Policy, Side, Order> &&
                should_pack_v<Policy, AtomT, Side, Spec>) {
    if (allow_full_k) {
      return with_operand_panel<Policy, AtomT, Side>(
          scope, spec, std::forward<Fn>(fn));
    }
    return std::forward<Fn>(fn)(spec);
  } else {
    return std::forward<Fn>(fn)(spec);
  }
}

/** One-side packing adapter; A and B never form a Cartesian branch. */
template <Operand Side, typename Policy, typename Order, typename AtomT>
struct OperandController {
  static constexpr PackingExtent extent =
      resolved_packing_extent_v<Policy, Side, Order>;

  template <execution::ExecutionScope Scope,
            tensor::InputSpecLike Spec, typename Fn>
  VECOPS_ALWAYS_INLINE static decltype(auto) with_whole_operand(
      Scope& scope, const Spec& spec, bool allow_full_k, Fn&& fn) {
    return with_full_k_operand<Policy, Order, AtomT, Side>(
        scope, spec, allow_full_k, std::forward<Fn>(fn));
  }

  template <execution::ExecutionScope Scope,
            tensor::InputSpecLike Spec, typename Fn>
  VECOPS_ALWAYS_INLINE static decltype(auto) with_panel(
      Scope& scope, const Spec& spec, Fn&& fn) {
    return with_operand_panel<Policy, AtomT, Side>(
        scope, spec, std::forward<Fn>(fn));
  }
};

} // namespace vecops::matmul::details

#endif // VECOPS_MATMUL_DETAILS_OPERAND_CONTROLLER_H
