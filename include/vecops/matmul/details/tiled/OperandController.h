//
// Copyright (c) vecops contributors.
//

#ifndef VECOPS_MATMUL_DETAILS_OPERAND_CONTROLLER_H
#define VECOPS_MATMUL_DETAILS_OPERAND_CONTROLLER_H

#include <type_traits>
#include <utility>

#include "vecops/matmul/Packing.h"
#include "vecops/matmul/details/packing/Plan.h"
#include "vecops/matmul/details/tiled/PolicyTraits.h"
#include "vecops/tensor/DataAccess.h"

/**
 * @file vecops/matmul/details/tiled/OperandController.h
 * @brief Per-operand packing decisions for the generic tiled matmul: packed
 *        block narrowing, whether to pack at all, and how long one packed
 *        copy lives (per KC block vs the whole operand).
 *
 * A and B are controlled independently -- one side packing never forces the
 * other (no Cartesian branch) -- and each side is either consumed as a plain
 * narrowed view of the user's tensor or repacked into workspace through the
 * matmul packing Plan layer. The `PackingExtent` resolved in PolicyTraits.h
 * decides whether the packed copy is created inside the K loop (per panel) or
 * hoisted around the whole loop nest (whole operand).
 */

namespace vecops::matmul::details {

/// Whether this input spec already carries the atom's packed layout for one
/// side (never re-packed, under any PackingMode).
template <typename AtomT, Operand Side, typename Spec>
inline constexpr bool is_packed_spec_v = is_packed_layout<
    AtomT, Side, typename Spec::InputLayout>();

/**
 * @brief Narrow one operand to the current spatial x K block.
 *
 * Packed inputs cannot be sliced row by row: the packed layout is
 * `[panel, k-group, row(panel), k(KPack)]` and a block boundary only ever
 * lands on whole panels and whole K groups. The narrowing therefore advances
 * only the two outermost dimensions -- `(spatial_origin / panel)` steps in
 * dim 0 and `(k_origin / KStep)` steps in dim 1 -- and re-bases the data
 * pointer, keeping dims 2..3 (row within panel, k within group) restarting
 * at zero for the new block. Unpacked inputs take two ordinary narrow
 * views.
 */
template <typename AtomT, Operand Side, tensor::InputSpecLike Spec,
          meta::ValueType Spatial, meta::ValueType Reduction>
VECOPS_INLINE auto narrow_input(
    const Spec& spec, nint_t spatial_origin, Spatial spatial,
    nint_t k_origin, Reduction reduction) {
  if constexpr (is_packed_spec_v<AtomT, Side, Spec>) {
    using Packing = packing_t<AtomT, Side>;
    // The two packing formats expose their constants differently (AMX has
    // static Panel/KTile members, SME a panel() function and KPack only);
    // probe for the richer spelling and fall back to the common one.
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
    // Re-base only: the first two packed dims advance (panel index via
    // strides[0], k-group index via strides[1]); the row/k dims restart at 0.
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

/**
 * @brief Whether one operand should be repacked into workspace.
 *
 * Already-packed inputs are never re-packed; `always`/`never` are honored
 * literally; `automatic` packs only when the operand's innermost (K) stride
 * is not a compile-time 1 -- i.e. the input is not known K-contiguous and
 * packing is what makes it consumable by the packed-input kernel paths.
 */
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
    // Automatic: pack exactly when the K axis is not unit-stride at compile
    // time. A K-contiguous operand already matches the friendly layout.
    return !std::same_as<
        tensor::stride_type_t<Rank - 1, typename Spec::InputLayout>,
        meta::Const<1>>;
  }
}();

/**
 * @brief Run `fn` with one operand made panel-friendly for the current
 *        cache block.
 *
 * If packing is not warranted, `fn` receives the spec unchanged. Otherwise
 * the operand is packed into execution workspace (mark/rewind scoped: the
 * packed copy lives exactly for `fn`'s duration) and `fn` receives the
 * packed input spec.
 */
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
    run_matmul_pack<AtomT, Side>(
        scope, spec, packed);
    auto packed_spec = tensor::input<Element>(packed);
    // Rewind inside both return paths so the packed copy is released even
    // when fn returns a value.
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

/**
 * @brief Hoist packing of a `full_k` operand out of the whole loop nest --
 *        when allowed.
 *
 * Elevation requires all three: the resolved extent is `full_k`, the operand
 * is packable at all, and the caller passed `allow_full_k` (the Tiler grants
 * it when an explicitly-requested full_k policy overrides the budget, or the
 * operand's whole-K packed size fits the L3 working-set budget). Without it,
 * the plain spec flows through and packing happens per panel inside the
 * nest instead.
 */
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

/**
 * @brief One-side packing adapter; A and B never form a Cartesian branch.
 *
 * Exposes the two packing lifetimes as one type: `with_whole_operand` is the
 * hoisted full-K entry (called once around the whole loop nest, honoring the
 * L3 budget flag), `with_panel` is the per-KC-block entry (called inside the
 * nest for cache_k packing).
 */
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
