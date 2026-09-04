//
// Copyright (c) vecops contributors.
//

#ifndef VECOPS_MATMUL_DETAILS_ACCUMULATOR_H
#define VECOPS_MATMUL_DETAILS_ACCUMULATOR_H

#include <cstdint>
#include <type_traits>

#include "vecops/matmul/Config.h"
#include "vecops/tensor/DataAccess.h"

/**
 * @file vecops/matmul/details/tiled/Accumulator.h
 * @brief Split-K accumulator placement: dedicated workspace vs reusing the
 *        output tensor's own storage.
 *
 * When the K loop splits (logical K exceeds KC), the partial sums of an
 * output block must survive across K blocks. They can live in one of two
 * places:
 *
 * - workspace: a dedicated `TAcc` tensor allocated for the operation; always
 *   applicable, costs allocation size;
 * - the output tensor itself: valid when the output can faithfully *store*
 *   `TAcc` values -- every output element is at least as wide as `TAcc`
 *   (`sizeof(C) >= sizeof(TAcc)`), no output transform reinterprets values
 *   on the way in or out, and the output's alignment and byte strides are
 *   divisible by `sizeof(TAcc)` (checked below). The accumulator is then
 *   just a re-typed view of the output storage and needs zero extra memory.
 */

namespace vecops::matmul::details {

/**
 * @brief Reinterpret an output spec's storage as a rank-2 `Acc` accumulator
 *        tensor over the full logical M x N extent.
 *
 * The output layout's strides are in `MemoryElement` units; the returned
 * tensor's shape/strides are in `Acc` element units, so the conversion goes
 * through bytes: `row_bytes = strides[0] * sizeof(MemoryElement)` and the
 * accumulator stride is `row_bytes / sizeof(Acc)`. Always-on checks guard the
 * two requirements the reinterpretation silently relies on: the base address
 * is `Acc`-aligned, and both byte strides divide evenly by `sizeof(Acc)`
 * (otherwise some `Acc` elements would straddle two output elements).
 */
template <typename Acc, tensor::OutputSpecLike Spec>
VECOPS_INLINE auto output_acc_storage(
    const Spec& output, nint_t m, nint_t n) {
  using Memory = typename Spec::MemoryElement;
  const auto address = reinterpret_cast<std::uintptr_t>(
      output.tensor().data());
  VECOPS_CHECK(address % alignof(Acc) == 0,
               "output accumulator storage is insufficiently aligned");
  // Strides are in MemoryElement units here and in Acc units below; the
  // byte round trip is what makes the divisibility requirement explicit.
  const nint_t row_bytes = output.output_layout().strides()[0] *
      static_cast<nint_t>(sizeof(Memory));
  const nint_t column_bytes = output.output_layout().strides()[1] *
      static_cast<nint_t>(sizeof(Memory));
  VECOPS_CHECK(row_bytes % static_cast<nint_t>(sizeof(Acc)) == 0 &&
               column_bytes % static_cast<nint_t>(sizeof(Acc)) == 0,
               "output strides cannot represent accumulator storage");
  auto layout = tensor::make_layout(
      tensor::make_shape(meta::Any{m}, meta::Any{n}),
      tensor::make_strides(
          meta::Any{row_bytes / static_cast<nint_t>(sizeof(Acc))},
          meta::Any{column_bytes / static_cast<nint_t>(sizeof(Acc))}));
  return tensor::make_tensor(
      reinterpret_cast<Acc*>(output.tensor().data()), layout);
}

/// Automatic output-reuse is only taken for a perfect stand-in: identical
/// memory element and no output transform, so accumulation into the output
/// and reading it back are exact value round trips.
template <typename Config, tensor::OutputSpecLike Spec>
inline constexpr bool can_reuse_output_automatically_v =
    std::same_as<typename Spec::MemoryElement,
                 typename Config::Atom::TAcc> &&
    std::same_as<typename Spec::TransformType, tensor::NoTransform>;

/**
 * @brief Whether the split-K accumulator lives in the output tensor's own
 *        storage for this configuration and output spec.
 *
 * Dispatches `AccBufferMode`: `output` forces reuse (statically requiring
 * `sizeof(C) >= sizeof(TAcc)`; alignment/stride feasibility is checked at
 * run time in `output_acc_storage`), `automatic` reuses only the perfect
 * stand-in case above, and `workspace` never reuses.
 */
template <typename Config, tensor::OutputSpecLike Spec>
inline constexpr bool uses_output_accumulator_v = [] {
  constexpr auto Mode = Config::GenericTuning::acc_buffer_mode;
  if constexpr (Mode == AccBufferMode::output) {
    static_assert(sizeof(typename Spec::MemoryElement) >=
                  sizeof(typename Config::Atom::TAcc),
                  "forced output accumulator reuse needs sizeof(C) >= sizeof(TAcc)");
    return true;
  } else if constexpr (Mode == AccBufferMode::automatic) {
    return can_reuse_output_automatically_v<Config, Spec>;
  } else {
    return false;
  }
}();

} // namespace vecops::matmul::details

#endif // VECOPS_MATMUL_DETAILS_ACCUMULATOR_H
