//
// Copyright (c) vecops contributors.
//

#ifndef VECOPS_MATMUL_DETAILS_ACCUMULATOR_H
#define VECOPS_MATMUL_DETAILS_ACCUMULATOR_H

#include <cstdint>
#include <type_traits>

#include "vecops/matmul/Config.h"
#include "vecops/tensor/DataAccess.h"

namespace vecops::matmul::details {

template <typename Acc, tensor::OutputSpecLike Spec>
VECOPS_INLINE auto output_acc_storage(
    const Spec& output, nint_t m, nint_t n) {
  using Memory = typename Spec::MemoryElement;
  const auto address = reinterpret_cast<std::uintptr_t>(
      output.tensor().data());
  VECOPS_ASSERT(address % alignof(Acc) == 0,
                "output accumulator storage is insufficiently aligned");
  const nint_t row_bytes = output.output_layout().strides()[0] *
      static_cast<nint_t>(sizeof(Memory));
  const nint_t column_bytes = output.output_layout().strides()[1] *
      static_cast<nint_t>(sizeof(Memory));
  VECOPS_ASSERT(row_bytes % static_cast<nint_t>(sizeof(Acc)) == 0 &&
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

template <typename Config, tensor::OutputSpecLike Spec>
inline constexpr bool can_reuse_output_automatically_v =
    std::same_as<typename Spec::MemoryElement,
                 typename Config::Atom::TAcc> &&
    std::same_as<typename Spec::TransformType, tensor::NoTransform>;

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
