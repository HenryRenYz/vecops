//
// Copyright (c) vecops contributors.
//

#ifndef VECOPS_GEMM_PACKING_H
#define VECOPS_GEMM_PACKING_H

#include "vecops/gemm/KernelBase.h"
#include "vecops/gemm/details/packing/Utilities.h"

namespace vecops::gemm {

template <Atom AtomT, Operand Side>
using packing_t = typename AtomT::template Packing<Side>;

template <Atom AtomT, Operand Side, tensor::LayoutLike InputLayout>
VECOPS_INLINE auto packed_layout(const InputLayout& input) {
  return packing_t<AtomT, Side>::packed_layout(input);
}

template <Atom AtomT, Operand Side, tensor::LayoutLike Layout>
consteval bool is_packed_layout() {
  return packing_t<AtomT, Side>::template is_packed_layout<Layout>();
}

template <Atom AtomT, Operand Side,
          tensor::LayoutLike InputLayout,
          tensor::LayoutLike OutputLayout>
VECOPS_INLINE bool is_corresponding_packed_layout(
    const InputLayout& input, const OutputLayout& output) {
  if constexpr (!is_packed_layout<AtomT, Side, OutputLayout>()) {
    return false;
  } else {
    const auto expected = packed_layout<AtomT, Side>(input);
    return packing_details::same_layout_values(expected, output);
  }
}

} // namespace vecops::gemm

#endif // VECOPS_GEMM_PACKING_H
