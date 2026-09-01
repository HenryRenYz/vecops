//
// Copyright (c) vecops contributors.
//

#ifndef VECOPS_MATMUL_PACKING_H
#define VECOPS_MATMUL_PACKING_H

#include "vecops/matmul/Atom.h"
#include "vecops/tensor/Layout.h"

#if defined(ARCH_X86_FAMILY)
#include "vecops/matmul/details/amx/Packing.h"
#endif

#if defined(HAS_SME)
#include "vecops/matmul/details/sme/Packing.h"
#endif

namespace vecops::matmul {

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
    static_assert(decltype(expected)::Ndim == OutputLayout::Ndim);
    for (int d = 0; d < OutputLayout::Ndim; ++d) {
      if (expected.shape()[d] != output.shape()[d] ||
          expected.strides()[d] != output.strides()[d]) {
        return false;
      }
    }
    return true;
  }
}

} // namespace vecops::matmul

#endif // VECOPS_MATMUL_PACKING_H
