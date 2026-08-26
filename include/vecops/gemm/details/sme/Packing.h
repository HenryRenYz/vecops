//
// Copyright (c) vecops contributors.
//

#ifndef VECOPS_GEMM_DETAILS_SME_PACKING_H
#define VECOPS_GEMM_DETAILS_SME_PACKING_H

#include <arm_sme.h>

#include <concepts>

#include "vecops/gemm/KernelBase.h"
#include "vecops/gemm/details/packing/Utilities.h"

namespace vecops::gemm::details::sme {

struct Format {};

template <typename T>
inline constexpr bool supported_element_v =
    std::same_as<T, float32_t> || std::same_as<T, bfloat16_t> ||
    std::same_as<T, float16_t> || std::same_as<T, int8_t> ||
    std::same_as<T, uint8_t>;

VECOPS_INLINE nint_t panel_lanes() {
  return 2 * static_cast<nint_t>(svcntsw());
}

template <typename ElementT, Operand Side>
struct Packing {
  using Element = ElementT;
  using FormatType = Format;
  static_assert(supported_element_v<Element>);

  static constexpr int VectorAxis = 0;
  static constexpr nint_t KPack = 4 / sizeof(Element);
  static_assert(KPack > 0);

  template <tensor::LayoutLike InputLayout>
  VECOPS_INLINE static auto packed_layout(const InputLayout& input) {
    static_assert(InputLayout::Ndim == 2,
                  "matrix packing accepts a rank-two input layout");
    const nint_t spatial = tensor::size<0>(input);
    const nint_t k = tensor::size<1>(input);
    const nint_t panel = panel_lanes();
    packing_details::validate_packed_size(spatial, k, panel, KPack);
    return tensor::make_layout(tensor::make_shape(
        meta::Any{ceil_div(spatial, panel)},
        meta::Any{ceil_div(k, KPack)},
        meta::Any{panel}, meta::cint<KPack>));
  }

  template <tensor::LayoutLike Layout>
  static consteval bool is_packed_layout() {
    if constexpr (Layout::Ndim != 4) return false;
    else return
        std::same_as<tensor::size_type_t<3, Layout>, meta::Const<KPack>> &&
        std::same_as<tensor::stride_type_t<2, Layout>,
                     meta::Const<KPack>> &&
        std::same_as<tensor::stride_type_t<3, Layout>, meta::Const<1>>;
  }
};

} // namespace vecops::gemm::details::sme

#endif // VECOPS_GEMM_DETAILS_SME_PACKING_H
