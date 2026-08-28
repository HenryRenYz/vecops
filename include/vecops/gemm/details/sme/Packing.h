//
// Copyright (c) vecops contributors.
//

#ifndef VECOPS_GEMM_DETAILS_SME_PACKING_H
#define VECOPS_GEMM_DETAILS_SME_PACKING_H

#include <concepts>

#include "vecops/gemm/KernelBase.h"
#include "vecops/gemm/details/packing/Utilities.h"
#include "vecops/vec/details/sme/State.h"

namespace vecops::gemm::details::sme {

struct Format {};

template <typename T>
inline constexpr bool supported_element_v =
    std::same_as<T, float32_t> || std::same_as<T, bfloat16_t> ||
    std::same_as<T, float16_t> || std::same_as<T, int8_t> ||
    std::same_as<T, uint8_t> || std::same_as<T, float64_t>;

VECOPS_INLINE auto panel_lanes() {
  return meta::cint<2> *
      vec::details::sme::streaming_lanes_value<float32_t>();
}

template <typename ElementT, Operand Side>
struct Packing {
  using Element = ElementT;
  using FormatType = Format;
  static_assert(supported_element_v<Element>);

  static constexpr int VectorAxis = 0;
  static constexpr nint_t KPack = sizeof(Element) > 4 ? 1 : 4 / sizeof(Element);
  static_assert(KPack > 0);

  VECOPS_INLINE static auto panel() {
    if constexpr (sizeof(Element) == 8)
      return meta::cint<2> *
          vec::details::sme::streaming_lanes_value<float64_t>();
    else
      return panel_lanes();
  }

  template <tensor::LayoutLike InputLayout>
  VECOPS_INLINE static auto packed_layout(const InputLayout& input) {
    static_assert(InputLayout::Ndim == 2,
                  "matrix packing accepts a rank-two input layout");
    using Spatial = tensor::size_type_t<0, InputLayout>;
    using K = tensor::size_type_t<1, InputLayout>;
    const Spatial spatial{tensor::size<0>(input)};
    const K k{tensor::size<1>(input)};
    const auto panel = Packing::panel();
    packing_details::validate_packed_size(
        static_cast<nint_t>(spatial), static_cast<nint_t>(k),
        static_cast<nint_t>(panel), KPack);
    return tensor::make_layout(tensor::make_shape(
        packing_details::ceil_div_value(spatial, panel),
        packing_details::ceil_div_value(k, meta::cint<KPack>),
        panel, meta::cint<KPack>));
  }

  template <tensor::LayoutLike Layout>
  static consteval bool is_packed_layout() {
    if constexpr (Layout::Ndim != 4) return false;
    else return
        std::same_as<tensor::size_type_t<2, Layout>,
                     decltype(Packing::panel())> &&
        std::same_as<tensor::size_type_t<3, Layout>, meta::Const<KPack>> &&
        std::same_as<tensor::stride_type_t<2, Layout>,
                     meta::Const<KPack>> &&
        std::same_as<tensor::stride_type_t<3, Layout>, meta::Const<1>>;
  }
};

} // namespace vecops::gemm::details::sme

#endif // VECOPS_GEMM_DETAILS_SME_PACKING_H
