//
// Copyright (c) vecops contributors.
//

#ifndef VECOPS_GEMM_DETAILS_AMX_PACKING_H
#define VECOPS_GEMM_DETAILS_AMX_PACKING_H

#include <concepts>
#include <type_traits>

#include "vecops/gemm/KernelBase.h"
#include "vecops/gemm/details/packing/Utilities.h"

namespace vecops::gemm::details::amx {

struct Format {};

template <typename T>
inline constexpr bool supported_element_v =
    std::same_as<T, bfloat16_t> || std::same_as<T, float16_t> ||
    std::same_as<T, int8_t> || std::same_as<T, uint8_t>;

template <typename ElementT, Operand Side>
struct Packing {
  using Element = ElementT;
  using FormatType = Format;
  static_assert(supported_element_v<Element>);

  static constexpr int VectorAxis = Side == Operand::A ? 1 : 0;
  static constexpr nint_t Panel = 16;
  static constexpr nint_t KTile = 64 / sizeof(Element);
  static constexpr nint_t KPack = 4 / sizeof(Element);
  static_assert(KTile > 0 && KPack > 0 && KTile % KPack == 0);

  template <tensor::LayoutLike InputLayout>
  VECOPS_INLINE static constexpr auto packed_layout(
      const InputLayout& input) {
    static_assert(InputLayout::Ndim == 2,
                  "matrix packing accepts a rank-two input layout");
    const auto spatial = tensor::size<0>(input);
    const auto k = tensor::size<1>(input);
    packing_details::validate_packed_size(spatial, k, Panel, KTile);
    using Spatial = tensor::size_type_t<0, InputLayout>;
    using K = tensor::size_type_t<1, InputLayout>;
    const Spatial spatial_value{spatial};
    const K k_value{k};
    if constexpr (Side == Operand::A) {
      return tensor::make_layout(tensor::make_shape(
          packing_details::ceil_div_value(
              spatial_value, meta::cint<Panel>),
          packing_details::ceil_div_value(k_value, meta::cint<KTile>),
          meta::cint<Panel>, meta::cint<KTile>));
    } else {
      return tensor::make_layout(tensor::make_shape(
          packing_details::ceil_div_value(
              spatial_value, meta::cint<Panel>),
          packing_details::ceil_div_value(k_value, meta::cint<KTile>),
          meta::cint<KTile / KPack>,
          meta::cint<Panel>, meta::cint<KPack>));
    }
  }

  template <tensor::LayoutLike Layout>
  static consteval bool is_packed_layout() {
    if constexpr (Side == Operand::A) {
      if constexpr (Layout::Ndim != 4) return false;
      else return
          std::same_as<tensor::size_type_t<2, Layout>, meta::Const<Panel>> &&
          std::same_as<tensor::size_type_t<3, Layout>, meta::Const<KTile>> &&
          std::same_as<tensor::stride_type_t<1, Layout>,
                       meta::Const<Panel * KTile>> &&
          std::same_as<tensor::stride_type_t<2, Layout>,
                       meta::Const<KTile>> &&
          std::same_as<tensor::stride_type_t<3, Layout>, meta::Const<1>>;
    } else {
      if constexpr (Layout::Ndim != 5) return false;
      else return
          std::same_as<tensor::size_type_t<2, Layout>,
                       meta::Const<KTile / KPack>> &&
          std::same_as<tensor::size_type_t<3, Layout>, meta::Const<Panel>> &&
          std::same_as<tensor::size_type_t<4, Layout>, meta::Const<KPack>> &&
          std::same_as<tensor::stride_type_t<1, Layout>,
                       meta::Const<Panel * KTile>> &&
          std::same_as<tensor::stride_type_t<2, Layout>,
                       meta::Const<Panel * KPack>> &&
          std::same_as<tensor::stride_type_t<3, Layout>,
                       meta::Const<KPack>> &&
          std::same_as<tensor::stride_type_t<4, Layout>, meta::Const<1>>;
    }
  }
};

} // namespace vecops::gemm::details::amx

#endif // VECOPS_GEMM_DETAILS_AMX_PACKING_H
