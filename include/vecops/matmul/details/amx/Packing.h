//
// Copyright (c) vecops contributors.
//

#ifndef VECOPS_MATMUL_DETAILS_AMX_PACKING_H
#define VECOPS_MATMUL_DETAILS_AMX_PACKING_H

#include <concepts>
#include <limits>
#include <type_traits>

#include "vecops/matmul/Atom.h"
#include "vecops/tensor/Layout.h"

namespace vecops::matmul::details::amx {

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

private:
  VECOPS_INLINE static constexpr void validate_extents(
      nint_t spatial, nint_t k) {
    VECOPS_ASSERT(spatial >= 0 && k >= 0,
                  "matrix packing extents must be non-negative");
    const nint_t spatial_groups = ceil_div(spatial, Panel);
    const nint_t k_groups = ceil_div(k, KTile);
    constexpr nint_t Limit = std::numeric_limits<nint_t>::max();
    VECOPS_ASSERT(spatial_groups <= Limit / Panel &&
                      k_groups <= Limit / KTile,
                  "packed matrix extent overflows nint_t");
    const nint_t padded_spatial = spatial_groups * Panel;
    const nint_t padded_k = k_groups * KTile;
    VECOPS_ASSERT(padded_spatial == 0 ||
                      padded_k <= Limit / padded_spatial,
                  "packed matrix element count overflows nint_t");
  }

public:
  template <tensor::LayoutLike InputLayout>
  VECOPS_INLINE static constexpr auto packed_layout(
      const InputLayout& input) {
    static_assert(InputLayout::Ndim == 2,
                  "matrix packing accepts a rank-two input layout");
    const auto spatial = tensor::size_value<0>(input);
    const auto k = tensor::size_value<1>(input);
    validate_extents(static_cast<nint_t>(spatial),
                     static_cast<nint_t>(k));
    if constexpr (Side == Operand::A) {
      return tensor::make_layout(tensor::make_shape(
          ceil_div(spatial, meta::cint<Panel>),
          ceil_div(k, meta::cint<KTile>),
          meta::cint<Panel>, meta::cint<KTile>));
    } else {
      return tensor::make_layout(tensor::make_shape(
          ceil_div(spatial, meta::cint<Panel>),
          ceil_div(k, meta::cint<KTile>),
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

} // namespace vecops::matmul::details::amx

#endif // VECOPS_MATMUL_DETAILS_AMX_PACKING_H
