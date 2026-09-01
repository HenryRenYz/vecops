//
// Copyright (c) vecops contributors.
//

#ifndef VECOPS_MATMUL_DETAILS_SME_PACKING_H
#define VECOPS_MATMUL_DETAILS_SME_PACKING_H

#include <concepts>
#include <limits>

#include "vecops/matmul/Atom.h"
#include "vecops/tensor/Layout.h"
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

private:
  VECOPS_INLINE static constexpr void validate_extents(
      nint_t spatial, nint_t k, nint_t panel) {
    VECOPS_ASSERT(spatial >= 0 && k >= 0,
                  "matrix packing extents must be non-negative");
    VECOPS_ASSERT(panel > 0,
                  "matrix packing blocks must be positive");
    const nint_t spatial_groups = ceil_div(spatial, panel);
    const nint_t k_groups = ceil_div(k, KPack);
    constexpr nint_t Limit = std::numeric_limits<nint_t>::max();
    VECOPS_ASSERT(spatial_groups <= Limit / panel &&
                      k_groups <= Limit / KPack,
                  "packed matrix extent overflows nint_t");
    const nint_t padded_spatial = spatial_groups * panel;
    const nint_t padded_k = k_groups * KPack;
    VECOPS_ASSERT(padded_spatial == 0 ||
                      padded_k <= Limit / padded_spatial,
                  "packed matrix element count overflows nint_t");
  }

public:
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
    const Spatial spatial{tensor::size_value<0>(input)};
    const K k{tensor::size_value<1>(input)};
    const auto panel = Packing::panel();
    validate_extents(static_cast<nint_t>(spatial),
                     static_cast<nint_t>(k),
                     static_cast<nint_t>(panel));
    return tensor::make_layout(tensor::make_shape(
        ceil_div(spatial, panel),
        ceil_div(k, meta::cint<KPack>),
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

#endif // VECOPS_MATMUL_DETAILS_SME_PACKING_H
