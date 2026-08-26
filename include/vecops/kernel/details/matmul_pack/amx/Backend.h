//
// Copyright (c) vecops contributors.
//

#ifndef VECOPS_KERNEL_DETAILS_MATMUL_PACK_AMX_BACKEND_H
#define VECOPS_KERNEL_DETAILS_MATMUL_PACK_AMX_BACKEND_H

#include <limits>

#include "vecops/execution/ExecutionSession.h"
#include "vecops/gemm/details/amx/Packing.h"
#include "vecops/kernel/details/matmul_pack/amx/Pack.h"
#include "vecops/kernel/details/matmul_pack/generic/Pack.h"
#include "vecops/vec/Capabilities.h"

namespace vecops::kernel::matmul_pack_details {

template <typename T>
consteval int amx_panel_scale_power() {
#if VEC_WIDTH > 0
  nint_t native_bytes = VEC_WIDTH / 8;
#else
  nint_t native_bytes = 16;
#endif
  nint_t requested_bytes = 16 * static_cast<nint_t>(sizeof(T));
  int power = 0;
  while (requested_bytes > native_bytes) {
    native_bytes *= 2;
    ++power;
  }
  while (requested_bytes < native_bytes) {
    requested_bytes *= 2;
    --power;
  }
  return power;
}

template <>
struct Backend<
    gemm::details::amx::Format,
    matmul_pack_implementation::Vector> {
  using ResourceRequirements =
      typename execution::details::current_backend_t::DefaultRequirements;

  template <typename InputSpec, typename OutputSpec>
  static constexpr bool eligible = false;

  template <gemm::Atom Atom, gemm::Operand Side,
            execution::ExecutionScope Scope,
            typename Source, typename Destination>
  VECOPS_ALWAYS_INLINE static void run(
      Scope&, const Source& source, Destination& destination) {
    using Packing = gemm::packing_t<Atom, Side>;
    using T = typename Packing::Element;
    static_assert(std::same_as<typename Source::ComputeType, T>);
    const auto& layout = source.spec().input_layout();
    const nint_t spatial = layout.shape()[0];
    const nint_t k = layout.shape()[1];
    auto* output = reinterpret_cast<T*>(destination.raw_data());
    using Tag = vec::ScalableTag<T, amx_panel_scale_power<T>()>;
    constexpr bool Direct = VEC_WIDTH >= 512 &&
        generic::RawDirectAccess<Source> &&
        std::same_as<
            tensor::stride_type_t<
                1, typename generic::SpecOf<Source>::InputLayout>,
            meta::Const<1>>;
    if constexpr (Direct) {
      const auto* input = reinterpret_cast<const T*>(source.raw_data());
      const nint_t row_stride = source.raw_strides()[0];
      if constexpr (Side == gemm::Operand::A) {
        amx::pack_a_direct(
            input, row_stride, output, spatial, k,
            Packing::Panel, Packing::KTile);
      } else {
        constexpr nint_t MaxGatherStride =
            std::numeric_limits<int32_t>::max() / (Packing::Panel - 1);
        if (k >= Packing::KPack && row_stride >= 0 &&
            row_stride <= MaxGatherStride) {
          amx::pack_b_direct<Packing::KPack>(
              input, row_stride, output, spatial, k,
              Packing::Panel, Packing::KTile);
        } else {
          const nint_t padded_groups =
              ceil_div(k, Packing::KTile) *
              (Packing::KTile / Packing::KPack);
          generic::pack_interleaved_panels<Packing::KPack, Tag>(
              source, output, spatial, k,
              Packing::Panel, padded_groups);
        }
      }
    } else if constexpr (Side == gemm::Operand::A) {
      generic::pack_blocked_rows<Tag>(source, output, spatial, k,
                                      Packing::Panel, Packing::KTile);
    } else {
      const nint_t padded_groups =
          ceil_div(k, Packing::KTile) *
          (Packing::KTile / Packing::KPack);
      generic::pack_interleaved_panels<Packing::KPack, Tag>(
          source, output, spatial, k, Packing::Panel, padded_groups);
    }
  }
};

} // namespace vecops::kernel::matmul_pack_details

#endif // VECOPS_KERNEL_DETAILS_MATMUL_PACK_AMX_BACKEND_H
