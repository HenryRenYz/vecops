//
// Copyright (c) vecops contributors.
//

#ifndef VECOPS_MATMUL_DETAILS_PACK_AMX_BACKEND_H
#define VECOPS_MATMUL_DETAILS_PACK_AMX_BACKEND_H

#include "vecops/execution/ExecutionSession.h"
#include "vecops/matmul/details/amx/Packing.h"
#include "vecops/matmul/details/pack/amx/Pack.h"
#include "vecops/matmul/details/pack/generic/Pack.h"
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
  static constexpr bool supports_column_compensation = true;

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
    const auto spatial = tensor::size_value<0>(layout);
    const auto k = tensor::size_value<1>(layout);
    using Spatial = std::remove_cvref_t<decltype(spatial)>;
    using K = std::remove_cvref_t<decltype(k)>;
    constexpr bool SpatialGuaranteed = Spatial::aligns(Packing::Panel);
    constexpr bool KGuaranteed = K::aligns(Packing::KTile);
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
      const nint_t row_stride = static_cast<nint_t>(
          tensor::stride_value<0>(layout));
      if constexpr (Side == gemm::Operand::A) {
        if constexpr (!SpatialGuaranteed && !KGuaranteed) {
          amx::pack_a_direct_dynamic(
              input, row_stride, output,
              static_cast<nint_t>(spatial), static_cast<nint_t>(k),
              Packing::Panel, Packing::KTile);
        } else {
          amx::pack_a_direct<SpatialGuaranteed, KGuaranteed>(
              input, row_stride, output,
              static_cast<nint_t>(spatial), static_cast<nint_t>(k));
        }
      } else {
        amx::pack_b_direct<
            Packing::KPack, SpatialGuaranteed, KGuaranteed>(
            input, row_stride, output,
            static_cast<nint_t>(spatial), static_cast<nint_t>(k));
      }
    } else if constexpr (Side == gemm::Operand::A) {
      if constexpr (VEC_WIDTH >= 512) {
        using ATag = vec::ScalableTag<T, 0>;
        amx::pack_a_access<Packing::KTile, ATag>(
            source, output, spatial, k);
      } else {
        generic::pack_blocked_rows<Tag>(
                                        source, output,
                                        static_cast<nint_t>(spatial),
                                        static_cast<nint_t>(k),
                                        Packing::Panel, Packing::KTile);
      }
    } else {
      if constexpr (VEC_WIDTH >= 512) {
        using Transform =
            typename generic::SpecOf<Source>::TransformType;
        constexpr bool RowwiseCompatibleTransform = [] {
          if constexpr (std::same_as<Transform, tensor::NoTransform>) {
            return true;
          } else {
            return Transform::is_elementwise &&
                Transform::permutation_equivariant;
          }
        }();
        constexpr bool RowMajorInput = std::same_as<
            tensor::stride_type_t<
                1, typename generic::SpecOf<Source>::InputLayout>,
            meta::Const<1>> && RowwiseCompatibleTransform;
        if constexpr (RowMajorInput) {
          amx::pack_b_row_major_access<
              Packing::KPack, Packing::KTile>(
                  source, output, spatial, k);
        } else {
          amx::pack_b_access<
              Packing::KPack, Packing::KTile, Tag>(
                  source, output, spatial, k);
        }
      } else {
        const nint_t padded_groups =
            ceil_div(k, Packing::KTile) *
            (Packing::KTile / Packing::KPack);
        generic::pack_interleaved_panels<Packing::KPack, Tag>(
            source, output,
            static_cast<nint_t>(spatial), static_cast<nint_t>(k),
            Packing::Panel, padded_groups);
      }
    }
  }

  template <gemm::Atom Atom,
            execution::ExecutionScope Scope,
            typename Source, typename Destination,
            typename CompensationDestination>
  VECOPS_ALWAYS_INLINE static void run_compensated(
      Scope& scope, const Source& source, Destination& destination,
      CompensationDestination& compensation, int32_t a_zero_point) {
    using Packing = gemm::packing_t<Atom, gemm::Operand::B>;
    static_assert(std::same_as<typename Atom::TA, uint8_t>);
    static_assert(std::same_as<typename Atom::TB, int8_t>);
    static_assert(std::same_as<typename Atom::TAcc, int32_t>);
    static_assert(std::same_as<typename Source::ComputeType, int8_t>);
    static_assert(std::same_as<typename Destination::ComputeType, int8_t>);
    static_assert(std::same_as<
                  typename CompensationDestination::ComputeType, int32_t>);
    const auto& layout = source.spec().input_layout();
    const auto spatial = tensor::size_value<0>(layout);
    const auto k = tensor::size_value<1>(layout);
    using Spatial = std::remove_cvref_t<decltype(spatial)>;
    using K = std::remove_cvref_t<decltype(k)>;
    constexpr bool SpatialGuaranteed = Spatial::aligns(Packing::Panel);
    constexpr bool KGuaranteed = K::aligns(Packing::KTile);
    auto* output = reinterpret_cast<int8_t*>(destination.raw_data());
    auto* correction =
        reinterpret_cast<int32_t*>(compensation.raw_data());
    constexpr bool Direct = VEC_WIDTH >= 512 &&
        generic::RawDirectAccess<Source> &&
        std::same_as<
            tensor::stride_type_t<
                1, typename generic::SpecOf<Source>::InputLayout>,
            meta::Const<1>>;
    if constexpr (Direct) {
      const auto* input =
          reinterpret_cast<const int8_t*>(source.raw_data());
      amx::pack_b_direct_compensated<
          Packing::KPack, SpatialGuaranteed, KGuaranteed>(
          input, static_cast<nint_t>(tensor::stride_value<0>(layout)),
          output, correction,
          static_cast<nint_t>(spatial), static_cast<nint_t>(k),
          a_zero_point);
    } else if constexpr (VEC_WIDTH >= 512) {
      using Transform = typename generic::SpecOf<Source>::TransformType;
      constexpr bool RowwiseCompatibleTransform = [] {
        if constexpr (std::same_as<Transform, tensor::NoTransform>) {
          return true;
        } else {
          return Transform::is_elementwise &&
              Transform::permutation_equivariant;
        }
      }();
      constexpr bool RowMajorInput = std::same_as<
          tensor::stride_type_t<
              1, typename generic::SpecOf<Source>::InputLayout>,
          meta::Const<1>> && RowwiseCompatibleTransform;
      if constexpr (RowMajorInput) {
        amx::pack_b_row_major_access_compensated<
            Packing::KPack, Packing::KTile>(
                source, output, correction, spatial, k, a_zero_point);
      } else {
        run<Atom, gemm::Operand::B>(scope, source, destination);
        amx::compensate_packed_s8_b(
            output, correction,
            static_cast<nint_t>(spatial), static_cast<nint_t>(k),
            a_zero_point);
      }
    } else {
      run<Atom, gemm::Operand::B>(scope, source, destination);
      amx::compensate_packed_s8_b(
          output, correction,
          static_cast<nint_t>(spatial), static_cast<nint_t>(k),
          a_zero_point);
    }
  }
};

} // namespace vecops::kernel::matmul_pack_details

#endif // VECOPS_MATMUL_DETAILS_PACK_AMX_BACKEND_H
