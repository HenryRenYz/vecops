//
// Copyright (c) vecops contributors.
//

#ifndef VECOPS_MATMUL_DETAILS_SME_RUNTIME_QUANT_INT8_H
#define VECOPS_MATMUL_DETAILS_SME_RUNTIME_QUANT_INT8_H

#include <type_traits>
#include <utility>

#include "vecops/matmul/Packing.h"
#include "vecops/matmul/details/kernel/sme/Atoms.h"
#include "vecops/matmul/Quantization.h"
#include "vecops/tensor/DataAccess.h"

namespace vecops::kernel::matmul_details::sme {

#if defined(__ARM_FEATURE_SVE_MATMUL_INT8)

namespace runtime_quant_int8_details {

template <typename Access>
using SpecOf = std::remove_cvref_t<decltype(
    std::declval<const std::remove_cvref_t<Access>&>().spec())>;

template <typename Access>
using InputLayoutOf = typename SpecOf<Access>::InputLayout;

template <::vecops::matmul::Atom Atom, ::vecops::matmul::Operand Side, typename Access>
inline constexpr bool is_packed_access_v =
    ::vecops::matmul::is_packed_layout<Atom, Side, InputLayoutOf<Access>>();

} // namespace runtime_quant_int8_details

void fused_runtime_quant_int8_packed_b_gemv(
    const float32_t* a, const int8_t* packed_b,
    nint_t b_outer_stride, nint_t b_group_stride,
    nint_t b_spatial_stride, nint_t packed_panel,
    const int32_t* correction, float32_t* output,
    const float32_t* column_scales,
    nint_t logical_n, nint_t logical_k,
    float32_t quant_multiplier, int32_t input_zero_point,
    float32_t row_dequant_scale);

template <::vecops::matmul::Atom Atom,
          typename A, typename B, typename CInput, typename COutput>
inline constexpr bool fused_runtime_quant_int8_candidate_v =
    std::same_as<Atom, ::vecops::matmul::SME_I8I32<uint8_t, int8_t>> &&
    A::Rank == 2 && B::Rank == 4 &&
    CInput::Rank == 2 && COutput::Rank == 2 &&
    std::same_as<typename A::MemoryElement, float32_t> &&
    std::same_as<typename A::ComputeType, uint8_t> &&
    is_runtime_per_row_asymmetric_quantize_transform_v<
        typename A::Transform> &&
    runtime_quant_int8_details::is_packed_access_v<
        Atom, ::vecops::matmul::Operand::B, B> &&
    std::same_as<typename B::ComputeType, int8_t> &&
    std::same_as<typename CInput::MemoryElement, int32_t> &&
    std::same_as<typename CInput::ComputeType, int32_t> &&
    std::same_as<typename CInput::Transform, tensor::NoTransform> &&
    std::same_as<typename COutput::MemoryElement, float32_t> &&
    std::same_as<typename COutput::ComputeType, int32_t> &&
    is_runtime_per_row_column_dequantize_transform_v<
        typename COutput::Transform>;

template <::vecops::matmul::Atom Atom,
          typename A, typename B, typename CInput, typename COutput>
VECOPS_NOINLINE VECOPS_FUNCTION_ALIGN(64) bool
try_fused_runtime_quant_int8_packed_b_gemv(
    nint_t logical_m, nint_t logical_n, nint_t logical_k,
    const A& a, const B& b,
    const CInput& c_input, COutput& c_output) {
  if constexpr (!fused_runtime_quant_int8_candidate_v<
                    Atom, A, B, CInput, COutput>) {
    return false;
  } else {
    const nint_t packed_panel = static_cast<nint_t>(
        ::vecops::matmul::packing_t<Atom, ::vecops::matmul::Operand::B>::panel());
    if (logical_m <= 0 || logical_n <= 0 || logical_k < 0 ||
        packed_panel <= 0) {
      return false;
    }

    const auto a_strides = a.raw_strides();
    const auto b_strides = b.raw_strides();
    const auto correction_strides = c_input.raw_strides();
    const auto output_strides = c_output.raw_strides();
    if (a_strides[1] != 1 || correction_strides[1] != 1 ||
        output_strides[1] != 1) {
      return false;
    }

    const auto& quant = a.spec().transform().parameters();
    const auto& dequant = c_output.spec().transform().parameters();
    for (nint_t row = 0; row < logical_m; ++row) {
      const auto local_origin = tensor::coord(row, nint_t{0});
      const auto a_original = a.spec().projection().project(local_origin);
      const auto output_original =
          c_output.spec().projection().project(local_origin);
      const nint_t quant_row = quant.row_scale_index(a_original);
      const nint_t dequant_row = dequant.row_scale_index(output_original);
      const nint_t column = output_original[output_original.size() - 1];

      fused_runtime_quant_int8_packed_b_gemv(
          reinterpret_cast<const float32_t*>(a.raw_data()) +
              row * a_strides[0],
          reinterpret_cast<const int8_t*>(b.raw_data()),
          b_strides[0], b_strides[1], b_strides[2], packed_panel,
          reinterpret_cast<const int32_t*>(c_input.raw_data()) +
              row * correction_strides[0],
          reinterpret_cast<float32_t*>(c_output.raw_data()) +
              row * output_strides[0],
          dequant.column_scales + column,
          logical_n, logical_k,
          quant.multipliers[quant_row], *quant.zero_point,
          dequant.row_scales[dequant_row]);
    }
    return true;
  }
}

#endif

} // namespace vecops::kernel::matmul_details::sme

#endif // VECOPS_MATMUL_DETAILS_SME_RUNTIME_QUANT_INT8_H
