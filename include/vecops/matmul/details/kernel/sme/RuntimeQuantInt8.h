// SPDX-FileCopyrightText: 2026 HenryRenYz and vecops contributors
// SPDX-License-Identifier: MIT
//
//

#ifndef VECOPS_MATMUL_DETAILS_SME_RUNTIME_QUANT_INT8_H
#define VECOPS_MATMUL_DETAILS_SME_RUNTIME_QUANT_INT8_H

#include <type_traits>
#include <utility>

#include "vecops/matmul/Packing.h"
#include "vecops/matmul/details/kernel/sme/Atoms.h"
#include "vecops/matmul/Quantization.h"
#include "vecops/tensor/DataAccess.h"

/**
 * @file vecops/matmul/details/kernel/sme/RuntimeQuantInt8.h
 * @brief Fused runtime-quantization INT8 GEMV leaf for the SME backend.
 *
 * Fusion chain (all stages inside one leaf, no intermediate tensors):
 *
 * 1. each A row (fp32 memory) is quantized with the runtime per-row
 *    asymmetric transform `q = a * multiplier[row] + zero_point`
 *    (uint8 output domain);
 * 2. the quantized row drives an INT8/UINT8 widening-dot GEMV against a
 *    packed INT8 B;
 * 3. a precomputed per-column `correction` (carried by the C-input tensor,
 *    int32, untransformed) repairs the zero-point bias term
 *    `zero_point * sum(b[, k])` introduced by step 1;
 * 4. the final int32 result is dequantized on the way out with the runtime
 *    per-row/per-column transform `c *= row_scale[row] * column_scale[col]`;
 *    split-K first/middle phases instead keep the corrected int32 sum.
 *
 * The vector core lives in the .cpp (`fused_runtime_quant_int8_packed_b_
 * gemv`, declared here) and handles arbitrary N/K tails with masked loads;
 * this header adds the row driver, the compile-time candidate predicate,
 * and the run-time shape/stride gate `try_fused_...`, which selects the
 * family when compile-time facts already prove the leaf applies (see
 * `select_automatic_dispatch_owner` in kernel/sme/Backend.h). Only the SVE
 * INT8 dot feature is required -- the kernel needs neither ZA nor FA64.
 */

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

/**
 * @brief One fused quantize -> GEMV -> correct -> dequantize pass over one
 *        output row of an N x K problem against a packed INT8 B.
 *
 * Called once per output row by `try_fused_runtime_quant_int8_packed_b_gemv`,
 * which supplies the per-row quant/dequant scale table entries. Parameters
 * map onto the packed B 4-D layout `[panel][k-group][row(panel)][k(KPack)]`
 * (see packing/sme/Format.h): `b_outer_stride` steps panels (dim 0),
 * `b_group_stride` steps K groups (dim 1), `b_spatial_stride` steps rows
 * within the panel's group (dim 2), and `packed_panel` is the panel height
 * (bounds dims 2..3). `correction` is the int32 zero-point-bias row;
 * `column_scales` the dequantize column factors. Defined out of line in the
 * SME backend TU.
 */
void fused_runtime_quant_int8_packed_b_gemv(
    const float32_t* a, const int8_t* packed_b,
    nint_t b_outer_stride, nint_t b_group_stride,
    nint_t b_spatial_stride, nint_t packed_panel,
    const int32_t* correction, float32_t* output,
    const float32_t* column_scales,
    nint_t logical_n, nint_t logical_k,
    float32_t quant_multiplier, int32_t input_zero_point,
    float32_t row_dequant_scale);

/// Split-K phase twin: retain the corrected running sum as native int32
/// instead of applying the final dequantization transform.
void runtime_quant_int8_packed_b_gemv_accumulate(
    const float32_t* a, const int8_t* packed_b,
    nint_t b_outer_stride, nint_t b_group_stride,
    nint_t b_spatial_stride, nint_t packed_panel,
    const int32_t* prior, int32_t* output,
    nint_t logical_n, nint_t logical_k,
    float32_t quant_multiplier, int32_t input_zero_point);

/**
 * @brief Compile-time candidate predicate for the fused runtime-quant leaf.
 *
 * Pins the exact fusion shape: SME_I8I32<uint8, int8> atom, rank-2 A whose
 * memory is fp32 behind a runtime per-row asymmetric quantize transform,
 * rank-4 packed B (int8), an int32 untransformed C input (the correction),
 * and either an fp32-memory output behind a runtime per-row/column
 * dequantize transform or a native int32 split-K accumulator output.
 * Anything else is rejected at compile time.
 */
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
    std::same_as<typename COutput::ComputeType, int32_t> &&
    ((std::same_as<typename COutput::MemoryElement, float32_t> &&
      is_runtime_per_row_column_dequantize_transform_v<
          typename COutput::Transform>) ||
     (std::same_as<typename COutput::MemoryElement, int32_t> &&
      std::same_as<typename COutput::Transform, tensor::NoTransform>));

/**
 * @brief Run-time shape/stride gate and row driver for the fused
 *        runtime-quant leaf.
 *
 * Returns false (and lets the caller fall back) when the leaf cannot serve
 * this problem. Since the vector core masks its tails, the remaining
 * run-time checks are only positivity and unit inner-dimension strides on
 * the row-addressed operands. On success every output row is computed by
 * one `fused_runtime_quant_int8_packed_b_gemv` call.
 */
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
    constexpr bool FinalDequantize =
        is_runtime_per_row_column_dequantize_transform_v<
            typename COutput::Transform>;
    // The quantize/dequantize scale tables are indexed in the *original*
    // (pre-narrow) coordinate space. This leaf receives already-narrowed
    // access objects, so project each row's local origin (row, 0) back
    // through the spec's projection to recover the row/column this block
    // actually occupies, and read the scale indices from those original
    // coordinates.
    for (nint_t row = 0; row < logical_m; ++row) {
      const auto local_origin = tensor::coord(row, nint_t{0});
      const auto a_original = a.spec().projection().project(local_origin);
      const nint_t quant_row = quant.row_scale_index(a_original);
      const auto* a_row = reinterpret_cast<const float32_t*>(a.raw_data()) +
          row * a_strides[0];
      const auto* prior = reinterpret_cast<const int32_t*>(
          c_input.raw_data()) + row * correction_strides[0];
      if constexpr (FinalDequantize) {
        const auto& dequant = c_output.spec().transform().parameters();
        const auto output_original =
            c_output.spec().projection().project(local_origin);
        const nint_t dequant_row = dequant.row_scale_index(output_original);
        const nint_t column = output_original[output_original.size() - 1];
        fused_runtime_quant_int8_packed_b_gemv(
            a_row, reinterpret_cast<const int8_t*>(b.raw_data()),
            b_strides[0], b_strides[1], b_strides[2], packed_panel,
            prior,
            reinterpret_cast<float32_t*>(c_output.raw_data()) +
                row * output_strides[0],
            dequant.column_scales + column,
            logical_n, logical_k,
            quant.multipliers[quant_row], *quant.zero_point,
            dequant.row_scales[dequant_row]);
      } else {
        runtime_quant_int8_packed_b_gemv_accumulate(
            a_row, reinterpret_cast<const int8_t*>(b.raw_data()),
            b_strides[0], b_strides[1], b_strides[2], packed_panel,
            prior,
            reinterpret_cast<int32_t*>(c_output.raw_data()) +
                row * output_strides[0],
            logical_n, logical_k,
            quant.multipliers[quant_row], *quant.zero_point);
      }
    }
    return true;
  }
}

#endif

} // namespace vecops::kernel::matmul_details::sme

#endif // VECOPS_MATMUL_DETAILS_SME_RUNTIME_QUANT_INT8_H
