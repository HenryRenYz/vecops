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

/**
 * @file vecops/matmul/details/kernel/sme/RuntimeQuantInt8.h
 * @brief Fused runtime-quantization INT8 GEMV leaf for the SME backend.
 *
 * Fusion chain (all stages inside one leaf, no intermediate tensors):
 *
 * 1. A (fp32 memory, one row since M == 1) is quantized with the runtime
 *    per-row asymmetric transform `q = a * multiplier[row] + zero_point`
 *    (uint8 output domain);
 * 2. the quantized row drives an INT8/UINT8 dot-product GEMV against a
 *    packed INT8 B;
 * 3. a precomputed per-column `correction` (carried by the C-input tensor,
 *    int32, untransformed) repairs the zero-point bias term
 *    `zero_point * sum(b[, k])` introduced by step 1;
 * 4. the int32 result is dequantized on the way out with the runtime
 *    per-row/per-column transform `c *= row_scale[row] * column_scale[col]`.
 *
 * The heavy lifting lives in the .cpp (`fused_runtime_quant_int8_
 * packed_b_gemv_1x4vl`, declared here); this header only carries the
 * candidate predicate and the run-time shape/stride gate `try_fused_...`,
 * which selects the family when compile-time facts already prove the leaf
 * applies (see `select_automatic_dispatch_owner` in kernel/sme/Backend.h).
 */

namespace vecops::kernel::matmul_details::sme {

#if defined(HAS_SME_FA64) && defined(__ARM_FEATURE_SVE_MATMUL_INT8)

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
 * @brief One fused quantize -> GEMV -> correct -> dequantize pass over a
 *        1 x N x K problem against a packed INT8 B.
 *
 * Parameters map onto the packed B 4-D layout
 * `[panel][k-group][row(panel)][k(KPack)]` (see packing/sme/Format.h):
 * `b_outer_stride` steps panels (dim 0), `b_group_stride` steps K groups
 * (dim 1), `b_spatial_stride` steps rows within the panel's group (dim 2),
 * and `packed_panel` is the panel height (bounds dims 2..3). `correction`
 * is the int32 zero-point-bias row; `column_scales` the dequantize column
 * factors. Defined out of line in the SME backend TU.
 */
void fused_runtime_quant_int8_packed_b_gemv_1x4vl(
    const float32_t* a, const int8_t* packed_b,
    nint_t b_outer_stride, nint_t b_group_stride,
    nint_t b_spatial_stride, nint_t packed_panel,
    const int32_t* correction, float32_t* output,
    const float32_t* column_scales,
    nint_t logical_n, nint_t logical_k,
    float32_t quant_multiplier, int32_t input_zero_point,
    float32_t row_dequant_scale);

/**
 * @brief Compile-time candidate predicate for the fused runtime-quant leaf.
 *
 * Pins the exact fusion shape: SME_I8I32<uint8, int8> atom, rank-2 A whose
 * memory is fp32 behind a runtime per-row asymmetric quantize transform,
 * rank-4 packed B (int8), an int32 untransformed C input (the correction),
 * and an fp32-memory output behind a runtime per-row/column dequantize
 * transform. Anything else is rejected at compile time.
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
    std::same_as<typename COutput::MemoryElement, float32_t> &&
    std::same_as<typename COutput::ComputeType, int32_t> &&
    is_runtime_per_row_column_dequantize_transform_v<
        typename COutput::Transform>;

/**
 * @brief Run-time shape/stride gate for the fused runtime-quant leaf.
 *
 * Returns false (and lets the caller fall back) when the leaf's hand-written
 * kernel cannot serve this problem; the checks encode the geometry the
 * kernel body is unrolled for.
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
    using ATag = vec::ScalableTag<uint8_t, 0>;
    using AccTag = vec::ScalableTag<int32_t, 0>;
    constexpr nint_t KChunk = 64;
    const nint_t output_block = 4 * vec::size(AccTag{});
    const nint_t ordinary_vl_bytes = vec::size(ATag{});
    const nint_t packed_panel = static_cast<nint_t>(
        ::vecops::matmul::packing_t<Atom, ::vecops::matmul::Operand::B>::panel());
    // Shape/vector-length coupling the unrolled kernel assumes: a single
    // output row (GEMV); N a multiple of one 4-vector output block; K a
    // multiple of the 64-byte K chunk; and the ordinary SVE vector length
    // tied to both the chunk and the packed panel -- the packed panel is
    // SVL/2 rows (panel_lanes), so `ordinary_vl_bytes == 2 * packed_panel`
    // additionally pins ordinary VL == streaming SVL.
    if (logical_m != 1 || logical_n <= 0 || logical_k < 0 ||
        logical_n % output_block != 0 || logical_k % KChunk != 0 ||
        ordinary_vl_bytes != KChunk ||
        ordinary_vl_bytes != 2 * packed_panel) {
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

    // The quantize/dequantize scale tables are indexed in the *original*
    // (pre-narrow) coordinate space. This leaf receives already-narrowed
    // access objects, so project the local origin (0,0) back through each
    // spec's projection to recover the row/column this block actually
    // occupies, and read the scale indices from those original coordinates.
    const auto local_origin = tensor::coord(nint_t{0}, nint_t{0});
    const auto a_original = a.spec().projection().project(local_origin);
    const auto output_original =
        c_output.spec().projection().project(local_origin);
    const auto& quant = a.spec().transform().parameters();
    const auto& dequant = c_output.spec().transform().parameters();
    const nint_t quant_row = quant.row_scale_index(a_original);
    const nint_t dequant_row = dequant.row_scale_index(output_original);
    const nint_t column = output_original[output_original.size() - 1];

    fused_runtime_quant_int8_packed_b_gemv_1x4vl(
        reinterpret_cast<const float32_t*>(a.raw_data()),
        reinterpret_cast<const int8_t*>(b.raw_data()),
        b_strides[0], b_strides[1], b_strides[2], packed_panel,
        reinterpret_cast<const int32_t*>(c_input.raw_data()),
        reinterpret_cast<float32_t*>(c_output.raw_data()),
        dequant.column_scales + column,
        logical_n, logical_k,
        quant.multipliers[quant_row], *quant.zero_point,
        dequant.row_scales[dequant_row]);
    return true;
  }
}

#endif

} // namespace vecops::kernel::matmul_details::sme

#endif // VECOPS_MATMUL_DETAILS_SME_RUNTIME_QUANT_INT8_H
