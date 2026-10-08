// SPDX-FileCopyrightText: 2026 HenryRenYz and vecops contributors
// SPDX-License-Identifier: MIT
//
//
// Runtime-quantized INT8 packed-B GEMV leaf. It accepts arbitrary positive M
// through a shared row driver and arbitrary N/K tails in this vector core.
// intentionally excluded from generic libvecops.a and compiled in the shared
// ISA-specific SME matmul archive.

#include <utility>

#include "vecops/CoreTypes.h"
#include "vecops/util/Math.h"
#include "vecops/util/ScalarConvert.h"
#include "vecops/vec/Vec.h"

namespace vecops::kernel::matmul_details::sme {

#if defined(__ARM_FEATURE_SVE_MATMUL_INT8)

template <bool Dequantize>
VECOPS_ALWAYS_INLINE void runtime_quant_int8_packed_b_gemv_core(
    const float32_t* a, const int8_t* packed_b,
    nint_t b_outer_stride, nint_t b_group_stride,
    nint_t b_spatial_stride, nint_t packed_panel,
    const int32_t* correction, void* output,
    const float32_t* column_scales,
    nint_t logical_n, nint_t logical_k,
    float32_t quant_multiplier, int32_t input_zero_point,
    float32_t row_dequant_scale) {
  using BTag = vec::ScalableTag<int8_t, 0>;
  using ATag = vec::ViewAs<uint8_t, BTag>;
  using GroupTag = vec::ViewAs<uint32_t, BTag>;
  using AccTag = vec::ViewAs<int32_t, BTag>;
  using OutputTag = vec::ViewAs<float32_t, BTag>;
  constexpr nint_t KPack = 4;
  const nint_t lanes = vec::size(AccTag{});

  const auto packed_at = [&](nint_t spatial) VECOPS_INLINE_LAMBDA {
    const nint_t offset =
        (spatial / packed_panel) * b_outer_stride +
        (spatial % packed_panel) * b_spatial_stride;
    return packed_b + offset;
  };

  VECOPS_LOOP_ALIGN(64) for (
      nint_t output_origin = 0;
      output_origin < logical_n;) {
    const nint_t panel_remaining =
        packed_panel - output_origin % packed_panel;
    const nint_t active = vecops::min(
        lanes, vecops::min(logical_n - output_origin, panel_remaining));
    const auto* packed = packed_at(output_origin);
    auto sum = vec::load(
        AccTag{}, correction + output_origin,
        vec::opt::first(active), vec::opt::zero);

    const nint_t groups = ceil_div(logical_k, KPack);
    VECOPS_LOOP_ALIGN(64) for (nint_t group = 0; group < groups; ++group) {
      uint32_t group_bits = 0;
      for (nint_t lane = 0; lane < KPack; ++lane) {
        const nint_t kk = group * KPack + lane;
        const uint8_t quantized = kk < logical_k
            ? vecops::convert<uint8_t>(
                  a[kk] * quant_multiplier +
                  static_cast<float32_t>(input_zero_point))
            : uint8_t{0};
        group_bits |= static_cast<uint32_t>(quantized) << (8 * lane);
      }
      const auto repeated_group = vec::bitcast(
          ATag{}, GroupTag{}, vec::fill(GroupTag{}, group_bits));
      const auto packed_values = vec::load(
          BTag{}, packed + group * b_group_stride,
          vec::opt::first(active * KPack), vec::opt::zero);
      sum = vec::widening_dot(
          AccTag{}, BTag{}, ATag{},
          packed_values, repeated_group, sum);
    }

    if constexpr (Dequantize) {
      auto value = vec::convert(OutputTag{}, AccTag{}, sum);
      value = vec::mul(
          OutputTag{}, value,
          vec::fill(OutputTag{}, row_dequant_scale));
      value = vec::mul(
          OutputTag{}, value,
          vec::load(
              OutputTag{}, column_scales + output_origin,
              vec::opt::first(active), vec::opt::zero));
      vec::store(
          OutputTag{}, static_cast<float32_t*>(output) + output_origin, value,
          vec::opt::first(active));
    } else {
      vec::store(
          AccTag{}, static_cast<int32_t*>(output) + output_origin, sum,
          vec::opt::first(active));
    }
    output_origin += active;
  }
}

VECOPS_NOINLINE VECOPS_FUNCTION_ALIGN(64) void
fused_runtime_quant_int8_packed_b_gemv(
    const float32_t* a, const int8_t* packed_b,
    nint_t b_outer_stride, nint_t b_group_stride,
    nint_t b_spatial_stride, nint_t packed_panel,
    const int32_t* correction, float32_t* output,
    const float32_t* column_scales,
    nint_t logical_n, nint_t logical_k,
    float32_t quant_multiplier, int32_t input_zero_point,
    float32_t row_dequant_scale) {
  runtime_quant_int8_packed_b_gemv_core<true>(
      a, packed_b, b_outer_stride, b_group_stride,
      b_spatial_stride, packed_panel, correction, output, column_scales,
      logical_n, logical_k, quant_multiplier, input_zero_point,
      row_dequant_scale);
}

VECOPS_NOINLINE VECOPS_FUNCTION_ALIGN(64) void
runtime_quant_int8_packed_b_gemv_accumulate(
    const float32_t* a, const int8_t* packed_b,
    nint_t b_outer_stride, nint_t b_group_stride,
    nint_t b_spatial_stride, nint_t packed_panel,
    const int32_t* prior, int32_t* output,
    nint_t logical_n, nint_t logical_k,
    float32_t quant_multiplier, int32_t input_zero_point) {
  runtime_quant_int8_packed_b_gemv_core<false>(
      a, packed_b, b_outer_stride, b_group_stride,
      b_spatial_stride, packed_panel, prior, output, nullptr,
      logical_n, logical_k, quant_multiplier, input_zero_point, 1.0f);
}

#endif

} // namespace vecops::kernel::matmul_details::sme
