//
// Copyright (c) vecops contributors.
//
// Kleidi-style runtime-quantized INT8 1x4VL GEMV leaf. This source is
// intentionally excluded from generic libvecops.a and compiled in the shared
// ISA-specific SME matmul archive.

#include <utility>

#include "vecops/CoreTypes.h"
#include "vecops/vec/Vec.h"
#include "vecops/vec/details/sve/Basic.h"

namespace vecops::kernel::matmul_details::sme {

#if defined(HAS_SME_FA64) && defined(__ARM_FEATURE_SVE_MATMUL_INT8)

// TODO: Keep this mixed-sign lane-DOT helper SME-backend-local until its
// packed-B segment contract and signedness semantics have another backend
// consumer. Do not expose it through the public vec API.
template <int Lane>
VECOPS_ALWAYS_INLINE svint32_t runtime_quant_sudot_lane(
    svint32_t accumulator, svint8_t packed_b,
    svuint8_t quantized_a) {
  static_assert(0 <= Lane && Lane < 4);
  return svsudot_lane_s32(
      accumulator, packed_b, quantized_a, Lane);
}

VECOPS_NOINLINE VECOPS_FUNCTION_ALIGN(64) void
fused_runtime_quant_int8_packed_b_gemv_1x4vl(
    const float32_t* a, const int8_t* packed_b,
    nint_t b_outer_stride, nint_t b_group_stride,
    nint_t b_spatial_stride, nint_t packed_panel,
    const int32_t* correction, float32_t* output,
    const float32_t* column_scales,
    nint_t logical_n, nint_t logical_k,
    float32_t quant_multiplier, int32_t input_zero_point,
    float32_t row_dequant_scale) {
  using ATag = vec::ScalableTag<uint8_t, 0>;
  using AWideTag = vec::Rebind<float32_t, ATag>;
  using BTag = vec::ScalableTag<int8_t, 0>;
  using AccTag = vec::ScalableTag<int32_t, 0>;
  using OutputTag = vec::ScalableTag<float32_t, 0>;
  constexpr nint_t KPack = 4;
  constexpr nint_t KChunk = 64;
  const nint_t lanes = vec::size(AccTag{});
  const nint_t output_block = 4 * lanes;

  const auto packed_at = [&](nint_t spatial) VECOPS_INLINE_LAMBDA {
    const nint_t offset =
        (spatial / packed_panel) * b_outer_stride +
        (spatial % packed_panel) * b_spatial_stride;
    return packed_b + offset;
  };

  VECOPS_LOOP_ALIGN(64) for (
      nint_t output_origin = 0;
      output_origin < logical_n; output_origin += output_block) {
    const auto* packed0 = packed_at(output_origin);
    const auto* packed1 = packed_at(output_origin + lanes);
    const auto* packed2 = packed_at(output_origin + 2 * lanes);
    const auto* packed3 = packed_at(output_origin + 3 * lanes);

    auto sum0 = vec::details::sve_basic_raw_word(vec::load(
        AccTag{}, correction + output_origin));
    auto sum1 = vec::details::sve_basic_raw_word(vec::load(
        AccTag{}, correction + output_origin + lanes));
    auto sum2 = vec::details::sve_basic_raw_word(vec::load(
        AccTag{}, correction + output_origin + 2 * lanes));
    auto sum3 = vec::details::sve_basic_raw_word(vec::load(
        AccTag{}, correction + output_origin + 3 * lanes));

    VECOPS_LOOP_ALIGN(64) for (
        nint_t k_origin = 0; k_origin < logical_k;
        k_origin += KChunk) {
      auto a_values = vec::load(AWideTag{}, a + k_origin);
      a_values = vec::add(
          AWideTag{},
          vec::mul(
              AWideTag{}, a_values,
              vec::fill(AWideTag{}, quant_multiplier)),
          vec::fill(
              AWideTag{}, static_cast<float32_t>(input_zero_point)));
      const auto quantized_a = vec::details::sve_basic_raw_word(
          vec::convert(ATag{}, AWideTag{}, a_values));
      const nint_t first_group = k_origin / KPack;

      const auto accumulate_segment = [&]<int Segment>(
          svint32_t accumulator, const int8_t* packed)
          VECOPS_INLINE_LAMBDA {
        const auto shared_quad = svdupq_lane_u8(quantized_a, Segment);
        [&]<int... Lane>(std::integer_sequence<int, Lane...>)
            VECOPS_INLINE_LAMBDA {
          ((accumulator = runtime_quant_sudot_lane<Lane>(
                accumulator,
                vec::details::sve_basic_raw_word(vec::load(
                    BTag{},
                    packed +
                        (first_group + Segment * 4 + Lane) *
                            b_group_stride)),
                shared_quad)), ...);
        }(std::integer_sequence<int, 0, 1, 2, 3>{});
        return accumulator;
      };
      const auto accumulate16 = [&](
          svint32_t accumulator, const int8_t* packed)
          VECOPS_INLINE_LAMBDA {
        accumulator =
            accumulate_segment.template operator()<0>(accumulator, packed);
        accumulator =
            accumulate_segment.template operator()<1>(accumulator, packed);
        accumulator =
            accumulate_segment.template operator()<2>(accumulator, packed);
        return accumulate_segment.template operator()<3>(
            accumulator, packed);
      };
      sum0 = accumulate16(sum0, packed0);
      sum1 = accumulate16(sum1, packed1);
      sum2 = accumulate16(sum2, packed2);
      sum3 = accumulate16(sum3, packed3);
    }

    const auto store = [&](int block, svint32_t accumulator)
        VECOPS_INLINE_LAMBDA {
      const nint_t origin = output_origin + block * lanes;
      auto value = vec::convert(
          OutputTag{}, AccTag{},
          vec::details::sve_basic_wrap_word<AccTag>(accumulator));
      value = vec::mul(
          OutputTag{}, value,
          vec::fill(OutputTag{}, row_dequant_scale));
      value = vec::mul(
          OutputTag{}, value,
          vec::load(OutputTag{}, column_scales + origin));
      vec::store(OutputTag{}, output + origin, value);
    };
    store(0, sum0);
    store(1, sum1);
    store(2, sum2);
    store(3, sum3);
  }
}

#endif

} // namespace vecops::kernel::matmul_details::sme
