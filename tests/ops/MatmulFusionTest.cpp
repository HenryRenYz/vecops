// @vecops-target-shards-x86: 2
// @vecops-target-shards-ARM: 8

#include "vecops/Features.h"
#include "MatmulTestArch.h"

#if defined(ARCH_X86_FAMILY)

#if defined(VECOPS_TARGET_SHARD_ACTIVE)

#include <gtest/gtest.h>


#include "MatmulConversionTestCommon.h"

#include "vecops/Features.h"
#include "vecops/gemm/Atoms.h"

namespace {

static_assert(VECOPS_TARGET_SHARD_COUNT == 2);


#if VECOPS_TARGET_SHARD_INDEX == 0

// Floating-point fusion coverage.
TEST(MatmulFusionTest, RuntimePerColumnScaleAndClampAreFused) {
  ASSERT_TRUE(vecops::test::matmul::MatmulTestArchTraits::enable());
  vecops::test::matmul::check_native_extent_pair<
      vecops::gemm::AMX_BF16F32, 19, 23, 65>();
  vecops::test::matmul::check_runtime_and_per_column_scale<
      vecops::gemm::AMX_BF16F32>(19, 23, 65);
  vecops::test::matmul::check_bias_clamp<
      vecops::gemm::AMX_BF16F32>(19, 23, 65);
#if defined(HAS_AMX_FP16)
  vecops::test::matmul::check_runtime_and_per_column_scale<
      vecops::gemm::AMX_F16F32>(19, 23, 65);
  vecops::test::matmul::check_bias_clamp<
      vecops::gemm::AMX_F16F32>(19, 23, 65);
#endif
}

TEST(MatmulFusionTest, MixedBiasNarrowAndActivationsAreFused) {
  ASSERT_TRUE(vecops::test::matmul::MatmulTestArchTraits::enable());
  vecops::test::matmul::check_native_extent_pair<
      vecops::gemm::AMX_BF16F32, 1, 8, 129>();
  vecops::test::matmul::check_mixed_bias_relu<
      vecops::gemm::AMX_BF16F32,
      vecops::bfloat16_t, vecops::bfloat16_t, vecops::bfloat16_t>(
          19, 23, 65);
  vecops::test::matmul::check_relu_and_sigmoid<
      vecops::gemm::AMX_BF16F32>(19, 23, 65);
  vecops::test::matmul::check_relu_and_sigmoid<
      vecops::gemm::AMX_BF16F32>(1, 8, 129);
  vecops::test::matmul::check_silu_and_bias_silu<
      vecops::gemm::AMX_BF16F32>(1, 8, 129);
  vecops::test::matmul::check_relu_and_sigmoid<
      vecops::gemm::AMX_BF16F32>(8, 1, 129);
#if defined(HAS_AMX_FP16)
  vecops::test::matmul::check_mixed_bias_relu<
      vecops::gemm::AMX_F16F32,
      vecops::float16_t, vecops::float16_t, vecops::float16_t>(
          19, 23, 65);
#endif
}

#elif VECOPS_TARGET_SHARD_INDEX == 1

// Integer and quantized fusion coverage.
TEST(MatmulFusionTest, DualAsymmetricU8UsesRowAndColumnCorrection) {
  ASSERT_TRUE(vecops::test::matmul::MatmulTestArchTraits::enable());
#if defined(HAS_AMX_INT8)
  vecops::test::matmul::check_native_extent_pair<
      vecops::gemm::AMX_I8I32<uint8_t, uint8_t>, 19, 23, 65>();
  vecops::test::matmul::check_dual_asymmetric_quantized<
      vecops::gemm::AMX_I8I32<uint8_t, uint8_t>>(19, 23, 65);
  vecops::test::matmul::check_dual_asymmetric_quantized<
      vecops::gemm::AMX_I8I32<uint8_t, uint8_t>>(
          vecops::meta::cint<32>, vecops::meta::cint<64>,
          vecops::meta::cint<128>, true);
#endif
}

TEST(MatmulFusionTest, IndependentQuantizationAndNativeI8Accumulate) {
  ASSERT_TRUE(vecops::test::matmul::MatmulTestArchTraits::enable());
#if defined(HAS_AMX_INT8)
  vecops::test::matmul::check_native_extent_pair<
      vecops::gemm::AMX_I8I32<int8_t, uint8_t>, 19, 23, 65>();
  vecops::test::matmul::check_quantized_a_direct_b<
      vecops::gemm::AMX_I8I32<int8_t, int8_t>>(19, 23, 65);
  vecops::test::matmul::check_native_integer_accumulate<
      vecops::gemm::AMX_I8I32<int8_t, int8_t>>(19, 23, 65);
  vecops::test::matmul::check_native_integer_accumulate<
      vecops::gemm::AMX_I8I32<int8_t, uint8_t>>(19, 23, 65);
  vecops::test::matmul::check_native_integer_accumulate<
      vecops::gemm::AMX_I8I32<uint8_t, int8_t>>(19, 23, 65);
  vecops::test::matmul::check_native_integer_accumulate<
      vecops::gemm::AMX_I8I32<uint8_t, uint8_t>>(19, 23, 65);
  vecops::test::matmul::check_asymmetric_quantized<
      vecops::gemm::AMX_I8I32<uint8_t, int8_t>, true>(19, 23, 65);
  vecops::test::matmul::check_batched_runtime_per_row_column_quantization<
      vecops::gemm::AMX_I8I32<uint8_t, int8_t>, false, true>(
          3, 5, 7, 65);
  vecops::test::matmul::check_batched_runtime_per_row_column_quantization<
      vecops::gemm::AMX_I8I32<uint8_t, int8_t>, true, false>(
          3, 5, 7, 65);
  vecops::test::matmul::check_integer_bias_per_column_quantization<
      vecops::gemm::AMX_I8I32<int8_t, int8_t>>(1, 8, 129);
#endif
}

#else
#error "Unexpected MatmulFusionTest shard index"
#endif

} // namespace

#endif

#else

#if defined(VECOPS_TARGET_SHARD_ACTIVE)

#include <gtest/gtest.h>

#include "MatmulConversionTestCommon.h"

#include "vecops/gemm/Atoms.h"

namespace {

static_assert(VECOPS_TARGET_SHARD_COUNT == 8);

#if VECOPS_TARGET_SHARD_INDEX == 0

// Non-F64 floating-point fusion coverage.
TEST(MatmulFusionTest, RuntimePerColumnScaleAndClampAreFused) {
  vecops::test::matmul::check_native_extent_pair<
      vecops::gemm::SME_BF16F32, 19, 23, 17>();
  vecops::test::matmul::check_runtime_and_per_column_scale<
      vecops::gemm::SME_BF16F32>(19, 23, 17);
  vecops::test::matmul::check_bias_clamp<
      vecops::gemm::SME_BF16F32>(19, 23, 17);
  vecops::test::matmul::check_runtime_and_per_column_scale<
      vecops::gemm::SME_BF16F32>(
          vecops::meta::cint<1>, vecops::meta::cint<8>,
          vecops::meta::cint<129>);
  vecops::test::matmul::check_runtime_and_per_column_scale<
      vecops::gemm::SME_BF16F32>(
          vecops::meta::cint<8>, vecops::meta::cint<1>,
          vecops::meta::cint<129>);
  vecops::test::matmul::check_runtime_and_per_column_scale<
      vecops::gemm::SME_BF16F32>(
          vecops::meta::dyn<1, 1, 1>(1),
          vecops::meta::dyn<1, 1, 8>(8),
          vecops::meta::dyn<1, 0, 129>(129));
}

#elif VECOPS_TARGET_SHARD_INDEX == 1

TEST(MatmulFusionTest, MixedBiasNarrowAndActivationsAreFused) {
  vecops::test::matmul::check_native_extent_pair<
      vecops::gemm::SME_F32F32, 19, 23, 9>();
  vecops::test::matmul::check_mixed_bias_relu<
      vecops::gemm::SME_BF16F32,
      vecops::bfloat16_t, vecops::bfloat16_t, vecops::bfloat16_t>(
          19, 23, 17);
  vecops::test::matmul::check_mixed_bias_relu<
      vecops::gemm::SME_F16F32,
      vecops::float16_t, vecops::float16_t, vecops::float16_t>(
          19, 23, 17);
  vecops::test::matmul::check_mixed_bias_relu<
      vecops::gemm::SME_F32F32,
      vecops::bfloat16_t, vecops::bfloat16_t, vecops::bfloat16_t>(
          19, 23, 9);
  vecops::test::matmul::check_mixed_bias_relu<
      vecops::gemm::SME_F32F32,
      vecops::float16_t, vecops::float16_t, vecops::float16_t>(
          19, 23, 9);
  vecops::test::matmul::check_relu_and_sigmoid<
      vecops::gemm::SME_BF16F32>(19, 23, 17);
  vecops::test::matmul::check_relu_and_sigmoid<
      vecops::gemm::SME_BF16F32>(1, 8, 129);
  vecops::test::matmul::check_silu_and_bias_silu<
      vecops::gemm::SME_BF16F32>(1, 8, 129);
  vecops::test::matmul::check_relu_and_sigmoid<
      vecops::gemm::SME_BF16F32>(8, 1, 129);
}

#elif VECOPS_TARGET_SHARD_INDEX == 2

// Integer and quantized fusion coverage.
TEST(MatmulFusionTest, DualAsymmetricU8UsesRowAndColumnCorrection) {
  vecops::test::matmul::check_native_extent_pair<
      vecops::gemm::SME_I8I32<uint8_t, uint8_t>, 19, 23, 33>();
  vecops::test::matmul::check_dual_asymmetric_quantized<
      vecops::gemm::SME_I8I32<uint8_t, uint8_t>>(19, 23, 33);
  vecops::test::matmul::check_dual_asymmetric_quantized<
      vecops::gemm::SME_I8I32<uint8_t, uint8_t>>(
          vecops::meta::cint<32>, vecops::meta::cint<64>,
          vecops::meta::cint<128>, true);
}

#elif VECOPS_TARGET_SHARD_INDEX == 3

TEST(MatmulFusionTest, IndependentQuantizationAndNativeI8Accumulate) {
  vecops::test::matmul::check_native_extent_pair<
      vecops::gemm::SME_I8I32<int8_t, uint8_t>, 19, 23, 33>();
  vecops::test::matmul::check_quantized_a_direct_b<
      vecops::gemm::SME_I8I32<int8_t, int8_t>>(19, 23, 33);
  vecops::test::matmul::check_native_integer_accumulate<
      vecops::gemm::SME_I8I32<int8_t, int8_t>>(19, 23, 33);
  vecops::test::matmul::check_native_integer_accumulate<
      vecops::gemm::SME_I8I32<int8_t, uint8_t>>(19, 23, 33);
  vecops::test::matmul::check_native_integer_accumulate<
      vecops::gemm::SME_I8I32<uint8_t, int8_t>>(19, 23, 33);
  vecops::test::matmul::check_native_integer_accumulate<
      vecops::gemm::SME_I8I32<uint8_t, uint8_t>>(19, 23, 33);
  vecops::test::matmul::check_asymmetric_quantized<
      vecops::gemm::SME_I8I32<uint8_t, int8_t>, true>(19, 23, 33);
  vecops::test::matmul::check_batched_runtime_per_row_column_quantization<
      vecops::gemm::SME_I8I32<uint8_t, int8_t>, false, true>(
          3, 5, 7, 33);
  vecops::test::matmul::check_batched_runtime_per_row_column_quantization<
      vecops::gemm::SME_I8I32<uint8_t, int8_t>, true, false>(
          3, 5, 7, 33);
  vecops::test::matmul::check_integer_bias_per_column_quantization<
      vecops::gemm::SME_I8I32<int8_t, int8_t>>(1, 8, 129);
}

#elif VECOPS_TARGET_SHARD_INDEX == 4

// F64 runtime fusion and the heavy auto-packing conversion stay isolated from
// the ordinary floating-point shard.
#if defined(HAS_SME_F64F64)
TEST(MatmulFusionTest, F64RuntimePerColumnScaleAndClampAreFused) {
  vecops::test::matmul::check_native_extent_pair<
      vecops::gemm::SME_F64F64, 19, 23, 9>();
  vecops::test::matmul::check_runtime_and_per_column_scale<
      vecops::gemm::SME_F64F64>(19, 23, 9);
  vecops::test::matmul::check_bias_clamp<
      vecops::gemm::SME_F64F64>(19, 23, 9);
}

#endif

#elif VECOPS_TARGET_SHARD_INDEX == 5

#if defined(HAS_SME_F64F64)
TEST(MatmulFusionTest, LargeF32ToF64ConversionUsesAutoPacking) {
  vecops::test::matmul::check_native_extent_pair<
      vecops::gemm::SME_F64F64, 64, 256, 256>();
  vecops::test::matmul::check_conversion<
      vecops::gemm::SME_F64F64,
      vecops::float32_t, vecops::float32_t, vecops::float32_t>(
          vecops::meta::cint<64>, vecops::meta::cint<256>,
          vecops::meta::cint<256>, true);
}
#endif

#elif VECOPS_TARGET_SHARD_INDEX == 6

TEST(MatmulFusionTest, RuntimePerColumnScaleAndClampF32AreFused) {
  vecops::test::matmul::check_native_extent_pair<
      vecops::gemm::SME_F32F32, 19, 23, 9>();
  vecops::test::matmul::check_runtime_and_per_column_scale<
      vecops::gemm::SME_F32F32>(19, 23, 9);
  vecops::test::matmul::check_bias_clamp<
      vecops::gemm::SME_F32F32>(19, 23, 9);
}

#elif VECOPS_TARGET_SHARD_INDEX == 7

TEST(MatmulFusionTest, RuntimePerColumnScaleAndClampF16AreFused) {
  vecops::test::matmul::check_native_extent_pair<
      vecops::gemm::SME_F16F32, 19, 23, 17>();
  vecops::test::matmul::check_runtime_and_per_column_scale<
      vecops::gemm::SME_F16F32>(19, 23, 17);
  vecops::test::matmul::check_bias_clamp<
      vecops::gemm::SME_F16F32>(19, 23, 17);
}

#else
#error "Unexpected MatmulFusionTest shard index"
#endif

} // namespace

#endif

#endif
