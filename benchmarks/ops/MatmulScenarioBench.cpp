// @vecops-target-shards: 306

#include <cerrno>
#include <cstring>
#include <iostream>
#include <sys/syscall.h>
#include <unistd.h>
#include <utility>

#define VECOPS_SCENARIO_CATALOG_CASE_SHARDS 9
#include "MatmulScenarioBenchCommon.h"

#include "vecops/Features.h"
#include "vecops/matmul/Atom.h"

namespace vecops::bench::matmul {

inline constexpr int MatmulScenarioShardCount = 306;

template <int Shard>
void register_matmul_scenario_shard();

template <std::size_t Begin, std::size_t End>
void register_matmul_scenario_range() {
  if constexpr (Begin + 1 == End) {
    register_matmul_scenario_shard<static_cast<int>(Begin)>();
  } else if constexpr (Begin < End) {
    constexpr std::size_t Middle = Begin + (End - Begin) / 2;
    register_matmul_scenario_range<Begin, Middle>();
    register_matmul_scenario_range<Middle, End>();
  }
}

} // namespace vecops::bench::matmul

#if defined(VECOPS_TARGET_SHARD_ACTIVE)

namespace vecops::bench::matmul {

#if defined(ARCH_X86_FAMILY)
template <> struct AtomName<gemm::AMX_BF16F32> {
  static constexpr const char* value = "AMX_BF16F32";
};
#if defined(HAS_AMX_FP16)
template <> struct AtomName<gemm::AMX_F16F32> {
  static constexpr const char* value = "AMX_F16F32";
};
#endif
template <typename A, typename B>
using NativeI8 = gemm::AMX_I8I32<A, B>;
#else
template <> struct AtomName<gemm::SME_BF16F32> {
  static constexpr const char* value = "SME_BF16F32";
};
template <> struct AtomName<gemm::SME_F16F32> {
  static constexpr const char* value = "SME_F16F32";
};
template <> struct AtomName<gemm::SME_F32F32> {
  static constexpr const char* value = "SME_F32F32";
};
#if defined(HAS_SME_F64F64)
template <> struct AtomName<gemm::SME_F64F64> {
  static constexpr const char* value = "SME_F64F64";
};
#endif
template <typename A, typename B>
using NativeI8 = gemm::SME_I8I32<A, B>;
#endif

using I8S8S8 = NativeI8<int8_t, int8_t>;
using I8S8U8 = NativeI8<int8_t, uint8_t>;
using I8U8S8 = NativeI8<uint8_t, int8_t>;
using I8U8U8 = NativeI8<uint8_t, uint8_t>;

template <> struct AtomName<I8S8S8> {
  static constexpr const char* value = "I8I32_s8s8";
};
template <> struct AtomName<I8S8U8> {
  static constexpr const char* value = "I8I32_s8u8";
};
template <> struct AtomName<I8U8S8> {
  static constexpr const char* value = "I8I32_u8s8";
};
template <> struct AtomName<I8U8U8> {
  static constexpr const char* value = "I8I32_u8u8";
};

template <int TargetShard>
void register_matmul_scenario_shard() {
  static_assert(0 <= TargetShard && TargetShard < MatmulScenarioShardCount);
  constexpr int Shard = TargetShard /
      (2 * VECOPS_SCENARIO_CATALOG_CASE_SHARDS);
#if defined(ARCH_X86_FAMILY)
  constexpr nint_t KStep = 32;
  constexpr nint_t I8KStep = 64;
  const auto cases = scenario_cases(KStep);
  const auto batch_cases = batched_scenario_cases(KStep);
  const auto i8_cases = scenario_cases(I8KStep);
  const auto i8_batch_cases = batched_scenario_cases(I8KStep);
  if constexpr (Shard == 0) {
    register_scenario<gemm::AMX_BF16F32, bfloat16_t, bfloat16_t, float32_t>(cases);
    register_batched_scenario<
        gemm::AMX_BF16F32, bfloat16_t, bfloat16_t, float32_t>(batch_cases);
  } else if constexpr (Shard == 1) {
#if defined(HAS_AMX_FP16)
    register_scenario<gemm::AMX_F16F32, float16_t, float16_t, float32_t>(cases);
    register_batched_scenario<
        gemm::AMX_F16F32, float16_t, float16_t, float32_t>(batch_cases);
#endif
  } else if constexpr (Shard == 2) {
    register_scenario<I8S8S8, int8_t, int8_t, int32_t>(i8_cases);
    register_scenario<I8S8U8, int8_t, uint8_t, int32_t>(i8_cases);
    register_batched_scenario<I8S8S8, int8_t, int8_t, int32_t>(i8_batch_cases);
    register_batched_scenario<I8S8U8, int8_t, uint8_t, int32_t>(i8_batch_cases);
  } else if constexpr (Shard == 3) {
    register_scenario<I8U8S8, uint8_t, int8_t, int32_t>(i8_cases);
    register_scenario<I8U8U8, uint8_t, uint8_t, int32_t>(i8_cases);
    register_batched_scenario<I8U8S8, uint8_t, int8_t, int32_t>(i8_batch_cases);
    register_batched_scenario<I8U8U8, uint8_t, uint8_t, int32_t>(i8_batch_cases);
  } else if constexpr (Shard == 4) {
    register_scenario<gemm::AMX_BF16F32, float32_t, float32_t, float32_t>(cases);
    register_batched_scenario<
        gemm::AMX_BF16F32, float32_t, float32_t, float32_t>(batch_cases);
  } else if constexpr (Shard == 5) {
#if defined(HAS_AMX_FP16)
    register_scenario<gemm::AMX_F16F32, float32_t, float32_t, float32_t>(cases);
    register_batched_scenario<
        gemm::AMX_F16F32, float32_t, float32_t, float32_t>(batch_cases);
#endif
  } else if constexpr (Shard == 6) {
    register_scenario<gemm::AMX_BF16F32, bfloat16_t, bfloat16_t, bfloat16_t>(cases);
    register_scenario<gemm::AMX_BF16F32, bfloat16_t, bfloat16_t, float16_t>(cases);
    register_batched_scenario<
        gemm::AMX_BF16F32, bfloat16_t, bfloat16_t, bfloat16_t>(batch_cases);
    register_batched_scenario<
        gemm::AMX_BF16F32, bfloat16_t, bfloat16_t, float16_t>(batch_cases);
  } else if constexpr (Shard == 7) {
    register_scenario<gemm::AMX_BF16F32, bfloat16_t, bfloat16_t, float32_t,
                      InputPipeline::Convert, OutputPipeline::Accumulate>(cases);
    register_scenario<gemm::AMX_BF16F32, bfloat16_t, bfloat16_t, float32_t,
                      InputPipeline::Convert, OutputPipeline::Relu>(cases);
    register_scenario<gemm::AMX_BF16F32, bfloat16_t, bfloat16_t, float32_t,
                      InputPipeline::Convert, OutputPipeline::Scale>(cases);
    register_batched_scenario<
        gemm::AMX_BF16F32, bfloat16_t, bfloat16_t, float32_t,
        InputPipeline::Convert, OutputPipeline::Accumulate>(batch_cases);
    register_batched_scenario<
        gemm::AMX_BF16F32, bfloat16_t, bfloat16_t, float32_t,
        InputPipeline::Convert, OutputPipeline::Relu>(batch_cases);
    register_batched_scenario<
        gemm::AMX_BF16F32, bfloat16_t, bfloat16_t, float32_t,
        InputPipeline::Convert, OutputPipeline::Scale>(batch_cases);
  } else if constexpr (Shard == 8) {
    register_scenario<gemm::AMX_BF16F32, bfloat16_t, bfloat16_t, float32_t,
                      InputPipeline::Convert, OutputPipeline::Sigmoid>(cases);
    register_scenario<gemm::AMX_BF16F32, bfloat16_t, bfloat16_t, float32_t,
                      InputPipeline::Convert,
                      OutputPipeline::AccumulateReluScale>(cases);
    register_batched_scenario<
        gemm::AMX_BF16F32, bfloat16_t, bfloat16_t, float32_t,
        InputPipeline::Convert, OutputPipeline::Sigmoid>(batch_cases);
    register_batched_scenario<
        gemm::AMX_BF16F32, bfloat16_t, bfloat16_t, float32_t,
        InputPipeline::Convert,
        OutputPipeline::AccumulateReluScale>(batch_cases);
    register_scenario<
        gemm::AMX_BF16F32, bfloat16_t, bfloat16_t, float32_t,
        InputPipeline::Convert, OutputPipeline::Bias>(cases);
    register_scenario<
        gemm::AMX_BF16F32, bfloat16_t, bfloat16_t, float32_t,
        InputPipeline::Convert, OutputPipeline::BiasRelu>(cases);
    register_batched_scenario<
        gemm::AMX_BF16F32, bfloat16_t, bfloat16_t, float32_t,
        InputPipeline::Convert, OutputPipeline::Bias>(batch_cases);
    register_batched_scenario<
        gemm::AMX_BF16F32, bfloat16_t, bfloat16_t, float32_t,
        InputPipeline::Convert, OutputPipeline::BiasRelu>(batch_cases);
  } else if constexpr (Shard == 9) {
    register_scenario<I8S8U8, float32_t, float32_t, float32_t,
                      InputPipeline::Quantize4,
                      OutputPipeline::Dequantize>(i8_cases);
    register_batched_scenario<
        I8S8U8, float32_t, float32_t, float32_t,
        InputPipeline::Quantize4,
        OutputPipeline::Dequantize>(i8_batch_cases);
  } else if constexpr (Shard == 10) {
    register_scenario<I8S8U8, float32_t, float32_t, int8_t,
                      InputPipeline::Quantize4,
                      OutputPipeline::ReluRequantize>(i8_cases);
    register_batched_scenario<
        I8S8U8, float32_t, float32_t, int8_t,
        InputPipeline::Quantize4,
        OutputPipeline::ReluRequantize>(i8_batch_cases);
  } else if constexpr (Shard == 11) {
    register_scenario<
        I8U8S8, float32_t, float32_t, float32_t,
        InputPipeline::AsymmetricQuantize4Zp3,
        OutputPipeline::AsymmetricDequantize>(i8_cases);
    register_batched_scenario<
        I8U8S8, float32_t, float32_t, float32_t,
        InputPipeline::AsymmetricQuantize4Zp3,
        OutputPipeline::AsymmetricDequantize>(i8_batch_cases);
    register_scenario<
        I8U8S8, uint8_t, int8_t, float32_t,
        InputPipeline::PrequantizedAsymmetricZp3,
        OutputPipeline::AsymmetricDequantize>(i8_cases);
    register_batched_scenario<
        I8U8S8, uint8_t, int8_t, float32_t,
        InputPipeline::PrequantizedAsymmetricZp3,
        OutputPipeline::AsymmetricDequantize>(i8_batch_cases);
  } else if constexpr (Shard == 12) {
    register_scenario<
        I8S8S8, float32_t, int8_t, float32_t,
        InputPipeline::Quantize4, OutputPipeline::Dequantize,
        InputPipeline::Convert>(i8_cases);
    register_batched_scenario<
        I8S8S8, float32_t, int8_t, float32_t,
        InputPipeline::Quantize4, OutputPipeline::Dequantize, true,
        InputPipeline::Convert>(i8_batch_cases);
  } else if constexpr (Shard == 13) {
    register_scenario<
        gemm::AMX_BF16F32, bfloat16_t, bfloat16_t, bfloat16_t,
        InputPipeline::Convert, OutputPipeline::BiasRelu,
        InputPipeline::Convert, float32_t>(cases);
    register_batched_scenario<
        gemm::AMX_BF16F32, bfloat16_t, bfloat16_t, bfloat16_t,
        InputPipeline::Convert, OutputPipeline::BiasRelu, true,
        InputPipeline::Convert, float32_t>(batch_cases);
  } else if constexpr (Shard == 14) {
    register_scenario<
        gemm::AMX_BF16F32, float32_t, bfloat16_t, float32_t>(cases);
    register_scenario<
        gemm::AMX_BF16F32, bfloat16_t, float32_t, float32_t>(cases);
    register_batched_scenario<
        gemm::AMX_BF16F32, float32_t, bfloat16_t, float32_t>(batch_cases);
    register_batched_scenario<
        gemm::AMX_BF16F32, bfloat16_t, float32_t, float32_t>(batch_cases);
  } else if constexpr (Shard == 15) {
#if defined(HAS_AMX_FP16)
    register_scenario<
        gemm::AMX_F16F32, float16_t, float16_t, float16_t,
        InputPipeline::Convert, OutputPipeline::BiasRelu,
        InputPipeline::Convert, float32_t>(cases);
    register_batched_scenario<
        gemm::AMX_F16F32, float16_t, float16_t, float16_t,
        InputPipeline::Convert, OutputPipeline::BiasRelu, true,
        InputPipeline::Convert, float32_t>(batch_cases);
#endif
  } else {
    register_scenario<
        I8U8S8, float32_t, int8_t, float32_t,
        InputPipeline::AsymmetricQuantize4Zp3,
        OutputPipeline::AsymmetricDequantize,
        InputPipeline::Convert>(i8_cases);
    register_batched_scenario<
        I8U8S8, float32_t, int8_t, float32_t,
        InputPipeline::AsymmetricQuantize4Zp3,
        OutputPipeline::AsymmetricDequantize, true,
        InputPipeline::Convert>(i8_batch_cases);
  }
#else
  constexpr nint_t F32KStep = 1;
  constexpr nint_t BF16KStep = 2;
  constexpr nint_t I8KStep = 4;
  if constexpr (Shard == 0) {
    register_scenario<gemm::SME_F32F32, float32_t, float32_t, float32_t>(
        scenario_cases(F32KStep));
    register_batched_scenario<
        gemm::SME_F32F32, float32_t, float32_t, float32_t>(
            batched_scenario_cases(F32KStep));
  } else if constexpr (Shard == 1) {
    register_scenario<gemm::SME_BF16F32, bfloat16_t, bfloat16_t, float32_t>(
        scenario_cases(BF16KStep));
    register_batched_scenario<
        gemm::SME_BF16F32, bfloat16_t, bfloat16_t, float32_t>(
            batched_scenario_cases(BF16KStep));
  } else if constexpr (Shard == 2) {
    register_scenario<gemm::SME_F16F32, float16_t, float16_t, float32_t>(
        scenario_cases(BF16KStep));
    register_batched_scenario<
        gemm::SME_F16F32, float16_t, float16_t, float32_t>(
            batched_scenario_cases(BF16KStep));
  } else if constexpr (Shard == 3) {
#if defined(HAS_SME_F64F64)
    register_scenario<gemm::SME_F64F64, float64_t, float64_t, float64_t>(
        scenario_cases(F32KStep));
    register_scenario<gemm::SME_F64F64, float32_t, float32_t, float32_t>(
        scenario_cases(F32KStep));
    register_batched_scenario<
        gemm::SME_F64F64, float64_t, float64_t, float64_t>(
            batched_scenario_cases(F32KStep));
    register_batched_scenario<
        gemm::SME_F64F64, float32_t, float32_t, float32_t>(
            batched_scenario_cases(F32KStep));
#endif
  } else if constexpr (Shard == 4) {
    register_scenario<I8S8S8, int8_t, int8_t, int32_t>(scenario_cases(I8KStep));
    register_scenario<I8S8U8, int8_t, uint8_t, int32_t>(scenario_cases(I8KStep));
    register_batched_scenario<I8S8S8, int8_t, int8_t, int32_t>(
        batched_scenario_cases(I8KStep));
    register_batched_scenario<I8S8U8, int8_t, uint8_t, int32_t>(
        batched_scenario_cases(I8KStep));
  } else if constexpr (Shard == 5) {
    register_scenario<I8U8S8, uint8_t, int8_t, int32_t>(scenario_cases(I8KStep));
    register_scenario<I8U8U8, uint8_t, uint8_t, int32_t>(scenario_cases(I8KStep));
    register_batched_scenario<I8U8S8, uint8_t, int8_t, int32_t>(
        batched_scenario_cases(I8KStep));
    register_batched_scenario<I8U8U8, uint8_t, uint8_t, int32_t>(
        batched_scenario_cases(I8KStep));
  } else if constexpr (Shard == 6) {
    register_scenario<gemm::SME_BF16F32, float32_t, float32_t, float32_t>(
        scenario_cases(BF16KStep));
    register_batched_scenario<
        gemm::SME_BF16F32, float32_t, float32_t, float32_t>(
            batched_scenario_cases(BF16KStep));
  } else if constexpr (Shard == 7) {
    register_scenario<gemm::SME_F32F32, bfloat16_t, bfloat16_t, bfloat16_t>(
        scenario_cases(F32KStep));
    register_scenario<gemm::SME_F32F32, float16_t, float16_t, float16_t>(
        scenario_cases(F32KStep));
    register_batched_scenario<
        gemm::SME_F32F32, bfloat16_t, bfloat16_t, bfloat16_t>(
            batched_scenario_cases(F32KStep));
    register_batched_scenario<
        gemm::SME_F32F32, float16_t, float16_t, float16_t>(
            batched_scenario_cases(F32KStep));
  } else if constexpr (Shard == 8) {
    const auto cases = scenario_cases(BF16KStep);
    register_scenario<gemm::SME_BF16F32, bfloat16_t, bfloat16_t, float32_t,
                      InputPipeline::Convert, OutputPipeline::Accumulate>(cases);
    register_scenario<gemm::SME_BF16F32, bfloat16_t, bfloat16_t, float32_t,
                      InputPipeline::Convert, OutputPipeline::Relu>(cases);
    register_scenario<gemm::SME_BF16F32, bfloat16_t, bfloat16_t, float32_t,
                      InputPipeline::Convert, OutputPipeline::Scale>(cases);
    const auto batch_cases = batched_scenario_cases(BF16KStep);
    register_batched_scenario<
        gemm::SME_BF16F32, bfloat16_t, bfloat16_t, float32_t,
        InputPipeline::Convert, OutputPipeline::Accumulate>(batch_cases);
    register_batched_scenario<
        gemm::SME_BF16F32, bfloat16_t, bfloat16_t, float32_t,
        InputPipeline::Convert, OutputPipeline::Relu>(batch_cases);
    register_batched_scenario<
        gemm::SME_BF16F32, bfloat16_t, bfloat16_t, float32_t,
        InputPipeline::Convert, OutputPipeline::Scale>(batch_cases);
    register_scenario<
        gemm::SME_BF16F32, bfloat16_t, bfloat16_t, float32_t,
        InputPipeline::Convert, OutputPipeline::Bias>(cases);
    register_scenario<
        gemm::SME_BF16F32, bfloat16_t, bfloat16_t, float32_t,
        InputPipeline::Convert, OutputPipeline::BiasRelu>(cases);
    register_batched_scenario<
        gemm::SME_BF16F32, bfloat16_t, bfloat16_t, float32_t,
        InputPipeline::Convert, OutputPipeline::Bias>(batch_cases);
    register_batched_scenario<
        gemm::SME_BF16F32, bfloat16_t, bfloat16_t, float32_t,
        InputPipeline::Convert, OutputPipeline::BiasRelu>(batch_cases);
  } else if constexpr (Shard == 9) {
    const auto cases = scenario_cases(F32KStep);
    register_scenario<gemm::SME_F32F32, float32_t, float32_t, float32_t,
                      InputPipeline::Convert, OutputPipeline::Sigmoid>(cases);
    register_scenario<gemm::SME_F32F32, float32_t, float32_t, float32_t,
                      InputPipeline::Convert,
                      OutputPipeline::AccumulateReluScale>(cases);
    const auto batch_cases = batched_scenario_cases(F32KStep);
    register_batched_scenario<
        gemm::SME_F32F32, float32_t, float32_t, float32_t,
        InputPipeline::Convert, OutputPipeline::Sigmoid>(batch_cases);
    register_batched_scenario<
        gemm::SME_F32F32, float32_t, float32_t, float32_t,
        InputPipeline::Convert,
        OutputPipeline::AccumulateReluScale>(batch_cases);
  } else if constexpr (Shard == 10) {
    register_scenario<I8S8U8, float32_t, float32_t, float32_t,
                      InputPipeline::Quantize4,
                      OutputPipeline::Dequantize>(scenario_cases(I8KStep));
    register_batched_scenario<
        I8S8U8, float32_t, float32_t, float32_t,
        InputPipeline::Quantize4, OutputPipeline::Dequantize>(
            batched_scenario_cases(I8KStep));
  } else if constexpr (Shard == 11) {
    register_scenario<I8S8U8, float32_t, float32_t, int8_t,
                      InputPipeline::Quantize4,
                      OutputPipeline::ReluRequantize>(scenario_cases(I8KStep));
    register_batched_scenario<
        I8S8U8, float32_t, float32_t, int8_t,
        InputPipeline::Quantize4, OutputPipeline::ReluRequantize>(
            batched_scenario_cases(I8KStep));
    register_scenario<
        I8U8S8, float32_t, float32_t, float32_t,
        InputPipeline::AsymmetricQuantize4Zp3,
        OutputPipeline::AsymmetricDequantize>(scenario_cases(I8KStep));
    register_batched_scenario<
        I8U8S8, float32_t, float32_t, float32_t,
        InputPipeline::AsymmetricQuantize4Zp3,
        OutputPipeline::AsymmetricDequantize>(
            batched_scenario_cases(I8KStep));
    register_scenario<
        I8U8S8, uint8_t, int8_t, float32_t,
        InputPipeline::PrequantizedAsymmetricZp3,
        OutputPipeline::AsymmetricDequantize>(scenario_cases(I8KStep));
    register_batched_scenario<
        I8U8S8, uint8_t, int8_t, float32_t,
        InputPipeline::PrequantizedAsymmetricZp3,
        OutputPipeline::AsymmetricDequantize>(
            batched_scenario_cases(I8KStep));
  } else if constexpr (Shard == 12) {
    register_scenario<gemm::SME_F16F32, float32_t, float32_t, float32_t>(
        scenario_cases(BF16KStep));
    register_batched_scenario<
        gemm::SME_F16F32, float32_t, float32_t, float32_t>(
            batched_scenario_cases(BF16KStep));
  } else if constexpr (Shard == 13) {
    register_scenario<gemm::SME_F32F32, bfloat16_t, bfloat16_t, float32_t>(
        scenario_cases(F32KStep));
    register_scenario<gemm::SME_F32F32, float16_t, float16_t, float32_t>(
        scenario_cases(F32KStep));
    register_batched_scenario<
        gemm::SME_F32F32, bfloat16_t, bfloat16_t, float32_t>(
            batched_scenario_cases(F32KStep));
    register_batched_scenario<
        gemm::SME_F32F32, float16_t, float16_t, float32_t>(
            batched_scenario_cases(F32KStep));
  } else if constexpr (Shard == 14) {
    register_scenario<
        I8S8S8, float32_t, int8_t, float32_t,
        InputPipeline::Quantize4, OutputPipeline::Dequantize,
        InputPipeline::Convert>(scenario_cases(I8KStep));
    register_batched_scenario<
        I8S8S8, float32_t, int8_t, float32_t,
        InputPipeline::Quantize4, OutputPipeline::Dequantize, true,
        InputPipeline::Convert>(batched_scenario_cases(I8KStep));
  } else if constexpr (Shard == 15) {
    register_scenario<
        gemm::SME_BF16F32, bfloat16_t, bfloat16_t, bfloat16_t,
        InputPipeline::Convert, OutputPipeline::BiasRelu,
        InputPipeline::Convert, float32_t>(scenario_cases(BF16KStep));
    register_batched_scenario<
        gemm::SME_BF16F32, bfloat16_t, bfloat16_t, bfloat16_t,
        InputPipeline::Convert, OutputPipeline::BiasRelu, true,
        InputPipeline::Convert, float32_t>(
            batched_scenario_cases(BF16KStep));
    register_scenario<
        gemm::SME_F16F32, float16_t, float16_t, float16_t,
        InputPipeline::Convert, OutputPipeline::BiasRelu,
        InputPipeline::Convert, float32_t>(scenario_cases(BF16KStep));
    register_batched_scenario<
        gemm::SME_F16F32, float16_t, float16_t, float16_t,
        InputPipeline::Convert, OutputPipeline::BiasRelu, true,
        InputPipeline::Convert, float32_t>(
            batched_scenario_cases(BF16KStep));
  } else {
    register_scenario<
        I8U8S8, float32_t, int8_t, float32_t,
        InputPipeline::AsymmetricQuantize4Zp3,
        OutputPipeline::AsymmetricDequantize,
        InputPipeline::Convert>(scenario_cases(I8KStep));
    register_batched_scenario<
        I8U8S8, float32_t, int8_t, float32_t,
        InputPipeline::AsymmetricQuantize4Zp3,
        OutputPipeline::AsymmetricDequantize, true,
        InputPipeline::Convert>(batched_scenario_cases(I8KStep));
  }
#endif
}

} // namespace vecops::bench::matmul

static_assert(
    VECOPS_TARGET_SHARD_COUNT ==
    vecops::bench::matmul::MatmulScenarioShardCount);
template void vecops::bench::matmul::register_matmul_scenario_shard<
    VECOPS_TARGET_SHARD_INDEX>();

#else

namespace {

#if defined(ARCH_X86_FAMILY)
bool enable_matrix_extension() {
  constexpr long ArchRequestXcompPerm = 0x1023;
  constexpr long XfeatureTileData = 18;
  return syscall(
      SYS_arch_prctl, ArchRequestXcompPerm, XfeatureTileData) == 0;
}
#else
bool enable_matrix_extension() { return true; }
#endif

} // namespace

int main(int argc, char** argv) {
  if (!enable_matrix_extension()) {
    std::cerr << "Unable to enable matrix extension: "
              << std::strerror(errno) << '\n';
    return 1;
  }
  vecops::bench::matmul::register_matmul_scenario_range<
      0, vecops::bench::matmul::MatmulScenarioShardCount>();
  return vecops::bench::matmul::run_benchmarks(
      argc, argv, "matmul_scenario");
}

#endif
