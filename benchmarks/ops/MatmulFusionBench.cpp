// @vecops-target-shards: 396

#include <cerrno>
#include <cstring>
#include <iostream>
#include <stdexcept>
#include <sys/syscall.h>
#include <unistd.h>
#include <utility>

#define VECOPS_SCENARIO_CATALOG_CASE_SHARDS 6
#include "MatmulScenarioBenchCommon.h"

#include "vecops/platform/Features.h"
#include "vecops/matmul/Atom.h"

namespace vecops::bench::matmul {

inline constexpr int MatmulFusionScenarioShardCount = 396;

template <int Shard>
void register_matmul_fusion_scenario_shard();

template <std::size_t Begin, std::size_t End>
void register_matmul_fusion_scenario_range() {
  if constexpr (Begin + 1 == End) {
    register_matmul_fusion_scenario_shard<static_cast<int>(Begin)>();
  } else if constexpr (Begin < End) {
    constexpr std::size_t Middle = Begin + (End - Begin) / 2;
    register_matmul_fusion_scenario_range<Begin, Middle>();
    register_matmul_fusion_scenario_range<Middle, End>();
  }
}

} // namespace vecops::bench::matmul

#if defined(VECOPS_TARGET_SHARD_ACTIVE)

namespace vecops::bench::matmul {

#if defined(ARCH_X86_FAMILY)
using FusionFloatAtom = ::vecops::matmul::AMX_BF16F32;
template <> struct AtomName<::vecops::matmul::AMX_BF16F32> {
  static constexpr const char* value = "AMX_BF16F32";
};
#if defined(HAS_AMX_FP16)
template <> struct AtomName<::vecops::matmul::AMX_F16F32> {
  static constexpr const char* value = "AMX_F16F32";
};
#endif
template <typename A, typename B>
using FusionI8 = ::vecops::matmul::AMX_I8I32<A, B>;
#else
using FusionFloatAtom = ::vecops::matmul::SME_BF16F32;
template <> struct AtomName<::vecops::matmul::SME_BF16F32> {
  static constexpr const char* value = "SME_BF16F32";
};
template <> struct AtomName<::vecops::matmul::SME_F16F32> {
  static constexpr const char* value = "SME_F16F32";
};
template <> struct AtomName<::vecops::matmul::SME_F32F32> {
  static constexpr const char* value = "SME_F32F32";
};
#if defined(HAS_SME_F64F64)
template <> struct AtomName<::vecops::matmul::SME_F64F64> {
  static constexpr const char* value = "SME_F64F64";
};
#endif
template <typename A, typename B>
using FusionI8 = ::vecops::matmul::SME_I8I32<A, B>;
#endif

using FusionI8S8S8 = FusionI8<int8_t, int8_t>;
using FusionI8S8U8 = FusionI8<int8_t, uint8_t>;
using FusionI8U8S8 = FusionI8<uint8_t, int8_t>;
using FusionI8U8U8 = FusionI8<uint8_t, uint8_t>;

template <> struct AtomName<FusionI8S8S8> {
  static constexpr const char* value = "I8I32_s8s8";
};
template <> struct AtomName<FusionI8S8U8> {
  static constexpr const char* value = "I8I32_s8u8";
};
template <> struct AtomName<FusionI8U8S8> {
  static constexpr const char* value = "I8I32_u8s8";
};
template <> struct AtomName<FusionI8U8U8> {
  static constexpr const char* value = "I8I32_u8u8";
};

template <typename Atom, typename MemoryA, typename MemoryB, typename MemoryC,
          InputPipeline InPipeline, OutputPipeline OutPipeline,
          InputPipeline InPipelineB = InPipeline,
          typename MemoryCInput = MemoryC>
void register_rank2_and_batched_fusion(nint_t k_step) {
  constexpr nint_t CatalogKStep =
      std::remove_cv_t<decltype(Atom::K_R)>::value;
  if (k_step != CatalogKStep) {
    throw std::logic_error("fusion catalog K step does not match Atom::K_R");
  }
  register_scenario<
      Atom, MemoryA, MemoryB, MemoryC, InPipeline, OutPipeline,
      InPipelineB, MemoryCInput>(
          fusion_scenario_cases(CatalogKStep));
  register_batched_scenario<
      Atom, MemoryA, MemoryB, MemoryC, InPipeline, OutPipeline, false,
      InPipelineB, MemoryCInput>(
          batched_fusion_scenario_cases(CatalogKStep));
}

template <typename Atom, typename MemoryA, typename MemoryB, typename MemoryC,
          InputPipeline InPipeline, OutputPipeline OutPipeline,
          InputPipeline InPipelineB = InPipeline,
          typename MemoryCInput = MemoryC>
void register_rank2_fusion(nint_t k_step) {
  constexpr nint_t CatalogKStep =
      std::remove_cv_t<decltype(Atom::K_R)>::value;
  if (k_step != CatalogKStep) {
    throw std::logic_error("fusion catalog K step does not match Atom::K_R");
  }
  register_scenario<Atom, MemoryA, MemoryB, MemoryC, InPipeline, OutPipeline,
                    InPipelineB, MemoryCInput>(
      fusion_scenario_cases(CatalogKStep));
}

template <typename Atom, typename MemoryA, typename MemoryB, typename MemoryC,
          InputPipeline InPipeline, OutputPipeline OutPipeline,
          InputPipeline InPipelineB = InPipeline,
          typename MemoryCInput = MemoryC>
void register_batched_fusion(nint_t k_step) {
  constexpr nint_t CatalogKStep =
      std::remove_cv_t<decltype(Atom::K_R)>::value;
  if (k_step != CatalogKStep) {
    throw std::logic_error("fusion catalog K step does not match Atom::K_R");
  }
  register_batched_scenario<Atom, MemoryA, MemoryB, MemoryC, InPipeline,
                            OutPipeline, false, InPipelineB, MemoryCInput>(
      batched_fusion_scenario_cases(CatalogKStep));
}

template <int TargetShard>
void register_matmul_fusion_scenario_shard() {
  static_assert(
      0 <= TargetShard && TargetShard < MatmulFusionScenarioShardCount);
  constexpr int Shard = TargetShard /
      (2 * VECOPS_SCENARIO_CATALOG_CASE_SHARDS);
#if defined(ARCH_X86_FAMILY)
  constexpr nint_t FloatKStep = 32;
  constexpr nint_t I8KStep = 64;
#else
  constexpr nint_t FloatKStep = 2;
  constexpr nint_t I8KStep = 4;
#endif

  if constexpr (Shard == 0) {
    register_rank2_and_batched_fusion<
        FusionFloatAtom, bfloat16_t, bfloat16_t, float32_t,
        InputPipeline::Convert, OutputPipeline::Convert>(FloatKStep);
    register_rank2_and_batched_fusion<
        FusionFloatAtom, bfloat16_t, bfloat16_t, float32_t,
        InputPipeline::Convert, OutputPipeline::Clamp>(FloatKStep);
    register_rank2_and_batched_fusion<
        FusionFloatAtom, bfloat16_t, bfloat16_t, float32_t,
        InputPipeline::Convert, OutputPipeline::BiasClamp>(FloatKStep);
  } else if constexpr (Shard == 1) {
    register_rank2_and_batched_fusion<
        FusionFloatAtom, bfloat16_t, bfloat16_t, float32_t,
        InputPipeline::Convert, OutputPipeline::Scale>(FloatKStep);
    register_rank2_and_batched_fusion<
        FusionFloatAtom, bfloat16_t, bfloat16_t, float32_t,
        InputPipeline::Convert, OutputPipeline::DynamicScale>(FloatKStep);
    register_rank2_and_batched_fusion<
        FusionFloatAtom, bfloat16_t, bfloat16_t, float32_t,
        InputPipeline::Convert, OutputPipeline::PerColumnScale>(FloatKStep);
  } else if constexpr (Shard == 2) {
    register_rank2_and_batched_fusion<
        FusionFloatAtom, float32_t, float32_t, float32_t,
        InputPipeline::Convert, OutputPipeline::DynamicScale>(FloatKStep);
    register_rank2_and_batched_fusion<
        FusionFloatAtom, float32_t, float32_t, float32_t,
        InputPipeline::Convert, OutputPipeline::PerColumnScale>(FloatKStep);
  } else if constexpr (Shard == 3) {
    register_rank2_and_batched_fusion<
        FusionFloatAtom, bfloat16_t, bfloat16_t, bfloat16_t,
        InputPipeline::Convert, OutputPipeline::Clamp>(FloatKStep);
    register_rank2_and_batched_fusion<
        FusionFloatAtom, bfloat16_t, bfloat16_t, float16_t,
        InputPipeline::Convert, OutputPipeline::Clamp>(FloatKStep);
  } else if constexpr (Shard == 4) {
    register_rank2_and_batched_fusion<
        FusionI8S8U8, float32_t, float32_t, uint8_t,
        InputPipeline::Quantize4,
        OutputPipeline::RequantizeU8Zp7>(I8KStep);
    register_rank2_and_batched_fusion<
        FusionI8U8U8, float32_t, float32_t, float32_t,
        InputPipeline::AsymmetricQuantize4Zp3Zp5,
        OutputPipeline::AsymmetricDequantize>(I8KStep);
  } else if constexpr (Shard == 5) {
#if defined(ARCH_X86_FAMILY)
#if defined(HAS_AMX_FP16)
    register_rank2_and_batched_fusion<
        ::vecops::matmul::AMX_F16F32, float16_t, float16_t, float32_t,
        InputPipeline::Convert, OutputPipeline::Clamp>(FloatKStep);
    register_rank2_and_batched_fusion<
        ::vecops::matmul::AMX_F16F32, float16_t, float16_t, float32_t,
        InputPipeline::Convert, OutputPipeline::DynamicScale>(FloatKStep);
    register_rank2_and_batched_fusion<
        ::vecops::matmul::AMX_F16F32, float16_t, float16_t, float32_t,
        InputPipeline::Convert, OutputPipeline::PerColumnScale>(FloatKStep);
#endif
#else
    register_rank2_and_batched_fusion<
        ::vecops::matmul::SME_F32F32, float32_t, float32_t, float32_t,
        InputPipeline::Convert, OutputPipeline::Clamp>(1);
    register_rank2_and_batched_fusion<
        ::vecops::matmul::SME_F32F32, float32_t, float32_t, float32_t,
        InputPipeline::Convert, OutputPipeline::DynamicScale>(1);
    register_rank2_and_batched_fusion<
        ::vecops::matmul::SME_F32F32, float32_t, float32_t, float32_t,
        InputPipeline::Convert, OutputPipeline::PerColumnScale>(1);
#endif
  } else if constexpr (Shard == 6) {
#if defined(ARCH_X86_FAMILY)
#if defined(HAS_AMX_FP16)
    register_rank2_and_batched_fusion<
        ::vecops::matmul::AMX_F16F32, float32_t, float32_t, float32_t,
        InputPipeline::Convert, OutputPipeline::Clamp>(FloatKStep);
    register_rank2_and_batched_fusion<
        ::vecops::matmul::AMX_F16F32, float32_t, float32_t, float32_t,
        InputPipeline::Convert, OutputPipeline::DynamicScale>(FloatKStep);
    register_rank2_and_batched_fusion<
        ::vecops::matmul::AMX_F16F32, float32_t, float32_t, float32_t,
        InputPipeline::Convert, OutputPipeline::PerColumnScale>(FloatKStep);
#endif
#else
    register_rank2_and_batched_fusion<
        ::vecops::matmul::SME_F16F32, float16_t, float16_t, float32_t,
        InputPipeline::Convert, OutputPipeline::Clamp>(FloatKStep);
    register_rank2_and_batched_fusion<
        ::vecops::matmul::SME_F16F32, float16_t, float16_t, float32_t,
        InputPipeline::Convert, OutputPipeline::DynamicScale>(FloatKStep);
    register_rank2_and_batched_fusion<
        ::vecops::matmul::SME_F16F32, float16_t, float16_t, float32_t,
        InputPipeline::Convert, OutputPipeline::PerColumnScale>(FloatKStep);
#endif
  } else if constexpr (Shard == 7) {
#if !defined(ARCH_X86_FAMILY) && defined(HAS_SME_F64F64)
    register_rank2_fusion<
        ::vecops::matmul::SME_F64F64, float64_t, float64_t, float64_t,
        InputPipeline::Convert, OutputPipeline::Clamp>(1);
#endif
  } else if constexpr (Shard == 8) {
#if !defined(ARCH_X86_FAMILY) && defined(HAS_SME_F64F64)
    register_batched_fusion<
        ::vecops::matmul::SME_F64F64, float32_t, float32_t, float32_t,
        InputPipeline::Convert, OutputPipeline::Clamp>(1);
#endif
  } else if constexpr (Shard == 9) {
#if !defined(ARCH_X86_FAMILY)
    register_rank2_and_batched_fusion<
        ::vecops::matmul::SME_F32F32, bfloat16_t, bfloat16_t, bfloat16_t,
        InputPipeline::Convert, OutputPipeline::BiasClamp>(1);
    register_rank2_and_batched_fusion<
        ::vecops::matmul::SME_F32F32, float16_t, float16_t, float16_t,
        InputPipeline::Convert, OutputPipeline::BiasClamp>(1);
#endif
  } else if constexpr (Shard == 10) {
    register_rank2_and_batched_fusion<
        FusionFloatAtom, bfloat16_t, bfloat16_t, bfloat16_t,
        InputPipeline::Convert, OutputPipeline::Bias,
        InputPipeline::Convert, float32_t>(FloatKStep);
    register_rank2_and_batched_fusion<
        FusionFloatAtom, bfloat16_t, bfloat16_t, bfloat16_t,
        InputPipeline::Convert, OutputPipeline::BiasRelu,
        InputPipeline::Convert, float32_t>(FloatKStep);
  } else if constexpr (Shard == 11) {
    register_rank2_and_batched_fusion<
        FusionFloatAtom, bfloat16_t, bfloat16_t, float32_t,
        InputPipeline::Convert, OutputPipeline::Relu>(FloatKStep);
    register_rank2_and_batched_fusion<
        FusionFloatAtom, bfloat16_t, bfloat16_t, float32_t,
        InputPipeline::Convert, OutputPipeline::Sigmoid>(FloatKStep);
  } else if constexpr (Shard == 12) {
#if defined(ARCH_X86_FAMILY)
#if defined(HAS_AMX_FP16)
    register_rank2_and_batched_fusion<
        ::vecops::matmul::AMX_F16F32, float16_t, float16_t, float16_t,
        InputPipeline::Convert, OutputPipeline::Bias,
        InputPipeline::Convert, float32_t>(FloatKStep);
    register_rank2_and_batched_fusion<
        ::vecops::matmul::AMX_F16F32, float16_t, float16_t, float16_t,
        InputPipeline::Convert, OutputPipeline::BiasRelu,
        InputPipeline::Convert, float32_t>(FloatKStep);
#endif
#else
    register_rank2_and_batched_fusion<
        ::vecops::matmul::SME_F16F32, float16_t, float16_t, float16_t,
        InputPipeline::Convert, OutputPipeline::Bias,
        InputPipeline::Convert, float32_t>(FloatKStep);
    register_rank2_and_batched_fusion<
        ::vecops::matmul::SME_F16F32, float16_t, float16_t, float16_t,
        InputPipeline::Convert, OutputPipeline::BiasRelu,
        InputPipeline::Convert, float32_t>(FloatKStep);
#endif
  } else if constexpr (Shard == 13) {
#if !defined(ARCH_X86_FAMILY)
    register_rank2_and_batched_fusion<
        ::vecops::matmul::SME_F32F32, bfloat16_t, bfloat16_t, bfloat16_t,
        InputPipeline::Convert, OutputPipeline::BiasRelu,
        InputPipeline::Convert, float32_t>(1);
    register_rank2_and_batched_fusion<
        ::vecops::matmul::SME_F32F32, float16_t, float16_t, float16_t,
        InputPipeline::Convert, OutputPipeline::BiasRelu,
        InputPipeline::Convert, float32_t>(1);
#endif
  } else if constexpr (Shard == 14) {
    register_rank2_and_batched_fusion<
        FusionI8S8S8, float32_t, int8_t, float32_t,
        InputPipeline::Quantize4, OutputPipeline::Dequantize,
        InputPipeline::Convert>(I8KStep);
  } else if constexpr (Shard == 15) {
    register_rank2_and_batched_fusion<
        FusionI8S8S8, int8_t, int8_t, int32_t,
        InputPipeline::Convert, OutputPipeline::Accumulate>(I8KStep);
    register_rank2_and_batched_fusion<
        FusionI8S8U8, int8_t, uint8_t, int32_t,
        InputPipeline::Convert, OutputPipeline::Accumulate>(I8KStep);
  } else if constexpr (Shard == 16) {
    register_rank2_and_batched_fusion<
        FusionI8U8S8, uint8_t, int8_t, int32_t,
        InputPipeline::Convert, OutputPipeline::Accumulate>(I8KStep);
    register_rank2_and_batched_fusion<
        FusionI8U8U8, uint8_t, uint8_t, int32_t,
        InputPipeline::Convert, OutputPipeline::Accumulate>(I8KStep);
  } else if constexpr (Shard == 17) {
    register_rank2_and_batched_fusion<
        FusionI8U8S8, float32_t, int8_t, float32_t,
        InputPipeline::AsymmetricQuantize4Zp3,
        OutputPipeline::AsymmetricDequantize,
        InputPipeline::Convert>(I8KStep);
  } else if constexpr (Shard == 18) {
    register_rank2_and_batched_fusion<
        FusionFloatAtom, bfloat16_t, bfloat16_t, float32_t,
        InputPipeline::Convert, OutputPipeline::Silu>(FloatKStep);
  } else if constexpr (Shard == 19) {
    register_rank2_and_batched_fusion<
        FusionFloatAtom, bfloat16_t, bfloat16_t, float32_t,
        InputPipeline::Convert, OutputPipeline::BiasSilu>(FloatKStep);
  } else if constexpr (Shard == 20) {
    register_rank2_and_batched_fusion<
        FusionI8S8S8, float32_t, int8_t, float32_t,
        InputPipeline::Quantize4,
        OutputPipeline::BiasPerColumnDequantize,
        InputPipeline::Convert, int32_t>(I8KStep);
  } else if constexpr (Shard == 21) {
    register_rank2_and_batched_fusion<
        FusionI8S8S8, float32_t, int8_t, uint8_t,
        InputPipeline::Quantize4,
        OutputPipeline::BiasPerColumnRequantizeU8,
        InputPipeline::Convert, int32_t>(I8KStep);
  } else if constexpr (Shard == 22) {
    register_rank2_and_batched_fusion<
        FusionI8U8S8, float32_t, int8_t, float32_t,
        InputPipeline::RuntimePerRowAsymmetricQuantize,
        OutputPipeline::RuntimePerRowColumnDequantize,
        InputPipeline::Convert>(I8KStep);
    register_batched_scenario_extent_pair<
        FusionI8U8S8, float32_t, int8_t, float32_t,
        InputPipeline::RuntimePerRowAsymmetricQuantize,
        OutputPipeline::RuntimePerRowColumnDequantize,
        false, true, InputMode::Raw, InputPipeline::Convert, float32_t,
        3, 5, 7, 2 * I8KStep + 1>(
            "MatmulFusion", "fusion_runtime_shared_activation_tail");
    register_batched_scenario_extent_pair<
        FusionI8U8S8, float32_t, int8_t, float32_t,
        InputPipeline::RuntimePerRowAsymmetricQuantize,
        OutputPipeline::RuntimePerRowColumnDequantize,
        true, false, InputMode::OnlinePackedB, InputPipeline::Convert,
        float32_t, 3, 5, 7, 2 * I8KStep + 1>(
            "MatmulFusion", "fusion_runtime_shared_weight_online");
  } else if constexpr (Shard == 23) {
#if !defined(ARCH_X86_FAMILY) && defined(HAS_SME_F64F64)
    register_batched_fusion<
        ::vecops::matmul::SME_F64F64, float64_t, float64_t, float64_t,
        InputPipeline::Convert, OutputPipeline::Clamp>(1);
#endif
  } else if constexpr (Shard == 24) {
#if !defined(ARCH_X86_FAMILY) && defined(HAS_SME_F64F64)
    register_rank2_fusion<
        ::vecops::matmul::SME_F64F64, float64_t, float64_t, float64_t,
        InputPipeline::Convert, OutputPipeline::DynamicScale>(1);
#endif
  } else if constexpr (Shard == 25) {
#if !defined(ARCH_X86_FAMILY) && defined(HAS_SME_F64F64)
    register_batched_fusion<
        ::vecops::matmul::SME_F64F64, float64_t, float64_t, float64_t,
        InputPipeline::Convert, OutputPipeline::DynamicScale>(1);
#endif
  } else if constexpr (Shard == 26) {
#if !defined(ARCH_X86_FAMILY) && defined(HAS_SME_F64F64)
    register_rank2_fusion<
        ::vecops::matmul::SME_F64F64, float64_t, float64_t, float64_t,
        InputPipeline::Convert, OutputPipeline::PerColumnScale>(1);
#endif
  } else if constexpr (Shard == 27) {
#if !defined(ARCH_X86_FAMILY) && defined(HAS_SME_F64F64)
    register_batched_fusion<
        ::vecops::matmul::SME_F64F64, float64_t, float64_t, float64_t,
        InputPipeline::Convert, OutputPipeline::PerColumnScale>(1);
#endif
  } else if constexpr (Shard == 28) {
#if !defined(ARCH_X86_FAMILY) && defined(HAS_SME_F64F64)
    register_rank2_fusion<
        ::vecops::matmul::SME_F64F64, float32_t, float32_t, float32_t,
        InputPipeline::Convert, OutputPipeline::Clamp>(1);
#endif
  } else if constexpr (Shard == 29) {
#if !defined(ARCH_X86_FAMILY) && defined(HAS_SME_F64F64)
    register_rank2_fusion<
        ::vecops::matmul::SME_F64F64, float32_t, float32_t, float32_t,
        InputPipeline::Convert, OutputPipeline::DynamicScale>(1);
#endif
  } else if constexpr (Shard == 30) {
#if !defined(ARCH_X86_FAMILY) && defined(HAS_SME_F64F64)
    register_batched_fusion<
        ::vecops::matmul::SME_F64F64, float32_t, float32_t, float32_t,
        InputPipeline::Convert, OutputPipeline::DynamicScale>(1);
#endif
  } else if constexpr (Shard == 31) {
#if !defined(ARCH_X86_FAMILY) && defined(HAS_SME_F64F64)
    register_rank2_fusion<
        ::vecops::matmul::SME_F64F64, float32_t, float32_t, float32_t,
        InputPipeline::Convert, OutputPipeline::PerColumnScale>(1);
#endif
  } else {
#if !defined(ARCH_X86_FAMILY) && defined(HAS_SME_F64F64)
    register_batched_fusion<
        ::vecops::matmul::SME_F64F64, float32_t, float32_t, float32_t,
        InputPipeline::Convert, OutputPipeline::PerColumnScale>(1);
#endif
  }
}

} // namespace vecops::bench::matmul

static_assert(
    VECOPS_TARGET_SHARD_COUNT ==
    vecops::bench::matmul::MatmulFusionScenarioShardCount);
template void vecops::bench::matmul::register_matmul_fusion_scenario_shard<
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
  vecops::bench::matmul::register_matmul_fusion_scenario_range<
      0, vecops::bench::matmul::MatmulFusionScenarioShardCount>();
  return vecops::bench::matmul::run_benchmarks(
      argc, argv, "matmul_fusion");
}

#endif
