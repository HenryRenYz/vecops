// @vecops-target-shards: 480

#include <cerrno>
#include <cstring>
#include <iostream>
#include <sys/syscall.h>
#include <unistd.h>
#include <utility>

#define VECOPS_SCENARIO_CATALOG_CASE_SHARDS 12
#include "MatmulScenarioBenchCommon.h"

#include "vecops/Features.h"
#include "vecops/gemm/Atoms.h"

namespace vecops::bench::matmul {

inline constexpr int MatmulPackedScenarioShardCount = 480;

template <int Shard>
void register_matmul_packed_scenario_shard();

template <std::size_t Begin, std::size_t End>
void register_matmul_packed_scenario_range() {
  if constexpr (Begin + 1 == End) {
    register_matmul_packed_scenario_shard<static_cast<int>(Begin)>();
  } else if constexpr (Begin < End) {
    constexpr std::size_t Middle = Begin + (End - Begin) / 2;
    register_matmul_packed_scenario_range<Begin, Middle>();
    register_matmul_packed_scenario_range<Middle, End>();
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
using PackedScenarioI8 = gemm::AMX_I8I32<A, B>;
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
using PackedScenarioI8 = gemm::SME_I8I32<A, B>;
#endif

using PackedI8S8S8 = PackedScenarioI8<int8_t, int8_t>;
using PackedI8S8U8 = PackedScenarioI8<int8_t, uint8_t>;
using PackedI8U8S8 = PackedScenarioI8<uint8_t, int8_t>;
using PackedI8U8U8 = PackedScenarioI8<uint8_t, uint8_t>;

template <> struct AtomName<PackedI8S8S8> {
  static constexpr const char* value = "I8I32_s8s8";
};
template <> struct AtomName<PackedI8S8U8> {
  static constexpr const char* value = "I8I32_s8u8";
};
template <> struct AtomName<PackedI8U8S8> {
  static constexpr const char* value = "I8I32_u8s8";
};
template <> struct AtomName<PackedI8U8U8> {
  static constexpr const char* value = "I8I32_u8u8";
};

template <nint_t KStep>
struct PackedDTypeCaseCatalog {};

#define packed_dtype_probe_cases(KStep) PackedDTypeCaseCatalog<(KStep)>{}

template <typename Atom, typename MemoryA, typename MemoryB, typename MemoryC,
          InputPipeline InPipeline, OutputPipeline OutPipeline,
          InputMode Mode, InputPipeline InPipelineB, typename MemoryCInput,
          nint_t KStep, int CaseBase>
void register_packed_dtype_mode() {
  register_scenario_extent_pair<
      Atom, MemoryA, MemoryB, MemoryC, InPipeline, OutPipeline, Mode,
      InPipelineB, MemoryCInput, 19, 23, 8 * KStep + 1, CaseBase + 0>(
          "MatmulPacked", "tail");
  register_scenario_extent_pair<
      Atom, MemoryA, MemoryB, MemoryC, InPipeline, OutPipeline, Mode,
      InPipelineB, MemoryCInput, 1, 257, 4 * KStep + 1, CaseBase + 1>(
          "MatmulPacked", "gemv_tail");
}

template <typename Atom, typename MemoryA, typename MemoryB, typename MemoryC,
          InputPipeline InPipeline = InputPipeline::Convert,
          OutputPipeline OutPipeline = OutputPipeline::Convert,
          InputPipeline InPipelineB = InPipeline,
          typename MemoryCInput = MemoryC, nint_t KStep>
void register_all_packing_modes(PackedDTypeCaseCatalog<KStep>) {
  register_packed_dtype_mode<
      Atom, MemoryA, MemoryB, MemoryC, InPipeline, OutPipeline,
      InputMode::Raw, InPipelineB, MemoryCInput, KStep, 0>();
  register_packed_dtype_mode<
      Atom, MemoryA, MemoryB, MemoryC, InPipeline, OutPipeline,
      InputMode::PackedA, InPipelineB, MemoryCInput, KStep, 2>();
  register_packed_dtype_mode<
      Atom, MemoryA, MemoryB, MemoryC, InPipeline, OutPipeline,
      InputMode::PackedB, InPipelineB, MemoryCInput, KStep, 4>();
  register_packed_dtype_mode<
      Atom, MemoryA, MemoryB, MemoryC, InPipeline, OutPipeline,
      InputMode::PackedAB, InPipelineB, MemoryCInput, KStep, 6>();
  register_packed_dtype_mode<
      Atom, MemoryA, MemoryB, MemoryC, InPipeline, OutPipeline,
      InputMode::OnlinePackedA, InPipelineB, MemoryCInput, KStep, 8>();
  register_packed_dtype_mode<
      Atom, MemoryA, MemoryB, MemoryC, InPipeline, OutPipeline,
      InputMode::OnlinePackedB, InPipelineB, MemoryCInput, KStep, 10>();
  register_packed_dtype_mode<
      Atom, MemoryA, MemoryB, MemoryC, InPipeline, OutPipeline,
      InputMode::OnlinePackedAB, InPipelineB, MemoryCInput, KStep, 12>();
}

template <int TargetShard>
void register_matmul_packed_scenario_shard() {
  static_assert(
      0 <= TargetShard && TargetShard < MatmulPackedScenarioShardCount);
  constexpr int Shard = TargetShard /
      (2 * VECOPS_SCENARIO_CATALOG_CASE_SHARDS);
#if defined(ARCH_X86_FAMILY)
  constexpr nint_t BF16KStep = 32;
  constexpr nint_t I8KStep = 64;
#else
  constexpr nint_t BF16KStep = 2;
  constexpr nint_t I8KStep = 4;
#endif
  if constexpr (Shard == 0) {
    register_all_packing_modes<
#if defined(ARCH_X86_FAMILY)
        gemm::AMX_BF16F32,
#else
        gemm::SME_BF16F32,
#endif
        bfloat16_t, bfloat16_t, float32_t,
        InputPipeline::Convert, OutputPipeline::BiasRelu>(
            scenario_cases(BF16KStep));
  } else if constexpr (Shard == 1) {
    register_all_packing_modes<
#if defined(ARCH_X86_FAMILY)
        gemm::AMX_BF16F32,
#else
        gemm::SME_BF16F32,
#endif
        float32_t, float32_t, float32_t,
        InputPipeline::Convert, OutputPipeline::Relu>(
            scenario_cases(BF16KStep));
  } else if constexpr (Shard == 2) {
#if defined(ARCH_X86_FAMILY)
    register_all_packing_modes<
        gemm::AMX_BF16F32,
        bfloat16_t, bfloat16_t, bfloat16_t,
        InputPipeline::Convert, OutputPipeline::BiasRelu>(
            scenario_cases(BF16KStep));
#if defined(HAS_AMX_FP16)
    register_all_packing_modes<
        gemm::AMX_F16F32,
        float16_t, float16_t, float16_t,
        InputPipeline::Convert, OutputPipeline::BiasRelu>(
            scenario_cases(BF16KStep));
#endif
#else
    register_all_packing_modes<
        gemm::SME_F32F32,
        bfloat16_t, bfloat16_t, bfloat16_t,
        InputPipeline::Convert, OutputPipeline::BiasRelu>(
            scenario_cases(1));
    register_all_packing_modes<
        gemm::SME_F32F32,
        float16_t, float16_t, float16_t,
        InputPipeline::Convert, OutputPipeline::BiasRelu>(
            scenario_cases(1));
#endif
  } else if constexpr (Shard == 3) {
    register_all_packing_modes<
        PackedI8S8U8, float32_t, float32_t, float32_t,
        InputPipeline::Quantize4, OutputPipeline::Dequantize>(
            scenario_cases(I8KStep));
  } else if constexpr (Shard == 4) {
    register_all_packing_modes<
        PackedI8S8U8, float32_t, float32_t, int8_t,
        InputPipeline::Quantize4, OutputPipeline::ReluRequantize>(
            scenario_cases(I8KStep));
  } else if constexpr (Shard == 5) {
    register_all_packing_modes<
        PackedI8U8S8, float32_t, float32_t, float32_t,
        InputPipeline::AsymmetricQuantize4Zp3,
        OutputPipeline::AsymmetricDequantize>(scenario_cases(I8KStep));
    register_all_packing_modes<
        PackedI8U8S8, uint8_t, int8_t, float32_t,
        InputPipeline::PrequantizedAsymmetricZp3,
        OutputPipeline::AsymmetricDequantize>(scenario_cases(I8KStep));
  } else if constexpr (Shard == 6) {
    register_all_batched_packed_b_modes<
#if defined(ARCH_X86_FAMILY)
        gemm::AMX_BF16F32,
#else
        gemm::SME_BF16F32,
#endif
        bfloat16_t, bfloat16_t, float32_t,
        InputPipeline::Convert, OutputPipeline::BiasRelu>(
            batched_scenario_cases(BF16KStep));
  } else if constexpr (Shard == 7) {
    register_all_batched_packed_b_modes<
#if defined(ARCH_X86_FAMILY)
        gemm::AMX_BF16F32,
#else
        gemm::SME_BF16F32,
#endif
        float32_t, float32_t, float32_t,
        InputPipeline::Convert, OutputPipeline::Relu>(
            batched_scenario_cases(BF16KStep));
  } else if constexpr (Shard == 8) {
    register_all_batched_packed_b_modes<
        PackedI8S8U8, float32_t, float32_t, float32_t,
        InputPipeline::Quantize4, OutputPipeline::Dequantize>(
            batched_scenario_cases(I8KStep));
  } else if constexpr (Shard == 9) {
    register_all_packing_modes<
#if defined(ARCH_X86_FAMILY)
        gemm::AMX_BF16F32,
#else
        gemm::SME_BF16F32,
#endif
        bfloat16_t, bfloat16_t, bfloat16_t,
        InputPipeline::Convert, OutputPipeline::BiasRelu,
        InputPipeline::Convert, float32_t>(
            packed_dtype_probe_cases(BF16KStep));
  } else if constexpr (Shard == 10) {
#if defined(ARCH_X86_FAMILY)
#if defined(HAS_AMX_FP16)
    register_all_packing_modes<
        gemm::AMX_F16F32, float16_t, float16_t, float32_t,
        InputPipeline::Convert, OutputPipeline::BiasRelu>(
            packed_dtype_probe_cases(BF16KStep));
#endif
#else
    register_all_packing_modes<
        gemm::SME_F16F32, float16_t, float16_t, float32_t,
        InputPipeline::Convert, OutputPipeline::BiasRelu>(
            packed_dtype_probe_cases(BF16KStep));
#endif
  } else if constexpr (Shard == 11) {
#if defined(ARCH_X86_FAMILY)
#if defined(HAS_AMX_FP16)
    register_all_packing_modes<
        gemm::AMX_F16F32, float16_t, float16_t, float16_t,
        InputPipeline::Convert, OutputPipeline::BiasRelu,
        InputPipeline::Convert, float32_t>(
            packed_dtype_probe_cases(BF16KStep));
#endif
#else
    register_all_packing_modes<
        gemm::SME_F16F32, float16_t, float16_t, float16_t,
        InputPipeline::Convert, OutputPipeline::BiasRelu,
        InputPipeline::Convert, float32_t>(
            packed_dtype_probe_cases(BF16KStep));
#endif
  } else if constexpr (Shard == 12) {
#if !defined(ARCH_X86_FAMILY)
    register_all_packing_modes<
        gemm::SME_F32F32, float32_t, float32_t, float32_t,
        InputPipeline::Convert, OutputPipeline::BiasRelu>(
            packed_dtype_probe_cases(1));
#endif
  } else if constexpr (Shard == 13) {
#if !defined(ARCH_X86_FAMILY) && defined(HAS_SME_F64F64)
    register_all_packing_modes<
        gemm::SME_F64F64, float64_t, float64_t, float64_t,
        InputPipeline::Convert, OutputPipeline::BiasRelu>(
            packed_dtype_probe_cases(1));
#endif
  } else if constexpr (Shard == 14) {
    register_all_packing_modes<
        PackedI8S8S8, int8_t, int8_t, int32_t,
        InputPipeline::Convert, OutputPipeline::Convert>(
            packed_dtype_probe_cases(I8KStep));
  } else if constexpr (Shard == 15) {
    register_all_packing_modes<
        PackedI8S8U8, int8_t, uint8_t, int32_t,
        InputPipeline::Convert, OutputPipeline::Convert>(
            packed_dtype_probe_cases(I8KStep));
  } else if constexpr (Shard == 16) {
    register_all_packing_modes<
        PackedI8U8S8, uint8_t, int8_t, int32_t,
        InputPipeline::Convert, OutputPipeline::Convert>(
            packed_dtype_probe_cases(I8KStep));
  } else if constexpr (Shard == 17) {
    register_all_packing_modes<
        PackedI8U8U8, uint8_t, uint8_t, int32_t,
        InputPipeline::Convert, OutputPipeline::Convert>(
            packed_dtype_probe_cases(I8KStep));
  } else if constexpr (Shard == 18) {
    register_all_packing_modes<
        PackedI8S8S8, float32_t, int8_t, float32_t,
        InputPipeline::Quantize4, OutputPipeline::Dequantize,
        InputPipeline::Convert>(packed_dtype_probe_cases(I8KStep));
  } else {
    register_all_packing_modes<
        PackedI8U8S8, float32_t, int8_t, float32_t,
        InputPipeline::AsymmetricQuantize4Zp3,
        OutputPipeline::AsymmetricDequantize,
        InputPipeline::Convert>(packed_dtype_probe_cases(I8KStep));
  }
}

} // namespace vecops::bench::matmul

static_assert(
    VECOPS_TARGET_SHARD_COUNT ==
    vecops::bench::matmul::MatmulPackedScenarioShardCount);
template void vecops::bench::matmul::register_matmul_packed_scenario_shard<
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
  vecops::bench::matmul::register_matmul_packed_scenario_range<
      0, vecops::bench::matmul::MatmulPackedScenarioShardCount>();
  return vecops::bench::matmul::run_benchmarks(
      argc, argv, "matmul_packed");
}

#endif
