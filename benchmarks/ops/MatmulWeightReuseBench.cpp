// @vecops-target-shards: 260

#include <cerrno>
#include <cstring>
#include <iostream>
#include <sys/syscall.h>
#include <unistd.h>
#include <utility>

// Weight-reuse shards are already partitioned by pipeline, catalog part and
// dtype probe.  Parse the common registrar in combined-extent mode so a single
// TU owns both variants and no macro-dependent template body crosses a shard
// ODR boundary; restore the target-shard marker for this source's dispatcher.
#if defined(VECOPS_TARGET_SHARD_ACTIVE)
#define VECOPS_WEIGHT_REUSE_TARGET_SHARD_ACTIVE 1
#undef VECOPS_TARGET_SHARD_ACTIVE
#endif
#include "MatmulScenarioBenchCommon.h"
#if defined(VECOPS_WEIGHT_REUSE_TARGET_SHARD_ACTIVE)
#define VECOPS_TARGET_SHARD_ACTIVE 1
#undef VECOPS_WEIGHT_REUSE_TARGET_SHARD_ACTIVE
#endif

#include "vecops/Features.h"
#include "vecops/gemm/Atoms.h"

#ifndef VECOPS_WEIGHT_REUSE_RESULT_STEM
#define VECOPS_WEIGHT_REUSE_RESULT_STEM "matmul_weight_reuse"
#endif

namespace vecops::bench::matmul {

inline constexpr int MatmulWeightReuseShardCount = 260;

template <int Shard>
void register_matmul_weight_reuse_shard();

template <std::size_t Begin, std::size_t End>
void register_matmul_weight_reuse_range() {
  if constexpr (Begin + 1 == End) {
    register_matmul_weight_reuse_shard<static_cast<int>(Begin)>();
  } else if constexpr (Begin < End) {
    constexpr std::size_t Middle = Begin + (End - Begin) / 2;
    register_matmul_weight_reuse_range<Begin, Middle>();
    register_matmul_weight_reuse_range<Middle, End>();
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
using WeightReuseI8 = gemm::AMX_I8I32<A, B>;
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
using WeightReuseI8 = gemm::SME_I8I32<A, B>;
#endif

using WeightReuseI8S8S8 = WeightReuseI8<int8_t, int8_t>;
using WeightReuseI8S8U8 = WeightReuseI8<int8_t, uint8_t>;
using WeightReuseI8U8S8 = WeightReuseI8<uint8_t, int8_t>;
using WeightReuseI8U8U8 = WeightReuseI8<uint8_t, uint8_t>;

template <> struct AtomName<WeightReuseI8S8S8> {
  static constexpr const char* value = "I8I32_s8s8";
};
template <> struct AtomName<WeightReuseI8S8U8> {
  static constexpr const char* value = "I8I32_s8u8";
};
template <> struct AtomName<WeightReuseI8U8S8> {
  static constexpr const char* value = "I8I32_u8s8";
};
template <> struct AtomName<WeightReuseI8U8U8> {
  static constexpr const char* value = "I8I32_u8u8";
};

struct WeightReuseDTypeProbeCatalog {};

#define weight_reuse_dtype_probe_cases() WeightReuseDTypeProbeCatalog{}

template <typename Atom, typename MemoryA, typename MemoryB, typename MemoryC,
          InputPipeline InPipeline, OutputPipeline OutPipeline,
          InputPipeline InPipelineB, typename MemoryCInput,
          nint_t Batch, nint_t M, nint_t N, nint_t K, nint_t Calls>
void register_weight_reuse_dtype_case(const char* name) {
  register_weight_reuse_extent_pair<
      Atom, MemoryA, MemoryB, MemoryC, InPipeline, OutPipeline,
      InputMode::Raw, InPipelineB, MemoryCInput,
      Batch, M, N, K, Calls>(name);
  register_weight_reuse_extent_pair<
      Atom, MemoryA, MemoryB, MemoryC, InPipeline, OutPipeline,
      InputMode::OnlinePackedB, InPipelineB, MemoryCInput,
      Batch, M, N, K, Calls>(name);
}

template <typename Atom, typename MemoryA, typename MemoryB, typename MemoryC,
          InputPipeline InPipeline = InputPipeline::Convert,
          OutputPipeline OutPipeline = OutputPipeline::Convert,
          InputPipeline InPipelineB = InPipeline,
          typename MemoryCInput = MemoryC, int Part>
void register_all_batched_weight_reuse_modes(WeightReuseDTypeProbeCatalog) {
  static_assert(0 <= Part && Part < 8);
#define VECOPS_REGISTER_DTYPE_REUSE(NAME, B, M, N, K, CALLS) \
  register_weight_reuse_dtype_case< \
      Atom, MemoryA, MemoryB, MemoryC, InPipeline, OutPipeline, \
      InPipelineB, MemoryCInput, B, M, N, K, CALLS>(NAME)
  if constexpr (Part == 0) {
    VECOPS_REGISTER_DTYPE_REUSE("decode", 1, 1, 256, 256, 1);
  } else if constexpr (Part == 1) {
    VECOPS_REGISTER_DTYPE_REUSE("decode", 1, 1, 256, 256, 16);
  } else if constexpr (Part == 2) {
    VECOPS_REGISTER_DTYPE_REUSE("decode", 8, 1, 256, 256, 1);
  } else if constexpr (Part == 3) {
    VECOPS_REGISTER_DTYPE_REUSE("decode", 8, 1, 256, 256, 16);
  } else if constexpr (Part == 4) {
    VECOPS_REGISTER_DTYPE_REUSE("small", 8, 1, 64, 128, 1);
  } else if constexpr (Part == 5) {
    VECOPS_REGISTER_DTYPE_REUSE("small", 8, 1, 64, 128, 16);
  } else if constexpr (Part == 6) {
    VECOPS_REGISTER_DTYPE_REUSE("mlp_projection", 8, 1, 1024, 4096, 4);
  } else {
    VECOPS_REGISTER_DTYPE_REUSE("qwen_o_proj", 8, 1, 3584, 3584, 2);
  }
#undef VECOPS_REGISTER_DTYPE_REUSE
}

template <int Part, int Slice, int Mode>
struct WeightReusePartition {};

template <typename Atom, typename MemoryA, typename MemoryB, typename MemoryC,
          InputPipeline InPipeline, OutputPipeline OutPipeline,
          InputMode Mode, InputPipeline InPipelineB, typename MemoryCInput,
          int Part, int Slice>
void register_weight_reuse_partition_mode() {
  static_assert(0 <= Part && Part < 5);
  static_assert(0 <= Slice && Slice < 6);
  constexpr std::array<nint_t, 5> BatchSizes{1, 2, 4, 8, 16};
  constexpr nint_t Batch = BatchSizes[Part];
  if constexpr (Slice == 0) {
    register_weight_reuse_counts<
        Atom, MemoryA, MemoryB, MemoryC, InPipeline, OutPipeline, Mode,
        InPipelineB, MemoryCInput, Batch, 1, 256, 256>(
            "decode", std::integer_sequence<nint_t, 1, 2>{});
  } else if constexpr (Slice == 1) {
    register_weight_reuse_counts<
        Atom, MemoryA, MemoryB, MemoryC, InPipeline, OutPipeline, Mode,
        InPipelineB, MemoryCInput, Batch, 1, 256, 256>(
            "decode", std::integer_sequence<nint_t, 4, 8>{});
  } else if constexpr (Slice == 2) {
    register_weight_reuse_counts<
        Atom, MemoryA, MemoryB, MemoryC, InPipeline, OutPipeline, Mode,
        InPipelineB, MemoryCInput, Batch, 1, 256, 256>(
            "decode", std::integer_sequence<nint_t, 16>{});
    register_weight_reuse_counts<
        Atom, MemoryA, MemoryB, MemoryC, InPipeline, OutPipeline, Mode,
        InPipelineB, MemoryCInput, Batch, 1, 64, 128>(
            "small", std::integer_sequence<nint_t, 1>{});
  } else if constexpr (Slice == 3) {
    register_weight_reuse_counts<
        Atom, MemoryA, MemoryB, MemoryC, InPipeline, OutPipeline, Mode,
        InPipelineB, MemoryCInput, Batch, 1, 64, 128>(
            "small", std::integer_sequence<nint_t, 4, 16>{});
  } else if constexpr (Slice == 4 && Part == 3) {
    register_weight_reuse_counts<
        Atom, MemoryA, MemoryB, MemoryC, InPipeline, OutPipeline, Mode,
        InPipelineB, MemoryCInput, 8, 1, 1024, 4096>(
            "mlp_projection", std::integer_sequence<nint_t, 1, 2>{});
  } else if constexpr (Slice == 5 && Part == 3) {
    register_weight_reuse_counts<
        Atom, MemoryA, MemoryB, MemoryC, InPipeline, OutPipeline, Mode,
        InPipelineB, MemoryCInput, 8, 1, 1024, 4096>(
            "mlp_projection", std::integer_sequence<nint_t, 4, 8>{});
  } else if constexpr (Slice == 4 && Part == 4) {
    register_weight_reuse_counts<
        Atom, MemoryA, MemoryB, MemoryC, InPipeline, OutPipeline, Mode,
        InPipelineB, MemoryCInput, 8, 1, 3584, 3584>(
            "qwen_o_proj", std::integer_sequence<nint_t, 1, 2>{});
  } else if constexpr (Slice == 5 && Part == 4) {
    register_weight_reuse_counts<
        Atom, MemoryA, MemoryB, MemoryC, InPipeline, OutPipeline, Mode,
        InPipelineB, MemoryCInput, 8, 1, 3584, 3584>(
            "qwen_o_proj", std::integer_sequence<nint_t, 4>{});
  }
}

template <typename Atom, typename MemoryA, typename MemoryB, typename MemoryC,
          InputPipeline InPipeline = InputPipeline::Convert,
          OutputPipeline OutPipeline = OutputPipeline::Convert,
          InputPipeline InPipelineB = InPipeline,
          typename MemoryCInput = MemoryC, int Part, int Slice, int Mode>
void register_all_batched_weight_reuse_modes(
    WeightReusePartition<Part, Slice, Mode>) {
  if constexpr (Mode == 0) {
    register_weight_reuse_partition_mode<
        Atom, MemoryA, MemoryB, MemoryC, InPipeline, OutPipeline,
        InputMode::Raw, InPipelineB, MemoryCInput, Part, Slice>();
  } else {
    register_weight_reuse_partition_mode<
        Atom, MemoryA, MemoryB, MemoryC, InPipeline, OutPipeline,
        InputMode::OnlinePackedB, InPipelineB, MemoryCInput, Part, Slice>();
  }
}

template <int TargetShard>
void register_matmul_weight_reuse_shard() {
  static_assert(0 <= TargetShard && TargetShard < MatmulWeightReuseShardCount);
  constexpr int Shard = TargetShard;
  constexpr int LifecyclePipeline = Shard < 180 ? Shard / 60 : -1;
  constexpr int LifecyclePart = Shard < 180 ? (Shard % 60) / 12 : -1;
  constexpr int LifecycleSlice = Shard < 180 ? (Shard % 12) / 2 : -1;
  constexpr int LifecycleMode = Shard < 180 ? Shard % 2 : -1;
  constexpr int DTypeShard = Shard >= 180 ? (Shard - 180) / 8 : -1;
  constexpr int DTypePart = Shard >= 180 ? (Shard - 180) % 8 : -1;
#if defined(VECOPS_WEIGHT_REUSE_PIPELINE)
  static_assert(
      0 <= VECOPS_WEIGHT_REUSE_PIPELINE &&
      VECOPS_WEIGHT_REUSE_PIPELINE < 3);
#endif
#if !defined(VECOPS_WEIGHT_REUSE_PIPELINE) || \
    VECOPS_WEIGHT_REUSE_PIPELINE == 0
  if constexpr (LifecyclePipeline == 0) {
    register_all_batched_weight_reuse_modes<
#if defined(ARCH_X86_FAMILY)
        gemm::AMX_BF16F32,
#else
        gemm::SME_BF16F32,
#endif
        bfloat16_t, bfloat16_t, float32_t,
        InputPipeline::Convert, OutputPipeline::BiasRelu>(
            WeightReusePartition<
                LifecyclePart, LifecycleSlice, LifecycleMode>{});
  } else
#endif
#if !defined(VECOPS_WEIGHT_REUSE_PIPELINE) || \
    VECOPS_WEIGHT_REUSE_PIPELINE == 1
  if constexpr (LifecyclePipeline == 1) {
    register_all_batched_weight_reuse_modes<
#if defined(ARCH_X86_FAMILY)
        gemm::AMX_BF16F32,
#else
        gemm::SME_BF16F32,
#endif
        float32_t, float32_t, float32_t,
        InputPipeline::Convert, OutputPipeline::Relu>(
            WeightReusePartition<
                LifecyclePart, LifecycleSlice, LifecycleMode>{});
  } else
#endif
#if !defined(VECOPS_WEIGHT_REUSE_PIPELINE) || \
    VECOPS_WEIGHT_REUSE_PIPELINE == 2
  if constexpr (LifecyclePipeline == 2) {
    register_all_batched_weight_reuse_modes<
        WeightReuseI8S8U8, float32_t, float32_t, float32_t,
        InputPipeline::Quantize4, OutputPipeline::Dequantize>(
            WeightReusePartition<
                LifecyclePart, LifecycleSlice, LifecycleMode>{});
  } else
#endif
#if !defined(VECOPS_WEIGHT_REUSE_PIPELINE)
  if constexpr (DTypeShard == 0) {
    register_all_batched_weight_reuse_modes<
#if defined(ARCH_X86_FAMILY)
        gemm::AMX_BF16F32,
#else
        gemm::SME_BF16F32,
#endif
        bfloat16_t, bfloat16_t, bfloat16_t,
        InputPipeline::Convert, OutputPipeline::BiasRelu,
        InputPipeline::Convert, float32_t, DTypePart>(
            weight_reuse_dtype_probe_cases());
  } else if constexpr (DTypeShard == 1) {
#if defined(ARCH_X86_FAMILY)
#if defined(HAS_AMX_FP16)
    register_all_batched_weight_reuse_modes<
        gemm::AMX_F16F32, float16_t, float16_t, float16_t,
        InputPipeline::Convert, OutputPipeline::BiasRelu,
        InputPipeline::Convert, float32_t, DTypePart>(
            weight_reuse_dtype_probe_cases());
#endif
#else
    register_all_batched_weight_reuse_modes<
        gemm::SME_F16F32, float16_t, float16_t, float16_t,
        InputPipeline::Convert, OutputPipeline::BiasRelu,
        InputPipeline::Convert, float32_t, DTypePart>(
            weight_reuse_dtype_probe_cases());
#endif
  } else if constexpr (DTypeShard == 2) {
#if !defined(ARCH_X86_FAMILY)
    register_all_batched_weight_reuse_modes<
        gemm::SME_F32F32, float32_t, float32_t, float32_t,
        InputPipeline::Convert, OutputPipeline::BiasRelu,
        InputPipeline::Convert, float32_t, DTypePart>(
            weight_reuse_dtype_probe_cases());
#endif
  } else if constexpr (DTypeShard == 3) {
#if !defined(ARCH_X86_FAMILY) && defined(HAS_SME_F64F64)
    register_all_batched_weight_reuse_modes<
        gemm::SME_F64F64, float64_t, float64_t, float64_t,
        InputPipeline::Convert, OutputPipeline::BiasRelu,
        InputPipeline::Convert, float64_t, DTypePart>(
            weight_reuse_dtype_probe_cases());
#endif
  } else if constexpr (DTypeShard == 4) {
    register_all_batched_weight_reuse_modes<
        WeightReuseI8S8S8, int8_t, int8_t, int32_t,
        InputPipeline::Convert, OutputPipeline::Convert,
        InputPipeline::Convert, int32_t, DTypePart>(
            weight_reuse_dtype_probe_cases());
  } else if constexpr (DTypeShard == 5) {
    register_all_batched_weight_reuse_modes<
        WeightReuseI8S8U8, int8_t, uint8_t, int32_t,
        InputPipeline::Convert, OutputPipeline::Convert,
        InputPipeline::Convert, int32_t, DTypePart>(
            weight_reuse_dtype_probe_cases());
  } else if constexpr (DTypeShard == 6) {
    register_all_batched_weight_reuse_modes<
        WeightReuseI8U8S8, uint8_t, int8_t, int32_t,
        InputPipeline::Convert, OutputPipeline::Convert,
        InputPipeline::Convert, int32_t, DTypePart>(
            weight_reuse_dtype_probe_cases());
  } else if constexpr (DTypeShard == 7) {
    register_all_batched_weight_reuse_modes<
        WeightReuseI8U8U8, uint8_t, uint8_t, int32_t,
        InputPipeline::Convert, OutputPipeline::Convert,
        InputPipeline::Convert, int32_t, DTypePart>(
            weight_reuse_dtype_probe_cases());
  } else if constexpr (DTypeShard == 8) {
    register_all_batched_weight_reuse_modes<
        WeightReuseI8S8S8, float32_t, int8_t, float32_t,
        InputPipeline::Quantize4, OutputPipeline::Dequantize,
        InputPipeline::Convert, float32_t, DTypePart>(
            weight_reuse_dtype_probe_cases());
  } else if constexpr (DTypeShard == 9) {
    register_all_batched_weight_reuse_modes<
        WeightReuseI8U8S8, float32_t, int8_t, float32_t,
        InputPipeline::AsymmetricQuantize4Zp3,
        OutputPipeline::AsymmetricDequantize,
        InputPipeline::Convert, float32_t, DTypePart>(
            weight_reuse_dtype_probe_cases());
  } else
#endif
  {
  }
}

} // namespace vecops::bench::matmul

static_assert(
    VECOPS_TARGET_SHARD_COUNT ==
    vecops::bench::matmul::MatmulWeightReuseShardCount);
template void vecops::bench::matmul::register_matmul_weight_reuse_shard<
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
  vecops::bench::matmul::register_matmul_weight_reuse_range<
      0, vecops::bench::matmul::MatmulWeightReuseShardCount>();
  return vecops::bench::matmul::run_benchmarks(
      argc, argv, VECOPS_WEIGHT_REUSE_RESULT_STEM);
}

#endif
