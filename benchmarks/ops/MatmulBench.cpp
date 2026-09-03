// @vecops-target-shards-x86: 5
// @vecops-target-shards-ARM: 176

#include "vecops/Features.h"

#if defined(ARCH_X86_FAMILY)
#define VECOPS_MATMUL_CATALOG_CASE_SHARDS 1
#else
#define VECOPS_MATMUL_CATALOG_CASE_SHARDS 16
#endif

#include "MatmulBenchCommon.h"

#include <cerrno>
#include <cstring>
#include <iostream>
#include <utility>

#if defined(ARCH_X86_FAMILY)
#include <sys/syscall.h>
#include <unistd.h>
#endif

#include "vecops/matmul/Atom.h"

namespace vecops::bench::matmul {

struct MatmulBenchArchTraits {
#if defined(ARCH_X86_FAMILY)
  static constexpr int ShardCount = 5;
#else
  static constexpr int ShardCount = 176;
#endif

  static bool enable() {
#if defined(ARCH_X86_FAMILY)
    constexpr long ArchRequestXcompPerm = 0x1023;
    constexpr long XfeatureTileData = 18;
    return syscall(
        SYS_arch_prctl, ArchRequestXcompPerm, XfeatureTileData) == 0;
#else
    return true;
#endif
  }

  static const char* enable_error() {
#if defined(ARCH_X86_FAMILY)
    return "Unable to request Linux XTILEDATA permission";
#else
    return "Unable to initialize Matmul benchmark architecture";
#endif
  }
};

template <int Shard>
void register_matmul_bench_shard();

} // namespace vecops::bench::matmul

#if defined(VECOPS_TARGET_SHARD_ACTIVE)

namespace vecops::bench::matmul {

#if defined(ARCH_X86_FAMILY)

template <>
struct AtomName<::vecops::matmul::AMX_BF16F32> {
  static constexpr const char* value = "AMX_BF16F32";
};

#if defined(HAS_AMX_FP16)
template <>
struct AtomName<::vecops::matmul::AMX_F16F32> {
  static constexpr const char* value = "AMX_F16F32";
};
#endif

using NativeI8S8U8 = ::vecops::matmul::AMX_I8I32<int8_t, uint8_t>;

template <>
struct AtomName<NativeI8S8U8> {
  static constexpr const char* value = "AMX_I8I32_s8u8";
};

#else

template <>
struct AtomName<::vecops::matmul::SME_BF16F32> {
  static constexpr const char* value = "SME_BF16F32";
};

template <>
struct AtomName<::vecops::matmul::SME_F16F32> {
  static constexpr const char* value = "SME_F16F32";
};

template <>
struct AtomName<::vecops::matmul::SME_F32F32> {
  static constexpr const char* value = "SME_F32F32";
};

#if defined(HAS_SME_F64F64)
template <>
struct AtomName<::vecops::matmul::SME_F64F64> {
  static constexpr const char* value = "SME_F64F64";
};
#endif

using NativeI8S8U8 = ::vecops::matmul::SME_I8I32<int8_t, uint8_t>;
using NativeI8U8S8 = ::vecops::matmul::SME_I8I32<uint8_t, int8_t>;
using NativeI8S8S8 = ::vecops::matmul::SME_I8I32<int8_t, int8_t>;
using NativeI8U8U8 = ::vecops::matmul::SME_I8I32<uint8_t, uint8_t>;

template <>
struct AtomName<NativeI8S8U8> {
  static constexpr const char* value = "SME_I8I32_s8u8";
};

template <>
struct AtomName<NativeI8U8S8> {
  static constexpr const char* value = "SME_I8I32_u8s8";
};

template <>
struct AtomName<NativeI8S8S8> {
  static constexpr const char* value = "SME_I8I32_s8s8";
};

template <>
struct AtomName<NativeI8U8U8> {
  static constexpr const char* value = "SME_I8I32_u8u8";
};

inline std::vector<MatmulCase> extended_integer_skinny_probe_cases() {
  auto cases = integer_skinny_probe_cases();
  constexpr uint32_t Raw = mode_bit(InputMode::Raw);
  cases.push_back({
      "dispatch_integer_skinny", "m1_n8_k4097", 1, 8, 4097, Raw});
  cases.push_back({
      "dispatch_integer_skinny", "m8_n1_k4097", 8, 1, 4097, Raw});
  return cases;
}

#endif

template <int TargetShard>
void register_matmul_bench_shard() {
  static_assert(
      0 <= TargetShard && TargetShard < MatmulBenchArchTraits::ShardCount);
#if defined(ARCH_X86_FAMILY)
  if constexpr (TargetShard == 0) {
    register_representative_extent_pairs<::vecops::matmul::AMX_BF16F32, 16, 32, 3>();
  } else if constexpr (TargetShard == 1) {
    register_workload_extent_pairs<::vecops::matmul::AMX_BF16F32, 16, 32>();
  } else if constexpr (TargetShard == 2) {
    register_dispatch_extent_pairs<::vecops::matmul::AMX_BF16F32, 16>();
  } else if constexpr (TargetShard == 3) {
#if defined(HAS_AMX_FP16)
    register_dtype_probe_extent_pairs<::vecops::matmul::AMX_F16F32, 16, 32>();
#endif
  } else if constexpr (TargetShard == 4) {
#if defined(HAS_AMX_INT8)
    register_dtype_probe_extent_pairs<NativeI8S8U8, 16, 64>();
#endif
  }
#else
  constexpr int CatalogShard =
      TargetShard / VECOPS_MATMUL_CATALOG_CASE_SHARDS;
  if constexpr (CatalogShard == 0) {
    register_representative_extent_pairs<
        ::vecops::matmul::SME_BF16F32, 16, 2, 4, TargetShard>();
  } else if constexpr (CatalogShard == 1) {
    register_workload_extent_pairs<
        ::vecops::matmul::SME_BF16F32, 16, 2, TargetShard>();
  } else if constexpr (CatalogShard == 2) {
    register_dispatch_extent_pairs<::vecops::matmul::SME_BF16F32, 16, TargetShard>();
  } else if constexpr (CatalogShard == 3) {
    register_packed_dot_extent_pairs<
        ::vecops::matmul::SME_BF16F32, 0, TargetShard>();
  } else if constexpr (CatalogShard == 4) {
    register_dtype_probe_extent_pairs<
        ::vecops::matmul::SME_F16F32, 16, 2, 0, TargetShard>();
    register_skinny_extent_pairs<
        ::vecops::matmul::SME_F16F32, mode_bit(InputMode::Raw), 3, TargetShard>(
        "dispatch_float_skinny");
  } else if constexpr (CatalogShard == 5) {
    register_dtype_probe_extent_pairs<
        ::vecops::matmul::SME_F32F32, 16, 1, 0, TargetShard>();
    register_skinny_extent_pairs<
        ::vecops::matmul::SME_F32F32, mode_bit(InputMode::Raw), 3, TargetShard>(
        "dispatch_float_skinny");
  } else if constexpr (CatalogShard == 6) {
    register_extended_integer_skinny_extent_pairs<
        NativeI8S8S8, 0, TargetShard>();
    register_packed_dot_extent_pairs<NativeI8S8S8, 8, TargetShard>();
  } else if constexpr (CatalogShard == 7) {
    register_extended_integer_skinny_extent_pairs<
        NativeI8U8U8, 0, TargetShard>();
    register_packed_dot_extent_pairs<NativeI8U8U8, 8, TargetShard>();
  } else if constexpr (CatalogShard == 8) {
#if defined(HAS_SME_F64F64)
    register_dtype_probe_extent_pairs<
        ::vecops::matmul::SME_F64F64, 8, 1, 0, TargetShard>();
    register_skinny_extent_pairs<
        ::vecops::matmul::SME_F64F64, mode_bit(InputMode::Raw), 3, TargetShard>(
        "dispatch_float_skinny");
#endif
  } else if constexpr (CatalogShard == 9) {
    register_dtype_probe_extent_pairs<
        NativeI8S8U8, 16, 4, 0, TargetShard>();
    register_dtype_probe_extent_pairs<
        NativeI8U8S8, 16, 4, 0, TargetShard>();
  } else if constexpr (CatalogShard == 10) {
    register_extended_integer_skinny_extent_pairs<
        NativeI8S8U8, 0, TargetShard>();
    register_extended_integer_skinny_extent_pairs<
        NativeI8U8S8, 0, TargetShard>();
  }
#endif
}

} // namespace vecops::bench::matmul

static_assert(
    VECOPS_TARGET_SHARD_COUNT ==
    vecops::bench::matmul::MatmulBenchArchTraits::ShardCount);
template void vecops::bench::matmul::register_matmul_bench_shard<
    VECOPS_TARGET_SHARD_INDEX>();

#else

int main(int argc, char** argv) {
  using Traits = vecops::bench::matmul::MatmulBenchArchTraits;
  if (!Traits::enable()) {
    std::cerr << Traits::enable_error() << ": " << std::strerror(errno)
              << '\n';
    return 1;
  }
  []<std::size_t... I>(std::index_sequence<I...>) {
    (vecops::bench::matmul::register_matmul_bench_shard<
         static_cast<int>(I)>(), ...);
  }(std::make_index_sequence<Traits::ShardCount>{});
  return vecops::bench::matmul::run_benchmarks(argc, argv, "matmul");
}

#endif
