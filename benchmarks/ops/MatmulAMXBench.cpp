#include <cerrno>
#include <cstring>
#include <iostream>
#include <sys/syscall.h>
#include <unistd.h>

#include "MatmulBenchCommon.h"

#include "vecops/Features.h"
#include "vecops/gemm/Atoms.h"

namespace vecops::bench::matmul {

template <>
struct AtomName<gemm::AMX_BF16F32> {
  static constexpr const char* value = "AMX_BF16F32";
};

#if defined(HAS_AMX_FP16)
template <>
struct AtomName<gemm::AMX_F16F32> {
  static constexpr const char* value = "AMX_F16F32";
};
#endif

using AMXI8 = gemm::AMX_I8I32<int8_t, uint8_t>;

template <>
struct AtomName<AMXI8> {
  static constexpr const char* value = "AMX_I8I32_s8u8";
};

bool enable_amx() {
  constexpr long ArchRequestXcompPerm = 0x1023;
  constexpr long XfeatureTileData = 18;
  return syscall(
      SYS_arch_prctl, ArchRequestXcompPerm, XfeatureTileData) == 0;
}

} // namespace vecops::bench::matmul

int main(int argc, char** argv) {
  using namespace vecops::bench::matmul;
  if (!enable_amx()) {
    std::cerr << "Unable to request Linux XTILEDATA permission: "
              << std::strerror(errno) << '\n';
    return 1;
  }

  constexpr vecops::nint_t tile = 16;
  register_cases<vecops::gemm::AMX_BF16F32>(
      representative_cases(tile, 32, 3));
#if defined(HAS_AMX_FP16)
  register_probe_cases<vecops::gemm::AMX_F16F32>(
      dtype_probe_cases(tile, 32));
#endif
#if defined(HAS_AMX_INT8)
  register_probe_cases<AMXI8>(dtype_probe_cases(tile, 64));
#endif
  return run_benchmarks(argc, argv, "matmul_amx");
}
