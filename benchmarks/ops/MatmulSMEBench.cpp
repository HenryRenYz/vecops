#include <arm_sme.h>

#include "MatmulBenchCommon.h"

#include "vecops/Features.h"
#include "vecops/gemm/Atoms.h"

namespace vecops::bench::matmul {

template <>
struct AtomName<gemm::SME_BF16F32> {
  static constexpr const char* value = "SME_BF16F32";
};

template <>
struct AtomName<gemm::SME_F16F32> {
  static constexpr const char* value = "SME_F16F32";
};

template <>
struct AtomName<gemm::SME_F32F32> {
  static constexpr const char* value = "SME_F32F32";
};

#if defined(HAS_SME_F64F64)
template <>
struct AtomName<gemm::SME_F64F64> {
  static constexpr const char* value = "SME_F64F64";
};
#endif

using SMEI8 = gemm::SME_I8I32<int8_t, uint8_t>;

template <>
struct AtomName<SMEI8> {
  static constexpr const char* value = "SME_I8I32_s8u8";
};

} // namespace vecops::bench::matmul

int main(int argc, char** argv) {
  using namespace vecops::bench::matmul;
  const vecops::nint_t tile = static_cast<vecops::nint_t>(svcntsw());
  register_cases<vecops::gemm::SME_BF16F32>(
      representative_cases(tile, 2, 4));
  register_probe_cases<vecops::gemm::SME_F16F32>(
      dtype_probe_cases(tile, 2));
  register_probe_cases<vecops::gemm::SME_F32F32>(
      dtype_probe_cases(tile, 1));
  register_probe_cases<SMEI8>(dtype_probe_cases(tile, 4));
#if defined(HAS_SME_F64F64)
  const vecops::nint_t tile64 = static_cast<vecops::nint_t>(svcntsd());
  register_probe_cases<vecops::gemm::SME_F64F64>(
      dtype_probe_cases(tile64, 1));
#endif
  return run_benchmarks(argc, argv, "matmul_sme");
}
