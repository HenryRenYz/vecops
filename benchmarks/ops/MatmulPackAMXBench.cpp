#include "MatmulPackBenchCommon.h"

#include "vecops/gemm/Atoms.h"

namespace vecops::bench::matmul_pack {

template <>
struct AtomName<gemm::AMX_BF16F32> {
  static constexpr const char* value = "AMX_BF16F32";
};

template <>
struct AtomName<gemm::AMX_F16F32> {
  static constexpr const char* value = "AMX_F16F32";
};

using AMXI8 = gemm::AMX_I8I32<int8_t, uint8_t>;

template <>
struct AtomName<AMXI8> {
  static constexpr const char* value = "AMX_I8I32";
};

} // namespace vecops::bench::matmul_pack

int main(int argc, char** argv) {
  using namespace vecops::bench::matmul_pack;
  register_cases<vecops::gemm::AMX_BF16F32>(RepresentativeCases{});
  register_cases<vecops::gemm::AMX_F16F32>(DTypeProbeCases{});
  register_cases<AMXI8>(DTypeProbeCases{});
  return run_benchmarks(argc, argv, "matmul_pack_amx");
}
