#include "MatmulBenchCommon.h"

#include "vecops/gemm/Atoms.h"

namespace vecops::bench::matmul {

template <>
struct AtomName<gemm::SME_BF16F32> {
  static constexpr const char* value = "SME_BF16F32";
};

template <typename Policy>
void register_policy(const char* policy) {
  using Atom = gemm::SME_BF16F32;
  register_fixed_policy_mode<
      Atom, InputMode::PackedAB, Policy, 16, 16, 8>(
          "fixed_microkernel", "acc_1x1", policy);
  register_fixed_policy_mode<
      Atom, InputMode::PackedAB, Policy, 16, 64, 8>(
          "fixed_microkernel", "acc_1x4", policy);
  register_fixed_policy_mode<
      Atom, InputMode::PackedAB, Policy, 64, 16, 8>(
          "fixed_microkernel", "acc_4x1", policy);

  register_fixed_policy_mode<
      Atom, InputMode::Raw, Policy, 35, 53, 17>(
          "fixed_tail", "mnk_tail", policy);
  register_fixed_policy_mode<
      Atom, InputMode::PackedAB, Policy, 35, 53, 17>(
          "fixed_tail", "mnk_tail", policy);

  register_fixed_policy_mode<
      Atom, InputMode::Raw, Policy, 1, 1152, 896>(
          "fixed_representative", "decode_projection", policy);
  register_fixed_policy_mode<
      Atom, InputMode::PackedB, Policy, 1, 1152, 896>(
          "fixed_representative", "decode_projection", policy);
  register_fixed_policy_mode<
      Atom, InputMode::PackedAB, Policy, 1, 1152, 896>(
          "fixed_representative", "decode_projection", policy);

  register_fixed_policy_mode<
      Atom, InputMode::Raw, Policy, 256, 256, 256>(
          "fixed_representative", "square", policy);
  register_fixed_policy_mode<
      Atom, InputMode::PackedAB, Policy, 256, 256, 256>(
          "fixed_representative", "square", policy);
  register_fixed_policy_mode<
      Atom, InputMode::Raw, Policy, 128, 1024, 768>(
          "fixed_representative", "batch_projection", policy);
  register_fixed_policy_mode<
      Atom, InputMode::PackedAB, Policy, 128, 1024, 768>(
          "fixed_representative", "batch_projection", policy);
}

} // namespace vecops::bench::matmul

int main(int argc, char** argv) {
  using namespace vecops::bench::matmul;
  register_policy<vecops::kernel::loop::tile2d_policy::RuntimeExactArea4>(
      "runtime_exact");
  register_policy<vecops::kernel::loop::tile2d_policy::FourRegions>(
      "four_regions");
  register_policy<vecops::kernel::matmul_policy::Automatic>(
      "automatic");
  return run_benchmarks(argc, argv, "matmul_sme_fixed_policy");
}
