#include "MatmulBenchCommon.h"

#include "vecops/gemm/Atoms.h"

namespace vecops::bench::matmul {

template <>
struct AtomName<gemm::SME_BF16F32> {
  static constexpr const char* value = "SME_BF16F32";
};

} // namespace vecops::bench::matmul

int main(int argc, char** argv) {
  using namespace vecops::bench::matmul;
  using Atom = vecops::gemm::SME_BF16F32;

  register_fixed_mode<Atom, InputMode::PackedAB, 16, 16, 8>(
      "fixed_microkernel", "acc_1x1");
  register_fixed_mode<Atom, InputMode::PackedAB, 16, 64, 8>(
      "fixed_microkernel", "acc_1x4");
  register_fixed_mode<Atom, InputMode::PackedAB, 64, 16, 8>(
      "fixed_microkernel", "acc_4x1");

  register_fixed_mode<Atom, InputMode::Raw, 35, 53, 17>(
      "fixed_tail", "mnk_tail");
  register_fixed_mode<Atom, InputMode::PackedAB, 35, 53, 17>(
      "fixed_tail", "mnk_tail");

  register_fixed_mode<Atom, InputMode::Raw, 1, 1152, 896>(
      "fixed_representative", "decode_projection");
  register_fixed_mode<Atom, InputMode::PackedB, 1, 1152, 896>(
      "fixed_representative", "decode_projection");
  register_fixed_mode<Atom, InputMode::PackedAB, 1, 1152, 896>(
      "fixed_representative", "decode_projection");

  register_fixed_mode<Atom, InputMode::Raw, 256, 256, 256>(
      "fixed_representative", "square");
  register_fixed_mode<Atom, InputMode::PackedAB, 256, 256, 256>(
      "fixed_representative", "square");
  register_fixed_mode<Atom, InputMode::Raw, 128, 1024, 768>(
      "fixed_representative", "batch_projection");
  register_fixed_mode<Atom, InputMode::PackedAB, 128, 1024, 768>(
      "fixed_representative", "batch_projection");

  return run_benchmarks(argc, argv, "matmul_sme_fixed");
}
