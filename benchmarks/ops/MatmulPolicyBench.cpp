// SPDX-FileCopyrightText: 2026 HenryRenYz and vecops contributors
// SPDX-License-Identifier: MIT
// @vecops-target-shards: 3

#include "MatmulBenchCommon.h"

#include "vecops/matmul/Atom.h"

namespace vecops::bench::matmul {
template <int Shard>
void register_matmul_policy_shard();
}

#if defined(VECOPS_TARGET_SHARD_ACTIVE)

namespace vecops::bench::matmul {

template <>
struct AtomName<::vecops::matmul::SME_BF16F32> {
  static constexpr const char* value = "SME_BF16F32";
};

template <typename Policy>
void register_policy(const char* policy) {
  using Atom = ::vecops::matmul::SME_BF16F32;
  register_policy_extent_pair<
      Atom, InputMode::PackedAB, Policy, 16, 16, 8>(
          "fixed_microkernel", "acc_1x1", policy);
  register_policy_extent_pair<
      Atom, InputMode::PackedAB, Policy, 16, 64, 8>(
          "fixed_microkernel", "acc_1x4", policy);
  register_policy_extent_pair<
      Atom, InputMode::PackedAB, Policy, 64, 16, 8>(
          "fixed_microkernel", "acc_4x1", policy);

  register_policy_extent_pair<
      Atom, InputMode::Raw, Policy, 35, 53, 17>(
          "fixed_tail", "mnk_tail", policy);
  register_policy_extent_pair<
      Atom, InputMode::PackedAB, Policy, 35, 53, 17>(
          "fixed_tail", "mnk_tail", policy);

  register_policy_extent_pair<
      Atom, InputMode::Raw, Policy, 1, 1152, 896>(
          "fixed_representative", "decode_projection", policy);
  register_policy_extent_pair<
      Atom, InputMode::PackedB, Policy, 1, 1152, 896>(
          "fixed_representative", "decode_projection", policy);
  register_policy_extent_pair<
      Atom, InputMode::PackedAB, Policy, 1, 1152, 896>(
          "fixed_representative", "decode_projection", policy);

  register_policy_extent_pair<
      Atom, InputMode::Raw, Policy, 256, 256, 256>(
          "fixed_representative", "square", policy);
  register_policy_extent_pair<
      Atom, InputMode::PackedAB, Policy, 256, 256, 256>(
          "fixed_representative", "square", policy);
  register_policy_extent_pair<
      Atom, InputMode::Raw, Policy, 128, 1024, 768>(
          "fixed_representative", "batch_projection", policy);
  register_policy_extent_pair<
      Atom, InputMode::PackedAB, Policy, 128, 1024, 768>(
          "fixed_representative", "batch_projection", policy);
}

template <int Shard>
void register_matmul_policy_shard() {
  static_assert(0 <= Shard && Shard < 3);
  if constexpr (Shard == 0) {
    register_policy<kernel::loop::tile2d_policy::ExactCover>(
        "exact_cover");
  } else if constexpr (Shard == 1) {
    register_policy<kernel::loop::tile2d_policy::FourRegions>(
        "four_regions");
  } else {
    register_policy<kernel::matmul_policy::Automatic>("automatic");
  }
}

} // namespace vecops::bench::matmul

template void vecops::bench::matmul::register_matmul_policy_shard<
    VECOPS_TARGET_SHARD_INDEX>();

#else

int main(int argc, char** argv) {
  using namespace vecops::bench::matmul;
  []<std::size_t... I>(std::index_sequence<I...>) {
    (register_matmul_policy_shard<static_cast<int>(I)>(), ...);
  }(std::make_index_sequence<3>{});
  return run_benchmarks(argc, argv, "matmul_policy");
}

#endif
