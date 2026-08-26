//
// Copyright (c) vecops contributors.
//

#ifndef VECOPS_EMUAMX_H
#define VECOPS_EMUAMX_H

#include "vecops/gemm/KernelBase.h"
#include "vecops/gemm/details/amx/Packing.h"

namespace vecops::gemm::EmuAMX {

/** Emulated kernels need no AMX execution-state resource. */
struct KernelKind {
  using ResourceRequirements =
      typename execution::details::current_backend_t::DefaultRequirements;
};

/**
 * Emulated BF16 atom using the real AMX packed-format contract.
 * The format is independent of whether a consuming kernel uses AMX tiles.
 */
struct AtomBF16BF16F32 {
  using KernelKind = EmuAMX::KernelKind;
  using TA = bfloat16_t;
  using TB = bfloat16_t;
  using TC = float32_t;
  using TAcc = float32_t;

  static constexpr auto M_R = meta::cint<16>;
  static constexpr auto N_R = meta::cint<16>;
  static constexpr auto K_R = meta::cint<32>;

  template <Operand Side>
  using Packing = details::amx::Packing<
      std::conditional_t<Side == Operand::A, TA, TB>, Side>;
};

static_assert(gemm::Atom<AtomBF16BF16F32>);

} // namespace vecops::gemm::EmuAMX

#endif // VECOPS_EMUAMX_H
