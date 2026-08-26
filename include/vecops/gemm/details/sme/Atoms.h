//
// Copyright (c) vecops contributors.
//

#ifndef VECOPS_GEMM_DETAILS_SME_ATOMS_H
#define VECOPS_GEMM_DETAILS_SME_ATOMS_H

#include <concepts>

#include "vecops/execution/details/ResourceSet.h"
#include "vecops/execution/details/arm/Resources.h"
#include "vecops/gemm/details/sme/Packing.h"

namespace vecops::gemm {

struct SMEKernelKind {
  using ResourceRequirements = execution::details::ResourceSet<
      execution::details::arm::Streaming>;
};

template <typename A, typename B, typename Acc>
struct SMEAtomBase {
  using KernelKind = SMEKernelKind;
  using TA = A;
  using TB = B;
  using TC = Acc;
  using TAcc = Acc;

  // One ZA.S tile is SVL.W by SVL.W. Packing uses two such spatial
  // sub-panels, but the micro-kernel base tile must describe one ZA tile.
  inline static const meta::Any M_R{[] {
    if constexpr (sizeof(Acc) == 8)
      return static_cast<nint_t>(svcntsd());
    else
      return static_cast<nint_t>(svcntsw());
  }()};
  inline static const meta::Any N_R{static_cast<nint_t>(M_R)};
  static constexpr auto K_R = meta::cint<
      (sizeof(A) > 4 ? 1 : 4 / sizeof(A))>;

  template <Operand Side>
  using Packing = details::sme::Packing<
      std::conditional_t<Side == Operand::A, TA, TB>, Side>;
};

struct SME_F32F32 : SMEAtomBase<float32_t, float32_t, float32_t> {};
struct SME_BF16F32 : SMEAtomBase<bfloat16_t, bfloat16_t, float32_t> {};
struct SME_F16F32 : SMEAtomBase<float16_t, float16_t, float32_t> {};

#if defined(HAS_SME_F64F64)
struct SME_F64F64 : SMEAtomBase<float64_t, float64_t, float64_t> {};
#endif

template <typename A, typename B>
  requires ((std::same_as<A, int8_t> || std::same_as<A, uint8_t>) &&
            (std::same_as<B, int8_t> || std::same_as<B, uint8_t>))
struct SME_I8I32 : SMEAtomBase<A, B, int32_t> {};

static_assert(Atom<SME_F32F32>);

} // namespace vecops::gemm

#endif // VECOPS_GEMM_DETAILS_SME_ATOMS_H
