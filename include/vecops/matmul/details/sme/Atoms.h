//
// Copyright (c) vecops contributors.
//

#ifndef VECOPS_MATMUL_DETAILS_SME_ATOMS_H
#define VECOPS_MATMUL_DETAILS_SME_ATOMS_H

#include <concepts>
#include <type_traits>

#include "vecops/execution/details/ResourceSet.h"
#include "vecops/execution/details/arm/Resources.h"
#include "vecops/matmul/Atom.h"
#include "vecops/vec/details/sme/State.h"

namespace vecops::matmul::details::sme {

template <typename ElementT, Operand Side>
struct Packing;

} // namespace vecops::matmul::details::sme

namespace vecops::matmul {

struct SMEKernelKind {
  using ResourceRequirements = execution::details::ResourceSet<
      execution::details::arm::StreamingZA>;
};

template <typename A, typename B, typename Acc>
struct SMEAtomBase {
  using KernelKind = SMEKernelKind;
  using TA = A;
  using TB = B;
  using TC = Acc;
  using TAcc = Acc;

  // One ZA tile is SVL/sizeof(Acc) by SVL/sizeof(Acc).  In a fixed-SVL build
  // these are Const values; otherwise they retain the architectural SVL byte
  // alignment and bounds through Meta arithmetic.
#if defined(HAS_FIXED_STREAMING_SVE_BITS)
  static constexpr auto M_R =
      vec::details::sme::streaming_lanes_value<Acc>();
  static constexpr auto N_R = M_R;
#else
  inline static const auto M_R =
      vec::details::sme::streaming_lanes_value<Acc>();
  inline static const auto N_R = M_R;
#endif
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

} // namespace vecops::matmul

#endif // VECOPS_MATMUL_DETAILS_SME_ATOMS_H
