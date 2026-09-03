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

/**
 * @file vecops/matmul/details/kernel/sme/Atoms.h
 * @brief Concrete SME matmul atoms: one ZA outer-product family per element
 *        type.
 *
 * Every atom here is built from `SMEAtomBase`, which derives the native
 * instruction extents from the ZA tile geometry (see the comment on
 * `SMEAtomBase`). The kernel backend (kernel/sme/Backend.h) selects the
 * FMOPA/BFMOPA/SMOPA/UMOPA/FMOPA-64 instruction through the atom's
 * TA/TB/TAcc types. Concrete atoms become available in `vecops::matmul`
 * only under `HAS_SME` (see matmul/Atom.h).
 */

namespace vecops::matmul::details::sme {

template <typename ElementT, Operand Side>
struct Packing;

} // namespace vecops::matmul::details::sme

namespace vecops::matmul {

/// Kernel kind for every SME atom: its kernels own one Streaming+ZA
/// interval around the tile traversal (see execution/details/arm/Resources.h).
struct SMEKernelKind {
  using ResourceRequirements = execution::details::ResourceSet<
      execution::details::arm::StreamingZA>;
};

/**
 * @brief Native extents and packing format for one SME matmul element
 *        triple.
 *
 * Extents and their hardware origin:
 *
 * - `M_R`/`N_R` = `SVL / sizeof(Acc)` (via `streaming_lanes_value`): one ZA
 *   tile is a square `SVL/sizeof(Acc)` x `SVL/sizeof(Acc)` accumulator, so
 *   one outer product covers that many rows and columns. See the two-state
 *   comment below for the fixed-SVL vs variable-SVL spelling.
 * - `K_R` = `sizeof(A) > 4 ? 1 : 4 / sizeof(A)`: the SME outer products
 *   (FMOPA/SMOPA family) consume K in 32-bit groups -- four i8, two i16
 *   (fp16/bf16), or one fp32 per group -- while the fp64 outer product
 *   (FMOPA .d) consumes a single element per step.
 *
 * @tparam A    Input element type (both operands share it per atom).
 * @tparam B    Input element type of the second operand.
 * @tparam Acc  Accumulator and output element type.
 */
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
  // 32-bit K groups for the SME outer products; 1 element per step for fp64.
  static constexpr auto K_R = meta::cint<
      (sizeof(A) > 4 ? 1 : 4 / sizeof(A))>;

  template <Operand Side>
  using Packing = details::sme::Packing<
      std::conditional_t<Side == Operand::A, TA, TB>, Side>;
};

/// fp32 x fp32 -> fp32 (FMOPA .s): K groups of one fp32. Self-swapped: A
/// and B share the element type, so orientation exchange maps to itself.
struct SME_F32F32 : SMEAtomBase<float32_t, float32_t, float32_t> {
  using SwappedAtom = SME_F32F32;
};
/// bf16 x bf16 -> fp32 (BFMOPA): K groups of two bf16. Self-swapped.
struct SME_BF16F32 : SMEAtomBase<bfloat16_t, bfloat16_t, float32_t> {
  using SwappedAtom = SME_BF16F32;
};
/// fp16 x fp16 -> fp32 (FMOPA .h): K groups of two fp16. Self-swapped.
struct SME_F16F32 : SMEAtomBase<float16_t, float16_t, float32_t> {
  using SwappedAtom = SME_F16F32;
};

#if defined(HAS_SME_F64F64)
/// fp64 x fp64 -> fp64 (FMOPA .d): one fp64 per K step. Self-swapped.
struct SME_F64F64 : SMEAtomBase<float64_t, float64_t, float64_t> {
  using SwappedAtom = SME_F64F64;
};
#endif

/// (u)i8 x (u)i8 -> i32 (S/UMOPA): K groups of four bytes; the mixed-sign
/// pairs select SUMOPA/USMOPA in the ZA wrapper. Both operands must carry
/// the same signedness here -- mixed-sign atoms are spelled by the template
/// arguments and validated by the requires clause. Swapping A and B also
/// swaps the signedness slots, so SwappedAtom names SME_I8I32<B, A>.
template <typename A, typename B>
  requires ((std::same_as<A, int8_t> || std::same_as<A, uint8_t>) &&
            (std::same_as<B, int8_t> || std::same_as<B, uint8_t>))
struct SME_I8I32 : SMEAtomBase<A, B, int32_t> {
  using SwappedAtom = SME_I8I32<B, A>;
};

static_assert(Atom<SME_F32F32>);

} // namespace vecops::matmul

#endif // VECOPS_MATMUL_DETAILS_SME_ATOMS_H
