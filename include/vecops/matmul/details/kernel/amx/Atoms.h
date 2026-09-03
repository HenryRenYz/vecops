//
// Copyright (c) vecops contributors.
//

#ifndef VECOPS_MATMUL_DETAILS_AMX_ATOMS_H
#define VECOPS_MATMUL_DETAILS_AMX_ATOMS_H

/**
 * @file vecops/matmul/details/kernel/amx/Atoms.h
 * @brief AMX Atom catalog: microkernel shapes fixed by the tile hardware.
 *
 * Design intent: one AMX tile register is architecturally 16 rows by
 * 64 bytes, independent of element type. AMXAtomBase derives every Atom
 * extent from those two constants: the register block is 16x16 for every
 * accumulator type (the 64-byte column budget covers 16 f32 lanes), and
 * one tileload fills a whole 64-byte row, so the K step is 64/sizeof(A)
 * elements. The declared Packing alias forwards each side to the amx
 * packing format so planners can check panel shapes without naming the
 * backend namespace.
 */

#include <concepts>
#include <type_traits>

#include "vecops/execution/details/ResourceSet.h"
#include "vecops/execution/details/x86/Resources.h"
#include "vecops/matmul/Atom.h"

namespace vecops::matmul::details::amx {

template <typename ElementT, Operand Side>
struct Packing;

} // namespace vecops::matmul::details::amx

namespace vecops::matmul {

/// Kernel-kind marker: an AMX leaf needs the x86 tile resources held for
/// the whole traversal (one TILECFG image + tile registers).
struct AMXKernelKind {
  using ResourceRequirements = execution::details::ResourceSet<
      execution::details::x86::Tiles>;
};

/// Extents and packing hook shared by every AMX Atom.
///
/// M_R/N_R: one tile register holds 16 rows x 16 accumulator lanes (64
/// bytes per row / sizeof(Acc)). K_R: elements of A consumed per dot
/// step — one full 64-byte tileload row.
template <typename A, typename B, typename Acc>
struct AMXAtomBase {
  using KernelKind = AMXKernelKind;
  using TA = A;
  using TB = B;
  using TC = Acc;
  using TAcc = Acc;

  static constexpr auto M_R = meta::cint<16>;
  static constexpr auto N_R = meta::cint<16>;
  static constexpr auto K_R = meta::cint<64 / sizeof(A)>;

  template <Operand Side>
  using Packing = details::amx::Packing<
      std::conditional_t<Side == Operand::A, TA, TB>, Side>;
};

/// BF16 inputs, f32 accumulate (TMUL bf16 dot). Self-swapped: A and B
/// share the element type, so orientation exchange maps the atom to itself.
struct AMX_BF16F32 : AMXAtomBase<bfloat16_t, bfloat16_t, float32_t> {
  using SwappedAtom = AMX_BF16F32;
};
/// FP16 inputs, f32 accumulate (TMUL fp16 dot). Self-swapped like above.
struct AMX_F16F32 : AMXAtomBase<float16_t, float16_t, float32_t> {
  using SwappedAtom = AMX_F16F32;
};

/// 8-bit inputs of either sign mix, i32 accumulate (TMUL int8 dot).
/// Swapping A and B also swaps the signedness slots, so SwappedAtom names
/// the atom with TA/TB exchanged.
template <typename A, typename B>
  requires ((std::same_as<A, int8_t> || std::same_as<A, uint8_t>) &&
            (std::same_as<B, int8_t> || std::same_as<B, uint8_t>))
struct AMX_I8I32 : AMXAtomBase<A, B, int32_t> {
  using SwappedAtom = AMX_I8I32<B, A>;
};

static_assert(Atom<AMX_BF16F32>);

} // namespace vecops::matmul

#endif // VECOPS_MATMUL_DETAILS_AMX_ATOMS_H
