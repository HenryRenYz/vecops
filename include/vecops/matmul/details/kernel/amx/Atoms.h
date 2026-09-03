//
// Copyright (c) vecops contributors.
//

#ifndef VECOPS_MATMUL_DETAILS_AMX_ATOMS_H
#define VECOPS_MATMUL_DETAILS_AMX_ATOMS_H

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

struct AMXKernelKind {
  using ResourceRequirements = execution::details::ResourceSet<
      execution::details::x86::Tiles>;
};

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

struct AMX_BF16F32 : AMXAtomBase<bfloat16_t, bfloat16_t, float32_t> {
  using SwappedAtom = AMX_BF16F32;
};
struct AMX_F16F32 : AMXAtomBase<float16_t, float16_t, float32_t> {
  using SwappedAtom = AMX_F16F32;
};

template <typename A, typename B>
  requires ((std::same_as<A, int8_t> || std::same_as<A, uint8_t>) &&
            (std::same_as<B, int8_t> || std::same_as<B, uint8_t>))
struct AMX_I8I32 : AMXAtomBase<A, B, int32_t> {
  using SwappedAtom = AMX_I8I32<B, A>;
};

static_assert(Atom<AMX_BF16F32>);

} // namespace vecops::matmul

#endif // VECOPS_MATMUL_DETAILS_AMX_ATOMS_H
