//
// Copyright (c) vecops contributors.
//

#ifndef VECOPS_MATMUL_ATOM_H
#define VECOPS_MATMUL_ATOM_H

#include <concepts>
#include <type_traits>

#include "vecops/CoreTypes.h"
#include "vecops/Features.h"
#include "vecops/Meta.h"

namespace vecops::gemm {

/**
 * @brief Identifies one input of the logical product
 * `C[M,N] = A[M,K] * B[N,K]^T`.
 */
enum class Operand { A, B };

/**
 * @brief Describes one hardware matrix-instruction family.
 *
 * An Atom is the compile-time contract shared by kernels built from the same
 * instruction and operand types.  It does not represent one kernel shape or
 * one invocation.  Concrete atoms such as `AMX_BF16F32` and `SME_F16F32`
 * provide:
 *
 * - `KernelKind`, including the execution resources required by the ISA;
 * - `TA`, `TB`, `TC`, and `TAcc` operand and accumulator element types;
 * - `M_R`, `N_R`, and `K_R` native instruction extents; and
 * - an operand-specific `Packing` format for A and B.
 *
 * Atom types are concrete, unqualified types.  Parameterized instruction
 * families may use a class template to produce such a concrete type, as done
 * by the signedness variants of the integer atoms.
 */
template <typename T>
concept Atom = requires {
  typename T::KernelKind;
  typename T::TA;
  typename T::TB;
  typename T::TC;
  typename T::TAcc;
  typename T::template Packing<Operand::A>;
  typename T::template Packing<Operand::B>;
  T::M_R;
  T::N_R;
  T::K_R;
} && std::same_as<T, std::remove_cvref_t<T>>;

} // namespace vecops::gemm

#if defined(ARCH_X86_FAMILY)
#include "vecops/matmul/details/amx/Atoms.h"
#endif

#if defined(HAS_SME)
#include "vecops/matmul/details/sme/Atoms.h"
#endif

#endif // VECOPS_MATMUL_ATOM_H
