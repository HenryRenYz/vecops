//
// Copyright (c) vecops contributors.
//

#ifndef VECOPS_MATMUL_ATOM_H
#define VECOPS_MATMUL_ATOM_H

#include <concepts>
#include <type_traits>

#include "vecops/CoreTypes.h"
#include "vecops/platform/Features.h"
#include "vecops/Meta.h"

/**
 * @file vecops/matmul/Atom.h
 * @brief Hardware matrix-instruction "atom" contract for matmul.
 *
 * This header defines the `Atom` concept — the compile-time vocabulary that
 * every matmul kernel, planner, and tuning type is parameterized on — plus
 * the `Operand` enumerator that distinguishes the two inputs. An Atom names
 * one hardware matrix-instruction family (e.g. Intel AMX bf16, Arm SME fp32)
 * and carries everything downstream code needs to specialize for it:
 * execution resources, element types, native instruction extents, and the
 * packed input formats.
 *
 * On supported targets this header also pulls the concrete atom types
 * (`AMX_BF16F32`, `SME_F32F32`, the signedness-parameterized `*_I8I32`
 * variants, ...) into `vecops::matmul`, so user code can spell them directly
 * as template arguments to `ops::MatmulConfig`.
 *
 * ## Key components
 *
 * | Component  | Purpose                                              |
 * |------------|------------------------------------------------------|
 * | `Operand`  | Selects input A (`[M,K]`) or B (`[N,K]`, transposed) |
 * | `Atom`     | Concept every concrete atom type satisfies           |
 * | arch atoms | `AMX_*` (x86) / `SME_*` (Arm) types behind arch guards |
 *
 * ## Usage overview
 *
 * @code
 * #include "vecops/matmul/Atom.h"
 *
 * using vecops::matmul::AMX_BF16F32;   // available on x86 with AMX
 * static_assert(vecops::matmul::Atom<AMX_BF16F32>);
 *
 * // Atoms parameterize the public matmul entry point:
 * // ops::MatmulConfig<AMX_BF16F32> config;
 * @endcode
 *
 * ## Pitfalls
 *
 * - Concrete atom availability is architecture-conditional: the `AMX_*`
 *   types exist only under `ARCH_X86_FAMILY`, the `SME_*` types only under
 *   `HAS_SME`. Guard user code accordingly.
 * - `M_R`/`N_R`/`K_R` are Meta `Value` types, not plain integers. AMX atoms
 *   use `meta::Const`, while SME atoms without a fixed streaming-SVE vector
 *   length use runtime values whose Meta type still carries the
 *   architectural alignment and bounds.
 * - An Atom must be a concrete, unqualified type; `const T`, `T&`, and
 *   related spellings do not satisfy the concept.
 */
namespace vecops::matmul {

/**
 * @brief Identifies one input of the logical product
 * `C[M,N] = A[M,K] * B[N,K]^T`.
 *
 * @note `Operand::B` addresses B through its `[N,K]` layout and the explicit
 *       transpose in the product formula — B is never stored or loaded
 *       transposed by this enumeration alone.
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
 * An atom may additionally provide `SwappedAtom`. Orientation-aware planning
 * uses it when exchanging A and B; mixed-signedness atoms map this alias to the
 * corresponding atom with TA/TB exchanged.
 *
 * Atom types are concrete, unqualified types.  Parameterized instruction
 * families may use a class template to produce such a concrete type, as done
 * by the signedness variants of the integer atoms.
 *
 * @code
 * template <vecops::matmul::Atom AtomT>
 * void run(const typename AtomT::TA* a, const typename AtomT::TB* b,
 *          typename AtomT::TC* c) {
 *   // AtomT::M_R / AtomT::N_R / AtomT::K_R: native instruction extents,
 *   // each a Meta Value type (Const where the ISA fixes it).
 * }
 * @endcode
 *
 * @note The trailing `std::same_as<T, std::remove_cvref_t<T>>` requirement
 *       rejects cv-qualified or reference-adjusted spellings (`Atom<const T>`,
 *       `Atom<T&>`): an Atom is meant to be instantiated and stored by value
 *       throughout the configuration chain, so only the plain type is
 *       accepted.
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

} // namespace vecops::matmul

// Arch-conditional concrete atoms: the concept above is backend-independent,
// but the usable atom types (AMX_* on x86, SME_* on Arm) come from the
// backend headers included here. This is the single point where including
// "vecops/matmul/Atom.h" makes concrete atom names available.
#if defined(ARCH_X86_FAMILY)
#include "vecops/matmul/details/kernel/amx/Atoms.h"
#endif

#if defined(HAS_SME)
#include "vecops/matmul/details/kernel/sme/Atoms.h"
#endif

#endif // VECOPS_MATMUL_ATOM_H
