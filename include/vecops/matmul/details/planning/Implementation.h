// SPDX-FileCopyrightText: 2026 HenryRenYz and vecops contributors
// SPDX-License-Identifier: MIT
//
//

#ifndef VECOPS_MATMUL_DETAILS_IMPLEMENTATION_H
#define VECOPS_MATMUL_DETAILS_IMPLEMENTATION_H

/**
 * @file vecops/matmul/details/planning/Implementation.h
 * @brief Compile-time mapping from a public Atom's KernelKind tag to its
 *        backend kernel implementation leaf.
 *
 * The public Atom vocabulary (matmul/Atom.h) names an instruction family via
 * a tag type such as AMXKernelKind or SMEKernelKind, while the kernel layer
 * keys its Backend machinery on concrete implementation types in
 * kernel::matmul_implementation.  This header is the single translation
 * point between the two spellings.
 *
 * Specializations exist only behind the ISA feature macros that enable their
 * backend (HAS_AMX_TILE / HAS_SME).  On a platform without the feature the
 * primary template stays undefined, so using SelectedImplementation fails
 * at compile time instead of silently selecting a different implementation.
 */

#include "vecops/matmul/Atom.h"
#include "vecops/matmul/details/kernel/Kernel.h"

namespace vecops::matmul::details {

/// Resolves a KernelKind tag to its kernel implementation type; undefined
/// unless a backend specialization for the kind exists on this platform.
template <typename KernelKind>
struct SelectImplementation;

#if defined(HAS_AMX_TILE)
/// AMX tile kernels, available with HAS_AMX_TILE.
template <>
struct SelectImplementation<::vecops::matmul::AMXKernelKind> {
  using type = kernel::matmul_implementation::AMX;
};
#endif

#if defined(HAS_SME)
/// SME outer-product kernels, available with HAS_SME.
template <>
struct SelectImplementation<::vecops::matmul::SMEKernelKind> {
  using type = kernel::matmul_implementation::SME;
};
#endif

/// The kernel implementation leaf selected for one concrete Atom.
template <::vecops::matmul::Atom Atom>
using SelectedImplementation =
    typename SelectImplementation<typename Atom::KernelKind>::type;

} // namespace vecops::matmul::details

#endif // VECOPS_MATMUL_DETAILS_IMPLEMENTATION_H
