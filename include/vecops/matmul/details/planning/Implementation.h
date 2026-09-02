//
// Copyright (c) vecops contributors.
//

#ifndef VECOPS_MATMUL_DETAILS_IMPLEMENTATION_H
#define VECOPS_MATMUL_DETAILS_IMPLEMENTATION_H

#include "vecops/matmul/Atom.h"
#include "vecops/matmul/details/kernel/Kernel.h"

namespace vecops::matmul::details {

template <typename KernelKind>
struct SelectImplementation;

#if defined(HAS_AMX_TILE)
template <>
struct SelectImplementation<::vecops::matmul::AMXKernelKind> {
  using type = kernel::matmul_implementation::AMX;
};
#endif

#if defined(HAS_SME)
template <>
struct SelectImplementation<::vecops::matmul::SMEKernelKind> {
  using type = kernel::matmul_implementation::SME;
};
#endif

template <::vecops::matmul::Atom Atom>
using SelectedImplementation =
    typename SelectImplementation<typename Atom::KernelKind>::type;

} // namespace vecops::matmul::details

#endif // VECOPS_MATMUL_DETAILS_IMPLEMENTATION_H
