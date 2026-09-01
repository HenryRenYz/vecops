//
// Copyright (c) vecops contributors.
//

#ifndef VECOPS_MATMUL_DETAILS_OPERATION_COMMON_H
#define VECOPS_MATMUL_DETAILS_OPERATION_COMMON_H

#include <type_traits>
#include <utility>

#include "vecops/matmul/Atom.h"
#include "vecops/matmul/details/Kernel.h"

namespace vecops::ops::matmul_details {

template <typename T>
concept Extent = meta::ValueType<std::remove_cvref_t<T>> ||
    is_int_v<std::remove_cvref_t<T>>;

template <Extent T>
VECOPS_INLINE constexpr auto extent_value(T&& value) {
  using V = meta::to_value_t<std::remove_cvref_t<T>>;
  if constexpr (meta::ValueType<std::remove_cvref_t<T>>) {
    return std::forward<T>(value);
  } else {
    return V{static_cast<nint_t>(value)};
  }
}

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

} // namespace vecops::ops::matmul_details

#endif // VECOPS_MATMUL_DETAILS_OPERATION_COMMON_H
