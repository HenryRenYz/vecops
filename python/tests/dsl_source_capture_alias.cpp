// SPDX-FileCopyrightText: 2026 HenryRenYz and vecops contributors
// SPDX-License-Identifier: MIT
/** @file dsl_source_capture_alias.cpp
 * @brief Source-constexpr aliases must survive schema generation and batching.
 */
#include "vecops/Kernel.h"

using NativeA = vecops::spec::NativeA;
using NativeB = vecops::spec::NativeB;
static_assert(std::same_as<NativeA, NativeB>);
static_assert(vecops::spec::ScaleA == vecops::spec::ScaleB);
static_assert(vecops::spec::EnabledA == vecops::spec::EnabledB);

void __kernel__(vecops::TensorLike auto input, vecops::TensorLike auto output) {
  for (vecops::nint_t i = 0; i < input.size(0); ++i) {
    const auto x = static_cast<vecops::float64_t>(input(i));
    output(i) = static_cast<NativeB>(x * vecops::spec::ScaleA + vecops::spec::ScaleB +
                                   (vecops::spec::EnabledA ? 1 : 0) + (vecops::spec::EnabledB ? 2 : 0));
  }
}
