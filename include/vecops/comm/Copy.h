// SPDX-FileCopyrightText: 2026 HenryRenYz and vecops contributors
// SPDX-License-Identifier: MIT
#pragma once
/** @file Copy.h
 * Prevalidated disjoint interleaved rows. Sources are compact snapshots; the
 * destination stores source p's run at row*row_stride+p*run. Callers retain
 * all pointers until the synchronous team returns. No allocation or ISA
 * discovery occurs in worker callbacks. This is byte transport, not memmove.
 */
#include <cstddef>
#include <cstdint>
namespace vecops::comm::detail {
struct CopyRows {
  const void* const* sources;
  void* destination;
  std::size_t peers, rows, run, row_stride;
};
/** Every payload path uses vec, including contiguous/strided snapshots and
 * replicas. The ISA implementation is selected once, outside worker callbacks.
 * Ranges must be disjoint. Zero byte/row counts do not dereference pointers.
 */
struct CopyBackend {
  void (*bytes)(void*, const void*, std::size_t) noexcept;
  void (*strided)(void*, const void*, const std::size_t*, std::size_t,
                  std::size_t, std::size_t, bool) noexcept;
  void (*rows)(void*, std::uint32_t, std::uint32_t) noexcept;
};
const CopyBackend& copy_backend() noexcept;
}
