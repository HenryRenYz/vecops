//
// Copyright (c) vecops contributors.
//

#ifndef VECOPS_EXECUTION_DETAILS_X86_RESOURCES_H
#define VECOPS_EXECUTION_DETAILS_X86_RESOURCES_H

#include <cstddef>
#include <cstdint>

#include "vecops/CoreDefs.h"

#if defined(HAS_AMX_TILE)
#include "vecops/vec/details/amx/AMX.h"
#endif

/**
 * @file Resources.h
 * @brief Type-level AMX ownership tag, TILECFG image, and release guard.
 */

namespace vecops::execution::details::x86 {

/** Resource tag proving that the scope owns live Intel AMX tile state. */
struct Tiles {};

/**
 * @brief Architectural 64-byte Intel AMX TILECFG image.
 *
 * `column_bytes[i]` and `rows[i]` configure tile `i`. `load()` requires an
 * active `Tiles` execution resource and emits one LDTILECFG. The type is
 * aligned so its address can be passed directly to the intrinsic.
 */
struct alignas(64) TileConfiguration {
  using RequiredResource = Tiles;

  uint8_t palette = 1;
  uint8_t start_row = 0;
  uint8_t reserved0[14]{};
  // The architectural image reserves descriptors for 16 tiles even though
  // palette 1 currently exposes eight data tile registers.
  uint16_t column_bytes[16]{};
  uint8_t rows[16]{};

#if defined(HAS_AMX_TILE)
  /** Load this configuration into the current thread's AMX tile state. */
  VECOPS_ALWAYS_INLINE void load() const {
    vec::details::amx::load_configuration(this);
  }
#else
  void load() const = delete;
#endif
};

static_assert(sizeof(TileConfiguration) == 64);
static_assert(alignof(TileConfiguration) == 64);

#if defined(HAS_AMX_TILE)
/** Scope-exit guard emitting TILERELEASE exactly once for a new tile region. */
struct TileReleaseGuard {
  TileReleaseGuard() = default;
  TileReleaseGuard(const TileReleaseGuard&) = delete;
  TileReleaseGuard& operator=(const TileReleaseGuard&) = delete;
  VECOPS_ALWAYS_INLINE ~TileReleaseGuard() { vec::details::amx::release(); }
};
#endif

} // namespace vecops::execution::details::x86

#endif // VECOPS_EXECUTION_DETAILS_X86_RESOURCES_H
