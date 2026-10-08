// SPDX-FileCopyrightText: 2026 HenryRenYz and vecops contributors
// SPDX-License-Identifier: MIT
#ifndef VECOPS_VEC_AMX_H
#define VECOPS_VEC_AMX_H

#include "vecops/CoreDefs.h"
#include "vecops/platform/Features.h"

/**
 * @file AMX.h
 * @brief Public Intel AMX instruction primitives.
 *
 * zero/load/stream_load/store/dot require an active execution::amx::Region
 * and loaded TileConfiguration. Strides are bytes, tile indices constants
 * in [0,8). Load/store do not transpose: AVX/SVE register transposes remain
 * distinct from matrix engines. dot's dtype/target and literal-register
 * combinations are checked by the backend. ExecutionSession owns permission,
 * configuration and release; raw release/load_configuration are expert APIs.
 * Loads include compiler ordering for freshly produced packing buffers;
 * this adds no hardware fence and does not provide inter-thread synchronization.
 */
#if defined(HAS_AMX_TILE)
#include "vecops/vec/details/amx/AMX.h"
namespace vecops::vec::amx {
using details::amx::load_configuration;
using details::amx::release;
using details::amx::zero;
using details::amx::load;
using details::amx::stream_load;
using details::amx::store;
using details::amx::dot;
} // namespace vecops::vec::amx
#endif
#endif
