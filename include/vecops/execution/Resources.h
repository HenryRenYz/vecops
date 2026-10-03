#ifndef VECOPS_EXECUTION_RESOURCES_H
#define VECOPS_EXECUTION_RESOURCES_H

#include "vecops/execution/ExecutionSession.h"
#include "vecops/execution/details/arm/Resources.h"
#include "vecops/execution/details/x86/Resources.h"

/**
 * @file Resources.h
 * @brief Public architecture-specific lexical hardware-state contracts.
 *
 * session.with_resources(sme::Region{}, inline_noexcept_callback) owns SM/ZA.
 * session.with_resources(amx::Region{}, callback) obtains per-thread Linux
 * XTILEDATA permission and releases tiles on normal/exceptional exit. Inside
 * it, scope.with_configuration(TileConfiguration, callback) loads TILECFG.
 * Nested scopes inherit already-active resources without resetting them.
 * An AMX configuration is destructive: never change it with live tile data.
 * SME callbacks must contain no allocation, exceptions or non-inlined calls.
 */
namespace vecops::execution::sme {
using StreamingZA = details::arm::StreamingZA;
using Region = details::arm::StreamingZARegion;
}
namespace vecops::execution::amx {
using Tiles = details::x86::Tiles;
using TileConfiguration = details::x86::TileConfiguration;
struct Region { using ResourceRequirements = details::ResourceSet<Tiles>; };
}
#endif
