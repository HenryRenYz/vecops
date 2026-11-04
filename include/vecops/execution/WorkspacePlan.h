// SPDX-FileCopyrightText: 2026 HenryRenYz and vecops contributors
// SPDX-License-Identifier: MIT
#ifndef VECOPS_EXECUTION_WORKSPACE_PLAN_H
#define VECOPS_EXECUTION_WORKSPACE_PLAN_H
/** @file WorkspacePlan.h
 * Source-compatible workspace frontend names for the common BufferPlan IR.
 * No separate workspace placement or packing implementation is retained.
 */
#include "vecops/execution/BufferPlan.h"
namespace vecops::execution {
namespace workspace_plan_details = buffer_plan_details;
using WorkspaceSiteId = BufferSiteId;
using WorkspacePlacementPolicy = BufferPlacementPolicy;
using WorkspaceTier = BufferTier;
using WorkspaceDomain = BufferDomain;
using WorkspaceAllocationRequest = BufferAllocationRequest;
using WorkspaceLifetime = BufferLifetime;
using LogicalWorkspaceAllocation = LogicalBufferAllocation;
using LogicalWorkspacePlan = LogicalBufferPlan;
using WorkspaceTrace = BufferTrace;
using WorkspacePlacementEntry = BufferPlacementEntry;
using WorkspacePlacement = BufferPlacement;
using BoundWorkspaceSlot = BoundBufferSlot;
using BoundWorkspacePlan = BoundBufferPlan;
inline auto workspace_site_id(std::string_view recipe, std::string_view name, WorkspaceDomain domain) {
  return buffer_site_id(recipe,name,domain);
}
inline auto workspace_site_id(std::uint64_t scope, std::string_view name, WorkspaceDomain domain) {
  return buffer_site_id(scope,name,domain);
}
inline WorkspacePlacement place_workspace(const LogicalWorkspacePlan& logical,
    nint_t fast = std::numeric_limits<nint_t>::max(), nint_t slow = std::numeric_limits<nint_t>::max(),
    const std::vector<BufferReservation>& reserved = {}) {
  return place_buffers(logical,fast,slow,reserved);
}
} // namespace vecops::execution
#endif
