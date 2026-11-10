// SPDX-FileCopyrightText: 2026 HenryRenYz and vecops contributors
// SPDX-License-Identifier: MIT
#pragma once

#include "vecops/memory/Model.h"

namespace vecops::memory::model {
/** Immutable hardware-independent model. IDs and rules are opaque to this layer.
 * Scope projection retains the capacity of shared services. Recipe validation
 * concerns byte mappings only; graph lifetimes belong to the graph compiler.
 * Methods report errors before any allocation or executable work occurs.
 */
class ResourceModel final : public MemoryCostModel {
public:
  explicit ResourceModel(ResourceGraph graph);
  const ResourceGraph& graph() const noexcept {
    return graph_;
  }
  Result<ResourceView> view(const Scope&, std::uint64_t epoch = 0) const;
  Result<ResolvedPlacement> resolve_placement(const ResourceView&, const PlacementPlan&, std::uint64_t bytes) const;
  Result<bool> compatible_overlap(const ResolvedPlacement&, std::uint64_t left_offset, const ResolvedPlacement&,
                                  std::uint64_t right_offset, std::uint64_t bytes) const;
  Result<void> validate_recipe(const ResourceView&, const MemoryRecipe&) const;
  Result<OperandEnvironment> describe_operands(const ResourceView&, const MemoryRecipe&,
                                               std::span<const MemoryDescriptor> external = {}) const;
  Result<void> validate_commands(const ResourceView&, const MemoryCommandPlan&, const OperandEnvironment&) const;
  Result<CostPlan> cost_plan(const ResourceView&, const MemoryCommandPlan&, const OperandEnvironment&) const;
  Result<std::vector<TransferCandidate>> candidates(const ResourceView&, const TransferEndpoint&,
                                                    const TransferEndpoint&) const override;
  Result<CostEstimate> estimate(const ResourceView&, const TransferQuery&, const CostContext&) const override;
  Result<CostEstimate> estimate_access(const ResourceView&, const AccessQuery&, const CostContext&) const override;
  Result<CostEstimate> estimate_schedule(const ResourceView&, const CostPlan&, const CostContext&) const override;

private:
  ResourceGraph graph_;
};

/** Checked page charge. Throws invalid_argument/overflow_error on invalid input. */
std::uint64_t page_charge(std::uint64_t bytes, std::uint64_t quantum);
/** Admissible bytes including reusable cache, with outstanding grants deducted. */
Result<std::uint64_t> budget_available(const PoolBudget&);
} // namespace vecops::memory::model
