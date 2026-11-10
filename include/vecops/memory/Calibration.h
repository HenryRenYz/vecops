// SPDX-FileCopyrightText: 2026 HenryRenYz and vecops contributors
// SPDX-License-Identifier: MIT
#pragma once
#include "vecops/memory/HardwareTemplate.h"
#include "vecops/memory/ResourceModel.h"
#include <functional>

namespace vecops::memory::model {
/** Actual registered workload controls, not CPU affinity or execution-slot grants.
 * An implementation name is an opaque producer/consumer contract. A profile
 * with a nonempty implementation is eligible only for that implementation.
 */
struct CalibrationControl {
  std::optional<std::uint64_t> cpu_cores;
  std::string implementation;
};
using CalibrationControls = std::map<std::string, CalibrationControl>; // concrete executor ID
/** Temporary template relations. These callbacks are consumed during parsing;
 * no callbacks, addresses or template ancestry enter PerformanceDatabase.
 */
struct CalibrationSelectors {
  std::function<std::vector<std::string>(std::string_view)> select;
  std::function<bool(std::string_view, std::string_view, std::string_view)> relation;
};
struct CalibrationUse {
  std::string target, operation, profile, source;
  CostQuality provenance = CostQuality::reference;
  bool interpolated = false;
  double bandwidth = 0; // GiB/s
};
class Calibration;
/** Strict memory-calibration/0.4 subset: read/write access curves, copy
 * transfer curves/cases, and copy write_cost. Unsupported compute/triad
 * semantics are errors. Historical profiles default to reference provenance.
 */
Result<std::shared_ptr<const Calibration>> parse_calibration(std::string_view, const ResourceGraph&,
                                                             const CalibrationSelectors&);
struct CalibratedHardwareExpansion {
  HardwareExpansion hardware;
  std::shared_ptr<const Calibration> calibration;
};
/** Expand hardware and calibration using the very same temporary relationships.
 * Binding discovery and scope projection happen after this full expansion.
 */
Result<CalibratedHardwareExpansion> expand_calibrated_hardware_template(std::string_view hardware,
                                                                        std::string_view calibration,
                                                                        const RuntimeCounts& counts = {});
const PerformanceDatabase& calibration_database(const Calibration&);

/** Immutable cost facade. Physical graph and ResourceModel layout are unchanged.
 * Each call uses local caps; shared physical service demands still compete.
 * Aggregate quality remains reference because those physical services and
 * startup/latency references also participate. explain_access reports the
 * provenance of the selected curve independently of aggregate cost quality.
 * No extrapolation: unavailable controls/ranges leave the reference model.
 */
class CalibratedCostModel final : public MemoryCostModel {
public:
  CalibratedCostModel(const ResourceModel&, std::shared_ptr<const Calibration>, CalibrationControls = {});
  Result<std::vector<TransferCandidate>> candidates(const ResourceView&, const TransferEndpoint&,
                                                    const TransferEndpoint&) const override;
  Result<CostEstimate> estimate(const ResourceView&, const TransferQuery&, const CostContext&) const override;
  Result<CostEstimate> estimate_access(const ResourceView&, const AccessQuery&, const CostContext&) const override;
  Result<CostEstimate> estimate_schedule(const ResourceView&, const CostPlan&, const CostContext&) const override;
  Result<CostPlan> cost_plan(const ResourceView&, const MemoryCommandPlan&, const OperandEnvironment&) const;
  Result<std::vector<CalibrationUse>> explain_access(const ResourceView&, const AccessQuery&) const;

private:
  const ResourceModel& base_; // must outlive the facade
  std::shared_ptr<const Calibration> calibration_;
  CalibrationControls controls_;
};
} // namespace vecops::memory::model
