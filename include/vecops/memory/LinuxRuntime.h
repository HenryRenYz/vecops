// SPDX-FileCopyrightText: 2026 HenryRenYz and vecops contributors
// SPDX-License-Identifier: MIT
#pragma once
#include "vecops/memory/LinuxDiscovery.h"
#include "vecops/memory/LinuxTransfer.h"
#include "vecops/memory/Calibration.h"

namespace vecops::memory::model {
enum class SharedOperation { begin, publish, acquire, retire };
/** Same-host control adapter. Implementations compare the complete canonical
 * layout, exchange export metadata, and implement all-participant retire.
 * Tensor/operator semantics never cross this interface.
 */
class LinuxMemoryControl {
public:
  virtual ~LinuxMemoryControl() = default;
  virtual Result<void> agree(const DistributedLayout&, std::string_view canonical) = 0;
  virtual Result<std::map<std::string, LinuxBackingHandle>>
  exchange(const DistributedLayout&, const std::map<std::string, LinuxBackingHandle>& owned) = 0;
  virtual Result<CompletionToken> shared(SharedOperation, const SharedRange&, std::span<const CompletionToken>,
                                         EventService&) = 0;
};
struct LinuxRuntimeOptions {
  std::string participant; // empty: generate a process-local participant identity
  std::map<std::string, std::uint64_t> group_hard_limits;
  std::vector<BudgetConstraint> shared_limits;
  std::map<std::string, std::string> cpu_queues; // queue symbol -> executable Agent ID
  std::vector<std::string> cpu_copy_methods; // opaque method IDs bound to host memcpy
  std::map<std::string, LinuxTransferQueue> transfer_queues;
  // Bounded caller-driven completion window, after which an unfinished device
  // operation is reported failed/not-stopped and retained for later event polls.
  std::uint64_t transfer_wait_ms = 30000;
  std::shared_ptr<LinuxMemoryControl> control;
  // Job-local directory shared by participants on this host; owner-only 0700.
  // Empty selects an independent process runtime. Multi-process launchers must
  // supply the same directory and limits before any root allocation.
  std::optional<std::string> capacity_directory;
  RuntimeCounts runtime_counts; // explicit template runtime counts; local roots default to one
  std::shared_ptr<const Calibration> calibration; // compiled against the complete physical template
  CalibrationControls calibration_controls; // actual workload controls, never inferred from grants
};
/** Opens a CPU runtime from already certified discovery metadata. All direct
 * allocation, arenas and plan leases share one physical-capacity ledger.
 */
Result<std::unique_ptr<MemoryRuntime>> make_linux_runtime(LinuxDiscovery, LinuxRuntimeOptions = {});
/** No hardware file: use generic Linux CPU discovery (unknown rates stay unknown).
 * Supply a template + explicit profile for LX2. Policy installs domain caps and
 * arenas before the caller can allocate. Files are read locally on this host.
 */
Result<std::unique_ptr<MemoryRuntime>>
discover_linux_runtime(const std::optional<std::string>& hardware_file = std::nullopt,
                       LinuxCpuProfile profile = LinuxCpuProfile::generic_cpu, LinuxRuntimeOptions = {});
/** Hardware + application policy with explicit launch-time coordination.
 * Optional calibration is a strict supported memory-calibration/0.4 overlay,
 * expanded before discovery/scope projection. Unsupported profile semantics
 * fail before OS probing; workload controls must be explicitly registered in
 * options.calibration_controls and are never inferred from allocation grants.
 */
Result<std::unique_ptr<MemoryRuntime>>
open_linux_runtime(const std::string& hardware_file, const std::optional<std::string>& policy_file,
                   const std::optional<std::string>& calibration_file, LinuxRuntimeOptions);
} // namespace vecops::memory::model
