// SPDX-FileCopyrightText: 2026 HenryRenYz and vecops contributors
// SPDX-License-Identifier: MIT
#pragma once
#include "vecops/memory/Model.h"

namespace vecops::memory::model {
/** A device operation owns its registrations and operand owner until stopped.
 * A failed submission may still access memory. query() must preserve that
 * distinction and retry device completion after an error. No implicit fallback.
 */
struct LinuxTransferStatistics {
  std::uint64_t payload_bytes = 0, pinned_bytes = 0, registrations = 0;
};
class LinuxTransferOperation {
public:
  virtual ~LinuxTransferOperation() = default;
  virtual Result<EventStatus> query() = 0;
  // Prevent future chunks, without claiming current DMA has stopped.
  virtual void stop_submitting(Error) = 0;
  virtual LinuxTransferStatistics statistics() const { return {}; }
};
class LinuxTransferExecutor {
public:
  virtual ~LinuxTransferExecutor() = default;
  virtual ProgressCapabilities progress() const = 0;
  virtual Result<std::shared_ptr<LinuxTransferOperation>> copy(void* target, const void* source,
                                                               std::uint64_t bytes,
                                                               std::shared_ptr<void> owner) = 0;
};
/** Explicit opaque method binding; controller is the host address-space agent
 * submitting work, agent is the device executor represented by the graph route.
 */
struct LinuxTransferQueue {
  std::string agent;
  std::string controller;
  std::vector<std::string> methods;
  std::shared_ptr<LinuxTransferExecutor> executor;
};
/** Retry uncertain submissions retained after Runtime teardown. Returns the
 * observed token states, releasing quarantine owners only after stopped. Never
 * cancels or guesses completion. Permanent failures remain auditable/retained.
 */
struct LinuxTransferDiagnostic {
  CompletionToken token;
  EventStatus status;
  LinuxTransferStatistics retained;
};
Result<std::vector<LinuxTransferDiagnostic>> poll_linux_transfer_quarantine();
} // namespace vecops::memory::model
