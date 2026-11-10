// SPDX-FileCopyrightText: 2026 HenryRenYz and vecops contributors
// SPDX-License-Identifier: MIT
#pragma once
#include "vecops/memory/Model.h"
#include <functional>

namespace vecops::memory::model {
/** Adapter payload for ExternalEvent{backend="host-callback", handle=callbacks}.
 * Query/wait must report stopped accesses independently of logical failure or
 * cancellation. A cancellation callback requests cancellation only. A missing
 * cancel function is a successful no-op. Query and wait are mandatory.
 */
struct HostEventCallbacks {
  std::function<Result<EventStatus>()> query;
  std::function<Result<EventStatus>()> wait;
  std::function<Result<void>()> cancel;
};
/** Authenticated observation ownership, independent of the public token registry.
 * A reference remains observable after release(token), and can outlive the
 * service. Service destruction still drains live records before returning.
 * References retain the record, not the service or a service address.
 */
class HostEventReference {
public:
  virtual ~HostEventReference() = default;
  virtual Result<EventStatus> query() const = 0;
  virtual Result<EventStatus> wait() = 0;
};
/** Thread-safe host completion registry with explicitly released tokens.
 * Token IDs are service-unique and never reused. All three token fields are
 * authenticated. Callers give each replay a unique InvocationStamp::id; live
 * stamps cannot associate one ID with conflicting binding generations.
 * Joins may reference authenticated producer events from older invocations.
 *
 * Callbacks run without the registry or record mutex. Operations on one record
 * serialize; callback reentry into an already-active record reports an error
 * instead of waiting, preventing same-thread and cross-callback wait cycles.
 * Different records can progress concurrently. Cancellation runs separately so
 * it can interrupt a blocking wait; stopped is not published during a running
 * cancellation callback. wait() returns
 * the adapter's observed status; failure/cancellation without stopped accesses
 * still forbids release and memory reuse.
 *
 * Destruction drains every remaining record until terminal AND accesses_stopped.
 * It never cancels implicitly or abandons live owners. An adapter that cannot
 * certify quiescence can therefore keep destruction waiting indefinitely; its
 * owner/callbacks must remain usable during this drain. Adapter errors are not
 * evidence of stopped accesses and are retried during destruction.
 */
class HostEventService final : public EventService {
public:
  HostEventService();
  ~HostEventService() override;
  HostEventService(const HostEventService&) = delete;
  HostEventService& operator=(const HostEventService&) = delete;
  Result<CompletionToken> import_event(const ExternalEvent&, const InvocationContext&) override;
  Result<EventStatus> query(const CompletionToken&) const override;
  Result<EventStatus> wait(const CompletionToken&) override;
  Result<CompletionToken> join(std::span<const CompletionToken>, const InvocationContext&) override;
  Result<void> request_cancel(const CompletionToken&) override;
  Result<void> release(std::span<const CompletionToken>) override;
  /** Mint a synchronous event; only terminal + stopped statuses are accepted. */
  Result<CompletionToken> complete(const InvocationStamp&, const EventStatus&);
  /** Retain a backing/DSO/other owner until accesses stop. Already-stopped
   * records accept the call without retaining the new owner. */
  Result<void> retain_until(const CompletionToken&, std::shared_ptr<void>);
  /** Authenticate once and keep stable observation ownership. This does not
   * prevent explicit token release or retain registry entries. */
  Result<std::shared_ptr<HostEventReference>> retain_event(const CompletionToken&);
  std::size_t token_count() const;

private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};
} // namespace vecops::memory::model
