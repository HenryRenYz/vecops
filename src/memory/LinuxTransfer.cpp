// SPDX-FileCopyrightText: 2026 HenryRenYz and vecops contributors
// SPDX-License-Identifier: MIT
#include "details/RuntimeInternal.h"
namespace vecops::memory::model {
namespace {
struct Entry { std::shared_ptr<HostEventService> service; CompletionToken token; std::shared_ptr<LinuxTransferOperation> operation; std::shared_ptr<HostEventReference> reference; };
struct Quarantine { std::mutex mutex; std::vector<std::shared_ptr<Entry>> entries; };
Quarantine& registry() {
  // Intentional process lifetime: static destruction must not destroy mappings
  // still accessible by a failed device or block waiting for that device.
  static auto* state = new Quarantine;
  return *state;
}
}
void details::quarantine_transfer(std::shared_ptr<HostEventService> service, CompletionToken token,
                                  std::shared_ptr<LinuxTransferOperation> operation) {
  auto reference = details::take(service->retain_event(token));
  auto entry = std::make_shared<Entry>(Entry{std::move(service), std::move(token), std::move(operation), std::move(reference)});
  std::lock_guard lock(registry().mutex);
  registry().entries.push_back(std::move(entry));
}
Result<std::vector<LinuxTransferDiagnostic>> poll_linux_transfer_quarantine() {
  return details::protect([] {
    std::vector<std::shared_ptr<Entry>> snapshot, done;
    { std::lock_guard lock(registry().mutex); snapshot = registry().entries; }
    std::vector<LinuxTransferDiagnostic> result;
    for (const auto& entry : snapshot) {
      auto s = entry->reference->query();
      EventStatus status = s ? s.value() : EventStatus{EventState::failed, false, std::get<Error>(s.state)};
      result.push_back({entry->token, status, entry->operation->statistics()});
      if (details::stopped(status)) done.push_back(entry);
    }
    {
      std::lock_guard lock(registry().mutex);
      for (const auto& entry : done)
        std::erase(registry().entries, entry);
    }
    // snapshot keeps Service alive until outside all callbacks and registry lock.
    return result;
  });
}
} // namespace vecops::memory::model
