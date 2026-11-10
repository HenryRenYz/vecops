// SPDX-FileCopyrightText: 2026 HenryRenYz and vecops contributors
// SPDX-License-Identifier: MIT
#include "details/RuntimeInternal.h"

namespace vecops::memory::model {
struct Allocation::Impl {
  AllocationInfo info;
  std::shared_ptr<details::AllocationStorage> storage;
  std::shared_ptr<HostEventService> events;
  const void* arena = nullptr;
  // Event records retain storage until certified quiescence; free is nonblocking.
  ~Impl() = default;
};
Allocation::Allocation(std::unique_ptr<Impl> p) noexcept
  : impl_(std::move(p)) {
}
Allocation::Allocation(Allocation&&) noexcept = default;
Allocation& Allocation::operator=(Allocation&&) noexcept = default;
Allocation::~Allocation() = default;
const AllocationInfo& Allocation::info() const {
  details::require(impl_ != nullptr, "empty allocation");
  return impl_->info;
}
Result<void*> Allocation::data() const {
  return details::protect([&] {
    details::require(impl_ != nullptr, "empty allocation");
    return details::take(impl_->storage->mapping->data());
  });
}
Result<std::shared_ptr<SharedBacking>> Allocation::export_backing() const {
  return details::protect([&] {
    details::require(impl_ && impl_->info.shareable && impl_->info.remappable,
                     "allocation is not shareable/remappable");
    return impl_->storage->backing;
  });
}
namespace details {
Allocation AllocationAccess::make(AllocationInfo info, std::shared_ptr<AllocationStorage> storage,
                                  std::shared_ptr<HostEventService> events, const void* arena) {
  auto p = std::make_unique<Allocation::Impl>();
  p->info = std::move(info);
  p->storage = std::move(storage);
  p->events = std::move(events);
  p->arena = arena;
  return Allocation(std::move(p));
}
const void* AllocationAccess::arena(const Allocation& a) {
  require(a.impl_ != nullptr, "empty allocation");
  return a.impl_->arena;
}
std::shared_ptr<AllocationStorage> AllocationAccess::storage(const Allocation& a) {
  require(a.impl_ != nullptr, "empty allocation");
  return a.impl_->storage;
}
void AllocationAccess::record(Allocation& a, CompletionToken token) {
  require(a.impl_ != nullptr, "empty allocation");
  require(token.binding_generation == a.impl_->info.generation, "stale allocation-use generation");
  auto reference = take(a.impl_->events->retain_event(token));
  if (!stopped(take(reference->query()))) {
    auto retained = a.impl_->events->retain_until(token, a.impl_->storage);
    if (!retained && !stopped(take(reference->query())))
      take(retained);
  }
}
} // namespace details
} // namespace vecops::memory::model
