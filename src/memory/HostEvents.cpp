// SPDX-FileCopyrightText: 2026 HenryRenYz and vecops contributors
// SPDX-License-Identifier: MIT
#include "vecops/memory/HostEvents.h"
#include <chrono>
#include <condition_variable>
#include <limits>
#include <mutex>
#include <set>
#include <stdexcept>
#include <thread>

namespace vecops::memory::model {
namespace {
using U = std::uint64_t;
bool terminal(EventState state) {
  return state != EventState::pending;
}
bool finished(const EventStatus& status) {
  return terminal(status.state) && status.accesses_stopped;
}
bool valid_state(EventState state) {
  switch (state) {
  case EventState::pending:
  case EventState::succeeded:
  case EventState::failed:
  case EventState::cancelled:
    return true;
  }
  return false;
}
Error invalid_token() {
  return {"event.invalid_token", "unknown/released token or invocation/generation mismatch"};
}
Error reentrant() {
  return {"event.reentrant", "recursive operation on the active event callback"};
}
U service_identity() {
  static std::mutex mutex;
  static U next = 1;
  std::lock_guard lock(mutex);
  if (next == std::numeric_limits<U>::max())
    throw std::overflow_error("host event service identity exhausted");
  return next++;
}
thread_local unsigned callback_depth = 0;
struct CallbackScope {
  CallbackScope() {
    ++callback_depth;
  }
  ~CallbackScope() {
    --callback_depth;
  }
};
template <class F>
auto adapter_call(F&& call) -> decltype(call()) {
  try {
    CallbackScope scope;
    return call();
  } catch (const std::exception& e) {
    return {Error{"event.adapter_exception", e.what()}};
  } catch (...) {
    return {Error{"event.adapter_exception", "non-standard exception from host event adapter"}};
  }
}
} // namespace
struct HostEventService::Impl {
  struct Record {
    CompletionToken token;
    mutable std::mutex mutex;
    mutable std::condition_variable ready;
    bool running = false;
    std::thread::id running_thread;
    bool cancelling = false;
    std::thread::id cancelling_thread;
    bool observed = false;
    EventStatus status{EventState::pending, false, {}};
    std::optional<HostEventCallbacks> adapter;
    std::shared_ptr<void> external_owner;
    std::vector<std::shared_ptr<void>> retained;
    std::vector<std::shared_ptr<Record>> dependencies;
  };
  struct Invocation {
    U generation, count;
  };
  const U identity = service_identity();
  U next = 1;
  mutable std::mutex mutex;
  bool closing = false;
  std::map<std::string, std::shared_ptr<Record>> records;
  std::map<std::string, Invocation> invocations;

  std::shared_ptr<Record> lookup_locked(const CompletionToken& token) const {
    const auto found = records.find(token.id);
    if (found == records.end())
      return {};
    const auto& actual = found->second->token;
    if (actual.invocation != token.invocation || actual.binding_generation != token.binding_generation)
      return {};
    return found->second;
  }
  std::shared_ptr<Record> lookup(const CompletionToken& token) const {
    std::lock_guard lock(mutex);
    return lookup_locked(token);
  }
  Result<CompletionToken> mint(std::shared_ptr<Record> record, const InvocationStamp& stamp) {
    if (stamp.id.empty())
      return {Error{"event.invocation", "empty invocation identity"}};
    std::lock_guard lock(mutex);
    if (closing)
      return {Error{"event.closed", "event service is draining"}};
    const auto inv = invocations.find(stamp.id);
    if (inv != invocations.end() && inv->second.generation != stamp.binding_generation)
      return {Error{"event.invocation", "live invocation identity has a different binding generation"}};
    if (next == std::numeric_limits<U>::max())
      return {Error{"event.exhausted", "event token identity exhausted"}};
    record->token = {"host-event/" + std::to_string(identity) + "/" + std::to_string(next++), stamp.id,
                     stamp.binding_generation};
    records.emplace(record->token.id, record);
    auto [at, inserted] = invocations.emplace(stamp.id, Invocation{stamp.binding_generation, 0});
    (void)inserted;
    ++at->second.count;
    return {record->token};
  }
  static Result<void> begin(Record& record, std::unique_lock<std::mutex>& lock) {
    while (record.running) {
      if (record.running_thread == std::this_thread::get_id() || callback_depth != 0)
        return {reentrant()};
      record.ready.wait(lock, [&] { return !record.running; });
    }
    record.running = true;
    record.running_thread = std::this_thread::get_id();
    return {};
  }
  static void end(Record& record) {
    record.running = false;
    record.running_thread = {};
  }
  struct Operation {
    Record& record;
    std::unique_lock<std::mutex>& lock;
    bool cancellation = false;
    bool active = true;
    void finish() {
      if (!active)
        return;
      if (!lock.owns_lock())
        lock.lock();
      if (cancellation) {
        record.cancelling = false;
        record.cancelling_thread = {};
      } else
        end(record);
      active = false;
      lock.unlock();
      record.ready.notify_all();
    }
    ~Operation() {
      finish();
    }
  };
  static Result<EventStatus> poll_record(const std::shared_ptr<Record>& record, bool wait) {
    std::unique_lock lock(record->mutex);
    if (finished(record->status))
      return {record->status};
    auto started = begin(*record, lock);
    if (!started)
      return {*started.error};
    Operation operation{*record, lock};
    // Another serialized observer may have finished while begin() waited.
    if (finished(record->status)) {
      auto status = record->status;
      operation.finish();
      return {status};
    }
    auto callbacks = record->adapter;
    auto dependencies = record->dependencies;
    lock.unlock();
    Result<EventStatus> result{Error{"event.adapter", "event has no observation adapter"}};
    if (callbacks)
      result = adapter_call([&] { return wait ? callbacks->wait() : callbacks->query(); });
    else if (!dependencies.empty()) {
      EventStatus aggregate{EventState::succeeded, true, {}};
      bool pending = false, cancelled = false, failed = false;
      std::optional<Error> first_error;
      for (const auto& dependency : dependencies) {
        auto status = poll(dependency, wait);
        if (!status) {
          if (!first_error)
            first_error = std::get<Error>(status.state);
          continue;
        }
        const auto& value = status.value();
        aggregate.accesses_stopped = aggregate.accesses_stopped && value.accesses_stopped;
        pending = pending || value.state == EventState::pending;
        cancelled = cancelled || value.state == EventState::cancelled;
        if (value.state == EventState::failed && !failed)
          aggregate.error = value.error;
        failed = failed || value.state == EventState::failed;
      }
      aggregate.state = failed      ? EventState::failed
                        : cancelled ? EventState::cancelled
                        : pending   ? EventState::pending
                                    : EventState::succeeded;
      result = first_error ? Result<EventStatus>{*first_error} : Result<EventStatus>{aggregate};
    }
    std::vector<std::shared_ptr<void>> retired;
    std::shared_ptr<void> external;
    std::optional<HostEventCallbacks> retired_adapter;
    std::vector<std::shared_ptr<Record>> retired_dependencies;
    lock.lock();
    if (result) {
      // Cancellation callbacks can drive a blocking wait. Their own accesses
      // must finish before the service publishes stopped and releases owners.
      if (record->cancelling)
        result.value().accesses_stopped = false;
      const auto& incoming = result.value();
      if (!valid_state(incoming.state) ||
          (record->observed && record->status.accesses_stopped && !incoming.accesses_stopped) ||
          (record->observed && terminal(record->status.state) && incoming.state == EventState::pending) ||
          (record->observed && record->status.state == EventState::failed && incoming.state != EventState::failed) ||
          (record->observed && record->status.state == EventState::cancelled &&
           incoming.state == EventState::succeeded))
        result = {
          Error{"event.invalid_status", "adapter regressed a terminal/stopped status or returned an invalid state"}};
      else {
        record->status = incoming;
        record->observed = true;
        if (incoming.accesses_stopped)
          retired.swap(record->retained);
        if (finished(incoming)) {
          external = std::move(record->external_owner);
          retired_adapter = std::move(record->adapter);
          record->adapter.reset();
          retired_dependencies.swap(record->dependencies);
        }
      }
    }
    operation.finish();
    // Destruct callback/DSO/backing owners with no registry or record mutex held.
    return result;
  }
  static Result<EventStatus> poll(const std::shared_ptr<Record>& record, bool wait) {
    try {
      return poll_record(record, wait);
    } catch (const std::exception& e) {
      return {Error{"event.observation_exception", e.what()}};
    } catch (...) {
      return {Error{"event.observation_exception", "non-standard exception while observing host event"}};
    }
  }
  struct Reference final : HostEventReference {
    std::shared_ptr<Record> record;
    explicit Reference(std::shared_ptr<Record> value) : record(std::move(value)) {}
    Result<EventStatus> query() const override {
      return Impl::poll(record, false);
    }
    Result<EventStatus> wait() override {
      return Impl::poll(record, true);
    }
  };
  Result<void> cancel_record(const std::shared_ptr<Record>& record) const {
    std::unique_lock lock(record->mutex);
    if (record->status.accesses_stopped)
      return {};
    while (record->cancelling) {
      if (record->cancelling_thread == std::this_thread::get_id() || callback_depth != 0)
        return {reentrant()};
      record->ready.wait(lock, [&] { return !record->cancelling; });
    }
    if (record->status.accesses_stopped)
      return {};
    record->cancelling = true;
    record->cancelling_thread = std::this_thread::get_id();
    Operation operation{*record, lock, true};
    auto callbacks = record->adapter;
    auto dependencies = record->dependencies;
    auto external_owner = record->external_owner;
    lock.unlock();
    Result<void> result;
    if (callbacks && callbacks->cancel)
      result = adapter_call([&] { return callbacks->cancel(); });
    for (const auto& dependency : dependencies) {
      auto cancelled = cancel(dependency);
      if (!cancelled && !result.error)
        result = cancelled;
    }
    operation.finish();
    return result;
  }
  Result<void> cancel(const std::shared_ptr<Record>& record) const {
    try {
      return cancel_record(record);
    } catch (const std::exception& e) {
      return {Error{"event.cancellation_exception", e.what()}};
    } catch (...) {
      return {Error{"event.cancellation_exception", "non-standard exception while cancelling host event"}};
    }
  }
  void drain() {
    std::vector<std::shared_ptr<Record>> remaining;
    {
      std::lock_guard lock(mutex);
      closing = true;
      for (const auto& [id, record] : records) {
        (void)id;
        remaining.push_back(record);
      }
    }
    while (!remaining.empty()) {
      std::erase_if(remaining, [&](const auto& record) {
        auto status = poll(record, true);
        return status && finished(status.value());
      });
      if (!remaining.empty())
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
    std::map<std::string, std::shared_ptr<Record>> retired;
    {
      std::lock_guard lock(mutex);
      retired.swap(records);
      invocations.clear();
    }
  }
};
HostEventService::HostEventService()
  : impl_(std::make_unique<Impl>()) {
}
HostEventService::~HostEventService() {
  impl_->drain();
}
Result<CompletionToken> HostEventService::import_event(const ExternalEvent& event, const InvocationContext& context) {
  if (event.backend != "host-callback")
    return {Error{"event.backend", "unsupported external event backend"}};
  const auto* callbacks = std::any_cast<HostEventCallbacks>(&event.handle);
  if (!callbacks || !callbacks->query || !callbacks->wait)
    return {Error{"event.adapter", "host-callback requires query and wait callbacks"}};
  auto record = std::make_shared<Impl::Record>();
  record->adapter = *callbacks;
  record->external_owner = event.owner;
  return impl_->mint(std::move(record), context.stamp);
}
Result<EventStatus> HostEventService::query(const CompletionToken& token) const {
  auto record = impl_->lookup(token);
  if (!record)
    return {invalid_token()};
  return impl_->poll(record, false);
}
Result<EventStatus> HostEventService::wait(const CompletionToken& token) {
  auto record = impl_->lookup(token);
  if (!record)
    return {invalid_token()};
  return impl_->poll(record, true);
}
Result<CompletionToken> HostEventService::join(std::span<const CompletionToken> tokens,
                                               const InvocationContext& context) {
  auto record = std::make_shared<Impl::Record>();
  std::set<std::string> seen;
  {
    std::lock_guard lock(impl_->mutex);
    for (const auto& token : tokens) {
      if (!seen.insert(token.id).second)
        return {Error{"event.duplicate", "duplicate join dependency"}};
      auto dependency = impl_->lookup_locked(token);
      if (!dependency)
        return {invalid_token()};
      record->dependencies.push_back(std::move(dependency));
    }
  }
  if (tokens.empty()) {
    record->status = {EventState::succeeded, true, {}};
    record->observed = true;
  }
  return impl_->mint(std::move(record), context.stamp);
}
Result<void> HostEventService::request_cancel(const CompletionToken& token) {
  auto record = impl_->lookup(token);
  if (!record)
    return {invalid_token()};
  return impl_->cancel(record);
}
Result<void> HostEventService::release(std::span<const CompletionToken> tokens) {
  std::vector<std::shared_ptr<Impl::Record>> records;
  std::set<std::string> seen;
  {
    std::lock_guard lock(impl_->mutex);
    for (const auto& token : tokens) {
      if (!seen.insert(token.id).second)
        return {Error{"event.duplicate", "duplicate release token"}};
      auto record = impl_->lookup_locked(token);
      if (!record)
        return {invalid_token()};
      records.push_back(std::move(record));
    }
  }
  for (const auto& record : records) {
    auto status = impl_->poll(record, false);
    if (!status)
      return {std::get<Error>(status.state)};
    if (!finished(status.value()))
      return {Error{"event.not_quiescent", "release requires terminal state and stopped accesses"}};
  }
  {
    std::lock_guard lock(impl_->mutex);
    for (std::size_t i = 0; i < tokens.size(); ++i)
      if (impl_->lookup_locked(tokens[i]) != records[i])
        return {invalid_token()};
    for (const auto& token : tokens) {
      impl_->records.erase(token.id);
      auto at = impl_->invocations.find(token.invocation);
      if (!--at->second.count)
        impl_->invocations.erase(at);
    }
  }
  return {};
}
Result<CompletionToken> HostEventService::complete(const InvocationStamp& stamp, const EventStatus& status) {
  if (!valid_state(status.state) || !finished(status))
    return {Error{"event.not_quiescent", "complete requires a terminal state with stopped accesses"}};
  auto record = std::make_shared<Impl::Record>();
  record->status = status;
  record->observed = true;
  return impl_->mint(std::move(record), stamp);
}
Result<void> HostEventService::retain_until(const CompletionToken& token, std::shared_ptr<void> owner) {
  auto record = impl_->lookup(token);
  if (!record)
    return {invalid_token()};
  if (!owner)
    return {Error{"event.owner", "cannot retain a null owner"}};
  std::lock_guard lock(record->mutex);
  if (!record->status.accesses_stopped)
    record->retained.push_back(std::move(owner));
  return {};
}
Result<std::shared_ptr<HostEventReference>> HostEventService::retain_event(const CompletionToken& token) {
  auto record = impl_->lookup(token);
  if (!record)
    return {invalid_token()};
  return {std::shared_ptr<HostEventReference>(std::make_shared<Impl::Reference>(std::move(record)))};
}
std::size_t HostEventService::token_count() const {
  std::lock_guard lock(impl_->mutex);
  return impl_->records.size();
}
} // namespace vecops::memory::model
