// SPDX-FileCopyrightText: 2026 HenryRenYz and vecops contributors
// SPDX-License-Identifier: MIT
#include "vecops/memory/HostEvents.h"
#include <gtest/gtest.h>
#include <atomic>
#include <chrono>
#include <future>
#include <thread>

namespace vecops::memory::model {
namespace {
using namespace std::chrono_literals;
InvocationContext context(std::string id = "replay", std::uint64_t generation = 1) {
  return {{std::move(id), generation}, {}, {}};
}
CompletionToken value(Result<CompletionToken> result) {
  EXPECT_TRUE(result);
  return result ? std::move(result).value() : CompletionToken{};
}
struct Manual {
  std::atomic<EventState> state{EventState::pending};
  std::atomic<bool> stopped{false};
  std::atomic<unsigned> cancels{0}, waits{0};
  EventStatus status() const {
    const auto s = state.load();
    return {s, stopped.load(),
            s == EventState::failed ? std::optional<Error>{{"test.failure", "backend failed"}} : std::nullopt};
  }
  void finish(EventState s = EventState::succeeded) {
    state = s;
    stopped = true;
  }
};
HostEventCallbacks callbacks(const std::shared_ptr<Manual>& manual) {
  return {[manual]() -> Result<EventStatus> { return {manual->status()}; },
          [manual]() -> Result<EventStatus> {
            ++manual->waits;
            return {manual->status()};
          },
          [manual]() -> Result<void> {
            ++manual->cancels;
            return {};
          }};
}
CompletionToken import(HostEventService& service, const std::shared_ptr<Manual>& manual,
                       const InvocationContext& invocation = context(), std::shared_ptr<void> owner = {}) {
  return value(service.import_event({"host-callback", callbacks(manual), std::move(owner)}, invocation));
}
void release(HostEventService& service, const CompletionToken& token) {
  EXPECT_TRUE(service.release({&token, 1}));
}

TEST(HostEvents, FailureIsIndependentOfStoppedAndReleaseIsAtomic) {
  HostEventService service;
  auto manual = std::make_shared<Manual>();
  auto owner = std::make_shared<int>(4);
  std::weak_ptr<int> weak = owner;
  auto token = import(service, manual, context(), owner);
  owner.reset();
  manual->state = EventState::failed;
  auto failure = service.query(token);
  ASSERT_TRUE(failure);
  EXPECT_EQ(failure.value().state, EventState::failed);
  EXPECT_FALSE(failure.value().accesses_stopped);
  EXPECT_FALSE(weak.expired());
  const auto ready = value(service.complete({"other", 2}, {EventState::succeeded, true, {}}));
  const std::vector<CompletionToken> batch{ready, token};
  EXPECT_FALSE(service.release(batch));
  EXPECT_TRUE(service.query(ready));
  EXPECT_EQ(service.token_count(), 2u);
  ASSERT_TRUE(service.request_cancel(token));
  EXPECT_EQ(manual->cancels, 1u);
  EXPECT_FALSE(service.query(token).value().accesses_stopped);
  manual->stopped = true;
  auto done = service.wait(token);
  ASSERT_TRUE(done);
  EXPECT_EQ(done.value().state, EventState::failed);
  EXPECT_TRUE(done.value().accesses_stopped);
  EXPECT_TRUE(weak.expired());
  EXPECT_TRUE(service.release(batch));
  EXPECT_EQ(service.token_count(), 0u);
  EXPECT_FALSE(service.query(token));
}
TEST(HostEvents, JoinRetainsReleasedDependencyAndWaitsForAllAccesses) {
  HostEventService service;
  auto a = std::make_shared<Manual>(), b = std::make_shared<Manual>();
  const auto first = import(service, a, context("producer-A", 3)),
             second = import(service, b, context("producer-B", 7));
  const std::vector<CompletionToken> dependencies{first, second};
  const auto joined = value(service.join(dependencies, context("consumer", 8)));
  EXPECT_EQ(joined.invocation, "consumer");
  EXPECT_EQ(joined.binding_generation, 8u);
  a->state = EventState::failed;
  auto status = service.query(joined);
  ASSERT_TRUE(status);
  EXPECT_EQ(status.value().state, EventState::failed);
  EXPECT_FALSE(status.value().accesses_stopped);
  a->stopped = true;
  release(service, first);
  EXPECT_FALSE(service.query(first));
  EXPECT_EQ(service.token_count(), 2u);
  status = service.query(joined);
  ASSERT_TRUE(status);
  EXPECT_EQ(status.value().state, EventState::failed);
  EXPECT_FALSE(status.value().accesses_stopped);
  b->finish();
  status = service.wait(joined);
  ASSERT_TRUE(status);
  EXPECT_EQ(status.value().state, EventState::failed);
  EXPECT_TRUE(status.value().accesses_stopped);
  EXPECT_EQ(status.value().error->code, "test.failure");
  release(service, second);
  release(service, joined);
  EXPECT_EQ(service.token_count(), 0u);
}
TEST(HostEvents, TokenIdentityAndInvocationGenerationAreAuthenticated) {
  HostEventService service, other;
  const auto token = value(service.complete({"invocation", 4}, {EventState::succeeded, true, {}}));
  auto forged = token;
  forged.invocation = "elsewhere";
  EXPECT_FALSE(service.query(forged));
  EXPECT_FALSE(service.wait(forged));
  EXPECT_FALSE(service.request_cancel(forged));
  forged = token;
  forged.binding_generation = 5;
  EXPECT_FALSE(service.query(forged));
  EXPECT_FALSE(service.release({&forged, 1}));
  EXPECT_FALSE(service.join({&forged, 1}, context("consumer", 8)));
  EXPECT_FALSE(other.query(token));
  EXPECT_FALSE(service.complete({"invocation", 5}, {EventState::succeeded, true, {}}));
  const auto joined = value(service.join({&token, 1}, context("new-consumer", 12)));
  EXPECT_EQ(joined.binding_generation, 12u);
  EXPECT_TRUE(service.wait(joined));
  const std::vector<CompletionToken> duplicate{token, token};
  EXPECT_FALSE(service.release(duplicate));
  EXPECT_TRUE(service.query(token));
  auto invalid = token;
  invalid.id = "missing";
  const std::vector<CompletionToken> mixed{token, invalid};
  EXPECT_FALSE(service.release(mixed));
  EXPECT_TRUE(service.query(token));
  release(service, token);
  release(service, joined);
  EXPECT_FALSE(service.complete({"", 1}, {EventState::succeeded, true, {}}));
  EXPECT_FALSE(service.complete({"bad", 1}, {EventState::pending, true, {}}));
  EXPECT_FALSE(service.complete({"bad", 1}, {EventState::succeeded, false, {}}));
  EXPECT_FALSE(service.complete({"bad", 1}, {static_cast<EventState>(255), true, {}}));
}
TEST(HostEvents, CancelNoOpDoesNotStopOrDropRetainedOwners) {
  HostEventService service;
  auto manual = std::make_shared<Manual>();
  auto cb = callbacks(manual);
  cb.cancel = {};
  const auto token = value(service.import_event({"host-callback", cb, {}}, context()));
  auto retained = std::make_shared<int>(7);
  std::weak_ptr<int> weak = retained;
  ASSERT_TRUE(service.retain_until(token, retained));
  retained.reset();
  ASSERT_TRUE(service.request_cancel(token));
  EXPECT_FALSE(service.query(token).value().accesses_stopped);
  EXPECT_FALSE(weak.expired());
  EXPECT_FALSE(service.release({&token, 1}));
  manual->finish(EventState::cancelled);
  auto status = service.wait(token);
  ASSERT_TRUE(status);
  EXPECT_EQ(status.value().state, EventState::cancelled);
  EXPECT_TRUE(weak.expired());
  retained = std::make_shared<int>(8);
  weak = retained;
  EXPECT_TRUE(service.retain_until(token, retained));
  retained.reset();
  EXPECT_TRUE(weak.expired());
  release(service, token);
  EXPECT_FALSE(service.retain_until(token, std::make_shared<int>()));
}
TEST(HostEvents, JoinCancellationOnlyRequestsAndPreservesOtherPendingWork) {
  HostEventService service;
  auto a = std::make_shared<Manual>(), b = std::make_shared<Manual>();
  const std::vector<CompletionToken> deps{import(service, a), import(service, b)};
  const auto joined = value(service.join(deps, context("join", 2)));
  EXPECT_TRUE(service.request_cancel(joined));
  EXPECT_EQ(a->cancels, 1u);
  EXPECT_EQ(b->cancels, 1u);
  EXPECT_FALSE(service.query(joined).value().accesses_stopped);
  a->finish(EventState::cancelled);
  EXPECT_EQ(service.query(joined).value().state, EventState::cancelled);
  EXPECT_FALSE(service.query(joined).value().accesses_stopped);
  b->finish(EventState::failed);
  auto done = service.query(joined);
  ASSERT_TRUE(done);
  EXPECT_EQ(done.value().state, EventState::failed);
  EXPECT_TRUE(done.value().accesses_stopped);
  EXPECT_TRUE(service.release(deps));
  release(service, joined);
}
TEST(HostEvents, IndependentCallbacksProgressConcurrentlyWithoutRegistryLock) {
  HostEventService service;
  std::promise<void> release_callbacks;
  const auto unblock = release_callbacks.get_future().share();
  std::promise<void> entered_a, entered_b;
  auto ea = entered_a.get_future(), eb = entered_b.get_future();
  auto adapter = [&](std::promise<void>& entered) {
    return HostEventCallbacks{[&entered, unblock]() -> Result<EventStatus> {
                                entered.set_value();
                                unblock.wait();
                                return {EventStatus{EventState::succeeded, true, {}}};
                              },
                              []() -> Result<EventStatus> { return {EventStatus{EventState::succeeded, true, {}}}; },
                              {}};
  };
  const auto a = value(service.import_event({"host-callback", adapter(entered_a), {}}, context("a")));
  const auto b = value(service.import_event({"host-callback", adapter(entered_b), {}}, context("b")));
  auto fa = std::async(std::launch::async, [&] { return service.query(a); });
  auto fb = std::async(std::launch::async, [&] { return service.query(b); });
  const bool concurrent = ea.wait_for(1s) == std::future_status::ready && eb.wait_for(1s) == std::future_status::ready;
  release_callbacks.set_value();
  EXPECT_TRUE(concurrent);
  EXPECT_TRUE(fa.get());
  EXPECT_TRUE(fb.get());
  release(service, a);
  release(service, b);
}
TEST(HostEvents, SameRecordCallbacksSerializeAndReentryReportsError) {
  HostEventService service;
  std::atomic<unsigned> active{0}, maximum{0};
  auto manual = std::make_shared<Manual>();
  auto observe = [&]() -> Result<EventStatus> {
    const auto n = ++active;
    maximum.store(std::max(maximum.load(), n));
    std::this_thread::sleep_for(1ms);
    --active;
    return {manual->status()};
  };
  const auto token =
    value(service.import_event({"host-callback", HostEventCallbacks{observe, observe, {}}, {}}, context()));
  std::vector<std::future<Result<EventStatus>>> futures;
  for (unsigned i = 0; i < 8; ++i)
    futures.push_back(std::async(std::launch::async, [&] { return service.query(token); }));
  for (auto& future : futures)
    EXPECT_TRUE(future.get());
  EXPECT_EQ(maximum, 1u);
  manual->finish();
  EXPECT_TRUE(service.wait(token));
  release(service, token);
  CompletionToken recursive;
  std::atomic<bool> rejected{false};
  auto recurse = [&]() -> Result<EventStatus> {
    const auto inner = service.query(recursive);
    rejected = !inner && std::get<Error>(inner.state).code == "event.reentrant";
    return {EventStatus{EventState::succeeded, true, {}}};
  };
  recursive =
    value(service.import_event({"host-callback", HostEventCallbacks{recurse, recurse, {}}, {}}, context("recursive")));
  EXPECT_TRUE(service.query(recursive));
  EXPECT_TRUE(rejected);
  release(service, recursive);
}
TEST(HostEvents, CallbackReentryIntoAnotherBusyRecordDoesNotWait) {
  HostEventService service;
  std::promise<void> entered, unblock;
  auto entered_future = entered.get_future();
  auto gate = unblock.get_future().share();
  auto blocked = [&]() -> Result<EventStatus> {
    entered.set_value();
    gate.wait();
    return {EventStatus{EventState::succeeded, true, {}}};
  };
  const auto first =
    value(service.import_event({"host-callback", HostEventCallbacks{blocked, blocked, {}}, {}}, context("first")));
  auto running = std::async(std::launch::async, [&] { return service.query(first); });
  const bool started = entered_future.wait_for(1s) == std::future_status::ready;
  std::atomic<bool> refused_wait{false};
  auto observe = [&]() -> Result<EventStatus> {
    auto status = service.query(first);
    refused_wait = !status && std::get<Error>(status.state).code == "event.reentrant";
    return {EventStatus{EventState::succeeded, true, {}}};
  };
  const auto second =
    value(service.import_event({"host-callback", HostEventCallbacks{observe, observe, {}}, {}}, context("second")));
  auto reentry = std::async(std::launch::async, [&] { return service.query(second); });
  const bool returned = reentry.wait_for(1s) == std::future_status::ready;
  unblock.set_value();
  EXPECT_TRUE(started);
  EXPECT_TRUE(returned);
  EXPECT_TRUE(reentry.get());
  EXPECT_TRUE(running.get());
  EXPECT_TRUE(refused_wait);
  release(service, first);
  release(service, second);
}

TEST(HostEvents, CancellationCanInterruptBlockingWaitWithoutPublishingStoppedEarly) {
  HostEventService service;
  std::promise<void> waiting, cancelled, leave_cancel;
  auto waiting_future = waiting.get_future();
  auto cancel_signal = cancelled.get_future().share();
  auto cancel_exit = leave_cancel.get_future().share();
  auto owner = std::make_shared<int>();
  std::weak_ptr<int> weak = owner;
  auto observe = []() -> Result<EventStatus> { return {EventStatus{EventState::cancelled, true, {}}}; };
  HostEventCallbacks adapter{observe,
                             [&]() -> Result<EventStatus> {
                               waiting.set_value();
                               cancel_signal.wait();
                               return {EventStatus{EventState::cancelled, true, {}}};
                             },
                             [&]() -> Result<void> {
                               cancelled.set_value();
                               cancel_exit.wait();
                               return {};
                             }};
  const auto token = value(service.import_event({"host-callback", adapter, owner}, context()));
  owner.reset();
  auto waiter = std::async(std::launch::async, [&] { return service.wait(token); });
  EXPECT_EQ(waiting_future.wait_for(1s), std::future_status::ready);
  auto cancel = std::async(std::launch::async, [&] { return service.request_cancel(token); });
  const bool interrupted = waiter.wait_for(1s) == std::future_status::ready;
  if (interrupted) {
    auto status = waiter.get();
    EXPECT_TRUE(status);
    if (status) {
      EXPECT_FALSE(status.value().accesses_stopped);
    }
    EXPECT_FALSE(weak.expired());
    EXPECT_FALSE(service.release({&token, 1}));
  }
  leave_cancel.set_value();
  EXPECT_TRUE(interrupted);
  EXPECT_TRUE(cancel.get());
  if (!interrupted) {
    EXPECT_TRUE(waiter.get());
  }
  auto done = service.query(token);
  ASSERT_TRUE(done);
  EXPECT_TRUE(done.value().accesses_stopped);
  EXPECT_TRUE(weak.expired());
  release(service, token);
}

TEST(HostEvents, OwnerDestructionCanReenterServiceAfterStopped) {
  HostEventService service;
  CompletionToken token;
  std::atomic<bool> destructor_observed_stop{false};
  auto manual = std::make_shared<Manual>();
  std::shared_ptr<void> owner(new int(0), [&](void* pointer) {
    auto result = service.query(token);
    destructor_observed_stop = result && result.value().accesses_stopped;
    delete static_cast<int*>(pointer);
  });
  token = import(service, manual, context(), owner);
  owner.reset();
  manual->finish();
  EXPECT_TRUE(service.query(token));
  EXPECT_TRUE(destructor_observed_stop);
  release(service, token);
}
TEST(HostEvents, DestructorDrainsFailureWithoutTreatingCancellationAsProof) {
  auto service = std::make_unique<HostEventService>();
  auto manual = std::make_shared<Manual>();
  manual->state = EventState::failed;
  auto owner = std::make_shared<int>(2);
  std::weak_ptr<int> weak = owner;
  const auto token = import(*service, manual, context(), owner);
  owner.reset();
  ASSERT_TRUE(service->query(token));
  std::atomic<bool> destructor_returned{false};
  auto destruction = std::async(std::launch::async, [&] {
    service.reset();
    destructor_returned = true;
  });
  for (unsigned i = 0; i < 100 && manual->waits.load() == 0; ++i)
    std::this_thread::sleep_for(1ms);
  EXPECT_GT(manual->waits, 0u);
  EXPECT_FALSE(destructor_returned);
  EXPECT_FALSE(weak.expired());
  EXPECT_EQ(manual->cancels, 0u);
  manual->stopped = true;
  EXPECT_EQ(destruction.wait_for(1s), std::future_status::ready);
  destruction.get();
  EXPECT_TRUE(weak.expired());
}
TEST(HostEvents, AdapterErrorsExceptionsAndRegressionsDoNotAuthorizeRelease) {
  HostEventService service;
  auto manual = std::make_shared<Manual>();
  auto cb = callbacks(manual);
  cb.query = []() -> Result<EventStatus> { throw std::runtime_error("adapter exploded"); };
  const auto token = value(service.import_event({"host-callback", cb, {}}, context()));
  auto failed = service.query(token);
  ASSERT_FALSE(failed);
  EXPECT_EQ(std::get<Error>(failed.state).code, "event.adapter_exception");
  EXPECT_FALSE(service.release({&token, 1}));
  manual->finish();
  EXPECT_TRUE(service.wait(token));
  release(service, token);
  const auto regression = import(service, manual, context("regress"));
  manual->state = EventState::failed;
  manual->stopped = false;
  EXPECT_TRUE(service.query(regression));
  manual->state = EventState::pending;
  EXPECT_FALSE(service.query(regression));
  manual->finish(EventState::failed);
  EXPECT_TRUE(service.wait(regression));
  release(service, regression);
  EXPECT_FALSE(service.import_event({"unknown", HostEventCallbacks{}, {}}, context()));
  EXPECT_FALSE(service.import_event({"host-callback", 7, {}}, context()));
  EXPECT_FALSE(service.import_event({"host-callback", HostEventCallbacks{}, {}}, context()));
}
TEST(HostEvents, RetainedReferenceSurvivesTokenReleaseAndServiceDestruction) {
  auto service = std::make_unique<HostEventService>();
  auto manual = std::make_shared<Manual>();
  const auto token = import(*service, manual, context());
  auto forged = token;
  ++forged.binding_generation;
  EXPECT_FALSE(service->retain_event(forged));
  auto retained = service->retain_event(token);
  ASSERT_TRUE(retained);
  auto reference = std::move(retained).value();
  auto pending = reference->query();
  ASSERT_TRUE(pending);
  EXPECT_EQ(pending.value().state, EventState::pending);
  EXPECT_FALSE(service->release({&token, 1}));
  manual->finish(EventState::failed);
  auto stopped = reference->wait();
  ASSERT_TRUE(stopped);
  EXPECT_EQ(stopped.value().state, EventState::failed);
  EXPECT_TRUE(stopped.value().accesses_stopped);
  release(*service, token);
  EXPECT_EQ(service->token_count(), 0u);
  EXPECT_FALSE(service->query(token));
  EXPECT_FALSE(service->retain_event(token));
  EXPECT_TRUE(reference->query());
  service.reset();
  auto after = reference->wait();
  ASSERT_TRUE(after);
  EXPECT_EQ(after.value().state, EventState::failed);
  EXPECT_TRUE(after.value().accesses_stopped);
}
TEST(HostEvents, AdapterCopyExceptionsDoNotLeaveObservationOrCancellationBusy) {
  struct Observer {
    std::shared_ptr<std::atomic<bool>> fail;
    std::shared_ptr<Manual> manual;
    Observer(std::shared_ptr<std::atomic<bool>> f, std::shared_ptr<Manual> m)
      : fail(std::move(f)), manual(std::move(m)) {}
    Observer(const Observer& other) : fail(other.fail), manual(other.manual) {
      if (*fail)
        throw std::runtime_error("callback copy failed");
    }
    Result<EventStatus> operator()() const {
      return {manual->status()};
    }
  };
  HostEventService service;
  auto manual = std::make_shared<Manual>();
  auto fail = std::make_shared<std::atomic<bool>>(false);
  HostEventCallbacks cb{Observer{fail, manual}, Observer{fail, manual}, {}};
  const auto token = value(service.import_event({"host-callback", cb, {}}, context()));
  *fail = true;
  auto failed = service.query(token);
  ASSERT_FALSE(failed);
  EXPECT_EQ(std::get<Error>(failed.state).code, "event.observation_exception");
  auto cancelled = service.request_cancel(token);
  ASSERT_FALSE(cancelled);
  EXPECT_EQ(cancelled.error->code, "event.cancellation_exception");
  *fail = false;
  ASSERT_TRUE(service.query(token));
  EXPECT_TRUE(service.request_cancel(token));
  manual->finish();
  EXPECT_TRUE(service.wait(token));
  release(service, token);
}
TEST(HostEvents, RetainedReferenceDoesNotBypassServiceDestructorDrain) {
  auto service = std::make_unique<HostEventService>();
  auto manual = std::make_shared<Manual>();
  auto owner = std::make_shared<int>(1);
  std::weak_ptr<int> weak = owner;
  const auto token = import(*service, manual, context(), owner);
  auto retained = service->retain_event(token);
  ASSERT_TRUE(retained);
  auto reference = std::move(retained).value();
  owner.reset();
  auto destruction = std::async(std::launch::async, [&] { service.reset(); });
  for (unsigned i = 0; i < 100 && manual->waits.load() == 0; ++i)
    std::this_thread::sleep_for(1ms);
  EXPECT_GT(manual->waits, 0u);
  EXPECT_FALSE(weak.expired());
  manual->finish();
  EXPECT_EQ(destruction.wait_for(1s), std::future_status::ready);
  destruction.get();
  EXPECT_TRUE(weak.expired());
  EXPECT_TRUE(reference->query());
}
TEST(HostEvents, ExplicitReleaseKeepsRegistryBoundedAcrossManyReplays) {
  HostEventService service;
  CompletionToken old;
  for (unsigned i = 0; i < 100000; ++i) {
    const auto token = value(service.complete({"replay-" + std::to_string(i), i}, {EventState::succeeded, true, {}}));
    EXPECT_EQ(service.token_count(), 1u);
    release(service, token);
    EXPECT_EQ(service.token_count(), 0u);
    if (i == 0)
      old = token;
  }
  EXPECT_FALSE(service.query(old));
  const auto empty = value(service.join({}, context("empty")));
  auto status = service.query(empty);
  ASSERT_TRUE(status);
  EXPECT_EQ(status.value().state, EventState::succeeded);
  EXPECT_TRUE(status.value().accesses_stopped);
  release(service, empty);
}
} // namespace
} // namespace vecops::memory::model
