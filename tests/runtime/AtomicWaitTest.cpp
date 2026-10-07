// Copyright (c) vecops contributors.

#include "vecops/runtime/detail/AtomicWait.h"

#include <gtest/gtest.h>

#include <atomic>
#include <cstdint>
#include <thread>

namespace {

using vecops::runtime::detail::AtomicWait32;

TEST(AtomicWaitTest, NotifyBeforeWaitDoesNotBlock) {
  AtomicWait32 generation{0};
  generation.store(1, std::memory_order_release);
  generation.notify_all();
  generation.wait(0, std::memory_order_acquire);
  EXPECT_EQ(generation.load(std::memory_order_acquire), 1u);
}

TEST(AtomicWaitTest, WaiterParksUntilGenerationChanges) {
  AtomicWait32 generation{0};
  std::atomic<bool> entered{false};
  std::atomic<bool> returned{false};
  std::thread waiter([&] {
    entered.store(true, std::memory_order_release);
    generation.wait(0, std::memory_order_acquire);
    returned.store(true, std::memory_order_release);
  });

  while (!entered.load(std::memory_order_acquire))
    std::this_thread::yield();
  generation.store(1, std::memory_order_release);
  generation.notify_all();
  waiter.join();
  EXPECT_TRUE(returned.load(std::memory_order_acquire));
}

TEST(AtomicWaitTest, RepeatedGenerationsWakeAllWaiters) {
  constexpr std::uint32_t iterations = 1000;
  AtomicWait32 generation{0};
  std::atomic<std::uint32_t> completed{0};
  std::thread waiter([&] {
    for (std::uint32_t expected = 0; expected < iterations; ++expected) {
      generation.wait(expected, std::memory_order_acquire);
      completed.fetch_add(1, std::memory_order_relaxed);
    }
  });

  for (std::uint32_t next = 1; next <= iterations; ++next) {
    generation.store(next, std::memory_order_release);
    generation.notify_all();
  }
  waiter.join();
  EXPECT_EQ(completed.load(std::memory_order_relaxed), iterations);
}

} // namespace
