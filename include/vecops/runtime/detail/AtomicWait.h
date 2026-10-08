// SPDX-FileCopyrightText: 2026 HenryRenYz and vecops contributors
// SPDX-License-Identifier: MIT
// Copyright (c) vecops contributors.

#ifndef VECOPS_RUNTIME_DETAIL_ATOMIC_WAIT_H
#define VECOPS_RUNTIME_DETAIL_ATOMIC_WAIT_H

#include <atomic>
#include <condition_variable>
#include <cstdint>
#include <mutex>
#include <version>

#if defined(__linux__)
#  include <cerrno>
#  include <linux/futex.h>
#  include <sys/syscall.h>
#  include <unistd.h>
#endif

#if defined(__cpp_lib_atomic_wait) && __cpp_lib_atomic_wait >= 201907L && \
    !defined(VECOPS_FORCE_PORTABLE_ATOMIC_WAIT)
#  define VECOPS_RUNTIME_HAS_ATOMIC_WAIT 1
#else
#  define VECOPS_RUNTIME_HAS_ATOMIC_WAIT 0
#endif

namespace vecops::runtime::detail {

/** A 32-bit generation word with C++20 atomic-wait or portable parking.
 *
 * On Linux without library atomic-wait support, a lock-free aligned 32-bit
 * atomic is used directly with private futexes. Other platforms park through
 * a condition variable. The condition-variable path always tests the value
 * while holding the same mutex used by notify, so notifications cannot be
 * lost. Callers must update the word before notifying.
 */
class AtomicWait32 {
public:
  explicit AtomicWait32(std::uint32_t initial = 0) noexcept : value_(initial) {}

  std::uint32_t load(std::memory_order order = std::memory_order_seq_cst) const noexcept {
    return value_.load(order);
  }

  void store(std::uint32_t value,
             std::memory_order order = std::memory_order_seq_cst) noexcept {
    value_.store(value, order);
  }

  std::uint32_t fetch_add(
      std::uint32_t value,
      std::memory_order order = std::memory_order_seq_cst) noexcept {
    return value_.fetch_add(value, order);
  }

  void wait(std::uint32_t old,
            std::memory_order order = std::memory_order_seq_cst) const noexcept {
#if VECOPS_RUNTIME_HAS_ATOMIC_WAIT
    value_.wait(old, order);
#elif defined(__linux__) && ATOMIC_INT_LOCK_FREE == 2 && \
    !defined(VECOPS_FORCE_CV_ATOMIC_WAIT)
    static_assert(std::atomic<std::uint32_t>::is_always_lock_free,
                  "futex wait requires a lock-free 32-bit atomic");
    while (value_.load(order) == old) {
      const auto result = ::syscall(
          SYS_futex, futex_address(), FUTEX_WAIT_PRIVATE, old,
          nullptr, nullptr, 0);
      if (result == 0 || errno == EAGAIN || errno == EINTR)
        continue;
      wait_with_condition_variable(old, order);
      return;
    }
#else
    wait_with_condition_variable(old, order);
#endif
  }

  void notify_all() noexcept {
#if VECOPS_RUNTIME_HAS_ATOMIC_WAIT
    value_.notify_all();
#elif defined(__linux__) && ATOMIC_INT_LOCK_FREE == 2 && \
    !defined(VECOPS_FORCE_CV_ATOMIC_WAIT)
    const auto result = ::syscall(SYS_futex, futex_address(), FUTEX_WAKE_PRIVATE,
                                  0x7fffffff, nullptr, nullptr, 0);
    // This zero-increment RMW pairs with CV wait registration. If notify sees
    // a registered waiter, it takes the CV mutex before notifying. If notify
    // sees zero, a later registrant acquires this RMW, so the preceding
    // generation update happens-before its predicate check. The normal futex
    // path remains mutex-free, without assuming WAIT/WAKE errno symmetry.
    const auto cv_waiters = cv_waiters_.fetch_add(0, std::memory_order_acq_rel);
    if (result < 0 || cv_waiters != 0) {
      std::lock_guard lock(wait_mutex_);
      wait_cv_.notify_all();
    }
#else
    {
      std::lock_guard lock(wait_mutex_);
      wait_cv_.notify_all();
    }
#endif
  }

private:
#if defined(__linux__) && !VECOPS_RUNTIME_HAS_ATOMIC_WAIT && \
    ATOMIC_INT_LOCK_FREE == 2 && !defined(VECOPS_FORCE_CV_ATOMIC_WAIT)
  std::uint32_t* futex_address() const noexcept {
    // The kernel treats uaddr as an aligned 32-bit word; C++ accesses remain
    // exclusively through value_ and the atomic's representation is checked.
    static_assert(sizeof(value_) == sizeof(std::uint32_t));
    return reinterpret_cast<std::uint32_t*>(
        const_cast<std::atomic<std::uint32_t>*>(&value_));
  }
#endif

  void wait_with_condition_variable(
      std::uint32_t old, std::memory_order order) const noexcept {
    std::unique_lock lock(wait_mutex_);
#if defined(__linux__) && !VECOPS_RUNTIME_HAS_ATOMIC_WAIT && \
    ATOMIC_INT_LOCK_FREE == 2 && !defined(VECOPS_FORCE_CV_ATOMIC_WAIT)
    cv_waiters_.fetch_add(1, std::memory_order_acq_rel);
#endif
    wait_cv_.wait(lock, [&] { return value_.load(order) != old; });
#if defined(__linux__) && !VECOPS_RUNTIME_HAS_ATOMIC_WAIT && \
    ATOMIC_INT_LOCK_FREE == 2 && !defined(VECOPS_FORCE_CV_ATOMIC_WAIT)
    cv_waiters_.fetch_sub(1, std::memory_order_acq_rel);
#endif
  }

  alignas(4) std::atomic<std::uint32_t> value_;
  mutable std::mutex wait_mutex_;
  mutable std::condition_variable wait_cv_;
#if defined(__linux__) && !VECOPS_RUNTIME_HAS_ATOMIC_WAIT && ATOMIC_INT_LOCK_FREE == 2
  mutable std::atomic<std::uint32_t> cv_waiters_{0};
#endif
};

} // namespace vecops::runtime::detail

#undef VECOPS_RUNTIME_HAS_ATOMIC_WAIT

#endif // VECOPS_RUNTIME_DETAIL_ATOMIC_WAIT_H
