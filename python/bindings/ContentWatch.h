// SPDX-FileCopyrightText: 2026 HenryRenYz and vecops contributors
// SPDX-License-Identifier: MIT
// Mutation-based hash-cache acceleration. Linux dnotify is a fallback when
// the account's inotify quota is exhausted. Never performs work in a signal
// handler beyond incrementing one lock-free atomic; no Python/GIL callbacks.
#pragma once
#include <algorithm>
#include <atomic>
#include <cstdint>
#include <filesystem>
#include <map>
#include <mutex>
#if defined(__linux__)
#  include <csignal>
#  include <fcntl.h>
#  include <unistd.h>
#  include <sys/resource.h>
#  include <sys/stat.h>
#endif

namespace {
class ContentWatch {
  std::atomic<std::uint64_t> generation_{0};
  std::mutex mutex_;
  std::map<std::string, int> descriptors_;
  int signal_ = -1;
#if defined(__linux__)
  const pid_t owner_ = ::getpid();
  static ContentWatch* instance_;
  static void changed(int) noexcept {
    instance_->generation_.fetch_add(1, std::memory_order_relaxed);
  }
  static std::size_t descriptor_budget() noexcept {
    struct rlimit limit {};
    if (::getrlimit(RLIMIT_NOFILE, &limit))
      return 0;
    // dnotify needs one persistent FD per directory. Leave most of the
    // process limit available to compilers, frameworks and failure reporting.
    return static_cast<std::size_t>(std::min<rlim_t>(128, limit.rlim_cur / 8));
  }
#endif
public:
  bool add(const std::filesystem::path& path) {
#if defined(__linux__)
    static_assert(std::atomic<std::uint64_t>::is_always_lock_free);
    std::lock_guard guard(mutex_);
    if (::getpid() != owner_)
      return false;
    const auto budget = descriptor_budget();
    if (descriptors_.size() > budget) {
      // A lowered limit must not leave cached hashes backed by closed watches.
      for (const auto& [_, descriptor] : descriptors_)
        ::close(descriptor);
      descriptors_.clear();
      generation_.fetch_add(1, std::memory_order_relaxed);
    }
    if (signal_ < 0) {
      struct sigaction overflow {};
      if (::sigaction(SIGIO, nullptr, &overflow) || overflow.sa_handler != SIG_DFL)
        return false;
      overflow.sa_handler = changed;
      ::sigemptyset(&overflow.sa_mask);
      overflow.sa_flags = SA_RESTART;
      instance_ = this;
      if (::sigaction(SIGIO, &overflow, nullptr))
        return false;
      for (int candidate = SIGRTMAX; candidate >= SIGRTMIN; --candidate) {
        struct sigaction prior {};
        if (::sigaction(candidate, nullptr, &prior) || prior.sa_handler != SIG_DFL)
          continue;
        struct sigaction action {};
        action.sa_handler = changed;
        ::sigemptyset(&action.sa_mask);
        action.sa_flags = SA_RESTART;
        instance_ = this;
        if (!::sigaction(candidate, &action, nullptr)) {
          signal_ = candidate;
          break;
        }
      }
    }
    if (signal_ < 0)
      return false;
    if (const auto found = descriptors_.find(path.string()); found != descriptors_.end()) {
      struct stat watched {
      }, current{};
      if (!::fstat(found->second, &watched) && !::stat(path.c_str(), &current) && watched.st_dev == current.st_dev &&
          watched.st_ino == current.st_ino)
        return true;
      ::close(found->second);
      descriptors_.erase(found);
      generation_.fetch_add(1, std::memory_order_relaxed);
    }
    if (descriptors_.size() >= budget)
      return false; // Unwatched paths are hashed from content, never trusted.
    const int descriptor = ::open(path.c_str(), O_RDONLY | O_DIRECTORY | O_CLOEXEC);
    if (descriptor < 0)
      return false;
    if (::fcntl(descriptor, F_SETOWN, owner_) < 0 || ::fcntl(descriptor, F_SETSIG, signal_) < 0 ||
        ::fcntl(descriptor, F_NOTIFY, DN_MODIFY | DN_CREATE | DN_DELETE | DN_RENAME | DN_ATTRIB | DN_MULTISHOT) < 0) {
      ::close(descriptor);
      return false;
    }
    descriptors_.emplace(path.string(), descriptor);
    return true;
#else
    return false;
#endif
  }
  std::int64_t epoch() const {
#if defined(__linux__)
    if (::getpid() != owner_)
      return -1;
    if (signal_ >= 0) {
      struct sigaction current {};
      if (::sigaction(signal_, nullptr, &current) || current.sa_handler != changed)
        return -1;
      if (::sigaction(SIGIO, nullptr, &current) || current.sa_handler != changed)
        return -1;
    }
    return static_cast<std::int64_t>(generation_.load(std::memory_order_relaxed));
#else
    return -1;
#endif
  }
};
#if defined(__linux__)
ContentWatch* ContentWatch::instance_ = nullptr;
#endif
ContentWatch content_watch;
} // namespace
