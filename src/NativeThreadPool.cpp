// SPDX-FileCopyrightText: 2026 HenryRenYz and vecops contributors
// SPDX-License-Identifier: MIT

#include "vecops/runtime/CallAbi.h"
#include "vecops/runtime/detail/AtomicWait.h"

#include <algorithm>
#include <atomic>
#include <bit>
#include <cerrno>
#include <condition_variable>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <chrono>
#include <exception>
#include <memory>
#include <limits>
#include <map>
#include <fstream>
#include <set>
#include <mutex>
#include <stdexcept>
#include <string>
#include <thread>
#include <utility>
#include <vector>

#if defined(__linux__)
#  include <pthread.h>
#  include <sched.h>
#  include <unistd.h>
#endif

#if defined(__x86_64__) || defined(_M_X64)
#  include <immintrin.h>
#endif

namespace {

thread_local bool inside_native_pool = false;
thread_local std::uint32_t cpu_thread_budget = 0;

inline void cpu_relax() noexcept {
#if defined(__aarch64__) || defined(__arm__)
  __asm__ __volatile__("yield" ::: "memory");
#elif defined(__x86_64__) || defined(_M_X64)
  _mm_pause();
#else
  std::atomic_signal_fence(std::memory_order_seq_cst);
#endif
}

void set_error(VecopsError* error, const char* message) noexcept {
  if (error == nullptr || error->struct_size < sizeof(VecopsError))
    return;
  error->code = VECOPS_STATUS_EXECUTION_ERROR;
  const auto size = std::strlen(message);
  error->message_required = size + 1;
  if (error->message == nullptr || error->message_capacity == 0)
    return;
  const auto copied = std::min(size, error->message_capacity - 1);
  std::memcpy(error->message, message, copied);
  error->message[copied] = '\0';
}

std::uint32_t configured_spin_count() noexcept {
  constexpr std::uint32_t default_spin_count = 256;
  const char* text = std::getenv("VECOPS_THREAD_POOL_SPIN_COUNT");
  if (text == nullptr || *text == '\0')
    return default_spin_count;
  char* end = nullptr;
  errno = 0;
  const auto parsed = std::strtoul(text, &end, 10);
  if (errno != 0 || end == text || *end != '\0')
    return default_spin_count;
  return static_cast<std::uint32_t>(std::min<unsigned long>(parsed, 1'000'000UL));
}

std::vector<int> allowed_cpus() {
  std::vector<int> result;
#if defined(__linux__)
  cpu_set_t set;
  CPU_ZERO(&set);
  if (sched_getaffinity(0, sizeof(set), &set) == 0) {
    for (int cpu = 0; cpu < CPU_SETSIZE; ++cpu) {
      if (CPU_ISSET(cpu, &set))
        result.push_back(cpu);
    }
  }
#endif
  return result;
}

std::vector<int> parse_cpu_list(const char* text) {
  std::vector<int> result;
#if defined(__linux__)
  if (text == nullptr || *text == '\0')
    return result;
  const char* cursor = text;
  while (*cursor != '\0') {
    char* end = nullptr;
    errno = 0;
    const auto first = std::strtol(cursor, &end, 10);
    if (errno != 0 || end == cursor || first < 0 || first >= CPU_SETSIZE)
      return {};
    auto last = first;
    if (*end == '-') {
      cursor = end + 1;
      errno = 0;
      last = std::strtol(cursor, &end, 10);
      if (errno != 0 || end == cursor || last < first || last >= CPU_SETSIZE)
        return {};
    }
    for (auto cpu = first; cpu <= last; ++cpu) {
      const auto value = static_cast<int>(cpu);
      if (std::find(result.begin(), result.end(), value) == result.end())
        result.push_back(value);
    }
    if (*end == '\0')
      break;
    if (*end != ',')
      return {};
    cursor = end + 1;
  }
#else
  (void)text;
#endif
  return result;
}

std::vector<int> pool_cpus(bool constrain_to_caller = false) {
  const char* configured = std::getenv("VECOPS_THREAD_POOL_CPU_LIST");
  if (configured == nullptr || *configured == '\0')
    return allowed_cpus();
  auto result = parse_cpu_list(configured);
  if (result.empty())
    throw std::runtime_error("invalid VECOPS_THREAD_POOL_CPU_LIST");
  const auto allowed = allowed_cpus();
  if (constrain_to_caller && std::any_of(result.begin(), result.end(), [&](int cpu) {
        return !allowed.empty() && std::find(allowed.begin(), allowed.end(), cpu) == allowed.end();
      }))
    throw std::runtime_error("VECOPS_THREAD_POOL_CPU_LIST exceeds caller CPU affinity");
  return result;
}

std::vector<int> physical_first(std::vector<int> cpus) {
#if defined(__linux__)
  static std::mutex mutex;
  static std::map<std::vector<int>, std::vector<int>> cache;
  std::lock_guard lock(mutex);
  if (const auto found = cache.find(cpus); found != cache.end())
    return found->second;
  const auto key = cpus;
  std::set<std::pair<int, int>> cores;
  std::vector<int> first, siblings;
  for (int cpu : cpus) {
    int package = -1, core = -1;
    const auto base = "/sys/devices/system/cpu/cpu" + std::to_string(cpu) + "/topology/";
    std::ifstream(base + "physical_package_id") >> package;
    std::ifstream(base + "core_id") >> core;
    if (package < 0 || core < 0)
      return cpus;
    (cores.emplace(package, core).second ? first : siblings).push_back(cpu);
  }
  first.insert(first.end(), siblings.begin(), siblings.end());
  cache.emplace(key, first);
  return first;
#else
  return cpus;
#endif
}

void pin_worker(const std::vector<int>& cpus, std::uint32_t ordinal) noexcept {
#if defined(__linux__)
  if (cpus.empty())
    return;
  cpu_set_t set;
  CPU_ZERO(&set);
  CPU_SET(cpus[static_cast<std::size_t>(ordinal) % cpus.size()], &set);
  (void)pthread_setaffinity_np(pthread_self(), sizeof(set), &set);
#else
  (void)cpus;
  (void)ordinal;
#endif
}

class CallerAffinity {
public:
  explicit CallerAffinity(const std::vector<int>& cpus) noexcept {
#if defined(__linux__)
    // Worker zero is the caller. Do not accidentally share its current CPU
    // with an explicitly pinned worker. Restore the embedding's full mask.
    if (!cpus.empty() && sched_getcpu() != cpus.front() &&
        pthread_getaffinity_np(pthread_self(), sizeof(previous_), &previous_) == 0) {
      cpu_set_t selected;
      CPU_ZERO(&selected);
      CPU_SET(cpus.front(), &selected);
      changed_ = pthread_setaffinity_np(pthread_self(), sizeof(selected), &selected) == 0;
    }
#else
    (void)cpus;
#endif
  }
  ~CallerAffinity() {
#if defined(__linux__)
    if (changed_)
      (void)pthread_setaffinity_np(pthread_self(), sizeof(previous_), &previous_);
#endif
  }

private:
#if defined(__linux__)
  cpu_set_t previous_{};
  bool changed_ = false;
#endif
};

class NativeThreadPool {
  struct TeamBarrier {
    explicit TeamBarrier(std::uint32_t count)
      : count(count) {
    }
    static std::int32_t wait(void* opaque, VecopsError* error) {
      auto& self = *static_cast<TeamBarrier*>(opaque);
      const auto generation = self.generation.load(std::memory_order_acquire);
      if (self.cancelled.load(std::memory_order_acquire)) {
        set_error(error, "native grid team cancelled");
        return VECOPS_STATUS_EXECUTION_ERROR;
      }
      if (self.arrived.fetch_add(1, std::memory_order_acq_rel) + 1 == self.count) {
        self.arrived.store(0, std::memory_order_relaxed);
        self.generation.fetch_add(1, std::memory_order_release);
        self.generation.notify_all();
      } else {
        for (unsigned spin = 0; spin < 2048 && self.generation.load(std::memory_order_acquire) == generation; ++spin)
          cpu_relax();
        self.generation.wait(generation, std::memory_order_acquire);
      }
      if (!self.cancelled.load(std::memory_order_acquire))
        return VECOPS_STATUS_OK;
      set_error(error, "native grid team cancelled");
      return VECOPS_STATUS_EXECUTION_ERROR;
    }
    void cancel() {
      cancelled.store(true, std::memory_order_release);
      self_wake();
    }
    void self_wake() {
      generation.fetch_add(1, std::memory_order_release);
      generation.notify_all();
    }
    const std::uint32_t count;
    std::atomic<std::uint32_t> arrived{0};
    vecops::runtime::detail::AtomicWait32 generation{0};
    std::atomic<bool> cancelled{false};
  };
  struct Submission {
    std::uint32_t task_count;
    std::uint32_t participants;
    void* context;
    VecopsParallelTaskFn body;
    VecopsTeamBodyFn team_body;
    std::atomic<std::uint32_t> remaining;
    std::mutex exception_mutex;
    std::exception_ptr exception;
    TeamBarrier barrier;
  };
  struct Worker {
    std::atomic<Submission*> inbox{nullptr};
    std::mutex mutex;
    std::condition_variable cv;
    std::thread thread;
  };

public:
  explicit NativeThreadPool(std::uint32_t threads, std::vector<int> cpus = pool_cpus())
    : threads_(std::max<std::uint32_t>(1, threads))
    , spin_count_(configured_spin_count())
    , cpus_(std::move(cpus)) {
    workers_.reserve(threads_ - 1);
    try {
      for (std::uint32_t ordinal = 1; ordinal < threads_; ++ordinal) {
        auto worker = std::make_unique<Worker>();
        auto* address = worker.get();
        worker->thread = std::thread([this, ordinal, address] { worker_loop(ordinal, *address); });
        workers_.push_back(std::move(worker));
      }
    } catch (...) {
      stop();
      throw;
    }
    while (ready_.load(std::memory_order_acquire) != threads_ - 1)
      std::this_thread::yield();
    provider_ = VecopsThreadPoolV1{sizeof(VecopsThreadPoolV1),
                                   VECOPS_THREAD_POOL_ABI_MAJOR,
                                   VECOPS_THREAD_POOL_ABI_MINOR,
                                   UINT64_C(0x6e6174697665),
                                   this,
                                   maximum,
                                   in_parallel,
                                   parallel_for,
                                   nullptr,
                                   nullptr,
                                   0,
                                   &team_executor_};
  }
  NativeThreadPool(const NativeThreadPool&) = delete;
  NativeThreadPool& operator=(const NativeThreadPool&) = delete;
  ~NativeThreadPool() {
    stop();
  }
  const VecopsThreadPoolV1* provider() const noexcept {
    return &provider_;
  }

  void begin_active() noexcept {
    std::lock_guard lock(submission_mutex_);
    if (active_scopes_.fetch_add(1, std::memory_order_acq_rel) == 0)
      wake_workers();
  }
  void end_active() noexcept {
    auto active = active_scopes_.load(std::memory_order_acquire);
    while (active != 0 && !active_scopes_.compare_exchange_weak(active, active - 1, std::memory_order_acq_rel,
                                                                std::memory_order_acquire)) {
    }
  }

private:
  void wake_workers() {
    for (auto& worker : workers_) {
      std::lock_guard lock(worker->mutex);
      worker->cv.notify_one();
    }
  }
  void stop() {
    stopping_.store(true, std::memory_order_release);
    wake_workers();
    for (auto& worker : workers_)
      if (worker->thread.joinable())
        worker->thread.join();
  }
  static std::uint32_t maximum(void* context) noexcept {
    return inside_native_pool ? 1 : static_cast<NativeThreadPool*>(context)->threads_;
  }
  static std::uint32_t in_parallel(void*) noexcept {
    return inside_native_pool ? 1 : 0;
  }
  static std::int32_t parallel_for(void* context, std::uint32_t tasks, void* opaque, VecopsParallelTaskFn body,
                                   VecopsError* error) noexcept {
    if (body == nullptr || tasks == 0)
      return VECOPS_STATUS_OK;
    try {
      if (tasks == 1 || inside_native_pool) {
        for (std::uint32_t task = 0; task < tasks; ++task)
          body(opaque, task, tasks);
      } else {
        static_cast<NativeThreadPool*>(context)->run(tasks, opaque, body, nullptr);
      }
      return VECOPS_STATUS_OK;
    } catch (const std::exception& exception) {
      set_error(error, exception.what());
    } catch (...) {
      set_error(error, "native thread pool failed");
    }
    return VECOPS_STATUS_EXECUTION_ERROR;
  }
  static std::int32_t run_team(void* context, std::uint32_t workers, void* opaque, VecopsTeamBodyFn body,
                               VecopsError* error) noexcept {
    if (body == nullptr)
      return VECOPS_STATUS_OK;
    try {
      if (workers <= 1 || inside_native_pool) {
        const VecopsTeamContextV1 team{sizeof(VecopsTeamContextV1), 0, 1, nullptr, nullptr};
        body(opaque, &team);
      } else {
        static_cast<NativeThreadPool*>(context)->run(workers, opaque, nullptr, body);
      }
      return VECOPS_STATUS_OK;
    } catch (const std::exception& exception) {
      set_error(error, exception.what());
    } catch (...) {
      set_error(error, "native grid team failed");
    }
    return VECOPS_STATUS_EXECUTION_ERROR;
  }
  inline static const VecopsTeamExecutorV1 team_executor_{sizeof(VecopsTeamExecutorV1), 1, run_team};

  void run(std::uint32_t tasks, void* opaque, VecopsParallelTaskFn body, VecopsTeamBodyFn team_body) {
#if defined(__linux__)
    if (process_id_ != getpid())
      throw std::runtime_error("native executor inherited across fork; use multiprocessing spawn/forkserver");
#endif
    std::lock_guard lock(submission_mutex_);
    CallerAffinity affinity(cpus_);
    const auto participants = std::min(tasks, threads_);
    // Only selected workers receive a mailbox. Idle workers never observe
    // the stack-backed submission, so its lifetime needs no idle-worker ack.
    Submission submission{
      tasks, participants, opaque, body, team_body, participants, {}, nullptr, TeamBarrier(participants)};
    for (std::uint32_t ordinal = 1; ordinal < participants; ++ordinal) {
      auto& worker = *workers_[ordinal - 1];
      std::lock_guard lock(worker.mutex);
      worker.inbox.store(&submission, std::memory_order_release);
      worker.cv.notify_one();
    }
    run_participant(submission, 0);
    submission.remaining.fetch_sub(1, std::memory_order_acq_rel);
    std::uint32_t spins = 0;
    while (submission.remaining.load(std::memory_order_acquire) != 0) {
      cpu_relax();
      if (++spins == 4096) {
        spins = 0;
        std::this_thread::yield();
      }
    }
    if (submission.exception != nullptr)
      std::rethrow_exception(submission.exception);
  }
  void run_participant(Submission& submission, std::uint32_t ordinal) noexcept {
    inside_native_pool = true;
    try {
      if (submission.team_body != nullptr) {
        const VecopsTeamContextV1 team{sizeof(VecopsTeamContextV1), ordinal, submission.participants,
                                       &submission.barrier, TeamBarrier::wait};
        submission.team_body(submission.context, &team);
      } else {
        for (std::uint64_t task = ordinal; task < submission.task_count; task += submission.participants)
          submission.body(submission.context, static_cast<std::uint32_t>(task), submission.task_count);
      }
    } catch (...) {
      {
        std::lock_guard lock(submission.exception_mutex);
        if (submission.exception == nullptr)
          submission.exception = std::current_exception();
      }
      submission.barrier.cancel();
    }
    inside_native_pool = false;
  }
  void worker_loop(std::uint32_t ordinal, Worker& worker) noexcept {
    pin_worker(cpus_, ordinal);
    ready_.fetch_add(1, std::memory_order_release);
    while (true) {
      auto* submission = worker.inbox.exchange(nullptr, std::memory_order_acquire);
      for (std::uint32_t spin = 0; submission == nullptr && spin < spin_count_; ++spin) {
        cpu_relax();
        submission = worker.inbox.exchange(nullptr, std::memory_order_acquire);
      }
      if (submission == nullptr) {
        if (stopping_.load(std::memory_order_acquire))
          return;
        if (active_scopes_.load(std::memory_order_acquire) != 0)
          continue;
        std::unique_lock lock(worker.mutex);
        worker.cv.wait(lock, [&] {
          return worker.inbox.load(std::memory_order_acquire) != nullptr || stopping_.load(std::memory_order_acquire) ||
                 active_scopes_.load(std::memory_order_acquire) != 0;
        });
        continue;
      }
      run_participant(*submission, ordinal);
      submission->remaining.fetch_sub(1, std::memory_order_acq_rel);
    }
  }

  const std::uint32_t threads_;
  const std::uint32_t spin_count_;
  const std::vector<int> cpus_;
  std::vector<std::unique_ptr<Worker>> workers_;
  std::mutex submission_mutex_;
  std::atomic<std::uint32_t> ready_{0};
  std::atomic<std::uint32_t> active_scopes_{0};
  std::atomic<bool> stopping_{false};
  VecopsThreadPoolV1 provider_{};
#if defined(__linux__)
  const pid_t process_id_ = getpid();
#endif
};
std::once_flag native_pool_once;
std::unique_ptr<NativeThreadPool> native_pool;

NativeThreadPool& get_native_pool(std::uint32_t threads) {
  std::call_once(native_pool_once, [&] { native_pool = std::make_unique<NativeThreadPool>(threads); });
  return *native_pool;
}

// The adaptive facade is separate from the legacy first-call-fixed factory.
// Its pointer is stable; sleeping pools are reused when execution budgets change.
std::uint32_t cgroup_capacity() {
#if defined(__linux__)
  // Cache only the quota (one second), never the caller's changing affinity.
  static std::mutex mutex;
  static auto refreshed = std::chrono::steady_clock::time_point::min();
  static auto capacity = std::numeric_limits<std::uint32_t>::max();
  std::lock_guard lock(mutex);
  const auto now = std::chrono::steady_clock::now();
  if (refreshed != std::chrono::steady_clock::time_point::min() && now - refreshed < std::chrono::seconds(1))
    return capacity;
  capacity = std::numeric_limits<std::uint32_t>::max();
  auto read_quota = [&](const std::string& path) {
    std::string quota;
    std::uint64_t period = 0;
    std::ifstream(path + "/cpu.max") >> quota >> period;
    if (quota.empty() || quota == "max" || period == 0)
      return;
    char* end = nullptr;
    errno = 0;
    const auto value = std::strtoull(quota.c_str(), &end, 10);
    if (errno || *end || value == 0)
      return;
    const auto workers = value / period + (value % period != 0);
    capacity = std::min<std::uint64_t>(capacity, std::max<std::uint64_t>(1, workers));
  };
  // Namespace root plus every visible v2 ancestor. No writes to cgroups.
  read_quota("/sys/fs/cgroup");
  std::string entry;
  std::ifstream membership("/proc/self/cgroup");
  while (std::getline(membership, entry)) {
    if (!entry.starts_with("0::/"))
      continue;
    auto relative = entry.substr(4);
    if (relative.find("..") != std::string::npos)
      continue;
    while (!relative.empty()) {
      read_quota("/sys/fs/cgroup/" + relative);
      const auto parent = relative.rfind('/');
      if (parent == std::string::npos)
        break;
      relative.resize(parent);
    }
  }
  refreshed = now;
  return capacity;
#else
  return std::numeric_limits<std::uint32_t>::max();
#endif
}

std::uint32_t cpu_capacity(void*) noexcept {
  if (inside_native_pool)
    return 1;
  try {
    const auto cpus = pool_cpus(true);
    const auto allowed = std::min(cgroup_capacity(), cpus.empty() ? std::max(1u, std::thread::hardware_concurrency())
                                                                  : static_cast<std::uint32_t>(cpus.size()));
    if (cpu_thread_budget != 0)
      return std::min(cpu_thread_budget, allowed);
    if (const char* text = std::getenv("VECOPS_NUM_THREADS"); text != nullptr && *text) {
      char* end = nullptr;
      errno = 0;
      const auto count = std::strtoul(text, &end, 10);
      if (errno == 0 && end != text && *end == '\0' && *text != '-' && count > 0)
        return static_cast<std::uint32_t>(std::min<unsigned long>(count, allowed));
      return 1; // malformed budgets must not silently oversubscribe
    }
#if defined(__linux__)
    static std::mutex topology_mutex;
    static std::map<std::vector<int>, std::uint32_t> topology_cache;
    std::lock_guard lock(topology_mutex);
    if (auto it = topology_cache.find(cpus); it != topology_cache.end())
      return std::min(it->second, allowed);
    std::set<std::pair<int, int>> cores;
    for (auto cpu : cpus) {
      int package = -1, core = -1;
      const auto base = "/sys/devices/system/cpu/cpu" + std::to_string(cpu) + "/topology/";
      std::ifstream(base + "physical_package_id") >> package;
      std::ifstream(base + "core_id") >> core;
      if (package < 0 || core < 0) {
        cores.clear();
        break;
      }
      cores.emplace(package, core);
    }
    const auto capacity =
      cores.empty() ? static_cast<std::uint32_t>(cpus.size()) : static_cast<std::uint32_t>(cores.size());
    topology_cache.emplace(cpus, capacity);
    return std::min(std::max(1u, capacity), allowed);
#else
    return allowed;
#endif
  } catch (...) {
    return 1;
  }
}

NativeThreadPool& adaptive_pool(std::uint32_t selected) {
  static std::mutex mutex;
  static std::map<std::pair<std::uint32_t, std::vector<int>>, std::unique_ptr<NativeThreadPool>> pools;
  const auto available = cpu_capacity(nullptr);
  const auto needed = std::max(1u, std::min(selected, available));
  // Cache geometric capacities, not one pool per task count. Varying grid
  // shapes must not accumulate O(P^2) sleeping threads under a fixed budget.
  // Only the requested prefix participates; extra capacity receives no mail.
  const auto rounded = std::bit_ceil(static_cast<std::uint64_t>(needed));
  const auto capacity = static_cast<std::uint32_t>(std::min<std::uint64_t>(rounded, available));
  auto cpus = physical_first(pool_cpus(true));
  std::lock_guard lock(mutex);
  auto& pool = pools[{capacity, cpus}];
  if (!pool)
    pool = std::make_unique<NativeThreadPool>(capacity, std::move(cpus));
  return *pool;
}

std::int32_t cpu_tasks(void*, std::uint32_t count, void* opaque, VecopsParallelTaskFn body, VecopsError* error) {
  if (body == nullptr || count == 0)
    return VECOPS_STATUS_OK;
  try {
    if (inside_native_pool) {
      for (std::uint32_t task = 0; task < count; ++task)
        body(opaque, task, count);
      return VECOPS_STATUS_OK;
    }
    const auto* pool = adaptive_pool(count).provider();
    return pool->parallel_for(pool->context, count, opaque, body, error);
  } catch (const std::exception& exception) {
    set_error(error, exception.what());
  } catch (...) {
    set_error(error, "CPU executor failed");
  }
  return VECOPS_STATUS_EXECUTION_ERROR;
}
std::int32_t cpu_team(void*, std::uint32_t count, void* opaque, VecopsTeamBodyFn body, VecopsError* error) {
  if (body == nullptr)
    return VECOPS_STATUS_OK;
  try {
    if (inside_native_pool) {
      const VecopsTeamContextV1 serial{sizeof(VecopsTeamContextV1), 0, 1, nullptr, nullptr};
      body(opaque, &serial);
      return VECOPS_STATUS_OK;
    }
    const auto* pool = adaptive_pool(count).provider();
    return pool->team_executor->run(pool->context, count, opaque, body, error);
  } catch (const std::exception& exception) {
    set_error(error, exception.what());
  } catch (...) {
    set_error(error, "CPU team executor failed");
  }
  return VECOPS_STATUS_EXECUTION_ERROR;
}

} // namespace

extern "C" VECOPS_RUNTIME_EXPORT const VecopsThreadPoolV1* vecops_cpu_thread_pool_v1() {
  static const VecopsTeamExecutorV1 team{sizeof(VecopsTeamExecutorV1), 1, cpu_team};
  static const VecopsThreadPoolV1 provider{sizeof(VecopsThreadPoolV1),
                                           VECOPS_THREAD_POOL_ABI_MAJOR,
                                           VECOPS_THREAD_POOL_ABI_MINOR,
                                           UINT64_C(0x63707567726964),
                                           nullptr,
                                           cpu_capacity,
                                           [](void*) -> std::uint32_t { return inside_native_pool ? 1 : 0; },
                                           cpu_tasks,
                                           nullptr,
                                           nullptr,
                                           0,
                                           &team};
  return &provider;
}

extern "C" VECOPS_RUNTIME_EXPORT std::uint32_t vecops_cpu_thread_limit(std::uint32_t limit) {
  return std::exchange(cpu_thread_budget, limit);
}

extern "C" VECOPS_RUNTIME_EXPORT const VecopsThreadPoolV1* vecops_native_thread_pool_v1(std::uint32_t threads) {
  return get_native_pool(threads).provider();
}

extern "C" VECOPS_RUNTIME_EXPORT void vecops_native_thread_pool_begin_active(std::uint32_t threads) {
  get_native_pool(threads).begin_active();
}

extern "C" VECOPS_RUNTIME_EXPORT void vecops_native_thread_pool_end_active() {
  get_native_pool(1).end_active();
}
