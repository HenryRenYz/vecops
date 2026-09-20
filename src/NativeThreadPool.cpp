// Copyright (c) vecops contributors.

#include "vecops/runtime/CallAbi.h"

#include <algorithm>
#include <atomic>
#include <cerrno>
#include <condition_variable>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <thread>
#include <utility>
#include <vector>

#if defined(__linux__)
#  include <pthread.h>
#  include <sched.h>
#endif

#if defined(__x86_64__) || defined(_M_X64)
#  include <immintrin.h>
#endif

namespace {

thread_local bool inside_native_pool = false;

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

std::vector<int> pool_cpus() {
  const char* configured = std::getenv("VECOPS_THREAD_POOL_CPU_LIST");
  if (configured == nullptr || *configured == '\0')
    return allowed_cpus();
  auto result = parse_cpu_list(configured);
  if (result.empty())
    throw std::runtime_error("invalid VECOPS_THREAD_POOL_CPU_LIST");
  return result;
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

class NativeThreadPool {
public:
  explicit NativeThreadPool(std::uint32_t threads)
    : threads_(std::max<std::uint32_t>(1, threads))
    , spin_count_(configured_spin_count())
    , cpus_(pool_cpus()) {
    workers_.reserve(threads_ - 1);
    for (std::uint32_t ordinal = 1; ordinal < threads_; ++ordinal)
      workers_.emplace_back([this, ordinal] { worker_loop(ordinal); });
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
                                   0};
  }

  NativeThreadPool(const NativeThreadPool&) = delete;
  NativeThreadPool& operator=(const NativeThreadPool&) = delete;

  ~NativeThreadPool() {
    {
      std::lock_guard lock(wake_mutex_);
      stopping_.store(true, std::memory_order_release);
      generation_.fetch_add(1, std::memory_order_release);
    }
    wake_cv_.notify_all();
    for (auto& worker : workers_)
      worker.join();
  }

  [[nodiscard]] const VecopsThreadPoolV1* provider() const noexcept {
    return &provider_;
  }

  void begin_active() noexcept {
    std::lock_guard submission_lock(submission_mutex_);
    if (active_scopes_.fetch_add(1, std::memory_order_acq_rel) == 0) {
      {
        std::lock_guard lock(wake_mutex_);
        generation_.fetch_add(1, std::memory_order_release);
      }
      wake_cv_.notify_all();
    }
  }

  void end_active() noexcept {
    auto active = active_scopes_.load(std::memory_order_acquire);
    while (active != 0 && !active_scopes_.compare_exchange_weak(active, active - 1, std::memory_order_acq_rel,
                                                                std::memory_order_acquire)) {
    }
  }

private:
  struct Submission {
    std::uint32_t task_count;
    std::uint32_t participants;
    void* body_context;
    VecopsParallelTaskFn body;
    std::atomic<std::uint32_t> remaining;
    std::mutex exception_mutex;
    std::exception_ptr exception;
  };

  static std::uint32_t maximum(void* context) noexcept {
    if (inside_native_pool)
      return 1;
    return static_cast<NativeThreadPool*>(context)->threads_;
  }

  static std::uint32_t in_parallel(void*) noexcept {
    return inside_native_pool ? 1u : 0u;
  }

  static std::int32_t parallel_for(void* context, std::uint32_t task_count, void* body_context,
                                   VecopsParallelTaskFn body, VecopsError* error) noexcept {
    if (body == nullptr || task_count == 0)
      return VECOPS_STATUS_OK;
    auto& self = *static_cast<NativeThreadPool*>(context);
    try {
      if (task_count == 1 || inside_native_pool) {
        for (std::uint32_t task = 0; task < task_count; ++task)
          body(body_context, task, task_count);
        return VECOPS_STATUS_OK;
      }
      self.run(task_count, body_context, body);
      return VECOPS_STATUS_OK;
    } catch (const std::exception& exception) {
      set_error(error, exception.what());
    } catch (...) {
      set_error(error, "native thread pool failed");
    }
    return VECOPS_STATUS_EXECUTION_ERROR;
  }

  void run(std::uint32_t task_count, void* body_context, VecopsParallelTaskFn body) {
    std::lock_guard submission_lock(submission_mutex_);
    const auto participants = std::min(task_count, threads_);
    // Every worker observes each generation, including workers whose ordinal
    // is outside this submission's participant set.  Keep the stack-backed
    // Submission alive until the entire team has acknowledged the generation;
    // otherwise a late non-participant can dereference a recycled stack slot
    // and call a stale body function pointer.
    Submission submission{task_count, participants, body_context, body, threads_, {}, nullptr};
    submission_.store(&submission, std::memory_order_release);
    {
      std::lock_guard lock(wake_mutex_);
      generation_.fetch_add(1, std::memory_order_release);
    }
    wake_cv_.notify_all();

    run_participant(submission, 0);
    submission.remaining.fetch_sub(1, std::memory_order_acq_rel);
    std::uint32_t wait_spins = 0;
    while (submission.remaining.load(std::memory_order_acquire) != 0) {
      cpu_relax();
      if (++wait_spins == 4096) {
        wait_spins = 0;
        std::this_thread::yield();
      }
    }
    submission_.store(nullptr, std::memory_order_release);
    if (submission.exception != nullptr)
      std::rethrow_exception(submission.exception);
  }

  void run_participant(Submission& submission, std::uint32_t ordinal) noexcept {
    if (ordinal >= submission.participants)
      return;
    inside_native_pool = true;
    try {
      for (std::uint32_t task = ordinal; task < submission.task_count; task += threads_)
        submission.body(submission.body_context, task, submission.task_count);
    } catch (...) {
      std::lock_guard lock(submission.exception_mutex);
      if (submission.exception == nullptr)
        submission.exception = std::current_exception();
    }
    inside_native_pool = false;
  }

  void worker_loop(std::uint32_t ordinal) noexcept {
    pin_worker(cpus_, ordinal);
    auto observed = generation_.load(std::memory_order_acquire);
    ready_.fetch_add(1, std::memory_order_release);
    while (true) {
      auto current = generation_.load(std::memory_order_acquire);
      if (current == observed) {
        if (active_scopes_.load(std::memory_order_acquire) != 0) {
          cpu_relax();
          continue;
        }
        for (std::uint32_t spin = 0; spin < spin_count_ && current == observed; ++spin) {
          cpu_relax();
          current = generation_.load(std::memory_order_acquire);
        }
        if (current == observed && active_scopes_.load(std::memory_order_acquire) == 0) {
          std::unique_lock lock(wake_mutex_);
          wake_cv_.wait(lock, [&] {
            return stopping_.load(std::memory_order_acquire) ||
                   generation_.load(std::memory_order_acquire) != observed ||
                   active_scopes_.load(std::memory_order_acquire) != 0;
          });
          current = generation_.load(std::memory_order_acquire);
        }
      }
      if (stopping_.load(std::memory_order_acquire))
        return;
      if (current == observed)
        continue;
      observed = current;
      auto* submission = submission_.load(std::memory_order_acquire);
      if (submission == nullptr)
        continue;
      run_participant(*submission, ordinal);
      submission->remaining.fetch_sub(1, std::memory_order_acq_rel);
    }
  }

  const std::uint32_t threads_;
  const std::uint32_t spin_count_;
  const std::vector<int> cpus_;
  std::vector<std::thread> workers_;
  std::mutex submission_mutex_;
  std::mutex wake_mutex_;
  std::condition_variable wake_cv_;
  std::atomic<Submission*> submission_{nullptr};
  std::atomic<std::uint64_t> generation_{0};
  std::atomic<std::uint32_t> ready_{0};
  std::atomic<std::uint32_t> active_scopes_{0};
  std::atomic<bool> stopping_{false};
  VecopsThreadPoolV1 provider_{};
};

std::once_flag native_pool_once;
std::unique_ptr<NativeThreadPool> native_pool;

NativeThreadPool& get_native_pool(std::uint32_t threads) {
  std::call_once(native_pool_once, [&] { native_pool = std::make_unique<NativeThreadPool>(threads); });
  return *native_pool;
}

} // namespace

extern "C" VECOPS_RUNTIME_EXPORT const VecopsThreadPoolV1* vecops_native_thread_pool_v1(std::uint32_t threads) {
  return get_native_pool(threads).provider();
}

extern "C" VECOPS_RUNTIME_EXPORT void vecops_native_thread_pool_begin_active(std::uint32_t threads) {
  get_native_pool(threads).begin_active();
}

extern "C" VECOPS_RUNTIME_EXPORT void vecops_native_thread_pool_end_active() {
  get_native_pool(1).end_active();
}
