// SPDX-FileCopyrightText: 2026 HenryRenYz and vecops contributors
// SPDX-License-Identifier: MIT
/** Protocol-free, contiguous AllToAll-shaped vec payload. Per-lane wall/CPU
 * clocks distinguish worker execution stalls from descheduling. The caller
 * owns all allocations; only the pool submission/callback is timed. */
#include "vecops/comm/Copy.h"
#include "vecops/runtime/CallAbi.h"
#include <algorithm>
#include <chrono>
#include <cstdint>
#include <ctime>
#include <thread>
#include <linux/perf_event.h>
#include <sys/syscall.h>
#include <sys/resource.h>
#include <unistd.h>

namespace {
struct Counters {
  int cycles = -1, instructions = -1;
  Counters() {
    perf_event_attr attr{};
    attr.size = sizeof(attr); attr.type = PERF_TYPE_HARDWARE;
    attr.config = PERF_COUNT_HW_CPU_CYCLES;
    attr.exclude_kernel = 1; attr.exclude_hv = 1;
    attr.read_format = PERF_FORMAT_GROUP | PERF_FORMAT_TOTAL_TIME_ENABLED | PERF_FORMAT_TOTAL_TIME_RUNNING;
    cycles = int(syscall(SYS_perf_event_open, &attr, 0, -1, -1, 0));
    if (cycles >= 0) {
      attr.config = PERF_COUNT_HW_INSTRUCTIONS;
      instructions = int(syscall(SYS_perf_event_open, &attr, 0, -1, cycles, 0));
    }
  }
  ~Counters() { if (instructions >= 0) close(instructions); if (cycles >= 0) close(cycles); }
  bool read_values(std::uint64_t (&values)[5]) const {
    return instructions >= 0 && read(cycles, values, sizeof(values)) == sizeof(values) && values[0] == 2;
  }
};
std::uint64_t cpu_ns() {
  timespec ts{};
  clock_gettime(CLOCK_THREAD_CPUTIME_ID, &ts);
  return std::uint64_t(ts.tv_sec) * 1000000000ULL + ts.tv_nsec;
}
struct Job {
  const vecops::comm::detail::CopyBackend* copy;
  const char* source;
  void* const* destinations;
  std::size_t peers, peer_bytes, tasks_per_peer;
  double* lane_wall;
  double* lane_cpu;
  double* lane_pmu;
};
void work(void* opaque, std::uint32_t lane, std::uint32_t lanes) {
  const auto& j = *static_cast<Job*>(opaque);
  Counters* counters = nullptr;
  std::uint64_t before[5]{}, after[5]{};
  rusage usage_before{}, usage_after{};
  if (j.lane_pmu) {
    static thread_local Counters perf; counters = &perf;
    getrusage(RUSAGE_THREAD, &usage_before);
    counters->read_values(before);
  }
  const auto begin = std::chrono::steady_clock::now();
  const auto cpu = cpu_ns();
  const auto count = j.peers * j.tasks_per_peer;
  // Same contiguous task boundaries as Communication.cpp::Batch::work.
  const auto begin_task = count * lane / lanes, end_task = count * (lane + 1) / lanes;
  for (std::size_t peer = 0; peer < j.peers; ++peer) {
    const auto base = peer * j.tasks_per_peer;
    const auto first_task = begin_task > base ? begin_task - base : 0;
    const auto last_task = std::min(end_task > base ? end_task - base : 0, j.tasks_per_peer);
    if (first_task < last_task) {
      const auto first = j.peer_bytes * first_task / j.tasks_per_peer;
      const auto last = j.peer_bytes * last_task / j.tasks_per_peer;
      j.copy->bytes(static_cast<char*>(j.destinations[peer]) + first,
                    j.source + peer * j.peer_bytes + first, last - first);
    }
  }
  j.lane_cpu[lane] = double(cpu_ns() - cpu);
  j.lane_wall[lane] = std::chrono::duration<double, std::nano>(
      std::chrono::steady_clock::now() - begin).count();
  if (counters && before[0] == 2 && counters->read_values(after)) {
    // cycles, instructions, enabled ns, running ns; no inferred scaling here.
    j.lane_pmu[8*lane] = double(after[3]-before[3]);
    j.lane_pmu[8*lane+1] = double(after[4]-before[4]);
    j.lane_pmu[8*lane+2] = double(after[1]-before[1]);
    j.lane_pmu[8*lane+3] = double(after[2]-before[2]);
  }
  if (j.lane_pmu) {
    getrusage(RUSAGE_THREAD, &usage_after);
    const auto ns = [](timeval t) { return double(t.tv_sec)*1e9+double(t.tv_usec)*1e3; };
    j.lane_pmu[8*lane+4] = ns(usage_after.ru_utime)-ns(usage_before.ru_utime);
    j.lane_pmu[8*lane+5] = ns(usage_after.ru_stime)-ns(usage_before.ru_stime);
    j.lane_pmu[8*lane+6] = double(usage_after.ru_nvcsw-usage_before.ru_nvcsw);
    j.lane_pmu[8*lane+7] = double(usage_after.ru_nivcsw-usage_before.ru_nivcsw);
  }
}
}
extern "C" int scatter_copy_pmu(const VecopsThreadPoolV1* pool, unsigned workers,
    const void* source, void* const* destinations, std::size_t peers,
    std::size_t peer_bytes, std::size_t chunk_bytes, double* elapsed,
    double* lane_wall, double* lane_cpu, double* lane_pmu) {
  if (!pool || !workers || workers > pool->max_parallelism(pool->context) ||
      !source || !destinations || !peers || !peer_bytes || !chunk_bytes) return -1;
  Job job{&vecops::comm::detail::copy_backend(), static_cast<const char*>(source),
          destinations, peers, peer_bytes, (peer_bytes + chunk_bytes - 1) / chunk_bytes,
          lane_wall, lane_cpu, lane_pmu};
  const auto begin = std::chrono::steady_clock::now();
  const int status = pool->parallel_for(pool->context, workers, &job, work, nullptr);
  *elapsed = std::chrono::duration<double, std::nano>(
      std::chrono::steady_clock::now() - begin).count();
  return status;
}

extern "C" int scatter_copy(const VecopsThreadPoolV1* pool, unsigned workers,
    const void* source, void* const* destinations, std::size_t peers,
    std::size_t peer_bytes, std::size_t chunk_bytes, double* elapsed,
    double* lane_wall, double* lane_cpu) {
  return scatter_copy_pmu(pool, workers, source, destinations, peers, peer_bytes,
                          chunk_bytes, elapsed, lane_wall, lane_cpu, nullptr);
}

/** Untimed admission for independently exec'd benchmark workers. The file is
 * initialized once by the supervisor; monotonic epochs avoid counter resets. */
extern "C" int scatter_admit(std::uint64_t* state, std::uint64_t target,
                              unsigned timeout_seconds) {
  static_assert(__atomic_always_lock_free(sizeof(std::uint64_t), nullptr));
  __atomic_fetch_add(state, 1, __ATOMIC_ACQ_REL);
  const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(timeout_seconds);
  unsigned spin = 0;
  while (__atomic_load_n(state, __ATOMIC_ACQUIRE) < target) {
    if (__atomic_load_n(state + 1, __ATOMIC_ACQUIRE)) return -1;
    if ((++spin & 1023) == 0) {
      if (std::chrono::steady_clock::now() > deadline) {
        __atomic_store_n(state + 1, 1, __ATOMIC_RELEASE);
        return -2;
      }
      std::this_thread::yield();
    }
  }
  return 0;
}
