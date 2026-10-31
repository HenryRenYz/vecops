// SPDX-FileCopyrightText: 2026 HenryRenYz and vecops contributors
// SPDX-License-Identifier: MIT
// Persistent pinned workers, private mappings and a shared timed window.
// Statistics are collected after completion fences, before the finish barrier.
#include "Bench.h"
#include <algorithm>
#include <atomic>
#include <condition_variable>
#include <cmath>
#include <cstring>
#include <exception>
#include <iomanip>
#include <iostream>
#include <limits>
#include <memory>
#include <mutex>
#include <numeric>
#include <random>
#include <set>
#include <stdexcept>
#include <sys/resource.h>
#include <sched.h>
#include <thread>

namespace memory_bench {
namespace {
// Keep synchronization outside timed intervals and remain compatible with the
// older libstdc++ shipped alongside BiSheng (which has no <barrier>).
class Barrier {
 public:
  explicit Barrier(std::size_t count) : parties(count), remaining(count) {}
  void arrive_and_wait() {
    std::unique_lock lock(mutex);
    const auto previous = generation;
    if (--remaining == 0) {
      remaining = parties;
      ++generation;
      lock.unlock();
      changed.notify_all();
    } else changed.wait(lock, [&] { return generation != previous; });
  }
 private:
  const std::size_t parties;
  std::size_t remaining, generation = 0;
  std::mutex mutex;
  std::condition_variable changed;
};
float input_a(std::size_t i) { return 1.0f + static_cast<float>(i % 64) / 64.0f; }
float input_b(std::size_t i) { return 0.25f + static_cast<float>(i % 32) / 32.0f; }
std::size_t array_count(Operation o) { return o == Operation::Triad ? 3 : o == Operation::Copy ? 2 : 1; }
std::string route_name(const Group& g) { return std::to_string(g.route.src) + ":" + std::to_string(g.route.dst); }
struct alignas(128) Worker {
  int cpu = -1, group = -1;
  bool latency = false;
  std::size_t bytes = 0, stride = 0, batch = 0, offset = 0;
  std::uint64_t elements = 0, total_elements = 0, begin = 0, end = 0;
  std::uint64_t minor = 0, major = 0, switches = 0;
  int end_cpu = -1;
  std::unique_ptr<Buffer> a, b, d;
  void* pointer = nullptr;
  std::exception_ptr error;
};
class Session {
 public:
  Session(const Options& o, const Topology& t, const std::vector<Group>& groups,
          Operation op, Kernel kernel, std::size_t bytes, bool latency)
      : options(o), topology(t), mapping(groups), op(op), kernel(std::move(kernel)) {
    std::set<int> cpus;
    for (std::size_t g = 0; g < mapping.size(); ++g) {
      auto per_array = bytes;
      if (options.size_scope == "group-array") {
        if (bytes % mapping[g].cpus.size()) throw std::runtime_error("Group array size must divide exactly by worker count");
        per_array /= mapping[g].cpus.size();
      }
      if (!latency || options.loaded) for (int cpu : mapping[g].cpus) {
        auto w = std::make_unique<Worker>(); w->cpu = cpu; w->group = g; w->bytes = per_array;
        if (!this->kernel.step_bytes || per_array < this->kernel.step_bytes || per_array % this->kernel.step_bytes)
          throw std::runtime_error("Per-thread array size must be a positive multiple of kernel step bytes");
        w->batch = std::max<std::size_t>(this->kernel.step_bytes / 4, 4096 / (this->kernel.step_bytes / 4) * (this->kernel.step_bytes / 4));
        cpus.insert(cpu); workers.push_back(std::move(w));
      }
      if (latency) {
        auto w = std::make_unique<Worker>(); w->latency = true; w->group = g;
        if (options.loaded) {
          auto chase_options = options;
          chase_options.group_cpus.clear();
          // Resolve the reserved relative offset with exactly the same CPU view.
          const auto latency_groups = memory_bench::groups(chase_options, topology, {options.latency_cpu});
          w->cpu = latency_groups[g].cpus[0];
        } else {
          if (mapping[g].cpus.size() != 1) throw std::runtime_error("Latency needs one CPU per group; use semicolon lists for sequential CPU tests");
          w->cpu = mapping[g].cpus[0];
        }
        if (!cpus.insert(w->cpu).second) throw std::runtime_error("Latency CPU overlaps a bandwidth worker");
        w->bytes = bytes;
        w->stride = options.stride ? options.stride : topology.line_bytes;
        if (w->stride < topology.line_bytes || w->stride % topology.line_bytes || bytes % w->stride || bytes / w->stride < 2)
          throw std::runtime_error("Chase size must hold at least two nodes and be a multiple of a cache-line-multiple stride");
        w->batch = 256;
        workers.push_back(std::move(w));
      }
    }
    // Account for every private array, including repeated memory targets.
    std::map<int, std::size_t> demand;
    for (const auto& w : workers) {
      auto add = [&](bool source) {
        const auto nodes = memory_nodes(options, mapping, w->group, source);
        const auto page = options.pages == "huge" ? 2 * 1024 * 1024UL : topology.page_bytes;
        if (w->bytes > std::numeric_limits<std::size_t>::max() - page) throw std::runtime_error("Allocation size overflow");
        const auto pages = (w->bytes + page - 1) / page;
        for (std::size_t i = 0; i < nodes.size(); ++i) {
          // Include one page/node of conservative phase-probe reservation slack.
          const auto amount = checked_mul(pages / nodes.size() + (i < pages % nodes.size()) + (nodes.size() > 1), page);
          if (demand[nodes[i]] > std::numeric_limits<std::size_t>::max() - amount) throw std::runtime_error("Total memory size overflow");
          demand[nodes[i]] += amount;
        }
      };
      if (w->latency || op != Operation::Write) add(true);
      if (!w->latency && op == Operation::Triad) add(true);
      if (!w->latency && op != Operation::Read) add(false);
    }
    for (const auto& [node, amount] : demand)
      if (amount > (options.pages == "huge" ? topology.node(node).huge_free_bytes
                                           : topology.node(node).free_bytes) * 0.9)
        throw std::runtime_error("Requested private arrays exceed 90% of the selected page pool on node " + std::to_string(node));
    barrier = std::make_unique<Barrier>(workers.size() + 1);
    try {
      for (auto& w : workers) threads.emplace_back([this, p = w.get()] {
        while (!launched.load(std::memory_order_acquire)) relax();
        if (cancel.load(std::memory_order_relaxed)) return;
        try { initialize(*p); } catch (...) { p->error = std::current_exception(); }
        barrier->arrive_and_wait();
        for (;;) {
          barrier->arrive_and_wait();
          if (stopping) break;
          try { measure(*p); } catch (...) { p->error = std::current_exception(); }
          barrier->arrive_and_wait();
        }
      });
    } catch (...) {
      cancel.store(true, std::memory_order_relaxed);
      launched.store(true, std::memory_order_release);
      for (auto& thread : threads) thread.join();
      throw;
    }
    launched.store(true, std::memory_order_release);
    barrier->arrive_and_wait();
  }
  ~Session() {
    stopping = true;
    barrier->arrive_and_wait();
    for (auto& thread : threads) thread.join();
  }
  void check_errors() const {
    for (const auto& w : workers) if (w->error) std::rethrow_exception(w->error);
  }
  void phase(double seconds) {
    check_errors();
    // Large broadcasts need more lead time to wake every pinned worker.
    // This is outside timing; the measured skew threshold is unchanged.
    const auto lead = std::max<std::uint64_t>(5000000, workers.size() * 100000ULL);
    start = now_ns() + lead;
    deadline = start + static_cast<std::uint64_t>(seconds * 1e9);
    barrier->arrive_and_wait();
    barrier->arrive_and_wait();
    check_errors();
  }
  void calibrate() {
    // Concurrent calibration captures contention between all active routes.
    phase(0.02);
    const double budget = std::max(500.0, std::min(20000.0, options.seconds * 1e9 * options.max_skew / 8));
    for (auto& w : workers) {
      const auto elapsed = std::max<std::uint64_t>(1, w->end - w->begin);
      auto batch = static_cast<std::size_t>(w->elements * budget / elapsed);
      const std::size_t step = w->latency ? 1 : kernel.step_bytes / 4;
      // Keep a finite upper bound even for cache-resident loops.
      batch = std::min<std::size_t>(batch, w->latency ? 16384 : 16 * 1024 * 1024);
      w->batch = std::max(step, batch / step * step);
    }
    if (options.warmup > 0) phase(options.warmup);
  }
  void verify() const {
    for (const auto& w : workers) {
      if (w->a) w->a->verify();
      if (w->b) w->b->verify();
      if (w->d) w->d->verify();
      if (w->latency) {
        // A full traversal must return to its origin; initialization made one cycle.
        if (chase(w->pointer, w->bytes / w->stride) != w->pointer)
          throw std::runtime_error("Pointer chain verification failed");
        continue;
      }
      const auto n = w->bytes / sizeof(float);
      const auto* a = w->a ? static_cast<const float*>(w->a->data) : nullptr;
      const auto* b = w->b ? static_cast<const float*>(w->b->data) : nullptr;
      const auto* d = w->d ? static_cast<const float*>(w->d->data) : nullptr;
      for (std::size_t i = 0; i < n; ++i) {
        if ((a && a[i] != input_a(i)) || (b && b[i] != input_b(i))) throw std::runtime_error("Input array corruption");
        if (d) {
          float expected = -17.0f;
          if (i < std::min<std::uint64_t>(w->total_elements, n)) {
            expected = op == Operation::Write ? 1.25f : input_a(i);
            if (op == Operation::Triad) expected = options.fma ? std::fma(0.5f, input_b(i), expected) : expected + 0.5f * input_b(i);
          }
          if (d[i] != expected) throw std::runtime_error("Output verification failed at element " + std::to_string(i));
        }
      }
    }
  }
  std::vector<std::unique_ptr<Worker>> workers;
  const Options& options;
  const Topology& topology;
  const std::vector<Group>& mapping;
  Operation op;
  Kernel kernel;
 private:
  void initialize(Worker& w) {
    pin(w.cpu);
    if (!w.latency && kernel.vector_bytes && native_vector_bytes() != kernel.vector_bytes)
      throw std::runtime_error("Worker vector length differs from the configured kernel step");
    auto allocate = [&](bool source) {
      return std::make_unique<Buffer>(w.bytes, memory_nodes(options, mapping, w.group, source), options.pages, topology.page_bytes);
    };
    if (w.latency) {
      w.a = allocate(true);
      const auto count = w.bytes / w.stride;
      std::vector<std::size_t> order(count);
      std::iota(order.begin(), order.end(), 0);
      std::mt19937_64 generator(options.seed);
      std::shuffle(order.begin(), order.end(), generator);
      for (std::size_t i = 0; i < count; ++i) {
        void* next = static_cast<char*>(w.a->data) + order[(i + 1) % count] * w.stride;
        std::memcpy(static_cast<char*>(w.a->data) + order[i] * w.stride, &next, sizeof(next));
      }
      w.pointer = static_cast<char*>(w.a->data) + order[0] * w.stride;
    } else {
      if (op != Operation::Write) w.a = allocate(true);
      if (op == Operation::Triad) w.b = allocate(true);
      if (op != Operation::Read) w.d = allocate(false);
      for (std::size_t i = 0; i < w.bytes / sizeof(float); ++i) {
        if (w.a) static_cast<float*>(w.a->data)[i] = input_a(i);
        if (w.b) static_cast<float*>(w.b->data)[i] = input_b(i);
        if (w.d) static_cast<float*>(w.d->data)[i] = -17.0f;
      }
    }
    if (w.a) w.a->verify();
    if (w.b) w.b->verify();
    if (w.d) w.d->verify();
    completion_fence();
    // Fault in timer/resource-accounting code before the measured samples.
    rusage usage{}; getrusage(RUSAGE_THREAD, &usage); (void)now_ns();
  }
  void measure(Worker& w) {
    rusage before{}, after{};
    w.elements = 0;
    completion_fence();
    while (now_ns() < start) relax();
    getrusage(RUSAGE_THREAD, &before);
    w.begin = now_ns();
    auto current = w.begin;
    const auto n = w.bytes / sizeof(float);
    const float* a = w.a ? static_cast<const float*>(w.a->data) : nullptr;
    const float* b = w.b ? static_cast<const float*>(w.b->data) : nullptr;
    float* d = w.d ? static_cast<float*>(w.d->data) : nullptr;
    while (current < deadline) {
      if (w.latency) w.pointer = chase(w.pointer, w.batch);
      else {
        std::size_t remaining = w.batch;
        while (remaining) {
          const auto count = std::min(remaining, n - w.offset);
          kernel.function(a ? a + w.offset : nullptr, b ? b + w.offset : nullptr,
                          d ? d + w.offset : nullptr, count);
          w.offset += count; if (w.offset == n) w.offset = 0;
          remaining -= count;
        }
      }
      w.elements += w.batch;
      current = now_ns();
    }
    completion_fence();
    w.end = now_ns();
    w.total_elements += w.elements;
    w.end_cpu = sched_getcpu();
    getrusage(RUSAGE_THREAD, &after);
    w.minor = after.ru_minflt - before.ru_minflt;
    w.major = after.ru_majflt - before.ru_majflt;
    w.switches = after.ru_nvcsw + after.ru_nivcsw - before.ru_nvcsw - before.ru_nivcsw;
  }
  std::vector<std::thread> threads;
  std::unique_ptr<Barrier> barrier;
  std::atomic<bool> launched{false}, cancel{false};
  bool stopping = false;
  std::uint64_t start = 0, deadline = 0;
};
Kernel select_kernel(const Options& o, Operation op, const std::string& name, const std::string& access) {
  const bool load = o.load_policy == "nt" || (o.load_policy == "auto" && access == "stream");
  const bool store = o.store_policy == "nt" || (o.store_policy == "auto" && access == "stream");
  if (name.starts_with("scalar")) return scalar_kernel(op, std::stoul(name.substr(6)), load, store, o.fma);
  const unsigned power = name == "vec1" ? 0 : name == "vec2" ? 1 : 2;
  return vector_kernel(op, power, load, store, o.fma);
}
void print_mapping(const std::vector<Group>& mapping, const Options& options) {
  for (std::size_t i = 0; i < mapping.size(); ++i) {
    const auto& g = mapping[i];
    std::cout << "  Group " << i << " | Memory " << route_name(g) << " | Execute node " << g.exec
              << " | CPUs " << join_cpus(g.cpus) << '\n';
    if (options.interleave)
      std::cout << "    Interleave source [" << join_cpus(memory_nodes(options, mapping, i, true))
                << "] | destination [" << join_cpus(memory_nodes(options, mapping, i, false))
                << "] | Page 0 starts at first node; one node degenerates to bind\n";
  }
}
Row worker_row(const Session& s, const Worker& w, Row base) {
  const auto& g = s.mapping[w.group];
  base.kind = w.latency ? "latency" : "bandwidth";
  base.scope = "thread:g" + std::to_string(w.group) + "/cpu" + std::to_string(w.cpu);
  base.group = w.group; base.src = g.route.src; base.dst = w.latency ? -1 : g.route.dst; base.exec = g.exec;
  base.cpus = std::to_string(w.cpu); base.threads = 1;
  base.array_bytes = w.bytes; base.working_bytes = checked_mul(w.bytes, w.latency ? 1 : array_count(s.op));
  base.page_policy = s.options.pages == "auto" ? "auto(THP-eligible)" : s.options.pages;
  base.placement = "all-pages-verified";
  auto add_pages = [](const Buffer* buffer, auto& nodes, auto& counts) {
    if (!buffer) return;
    nodes = buffer->nodes;
    for (const auto& [node, count] : buffer->page_counts()) counts[node] += count;
  };
  add_pages(w.a.get(), base.source_nodes, base.source_pages);
  add_pages(w.b.get(), base.source_nodes, base.source_pages);
  add_pages(w.d.get(), base.destination_nodes, base.destination_pages);
  base.start_ns = w.begin; base.end_ns = w.end;
  base.seconds = (w.end - w.begin) * 1e-9;
  base.minor_faults = w.minor; base.major_faults = w.major; base.switches = w.switches;
  if (!w.elements || !base.seconds) { base.status = "missed-window"; base.detail = "Worker completed no work in the common window"; }
  else if (w.end_cpu != w.cpu) { base.status = "migration"; base.detail = "Worker migrated from its pinned CPU"; }
  else if (w.minor || w.major) { base.status = "page-fault"; base.detail = "Faults occurred in timed worker interval"; }
  else if (w.switches) { base.status = "interrupted"; base.detail = "Worker was context-switched"; }
  if (w.latency) {
    base.operation = "chase"; base.kernel = "dependent-pointer"; base.access = "temporal";
    base.load_policy = "dependent-temporal"; base.store_policy = "none";
    base.instructions = "native-pointer-load"; base.step_bytes = sizeof(void*); base.vector_bytes = 0;
    base.stride_bytes = w.stride; base.accesses = w.elements;
    base.read_bytes = checked_mul(w.elements, sizeof(void*)); base.write_bytes = 0;
    base.ns = w.elements ? (w.end - w.begin) / static_cast<double>(w.elements) : 0;
  } else {
    base.step_bytes = s.kernel.step_bytes; base.vector_bytes = s.kernel.vector_bytes;
    base.load_policy = s.kernel.load_policy; base.store_policy = s.kernel.store_policy;
    base.instructions = s.kernel.instructions;
    base.read_bytes = checked_mul(w.elements, s.op == Operation::Write ? 0 : s.op == Operation::Triad ? 8 : 4);
    base.write_bytes = checked_mul(w.elements, s.op == Operation::Read ? 0 : 4);
  }
  base.read_gbs = base.seconds ? base.read_bytes / base.seconds / 1e9 : 0;
  base.write_gbs = base.seconds ? base.write_bytes / base.seconds / 1e9 : 0;
  base.gbs = base.read_gbs + base.write_gbs;
  if (s.op == Operation::Copy && !w.latency) base.payload_gbs = base.write_gbs;
  return base;
}
Row combine(const std::vector<Row>& rows, const std::string& scope, double duration, double skew) {
  Row r = rows.front(); r.scope = scope;
  r.read_bytes = r.write_bytes = r.accesses = 0; r.threads = r.working_bytes = 0;
  r.minor_faults = r.major_faults = r.switches = 0;
  r.start_ns = std::numeric_limits<std::uint64_t>::max(); r.end_ns = 0;
  std::uint64_t latest_start = 0, earliest_end = std::numeric_limits<std::uint64_t>::max();
  r.cpus.clear();
  r.source_nodes.clear(); r.destination_nodes.clear();
  r.source_pages.clear(); r.destination_pages.clear();
  for (const auto& row : rows) {
    r.source_nodes.insert(r.source_nodes.end(), row.source_nodes.begin(), row.source_nodes.end());
    r.destination_nodes.insert(r.destination_nodes.end(), row.destination_nodes.begin(), row.destination_nodes.end());
    for (const auto& [node, count] : row.source_pages) r.source_pages[node] += count;
    for (const auto& [node, count] : row.destination_pages) r.destination_pages[node] += count;
    r.read_bytes += row.read_bytes; r.write_bytes += row.write_bytes; r.accesses += row.accesses;
    r.threads += row.threads; r.working_bytes += row.working_bytes;
    r.minor_faults += row.minor_faults; r.major_faults += row.major_faults; r.switches += row.switches;
    r.start_ns = std::min(r.start_ns, row.start_ns); r.end_ns = std::max(r.end_ns, row.end_ns);
    latest_start = std::max(latest_start, row.start_ns); earliest_end = std::min(earliest_end, row.end_ns);
    if (!r.cpus.empty()) r.cpus += ',';
    r.cpus += row.cpus;
    if (row.status != "ok") { r.status = row.status; r.detail = row.detail; }
  }
  for (auto* nodes : {&r.source_nodes, &r.destination_nodes}) {
    std::sort(nodes->begin(), nodes->end()); nodes->erase(std::unique(nodes->begin(), nodes->end()), nodes->end());
  }
  r.start_skew_ns = latest_start - r.start_ns; r.end_skew_ns = r.end_ns - earliest_end;
  r.common_ns = earliest_end > latest_start ? earliest_end - latest_start : 0;
  r.seconds = (r.end_ns - r.start_ns) * 1e-9;
  r.read_gbs = r.read_bytes / r.seconds / 1e9; r.write_gbs = r.write_bytes / r.seconds / 1e9;
  r.gbs = r.read_gbs + r.write_gbs;
  r.payload_gbs = r.operation == "copy" ? r.write_gbs : 0;
  if (r.start_skew_ns > duration * 1e9 * skew || r.end_skew_ns > duration * 1e9 * skew || !r.common_ns) {
    r.status = "skew"; r.detail = "Worker interval spread exceeds configured threshold";
  }
  return r;
}
std::vector<Row> sample_rows(const Session& s, const Row& base) {
  std::vector<Row> result, bandwidth;
  for (const auto& w : s.workers) {
    auto r = worker_row(s, *w, base);
    result.push_back(r);
    if (w->latency) {
      r.scope = "chase:g" + std::to_string(w->group);
      result.push_back(r);
    } else bandwidth.push_back(r);
  }
  if (!bandwidth.empty()) {
    for (std::size_t g = 0; g < s.mapping.size(); ++g) {
      std::vector<Row> rows;
      for (const auto& r : bandwidth) if (r.group == static_cast<int>(g)) rows.push_back(r);
      result.push_back(combine(rows, "group:" + std::to_string(g), s.options.seconds, s.options.max_skew));
    }
    auto total = combine(bandwidth, "total", s.options.seconds, s.options.max_skew);
    total.group = total.src = total.dst = total.exec = -1;
    const bool valid = total.status == "ok";
    result.push_back(total);
    // A loaded-latency sample is accepted only with valid simultaneous traffic.
    if (!valid) for (auto& r : result) if (r.kind == "latency" && r.status == "ok") {
      r.status = total.status; r.detail = "Background traffic: " + total.detail;
    }
    if (valid) for (auto& r : result) if (r.kind == "latency") {
      const auto spread = s.options.seconds * 1e9 * s.options.max_skew;
      const auto start_delta = r.start_ns > total.start_ns ? r.start_ns - total.start_ns : total.start_ns - r.start_ns;
      const auto end_delta = r.end_ns > total.end_ns ? r.end_ns - total.end_ns : total.end_ns - r.end_ns;
      if (start_delta > spread || end_delta > spread) {
        r.status = "skew"; r.detail = "Chase and background traffic do not share the configured window";
      }
      const auto first_end = std::min(r.end_ns, total.end_ns);
      const auto last_start = std::max(r.start_ns, total.start_ns);
      r.common_ns = first_end > last_start ? first_end - last_start : 0;
    }
  }
  return result;
}
void run_samples(Session& s, Reporter& reporter, Row base, const std::string& label) {
  s.check_errors();
  if (s.options.interleave) {
    for (std::size_t g = 0; g < s.mapping.size(); ++g) {
      std::map<int, std::size_t> source, destination;
      for (const auto& w : s.workers) if (w->group == static_cast<int>(g)) {
        for (const auto* buffer : {w->a.get(), w->b.get()})
          if (buffer) for (const auto& [node, count] : buffer->page_counts()) source[node] += count;
        if (w->d) for (const auto& [node, count] : w->d->page_counts()) destination[node] += count;
      }
      std::cout << "  Verified group " << g << " pages | source [" << page_counts_text(source)
                << "] | destination [" << page_counts_text(destination) << "]\n";
    }
  }
  s.calibrate();
  std::vector<Row> all;
  for (int sample = 1; sample <= s.options.samples; ++sample) {
    base.sample = sample;
    for (int attempt = 0; attempt <= s.options.retries; ++attempt) {
      base.attempt = attempt;
      s.phase(s.options.seconds);
      auto rows = sample_rows(s, base);
      const bool valid = std::all_of(rows.begin(), rows.end(), [](const Row& r) { return r.status == "ok"; });
      if (!valid && attempt < s.options.retries) for (auto& r : rows) {
        if (r.status == "ok") { r.status = "retry"; r.detail = "Another participant invalidated this attempt"; }
      }
      all.insert(all.end(), rows.begin(), rows.end());
      if (valid || attempt == s.options.retries) break;
    }
  }
  // Do not publish successful measurements until allocation residence and
  // arithmetic are verified after every timed phase has finished.
  try { s.verify(); }
  catch (...) {
    for (auto& r : all) { r.status = "verification-failed"; r.detail = "Post-run verification failed"; reporter.add(r); }
    throw;
  }
  std::vector<Row> bw, lat;
  for (const auto& r : all) {
    reporter.add(r);
    if (!r.scope.starts_with("thread:")) (r.kind == "bandwidth" ? bw : lat).push_back(r);
  }
  if (!bw.empty()) reporter.summary(bw, label + (lat.empty() ? "" : " | background bandwidth"));
  if (!lat.empty()) reporter.summary(lat, label + " | dependent loads (ns/load)");
}
void run_configurations(const Options& o, const Topology& t, Reporter& reporter, bool latency) {
  if (latency && o.loaded && o.latency_cpu < 0) throw std::runtime_error("--loaded requires --latency-cpu");
  int config = 0;
  for (const auto& cpu_spec : split(o.cpu, ';')) {
    auto mapping = groups(o, t, cpu_list(cpu_spec));
    print_mapping(mapping, o);
    for (const auto& g : mapping) {
      if (g.route.paired && latency && !o.loaded)
        throw std::runtime_error("Idle latency accepts memory targets, not source:destination routes");
      if (latency && !o.loaded && g.cpus.size() != 1)
        throw std::runtime_error("Latency needs one CPU per group; use semicolon lists for sequential CPU tests");
    }
    if (latency && o.loaded) {
      auto chase_options = o;
      chase_options.group_cpus.clear();
      const auto reserved = groups(chase_options, t, {o.latency_cpu});
      std::set<int> traffic;
      for (const auto& g : mapping) traffic.insert(g.cpus.begin(), g.cpus.end());
      for (const auto& g : reserved) {
        if (traffic.count(g.cpus[0])) throw std::runtime_error("Latency CPU overlaps a bandwidth worker");
        std::cout << "  Reserved chase CPU: " << g.cpus[0] << '\n';
      }
    }
    const auto ops = latency && !o.loaded ? std::vector<std::string>{"read"} : o.operations;
    const auto names = latency && !o.loaded ? std::vector<std::string>{"vec1"} : o.kernels;
    const auto accesses = latency && !o.loaded ? std::vector<std::string>{"temporal"} : o.accesses;
    for (const auto& op_name : ops) for (const auto& name : names) for (const auto& access : accesses) for (auto size : o.sizes) {
      const auto op = operation(op_name);
      if (!latency && (op == Operation::Read || op == Operation::Write)) for (const auto& g : mapping)
        if (g.route.paired) throw std::runtime_error("Single-target operations do not accept source:destination routes");
      auto kernel = select_kernel(o, op, name, access);
      Row base; base.config = ++config; base.operation = op_name; base.kernel = name;
      base.access = access; base.requested_bytes = size; base.fma = o.fma;
      const auto label = (latency && !o.loaded ? std::string("chase") : op_name + " / " + name + " / " + access)
                       + " / " + human_size(size) + " / CPUs " + cpu_spec;
      if (!kernel.unsupported.empty()) {
        base.kind = "bandwidth"; base.status = "unsupported"; base.detail = kernel.unsupported;
        reporter.add(base); std::cout << "\n  SKIP " << label << ": " << kernel.unsupported << '\n'; continue;
      }
      std::cout << "\n  Config " << config << ": " << label << '\n';
      if (!latency || o.loaded) std::cout << "  Effective load=" << kernel.load_policy << ", store=" << kernel.store_policy
                                       << " | Step=" << kernel.step_bytes << " B\n";
      if (!latency || o.loaded) for (const auto& g : mapping) {
        if (o.size_scope == "group-array" && size % g.cpus.size())
          throw std::runtime_error("Group array size must divide exactly by worker count");
        const auto per_array = o.size_scope == "group-array" ? size / g.cpus.size() : size;
        if (per_array < kernel.step_bytes || per_array % kernel.step_bytes)
          throw std::runtime_error("Per-thread array size must be a positive multiple of kernel step bytes");
        const auto working = checked_mul(checked_mul(per_array, g.cpus.size()), array_count(op));
        std::cout << "  Memory " << route_name(g) << ": " << human_size(per_array)
                  << "/thread/array | Working set " << human_size(working) << '\n';
      }
      if (latency) {
        const auto stride = o.stride ? o.stride : t.line_bytes;
        if (stride < t.line_bytes || stride % t.line_bytes || size % stride || size / stride < 2)
          throw std::runtime_error("Chase size must hold at least two nodes and be a multiple of a cache-line-multiple stride");
      }
      std::cout << std::flush;
      if (o.dry_run) continue;
      Session session(o, t, mapping, op, kernel, size, latency);
      run_samples(session, reporter, base, label);
    }
  }
}
} // namespace
void run_bandwidth(const Options& o, const Topology& t, Reporter& reporter) { run_configurations(o, t, reporter, false); }
void run_latency(const Options& o, const Topology& t, Reporter& reporter) { run_configurations(o, t, reporter, true); }

void run_pingpong(const Options& o, const Topology& t, Reporter& reporter) {
  if (o.peer_cpu < 0) throw std::runtime_error("pingpong requires --peer-cpu");
  int config = 0;
  for (const auto& cpu_spec : split(o.cpu, ';')) {
    const auto mapping = groups(o, t, cpu_list(cpu_spec));
    if (mapping.size() != 1 || mapping[0].cpus.size() != 1) throw std::runtime_error("pingpong requires one route and one initiating CPU");
    auto peer_options = o;
    peer_options.group_cpus.clear();
    peer_options.exec = std::to_string(mapping[0].route.dst);
    const auto peer = groups(peer_options, t, {o.peer_cpu})[0].cpus[0];
    const auto cpu = mapping[0].cpus[0];
    if (cpu == peer) throw std::runtime_error("Ping-pong CPUs must differ");
    print_mapping(mapping, o);
    std::cout << "  Peer CPU: " << peer << " | Metric: ns/round-trip\n";
    if (o.dry_run) continue;
    Buffer memory(t.page_bytes, mapping[0].route.src, o.pages, t.page_bytes);
    auto* token = new (memory.data) std::atomic<std::uint64_t>(0);
    if (!token->is_lock_free()) throw std::runtime_error("Ping-pong requires lock-free 64-bit atomics");
    memory.verify();
    std::atomic<bool> done{false}, ready{false};
    std::exception_ptr peer_error;
    std::thread responder([&] {
      try {
        pin(peer); ready.store(true, std::memory_order_release);
        while (!done.load(std::memory_order_relaxed)) {
          if (token->load(std::memory_order_acquire) == 1) token->store(0, std::memory_order_release);
          else relax();
        }
      } catch (...) { peer_error = std::current_exception(); ready.store(true, std::memory_order_release); }
    });
    // Restore the main thread's affinity after each ping-pong configuration.
    struct RestoreAffinity {
      const std::vector<int>& cpus;
      ~RestoreAffinity() {
        const auto count = *std::max_element(cpus.begin(), cpus.end()) + 1;
        const auto bytes = CPU_ALLOC_SIZE(count); auto* mask = CPU_ALLOC(count);
        if (mask) { CPU_ZERO_S(bytes, mask); for (auto c : cpus) CPU_SET_S(c, bytes, mask); sched_setaffinity(0, bytes, mask); CPU_FREE(mask); }
      }
    } restore{t.allowed_cpus};
    try {
      pin(cpu);
      while (!ready.load(std::memory_order_acquire)) relax();
      if (peer_error) std::rethrow_exception(peer_error);
      auto measure = [&](double seconds) {
        std::uint64_t count = 0;
        rusage before{}, after{}; getrusage(RUSAGE_THREAD, &before);
        completion_fence(); const auto begin = now_ns(); const auto end = begin + static_cast<std::uint64_t>(seconds * 1e9);
        do {
          // Poll the timer once per batch, never once per handoff.
          for (int i = 0; i < 64; ++i) {
            token->store(1, std::memory_order_release);
            while (token->load(std::memory_order_acquire) != 0) relax();
            ++count;
          }
        } while (now_ns() < end);
        completion_fence(); const auto finish = now_ns(); getrusage(RUSAGE_THREAD, &after);
        Row r; r.kind = "pingpong"; r.operation = "pingpong"; r.kernel = "atomic-handoff";
        r.scope = "round-trip"; r.config = config; r.threads = 2;
        r.cpus = std::to_string(cpu) + "," + std::to_string(peer);
        r.src = mapping[0].route.src; r.dst = mapping[0].route.dst; r.exec = mapping[0].exec;
        r.accesses = count; r.start_ns = begin; r.end_ns = finish;
        r.seconds = (finish - begin) * 1e-9; r.ns = (finish - begin) / static_cast<double>(count);
        r.working_bytes = t.line_bytes; r.page_policy = o.pages; r.placement = "all-pages-verified";
        r.minor_faults = after.ru_minflt - before.ru_minflt; r.major_faults = after.ru_majflt - before.ru_majflt;
        r.switches = after.ru_nvcsw + after.ru_nivcsw - before.ru_nvcsw - before.ru_nivcsw;
        if (r.minor_faults || r.major_faults) r.status = "page-fault";
        else if (r.switches) r.status = "interrupted";
        return r;
      };
      ++config;
      (void)measure(std::max(0.01, o.warmup));
      std::vector<Row> rows;
      for (int sample = 1; sample <= o.samples; ++sample) for (int attempt = 0; attempt <= o.retries; ++attempt) {
        auto r = measure(o.seconds); r.sample = sample; r.attempt = attempt; rows.push_back(r);
        if (r.status == "ok") break;
      }
      done.store(true, std::memory_order_relaxed); responder.join();
      memory.verify();
      for (const auto& r : rows) reporter.add(r);
      reporter.summary(rows, "CPU " + std::to_string(cpu) + " <-> CPU " + std::to_string(peer) + " (ns/round-trip)");
    } catch (...) {
      done.store(true, std::memory_order_relaxed);
      if (responder.joinable()) responder.join();
      throw;
    }
  }
}
} // namespace memory_bench
