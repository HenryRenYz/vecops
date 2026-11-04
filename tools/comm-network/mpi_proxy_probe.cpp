// SPDX-FileCopyrightText: 2026 HenryRenYz and vecops contributors
// SPDX-License-Identifier: MIT
// Standalone two-node MPI/SPSC offload probe, not the vecops network backend.
// Compile with an explicit C++ compiler and the target MPI wrapper's flags.
#include <mpi.h>
#include <sched.h>
#include <atomic>
#include <algorithm>
#include <chrono>
#include <cstdlib>
#include <iostream>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

namespace {
void check(int code) {
  if (code != MPI_SUCCESS) {
    char text[MPI_MAX_ERROR_STRING]; int count = 0;
    MPI_Error_string(code, text, &count);
    std::cerr << std::string(text, count) << std::endl;
    MPI_Abort(MPI_COMM_WORLD, code);
    std::abort();
  }
}
void relax() {
#if defined(__aarch64__)
  asm volatile("yield" ::: "memory");
#elif defined(__x86_64__)
  asm volatile("pause" ::: "memory");
#else
  std::atomic_signal_fence(std::memory_order_seq_cst);
#endif
}
void pin(int cpu) {
  cpu_set_t mask; CPU_ZERO(&mask); CPU_SET(cpu, &mask);
  if (sched_setaffinity(0, sizeof(mask), &mask)) throw std::runtime_error("CPU binding failed");
}
std::string json_string(const std::string& s) {
  std::string result = "\"";
  for (unsigned char c : s) {
    if (c == '\n') result += "\\n";
    else if (c == '\r') result += "\\r";
    else if (c == '\t') result += "\\t";
    else if (c < 32) {
      constexpr char hex[] = "0123456789abcdef";
      result += "\\u00"; result += hex[c >> 4]; result += hex[c & 15];
    }
    else { if (c == '\\' || c == '"') result += '\\'; result += c; }
  }
  return result + '"';
}
struct Slot {
  alignas(128) std::atomic<uint64_t> posted{0};
  alignas(128) std::atomic<uint64_t> completed{0};
  int operation = 0, bytes = 0;
};
}
int main(int argc, char** argv) {
  cpu_set_t allowed; CPU_ZERO(&allowed);
  if (sched_getaffinity(0, sizeof(allowed), &allowed)) return 2;
  std::vector<int> cpus;
  for (int c = 0; c < CPU_SETSIZE; ++c) if (CPU_ISSET(c, &allowed)) cpus.push_back(c);
  int provided = 0;
  check(MPI_Init_thread(&argc, &argv, MPI_THREAD_SERIALIZED, &provided));
  check(MPI_Comm_set_errhandler(MPI_COMM_WORLD, MPI_ERRORS_RETURN));
  int rank, world; check(MPI_Comm_rank(MPI_COMM_WORLD, &rank)); check(MPI_Comm_size(MPI_COMM_WORLD, &world));
  if (provided < MPI_THREAD_SERIALIZED || cpus.size() != 2 || world != 2) {
    std::cerr << "Require two MPI ranks, two allowed CPUs per rank and SERIALIZED support\n";
    MPI_Abort(MPI_COMM_WORLD, 3); return 3;
  }
  pin(cpus[0]);
  char host[MPI_MAX_PROCESSOR_NAME]; int host_len;
  check(MPI_Get_processor_name(host, &host_len));
  char lib[MPI_MAX_LIBRARY_VERSION_STRING]; int lib_len;
  check(MPI_Get_library_version(lib, &lib_len));
  // Separate files preserve per-rank metadata; only rank0 emits timing rows.
  std::cout << "{\"type\":\"metadata\",\"rank\":" << rank
    << ",\"host\":" << json_string(std::string(host, host_len))
    << ",\"allowed_cpus\":[" << cpus[0] << ',' << cpus[1]
    << "],\"mpi_caller_cpu\":" << cpus[1] << ",\"other_spin_cpu\":" << cpus[0] << ",\"mpi_thread_level\":" << provided << ",\"mpi_library\":"
    << json_string(std::string(lib, lib_len)) << "}" << std::endl;
  constexpr int maximum = 64 << 20;
  std::vector<unsigned char> input(world * maximum, static_cast<unsigned char>(rank + 17));
  std::vector<unsigned char> output(world * maximum);
  auto operation = [&](int op, int bytes) {
    if (op == 3) check(MPI_Barrier(MPI_COMM_WORLD));
    else if (op == 0) check(MPI_Sendrecv(input.data(), bytes, MPI_BYTE, 1-rank, 17,
          output.data(), bytes, MPI_BYTE, 1-rank, 17, MPI_COMM_WORLD, MPI_STATUS_IGNORE));
    else if (op == 1) check(MPI_Allgather(input.data(), bytes, MPI_BYTE,
          output.data(), bytes, MPI_BYTE, MPI_COMM_WORLD));
    else check(MPI_Alltoall(input.data(), bytes, MPI_BYTE,
          output.data(), bytes, MPI_BYTE, MPI_COMM_WORLD));
  };
  Slot slot;
  std::thread worker([&] {
    pin(cpus[1]);
    uint64_t generation = 0;
    for (;;) {
      uint64_t next;
      while ((next = slot.posted.load(std::memory_order_acquire)) == generation) relax();
      generation = next;
      if (slot.operation == -1) break;
      if (slot.operation == 4) pin(cpus[slot.bytes]);
      else operation(slot.operation, slot.bytes);
      slot.completed.store(generation, std::memory_order_release);
    }
  });
  uint64_t generation = 0;
  auto invoke = [&](bool proxy, int op, int bytes) {
    if (!proxy) { operation(op, bytes); return; }
    slot.operation = op; slot.bytes = bytes;
    slot.posted.store(++generation, std::memory_order_release);
    // This synchronous probe deliberately spins its caller to measure queue
    // latency. It does NOT demonstrate compute/communication overlap or idle CPU.
    while (slot.completed.load(std::memory_order_acquire) != generation) relax();
  };
  const char* names[] = {"sendrecv", "allgather", "alltoall"};
  for (int op = 0; op != 3; ++op) for (int bytes : {1, 64, 4096, 65536, 1<<20, 16<<20, 64<<20}) {
    const int iterations = std::max(8, static_cast<int>(std::min<int64_t>(4096, (2LL<<30)/bytes)));
    for (int round = 0; round != 5; ++round) for (int order = 0; order != 2; ++order) {
      bool proxy = ((order + round) & 1) != 0;
      // Both modes execute MPI on the SAME physical CPU. The other CPU is
      // deliberately busy-polling in both modes; only the SPSC hop changes.
      pin(cpus[0]);
      invoke(true, 4, proxy ? 1 : 0);
      if (!proxy) pin(cpus[1]);
      for (int warm = 0; warm != 3; ++warm) invoke(proxy, op, bytes);
      invoke(proxy, 3, 0);
      const double begin = MPI_Wtime();
      for (int i = 0; i != iterations; ++i) invoke(proxy, op, bytes);
      double seconds = MPI_Wtime() - begin, maximum_seconds;
      // The worker is idle here; MPI calls never overlap across threads.
      check(MPI_Allreduce(&seconds, &maximum_seconds, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD));
      bool correct = true;
      for (size_t i = 0, end = static_cast<size_t>(bytes)*(op ? world : 1); i != end; ++i) {
        unsigned char expected = static_cast<unsigned char>((op ? i/bytes : 1-rank) + 17);
        correct &= output[i] == expected;
      }
      if (!correct) { std::cerr << "Payload validation failed\n"; MPI_Abort(MPI_COMM_WORLD, 4); }
      if (!rank) std::cout << "{\"type\":\"sample\",\"mode\":\"" << (proxy ? "spsc_proxy" : "direct_mpi")
        << "\",\"operation\":\"" << names[op] << "\",\"bytes_per_remote_peer\":" << bytes
        << ",\"iterations\":" << iterations << ",\"round\":" << round
        << ",\"seconds\":" << maximum_seconds << ",\"us\":" << maximum_seconds*1e6/iterations
        << ",\"remote_payload_GBps\":" << bytes*static_cast<double>(iterations)/maximum_seconds/1e9
        << ",\"correct\":true}" << std::endl;
    }
  }
  slot.operation = -1; slot.posted.store(++generation, std::memory_order_release);
  worker.join(); check(MPI_Finalize()); return 0;
}
