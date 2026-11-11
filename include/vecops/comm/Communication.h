// SPDX-FileCopyrightText: 2026 HenryRenYz and vecops contributors
// SPDX-License-Identifier: MIT
#pragma once
/** @file Framework-neutral, coherent Linux shared-memory collectives.
 * Bootstrap runs once through the embedding's exchange callback. No foreign
 * virtual addresses cross that boundary. Plans own layouts, not user buffers;
 * a synchronous execute retains access until return. Output regions must be
 * mutually disjoint; input/output aliasing is supported because inputs are
 * published before outputs are materialized. Instances are serialized
 * per communicator, and an error poisons the communicator before storage reuse.
 */
#include <array>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>
#include <string>
#include <vector>
#include "vecops/runtime/CallAbi.h"

namespace vecops::memory::model { class MemoryRuntime; }
namespace vecops::comm {
enum class Operation : uint32_t { Barrier, Broadcast, AllGather, AllToAll, SumFloat32 };
struct Layout {
  std::vector<int64_t> sizes, strides; // nonnegative element strides
  size_t element_bytes = 1;
};
struct Specification {
  Operation operation = Operation::Barrier;
  int root = 0;
  uint64_t element_type = 0; // embedding's wire-type identity; zero means opaque bytes
  std::vector<Layout> inputs, outputs;
  // Optional cold plan override, not alias permission. Must match the number
  // of configured levels and agree across ranks. Empty uses platform modes.
  std::vector<int> copy_levels;
};
struct Options {
  uint32_t threads = 1;
  // Root payload allocation uses this exact runtime and its shared authority.
  // The pool is an opaque public resource ID; backend NUMA binding belongs to
  // MemoryRuntime. Imported page aliases never incur a second root charge.
  std::shared_ptr<memory::model::MemoryRuntime> memory_runtime;
  std::string payload_pool;
  std::vector<int> locality_groups;
  // Coarse-to-fine nested partitions. A copy level makes one distributed
  // replica/source in each destination partition; share levels reuse the
  // preceding level's storage. Empty levels preserve locality_groups policy.
  std::vector<std::vector<int>> replica_levels;
  std::vector<int> copy_levels; // 0 share, 1 copy; one per replica level
  size_t initial_capacity = 1 << 20;
  size_t relay_threshold = 1 << 20;
  size_t chunk_bytes = 256 << 10;
  uint64_t timeout_ms = 120000;
  bool relay = true;
  // Auto recognizes equal interleaved AllGather outputs and gives complete
  // destination rows to one worker. Other layouts retain the generic path.
  // Every scheduling path uses vec byte transport with cold ISA selection.
  bool row_owned_copy = true;
  /** AllGather may copy its own snapshot before peers arrive, then validate
   * descriptors after the ready acquire. Previous readers must still retire
   * first; no output/remote inbox is written before descriptor validation.
   * This overlaps local publication with peer computation, not background DMA.
   */
  bool early_snapshot = false;
};
struct Statistics {
  // Logical explicit-copy payload, not read+write traffic or fabric bytes.
  uint64_t calls = 0, plans = 0, capacity_growths = 0, copy_bytes = 0;
  // Subset of copy_bytes processed with row ownership. Consumer kernels and
  // implicit loads are not counted here. All explicit copies use vec.
  uint64_t row_copy_calls = 0, row_copy_bytes = 0;
};
using Exchange = std::function<std::vector<std::string>(const std::string&)>;
class Plan;
class ExchangeBuffer;
enum class PagePlacement : uint32_t { Source, Destination };
class Communicator : public std::enable_shared_from_this<Communicator> {
public:
  /** All participants must call exchange in the same bootstrap order. */
  static std::shared_ptr<Communicator> create(int rank, int size, Options options,
      Exchange exchange, const VecopsThreadPoolV1* executor = nullptr);
  ~Communicator();
  /** A deferred plan registers demand without growing backing, even before
   * the first invocation. The cold provisioner must close its capacity before
   * a sealed execution. */
  std::shared_ptr<Plan> prepare(Specification specification, bool defer_capacity = false);
  /** Collectively provision the maximum publication size and the maximum
   * relayed payload for each registered source, rounded to the backing page
   * size. The embedding first stops submissions and drains its local queue.
   * All ranks must enter in the same cold order,
   * with the same completed invocation epoch. A control Barrier or earlier
   * payload invocation does not disable this preparation phase.
   *
   * No publication/borrowed consumer may remain active. The protocol proves
   * all previous readers stopped, removes every imported publication alias,
   * and only then retires/rebuilds roots. Existing publication contents and
   * borrowed addresses are invalidated; user input/output tensors are not.
   * New descriptors are published and all peer mappings acknowledged before
   * return. With growth allowed, returns the SAME bool on every rank: true if
   * any rank allocated publication/relay roots. With growth disabled, returns
   * true before changing backing or aliases when any rank needs to grow.
   * False means the registered capacity and peer mappings are ready.
   * Failures poison the communicator and retain unproven owners as usual.
   * No exchange, allocation or remapping is added to unchanged hot replay.
   */
  bool provision_registered_capacity(bool allow_growth = true);
  /** Cold all-rank AND of a local planning/admission result, using only the
   * metadata exchange. Call in the same order after stopping local submissions
   * and draining work. False is a normal consensus result, not an abort. This
   * does not allocate/map payloads, advance invocation epochs, or run a data
   * collective. Exchange failure or inconsistent call order poisons the group.
   */
  bool preparation_consensus(bool local_success);
  /** Collectively create fixed page aliases outside capture/replay. Producers
   * write peer-major blocks; consumers see source-major blocks at stable VAs.
   * Blocks are page padded, so callers must honor block_stride(). No payload
   * copy or mmap occurs in begin/publish/retire. Explicit producer/consumer
   * lifetime and read-only consumption are required; this is not a c10d output.
   */
  std::shared_ptr<ExchangeBuffer> prepare_exchange(size_t block_bytes,
      uint32_t slots = 1, PagePlacement placement = PagePlacement::Destination);
  Statistics statistics() const;
  /** Freeze capacity and plan registration after explicit model preparation.
   * Existing prepared plans remain reusable with new buffer addresses. */
  void seal();
  /** Collective cold shutdown after the embedding stops every local reader.
   * Releases roots only after every rank certifies that it has stopped.
   * Abort poisons data operations but does not replace this stopped handshake;
   * peers may collectively close an idle aborted communicator. A missing peer
   * or active exchange lease prevents refund and retains roots conservatively.
   * Closing ends PageExchange access permission and removes its page aliases;
   * producer/consumer tensors must not be used after process-group shutdown.
   */
  void close();
  void abort() noexcept;
  int rank() const noexcept;
  int size() const noexcept;
private:
  struct State;
  std::unique_ptr<State> state_;
  explicit Communicator(std::unique_ptr<State>);
  friend class Plan;
  friend class ExchangeBuffer;
};
class Plan {
public:
  using Consumer = std::function<void(const void* const*, size_t)>;
  ~Plan();
  /** Pointer arrays follow Specification order; no hot allocation or layout
   * construction. Null pointers are valid only for empty layouts. */
  void execute(void* const* outputs, const void* const* inputs);
  /** AllGather into a synchronous read-only lease instead of private outputs.
   * The pointers describe contiguous source snapshots in rank order, with the
   * same byte counts as specification().outputs. The callback must finish all
   * readers before returning, may not retain addresses or reenter this
   * communicator, and may write only separately owned storage. Exceptions
   * poison the communicator. Publication, optional domain replicas and lease
   * retirement are one native graph effect; no full gathered tensor is made.
   */
  void consume(const void* const* inputs, const Consumer& consumer);
  /** Execute on an already resident physical team. Every lane must enter in
   * the same order; idle copy lanes still join dependencies. Lane zero owns
   * protocol publication/polling while other lanes wait locally. No nested
   * pool submission occurs. A team consumer is invoked on ALL lanes and must
   * join every reader before return; it may use the same team's barriers.
   * Exceptions poison the communicator; the embedding must cancel its team
   * when a lane throws. Arrays/consumer and their owners outlive all lanes.
   */
  using TeamConsumer = std::function<void(const void* const*, size_t, const VecopsTeamContextV1&)>;
  void execute_on_team(void* const* outputs, const void* const* inputs,
      const VecopsTeamContextV1& team, const TeamConsumer* consumer = nullptr);
  /** The retained provider and maximum width used by this plan. */
  const VecopsThreadPoolV1* thread_pool() const noexcept;
  uint32_t workers() const noexcept;
  const Specification& specification() const noexcept;
private:
  struct State;
  std::unique_ptr<State> state_;
  explicit Plan(std::unique_ptr<State>);
  void execute_impl(void* const*, const void* const*, const Consumer*,
      const VecopsTeamContextV1* = nullptr, const TeamConsumer* = nullptr);
  friend class Communicator;
};
class ExchangeBuffer {
public:
  ~ExchangeBuffer();
  size_t block_stride() const noexcept;
  size_t block_bytes() const noexcept;
  uint32_t slots() const noexcept;
  void* producer(uint32_t slot) const;
  const void* consumer(uint32_t slot) const;
  /** begin waits for every previous reader of this slot. publish releases
   * producer writes and acquires every peer's publication. retire follows the
   * last consumer, including asynchronous/native worker reads. A slot follows
   * begin -> producer writes -> publish -> read-only consumers -> retire.
   * Different slots may remain live together; no unbounded version creation.
   * Distinct slots have independent phase locks; a wait on one slot cannot
   * prevent retirement of another. Same-slot phase calls remain serialized.
   * Protocol misuse or timeout poisons the buffer; abort wakes peer waiters.
   */
  void begin(uint32_t slot);
  void publish(uint32_t slot);
  void retire(uint32_t slot);
  /** Cold all-participant stopped handshake, including after abort. Active
   * producer/consumer leases must first be retired; no further phase calls are
   * legal after close. A missing stopped acknowledgement remains a failure.
   */
  void close();
  void abort() noexcept;
private:
  struct State;
  std::unique_ptr<State> state_;
  explicit ExchangeBuffer(std::unique_ptr<State>);
  friend class Communicator;
};
} // namespace vecops::comm
