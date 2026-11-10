// SPDX-FileCopyrightText: 2026 HenryRenYz and vecops contributors
// SPDX-License-Identifier: MIT
/** @file KernelPlan.h
 * @brief Capture-owned C++ grids, explicit arguments and resident replay.
 *
 * A source __kernel__ builds a KernelPlan. Only registered grid bodies execute
 * on replay. Bodies must be stateless; inputs, scalars and prepared state enter
 * through explicit bindings. Tensor metadata retains its original C++ types.
 * Source grids complete synchronously and must not export/retain borrowed
 * tensor pointers or launch work that outlives grid completion. Asynchronous
 * or peer-shared ownership requires a specialized provider with explicit
 * completion/lifetime contracts; graph donation treats it conservatively.
 * A plan is immutable and shareable; each KernelInstance owns separate scratch,
 * prepared state and an allocation-free, non-reentrant replay frame.
 *
 * Grid completion is a conservative full-team dependency. Inactive lanes skip
 * computation, not synchronization. Resident replay requires a real team;
 * logical-task-only providers execute individual synchronous grids instead.
 * Hardware resource scopes belong inside the compiled grid body, on workers.
 */
#ifndef VECOPS_EXECUTION_KERNEL_PLAN_H
#define VECOPS_EXECUTION_KERNEL_PLAN_H

#include <atomic>
#include <cstring>
#include <memory>
#include <string>
#include <tuple>
#include <typeinfo>
#include <type_traits>
#include <vector>
#include "vecops/execution/Grid.h"
#include "vecops/runtime/KernelPlanAbi.h"
#include "vecops/execution/WorkspaceArena.h"
#include "vecops/execution/WorkspaceTensor.h"
#include "vecops/tensor/DataAccess.h"

namespace vecops::execution {

class KernelBuilder;
class KernelPlan;
class KernelInstance;
class KernelSequence;
class KernelProgram;

/** One active grid lane. programs() is synchronous; local references captured
 * by its nested callback remain valid. This context must not escape its body.
 * Put worker prologue/epilogue around programs(), and keep architectural state
 * in a lexical, compiler-checked execution-resource region on this worker.
 */
template <nint_t Parallelism, typename Chunk, typename Count>
class GridWorkerContext {
public:
  using Partition = decltype(plan_task_partition(meta::cint<Parallelism>, std::declval<Count>()));
  GridWorkerContext(nint_t lane, nint_t first, nint_t last, Chunk chunk, Count count, const Partition& partition)
    : lane_(lane)
    , begin_(first)
    , end_(last)
    , chunk_(chunk)
    , count_(count)
    , partition_(partition) {
  }
  [[nodiscard]] nint_t lane_id() const noexcept {
    return lane_;
  }
  [[nodiscard]] TaskContext<Parallelism> task() const noexcept {
    return TaskContext<Parallelism>{lane_};
  }
  template <typename Tensor>
  auto local(Tensor tensor) const {
    return task().local(tensor);
  }
  template <typename Fn>
  void programs(Fn&& body) const {
    using LaneIndex = meta::Dynamic<1, 0, Parallelism - 1>;
    visit_task_partition_shard(partition_, LaneIndex{lane_}, [&](const auto& shard) {
      const nint_t start = static_cast<nint_t>(shard.begin);
      const nint_t size = static_cast<nint_t>(shard.task_count);
      for (nint_t i = 0; i < size; ++i) {
        const nint_t program = start + i;
        const nint_t first = begin_ + program * static_cast<nint_t>(chunk_);
        const nint_t last = first + std::min(static_cast<nint_t>(chunk_), end_ - first);
        std::invoke(body, task(), RangeWorkItem<Chunk, Count>{program, count_, first, last, chunk_});
      }
    });
  }

private:
  nint_t lane_, begin_, end_;
  Chunk chunk_;
  Count count_;
  const Partition& partition_;
};

namespace kernel_plan_details {
struct Slot {
  void* address = nullptr;
  alignas(16) std::byte scalar[16]{};
  bool bound = false;
};
struct Frame {
  const void* owner = nullptr;
  std::shared_ptr<WorkspaceArenaProvider> provider;
  std::vector<Slot> slots;
  std::vector<std::shared_ptr<void>> states;
};
enum class SlotKind { Tensor, Scalar, Buffer };
struct SlotSpec {
  SlotKind kind;
  nint_t bytes = 0;
  nint_t alignment = 64;
  const std::type_info* scalar_type = nullptr;
  BufferPlacementPolicy placement = BufferPlacementPolicy::FastPreferred;
  std::string name;
  double estimated_traffic_bytes = 0;
  nint_t replicas = 1;
  nint_t worker_bytes = 0;
};
enum class AccessMode { Unknown, Read, Write, ReadWrite, Overwrite, CallTemporary };
struct ResourceAccess {
  std::size_t slot;
  AccessMode mode = AccessMode::Unknown;
  nint_t offset = 0, bytes = 0;
};
struct Resources {
  std::vector<ResourceAccess> buffers;
  std::vector<std::size_t> states;
};
struct StateFactory {
  Resources resources;
  bool transient_workspace = false;
  virtual ~StateFactory() = default;
  virtual std::shared_ptr<void> make(const Frame&) const = 0;
};
struct Action {
  std::string name;
  nint_t active = 0;
  void (*run)(const Action&, const Frame&, nint_t) = nullptr;
  virtual ~Action() = default;
};
struct PlanData {
  nint_t workers = 1;
  std::vector<SlotSpec> slots;
  std::vector<std::shared_ptr<const StateFactory>> factories;
  std::vector<std::shared_ptr<const Action>> actions;
  std::vector<Resources> action_resources;
  std::vector<std::pair<std::size_t, std::size_t>> consumes;
  LogicalBufferPlan buffers;
  std::vector<bool> reusable;
  std::vector<VecopsKernelInplaceV1> inplace;

};
} // namespace kernel_plan_details

/** Symbolic input/buffer view. It exposes metadata, never capture-time data. */
template <tensor::UnboundTensorLike Pattern>
class TensorArgument {
public:
  using ElementType = typename Pattern::ElementType;
  static constexpr int Ndim = Pattern::Ndim;
  TensorArgument(const void* owner, std::size_t slot, Pattern pattern)
    : owner_(owner)
    , slot_(slot)
    , pattern_(std::move(pattern)) {
  }
  [[nodiscard]] const Pattern& pattern() const noexcept {
    return pattern_;
  }
  [[nodiscard]] auto layout() const {
    return pattern_.layout();
  }
  template <int I>
  [[nodiscard]] auto size() const {
    return pattern_.template size<I>();
  }
  [[nodiscard]] nint_t size(int i) const {
    return pattern_.size(i);
  }
  [[nodiscard]] nint_t stride(int i) const {
    return pattern_.stride(i);
  }
  template <int I>
  [[nodiscard]] auto stride() const {
    return pattern_.template stride<I>();
  }
  [[nodiscard]] std::size_t slot() const noexcept {
    return slot_;
  }
  [[nodiscard]] const void* owner() const noexcept {
    return owner_;
  }
  template <typename... Indices>
  auto operator()(Indices... indices) const {
    return TensorArgument<decltype(pattern_(indices...))>{owner_, slot_, pattern_(indices...)};
  }
  [[nodiscard]] auto resolve(const kernel_plan_details::Frame& frame) const {
    auto* base = static_cast<ElementType*>(frame.slots[slot_].address);
    return tensor::make_tensor(pattern_.element_offset() ? base + pattern_.element_offset() : base, pattern_.layout());
  }

private:
  const void* owner_;
  std::size_t slot_;
  Pattern pattern_;
};

/** Grid-local memory effect. Overwrite promises every subsequently read byte
 * is initialized in this grid on every replay, including dynamic tails. These
 * wrappers preserve the original resolved tensor and all Const metadata. A
 * plain binding remains opaque and pins storage for the complete instance.
 * No ownership or alias permission is implied for external inputs.
 */
template <typename Argument, kernel_plan_details::AccessMode Mode>
class TensorAccess {
public:
  explicit TensorAccess(Argument argument) : argument_(std::move(argument)) {}
  auto owner() const noexcept { return argument_.owner(); }
  auto slot() const noexcept { return argument_.slot(); }
  const auto& pattern() const noexcept { return argument_.pattern(); }
  static constexpr auto access_mode = Mode;
  auto resolve(const kernel_plan_details::Frame& frame) const { return argument_.resolve(frame); }
private:
  Argument argument_;
};
template <tensor::UnboundTensorLike P>
auto read(TensorArgument<P> arg) {
  return TensorAccess<TensorArgument<P>, kernel_plan_details::AccessMode::Read>{std::move(arg)};
}
template <tensor::UnboundTensorLike P>
auto write(TensorArgument<P> arg) {
  static_assert(!std::is_const_v<typename P::ElementType>);
  return TensorAccess<TensorArgument<P>, kernel_plan_details::AccessMode::Write>{std::move(arg)};
}
template <tensor::UnboundTensorLike P>
auto read_write(TensorArgument<P> arg) {
  static_assert(!std::is_const_v<typename P::ElementType>);
  return TensorAccess<TensorArgument<P>, kernel_plan_details::AccessMode::ReadWrite>{std::move(arg)};
}
template <tensor::UnboundTensorLike P>
auto overwrite(TensorArgument<P> arg) {
  static_assert(!std::is_const_v<typename P::ElementType>);
  return TensorAccess<TensorArgument<P>, kernel_plan_details::AccessMode::Overwrite>{std::move(arg)};
}

/** Scratch whose contents never escape one grid invocation. Every byte read
 * inside that grid must first be written there, including each dynamic tail
 * and worker task. No incoming contents or factory initialization may be used.
 * The storage may be reused in gaps between grids and between replays.
 */
template <tensor::UnboundTensorLike P>
auto call_temporary(TensorArgument<P> arg) {
  static_assert(!std::is_const_v<typename P::ElementType>);
  return TensorAccess<TensorArgument<P>, kernel_plan_details::AccessMode::CallTemporary>{std::move(arg)};
}

/** Grid boundary in one builder. Valid only for metadata declarations. */
struct KernelStage {
  const void* owner;
  std::size_t index;
};

/** Runtime scalar slot: its value is read at execution, not plan construction. */
template <typename T>
class ScalarArgument {
  static_assert(std::is_arithmetic_v<T> && sizeof(T) <= 16);

public:
  using value_type = T;
  ScalarArgument(const void* owner, std::size_t slot)
    : owner_(owner)
    , slot_(slot) {
  }
  [[nodiscard]] std::size_t slot() const noexcept {
    return slot_;
  }
  [[nodiscard]] const void* owner() const noexcept {
    return owner_;
  }
  [[nodiscard]] T resolve(const kernel_plan_details::Frame& frame) const {
    T result;
    std::memcpy(&result, frame.slots[slot_].scalar, sizeof(result));
    return result;
  }

private:
  const void* owner_;
  std::size_t slot_;
};

/** State owned by an instance; factories run once after buffer allocation. */
template <typename T>
class StateArgument {
public:
  StateArgument(const void* owner, std::size_t slot)
    : owner_(owner)
    , slot_(slot) {
  }
  [[nodiscard]] const void* owner() const noexcept {
    return owner_;
  }
  [[nodiscard]] std::size_t state_slot() const noexcept { return slot_; }
  [[nodiscard]] const T& resolve(const kernel_plan_details::Frame& frame) const {
    return *static_cast<const T*>(frame.states[slot_].get());
  }

private:
  const void* owner_;
  std::size_t slot_;
};

/** Explicit immutable value. Authors must not hide borrowed pointers in T. */
template <typename T>
class KernelConstant {
  static_assert(std::is_copy_constructible_v<T> && !std::is_pointer_v<T>);
  static_assert(
    !requires(const T& value) { value.owner(); }, "runtime arguments must be bound directly, not hidden in a constant");
  static_assert(
    [] {
      if constexpr (requires { T::is_bound; })
        return !T::is_bound;
      else
        return true;
    }(),
    "bound tensor data must use an input slot");

public:
  explicit KernelConstant(T value)
    : value_(std::move(value)) {
  }
  [[nodiscard]] const T& resolve(const kernel_plan_details::Frame&) const {
    return value_;
  }

private:
  T value_;
};
template <typename T>
auto kernel_constant(T value) {
  return KernelConstant<T>{std::move(value)};
}

/** Owned argument packets are the only state passed to registered bodies. */
template <typename... Args>
class KernelBindings {
public:
  explicit KernelBindings(Args... args)
    : args_(std::move(args)...) {
  }
  void validate_owner(const kernel_plan_details::PlanData& plan, bool prepare = false) const {
    std::apply(
      [&](const auto&... args) {
        auto check = [&](const auto& arg) {
          if constexpr (requires { arg.owner(); }) {
            if (arg.owner() != &plan)
              throw std::invalid_argument("binding belongs to a different kernel plan");
            if constexpr (requires { arg.slot(); }) {
              if (arg.slot() >= plan.slots.size())
                throw std::invalid_argument("binding slot exceeds kernel plan capacity");
              if (prepare && plan.slots[arg.slot()].kind != kernel_plan_details::SlotKind::Buffer)
                throw std::invalid_argument("prepare requires patterns/constants, not runtime input values");
            }
            if constexpr (requires { arg.state_slot(); }) {
              if (arg.state_slot() >= plan.factories.size())
                throw std::invalid_argument("binding state exceeds kernel plan capacity");
            }
          }
        };
        (check(args), ...);
      },
      args_);
  }
  [[nodiscard]] kernel_plan_details::Resources resources() const {
    kernel_plan_details::Resources result;
    std::apply([&](const auto&... args) {
      auto collect = [&](const auto& arg) {
        if constexpr (requires { arg.pattern(); arg.slot(); }) {
          constexpr auto mode = [] {
            if constexpr (requires { std::remove_cvref_t<decltype(arg)>::access_mode; })
              return std::remove_cvref_t<decltype(arg)>::access_mode;
            else return kernel_plan_details::AccessMode::Unknown;
          }();
          const auto& pattern = arg.pattern();
          using Element = std::remove_const_t<typename std::remove_cvref_t<decltype(pattern)>::ElementType>;
          result.buffers.push_back({arg.slot(), mode,
            pattern.element_offset() * static_cast<nint_t>(sizeof(Element)),
            [&] {
              if constexpr (mode == kernel_plan_details::AccessMode::Overwrite)
                return workspace_tensor_details::storage_bytes<Element>(pattern.layout());
              else return nint_t{0};
            }()});
        } else if constexpr (requires { arg.state_slot(); }) {
          result.states.push_back(arg.state_slot());
        }
      };
      (collect(args), ...);
    }, args_);
    return result;
  }
  template <typename Fn>
  decltype(auto) apply(const kernel_plan_details::Frame& frame, Fn&& fn) const {
    return std::apply(
      [&](const auto&... args) -> decltype(auto) {
        auto resolve = [&](const auto& arg) -> decltype(auto) {
          if constexpr (tensor::is_nullopt_v<decltype(arg)>)
            return tensor::nullopt;
          else
            return arg.resolve(frame);
        };
        return std::invoke(std::forward<Fn>(fn), resolve(args)...);
      },
      args_);
  }

private:
  std::tuple<Args...> args_;
};
template <typename... Args>
auto kernel_bind(Args... args) {
  auto own = [](auto value) {
    if constexpr (tensor::is_nullopt_v<decltype(value)> ||
                  requires(const kernel_plan_details::Frame& frame) { value.resolve(frame); })
      return value;
    else
      return kernel_constant(std::move(value));
  };
  return KernelBindings<decltype(own(args))...>{own(std::move(args))...};
}

/** Instance-time scratch authority for the existing address-free ops::prepare
 * surface. Its arenas outlive the resulting prepared operator. No allocation
 * authority is exposed to grid bodies. Sites are conservatively independent.
 */
class KernelScratch {
public:
  KernelScratch(std::vector<kernel_plan_details::SlotSpec>& slots, std::vector<std::size_t>& indices)
    : declarations_(&slots)
    , indices_(&indices) {
  }
  KernelScratch(const kernel_plan_details::Frame& frame, const std::vector<std::size_t>& indices)
    : frame_(&frame)
    , bindings_(&indices) {
  }
  template <typename T, tensor::LayoutLike Layout>
  auto tensor(std::string_view name, Layout layout, WorkspaceTensorOptions options = {}) {
    workspace_tensor_details::validate_options(options);
    const auto bytes = workspace_tensor_details::storage_bytes<T>(layout);
    const auto alignment = std::max<nint_t>(64, workspace_tensor_details::allocation_alignment<T>(options));
    void* data = nullptr;
    if (declarations_) {
      indices_->push_back(declarations_->size());
      declarations_->push_back({kernel_plan_details::SlotKind::Buffer, bytes, alignment, nullptr, options.placement,
                                std::string(name), options.estimated_traffic_bytes});
    } else {
      if (cursor_ >= bindings_->size())
        throw std::logic_error("prepared scratch topology changed");
      const auto slot = (*bindings_)[cursor_++];
      const auto& declaration = static_cast<const kernel_plan_details::PlanData*>(frame_->owner)->slots[slot];
      if (declaration.bytes != bytes || declaration.alignment != alignment ||
          declaration.placement != options.placement)
        throw std::logic_error("prepared scratch shape changed during binding");
      data = frame_->slots[slot].address;
    }
    return tensor::make_tensor(static_cast<T*>(data), std::move(layout));
  }
  template <typename T, typename Shape>
    requires tensor::is_shape_v<Shape>
  auto tensor(std::string_view name, Shape shape, WorkspaceTensorOptions options = {}) {
    return tensor<T>(name, tensor::make_layout(std::move(shape)), options);
  }
  template <typename T, nint_t Replicas, tensor::LayoutLike Layout>
  auto worker_tensor(std::string_view name, Layout layout, WorkspaceTensorOptions options = {}) {
    static_assert(Replicas > 0 && (sizeof(T) & (sizeof(T) - 1)) == 0);
    const auto bytes = workspace_tensor_details::storage_bytes<T>(layout);
    const auto stride = buffer_plan_details::checked_align_up(
      bytes, std::max<nint_t>(64, workspace_tensor_details::allocation_alignment<T>(options)));
    auto result = tensor<T>(name,
      workspace_tensor_details::prepend_worker_layout(meta::cint<Replicas>, layout, stride / sizeof(T)), options);
    if (declarations_) {
      declarations_->back().replicas = Replicas;
      declarations_->back().worker_bytes = bytes;
    }
    return result;
  }
  template <typename T, nint_t Replicas, typename Shape>
    requires tensor::is_shape_v<Shape>
  auto worker_tensor(std::string_view name, Shape shape, WorkspaceTensorOptions options = {}) {
    return worker_tensor<T, Replicas>(name, tensor::make_layout(std::move(shape)), options);
  }
  void finish() const {
    if (bindings_ && cursor_ != bindings_->size())
      throw std::logic_error("prepared scratch topology changed");
  }

private:
  std::vector<kernel_plan_details::SlotSpec>* declarations_ = nullptr;
  std::vector<std::size_t>* indices_ = nullptr;
  const kernel_plan_details::Frame* frame_ = nullptr;
  const std::vector<std::size_t>* bindings_ = nullptr;
  std::size_t cursor_ = 0;
};

/** Prepared op plus its scratch owner; the forwarding call is inlineable. */
template <typename Op>
class PreparedOperation {
public:
  PreparedOperation(std::shared_ptr<KernelScratch> scratch, Op op)
    : scratch_(std::move(scratch))
    , op_(std::move(op)) {
  }
  template <typename... Args>
  VECOPS_INLINE decltype(auto) operator()(Args&&... args) const {
    return op_(std::forward<Args>(args)...);
  }
  [[nodiscard]] const Op& operation() const noexcept {
    return op_;
  }

private:
  std::shared_ptr<KernelScratch> scratch_;
  Op op_;
};

/** Immutable executable recipe. Default instances own independent scratch;
 * explicitly shared arena providers require embedding-side serialization.
 */
class KernelPlan {
public:
  [[nodiscard]] nint_t workers() const noexcept {
    return data_->workers;
  }
  [[nodiscard]] std::size_t grid_count() const noexcept {
    return data_->actions.size();
  }
  [[nodiscard]] std::size_t state_count() const noexcept {
    return data_->factories.size();
  }
  [[nodiscard]] const std::vector<VecopsKernelInplaceV1>& inplace_contracts() const noexcept {
    return data_->inplace;
  }
  /** Prepared logical storage and grid-level lifetimes; no addresses.
   * Each canonical path is a deterministic, unique frame-slot symbol followed
   * by its descriptive label. Repeated operator scratch labels remain distinct.
   */
  [[nodiscard]] const LogicalBufferPlan& buffer_plan() const noexcept { return data_->buffers; }
  /** True only after audited effects prove overwrite/reuse. Otherwise prepared
   * state may preserve initialized contents until its instance is destroyed. */
  [[nodiscard]] bool buffer_reusable(std::size_t index) const {
    return data_->reusable.at(data_->buffers.allocations.at(index).site.path_hash-1);
  }
  [[nodiscard]] std::size_t slot_count() const noexcept {
    return data_->slots.size();
  }

private:
  friend class KernelBuilder;
  friend class KernelInstance;
  friend class KernelProgram;
  explicit KernelPlan(std::shared_ptr<const kernel_plan_details::PlanData> data)
    : data_(std::move(data)) {
  }
  std::shared_ptr<const kernel_plan_details::PlanData> data_;
};

/** Plan-time declaration authority; no program is executed by this object. */
class KernelBuilder {
public:
  explicit KernelBuilder(nint_t workers);
  [[nodiscard]] nint_t parallelism() const {
    check_open();
    return data_->workers;
  }
  template <tensor::TensorLike Tensor>
  auto input(const Tensor& example);
  template <typename T>
  auto scalar();
  template <typename T, tensor::LayoutLike Layout>
  auto tensor(std::string_view name, Layout layout, WorkspaceTensorOptions options = {});
  template <typename T, typename Shape>
    requires tensor::is_shape_v<Shape>
  auto tensor(std::string_view name, Shape shape, WorkspaceTensorOptions options = {}) {
    return tensor<T>(name, tensor::make_layout(std::move(shape)), options);
  }
  template <typename T, nint_t Replicas, tensor::LayoutLike Layout>
  auto worker_tensor(std::string_view name, Layout layout);
  template <typename T, nint_t Replicas, typename Shape>
    requires tensor::is_shape_v<Shape>
  auto worker_tensor(std::string_view name, Shape shape) {
    return worker_tensor<T, Replicas>(name, tensor::make_layout(std::move(shape)));
  }
  template <typename Bindings, typename Factory>
  auto prepare(Bindings bindings, Factory factory);
  /** Prepare an ops object from address-free patterns once per instance. */
  template <nint_t Parallelism, typename Op, typename... Args>
  auto prepare_operation(std::string_view name, Op op, Args... args);

  /** Record a compact data-dependent program domain. count is a one-element
   * integer tensor written by an earlier grid; maximum is an owned static
   * capacity. Storage/prepare topology remains fixed while the live task count
   * may change on every replay. Program chunk is one.
   */
  template <nint_t Parallelism, meta::ValueInput Maximum, typename CountArgument, typename Bindings, typename Fn>
  KernelStage indirect_grid(std::string_view name, Maximum maximum, CountArgument count, Bindings bindings, Fn body);
  template <nint_t Parallelism, meta::ValueInput Begin, meta::ValueInput End, meta::ValueInput Chunk, typename Bindings,
            typename Fn>
  KernelStage grid(std::string_view name, Begin begin, End end, Chunk chunk, Bindings bindings, Fn body);
  template <nint_t Parallelism, meta::ValueInput Begin, meta::ValueInput End, meta::ValueInput Chunk, typename Bindings,
            typename Fn>
  KernelStage worker_grid(std::string_view name, Begin begin, End end, Chunk chunk, Bindings bindings, Fn body);
  template <typename Bindings, typename Fn>
  KernelStage single(std::string_view name, Bindings bindings, Fn body);
  /** Author-audited permission to bind an output to exactly the input view.
   * Every old input value must be read before its first overwrite, whether
   * through same-index worker ordering or an existing completed grid. Other
   * inputs must not overlap this view. This declaration never changes eager
   * bindings; the graph must prove ownership and last use independently.
   * Only complete external argument views of identical dtype/layout qualify.
   */
  template <tensor::UnboundTensorLike P, tensor::UnboundTensorLike Q>
  void allow_inplace(TensorArgument<P> input, TensorArgument<Q> output) {
    check_open();
    static_assert(std::is_same_v<std::remove_const_t<typename P::ElementType>,
                                std::remove_const_t<typename Q::ElementType>>);
    static_assert(!std::is_const_v<typename Q::ElementType>);
    if (input.owner() != data_.get() || output.owner() != data_.get() || input.slot() == output.slot() ||
        input.slot() >= data_->slots.size() || output.slot() >= data_->slots.size() ||
        data_->slots[input.slot()].kind != kernel_plan_details::SlotKind::Tensor ||
        data_->slots[output.slot()].kind != kernel_plan_details::SlotKind::Tensor ||
        input.pattern().element_offset() || output.pattern().element_offset())
      throw std::invalid_argument("inplace contract requires root external arguments in this plan");
    if constexpr (P::Ndim != Q::Ndim) {
      throw std::invalid_argument("inplace argument ranks differ");
    } else {
      for (int axis = 0; axis < P::Ndim; ++axis)
        if (input.size(axis) != output.size(axis) || input.stride(axis) != output.stride(axis))
          throw std::invalid_argument("inplace argument layouts differ");
    }
    if (input.slot() > UINT32_MAX || output.slot() > UINT32_MAX)
      throw std::overflow_error("inplace argument index exceeds ABI capacity");
    data_->inplace.push_back({static_cast<std::uint32_t>(input.slot()), static_cast<std::uint32_t>(output.slot())});
  }
  /** Assert the final local use, including aliases and prepared-state uses.
   * This is checked at finish(), never an eager free or external donation.
   * Unknown/retained state still prevents reuse even when consumed.
   */
  template <tensor::UnboundTensorLike P>
  void consume(TensorArgument<P> arg, KernelStage after) {
    check_open();
    if (arg.owner() != data_.get() || arg.slot() >= data_->slots.size() ||
        after.owner != data_.get() || after.index >= data_->actions.size())
      throw std::invalid_argument("consume resource/stage belongs to a different plan");
    data_->consumes.emplace_back(arg.slot(), after.index);
  }
  [[nodiscard]] KernelPlan finish();

private:
  void check_open() const;
  KernelStage record(std::shared_ptr<const kernel_plan_details::Action>, kernel_plan_details::Resources);
  std::shared_ptr<kernel_plan_details::PlanData> data_;
  bool finished_ = false;
};

/** Preallocated bindings and persistent resources for one plan.
 * Input addresses may change between calls. Their guarded metadata must match
 * the plan. set() retains no external owner: the caller must keep input/output
 * storage alive until replay returns. A graph owner must retain its Tensor owners.
 */
struct KernelBufferBinding {
  /** Exact canonical symbol from KernelPlan::buffer_plan(), not a debug label. */
  std::string name;
  void* address;
  std::uint64_t bytes, replica_stride;
  std::shared_ptr<void> owner;
};
class KernelInstance {
public:
  explicit KernelInstance(KernelPlan plan, const VecopsThreadPoolV1* pool = nullptr,
                          std::shared_ptr<WorkspaceArenaProvider> provider = default_workspace_arena_provider());
  /** Bind already planned storage. This constructor performs no placement and
   * never obtains tensor/workspace storage from an allocator. Each binding owner
   * survives all prepared-state destructors. Unknown/duplicate/missing buffers,
   * bad alignment or overlapping live virtual byte ranges are rejected. */
  KernelInstance(KernelPlan, const VecopsThreadPoolV1*, std::vector<KernelBufferBinding>);
  [[nodiscard]] bool running() const noexcept { return running_.test(std::memory_order_acquire); }
  KernelInstance(const KernelInstance&) = delete;
  KernelInstance& operator=(const KernelInstance&) = delete;
  ~KernelInstance();
  template <tensor::UnboundTensorLike Pattern>
  void set(TensorArgument<Pattern> arg, typename Pattern::ElementType* address);
  template <typename T>
  void set(ScalarArgument<T> arg, T value);
  void replay();
  /** Expert embedding protocol: enter, execute every real team lane, leave.
   * execute_on_team does not submit a team. leave must follow team completion,
   * including cancelled execution, and must be called exactly once per enter.
   */
  void enter();
  void execute_on_team(const GridTeam& team) const;
  void leave(bool successful) noexcept;
  [[nodiscard]] const KernelPlan& plan() const noexcept {
    return plan_;
  }
  [[nodiscard]] const VecopsThreadPoolV1* thread_pool() const noexcept {
    return pool_;
  }
  [[nodiscard]] std::uint64_t replay_count() const noexcept {
    return replay_count_;
  }

private:
  friend class KernelSequence;
  friend class KernelProgram;
  KernelInstance(KernelPlan plan, const VecopsThreadPoolV1* pool,
                 std::shared_ptr<WorkspaceArenaProvider> provider, const BufferPlacement& placement,
                 WorkspaceArena fast, WorkspaceArena slow);
  void initialize_frame();
  void bind_storage(const BufferPlacement&, WorkspaceArena fast, WorkspaceArena slow);
  void acquire();
  void release() noexcept;
  void run_lane(nint_t lane, const GridTeam& team) const;
  void validate_bindings() const;
  void run_synchronous();
  KernelPlan plan_;
  const VecopsThreadPoolV1* pool_;
  std::shared_ptr<WorkspaceArenaProvider> provider_;
  std::vector<WorkspaceArena> buffers_;
  bool external_buffers_ = false;
  kernel_plan_details::Frame frame_;
  std::atomic_flag running_ = ATOMIC_FLAG_INIT;
  std::uint64_t replay_count_ = 0;
};

/** Flatten consecutive prepared kernels into one resident-team submission.
 * Instances retain all resources. No wrapper/recipe/prepare runs on replay.
 * The team width is the maximum required by its members; smaller grids skip
 * computation on excess workers while all workers participate in barriers.
 * The sequence and its member instances are non-reentrant; bindings are updated
 * before replay. Appending the same instance twice preserves sequential reuse.
 */
class KernelSequence {
public:
  void append(std::shared_ptr<KernelInstance> instance);
  void replay();

private:
  std::vector<std::shared_ptr<KernelInstance>> instances_;
  std::vector<KernelInstance*> unique_;
  std::atomic_flag running_ = ATOMIC_FLAG_INIT;
};

/** Cold link of serial kernels using one BufferPlan and one pair of arenas.
 * Only buffers with an audited defining overwrite and known grid effects can
 * overlap across phases. Prepared state and unknown effects stay live for the
 * complete program, including construction/destruction. External tensors keep
 * caller-owned storage; this API never donates or mutates them implicitly.
 * Replay is one resident-team sequence; there is no placement or allocation.
 * Programs are non-reentrant. Bindings must be set before replay, and neither
 * a program nor any of its borrowed input/output storage may escape its owner.
 */
class KernelProgram {
public:
  explicit KernelProgram(std::vector<KernelPlan> plans, const VecopsThreadPoolV1* pool = nullptr,
                         std::shared_ptr<WorkspaceArenaProvider> provider = default_workspace_arena_provider());
  template <tensor::UnboundTensorLike P>
  void set(std::size_t phase, TensorArgument<P> arg, typename P::ElementType* address) {
    instances_.at(phase)->set(arg, address);
  }
  template <typename T>
  void set(std::size_t phase, ScalarArgument<T> arg, T value) { instances_.at(phase)->set(arg, value); }
  void replay() { sequence_.replay(); }
  [[nodiscard]] const LogicalBufferPlan& buffer_plan() const noexcept { return buffers_; }
  [[nodiscard]] const BufferPlacement& placement() const noexcept { return placement_; }
  [[nodiscard]] std::uint64_t replay_count(std::size_t phase) const { return instances_.at(phase)->replay_count(); }
private:
  LogicalBufferPlan buffers_;
  BufferPlacement placement_;
  std::vector<std::shared_ptr<KernelInstance>> instances_;
  KernelSequence sequence_;
};

} // namespace vecops::execution

namespace vecops::tensor {
/** Reinterpret an existing allocation with a typed layout, without a copy or
 * dtype conversion. offset is measured in elements from this view's origin.
 * The author guarantees that the new view stays inside the backing allocation.
 */
template <TensorLike Tensor, LayoutLike Layout>
auto layout_view(const Tensor& source, Layout layout, nint_t offset = 0) {
  if constexpr (Tensor::is_bound) {
    auto* base = source.data();
    return make_tensor(offset ? base + offset : base, layout);
  } else {
    using L = std::remove_cvref_t<Layout>;
    return tensor::Tensor<typename Tensor::ElementType, typename L::Shape, typename L::Strides, UnboundBinding>(
      UnboundBinding{source.element_offset() + offset}, layout);
  }
}
template <UnboundTensorLike Pattern, LayoutLike Layout>
auto layout_view(const execution::TensorArgument<Pattern>& source, Layout layout, nint_t offset = 0) {
  auto pattern = layout_view(source.pattern(), layout, offset);
  return execution::TensorArgument<decltype(pattern)>{source.owner(), source.slot(), std::move(pattern)};
}
template <typename Tensor, typename Shape>
  requires is_shape_v<Shape>
auto layout_view(const Tensor& source, Shape shape, nint_t offset = 0) {
  return layout_view(source, make_layout(shape), offset);
}
template <int I, int J, UnboundTensorLike Pattern>
auto transpose(const execution::TensorArgument<Pattern>& source) {
  auto pattern = transpose<I, J>(source.pattern());
  return execution::TensorArgument<decltype(pattern)>{source.owner(), source.slot(), std::move(pattern)};
}
/** Symbolic arguments participate in pattern preparation without addresses. */
template <UnboundTensorLike Pattern>
auto unbind(const execution::TensorArgument<Pattern>& arg) {
  return arg.pattern();
}
template <typename Compute, UnboundTensorLike Pattern, typename... Args>
auto input(const execution::TensorArgument<Pattern>& arg, Args... args) {
  return input<Compute>(arg.pattern(), std::move(args)...);
}
template <typename Compute, UnboundTensorLike Pattern, typename... Args>
auto output(const execution::TensorArgument<Pattern>& arg, Args... args) {
  return output<Compute>(arg.pattern(), std::move(args)...);
}
} // namespace vecops::tensor
#include "vecops/execution/details/KernelPlan.h"
#endif
