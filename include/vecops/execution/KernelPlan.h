// SPDX-FileCopyrightText: 2026 HenryRenYz and vecops contributors
// SPDX-License-Identifier: MIT
/** @file KernelPlan.h
 * @brief Capture-owned C++ grids, explicit arguments and resident replay.
 *
 * A source __kernel__ builds a KernelPlan. Only registered grid bodies execute
 * on replay. Bodies must be stateless; inputs, scalars and prepared state enter
 * through explicit bindings. Tensor metadata retains its original C++ types.
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
#include "vecops/execution/WorkspaceArena.h"
#include "vecops/execution/WorkspaceTensor.h"
#include "vecops/tensor/DataAccess.h"

namespace vecops::execution {

class KernelBuilder;
class KernelPlan;
class KernelInstance;
class KernelSequence;

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
  WorkspacePlacementPolicy placement = WorkspacePlacementPolicy::FastPreferred;
};
struct StateFactory {
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
            if (prepare) {
              if constexpr (requires { arg.slot(); }) {
                if (plan.slots[arg.slot()].kind != kernel_plan_details::SlotKind::Buffer)
                  throw std::invalid_argument("prepare requires patterns/constants, not runtime input values");
              }
            }
          }
        };
        (check(args), ...);
      },
      args_);
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
  auto tensor(std::string_view, Layout layout) {
    const auto bytes = workspace_tensor_details::storage_bytes<T>(layout);
    void* data = nullptr;
    if (declarations_) {
      indices_->push_back(declarations_->size());
      declarations_->push_back({kernel_plan_details::SlotKind::Buffer, bytes, std::max<nint_t>(64, alignof(T))});
    } else {
      if (cursor_ >= bindings_->size())
        throw std::logic_error("prepared scratch topology changed");
      const auto slot = (*bindings_)[cursor_++];
      const auto& declaration = static_cast<const kernel_plan_details::PlanData*>(frame_->owner)->slots[slot];
      if (declaration.bytes != bytes || declaration.alignment < static_cast<nint_t>(alignof(T)))
        throw std::logic_error("prepared scratch shape changed during binding");
      data = frame_->slots[slot].address;
    }
    return tensor::make_tensor(static_cast<T*>(data), std::move(layout));
  }
  template <typename T, typename Shape>
    requires tensor::is_shape_v<Shape>
  auto tensor(std::string_view name, Shape shape) {
    return tensor<T>(name, tensor::make_layout(std::move(shape)));
  }
  template <typename T, nint_t Replicas, tensor::LayoutLike Layout>
  auto worker_tensor(std::string_view name, Layout layout) {
    static_assert(Replicas > 0 && (sizeof(T) & (sizeof(T) - 1)) == 0);
    const auto bytes = workspace_tensor_details::storage_bytes<T>(layout);
    const auto stride = workspace_plan_details::checked_align_up(bytes, std::max<nint_t>(64, alignof(T)));
    return tensor<T>(name,
                     workspace_tensor_details::prepend_worker_layout(meta::cint<Replicas>, layout, stride / sizeof(T)));
  }
  template <typename T, nint_t Replicas, typename Shape>
    requires tensor::is_shape_v<Shape>
  auto worker_tensor(std::string_view name, Shape shape) {
    return worker_tensor<T, Replicas>(name, tensor::make_layout(std::move(shape)));
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
  [[nodiscard]] std::size_t slot_count() const noexcept {
    return data_->slots.size();
  }

private:
  friend class KernelBuilder;
  friend class KernelInstance;
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
  void indirect_grid(std::string_view name, Maximum maximum, CountArgument count, Bindings bindings, Fn body);
  template <nint_t Parallelism, meta::ValueInput Begin, meta::ValueInput End, meta::ValueInput Chunk, typename Bindings,
            typename Fn>
  void grid(std::string_view name, Begin begin, End end, Chunk chunk, Bindings bindings, Fn body);
  template <nint_t Parallelism, meta::ValueInput Begin, meta::ValueInput End, meta::ValueInput Chunk, typename Bindings,
            typename Fn>
  void worker_grid(std::string_view name, Begin begin, End end, Chunk chunk, Bindings bindings, Fn body);
  template <typename Bindings, typename Fn>
  void single(std::string_view name, Bindings bindings, Fn body);
  [[nodiscard]] KernelPlan finish();

private:
  void check_open() const;
  std::shared_ptr<kernel_plan_details::PlanData> data_;
  bool finished_ = false;
};

/** Preallocated bindings and persistent resources for one plan.
 * Input addresses may change between calls. Their guarded metadata must match
 * the plan. set() retains no external owner: the caller must keep input/output
 * storage alive until replay returns. A graph owner must retain its Tensor owners.
 */
class KernelInstance {
public:
  explicit KernelInstance(KernelPlan plan, const VecopsThreadPoolV1* pool = nullptr,
                          std::shared_ptr<WorkspaceArenaProvider> provider = default_workspace_arena_provider());
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
  void acquire();
  void release() noexcept;
  void run_lane(nint_t lane, const GridTeam& team) const;
  void validate_bindings() const;
  void run_synchronous();
  KernelPlan plan_;
  const VecopsThreadPoolV1* pool_;
  std::shared_ptr<WorkspaceArenaProvider> provider_;
  std::vector<WorkspaceArena> buffers_;
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
