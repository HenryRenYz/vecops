// SPDX-FileCopyrightText: 2026 HenryRenYz and vecops contributors
// SPDX-License-Identifier: MIT
/** @file KernelPlan.h @brief Typed packet/loop lowering for resident plans. */
#ifndef VECOPS_EXECUTION_DETAILS_KERNEL_PLAN_H
#define VECOPS_EXECUTION_DETAILS_KERNEL_PLAN_H
namespace vecops::execution {
namespace kernel_plan_details {

template <typename Bindings, typename Factory>
struct TypedFactory final : StateFactory {
  using State = std::remove_cvref_t<decltype(std::declval<Bindings>().apply(std::declval<const Frame&>(),
                                                                            std::declval<Factory>()))>;
  Bindings bindings;
  Factory factory;
  TypedFactory(Bindings b, Factory f)
    : bindings(std::move(b))
    , factory(std::move(f)) {
    resources = bindings.resources();
  }
  std::shared_ptr<void> make(const Frame& frame) const override {
    return bindings.apply(frame, [&](auto&&... args) -> std::shared_ptr<void> {
      return std::make_shared<State>(std::invoke(factory, std::forward<decltype(args)>(args)...));
    });
  }
};

template <nint_t Parallelism, typename Op, typename... Args>
struct OperationFactory final : StateFactory {
  using Prepared = decltype(std::declval<const Op&>().template prepare<Parallelism>(
    std::declval<KernelScratch&>(), std::declval<std::string_view>(), std::declval<const Args&>()...));
  using State = PreparedOperation<Prepared>;
  std::string name;
  Op op;
  std::tuple<Args...> args;
  std::vector<std::size_t> scratch_slots;
  OperationFactory(std::vector<SlotSpec>& slots, std::string_view n, Op o, Args... a)
    : name(n)
    , op(std::move(o))
    , args(std::move(a)...) {
    KernelScratch declarations{slots, scratch_slots};
    std::apply(
      [&](const auto&... args) {
        // Address-free patterns are sufficient to declare all scratch. No
        // execution is allowed here; binding runs once after arena placement.
        [[maybe_unused]] auto prepared = op.template prepare<Parallelism>(declarations, name, args...);
      },
      args);
    if constexpr (requires { Prepared::transient_workspace; })
      transient_workspace = Prepared::transient_workspace;
    for (auto slot : scratch_slots)
      resources.buffers.push_back({slot, transient_workspace ? AccessMode::CallTemporary : AccessMode::Unknown});
  }
  std::shared_ptr<void> make(const Frame& frame) const override {
    auto scratch = std::make_shared<KernelScratch>(frame, scratch_slots);
    return std::apply(
      [&](const auto&... args) -> std::shared_ptr<void> {
        auto prepared = op.template prepare<Parallelism>(*scratch, name, args...);
        scratch->finish();
        return std::make_shared<State>(std::move(scratch), std::move(prepared));
      },
      args);
  }
};

template <nint_t Parallelism, typename Chunk, typename Count, typename Bindings, typename Fn, bool WorkerBody = false>
struct TypedGrid final : Action {
  nint_t begin, end;
  Chunk chunk;
  Count count;
  decltype(plan_task_partition(meta::cint<Parallelism>, std::declval<Count>())) partition;
  Bindings bindings;
  Fn body;
  TypedGrid(std::string_view label, nint_t first, nint_t last, Chunk c, Count n, Bindings b, Fn f)
    : begin(first)
    , end(last)
    , chunk(c)
    , count(n)
    , partition(plan_task_partition(meta::cint<Parallelism>, n))
    , bindings(std::move(b))
    , body(std::move(f)) {
    name = label;
    active = std::min<nint_t>(Parallelism, static_cast<nint_t>(count));
    run = [](const Action& action, const Frame& frame, nint_t lane) {
      const auto& self = static_cast<const TypedGrid&>(action);
      self.bindings.apply(frame, [&](auto&&... args) {
        GridWorkerContext<Parallelism, Chunk, Count> worker{lane,       self.begin, self.end,
                                                            self.chunk, self.count, self.partition};
        if constexpr (WorkerBody)
          std::invoke(self.body, worker, args...);
        else
          worker.programs([&](auto task, const auto& item) { std::invoke(self.body, task, item, args...); });
      });
    };
  }
};

template <nint_t Parallelism, typename CountArgument, typename Bindings, typename Fn>
struct TypedIndirectGrid final : Action {
  nint_t maximum;
  CountArgument count;
  Bindings bindings;
  Fn body;
  TypedIndirectGrid(std::string_view label, nint_t max, CountArgument c, Bindings b, Fn f)
    : maximum(max)
    , count(std::move(c))
    , bindings(std::move(b))
    , body(std::move(f)) {
    name = label;
    active = std::min<nint_t>(Parallelism, maximum);
    run = [](const Action& action, const Frame& frame, nint_t lane) {
      const auto& self = static_cast<const TypedIndirectGrid&>(action);
      const nint_t live = self.count.resolve(frame)(0);
      if (live < 0 || live > self.maximum)
        throw std::out_of_range("indirect grid count exceeds recorded capacity");
      using Count = meta::Dynamic<1, 0>;
      const Count count{live};
      const auto partition = plan_task_partition(meta::cint<Parallelism>, count);
      self.bindings.apply(frame, [&](auto&&... args) {
        GridWorkerContext<Parallelism, meta::Const<1>, Count> worker{lane, 0, live, meta::cint<1>, count, partition};
        worker.programs([&](auto task, const auto& item) { std::invoke(self.body, task, item, args...); });
      });
    };
  }
};
} // namespace kernel_plan_details

template <nint_t Parallelism, meta::ValueInput Maximum, typename CountArgument, typename Bindings, typename Fn>
KernelStage KernelBuilder::indirect_grid(std::string_view name, Maximum maximum, CountArgument count, Bindings bindings,
                                  Fn body) {
  check_open();
  static_assert(Parallelism > 0 && std::is_empty_v<Fn>, "indirect grids require positive width and a stateless body");
  if (Parallelism > data_->workers || static_cast<nint_t>(maximum) < 0 || count.size(0) != 1)
    throw std::invalid_argument("invalid indirect grid capacity or count shape");
  kernel_bind(count).validate_owner(*data_, false);
  bindings.validate_owner(*data_, false);
  using Grid = kernel_plan_details::TypedIndirectGrid<Parallelism, CountArgument, Bindings, Fn>;
  auto resources = bindings.resources();
  auto implicit = kernel_bind(read(count)).resources();
  resources.buffers.insert(resources.buffers.end(), implicit.buffers.begin(), implicit.buffers.end());
  return record(std::make_shared<Grid>(name, static_cast<nint_t>(maximum), std::move(count),
                                     std::move(bindings), std::move(body)), std::move(resources));
}

template <tensor::TensorLike Tensor>
auto KernelBuilder::input(const Tensor& example) {
  check_open();
  auto pattern = tensor::unbind(example);
  const auto slot = data_->slots.size();
  bool nonempty = true;
  for (int i = 0; i < Tensor::Ndim; ++i)
    nonempty &= example.size(i) != 0;
  data_->slots.push_back({kernel_plan_details::SlotKind::Tensor, nonempty ? 1 : 0});
  return TensorArgument<decltype(pattern)>{data_.get(), slot, std::move(pattern)};
}

template <typename T>
auto KernelBuilder::scalar() {
  check_open();
  const auto slot = data_->slots.size();
  data_->slots.push_back({kernel_plan_details::SlotKind::Scalar, 0, 64, &typeid(T)});
  return ScalarArgument<T>{data_.get(), slot};
}

template <typename T, tensor::LayoutLike Layout>
auto KernelBuilder::tensor(std::string_view name, Layout layout, WorkspaceTensorOptions options) {
  check_open();
  if (name.empty())
    throw std::invalid_argument("kernel buffer name is empty");
  const auto slot = data_->slots.size();
  const auto bytes = workspace_tensor_details::storage_bytes<T>(layout);
  workspace_tensor_details::validate_options(options);
  const auto alignment = std::max<nint_t>(64, workspace_tensor_details::allocation_alignment<T>(options));
  data_->slots.push_back({kernel_plan_details::SlotKind::Buffer, bytes, alignment, nullptr, options.placement,
                          std::string(name), options.estimated_traffic_bytes});
  auto pattern = tensor::make_unbound_tensor<T>(std::move(layout));
  return TensorArgument<decltype(pattern)>{data_.get(), slot, std::move(pattern)};
}

template <typename T, nint_t Replicas, tensor::LayoutLike Layout>
auto KernelBuilder::worker_tensor(std::string_view name, Layout layout) {
  static_assert(Replicas > 0);
  static_assert((sizeof(T) & (sizeof(T) - 1)) == 0);
  const auto bytes = workspace_tensor_details::storage_bytes<T>(layout);
  const auto aligned = workspace_plan_details::checked_align_up(bytes, std::max<nint_t>(64, alignof(T)));
  auto replicated = workspace_tensor_details::prepend_worker_layout(meta::cint<Replicas>, layout, aligned / sizeof(T));
  auto result = tensor<T>(name, std::move(replicated));
  data_->slots.back().replicas = Replicas;
  data_->slots.back().worker_bytes = bytes;
  return result;
}

template <typename Bindings, typename Factory>
auto KernelBuilder::prepare(Bindings bindings, Factory factory) {
  check_open();
  static_assert(std::is_empty_v<Factory>, "prepare factories must be stateless; use explicit bindings");
  bindings.validate_owner(*data_, true);
  using FactoryType = kernel_plan_details::TypedFactory<Bindings, Factory>;
  const auto slot = data_->factories.size();
  data_->factories.push_back(std::make_shared<FactoryType>(std::move(bindings), std::move(factory)));
  return StateArgument<typename FactoryType::State>{data_.get(), slot};
}

template <nint_t Parallelism, typename Op, typename... Args>
auto KernelBuilder::prepare_operation(std::string_view name, Op op, Args... args) {
  check_open();
  static_assert(Parallelism > 0);
  if (Parallelism > data_->workers)
    throw std::invalid_argument("prepared operation exceeds plan width");
  static_assert((!std::is_pointer_v<Args> && ...), "prepared arguments must own their metadata");
  static_assert(([] {
                  if constexpr (requires { Args::is_bound; })
                    return !Args::is_bound;
                  else
                    return !requires(const Args& arg) { arg.owner(); };
                }() &&
                 ...),
                "prepared operations require address-free patterns, not bound tensors or runtime slots");
  using Factory = kernel_plan_details::OperationFactory<Parallelism, Op, Args...>;
  const auto slot = data_->factories.size();
  const auto slots = data_->slots.size();
  try {
    data_->factories.push_back(std::make_shared<Factory>(data_->slots, name, std::move(op), std::move(args)...));
  } catch (...) {
    data_->slots.resize(slots);
    throw;
  }
  return StateArgument<typename Factory::State>{data_.get(), slot};
}

template <nint_t Parallelism, meta::ValueInput Begin, meta::ValueInput End, meta::ValueInput Chunk, typename Bindings,
          typename Fn>
KernelStage KernelBuilder::grid(std::string_view name, Begin begin, End end, Chunk chunk, Bindings bindings, Fn body) {
  static_assert(Parallelism > 0);
  static_assert(std::is_empty_v<Fn>, "grid bodies must be stateless; use explicit bindings");
  check_open();
  bindings.validate_owner(*data_);
  if (Parallelism > data_->workers)
    throw std::invalid_argument("grid lanes exceed resident team width");
  auto b = meta::to_value(begin);
  auto e = meta::to_value(end);
  auto c = meta::to_value(chunk);
  if (static_cast<nint_t>(c) <= 0 || static_cast<nint_t>(e) < static_cast<nint_t>(b))
    throw std::invalid_argument("invalid kernel grid range");
  // Validate subtraction before applying Meta arithmetic (which may be signed).
  if (static_cast<nint_t>(b) < 0 &&
      static_cast<nint_t>(e) > std::numeric_limits<nint_t>::max() + static_cast<nint_t>(b))
    throw std::overflow_error("kernel grid range overflows");
  auto count = ::vecops::ceil_div(e - b, c);
  using Grid = kernel_plan_details::TypedGrid<Parallelism, decltype(c), decltype(count), Bindings, Fn>;
  auto resources = bindings.resources();
  return record(std::make_shared<Grid>(name, static_cast<nint_t>(b), static_cast<nint_t>(e), c, count,
                                      std::move(bindings), std::move(body)), std::move(resources));
}

template <nint_t Parallelism, meta::ValueInput Begin, meta::ValueInput End, meta::ValueInput Chunk, typename Bindings,
          typename Fn>
KernelStage KernelBuilder::worker_grid(std::string_view name, Begin begin, End end, Chunk chunk, Bindings bindings, Fn body) {
  static_assert(Parallelism > 0);
  static_assert(std::is_empty_v<Fn>, "worker bodies must be stateless; use explicit bindings");
  check_open();
  bindings.validate_owner(*data_);
  if (Parallelism > data_->workers)
    throw std::invalid_argument("grid lanes exceed resident team width");
  auto b = meta::to_value(begin);
  auto e = meta::to_value(end);
  auto c = meta::to_value(chunk);
  if (static_cast<nint_t>(c) <= 0 || static_cast<nint_t>(e) < static_cast<nint_t>(b))
    throw std::invalid_argument("invalid worker grid range");
  if (static_cast<nint_t>(b) < 0 &&
      static_cast<nint_t>(e) > std::numeric_limits<nint_t>::max() + static_cast<nint_t>(b))
    throw std::overflow_error("kernel grid range overflows");
  auto count = ::vecops::ceil_div(e - b, c);
  using Grid = kernel_plan_details::TypedGrid<Parallelism, decltype(c), decltype(count), Bindings, Fn, true>;
  auto resources = bindings.resources();
  return record(std::make_shared<Grid>(name, static_cast<nint_t>(b), static_cast<nint_t>(e), c, count,
                                      std::move(bindings), std::move(body)), std::move(resources));
}

template <typename Bindings, typename Fn>
KernelStage KernelBuilder::single(std::string_view name, Bindings bindings, Fn body) {
  static_assert(std::is_empty_v<Fn>, "single bodies must be stateless; use explicit bindings");
  // A single has its own typed action, avoiding a captured forwarding closure.
  struct Single final : kernel_plan_details::Action {
    Bindings bindings;
    Fn body;
    Single(std::string_view label, Bindings b, Fn f)
      : bindings(std::move(b))
      , body(std::move(f)) {
      name = label;
      active = 1;
      run = [](const kernel_plan_details::Action& action, const kernel_plan_details::Frame& frame, nint_t) {
        const auto& self = static_cast<const Single&>(action);
        self.bindings.apply(frame, self.body);
      };
    }
  };
  check_open();
  bindings.validate_owner(*data_);
  auto resources = bindings.resources();
  return record(std::make_shared<Single>(name, std::move(bindings), std::move(body)), std::move(resources));
}

template <tensor::UnboundTensorLike Pattern>
void KernelInstance::set(TensorArgument<Pattern> arg, typename Pattern::ElementType* address) {
  acquire();
  const auto slot = arg.slot();
  if (arg.owner() != plan_.data_.get() || slot >= frame_.slots.size() ||
      plan_.data_->slots[slot].kind != kernel_plan_details::SlotKind::Tensor) {
    release();
    throw std::invalid_argument("invalid kernel tensor input slot");
  }
  frame_.slots[slot].address = const_cast<std::remove_const_t<typename Pattern::ElementType>*>(address);
  frame_.slots[slot].bound = true;
  release();
}

template <typename T>
void KernelInstance::set(ScalarArgument<T> arg, T value) {
  acquire();
  const auto slot = arg.slot();
  if (arg.owner() != plan_.data_.get() || slot >= frame_.slots.size() ||
      plan_.data_->slots[slot].kind != kernel_plan_details::SlotKind::Scalar ||
      *plan_.data_->slots[slot].scalar_type != typeid(T)) {
    release();
    throw std::invalid_argument("invalid kernel scalar input slot");
  }
  std::memcpy(frame_.slots[slot].scalar, &value, sizeof(value));
  frame_.slots[slot].bound = true;
  release();
}
} // namespace vecops::execution
#endif
