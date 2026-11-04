// SPDX-FileCopyrightText: 2026 HenryRenYz and vecops contributors
// SPDX-License-Identifier: MIT
/** @file SourceKernelPlan.h
 * @brief Generated source adapter support; decoding is outside resident grids.
 *
 * Define VECOPS_RECORDABLE_KERNEL in a source file whose __kernel__ accepts a
 * KernelBuilder and symbolic arguments. The compiler exports the optional plan
 * ABI. Ordinary eager calls cache instances; graph lowering retains a dedicated
 * prepared object and replays its internal grids directly.
 */
#ifndef VECOPS_EXECUTION_SOURCE_KERNEL_PLAN_H
#define VECOPS_EXECUTION_SOURCE_KERNEL_PLAN_H
#include <algorithm>
#include <array>
#include "vecops/execution/KernelPlan.h"
#include "vecops/runtime/KernelPlanAbi.h"

namespace vecops::execution::source_plan_details {

inline void error(VecopsError* output, const char* message) noexcept {
  if (!output || output->struct_size < sizeof(VecopsError))
    return;
  output->code = VECOPS_STATUS_EXECUTION_ERROR;
  const auto length = std::strlen(message);
  output->message_required = length + 1;
  if (output->message && output->message_capacity) {
    const auto n = std::min(length, output->message_capacity - 1);
    std::memcpy(output->message, message, n);
    output->message[n] = 0;
  }
}
template <typename Fn>
int32_t guarded(VecopsError* output, Fn&& fn) noexcept {
  try {
    std::invoke(std::forward<Fn>(fn));
    return VECOPS_STATUS_OK;
  } catch (const std::exception& exception) {
    error(output, exception.what());
  } catch (...) {
    error(output, "prepared source kernel failed");
  }
  return VECOPS_STATUS_EXECUTION_ERROR;
}

/** An owning, exact metadata snapshot. Scalar values and tensor addresses vary. */
class CallGuard {
  struct Value {
    uint32_t kind = 0, dtype = 0, device = 0;
    int32_t device_index = 0;
    uint64_t flags = 0;
    std::vector<int64_t> sizes, strides;
  };
  std::vector<Value> values_;
  const VecopsThreadPoolV1* pool_ = nullptr;
  uint64_t provider_identity_ = 0;
  uint32_t requested_threads_ = 0;

public:
  explicit CallGuard(const VecopsCall& call) {
    if (call.workspace || call.workspace_size)
      throw std::invalid_argument("recordable kernels own their workspace");
    pool_ = pool(call);
    const auto* arena = provider(call);
    provider_identity_ = arena ? arena->identity : 0;
    requested_threads_ = requested(call);
    values_.reserve(call.num_values);
    for (uint32_t i = 0; i < call.num_values; ++i) {
      const auto& item = call.values[i];
      Value value;
      value.kind = item.kind;
      if (item.kind == VECOPS_VALUE_TENSOR) {
        const auto& t = item.value.tensor;
        value.dtype = t.dtype;
        value.device = t.device_type;
        value.device_index = t.device_index;
        value.flags = t.flags;
        if (t.rank) {
          value.sizes.assign(t.sizes, t.sizes + t.rank);
          value.strides.assign(t.strides, t.strides + t.rank);
        }
      } else if (item.kind == VECOPS_VALUE_SCALAR)
        value.dtype = item.value.scalar.dtype;
      values_.push_back(std::move(value));
    }
  }
  static const VecopsThreadPoolV1* pool(const VecopsCall& call) {
    return call.context && call.context->struct_size >= offsetof(VecopsExecutionContext, thread_pool) + sizeof(void*) &&
               call.context->thread_pool
             ? call.context->thread_pool
             : vecops_cpu_thread_pool_v1();
  }
  static const VecopsWorkspaceArenaProvider* provider(const VecopsCall& call) {
    return call.context &&
               call.context->struct_size >= offsetof(VecopsExecutionContext, workspace_provider) + sizeof(void*)
             ? call.context->workspace_provider
             : nullptr;
  }
  static uint32_t requested(const VecopsCall& call) {
    return call.context &&
               call.context->struct_size >= offsetof(VecopsExecutionContext, requested_threads) + sizeof(uint32_t)
             ? call.context->requested_threads
             : 0;
  }
  bool matches(const VecopsCall& call) const {
    if (call.struct_size < sizeof(VecopsCall) || !call.values || call.num_values != values_.size() || call.workspace ||
        call.workspace_size || pool(call) != pool_ || requested(call) != requested_threads_)
      return false;
    const auto* arena = provider(call);
    if ((arena ? arena->identity : 0) != provider_identity_)
      return false;
    for (uint32_t i = 0; i < call.num_values; ++i) {
      const auto& item = call.values[i];
      const auto& expected = values_[i];
      if (item.struct_size < sizeof(VecopsValue) || item.kind != expected.kind)
        return false;
      if (item.kind == VECOPS_VALUE_TENSOR) {
        const auto& t = item.value.tensor;
        if (t.struct_size < sizeof(VecopsTensorView) || t.dtype != expected.dtype || t.device_type != expected.device ||
            t.device_index != expected.device_index || t.flags != expected.flags || t.rank != expected.sizes.size() ||
            (t.rank && (!t.sizes || !t.strides || !std::equal(expected.sizes.begin(), expected.sizes.end(), t.sizes) ||
                        !std::equal(expected.strides.begin(), expected.strides.end(), t.strides))))
          return false;
      } else if (item.kind == VECOPS_VALUE_SCALAR &&
                 (item.value.scalar.struct_size < sizeof(VecopsScalar) || item.value.scalar.dtype != expected.dtype))
        return false;
    }
    return true;
  }
};

template <tensor::TensorLike T>
auto argument(KernelBuilder& builder, const T& tensor) {
  return builder.input(tensor);
}
template <typename T>
  requires std::is_arithmetic_v<T>
auto argument(KernelBuilder& builder, T) {
  return builder.scalar<T>();
}
inline auto argument(KernelBuilder&, tensor::nullopt_t) {
  return tensor::nullopt;
}

template <tensor::UnboundTensorLike Pattern>
void bind(KernelInstance& instance, TensorArgument<Pattern> arg, const VecopsValue& value) {
  using T = typename Pattern::ElementType;
  const auto& tensor = value.value.tensor;
  auto* base = static_cast<std::byte*>(tensor.data);
  instance.set(arg, reinterpret_cast<T*>(tensor.byte_offset ? base + tensor.byte_offset : base));
}
template <typename T>
void bind(KernelInstance& instance, ScalarArgument<T> arg, const VecopsValue& value) {
  const auto& s = value.value.scalar.value;
  if constexpr (std::is_floating_point_v<T>)
    instance.set(arg, static_cast<T>(s.f64));
  else if constexpr (std::is_unsigned_v<T> || std::is_same_v<T, bool>)
    instance.set(arg, static_cast<T>(s.u64));
  else
    instance.set(arg, static_cast<T>(s.i64));
}
inline void bind(KernelInstance&, tensor::nullopt_t, const VecopsValue&) {
}

class PreparedBase {
public:
  explicit PreparedBase(const VecopsCall& call)
    : guard(call)
    , vector_lengths_(current_thread_vector_lengths()) {
    if (call.context)
      std::memcpy(&context_, call.context, std::min<std::size_t>(sizeof(context_), call.context->struct_size));
    context_.struct_size = sizeof(context_);
    context_.thread_pool = CallGuard::pool(call);
    if (const auto* provider = CallGuard::provider(call)) {
      if (provider->struct_size < offsetof(VecopsWorkspaceArenaProvider, place_buffers))
        throw std::invalid_argument("workspace provider ABI is truncated");
      provider_ = *provider;
      context_.workspace_provider = &provider_;
    }
  }
  virtual ~PreparedBase() = default;
  virtual void bind_call(const VecopsCall&, bool validate = true) = 0;
  virtual KernelInstance& instance() = 0;
  virtual const std::vector<VecopsKernelInplaceV1>& inplace_contracts() const = 0;
  CallGuard guard;
  VecopsPreparedKernelV1 abi() {
    return {
      sizeof(VecopsPreparedKernelV1),
      1,
      this,
      static_cast<uint32_t>(instance().plan().workers()),
      static_cast<uint32_t>(instance().plan().grid_count()),
      instance().thread_pool(),
      [](void* p, const VecopsCall* call, VecopsError* e) {
        return guarded(e, [&] {
          if (!call)
            throw std::invalid_argument("null prepared call");
          static_cast<PreparedBase*>(p)->bind_call(*call);
        });
      },
      [](void* p, const VecopsCall* call, VecopsError* e) {
        return guarded(e, [&] {
          if (!call)
            throw std::invalid_argument("null prepared call");
          static_cast<PreparedBase*>(p)->bind_call(*call, false);
        });
      },
      [](void* p, VecopsError* e) { return guarded(e, [&] { static_cast<PreparedBase*>(p)->instance().replay(); }); },
      [](void* p, VecopsError* e) { return guarded(e, [&] { static_cast<PreparedBase*>(p)->instance().enter(); }); },
      [](void* p, const VecopsTeamContextV1* team, VecopsError* e) {
        return guarded(e, [&] {
          if (!team || team->struct_size < sizeof(VecopsTeamContextV1))
            throw std::invalid_argument("invalid resident team");
          auto& self = *static_cast<PreparedBase*>(p);
          const ScopedArmVectorLengths lengths(self.vector_lengths_);
          apply_process_flush_subnormals();
          self.instance().execute_on_team(GridTeam{*team});
        });
      },
      [](void* p, uint32_t ok) { static_cast<PreparedBase*>(p)->instance().leave(ok != 0); },
      [](void* p) { delete static_cast<PreparedBase*>(p); },
      nullptr,
      &context_,
      static_cast<uint32_t>(inplace_contracts().size()),
      inplace_contracts().data()};
  }

private:
  VecopsExecutionContext context_{};
  VecopsWorkspaceArenaProvider provider_{};
  ArmVectorLengths vector_lengths_;
};

template <typename Arguments>
class Prepared final : public PreparedBase {
public:
  Prepared(const VecopsCall& call, KernelPlan plan, Arguments arguments)
    : PreparedBase(call)
    , arguments_(std::move(arguments))
    , instance_(std::move(plan), CallGuard::pool(call),
                CallGuard::provider(call) ? workspace_arena_provider_from_abi(*CallGuard::provider(call))
                                          : default_workspace_arena_provider()) {
    // Symbolic slots omit nullopt arguments. Export call-value indices,
    // never the private frame-slot indices used inside KernelBuilder.
    for (const auto& contract : instance_.plan().inplace_contracts()) {
      VecopsKernelInplaceV1 mapped{UINT32_MAX, UINT32_MAX};
      [&]<std::size_t... I>(std::index_sequence<I...>) {
        auto map = [&](const auto& arg, std::size_t index) {
          if constexpr (requires { arg.slot(); }) {
            if (arg.slot() == contract.input) mapped.input = static_cast<uint32_t>(index);
            if (arg.slot() == contract.output) mapped.output = static_cast<uint32_t>(index);
          }
        };
        (map(std::get<I>(arguments_), I), ...);
      }(std::make_index_sequence<std::tuple_size_v<Arguments>>{});
      if (mapped.input == UINT32_MAX || mapped.output == UINT32_MAX)
        throw std::logic_error("inplace contract does not refer to call arguments");
      inplace_.push_back(mapped);
    }
    bind_call(call);
  }
  const std::vector<VecopsKernelInplaceV1>& inplace_contracts() const override { return inplace_; }
  KernelInstance& instance() override {
    return instance_;
  }
  void bind_call(const VecopsCall& call, bool validate = true) override {
    if (validate && !guard.matches(call))
      throw std::invalid_argument("prepared kernel metadata/context changed; instantiate another plan");
    [&]<std::size_t... I>(std::index_sequence<I...>) {
      (source_plan_details::bind(instance_, std::get<I>(arguments_), call.values[I]), ...);
    }(std::make_index_sequence<std::tuple_size_v<Arguments>>{});
  }

private:
  Arguments arguments_;
  KernelInstance instance_;
  std::vector<VecopsKernelInplaceV1> inplace_;
};

/** Build once or execute a cached instance. A nonnull output transfers ownership. */
template <nint_t Workers, typename Build, typename... Args>
void launch(const VecopsCall* call, VecopsPreparedKernelV1* output, Build build, const Args&... args) {
  static_assert(std::is_empty_v<Build>);
  using Arguments = std::tuple<decltype(argument(std::declval<KernelBuilder&>(), args))...>;
  using Instance = Prepared<Arguments>;
  auto prepare = [&] {
    KernelBuilder builder{Workers};
    Arguments arguments{argument(builder, args)...};
    std::apply([&](auto... inputs) { build(builder, inputs...); }, arguments);
    return std::make_unique<Instance>(*call, builder.finish(), std::move(arguments));
  };
  if (output) {
    auto object = prepare();
    auto abi = object->abi();
    abi.struct_size = std::min<uint32_t>(output->struct_size, sizeof(abi));
    std::memcpy(output, &abi, abi.struct_size);
    object.release();
    return;
  }
  static thread_local std::array<std::unique_ptr<Instance>, 4> cache;
  for (auto& item : cache) {
    if (item && item->guard.matches(*call)) {
      item->bind_call(*call);
      item->instance().replay();
      return;
    }
  }
  auto object = prepare();
  object->instance().replay();
  // Round-robin replacement is bounded and never occurs on a matching replay.
  static thread_local std::size_t next = 0;
  cache[next++ % cache.size()] = std::move(object);
}
} // namespace vecops::execution::source_plan_details
#endif
