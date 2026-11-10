// SPDX-FileCopyrightText: 2026 HenryRenYz and vecops contributors
// SPDX-License-Identifier: MIT
#include "vecops/execution/ResourceWorkspace.h"
#include <algorithm>
#include <atomic>
#include <cstdlib>
#include <mutex>
#include <random>
#ifdef VECOPS_HAS_MEMORY_RUNTIME
#  include "vecops/memory/LinuxRuntime.h"
#  include "vecops/memory/ResourceModel.h"
#  include <unistd.h>
#endif

namespace vecops::execution {
namespace {
thread_local std::shared_ptr<WorkspaceArenaProvider> selected;
thread_local bool planned_workspace = false;
#ifdef VECOPS_HAS_MEMORY_RUNTIME
namespace m = memory::model;
template <class T>
T checked(m::Result<T> result) {
  if (!result) {
    const auto& error = std::get<m::Error>(result.state);
    throw WorkspaceArenaUnavailable(error.code + ": " + error.message);
  }
  return std::move(result).value();
}
std::optional<std::string> environment(const char* name) {
  const char* value = std::getenv(name);
  return value && *value ? std::optional<std::string>(value) : std::nullopt;
}
class ResourceProvider final : public WorkspaceArenaProvider {
  std::shared_ptr<m::MemoryRuntime> runtime;
  m::ResourceView view;
  std::vector<std::string> pools;
  std::string namespace_id;
  std::atomic<std::uint64_t> next{1};

public:
  ResourceProvider(std::shared_ptr<m::MemoryRuntime> r, std::vector<std::string> allowed)
    : runtime(std::move(r)) {
    if (!runtime)
      throw std::invalid_argument("workspace runtime is null");
    view = checked(runtime->view(checked(runtime->current_scope())));
    for (const auto& id : allowed)
      if (std::find(view.allocatable_pools.begin(), view.allocatable_pools.end(), id) == view.allocatable_pools.end())
        throw std::invalid_argument("workspace pool is outside allocation grant");
    for (const auto& id : view.allocatable_pools) {
      if (!allowed.empty() && std::find(allowed.begin(), allowed.end(), id) == allowed.end())
        continue;
      const auto pool =
        std::find_if(view.graph.pools.begin(), view.graph.pools.end(), [&](const auto& p) { return p.id == id; });
      bool host_executor = false, accessible = true;
      for (const auto& agent : view.graph.agents) {
        if (std::find(view.executable_agents.begin(), view.executable_agents.end(), agent.id) ==
              view.executable_agents.end() ||
            std::find(agent.capabilities.begin(), agent.capabilities.end(), "host_load_store") ==
              agent.capabilities.end())
          continue;
        host_executor = true;
        const bool direct = std::any_of(view.graph.access.begin(), view.graph.access.end(), [&](const auto& access) {
          return access.pool == id && access.agent == agent.id && access.mapping == m::MappingMode::direct &&
                 std::find(access.access.begin(), access.access.end(), m::AccessMode::read) != access.access.end() &&
                 std::find(access.access.begin(), access.access.end(), m::AccessMode::write) != access.access.end();
        });
        accessible &= direct;
      }
      if (host_executor && accessible && pool->address_space == "host")
        pools.push_back(id);
    }
    if (pools.empty())
      throw WorkspaceArenaUnavailable("no directly host-addressable workspace pool");
    std::random_device random;
    namespace_id =
      runtime->participant() + "/native-workspace/" + std::to_string(random()) + "-" + std::to_string(random());
  }
  nint_t capacity(BufferTier) const noexcept override {
    try {
      std::uint64_t largest = 0;
      for (const auto& p : runtime->snapshot(view).pools)
        if (std::find(pools.begin(), pools.end(), p.pool) != pools.end())
          largest = std::max(largest, p.allocatable_bytes);
      return static_cast<nint_t>(std::min<std::uint64_t>(largest, std::numeric_limits<nint_t>::max()));
    } catch (...) {
      return 0;
    }
  }
  WorkspaceArena allocate(BufferTier, nint_t bytes, nint_t alignment) override {
    if (bytes < 0 || alignment <= 0 || (alignment & (alignment - 1)))
      throw std::invalid_argument("invalid native workspace size/alignment");
    if (!bytes)
      return {};
    std::string failure = "workspace capacity exhausted";
    for (const auto& id : pools) {
      const auto& pool =
        *std::find_if(view.graph.pools.begin(), view.graph.pools.end(), [&](const auto& p) { return p.id == id; });
      std::uint64_t quantum = std::max<std::uint64_t>(alignment, pool.granularity_bytes.value_or(1));
      for (auto page : pool.page_sizes_bytes)
        quantum = std::max(quantum, page);
      const auto size = m::page_charge(bytes, quantum);
      if (size > static_cast<std::uint64_t>(std::numeric_limits<nint_t>::max()))
        throw std::overflow_error("native workspace exceeds host address range");
      m::BackingRequest request{namespace_id + "/" + std::to_string(next.fetch_add(1)),
                                runtime->participant(),
                                id,
                                size,
                                quantum,
                                m::BackingMode::private_memory,
                                {}};
      auto result = runtime->allocate(view, request);
      if (!result) {
        const auto& e = std::get<m::Error>(result.state);
        failure = e.code + ": " + e.message;
        continue;
      }
      auto owner = std::make_shared<m::Allocation>(std::move(result).value());
      auto* address = checked(owner->data());
      return {address, static_cast<nint_t>(size), std::move(owner)};
    }
    throw WorkspaceArenaUnavailable(failure);
  }
};
#endif
} // namespace
std::shared_ptr<const VecopsWorkspaceArenaProvider>
workspace_arena_provider_to_abi(std::shared_ptr<WorkspaceArenaProvider> provider) {
  if (!provider)
    throw std::invalid_argument("workspace provider is null");
  struct State {
    std::atomic<std::size_t> references{1};
    std::shared_ptr<WorkspaceArenaProvider> provider;
    VecopsWorkspaceArenaProvider abi{};
    static void retain(void* value) {
      ++static_cast<State*>(value)->references;
    }
    static void release(void* value) {
      auto* self = static_cast<State*>(value);
      if (self->references.fetch_sub(1) == 1)
        delete self;
    }
  };
  auto* state = new State;
  state->provider = std::move(provider);
  state->abi = {
    sizeof(VecopsWorkspaceArenaProvider),
    0,
    static_cast<std::uint64_t>(reinterpret_cast<std::uintptr_t>(state)),
    state,
    [](void* p, std::uint32_t tier) -> std::uint64_t {
      if (tier > 1)
        return 0;
      return std::max<nint_t>(0, static_cast<State*>(p)->provider->capacity(static_cast<BufferTier>(tier)));
    },
    [](void* p, std::uint32_t tier, std::uint64_t bytes, std::uint64_t alignment, VecopsWorkspaceArena* out,
       VecopsError* error) -> std::int32_t {
      try {
        if (!out || out->struct_size < sizeof(VecopsWorkspaceArena) || tier > 1 ||
            bytes > static_cast<std::uint64_t>(std::numeric_limits<nint_t>::max()) || !alignment ||
            (alignment & (alignment - 1)) || alignment > static_cast<std::uint64_t>(std::numeric_limits<nint_t>::max()))
          throw std::invalid_argument("invalid workspace ABI allocation");
        auto value = static_cast<State*>(p)->provider->allocate(static_cast<BufferTier>(tier), bytes, alignment);
        auto owner = std::make_unique<WorkspaceArena>(std::move(value));
        *out = {sizeof(VecopsWorkspaceArena), 0, owner->data, static_cast<std::uint64_t>(owner->capacity), owner.get()};
        owner.release();
        return VECOPS_STATUS_OK;
      } catch (const std::exception& failure) {
        if (error && error->struct_size >= sizeof(VecopsError)) {
          error->code = VECOPS_STATUS_EXECUTION_ERROR;
          const auto size = std::strlen(failure.what());
          error->message_required = size + 1;
          if (error->message && error->message_capacity) {
            const auto copied = std::min<std::size_t>(size, error->message_capacity - 1);
            std::memcpy(error->message, failure.what(), copied);
            error->message[copied] = 0;
          }
        }
        return VECOPS_STATUS_EXECUTION_ERROR;
      } catch (...) {
        if (error && error->struct_size >= sizeof(VecopsError)) {
          constexpr char message[] = "workspace provider failed";
          error->code = VECOPS_STATUS_EXECUTION_ERROR;
          error->message_required = sizeof(message);
          if (error->message && error->message_capacity) {
            const auto copied = std::min<std::size_t>(sizeof(message) - 1, error->message_capacity - 1);
            std::memcpy(error->message, message, copied);
            error->message[copied] = 0;
          }
        }
        return VECOPS_STATUS_EXECUTION_ERROR;
      }
    },
    [](void*, void* owner) { delete static_cast<WorkspaceArena*>(owner); },
    State::retain,
    State::release,
    0,
    nullptr};
  return std::shared_ptr<const VecopsWorkspaceArenaProvider>(&state->abi,
                                                             [state](const auto*) { State::release(state); });
}
std::shared_ptr<WorkspaceArenaProvider>
exchange_workspace_arena_provider(std::shared_ptr<WorkspaceArenaProvider> value) {
  return std::exchange(selected, std::move(value));
}
WorkspaceProviderScope::WorkspaceProviderScope(const VecopsWorkspaceArenaProvider* provider) {
  if (provider) {
    previous_ = exchange_workspace_arena_provider(workspace_arena_provider_from_abi(*provider));
    active_ = true;
  }
}
WorkspaceProviderScope::~WorkspaceProviderScope() {
  if (active_)
    exchange_workspace_arena_provider(std::move(previous_));
}
PlannedWorkspaceScope::PlannedWorkspaceScope()
  : previous_(std::exchange(planned_workspace, true)) {
}
PlannedWorkspaceScope::~PlannedWorkspaceScope() {
  planned_workspace = previous_;
}
std::shared_ptr<memory::model::MemoryRuntime> default_workspace_runtime() {
#ifdef VECOPS_HAS_MEMORY_RUNTIME
  struct State {
    std::mutex mutex;
    pid_t pid = getpid();
    std::shared_ptr<m::MemoryRuntime> runtime;
  };
  static State state;
  if (state.pid != getpid())
    throw WorkspaceArenaUnavailable("reopen a resource runtime after fork");
  std::lock_guard lock(state.mutex);
  if (!state.runtime) {
    m::LinuxRuntimeOptions options;
    options.participant = environment("VECOPS_MEMORY_PARTICIPANT").value_or("");
    options.capacity_directory = environment("VECOPS_MEMORY_CAPACITY_DIRECTORY");
    const auto hardware = environment("VECOPS_MEMORY_HARDWARE"), policy = environment("VECOPS_MEMORY_POLICY"),
               calibration = environment("VECOPS_MEMORY_CALIBRATION");
    if (!hardware && (policy || calibration))
      throw std::invalid_argument("workspace policy/calibration requires hardware");
    auto opened = hardware
                    ? m::open_linux_runtime(*hardware, policy, calibration, std::move(options))
                    : m::discover_linux_runtime(std::nullopt, m::LinuxCpuProfile::generic_cpu, std::move(options));
    state.runtime = std::shared_ptr<m::MemoryRuntime>(checked(std::move(opened)));
  }
  return state.runtime;
#else
  throw WorkspaceArenaUnavailable("native workspace allocation requires the memory runtime");
#endif
}
std::shared_ptr<WorkspaceArenaProvider>
resource_workspace_arena_provider(std::shared_ptr<memory::model::MemoryRuntime> runtime,
                                  std::vector<std::string> pools) {
#ifdef VECOPS_HAS_MEMORY_RUNTIME
  return std::make_shared<ResourceProvider>(std::move(runtime), std::move(pools));
#else
  (void)runtime;
  (void)pools;
  throw WorkspaceArenaUnavailable("native workspace allocation requires the memory runtime");
#endif
}
std::shared_ptr<WorkspaceArenaProvider> default_workspace_arena_provider() {
  if (planned_workspace)
    throw WorkspaceArenaUnavailable("recorded grid actions require declared buffers or borrowed workspace views");
  if (selected)
    return selected;
  // Resolve lazily: metadata-only plans must not allocate or open a runtime.
  static thread_local auto provider = resource_workspace_arena_provider(default_workspace_runtime());
  return provider;
}
} // namespace vecops::execution
