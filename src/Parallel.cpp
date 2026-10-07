// SPDX-FileCopyrightText: 2026 HenryRenYz and vecops contributors
// SPDX-License-Identifier: MIT
//
//

/** @file Parallel.cpp @brief Backend-neutral logical-task execution. */

#include "vecops/execution/Parallel.h"
#include "vecops/execution/FloatControl.h"

#include <algorithm>
#include <array>
#include <exception>
#include <limits>
#include <mutex>
#include <stdexcept>

namespace vecops::execution {

namespace {

bool valid_pool(const VecopsThreadPoolV1* pool) noexcept {
  return pool != nullptr && pool->struct_size >= offsetof(VecopsThreadPoolV1, flags) + sizeof(pool->flags) &&
         pool->abi_major == VECOPS_THREAD_POOL_ABI_MAJOR && pool->max_parallelism != nullptr &&
         pool->parallel_for != nullptr;
}

} // namespace

nint_t max_parallelism() noexcept {
  return 1;
}

namespace details {

nint_t max_parallelism(const VecopsThreadPoolV1* pool) noexcept {
  if (!valid_pool(pool))
    return execution::max_parallelism();
  return std::max<nint_t>(1, static_cast<nint_t>(pool->max_parallelism(pool->context)));
}

void parallel_for_erased(nint_t requested_threads, void* object, ParallelBody body) {
  if (body == nullptr)
    return;

  (void)requested_threads;
  apply_process_flush_subnormals();
  body(object, ParallelContext{0, 1});
}

void parallel_tasks_erased(const VecopsThreadPoolV1* pool, nint_t task_count, void* object, ParallelBody body) {
  if (body == nullptr || task_count <= 0)
    return;
  if (static_cast<std::uint64_t>(task_count) > std::numeric_limits<std::uint32_t>::max())
    throw std::invalid_argument("vecops logical task count exceeds the thread-pool ABI");

  if (valid_pool(pool)) {
    struct CallbackState {
      void* object;
      ParallelBody body;
      std::exception_ptr exception;
      std::mutex mutex;
    } state{object, body};
    std::array<char, 512> message{};
    VecopsError error{sizeof(VecopsError), 0, message.data(), message.size(), 0};
    const auto status = pool->parallel_for(
      pool->context, static_cast<std::uint32_t>(task_count), &state,
      [](void* opaque, std::uint32_t task_id, std::uint32_t count) {
        auto& callback = *static_cast<CallbackState*>(opaque);
        try {
          apply_process_flush_subnormals();
          callback.body(callback.object,
                        ParallelContext{static_cast<nint_t>(task_id), static_cast<nint_t>(count)});
        } catch (...) {
          std::lock_guard lock(callback.mutex);
          if (callback.exception == nullptr)
            callback.exception = std::current_exception();
        }
      },
      &error);
    if (state.exception != nullptr)
      std::rethrow_exception(state.exception);
    if (status != VECOPS_STATUS_OK)
      throw std::runtime_error(message[0] == '\0' ? "embedding thread pool failed" : message.data());
    return;
  }

  apply_process_flush_subnormals();
  for (nint_t task = 0; task < task_count; ++task)
    body(object, ParallelContext{task, task_count});
}

} // namespace details
} // namespace vecops::execution
