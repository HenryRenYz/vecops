// SPDX-FileCopyrightText: 2026 HenryRenYz and vecops contributors
// SPDX-License-Identifier: MIT

#include "vecops/runtime/CallAbi.h"

#include <algorithm>
#include <cstring>
#include <exception>
#include <mutex>

#include <omp.h>

namespace {

uint32_t maximum(void*) {
  return omp_in_parallel() ? 1u : static_cast<uint32_t>(std::max(1, omp_get_max_threads()));
}

uint32_t in_parallel(void*) {
  return omp_in_parallel() ? 1u : 0u;
}

void set_error(VecopsError* error, const char* message) noexcept {
  if (error == nullptr || error->struct_size < sizeof(VecopsError))
    return;
  error->code = VECOPS_STATUS_EXECUTION_ERROR;
  const auto size = std::strlen(message);
  error->message_required = size + 1;
  if (error->message == nullptr || error->message_capacity == 0)
    return;
  const auto copied = std::min(size, error->message_capacity - 1);
  std::memcpy(error->message, message, copied);
  error->message[copied] = '\0';
}

int32_t parallel_for(void*, uint32_t task_count, void* body_context, VecopsParallelTaskFn body,
                     VecopsError* error) noexcept {
  if (body == nullptr || task_count == 0)
    return VECOPS_STATUS_OK;
  try {
    if (task_count == 1 || omp_in_parallel()) {
      for (uint32_t task = 0; task < task_count; ++task)
        body(body_context, task, task_count);
      return VECOPS_STATUS_OK;
    }
    std::exception_ptr exception;
#pragma omp parallel for schedule(static) shared(exception)
    for (uint32_t task = 0; task < task_count; ++task) {
      try {
        body(body_context, task, task_count);
      } catch (...) {
#pragma omp critical(vecops_openmp_pool_exception)
        {
          if (exception == nullptr)
            exception = std::current_exception();
        }
      }
    }
    if (exception != nullptr)
      std::rethrow_exception(exception);
    return VECOPS_STATUS_OK;
  } catch (const std::exception& exception) {
    set_error(error, exception.what());
  } catch (...) {
    set_error(error, "OpenMP thread pool failed");
  }
  return VECOPS_STATUS_EXECUTION_ERROR;
}

const VecopsThreadPoolV1 provider{
  sizeof(VecopsThreadPoolV1), VECOPS_THREAD_POOL_ABI_MAJOR, VECOPS_THREAD_POOL_ABI_MINOR,
  UINT64_C(0x6f70656e6d70), nullptr, maximum, in_parallel, parallel_for, nullptr, nullptr, 0};

} // namespace

extern "C" VECOPS_RUNTIME_EXPORT const VecopsThreadPoolV1* vecops_openmp_thread_pool_v1() {
  return &provider;
}
