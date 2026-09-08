//
// Copyright (c) vecops contributors.
//

/** @file Parallel.cpp @brief OpenMP-backed worker-team execution. */

#include "vecops/execution/Parallel.h"

#include <algorithm>
#include <exception>

#if defined(VECOPS_ENABLE_OPENMP)
#  include <omp.h>
#endif

namespace vecops::execution {

nint_t max_parallelism() noexcept {
#if defined(VECOPS_ENABLE_OPENMP)
  if (omp_in_parallel())
    return 1;
  return std::max<nint_t>(1, omp_get_max_threads());
#else
  return 1;
#endif
}

namespace details {

void parallel_for_erased(nint_t requested_threads, void* object, ParallelBody body) {
  if (body == nullptr)
    return;

#if defined(VECOPS_ENABLE_OPENMP)
  const nint_t maximum = std::max<nint_t>(1, omp_get_max_threads());
  const nint_t requested = requested_threads <= 0 ? maximum : std::min(requested_threads, maximum);

  if (requested <= 1 || omp_in_parallel()) {
    body(object, ParallelContext{0, 1});
    return;
  }

  std::exception_ptr first_exception;
#  pragma omp parallel num_threads(requested) shared(first_exception)
  {
    try {
      body(object,
           ParallelContext{static_cast<nint_t>(omp_get_thread_num()), static_cast<nint_t>(omp_get_num_threads())});
    } catch (...) {
#  pragma omp critical(vecops_parallel_for_exception)
      {
        if (first_exception == nullptr)
          first_exception = std::current_exception();
      }
    }
  }

  if (first_exception != nullptr)
    std::rethrow_exception(first_exception);
#else
  (void)requested_threads;
  body(object, ParallelContext{0, 1});
#endif
}

} // namespace details
} // namespace vecops::execution
