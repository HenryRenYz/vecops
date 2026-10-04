/** @file Atomic.h
 * @brief Scalar CPU atomics for explicitly shared storage.
 *
 * Supported types are 32/64-bit integers and float/double. Every concurrent
 * access to an atomic location must use these operations or be separated by
 * a team barrier. References must meet atomic_ref::required_alignment.
 * min/max use ordinary < comparisons: NaN candidates do not replace a value,
 * and a stored NaN remains NaN. compare_exchange returns the observed value.
 */
#ifndef VECOPS_EXECUTION_ATOMIC_H
#define VECOPS_EXECUTION_ATOMIC_H

#include <atomic>
#include <concepts>
#include <cstdint>
#include <stdexcept>
#include "vecops/CoreDefs.h"

namespace vecops::execution {
template <class T>
concept AtomicElement =
  ((std::integral<T> && !std::same_as<T, bool>) || std::same_as<T, float> || std::same_as<T, double>) &&
  (sizeof(T) == 4 || sizeof(T) == 8);

template <AtomicElement T>
[[nodiscard]] VECOPS_ALWAYS_INLINE std::atomic_ref<T> atomic_reference(T* pointer) {
  if (reinterpret_cast<std::uintptr_t>(pointer) % std::atomic_ref<T>::required_alignment != 0)
    throw std::invalid_argument("unaligned atomic reference");
  return std::atomic_ref<T>(*pointer);
}

template <AtomicElement T>
VECOPS_ALWAYS_INLINE T atomic_add(T* p, T value, std::memory_order order = std::memory_order_relaxed) {
  return atomic_reference(p).fetch_add(value, order);
}
template <AtomicElement T>
VECOPS_ALWAYS_INLINE T atomic_exchange(T* p, T value, std::memory_order order = std::memory_order_relaxed) {
  return atomic_reference(p).exchange(value, order);
}
template <AtomicElement T>
VECOPS_ALWAYS_INLINE T atomic_compare_exchange(T* p, T expected, T desired,
                                               std::memory_order order = std::memory_order_relaxed) {
  atomic_reference(p).compare_exchange_strong(expected, desired, order);
  return expected;
}
template <bool Minimum, AtomicElement T>
VECOPS_ALWAYS_INLINE T atomic_extreme(T* p, T value, std::memory_order order) {
  auto ref = atomic_reference(p);
  auto old = ref.load(order == std::memory_order_acq_rel ? std::memory_order_acquire : order);
  while ((Minimum ? value < old : old < value) && !ref.compare_exchange_weak(old, value, order)) {
  }
  return old;
}
template <AtomicElement T>
VECOPS_ALWAYS_INLINE T atomic_min(T* p, T value, std::memory_order order = std::memory_order_relaxed) {
  return atomic_extreme<true>(p, value, order);
}
template <AtomicElement T>
VECOPS_ALWAYS_INLINE T atomic_max(T* p, T value, std::memory_order order = std::memory_order_relaxed) {
  return atomic_extreme<false>(p, value, order);
}
} // namespace vecops::execution
#endif
