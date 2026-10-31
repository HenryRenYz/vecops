// SPDX-FileCopyrightText: 2026 HenryRenYz and vecops contributors
// SPDX-License-Identifier: MIT
#pragma once

#include <cstddef>
#include <cstdint>
#include <string>

namespace memory_bench {
enum class Operation { Read, Write, Copy, Triad };
struct Kernel {
  using Function = void (*)(const float*, const float*, float*, std::size_t);
  Function function = nullptr;
  std::size_t step_bytes = 0;
  std::size_t vector_bytes = 0;
  std::string load_policy;
  std::string store_policy;
  std::string instructions;
  std::string unsupported;
};
Kernel scalar_kernel(Operation op, unsigned bytes, bool nt_load, bool nt_store, bool fma);
Kernel vector_kernel(Operation op, unsigned power, bool nt_load, bool nt_store, bool fma);
std::size_t native_vector_bytes();
const char* vector_backend();
void completion_fence();
void* chase(void* pointer, std::size_t loads);
inline void compiler_barrier() { asm volatile("" ::: "memory"); }
inline void relax() {
#if defined(__x86_64__)
  asm volatile("pause");
#else
  asm volatile("yield");
#endif
}
} // namespace memory_bench
