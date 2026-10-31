// SPDX-FileCopyrightText: 2026 HenryRenYz and vecops contributors
// SPDX-License-Identifier: MIT
#include "Kernels.h"

namespace memory_bench {
void completion_fence() {
#if defined(__x86_64__)
  asm volatile("mfence\n\tlfence" ::: "memory");
#else
  asm volatile("dsb ish\n\tisb" ::: "memory");
#endif
}
__attribute__((noinline)) void* chase(void* pointer, std::size_t loads) {
  for (std::size_t i = 0; i < loads; ++i) {
#if defined(__x86_64__)
    asm volatile("mov (%0), %0" : "+r"(pointer) : : "memory");
#else
    asm volatile("ldr %0, [%0]" : "+r"(pointer) : : "memory");
#endif
  }
  return pointer;
}
} // namespace memory_bench
