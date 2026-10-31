// SPDX-FileCopyrightText: 2026 HenryRenYz and vecops contributors
// SPDX-License-Identifier: MIT
/** Baseline-safe, cold ISA selection. All alternatives implement the same vec
 * operations; there is no libc transport or user-selectable copy backend. */
#include "vecops/comm/Copy.h"
#if defined(__aarch64__)
#include <sys/auxv.h>
#include <asm/hwcap.h>
#endif
namespace vecops::comm::detail {
namespace portable { const CopyBackend& backend() noexcept; }
#if defined(VECOPS_COMM_COPY_AVX512)
namespace avx512 { const CopyBackend& backend() noexcept; }
#endif
#if defined(VECOPS_COMM_COPY_AVX2)
namespace avx2 { const CopyBackend& backend() noexcept; }
#endif
#if defined(VECOPS_COMM_COPY_SVE)
namespace sve { const CopyBackend& backend() noexcept; }
#endif
const CopyBackend& copy_backend() noexcept {
  static const CopyBackend* selected=[] {
#if defined(VECOPS_COMM_COPY_AVX512)
    if(__builtin_cpu_supports("avx512f") && __builtin_cpu_supports("avx512bw") &&
       __builtin_cpu_supports("avx512vl") && __builtin_cpu_supports("avx512dq") &&
       __builtin_cpu_supports("avx512cd"))return &avx512::backend();
#endif
#if defined(VECOPS_COMM_COPY_AVX2)
    if(__builtin_cpu_supports("avx2"))return &avx2::backend();
#endif
#if defined(VECOPS_COMM_COPY_SVE)
    if(getauxval(AT_HWCAP)&HWCAP_SVE)return &sve::backend();
#endif
    return &portable::backend();
  }();
  return *selected;
}
}
