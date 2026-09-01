//
// Copyright (c) vecops contributors.
//

#ifndef VECOPS_PLATFORM_CACHE_INFO_H
#define VECOPS_PLATFORM_CACHE_INFO_H

#include "vecops/CoreTypes.h"

namespace vecops::platform {

struct CacheInfo {
  nint_t line_bytes = 64;
  nint_t l1d_bytes = 32 * 1024;
  nint_t l2_bytes = 1024 * 1024;
  nint_t l3_bytes = 8 * 1024 * 1024;
  bool fallback = true;
};

/** Detect the cache hierarchy of the current CPU. */
CacheInfo detect_cache_info();

/** Process-wide immutable cache information, initialized on first use. */
const CacheInfo& cache_info();

struct SystemCacheInfoProvider {
  const CacheInfo& operator()() const { return cache_info(); }
};

} // namespace vecops::platform

#endif // VECOPS_PLATFORM_CACHE_INFO_H
