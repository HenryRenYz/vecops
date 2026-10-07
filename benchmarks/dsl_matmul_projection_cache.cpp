// The target process's exact public cache data, outside all timed kernels.
#include "vecops/Kernel.h"
#include "vecops/platform/CacheInfo.h"

inline void __kernel__(auto Information) {
  const auto& cache = vecops::platform::cache_info();
  auto* output = Information.data();
  output[0] = cache.line_bytes;
  output[1] = cache.l1d_bytes;
  output[2] = cache.l2_bytes;
  output[3] = cache.l3_bytes;
  output[4] = cache.llc_bytes;
  output[5] = cache.llc_level;
  output[6] = cache.l3_present;
  output[7] = cache.fallback;
}
