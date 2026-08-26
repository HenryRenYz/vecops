#include "TransposeBenchCommon.h"

namespace vecops::bench::transpose {

void register_bf16_benchmarks() {
#if defined(HAS_BFLOAT16) || defined(ARCH_X86_FAMILY)
  register_same_dtype<vecops::bfloat16_t>();
#endif
}

} // namespace vecops::bench::transpose
