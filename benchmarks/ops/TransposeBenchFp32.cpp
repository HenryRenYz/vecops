#include "TransposeBenchCommon.h"

namespace vecops::bench::transpose {

void register_fp32_benchmarks() {
  register_same_dtype<vecops::float32_t>();
}

} // namespace vecops::bench::transpose
