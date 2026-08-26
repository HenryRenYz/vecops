#include "TransposeBenchCommon.h"

namespace vecops::bench::transpose {

void register_compare_fp32_benchmarks() {
  register_compare_same_dtype<vecops::float32_t>();
}

} // namespace vecops::bench::transpose
