#include "TransposeBenchCommon.h"

namespace vecops::bench::transpose {

void register_fp64_benchmarks() {
  register_same_dtype<vecops::float64_t>();
}

} // namespace vecops::bench::transpose
