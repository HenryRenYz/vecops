#include "TransposeBenchCommon.h"

namespace vecops::bench::transpose {

void register_compare_fp64_benchmarks() {
  register_compare_same_dtype<vecops::float64_t>();
}

} // namespace vecops::bench::transpose
