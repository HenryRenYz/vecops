#include "TransposeBenchCommon.h"

namespace vecops::bench::transpose {

void register_compare_fp16_benchmarks() {
  register_compare_same_dtype<vecops::float16_t>();
}

} // namespace vecops::bench::transpose
