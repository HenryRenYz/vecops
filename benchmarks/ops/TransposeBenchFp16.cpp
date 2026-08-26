#include "TransposeBenchCommon.h"

namespace vecops::bench::transpose {

void register_fp16_benchmarks() {
  register_same_dtype<vecops::float16_t>();
}

} // namespace vecops::bench::transpose
