#include "TransposeBenchCommon.h"

namespace vecops::bench::transpose {

void register_int8_benchmarks() {
  register_same_dtype<int8_t>();
}

} // namespace vecops::bench::transpose
