#include "TransposeBenchCommon.h"

namespace vecops::bench::transpose {

void register_conversion_benchmarks() {
  register_int8_conversion_benchmarks();
  register_fp16_conversion_benchmarks();
  register_fp32_conversion_benchmarks();
  register_fp64_conversion_benchmarks();
}

} // namespace vecops::bench::transpose
