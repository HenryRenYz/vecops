#include "TransposeBenchCommon.h"

namespace vecops::bench::transpose {

void register_fp16_conversion_benchmarks() {
  register_conversion_cases(ConversionCases{}, Fp16ConversionPairs{});
}

} // namespace vecops::bench::transpose
