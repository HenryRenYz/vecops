#include "TransposeBenchCommon.h"

namespace vecops::bench::transpose {

void register_fp64_conversion_benchmarks() {
  register_conversion_cases(ConversionCases{}, Fp64ConversionPairs{});
}

} // namespace vecops::bench::transpose
