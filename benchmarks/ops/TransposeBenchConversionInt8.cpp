#include "TransposeBenchCommon.h"

namespace vecops::bench::transpose {

void register_int8_conversion_benchmarks() {
  register_conversion_cases(ConversionCases{}, Int8ConversionPairs{});
}

} // namespace vecops::bench::transpose
