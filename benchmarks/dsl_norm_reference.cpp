// Benchmark-only wrapper: LayerNorm's prepared-pattern overloads currently
// require TaskContext to be visible before its public header is parsed.
// Keep the original C++ algorithms unchanged for the comparison.
#include "vecops/execution/Parallel.h"
#include "../tests/runtime/NormOperatorKernel.cpp"
