//
// Created by renyz on 2026/3/13.
//

#ifndef VECOPS_CORETYPES_H
#define VECOPS_CORETYPES_H

#include <cstdint>  // for standard int defs

#include "vecops/Features.h"
#include "vecops/CoreDefs.h"
#include "vecops/util/BFloat16.h"
#include "vecops/util/Float16.h"

namespace vecops {

/**
 * Float types
 */
using bfloat16_t = BFloat16;
using float16_t = Float16;
using float32_t = float;
using float64_t = double;
/**
 * Signed native int, having the same width as machine word.
 */
using nint_t = ptrdiff_t;
/**
 * Unsigned native int, having the same width as machine word.
 */
using nuint_t = size_t;

/**
 * Integer types
 */
using std::int8_t;
using std::uint8_t;
using std::int16_t;
using std::uint16_t;
using std::int32_t;
using std::uint32_t;
using std::int64_t;
using std::uint64_t;

} // namespace vecops

#endif //VECOPS_CORETYPES_H
