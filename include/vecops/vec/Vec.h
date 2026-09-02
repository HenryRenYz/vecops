#ifndef VECOPS_VEC_VEC_H
#define VECOPS_VEC_VEC_H

/**
 * @file Vec.h
 * @brief Umbrella header for the VecOps vector abstraction layer.
 *
 * The vec module provides a portable, architecture-independent interface for
 * SIMD vector operations. The key abstraction layers are:
 *
 * 1. **Tags** (Tag.h): Compile-time descriptors like FixedTag<T,N> and
 *    ScalableTag<T,P> that describe a vector's element type and logical
 *    lane count without committing to a specific ISA representation.
 *
 * 2. **Representations** (VecBase.h): Vec<Tag> and Mask<Tag> are the
 *    concrete types selected by the active backend (scalar, x86, or SVE)
 *    from a Tag.
 *
 * 3. **Operations** (Arithmetic.h, Rounding.h, Sequence.h, Basic.h, Bit.h,
 *    Comparison.h, Conversion.h, WideningDot.h, Math.h, Memory.h,
 *    Reduction.h): Each operation is a callable CPO object (e.g., add, mul,
 *    load) dispatched
 *    through backend-specialized NativeImpl / NativeWordImpl with automatic
 *    multi-word batching.
 *
 * 4. **Options** (Options.h): Tag types in namespaces opt, cvt, and mem
 *    control masking, inactive-lane population, memory alignment/
 *    temporality, and addressing mode for each operation.
 */

#include "vecops/vec/VecBase.h"
#include "vecops/vec/Basic.h"
#include "vecops/vec/Arithmetic.h"
#include "vecops/vec/Sequence.h"
#include "vecops/vec/Rounding.h"
#include "vecops/vec/Bit.h"
#include "vecops/vec/Comparison.h"
#include "vecops/vec/Conversion.h"
#include "vecops/vec/WideningDot.h"
#include "vecops/vec/ConversionMemory.h"
#include "vecops/vec/Math.h"
#include "vecops/vec/Memory.h"
#include "vecops/vec/Prefetch.h"
#include "vecops/vec/Reduction.h"

#endif // VECOPS_VEC_VEC_H
