#ifndef VECOPS_VEC_SVE_H
#define VECOPS_VEC_SVE_H

#include "vecops/vec/Vec.h"

/**
 * @file SVE.h
 * @brief Public ordinary-SVE, 128-bit-segment matrix/vector instructions.
 *
 * These are register primitives, not a matrix operator or a state owner.
 * All tags are ScalableTag<T, 0>: one physical, ordinary-SVE register. Fixed
 * tags, subwords and multiword tags are deliberately not accepted. They work
 * with both sizeless and vector-length-specific C++ representations.
 *
 * Execute outside streaming mode. Callers must keep vectors inside that
 * execution domain and validate ordinary VL independently of SME SVL.
 * Including the header is portable; the names are available only on SVE.
 * Feature-specific mmla overloads additionally require SVE BF16 or I8MM.
 *
 * join_k_groups interleaves the low halves of two registers as 32-bit groups.
 * For MOPA-packed input, each group contains KPack=2 BF16 or KPack=4 byte
 * elements for one spatial row. Joining adjacent K groups produces the
 * MMLA 128-bit segment layout (two rows by two K groups).
 *
 * broadcast_segment<Index> duplicates a 128-bit segment. The caller must
 * prove Index < ordinary_VL_bytes/16; an insufficient VL is not repaired by
 * this instruction. row_pair<Row> extracts row 0 or 1 of each MMLA 2x2
 * accumulator segment, yielding contiguous column pairs in the low half.
 *
 * mmla computes one matrix-product accumulate per 128-bit segment: BF16
 * inputs widen to FP32; signed/signed and unsigned/unsigned bytes accumulate
 * modulo 2^32 and return an int32 bit view. Mixed signedness is intentionally
 * unsupported. The accumulator starts/continues exactly as supplied; MMLA
 * floating-point accumulation follows the instruction's semantics and need
 * not match a reassociated scalar or vector-dot reduction bit for bit.
 */
#if defined(HAS_SVE)
#include "vecops/vec/details/sve/SVE.h"
namespace vecops::vec::sve {
using details::sve::join_k_groups;
using details::sve::broadcast_segment;
using details::sve::row_pair;
#if defined(__ARM_FEATURE_SVE_BF16) || defined(__ARM_FEATURE_SVE_MATMUL_INT8)
using details::sve::mmla;
#endif
} // namespace vecops::vec::sve
#endif

#endif // VECOPS_VEC_SVE_H
