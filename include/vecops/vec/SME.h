#ifndef VECOPS_VEC_SME_H
#define VECOPS_VEC_SME_H

#include "vecops/vec/Vec.h"

/**
 * @file SME.h
 * @brief Public, architecture-specific SME vector/ZA primitives.
 *
 * These are instructions, not state-owning operators. ZA moves require an
 * active execution::sme::Region; their vectors and predicates must be created
 * inside that region (SVL need not equal VL). No value may cross its boundary.
 * Sized SVE Vec types commit to the ordinary VL: only use them in SM when
 * fixed VL and fixed SVL are guaranteed equal; otherwise use VLA vector types.
 * The manual state owner forbids throwing, instrumentation, LTO and arbitrary
 * out-of-line calls while streaming is active. Its callback and every compute
 * helper must actually inline. With unequal or unbound vector domains the
 * owner makes one resolved private scalar-only call, entering SM before the
 * compute frame is allocated in SVL; ZA is confined to that compute body.
 * This compiler-specific bridge is not a general streaming function ABI.
 * Capture only scalar/pointer/metadata values, never pre-existing Vec/Mask.
 * Prefer ExecutionSession to raw ownership; entry and exit require SM=ZA=0.
 * For vector exp/log/rcp/rsqrt in a manual owner, define
 * VECOPS_SVE_INLINE_MATH_SPECIALS=1 before any vecops header, consistently
 * across translation units. It preserves exceptional-value semantics while
 * keeping those paths inline; it does not legalize scalar libm calls or
 * replace interprocedural state/frame and no-compute-call validation.
 * Availability is compile-time: including this header is portable, using its
 * primitives on a non-SME target is not. Tile indices are compile-time values.
 * load_hor/load_ver and store_hor/store_ver move contiguous unsigned-bit
 * memory slices directly to/from ZA (no Z temporary). Their predicates
 * select columns for horizontal slices and rows for vertical slices. Cast
 * same-width signed/floating pointers to unsigned containers for raw moves;
 * use the vector read/write forms when conversion or arithmetic is needed.
 */
#if defined(HAS_SME)
#include "vecops/vec/details/sme/SME.h"
namespace vecops::vec::sme {
using details::sme::streaming_vector_bytes;
using details::sme::streaming_vector_bytes_value;
using details::sme::streaming_lanes;
using details::sme::streaming_lanes_value;
using details::sme::read_svcr;
// Expert state owner; its callback must satisfy the contract above.
using details::sme::with_streaming_za;
// Force-inlined scalar helpers for the manual-owner no-out-of-line-call rule.
using details::sme::min_value;
using details::sme::max_value;
using details::sme::clamp_value;
using details::sme::zero_za;
using details::sme::write_hor;
using details::sme::write_ver;
using details::sme::read_hor;
using details::sme::read_ver;
using details::sme::load_hor;
using details::sme::load_ver;
using details::sme::store_hor;
using details::sme::store_ver;
using details::sme::addha;
using details::sme::mopa;
} // namespace vecops::vec::sme
#endif
#endif
