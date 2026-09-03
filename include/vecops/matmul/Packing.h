//
// Copyright (c) vecops contributors.
//

#ifndef VECOPS_MATMUL_PACKING_H
#define VECOPS_MATMUL_PACKING_H

#include "vecops/matmul/Atom.h"
#include "vecops/tensor/Layout.h"

// Arch-conditional concrete formats: each backend's Packing<Element, Side>
// implementation lives behind these guards. The functions below stay
// backend-independent and dispatch through the Atom's Packing member.
#if defined(ARCH_X86_FAMILY)
#include "vecops/matmul/details/packing/amx/Format.h"
#endif

#if defined(HAS_SME)
#include "vecops/matmul/details/packing/sme/Format.h"
#endif

/**
 * @file vecops/matmul/Packing.h
 * @brief Backend-independent queries over an atom's packed input formats.
 *
 * Hardware matrix instructions consume operands in fixed block layouts that
 * differ per atom and per operand (AMX keeps A rows contiguous but
 * interleaves B's K into 32-bit VNNI groups; SME tiles both operands into
 * panel x KPack blocks). An atom exposes its format as
 * `Atom::Packing<Operand::A/B>`; this header lifts that contract into the
 * shared vocabulary used by the packing planners and every backend:
 *
 * - `packing_t` — name the format type for one atom and operand side;
 * - `packed_layout` — compute the layout of the packed copy of an input;
 * - `is_packed_layout` — statically recognize a layout shaped by a format;
 * - `is_corresponding_packed_layout` — verify that one layout is exactly
 *   the packed image of another.
 *
 * ## Pitfalls
 *
 * - "Packed" is defined per atom **and** operand side; a layout packed for
 *   one atom (or for A) is generally not packed for another.
 * - `packed_layout` accepts rank-two inputs and its result may have more
 *   dimensions than the input, with tail blocks rounded up — the packed
 *   copy can be larger than the source.
 * - `is_corresponding_packed_layout` demands strict per-dimension equality
 *   of shapes and strides; it is a recognition check, not a compatibility
 *   check.
 */
namespace vecops::matmul {

/// Alias naming the packed-input format an atom defines for one operand
/// side (`AtomT::Packing<Side>`); the format type provides the block
/// geometry constants and the static layout rules.
template <Atom AtomT, Operand Side>
using packing_t = typename AtomT::template Packing<Side>;

/**
 * @brief Compute the layout of the packed copy of a rank-two input.
 *
 * The result's rank, dimension meanings, and block geometry are defined by
 * the atom's format (see the backend `Format.h` headers); the packed output
 * may be larger than the input because tail blocks are rounded up.
 *
 * @tparam AtomT        The atom whose packed format applies.
 * @tparam Side         Whether the input is operand A or B.
 * @tparam InputLayout  A `tensor::LayoutLike` of rank two (spatial x K).
 * @param input         Layout of the unpacked source operand.
 * @return              The layout to allocate for the packed copy.
 */
template <Atom AtomT, Operand Side, tensor::LayoutLike InputLayout>
VECOPS_INLINE auto packed_layout(const InputLayout& input) {
  return packing_t<AtomT, Side>::packed_layout(input);
}

/**
 * @brief Statically recognize a layout shaped by an atom's packed format.
 *
 * This is a compile-time shape check: the layout's inner dimension sizes
 * and strides must carry the format's exact compile-time constants, while
 * outer block counts may remain dynamic. Runtime values are not inspected.
 *
 * @tparam AtomT   The atom whose packed format applies.
 * @tparam Side    Whether the layout holds operand A or B.
 * @tparam Layout  The candidate `tensor::LayoutLike`.
 * @return         `true` if `Layout` has the format's packed shape.
 */
template <Atom AtomT, Operand Side, tensor::LayoutLike Layout>
consteval bool is_packed_layout() {
  return packing_t<AtomT, Side>::template is_packed_layout<Layout>();
}

/**
 * @brief Check that `output` is exactly the packed image of `input`.
 *
 * Defines the strict "correspondence" used when a backend must confirm
 * that a prepacked operand matches the input it stands for: the output
 * must be a packed layout for this atom and side, and every dimension's
 * shape and stride must equal what `packed_layout(input)` would produce.
 *
 * @tparam AtomT         The atom whose packed format applies.
 * @tparam Side          Whether the operands are A or B.
 * @tparam InputLayout   Layout of the unpacked source operand.
 * @tparam OutputLayout  Candidate packed layout.
 * @param input          Layout of the unpacked source.
 * @param output         The layout being recognized.
 * @return               `true` if `output` is `input`'s packed image.
 */
template <Atom AtomT, Operand Side,
          tensor::LayoutLike InputLayout,
          tensor::LayoutLike OutputLayout>
VECOPS_INLINE bool is_corresponding_packed_layout(
    const InputLayout& input, const OutputLayout& output) {
  // Two-stage check: first reject layouts whose compile-time shape is not
  // a packed layout for this atom/side at all, then compare against the
  // layout this exact input would pack into.
  if constexpr (!is_packed_layout<AtomT, Side, OutputLayout>()) {
    return false;
  } else {
    const auto expected = packed_layout<AtomT, Side>(input);
    // Defensive: the is_packed_layout branch above already fixed the rank,
    // so this can only fire if the format's packed and checked ranks
    // disagree (a format bug, not a user error).
    static_assert(decltype(expected)::Ndim == OutputLayout::Ndim);
    // Correspondence is strict: every dimension must match in both shape
    // and stride — equivalent block counts are not enough.
    for (int d = 0; d < OutputLayout::Ndim; ++d) {
      if (expected.shape()[d] != output.shape()[d] ||
          expected.strides()[d] != output.strides()[d]) {
        return false;
      }
    }
    return true;
  }
}

} // namespace vecops::matmul

#endif // VECOPS_MATMUL_PACKING_H
