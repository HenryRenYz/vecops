#ifndef VECOPS_VEC_DETAILS_SCALAR_MEMORY_H
#define VECOPS_VEC_DETAILS_SCALAR_MEMORY_H

#include <algorithm>

#include "vecops/vec/details/Dispatch.h"
#include "vecops/vec/details/Options.h"

namespace vecops::vec::details {


/* **************************************************************************** */
//                    Load and store word implementations                     //
/* **************************************************************************** */

template <nint_t Index, VectorTag Tag>
VECOPS_ALWAYS_INLINE constexpr nint_t scalar_valid_memory_lanes(Tag tag) {
  const nint_t word_lanes = native_word_size(tag);
  return std::clamp<nint_t>(
      size(tag) - Index * word_lanes, 0, word_lanes);
}

template <>
struct NativeWordImpl<ScalarBackend, LoadOp> {
  template <nint_t Index, VectorTag Tag,
            typename Alignment, typename Temporality>
    requires ((IsAlignedOption<Alignment>::value ||
               IsUnalignedOption<Alignment>::value) &&
              (IsTemporalOption<Temporality>::value ||
               IsNonTemporalOption<Temporality>::value))
  static VECOPS_ALWAYS_INLINE NativeWordVec<Tag> call(
      LoadOp, Tag tag, const ElementOf<Tag>* pointer,
      Alignment, Temporality) {
    NativeWordVec<Tag> result{};
    const nint_t valid = scalar_valid_memory_lanes<Index>(tag);
    for (nint_t lane = 0; lane < valid; ++lane) result[lane] = pointer[lane];
    return result;
  }

  template <nint_t Index, VectorTag Tag,
            typename Alignment, typename Temporality>
    requires ((IsAlignedOption<Alignment>::value ||
               IsUnalignedOption<Alignment>::value) &&
              (IsTemporalOption<Temporality>::value ||
               IsNonTemporalOption<Temporality>::value))
  static VECOPS_ALWAYS_INLINE NativeWordVec<Tag> call(
      LoadOp, Tag tag, const ElementOf<Tag>* pointer,
      NativeWordMask<Tag> mask, NativeWordVec<Tag> inactive,
      Alignment, Temporality) {
    const nint_t valid = scalar_valid_memory_lanes<Index>(tag);
    for (nint_t lane = 0; lane < valid; ++lane) {
      if (mask.bits.test(static_cast<std::size_t>(lane)))
        inactive[lane] = pointer[lane];
    }
    return inactive;
  }
};

template <>
struct NativeWordImpl<ScalarBackend, StoreOp> {
  template <nint_t Index, VectorTag Tag,
            typename Alignment, typename Temporality>
    requires ((IsAlignedOption<Alignment>::value ||
               IsUnalignedOption<Alignment>::value) &&
              (IsTemporalOption<Temporality>::value ||
               IsNonTemporalOption<Temporality>::value))
  static VECOPS_ALWAYS_INLINE void call(
      StoreOp, Tag tag, ElementOf<Tag>* pointer, NativeWordVec<Tag> value,
      Alignment, Temporality) {
    const nint_t valid = scalar_valid_memory_lanes<Index>(tag);
    for (nint_t lane = 0; lane < valid; ++lane) pointer[lane] = value[lane];
  }

  template <nint_t Index, VectorTag Tag,
            typename Alignment, typename Temporality>
    requires ((IsAlignedOption<Alignment>::value ||
               IsUnalignedOption<Alignment>::value) &&
              (IsTemporalOption<Temporality>::value ||
               IsNonTemporalOption<Temporality>::value))
  static VECOPS_ALWAYS_INLINE void call(
      StoreOp, Tag tag, ElementOf<Tag>* pointer,
      NativeWordMask<Tag> mask, NativeWordVec<Tag> value,
      Alignment, Temporality) {
    const nint_t valid = scalar_valid_memory_lanes<Index>(tag);
    for (nint_t lane = 0; lane < valid; ++lane) {
      if (mask.bits.test(static_cast<std::size_t>(lane))) pointer[lane] = value[lane];
    }
  }
};

} // namespace vecops::vec::details

#endif // VECOPS_VEC_DETAILS_SCALAR_MEMORY_H
