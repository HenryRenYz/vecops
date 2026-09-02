#ifndef VECOPS_VEC_DETAILS_SEQUENCE_H
#define VECOPS_VEC_DETAILS_SEQUENCE_H

/**
 * @file Sequence.h
 * @brief Multi-word batching for index-sensitive lane sequences.
 */

#include "vecops/vec/details/Wordwise.h"

namespace vecops::vec::details {

template <typename Backend, VectorTag Tag>
  requires (RepresentationTraits<Backend, Tag>::word_count > 1)
struct GenericImpl<Backend, IotaOp, Tag> {
  static VECOPS_ALWAYS_INLINE Vec<Tag> call(
      IotaOp op, Tag tag, ElementOf<Tag> start) {
    return construct_words<Backend>(
        tag, [&]<nint_t Index>(Tag parent) VECOPS_INLINE_LAMBDA {
      return execute_word<Index, Backend>(op, parent, start);
    });
  }

  static VECOPS_ALWAYS_INLINE Vec<Tag> call(
      IotaOp op, Tag tag, ElementOf<Tag> start, ElementOf<Tag> step) {
    return construct_words<Backend>(
        tag, [&]<nint_t Index>(Tag parent) VECOPS_INLINE_LAMBDA {
      return execute_word<Index, Backend>(op, parent, start, step);
    });
  }
};

} // namespace vecops::vec::details

#endif // VECOPS_VEC_DETAILS_SEQUENCE_H
