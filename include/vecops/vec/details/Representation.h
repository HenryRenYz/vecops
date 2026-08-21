#ifndef VECOPS_VEC_DETAILS_REPRESENTATION_H
#define VECOPS_VEC_DETAILS_REPRESENTATION_H

#include <concepts>
#include <type_traits>

#include "vecops/vec/Tag.h"
#include "vecops/vec/details/Backend.h"

namespace vecops::vec::details {

/**
 * Maps a complete Tag to backend storage and physical-layout information.
 * Backend headers specialize this template without an inheritance hierarchy.
 */
template <typename Backend, VectorTag Tag>
struct RepresentationTraits;

template <typename T>
struct IsVectorRepresentation : std::false_type {};

template <typename T>
struct IsMaskRepresentation : std::false_type {};

/**
 * Lossy reverse mapping from a Vec representation to a canonical physical Tag.
 *
 * Subword information is intentionally absent from the representation and
 * cannot be recovered here. Operations whose semantics depend on logical lane
 * count must receive the original Tag instead of using this trait.
 * No corresponding Mask-to-Tag trait exists: mask representation never
 * carries enough element-type information to reconstruct a Tag.
 */
template <typename V>
struct InferredTagTraits;

template <typename V>
concept HasInferredTag = requires {
  typename InferredTagTraits<std::remove_cvref_t<V>>::type;
};

} // namespace vecops::vec::details

namespace vecops::vec {

template <typename V>
concept VectorValue = details::IsVectorRepresentation<
    std::remove_cvref_t<V>>::value;

template <typename M>
concept MaskValue = details::IsMaskRepresentation<
    std::remove_cvref_t<M>>::value;

template <VectorValue V>
  requires details::HasInferredTag<V>
using InferredTagOf = typename details::InferredTagTraits<
    std::remove_cvref_t<V>>::type;

} // namespace vecops::vec

#endif // VECOPS_VEC_DETAILS_REPRESENTATION_H
