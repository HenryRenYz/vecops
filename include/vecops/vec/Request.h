#ifndef VECOPS_VEC_REQUEST_H
#define VECOPS_VEC_REQUEST_H

#include "vecops/CoreTypes.h"
#include "vecops/vec/Options.h"
#include "vecops/vec/Tag.h"
#include "vecops/vec/VecBase.h"

/**
 * @file Request.h
 * @brief Resolved, parse-free access descriptors for memory operations.
 *
 * A Request is the flattened form of an option pack: the parsing layers fold
 * `Options...` exactly once into a `LoadRequest` / `StoreRequest`, and every
 * layer below the entry point receives the struct instead of re-walking the
 * pack. Kinds that select code paths (mask form, addressing form, inactive
 * population) and index scale are template parameters so backends can branch
 * with `if constexpr`; values that vary at runtime (stride, first-count,
 * mask and index vectors, merge values) are fields, so numeric combinations
 * no longer multiply template instantiations.
 *
 * Only runtime fields selected by those kinds are present in the descriptor.
 * Vector values (mask, merge vector, indices) are referenced, never stored:
 * scalable-SVE vectors are sizeless and cannot be data members, so requests
 * borrow caller-owned storage exactly like opt::masked does. Null vector
 * pointers simply mean the matching kind is not selected.
 */

namespace vecops::vec {

/** Inactive-lane addressing policy of a memory or elementwise operation. */
enum class Active {
  Unmasked,  ///< Every logical lane participates.
  First,     ///< Lanes `[0, first_count)` participate.
  Masked,    ///< Lanes selected by the mask vector participate.
};

/** Address arithmetic applied to the base pointer. */
enum class Addressing {
  Contiguous,  ///< Consecutive elements.
  Strided,     ///< Element offsets `lane * stride`.
  Indexed,     ///< Per-lane element offsets from an index vector.
};

/** Inactive-lane result population for masked loads. */
enum class Populate {
  Zero,          ///< Inactive lanes read as zero.
  MergeVector,   ///< Inactive lanes read the merge vector.
  MergeScalar,   ///< Inactive lanes read the merge scalar.
};

/** Inactive-lane population policy for elementwise operations. */
enum class Inactive {
  PreserveInput,  ///< Inactive lanes keep the first operand's value.
  Zero,           ///< Inactive lanes read as zero.
  MergeVector,    ///< Inactive lanes read the merge vector.
  MergeScalar,    ///< Inactive lanes read the merge scalar.
  MergeMask,      ///< Mask-result operations merge the inactive mask bits.
};

namespace details::request_storage {

template <Active A, typename MaskVector>
struct ActiveFields {};

template <typename MaskVector>
struct ActiveFields<Active::First, MaskVector> {
  nint_t first_count = 0;
};

template <typename MaskVector>
struct ActiveFields<Active::Masked, MaskVector> {
  const MaskVector* mask = nullptr;
};

template <Addressing Addr, typename IndexVector>
struct AddressingFields {};

template <typename IndexVector>
struct AddressingFields<Addressing::Strided, IndexVector> {
  nint_t stride = 1;
};

template <typename IndexVector>
struct AddressingFields<Addressing::Indexed, IndexVector> {
  const IndexVector* indices = nullptr;
};

template <Populate P, VectorTag Tag>
struct PopulateFields {};

template <VectorTag Tag>
struct PopulateFields<Populate::MergeVector, Tag> {
  const Vec<Tag>* merge_vector = nullptr;
};

template <VectorTag Tag>
struct PopulateFields<Populate::MergeScalar, Tag> {
  ElementOf<Tag> merge_scalar = ElementOf<Tag>{};
};

template <Inactive I, VectorTag Tag>
struct InactiveFields {};

template <VectorTag Tag>
struct InactiveFields<Inactive::MergeVector, Tag> {
  const Vec<Tag>* merge_vector = nullptr;
};

template <VectorTag Tag>
struct InactiveFields<Inactive::MergeScalar, Tag> {
  ElementOf<Tag> merge_scalar = ElementOf<Tag>{};
};

template <VectorTag Tag>
struct InactiveFields<Inactive::MergeMask, Tag> {
  const Mask<Tag>* mask_merge = nullptr;
};

} // namespace details::request_storage

/**
 * @brief Resolved descriptor for `load`.
 *
 * The defaults reproduce the plain unmasked, unaligned, temporal,
 * contiguous load.
 */
template <
    VectorTag Tag,
    Active A = Active::Unmasked,
    Addressing Addr = Addressing::Contiguous,
    Populate P = Populate::Zero,
    typename Alignment = mem::Unaligned,
    typename Temporality = mem::Temporal,
    int IndexScale = 0,
    VectorValue IndexVector = Vec<IndexTag<Tag>>>
struct LoadRequest
    : details::request_storage::ActiveFields<A, Mask<Tag>>,
      details::request_storage::AddressingFields<Addr, IndexVector>,
      details::request_storage::PopulateFields<P, Tag> {
  using TagType = Tag;
  static constexpr Active active_kind = A;
  static constexpr Addressing addressing_kind = Addr;
  static constexpr Populate populate_kind = P;
  using AlignmentOption = Alignment;
  using TemporalityOption = Temporality;
  static constexpr int index_scale = IndexScale;
  using IndexVectorType = IndexVector;

};

/**
 * @brief Resolved descriptor for `store`.
 *
 * Stores have no inactive output lanes, so there is no populate axis.
 */
template <
    VectorTag Tag,
    Active A = Active::Unmasked,
    Addressing Addr = Addressing::Contiguous,
    typename Alignment = mem::Unaligned,
    typename Temporality = mem::Temporal,
    int IndexScale = 0,
    VectorValue IndexVector = Vec<IndexTag<Tag>>>
struct StoreRequest
    : details::request_storage::ActiveFields<A, Mask<Tag>>,
      details::request_storage::AddressingFields<Addr, IndexVector> {
  using TagType = Tag;
  static constexpr Active active_kind = A;
  static constexpr Addressing addressing_kind = Addr;
  using AlignmentOption = Alignment;
  using TemporalityOption = Temporality;
  static constexpr int index_scale = IndexScale;
  using IndexVectorType = IndexVector;

};

/**
 * @brief Resolved descriptor for elementwise operations.
 *
 * Covers arithmetic, bit, and comparison option surfaces: an active kind
 * (masked/unmasked — elementwise calls have no first-count form) and an
 * inactive-lane policy. Vector values are referenced, never stored, exactly
 * like the memory requests.
 */
template <
    VectorTag Tag,
    Active A = Active::Unmasked,
    Inactive I = Inactive::PreserveInput>
struct OpRequest
    : details::request_storage::ActiveFields<A, Mask<Tag>>,
      details::request_storage::InactiveFields<I, Tag> {
  using TagType = Tag;
  static constexpr Active active_kind = A;
  static constexpr Inactive inactive_kind = I;

};

/**
 * @brief Resolved descriptor for `load_convert`.
 *
 * Conversion layout and value policies are type axes exactly like alignment
 * and temporality. The mask domain follows the layout: ordered conversion
 * masks are in the output domain, unordered conversion masks are in the
 * memory-side domain. Unordered conversion accepts no populate option.
 */
template <
    VectorTag ToTag,
    Element From,
    Active A = Active::Unmasked,
    Addressing Addr = Addressing::Contiguous,
    Populate P = Populate::Zero,
    typename Alignment = mem::Unaligned,
    typename Temporality = mem::Temporal,
    int IndexScale = 0,
    VectorValue IndexVector = Vec<IndexTag<ToTag>>,
    typename Layout = cvt::Ordered,
    typename ValuePolicy = cvt::Saturate,
    MaskValue MaskVector = Mask<ToTag>>
struct LoadConvertRequest
    : details::request_storage::ActiveFields<A, MaskVector>,
      details::request_storage::AddressingFields<Addr, IndexVector>,
      details::request_storage::PopulateFields<P, ToTag> {
  using TagType = ToTag;
  using FromElement = From;
  static constexpr Active active_kind = A;
  static constexpr Addressing addressing_kind = Addr;
  static constexpr Populate populate_kind = P;
  using AlignmentOption = Alignment;
  using TemporalityOption = Temporality;
  static constexpr int index_scale = IndexScale;
  using IndexVectorType = IndexVector;
  using LayoutOption = Layout;
  using ValuePolicyOption = ValuePolicy;
  /// Natural mask domain for the layout: memory-side for unordered
  /// conversion, output-side otherwise.
  using MaskTag = std::conditional_t<
      std::same_as<Layout, cvt::Unordered>,
      Rebind<From, ToTag>,
      ToTag>;

};

/**
 * @brief Resolved descriptor for `store_convert`.
 *
 * The mask domain depends on the conversion layout: unordered stores mask in
 * the memory-side domain, ordered stores accept either domain and convert
 * when needed, so the mask vector type stays a template parameter.
 */
template <
    VectorTag FromTag,
    Element To,
    Active A = Active::Unmasked,
    Addressing Addr = Addressing::Contiguous,
    typename Alignment = mem::Unaligned,
    typename Temporality = mem::Temporal,
    int IndexScale = 0,
    VectorValue IndexVector = Vec<IndexTag<FromTag>>,
    typename Layout = cvt::Ordered,
    typename ValuePolicy = cvt::Saturate,
    typename Packing = mem::Packed,
    MaskValue MaskVector = Mask<Rebind<To, FromTag>>>
struct StoreConvertRequest
    : details::request_storage::ActiveFields<A, MaskVector>,
      details::request_storage::AddressingFields<Addr, IndexVector> {
  using TagType = FromTag;
  using ToElement = To;
  static constexpr Active active_kind = A;
  static constexpr Addressing addressing_kind = Addr;
  using AlignmentOption = Alignment;
  using TemporalityOption = Temporality;
  static constexpr int index_scale = IndexScale;
  using IndexVectorType = IndexVector;
  using LayoutOption = Layout;
  using ValuePolicyOption = ValuePolicy;
  using PackingOption = Packing;
  using MaskVectorType = MaskVector;

};

/**
 * @brief Resolved descriptor for reductions.
 *
 * Reductions filter participating lanes with a single active kind; there
 * are no population options because a reduction has no inactive output
 * lanes.
 */
template <VectorTag Tag, Active A = Active::Unmasked>
struct ReduceRequest
    : details::request_storage::ActiveFields<A, Mask<Tag>> {
  using TagType = Tag;
  static constexpr Active active_kind = A;
};

} // namespace vecops::vec

#endif // VECOPS_VEC_REQUEST_H
