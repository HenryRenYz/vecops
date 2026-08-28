#ifndef VECOPS_VEC_OPTIONS_H
#define VECOPS_VEC_OPTIONS_H

#include <type_traits>

#include "vecops/Meta.h"
#include "vecops/vec/VecBase.h"

/**
 * @file Options.h
 * @brief Tag types that control operation behavior: masking, population,
 * conversion policy, and memory addressing.
 *
 * Options are organized in three namespaces:
 *
 * - **opt**: computation-level options for masking (masked/unmasked),
 *   inactive-lane population (zero/merge), math accuracy (opt::math), lane
 *   selection (first), lane ordering (LaneOrder), scale (Scale), and memory
 *   addressing (indexed, strided).
 *
 * - **cvt**: conversion-policy options controlling lane order
 *   (ordered/unordered/lane), value behavior (saturate/wrap), and narrowing
 *   phase.
 *
 * - **mem**: memory-access qualifiers for alignment (aligned/unaligned),
 *   temporality (temporal/non_temporal), narrow-store layout (packed/split),
 *   and prefetch hints.
 *
 * Options are validated at compile time by detection traits in
 * details/Options.h. Each operation header (Arithmetic.h, Bit.h, Memory.h,
 * etc.) declares its own valid-option set via constexpr predicates.
 */

namespace vecops::vec {

/**
 * Compile-time accuracy policy shared by vector math operations and operators
 * which expose their math-accuracy choice.
 *
 * The meaning of each level, including its error metric and accuracy bound, is
 * defined by the operation that accepts the policy.  The levels select an
 * implementation contract; they do not impose a cross-operation or
 * cross-target ordering on measured accuracy or performance.  In particular,
 * a more relaxed implementation may be as accurate as a stricter one and is
 * not guaranteed to be faster on every target or input.
 */
enum class Accuracy { Strict, Fast, Estimate };

} // namespace vecops::vec

namespace vecops::vec::opt {

/** Compile-time proof of the execution resources active at a memory access. */
template <typename Set>
struct Resources {
  using type = Set;
};

template <typename Set>
inline constexpr Resources<Set> resources{};

/** Compile-time options which select an operation-defined math accuracy. */
namespace math {

/** Associates a math option with one value of vec::Accuracy. */
template <Accuracy A>
struct AccuracyOption {
  static constexpr Accuracy value = A;
};

/** Generic accuracy option for forwarding a compile-time Accuracy value. */
template <Accuracy A>
inline constexpr AccuracyOption<A> accuracy{};

/** Requests the operation's strict accuracy contract. */
inline constexpr auto strict = accuracy<Accuracy::Strict>;

/** Requests the operation's fast accuracy contract, when distinguished. */
inline constexpr auto fast = accuracy<Accuracy::Fast>;

/** Requests the operation's estimate accuracy contract, when distinguished. */
inline constexpr auto estimate = accuracy<Accuracy::Estimate>;

} // namespace math

/**
 * Selects zero for every inactive output lane of a masked vector operation.
 * Mutually exclusive with any opt::merge option.
 */
struct Zero {};

inline constexpr Zero zero{};

/**
 * Explicitly selects every logical lane without applying a mask.
 * The counterpart of opt::masked; unmasked and masked in the same call is
 * a compile-time error.
 */
struct Unmasked {};

inline constexpr Unmasked unmasked{};

/**
 * Selects lanes in the logical element space of the receiving abstraction.
 * A higher-level accessor may translate the predicate before issuing the
 * eventual physical memory instruction.
 */
template <MaskValue M>
struct Masked {
  const M& value;
};

/**
 * Selects the lanes whose corresponding mask bit is true.
 *
 * The mask must be a named lvalue. Keeping only a reference is required for
 * sizeless SVE predicates, which cannot be stored as ordinary data members.
 */
template <MaskValue M>
VECOPS_ALWAYS_INLINE Masked<M> masked(const M& value) {
  return {value};
}

template <typename M>
  requires MaskValue<M> && (!std::is_lvalue_reference_v<M>)
Masked<std::remove_cvref_t<M>> masked(M&&) = delete;

/**
 * Selects the first count logical elements. count is clamped to [0, size(tag)]
 * at the call site: negative values act as 0, excess as size(tag).
 * @see mwhilelt
 */
struct First {
  nint_t count;
};

VECOPS_ALWAYS_INLINE constexpr First first(nint_t count) {
  return {count};
}

/** Supplies a vector value for inactive lanes. */
template <VectorValue V>
struct VectorMerge {
  const V& value;
};

/**
 * Preserves lanes from a named vector lvalue where an operation is inactive.
 * A reference wrapper also keeps this option usable with sizeless SVE values.
 */
template <VectorValue V>
VECOPS_ALWAYS_INLINE VectorMerge<V> merge(const V& value) {
  return {value};
}

template <typename V>
  requires VectorValue<V> && (!std::is_lvalue_reference_v<V>)
VectorMerge<std::remove_cvref_t<V>> merge(V&&) = delete;

/** Supplies a mask value for inactive lanes of a mask-producing operation. */
template <MaskValue M>
struct MaskMerge {
  const M& value;
};

template <MaskValue M>
VECOPS_ALWAYS_INLINE MaskMerge<M> merge(const M& value) {
  return {value};
}

template <typename M>
  requires MaskValue<M> && (!std::is_lvalue_reference_v<M>)
MaskMerge<std::remove_cvref_t<M>> merge(M&&) = delete;

/**
 * Supplies one scalar value broadcast to every inactive lane.
 * Valid only in operations that produce vector output (arithmetic, bit, fill,
 * load, etc.), not in mask-producing or reduction operations.
 */
template <Element T>
struct ScalarMerge {
  T value;
};

template <Element T>
VECOPS_ALWAYS_INLINE constexpr ScalarMerge<T> merge(T value) {
  return {value};
}

/**
 * Compile-time lane order for local_shuf operations.
 * All indices must be in [0, 16/sizeof(T)); the pattern repeats across every
 * 16-byte block of the input.
 * @see LocalShuffleOp
 */
template <int... Indices>
struct LaneOrder {};

template <int... Indices>
inline constexpr LaneOrder<Indices...> lanes{};

/** Compile-time byte scale for indexed memory addressing. */
template <int S>
struct Scale {
  static_assert(
      S == 1 || S == 2 || S == 4 || S == 8,
      "indexed scale must be 1, 2, 4, or 8");
  static constexpr int value = S;
};

template <int S>
inline constexpr Scale<S> scale{};

/**
 * Selects indexed element addressing with one signed i32/i64 index per lane.
 *
 * At the raw vec layer a scale of zero means `sizeof(memory element)` and an
 * explicit scale is a byte multiplier. A tensor accessor may instead lower
 * scale-zero logical offsets through its layout, so this option alone does
 * not guarantee a physical gather/scatter instruction. The named index vector
 * is retained by reference so this also works with sizeless SVE types.
 */
template <VectorValue V, int S = 0>
struct Indexed {
  static_assert(
      S == 0 || S == 1 || S == 2 || S == 4 || S == 8,
      "indexed scale must be 0 (auto), 1, 2, 4, or 8");
  static constexpr int scale = S;
  const V& indices;
};

template <VectorValue V>
VECOPS_ALWAYS_INLINE Indexed<V> indexed(const V& indices) {
  return {indices};
}

template <VectorValue V, int S>
VECOPS_ALWAYS_INLINE Indexed<V, S> indexed(
    const V& indices, Scale<S>) {
  return {indices};
}

template <typename V>
  requires VectorValue<std::remove_cvref_t<V>> &&
           (!std::is_lvalue_reference_v<V>)
Indexed<std::remove_cvref_t<V>> indexed(V&&) = delete;

template <typename V, int S>
  requires VectorValue<std::remove_cvref_t<V>> &&
           (!std::is_lvalue_reference_v<V>)
Indexed<std::remove_cvref_t<V>, S> indexed(V&&, Scale<S>) =
    delete;

namespace details {

template <typename>
struct IsStrideMetadata : std::false_type {};

template <nint_t N>
struct IsStrideMetadata<meta::Const<N>> : std::true_type {};

template <nint_t Alignment, nint_t Lo, nint_t Hi>
struct IsStrideMetadata<meta::Dynamic<Alignment, Lo, Hi>> : std::true_type {};

} // namespace details

/**
 * Selects a constant or runtime stride in the logical element space of the
 * receiving abstraction. Raw vec memory operations apply it to pointer
 * elements; a tensor accessor composes it with the selected tensor-axis
 * stride. The option alone does not determine load versus gather lowering.
 * Accepts meta::Const<N> for compile-time strides, meta::Dynamic for
 * bounded runtime strides, or a plain nint_t (via meta::Any). Strided and
 * indexed addressing are mutually exclusive and cannot be combined with
 * alignment options.
 */
template <typename Stride, int S = 0>
  requires details::IsStrideMetadata<std::remove_cvref_t<Stride>>::value
struct Strided {
  static_assert(
      S == 0 || S == 1 || S == 2 || S == 4 || S == 8,
      "strided scale must be 0 (pointer elements), 1, 2, 4, or 8 bytes");
  using StrideType = std::remove_cvref_t<Stride>;
  static constexpr int scale = S;
  StrideType stride;
};

template <nint_t N>
VECOPS_ALWAYS_INLINE constexpr Strided<meta::Const<N>> strided(
    meta::Const<N> stride) {
  return {stride};
}

template <nint_t Alignment, nint_t Lo, nint_t Hi>
VECOPS_ALWAYS_INLINE constexpr Strided<meta::Dynamic<Alignment, Lo, Hi>>
strided(meta::Dynamic<Alignment, Lo, Hi> stride) {
  return {stride};
}

VECOPS_ALWAYS_INLINE constexpr Strided<meta::Any> strided(nint_t stride) {
  return {meta::Any{stride}};
}

template <int S>
VECOPS_ALWAYS_INLINE constexpr Strided<meta::Any, S> strided(
    nint_t stride, Scale<S>) {
  return {meta::Any{stride}};
}

template <typename Stride, int S>
  requires details::IsStrideMetadata<std::remove_cvref_t<Stride>>::value
VECOPS_ALWAYS_INLINE constexpr Strided<std::remove_cvref_t<Stride>, S>
strided(Stride stride, Scale<S>) {
  return {stride};
}

} // namespace vecops::vec::opt

namespace vecops::vec {

// Addressing factories are also available unqualified alongside load/store.
using opt::indexed;
using opt::resources;
using opt::scale;
using opt::strided;

} // namespace vecops::vec

namespace vecops::vec::cvt {

/**
 * Preserves the natural logical lane order during conversion.
 * output[i] is derived from input[i] for equal-lane-shape Tags.
 * This is the default layout.
 */
struct Ordered {};

/**
 * Allows a stable backend-native lane permutation.
 *
 * Equal-width element conversion has ordered lane provenance. For any three
 * compatible element types A, B, and C, converting A->B->C has the same lane
 * provenance as converting A->C directly. Consequently A->B->A restores the
 * original lane positions. This contract concerns ordering only; value loss,
 * rounding, saturation, and wrapping follow the selected value policy.
 */
struct Unordered {};

/**
 * Selects or inserts one lane phase of a widening or narrowing conversion.
 *
 * For equal-byte Tags: widening selects input lane ratio*i+Phase, while
 * narrowing writes converted input lane i to output lane ratio*i+Phase.
 * Phase 0 is always supported; Phase 1 requires a 2x element-width ratio.
 */
template <int Phase>
struct Lane {
  static constexpr int phase = Phase;
};

/**
 * Clamps values that are outside a narrowing destination range.
 * For float→int: clamps to [min, max] of the integer destination type.
 * For int→narrower-int: clamps to the narrower range.
 * This is the default value policy.
 */
struct Saturate {};

/**
 * Keeps only the low destination-width bits of a narrowing integer
 * conversion. Valid only for integer-source, narrower-integer-destination
 * conversions. Equivalent to a C-style truncating cast.
 */
struct Wrap {};

inline constexpr Ordered ordered{};
inline constexpr Unordered unordered{};
template <int Phase>
inline constexpr Lane<Phase> lane{};
inline constexpr Saturate saturate{};
inline constexpr Wrap wrap{};
inline constexpr Wrap truncate{};

} // namespace vecops::vec::cvt

namespace vecops::vec::mem {

/** Makes no stronger alignment promise than alignof(element). This is the default. */
struct Unaligned {};

/**
 * Requires the base address to satisfy memory_alignment(tag) byte alignment.
 * An assertion validates alignment at runtime in debug builds.
 */
struct Aligned {};

/** Uses ordinary cacheable memory access semantics. This is the default. */
struct Temporal {};

/**
 * Requests a non-temporal (streaming) access when the selected backend
 * supports one. Silently degrades to a temporal access on unsupported
 * hardware.
 */
struct NonTemporal {};

/**
 *  store_convert narrowing layout: merge all source words into one destination
 *  word before writing.
 *
 *  When a narrowing store_convert spans multiple source words (e.g. two
 *  512-bit f32 words → one 512-bit bf16 word), packed converts every source
 *  half first, concatenates the resulting narrow vectors, and emits a single
 *  wide store.  This avoids multiple partial cache-line writes that would
 *  degrade write throughput on some hardware.
 *
 *  This is the default layout for store_convert.
 */
struct Packed {};

/**
 *  store_convert narrowing layout: write each source word's narrowed result
 *  independently, without merging.
 *
 *  Each source word is converted to its destination representation and stored
 *  immediately at the corresponding offset in the output buffer.  This gives
 *  the backend freedom to select the most natural word-level layout but may
 *  result in multiple narrower stores where packed would have issued one wide
 *  store.
 */
struct Split {};

inline constexpr Unaligned unaligned{};
inline constexpr Aligned aligned{};
inline constexpr Temporal temporal{};
inline constexpr NonTemporal non_temporal{};
inline constexpr Packed packed{};
inline constexpr Split split{};

/**
 * Requests that a prefetched cache line be placed in L1.
 * This is the default cache level for prefetch.
 */
struct PrefetchL1 {};

/** Requests that a prefetched cache line be placed in L2. */
struct PrefetchL2 {};

/**
 * Requests temporal prefetch behavior that keeps the cache line.
 * This is the default temporality for prefetch.
 */
struct PrefetchKeep {};

/**
 * Requests streaming (non-temporal) prefetch behavior.
 * The cache line is evicted after use, avoiding cache pollution.
 */
struct PrefetchStream {};

/**
 * Declares that a prefetch is intended for a subsequent read.
 * This is the default intent for prefetch.
 */
struct PrefetchRead {};

/** Declares that a prefetch is intended for a subsequent write. */
struct PrefetchWrite {};

inline constexpr PrefetchL1 prefetch_l1{};
inline constexpr PrefetchL2 prefetch_l2{};
inline constexpr PrefetchKeep prefetch_keep{};
inline constexpr PrefetchStream prefetch_stream{};
inline constexpr PrefetchRead prefetch_read{};
inline constexpr PrefetchWrite prefetch_write{};

} // namespace vecops::vec::mem

#endif // VECOPS_VEC_OPTIONS_H
