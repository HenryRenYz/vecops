#ifndef VECOPS_VEC_CONVERSION_MEMORY_H
#define VECOPS_VEC_CONVERSION_MEMORY_H

#include <type_traits>
#include <utility>

#include "vecops/vec/Conversion.h"
#include "vecops/vec/Memory.h"
#include "vecops/vec/Request.h"
#include "vecops/vec/details/Request.h"

/**
 * @file ConversionMemory.h
 * @brief Load/store operations fused with element conversion.
 *
 * `load_convert(to_tag, pointer, options...)` and
 * `store_convert(from_tag, pointer, value, options...)` preserve the logical
 * lane count of the caller's Tag while the memory element type may differ.
 * Backends may use a fused instruction or an equivalent load/convert sequence;
 * the API promises semantics, not one opcode.
 * When the memory and logical element types are identical, the public boundary
 * directly forwards to `load` or `store`. Conversion-only options are consumed
 * there, while all ordinary memory options are preserved. Identity conversion
 * therefore has exactly the same lowering as the corresponding memory API and
 * never enters backend conversion machinery.
 *
 * Any otherwise legal caller Tag is accepted even when rebinding that Tag to
 * the memory dtype would exceed the backend's representable POW2 range. A
 * backend whole-operation lowering is preferred whenever it can process the
 * request without materializing that oversized memory-side representation.
 * Only otherwise is the request recursively partitioned into legal logical
 * chunks and reassembled (load) or emitted (store). For active non-contiguous
 * boundary cases, the generic fallback may operate lane-by-lane to preserve
 * exact mask/address semantics.
 *
 * ## Important mask domains
 *
 * Ordered conversion uses a caller-logical mask. Unordered conversion normally
 * uses the memory-side rebound Tag's mask because the backend may permute lanes.
 * Its lane permutation is the same stable, compositional layout specified by
 * `convert`; fused and load/convert or convert/store lowerings are equivalent.
 * When that rebound Tag is not representable, only the caller-logical mask
 * type is valid, whether a backend handles the whole request directly or the
 * generic boundary partitions it. `first(n)` avoids this distinction and is
 * generally the simplest tail interface.
 *
 * ## Pitfalls
 *
 * - Ordered/unordered and saturate/wrap are semantic choices, not mere hints.
 * - Explicit indexed scale is in bytes at this raw pointer layer; scale zero
 *   means one memory element. Tensor DataAccess assigns different higher-level
 *   logical semantics before lowering to this API.
 * - Masked-off addresses are not accessed. Inactive load lanes are zero unless
 *   a merge population is provided; inactive stores leave memory unchanged.
 * - Alignment and temporality are hints and may be ignored by a backend while
 *   preserving the conversion and memory semantics.
 */
namespace vecops::vec {

namespace details {
template <VectorTag ToTag, Element From, bool IsStore, typename... Options>
consteval bool valid_memory_conversion_options();

template <VectorTag LogicalTag, Element Other>
consteval bool memory_rebind_supported();

/**
 * Backend capability for a conversion whose memory-side Rebind exceeds the
 * public scalable-Tag representation limit.
 *
 * A specialization promises that the backend's whole-operation lowering can
 * consume the original public Tag without materializing the oversized
 * memory-side Vec/Mask. The capability is option-sensitive because contiguous
 * ordered conversion may be supported when indexed or unordered conversion is
 * not.
 */
template <typename Backend, typename Op, VectorTag LogicalTag, Element Other,
          typename... Options>
struct HasOversizedMemoryConversionLowering : std::false_type {};

template <typename Backend, typename Op, VectorTag LogicalTag, Element Other,
          typename... Options>
inline constexpr bool has_oversized_memory_conversion_lowering_v =
    HasOversizedMemoryConversionLowering<
        Backend, Op, LogicalTag, Other,
        std::remove_cvref_t<Options>...>::value;
}

/* **************************************************************************** */
//    Load and store with element conversion                              //
/* **************************************************************************** */

struct LoadConvertOp {
  template <VectorTag ToTag, Element From, typename... Options>
    requires (details::valid_memory_conversion_options<
              ToTag, From, false, Options...>())
  VECOPS_ALWAYS_INLINE Vec<ToTag> operator()(
      ToTag to, const From* pointer, Options&&... options) const;

  /// Direct entry with an already-resolved request; skips option parsing.
  /// Requires an identity pair or a representable memory-side rebind; the
  /// oversized partitioner remains reachable through the option-pack entry.
  template <VectorTag ToTag, Element From, Active A, Addressing Addr,
            Populate P, typename Alignment, typename Temporality,
            int IndexScale, VectorValue IndexVector, typename Layout,
            typename ValuePolicy, MaskValue MaskVector, typename Resources>
    requires (std::same_as<From, ElementOf<ToTag>> ||
              details::memory_rebind_supported<ToTag, From>())
  VECOPS_ALWAYS_INLINE Vec<ToTag> operator()(
      ToTag to, const From* pointer,
      LoadConvertRequest<
          ToTag, From, A, Addr, P, Alignment, Temporality, IndexScale,
          IndexVector, Layout, ValuePolicy, MaskVector, Resources> request) const;
};

struct StoreConvertOp {
  template <VectorTag FromTag, Element To, typename... Options>
    requires (details::valid_memory_conversion_options<
              FromTag, To, true, Options...>())
  VECOPS_ALWAYS_INLINE void operator()(
      FromTag from, To* pointer, Vec<FromTag> value,
      Options&&... options) const;

  /// Direct entry with an already-resolved request; skips option parsing.
  /// Requires an identity pair or a representable memory-side rebind.
  template <VectorTag FromTag, Element To, Active A, Addressing Addr,
            typename Alignment, typename Temporality, int IndexScale,
            VectorValue IndexVector, typename Layout, typename ValuePolicy,
            typename Packing, MaskValue MaskVector, typename Resources>
    requires (std::same_as<To, ElementOf<FromTag>> ||
              details::memory_rebind_supported<FromTag, To>())
  VECOPS_ALWAYS_INLINE void operator()(
      FromTag from, To* pointer, Vec<FromTag> value,
      StoreConvertRequest<
          FromTag, To, A, Addr, Alignment, Temporality, IndexScale,
          IndexVector, Layout, ValuePolicy, Packing, MaskVector, Resources>
          request) const;
};

} // namespace vecops::vec

#if defined(ARCH_X86_FAMILY) && !defined(CPU_CAPABILITY_GENERIC)
#include "vecops/vec/details/x86/ConversionMemory.h"
#elif defined(CPU_CAPABILITY_SVE)
#include "vecops/vec/details/sve/ConversionMemory.h"
#endif

#include "vecops/vec/details/ConversionMemory.h"

namespace vecops::vec {

/**
 * Loads consecutive `From` elements and converts them to `ElementOf<ToTag>`.
 * The source Tag is `Rebind<From, ToTag>`, preserving the output's logical
 * lane shape. Defaults are ordered, saturating, unaligned, temporal, and
 * packed. `cvt::wrap` is available only for integer narrowing.
 *
 * `opt::masked` and `opt::first` filter memory accesses; inactive addresses
 * are never read. For ordered conversion a mask has type `Mask<ToTag>` and an
 * inactive output lane is zero unless `opt::merge(scalar/vector)` is supplied.
 * For unordered conversion the mask normally has the memory-side source Tag
 * and no population option is accepted. If that rebound Tag would exceed the
 * backend maximum POW, the public logical-Tag mask is used while either a
 * backend whole-request lowering or the generic partitioner handles it.
 * `mem::aligned` promises source-Tag memory
 * alignment. Alignment and temporality are hints and may be ignored while the
 * backend retains its fused conversion-load path. `mem::split` is not a valid
 * load option. `opt::indexed(indices)` and `opt::strided(stride)` select
 * non-contiguous source addresses and are mutually exclusive with alignment;
 * an explicit indexed scale is measured in bytes and scale zero uses
 * `sizeof(From)`.
 *
 * @see load for same-type contiguous load.
 * @see store_convert for converting stores.
 */
template <VectorTag ToTag, Element From, typename... Options>
  requires (details::valid_memory_conversion_options<
            ToTag, From, false, Options...>())
VECOPS_ALWAYS_INLINE Vec<ToTag> LoadConvertOp::operator()(
    ToTag to, const From* pointer, Options&&... options) const {
  if constexpr (std::same_as<From, ElementOf<ToTag>>) {
    auto invoke = [&](auto&&... memory_options) VECOPS_INLINE_LAMBDA {
      return LoadOp{}(
          to, pointer,
          std::forward<decltype(memory_options)>(memory_options)...);
    };
    return details::apply_identity_memory_options<false, ToTag>(
        invoke, std::forward<Options>(options)...);
  } else if constexpr (
      details::memory_rebind_supported<ToTag, From>() ||
      details::has_oversized_memory_conversion_lowering_v<
          details::CurrentBackend, LoadConvertOp, ToTag, From, Options...>) {
    if constexpr (details::memory_rebind_supported<ToTag, From>()) {
      return details::execute_load_convert_request(
          *this, to, pointer,
          details::resolve_load_convert_request<ToTag, From>(
              std::forward<Options>(options)...));
    } else {
      return details::execute_load_convert_options(
          *this, to, pointer, std::forward<Options>(options)...);
    }
  } else {
    return details::execute_large_load_convert(
        to, pointer, std::forward<Options>(options)...);
  }
}

template <VectorTag ToTag, Element From, Active A, Addressing Addr,
          Populate P, typename Alignment, typename Temporality,
          int IndexScale, VectorValue IndexVector, typename Layout,
          typename ValuePolicy, MaskValue MaskVector, typename Resources>
  requires (std::same_as<From, ElementOf<ToTag>> ||
            details::memory_rebind_supported<ToTag, From>())
VECOPS_ALWAYS_INLINE Vec<ToTag> LoadConvertOp::operator()(
    ToTag to, const From* pointer,
    LoadConvertRequest<
        ToTag, From, A, Addr, P, Alignment, Temporality, IndexScale,
        IndexVector, Layout, ValuePolicy, MaskVector, Resources> request) const {
  if constexpr (std::same_as<From, ElementOf<ToTag>>) {
    return details::execute_load_request(
        LoadOp{}, to, pointer, request);
  } else {
    return details::execute_load_convert_request(
        *this, to, pointer, request);
  }
}

/**
 * Converts `value` and stores consecutive `To` elements. The memory-side Tag
 * is `Rebind<To, FromTag>`, preserving the input's logical lane shape.
 * Defaults are ordered, saturating, unaligned, temporal, and packed.
 *
 * `opt::masked` and `opt::first` filter writes, so inactive addresses remain
 * untouched. Ordered masks have type `Mask<FromTag>`; unordered masks normally
 * have the memory-side Tag. For a rebound Tag beyond the backend maximum POW,
 * unordered fallback uses a logical `Mask<FromTag>` instead.
 * Population options are invalid. `mem::split` is accepted
 * only for ordered, saturating stores and has the same contiguous logical
 * result as `mem::packed`; it permits a backend-native wordwise implementation.
 * `mem::aligned` promises memory-side Tag alignment. Alignment and temporality
 * are hints and may be ignored while retaining a fused conversion-store path.
 * `opt::indexed(indices)` and `opt::strided(stride)` select non-contiguous
 * destination addresses and are mutually exclusive with alignment; scale zero
 * uses `sizeof(To)`.
 *
 * @see store for same-type contiguous store.
 * @see load_convert for converting loads.
 */
template <VectorTag FromTag, Element To, typename... Options>
  requires (details::valid_memory_conversion_options<
            FromTag, To, true, Options...>())
VECOPS_ALWAYS_INLINE void StoreConvertOp::operator()(
    FromTag from, To* pointer, Vec<FromTag> value,
    Options&&... options) const {
  if constexpr (std::same_as<To, ElementOf<FromTag>>) {
    auto invoke = [&](auto&&... memory_options) VECOPS_INLINE_LAMBDA {
      StoreOp{}(
          from, pointer, value,
          std::forward<decltype(memory_options)>(memory_options)...);
    };
    details::apply_identity_memory_options<true, FromTag>(
        invoke, std::forward<Options>(options)...);
  } else if constexpr (
      details::memory_rebind_supported<FromTag, To>() ||
      details::has_oversized_memory_conversion_lowering_v<
          details::CurrentBackend, StoreConvertOp, FromTag, To, Options...>) {
    if constexpr (details::memory_rebind_supported<FromTag, To>()) {
      details::execute_store_convert_request(
          *this, from, pointer, value,
          details::resolve_store_convert_request<FromTag, To>(
              std::forward<Options>(options)...));
    } else {
      details::execute_store_convert_options(
          *this, from, pointer, value,
          std::forward<Options>(options)...);
    }
  } else {
    details::execute_large_store_convert(
        from, pointer, value, std::forward<Options>(options)...);
  }
}

template <VectorTag FromTag, Element To, Active A, Addressing Addr,
          typename Alignment, typename Temporality, int IndexScale,
          VectorValue IndexVector, typename Layout, typename ValuePolicy,
          typename Packing, MaskValue MaskVector, typename Resources>
  requires (std::same_as<To, ElementOf<FromTag>> ||
            details::memory_rebind_supported<FromTag, To>())
VECOPS_ALWAYS_INLINE void StoreConvertOp::operator()(
    FromTag from, To* pointer, Vec<FromTag> value,
    StoreConvertRequest<
        FromTag, To, A, Addr, Alignment, Temporality, IndexScale,
        IndexVector, Layout, ValuePolicy, Packing, MaskVector, Resources>
        request) const {
  if constexpr (std::same_as<To, ElementOf<FromTag>>) {
    details::execute_store_request(
        StoreOp{}, from, pointer, value, request);
  } else {
    details::execute_store_convert_request(
        *this, from, pointer, value, request);
  }
}

inline constexpr LoadConvertOp load_convert{};
inline constexpr StoreConvertOp store_convert{};

} // namespace vecops::vec

#endif // VECOPS_VEC_CONVERSION_MEMORY_H
