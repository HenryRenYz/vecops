#ifndef VECOPS_VEC_CONVERSION_MEMORY_H
#define VECOPS_VEC_CONVERSION_MEMORY_H

#include <utility>

#include "vecops/vec/Conversion.h"
#include "vecops/vec/Memory.h"

namespace vecops::vec {

namespace details {
template <VectorTag ToTag, Element From, bool IsStore, typename... Options>
consteval bool valid_memory_conversion_options();
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
};

struct StoreConvertOp {
  template <VectorTag FromTag, Element To, typename... Options>
    requires (details::valid_memory_conversion_options<
              FromTag, To, true, Options...>())
  VECOPS_ALWAYS_INLINE void operator()(
      FromTag from, To* pointer, Vec<FromTag> value,
      Options&&... options) const;
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
 * For unordered conversion the mask has the memory-side source Tag and no
 * population option is accepted. `mem::aligned` promises source-Tag memory
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
  return details::execute_load_convert_options(
      *this, to, pointer, std::forward<Options>(options)...);
}

/**
 * Converts `value` and stores consecutive `To` elements. The memory-side Tag
 * is `Rebind<To, FromTag>`, preserving the input's logical lane shape.
 * Defaults are ordered, saturating, unaligned, temporal, and packed.
 *
 * `opt::masked` and `opt::first` filter writes, so inactive addresses remain
 * untouched. Ordered masks have type `Mask<FromTag>`; unordered masks have the
 * memory-side Tag. Population options are invalid. `mem::split` is accepted
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
  details::execute_store_convert_options(
      *this, from, pointer, value, std::forward<Options>(options)...);
}

inline constexpr LoadConvertOp load_convert{};
inline constexpr StoreConvertOp store_convert{};

} // namespace vecops::vec

#endif // VECOPS_VEC_CONVERSION_MEMORY_H
