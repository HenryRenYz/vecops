#ifndef VECOPS_VEC_REDUCTION_H
#define VECOPS_VEC_REDUCTION_H

#include <limits>
#include <utility>

#include "vecops/vec/Arithmetic.h"
#include "vecops/vec/Request.h"

namespace vecops::vec {

namespace details {

template <typename Tag, typename... Options>
concept ReductionFilterOptions =
    VectorTag<Tag> && (sizeof...(Options) > 0) &&
    ((is_masked_option_for_v<Tag, Options> ||
      is_unmasked_option_v<std::remove_cvref_t<Options>>) && ...) &&
    (option_count_v<IsMaskedOption, Options...> +
         option_count_v<IsUnmaskedOption, Options...> == 1);

} // namespace details

/* **************************************************************************** */
//    Horizontal reduction: reduce_add, reduce_max, reduce_min           //
/* **************************************************************************** */

#define VECOPS_VEC_DECLARE_REDUCTION_OP(OpType)                        \
  struct OpType {                                                       \
    template <VectorTag Tag>                                            \
    VECOPS_ALWAYS_INLINE ElementOf<Tag> operator()(                     \
        Tag tag, Vec<Tag> value) const;                                 \
    template <VectorTag Tag, typename... Options>                       \
      requires details::ReductionFilterOptions<Tag, Options...>        \
    VECOPS_ALWAYS_INLINE ElementOf<Tag> operator()(                     \
        Tag tag, Vec<Tag> value, Options&&... options) const;           \
    template <VectorTag Tag, Active A>                                  \
    VECOPS_ALWAYS_INLINE ElementOf<Tag> operator()(                     \
        Tag tag, Vec<Tag> value,                                       \
        const ReduceRequest<Tag, A>& request) const;                    \
  }

VECOPS_VEC_DECLARE_REDUCTION_OP(ReduceAddOp);
VECOPS_VEC_DECLARE_REDUCTION_OP(ReduceMaxOp);
VECOPS_VEC_DECLARE_REDUCTION_OP(ReduceMinOp);

#undef VECOPS_VEC_DECLARE_REDUCTION_OP

/* **************************************************************************** */
//    Reduction identity elements                                         //
/* **************************************************************************** */

namespace details {

template <typename Op, Element T>
VECOPS_ALWAYS_INLINE T reduction_identity() {
  if constexpr (std::same_as<Op, ReduceAddOp>) {
    return T{};
  } else if constexpr (std::same_as<Op, ReduceMaxOp>) {
    if constexpr (::vecops::is_float_v<T>) {
      return static_cast<T>(-std::numeric_limits<double>::infinity());
    } else {
      return std::numeric_limits<T>::lowest();
    }
  } else if constexpr (std::same_as<Op, ReduceMinOp>) {
    if constexpr (::vecops::is_float_v<T>) {
      return static_cast<T>(std::numeric_limits<double>::infinity());
    } else {
      return std::numeric_limits<T>::max();
    }
  } else {
    static_assert(dispatch_dependent_false<Op>, "unsupported reduction op");
  }
}

} // namespace details

} // namespace vecops::vec

#include "vecops/vec/details/Dispatch.h"
#include "vecops/vec/details/Reduction.h"
#include "vecops/vec/details/scalar/Reduction.h"

#if defined(ARCH_X86_FAMILY) && !defined(CPU_CAPABILITY_GENERIC)
#include "vecops/vec/details/x86/Reduction.h"
#elif defined(CPU_CAPABILITY_SVE)
#include "vecops/vec/details/sve/Reduction.h"
#endif

namespace vecops::vec {

/**
 * Returns the sum of all logical lanes in value.
 *
 * Integer addition wraps modulo the element width. Floating-point association
 * is backend-defined. Multi-word values are folded with vector additions and
 * then horizontally reduced once. An empty masked selection returns zero
 * (the additive identity).
 *
 * @see reduce_max, reduce_min for other reduction operations.
 * @see add for the lane-wise operation used in the fold.
 */
template <VectorTag Tag>
VECOPS_ALWAYS_INLINE ElementOf<Tag> ReduceAddOp::operator()(
    Tag tag, Vec<Tag> value) const {
  return details::execute(*this, tag, value);
}

/**
 * With opt::unmasked, returns the sum of every lane. With exactly one
 * opt::masked(mask), returns the sum of the selected lanes.
 * Unselected lanes do not participate. An empty selection returns zero.
 * Population options such as opt::zero and opt::merge are not valid because
 * a reduction has no inactive output lanes.
 */
template <VectorTag Tag, typename... Options>
  requires details::ReductionFilterOptions<Tag, Options...>
VECOPS_ALWAYS_INLINE ElementOf<Tag> ReduceAddOp::operator()(
    Tag tag, Vec<Tag> value, Options&&... options) const {
  return details::execute_reduction_options(
      *this, tag, value, std::forward<Options>(options)...);
}
template <VectorTag Tag, Active A>
VECOPS_ALWAYS_INLINE ElementOf<Tag> ReduceAddOp::operator()(
    Tag tag, Vec<Tag> value,
    const ReduceRequest<Tag, A>& request) const {
  return details::execute_reduction_request(*this, tag, value, request);
}

inline constexpr ReduceAddOp reduce_add{};

/**
 * Returns the greatest logical lane. Multi-word values are combined with
 * lane-wise maxima before one horizontal reduction. Floating NaN and
 * signed-zero selection follows the active backend. An empty masked selection
 * returns negative infinity for floating elements or the lowest integer value.
 *
 * @see reduce_add, reduce_min for other reduction operations.
 * @see max for the lane-wise operation used in the fold.
 */
template <VectorTag Tag>
VECOPS_ALWAYS_INLINE ElementOf<Tag> ReduceMaxOp::operator()(
    Tag tag, Vec<Tag> value) const {
  return details::execute(*this, tag, value);
}

/**
 * With opt::unmasked, returns the greatest lane. With exactly one
 * opt::masked(mask), returns the greatest selected lane.
 * Unselected lanes do not participate. An empty selection returns negative
 * infinity for floating elements or the lowest integer value. Population
 * options are invalid because a reduction has no inactive output lanes.
 */
template <VectorTag Tag, typename... Options>
  requires details::ReductionFilterOptions<Tag, Options...>
VECOPS_ALWAYS_INLINE ElementOf<Tag> ReduceMaxOp::operator()(
    Tag tag, Vec<Tag> value, Options&&... options) const {
  return details::execute_reduction_options(
      *this, tag, value, std::forward<Options>(options)...);
}
template <VectorTag Tag, Active A>
VECOPS_ALWAYS_INLINE ElementOf<Tag> ReduceMaxOp::operator()(
    Tag tag, Vec<Tag> value,
    const ReduceRequest<Tag, A>& request) const {
  return details::execute_reduction_request(*this, tag, value, request);
}

inline constexpr ReduceMaxOp reduce_max{};

/**
 * Returns the least logical lane. Multi-word values are combined with
 * lane-wise minima before one horizontal reduction. Floating NaN and
 * signed-zero selection follows the active backend. An empty masked selection
 * returns positive infinity for floating elements or the greatest integer
 * value.
 *
 * @see reduce_add, reduce_max for other reduction operations.
 * @see min for the lane-wise operation used in the fold.
 */
template <VectorTag Tag>
VECOPS_ALWAYS_INLINE ElementOf<Tag> ReduceMinOp::operator()(
    Tag tag, Vec<Tag> value) const {
  return details::execute(*this, tag, value);
}

/**
 * With opt::unmasked, returns the least lane. With exactly one
 * opt::masked(mask), returns the least selected lane.
 * Unselected lanes do not participate. An empty selection returns positive
 * infinity for floating elements or the greatest integer value. Population
 * options are invalid because a reduction has no inactive output lanes.
 */
template <VectorTag Tag, typename... Options>
  requires details::ReductionFilterOptions<Tag, Options...>
VECOPS_ALWAYS_INLINE ElementOf<Tag> ReduceMinOp::operator()(
    Tag tag, Vec<Tag> value, Options&&... options) const {
  return details::execute_reduction_options(
      *this, tag, value, std::forward<Options>(options)...);
}
template <VectorTag Tag, Active A>
VECOPS_ALWAYS_INLINE ElementOf<Tag> ReduceMinOp::operator()(
    Tag tag, Vec<Tag> value,
    const ReduceRequest<Tag, A>& request) const {
  return details::execute_reduction_request(*this, tag, value, request);
}

inline constexpr ReduceMinOp reduce_min{};

} // namespace vecops::vec

#endif // VECOPS_VEC_REDUCTION_H
