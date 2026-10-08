// SPDX-FileCopyrightText: 2026 HenryRenYz and vecops contributors
// SPDX-License-Identifier: MIT
#ifndef VECOPS_VEC_ROUNDING_H
#define VECOPS_VEC_ROUNDING_H

/**
 * @file Rounding.h
 * @brief Floating-point, same-type lane-wise rounding operations.
 */

#include "vecops/vec/Arithmetic.h"

namespace vecops::vec {

#define VECOPS_VEC_DECLARE_ROUNDING_OP(OpType)                          \
  struct OpType {                                                       \
    template <FloatingTag Tag>                                         \
    VECOPS_ALWAYS_INLINE Vec<Tag> operator()(                           \
        Tag tag, Vec<Tag> value) const;                                 \
    template <FloatingTag Tag, typename... Options>                     \
      requires (sizeof...(Options) > 0)                                \
    VECOPS_ALWAYS_INLINE Vec<Tag> operator()(                           \
        Tag tag, Vec<Tag> value, Options&&... options) const;           \
    template <FloatingTag Tag, Active A, Inactive I>                    \
    VECOPS_ALWAYS_INLINE Vec<Tag> operator()(                           \
        Tag tag, Vec<Tag> value,                                       \
        const OpRequest<Tag, A, I>& request) const;                     \
    template <FloatingVectorValue V, typename... Options>               \
    VECOPS_ALWAYS_INLINE V operator()(                                  \
        V value, Options&&... options) const {                          \
      return (*this)(                                                   \
          VecToTag<V>{}, value, std::forward<Options>(options)...);    \
    }                                                                   \
    template <FloatingTag Tag, typename Policy>                         \
      requires (details::arithmetic_inactive_policy<Policy>)            \
    inline Vec<Tag> operator()(                                         \
        Tag tag, Vec<Tag> value, Mask<Tag> mask,                        \
        Vec<Tag> inactive, Policy policy) const {                       \
      return details::execute(                                          \
          *this, tag, value, mask, inactive, policy);                   \
    }                                                                   \
    template <FloatingTag Tag>                                          \
      requires (details::multi_word_tag_v<Tag>)                         \
    inline NativeWordVec<Tag> operator()(                               \
        Tag tag, NativeWordVec<Tag> value) const {                      \
      return details::execute_word<0, details::CurrentBackend>(         \
          *this, tag, value);                                           \
    }                                                                   \
  }

VECOPS_VEC_DECLARE_ROUNDING_OP(FloorOp);
VECOPS_VEC_DECLARE_ROUNDING_OP(CeilOp);
VECOPS_VEC_DECLARE_ROUNDING_OP(TruncOp);
VECOPS_VEC_DECLARE_ROUNDING_OP(RoundOp);
VECOPS_VEC_DECLARE_ROUNDING_OP(RoundEvenOp);
VECOPS_VEC_DECLARE_ROUNDING_OP(NearbyIntOp);
VECOPS_VEC_DECLARE_ROUNDING_OP(RintOp);

#undef VECOPS_VEC_DECLARE_ROUNDING_OP

inline constexpr FloorOp floor{};
inline constexpr CeilOp ceil{};
inline constexpr TruncOp trunc{};
inline constexpr RoundOp round{};
inline constexpr RoundEvenOp round_even{};
inline constexpr NearbyIntOp nearbyint{};
inline constexpr RintOp rint{};

} // namespace vecops::vec

#include "vecops/vec/details/Dispatch.h"
#include "vecops/vec/details/Rounding.h"
#include "vecops/vec/details/scalar/Rounding.h"

#if defined(ARCH_X86_FAMILY) && !defined(CPU_CAPABILITY_GENERIC)
#include "vecops/vec/details/x86/Rounding.h"
#elif defined(CPU_CAPABILITY_SVE)
#include "vecops/vec/details/sve/Rounding.h"
#endif

namespace vecops::vec {

#define VECOPS_VEC_DEFINE_ROUNDING_OP(OpType)                           \
  template <FloatingTag Tag>                                           \
  VECOPS_ALWAYS_INLINE Vec<Tag> OpType::operator()(                     \
      Tag tag, Vec<Tag> value) const {                                  \
    return details::execute(*this, tag, value);                         \
  }                                                                     \
  template <FloatingTag Tag, typename... Options>                       \
    requires (sizeof...(Options) > 0)                                  \
  VECOPS_ALWAYS_INLINE Vec<Tag> OpType::operator()(                     \
      Tag tag, Vec<Tag> value, Options&&... options) const {            \
    return details::execute_unary_arithmetic_options(                  \
        *this, tag, value, std::forward<Options>(options)...);          \
  }                                                                     \
  template <FloatingTag Tag, Active A, Inactive I>                      \
  VECOPS_ALWAYS_INLINE Vec<Tag> OpType::operator()(                     \
      Tag tag, Vec<Tag> value,                                         \
      const OpRequest<Tag, A, I>& request) const {                      \
    return details::execute_unary_arithmetic_request(                  \
        *this, tag, value, request);                                    \
  }

/** Rounds each lane toward negative infinity. */
VECOPS_VEC_DEFINE_ROUNDING_OP(FloorOp);
/** Rounds each lane toward positive infinity. */
VECOPS_VEC_DEFINE_ROUNDING_OP(CeilOp);
/** Rounds each lane toward zero. */
VECOPS_VEC_DEFINE_ROUNDING_OP(TruncOp);
/** Rounds each lane to nearest, with halfway cases away from zero. */
VECOPS_VEC_DEFINE_ROUNDING_OP(RoundOp);
/** Rounds each lane to nearest, with halfway cases to even. */
VECOPS_VEC_DEFINE_ROUNDING_OP(RoundEvenOp);
/** Rounds using the current floating-point mode without raising inexact. */
VECOPS_VEC_DEFINE_ROUNDING_OP(NearbyIntOp);
/** Rounds using the current floating-point mode and may raise inexact. */
VECOPS_VEC_DEFINE_ROUNDING_OP(RintOp);

#undef VECOPS_VEC_DEFINE_ROUNDING_OP

} // namespace vecops::vec

#endif // VECOPS_VEC_ROUNDING_H
