#ifndef VECOPS_VEC_DETAILS_MATH_H
#define VECOPS_VEC_DETAILS_MATH_H

/**
 * @file Math.h
 * @brief Backend-independent math infrastructure: EnableElementwiseWordBatching
 * opt-ins (all six exp operations support word batching) and GenericImpl
 * (multi-word fallback reusing UnaryArithmeticGenericImpl).
 *
 * The actual precision-tier logic lives in each backend's NativeWordImpl
 * specialization.  This file just registers the exp operations as eligible
 * for automatic word batching and provides the multi-word GenericImpl.
 */

#include "vecops/vec/details/Arithmetic.h"

namespace vecops::vec::details {

#define VECOPS_VEC_ENABLE_EXP_BATCHING(OpType)                         \
  template <>                                                          \
  struct EnableElementwiseWordBatching<OpType> : std::true_type {}

VECOPS_VEC_ENABLE_EXP_BATCHING(ExpOp);
VECOPS_VEC_ENABLE_EXP_BATCHING(ExpFastOp);
VECOPS_VEC_ENABLE_EXP_BATCHING(ExpEstOp);
VECOPS_VEC_ENABLE_EXP_BATCHING(ExpNegOp);
VECOPS_VEC_ENABLE_EXP_BATCHING(ExpNegFastOp);
VECOPS_VEC_ENABLE_EXP_BATCHING(ExpNegEstOp);

#undef VECOPS_VEC_ENABLE_EXP_BATCHING

#define VECOPS_VEC_DEFINE_EXP_GENERIC(OpType)                          \
  template <typename Backend, VectorTag Tag>                           \
  struct GenericImpl<Backend, OpType, Tag>                             \
      : UnaryArithmeticGenericImpl<Backend, OpType, Tag> {}

VECOPS_VEC_DEFINE_EXP_GENERIC(ExpOp);
VECOPS_VEC_DEFINE_EXP_GENERIC(ExpFastOp);
VECOPS_VEC_DEFINE_EXP_GENERIC(ExpEstOp);
VECOPS_VEC_DEFINE_EXP_GENERIC(ExpNegOp);
VECOPS_VEC_DEFINE_EXP_GENERIC(ExpNegFastOp);
VECOPS_VEC_DEFINE_EXP_GENERIC(ExpNegEstOp);

#undef VECOPS_VEC_DEFINE_EXP_GENERIC

} // namespace vecops::vec::details

#endif // VECOPS_VEC_DETAILS_MATH_H
