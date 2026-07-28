#ifndef VECOPS_NVEC_DETAILS_ARITHMETIC_H
#define VECOPS_NVEC_DETAILS_ARITHMETIC_H

#include "vecops/nvec/details/Elementwise.h"

namespace vecops::nvec::details {

template <>
struct EnableElementwiseWordBatching<AddOp> : std::true_type {};

} // namespace vecops::nvec::details

#endif // VECOPS_NVEC_DETAILS_ARITHMETIC_H
