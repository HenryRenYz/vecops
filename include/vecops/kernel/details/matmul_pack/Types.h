//
// Copyright (c) vecops contributors.
//

#ifndef VECOPS_KERNEL_DETAILS_MATMUL_PACK_TYPES_H
#define VECOPS_KERNEL_DETAILS_MATMUL_PACK_TYPES_H

namespace vecops::kernel::matmul_pack_implementation {

struct Vector {};
struct SME {};
struct SMEPostprocess {};
struct SMEStagedTransform {};
struct SMEStagedFP16ToFP32 {};

} // namespace vecops::kernel::matmul_pack_implementation

#endif // VECOPS_KERNEL_DETAILS_MATMUL_PACK_TYPES_H
