//
// Copyright (c) vecops contributors.
//

#ifndef VECOPS_MATMUL_DETAILS_PACK_TYPES_H
#define VECOPS_MATMUL_DETAILS_PACK_TYPES_H

namespace vecops::kernel::matmul_pack_implementation {

struct Vector {};
struct SME {};
struct SMEPostprocess {};
struct SMEStagedTransform {};
struct SMEStagedFP16ToFP32 {};
struct SMEFP32ToFP64 {};
struct SMEFP32ToFP64Single {};

} // namespace vecops::kernel::matmul_pack_implementation

#endif // VECOPS_MATMUL_DETAILS_PACK_TYPES_H
