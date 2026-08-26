//
// Copyright (c) vecops contributors.
//

#ifndef VECOPS_KERNEL_DETAILS_MATMUL_TYPES_H
#define VECOPS_KERNEL_DETAILS_MATMUL_TYPES_H

namespace vecops::kernel::matmul_implementation {

/** Select the Intel AMX tile backend. */
struct AMX {};

/** Select the Arm SME ZA backend. */
struct SME {};

} // namespace vecops::kernel::matmul_implementation

#endif // VECOPS_KERNEL_DETAILS_MATMUL_TYPES_H
