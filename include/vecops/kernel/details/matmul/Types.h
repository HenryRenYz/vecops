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

namespace vecops::kernel::matmul_policy {

/**
 * Keep the architecture's benchmark-tuned traversal. Passing one of
 * kernel::loop::tile2d_policy's policy types to the ops API instead exposes
 * tile2d's corresponding traversal directly.
 */
struct Automatic {};

} // namespace vecops::kernel::matmul_policy

#endif // VECOPS_KERNEL_DETAILS_MATMUL_TYPES_H
