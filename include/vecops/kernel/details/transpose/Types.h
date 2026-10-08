// SPDX-FileCopyrightText: 2026 HenryRenYz and vecops contributors
// SPDX-License-Identifier: MIT
//
//

#ifndef VECOPS_KERNEL_DETAILS_TRANSPOSE_TYPES_H
#define VECOPS_KERNEL_DETAILS_TRANSPOSE_TYPES_H

/**
 * @file Types.h
 * @brief Architecture-neutral policy and implementation tags for Transpose2D.
 *
 * Tags live separately so the public entry, generic algorithm, operation-level
 * selection, and platform backend can share them without cyclic includes.
 */

namespace vecops::kernel {

namespace transpose2d_policy {

/** Select the current transpose backend's normal register kernel. */
struct Automatic {};

/**
 * Force the generic gather/contiguous-store implementation.
 * This policy never selects SME and is useful for non-direct DataAccess views.
 */
struct Gather {};

} // namespace transpose2d_policy

namespace transpose2d_implementation {

/** Vector implementation selected by the current transpose backend. */
struct Vector {};

/** SME/ZA implementation selected independently of the vector backend. */
struct SME {};

} // namespace transpose2d_implementation

} // namespace vecops::kernel

#endif // VECOPS_KERNEL_DETAILS_TRANSPOSE_TYPES_H
