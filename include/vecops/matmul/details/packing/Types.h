// SPDX-FileCopyrightText: 2026 HenryRenYz and vecops contributors
// SPDX-License-Identifier: MIT
//
//

#ifndef VECOPS_MATMUL_DETAILS_PACK_TYPES_H
#define VECOPS_MATMUL_DETAILS_PACK_TYPES_H

/**
 * @file vecops/matmul/details/packing/Types.h
 * @brief Implementation tags for the packing layer's Backend contract.
 *
 * One tag per packing implementation; packing/Backend.h maps each to a
 * Backend specialization and packing/Plan.h probes them as the selection
 * chain (most specialized first, plain Vector as the fallback).
 */

namespace vecops::kernel::matmul_pack_implementation {

/** Portable vector-loop packer; the always-eligible fallback. */
struct Vector {};
/** SME ZA-based packer: transpose through ZA tiles inside Streaming+ZA. */
struct SME {};
/** Pack through ZA, fusing the elementwise transform on the ZA read-out. */
struct SMEPostprocess {};
/** Pack through ZA, then run the operand transform in place afterwards
 *  (exits streaming to execute ordinary vector code). */
struct SMEStagedTransform {};
/** FP16 operand: pack to the FP16 panel, then expand to FP32 in place. */
struct SMEStagedFP16ToFP32 {};
/** FP32->FP64: convert through a double-buffered ZA tile pair. */
struct SMEFP32ToFP64 {};
/** FP32->FP64 with a single ZA tile (runtime-detected alternative). */
struct SMEFP32ToFP64Single {};

} // namespace vecops::kernel::matmul_pack_implementation

#endif // VECOPS_MATMUL_DETAILS_PACK_TYPES_H
