/*
 * The fixed 16x64 BF16 schedule in this provider follows KUPL's
 * KP36_16x64x2_BF16BF16F32 MMA primitive.
 *
 * Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
 * Licensed under the Mulan PSL v2.
 */

#ifndef VECOPS_MATMUL_DETAILS_KERNEL_SME_KUPL_MMA_H
#define VECOPS_MATMUL_DETAILS_KERNEL_SME_KUPL_MMA_H

#include "vecops/CoreTypes.h"
#include "vecops/kernel/Tile2D.h"

namespace vecops::kernel::matmul_details::sme::kupl_mma {

/** KUPL KP36 BF16 covers one fp32 ZA row by four ZA columns. */
inline constexpr int MBlocks = 1;
inline constexpr int NBlocks = 4;

/**
 * Explicit Tile2D provider used only by a KUPL-enabled build.
 *
 * The 1x4 family wins every full-width BF16 region. Smaller native families
 * remain available solely for dimensions that a fixed KUPL tile cannot cover.
 * There is no runtime or shape-based provider selection.
 */
struct KernelProvider {
  static constexpr bool four_regions_exact_constraints = true;
  static constexpr kernel::loop::Tile2DExactGridMode exact_grid_mode =
      kernel::loop::Tile2DExactGridMode::runtime;
  static constexpr nint_t exact_meta_block_limit = nint_t{8};

  template <int A, int B, kernel::loop::Tile2DMaskMode,
            kernel::loop::Tile2DMaskMode>
  static consteval int power() {
    if constexpr (A == MBlocks && B == NBlocks) {
      return 10'000;
    } else if constexpr (A * B <= 4) {
      constexpr int imbalance = A > B ? A - B : B - A;
      return 100 * A * B + 4 * (A + B) - 8 * imbalance;
    } else {
      return -1;
    }
  }
};

using Catalog = kernel::loop::Tile2DGeneratedCatalog<
    KernelProvider, kernel::loop::Tile2DSearchSpace<4, 4, 4>>;

} // namespace vecops::kernel::matmul_details::sme::kupl_mma

#endif // VECOPS_MATMUL_DETAILS_KERNEL_SME_KUPL_MMA_H
