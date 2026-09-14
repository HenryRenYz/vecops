//
// Copyright (c) vecops contributors.
//

// KUPL's public tiled API represents K as a compile-time Shape parameter,
// while vecops supports runtime K.  The library's exported BF16 leaf already
// accepts runtime size_k; this isolated adapter is compiled with
// -fno-access-control so it can call that leaf without exposing KUPL types in
// vecops headers.  Link failures expose an incompatible KUPL build early.

#include <tuple>
#include <type_traits>
#include <utility>

#include <arm_bf16.h>
#include <kupl_mma.h>

#include "vecops/CoreTypes.h"
#include "vecops/matmul/details/kernel/sme/KuplMma.h"

namespace vecops::kernel::matmul_details::sme::kupl_mma {

void run_bf16_tile(const void* packed_a, const void* packed_b,
                   float32_t* c, nint_t padded_k) {
  static_assert(sizeof(bfloat16_t) == sizeof(::bfloat16_t));
  using namespace ::kupl::tensor;
  using AStride = Stride<Int<2>, Stride<Int<1>, Int<32>>>;
  using BStride = Stride<Stride<Int<1>, Int<128>>, Int<2>>;
  using CStride = Stride<Int<64>, Int<1>>;
  auto* a = reinterpret_cast<::bfloat16_t*>(const_cast<void*>(packed_a));
  auto* b = reinterpret_cast<::bfloat16_t*>(const_cast<void*>(packed_b));
  TiledCallFunc::call_mma<
      16, 64, AStride, BStride, CStride,
      ::bfloat16_t, ::bfloat16_t, float>(
          a, b, c, static_cast<int>(padded_k));
  TiledCallFunc::call_store<16, 64, CStride, float>(c);
}

} // namespace vecops::kernel::matmul_details::sme::kupl_mma
