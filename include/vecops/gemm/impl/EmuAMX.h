//
// Created by renyz on 2026/6/10.
//

#ifndef VECOPS_EMUAMX_H
#define VECOPS_EMUAMX_H

#include "vecops/CoreTypes.h"
#include "vecops/vec/Vec.h"
#include "vecops/util/Math.h"
#include "vecops/Meta.h"

namespace vecops::gemm::EmuAMX {

struct KernelKind {
  static void acquire() {}
  static void release() {}
};

namespace details {

/**
 * Block layout
 */
template <typename TLayout, int dtype_size>
consteval auto infer_A_packed_layout() {
  constexpr auto M_R = meta::cint<16>;
  constexpr auto K_R = meta::cint<64 / dtype_size>;

  auto layout = std::declval<TLayout>();
  static_assert(layout.ndim() == 2, "Ndim mismatch");
  auto M = size<0>(layout);
  auto K = size<1>(layout);
  auto nM_R = cdiv(M, M_R);
  auto nK_R = cdiv(K, K_R);
  return make_layout(make_shape(nM_R, nK_R, M_R, K_R));
}

template <typename TLayout, int dtype_size>
consteval bool is_A_packed_layout() {
  constexpr auto M_R = meta::cint<16>;
  constexpr auto K_R = meta::cint<64 / dtype_size>;

  auto layout = std::declval<TLayout>();
  // TLayout == shape (*, *, cint<M_R>, cint<K_R>), stride (*, *, cint<K_R>, cint<1>)
  if (layout.ndim() != 4) return false;
  if (is_runtime<2>(layout.shape()) || size<2>(layout) != M_R) return false;
  if (is_runtime<3>(layout.shape()) || size<3>(layout) != K_R) return false;
  if (is_runtime<2>(layout.strides()) || stride<2>(layout) != K_R) return false;
  if (is_runtime<3>(layout.strides()) || stride<3>(layout) != 1) return false;
  return true;
}

template <typename TInArray, typename TOutArray, typename Prologue>
VECOPS_INLINE void pack_A(const TInArray &src, TOutArray &dst, const Prologue &prologue) {
  static_assert(is_tensor<TInArray> && is_tensor<TOutArray>);
  using EIn = TInArray::ElementType;
  using EOut = TOutArray::ElementType;
  constexpr auto M_R = meta::cint<16>;
  constexpr auto K_R = meta::cint<64 / sizeof(EOut)>;

  auto in_p = src.data();
  auto out_p = dst.data();
  auto in_layout = src.layout();
  auto out_layout = dst.layout();

  static_assert(in_layout.ndim() == 2);
  static_assert(is_A_packed_layout<decltype(out_layout), sizeof(EOut)>());

  auto M = size<0>(in_layout);
  auto K = size<1>(in_layout);
  auto nM_R = size<0>(out_layout);
  auto nK_R = size<1>(out_layout);

  VECOPS_ASSERT(nM_R == cdiv(M, M_R), "M dim size mismatch: expected %zd, got %zd", cdiv(M, M_R), nM_R);
  VECOPS_ASSERT(nK_R == cdiv(K, K_R), "K dim size mismatch: expected %zd, got %zd", cdiv(K, K_R), nK_R);

  nint_t i;
  for (nint_t i = 0; i < M; i += nM_R) {
    for (nint_t k = 0; k < K; k += nK_R) {

    }
  }

}

/**
 * Blocked VNNI layout
 */
template <typename TLayout, int dtype_size>
consteval auto infer_B_packed_layout() {
  constexpr auto N_R = meta::cint<16>;
  constexpr auto K_R = meta::cint<64 / dtype_size>;
  constexpr auto K_P = meta::cint<4 / dtype_size>;
  static_assert(K_R % K_P == 0);

  auto layout = std::declval<TLayout>();
  static_assert(layout.ndim() == 2, "Ndim mismatch");
  auto N = size<0>(layout);
  auto K = size<1>(layout);
  auto nN_R = cdiv(N, N_R);
  auto nK_R = cdiv(K, K_R);
  return make_layout(make_shape(nN_R, nK_R, K_R / K_P, N_R, K_P));
}

template <typename TLayout, int dtype_size>
consteval bool is_B_packed_layout() {
  constexpr auto N_R = meta::cint<16>;
  constexpr auto K_R = meta::cint<64 / dtype_size>;
  constexpr auto K_P = meta::cint<4 / dtype_size>;
  static_assert(K_R % K_P == 0);

  auto layout = std::declval<TLayout>();
  // TLayout == shape (*, *, cint<K_R/K_P>, cint<N_R>, cint<K_P>), stride (*, *, cint<N_R * K_P>, cint<K_P>, cint<1>)
  if (layout.ndim() != 5) return false;
  if (is_runtime<2>(layout.shape()) || size<2>(layout) != K_R / K_P) return false;
  if (is_runtime<3>(layout.shape()) || size<3>(layout) != N_R) return false;
  if (is_runtime<4>(layout.shape()) || size<4>(layout) != K_P) return false;
  if (is_runtime<2>(layout.strides()) || stride<2>(layout) != K_P * N_R) return false;
  if (is_runtime<3>(layout.strides()) || stride<3>(layout) != K_P) return false;
  if (is_runtime<4>(layout.strides()) || stride<4>(layout) != 1) return false;
  return true;
}

} // namespace details

struct AtomBF16BF16F32 {
  using KernelKind = KernelKind;
  static constexpr auto M_R = meta::cint<16>;
  static constexpr auto N_R = meta::cint<16>;
  static constexpr auto K_R = meta::cint<32>;
  using TA = bfloat16_t;
  using TB = bfloat16_t;
  using TC = float32_t;
  using TAcc = float32_t;

  template <typename TLayout>
  using APackedLayout = decltype(details::infer_A_packed_layout<TLayout, sizeof(TA)>());

  template <typename TLayout>
  static constexpr bool is_A_packed_layout = details::is_A_packed_layout<TLayout, sizeof(TA)>();

  template <
      typename TSrcA, typename SrcALayout, typename SrcAPackedLayout, typename Prologue>
  void pack_A(const TSrcA *src, SrcALayout src_layout, TA *dst, SrcAPackedLayout dst_layout, const Prologue &prologue);

  template <typename TLayout>
  using BPackedLayout = decltype(details::infer_B_packed_layout<TLayout, sizeof(TB)>());

  template <typename TLayout>
  static constexpr bool is_B_packed_layout = details::is_B_packed_layout<TLayout, sizeof(TB)>();

  template <
      typename TSrcB, typename SrcBLayout, typename SrcBPackedLayout, typename Prologue>
  void pack_B(const TSrcB *src, SrcBLayout src_layout, TB *dst, SrcBPackedLayout dst_layout, const Prologue &prologue);


};

} // namespace vecops::gemm::EmuAMX

#endif //VECOPS_EMUAMX_H
