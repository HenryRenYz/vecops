//
// Created by renyz on 2026/6/10.
//

#ifndef VECOPS_KERNELBASE_H
#define VECOPS_KERNELBASE_H

#include "vecops/CoreTypes.h"
#include "./Layout.h"

namespace vecops::gemm {

/**
 * 微内核种类，表示实现内核的硬件种类，比如SME, AMX这种。实际类型，不是模板，因此没有参数。
 * 包含属于此种类代表的加速器的硬件特征和状态管理函数。
 */
struct KernelKind {
  /**
   * 获取/释放当前Atom对应的内核所需的硬件资源。上层调用属于此Atom的内核之前必须调用
   * acquire()，之后必须释放release()，否则执行行为未定义。
   *
   * 比如AMX的标志位设置、SME进入streaming mode开启ZA寄存器。
   */
  static void acquire();
  static void release();
};

/**
 * 定义单条硬件指令的固定参数和打包格式，所有同种类的 Kernel 共享同一个 Atom。包含Arch+Dtype标签的
 * 命名方式，直接定义了底层实现和累加器类型。如AMX_BF16F32, AMX_I8U8I32, SME_F16F32, SME_F64F64
 * 这种命名方式。实际类型，不是模板，因此没有参数。
 */
struct Atom {
  using KernelKind = gemm::KernelKind;
  /**
   * 硬件累加器M N分块大小，可能是常量或者运行时量（如SME）
   */
  constexpr static auto M_R = cint<16>;
  constexpr static auto N_R = cint<16>;
  /**
   * 单条指令 K 轴长度 (SME: 1, AMX BF16: 32, AMX INT8: 64)。
   */
  constexpr static auto K_R = cint<16>;

  /**
   * 内核期望的A, B, C元素类型。如果内核传入的实际A, B, C类型和此保持一致则不需要类型转换。
   */
  using TA = float32_t;
  using TB = float32_t;
  using TC = float32_t;

  /**
   * 累加器类型。一般和TC保持一致，但不绝对。
   */
  using TAcc = float32_t;

  /**
   * 打包布局为ALayout的A后的打包布局。
   * 输入类型ALayout: 必须为Shape<xM, xK>, Stride<*, *>形式，其中xM xK任意。
   * 返回布局为Shape<ceil(xM/M_R), ceil(xK/K_R), ...>, Stride<Dynamic<64> | Int<*>, Dynamic<64> | Int<*>>的特化。
   * 注：可能并不总是连续的，在一些输入下为了防止cache thrashing可能会特意留意些空隙。
   */
  template <typename TLayout, std::enable_if_t<is_layout<TLayout>, bool> = true>
  using APackedLayout = TLayout;

  /**
   * ALayout是否为打包布局，即判断其是否为APackedLayout<ALayout>的宽松类型：
   * Shape<*, *, ...>, Stride<Dynamic<64> | Int<*>, Dynamic<64> | Int<*>>。
   */
  template <typename TLayout, std::enable_if_t<is_layout<TLayout>, bool> = true>
  static constexpr bool is_A_packed_layout = true;

  template <
      typename TSrcA,
      typename SrcALayout,
      typename SrcAPackedLayout,
      typename Prologue>
  void pack_A(const TSrcA *src, SrcALayout src_layout, TA *dst, SrcAPackedLayout dst_layout, const Prologue &prologue);


  template <typename TLayout, std::enable_if_t<is_layout<TLayout>, bool> = true>
  using BPackedLayout = TLayout;

  template <typename TLayout, std::enable_if_t<is_layout<TLayout>, bool> = true>
  static constexpr bool is_B_packed_layout = true;

  template <
      typename TSrcB,
      typename SrcBLayout,
      typename SrcBPackedLayout,
      typename Prologue>
  void pack_B(const TSrcB *src, SrcBLayout src_layout, TA *dst, SrcBPackedLayout dst_layout, const Prologue &prologue);

  /**
   * 用于不打包时 Kernel 内部的向量加载优化路径选择
   *   - 0: 偏好第一维（M 或 N）连续 → column-major
   *   - 1: 偏好第二维（K）连续 → row-major
   */
  static constexpr int A_prefers_contiguous_dim = 1;
  static constexpr int B_prefers_contiguous_dim = 1;

};

enum MaskMode {
  Masked, Unmasked
};

/**
 * 一个K累加循环的实现。模板类型，但每个特化需要手动实现。
 * @tparam Atom Kernel的Atom
 * @tparam nM_R, nN_R 一个内核分块M N方向各包含几个累加器分块。大于0
 * @tparam M_mask, N_mask 定义M N方向是否开启运行时mask
 */
template <
    typename Atom_,
    int nM_R_, int nN_R_,
    MaskMode M_mask_, MaskMode N_mask_>
struct Kernel {
  using Atom = Atom_;
  static constexpr int nM_R = nM_R_, nN_R = nN_R_;
  static constexpr MaskMode M_mask = M_mask_, N_mask = N_mask_;
  static constexpr int compute_power = 0;

};


} // namespace vecops::gemm

#endif //VECOPS_KERNELBASE_H
