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
   * 可能不是constexpr
   */
  static constexpr auto M_R = cint<16>;
  static constexpr auto N_R = cint<16>;
  /**
   * 单条指令 K 轴长度 (SME: 1, AMX BF16: 32, AMX INT8: 64)。
   */
  static constexpr auto K_R = cint<16>;

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
      typename TSrcA, typename SrcALayout, typename SrcAPackedLayout, typename Prologue>
  void pack_A(const TSrcA *src, SrcALayout src_layout, TA *dst, SrcAPackedLayout dst_layout, const Prologue &prologue);


  template <typename TLayout, std::enable_if_t<is_layout<TLayout>, bool> = true>
  using BPackedLayout = TLayout;

  template <typename TLayout, std::enable_if_t<is_layout<TLayout>, bool> = true>
  static constexpr bool is_B_packed_layout = true;

  template <
      typename TSrcB, typename SrcBLayout, typename SrcBPackedLayout, typename Prologue>
  void pack_B(const TSrcB *src, SrcBLayout src_layout, TB *dst, SrcBPackedLayout dst_layout, const Prologue &prologue);

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
  static constexpr auto nM_R = cint<nM_R_>;
  static constexpr auto nN_R = cint<nN_R_>;
  static constexpr MaskMode M_mask = M_mask_;
  static constexpr MaskMode N_mask = N_mask_;
  /**
   * 一次内核调用处理的元素数。
   */
  static constexpr auto tile_M = nM_R * Atom::M_R;
  static constexpr auto tile_N = nN_R * Atom::N_R;
  /**
   * 算力参考值，手动设定，用于组合搜索。
   */
  static constexpr int compute_power = 0;

  /**
   * 执行K累加循环核心函数。
   * @tparam TSrcA, TSrcB, TSrcDstC A B C类型。
   * @tparam SrcALayout 输入A的布局，必须为非打包布局Shape<xM, xK>, Stride<*, *>或者打包布局，
   *                    即满足Atom::is_A_packed_layout<SrcALayout>。其中xM, xN当前分块大小，
   *                    不大于内核分块大小M_R, N_R，xK为K轴累加长度，任意，下同。
   * @tparam APrologue 用于非打包A的前处理，A打包时必须为identity。一般使用identity。
   * @tparam SrcBLayout 输入B的布局，必须为非打包布局Shape<xN, xK>, Stride<*, *>或者打包布局，
   *                    即满足Atom::is_B_packed_layout<SrcBLayout>。
   * @tparam BPrologue 用于非打包B的前处理，B打包时必须为identity。一般使用identity。
   * @tparam SrcDstCLayout 输入C的布局，必须为非打包布局Shape<xM, xN>, Stride<*, *>。
   * @tparam CPrologue 用于非ld_acc时的C的前处理，一般使用zeros（零初始化）。
   * @tparam LdAccFlag bool或者std::bool_constant
   * @tparam StAccFlag bool或者std::bool_constant
   * @tparam reuse_buffer 是否重用C作为K分块缓冲区。如果开启，则必须满足：
   *                      - sizeof(Atom::TAcc)<=sizeof(TC)，即硬件累加器精度不能高于C精度。
   *                      - SrcDstCLayout.stride[1]==Int<1>，即缓冲区必须是行主序的，保证读写效率。
   *                      - 注：由于reuse_buffer保证C必须是行主序的，且C元素不不比硬件累加器精度低，因此C作为分块缓冲区时我们会直接将加载/存储的C块的每一行看成Atom::TAcc类型，从而省掉作为缓冲区加载存储时的数据转换以及节省潜在带宽。
   * @param acc K分块缓冲区，可空，但(ld_acc||st_acc) && !reuse_buffer为真是必须非空。
   *            布局必须为Shape<nM_R, Atom::M_R, nN_R, Atom::N_R>, Stride<Int<Atom::M_R>, *, Int<Atom::N_R>, Int<1>>。
   * @param ld_acc 真+reuse_buffer假：从缓冲区中unmasked加载数据到硬件累加器。
   *               真+reuse_buffer真：从C中masked加载数据到硬件累加器
   *               假+CPrologue==zeros：直接硬件0初始化硬件累加器。
   *               假+CPrologue!=zeros：从C中masked加载数据，经过CPrologue变换后加载到硬件累加器。
   * @param st_acc 真+reuse_buffer假：硬件累加器数据unmasked写入acc。
   *               真+reuse_buffer真：硬件累加器数据masked写入C。
   *               假：硬件累加器数据经Epilogue变换后masked写入C。
   * @param epilogue
   *
   * @note 关于mask相关的约束和处理：
   *       - 在A/B打包时，不管对应M/N如何设置，都直接按满数据（Atom::M_R/Atom::N_R）加载。
   *       - 在A非打包时，形状参数为Shape<xM, xK>，则M_mask关闭时需满足xM==Atom::M_R（assert检查）。
   *         M_mask开启时则加载A数据时按照xM元素数量应用mask，其余元素设为0。
   *       - 在B非打包时，形状参数为Shape<xN, xK>，则N_mask关闭时需满足xN==Atom::N_R（assert检查）。
   *         N_mask开启时则加载B数据时按照xN元素数量应用mask，其余元素设为0。
   */
  template <
      typename TSrcA, typename SrcALayout, typename APrologue,
      typename TSrcB, typename SrcBLayout, typename BPrologue,
      typename TSrcDstC, typename SrcDstCLayout, typename CPrologue,
      typename LdAccFlag, typename StAccFlag,
      typename Epilogue, bool reuse_buffer>
  void run(
      const TSrcA *srcA, SrcALayout srcA_layout, const APrologue &srcA_prologue,
      const TSrcB *srcB, SrcBLayout srcB_layout, const BPrologue &srcB_prologue,
      const TSrcDstC *C, SrcDstCLayout C_layout, const CPrologue &C_prologue,
      typename Atom::TAcc *acc, LdAccFlag ld_acc, StAccFlag st_acc, const Epilogue &epilogue) const;

  /**
   * 获取/释放运行此kernel所需的硬件资源，进入kernel前必须调用begin_kernel()，退出kernel后必须调用end_kernel()，
   * 否则执行结果未定义。
   *
   * 多次调用同一种内核之间不需要调用end_kernel+begin_kernel。
   *   - 比如AMX的LDTILECFG和设置tile形状。
   */
  void begin_kernel() const;
  void end_kernel() const;
};


} // namespace vecops::gemm

#endif //VECOPS_KERNELBASE_H
