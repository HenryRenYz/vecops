//
// Created by renyz on 2026/6/2.
//

#ifndef VECOPS_LAYOUT_H
#define VECOPS_LAYOUT_H

#include "vecops/CoreTypes.h"
#include "vecops/Assertion.h"

namespace vecops::array {

/**
 * 用于表示一个张量元数据中的值，比如Shape或者Stride中的一个值。
 * 使用类型而不是运行时nint_t是用于存储编译期常量和对齐信息，用于编译期优化。
 * 对于张量元数据而言，所有数值均为常量，一旦张量初始化
 */
struct Value {
  /**
   * 表示此值是否为编译期常量。
   */
  static constexpr bool is_const = false;
  /**
   *
   */
  static constexpr bool is_runtime = false;
  static constexpr bool conforms(nint_t v) { return false; }
  static constexpr bool aligns(nint_t v) { return false; }
};

template <nint_t N>
struct Const : public Value {
  static constexpr bool is_const = true;
  static constexpr bool is_runtime = false;
  static constexpr nint_t value = N;
  static constexpr bool conforms(nint_t v) { return v == N; }

  explicit Const(nint_t v = N) {
    VECOPS_ASSERT(v == N, "%zd != %zd", v, N);
  }
};

template <nint_t N>
static constexpr Const<N> cint {N};

template <nint_t Alignment>
struct Aligned : public Value {
  static_assert((Alignment & (Alignment - 1)) == 0 && Alignment > 0, "Alignment must be positive and power of 2");
  static constexpr bool is_const = false;
  static constexpr bool is_runtime = true;
  static constexpr int alignment = Alignment;
  static constexpr bool conforms(int v) { return (v & (alignment - 1)) == 0; }

  explicit Aligned(int value) : value(value) { }

  const int value;
};

using Any = Aligned<1>;



} // namespace vecops

#endif //VECOPS_LAYOUT_H
