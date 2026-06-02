//
// Created by renyz on 2026/6/2.
//

#ifndef VECOPS_LAYOUT_H
#define VECOPS_LAYOUT_H

#include <array>

#include "vecops/CoreTypes.h"
#include "vecops/Assertion.h"
#include "vecops/util/TypeTraits.h"

namespace vecops::array {

/**
 * 用于表示一个张量元数据中的值，比如Shape或者Stride中的一个值。
 * 使用类型而不是运行时nint_t是用于存储编译期常量、对齐信息和约束，用于编译期优化。
 * 对于张量元数据而言，所有数值均为常量，一旦张量初始化其元数据则不再变化。
 */
struct Value {
  /**
   * 表示此值是否为编译期常量。
   */
  static constexpr bool is_const = false;
  /**
   * 表示此值是否为运行时常量。
   */
  static constexpr bool is_runtime = false;
  /**
   * 对于传入值v，其是否符合当前Value定义的约束（比如是否对齐等等）
   */
  static constexpr bool conforms(nint_t v) { return false; }
  /**
   * 检查此Value定义是否一定符合对齐到v的要求
   * 此函数返回false不代表Value携带的值一定不符合对齐要求（比如实际的运行时量可能是对齐的）
   */
  static constexpr bool aligns(nint_t v) { return false; }
  /**
   * 检查此Value携带的值是否符合对齐到v的要求
   * 可能会出现aligns(v) == false但this->is_aligned(v) == true的情况：Any{4}.is_aligned(4)
   */
  bool is_aligned(nint_t v) { return false; }
};

/**
 * 表示编译器常量值，其值必定为N
 */
template <nint_t N>
struct Const : public Value {
  static constexpr bool is_const = true;
  static constexpr bool is_runtime = false;
  static constexpr nint_t value = N;
  static constexpr bool conforms(nint_t v) { return v == N; }
  static constexpr bool aligns(nint_t v) { return N % v == 0; }

  explicit Const(nint_t v = N) {
    VECOPS_ASSERT(v == N, "%zd != %zd", v, N);
  }

  constexpr bool is_aligned(nint_t v) { return N % v == 0; }
};

/**
 * constexpr field template用于方便初始化
 */
template <nint_t N>
static constexpr Const<N> cint {N};

/**
 * 表示满足对齐到Alignment的运行时量。
 * @tparam Alignment 对齐量，必须是2的整数幂且不能是0
 */
template <nint_t Alignment>
struct Aligned : public Value {
  static_assert((Alignment & (Alignment - 1)) == 0 && Alignment > 0, "Alignment must be positive and power of 2");
  static constexpr bool is_const = false;
  static constexpr bool is_runtime = true;
  static constexpr int alignment = Alignment;
  static constexpr bool conforms(int v) { return (v & (alignment - 1)) == 0; }
  static constexpr bool aligns(nint_t v) { return Alignment % v == 0; }

  explicit Aligned(nint_t value) : value(value) {
    VECOPS_ASSERT((value & (Alignment - 1)) == 0, "%zd not aligned to %zd", value, Alignment);
  }

  constexpr bool is_aligned(nint_t v) { return value % v == 0; }

  const nint_t value;
};

/**
 * 表示任意运行时量（无约束）。
 */
using Any = Aligned<1>;

namespace details {

/**
 * 将用户参数输入中的整数v转化为Any{v}
 * @tparam T
 */
template <typename T, typename = void/*SFINAE*/>
struct ValuePromote { using Type = T; };
template <typename T>
struct ValuePromote<T, std::enable_if_t<is_int<T>>> { using Type = Any; };

/**
 * 生成一个将Is下标映射到压缩数组下标的映射表
 * @return (映射表数组, 压缩数组长度)
 */
template <typename... Is>
consteval std::pair<std::array<int, sizeof...(Is)>, int> make_storage_offset() {
  constexpr bool flags[] = {Is::is_runtime...};
  std::array<int, sizeof...(Is)> arr;

  int off = 0;
  for (int i = 0; i < sizeof...(Is); ++i) {
    arr[i] = off;
    off += int(flags[i]);
  }
  return std::make_pair(arr, off);
}

} // namespace details

template <typename T>
using ToValue = details::ValuePromote<T>::Type;

template <typename... Is>
struct ArrayMeta {
  static_assert((std::is_base_of_v<Value, Is> && ...), "Is is not Value");
  static_assert(sizeof...(Is) > 0, "ndim cannot be 0");
  static constexpr int Ndim = sizeof...(Is);

};


} // namespace vecops

#endif //VECOPS_LAYOUT_H
