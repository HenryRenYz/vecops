//
// Created by renyz on 2026/6/2.
//

#ifndef VECOPS_LAYOUT_H
#define VECOPS_LAYOUT_H

#include <array>

#include "vecops/CoreTypes.h"
#include "vecops/Assertion.h"
#include "vecops/util/TypeTraits.h"

namespace vecops::array2 {

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

  constexpr operator nint_t() const { return -100; }
}; // struct Value

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
  constexpr operator nint_t() const { return N; }
}; // struct Const

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
  constexpr operator nint_t() const { return value; }

  const nint_t value;
}; // struct Aligned

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

template <typename T>
struct IsConst : std::false_type {};

template <nint_t N>
struct IsConst<Const<N>> : std::true_type {};

template <typename T>
struct PickConstValue { static constexpr nint_t value = 0; };

template <nint_t N>
struct PickConstValue<Const<N>> { static constexpr nint_t value = N; };

template <int N, int I, typename = void/*SFINAE*/, typename... Is>
struct SetPackedValue {};

template <int N, typename... Is>
struct SetPackedValue<N, N, void, Is...> {
};

template <int N, int I, typename Ix, typename... Is>
struct SetPackedValue<N, I, std::enable_if_t<IsConst<Ix>{}>, Ix, Is...> {
  void operator()(nint_t * values, Ix v, Is... v_rem) const {
    SetPackedValue<N, I + 1, void, Is...>{}(values, v_rem...);
  }
  void operator()(nint_t * values, const nint_t * vs) const {
    SetPackedValue<N, I + 1, void, Is...>{}(values, vs + 1);
  }
};

template <int N, int I, typename Ix, typename... Is>
struct SetPackedValue<N, I, std::enable_if_t<!IsConst<Ix>{}>, Ix, Is...> {
  void operator()(nint_t * values, Ix v, Is... v_rem) const {
    values[0] = nint_t(v);
    SetPackedValue<N, I + 1, void, Is...>{}(values + 1, v_rem...);
  }
  void operator()(nint_t * values, const nint_t * vs) const {
    values[0] = vs[0];
    SetPackedValue<N, I + 1, void, Is...>{}(values + 1, vs + 1);
  }
};

template <>

template <typename... Is>
struct PackedStorage {
  static constexpr bool is_runtime[] = {Is::is_runtime...};
  static constexpr bool n_dim = sizeof...(Is);
private:
  static constexpr nint_t const_values[] = {PickConstValue<Is>::value...};
  /**
  * 生成一个将Is下标映射到压缩数组下标的映射表
  * @return (映射表数组, 压缩数组长度)
  */
  static constexpr auto stor_data = []{
    std::array<int, sizeof...(Is)> arr;
    int off = 0;
    for (int i = 0; i < sizeof...(Is); ++i) {
      arr[i] = off;
      off += int(is_runtime[i]);
    }
    return std::make_pair(arr, off);
  }();
public:
  static constexpr std::array<int, sizeof...(Is)> offsets = stor_data.first;
  static constexpr int num_stor = stor_data.second;

  constexpr PackedStorage() = default;
  constexpr PackedStorage(Is... values) {
    VECOPS_ASSERT((Is::conforms(nint_t(values)) && ...), "values do not conform to type constraints");
    SetPackedValue<n_dim, 0, void, Is...>{}(this->values, values...);
  }
  constexpr PackedStorage(const nint_t (&values)[n_dim]) {
    auto iseq = std::make_index_sequence<n_dim>{};
    VECOPS_ASSERT((Is::conforms(nint_t(values)) && ...), "values do not conform to type constraints");
    SetPackedValue<n_dim, 0, void, Is...>{}(this->values, values...);
  }

  nint_t values[num_stor];

  /**
   * 拿到第I维的数据，如果是常量则函数返回constexpr，否则返回运行时值。
   */
  template <int I>
  [[nodiscard]] constexpr nint_t get() const {
    if constexpr (is_runtime[I]) {
      return values[offsets[I]];
    } else {
      return const_values[I];
    }
  }
}; // struct PackedStorage

} // namespace details

/**
 * 将类型转换为Value包装，比如整数会被转为Any{v}
 */
template <typename T>
using ToValue = details::ValuePromote<T>::Type;

/**
 * 多维数组元数据基类，比如Shape或者Stride
 * @tparam Is
 */
template <typename... Is>
struct ArrayMeta {
  static_assert((std::is_base_of_v<Value, Is> && ...), "Is is not Value");
  static_assert(sizeof...(Is) > 0, "ndim cannot be 0");
  static constexpr int Ndim = sizeof...(Is);

  constexpr ArrayMeta() = default;

  template <typename... Ints>
  constexpr explicit ArrayMeta(Ints... vs) {
    static_assert(sizeof...(Ints) == sizeof...(Is), "MatrixMeta: argument count mismatch");

    _stor = details::PackedStorage<Is...>{Is{vs}...};
  }

  template <int I>
  constexpr nint_t get() const {
    return _stor.template get<I>();
  }

  template <int I>
  constexpr nint_t is_const() const {
    return !this->template is_runtime<I>();
  }

  template <int I>
  constexpr nint_t is_runtime() const {
    return decltype(_stor)::is_runtime[I];
  }

  constexpr int ndim() const {
    return Ndim;
  }

protected:
  details::PackedStorage<Is...> _stor;
}; // struct ArrayMeta

/**
 * 高维数组形状参数，每一维度参数必须非负
 */
template <typename... Is>
struct Shape : public ArrayMeta<Is...> {
  template <typename... Ints>
  constexpr Shape(Ints... is) : ArrayMeta<Is...>(is...) {
    VECOPS_ASSERT(((nint_t(is) >= 0) && ...), "is must be non-negative");
  }

  constexpr Shape() = default;
}; // struct Shape

/**
 * 高维数组步长参数，每一维参数可为负数
 */
template <typename... Is>
struct Strides : public ArrayMeta<Is...> {
  template <typename... Ints>
  constexpr Strides(Ints... is) : ArrayMeta<Is...>(is...) {
  }

  constexpr Strides() = default;
}; // class Strides

namespace details {

template <typename T>
struct IsArrayMeta : std::false_type { };
template <typename... Is>
struct IsArrayMeta<ArrayMeta<Is...>> : std::true_type { };

template <typename T>
struct IsShape : std::false_type { };
template <typename... Is>
struct IsShape<Shape<Is...>> : std::true_type { };

template <typename T>
struct IsStrides : std::false_type { };
template <typename... Is>
struct IsStrides<Strides<Is...>> : std::true_type { };

template <
    template <typename... xIs> typename Meta, // 目标Meta类型，模板类型用于随后填入类型值
    int N, // 原始Is长度
    int I, // 当前下标
    typename = void, // SFINAE
    typename... InIs // 剩余未处理的类型值
> struct ArrayMetaRemoveAxis {
  static_assert(sizeof(N) == 0, "Unreachable");
};

template <
    template <typename... xIs> typename Meta,
    int N,
    int I,
    typename InI0,
    typename... InIs
> struct ArrayMetaRemoveAxis<Meta, N, I, std::enable_if_t<(I < N)>, InI0, InIs...> {
  static_assert(0 <= I && I < N, "I out of range");

  // 由于有两个variadic template args，所以需要套两层
  template <typename... OutIs> // 已处理的类型值
  struct Holder {
    // 直接从高一级转发
    using Type = typename ArrayMetaRemoveAxis<Meta, N, I + 1, void, InIs...>
        ::template Holder<OutIs..., InI0>::Type;
  };
};

template <
    template <typename... xIs> typename Meta,
    int N,
    int I,
    typename InI0,
    typename... InIs
> struct ArrayMetaRemoveAxis<Meta, N, I, std::enable_if_t<(I == N)>, InI0, InIs...> {
  // 由于有两个variadic template args，所以需要套两层
  template <typename... OutIs> // 已处理的类型值
  struct Holder {
    using Type = Meta<OutIs..., InIs...>; // 输入InI0被移除

    constexpr Type transform(const Meta<OutIs..., InI0, InIs...>& m)
  };
};

} // namespace details

/**
 * 检查类型T是否是ArrayMeta
 */
template <typename T>
static constexpr bool is_array_meta = details::IsArrayMeta<T>::value;
/**
 * 检查类型T是否是Shape
 */
template <typename T>
static constexpr bool is_shape = details::IsShape<T>::value;
/**
 * 检查类型T是否是Strides
 */
template <typename T>
static constexpr bool is_strides = details::IsStrides<T>::value;

/**
 * 获取ArrayMeta (Shape, Stride)中第I维的数据
 * 如果第I维是Const，则函数返回constexpr，否则返回运行时量
 */
template <int I, typename TMeta, std::enable_if_t<is_array_meta<TMeta>, bool> = true>
constexpr nint_t get(const TMeta& m) {
  return m.template get<I>();
}
/**
 * 获取ArrayMeta中第I维的数据是否为Const
 */
template <int I, typename TMeta, std::enable_if_t<is_array_meta<TMeta>, bool> = true>
constexpr nint_t is_const(const TMeta& m) {
  return m.template is_const<I>();
}
/**
 * 获取ArrayMeta中第I维的数据是否为运行时量
 */
template <int I, typename TMeta, std::enable_if_t<is_array_meta<TMeta>, bool> = true>
constexpr nint_t is_runtime(const TMeta& m) {
  return m.template is_runtime<I>();
}


/**
 * 一个高维数组的内存布局，包含Shape和Stride。
 * @tparam TShape
 * @tparam TStrides
 */
template <typename TShape, typename TStrides>
struct Layout {
  static_assert(is_shape<TShape>, "TShape must be Shape<_,_>");
  static_assert(is_strides<TStrides>, "TStride must be Stride<_,_>");
  static_assert(TShape::Ndim == TStrides::Ndim, "TShape and TStride must have the same rank");
  static constexpr int Ndim = TShape::Ndim;

  using Shape = TShape;
  using Stride = TStrides;

  Layout(TShape shape, TStrides stride) : _shape(shape), _stride(stride) {
  }

  const TShape& shape() const {
    return _shape;
  }

  const TStrides& stride() const {
    return _stride;
  }

  constexpr int ndim() const {
    return Ndim;
  }

private:
  TShape _shape;
  TStrides _stride;
}; // struct Layout

namespace details {

template <typename T>
struct IsLayout : std::false_type { };
template <typename TShape, typename TStride>
struct IsLayout<Layout<TShape, TStride>> : std::true_type { };

} // namespace details

/**
 * 检查类型T是否是Layout
 */
template <typename T>
static constexpr bool is_layout = details::IsLayout<T>::value;

/**
 * 获取布局layout第I维的大小（Shape数值），此值应是非负的。
 */
template <int I, typename TLayout, std::enable_if_t<is_layout<TLayout>, bool> = true>
constexpr nint_t size(const TLayout& layout) {
  return get<I>(layout.size());
}

/**
 * 获取布局layout第I维的步长（Stride数值）
 */
template <int I, typename TLayout, std::enable_if_t<is_layout<TLayout>, bool> = true>
constexpr nint_t stride(const TLayout& layout) {
  return get<I>(layout.stride());
}


} // namespace vecops::array

#endif //VECOPS_LAYOUT_H
