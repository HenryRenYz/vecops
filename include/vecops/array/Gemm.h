//
// Created by renyz on 2026/5/9.
//

#ifndef VECOPS_GEMM_H
#define VECOPS_GEMM_H

#include <type_traits>

#include "vecops/CoreDefs.h"

namespace vecops::array {

struct ValueDef {
  static constexpr bool is_const = false;
  static constexpr bool is_runtime = false;
  static constexpr bool conforms(int v) { return false; }
};

template <int N>
struct Int : public ValueDef {
  static constexpr bool is_const = true;
  static constexpr bool is_runtime = false;
  static constexpr int value = N;
  static constexpr bool conforms(int v) { return v == N; }

  explicit Int(int _ = N) {}
};

template <int Alignment>
struct Aligned : public ValueDef {
  static_assert((Alignment & (Alignment - 1)) == 0 && Alignment > 0, "Alignment must be positive and power of 2");
  static constexpr bool is_const = false;
  static constexpr bool is_runtime = true;
  static constexpr int alignment = Alignment;
  static constexpr bool conforms(int v) { return (v & (alignment - 1)) == 0; }

  explicit Aligned(int value) : value(value) { }

  const int value;
};

using Any = Aligned<1>;

namespace details {
template <typename T>
struct ValueDefPromote { using Type = T; };
template <>
struct ValueDefPromote<int> { using Type = Any; };
} // namespace details

template <typename T>
using ToValueDef = details::ValueDefPromote<T>::Type;

namespace details {

template <typename... Is>
struct IndexSetter;

template <>
struct IndexSetter<> {
  void operator()(const int * in, int * out) { }
};

template <typename I, typename... Is>
struct IndexSetter<I, Is...> {
  void operator()(const int * in, int * out) {
    if constexpr (I::is_const) {
      IndexSetter<Is...>{}(in + 1, out);
    } else {
      out[0] = in[0];
      IndexSetter<Is...>{}(in + 1, out + 1);
    }
  }
};

template <int N, typename... Is>
struct IndexGetter;

template <typename I, typename... Is>
struct IndexGetter<0, I, Is...> {
  constexpr int operator()(const int * in) {
    if constexpr (I::is_const) {
      return I::value;
    } else {
      return in[0];
    }
  }
};

template <int N, typename I, typename... Is>
struct IndexGetter<N, I, Is...> {
  static_assert(N > 0, "Negative index");
  constexpr int operator()(const int * in) {
    if constexpr (I::is_const) {
      return IndexGetter<N - 1, Is...>{}(in);
    } else {
      return IndexGetter<N - 1, I, Is...>{}(in + 1);
    }
  }
};

} // namespace details


template <typename... Is>
struct MatrixMeta {
  static_assert((std::is_base_of_v<ValueDef, Is> && ...), "Is is not base of ValueDef");
  static_assert(sizeof...(Is) > 0, "ndim cannot be 0");
  static constexpr int Ndim = sizeof...(Is);

  template <typename... Ints>
  MatrixMeta(Ints... is) {
    static_assert((std::is_same_v<Ints, int> && ...), "Ints must be int");
    VECOPS_ASSERT((Is::conforms(is) && ...), "`is` does not conform the constraint of Is");
    int is_arr[sizeof...(is)] = {is...};
    details::IndexSetter<Is...>()(is_arr, _is);
  }

  template <int N>
  constexpr int get() const {
    return details::IndexGetter<N, Is...>{}(_is);
  }

  constexpr int ndim() const {
    return Ndim;
  }
protected:
  int _is[(int(Is::is_runtime) + ...)];
};

template <typename... Is>
struct Shape : public MatrixMeta<Is...> {
  template <typename... Ints>
  Shape(Ints... is) : MatrixMeta<Is...>(is...) {
    VECOPS_ASSERT(((is >= 0) && ...), "is must be non-negative");
  }
};

template <typename... Is>
struct Stride : public MatrixMeta<Is...> {
  template <typename... Ints>
  Stride(Ints... is) : MatrixMeta<Is...>(is...) {
  }
};

namespace details {
template <typename T>
struct ShapeChecker : public std::false_type {};
template <typename... Is>
struct ShapeChecker<Shape<Is...>> : public std::true_type {};

template <typename T>
struct StrideChecker : public std::false_type {};
template <typename... Is>
struct StrideChecker<Stride<Is...>> : public std::true_type {};
} // namespace details

template <typename T>
static constexpr bool is_shape = details::ShapeChecker<T>::value;

template <typename T>
static constexpr bool is_stride = details::StrideChecker<T>::value;

template <typename TShape, typename TStride>
struct Layout {
  static_assert(is_shape<TShape>, "TShape must be Shape<_,_>");
  static_assert(is_stride<TStride>, "TStride must be Stride<_,_>");
  static_assert(TShape::Ndim == TStride::Ndim, "TShape and TStride must have the same rank");

  using Shape = TShape;
  using Stride = TStride;

  Layout(TShape shape, TStride stride) : _shape(shape), _stride(stride) {
  }

  const TShape& shape() const {
    return _shape;
  }

  const TStride& stride() const {
    return _stride;
  }

private:
  TShape _shape;
  TStride _stride;
};

namespace details {
template <typename T>
struct LayoutChecker : std::false_type {};
template <typename TShape, typename TStride>
struct LayoutChecker<Layout<TShape, TStride>> : std::true_type {};
} // namespace details

template <typename T>
static constexpr bool is_layout = details::LayoutChecker<T>::value;

template <typename... IsOrInts>
static constexpr Shape<ToValueDef<IsOrInts>...> make_shape(IsOrInts... vs) {
  return Shape<ToValueDef<IsOrInts>...>{ToValueDef<IsOrInts>{vs}.value...};
}

template <typename... IsOrInts>
static constexpr Stride<ToValueDef<IsOrInts>...> make_stride(IsOrInts... vs) {
  return Stride<ToValueDef<IsOrInts>...>{ToValueDef<IsOrInts>{vs}.value...};
}

template <typename TShape, typename TStride>
static constexpr Layout<TShape, TStride> make_layout(const TShape& shape, const TStride& stride) {
  return Layout<TShape, TStride>{shape, stride};
}

template <typename ALayout, typename BLayout, typename CLayout, typename EpilogFn>
struct Kernel {
  using TA = void;
  using TB = void;
  using TC = void;

  static constexpr int Mtile = 0;
  static constexpr int Ntile = 0;
  static constexpr int Ktile = 0;

  /**
   * @param A array with layout ALayout
   * @param B array with layout BLayout
   * @param C array with layout CLayout
   * @param fn epilog fn
   */
  void run(
      const TA * A, ALayout A_layout,
      const TB * B, BLayout B_layout,
      TC * C, CLayout C_layout,
      const EpilogFn& fn
  );
};

template <
    template <typename KALayout, typename KBLayout, typename KCLayout, typename KEpilogFn>
    typename Kernel,
    typename ProblemShape,
    typename TA,
    typename ALayout,
    typename TB,
    typename BLayout,
    typename TC,
    typename CLayout,
    typename EpilogFn
>
void gemm(
    ProblemShape shape_MNK,
    const TA * A, ALayout A_layout,
    const TB * B, BLayout B_layout,
    TC * C, CLayout C_layout,
    const EpilogFn& fn
) {

}

}

#endif //VECOPS_GEMM_H
