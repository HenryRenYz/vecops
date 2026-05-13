//
// Created by renyz on 2026/5/9.
//

#ifndef VECOPS_GEMM_H
#define VECOPS_GEMM_H

#include <type_traits>
#include <algorithm>
#include <cstdlib>

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

template <typename T>
struct IntDefChecker : std::false_type {};
template <int N>
struct IntDefChecker<Int<N>> : std::true_type {};
} // namespace details

template <typename T>
using ToValueDef = details::ValueDefPromote<T>::Type;

template <typename T>
static constexpr bool is_int = details::IntDefChecker<T>::value;

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

template <int I, typename... Is>
struct IndexGetter<0, Int<I>, Is...> {
  constexpr int operator()(const int *) {
    return I;
  }
};

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

template <typename... Is>
struct ConformCheck;

template <>
struct ConformCheck<> {
  static constexpr bool check(const int *) { return true; }
};

template <typename I0, typename... Is>
struct ConformCheck<I0, Is...> {
  static constexpr bool check(const int *arr) {
    return I0::conforms(arr[0]) && ConformCheck<Is...>::check(arr + 1);
  }
};

template <
    int I,
    template<typename... xIs> typename Meta,
    typename = void/*SFINAE*/,
    typename... IsRem
>
struct MetaTakeHelper;

template <
    template<typename... xIs> typename Meta,
    typename I0,
    typename... IsRem
>
struct MetaTakeHelper<0, Meta, void, I0, IsRem...> {
  template <typename... IsProcessed>
  struct Holder {
    using Type = Meta<IsProcessed..., IsRem...>;
    static void set(const int * is, int * os, int n) {
      if constexpr (I0::is_runtime) {
        is += 1;
      }
      for (int i = 0; i < n - 1; ++i) {
        os[i] = is[i];
      }
    }
  };
};

template <
    int I,
    template<typename... xIs> typename Meta,
    typename I0,
    typename... IsRem
>
struct MetaTakeHelper<I, Meta, std::enable_if_t<(I > 0)>, I0, IsRem...> {
  template <typename... IsProcessed>
  struct Holder {
    using Next = MetaTakeHelper<I - 1, Meta, IsRem...>::template Holder<IsProcessed..., I0>;
    using Type = typename Next::Type;
    static void set(const int * is, int * os, int n) {
      if constexpr (I0::is_runtime) {
        os[0] = is[0];
        Next::set(is + 1, os + 1, n - 1);
      } else {
        Next::set(is, os, n);
      }
    }
  };
};

template <
    int I,
    template<typename... xIs> typename Meta,
    typename = void/*SFINAE*/,
    typename... IsRem
>
struct ShapeTileHelper;

template <
    template<typename... xIs> typename Meta,
    typename I0,
    typename... IsRem
>
struct ShapeTileHelper<0, Meta, void, I0, IsRem...> {
  template <typename... IsProcessed>
  struct Holder {
    // index I is a runtime value, or it is not divisible by tile size
    template <typename Ts>
    static constexpr bool has_tail = !I0::is_const || !is_int<Ts> || (I0::value % Ts::value != 0);
    template <typename Ts>
    using FullType = Meta<IsProcessed..., std::conditional_t<is_int<Ts>, Ts, Any>, IsRem...>;
    template <typename Ts>
    using TailType = Meta<IsProcessed..., std::conditional_t<I0::is_const && is_int<Ts>, Int<I0::value % Ts::value>, Any>, IsRem...>;
    template <typename Ts>
    static void set_full(const int * is, int * os, int n, Ts ts) {
      if constexpr (I0::is_runtime) {
        is += 1;
      }
      if constexpr (!is_int<Ts>) {
        os[0] = ToValueDef<Ts>{ts}.value;
        ++os;
      }
      for (int i = 0; i < n - 1; ++i) {
        os[i] = is[i];
      }
    }
    template <typename Ts>
    static void set_tail(const int * is, int * os, int n, Ts ts) {
      if constexpr (I0::is_runtime) {
        os[0] = is[0] % ToValueDef<Ts>{ts}.value;
        os += 1;
        is += 1;
      }
      for (int i = 0; i < n - 1; ++i) {
        os[i] = is[i];
      }
    }
  };
};

template <
    int I,
    template<typename... xIs> typename Meta,
    typename I0,
    typename... IsRem
>
struct ShapeTileHelper<I, Meta, std::enable_if_t<(I > 0)>, I0, IsRem...> {
  template <typename... IsProcessed>
  struct Holder {
    using Next = typename ShapeTileHelper<I - 1, Meta, IsRem...>::template Holder<IsProcessed..., I0>;
    template <typename Ts>
    static constexpr bool has_tail = Next::template has_tail<Ts>;
    template <typename Ts>
    using FullType = typename Next::template FullType<Ts>;
    template <typename Ts>
    using TailType = typename Next::template TailType<Ts>;
    template <typename Ts>
    static void set_full(const int * is, int * os, int n, Ts ts) {
      if constexpr (I0::is_runtime) {
        os[0] = is[0];
        Next::template set_full<Ts>(is + 1, os + 1, n - 1, ts);
      } else {
        Next::template set_full<Ts>(is, os, n, ts);
      }
    }
    template <typename Ts>
    static void set_tail(const int * is, int * os, int n, Ts ts) {
      if constexpr (I0::is_runtime) {
        os[0] = is[0];
        Next::template set_tail<Ts>(is + 1, os + 1, n - 1, ts);
      } else {
        Next::template set_tail<Ts>(is, os, n, ts);
      }
    }
  };
};

template <typename Meta, nuint_t... Is>
nint_t numel_of(const Meta& m, std::index_sequence<Is...>) {
  nint_t x = 1;
  ((x *= nint_t(m.template get<Is>())) && ...);
  return x;
}

} // namespace details


template <typename... Is>
struct MatrixMeta {
  static_assert((std::is_base_of_v<ValueDef, Is> && ...), "Is is not base of ValueDef");
  static_assert(sizeof...(Is) > 0, "ndim cannot be 0");
  static constexpr int Ndim = sizeof...(Is);

  template <typename... Ints>
  explicit MatrixMeta(Ints... vs) {
    static_assert(sizeof...(Ints) == sizeof...(Is), "MatrixMeta: argument count mismatch");
    int arr[] = {int(vs)...};
    VECOPS_ASSERT(details::ConformCheck<Is...>::check(arr), "values do not conform to type constraints");
    details::IndexSetter<Is...>()(arr, _is);
  }

  MatrixMeta() = default;

  template <int N>
  constexpr int get() const {
    return details::IndexGetter<N, Is...>{}(_is);
  }

  constexpr int ndim() const {
    return Ndim;
  }
protected:
  static constexpr int _IsLen = (int(Is::is_runtime) + ...);
  int _is[_IsLen];
};

template <typename... Is>
struct Shape : public MatrixMeta<Is...> {
  template <typename... Ints>
  Shape(Ints... is) : MatrixMeta<Is...>(is...) {
    VECOPS_ASSERT(((is >= 0) && ...), "is must be non-negative");
  }

  Shape() = default;

  template <int I = 0>
  constexpr auto take() const {
    using H = details::MetaTakeHelper<I, Shape, Is...>::template Holder<>;
    auto r = typename H::Type{};
    H::set(this->_is, r->_is, this->_IsLen);
    return r;
  }

  template <int I = 0, typename Ts>
  constexpr auto tile_full(Ts ts) const {
    using H = details::ShapeTileHelper<I, Shape, Is...>::template Holder<>;
    auto r = typename H::template FullType<Ts>{};
    H::template set_full<Ts>(this->_is, r->_is, this->_IsLen, ts);
    return r;
  }

  template <int I = 0, typename Ts>
  constexpr auto tile_tail(Ts ts) const {
    using H = details::ShapeTileHelper<I, Shape, Is...>::template Holder<>;
    auto r = typename H::template TailType<Ts>{};
    H::template set_tail<Ts>(this->_is, r->_is, this->_IsLen, ts);
    return r;
  }

  nint_t numel() const {
    return details::numel_of(*this, std::index_sequence_for<Is...>{});
  }
};

template <typename... Is>
struct Stride : public MatrixMeta<Is...> {
  template <typename... Ints>
  Stride(Ints... is) : MatrixMeta<Is...>(is...) {
  }

  Stride() = default;

  template <int I = 0>
  constexpr auto take() const {
    using H = details::MetaTakeHelper<I, Stride, Is...>::template Holder<>;
    auto r = typename H::Type{};
    H::set(this->_is, r->_is, this->_IsLen);
    return r;
  }

  template <int Ts, int I = 0>
  constexpr auto tile_full() const {
    auto r = Stride{};
    std::copy(this->_is, this->_is + this->_IsLen, r->_is);
    return r;
  }

  template <int Ts, int I = 0>
  constexpr auto tile_tail() const {
    auto r = Stride{};
    std::copy(this->_is, this->_is + this->_IsLen, r->_is);
    return r;
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
  static constexpr int Ndim = TShape::Ndim;

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

  constexpr int ndim() const {
    return Ndim;
  }

  template <int I = 0>
  constexpr auto take() const {
    auto shape = _shape.template take<I>();
    auto stride = _stride.template take<I>();
    return Layout(shape, stride);
  }

  template <int Ts, int I = 0>
  constexpr auto tile_full() const {
    auto shape = _shape.template tile_full<Ts, I>();
    auto stride = _stride.template tile_full<Ts, I>();
    return Layout(shape, stride);
  }

  template <int Ts, int I = 0>
  constexpr auto tile_tail() const {
    auto shape = _shape.template tile_tail<Ts, I>();
    auto stride = _stride.template tile_tail<Ts, I>();
    return Layout(shape, stride);
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

namespace details {

template <typename ShapeT>
struct KDimType;
template <typename D0, typename D1>
struct KDimType<Shape<D0, D1>> { using type = D1; };

template <typename OrigShape, typename NewD0, typename NewD1>
struct SubShape2D;
template <typename D0_orig, typename D1_orig, typename NewD0, typename NewD1>
struct SubShape2D<Shape<D0_orig, D1_orig>, NewD0, NewD1> {
  using Type = Shape<NewD0, NewD1>;
  static Type create(int d0_val, int d1_val) {
    if constexpr (NewD0::is_const && NewD1::is_const)
      return Type{};
    else if constexpr (NewD0::is_const && NewD1::is_runtime)
      return Type{NewD0::value, d1_val};
    else if constexpr (NewD0::is_runtime && NewD1::is_const)
      return Type{d0_val, NewD1::value};
    else
      return Type{d0_val, d1_val};
  }
};

template <int I, typename Meta>
struct MetaParamAt;

template <int I, template<typename...> typename Meta, typename I0, typename... Is>
struct MetaParamAt<I, Meta<I0, Is...>> {
  using Type = typename MetaParamAt<I - 1, Meta<Is...>>::Type;
};

template <template<typename...> typename Meta, typename I0, typename... Is>
struct MetaParamAt<0, Meta<I0, Is...>> {
  using Type = I0;
};

template <typename T>
struct IsConstOne : std::false_type {};

template <int N>
struct IsConstOne<Int<N>> : std::bool_constant<N == 1> {};

template <typename StrideT, int Dim>
struct StrideDimConst1 : std::false_type {};

template <typename S0, typename S1>
struct StrideDimConst1<Stride<S0, S1>, 0>
    : std::bool_constant<IsConstOne<S0>::value> {};

template <typename S0, typename S1>
struct StrideDimConst1<Stride<S0, S1>, 1>
    : std::bool_constant<IsConstOne<S1>::value> {};

template <typename StrideT>
struct StrideIsCompileTime : std::false_type {};

template <typename... Is>
struct StrideIsCompileTime<Stride<Is...>>
    : std::bool_constant<(is_int<Is> && ...)> {};

template <typename ShapeT, int Dim>
struct ShapeDim;

template <typename D0, typename D1>
struct ShapeDim<Shape<D0, D1>, 0> {
  static constexpr int value = D0::value;
};

template <typename D0, typename D1>
struct ShapeDim<Shape<D0, D1>, 1> {
  static constexpr int value = D1::value;
};

} // namespace details

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

/**
 * Kernel模板，这里就是一个接口示意
 */
struct Kernel {
  // 内核分块大小，实际内核可能依次处理多个分块
  static constexpr int Mtile = 0;
  static constexpr int Ntile = 0;
  static constexpr int Ktile = 0;
  using TAccumulator = float;

  // Whether M (for pack_A) or N (for pack_B) is on the second-to-last
  // dimension of the packed layout.  If false, M/N is on the last dim.
  static constexpr bool pack_A_M_on_dim2 = false;
  static constexpr bool pack_B_N_on_dim2 = false;

  // Tile-count shapes: Shape<(Mtiled, Ntiled)> where the actual kernel tile size
  // processed by one call is (Mtiled * Mtile, Ntiled * Ntile).
  //   shape_upper_left  — both M,N non-boundary (bulk, largest tile)
  //   shape_upper_right — M non-boundary, N may be boundary
  //   shape_lower_left  — M may be boundary, N non-boundary
  //   shape_lower_right — both M,N may be boundary (smallest tile)
  // All tile-count dimensions must satisfy: upper_left >= others; all divisible by
  // lower_right (ul_m % lr_m == 0 etc.).
  static constexpr Shape<Int<1>, Int<1>> shape_upper_left  {};
  static constexpr Shape<Int<1>, Int<1>> shape_upper_right {};
  static constexpr Shape<Int<1>, Int<1>> shape_lower_left  {};
  static constexpr Shape<Int<1>, Int<1>> shape_lower_right {};

  /**
   * @tparam Accumulate  if true, read from acc and accumulate; if false, start fresh
   * @param A array pointed to start of the tile
   * @param A_layout  shape (* <= Mtile, K), stride (*, *)
   * @param B array pointed to start of the tile
   * @param B_layout  shape (* <= Ntile, K), stride (*, *)
   * @param acc  accumulator buffer (row-major, stride = acc_ld), or C itself in bypass mode
   * @param acc_ld  row stride of accumulator buffer in elements
   */
  template <bool Accumulate,
            typename TA, typename ALayout,
            typename TB, typename BLayout>
  void run(
      const TA * A, ALayout A_layout,
      const TB * B, BLayout B_layout,
      TAccumulator * acc, int acc_ld
  );

  /**
   * @tparam Layout (M, K), stride (*, *)
   * @returns layout of (Mtiled=ceil(M/Mtile), Mtile, <specified by kernel, arbitrary dims but >= 1>), stride (all contiguous)
   */
  template <typename Layout>
  using PackedALayout = void;

  /**
   * @tparam Layout (N, K), stride (*, *)
   * @returns layout of (Ntiled=ceil(N/Ntile), Ntile, <specified by kernel, arbitrary dims but >= 1>), stride (all contiguous)
   */
  template <typename Layout>
  using PackedBLayout = void;

  template <typename T, typename Layout, typename PackedLayout = PackedALayout<Layout>>
  void pack_A(const T* A, Layout A_layout, T * A_packed, PackedLayout A_packed_layout);

  template <typename T, typename Layout, typename PackedLayout = PackedBLayout<Layout>>
  void pack_B(const T* B, Layout B_layout, T * B_packed, PackedLayout B_packed_layout);
};

/**
 * SchedulerMaxCases: partitions the M×N plane into 4 regions handled by
 * shape_upper_left, shape_upper_right, shape_lower_left, shape_lower_right.
 * Maximizes the use of specialized kernel configurations for each boundary case.
 */
struct SchedulerMaxCases {};

/**
 * SchedulerMinCases: partitions the M×N plane into 2 regions using
 * shape_upper_left for the main bulk and shape_lower_right for all boundaries.
 * Minimizes the number of kernel configurations invoked.
 */
struct SchedulerMinCases {};

// ============================================================================
// InputAdapter: ensures A and B are in a layout the kernel can consume
// ============================================================================
//
// When the input stride is fully compile-time known (all dims are Int<N>),
// InputAdapter is a no-op — the original pointer and layout are passed to the
// kernel, which handles online-transpose via its own stride-type dispatch.
//
// When the stride is not compile-time known (any dim is Any/Aligned), the
// adapter performs a runtime check: if already row-major (stride[1]==1) it
// passes through; otherwise it gathers into a row-major contiguous buffer.
//
// Post-condition: the layout returned by prepare() always has stride[1]==1
// (inner dimension contiguous), which is the kernel's default input format.

template <typename T, typename StrideT,
          bool CompileTimeKnown = details::StrideIsCompileTime<StrideT>::value>
struct InputAdapter {
  static constexpr bool needs_buffer() { return false; }

  static const T* prepare(const T* base, int /*rows*/, int /*cols*/,
                          nint_t /*sr*/, nint_t /*sc*/,
                          T* /*buf*/,
                          auto& /*out_layout*/) {
    return base;
  }
};

template <typename T, typename S0, typename S1>
struct InputAdapter<T, Stride<S0, S1>, false> {
  static constexpr bool needs_buffer() { return true; }

  static const T* prepare(const T* base, int rows, int cols,
                          nint_t sr, nint_t sc,
                          T* buf,
                          auto& out_layout) {
    if (sc == 1) {
      out_layout = make_layout(make_shape(rows, cols),
                               make_stride(int(sr), int(sc)));
      return base;
    }
    for (int r = 0; r < rows; ++r)
      for (int c = 0; c < cols; ++c)
        buf[r * cols + c] = base[r * sr + c * sc];
    out_layout = make_layout(make_shape(rows, cols),
                             make_stride(cols, 1));
    return buf;
  }
};

// ============================================================================
// OutputAdapter: drains the accumulator buffer into C, applying EpilogFn
// ============================================================================
//
// Selected at compile time based on C's stride type. Three specializations:
//  - Row-major (stride[1] == Int<1>): direct copy
//  - Col-major (stride[0] == Int<1>): transpose copy
//  - Scattered (neither dim == 1 at compile time, or runtime unknown): scatter
//
// For the runtime-unknown case, a runtime check on sc_n/sc_m selects the path.
//
// When the accumulator IS C (buffer bypass: no K-tiling, TC==TAcc, row-major),
// this performs an in-place element-wise transform via EpilogFn, which the
// compiler can eliminate if EpilogFn is identity and TC==TAcc.

template <typename TC, typename TAcc, typename CStride,
          bool CompileTimeKnown = details::StrideIsCompileTime<CStride>::value>
struct OutputAdapter;

template <typename TC, typename TAcc, typename S0>
    requires (!details::IsConstOne<S0>::value)
struct OutputAdapter<TC, TAcc, Stride<S0, Int<1>>, true> {
  template <typename Fn>
  static void drain(const TAcc* acc, int acc_ld, int curM, int curN,
                    TC* C, nint_t sc_m, nint_t sc_n,
                    int offM, int offN, const Fn& fn) {
    for (int m = 0; m < curM; ++m)
      for (int n = 0; n < curN; ++n)
        C[(offM + m) * sc_m + (offN + n) * sc_n] =
            static_cast<TC>(fn(offM + m, offN + n,
                               acc[m * acc_ld + n]));
  }
};

template <typename TC, typename TAcc, typename S1>
    requires (!details::IsConstOne<S1>::value)
struct OutputAdapter<TC, TAcc, Stride<Int<1>, S1>, true> {
  template <typename Fn>
  static void drain(const TAcc* acc, int acc_ld, int curM, int curN,
                    TC* C, nint_t sc_m, nint_t sc_n,
                    int offM, int offN, const Fn& fn) {
    for (int m = 0; m < curM; ++m)
      for (int n = 0; n < curN; ++n)
        C[(offM + m) * sc_m + (offN + n) * sc_n] =
            static_cast<TC>(fn(offM + m, offN + n,
                               acc[m * acc_ld + n]));
  }
};

template <typename TC, typename TAcc>
struct OutputAdapter<TC, TAcc, Stride<Int<1>, Int<1>>, true> {
  template <typename Fn>
  static void drain(const TAcc* acc, int acc_ld, int curM, int curN,
                    TC* C, nint_t sc_m, nint_t sc_n,
                    int offM, int offN, const Fn& fn) {
    for (int m = 0; m < curM; ++m)
      for (int n = 0; n < curN; ++n)
        C[(offM + m) * sc_m + (offN + n) * sc_n] =
            static_cast<TC>(fn(offM + m, offN + n,
                               acc[m * acc_ld + n]));
  }
};

template <typename TC, typename TAcc, typename S0, typename S1>
struct OutputAdapter<TC, TAcc, Stride<S0, S1>, true> {
  template <typename Fn>
  static void drain(const TAcc* acc, int acc_ld, int curM, int curN,
                    TC* C, nint_t sc_m, nint_t sc_n,
                    int offM, int offN, const Fn& fn) {
    for (int m = 0; m < curM; ++m)
      for (int n = 0; n < curN; ++n)
        C[(offM + m) * sc_m + (offN + n) * sc_n] =
            static_cast<TC>(fn(offM + m, offN + n,
                               acc[m * acc_ld + n]));
  }
};

template <typename TC, typename TAcc, typename S0, typename S1>
struct OutputAdapter<TC, TAcc, Stride<S0, S1>, false> {
  template <typename Fn>
  static void drain(const TAcc* acc, int acc_ld, int curM, int curN,
                    TC* C, nint_t sc_m, nint_t sc_n,
                    int offM, int offN, const Fn& fn) {
    if (sc_n == 1) {
      for (int m = 0; m < curM; ++m)
        for (int n = 0; n < curN; ++n)
          C[(offM + m) * sc_m + (offN + n)] =
              static_cast<TC>(fn(offM + m, offN + n,
                                 acc[m * acc_ld + n]));
    } else if (sc_m == 1) {
      for (int m = 0; m < curM; ++m)
        for (int n = 0; n < curN; ++n)
          C[(offM + m) + (offN + n) * sc_n] =
              static_cast<TC>(fn(offM + m, offN + n,
                                 acc[m * acc_ld + n]));
    } else {
      for (int m = 0; m < curM; ++m)
        for (int n = 0; n < curN; ++n)
          C[(offM + m) * sc_m + (offN + n) * sc_n] =
              static_cast<TC>(fn(offM + m, offN + n,
                                 acc[m * acc_ld + n]));
    }
  }
};

// ============================================================================
// gemm: the GEMM orchestrator
// ============================================================================
//
// Decomposes C <- A * B into a hierarchy of tiles.
//
// Outer level (TilesShape): splits (M,N,K) into outer tiles of size (Mt,Nt,Kt).
// If TilesShape is 2D (M,N only), Kt defaults to K (single K pass).
//
// Inner level (Scheduler + Kernel shapes): within one outer tile, the Scheduler
// partitions the (curM,curN) plane using the Kernel's four shape constants.
//
// Accumulation (across K tiles): the kernel always writes to an accumulator
// buffer.  For K-tiling the buffer is reused (Accumulate=false for first pass,
// true thereafter).  After all K passes, the OutputAdapter drains the buffer
// to C, applying the user-provided EpilogFn per element.
//
// Buffer-bypass: when C is row-major AND TC == TAcc, C itself serves as the
// accumulator.  No separate buffer is allocated.  OutputAdapter applies
// EpilogFn in-place.
//
// InputAdapter: if strides are fully compile-time known (all Int<N>), the
// original layout goes directly to the kernel.  Otherwise a runtime check
// decides: row-major -> pass through; else -> gather into contiguous buffer.
// The kernel's default input format is row-major (stride[1]==1).
//
// OutputAdapter: selected at compile time from C's stride type.  Row-major,
// col-major, and scattered each have a specialisation.  For runtime strides,
// a runtime if/else selects the write path.
//
// Constraints:
//   - Every input matrix must have at least one contiguous (stride==1) axis.
//   - TilesShape (Mt,Nt) must be multiples of Kernel::Mtile / Ntile
//     respectively, so outer→inner tile subdivision is well-defined.
//   - Tile dimensions must divide the problem's bulk region evenly.
//   - Pre-packed A/B (ALayout::Ndim > 2) is not yet implemented.
//
// Pitfalls:
//   - acc_ld == curN in non-bypass mode; kernel tile N may differ when the
//     scheduler splits N, so acc_ld must be passed to the kernel explicitly.
//   - EpilogFn receives GLOBAL (m,n) coordinates, not tile-local.
//   - Gather buffers are sized (Mt*Kt) / (Nt*Kt), reused across outer tiles.

template <
    typename ProblemShape,
    typename TA, typename ALayout,
    typename TB, typename BLayout,
    typename TC, typename CLayout,
    typename TilesShape,
    typename Kernel,
    typename Scheduler,
    typename EpilogFn
>
void gemm(
    ProblemShape shape_MNK,
    const TA * A, ALayout A_layout,
    const TB * B, BLayout B_layout,
    TC * C, CLayout C_layout,
    TilesShape tiles_shape,
    Kernel kernel,
    [[maybe_unused]] Scheduler scheduler,
    const EpilogFn& fn
) {
  int M = shape_MNK.template get<0>();
  int N = shape_MNK.template get<1>();
  int K = shape_MNK.template get<2>();

  int Mt = tiles_shape.template get<0>();
  int Nt = tiles_shape.template get<1>();
  int Kt = K;
  if constexpr (TilesShape::Ndim >= 3) {
    Kt = tiles_shape.template get<2>();
  }

  constexpr int kMt = Kernel::Mtile;
  constexpr int kNt = Kernel::Ntile;

  VECOPS_ASSERT(Mt % kMt == 0,
                "TilesShape M-tile must be divisible by Kernel::Mtile");
  VECOPS_ASSERT(Nt % kNt == 0,
                "TilesShape N-tile must be divisible by Kernel::Ntile");

  constexpr bool is_Apacked = ALayout::Ndim > 2;
  constexpr bool is_Bpacked = BLayout::Ndim > 2;

  constexpr int ul_m = details::ShapeDim<std::remove_const_t<decltype(Kernel::shape_upper_left)>, 0>::value;
  constexpr int ul_n = details::ShapeDim<std::remove_const_t<decltype(Kernel::shape_upper_left)>, 1>::value;
  constexpr int ur_m = details::ShapeDim<std::remove_const_t<decltype(Kernel::shape_upper_right)>, 0>::value;
  constexpr int ur_n = details::ShapeDim<std::remove_const_t<decltype(Kernel::shape_upper_right)>, 1>::value;
  constexpr int ll_m = details::ShapeDim<std::remove_const_t<decltype(Kernel::shape_lower_left)>, 0>::value;
  constexpr int ll_n = details::ShapeDim<std::remove_const_t<decltype(Kernel::shape_lower_left)>, 1>::value;
  constexpr int lr_m = details::ShapeDim<std::remove_const_t<decltype(Kernel::shape_lower_right)>, 0>::value;
  constexpr int lr_n = details::ShapeDim<std::remove_const_t<decltype(Kernel::shape_lower_right)>, 1>::value;
  static_assert(ul_m % lr_m == 0 && ur_m % lr_m == 0 && ll_m % lr_m == 0, "lr_m is not the smallest tile");
  static_assert(ul_n % lr_n == 0 && ur_n % lr_n == 0 && ll_n % lr_n == 0, "lr_n is not the smallest tile");

  constexpr int ul_M = ul_m * kMt;
  constexpr int ul_N = ul_n * kNt;
  constexpr int ur_M = ur_m * kMt;
  constexpr int ur_N = ur_n * kNt;
  constexpr int ll_M = ll_m * kMt;
  constexpr int ll_N = ll_n * kNt;
  constexpr int lr_M = lr_m * kMt;
  constexpr int lr_N = lr_n * kNt;

  constexpr int kKt = Kernel::Ktile;

  using TAcc = typename Kernel::TAccumulator;

  using AStride = typename ALayout::Stride;
  using BStride = typename BLayout::Stride;
  using CStride = typename CLayout::Stride;

  using InA = InputAdapter<TA, AStride>;
  using InB = InputAdapter<TB, BStride>;
  using Out = OutputAdapter<TC, TAcc, CStride>;

  using InAPacked = InputAdapter<TA, Stride<Any, Any>>;
  using InBPacked = InputAdapter<TB, Stride<Any, Any>>;

  nint_t sc_m = C_layout.stride().template get<0>();
  nint_t sc_n = C_layout.stride().template get<1>();

  constexpr bool c_is_row_major = details::StrideDimConst1<CStride, 1>::value;
  constexpr bool buffer_bypass =
      std::is_same_v<TC, TAcc> && c_is_row_major;

  TAcc * acc_buf = nullptr;
  if (!buffer_bypass) {
    nuint_t buf_elems = nuint_t(Mt) * nuint_t(Nt);
    constexpr nuint_t align = 64;
    nuint_t alloc_size = ((buf_elems * sizeof(TAcc) + align - 1) / align) * align;
    acc_buf = (TAcc *)std::aligned_alloc(align, alloc_size);
    VECOPS_ASSERT(acc_buf, "aligned_alloc failed");
  }

  TA * a_gather = nullptr;
  TB * b_gather = nullptr;

  if constexpr (is_Apacked || is_Bpacked) {
    VECOPS_ASSERT(Mt == kMt, "When A or B is packed, tiles_shape Mt must equal Kernel Mtile");
    VECOPS_ASSERT(Nt == kNt, "When A or B is packed, tiles_shape Nt must equal Kernel Ntile");
    VECOPS_ASSERT(Kt % kKt == 0, "When A or B is packed, Kt must be multiple of Kernel Ktile");
  }

  if constexpr (is_Apacked || is_Bpacked) {
    if constexpr (InAPacked::needs_buffer()) {
      nuint_t sz = nuint_t(Mt) * nuint_t(Kt);
      a_gather = (TA *)std::aligned_alloc(64, sz * sizeof(TA));
      VECOPS_ASSERT(a_gather, "aligned_alloc A gather failed");
    }
    if constexpr (InBPacked::needs_buffer()) {
      nuint_t sz = nuint_t(Nt) * nuint_t(Kt);
      b_gather = (TB *)std::aligned_alloc(64, sz * sizeof(TB));
      VECOPS_ASSERT(b_gather, "aligned_alloc B gather failed");
    }
  } else {
    nint_t sa_m = A_layout.stride().template get<0>();
    nint_t sa_k = A_layout.stride().template get<1>();
    nint_t sb_n = B_layout.stride().template get<0>();
    nint_t sb_k = B_layout.stride().template get<1>();
    if constexpr (InA::needs_buffer()) {
      nuint_t sz = nuint_t(Mt) * nuint_t(Kt);
      a_gather = (TA *)std::aligned_alloc(64, sz * sizeof(TA));
      VECOPS_ASSERT(a_gather, "aligned_alloc A gather failed");
    }
    if constexpr (InB::needs_buffer()) {
      nuint_t sz = nuint_t(Nt) * nuint_t(Kt);
      b_gather = (TB *)std::aligned_alloc(64, sz * sizeof(TB));
      VECOPS_ASSERT(b_gather, "aligned_alloc B gather failed");
    }
  }

  auto run_scheduler = [&](
      int curM, int curN, int curK,
      int offM, int offN, int offK,
      bool accumulate, TAcc * acc_buf, int acc_ld,
      const TA * a_ptr, const auto& a_layout,
      const TB * b_ptr, const auto& b_layout)
  {
    int m1 = (curM / ul_M) * ul_M;
    int n1 = (curN / ul_N) * ul_N;
    int m2 = (curM / ur_M) * ur_M;
    int n2 = (curN / ll_N) * ll_N;

    nint_t a_sm = a_layout.stride().template get<0>();
    nint_t b_sn = b_layout.stride().template get<0>();

    auto call_tile = [&]<typename MType, typename NType>(
        int m_loc, int n_loc, int tile_m, int tile_n)
    {
      using SubAS = details::SubShape2D<Shape<Any, Any>, MType, Any>;
      using SubBS = details::SubShape2D<Shape<Any, Any>, NType, Any>;

      auto sub_a_shape = SubAS::create(tile_m, curK);
      auto sub_b_shape = SubBS::create(tile_n, curK);

      auto sub_a_layout = make_layout(sub_a_shape, a_layout.stride());
      auto sub_b_layout = make_layout(sub_b_shape, b_layout.stride());

      TAcc * tile_buf = acc_buf
          ? acc_buf + static_cast<nuint_t>(m_loc) * static_cast<nuint_t>(acc_ld)
                     + static_cast<nuint_t>(n_loc)
          : nullptr;

      if (accumulate)
        kernel.template run<true>(a_ptr + m_loc * a_sm, sub_a_layout,
                                   b_ptr + n_loc * b_sn, sub_b_layout,
                                   tile_buf, acc_ld);
      else
        kernel.template run<false>(a_ptr + m_loc * a_sm, sub_a_layout,
                                    b_ptr + n_loc * b_sn, sub_b_layout,
                                    tile_buf, acc_ld);
    };

    if constexpr (std::is_same_v<Scheduler, SchedulerMaxCases>) {
      for (int m = 0; m < m1; m += ul_M)
        for (int n = 0; n < n1; n += ul_N)
          call_tile.template operator()<Int<ul_M>, Int<ul_N>>(m, n, ul_M, ul_N);

      for (int n = n1; n < curN; n += ur_N) {
        int tile_n = std::min(ur_N, curN - n);
        for (int m = 0; m < m2; m += ur_M)
          call_tile.template operator()<Int<ur_M>, Any>(m, n, ur_M, tile_n);
      }

      for (int n = n1; n < curN; n += lr_N) {
        int tile_n = std::min(lr_N, curN - n);
        for (int m = m2; m < m1; m += lr_M)
          call_tile.template operator()<Int<lr_M>, Any>(m, n, lr_M, tile_n);
      }

      for (int m = m1; m < curM; m += ll_M) {
        int tile_m = std::min(ll_M, curM - m);
        for (int n = 0; n < n2; n += ll_N)
          call_tile.template operator()<Any, Int<ll_N>>(m, n, tile_m, ll_N);
      }

      for (int m = m1; m < curM; m += lr_M) {
        int tile_m = std::min(lr_M, curM - m);
        for (int n = n2; n < curN; n += lr_N) {
          int tile_n = std::min(lr_N, curN - n);
          call_tile.template operator()<Any, Any>(m, n, tile_m, tile_n);
        }
      }
    } else {
      static_assert(std::is_same_v<Scheduler, SchedulerMinCases>);

      for (int m = 0; m < m1; m += ul_M)
        for (int n = 0; n < n1; n += ul_N)
          call_tile.template operator()<Int<ul_M>, Int<ul_N>>(m, n, ul_M, ul_N);

      for (int m = 0; m < m1; m += lr_M)
        for (int n = n1; n < curN; n += lr_N) {
          int tile_n = std::min(lr_N, curN - n);
          call_tile.template operator()<Any, Any>(m, n, lr_M, tile_n);
        }

      for (int m = m1; m < curM; m += lr_M) {
        int tile_m = std::min(lr_M, curM - m);
        for (int n = 0; n < curN; n += lr_N) {
          int tile_n = std::min(lr_N, curN - n);
          call_tile.template operator()<Any, Any>(m, n, tile_m, tile_n);
        }
      }
    }
  };

  nint_t sa_m, sa_k, sb_n, sb_k;
  if constexpr (!is_Apacked) {
    sa_m = A_layout.stride().template get<0>();
    sa_k = A_layout.stride().template get<1>();
  }
  if constexpr (!is_Bpacked) {
    sb_n = B_layout.stride().template get<0>();
    sb_k = B_layout.stride().template get<1>();
  }

  auto prepare_packed_A = [&](int mi, int ki) -> const TA* {
    int a_im = mi / kMt;
    int a_ik = ki / kKt;
    return A + a_im * A_layout.stride().template get<0>()
             + a_ik * A_layout.stride().template get<1>();
  };

  auto prepare_packed_B = [&](int ni, int ki) -> const TB* {
    int b_in = ni / kNt;
    int b_ik = ki / kKt;
    return B + b_in * B_layout.stride().template get<0>()
             + b_ik * B_layout.stride().template get<1>();
  };

  for (int mi = 0; mi < M; mi += Mt) {
    int curM = std::min(Mt, M - mi);
    for (int ni = 0; ni < N; ni += Nt) {
      int curN = std::min(Nt, N - ni);

      TAcc * tile_acc = buffer_bypass
          ? reinterpret_cast<TAcc *>(C + mi * sc_m + ni * sc_n)
          : acc_buf;
      int acc_ld = buffer_bypass ? static_cast<int>(sc_m) : curN;

      for (int ki = 0; ki < K; ki += Kt) {
        int curK = std::min(Kt, K - ki);
        bool accumulate = (ki > 0);

        const TA * a_base;
        const TB * b_base;
        const TA * a_run;
        const TB * b_run;

        if constexpr (is_Apacked && is_Bpacked) {
          a_base = prepare_packed_A(mi, ki);
          b_base = prepare_packed_B(ni, ki);
          nint_t a_s, a_k, b_s, b_k;
          if constexpr (Kernel::pack_A_M_on_dim2) { a_s = kKt; a_k = 1; } else { a_s = 1; a_k = kMt; }
          if constexpr (Kernel::pack_B_N_on_dim2) { b_s = kKt; b_k = 1; } else { b_s = 1; b_k = kNt; }
          auto a_l = make_layout(make_shape(curM, curK), make_stride(int(a_s), int(a_k)));
          auto b_l = make_layout(make_shape(curN, curK), make_stride(int(b_s), int(b_k)));
          a_run = InAPacked::prepare(a_base, curM, curK, a_s, a_k, a_gather, a_l);
          b_run = InBPacked::prepare(b_base, curN, curK, b_s, b_k, b_gather, b_l);
          run_scheduler(curM, curN, curK, mi, ni, ki,
                        accumulate, tile_acc, acc_ld,
                        a_run, a_l, b_run, b_l);
        } else if constexpr (is_Apacked) {
          a_base = prepare_packed_A(mi, ki);
          b_base = B + ni * sb_n + ki * sb_k;
          nint_t a_s, a_k;
          if constexpr (Kernel::pack_A_M_on_dim2) { a_s = kKt; a_k = 1; } else { a_s = 1; a_k = kMt; }
          auto a_l = make_layout(make_shape(curM, curK), make_stride(int(a_s), int(a_k)));
          a_run = InAPacked::prepare(a_base, curM, curK, a_s, a_k, a_gather, a_l);
          auto b_lay = B_layout;
          b_run = InB::prepare(b_base, curN, curK, sb_n, sb_k, b_gather, b_lay);
          run_scheduler(curM, curN, curK, mi, ni, ki,
                        accumulate, tile_acc, acc_ld,
                        a_run, a_l, b_run, b_lay);
        } else if constexpr (is_Bpacked) {
          a_base = A + mi * sa_m + ki * sa_k;
          b_base = prepare_packed_B(ni, ki);
          nint_t b_s, b_k;
          if constexpr (Kernel::pack_B_N_on_dim2) { b_s = kKt; b_k = 1; } else { b_s = 1; b_k = kNt; }
          auto b_l = make_layout(make_shape(curN, curK), make_stride(int(b_s), int(b_k)));
          auto a_lay = A_layout;
          a_run = InA::prepare(a_base, curM, curK, sa_m, sa_k, a_gather, a_lay);
          b_run = InBPacked::prepare(b_base, curN, curK, b_s, b_k, b_gather, b_l);
          run_scheduler(curM, curN, curK, mi, ni, ki,
                        accumulate, tile_acc, acc_ld,
                        a_run, a_lay, b_run, b_l);
        } else {
          a_base = A + mi * sa_m + ki * sa_k;
          b_base = B + ni * sb_n + ki * sb_k;
          auto a_lay = A_layout;
          auto b_lay = B_layout;
          a_run = InA::prepare(a_base, curM, curK, sa_m, sa_k, a_gather, a_lay);
          b_run = InB::prepare(b_base, curN, curK, sb_n, sb_k, b_gather, b_lay);
          run_scheduler(curM, curN, curK, mi, ni, ki,
                        accumulate, tile_acc, acc_ld,
                        a_run, a_lay, b_run, b_lay);
        }
      }

      Out::drain(tile_acc, acc_ld, curM, curN,
                 C, sc_m, sc_n, mi, ni, fn);
    }
  }

  if (acc_buf) std::free(acc_buf);
  if constexpr (is_Apacked || is_Bpacked) {
    if constexpr (InAPacked::needs_buffer()) std::free(a_gather);
    if constexpr (InBPacked::needs_buffer()) std::free(b_gather);
  } else {
    if constexpr (InA::needs_buffer()) std::free(a_gather);
    if constexpr (InB::needs_buffer()) std::free(b_gather);
  }
}

}

#endif //VECOPS_GEMM_H
