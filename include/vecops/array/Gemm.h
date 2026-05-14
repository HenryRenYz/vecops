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
      return IndexGetter<N - 1, Is...>{}(in + 1);
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

template <int N, bool IsPacked, int TileSize>
using InteriorType = Int<N>;

template <int N, bool IsPacked>
using BoundaryType = std::conditional_t<IsPacked, Int<N>, Int<1>>;

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

  // Packed layout inner block dimensions (compile-time Int<N> values)
  static constexpr int A_block_dim2 = 0;
  static constexpr int A_block_dim3 = 0;
  static constexpr int B_block_dim2 = 0;
  static constexpr int B_block_dim3 = 0;

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
   * @param A_layout  if Ndim==2: shape (* <= Mtile, K), stride (*, *)
   *                  if Ndim==4: shape (tile_count, K, block_dim2, block_dim3), stride packed
   * @param B array pointed to start of the tile
   * @param B_layout  same distinction as A_layout
   * @param acc  accumulator buffer (row-major, stride = acc_ld), or C itself in bypass mode
   * @param acc_ld  row stride of accumulator buffer in elements
   * @param C   output matrix pointer (for direct-to-C write + Epilog)
   * @param C_layout  output matrix layout
   * @param offM, offN  global (M,N) offset of this tile in C
   * @param fn  Epilog function, applied when writing to C
   * @param validM, validN  actual element counts for this sub-tile (may be less than packed tile_count * tile_size)
   */
  template <bool Accumulate,
            typename TA, typename ALayout,
            typename TB, typename BLayout,
            typename TC, typename CLayout,
            typename EpilogFn>
  void run(
      const TA * A, ALayout A_layout,
      const TB * B, BLayout B_layout,
      TAccumulator * acc, int acc_ld,
      TC * C, CLayout C_layout,
      int offM, int offN,
      const EpilogFn& fn,
      int validM, int validN
  ) const;

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

template <typename, typename, typename, typename,
          typename, typename, typename, typename,
          typename, typename, typename>
struct GemmOrchestrator;

template <typename Kernel>
struct SubLayoutBuilder {
  template <bool IsPacked, typename TType, typename Layout>
  VECOPS_ALWAYS_INLINE static auto a_sub(int curK, int /*tile*/, const Layout& layout) {
    if constexpr (IsPacked) {
      int nK = (curK + Kernel::Ktile - 1) / Kernel::Ktile;
      auto s = make_shape(Int<TType::value>{}, nK,
                          Int<Kernel::A_block_dim2>{},
                          Int<Kernel::A_block_dim3>{});
      return make_layout(s, layout.stride());
    } else {
      nint_t s0 = layout.stride().template get<0>();
      nint_t s1 = layout.stride().template get<1>();
      auto s  = make_shape(Int<TType::value>{}, Int<Kernel::Mtile>{}, curK);
      auto st = make_stride(int(s0 * Kernel::Mtile), int(s0), int(s1));
      return make_layout(s, st);
    }
  }

  template <bool IsPacked, typename TType, typename Layout>
  VECOPS_ALWAYS_INLINE static auto b_sub(int curK, int /*tile*/, const Layout& layout) {
    if constexpr (IsPacked) {
      int nK = (curK + Kernel::Ktile - 1) / Kernel::Ktile;
      auto s = make_shape(Int<TType::value>{}, nK,
                          Int<Kernel::B_block_dim2>{},
                          Int<Kernel::B_block_dim3>{});
      return make_layout(s, layout.stride());
    } else {
      nint_t s0 = layout.stride().template get<0>();
      nint_t s1 = layout.stride().template get<1>();
      auto s  = make_shape(Int<TType::value>{}, Int<Kernel::Ntile>{}, curK);
      auto st = make_stride(int(s0 * Kernel::Ntile), int(s0), int(s1));
      return make_layout(s, st);
    }
  }
};

struct TileInvoker {
  template <typename MType, typename NType, typename SubLayout>
  VECOPS_ALWAYS_INLINE static void invoke(
      int m_global, int n_global, int tile_m, int tile_n, int curK,
      const auto* a_loc, const auto& a_layout,
      const auto* b_loc, const auto& b_layout,
      auto* tile_buf, int acc_ld,
      auto* C, const auto& C_layout,
      const auto& fn, const auto& kernel, auto accumulate)
  {
    constexpr bool is_Apacked = (std::decay_t<decltype(a_layout)>::Ndim > 2);
    constexpr bool is_Bpacked = (std::decay_t<decltype(b_layout)>::Ndim > 2);

    auto sub_a_l = SubLayout::template a_sub<is_Apacked, MType>(curK, tile_m, a_layout);
    auto sub_b_l = SubLayout::template b_sub<is_Bpacked, NType>(curK, tile_n, b_layout);

    nint_t sc_m = C_layout.stride().template get<0>();
    nint_t sc_n = C_layout.stride().template get<1>();
    auto C_local = C + m_global * sc_m + n_global * sc_n;
    auto C_local_layout = make_layout(make_shape(tile_m, tile_n),
                                      make_stride(int(sc_m), int(sc_n)));

    kernel.run(
        a_loc, sub_a_l,
        b_loc, sub_b_l,
        tile_buf, acc_ld,
        C_local, C_local_layout, m_global, n_global, fn,
        tile_m, tile_n,
        accumulate);
  }
};

template <typename SchedTag, typename Orchestrator>
struct TileScheduler;

template <typename Orchestrator>
struct TileScheduler<SchedulerMaxCases, Orchestrator> {
  VECOPS_ALWAYS_INLINE static void run(
      int curM, int curN, int curK,
      int offM, int offN, int offK,
      auto accumulate, auto* acc_buf, int acc_ld,
      const auto* a_ptr, auto a_layout,
      const auto* b_ptr, auto b_layout,
      auto* C, auto C_layout,
      const auto& kernel, const auto& fn)
  {
    using TAcc = typename Orchestrator::TAcc;
    using SL = SubLayoutBuilder<typename Orchestrator::KernelT>;

    constexpr int kMt = Orchestrator::kMt;
    constexpr int kNt = Orchestrator::kNt;
    constexpr bool is_Apacked = (std::decay_t<decltype(a_layout)>::Ndim >= 4);
    constexpr bool is_Bpacked = (std::decay_t<decltype(b_layout)>::Ndim >= 4);
    constexpr int ul_M = Orchestrator::ul_M, ul_N = Orchestrator::ul_N;
    constexpr int ur_M = Orchestrator::ur_M, ur_N = Orchestrator::ur_N;
    constexpr int ll_M = Orchestrator::ll_M, ll_N = Orchestrator::ll_N;
    constexpr int lr_M = Orchestrator::lr_M, lr_N = Orchestrator::lr_N;
    constexpr int ul_m = Orchestrator::ul_m, ul_n = Orchestrator::ul_n;
    constexpr int ur_m = Orchestrator::ur_m, ur_n = Orchestrator::ur_n;
    constexpr int ll_m = Orchestrator::ll_m, ll_n = Orchestrator::ll_n;
    constexpr int lr_m = Orchestrator::lr_m, lr_n = Orchestrator::lr_n;

    int m1 = (curM / ul_M) * ul_M;
    int n1 = (curN / ul_N) * ul_N;
    int m2 = (curM / ur_M) * ur_M;
    int n2 = (curN / ll_N) * ll_N;

    nint_t a_tile_s0 = a_layout.stride().template get<0>();
    if constexpr (std::decay_t<decltype(a_layout)>::Ndim == 2) a_tile_s0 *= kMt;
    nint_t b_tile_s0 = b_layout.stride().template get<0>();
    if constexpr (std::decay_t<decltype(b_layout)>::Ndim == 2) b_tile_s0 *= kNt;

    using UlM = details::InteriorType<ul_m, is_Apacked, kMt>;
    using UlN = details::InteriorType<ul_n, is_Bpacked, kNt>;
    {
      nint_t buf_step_m = acc_buf ? ul_M * acc_ld : 0;
      nint_t buf_step_n = acc_buf ? ul_N : 0;
      auto* buf_m = acc_buf ? acc_buf : nullptr;
      const auto* a_m = a_ptr;
      for (int m = 0; m < m1; m += ul_M, a_m += ul_m * a_tile_s0, buf_m += buf_step_m) {
        auto* buf_n = buf_m;
        const auto* b_n = b_ptr;
        for (int n = 0; n < n1; n += ul_N, b_n += ul_n * b_tile_s0, buf_n += buf_step_n)
          TileInvoker::invoke<UlM, UlN, SL>(
              offM + m, offN + n, ul_M, ul_N, curK,
              a_m, a_layout, b_n, b_layout,
              buf_n, acc_ld,
              C, C_layout, fn, kernel, accumulate);
      }
    }

    using UrM = details::InteriorType<ur_m, is_Apacked, kMt>;
    using UrN = details::BoundaryType<ur_n, is_Bpacked>;
    {
      nint_t buf_step_n = acc_buf ? ur_N : 0;
      nint_t buf_step_m = acc_buf ? ur_M * acc_ld : 0;
      auto* buf_n = acc_buf ? acc_buf + n1 : nullptr;
      const auto* b_n = b_ptr + (n1 / kNt) * b_tile_s0;
      for (int n = n1; n < curN; n += ur_N, b_n += ur_n * b_tile_s0, buf_n += buf_step_n) {
        int tile_n = std::min(ur_N, curN - n);
        auto* buf_m = buf_n;
        const auto* a_m = a_ptr;
        for (int m = 0; m < m2; m += ur_M, a_m += ur_m * a_tile_s0, buf_m += buf_step_m)
          TileInvoker::invoke<UrM, UrN, SL>(
              offM + m, offN + n, ur_M, tile_n, curK,
              a_m, a_layout, b_n, b_layout,
              buf_m, acc_ld,
              C, C_layout, fn, kernel, accumulate);
      }
    }

    using LrM = details::BoundaryType<lr_m, is_Apacked>;
    using LrN = details::BoundaryType<lr_n, is_Bpacked>;
    {
      nint_t buf_step_n = acc_buf ? lr_N : 0;
      nint_t buf_step_m = acc_buf ? lr_M * acc_ld : 0;
      auto* buf_n = acc_buf ? acc_buf + n1 : nullptr;
      const auto* b_n = b_ptr + (n1 / kNt) * b_tile_s0;
      for (int n = n1; n < curN; n += lr_N, b_n += lr_n * b_tile_s0, buf_n += buf_step_n) {
        int tile_n = std::min(lr_N, curN - n);
        auto* buf_m = acc_buf ? buf_n + m2 * acc_ld : nullptr;
        const auto* a_m = a_ptr + (m2 / kMt) * a_tile_s0;
        for (int m = m2; m < m1; m += lr_M, a_m += lr_m * a_tile_s0, buf_m += buf_step_m)
          TileInvoker::invoke<LrM, LrN, SL>(
              offM + m, offN + n, lr_M, tile_n, curK,
              a_m, a_layout, b_n, b_layout,
              buf_m, acc_ld,
              C, C_layout, fn, kernel, accumulate);
      }
    }

    using LlM = details::BoundaryType<ll_m, is_Apacked>;
    using LlN = details::InteriorType<ll_n, is_Bpacked, kNt>;
    {
      nint_t buf_step_m = acc_buf ? ll_M * acc_ld : 0;
      nint_t buf_step_n = acc_buf ? ll_N : 0;
      auto* buf_m = acc_buf ? acc_buf + m1 * acc_ld : nullptr;
      const auto* a_m = a_ptr + (m1 / kMt) * a_tile_s0;
      for (int m = m1; m < curM; m += ll_M, a_m += ll_m * a_tile_s0, buf_m += buf_step_m) {
        int tile_m = std::min(ll_M, curM - m);
        auto* buf_n = buf_m;
        const auto* b_n = b_ptr;
        for (int n = 0; n < n2; n += ll_N, b_n += ll_n * b_tile_s0, buf_n += buf_step_n)
          TileInvoker::invoke<LlM, LlN, SL>(
              offM + m, offN + n, tile_m, ll_N, curK,
              a_m, a_layout, b_n, b_layout,
              buf_n, acc_ld,
              C, C_layout, fn, kernel, accumulate);
      }
    }

    {
      nint_t buf_step_m = acc_buf ? lr_M * acc_ld : 0;
      nint_t buf_step_n = acc_buf ? lr_N : 0;
      auto* buf_m = acc_buf ? acc_buf + m1 * acc_ld : nullptr;
      const auto* a_m = a_ptr + (m1 / kMt) * a_tile_s0;
      for (int m = m1; m < curM; m += lr_M, a_m += lr_m * a_tile_s0, buf_m += buf_step_m) {
        int tile_m_el = std::min(lr_M, curM - m);
        auto* buf_n = acc_buf ? buf_m + n2 : nullptr;
        const auto* b_n = b_ptr + (n2 / kNt) * b_tile_s0;
        for (int n = n2; n < curN; n += lr_N, b_n += lr_n * b_tile_s0, buf_n += buf_step_n) {
          int tile_n_el = std::min(lr_N, curN - n);
          TileInvoker::invoke<LrM, LrN, SL>(
              offM + m, offN + n, tile_m_el, tile_n_el, curK,
              a_m, a_layout, b_n, b_layout,
              buf_n, acc_ld,
              C, C_layout, fn, kernel, accumulate);
        }
      }
    }
  }
};

template <typename Orchestrator>
struct TileScheduler<SchedulerMinCases, Orchestrator> {
  VECOPS_ALWAYS_INLINE static void run(
      int curM, int curN, int curK,
      int offM, int offN, int offK,
      auto accumulate, auto* acc_buf, int acc_ld,
      const auto* a_ptr, auto a_layout,
      const auto* b_ptr, auto b_layout,
      auto* C, auto C_layout,
      const auto& kernel, const auto& fn)
  {
    using TAcc = typename Orchestrator::TAcc;
    using SL = SubLayoutBuilder<typename Orchestrator::KernelT>;

    constexpr int kMt = Orchestrator::kMt;
    constexpr int kNt = Orchestrator::kNt;
    constexpr bool is_Apacked = (std::decay_t<decltype(a_layout)>::Ndim >= 4);
    constexpr bool is_Bpacked = (std::decay_t<decltype(b_layout)>::Ndim >= 4);
    constexpr int ul_M = Orchestrator::ul_M, ul_N = Orchestrator::ul_N;
    constexpr int lr_M = Orchestrator::lr_M, lr_N = Orchestrator::lr_N;
    constexpr int ul_m = Orchestrator::ul_m, ul_n = Orchestrator::ul_n;
    constexpr int lr_m = Orchestrator::lr_m, lr_n = Orchestrator::lr_n;

    int m1 = (curM / ul_M) * ul_M;
    int n1 = (curN / ul_N) * ul_N;

    nint_t a_tile_s0 = a_layout.stride().template get<0>();
    if constexpr (std::decay_t<decltype(a_layout)>::Ndim == 2) a_tile_s0 *= kMt;
    nint_t b_tile_s0 = b_layout.stride().template get<0>();
    if constexpr (std::decay_t<decltype(b_layout)>::Ndim == 2) b_tile_s0 *= kNt;

    using UlM = details::InteriorType<ul_m, is_Apacked, kMt>;
    using UlN = details::InteriorType<ul_n, is_Bpacked, kNt>;
    {
      nint_t buf_step_m = acc_buf ? ul_M * acc_ld : 0;
      nint_t buf_step_n = acc_buf ? ul_N : 0;
      auto* buf_m = acc_buf ? acc_buf : nullptr;
      const auto* a_m = a_ptr;
      for (int m = 0; m < m1; m += ul_M, a_m += ul_m * a_tile_s0, buf_m += buf_step_m) {
        auto* buf_n = buf_m;
        const auto* b_n = b_ptr;
        for (int n = 0; n < n1; n += ul_N, b_n += ul_n * b_tile_s0, buf_n += buf_step_n)
          TileInvoker::invoke<UlM, UlN, SL>(
              offM + m, offN + n, ul_M, ul_N, curK,
              a_m, a_layout, b_n, b_layout,
              buf_n, acc_ld,
              C, C_layout, fn, kernel, accumulate);
      }
    }

    using LrM = details::BoundaryType<lr_m, is_Apacked>;
    using LrN = details::BoundaryType<lr_n, is_Bpacked>;

    {
      nint_t buf_step_n = acc_buf ? lr_N : 0;
      nint_t buf_step_m = acc_buf ? lr_M * acc_ld : 0;
      auto* buf_n = acc_buf ? acc_buf + n1 : nullptr;
      const auto* b_n = b_ptr + (n1 / kNt) * b_tile_s0;
      for (int n = n1; n < curN; n += lr_N, b_n += lr_n * b_tile_s0, buf_n += buf_step_n) {
        int tile_n = std::min(lr_N, curN - n);
        auto* buf_m = buf_n;
        const auto* a_m = a_ptr;
        for (int m = 0; m < m1; m += lr_M, a_m += lr_m * a_tile_s0, buf_m += buf_step_m)
          TileInvoker::invoke<LrM, LrN, SL>(
              offM + m, offN + n, lr_M, tile_n, curK,
              a_m, a_layout, b_n, b_layout,
              buf_m, acc_ld,
              C, C_layout, fn, kernel, accumulate);
      }
    }

    {
      nint_t buf_step_m = acc_buf ? lr_M * acc_ld : 0;
      nint_t buf_step_n = acc_buf ? lr_N : 0;
      auto* buf_m = acc_buf ? acc_buf + m1 * acc_ld : nullptr;
      const auto* a_m = a_ptr + (m1 / kMt) * a_tile_s0;
      for (int m = m1; m < curM; m += lr_M, a_m += lr_m * a_tile_s0, buf_m += buf_step_m) {
        int tile_m = std::min(lr_M, curM - m);
        auto* buf_n = buf_m;
        const auto* b_n = b_ptr;
        for (int n = 0; n < curN; n += lr_N, b_n += lr_n * b_tile_s0, buf_n += buf_step_n) {
          int tile_n = std::min(lr_N, curN - n);
          TileInvoker::invoke<LrM, LrN, SL>(
              offM + m, offN + n, tile_m, tile_n, curK,
              a_m, a_layout, b_n, b_layout,
              buf_n, acc_ld,
              C, C_layout, fn, kernel, accumulate);
        }
      }
    }
  }
};

template <typename Orchestrator>
struct L2Tiler {
  using TAcc = typename Orchestrator::TAcc;
  using Out  = typename Orchestrator::Out;

  VECOPS_ALWAYS_INLINE static void run(
      int M, int N, int K,
      int Mt, int Nt, int Kt,
      const auto* A, auto A_layout,
      nint_t sa_m, nint_t sa_k,
      nint_t a_step_M, nint_t a_step_K,
      auto* a_gather, auto* a_packed,
      const auto* B, auto B_layout,
      nint_t sb_n, nint_t sb_k,
      nint_t b_step_N, nint_t b_step_K,
      auto* b_gather, auto* b_packed,
      auto* C, auto C_layout,
      nint_t sc_m, nint_t sc_n,
      TAcc* acc_buf,
      const auto& kernel, const auto& fn)
  {
    constexpr bool is_Apacked = Orchestrator::is_Apacked;
    constexpr bool is_Bpacked = Orchestrator::is_Bpacked;
    constexpr bool a_pack_otf = Orchestrator::a_pack_otf;
    constexpr bool b_pack_otf = Orchestrator::b_pack_otf;
    constexpr bool buffer_bypass = Orchestrator::buffer_bypass;
    constexpr bool is_kTiling = Orchestrator::is_kTiling;
    constexpr int kMt = Orchestrator::kMt;
    constexpr int kNt = Orchestrator::kNt;
    constexpr int kKt = Orchestrator::kKt;
    using InA = typename Orchestrator::InA;
    using InB = typename Orchestrator::InB;
    using KernelT = typename Orchestrator::KernelT;

    auto pack_A_otf = [&](const auto* A_src, int curM, int curK,
                          nint_t sa_m, nint_t sa_k, auto* buf) {
      constexpr int Mt = KernelT::Mtile;
      constexpr int Kt_k = KernelT::Ktile;
      int nM = (curM + Mt - 1) / Mt;
      int nK = (curK + Kt_k - 1) / Kt_k;
      int a_inner = Kt_k;
      for (int im = 0; im < nM; ++im)
        for (int ik = 0; ik < nK; ++ik)
          for (int m = 0; m < Mt; ++m)
            for (int k = 0; k < Kt_k; ++k) {
              int mg = im * Mt + m;
              int kg = ik * Kt_k + k;
              auto v = (mg < curM && kg < curK)
                  ? A_src[mg * sa_m + kg * sa_k] : decltype(*A_src)(0);
              buf[im * nK * Mt * a_inner + ik * Mt * a_inner + m * a_inner + k] = v;
            }
      return make_layout(
          make_shape(nM, nK,
                     Int<KernelT::A_block_dim2>{},
                     Int<KernelT::A_block_dim3>{}),
          make_stride(nK * Mt * a_inner, Mt * a_inner,
                      Int<KernelT::Ktile>{}, Int<1>{}));
    };

    auto pack_B_otf = [&](const auto* B_src, int curN, int curK,
                          nint_t sb_n, nint_t sb_k, auto* buf) {
      constexpr int Nt = KernelT::Ntile;
      constexpr int Kt_k = KernelT::Ktile;
      constexpr int Wd = (KernelT::B_block_dim2 > 0)
          ? (Kt_k / KernelT::B_block_dim2) : 2;
      constexpr int kWt = Kt_k / Wd;
      constexpr int kNw = Nt * Wd;
      int nN = (curN + Nt - 1) / Nt;
      int nK = (curK + Kt_k - 1) / Kt_k;
      for (int in = 0; in < nN; ++in)
        for (int ik = 0; ik < nK; ++ik)
          for (int kw = 0; kw < kWt; ++kw)
            for (int x = 0; x < kNw; ++x) {
              int n = x / Wd, ko = x % Wd;
              int ng = in * Nt + n;
              int kg = ik * Kt_k + kw * Wd + ko;
              auto v = (ng < curN && kg < curK)
                  ? B_src[ng * sb_n + kg * sb_k] : decltype(*B_src)(0);
              buf[in * nK * kWt * kNw + ik * kWt * kNw + kw * kNw + x] = v;
            }
      return make_layout(
          make_shape(nN, nK,
                     Int<KernelT::B_block_dim2>{},
                     Int<KernelT::B_block_dim3>{}),
          make_stride(nK * kWt * kNw, kWt * kNw,
                      Int<KernelT::B_block_dim3>{}, Int<1>{}));
    };

    auto make_accumulate = [](int ki) {
      if constexpr (is_kTiling)
        return bool(ki > 0);
      else
        return std::bool_constant<false>{};
    };

    // ═══ Pre-pack B: all N-tiles, full K, uniform sizing ═══
    nuint_t b_stride = 0;
    int Kt_use = Kt;
    if constexpr (a_pack_otf || b_pack_otf) Kt_use = K;
    if constexpr (b_pack_otf) {
      constexpr int NtK = KernelT::Ntile;
      constexpr int Wd_b = (KernelT::B_block_dim2 > 0)
          ? (kKt / KernelT::B_block_dim2) : 2;
      constexpr int kWt_b = kKt / Wd_b;
      constexpr int kNw_b = NtK * Wd_b;
      int nK_pack = (K + kKt - 1) / kKt;
      int nN_per_tile = Nt / NtK;
      b_stride = (nuint_t)nN_per_tile * (nuint_t)nK_pack
               * (nuint_t)kWt_b * (nuint_t)kNw_b;
      const auto* B_ni = B;
      for (int ni = 0; ni < N; ni += Nt, B_ni += b_step_N) {
        int curN_val = std::min(Nt, N - ni);
        pack_B_otf(B_ni, curN_val, K, sb_n, sb_k,
                   b_packed + (ni / Nt) * b_stride);
      }
    }

    const auto* A_m = A;
    for (int mi = 0; mi < M; mi += Mt, A_m += a_step_M) {
      int curM = std::min(Mt, M - mi);

      // ═══ Pack A: current M-tile, full K ═══
      if constexpr (a_pack_otf) {
        pack_A_otf(A_m, curM, K, sa_m, sa_k, a_packed);
      }

      const auto* B_n = B;
      for (int ni = 0; ni < N; ni += Nt, B_n += b_step_N) {
        int curN = std::min(Nt, N - ni);

        TAcc * tile_acc;
        int acc_ld;
        if constexpr (!buffer_bypass) {
          tile_acc = acc_buf;
          acc_ld = curN;
        } else if constexpr (is_kTiling) {
          tile_acc = reinterpret_cast<TAcc*>(C + mi * sc_m + ni * sc_n);
          acc_ld = static_cast<int>(sc_m);
        } else {
          tile_acc = nullptr;
          acc_ld = static_cast<int>(sc_m);
        }

        const auto* A_mk = A_m;
        const auto* B_nk = B_n;
        for (int ki = 0; ki < K; ki += Kt_use, A_mk += a_step_K, B_nk += b_step_K) {
          int curK = std::min(Kt_use, K - ki);
          auto accumulate = make_accumulate(ki);

          if constexpr (a_pack_otf) {
            constexpr int Am = KernelT::Mtile;
            int nK = (K + kKt - 1) / kKt;
            int nM = (curM + Am - 1) / Am;
            auto a_lay = make_layout(
                make_shape(nM, nK,
                           Int<KernelT::A_block_dim2>{},
                           Int<KernelT::A_block_dim3>{}),
                make_stride(nK * Am * kKt, Am * kKt,
                            Int<kKt>{}, Int<1>{}));
            const auto* a_ptr = a_packed + (ki / kKt) * (nuint_t)Am * (nuint_t)kKt;

            if constexpr (b_pack_otf) {
              constexpr int NtKb = KernelT::Ntile;
              constexpr int Wd_b = (KernelT::B_block_dim2 > 0)
                  ? (kKt / KernelT::B_block_dim2) : 2;
              constexpr int kWt_b = kKt / Wd_b;
              constexpr int kNw_b = NtKb * Wd_b;
              int nK_b = (K + kKt - 1) / kKt;
              int nN_u = Nt / NtKb;
              auto b_lay = make_layout(
                  make_shape(nN_u, nK_b,
                             Int<KernelT::B_block_dim2>{},
                             Int<KernelT::B_block_dim3>{}),
                  make_stride(nK_b * kWt_b * kNw_b, kWt_b * kNw_b,
                              Int<KernelT::B_block_dim3>{}, Int<1>{}));
              const auto* b_ptr = b_packed + (ni / Nt) * b_stride
                                + (ki / kKt) * (nuint_t)kWt_b * (nuint_t)kNw_b;
              TileScheduler<typename Orchestrator::SchedTag, Orchestrator>::run(
                  curM, curN, curK, mi, ni, ki,
                  accumulate, tile_acc, acc_ld,
                  a_ptr, a_lay, b_ptr, b_lay,
                  C, C_layout, kernel, fn);
            } else if constexpr (!is_Bpacked) {
              decltype(B_nk) b_run = InB::prepare(B_nk, curN, curK,
                                                   sb_n, sb_k, b_gather, B_layout);
              TileScheduler<typename Orchestrator::SchedTag, Orchestrator>::run(
                  curM, curN, curK, mi, ni, ki,
                  accumulate, tile_acc, acc_ld,
                  a_ptr, a_lay, b_run, B_layout,
                  C, C_layout, kernel, fn);
            } else {
              TileScheduler<typename Orchestrator::SchedTag, Orchestrator>::run(
                  curM, curN, curK, mi, ni, ki,
                  accumulate, tile_acc, acc_ld,
                  a_ptr, a_lay, B_nk, B_layout,
                  C, C_layout, kernel, fn);
            }
          } else if constexpr (b_pack_otf) {
            constexpr int NtKb = KernelT::Ntile;
            constexpr int Wd_b = (KernelT::B_block_dim2 > 0)
                ? (kKt / KernelT::B_block_dim2) : 2;
            constexpr int kWt_b = kKt / Wd_b;
            constexpr int kNw_b = NtKb * Wd_b;
            int nK_b = (K + kKt - 1) / kKt;
            int nN_u = Nt / NtKb;
            auto b_lay = make_layout(
                make_shape(nN_u, nK_b,
                           Int<KernelT::B_block_dim2>{},
                           Int<KernelT::B_block_dim3>{}),
                make_stride(nK_b * kWt_b * kNw_b, kWt_b * kNw_b,
                            Int<KernelT::B_block_dim3>{}, Int<1>{}));
            const auto* b_ptr = b_packed + (ni / Nt) * b_stride
                              + (ki / kKt) * (nuint_t)kWt_b * (nuint_t)kNw_b;

            if constexpr (!is_Apacked) {
              decltype(A_mk) a_run = InA::prepare(A_mk, curM, curK,
                                                   sa_m, sa_k, a_gather, A_layout);
              TileScheduler<typename Orchestrator::SchedTag, Orchestrator>::run(
                  curM, curN, curK, mi, ni, ki,
                  accumulate, tile_acc, acc_ld,
                  a_run, A_layout, b_ptr, b_lay,
                  C, C_layout, kernel, fn);
            } else {
              TileScheduler<typename Orchestrator::SchedTag, Orchestrator>::run(
                  curM, curN, curK, mi, ni, ki,
                  accumulate, tile_acc, acc_ld,
                  A_mk, A_layout, b_ptr, b_lay,
                  C, C_layout, kernel, fn);
            }
          } else {
            decltype(A_mk) a_run = A_mk;
            if constexpr (!is_Apacked) {
              a_run = InA::prepare(A_mk, curM, curK, sa_m, sa_k, a_gather, A_layout);
            }
            decltype(B_nk) b_run = B_nk;
            if constexpr (!is_Bpacked) {
              b_run = InB::prepare(B_nk, curN, curK, sb_n, sb_k, b_gather, B_layout);
            }
            TileScheduler<typename Orchestrator::SchedTag, Orchestrator>::run(
                curM, curN, curK, mi, ni, ki,
                accumulate, tile_acc, acc_ld,
                a_run, A_layout, b_run, B_layout,
                C, C_layout, kernel, fn);
          }
        }

        if constexpr (!buffer_bypass || is_kTiling) {
          Out::drain(tile_acc, acc_ld, curM, curN,
                     C, sc_m, sc_n, mi, ni, fn);
        }
      }
    }
  }
};

// ============================================================================
// GemmOrchestrator
// ============================================================================
//
// Holds all compile-time constants, allocates buffers, pre-computes pointer
// steps, and delegates to L2Tiler for the actual tile traversal.

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
struct GemmOrchestrator {
  using TAcc = typename Kernel::TAccumulator;
  using AStride = typename ALayout::Stride;
  using BStride = typename BLayout::Stride;
  using CStride = typename CLayout::Stride;
  using InA = InputAdapter<TA, AStride>;
  using InB = InputAdapter<TB, BStride>;
  using Out = OutputAdapter<TC, TAcc, CStride>;
  using SchedTag = Scheduler;
  using KernelT = Kernel;
  using L2 = L2Tiler<GemmOrchestrator>;

  static constexpr int kMt = Kernel::Mtile;
  static constexpr int kNt = Kernel::Ntile;
  static constexpr int kKt = Kernel::Ktile;
  static constexpr bool is_kTiling = TilesShape::Ndim >= 3;
  static constexpr bool is_Apacked = ALayout::Ndim > 2;
  static constexpr bool is_Bpacked = BLayout::Ndim > 2;
  static constexpr bool a_pack_otf = !is_Apacked;
  static constexpr bool b_pack_otf = !is_Bpacked;
  static_assert(!a_pack_otf || !is_Apacked,
      "A is already packed; pack-OTF for A makes no sense");
  static_assert(!b_pack_otf || !is_Bpacked,
      "B is already packed; pack-OTF for B makes no sense");
  static constexpr bool c_is_row_major = details::StrideDimConst1<CStride, 1>::value;
  static constexpr bool buffer_bypass = std::is_same_v<TC, TAcc> && c_is_row_major;

  static constexpr int ul_m = details::ShapeDim<std::remove_const_t<decltype(Kernel::shape_upper_left)>, 0>::value;
  static constexpr int ul_n = details::ShapeDim<std::remove_const_t<decltype(Kernel::shape_upper_left)>, 1>::value;
  static constexpr int ur_m = details::ShapeDim<std::remove_const_t<decltype(Kernel::shape_upper_right)>, 0>::value;
  static constexpr int ur_n = details::ShapeDim<std::remove_const_t<decltype(Kernel::shape_upper_right)>, 1>::value;
  static constexpr int ll_m = details::ShapeDim<std::remove_const_t<decltype(Kernel::shape_lower_left)>, 0>::value;
  static constexpr int ll_n = details::ShapeDim<std::remove_const_t<decltype(Kernel::shape_lower_left)>, 1>::value;
  static constexpr int lr_m = details::ShapeDim<std::remove_const_t<decltype(Kernel::shape_lower_right)>, 0>::value;
  static constexpr int lr_n = details::ShapeDim<std::remove_const_t<decltype(Kernel::shape_lower_right)>, 1>::value;
  static_assert(ul_m % lr_m == 0 && ur_m % lr_m == 0 && ll_m % lr_m == 0, "lr_m is not the smallest tile");
  static_assert(ul_n % lr_n == 0 && ur_n % lr_n == 0 && ll_n % lr_n == 0, "lr_n is not the smallest tile");

  static constexpr int ul_M = ul_m * kMt;
  static constexpr int ul_N = ul_n * kNt;
  static constexpr int ur_M = ur_m * kMt;
  static constexpr int ur_N = ur_n * kNt;
  static constexpr int ll_M = ll_m * kMt;
  static constexpr int ll_N = ll_n * kNt;
  static constexpr int lr_M = lr_m * kMt;
  static constexpr int lr_N = lr_n * kNt;

  VECOPS_ALWAYS_INLINE static void run(
      ProblemShape shape_MNK,
      const TA* A, ALayout A_layout,
      const TB* B, BLayout B_layout,
      TC* C, CLayout C_layout,
      TilesShape tiles_shape,
      Kernel kernel,
      [[maybe_unused]] Scheduler scheduler,
      const EpilogFn& fn)
  {
    int M = shape_MNK.template get<0>();
    int N = shape_MNK.template get<1>();
    int K = shape_MNK.template get<2>();

    int Mt = tiles_shape.template get<0>();
    int Nt = tiles_shape.template get<1>();
    int Kt = K;
    if constexpr (is_kTiling) {
      Kt = tiles_shape.template get<2>();
    }

    VECOPS_ASSERT(Mt % kMt == 0, "TilesShape M-tile must be divisible by Kernel::Mtile");
    VECOPS_ASSERT(Nt % kNt == 0, "TilesShape N-tile must be divisible by Kernel::Ntile");

    if constexpr (is_Apacked || is_Bpacked) {
      VECOPS_ASSERT(Mt % kMt == 0, "Mt must be divisible by Kernel Mtile");
      VECOPS_ASSERT(Nt % kNt == 0, "Nt must be divisible by Kernel Ntile");
      VECOPS_ASSERT(Kt % kKt == 0, "Kt must be multiple of Kernel Ktile");
    }

    nint_t sc_m = C_layout.stride().template get<0>();
    nint_t sc_n = C_layout.stride().template get<1>();

    TAcc* acc_buf = nullptr;
    if constexpr (!buffer_bypass) {
      nuint_t buf_elems = nuint_t(Mt) * nuint_t(Nt);
      constexpr nuint_t align = 64;
      nuint_t alloc_size = ((buf_elems * sizeof(TAcc) + align - 1) / align) * align;
      acc_buf = (TAcc*)std::aligned_alloc(align, alloc_size);
      VECOPS_ASSERT(acc_buf, "aligned_alloc failed");
    }

    TA* a_gather = nullptr;
    TB* b_gather = nullptr;
    TA* a_packed = nullptr;
    TB* b_packed = nullptr;

    nint_t sa_m = 0, sa_k = 0, sb_n = 0, sb_k = 0;

    if constexpr (!is_Apacked) {
      sa_m = A_layout.stride().template get<0>();
      sa_k = A_layout.stride().template get<1>();
      if constexpr (InA::needs_buffer()) {
        nuint_t sz = nuint_t(Mt) * nuint_t(Kt);
        a_gather = (TA*)std::aligned_alloc(64, sz * sizeof(TA));
        VECOPS_ASSERT(a_gather, "aligned_alloc A gather failed");
      }
    }
    if constexpr (!is_Bpacked) {
      sb_n = B_layout.stride().template get<0>();
      sb_k = B_layout.stride().template get<1>();
      if constexpr (InB::needs_buffer()) {
        nuint_t sz = nuint_t(Nt) * nuint_t(Kt);
        b_gather = (TB*)std::aligned_alloc(64, sz * sizeof(TB));
        VECOPS_ASSERT(b_gather, "aligned_alloc B gather failed");
      }
    }

    if constexpr (a_pack_otf) {
      nuint_t nK_pack = (nuint_t(K) + Kernel::Ktile - 1) / Kernel::Ktile;
      nuint_t sz = nuint_t(Mt) * nK_pack * Kernel::Ktile;
      a_packed = (TA*)std::aligned_alloc(64, sz * sizeof(TA));
      VECOPS_ASSERT(a_packed, "aligned_alloc A pack-OTF failed");
    }
    if constexpr (b_pack_otf) {
      nuint_t nNtiles = (nuint_t(N) + nuint_t(Nt) - 1) / nuint_t(Nt);
      nuint_t nK_pack = (nuint_t(K) + Kernel::Ktile - 1) / Kernel::Ktile;
      nuint_t nN_per_tile = nuint_t(Nt) / Kernel::Ntile;
      nuint_t sz = nNtiles * nN_per_tile * nK_pack * Kernel::Ntile * Kernel::Ktile;
      b_packed = (TB*)std::aligned_alloc(64, sz * sizeof(TB));
      VECOPS_ASSERT(b_packed, "aligned_alloc B pack-OTF failed");
    }

    nint_t a_step_M = is_Apacked ? (Mt / kMt) * A_layout.stride().template get<0>() : Mt * sa_m;
    nint_t a_step_K = a_pack_otf ? 0
                     : is_Apacked ? (Kt / kKt) * A_layout.stride().template get<1>()
                     : Kt * sa_k;
    nint_t b_step_N = is_Bpacked ? (Nt / kNt) * B_layout.stride().template get<0>() : Nt * sb_n;
    nint_t b_step_K = b_pack_otf ? 0
                     : is_Bpacked ? (Kt / kKt) * B_layout.stride().template get<1>()
                     : Kt * sb_k;

    L2::run(M, N, K, Mt, Nt, Kt,
            A, A_layout, sa_m, sa_k,
            a_step_M, a_step_K, a_gather, a_packed,
            B, B_layout, sb_n, sb_k,
            b_step_N, b_step_K, b_gather, b_packed,
            C, C_layout, sc_m, sc_n,
            acc_buf, kernel, fn);

    if (acc_buf) std::free(acc_buf);
    if (a_gather) std::free(a_gather);
    if (b_gather) std::free(b_gather);
    if (a_packed) std::free(a_packed);
    if (b_packed) std::free(b_packed);
  }
};

// ============================================================================
// Gemm entry point
// ============================================================================

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
  GemmOrchestrator<ProblemShape, TA, ALayout, TB, BLayout, TC, CLayout,
                   TilesShape, Kernel, Scheduler, EpilogFn>
      ::run(shape_MNK, A, A_layout, B, B_layout,
            C, C_layout, tiles_shape, kernel, scheduler, fn);
}

}

#endif //VECOPS_GEMM_H
