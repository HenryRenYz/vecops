//
// Created by renyz on 2026/5/9.
//

#ifndef VECOPS_GEMM_H
#define VECOPS_GEMM_H

#include <type_traits>
#include <algorithm>

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

  // 四种kernel大小，用于处理不同分段，Shape<Mtiled, Ntiled>代表M N轴方向一次计算的内核分块大小，即实际kernel大小为(Mtiled*Mtile, Ntiled*Ntile)
  static constexpr Shape<Int<1>, Int<1>> shape_upper_left {}; // 左上角的块，最大的块，效率最高，M N轴均非边界情况
  static constexpr Shape<Int<1>, Int<1>> shape_upper_right {}; // 右上角的块，M轴非边界，N轴可能是边界
  static constexpr Shape<Int<1>, Int<1>> shape_lower_left {}; // 左下角的块，M轴可能是边界，N轴非边界
  static constexpr Shape<Int<1>, Int<1>> shape_lower_right {}; // 右下角的块，最小的块，M N轴均可能是边界

  /**
   * @param A array with layout ALayout
   * @param A_layout (* <= M_tile, K), stride (*, *), or PackedALayout of that
   * @param B array with layout BLayout
   * @param B_layout (* <= N_tile, K), stride (*, *), or PackedBLayout of that
   * @param C array with layout CLayout
   * @param C_layout (* <= M_tile, * <= N_tile), stride (*, *)
   * @param fn epilog fn, signature
   *        vec::Vec<vec::ScalableTag>(int m, int n, vec::Vec<vec::ScalableTag>),
   *            where m, n is the local offset in 0:M_tile, 0:N_tile
   * @param accumulate if true, accumulate partial sum into C (used for K-tiling)
   * @param accumulator_buffer optional buffer for partial accumulation across K tiles
   */
  template <
      typename TA, typename ALayout,
      typename TB, typename BLayout,
      typename TC, typename CLayout,
      typename EpilogFn
  >
  void run(
      const TA * A, ALayout A_layout,
      const TB * B, BLayout B_layout,
      TC * C, CLayout C_layout,
      const EpilogFn& fn,
      bool accumulate = false,
      TAccumulator * accumulator_buffer = nullptr
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

template <
    typename ProblemShape,
    typename TA, typename ALayout,
    typename TB, typename BLayout,
    typename TC, typename CLayout,
    typename Kernel,
    typename Scheduler,
    typename EpilogFn
>
void gemm(
    ProblemShape shape_MNK,
    const TA * A, ALayout A_layout,
    const TB * B, BLayout B_layout,
    TC * C, CLayout C_layout,
    Kernel kernel,
    [[maybe_unused]] Scheduler scheduler,
    const EpilogFn& fn
) {
  int M = shape_MNK.template get<0>();
  int N = shape_MNK.template get<1>();
  int K = shape_MNK.template get<2>();

  constexpr int kMt = Kernel::Mtile;
  constexpr int kNt = Kernel::Ntile;

  constexpr bool is_Atiled = ALayout::Ndim > 2;
  constexpr bool is_Btiled = BLayout::Ndim > 2;

  if constexpr (is_Atiled || is_Btiled) {
    VECOPS_ASSERT(false, "Pre-packed A/B not yet implemented");
  } else {
    int sa_m = A_layout.stride().template get<0>();
    int sa_k = A_layout.stride().template get<1>();
    int sb_n = B_layout.stride().template get<0>();
    int sb_k = B_layout.stride().template get<1>();
    int sc_m = C_layout.stride().template get<0>();
    int sc_n = C_layout.stride().template get<1>();

    using AShape = typename ALayout::Shape;
    using BShape = typename BLayout::Shape;
    using CShape = typename CLayout::Shape;
    using AKDim  = typename details::KDimType<AShape>::type;
    using BKDim  = typename details::KDimType<BShape>::type;

    auto call_tile = [&]<typename MType, typename NType>(
        int m_off, int n_off, int tile_m, int tile_n)
    {
      using SubA = details::SubShape2D<AShape, MType, AKDim>;
      using SubB = details::SubShape2D<BShape, NType, BKDim>;
      using SubC = details::SubShape2D<CShape, MType, NType>;

      auto sub_a_shape = SubA::create(tile_m, K);
      auto sub_b_shape = SubB::create(tile_n, K);
      auto sub_c_shape = SubC::create(tile_m, tile_n);

      auto sub_a_layout = make_layout(sub_a_shape, A_layout.stride());
      auto sub_b_layout = make_layout(sub_b_shape, B_layout.stride());
      auto sub_c_layout = make_layout(sub_c_shape, C_layout.stride());

      const TA * a_ptr = A + m_off * sa_m;
      const TB * b_ptr = B + n_off * sb_n;
      TC * c_ptr = C + m_off * sc_m + n_off * sc_n;

      auto epilog_global = [&](int m_local, int n_local, auto val) {
        return fn(m_off + m_local, n_off + n_local, val);
      };

      kernel.run(a_ptr, sub_a_layout, b_ptr, sub_b_layout,
                 c_ptr, sub_c_layout, epilog_global);
    };

    constexpr int ul_m = decltype(Kernel::shape_upper_left){}.template get<0>();
    constexpr int ul_n = decltype(Kernel::shape_upper_left){}.template get<1>();
    constexpr int ur_m = decltype(Kernel::shape_upper_right){}.template get<0>();
    constexpr int ur_n = decltype(Kernel::shape_upper_right){}.template get<1>();
    constexpr int ll_m = decltype(Kernel::shape_lower_left){}.template get<0>();
    constexpr int ll_n = decltype(Kernel::shape_lower_left){}.template get<1>();
    constexpr int lr_m = decltype(Kernel::shape_lower_right){}.template get<0>();
    constexpr int lr_n = decltype(Kernel::shape_lower_right){}.template get<1>();
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

    int m1 = (M / ul_M) * ul_M;
    int n1 = (N / ul_N) * ul_N;
    int m2 = (M / ur_M) * ur_M;
    int n2 = (N / ll_N) * ll_N;

    if constexpr (std::is_same_v<Scheduler, SchedulerMaxCases>) {
      // ┌────────────────┬────────────┐
      // │ 1. UL          │ 2. UR      │
      // │ Int<M,N>       │ Int<M>,Any │
      // │                ├────────────┤
      // │                │ 4.LR-vert  │
      // │                │ Int<M>,Any │
      // ├─────────────┬──┴────────────┤
      // │ 3. LL       │ 5.LR-bottom   │
      // │ Any,Int<N>  │ Any,Any       │
      // └─────────────┴───────────────┘

      // 1. M [0, m1), N [0, n1)
      for (int m = 0; m < m1; m += ul_M) {
        for (int n = 0; n < n1; n += ul_N) {
          call_tile.template operator()<Int<ul_M>, Int<ul_N>>(m, n, ul_M, ul_N);
        }
      }

      // 2. M [0, m2), N [n1, N)
      for (int n = n1; n < N; n += ur_N) {
        int tile_n = std::min(ur_N, N - n);
        for (int m = 0; m < m2; m += ur_M) {
          call_tile.template operator()<Int<ur_M>, Any>(m, n, ur_M, tile_n);
        }

      }

      // 4. M [m2, m1), N [n1, N)
      for (int n = n1; n < N; n += lr_N) {
        int tile_n = std::min(lr_N, N - n);
        for (int m = m2; m < m1; m += lr_M) {
          call_tile.template operator()<Int<lr_M>, Any>(m, n, lr_M, tile_n);
        }
      }

      // 3. M [m1, M), N [0, n2)
      for (int m = m1; m < M; m += ll_M) {
        int tile_m = std::min(ll_M, M - m);
        for (int n = 0; n < n2; n += ll_N) {
          call_tile.template operator()<Any, Int<ll_N>>(m, n, tile_m, ll_N);
        }
      }

      // 5. M [m1, M), N [n2, N)
      for (int m = m1; m < M; m += lr_M) {
        int tile_m = std::min(lr_M, M - m);
        for (int n = n2; n < N; n += lr_N) {
          int tile_n = std::min(lr_N, N - n);
          call_tile.template operator()<Any, Any>(m, n, tile_m, tile_n);
        }
      }
    } else {
      static_assert(std::is_same_v<Scheduler, SchedulerMinCases>,
                    "Unknown Scheduler type");
      // ┌────────────────┬────────────┐
      // │ 1. UL          │ 2. LR-vert │
      // │ Int<M,N>       │ Any,Any    │
      // ├────────────────┴────────────┤
      // │ 3.LR-bottom                 │
      // │ Any,Any                     │
      // └─────────────────────────────┘

      // 1. M [0, m1), N [0, n1)
      for (int m = 0; m < m1; m += ul_M) {
        for (int n = 0; n < n1; n += ul_N) {
          call_tile.template operator()<Int<ul_M>, Int<ul_N>>(m, n, ul_M, ul_N);
        }
      }

      // 2. M [0, m1), N [n1, N)
      for (int m = 0; m < m1; m += lr_M) {
        for (int n = n1; n < N; n += lr_N) {
          int tile_n = std::min(lr_N, N - n);
          call_tile.template operator()<Any, Any>(m, n, lr_M, tile_n);
        }
      }

      // 3. M [m1, M), N [0, N)
      for (int m = m1; m < M; m += lr_M) {
        int tile_m = std::min(lr_M, M - m);
        for (int n = 0; n < N; n += lr_N) {
          int tile_n = std::min(lr_N, N - n);
          call_tile.template operator()<Any, Any>(m, n, tile_m, tile_n);
        }
      }
    }
  }
}

}

#endif //VECOPS_GEMM_H
