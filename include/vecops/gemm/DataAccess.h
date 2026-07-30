//
// Created by renyz on 2026/7/8.
//

#ifndef VECOPS_DATAACCESS_H
#define VECOPS_DATAACCESS_H

#include "vecops/VecTransform.h"
#include "vecops/util/TypeTraits.h"
#include "vecops/vec/Vec.h"
#include "./Workspace.h"
#include "./Tensor.h"
#include "./HOP.h"

#include <algorithm>
#include <array>
#include <cstddef>
#include <tuple>
#include <type_traits>
#include <utility>

namespace vecops::gemm {

/**
 * Marks a data-access request whose vector is known to be completely active.
 *
 * Unlike a runtime count, this marker deliberately has no numeric conversion:
 * callers must establish the full-vector precondition structurally (normally
 * with a loop bound), and dispatch can then omit predicate construction even
 * for runtime-sized scalable vectors.
 */
struct FullVectorCount {};

namespace details {

struct AccessKindLastContiguous {};
struct AccessKindSecondLastContiguous {};
struct AccessKindStrided {};

template <typename Meta, int Pos, typename Pattern, typename Seq>
struct IsLenientAtImpl;

template <typename Meta, int Pos, typename Pattern, size_t... Idx>
struct IsLenientAtImpl<Meta, Pos, Pattern, std::index_sequence<Idx...>>
    : std::bool_constant<is_lenient_v<
          Meta,
          std::conditional_t<static_cast<int>(Idx) == Pos, Pattern, any>...>> {};

template <typename Meta, int Pos, typename Pattern>
struct IsLenientAt
    : IsLenientAtImpl<Meta, Pos, Pattern, std::make_index_sequence<Meta::Ndim>> {};

template <typename TLayout>
struct IsLastDimContiguous
    : IsLenientAt<typename std::remove_cvref_t<TLayout>::Strides,
                  std::remove_cvref_t<TLayout>::Ndim - 1,
                  Const<1>> {};

template <typename TLayout>
struct IsSecondLastDimContiguous
    : std::bool_constant<
          (std::remove_cvref_t<TLayout>::Ndim >= 2) &&
          IsLenientAt<typename std::remove_cvref_t<TLayout>::Strides,
                      std::remove_cvref_t<TLayout>::Ndim - 2,
                      Const<1>>::value> {};

template <typename TLayout>
using SelectAccessKind = chain_if_t<
    chain_opt<IsLastDimContiguous<TLayout>::value, AccessKindLastContiguous>,
    chain_opt<IsSecondLastDimContiguous<TLayout>::value, AccessKindSecondLastContiguous>,
    chain_opt<true, AccessKindStrided>>;

VECOPS_INLINE constexpr nint_t round_up(nint_t value, nint_t alignment) {
  return (value + alignment - 1) & ~(alignment - 1);
}

VECOPS_INLINE consteval int element_size_shift(
    std::size_t from, std::size_t to) {
  int shift = 0;
  while (from < to) {
    from *= 2;
    ++shift;
  }
  while (from > to) {
    to *= 2;
    --shift;
  }
  return shift;
}

template <typename T>
VECOPS_INLINE constexpr nint_t aux_vector_lanes() {
  return vec::size(vec::ScalableTag<T, VEC_MAX_POW>{});
}

template <typename TLayout, typename TAux>
VECOPS_INLINE nint_t padded_last_size(const TLayout& layout) {
  return round_up(layout.shape()[TLayout::Ndim - 1], aux_vector_lanes<TAux>());
}

template <typename TLayout, typename TAux>
VECOPS_INLINE nint_t aux_numel(const TLayout& layout) {
  nint_t n = padded_last_size<TLayout, TAux>(layout);
  VECOPS_UNROLL
  for (int d = 0; d < TLayout::Ndim - 1; ++d) n *= layout.shape()[d];
  return n;
}

template <typename TLayout, typename TAux>
VECOPS_INLINE nint_t aux_required_bytes(const TLayout& layout) {
  return round_up(aux_numel<TLayout, TAux>(layout) * static_cast<nint_t>(sizeof(TAux)),
                  vec::DEFAULT_ALIGNMENT);
}

template <typename TLayout, typename TAux>
VECOPS_INLINE auto make_aux_layout(const TLayout& layout) {
  constexpr int last_dim = TLayout::Ndim - 1;
  auto aux_shape = set<last_dim>(layout.shape(), Any{padded_last_size<TLayout, TAux>(layout)});
  return make_layout(aux_shape);
}

template <typename F, typename Tuple, std::size_t... I>
VECOPS_ALWAYS_INLINE constexpr decltype(auto) apply_impl(F&& f, Tuple&& t, std::index_sequence<I...>) {
  return std::forward<F>(f)(std::get<I>(std::forward<Tuple>(t))...);
}

// Note: std::apply may fail to inline at some point
template <typename F, typename Tuple>
VECOPS_ALWAYS_INLINE constexpr decltype(auto) apply_inline(F&& f, Tuple&& t) {
  return apply_impl(
      std::forward<F>(f), std::forward<Tuple>(t),
      std::make_index_sequence<
          std::tuple_size_v<std::decay_t<Tuple>>>{}
  );
}

template <typename TransformFn, typename To, typename VIn, typename PrefixTuple>
VECOPS_ALWAYS_INLINE auto call_transform_with_last_coord(
    const TransformFn& fn,
    To to,
    VIn v_in,
    const PrefixTuple& prefix,
    nint_t last) {
  return apply_inline(
      [&](auto... prefix_coords) VECOPS_INLINE_LAMBDA {
        return fn(to, v_in, prefix_coords..., last);
      },
      prefix);
}

template <typename TransformFn, typename PrefixTuple>
struct PrefixVecTransform {
  using TIn = typename TransformFn::TIn;
  using TOut = typename TransformFn::TOut;

  TransformFn fn;
  PrefixTuple prefix;

  template <typename To, typename VIn, typename... Is>
  VECOPS_ALWAYS_INLINE auto operator()(To to, VIn v_in, Is... is) const {
    return apply_inline(
        [&](auto... prefix_coords) VECOPS_INLINE_LAMBDA {
          return fn(to, v_in, prefix_coords..., static_cast<nint_t>(is)...);
        },
        prefix);
  }
};

template <typename TransformFn, typename PrefixTuple>
VECOPS_INLINE auto prepend_transform_prefix(const TransformFn& fn, const PrefixTuple& prefix) {
  return PrefixVecTransform<TransformFn, PrefixTuple>{fn, prefix};
}

template <typename N>
struct StaticCount : std::false_type {};

template <nint_t N>
struct StaticCount<Const<N>> : std::true_type {
  static constexpr nint_t value = N;
};

template <typename N>
static constexpr bool is_static_count_v = StaticCount<std::remove_cvref_t<N>>::value;

template <typename N, typename T>
struct UseUnmaskedPath : std::false_type {};

template <nint_t N, vec::VectorTag T>
struct UseUnmaskedPath<Const<N>, T> : std::bool_constant<[] {
  if constexpr (vec::is_runtime_size<T>) {
    return false;
  } else {
    return N >= vec::native_word_size(T{}) * vec::num_words(T{});
  }
}()> {};

template <vec::VectorTag T>
struct UseUnmaskedPath<FullVectorCount, T> : std::true_type {};

template <typename N>
VECOPS_ALWAYS_INLINE constexpr nint_t count_value(N n) {
  if constexpr (is_static_count_v<N>) {
    return StaticCount<std::remove_cvref_t<N>>::value;
  } else {
    return static_cast<nint_t>(n);
  }
}

template <typename N, vec::VectorTag T>
static constexpr bool use_unmasked_path_v = UseUnmaskedPath<std::remove_cvref_t<N>, T>::value;

template <typename N, vec::VectorTag T>
VECOPS_ALWAYS_INLINE vec::Vec<T> load_dispatch(
    T t, const vec::ElementOf<T>* p, N n) {
  if constexpr (use_unmasked_path_v<N, T>) {
    return vec::load(t, p);
  } else {
    return vec::load(t, p, vec::opt::first(count_value(n)));
  }
}

template <typename N, vec::VectorTag T>
VECOPS_ALWAYS_INLINE vec::Vec<T> zero_inactive_lanes(
    T t, vec::Vec<T> v, N n) {
  if constexpr (use_unmasked_path_v<N, T>) {
    return v;
  } else {
    return vec::blend(
        t, vec::zeros(t), vec::mwhilelt(t, 0, count_value(n)), v);
  }
}

template <typename N, vec::VectorTag T>
VECOPS_INLINE void store_dispatch(
    T t, vec::ElementOf<T>* p, N n, vec::Vec<T> v) {
  if constexpr (use_unmasked_path_v<N, T>) {
    vec::store(t, p, v);
  } else {
    vec::store(t, p, v, vec::opt::first(count_value(n)));
  }
}

template <typename T>
using MemoryIndex = std::conditional_t<
    (sizeof(T) < sizeof(int64_t)), int32_t, int64_t>;

template <typename T, vec::VectorTag Ti>
VECOPS_INLINE vec::Vec<vec::Rebind<MemoryIndex<T>, Ti>>
make_index_vector(
    Ti ti,
    nint_t base_offset,
    nint_t stride) {
  using IndexTag = vec::Rebind<MemoryIndex<T>, Ti>;
  using Index = vec::ElementOf<IndexTag>;
  IndexTag it;
  auto idx = vec::fill(it, static_cast<Index>(base_offset));
  for (nint_t lane = 0; lane < vec::size(it); ++lane) {
    idx = vec::set(
        it, idx, lane, static_cast<Index>(base_offset + lane * stride));
  }
  return idx;
}

template <typename T, vec::VectorTag Ti>
static constexpr bool can_materialize_memory_index_v = [] {
#if defined(CPU_CAPABILITY_SVE)
  using IndexTag = vec::Rebind<MemoryIndex<T>, Ti>;
  if constexpr (vec::is_scalable_tag<IndexTag>) {
    return vec::scale_power<IndexTag> <= VEC_MAX_POW;
  } else {
    return true;
  }
#else
  return true;
#endif
}();

template <typename N, typename T, vec::VectorTag Ti>
VECOPS_ALWAYS_INLINE vec::Vec<Ti> indexed_load_leaf(
    Ti ti,
    const T* p,
    nint_t base_offset,
    nint_t stride,
    N n) {
  auto idx = make_index_vector<T>(ti, base_offset, stride);
  if constexpr (use_unmasked_path_v<N, Ti>) {
    return vec::load(ti, p, vec::indexed(idx));
  } else {
    const auto mask = vec::mwhilelt(ti, 0, count_value(n));
    return vec::load(
        ti, p, vec::indexed(idx), vec::opt::masked(mask),
        vec::opt::zero);
  }
}

template <typename N, typename T, vec::VectorTag Ti>
VECOPS_ALWAYS_INLINE void indexed_store_leaf(
    Ti ti,
    T* p,
    nint_t base_offset,
    nint_t stride,
    N n,
    vec::Vec<Ti> v) {
  auto idx = make_index_vector<T>(ti, base_offset, stride);
  if constexpr (use_unmasked_path_v<N, Ti>) {
    vec::store(ti, p, v, vec::indexed(idx));
  } else {
    const auto mask = vec::mwhilelt(ti, 0, count_value(n));
    vec::store(ti, p, v, vec::indexed(idx), vec::opt::masked(mask));
  }
}

template <typename N, typename T, vec::VectorTag Ti>
VECOPS_INLINE vec::Vec<Ti> indexed_load_dispatch(
    Ti ti,
    const T* p,
    nint_t base_offset,
    nint_t stride,
    N n) {
  if constexpr (!can_materialize_memory_index_v<T, Ti>) {
    // SVE supports up to x4 tuples. A narrow data vector may need a wider
    // i32/i64 index tuple, so split before instantiating the index tag.
    using Th = vec::Half<Ti>;
    Th th;
    const nint_t half_size = vec::size(th);
    if constexpr (use_unmasked_path_v<N, Ti>) {
      auto lo = indexed_load_dispatch(th, p, base_offset, stride, n);
      auto hi = indexed_load_dispatch(
          th, p, base_offset + half_size * stride, stride, n);
      return vec::concat(ti, lo, hi);
    } else {
      const nint_t count = count_value(n);
      auto lo = indexed_load_dispatch(th, p, base_offset, stride, count);
      auto hi = indexed_load_dispatch(
          th, p, base_offset + half_size * stride, stride,
          count - half_size);
      return vec::concat(ti, lo, hi);
    }
  } else {
    return indexed_load_leaf(ti, p, base_offset, stride, n);
  }
}

template <typename N, typename T, vec::VectorTag Ti>
VECOPS_INLINE void indexed_store_dispatch(
    Ti ti,
    T* p,
    nint_t base_offset,
    nint_t stride,
    N n,
    vec::Vec<Ti> v) {
  if constexpr (!can_materialize_memory_index_v<T, Ti>) {
    // Keep the leaf path from forming an unsupported SVE index tuple.
    using Th = vec::Half<Ti>;
    Th th;
    const nint_t half_size = vec::size(th);
    if constexpr (use_unmasked_path_v<N, Ti>) {
      indexed_store_dispatch(
          th, p, base_offset, stride, n, vec::lower(ti, v));
      indexed_store_dispatch(
          th, p, base_offset + half_size * stride, stride, n,
          vec::upper(ti, v));
    } else {
      const nint_t count = count_value(n);
      indexed_store_dispatch(
          th, p, base_offset, stride, count, vec::lower(ti, v));
      indexed_store_dispatch(
          th, p, base_offset + half_size * stride, stride,
          count - half_size, vec::upper(ti, v));
    }
    return;
  } else {
    indexed_store_leaf(ti, p, base_offset, stride, n, v);
  }
}

template <typename TOut, typename TIn, typename InLayout, typename AuxLayout, typename TransformFn>
VECOPS_INLINE void precompute_input_aux(
    const TIn* p,
    const InLayout& layout,
    const AuxLayout& aux_layout,
    const TransformFn& fn,
  TOut* aux) {
  std::fill(aux, aux + aux_numel<InLayout, TOut>(layout), TOut{});
  constexpr int input_pow_shift =
      element_size_shift(sizeof(TOut), sizeof(TIn));
  constexpr int output_pow = input_pow_shift > VEC_MAX_POW
      ? VEC_MAX_POW - input_pow_shift
      : 0;
  using To = vec::ScalableTag<TOut, output_pow>;
  using Ti = vec::Rebind<TIn, To>;
  To to;
  Ti ti;
  const auto last = size<InLayout::Ndim - 1>(layout);
  auto src_tensor = make_tensor(p, layout);
  auto aux_tensor = make_tensor(aux, aux_layout);

  hop::for_each_dims_with_index_tuple<InLayout::Ndim - 1>(
      [&](const auto& prefix, auto&& src_row, auto&& aux_row) VECOPS_INLINE_LAMBDA {
        const nint_t stride = src_row.stride(0);
        hop::map(last, vec::size(to), [&](nint_t i, auto n) {
          auto v_in = indexed_load_dispatch(
              ti, src_row.data(), i * stride, stride, Any{n});
          auto v_out = call_transform_with_last_coord(fn, to, v_in, prefix, i);
          store_dispatch(to, aux_row.data() + i, n, v_out);
        });
      },
      src_tensor,
      aux_tensor);
}

template <typename Kind, typename TOut, typename InTensor, typename TransformFn>
struct DataInputImpl;

template <typename TOut, typename InTensor, typename TransformFn>
struct DataInputImpl<AccessKindLastContiguous, TOut, InTensor, TransformFn> {
  using TIn = typename InTensor::ElementType;
  using InLayout = typename InTensor::Layout;

  VECOPS_INLINE static constexpr nint_t required_workspace(const InLayout&) { return 0; }

  VECOPS_INLINE DataInputImpl(const TIn* p, const InLayout& layout, const TransformFn& fn, void*)
      : _p(p), _layout(layout), _fn(fn) {}

  template <vec::VectorTag To, typename N, typename... Is>
  VECOPS_INLINE vec::Vec<To> operator()(To t, N n, Is... is) const {
    static_assert(sizeof...(Is) == InLayout::Ndim, "coordinate count must match input rank");
    using Ti = vec::Rebind<TIn, To>;
    std::array<nint_t, InLayout::Ndim> coords{static_cast<nint_t>(is)...};
    const nint_t offset = offset_at(_layout, coords);
    auto v_in = load_dispatch(Ti{}, _p + offset, n);
    auto v_out = _fn(t, v_in, static_cast<nint_t>(is)...);
    return zero_inactive_lanes(t, v_out, n);
  }

  const TIn* _p;
  InLayout _layout;
  TransformFn _fn;
};

template <typename Kind, typename TOut, typename InTensor, typename TransformFn>
struct DataInputAuxImpl {
  using TIn = typename InTensor::ElementType;
  using InLayout = typename InTensor::Layout;
  using AuxLayout = decltype(make_aux_layout<InLayout, TOut>(std::declval<const InLayout&>()));

  VECOPS_INLINE static nint_t required_workspace(const InLayout& layout) {
    return aux_required_bytes<InLayout, TOut>(layout);
  }

  VECOPS_INLINE DataInputAuxImpl(const TIn* p, const InLayout& layout, const TransformFn& fn, void* aux)
      : _p(p),
        _layout(layout),
        _aux_layout(make_aux_layout<InLayout, TOut>(layout)),
        _aux(static_cast<TOut*>(aux)) {
    VECOPS_ASSERT(aux != nullptr, "DataInput aux buffer is required for non-last-contiguous layouts");
    precompute_input_aux<TOut>(p, _layout, _aux_layout, fn, _aux);
  }

  template <vec::VectorTag To, typename N, typename... Is>
  VECOPS_INLINE vec::Vec<To> operator()(To t, N n, Is... is) const {
    static_assert(sizeof...(Is) == InLayout::Ndim, "coordinate count must match input rank");
    std::array<nint_t, InLayout::Ndim> coords{static_cast<nint_t>(is)...};
    const nint_t offset = offset_at(_aux_layout, coords);
    auto v = load_dispatch(t, _aux + offset, n);
    return v;
  }

  const TIn* _p;
  InLayout _layout;
  AuxLayout _aux_layout;
  TOut* _aux;
};

template <typename TOut, typename InTensor, typename TransformFn>
struct DataInputImpl<AccessKindSecondLastContiguous, TOut, InTensor, TransformFn>
    : DataInputAuxImpl<AccessKindSecondLastContiguous, TOut, InTensor, TransformFn> {
  using DataInputAuxImpl<AccessKindSecondLastContiguous, TOut, InTensor, TransformFn>::DataInputAuxImpl;
};

template <typename TOut, typename InTensor, typename TransformFn>
struct DataInputImpl<AccessKindStrided, TOut, InTensor, TransformFn>
    : DataInputAuxImpl<AccessKindStrided, TOut, InTensor, TransformFn> {
  using DataInputAuxImpl<AccessKindStrided, TOut, InTensor, TransformFn>::DataInputAuxImpl;
};

template <typename Kind, typename TIn, typename OutTensor, typename TransformFn>
struct DataOutputImpl;

template <typename TIn, typename OutTensor, typename TransformFn>
struct DataOutputImpl<AccessKindLastContiguous, TIn, OutTensor, TransformFn> {
  using TOut = typename OutTensor::ElementType;
  using OutLayout = typename OutTensor::Layout;

  VECOPS_INLINE static constexpr nint_t required_workspace(const OutLayout&) { return 0; }

  VECOPS_INLINE DataOutputImpl(TOut* p, const OutLayout& layout, const TransformFn& fn, void*)
      : _p(p), _layout(layout), _fn(fn) {}

  template <vec::VectorTag Ti, typename N, typename... Is>
  VECOPS_INLINE void operator()(Ti ti, vec::Vec<Ti> v, N n, Is... is) const {
    static_assert(sizeof...(Is) == OutLayout::Ndim, "coordinate count must match output rank");
    using To = vec::Rebind<TOut, Ti>;
    std::array<nint_t, OutLayout::Ndim> coords{static_cast<nint_t>(is)...};
    const nint_t offset = offset_at(_layout, coords);
    auto v_out = _fn(To{}, v, static_cast<nint_t>(is)...);
    store_dispatch(To{}, _p + offset, n, v_out);
  }

  TOut* _p;
  OutLayout _layout;
  TransformFn _fn;
};

template <typename TIn, typename OutTensor, typename TransformFn>
struct DataOutputImpl<AccessKindSecondLastContiguous, TIn, OutTensor, TransformFn> {
  using TOut = typename OutTensor::ElementType;
  using OutLayout = typename OutTensor::Layout;
  using AuxLayout = decltype(make_aux_layout<OutLayout, TOut>(std::declval<const OutLayout&>()));

  VECOPS_INLINE static nint_t required_workspace(const OutLayout& layout) {
    return aux_required_bytes<OutLayout, TOut>(layout);
  }

  VECOPS_INLINE DataOutputImpl(TOut* p, const OutLayout& layout, const TransformFn& fn, void* aux)
      : _p(p),
        _layout(layout),
        _aux_layout(make_aux_layout<OutLayout, TOut>(layout)),
        _fn(fn),
        _aux(static_cast<TOut*>(aux)) {
    VECOPS_ASSERT(aux != nullptr, "DataOutput aux buffer is required for second-last-contiguous layouts");
    std::fill(_aux, _aux + aux_numel<OutLayout, TOut>(_layout), TOut{});
  }

  VECOPS_INLINE ~DataOutputImpl() {
    auto out_tensor = make_tensor(_p, _layout);
    auto aux_tensor = make_tensor(_aux, _aux_layout);
    hop::for_each_dims_with_index_tuple<OutLayout::Ndim - 1>(
        [](const auto&, auto&& out_row, auto&& aux_row) VECOPS_INLINE_LAMBDA {
          const nint_t last = out_row.size(0);
          for (nint_t y = 0; y < last; ++y) {
            out_row(y) = aux_row(y);
          }
        },
        out_tensor,
        aux_tensor);
  }

  template <vec::VectorTag Ti, typename N, typename... Is>
  VECOPS_INLINE void operator()(Ti ti, vec::Vec<Ti> v, N n, Is... is) const {
    static_assert(sizeof...(Is) == OutLayout::Ndim, "coordinate count must match output rank");
    using To = vec::Rebind<TOut, Ti>;
    std::array<nint_t, OutLayout::Ndim> coords{static_cast<nint_t>(is)...};
    auto v_out = _fn(To{}, v, static_cast<nint_t>(is)...);
    const nint_t offset = offset_at(_aux_layout, coords);
    store_dispatch(To{}, _aux + offset, n, v_out);
  }

  TOut* _p;
  OutLayout _layout;
  AuxLayout _aux_layout;
  TransformFn _fn;
  TOut* _aux;
};

template <typename TIn, typename OutTensor, typename TransformFn>
struct DataOutputImpl<AccessKindStrided, TIn, OutTensor, TransformFn> {
  using TOut = typename OutTensor::ElementType;
  using OutLayout = typename OutTensor::Layout;

  VECOPS_INLINE static constexpr nint_t required_workspace(const OutLayout&) { return 0; }

  VECOPS_INLINE DataOutputImpl(TOut* p, const OutLayout& layout, const TransformFn& fn, void*)
      : _p(p), _layout(layout), _fn(fn) {}

  template <vec::VectorTag Ti, typename N, typename... Is>
  VECOPS_INLINE void operator()(Ti ti, vec::Vec<Ti> v, N n, Is... is) const {
    static_assert(sizeof...(Is) == OutLayout::Ndim, "coordinate count must match output rank");
    using To = vec::Rebind<TOut, Ti>;
    std::array<nint_t, OutLayout::Ndim> coords{static_cast<nint_t>(is)...};
    const nint_t base = offset_at(_layout, coords);
    auto v_out = _fn(To{}, v, static_cast<nint_t>(is)...);
    indexed_store_dispatch(
        To{}, _p, base, _layout.strides()[OutLayout::Ndim - 1], n, v_out);
  }

  TOut* _p;
  OutLayout _layout;
  TransformFn _fn;
};

template <
    typename Kind,
    typename TOut,
    typename InTensor,
    typename TransformFn,
    typename AuxLayout = void>
struct DataInputViewImpl;

template <typename TOut, typename InTensor, typename TransformFn>
struct DataInputViewImpl<AccessKindLastContiguous, TOut, InTensor, TransformFn, void> {
  using TIn = typename InTensor::ElementType;
  using InLayout = typename InTensor::Layout;

  VECOPS_INLINE DataInputViewImpl(const TIn* p, const InLayout& layout, const TransformFn& fn)
      : _p(p), _layout(layout), _fn(fn) {}

  template <vec::VectorTag To, typename N, typename... Is>
  VECOPS_INLINE vec::Vec<To> operator()(To t, N n, Is... is) const {
    static_assert(sizeof...(Is) == InLayout::Ndim, "coordinate count must match input rank");
    using Ti = vec::Rebind<TIn, To>;
    std::array<nint_t, InLayout::Ndim> coords{static_cast<nint_t>(is)...};
    const nint_t offset = offset_at(_layout, coords);
    auto v_in = load_dispatch(Ti{}, _p + offset, n);
    auto v_out = _fn(t, v_in, static_cast<nint_t>(is)...);
    return zero_inactive_lanes(t, v_out, n);
  }

  const TIn* _p;
  InLayout _layout;
  TransformFn _fn;
};

template <typename Kind, typename TOut, typename InTensor, typename TransformFn, typename AuxLayout>
struct DataInputAuxViewImpl {
  using TIn = typename InTensor::ElementType;
  using InLayout = typename InTensor::Layout;

  VECOPS_INLINE DataInputAuxViewImpl(const TIn* p, const InLayout& layout, const AuxLayout& aux_layout, TOut* aux)
      : _p(p), _layout(layout), _aux_layout(aux_layout), _aux(aux) {}

  template <vec::VectorTag To, typename N, typename... Is>
  VECOPS_INLINE vec::Vec<To> operator()(To t, N n, Is... is) const {
    static_assert(sizeof...(Is) == InLayout::Ndim, "coordinate count must match input rank");
    std::array<nint_t, InLayout::Ndim> coords{static_cast<nint_t>(is)...};
    const nint_t offset = offset_at(_aux_layout, coords);
    return load_dispatch(t, _aux + offset, n);
  }

  const TIn* _p;
  InLayout _layout;
  AuxLayout _aux_layout;
  TOut* _aux;
};

template <typename TOut, typename InTensor, typename TransformFn, typename AuxLayout>
struct DataInputViewImpl<AccessKindSecondLastContiguous, TOut, InTensor, TransformFn, AuxLayout>
    : DataInputAuxViewImpl<AccessKindSecondLastContiguous, TOut, InTensor, TransformFn, AuxLayout> {
  using DataInputAuxViewImpl<
      AccessKindSecondLastContiguous,
      TOut,
      InTensor,
      TransformFn,
      AuxLayout>::DataInputAuxViewImpl;
};

template <typename TOut, typename InTensor, typename TransformFn, typename AuxLayout>
struct DataInputViewImpl<AccessKindStrided, TOut, InTensor, TransformFn, AuxLayout>
    : DataInputAuxViewImpl<AccessKindStrided, TOut, InTensor, TransformFn, AuxLayout> {
  using DataInputAuxViewImpl<
      AccessKindStrided,
      TOut,
      InTensor,
      TransformFn,
      AuxLayout>::DataInputAuxViewImpl;
};

template <
    typename Kind,
    typename TIn,
    typename OutTensor,
    typename TransformFn,
    typename AuxLayout = void>
struct DataOutputViewImpl;

template <typename TIn, typename OutTensor, typename TransformFn>
struct DataOutputViewImpl<AccessKindLastContiguous, TIn, OutTensor, TransformFn, void> {
  using TOut = typename OutTensor::ElementType;
  using OutLayout = typename OutTensor::Layout;

  VECOPS_INLINE DataOutputViewImpl(TOut* p, const OutLayout& layout, const TransformFn& fn)
      : _p(p), _layout(layout), _fn(fn) {}

  template <vec::VectorTag Ti, typename N, typename... Is>
  VECOPS_INLINE void operator()(Ti ti, vec::Vec<Ti> v, N n, Is... is) const {
    static_assert(sizeof...(Is) == OutLayout::Ndim, "coordinate count must match output rank");
    using To = vec::Rebind<TOut, Ti>;
    std::array<nint_t, OutLayout::Ndim> coords{static_cast<nint_t>(is)...};
    const nint_t offset = offset_at(_layout, coords);
    auto v_out = _fn(To{}, v, static_cast<nint_t>(is)...);
    store_dispatch(To{}, _p + offset, n, v_out);
  }

  TOut* _p;
  OutLayout _layout;
  TransformFn _fn;
};

template <typename TIn, typename OutTensor, typename TransformFn, typename AuxLayout>
struct DataOutputViewImpl<
    AccessKindSecondLastContiguous,
    TIn,
    OutTensor,
    TransformFn,
    AuxLayout> {
  using TOut = typename OutTensor::ElementType;
  using OutLayout = typename OutTensor::Layout;

  VECOPS_INLINE DataOutputViewImpl(TOut* p, const OutLayout& layout, const AuxLayout& aux_layout, const TransformFn& fn, TOut* aux)
      : _p(p), _layout(layout), _aux_layout(aux_layout), _fn(fn), _aux(aux) {}

  template <vec::VectorTag Ti, typename N, typename... Is>
  VECOPS_INLINE void operator()(Ti ti, vec::Vec<Ti> v, N n, Is... is) const {
    static_assert(sizeof...(Is) == OutLayout::Ndim, "coordinate count must match output rank");
    using To = vec::Rebind<TOut, Ti>;
    std::array<nint_t, OutLayout::Ndim> coords{static_cast<nint_t>(is)...};
    auto v_out = _fn(To{}, v, static_cast<nint_t>(is)...);
    const nint_t offset = offset_at(_aux_layout, coords);
    store_dispatch(To{}, _aux + offset, n, v_out);
  }

  TOut* _p;
  OutLayout _layout;
  AuxLayout _aux_layout;
  TransformFn _fn;
  TOut* _aux;
};

template <typename TIn, typename OutTensor, typename TransformFn>
struct DataOutputViewImpl<AccessKindStrided, TIn, OutTensor, TransformFn, void> {
  using TOut = typename OutTensor::ElementType;
  using OutLayout = typename OutTensor::Layout;

  VECOPS_INLINE DataOutputViewImpl(TOut* p, const OutLayout& layout, const TransformFn& fn)
      : _p(p), _layout(layout), _fn(fn) {}

  template <vec::VectorTag Ti, typename N, typename... Is>
  VECOPS_INLINE void operator()(Ti ti, vec::Vec<Ti> v, N n, Is... is) const {
    static_assert(sizeof...(Is) == OutLayout::Ndim, "coordinate count must match output rank");
    using To = vec::Rebind<TOut, Ti>;
    std::array<nint_t, OutLayout::Ndim> coords{static_cast<nint_t>(is)...};
    const nint_t base = offset_at(_layout, coords);
    auto v_out = _fn(To{}, v, static_cast<nint_t>(is)...);
    indexed_store_dispatch(
        To{}, _p, base, _layout.strides()[OutLayout::Ndim - 1], n, v_out);
  }

  TOut* _p;
  OutLayout _layout;
  TransformFn _fn;
};

} // namespace details

/**
 * @brief Standard input accessor for GEMM and fused operator kernels.
 *
 * `DataInput` hides the physical layout of an input tensor behind a uniform
 * vector-load interface. The implementation is selected from compile-time
 * stride metadata:
 *
 * - last dimension contiguous: load directly from source memory and apply the
 *   transform in `operator()`;
 * - second-last dimension contiguous: materialize a padded, last-contiguous
 *   aux buffer in the constructor, then serve `operator()` from aux;
 * - neither trailing dimension contiguous: indexed-load each logical row into
 *   aux in the constructor, then serve `operator()` from aux.
 *
 * The aux buffer, when required, covers the whole logical layout. Its last
 * dimension is padded to a vector lane count so callers can issue vector
 * loads without crossing into the next logical row. Inactive tail lanes are
 * returned as zero after the transform.
 *
 * Transforms are invoked as `fn(out_tag, in_vec, coords...)`, where
 * `coords...` is the full logical coordinate pack supplied by the caller.
 *
 * @warning The current transpose materialization uses a simple element loop
 * for correctness. It is the replaceable hook for future low-level block
 * transpose kernels; strided rows already use the vector indexed-load path.
 */
template <typename TOut, typename InTensor, typename TransformFn>
struct DataInput {
  using TIn = typename InTensor::ElementType;
  using InLayout = typename InTensor::Layout;
  using InputTensor = InTensor;
  using OutputElement = TOut;
  using Transform = std::remove_cvref_t<TransformFn>;
  using Kind = details::SelectAccessKind<InLayout>;
  using AccessKind = Kind;
  using Impl = details::DataInputImpl<Kind, TOut, InTensor, Transform>;

  VECOPS_INLINE static nint_t required_workspace(const InLayout& layout) {
    return Impl::required_workspace(layout);
  }

  VECOPS_INLINE DataInput(const TIn* p, const InLayout& layout, const Transform& fn, void* aux)
      : _impl(p, layout, fn, aux) {}

  template <vec::VectorTag To, typename N, typename... Is>
    requires is_any<vec::ElementOf<To>, TOut>
  VECOPS_INLINE vec::Vec<To> operator()(To t, N n, Is... is) const {
    return _impl(t, n, is...);
  }

  Impl _impl;
};

template <
    typename TOut,
    typename InTensor,
    typename TransformFn,
    typename Kind,
    typename AuxLayout = void>
struct DataInputView {
  using TIn = typename InTensor::ElementType;
  using InLayout = typename InTensor::Layout;
  using InputTensor = InTensor;
  using OutputElement = TOut;
  using Transform = std::remove_cvref_t<TransformFn>;
  using AccessKind = Kind;
  using KindType = Kind;
  using Impl = details::DataInputViewImpl<Kind, TOut, InTensor, Transform, AuxLayout>;

  template <typename... Args>
  VECOPS_INLINE explicit DataInputView(Args&&... args)
      : _impl(std::forward<Args>(args)...) {}

  template <vec::VectorTag To, typename N, typename... Is>
    requires is_any<vec::ElementOf<To>, TOut>
  VECOPS_INLINE vec::Vec<To> operator()(To t, N n, Is... is) const {
    return _impl(t, n, is...);
  }

  Impl _impl;
};

template <
    typename TOut,
    typename Arg1,
    typename Arg2 = void,
    typename Arg3 = void,
    typename Enable = void>
struct InputSpec;

/**
 * @brief Full input operand description: tensor view + transform.
 */
template <typename TOut, typename InTensor, typename TransformFn>
struct InputSpec<
    TOut,
    InTensor,
    TransformFn,
    void,
    std::enable_if_t<is_tensor<std::remove_cvref_t<InTensor>>>> {
  using InputTensor = std::remove_cvref_t<InTensor>;
  using TIn = typename InputTensor::ElementType;
  using InputLayout = typename InputTensor::Layout;
  using OutputElement = TOut;
  using Transform = std::remove_cvref_t<TransformFn>;
  using InputAccessor = DataInput<TOut, InputTensor, Transform>;

  static_assert(is_any<typename Transform::TIn, TIn>, "Input type of transform fn mismatch");
  static_assert(is_any<typename Transform::TOut, TOut>, "Output type of transform fn mismatch");

  VECOPS_INLINE static nint_t required_workspace(const InputLayout& layout) {
    return InputAccessor::required_workspace(layout);
  }

  VECOPS_INLINE InputSpec(const InputTensor& tensor, const Transform& fn)
      : _tensor(tensor), _fn(fn) {}

  template <typename T = Transform, std::enable_if_t<std::is_default_constructible_v<T>, bool> = true>
  VECOPS_INLINE explicit InputSpec(const InputTensor& tensor)
      : _tensor(tensor), _fn{} {}

  VECOPS_INLINE const InputTensor& tensor() const { return _tensor; }
  VECOPS_INLINE const Transform& transform() const { return _fn; }
  VECOPS_INLINE const InputLayout& input_layout() const { return _tensor.layout(); }

  VECOPS_INLINE nint_t required_workspace() const {
    return InputAccessor::required_workspace(_tensor.layout());
  }

  VECOPS_INLINE InputAccessor bind(void* aux) const {
    return InputAccessor(_tensor.data(), _tensor.layout(), _fn, aux);
  }

  VECOPS_INLINE InputAccessor bind(WorkspaceView& ws) const {
    const nint_t bytes = required_workspace();
    return bind(bytes == 0 ? nullptr : ws.allocate(bytes));
  }

  VECOPS_INLINE InputAccessor make_input(void* aux) const { return bind(aux); }

private:
  InputTensor _tensor;
  Transform _fn;
};

template <typename TOut, typename InTensor>
struct InputSpec<
    TOut,
    InTensor,
    void,
    void,
    std::enable_if_t<is_tensor<std::remove_cvref_t<InTensor>>>>
    : InputSpec<
          TOut,
          std::remove_cvref_t<InTensor>,
          IdentityVecTransform<TOut, typename std::remove_cvref_t<InTensor>::ElementType>> {
  using InputTensor = std::remove_cvref_t<InTensor>;
  using TIn = typename InputTensor::ElementType;
  using Base = InputSpec<TOut, InputTensor, IdentityVecTransform<TOut, TIn>>;
  using OutputElement = TOut;
  using Transform = typename Base::Transform;
  using Base::Base;
};

/**
 * @brief Legacy input layout description.
 *
 * This form keeps the original `InputSpec<TOut, TIn, Layout, Transform>` API:
 * it stores only layout + transform and binds the data pointer later with
 * `make_input()`. New user-facing code should prefer `input<TOut>(tensor)`.
 */
template <typename TOut, typename TIn, typename InLayout, typename TransformFn>
struct InputSpec<
    TOut,
    TIn,
    InLayout,
    TransformFn,
    std::enable_if_t<!is_tensor<std::remove_cvref_t<TIn>> &&
                     is_layout<std::remove_cvref_t<InLayout>>>> {
  using InputLayout = std::remove_cvref_t<InLayout>;
  using OutputElement = TOut;
  using Transform = std::conditional_t<
      std::is_void_v<TransformFn>,
      IdentityVecTransform<TOut, TIn>,
      std::remove_cvref_t<TransformFn>>;
  using InputTensor = Tensor<TIn, typename InputLayout::Shape, typename InputLayout::Strides>;
  using InputAccessor = DataInput<TOut, InputTensor, Transform>;

  static_assert(is_any<typename Transform::TIn, TIn>, "Input type of transform fn mismatch");
  static_assert(is_any<typename Transform::TOut, TOut>, "Output type of transform fn mismatch");

  VECOPS_INLINE static nint_t required_workspace(const InputLayout& layout) {
    return InputAccessor::required_workspace(layout);
  }

  VECOPS_INLINE InputSpec(const InputLayout& layout, const Transform& fn)
      : _in_layout(layout), _fn(fn) {}

  template <typename T = Transform, std::enable_if_t<std::is_default_constructible_v<T>, bool> = true>
  VECOPS_INLINE explicit InputSpec(const InputLayout& layout)
      : _in_layout(layout), _fn{} {}

  VECOPS_INLINE const InputLayout& input_layout() const { return _in_layout; }
  VECOPS_INLINE const Transform& transform() const { return _fn; }

  VECOPS_INLINE nint_t required_workspace() const {
    return InputAccessor::required_workspace(_in_layout);
  }

  VECOPS_INLINE InputAccessor make_input(const TIn* p, void* aux) const {
    return InputAccessor(p, _in_layout, _fn, aux);
  }

  VECOPS_INLINE InputAccessor make_input(const TIn* p, WorkspaceView& ws) const {
    const nint_t bytes = required_workspace();
    return make_input(p, bytes == 0 ? nullptr : ws.allocate(bytes));
  }

private:
  InputLayout _in_layout;
  Transform _fn;
};

/**
 * @brief Standard output accessor for GEMM and fused operator kernels.
 *
 * `DataOutput` mirrors `DataInput` for stores:
 *
 * - last dimension contiguous: transform and store directly in `operator()`;
 * - second-last dimension contiguous: transform into aux in `operator()`, then
 *   flush aux back to the destination layout in the destructor;
 * - neither trailing dimension contiguous: transform and indexed-store
 *   directly in `operator()`.
 *
 * The second-last-contiguous path uses the same padded aux layout as
 * `DataInput`. Only logical elements are flushed; padded tail cells remain
 * internal scratch space.
 *
 * Transforms are invoked as `fn(out_tag, in_vec, coords...)`, where
 * `coords...` is the full logical coordinate pack supplied by the caller.
 *
 * @warning Scatter with duplicate logical addresses has the same ordering
 * caveat as the underlying vector indexed-store primitive.
 */
template <typename TIn, typename OutTensor, typename TransformFn>
struct DataOutput {
  using TOut = typename OutTensor::ElementType;
  using OutLayout = typename OutTensor::Layout;
  using OutputTensor = OutTensor;
  using InputElement = TIn;
  using Transform = std::remove_cvref_t<TransformFn>;
  using Kind = details::SelectAccessKind<OutLayout>;
  using AccessKind = Kind;
  using Impl = details::DataOutputImpl<Kind, TIn, OutTensor, Transform>;

  VECOPS_INLINE static nint_t required_workspace(const OutLayout& layout) {
    return Impl::required_workspace(layout);
  }

  VECOPS_INLINE DataOutput(TOut* p, const OutLayout& layout, const Transform& fn, void* aux)
      : _impl(p, layout, fn, aux) {}

  template <vec::VectorTag Ti, typename N, typename... Is>
    requires is_any<vec::ElementOf<Ti>, TIn>
  VECOPS_INLINE void operator()(Ti t, vec::Vec<Ti> v, N n, Is... is) const {
    _impl(t, v, n, is...);
  }

  Impl _impl;
};

template <
    typename TIn,
    typename OutTensor,
    typename TransformFn,
    typename Kind,
    typename AuxLayout = void>
struct DataOutputView {
  using TOut = typename OutTensor::ElementType;
  using OutLayout = typename OutTensor::Layout;
  using OutputTensor = OutTensor;
  using InputElement = TIn;
  using Transform = std::remove_cvref_t<TransformFn>;
  using AccessKind = Kind;
  using KindType = Kind;
  using Impl = details::DataOutputViewImpl<Kind, TIn, OutTensor, Transform, AuxLayout>;

  template <typename... Args>
  VECOPS_INLINE explicit DataOutputView(Args&&... args)
      : _impl(std::forward<Args>(args)...) {}

  template <vec::VectorTag Ti, typename N, typename... Is>
    requires is_any<vec::ElementOf<Ti>, TIn>
  VECOPS_INLINE void operator()(Ti t, vec::Vec<Ti> v, N n, Is... is) const {
    _impl(t, v, n, is...);
  }

  Impl _impl;
};

template <
    typename TIn,
    typename Arg1,
    typename Arg2 = void,
    typename Arg3 = void,
    typename Enable = void>
struct OutputSpec;

/**
 * @brief Full output operand description: tensor view + transform.
 */
template <typename TIn, typename OutTensor, typename TransformFn>
struct OutputSpec<
    TIn,
    OutTensor,
    TransformFn,
    void,
    std::enable_if_t<is_tensor<std::remove_cvref_t<OutTensor>>>> {
  using OutputTensor = std::remove_cvref_t<OutTensor>;
  using TOut = typename OutputTensor::ElementType;
  using OutputLayout = typename OutputTensor::Layout;
  using InputElement = TIn;
  using Transform = std::remove_cvref_t<TransformFn>;
  using OutputAccessor = DataOutput<TIn, OutputTensor, Transform>;

  static_assert(is_any<typename Transform::TIn, TIn>, "Input type of transform fn mismatch");
  static_assert(is_any<typename Transform::TOut, TOut>, "Output type of transform fn mismatch");

  VECOPS_INLINE static nint_t required_workspace(const OutputLayout& layout) {
    return OutputAccessor::required_workspace(layout);
  }

  VECOPS_INLINE OutputSpec(const OutputTensor& tensor, const Transform& fn)
      : _tensor(tensor), _fn(fn) {}

  template <typename T = Transform, std::enable_if_t<std::is_default_constructible_v<T>, bool> = true>
  VECOPS_INLINE explicit OutputSpec(const OutputTensor& tensor)
      : _tensor(tensor), _fn{} {}

  VECOPS_INLINE const OutputTensor& tensor() const { return _tensor; }
  VECOPS_INLINE const Transform& transform() const { return _fn; }
  VECOPS_INLINE const OutputLayout& output_layout() const { return _tensor.layout(); }

  VECOPS_INLINE nint_t required_workspace() const {
    return OutputAccessor::required_workspace(_tensor.layout());
  }

  VECOPS_INLINE OutputAccessor bind(void* aux) const {
    return OutputAccessor(const_cast<TOut*>(_tensor.data()), _tensor.layout(), _fn, aux);
  }

  VECOPS_INLINE OutputAccessor bind(WorkspaceView& ws) const {
    const nint_t bytes = required_workspace();
    return bind(bytes == 0 ? nullptr : ws.allocate(bytes));
  }

  VECOPS_INLINE OutputAccessor make_output(void* aux) const { return bind(aux); }

private:
  OutputTensor _tensor;
  Transform _fn;
};

template <typename TIn, typename OutTensor>
struct OutputSpec<
    TIn,
    OutTensor,
    void,
    void,
    std::enable_if_t<is_tensor<std::remove_cvref_t<OutTensor>>>>
    : OutputSpec<
          TIn,
          std::remove_cvref_t<OutTensor>,
          IdentityVecTransform<typename std::remove_cvref_t<OutTensor>::ElementType, TIn>> {
  using OutputTensor = std::remove_cvref_t<OutTensor>;
  using TOut = typename OutputTensor::ElementType;
  using Base = OutputSpec<TIn, OutputTensor, IdentityVecTransform<TOut, TIn>>;
  using InputElement = TIn;
  using Transform = typename Base::Transform;
  using Base::Base;
};

/**
 * @brief Legacy output layout description.
 */
template <typename TIn, typename TOut, typename OutLayout, typename TransformFn>
struct OutputSpec<
    TIn,
    TOut,
    OutLayout,
    TransformFn,
    std::enable_if_t<!is_tensor<std::remove_cvref_t<TOut>> &&
                     is_layout<std::remove_cvref_t<OutLayout>>>> {
  using OutputLayout = std::remove_cvref_t<OutLayout>;
  using InputElement = TIn;
  using Transform = std::conditional_t<
      std::is_void_v<TransformFn>,
      IdentityVecTransform<TOut, TIn>,
      std::remove_cvref_t<TransformFn>>;
  using OutputTensor = Tensor<TOut, typename OutputLayout::Shape, typename OutputLayout::Strides>;
  using OutputAccessor = DataOutput<TIn, OutputTensor, Transform>;

  static_assert(is_any<typename Transform::TIn, TIn>, "Input type of transform fn mismatch");
  static_assert(is_any<typename Transform::TOut, TOut>, "Output type of transform fn mismatch");

  VECOPS_INLINE static nint_t required_workspace(const OutputLayout& layout) {
    return OutputAccessor::required_workspace(layout);
  }

  VECOPS_INLINE OutputSpec(const OutputLayout& layout, const Transform& fn)
      : _out_layout(layout), _fn(fn) {}

  template <typename T = Transform, std::enable_if_t<std::is_default_constructible_v<T>, bool> = true>
  VECOPS_INLINE explicit OutputSpec(const OutputLayout& layout)
      : _out_layout(layout), _fn{} {}

  VECOPS_INLINE const OutputLayout& output_layout() const { return _out_layout; }
  VECOPS_INLINE const Transform& transform() const { return _fn; }

  VECOPS_INLINE nint_t required_workspace() const {
    return OutputAccessor::required_workspace(_out_layout);
  }

  VECOPS_INLINE OutputAccessor make_output(TOut* p, void* aux) const {
    return OutputAccessor(p, _out_layout, _fn, aux);
  }

  VECOPS_INLINE OutputAccessor make_output(TOut* p, WorkspaceView& ws) const {
    const nint_t bytes = required_workspace();
    return make_output(p, bytes == 0 ? nullptr : ws.allocate(bytes));
  }

private:
  OutputLayout _out_layout;
  Transform _fn;
};

template <typename TOut, typename Tensor>
VECOPS_INLINE auto input(Tensor&& tensor) {
  using TensorT = std::remove_cvref_t<Tensor>;
  return InputSpec<TOut, TensorT>(std::forward<Tensor>(tensor));
}

template <typename TOut, typename Tensor, typename TransformFn>
VECOPS_INLINE auto input(Tensor&& tensor, TransformFn&& fn) {
  using TensorT = std::remove_cvref_t<Tensor>;
  using TIn = typename TensorT::ElementType;
  auto adapted = adapt_vec_transform<TOut, TIn>(std::forward<TransformFn>(fn));
  using Transform = decltype(adapted);
  return InputSpec<TOut, TensorT, Transform>(std::forward<Tensor>(tensor), adapted);
}

template <typename TOut, typename Tensor>
VECOPS_INLINE auto in(Tensor&& tensor) {
  return input<TOut>(std::forward<Tensor>(tensor));
}

template <typename TOut, typename Tensor, typename TransformFn>
VECOPS_INLINE auto in(Tensor&& tensor, TransformFn&& fn) {
  return input<TOut>(std::forward<Tensor>(tensor), std::forward<TransformFn>(fn));
}

template <typename TIn, typename Tensor>
VECOPS_INLINE auto output(Tensor&& tensor) {
  using TensorT = std::remove_cvref_t<Tensor>;
  return OutputSpec<TIn, TensorT>(std::forward<Tensor>(tensor));
}

template <typename TIn, typename Tensor, typename TransformFn>
VECOPS_INLINE auto output(Tensor&& tensor, TransformFn&& fn) {
  using TensorT = std::remove_cvref_t<Tensor>;
  using TOut = typename TensorT::ElementType;
  auto adapted = adapt_vec_transform<TOut, TIn>(std::forward<TransformFn>(fn));
  using Transform = decltype(adapted);
  return OutputSpec<TIn, TensorT, Transform>(std::forward<Tensor>(tensor), adapted);
}

template <typename TIn, typename Tensor>
VECOPS_INLINE auto out(Tensor&& tensor) {
  return output<TIn>(std::forward<Tensor>(tensor));
}

template <typename TIn, typename Tensor, typename TransformFn>
VECOPS_INLINE auto out(Tensor&& tensor, TransformFn&& fn) {
  return output<TIn>(std::forward<Tensor>(tensor), std::forward<TransformFn>(fn));
}

template <typename... Specs>
VECOPS_INLINE nint_t required_workspace(const Specs&... specs) {
  nint_t total = 0;
  ((total = details::workspace_round_up(total, vec::DEFAULT_ALIGNMENT) +
            specs.required_workspace()), ...);
  return details::workspace_round_up(total, vec::DEFAULT_ALIGNMENT);
}

namespace details {

template <int ActualDim, typename Transform>
VECOPS_INLINE auto prefixed_transform(const Transform& fn, nint_t index) {
  return prepend_transform_prefix(fn, std::tuple<nint_t>{index});
}

template <int ActualDim, typename TOut, typename InTensor, typename Transform>
VECOPS_INLINE auto slice_data_input_last(
    const DataInput<TOut, InTensor, Transform>& input,
    nint_t index) {
  const auto& impl = input._impl;
  auto tensor = make_tensor(impl._p, impl._layout);
  auto sliced = hop::details::slice_at_actual_dim<ActualDim>(tensor, index);
  auto fn = prefixed_transform<ActualDim>(impl._fn, index);
  using SlicedTensor = std::remove_cvref_t<decltype(sliced)>;
  using PrefixedTransform = decltype(fn);
  return DataInputView<TOut, SlicedTensor, PrefixedTransform, AccessKindLastContiguous>(
      sliced.data(),
      sliced.layout(),
      fn);
}

template <int ActualDim, typename Accessor>
VECOPS_INLINE auto slice_data_input_view_last(const Accessor& input, nint_t index) {
  const auto& impl = input._impl;
  auto tensor = make_tensor(impl._p, impl._layout);
  auto sliced = hop::details::slice_at_actual_dim<ActualDim>(tensor, index);
  auto fn = prefixed_transform<ActualDim>(impl._fn, index);
  using SlicedTensor = std::remove_cvref_t<decltype(sliced)>;
  using PrefixedTransform = decltype(fn);
  return DataInputView<
      typename Accessor::OutputElement,
      SlicedTensor,
      PrefixedTransform,
      AccessKindLastContiguous>(
      sliced.data(),
      sliced.layout(),
      fn);
}

template <int ActualDim, typename Accessor>
VECOPS_INLINE auto slice_data_input_aux(const Accessor& input, nint_t index) {
  const auto& impl = input._impl;
  auto src_tensor = make_tensor(impl._p, impl._layout);
  auto aux_tensor = make_tensor(impl._aux, impl._aux_layout);
  auto sliced_src = hop::details::slice_at_actual_dim<ActualDim>(src_tensor, index);
  auto sliced_aux = hop::details::slice_at_actual_dim<ActualDim>(aux_tensor, index);
  using SlicedTensor = std::remove_cvref_t<decltype(sliced_src)>;
  using SlicedAuxLayout = std::remove_cvref_t<decltype(sliced_aux.layout())>;
  return DataInputView<
      typename Accessor::OutputElement,
      SlicedTensor,
      typename Accessor::Transform,
      typename Accessor::AccessKind,
      SlicedAuxLayout>(
      sliced_src.data(),
      sliced_src.layout(),
      sliced_aux.layout(),
      const_cast<typename Accessor::OutputElement*>(sliced_aux.data()));
}

template <int ActualDim, typename TIn, typename OutTensor, typename Transform>
VECOPS_INLINE auto slice_data_output_last(
    const DataOutput<TIn, OutTensor, Transform>& output,
    nint_t index) {
  const auto& impl = output._impl;
  auto tensor = make_tensor(impl._p, impl._layout);
  auto sliced = hop::details::slice_at_actual_dim<ActualDim>(tensor, index);
  auto fn = prefixed_transform<ActualDim>(impl._fn, index);
  using SlicedTensor = std::remove_cvref_t<decltype(sliced)>;
  using PrefixedTransform = decltype(fn);
  return DataOutputView<TIn, SlicedTensor, PrefixedTransform, AccessKindLastContiguous>(
      const_cast<typename SlicedTensor::ElementType*>(sliced.data()),
      sliced.layout(),
      fn);
}

template <int ActualDim, typename Accessor>
VECOPS_INLINE auto slice_data_output_view_last(const Accessor& output, nint_t index) {
  const auto& impl = output._impl;
  auto tensor = make_tensor(impl._p, impl._layout);
  auto sliced = hop::details::slice_at_actual_dim<ActualDim>(tensor, index);
  auto fn = prefixed_transform<ActualDim>(impl._fn, index);
  using SlicedTensor = std::remove_cvref_t<decltype(sliced)>;
  using PrefixedTransform = decltype(fn);
  return DataOutputView<
      typename Accessor::InputElement,
      SlicedTensor,
      PrefixedTransform,
      AccessKindLastContiguous>(
      const_cast<typename SlicedTensor::ElementType*>(sliced.data()),
      sliced.layout(),
      fn);
}

template <int ActualDim, typename Accessor>
VECOPS_INLINE auto slice_data_output_strided(const Accessor& output, nint_t index) {
  const auto& impl = output._impl;
  auto tensor = make_tensor(impl._p, impl._layout);
  auto sliced = hop::details::slice_at_actual_dim<ActualDim>(tensor, index);
  auto fn = prefixed_transform<ActualDim>(impl._fn, index);
  using SlicedTensor = std::remove_cvref_t<decltype(sliced)>;
  using PrefixedTransform = decltype(fn);
  return DataOutputView<
      typename Accessor::InputElement,
      SlicedTensor,
      PrefixedTransform,
      AccessKindStrided>(
      const_cast<typename SlicedTensor::ElementType*>(sliced.data()),
      sliced.layout(),
      fn);
}

template <int ActualDim, typename Accessor>
VECOPS_INLINE auto slice_data_output_second_last(const Accessor& output, nint_t index) {
  const auto& impl = output._impl;
  auto out_tensor = make_tensor(impl._p, impl._layout);
  auto aux_tensor = make_tensor(impl._aux, impl._aux_layout);
  auto sliced_out = hop::details::slice_at_actual_dim<ActualDim>(out_tensor, index);
  auto sliced_aux = hop::details::slice_at_actual_dim<ActualDim>(aux_tensor, index);
  auto fn = prefixed_transform<ActualDim>(impl._fn, index);
  using SlicedTensor = std::remove_cvref_t<decltype(sliced_out)>;
  using SlicedAuxLayout = std::remove_cvref_t<decltype(sliced_aux.layout())>;
  using PrefixedTransform = decltype(fn);
  return DataOutputView<
      typename Accessor::InputElement,
      SlicedTensor,
      PrefixedTransform,
      AccessKindSecondLastContiguous,
      SlicedAuxLayout>(
      const_cast<typename SlicedTensor::ElementType*>(sliced_out.data()),
      sliced_out.layout(),
      sliced_aux.layout(),
      fn,
      const_cast<typename SlicedTensor::ElementType*>(sliced_aux.data()));
}

template <int ActualDim, typename Spec>
VECOPS_INLINE auto slice_input_spec(const Spec& spec, nint_t index) {
  auto sliced = hop::details::slice_at_actual_dim<ActualDim>(spec.tensor(), index);
  auto fn = prefixed_transform<ActualDim>(spec.transform(), index);
  using SlicedTensor = std::remove_cvref_t<decltype(sliced)>;
  using PrefixedTransform = decltype(fn);
  return InputSpec<typename Spec::OutputElement, SlicedTensor, PrefixedTransform>(
      sliced,
      fn);
}

template <int ActualDim, typename Spec>
VECOPS_INLINE auto slice_output_spec(const Spec& spec, nint_t index) {
  auto sliced = hop::details::slice_at_actual_dim<ActualDim>(spec.tensor(), index);
  auto fn = prefixed_transform<ActualDim>(spec.transform(), index);
  using SlicedTensor = std::remove_cvref_t<decltype(sliced)>;
  using PrefixedTransform = decltype(fn);
  return OutputSpec<typename Spec::InputElement, SlicedTensor, PrefixedTransform>(
      sliced,
      fn);
}

} // namespace details

namespace hop::details {

template <typename TOut, typename InTensor, typename Transform>
struct SliceTraits<DataInput<TOut, InTensor, Transform>> {
  using Accessor = DataInput<TOut, InTensor, Transform>;
  static constexpr int rank = InTensor::Ndim;
  static constexpr bool is_sliceable = true;

  template <int I>
  using shape_dim = typename TensorShapeDim<I, InTensor>::type;

  VECOPS_INLINE static nint_t size(const Accessor& input, int dim) {
    return input._impl._layout.shape()[dim];
  }

  template <int ActualDim, typename U>
  VECOPS_INLINE static auto slice(U&& input, nint_t index) {
    using Kind = typename Accessor::AccessKind;
    if constexpr (std::is_same_v<Kind, vecops::gemm::details::AccessKindLastContiguous>) {
      return vecops::gemm::details::slice_data_input_last<ActualDim>(input, index);
    } else {
      return vecops::gemm::details::slice_data_input_aux<ActualDim>(input, index);
    }
  }
};

template <typename TOut, typename InTensor, typename Transform, typename Kind, typename AuxLayout>
struct SliceTraits<DataInputView<TOut, InTensor, Transform, Kind, AuxLayout>> {
  using Accessor = DataInputView<TOut, InTensor, Transform, Kind, AuxLayout>;
  static constexpr int rank = InTensor::Ndim;
  static constexpr bool is_sliceable = true;

  template <int I>
  using shape_dim = typename TensorShapeDim<I, InTensor>::type;

  VECOPS_INLINE static nint_t size(const Accessor& input, int dim) {
    return input._impl._layout.shape()[dim];
  }

  template <int ActualDim, typename U>
  VECOPS_INLINE static auto slice(U&& input, nint_t index) {
    if constexpr (std::is_same_v<Kind, vecops::gemm::details::AccessKindLastContiguous>) {
      return vecops::gemm::details::slice_data_input_view_last<ActualDim>(input, index);
    } else {
      return vecops::gemm::details::slice_data_input_aux<ActualDim>(input, index);
    }
  }
};

template <typename TIn, typename OutTensor, typename Transform>
struct SliceTraits<DataOutput<TIn, OutTensor, Transform>> {
  using Accessor = DataOutput<TIn, OutTensor, Transform>;
  static constexpr int rank = OutTensor::Ndim;
  static constexpr bool is_sliceable = true;

  template <int I>
  using shape_dim = typename TensorShapeDim<I, OutTensor>::type;

  VECOPS_INLINE static nint_t size(const Accessor& output, int dim) {
    return output._impl._layout.shape()[dim];
  }

  template <int ActualDim, typename U>
  VECOPS_INLINE static auto slice(U&& output, nint_t index) {
    using Kind = typename Accessor::AccessKind;
    if constexpr (std::is_same_v<Kind, vecops::gemm::details::AccessKindLastContiguous>) {
      return vecops::gemm::details::slice_data_output_last<ActualDim>(output, index);
    } else if constexpr (std::is_same_v<Kind, vecops::gemm::details::AccessKindSecondLastContiguous>) {
      return vecops::gemm::details::slice_data_output_second_last<ActualDim>(output, index);
    } else {
      return vecops::gemm::details::slice_data_output_strided<ActualDim>(output, index);
    }
  }
};

template <typename TIn, typename OutTensor, typename Transform, typename Kind, typename AuxLayout>
struct SliceTraits<DataOutputView<TIn, OutTensor, Transform, Kind, AuxLayout>> {
  using Accessor = DataOutputView<TIn, OutTensor, Transform, Kind, AuxLayout>;
  static constexpr int rank = OutTensor::Ndim;
  static constexpr bool is_sliceable = true;

  template <int I>
  using shape_dim = typename TensorShapeDim<I, OutTensor>::type;

  VECOPS_INLINE static nint_t size(const Accessor& output, int dim) {
    return output._impl._layout.shape()[dim];
  }

  template <int ActualDim, typename U>
  VECOPS_INLINE static auto slice(U&& output, nint_t index) {
    if constexpr (std::is_same_v<Kind, vecops::gemm::details::AccessKindLastContiguous>) {
      return vecops::gemm::details::slice_data_output_view_last<ActualDim>(output, index);
    } else if constexpr (std::is_same_v<Kind, vecops::gemm::details::AccessKindSecondLastContiguous>) {
      return vecops::gemm::details::slice_data_output_second_last<ActualDim>(output, index);
    } else {
      return vecops::gemm::details::slice_data_output_strided<ActualDim>(output, index);
    }
  }
};

template <typename TOut, typename InTensor, typename Transform>
struct SliceTraits<
    InputSpec<
        TOut,
        InTensor,
        Transform,
        void,
        std::enable_if_t<is_tensor<std::remove_cvref_t<InTensor>>>>> {
  using Spec = InputSpec<
      TOut,
      InTensor,
      Transform,
      void,
      std::enable_if_t<is_tensor<std::remove_cvref_t<InTensor>>>>;
  static constexpr int rank = Spec::InputTensor::Ndim;
  static constexpr bool is_sliceable = true;

  template <int I>
  using shape_dim = typename TensorShapeDim<I, typename Spec::InputTensor>::type;

  VECOPS_INLINE static nint_t size(const Spec& spec, int dim) {
    return spec.tensor().size(dim);
  }

  template <int ActualDim, typename U>
  VECOPS_INLINE static auto slice(U&& spec, nint_t index) {
    return vecops::gemm::details::slice_input_spec<ActualDim>(spec, index);
  }
};

template <typename TIn, typename OutTensor, typename Transform>
struct SliceTraits<
    OutputSpec<
        TIn,
        OutTensor,
        Transform,
        void,
        std::enable_if_t<is_tensor<std::remove_cvref_t<OutTensor>>>>> {
  using Spec = OutputSpec<
      TIn,
      OutTensor,
      Transform,
      void,
      std::enable_if_t<is_tensor<std::remove_cvref_t<OutTensor>>>>;
  static constexpr int rank = Spec::OutputTensor::Ndim;
  static constexpr bool is_sliceable = true;

  template <int I>
  using shape_dim = typename TensorShapeDim<I, typename Spec::OutputTensor>::type;

  VECOPS_INLINE static nint_t size(const Spec& spec, int dim) {
    return spec.tensor().size(dim);
  }

  template <int ActualDim, typename U>
  VECOPS_INLINE static auto slice(U&& spec, nint_t index) {
    return vecops::gemm::details::slice_output_spec<ActualDim>(spec, index);
  }
};

} // namespace hop::details

namespace details {

template <typename T>
struct IsInputSpec : std::false_type {};
template <typename... Args>
struct IsInputSpec<InputSpec<Args...>> : std::true_type {};

template <typename T>
struct IsOutputSpec : std::false_type {};
template <typename... Args>
struct IsOutputSpec<OutputSpec<Args...>> : std::true_type {};

} // namespace details

/**
 * @brief True when `Spec` is a `gemm::InputSpec` instantiation.
 *
 * This trait ignores cv/ref qualifiers, so `InputSpec<...>`,
 * `const InputSpec<...>&`, and `InputSpec<...>&&` are all recognized.
 * It is intended for operator front-ends that accept the user-facing
 * DataAccess spec API and want a clear compile-time contract.
 */
template <typename Spec>
static constexpr bool is_input_spec_v =
    details::IsInputSpec<std::remove_cvref_t<Spec>>::value;

/**
 * @brief True when `Spec` is a `gemm::OutputSpec` instantiation.
 *
 * This trait ignores cv/ref qualifiers, so `OutputSpec<...>`,
 * `const OutputSpec<...>&`, and `OutputSpec<...>&&` are all recognized.
 * It is intended for operator front-ends that accept the user-facing
 * DataAccess spec API and want a clear compile-time contract.
 */
template <typename Spec>
static constexpr bool is_output_spec_v =
    details::IsOutputSpec<std::remove_cvref_t<Spec>>::value;

/**
 * @brief True when an InputSpec transform is the zero-vector function.
 */
template <typename Spec>
static constexpr bool is_zero_input_spec_v =
    is_zero_vec_transform_v<typename std::remove_cvref_t<Spec>::Transform>;

/**
 * @brief True when an OutputSpec transform is the zero-vector function.
 */
template <typename Spec>
static constexpr bool is_zero_output_spec_v =
    is_zero_vec_transform_v<typename std::remove_cvref_t<Spec>::Transform>;

/**
 * @brief True when an InputSpec is identity and can read last-dim contiguous memory.
 */
template <typename Spec>
static constexpr bool is_identity_last_contiguous_input_spec_v =
    is_identity_vec_transform_v<typename std::remove_cvref_t<Spec>::Transform> &&
    details::IsLastDimContiguous<typename std::remove_cvref_t<Spec>::InputLayout>::value;

/**
 * @brief True when an OutputSpec is identity and can write last-dim contiguous memory.
 */
template <typename Spec>
static constexpr bool is_identity_last_contiguous_output_spec_v =
    is_identity_vec_transform_v<typename std::remove_cvref_t<Spec>::Transform> &&
    details::IsLastDimContiguous<typename std::remove_cvref_t<Spec>::OutputLayout>::value;

/**
 * @brief True when an InputSpec is identity and matches the second-last contiguous path.
 */
template <typename Spec>
static constexpr bool is_identity_second_last_contiguous_input_spec_v =
    is_identity_vec_transform_v<typename std::remove_cvref_t<Spec>::Transform> &&
    !details::IsLastDimContiguous<typename std::remove_cvref_t<Spec>::InputLayout>::value &&
    details::IsSecondLastDimContiguous<typename std::remove_cvref_t<Spec>::InputLayout>::value;

/**
 * @brief True when an OutputSpec is identity and matches the second-last contiguous path.
 */
template <typename Spec>
static constexpr bool is_identity_second_last_contiguous_output_spec_v =
    is_identity_vec_transform_v<typename std::remove_cvref_t<Spec>::Transform> &&
    !details::IsLastDimContiguous<typename std::remove_cvref_t<Spec>::OutputLayout>::value &&
    details::IsSecondLastDimContiguous<typename std::remove_cvref_t<Spec>::OutputLayout>::value;

} // namespace vecops::gemm

#endif // VECOPS_DATAACCESS_H
