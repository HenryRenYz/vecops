//
// Created by renyz on 2026/7/8.
//

#ifndef VECOPS_DATAACCESS_H
#define VECOPS_DATAACCESS_H

#include "vecops/util/TypeTraits.h"
#include "vecops/vec/Vec.h"
#include "./Attachment.h"
#include "./Tensor.h"

#include <algorithm>
#include <array>
#include <cstddef>
#include <type_traits>

namespace vecops::gemm {
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

constexpr nint_t round_up(nint_t value, nint_t alignment) {
  return (value + alignment - 1) & ~(alignment - 1);
}

template <typename T>
constexpr nint_t aux_vector_lanes() {
  return vec::size(vec::ScalableTag<T, VEC_MAX_POW>{});
}

template <typename TLayout, typename TAux>
nint_t padded_last_size(const TLayout& layout) {
  return round_up(layout.shape()[TLayout::Ndim - 1], aux_vector_lanes<TAux>());
}

template <typename TLayout, typename TAux>
nint_t aux_numel(const TLayout& layout) {
  nint_t n = padded_last_size<TLayout, TAux>(layout);
  VECOPS_UNROLL
  for (int d = 0; d < TLayout::Ndim - 1; ++d) n *= layout.shape()[d];
  return n;
}

template <typename TLayout, typename TAux>
nint_t aux_required_bytes(const TLayout& layout) {
  return round_up(aux_numel<TLayout, TAux>(layout) * static_cast<nint_t>(sizeof(TAux)),
                  vec::DEFAULT_ALIGNMENT);
}

template <typename TLayout>
nint_t offset_from_coords(const TLayout& layout, const std::array<nint_t, TLayout::Ndim>& coords) {
  nint_t offset = 0;
  VECOPS_UNROLL for (int d = 0; d < TLayout::Ndim; ++d)
    offset += coords[d] * layout.strides()[d];
  return offset;
}

template <typename TLayout>
nint_t offset_from_coords_and_last(
    const TLayout& layout,
    const std::array<nint_t, TLayout::Ndim>& coords,
    nint_t last) {
  nint_t offset = 0;
  VECOPS_UNROLL for (int d = 0; d < TLayout::Ndim - 1; ++d)
    offset += coords[d] * layout.strides()[d];
  offset += last * layout.strides()[TLayout::Ndim - 1];
  return offset;
}

template <typename TLayout, typename TAux>
nint_t aux_offset_from_coords(
    const TLayout& layout,
    const std::array<nint_t, TLayout::Ndim>& coords) {
  nint_t offset = 0;
  nint_t stride = 1;
  const nint_t padded_last = padded_last_size<TLayout, TAux>(layout);
  VECOPS_UNROLL for (int d = TLayout::Ndim - 1; d >= 0; --d) {
    offset += coords[d] * stride;
    stride *= (d == TLayout::Ndim - 1) ? padded_last : layout.shape()[d];
  }
  return offset;
}

template <typename TLayout, typename F>
void for_each_prefix_row(const TLayout& layout, F&& f) {
  constexpr int ndim = TLayout::Ndim;
  nint_t rows = 1;
  VECOPS_UNROLL for (int d = 0; d < ndim - 1; ++d)
    rows *= layout.shape()[d];

  for (nint_t linear = 0; linear < rows; ++linear) {
    std::array<nint_t, ndim> coords{};
    nint_t rem = linear;
    VECOPS_UNROLL
    for (int d = ndim - 2; d >= 0; --d) {
      const nint_t extent = layout.shape()[d];
      coords[d] = rem % extent;
      rem /= extent;
    }
    f(coords);
  }
}

template <typename TLayout>
nint_t transform_x(const std::array<nint_t, TLayout::Ndim>& coords) {
  if constexpr (TLayout::Ndim >= 2) {
    return coords[TLayout::Ndim - 2];
  } else {
    return 0;
  }
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

template <nint_t N, TLV_DECL_TAG(T)>
struct UseUnmaskedPath<Const<N>, T>
    : std::bool_constant<(N > vec::max_word_size(T{}) * vec::num_words(T{}))> {};

template <typename N>
constexpr nint_t count_value(N n) {
  if constexpr (is_static_count_v<N>) {
    return StaticCount<std::remove_cvref_t<N>>::value;
  } else {
    return static_cast<nint_t>(n);
  }
}

template <typename N, TLV_DECL_TAG(T)>
static constexpr bool use_unmasked_path_v = UseUnmaskedPath<std::remove_cvref_t<N>, T>::value;

template <typename N, TLV_DECL_TAG(T)>
vec::Vec<T> loadu_dispatch(T t, const vec::TypeOf<T>* p, N n) {
  if constexpr (use_unmasked_path_v<N, T>) {
    return vec::loadu(t, p);
  } else {
    return vec::loadu(t, p, vec::mwhilelt(t, 0, count_value(n)), vec::zeros(t));
  }
}

template <typename N, TLV_DECL_TAG(T)>
void storeu_dispatch(T t, vec::TypeOf<T>* p, N n, vec::Vec<T> v) {
  if constexpr (use_unmasked_path_v<N, T>) {
    vec::storeu(t, p, v);
  } else {
    vec::storeu(t, p, vec::mwhilelt(t, 0, count_value(n)), v);
  }
}

template <typename T, TLV_DECL_TAG(Ti)>
vec::Vec<vec::Rebind<vec::GatherScatterIndex<T>, Ti>>
make_index_vector(
    Ti ti,
    nint_t base_offset,
    nint_t stride) {
  using IndexTag = vec::Rebind<vec::GatherScatterIndex<T>, Ti>;
  using Index = vec::TypeOf<IndexTag>;
  IndexTag it;
  auto idx = vec::fill(it, static_cast<Index>(base_offset));
  for (nint_t lane = 0; lane < vec::size(it); ++lane) {
    idx = vec::set(it, idx, lane, static_cast<Index>(base_offset + lane * stride));
  }
  return idx;
}

template <typename N, typename T, TLV_DECL_TAG(Ti)>
vec::Vec<Ti> gather_dispatch(
    Ti ti,
    const T* p,
    nint_t base_offset,
    nint_t stride,
    N n) {
  auto idx = make_index_vector<T>(ti, base_offset, stride);
  if constexpr (use_unmasked_path_v<N, Ti>) {
    return vec::gather(ti, p, idx);
  } else {
    return vec::gather(ti, p, idx, vec::mwhilelt(ti, 0, count_value(n)), vec::zeros(ti));
  }
}

template <typename N, typename T, TLV_DECL_TAG(Ti)>
void scatter_dispatch(
    Ti ti,
    T* p,
    nint_t base_offset,
    nint_t stride,
    N n,
    vec::Vec<Ti> v) {
  auto idx = make_index_vector<T>(ti, base_offset, stride);
  if constexpr (use_unmasked_path_v<N, Ti>) {
    vec::scatter(ti, p, idx, v);
  } else {
    vec::scatter(ti, p, idx, vec::mwhilelt(ti, 0, count_value(n)), v);
  }
}

template <typename TOut, typename TIn, typename InLayout, typename TransformFn>
void precompute_input_aux(
    const TIn* p,
    const InLayout& layout,
    const TransformFn& fn,
  TOut* aux) {
  std::fill(aux, aux + aux_numel<InLayout, TOut>(layout), TOut{});
  using To = vec::ScalableTag<TOut, 0>;
  using Ti = vec::Rebind<TIn, To>;
  To to;
  Ti ti;
  const nint_t last = layout.shape()[InLayout::Ndim - 1];

  for_each_prefix_row(layout, [&](auto coords) {
    for (nint_t y = 0; y < last; y += vec::size(to)) {
      const nint_t n = std::min(vec::size(to), last - y);
      coords[InLayout::Ndim - 1] = y;
      const nint_t src_offset = offset_from_coords_and_last(layout, coords, y);
      auto v_in = gather_dispatch(ti, p, src_offset, layout.strides()[InLayout::Ndim - 1], Any{n});
      auto v_out = fn.call(to, v_in, transform_x<InLayout>(coords), y);
      storeu_dispatch(to, aux + aux_offset_from_coords<InLayout, TOut>(layout, coords), Any{n}, v_out);
    }
  });
}

template <typename Kind, typename TOut, typename InTensor, typename TransformFn>
struct DataInputImpl;

template <typename TOut, typename InTensor, typename TransformFn>
struct DataInputImpl<AccessKindLastContiguous, TOut, InTensor, TransformFn> {
  using TIn = typename InTensor::ElementType;
  using InLayout = typename InTensor::Layout;

  static constexpr nint_t required_aux_size(const InLayout&) { return 0; }

  DataInputImpl(const TIn* p, const InLayout& layout, const TransformFn& fn, void*)
      : _p(p), _layout(layout), _fn(fn) {}

  template <TLV_DECL_TAG(To), typename N, typename... Is>
  vec::Vec<To> operator()(To t, N n, Is... is) const {
    static_assert(sizeof...(Is) == InLayout::Ndim, "coordinate count must match input rank");
    using Ti = vec::Rebind<TIn, To>;
    std::array<nint_t, InLayout::Ndim> coords{static_cast<nint_t>(is)...};
    const nint_t offset = offset_from_coords(_layout, coords);
    auto v_in = loadu_dispatch(Ti{}, _p + offset, n);
    auto v_out = _fn.call(t, v_in, transform_x<InLayout>(coords), coords[InLayout::Ndim - 1]);
    return v_out;
  }

  const TIn* _p;
  InLayout _layout;
  TransformFn _fn;
};

template <typename Kind, typename TOut, typename InTensor, typename TransformFn>
struct DataInputAuxImpl {
  using TIn = typename InTensor::ElementType;
  using InLayout = typename InTensor::Layout;

  static nint_t required_aux_size(const InLayout& layout) {
    return aux_required_bytes<InLayout, TOut>(layout);
  }

  DataInputAuxImpl(const TIn* p, const InLayout& layout, const TransformFn& fn, void* aux)
      : _layout(layout), _aux(static_cast<TOut*>(aux)) {
    VECOPS_ASSERT(aux != nullptr, "DataInput aux buffer is required for non-last-contiguous layouts");
    precompute_input_aux<TOut>(p, _layout, fn, _aux);
  }

  template <TLV_DECL_TAG(To), typename N, typename... Is>
  vec::Vec<To> operator()(To t, N n, Is... is) const {
    static_assert(sizeof...(Is) == InLayout::Ndim, "coordinate count must match input rank");
    std::array<nint_t, InLayout::Ndim> coords{static_cast<nint_t>(is)...};
    const nint_t offset = aux_offset_from_coords<InLayout, TOut>(_layout, coords);
    auto v = loadu_dispatch(t, _aux + offset, n);
    return v;
  }

  InLayout _layout;
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

  static constexpr nint_t required_aux_size(const OutLayout&) { return 0; }

  DataOutputImpl(TOut* p, const OutLayout& layout, const TransformFn& fn, void*)
      : _p(p), _layout(layout), _fn(fn) {}

  template <TLV_DECL_TAG(Ti), typename N, typename... Is>
  void operator()(Ti ti, vec::Vec<Ti> v, N n, Is... is) const {
    static_assert(sizeof...(Is) == OutLayout::Ndim, "coordinate count must match output rank");
    using To = vec::Rebind<TOut, Ti>;
    std::array<nint_t, OutLayout::Ndim> coords{static_cast<nint_t>(is)...};
    const nint_t offset = offset_from_coords(_layout, coords);
    auto v_out = _fn.call(To{}, v, transform_x<OutLayout>(coords), coords[OutLayout::Ndim - 1]);
    storeu_dispatch(To{}, _p + offset, n, v_out);
  }

  TOut* _p;
  OutLayout _layout;
  TransformFn _fn;
};

template <typename TIn, typename OutTensor, typename TransformFn>
struct DataOutputImpl<AccessKindSecondLastContiguous, TIn, OutTensor, TransformFn> {
  using TOut = typename OutTensor::ElementType;
  using OutLayout = typename OutTensor::Layout;

  static nint_t required_aux_size(const OutLayout& layout) {
    return aux_required_bytes<OutLayout, TOut>(layout);
  }

  DataOutputImpl(TOut* p, const OutLayout& layout, const TransformFn& fn, void* aux)
      : _p(p), _layout(layout), _fn(fn), _aux(static_cast<TOut*>(aux)) {
    VECOPS_ASSERT(aux != nullptr, "DataOutput aux buffer is required for second-last-contiguous layouts");
    std::fill(_aux, _aux + aux_numel<OutLayout, TOut>(_layout), TOut{});
  }

  ~DataOutputImpl() {
    for_each_prefix_row(_layout, [&](auto coords) {
      const nint_t last = _layout.shape()[OutLayout::Ndim - 1];
      for (nint_t y = 0; y < last; ++y) {
        coords[OutLayout::Ndim - 1] = y;
        _p[offset_from_coords(_layout, coords)] =
            _aux[aux_offset_from_coords<OutLayout, TOut>(_layout, coords)];
      }
    });
  }

  template <TLV_DECL_TAG(Ti), typename N, typename... Is>
  void operator()(Ti ti, vec::Vec<Ti> v, N n, Is... is) const {
    static_assert(sizeof...(Is) == OutLayout::Ndim, "coordinate count must match output rank");
    using To = vec::Rebind<TOut, Ti>;
    std::array<nint_t, OutLayout::Ndim> coords{static_cast<nint_t>(is)...};
    auto v_out = _fn.call(To{}, v, transform_x<OutLayout>(coords), coords[OutLayout::Ndim - 1]);
    const nint_t offset = aux_offset_from_coords<OutLayout, TOut>(_layout, coords);
    storeu_dispatch(To{}, _aux + offset, n, v_out);
  }

  TOut* _p;
  OutLayout _layout;
  TransformFn _fn;
  TOut* _aux;
};

template <typename TIn, typename OutTensor, typename TransformFn>
struct DataOutputImpl<AccessKindStrided, TIn, OutTensor, TransformFn> {
  using TOut = typename OutTensor::ElementType;
  using OutLayout = typename OutTensor::Layout;

  static constexpr nint_t required_aux_size(const OutLayout&) { return 0; }

  DataOutputImpl(TOut* p, const OutLayout& layout, const TransformFn& fn, void*)
      : _p(p), _layout(layout), _fn(fn) {}

  template <TLV_DECL_TAG(Ti), typename N, typename... Is>
  void operator()(Ti ti, vec::Vec<Ti> v, N n, Is... is) const {
    static_assert(sizeof...(Is) == OutLayout::Ndim, "coordinate count must match output rank");
    using To = vec::Rebind<TOut, Ti>;
    std::array<nint_t, OutLayout::Ndim> coords{static_cast<nint_t>(is)...};
    const nint_t base = offset_from_coords(_layout, coords);
    auto v_out = _fn.call(To{}, v, transform_x<OutLayout>(coords), coords[OutLayout::Ndim - 1]);
    scatter_dispatch(To{}, _p, base, _layout.strides()[OutLayout::Ndim - 1], n, v_out);
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
 * - neither trailing dimension contiguous: gather each logical row into aux in
 *   the constructor, then serve `operator()` from aux.
 *
 * The aux buffer, when required, covers the whole logical layout. Its last
 * dimension is padded to a vector lane count so callers can issue vector
 * loads without crossing into the next logical row. Inactive tail lanes are
 * returned as zero after the transform.
 *
 * @warning The current transpose materialization uses a simple element loop
 * for correctness. It is the replaceable hook for future low-level block
 * transpose kernels; strided rows already use the vector gather path.
 */
template <typename TOut, typename InTensor, typename TransformFn>
struct DataInput {
  using TIn = typename InTensor::ElementType;
  using InLayout = typename InTensor::Layout;
  using Transform = std::remove_cvref_t<TransformFn>;
  using Kind = details::SelectAccessKind<InLayout>;
  using Impl = details::DataInputImpl<Kind, TOut, InTensor, Transform>;

  static nint_t required_aux_size(const InLayout& layout) {
    return Impl::required_aux_size(layout);
  }

  DataInput(const TIn* p, const InLayout& layout, const Transform& fn, void* aux)
      : _impl(p, layout, fn, aux) {}

  template <TLV_DECL_TAG(To), TL_IF(is_any<vec::TypeOf<To>, TOut>), typename N, typename... Is>
  vec::Vec<To> operator()(To t, N n, Is... is) const {
    return _impl(t, n, is...);
  }

  Impl _impl;
};

/**
 * @brief Compile-time description of an input stream.
 *
 * `InputSpec` bundles the source element type, output element type, input
 * layout and transform function. Kernels can inspect the type to specialize
 * fast paths, and can call `make_input()` to bind a concrete pointer and aux
 * buffer for execution.
 */
template <
    typename TOut,
    typename TIn,
    typename InLayout,
    typename TransformFn = IdentityConversionVecFn<TOut, TIn>
> struct InputSpec {
  using InputLayout = InLayout;
  using Transform = std::remove_cvref_t<TransformFn>;
  using InputTensor = Tensor<TIn, typename InLayout::Shape, typename InLayout::Strides>;
  using InputAccessor = DataInput<TOut, InputTensor, Transform>;

  static_assert(is_any<typename Transform::TIn, TIn>, "Input type of transform fn mismatch");
  static_assert(is_any<typename Transform::TOut, TOut>, "Output type of transform fn mismatch");

  static nint_t required_aux_size(const InLayout& layout) {
    return InputAccessor::required_aux_size(layout);
  }

  InputSpec(const InLayout& layout, const Transform& fn = identity<TOut, TIn>)
      : _in_layout(layout), _fn(fn) {}

  const InLayout& input_layout() const { return _in_layout; }

  InputAccessor make_input(const TIn* p, void* aux) const {
    return InputAccessor(p, _in_layout, _fn, aux);
  }

private:
  InLayout _in_layout;
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
 * - neither trailing dimension contiguous: transform and scatter directly in
 *   `operator()`.
 *
 * The second-last-contiguous path uses the same padded aux layout as
 * `DataInput`. Only logical elements are flushed; padded tail cells remain
 * internal scratch space.
 *
 * @warning Scatter with duplicate logical addresses has the same ordering
 * caveat as the underlying vector scatter primitive.
 */
template <typename TIn, typename OutTensor, typename TransformFn>
struct DataOutput {
  using TOut = typename OutTensor::ElementType;
  using OutLayout = typename OutTensor::Layout;
  using Transform = std::remove_cvref_t<TransformFn>;
  using Kind = details::SelectAccessKind<OutLayout>;
  using Impl = details::DataOutputImpl<Kind, TIn, OutTensor, Transform>;

  static nint_t required_aux_size(const OutLayout& layout) {
    return Impl::required_aux_size(layout);
  }

  DataOutput(TOut* p, const OutLayout& layout, const Transform& fn, void* aux)
      : _impl(p, layout, fn, aux) {}

  template <TLV_DECL_TAG(Ti), TL_IF(is_any<vec::TypeOf<Ti>, TIn>), typename N, typename... Is>
  void operator()(Ti t, vec::Vec<Ti> v, N n, Is... is) const {
    _impl(t, v, n, is...);
  }

  Impl _impl;
};

/**
 * @brief Compile-time description of an output stream.
 *
 * `OutputSpec` bundles the accumulator/input element type, stored output
 * element type, output layout and transform function. `make_output()` binds a
 * destination pointer and optional aux buffer to produce a `DataOutput`.
 */
template <
    typename TIn,
    typename TOut,
    typename OutLayout,
    typename TransformFn = IdentityConversionVecFn<TOut, TIn>
> struct OutputSpec {
  using OutputLayout = OutLayout;
  using Transform = std::remove_cvref_t<TransformFn>;
  using OutputTensor = Tensor<TOut, typename OutLayout::Shape, typename OutLayout::Strides>;
  using OutputAccessor = DataOutput<TIn, OutputTensor, Transform>;

  static_assert(is_any<typename Transform::TIn, TIn>, "Input type of transform fn mismatch");
  static_assert(is_any<typename Transform::TOut, TOut>, "Output type of transform fn mismatch");

  static nint_t required_aux_size(const OutLayout& layout) {
    return OutputAccessor::required_aux_size(layout);
  }

  OutputSpec(const OutLayout& layout, const Transform& fn = identity<TOut, TIn>)
      : _out_layout(layout), _fn(fn) {}

  const OutLayout& output_layout() const { return _out_layout; }

  OutputAccessor make_output(TOut* p, void* aux) const {
    return OutputAccessor(p, _out_layout, _fn, aux);
  }

private:
  OutLayout _out_layout;
  Transform _fn;
};

namespace details {

template <typename T>
struct IsInputSpec : std::false_type {};
template <typename TOut, typename TIn, typename L, typename Fn>
struct IsInputSpec<InputSpec<TOut, TIn, L, Fn>> : std::true_type {};

template <typename T>
struct IsOutputSpec : std::false_type {};
template <typename TIn, typename TOut, typename L, typename Fn>
struct IsOutputSpec<OutputSpec<TIn, TOut, L, Fn>> : std::true_type {};

} // namespace details

/**
 * @brief True when an InputSpec transform is the zero-vector function.
 */
template <typename Spec>
static constexpr bool is_zero_input_spec_v =
    is_zeros_fn<typename std::remove_cvref_t<Spec>::Transform>;

/**
 * @brief True when an OutputSpec transform is the zero-vector function.
 */
template <typename Spec>
static constexpr bool is_zero_output_spec_v =
    is_zeros_fn<typename std::remove_cvref_t<Spec>::Transform>;

/**
 * @brief True when an InputSpec is identity and can read last-dim contiguous memory.
 */
template <typename Spec>
static constexpr bool is_identity_last_contiguous_input_spec_v =
    is_identity_fn<typename std::remove_cvref_t<Spec>::Transform> &&
    details::IsLastDimContiguous<typename std::remove_cvref_t<Spec>::InputLayout>::value;

/**
 * @brief True when an OutputSpec is identity and can write last-dim contiguous memory.
 */
template <typename Spec>
static constexpr bool is_identity_last_contiguous_output_spec_v =
    is_identity_fn<typename std::remove_cvref_t<Spec>::Transform> &&
    details::IsLastDimContiguous<typename std::remove_cvref_t<Spec>::OutputLayout>::value;

/**
 * @brief True when an InputSpec is identity and matches the second-last contiguous path.
 */
template <typename Spec>
static constexpr bool is_identity_second_last_contiguous_input_spec_v =
    is_identity_fn<typename std::remove_cvref_t<Spec>::Transform> &&
    !details::IsLastDimContiguous<typename std::remove_cvref_t<Spec>::InputLayout>::value &&
    details::IsSecondLastDimContiguous<typename std::remove_cvref_t<Spec>::InputLayout>::value;

/**
 * @brief True when an OutputSpec is identity and matches the second-last contiguous path.
 */
template <typename Spec>
static constexpr bool is_identity_second_last_contiguous_output_spec_v =
    is_identity_fn<typename std::remove_cvref_t<Spec>::Transform> &&
    !details::IsLastDimContiguous<typename std::remove_cvref_t<Spec>::OutputLayout>::value &&
    details::IsSecondLastDimContiguous<typename std::remove_cvref_t<Spec>::OutputLayout>::value;

} // namespace vecops::gemm

#endif // VECOPS_DATAACCESS_H
