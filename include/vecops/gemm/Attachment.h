//
// Created by renyz on 2026/6/10.
//

#ifndef VECOPS_ATTACHMENT_H
#define VECOPS_ATTACHMENT_H

#include "vecops/CoreTypes.h"
#include "vecops/vec/Vec.h"

#include <type_traits>

namespace vecops::gemm {
namespace details {

template <typename To, typename Fn, typename Ei, typename = void>
struct can_call_positional : std::false_type {};

template <typename To, typename Fn, typename Ei>
struct can_call_positional<To, Fn, Ei, std::void_t<decltype(
    std::declval<Fn>()(std::declval<To>(),
                       std::declval<vec::Vec<vec::Rebind<Ei, To>>>(),
                       std::declval<nint_t>(),
                       std::declval<nint_t>())
)>> : std::true_type {};

template <typename To, typename Fn, typename Ei, typename = void>
struct can_call_elementwise : std::false_type {};

template <typename To, typename Fn, typename Ei>
struct can_call_elementwise<To, Fn, Ei, std::void_t<decltype(
    std::declval<Fn>()(std::declval<To>(),
                       std::declval<vec::Vec<vec::Rebind<Ei, To>>>())
)>> : std::true_type {};

} // namespace details

/**
 * 用于各种Prologue，Epilogue的转换函数，抽象上看行为类似类型转换函数vec::promote, vec::demote, vec::convert等。
 * 接受各种长度的输入输出（在向量系统可以接收的范围内），以及一个抽象二维座标，座标含义由函数调用方和函数实现方约定，这里不假定其含义。
 * 此函数一般用于实现Gemm的后继激活函数融合、前后量化/反量化、FlashAttention的rowwise scale融合等操作。
 *
 * 此函数的实现可以包含成员变量和构造器参数，然后在call函数中使用这些信息，因此函数可以不是纯的，但函数本身不能修改任何数据。即其对于
 * 外部数据的依赖仅限于读取，不能进行写入操作。
 */
template <typename Eo, typename Ei>
struct PositionedVecFn {
  static_assert(vec::is_element_type<Ei> && vec::is_element_type<Eo>, "Ei or Eo is not supported element type");
  using TIn = Ei;
  using TOut = Eo;

  /**
   * 是否是点对点操作
   */
  static constexpr bool is_elementwise = false;

  static constexpr bool is_widening = sizeof(Eo) > sizeof(Ei);
  static constexpr bool is_narrowing = sizeof(Eo) < sizeof(Ei);
  static constexpr int size_ratio = is_widening ? sizeof(Eo) / sizeof(Ei) : sizeof(Ei) / sizeof(Eo);
  static constexpr int pow2_shift = log2_floor(size_ratio);

  static constexpr int max_input_pow2 = is_widening ? VEC_MAX_POW - pow2_shift : VEC_MAX_POW;
  static constexpr int min_input_pow2 = is_widening ? VEC_HW_MIN_POW - pow2_shift : VEC_HW_MIN_POW;
  static constexpr int max_output_pow2 = is_narrowing ? VEC_MAX_POW - pow2_shift : VEC_MAX_POW;
  static constexpr int min_output_pow2 = is_narrowing ? VEC_HW_MIN_POW - pow2_shift : VEC_HW_MIN_POW;

  /**
   * 转换函数，包含从min_output_pow2到max_output_pow2的各种长度的重载
   */
  template <TLV_DECL_TAG(To),
            TL_IF(is_any<vec::TypeOf<To>, Eo>),
            TL_IF(min_output_pow2 <= vec::scalable_pow2_of<To> && vec::scalable_pow2_of<To> <= max_output_pow2)>
  vec::Vec<To> call(To t, vec::Vec<vec::Rebind<Ei, To>> v_in, nint_t x, nint_t y) const;
};

/**
 * 点对点操作，一般用于Gemm中A, B的Prologue操作。
 */
template <typename Eo, typename Ei>
struct ElementwiseVecFn : public PositionedVecFn<Eo, Ei> {
  static constexpr bool is_elementwise = true;

  /**
   * 转换函数，包含从min_output_pow2到max_output_pow2的各种长度的重载
   */
  template <TLV_DECL_TAG(To),
      TL_IF(is_any<vec::TypeOf<To>, Eo>),
      TL_IF(ElementwiseVecFn::min_output_pow2 <= vec::scalable_pow2_of<To> && vec::scalable_pow2_of<To> <= ElementwiseVecFn::max_output_pow2)>
  vec::Vec<To> call(To t, vec::Vec<vec::Rebind<Ei, To>> v_in) const;

  /**
   * Adapter，elementwise操作等价于忽略座标的positioned操作
   */
  template <TLV_DECL_TAG(To),
      TL_IF(is_any<vec::TypeOf<To>, Eo>),
      TL_IF(ElementwiseVecFn::min_output_pow2 <= vec::scalable_pow2_of<To> && vec::scalable_pow2_of<To> <= ElementwiseVecFn::max_output_pow2)>
  vec::Vec<To> call(To t, vec::Vec<vec::Rebind<Ei, To>> v_in, nint_t x, nint_t y) const {
    ((void) x); // UNUSED
    ((void) y); // UNUSED
    return this->call(t, v_in);
  }
};

/**
 * 类型转换适配器，通过自动batch将原始VecFn通过默认类型转换修改为输入Ei, 输出Eo的外观。
 */
template <typename Eo, typename Ei, typename VecFn>
struct ConversionVecAdapter : public PositionedVecFn<Eo, Ei> {
  static constexpr bool is_elementwise = VecFn::is_elementwise;

  ConversionVecAdapter(VecFn&& fn) : _fn(std::move(fn)) { }
  ConversionVecAdapter(const VecFn& fn) : _fn(fn) { }

  template <TLV_DECL_TAG(To),
      TL_IF(is_any<vec::TypeOf<To>, Eo>),
      TL_IF(ConversionVecAdapter::min_output_pow2 <= vec::scalable_pow2_of<To> && vec::scalable_pow2_of<To> <= ConversionVecAdapter::max_output_pow2)>
  vec::Vec<To> call(To t, vec::Vec<vec::Rebind<Ei, To>> v_in, nint_t x, nint_t y) const {
    using Ti = vec::Rebind<Ei, To>;
    using InnerEi = typename VecFn::TIn;
    using InnerEo = typename VecFn::TOut;
    constexpr int pow2_in = vec::scalable_pow2_of<Ti>;

    // 输入转换为VecFn输入后可以装下
    if constexpr (pow2_in <= VecFn::max_input_pow2) {
      if (VecFn::min_input_pow2 <= pow2_in) {
        vec::Rebind<InnerEi, Ti> t_ii;
        vec::Rebind<InnerEo, Ti> t_io;
        auto inner_in = vec::xconvert(t_ii, v_in);
        auto inner_out = _fn.call(t_io, inner_in, x, y);
        return vec::xconvert(t, inner_out);
      } else {
        vec::Rebind<InnerEi, Ti> t_ii;
        vec::ScalableTag<InnerEi, VecFn::min_input_pow2> t_ix;
        vec::Rebind<InnerEo, decltype(t_ix)> t_ox;
        vec::Rebind<InnerEo, Ti> t_io;
        auto inner_in = vec::bitcast(t_ix, vec::xconvert(t_ii, v_in)); // TODO is zero-extending better?
        auto inner_out = _fn.call(t_ox, inner_in, x, y);
        return vec::xconvert(t, vec::bitcast(t_io, inner_out));
      }
    } else { // 否则需要劈成两半处理，且需要递归处理
      vec::Half<Ti> t_h;
      auto v_lo = this->call(t_h, vec::lower(t, v_in), x, y);
      auto v_hi = this->call(t_h, vec::upper(t, v_lo), x, y);
      return vec::concat(t, v_lo, v_hi);
    }
  }

  template <TLV_DECL_TAG(To),
      TL_IF(is_any<vec::TypeOf<To>, Eo>),
      TL_IF(is_elementwise),
      TL_IF(ConversionVecAdapter::min_output_pow2 <= vec::scalable_pow2_of<To> && vec::scalable_pow2_of<To> <= ConversionVecAdapter::max_output_pow2)>
  vec::Vec<To> call(To t, vec::Vec<vec::Rebind<Ei, To>> v_in) const {
    return this->call(t, v_in, 0, 0);
  }

private:
  VecFn _fn;
};

/**
 * Lambda适配器，将仿函数Fn包装为PositionedVecFn外观。
 * Fn可能只实现了部分POW2级别的转换，本适配器通过向上bitcast和向下拆分自动找到可用的POW2级别。
 */
template <typename Eo, typename Ei, typename Fn>
struct PositionalLambdaVecAdapter : public PositionedVecFn<Eo, Ei> {

  PositionalLambdaVecAdapter(Fn&& fn) : _fn(std::move(fn)) { }
  PositionalLambdaVecAdapter(const Fn& fn) : _fn(fn) { }

  template <TLV_DECL_TAG(To),
      TL_IF(is_any<vec::TypeOf<To>, Eo>),
      TL_IF(PositionalLambdaVecAdapter::min_output_pow2 <= vec::scalable_pow2_of<To> && vec::scalable_pow2_of<To> <= PositionalLambdaVecAdapter::max_output_pow2)>
  vec::Vec<To> call(To t, vec::Vec<vec::Rebind<Ei, To>> v_in, nint_t x, nint_t y) const {
    constexpr int pow2 = vec::scalable_pow2_of<To>;

    if constexpr (details::can_call_positional<To, Fn, Ei>::value) {
      return _fn(t, v_in, x, y);
    } else {
      return _try_upward<To, pow2 + 1>(t, v_in, x, y);
    }
  }

private:
  Fn _fn;

  template <typename To, int TryPow2>
  vec::Vec<To> _try_upward(To t, vec::Vec<vec::Rebind<Ei, To>> v_in, nint_t x, nint_t y) const {
    if constexpr (TryPow2 > PositionalLambdaVecAdapter::max_input_pow2) {
      return _try_downward<To>(t, v_in, x, y);
    } else {
      using TryOut = vec::ScalableTag<Eo, TryPow2>;
      using TryIn  = vec::Rebind<Ei, TryOut>;

      if constexpr (details::can_call_positional<TryOut, Fn, Ei>::value) {
        auto cast_in  = vec::bitcast(TryIn{}, v_in);
        auto cast_out = _fn(TryOut{}, cast_in, x, y);
        return vec::bitcast(To{}, cast_out);
      } else {
        return _try_upward<To, TryPow2 + 1>(t, v_in, x, y);
      }
    }
  }

  template <typename To>
  vec::Vec<To> _try_downward(To t, vec::Vec<vec::Rebind<Ei, To>> v_in, nint_t x, nint_t y) const {
    constexpr int pow2 = vec::scalable_pow2_of<To>;

    if constexpr (details::can_call_positional<To, Fn, Ei>::value) {
      return _fn(t, v_in, x, y);
    } else if constexpr (pow2 > PositionalLambdaVecAdapter::min_input_pow2) {
      using HalfTo = vec::Half<To>;
      using Ti     = vec::Rebind<Ei, To>;
      HalfTo t_h;
      auto v_lo = _try_downward<HalfTo>(
          t_h, vec::lower(Ti{}, v_in), x, y);
      auto v_hi = _try_downward<HalfTo>(
          t_h, vec::upper(Ti{}, v_in), x, y);
      return vec::concat(To{}, v_lo, v_hi);
    } else {
      static_assert(pow2 > PositionalLambdaVecAdapter::min_input_pow2,
          "PositionalLambdaVecAdapter: Fn does not support any POW2 in range");
      return {};
    }
  }
};

/**
 * 点对点Lambda适配器，将仿函数Fn包装为ElementwiseVecFn外观。
 */
template <typename Eo, typename Ei, typename Fn>
struct ElementwiseLambdaVecAdapter : public PositionedVecFn<Eo, Ei> {
  static constexpr bool is_elementwise = true;

  ElementwiseLambdaVecAdapter(Fn&& fn) : _fn(std::move(fn)) { }
  ElementwiseLambdaVecAdapter(const Fn& fn) : _fn(fn) { }

  template <TLV_DECL_TAG(To),
      TL_IF(is_any<vec::TypeOf<To>, Eo>),
      TL_IF(ElementwiseLambdaVecAdapter::min_output_pow2 <= vec::scalable_pow2_of<To> && vec::scalable_pow2_of<To> <= ElementwiseLambdaVecAdapter::max_output_pow2)>
  vec::Vec<To> call(To t, vec::Vec<vec::Rebind<Ei, To>> v_in) const {
    constexpr int pow2 = vec::scalable_pow2_of<To>;

    if constexpr (details::can_call_elementwise<To, Fn, Ei>::value) {
      return _fn(t, v_in);
    } else {
      return _try_upward<To, pow2 + 1>(t, v_in);
    }
  }

  template <TLV_DECL_TAG(To),
      TL_IF(is_any<vec::TypeOf<To>, Eo>),
      TL_IF(ElementwiseLambdaVecAdapter::min_output_pow2 <= vec::scalable_pow2_of<To> && vec::scalable_pow2_of<To> <= ElementwiseLambdaVecAdapter::max_output_pow2)>
  vec::Vec<To> call(To t, vec::Vec<vec::Rebind<Ei, To>> v_in, nint_t x, nint_t y) const {
    ((void) x);
    ((void) y);
    return this->call(t, v_in);
  }

private:
  Fn _fn;

  template <typename To, int TryPow2>
  vec::Vec<To> _try_upward(To t, vec::Vec<vec::Rebind<Ei, To>> v_in) const {
    if constexpr (TryPow2 > ElementwiseLambdaVecAdapter::max_input_pow2) {
      return _try_downward<To>(t, v_in);
    } else {
      using TryOut = vec::ScalableTag<Eo, TryPow2>;
      using TryIn  = vec::Rebind<Ei, TryOut>;

      if constexpr (details::can_call_elementwise<TryOut, Fn, Ei>::value) {
        auto cast_in  = vec::bitcast(TryIn{}, v_in);
        auto cast_out = _fn(TryOut{}, cast_in);
        return vec::bitcast(To{}, cast_out);
      } else {
        return _try_upward<To, TryPow2 + 1>(t, v_in);
      }
    }
  }

  template <typename To>
  vec::Vec<To> _try_downward(To t, vec::Vec<vec::Rebind<Ei, To>> v_in) const {
    constexpr int pow2 = vec::scalable_pow2_of<To>;

    if constexpr (details::can_call_elementwise<To, Fn, Ei>::value) {
      return _fn(t, v_in);
    } else if constexpr (pow2 > ElementwiseLambdaVecAdapter::min_input_pow2) {
      using HalfTo = vec::Half<To>;
      using Ti     = vec::Rebind<Ei, To>;
      HalfTo t_h;
      auto v_lo = _try_downward<HalfTo>(
          t_h, vec::lower(Ti{}, v_in));
      auto v_hi = _try_downward<HalfTo>(
          t_h, vec::upper(Ti{}, v_in));
      return vec::concat(To{}, v_lo, v_hi);
    } else {
      static_assert(pow2 > ElementwiseLambdaVecAdapter::min_input_pow2,
          "ElementwiseLambdaVecAdapter: Fn does not support any POW2 in range");
      return {};
    }
  }
};

} // namespace vecops::gemm

#endif //VECOPS_ATTACHMENT_H
