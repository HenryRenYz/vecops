#ifndef VECOPS_VEC_DETAILS_WIDENINGDOT_H
#define VECOPS_VEC_DETAILS_WIDENINGDOT_H

/**
 * @file WideningDot.h
 * @brief Legality predicates and the backend-independent grouped widening
 * dot fallback.
 */

#include <type_traits>
#include <utility>

namespace vecops::vec::details {

template <VectorTag ToTag, VectorTag FromTag1, VectorTag FromTag2>
consteval bool valid_widening_dot() {
  using To = ElementOf<ToTag>;
  using From1 = ElementOf<FromTag1>;
  using From2 = ElementOf<FromTag2>;

  if constexpr (
      std::same_as<To, From1> && std::same_as<To, From2>) {
    return std::same_as<ToTag, FromTag1> &&
        std::same_as<ToTag, FromTag2>;
  } else {
    constexpr bool all_float =
        ::vecops::is_float_v<To> &&
        ::vecops::is_float_v<From1> &&
        ::vecops::is_float_v<From2>;
    constexpr bool all_integer =
        std::integral<To> && std::integral<From1> &&
        std::integral<From2>;
    return (all_float || all_integer) &&
        sizeof(From1) == sizeof(From2) &&
        sizeof(To) > sizeof(From1) &&
        same_logical_bytes_v<ToTag, FromTag1> &&
        same_logical_bytes_v<ToTag, FromTag2>;
  }
}

template <VectorTag ToTag, TagInferableVector V, bool WidthCompatible>
struct InferredWideningDotSourceImpl {};

template <VectorTag ToTag, TagInferableVector V>
  requires std::same_as<
      std::remove_cvref_t<V>,
      Vec<ViewAs<ElementOf<VecToTag<V>>, ToTag>>>
struct InferredWideningDotSourceImpl<ToTag, V, true> {
  using Element = ElementOf<VecToTag<V>>;
  using Candidate = ViewAs<Element, ToTag>;
  using Tag = Candidate;
};

template <VectorTag ToTag, TagInferableVector V>
struct InferredWideningDotSource
    : InferredWideningDotSourceImpl<
          ToTag, V,
          (sizeof(ElementOf<VecToTag<V>>) <= sizeof(ElementOf<ToTag>))> {};

template <int Phase, int Levels, VectorTag Tag>
VECOPS_ALWAYS_INLINE auto widening_dot_select_phase(
    Tag tag, Vec<Tag> value) {
  static_assert(Phase >= 0 && Phase < (1 << Levels));
  if constexpr (Levels == 0) {
    return value;
  } else if constexpr ((Phase & 1) == 0) {
    return widening_dot_select_phase<(Phase >> 1), Levels - 1>(
        Half<Tag>{}, even(tag, value));
  } else {
    return widening_dot_select_phase<(Phase >> 1), Levels - 1>(
        Half<Tag>{}, odd(tag, value));
  }
}

template <int Levels, VectorTag Tag>
struct WideningDotSelectedTag {
  using type = typename WideningDotSelectedTag<Levels - 1, Half<Tag>>::type;
};

template <VectorTag Tag>
struct WideningDotSelectedTag<0, Tag> {
  using type = Tag;
};

template <int Levels, VectorTag Tag>
using WideningDotSelectedTagT =
    typename WideningDotSelectedTag<Levels, Tag>::type;

template <int Levels, typename Backend, VectorTag ToTag,
          VectorTag FromTag1, VectorTag FromTag2, std::size_t... Phase>
VECOPS_ALWAYS_INLINE Vec<ToTag> widening_dot_generic_phases(
    ToTag to, FromTag1 from1, FromTag2 from2,
    Vec<FromTag1> a, Vec<FromTag2> b, Vec<ToTag> initial,
    std::index_sequence<Phase...>) {
  static_assert(sizeof...(Phase) == (std::size_t{1} << Levels));
  auto result = initial;
  ([&]() VECOPS_INLINE_LAMBDA {
    constexpr int phase = static_cast<int>(Phase);
    const auto widened_a = [&]() VECOPS_INLINE_LAMBDA {
      if constexpr (Levels == 1) {
        return convert(to, from1, a, cvt::lane<phase>);
      } else {
        const auto selected =
            widening_dot_select_phase<phase, Levels>(from1, a);
        using SelectedTag = WideningDotSelectedTagT<Levels, FromTag1>;
        return convert(to, SelectedTag{}, selected);
      }
    }();
    const auto widened_b = [&]() VECOPS_INLINE_LAMBDA {
      if constexpr (Levels == 1) {
        return convert(to, from2, b, cvt::lane<phase>);
      } else {
        const auto selected =
            widening_dot_select_phase<phase, Levels>(from2, b);
        using SelectedTag = WideningDotSelectedTagT<Levels, FromTag2>;
        return convert(to, SelectedTag{}, selected);
      }
    }();
    result = fmadd(to, widened_a, widened_b, result);
  }(), ...);
  return result;
}

template <typename Backend, VectorTag ToTag>
struct GenericImpl<Backend, WideningDotOp, ToTag> {
  template <VectorTag FromTag1, VectorTag FromTag2>
    requires (valid_widening_dot<ToTag, FromTag1, FromTag2>()) &&
             (!std::same_as<ToTag, FromTag1> ||
              !std::same_as<ToTag, FromTag2>)
  static VECOPS_ALWAYS_INLINE Vec<ToTag> call(
      WideningDotOp op, ToTag to, FromTag1 from1, FromTag2 from2,
      Vec<FromTag1> a, Vec<FromTag2> b) {
    return call(op, to, from1, from2, a, b, zeros(to));
  }

  template <VectorTag FromTag1, VectorTag FromTag2>
    requires (valid_widening_dot<ToTag, FromTag1, FromTag2>()) &&
             (!std::same_as<ToTag, FromTag1> ||
              !std::same_as<ToTag, FromTag2>)
  static VECOPS_ALWAYS_INLINE Vec<ToTag> call(
      WideningDotOp, ToTag to, FromTag1 from1, FromTag2 from2,
      Vec<FromTag1> a, Vec<FromTag2> b, Vec<ToTag> c) {
    constexpr int ratio =
        static_cast<int>(sizeof(ElementOf<ToTag>) /
                         sizeof(ElementOf<FromTag1>));
    constexpr int levels = ratio == 2 ? 1 : ratio == 4 ? 2 : 3;
    static_assert(ratio == 2 || ratio == 4 || ratio == 8);
    return widening_dot_generic_phases<levels, Backend>(
        to, from1, from2, a, b, c,
        std::make_index_sequence<static_cast<std::size_t>(ratio)>{});
  }
};

} // namespace vecops::vec::details

#endif // VECOPS_VEC_DETAILS_WIDENINGDOT_H
