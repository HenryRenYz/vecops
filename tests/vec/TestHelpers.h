#ifndef VECOPS_TESTS_NVEC_TESTHELPERS_H
#define VECOPS_TESTS_NVEC_TESTHELPERS_H

#include <cstdint>
#include <type_traits>
#include <utility>

#include "TestTypes.h"
#include "vecops/vec/VecBase.h"

namespace vec_test {

template <typename Visitor>
void for_each_element_type(Visitor&& visitor) {
  visitor.template operator()<vecops::bfloat16_t>();
  visitor.template operator()<vecops::float16_t>();
  visitor.template operator()<vecops::float32_t>();
  visitor.template operator()<vecops::float64_t>();
  visitor.template operator()<vecops::int8_t>();
  visitor.template operator()<vecops::uint8_t>();
  visitor.template operator()<vecops::int16_t>();
  visitor.template operator()<vecops::uint16_t>();
  visitor.template operator()<vecops::int32_t>();
  visitor.template operator()<vecops::uint32_t>();
  visitor.template operator()<vecops::int64_t>();
  visitor.template operator()<vecops::uint64_t>();
}

template <typename Visitor>
void for_each_element_pair(Visitor&& visitor) {
  for_each_element_type([&]<typename From>() {
    for_each_element_type([&]<typename To>() {
      visitor.template operator()<From, To>();
    });
  });
}

template <typename T>
bool values_identical(T expected, T actual) {
  if constexpr (
      std::same_as<T, vecops::bfloat16_t> ||
      std::same_as<T, vecops::float16_t>) {
    return expected.to_bits() == actual.to_bits();
  } else if constexpr (std::same_as<T, vecops::float32_t>) {
    return ::vecops::bitcast<std::uint32_t>(expected) ==
           ::vecops::bitcast<std::uint32_t>(actual);
  } else if constexpr (std::same_as<T, vecops::float64_t>) {
    return ::vecops::bitcast<std::uint64_t>(expected) ==
           ::vecops::bitcast<std::uint64_t>(actual);
  } else {
    return expected == actual;
  }
}

namespace details {

consteval int floor_log2(vecops::nint_t value) {
  int result = 0;
  while (value > 1) {
    value >>= 1;
    ++result;
  }
  return result;
}

template <typename T>
inline constexpr vecops::nint_t minimum_scalable_word_bytes =
#if defined(CPU_CAPABILITY_SVE) && !defined(HAS_FIXED_SVE_BITS)
    16;
#elif defined(CPU_CAPABILITY_SVE) && defined(HAS_FIXED_SVE_BITS)
    FIXED_SVE_BITS / 8;
#else
    VEC_WIDTH / 8;
#endif

template <typename T>
inline constexpr int minimum_scalable_power =
    -floor_log2(minimum_scalable_word_bytes<T> /
                static_cast<vecops::nint_t>(sizeof(T)));

inline constexpr int maximum_scalable_power = VEC_MAX_POW;

template <typename T, int Power, int Last, typename Visitor>
void visit_scalable_powers(Visitor& visitor) {
  visitor.template operator()<vecops::vec::ScalableTag<T, Power>>();
  if constexpr (Power < Last)
    visit_scalable_powers<T, Power + 1, Last>(visitor);
}

template <typename From, typename To, int Power, int Last, typename Visitor>
void visit_scalable_conversion_powers(Visitor& visitor) {
  using FromTag = vecops::vec::ScalableTag<From, Power>;
  using ToTag = vecops::vec::Rebind<To, FromTag>;
  constexpr int to_power = vecops::vec::scale_power<ToTag>;
  if constexpr (
      to_power >= minimum_scalable_power<To> &&
      to_power <= maximum_scalable_power)
    visitor.template operator()<FromTag, ToTag>();
  if constexpr (Power < Last)
    visit_scalable_conversion_powers<From, To, Power + 1, Last>(visitor);
}

template <typename From, typename To, vecops::nint_t N, typename Visitor>
void visit_fixed_conversion_powers(Visitor& visitor) {
  using FromTag = vecops::vec::FixedTag<From, N>;
  using ToTag = vecops::vec::Rebind<To, FromTag>;
  constexpr auto from_words = vecops::vec::num_words(FromTag{});
  constexpr auto to_words = vecops::vec::num_words(ToTag{});
  if constexpr (from_words <= 32 && to_words <= 32)
    visitor.template operator()<FromTag, ToTag>();
  if constexpr (from_words < 32 && to_words < 32)
    visit_fixed_conversion_powers<From, To, N * 2>(visitor);
}

} // namespace details

/** Visits every backend-supported scalable shape, including one-lane subwords. */
template <typename T, typename Visitor>
void for_each_scalable_shape(Visitor&& visitor) {
  auto& stable_visitor = visitor;
  details::visit_scalable_powers<
      T, details::minimum_scalable_power<T>,
      details::maximum_scalable_power>(stable_visitor);
}

/** Visits every scalable From/To pair whose two representations are supported. */
template <typename From, typename To, typename Visitor>
void for_each_scalable_conversion_shape(Visitor&& visitor) {
  auto& stable_visitor = visitor;
  details::visit_scalable_conversion_powers<
      From, To, details::minimum_scalable_power<From>,
      details::maximum_scalable_power>(stable_visitor);
}

/** Shapes on which every orthogonal Options population is instantiated. */
template <vecops::vec::VectorTag Tag>
inline constexpr bool exhaustive_options_shape = [] {
  if constexpr (vecops::vec::is_scalable_tag<Tag>) {
    using T = vecops::vec::ElementOf<Tag>;
    constexpr int power = vecops::vec::scale_power<Tag>;
    return power == details::minimum_scalable_power<T> ||
        power == 0 || power == details::maximum_scalable_power;
  } else {
    return true;
  }
}();

/** Visits every legal power-of-two FixedTag extent through 32 physical words. */
template <typename From, typename To, typename Visitor>
void for_each_fixed_conversion_shape(Visitor&& visitor) {
#if !defined(CPU_CAPABILITY_SVE) || defined(HAS_FIXED_SVE_BITS)
  auto& stable_visitor = visitor;
  details::visit_fixed_conversion_powers<From, To, 1>(stable_visitor);
#else
  (void)visitor;
#endif
}

} // namespace vec_test

#endif // VECOPS_TESTS_NVEC_TESTHELPERS_H
