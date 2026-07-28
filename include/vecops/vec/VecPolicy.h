#ifndef VECOPS_VECPOLICY_H
#define VECOPS_VECPOLICY_H

#include <type_traits>
#include <utility>

#include "vecops/CoreTypes.h"

namespace vecops::vec {

namespace cvt {

struct ordered_t {};
struct unordered_t {};

template <int Phase>
struct lane_t {
  static constexpr int phase = Phase;
};

struct saturate_t {};
struct wrap_t {};

inline constexpr ordered_t ordered{};
inline constexpr unordered_t unordered{};
template <int Phase>
inline constexpr lane_t<Phase> lane{};
inline constexpr saturate_t saturate{};
inline constexpr wrap_t wrap{};
// Integer truncation is precisely the low-bit/wrapping policy.
inline constexpr wrap_t truncate{};

} // namespace cvt

namespace mem {

struct unaligned_t {};
struct aligned_t {};
struct temporal_t {};
struct non_temporal_t {};
struct packed_t {};
struct split_t {};

inline constexpr unaligned_t unaligned{};
inline constexpr aligned_t aligned{};
inline constexpr temporal_t temporal{};
inline constexpr non_temporal_t non_temporal{};
inline constexpr packed_t packed{};
inline constexpr split_t split{};

} // namespace mem

namespace opt {

struct zero_t {};

template <typename M, bool IsSized = requires { sizeof(M); }>
struct masked_t {
  M value;
};

template <typename M>
struct masked_t<M, false> {
  M& value;
};

struct first_t {
  nint_t value;
};

template <typename V, bool IsSized = requires { sizeof(V); }>
struct merge_t {
  V value;
};

template <typename V>
struct merge_t<V, false> {
  V& value;
};

inline constexpr zero_t zero{};

template <typename M>
constexpr auto masked(M&& value) {
  using Raw = std::remove_reference_t<M>;
  if constexpr (requires { sizeof(Raw); }) {
    using D = std::decay_t<M>;
    return masked_t<D>{std::forward<M>(value)};
  } else {
    static_assert(
        std::is_lvalue_reference_v<M&&>,
        "sizeless masks passed to opt::masked must be named lvalues");
    return masked_t<Raw>{value};
  }
}

constexpr auto first(nint_t value) {
  return first_t{value};
}

template <typename V>
constexpr auto merge(V&& value) {
  using Raw = std::remove_reference_t<V>;
  if constexpr (requires { sizeof(Raw); }) {
    using D = std::decay_t<V>;
    return merge_t<D>{std::forward<V>(value)};
  } else {
    static_assert(
        std::is_lvalue_reference_v<V&&>,
        "sizeless vectors passed to opt::merge must be named lvalues");
    return merge_t<Raw>{value};
  }
}

} // namespace opt

namespace policy_details {

template <typename T>
using remove_cvref_t = std::remove_cv_t<std::remove_reference_t<T>>;

template <typename T>
inline constexpr bool is_ordered =
    std::is_same_v<remove_cvref_t<T>, cvt::ordered_t>;

template <typename T>
inline constexpr bool is_unordered =
    std::is_same_v<remove_cvref_t<T>, cvt::unordered_t>;

template <typename T>
struct is_lane_impl : std::false_type {};

template <int Phase>
struct is_lane_impl<cvt::lane_t<Phase>> : std::true_type {};

template <typename T>
inline constexpr bool is_lane = is_lane_impl<remove_cvref_t<T>>::value;

template <typename T>
inline constexpr bool is_layout = is_ordered<T> || is_unordered<T> || is_lane<T>;

template <typename T>
inline constexpr bool is_saturate =
    std::is_same_v<remove_cvref_t<T>, cvt::saturate_t>;

template <typename T>
inline constexpr bool is_wrap =
    std::is_same_v<remove_cvref_t<T>, cvt::wrap_t>;

template <typename T>
inline constexpr bool is_value_policy = is_saturate<T> || is_wrap<T>;

template <typename T>
inline constexpr bool is_aligned =
    std::is_same_v<remove_cvref_t<T>, mem::aligned_t>;

template <typename T>
inline constexpr bool is_unaligned =
    std::is_same_v<remove_cvref_t<T>, mem::unaligned_t>;

template <typename T>
inline constexpr bool is_alignment = is_aligned<T> || is_unaligned<T>;

template <typename T>
inline constexpr bool is_temporal =
    std::is_same_v<remove_cvref_t<T>, mem::temporal_t>;

template <typename T>
inline constexpr bool is_non_temporal =
    std::is_same_v<remove_cvref_t<T>, mem::non_temporal_t>;

template <typename T>
inline constexpr bool is_temporality = is_temporal<T> || is_non_temporal<T>;

template <typename T>
inline constexpr bool is_packed =
    std::is_same_v<remove_cvref_t<T>, mem::packed_t>;

template <typename T>
inline constexpr bool is_split =
    std::is_same_v<remove_cvref_t<T>, mem::split_t>;

template <typename T>
inline constexpr bool is_packing = is_packed<T> || is_split<T>;

template <typename T>
struct is_masked_impl : std::false_type {};

template <typename M, bool IsSized>
struct is_masked_impl<opt::masked_t<M, IsSized>> : std::true_type {};

template <typename T>
inline constexpr bool is_masked = is_masked_impl<remove_cvref_t<T>>::value;

template <typename T>
inline constexpr bool is_first =
    std::is_same_v<remove_cvref_t<T>, opt::first_t>;

template <typename T>
inline constexpr bool is_zero =
    std::is_same_v<remove_cvref_t<T>, opt::zero_t>;

template <typename T>
struct is_merge_impl : std::false_type {};

template <typename V, bool IsSized>
struct is_merge_impl<opt::merge_t<V, IsSized>> : std::true_type {};

template <typename T>
inline constexpr bool is_merge = is_merge_impl<remove_cvref_t<T>>::value;

template <typename T>
inline constexpr bool is_population = is_zero<T> || is_merge<T>;

template <typename T>
inline constexpr bool is_active = is_masked<T> || is_first<T>;

template <typename T>
inline constexpr bool is_memory_option =
    is_alignment<T> || is_temporality<T> || is_active<T> || is_population<T>;

template <typename T>
inline constexpr bool is_memory_conversion_option =
    is_memory_option<T> || is_packing<T>;

template <typename T>
inline constexpr bool is_conversion_option =
    is_layout<T> || is_value_policy<T> || is_population<T>;

template <template <typename> class Pred, typename... Options>
inline constexpr int count_options = (int(Pred<Options>::value) + ... + 0);

template <typename T>
struct IsLayout : std::bool_constant<is_layout<T>> {};
template <typename T>
struct IsValuePolicy : std::bool_constant<is_value_policy<T>> {};
template <typename T>
struct IsAlignment : std::bool_constant<is_alignment<T>> {};
template <typename T>
struct IsTemporality : std::bool_constant<is_temporality<T>> {};
template <typename T>
struct IsPacking : std::bool_constant<is_packing<T>> {};
template <typename T>
struct IsActive : std::bool_constant<is_active<T>> {};
template <typename T>
struct IsPopulation : std::bool_constant<is_population<T>> {};

template <template <typename> class Pred, typename Default>
constexpr decltype(auto) select_option(Default&& value) {
  return std::forward<Default>(value);
}

template <template <typename> class Pred, typename Default,
          typename First, typename... Rest>
constexpr decltype(auto) select_option(
    Default&& default_value, First&& first, Rest&&... rest) {
  if constexpr (Pred<remove_cvref_t<First>>::value) {
    return std::forward<First>(first);
  } else {
    return select_option<Pred>(
        std::forward<Default>(default_value),
        std::forward<Rest>(rest)...);
  }
}

template <template <typename> class Pred, typename Default,
          typename... Options>
using selected_option_t = remove_cvref_t<decltype(select_option<Pred>(
    std::declval<Default>(), std::declval<Options>()...))>;

} // namespace policy_details
} // namespace vecops::vec

#endif // VECOPS_VECPOLICY_H
