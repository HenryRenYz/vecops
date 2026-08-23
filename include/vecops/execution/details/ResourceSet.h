//
// Copyright (c) vecops contributors.
//

#ifndef VECOPS_EXECUTION_DETAILS_RESOURCE_SET_H
#define VECOPS_EXECUTION_DETAILS_RESOURCE_SET_H

#include <concepts>
#include <type_traits>

namespace vecops::execution::details {

/**
 * @file ResourceSet.h
 * @brief Type-level sets used as compile-time proofs of active CPU resources.
 *
 * Resource tags have no runtime payload. Concatenation intentionally preserves
 * duplicates: membership is sufficient for dispatch, and backend validation
 * diagnoses incompatible combinations such as ARM Streaming + NonStreaming.
 */

/** Type list of resources required by or active in an execution scope. */
template <typename... Resources>
struct ResourceSet {};

/** Metafunction concatenating zero or more `ResourceSet` instances. */
template <typename... Sets>
struct ConcatResourceSets;

template <>
struct ConcatResourceSets<> {
  using type = ResourceSet<>;
};

template <typename... Resources>
struct ConcatResourceSets<ResourceSet<Resources...>> {
  using type = ResourceSet<Resources...>;
};

template <typename... Resources, typename... Rest>
struct ConcatResourceSets<ResourceSet<Resources...>, ResourceSet<Rest...>> {
  using type = ResourceSet<Resources..., Rest...>;
};

template <typename... Resources, typename... Rest, typename... Tail>
struct ConcatResourceSets<
    ResourceSet<Resources...>, ResourceSet<Rest...>, Tail...> {
  using type = typename ConcatResourceSets<
      ResourceSet<Resources..., Rest...>, Tail...>::type;
};

template <typename... Sets>
using concat_resource_sets_t = typename ConcatResourceSets<Sets...>::type;

/** Metafunction testing whether `Resource` occurs in `Set`. */
template <typename Resource, typename Set>
struct HasResource;

template <typename Resource, typename... Resources>
struct HasResource<Resource, ResourceSet<Resources...>>
    : std::bool_constant<(std::same_as<Resource, Resources> || ...)> {};

template <typename Resource, typename Set>
inline constexpr bool has_resource_v = HasResource<Resource, Set>::value;

/** Dependent false value used to delay unsupported-backend diagnostics. */
template <typename...>
inline constexpr bool dependent_false_v = false;

} // namespace vecops::execution::details

#endif // VECOPS_EXECUTION_DETAILS_RESOURCE_SET_H
