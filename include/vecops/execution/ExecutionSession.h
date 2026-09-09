//
// Copyright (c) vecops contributors.
//

#ifndef VECOPS_EXECUTION_EXECUTION_SESSION_H
#define VECOPS_EXECUTION_EXECUTION_SESSION_H

#include <concepts>
#include <type_traits>
#include <utility>

#include "vecops/CoreDefs.h"
#include "vecops/execution/details/Backend.h"
#include "vecops/execution/details/ResourceSet.h"
#include "vecops/kernel/Workspace.h"

/**
 * @file vecops/execution/ExecutionSession.h
 * @brief Lexically scoped, compile-time CPU resource and configuration model.
 *
 * Operators declare a nested `ResourceRequirements` type. `with_region()` and
 * `with_resources()` combine those type-level requirements, ask the selected
 * execution backend to validate them, enter only resources not already active,
 * and pass a more specific scope type to the callback. Nested scopes therefore
 * compile away when their requirements are already satisfied.
 *
 * The model deliberately performs no runtime capability or current-state
 * detection. Missing compile-time information selects the conservative backend
 * contract or fails compilation; it never creates a runtime fast-path test.
 *
 * ## Usage
 *
 * @code
 * #include "vecops/execution/ExecutionSession.h"
 *
 * kernel::Workspace storage(bytes);
 * auto view = storage.view();
 * ExecutionSession session{view};
 *
 * session.with_region(op0, op1, [&](auto& scope) {
 *   // Inside the region the scope type proves op0/op1 resources are active;
 *   // operator calls taking a scope add no further hardware transition.
 *   op0(scope, a, b);
 *   op1(scope, c, d);
 *   // Scratch allocations share the worker workspace via mark/rewind.
 *   auto mark = scope.workspace_view().mark();
 *   ...
 *   scope.workspace_view().rewind(mark);
 * });
 * @endcode
 *
 * ## Pitfalls
 *
 * - Scope values must not escape their callback: the active-resource proof
 *   is lexical, and the underlying hardware mode (e.g. SME Streaming+ZA)
 *   is released when the region ends.
 * - `workspace_view()` asserts unless the session/scope was created with a
 *   workspace.
 * - `with_configuration()` is only valid when the configuration's coarse
 *   resource is already active (`@pre` on the method).
 * - Backends may require the region callback to be `noexcept` (ARM
 *   Streaming+ZA restores processor state in an unwind path that cannot
 *   tolerate exceptions); this is enforced by static_assert, not caught
 *   at runtime.
 */

namespace vecops::execution {

namespace details {

/** Extract an operator's requirements, defaulting to the backend contract.
 *  The void_t probe below only engages when the operator declares a nested
 *  `ResourceRequirements`; operators without one run under the backend's
 *  default requirements instead of failing to compile. */
template <typename T, typename = void>
struct RequirementsOf {
  using type = typename current_backend_t::DefaultRequirements;
};

template <typename T>
struct RequirementsOf<T, std::void_t<typename std::remove_cvref_t<T>::ResourceRequirements>> {
  using type = typename std::remove_cvref_t<T>::ResourceRequirements;
};

template <typename T>
using requirements_of_t = typename RequirementsOf<T>::type;

/** Concatenate the requirements of all operators in one region. */
template <typename... Operators>
using requirements_for_t = concat_resource_sets_t<requirements_of_t<Operators>...>;

template <typename ActiveResources, typename ActiveConfiguration = void>
class Scope;

template <typename Current, typename Required, typename Fn>
/** Forward declaration; the backend-neutral lexical transition defined
 *  below combines Current+Required, validates, and enters via the backend. */
VECOPS_ALWAYS_INLINE decltype(auto) enter_resources(kernel::WorkspaceView* workspace, Fn&& fn);

} // namespace details

/**
 * @brief Object carrying compile-time execution proofs and a worker workspace.
 *
 * A scope exposes its active resource/configuration types and provides access
 * to the per-worker workspace. Scope values must not escape their callback.
 */
template <typename T>
inline constexpr bool is_execution_scope_v = requires(T& scope) {
  typename std::remove_cvref_t<T>::ActiveResources;
  typename std::remove_cvref_t<T>::ActiveConfiguration;
  { scope.workspace_view() } -> std::same_as<kernel::WorkspaceView&>;
};

/** @brief Constraint-facing form of `is_execution_scope_v`: any type that
 *  exposes active resource/configuration types and a worker workspace. */
template <typename T>
concept ExecutionScope = is_execution_scope_v<T>;

/**
 * @brief Invoke a callback with the same compile-time resource proof as an
 * existing scope but a different worker-local workspace.
 *
 * Stateful operations use this adapter after binding their scratch storage at
 * construction time.  It performs no hardware transition: the parent scope
 * already proves that its resources/configuration are active, and only the
 * non-owning workspace pointer changes for the callback lifetime.
 */
template <ExecutionScope Scope, typename Fn>
VECOPS_ALWAYS_INLINE decltype(auto) with_workspace(Scope&, kernel::WorkspaceView& workspace, Fn&& fn) {
  using Parent = std::remove_cvref_t<Scope>;
  details::Scope<typename Parent::ActiveResources, typename Parent::ActiveConfiguration> rebound{&workspace};
  return std::forward<Fn>(fn)(rebound);
}

namespace details {

/**
 * @brief Callback-scoped proof that CPU resources/configuration are active.
 * @tparam Resources Type-level set of active coarse hardware resources.
 * @tparam CurrentConfiguration Active micro-kernel configuration type or void.
 *
 * The object itself only carries the optional per-worker workspace. Hardware
 * ownership is lexical and implemented by the selected backend around the
 * callback that receives this scope.
 */
template <typename Resources, typename CurrentConfiguration>
class Scope {
public:
  using ActiveResources = Resources;
  using ActiveConfiguration = CurrentConfiguration;

  VECOPS_ALWAYS_INLINE constexpr Scope() = default;
  VECOPS_ALWAYS_INLINE constexpr explicit Scope(kernel::WorkspaceView* workspace)
    : workspace_(workspace) {
  }

  /**
   * @brief Return the worker-local workspace associated with this scope.
   * @return Mutable workspace view.
   * @pre The creating session was constructed with a workspace.
   */
  VECOPS_ALWAYS_INLINE kernel::WorkspaceView& workspace_view() const {
    VECOPS_ASSERT(workspace_ != nullptr, "execution scope has no workspace");
    return *workspace_;
  }

  template <typename Operator, typename Fn>
  /**
   * @brief Enter resources required by one nested operator.
   * @param fn Callback receiving a scope whose type proves the union of current
   *           and requested resources.
   * @return The callback result.
   */
  VECOPS_ALWAYS_INLINE decltype(auto) with_resources(const Operator&, Fn&& fn) const {
    using Required = requirements_of_t<Operator>;
    return enter_resources<Resources, Required>(workspace_, std::forward<Fn>(fn));
  }

  template <typename NewConfiguration, typename Fn>
  /**
   * @brief Load a backend configuration and invoke `fn` under its type proof.
   * @param configuration Backend configuration object, for example TILECFG.
   * @param fn Callback receiving a scope with `ActiveConfiguration` equal to
   *           `NewConfiguration`.
   * @return The callback result.
   * @pre The configuration's required coarse resource is already active.
   */
  VECOPS_ALWAYS_INLINE decltype(auto) with_configuration(const NewConfiguration& configuration, Fn&& fn) const {
    auto invoke = [&]() VECOPS_INLINE_LAMBDA -> decltype(auto) {
      Scope<Resources, NewConfiguration> configured{workspace_};
      return std::forward<Fn>(fn)(configured);
    };
    return current_backend_t::template configure<Resources>(configuration, invoke);
  }

private:
  kernel::WorkspaceView* workspace_ = nullptr;
};

template <typename Current, typename Required, typename Fn>
/**
 * @brief Backend-neutral implementation of a lexical resource transition.
 *
 * `current_backend_t::validate` runs at compile time. The backend's `enter`
 * receives both the current and requested type sets, allowing it to omit a
 * nested transition with `if constexpr`.
 */
VECOPS_ALWAYS_INLINE decltype(auto) enter_resources(kernel::WorkspaceView* workspace, Fn&& fn) {
  using Active = concat_resource_sets_t<Current, Required>;
  current_backend_t::template validate<Active>();

  auto invoke = [&]() VECOPS_INLINE_LAMBDA_NOEXCEPT_IF(
                  noexcept(std::forward<Fn>(fn)(std::declval<Scope<Active>&>()))) -> decltype(auto) {
    Scope<Active> active{workspace};
    return std::forward<Fn>(fn)(active);
  };

  return current_backend_t::template enter<Current, Required>(invoke);
}

} // namespace details

/**
 * @brief Root of one worker's execution-resource and workspace lifetime.
 *
 * The root owns no active hardware mode in its public type. `with_region()`
 * creates lexical resource-bearing scopes. A default-constructed session is
 * valid for direct operations requiring no workspace; `workspace_view()` is
 * only valid when a view was supplied to the constructor.
 */
class ExecutionSession {
public:
  using ActiveResources = details::ResourceSet<>;
  using ActiveConfiguration = void;

  constexpr ExecutionSession() = default;

  /** Associate a non-owning worker-local workspace with this session. */
  constexpr explicit ExecutionSession(kernel::WorkspaceView& workspace)
    : workspace_(&workspace) {
  }

  /**
   * @brief Return the associated worker-local workspace.
   * @pre A workspace was supplied at construction.
   */
  VECOPS_ALWAYS_INLINE kernel::WorkspaceView& workspace_view() const {
    VECOPS_ASSERT(workspace_ != nullptr, "execution session has no workspace");
    return *workspace_;
  }

  template <typename Operator, typename Fn>
  /**
   * @brief Open a region for one operator and invoke `fn` once.
   * @param fn Callback receiving the active typed scope.
   * @return The callback result.
   */
  VECOPS_ALWAYS_INLINE decltype(auto) with_region(const Operator&, Fn&& fn) const {
    return details::enter_resources<ActiveResources, details::requirements_for_t<Operator>>(workspace_,
                                                                                            std::forward<Fn>(fn));
  }

  template <typename Operator, typename Fn>
  /** Alias of `with_region()` used by self-managing operator call surfaces. */
  VECOPS_ALWAYS_INLINE decltype(auto) with_resources(const Operator& operation, Fn&& fn) const {
    return with_region(operation, std::forward<Fn>(fn));
  }

  template <typename Op0, typename Op1, typename Fn>
  /** Open one region containing the union of two operators' requirements. */
  VECOPS_ALWAYS_INLINE decltype(auto) with_region(const Op0&, const Op1&, Fn&& fn) const {
    return details::enter_resources<ActiveResources, details::requirements_for_t<Op0, Op1>>(workspace_,
                                                                                            std::forward<Fn>(fn));
  }

  template <typename Op0, typename Op1, typename Op2, typename Fn>
  /** Open one region containing the union of three operators' requirements. */
  VECOPS_ALWAYS_INLINE decltype(auto) with_region(const Op0&, const Op1&, const Op2&, Fn&& fn) const {
    return details::enter_resources<ActiveResources, details::requirements_for_t<Op0, Op1, Op2>>(workspace_,
                                                                                                 std::forward<Fn>(fn));
  }

  template <typename Op0, typename Op1, typename Op2, typename Op3, typename Fn>
  /** Open one region containing the union of four operators' requirements. */
  VECOPS_ALWAYS_INLINE decltype(auto) with_region(const Op0&, const Op1&, const Op2&, const Op3&, Fn&& fn) const {
    return details::enter_resources<ActiveResources, details::requirements_for_t<Op0, Op1, Op2, Op3>>(
      workspace_, std::forward<Fn>(fn));
  }

private:
  kernel::WorkspaceView* workspace_ = nullptr;
};

static_assert(ExecutionScope<ExecutionSession>);

template <typename Resource, ExecutionScope Scope>
/** True when `Scope` statically proves that `Resource` is active. */
inline constexpr bool has_resource_v =
  details::has_resource_v<Resource, typename std::remove_cvref_t<Scope>::ActiveResources>;

} // namespace vecops::execution

namespace vecops {
using execution::ExecutionSession;
}

#endif // VECOPS_EXECUTION_EXECUTION_SESSION_H
