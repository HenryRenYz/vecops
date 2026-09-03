//
// Copyright (c) vecops contributors.
//

#ifndef VECOPS_MATMUL_FAMILY_H
#define VECOPS_MATMUL_FAMILY_H

#include <concepts>
#include <string_view>
#include <type_traits>

/**
 * @file vecops/matmul/Family.h
 * @brief Matmul kernel family tags, their metadata, and the user-facing
 *        family selection policy.
 *
 * A "family" names one implementation strategy for the matmul product:
 * either an architecture backend leaf (the `kernel_family::ArchitectureFamily`
 * tags) or the backend-independent generic cache tiler (`GenericTiled`).
 * `WholeProblem` is special: it is the composite automatic planner that
 * decomposes a problem into the backend's specialized leaves.
 *
 * Users rarely spell family tags directly; they select one through the
 * `family_selection` policy (`Automatic`, `Prefer<F>`, `Require<F>`) passed
 * to `ops::MatmulConfig`, and read back which family served a call through
 * `kernel_family::Info<F>::name`.
 *
 * ## Key components
 *
 * | Component                          | Purpose                           |
 * |------------------------------------|-----------------------------------|
 * | `kernel_family::*` tags            | One tag per implementation strategy |
 * | `kernel_family::ArchitectureFamily`| Concept for backend leaf families |
 * | `kernel_family::Family`            | Architecture families + `GenericTiled` |
 * | `kernel_family::Info<F>`           | Per-family metadata (`name`, `composite`) |
 * | `family_selection::*`              | Automatic / Prefer / Require policy |
 * | `details::FamilyDispatch`          | Internal plan-to-leaf contract    |
 *
 * ## Usage overview
 *
 * @code
 * #include "vecops/matmul/Family.h"
 *
 * using vecops::matmul::kernel_family;
 * using vecops::matmul::family_selection;
 *
 * // Prefer the packed-input MMLA kernels, but keep the automatic planner
 * // as a fallback for shapes it cannot serve:
 * using Selection = family_selection::Prefer<kernel_family::PackedMMLA>;
 * // ops::MatmulConfig<Atom, Selection, ...> config;
 *
 * static_assert(kernel_family::Family<Selection::Family>);
 * @endcode
 *
 * ## Pitfalls
 *
 * - Family availability is backend- and shape-dependent; whether a given
 *   leaf can serve a call is decided by the planner, not by this header.
 * - `Require<F>` turns an inapplicable family into a hard configuration
 *   error (compile-time static_assert or a runtime assertion in the backend
 *   leaf); use `Prefer<F>` when a fallback is acceptable.
 * - `GenericTiled` is deliberately **not** an `ArchitectureFamily`: it is a
 *   portable composite tiler, not a backend leaf. Both are `Family`s.
 */

namespace vecops::matmul {

/**
 * @brief Tags naming each matmul implementation strategy.
 *
 * See the file header for the selection flow; the tags themselves are empty
 * marker types.
 */
namespace kernel_family {

/** Automatic whole-problem planner, including architecture-specialized leaves. */
struct WholeProblem {};

/** General MC/NC/KC cache tiler feeding the backend Tile2D scheduler. */
struct GenericTiled {};

/** The architecture backend's general whole-problem tile kernel. */
struct General {};

/** Small or skinny vector kernels that bypass matrix-tile setup. */
struct SmallVector {};

/** Fused FP32-to-INT8 activation quantization with a packed INT8 weight. */
struct RuntimeQuantInt8 {};

/** Ordinary-vector packed-input MMLA kernels. */
struct PackedMMLA {};

/** Architecture-specific packed tail decomposition. */
struct PackedTail {};

/**
 * @brief A matmul family provided by the architecture backend.
 *
 * This covers the backend's executable leaf kernels (e.g. `PackedMMLA`,
 * `RuntimeQuantInt8`) as well as its own `WholeProblem` planner. The
 * portable `GenericTiled` tiler is deliberately excluded: it composes the
 * backend's tile kernels from outside and is not a backend itself, so it is
 * only a `Family`.
 */
template <typename T>
concept ArchitectureFamily =
    std::same_as<T, WholeProblem> || std::same_as<T, General> ||
    std::same_as<T, SmallVector> ||
    std::same_as<T, RuntimeQuantInt8> ||
    std::same_as<T, PackedMMLA> || std::same_as<T, PackedTail>;

/**
 * @brief Any selectable matmul family: an architecture leaf or the generic
 *        cache tiler.
 */
template <typename T>
concept Family = ArchitectureFamily<T> || std::same_as<T, GenericTiled>;

/**
 * @brief Static metadata for one kernel family.
 *
 * The primary template is never instantiated; one specialization exists per
 * family tag, filling in:
 *
 * - `name`: a stable, human-readable identifier (e.g. `"whole_problem"`),
 *   used for diagnostics and reported by the public ops layer; and
 * - `composite`: `true` when the family is a planner that decomposes the
 *   problem into other families rather than executing a leaf kernel itself
 *   (only `WholeProblem` today), `false` for directly executable leaves.
 *
 * @tparam FamilyT  The family tag to describe.
 */
template <Family FamilyT>
struct Info;

// One specialization per family tag; each only fills in `name` and
// `composite` as documented on the primary template.
template <>
struct Info<WholeProblem> {
  static constexpr std::string_view name = "whole_problem";
  static constexpr bool composite = true;
};

template <>
struct Info<GenericTiled> {
  static constexpr std::string_view name = "generic_tiled";
  static constexpr bool composite = false;
};

template <>
struct Info<General> {
  static constexpr std::string_view name = "general";
  static constexpr bool composite = false;
};

template <>
struct Info<SmallVector> {
  static constexpr std::string_view name = "small_vector";
  static constexpr bool composite = false;
};

template <>
struct Info<RuntimeQuantInt8> {
  static constexpr std::string_view name = "runtime_quant_int8";
  static constexpr bool composite = false;
};

template <>
struct Info<PackedMMLA> {
  static constexpr std::string_view name = "packed_mmla";
  static constexpr bool composite = false;
};

template <>
struct Info<PackedTail> {
  static constexpr std::string_view name = "packed_tail";
  static constexpr bool composite = false;
};

} // namespace kernel_family

/**
 * @brief User-facing policy for which kernel family serves a matmul.
 *
 * The selected policy is the second template argument of
 * `ops::MatmulConfig` and resolves to one `kernel_family` tag.
 */
namespace family_selection {

/**
 * @brief How strictly a family selection is enforced.
 *
 * - `automatic`: let the planner choose (resolves to `WholeProblem`).
 * - `prefer`: use the requested family when it applies; fall back to the
 *   automatic planner otherwise.
 * - `require`: the requested family must serve the call; inapplicable
 *   families become configuration errors instead of falling back.
 */
enum class Mode {
  automatic,
  prefer,
  require,
};

/** Let the library choose among every family made available by the backend. */
struct Automatic {
  static constexpr Mode mode = Mode::automatic;
};

/**
 * @brief Prefer one family and retain the automatic planner as its fallback.
 *
 * If the family is applicable to the problem it is used; otherwise the
 * planner silently falls back to the automatic whole-problem route.
 *
 * @tparam FamilyT  The family to prefer; must satisfy
 *                  `kernel_family::Family` (static_assert).
 */
template <typename FamilyT>
struct Prefer {
  static_assert(kernel_family::Family<FamilyT>,
                "unknown matmul kernel family");
  using Family = FamilyT;
  static constexpr Mode mode = Mode::prefer;
};

/**
 * @brief Require one family; an inapplicable family is a configuration error.
 *
 * Unlike `Prefer`, there is no fallback: a required family that cannot
 * serve the problem fails with a compile-time static_assert (or a runtime
 * assertion in backend leaves whose applicability is only known at run
 * time). Use this to pin down codegen in benchmarks and tests.
 *
 * @tparam FamilyT  The family to require; must satisfy
 *                  `kernel_family::Family` (static_assert).
 */
template <typename FamilyT>
struct Require {
  static_assert(kernel_family::Family<FamilyT>,
                "unknown matmul kernel family");
  using Family = FamilyT;
  static constexpr Mode mode = Mode::require;
};

} // namespace family_selection

namespace details {

/** Compile-time contract carried from the operation plan into a backend leaf. */
template <kernel_family::ArchitectureFamily FamilyT,
          family_selection::Mode ModeV>
struct FamilyDispatch {
  using Family = FamilyT;
  static constexpr family_selection::Mode mode = ModeV;
  // Backends static_assert against `required` to reject an inapplicable
  // family outright instead of silently falling back to another leaf.
  static constexpr bool required = mode == family_selection::Mode::require;
};

/// The dispatch used when the user left family selection automatic:
/// the whole-problem planner with fallback allowed.
using AutomaticFamilyDispatch = FamilyDispatch<
    kernel_family::WholeProblem, family_selection::Mode::automatic>;

} // namespace details

} // namespace vecops::matmul

#endif // VECOPS_MATMUL_FAMILY_H
