//
// Copyright (c) vecops contributors.
//

#ifndef VECOPS_MATMUL_FAMILY_H
#define VECOPS_MATMUL_FAMILY_H

#include <concepts>
#include <string_view>
#include <type_traits>

namespace vecops::matmul {

namespace kernel_family {

/**
 * Automatic whole-problem planner, including architecture-specialized leaves.
 * Only profitability-proven leaves are selected automatically.
 */
struct WholeProblem {};

/** General MC/NC/KC cache tiler feeding the backend Tile2D scheduler. */
struct GenericTiled {};

/** The architecture backend's general whole-problem tile kernel. */
struct General {};

/**
 * Small or skinny vector kernels that bypass matrix-tile setup. The supported
 * shape domain can be wider than the subset selected by WholeProblem.
 */
struct SmallVector {};

/** Fused FP32-to-INT8 activation quantization with a packed INT8 weight. */
struct RuntimeQuantInt8 {};

/** Packed-input vector dot kernels, including ISA-specific MMLA leaves. */
struct PackedDot {};

/** Backend-independent bulk/residual decomposition. */
struct ResidualSplit {};

template <typename T>
concept ArchitectureFamily =
    std::same_as<T, WholeProblem> || std::same_as<T, General> ||
    std::same_as<T, SmallVector> ||
    std::same_as<T, RuntimeQuantInt8> ||
    std::same_as<T, PackedDot> || std::same_as<T, ResidualSplit>;

template <typename T>
concept Family = ArchitectureFamily<T> || std::same_as<T, GenericTiled>;

template <Family FamilyT>
struct Info;

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
struct Info<PackedDot> {
  static constexpr std::string_view name = "packed_dot";
  static constexpr bool composite = false;
};

template <>
struct Info<ResidualSplit> {
  static constexpr std::string_view name = "residual_split";
  static constexpr bool composite = false;
};

} // namespace kernel_family

namespace family_selection {

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
 * Try one family throughout its correctness-supported domain, then retain the
 * automatic planner as the fallback. Profitability thresholds do not reject a
 * supported preferred family.
 */
template <typename FamilyT>
struct Prefer {
  static_assert(kernel_family::Family<FamilyT>,
                "unknown matmul kernel family");
  using Family = FamilyT;
  static constexpr Mode mode = Mode::prefer;
};

/**
 * Require one family throughout its correctness-supported domain. A family
 * proven unsupported from static metadata is a compile-time error; a dynamic
 * layout/shape rejection hits an always-on runtime check in every build.
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

/**
 * Result of proving a family predicate from compile-time metadata.
 *
 * `always` and `never` must emit no runtime test. `runtime` is reserved for
 * predicates whose truth varies within the admitted Meta extent or layout
 * contract.
 */
enum class Applicability {
  never,
  runtime,
  always,
};

inline constexpr Applicability operator&&(
    Applicability lhs, Applicability rhs) {
  if (lhs == Applicability::never || rhs == Applicability::never)
    return Applicability::never;
  if (lhs == Applicability::always && rhs == Applicability::always)
    return Applicability::always;
  return Applicability::runtime;
}

inline constexpr Applicability operator||(
    Applicability lhs, Applicability rhs) {
  if (lhs == Applicability::always || rhs == Applicability::always)
    return Applicability::always;
  if (lhs == Applicability::never && rhs == Applicability::never)
    return Applicability::never;
  return Applicability::runtime;
}

/** Compile-time contract carried from the operation plan into a backend leaf. */
template <kernel_family::ArchitectureFamily FamilyT,
          family_selection::Mode ModeV>
struct FamilyDispatch {
  using Family = FamilyT;
  static constexpr family_selection::Mode mode = ModeV;
  static constexpr bool automatic = mode == family_selection::Mode::automatic;
  static constexpr bool preferred = mode == family_selection::Mode::prefer;
  static constexpr bool required = mode == family_selection::Mode::require;
};

using AutomaticFamilyDispatch = FamilyDispatch<
    kernel_family::WholeProblem, family_selection::Mode::automatic>;

} // namespace details

} // namespace vecops::matmul

#endif // VECOPS_MATMUL_FAMILY_H
