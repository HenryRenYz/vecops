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

template <typename T>
concept ArchitectureFamily =
    std::same_as<T, WholeProblem> || std::same_as<T, General> ||
    std::same_as<T, SmallVector> ||
    std::same_as<T, RuntimeQuantInt8> ||
    std::same_as<T, PackedMMLA> || std::same_as<T, PackedTail>;

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

/** Prefer one family and retain the automatic planner as its fallback. */
template <typename FamilyT>
struct Prefer {
  static_assert(kernel_family::Family<FamilyT>,
                "unknown matmul kernel family");
  using Family = FamilyT;
  static constexpr Mode mode = Mode::prefer;
};

/** Require one family; an inapplicable family is a configuration error. */
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
  static constexpr bool required = mode == family_selection::Mode::require;
};

using AutomaticFamilyDispatch = FamilyDispatch<
    kernel_family::WholeProblem, family_selection::Mode::automatic>;

} // namespace details

} // namespace vecops::matmul

#endif // VECOPS_MATMUL_FAMILY_H
