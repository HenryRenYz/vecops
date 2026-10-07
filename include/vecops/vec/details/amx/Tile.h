// SPDX-FileCopyrightText: 2026 HenryRenYz and vecops contributors
// SPDX-License-Identifier: MIT
//
//

#ifndef VECOPS_VEC_DETAILS_AMX_TILE_H
#define VECOPS_VEC_DETAILS_AMX_TILE_H

#include <concepts>
#include <cstdint>
#include <type_traits>

#include <immintrin.h>

#include "vecops/CoreDefs.h"
#include "vecops/CoreTypes.h"

#if !defined(HAS_AMX_TILE)
#error "AMX tile primitives require an AMX-TILE target"
#endif

namespace vecops::vec::details::amx {

template <typename...>
inline constexpr bool dependent_false_v = false;

template <int Tile>
consteval void validate_tile() {
  static_assert(Tile >= 0 && Tile < 8, "AMX tile number must be in [0, 8)");
}

/** Load one architectural 64-byte TILECFG image. */
VECOPS_ALWAYS_INLINE void load_configuration(const void* configuration) {
  _tile_loadconfig(configuration);
}

/** Release the current thread's AMX tile state. */
VECOPS_ALWAYS_INLINE void release() {
  _tile_release();
}

template <int Tile>
VECOPS_ALWAYS_INLINE void zero() {
  validate_tile<Tile>();
  if constexpr (Tile == 0) _tile_zero(0);
  else if constexpr (Tile == 1) _tile_zero(1);
  else if constexpr (Tile == 2) _tile_zero(2);
  else if constexpr (Tile == 3) _tile_zero(3);
  else if constexpr (Tile == 4) _tile_zero(4);
  else if constexpr (Tile == 5) _tile_zero(5);
  else if constexpr (Tile == 6) _tile_zero(6);
  else _tile_zero(7);
}

template <int Tile>
VECOPS_ALWAYS_INLINE void load(const void* pointer, std::intptr_t stride) {
  validate_tile<Tile>();
  if constexpr (Tile == 0) _tile_loadd(0, pointer, stride);
  else if constexpr (Tile == 1) _tile_loadd(1, pointer, stride);
  else if constexpr (Tile == 2) _tile_loadd(2, pointer, stride);
  else if constexpr (Tile == 3) _tile_loadd(3, pointer, stride);
  else if constexpr (Tile == 4) _tile_loadd(4, pointer, stride);
  else if constexpr (Tile == 5) _tile_loadd(5, pointer, stride);
  else if constexpr (Tile == 6) _tile_loadd(6, pointer, stride);
  else _tile_loadd(7, pointer, stride);
}

template <int Tile>
VECOPS_ALWAYS_INLINE void stream_load(
    const void* pointer, std::intptr_t stride) {
  validate_tile<Tile>();
  if constexpr (Tile == 0) _tile_stream_loadd(0, pointer, stride);
  else if constexpr (Tile == 1) _tile_stream_loadd(1, pointer, stride);
  else if constexpr (Tile == 2) _tile_stream_loadd(2, pointer, stride);
  else if constexpr (Tile == 3) _tile_stream_loadd(3, pointer, stride);
  else if constexpr (Tile == 4) _tile_stream_loadd(4, pointer, stride);
  else if constexpr (Tile == 5) _tile_stream_loadd(5, pointer, stride);
  else if constexpr (Tile == 6) _tile_stream_loadd(6, pointer, stride);
  else _tile_stream_loadd(7, pointer, stride);
}

template <int Tile>
VECOPS_ALWAYS_INLINE void store(void* pointer, std::intptr_t stride) {
  validate_tile<Tile>();
  if constexpr (Tile == 0) _tile_stored(0, pointer, stride);
  else if constexpr (Tile == 1) _tile_stored(1, pointer, stride);
  else if constexpr (Tile == 2) _tile_stored(2, pointer, stride);
  else if constexpr (Tile == 3) _tile_stored(3, pointer, stride);
  else if constexpr (Tile == 4) _tile_stored(4, pointer, stride);
  else if constexpr (Tile == 5) _tile_stored(5, pointer, stride);
  else if constexpr (Tile == 6) _tile_stored(6, pointer, stride);
  else _tile_stored(7, pointer, stride);
}

#if defined(HAS_AMX_BF16)
#define VECOPS_AMX_BF16_LITERAL(Dst, SrcA, SrcB, ...) \
  _tile_dpbf16ps(Dst, SrcA, SrcB)
#else
#define VECOPS_AMX_BF16_LITERAL(Dst, SrcA, SrcB, ...)                    \
  static_assert(                                                         \
      ::vecops::vec::details::amx::dependent_false_v<__VA_ARGS__>,       \
      "AMX BF16 is not enabled for this target")
#endif

#if defined(HAS_AMX_FP16)
#define VECOPS_AMX_FP16_LITERAL(Dst, SrcA, SrcB, ...) \
  _tile_dpfp16ps(Dst, SrcA, SrcB)
#else
#define VECOPS_AMX_FP16_LITERAL(Dst, SrcA, SrcB, ...)                    \
  static_assert(                                                         \
      ::vecops::vec::details::amx::dependent_false_v<__VA_ARGS__>,       \
      "AMX FP16 is not enabled for this target")
#endif

#if defined(HAS_AMX_INT8)
#define VECOPS_AMX_SS_LITERAL(Dst, SrcA, SrcB, ...) \
  _tile_dpbssd(Dst, SrcA, SrcB)
#define VECOPS_AMX_SU_LITERAL(Dst, SrcA, SrcB, ...) \
  _tile_dpbsud(Dst, SrcA, SrcB)
#define VECOPS_AMX_US_LITERAL(Dst, SrcA, SrcB, ...) \
  _tile_dpbusd(Dst, SrcA, SrcB)
#define VECOPS_AMX_UU_LITERAL(Dst, SrcA, SrcB, ...) \
  _tile_dpbuud(Dst, SrcA, SrcB)
#else
#define VECOPS_AMX_INT8_LITERAL_UNAVAILABLE(Dst, SrcA, SrcB, ...)        \
  static_assert(                                                         \
      ::vecops::vec::details::amx::dependent_false_v<__VA_ARGS__>,       \
      "AMX INT8 is not enabled for this target")
#define VECOPS_AMX_SS_LITERAL VECOPS_AMX_INT8_LITERAL_UNAVAILABLE
#define VECOPS_AMX_SU_LITERAL VECOPS_AMX_INT8_LITERAL_UNAVAILABLE
#define VECOPS_AMX_US_LITERAL VECOPS_AMX_INT8_LITERAL_UNAVAILABLE
#define VECOPS_AMX_UU_LITERAL VECOPS_AMX_INT8_LITERAL_UNAVAILABLE
#endif

// GCC implements the AMX tile intrinsics with inline-assembly macros that
// stringify the tile arguments. Generate the literal overloads used by the
// AMX kernel catalog here, then expose only an ordinary force-inlined function.
#define VECOPS_AMX_DEFINE_DOT_LITERAL(Dst, SrcA, SrcB)                   \
  template <typename TAcc, typename TA, typename TB>                    \
  VECOPS_ALWAYS_INLINE void dot_literal(                                \
      std::integral_constant<int, Dst>, std::integral_constant<int, SrcA>,\
      std::integral_constant<int, SrcB>) {                              \
    if constexpr (                                                       \
        std::same_as<TAcc, ::vecops::float32_t> &&                       \
        std::same_as<TA, ::vecops::bfloat16_t> &&                        \
        std::same_as<TB, ::vecops::bfloat16_t>) {                        \
      VECOPS_AMX_BF16_LITERAL(Dst, SrcA, SrcB, TAcc, TA, TB);           \
    } else if constexpr (                                                \
        std::same_as<TAcc, ::vecops::float32_t> &&                       \
        std::same_as<TA, ::vecops::float16_t> &&                         \
        std::same_as<TB, ::vecops::float16_t>) {                         \
      VECOPS_AMX_FP16_LITERAL(Dst, SrcA, SrcB, TAcc, TA, TB);           \
    } else if constexpr (                                                \
        std::same_as<TAcc, std::int32_t> &&                              \
        std::same_as<TA, std::int8_t> && std::same_as<TB, std::int8_t>) {\
      VECOPS_AMX_SS_LITERAL(Dst, SrcA, SrcB, TAcc, TA, TB);             \
    } else if constexpr (                                                \
        std::same_as<TAcc, std::int32_t> &&                              \
        std::same_as<TA, std::int8_t> && std::same_as<TB, std::uint8_t>) {\
      VECOPS_AMX_SU_LITERAL(Dst, SrcA, SrcB, TAcc, TA, TB);             \
    } else if constexpr (                                                \
        std::same_as<TAcc, std::int32_t> &&                              \
        std::same_as<TA, std::uint8_t> && std::same_as<TB, std::int8_t>) {\
      VECOPS_AMX_US_LITERAL(Dst, SrcA, SrcB, TAcc, TA, TB);             \
    } else if constexpr (                                                \
        std::same_as<TAcc, std::int32_t> &&                              \
        std::same_as<TA, std::uint8_t> &&                                \
        std::same_as<TB, std::uint8_t>) {                                \
      VECOPS_AMX_UU_LITERAL(Dst, SrcA, SrcB, TAcc, TA, TB);             \
    } else {                                                             \
      static_assert(                                                     \
          ::vecops::vec::details::amx::dependent_false_v<TAcc, TA, TB>,  \
          "unsupported AMX dot-product dtype combination");            \
    }                                                                    \
  }

VECOPS_AMX_DEFINE_DOT_LITERAL(0, 1, 2)
VECOPS_AMX_DEFINE_DOT_LITERAL(0, 2, 3)
VECOPS_AMX_DEFINE_DOT_LITERAL(1, 2, 4)
VECOPS_AMX_DEFINE_DOT_LITERAL(0, 3, 4)
VECOPS_AMX_DEFINE_DOT_LITERAL(1, 3, 5)
VECOPS_AMX_DEFINE_DOT_LITERAL(2, 3, 6)
VECOPS_AMX_DEFINE_DOT_LITERAL(0, 2, 4)
VECOPS_AMX_DEFINE_DOT_LITERAL(1, 3, 4)
VECOPS_AMX_DEFINE_DOT_LITERAL(0, 3, 6)
VECOPS_AMX_DEFINE_DOT_LITERAL(1, 4, 6)
VECOPS_AMX_DEFINE_DOT_LITERAL(2, 5, 6)
VECOPS_AMX_DEFINE_DOT_LITERAL(0, 4, 6)
VECOPS_AMX_DEFINE_DOT_LITERAL(1, 4, 7)
VECOPS_AMX_DEFINE_DOT_LITERAL(3, 5, 7)

#undef VECOPS_AMX_DEFINE_DOT_LITERAL
#undef VECOPS_AMX_BF16_LITERAL
#undef VECOPS_AMX_FP16_LITERAL
#undef VECOPS_AMX_SS_LITERAL
#undef VECOPS_AMX_SU_LITERAL
#undef VECOPS_AMX_US_LITERAL
#undef VECOPS_AMX_UU_LITERAL
#if defined(VECOPS_AMX_INT8_LITERAL_UNAVAILABLE)
#undef VECOPS_AMX_INT8_LITERAL_UNAVAILABLE
#endif

template <typename TAcc, typename TA, typename TB,
          int Dst, int SrcA, int SrcB>
VECOPS_ALWAYS_INLINE void dot() {
  validate_tile<Dst>();
  validate_tile<SrcA>();
  validate_tile<SrcB>();
  dot_literal<TAcc, TA, TB>(
      std::integral_constant<int, Dst>{},
      std::integral_constant<int, SrcA>{},
      std::integral_constant<int, SrcB>{});
}

} // namespace vecops::vec::details::amx

#endif // VECOPS_VEC_DETAILS_AMX_TILE_H
