//
// Copyright (c) vecops contributors.
//

#ifndef VECOPS_VEC_DETAILS_SME_ZA_H
#define VECOPS_VEC_DETAILS_SME_ZA_H

#include <cstdint>
#include <type_traits>

#include "vecops/CoreDefs.h"
#include "vecops/CoreTypes.h"
#include "vecops/vec/Vec.h"
#include "vecops/vec/details/sme/State.h"

namespace vecops::vec::details::sme {

template <typename...>
inline constexpr bool dependent_false_v = false;

template <typename T>
using WordTag = ScalableTag<T, 0>;

template <typename T>
VECOPS_ALWAYS_INLINE auto raw_word(Vec<WordTag<T>> value) noexcept {
  return ::vecops::vec::details::sve_basic_raw_word(value);
}

template <typename T, typename Raw>
VECOPS_ALWAYS_INLINE Vec<WordTag<T>> wrap_word(Raw value) noexcept {
  return ::vecops::vec::details::sve_basic_wrap_word<WordTag<T>>(value);
}

template <typename T>
VECOPS_ALWAYS_INLINE auto raw_mask(Mask<WordTag<T>> value) noexcept {
  return ::vecops::vec::get_word<0>(WordTag<T>{}, value);
}

template <typename T>
consteval char element_suffix() {
  if constexpr (sizeof(T) == 1) return 'b';
  else if constexpr (sizeof(T) == 2) return 'h';
  else if constexpr (sizeof(T) == 4) return 's';
  else {
    static_assert(sizeof(T) == 8, "ZA moves support 8/16/32/64-bit elements");
    return 'd';
  }
}

template <int Tile, typename T>
consteval void validate_tile() {
  static_assert(Tile >= 0);
  static_assert(Tile < static_cast<int>(sizeof(T)),
                "ZA tile number is invalid for the element width");
}

VECOPS_ALWAYS_INLINE void zero_za() noexcept {
  asm volatile("zero {za}" ::: "za");
}

#define VECOPS_SME_DEFINE_ZA_MOVE_WIDTH(Type, Suffix)                    \
  template <int Tile>                                                    \
  VECOPS_ALWAYS_INLINE void write_hor(                                   \
      std::uint32_t slice, Mask<WordTag<Type>> pg,                       \
      Vec<WordTag<Type>> value) noexcept {                               \
    validate_tile<Tile, Type>();                                         \
    const auto raw = raw_word<Type>(value);                              \
    const auto predicate = raw_mask<Type>(pg);                           \
    asm volatile(                                                        \
        "mov w12, %w[slice]\n\t"                                      \
        "mov za%c[tile]h." #Suffix "[w12, 0], %[pg]/m, %[value]."      \
            #Suffix                                                       \
        :                                                                 \
        : [slice] "r"(slice), [tile] "i"(Tile),                         \
          [pg] "Upl"(predicate), [value] "w"(raw)                       \
        : "x12", "za");                                                \
  }                                                                       \
  template <int Tile>                                                     \
  VECOPS_ALWAYS_INLINE void write_ver(                                    \
      std::uint32_t slice, Mask<WordTag<Type>> pg,                        \
      Vec<WordTag<Type>> value) noexcept {                                \
    validate_tile<Tile, Type>();                                          \
    const auto raw = raw_word<Type>(value);                               \
    const auto predicate = raw_mask<Type>(pg);                            \
    asm volatile(                                                         \
        "mov w12, %w[slice]\n\t"                                       \
        "mov za%c[tile]v." #Suffix "[w12, 0], %[pg]/m, %[value]."       \
            #Suffix                                                       \
        :                                                                 \
        : [slice] "r"(slice), [tile] "i"(Tile),                         \
          [pg] "Upl"(predicate), [value] "w"(raw)                       \
        : "x12", "za");                                                \
  }                                                                       \
  template <int Tile>                                                     \
  VECOPS_ALWAYS_INLINE Vec<WordTag<Type>> read_hor(                       \
      WordTag<Type>, std::uint32_t slice,                                \
      Mask<WordTag<Type>> pg) noexcept {                                  \
    validate_tile<Tile, Type>();                                          \
    auto result = raw_word<Type>(::vecops::vec::zeros(WordTag<Type>{}));  \
    const auto predicate = raw_mask<Type>(pg);                            \
    asm volatile(                                                         \
        "mov w12, %w[slice]\n\t"                                       \
        "mov %[result]." #Suffix ", %[pg]/m, za%c[tile]h." #Suffix     \
            "[w12, 0]"                                                   \
        : [result] "+w"(result)                                          \
        : [slice] "r"(slice), [tile] "i"(Tile),                         \
          [pg] "Upl"(predicate)                                          \
        : "x12", "za");                                                \
    return wrap_word<Type>(result);                                       \
  }                                                                       \
  template <int Tile>                                                     \
  VECOPS_ALWAYS_INLINE Vec<WordTag<Type>> read_ver(                       \
      WordTag<Type>, std::uint32_t slice,                                \
      Mask<WordTag<Type>> pg) noexcept {                                  \
    validate_tile<Tile, Type>();                                          \
    auto result = raw_word<Type>(::vecops::vec::zeros(WordTag<Type>{}));  \
    const auto predicate = raw_mask<Type>(pg);                            \
    asm volatile(                                                         \
        "mov w12, %w[slice]\n\t"                                       \
        "mov %[result]." #Suffix ", %[pg]/m, za%c[tile]v." #Suffix     \
            "[w12, 0]"                                                   \
        : [result] "+w"(result)                                          \
        : [slice] "r"(slice), [tile] "i"(Tile),                         \
          [pg] "Upl"(predicate)                                          \
        : "x12", "za");                                                \
    return wrap_word<Type>(result);                                       \
  }

VECOPS_SME_DEFINE_ZA_MOVE_WIDTH(std::uint8_t, b)
VECOPS_SME_DEFINE_ZA_MOVE_WIDTH(std::uint16_t, h)
VECOPS_SME_DEFINE_ZA_MOVE_WIDTH(std::uint32_t, s)
VECOPS_SME_DEFINE_ZA_MOVE_WIDTH(std::uint64_t, d)
VECOPS_SME_DEFINE_ZA_MOVE_WIDTH(float32_t, s)
VECOPS_SME_DEFINE_ZA_MOVE_WIDTH(float64_t, d)
VECOPS_SME_DEFINE_ZA_MOVE_WIDTH(std::int32_t, s)

#undef VECOPS_SME_DEFINE_ZA_MOVE_WIDTH

#define VECOPS_SME_DEFINE_ZA_MEMORY_WIDTH(                               \
    Type, LoadMnemonic, MemorySuffix, ZASuffix)                          \
  template <int Tile>                                                    \
  VECOPS_ALWAYS_INLINE void load_hor(                                    \
      std::uint32_t row, Mask<WordTag<Type>> pg,                         \
      const Type* pointer) noexcept {                                    \
    validate_tile<Tile, Type>();                                         \
    const auto predicate = raw_mask<Type>(pg);                           \
    asm volatile(                                                        \
        "mov w12, %w[row]\n\t"                                        \
        #LoadMnemonic " {za%c[tile]h." #ZASuffix                       \
            "[w12, 0]}, %[pg]/z, "                                      \
            "[%[pointer]]"                                              \
        :                                                                \
        : [row] "r"(row), [tile] "i"(Tile), [pg] "Upl"(predicate),    \
          [pointer] "r"(pointer)                                        \
        : "x12", "za", "memory");                                    \
  }                                                                      \
  template <int Tile>                                                    \
  VECOPS_ALWAYS_INLINE void store_hor(                                   \
      std::uint32_t row, Mask<WordTag<Type>> pg, Type* pointer) noexcept {\
    validate_tile<Tile, Type>();                                         \
    const auto predicate = raw_mask<Type>(pg);                           \
    asm volatile(                                                        \
        "mov w12, %w[row]\n\t"                                        \
        "st1" #MemorySuffix " {za%c[tile]h." #ZASuffix               \
            "[w12, 0]}, %[pg], [%[pointer]]"                            \
        :                                                                \
        : [row] "r"(row), [tile] "i"(Tile), [pg] "Upl"(predicate),    \
          [pointer] "r"(pointer)                                        \
        : "x12", "za", "memory");                                    \
  }

// These are the direct ZA-memory forms used by the current matmul backend.
VECOPS_SME_DEFINE_ZA_MEMORY_WIDTH(std::uint32_t, ld1w, w, s)
VECOPS_SME_DEFINE_ZA_MEMORY_WIDTH(std::uint64_t, ld1d, d, d)

#undef VECOPS_SME_DEFINE_ZA_MEMORY_WIDTH

template <int Tile, typename TAcc, typename TA, typename TB>
VECOPS_ALWAYS_INLINE void mopa(
    Mask<WordTag<TA>> pga, Mask<WordTag<TB>> pgb,
    Vec<WordTag<TA>> a, Vec<WordTag<TB>> b) noexcept {
  validate_tile<Tile, TAcc>();
  const auto pa = raw_mask<TA>(pga);
  const auto pb = raw_mask<TB>(pgb);
  const auto va = raw_word<TA>(a);
  const auto vb = raw_word<TB>(b);
#define VECOPS_SME_MOPA_ASM(Mnemonic, AccSuffix, InputSuffix)            \
  asm volatile(                                                         \
      #Mnemonic " za%c[tile]." #AccSuffix ", %[pa]/m, %[pb]/m, "      \
          "%[a]." #InputSuffix ", %[b]." #InputSuffix                 \
      :                                                                 \
      : [tile] "i"(Tile), [pa] "Upl"(pa), [pb] "Upl"(pb),             \
        [a] "w"(va), [b] "w"(vb)                                      \
      : "za")
  if constexpr (std::same_as<TAcc, float32_t> &&
                std::same_as<TA, float32_t> && std::same_as<TB, float32_t>) {
    VECOPS_SME_MOPA_ASM(fmopa, s, s);
  } else if constexpr (std::same_as<TAcc, float32_t> &&
                       std::same_as<TA, float16_t> &&
                       std::same_as<TB, float16_t>) {
    VECOPS_SME_MOPA_ASM(fmopa, s, h);
  } else if constexpr (std::same_as<TAcc, float32_t> &&
                       std::same_as<TA, bfloat16_t> &&
                       std::same_as<TB, bfloat16_t>) {
    VECOPS_SME_MOPA_ASM(bfmopa, s, h);
  } else if constexpr (std::same_as<TAcc, std::int32_t> &&
                       std::same_as<TA, std::int8_t> &&
                       std::same_as<TB, std::int8_t>) {
    VECOPS_SME_MOPA_ASM(smopa, s, b);
  } else if constexpr ((std::same_as<TAcc, std::uint32_t> ||
                        std::same_as<TAcc, std::int32_t>) &&
                       std::same_as<TA, std::uint8_t> &&
                       std::same_as<TB, std::uint8_t>) {
    VECOPS_SME_MOPA_ASM(umopa, s, b);
  } else if constexpr (std::same_as<TAcc, std::int32_t> &&
                       std::same_as<TA, std::int8_t> &&
                       std::same_as<TB, std::uint8_t>) {
    VECOPS_SME_MOPA_ASM(sumopa, s, b);
  } else if constexpr (std::same_as<TAcc, std::int32_t> &&
                       std::same_as<TA, std::uint8_t> &&
                       std::same_as<TB, std::int8_t>) {
    VECOPS_SME_MOPA_ASM(usmopa, s, b);
  } else if constexpr (std::same_as<TAcc, float64_t> &&
                       std::same_as<TA, float64_t> &&
                       std::same_as<TB, float64_t>) {
    VECOPS_SME_MOPA_ASM(fmopa, d, d);
  } else {
    static_assert(dependent_false_v<TAcc, TA, TB>,
                  "unsupported SME outer-product dtype combination");
  }
#undef VECOPS_SME_MOPA_ASM
}

} // namespace vecops::vec::details::sme

#endif // VECOPS_VEC_DETAILS_SME_ZA_H
