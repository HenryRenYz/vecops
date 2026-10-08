// SPDX-FileCopyrightText: 2026 HenryRenYz and vecops contributors
// SPDX-License-Identifier: MIT
#ifndef VECOPS_VEC_DETAILS_STORAGE_H
#define VECOPS_VEC_DETAILS_STORAGE_H

/**
 * @file Storage.h
 * @brief Low-level storage types: WordArray (fixed multi-word container),
 * ScalarVector (aligned array-based vector word), ScalarMask (bitset-based
 * mask word), and the get_word/set_word accessors that uniformly handle
 * both single-word and multi-word values.
 */

#include <array>
#include <bitset>
#include <cassert>
#include <concepts>
#include <cstddef>
#include <type_traits>
#include <utility>

#include "vecops/CoreTypes.h"

namespace vecops::vec::details {

/** Distinct storage for a fixed set of backend words. */
template <typename Word, nint_t Count>
struct WordArray {
  static_assert(Count > 1, "WordArray is reserved for multi-word values");

  std::array<Word, static_cast<std::size_t>(Count)> words;

  constexpr Word& operator[](nint_t index) {
    return words[static_cast<std::size_t>(index)];
  }

  constexpr const Word& operator[](nint_t index) const {
    return words[static_cast<std::size_t>(index)];
  }
};

template <typename T, nint_t N, nint_t Alignment>
struct alignas(Alignment) ScalarVector {
  std::array<T, static_cast<std::size_t>(N)> lanes;

  constexpr T& operator[](nint_t index) {
    return lanes[static_cast<std::size_t>(index)];
  }

  constexpr const T& operator[](nint_t index) const {
    return lanes[static_cast<std::size_t>(index)];
  }
};

template <nint_t ElementBytes, nint_t N>
struct ScalarMask {
  static constexpr nint_t element_bytes = ElementBytes;
  static constexpr nint_t lanes = N;
  std::bitset<static_cast<std::size_t>(N)> bits;
};

template <typename T>
struct IsWordArray : std::false_type {};

template <typename Word, nint_t Count>
struct IsWordArray<WordArray<Word, Count>> : std::true_type {};

template <typename T>
inline constexpr bool is_word_array_v =
    IsWordArray<std::remove_cvref_t<T>>::value;

template <typename Word, nint_t Count>
using SingleOrArray = std::conditional_t<
    Count == 1, Word, WordArray<Word, Count>>;

template <nint_t Index, typename Word>
VECOPS_ALWAYS_INLINE constexpr Word get_word(const Word& value) {
  static_assert(Index == 0);
  return value;
}

template <nint_t Index, typename Word, nint_t Count>
VECOPS_ALWAYS_INLINE constexpr Word get_word(
    const WordArray<Word, Count>& value) {
  static_assert(Index >= 0 && Index < Count);
  return value.words[static_cast<std::size_t>(Index)];
}

/** Returns a physical word selected by a runtime ordinal from sized storage. */
template <typename Word>
VECOPS_ALWAYS_INLINE constexpr Word get_word(
    const Word& value, nint_t ordinal) {
  assert(ordinal == 0);
  return value;
}

template <typename Word, nint_t Count>
VECOPS_ALWAYS_INLINE constexpr Word get_word(
    const WordArray<Word, Count>& value, nint_t ordinal) {
  assert(ordinal >= 0 && ordinal < Count);
  return value.words[static_cast<std::size_t>(ordinal)];
}

template <nint_t Index, typename Word>
VECOPS_ALWAYS_INLINE constexpr Word set_word(Word, Word word) {
  static_assert(Index == 0);
  return word;
}

template <nint_t Index, typename Word, nint_t Count>
VECOPS_ALWAYS_INLINE constexpr WordArray<Word, Count> set_word(
    WordArray<Word, Count> value, Word word) {
  static_assert(Index >= 0 && Index < Count);
  value.words[static_cast<std::size_t>(Index)] = word;
  return value;
}

/** Replaces a physical word selected by a runtime ordinal in sized storage. */
template <typename Word>
VECOPS_ALWAYS_INLINE constexpr Word set_word(
    Word, nint_t ordinal, Word word) {
  assert(ordinal == 0);
  return word;
}

template <typename Word, nint_t Count>
VECOPS_ALWAYS_INLINE constexpr WordArray<Word, Count> set_word(
    WordArray<Word, Count> value, nint_t ordinal, Word word) {
  assert(ordinal >= 0 && ordinal < Count);
  value.words[static_cast<std::size_t>(ordinal)] = word;
  return value;
}

/** Customization point for sizeless backend word-group construction. */
template <typename... Words>
void make_word_group(Words...) = delete;

} // namespace vecops::vec::details

#endif // VECOPS_VEC_DETAILS_STORAGE_H
