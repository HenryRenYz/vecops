#ifndef VECOPS_NVEC_DETAILS_STORAGE_H
#define VECOPS_NVEC_DETAILS_STORAGE_H

#include <array>
#include <bitset>
#include <concepts>
#include <cstddef>
#include <type_traits>
#include <utility>

#include "vecops/CoreTypes.h"

namespace vecops::nvec::details {

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
inline constexpr bool is_word_array =
    IsWordArray<std::remove_cvref_t<T>>::value;

template <typename Word, nint_t Count>
using SingleOrArray = std::conditional_t<
    Count == 1, Word, WordArray<Word, Count>>;

template <nint_t Index, typename Word>
constexpr Word get_word(const Word& value) {
  static_assert(Index == 0);
  return value;
}

template <nint_t Index, typename Word, nint_t Count>
constexpr Word get_word(const WordArray<Word, Count>& value) {
  static_assert(Index >= 0 && Index < Count);
  return value.words[static_cast<std::size_t>(Index)];
}

template <nint_t Index, typename Word>
constexpr Word set_word(Word, Word word) {
  static_assert(Index == 0);
  return word;
}

template <nint_t Index, typename Word, nint_t Count>
constexpr WordArray<Word, Count> set_word(
    WordArray<Word, Count> value, Word word) {
  static_assert(Index >= 0 && Index < Count);
  value.words[static_cast<std::size_t>(Index)] = word;
  return value;
}

} // namespace vecops::nvec::details

#endif // VECOPS_NVEC_DETAILS_STORAGE_H
