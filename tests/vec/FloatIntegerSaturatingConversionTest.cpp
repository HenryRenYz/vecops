// SPDX-FileCopyrightText: 2026 HenryRenYz and vecops contributors
// SPDX-License-Identifier: MIT
// Standalone scalar-oracle regression for floating-to-integer conversion.
#include <array>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <limits>
#include "vecops/vec/ConversionMemory.h"
#include "vecops/util/ScalarConvert.h"

namespace vec = vecops::vec;

template <typename From, typename To, int Lanes>
bool verify() {
  using Tag = vec::FixedTag<From, Lanes>;
  using ToTag = vec::Rebind<To, Tag>;
  using IndexTag = vec::Rebind<int64_t, Tag>;
  constexpr auto hi = static_cast<From>(std::numeric_limits<To>::max());
  constexpr auto lo = static_cast<From>(std::numeric_limits<To>::min());
  const std::array<From, 16> cases = {
    -std::numeric_limits<From>::infinity(), std::numeric_limits<From>::infinity(),
    std::numeric_limits<From>::quiet_NaN(), lo, hi,
    std::nextafter(lo, -std::numeric_limits<From>::infinity()),
    std::nextafter(hi, std::numeric_limits<From>::infinity()),
    std::nextafter(lo, From{0}), std::nextafter(hi, From{0}),
    From{-1.9}, From{1.9}, From{-1}, From{0}, From{1},
    -std::numeric_limits<From>::max(), std::numeric_limits<From>::max()
  };
  std::array<From, Lanes> input{};
  std::array<int64_t, Lanes> indices{};
  std::array<To, Lanes> converted{}, stored{}, indexed{};
  for (int i = 0; i < Lanes; ++i) {
    input[i] = cases[i % cases.size()];
    indices[i] = Lanes - i - 1;
  }
  const auto value = vec::load(Tag{}, input.data());
  vec::store(ToTag{}, converted.data(), vec::convert(ToTag{}, Tag{}, value));
  vec::store_convert(Tag{}, stored.data(), value);
  const auto index = vec::load(IndexTag{}, indices.data());
  vec::store_convert(Tag{}, indexed.data(), value, vec::opt::indexed(index));
  for (int i = 0; i < Lanes; ++i) {
    const auto expected = vecops::convert<To>(input[i]);
    if (converted[i] != expected || stored[i] != expected || indexed[indices[i]] != expected) {
      std::cerr << "conversion mismatch: source bytes=" << sizeof(From)
                << " destination bytes=" << sizeof(To) << " signed=" << std::is_signed_v<To>
                << " lanes=" << Lanes << " lane=" << i << '\n';
      return false;
    }
  }
  return true;
}

template <typename From, typename To>
bool pair() { return verify<From, To, 16>() && verify<From, To, 64>(); }

template <typename From>
bool all() {
  return pair<From, int8_t>() && pair<From, uint8_t>() &&
         pair<From, int16_t>() && pair<From, uint16_t>() &&
         pair<From, int32_t>() && pair<From, uint32_t>() &&
         pair<From, int64_t>() && pair<From, uint64_t>();
}

int main() { return all<float>() && all<double>() ? 0 : 1; }
