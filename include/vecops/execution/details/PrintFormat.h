// SPDX-FileCopyrightText: 2026 HenryRenYz and vecops contributors
// SPDX-License-Identifier: MIT
/** @file PrintFormat.h
 * @brief Bounded numeric formatter used only by the public PrintRecord API.
 *
 * Standard floating formatters may allocate a temporary string for large
 * fixed-point values. This formatter uses to_chars into a fixed byte buffer
 * while remaining in the same format_to pipeline. Precision is bounded to 64;
 * locale, dynamic specifications and multibyte fill characters are rejected.
 */
#pragma once

#include <algorithm>
#include <charconv>
#include <cstddef>
#include <cmath>
#include <cstring>
#include <format>
#include <string_view>
#include <type_traits>
#include <utility>

namespace vecops::execution::details {
struct PrintFloat {
  double value;
};

template <class T>
auto print_argument(T&& value) {
  if constexpr (std::is_same_v<std::remove_cvref_t<T>, float> || std::is_same_v<std::remove_cvref_t<T>, double>)
    return PrintFloat{static_cast<double>(value)};
  else if constexpr (std::is_convertible_v<T, std::string_view>)
    return std::string_view{value};
  else
    return std::forward<T>(value);
}
} // namespace vecops::execution::details

template <>
struct std::formatter<vecops::execution::details::PrintFloat> {
  char fill = ' ', alignment = 0, sign = '-', presentation = 0;
  bool alternate = false, zero = false;
  int width = 0, precision = -1;

  static constexpr bool align(char c) {
    return c == '<' || c == '>' || c == '^';
  }
  static constexpr bool digit(char c) {
    return c >= '0' && c <= '9';
  }

  constexpr auto parse(std::format_parse_context& context) {
    auto position = context.begin();
    const auto end = context.end();
    if (position != end && *position != '}') {
      if (position + 1 != end && align(position[1])) {
        fill = *position++;
        alignment = *position++;
        if (static_cast<unsigned char>(fill) >= 128 || fill == '{')
          throw std::format_error("kernel print requires an ASCII fill character");
      } else if (align(*position))
        alignment = *position++;
    }
    if (position != end && (*position == '+' || *position == '-' || *position == ' '))
      sign = *position++;
    if (position != end && *position == '#') {
      alternate = true;
      ++position;
    }
    if (position != end && *position == '0') {
      zero = true;
      ++position;
    }
    while (position != end && digit(*position)) {
      width = width * 10 + (*position++ - '0');
      if (width > 1048576)
        throw std::format_error("kernel print width exceeds 1048576");
    }
    if (position != end && *position == '.') {
      ++position;
      precision = 0;
      if (position == end || !digit(*position))
        throw std::format_error("kernel print requires constant precision");
      while (position != end && digit(*position)) {
        precision = precision * 10 + (*position++ - '0');
        if (precision > 64)
          throw std::format_error("kernel print precision exceeds 64");
      }
    }
    if (position != end && *position != '}') {
      const auto c = *position++;
      if (c != 'a' && c != 'A' && c != 'e' && c != 'E' && c != 'f' && c != 'F' && c != 'g' && c != 'G')
        throw std::format_error("unsupported allocation-free kernel float format");
      presentation = c;
    }
    if (position != end && *position != '}')
      throw std::format_error("unsupported allocation-free kernel float format");
    return position;
  }

  template <class Context>
  auto format(vecops::execution::details::PrintFloat input, Context& context) const {
    // binary64 needs at most 309 integral digits plus 64 fractional digits.
    char bytes[512];
    const bool negative = std::signbit(input.value);
    const auto value = std::fabs(input.value);
    const char type = presentation >= 'A' && presentation <= 'Z' ? presentation + ('a' - 'A') : presentation;
    const auto mode = type == 'f'   ? std::chars_format::fixed
                      : type == 'e' ? std::chars_format::scientific
                      : type == 'a' ? std::chars_format::hex
                                    : std::chars_format::general;
    const auto digits = precision >= 0 ? precision : 6;
    std::to_chars_result result;
    if (precision >= 0 || type == 'e' || type == 'f' || type == 'g')
      result = std::to_chars(bytes, bytes + sizeof(bytes), value, mode, type == 'g' && !digits ? 1 : digits);
    else if (type == 'a')
      result = std::to_chars(bytes, bytes + sizeof(bytes), value, mode);
    else
      result = std::to_chars(bytes, bytes + sizeof(bytes), value);
    if (result.ec != std::errc{})
      throw std::format_error("bounded kernel float conversion failed");
    auto length = result.ptr - bytes;
    if (alternate && std::isfinite(value)) {
      auto exponent = length;
      bool point = false, significant = false;
      int significant_digits = 0;
      for (std::ptrdiff_t i = 0; i < length; ++i) {
        if (bytes[i] == (type == 'a' ? 'p' : 'e')) {
          exponent = i;
          break;
        }
        point |= bytes[i] == '.';
        if (digit(bytes[i])) {
          significant |= bytes[i] != '0';
          significant_digits += significant;
        }
      }
      if (!significant_digits)
        significant_digits = 1;
      const auto zeros = type == 'g' ? std::max(0, std::max(1, digits) - significant_digits) : 0;
      const auto extra = zeros + !point;
      std::memmove(bytes + exponent + extra, bytes + exponent, length - exponent);
      if (!point)
        bytes[exponent++] = '.';
      for (int i = 0; i < zeros; ++i)
        bytes[exponent++] = '0';
      length += extra;
    }
    if (presentation >= 'A' && presentation <= 'Z')
      for (std::ptrdiff_t i = 0; i < length; ++i)
        if (bytes[i] >= 'a' && bytes[i] <= 'z')
          bytes[i] -= 'a' - 'A';
    const char prefix = negative ? '-' : sign != '-' ? sign : 0;
    const auto padding = std::max<std::ptrdiff_t>(0, width - length - (prefix != 0));
    const bool zero_padding = zero && !alignment && std::isfinite(value);
    const auto left = alignment == '<' || zero_padding ? 0 : alignment == '^' ? padding / 2 : padding;
    auto output = context.out();
    for (std::ptrdiff_t i = 0; i < left; ++i)
      *output++ = fill;
    if (prefix)
      *output++ = prefix;
    if (zero_padding)
      for (std::ptrdiff_t i = 0; i < padding; ++i)
        *output++ = '0';
    for (std::ptrdiff_t i = 0; i < length; ++i)
      *output++ = bytes[i];
    if (!zero_padding)
      for (std::ptrdiff_t i = left; i < padding; ++i)
        *output++ = fill;
    return output;
  }
};
