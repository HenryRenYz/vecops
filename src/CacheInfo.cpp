// SPDX-FileCopyrightText: 2026 HenryRenYz and vecops contributors
// SPDX-License-Identifier: MIT
//
//

#include "vecops/platform/CacheInfo.h"

#include <algorithm>
#include <cstdio>
#include <cstdlib>

#if defined(__GNUC__) && (defined(__x86_64__) || defined(__i386__))
#include <cpuid.h>
#endif

namespace vecops::platform {
namespace {

CacheInfo fallback_cache_info() { return {}; }

CacheInfo complete_detected_cache_info(
    CacheInfo result, bool found_l1, bool found_l2, bool found_l3) {
  const CacheInfo fallback = fallback_cache_info();
  if (result.line_bytes <= 0) result.line_bytes = fallback.line_bytes;
  if (!found_l1) result.l1d_bytes = fallback.l1d_bytes;
  if (!found_l2) result.l2_bytes = result.l1d_bytes;
  if (!found_l3) result.l3_bytes = result.l2_bytes;
  result.l2_bytes = std::max(result.l2_bytes, result.l1d_bytes);
  result.l3_bytes = std::max(result.l3_bytes, result.l2_bytes);
  result.fallback = !(found_l1 && found_l2);
  return result;
}

#if defined(__GNUC__) && (defined(__x86_64__) || defined(__i386__))
CacheInfo detect_x86_cache_info() {
  CacheInfo result{0, 0, 0, 0, true};
  unsigned int max_leaf = __get_cpuid_max(0, nullptr);
  if (max_leaf < 4) return fallback_cache_info();

  bool found_l1 = false;
  bool found_l2 = false;
  bool found_l3 = false;
  for (unsigned int index = 0;; ++index) {
    unsigned int eax = 0;
    unsigned int ebx = 0;
    unsigned int ecx = 0;
    unsigned int edx = 0;
    __cpuid_count(4, index, eax, ebx, ecx, edx);
    const unsigned int type = eax & 0x1fU;
    if (type == 0) break;
    const unsigned int level = (eax >> 5U) & 0x7U;
    const nint_t line = static_cast<nint_t>((ebx & 0xfffU) + 1U);
    const nint_t partitions =
        static_cast<nint_t>(((ebx >> 12U) & 0x3ffU) + 1U);
    const nint_t ways =
        static_cast<nint_t>(((ebx >> 22U) & 0x3ffU) + 1U);
    const nint_t sets = static_cast<nint_t>(ecx) + 1;
    const nint_t bytes = line * partitions * ways * sets;
    result.line_bytes = std::max(result.line_bytes, line);
    if (level == 1 && (type == 1 || type == 3)) {
      result.l1d_bytes = std::max(result.l1d_bytes, bytes);
      found_l1 = true;
    } else if (level == 2) {
      result.l2_bytes = std::max(result.l2_bytes, bytes);
      found_l2 = true;
    } else if (level == 3) {
      result.l3_bytes = std::max(result.l3_bytes, bytes);
      found_l3 = true;
    }
  }
  return complete_detected_cache_info(
      result, found_l1, found_l2, found_l3);
}
#endif

#if defined(__linux__)
nint_t parse_cache_size(const char* text) {
  if (text == nullptr || *text == '\0') return 0;
  char* suffix = nullptr;
  const long long value = std::strtoll(text, &suffix, 10);
  if (suffix == text || value <= 0) return 0;
  while (*suffix == ' ' || *suffix == '\t') ++suffix;
  nint_t multiplier = 1;
  if (*suffix == 'K' || *suffix == 'k') {
    multiplier = 1024;
  } else if (*suffix == 'M' || *suffix == 'm') {
    multiplier = 1024 * 1024;
  }
  return static_cast<nint_t>(value) * multiplier;
}

template <std::size_t N>
bool read_line(const char* path, char (&value)[N]) {
  FILE* stream = std::fopen(path, "r");
  if (stream == nullptr) return false;
  const bool read = std::fgets(value, static_cast<int>(N), stream) != nullptr;
  std::fclose(stream);
  return read;
}

CacheInfo detect_sysfs_cache_info() {
  CacheInfo result{0, 0, 0, 0, true};
  bool found_l1 = false;
  bool found_l2 = false;
  bool found_l3 = false;
  constexpr const char* Root = "/sys/devices/system/cpu/cpu0/cache/index";
  for (int index = 0; index < 16; ++index) {
    char path[160];
    char level_text[16];
    char type[32];
    char size_text[32];
    std::snprintf(path, sizeof(path), "%s%d/level", Root, index);
    if (!read_line(path, level_text)) continue;
    std::snprintf(path, sizeof(path), "%s%d/type", Root, index);
    if (!read_line(path, type)) continue;
    std::snprintf(path, sizeof(path), "%s%d/size", Root, index);
    if (!read_line(path, size_text)) {
      continue;
    }
    const int level = level_text[0] - '0';
    const nint_t bytes = parse_cache_size(size_text);
    if (bytes <= 0) continue;
    if (level == 1 &&
        (type[0] == 'D' || type[0] == 'U')) {
      result.l1d_bytes = std::max(result.l1d_bytes, bytes);
      found_l1 = true;
    } else if (level == 2) {
      result.l2_bytes = std::max(result.l2_bytes, bytes);
      found_l2 = true;
    } else if (level == 3) {
      result.l3_bytes = std::max(result.l3_bytes, bytes);
      found_l3 = true;
    }
    char line_text[32];
    std::snprintf(
        path, sizeof(path), "%s%d/coherency_line_size", Root, index);
    if (read_line(path, line_text)) {
      result.line_bytes = std::max(
          result.line_bytes, parse_cache_size(line_text));
    }
  }
  return complete_detected_cache_info(
      result, found_l1, found_l2, found_l3);
}
#endif

} // namespace

CacheInfo detect_cache_info() {
#if defined(__GNUC__) && (defined(__x86_64__) || defined(__i386__))
  auto result = detect_x86_cache_info();
  if (!result.fallback) return result;
#endif
#if defined(__linux__)
  return detect_sysfs_cache_info();
#else
  return fallback_cache_info();
#endif
}

const CacheInfo& cache_info() {
  static const CacheInfo info = detect_cache_info();
  return info;
}

} // namespace vecops::platform
