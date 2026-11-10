// SPDX-FileCopyrightText: 2026 HenryRenYz and vecops contributors
// SPDX-License-Identifier: MIT
#pragma once
#include <algorithm>
#include <charconv>
#include <cstdint>
#include <limits>
#include <map>
#include <optional>
#include <set>
#include <stdexcept>
#include <string>
#include <string_view>
#include <system_error>
#include <utility>
#include <vector>

namespace vecops::memory::model::details {
namespace linux_free_capacity_detail {
using Pages = std::uint64_t;
inline bool whitespace(char c) {
  return c == ' ' || c == '\t' || c == '\r' || c == '\n' || c == '\f' || c == '\v';
}
inline std::string_view trim(std::string_view s) {
  while (!s.empty() && whitespace(s.front())) s.remove_prefix(1);
  while (!s.empty() && whitespace(s.back())) s.remove_suffix(1);
  return s;
}
inline std::vector<std::string_view> tokens(std::string_view s) {
  std::vector<std::string_view> result;
  while (!(s = trim(s)).empty()) {
    const auto end = s.find_first_of(" \t\r\n\f\v");
    result.push_back(s.substr(0, end));
    if (end == std::string_view::npos) break;
    s.remove_prefix(end);
  }
  return result;
}
[[noreturn]] inline void malformed(const char* message) {
  throw std::invalid_argument(std::string("invalid Linux zoneinfo: ") + message);
}
inline Pages number(std::string_view s) {
  if (s.empty()) malformed("expected unsigned integer");
  Pages value{};
  const auto [end, error] = std::from_chars(s.data(), s.data() + s.size(), value);
  if (error == std::errc::result_out_of_range) throw std::overflow_error("Linux zoneinfo integer overflow");
  if (error != std::errc{} || end != s.data() + s.size()) malformed("expected unsigned integer");
  return value;
}
inline unsigned index(std::string_view s) {
  const auto value = number(s);
  if (value > std::numeric_limits<unsigned>::max()) throw std::overflow_error("Linux zoneinfo index overflow");
  return static_cast<unsigned>(value);
}
inline Pages add(Pages a, Pages b) {
  if (b > std::numeric_limits<Pages>::max() - a) throw std::overflow_error("Linux zoneinfo page sum overflow");
  return a + b;
}
inline void field(std::optional<Pages>& target, Pages value) {
  if (target) malformed("duplicate zone field");
  target = value;
}
inline Pages protection(std::string_view s) {
  s = trim(s);
  if (s.size() < 3 || s.front() != '(' || s.back() != ')') malformed("invalid protection tuple");
  s.remove_prefix(1); s.remove_suffix(1);
  Pages maximum{};
  for (;;) {
    const auto comma = s.find(',');
    maximum = std::max(maximum, number(trim(s.substr(0, comma))));
    if (comma == std::string_view::npos) break;
    s.remove_prefix(comma + 1);
  }
  return maximum;
}
struct Zone {
  unsigned node{};
  std::optional<Pages> free, high, protection, highatomic;
  Pages drift{};
  bool pagesets = false;
  std::optional<unsigned> cpu;
  bool threshold = false;
  std::set<unsigned> cpus;
  std::size_t missing_thresholds{}, thresholds{};
  void finish_cpu() {
    if (cpu && !threshold) ++missing_thresholds;
    cpu.reset(); threshold = false;
  }
  Pages available() const {
    if (!free || !high || !protection) malformed("missing pages free, high or protection");
    // Empty zones legitimately have no pagesets. A positive free count without
    // any CPU statistics cannot establish the required counter-drift bound.
    if ((*free || pagesets) && cpus.empty()) malformed("populated zone lacks per-CPU vm statistics");
    if (missing_thresholds && (cpus.size() != 1 || *cpus.begin() != 0 || missing_thresholds != 1 || thresholds))
      malformed("missing per-CPU vm stats threshold");
    const auto reserve = add(add(*high, *protection), add(highatomic.value_or(0), drift));
    return *free > reserve ? *free - reserve : 0;
  }
};
} // namespace linux_free_capacity_detail

/** Conservative, order-0 free-page budget per node from a /proc/zoneinfo snapshot.
 * Each zone contributes max(0, pages_free - high - max(protection) - optional
 * highatomic reservation - sum(per-CPU vm_stats_threshold)). The printed high
 * watermark already includes boost: boost is deliberately not subtracted again.
 * CPU pageset high: is a different field and is never a zone watermark.
 *
 * Empty zones need the same three zone fields but may omit pagesets. Nonempty
 * zones require CPU records. SMP records require exactly one threshold per CPU.
 * A consistent UP form (only cpu0 in every populated zone, no thresholds in the
 * entire snapshot) has zero per-CPU counter drift. Mixed/partial omissions fail.
 * Zone names are opaque. Unrelated counters are ignored; malformed/duplicate or
 * missing required fields throw invalid_argument, integer/sum overflow throws
 * overflow_error. This is a conservative snapshot, not an allocation guarantee.
 *
 * Format/semantics references only; no kernel implementation code is imported:
 * https://docs.kernel.org/admin-guide/sysctl/vm.html#lowmem-reserve-ratio
 * https://github.com/torvalds/linux/blob/v5.10/mm/vmstat.c
 */
inline std::map<unsigned, std::uint64_t> linux_zone_free_pages(std::string_view text) {
  using namespace linux_free_capacity_detail;
  std::map<unsigned, Pages> result;
  std::set<std::pair<unsigned, std::string>> seen;
  std::optional<Zone> zone;
  bool has_thresholds = false, has_up_form = false;
  auto finish = [&] {
    if (zone) {
      zone->finish_cpu();
      const auto available = zone->available();
      has_thresholds = has_thresholds || zone->thresholds != 0;
      has_up_form = has_up_form || zone->missing_thresholds != 0;
      result[zone->node] = add(result[zone->node], available);
    }
  };
  while (!text.empty()) {
    const auto newline = text.find('\n');
    const auto line = trim(text.substr(0, newline));
    if (newline == std::string_view::npos) text = {};
    else text.remove_prefix(newline + 1);
    if (line.empty()) continue;
    const auto words = tokens(line);
    if (words[0] == "Node") {
      finish();
      if (words.size() != 4 || words[1].size() < 2 || words[1].back() != ',' || words[2] != "zone")
        malformed("invalid node/zone header");
      const auto node = index(words[1].substr(0, words[1].size() - 1));
      if (!seen.emplace(node, std::string(words[3])).second) malformed("duplicate node/zone");
      zone.emplace(); zone->node = node;
      continue;
    }
    if (!zone) malformed("content before first node/zone");
    auto scalar = [&](std::size_t count) {
      if (words.size() != count) malformed("invalid scalar field");
      return number(words.back());
    };
    if (words[0] == "pages" && words.size() > 1 && words[1] == "free") {
      if (zone->pagesets) malformed("zone field inside pagesets");
      field(zone->free, scalar(3));
    } else if (words[0] == "high") {
      if (zone->pagesets) malformed("zone high inside pagesets");
      field(zone->high, scalar(2));
    } else if (words[0] == "protection:") {
      if (zone->pagesets) malformed("protection inside pagesets");
      field(zone->protection, protection(line.substr(words[0].size())));
    } else if (words[0] == "reserved_highatomic" || words[0] == "nr_reserved_highatomic") {
      if (zone->pagesets) malformed("highatomic reservation inside pagesets");
      // The two spellings identify one optional reservation, not two charges.
      field(zone->highatomic, scalar(2));
    } else if (words[0] == "pagesets") {
      if (words.size() != 1 || zone->pagesets) malformed("duplicate/invalid pagesets section");
      zone->pagesets = true;
    } else if (words[0] == "cpu:" || words[0] == "cpu") {
      if (words[0] != "cpu:" || words.size() != 2 || !zone->pagesets) malformed("invalid CPU section");
      zone->finish_cpu();
      const auto cpu = index(words[1]);
      if (!zone->cpus.insert(cpu).second) malformed("duplicate CPU section");
      zone->cpu = cpu; zone->threshold = false;
    } else if (words[0] == "vm" && words.size() > 1 && words[1] == "stats") {
      if (words.size() != 4 || words[2] != "threshold:" || !zone->cpu || zone->threshold)
        malformed("invalid/duplicate per-CPU vm stats threshold");
      zone->drift = add(zone->drift, number(words[3]));
      zone->threshold = true;
      ++zone->thresholds;
    }
  }
  finish();
  if (result.empty()) malformed("no node/zone records");
  if (has_thresholds && has_up_form) malformed("inconsistent per-CPU vm stats threshold availability");
  return result;
}
} // namespace vecops::memory::model::details
