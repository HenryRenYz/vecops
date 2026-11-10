// SPDX-FileCopyrightText: 2026 HenryRenYz and vecops contributors
// SPDX-License-Identifier: MIT
// Strict, address-free TOML wire format. Hardware/runtime validation is separate.
#include "vecops/memory/Model.h"
#include "../../third_party/tomlplusplus/toml.hpp"

#include <algorithm>
#include <limits>
#include <set>
#include <sstream>
#include <stdexcept>
#include <tuple>
#include <type_traits>

namespace vecops::memory::model {
namespace {
using U = std::uint64_t;
struct CodecError : std::runtime_error {
  std::string code;
  CodecError(std::string c, std::string message)
    : std::runtime_error(std::move(message))
    , code(std::move(c)) {
  }
};
[[noreturn]] void fail(std::string_view path, std::string_view why) {
  throw CodecError("toml.schema", std::string(path) + ": " + std::string(why));
}
void require(bool condition, std::string_view path, std::string_view message) {
  if (!condition)
    fail(path, message);
}
std::string child(std::string_view path, std::string_view key) {
  return std::string(path) + "." + std::string(key);
}

template <class T>
struct Optional : std::false_type {};
template <class T>
struct Optional<std::optional<T>> : std::true_type {
  using Value = T;
};
template <class T>
struct Vector : std::false_type {};
template <class T>
struct Vector<std::vector<T>> : std::true_type {
  using Value = T;
};
template <class T>
struct Record;
// One field list drives both directions and unknown-field validation.
#define FIELD(name) visit(#name, value.name);
#define RECORD(type, fields)                \
  template <>                               \
  struct Record<type> {                     \
    template <class V, class F>             \
    static void each(V& value, F&& visit) { \
      fields                                \
    }                                       \
  }
RECORD(PlacedSegment, FIELD(offset_bytes) FIELD(size_bytes) FIELD(pool));
RECORD(ResolvedPlacement, FIELD(key) FIELD(address_space) FIELD(page_bytes) FIELD(segments));
RECORD(VirtualRegion, FIELD(id) FIELD(address_space) FIELD(bytes) FIELD(page_bytes) FIELD(base_alignment_bytes));
RECORD(SharingContract, FIELD(id) FIELD(participants) FIELD(producers) FIELD(consumers) FIELD(protocol));
RECORD(AddressSlot,
       FIELD(id) FIELD(region) FIELD(offset_bytes) FIELD(size_bytes) FIELD(placement) FIELD(sharing_contract));
RECORD(AddressLayout, FIELD(regions) FIELD(slots));
RECORD(BackingRequest, FIELD(id) FIELD(owner) FIELD(pool) FIELD(bytes) FIELD(page_bytes) FIELD(mode) FIELD(export_to));
RECORD(PageMapping, FIELD(region) FIELD(virtual_offset_bytes) FIELD(backing) FIELD(backing_offset_bytes)
                      FIELD(size_bytes) FIELD(access));
RECORD(MemoryRecipe, FIELD(participant) FIELD(layout_id) FIELD(layout_generation) FIELD(sharing_contracts)
                       FIELD(addresses) FIELD(backings) FIELD(mappings));
RECORD(ParticipantLayout, FIELD(participant) FIELD(addresses) FIELD(mappings));
RECORD(DistributedLayout, FIELD(id) FIELD(generation) FIELD(sharing_contracts) FIELD(backings) FIELD(participants));
#undef RECORD
#undef FIELD

std::string_view spelling(AccessMode value) {
  switch (value) {
  case AccessMode::read:
    return "read";
  case AccessMode::write:
    return "write";
  }
  fail("access", "invalid C++ AccessMode value");
}
std::string_view spelling(BackingMode value) {
  switch (value) {
  case BackingMode::private_memory:
    return "private_memory";
  case BackingMode::shareable_remappable:
    return "shareable_remappable";
  case BackingMode::private_movable:
    return "private_movable";
  }
  fail("mode", "invalid C++ BackingMode value");
}

template <class T>
T decode(const toml::node& node, std::string_view path) {
  if constexpr (std::is_same_v<T, std::string>) {
    const auto* value = node.as_string();
    require(value != nullptr, path, "expected string");
    return value->get();
  } else if constexpr (std::is_same_v<T, U>) {
    const auto* value = node.as_integer();
    require(value != nullptr, path, "expected nonnegative integer (not float/string/bool)");
    require(value->get() >= 0, path, "expected nonnegative integer");
    return static_cast<U>(value->get());
  } else if constexpr (std::is_same_v<T, AccessMode>) {
    const auto value = decode<std::string>(node, path);
    if (value == "read")
      return AccessMode::read;
    if (value == "write")
      return AccessMode::write;
    fail(path, "unknown AccessMode");
  } else if constexpr (std::is_same_v<T, BackingMode>) {
    const auto value = decode<std::string>(node, path);
    if (value == "private_memory")
      return BackingMode::private_memory;
    if (value == "shareable_remappable")
      return BackingMode::shareable_remappable;
    if (value == "private_movable")
      return BackingMode::private_movable;
    fail(path, "unknown BackingMode");
  } else if constexpr (Vector<T>::value) {
    const auto* array = node.as_array();
    require(array != nullptr, path, "expected array");
    T result;
    result.reserve(array->size());
    for (std::size_t i = 0; i < array->size(); ++i)
      result.push_back(
        decode<typename Vector<T>::Value>((*array)[i], std::string(path) + "[" + std::to_string(i) + "]"));
    return result;
  } else {
    const auto* table = node.as_table();
    require(table != nullptr, path, "expected table");
    T result{};
    std::set<std::string> known;
    Record<T>::each(result, [&](std::string_view key, auto& field) {
      using F = std::remove_cvref_t<decltype(field)>;
      known.insert(std::string(key));
      const auto* value = table->get(key);
      if constexpr (Optional<F>::value) {
        if (value)
          field = decode<typename Optional<F>::Value>(*value, child(path, key));
      } else {
        require(value != nullptr, child(path, key), "missing required field");
        field = decode<F>(*value, child(path, key));
      }
    });
    for (const auto& [key, value] : *table) {
      (void)value;
      require(known.contains(std::string(key.str())), child(path, key.str()), "unknown field");
    }
    return result;
  }
}

template <class T>
auto encode(const T& value) {
  if constexpr (std::is_same_v<T, std::string>) {
    return value;
  } else if constexpr (std::is_same_v<T, U>) {
    // TOML 1.0 requires signed 64-bit integers; never silently truncate uint64.
    if (value > static_cast<U>(std::numeric_limits<std::int64_t>::max()))
      throw CodecError("toml.range", "integer exceeds the TOML signed-64-bit range");
    return static_cast<std::int64_t>(value);
  } else if constexpr (std::is_enum_v<T>) {
    return std::string(spelling(value));
  } else if constexpr (Vector<T>::value) {
    toml::array result;
    for (const auto& item : value)
      result.push_back(encode(item));
    return result;
  } else {
    toml::table result;
    Record<T>::each(value, [&](std::string_view key, const auto& field) {
      using F = std::remove_cvref_t<decltype(field)>;
      if constexpr (Optional<F>::value) {
        if (field)
          result.insert(key, encode(*field));
      } else
        result.insert(key, encode(field));
    });
    return result;
  }
}

bool power2(U n) {
  return n && !(n & (n - 1));
}
U end(U offset, U bytes, std::string_view path) {
  require(bytes != 0, path, "empty range");
  require(offset <= std::numeric_limits<U>::max() - bytes, path, "range addition overflows uint64");
  return offset + bytes;
}
void identifier(std::string_view value, std::string_view path) {
  require(!value.empty(), path, "empty identifier");
}
void strings(std::vector<std::string>& values, std::string_view path) {
  for (const auto& v : values)
    identifier(v, path);
  std::sort(values.begin(), values.end());
  require(std::adjacent_find(values.begin(), values.end()) == values.end(), path, "duplicate identifier");
}
template <class T, class Key>
void sort_unique(std::vector<T>& values, Key key, std::string_view path) {
  std::sort(values.begin(), values.end(), [&](const auto& a, const auto& b) { return key(a) < key(b); });
  for (std::size_t i = 0; i < values.size(); ++i) {
    identifier(key(values[i]), path);
    require(!i || key(values[i - 1]) != key(values[i]), path, "duplicate identifier");
  }
}
void contracts(std::vector<SharingContract>& values) {
  sort_unique(values, [](const auto& v) { return v.id; }, "sharing_contracts");
  for (auto& c : values) {
    strings(c.participants, "sharing_contracts.participants");
    strings(c.producers, "sharing_contracts.producers");
    strings(c.consumers, "sharing_contracts.consumers");
    identifier(c.protocol, "sharing_contracts.protocol");
    for (const auto* roles : {&c.producers, &c.consumers})
      for (const auto& p : *roles)
        require(std::binary_search(c.participants.begin(), c.participants.end(), p), "sharing_contracts",
                "role outside participants");
  }
}
void backings(std::vector<BackingRequest>& values) {
  sort_unique(values, [](const auto& v) { return v.id; }, "backings");
  for (auto& b : values) {
    identifier(b.owner, "backings.owner");
    identifier(b.pool, "backings.pool");
    require(power2(b.page_bytes) && b.bytes && b.bytes % b.page_bytes == 0, "backings", "invalid page geometry");
    (void)spelling(b.mode);
    strings(b.export_to, "backings.export_to");
    require(b.export_to.empty() || b.mode == BackingMode::shareable_remappable, "backings",
            "private backing cannot be exported");
  }
}
void addresses(AddressLayout& value) {
  sort_unique(value.regions, [](const auto& v) { return v.id; }, "addresses.regions");
  sort_unique(value.slots, [](const auto& v) { return v.id; }, "addresses.slots");
  std::map<std::string, const VirtualRegion*> regions;
  for (const auto& r : value.regions) {
    identifier(r.address_space, "regions.address_space");
    require(power2(r.page_bytes) && r.bytes && r.bytes % r.page_bytes == 0, "regions", "invalid page geometry");
    require(r.base_alignment_bytes == 0 || power2(r.base_alignment_bytes), "regions.base_alignment_bytes",
            "expected zero/default or a power of two");
    regions.emplace(r.id, &r);
  }
  for (auto& s : value.slots) {
    auto it = regions.find(s.region);
    require(it != regions.end(), "slots.region", "unknown virtual region");
    const auto& r = *it->second;
    const auto& p = s.placement;
    identifier(p.key, "placement.key");
    require(p.address_space == r.address_space && p.page_bytes == r.page_bytes, "placement",
            "region/placement address-space or page mismatch");
    require(s.offset_bytes % r.page_bytes == 0 && s.size_bytes % r.page_bytes == 0 &&
              end(s.offset_bytes, s.size_bytes, "slots") <= r.bytes,
            "slots", "unaligned or out-of-bounds range");
    if (s.sharing_contract)
      identifier(*s.sharing_contract, "slots.sharing_contract");
    auto& segments = s.placement.segments;
    std::sort(segments.begin(), segments.end(), [](const auto& a, const auto& b) {
      return std::tie(a.offset_bytes, a.size_bytes, a.pool) < std::tie(b.offset_bytes, b.size_bytes, b.pool);
    });
    U cursor = 0;
    for (const auto& segment : segments) {
      identifier(segment.pool, "segments.pool");
      require(segment.offset_bytes == cursor && segment.size_bytes % p.page_bytes == 0, "segments",
              "placement hole, overlap or unaligned size");
      cursor = end(segment.offset_bytes, segment.size_bytes, "segments");
      require(cursor <= s.size_bytes, "segments", "segment exceeds slot");
    }
    require(cursor == s.size_bytes, "segments", "placement does not cover slot");
  }
}
void mappings(std::vector<PageMapping>& values, const AddressLayout& layout,
              const std::vector<BackingRequest>& storage) {
  std::map<std::string, const VirtualRegion*> regions;
  std::map<std::string, const BackingRequest*> backing;
  for (const auto& r : layout.regions)
    regions.emplace(r.id, &r);
  for (const auto& b : storage)
    backing.emplace(b.id, &b);
  std::sort(values.begin(), values.end(), [](const auto& a, const auto& b) {
    return std::tie(a.region, a.virtual_offset_bytes, a.backing, a.backing_offset_bytes, a.size_bytes) <
           std::tie(b.region, b.virtual_offset_bytes, b.backing, b.backing_offset_bytes, b.size_bytes);
  });
  std::string previous_region;
  U previous_end = 0;
  std::map<std::string, unsigned> movable_maps;
  for (auto& m : values) {
    const auto ri = regions.find(m.region);
    const auto bi = backing.find(m.backing);
    require(ri != regions.end() && bi != backing.end(), "mappings", "unknown virtual region or backing");
    const auto& r = *ri->second;
    const auto& b = *bi->second;
    if (b.mode == BackingMode::private_movable) {
      require(m.backing_offset_bytes == 0 && m.size_bytes == b.bytes && ++movable_maps[b.id] == 1,
              "mappings", "private_movable requires a single whole-root mapping");
      for (const auto& slot : layout.slots) if (slot.region == m.region)
        require(!slot.sharing_contract, "mappings", "private_movable cannot have a sharing contract");
    }
    const U q = std::max(r.page_bytes, b.page_bytes);
    require(r.page_bytes >= b.page_bytes && m.virtual_offset_bytes % q == 0 && m.backing_offset_bytes % q == 0 &&
              m.size_bytes % q == 0,
            "mappings", "unaligned mapping");
    const U ve = end(m.virtual_offset_bytes, m.size_bytes, "mappings.virtual_range");
    require(ve <= r.bytes && end(m.backing_offset_bytes, m.size_bytes, "mappings.backing_range") <= b.bytes, "mappings",
            "mapping exceeds region or backing");
    require(previous_region != m.region || m.virtual_offset_bytes >= previous_end, "mappings",
            "overlapping virtual mappings");
    previous_region = m.region;
    previous_end = ve;
    require(!m.access.empty(), "mappings.access", "empty access permissions");
    for (auto access : m.access)
      (void)spelling(access);
    std::sort(m.access.begin(), m.access.end());
    require(std::adjacent_find(m.access.begin(), m.access.end()) == m.access.end(), "mappings.access",
            "duplicate access permission");
  }
}
void shared_slots(const AddressLayout& layout, const std::vector<SharingContract>& values,
                  const std::string& participant) {
  std::map<std::string, const SharingContract*> index;
  for (const auto& c : values)
    index.emplace(c.id, &c);
  for (const auto& slot : layout.slots) {
    if (!slot.sharing_contract)
      continue;
    const auto it = index.find(*slot.sharing_contract);
    require(it != index.end(), "slots.sharing_contract", "unknown sharing contract");
    const auto& members = it->second->participants;
    require(std::binary_search(members.begin(), members.end(), participant), "slots.sharing_contract",
            "local participant outside sharing contract");
  }
}
void normalize(MemoryRecipe& value) {
  identifier(value.participant, "participant");
  if (value.layout_id)
    identifier(*value.layout_id, "layout_id");
  contracts(value.sharing_contracts);
  backings(value.backings);
  addresses(value.addresses);
  mappings(value.mappings, value.addresses, value.backings);
  shared_slots(value.addresses, value.sharing_contracts, value.participant);
  for (const auto& b : value.backings)
    require(b.owner == value.participant ||
              std::binary_search(b.export_to.begin(), b.export_to.end(), value.participant),
            "backings.export_to", "foreign backing not exported to local participant");
}
void normalize(DistributedLayout& value) {
  identifier(value.id, "id");
  contracts(value.sharing_contracts);
  backings(value.backings);
  sort_unique(value.participants, [](const auto& v) { return v.participant; }, "participants");
  for (auto& p : value.participants) {
    addresses(p.addresses);
    mappings(p.mappings, p.addresses, value.backings);
    shared_slots(p.addresses, value.sharing_contracts, p.participant);
    for (const auto& m : p.mappings) {
      const auto& b = *std::find_if(value.backings.begin(), value.backings.end(),
                                    [&](const auto& item) { return item.id == m.backing; });
      require(b.owner == p.participant || std::binary_search(b.export_to.begin(), b.export_to.end(), p.participant),
              "participants.mappings", "foreign backing not exported to participant");
    }
  }
  std::set<std::string> participants;
  for (const auto& p : value.participants)
    participants.insert(p.participant);
  for (const auto& b : value.backings) {
    require(participants.contains(b.owner), "backings.owner", "owner outside layout participants");
    for (const auto& peer : b.export_to)
      require(participants.contains(peer), "backings.export_to", "export outside layout participants");
  }
  for (const auto& c : value.sharing_contracts)
    for (const auto& p : c.participants)
      require(participants.contains(p), "sharing_contracts.participants", "participant outside layout");
}

template <class T>
Result<std::string> write(const T& input, std::string_view kind) {
  try {
    auto value = input;
    normalize(value);
    auto table = encode(value);
    table.insert("schema_version", std::int64_t{1});
    table.insert("kind", kind);
    std::ostringstream stream;
    stream << toml::toml_formatter(table);
    return {stream.str() + "\n"};
  } catch (const CodecError& e) {
    return {Error{e.code, e.what()}};
  }
}
template <class T>
Result<T> read(std::string_view text, std::string_view kind) {
  try {
    auto table = toml::parse(text);
    const auto* version = table.get("schema_version");
    require(version && version->is_integer() && version->as_integer()->get() == 1, "schema_version",
            "expected integer schema version 1");
    const auto* type = table.get("kind");
    require(type && type->is_string() && type->as_string()->get() == kind, "kind", "incorrect document kind");
    table.erase("schema_version");
    table.erase("kind");
    T value = decode<T>(table, "root");
    normalize(value);
    return {std::move(value)};
  } catch (const toml::parse_error& e) {
    std::ostringstream stream;
    stream << e;
    return {Error{"toml.parse", stream.str()}};
  } catch (const CodecError& e) {
    return {Error{e.code, e.what()}};
  }
}
} // namespace

Result<std::string> recipe_to_toml(const MemoryRecipe& value) {
  return write(value, "memory_recipe");
}
Result<MemoryRecipe> recipe_from_toml(std::string_view text) {
  return read<MemoryRecipe>(text, "memory_recipe");
}
Result<std::string> layout_to_toml(const DistributedLayout& value) {
  return write(value, "distributed_layout");
}
Result<DistributedLayout> layout_from_toml(std::string_view text) {
  return read<DistributedLayout>(text, "distributed_layout");
}
} // namespace vecops::memory::model
