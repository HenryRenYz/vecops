// SPDX-FileCopyrightText: 2026 HenryRenYz and vecops contributors
// SPDX-License-Identifier: MIT
#include "vecops/memory/Model.h"
#include <gtest/gtest.h>
#include <algorithm>
#include <limits>

namespace vecops::memory::model {
namespace {
constexpr std::uint64_t page = 4096;
MemoryRecipe sample() {
  MemoryRecipe r{};
  r.participant = "rank.0";
  r.layout_id = "layout\n\"quoted\"\\shared";
  r.layout_generation = 7;
  r.sharing_contracts = {{"contract", {"rank.1", "rank.0"}, {"rank.0"}, {"rank.1"}, "publish/acquire/retire"}};
  r.addresses.regions = {{"virtual", "process-space", 4 * page, page}};
  r.addresses.slots = {{"view",
                        "virtual",
                        0,
                        2 * page,
                        {"opaque-placement", "process-space", page, {{page, page, "ddr"}, {0, page, "cowos_hbm"}}},
                        "contract"}};
  r.backings = {{"z-ddr", "rank.0", "ddr", 2 * page, page, BackingMode::private_memory, {}},
                {"a-hbm", "rank.0", "cowos_hbm", 2 * page, page, BackingMode::shareable_remappable, {"rank.1"}}};
  r.mappings = {{"virtual", page, "z-ddr", page, page, {AccessMode::read}},
                {"virtual", 0, "a-hbm", 0, page, {AccessMode::write, AccessMode::read}}};
  return r;
}
std::string text(MemoryRecipe r = sample()) {
  auto encoded = recipe_to_toml(r);
  EXPECT_TRUE(encoded);
  if (!encoded)
    return {};
  return encoded.value();
}
std::string replace_first(std::string value, std::string_view before, std::string_view after) {
  const auto at = value.find(before);
  EXPECT_NE(at, std::string::npos) << before << " in " << value;
  if (at != std::string::npos)
    value.replace(at, before.size(), after);
  return value;
}
void rejected(std::string_view input) {
  const auto result = recipe_from_toml(input);
  EXPECT_FALSE(result) << input;
}

TEST(MemorySerialization, PrivateMovableRoundTripAndWholeRootValidation) {
  MemoryRecipe r{};r.participant="p0";
  r.addresses.regions={{"va","host",2*page,page}};
  r.backings={{"root","p0","pool",2*page,page,BackingMode::private_movable,{}}};
  r.mappings={{"va",0,"root",0,2*page,{AccessMode::read,AccessMode::write}}};
  auto encoded=recipe_to_toml(r);ASSERT_TRUE(encoded);
  auto decoded=recipe_from_toml(encoded.value());ASSERT_TRUE(decoded);
  EXPECT_EQ(decoded.value().backings.front().mode,BackingMode::private_movable);
  r.mappings.front().size_bytes=page;EXPECT_FALSE(recipe_to_toml(r));
}

TEST(MemorySerialization, RegionBaseAlignmentRoundTripsIndependentlyOfLogicalPage) {
  auto r = sample();
  r.addresses.regions.front().base_alignment_bytes = std::uint64_t{1} << 22;
  const auto wire = text(r);
  auto decoded = recipe_from_toml(wire);ASSERT_TRUE(decoded);
  const auto& region = decoded.value().addresses.regions.front();
  EXPECT_EQ(region.base_alignment_bytes, std::uint64_t{1} << 22);
  EXPECT_EQ(region.page_bytes, page);
  EXPECT_EQ(region.bytes, 4 * page);
  EXPECT_EQ(recipe_to_toml(decoded.value()).value(), wire);
  r.addresses.regions.front().base_alignment_bytes = 3;
  EXPECT_FALSE(recipe_to_toml(r));
  rejected(replace_first(wire, "base_alignment_bytes = 4194304", "base_alignment_bytes = 3"));
  rejected(replace_first(wire, "base_alignment_bytes = 4194304", "base_alignment_bytes = -1"));
  rejected(replace_first(wire, "base_alignment_bytes = 4194304", "base_alignment_bytes = true"));
  const auto zero = text();
  auto missing = replace_first(zero, "base_alignment_bytes = 0", "");
  rejected(missing); // The current wire record requires the field, including zero.
}

TEST(MemorySerialization, RecipeRoundTripPreservesExactAddressFreeData) {
  const auto wire = text();
  auto parsed = recipe_from_toml(wire);
  ASSERT_TRUE(parsed);
  const auto& r = parsed.value();
  EXPECT_EQ(r.participant, "rank.0");
  EXPECT_EQ(r.layout_id, sample().layout_id);
  EXPECT_EQ(r.layout_generation, 7u);
  ASSERT_EQ(r.mappings.size(), 2u);
  EXPECT_EQ(r.mappings[1].backing_offset_bytes, page);
  EXPECT_EQ(r.mappings[1].access, (std::vector{AccessMode::read}));
  ASSERT_EQ(r.addresses.slots[0].placement.segments.size(), 2u);
  EXPECT_EQ(r.addresses.slots[0].placement.segments[0].pool, "cowos_hbm");
  auto again = recipe_to_toml(r);
  ASSERT_TRUE(again);
  EXPECT_EQ(again.value(), wire);
}
TEST(MemorySerialization, LayoutRoundTripAndParticipantOrder) {
  auto r = sample();
  DistributedLayout l{
    *r.layout_id,
    9,
    r.sharing_contracts,
    r.backings,
    {{"rank.1", {{{"v1", "peer-space", 2 * page, page}}, {}}, {}}, {r.participant, r.addresses, r.mappings}}};
  auto encoded = layout_to_toml(l);
  ASSERT_TRUE(encoded);
  auto decoded = layout_from_toml(encoded.value());
  ASSERT_TRUE(decoded);
  EXPECT_EQ(decoded.value().participants[0].participant, "rank.0");
  EXPECT_EQ(decoded.value().generation, 9u);
  auto again = layout_to_toml(decoded.value());
  ASSERT_TRUE(again);
  EXPECT_EQ(again.value(), encoded.value());
  EXPECT_FALSE(recipe_from_toml(encoded.value()));
  EXPECT_FALSE(layout_from_toml(text()));
  EXPECT_FALSE(layout_from_toml(replace_first(encoded.value(), "participant = 'rank.1'", "participant = 'rank.0'")));
}
TEST(MemorySerialization, CanonicalOutputIndependentOfSetAndObjectOrder) {
  auto a = sample();
  auto b = a;
  std::reverse(b.backings.begin(), b.backings.end());
  std::reverse(b.mappings.begin(), b.mappings.end());
  std::reverse(b.sharing_contracts[0].participants.begin(), b.sharing_contracts[0].participants.end());
  std::reverse(b.addresses.slots[0].placement.segments.begin(), b.addresses.slots[0].placement.segments.end());
  std::reverse(b.mappings[1].access.begin(), b.mappings[1].access.end());
  EXPECT_EQ(text(a), text(b));
  // Normalization must not mutate callers' objects.
  EXPECT_EQ(a.backings[0].id, "z-ddr");
  EXPECT_EQ(a.addresses.slots[0].placement.segments[0].offset_bytes, page);
}
TEST(MemorySerialization, CompleteTomlSyntaxHandledByUpstreamParser) {
  const std::string fixture = R"toml(
# UTF-8, multiline strings, dotted keys, literal keys and hexadecimal integers.
schema_version = 1
kind = 'memory_recipe'
participant = "进程 α"
layout_id = """a
b"""
layout_generation = 0x7fff_ffff_ffff_ffff
sharing_contracts = []
backings = []
mappings = []
addresses.regions = []
addresses.slots = []
)toml";
  auto result = recipe_from_toml(fixture);
  ASSERT_TRUE(result);
  EXPECT_EQ(result.value().participant, "进程 α");
  EXPECT_EQ(result.value().layout_id, "a\nb");
  EXPECT_EQ(result.value().layout_generation, static_cast<std::uint64_t>(std::numeric_limits<std::int64_t>::max()));
  ASSERT_TRUE(recipe_to_toml(result.value()));
}
TEST(MemorySerialization, ExactIntegerRulesRejectFloatsStringsBooleansAndOverflow) {
  const auto wire = text();
  for (auto bad : {"-1", "0.0", "nan", "inf", "true", "'4096'", "9223372036854775808", "18446744073709551615"})
    rejected(replace_first(wire, "layout_generation = 7", std::string("layout_generation = ") + bad));
  for (auto bad : {"-1", "0.5", "nan", "false", "'0'"})
    rejected(replace_first(wire, "virtual_offset_bytes = 0", std::string("virtual_offset_bytes = ") + bad));
  auto r = sample();
  r.layout_generation = std::numeric_limits<std::uint64_t>::max();
  auto result = recipe_to_toml(r);
  ASSERT_FALSE(result);
  EXPECT_EQ(std::get<Error>(result.state).code, "toml.range");
}
TEST(MemorySerialization, SchemaUnknownMissingDuplicateAndWrongTypesRejected) {
  const auto wire = text();
  rejected(replace_first(wire, "schema_version = 1", "schema_version = 2"));
  rejected(replace_first(wire, "schema_version = 1", "schema_version = 1.0"));
  rejected(replace_first(wire, "schema_version = 1", ""));
  rejected(replace_first(wire, "layout_generation = 7", "runtime_address = 123\nlayout_generation = 7"));
  rejected(replace_first(wire, "layout_generation = 7", "layout_generation = 7\nlayout_generation = 7"));
  rejected(replace_first(wire, "layout_generation = 7", "layout_generation = []"));
  rejected(replace_first(wire, "[[addresses.regions]]", "[addresses]\nfd = 4\n[[addresses.regions]]"));
  rejected(replace_first(wire, "pool = 'cowos_hbm'", "pool = 'cowos_hbm'\nhandle = 'native'"));
  rejected(replace_first(wire, "mode = 'shareable_remappable'", "mode = 'mmap'"));
  rejected(replace_first(wire, "access = [ 'read', 'write' ]", "access = ['read', 'execute']"));
  rejected(replace_first(wire, "participant = 'rank.0'", "participant = 0"));
  rejected("schema_version = 1\nkind = 'memory_recipe'\nparticipant = 'p'\n");
}
TEST(MemorySerialization, InvalidGeometryDuplicatesAndReferencesRejectedBothDirections) {
  auto r = sample();
  r.addresses.regions[0].page_bytes = 3;
  EXPECT_FALSE(recipe_to_toml(r));
  r = sample();
  r.backings.push_back(r.backings[0]);
  EXPECT_FALSE(recipe_to_toml(r));
  r = sample();
  r.addresses.slots.push_back(r.addresses.slots[0]);
  EXPECT_FALSE(recipe_to_toml(r));
  r = sample();
  r.sharing_contracts[0].participants.push_back("rank.0");
  EXPECT_FALSE(recipe_to_toml(r));
  r = sample();
  r.backings[1].mode = BackingMode::private_memory;
  EXPECT_FALSE(recipe_to_toml(r));
  r = sample();
  r.backings[1].mode = static_cast<BackingMode>(255);
  EXPECT_FALSE(recipe_to_toml(r));
  r = sample();
  r.mappings[0].access = {static_cast<AccessMode>(255)};
  EXPECT_FALSE(recipe_to_toml(r));
  r = sample();
  r.mappings[0].backing = "unknown";
  EXPECT_FALSE(recipe_to_toml(r));
  r = sample();
  r.addresses.slots[0].sharing_contract = "unknown";
  EXPECT_FALSE(recipe_to_toml(r));
  r = sample();
  r.backings[0].owner = "rank.1";
  EXPECT_FALSE(recipe_to_toml(r));
  r = sample();
  r.mappings[0].virtual_offset_bytes = 0;
  EXPECT_FALSE(recipe_to_toml(r));
  r = sample();
  r.mappings[0].size_bytes = 0;
  EXPECT_FALSE(recipe_to_toml(r));
  r = sample();
  r.mappings[0].access = {AccessMode::read, AccessMode::read};
  EXPECT_FALSE(recipe_to_toml(r));
  r = sample();
  r.addresses.slots[0].placement.segments[0].offset_bytes = 2 * page;
  EXPECT_FALSE(recipe_to_toml(r));
  const auto wire = text();
  rejected(replace_first(wire, "backing = 'a-hbm'", "backing = 'unknown'"));
  rejected(replace_first(wire, "virtual_offset_bytes = 4096", "virtual_offset_bytes = 0"));
  rejected(replace_first(wire, "size_bytes = 4096", "size_bytes = 0"));
}
TEST(MemorySerialization, OptionalFieldsAbsentAreNotInvented) {
  MemoryRecipe r{};
  r.participant = "empty";
  auto wire = recipe_to_toml(r);
  ASSERT_TRUE(wire);
  EXPECT_EQ(wire.value().find("layout_id"), std::string::npos);
  auto result = recipe_from_toml(wire.value());
  ASSERT_TRUE(result);
  EXPECT_FALSE(result.value().layout_id);
  EXPECT_TRUE(result.value().addresses.slots.empty());
}
} // namespace
} // namespace vecops::memory::model
