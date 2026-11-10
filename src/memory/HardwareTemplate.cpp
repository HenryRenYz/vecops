// SPDX-FileCopyrightText: 2026 HenryRenYz and vecops contributors
// SPDX-License-Identifier: MIT
#include "vecops/memory/HardwareTemplate.h"
#include "vecops/memory/Calibration.h"
#include "../../third_party/tomlplusplus/toml.hpp"
#include <algorithm>
#include <charconv>
#include <cmath>
#include <deque>
#include <limits>
#include <regex>
#include <set>
#include <sstream>
#include <stdexcept>
#include <tuple>

namespace vecops::memory::model {
namespace {
using U = std::uint64_t;
void check(bool condition, const std::string& message) {
  if (!condition)
    throw std::invalid_argument(message);
}
template <class F>
auto checked(F&& function) -> Result<decltype(function())> {
  try {
    return {function()};
  } catch (const std::invalid_argument& e) {
    return {Error{"hardware.schema", e.what()}};
  } catch (const std::overflow_error& e) {
    return {Error{"hardware.range", e.what()}};
  } catch (const toml::parse_error& e) {
    std::ostringstream message;
    message << e;
    return {Error{"hardware.toml", message.str()}};
  }
}
U add(U a, U b) {
  if (b > std::numeric_limits<U>::max() - a)
    throw std::overflow_error("integer addition overflow");
  return a + b;
}
U multiply(U a, U b) {
  if (b && a > std::numeric_limits<U>::max() / b)
    throw std::overflow_error("integer multiplication overflow");
  return a * b;
}
U decimal(std::string_view text) {
  U result{};
  auto parsed = std::from_chars(text.data(), text.data() + text.size(), result);
  if (parsed.ec == std::errc::result_out_of_range)
    throw std::overflow_error("decimal integer exceeds uint64");
  check(parsed.ec == std::errc{} && parsed.ptr == text.data() + text.size() && !text.empty(),
        "invalid decimal integer");
  return result;
}
U size_bytes(std::string_view text) {
  static const std::regex pattern(R"(^([0-9]+)(?:\.([0-9]+))?\s*(GiB|MiB|KiB|B)$)");
  std::match_results<std::string_view::const_iterator> match;
  check(std::regex_match(text.begin(), text.end(), match, pattern),
        "invalid explicit memory size: " + std::string(text));
  const std::string unit = match[3];
  const U scale = unit == "GiB" ? 1ULL << 30 : unit == "MiB" ? 1ULL << 20 : unit == "KiB" ? 1ULL << 10 : 1;
  U result = multiply(decimal(std::string_view(match[1].first, match[1].length())), scale);
  if (match[2].matched) {
    // Repeated decimal long multiplication avoids a floating-point roundtrip,
    // and avoids overflowing a 10^N denominator for long decimal spellings.
    U carry = 0;
    const std::string fraction = match[2];
    for (auto it = fraction.rbegin(); it != fraction.rend(); ++it) {
      const U digit = static_cast<U>(*it - '0');
      const U next = digit * scale + carry; // scale <= 2^30
      check(next % 10 == 0, "memory size has fractional bytes");
      carry = next / 10;
    }
    result = add(result, carry);
  }
  check(result > 0, "memory size must be positive");
  return result;
}
const toml::node& required(const toml::table& table, std::string_view key) {
  const auto* value = table.get(key);
  check(value != nullptr, "missing field: " + std::string(key));
  return *value;
}
const toml::table& table(const toml::node& node) {
  const auto* value = node.as_table();
  check(value != nullptr, "expected table");
  return *value;
}
std::string string(const toml::node& node) {
  const auto* value = node.as_string();
  check(value != nullptr, "expected string");
  return value->get();
}
std::string str(const toml::table& t, std::string_view key) {
  return string(required(t, key));
}
std::string str(const toml::table& t, std::string_view key, std::string fallback) {
  const auto* n = t.get(key);
  return n ? string(*n) : std::move(fallback);
}
U integer(const toml::node& node) {
  const auto* value = node.as_integer();
  check(value && value->get() >= 0, "expected nonnegative integer");
  return static_cast<U>(value->get());
}
bool boolean(const toml::table& t, std::string_view key, bool fallback = false) {
  const auto* n = t.get(key);
  if (!n)
    return fallback;
  check(n->is_boolean(), "expected boolean: " + std::string(key));
  return n->as_boolean()->get();
}
double number(const toml::node& n, bool zero = false) {
  double value;
  if (n.is_integer())
    value = static_cast<double>(n.as_integer()->get());
  else {
    check(n.is_floating_point(), "expected numeric bandwidth/time");
    value = n.as_floating_point()->get();
  }
  check(std::isfinite(value) && (zero ? value >= 0 : value > 0), "nonfinite or invalid bandwidth/time");
  return value;
}
std::optional<double> numeric(const toml::table& t, std::string_view key, bool zero = false) {
  const auto* n = t.get(key);
  return n ? std::optional{number(*n, zero)} : std::nullopt;
}
void fields(const toml::table& t, std::initializer_list<std::string_view> allowed, std::string_view path) {
  for (const auto& [key, value] : t) {
    (void)value;
    check(std::find(allowed.begin(), allowed.end(), key.str()) != allowed.end(),
          std::string(path) + ": unknown field " + std::string(key.str()));
  }
}
std::vector<std::string> string_array(const toml::node& n, bool allow_scalar = false) {
  if (allow_scalar && n.is_string())
    return {string(n)};
  const auto* array = n.as_array();
  check(array, "expected string array");
  std::vector<std::string> result;
  for (const auto& x : *array) {
    auto s = string(x);
    check(!s.empty(), "empty string selection");
    check(std::find(result.begin(), result.end(), s) == result.end(), "duplicate string selection");
    result.push_back(std::move(s));
  }
  return result;
}
std::vector<std::string> optional_strings(const toml::table& t, std::string_view key) {
  const auto* n = t.get(key);
  return n ? string_array(*n) : std::vector<std::string>{};
}
std::map<std::string, double> rates(const toml::node& n, bool allow_zero = false) {
  const auto& t = table(n);
  fields(t, {"read", "write"}, "read/write rates");
  std::map<std::string, double> result;
  for (const auto& [key, value] : t)
    result.emplace(key.str(), number(value, allow_zero));
  return result;
}
HardwareFact fact(const toml::node& n) {
  if (n.is_string())
    return string(n);
  if (n.is_boolean())
    return n.as_boolean()->get();
  return integer(n);
}
bool power2(U n) {
  return n && !(n & (n - 1));
}
struct Object {
  std::string id, prototype, section;
  const toml::table* spec;
  std::vector<std::pair<std::string, std::string>> ancestry;
  std::optional<std::string> ancestor(std::string_view name) const {
    for (const auto& [key, value] : ancestry)
      if (key == name)
        return value;
    return std::nullopt;
  }
};
struct ExpandedResource {
  Resource value;
  std::optional<std::string> owner, source, target;
};
class Compiler {
public:
  explicit Compiler(toml::table source, const RuntimeCounts& counts)
    : source_(std::move(source))
    , counts_(counts) {
  }
  HardwareExpansion run() {
    validate();
    objects();
    resources();
    paths();
    output();
    transfers();
    domains();
    return std::move(result_);
  }
  CalibrationSelectors calibration_selectors() const {
    return {[this](std::string_view token) { return select(token); },
            [this](std::string_view agent, std::string_view pool, std::string_view test) {
              const auto a = objects_.find(std::string(agent)), b = objects_.find(std::string(pool));
              check(a != objects_.end() && b != objects_.end(), "unknown calibration relation object");
              return relation(a->second, b->second, test);
            }};
  }

private:
  toml::table source_;
  const RuntimeCounts& counts_;
  HardwareExpansion result_;
  std::map<std::string, const toml::table*> sections_;
  std::map<std::string, Object> objects_;
  std::map<std::string, std::vector<std::string>> instances_;
  std::vector<std::string> expansion_order_;
  std::vector<ExpandedResource> resources_;
  std::map<std::pair<std::string, std::string>, std::string> services_;
  const toml::table& section(const std::string& name) const {
    return *sections_.at(name);
  }
  void validate() {
    fields(
      source_,
      {"schema", "name", "units", "require", "groups", "agents", "pools", "domains", "resources", "paths", "transfers"},
      "root");
    check(str(source_, "schema") == "memory-model/0.3", "unsupported hardware schema");
    result_.name = str(source_, "name");
    check(!result_.name.empty(), "empty template name");
    const auto& units = table(required(source_, "units"));
    fields(units, {"bandwidth"}, "units");
    check(str(units, "bandwidth") == "GiB/s", "bandwidth unit must be GiB/s");
    if (const auto* req = source_.get("require"))
      for (const auto& [key, value] : table(*req))
        result_.requirements.emplace(key.str(), fact(value));
    for (const auto& name : {"groups", "agents", "pools", "domains", "resources", "paths", "transfers"}) {
      if (!source_.contains(name))
        source_.insert(name, toml::table{});
      sections_[name] = &table(required(source_, name));
      for (const auto& [key, value] : section(name)) {
        check(std::regex_match(std::string(key.str()), std::regex("[A-Za-z_][A-Za-z0-9_-]*")),
              "invalid prototype/rule name");
        const auto& t = table(value);
        if (name == std::string_view("groups"))
          fields(t, {"count", "per"}, name);
        else if (name == std::string_view("agents"))
          fields(t, {"count", "per", "bind", "capabilities"}, name);
        else if (name == std::string_view("pools"))
          fields(t,
                 {"count", "per", "bind", "capacity", "granularity", "page_sizes", "service", "read_service",
                  "write_service"},
                 name);
        else if (name == std::string_view("domains"))
          fields(t, {"per", "members"}, name);
        else if (name == std::string_view("resources"))
          fields(
            t,
            {"per", "between", "within", "topology", "edges", "bandwidth", "read", "write", "sharing", "kind", "slots"},
            name);
        else if (name == std::string_view("paths"))
          fields(t,
                 {"agent", "pool", "within", "where", "access", "mapping", "prerequisites", "bandwidth", "latency_ns",
                  "use_fabric", "priority"},
                 name);
        else
          fields(t,
                 {"from", "to", "by", "within", "where", "use", "traffic", "bandwidth", "startup_us", "controller",
                  "registration_bits"},
                 name);
      }
    }
  }
  void objects() {
    struct Pending {
      std::string section;
      const toml::table* spec;
    };
    std::map<std::string, Pending> pending;
    std::set<std::string> runtime;
    for (const auto& name : {"groups", "agents", "pools"})
      for (const auto& [key, value] : section(name)) {
        check(pending.emplace(key.str(), Pending{name, &table(value)}).second, "duplicate object prototype");
        const auto* count = table(value).get("count");
        if (count && count->is_string()) {
          check(string(*count) == "runtime", "invalid runtime count token");
          runtime.emplace(key.str());
        }
      }
    for (const auto& [name, count] : counts_) {
      (void)count;
      check(runtime.contains(name), "unknown runtime count: " + name);
    }
    for (const auto& name : runtime)
      check(counts_.contains(name), "missing runtime count: " + name);
    for (const auto& [name, item] : pending)
      if (const auto* parent = item.spec->get("per"))
        check(pending.contains(string(*parent)), "unknown per prototype: " + name);
    while (!pending.empty()) {
      std::vector<std::string> ready;
      for (const auto& [name, item] : pending)
        if (!item.spec->contains("per") || instances_.contains(str(*item.spec, "per")))
          ready.push_back(name);
      check(!ready.empty(), "cyclic per references");
      for (const auto& name : ready) {
        const auto item = pending.at(name);
        pending.erase(name);
        const auto& spec = *item.spec;
        std::vector<std::string> parents =
          spec.contains("per") ? instances_.at(str(spec, "per")) : std::vector<std::string>{""};
        U count = 1;
        if (const auto* n = spec.get("count"))
          count = n->is_string() ? counts_.at(name) : integer(*n);
        check(count > 0, "repeat count must be positive");
        // Bound materialization, rather than allowing an untrusted file to allocate without limit.
        check(count <= 1000000 && parents.size() <= 1000000 / count &&
                objects_.size() + parents.size() * count <= 1000000,
              "expanded object count exceeds 1000000");
        instances_[name] = {};
        for (const auto& parent : parents)
          for (U i = 0; i < count; ++i) {
            const auto id = (parent.empty() ? "" : parent + "/") + name + "[" + std::to_string(i) + "]";
            Object object{id, name, item.section, item.spec, {}};
            if (!parent.empty())
              object.ancestry = objects_.at(parent).ancestry;
            object.ancestry.emplace_back(name, id);
            objects_.emplace(id, std::move(object));
            instances_[name].push_back(id);
            expansion_order_.push_back(id);
          }
      }
    }
  }
  std::string root(const Object& object) const {
    for (const auto& [name, id] : object.ancestry) {
      (void)name;
      if (objects_.at(id).section == "groups")
        return id;
    }
    return "@global";
  }
  std::vector<std::string> select(std::string_view token) const {
    static const std::regex pattern(R"(^([A-Za-z_][A-Za-z0-9_-]*)(?:\[([0-9]+)\])?$)");
    std::match_results<std::string_view::const_iterator> match;
    check(std::regex_match(token.begin(), token.end(), match, pattern), "invalid selector: " + std::string(token));
    const std::string prototype = match[1];
    check(instances_.contains(prototype), "unknown selector: " + std::string(token));
    if (!match[2].matched)
      return instances_.at(prototype);
    const U index = decimal(std::string_view(match[2].first, match[2].length()));
    std::map<std::string, std::vector<std::string>> components;
    for (const auto& id : instances_.at(prototype))
      components[root(objects_.at(id))].push_back(id);
    std::vector<std::string> result;
    for (const auto& [name, items] : components) {
      (void)name;
      check(index < items.size(), "selector index out of range: " + std::string(token));
      result.push_back(items[static_cast<std::size_t>(index)]);
    }
    return result;
  }
  bool relation(const Object& a, const Object& b, std::string_view test) const {
    if (test == "any")
      return true;
    if (test == "local")
      return a.ancestor(a.prototype) == b.ancestor(a.prototype) && b.ancestor(a.prototype).has_value();
    if (test == "same_pool")
      return a.id == b.id;
    static const std::regex pattern(R"(^(same|different)\(([A-Za-z_][A-Za-z0-9_-]*)\)$)");
    std::match_results<std::string_view::const_iterator> match;
    check(std::regex_match(test.begin(), test.end(), match, pattern), "invalid relation: " + std::string(test));
    const auto left = a.ancestor(std::string(match[2]));
    const auto right = b.ancestor(std::string(match[2]));
    check(left && right, "relation names an absent ancestor: " + std::string(test));
    return match[1] == "same" ? left == right : left != right;
  }
  bool matches(const Object& a, const Object& b, const toml::table& spec) const {
    if (spec.contains("within") && !relation(a, b, "same(" + str(spec, "within") + ")"))
      return false;
    return relation(a, b, str(spec, "where", "any"));
  }
  bool compatible(const Object& a, const Object& b) const {
    for (const auto& [key, id] : a.ancestry)
      if (const auto other = b.ancestor(key); other && *other != id)
        return false;
    return true;
  }
  Resource resource(std::string id, std::string rule, const toml::table& spec,
                    std::optional<double> edge_rate = {}) const {
    const auto sharing = str(spec, "sharing", "shared");
    check(sharing == "shared" || sharing == "read_write", "unknown sharing mode");
    const auto kind = str(spec, "kind", "bandwidth");
    check(kind == "bandwidth" || kind == "execution_slots", "unknown resource kind");
    Resource result{std::move(id),
                    std::move(rule),
                    edge_rate,
                    numeric(spec, "read"),
                    numeric(spec, "write"),
                    sharing == "shared" ? SharingMode::shared : SharingMode::read_write,
                    kind == "bandwidth" ? ResourceKind::bandwidth : ResourceKind::execution_slots,
                    {}};
    if (!edge_rate && spec.contains("bandwidth"))
      result.bandwidth = number(required(spec, "bandwidth"));
    if (const auto* n = spec.get("slots"))
      result.slots = integer(*n);
    if (result.kind == ResourceKind::execution_slots)
      check(result.slots && *result.slots && !result.bandwidth && !result.read && !result.write,
            "slot resource needs positive slots and no bandwidth");
    else
      check(!result.slots, "bandwidth resource cannot contain slots");
    return result;
  }
  void resources() {
    for (const auto& [key, n] : section("resources")) {
      const std::string name(key.str());
      const auto& spec = table(n);
      if (spec.contains("per")) {
        check(!spec.contains("between") && !spec.contains("within") && !spec.contains("edges") &&
                !spec.contains("topology"),
              "per resource cannot also describe fabric");
        for (const auto& id : select(str(spec, "per"))) {
          const std::string rid = id + "/" + name;
          resources_.push_back({resource(rid, name, spec), id, {}, {}});
          services_.emplace(std::pair{name, id}, rid);
        }
      } else {
        const auto axis = str(spec, "between"), scope = str(spec, "within");
        check(instances_.contains(axis) && instances_.contains(scope), "unknown fabric axis/scope");
        const auto sharing = str(spec, "sharing", "shared");
        check(sharing == "shared" || sharing == "read_write", "unknown sharing mode");
        const auto topology = str(spec, "topology", "complete");
        check(topology == "complete" || topology == "explicit", "unknown fabric topology");
        check(topology == "complete" || spec.contains("edges"), "explicit topology needs edges");
        check(!spec.contains("read") && !spec.contains("write") && !spec.contains("slots") &&
                str(spec, "kind", "bandwidth") == "bandwidth",
              "fabric edges require bandwidth resources");
        const auto& bandwidth = required(spec, "bandwidth");
        if (bandwidth.is_array()) {
          for (const auto& rate : *bandwidth.as_array())
            (void)number(rate);
        } else
          (void)number(bandwidth);
        for (const auto& owner : instances_.at(scope)) {
          std::vector<std::string> children;
          for (const auto& id : instances_.at(axis))
            if (objects_.at(id).ancestor(scope) == owner)
              children.push_back(id);
          check(!children.empty(), "fabric scope contains no between instances");
          std::vector<std::pair<U, U>> pairs;
          if (const auto* edges = spec.get("edges")) {
            check(edges->is_array(), "edges must be array");
            for (const auto& pair : *edges->as_array()) {
              check(pair.is_array() && pair.as_array()->size() == 2, "directed edge needs two indices");
              pairs.emplace_back(integer((*pair.as_array())[0]), integer((*pair.as_array())[1]));
            }
          } else
            for (U i = 0; i < children.size(); ++i)
              for (U j = 0; j < children.size(); ++j)
                if (i != j)
                  pairs.emplace_back(i, j);
          if (bandwidth.is_array())
            check(bandwidth.as_array()->size() == pairs.size(), "one bandwidth per directed edge required");
          std::set<std::pair<U, U>> seen;
          for (std::size_t i = 0; i < pairs.size(); ++i) {
            const auto [s, t] = pairs[i];
            check(s != t && s < children.size() && t < children.size() && seen.insert(pairs[i]).second,
                  "invalid/duplicate directed edge");
            const auto& a = children[static_cast<std::size_t>(s)];
            const auto& b = children[static_cast<std::size_t>(t)];
            const double rate = number(bandwidth.is_array() ? (*bandwidth.as_array())[i] : bandwidth);
            resources_.push_back({resource(a + "->" + b + "/" + name, name, spec, rate), {}, a, b});
          }
        }
      }
    }
  }
  std::optional<std::string> memory_service(const Object& object, std::string_view operation) const {
    const auto name = str(*object.spec, std::string(operation) + "_service", str(*object.spec, "service", ""));
    return name.empty() ? std::nullopt : std::optional{service(object, name)};
  }
  std::string service(const Object& object, const std::string& name) const {
    const auto* n = section("resources").get(name);
    check(n, "unknown resource selector: " + name);
    const auto& spec = table(*n);
    check(spec.contains("per"), "endpoint resource must use per: " + name);
    std::vector<std::string> owners;
    for (const auto& id : select(str(spec, "per"))) {
      const auto& owner = objects_.at(id);
      if (object.ancestor(owner.prototype) == id)
        owners.push_back(id);
    }
    check(owners.size() == 1, "resource owner missing/ambiguous: " + name);
    return services_.at({name, owners[0]});
  }
  std::vector<std::string> fabric(const std::string& rule, const std::string& source, const std::string& target) const {
    std::deque<std::pair<std::string, std::vector<std::string>>> queue{{source, {}}};
    std::set<std::string> seen{source};
    while (!queue.empty()) {
      auto [at, path] = std::move(queue.front());
      queue.pop_front();
      if (at == target)
        return path;
      for (const auto& edge : resources_)
        if (edge.value.rule == rule && edge.source == at && edge.target && seen.insert(*edge.target).second) {
          auto next = path;
          next.push_back(edge.value.id);
          queue.emplace_back(*edge.target, std::move(next));
        }
    }
    throw std::invalid_argument("no directed fabric path for " + source + " -> " + target);
  }
  void paths() {
    using Score = std::tuple<std::int64_t, int, std::size_t>;
    struct Choice {
      Score score;
      AccessConnection edge;
      bool use_fabric;
      bool ambiguous;
    };
    std::map<std::pair<std::string, std::string>, Choice> connections;
    for (const auto& [key, n] : section("paths")) {
      const std::string name(key.str());
      const auto& spec = table(n);
      const auto agents = select(str(spec, "agent")), pools = select(str(spec, "pool"));
      std::int64_t priority = 0;
      if (const auto* p = spec.get("priority")) {
        check(p->is_integer(), "priority must be integer");
        priority = p->as_integer()->get();
      }
      const auto test = str(spec, "where", "any");
      (void)relation(objects_.at(agents.front()), objects_.at(pools.front()), test);
      const auto mode = str(spec, "mapping", "unknown");
      check(mode == "unknown" || mode == "none" || mode == "direct" || mode == "requires_mapping",
            "unknown mapping mode");
      const auto mapping = mode == "none"               ? MappingMode::none
                           : mode == "direct"           ? MappingMode::direct
                           : mode == "requires_mapping" ? MappingMode::requires_mapping
                                                        : MappingMode::unknown;
      std::vector<AccessMode> access;
      const auto permissions =
        spec.contains("access") ? string_array(required(spec, "access")) : std::vector<std::string>{"read", "write"};
      check(!permissions.empty(), "empty access rights");
      for (const auto& permission : permissions) {
        check(permission == "read" || permission == "write", "unknown access permission");
        access.push_back(permission == "read" ? AccessMode::read : AccessMode::write);
      }
      const bool use_fabric = boolean(spec, "use_fabric");
      const auto prerequisites = optional_strings(spec, "prerequisites");
      std::optional<std::map<std::string, double>> bandwidth;
      if (const auto* rate = spec.get("bandwidth"))
        bandwidth = rates(*rate);
      const auto latency = numeric(spec, "latency_ns", true);
      for (const auto& aid : agents)
        for (const auto& pid : pools) {
          const auto& a = objects_.at(aid);
          const auto& p = objects_.at(pid);
          check(a.section == "agents" && p.section == "pools", "path endpoints must be agents and pools");
          if (!matches(a, p, spec))
            continue;
          std::size_t specificity = 0;
          if (test == "local")
            specificity = a.ancestry.size() + 1;
          else if (test.starts_with("same(")) {
            const auto axis = test.substr(5, test.size() - 6);
            for (std::size_t i = 0; i < a.ancestry.size(); ++i)
              if (a.ancestry[i].first == axis)
                specificity = i + 1;
          }
          const Score score{priority,
                            static_cast<int>(str(spec, "agent").find('[') != std::string::npos) +
                              static_cast<int>(str(spec, "pool").find('[') != std::string::npos),
                            specificity};
          auto existing = connections.find({aid, pid});
          if (existing != connections.end()) {
            if (score == existing->second.score) {
              existing->second.ambiguous = true;
              continue;
            }
            if (score < existing->second.score)
              continue;
          }
          AccessConnection edge{aid + "->" + pid + "/" + name,
                                name,
                                aid,
                                pid,
                                access,
                                bandwidth,
                                latency,
                                mapping,
                                prerequisites,
                                memory_service(p, "read"),
                                memory_service(p, "write"),
                                {},
                                {}};
          connections[{aid, pid}] = {score, std::move(edge), use_fabric, false};
        }
    }
    for (auto& [key, choice] : connections) {
      (void)key;
      check(!choice.ambiguous, "ambiguous highest-priority access rules: " + choice.edge.id);
      auto& edge = choice.edge;
      const auto& a = objects_.at(edge.agent);
      const auto& p = objects_.at(edge.pool);
      if (choice.use_fabric)
        for (const auto& [rname, rspec] : section("resources")) {
          const auto& rs = table(rspec);
          if (!rs.contains("between"))
            continue;
          const auto axis = str(rs, "between"), scope = str(rs, "within");
          const auto start = a.ancestor(axis), stop = p.ancestor(axis);
          if (!start || !stop || start == stop || a.ancestor(scope) != p.ancestor(scope))
            continue;
          if (std::find(edge.access.begin(), edge.access.end(), AccessMode::read) != edge.access.end()) {
            auto route = fabric(std::string(rname.str()), *stop, *start);
            edge.read_fabric.insert(edge.read_fabric.end(), route.begin(), route.end());
          }
          if (std::find(edge.access.begin(), edge.access.end(), AccessMode::write) != edge.access.end()) {
            auto route = fabric(std::string(rname.str()), *start, *stop);
            edge.write_fabric.insert(edge.write_fabric.end(), route.begin(), route.end());
          }
        }
      result_.graph.access.push_back(std::move(edge));
    }
  }
  void output() {
    for (const auto& id : expansion_order_) {
      const auto& o = objects_.at(id);
      const auto& spec = *o.spec;
      if (o.section == "agents") {
        result_.graph.agents.push_back({id, optional_strings(spec, "capabilities"), {}});
        const auto role = str(spec, "bind");
        check(!role.empty(), "empty discovery role");
        result_.binding_requests.emplace(id, role);
      } else if (o.section == "pools") {
        const auto capacity_text = str(spec, "capacity");
        const auto cap = capacity_text == "discover" ? std::optional<U>{} : std::optional{size_bytes(capacity_text)};
        std::optional<U> granularity;
        if (spec.contains("granularity"))
          granularity = size_bytes(str(spec, "granularity"));
        std::vector<U> pages;
        for (const auto& q : optional_strings(spec, "page_sizes"))
          pages.push_back(size_bytes(q));
        check(!granularity || power2(*granularity), "granularity must be power of two");
        for (auto q : pages)
          check(power2(q), "page size must be power of two");
        std::set<U> unique(pages.begin(), pages.end());
        check(unique.size() == pages.size(), "duplicate page size");
        result_.graph.pools.push_back(
          {id, cap, {}, granularity, pages, memory_service(o, "read"), memory_service(o, "write"), id});
        result_.graph.capacity_groups.push_back({id, cap});
        const auto role = str(spec, "bind");
        check(!role.empty(), "empty discovery role");
        result_.binding_requests.emplace(id, role);
      }
    }
    for (const auto& r : resources_)
      result_.graph.resources.push_back(r.value);
  }
  std::vector<std::string> pool_selection(const toml::node& n) const {
    std::vector<std::string> result;
    for (const auto& token : string_array(n, true))
      for (const auto& id : select(token)) {
        check(objects_.at(id).section == "pools", "transfer endpoints must be pools");
        check(std::find(result.begin(), result.end(), id) == result.end(), "overlapping transfer selectors");
        result.push_back(id);
      }
    check(!result.empty(), "empty transfer pool selection");
    return result;
  }
  std::vector<ResourceDemand> demands(const Object& source, const Object& target, const Object* executor,
                                      const toml::table& spec) const {
    const auto traffic = spec.contains("traffic") ? rates(required(spec, "traffic"), true)
                                                  : std::map<std::string, double>{{"read", 1}, {"write", 1}};
    std::vector<ResourceDemand> result;
    for (const auto& [side, op] : {std::pair{&source, std::string("read")}, std::pair{&target, std::string("write")}}) {
      const auto it = traffic.find(op);
      const auto resource = memory_service(*side, op);
      if (resource && it != traffic.end() && it->second)
        result.push_back({*resource, op == "read" ? DemandKind::read : DemandKind::write, it->second, 0});
    }
    for (const auto& token : optional_strings(spec, "use")) {
      const auto at = token.find('.');
      check(at != std::string::npos, "invalid resource endpoint selector");
      const auto side = token.substr(0, at), name = token.substr(at + 1);
      const Object* endpoint = side == "source"     ? &source
                               : side == "target"   ? &target
                               : side == "executor" ? executor
                                                    : nullptr;
      check(endpoint != nullptr, "resource endpoint missing/invalid: " + token);
      const auto rid = service(*endpoint, name);
      const auto& rs = table(required(section("resources"), name));
      result.push_back(
        {rid, str(rs, "kind", "bandwidth") == "execution_slots" ? DemandKind::occupancy : DemandKind::traffic, 1, 0});
    }
    return result;
  }
  std::string scope_id(const Object& object, const toml::table& spec) const {
    if (!spec.contains("within"))
      return root(object);
    const auto id = object.ancestor(str(spec, "within"));
    check(id.has_value(), "controller scope absent from endpoint");
    return *id;
  }
  Route route(const std::string& name, const Object& source, const Object& target, const Object* executor,
              const toml::table& spec) const {
    std::optional<std::string> controller;
    if (spec.contains("controller")) {
      for (const auto& id : select(str(spec, "controller"))) {
        const auto& object = objects_.at(id);
        check(object.section == "agents", "controller must be agent");
        if (scope_id(object, spec) == scope_id(source, spec)) {
          check(!controller, "ambiguous controller role");
          controller = id;
        }
      }
      check(controller.has_value(), "missing controller role");
    }
    const std::optional<std::string> agent = executor ? std::optional{executor->id} : std::nullopt;
    return {(agent ? *agent : "$request") + ":" + source.id + "->" + target.id + "/" + name,
            agent,
            controller,
            source.id,
            target.id,
            demands(source, target, executor, spec)};
  }
  void transfers() {
    for (const auto& [key, n] : section("transfers")) {
      const std::string name(key.str());
      const auto& spec = table(n);
      const auto by = str(spec, "by");
      const auto sources = pool_selection(required(spec, "from")), targets = pool_selection(required(spec, "to"));
      (void)relation(objects_.at(sources.front()), objects_.at(targets.front()), str(spec, "where", "any"));
      if (spec.contains("traffic"))
        (void)rates(required(spec, "traffic"), true);
      for (const auto& ref : optional_strings(spec, "use")) {
        const auto at = ref.find('.');
        check(at != std::string::npos, "invalid resource endpoint selector");
        const auto side = ref.substr(0, at), resource = ref.substr(at + 1);
        check(side == "source" || side == "target" || (side == "executor" && by != "request.agent"),
              "invalid/unresolved resource endpoint");
        const auto& rs = table(required(section("resources"), resource));
        check(rs.contains("per"), "endpoint resource must use per");
      }
      Method method{name, by == "request.agent", numeric(spec, "bandwidth"), numeric(spec, "startup_us", true), {}, {}};
      if (const auto* bits = spec.get("registration_bits")) {
        method.registration_bits = integer(*bits);
        check(*method.registration_bits > 0, "registration bits must be positive");
      }
      const bool endpoint_executor = by.starts_with("source.") || by.starts_with("target.");
      std::vector<std::string> executors;
      if (by != "request.agent") {
        const auto token = endpoint_executor ? by.substr(by.find('.') + 1) : by;
        executors = select(token);
        for (const auto& id : executors)
          check(objects_.at(id).section == "agents", "executor must be agent");
      }
      if (spec.contains("controller"))
        (void)select(str(spec, "controller"));
      for (const auto& sid : sources)
        for (const auto& tid : targets) {
          const auto& s = objects_.at(sid);
          const auto& t = objects_.at(tid);
          if (!matches(s, t, spec))
            continue;
          if (by == "request.agent")
            method.routes.push_back(route(name, s, t, nullptr, spec));
          else if (endpoint_executor) {
            const auto& endpoint = by.starts_with("source.") ? s : t;
            const Object* executor = nullptr;
            for (const auto& id : executors)
              if (compatible(objects_.at(id), endpoint)) {
                check(executor == nullptr, "ambiguous endpoint executor role");
                executor = &objects_.at(id);
              }
            check(executor, "missing endpoint executor role");
            method.routes.push_back(route(name, s, t, executor, spec));
          } else
            for (const auto& id : executors)
              if (compatible(objects_.at(id), s))
                method.routes.push_back(route(name, s, t, &objects_.at(id), spec));
        }
      result_.graph.methods.push_back(std::move(method));
    }
  }
  void domains() {
    for (const auto& [key, n] : section("domains")) {
      const auto& spec = table(n);
      const auto per = str(spec, "per");
      const auto members = select(str(spec, "members"));
      for (const auto& id : members)
        check(objects_.at(id).section == "pools", "domain members must be pools");
      for (const auto& owner : select(per)) {
        AllocationDomain domain{owner + "/" + std::string(key.str()), std::string(key.str()), {}};
        const auto prototype = objects_.at(owner).prototype;
        for (const auto& id : members)
          if (objects_.at(id).ancestor(prototype) == owner)
            domain.members.push_back(id);
        check(!domain.members.empty(), "empty allocation domain");
        result_.graph.domains.push_back(std::move(domain));
      }
    }
  }
};
} // namespace
Result<U> parse_memory_size(std::string_view text) {
  return checked([&] { return size_bytes(text); });
}
Result<HardwareExpansion> expand_hardware_template(std::string_view text, const RuntimeCounts& counts) {
  return checked([&] { return Compiler(toml::parse(text), counts).run(); });
}
Result<CalibratedHardwareExpansion> expand_calibrated_hardware_template(std::string_view hardware,
  std::string_view calibration, const RuntimeCounts& counts) {
  auto expanded = checked([&] {
    Compiler compiler(toml::parse(hardware), counts);
    auto expansion = compiler.run();
    auto parsed = parse_calibration(calibration, expansion.graph, compiler.calibration_selectors());
    return std::pair{std::move(expansion), std::move(parsed)};
  });
  if (!expanded) return {std::get<Error>(expanded.state)};
  auto [hardware_expansion, parsed] = std::move(expanded).value();
  if (!parsed) return {std::get<Error>(parsed.state)};
  return {CalibratedHardwareExpansion{std::move(hardware_expansion), std::move(parsed).value()}};
}
Result<ResourceGraph> bind_hardware_inventory(const HardwareExpansion& expanded, const HardwareInventory& inventory) {
  return checked([&] {
    for (const auto& [key, expected] : expanded.requirements) {
      const auto found = inventory.facts.find(key);
      check(found != inventory.facts.end() && found->second == expected, "incompatible/missing hardware fact: " + key);
    }
    check(inventory.bindings.size() == expanded.binding_requests.size(),
          "discovery bindings do not exactly cover logical roles");
    for (const auto& [key, role] : expanded.binding_requests) {
      check(!role.empty(), "empty discovery role");
      check(inventory.bindings.contains(key), "missing discovery binding: " + key);
    }
    ResourceGraph graph = expanded.graph;
    for (auto& agent : graph.agents) {
      const auto& info = inventory.bindings.at(agent.id);
      check(!info.capacity_bytes && !info.capacity_group && !info.capacity_group_bytes && !info.granularity_bytes &&
              info.page_sizes_bytes.empty(),
            "agent binding contains pool properties");
      if (info.address_space)
        check(!info.address_space->empty(), "empty agent address space");
      agent.address_space = info.address_space;
      if (info.capabilities) {
        std::set<std::string> capabilities(info.capabilities->begin(), info.capabilities->end());
        check(capabilities.size() == info.capabilities->size() && !capabilities.contains(""),
              "invalid discovered capabilities");
        for (const auto& required : agent.capabilities)
          check(capabilities.contains(required), "required agent capability unavailable");
        agent.capabilities = *info.capabilities;
      }
    }
    std::map<std::string, U> groups;
    for (auto& pool : graph.pools) {
      const auto& info = inventory.bindings.at(pool.id);
      check(!info.capabilities, "pool binding contains agent capabilities");
      check(info.capacity_bytes && *info.capacity_bytes > 0 &&
              (!pool.capacity_bytes || *info.capacity_bytes >= *pool.capacity_bytes),
            "invalid/insufficient discovered capacity");
      if (!pool.capacity_bytes)
        pool.capacity_bytes = info.capacity_bytes;
      check(info.address_space && !info.address_space->empty(), "pool needs discovered address space");
      pool.address_space = info.address_space;
      if (info.granularity_bytes) {
        check(!pool.granularity_bytes || pool.granularity_bytes == info.granularity_bytes,
              "allocation granularity mismatch");
        pool.granularity_bytes = info.granularity_bytes;
      }
      if (!info.page_sizes_bytes.empty()) {
        auto expected = pool.page_sizes_bytes, actual = info.page_sizes_bytes;
        std::sort(expected.begin(), expected.end());
        std::sort(actual.begin(), actual.end());
        check(expected.empty() || expected == actual, "page-size list mismatch");
        pool.page_sizes_bytes = std::move(actual);
      }
      check(!pool.page_sizes_bytes.empty() && pool.granularity_bytes,
            "pool needs discovered pages and allocation granularity");
      check(power2(*pool.granularity_bytes), "invalid allocation granularity");
      U quantum = *pool.granularity_bytes;
      std::set<U> pages;
      for (auto q : pool.page_sizes_bytes) {
        check(power2(q) && pages.insert(q).second, "invalid/duplicate page size");
        quantum = std::max(quantum, q);
      }
      pool.granularity_bytes = quantum;
      pool.capacity_group = info.capacity_group.value_or(pool.id);
      check(!pool.capacity_group.empty(), "empty capacity group");
      const U cap = info.capacity_group_bytes.value_or(*pool.capacity_bytes);
      check(cap >= *pool.capacity_bytes, "capacity group is smaller than pool");
      const auto [it, inserted] = groups.emplace(pool.capacity_group, cap);
      check(inserted || it->second == cap, "inconsistent shared capacity aliases");
    }
    graph.capacity_groups.clear();
    for (const auto& [id, capacity] : groups)
      graph.capacity_groups.push_back({id, capacity});
    std::set<std::string> access_ids;
    for (auto& access : graph.access) {
      access_ids.insert(access.id);
      if (const auto found = inventory.access.find(access.id); found != inventory.access.end()) {
        if (found->second.mapping) {
          switch (*found->second.mapping) {
          case MappingMode::unknown:
          case MappingMode::none:
          case MappingMode::direct:
          case MappingMode::requires_mapping:
            break;
          default:
            throw std::invalid_argument("invalid discovered mapping enum");
          }
          access.mapping = *found->second.mapping;
        }
        if (found->second.prerequisites) {
          std::set<std::string> keys(found->second.prerequisites->begin(), found->second.prerequisites->end());
          check(!keys.contains("") && keys.size() == found->second.prerequisites->size(),
                "invalid discovered prerequisites");
          access.prerequisites = *found->second.prerequisites;
        }
      }
    }
    for (const auto& [id, info] : inventory.access) {
      (void)info;
      check(access_ids.contains(id), "unknown access discovery target");
    }
    return graph;
  });
}
Result<ResourceGraph> compile_hardware_template(std::string_view text, const RuntimeCounts& counts,
                                                const HardwareInventory& inventory) {
  auto expanded = expand_hardware_template(text, counts);
  if (!expanded)
    return {std::get<Error>(expanded.state)};
  return bind_hardware_inventory(expanded.value(), inventory);
}
} // namespace vecops::memory::model
