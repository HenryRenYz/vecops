// SPDX-FileCopyrightText: 2026 HenryRenYz and vecops contributors
// SPDX-License-Identifier: MIT
#include "vecops/memory/RuntimePolicy.h"
#include "details/RuntimeInternal.h"
#include "../../third_party/tomlplusplus/toml.hpp"
#include <cmath>
#include <fstream>
#include <regex>
#include <sstream>

namespace vecops::memory::model {
namespace {
using namespace details;
void check(bool yes, const std::string& message) {
  if (!yes)
    throw Failure{Error{"policy.schema", message}};
}
template <class F>
auto checked(F&& fn) -> Result<decltype(fn())> {
  return protect([&]() -> decltype(fn()) {
    try {
      return fn();
    } catch (const toml::parse_error& e) {
      std::ostringstream message;
      message << e;
      throw Failure{Error{"policy.toml", message.str()}};
    }
  });
}
const toml::node& required(const toml::table& t, std::string_view key) {
  const auto* n = t.get(key);
  check(n != nullptr, "missing field: " + std::string(key));
  return *n;
}
const toml::table& table(const toml::node& n) {
  check(n.is_table(), "expected table");
  return *n.as_table();
}
std::string string(const toml::node& n) {
  check(n.is_string(), "expected string");
  return n.as_string()->get();
}
void fields(const toml::table& t, std::initializer_list<std::string_view> allowed) {
  for (const auto& [key, value] : t) {
    (void)value;
    check(std::find(allowed.begin(), allowed.end(), key.str()) != allowed.end(),
          "unknown field: " + std::string(key.str()));
  }
}
U size(const toml::node& n, bool zero = true) {
  const auto text = string(n);
  static const std::regex zeros(R"(^0+(?:\.0+)?\s*(GiB|MiB|KiB|B)$)");
  if (zero && std::regex_match(text, zeros))
    return 0;
  return take(parse_memory_size(text));
}
double fraction(const toml::node& n) {
  check(n.is_floating_point() || n.is_integer(), "expected numeric fraction");
  const double x = n.is_integer() ? static_cast<double>(n.as_integer()->get()) : n.as_floating_point()->get();
  check(std::isfinite(x) && x >= 0 && x <= 1, "fraction must be finite and in [0, 1]");
  return x;
}
std::variant<U, double> pair(const toml::table& t, const char* bytes, const char* ratio) {
  const auto* b = t.get(bytes);
  const auto* r = t.get(ratio);
  check(bool(b) != bool(r), "exactly one of " + std::string(bytes) + " / " + ratio + " is required");
  return b ? std::variant<U, double>{size(*b)} : std::variant<U, double>{fraction(*r)};
}
U resolve(const std::variant<U, double>& v, U base) {
  if (const auto* n = std::get_if<U>(&v))
    return *n;
  const double f = std::get<double>(v);
  check(std::isfinite(f) && f >= 0 && f <= 1, "fraction must be finite and in [0, 1]");
  // Limit values at one explicitly: rounding UINT64_MAX through a floating-point
  // type must not produce an out-of-range uint64 conversion on other platforms.
  if (f == 1)
    return base;
  return static_cast<U>(static_cast<long double>(base) * f);
}
std::string read_file(const std::string& path) {
  std::ifstream stream(path);
  check(bool(stream), "cannot open TOML file: " + path);
  std::stringstream content;
  content << stream.rdbuf();
  check(!stream.bad(), "cannot read TOML file: " + path);
  return content.str();
}
RuntimeCounts local_counts(std::string_view hardware, RuntimeCounts explicit_counts) {
  const auto t = toml::parse(hardware);
  const auto* groups = t.get_as<toml::table>("groups");
  if (groups)
    for (const auto& [name, value] : *groups)
      if (const auto* group = value.as_table())
        if (!group->contains("per")) {
          const auto* count = group->get("count");
          if (count && count->is_string() && count->as_string()->get() == "runtime")
            explicit_counts.try_emplace(std::string(name.str()), 1);
        }
  return explicit_counts;
}
LinuxDiscovery discover(const std::optional<std::string>& file, LinuxCpuProfile profile, LinuxRuntimeOptions& options,
                        bool infer_profile, const std::optional<std::string>& calibration = std::nullopt) {
  std::optional<HardwareExpansion> expansion;
  if (file) {
    const auto text = read_file(*file);
    if (calibration) {
      check(!options.calibration, "calibration supplied both as file and compiled options");
      auto compiled = take(expand_calibrated_hardware_template(text, *calibration,
                                                              local_counts(text, options.runtime_counts)));
      expansion = std::move(compiled.hardware);
      options.calibration = std::move(compiled.calibration);
    } else
      expansion = take(expand_hardware_template(text, local_counts(text, options.runtime_counts)));
    if (infer_profile) {
      const auto model = expansion->requirements.find("cpu_model");
      if (model != expansion->requirements.end())
        if (const auto* name = std::get_if<std::string>(&model->second); name && *name == "LX2")
          profile = LinuxCpuProfile::lx2_flat;
    }
  } else
    check(options.runtime_counts.empty(), "runtime counts require a hardware template");
  auto result = take(discover_linux_cpu(take(probe_linux_topology()), expansion ? &*expansion : nullptr, profile));
  // These names are Linux backend bindings, independent of the hardware template
  // parser. An arbitrary requester-executed method may be a network method and
  // must never be inferred to mean memcpy.
  if (options.cpu_copy_methods.empty())
    if (std::any_of(result.graph.methods.begin(), result.graph.methods.end(),
                    [](const auto& m) { return m.id == "cpu_copy"; }))
      options.cpu_copy_methods.push_back("cpu_copy");
  if (options.cpu_queues.empty())
    for (const auto& agent : result.graph.agents)
      if (has(result.scope.executable_agents, agent.id) && has(agent.capabilities, std::string("host_load_store")))
        options.cpu_queues.emplace(agent.id, agent.id);
  return result;
}
} // namespace
Result<RuntimePolicy> parse_runtime_policy(std::string_view text) {
  return checked([&] {
    const auto root = toml::parse(text);
    fields(root, {"schema", "arena"});
    check(string(required(root, "schema")) == "memory-runtime/0.1", "unsupported runtime policy schema");
    const auto& a = table(required(root, "arena"));
    fields(a, {"domain_rule", "hard_limit_bytes", "hard_limit_fraction", "preallocate_bytes",
               "preallocate_fraction", "grow_by", "retain_idle", "page_size", "backing", "allow_growth"});
    RuntimeArenaPolicy p;
    p.domain_rule = string(required(a, "domain_rule"));
    check(!p.domain_rule.empty(), "domain_rule cannot be empty");
    p.hard_limit = {pair(a, "hard_limit_bytes", "hard_limit_fraction")};
    p.initial = {pair(a, "preallocate_bytes", "preallocate_fraction")};
    p.grow_by_bytes = size(required(a, "grow_by"));
    p.retain_idle_bytes = size(required(a, "retain_idle"));
    p.page_bytes = size(required(a, "page_size"), false);
    check(power2(p.page_bytes), "page_size must be a power of two");
    const auto mode = string(required(a, "backing"));
    check(mode == "private_memory" || mode == "shareable_remappable" || mode == "private_movable", "unknown backing mode");
    p.backing = mode == "private_memory" ? BackingMode::private_memory : mode == "private_movable" ? BackingMode::private_movable : BackingMode::shareable_remappable;
    const auto& grow = required(a, "allow_growth");
    check(grow.is_boolean(), "allow_growth must be boolean");
    p.allow_growth = grow.as_boolean()->get();
    return RuntimePolicy{std::move(p)};
  });
}
Result<std::vector<ArenaOptions>> apply_runtime_policy(const LinuxDiscovery& discovery, LinuxRuntimeOptions& options,
                                                       const RuntimePolicy& policy) {
  return checked([&] {
    const auto& rule = policy.arena;
    check(!rule.domain_rule.empty(), "domain_rule cannot be empty");
    check(power2(rule.page_bytes), "page_size must be a power of two");
    check(rule.backing == BackingMode::private_memory || rule.backing == BackingMode::shareable_remappable || rule.backing == BackingMode::private_movable,
          "unknown backing mode");
    std::vector<ArenaOptions> arenas;
    auto constraints = options.shared_limits;
    std::set<std::string> ids;
    for (const auto& c : constraints)
      check(ids.insert(c.id).second, "duplicate existing budget constraint");
    for (const auto& domain : discovery.graph.domains) {
      if (domain.rule != rule.domain_rule)
        continue;
      std::vector<std::string> pools, groups, local_groups;
      U capacity = 0, local_capacity = 0;
      // The physical domain cap is independent of a participant's scope. A
      // restricted process sees fewer allocation candidates, not another copy
      // of the shared domain budget or a different authority configuration.
      for (const auto& id : domain.members) {
        const auto& p = find(discovery.graph.pools, id);
        if (!has(groups, p.capacity_group)) {
          const auto& group = find(discovery.graph.capacity_groups, p.capacity_group);
          check(group.capacity_bytes.has_value(), "policy needs discovered unique capacity");
          capacity = add(capacity, *group.capacity_bytes);
          groups.push_back(group.id);
        }
        if (has(discovery.scope.allocatable_pools, id)) {
          check(rule.page_bytes >= quantum(p), "policy page size below pool quantum");
          pools.push_back(id);
          if (!has(local_groups,p.capacity_group)) {
            local_capacity=add(local_capacity,*find(discovery.graph.capacity_groups,p.capacity_group).capacity_bytes);
            local_groups.push_back(p.capacity_group);
          }
        }
      }
      const U hard = resolve(rule.hard_limit.bytes_or_capacity_fraction, capacity);
      check(hard <= capacity, "domain hard limit exceeds managed capacity");
      const U domain_hard = hard / rule.page_bytes * rule.page_bytes;
      const auto id = "runtime-policy/" + domain.id;
      check(ids.insert(id).second, "policy budget constraint identity collision");
      constraints.push_back({id, groups, domain_hard});
      if (pools.empty()) continue;
      const U rounded_hard = std::min(domain_hard,local_capacity / rule.page_bytes * rule.page_bytes);
      U initial = resolve(rule.initial.bytes_or_limit_fraction, rounded_hard);
      check(initial <= rounded_hard, "preallocation exceeds domain hard limit");
      if (initial) initial = page_charge(initial,rule.page_bytes);
      check(initial <= rounded_hard,"rounded preallocation exceeds domain hard limit");
      arenas.push_back({domain.id, std::move(pools), Limit{rounded_hard}, Preallocate{initial}, rule.grow_by_bytes,
                        rule.retain_idle_bytes, rule.page_bytes, rule.backing, rule.allow_growth});
    }
    check(!arenas.empty(), "domain_rule matches no allocatable domains");
    options.shared_limits = std::move(constraints);
    return arenas;
  });
}
Result<std::unique_ptr<MemoryRuntime>> make_policy_linux_runtime(LinuxDiscovery discovery, const RuntimePolicy& policy,
                                                                 LinuxRuntimeOptions options) {
  return checked([&] {
    auto arenas = take(apply_runtime_policy(discovery, options, policy));
    auto runtime = take(make_linux_runtime(std::move(discovery), std::move(options)));
    auto view = take(runtime->view(take(runtime->current_scope())));
    for (const auto& arena : arenas)
      take(runtime->create_arena(view, arena));
    return runtime;
  });
}
Result<std::unique_ptr<MemoryRuntime>> discover_linux_runtime(const std::optional<std::string>& hardware_file,
                                                              LinuxCpuProfile profile, LinuxRuntimeOptions options) {
  return checked([&] {
    auto d = discover(hardware_file, profile, options, false);
    return take(make_linux_runtime(std::move(d), std::move(options)));
  });
}
Result<std::unique_ptr<MemoryRuntime>> open_runtime(const std::string& hardware_file,
                                                    const std::optional<std::string>& policy_file,
                                                    const std::optional<std::string>& calibration_file) {
  return open_linux_runtime(hardware_file,policy_file,calibration_file,{});
}
Result<std::unique_ptr<MemoryRuntime>> open_linux_runtime(const std::string& hardware_file,
  const std::optional<std::string>& policy_file, const std::optional<std::string>& calibration_file,
  LinuxRuntimeOptions options) {
  return checked([&] {
    check(!hardware_file.empty(), "hardware TOML file path cannot be empty");
    std::optional<RuntimePolicy> policy;
    if (policy_file)
      policy = take(parse_runtime_policy(read_file(*policy_file)));
    const auto calibration = calibration_file ? std::optional<std::string>{read_file(*calibration_file)} : std::nullopt;
    auto d = discover(hardware_file, LinuxCpuProfile::generic_cpu, options, true, calibration);
    return policy ? take(make_policy_linux_runtime(std::move(d), *policy, std::move(options)))
                  : take(make_linux_runtime(std::move(d), std::move(options)));
  });
}
} // namespace vecops::memory::model
