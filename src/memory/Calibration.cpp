// SPDX-FileCopyrightText: 2026 HenryRenYz and vecops contributors
// SPDX-License-Identifier: MIT
#include "vecops/memory/Calibration.h"
#include "../../third_party/tomlplusplus/toml.hpp"
#include <algorithm>
#include <cmath>
#include <limits>
#include <set>
#include <sstream>
#include <stdexcept>
#include <tuple>

namespace vecops::memory::model {
namespace {
using U = std::uint64_t;
struct Failure {
  Error error;
};
void check(bool yes, const std::string& message) {
  if (!yes)
    throw Failure{Error{"calibration.schema", message}};
}
[[noreturn]] void unsupported(const std::string& message) {
  throw Failure{Error{"calibration.unsupported", message}};
}
template <class F>
auto checked(F&& f) -> Result<decltype(f())> {
  try {
    return {f()};
  } catch (const Failure& e) {
    return {e.error};
  } catch (const toml::parse_error& e) {
    std::ostringstream out;
    out << e;
    return {Error{"calibration.toml", out.str()}};
  } catch (const std::exception& e) {
    return {Error{"calibration.schema", e.what()}};
  }
}
template <class T>
T take(Result<T> r) {
  if (!r)
    throw Failure{std::get<Error>(r.state)};
  return std::move(r).value();
}
const toml::node& required(const toml::table& t, std::string_view key) {
  const auto* n = t.get(key);
  check(n, "missing field: " + std::string(key));
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
std::string str(const toml::table& t, std::string_view key) {
  return string(required(t, key));
}
void fields(const toml::table& t, std::initializer_list<std::string_view> allowed) {
  for (const auto& [k, v] : t) {
    (void)v;
    check(std::find(allowed.begin(), allowed.end(), k.str()) != allowed.end(),
          "unknown field: " + std::string(k.str()));
  }
}
double number(const toml::node& n) {
  check(n.is_integer() || n.is_floating_point(), "expected positive number");
  const auto x = n.is_integer() ? static_cast<double>(n.as_integer()->get()) : n.as_floating_point()->get();
  check(std::isfinite(x) && x > 0, "number must be finite and positive");
  return x;
}
U integer(const toml::node& n) {
  check(n.is_integer() && n.as_integer()->get() > 0, "expected positive integer");
  return static_cast<U>(n.as_integer()->get());
}
const toml::array& array(const toml::node& n) {
  check(n.is_array(), "expected array");
  return *n.as_array();
}
std::any plain(const toml::node& n) {
  if (n.is_table()) {
    std::map<std::string, std::any> out;
    for (const auto& [k, v] : *n.as_table())
      out.emplace(k.str(), plain(v));
    return out;
  }
  if (n.is_array()) {
    std::vector<std::any> out;
    for (const auto& v : *n.as_array())
      out.push_back(plain(v));
    return out;
  }
  if (n.is_string())
    return n.as_string()->get();
  if (n.is_integer())
    return n.as_integer()->get();
  if (n.is_floating_point())
    return n.as_floating_point()->get();
  if (n.is_boolean())
    return n.as_boolean()->get();
  throw Failure{Error{"calibration.schema", "unsupported TOML value"}};
}
struct Provenance {
  CostQuality quality = CostQuality::reference;
  std::string source, implementation;
};
Provenance provenance(const toml::table& p) {
  Provenance out;
  if (const auto* n = p.get("provenance")) {
    const auto& t = table(*n);
    fields(t, {"quality", "source", "implementation"});
    if (t.contains("source"))
      out.source = str(t, "source");
    if (t.contains("implementation"))
      out.implementation = str(t, "implementation");
    if (t.contains("quality")) {
      const auto q = str(t, "quality");
      check(q == "reference" || q == "calibrated", "provenance quality must be reference or calibrated");
      if (q == "calibrated") {
        check(!out.source.empty() && !out.implementation.empty(),
              "calibrated provenance requires source and implementation");
        out.quality = CostQuality::calibrated;
      }
    }
  }
  return out;
}
struct Curve {
  enum Axis { constant, cores, bytes } axis = constant;
  std::vector<U> x;
  std::vector<double> bandwidth;
  Provenance origin;
};
Curve curve(const toml::table& t, Provenance origin) {
  fields(t, {"cpu_cores", "size", "bandwidth"});
  Curve out;
  out.origin = std::move(origin);
  check(!(t.contains("cpu_cores") && t.contains("size")), "two-dimensional curves are unsupported");
  const auto* axis = t.get("cpu_cores");
  if (axis)
    out.axis = Curve::cores;
  else if ((axis = t.get("size")))
    out.axis = Curve::bytes;
  check(axis, "curve requires cpu_cores or size axis");
  for (const auto& n : array(*axis))
    out.x.push_back(out.axis == Curve::cores ? integer(n) : take(parse_memory_size(string(n))));
  for (const auto& n : array(required(t, "bandwidth")))
    out.bandwidth.push_back(number(n));
  check(!out.x.empty() && out.x.size() == out.bandwidth.size(), "curve axis/bandwidth lengths differ or are empty");
  check(std::adjacent_find(out.x.begin(), out.x.end(), std::greater_equal<U>{}) == out.x.end(),
        "curve axis must be strictly increasing");
  return out;
}
struct Sample {
  std::string method, executor, source, target;
  Curve value;
};
struct Profile {
  std::map<std::string, Curve> access;
  std::optional<Curve> transfer;
  std::vector<Sample> samples;
  std::optional<double> write_cost;
  Provenance origin;
};
struct Selected {
  double bandwidth;
  bool interpolated;
};
std::optional<Selected> select(const Curve& c, U bytes, const CalibrationControl* control) {
  if (!c.origin.implementation.empty() && (!control || control->implementation != c.origin.implementation))
    return {};
  if (c.axis == Curve::constant)
    return Selected{c.bandwidth.front(), false};
  U x = bytes;
  if (c.axis == Curve::cores) {
    if (!control || !control->cpu_cores)
      return {};
    x = *control->cpu_cores;
  }
  const auto hi = std::lower_bound(c.x.begin(), c.x.end(), x);
  if (hi == c.x.end())
    return {};
  const auto i = static_cast<std::size_t>(hi - c.x.begin());
  if (*hi == x)
    return Selected{c.bandwidth[i], false};
  // Core-count curves require an exact control. Size curves interpolate service
  // time, rather than averaging bandwidth. Neither axis extrapolates.
  if (hi == c.x.begin() || c.axis == Curve::cores)
    return {};
  const long double t0 = static_cast<long double>(c.x[i - 1]) / c.bandwidth[i - 1];
  const long double t1 = static_cast<long double>(c.x[i]) / c.bandwidth[i];
  const long double a = static_cast<long double>(x - c.x[i - 1]) / (c.x[i] - c.x[i - 1]);
  const double b = static_cast<double>(x / (t0 + a * (t1 - t0)));
  check(std::isfinite(b) && b > 0, "curve interpolation overflow");
  return Selected{b, true};
}
const CalibrationControl* control(const CalibrationControls& controls, const std::optional<std::string>& id) {
  if (!id)
    return nullptr;
  const auto i = controls.find(*id);
  return i == controls.end() ? nullptr : &i->second;
}
template <class T>
const T* find(const std::vector<T>& v, const std::string& id) {
  const auto i = std::find_if(v.begin(), v.end(), [&](const auto& x) { return x.id == id; });
  return i == v.end() ? nullptr : &*i;
}
std::pair<std::string, std::string> endpoint(const std::string& token) {
  const auto at = token.find(':');
  auto a = token.substr(0, at), b = at == std::string::npos ? "any" : token.substr(at + 1);
  check(!a.empty() && !b.empty(), "empty endpoint selector/predicate");
  return {a, b};
}
bool predicate(const CalibrationSelectors& selectors, const std::string& agent, const std::string& pool,
               const std::string& tests) {
  bool out = true;
  std::size_t start = 0;
  do {
    const auto end = tests.find('&', start);
    auto term = tests.substr(start, end == std::string::npos ? std::string::npos : end - start);
    check(!term.empty(), "empty predicate term");
    bool negate = term.front() == '!';
    if (negate)
      term.erase(0, 1);
    check(!term.empty(), "empty negated predicate");
    // Evaluate every term even if another term was false, to reject misspelled
    // ancestors independently of concrete topology/order.
    bool value = selectors.relation(agent, pool, term);
    out = (negate ? !value : value) && out;
    if (end == std::string::npos)
      break;
    start = end + 1;
  } while (true);
  return out;
}
} // namespace

class Calibration {
public:
  ResourceGraph expected_graph; // topology only; discovered capacity/address bindings may differ
  PerformanceDatabase database;
  std::map<std::string, Profile> profiles;
  std::map<std::pair<ModelTargetKind, std::string>, std::string> targets;
};
const PerformanceDatabase& calibration_database(const Calibration& c) {
  return c.database;
}

Result<std::shared_ptr<const Calibration>> parse_calibration(std::string_view text, const ResourceGraph& graph,
                                                             const CalibrationSelectors& selectors) {
  return checked([&]() -> std::shared_ptr<const Calibration> {
    const auto root = toml::parse(text);
    fields(root, {"schema", "name", "units", "models", "profiles", "methods"});
    check(str(root, "schema") == "memory-calibration/0.4", "unsupported calibration schema");
    check(!str(root, "name").empty(), "empty calibration name");
    const auto& units = table(required(root, "units"));
    fields(units, {"bandwidth"});
    check(str(units, "bandwidth") == "GiB/s", "bandwidth unit must be GiB/s");
    if (const auto* m = root.get("methods")) {
      check(m->is_table(), "methods must be a table");
      if (!m->as_table()->empty())
        unsupported("compute/fallback method models are not implemented; use explicit transfer bindings");
    }
    auto out = std::make_shared<Calibration>();
    out->expected_graph = graph;
    for (const auto& [key, value] : table(required(root, "profiles"))) {
      const std::string name(key.str());
      check(!name.empty(), "empty profile name");
      const auto& p = table(value);
      fields(p, {"read", "write", "size", "bandwidth", "samples", "write_cost", "provenance"});
      Profile profile;
      profile.origin = provenance(p);
      for (const auto op : {"read", "write"})
        if (const auto* n = p.get(op))
          profile.access.emplace(op, curve(table(*n), profile.origin));
      if (p.contains("size") || p.contains("bandwidth")) {
        toml::table t;
        t.insert("size", required(p, "size"));
        t.insert("bandwidth", required(p, "bandwidth"));
        profile.transfer = curve(t, profile.origin);
      }
      if (const auto* n = p.get("write_cost")) {
        const auto& t = table(*n);
        fields(t, {"copy"});
        profile.write_cost = number(required(t, "copy"));
      }
      if (const auto* n = p.get("samples"))
        for (const auto& item : array(*n)) {
          const auto& t = table(item);
          fields(t, {"method", "executor", "source", "target", "operation", "cpu_cores", "size", "bandwidth"});
          if (str(t, "operation") != "copy")
            unsupported("only operation=copy transfer samples are supported");
          Sample s{str(t, "method"), str(t, "executor"), str(t, "source"), str(t, "target"), {}};
          s.value.origin = profile.origin;
          check(!(t.contains("size") && t.contains("cpu_cores")), "two-dimensional samples are unsupported");
          if (const auto* cores = t.get("cpu_cores")) {
            s.value.axis = Curve::cores;
            s.value.x = {integer(*cores)};
          }
          if (const auto* bytes = t.get("size")) {
            s.value.axis = Curve::bytes;
            s.value.x = {take(parse_memory_size(string(*bytes)))};
          }
          s.value.bandwidth = {number(required(t, "bandwidth"))};
          profile.samples.push_back(std::move(s));
        }
      const int kinds =
        (!profile.access.empty()) + bool(profile.transfer || !profile.samples.empty()) + bool(profile.write_cost);
      check(kinds == 1 && !(profile.transfer && !profile.samples.empty()),
            "profile must have exactly one supported purpose");
      out->profiles.emplace(name, std::move(profile));
      auto public_profile = std::any_cast<std::map<std::string, std::any>>(plain(value));
      if (auto samples = public_profile.find("samples"); samples != public_profile.end())
        for (auto& item : std::any_cast<std::vector<std::any>&>(samples->second)) {
          auto& point = std::any_cast<std::map<std::string, std::any>&>(item);
          for (const auto key : {"method", "executor", "source", "target"})
            point.erase(key); // concrete case IDs replace all temporary selectors
        }
      out->database.profiles.emplace(name, std::move(public_profile));
    }
    std::set<std::string> used;
    for (const auto& [key, value] : table(required(root, "models"))) {
      const std::string model(key.str()), profile = string(value);
      check(out->profiles.contains(profile), "unknown profile: " + profile);
      const auto colon = model.find(':');
      check(colon != std::string::npos, "model key requires target:rule");
      const auto prefix = model.substr(0, colon), rule = model.substr(colon + 1);
      check(!rule.empty(), "empty target rule");
      std::vector<std::string> targets;
      ModelTargetKind kind;
      const auto& p = out->profiles.at(profile);
      if (prefix == "access") {
        kind = ModelTargetKind::access;
        check(!p.access.empty(), "access binding requires read/write profile");
        for (const auto& x : graph.access)
          if (x.rule == rule)
            targets.push_back(x.id);
      } else if (prefix == "resource") {
        kind = ModelTargetKind::resource;
        check(p.write_cost.has_value(), "resource binding requires copy write_cost");
        for (const auto& x : graph.resources)
          if (x.rule == rule) {
            check(x.kind == ResourceKind::bandwidth, "write_cost requires bandwidth resource");
            targets.push_back(x.id);
          }
      } else if (prefix == "transfer") {
        kind = ModelTargetKind::method;
        check(p.transfer || !p.samples.empty(), "transfer binding requires transfer profile");
        if (find(graph.methods, rule))
          targets.push_back(rule);
      } else {
        check(false, "unknown model target: " + prefix);
        kind = ModelTargetKind::access;
      }
      check(!targets.empty(), "model target matches nothing: " + model);
      used.insert(profile);
      for (const auto& target : targets) {
        check(out->targets.emplace(std::make_pair(kind, target), profile).second, "duplicate model binding");
        out->database.bindings.push_back({kind, target, profile});
      }
    }
    check(used.size() == out->profiles.size(), "unbound profile");
    std::map<std::tuple<std::string, std::string, std::string>, std::vector<std::pair<Curve::Axis, U>>> unique;
    for (const auto& [name, p] : out->profiles)
      for (std::size_t i = 0; i < p.samples.size(); ++i) {
        check(bool(selectors.select) && bool(selectors.relation),
              "sample predicates require hardware template relations");
        const auto& sample = p.samples[i];
        const auto* method = find(graph.methods, sample.method);
        check(method, "sample names unknown method");
        const auto binding = out->targets.find({ModelTargetKind::method, sample.method});
        check(binding != out->targets.end() && binding->second == name, "sample method not bound to this profile");
        const auto [source, source_test] = endpoint(sample.source);
        const auto [target, target_test] = endpoint(sample.target);
        const auto sources = selectors.select(source), targets = selectors.select(target),
                   agents = selectors.select(sample.executor);
        check(!sources.empty() && !targets.empty() && !agents.empty(), "empty sample selector");
        for (const auto& id : sources)
          check(find(graph.pools, id), "source selector must select pools");
        for (const auto& id : targets)
          check(find(graph.pools, id), "target selector must select pools");
        bool matched = false;
        for (const auto& agent : agents) {
          check(find(graph.agents, agent), "executor selector must select agents");
          std::set<std::string> ss, tt;
          for (const auto& id : sources)
            if (predicate(selectors, agent, id, source_test))
              ss.insert(id);
          for (const auto& id : targets)
            if (predicate(selectors, agent, id, target_test))
              tt.insert(id);
          for (const auto& r : method->routes) {
            const auto bound = method->request_executor ? std::optional<std::string>{}
                               : r.controller           ? r.controller
                                                        : r.agent;
            if (bound && *bound != agent)
              continue;
            if (!ss.contains(r.source) || !tt.contains(r.target))
              continue;
            matched = true;
            const U x = sample.value.x.empty() ? 0 : sample.value.x.front();
            auto& existing = unique[{method->id, r.id, agent}];
            for (const auto& [axis, point] : existing)
              check(axis == sample.value.axis && axis != Curve::constant && point != x,
                    "ambiguous overlapping calibration cases");
            existing.emplace_back(sample.value.axis, x);
            out->database.cases.push_back({method->id, r.id, agent, "copy", name, static_cast<U>(i)});
          }
        }
        check(matched, "sample matches no legal route");
      }
    std::sort(out->database.bindings.begin(), out->database.bindings.end(), [](const auto& a, const auto& b) {
      return std::tie(a.target_kind, a.target, a.profile) < std::tie(b.target_kind, b.target, b.profile);
    });
    std::sort(out->database.cases.begin(), out->database.cases.end(), [](const auto& a, const auto& b) {
      return std::tie(a.method, a.route, a.executor, a.profile, a.sample) <
             std::tie(b.method, b.route, b.executor, b.profile, b.sample);
    });
    return out;
  });
}

namespace {
const Profile* bound(const Calibration& c, ModelTargetKind kind, const std::string& id, std::string* name = nullptr) {
  const auto i = c.targets.find({kind, id});
  if (i == c.targets.end())
    return nullptr;
  if (name)
    *name = i->second;
  return &c.profiles.at(i->second);
}
std::optional<Selected> access_rate(const Calibration& c, const AccessConnection& edge, const std::string& op, U bytes,
                                    const CalibrationControls& controls, CalibrationUse* use = nullptr) {
  std::string name;
  const auto* p = bound(c, ModelTargetKind::access, edge.id, &name);
  if (!p)
    return {};
  const auto i = p->access.find(op);
  if (i == p->access.end())
    return {};
  auto result = select(i->second, bytes, control(controls, edge.agent));
  if (result && use)
    *use = {edge.id,          op, name, i->second.origin.source, i->second.origin.quality, result->interpolated,
            result->bandwidth};
  return result;
}
std::optional<double> transfer_rate(const Calibration& c, const Method& m, const Route* route,
                                    const std::optional<std::string>& executor, U bytes,
                                    const CalibrationControls& controls) {
  const auto* p = bound(c, ModelTargetKind::method, m.id);
  if (!p)
    return {};
  const auto selected_executor = m.request_executor           ? executor
                                 : route && route->controller ? route->controller
                                 : route && route->agent      ? route->agent
                                                              : executor;
  const auto* actual = control(controls, selected_executor);
  if (p->transfer) {
    const auto value = select(*p->transfer, bytes, actual);
    if (value)
      return value->bandwidth;
  }
  std::optional<double> out;
  if (route)
    for (const auto& entry : c.database.cases) {
      if (entry.method != m.id || entry.route != route->id || entry.executor != selected_executor)
        continue;
      const auto value = select(c.profiles.at(entry.profile).samples.at(entry.sample).value, bytes, actual);
      if (value) {
        check(!out, "multiple eligible transfer samples");
        out = value->bandwidth;
      }
    }
  return out;
}
void apply_write_cost(const Calibration& c, std::vector<ResourceDemand>& demands, const CalibrationControl* actual) {
  for (auto& d : demands)
    if (d.service == DemandKind::write) {
      const auto* p = bound(c, ModelTargetKind::resource, d.resource);
      if (!p)
        continue;
      if (!p->origin.implementation.empty() && (!actual || actual->implementation != p->origin.implementation))
        continue;
      d.amount *= *p->write_cost;
      const long double fixed = std::ceil(static_cast<long double>(d.per_invocation_bytes) * *p->write_cost);
      check(std::isfinite(d.amount) && fixed <= std::numeric_limits<U>::max(), "write_cost overflow");
      d.per_invocation_bytes = static_cast<U>(fixed);
    }
}
std::string synthetic(const ResourceGraph& graph, std::string prefix) {
  U i = 0;
  std::string id;
  do {
    id = prefix + std::to_string(i++);
  } while (find(graph.resources, id) || find(graph.methods, id));
  return id;
}
void decorate_activities(ResourceView& local, std::vector<Activity>& activities, const ResourceView& original_view,
                         const Calibration& calibration, const CalibrationControls& controls) {
  for (auto& a : activities) {
    if (a.method.empty() || !a.payload_bytes)
      continue;
    const auto* original = find(original_view.graph.methods, a.method);
    check(original, "unknown activity method");
    const auto* route = a.route ? find(original->routes, *a.route) : nullptr;
    check(!a.route || route, "unknown activity route");
    std::optional<double> rate = transfer_rate(calibration, *original, route, a.executor, a.payload_bytes, controls);
    if (original->bandwidth)
      rate = std::min(rate.value_or(*original->bandwidth), *original->bandwidth);
    if (original->request_executor && route) {
      check(a.executor.has_value(), "costed route needs executor");
      for (const auto mode : {AccessMode::read, AccessMode::write}) {
        const auto& pool = mode == AccessMode::read ? route->source : route->target;
        const std::string op = mode == AccessMode::read ? "read" : "write";
        bool found = false;
        for (const auto& e : original_view.graph.access)
          if (e.agent == *a.executor && e.pool == pool &&
              std::find(e.access.begin(), e.access.end(), mode) != e.access.end()) {
            found = true;
            std::optional<double> cap;
            if (const auto selected = access_rate(calibration, e, op, a.payload_bytes, controls))
              cap = selected->bandwidth;
            else if (e.bandwidth && e.bandwidth->contains(op))
              cap = e.bandwidth->at(op);
            if (cap)
              rate = std::min(rate.value_or(*cap), *cap);
          }
        check(found, "missing activity access path");
      }
    }
    // Each activity receives a private cap. Shared bus/service demands remain
    // unchanged; separate activities do not compete on a calibration artifact.
    Method capped = *original;
    capped.id = synthetic(local.graph, "calibration-method/");
    capped.bandwidth = rate;
    capped.request_executor = false;
    a.method = capped.id;
    local.graph.methods.push_back(std::move(capped));
  }
}
} // namespace

CalibratedCostModel::CalibratedCostModel(const ResourceModel& base, std::shared_ptr<const Calibration> c,
                                         CalibrationControls controls)
  : base_(base)
  , calibration_(std::move(c))
  , controls_(std::move(controls)) {
  if (!calibration_)
    throw std::invalid_argument("null calibration");
  // A compiled overlay is tied to the full expanded topology, not merely to
  // similarly spelled IDs. Discovery may resolve capacities/address spaces.
  for (const auto& [target, name] : calibration_->targets) {
    (void)name;
    const auto& [kind, id] = target;
    bool compatible = false;
    if (kind == ModelTargetKind::access) {
      const auto* expected = find(calibration_->expected_graph.access, id);
      const auto* actual = find(base_.graph().access, id);
      compatible = actual && expected && actual->agent == expected->agent && actual->pool == expected->pool &&
                   actual->rule == expected->rule;
    } else if (kind == ModelTargetKind::resource) {
      const auto* expected = find(calibration_->expected_graph.resources, id);
      const auto* actual = find(base_.graph().resources, id);
      compatible = actual && expected && actual->kind == expected->kind && actual->sharing == expected->sharing &&
                   actual->rule == expected->rule;
    } else {
      const auto* expected = find(calibration_->expected_graph.methods, id);
      const auto* actual = find(base_.graph().methods, id);
      compatible = actual && expected && actual->request_executor == expected->request_executor &&
                   actual->routes.size() == expected->routes.size();
      if (compatible)
        for (const auto& route : expected->routes) {
          const auto* found = find(actual->routes, route.id);
          if (!found || std::tie(found->agent, found->controller, found->source, found->target) !=
                          std::tie(route.agent, route.controller, route.source, route.target)) {
            compatible = false;
            break;
          }
        }
    }
    if (!compatible)
      throw std::invalid_argument("calibration topology mismatch: " + id);
  }
  for (const auto& [id, value] : controls_) {
    if (!find(base_.graph().agents, id) || (value.cpu_cores && !*value.cpu_cores))
      throw std::invalid_argument("invalid calibration executor controls");
  }
}
Result<std::vector<TransferCandidate>> CalibratedCostModel::candidates(const ResourceView& v, const TransferEndpoint& s,
                                                                       const TransferEndpoint& t) const {
  return base_.candidates(v, s, t);
}
Result<CostEstimate> CalibratedCostModel::estimate_access(const ResourceView& v, const AccessQuery& q,
                                                          const CostContext& context) const {
  return checked([&] {
    U remaining = q.payload_bytes;
    const std::string op = q.access == AccessMode::read ? "read" : "write";
    // Repeated segments of an edge use the lowest eligible cap, so no segment
    // is accidentally assigned the rate of a differently sized segment.
    std::map<std::string, double> caps;
    for (const auto& s : q.memory.segments) {
      const U n = std::min(s.size_bytes, remaining);
      if (!n)
        break;
      for (const auto& e : v.graph.access)
        if (e.agent == q.agent && e.pool == s.pool)
          if (const auto value = access_rate(*calibration_, e, op, n, controls_)) {
            auto [i, inserted] = caps.emplace(e.id, value->bandwidth);
            if (!inserted)
              i->second = std::min(i->second, value->bandwidth);
          }
      remaining -= n;
    }
    if (caps.empty() && context.active.empty())
      return take(base_.estimate_access(v, q, context));
    auto local = v;
    for (auto& e : local.graph.access)
      if (caps.contains(e.id)) {
        if (!e.bandwidth)
          e.bandwidth.emplace();
        (*e.bandwidth)[op] = caps.at(e.id);
      }
    auto ctx = context;
    decorate_activities(local, ctx.active, v, *calibration_, controls_);
    return take(base_.estimate_access(local, q, ctx));
  });
}
Result<std::vector<CalibrationUse>> CalibratedCostModel::explain_access(const ResourceView& v,
                                                                        const AccessQuery& q) const {
  return checked([&] {
    std::vector<CalibrationUse> out;
    U left = q.payload_bytes;
    for (const auto& s : q.memory.segments) {
      const auto n = std::min(left, s.size_bytes);
      if (!n)
        break;
      for (const auto& e : v.graph.access)
        if (e.agent == q.agent && e.pool == s.pool) {
          CalibrationUse use;
          if (access_rate(*calibration_, e, q.access == AccessMode::read ? "read" : "write", n, controls_, &use))
            out.push_back(std::move(use));
        }
      left -= n;
    }
    // Validate the same endpoint/agent contract as the actual query.
    take(base_.estimate_access(v, q, CostContext{v.epoch, {}, {}, false}));
    return out;
  });
}
Result<CostEstimate> CalibratedCostModel::estimate(const ResourceView& v, const TransferQuery& q,
                                                   const CostContext& context) const {
  return checked([&] {
    auto local = v;
    auto ctx = context;
    decorate_activities(local, ctx.active, v, *calibration_, controls_);
    auto* m = const_cast<Method*>(find(local.graph.methods, q.method));
    check(m, "unknown transfer method");
    // Conservative cap for any route used more than once with different sizes.
    std::map<std::string, std::vector<U>> route_bytes;
    std::map<std::pair<std::string, std::string>, double> access_caps;
    U done = 0;
    std::size_t i = 0, j = 0;
    while (done < q.payload_bytes) {
      check(i < q.source.segments.size() && j < q.target.segments.size(), "transfer endpoint coverage is incomplete");
      const auto& s = q.source.segments[i];
      const auto& t = q.target.segments[j];
      check(s.offset_bytes <= done && t.offset_bytes <= done &&
              s.size_bytes <= std::numeric_limits<U>::max() - s.offset_bytes &&
              t.size_bytes <= std::numeric_limits<U>::max() - t.offset_bytes,
            "invalid endpoint range");
      const auto end = std::min({q.payload_bytes, s.offset_bytes + s.size_bytes, t.offset_bytes + t.size_bytes});
      check(end > done, "invalid endpoint coverage");
      const U n = end - done;
      for (const auto& r : m->routes)
        if (r.source == s.pool && r.target == t.pool && (!r.agent || r.agent == q.executor) &&
            (!m->request_executor || q.executor)) {
          route_bytes[r.id].push_back(n);
        }
      if (m->request_executor && q.executor) {
        for (const auto& [pool, op] : {std::pair{s.pool, std::string("read")}, std::pair{t.pool, std::string("write")}})
          for (const auto& edge : v.graph.access)
            if (edge.agent == *q.executor && edge.pool == pool)
              if (const auto cap = access_rate(*calibration_, edge, op, n, controls_)) {
                auto [entry, inserted] = access_caps.emplace(std::make_pair(edge.id, op), cap->bandwidth);
                if (!inserted)
                  entry->second = std::min(entry->second, cap->bandwidth);
              }
      }
      done = end;
      if (done == s.offset_bytes + s.size_bytes)
        ++i;
      if (done == t.offset_bytes + t.size_bytes)
        ++j;
    }
    for (auto& r : m->routes)
      if (route_bytes.contains(r.id)) {
        const auto executor = m->request_executor ? q.executor
                              : r.controller      ? r.controller
                              : r.agent           ? r.agent
                                                  : q.executor;
        apply_write_cost(*calibration_, r.demands, control(controls_, executor));
        std::optional<double> rate;
        for (const auto bytes : route_bytes.at(r.id))
          if (const auto value = transfer_rate(*calibration_, *m, &r, q.executor, bytes, controls_))
            rate = std::min(rate.value_or(*value), *value);
        if (rate) {
          const auto id = synthetic(local.graph, "calibration-cap/");
          local.graph.resources.push_back({id, "", *rate, {}, {}, SharingMode::shared, ResourceKind::bandwidth, {}});
          r.demands.push_back({id, DemandKind::traffic, 1, 0});
        }
      }
    for (auto& edge : local.graph.access)
      for (const auto op : {"read", "write"})
        if (const auto cap = access_caps.find({edge.id, op}); cap != access_caps.end()) {
          if (!edge.bandwidth)
            edge.bandwidth.emplace();
          (*edge.bandwidth)[op] = cap->second;
        }
    return take(base_.estimate(local, q, ctx));
  });
}
Result<CostPlan> CalibratedCostModel::cost_plan(const ResourceView& v, const MemoryCommandPlan& commands,
                                                const OperandEnvironment& env) const {
  return checked([&] {
    auto plan = take(base_.cost_plan(v, commands, env));
    for (std::size_t i = 0; i < commands.commands.size(); ++i)
      if (commands.commands[i].opcode == MemoryOpcode::copy) {
        auto& a = plan.activities[i];
        const auto* m = find(v.graph.methods, a.method);
        const auto* route = m && a.route ? find(m->routes, *a.route) : nullptr;
        const auto executor = m && m->request_executor     ? a.executor
                              : route && route->controller ? route->controller
                              : route && route->agent      ? route->agent
                                                           : a.executor;
        apply_write_cost(*calibration_, a.demands, control(controls_, executor));
      }
    return plan;
  });
}
Result<CostEstimate> CalibratedCostModel::estimate_schedule(const ResourceView& v, const CostPlan& plan,
                                                            const CostContext& context) const {
  return checked([&] {
    auto local = v;
    auto work = plan;
    auto ctx = context;
    decorate_activities(local, ctx.active, v, *calibration_, controls_);
    decorate_activities(local, work.activities, v, *calibration_, controls_);
    return take(base_.estimate_schedule(local, work, ctx));
  });
}
} // namespace vecops::memory::model
