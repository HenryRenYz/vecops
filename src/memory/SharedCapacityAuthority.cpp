// SPDX-FileCopyrightText: 2026 HenryRenYz and vecops contributors
// SPDX-License-Identifier: MIT
#include "vecops/memory/details/SharedCapacityAuthority.h"
#include "../../third_party/tomlplusplus/toml.hpp"
#include <algorithm>
#include <array>
#include <cerrno>
#include <charconv>
#include <fcntl.h>
#include <fstream>
#include <limits>
#include <set>
#include <signal.h>
#include <sstream>
#include <sys/file.h>
#include <sys/random.h>
#include <sys/stat.h>
#include <system_error>
#include <unistd.h>

namespace vecops::memory::model::details {
namespace {
using U = std::uint64_t;
struct Failure { Error error; };
[[noreturn]] void fail(const char* code, const std::string& message) {
  throw Failure{Error{std::string("shared_capacity.") + code, message}};
}
void check(bool yes, const char* code, const std::string& message) {
  if (!yes)
    fail(code, message);
}
[[noreturn]] void os_error(const char* operation) {
  throw std::system_error(errno, std::generic_category(), operation);
}
template <class F>
auto checked(F&& fn) -> Result<decltype(fn())> {
  using T = decltype(fn());
  try {
    if constexpr (std::is_void_v<T>) { fn(); return {}; }
    else return {fn()};
  } catch (const Failure& e) { return {e.error}; }
  catch (const toml::parse_error& e) {
    std::ostringstream out; out << e;
    return {Error{"shared_capacity.metadata", out.str()}};
  } catch (const std::system_error& e) { return {Error{"shared_capacity.io", e.what()}}; }
  catch (const std::bad_alloc&) { return {Error{"shared_capacity.out_of_memory", "allocation failed"}}; }
  catch (const std::exception& e) { return {Error{"shared_capacity.metadata", e.what()}}; }
}
template <class T>
T take(Result<T> r) {
  if (!r) throw Failure{std::get<Error>(r.state)};
  return std::move(r).value();
}
U add(U a, U b) {
  check(b <= std::numeric_limits<U>::max() - a, "range", "capacity arithmetic overflow");
  return a + b;
}
struct Fd {
  int value = -1;
  explicit Fd(int fd = -1) : value(fd) {}
  ~Fd() { if (value >= 0) ::close(value); }
  Fd(const Fd&) = delete;
  Fd& operator=(const Fd&) = delete;
};
void secure_file(int fd) {
  struct stat info{};
  if (::fstat(fd, &info)) os_error("fstat capacity file");
  check(S_ISREG(info.st_mode) && info.st_uid == ::geteuid() && (info.st_mode & 07777) == 0600,
        "permissions", "capacity files must be regular, owner-only mode 0600 files");
  check(info.st_nlink == 1, "permissions", "capacity files cannot have extra hard links");
}
struct Lock {
  int fd;
  explicit Lock(int value) : fd(value) {
    while (::flock(fd, LOCK_EX)) if (errno != EINTR) os_error("flock capacity authority");
  }
  ~Lock() { ::flock(fd, LOCK_UN); }
};
std::string read_text(const std::string& path) {
  std::ifstream file(path);
  if (!file) fail("host", "cannot read host/process identity: " + path);
  std::string value{std::istreambuf_iterator<char>(file), std::istreambuf_iterator<char>()};
  while (!value.empty() && (value.back() == '\n' || value.back() == '\r')) value.pop_back();
  return value;
}
U decimal(std::string_view text) {
  U result{};
  const auto parsed = std::from_chars(text.data(), text.data() + text.size(), result);
  check(!text.empty() && parsed.ec == std::errc{} && parsed.ptr == text.data() + text.size(),
        "metadata", "expected exact uint64 decimal string");
  return result;
}
std::optional<U> process_start(U pid) {
  std::ifstream file("/proc/" + std::to_string(pid) + "/stat");
  if (!file) return {};
  std::string text{std::istreambuf_iterator<char>(file), std::istreambuf_iterator<char>()};
  const auto end = text.rfind(')');
  if (end == std::string::npos) return {};
  std::istringstream fields(text.substr(end + 1));
  std::string value;
  for (unsigned field = 3; field <= 22; ++field) {
    if (!(fields >> value)) return {};
    if (field == 22) {
      U result{};
      const auto parsed = std::from_chars(value.data(), value.data() + value.size(), result);
      if (parsed.ec != std::errc{} || parsed.ptr != value.data() + value.size()) return {};
      return result;
    }
  }
  return {};
}
SharedCapacityOwnerState owner_state(const SharedCapacityOwner& owner) {
  if (auto start = process_start(owner.pid))
    return *start == owner.start_time_ticks ? SharedCapacityOwnerState::alive : SharedCapacityOwnerState::owner_dead;
  if (owner.pid <= static_cast<U>(std::numeric_limits<pid_t>::max()) &&
      ::kill(static_cast<pid_t>(owner.pid), 0) && errno == ESRCH)
    return SharedCapacityOwnerState::owner_dead;
  return SharedCapacityOwnerState::unknown;
}
bool same_owner(const SharedCapacityOwner& a, const SharedCapacityOwner& b) {
  return a.participant == b.participant && a.pid == b.pid && a.start_time_ticks == b.start_time_ticks;
}
std::string nonce() {
  std::array<unsigned char, 16> bytes{};
  std::size_t at = 0;
  while (at < bytes.size()) {
    const auto n = ::getrandom(bytes.data() + at, bytes.size() - at, 0);
    if (n < 0) { if (errno == EINTR) continue; os_error("getrandom capacity receipt"); }
    check(n > 0, "host", "empty random identity");
    at += static_cast<std::size_t>(n);
  }
  std::string value;
  static constexpr char digits[] = "0123456789abcdef";
  for (auto b : bytes) { value += digits[b >> 4]; value += digits[b & 15]; }
  return value;
}
struct Config {
  std::vector<SharedCapacityGroup> groups;
  std::vector<BudgetConstraint> constraints;
};
Config canonical(Config c) {
  std::sort(c.groups.begin(), c.groups.end(), [](const auto& a, const auto& b) { return a.id < b.id; });
  std::set<std::string> ids;
  check(!c.groups.empty(), "config", "at least one capacity group is required");
  for (const auto& g : c.groups)
    check(!g.id.empty() && ids.insert(g.id).second && g.capacity_bytes && g.hard_limit_bytes <= g.capacity_bytes,
          "config", "invalid or duplicate capacity group");
  std::sort(c.constraints.begin(), c.constraints.end(), [](const auto& a, const auto& b) { return a.id < b.id; });
  std::set<std::string> constraint_ids;
  for (auto& constraint : c.constraints) {
    check(!constraint.id.empty() && constraint_ids.insert(constraint.id).second && !constraint.capacity_groups.empty(),
          "config", "invalid or duplicate shared constraint");
    std::sort(constraint.capacity_groups.begin(), constraint.capacity_groups.end());
    std::string previous;
    for (const auto& group : constraint.capacity_groups) {
      check(ids.contains(group) && group != previous, "config", "unknown or duplicate constrained group");
      previous = group;
    }
  }
  return c;
}
bool equal(const Config& a, const Config& b) {
  if (a.groups.size() != b.groups.size() || a.constraints.size() != b.constraints.size()) return false;
  for (std::size_t i = 0; i < a.groups.size(); ++i) {
    const auto& x = a.groups[i]; const auto& y = b.groups[i];
    if (x.id != y.id || x.capacity_bytes != y.capacity_bytes || x.hard_limit_bytes != y.hard_limit_bytes) return false;
  }
  for (std::size_t i = 0; i < a.constraints.size(); ++i) {
    const auto& x = a.constraints[i]; const auto& y = b.constraints[i];
    if (x.id != y.id || x.capacity_groups != y.capacity_groups || x.remaining_bytes != y.remaining_bytes) return false;
  }
  return true;
}
struct Metadata {
  std::string boot, host;
  U epoch = 1;
  Config config;
  std::map<std::string, SharedCapacityGrantReceipt> grants;
  std::map<std::string, SharedCapacityRootReceipt> roots;
};
const toml::node& required(const toml::table& t, std::string_view key) {
  const auto* n = t.get(key);
  check(n, "metadata", "missing field: " + std::string(key));
  return *n;
}
const toml::table& table(const toml::node& n) {
  check(n.is_table(), "metadata", "expected table");
  return *n.as_table();
}
const toml::array& array(const toml::node& n) {
  check(n.is_array(), "metadata", "expected array");
  return *n.as_array();
}
std::string string(const toml::node& n) {
  check(n.is_string(), "metadata", "expected string");
  return n.as_string()->get();
}
U integer(const toml::node& n) { return decimal(string(n)); }
void fields(const toml::table& t, std::initializer_list<std::string_view> allowed) {
  for (const auto& [key, value] : t) {
    (void)value;
    check(std::find(allowed.begin(), allowed.end(), key.str()) != allowed.end(), "metadata",
          "unknown metadata field: " + std::string(key.str()));
  }
}
SharedCapacityOwner parse_owner(const toml::node& value) {
  const auto& t = table(value);
  fields(t, {"participant", "pid", "start_time_ticks"});
  SharedCapacityOwner result{string(required(t, "participant")), integer(required(t, "pid")),
                            integer(required(t, "start_time_ticks")), SharedCapacityOwnerState::unknown};
  check(!result.participant.empty() && result.pid && result.pid <= static_cast<U>(std::numeric_limits<pid_t>::max()),
        "metadata", "invalid receipt owner");
  return result;
}
Metadata parse(std::string_view text) {
  const auto t = toml::parse(text);
  fields(t, {"schema", "boot_id", "host_id", "epoch", "groups", "constraints", "grants", "roots"});
  check(string(required(t, "schema")) == "shared-capacity/0.1", "metadata", "unknown metadata schema");
  Metadata m;
  m.boot = string(required(t, "boot_id")); m.host = string(required(t, "host_id"));
  m.epoch = integer(required(t, "epoch"));
  check(m.epoch && !m.boot.empty() && !m.host.empty(), "metadata", "invalid host or epoch");
  for (const auto& n : array(required(t, "groups"))) {
    const auto& g = table(n); fields(g, {"id", "capacity_bytes", "hard_limit_bytes"});
    m.config.groups.push_back({string(required(g, "id")), integer(required(g, "capacity_bytes")),
                               integer(required(g, "hard_limit_bytes"))});
  }
  for (const auto& n : array(required(t, "constraints"))) {
    const auto& c = table(n); fields(c, {"id", "capacity_groups", "limit_bytes"});
    BudgetConstraint constraint{string(required(c, "id")), {}, integer(required(c, "limit_bytes"))};
    for (const auto& group : array(required(c, "capacity_groups"))) constraint.capacity_groups.push_back(string(group));
    m.config.constraints.push_back(std::move(constraint));
  }
  m.config = canonical(std::move(m.config));
  for (const auto& n : array(required(t, "grants"))) {
    const auto& g = table(n); fields(g, {"id", "owner", "remaining"});
    SharedCapacityGrantReceipt grant{string(required(g, "id")), parse_owner(required(g, "owner")), {}};
    for (const auto& [key, value] : table(required(g, "remaining")))
      grant.remaining.emplace(std::string(key.str()), integer(value));
    check(!grant.id.empty() && !grant.remaining.empty(), "metadata", "empty grant receipt");
    check(m.grants.emplace(grant.id, grant).second, "metadata", "duplicate grant receipt");
  }
  for (const auto& n : array(required(t, "roots"))) {
    const auto& r = table(n); fields(r, {"id", "owner", "group", "bytes", "physical_identity"});
    SharedCapacityRootReceipt root{string(required(r, "id")), parse_owner(required(r, "owner")),
                                   string(required(r, "group")), integer(required(r, "bytes")),
                                   string(required(r, "physical_identity"))};
    check(!root.id.empty() && !root.physical_identity.empty() && root.bytes, "metadata", "invalid root receipt");
    check(m.roots.emplace(root.id, root).second, "metadata", "duplicate root receipt");
  }
  return m;
}
toml::table owner_table(const SharedCapacityOwner& owner) {
  return toml::table{ {"participant", owner.participant}, {"pid", std::to_string(owner.pid)},
                     {"start_time_ticks", std::to_string(owner.start_time_ticks)} };
}
std::string serialize(const Metadata& m) {
  toml::array groups, constraints, grants, roots;
  for (const auto& g : m.config.groups)
    groups.push_back(toml::table{{"id", g.id}, {"capacity_bytes", std::to_string(g.capacity_bytes)},
                                 {"hard_limit_bytes", std::to_string(g.hard_limit_bytes)}});
  for (const auto& c : m.config.constraints) {
    toml::array members; for (const auto& g : c.capacity_groups) members.push_back(g);
    constraints.push_back(toml::table{{"id", c.id}, {"capacity_groups", std::move(members)},
                                      {"limit_bytes", std::to_string(c.remaining_bytes)}});
  }
  for (const auto& [id, g] : m.grants) {
    toml::table remaining; for (const auto& [group, n] : g.remaining) remaining.insert(group, std::to_string(n));
    grants.push_back(toml::table{{"id", id}, {"owner", owner_table(g.owner)}, {"remaining", std::move(remaining)}});
  }
  for (const auto& [id, r] : m.roots)
    roots.push_back(toml::table{{"id", id}, {"owner", owner_table(r.owner)}, {"group", r.group},
                               {"bytes", std::to_string(r.bytes)}, {"physical_identity", r.physical_identity}});
  toml::table t{{"schema", "shared-capacity/0.1"}, {"boot_id", m.boot}, {"host_id", m.host},
                {"epoch", std::to_string(m.epoch)}, {"groups", std::move(groups)},
                {"constraints", std::move(constraints)}, {"grants", std::move(grants)}, {"roots", std::move(roots)}};
  std::ostringstream out; out << toml::toml_formatter{t}; return out.str();
}
SharedCapacitySnapshot summarize(const Metadata& m, bool owners) {
  SharedCapacitySnapshot result{}; result.epoch = m.epoch;
  std::map<std::string, U> committed, pending;
  std::set<std::string> group_ids, physical_ids;
  for (const auto& g : m.config.groups) group_ids.insert(g.id);
  for (const auto& [id, grant] : m.grants) {
    (void)id;
    auto copy = grant; if (owners) copy.owner.state = owner_state(copy.owner);
    result.grants.push_back(std::move(copy));
    for (const auto& [group, n] : grant.remaining) {
      check(group_ids.contains(group) && n, "metadata", "invalid grant group or empty charge");
      pending[group] = add(pending[group], n);
    }
  }
  for (const auto& [id, root] : m.roots) {
    (void)id;
    check(group_ids.contains(root.group) && physical_ids.insert(root.physical_identity).second,
          "metadata", "unknown root group or duplicate physical identity");
    committed[root.group] = add(committed[root.group], root.bytes);
    auto copy = root; if (owners) copy.owner.state = owner_state(copy.owner);
    result.roots.push_back(std::move(copy));
  }
  for (const auto& group : m.config.groups) {
    const U used = add(committed[group.id], pending[group.id]);
    check(used <= group.hard_limit_bytes, "metadata", "receipts exceed group hard limit");
    result.groups.push_back({group.id, group.capacity_bytes, group.hard_limit_bytes,
                             committed[group.id], pending[group.id], group.hard_limit_bytes - used});
  }
  for (const auto& c : m.config.constraints) {
    U used = 0;
    for (const auto& group : c.capacity_groups) used = add(used, add(committed[group], pending[group]));
    check(used <= c.remaining_bytes, "metadata", "receipts exceed shared hard constraint");
    result.constraints.push_back({c.id, c.capacity_groups, c.remaining_bytes, c.remaining_bytes - used});
  }
  return result;
}
void advance(Metadata& m) {
  check(m.epoch < std::numeric_limits<U>::max(), "epoch", "shared capacity epoch exhausted");
  ++m.epoch;
}
} // namespace

struct SharedCapacityState : std::enable_shared_from_this<SharedCapacityState> {
  Config config;
  SharedCapacityOwner owner;
  std::string boot, host, instance, namespace_id;
  Fd directory, lock_file;
  mutable std::mutex mutex;
  U serial = 0;
  std::optional<Error> cleanup_failure;
  SharedCapacityState(std::filesystem::path path, Config c, std::string participant) : config(std::move(c)) {
    check(!participant.empty(), "config", "participant cannot be empty");
    boot = read_text("/proc/sys/kernel/random/boot_id");
    std::array<char, 256> hostname{};
    if (::gethostname(hostname.data(), hostname.size())) os_error("gethostname capacity authority");
    check(hostname.back() == '\0', "host", "hostname is too long"); host = hostname.data();
    const auto start = process_start(static_cast<U>(::getpid()));
    check(start.has_value(), "host", "cannot certify current process start time");
    owner = {std::move(participant), static_cast<U>(::getpid()), *start, SharedCapacityOwnerState::alive};
    instance = nonce();
    if (::mkdir(path.c_str(), 0700) && errno != EEXIST) os_error("mkdir capacity authority");
    directory.value = ::open(path.c_str(), O_RDONLY | O_DIRECTORY | O_NOFOLLOW | O_CLOEXEC);
    if (directory.value < 0) os_error("open capacity directory");
    struct stat info{}; if (::fstat(directory.value, &info)) os_error("fstat capacity directory");
    check(info.st_uid == ::geteuid() && (info.st_mode & 07777) == 0700,
          "permissions", "authority directory must be owned by current UID and mode 0700");
    lock_file.value = ::openat(directory.value, "lock", O_RDWR | O_CREAT | O_NOFOLLOW | O_CLOEXEC, 0600);
    if (lock_file.value < 0) os_error("open capacity lock");
    secure_file(lock_file.value);
    if (::fstat(lock_file.value, &info)) os_error("fstat capacity lock");
    namespace_id = host + ":" + boot + ":" + std::to_string(info.st_dev) + ":" + std::to_string(info.st_ino);
    Lock lock(lock_file.value);
    auto existing = load(false);
    if (!existing) {
      Metadata m; m.boot = boot; m.host = host; m.config = config; store(m);
    } else {
      check_host(*existing);
      summarize(*existing, false);
      if (!equal(existing->config, config)) {
        check(existing->grants.empty() && existing->roots.empty(), "config_active",
              "different capacity configuration has active ownership receipts");
        existing->config = config; advance(*existing); store(*existing);
      }
    }
  }
  std::string id(const char* kind) {
    check(serial < std::numeric_limits<U>::max(), "epoch", "receipt identities exhausted");
    return std::string(kind) + "/" + instance + "/" + std::to_string(++serial);
  }
  void check_process() const {
    check(owner.pid == static_cast<U>(::getpid()), "fork", "reopen capacity authority in the forked child");
  }
  void check_host(const Metadata& m) const {
    check(m.boot == boot && m.host == host, "host", "capacity namespace belongs to a different host or boot");
  }
  std::optional<Metadata> load(bool required_file = true) const {
    Fd fd(::openat(directory.value, "capacity.toml", O_RDONLY | O_NOFOLLOW | O_CLOEXEC));
    if (fd.value < 0) {
      if (!required_file && errno == ENOENT) return {};
      os_error("open capacity metadata");
    }
    secure_file(fd.value);
    struct stat info{}; if (::fstat(fd.value, &info)) os_error("fstat capacity metadata");
    check(info.st_size >= 0 && info.st_size <= 64 * 1024 * 1024, "metadata", "capacity metadata exceeds size limit");
    std::string text(static_cast<std::size_t>(info.st_size), '\0');
    std::size_t at = 0;
    while (at < text.size()) {
      const auto n = ::read(fd.value, text.data() + at, text.size() - at);
      if (n < 0) { if (errno == EINTR) continue; os_error("read capacity metadata"); }
      check(n > 0, "metadata", "truncated capacity metadata"); at += static_cast<std::size_t>(n);
    }
    return parse(text);
  }
  void store(const Metadata& m) {
    const auto text = serialize(m);
    const auto temp = ".capacity-" + instance + "-" + std::to_string(m.epoch) + ".tmp";
    Fd fd(::openat(directory.value, temp.c_str(), O_WRONLY | O_CREAT | O_EXCL | O_NOFOLLOW | O_CLOEXEC, 0600));
    if (fd.value < 0) os_error("create capacity metadata temporary");
    struct Remove { int directory; const std::string& name; ~Remove() { ::unlinkat(directory, name.c_str(), 0); } } remove{directory.value, temp};
    secure_file(fd.value);
    std::size_t at = 0;
    while (at < text.size()) {
      const auto n = ::write(fd.value, text.data() + at, text.size() - at);
      if (n < 0) { if (errno == EINTR) continue; os_error("write capacity metadata"); }
      check(n > 0, "io", "metadata write made no progress"); at += static_cast<std::size_t>(n);
    }
    if (::fsync(fd.value)) os_error("fsync capacity metadata");
    if (::renameat(directory.value, temp.c_str(), directory.value, "capacity.toml")) os_error("commit capacity metadata");
    if (::fsync(directory.value)) fail("commit_uncertain", "metadata rename committed, directory durability sync failed");
  }
  template <class F>
  auto transaction(F&& fn, bool write) {
    check_process();
    std::lock_guard thread_lock(mutex); Lock process_lock(lock_file.value);
    auto m = *load(); check_host(m);
    check(equal(m.config, config), "config_changed", "authority configuration changed; reopen this authority");
    summarize(m, false);
    auto result = fn(m);
    if (write) { advance(m); summarize(m, false); store(m); }
    return result;
  }
  void refund(const std::string& id, bool root) {
    if (owner.pid != static_cast<U>(::getpid())) return; // inherited copies never own a child's refund
    auto result = checked([&] {
      transaction([&](Metadata& m) {
        if (root) {
          auto it = m.roots.find(id);
          check(it != m.roots.end() && same_owner(it->second.owner, owner), "receipt", "root refund owner mismatch");
          m.roots.erase(it);
        } else if (auto it = m.grants.find(id); it != m.grants.end()) {
          check(same_owner(it->second.owner, owner), "receipt", "grant refund owner mismatch"); m.grants.erase(it);
        }
        return 0;
      }, true);
    });
    if (!result) {
      try {
        std::lock_guard lock(mutex);
        if (!cleanup_failure) cleanup_failure = *result.error;
      } catch (...) {} // persisted receipt remains conservative even under OOM
    }
  }
  std::unique_ptr<SharedCapacityGrant> reserve(
      const std::map<std::string, U>& bytes, std::optional<U> epoch,
      const std::function<Result<std::map<std::string, U>>()>& sampler) {
    std::unique_ptr<SharedCapacityGrant> grant;
    transaction([&](Metadata& m) {
      check(!epoch || *epoch == m.epoch, "stale", "stale shared capacity snapshot");
      check(!bytes.empty(), "request", "empty shared capacity reservation");
      // Sampling before taking this lock, then refreshing the epoch, would miss
      // roots materialized and committed between those operations. Sampling
      // here keeps all simultaneous uncommitted materializations pending.
      const auto samples = take(sampler());
      const auto snapshot = summarize(m, false);
      for (const auto& [group, sample] : samples) {
        auto g = std::find_if(snapshot.groups.begin(), snapshot.groups.end(), [&](const auto& x) { return x.id == group; });
        check(g != snapshot.groups.end() && sample <= g->capacity_bytes, "sample", "unknown or oversized physical free sample");
      }
      for (const auto& [group, n] : bytes) {
        auto g = std::find_if(snapshot.groups.begin(), snapshot.groups.end(), [&](const auto& x) { return x.id == group; });
        check(g != snapshot.groups.end() && n, "request", "unknown group or zero reservation");
        check(n <= g->hard_remaining_bytes, "exhausted", "shared group hard capacity exhausted");
        auto free = samples.find(group);
        check(free != samples.end(), "sample", "missing physical free sample");
        if(g->pending_bytes > free->second || n > free->second - g->pending_bytes)
          fail("physical_free", "physical free sample minus pending reservations is exhausted: group="+group+
               ", requested_bytes="+std::to_string(n)+", sampled_free_bytes="+std::to_string(free->second)+
               ", pending_bytes="+std::to_string(g->pending_bytes));
      }
      for (const auto& constraint : snapshot.constraints) {
        U requested = 0;
        for (const auto& group : constraint.capacity_groups)
          if (auto it = bytes.find(group); it != bytes.end()) requested = add(requested, it->second);
        check(requested <= constraint.remaining_bytes, "exhausted", "overlapping shared capacity constraint exhausted");
      }
      const auto key = id("grant");
      grant = std::unique_ptr<SharedCapacityGrant>(new SharedCapacityGrant(shared_from_this(), key));
      m.grants.emplace(key, SharedCapacityGrantReceipt{key, owner, bytes});
      return 0;
    }, true);
    grant->active_ = true;
    return grant;
  }
  std::shared_ptr<SharedCapacityCharge> commit(const std::string& grant, const std::string& group, U bytes,
                                               const std::string& identity) {
    std::shared_ptr<SharedCapacityCharge> charge;
    transaction([&](Metadata& m) {
      check(bytes && !identity.empty(), "request", "commit requires positive bytes and canonical physical identity");
      auto it = m.grants.find(grant);
      check(it != m.grants.end() && same_owner(it->second.owner, owner), "receipt", "unknown/consumed/foreign grant");
      auto remaining = it->second.remaining.find(group);
      check(remaining != it->second.remaining.end() && bytes <= remaining->second, "request", "commit exceeds reserved group bytes");
      for (const auto& [id, root] : m.roots) {
        (void)id;
        check(root.physical_identity != identity, "duplicate_physical", "physical backing already has an ownership charge");
      }
      const auto key = id("root");
      SharedCapacityRootReceipt receipt{key, owner, group, bytes, identity};
      charge = std::shared_ptr<SharedCapacityCharge>(new SharedCapacityCharge(shared_from_this(), receipt));
      m.roots.emplace(key, std::move(receipt));
      remaining->second -= bytes;
      if (!remaining->second) it->second.remaining.erase(remaining);
      if (it->second.remaining.empty()) m.grants.erase(it);
      return 0;
    }, true);
    charge->active_ = true;
    return charge;
  }
};

SharedCapacityGrant::SharedCapacityGrant(std::shared_ptr<SharedCapacityState> state, std::string id)
  : state_(std::move(state)), id_(std::move(id)) {}
SharedCapacityGrant::~SharedCapacityGrant() { if (active_) state_->refund(id_, false); }
const std::string& SharedCapacityGrant::id() const noexcept { return id_; }
Result<std::shared_ptr<SharedCapacityCharge>> SharedCapacityGrant::commit(const std::string& group, U bytes,
                                                                          const std::string& identity) {
  return checked([&] { std::lock_guard lock(mutex_); return state_->commit(id_, group, bytes, identity); });
}
SharedCapacityCharge::SharedCapacityCharge(std::shared_ptr<SharedCapacityState> state, SharedCapacityRootReceipt receipt)
  : state_(std::move(state)), receipt_(std::move(receipt)) {}
SharedCapacityCharge::~SharedCapacityCharge() { if (active_) state_->refund(receipt_.id, true); }
const SharedCapacityRootReceipt& SharedCapacityCharge::receipt() const noexcept { return receipt_; }
SharedCapacityAuthority::SharedCapacityAuthority(std::shared_ptr<SharedCapacityState> state) : state_(std::move(state)) {}
Result<std::shared_ptr<SharedCapacityAuthority>>
SharedCapacityAuthority::open(const std::filesystem::path& directory, std::vector<SharedCapacityGroup> groups,
                              std::vector<BudgetConstraint> constraints, std::string participant) {
  return checked([&] {
    auto state = std::make_shared<SharedCapacityState>(directory, canonical({std::move(groups), std::move(constraints)}),
                                                       std::move(participant));
    return std::shared_ptr<SharedCapacityAuthority>(new SharedCapacityAuthority(std::move(state)));
  });
}
Result<SharedCapacitySnapshot> SharedCapacityAuthority::snapshot() const {
  return checked([&] { return state_->transaction([](Metadata& m) { return summarize(m, true); }, false); });
}
Result<std::unique_ptr<SharedCapacityGrant>>
SharedCapacityAuthority::reserve(const std::map<std::string, U>& bytes, U epoch, const std::map<std::string, U>& samples) {
  return checked([&] { return state_->reserve(bytes, epoch, [&]() -> Result<std::map<std::string, U>> { return {samples}; }); });
}
Result<std::unique_ptr<SharedCapacityGrant>> SharedCapacityAuthority::reserve_current(
    const std::map<std::string, U>& bytes,
    const std::function<Result<std::map<std::string, U>>()>& sampler) {
  return checked([&] {
    check(static_cast<bool>(sampler), "sample", "missing physical free sampler");
    return state_->reserve(bytes, {}, sampler);
  });
}
const std::string& SharedCapacityAuthority::participant() const noexcept { return state_->owner.participant; }
const std::string& SharedCapacityAuthority::namespace_id() const noexcept { return state_->namespace_id; }
std::optional<Error> SharedCapacityAuthority::cleanup_error() const {
  std::lock_guard lock(state_->mutex); return state_->cleanup_failure;
}
} // namespace vecops::memory::model::details
