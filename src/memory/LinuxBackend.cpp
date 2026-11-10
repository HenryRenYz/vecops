// SPDX-FileCopyrightText: 2026 HenryRenYz and vecops contributors
// SPDX-License-Identifier: MIT
#ifndef _GNU_SOURCE
#  define _GNU_SOURCE
#endif
#include "vecops/memory/LinuxBackend.h"
#include "vecops/memory/ResourceModel.h"

#include <algorithm>
#include <atomic>
#include <cerrno>
#include <cstring>
#include <fstream>
#include <limits>
#include <map>
#include <mutex>
#include <stdexcept>
#include <system_error>
#include <sstream>

#if defined(__linux__)
#  include <fcntl.h>
#  include <linux/memfd.h>
#  include <linux/mempolicy.h>
#  include <sys/mman.h>
#  include <sys/stat.h>
#  include <sys/syscall.h>
#  include <sys/sysmacros.h>
#  include <unistd.h>
#endif

namespace vecops::memory::model {
namespace {
using U = std::uint64_t;
void require(bool ok, const char* message) {
  if (!ok)
    throw std::invalid_argument(message);
}
void os_require(bool ok, const char* operation) {
  if (!ok)
    throw std::system_error(errno, std::generic_category(), operation);
}
std::string kernel_uuid(const char* failure) {
  std::ifstream random("/proc/sys/kernel/random/uuid");
  std::string nonce;
  std::getline(random, nonce);
  require(nonce.size() == 36 && std::all_of(nonce.begin(), nonce.end(), [](unsigned char ch) {
            return (ch >= '0' && ch <= '9') || (ch >= 'a' && ch <= 'f') || ch == '-';
          }), failure);
  return nonce;
}
U end(U offset, U bytes) {
  require(bytes && bytes <= std::numeric_limits<U>::max() - offset, "invalid byte range");
  return offset + bytes;
}
bool power2(U n) {
  return n && !(n & (n - 1));
}
template <class F>
auto checked(F&& fn) -> Result<decltype(fn())> {
  using T = decltype(fn());
  try {
    if constexpr (std::is_void_v<T>) {
      fn();
      return {};
    } else
      return {fn()};
  } catch (const std::system_error& e) {
    return {Error{"os_error", e.what()}};
  } catch (const std::invalid_argument& e) {
    return {Error{"invalid_argument", e.what()}};
  } catch (const std::overflow_error& e) {
    return {Error{"overflow", e.what()}};
  }
}
#if defined(__linux__)
U system_page() {
  const auto n = ::sysconf(_SC_PAGESIZE);
  require(n > 0, "unknown system page size");
  return static_cast<U>(n);
}
void geometry(U bytes, U q) {
  require(power2(q) && q >= system_page() && bytes && bytes % q == 0, "invalid mapping geometry");
  require(bytes <= static_cast<U>(std::numeric_limits<std::ptrdiff_t>::max()) &&
            q <= static_cast<U>(std::numeric_limits<std::ptrdiff_t>::max()) - bytes,
          "mapping exceeds process address range");
}
std::string inode_identity(int fd) {
  struct stat st {};
  os_require(::fstat(fd, &st) == 0, "fstat backing");
  std::ifstream boot("/proc/sys/kernel/random/boot_id");
  std::string id;
  std::getline(boot, id);
  require(!id.empty(), "cannot certify host boot identity");
  return "linux:" + id + ":" + std::to_string(st.st_dev) + ":" + std::to_string(st.st_ino);
}
U huge_alignment() {
  static const U result = [] {
    U n = 0;
    std::ifstream input("/sys/kernel/mm/transparent_hugepage/hpage_pmd_size");
    input >> n;
    return power2(n) && n >= system_page() && n <= (1ULL << 30) ? n : system_page();
  }();
  return result;
}
void* aligned_reservation(U bytes, U alignment) {
  require(alignment <= static_cast<U>(std::numeric_limits<std::ptrdiff_t>::max()) &&
            bytes <= static_cast<U>(std::numeric_limits<std::ptrdiff_t>::max()) - alignment,
          "aligned reservation exceeds process address range");
  const U total = bytes + alignment;
  void* base = ::mmap(nullptr, total, PROT_NONE, MAP_PRIVATE | MAP_ANONYMOUS | MAP_NORESERVE, -1, 0);
  os_require(base != MAP_FAILED, "reserve aligned address");
  const auto raw = reinterpret_cast<std::uintptr_t>(base);
  const U aligned = (raw + alignment - 1) & ~(alignment - 1);
  const U prefix = aligned - raw, suffix = total - prefix - bytes;
  if (prefix) ::munmap(base, prefix);
  if (suffix) ::munmap(reinterpret_cast<void*>(aligned + bytes), suffix);
  return reinterpret_cast<void*>(aligned);
}
struct AnonymousRoot {
  void* parking = MAP_FAILED;
  U bytes = 0;
  pid_t creator = ::getpid();
  bool active = false;
  void* location = MAP_FAILED;
  // Allocate the holder before fork. Child teardown deliberately drops no
  // creator-owned callback/ledger lock; its copied holder dies with the process.
  std::unique_ptr<std::shared_ptr<void>> lifetime;
  std::mutex mutex;
  ~AnonymousRoot() {
    if (creator != ::getpid()) { (void)lifetime.release(); return; }
    // DONTFORK prevents child COW allocation; never touch inherited stale VAs.
    if (creator == ::getpid() && parking != MAP_FAILED) {
      if (active) std::terminate();
      ::munmap(parking, bytes);
    }
  }
};
void same_process(const AnonymousRoot& root) {
  require(root.creator == ::getpid(), "private_movable backing cannot be inherited across fork");
}
/** Source and destination must be wholly owned, locked and quiescent. On
 * failure Linux may already have removed the target. Recover only with
 * NOREPLACE; uncertainty/rollback failure uses the existing backend fail-stop
 * policy rather than overwriting a foreign VMA or returning an invalid range. */
void move_anonymous(void* source, void* target, U bytes) {
#if defined(MREMAP_DONTUNMAP) && defined(MAP_FIXED_NOREPLACE)
  void* moved = ::mremap(source, bytes, bytes, MREMAP_MAYMOVE | MREMAP_FIXED | MREMAP_DONTUNMAP, target);
  if (moved == MAP_FAILED) {
    const int error = errno;
    void* placeholder = ::mmap(target, bytes, PROT_NONE,
                                MAP_PRIVATE | MAP_ANONYMOUS | MAP_NORESERVE | MAP_FIXED_NOREPLACE, -1, 0);
    if (placeholder != target) std::terminate();
    throw std::system_error(error, std::generic_category(), "move private_movable backing");
  }
  if (moved != target || ::mprotect(source, bytes, PROT_NONE)) std::terminate();
#else
  (void)source; (void)target; (void)bytes;
  throw std::invalid_argument("private_movable requires Linux DONTUNMAP/NOREPLACE support");
#endif
}
void attach_anonymous(AnonymousRoot& root, void* target, int prot) {
  same_process(root);
  require(!root.active, "private_movable permits only one active whole-root mapping");
  move_anonymous(root.parking, target, root.bytes);
  if (::mprotect(target, root.bytes, prot)) {
    const int error = errno;
    try { move_anonymous(target, root.parking, root.bytes); }
    catch (...) { std::terminate(); }
    if (::mprotect(root.parking, root.bytes, PROT_READ | PROT_WRITE)) std::terminate();
    throw std::system_error(error, std::generic_category(), "protect private_movable mapping");
  }
  root.active = true;
  root.location = target;
}
void park_anonymous(AnonymousRoot& root, void* target, int old_prot) {
  same_process(root);
  require(root.active && root.location == target, "private_movable active mapping differs");
  move_anonymous(target, root.parking, root.bytes);
  if (::mprotect(root.parking, root.bytes, PROT_READ | PROT_WRITE)) {
    const int error = errno;
    try { move_anonymous(root.parking, target, root.bytes); }
    catch (...) { std::terminate(); }
    if (::mprotect(target, root.bytes, old_prot)) std::terminate();
    throw std::system_error(error, std::generic_category(), "protect parked private_movable backing");
  }
  root.active = false;
  root.location = root.parking;
}
std::vector<U> anonymous_page_sizes(void* address, U bytes) {
  // Cold observation only, not a persistent THP guarantee. Do not attribute
  // merged-VMA totals to a subrange or infer physical pages from logical q.
  std::ifstream file("/proc/self/smaps");
  std::string line;
  bool exact = false;
  while (std::getline(file, line)) {
    if (line.find('-') != std::string::npos && line.find(':') > line.find(' ')) {
      std::istringstream row(line); std::string extent; row >> extent;
      const auto dash = extent.find('-');
      if (dash == std::string::npos) continue;
      const U a = std::stoull(extent.substr(0, dash), nullptr, 16),
              z = std::stoull(extent.substr(dash + 1), nullptr, 16);
      exact = a == reinterpret_cast<std::uintptr_t>(address) && z - a == bytes;
    } else if (exact && line.starts_with("AnonHugePages:")) {
      U kb = 0; std::istringstream row(line.substr(14)); row >> kb;
      if (!row || kb > bytes / 1024) return {};
      if (!kb) return {system_page()};
      if (kb * 1024 == bytes) return {huge_alignment()};
      return {system_page(), huge_alignment()};
    }
  }
  return {};
}
struct Descriptor {
  int fd = -1;
  std::string unlink_name;
  pid_t creator_pid = ::getpid();
  std::shared_ptr<void> lifetime;
  ~Descriptor() {
    if (fd >= 0)
      ::close(fd);
    // A forked child's local teardown closes its inherited fd, but the parent
    // still owns publication of this backing to later importers.
    if (!unlink_name.empty() && creator_pid == ::getpid())
      ::shm_unlink(unlink_name.c_str());
  }
};
class Backing final : public SharedBacking {
public:
  AllocationInfo allocation;
  std::shared_ptr<AnonymousRoot> anonymous;
  std::shared_ptr<Descriptor> descriptor;
  // Keep the immediate parent, not only its fd: it may own an arena slice
  // ticket in addition to the descriptor's root physical-capacity charge.
  std::shared_ptr<Backing> parent;
  std::shared_ptr<void> lifetime;
  std::string identity, export_name;
  U file_offset = 0;
  const AllocationInfo& info() const override {
    return allocation;
  }
  Result<std::any> export_handle() const override {
    if (!allocation.shareable || export_name.empty())
      return {Error{"not_shareable", "backing has no export name"}};
    return {std::any{LinuxBackingHandle{export_name, identity, allocation, file_offset}}};
  }
};
struct ReservationState {
  void* address = MAP_FAILED;
  U bytes = 0, q = 0;
  std::mutex mutex;
  std::map<U, U> mapped;
  pid_t creator = ::getpid();
  ~ReservationState() {
    if (creator == ::getpid() && address != MAP_FAILED)
      ::munmap(address, static_cast<std::size_t>(bytes));
  }
};
class Reservation final : public VirtualReservation {
public:
  std::shared_ptr<ReservationState> state;
  U size_bytes() const override {
    return state->bytes;
  }
  U page_bytes() const override {
    return state->q;
  }
};
int protection(std::span<const AccessMode> access) {
  if (access.empty())
    return PROT_READ | PROT_WRITE;
  int p = 0;
  for (auto a : access) {
    require(a == AccessMode::read || a == AccessMode::write, "unknown mapping right");
    const int bit = a == AccessMode::read ? PROT_READ : PROT_WRITE;
    require(!(p & bit), "duplicate mapping right");
    p |= bit;
  }
  return p;
}
class Mapping final : public MappedRange {
public:
  std::shared_ptr<ReservationState> reservation;
  std::shared_ptr<Backing> backing;
  MapRequest request{};
  int prot = PROT_READ | PROT_WRITE;
  std::atomic<U> version{1};
  ~Mapping() override {
    if (!reservation || reservation->creator != ::getpid())
      return;
    std::lock_guard lock(reservation->mutex);
    auto* address = static_cast<char*>(reservation->address) + request.target_offset_bytes;
    if (backing->anonymous) {
      std::lock_guard root_lock(backing->anonymous->mutex);
      try { park_anonymous(*backing->anonymous, address, prot); }
      catch (...) { std::terminate(); }
      reservation->mapped.erase(request.target_offset_bytes);
      return;
    }
    // Existing reservation, size and alignment were certified at map time.
    // mmap replacing an existing VMA atomically removes the physical mapping.
    const auto result = ::mmap(address, static_cast<std::size_t>(request.size_bytes), PROT_NONE,
                               MAP_PRIVATE | MAP_ANONYMOUS | MAP_FIXED | MAP_NORESERVE, -1, 0);
    if (result == MAP_FAILED)
      std::terminate(); // never leave a stale live mapping silently
    reservation->mapped.erase(request.target_offset_bytes);
  }
  Result<void*> data() const override {
    if (reservation->creator != ::getpid()) return {Error{"invalid_argument", "mapping cannot be inherited across fork"}};
    return {static_cast<char*>(reservation->address) + request.target_offset_bytes};
  }
  U generation() const override {
    return version.load();
  }
};
std::shared_ptr<Backing> as_backing(std::shared_ptr<SharedBacking> b) {
  auto p = std::dynamic_pointer_cast<Backing>(std::move(b));
  require(p != nullptr, "backing belongs to a different backend");
  return p;
}
void validate_map(const ReservationState& s, const Backing& b, const MapRequest& r) {
  require(s.creator == ::getpid(), "virtual reservation cannot be inherited across fork");
  require(end(r.target_offset_bytes, r.size_bytes) <= s.bytes &&
            end(r.backing_offset_bytes, r.size_bytes) <= b.info().mapped_bytes,
          "mapping outside reservation/backing");
  if (b.anonymous) {
    same_process(*b.anonymous);
    require(r.backing_offset_bytes == 0 && r.size_bytes == b.info().mapped_bytes,
            "private_movable mapping must cover its whole root");
  }
  const U q = std::max(s.q, b.info().page_bytes);
  require(s.q >= b.info().page_bytes && r.target_offset_bytes % q == 0 && r.backing_offset_bytes % q == 0 &&
            r.size_bytes % q == 0,
          "unaligned backing mapping");
}
void map_at(const ReservationState& s, const Backing& b, const MapRequest& r, int prot) {
  auto* address = static_cast<char*>(s.address) + r.target_offset_bytes;
  void* p = ::mmap(address, static_cast<std::size_t>(r.size_bytes), prot, MAP_SHARED | MAP_FIXED, b.descriptor->fd,
                   static_cast<off_t>(b.file_offset + r.backing_offset_bytes));
  os_require(p != MAP_FAILED, "map backing into reserved address");
}
void bind_address(void* p, U bytes, const std::vector<unsigned>& nodes) {
  if (!nodes.empty()) {
    const unsigned highest = *std::max_element(nodes.begin(), nodes.end());
    require(highest < 1048576, "NUMA identifier exceeds backend mask limit");
    constexpr auto bits = 8 * sizeof(unsigned long);
    std::vector<unsigned long> mask(highest / bits + 1);
    for (auto n : nodes)
      mask[n / bits] |= 1UL << (n % bits);
    // The Linux nodemask ABI reserves one sentinel bit beyond the usable mask.
    // Passing highest+1 drops that highest selected bit (node0 becomes empty).
    os_require(::syscall(SYS_mbind, p, static_cast<unsigned long>(bytes), MPOL_BIND, mask.data(),
                         static_cast<unsigned long>(mask.size() * bits) + 1, MPOL_MF_STRICT) == 0,
               "bind backing NUMA policy");
  }
 }
void bind_and_touch(int fd, U bytes, const std::vector<unsigned>& nodes) {
  void* p = ::mmap(nullptr, bytes, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
  os_require(p != MAP_FAILED, "materialize backing");
  struct Guard { void* p; U n; ~Guard() { ::munmap(p, n); } } guard{p, bytes};
  bind_address(p, bytes, nodes);
  const int admitted = ::posix_fallocate(fd, 0, static_cast<off_t>(bytes));
  if (admitted)
    throw std::system_error(admitted, std::generic_category(), "admit physical backing");
  // Force allocation now so capacity admission does not masquerade as physical
  // commitment. Later aliases refer to these already materialized shared pages.
  auto* dst = static_cast<volatile unsigned char*>(p);
  for (U i = 0; i < bytes; i += system_page())
    dst[i] = 0;
}
#endif
} // namespace
LinuxPageStore::LinuxPageStore(std::vector<LinuxPoolBinding> bindings)
  : bindings_(std::move(bindings)) {
  std::map<std::string, bool> seen;
  for (const auto& b : bindings_)
    require(!b.pool.empty() && !b.address_space.empty() && seen.emplace(b.pool, true).second,
            "duplicate or empty pool backend binding");
}
bool LinuxPageStore::private_movable_supported() {
#if defined(__linux__) && defined(MREMAP_DONTUNMAP) && defined(MAP_FIXED_NOREPLACE)
  static const bool supported = [] {
    const U n = system_page();
    void* source = ::mmap(nullptr, n, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS | MAP_NORESERVE, -1, 0);
    if (source == MAP_FAILED) return false;
    // Probe NOREPLACE without releasing an existing VMA, including old kernels
    // that ignore unknown mmap flags and return a different address.
    errno = 0;
    void* collision = ::mmap(source, n, PROT_NONE,
                              MAP_PRIVATE | MAP_ANONYMOUS | MAP_NORESERVE | MAP_FIXED_NOREPLACE, -1, 0);
    const bool noreplace = collision == MAP_FAILED && errno == EEXIST;
    if (collision != MAP_FAILED) ::munmap(collision, n);
    bool success = false;
    if (noreplace && ::madvise(source, n, MADV_DONTFORK) == 0) {
      *static_cast<volatile unsigned char*>(source) = 93;
      // Kernel-selected target means unsupported flags cannot destroy a caller
      // placeholder during this capability probe.
      // libc consumes the fifth argument for DONTUNMAP even without FIXED.
      // An explicit null selects the kernel target on every supported ABI.
      void* target = ::mremap(source, n, n, MREMAP_MAYMOVE | MREMAP_DONTUNMAP, nullptr);
      if (target != MAP_FAILED) {
        success = *static_cast<volatile unsigned char*>(target) == 93 && ::mprotect(source, n, PROT_NONE) == 0;
        ::munmap(target, n);
      }
    }
    ::munmap(source, n);
    return success;
  }();
  return supported;
#else
  return false;
#endif
}
Result<std::shared_ptr<SharedBacking>> LinuxPageStore::allocate(const BackingRequest& r,
                                                                std::shared_ptr<void> lifetime) const {
  return checked([&]() -> std::shared_ptr<SharedBacking> {
#if defined(__linux__)
    geometry(r.bytes, r.page_bytes);
    require(r.mode == BackingMode::private_memory || r.mode == BackingMode::shareable_remappable || r.mode == BackingMode::private_movable,
            "unknown backing mode");
    require(!r.id.empty() && !r.owner.empty(), "empty backing identity");
    require(r.bytes <= static_cast<U>(std::numeric_limits<off_t>::max()), "backing exceeds off_t range");
    auto out = std::make_shared<Backing>();
    if (r.mode == BackingMode::private_movable) {
      require(private_movable_supported(), "private_movable unsupported by this Linux kernel");
      require(r.export_to.empty(), "private_movable cannot be exported");
      const auto it = std::find_if(bindings_.begin(), bindings_.end(), [&](const auto& b) { return b.pool == r.pool; });
      require(it != bindings_.end(), "unknown pool backend binding");
      auto root = std::make_shared<AnonymousRoot>();
      root->bytes = r.bytes; root->lifetime = std::make_unique<std::shared_ptr<void>>(std::move(lifetime));
      root->parking = aligned_reservation(r.bytes, std::max(r.page_bytes, huge_alignment()));
      void* memory = root->parking;
      os_require(::mprotect(memory, r.bytes, PROT_READ | PROT_WRITE) == 0, "materialize private_movable backing");
      os_require(::madvise(memory, r.bytes, MADV_DONTFORK) == 0, "exclude private_movable from fork");
      bind_address(memory, r.bytes, it->numa_nodes);
      // Advisory only; ordinary physical pages remain a valid result.
      ::madvise(memory, r.bytes, MADV_HUGEPAGE);
      auto* dst = static_cast<volatile unsigned char*>(memory);
      for (U i = 0; i < r.bytes; i += system_page()) dst[i] = 0;
      root->location = root->parking;
      out->anonymous = std::move(root);
      std::ifstream boot("/proc/sys/kernel/random/boot_id"); std::string host; std::getline(boot, host);
      require(!host.empty(), "cannot certify host boot identity");
      const auto nonce = kernel_uuid("cannot mint private_movable kernel root identity");
      out->identity = "linux-anon:" + host + ":" + nonce;
      out->allocation = {r.id, it->address_space, r.bytes, r.bytes, r.page_bytes,
                         anonymous_page_sizes(memory, r.bytes), {{0, r.bytes, r.pool}}, false, true, 1, r.mode};
      return out;
    }
    out->descriptor = std::make_shared<Descriptor>();
    out->descriptor->lifetime = std::move(lifetime);
    if (r.mode == BackingMode::shareable_remappable) {
      // Independent local SDK DSOs share a PID, but not their static counters.
      // Kernel UUIDs preserve opaque publication names without a shared counter.
      for (unsigned attempt = 0; attempt < 100; ++attempt) {
        const auto name = "/vecops-" + std::to_string(::getpid()) + "-" +
                          kernel_uuid("cannot mint shareable kernel root identity");
        const int fd = ::shm_open(name.c_str(), O_RDWR | O_CREAT | O_EXCL | O_CLOEXEC, 0600);
        if (fd >= 0) {
          out->descriptor->fd = fd;
          out->descriptor->unlink_name = name;
          out->export_name = name;
          break;
        }
        if (errno != EEXIST)
          os_require(false, "create shared backing");
      }
      require(out->descriptor->fd >= 0, "shared backing name collision limit");
    } else {
      out->descriptor->fd =
        static_cast<int>(::syscall(SYS_memfd_create, "vecops-private", MFD_CLOEXEC | MFD_ALLOW_SEALING));
      os_require(out->descriptor->fd >= 0, "create private backing");
    }
    os_require(::ftruncate(out->descriptor->fd, static_cast<off_t>(r.bytes)) == 0, "size backing");
    const auto it = std::find_if(bindings_.begin(), bindings_.end(), [&](const auto& b) { return b.pool == r.pool; });
    require(it != bindings_.end(), "unknown pool backend binding");
    bind_and_touch(out->descriptor->fd, r.bytes, it->numa_nodes);
    out->identity = inode_identity(out->descriptor->fd);
    out->allocation = {r.id,
                       it->address_space,
                       r.bytes,
                       r.bytes,
                       r.page_bytes,
                       {system_page()},
                       {{0, r.bytes, r.pool}},
                       r.mode == BackingMode::shareable_remappable,
                       true,
                       1, r.mode};
    return out;
#else
    (void)r;
    (void)lifetime;
    throw std::invalid_argument("Linux page store unavailable on this platform");
#endif
  });
}
Result<std::shared_ptr<SharedBacking>> LinuxPageStore::import_backing(const LinuxBackingHandle& handle) const {
  return checked([&]() -> std::shared_ptr<SharedBacking> {
#if defined(__linux__)
    require(handle.name.starts_with("/vecops-") && handle.name.find('/', 1) == std::string::npos,
            "invalid shared backing name");
    require(handle.allocation.mode == BackingMode::shareable_remappable && handle.allocation.shareable && handle.allocation.remappable, "invalid shared backing metadata");
    geometry(handle.allocation.mapped_bytes, handle.allocation.page_bytes);
    auto out = std::make_shared<Backing>();
    out->descriptor = std::make_shared<Descriptor>();
    out->descriptor->fd = ::shm_open(handle.name.c_str(), O_RDWR | O_CLOEXEC, 0);
    os_require(out->descriptor->fd >= 0, "open shared backing");
    out->identity = inode_identity(out->descriptor->fd);
    require(out->identity == handle.physical_identity, "shared backing identity mismatch");
    struct stat st {};
    os_require(::fstat(out->descriptor->fd, &st) == 0, "stat shared backing");
    require(st.st_size >= 0 &&
              end(handle.backing_offset_bytes, handle.allocation.mapped_bytes) <= static_cast<U>(st.st_size),
            "shared backing size mismatch");
    require(handle.backing_offset_bytes % handle.allocation.page_bytes == 0, "unaligned exported backing offset");
    out->file_offset = handle.backing_offset_bytes;
    out->allocation = handle.allocation;
    out->export_name = handle.name;
    return out;
#else
    (void)handle;
    throw std::invalid_argument("Linux page store unavailable on this platform");
#endif
  });
}
Result<std::unique_ptr<VirtualReservation>> LinuxPageStore::reserve(U bytes, U q, U base_alignment) const {
  return checked([&]() -> std::unique_ptr<VirtualReservation> {
#if defined(__linux__)
    geometry(bytes, q);
    require(base_alignment == 0 || power2(base_alignment), "invalid virtual base alignment");
    const U alignment = std::max(q, base_alignment);
    auto state = std::make_shared<ReservationState>();
    const U preferred = bytes >= huge_alignment() ? std::max(alignment, huge_alignment()) : alignment;
    state->address = aligned_reservation(bytes, preferred);
    state->bytes = bytes;
    state->q = q;
    auto out = std::make_unique<Reservation>();
    out->state = std::move(state);
    return out;
#else
    (void)bytes;
    (void)q;
    (void)base_alignment;
    throw std::invalid_argument("Linux page store unavailable on this platform");
#endif
  });
}
Result<std::unique_ptr<MappedRange>> LinuxPageStore::map(VirtualReservation& target,
                                                         std::shared_ptr<SharedBacking> backing,
                                                         const MapRequest& request,
                                                         std::span<const AccessMode> access) const {
  return checked([&]() -> std::unique_ptr<MappedRange> {
#if defined(__linux__)
    auto* reservation = dynamic_cast<Reservation*>(&target);
    require(reservation != nullptr, "reservation belongs to different backend");
    auto b = as_backing(std::move(backing));
    auto out = std::make_unique<Mapping>();
    auto state = reservation->state;
    std::lock_guard lock(state->mutex);
    validate_map(*state, *b, request);
    auto next = state->mapped.lower_bound(request.target_offset_bytes);
    require(next == state->mapped.end() || end(request.target_offset_bytes, request.size_bytes) <= next->first,
            "mapping overlaps existing range");
    require(next == state->mapped.begin() || std::prev(next)->second <= request.target_offset_bytes,
            "mapping overlaps existing range");
    const int prot = protection(access);
    // Reserve bookkeeping before OS mutation, so allocation failure cannot orphan
    // a mapping with an unowned lifetime after successful MAP_FIXED.
    auto inserted =
      state->mapped.emplace(request.target_offset_bytes, end(request.target_offset_bytes, request.size_bytes));
    try {
      if (b->anonymous) {
        std::lock_guard root_lock(b->anonymous->mutex);
        attach_anonymous(*b->anonymous, static_cast<char*>(state->address) + request.target_offset_bytes, prot);
      } else map_at(*state, *b, request, prot);
    } catch (...) {
      state->mapped.erase(inserted.first);
      throw;
    }
    out->reservation = state;
    out->backing = std::move(b);
    out->request = request;
    out->prot = prot;
    return out;
#else
    (void)target;
    (void)backing;
    (void)request;
    (void)access;
    throw std::invalid_argument("Linux page store unavailable on this platform");
#endif
  });
}
Result<void> LinuxPageStore::replace_quiescent(MappedRange& target, std::shared_ptr<SharedBacking> backing,
                                               const MapRequest& request, std::span<const AccessMode> access) const {
  return checked([&] {
#if defined(__linux__)
    auto* range = dynamic_cast<Mapping*>(&target);
    require(range != nullptr, "mapping belongs to different backend");
    auto b = as_backing(std::move(backing));
    auto state = range->reservation;
    std::lock_guard lock(state->mutex);
    validate_map(*state, *b, request);
    require(range->request.target_offset_bytes == request.target_offset_bytes &&
              range->request.size_bytes == request.size_bytes,
            "rebind must preserve reserved range");
    require(range->version != std::numeric_limits<U>::max(), "mapping generation exhausted");
    const int prot = protection(access);
    auto old = range->backing;
    std::unique_lock<std::mutex> old_lock, new_lock;
    if (old->anonymous) old_lock = std::unique_lock<std::mutex>(old->anonymous->mutex, std::defer_lock);
    if (b->anonymous && b->anonymous != old->anonymous)
      new_lock = std::unique_lock<std::mutex>(b->anonymous->mutex, std::defer_lock);
    if (old_lock.mutex() && new_lock.mutex()) std::lock(old_lock, new_lock);
    else if (old_lock.mutex()) old_lock.lock();
    else if (new_lock.mutex()) new_lock.lock();
    auto* address = static_cast<char*>(state->address) + request.target_offset_bytes;
    if (b->anonymous && b->anonymous == old->anonymous) {
      same_process(*b->anonymous);
      os_require(::mprotect(address, request.size_bytes, prot) == 0, "protect existing private_movable mapping");
    } else {
      if (b->anonymous) {
        same_process(*b->anonymous);
        require(!b->anonymous->active, "private_movable already actively mapped");
      }
      // Parking failure restores its own target placeholder and leaves old
      // active data unchanged; do not begin replacing until parking succeeds.
      if (old->anonymous) park_anonymous(*old->anonymous, address, range->prot);
      try {
        if (b->anonymous) attach_anonymous(*b->anonymous, address, prot);
        else map_at(*state, *b, request, prot);
      } catch (...) {
        try {
          if (old->anonymous) attach_anonymous(*old->anonymous, address, range->prot);
          else map_at(*state, *old, range->request, range->prot);
        } catch (...) { std::terminate(); }
        throw;
      }
    }
    range->prot = prot;
    range->backing = std::move(b);
    range->request.backing_offset_bytes = request.backing_offset_bytes;
    ++range->version;
#else
    (void)target;
    (void)backing;
    (void)request;
    (void)access;
    throw std::invalid_argument("Linux page store unavailable on this platform");
#endif
  });
}
Result<std::string> LinuxPageStore::physical_identity(const SharedBacking& backing) {
  return checked([&] {
#if defined(__linux__)
    const auto* b = dynamic_cast<const Backing*>(&backing);
    require(b != nullptr, "backing belongs to different backend");
    if (b->anonymous) same_process(*b->anonymous);
    return b->identity;
#else
    (void)backing;
    throw std::invalid_argument("Linux page store unavailable on this platform");
    return std::string{};
#endif
  });
}

Result<std::shared_ptr<SharedBacking>> LinuxPageStore::slice(std::shared_ptr<SharedBacking> owner, U offset, U bytes,
                                                             const std::string& id,
                                                             std::shared_ptr<void> lifetime) const {
  return checked([&]() -> std::shared_ptr<SharedBacking> {
#if defined(__linux__)
    auto b = as_backing(std::move(owner));
    require(!b->anonymous, "private_movable cannot be sliced");
    require(!id.empty() && end(offset, bytes) <= b->allocation.mapped_bytes && offset % b->allocation.page_bytes == 0 &&
              bytes % b->allocation.page_bytes == 0,
            "invalid backing slice");
    auto out = std::make_shared<Backing>();
    out->descriptor = b->descriptor;
    out->identity = b->identity;
    out->export_name = b->export_name;
    out->file_offset = b->file_offset + offset;
    out->parent = b;
    out->lifetime = std::move(lifetime);
    out->allocation = b->allocation;
    out->allocation.id = id;
    out->allocation.requested_bytes = bytes;
    out->allocation.mapped_bytes = bytes;
    out->allocation.segments.clear();
    for (const auto& s : b->allocation.segments) {
      const U a = std::max(offset, s.offset_bytes), z = std::min(offset + bytes, s.offset_bytes + s.size_bytes);
      if (a < z)
        out->allocation.segments.push_back({a - offset, z - a, s.pool});
    }
    return out;
#else
    (void)owner;
    (void)offset;
    (void)bytes;
    (void)id;
    (void)lifetime;
    throw std::invalid_argument("Linux page store unavailable");
#endif
  });
}
Result<U> LinuxPageStore::physical_offset(const SharedBacking& backing) {
  return checked([&]() -> U {
#if defined(__linux__)
    const auto* b = dynamic_cast<const Backing*>(&backing);
    require(b != nullptr, "backing belongs to a different backend");
    if (b->anonymous) same_process(*b->anonymous);
    return b->file_offset;
#else
    (void)backing;
    throw std::invalid_argument("Linux page store unavailable");
#endif
  });
}
Result<void*> LinuxPageStore::address(VirtualReservation& reservation) {
  return checked([&]() -> void* {
#if defined(__linux__)
    auto* r = dynamic_cast<Reservation*>(&reservation);
    require(r != nullptr, "reservation belongs to a different backend");
    require(r->state->creator == ::getpid(), "virtual reservation cannot be inherited across fork");
    return r->state->address;
#else
    (void)reservation;
    throw std::invalid_argument("Linux page store unavailable");
#endif
  });
}
Result<MemoryDescriptor> LinuxPageStore::describe_external(const ExternalMemoryView& view, const std::string& symbol,
                                                           U generation) {
  return checked([&] {
#if defined(__linux__)
    require(!symbol.empty() && !view.address.address_space.empty() && view.owner != nullptr,
            "external memory needs symbol, address space and stable owner");
    const U stop = end(view.address.value, view.bytes);
    std::ifstream maps("/proc/self/maps");
    require(maps.good(), "cannot inspect external mappings");
    std::ifstream boot("/proc/sys/kernel/random/boot_id");
    std::string host;
    std::getline(boot, host);
    require(!host.empty(), "cannot certify external host identity");
    MemoryDescriptor out{MemoryRefKind::external, symbol, view.address.address_space, view.bytes, generation, {}};
    U cursor = view.address.value;
    std::string line;
    while (cursor < stop && std::getline(maps, line)) {
      std::istringstream row(line);
      std::string extent, permissions, off, device;
      U inode = 0;
      row >> extent >> permissions >> off >> device >> inode;
      const auto dash = extent.find('-');
      if (dash == std::string::npos)
        continue;
      const U a = std::stoull(extent.substr(0, dash), nullptr, 16),
              b = std::stoull(extent.substr(dash + 1), nullptr, 16);
      if (b <= cursor)
        continue;
      require(a <= cursor, "external memory crosses an unmapped hole");
      const U n = std::min(stop, b) - cursor;
      std::vector<AccessMode> access;
      if (permissions[0] == 'r')
        access.push_back(AccessMode::read);
      if (permissions[1] == 'w' && !view.read_only)
        access.push_back(AccessMode::write);
      require(!access.empty(), "external memory has no requested accessible bytes");
      std::optional<std::string> identity;
      U physical = cursor;
      if (permissions.size() >= 4 && permissions[3] == 'p') {
        // A private/COW VMA has process-private byte identity even when resident
        // pages happen to be shared. Writes cannot alter another private alias.
        identity = "linux-private:" + host + ":" + std::to_string(::getpid());
      } else if (inode) {
        const auto colon = device.find(':');
        require(colon != std::string::npos, "invalid shared mapping device");
        const auto dev = ::makedev(static_cast<unsigned>(std::stoul(device.substr(0, colon), nullptr, 16)),
                                   static_cast<unsigned>(std::stoul(device.substr(colon + 1), nullptr, 16)));
        identity = "linux:" + host + ":" + std::to_string(dev) + ":" + std::to_string(inode);
        physical = std::stoull(off, nullptr, 16) + cursor - a;
      }
      out.spans.push_back({cursor - view.address.value, n, identity, physical, std::move(access), view.known_pool});
      cursor += n;
    }
    require(cursor == stop, "external memory is not completely mapped");
    return out;
#else
    (void)view;
    (void)symbol;
    (void)generation;
    throw std::invalid_argument("Linux page store unavailable");
    return MemoryDescriptor{};
#endif
  });
}
Result<ResidencyObservation> LinuxPageStore::observe_external(
    const ExternalMemoryView& input, std::span<const std::string> accessible_pools) const {
  return checked([&] {
#if defined(__linux__)
    static std::atomic<U> sample{0};
    auto view=input;
    view.known_pool.reset();
    require(view.address.address_space=="host", "Linux residency requires host address space");
    auto described=describe_external(view,"residency",0);
    if(!described) throw std::invalid_argument(std::get<Error>(described.state).message);
    const auto& spans=described.value().spans;
    std::string identity;
    if(!spans.empty() && spans.front().physical_identity) {
      identity=*spans.front().physical_identity;
      for(const auto& span:spans)
        if(span.physical_identity!=std::optional<std::string>{identity}) {identity.clear();break;}
    }
    ResidencyObservation out{identity,0,{},true,++sample};
    const U stop=end(view.address.value,view.bytes), q=system_page();
    U cursor=view.address.value/q*q, resident=0;
    bool residency_known=true;
    constexpr U batch=4096;
    std::vector<void*> pages;
    std::vector<int> statuses;
    pages.reserve(batch);statuses.reserve(batch);
    while(cursor<stop) {
      const U count=std::min(batch,(stop-cursor-1)/q+1);
      pages.clear();statuses.assign(count,-EIO);
      for(U i=0;i<count;++i) pages.push_back(reinterpret_cast<void*>(cursor+i*q));
      // nodes=null queries placement only. It neither faults nor migrates data.
      const long result=::syscall(SYS_move_pages,0,static_cast<unsigned long>(count),pages.data(),
                                  nullptr,statuses.data(),0);
      if(result<0) {
        out.complete=false;
        residency_known=false;
        break;
      }
      for(U i=0;i<count;++i) {
        const U address=cursor+i*q;
        const U start=std::max(view.address.value,address), finish=stop-address<q?stop:address+q;
        const U bytes=finish-start;
        const int node=statuses[i];
        if(node<0) {
          out.complete=false;
          if(node!=-ENOENT) residency_known=false;
          continue;
        }
        resident+=bytes;
        std::optional<std::string> pool;
        bool ambiguous=false;
        for(const auto& binding:bindings_)
          if(binding.address_space=="host" &&
             std::find(binding.numa_nodes.begin(),binding.numa_nodes.end(),static_cast<unsigned>(node))!=binding.numa_nodes.end()) {
            if(pool && *pool!=binding.pool) {ambiguous=true;break;}
            pool=binding.pool;
          }
        if(!pool || ambiguous ||
           std::find(accessible_pools.begin(),accessible_pools.end(),*pool)==accessible_pools.end()) {
          out.complete=false;continue;
        }
        const U offset=start-view.address.value;
        if(!out.observed_placement.empty() && out.observed_placement.back().pool==*pool &&
           out.observed_placement.back().offset_bytes+out.observed_placement.back().size_bytes==offset)
          out.observed_placement.back().size_bytes+=bytes;
        else out.observed_placement.push_back({offset,bytes,*pool});
      }
      // Avoid overflow when the final queried page abuts the address-space end.
      if(stop-cursor<=count*q) break;
      cursor+=count*q;
    }
    if(residency_known) out.resident_bytes=resident;
    else out.resident_bytes.reset();
    return out;
#else
    (void)input;(void)accessible_pools;
    throw std::invalid_argument("Linux page residency unavailable");
    return ResidencyObservation{};
#endif
  });
}
} // namespace vecops::memory::model
