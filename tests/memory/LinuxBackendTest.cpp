// SPDX-FileCopyrightText: 2026 HenryRenYz and vecops contributors
// SPDX-License-Identifier: MIT
#include "vecops/memory/LinuxBackend.h"
#include "vecops/memory/LinuxDiscovery.h"
#include <gtest/gtest.h>
#include <cstring>
#include <algorithm>
#include <atomic>
#include <fstream>
#include <sstream>
#include <limits>
#if defined(__linux__)
#  include <fcntl.h>
#  include <sys/mman.h>
#  include <sys/resource.h>
#  include <csignal>
#  include <exception>
#  include <cstdarg>
#  include <sys/syscall.h>
#  include <sys/wait.h>
#  include <unistd.h>
#endif

#if defined(__linux__) && defined(MREMAP_DONTUNMAP) && defined(MAP_FIXED_NOREPLACE)
namespace {
// Deterministic syscall-boundary failures in this test executable only. All
// non-injected calls use the actual Linux syscall, including real page moves.
std::atomic<unsigned> injected_move_countdown{0};
bool injected_foreign_target = false;
void* foreign_target = nullptr;
}
extern "C" void* mremap(void* old_address, std::size_t old_bytes, std::size_t new_bytes, int flags, ...) noexcept {
  void* target = nullptr;
  if (flags & (MREMAP_FIXED | MREMAP_DONTUNMAP)) { va_list args; va_start(args, flags); target=va_arg(args,void*); va_end(args); }
  if ((flags & (MREMAP_FIXED | MREMAP_DONTUNMAP)) == (MREMAP_FIXED | MREMAP_DONTUNMAP)) {
    const auto count=injected_move_countdown.load();
    if(count && injected_move_countdown.fetch_sub(1)==1) {
      if(::munmap(target,new_bytes)) ::_exit(61);
      if(injected_foreign_target) {
        foreign_target=::mmap(target,new_bytes,PROT_READ|PROT_WRITE,MAP_PRIVATE|MAP_ANONYMOUS|MAP_FIXED_NOREPLACE,-1,0);
        if(foreign_target!=target) ::_exit(62);
        *static_cast<volatile unsigned char*>(foreign_target)=71;
      }
      errno=ENOMEM;return MAP_FAILED;
    }
  }
  return reinterpret_cast<void*>(::syscall(SYS_mremap,old_address,old_bytes,new_bytes,flags,target));
}
#endif

namespace vecops::memory::model {
namespace {
#if defined(__linux__)
std::uint64_t quantum() {
  return static_cast<std::uint64_t>(::sysconf(_SC_PAGESIZE));
}
BackingRequest request(std::uint64_t bytes = 0, BackingMode mode = BackingMode::shareable_remappable) {
  const auto q = quantum();
  return {"backing", "p0", "memory", bytes ? bytes : 4 * q, q, mode, {}};
}
LinuxPageStore store() {
  return LinuxPageStore{{{"memory", {}}}};
}

template <class T>
T take(Result<T> result) {
  if (!result)
    throw std::runtime_error(std::get<Error>(result.state).message);
  return std::move(result).value();
}

TEST(LinuxPageStore, SmallStronglyAlignedReservationKeepsBasePageMappingGeometry) {
  auto os = store();
  const auto q = quantum();
  for (const std::uint64_t alignment : {std::uint64_t{1} << 16, std::uint64_t{1} << 22}) {
    auto backing = take(os.allocate(request(q)));
    auto va = take(os.reserve(q, q, alignment));
    const auto base = reinterpret_cast<std::uintptr_t>(take(LinuxPageStore::address(*va)));
    EXPECT_EQ(base % std::max(q, alignment), 0u);
    // Mapping one q proves the base-alignment minimum did not become mapping q.
    auto mapping = take(os.map(*va, backing, {0, 0, q}));
    auto* data = static_cast<unsigned char*>(take(mapping->data()));
    EXPECT_EQ(reinterpret_cast<std::uintptr_t>(data), base);
    data[q - 1] = 73;
    EXPECT_EQ(data[q - 1], 73);
    EXPECT_EQ(backing->info().mapped_bytes, q);
    EXPECT_EQ(backing->info().page_bytes, q);
  }
  EXPECT_FALSE(os.reserve(q, q, 3));
  EXPECT_FALSE(os.reserve(q, q, std::uint64_t{1} << 63));
}

TEST(LinuxPageStore, StrictBindingIncludesHighestSelectedNodeInKernelMask) {
  auto topology = take(probe_linux_topology());
  ASSERT_FALSE(topology.allowed_memory_nodes.empty());
  const auto node = topology.allowed_memory_nodes.front();
  LinuxPageStore os{{{"memory", {node}}}};
  auto backing = take(os.allocate(request(quantum())));
  auto va = take(os.reserve(quantum(), quantum()));
  auto mapping = take(os.map(*va, backing, {0, 0, quantum()}));
  auto* bytes = static_cast<unsigned char*>(take(mapping->data()));
  bytes[0] = 89;
  EXPECT_EQ(bytes[0], 89);
}
TEST(LinuxPageStore, ResidencyObservesRealPagesWithoutTrustingClaimedPoolOrFaulting) {
  const auto q=quantum();
  void* address=::mmap(nullptr,3*q,PROT_READ|PROT_WRITE,MAP_PRIVATE|MAP_ANONYMOUS,-1,0);
  ASSERT_NE(address,MAP_FAILED);
  auto owner=std::shared_ptr<void>(address,[q](void* p) {::munmap(p,3*q);});
  auto topology=take(probe_linux_topology());
  std::vector<unsigned> nodes;
  for(const auto& node:topology.nodes) nodes.push_back(node.os_numa);
  LinuxPageStore os{{{"observed",nodes}}};
  std::string pools[]={"observed"};
  ExternalMemoryView view{"external",{"host",reinterpret_cast<std::uintptr_t>(address)+17},
    3*q-34,"invented-pool",owner,false,{}};
  unsigned char residency[3]{};
  ASSERT_EQ(::mincore(address,3*q,residency),0);
  for(auto bits:residency) EXPECT_EQ(bits&1,0);
  const auto untouched=take(os.observe_external(view,pools));
  EXPECT_FALSE(untouched.complete);
  EXPECT_TRUE(untouched.observed_placement.empty());
  ASSERT_EQ(::mincore(address,3*q,residency),0);
  for(auto bits:residency) EXPECT_EQ(bits&1,0);
  std::memset(address,93,3*q);
  const auto observed=take(os.observe_external(view,pools));
  EXPECT_GT(observed.sample_epoch,untouched.sample_epoch);
  if(observed.resident_bytes) {
    EXPECT_TRUE(observed.complete);
    EXPECT_EQ(*observed.resident_bytes,view.bytes);
    ASSERT_EQ(observed.observed_placement.size(),1u);
    EXPECT_EQ(observed.observed_placement[0].pool,"observed");
    EXPECT_EQ(observed.observed_placement[0].offset_bytes,0u);
    EXPECT_EQ(observed.observed_placement[0].size_bytes,view.bytes);
    LinuxPageStore ambiguous{{{"observed",nodes},{"other",nodes}}};
    std::string both[]={"observed","other"};
    const auto unknown=take(ambiguous.observe_external(view,both));
    EXPECT_FALSE(unknown.complete);
    EXPECT_TRUE(unknown.observed_placement.empty());
    const auto hidden=take(ambiguous.observe_external(view,pools));
    EXPECT_FALSE(hidden.complete); // Hiding a pool does not resolve physical ambiguity.
    EXPECT_TRUE(hidden.observed_placement.empty());
  } else {
    EXPECT_FALSE(observed.complete); // e.g. sandbox/OS denies move_pages query.
    EXPECT_TRUE(observed.observed_placement.empty());
  }
  EXPECT_EQ(static_cast<unsigned char*>(address)[q],93);
  view.owner.reset();
  EXPECT_FALSE(os.observe_external(view,pools));
}
TEST(LinuxPageStore, SharedAliasUsesOnePhysicalBackingWithOffsetMapping) {
  auto os = store();
  const auto q = quantum();
  auto backing = take(os.allocate(request()));
  auto va = take(os.reserve(4 * q, q)), other = take(os.reserve(2 * q, q));
  auto a = take(os.map(*va, backing, {q, q, 2 * q}));
  auto b = take(os.map(*other, backing, {q, 0, 2 * q}));
  auto* pa = static_cast<unsigned char*>(take(a->data()));
  auto* pb = static_cast<unsigned char*>(take(b->data()));
  EXPECT_EQ(reinterpret_cast<std::uintptr_t>(pa) % q, 0u);
  pa[7] = 91;
  EXPECT_EQ(pb[7], 91);
  pa[q + 13] = 72;
  EXPECT_EQ(pb[q + 13], 72);
  EXPECT_FALSE(os.map(*va, backing, {0, 2 * q, q})); // overlaps the first mapping
}
TEST(LinuxPageStore, ForkedChildDestructionDoesNotUnpublishParentBacking) {
  auto os = store();
  auto backing = take(os.allocate(request()));
  auto handle = std::any_cast<LinuxBackingHandle>(take(backing->export_handle()));
  const auto child = ::fork();
  ASSERT_GE(child, 0);
  if (child == 0) {
    backing.reset();
    ::_exit(0);
  }
  int status = 0;
  ASSERT_EQ(::waitpid(child, &status, 0), child);
  ASSERT_TRUE(WIFEXITED(status));
  ASSERT_EQ(WEXITSTATUS(status), 0);
  auto imported = os.import_backing(handle);
  ASSERT_TRUE(imported);
  EXPECT_EQ(take(LinuxPageStore::physical_identity(*imported.value())),
            take(LinuxPageStore::physical_identity(*backing)));
}
TEST(LinuxPageStore, LargeLogicalQuantumAlignsSmallPhysicalPages) {
  auto os = store();
  const std::uint64_t q = 2 * 1024 * 1024;
  auto req = request(q);
  req.page_bytes = q;
  auto backing = take(os.allocate(req));
  auto va = take(os.reserve(q, q));
  auto mapped = take(os.map(*va, backing, {0, 0, q}));
  EXPECT_EQ(reinterpret_cast<std::uintptr_t>(take(mapped->data())) % q, 0u);
  EXPECT_EQ(backing->info().page_bytes, q);
  EXPECT_EQ(backing->info().actual_page_sizes_bytes, (std::vector<std::uint64_t>{quantum()}));
}
TEST(LinuxPageStore, ReservationAndBackingAreRetainedByLiveMappings) {
  auto os = store();
  const auto q = quantum();
  auto backing = take(os.allocate(request()));
  auto va = take(os.reserve(q, q));
  auto mapping = take(os.map(*va, backing, {0, 0, q}));
  std::weak_ptr<SharedBacking> weak = backing;
  va.reset();
  backing.reset();
  EXPECT_FALSE(weak.expired());
  auto* p = static_cast<unsigned char*>(take(mapping->data()));
  p[5] = 34;
  EXPECT_EQ(p[5], 34);
  mapping.reset();
  EXPECT_TRUE(weak.expired());
}
TEST(LinuxPageStore, RebindPreservesAddressAndRejectsInvalidRangeBeforeMutation) {
  auto os = store();
  const auto q = quantum();
  auto one = take(os.allocate(request())), two = take(os.allocate(request()));
  auto va = take(os.reserve(q, q)), seed_va = take(os.reserve(q, q));
  auto a = take(os.map(*va, one, {0, 0, q}));
  auto seed = take(os.map(*seed_va, two, {0, 0, q}));
  auto* before = static_cast<unsigned char*>(take(a->data()));
  before[2] = 11;
  static_cast<unsigned char*>(take(seed->data()))[2] = 99;
  EXPECT_FALSE(os.replace_quiescent(*a, two, {0, q, q}));
  EXPECT_EQ(before[2], 11);
  EXPECT_EQ(a->generation(), 1u);
  ASSERT_TRUE(os.replace_quiescent(*a, two, {0, 0, q}));
  EXPECT_EQ(take(a->data()), before);
  EXPECT_EQ(before[2], 99);
  EXPECT_EQ(a->generation(), 2u);
}
TEST(LinuxPageStore, DestroyedMappingRestoresReservedAddressForAnotherBacking) {
  auto os = store();
  const auto q = quantum();
  auto backing = take(os.allocate(request()));
  auto va = take(os.reserve(q, q));
  auto mapping = take(os.map(*va, backing, {0, 0, q}));
  auto* first = take(mapping->data());
  mapping.reset();
  unsigned char residency = 0;
  EXPECT_EQ(::mincore(first, static_cast<std::size_t>(q), &residency), 0); // still a VMA
  auto next = take(os.map(*va, backing, {0, 0, q}));
  EXPECT_EQ(take(next->data()), first);
}
TEST(LinuxPageStore, ExportImportCertifiesIdentityAndRetainsPagesAfterExporterRelease) {
  auto os = store();
  const auto q = quantum();
  auto backing = take(os.allocate(request()));
  auto handle = std::any_cast<LinuxBackingHandle>(take(backing->export_handle()));
  auto imported = take(os.import_backing(handle));
  EXPECT_EQ(take(LinuxPageStore::physical_identity(*backing)), take(LinuxPageStore::physical_identity(*imported)));
  handle.physical_identity += "forged";
  EXPECT_FALSE(os.import_backing(handle));
  backing.reset();
  auto va = take(os.reserve(q, q));
  auto mapped = take(os.map(*va, imported, {0, 0, q}));
  auto* p = static_cast<unsigned char*>(take(mapped->data()));
  p[0] = 45;
  EXPECT_EQ(p[0], 45);
}
TEST(LinuxPageStore, SeparateProcessImportsWholeBackingAndMapsItsOwnOffsets) {
  auto os = store();
  const auto q = quantum();
  auto backing = take(os.allocate(request()));
  auto handle = std::any_cast<LinuxBackingHandle>(take(backing->export_handle()));
  auto va = take(os.reserve(q, q));
  auto mapped = take(os.map(*va, backing, {q, 0, q}));
  auto* p = static_cast<unsigned char*>(take(mapped->data()));
  p[8] = 17;
  const auto child = ::fork();
  ASSERT_GE(child, 0);
  if (child == 0) {
    auto imported = os.import_backing(handle);
    if (!imported)
      ::_exit(10);
    auto child_va = os.reserve(q, q);
    if (!child_va)
      ::_exit(11);
    auto alias = os.map(*child_va.value(), imported.value(), {q, 0, q});
    if (!alias)
      ::_exit(12);
    auto address = alias.value()->data();
    if (!address)
      ::_exit(13);
    auto* bytes = static_cast<unsigned char*>(address.value());
    if (bytes[8] != 17)
      ::_exit(14);
    bytes[8] = 87;
    ::_exit(0);
  }
  int status = 0;
  ASSERT_EQ(::waitpid(child, &status, 0), child);
  ASSERT_TRUE(WIFEXITED(status));
  EXPECT_EQ(WEXITSTATUS(status), 0);
  EXPECT_EQ(p[8], 87);
}
TEST(LinuxPageStore, KernelEnforcesReadonlyConsumerMapping) {
  auto os = store();
  const auto q = quantum();
  auto backing = take(os.allocate(request()));
  auto va = take(os.reserve(q, q));
  const AccessMode read = AccessMode::read;
  auto mapping = take(os.map(*va, backing, {0, 0, q}, std::span{&read, 1}));
  const auto* p = static_cast<const unsigned char*>(take(mapping->data()));
  EXPECT_EQ(p[0], 0);
  // /proc maps verifies the OS contract without an intentional sanitizer fault.
  std::ifstream maps("/proc/self/maps");
  std::string line;
  bool found = false;
  const auto address = reinterpret_cast<std::uintptr_t>(p);
  while (std::getline(maps, line)) {
    std::istringstream in(line);
    std::string extent, permissions;
    in >> extent >> permissions;
    const auto dash = extent.find('-');
    if (dash != std::string::npos && std::stoull(extent.substr(0, dash), nullptr, 16) <= address &&
        address < std::stoull(extent.substr(dash + 1), nullptr, 16)) {
      found = true;
      EXPECT_EQ(permissions[0], 'r');
      EXPECT_EQ(permissions[1], '-');
      break;
    }
  }
  EXPECT_TRUE(found);
}
TEST(LinuxPageStore, NestedSliceExportImportPreservesNonzeroFileOffset) {
  auto os = store();
  const auto q = quantum();
  auto root = take(os.allocate(request(6 * q)));
  auto root_va = take(os.reserve(6 * q, q));
  auto root_map = take(os.map(*root_va, root, {0, 0, 6 * q}));
  auto* data = static_cast<unsigned char*>(take(root_map->data()));
  for (std::uint64_t i = 0; i < 6; ++i)
    data[i * q + 9] = static_cast<unsigned char>(30 + i);
  auto first = take(os.slice(root, q, 4 * q, "first"));
  auto nested = take(os.slice(first, 2 * q, q, "nested"));
  EXPECT_EQ(take(LinuxPageStore::physical_offset(*nested)), 3 * q);
  EXPECT_EQ(take(LinuxPageStore::physical_identity(*nested)), take(LinuxPageStore::physical_identity(*root)));
  auto handle = std::any_cast<LinuxBackingHandle>(take(nested->export_handle()));
  EXPECT_EQ(handle.backing_offset_bytes, 3 * q);
  EXPECT_EQ(handle.allocation.mapped_bytes, q);
  ASSERT_EQ(handle.allocation.segments.size(), 1u);
  EXPECT_EQ(handle.allocation.segments[0].offset_bytes, 0u);
  auto imported = take(os.import_backing(handle));
  EXPECT_EQ(take(LinuxPageStore::physical_offset(*imported)), 3 * q);
  auto va = take(os.reserve(2 * q, q));
  auto mapped = take(os.map(*va, imported, {0, q, q}));
  auto* alias = static_cast<unsigned char*>(take(mapped->data()));
  EXPECT_EQ(alias[9], 33);
  alias[13] = 98;
  EXPECT_EQ(data[3 * q + 13], 98);
  auto bad = handle;
  bad.backing_offset_bytes = 1;
  EXPECT_FALSE(os.import_backing(bad));
  bad = handle;
  bad.backing_offset_bytes = 6 * q;
  EXPECT_FALSE(os.import_backing(bad));
  bad = handle;
  bad.backing_offset_bytes = std::numeric_limits<std::uint64_t>::max();
  EXPECT_FALSE(os.import_backing(bad));
  EXPECT_FALSE(os.slice(first, 4 * q, q, "outside"));
  EXPECT_FALSE(os.slice(first, 1, q, "unaligned"));
}
TEST(LinuxPageStore, SliceMappingRetainsParentRootChargeAndPerSliceTicket) {
  auto os = store();
  const auto q = quantum();
  auto charge = std::make_shared<int>(1), ticket = std::make_shared<int>(2);
  std::weak_ptr<int> weak_charge = charge, weak_ticket = ticket;
  auto root = take(os.allocate(request(), charge));
  std::weak_ptr<SharedBacking> weak_root = root;
  auto slice = take(os.slice(root, q, q, "arena-allocation", ticket));
  std::weak_ptr<SharedBacking> weak_slice = slice;
  auto va = take(os.reserve(q, q));
  auto mapped = take(os.map(*va, slice, {0, 0, q}));
  root.reset();
  slice.reset();
  charge.reset();
  ticket.reset();
  EXPECT_FALSE(weak_root.expired());
  EXPECT_FALSE(weak_slice.expired());
  EXPECT_FALSE(weak_charge.expired());
  EXPECT_FALSE(weak_ticket.expired());
  static_cast<unsigned char*>(take(mapped->data()))[0] = 47;
  mapped.reset();
  EXPECT_TRUE(weak_slice.expired());
  EXPECT_TRUE(weak_root.expired());
  EXPECT_TRUE(weak_ticket.expired());
  EXPECT_TRUE(weak_charge.expired());
}
TEST(LinuxPageStore, DescendantSliceRetainsAllAncestorTicketsUntilFinalMappingEnds) {
  auto os = store();
  const auto q = quantum();
  auto charge = std::make_shared<int>(), parent_ticket = std::make_shared<int>(),
       child_ticket = std::make_shared<int>();
  std::weak_ptr<int> root_life = charge, parent_life = parent_ticket, child_life = child_ticket;
  auto root = take(os.allocate(request(), charge));
  auto parent = take(os.slice(root, q, 2 * q, "parent", parent_ticket));
  auto child = take(os.slice(parent, q, q, "child", child_ticket));
  auto va = take(os.reserve(q, q));
  auto mapped = take(os.map(*va, child, {0, 0, q}));
  root.reset();
  parent.reset();
  child.reset();
  charge.reset();
  parent_ticket.reset();
  child_ticket.reset();
  EXPECT_FALSE(root_life.expired());
  EXPECT_FALSE(parent_life.expired());
  EXPECT_FALSE(child_life.expired());
  mapped.reset();
  EXPECT_TRUE(root_life.expired());
  EXPECT_TRUE(parent_life.expired());
  EXPECT_TRUE(child_life.expired());
}
ExternalMemoryView external(void* address, std::uint64_t bytes, std::shared_ptr<void> owner, bool read_only = false) {
  return {"diagnostic-import",
          {"host", reinterpret_cast<std::uintptr_t>(address)},
          bytes,
          "memory",
          std::move(owner),
          read_only,
          {}};
}
TEST(LinuxPageStore, ExternalSharedAliasesUseCertifiedIdentityAndFileOffsets) {
  auto os = store();
  const auto q = quantum();
  auto backing = take(os.allocate(request()));
  auto a_va = take(os.reserve(4 * q, q)), b_va = take(os.reserve(2 * q, q));
  std::shared_ptr<MappedRange> a = take(os.map(*a_va, backing, {0, 0, 4 * q}));
  std::shared_ptr<MappedRange> b = take(os.map(*b_va, backing, {q, 0, 2 * q}));
  auto* pa = static_cast<unsigned char*>(take(a->data()));
  auto* pb = static_cast<unsigned char*>(take(b->data()));
  auto da = take(LinuxPageStore::describe_external(external(pa + q + 17, q, a), "a", 9));
  auto db = take(LinuxPageStore::describe_external(external(pb + 17, q, b, true), "b", 10));
  ASSERT_EQ(da.spans.size(), 1u);
  ASSERT_EQ(db.spans.size(), 1u);
  EXPECT_EQ(da.spans[0].physical_identity, take(LinuxPageStore::physical_identity(*backing)));
  EXPECT_EQ(da.spans[0].physical_identity, db.spans[0].physical_identity);
  EXPECT_EQ(da.spans[0].physical_offset_bytes, q + 17);
  EXPECT_EQ(db.spans[0].physical_offset_bytes, q + 17);
  EXPECT_EQ(da.generation, 9u);
  EXPECT_EQ(db.generation, 10u);
  EXPECT_EQ(da.spans[0].access, (std::vector{AccessMode::read, AccessMode::write}));
  EXPECT_EQ(db.spans[0].access, (std::vector{AccessMode::read}));
  const auto read = AccessMode::read;
  auto ro_va = take(os.reserve(q, q));
  std::shared_ptr<MappedRange> ro = take(os.map(*ro_va, backing, {0, 0, q}, {&read, 1}));
  auto description = take(LinuxPageStore::describe_external(external(take(ro->data()), q, ro), "read-only", 1));
  EXPECT_EQ(description.spans[0].access, (std::vector{AccessMode::read}));
}
TEST(LinuxPageStore, ExternalSpanSplitsMixedPermissionVmasWithoutLosingIdentity) {
  auto os = store();
  const auto q = quantum();
  auto backing = take(os.allocate(request()));
  auto va = take(os.reserve(3 * q, q));
  std::shared_ptr<MappedRange> mapping = take(os.map(*va, backing, {0, 0, 3 * q}));
  auto* p = static_cast<unsigned char*>(take(mapping->data()));
  ASSERT_EQ(::mprotect(p + q, static_cast<std::size_t>(q), PROT_READ), 0);
  auto described = take(LinuxPageStore::describe_external(external(p, 3 * q, mapping), "mixed", 2));
  ASSERT_EQ(described.spans.size(), 3u);
  for (std::size_t i = 0; i < described.spans.size(); ++i) {
    EXPECT_EQ(described.spans[i].offset_bytes, i * q);
    EXPECT_EQ(described.spans[i].size_bytes, q);
    EXPECT_EQ(described.spans[i].physical_offset_bytes, i * q);
    EXPECT_EQ(described.spans[i].physical_identity, take(LinuxPageStore::physical_identity(*backing)));
  }
  EXPECT_EQ(described.spans[1].access, (std::vector{AccessMode::read}));
  EXPECT_EQ(described.spans[0].access.size(), 2u);
  EXPECT_EQ(described.spans[2].access.size(), 2u);
}
TEST(LinuxPageStore, ExternalPrivateCowVmasAreDistinctByteIdentities) {
  auto os = store();
  const auto q = quantum();
  auto root = take(os.allocate(request()));
  const auto handle = std::any_cast<LinuxBackingHandle>(take(root->export_handle()));
  const int fd = ::shm_open(handle.name.c_str(), O_RDWR | O_CLOEXEC, 0);
  ASSERT_GE(fd, 0);
  void* a =
    ::mmap(nullptr, static_cast<std::size_t>(q), PROT_READ | PROT_WRITE, MAP_PRIVATE, fd, static_cast<off_t>(q));
  void* b =
    ::mmap(nullptr, static_cast<std::size_t>(q), PROT_READ | PROT_WRITE, MAP_PRIVATE, fd, static_cast<off_t>(q));
  ::close(fd);
  ASSERT_NE(a, MAP_FAILED);
  ASSERT_NE(b, MAP_FAILED);
  auto release_page = [q](void* p) { ::munmap(p, static_cast<std::size_t>(q)); };
  std::shared_ptr<void> a_owner(a, release_page), b_owner(b, release_page);
  static_cast<unsigned char*>(a)[5] = 87;
  EXPECT_EQ(static_cast<unsigned char*>(b)[5], 0);
  auto da = take(LinuxPageStore::describe_external(external(a, q, a_owner), "cow-a", 1));
  auto db = take(LinuxPageStore::describe_external(external(b, q, b_owner), "cow-b", 1));
  ASSERT_EQ(da.spans.size(), 1u);
  ASSERT_EQ(db.spans.size(), 1u);
  EXPECT_TRUE(da.spans[0].physical_identity->starts_with("linux-private:"));
  EXPECT_EQ(da.spans[0].physical_identity, db.spans[0].physical_identity);
  EXPECT_NE(da.spans[0].physical_identity, take(LinuxPageStore::physical_identity(*root)));
  EXPECT_EQ(da.spans[0].physical_offset_bytes, reinterpret_cast<std::uintptr_t>(a));
  EXPECT_EQ(db.spans[0].physical_offset_bytes, reinterpret_cast<std::uintptr_t>(b));
  EXPECT_NE(da.spans[0].physical_offset_bytes, db.spans[0].physical_offset_bytes);
  auto sub =
    take(LinuxPageStore::describe_external(external(static_cast<unsigned char*>(a) + 13, q - 13, a_owner), "sub", 1));
  EXPECT_EQ(sub.spans[0].physical_offset_bytes, da.spans[0].physical_offset_bytes + 13);
}
TEST(LinuxPageStore, ExternalUnmappedInaccessibleAndOverflowRangesAreRejected) {
  auto os = store();
  const auto q = quantum();
  auto va = take(os.reserve(q, q));
  auto owner = std::make_shared<int>();
  auto* p = take(LinuxPageStore::address(*va));
  EXPECT_FALSE(LinuxPageStore::describe_external(external(p, q, owner), "reserved", 1));
  auto bad = external(p, q, owner);
  bad.owner.reset();
  EXPECT_FALSE(LinuxPageStore::describe_external(bad, "no-owner", 1));
  bad = external(p, 0, owner);
  EXPECT_FALSE(LinuxPageStore::describe_external(bad, "empty", 1));
  bad = external(p, q, owner);
  bad.address.value = std::numeric_limits<std::uint64_t>::max() - 1;
  EXPECT_FALSE(LinuxPageStore::describe_external(bad, "overflow", 1));
  EXPECT_FALSE(LinuxPageStore::describe_external(external(nullptr, q, owner), "unmapped", 1));
}

TEST(LinuxPageStore, PrivateMemoryCannotBeExportedAndInvalidGeometryFails) {
  auto os = store();
  const auto q = quantum();
  auto backing = take(os.allocate(request(0, BackingMode::private_memory)));
  EXPECT_FALSE(backing->export_handle());
  EXPECT_FALSE(os.reserve(0, q));
  EXPECT_FALSE(os.reserve(q, q / 2));
  EXPECT_FALSE(os.reserve(q - 1, q));
  auto bad = request();
  bad.pool = "missing";
  EXPECT_FALSE(os.allocate(bad));
  auto va = take(os.reserve(q, q));
  EXPECT_FALSE(os.map(*va, backing, {1, 0, q}));
  EXPECT_FALSE(os.map(*va, backing, {0, 0, 2 * q}));
}

TEST(LinuxPageStore, PrivateMovableWholeRootParkingContentAndLifetime) {
  // Compare with the real syscall so an ABI bug in the capability probe
  // cannot hide behind either this test's interposed libc wrapper or a skip.
  const auto probe_bytes = quantum();
  void* probe_source = ::mmap(nullptr, probe_bytes, PROT_READ | PROT_WRITE,
      MAP_PRIVATE | MAP_ANONYMOUS | MAP_NORESERVE, -1, 0);
  ASSERT_NE(probe_source, MAP_FAILED);
  ASSERT_EQ(::madvise(probe_source, probe_bytes, MADV_DONTFORK), 0);
  *static_cast<volatile unsigned char*>(probe_source) = 93;
  void* probe_target = reinterpret_cast<void*>(::syscall(SYS_mremap, probe_source,
      probe_bytes, probe_bytes, MREMAP_MAYMOVE | MREMAP_DONTUNMAP, nullptr));
  const bool kernel_supported = probe_target != MAP_FAILED;
  if (kernel_supported) {
    EXPECT_EQ(*static_cast<volatile unsigned char*>(probe_target), 93);
    EXPECT_EQ(::munmap(probe_target, probe_bytes), 0);
  }
  EXPECT_EQ(::munmap(probe_source, probe_bytes), 0);
  EXPECT_EQ(LinuxPageStore::private_movable_supported(), kernel_supported);
  if (!kernel_supported) GTEST_SKIP() << "kernel lacks DONTUNMAP";
  ASSERT_TRUE(LinuxPageStore::private_movable_supported());
  auto os = store(); const auto q = quantum(); const auto bytes = (2ULL << 20) + q;
  auto owner = std::make_shared<int>(7); std::weak_ptr<int> weak = owner;
  auto backing = take(os.allocate(request(bytes, BackingMode::private_movable), owner));
  owner.reset(); EXPECT_FALSE(weak.expired());
  EXPECT_EQ(backing->info().mode, BackingMode::private_movable);
  EXPECT_EQ(backing->info().page_bytes, q); EXPECT_EQ(backing->info().mapped_bytes, bytes);
  EXPECT_FALSE(backing->info().shareable); EXPECT_TRUE(backing->info().remappable);
  EXPECT_FALSE(backing->export_handle()); EXPECT_FALSE(os.slice(backing, 0, q, "slice"));
  auto address = take(os.reserve(bytes, q)); auto other = take(os.reserve(bytes, q));
  EXPECT_FALSE(os.map(*address, backing, {q, 0, bytes - q}));
  auto mapping = take(os.map(*address, backing, {0, 0, bytes}));
  auto* data = static_cast<unsigned char*>(take(mapping->data()));
  for (std::uint64_t i = 0; i < bytes; ++i) data[i] = static_cast<unsigned char>((i * 17 + 3) % 251);
  EXPECT_FALSE(os.map(*other, backing, {0, 0, bytes}));
  mapping.reset(); // Returns the same physical root to parking, preserving bytes.
  mapping = take(os.map(*other, backing, {0, 0, bytes}));
  data = static_cast<unsigned char*>(take(mapping->data()));
  for (std::uint64_t i = 0; i < bytes; ++i) ASSERT_EQ(data[i], (i * 17 + 3) % 251) << i;
  backing.reset(); EXPECT_FALSE(weak.expired());
  mapping.reset(); EXPECT_TRUE(weak.expired());
}
TEST(LinuxPageStore, PrivateMovableMbindAndWholeRootMovePreserveAllPagePlacement) {
  if(!LinuxPageStore::private_movable_supported())GTEST_SKIP();
  auto topology=take(probe_linux_topology());ASSERT_FALSE(topology.allowed_memory_nodes.empty());
  LinuxPageStore os{{{"memory",{topology.allowed_memory_nodes.front()}}}};
  const auto q=quantum();const auto bytes=(2ULL<<20)+q;
  auto backing=take(os.allocate(request(bytes,BackingMode::private_movable)));
  auto first=take(os.reserve(bytes,q)),second=take(os.reserve(bytes,q));
  auto mapping=take(os.map(*first,backing,{0,0,bytes}));
  auto observe=[&] {
    ExternalMemoryView view{};view.owner=backing;view.address={"host",reinterpret_cast<std::uintptr_t>(take(mapping->data()))};view.bytes=bytes;
    const std::string pools[]={"memory"};auto result=take(os.observe_external(view,pools));
    // Query precedes any consumer access: residency proves eager acquire rather
    // than lazily faulting the graph mapping on its first replay.
    ASSERT_TRUE(result.complete);ASSERT_TRUE(result.resident_bytes);EXPECT_EQ(*result.resident_bytes,bytes);
    ASSERT_EQ(result.observed_placement.size(),1u);EXPECT_EQ(result.observed_placement[0].pool,"memory");
  };
  observe();mapping.reset();mapping=take(os.map(*second,backing,{0,0,bytes}));observe();
}
TEST(LinuxPageStore, PrivateMovableRebindsBetweenAnonymousAndFdWithoutChangingAddress) {
  if (!LinuxPageStore::private_movable_supported()) GTEST_SKIP();
  auto os = store(); const auto q = quantum(); const auto n = 4*q;
  auto a = take(os.allocate(request(n, BackingMode::private_movable)));
  auto b = take(os.allocate(request(n, BackingMode::private_movable)));
  auto fd = take(os.allocate(request(n)));
  auto va = take(os.reserve(n, q)), vb = take(os.reserve(n, q));
  auto one = take(os.map(*va, a, {0, 0, n})); auto seed = take(os.map(*vb, b, {0, 0, n}));
  std::memset(take(one->data()), 13, n); std::memset(take(seed->data()), 27, n);
  EXPECT_FALSE(os.replace_quiescent(*one, b, {0, 0, n})); // Still active elsewhere.
  EXPECT_EQ(*static_cast<unsigned char*>(take(one->data())), 13);
  seed.reset();
  ASSERT_TRUE(os.replace_quiescent(*one, b, {0, 0, n}));
  EXPECT_EQ(take(one->data()), take(LinuxPageStore::address(*va)));
  EXPECT_EQ(*static_cast<unsigned char*>(take(one->data())), 27);
  EXPECT_FALSE(os.replace_quiescent(*one, a, {q, 0, n-q}));
  EXPECT_EQ(*static_cast<unsigned char*>(take(one->data())), 27);
  ASSERT_TRUE(os.replace_quiescent(*one, fd, {0, 0, n}));
  std::memset(take(one->data()), 38, n);
  ASSERT_TRUE(os.replace_quiescent(*one, a, {0, 0, n}));
  EXPECT_EQ(*static_cast<unsigned char*>(take(one->data())), 13);
  auto parked_b = take(os.map(*vb, b, {0, 0, n}));
  EXPECT_EQ(*static_cast<unsigned char*>(take(parked_b->data())), 27);
}
TEST(LinuxPageStore, PrivateMovableReadonlyAndForkedDestructionNeverTouchChildReplacement) {
  if (!LinuxPageStore::private_movable_supported()) GTEST_SKIP();
  auto os = store(); const auto q = quantum(); const auto parent=::getpid();
  auto lifetime=std::shared_ptr<int>(new int(7), [parent](int* p) { if (::getpid()!=parent) std::terminate(); delete p; });
  auto backing = take(os.allocate(request(q, BackingMode::private_movable),lifetime));lifetime.reset();
  auto va = take(os.reserve(q, q)); const AccessMode rights[]={AccessMode::read};
  auto mapping = take(os.map(*va, backing, {0, 0, q}, rights));
  auto* data = static_cast<unsigned char*>(take(mapping->data()));
  ExternalMemoryView e{}; e.owner=backing; e.address={"host", reinterpret_cast<std::uintptr_t>(data)}; e.bytes=q;
  auto described=take(LinuxPageStore::describe_external(e,"read-only",1));
  EXPECT_EQ(described.spans.front().access, (std::vector<AccessMode>{AccessMode::read}));
  e.owner.reset();
  const auto child=::fork(); ASSERT_GE(child,0);
  if (!child) {
    unsigned char resident=0;
    if (::mincore(data,q,&resident)!=-1 || errno!=ENOMEM) ::_exit(10);
    if (mapping->data() || LinuxPageStore::address(*va)) ::_exit(11);
    void* reused=::mmap(data,q,PROT_READ|PROT_WRITE,MAP_PRIVATE|MAP_ANONYMOUS|MAP_FIXED_NOREPLACE,-1,0);
    if (reused!=data) ::_exit(12);
    *data=73;
    mapping.reset(); va.reset(); backing.reset();
    if (*data!=73) ::_exit(13);
    ::munmap(data,q); ::_exit(0);
  }
  int status=0; ASSERT_EQ(::waitpid(child,&status,0),child); ASSERT_TRUE(WIFEXITED(status)); EXPECT_EQ(WEXITSTATUS(status),0);
  EXPECT_EQ(*data,0);
}
TEST(LinuxPageStore, PrivateMovableRealFailedFixedMoveNeverReturnsWithLostReservation) {
  if (!LinuxPageStore::private_movable_supported()) GTEST_SKIP();
  ASSERT_EXIT(([] {
    std::set_terminate([] { ::_exit(78); });
    auto os=store();const auto n=8ULL<<20;const auto q=quantum();
    auto backing=take(os.allocate(request(n,BackingMode::private_movable)));
    auto target=take(os.reserve(n,q));
    std::ifstream input("/proc/self/status");std::string line;std::uint64_t vm=0;
    while(std::getline(input,line))if(line.starts_with("VmSize:")){std::istringstream row(line.substr(7));row>>vm;break;}
    struct rlimit old{}; if(!vm||::getrlimit(RLIMIT_AS,&old))::_exit(20);
    struct rlimit limit{vm*1024-n/2,old.rlim_max};
    if(::setrlimit(RLIMIT_AS,&limit))::_exit(21);
    // FIXED can remove target before failure. NOREPLACE recovery is itself
    // prohibited by the lowered limit, so fail-stop is the only safe outcome.
    auto moved=os.map(*target,backing,{0,0,n});(void)moved;::_exit(22);
  }()), ::testing::ExitedWithCode(78), "");
}

#if defined(MREMAP_DONTUNMAP) && defined(MAP_FIXED_NOREPLACE)
TEST(LinuxPageStore, PrivateMovableFailedIncomingMoveRestoresOldAnonymousContentAndGeneration) {
  if(!LinuxPageStore::private_movable_supported())GTEST_SKIP();
  auto os=store();const auto n=4*quantum();
  auto first=take(os.allocate(request(n,BackingMode::private_movable)));
  auto second=take(os.allocate(request(n,BackingMode::private_movable)));
  auto va=take(os.reserve(n,quantum())),other=take(os.reserve(n,quantum()));
  auto mapping=take(os.map(*va,first,{0,0,n}));auto seed=take(os.map(*other,second,{0,0,n}));
  std::memset(take(mapping->data()),31,n);std::memset(take(seed->data()),57,n);seed.reset();
  const auto generation=mapping->generation();auto* address=take(mapping->data());
  injected_move_countdown=2; // old root parks successfully, incoming root fails.
  auto result=os.replace_quiescent(*mapping,second,{0,0,n});
  EXPECT_FALSE(result);EXPECT_EQ(injected_move_countdown.load(),0u);
  EXPECT_EQ(mapping->generation(),generation);EXPECT_EQ(take(mapping->data()),address);
  for(std::uint64_t i=0;i<n;++i)ASSERT_EQ(static_cast<unsigned char*>(address)[i],31);
  auto new_seed=take(os.map(*other,second,{0,0,n}));
  for(std::uint64_t i=0;i<n;++i)ASSERT_EQ(static_cast<unsigned char*>(take(new_seed->data()))[i],57);
}
TEST(LinuxPageStore, PrivateMovableForeignTargetOnFailureIsNotOverwritten) {
  if(!LinuxPageStore::private_movable_supported())GTEST_SKIP();
  ASSERT_EXIT(([] {
    std::set_terminate([] {::_exit(foreign_target && *static_cast<volatile unsigned char*>(foreign_target)==71 ? 79 : 80);});
    auto os=store();const auto n=4*quantum();auto backing=take(os.allocate(request(n,BackingMode::private_movable)));
    auto va=take(os.reserve(n,quantum()));injected_foreign_target=true;injected_move_countdown=1;
    auto result=os.map(*va,backing,{0,0,n});(void)result;::_exit(81);
  }()),::testing::ExitedWithCode(79),"");
}
#endif
#endif
} // namespace
} // namespace vecops::memory::model
