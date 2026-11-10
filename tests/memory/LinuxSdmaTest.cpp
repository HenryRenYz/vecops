// SPDX-FileCopyrightText: 2026 HenryRenYz and vecops contributors
// SPDX-License-Identifier: MIT
#include "vecops/memory/LinuxSdma.h"
#include "vecops/memory/LinuxRuntime.h"
#include <gtest/gtest.h>
#include <atomic>
#include <cstring>
#include <future>
#include <thread>
#include <unistd.h>
namespace vecops::memory::model {
namespace {
using U = std::uint64_t;
template<class T> T get(Result<T> r) {
  if (!r) throw std::runtime_error(std::get<Error>(r.state).message);
  return std::move(r).value();
}
void get(Result<void> r) { if (!r) throw std::runtime_error(r.error->message); }
struct Control {
  bool submit_error = false, complete = true, query_error = false, copy = true;
  unsigned pin_failure = 0, pins = 0, unpins = 0, submissions = 0, destroyed = 0;
  std::vector<std::pair<U,U>> registrations;
  std::vector<U> chunks;
};
class Driver final : public LinuxSdmaDriver {
  std::shared_ptr<Control> c_;
public:
  explicit Driver(std::shared_ptr<Control> c) : c_(std::move(c)) {}
  ~Driver() override { ++c_->destroyed; }
  Result<U> pin(void* p, std::uint32_t n) override {
    ++c_->pins;
    if (c_->pins == c_->pin_failure) return {Error{"test.pin", "injected pin failure"}};
    c_->registrations.emplace_back(reinterpret_cast<std::uintptr_t>(p), n);
    return {U(c_->pins)};
  }
  Result<void> unpin(U) override { ++c_->unpins; return {}; }
  Result<void> submit(void* t, const void* s, std::uint32_t n) override {
    ++c_->submissions; c_->chunks.push_back(n);
    if (c_->copy) std::memcpy(t,s,n);
    if (c_->submit_error) return {Error{"test.partial", "partial enqueue"}};
    return {};
  }
  Result<bool> completed() override {
    if (c_->query_error) return {Error{"test.query", "device still uncertain"}};
    return {c_->complete};
  }
};
std::shared_ptr<LinuxTransferExecutor> executor(std::shared_ptr<Control> c, U chunk = 1024) {
  LinuxSdmaOptions o; o.chunk_bytes = chunk;
  return get(make_linux_sdma_executor(o, [c]() -> Result<std::unique_ptr<LinuxSdmaDriver>> {
    return {std::make_unique<Driver>(c)};
  }));
}
EventStatus finish(const std::shared_ptr<LinuxTransferOperation>& op) {
  for (unsigned i=0;i<100;++i) {
    auto s=get(op->query());
    if (s.accesses_stopped) return s;
  }
  throw std::runtime_error("test operation did not finish");
}
TEST(LinuxSdma, ChunkedCopyPinsPageEnvelopesAndReleasesAfterCompletion) {
  auto c=std::make_shared<Control>(); auto owner=std::make_shared<int>(1);
  std::vector<unsigned char> a(9000,0x73),b(9000);
  auto op=get(executor(c)->copy(b.data()+7,a.data()+3,8000,owner));
  EXPECT_EQ(c->unpins,0u);
  EXPECT_EQ(finish(op).state,EventState::succeeded);
  EXPECT_EQ(std::memcmp(b.data()+7,a.data()+3,8000),0);
  EXPECT_EQ(c->submissions,8u); EXPECT_EQ(c->unpins,c->pins);
  for(auto [p,n]:c->registrations) { EXPECT_EQ(p%4096,0u); EXPECT_EQ(n%4096,0u); }
  EXPECT_EQ(op->statistics().pinned_bytes,0u);
}
TEST(LinuxSdma, FailedSubmitDoesNotUnpinBeforeLaterQuiescence) {
  auto c=std::make_shared<Control>(); c->submit_error=true; c->complete=false;
  auto owner=std::make_shared<int>(1); std::weak_ptr<int> weak=owner;
  std::vector<char> a(8192),b(8192);
  auto op=get(executor(c)->copy(b.data(),a.data(),8192,owner)); owner.reset();
  auto s=get(op->query()); EXPECT_EQ(s.state,EventState::failed); EXPECT_FALSE(s.accesses_stopped);
  c->query_error=true; s=get(op->query()); EXPECT_FALSE(s.accesses_stopped); EXPECT_EQ(c->unpins,0u);
  EXPECT_FALSE(weak.expired()); c->query_error=false; c->complete=true;
  s=get(op->query()); EXPECT_EQ(s.state,EventState::failed); EXPECT_TRUE(s.accesses_stopped);
  EXPECT_TRUE(weak.expired()); EXPECT_EQ(c->unpins,c->pins); EXPECT_EQ(c->submissions,1u);
}
TEST(LinuxSdma, PreparationFailureHasNoDeviceWorkAndRollsBackPins) {
  auto c=std::make_shared<Control>(); c->pin_failure=2;
  std::vector<char>a(8192),b(8192);
  EXPECT_FALSE(executor(c)->copy(b.data(),a.data(),8192,std::make_shared<int>(1)));
  EXPECT_EQ(c->submissions,0u); EXPECT_EQ(c->unpins,1u); EXPECT_EQ(c->destroyed,1u);
}
TEST(LinuxSdma, PinSegmentsRespectUint32ForLargeRanges) {
  auto c=std::make_shared<Control>(); c->copy=false;
  auto op=get(executor(c,UINT32_MAX)->copy(reinterpret_cast<void*>(U(1)<<36),
      reinterpret_cast<void*>(U(1)<<40),(U(1)<<32)+4096,std::make_shared<int>(1)));
  EXPECT_EQ(c->pins,4u);
  for(auto [p,n]:c->registrations) { EXPECT_EQ(p%4096,0u); EXPECT_LE(n,UINT32_MAX); EXPECT_GT(n,0u); }
  EXPECT_EQ(finish(op).state,EventState::succeeded); EXPECT_EQ(c->submissions,2u);
}
TEST(LinuxSdma, InvalidOptionsAndOverlapAreRejected) {
  LinuxSdmaOptions o; o.page_bytes=3; EXPECT_FALSE(make_linux_sdma_executor(o));
  o={}; o.chunk_bytes=U(UINT32_MAX)+1; EXPECT_FALSE(make_linux_sdma_executor(o));
  auto c=std::make_shared<Control>(); std::vector<char>a(8192);
  EXPECT_FALSE(executor(c)->copy(a.data()+10,a.data(),4096,std::make_shared<int>(1)));
  EXPECT_EQ(c->pins,0u);
  if (!linux_sdma_sdk_compiled()) {
    EXPECT_FALSE(make_linux_sdma_executor());
  }
}
struct RuntimeFixture {
  U q=U(::sysconf(_SC_PAGESIZE));
  std::unique_ptr<MemoryRuntime> runtime;
  ResourceView view;
  std::shared_ptr<PreparedAddresses> address;
  std::shared_ptr<BackingSet> backings;
  std::unique_ptr<MemoryBinding> binding;
  RuntimeFixture(std::shared_ptr<Control> c) {
    LinuxDiscovery d;
    d.graph.agents={{"host",{"host_load_store"},"host-as"},{"engine",{"dma","copy"},{}}};
    d.graph.capacity_groups={{"cap",64*q}};
    d.graph.pools={{"pool",64*q,"host-as",q,{q},{},{},"cap"}};
    d.graph.access={{"direct","opaque","host","pool",{AccessMode::read,AccessMode::write},{},{},MappingMode::direct,{},{},{},{},{}}};
    d.graph.methods={{"device-copy",true,{},{},32,{{"route","engine","host","pool","pool",{}}}}};
    d.scope={{"host","engine"},{"pool"},{"pool"},{{"host",1},{"engine",1}}};
    d.pool_bindings={{"pool",{}}}; d.pool_available_bytes={{"pool",64*q}};
    LinuxRuntimeOptions o; o.participant="p"; o.transfer_wait_ms=1;
    o.transfer_queues={{"device-q",{"engine","host",{"device-copy"},executor(c)}}};
    runtime=get(make_linux_runtime(d,o)); view=get(runtime->view(get(runtime->current_scope())));
    MemoryRecipe r{"p",{},1,{},{{{"va","host-as",2*q,q}},
      {{"src","va",0,q,{"p","host-as",q,{{0,q,"pool"}}},{}},
       {"dst","va",q,q,{"p","host-as",q,{{0,q,"pool"}}},{}}}},
      {{"root","p","pool",2*q,q,BackingMode::shareable_remappable,{}}},
      {{"va",0,"root",0,2*q,{AccessMode::read,AccessMode::write}}}};
    address=get(runtime->prepare_addresses(view,r.addresses));
    auto lease=get(runtime->reserve(view,r,runtime->snapshot(view,{}).epoch,{}));
    backings=get(runtime->acquire_backing(view,r,*lease));
    binding=get(runtime->bind(view,r,address,backings));
    std::memset(get(binding->slot("src")),0x47,q);
  }
  Submission submit() {
    MemoryCommand command{"c",MemoryOpcode::copy,ByteRangeRef{MemoryRefKind::slot,"src",0,q},
      ByteRangeRef{MemoryRefKind::slot,"dst",0,q},0,"device-copy",{},"device-q",{}, {}};
    InvocationContext context{{"test",binding->generation()},{},{}};
    return get(runtime->submit(view,{{command}},*binding,context));
  }
};
TEST(LinuxSdma, RuntimeExecutesExplicitDeviceMethodWithoutCpuFallback) {
  auto c=std::make_shared<Control>(); RuntimeFixture f(c);
  auto out=f.submit(); EXPECT_FALSE(out.submission_error);
  EXPECT_EQ(get(f.runtime->events().query(out.finished)).state,EventState::succeeded);
  EXPECT_EQ(std::memcmp(get(f.binding->slot("src")),get(f.binding->slot("dst")),f.q),0);
  EXPECT_EQ(c->submissions,4u);
  get(f.runtime->events().release(std::span<const CompletionToken>(&out.finished,1)));
}
TEST(LinuxSdma, RuntimeTimeoutDoesNotFallbackOrReleaseUntilStopped) {
  auto c=std::make_shared<Control>(); c->complete=false; c->copy=false;
  RuntimeFixture f(c); std::memset(get(f.binding->slot("dst")),0x15,f.q);
  auto out=f.submit(); ASSERT_TRUE(out.submission_error);
  EXPECT_EQ(out.submission_error->code,"sdma_quiescence_unknown");
  EXPECT_EQ(*static_cast<unsigned char*>(get(f.binding->slot("dst"))),0x15);
  EXPECT_EQ(c->unpins,0u); EXPECT_EQ(c->submissions,1u);
  c->complete=true;
  auto states=get(poll_linux_transfer_quarantine()); ASSERT_EQ(states.size(),1u);
  EXPECT_TRUE(states.front().status.accesses_stopped);
  EXPECT_EQ(states.front().status.state,EventState::failed);
  EXPECT_EQ(c->submissions,1u); // no remaining chunks after the failed command
  get(f.runtime->events().release(std::span<const CompletionToken>(&out.finished,1)));
}
TEST(LinuxSdma, QuarantineUsesStableReferenceAfterCallerReleasesFinishedToken) {
  auto c=std::make_shared<Control>(); c->submit_error=true; c->complete=false;
  RuntimeFixture f(c); auto out=f.submit(); ASSERT_TRUE(out.submission_error);
  c->complete=true;
  EXPECT_TRUE(get(f.runtime->events().query(out.finished)).accesses_stopped);
  get(f.runtime->events().release(std::span<const CompletionToken>(&out.finished,1)));
  auto states=get(poll_linux_transfer_quarantine()); ASSERT_EQ(states.size(),1u);
  EXPECT_TRUE(states.front().status.accesses_stopped);
  EXPECT_TRUE(get(poll_linux_transfer_quarantine()).empty());
}
TEST(LinuxSdma, RuntimeDestructionPreservesUnknownWorkAndExplicitPollRecovers) {
  auto c=std::make_shared<Control>(); c->submit_error=true; c->complete=false;
  {
    RuntimeFixture f(c); auto out=f.submit(); ASSERT_TRUE(out.submission_error);
    EXPECT_FALSE(get(f.runtime->events().query(out.finished)).accesses_stopped);
    auto stats=get(poll_linux_transfer_quarantine()); ASSERT_EQ(stats.size(),1u);
    EXPECT_EQ(stats[0].retained.payload_bytes,f.q); EXPECT_EQ(stats[0].retained.registrations,2u);
  }
  EXPECT_EQ(c->unpins,0u); EXPECT_EQ(c->destroyed,0u);
  std::vector<std::thread> pollers;
  std::atomic<unsigned> observed{0};
  for(unsigned i=0;i<8;++i) pollers.emplace_back([&] {
    for(unsigned n=0;n<10;++n) {
      auto states=get(poll_linux_transfer_quarantine());
      if(states.size()==1 && !states.front().status.accesses_stopped) ++observed;
    }
  });
  for(auto& t:pollers)t.join();
  EXPECT_EQ(observed,80u);
  auto first=get(poll_linux_transfer_quarantine()); ASSERT_EQ(first.size(),1u);
  EXPECT_FALSE(first[0].status.accesses_stopped);
  c->complete=true;
  auto done=get(poll_linux_transfer_quarantine()); ASSERT_EQ(done.size(),1u);
  EXPECT_TRUE(done[0].status.accesses_stopped); EXPECT_EQ(done[0].status.state,EventState::failed);
  EXPECT_TRUE(get(poll_linux_transfer_quarantine()).empty());
  EXPECT_EQ(c->unpins,c->pins); EXPECT_EQ(c->destroyed,1u);
}
} // namespace
} // namespace vecops::memory::model
