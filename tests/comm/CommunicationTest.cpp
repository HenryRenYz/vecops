// SPDX-FileCopyrightText: 2026 HenryRenYz and vecops contributors
// SPDX-License-Identifier: MIT
// Standalone fork test: no Torch/Python/GTest or external transport dependency.
#include "vecops/comm/Communication.h"
#include "vecops/comm/details/ReplicaRouting.h"
#include "vecops/comm/Copy.h"
#include "vecops/memory/LinuxRuntime.h"
#include "vecops/memory/details/SharedCapacityAuthority.h"
#include <filesystem>
#include <fstream>
#include <algorithm>
#include <array>
#include <atomic>
#include <barrier>
#include <chrono>
#include <cstring>
#include <cstdlib>
#include <iostream>
#include <stdexcept>
#include <thread>
#include <mutex>
#include <source_location>
#include <sys/mman.h>
#include <sys/wait.h>
#include <sys/syscall.h>
#include <sys/stat.h>
#include <linux/mempolicy.h>
#include <unistd.h>
using namespace vecops::comm;
namespace vecops::comm::detail::portable { const CopyBackend& backend() noexcept; }
#if defined(VECOPS_TEST_COMM_COPY_AVX2)
namespace vecops::comm::detail::avx2 { const CopyBackend& backend() noexcept; }
#endif
namespace {
constexpr int ranks=3, phases=128;
struct Bootstrap {
  struct Record {std::atomic<int> ready{0};char value[1024]{};} values[ranks][phases];
  std::atomic<int> cold_provision_entered{0},cold_reader_stopped{0};
};
void check(bool result,const std::source_location where=std::source_location::current()) {
  if(!result)throw std::runtime_error("standalone communication assertion at line "+std::to_string(where.line()));
}
void replica_routing() {
  const std::vector<uint64_t> bytes(8,1024);
  const std::vector<int> contiguous{0,0,0,0,1,1,1,1};
  check(detail::balanced_relay_targets(contiguous,0,bytes)==std::vector<int>({-1,-1,-1,-1,0,1,2,3}));
  const std::vector<int> interleaved{0,1,0,1,0,1,0,1};
  const std::vector<int> expected{-1,0,-1,2,-1,4,-1,6};
  for(int rank:{0,2,4,6})check(detail::balanced_relay_targets(interleaved,rank,bytes)==expected);
  check(detail::balanced_relay_targets(interleaved,1,bytes)==std::vector<int>({1,-1,3,-1,5,-1,7,-1}));
  // Unequal rank counts and payloads, with non-dense domain labels.
  const std::vector<int> unequal{8,9,8,9,9};
  const std::vector<uint64_t> ragged{10,100,10,20,10};
  check(detail::balanced_relay_targets(unequal,0,ragged)==std::vector<int>({-1,0,-1,2,2}));
  bool rejected=false;
  try{detail::balanced_relay_targets(unequal,-1,ragged);}catch(const std::invalid_argument&){rejected=true;}
  check(rejected);
  const std::vector<int> one{0,1,1};
  const std::vector<uint64_t> overflow{0,UINT64_MAX,1};rejected=false;
  try{detail::balanced_relay_targets(one,0,overflow);}catch(const std::overflow_error&){rejected=true;}
  check(rejected);
}
struct Guarded {
  void* mapping;
  size_t capacity, page;
  uint8_t* data;
  explicit Guarded(size_t bytes) {
    page=static_cast<size_t>(sysconf(_SC_PAGESIZE));
    capacity=((bytes+64+page-1)/page)*page;
    mapping=mmap(nullptr,capacity+page,PROT_READ|PROT_WRITE,MAP_PRIVATE|MAP_ANONYMOUS,-1,0);
    check(mapping!=MAP_FAILED);
    check(mprotect(static_cast<char*>(mapping)+capacity,page,PROT_NONE)==0);
    data=static_cast<uint8_t*>(mapping)+capacity-bytes;
    std::memset(mapping,99,capacity);
  }
  ~Guarded(){munmap(mapping,capacity+page);}
};
void copy_boundaries(const detail::CopyBackend& copy) {
  for(size_t bytes:{0UL,1UL,3UL,15UL,16UL,17UL,31UL,32UL,63UL,64UL,65UL,
                    127UL,128UL,191UL,192UL,193UL,255UL,256UL,257UL,
                    511UL,512UL,513UL,4097UL,1048579UL}) {
    Guarded source(bytes),destination(bytes);
    for(size_t i=0;i<bytes;++i)source.data[i]=static_cast<uint8_t>(i*13+7);
    copy.bytes(destination.data,source.data,bytes);
    check(bytes==0 || std::memcmp(source.data,destination.data,bytes)==0);
    for(size_t i=1;i<=64;++i)check(destination.data[-static_cast<ptrdiff_t>(i)]==99);
  }
  for(size_t run:{1UL,17UL,32UL,64UL,192UL,256UL,513UL}) {
    constexpr size_t rows=13,peers=3;
    const size_t stride=run*peers+7;
    std::array<Guarded,peers> sources{Guarded(rows*run),Guarded(rows*run),Guarded(rows*run)};
    Guarded destination(rows*stride);
    std::array<const void*,peers> pointers;
    for(size_t peer=0;peer<peers;++peer) {
      pointers[peer]=sources[peer].data;
      for(size_t i=0;i<rows*run;++i)sources[peer].data[i]=static_cast<uint8_t>(i+peer*23);
    }
    detail::CopyRows job{pointers.data(),destination.data,peers,rows,run,stride};
    for(uint32_t lane=0;lane<4;++lane)copy.rows(&job,lane,4);
    for(size_t row=0;row<rows;++row) {
      for(size_t peer=0;peer<peers;++peer)
        check(std::memcmp(destination.data+row*stride+peer*run,sources[peer].data+row*run,run)==0);
      for(size_t i=run*peers;i<stride;++i)check(destination.data[row*stride+i]==99);
    }
  }
}
namespace mm=vecops::memory::model;
template<class T>T take(mm::Result<T> result) {
  if(!result){const auto& error=std::get<mm::Error>(result.state);throw std::runtime_error(error.code+": "+error.message);}
  return std::move(result).value();
}
void worker(int rank,Bootstrap* store,const std::string& directory) {
  size_t phase=0;
  Exchange exchange=[&](const std::string& value) {
    check(phase<phases && value.size()<1024);
    auto& record=store->values[rank][phase];std::memcpy(record.value,value.c_str(),value.size()+1);
    record.ready.store(1,std::memory_order_release);
    std::vector<std::string> values;
    auto deadline=std::chrono::steady_clock::now()+std::chrono::seconds(5);
    for(int peer=0;peer<ranks;++peer) {
      while(!store->values[peer][phase].ready.load(std::memory_order_acquire)) {
        if(std::chrono::steady_clock::now()>deadline)throw std::runtime_error("test bootstrap timeout");
        std::this_thread::yield();
      }
      values.emplace_back(store->values[peer][phase].value);
    }
    ++phase;return values;
  };
  Options options;options.early_snapshot=std::getenv("VECOPS_TEST_EARLY_SNAPSHOT")!=nullptr;options.timeout_ms=5000;options.relay_threshold=0;
  const auto page=static_cast<uint64_t>(sysconf(_SC_PAGESIZE));
  mm::LinuxDiscovery discovery;
  discovery.graph.agents={{"cpu",{"host_load_store"},"host"}};
  discovery.graph.capacity_groups={{"physical",64ULL<<20}};
  discovery.graph.pools={{"opaque-payload",64ULL<<20,"host",page,{page},{},{},"physical"}};
  discovery.graph.access={{"direct","cpu-map","cpu","opaque-payload",{mm::AccessMode::read,mm::AccessMode::write},
                           {},{},mm::MappingMode::direct,{},{},{},{},{}}};
  discovery.scope={{"cpu"},{"opaque-payload"},{"opaque-payload"},{{"cpu",1}}};
  std::vector<unsigned> nodes;
  const auto* requested_node=std::getenv("VECOPS_TEST_MEMORY_NODE");
  if(requested_node)nodes.push_back(std::stoul(requested_node));
  discovery.pool_bindings={{"opaque-payload",nodes,"host"}};
  discovery.pool_available_bytes={{"opaque-payload",64ULL<<20}};
  mm::LinuxRuntimeOptions memory_options;
  memory_options.participant="rank-"+std::to_string(rank);memory_options.capacity_directory=directory;
  options.memory_runtime=take(mm::make_linux_runtime(std::move(discovery),memory_options));
  options.payload_pool="opaque-payload";
  auto authority=take(mm::details::SharedCapacityAuthority::open(directory,
      {{"physical",64ULL<<20,64ULL<<20}},{},"audit-rank-"+std::to_string(rank)));
  auto charged=[&] {
    uint64_t bytes=0;for(const auto& group:take(authority->snapshot()).groups)bytes+=group.committed_bytes;
    return bytes;
  };
  options.replica_levels={{0,0,1},{0,1,2}};options.copy_levels={1,1};
  options.threads=4;
  VecopsThreadPoolV1 executor{};
  executor.abi_major=VECOPS_THREAD_POOL_ABI_MAJOR;
  executor.struct_size=sizeof(executor);
  executor.parallel_for=[](void*,uint32_t lanes,void* payload,VecopsParallelTaskFn task,VecopsError*) -> int {
    std::vector<std::thread> workers;
    for(uint32_t lane=1;lane<lanes;++lane)workers.emplace_back([=]{task(payload,lane,lanes);});
    task(payload,0,lanes);
    for(auto& thread:workers)thread.join();
    return 0;
  };
  auto on_team=[](auto&& body) {
    struct Team {std::barrier<> phase{4};} team;
    std::mutex mutex;std::exception_ptr failure;std::vector<std::thread> lanes;
    auto work=[&](uint32_t lane) {
      struct Lane {Team* team;uint32_t id;} local{&team,lane};
      VecopsTeamContextV1 context{sizeof(VecopsTeamContextV1),lane,4,&local,
        [](void* opaque,VecopsError*)->int{
          auto& local=*static_cast<Lane*>(opaque);
          local.team->phase.arrive_and_wait();
          // A join releases workers independently. Deliberately delay one
          // return to expose reuse of mutable task metadata after empty phases.
          if(local.id==3)std::this_thread::sleep_for(std::chrono::microseconds(20));
          return 0;
        }};
      try{body(context);}catch(...){team.phase.arrive_and_drop();std::lock_guard lock(mutex);if(!failure)failure=std::current_exception();}
    };
    for(uint32_t lane=1;lane<4;++lane)lanes.emplace_back(work,lane);
    work(0);for(auto& lane:lanes)lane.join();if(failure)std::rethrow_exception(failure);
  };
  {
    // A real payload broadcast, not just a control barrier, precedes metadata
    // registration in applications that distribute weights/inputs cold.
    auto cold_options=options;cold_options.initial_capacity=page;
    cold_options.copy_levels={0,0};
    auto cold=Communicator::create(rank,ranks,cold_options,exchange,&executor);
    const auto before_consensus=cold->statistics();
    check(cold->preparation_consensus(true));
    check(!cold->preparation_consensus(rank!=1));
    check(!cold->preparation_consensus(rank==0));
    check(cold->preparation_consensus(true));
    check(cold->statistics().calls==before_consensus.calls &&
        cold->statistics().capacity_growths==before_consensus.capacity_growths);
    Specification deferred;deferred.operation=Operation::AllGather;
    deferred.inputs={Layout{{static_cast<int64_t>(page*4+17)},{1},1}};
    deferred.outputs.assign(ranks,deferred.inputs.front());deferred.copy_levels={0,0};
    auto deferred_plan=cold->prepare(deferred, /*defer_capacity=*/true);
    check(cold->statistics().capacity_growths==before_consensus.capacity_growths);
    Specification barrier;barrier.operation=Operation::Barrier;
    auto control=cold->prepare(barrier);control->execute(nullptr,nullptr);
    Specification broadcast;broadcast.operation=Operation::Broadcast;
    broadcast.inputs={Layout{{17},{1},1}};broadcast.outputs=broadcast.inputs;
    auto publication=cold->prepare(broadcast);
    std::array<uint8_t,17> small{};if(!rank)small.fill(73);
    const void* small_in[]{small.data()};void* small_out[]{small.data()};
    publication->execute(small_out,small_in);for(auto byte:small)check(byte==73);
    const size_t bytes=page*4+17;
    Specification flat;flat.operation=Operation::AllGather;
    flat.inputs={Layout{{static_cast<int64_t>(bytes)},{1},1}};
    flat.outputs.assign(ranks,flat.inputs.front());flat.copy_levels={0,0};
    auto gather=cold->prepare(flat);
    auto relayed=flat;relayed.copy_levels={1,1};
    auto relay=cold->prepare(relayed);
    Specification borrowed=flat;borrowed.inputs=broadcast.inputs;
    borrowed.outputs.assign(ranks,broadcast.inputs.front());
    auto slow=cold->prepare(borrowed);
    small.fill(static_cast<uint8_t>(rank+31));
    slow->consume(small_in,[&](const void* const* sources,size_t count) {
      check(count==ranks);
      if(rank==2) {
        const auto deadline=std::chrono::steady_clock::now()+std::chrono::seconds(5);
        while(!store->cold_provision_entered.load(std::memory_order_acquire)) {
          if(std::chrono::steady_clock::now()>deadline)throw std::runtime_error("slow-reader preparation timeout");
          std::this_thread::yield();
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(20));
      }
      for(int peer=0;peer<ranks;++peer)for(size_t i=0;i<small.size();++i)
        check(static_cast<const uint8_t*>(sources[peer])[i]==peer+31);
      if(rank==2)store->cold_reader_stopped.store(1,std::memory_order_release);
    });
    if(!rank)store->cold_provision_entered.store(1,std::memory_order_release);
    const auto before_probe=cold->statistics().capacity_growths;
    check(cold->provision_registered_capacity(false));
    check(cold->statistics().capacity_growths==before_probe);
    check(cold->provision_registered_capacity());
    check(store->cold_reader_stopped.load(std::memory_order_acquire)==1);
    check(!cold->provision_registered_capacity());
    // A correct fixed point charges one source plus exactly the union of its
    // registered relay segments. Previous generations/imports have all retired.
    uint64_t expected=0;
    for(int owner=0;owner<ranks;++owner) {
      std::vector<bool> needed(ranks,false);
      for(const auto& partition:cold_options.replica_levels) {
        const auto targets=detail::balanced_relay_targets(partition,owner,std::vector<uint64_t>(ranks,bytes));
        for(int source=0;source<ranks;++source)needed[source]=needed[source] || targets[source]==owner;
      }
      size_t rounded=page;while(rounded<bytes)rounded*=2;
      expected+=rounded*(1+std::count(needed.begin(),needed.end(),true));
    }
    exchange("cold-exact-union-observed");check(charged()==expected);
    std::shared_ptr<mm::Allocation> filler;
    if(!rank) {
      const auto remaining=take(authority->snapshot()).groups.front().hard_remaining_bytes;
      check(remaining>page);
      auto view=take(options.memory_runtime->view(take(options.memory_runtime->current_scope())));
      mm::BackingRequest request{"cold-near-capacity",memory_options.participant,"opaque-payload",remaining-page,
          page,mm::BackingMode::private_memory,{}};
      filler=std::make_shared<mm::Allocation>(take(options.memory_runtime->allocate(view,request)));
    }
    exchange("cold-budget-nearly-full");check(take(authority->snapshot()).groups.front().hard_remaining_bytes==page);
    cold->seal();check(!cold->provision_registered_capacity(false));
    const auto growths=cold->statistics().capacity_growths;
    std::vector<uint8_t> input(bytes),output(bytes*ranks);
    for(size_t i=0;i<bytes;++i)input[i]=static_cast<uint8_t>(i+rank*17);
    const void* in[]{input.data()};std::array<void*,ranks> out;
    for(int peer=0;peer<ranks;++peer)out[peer]=output.data()+peer*bytes;
    for(const auto& plan:{gather,relay}) {
      plan->execute(out.data(),in);
      for(int peer=0;peer<ranks;++peer)for(size_t i=0;i<bytes;++i)
        check(output[peer*bytes+i]==static_cast<uint8_t>(i+peer*17));
      check(cold->statistics().capacity_growths==growths);
    }
    cold->close();filler.reset();exchange("cold-publications-and-filler-refunded");
    const auto snapshot=take(authority->snapshot());
    check(snapshot.roots.empty() && snapshot.grants.empty());
    check(snapshot.groups.front().committed_bytes==0 && snapshot.groups.front().pending_bytes==0);
  }
  auto comm=Communicator::create(rank,ranks,options,exchange,&executor);
  exchange("initial-source-roots");
  // Bootstrap allocates one real source root/rank. Potential relay address
  // space is sparse until a registered route actually requires its segment.
  const auto initial_charge=charged();exchange("initial-source-roots-observed");
  check(initial_charge==ranks*options.initial_capacity);
  // Three ranks, independent pointer rebinding, non-vector tails, empty and
  // single rows, canary padding, and unsupported large-run fallback.
  for(int run:{16,17,32,192,256,512})for(int rows:{0,1,13}) {
    const int stride=ranks*run+7;
    Specification s;s.operation=Operation::AllGather;
    s.inputs={Layout{{rows,run},{run,1},1}};
    s.outputs.assign(ranks,Layout{{rows,run},{stride,1},1});
    auto gather=comm->prepare(s);
    for(int iteration=0;iteration<2;++iteration) {
      std::vector<uint8_t> source(rows*run),destination(rows*stride+64,99);
      for(size_t i=0;i<source.size();++i)source[i]=uint8_t(rank*17+i+iteration);
      std::array<void*,ranks> destinations;
      for(int peer=0;peer<ranks;++peer)destinations[peer]=destination.data()+peer*run;
      const void* input[]{source.data()};
      if(iteration==0)gather->execute(destinations.data(),input);
      else on_team([&](const auto& team){gather->execute_on_team(destinations.data(),input,team);});
      for(int row=0;row<rows;++row) {
        for(int peer=0;peer<ranks;++peer)for(int byte=0;byte<run;++byte)
          check(destination[row*stride+peer*run+byte]==uint8_t(peer*17+row*run+byte+iteration));
        for(int byte=ranks*run;byte<stride;++byte)check(destination[row*stride+byte]==99);
      }
      for(size_t i=rows*stride;i<destination.size();++i)check(destination[i]==99);
    }
  }
  check(comm->statistics().row_copy_calls>0);
  std::array<float,17> input;input.fill(rank+1);
  Specification spec;spec.operation=Operation::AllGather;
  spec.inputs={Layout{{17},{1},4}};spec.outputs.assign(ranks,spec.inputs[0]);
  auto plan=comm->prepare(spec);
  const void* inputs[]{input.data()};
  plan->consume(inputs,[&](const void* const* sources,size_t size) {
    check(size==ranks);for(int r=0;r<ranks;++r)for(int i=0;i<17;++i)check(static_cast<const float*>(sources[r])[i]==r+1);
  });
  Plan::TeamConsumer consumer=[&](const void* const* sources,size_t size,const VecopsTeamContextV1&) {
    check(size==ranks);for(int r=0;r<ranks;++r)for(int i=0;i<17;++i)check(static_cast<const float*>(sources[r])[i]==r+1);
  };
  on_team([&](const auto& team){plan->execute_on_team(nullptr,inputs,team,&consumer);});
  std::vector<std::shared_ptr<ExchangeBuffer>> buffers;
  for(auto placement:{PagePlacement::Source,PagePlacement::Destination}) {
    auto buffer=comm->prepare_exchange(17*sizeof(float),2,placement);
    if(requested_node)for(int peer=0;peer<ranks;++peer) {
      int policy=0;unsigned long nodes[16]{};
      auto* address=static_cast<const char*>(buffer->consumer(0))+peer*buffer->block_stride();
      check(syscall(SYS_get_mempolicy,&policy,nodes,1024,address,MPOL_F_ADDR)==0);
      check((policy&~MPOL_F_STATIC_NODES)==MPOL_BIND);
      check(nodes[std::stoul(requested_node)/64] == (1UL<<(std::stoul(requested_node)%64)));
    }
    for(int i=0;i<16;++i) {
      auto slot=i%2;buffer->begin(slot);
      auto* producer=static_cast<char*>(buffer->producer(slot));
      for(int peer=0;peer<ranks;++peer)for(int j=0;j<17;++j)
        reinterpret_cast<float*>(producer+peer*buffer->block_stride())[j]=rank*100+peer+i*1000;
      buffer->publish(slot);
      if(rank==2 && i==3)std::this_thread::sleep_for(std::chrono::milliseconds(20));
      auto* consumer=static_cast<const char*>(buffer->consumer(slot));
      for(int peer=0;peer<ranks;++peer)for(int j=0;j<17;++j)
        check(reinterpret_cast<const float*>(consumer+peer*buffer->block_stride())[j]==peer*100+rank+i*1000);
      buffer->retire(slot);
    }
    buffers.push_back(buffer);
  }
  // A waiting begin on one slot must not block retirement of another slot.
  auto concurrent=buffers.front();
  for(int slot=0;slot<2;++slot){concurrent->begin(slot);concurrent->publish(slot);}
  int waiting=rank%2,delayed=1-waiting;
  concurrent->retire(waiting);
  std::thread reader([&]{std::this_thread::sleep_for(std::chrono::milliseconds(20));concurrent->retire(delayed);});
  concurrent->begin(waiting);reader.join();
  concurrent->begin(delayed);
  for(int slot=0;slot<2;++slot){concurrent->publish(slot);concurrent->retire(slot);}
  comm->seal();
  // All regions and plans remain usable after freezing; no new descriptors.
  for(auto& buffer:buffers) {buffer->begin(0);buffer->publish(0);buffer->retire(0);}
  check(charged()>0);
  comm->close();exchange("all-roots-refunded");
  check(charged()==0);
  try{comm->prepare(spec);check(false);}catch(const std::runtime_error&){}
  // A shared planning failure can abort every otherwise idle group. Staggered
  // cleanup must await stopped acknowledgements rather than reject the abort
  // flags and leak named physical roots. Exercise page aliases as well.
  auto idle=Communicator::create(rank,ranks,options,exchange,&executor);
  auto idle_exchange=idle->prepare_exchange(17*sizeof(float),1,PagePlacement::Destination);
  check(charged()>0);
  idle->abort();idle_exchange->abort();exchange("idle-aborted");
  if(rank==1)std::this_thread::sleep_for(std::chrono::milliseconds(40));
  idle->close();exchange("idle-aborted-roots-refunded");
  check(charged()==0);
  auto failing=Communicator::create(rank,ranks,options,exchange,&executor);
  auto bad=failing->prepare_exchange(17*sizeof(float),1,PagePlacement::Destination);
  if(rank==0) {
    bad->begin(0);
    try{bad->begin(0);check(false);}catch(const std::runtime_error&){}
  } else {
    try{bad->begin(0);bad->publish(0);check(false);}catch(const std::runtime_error&){}
  }
}
}
int main(int argc,char** argv) {
  (void)argc;
  try {
    replica_routing();
    copy_boundaries(detail::copy_backend());
    copy_boundaries(detail::portable::backend());
#if defined(VECOPS_TEST_COMM_COPY_AVX2)
    if(__builtin_cpu_supports("avx2"))copy_boundaries(detail::avx2::backend());
#endif
  } catch(const std::exception& error) {std::cerr<<error.what()<<'\n';return 1;}
  auto* mapping=mmap(nullptr,sizeof(Bootstrap),PROT_READ|PROT_WRITE,MAP_SHARED|MAP_ANONYMOUS,-1,0);
  if(mapping==MAP_FAILED)return 1;
  auto* store=new(mapping) Bootstrap;
  const auto directory=std::filesystem::absolute(argv[0]).parent_path()/("comm-capacity-"+std::to_string(getpid()));
  std::filesystem::create_directory(directory);
  std::filesystem::permissions(directory,std::filesystem::perms::owner_all,std::filesystem::perm_options::replace);
  std::array<pid_t,ranks> children;
  for(int rank=0;rank<ranks;++rank) {
    auto pid=fork();if(pid<0)return 1;
    if(!pid) {try{worker(rank,store,directory.string());_exit(0);}catch(const std::exception& e){std::cerr<<rank<<": "<<e.what()<<'\n';_exit(1);}}
    children[rank]=pid;
  }
  bool okay=true;
  for(auto pid:children){int status=0;waitpid(pid,&status,0);okay &= WIFEXITED(status) && WEXITSTATUS(status)==0;}
  if(okay) {
    try {
      auto authority=take(mm::details::SharedCapacityAuthority::open(directory,
          {{"physical",64ULL<<20,64ULL<<20}},{},"audit"));
      const auto snapshot=take(authority->snapshot());
      check(!snapshot.roots.empty() && snapshot.groups.front().committed_bytes>0);
      check(snapshot.groups.front().pending_bytes==0);
      for(const auto& root:snapshot.roots)
        check(root.owner.state==mm::details::SharedCapacityOwnerState::owner_dead);
      std::cout<<"3-rank Runtime payload/relay/page-alias tests passed; normal shutdown refunded all roots; "
               <<snapshot.roots.size()<<" aborted root receipts retained conservatively\n";
      // Every test child has been joined. The active-lease negative case must
      // retain receipts, but its named files must not leak into later tests.
      // Match kernel identity and test-child creator, never unlink by prefix
      // alone. Live production owners are deliberately not reclaimed this way.
      std::string boot;std::ifstream("/proc/sys/kernel/random/boot_id")>>boot;
      check(!boot.empty());
      size_t removed=0;
      for(const auto& entry:std::filesystem::directory_iterator("/dev/shm")) {
        const auto name=entry.path().filename().string();
        if(std::none_of(children.begin(),children.end(),[&](auto pid){return name.starts_with("vecops-"+std::to_string(pid)+"-");}))continue;
        struct stat info{};check(lstat(entry.path().c_str(),&info)==0);
        check(S_ISREG(info.st_mode) && info.st_uid==getuid());
        const auto identity="linux:"+boot+":"+std::to_string(info.st_dev)+":"+std::to_string(info.st_ino);
        const auto receipt=std::find_if(snapshot.roots.begin(),snapshot.roots.end(),[&](const auto& r){return r.physical_identity==identity;});
        check(receipt!=snapshot.roots.end() && receipt->bytes==static_cast<uint64_t>(info.st_size));
        check(unlink(entry.path().c_str())==0);++removed;
      }
      check(removed==snapshot.roots.size());
    }catch(const std::exception& error){std::cerr<<error.what()<<'\n';okay=false;}
  }
  store->~Bootstrap();munmap(mapping,sizeof(Bootstrap));
  if(okay)std::filesystem::remove_all(directory);
  else std::cerr<<"Failed communication test metadata retained at "<<directory<<'\n';
  return okay?0:1;
}
