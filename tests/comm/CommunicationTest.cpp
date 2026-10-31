// SPDX-FileCopyrightText: 2026 HenryRenYz and vecops contributors
// SPDX-License-Identifier: MIT
// Standalone fork test: no Torch/Python/GTest or external transport dependency.
#include "vecops/comm/Communication.h"
#include "vecops/comm/Copy.h"
#include <array>
#include <atomic>
#include <chrono>
#include <cstring>
#include <cstdlib>
#include <iostream>
#include <stdexcept>
#include <thread>
#include <sys/mman.h>
#include <sys/wait.h>
#include <sys/syscall.h>
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
};
void check(bool result){if(!result)throw std::runtime_error("standalone communication assertion");}
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
void worker(int rank,Bootstrap* store) {
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
  Options options;options.timeout_ms=5000;options.relay_threshold=0;
  if(auto* node=std::getenv("VECOPS_TEST_MEMORY_NODE")){options.memory_node=std::atoi(node);options.require_placement=true;}
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
  auto comm=Communicator::create(rank,ranks,options,exchange,&executor);
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
      gather->execute(destinations.data(),input);
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
  std::vector<std::shared_ptr<ExchangeBuffer>> buffers;
  for(auto placement:{PagePlacement::Source,PagePlacement::Destination}) {
    auto buffer=comm->prepare_exchange(17*sizeof(float),2,placement);
    if(options.require_placement)for(int peer=0;peer<ranks;++peer) {
      int policy=0;unsigned long nodes[16]{};
      auto* address=static_cast<const char*>(buffer->consumer(0))+peer*buffer->block_stride();
      check(syscall(SYS_get_mempolicy,&policy,nodes,1024,address,MPOL_F_ADDR)==0);
      check((policy&~MPOL_F_STATIC_NODES)==MPOL_BIND);
      check(nodes[options.memory_node/64] == (1UL<<(options.memory_node%64)));
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
  auto bad=buffers.back();
  if(rank==0) {
    bad->begin(0);
    try{bad->begin(0);check(false);}catch(const std::runtime_error&){}
  } else {
    try{bad->begin(0);bad->publish(0);check(false);}catch(const std::runtime_error&){}
  }
}
}
int main() {
  try {
    copy_boundaries(detail::copy_backend());
    copy_boundaries(detail::portable::backend());
#if defined(VECOPS_TEST_COMM_COPY_AVX2)
    if(__builtin_cpu_supports("avx2"))copy_boundaries(detail::avx2::backend());
#endif
  } catch(const std::exception& error) {std::cerr<<error.what()<<'\n';return 1;}
  auto* mapping=mmap(nullptr,sizeof(Bootstrap),PROT_READ|PROT_WRITE,MAP_SHARED|MAP_ANONYMOUS,-1,0);
  if(mapping==MAP_FAILED)return 1;
  auto* store=new(mapping) Bootstrap;
  std::array<pid_t,ranks> children;
  for(int rank=0;rank<ranks;++rank) {
    auto pid=fork();if(pid<0)return 1;
    if(!pid) {try{worker(rank,store);_exit(0);}catch(const std::exception& e){std::cerr<<rank<<": "<<e.what()<<'\n';_exit(1);}}
    children[rank]=pid;
  }
  bool okay=true;
  for(auto pid:children){int status=0;waitpid(pid,&status,0);okay &= WIFEXITED(status) && WEXITSTATUS(status)==0;}
  store->~Bootstrap();munmap(mapping,sizeof(Bootstrap));return okay?0:1;
}
