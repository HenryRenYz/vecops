// SPDX-FileCopyrightText: 2026 HenryRenYz and vecops contributors
// SPDX-License-Identifier: MIT
// Direct SDMA copy runner. SDK completion APIs own device ordering; compiler/CPU
// fences supplement them and never substitute for a completed device request.
// Each channel has a bounded FIFO of requests and disjoint in-flight chunks.
#include "Bench.h"
#include <algorithm>
#include <atomic>
#include <condition_variable>
#include <cstring>
#include <cstdlib>
#include <dlfcn.h>
#include <fcntl.h>
#include <filesystem>
#include <iostream>
#include <limits>
#include <memory>
#include <mutex>
#include <set>
#include <stdexcept>
#include <sys/resource.h>
#include <sched.h>
#include <thread>
#include <time.h>
#include <unistd.h>
#ifdef VECOPS_MEMORY_HAVE_SDMA
#include <mdk_sdma.h>
#endif

namespace memory_bench {
namespace {
std::string text_file(const std::string& p) { std::ifstream f(p); std::string s; std::getline(f,s); return s; }
std::vector<int> device_ids() {
  std::vector<int> result;
  for (const auto& p : std::filesystem::directory_iterator("/sys/class")) {
    if (p.path().filename() != "sdma") continue;
    for (const auto& e : std::filesystem::directory_iterator(p)) {
      auto s = e.path().filename().string();
      if (s.size() > 4 && s.substr(0,4) == "sdma" && s.find_first_not_of("0123456789",4) == std::string::npos)
        result.push_back(std::stoi(s.substr(4)));
    }
  }
  std::sort(result.begin(),result.end()); return result;
}
std::string device_numa(int id) {
  const auto base = "/sys/class/sdma/sdma" + std::to_string(id);
  for (const auto& suffix : {"/device/numa_node", "/numa_node"}) {
    auto value = text_file(base + suffix);
    if (!value.empty() && value != "-1") return value;
  }
  return "unknown";
}
#ifdef VECOPS_MEMORY_HAVE_SDMA
struct Api {
  void* lib = nullptr;
  std::string path;
#define SDMA_API(name) decltype(&::name) name = nullptr
  SDMA_API(sdma_alloc_chn); SDMA_API(sdma_init_chn);
  SDMA_API(sdma_free_chn); SDMA_API(sdma_deinit_chn);
  SDMA_API(sdma_copy_data); SDMA_API(sdma_icopy_data);
  SDMA_API(sdma_wait_chn); SDMA_API(sdma_query_chn); SDMA_API(sdma_iquery_chn);
  SDMA_API(sdma_get_process_id); SDMA_API(sdma_devices_num);
  SDMA_API(sdma_pin_umem); SDMA_API(sdma_unpin_umem);
#undef SDMA_API
  Api() {
    lib = dlopen("libsdma_dk.so", RTLD_NOW | RTLD_LOCAL);
    if (!lib) throw std::runtime_error(std::string("Cannot load SDMA SDK: ") + dlerror());
    try {
#define LOAD(name) name = reinterpret_cast<decltype(name)>(dlsym(lib,#name)); if (!name) throw std::runtime_error("SDMA SDK missing " #name)
      LOAD(sdma_alloc_chn); LOAD(sdma_init_chn); LOAD(sdma_free_chn); LOAD(sdma_deinit_chn);
      LOAD(sdma_copy_data); LOAD(sdma_icopy_data); LOAD(sdma_wait_chn);
      LOAD(sdma_query_chn); LOAD(sdma_iquery_chn); LOAD(sdma_get_process_id);
      LOAD(sdma_devices_num); LOAD(sdma_pin_umem); LOAD(sdma_unpin_umem);
#undef LOAD
      Dl_info info{};
      if (dladdr(reinterpret_cast<void*>(sdma_init_chn), &info) && info.dli_fname) path = info.dli_fname;
    } catch (...) { dlclose(lib); lib = nullptr; throw; }
  }
  ~Api() { if (lib) dlclose(lib); }
};
void check(int code, const char* operation) {
  if (code) throw std::runtime_error(std::string(operation) + " failed, SDK code " + std::to_string(code));
}
class Gate {
  std::mutex mutex; std::condition_variable cv;
  std::size_t parties, left, generation = 0;
 public:
  explicit Gate(std::size_t n) : parties(n), left(n) {}
  void wait() {
    std::unique_lock lock(mutex); auto g = generation;
    if (--left == 0) { left = parties; ++generation; lock.unlock(); cv.notify_all(); }
    else cv.wait(lock,[&]{ return generation != g; });
  }
};
struct Slot {
  std::vector<sdma_sqe_task_t> tasks;
  sdma_request_t request{};
  std::uint64_t bytes = 0;
};
struct Channel {
  void* handle = nullptr;
  int index = -1;
  std::size_t first = 0, length = 0, cursor = 0, head = 0, tail = 0, pending = 0;
  std::uint64_t done_bytes = 0;
  std::vector<Slot> slots;
};
struct Worker {
  Api& api; const Options& o;
  int cpu, group, device, fd = -1, end_cpu = -1;
  std::size_t bytes;
  std::uint32_t pasid = 0;
  std::vector<Channel> channels;
  std::unique_ptr<Buffer> src, dst;
  std::uint64_t cookies[2]{}; bool pinned[2]{};
  std::exception_ptr error;
  std::uint64_t begin=0,end=0,drain=0,payload=0,descriptors=0,submissions=0;
  std::uint64_t minor=0,major=0,switches=0;
  double setup=0,cpu_time=0;
  unsigned epoch=0;
  Worker(Api& a,const Options& options,int c,int g,int d,std::size_t b) : api(a),o(options),cpu(c),group(g),device(d),bytes(b) {}
  ~Worker() {
    for (auto& c : channels) if (c.handle) {
      const int rc = o.sdma_mode == "shared" ? api.sdma_deinit_chn(c.handle) : api.sdma_free_chn(c.handle);
      if (rc) { std::cerr << "SDMA channel cleanup failed: " << rc << '\n'; std::_Exit(3); }
    }
    for (int i=0;i<2;++i) if (pinned[i] && api.sdma_unpin_umem(fd,cookies[i])) {
      std::cerr << "SDMA memory unpin failed\n"; std::_Exit(3);
    }
    if (fd >= 0) close(fd);
  }
};
std::uint32_t pattern(std::size_t i,int cpu) {
  // Bounded FP32 bit patterns, unique across chunks, workers and cache lines.
  return 0x3f000000U | ((i * 2654435761U + unsigned(cpu) * 104729U) & 0x007fffffU);
}
class Session {
 public:
  Api& api; const Options& o; const Topology& t; const std::vector<Group>& mapping;
  std::size_t depth,batch,chunk;
  std::vector<std::unique_ptr<Worker>> workers;
  std::vector<std::thread> threads;
  std::unique_ptr<Gate> gate;
  std::atomic<bool> launched{false},cancel{false};
  std::mutex sdk_setup;
  bool stop=false; std::uint64_t start=0,deadline=0;
  Session(Api& api,const Options& o,const Topology& t,const std::vector<Group>& mapping,
          const std::vector<int>& devices,std::size_t bytes,int count,int depth,int batch,std::size_t chunk)
      : api(api),o(o),t(t),mapping(mapping),depth(depth),batch(batch),chunk(chunk) {
    std::map<int,int> next_channel;
    std::map<int,std::size_t> demand;
    auto shared = text_file("/sys/module/sdma_dae/parameters/share_chns");
    for (std::size_t g=0;g<mapping.size();++g) {
      auto n=mapping[g].cpus.size();
      if (std::size_t(count)<n) throw std::runtime_error("SDMA channels/group must be at least the controller thread count");
      if (o.sdma_mode == "shared" && !shared.empty() && next_channel[devices[g]] + count > std::stoi(shared))
        throw std::runtime_error("Requested distinct shared-channel indexes exceed the driver share_chns parameter");
      for (std::size_t i=0;i<n;++i) {
        auto b=bytes;
        if (o.size_scope=="group-array") { if (b%n) throw std::runtime_error("Group array must divide exactly by workers"); b/=n; }
        if (b%4 || b>UINT32_MAX) throw std::runtime_error("SDMA worker array must be FP32-aligned and <= UINT32_MAX (SDK pin size)");
        auto w=std::make_unique<Worker>(api,o,mapping[g].cpus[i],g,devices[g],b);
        const auto assigned=count/n+(i<std::size_t(count)%n);
        if (b/4<assigned) throw std::runtime_error("Array is too small for the requested channels");
        std::size_t first=0;
        for (std::size_t j=0;j<assigned;++j) {
          auto length=(b/4/assigned+(j<b/4%assigned))*4;
          if ((length+chunk-1)/chunk<std::size_t(depth)) throw std::runtime_error("Buffer/channel must hold at least depth distinct chunks");
          Channel c; c.index=next_channel[devices[g]]++; c.first=first;c.length=length;
          c.slots.resize(depth/batch);
          for(auto& s:c.slots) s.tasks.resize(batch);
          w->channels.push_back(std::move(c)); first+=length;
        }
        for (bool source:{true,false}) {
          const auto nodes=memory_nodes(o,mapping,g,source);
          const auto page=o.pages=="huge"?2*1024*1024UL:t.page_bytes;
          const auto pages=(b+page-1)/page;
          for(std::size_t k=0;k<nodes.size();++k)
            demand[nodes[k]]+=checked_mul(pages/nodes.size()+(k<pages%nodes.size())+(nodes.size()>1),page);
        }
        workers.push_back(std::move(w));
      }
    }
    for(auto [node,b]:demand) if(b>(o.pages=="huge"?t.node(node).huge_free_bytes:t.node(node).free_bytes)*.9)
      throw std::runtime_error("SDMA arrays exceed 90% of the selected page pool on node "+std::to_string(node));
    gate=std::make_unique<Gate>(workers.size()+1);
    try {
      for(auto& w:workers) threads.emplace_back([this,p=w.get()] {
        while(!launched.load(std::memory_order_acquire)) relax();
        if(cancel.load()) return;
        try { initialize(*p); } catch(...) {p->error=std::current_exception();}
        gate->wait();
        for(;;) {
          gate->wait(); if(stop) break;
          try { measure(*p); } catch(...) {
            // Never unmap/pin-release buffers that may still be in use by DMA.
            for(auto& c:p->channels) if(c.pending) {
              std::cerr << "SDMA failure with in-flight DMA; terminating process without user-space cleanup\n";
              try { throw; } catch(const std::exception& e) { std::cerr << e.what() << std::endl; }
              std::_Exit(3);
            }
            p->error=std::current_exception();
          }
          gate->wait();
        }
      });
    } catch(...) {
      cancel=true; launched.store(true,std::memory_order_release);
      for(auto& th:threads) th.join(); throw;
    }
    launched.store(true,std::memory_order_release);gate->wait();
  }
  ~Session() {stop=true;gate->wait();for(auto& th:threads) th.join();}
  void errors() {for(auto& w:workers) if(w->error) std::rethrow_exception(w->error);}
  void phase(double seconds) {
    errors();
    for(auto& w:workers) {
      ++w->epoch;
      auto* a=static_cast<std::uint32_t*>(w->src->data);auto* d=static_cast<std::uint32_t*>(w->dst->data);
      for(std::size_t i=0;i<w->bytes/4;++i){a[i]=pattern(i,w->cpu+int(w->epoch)*613);d[i]=0xdeadbeefU;}
      for(auto& c:w->channels){c.cursor=c.head=c.tail=c.pending=0;c.done_bytes=0;}
      w->src->verify();w->dst->verify();
    }
    completion_fence();start=now_ns()+std::max<std::uint64_t>(10000000,workers.size()*100000);
    deadline=start+std::uint64_t(seconds*1e9);gate->wait();gate->wait();errors();verify();
  }
  void initialize(Worker& w) {
    pin(w.cpu); auto begin=now_ns();
    w.src=std::make_unique<Buffer>(w.bytes,memory_nodes(o,mapping,w.group,true),o.pages,t.page_bytes);
    w.dst=std::make_unique<Buffer>(w.bytes,memory_nodes(o,mapping,w.group,false),o.pages,t.page_bytes);
    auto* a=static_cast<std::uint32_t*>(w.src->data); auto* d=static_cast<std::uint32_t*>(w.dst->data);
    for(std::size_t i=0;i<w.bytes/4;++i) {a[i]=pattern(i,w.cpu);d[i]=0xdeadbeefU;}
    w.src->verify();w.dst->verify();completion_fence();
    std::lock_guard lock(sdk_setup);
    if(w.src->mapped_bytes>UINT32_MAX||w.dst->mapped_bytes>UINT32_MAX)throw std::runtime_error("Page-rounded mapping exceeds SDK pin range");
    auto path="/dev/sdma"+std::to_string(w.device);
    w.fd=open(path.c_str(),O_RDWR|O_CLOEXEC);if(w.fd<0) throw std::runtime_error("Cannot open "+path);
    check(api.sdma_get_process_id(w.fd,&w.pasid),"sdma_get_process_id");
    check(api.sdma_pin_umem(w.fd,w.src->data,w.src->mapped_bytes,&w.cookies[0]),"sdma_pin_umem(source)");w.pinned[0]=true;
    check(api.sdma_pin_umem(w.fd,w.dst->data,w.dst->mapped_bytes,&w.cookies[1]),"sdma_pin_umem(destination)");w.pinned[1]=true;
    for(auto& c:w.channels) {
      c.handle=o.sdma_mode=="shared"?api.sdma_init_chn(w.fd,c.index):api.sdma_alloc_chn(w.fd);
      if(!c.handle) throw std::runtime_error("Cannot allocate SDMA "+o.sdma_mode+" channel on "+path);
    }
    // Full untimed correctness pass, including partial chunks and all array tails.
    for(auto& c:w.channels) for(std::size_t offset=0;offset<c.length;) {
      sdma_sqe_task_t task{};
      task.src_addr=reinterpret_cast<uintptr_t>(w.src->data)+c.first+offset;
      task.dst_addr=reinterpret_cast<uintptr_t>(w.dst->data)+c.first+offset;
      task.src_process_id=task.dst_process_id=w.pasid;task.length=std::min(chunk,c.length-offset);
      if(o.sdma_mode=="shared") {sdma_request_t request{};check(api.sdma_icopy_data(c.handle,&task,1,&request),"validation submit");
        const auto limit=now_ns()+30000000000ULL;int rc;
        while(((rc=api.sdma_iquery_chn(c.handle,&request))==SDMA_TASK_UNFINISH || rc==SDMA_RNDCNT_ERR) && now_ns()<limit) relax();
        if(rc) {std::cerr<<"Validation DMA did not complete safely: "<<rc<<std::endl;std::_Exit(3);}
      } else {check(api.sdma_copy_data(c.handle,&task,1),"validation submit");
        if(api.sdma_wait_chn(c.handle,1)) {std::cerr<<"Validation DMA failed\n";std::_Exit(3);}}
      offset+=task.length;c.done_bytes+=task.length;
    }
    completion_fence();w.setup=(now_ns()-begin)*1e-9;
    rusage r{};getrusage(RUSAGE_THREAD,&r);(void)now_ns();
  }
  void verify() {
    for(auto& w:workers) {
      completion_fence();w->src->verify();w->dst->verify();
      auto* a=static_cast<const std::uint32_t*>(w->src->data);
      auto* d=static_cast<const std::uint32_t*>(w->dst->data);
      for(auto& c:w->channels)for(std::size_t offset=0;offset<c.length;offset+=4) {
        const auto i=(c.first+offset)/4;
        const auto expected=offset<std::min<std::uint64_t>(c.done_bytes,c.length)?a[i]:0xdeadbeefU;
        if(a[i]!=pattern(i,w->cpu+int(w->epoch)*613)||d[i]!=expected)
          throw std::runtime_error("SDMA generation/sentinel bitwise verification failed at word "+std::to_string(i));
      }
    }
  }
  void submit(Worker& w,Channel& c) {
    auto& s=c.slots[c.tail];s.bytes=0;
    for(std::size_t i=0;i<batch;++i) {
      auto& task=s.tasks[i]; task={};
      task.src_addr=reinterpret_cast<uintptr_t>(w.src->data)+c.first+c.cursor;
      task.dst_addr=reinterpret_cast<uintptr_t>(w.dst->data)+c.first+c.cursor;
      task.src_process_id=task.dst_process_id=w.pasid;
      task.length=std::min(chunk,c.length-c.cursor);s.bytes+=task.length;
      task.next_sqe=i+1<batch?&s.tasks[i+1]:nullptr;
      c.cursor+=task.length;if(c.cursor==c.length)c.cursor=0;
    }
    // Mark before entering SDK: a failing submit may have partially enqueued work.
    ++c.pending;
    if(o.sdma_mode=="shared") check(api.sdma_icopy_data(c.handle,s.tasks.data(),batch,&s.request),"sdma_icopy_data");
    else check(api.sdma_copy_data(c.handle,s.tasks.data(),batch),"sdma_copy_data");
    c.tail=(c.tail+1)%c.slots.size();++w.submissions;
  }
  bool retire(Worker& w,Channel& c) {
    if(!c.pending)return false;
    auto& s=c.slots[c.head];
    auto rc=o.sdma_mode=="shared"?api.sdma_iquery_chn(c.handle,&s.request):api.sdma_query_chn(c.handle,batch);
    // sdma-dk shared query uses RNDCNT_ERR when the requested CQ generation
    // has not arrived yet (the public implementation's rndcnt_invalid test).
    // Retain the request; a bounded drain and generation/data checks remain mandatory.
    if(rc==SDMA_TASK_UNFINISH || (o.sdma_mode=="shared" && rc==SDMA_RNDCNT_ERR))return false;
    check(rc,"SDMA completion query");
    if(o.sdma_mode=="exclusive")check(api.sdma_wait_chn(c.handle,batch),"sdma_wait_chn");
    c.done_bytes+=s.bytes;w.payload+=s.bytes;w.descriptors+=batch;--c.pending;c.head=(c.head+1)%c.slots.size();return true;
  }
  void measure(Worker& w) {
    w.payload=w.descriptors=w.submissions=0;
    rusage before{},after{};
    completion_fence();while(now_ns()<start)relax();
    getrusage(RUSAGE_THREAD,&before);
    timespec cpu_before{},cpu_after{};
    if(clock_gettime(CLOCK_THREAD_CPUTIME_ID,&cpu_before))throw std::runtime_error("Thread CPU clock unavailable");
    w.begin=now_ns();
    auto current=w.begin;
    while(current<deadline) {
      for(auto& c:w.channels) {
        retire(w,c);
        if(c.pending<c.slots.size() && now_ns()<deadline)submit(w,c);
      }
      current=now_ns();
    }
    auto drain_start=now_ns();
    for(;;) {
      bool pending=false;
      for(auto& c:w.channels){retire(w,c);pending|=c.pending!=0;}
      if(!pending)break;
      if(now_ns()-drain_start>30000000000ULL)throw std::runtime_error("SDMA drain timeout (30 s)");
      relax();
    }
    completion_fence();w.end=now_ns();w.drain=w.end-drain_start;w.end_cpu=sched_getcpu();
    if(clock_gettime(CLOCK_THREAD_CPUTIME_ID,&cpu_after))throw std::runtime_error("Thread CPU clock unavailable");
    getrusage(RUSAGE_THREAD,&after);
    w.minor=after.ru_minflt-before.ru_minflt;w.major=after.ru_majflt-before.ru_majflt;
    w.switches=after.ru_nvcsw+after.ru_nivcsw-before.ru_nvcsw-before.ru_nivcsw;
    w.cpu_time=(cpu_after.tv_sec-cpu_before.tv_sec)+(cpu_after.tv_nsec-cpu_before.tv_nsec)*1e-9;
  }
};
void rates(Row& r) {
  r.seconds=(r.end_ns-r.start_ns)*1e-9;
  r.read_gbs=r.read_bytes/r.seconds/1e9;r.write_gbs=r.write_bytes/r.seconds/1e9;
  r.payload_gbs=r.write_gbs;r.gbs=r.read_gbs+r.write_gbs;
}
Row worker_row(const Session& s,const Worker& w,Row r) {
  const auto& g=s.mapping[w.group];r.scope="thread:g"+std::to_string(w.group)+"/cpu"+std::to_string(w.cpu);
  r.group=w.group;r.src=g.route.src;r.dst=g.route.dst;r.exec=g.exec;r.cpus=std::to_string(w.cpu);r.threads=1;
  r.array_bytes=w.bytes;r.working_bytes=checked_mul(w.bytes,2);r.source_nodes=w.src->nodes;r.destination_nodes=w.dst->nodes;
  r.source_pages=w.src->page_counts();r.destination_pages=w.dst->page_counts();r.placement="all-pages-verified";
  r.start_ns=w.begin;r.end_ns=w.end;r.read_bytes=r.write_bytes=w.payload;
  r.sdma_devices=std::to_string(w.device);r.sdma_library=s.api.path;r.dma_channels=w.channels.size();
  for(auto& c:w.channels) {if(!r.sdma_channel_ids.empty())r.sdma_channel_ids+='|';
    r.sdma_channel_ids+=std::to_string(w.device)+":"+(s.o.sdma_mode=="shared"?"shared-index-":"opaque-allocation-")+std::to_string(c.index);}
  r.dma_depth=s.depth;r.dma_batch=s.batch;r.dma_chunk=s.chunk;r.dma_descriptors=w.descriptors;r.dma_submissions=w.submissions;
  r.drain_ns=w.drain;r.cpu_seconds=w.cpu_time;r.setup_seconds=w.setup;
  r.minor_faults=w.minor;r.major_faults=w.major;r.switches=w.switches;
  if(!w.payload || w.begin>=w.end)r.status="missed-window";
  else if(w.end_cpu!=w.cpu)r.status="migration";
  else if(w.minor||w.major)r.status="page-fault";
  else if(w.switches)r.status="interrupted";
  rates(r);return r;
}
Row aggregate(const std::vector<Row>& rows,const std::string& scope,const Options& o) {
  Row r=rows.front();r.scope=scope;r.cpus=r.sdma_devices=r.sdma_channel_ids="";
  r.source_nodes.clear();r.destination_nodes.clear();r.source_pages.clear();r.destination_pages.clear();
  r.read_bytes=r.write_bytes=r.dma_descriptors=r.dma_submissions=r.drain_ns=0;
  r.threads=r.working_bytes=r.dma_channels=0;r.minor_faults=r.major_faults=r.switches=0;
  r.cpu_seconds=r.setup_seconds=0;r.start_ns=UINT64_MAX;r.end_ns=0;
  std::uint64_t latest=0,earliest=UINT64_MAX;
  for(auto& a:rows) {
    for(auto pair:{std::pair{&r.cpus,&a.cpus},std::pair{&r.sdma_devices,&a.sdma_devices},std::pair{&r.sdma_channel_ids,&a.sdma_channel_ids}}) {
      if(!pair.first->empty())*pair.first+='|';*pair.first+=*pair.second;
    }
    r.read_bytes+=a.read_bytes;r.write_bytes+=a.write_bytes;r.dma_descriptors+=a.dma_descriptors;r.dma_submissions+=a.dma_submissions;
    r.threads+=a.threads;r.working_bytes+=a.working_bytes;r.dma_channels+=a.dma_channels;
    r.cpu_seconds+=a.cpu_seconds;r.setup_seconds=std::max(r.setup_seconds,a.setup_seconds);r.drain_ns=std::max(r.drain_ns,a.drain_ns);
    r.minor_faults+=a.minor_faults;r.major_faults+=a.major_faults;r.switches+=a.switches;
    r.start_ns=std::min(r.start_ns,a.start_ns);r.end_ns=std::max(r.end_ns,a.end_ns);
    latest=std::max(latest,a.start_ns);earliest=std::min(earliest,a.end_ns);
    for(auto [n,count]:a.source_pages)r.source_pages[n]+=count;
    for(auto [n,count]:a.destination_pages)r.destination_pages[n]+=count;
    r.source_nodes.insert(r.source_nodes.end(),a.source_nodes.begin(),a.source_nodes.end());
    r.destination_nodes.insert(r.destination_nodes.end(),a.destination_nodes.begin(),a.destination_nodes.end());
    if(a.status!="ok")r.status=a.status;
  }
  for(auto* nodes:{&r.source_nodes,&r.destination_nodes}) {std::sort(nodes->begin(),nodes->end());nodes->erase(std::unique(nodes->begin(),nodes->end()),nodes->end());}
  r.start_skew_ns=latest-r.start_ns;r.end_skew_ns=r.end_ns-earliest;r.common_ns=earliest>latest?earliest-latest:0;
  if(r.start_skew_ns>o.seconds*1e9*o.max_skew||r.end_skew_ns>o.seconds*1e9*o.max_skew||!r.common_ns)r.status="skew";
  if(scope=="total")r.group=r.src=r.dst=r.exec=-1;
  rates(r);return r;
}
#endif
} // namespace
void print_sdma_topology(const Topology&) {
  const auto ids=device_ids();
  std::cout<<"\n  SDMA visible devices: "<<ids.size()<<" (physical engine count: unknown)\n";
  for(auto id:ids)std::cout<<"  /dev/sdma"<<id<<" | NUMA: "<<device_numa(id)<<" | sysfs: /sys/class/sdma/sdma"<<id<<'\n';
  std::cout<<"  Driver share_chns: "<<text_file("/sys/module/sdma_dae/parameters/share_chns")
           <<" | safe_mode: "<<text_file("/sys/module/sdma_dae/parameters/safe_mode")<<'\n';
#ifdef VECOPS_MEMORY_HAVE_SDMA
  Api api;std::cout<<"  SDK: "<<api.path<<'\n';
  if(!ids.empty()) {
    auto fd=open(("/dev/sdma"+std::to_string(ids.front())).c_str(),O_RDWR|O_CLOEXEC);
    if(fd<0)throw std::runtime_error("Cannot open enumerated SDMA device");
    auto n=api.sdma_devices_num(fd);close(fd);if(n<0)throw std::runtime_error("SDK device query failed");
    std::cout<<"  SDK device count: "<<n<<"\n";
    if(n!=int(ids.size()))throw std::runtime_error("SDK/sysfs visible device counts disagree");
  }
#else
  throw std::runtime_error("SDMA backend was not built: install sdma-dk headers and configure VECOPS_MEMORY_SDMA=ON");
#endif
}
void run_sdma(const Options& o,const Topology& t,Reporter& reporter) {
#ifndef VECOPS_MEMORY_HAVE_SDMA
  (void)o;(void)t;(void)reporter;
  throw std::runtime_error("SDMA backend unavailable: configure with installed sdma-dk headers");
#else
  Api api;auto visible=device_ids();int config=0;
  for(auto cpu_spec:split(o.cpu,';')) {
    auto mapping=groups(o,t,cpu_list(cpu_spec));
    auto specs=split(o.sdma_device,'|');if(specs.size()!=1&&specs.size()!=mapping.size())throw std::runtime_error("Supply one SDMA device or one ID per group");
    std::vector<int> devices;
    for(std::size_t g=0;g<mapping.size();++g) {
      auto ids=cpu_list(specs[specs.size()==1?0:g]);if(ids.size()!=1)throw std::runtime_error("Each SDMA device field must contain exactly one ID");
      if(std::find(visible.begin(),visible.end(),ids[0])==visible.end())throw std::runtime_error("Requested SDMA device is not enumerated");
      devices.push_back(ids[0]);
      std::cout<<"  Group "<<g<<" | Memory "<<mapping[g].route.src<<":"<<mapping[g].route.dst<<" | CPU "<<join_cpus(mapping[g].cpus)<<" | SDMA "<<ids[0]<<" | Device NUMA "<<device_numa(ids[0])<<'\n';
    }
    for(auto bytes:o.sizes)for(auto count:o.sdma_channels)for(auto depth:o.sdma_depths)for(auto batch:o.sdma_batches)for(auto chunk:o.sdma_chunks) {
      Row base;base.config=++config;base.kind="bandwidth";base.operation="copy";base.kernel="sdma";base.access="device";
      base.load_policy=base.store_policy="device-managed";base.instructions="sdma-dk-completed-copy";base.page_policy=o.pages;base.requested_bytes=bytes;
      auto label="SDMA "+o.sdma_mode+" | "+human_size(bytes)+"/worker | channels/group="+std::to_string(count)+" depth="+std::to_string(depth)+" batch="+std::to_string(batch)+" chunk="+human_size(chunk);
      std::cout<<"\n  "<<label<<std::endl;
      if(o.dry_run)continue;
      Session session(api,o,t,mapping,devices,bytes,count,depth,batch,chunk);session.errors();session.verify();
      if(o.warmup)session.phase(o.warmup);
      std::vector<Row> all;
      for(int sample=0;sample<o.samples;++sample)for(int attempt=0;attempt<=o.retries;++attempt) {
        session.phase(o.seconds);base.sample=sample;base.attempt=attempt;std::vector<Row> rows;
        for(auto& w:session.workers)rows.push_back(worker_row(session,*w,base));
        auto total=aggregate(rows,"total",o);bool valid=total.status=="ok";
        auto raw=rows;
        for(std::size_t g=0;g<mapping.size();++g) {
          std::vector<Row> own;for(auto& row:rows)if(row.group==int(g))own.push_back(row);
          raw.push_back(aggregate(own,"group:"+std::to_string(g),o));
        }
        raw.push_back(total);
        for(auto& row:raw) {reporter.add(row);all.push_back(row);}
        if(valid)break;
      }
      reporter.summary(all,label);
      std::vector<double> payload;
      for(auto& r:all)if(r.scope=="total"&&r.status=="ok")payload.push_back(r.payload_gbs);
      if(!payload.empty()) {std::sort(payload.begin(),payload.end());std::cout<<"  Payload median: "<<(payload[(payload.size()-1)/2]+payload[payload.size()/2])/2<<" GB/s (copied bytes once)\n";}
    }
  }
#endif
}
} // namespace memory_bench
