// SPDX-FileCopyrightText: 2026 HenryRenYz and vecops contributors
// SPDX-License-Identifier: MIT
#include "vecops/comm/Communication.h"
#include "vecops/comm/Copy.h"
#include "vecops/memory/LinuxBackend.h"
#include "vecops/comm/details/ReplicaRouting.h"
#include <algorithm>
#include <atomic>
#include <chrono>
#include <fcntl.h>
#include <limits>
#include <mutex>
#include <numeric>
#include <cstring>
#include <map>
#include <stdexcept>
#include <system_error>
#include <thread>
#include <unistd.h>
#include <sys/mman.h>
#include <sys/random.h>
#include <sys/syscall.h>
#include <linux/mempolicy.h>

namespace vecops::comm {
namespace {
namespace mm=vecops::memory::model;
template<class T>T take(mm::Result<T> result) {
  if(!result){const auto& error=std::get<mm::Error>(result.state);throw std::runtime_error(error.code+": "+error.message);}
  return std::move(result).value();
}
constexpr size_t max_ranks = 256;
constexpr size_t max_plans = 1024;
struct alignas(128) Header {
  uint64_t arrived = 0, ready = 0, relayed = 0, done = 0;
  uint64_t aborted = 0, capacity = 0, operation = 0, root = 0;
  uint64_t stopped = 0, payload_generation = 0;
  uint64_t descriptor = 0;
  uint64_t stages[16]{};
};
struct alignas(128) Descriptor {
  uint64_t operation = 0, root = 0, element_type = 0, route = 0;
  uint64_t sends[max_ranks]{}, receives[max_ranks]{};
};
// Immutable per-plan descriptors occupy a sparse DDR control mapping. Only
// initialized records fault pages. Payload retains 4/16/64 KiB alignment.
constexpr size_t descriptor_offset = 65536;
constexpr size_t header_bytes = 8 << 20;
static_assert(descriptor_offset+max_plans*sizeof(Descriptor)<=header_bytes);
Descriptor* descriptor(Header* header,size_t index) {
  if(index>=max_plans)throw std::runtime_error("invalid remote communication descriptor");
  return reinterpret_cast<Descriptor*>(reinterpret_cast<char*>(header)+descriptor_offset)+index;
}
uint64_t load(const uint64_t* p) { return __atomic_load_n(p, __ATOMIC_ACQUIRE); }
void store(uint64_t* p, uint64_t v) { __atomic_store_n(p, v, __ATOMIC_RELEASE); }
void hint() {
#if defined(__aarch64__)
  asm volatile("yield" ::: "memory");
#elif defined(__x86_64__) || defined(__i386__)
  asm volatile("pause" ::: "memory");
#else
  std::this_thread::yield();
#endif
}
// A failed operation and a completed shutdown are independent facts. During
// close every peer must still drop aliases and publish stopped, even if abort
// has already poisoned data operations. Never substitute abort for that proof.
void wait_stopped(const uint64_t* token,const std::chrono::steady_clock::time_point& deadline) {
  size_t spin=0;
  while(!load(token)) {
    hint();
    if((++spin&1023)==0) {
      if(std::chrono::steady_clock::now()>deadline)
        throw std::runtime_error("vecops comm stopped handshake timeout");
      if((spin&65535)==0)std::this_thread::yield();
    }
  }
}
size_t add(size_t a, size_t b) {
  if (b > SIZE_MAX-a) throw std::overflow_error("vecops comm byte count overflow");
  return a+b;
}
size_t mul(size_t a, size_t b) {
  if (a && b > SIZE_MAX/a) throw std::overflow_error("vecops comm byte count overflow");
  return a*b;
}
void system_check(bool ok, const char* operation) {
  if (!ok) throw std::system_error(errno, std::generic_category(), operation);
}
// Payload metadata is in a separate part of the existing control mapping.
// No root descriptor is interpreted on an unchanged hot replay generation.
struct PayloadWire {
  uint64_t bytes=0,page=0,offset=0;
  char name[128]{},identity[256]{},pool[512]{},address_space[256]{};
};
constexpr size_t payload_offset=7<<20;
static_assert(descriptor_offset+max_plans*sizeof(Descriptor)<=payload_offset);
static_assert(payload_offset+sizeof(PayloadWire)*(max_ranks+1)<=header_bytes);
PayloadWire* payloads(Header* header) {
  return reinterpret_cast<PayloadWire*>(reinterpret_cast<char*>(header)+payload_offset);
}
void wire_text(char* out,size_t capacity,const std::string& text) {
  if(text.size()>=capacity)throw std::invalid_argument("communication export metadata exceeds control wire limit");
  std::memcpy(out,text.c_str(),text.size()+1);
}
std::string wire_text(const char* text,size_t capacity) {
  const auto* end=static_cast<const char*>(std::memchr(text,0,capacity));
  if(!end)throw std::invalid_argument("unterminated communication export metadata");
  return {text,end};
}
struct Root {
  std::shared_ptr<mm::Allocation> allocation;
  std::shared_ptr<mm::SharedBacking> backing;
};
struct PayloadStore {
  std::shared_ptr<mm::MemoryRuntime> runtime;
  mm::ResourceView view;
  std::string pool;
  size_t page=0;
  mm::LinuxPageStore pages;
  PayloadStore(std::shared_ptr<mm::MemoryRuntime> value,const std::string& id):runtime(std::move(value)),pool(id) {
    if(!runtime || pool.empty())throw std::invalid_argument("communication requires a MemoryRuntime and explicit payload pool");
    view=take(runtime->view(take(runtime->current_scope())));
    const std::vector<std::string> pools{pool};
    const auto capability=take(runtime->mapping_capabilities(view,pools));
    if(!capability.shareable || !capability.remappable || !capability.partial_backing_map)
      throw std::invalid_argument("communication requires shareable remappable partial page mappings");
    page=capability.page_bytes;
    if(!page || page%static_cast<size_t>(sysconf(_SC_PAGESIZE)))
      throw std::invalid_argument("invalid communication backend page geometry");
  }
  Root allocate(size_t bytes,PayloadWire& wire) {
    static std::atomic<uint64_t> sequence{1};
    mm::BackingRequest request{"comm-"+std::to_string(sequence.fetch_add(1)),runtime->participant(),
      pool,bytes,page,mm::BackingMode::shareable_remappable,{}};
    auto allocation=std::make_shared<mm::Allocation>(take(runtime->allocate(view,request)));
    auto backing=take(allocation->export_backing());
    const auto handle=std::any_cast<mm::LinuxBackingHandle>(take(backing->export_handle()));
    PayloadWire next;next.bytes=bytes;next.page=page;next.offset=handle.backing_offset_bytes;
    wire_text(next.name,sizeof(next.name),handle.name);
    wire_text(next.identity,sizeof(next.identity),handle.physical_identity);
    wire_text(next.pool,sizeof(next.pool),pool);
    wire_text(next.address_space,sizeof(next.address_space),handle.allocation.address_space);
    wire=next;
    return {std::move(allocation),std::move(backing)};
  }
  std::shared_ptr<mm::SharedBacking> import(const PayloadWire& wire) {
    if(!wire.bytes || !wire.page || wire.bytes%wire.page || wire.page!=page)
      throw std::invalid_argument("invalid communication payload geometry");
    const auto name=wire_text(wire.name,sizeof(wire.name));
    const auto identity=wire_text(wire.identity,sizeof(wire.identity));
    const auto source_pool=wire_text(wire.pool,sizeof(wire.pool));
    const auto address_space=wire_text(wire.address_space,sizeof(wire.address_space));
    if(std::find(view.accessible_pools.begin(),view.accessible_pools.end(),source_pool)==view.accessible_pools.end())
      throw std::invalid_argument("communication payload pool is outside the runtime memory grant");
    mm::AllocationInfo allocation{name,address_space,wire.bytes,wire.bytes,wire.page,{wire.page},
      {{0,wire.bytes,source_pool}},true,true,1,mm::BackingMode::shareable_remappable};
    return take(pages.import_backing({name,identity,std::move(allocation),wire.offset}));
  }
};
struct PayloadWindow {
  std::unique_ptr<mm::VirtualReservation> reservation;
  std::vector<std::unique_ptr<mm::MappedRange>> mappings;
  std::vector<std::shared_ptr<mm::ImportedMemory>> imports;
  char* address=nullptr;
  size_t bytes=0;
  void reserve(PayloadStore& store,size_t total) {
    reservation=take(store.pages.reserve(total,store.page));
    address=static_cast<char*>(take(mm::LinuxPageStore::address(*reservation)));bytes=total;
  }
  void map(PayloadStore& store,size_t offset,size_t count,std::shared_ptr<mm::SharedBacking> backing,
           size_t physical_offset,bool write) {
    const std::vector<mm::AccessMode> access=write?std::vector{mm::AccessMode::read,mm::AccessMode::write}
                                                  :std::vector{mm::AccessMode::read};
    auto mapping=take(store.pages.map(*reservation,backing,{physical_offset,offset,count},access));
    const auto& info=backing->info();
    mm::ExternalMemoryView external{"comm-page-alias",{info.address_space,
      reinterpret_cast<uint64_t>(take(mapping->data()))},count,{},backing,!write,{}};
    if(info.segments.size()==1)external.known_pool=info.segments[0].pool;
    imports.push_back(take(store.runtime->import_memory(store.view,external)));
    mappings.push_back(std::move(mapping));
  }
};
struct Mapping {
  int fd=-1;
  Header* header=nullptr;
  char* data=nullptr;
  size_t mapped=0;
  uint64_t payload_generation=0;
  std::unique_ptr<PayloadWindow> window;
  Mapping()=default;
  Mapping(const Mapping&)=delete;
  Mapping& operator=(const Mapping&)=delete;
  Mapping(Mapping&& other)noexcept {
    std::swap(fd,other.fd);std::swap(header,other.header);std::swap(data,other.data);
    std::swap(mapped,other.mapped);std::swap(payload_generation,other.payload_generation);window=std::move(other.window);
  }
  ~Mapping() {
    window.reset();
    if(header)munmap(header,header_bytes);
    if(fd>=0)::close(fd);
  }
  void map_data(PayloadStore& store,size_t capacity,size_t segments) {
    const auto total=mul(capacity,segments);
    const auto generation=load(&header->payload_generation);
    if(mapped==total && payload_generation==generation)return;
    auto next=std::make_unique<PayloadWindow>();next->reserve(store,total);
    for(size_t index=0;index<segments;++index) {
      const auto& wire=payloads(header)[index];
      if(!wire.bytes)continue; // absent relay segments consume no physical backing
      if(wire.bytes>capacity || (!index && wire.bytes!=capacity))
        throw std::invalid_argument("communication payload capacity mismatch");
      next->map(store,mul(index,capacity),wire.bytes,store.import(wire),0,true);
    }
    data=next->address;mapped=total;payload_generation=generation;window=std::move(next);
  }
};
struct CopyShape {
  size_t bytes = 0, run = 1;
  std::vector<size_t> offsets;
  explicit CopyShape(const Layout& layout) {
    if (!layout.element_bytes || layout.sizes.size()!=layout.strides.size())
      throw std::invalid_argument("invalid comm tensor layout");
    size_t count=1;
    for (size_t i=0;i<layout.sizes.size();++i) {
      if (layout.sizes[i]<0 || layout.strides[i]<0)
        throw std::invalid_argument("negative comm tensor size/stride");
      count=mul(count,layout.sizes[i]);
    }
    bytes=mul(count,layout.element_bytes); run=layout.element_bytes;
    if (!count) return;
    size_t prefix=layout.sizes.size();
    while (prefix && (layout.sizes[prefix-1]==1 ||
           mul(layout.strides[prefix-1],layout.element_bytes)==run)) {
      --prefix; run=mul(run,layout.sizes[prefix]);
    }
    const size_t rows=bytes/run;
    offsets.reserve(rows);
    for (size_t row=0;row<rows;++row) {
      size_t index=row, offset=0;
      for (size_t d=prefix;d-->0;) {
        offset=add(offset,mul(index%layout.sizes[d],layout.strides[d]));
        index/=layout.sizes[d];
      }
      offsets.push_back(mul(offset,layout.element_bytes));
    }
  }
};
struct Job {
  const CopyShape* shape;
  const char* source;
  char* destination;
  bool pack;
  size_t tasks;
};
struct Batch {
  const std::vector<Job>* jobs;
  size_t count;
  const detail::CopyBackend* copy;
  static void work(void* opaque, uint32_t lane, uint32_t lanes) noexcept {
    const auto& batch=*static_cast<Batch*>(opaque);
    size_t begin=batch.count*lane/lanes, end=batch.count*(lane+1)/lanes, base=0;
    for (const auto& job:*batch.jobs) {
      size_t first=begin>base ? begin-base:0;
      size_t last=std::min(end>base ? end-base:0,job.tasks);
      if (first<last) {
        const auto& s=*job.shape;
        if (s.offsets.size()==1) {
          size_t a=s.bytes*first/job.tasks, b=s.bytes*last/job.tasks;
          batch.copy->bytes(job.destination+a,job.source+a,b-a);
        } else {
          size_t a=s.offsets.size()*first/job.tasks, b=s.offsets.size()*last/job.tasks;
          batch.copy->strided(job.destination,job.source,s.offsets.data(),a,b,s.run,job.pack);
        }
      }
      base+=job.tasks;
      if (base>=end) break;
    }
  }
};
} // namespace

// Failure cannot certify remote quiescence. Keep the root receipts alive for
// the process lifetime; authority crash inspection reports these conservatively.
void retain_unproven(std::vector<Root> roots,std::shared_ptr<mm::MemoryRuntime> runtime) {
  struct Retained {std::vector<Root> roots;std::shared_ptr<mm::MemoryRuntime> runtime;};
  static auto* mutex=new std::mutex;
  static auto* records=new std::vector<Retained>;
  std::lock_guard lock(*mutex);records->push_back({std::move(roots),std::move(runtime)});
}
struct Communicator::State {
  int rank, size;
  Options options;
  Exchange exchange;
  const VecopsThreadPoolV1* executor;
  std::vector<Mapping> peers;
  std::unique_ptr<PayloadStore> payload;
  std::vector<Root> roots;
  std::vector<std::pair<uint64_t,Root>> retired_roots;
  std::vector<std::weak_ptr<ExchangeBuffer>> exchanges;
  bool closed=false;
  std::mutex mutex;
  uint64_t epoch=0;
  uint64_t provision_epoch=0;
  uint64_t preparation_epoch=0;
  size_t registered_required=0;
  std::vector<size_t> registered_relays;
  bool sealed=false;
  size_t descriptor_count=0;
  Statistics stats;
  const detail::CopyBackend* copy=&detail::copy_backend();
  std::chrono::steady_clock::time_point deadline;
  ~State() {
    if(!closed && payload) {
      for(auto& [epoch,root]:retired_roots){(void)epoch;roots.push_back(std::move(root));}
      std::erase_if(roots,[](const auto& root){return !root.allocation;});
      if(!roots.empty())retain_unproven(std::move(roots),payload->runtime);
    }
    if(executor && executor->release)executor->release(executor->context);
  }
  void wait(const uint64_t* token, uint64_t expected) {
    size_t spin=0;
    while (load(token)<expected) {
      hint();
      if ((++spin&1023)==0) {
        for (auto& peer:peers)
          if (load(&peer.header->aborted)) throw std::runtime_error("vecops comm peer aborted");
        if (std::chrono::steady_clock::now()>deadline)
          throw std::runtime_error("vecops comm sequence timeout");
        if ((spin&65535)==0) std::this_thread::yield();
      }
    }
  }
  void reserve(size_t required,const std::vector<size_t>& needed={}) {
    auto& own=peers[rank];
    const size_t cap=load(&own.header->capacity);
    bool growth=required>cap || !cap;
    for(size_t source=0;source<needed.size();++source)
      growth|=needed[source]>payloads(own.header)[source+1].bytes;
    if(!growth)return;
    if(sealed)throw std::runtime_error("vecops comm sealed capacity exceeded; prepare first");
    size_t next=std::max(cap,payload->page);
    while(next<required)next=mul(next,2);
    std::vector<PayloadWire> descriptors(size+1);
    std::copy_n(payloads(own.header),size+1,descriptors.begin());
    // All slots retain the primary capacity as their virtual stride. Only
    // the primary publication needs its full size. A relay stores one
    // AllGather source, independent of the largest AllToAll inbox.
    roots.resize(size+1);
    std::vector<Root> replacements(size+1);
    if(next!=cap)replacements[0]=payload->allocate(next,descriptors[0]);
    for(size_t source=0;source<needed.size();++source) {
      auto& wire=descriptors[source+1];
      if(needed[source]<=wire.bytes)continue;
      const size_t bytes=mul(add(needed[source],payload->page-1)/payload->page,payload->page);
      if(bytes>next)throw std::logic_error("relay payload exceeds publication stride");
      replacements[source+1]=payload->allocate(bytes,wire);
    }
    for(size_t index=0;index<replacements.size();++index) {
      if(!replacements[index].allocation)continue;
      if(roots[index].allocation)
        retired_roots.emplace_back(std::max<uint64_t>(1,epoch),std::move(roots[index]));
      roots[index]=std::move(replacements[index]);
    }
    std::copy(descriptors.begin(),descriptors.end(),payloads(own.header));
    store(&own.header->capacity,next);
    store(&own.header->payload_generation,load(&own.header->payload_generation)+1);
    own.map_data(*payload,next,size+1);
    ++stats.capacity_growths;
  }
  void batch(std::vector<Job>& jobs) {
    size_t tasks=prepare_batch(jobs);
    if (!tasks) return;
    Batch b{&jobs,tasks,copy};
    uint32_t lanes=std::min<size_t>(options.threads,tasks);
    if(executor && lanes>1) {
      char message[512]{}; VecopsError error{sizeof(VecopsError),0,message,sizeof(message),0};
      int status=executor->parallel_for(executor->context,lanes,&b,Batch::work,&error);
      if(status) throw std::runtime_error(std::string("vecops comm executor: ")+message);
    } else Batch::work(&b,0,1);
  }
  size_t prepare_batch(std::vector<Job>& jobs) {
    size_t tasks=0;
    for(auto& job:jobs) {
      if (!job.shape->bytes) {job.tasks=0;continue;}
      job.tasks=std::max<size_t>(1,(job.shape->bytes+options.chunk_bytes-1)/options.chunk_bytes);
      if(job.shape->offsets.size()!=1) job.tasks=std::min(job.tasks,job.shape->offsets.size());
      tasks=add(tasks,job.tasks); stats.copy_bytes+=job.shape->bytes;
    }
    return tasks;
  }
  bool row_batch(std::vector<Job>& jobs,const std::vector<const void*>& sources,size_t rows,size_t stride) {
    if(!options.row_owned_copy || !rows || jobs.size()<2)return false;
    const size_t run=jobs.front().shape->run;
    auto* first=jobs.front().destination;
    for(size_t peer=0;peer<jobs.size();++peer)
      if(reinterpret_cast<uintptr_t>(jobs[peer].destination)!=reinterpret_cast<uintptr_t>(first)+peer*run)
        return false;
    detail::CopyRows job{sources.data(),first,jobs.size(),rows,run,stride};
    const auto work=copy->rows;
    const uint32_t lanes=std::min<size_t>(options.threads,rows);
    if(executor && lanes>1) {
      char message[512]{};VecopsError error{sizeof(VecopsError),0,message,sizeof(message),0};
      if(executor->parallel_for(executor->context,lanes,&job,work,&error))
        throw std::runtime_error(std::string("vecops comm row executor: ")+message);
    } else work(&job,0,1);
    const auto bytes=mul(mul(rows,run),jobs.size());
    stats.copy_bytes+=bytes;stats.row_copy_bytes+=bytes;++stats.row_copy_calls;
    return true;
  }
};
struct Plan::State {
  std::shared_ptr<Communicator> owner;
  Specification spec;
  std::vector<CopyShape> inputs, outputs;
  std::vector<size_t> send_offsets, receive_offsets;
  std::vector<Job> jobs;
  std::vector<CopyShape> relay_shapes;
  std::vector<uint64_t> validated_descriptors;
  std::vector<size_t> remote_inbox_offsets;
  std::vector<const void*> borrowed;
  std::vector<int> origins;
  std::vector<uint8_t> replicas;
  size_t descriptor_id=0, required=0, largest=0;
  bool relay=false;
  size_t interleaved_rows=0,interleaved_stride=0;
  size_t team_tasks=0;
  uint64_t team_epoch=0;
  bool team_rows=false;
  std::vector<int> copy_levels;
  std::vector<std::vector<int>> relay_targets;
  std::vector<size_t> relay_needed;
  explicit State(std::shared_ptr<Communicator> comm, Specification value):owner(std::move(comm)),spec(std::move(value)) {
    copy_levels=spec.copy_levels.empty()?owner->state_->options.copy_levels:spec.copy_levels;
    for(const auto& l:spec.inputs) inputs.emplace_back(l);
    for(const auto& l:spec.outputs) outputs.emplace_back(l);
    // Cold proof of a regular peer-interleaved destination. Ragged/irregular
    // shapes retain generic scheduling; runtime checks only pointer rebinding.
    if(spec.operation==Operation::AllGather && outputs.size()>1) {
      const auto& first=outputs.front();
      bool regular=first.run>0 && first.run<=256 && first.offsets.size()>1 && first.offsets[0]==0;
      const size_t stride=regular?first.offsets[1]:0;
      regular &= stride>=mul(outputs.size(),first.run);
      if(regular)for(size_t row=0;row<first.offsets.size();++row)
        regular &= first.offsets[row]==mul(row,stride);
      if(regular)for(const auto& shape:outputs)
        regular &= shape.run==first.run && shape.offsets==first.offsets;
      if(regular){interleaved_rows=first.offsets.size();interleaved_stride=stride;}
    }
    jobs.reserve(max_ranks); relay_shapes.reserve(max_ranks);
    send_offsets.resize(max_ranks); receive_offsets.resize(max_ranks);
    validated_descriptors.assign(max_ranks,UINT64_MAX);
    remote_inbox_offsets.resize(max_ranks);
    borrowed.resize(owner->size());origins.resize(owner->size());replicas.resize(owner->size());
    for(const auto& output:outputs)
      relay_shapes.emplace_back(Layout{{static_cast<int64_t>(output.bytes)},{1},1});
    size_t send=0,receive=0;
    for(size_t i=0;i<inputs.size();++i){send_offsets[i]=send;send=add(send,inputs[i].bytes);}
    for(size_t i=0;i<outputs.size();++i){receive_offsets[i]=receive;receive=add(receive,outputs[i].bytes);largest=std::max(largest,outputs[i].bytes);}
    required=std::max(send,spec.operation==Operation::AllToAll?receive:size_t(0));
    if(spec.operation==Operation::AllGather)required=std::max(required,largest);
    // Immutable for every lane: it must not depend on a leader publication
    // between ready and the following copy-preparation barrier.
    relay=spec.operation==Operation::AllGather && owner->state_->options.relay &&
          largest>=owner->state_->options.relay_threshold;
    relay_needed.assign(owner->size(),0);
    if(relay) {
      std::vector<uint64_t> bytes;bytes.reserve(outputs.size());
      for(const auto& output:outputs)bytes.push_back(output.bytes);
      for(const auto& partition:owner->state_->options.replica_levels)
        relay_targets.push_back(detail::balanced_relay_targets(partition,owner->rank(),bytes));
      for(size_t level=0;level<relay_targets.size();++level)if(copy_levels[level])
        for(size_t source=0;source<relay_needed.size();++source)
          if(relay_targets[level][source]==owner->rank())relay_needed[source]=outputs[source].bytes;
    }
  }
};
Communicator::Communicator(std::unique_ptr<State> state):state_(std::move(state)) {}
Communicator::~Communicator()=default;
Plan::Plan(std::unique_ptr<State> state):state_(std::move(state)) {}
Plan::~Plan()=default;
std::shared_ptr<Communicator> Communicator::create(int rank,int size,Options options,
    Exchange exchange,const VecopsThreadPoolV1* executor) {
  if(size<1 || size>static_cast<int>(max_ranks) || rank<0 || rank>=size || !options.threads || !options.chunk_bytes ||
     !options.timeout_ms || !options.memory_runtime || options.payload_pool.empty())
    throw std::invalid_argument("invalid vecops communicator dimensions/options");
  if(options.locality_groups.empty()) options.locality_groups.assign(size,0);
  if(options.locality_groups.size()!=static_cast<size_t>(size)) throw std::invalid_argument("invalid locality group map");
  if(options.replica_levels.empty()) {
    options.replica_levels.push_back(options.locality_groups);
    options.copy_levels.push_back(options.relay?1:0);
  }
  if(options.replica_levels.size()>16 || options.copy_levels.size()!=options.replica_levels.size())
    throw std::invalid_argument("invalid replica level count");
  for(size_t level=0;level<options.replica_levels.size();++level) {
    if(options.replica_levels[level].size()!=static_cast<size_t>(size) || (options.copy_levels[level]!=0 && options.copy_levels[level]!=1))
      throw std::invalid_argument("invalid replica partition/mode");
    if(level)for(int a=0;a<size;++a)for(int b=0;b<a;++b)
      if(options.replica_levels[level][a]==options.replica_levels[level][b] &&
         options.replica_levels[level-1][a]!=options.replica_levels[level-1][b])
        throw std::invalid_argument("replica partitions must be nested coarse-to-fine");
  }
  if(executor && (executor->abi_major!=VECOPS_THREAD_POOL_ABI_MAJOR || !executor->parallel_for))
    throw std::invalid_argument("unsupported vecops comm executor ABI");
  auto s=std::make_unique<State>(); s->rank=rank;s->size=size;s->options=std::move(options);s->executor=executor;s->exchange=exchange;
  if(executor && executor->retain) executor->retain(executor->context);
  s->payload=std::make_unique<PayloadStore>(s->options.memory_runtime,s->options.payload_pool);
  uint64_t random=0;
  system_check(getrandom(&random,sizeof(random),0)==sizeof(random),"vecops comm random identifier");
  std::string name="/vecops-"+std::to_string(getuid())+"-"+std::to_string(random);
  int fd=shm_open(name.c_str(),O_CREAT|O_EXCL|O_RDWR|O_CLOEXEC,0600);
  system_check(fd>=0,"vecops comm shm_open");
  struct FileGuard {int fd;~FileGuard(){if(fd>=0)::close(fd);}} file_guard{fd};
  struct Cleanup {std::string name;~Cleanup(){shm_unlink(name.c_str());}} cleanup{name};
  system_check(ftruncate(fd,header_bytes)==0,"vecops comm control size");
  auto names=exchange(name);
  if(names.size()!=static_cast<size_t>(size)) throw std::invalid_argument("invalid bootstrap exchange");
  s->peers.reserve(size);
  for(int p=0;p<size;++p) {
    Mapping m; m.fd=p==rank?fd:shm_open(names[p].c_str(),O_RDWR|O_CLOEXEC,0);
    if(p==rank)file_guard.fd=-1;
    system_check(m.fd>=0,"vecops comm import window");
    auto* control=mmap(nullptr,header_bytes,PROT_READ|PROT_WRITE,MAP_SHARED,m.fd,0);
    system_check(control!=MAP_FAILED,"vecops comm map control");
    m.header=static_cast<Header*>(control);
    s->peers.push_back(std::move(m));
  }
  if(!__atomic_is_lock_free(sizeof(uint64_t),&s->peers[rank].header->arrived))
    throw std::runtime_error("vecops comm requires lock-free shared uint64 atomics");
  // All peers own an fd before names are unlinked. No named SHM remains after bootstrap.
  exchange("mapped");
  s->reserve(std::max<size_t>(1,s->options.initial_capacity));
  exchange("allocated");
  for(auto& peer:s->peers)peer.map_data(*s->payload,load(&peer.header->capacity),size+1);
  return std::shared_ptr<Communicator>(new Communicator(std::move(s)));
}
std::shared_ptr<Plan> Communicator::prepare(Specification spec, bool defer_capacity) {
  const auto op=spec.operation;
  if(static_cast<uint32_t>(op)>static_cast<uint32_t>(Operation::SumFloat32))
    throw std::invalid_argument("unknown collective operation");
  if(spec.root<0 || spec.root>=state_->size) throw std::invalid_argument("invalid collective root");
  if((op==Operation::AllGather && (spec.inputs.size()!=1 || spec.outputs.size()!=static_cast<size_t>(state_->size))) ||
     (op==Operation::AllToAll && (spec.inputs.size()!=static_cast<size_t>(state_->size) || spec.outputs.size()!=static_cast<size_t>(state_->size))) ||
     ((op==Operation::Broadcast || op==Operation::SumFloat32) && (spec.inputs.size()!=1 || spec.outputs.size()!=1)) ||
     (op==Operation::Barrier && (!spec.inputs.empty() || !spec.outputs.empty())))
    throw std::invalid_argument("invalid collective tensor arity");
  if(!spec.copy_levels.empty()) {
    if(spec.copy_levels.size()!=state_->options.replica_levels.size() ||
       std::any_of(spec.copy_levels.begin(),spec.copy_levels.end(),[](int v){return v!=0 && v!=1;}))
      throw std::invalid_argument("invalid plan replica modes");
  }
  auto plan=std::shared_ptr<Plan>(new Plan(std::make_unique<Plan::State>(shared_from_this(),std::move(spec))));
  if(op==Operation::SumFloat32 && (plan->state_->spec.inputs[0].element_bytes!=4 ||
      plan->state_->spec.outputs[0].element_bytes!=4 ||
      (plan->state_->outputs[0].bytes && plan->state_->outputs[0].offsets.size()!=1)))
    throw std::invalid_argument("FP32 SUM requires float32 input and contiguous float32 output");
  for(const auto& shape:plan->state_->outputs) {
    auto offsets=shape.offsets;
    std::sort(offsets.begin(),offsets.end());
    for(size_t i=1;i<offsets.size();++i)
      if(offsets[i]-offsets[i-1]<shape.run)
        throw std::invalid_argument("overlapping collective output view");
  }
  {
    std::lock_guard lock(state_->mutex);
    if(state_->closed)throw std::runtime_error("cannot prepare a collective after communicator close");
    if(state_->sealed)throw std::runtime_error("cannot prepare a collective after communicator seal");
    if(state_->descriptor_count==max_plans)throw std::runtime_error("vecops communication plan capacity exceeded");
    auto& p=*plan->state_;
    state_->registered_required=std::max(state_->registered_required,p.required);
    if(state_->registered_relays.empty())state_->registered_relays.assign(state_->size,0);
    for(size_t source=0;source<p.relay_needed.size();++source)
      state_->registered_relays[source]=std::max(state_->registered_relays[source],p.relay_needed[source]);
    p.descriptor_id=state_->descriptor_count++;
    auto* d=descriptor(state_->peers[state_->rank].header,p.descriptor_id);
    d->operation=static_cast<uint64_t>(p.spec.operation);d->root=p.spec.root;
    d->element_type=p.spec.element_type;
    if(op==Operation::AllGather) {
      uint64_t route=1469598103934665603ULL;
      for(size_t level=0;level<p.copy_levels.size();++level) {
        route=(route^static_cast<uint64_t>(p.copy_levels[level]))*1099511628211ULL;
        for(int group:state_->options.replica_levels[level])route=(route^static_cast<uint64_t>(group))*1099511628211ULL;
      }
      route=(route^static_cast<uint64_t>(state_->options.relay && p.largest>=state_->options.relay_threshold))*1099511628211ULL;
      d->route=route;
    }
    for(int i=0;i<state_->size;++i) {
      d->sends[i]=op==Operation::AllToAll?p.inputs[i].bytes:(p.inputs.empty()?0:p.inputs[0].bytes);
      d->receives[i]=op==Operation::AllGather || op==Operation::AllToAll?p.outputs[i].bytes:(p.outputs.empty()?0:p.outputs[0].bytes);
    }
    // Before the first invocation there are no publication readers. Provision
    // exact registered relay requirements cold, so metadata-only preparation
    // followed by seal does not depend on a hidden warm-up allocation.
    if(!state_->epoch && !defer_capacity) {
      state_->reserve(p.required,p.relay_needed);
    }
    ++state_->stats.plans;
  }
  return plan;
}
bool Communicator::provision_registered_capacity(bool allow_growth) {
  auto& c=*state_;
  std::lock_guard lock(c.mutex);
  const auto prefix="vecops-capacity/v1/"+std::to_string(++c.provision_epoch)+"/"+std::to_string(c.epoch)+"/";
  // Each participant reports local failures at the next checkpoint rather
  // than leaving peers blocked waiting for a publication that will not exist.
  auto checkpoint=[&](const char* phase,const std::string& error={},bool grew=false) {
    const auto tag=prefix+phase+"/";
    auto values=c.exchange(tag+(error.empty()?(grew?"grown":"ok"):"error:"+error));
    if(values.size()!=static_cast<size_t>(c.size))throw std::runtime_error("invalid capacity preparation exchange");
    bool any=false;
    for(const auto& value:values) {
      if(value==tag+"grown")any=true;
      else if(value!=tag+"ok")throw std::runtime_error("vecops capacity preparation mismatch or peer failure: "+value);
    }
    return any;
  };
  auto attempt=[](auto&& action) {
    try{action();return std::string{};}
    catch(const std::exception& error){return std::string(error.what());}
    catch(...){return std::string("unknown capacity preparation failure");}
  };
  try {
    const auto entry=attempt([&] {
      if(c.closed || load(&c.peers[c.rank].header->stopped))throw std::runtime_error("communicator is closed");
      if(load(&c.peers[c.rank].header->aborted))throw std::runtime_error("communicator is aborted");
      if(load(&c.peers[c.rank].header->done)!=c.epoch)throw std::runtime_error("local publication has not stopped");
    });
    checkpoint("entered",entry);
    c.deadline=std::chrono::steady_clock::now()+std::chrono::milliseconds(c.options.timeout_ms);
    checkpoint("stopped",attempt([&] {
      for(auto& peer:c.peers)c.wait(&peer.header->done,c.epoch);
    }));
    auto& own=c.peers[c.rank];
    const auto cap=load(&own.header->capacity);
    const auto target=std::max(cap,c.registered_required);
    auto needed=c.registered_relays;
    if(needed.empty())needed.assign(c.size,0);
    bool rebuild=target>cap || !cap;
    for(int source=0;source<c.size;++source) {
      const size_t existing=payloads(own.header)[source+1].bytes;
      rebuild|=needed[source]>existing;
      needed[source]=std::max(needed[source],existing);
    }
    const bool needs_growth=checkpoint(allow_growth ? "validated" : "probe",attempt([&] {
      if(rebuild && c.sealed)throw std::runtime_error("sealed communicator requires unregistered capacity");
    }),rebuild);
    if(needs_growth && !allow_growth)return true;
    // A remote imported mapping keeps physical pages alive even after an
    // owner's receipt is released. Remove all aliases BEFORE refunding roots.
    for(auto& peer:c.peers){peer.window.reset();peer.data=nullptr;peer.mapped=0;}
    checkpoint("unmapped");
    c.retired_roots.clear();
    const auto growths=c.stats.capacity_growths;
    const auto allocation_error=attempt([&] {
      if(rebuild) {
        c.roots.clear();
        std::fill_n(payloads(own.header),c.size+1,PayloadWire{});
        store(&own.header->capacity,0);
        store(&own.header->payload_generation,load(&own.header->payload_generation)+1);
        c.reserve(std::max<size_t>(1,target),needed);
      }
    });
    const bool grew=checkpoint("published",allocation_error,c.stats.capacity_growths!=growths);
    checkpoint("mapped",attempt([&] {
      for(auto& peer:c.peers)peer.map_data(*c.payload,load(&peer.header->capacity),c.size+1);
    }));
    return grew;
  }catch(...){abort();throw;}
}
bool Communicator::preparation_consensus(bool local_success) {
  auto& c=*state_;
  std::lock_guard lock(c.mutex);
  const auto prefix="vecops-preparation/v1/"+std::to_string(++c.preparation_epoch)+"/";
  try {
    const bool stopped=c.closed || load(&c.peers[c.rank].header->stopped) || load(&c.peers[c.rank].header->aborted);
    const auto values=c.exchange(prefix+(stopped?"error:communicator stopped":(local_success?"true":"false")));
    if(values.size()!=static_cast<size_t>(c.size))throw std::runtime_error("invalid preparation consensus exchange");
    bool result=true;
    for(const auto& value:values) {
      if(value==prefix+"false")result=false;
      else if(value!=prefix+"true")throw std::runtime_error("vecops preparation consensus failure or order mismatch: "+value);
    }
    return result;
  }catch(...){abort();throw;}
}
Statistics Communicator::statistics() const {std::lock_guard lock(state_->mutex);return state_->stats;}
void Communicator::seal(){std::lock_guard lock(state_->mutex);state_->sealed=true;}
void Communicator::abort() noexcept {if(state_ && !state_->peers.empty())store(&state_->peers[state_->rank].header->aborted,1);}
int Communicator::rank()const noexcept{return state_->rank;}
int Communicator::size()const noexcept{return state_->size;}
const Specification& Plan::specification()const noexcept{return state_->spec;}
void Plan::execute(void* const* outputs,const void* const* inputs) {
  execute_impl(outputs,inputs,nullptr);
}
void Plan::consume(const void* const* inputs,const Consumer& consumer) {
  if(state_->spec.operation!=Operation::AllGather || !consumer)
    throw std::invalid_argument("a borrowed consumer requires AllGather and a callback");
  execute_impl(nullptr,inputs,&consumer);
}
const VecopsThreadPoolV1* Plan::thread_pool()const noexcept{return state_->owner->state_->executor;}
uint32_t Plan::workers()const noexcept{return state_->owner->state_->options.threads;}
void Plan::execute_on_team(void* const* outputs,const void* const* inputs,
    const VecopsTeamContextV1& team,const TeamConsumer* consumer) {
  if(!team.worker_count || team.worker_id>=team.worker_count || (team.worker_count>1 && !team.barrier))
    throw std::invalid_argument("invalid resident communication team");
  if(consumer && (state_->spec.operation!=Operation::AllGather || !*consumer))
    throw std::invalid_argument("a team consumer requires AllGather and a callback");
  execute_impl(outputs,inputs,nullptr,&team,consumer);
}
void Plan::execute_impl(void* const* outputs,const void* const* inputs,const Consumer* consumer,
    const VecopsTeamContextV1* team,const TeamConsumer* team_consumer) {
  auto& p=*state_; auto& c=*p.owner->state_;
  const uint32_t lane=team?team->worker_id:0;
  const bool borrowed=consumer || team_consumer;
  const auto op=p.spec.operation; const int rank=c.rank,size=c.size;
  const bool early_snapshot=c.options.early_snapshot && op==Operation::AllGather;
  auto* own=c.peers[rank].header;
  std::unique_lock lock(c.mutex,std::defer_lock);
  auto sync=[&] {
    if(team && team->worker_count>1) {
      char message[256]{};VecopsError error{sizeof(VecopsError),0,message,sizeof(message),0};
      if(team->barrier(team->context,&error))throw std::runtime_error(message[0]?message:"communication team cancelled");
    }
  };
  // Protocol work is a one-lane action in the SAME resident submission. Other
  // lanes wait on the local generation, never all polling remote cache lines.
  auto control=[&](auto&& work){if(lane==0)work();sync();};
  auto prepare_copy=[&]{if(team)p.team_tasks=c.prepare_batch(p.jobs);};
  auto copy=[&] {
    if(!team){c.batch(p.jobs);return;}
    const uint32_t lanes=std::min<size_t>({c.options.threads,team->worker_count,p.team_tasks});
    if(lane<lanes){Batch b{&p.jobs,p.team_tasks,c.copy};Batch::work(&b,lane,lanes);}
    sync();
  };
  // AllGather's source snapshot touches only its own private publication
  // slot. It can precede peer arrival; remote maps/descriptor checks must
  // complete after ready acquires and before any consumer/output access.
  auto validate_and_map=[&] {
    const auto* local=descriptor(own,p.descriptor_id);
      for(int peer=0;peer<size;++peer) {
        const auto* h=c.peers[peer].header;
        if(p.validated_descriptors[peer]!=h->descriptor) {
          const auto* remote=descriptor(c.peers[peer].header,h->descriptor);
          if(remote->operation!=local->operation || remote->root!=local->root || remote->element_type!=local->element_type || remote->route!=local->route ||
             remote->sends[rank]!=local->receives[peer] || local->sends[peer]!=remote->receives[rank])
            throw std::runtime_error("vecops collective sequence/count mismatch");
          size_t offset=0;for(int source=0;source<rank;++source)offset=add(offset,remote->receives[source]);
          p.remote_inbox_offsets[peer]=offset;p.validated_descriptors[peer]=h->descriptor;
        }
        c.peers[peer].map_data(*c.payload,load(&h->capacity),size+1);
      }
  };
  try {
    control([&] {
      lock.lock();
      if(c.closed || load(&own->stopped))throw std::runtime_error("vecops communicator is closed");
      if(load(&own->aborted)) throw std::runtime_error("vecops communicator is aborted");
      c.deadline=std::chrono::steady_clock::now()+std::chrono::milliseconds(c.options.timeout_ms);
      p.team_epoch=++c.epoch;
      for(auto& peer:c.peers)c.wait(&peer.header->done,p.team_epoch-1);
      std::erase_if(c.retired_roots,[&](const auto& retired){return retired.first<p.team_epoch;});
      c.reserve(p.required,p.relay_needed);
      own->descriptor=p.descriptor_id;
      store(&own->arrived,p.team_epoch);
      if(!early_snapshot) {
      for(auto& peer:c.peers)c.wait(&peer.header->arrived,p.team_epoch);
      validate_and_map();
      }
      p.jobs.clear();
      if(op==Operation::AllToAll) {
        for(int peer=0;peer<size;++peer)
          p.jobs.push_back({&p.inputs[peer],static_cast<const char*>(inputs[peer]),c.peers[peer].data+p.remote_inbox_offsets[peer],true,0});
      } else if(op!=Operation::Barrier && (op!=Operation::Broadcast || rank==p.spec.root))
        p.jobs.push_back({&p.inputs[0],static_cast<const char*>(inputs[0]),c.peers[rank].data,true,0});
      prepare_copy();
    });
    if(op!=Operation::Barrier)copy();
    // Only lane 0 touches protocol state. All lanes next meet at the copy
    // preparation below; joining once here as well would be redundant.
    if(lane==0) {
      store(&own->ready,p.team_epoch);
      for(auto& peer:c.peers)c.wait(&peer.header->ready,p.team_epoch);
      if(early_snapshot)validate_and_map();
      std::iota(p.origins.begin(),p.origins.end(),0);std::fill(p.replicas.begin(),p.replicas.end(),0);
    }
    if(p.relay)for(size_t level=0;level<c.options.replica_levels.size();++level) {
      if(!p.copy_levels[level])continue;
      control([&] {
        const auto& groups=c.options.replica_levels[level];
        p.jobs.clear();
        for(int source=0;source<size;++source) {
          if(groups[source]==groups[rank])continue;
          const int rep=p.relay_targets[level][source];
          const int origin=p.origins[source];
          const char* from=c.peers[origin].data+(p.replicas[source]?(source+1)*load(&c.peers[origin].header->capacity):0);
          char* to=c.peers[rep].data+(source+1)*load(&c.peers[rep].header->capacity);
          if(rep==rank && from!=to)p.jobs.push_back({&p.relay_shapes[source],from,to,true,0});
          p.origins[source]=rep;p.replicas[source]=1;
        }
        prepare_copy();
      });
      copy();
      if(lane==0) {
        store(&own->stages[level],p.team_epoch);
        // Includes readers in every group before a later level reuses a slot.
        for(auto& peer:c.peers)c.wait(&peer.header->stages[level],p.team_epoch);
      }
    }
    control([&] {
      p.jobs.clear();p.team_rows=false;
      if(op==Operation::AllGather) {
        for(int source=0;source<size;++source) {
          const char* base=c.peers[source].data;
          if(p.replicas[source]) {
            int replica=p.origins[source];
            base=c.peers[replica].data+(source+1)*load(&c.peers[replica].header->capacity);
          }
          p.borrowed[source]=base;
          if(!borrowed)p.jobs.push_back({&p.outputs[source],base,static_cast<char*>(outputs[source]),false,0});
        }
      } else if(op==Operation::AllToAll) {
        for(int source=0;source<size;++source)p.jobs.push_back({&p.outputs[source],c.peers[rank].data+p.receive_offsets[source],static_cast<char*>(outputs[source]),false,0});
      } else if(op==Operation::Broadcast)
        p.jobs.push_back({&p.outputs[0],c.peers[p.spec.root].data,static_cast<char*>(outputs[0]),false,0});
      if(team && !borrowed && p.interleaved_rows && c.options.row_owned_copy && p.jobs.size()>1) {
        p.team_rows=true;const auto run=p.jobs.front().shape->run;
        for(size_t peer=0;peer<p.jobs.size();++peer)
          p.team_rows &= reinterpret_cast<uintptr_t>(p.jobs[peer].destination)==reinterpret_cast<uintptr_t>(p.jobs.front().destination)+peer*run;
        if(p.team_rows){const auto bytes=mul(mul(p.interleaved_rows,run),p.jobs.size());c.stats.copy_bytes+=bytes;c.stats.row_copy_bytes+=bytes;++c.stats.row_copy_calls;}
      }
      if(!p.team_rows)prepare_copy();
    });
    if(op==Operation::SumFloat32) {
      auto* out=static_cast<float*>(outputs[0]);const size_t count=p.outputs[0].bytes/4;
      const auto lanes=team?std::min(team->worker_count,c.options.threads):1u;
      if(lane<lanes)for(size_t i=count*lane/lanes;i<count*(lane+1)/lanes;++i) {
        float sum=0;for(int r=0;r<size;++r)sum+=reinterpret_cast<const float*>(c.peers[r].data)[i];out[i]=sum;
      }
      sync();
    } else if(team && p.team_rows) {
      detail::CopyRows job{p.borrowed.data(),p.jobs.front().destination,p.jobs.size(),p.interleaved_rows,p.jobs.front().shape->run,p.interleaved_stride};
      const auto lanes=std::min<size_t>({c.options.threads,team->worker_count,p.interleaved_rows});
      if(lane<lanes)c.copy->rows(&job,lane,lanes);
      sync();
    } else if(team){if(!borrowed)copy();}
    else if(borrowed || !p.interleaved_rows || !c.row_batch(p.jobs,p.borrowed,p.interleaved_rows,p.interleaved_stride))c.batch(p.jobs);
    if(consumer)(*consumer)(p.borrowed.data(),p.borrowed.size());
    if(team_consumer){(*team_consumer)(p.borrowed.data(),p.borrowed.size(),*team);sync();}
    control([&] {
      if(load(&own->aborted))throw std::runtime_error("vecops communication cancelled");
      store(&own->done,p.team_epoch);++c.stats.calls;
    });
  }catch(...){p.owner->abort();throw;}
}

namespace {
constexpr uint32_t max_slots=64;
struct alignas(128) SlotControl {uint64_t ready=0,retired=0;};
struct LeaseControl {uint64_t aborted=0,stopped=0;SlotControl slot[max_slots];};

}
struct ExchangeBuffer::State {
  std::shared_ptr<Communicator> owner;
  std::vector<Mapping> peers;
  PayloadWindow producer,consumer;
  Root root;
  ~State() {
    if(!closed && root.backing) {
      std::vector<Root> roots;roots.push_back(std::move(root));
      retain_unproven(std::move(roots),owner->state_->payload->runtime);
    }
  }
  bool closed=false;
  std::mutex close_mutex;
  size_t bytes=0,stride=0,slab=0;
  uint32_t count=0;
  std::vector<uint64_t> versions;
  std::vector<uint8_t> phase;
  std::array<std::mutex,max_slots> mutex;
  LeaseControl* control(int rank)const{return reinterpret_cast<LeaseControl*>(peers[rank].header);}
  void check(uint32_t slot)const {
    if(closed || load(&control(owner->rank())->stopped))throw std::runtime_error("vecops exchange is closed");
    if(slot>=count)throw std::out_of_range("vecops exchange slot index");
    if(load(&control(owner->rank())->aborted))throw std::runtime_error("vecops exchange aborted");
  }
  void wait(const uint64_t* p,uint64_t version,const std::chrono::steady_clock::time_point& deadline) {
    size_t spin=0;
    while(load(p)<version) {
      hint();
      if((++spin&1023)==0) {
        for(auto& peer:peers)if(load(&reinterpret_cast<LeaseControl*>(peer.header)->aborted))
          throw std::runtime_error("vecops exchange peer aborted");
        for(auto& peer:owner->state_->peers)if(load(&peer.header->aborted))
          throw std::runtime_error("vecops communicator aborted");
        if(std::chrono::steady_clock::now()>deadline)throw std::runtime_error("vecops exchange lease timeout");
        if((spin&65535)==0)std::this_thread::yield();
      }
    }
  }
};
ExchangeBuffer::ExchangeBuffer(std::unique_ptr<State> state):state_(std::move(state)){}
ExchangeBuffer::~ExchangeBuffer(){
  try{close();}catch(...){
    abort();
    if(state_->root.backing) {
      std::vector<Root> roots;roots.push_back(std::move(state_->root));
      retain_unproven(std::move(roots),state_->owner->state_->payload->runtime);
    }
  }
}
size_t ExchangeBuffer::block_stride()const noexcept{return state_->stride;}
size_t ExchangeBuffer::block_bytes()const noexcept{return state_->bytes;}
uint32_t ExchangeBuffer::slots()const noexcept{return state_->count;}
void* ExchangeBuffer::producer(uint32_t slot)const{state_->check(slot);return state_->producer.address+slot*state_->slab;}
const void* ExchangeBuffer::consumer(uint32_t slot)const{state_->check(slot);return state_->consumer.address+slot*state_->slab;}
void ExchangeBuffer::abort()noexcept{store(&state_->control(state_->owner->rank())->aborted,1);}
void ExchangeBuffer::begin(uint32_t slot) {
  auto& s=*state_;
  try {
    s.check(slot);
    std::lock_guard lock(s.mutex[slot]);
    s.check(slot);
    if(s.phase[slot])throw std::runtime_error("exchange slot still leased or already writing");
    const auto deadline=std::chrono::steady_clock::now()+std::chrono::milliseconds(s.owner->state_->options.timeout_ms);
    for(int r=0;r<s.owner->size();++r)s.wait(&s.control(r)->slot[slot].retired,s.versions[slot],deadline);
    if(s.versions[slot]==UINT64_MAX)throw std::overflow_error("exchange version overflow");
    ++s.versions[slot];s.phase[slot]=1;
  }catch(...){abort();throw;}
}
void ExchangeBuffer::publish(uint32_t slot) {
  auto& s=*state_;
  try {
    s.check(slot);
    std::lock_guard lock(s.mutex[slot]);
    s.check(slot);
    if(s.phase[slot]!=1)throw std::runtime_error("exchange publish requires begin");
    store(&s.control(s.owner->rank())->slot[slot].ready,s.versions[slot]);
    const auto deadline=std::chrono::steady_clock::now()+std::chrono::milliseconds(s.owner->state_->options.timeout_ms);
    for(int r=0;r<s.owner->size();++r)s.wait(&s.control(r)->slot[slot].ready,s.versions[slot],deadline);
    s.phase[slot]=2;
  }catch(...){abort();throw;}
}
void ExchangeBuffer::retire(uint32_t slot) {
  auto& s=*state_;
  try {
    s.check(slot);
    std::lock_guard lock(s.mutex[slot]);
    s.check(slot);
    if(s.phase[slot]!=2)throw std::runtime_error("exchange retire requires a published lease");
    store(&s.control(s.owner->rank())->slot[slot].retired,s.versions[slot]);s.phase[slot]=0;
  }catch(...){abort();throw;}
}
std::shared_ptr<ExchangeBuffer> Communicator::prepare_exchange(size_t bytes,uint32_t slots,PagePlacement placement) {
  auto& c=*state_;std::lock_guard lock(c.mutex);
  if(c.closed)throw std::runtime_error("cannot allocate exchange storage after close");
  if(c.sealed)throw std::runtime_error("cannot allocate exchange storage after seal");
  if(!slots || slots>max_slots || (placement!=PagePlacement::Source && placement!=PagePlacement::Destination))
    throw std::invalid_argument("invalid exchange storage options");
  auto s=std::make_unique<ExchangeBuffer::State>();s->owner=shared_from_this();s->bytes=bytes;s->count=slots;
  const size_t page=c.payload->page;
  s->stride=mul(std::max<size_t>(1,add(bytes,page-1)/page),page);
  s->slab=mul(s->stride,c.size);size_t total=mul(s->slab,slots);
  s->versions.assign(slots,0);s->phase.assign(slots,0);
  const auto metadata=std::to_string(bytes)+":"+std::to_string(slots)+":"+std::to_string(static_cast<uint32_t>(placement));
  for(const auto& remote:c.exchange(metadata))if(remote!=metadata)throw std::invalid_argument("exchange storage metadata mismatch");
  uint64_t random=0;system_check(getrandom(&random,sizeof(random),0)==sizeof(random),"vecops exchange random");
  std::string name="/vecops-exchange-"+std::to_string(getuid())+"-"+std::to_string(random);
  struct Cleanup {std::string name;~Cleanup(){shm_unlink(name.c_str());}} cleanup{name};
  Mapping own;own.fd=shm_open(name.c_str(),O_CREAT|O_EXCL|O_RDWR|O_CLOEXEC,0600);
  system_check(own.fd>=0,"vecops exchange shm_open");
  system_check(ftruncate(own.fd,header_bytes)==0,"vecops exchange size");
  auto names=c.exchange(name);s->peers.reserve(c.size);
  for(int r=0;r<c.size;++r) {
    Mapping m;m.fd=r==c.rank?own.fd:shm_open(names.at(r).c_str(),O_RDWR|O_CLOEXEC,0);
    if(r==c.rank)own.fd=-1;
    system_check(m.fd>=0,"vecops exchange import");
    auto* control=mmap(nullptr,header_bytes,PROT_READ|PROT_WRITE,MAP_SHARED,m.fd,0);
    system_check(control!=MAP_FAILED,"vecops exchange control");m.header=static_cast<Header*>(control);
    s->peers.push_back(std::move(m));
  }
  c.exchange("mapped");
  auto& local=s->peers[c.rank];
  s->root=c.payload->allocate(total,payloads(local.header)[0]);
  local.map_data(*c.payload,total,1);
  c.exchange("placed");
  s->producer.reserve(*c.payload,total);s->consumer.reserve(*c.payload,total);
  std::vector<std::shared_ptr<mm::SharedBacking>> imported;
  for(auto& peer:s->peers)imported.push_back(c.payload->import(payloads(peer.header)[0]));
  for(uint32_t slot=0;slot<slots;++slot)for(int peer=0;peer<c.size;++peer) {
    size_t offset=slot*s->slab+peer*s->stride;
    int producer_rank=placement==PagePlacement::Source?c.rank:peer;
    int consumer_rank=placement==PagePlacement::Source?peer:c.rank;
    size_t producer_peer=placement==PagePlacement::Source?peer:c.rank;
    size_t consumer_peer=placement==PagePlacement::Source?c.rank:peer;
    s->producer.map(*c.payload,offset,s->stride,imported[producer_rank],slot*s->slab+producer_peer*s->stride,true);
    s->consumer.map(*c.payload,offset,s->stride,imported[consumer_rank],slot*s->slab+consumer_peer*s->stride,false);
  }
  // Fault both alias page tables cold. Physical pages were already placed by
  // their owners; these reads do not move pages or make a second payload copy.
  volatile unsigned char touched=0;
  for(size_t i=0;i<total;i+=page) {
    touched=static_cast<unsigned char>(touched^s->producer.address[i]);
    touched=static_cast<unsigned char>(touched^s->consumer.address[i]);
  }
  (void)touched;c.exchange("prepared");
  auto result=std::shared_ptr<ExchangeBuffer>(new ExchangeBuffer(std::move(s)));
  c.exchanges.push_back(result);return result;
}
void ExchangeBuffer::close() {
  auto& s=*state_;
  std::lock_guard close_lock(s.close_mutex);
  if(s.closed)return;
  std::vector<std::unique_lock<std::mutex>> locks;
  for(uint32_t slot=0;slot<s.count;++slot)locks.emplace_back(s.mutex[slot]);
  if(std::any_of(s.phase.begin(),s.phase.end(),[](auto p){return p!=0;}))
    throw std::runtime_error("cannot close exchange with active producer/consumer leases");
  // Closing certifies that local readers/writers have stopped. Drop every
  // imported alias before acknowledging, so the final owner can refund roots.
  s.producer.imports.clear();s.consumer.imports.clear();
  s.producer.mappings.clear();s.consumer.mappings.clear();
  for(auto& peer:s.peers){peer.window.reset();peer.data=nullptr;peer.mapped=0;}
  store(&s.control(s.owner->rank())->stopped,1);
  const auto deadline=std::chrono::steady_clock::now()+std::chrono::milliseconds(s.owner->state_->options.timeout_ms);
  for(int rank=0;rank<s.owner->size();++rank)wait_stopped(&s.control(rank)->stopped,deadline);
  s.root={};s.closed=true;
}
void Communicator::close() {
  auto& c=*state_;
  std::lock_guard lock(c.mutex);
  if(c.closed)return;
  // The embedding drains its submission queue before this cold operation.
  for(auto& weak:c.exchanges)if(auto exchange=weak.lock())exchange->close();
  for(auto& peer:c.peers){peer.window.reset();peer.data=nullptr;peer.mapped=0;}
  store(&c.peers[c.rank].header->stopped,1);
  c.deadline=std::chrono::steady_clock::now()+std::chrono::milliseconds(c.options.timeout_ms);
  for(auto& peer:c.peers)wait_stopped(&peer.header->stopped,c.deadline);
  c.retired_roots.clear();c.roots.clear();c.closed=true;
}
} // namespace vecops::comm
