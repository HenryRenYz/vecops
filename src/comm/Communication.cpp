// SPDX-FileCopyrightText: 2026 HenryRenYz and vecops contributors
// SPDX-License-Identifier: MIT
#include "vecops/comm/Communication.h"
#include "vecops/comm/Copy.h"
#include "vecops/comm/details/ReplicaRouting.h"
#include <algorithm>
#include <atomic>
#include <chrono>
#include <fcntl.h>
#include <limits>
#include <mutex>
#include <numeric>
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
constexpr size_t max_ranks = 256;
constexpr size_t max_plans = 1024;
struct alignas(128) Header {
  uint64_t arrived = 0, ready = 0, relayed = 0, done = 0;
  uint64_t aborted = 0, capacity = 0, operation = 0, root = 0;
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
struct Mapping {
  int fd = -1;
  Header* header = nullptr;
  char* data = nullptr;
  size_t mapped = 0;
  Mapping() = default;
  Mapping(const Mapping&) = delete;
  Mapping& operator=(const Mapping&) = delete;
  Mapping(Mapping&& other) noexcept {
    std::swap(fd,other.fd); std::swap(header,other.header);
    std::swap(data,other.data); std::swap(mapped,other.mapped);
  }
  ~Mapping() {
    if (data) munmap(data,mapped);
    if (header) munmap(header,header_bytes);
    if (fd>=0) close(fd);
  }
  void map_data(size_t bytes) {
    if (mapped == bytes) return;
    auto* next = mmap(nullptr, bytes, PROT_READ|PROT_WRITE, MAP_SHARED,fd,header_bytes);
    system_check(next != MAP_FAILED,"vecops comm mmap payload");
    if (data) munmap(data,mapped);
    data = static_cast<char*>(next); mapped = bytes;
  }
};
// Temporarily bind ONLY the allocating caller. Existing workers retain their
// policy. The exact touched pages are subsequently queried when strict.
struct Placement {
  std::vector<unsigned long> previous, desired;
  int mode = MPOL_DEFAULT;
  bool changed = false, strict = false;
  int node = -1;
  explicit Placement(int target, bool require): strict(require), node(target) {
    if (target<0) return;
    const size_t bits = std::max<size_t>(1024, target+1);
    previous.resize((bits+63)/64); desired.resize(previous.size());
    if (syscall(SYS_get_mempolicy,&mode,previous.data(),previous.size()*64,nullptr,0)) {
      if (strict) system_check(false,"vecops comm get_mempolicy");
      return;
    }
    desired[target/64] |= 1UL<<(target%64);
    changed = syscall(SYS_set_mempolicy,MPOL_BIND,desired.data(),desired.size()*64)==0;
    if (!changed && strict) system_check(false,"vecops comm set_mempolicy");
  }
  ~Placement() {
    if (changed) syscall(SYS_set_mempolicy,mode,previous.data(),previous.size()*64);
  }
  void touch(char* first, size_t bytes) {
    const size_t page = static_cast<size_t>(sysconf(_SC_PAGESIZE));
    // shm/tmpfs policy belongs to the shared object, not only this caller.
    // Install it before first touch so all mappings retain the chosen target.
    if (changed && bytes) {
      const int result=syscall(SYS_mbind,first,bytes,MPOL_BIND|MPOL_F_STATIC_NODES,
                               desired.data(),desired.size()*64,0);
      if (result && strict) system_check(false,"vecops comm bind shared object");
    }
    for (size_t i=0;i<bytes;i+=page) first[i]=0;
    if (strict && node>=0 && bytes) {
      int policy=0;std::vector<unsigned long> mask(desired.size());
      system_check(syscall(SYS_get_mempolicy,&policy,mask.data(),mask.size()*64,first,MPOL_F_ADDR)==0,
                   "vecops comm query shared policy");
      if ((policy&~MPOL_F_STATIC_NODES)!=MPOL_BIND || mask!=desired)
        throw std::runtime_error("vecops comm shared-object policy mismatch");
      for (size_t i=0;i<bytes;i+=page) {
        void* address = first+i; int status=-1;
        system_check(syscall(SYS_move_pages,0,1,&address,nullptr,&status,0)==0,
                     "vecops comm query placement");
        if (status!=node) throw std::runtime_error("vecops comm shared-page placement mismatch");
      }
    }
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

struct Communicator::State {
  int rank, size;
  Options options;
  Exchange exchange;
  const VecopsThreadPoolV1* executor;
  std::vector<Mapping> peers;
  std::mutex mutex;
  uint64_t epoch=0;
  bool sealed=false;
  size_t descriptor_count=0;
  Statistics stats;
  const detail::CopyBackend* copy=&detail::copy_backend();
  std::chrono::steady_clock::time_point deadline;
  ~State() { if (executor && executor->release) executor->release(executor->context); }
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
  void reserve(size_t required) {
    auto& own=peers[rank];
    size_t cap=load(&own.header->capacity);
    if (required<=cap) return;
    if (sealed) throw std::runtime_error("vecops comm sealed capacity exceeded; prepare first");
    size_t next=std::max<size_t>(cap,static_cast<size_t>(sysconf(_SC_PAGESIZE)));
    while(next<required) next=mul(next,2);
    size_t bytes=mul(next,size+1);
    system_check(ftruncate(own.fd,add(header_bytes,bytes))==0,"vecops comm resize");
    own.map_data(bytes);
    Placement placement(options.memory_node,options.require_placement);
    // Touch once outside hot replay; placement follows the owner's policy.
    placement.touch(own.data,next);
    for(int source=0;source<size;++source) {
      bool needed=false;
      for(size_t level=0;level<options.replica_levels.size();++level) {
        const auto& groups=options.replica_levels[level];
        if(!options.relay || groups[source]==groups[rank])continue;
        int members=0,index=0;
        for(int r=0;r<size;++r)if(groups[r]==groups[rank]){if(r<rank)++index;++members;}
        needed|=source%members==index;
      }
      if(needed)placement.touch(own.data+(source+1)*next,next);
    }
    store(&own.header->capacity,next);
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
    if(relay) {
      std::vector<uint64_t> bytes;bytes.reserve(outputs.size());
      for(const auto& output:outputs)bytes.push_back(output.bytes);
      for(const auto& partition:owner->state_->options.replica_levels)
        relay_targets.push_back(detail::balanced_relay_targets(partition,owner->rank(),bytes));
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
     !options.timeout_ms || options.memory_node < -1 || (options.require_placement && options.memory_node < 0))
    throw std::invalid_argument("invalid vecops communicator dimensions/options");
  if(options.locality_groups.empty()) options.locality_groups.assign(size,0);
  if(options.locality_groups.size()!=size) throw std::invalid_argument("invalid locality group map");
  if(options.replica_levels.empty()) {
    options.replica_levels.push_back(options.locality_groups);
    options.copy_levels.push_back(options.relay?1:0);
  }
  if(options.replica_levels.size()>16 || options.copy_levels.size()!=options.replica_levels.size())
    throw std::invalid_argument("invalid replica level count");
  for(size_t level=0;level<options.replica_levels.size();++level) {
    if(options.replica_levels[level].size()!=size || (options.copy_levels[level]!=0 && options.copy_levels[level]!=1))
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
  uint64_t random=0;
  system_check(getrandom(&random,sizeof(random),0)==sizeof(random),"vecops comm random identifier");
  std::string name="/vecops-"+std::to_string(getuid())+"-"+std::to_string(random);
  int fd=shm_open(name.c_str(),O_CREAT|O_EXCL|O_RDWR|O_CLOEXEC,0600);
  system_check(fd>=0,"vecops comm shm_open");
  struct FileGuard {int fd;~FileGuard(){if(fd>=0)close(fd);}} file_guard{fd};
  struct Cleanup {std::string name;~Cleanup(){shm_unlink(name.c_str());}} cleanup{name};
  system_check(ftruncate(fd,header_bytes)==0,"vecops comm control size");
  auto names=exchange(name);
  if(names.size()!=size) throw std::invalid_argument("invalid bootstrap exchange");
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
  for(auto& peer:s->peers) peer.map_data(mul(load(&peer.header->capacity),size+1));
  return std::shared_ptr<Communicator>(new Communicator(std::move(s)));
}
std::shared_ptr<Plan> Communicator::prepare(Specification spec) {
  const auto op=spec.operation;
  if(static_cast<uint32_t>(op)>static_cast<uint32_t>(Operation::SumFloat32))
    throw std::invalid_argument("unknown collective operation");
  if(spec.root<0 || spec.root>=state_->size) throw std::invalid_argument("invalid collective root");
  if((op==Operation::AllGather && (spec.inputs.size()!=1 || spec.outputs.size()!=state_->size)) ||
     (op==Operation::AllToAll && (spec.inputs.size()!=state_->size || spec.outputs.size()!=state_->size)) ||
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
    if(state_->sealed)throw std::runtime_error("cannot prepare a new collective after communicator seal");
    if(state_->descriptor_count==max_plans)throw std::runtime_error("vecops communication plan capacity exceeded");
    auto& p=*plan->state_;
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
    ++state_->stats.plans;
  }
  return plan;
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
        c.peers[peer].map_data(mul(load(&h->capacity),size+1));
      }
  };
  try {
    control([&] {
      lock.lock();
      if(load(&own->aborted)) throw std::runtime_error("vecops communicator is aborted");
      c.deadline=std::chrono::steady_clock::now()+std::chrono::milliseconds(c.options.timeout_ms);
      p.team_epoch=++c.epoch;
      for(auto& peer:c.peers)c.wait(&peer.header->done,p.team_epoch-1);
      c.reserve(p.required);
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
struct LeaseControl {uint64_t aborted=0;SlotControl slot[max_slots];};
struct Alias {
  char* address=nullptr;size_t bytes=0;
  Alias()=default;
  Alias(const Alias&)=delete;
  ~Alias(){if(address)munmap(address,bytes);}
  void reserve(size_t n) {
    auto* p=mmap(nullptr,n,PROT_NONE,MAP_PRIVATE|MAP_ANONYMOUS,-1,0);
    system_check(p!=MAP_FAILED,"vecops reserve alias range");address=static_cast<char*>(p);bytes=n;
  }
  void map(size_t offset,size_t bytes,int fd,size_t file_offset,int protection) {
    auto* p=mmap(address+offset,bytes,protection,MAP_SHARED|MAP_FIXED,fd,file_offset);
    system_check(p!=MAP_FAILED,"vecops fixed page alias");
  }
};
}
struct ExchangeBuffer::State {
  std::shared_ptr<Communicator> owner;
  std::vector<Mapping> peers;
  Alias producer,consumer;
  size_t bytes=0,stride=0,slab=0;
  uint32_t count=0;
  std::vector<uint64_t> versions;
  std::vector<uint8_t> phase;
  std::array<std::mutex,max_slots> mutex;
  LeaseControl* control(int rank)const{return reinterpret_cast<LeaseControl*>(peers[rank].header);}
  void check(uint32_t slot)const {
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
ExchangeBuffer::~ExchangeBuffer(){if(std::any_of(state_->phase.begin(),state_->phase.end(),[](auto p){return p!=0;}))abort();}
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
  if(c.sealed)throw std::runtime_error("cannot allocate exchange storage after seal");
  if(!slots || slots>max_slots || (placement!=PagePlacement::Source && placement!=PagePlacement::Destination))
    throw std::invalid_argument("invalid exchange storage options");
  auto s=std::make_unique<ExchangeBuffer::State>();s->owner=shared_from_this();s->bytes=bytes;s->count=slots;
  const size_t page=sysconf(_SC_PAGESIZE);
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
  system_check(ftruncate(own.fd,add(header_bytes,total))==0,"vecops exchange size");
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
  auto& local=s->peers[c.rank];local.map_data(total);
  {Placement policy(c.options.memory_node,c.options.require_placement);policy.touch(local.data,total);}
  c.exchange("placed");
  s->producer.reserve(total);s->consumer.reserve(total);
  for(uint32_t slot=0;slot<slots;++slot)for(int peer=0;peer<c.size;++peer) {
    size_t offset=slot*s->slab+peer*s->stride;
    int producer_rank=placement==PagePlacement::Source?c.rank:peer;
    int consumer_rank=placement==PagePlacement::Source?peer:c.rank;
    size_t producer_peer=placement==PagePlacement::Source?peer:c.rank;
    size_t consumer_peer=placement==PagePlacement::Source?c.rank:peer;
    s->producer.map(offset,s->stride,s->peers[producer_rank].fd,header_bytes+slot*s->slab+producer_peer*s->stride,PROT_READ|PROT_WRITE);
    s->consumer.map(offset,s->stride,s->peers[consumer_rank].fd,header_bytes+slot*s->slab+consumer_peer*s->stride,PROT_READ);
  }
  // Fault both alias page tables cold. Physical pages were already placed by
  // their owners; these reads do not move pages or make a second payload copy.
  volatile unsigned char touched=0;
  for(size_t i=0;i<total;i+=page) {
    touched=static_cast<unsigned char>(touched^s->producer.address[i]);
    touched=static_cast<unsigned char>(touched^s->consumer.address[i]);
  }
  (void)touched;c.exchange("prepared");
  return std::shared_ptr<ExchangeBuffer>(new ExchangeBuffer(std::move(s)));
}
} // namespace vecops::comm
