// SPDX-FileCopyrightText: 2026 HenryRenYz and vecops contributors
// SPDX-License-Identifier: MIT
// Optional c10d adapter. Shared-memory protocol and copy execution live in the
// framework-neutral core. Prepared graph nodes bypass c10d Work construction.
#include <ATen/ATen.h>
#include <torch/library.h>
#include <torch/csrc/distributed/c10d/Backend.hpp>
#include <torch/csrc/distributed/c10d/Store.hpp>
#include <torch/csrc/distributed/c10d/Work.hpp>
#include <torch/csrc/utils/pybind.h>
#include <pybind11/stl.h>
#include "vecops/comm/Communication.h"
#include <atomic>
#include <deque>
#include <mutex>
#include <sstream>
#include <unordered_map>

namespace vc=vecops::comm;
namespace py=pybind11;
namespace {
vc::Layout layout(const at::Tensor& t) {
  TORCH_CHECK(t.device().is_cpu() || t.device().is_meta(),"vecops communication requires CPU tensors");
  TORCH_CHECK(t.layout()==at::kStrided,"vecops communication requires strided tensors");
  TORCH_CHECK(!t.is_conj() && !t.is_neg(),"unresolved conjugate/negative communication view");
  for(size_t d=0;d<t.dim();++d)TORCH_CHECK(t.stride(d)>=0,"negative communication stride");
  return {t.sizes().vec(),t.strides().vec(),static_cast<size_t>(t.element_size())};
}
struct Ref {size_t argument,offset;};
struct Queue;
void drain_queue(const std::weak_ptr<Queue>&);
void abort_queue(const std::weak_ptr<Queue>&) noexcept;
struct Prepared {
  std::shared_ptr<vc::Plan> plan;
  std::vector<Ref> in_refs,out_refs;
  std::vector<const void*> in_pointers;
  std::vector<void*> out_pointers;
  std::vector<vc::Layout> input_metadata,output_metadata;
  std::vector<at::ScalarType> input_types,output_types;
  std::mutex mutex;
  std::weak_ptr<Queue> queue;
  int64_t id=0;
  void execute(at::TensorList outputs,at::TensorList inputs) {
    std::unique_lock lock(mutex,std::try_to_lock);
    TORCH_CHECK(lock.owns_lock(),"concurrent use of one prepared communication instance");
    TORCH_CHECK(inputs.size()==input_metadata.size() && outputs.size()==output_metadata.size(),"prepared communication arity changed");
    auto check=[](const at::Tensor& t,const vc::Layout& l,at::ScalarType type) {
      TORCH_CHECK(t.device().is_cpu() && t.scalar_type()==type &&
                  t.sizes().equals(l.sizes) && t.strides().equals(l.strides),
                  "prepared communication metadata changed; prepare a new plan");
    };
    for(size_t i=0;i<inputs.size();++i)check(inputs[i],input_metadata[i],input_types[i]);
    for(size_t i=0;i<outputs.size();++i)check(outputs[i],output_metadata[i],output_types[i]);
    for(size_t i=0;i<in_refs.size();++i) {
      const auto& r=in_refs[i];
      auto* base=static_cast<const char*>(inputs[r.argument].const_data_ptr());
      in_pointers[i]=base?base+r.offset:nullptr;
    }
    for(size_t i=0;i<out_refs.size();++i) {
      const auto& r=out_refs[i];
      auto* base=static_cast<char*>(outputs[r.argument].mutable_data_ptr());
      out_pointers[i]=base?base+r.offset:nullptr;
    }
    plan->execute(out_pointers.data(),in_pointers.data());
  }
};
std::mutex registry_mutex;
std::unordered_map<int64_t,std::weak_ptr<Prepared>> registry;
std::atomic<int64_t> next_id{1};
void run(int64_t id,at::TensorList outputs,at::TensorList inputs) {
  std::shared_ptr<Prepared> p;
  {std::lock_guard lock(registry_mutex);auto it=registry.find(id);if(it!=registry.end())p=it->second.lock();}
  TORCH_CHECK(p,"expired prepared communication owner");
  drain_queue(p->queue);
  try{p->execute(outputs,inputs);}catch(...){abort_queue(p->queue);throw;}
}
void meta_run(int64_t,at::TensorList,at::TensorList) {}
// The pointer table never crosses the Python boundary. A synchronous native
// consumer owns its output and joins every reader before the transport retires.
struct PreparedConsumer {
  std::shared_ptr<Prepared> publication;
  at::Tensor addresses;
  vc::Layout source_layout, destination_layout;
  at::ScalarType source_type, destination_type;
  c10::OperatorHandle operation;
  std::vector<int64_t> specs;
  c10::Stack stack;
  std::mutex mutex;
  int64_t id=0;
  PreparedConsumer(std::shared_ptr<Prepared> p,at::Tensor source,at::Tensor output,
                   const std::string& name,std::vector<int64_t> values,int size)
    :publication(std::move(p)),addresses(at::empty({size},at::kLong)),
     source_layout(layout(source)),destination_layout(layout(output)),
     source_type(source.scalar_type()),destination_type(output.scalar_type()),
     operation(c10::Dispatcher::singleton().findSchemaOrThrow(name.c_str(),"")),specs(std::move(values)) {
    auto& schema=operation.schema();
    TORCH_CHECK(schema.arguments().size()==2+specs.size() && schema.returns().empty(),
                "borrowed consumer requires (Tensor addresses, Tensor(a!) output, int...)->()");
    for(size_t i=0;i<schema.arguments().size();++i) {
      auto type=schema.arguments()[i].type();
      // DSL specialization parameters have int? schemas so callers can leave
      // them inferred. This prebound consumer supplies every concrete value.
      if(i>=2 && type->kind()==c10::TypeKind::OptionalType)
        type=type->cast<c10::OptionalType>()->getElementType();
      TORCH_CHECK(type->kind()==(i<2?c10::TypeKind::TensorType:c10::TypeKind::IntType),
                  "unsupported borrowed consumer argument type: ",schema.arguments()[i].type()->str());
    }
    TORCH_CHECK(schema.arguments()[1].alias_info() && schema.arguments()[1].alias_info()->isWrite(),
                "borrowed consumer output must be declared mutable");
    stack.reserve(2+specs.size());
  }
  void execute(at::Tensor out,at::Tensor in) {
    std::unique_lock lock(mutex,std::try_to_lock);
    TORCH_CHECK(lock.owns_lock(),"concurrent borrowed consumer use");
    auto check=[](const at::Tensor& t,const vc::Layout& l,at::ScalarType type) {
      TORCH_CHECK(t.device().is_cpu() && t.scalar_type()==type && t.sizes().equals(l.sizes) && t.strides().equals(l.strides),
                  "borrowed consumer metadata changed");
    };
    check(in,source_layout,source_type);check(out,destination_layout,destination_type);
    drain_queue(publication->queue);
    const void* input=in.const_data_ptr();
    publication->plan->consume(&input,[&](const void* const* pointers,size_t count) {
      static_assert(sizeof(void*)==sizeof(int64_t));
      auto* target=addresses.mutable_data_ptr<int64_t>();
      for(size_t peer=0;peer<count;++peer)
        target[peer]=static_cast<int64_t>(reinterpret_cast<uintptr_t>(pointers[peer]));
      stack.clear();stack.emplace_back(addresses);stack.emplace_back(out);
      for(auto v:specs)stack.emplace_back(v);
      operation.callBoxed(&stack);
    });
  }
};
std::unordered_map<int64_t,std::weak_ptr<PreparedConsumer>> consumers;
void consume(int64_t id,at::Tensor output,at::Tensor input) {
  std::shared_ptr<PreparedConsumer> p;
  {std::lock_guard lock(registry_mutex);auto it=consumers.find(id);if(it!=consumers.end())p=it->second.lock();}
  TORCH_CHECK(p,"expired borrowed consumer owner");
  try{p->execute(output,input);}catch(...){abort_queue(p->publication->queue);throw;}
}
void meta_consume(int64_t,at::Tensor,at::Tensor) {}
struct PreparedExchange {
  std::shared_ptr<vc::ExchangeBuffer> buffer;
  std::weak_ptr<Queue> queue;
  std::vector<at::Tensor> producers,consumers;
  int64_t id=0;
};
std::unordered_map<int64_t,std::weak_ptr<PreparedExchange>> exchanges;
void exchange_phase(int64_t id,int64_t phase,int64_t slot,at::TensorList storage) {
  std::shared_ptr<PreparedExchange> p;
  {std::lock_guard lock(registry_mutex);auto it=exchanges.find(id);if(it!=exchanges.end())p=it->second.lock();}
  TORCH_CHECK(p,"expired page exchange owner");
  try {
  TORCH_CHECK(slot>=0 && slot<p->producers.size(),"invalid exchange slot");
  // Explicit mutable arguments carry graph ordering and alias/liveness effects.
  TORCH_CHECK(storage.size()==2 && storage[0].is_alias_of(p->producers[slot]) && storage[1].is_alias_of(p->consumers[slot]),
              "exchange storage bindings changed");
  drain_queue(p->queue);
  if(phase==0)p->buffer->begin(slot);
  else if(phase==1)p->buffer->publish(slot);
  else if(phase==2)p->buffer->retire(slot);
  else TORCH_CHECK(false,"invalid exchange phase");
  }catch(...){p->buffer->abort();throw;}
}
void abort_exchange(int64_t id) {
  std::shared_ptr<PreparedExchange> p;
  {std::lock_guard lock(registry_mutex);auto it=exchanges.find(id);if(it!=exchanges.end())p=it->second.lock();}
  TORCH_CHECK(p,"expired page exchange owner");p->buffer->abort();
}
void meta_exchange(int64_t,int64_t,int64_t,at::TensorList){}
class Pending;
struct Queue {
  std::mutex mutex;
  std::deque<c10::intrusive_ptr<Pending>> works;
  std::shared_ptr<vc::Communicator> communicator;
  std::atomic<size_t> pending{0};
  void wait(Pending* target);
  void drain();
};
class Pending final:public c10d::Work {
public:
  std::weak_ptr<Queue> queue;
  std::shared_ptr<Prepared> prepared;
  std::vector<at::Tensor> outputs,inputs;
  Pending(int rank,std::shared_ptr<Queue> q,std::shared_ptr<Prepared> p,
      at::TensorList out,at::TensorList in):Work(rank,c10d::OpType::UNKNOWN),
      queue(q),prepared(std::move(p)),outputs(out.vec()),inputs(in.vec()) {}
  bool wait(std::chrono::milliseconds timeout=kNoTimeout) override {
    TORCH_CHECK(timeout==kNoTimeout,"per-Work timeout overrides are unsupported; configure communicator timeout");
    if(!isCompleted()) {
      auto q=queue.lock();TORCH_CHECK(q,"communication backend destroyed before Work completion");q->wait(this);
    }
    if(auto error=exception())std::rethrow_exception(error);
    return true;
  }
  void execute() {
    try{prepared->execute(outputs,inputs);finish();}catch(...){finish(std::current_exception());}
  }
  std::vector<at::Tensor> result()override{return outputs;}
  void abort() override {
    if(auto q=queue.lock())q->communicator->abort();
  }
};
void Queue::wait(Pending* target) {
  std::lock_guard lock(mutex);
  while(!target->isCompleted()) {
    TORCH_CHECK(!works.empty(),"missing communication queue entry");
    auto current=std::move(works.front());works.pop_front();current->execute();
    pending.fetch_sub(1,std::memory_order_release);
    if(auto error=current->exception())std::rethrow_exception(error);
  }
}
void Queue::drain() {
  if(!pending.load(std::memory_order_acquire))return;
  std::lock_guard lock(mutex);
  while(!works.empty()) {
    auto current=std::move(works.front());works.pop_front();current->execute();
    pending.fetch_sub(1,std::memory_order_release);
    if(auto error=current->exception())std::rethrow_exception(error);
  }
}
void drain_queue(const std::weak_ptr<Queue>& weak) {
  auto queue=weak.lock();TORCH_CHECK(queue,"prepared communication queue owner expired");
  queue->drain();
}
void abort_queue(const std::weak_ptr<Queue>& weak) noexcept {
  if(auto q=weak.lock())q->communicator->abort();
}
class Backend final:public c10d::Backend {
  std::shared_ptr<vc::Communicator> communicator_;
  std::shared_ptr<Queue> queue_=std::make_shared<Queue>();
  std::mutex plans_mutex_;
  bool sealed_=false;
  std::vector<int> route_override_;
  std::unordered_map<std::string,std::vector<int>> routes_;
  std::vector<std::shared_ptr<PreparedConsumer>> consumers_;
  std::unordered_map<std::string,std::shared_ptr<PreparedConsumer>> consumer_cache_;
  std::vector<std::shared_ptr<PreparedExchange>> exchanges_;
  std::unordered_map<std::string,std::shared_ptr<Prepared>> plans_;
public:
  std::string tuning_context;
  Backend(c10::intrusive_ptr<c10d::Store> store,int rank,int size,uint64_t timeout,
      uint32_t threads,uint64_t executor,int node,std::vector<int> groups,
      size_t initial,size_t threshold,size_t chunk,bool relay,bool strict,
      std::vector<std::vector<int>> levels,std::vector<int> modes,
      bool row_owned=true):c10d::Backend(rank,size) {
    auto phase=std::make_shared<size_t>(0);
    vc::Exchange exchange=[store,rank,size,phase](const std::string& own) {
      std::string prefix="vecops-comm/bootstrap/"+std::to_string((*phase)++)+"/";
      store->set(prefix+std::to_string(rank),std::vector<uint8_t>(own.begin(),own.end()));
      std::vector<std::string> result;result.reserve(size);
      for(int p=0;p<size;++p){auto data=store->get(prefix+std::to_string(p));result.emplace_back(data.begin(),data.end());}
      return result;
    };
    vc::Options options;options.threads=threads;options.memory_node=node;
    options.locality_groups=std::move(groups);options.timeout_ms=timeout;
    options.initial_capacity=initial;options.relay_threshold=threshold;
    options.chunk_bytes=chunk;options.relay=relay;options.require_placement=strict;
    options.replica_levels=std::move(levels);options.copy_levels=std::move(modes);
    options.row_owned_copy=row_owned;
    communicator_=vc::Communicator::create(rank,size,options,std::move(exchange),
        reinterpret_cast<const VecopsThreadPoolV1*>(executor));
    queue_->communicator=communicator_;
  }
  ~Backend()override {
    communicator_->abort();
    std::lock_guard lock(registry_mutex);
    for(auto& [key,p]:plans_)registry.erase(p->id);
    for(auto& p:consumers_)consumers.erase(p->id);
    for(auto& p:exchanges_)exchanges.erase(p->id);
  }
  const std::string getBackendName()const override{return "vecops";}
  std::shared_ptr<Prepared> prepare(const std::string& kind,at::TensorList outputs,
      at::TensorList inputs,std::vector<int64_t> out_splits={},std::vector<int64_t> in_splits={},int root=0,std::vector<int> requested_modes={}) {
    std::ostringstream stream;stream<<kind<<':'<<root;
    for(auto list:{outputs,inputs})for(const auto& t:list) {
      stream<<'/'<<t.scalar_type()<<':'<<t.sizes()<<':'<<t.strides();
    }
    for(auto n:out_splits)stream<<'o'<<n;
    for(auto n:in_splits)stream<<'i'<<n;
    std::string base_key=stream.str(),key=base_key;
    std::lock_guard lock(plans_mutex_);
    auto modes=requested_modes.empty()?route_override_:requested_modes;
    if(modes.empty())if(auto it=routes_.find(base_key);it!=routes_.end())modes=it->second;
    for(int mode:modes)key+="@"+std::to_string(mode);
    if(auto found=plans_.find(key);found!=plans_.end())return found->second;
    auto p=std::make_shared<Prepared>();p->queue=queue_;vc::Specification spec;spec.root=root;spec.copy_levels=modes;
    if(!inputs.empty())spec.element_type=static_cast<uint64_t>(inputs[0].scalar_type())+1;
    for(const auto& t:inputs){p->input_metadata.push_back(layout(t));p->input_types.push_back(t.scalar_type());}
    for(const auto& t:outputs){p->output_metadata.push_back(layout(t));p->output_types.push_back(t.scalar_type());}
    auto add_input=[&](size_t arg,vc::Layout l,size_t offset=0){spec.inputs.push_back(std::move(l));p->in_refs.push_back({arg,offset});};
    auto add_output=[&](size_t arg,vc::Layout l,size_t offset=0){spec.outputs.push_back(std::move(l));p->out_refs.push_back({arg,offset});};
    auto require_type=[&] {
      for(const auto& t:outputs)TORCH_CHECK(inputs.empty() || t.scalar_type()==inputs[0].scalar_type(),"collective dtype mismatch");
      for(const auto& t:inputs)TORCH_CHECK(t.scalar_type()==inputs[0].scalar_type(),"collective dtype mismatch");
    };
    require_type();
    if(kind=="barrier")spec.operation=vc::Operation::Barrier;
    else if(kind=="all_gather" || kind=="all_gather_into_tensor" || kind=="all_gatherv") {
      TORCH_CHECK(inputs.size()==1,"AllGather requires one input");spec.operation=vc::Operation::AllGather;
      add_input(0,layout(inputs[0]));
      if(kind=="all_gather")for(size_t i=0;i<outputs.size();++i)add_output(i,layout(outputs[i]));
      else {
        TORCH_CHECK(outputs.size()==1 && outputs[0].is_contiguous(),"joined AllGather output must be contiguous");
        if(out_splits.empty())out_splits.assign(getSize(),inputs[0].numel());
        TORCH_CHECK(out_splits.size()==getSize(),"invalid gather counts");
        size_t offset=0;for(auto n:out_splits){TORCH_CHECK(n>=0,"negative gather count");add_output(0,{{n},{1},static_cast<size_t>(inputs[0].element_size())},offset);offset+=n*inputs[0].element_size();}
        TORCH_CHECK(offset==outputs[0].nbytes(),"gather output capacity/count mismatch");
      }
    }else if(kind=="all_to_all" || kind=="all_to_all_single") {
      spec.operation=vc::Operation::AllToAll;
      if(kind=="all_to_all") {
        for(size_t i=0;i<inputs.size();++i)add_input(i,layout(inputs[i]));
        for(size_t i=0;i<outputs.size();++i)add_output(i,layout(outputs[i]));
      }else {
        TORCH_CHECK(inputs.size()==1 && outputs.size()==1,"AllToAllSingle requires one input/output");
        auto split=[&](const at::Tensor& t,std::vector<int64_t> counts,bool input) {
          TORCH_CHECK(t.dim()>0 && t.is_contiguous(),"AllToAllSingle requires contiguous nonscalar tensors");
          if(counts.empty()){TORCH_CHECK(t.size(0)%getSize()==0,"uneven split without explicit counts");counts.assign(getSize(),t.size(0)/getSize());}
          TORCH_CHECK(counts.size()==getSize(),"invalid AllToAll counts");
          int64_t total=0;size_t offset=0;
          size_t row=t.element_size();for(int d=1;d<t.dim();++d)row*=t.size(d);
          for(auto n:counts){TORCH_CHECK(n>=0,"negative AllToAll count");auto l=layout(t);l.sizes[0]=n;
            if(input)add_input(0,std::move(l),offset);else add_output(0,std::move(l),offset);
            offset+=n*row;total+=n;}
          TORCH_CHECK(total==t.size(0),"AllToAll counts do not match dim0");
        };split(inputs[0],in_splits,true);split(outputs[0],out_splits,false);
      }
    }else if(kind=="broadcast" || kind=="sum") {
      TORCH_CHECK(outputs.size()==1 && inputs.size()==1,"broadcast/reduce requires one tensor");
      spec.operation=kind=="sum"?vc::Operation::SumFloat32:vc::Operation::Broadcast;
      if(kind=="sum")TORCH_CHECK(inputs[0].scalar_type()==at::kFloat,"SUM supports float32 only");
      add_input(0,layout(inputs[0]));add_output(0,layout(outputs[0]));
    }else TORCH_CHECK(false,"unsupported vecops collective: ",kind);
    p->plan=communicator_->prepare(std::move(spec));
    p->in_pointers.resize(p->in_refs.size());p->out_pointers.resize(p->out_refs.size());
    p->id=next_id.fetch_add(1);
    {std::lock_guard guard(registry_mutex);registry[p->id]=p;}
    plans_.emplace(std::move(key),p);return p;
  }
  std::string plan_key(const std::string& kind,at::TensorList out,at::TensorList in) {
    std::ostringstream stream;stream<<kind<<':'<<0;
    for(auto list:{out,in})for(const auto& t:list)stream<<'/'<<t.scalar_type()<<':'<<t.sizes()<<':'<<t.strides();
    return stream.str();
  }
  void set_route(std::vector<int> modes) {
    std::lock_guard lock(plans_mutex_);TORCH_CHECK(!sealed_,"cannot tune routing after seal");route_override_=std::move(modes);
  }
  void configure_routes(std::unordered_map<std::string,std::vector<int>> routes) {
    std::lock_guard lock(plans_mutex_);TORCH_CHECK(!sealed_,"cannot configure routing after seal");routes_=std::move(routes);route_override_.clear();
  }
  py::tuple prepare_exchange(std::vector<int64_t> shape,at::ScalarType type,uint32_t slots,const std::string& placement) {
    TORCH_CHECK(placement=="source" || placement=="destination","page placement must be source or destination");
    size_t count=1;for(auto n:shape){TORCH_CHECK(n>=0 && (!n || count<=SIZE_MAX/size_t(n)),"invalid exchange shape");count*=n;}
    size_t item=c10::elementSize(type);TORCH_CHECK(count<=SIZE_MAX/item,"exchange byte count overflow");
    auto p=std::make_shared<PreparedExchange>();p->queue=queue_;
    p->buffer=communicator_->prepare_exchange(count*item,slots,placement=="source"?vc::PagePlacement::Source:vc::PagePlacement::Destination);
    TORCH_CHECK(p->buffer->block_stride()%item==0,"dtype does not divide page stride");
    auto sizes=shape;sizes.insert(sizes.begin(),getSize());
    std::vector<int64_t> strides(sizes.size());int64_t step=1;
    for(size_t d=shape.size();d>0;--d){strides[d]=step;step*=shape[d-1];}
    strides[0]=p->buffer->block_stride()/item;
    for(uint32_t slot=0;slot<slots;++slot) {
      auto owner=p->buffer;
      p->producers.push_back(at::from_blob(owner->producer(slot),sizes,strides,[owner](void*){},at::TensorOptions().dtype(type)));
      p->consumers.push_back(at::from_blob(const_cast<void*>(owner->consumer(slot)),sizes,strides,[owner](void*){},at::TensorOptions().dtype(type)));
    }
    p->id=next_id.fetch_add(1);
    {std::lock_guard lock(registry_mutex);exchanges[p->id]=p;}
    exchanges_.push_back(p);return py::make_tuple(p->id,p->producers,p->consumers);
  }
  int64_t prepare_consumer(at::Tensor source,at::Tensor output,const std::string& name,std::vector<int64_t> specs) {
    // Layout-only placeholders: no P-way full-result allocation.
    std::vector<at::Tensor> outputs(getSize(),source);
    auto publication=prepare("all_gather",outputs,{source});
    std::ostringstream key_stream;key_stream<<publication->id<<'/'<<name<<'/'<<output.scalar_type()<<':'<<output.sizes()<<':'<<output.strides();
    for(auto value:specs)key_stream<<'/'<<value;
    auto key=key_stream.str();std::lock_guard lock(plans_mutex_);
    if(auto it=consumer_cache_.find(key);it!=consumer_cache_.end())return it->second->id;
    TORCH_CHECK(!sealed_,"cannot prepare a new borrowed consumer after seal");
    auto p=std::make_shared<PreparedConsumer>(publication,source,output,name,std::move(specs),getSize());
    p->id=next_id.fetch_add(1);
    {std::lock_guard lock(registry_mutex);consumers[p->id]=p;}
    consumers_.push_back(p);consumer_cache_[key]=p;return p->id;
  }
  c10::intrusive_ptr<c10d::Work> submit(const std::string& kind,at::TensorList out,at::TensorList in,
      std::vector<int64_t> os={},std::vector<int64_t> is={},int root=0) {
    std::shared_ptr<Prepared> p;
    try {p=prepare(kind,out,in,std::move(os),std::move(is),root);}
    catch(...){communicator_->abort();throw;}
    auto work=c10::make_intrusive<Pending>(getRank(),queue_,std::move(p),out,in);
    {std::lock_guard lock(queue_->mutex);queue_->works.push_back(work);queue_->pending.fetch_add(1,std::memory_order_release);}return work;
  }
  c10::intrusive_ptr<c10d::Work> allgather(std::vector<std::vector<at::Tensor>>& out,std::vector<at::Tensor>& in,const c10d::AllgatherOptions&)override {
    TORCH_CHECK(out.size()==1,"vecops supports one CPU input per AllGather");return submit("all_gather",out[0],in);
  }
  c10::intrusive_ptr<c10d::Work> _allgather_base(at::Tensor& out,at::Tensor& in,const c10d::AllgatherOptions&)override{return submit("all_gather_into_tensor",{out},{in});}
  c10::intrusive_ptr<c10d::Work> alltoall(std::vector<at::Tensor>& out,std::vector<at::Tensor>& in,const c10d::AllToAllOptions&)override{return submit("all_to_all",out,in);}
  c10::intrusive_ptr<c10d::Work> alltoall_base(at::Tensor& out,at::Tensor& in,std::vector<int64_t>& os,std::vector<int64_t>& is,const c10d::AllToAllOptions&)override{return submit("all_to_all_single",{out},{in},os,is);}
  c10::intrusive_ptr<c10d::Work> broadcast(std::vector<at::Tensor>& t,const c10d::BroadcastOptions& o)override{return submit("broadcast",t,t,{}, {},o.rootRank);}
  c10::intrusive_ptr<c10d::Work> allreduce(std::vector<at::Tensor>& t,const c10d::AllreduceOptions& o)override {
    TORCH_CHECK(o.reduceOp==c10d::ReduceOp::SUM,"vecops supports SUM only");return submit("sum",t,t);
  }
  c10::intrusive_ptr<c10d::Work> barrier(const c10d::BarrierOptions&)override{return submit("barrier",{},{});}
  py::dict statistics()const {
    auto s=communicator_->statistics();py::dict d;d["calls"]=s.calls;d["plans"]=s.plans;
    d["capacity_growths"]=s.capacity_growths;d["copy_bytes"]=s.copy_bytes;
    d["row_copy_calls"]=s.row_copy_calls;d["row_copy_bytes"]=s.row_copy_bytes;
    return d;
  }
  void seal(){std::lock_guard lock(plans_mutex_);communicator_->seal();sealed_=true;}
  void abort(){communicator_->abort();for(auto& p:exchanges_)p->buffer->abort();}
};
} // namespace
TORCH_LIBRARY(vecops_comm,m) {
  m.def("run(int plan, Tensor(a!)[] outputs, Tensor[] inputs) -> ()");
  m.def("consume(int plan, Tensor(a!) output, Tensor input) -> ()");
  m.def("exchange(int plan, int phase, int slot, Tensor(a!)[] storage) -> ()");
}
TORCH_LIBRARY_IMPL(vecops_comm,CPU,m){m.impl("run",run);m.impl("consume",consume);m.impl("exchange",exchange_phase);}
TORCH_LIBRARY_IMPL(vecops_comm,Meta,m){m.impl("run",meta_run);m.impl("consume",meta_consume);m.impl("exchange",meta_exchange);}
// Barrier has no tensor dispatch key.
TORCH_LIBRARY_IMPL(vecops_comm,CatchAll,m){m.impl("run",run);m.impl("consume",consume);m.impl("exchange",exchange_phase);}
PYBIND11_MODULE(TORCH_EXTENSION_NAME,m) {
  m.def("abort_exchange",abort_exchange);
  py::class_<Backend,c10d::Backend,c10::intrusive_ptr<Backend>>(m,"Backend")
    .def(py::init<c10::intrusive_ptr<c10d::Store>,int,int,uint64_t,uint32_t,uint64_t,int,std::vector<int>,size_t,size_t,size_t,bool,bool,std::vector<std::vector<int>>,std::vector<int>>(),py::call_guard<py::gil_scoped_release>())
    .def(py::init<c10::intrusive_ptr<c10d::Store>,int,int,uint64_t,uint32_t,uint64_t,int,std::vector<int>,size_t,size_t,size_t,bool,bool,std::vector<std::vector<int>>,std::vector<int>,bool>(),py::call_guard<py::gil_scoped_release>())
    .def("prepare",[](Backend& b,const std::string& kind,const std::vector<at::Tensor>& out,const std::vector<at::Tensor>& in,std::vector<int64_t> os,std::vector<int64_t> is,int root){return b.prepare(kind,out,in,std::move(os),std::move(is),root)->id;})
    .def("gatherv",[](Backend& b,at::Tensor out,at::Tensor in,std::vector<int64_t> counts){return b.submit("all_gatherv",{out},{in},std::move(counts));},py::call_guard<py::gil_scoped_release>())
    .def_readwrite("tuning_context",&Backend::tuning_context)
    .def("prepare_routed",[](Backend& b,const std::string& k,const std::vector<at::Tensor>& o,const std::vector<at::Tensor>& i,std::vector<int> modes){return b.prepare(k,o,i,{}, {},0,std::move(modes))->id;})
    .def("plan_key",[](Backend& b,const std::string& k,const std::vector<at::Tensor>& o,const std::vector<at::Tensor>& i){return b.plan_key(k,o,i);})
    .def("set_route",&Backend::set_route)
    .def("configure_routes",&Backend::configure_routes)
    .def("prepare_exchange",&Backend::prepare_exchange)
    .def("prepare_consumer",&Backend::prepare_consumer)
    .def("statistics",&Backend::statistics)
    .def("abort",&Backend::abort)
    .def("seal",&Backend::seal);
}
