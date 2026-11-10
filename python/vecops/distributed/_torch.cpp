// SPDX-FileCopyrightText: 2026 HenryRenYz and vecops contributors
// SPDX-License-Identifier: MIT
// Optional c10d adapter. Shared-memory protocol and copy execution live in the
// framework-neutral core. Prepared graph nodes bypass c10d Work construction.
#include <ATen/ATen.h>
#include <ATen/record_function.h>
#include <torch/library.h>
#include <torch/csrc/distributed/c10d/Backend.hpp>
#include <torch/csrc/distributed/c10d/Store.hpp>
#include <torch/csrc/distributed/c10d/Work.hpp>
#include <torch/csrc/utils/pybind.h>
#include <pybind11/stl.h>
#include "vecops/comm/Communication.h"
#include "vecops/memory/Model.h"
#include "vecops/memory/RuntimeBridge.h"
#include "vecops/runtime/KernelPlanAbi.h"
#include <atomic>
#include <chrono>
#include <deque>
#include <mutex>
#include <sstream>
#include <unordered_map>
#include <cstring>

namespace vc=vecops::comm;
namespace py=pybind11;
namespace {
std::shared_ptr<vecops::memory::model::MemoryRuntime> import_runtime_bridge(py::handle value) {
  namespace bridge=vecops::memory::bridge;
  if(!PyCapsule_CheckExact(value.ptr()) || !PyCapsule_IsValid(value.ptr(),bridge::capsule_name))
    throw py::type_error("expected a named MemoryRuntime bridge capsule from runtime.export_bridge()");
  auto* descriptor=static_cast<bridge::Descriptor*>(PyCapsule_GetPointer(value.ptr(),bridge::capsule_name));
  return bridge::acquire(*descriptor);
}
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
  vc::Specification specification;
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
std::atomic<uint64_t> next_capacity_provider{1};
struct Queue {
  std::mutex mutex;
  std::deque<c10::intrusive_ptr<Pending>> works;
  std::shared_ptr<vc::Communicator> communicator;
  std::atomic<size_t> pending{0};
  const uint64_t capacity_provider_id=next_capacity_provider.fetch_add(1);
  void wait(Pending* target);
  void drain();
  bool provision_capacity();
  bool preparation_consensus(bool local_success);
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
bool Queue::provision_capacity() {
  // Holding the queue lock through the cold collective also prevents a new
  // Work from being published between drain and the stopped handshake.
  std::lock_guard lock(mutex);
  while(!works.empty()) {
    auto current=std::move(works.front());works.pop_front();current->execute();
    pending.fetch_sub(1,std::memory_order_release);
    if(auto error=current->exception())std::rethrow_exception(error);
  }
  return communicator->provision_registered_capacity();
}
bool Queue::preparation_consensus(bool local_success) {
  std::lock_guard lock(mutex);
  while(!works.empty()) {
    auto current=std::move(works.front());works.pop_front();current->execute();
    pending.fetch_sub(1,std::memory_order_release);
    if(auto error=current->exception())std::rethrow_exception(error);
  }
  return communicator->preparation_consensus(local_success);
}
void drain_queue(const std::weak_ptr<Queue>& weak) {
  auto queue=weak.lock();TORCH_CHECK(queue,"prepared communication queue owner expired");
  queue->drain();
}
void abort_queue(const std::weak_ptr<Queue>& weak) noexcept {
  if(auto q=weak.lock())q->communicator->abort();
}
// Per-graph-node bindings, not per-shape backend cache state. Several identical
// collectives in one resident island are bound before any of them executes.
struct ResidentKernel {
  std::atomic<uint32_t> references{1};
  std::shared_ptr<Prepared> publication;
  std::shared_ptr<PreparedConsumer> borrowed;
  std::shared_ptr<Queue> queue;
  std::vector<void*> data,out;
  std::vector<const void*> in;
  std::vector<vc::Layout> metadata;
  std::vector<uint32_t> types;
  std::vector<int64_t> addresses;
  std::vector<VecopsValue> consumer_values;
  VecopsPreparedKernelV1 consumer{},abi{};
  struct ConsumerOwner {
    VecopsPreparedKernelV1 abi{};
    ~ConsumerOwner(){if(abi.destroy)abi.destroy(abi.context);}
  } consumer_owner;
  VecopsExecutionContext execution{};
  vc::Plan::TeamConsumer team_consumer;
  std::mutex mutex;
  bool consumer_entered=false;
  size_t outputs=0;
  bool entered_once=false;
  std::vector<int> configured_modes;

  void configure_route(std::vector<int> modes) {
    std::unique_lock lock(mutex,std::try_to_lock);
    TORCH_CHECK(lock.owns_lock() && !entered_once,"configure communication placement before its first execution");
    auto replacement=std::make_shared<Prepared>();
    replacement->queue=publication->queue;
    replacement->specification=publication->specification;
    replacement->specification.copy_levels=modes;
    replacement->plan=queue->communicator->prepare(replacement->specification);
    replacement->in_refs=publication->in_refs;replacement->out_refs=publication->out_refs;
    replacement->input_metadata=publication->input_metadata;replacement->output_metadata=publication->output_metadata;
    replacement->input_types=publication->input_types;replacement->output_types=publication->output_types;
    replacement->in_pointers.resize(replacement->in_refs.size());
    replacement->out_pointers.resize(replacement->out_refs.size());
    publication=std::move(replacement);
    configured_modes=std::move(modes);
  }

  template<class Fn> static int guard(VecopsError* error,Fn&& fn) noexcept {
    try{fn();return VECOPS_STATUS_OK;}catch(const std::exception& e){
      if(error){error->code=VECOPS_STATUS_EXECUTION_ERROR;error->message_required=std::strlen(e.what())+1;
        if(error->message && error->message_capacity){auto n=std::min<size_t>(std::strlen(e.what()),error->message_capacity-1);std::memcpy(error->message,e.what(),n);error->message[n]=0;}}
      return VECOPS_STATUS_EXECUTION_ERROR;
    }catch(...){return VECOPS_STATUS_EXECUTION_ERROR;}
  }
  ResidentKernel(std::shared_ptr<Prepared> p,std::shared_ptr<PreparedConsumer> b,
      const VecopsPreparedKernelV1* source,std::vector<uint32_t> wire_types)
      :publication(std::move(p)),borrowed(std::move(b)),types(std::move(wire_types)) {
    queue=publication->queue.lock();TORCH_CHECK(queue,"expired resident communication queue");
    configured_modes=publication->specification.copy_levels;
    const auto* pool=publication->plan->thread_pool();
    TORCH_CHECK(pool && pool->struct_size>=sizeof(VecopsThreadPoolV1) && pool->team_executor &&
        pool->team_executor->abi_major==1 && pool->team_executor->run && pool->max_parallelism,"resident communication requires a physical team");
    if(borrowed){metadata={borrowed->destination_layout,borrowed->source_layout};outputs=1;}
    else{metadata=publication->output_metadata;outputs=metadata.size();metadata.insert(metadata.end(),publication->input_metadata.begin(),publication->input_metadata.end());}
    TORCH_CHECK(types.size()==metadata.size(),"resident wire metadata arity changed");
    data.resize(metadata.size());in.resize(publication->in_refs.size());out.resize(publication->out_refs.size());
    execution.struct_size=sizeof(execution);execution.requested_threads=publication->plan->workers();execution.thread_pool=pool;
    if(borrowed) {
      TORCH_CHECK(source && source->struct_size>=sizeof(VecopsPreparedKernelV1) && source->abi_major==1 &&
          source->thread_pool==pool && source->bind_prevalidated && source->enter && source->run_lane && source->leave && source->retain && source->destroy,
          "borrowed resident consumer must expose the same physical provider");
      consumer=*source;consumer.retain(consumer.context);consumer_owner.abi=consumer;
      addresses.resize(borrowed->addresses.numel());consumer_values.resize(2);
      for(auto& value:consumer_values){value.struct_size=sizeof(VecopsValue);value.kind=VECOPS_VALUE_TENSOR;value.value.tensor.struct_size=sizeof(VecopsTensorView);}
      team_consumer=[this](const void* const* pointers,size_t count,const VecopsTeamContextV1& team) {
        TORCH_CHECK(count==addresses.size(),"borrowed pointer count changed");
        if(team.worker_id==0)for(size_t i=0;i<count;++i)addresses[i]=static_cast<int64_t>(reinterpret_cast<uintptr_t>(pointers[i]));
        char message[512]{};VecopsError error{sizeof(VecopsError),0,message,sizeof(message),0};
        if(team.worker_count>1)TORCH_CHECK(team.barrier(team.context,&error)==VECOPS_STATUS_OK,"borrowed address publication cancelled: ",message);
        if(team.worker_id==0) {
          RECORD_FUNCTION("vecops_comm::resident_consumer_kernel",c10::ArrayRef<c10::IValue>());
          TORCH_CHECK(consumer.run_lane(consumer.context,&team,&error)==VECOPS_STATUS_OK,"resident consumer failed: ",message);
        } else TORCH_CHECK(consumer.run_lane(consumer.context,&team,&error)==VECOPS_STATUS_OK,"resident consumer failed: ",message);
      };
    }
    abi.struct_size=sizeof(abi);abi.abi_major=1;abi.context=this;
    abi.workers=std::max(std::min(publication->plan->workers(),pool->max_parallelism(pool->context)),consumer.workers);abi.grids=1;
    abi.thread_pool=pool;abi.execution_context=&execution;
    abi.retain=[](void* c){static_cast<ResidentKernel*>(c)->references.fetch_add(1,std::memory_order_relaxed);};
    abi.destroy=[](void* c){auto* self=static_cast<ResidentKernel*>(c);if(self->references.fetch_sub(1,std::memory_order_acq_rel)==1)delete self;};
    abi.bind_prevalidated=[](void* c,const VecopsCall* call,VecopsError* error){return guard(error,[&]{static_cast<ResidentKernel*>(c)->bind(call,false);});};
    abi.bind=[](void* c,const VecopsCall* call,VecopsError* error){return guard(error,[&]{static_cast<ResidentKernel*>(c)->bind(call,true);});};
    abi.enter=[](void* c,VecopsError* error){return guard(error,[&]{
      auto& self=*static_cast<ResidentKernel*>(c);
      TORCH_CHECK(self.mutex.try_lock(),"concurrent resident communication instance use");
      try{self.entered_once=true;self.queue->drain();if(self.borrowed){TORCH_CHECK(self.consumer.enter(self.consumer.context,error)==VECOPS_STATUS_OK,"consumer enter failed");self.consumer_entered=true;}}
      catch(...){self.queue->communicator->abort();self.mutex.unlock();throw;}
    });};
    abi.run_lane=[](void* c,const VecopsTeamContextV1* team,VecopsError* error){return guard(error,[&]{
      auto& self=*static_cast<ResidentKernel*>(c);
      if(team->worker_id==0) {
        RECORD_FUNCTION(self.borrowed?"vecops_comm::resident_consume":"vecops_comm::resident_run",c10::ArrayRef<c10::IValue>());
        self.publication->plan->execute_on_team(self.out.data(),self.in.data(),*team,self.borrowed?&self.team_consumer:nullptr);
      } else self.publication->plan->execute_on_team(self.out.data(),self.in.data(),*team,self.borrowed?&self.team_consumer:nullptr);
    });};
    abi.leave=[](void* c,uint32_t success){auto& self=*static_cast<ResidentKernel*>(c);if(!success)self.queue->communicator->abort();if(self.consumer_entered){self.consumer.leave(self.consumer.context,success);self.consumer_entered=false;}self.mutex.unlock();};
    abi.replay=[](void* c,VecopsError* error){auto& self=*static_cast<ResidentKernel*>(c);
      auto status=self.abi.enter(c,error);if(status)return status;
      status=self.abi.thread_pool->team_executor->run(self.abi.thread_pool->context,self.abi.workers,c,
        [](void* c,const VecopsTeamContextV1* team){auto& self=*static_cast<ResidentKernel*>(c);char message[512]{};VecopsError e{sizeof(VecopsError),0,message,sizeof(message),0};TORCH_CHECK(self.abi.run_lane(c,team,&e)==VECOPS_STATUS_OK,"resident collective failed: ",message);},error);
      self.abi.leave(c,status==VECOPS_STATUS_OK);return status;
    };
  }
  void bind(const VecopsCall* call,bool validate) {
    TORCH_CHECK(call && call->num_values==data.size(),"resident communication argument count changed");
    for(size_t i=0;i<data.size();++i) {
      const auto& value=call->values[i];TORCH_CHECK(value.kind==VECOPS_VALUE_TENSOR,"resident communication requires tensors");
      const auto& t=value.value.tensor;const auto& m=metadata[i];
      if(validate){TORCH_CHECK(t.dtype==types[i] && t.rank==m.sizes.size(),"resident communication dtype/rank changed");
        for(size_t d=0;d<m.sizes.size();++d)TORCH_CHECK(t.sizes[d]==m.sizes[d] && t.strides[d]==m.strides[d],"resident communication layout changed");}
      data[i]=t.data?static_cast<char*>(t.data)+t.byte_offset:nullptr;
    }
    for(size_t i=0;i<in.size();++i){auto r=publication->in_refs[i];auto* p=static_cast<const char*>(data[outputs+r.argument]);in[i]=p?p+r.offset:nullptr;}
    if(!borrowed)for(size_t i=0;i<out.size();++i){auto r=publication->out_refs[i];auto* p=static_cast<char*>(data[r.argument]);out[i]=p?p+r.offset:nullptr;}
    if(borrowed) {
      consumer_values[0].value.tensor.data=addresses.data();consumer_values[1].value.tensor.data=data[0];
      VecopsCall frame{sizeof(VecopsCall),2,consumer_values.data(),nullptr,0,consumer.execution_context};
      char message[512]{};VecopsError error{sizeof(VecopsError),0,message,sizeof(message),0};
      TORCH_CHECK(consumer.bind_prevalidated(consumer.context,&frame,&error)==VECOPS_STATUS_OK,"resident consumer binding failed: ",message);
    }
  }
};
struct ResidentLease {
  ResidentKernel* kernel;
  explicit ResidentLease(ResidentKernel* p):kernel(p){}
  ~ResidentLease(){kernel->abi.destroy(kernel);}
  uint64_t address()const{return reinterpret_cast<uintptr_t>(&kernel->abi);}
  void configure_route(std::vector<int> modes){kernel->configure_route(std::move(modes));}
  std::vector<int> route_modes()const{return kernel->configured_modes;}
  std::string capacity_provider_id()const{return "comm-capacity/"+std::to_string(kernel->queue->capacity_provider_id);}
  bool provision_capacity(){std::lock_guard lock(kernel->mutex);return kernel->queue->provision_capacity();}
  bool preparation_consensus(bool success){std::lock_guard lock(kernel->mutex);return kernel->queue->preparation_consensus(success);}
};
void check_metadata(at::TensorList actual,const std::vector<vc::Layout>& expected,const std::vector<at::ScalarType>& types) {
  TORCH_CHECK(actual.size()==expected.size(),"resident graph communication arity changed");
  for(size_t i=0;i<actual.size();++i){auto m=layout(actual[i]);TORCH_CHECK(actual[i].scalar_type()==types[i] && m.sizes==expected[i].sizes && m.strides==expected[i].strides,"resident graph communication metadata changed");}
}
std::shared_ptr<ResidentLease> resident_plan(int64_t id,const std::vector<at::Tensor>& out,const std::vector<at::Tensor>& in,std::vector<uint32_t> types,uint64_t consumer_address) {
  std::shared_ptr<Prepared> p;std::shared_ptr<PreparedConsumer> b;
  {std::lock_guard lock(registry_mutex);
    if(consumer_address){auto it=consumers.find(id);if(it!=consumers.end())b=it->second.lock();if(b)p=b->publication;}
    else{auto it=registry.find(id);if(it!=registry.end())p=it->second.lock();}}
  TORCH_CHECK(p,"expired resident communication plan");
  const auto* pool=p->plan->thread_pool();
  if(!pool || pool->struct_size<sizeof(VecopsThreadPoolV1) || !pool->max_parallelism || !pool->team_executor ||
      pool->team_executor->struct_size<sizeof(VecopsTeamExecutorV1) || pool->team_executor->abi_major!=1 || !pool->team_executor->run)return {};
  if(consumer_address && reinterpret_cast<const VecopsPreparedKernelV1*>(consumer_address)->thread_pool!=pool)return {};
  if(b){check_metadata(out,{b->destination_layout},{b->destination_type});check_metadata(in,{b->source_layout},{b->source_type});}
  else{check_metadata(out,p->output_metadata,p->output_types);check_metadata(in,p->input_metadata,p->input_types);}
  return std::make_shared<ResidentLease>(new ResidentKernel(std::move(p),std::move(b),
      reinterpret_cast<const VecopsPreparedKernelV1*>(consumer_address),std::move(types)));
}
py::tuple resident_consumer_info(int64_t id) {
  std::shared_ptr<PreparedConsumer> p;
  {std::lock_guard lock(registry_mutex);auto it=consumers.find(id);if(it!=consumers.end())p=it->second.lock();}
  TORCH_CHECK(p,"expired resident consumer");return py::make_tuple(p->operation.schema().name(),p->addresses,p->specs);
}
struct RoutingLifetime {std::atomic<bool> alive{true};};
std::atomic<uint64_t> next_routing_id{1};
class Backend final:public c10d::Backend {
  std::shared_ptr<vc::Communicator> communicator_;
  vc::Options sample_options_;
  vc::Exchange sample_exchange_;
  const VecopsThreadPoolV1* executor_=nullptr;
  std::shared_ptr<Queue> queue_=std::make_shared<Queue>();
  mutable std::mutex plans_mutex_;
  bool sealed_=false;
  bool shutdown_confirmed_=false;
  std::string shutdown_error_;
  std::vector<int> route_override_;
  std::unordered_map<std::string,std::vector<int>> routes_;
  std::vector<std::shared_ptr<PreparedConsumer>> consumers_;
  std::unordered_map<std::string,std::shared_ptr<PreparedConsumer>> consumer_cache_;
  std::vector<std::shared_ptr<PreparedExchange>> exchanges_;
  std::unordered_map<std::string,std::shared_ptr<Prepared>> plans_;
  std::shared_ptr<RoutingLifetime> routing_lifetime_=std::make_shared<RoutingLifetime>();
  uint64_t routing_id_=next_routing_id.fetch_add(1,std::memory_order_relaxed);
public:
  std::string tuning_context;
  std::string payload_pool;
  uint64_t payload_page=0;
  uint64_t routing_id()const{return routing_id_;}
  std::shared_ptr<RoutingLifetime> routing_lifetime()const{return routing_lifetime_;}
  Backend(c10::intrusive_ptr<c10d::Store> store,int rank,int size,uint64_t timeout,
      uint32_t threads,uint64_t executor,std::shared_ptr<vecops::memory::model::MemoryRuntime> runtime,
      std::string pool,std::vector<int> groups,
      size_t initial,size_t threshold,size_t chunk,bool relay,
      std::vector<std::vector<int>> levels,std::vector<int> modes,
      bool row_owned=true,bool early_snapshot=false):c10d::Backend(rank,size) {
    auto phase=std::make_shared<size_t>(0);
    vc::Exchange exchange=[store,rank,size,phase](const std::string& own) {
      std::string prefix="vecops-comm/bootstrap/"+std::to_string((*phase)++)+"/";
      store->set(prefix+std::to_string(rank),std::vector<uint8_t>(own.begin(),own.end()));
      std::vector<std::string> result;result.reserve(size);
      for(int p=0;p<size;++p){auto data=store->get(prefix+std::to_string(p));result.emplace_back(data.begin(),data.end());}
      return result;
    };
    vc::Options options;options.threads=threads;
    payload_pool=pool;
    options.memory_runtime=std::move(runtime);options.payload_pool=std::move(pool);
    options.locality_groups=std::move(groups);options.timeout_ms=timeout;
    options.initial_capacity=initial;options.relay_threshold=threshold;
    options.chunk_bytes=chunk;options.relay=relay;
    options.replica_levels=std::move(levels);options.copy_levels=std::move(modes);
    options.row_owned_copy=row_owned;options.early_snapshot=early_snapshot;
    sample_options_=options;sample_options_.initial_capacity=1;sample_options_.relay=false;
    sample_options_.replica_levels.clear();sample_options_.copy_levels.clear();
    sample_exchange_=exchange;executor_=reinterpret_cast<const VecopsThreadPoolV1*>(executor);
    communicator_=vc::Communicator::create(rank,size,options,std::move(exchange),
        reinterpret_cast<const VecopsThreadPoolV1*>(executor));
    queue_->communicator=communicator_;
  }
  ~Backend()override {
    routing_lifetime_->alive.store(false,std::memory_order_release);
    communicator_->abort();
    std::lock_guard lock(registry_mutex);
    for(auto& [key,p]:plans_)registry.erase(p->id);
    for(auto& p:consumers_)consumers.erase(p->id);
    for(auto& p:exchanges_)exchanges.erase(p->id);
  }
  const std::string getBackendName()const override{return "vecops";}
  std::shared_ptr<Prepared> prepare(const std::string& kind,at::TensorList outputs,
      at::TensorList inputs,std::vector<int64_t> out_splits={},std::vector<int64_t> in_splits={},int root=0,std::vector<int> requested_modes={},bool materialized_calibration=true) {
    std::ostringstream stream;stream<<kind<<':'<<root;
    for(auto list:{outputs,inputs})for(const auto& t:list) {
      stream<<'/'<<t.scalar_type()<<':'<<t.sizes()<<':'<<t.strides();
    }
    for(auto n:out_splits)stream<<'o'<<n;
    for(auto n:in_splits)stream<<'i'<<n;
    std::string base_key=stream.str(),key=base_key;
    std::lock_guard lock(plans_mutex_);
    auto modes=requested_modes.empty()?route_override_:requested_modes;
    if(materialized_calibration && modes.empty())if(auto it=routes_.find(base_key);it!=routes_.end())modes=it->second;
    if(!materialized_calibration)key="borrowed/"+key;
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
    p->specification=spec;
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
  std::vector<double> measure_barriers(unsigned warmup,unsigned rounds) {
    TORCH_CHECK(warmup<=31 && rounds>=3 && rounds<=31,"invalid bounded barrier sample counts");
    // A dedicated tiny control communicator keeps the real transport's first
    // payload invocation unstarted, so prepare+seal can still provision all
    // source/relay roots cold. Its actual page is charged by the same Runtime.
    auto control=vc::Communicator::create(getRank(),getSize(),sample_options_,sample_exchange_,executor_);
    vc::Specification specification;specification.operation=vc::Operation::Barrier;
    auto plan=control->prepare(std::move(specification));
    std::vector<double> samples;samples.reserve(rounds);
    for(unsigned i=0;i<warmup+rounds;++i) {
      auto start=std::chrono::steady_clock::now();
      plan->execute(nullptr,nullptr);
      if(i>=warmup)samples.push_back(std::chrono::duration<double,std::micro>(std::chrono::steady_clock::now()-start).count());
    }
    control->close();
    return samples;
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
    auto publication=prepare("all_gather",outputs,{source},{},{},0,{},false);
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
    std::lock_guard lock(plans_mutex_);
    auto s=communicator_->statistics();py::dict d;d["calls"]=s.calls;d["plans"]=s.plans;
    d["capacity_growths"]=s.capacity_growths;d["copy_bytes"]=s.copy_bytes;
    d["row_copy_calls"]=s.row_copy_calls;d["row_copy_bytes"]=s.row_copy_bytes;
    d["shutdown_confirmed"]=shutdown_confirmed_;d["shutdown_error"]=shutdown_error_;
    return d;
  }
  void shutdown()override {
    std::lock_guard lock(plans_mutex_);
    routing_lifetime_->alive.store(false,std::memory_order_release);
    try {
      queue_->drain();communicator_->close();
      shutdown_confirmed_=true;sealed_=true;
    }catch(const std::exception& error) {
      // Teardown of an already failed collective must not mask its original
      // error. No stopped proof means no refund; core owners remain retained.
      shutdown_error_=error.what();abort();
    }
  }
  void seal(){std::lock_guard lock(plans_mutex_);communicator_->seal();sealed_=true;}
  std::string capacity_provider_id()const{return "comm-capacity/"+std::to_string(queue_->capacity_provider_id);}
  bool provision_capacity(){std::lock_guard lock(plans_mutex_);return queue_->provision_capacity();}
  bool preparation_consensus(bool success){std::lock_guard lock(plans_mutex_);return queue_->preparation_consensus(success);}
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
// Barrier has no tensor dispatch key. Use an explicit fallback: CatchAll is
// registered as CompositeImplicitAutograd and FakeTensor decomposes it before
// considering Meta, which would execute raw communication on fake addresses.
TORCH_LIBRARY_IMPL(vecops_comm,CompositeExplicitAutograd,m){m.impl("run",run);m.impl("consume",consume);m.impl("exchange",exchange_phase);}
PYBIND11_MODULE(TORCH_EXTENSION_NAME,m) {
  m.def("abort_exchange",abort_exchange);
  m.def("resident_plan",resident_plan);
  m.def("resident_consumer_info",resident_consumer_info);
  py::class_<ResidentLease,std::shared_ptr<ResidentLease>>(m,"ResidentLease")
    .def_property_readonly("abi_address",&ResidentLease::address)
    .def_property_readonly("route_modes",&ResidentLease::route_modes)
    .def_property_readonly("capacity_provider_id",&ResidentLease::capacity_provider_id)
    .def("provision_capacity",&ResidentLease::provision_capacity,py::call_guard<py::gil_scoped_release>())
    .def("preparation_consensus",&ResidentLease::preparation_consensus,py::call_guard<py::gil_scoped_release>())
    .def("configure_route",&ResidentLease::configure_route);
  py::class_<RoutingLifetime,std::shared_ptr<RoutingLifetime>>(m,"_RoutingLifetime")
    .def_property_readonly("alive",[](const RoutingLifetime& state){return state.alive.load(std::memory_order_acquire);});
  py::class_<Backend,c10d::Backend,c10::intrusive_ptr<Backend>>(m,"Backend")
    .def(py::init([](c10::intrusive_ptr<c10d::Store> store,int rank,int size,uint64_t timeout,
        uint32_t threads,uint64_t executor,py::object capsule,std::string pool,std::vector<int> groups,
        size_t initial,size_t threshold,size_t chunk,bool relay,std::vector<std::vector<int>> levels,
        std::vector<int> modes,bool row_owned,bool early_snapshot) {
      auto runtime=import_runtime_bridge(capsule);
      py::gil_scoped_release release;
      return c10::make_intrusive<Backend>(std::move(store),rank,size,timeout,threads,executor,
          std::move(runtime),std::move(pool),std::move(groups),initial,threshold,chunk,relay,
          std::move(levels),std::move(modes),row_owned,early_snapshot);
    }))
    .def("prepare",[](Backend& b,const std::string& kind,const std::vector<at::Tensor>& out,const std::vector<at::Tensor>& in,std::vector<int64_t> os,std::vector<int64_t> is,int root){return b.prepare(kind,out,in,std::move(os),std::move(is),root)->id;})
    .def("gatherv",[](Backend& b,at::Tensor out,at::Tensor in,std::vector<int64_t> counts){return b.submit("all_gatherv",{out},{in},std::move(counts));},py::call_guard<py::gil_scoped_release>())
    .def_readwrite("tuning_context",&Backend::tuning_context)
    .def_readonly("payload_pool",&Backend::payload_pool)
    .def_readwrite("payload_page",&Backend::payload_page)
    .def_property_readonly("_routing_id",&Backend::routing_id)
    .def_property_readonly("_routing_lifetime",&Backend::routing_lifetime)
    .def_property_readonly("capacity_provider_id",&Backend::capacity_provider_id)
    .def("provision_capacity",&Backend::provision_capacity,py::call_guard<py::gil_scoped_release>())
    .def("preparation_consensus",&Backend::preparation_consensus,py::call_guard<py::gil_scoped_release>())
    .def("prepare_routed",[](Backend& b,const std::string& k,const std::vector<at::Tensor>& o,const std::vector<at::Tensor>& i,std::vector<int> modes){return b.prepare(k,o,i,{}, {},0,std::move(modes))->id;})
    .def("plan_key",[](Backend& b,const std::string& k,const std::vector<at::Tensor>& o,const std::vector<at::Tensor>& i){return b.plan_key(k,o,i);})
    .def("set_route",&Backend::set_route)
    .def("configure_routes",&Backend::configure_routes)
    .def("measure_barriers",&Backend::measure_barriers,py::call_guard<py::gil_scoped_release>())
    .def("prepare_exchange",&Backend::prepare_exchange)
    .def("prepare_consumer",&Backend::prepare_consumer)
    .def("statistics",&Backend::statistics)
    .def("shutdown",&Backend::shutdown,py::call_guard<py::gil_scoped_release>())
    .def("abort",&Backend::abort)
    .def("seal",&Backend::seal);
}
