// SPDX-FileCopyrightText: 2026 HenryRenYz and vecops contributors
// SPDX-License-Identifier: MIT
// Explicit SDK-on-host probe/benchmark. Never run on a cluster login node.
#include "vecops/memory/LinuxRuntime.h"
#include "vecops/memory/LinuxSdma.h"
#include <algorithm>
#include <chrono>
#include <cstring>
#include <fstream>
#include <iostream>
#include <sstream>
#include <sched.h>
#include <stdexcept>
#if defined(__linux__)
#include <linux/mempolicy.h>
#include <sys/syscall.h>
#include <unistd.h>
#endif
using namespace vecops::memory::model;
using U=std::uint64_t;
template<class T> T get(Result<T> r) {
  if(!r) { const auto& e=std::get<Error>(r.state); throw std::runtime_error(e.code+": "+e.message); }
  return std::move(r).value();
}
void get(Result<void> r) { if(!r) throw std::runtime_error(r.error->code+": "+r.error->message); }
int main(int argc,char**argv) {
 try {
  if(argc!=4 && argc!=7 && argc!=8) throw std::runtime_error(
    "usage: LinuxSdmaSmoke hardware.toml root-group --list | hardware.toml root-group method route '64 MiB' rounds [shareable_remappable|private_movable]");
  std::ifstream input(argv[1]); std::stringstream text; text<<input.rdbuf();
  auto expansion=get(expand_hardware_template(text.str(),{{argv[2],1}}));
  auto discovery=get(discover_linux_cpu(get(probe_linux_topology()),&expansion,LinuxCpuProfile::lx2_flat));
  if(argc==4) {
    if(std::string(argv[3])!="--list") throw std::runtime_error("expected --list");
    for(const auto&m:discovery.graph.methods) for(const auto&r:m.routes)
      if(r.agent && r.controller &&
         std::find(discovery.scope.executable_agents.begin(),discovery.scope.executable_agents.end(),*r.agent)!=discovery.scope.executable_agents.end() &&
         std::find(discovery.scope.executable_agents.begin(),discovery.scope.executable_agents.end(),*r.controller)!=discovery.scope.executable_agents.end() &&
         std::find(discovery.scope.allocatable_pools.begin(),discovery.scope.allocatable_pools.end(),r.source)!=discovery.scope.allocatable_pools.end() &&
         std::find(discovery.scope.allocatable_pools.begin(),discovery.scope.allocatable_pools.end(),r.target)!=discovery.scope.allocatable_pools.end()) std::cout<<m.id<<'\t'<<r.id<<'\t'<<r.source<<'\t'<<r.target<<'\t'<<*r.agent<<'\t'<<*r.controller<<'\n';
    return 0;
  }
  const auto method=std::find_if(discovery.graph.methods.begin(),discovery.graph.methods.end(),[&](const auto&m){return m.id==argv[3];});
  if(method==discovery.graph.methods.end()) throw std::runtime_error("unknown method");
  const auto selected=std::find_if(method->routes.begin(),method->routes.end(),[&](const auto&r){return r.id==argv[4];});
  if(selected==method->routes.end() || !selected->agent || !selected->controller) throw std::runtime_error("unknown/non-device route");
  const auto route=*selected;
  const std::string method_id=method->id;
  const auto descriptor=std::find_if(discovery.descriptors.begin(),discovery.descriptors.end(),[&](const auto&d){return d.object==*route.agent && d.backend=="linux_sdma";});
  if(descriptor==discovery.descriptors.end()) throw std::runtime_error("route agent lacks SDMA device binding");
  const auto controller=std::find_if(discovery.descriptors.begin(),discovery.descriptors.end(),
    [&](const auto&d){return d.object==*route.controller && d.backend=="linux_cpu";});
  if(controller==discovery.descriptors.end()) throw std::runtime_error("controller lacks certified CPU binding");
  auto controller_cpus=get(parse_linux_id_list(controller->labels.at("allowed_cpus")));
  if(controller_cpus.empty() || controller_cpus.front()>=CPU_SETSIZE)
    throw std::runtime_error("controller CPU is unavailable for smoke affinity");
  struct Affinity {
    cpu_set_t previous{};
    explicit Affinity(unsigned cpu) {
      if(::sched_getaffinity(0,sizeof(previous),&previous)) throw std::runtime_error("cannot read controller affinity");
      cpu_set_t selected;CPU_ZERO(&selected);CPU_SET(cpu,&selected);
      if(::sched_setaffinity(0,sizeof(selected),&selected)) throw std::runtime_error("cannot pin certified controller CPU");
    }
    ~Affinity() { (void)::sched_setaffinity(0,sizeof(previous),&previous); }
  } affinity(controller_cpus.front());
  LinuxSdmaOptions sdk; sdk.device=descriptor->labels.at("device"); sdk.page_bytes=U(::sysconf(_SC_PAGESIZE));
  LinuxRuntimeOptions options; options.participant="sdma-benchmark";
  options.cpu_copy_methods={"cpu_copy"}; options.cpu_queues={{"host-q",*route.controller}};
  options.transfer_queues={{"sdma-q",{*route.agent,*route.controller,{method_id},get(make_linux_sdma_executor(sdk))}}};
  auto runtime=get(make_linux_runtime(std::move(discovery),options));
  auto view=get(runtime->view(get(runtime->current_scope())));
  const std::string mode_name=argc==8?argv[7]:"shareable_remappable";
  if(mode_name!="shareable_remappable" && mode_name!="private_movable")
    throw std::runtime_error("unknown backing mode");
  const auto mode=mode_name=="private_movable"?BackingMode::private_movable:BackingMode::shareable_remappable;
  for(const auto& id:{route.source,route.target}) {
    const std::string endpoint_pool[]={id};
    if(!get(runtime->mapping_capabilities(view,endpoint_pool,mode)).reserve_without_backing)
      throw std::runtime_error("backing mode unsupported for endpoint");
  }
  std::string pools[]={route.source,route.target}; auto caps=get(runtime->mapping_capabilities(view,pools));
  U bytes=get(parse_memory_size(argv[5])); const unsigned rounds=unsigned(std::stoul(argv[6]));
  if(!rounds || rounds>1000 || bytes%caps.page_bytes) throw std::runtime_error("rounds must be 1..1000 and size page-aligned");
  const auto& pool=*std::find_if(view.graph.pools.begin(),view.graph.pools.end(),[&](const auto&p){return p.id==route.source;});
  const auto space=*pool.address_space; const U q=caps.page_bytes;
  MemoryRecipe recipe{runtime->participant(),{},1,{},
    {{{"s-va",space,bytes,q},{"t-va",space,bytes,q}},
     {{"source","s-va",0,bytes,{"source-placement",space,q,{{0,bytes,route.source}}},{}},
      {"target","t-va",0,bytes,{"target-placement",space,q,{{0,bytes,route.target}}},{}}}},
    {{"source-root",runtime->participant(),route.source,bytes,q,mode,{}},
     {"target-root",runtime->participant(),route.target,bytes,q,mode,{}}},
    {{"s-va",0,"source-root",0,bytes,{AccessMode::read,AccessMode::write}},
     {"t-va",0,"target-root",0,bytes,{AccessMode::read,AccessMode::write}}}};
  auto addresses=get(runtime->prepare_addresses(view,recipe.addresses));
  auto lease=get(runtime->reserve(view,recipe,runtime->snapshot(view).epoch));
  auto roots=get(runtime->acquire_backing(view,recipe,*lease));
  auto binding=get(runtime->bind(view,recipe,addresses,roots));
  auto* source=static_cast<unsigned char*>(get(binding->slot("source")));
  auto* target=static_cast<unsigned char*>(get(binding->slot("target")));
  for(U i=0;i<bytes;++i) source[i]=static_cast<unsigned char>((i*73+(i>>12)*37)%251);
  std::memset(target,0,bytes);
  auto numa=[](void* p) { int node=-1;
    if(::syscall(SYS_get_mempolicy,&node,nullptr,0,p,MPOL_F_NODE|MPOL_F_ADDR)) throw std::runtime_error("actual NUMA query failed");
    return node; };
  std::cout<<"device="<<sdk.device<<" agent="<<*route.agent<<" controller="<<*route.controller
           <<" controller_cpu="<<::sched_getcpu()<<" source="<<route.source<<" numa="<<numa(source)<<" target="<<route.target<<" numa="<<numa(target)
           <<" bytes="<<bytes<<" chunk="<<sdk.chunk_bytes<<" rounds="<<rounds<<" backing="<<mode_name<<'\n';
  auto run=[&](bool dma,unsigned round) {
    std::memset(target,0xa5,bytes);
    MemoryCommand command{"copy",MemoryOpcode::copy,ByteRangeRef{MemoryRefKind::slot,"source",0,bytes},
      ByteRangeRef{MemoryRefKind::slot,"target",0,bytes},0,dma?method_id:"cpu_copy",dma?std::optional<std::string>{route.id}:std::nullopt,dma?"sdma-q":"host-q",{}, {}};
    InvocationContext context{{std::string(dma?"dma-":"cpu-")+std::to_string(round),binding->generation()},{},{}};
    const auto begin=std::chrono::steady_clock::now();
    auto submission=get(runtime->submit(view,{{command}},*binding,context));
    const double seconds=std::chrono::duration<double>(std::chrono::steady_clock::now()-begin).count();
    const auto status=get(runtime->events().query(submission.finished));
    if(submission.submission_error || status.state!=EventState::succeeded || !status.accesses_stopped) {
      for(const auto&d:get(poll_linux_transfer_quarantine()))
        std::cerr<<"quarantine token="<<d.token.id<<" stopped="<<d.status.accesses_stopped
                 <<" payload="<<d.retained.payload_bytes<<" pins="<<d.retained.registrations<<'\n';
      throw std::runtime_error(submission.submission_error?submission.submission_error->message:"copy did not finish");
    }
    if(std::memcmp(source,target,bytes)) throw std::runtime_error("copy correctness mismatch");
    get(runtime->events().release(std::span<const CompletionToken>(&submission.finished,1)));
    return seconds;
  };
  run(false,0);run(true,0);
  std::vector<double> cpu,dma;
  for(unsigned i=1;i<=rounds;++i) { if(i%2){cpu.push_back(run(false,i));dma.push_back(run(true,i));}
    else{dma.push_back(run(true,i));cpu.push_back(run(false,i));} }
  for(auto* times:{&cpu,&dma})std::sort(times->begin(),times->end());
  for(const auto&pair:{std::pair{"cpu",cpu[rounds/2]},std::pair{"sdma",dma[rounds/2]}})
    std::cout<<"method="<<pair.first<<" median_ms="<<pair.second*1e3<<" effective_GiB_s="<<double(bytes)/(1ULL<<30)/pair.second<<" inclusive=submit+pin+channel+copy+completion+unpin\n";
 } catch(const std::exception&e) { std::cerr<<e.what()<<'\n'; return 1; }
}
