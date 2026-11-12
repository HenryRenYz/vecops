# SPDX-FileCopyrightText: 2026 HenryRenYz and vecops contributors
# SPDX-License-Identifier: MIT
"""Cold, empirical route tuning. No measurement or policy search during replay.

Profiles match complete tensor metadata, rank placement, worker count, compiler
binary, stable hardware and configured frequency policy. Unknown/stale cases keep the
resource-model routing. Shared-resource contention is measured with all ranks active,
not estimated by treating nominal link bandwidths as independent capacities.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from dataclasses import asdict
import hashlib
import json
import os
import platform
from pathlib import Path
import statistics
import math
import time
import warnings


@dataclass(frozen=True)
class ConsumerTensor:
  """Address-free tensor geometry for an opaque borrowed-source consumer."""
  shape: tuple[int, ...]
  strides: tuple[int, ...]
  dtype: str


@dataclass(frozen=True)
class ConsumerRequest:
  """One observed consumer case, before production plans are activated.

  ``group`` is an existing process-local control object; it is not serialized.
  The request grants no permission to execute an arbitrary consumer on synthetic
  values. Its owner must supply valid, bounded source and initial output data.
  """
  operator_name: str
  specs: tuple[int, ...]
  source: ConsumerTensor
  output: ConsumerTensor
  group: object = field(repr=False)


def consumer_requests(captures):
  """Query unique borrowed consumers from metadata captures without rebinding.

  Both deferred discovery effects and existing logical collective recipes are
  supported. Their original instruction order is preserved across captures.
  No native plan, tensor payload, group, workspace or mapping is created here.
  Call after compilation and outside capture, before executable preparation.
  """
  from . import _template
  from vecops.graph._replay import _Ref
  result=[];seen=set()
  for capture in captures:
    tape=getattr(capture,'_torch_replay_tape',None)
    execution=getattr(capture,'_torch_execution_graph',None)
    if tape is None or execution is None:continue
    relocations={}
    for ordinal,recipe in getattr(tape,'_collective_relocations',{}).items():
      index=int(ordinal)
      if str(index)!=str(ordinal) or not 0<=index<len(tape.instructions) or index in relocations:
        raise ValueError('invalid collective relocation instruction index')
      relocations[index]=recipe
    for index,instruction in enumerate(tape.instructions):
      operation=getattr(getattr(instruction.function,'_schema',None),'name',None)
      if operation!='vecops_comm::consume':continue
      if len(instruction.arguments)!=3:
        raise ValueError('borrowed consumer argument contract changed')
      recipe=relocations.get(index)
      if recipe is not None:
        if recipe['operation']!=operation or instruction.arguments[0]is not None:
          raise ValueError('collective relocation instruction changed')
      else:recipe=_template.describe(operation,instruction.arguments[0])
      group=_template.group_for(recipe)
      def geometry(ref):
        if not isinstance(ref,_Ref):raise ValueError('collective template tensor reference changed')
        metadata=execution.tensors[ref.id].metadata
        return ConsumerTensor(tuple(metadata.shape),tuple(metadata.strides),metadata.dtype)
      request=ConsumerRequest(recipe['operator_name'],tuple(recipe['specs']),
          geometry(instruction.arguments[2]),geometry(instruction.arguments[1]),group)
      if request not in seen:
        seen.add(request);result.append(request)
  return tuple(result)


def consumer_calibration_options(group=None):
  """Detached routing policy and generated candidates for a cold planner.

  Return None when this group has no live resource routing compiler. No
  hardware names, native handles or mapping permissions enter this query.
  """
  from . import backend
  from ._routing import for_backend
  planner=for_backend(backend(group))
  if planner is None:return None
  return dict(policy=dict(planner.policy),partitions=[list(row)for row in planner.compiler.topology.partitions],
      candidates=[list(row)for row in planner.compiler.candidates()],
      installed_consumer_profiles=len(planner.consumer_routes))


def runtime_signature():
  policy={}
  for path in sorted(Path('/sys/devices/system/cpu/cpufreq').glob('policy*')):
    policy[path.name]={key:(path/key).read_text().strip() for key in
        ('scaling_min_freq','scaling_max_freq','scaling_governor') if (path/key).exists()}
  return {'boot':Path('/proc/sys/kernel/random/boot_id').read_text().strip(), 'frequency_policy':policy,
          'user_tag':os.environ.get('VECOPS_COMM_TUNING_TAG', '')}


def context(policy, members, threads, binary):
  # Every rank compares the same group-wide context, including asymmetric
  # worker counts and per-rank manual frequency tags. Reference-file enablement
  # is checked during bootstrap but is not part of the measured workload.
  return {'policy':policy.fingerprint,
          'members':[{k:v for k,v in member.items() if k not in {
              'threads','runtime','calibration','shared_page_profile','shared_page_error',
              'shared_page_audit','shared_page_measure','shared_page_filter'}} for member in members],
          'threads':[member.get('threads',threads) for member in members],
          'binary':binary,
          'runtime':[member['runtime'] if 'runtime' in member else runtime_signature() for member in members]}


def _read_identity(path):
  try:
    return Path(path).read_text().replace('\0', '\n').strip()
  except (OSError, UnicodeError):
    return ''


def _hardware_identity():
  """Stable Linux CPU/firmware facts; no serials, boot IDs or live clocks.

  Unknown CPUs deliberately cannot certify a portable performance profile.
  Record all logical CPUs, so rank affinity never hides heterogeneous cores.
  """
  cpus=[]
  for block in _read_identity('/proc/cpuinfo').split('\n\n'):
    fields={key.strip():value.strip() for line in block.splitlines()
            if ':' in line for key,value in [line.split(':',1)]}
    if not fields.get('processor','').isdigit():continue
    x86=('vendor_id','cpu family','model','stepping','flags')
    arm=('CPU implementer','CPU architecture','CPU variant','CPU part','CPU revision','Features')
    if all(fields.get(key) for key in x86):keys=x86
    elif all(fields.get(key) for key in arm):keys=arm
    else:return None
    keys=(*keys,'model name','microcode','physical id','core id','cache size','cpu cores','siblings')
    cpu={key:fields[key] for key in keys if key in fields}
    for key in ('flags','Features'):
      if key in cpu:cpu[key]=' '.join(sorted(cpu[key].split()))
    cpus.append({'id':int(fields['processor']),**cpu})
  if not cpus:return None
  firmware={}
  for name in ('bios_vendor','bios_version','bios_date','sys_vendor','product_name',
               'product_version','board_vendor','board_name','board_version'):
    if value:=_read_identity('/sys/devices/virtual/dmi/id/'+name):firmware[name]=value
  for name in ('model','compatible'):
    if value:=_read_identity('/sys/firmware/devicetree/base/'+name):firmware['device_tree_'+name]=value
  return {'architecture':platform.machine(),'kernel':platform.release(),
          'cpus':sorted(cpus,key=lambda cpu:cpu['id']),'firmware':firmware}


def _cpu_list(text):
  result=[]
  try:
    for item in text.split(','):
      bounds=[int(value) for value in item.split('-')]
      if len(bounds) not in (1,2) or min(bounds)<0 or bounds[-1]<bounds[0]:return []
      for cpu in range(bounds[0],bounds[-1]+1):
        if cpu not in result:result.append(cpu)
  except ValueError:
    return []
  return result


def _local_performance_facts(runtime, threads):
  # OpenMP can narrow the caller to one CPU after a larger team was created.
  # Prefer the explicit native-team list or the runtime's discovered CPU grant.
  configured=os.environ.get('VECOPS_THREAD_POOL_CPU_LIST','').strip()
  grant=_cpu_list(configured) if configured else []
  authoritative=bool(grant)
  if not configured:
    view=runtime.view()
    for binding in runtime.native.describe_bindings():
      if binding.object in view.executable_agents and binding.backend=='linux_cpu':
        grant.extend(_cpu_list(binding.labels.get('allowed_cpus','')))
    grant=sorted(set(grant))
    authoritative=bool(grant)
    if not grant and hasattr(os,'sched_getaffinity'):grant=sorted(os.sched_getaffinity(0))
  # Native teams may share CPUs in a known grant; a narrowed caller mask alone
  # cannot prove that it describes the complete team's configured placement.
  if not authoritative and len(grant)<threads:grant=[]
  return {'hardware':_hardware_identity(),'cpu_grant':grant,
          'worker_binding':{'provider':os.environ.get('VECOPS_THREAD_POOL','native'),
                            'torch_resident_team':os.environ.get('VECOPS_TORCH_RESIDENT_TEAM','0')}}


def _performance_context(live, facts):
  """Project performance facts only; leave live group/alias authority untouched."""
  # JSON copy also keeps callers from mutating the registry's nested evidence.
  result=json.loads(json.dumps(live))
  result['members']=[{key:value for key,value in member.items()
                      if key not in {'host','capacity_authority'}} for member in result['members']]
  result['runtime']=[{key:value for key,value in runtime.items() if key!='boot'}
                     for runtime in result['runtime']]
  result['performance_members']=json.loads(json.dumps(facts))
  return result


def bind_profile_context(backend, live, runtime):
  """Cold all-rank evidence, retained by the existing native-lifetime registry."""
  from ._routing import for_backend
  planner=for_backend(backend)
  local=_local_performance_facts(runtime,live['threads'][planner.rank])
  prefix='vecops/performance-identity/'
  planner.store.set(prefix+str(planner.rank),json.dumps(local,sort_keys=True))
  facts=[json.loads(planner.store.get(prefix+str(rank)).decode()) for rank in range(planner.size)]
  performance=_performance_context(live,facts)
  # Mode positions refer to these exact generated cuts, not just their count.
  performance['generated_topology']=asdict(planner.compiler.topology)
  planner._performance_context=json.dumps(performance,sort_keys=True)


def profile_context(backend):
  """Detached performance match key; never a mapping or alias permission."""
  from ._routing import for_backend
  planner=for_backend(backend)
  if planner is None or not hasattr(planner,'_performance_context'):
    raise ValueError('communication performance context is unavailable')
  return json.loads(planner._performance_context)


def _complete_performance_context(value):
  facts=value.get('performance_members',[])
  return (bool(facts) and len(facts)==len(value.get('members',[]))
          and all(row.get('hardware') and row['hardware'].get('architecture')
                  and row['hardware'].get('cpus') and row.get('cpu_grant') for row in facts))


def load_profile(path, expected):
  """Match a schema-2 profile against profile_context(backend), not live authority."""
  data=json.loads(Path(path).expanduser().read_text())
  if data.get('schema_version')!=2:
    raise ValueError('unsupported communication calibration schema')
  if not _complete_performance_context(expected) or data.get('context')!=expected:
    warnings.warn('vecops communication calibration is stale; retaining resource-model routing',RuntimeWarning)
    return {},hashlib.sha256(json.dumps(data,sort_keys=True).encode()).hexdigest()
  return _profile_routes(data),hashlib.sha256(json.dumps(data,sort_keys=True).encode()).hexdigest()


def _profile_routes(data):
  routes={}
  for record in data.get('cases',[]):
    modes=record['selected']
    if not isinstance(record['key'],str) or any(type(v) is not int or v not in (0,1) for v in modes):
      raise ValueError('invalid calibrated communication route')
    if record['key'].startswith('consumer/'):
      from ._routing import consumer_profile_key
      if (record.get('measurement')!='borrowed-consumer' or
          consumer_profile_key(record.get('consumer_contract',[]))!=record['key']):
        raise ValueError('consumer calibration contract does not match its key')
    if record['key'] in routes:
      raise ValueError('duplicate calibrated communication case')
    routes[record['key']]=modes
  return routes


def _paired_route_evidence(baseline, candidate, minimum_gain):
  """Observed paired-round evidence; median-2*MAD is not a confidence interval."""
  if (len(baseline)!=len(candidate) or not baseline
      or any(type(x) not in (int,float) or not math.isfinite(x) or x<=0
             for x in (*baseline,*candidate))):
    raise ValueError('route timings must be equal-length finite positive samples')
  gains=[1-value/reference for reference,value in zip(baseline,candidate)]
  median=statistics.median(gains)
  mad=statistics.median(abs(value-median) for value in gains)
  return dict(rounds=len(gains),paired_gains=gains,median_paired_gain=median,
      paired_gain_mad=mad,robust_margin=median-2*mad,
      wins=sum(value>minimum_gain for value in gains),
      mean_time_gain=1-statistics.fmean(candidate)/statistics.fmean(baseline))


def _rotated_route_indices(indices, round_index):
  return [indices[(offset+round_index)%len(indices)] for offset in range(len(indices))]


def _select_route_with_confirmation(records, minimum_gain, measure_round):
  """Select once; confirm only the original best against baseline for five rounds.

  measure_round(round_index, indices) must append one max-rank timing per index
  and perform the caller's ordinary restoration/numerical checks. No candidate
  is reselected after confirmation. Every sample, including adverse tails, stays
  in both the raw record and the final mean-time guard.
  """
  if (not records or not 0<=minimum_gain<1 or len(records[0])<3
      or len({len(row) for row in records})!=1):
    raise ValueError('route selection needs equally sampled candidates and >=3 rounds')
  # Validate all candidates, not just the eventual best.
  for row in records:_paired_route_evidence(records[0],row,minimum_gain)
  initial_rounds=len(records[0])
  best=min(range(len(records)),key=lambda i:statistics.median(records[i]))
  initial=_paired_route_evidence(records[0],records[best],minimum_gain)
  result=dict(policy='paired-mad-mean-fixed5/1',fixed_best_index=best,
      minimum_gain=minimum_gain,initial_rounds=initial_rounds,
      confirmation_rounds=0,initial_statistics=initial,
      confirmation_statistics=None,combined_statistics=initial)
  selected=0
  clear=(initial['robust_margin']>minimum_gain and initial['mean_time_gain']>minimum_gain)
  if best==0:
    result['reason']='baseline_has_best_initial_median'
  elif initial_rounds>=5 and initial['wins']==initial_rounds and clear:
    selected=best;result['reason']='unanimous_initial_robust_and_mean_gain'
  elif (initial['median_paired_gain']<=minimum_gain or initial['wins']<initial_rounds-1
        or 10*(initial['wins']+5)<9*(initial_rounds+5)):
    result['reason']='insufficient_initial_paired_evidence'
  else:
    counts=[len(row) for row in records]
    for index in range(5):
      measure_round(initial_rounds+index,(0,best))
      if [len(row) for row in records]!=[count+(index+1 if i in (0,best) else 0)
                                             for i,count in enumerate(counts)]:
        raise ValueError('confirmation must append only one baseline/fixed-best sample per round')
    confirmation=_paired_route_evidence(records[0][initial_rounds:],records[best][initial_rounds:],minimum_gain)
    combined=_paired_route_evidence(records[0],records[best],minimum_gain)
    result.update(confirmation_rounds=5,confirmation_statistics=confirmation,combined_statistics=combined)
    if (confirmation['wins']==5 and 10*combined['wins']>=9*combined['rounds']
        and combined['robust_margin']>minimum_gain and combined['mean_time_gain']>minimum_gain):
      selected=best;result['reason']='fixed_best_confirmed_all5_robust_and_mean_gain'
    else:result['reason']='confirmation_inconclusive_keep_baseline'
  # Historical diagnostic definition is retained, even when a robust candidate
  # is accepted despite one outlier. It no longer alone decides the route.
  noise=max((max(row)-min(row))/max(statistics.median(row),1e-12)
            for row in (records[0],records[best]))
  result.update(selected_index=selected,candidate_round_counts=[len(row) for row in records],noise_fraction=noise)
  return selected,result


def tune_all_gather(source, outputs, *, candidates, group=None, iterations=10,
                    rounds=5, warmup=3, minimum_gain=.05):
  """Measure actual materialized layouts under simultaneous all-rank load.

  Call collectively before capture/seal. Candidates are share/copy bit vectors
  for the configured coarse-to-fine levels. Every candidate is prebound and
  warmed outside timing. Rounds rotate candidate order; choose by median of
  max-rank time. Paired gains require robust and all-sample mean margins; an
  uncertain original best receives at most five additional baseline/best rounds.
  This does not prove an end-to-end model gain: benchmark that before rollout.
  Returns a profile case; write_profile installs the complete calibrated table.
  """
  import torch
  import torch.distributed as dist
  from . import backend
  from vecops.graph.capture import current_capture
  if current_capture() is not None:
    raise RuntimeError('communication tuning must precede graph capture')
  if iterations<1 or rounds<3 or warmup<1 or not 0<=minimum_gain<1:
    raise ValueError('invalid communication tuning parameters')
  native=backend(group)
  candidates=list(dict.fromkeys(tuple(x) for x in candidates))
  if not candidates or any(any(type(v) is not int or v not in (0,1) for v in x) for x in candidates):
    raise ValueError('routing candidates must contain share/copy bit vectors')
  handles=[]
  try:
    for modes in candidates:
      native.set_route(list(modes))
      handle=native.prepare('all_gather',list(outputs),[source],[],[],0)
      for _ in range(warmup):torch.ops.vecops_comm.run.default(handle,list(outputs),[source])
      handles.append(handle)
    records=[[] for _ in candidates]
    samples=[torch.empty(1,dtype=torch.float64) for _ in range(dist.get_world_size(group))]
    value=torch.empty(1,dtype=torch.float64)
    def measure_round(round_index,indices):
      for i in _rotated_route_indices(indices,round_index):
        dist.barrier(group=group)
        started=time.perf_counter_ns()
        for _ in range(iterations):torch.ops.vecops_comm.run.default(handles[i],list(outputs),[source])
        value[0]=(time.perf_counter_ns()-started)/1e9/iterations
        dist.all_gather(samples,value,group=group)
        records[i].append(max(x.item() for x in samples))
    for round_index in range(rounds):measure_round(round_index,tuple(range(len(candidates))))
    selected,selection=_select_route_with_confirmation(records,minimum_gain,measure_round)
    medians=list(map(statistics.median,records))
    noise=selection['noise_fraction']
    key=native.plan_key('all_gather',list(outputs),[source])
    record={'key':key,'selected':list(candidates[selected]),'baseline':list(candidates[0]),
            'gain':1-medians[selected]/medians[0],'noise_fraction':noise,'selection':selection,
            'candidates':[{'modes':list(m),'maxrank_seconds':row,'median':median}
                          for m,row,median in zip(candidates,records,medians)]}
  finally:
    native.set_route([])
  return record


def write_profile(path, cases, *, group=None):
  """Atomically write rank-zero evidence; install the identical plan on all ranks."""
  import torch.distributed as dist
  from . import backend
  native=backend(group)
  performance=profile_context(native)
  if not _complete_performance_context(performance):
    raise ValueError('communication calibration requires stable CPU identity and complete CPU grants')
  data={'schema_version':2,'context':performance,
        'runtime_context':json.loads(native.tuning_context),'cases':list(cases)}
  routes=_profile_routes(data)
  if dist.get_rank(group)==0:
    path=Path(path).expanduser().resolve();path.parent.mkdir(parents=True,exist_ok=True)
    temporary=path.with_name(path.name+f'.{os.getpid()}.partial')
    fd=os.open(temporary,os.O_CREAT|os.O_EXCL|os.O_WRONLY,0o600)
    with os.fdopen(fd,'w') as stream:json.dump(data,stream,indent=2);stream.write('\n')
    temporary.replace(path)
  dist.barrier(group=group)
  from ._routing import install_profiles
  install_profiles(native,routes)
  return data


def tune_consumer(source, output, operator_name, specs=(), *, candidates, group=None,
                  output_placement=None, iterations=10, rounds=5, warmup=3,
                  minimum_gain=.05, rtol=1e-5, atol=1e-6):
  """Measure the real borrowed-source native consumer, including its workspace.

  Call collectively before capture/seal. The consumer must already be compiled.
  Its metadata-only lease is activated with a new memory Program, so native
  workspace allocation is charged by the same runtime as communication payload.
  Output pages must be observable or supplied as an actual Program placement
  (pool-to-byte weights, not a framework capacity-budget declaration).

  Each block restores the caller's initial output before timing, including
  read-modify-write consumers. All candidates execute the same repeat count;
  every block is numerically checked against the flat candidate outside timing.
  Both tuning entry points use the same paired robust/mean selection and one
  bounded fixed-best confirmation batch; no model inference is warmed here.
  Profiles match exact consumer DSO/metadata/specs/output placement and all-rank
  runtime context. Publisher input copying is a common term across these routes.
  """
  import torch
  import torch.distributed as dist
  from types import SimpleNamespace
  from vecops import _schema_bridge
  from vecops.graph.capture import current_capture
  from vecops.memory import current_runtime, default_runtime
  from vecops.planner.compiler import GraphMemoryCompiler,native_workspace_requirements,activate_native_buffers
  from . import backend
  from ._routing import for_backend,consumer_contract,consumer_profile_key
  if current_capture() is not None:
    raise RuntimeError('communication tuning must precede graph capture')
  if iterations<1 or rounds<3 or warmup<1 or not 0<=minimum_gain<1:
    raise ValueError('invalid communication tuning parameters')
  native=backend(group);planner=for_backend(native)
  if planner is None:
    raise ValueError('consumer tuning requires a live resource routing compiler')
  candidates=list(dict.fromkeys(tuple(row) for row in candidates))
  levels=len(planner.compiler.topology.partitions) or 1
  if (not candidates or any(len(row)!=levels or any(type(v) is not int or v not in (0,1) for v in row)
                            for row in candidates) or any(candidates[0])):
    raise ValueError('consumer candidates must start with the flat route and match generated levels')
  runtime=current_runtime() or default_runtime();view=runtime.view()
  if output_placement is None:
    if output.device.type!='cpu' or not output.is_contiguous() or output.nbytes==0:
      raise ValueError('consumer tuning requires explicit placement for a noncontiguous/empty output')
    from vecops.memory import native as model
    address=model.VirtualAddress();address.address_space='host';address.value=output.data_ptr()
    external=model.ExternalMemoryView('consumer-tuning-output',address,output.nbytes,output,None,False)
    observation=runtime.observe_external(view,external)
    if not observation.complete:
      raise ValueError('consumer output placement is not observable; supply actual Program placement')
    output_placement={}
    for segment in observation.observed_placement:
      output_placement[segment.pool]=output_placement.get(segment.pool,0)+segment.size_bytes
  if (not isinstance(output_placement,dict) or not output_placement or
      any(not isinstance(pool,str) or type(size) is not int or size<=0 for pool,size in output_placement.items())):
    raise ValueError('consumer output placement must contain positive exact pool byte counts')
  local={'kind':'borrowed-consumer','bytes':source.nbytes,'inputs':[None],
         'outputs':[output_placement],'consumer':consumer_contract(operator_name,source,output,specs)}
  ordinal=planner.ordinal;planner.ordinal+=1
  prefix=f'vecops/consumer-tune/{ordinal}/'
  planner.store.set(prefix+str(planner.rank),json.dumps(local,sort_keys=True))
  rows=[json.loads(planner.store.get(prefix+str(rank)).decode()) for rank in range(planner.size)]
  key=consumer_profile_key(rows)
  if key is None:
    raise ValueError('consumer profile lacks an exact DSO or output placement contract')
  namespace,name=operator_name.split('::',1)
  operation=getattr(getattr(torch.ops,namespace),name).default
  addresses=torch.empty(planner.size,dtype=torch.int64)
  def metadata(tensor):
    return SimpleNamespace(shape=tuple(tensor.shape),strides=tuple(tensor.stride()),dtype=str(tensor.dtype),device=str(tensor.device))
  described=_schema_bridge.prepare_replay_plan(operation,(addresses,output,*specs),metadata,
                                               describe_only=True,for_eager=True)
  if described is None:
    raise ValueError('consumer has no metadata-only native workspace description')
  lease=described[0];prepared=None
  try:
    consumers=tuple(agent.id for agent in view.graph.agents if agent.id in view.executable_agents
                    and 'host_load_store' in agent.capabilities)
    failure=None
    try:
      requirements=native_workspace_requirements(lease,prefix='consumer',consumers=consumers)
      program=GraphMemoryCompiler(runtime,view,runtime.snapshot(view)).compile(requirements)
      prepared=program.prepare(runtime,view);prepared.materialize()
      activate_native_buffers(lease,prepared,prefix='consumer')
    except Exception as error:
      failure=error
    if not native.preparation_consensus(failure is None):
      if failure is not None:raise failure
      raise RuntimeError('peer consumer workspace preparation failed')
    # These recipes contain offsets and pool identities, never live pointers.
    # Keep all ranks' actual workspace tiers alongside the measured evidence.
    planner.store.set(prefix+'workspace/'+str(planner.rank),program.dumps())
    workspace_recipes=[planner.store.get(prefix+'workspace/'+str(rank)).decode()for rank in range(planner.size)]
    initial=output.clone();handles=[];reference=None
    for modes in candidates:
      native.set_route(list(modes))
      handle=native.prepare_consumer(source,output,operator_name,list(specs));handles.append(handle)
      output.copy_(initial)
      for _ in range(warmup):torch.ops.vecops_comm.consume.default(handle,output,source)
      if reference is None:reference=output.clone()
      else:torch.testing.assert_close(output,reference,rtol=rtol,atol=atol)
    records=[[] for _ in candidates]
    samples=[torch.empty(1,dtype=torch.float64) for _ in range(planner.size)]
    value=torch.empty(1,dtype=torch.float64)
    def measure_round(round_index,indices):
      results={}
      for index in _rotated_route_indices(indices,round_index):
        output.copy_(initial);dist.barrier(group=group)
        started=time.perf_counter_ns()
        for _ in range(iterations):torch.ops.vecops_comm.consume.default(handles[index],output,source)
        value[0]=(time.perf_counter_ns()-started)/1e9/iterations
        dist.all_gather(samples,value,group=group)
        records[index].append(max(sample.item() for sample in samples))
        results[index]=output.clone()
      for result in results.values():torch.testing.assert_close(result,results[0],rtol=rtol,atol=atol)
    for round_index in range(rounds):measure_round(round_index,tuple(range(len(candidates))))
    selected,selection=_select_route_with_confirmation(records,minimum_gain,measure_round)
    medians=list(map(statistics.median,records))
    noise=selection['noise_fraction']
    return dict(key=key,measurement='borrowed-consumer',selected=list(candidates[selected]),baseline=list(candidates[0]),
                consumer_contract=rows,publisher_input='common to all routes',
                workspace_recipes=workspace_recipes,
                gain=1-medians[selected]/medians[0],noise_fraction=noise,selection=selection,
                candidates=[dict(modes=list(mode),maxrank_seconds=row,median=median)
                            for mode,row,median in zip(candidates,records,medians)])
  finally:
    native.set_route([])
    lease.close()
    if prepared is not None:prepared.close()
