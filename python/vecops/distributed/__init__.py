# SPDX-FileCopyrightText: 2026 HenryRenYz and vecops contributors
# SPDX-License-Identifier: MIT
"""Optional CPU c10d backend and prepared collective graph integration.

Importing the configuration types does not import Torch. Call register_backend
before init_process_group('vecops'). No dependency on another communication
library is required. Transport progress is cooperative and ordered at wait().
"""
from .config import Hints, Options, Policy


def _payload_runtime():
  """Select an opaque runtime pool using public CPU access costs."""
  import os
  from vecops import memory
  runtime=memory.current_runtime() or memory.default_runtime()
  native=runtime.native
  view=runtime.view()
  module=memory.native
  def value(name,**fields):
    result=getattr(module,name)()
    for key,item in fields.items():setattr(result,key,item)
    return result
  requested=os.environ.get('VECOPS_COMM_POOL','').strip()
  pools=[pool for pool in view.graph.pools if pool.id in view.allocatable_pools
         and (not requested or pool.id==requested)]
  if not pools:
    raise ValueError('VECOPS_COMM_POOL does not name an allocatable runtime pool' if requested
                     else 'communication requires an allocatable runtime pool')
  consumers=[agent.id for agent in view.graph.agents if agent.id in view.executable_agents
             and 'host_load_store' in agent.capabilities]
  if not consumers:raise ValueError('CPU communication requires an executable host_load_store agent')
  context=value('CostContext',view_epoch=view.epoch,execution_slots=view.execution_slots,active=[],
                include_preparation=False)
  choices=[]
  for index,pool in enumerate(pools):
    capability=native.mapping_capabilities(view,[pool.id])
    if not (capability.shareable and capability.remappable and capability.partial_backing_map):continue
    page=capability.page_bytes
    placement=value('PlacementPlan',address_space=pool.address_space,stripe_bytes=page,weights={pool.id:1})
    resolved=native.resolve_placement(view,placement,page,module.BackingMode.shareable_remappable,
                                    [module.AccessMode.read,module.AccessMode.write])
    endpoint=value('TransferEndpoint',address_space=resolved.address_space,segments=resolved.segments)
    unknown=False;cost=0.0
    for consumer in consumers:
      for access in (module.AccessMode.read,module.AccessMode.write):
        try:
          estimate=native.costs.estimate_access(view,value('AccessQuery',agent=consumer,memory=endpoint,
              access=access,payload_bytes=page),context)
        except RuntimeError:
          # The allocation/mapping backend has already certified this pool;
          # an absent reference path is an unknown cost, not zero bandwidth.
          unknown=True
        else:
          unknown|=estimate.quality==module.CostQuality.incomplete or bool(estimate.missing_facts)
          cost+=estimate.total_us
    choices.append(((unknown,cost,index),pool.id,page))
  if not choices:raise ValueError('no runtime pool supports shared communication page aliases')
  _,pool,page=min(choices)
  return runtime,pool,page



def register_backend():
  import torch.distributed as dist
  if 'vecops' not in dist.Backend.backend_list:
    dist.Backend.register_backend('vecops', _create, extended_api=True, devices=['cpu'])


def _create(dist_options, pg_options=None):
  import json
  import os
  from pathlib import Path
  import torch
  from vecops import _C
  from ._build import load
  options = pg_options or Options()
  if not isinstance(options, Options):
    raise TypeError('vecops backend expects vecops.distributed.Options')
  platform = Policy.load(options.config)
  runtime,payload_pool,payload_page=_payload_runtime()
  rank, size = dist_options.group_rank, dist_options.group_size
  store = dist_options.store
  extension = load()
  threads = options.threads or int(os.environ.get('VECOPS_NUM_THREADS', torch.get_num_threads()))
  if threads < 1:
    raise ValueError('vecops communication threads must be positive')
  policy = platform.communication
  from .tuning import context, load_profile, runtime_signature
  from .compiler import rank_facts, CommunicationCompiler
  local = {'resources':rank_facts(runtime,payload_pool,threads), 'host': Path('/proc/sys/kernel/random/boot_id').read_text().strip(),
           'platform': platform.fingerprint, 'binary': extension.build_id,
           'threads':threads, 'runtime':runtime_signature(),
           'payload_pool':payload_pool,'payload_page':payload_page,
           'capacity_authority':runtime.native.capacity_authority(),
           'calibration':bool(policy.get('calibration'))}
  store.set(f'vecops/identity/{rank}', json.dumps(local))
  members = [json.loads(store.get(f'vecops/identity/{r}').decode()) for r in range(size)]
  if len({x['host'] for x in members}) != 1:
    raise RuntimeError('vecops shared-memory backend requires one host; network transport is not enabled')
  if size>1 and (any(x['capacity_authority'] is None for x in members)
                 or len({x['capacity_authority'] for x in members})!=1):
    raise ValueError('multi-process communication requires one shared MemoryRuntime capacity authority; '
                     'configure the same job-local capacity_directory before init_process_group')
  if len({x['platform'] for x in members}) != 1:
    raise ValueError('vecops platform configurations differ across ranks')
  if len({x['binary'] for x in members}) != 1:
    raise ValueError('vecops communication binaries differ across ranks')
  if len({x['calibration'] for x in members}) != 1:
    raise ValueError('communication calibration enablement differs across ranks')
  if len({x['payload_page'] for x in members}) != 1:
    raise ValueError('communication participants require a common runtime page quantum')
  compiler=CommunicationCompiler([member['resources'] for member in members],
      stage_us=policy.get('stage_us'))
  groups=list(range(size))
  timeout = options.timeout_ms
  if timeout is None:
    timeout = int(policy.get('timeout_ms', dist_options.timeout.total_seconds()*1000))
  if timeout <= 0:
    raise ValueError('vecops communication timeout must be positive')
  levels=[list(level) for level in compiler.topology.partitions]
  modes=[0]*len(levels)
  # Use the identical process-local provider as source kernels when opting
  # into Torch resident teams; pointer identity is part of island composition.
  if (os.environ.get("VECOPS_THREAD_POOL", "native").strip().lower() == "torch"
      and os.environ.get("VECOPS_TORCH_RESIDENT_TEAM", "0") == "1"):
    from vecops._torch_pool import address
    pool_address = int(address().value)
  else:
    pool_address = int(_C.native_thread_pool_address(threads))
  backend = extension.Backend(store, rank, size, timeout, threads,
      pool_address, runtime.native.export_bridge(), payload_pool, groups,
      int(policy.get('initial_capacity_bytes', 1 << 20)),
      int(policy.get('relay_threshold_bytes', 1 << 20)),
      int(policy.get('chunk_bytes', 256 << 10)),
      bool(levels),
      levels, modes,
      policy.get('copy_schedule','auto')!='peer', bool(policy.get('early_snapshot',False)))
  backend.payload_page=payload_page
  from ._routing import install
  install(backend,compiler,store,rank,size,policy)
  signature = context(platform, members, threads, extension.build_id)
  backend.tuning_context = json.dumps(signature, sort_keys=True)
  # Performance portability must not weaken this full live authority context.
  from .tuning import bind_profile_context, profile_context
  bind_profile_context(backend,signature,runtime)
  if profile := policy.get('calibration'):
    path = Path(profile).expanduser()
    if not path.is_absolute():
      path = Path(platform.path).parent/path
    import hashlib
    try:
      routes, digest = load_profile(path, profile_context(backend))
      if any(len(route)!=(len(levels) or 1) for route in routes.values()):
        raise ValueError('calibrated route level count differs from generated resource partitions')
      status = {'digest':digest, 'routes':hashlib.sha256(
          json.dumps(routes,sort_keys=True).encode()).hexdigest(), 'error':None}
    except Exception as error:
      routes = {}
      status = {'digest':None, 'routes':None, 'error':str(error)}
    store.set(f'vecops/calibration/{rank}',json.dumps(status))
    statuses=[json.loads(store.get(f'vecops/calibration/{r}').decode()) for r in range(size)]
    if any(row['error'] is not None for row in statuses):
      backend.abort()
      raise ValueError('communication calibration failed: '+str([row['error'] for row in statuses]))
    if len({row['digest'] for row in statuses})!=1:
      backend.abort()
      raise ValueError('communication calibration profiles differ across ranks')
    if len({row['routes'] for row in statuses})!=1:
      # A profile is usable by the entire collective or by nobody. Never let
      # one rank enter a replica stage while another silently falls back.
      import warnings
      warnings.warn('calibration applicability differs across ranks; retaining resource-model routing',RuntimeWarning)
      routes={}
    from ._routing import install_profiles
    install_profiles(backend,routes)
  return backend


def backend(group=None):
  import torch
  import torch.distributed as dist
  group = group or dist.distributed_c10d._get_default_group()
  if str(dist.get_backend(group)).lower() != 'vecops':
    raise ValueError('process group is not a vecops communicator')
  return group._get_backend(torch.device('cpu'))


def statistics(group=None):
  native=backend(group)
  result=native.statistics()
  result['payload_pool']=native.payload_pool
  result['payload_page_bytes']=native.payload_page
  from ._routing import for_backend
  if compiler:=for_backend(native):
    result['routing']={'partitions':[list(p) for p in compiler.compiler.topology.partitions],
      'quality':compiler.compiler.topology.quality,'missing_facts':list(compiler.compiler.topology.missing_facts),
      'synchronization':compiler.synchronization,'plans':compiler.records}
  return result


def seal(group=None):
  """Freeze transport capacity and new plans after graph/model preparation."""
  backend(group).seal()


def native_operators(group=None):
  from ._build import load
  import torch.distributed as dist
  group = group or dist.distributed_c10d._get_default_group()
  if str(dist.get_backend(group)).lower() != 'vecops':
    import torch
    return {name:(torch._C, group) for name in (
        '_c10d_functional::all_gather_into_tensor',
        '_c10d_functional::all_to_all_single', '_c10d_functional::wait_tensor')}
  return {name: (load(), group) for name in ('vecops_comm::run', 'vecops_comm::consume', 'vecops_comm::exchange')}


def launch(kind, outputs, inputs, *, output=None, output_splits=(), input_splits=(), root=0,
           group=None, async_op=False, copy_levels=None):
  """Capture a prebound native plan; actual ordered progress happens at wait.

  Used by a graph adapter only. Preparing the handle allocates layout state but
  does not communicate or inspect Tensor addresses (including FakeTensor).
  Replay executes the native operator without Python/c10d Work allocation.
  """
  import torch
  if copy_levels is not None:
    if kind != 'all_gather' or root or output_splits or input_splits:
      raise ValueError('per-operation replica hints currently require list AllGather')
    handle = backend(group).prepare_routed(kind, list(outputs), list(inputs), list(copy_levels))
  else:
    handle = backend(group).prepare(kind, list(outputs), list(inputs),
        list(output_splits or ()), list(input_splits or ()), root)
  from ._template import remember
  remember('vecops_comm::run', handle, group, kind=kind,
      options=dict(output_splits=list(output_splits or ()), input_splits=list(input_splits or ()),
                   root=root, copy_levels=None if copy_levels is None else list(copy_levels)))
  from vecops.graph.capture import current_capture
  capture = current_capture()
  if capture is not None:
    tape = getattr(capture, '_torch_replay_tape', None)
    if tape is not None:
      tape.native_operators.update(native_operators(group))
  class Work:
    def __init__(self):
      self.complete = False
    def wait(self):
      if not self.complete:
        torch.ops.vecops_comm.run.default(handle, list(outputs), list(inputs))
        self.complete = True
      return output
  work = Work()
  return work if async_op else work.wait()


def all_gatherv_into_tensor(output, input, counts, *, group=None, async_op=False):
  work = backend(group).gatherv(output, input, list(counts))
  if async_op:
    return work
  work.wait()
  return output


class Group:
  """Materialized collective values; other backends ignore performance hints.

  Source/target axes describe logical sharding, never hardware locations.
  Capture uses native effects, including when falling back to c10d functional
  operators. An ordinary c10d call must never silently disappear from replay.
  """
  def __init__(self, process_group=None):
    import torch.distributed as dist
    self.process_group = process_group or dist.distributed_c10d._get_default_group()

  def all_gather(self, tensor, dim=0, *, hints=None):
    import torch
    import torch.distributed as dist
    if tensor.ndim == 0:
      raise ValueError('all_gather value API requires a nonscalar tensor')
    dim %= tensor.ndim
    world = dist.get_world_size(self.process_group)
    if str(dist.get_backend(self.process_group)).lower() == 'vecops':
      shape = list(tensor.shape)
      shape[dim] *= world
      result = torch.empty(shape, dtype=tensor.dtype, device=tensor.device)
      parts = [result.narrow(dim, peer*tensor.size(dim), tensor.size(dim)) for peer in range(world)]
      modes = hints.copy_levels if hints is not None else None
      if self._capturing() or modes is not None:
        return launch('all_gather', parts, [tensor], output=result, group=self.process_group, copy_levels=modes)
      dist.all_gather(parts, tensor, group=self.process_group)
      return result
    if self._capturing():
      self._register_functionals()
      source = tensor.contiguous().reshape(-1)
      gathered = torch.ops._c10d_functional.all_gather_into_tensor.default(
          source, world, dist.distributed_c10d._get_process_group_name(self.process_group))
      ready = torch.ops._c10d_functional.wait_tensor.default(gathered)
      return torch.cat(list(ready.reshape(world, *tensor.shape).unbind(0)), dim=dim)
    parts = [torch.empty_like(tensor) for _ in range(dist.get_world_size(self.process_group))]
    dist.all_gather(parts, tensor, group=self.process_group)
    return torch.cat(parts, dim=dim)

  def reshard(self, tensor, source_axis=0, target_axis=1, *, hints=None):
    """Gather the current sharded axis and evenly shard target_axis."""
    import torch
    import torch.distributed as dist
    world = dist.get_world_size(self.process_group)
    source_axis %= tensor.ndim
    target_axis %= tensor.ndim
    if source_axis == target_axis or tensor.size(target_axis) % world:
      raise ValueError('reshard requires distinct axes and an evenly divisible target axis')
    local = tensor.size(target_axis)//world
    inputs = [tensor.narrow(target_axis, r*local, local) for r in range(world)]
    if str(dist.get_backend(self.process_group)).lower() == 'vecops':
      shape = list(tensor.shape)
      shape[source_axis] *= world
      shape[target_axis] = local
      result = torch.empty(shape, dtype=tensor.dtype, device=tensor.device)
      outputs = [result.narrow(source_axis, r*tensor.size(source_axis), tensor.size(source_axis)) for r in range(world)]
      if self._capturing():
        return launch('all_to_all', outputs, inputs, output=result, group=self.process_group)
      dist.all_to_all(outputs, inputs, group=self.process_group)
      return result
    packed = torch.cat([part.contiguous().reshape(-1) for part in inputs])
    counts = [inputs[0].numel()]*world
    if self._capturing():
      self._register_functionals()
      received = torch.ops._c10d_functional.all_to_all_single.default(
          packed, counts, counts, dist.distributed_c10d._get_process_group_name(self.process_group))
      received = torch.ops._c10d_functional.wait_tensor.default(received)
    else:
      received = torch.empty_like(packed)
      dist.all_to_all_single(received, packed, counts, counts, group=self.process_group)
    parts = received.reshape(world, *inputs[0].shape).unbind(0)
    return torch.cat(list(parts), dim=source_axis)

  @staticmethod
  def _capturing():
    from vecops.graph.capture import current_capture
    capture = current_capture()
    return capture is not None and getattr(capture, '_torch_replay_tape', None) is not None

  def _register_functionals(self):
    from vecops.graph.capture import current_capture
    current_capture()._torch_replay_tape.native_operators.update(native_operators(self.process_group))


def group(process_group=None):
  return Group(process_group)


__all__ = ['Hints', 'Options', 'Policy', 'all_gatherv_into_tensor', 'backend', 'group',
           'launch', 'native_operators', 'register_backend', 'seal', 'statistics']


def consume_all_gather(source, output, operator_name, specs=(), *, group=None, owners=()):
  """Publish contiguous rank snapshots and run one synchronous native consumer.

  The registered CPU operator receives an int64 address table and mutable output,
  followed by integer specialization values. It must finish all source reads
  before returning and must not retain addresses or communicate recursively.
  Only output escapes this effect. Caller retains all compiled operator owners
  for the graph lifetime. This explicit API grants borrowing; Hints never do.
  """
  import torch
  from ._template import record_deferred_consumer
  if record_deferred_consumer(source,output,operator_name,specs,group,owners):
    return output
  # Preparing owns an internal address table, not part of the captured program.
  # Do not let its allocation become a graph factory or inspect its storage
  # through the capture's TorchFunction escape guard.
  with torch._C.DisableTorchFunction(), torch._C._DisableTorchDispatch():
    handle = backend(group).prepare_consumer(source, output, operator_name, list(specs))
  from ._template import remember
  remember('vecops_comm::consume', handle, group, operator_name=operator_name, specs=list(specs))
  from vecops.graph.capture import current_capture
  capture = current_capture()
  tape = getattr(capture, '_torch_replay_tape', None) if capture else None
  if tape is not None:
    tape.native_operators.update(native_operators(group))
    # Keep the consumer DSO/compiler and communicator alive with its effect.
    retained = getattr(tape, '_communication_consumer_owners', None)
    if retained is None:
      retained = tape._communication_consumer_owners = {}
    for owner in owners:
      retained[id(owner)] = owner
    tape.native_operators['vecops_comm::consume'] = (load_communication(), group, tuple(retained.values()))
  torch.ops.vecops_comm.consume.default(handle, output, source)
  return output


def load_communication():
  from ._build import load
  return load()

__all__ += ['consume_all_gather']


class PageExchange:
  """Explicit fixed-layout zero-copy AllToAll storage, prepared collectively.

  Construct before capture. Each slot has a peer-major producer Tensor and
  source-major read-only consumer Tensor; their first stride includes page
  padding. Producers must write directly into producer(slot) between begin and
  publish. Consumers must finish before retire. Keep tensors within the region
  lifetime and never mutate consumer storage (the mapping is OS read-only).
  Use private clone/copy for an independently writable or retained result.
  Slots are bounded versions, not dynamically allocated replay state.
  Process-group shutdown ends this region's access permission and removes its
  aliases. All producer/consumer readers must finish before shutdown.
  """
  def __init__(self, block_shape, dtype, *, slots=1, placement='destination', group=None):
    import torch.distributed as dist
    from vecops.graph.capture import current_capture
    if current_capture() is not None:
      raise RuntimeError('prepare PageExchange before graph capture')
    self.group = group or dist.distributed_c10d._get_default_group()
    self.handle, self.producers, self.consumers = backend(self.group).prepare_exchange(
        list(block_shape), dtype, slots, placement)

  def producer(self, slot=0):
    return self.producers[slot]

  def consumer(self, slot=0):
    return self.consumers[slot]

  def _phase(self, phase, slot):
    import torch
    from vecops.graph.capture import current_capture
    capture = current_capture()
    tape = getattr(capture, '_torch_replay_tape', None) if capture else None
    if tape is not None:
      tape.native_operators.update(native_operators(self.group))
      from vecops.graph import mark_retained
      mark_retained((self.producers[slot], self.consumers[slot]), reason='page exchange external shared aliases')
    torch.ops.vecops_comm.exchange.default(self.handle, phase, slot,
        [self.producers[slot], self.consumers[slot]])

  def begin(self, slot=0):
    self._phase(0, slot)

  def publish(self, slot=0):
    self._phase(1, slot)

  def retire(self, slot=0):
    self._phase(2, slot)

  def abort(self):
    """Poison this storage after a failed producer/consumer; wake peer waiters."""
    load_communication().abort_exchange(self.handle)

__all__ += ['PageExchange']
