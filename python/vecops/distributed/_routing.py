# SPDX-FileCopyrightText: 2026 HenryRenYz and vecops contributors
# SPDX-License-Identifier: MIT
"""Cold graph-placement handshake; repeated native replay never enters Python."""
from dataclasses import asdict
import json
import hashlib
from pathlib import Path
import statistics

_registry={}
_binary_hashes={}


def consumer_contract(operator,source,output,specs):
  """Exact consumer identity, including the executed DSO, without addresses."""
  from vecops import _schema_bridge
  registration=_schema_bridge._registrations.get(tuple(operator.split('::',1)))
  if registration is None or registration.library_path is None:
    return None
  path=Path(registration.library_path).resolve()
  stat=path.stat();receipt=(str(path),stat.st_ino,stat.st_size,stat.st_mtime_ns)
  if receipt not in _binary_hashes:
    with path.open('rb') as stream:
      _binary_hashes[receipt]=hashlib.file_digest(stream,'sha256').hexdigest()
  def metadata(tensor):
    return dict(shape=list(tensor.shape),strides=list(tensor.stride()),dtype=str(tensor.dtype),device=str(tensor.device.type))
  # Meta tensors stand for CPU payloads during graph preparation.
  result=dict(operator=operator,specs=list(specs),source=metadata(source),output=metadata(output),
              binary_sha256=_binary_hashes[receipt])
  result['source']['device']=result['output']['device']='cpu'
  return result


def consumer_profile_key(rows):
  """Publisher input copy is identical for all routes and excluded explicitly.

  Actual consumer identity, output placement, per-rank byte counts and all-rank
  runtime context remain exact. Unknown output placement cannot certify a case.
  """
  if not rows or any(row.get('consumer') is None or not row.get('outputs') or
         any(not endpoint for endpoint in row['outputs']) for row in rows):
    return None
  contract=[{key:row[key] for key in ('kind','bytes','outputs','consumer')} for row in rows]
  return 'consumer/'+hashlib.sha256(json.dumps(contract,sort_keys=True,separators=(',',':')).encode()).hexdigest()


def install_profiles(backend,routes):
  """Separate materialization profiles from exact borrowed-consumer evidence."""
  backend.configure_routes({key:value for key,value in routes.items() if not key.startswith('consumer/')})
  planner=for_backend(backend)
  if planner is not None:
    planner.consumer_routes={key:value for key,value in routes.items() if key.startswith('consumer/')}
    planner._cache.clear()


def for_backend(backend):
  # c10d may reconstruct the Python wrapper around its intrusive C++ Backend.
  # Neither Python object identity nor wrapper attributes are lifetime receipts.
  for identity,planner in list(_registry.items()):
    if not planner.lifetime.alive:del _registry[identity]
  return _registry.get(backend._routing_id)


class RoutePlanner:
  def __init__(self, backend, compiler, store, rank, size, policy, synchronization):
    self.lifetime=backend._routing_lifetime
    self.compiler,self.store,self.rank,self.size=compiler,store,rank,size
    self.policy,self.synchronization=policy,synchronization
    self.ordinal=0
    self.records=[]
    self._cache={}
    self.consumer_routes={}

  def configure(self, native, *, kind, inputs, outputs, input_bytes, consumer=None, previous=None):
    # Explicit per-operation routes and applicable empirical profiles already
    # selected by Backend.prepare are authoritative user performance policy.
    if native.route_modes and (previous is None or previous.get('source')=='explicit-or-calibrated'):
      return {'source':'explicit-or-calibrated','modes':list(native.route_modes)}
    if self.policy.get('algorithm','auto')=='flat' or not self.compiler.topology.partitions:
      return {'source':'flat','quality':self.compiler.topology.quality,
              'missing_facts':list(self.compiler.topology.missing_facts)}
    ordinal=self.ordinal;self.ordinal+=1
    local={'kind':kind,'bytes':input_bytes,'inputs':inputs,'outputs':outputs,'consumer':consumer}
    prefix=f'vecops/route-compile/{ordinal}/'
    self.store.set(prefix+str(self.rank),json.dumps(local,sort_keys=True))
    rows=[json.loads(self.store.get(prefix+str(rank)).decode()) for rank in range(self.size)]
    if len({row['kind'] for row in rows})!=1:
      raise ValueError('collective placement compilation order differs between ranks')
    key=json.dumps(rows,sort_keys=True)
    if key not in self._cache:
      if max((row['bytes'] for row in rows),default=0)<self.policy.get('relay_threshold_bytes',1<<20):
        result={'source':'policy-threshold','quality':'reference',
                'modes':[0]*len(self.compiler.topology.partitions)}
      elif kind=='borrowed-consumer' and (profile_key:=consumer_profile_key(rows)) in self.consumer_routes:
        result={'source':'empirical-consumer','quality':'measured',
                'modes':list(self.consumer_routes[profile_key]),'profile_key':profile_key,
                'cancelled_common_terms':['publisher input copy (same for every measured route)']}
      elif kind=='borrowed-consumer':
        # Its fused consumer may have private writes/compute overlapping loads.
        # A plain AllGather-copy estimate cannot certify those differences.
        result={'source':'flat','quality':'incomplete',
                'missing_facts':['fused consumer access/compute schedule is not described'],
                'modes':[0]*len(self.compiler.topology.partitions)}
      else:
        sizes=[row['bytes'] for row in rows]
        source=[row['inputs'][0] if len(row['inputs'])==1 else None for row in rows]
        destinations=[]
        for row in rows:
          endpoints=row['outputs']
          if kind=='all_gather' and len(endpoints)==self.size:
            destinations.append(endpoints)
          elif len(endpoints)==1 and endpoints[0] is not None and len(endpoints[0])==1:
            destinations.append([endpoints[0]]*self.size)
          else:
            destinations.append([None]*self.size)
        best,candidates=self.compiler.select(sizes,input_pools=source,output_pools=destinations,
            minimum_gain=self.policy.get('minimum_gain',.05))
        result={'source':'resource-model','selected':asdict(best),
                'candidates':[asdict(candidate) for candidate in candidates],
                'modes':list(best.modes),'synchronization':self.synchronization}
      self._cache[key]=result
    result=self._cache[key]
    current=list(native.route_modes) or [0]*len(result['modes'])
    if current!=result['modes']:
      native.configure_route(result['modes'])
    self.records.append(result)
    return result


def install(backend,compiler,store,rank,size,policy):
  synchronization=None
  if compiler.topology.partitions and compiler.stage_us is None and policy.get('algorithm','auto')=='auto':
    warmup,rounds=policy.get('sync_warmup',3),policy.get('sync_rounds',7)
    samples=backend.measure_barriers(warmup,rounds)
    prefix='vecops/route-synchronization/'
    store.set(prefix+str(rank),json.dumps(samples))
    rows=[json.loads(store.get(prefix+str(peer)).decode()) for peer in range(size)]
    maxima=[max(row[index] for row in rows) for index in range(rounds)]
    compiler.stage_us=statistics.median(maxima)
    synchronization={'source':'cold-native-barrier','unit':'us','warmup':warmup,
                     'max_rank_samples':maxima,'estimate_us':compiler.stage_us}
  elif compiler.stage_us is not None:
    synchronization={'source':'policy.stage_us','unit':'us','estimate_us':compiler.stage_us}
  for_backend(backend)  # prune stopped groups before retaining another cold store
  _registry[backend._routing_id]=RoutePlanner(backend,compiler,store,rank,size,policy,synchronization)
