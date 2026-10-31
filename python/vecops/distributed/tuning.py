# SPDX-FileCopyrightText: 2026 HenryRenYz and vecops contributors
# SPDX-License-Identifier: MIT
"""Cold, empirical route tuning. No measurement or policy search during replay.

Profiles match complete tensor metadata, rank placement, worker count, compiler
binary, boot and configured frequency policy. Unknown/stale cases keep the
platform default. Shared-resource contention is measured with all ranks active,
not estimated by treating nominal link bandwidths as independent capacities.
"""
from __future__ import annotations
import hashlib
import json
import os
from pathlib import Path
import statistics
import time
import warnings


def runtime_signature():
  policy={}
  for path in sorted(Path('/sys/devices/system/cpu/cpufreq').glob('policy*')):
    policy[path.name]={key:(path/key).read_text().strip() for key in
        ('scaling_min_freq','scaling_max_freq','scaling_governor') if (path/key).exists()}
  return {'boot':Path('/proc/sys/kernel/random/boot_id').read_text().strip(), 'frequency_policy':policy,
          'user_tag':os.environ.get('VECOPS_COMM_TUNING_TAG', '')}


def context(platform, members, threads, binary):
  # Every rank compares the same group-wide context, including asymmetric
  # worker counts and per-rank manual frequency tags. Reference-file enablement
  # is checked during bootstrap but is not part of the measured workload.
  return {'platform':platform.fingerprint,
          'members':[{k:v for k,v in member.items() if k not in {'threads','runtime','calibration'}} for member in members],
          'threads':[member.get('threads',threads) for member in members],
          'binary':binary,
          'runtime':[member['runtime'] if 'runtime' in member else runtime_signature() for member in members]}


def load_profile(path, expected):
  data=json.loads(Path(path).expanduser().read_text())
  if data.get('schema_version')!=1:
    raise ValueError('unsupported communication calibration schema')
  if data.get('context')!=expected:
    warnings.warn('vecops communication calibration is stale; retaining platform routing',RuntimeWarning)
    return {},hashlib.sha256(json.dumps(data,sort_keys=True).encode()).hexdigest()
  routes={}
  for record in data.get('cases',[]):
    modes=record['selected']
    if not isinstance(record['key'],str) or any(type(v) is not int or v not in (0,1) for v in modes):
      raise ValueError('invalid calibrated communication route')
    routes[record['key']]=modes
  return routes,hashlib.sha256(json.dumps(data,sort_keys=True).encode()).hexdigest()


def tune_all_gather(source, outputs, *, candidates, group=None, iterations=10,
                    rounds=5, warmup=3, minimum_gain=.05):
  """Measure actual materialized layouts under simultaneous all-rank load.

  Call collectively before capture/seal. Candidates are share/copy bit vectors
  for the configured coarse-to-fine levels. Every candidate is prebound and
  warmed outside timing. Rounds rotate candidate order; choose by median of
  max-rank time, requiring the requested gain above both the default and noise.
  This does not prove an end-to-end model gain: benchmark that before rollout.
  Returns a profile case; write_profile installs the complete calibrated table.
  """
  import torch
  import torch.distributed as dist
  from . import backend
  from vecops.graph.capture import current_capture
  if current_capture() is not None:
    raise RuntimeError('communication tuning must precede graph capture')
  if iterations<1 or rounds<3 or warmup<1 or not 0<minimum_gain<1:
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
    for round in range(rounds):
      for i in [(j+round)%len(candidates) for j in range(len(candidates))]:
        dist.barrier(group=group)
        started=time.perf_counter_ns()
        for _ in range(iterations):torch.ops.vecops_comm.run.default(handles[i],list(outputs),[source])
        value[0]=(time.perf_counter_ns()-started)/1e9/iterations
        dist.all_gather(samples,value,group=group)
        records[i].append(max(x.item() for x in samples))
    medians=list(map(statistics.median,records))
    best=min(range(len(candidates)),key=medians.__getitem__)
    noise=max((max(row)-min(row))/max(statistics.median(row),1e-12) for row in (records[0],records[best]))
    gain=1-medians[best]/medians[0]
    selected=best if gain>max(minimum_gain,noise) else 0
    key=native.plan_key('all_gather',list(outputs),[source])
    record={'key':key,'selected':list(candidates[selected]),'baseline':list(candidates[0]),
            'gain':1-medians[selected]/medians[0],'noise_fraction':noise,
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
  data={'schema_version':1,'context':json.loads(native.tuning_context),'cases':list(cases)}
  if dist.get_rank(group)==0:
    path=Path(path).expanduser().resolve();path.parent.mkdir(parents=True,exist_ok=True)
    temporary=path.with_name(path.name+f'.{os.getpid()}.partial')
    fd=os.open(temporary,os.O_CREAT|os.O_EXCL|os.O_WRONLY,0o600)
    with os.fdopen(fd,'w') as stream:json.dump(data,stream,indent=2);stream.write('\n')
    temporary.replace(path)
  dist.barrier(group=group)
  native.configure_routes({case['key']:case['selected'] for case in cases})
  return data
