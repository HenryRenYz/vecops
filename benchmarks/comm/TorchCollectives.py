# SPDX-FileCopyrightText: 2026 HenryRenYz and vecops contributors
# SPDX-License-Identifier: MIT
"""AF3-shaped CPU transport benchmark, including prebound native execution.

Run with torchrun. Binding is supplied explicitly as JSON CPU lists, and memory
targets come from the independent platform file. All timing excludes allocation,
plan construction, validation and warmup. Other backends use identical tensors.
"""
import argparse
import json
import os
from pathlib import Path
import statistics
import time


def main():
  parser = argparse.ArgumentParser()
  parser.add_argument('--backend', choices=['vecops', 'gloo'], required=True)
  parser.add_argument('--config', required=True)
  parser.add_argument('--cpus', required=True, help='JSON file containing one CPU list per rank')
  parser.add_argument('--tokens', type=int, choices=[128, 768, 1024], default=768)
  parser.add_argument('--iterations', type=int, default=50)
  parser.add_argument('--output', required=True)
  parser.add_argument('--tune-profile', help='cold empirical AllGather route profile; vecops only')
  parser.add_argument('--batch-dim', action='store_true', help='include AF3 singleton batch dimension')
  args = parser.parse_args()
  rank = int(os.environ['RANK'])
  cpus = json.loads(Path(args.cpus).read_text())[rank]
  os.sched_setaffinity(0, cpus)
  os.environ['VECOPS_NUM_THREADS'] = str(len(cpus))
  os.environ['VECOPS_THREAD_POOL_CPU_LIST'] = ','.join(map(str, cpus))
  os.environ['VECOPS_COMM_CONFIG'] = str(Path(args.config).resolve())
  import torch
  import torch.distributed as dist
  from vecops import memory
  from vecops.distributed.config import Platform, current_cpu_node
  torch.set_num_threads(1)
  if args.backend == 'vecops':
    import vecops.distributed as vd
    vd.register_backend()
  dist.init_process_group(args.backend)
  world = dist.get_world_size()
  assert args.tokens % world == 0
  local, channels = args.tokens//world, 128
  system = memory.System()
  _, target = Platform.load(args.config).resolve(current_cpu_node())
  owners = []
  def empty(shape):
    count = 1
    for n in shape:
      count *= n
    owner = system.allocate(count*2, os_numa_id=target if target>=0 else None, large_pages=False, fallback=None)
    owners.append(owner)
    return torch.frombuffer(owner, dtype=torch.bfloat16, count=count).reshape(shape)
  records = []
  calibration = []
  for pattern in ('outgoing', 'incoming', 'exchange'):
    owners.clear()
    if pattern == 'outgoing':
      source = empty((channels, local, args.tokens)).fill_(rank+1)
      target_tensor = empty((channels, args.tokens, args.tokens))
      inputs, outputs = [source], list(target_tensor.split(local, dim=1))
    elif pattern == 'incoming':
      source = empty((channels, args.tokens, local)).fill_(rank+1)
      target_tensor = empty((channels, args.tokens, args.tokens))
      inputs, outputs = [source], list(target_tensor.split(local, dim=2))
    else:
      source = empty((local, args.tokens, channels)).fill_(rank+1)
      target_tensor = empty((world, local, local, channels))
      inputs, outputs = list(source.split(local, dim=1)), list(target_tensor.unbind(0))
    if args.batch_dim and pattern != 'exchange':
      inputs = [x.unsqueeze(0) for x in inputs]
      outputs = [x.unsqueeze(0) for x in outputs]
      source = inputs[0]
    if args.tune_profile and pattern != 'exchange':
      if args.backend != 'vecops':
        raise ValueError('route tuning requires vecops')
      from vecops.distributed.tuning import tune_all_gather
      signature = json.loads(vd.backend().tuning_context)
      platform = Platform.load(args.config)
      levels, defaults = platform.partitions([x['cpu_node'] for x in signature['members']])
      if not levels:
        defaults = [int(platform.communication.get('algorithm') == 'hierarchical')]
      candidates = [defaults, [0]*len(defaults), [1]*len(defaults)]
      candidates += [[int(i==j) for i in range(len(defaults))] for j in range(len(defaults))]
      calibration.append(tune_all_gather(inputs[0],outputs,candidates=candidates,iterations=args.iterations))
    if args.backend == 'vecops':
      handle = vd.backend().prepare('all_to_all' if pattern == 'exchange' else 'all_gather', outputs, inputs, [], [], 0)
      def call():
        torch.ops.vecops_comm.run.default(handle, outputs, inputs)
    elif pattern == 'exchange':
      def call():
        dist.all_to_all(outputs, inputs)
    else:
      def call():
        dist.all_gather(outputs, source)
    for _ in range(8):
      call()
    samples = []
    for _ in range(5):
      dist.barrier()
      begin = time.perf_counter_ns()
      for _ in range(args.iterations):
        call()
      samples.append((time.perf_counter_ns()-begin)/1e9/args.iterations)
    for peer, output in enumerate(outputs):
      assert (output == peer+1).all(), (rank, pattern, peer)
    records.append({'pattern':pattern, 'seconds':samples, 'median':statistics.median(samples)})
  if args.tune_profile:
    from vecops.distributed.tuning import write_profile
    write_profile(args.tune_profile,calibration)
  gathered = [None for _ in range(world)]
  dist.all_gather_object(gathered, {'rank':rank, 'records':records})
  if rank == 0:
    Path(args.output).write_text(json.dumps({'backend':args.backend, 'tokens':args.tokens,
        'threads':len(cpus), 'iterations':args.iterations, 'ranks':gathered}, indent=2)+'\n')
  dist.destroy_process_group()


if __name__ == '__main__':
  main()
