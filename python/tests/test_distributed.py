# SPDX-FileCopyrightText: 2026 HenryRenYz and vecops contributors
# SPDX-License-Identifier: MIT
"""Real multi-process tests: strided/ragged traffic, queue ordering and replay."""
import os
from pathlib import Path
import pytest


def _worker(rank, size, rendezvous):
  os.environ['VECOPS_NUM_THREADS'] = '2'
  if lists := os.environ.get('VECOPS_TEST_CPU_LISTS'):
    import json
    os.sched_setaffinity(0, json.loads(lists)[rank])
  import torch
  import torch.distributed as dist
  import vecops.distributed as vd
  torch.set_num_threads(2)
  vd.register_backend()
  dist.init_process_group('vecops', init_method='file://'+rendezvous,
                          rank=rank, world_size=size)
  try:
    for dtype in (torch.uint8, torch.float32, torch.bfloat16, torch.int64):
      # Ragged, strided producer and strided destination with untouched padding.
      source = (torch.arange(3*(rank+1)).reshape(3,rank+1).t()+rank*70).to(dtype)
      backing = [torch.full((r+1, 6), 99, dtype=dtype) for r in range(size)]
      outputs = [x[:, ::2] for x in backing]
      dist.all_gather(outputs, source)
      for r, out in enumerate(outputs):
        expected=(torch.arange(3*(r+1)).reshape(3,r+1).t()+r*70).to(dtype)
        assert torch.equal(out, expected)
        assert (backing[r][:, 1::2] == 99).all()
    sends = [torch.full(((rank+r)%3, 3), 10*rank+r, dtype=torch.float32) for r in range(size)]
    receives = [torch.empty(((r+rank)%3, 6))[:, ::2] for r in range(size)]
    dist.all_to_all(receives, sends)
    for r, result in enumerate(receives):
      assert torch.equal(result, torch.full_like(result, 10*r+rank))
    value = torch.tensor([rank+1.])
    one, two = torch.empty(size), torch.empty(size)
    first = dist.all_gather_into_tensor(one, value, async_op=True)
    second = dist.all_gather_into_tensor(two, value, async_op=True)
    second.wait()  # Must progress the earlier submission first.
    first.wait()
    assert torch.equal(one, torch.arange(1., size+1))
    assert torch.equal(one, two)
    dist.all_reduce(value)
    assert value.item() == size*(size+1)/2
    b = torch.tensor([rank], dtype=torch.int64)
    dist.broadcast(b, src=size-1)
    assert b.item() == size-1
    large=torch.arange(300003,dtype=torch.float32)+rank*1000000
    joined=torch.empty(size*large.numel())
    dist.all_gather_into_tensor(joined,large)
    for peer,part in enumerate(joined.view(size,-1)):
      assert torch.equal(part,torch.arange(300003,dtype=torch.float32)+peer*1000000)
    # Native prepared node, varying pointers but fixed metadata.
    handle = vd.backend().prepare('all_gather_into_tensor', [one], [value], [], [], 0)
    torch.ops.vecops_comm.run.default(handle, [one], [value])
    from vecops.graph import GraphCapture
    class Gather(torch.nn.Module):
      def forward(self, x):
        result = torch.empty((size,), dtype=x.dtype)
        return vd.launch('all_gather_into_tensor', [result], [x], output=result)
    capture = GraphCapture('communication')
    capture.capture_torch(Gather(), value)
    executable = capture.compile_torch(native_operators=vd.native_operators())
    for i in range(2):
      result = executable(torch.tensor([rank+i+1.]))
      assert torch.equal(result, torch.arange(1., size+1)+i)
    dist.barrier()  # Prepare its descriptor before freezing the communicator.
    vd.seal()
    before = vd.statistics()
    for i in range(3):
      x = torch.tensor([rank+i+1.])
      torch.ops.vecops_comm.run.default(handle, [one], [x])
      assert torch.equal(one, torch.arange(1., size+1)+i)
    after = vd.statistics()
    assert after['capacity_growths'] == before['capacity_growths']
    assert after['plans'] == before['plans']
    dist.barrier()
  finally:
    dist.destroy_process_group()


@pytest.mark.parametrize('size', [1, 3])
def test_real_collectives(tmp_path, size):
  import torch.multiprocessing as mp
  from vecops.distributed._build import load
  load()  # Build once before spawning; workers use the content-addressed DSO.
  mp.spawn(_worker, args=(size, str(tmp_path/'store')), nprocs=size, join=True)


def test_platform_validation(tmp_path):
  from vecops.distributed import Platform
  path = tmp_path/'platform.toml'
  path.write_text('''[topology]
schema_version=1
[[topology.domains]]
id="s0"
kind="socket"
parent="host0"
[[topology.domains]]
id="d0"
kind="die"
parent="s0"
numa_nodes=[0]
[[memory.pools]]
id="h0"
kind="hbm"
owner="numa:0"
os_nodes=[16]
''')
  assert Platform.load(path).resolve(0) == ('d0', 16)
  path.write_text(path.read_text().replace('parent="s0"', 'parent="d0"'))
  with pytest.raises(ValueError, match='cyclic'):
    Platform.load(path)


@pytest.mark.parametrize('policy,match', [
  ('chunk_bytes=0', 'positive integer'),
  ('algorithm="borrowed"', 'unknown communication algorithm'),
  ('direction="push"', 'direction overrides are unsupported'),
  ('unknown_switch=true', 'unsupported communication policy fields'),
])
def test_policy_rejects_unimplemented_or_invalid_options(tmp_path, policy, match):
  from vecops.distributed import Platform
  path=tmp_path/'invalid.toml'
  path.write_text('[communication]\n'+policy+'\n')
  with pytest.raises(ValueError, match=match):
    Platform.load(path)
