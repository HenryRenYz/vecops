# SPDX-FileCopyrightText: 2026 HenryRenYz and vecops contributors
# SPDX-License-Identifier: MIT
"""Compute/collective islands: repeated plans, rebinding and lease readers."""
import os
from pathlib import Path
import pytest


def _worker(rank, size, store, provider):
  os.environ.update(VECOPS_THREAD_POOL=provider, VECOPS_NUM_THREADS='2',
                    VECOPS_TORCH_RESIDENT_TEAM='1', VECOPS_COMM_CONFIG=store+'.toml',
                    OMP_WAIT_POLICY='PASSIVE', GOMP_SPINCOUNT='0')
  if lists := os.environ.get('VECOPS_TEST_CPU_LISTS'):
    import json
    os.sched_setaffinity(0, json.loads(lists)[rank])
  import torch
  import torch.distributed as dist
  import vecops.distributed as vd
  from vecops.graph import GraphCapture
  torch.set_num_threads(2)
  vd.register_backend()
  dist.init_process_group('vecops', init_method='file://'+store,
                          rank=rank, world_size=size,
                          pg_options=vd.Options(timeout_ms=int(os.environ.get('VECOPS_TEST_COMM_TIMEOUT_MS', '15000'))))
  try:
    class Chain(torch.nn.Module):
      def __init__(self):
        super().__init__()
        self.a = torch.empty(size, 17)
        self.b = torch.empty(size, 17)

      def forward(self, x):
        vd.launch('all_gather', list(self.a.unbind()), [x+1])
        self.a.mul_(2)
        # Same cached collective plan, different pointers at BOTH call sites.
        vd.launch('all_gather', list(self.b.unbind()), [x+3])
        return self.a+self.b

    with torch.inference_mode():
      model = Chain()
      source = torch.arange(17, dtype=torch.float32)+rank*100
      capture = GraphCapture('resident-collective-chain')
      capture.capture_torch(model, source)
      executable = capture.compile_torch(native_pointwise=True, borrow_outputs=True,
                                          native_operators=vd.native_operators())
      assert max(executable.allocator_policy['native_island_sizes']) >= 5
      before = vd.statistics()
      for iteration in range(4):
        actual = executable(source+iteration)
        expected = torch.stack([3*(torch.arange(17)+peer*100+iteration)+5 for peer in range(size)])
        assert torch.equal(actual, expected)
      after = vd.statistics()
      assert after['calls']-before['calls'] == 8
      assert after['capacity_growths'] == before['capacity_growths']

      # Ragged/zero traffic also retains strided destination semantics.
      class Ragged(torch.nn.Module):
        def __init__(self):
          super().__init__()
          self.out = [torch.empty((peer+rank)%3, 6)[:, ::2] for peer in range(size)]

        def forward(self, x):
          vd.launch('all_to_all', self.out, list(x))
          return self.out

      sends = [torch.full(((rank+peer)%3, 3), rank*10+peer, dtype=torch.float32) for peer in range(size)]
      ragged = GraphCapture('resident-ragged')
      ragged.capture_torch(Ragged(), sends)
      replay = ragged.compile_torch(native_operators=vd.native_operators())
      for iteration in range(3):
        result = replay([x+iteration for x in sends])
        for peer, x in enumerate(result):
          assert torch.equal(x, torch.full_like(x, peer*10+rank+iteration))

      def sum_and_barrier(x):
        vd.launch('sum', [x], [x])
        vd.launch('barrier', [], [])
        return x
      summed = GraphCapture('resident-sum-and-barrier')
      summed.capture_torch(sum_and_barrier, source.clone())
      reduce_replay = summed.compile_torch(native_operators=vd.native_operators())
      assert max(reduce_replay.allocator_policy['native_island_sizes']) >= 2
      for iteration in range(3):
        expected = sum(torch.arange(17)+peer*100+iteration for peer in range(size))
        assert torch.equal(reduce_replay(source+iteration), expected)

      import vecops.language as T

      @T.jit(out_idx=[1])
      def pack_factory(peers, delay):
        Peers, Delay = peers, delay
        Count, Stride = T.dynamic(lower=0), T.dynamic(lower=0)
        @T.template(source=str(Path(__file__).with_name('ResidentPeerPack.cpp')))
        def kernel(addresses: T.Tensor((Peers,), (1,), T.int64),
                   out: T.Tensor((Peers, Count), (Stride, 1), T.float32)):
          Delay
        return kernel

      pack = pack_factory(size, int(rank == size-1))
      for count in (0, 17):
        sample = torch.arange(count, dtype=torch.float32)+rank*100
        # Materialize the real consumer bridge before graph capture.
        pack.explicit_torch_function()(torch.tensor([sample.data_ptr()]*size), torch.empty(size, count))
        operation, specs = pack.explicit_torch_binding()

        class Borrowed(torch.nn.Module):
          def __init__(self):
            super().__init__()
            self.a, self.b = torch.empty(size, count), torch.empty(size, count)

          def forward(self, x):
            vd.consume_all_gather(x+1, self.a, operation._schema.name, specs, owners=(pack,))
            vd.consume_all_gather(x+3, self.b, operation._schema.name, specs, owners=(pack,))
            return self.a+self.b

        borrowed = GraphCapture('resident-borrowed-consumers')
        borrowed.capture_torch(Borrowed(), sample)
        replay = borrowed.compile_torch(native_pointwise=True, borrow_outputs=True,
                                        native_operators=vd.native_operators())
        assert max(replay.allocator_policy['native_island_sizes']) >= (4 if count == 0 else 5), (rank, count, replay.allocator_policy)
        for iteration in range(3):
          actual = replay(sample+iteration)
          expected = torch.stack([2*(torch.arange(count)+peer*100+iteration)+4 for peer in range(size)])
          assert torch.equal(actual, expected)
      dist.barrier()
      if size == 1:
        vd.backend().abort()
      else:
        # Only the delayed rank fails, on lane 1 reading rank 1's snapshot.
        # Its peers reach the next collective and must observe the abort;
        # its other local lane must leave the cancelled barrier as well.
        failed_source = sample-1000 if rank == 1 else sample
        with pytest.raises(RuntimeError, match='aborted|cancelled|failed|failure'):
          replay(failed_source)
      with pytest.raises(RuntimeError, match='aborted|cancelled|failed'):
        executable(source)
  finally:
    dist.destroy_process_group()


@pytest.mark.parametrize('size', [1, 3])
@pytest.mark.parametrize('provider', ['native', 'torch'])
def test_resident_collectives(size, provider, tmp_path):
  import torch.multiprocessing as mp
  from vecops.distributed._build import load
  load()
  (tmp_path/'store.toml').write_text('[communication]\nearly_snapshot=true\n')
  mp.spawn(_worker, args=(size, str(tmp_path/'store'), provider), nprocs=size, join=True)
