# SPDX-FileCopyrightText: 2026 HenryRenYz and vecops contributors
# SPDX-License-Identifier: MIT
"""Ordinary collective lowering is geometry-, lifetime- and evidence-gated."""
import json
import os
from pathlib import Path

import pytest


def _worker(rank, directory):
  os.environ.update(VECOPS_NUM_THREADS='2', VECOPS_THREAD_POOL='native',
                    VECOPS_SHARED_PAGE_AUDIT=directory+'/audit')
  import torch
  import torch.distributed as dist
  from vecops import distributed as vd
  from vecops.distributed._routing import for_backend
  from vecops.graph import GraphCapture
  from vecops.graph._pointwise import kernel
  from vecops._arm_vector_state import get_vector_state
  torch.set_num_threads(2)
  vd.register_backend()
  dist.init_process_group('vecops', init_method='file://'+directory+'/store', rank=rank, world_size=2,
                          pg_options=vd.Options(timeout_ms=15000))
  try:
    native = vd.backend()
    planner = for_backend(native)
    count = native.payload_page//4

    def neg(shape):
      sample = torch.zeros(shape)
      out = torch.empty_like(sample)
      impl = kernel(6, shape, sample.stride(), 'float32', shape, sample.stride(), 'float32',
                    shape, out.stride(), 'float32', 2, get_vector_state().env_token)
      impl.explicit_torch_function()(sample,sample,out,0.)
      op, specs = impl.explicit_torch_binding()
      return lambda x, y: op(x,x,y,0.,*specs)

    neg_rows = neg((2,count))
    neg_row = neg((count,))
    for kind in ('all_gather','all_to_all','all_gather_into_tensor','all_to_all_single','all_gatherv'):
      gathered = kind.startswith('all_gather')
      def program():
        inp = torch.empty((count,) if gathered else (2,count))
        sender = torch.empty_like(inp)
        receiver = torch.empty((2,count))
        result = torch.empty_like(receiver)
        inp.fill_(rank+10)
        if gathered:
          neg_row(inp,sender)
        else:
          neg_rows(inp,sender)
        outputs = list(receiver.unbind()) if kind in ('all_gather','all_to_all') else [receiver]
        inputs = list(sender.unbind()) if kind == 'all_to_all' else [sender]
        options = {'output_splits':[count,count]} if kind == 'all_gatherv' else {}
        vd.launch(kind,outputs,inputs,**options)
        neg_rows(receiver,result)
        return result

      def compile_graph():
        graph = GraphCapture('ordinary-'+kind)
        graph.capture_torch(program)
        return graph.compile_torch(borrow_outputs=True,native_operators=vd.native_operators())

      planner.shared_profile.clear()
      original = compile_graph()
      assert original.allocator_policy['shared_page_collectives'] == []
      audit = [json.loads(line) for line in Path(f'{directory}/audit.{rank}.jsonl').read_text().splitlines()]
      row = audit[-1]
      assert row['case'] is not None, row
      # Controlled samples exercise compiler selection. The physical ARM
      # benchmark supplies the timings used to qualify a production case.
      planner.shared_profile[row['case']] = {'copy':[100,102,101], 'source':[110,101,90]}
      fallback = compile_graph()
      assert fallback.allocator_policy['shared_page_collectives'] == []
      planner.shared_profile[row['case']] = {'copy':[100,102,101], 'source':[70,72,71],
                                             'destination':[50,51,52]}
      shared = compile_graph()
      expected = torch.arange(10,12,dtype=torch.float32)[:,None].expand(2,count)
      choice, = shared.allocator_policy['shared_page_collectives']
      assert choice['route'] == ('source' if gathered else 'destination')
      assert max(shared.allocator_policy['native_island_sizes']) >= 5
      for _ in range(3):
        assert torch.equal(original(),expected)
        assert torch.equal(fallback(),expected)
        before = vd.statistics()['copy_bytes']
        assert torch.equal(shared(),expected)
        assert vd.statistics()['copy_bytes'] == before
      if kind == 'all_gather':
        for route in ('copy','source'):
          planner.shared_measure = route
          measured = compile_graph()
          assert torch.equal(measured(),expected)
          samples = [json.loads(line) for line in
              Path(f'{directory}/audit.samples.{os.getpid()}.jsonl').read_text().splitlines()]
          assert samples[-1]['route'] == route
          assert samples[-1]['case'] == row['case']
          assert samples[-1]['elapsed_us'] > 0
        planner.shared_measure = None

    neg_tiles = neg((2,2,count))
    def tiled():
      inp = torch.empty((2,2,count))
      results = []
      for step in (0,1000):
        sender = torch.empty_like(inp)
        receiver = torch.empty_like(inp)
        result = torch.empty_like(inp)
        inp.fill_(rank+10+step)
        inp[1].add_(100)
        inp[:,1].add_(200)
        neg_tiles(inp,sender)
        vd.launch('all_to_all',list(receiver.unbind(0)),list(sender.unbind(1)))
        neg_tiles(receiver,result)
        results.append(result)
      return tuple(results)

    def compile_tiled():
      graph = GraphCapture('interleaved-all-to-all')
      graph.capture_torch(tiled)
      return graph.compile_torch(borrow_outputs=True,native_operators=vd.native_operators())

    planner.shared_profile.clear()
    baseline = compile_tiled()
    audit = Path(f'{directory}/audit.{rank}.jsonl')
    cases = {json.loads(line)['case'] for line in audit.read_text().splitlines()[-2:]}
    assert cases and None not in cases
    expected = (torch.arange(10,12)[:,None,None] + torch.arange(2)[None,:,None]*100 +
                rank*200).expand(2,2,count).to(torch.float32)
    assert all(torch.equal(output,expected+step) for output,step in zip(baseline(),(0,1000)))
    for route in ('source','destination'):
      planner.shared_profile.update({case:{'copy':[100,102,101],route:[50,51,52]} for case in cases})
      shared = compile_tiled()
      assert [row['route'] for row in shared.allocator_policy['shared_page_collectives']] == [route,route]
      assert [json.loads(line)['shared_backing'] for line in audit.read_text().splitlines()[-2:]] == [0,0]
      before = vd.statistics()['copy_bytes']
      for _ in range(3):
        assert all(torch.equal(output,expected+step) for output,step in zip(shared(),(0,1000)))
      assert vd.statistics()['copy_bytes'] == before
  finally:
    dist.destroy_process_group()


def test_ordinary_collectives_compile_to_shared_pages(tmp_path):
  from vecops.distributed._build import load
  import torch.multiprocessing as mp
  load()
  mp.spawn(_worker, args=(str(tmp_path),), nprocs=2, join=True)
