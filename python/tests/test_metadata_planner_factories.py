# SPDX-FileCopyrightText: 2026 HenryRenYz and vecops contributors
# SPDX-License-Identifier: MIT
"""Metadata factories must not lease real planner storage or pin snapshots."""

import pytest
import torch
from torch._subclasses.fake_tensor import is_fake

from vecops.graph import GraphCapture
from vecops.graph._metadata import bind_metadata, metadata_mode
from test_graph_resource_program import runtime as resource_runtime


@pytest.mark.parametrize('like', [False, True])
def test_metadata_factories_record_dispatcher_nodes_without_physical_storage(resource_runtime, like):
  source = torch.arange(12.).reshape(3, 4)
  capture = GraphCapture('metadata-planned-factory')
  def fn(x):
    result = torch.empty_like(x) if like else torch.empty(x.shape, dtype=x.dtype)
    result.copy_(x)
    return result + 2
  with torch.inference_mode(), metadata_mode() as mode:
    fake = mode.from_tensor(source)
    assert is_fake(fn(fake))
    assert is_fake(capture.capture_torch_metadata(fn, fake))
  names = [i.function._schema.name for i in capture._torch_replay_tape.instructions]
  assert 'aten::empty_like' in names if like else 'aten::empty' in names
  bind_metadata(capture, (source,))
  executable = capture.compile_torch(runtime=resource_runtime,native_pointwise=True, native_donation=True)
  for x in (source, source + 5):
    torch.testing.assert_close(executable(x), x + 2, rtol=0, atol=0)


def test_direct_torch_factories_keep_storage_lifetimes_and_reuse_resource_offsets(resource_runtime):
  from vecops.memory import native
  from vecops.planner import GraphMemoryCompiler, requirements_from_graph
  source = torch.arange(8.,device='cpu')
  capture = GraphCapture('direct-torch-factory-reuse')
  def fn(x):
    first = torch.empty_like(x)
    first.copy_(x)
    left = first.sum()
    second = torch.empty_like(x)
    second.copy_(x+1)
    return left+second.sum()
  with torch.inference_mode(), metadata_mode() as mode:
    capture.capture_torch_metadata(fn,mode.from_tensor(source))
  execution = capture._torch_execution_graph
  factories = [instruction.result.id for instruction in capture._torch_replay_tape.instructions
               if instruction.function._schema.name == 'aten::empty_like']
  assert len(factories) == 2
  storages = [execution.storages[execution.tensors[key].storage_id] for key in factories]
  assert storages[0].id != storages[1].id
  assert storages[0].lifetime_end < storages[1].lifetime_start
  view = resource_runtime.view(resource_runtime.current_scope())
  requirements = [item for item in requirements_from_graph(execution) if item.allocated]
  program = GraphMemoryCompiler(resource_runtime,view,resource_runtime.snapshot(view),
      native_module=native).compile(requirements)
  bindings = [program.bindings[storage.id] for storage in storages]
  assert (bindings[0].slot,bindings[0].view_offset_bytes) == (bindings[1].slot,bindings[1].view_offset_bytes)
  assert sum(root.bytes for root in program.memory.backings) == 4096
  bind_metadata(capture,(source,))
  executable = capture.compile_torch(runtime=resource_runtime,memory_program=program,native_pointwise=False)
  assert executable.planned_factory_count == 2
  for shift in (0,7):
    value = source+shift
    torch.testing.assert_close(executable(value),value.sum()*2+value.numel(),rtol=0,atol=0)
