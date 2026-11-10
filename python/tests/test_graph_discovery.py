# SPDX-FileCopyrightText: 2026 HenryRenYz and vecops contributors
# SPDX-License-Identifier: MIT
"""Fake discovery shares the real selector and eliminates later kernel builds."""

import pytest
import torch
import vecops
from torch._subclasses.fake_tensor import FakeTensor, FakeTensorMode
from vecops.graph import GraphCapture, discover_pointwise_requests


def region(x):
  return (x.square() + 1).to(torch.bfloat16).float() * 2


@pytest.mark.parametrize("borrow", [False, True])
def test_fake_region_collects_without_executing_or_binding_storage(borrow):
  x = torch.ones(3, 64)
  with vecops.BuildSession(reuse_sdk=False) as build, FakeTensorMode() as mode:
    fake = mode.from_tensor(x)
    with vecops.collect_compile_requests() as collector:
      collector.discover_graph_artifacts = True
      result = discover_pointwise_requests(region, fake, borrow_outputs=borrow, include_grid_seed=True)
    assert isinstance(result, FakeTensor)
    assert result.shape == x.shape
    assert collector.graph_kernels
    assert collector.requests
    assert any(kernel.source.name == "_IntScalar.cpp" for kernel in collector.graph_kernels.values())
    # Earlier tests or invocations may already have prepared an interned
    # kernel. Discovery must not initiate builds, irrespective of that state.
    assert not any(event['kind'] in {'native', 'bridges', 'sdk', 'graph-executor'}
                   for event in build.events)


@pytest.mark.parametrize("memory_graph", [False, True])
def test_bulk_precompile_includes_pointwise_before_real_graph_binding(monkeypatch, memory_graph):
  monkeypatch.setenv("VECOPS_THREAD_POOL", "native")
  x = torch.arange(192, dtype=torch.float32).reshape(3, 64) / 193
  with vecops.BuildSession(parallelism=4) as build, torch.inference_mode():
    result = vecops.precompile(lambda x: discover_pointwise_requests(region, x, borrow_outputs=True), x, parallelism=4, capture_graph=memory_graph)
    assert result.collected >= 4
    if memory_graph:
      result.graph.validate()
      # Factory elaboration must not add compiler/backend bookkeeping kernels
      # to the model's semantic memory program.
      unexpected = [event.name for event in result.graph.events if event.kind == 'operator' and not event.name.startswith('aten.')]
      assert not unexpected, unexpected
    # Capture/lowering timers are diagnostic events, not artifact builds.
    def build_events():
      return [event for event in build.events if event['kind'] in {'native', 'bridges', 'sdk'}]
    frontiers = list(build_events())
    capture = GraphCapture("discovered-pointwise")
    capture.capture_torch(region, x)
    executable = capture.compile_torch(native_pointwise=True, borrow_outputs=True)
    assert build_events() == frontiers
    torch.testing.assert_close(executable(x), region(x), rtol=0, atol=0)
