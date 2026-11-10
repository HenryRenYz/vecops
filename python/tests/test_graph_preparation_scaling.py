# SPDX-FileCopyrightText: 2026 HenryRenYz and vecops contributors
# SPDX-License-Identifier: MIT
"""Preparation must not revisit the entire input set per pointwise output."""
import pytest
import torch

from vecops.graph import GraphCapture


@pytest.mark.parametrize('outputs',[8,96])
def test_many_retained_outputs_visit_graph_inputs_linearly(monkeypatch,outputs):
  monkeypatch.setenv('VECOPS_NUM_THREADS','2')
  value=torch.arange(4,dtype=torch.float32)
  capture=GraphCapture('many-retained-pointwise-outputs')
  with torch.inference_mode():
    capture.capture_torch(lambda x:tuple(x+index+1 for index in range(outputs)),value)
    original=torch._C.Graph.inputs
    visits=0
    def inputs(graph):
      nonlocal visits
      for item in original(graph):
        visits+=1
        yield item
    with monkeypatch.context()as probe:
      probe.setattr(torch._C.Graph,'inputs',inputs)
      executable=capture.compile_torch(native_pointwise=True,borrow_outputs=True)
    # Count metadata work, not noisy wall-clock time. Include room for other
    # linear preparation passes while rejecting a full scan per output.
    assert visits<=10*(outputs+1)
    result=executable(value)
    assert len(result)==outputs
    for index,actual in enumerate(result):
      torch.testing.assert_close(actual,value+index+1,rtol=0,atol=0)
