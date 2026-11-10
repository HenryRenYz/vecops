# SPDX-FileCopyrightText: 2026 HenryRenYz and vecops contributors
# SPDX-License-Identifier: MIT
"""FakeTensor must not decompose raw native communication into CPU execution."""
import pytest
import torch


def test_metadata_collectives_do_not_resolve_or_execute_real_plans():
  from vecops.distributed._build import load
  from vecops.graph import GraphCapture
  from vecops.graph._metadata import metadata_mode
  load()
  names = ('run', 'consume', 'exchange')
  for name in names:
    qualified = 'vecops_comm::' + name
    assert torch._C._dispatch_has_kernel_for_dispatch_key(qualified, 'Meta')
    assert not torch._C._dispatch_has_kernel_for_dispatch_key(
        qualified, 'CompositeImplicitAutograd')
  # An unknown plan is deliberate. Any accidental CPU submission fails before
  # touching fake addresses, rather than turning this regression into a crash.
  invalid = (1 << 62) - 1
  def communicate(x):
    out = torch.empty_like(x)
    torch.ops.vecops_comm.run.default(invalid, [out], [x])
    torch.ops.vecops_comm.run.default(invalid, [], [])  # Barrier: no device-bearing tensor.
    torch.ops.vecops_comm.consume.default(invalid, out, x)
    torch.ops.vecops_comm.exchange.default(invalid, 0, 0, [out])
    return out
  capture = GraphCapture('metadata-communication')
  with torch.inference_mode(), metadata_mode():
    x = torch.empty((3, 7))
    capture.capture_torch_metadata(communicate, x)
  observed = {instruction.function._schema.name
              for instruction in capture._torch_replay_tape.instructions}
  assert {'vecops_comm::' + name for name in names} <= observed
  # Zero-tensor control operations still have a real fallback. The invalid
  # handle must therefore be checked outside metadata recording.
  with pytest.raises(RuntimeError, match='expired prepared communication'):
    torch.ops.vecops_comm.run.default(invalid, [], [])
