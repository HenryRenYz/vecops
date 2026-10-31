# SPDX-FileCopyrightText: 2026 HenryRenYz and vecops contributors
# SPDX-License-Identifier: MIT
"""Scoped static tensors must coexist with retained dynamic leases."""
import pytest

torch = pytest.importorskip('torch')
import vecops
from vecops import graph, planner


class _System:
  def current_cpu_domain(self):
    return 0

  def allocate(self, size, **kwargs):
    return bytearray(size)


@pytest.mark.parametrize('wrapped_region', [False, True])
def test_scoped_replay_keeps_retained_call_ordinals_and_escaping_output(wrapped_region):
  def buffers(value, index):
    temporary = planner.empty_like(value)
    temporary.copy_(value + index)
    retained = planner.empty_like(value)
    retained.fill_(index)
    graph.mark_retained(retained, reason='cache spanning iterations')
    return temporary, retained

  if wrapped_region:
    buffers = graph.wrap_region(buffers, name='buffer_region')
  class Model(torch.nn.Module):
    def forward(self, value):
      self.held = []
      self.ids = []
      graph.enter_scope('other')
      other = planner.empty_like(value)
      other.copy_(value)
      result = other + 1
      graph.leave_scope()
      graph.enter_scope('diffusion')
      loop = graph.declare_loop('steps', 4)
      iterations = 2 if vecops.is_precompiling() else 4
      for index in range(iterations):
        role = 'prologue' if index == 0 else 'steady_sample'
        with loop.iteration(index, role=role, inputs=(result,)) as state:
          temporary, retained = buffers(value, index)
          self.held.append(retained)
          self.ids.append(id(temporary))
          result = result + temporary + retained
          state['outputs'] = (result,)
      loop.close()
      escaped = planner.empty_like(value)
      escaped.copy_(result)
      graph.leave_scope()
      return escaped

  model = Model().eval()
  value = torch.arange(16, dtype=torch.float32)
  execution = vecops.precompile(model, value, capture_graph=True).graph
  machine = planner.MachineProfile(
      fast_capacity=4096, slow_capacity=1 << 20,
      fast_bandwidth_mib_s=100, slow_bandwidth_mib_s=10,
      transfer_bandwidth_mib_s=100, alignment=64)
  plan = planner.plan_memory(execution, machine, config=planner.PlannerConfig(
      minimum_fast_bytes=1, require_replayable_temporaries=True,
      temporary_scope='diffusion', only_explicit_factory_temporaries=True))
  expected = model(value).clone()
  with planner.PlanSession(execution, plan, system=_System(), large_pages=False,
                           enable_factory_replay=True,
                           enable_dynamic_factories=True,
                           strict_factory_replay=True) as session:
    actual = model(value)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    for i, retained in enumerate(model.held):
      assert torch.all(retained == i)
    assert len(set(model.ids[1:])) == 1
    assert session.stats()['planned_factory_allocations'] == 4
    assert session.stats()['factory_fallbacks'] == 0
    assert session.stats()['dynamic_allocations'] >= 6
    assert session._factory_cursors == {}
  torch.testing.assert_close(actual, expected, rtol=0, atol=0)
  for i, retained in enumerate(model.held):
    assert torch.all(retained == i)


def test_sampled_loop_invariant_is_live_after_its_last_observed_read():
  class Model(torch.nn.Module):
    def forward(self, value):
      graph.enter_scope('diffusion')
      invariant = planner.empty_like(value)
      invariant.copy_(value)
      result = torch.zeros_like(value)
      loop = graph.declare_loop('denoise', 4)
      for index in range(2 if vecops.is_precompiling() else 4):
        role = 'prologue' if index == 0 else 'steady_sample'
        with loop.iteration(index, role=role, inputs=(result,)) as state:
          temporary = planner.empty_like(value)
          temporary.copy_(invariant)
          # The sampled invariant's last read is above. Its slot must not be
          # reused here: the next real iteration still reads that invariant.
          late = planner.empty_like(value)
          late.fill_(100 + index)
          result = result + temporary + late
          state['outputs'] = (result,)
      loop.close()
      graph.leave_scope()
      return result

  model = Model().eval()
  value = torch.arange(16, dtype=torch.float32)
  execution = vecops.precompile(model, value, capture_graph=True).graph
  machine = planner.MachineProfile(
      fast_capacity=4096, slow_capacity=1 << 20,
      fast_bandwidth_mib_s=100, slow_bandwidth_mib_s=10,
      transfer_bandwidth_mib_s=100, alignment=64)
  plan = planner.plan_memory(execution, machine, config=planner.PlannerConfig(
      minimum_fast_bytes=1, require_replayable_temporaries=True,
      temporary_scope='diffusion', only_explicit_factory_temporaries=True))
  factory = next(event for event in execution.events if event.attributes.get('explicit_factory'))
  storage_id = execution.tensors[factory.outputs[0]].storage_id
  loop_end = next(e.index for e in execution.events if e.id == execution.loops[0].end_event)
  assert plan.placements[storage_id].end_event >= loop_end
  expected = model(value).clone()
  with planner.PlanSession(execution, plan, system=_System(), large_pages=False,
                           enable_factory_replay=True, enable_dynamic_factories=True,
                           strict_factory_replay=True) as session:
    torch.testing.assert_close(model(value), expected, rtol=0, atol=0)
    assert session.stats()['factory_fallbacks'] == 0
  import copy
  stale = copy.deepcopy(plan)
  stale.placements[storage_id].end_event = execution.storages[storage_id].lifetime_end
  with pytest.raises(ValueError, match='loop invariant'):
    planner.validate_plan(execution, stale)
