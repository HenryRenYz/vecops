# SPDX-FileCopyrightText: 2026 HenryRenYz and vecops contributors
# SPDX-License-Identifier: MIT
from __future__ import annotations

from pathlib import Path

from vecops import graph, planner


class _Tensor:
  pass


def _metadata(elements: int) -> graph.TensorMetadata:
  return graph.TensorMetadata(
    shape=(elements,),
    strides=(1,),
    dtype="uint8",
    device="cpu",
    element_size=1,
    logical_bytes=elements,
  )


def _machine(fast_capacity: int) -> planner.MachineProfile:
  return planner.MachineProfile(
    fast_capacity=fast_capacity,
    slow_capacity=1 << 20,
    fast_bandwidth_mib_s=100_000,
    slow_bandwidth_mib_s=10_000,
    transfer_bandwidth_mib_s=100_000,
    alignment=64,
    name="test",
  )


def test_all_fit_plan_replicates_read_only_state_and_keeps_output_slow() -> None:
  capture = graph.GraphCapture("all-fit")
  input_value = _Tensor()
  weight = _Tensor()
  output = _Tensor()
  with capture:
    input_id = capture.tensor(
      input_value,
      _metadata(64),
      storage_identity=("input", 1),
      storage_bytes=64,
      kind=graph.TensorKind.input,
      external=True,
    )
    capture.mark_input(input_id)
    capture.tensor(
      weight,
      _metadata(128),
      storage_identity=("weight", 1),
      storage_bytes=128,
      name="weight",
      kind=graph.TensorKind.parameter,
      persistent=True,
      read_only=True,
      external=True,
    )
    capture.tensor(
      output,
      _metadata(128),
      storage_identity=("output", 1),
      storage_bytes=128,
    )
    capture.record_operator(
      "compute",
      inputs=[
        (input_value, graph.AccessKind.read, "input"),
        (weight, graph.AccessKind.read, "weight"),
      ],
      outputs=[output],
    )
    capture.mark_outputs([output])
  execution = capture.finish()
  result = planner.plan_memory(
    execution,
    _machine(256),
    config=planner.PlannerConfig(minimum_fast_bytes=1),
  )
  planner.validate_plan(execution, result)

  input_storage = execution.tensors[input_id].storage_id
  weight_storage = execution.tensors[capture.tensor_id(weight)].storage_id
  output_storage = execution.tensors[capture.tensor_id(output)].storage_id
  assert result.placements[input_storage].tier == planner.MemoryTier.fast
  assert result.placements[input_storage].replicated
  assert result.placements[weight_storage].tier == planner.MemoryTier.fast
  assert result.placements[weight_storage].replicated
  assert result.placements[output_storage].tier == planner.MemoryTier.slow
  assert any(
    action.kind == planner.ActionKind.copy
    and action.storage_id == weight_storage
    and action.source_tier == planner.MemoryTier.slow
    and action.target_tier == planner.MemoryTier.fast
    for action in result.actions
  )

  restored = planner.MemoryPlan.loads(result.dumps())
  planner.validate_plan(execution, restored)
  audit = planner.audit_plan(execution, result)
  assert audit["parameters"]["storages"] == 1
  assert audit["parameters"]["copy_to_fast_storages"] == 1
  assert audit["movement"] == {
    "copy_actions": 2,
    "copy_bytes": 192,
    "unique_copy_storages": 2,
    "duplicate_copy_actions": 0,
    "low_reuse_copy_actions": 2,
    "low_reuse_copy_bytes": 192,
    "copy_back_actions": 0,
    "copy_back_bytes": 0,
  }
  assert audit["unnamed_fast_state_storages"] == []


def test_cost_plan_prefers_more_frequently_accessed_storage() -> None:
  capture = graph.GraphCapture("benefit")
  hot = _Tensor()
  cold = _Tensor()
  with capture:
    for value, name in ((hot, "hot"), (cold, "cold")):
      capture.tensor(
        value,
        _metadata(128),
        storage_identity=(name, 1),
        storage_bytes=128,
        name=name,
        kind=graph.TensorKind.parameter,
        persistent=True,
        read_only=True,
        external=True,
      )
    capture.record_operator(
      "both", inputs=[(hot, "read", "hot"), (cold, "read", "cold")], outputs=[]
    )
    capture.record_operator("hot-again", inputs=[(hot, "read", "hot")], outputs=[])
    capture.record_operator("hot-third", inputs=[(hot, "read", "hot")], outputs=[])
  execution = capture.finish()
  result = planner.plan_memory(
    execution,
    _machine(128),
    config=planner.PlannerConfig(
      minimum_fast_bytes=1,
      prefer_all_fast_when_fit=False,
      score_by_residency=False,
    ),
  )
  hot_storage = execution.tensors[capture.tensor_id(hot)].storage_id
  cold_storage = execution.tensors[capture.tensor_id(cold)].storage_id
  assert result.placements[hot_storage].tier == planner.MemoryTier.fast
  assert result.placements[cold_storage].tier == planner.MemoryTier.slow


def test_physical_packing_reuses_offsets_for_disjoint_lifetimes() -> None:
  capture = graph.GraphCapture("packing")
  first = _Tensor()
  second = _Tensor()
  with capture:
    capture.tensor(
      first,
      _metadata(256),
      storage_identity=("temporary", 1),
      storage_bytes=256,
    )
    capture.record_operator("make-first", inputs=[], outputs=[first])
    capture.record_operator("use-first", inputs=[(first, "read", "input")], outputs=[])
    capture.tensor(
      second,
      _metadata(256),
      storage_identity=("temporary", 2),
      storage_bytes=256,
    )
    capture.record_operator("make-second", inputs=[], outputs=[second])
    capture.record_operator("use-second", inputs=[(second, "read", "input")], outputs=[])
  execution = capture.finish()
  result = planner.plan_memory(
    execution,
    _machine(256),
    config=planner.PlannerConfig(minimum_fast_bytes=1),
  )
  first_storage = execution.tensors[capture.tensor_id(first)].storage_id
  second_storage = execution.tensors[capture.tensor_id(second)].storage_id
  assert result.placements[first_storage].offset == 0
  assert result.placements[second_storage].offset == 0
  assert result.fast_arena_bytes == 256


def test_plan_session_materializes_and_restores_read_only_module_state() -> None:
  torch = __import__("pytest").importorskip("torch")
  import vecops

  class Model(torch.nn.Module):
    def __init__(self):
      super().__init__()
      self.weight = torch.nn.Parameter(torch.arange(16, dtype=torch.float32))

    def forward(self, value):
      return value + self.weight

  model = Model().eval()
  execution = vecops.precompile(
    model,
    torch.ones(16),
    capture_graph=True,
  ).graph
  assert execution is not None
  result = planner.plan_memory(
    execution,
    _machine(4096),
    config=planner.PlannerConfig(minimum_fast_bytes=1),
  )
  original_pointer = model.weight.data_ptr()
  expected = model.weight.detach().clone()
  with planner.PlanSession(
    execution,
    result,
    large_pages=False,
  ) as session:
    stats = session.bind_module_state(model)
    assert stats["copied_storages"] == 1
    assert stats["copied_bytes"] == expected.numel() * expected.element_size()
    assert model.weight.data_ptr() != original_pointer
    torch.testing.assert_close(model.weight, expected)
    assert session.stats()["copied_storages"] == 1
    assert session.stats()["fast_arena_bytes"] == result.fast_arena_bytes
  assert model.weight.data_ptr() == original_pointer
  torch.testing.assert_close(model.weight, expected)


def test_plan_session_replays_explicit_factory_from_shared_arena() -> None:
  torch = __import__("pytest").importorskip("torch")
  import vecops

  class Model(torch.nn.Module):
    def __init__(self):
      super().__init__()
      self.last = None

    def forward(self, value):
      output = planner.empty_like(value)
      output.copy_(value)
      self.last = output
      return output + 1

  model = Model().eval()
  value = torch.arange(16, dtype=torch.float32)
  execution = vecops.precompile(model, value, capture_graph=True).graph
  assert execution is not None
  result = planner.plan_memory(
    execution,
    _machine(4096),
    config=planner.PlannerConfig(
      minimum_fast_bytes=1,
      require_replayable_temporaries=True,
    ),
  )
  factory_event = next(
    event for event in execution.events if event.name == "aten.empty_like.default"
  )
  factory_output_id = factory_event.outputs[0]
  model.last = None
  with planner.PlanSession(
    execution,
    result,
    large_pages=False,
    enable_factory_replay=True,
  ) as session:
    actual = model(value)
    assert model.last is not None
    assert (
      model.last.untyped_storage().nbytes()
      == result.fast_arena_bytes
    )
    assert session.stats()["planned_factory_allocations"] == 1
    assert session.stats()["factory_allocated_bytes"] == value.numel() * value.element_size()
    torch.testing.assert_close(actual, value + 1)
    assert session.stats()["factory_reserved_bytes"] == result.fast_arena_bytes
    assert session.stats()["factory_owner_allocations"] == 1
    assert session.stats()["allocated_slow_arena_bytes"] == 0


def test_plan_session_replays_sampled_loop_from_one_shared_arena() -> None:
  torch = __import__("pytest").importorskip("torch")
  import vecops

  class Model(torch.nn.Module):
    def forward(self, value):
      loop = graph.declare_loop("test_loop", 4)
      iterations = 2 if vecops.is_precompiling() else 4
      result = value.clone()
      for index in range(iterations):
        role = "prologue" if index == 0 else "steady_sample"
        with loop.iteration(index, role=role, inputs=(result,)) as state:
          temporary = planner.empty_like(value)
          temporary.copy_(value + index)
          result = result + temporary
          state["outputs"] = (result,)
      loop.close()
      return result

  model = Model().eval()
  value = torch.arange(16, dtype=torch.float32)
  execution = vecops.precompile(model, value, capture_graph=True).graph
  assert execution is not None
  result = planner.plan_memory(
    execution,
    _machine(4096),
    config=planner.PlannerConfig(
      minimum_fast_bytes=1,
      require_replayable_temporaries=True,
    ),
  )
  expected = model(value)
  with planner.PlanSession(
    execution,
    result,
    large_pages=False,
    strict_factory_replay=True,
    enable_factory_replay=True,
  ) as session:
    actual = model(value)
    torch.testing.assert_close(actual, expected)
    assert session.stats()["planned_factory_allocations"] == 4
    assert session.stats()["factory_fallbacks"] == 0
    assert session.stats()["factory_owner_allocations"] == 1


def test_replayable_policy_rejects_steady_value_retained_past_iteration() -> None:
  capture = graph.GraphCapture("unsafe-loop-reuse")
  values = [_Tensor(), _Tensor()]
  with capture:
    loop = capture.declare_loop("loop", 4)
    for index, value in enumerate(values):
      role = "prologue" if index == 0 else "steady_sample"
      with loop.iteration(index, role=role):
        capture.tensor(
          value,
          _metadata(256),
          storage_identity=("loop", index),
          storage_bytes=256,
        )
        capture.record_operator("factory", inputs=[], outputs=[value])
        if index == 1:
          capture.mark_retained(value, reason="test")
    loop.close()
  execution = capture.finish()
  result = planner.plan_memory(
    execution,
    _machine(1024),
    config=planner.PlannerConfig(
      minimum_fast_bytes=1,
      require_replayable_temporaries=True,
    ),
  )
  steady_storage = execution.tensors[capture.tensor_id(values[1])].storage_id
  assert result.placements[steady_storage].tier == planner.MemoryTier.slow


def test_machine_profile_can_be_discovered_from_memory_system_shape() -> None:
  class System:
    def current_cpu_domain(self):
      return 3

    def tiers(self, domain, *, objective):
      assert domain == 3
      if objective == "bandwidth":
        return [
          {"rank": 0, "targets": [9], "value": 250_000},
          {"rank": 1, "targets": [3], "value": 25_000},
        ]
      return [{"rank": 0, "targets": [3], "value": 10}]

    def topology(self):
      return {"backend": "test-hwloc"}

  profile = planner.profile_from_system(
    System(),
    fast_capacity=3 << 30,
    slow_capacity=64 << 30,
  )
  assert profile.fast_bandwidth_mib_s == 250_000
  assert profile.slow_bandwidth_mib_s == 25_000
  assert profile.transfer_bandwidth_mib_s == 25_000
  assert profile.attributes["fast_targets"] == [9]
  assert profile.attributes["slow_targets"] == [3]
  assert profile.attributes["bandwidth_overridden"] is False

  calibrated = planner.profile_from_system(
    System(),
    fast_capacity=3 << 30,
    slow_capacity=64 << 30,
    fast_bandwidth_mib_s=295_578,
    slow_bandwidth_mib_s=153_324,
    transfer_bandwidth_mib_s=148_797,
  )
  assert calibrated.fast_bandwidth_mib_s == 295_578
  assert calibrated.slow_bandwidth_mib_s == 153_324
  assert calibrated.transfer_bandwidth_mib_s == 148_797
  assert calibrated.attributes["discovered_fast_bandwidth_mib_s"] == 250_000
  assert calibrated.attributes["discovered_slow_bandwidth_mib_s"] == 25_000
  assert calibrated.attributes["bandwidth_overridden"] is True


def test_plan_cache_rejects_corrupt_and_stale_entries(tmp_path: Path) -> None:
  capture = graph.GraphCapture("cache")
  value = _Tensor()
  with capture:
    capture.tensor(
      value,
      _metadata(64),
      storage_identity=("cache", 1),
      storage_bytes=64,
    )
    capture.record_operator("make", inputs=[], outputs=[value])
  execution = capture.finish()
  path = tmp_path / "plan.json"
  config = planner.PlannerConfig(minimum_fast_bytes=1)

  first, hit = planner.load_or_plan(
    execution, _machine(128), path, config=config
  )
  assert not hit
  second, hit = planner.load_or_plan(
    execution, _machine(128), path, config=config
  )
  assert hit
  assert second.to_dict() == first.to_dict()

  path.write_text("not json", encoding="utf-8")
  repaired, hit = planner.load_or_plan(
    execution, _machine(128), path, config=config
  )
  assert not hit
  planner.validate_plan(execution, repaired)
