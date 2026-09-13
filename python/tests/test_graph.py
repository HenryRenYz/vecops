from __future__ import annotations

import gc

from vecops import graph


class _Tensor:
  pass


def _metadata(shape=(8, 8)):
  return graph.TensorMetadata(
    shape=shape,
    strides=(shape[1], 1),
    dtype="float32",
    device="cpu",
    element_size=4,
    logical_bytes=shape[0] * shape[1] * 4,
  )


def test_framework_neutral_graph_tracks_alias_storage_lifetime_and_json() -> None:
  capture = graph.GraphCapture("manual")
  source = _Tensor()
  produced = _Tensor()
  view = _Tensor()

  with capture:
    source_id = capture.tensor(
      source,
      _metadata(),
      storage_identity=("manual", 1),
      storage_bytes=256,
      kind=graph.TensorKind.input,
      external=True,
    )
    capture.mark_input(source_id)
    capture.tensor(
      produced,
      _metadata(),
      storage_identity=("manual", 2),
      storage_bytes=256,
    )
    capture.record_operator(
      "make",
      inputs=[(source, graph.AccessKind.read, "source")],
      outputs=[produced],
    )
    capture.tensor(
      view,
      _metadata((4, 8)),
      storage_identity=("manual", 2),
      storage_bytes=256,
    )
    capture.record_operator(
      "view",
      inputs=[(produced, graph.AccessKind.read, "input")],
      outputs=[view],
    )
    capture.record_operator(
      "mutate",
      inputs=[(view, graph.AccessKind.read_write, "view")],
      outputs=[view],
    )
    graph.mark_retained(produced, reason="test cache")
    capture.mark_outputs([view])

  result = capture.finish()
  result.validate()
  assert (
    result.tensors[capture.tensor_id(produced)].storage_id
    == result.tensors[capture.tensor_id(view)].storage_id
  )
  assert result.tensors[capture.tensor_id(view)].alias_of == capture.tensor_id(produced)
  storage = result.storages[result.tensors[capture.tensor_id(view)].storage_id]
  assert storage.lifetime_start == result.tensors[capture.tensor_id(produced)].lifetime_start
  assert storage.lifetime_end == result.events[-1].index
  assert result.tensors[capture.tensor_id(produced)].annotations["retained"] == "test cache"
  assert (
    result.tensors[capture.tensor_id(produced)].lifetime_end
    == result.events[-1].index
  )
  assert any(
    use.version_after == 1
    for event in result.events
    for use in event.uses
    if event.name == "mutate"
  )

  restored = graph.ExecutionGraph.loads(result.dumps())
  assert restored.to_dict() == result.to_dict()


def test_loop_trace_preserves_declared_count_and_sampled_iterations() -> None:
  capture = graph.GraphCapture("loops")
  source = _Tensor()
  with capture:
    capture.tensor(
      source,
      _metadata(),
      storage_identity=("loop", 1),
      storage_bytes=256,
      external=True,
    )
    loop = graph.declare_loop("layers", 48, attributes={"reverse": False})
    with loop.iteration(0, role="prologue", inputs=[source]) as state:
      capture.record_marker("sample", "first")
      state["outputs"] = [source]
    with loop.iteration(1, role="steady_sample", inputs=[source]) as state:
      capture.record_marker("sample", "steady")
      state["outputs"] = [source]
    loop.close()
  result = capture.finish()

  assert len(result.loops) == 1
  assert result.loops[0].trip_count == 48
  assert result.loops[0].observed_iterations == [0, 1]
  assert [sample.role for sample in result.loops[0].samples] == ["prologue", "steady_sample"]
  assert all(sample.begin_event and sample.end_event for sample in result.loops[0].samples)
  assert result.loops[0].begin_event is not None
  assert result.loops[0].end_event is not None
  assert result.loops[0].samples[0].input_tensors == [capture.tensor_id(source)]
  assert result.loops[0].samples[0].output_tensors == [capture.tensor_id(source)]


def test_recycled_framework_storage_identity_does_not_create_false_alias() -> None:
  capture = graph.GraphCapture("reused-storage-identity")
  with capture:
    first = _Tensor()
    first_id = capture.tensor(
      first,
      _metadata(),
      storage_identity=("allocator-pointer", 7),
      storage_bytes=256,
    )
    first_storage = capture.tensors[first_id].storage_id
    del first
    gc.collect()

    second = _Tensor()
    second_id = capture.tensor(
      second,
      _metadata(),
      storage_identity=("allocator-pointer", 7),
      storage_bytes=256,
    )
    second_storage = capture.tensors[second_id].storage_id

  result = capture.finish()
  result.validate()
  assert first_storage != second_storage
  assert result.tensors[second_id].alias_of is None


def test_torch_precompile_capture_records_fake_aliases_mutation_and_state() -> None:
  torch = __import__("pytest").importorskip("torch")
  import vecops

  class Model(torch.nn.Module):
    def __init__(self):
      super().__init__()
      self.weight = torch.nn.Parameter(torch.ones(4, 4))

    def forward(self, value):
      matrix = value.view(4, 4)
      result = matrix + self.weight
      result.add_(2)
      graph.mark_tensor(result, name="annotated_result", annotations={"hot": True})
      return result[:, :2]

  result = vecops.precompile(Model().eval(), torch.ones(16), capture_graph=True, graph_name="torch-test")
  captured = result.graph
  assert captured is not None
  captured.validate()
  names = [event.name for event in captured.events if event.kind == "operator"]
  assert any("aten.view" in name for name in names)
  assert any("aten.add" in name for name in names)
  assert any("aten.add_" in name for name in names)
  allocation_events = [
    event
    for event in captured.events
    if any(use.access == graph.AccessKind.allocate for use in event.uses)
  ]
  assert allocation_events
  assert all("source" in event.attributes for event in allocation_events)
  weight = next(
    tensor
    for tensor in captured.tensors.values()
    if tensor.name == "weight" and tensor.kind == graph.TensorKind.parameter
  )
  add_event = next(event for event in captured.events if "aten.add.Tensor" in event.name)
  assert weight.id in {use.tensor_id for use in add_event.uses}
  assert any(tensor.alias_of is not None for tensor in captured.tensors.values())
  assert all(tensor.alias_of != tensor.id for tensor in captured.tensors.values())
  assert any(
    tensor.name == "annotated_result" and tensor.annotations.get("hot") is True
    for tensor in captured.tensors.values()
  )
  mutation = next(event for event in captured.events if "aten.add_.Tensor" in event.name)
  assert any(use.version_after == use.version_before + 1 for use in mutation.uses)
  assert captured.outputs


def test_torch_precompile_capture_also_supports_real_tensor_trace() -> None:
  torch = __import__("pytest").importorskip("torch")
  import vecops

  class Model(torch.nn.Module):
    def forward(self, value):
      return value.transpose(0, 1).contiguous()

  result = vecops.precompile(
    Model().eval(),
    torch.ones(2, 3),
    capture_graph=True,
    use_real_tensors=True,
  )
  captured = result.graph
  assert captured is not None
  captured.validate()
  assert len(captured.inputs) == 1
  assert len(captured.outputs) == 1
  output = captured.tensors[captured.outputs[0]]
  assert output.metadata.shape == (3, 2)
  assert output.alias_of is None
  output_storage = captured.storages[output.storage_id]
  assert output_storage.allocation_event is not None
  assert not output_storage.external
  assert captured.storages[captured.tensors[captured.inputs[0]].storage_id].external


def test_torch_capture_resolves_conditional_contiguous_and_to_aliasing() -> None:
  torch = __import__("pytest").importorskip("torch")
  import vecops

  class Model(torch.nn.Module):
    def forward(self, value):
      contiguous_alias = value.contiguous()
      contiguous_copy = value.transpose(0, 1).contiguous()
      to_alias = value.to(dtype=value.dtype, device=value.device)
      to_copy = value.to(torch.float64)
      return contiguous_alias, contiguous_copy, to_alias, to_copy

  captured = vecops.precompile(
    Model().eval(),
    torch.ones(2, 3),
    capture_graph=True,
  ).graph
  assert captured is not None
  captured.validate()

  # The two no-op expressions return the exact graph input. Boundary slots
  # preserve pytree arity, including two references to the same logical value.
  input_id = captured.inputs[0]
  assert captured.outputs.count(input_id) == 2
  assert len(captured.outputs) == 4
  fresh_outputs = [captured.tensors[item] for item in captured.outputs if item != input_id]
  assert {tensor.metadata.dtype for tensor in fresh_outputs} == {
    "torch.float32",
    "torch.float64",
  }
  assert all(
    tensor.storage_id != captured.tensors[input_id].storage_id
    for tensor in fresh_outputs
  )
  assert all(tensor.alias_of is None for tensor in fresh_outputs)


def test_torch_capture_preserves_repeated_boundary_slots_and_tied_state_names() -> None:
  torch = __import__("pytest").importorskip("torch")
  import vecops

  class Model(torch.nn.Module):
    def __init__(self):
      super().__init__()
      self.left = torch.nn.Parameter(torch.ones(2, 3))
      self.right = self.left

    def forward(self, first, second):
      return first + second + self.left

  value = torch.ones(2, 3)
  captured = vecops.precompile(
    Model().eval(),
    value,
    value,
    capture_graph=True,
  ).graph
  assert captured is not None
  captured.validate()

  # Boundary lists describe formal pytree slots rather than a set of values.
  assert len(captured.inputs) == 2
  assert captured.inputs[0] == captured.inputs[1]
  parameter = next(
    tensor
    for tensor in captured.tensors.values()
    if tensor.kind == graph.TensorKind.parameter
  )
  assert parameter.name == "left"
  assert parameter.annotations["names"] == ["left", "right"]
