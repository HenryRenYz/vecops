# SPDX-FileCopyrightText: 2026 HenryRenYz and vecops contributors
# SPDX-License-Identifier: MIT
"""End-to-end ordered dispatch through a generated mutable Torch bridge."""

from concurrent.futures import ThreadPoolExecutor
import os
from pathlib import Path

import numpy as np
import pytest
import torch
import vecops
import vecops.typing as vt

_DYNAMIC_OUTPUT_STRIDES = "Dynamic<1,D,1048576> 1"


@vecops.jit(source="../runtime/TestKernel.cpp")
def float32_recipe(
  input: vt.In(torch.float32)["B D", "stride_D 1"],
  output: vt.Out(torch.float32)["B D", _DYNAMIC_OUTPUT_STRIDES],
  factor: float = 2.0,
  add_one: bool = False,
  *,
  ComputeType: vt.DType = torch.float32,
) -> None: ...


@vecops.jit(source="../runtime/TestKernel.cpp")
def float64_recipe(
  input: vt.In(np.float64)["B D", "stride_D 1"],
  output: vt.Out(np.float64)["B D", _DYNAMIC_OUTPUT_STRIDES],
  factor: float = 2.0,
  add_one: bool = False,
  *,
  ComputeType: vt.DType = np.float64,
) -> None: ...


def test_generated_mutable_out_operator_and_ordered_dispatch(tmp_path: Path) -> None:
  compiler = vecops.Compiler(
    cache_mode="compile-only",
    build_dir=tmp_path / "kernels",
    target="Scalar",
    cc=os.environ.get("CC", "/usr/bin/gcc"),
    cxx=os.environ.get("CXX", "/usr/bin/g++"),
    jobs=2,
  )
  function = vecops.ops.torch.register(
    "vecops_generated_test::run",
    [float32_recipe, float64_recipe],
    compiler=compiler,
    build_directory=tmp_path / "bridge",
    return_outputs=True,
  )

  input = torch.arange(1, 9, dtype=torch.float64).reshape(2, 4)

  class Model(torch.nn.Module):
    def __init__(self) -> None:
      super().__init__()
      self.register_buffer("offset", torch.ones(2, 4, dtype=torch.float64))
      self.saw_precompile = False

    def forward(self, value):
      self.saw_precompile = vecops.is_precompiling()
      prepared_input = value + self.offset
      destination = torch.empty_like(prepared_input)
      function(prepared_input, destination, 3.0, True)
      # Exact duplicate metadata must become one request even when runtime
      # scalar payloads differ.
      function(prepared_input, destination, 4.0, False)
      return destination

  traced_model = Model()
  precompiled = vecops.precompile(
    traced_model, input, parallelism=2, capture_graph=True
  )
  assert traced_model.saw_precompile
  assert precompiled.collected == 1
  assert precompiled.prepared == 1
  assert precompiled.graph is not None
  precompiled.graph.validate()
  vecops_events = [
    event
    for event in precompiled.graph.events
    if event.name == "vecops_generated_test::run"
  ]
  assert len(vecops_events) == 2
  assert all(
    [use.access.value for use in event.uses if use.argument == "output"]
    == ["write"]
    for event in vecops_events
  )
  assert not vecops.is_precompiling()
  real_trace = vecops.precompile(
      traced_model, input, use_real_tensors=True, parallelism=2
  )
  assert real_trace.collected == 1

  output = torch.zeros_like(input)
  result = function(input, output, 3.0, True)

  assert result is output
  torch.testing.assert_close(output, input * 3 + 1)

  # A second call with identical tensor metadata takes the prepared bridge
  # path. Runtime scalar values are deliberately not part of the executable
  # specialization and must still be read from the current call frame.
  second_input = torch.arange(9, 17, dtype=torch.float64).reshape(2, 4)
  second_output = torch.zeros_like(second_input)
  assert function(second_input, second_output, 4.0, False) is second_output
  torch.testing.assert_close(second_output, second_input * 4)

  # A different specialization must miss the exact-metadata cache, resolve a
  # distinct artifact, and leave the first specialization reusable.
  wider_input = torch.arange(1, 17, dtype=torch.float64).reshape(2, 8)
  wider_output = torch.zeros_like(wider_input)
  assert function(wider_input, wider_output, 2.0, True) is wider_output
  torch.testing.assert_close(wider_output, wider_input * 2 + 1)
  output.zero_()
  assert function(input, output, 3.0, True) is output
  torch.testing.assert_close(output, input * 3 + 1)

  def threaded_call(offset: int) -> torch.Tensor:
    value = input + offset
    destination = torch.zeros_like(value)
    function(value, destination, 3.0, True)
    return destination

  with ThreadPoolExecutor(max_workers=4) as executor:
    threaded = list(executor.map(threaded_call, range(4)))
  for offset, value in enumerate(threaded):
    torch.testing.assert_close(value, (input + offset) * 3 + 1)

  invalid_output = torch.zeros((4, 2), dtype=torch.float64).T
  with pytest.raises(RuntimeError, match="stride"):
    function(input, invalid_output, 3.0, True)
  assert vecops.ops.torch.vecops_generated_test.run is function
  schema = torch._C._dispatch_find_schema_or_throw("vecops_generated_test::run", "").schema()
  assert "Tensor(a!) output" in str(schema)
  assert "-> Tensor" in str(schema)
  meta_input = torch.empty((2, 4), dtype=torch.float64, device="meta")
  meta_output = torch.empty_like(meta_input)
  assert function(meta_input, meta_output, 3.0, True) is meta_output
  assert torch._C._dispatch_has_kernel_for_dispatch_key("vecops_generated_test::run", "Meta")

  def compiled_call(value, destination):
    return function(value, destination, 3.0, True)

  compiled = torch.compile(compiled_call, backend="eager", fullgraph=True)
  compiled_output = torch.zeros_like(input)
  assert compiled(input, compiled_output) is compiled_output
  torch.testing.assert_close(compiled_output, input * 3 + 1)
