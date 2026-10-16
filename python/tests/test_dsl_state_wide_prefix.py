# SPDX-FileCopyrightText: 2026 HenryRenYz and vecops contributors
# SPDX-License-Identifier: MIT
"""Wider state consumers preserve the exact lane-projection semantics."""

import json
import subprocess
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import vecops.language as T
from dsl_state_consumer_utils import assert_closed_store_consumers, state_consumers
from dsl_test_utils import cpp_compile_command, make_compiler, x86_target_available_for_test
from vecops.language._compiler import ir as I
from vecops.language._compiler.backends.cpu.backend import CPUBackend
from vecops.language._compiler.backends.cpu.cost import resolve_target
from vecops.language._compiler.backends.cpu.cpp.state import StateEmitter
from vecops.language._compiler.backends.cpu.state import StatePiece, StatePlan, state_view


class ProjectionEmitter(StateEmitter):
  """Supply printing primitives while exercising the real state emitter."""

  def __init__(self, bits=512):
    self.environment = SimpleNamespace(codegen_target=SimpleNamespace(architecture="x86", vector_bits=bits))
    self.hardware_scope = None
    shape = (I.Expr(I.ScalarType("int64"), "raw", data="extent"),)
    self.state = I.TileState("saved", I.TileType("uint8", shape))
    self.value = I.Expr(self.state.type, "state_ref", data=self.state)
    self.plan = StatePlan(self.state, state_view(self.state), (StatePiece((0,), 0, scalable=True, power=0),))
    self.prepared = SimpleNamespace(states={"saved": self.plan})
    self.tag = "vecops::vec::ScalableTag<std::uint8_t, 1>"
    self.tag_dtypes = {self.tag: "uint8"}
    self.tag_geometry = {self.tag: (True, 1)}
    self.fixed_tag_lanes = {}
    self.chunk_cache = {"outside": object()}
    self.lines = []
    self.counter = 0

  def line(self, text):
    self.lines.append(text)

  def opening(self, text):
    self.line(text + " {")

  def close(self):
    self.line("}")

  def unique(self, prefix):
    self.counter += 1
    return f"{prefix}_{self.counter}"

  def coordinates(self, shape, root_shape, coords):
    return (f"((extent) == 1 ? 0 : ({coords[0]}))",)

  def scalar(self, value):
    return str(I.constant(value)) if I.constant(value) is not None else value.data

  def vector_size(self, tag):
    return str((self.environment.codegen_target.vector_bits // 8) << self.tag_geometry[tag][1])

  def same_geometry(self, first, second):
    return I.same_extent(first, second)

  def arguments(self):
    root = (I.literal(128),)
    return (
      self.plan,
      self.value,
      self.tag,
      root,
      ("origin",),
      ("own_origin",),
      self.state_tags(self.plan, "uint8"),
      "active",
      0,
    )


@pytest.mark.parametrize(
  "boundary",
  (
    "wide",
    "arm",
    "unknown-width",
    "hardware",
    "mask",
    "pieces",
    "fixed-piece",
    "power",
    "rank",
    "root-rank",
    "axis",
    "negative-axis",
    "state-dtype",
    "tag-dtype",
    "same-width",
    "fixed-target",
  ),
)
def test_wide_prefix_eligibility_and_runtime_projection_boundary(boundary):
  emitter = ProjectionEmitter()
  arguments = list(emitter.arguments())
  if boundary == "arm":
    emitter.environment.codegen_target.architecture = "arm"
  elif boundary == "unknown-width":
    emitter.environment.codegen_target.vector_bits = None
  elif boundary == "hardware":
    emitter.hardware_scope = object()
  elif boundary == "mask":
    arguments[1] = replace(emitter.value, type=replace(emitter.value.type, mask=True))
  elif boundary == "pieces":
    arguments[0] = replace(emitter.plan, pieces=emitter.plan.pieces * 2)
  elif boundary == "fixed-piece":
    arguments[0] = replace(emitter.plan, pieces=(replace(emitter.plan.pieces[0], scalable=False),))
  elif boundary == "power":
    arguments[0] = replace(emitter.plan, pieces=(replace(emitter.plan.pieces[0], power=1),))
  elif boundary == "rank":
    arguments[1] = replace(emitter.value, type=replace(emitter.value.type, shape=(I.literal(2), I.literal(8))))
  elif boundary == "root-rank":
    arguments[3] = (I.literal(2), I.literal(64))
  elif boundary == "axis":
    arguments[-1] = 1
  elif boundary == "negative-axis":
    arguments[-1] = -1
  elif boundary == "state-dtype":
    arguments[0] = replace(emitter.plan, state=replace(emitter.state, type=replace(emitter.state.type, dtype="int8")))
  elif boundary == "tag-dtype":
    emitter.tag_dtypes[emitter.tag] = "int8"
  elif boundary == "same-width":
    emitter.tag_geometry[emitter.tag] = (True, 0)
  elif boundary == "fixed-target":
    emitter.tag_geometry[emitter.tag] = (False, 128)
  cache = emitter.chunk_cache
  result = emitter.wide_register_prefix(*arguments)
  assert emitter.chunk_cache is cache
  if boundary != "wide":
    assert result is None and not emitter.lines
    return
  source = "\n".join(emitter.lines)
  assert source.count("static_assert") == 2
  assert "(active) > 0 && (active) <= 128 && (own_origin) == 0 && (extent) != 1" in source
  assert "vecops::vec::blend" in source and "vecops::vec::mwhilelt" in source
  assert "else /* shifted or singleton state projection */" in source
  assert source.index("else /* shifted") < source.index("for (vecops::nint_t")


def test_fixed_prefix_also_requires_the_normalized_axis():
  emitter = ProjectionEmitter()
  value = replace(emitter.value, type=replace(emitter.value.type, shape=(I.literal(2),)))
  tag = "vecops::vec::FixedTag<std::uint8_t, 2>"
  emitter.fixed_tag_lanes[tag] = 2
  emitter.tag_dtypes[tag] = "uint8"
  assert emitter.fixed_register_prefix(emitter.plan, value, tag, value.type.shape, ("0",), "2", -1) is None
  assert not emitter.lines


def test_wide_projection_failure_restores_branch_chunk_cache():
  emitter = ProjectionEmitter()
  cache = emitter.chunk_cache

  def broken_projection(*args):
    assert emitter.chunk_cache is not cache
    emitter.chunk_cache["branch"] = object()
    raise RuntimeError("projection failed")

  emitter.state_projection = broken_projection
  with pytest.raises(RuntimeError, match="projection failed"):
    emitter.wide_register_prefix(*emitter.arguments())
  assert emitter.chunk_cache is cache and "branch" not in cache


def packing_template():
  P, DS = T.dynamic(lower=0), T.dynamic(lower=0)

  @T.template
  def kernel(
    A: T.Tensor((64, 32), (1, 64), dtype="uint8"),
    O: T.Tensor((P,), dtype="int8", access="output"),
    D: T.Tensor((DS,), dtype="int32", access="output"),
  ):
    assert D.shape[0] == A.shape[1]
    native = T.cast(T.load(A), "int8")
    T.pack(native, O, side="B", format="amx", dtype="int8")
    sums = T.reduce_add(T.bitcast(T.cast(native, "int32"), "uint32"), axis=0)
    T.store(D, T.bitcast(T.cast(0, "uint32") - sums * T.cast(3, "uint32"), "int32"))

  return kernel


def test_public_packing_and_compensation_reaches_runtime_wide_prefix():
  with state_consumers() as records:
    artifact = CPUBackend().lower(packing_template().ir, (), target="AVX512")
  assert_closed_store_consumers(artifact.backend_ir, artifact.source, records, resized=True)


@pytest.mark.skipif(not x86_target_available_for_test(), reason="physical prefix bridge targets x86")
def test_native_public_packing_compensation_and_full_owners(tmp_path):
  compiler = make_compiler(jobs=2, cache_dir=tmp_path / "cache", build_dir=tmp_path / "build")
  with state_consumers() as records:
    compiled = T.compile(packing_template(), compiler=compiler)
  assert_closed_store_consumers(compiled.backend_ir, compiled.generated_source, records, resized=True)
  input_owner = np.full((32 * 64 + 38,), 0xA6, dtype=np.uint8)
  a = input_owner[19:-19].reshape(32, 64).T
  a[:] = (np.arange(64 * 32).reshape(64, 32) * 5 + 1) % 127
  before = input_owner.copy()
  output_owner = np.empty(32 * 64 + 38, dtype=np.int8)
  correction_owner = np.empty(32 + 38, dtype=np.int32)
  output, correction = output_owner[19:-19], correction_owner[19:-19]
  expected = np.empty_like(output)
  for spatial in range(32):
    for k in range(64):
      index = (spatial // 16) * 1024 + (k // 4) * 64 + (spatial % 16) * 4 + k % 4
      expected[index] = a[k, spatial]
  expected_correction = (np.uint32(0) - a.astype(np.uint32).sum(axis=0, dtype=np.uint32) * np.uint32(3)).view(np.int32)
  for poison in (0x5A, 0xC3):
    output_owner.view(np.uint8).fill(poison)
    correction_owner.view(np.uint8).fill(poison)
    compiled.operator.native(a, output, correction, **compiled._core.specialization_values)
    np.testing.assert_array_equal(output, expected)
    np.testing.assert_array_equal(correction, expected_correction)
    np.testing.assert_array_equal(input_owner, before)
    for owner in (output_owner, correction_owner):
      assert np.all(owner[:19].view(np.uint8) == poison)
      assert np.all(owner[-19:].view(np.uint8) == poison)


@pytest.mark.skipif(not x86_target_available_for_test(), reason="physical prefix bridge targets x86")
def test_native_emitted_wide_prefix_matches_projection_for_every_physical_lane(tmp_path):
  bits = resolve_target("Native").vector_bits
  assert bits
  lanes = bits // 8
  emitter = ProjectionEmitter(bits)
  shape = (I.literal(2 * lanes),)
  result = emitter.state_vector(emitter.value, emitter.tag, shape, ("origin",), "active", 0)
  generated = "\n".join(emitter.lines) + f"\nreturn {result};"
  assert "/* wide register prefix */" in generated
  projection = ProjectionEmitter(bits)
  result = projection.state_projection(
    projection.plan,
    projection.value,
    projection.tag,
    shape,
    ("origin",),
    projection.coordinates(projection.value.type.shape, shape, ("origin",)),
    projection.state_tags(projection.plan, "uint8"),
    "active",
    0,
  )
  fallback = "\n".join(projection.lines) + f"\nreturn {result};"
  cpp = tmp_path / "wide-prefix.cpp"
  cpp.write_text(
    "// SPDX-FileCopyrightText: 2026 HenryRenYz and vecops contributors\n"
    "// SPDX-License-Identifier: MIT\n"
    "#include <array>\n#include <cassert>\n#include <cstdint>\n#include <type_traits>\n#include <vecops/vec/Vec.h>\n"
    "using Source = vecops::vec::ScalableTag<std::uint8_t,0>;\n"
    "using Destination = vecops::vec::ScalableTag<std::uint8_t,1>;\n"
    f"auto prefix(vecops::vec::Vec<Source> saved_r0, int active, int origin, int extent) {{\n{generated}\n}}\n"
    f"auto original(vecops::vec::Vec<Source> saved_r0, int active, int origin, int extent) {{\n{fallback}\n}}\n"
    f"""int main() {{
      std::array<std::uint8_t,{lanes}> input;
      for (int i=0;i<{lanes};++i) input[i]=static_cast<std::uint8_t>(i*3+1);
      auto value=vecops::vec::load_convert(Source{{}},input.data(),vecops::vec::opt::unmasked);
      for (int extent : {{0,1,16}}) for (int origin : {{-2,0,1,{lanes - 1},{lanes}}})
        for (int active : {{-3,0,1,16,{lanes},{2 * lanes},{2 * lanes + 1}}}) {{
          // The original singleton loop requires a valid destination lane.
          if (extent==1 && active>{2 * lanes}) continue;
          auto expected=original(value,active,origin,extent),actual=prefix(value,active,origin,extent);
          for (int lane=0;lane<{2 * lanes};++lane)
            assert(vecops::vec::get(Destination{{}},expected,lane)==vecops::vec::get(Destination{{}},actual,lane));
        }}
    }}\n"""
  )
  root = Path(__file__).resolve().parents[2]
  executable = tmp_path / "wide-prefix"
  command = cpp_compile_command(
    cpp,
    executable,
    flags=(
      "-std=c++20",
      "-O3",
      "-march=native",
      "-ffp-contract=off",
      "-UNDEBUG",
      "-Wno-ignored-attributes",
      "-I",
      str(root / "include"),
      "-I",
      str(root / "include/vecops"),
    ),
  )
  (tmp_path / "command.json").write_text(json.dumps(command, indent=2))
  build = subprocess.run(command, capture_output=True, text=True, check=False)
  (tmp_path / "build.log").write_text(build.stdout + build.stderr)
  assert build.returncode == 0, build.stderr
  run = subprocess.run([str(executable)], capture_output=True, text=True, check=False)
  assert run.returncode == 0, run.stdout + run.stderr
