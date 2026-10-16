# SPDX-FileCopyrightText: 2026 HenryRenYz and vecops contributors
# SPDX-License-Identifier: MIT
"""Closed Fixed consumers preserve physical padding and lazy projection."""

import dataclasses

import pytest
import test_dsl_state_wide_prefix as m
from vecops.language._compiler import ir as I
from vecops.language._compiler.backends.cpu.state import StatePiece, StatePlan, state_view


@pytest.mark.parametrize(
  "boundary",
  (
    "wide",
    "fixed-target",
    "unknown-width",
    "arm",
    "hardware",
    "mask",
    "pieces",
    "not-closed",
    "scalable-piece",
    "state-dtype",
    "tag-dtype",
    "rank",
    "root-rank",
    "axis",
    "negative-axis",
    "dynamic-length",
    "literal-args",
    "literal-subclass",
    "boolean-literal",
    "singleton",
    "oversize",
    "target-smaller",
    "unknown-tag",
    "shifted",
    "inactive",
    "negative-active",
    "large-active",
    "cache-exception",
  ),
)
def test_closed_fixed_consumer_boundaries(boundary):
  rows = []
  e = m.ProjectionEmitter()
  typ = I.VectorType("uint8", (I.literal(16),), False, I.VectorTag(16, 0, "simd", "uint8"))
  state = I.TileState("saved", typ)
  value = I.Expr(typ, "state_ref", data=state)
  plan = StatePlan(state, state_view(state), (StatePiece((0,), 16),), True)
  root_shape = (I.literal(128),)
  own = ("own_origin",)
  coords = ("origin",)
  active = "active"
  axis = 0
  if boundary == "fixed-target":
    e.tag = "vecops::vec::FixedTag<std::uint8_t, 32>"
    e.tag_dtypes[e.tag] = "uint8"
    e.tag_geometry[e.tag] = (False, 32)
  elif boundary == "unknown-width":
    e.environment.codegen_target.vector_bits = None
  elif boundary == "arm":
    e.environment.codegen_target.architecture = "aarch64"
  elif boundary == "hardware":
    e.hardware_scope = object()
  elif boundary == "mask":
    value = dataclasses.replace(value, type=dataclasses.replace(typ, mask=True))
  elif boundary == "pieces":
    plan = dataclasses.replace(plan, pieces=plan.pieces * 2)
  elif boundary == "not-closed":
    plan = dataclasses.replace(plan, closed_fixed=False)
  elif boundary == "scalable-piece":
    plan = dataclasses.replace(plan, pieces=(StatePiece((0,), 0, True),))
  elif boundary == "state-dtype":
    plan = dataclasses.replace(plan, state=dataclasses.replace(state, type=dataclasses.replace(typ, dtype="int8")))
  elif boundary == "tag-dtype":
    e.tag_dtypes[e.tag] = "int8"
  elif boundary == "rank":
    value = dataclasses.replace(value, type=dataclasses.replace(typ, shape=(I.literal(2), I.literal(8))))
  elif boundary == "root-rank":
    root_shape = (I.literal(2), I.literal(64))
  elif boundary == "axis":
    axis = 1
  elif boundary == "negative-axis":
    axis = -1
  elif boundary == "dynamic-length":
    value = dataclasses.replace(
      value, type=dataclasses.replace(typ, shape=(I.Expr(I.ScalarType("int64"), "raw", data="length"),))
    )
  elif boundary == "literal-args":
    value = dataclasses.replace(
      value, type=dataclasses.replace(typ, shape=(I.Expr(I.ScalarType("int64"), "literal", (I.literal(1),), 16),))
    )
  elif boundary == "literal-subclass":

    class Unknown(I.Expr):
      pass

    value = dataclasses.replace(
      value, type=dataclasses.replace(typ, shape=(Unknown(I.ScalarType("int64"), "literal", (), 16),))
    )
  elif boundary == "boolean-literal":
    value = dataclasses.replace(
      value, type=dataclasses.replace(typ, shape=(I.Expr(I.ScalarType("int64"), "literal", (), True),))
    )
  elif boundary == "singleton":
    value = dataclasses.replace(value, type=dataclasses.replace(typ, shape=(I.literal(1),)))
    plan = dataclasses.replace(plan, pieces=(StatePiece((0,), 1),))
  elif boundary == "oversize":
    value = dataclasses.replace(value, type=dataclasses.replace(typ, shape=(I.literal(128),)))
    plan = dataclasses.replace(plan, pieces=(StatePiece((0,), 128),))
  elif boundary == "target-smaller":
    e.tag_geometry[e.tag] = (False, 8)
  elif boundary == "unknown-tag":
    e.tag_geometry.pop(e.tag)
  elif boundary == "shifted":
    own = ("1",)
  elif boundary == "inactive":
    active = "0"
  elif boundary == "negative-active":
    active = "-3"
  elif boundary == "large-active":
    active = "129"
  cache = e.chunk_cache
  if boundary == "cache-exception":

    def broken(*args):
      assert e.chunk_cache is not cache
      e.chunk_cache["inner"] = 1
      raise RuntimeError("projection failure")

    e.state_projection = broken
  try:
    result = e.closed_fixed_register_resize(
      plan, value, e.tag, root_shape, coords, own, e.state_tags(plan, "uint8"), active, axis
    )
  except RuntimeError as ex:
    assert (
      boundary == "cache-exception"
      and str(ex) == "projection failure"
      and e.chunk_cache is cache
      and "inner" not in cache
    )
    rows.append({"boundary": boundary, "exception_cache_restored": True})
    return
  assert e.chunk_cache is cache
  enters = boundary in {"wide", "fixed-target", "shifted", "inactive", "negative-active", "large-active"}
  if enters:
    assert result is not None and any("closed fixed register resize" in line for line in e.lines)
    assert any("< 16 ?" in line for line in e.lines)
    assert any("else /* shifted or inactive closed fixed projection */" in line for line in e.lines)
  else:
    assert result is None and not e.lines, boundary
  rows.append(
    {
      "boundary": boundary,
      "conditional_resize_emitted": enters,
      "runtime_guard": next((x for x in e.lines if "closed fixed register resize" in x), None),
      "cache_restored": True,
    }
  )


@pytest.mark.parametrize("fixed_target", (False, True))
@pytest.mark.skipif(not m.x86_target_available_for_test(), reason="closed Fixed bridge targets x86")
def test_native_closed_fixed_padding_and_shifted_projection(tmp_path, fixed_target):
  import json
  import subprocess
  from pathlib import Path

  from dsl_test_utils import cpp_compile_command
  from vecops.language._compiler.backends.cpu.cost import resolve_target

  bits = resolve_target("Native").vector_bits
  assert bits
  native = bits // 8
  target = 32 if fixed_target else 2 * native
  destination = (
    f"vecops::vec::FixedTag<std::uint8_t,{target}>" if fixed_target else "vecops::vec::ScalableTag<std::uint8_t,1>"
  )
  parts = []
  optimized = 0
  source_counts = (1, 2, 4, 8, 16) if fixed_target else (1, 2, 4, 8, 16, native)
  for count in source_counts:

    class Emitter(m.ProjectionEmitter):
      def vector_size(self, tag):
        scalable, width = self.tag_geometry[tag]
        return str((native << width) if scalable else width)

    emitter = Emitter(bits)
    emitter.tag = destination
    emitter.tag_dtypes[destination] = "uint8"
    emitter.tag_geometry[destination] = (not fixed_target, 1 if not fixed_target else target)
    typ = I.VectorType("uint8", (I.literal(count),), False, I.VectorTag(count, 0, "simd", "uint8"))
    state = I.TileState("saved", typ)
    value = I.Expr(typ, "state_ref", data=state)
    plan = StatePlan(state, state_view(state), (StatePiece((0,), count),), True)
    emitter.prepared.states = {"saved": plan}
    shape = (I.literal(target),)
    result = emitter.state_vector(value, destination, shape, ("origin",), "active", 0)
    generated = "\n".join(emitter.lines) + f"\nreturn {result};"
    # Prove this fixture exercises the changed emitter before compiling it.
    if 1 < count < target:
      assert "/* closed fixed register resize */" in generated
      assert "vecops::vec::concat(" in generated
      assert f"< {count} ?" in generated
      optimized += 1
    original = Emitter(bits)
    result = original.state_projection(
      plan,
      value,
      destination,
      shape,
      ("origin",),
      original.coordinates(typ.shape, shape, ("origin",)),
      original.state_tags(plan, "uint8"),
      "active",
      0,
    )
    reference = "\n".join(original.lines) + f"\nreturn {result};"
    assert "closed fixed register resize" not in reference and "vecops::vec::concat(" not in reference
    parts.append(
      f"using Source{count}=vecops::vec::FixedTag<std::uint8_t,{count}>;\nauto actual{count}(vecops::vec::Vec<Source{count}> saved_r0,int active,int origin,int extent){{\n{generated}\n}}\nauto original{count}(vecops::vec::Vec<Source{count}> saved_r0,int active,int origin,int extent){{\n{reference}\n}}\n"
    )
    physical = max(16, count)
    initialization = "".join(
      f"value=vecops::vec::set(Source{count}{{}},value,{i},std::uint8_t{{{i % 127 + 1}}});" for i in range(count)
    )
    parts.append(f"""void check{count}() {{
    using Physical=vecops::vec::FixedTag<std::uint8_t,{physical}>;
    auto poison=vecops::vec::fill(Physical{{}},std::uint8_t{{166}});
    auto value=vecops::vec::bitcast(Source{count}{{}},Physical{{}},poison);
    {initialization}
    for(int origin : {{-2,0,1,{count - 1},{count}}})
      for(int active : {{-3,0,1,{count - 1},{count},{count + 1},{native},{target},{target + 1}}}) {{
        if({count}==1 && active>{target}) continue;
        auto expected=original{count}(value,active,origin,{count});
        auto actual=actual{count}(value,active,origin,{count});
        for(int i=0;i<{target};++i)
          assert(vecops::vec::get(Destination{{}},actual,i)==vecops::vec::get(Destination{{}},expected,i));
      }}
    }}\n""")
  assert optimized == (4 if fixed_target else 5)
  cpp = tmp_path / "closed-fixed-padding.cpp"
  cpp.write_text(
    "// SPDX-FileCopyrightText: 2026 HenryRenYz and vecops contributors\n// SPDX-License-Identifier: MIT\n#include <cassert>\n#include <cstdint>\n#include <type_traits>\n#include <vecops/vec/Vec.h>\n"
    + f"using Destination={destination};\n"
    + "".join(parts)
    + "\nint main(){"
    + "".join(f"check{n}();" for n in source_counts)
    + "}\n"
  )
  root = Path(__file__).resolve().parents[2]
  executable = tmp_path / "closed-fixed-padding"
  command = cpp_compile_command(
    cpp,
    executable,
    flags=(
      "-std=c++20",
      "-O3",
      "-march=native",
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


@pytest.mark.parametrize("physical_lanes", (8, None))
def test_register_alias_geometry_uses_actual_descriptor(physical_lanes):
  from types import SimpleNamespace

  from vecops.language._compiler.backends.cpu.cpp.registers import RegistersEmitter
  from vecops.language._compiler.backends.cpu.cpp.state import StateEmitter

  class Emitter(RegistersEmitter, StateEmitter):
    def __init__(self):
      self.lines = []
      self.tag_dtypes = {}
      self.tag_geometry = {}
      self.fixed_tag_lanes = {}
      self.counter = 0

    def unique(self, name):
      self.counter += 1
      return f"{name}_{self.counter}"

    def line(self, text):
      self.lines.append(text)

  emitter = Emitter()
  # The supplied physical descriptor controls printing even when a logical
  # shape is different; a scalable descriptor must not become Fixed(16).
  typ = I.VectorType("uint8", (I.literal(16),), False, I.VectorTag(16, 0, "simd", "uint8"))
  value = I.Expr(typ, "state_ref", data=I.TileState("source", typ))
  program = SimpleNamespace(inputs=(value,), outputs=(value,), operations=(value,))
  descriptor = I.VectorTag(physical_lanes, 0, "simd", "uint8")
  alias, outputs = emitter.register_program(program, lambda index, tag: "loaded", physical_tags={id(value): descriptor})
  assert outputs == ("loaded",)
  assert emitter.fixed_tag_lanes[alias] == 16
  if physical_lanes is None:
    assert "ScalableTag<std::uint8_t, 0>" in emitter.lines[0]
    assert alias not in emitter.tag_geometry
  else:
    assert "FixedTag<std::uint8_t, 8>" in emitter.lines[0]
    assert emitter.tag_geometry[alias] == (False, 8)
    same = emitter.state_tag("uint8", False, 8)
    different = emitter.state_tag("uint8", False, 16)
    assert emitter.same_register_tag(alias, same)
    assert not emitter.same_register_tag(alias, different)
