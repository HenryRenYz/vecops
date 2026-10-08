# SPDX-FileCopyrightText: 2026 HenryRenYz and vecops contributors
# SPDX-License-Identifier: MIT
"""Cast/load fusion preserves lexical read captures across aliased writes.

The two casts are ordered rounding boundaries. Their original read remains a
dependency even though CPU code generation may use one fused load_convert.
Native checks use an independent byte-overlap simulation, not another kernel.
"""

from __future__ import annotations

import re
from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest
import vecops
import vecops.language as T
from dsl_test_utils import assert_full_sve_memory_words, assert_no_vec_mask_arrays, make_compiler
from test_dsl_matmul_sve_optimized import _cross
from vecops._arm_vector_state import VectorState
from vecops.language._compiler import ir as I
from vecops.language._compiler.backends.cpu.backend import CPUBackend
from vecops.language._compiler.ir.traversal import nodes, rewrite_expressions
from vecops.language._compiler.operations import current
from vecops.language._compiler.passes.kernel.canonicalize import canonicalize_memory

KINDS = ("direct", "helper", "loop", "branch")


@T.func
def _capture_helper(A, valid, masked):
  if masked:
    return T.load(A, mask=valid, other=-1.5)
  return T.load(A)


def capture_template(kind="direct", masked=False):
  N = T.dynamic(lower=0)
  AS, MS, CS, DS, ES = (T.dynamic() for _ in range(5))

  @T.template
  def kernel(
    A: T.Tensor((N,), (AS,), "float32"),
    Valid: T.Tensor((N,), (MS,), "int8"),
    C: T.Tensor((N,), (CS,), "float16", access="output"),
    D: T.Tensor((N,), (DS,), "float32", access="output"),
    E: T.Tensor((N,), (ES,), "float32", access="output"),
    count: int,
    flag: bool,
  ):
    valid = T.load(Valid) != 0
    if kind == "helper":
      captured = _capture_helper(A, valid, masked)
    elif masked:
      captured = T.load(A, mask=valid, other=-1.5)
    else:
      captured = T.load(A)
    if kind == "loop":
      for _ in T.serial(count, unroll=1):
        T.store(C, captured)
    elif kind == "branch":
      if flag:
        T.store(C, captured)
      else:
        T.store(C, captured)
    else:
      T.store(C, captured)
    # This new read belongs after the store; it must not be merged with the
    # captured pre-store read merely because storage/dtype/layout are equal.
    after = T.load(A)
    T.store(D, T.cast(T.cast(captured, "float16"), "float32"))
    T.store(E, after)

  return kernel


def _stores(kernel, storage):
  return [node for node in nodes(kernel.body) if isinstance(node, I.Store) and node.destination.storage == storage]


@pytest.mark.parametrize("kind", KINDS)
@pytest.mark.parametrize("masked", (False, True))
def test_captured_cast_retains_original_read_identity_and_fused_load_convert(kind, masked):
  artifact = CPUBackend().lower(capture_template(kind, masked).ir, (), target="AVX512")
  c_reads = {
    id(read): read
    for store in _stores(artifact.kernel, "C")
    for read in I.loads_unique(store.value)
    if read.data.storage == "A"
  }
  d = _stores(artifact.kernel, "D")[0]
  d_reads = [read for read in I.loads_unique(d.value) if read.data.storage == "A"]
  assert c_reads and len(d_reads) == 1
  assert d_reads[0] is c_reads[id(d_reads[0])]
  assert d_reads[0].type.dtype == d_reads[0].data.dtype == "float32"
  assert d.value.op == "cast" and d.value.args[0].op == "cast"
  assert d.value.args[0].type.dtype == "float16"
  after_reads = [read for read in I.loads_unique(_stores(artifact.kernel, "E")[0].value) if read.data.storage == "A"]
  assert len(after_reads) == 1 and after_reads[0] is not d_reads[0]
  source = artifact.source
  assert_no_vec_mask_arrays(source)
  half_tags = re.findall(r"using (\w+) = vecops::vec::ScalableTag<vecops::float16_t,", source)
  fused_half_load = re.compile(
    r"load_convert\((?:vecops::vec::Rebind<vecops::float16_t,[^\n]+?|(?:" + "|".join(half_tags) + r"))\{\},"
  )
  assert fused_half_load.search(source)
  assert "store_convert(" in source and "cvt::unordered" not in source
  # Earlier C writes preserve this original F32 read. The typed load proxy
  # must select that same snapshot, not only an unrelated current A-D guard.
  snapshot_loads = [line for line in source.splitlines() if fused_half_load.search(line) and "snapshot" in line]
  assert snapshot_loads


def _synthetic_ir(*, captured, masked=False):
  extent = (I.literal(7),)
  view = I.View("A", "float32", extent, (I.literal(1),), I.literal(0))
  if masked:
    mv = replace(view, storage="Valid", dtype="int8")
    mask = I.binary("ne", I.Expr(I.TileType("int8", extent), "load", data=mv), I.literal(0))
    args = (mask, I.literal(-1.5, "float32"))
  else:
    args = ()
  read = I.Expr(I.TileType("float32", extent), "load", args, view)
  cast = I.Expr(I.TileType("float16", extent), "cast", (read,))
  statements = [I.BindTile(read)] if captured else []
  statements += [
    I.Store(replace(view, storage="C", dtype="float16"), cast),
    I.Store(replace(view, storage="D", dtype="float16"), cast),
  ]
  return I.KernelIR(I.Block(statements), {}, {}), read


@pytest.mark.parametrize("captured", (False, True))
@pytest.mark.parametrize("masked", (False, True))
def test_canonicalization_preserves_only_captured_load_edges_and_shared_cast(captured, masked):
  kernel, read = _synthetic_ir(captured=captured, masked=masked)
  canonicalize_memory(kernel, None)
  first, second = _stores(kernel, "C")[0].value, _stores(kernel, "D")[0].value
  assert first is second
  if captured:
    assert first.op == "cast" and first.args[0] is read
    assert kernel.body.statements[0].value is read
  else:
    assert first.op == "load" and first.type.dtype == "float16"
    assert first.data.dtype == "float32" and first.args == read.args


def test_original_callback_observes_shared_masked_read_before_argument_clone():
  kernel, original = _synthetic_ir(captured=True, masked=True)
  observed = []

  def rewrite(changed, before):
    if before is original:
      observed.append((changed, before))
    if before.op == "literal" and before.data == -1.5:
      return I.literal(-2.5, "float32")
    return changed

  rewrite_expressions(kernel, rewrite, with_original=True)
  assert len(observed) == 1
  changed, before = observed[0]
  assert before is original and changed is not original
  assert changed.args[1].data == -2.5 and original.args[1].data == -1.5
  assert kernel.body.statements[0].value is changed
  assert _stores(kernel, "C")[0].value.args[0] is changed
  assert _stores(kernel, "D")[0].value.args[0] is changed


def test_default_rewrite_callback_keeps_existing_one_argument_contract():
  kernel, original = _synthetic_ir(captured=True, masked=True)
  calls = []

  def rewrite(value):
    calls.append(value)
    return value

  rewrite_expressions(kernel, rewrite)
  assert sum(value is original for value in calls) == 1
  assert kernel.body.statements[0].value is original


def test_capture_discovery_visits_shared_binding_dag_union_once():
  kernel, read = _synthetic_ir(captured=True)
  bindings, value = [], read
  for _ in range(28):
    value = I.Expr(read.type, "add", (value, value))
    bindings.append(I.BindTile(value))
  kernel.body.statements[1:1] = bindings
  catalog, calls = current(), []

  class CountingCatalog:
    def has(self, expression, trait):
      calls.append(id(expression))
      return catalog.has(expression, trait)

  context = SimpleNamespace(environment=SimpleNamespace(operations=CountingCatalog()))
  canonicalize_memory(kernel, context)
  assert len(calls) == len(set(calls)) == 29
  assert _stores(kernel, "C")[0].value.args[0] is read


def test_loop_carried_cast_state_retains_ordered_rounding_and_initial_read_capture():
  @T.template
  def kernel(
    A: T.Tensor((17,), dtype="float32"),
    C: T.Tensor((17,), dtype="float16", access="output"),
    D: T.Tensor((17,), dtype="float32", access="output"),
    count: int,
  ):
    carried = T.load(A)
    for _ in T.serial(count, unroll=1):
      T.store(C, carried)
      carried = T.cast(T.cast(carried, "float16"), "float32")
    T.store(D, carried)

  artifact = CPUBackend().lower(kernel.ir, (), target="AVX512")
  declarations = [node for node in nodes(artifact.kernel.body) if isinstance(node, I.DeclareState)]
  updates = [node for node in nodes(artifact.kernel.body) if isinstance(node, I.UpdateState)]
  assert declarations and updates
  initial_reads = [
    read for declaration in declarations for read in I.loads_unique(declaration.value) if read.data.storage == "A"
  ]
  assert len(initial_reads) == 1 and initial_reads[0].type.dtype == "float32"
  assert any(
    update.value.op == "cast" and update.value.args[0].op == "cast" and update.value.args[0].type.dtype == "float16"
    for update in updates
  )
  assert "vecops::vec::convert(" in artifact.source
  assert_no_vec_mask_arrays(artifact.source)


def _rounding_template(shared=False, masked=False):
  @T.template
  def kernel(
    A: T.Tensor((2048,), dtype="float32"),
    Valid: T.Tensor((2048,), dtype="int8"),
    C: T.Tensor((2048,), dtype="float32", access="output"),
  ):
    if masked:
      captured = T.load(A, mask=T.load(Valid) != 0, other=-1.5)
    else:
      captured = T.load(A)
    rounded = T.cast(T.cast(captured, "float16"), "float32")
    if shared:
      T.store(C, captured + rounded)
    else:
      T.store(C, rounded)

  return kernel


@pytest.mark.parametrize("shared", (False, True))
def test_shared_vector_read_uses_register_instead_of_reloading_for_cast(shared):
  artifact = CPUBackend().lower(_rounding_template(shared).ir, (), target="AVX512")
  regions = [
    region for region in artifact.lowered.regions if any(op.expression.op == "cast" for op in region.operations)
  ]
  assert regions
  for region in regions:
    loads = [op for op in region.operations if op.expression.op == "load" and op.expression.data.storage == "A"]
    assert len(loads) == 1
    source_index = loads[0].index
    users = sum(index == source_index for op in region.operations for index in op.operands)
    assert users == (2 if shared else 1)
  assert_no_vec_mask_arrays(artifact.source)
  if shared:
    assert "vecops::vec::convert(" in artifact.source
  else:
    # The outer widening can fuse with the F32 store; the inner narrowing
    # still happens in the F16 load_convert, not as an identity operation.
    assert "ScalableTag<vecops::float16_t," in artifact.source
    assert "vecops::vec::store_convert(" in artifact.source


def _rounding_wrapper(source):
  return f"""#include "vecops/Kernel.h"
namespace vecops::spec {{ inline constexpr auto Parallelism = vecops::meta::cint<1>; }}
{source}
extern "C" void cast_capture_probe(float* a, std::int8_t* valid, float* c) {{
  auto shape=vecops::tensor::make_shape(vecops::meta::cint<2048>);
  auto strides=vecops::tensor::make_strides(vecops::meta::cint<1>);
  auto av=vecops::tensor::make_tensor(a, vecops::tensor::Layout{{shape,strides}});
  auto mv=vecops::tensor::make_tensor(valid, vecops::tensor::Layout{{shape,strides}});
  auto cv=vecops::tensor::make_tensor(c, vecops::tensor::Layout{{shape,strides}});
  __kernel__(av,mv,cv);
}}
"""


def _source_memory_loads(code):
  """Select vector loads rooted at the F32 input pointer, including LD1B."""
  lines = [line.split("//", 1)[0].strip() for line in code.splitlines()]
  root = None
  for line in lines:
    match = re.match(r"ld1w\s+\{\s*z\d+\.s\s*\},\s*p\d+/z,\s*\[(x\d+)", line)
    if match:
      root = match[1]
      break
  if root is None:
    return []

  # The compiler may use a byte load to preserve the bits of later F32 words
  # while assembling a wider conversion group. Follow only pointer arithmetic
  # derived from the same base used by the initial word load; this excludes
  # Valid-mask and unrelated snapshot accesses.
  aliases = {root}
  changed = True
  while changed:
    changed = False
    for line in lines:
      match = re.match(r"(?:add|mov)\s+(x\d+),\s*(x\d+)(?:,|$)", line)
      if match and match[2] in aliases and match[1] not in aliases:
        aliases.add(match[1])
        changed = True

  selected = []
  for line in lines:
    match = re.match(r"ld1([bhwd])\s+\{\s*z\d+\.[bhsd]\s*\},\s*p\d+/z,\s*\[(x\d+)", line)
    if match and match[2] in aliases:
      selected.append(line)
  return selected


def _assert_contiguous_source_groups(path_instructions, source_loads, *, words):
  """Prove source accesses cover one F32 word group at contiguous byte offsets."""
  # Symbolic expressions are affine in the loop's lane index and CNTB.  The
  # byte-preserving conversion path uses a derived A+4*index pointer followed
  # by offsets VL, 2*VL, ...; track those exact values instead of treating a
  # matching instruction count as proof of a complete read.
  values = {}
  pointers = {}
  root = None
  comparison = None

  def scalar(register):
    immediate = re.fullmatch(r"#(-?(?:0x[0-9a-fA-F]+|\d+))", register)
    if immediate:
      return ((), int(immediate[1], 0))
    if register in {"xzr", "wzr"}:
      return ((), 0)
    return values.get(register, (((f"reg:{register}", 1),), 0))

  def combine(left, right, sign=1):
    terms = dict(left[0])
    for name, coefficient in right[0]:
      terms[name] = terms.get(name, 0) + sign * coefficient
    terms = tuple(sorted((name, value) for name, value in terms.items() if value))
    return terms, left[1] + sign * right[1]

  def scaled(expr, factor):
    return tuple((name, coefficient * factor) for name, coefficient in expr[0]), expr[1] * factor

  def operand_expr(operand):
    match = re.fullmatch(r"(x\d+)(?:,\s*lsl\s*#(\d+))?", operand.strip())
    assert match, f"unrecognized address operand: {operand}"
    return scaled(scalar(match[1]), 1 << int(match[2] or 0))

  selected_lines = set(source_loads)
  observed = []
  for raw in path_instructions:
    line = raw.split("//", 1)[0].strip()
    if not line:
      continue
    load = re.match(r"ld1([bhwd])\s+\{\s*z\d+\.[bhsd]\s*\},\s*p\d+/z,\s*\[([^\]]+)\]", line)
    if load and line in selected_lines:
      operands = [part.strip() for part in load[2].split(",")]
      base = operands[0]
      if root is None:
        root = base
        pointers[root] = ((), 0)
      assert base in pointers, f"input load is not rooted at A: {line}"
      address = pointers[base]
      if len(operands) > 1:
        address = combine(address, operand_expr(",".join(operands[1:])))
      observed.append((load[1], address, line))
      continue

    parts = line.split(None, 1)
    if len(parts) != 2:
      continue
    op, rest = parts
    args = [part.strip() for part in rest.split(",")]
    destination = args[0] if args else ""
    if op == "cmp" and len(args) == 2:
      right = scalar(args[1])
      left = scalar(args[0])
      comparison = "positive" if right == ((), 0) and left[1] > 0 and all(name == "vl" for name, _ in left[0]) else None
      continue
    recognized = op in {"cntb", "mov", "lsl", "add", "and", "csel"}
    if re.fullmatch(r"x\d+", destination) and not recognized:
      values.pop(destination, None)
      pointers.pop(destination, None)
    if op == "cntb" and len(args) == 1:
      values[args[0]] = ((("vl", 1),), 0)
    elif op == "mov" and len(args) >= 2 and re.fullmatch(r"x\d+", args[0]):
      values[args[0]] = scalar(args[1])
      if args[1] in pointers:
        pointers[args[0]] = pointers[args[1]]
    elif op == "lsl" and len(args) == 3:
      values[args[0]] = scaled(scalar(args[1]), 1 << int(args[2].lstrip("#")))
    elif op == "add" and len(args) >= 3 and re.fullmatch(r"x\d+", args[0]):
      right = scalar(args[2])
      if len(args) >= 4 and args[3].startswith("lsr #") and int(args[3][5:]) >= 32:
        right = ((), 0)  # Positive CNTB-sized values shifted by >=32 are zero.
      value = combine(scalar(args[1]), right)
      values[args[0]] = value
      if args[1] in pointers:
        pointers[args[0]] = combine(pointers[args[1]], right)
    elif op == "and" and len(args) == 3 and re.fullmatch(r"x\d+", args[0]):
      # CNTB and its integer multiples are 16-byte aligned on SVE; the
      # compiler's rounding-to-4-byte operation therefore preserves them.
      value = scalar(args[1])
      mask = int(args[2].lstrip("#"), 0)
      assert mask in {-4, 0xFFFFFFFFFFFFFFFC}, line
      assert value[1] % 4 == 0 and all(name == "vl" or coefficient % 4 == 0 for name, coefficient in value[0]), line
      values[args[0]] = value
    elif op == "csel" and len(args) == 4:
      if comparison == "positive" and args[3] in {"lt", "mi"}:
        chosen = args[2]
      elif comparison == "positive" and args[3] in {"ge", "pl"}:
        chosen = args[1]
      else:
        left, right = scalar(args[1]), scalar(args[2])
        chosen = args[1] if left == right else None
      if chosen in pointers:
        pointers[args[0]] = pointers[chosen]
      elif chosen is not None:
        values[args[0]] = scalar(chosen)
      comparison = None

  assert len(observed) == words, (len(observed), words, observed)
  lane_terms = {name: coefficient for name, coefficient in observed[0][1][0] if name != "vl"}
  assert len(lane_terms) == 1 and next(iter(lane_terms.values())) == 4, observed[0]
  lane = next(iter(lane_terms))
  for index, (_, address, line) in enumerate(observed):
    terms = {lane: 4}
    if index:
      terms["vl"] = index
    expected = (tuple(sorted(terms.items())), 0)
    assert address == expected, f"non-contiguous input group at word {index}: {line}: {address} != {expected}"


def _path_source_loads(path_instructions, *, include_byte_loads):
  """Follow address provenance in execution order, killing every redefined GPR."""
  values, pointers = {}, {}
  source_root = None
  comparison = None
  accesses = []

  def physical_gpr(register):
    match = re.fullmatch(r"[xw](\d+)", register)
    return f"x{match[1]}" if match else register

  def scalar(token):
    immediate = re.fullmatch(r"#(-?(?:0x[0-9a-fA-F]+|\d+))", token)
    if immediate:
      return ((), int(immediate[1], 0))
    if token in {"xzr", "wzr"}:
      return ((), 0)
    register = physical_gpr(token)
    return values.get(register, (((f"reg:{register}", 1),), 0))

  def combine(left, right, sign=1):
    terms = dict(left[0])
    for name, coefficient in right[0]:
      terms[name] = terms.get(name, 0) + sign * coefficient
    return tuple(sorted((name, coefficient) for name, coefficient in terms.items() if coefficient)), left[
      1
    ] + sign * right[1]

  def scaled(value, amount):
    return tuple((name, coefficient * amount) for name, coefficient in value[0]), value[1] * amount

  def effective_address(operands):
    result = pointers[physical_gpr(operands[0])]
    if len(operands) > 1:
      if re.fullmatch(r"z\d+\.[bhsd]", operands[1]):
        return combine(result, (((f"vector-index:{operands[1]}", 1),), 0))
      index = scalar(operands[1])
      if len(operands) > 2:
        shift = re.fullmatch(r"lsl\s*#(\d+)", operands[2])
        assert shift, f"unsupported load address modifier: {operands}"
        index = scaled(index, 1 << int(shift[1]))
      result = combine(result, index)
    return result

  for raw in path_instructions:
    line = raw.split("//", 1)[0].strip()
    if not line:
      continue
    load = re.match(r"ld1([bhwd])\s+\{\s*z\d+\.[bhsd]\s*\},\s*p\d+/z,\s*\[([^\]]+)\]", line)
    if load:
      operation = load[1]
      operands = [part.strip() for part in load[2].split(",")]
      if source_root is None and operation == "w":
        source_root = physical_gpr(operands[0])
        pointers[source_root] = ((), 0)
      if physical_gpr(operands[0]) in pointers and (include_byte_loads or operation == "w"):
        accesses.append((operation, effective_address(operands), line))
      continue

    parts = line.split(None, 1)
    if len(parts) != 2:
      continue
    operation, rest = parts
    arguments = [part.strip() for part in rest.split(",")]
    if operation == "cmp" and len(arguments) == 2:
      value, zero = scalar(arguments[0]), scalar(arguments[1])
      comparison = "positive" if zero[1] == 0 and value[1] > 0 and all(name == "vl" for name, _ in value[0]) else None
      continue

    destination = arguments[0] if arguments else ""
    if re.fullmatch(r"[xw]\d+", destination):
      # Wn writes the low word and zeroes Xn's high word, so both spellings
      # invalidate the same physical register's pointer/value provenance.
      destination = physical_gpr(destination)
      values.pop(destination, None)
      pointers.pop(destination, None)

    if operation == "cntb" and len(arguments) == 1:
      values[destination] = ((("vl", 1),), 0)
    elif operation == "mov" and len(arguments) >= 2:
      source = physical_gpr(arguments[1])
      if source in pointers:
        pointers[destination] = pointers[source]
      else:
        values[destination] = scalar(arguments[1])
    elif operation == "lsl" and len(arguments) == 3:
      values[destination] = scaled(scalar(arguments[1]), 1 << int(arguments[2].lstrip("#")))
    elif operation == "add" and len(arguments) >= 3:
      right = scalar(arguments[2])
      if len(arguments) >= 4 and arguments[3].startswith("lsr #"):
        shift = int(arguments[3][5:])
        if shift >= 32 and all(name == "vl" for name, _ in right[0]) and right[1] >= 0:
          right = ((), 0)
        else:
          right = (((f"shift:{arguments[2]}:{shift}", 1),), 0)
      if arguments[1] in pointers:
        pointers[destination] = combine(pointers[arguments[1]], right)
      else:
        values[destination] = combine(scalar(arguments[1]), right)
    elif operation == "and" and len(arguments) == 3:
      value, mask = scalar(arguments[1]), int(arguments[2].lstrip("#"), 0)
      if mask in {-4, 0xFFFFFFFFFFFFFFFC}:
        if value[1] % 4 == 0 and all(name == "vl" or coefficient % 4 == 0 for name, coefficient in value[0]):
          values[destination] = value
        else:
          values[destination] = (((f"and4:{arguments[1]}", 1),), 0)
      else:
        values[destination] = (((f"and:{arguments[1]}:{mask}", 1),), 0)
    elif operation == "csel" and len(arguments) == 4:
      condition = arguments[3]
      if comparison == "positive" and condition in {"lt", "mi"}:
        chosen = arguments[2]
      elif comparison == "positive" and condition in {"ge", "pl"}:
        chosen = arguments[1]
      else:
        left, right = scalar(arguments[1]), scalar(arguments[2])
        chosen = arguments[1] if left == right else None
      if chosen is not None:
        if chosen in pointers:
          pointers[destination] = pointers[chosen]
        else:
          values[destination] = scalar(chosen)
      comparison = None
    elif operation == "asr" and len(arguments) == 3:
      values[destination] = (((f"asr:{arguments[1]}:{arguments[2]}", 1),), 0)
    elif operation not in {
      "cmp",
      "b",
      "b.eq",
      "b.ne",
      "b.lt",
      "b.le",
      "b.gt",
      "b.ge",
      "b.lo",
      "b.hs",
      "ret",
    } and re.fullmatch(r"[xw]\d+", destination):
      # An unmodeled GPR definition invalidates previous provenance; never
      # let a stale alias survive register reuse.
      values[destination] = (((f"unknown:{destination}", 1),), 0)

  return accesses


def _assert_indexed_source_groups(function, header, path_instructions, source_loads, *, vl_bytes, words):
  """Prove predicated SVE gathers partition one loop tile without overlap."""
  lanes = vl_bytes // 4
  vector_values = {}

  def physical_z(register):
    match = re.fullmatch(r"z(\d+)\.[bhsd]", register.strip())
    return f"z{match[1]}" if match else register.strip()

  def immediate(token):
    match = re.fullmatch(r"#(-?(?:0x[0-9a-fA-F]+|\d+))", token.strip())
    assert match, f"expected immediate, got {token}"
    return int(match[1], 0)

  # Recover vector offsets from their reaching definitions before the loop.
  # Any unrecognized write kills the old value, so register reuse cannot make
  # an earlier INDEX/INCV definition appear to reach a later load.
  indexed_load = re.compile(r"ld1w\s+\{\s*z\d+\.s\s*\},\s*(p\d+)/z,\s*\[(x\d+),\s*(z\d+\.s),\s*sxtw\]")
  remaining = list(source_loads)
  accesses = []

  def process(raw):
    line = raw.split("//", 1)[0].strip()
    if not line:
      return
    match = indexed_load.fullmatch(line)
    if match and line in remaining:
      predicate, base, index_register = match.groups()
      index_key = physical_z(index_register)
      assert index_key in vector_values, f"no reaching INDEX/INCV for {index_register}: {line}"
      accesses.append((predicate, base, vector_values[index_key], line))
      remaining.remove(line)
    line = raw.split("//", 1)[0].strip()
    parts = line.split(None, 1)
    if len(parts) != 2:
      return
    operation, rest = parts
    arguments = [part.strip() for part in rest.split(",")]
    destination = arguments[0] if arguments else ""
    destination_registers = re.findall(r"z\d+\.[bhsd]", destination)
    if not destination_registers:
      return
    destination_register = destination_registers[0]
    destination_key = physical_z(destination_register)
    previous = vector_values.get(destination_key)
    for register in destination_registers:
      vector_values.pop(physical_z(register), None)
    if operation == "index" and len(arguments) == 3 and destination_register.endswith(".s"):
      start, step = immediate(arguments[1]), immediate(arguments[2])
      vector_values[destination_key] = tuple(start + index * step for index in range(lanes))
    elif operation == "mov" and len(arguments) == 2 and re.fullmatch(r"z\d+\.[bhsd]", destination_register):
      source = physical_z(arguments[1])
      if source in vector_values:
        vector_values[destination_key] = vector_values[source]
    elif operation == "incw" and destination_register.endswith(".s") and previous is not None:
      amount = lanes
      if len(arguments) >= 3 and arguments[1] == "all" and arguments[2].startswith("mul #"):
        amount *= immediate(arguments[2].removeprefix("mul "))
      vector_values[destination_key] = tuple(value + amount for value in previous)
    elif operation == "lsl" and len(arguments) == 3 and destination_register.endswith(".s"):
      source = physical_z(arguments[1])
      if source in vector_values:
        amount = 1 << immediate(arguments[2])
        vector_values[destination_key] = tuple(value * amount for value in vector_values[source])

  for raw in function[: header.start()].splitlines():
    process(raw)
  for raw in path_instructions:
    process(raw)

  assert not remaining, f"indexed source loads were not reached on the selected loop path: {remaining}"
  assert len(accesses) == words, (len(accesses), words, source_loads)
  predicates = {predicate for predicate, _, _, _ in accesses}
  bases = {base for _, base, _, _ in accesses}
  assert len(predicates) == 1, f"indexed groups use different masks: {accesses}"
  assert len(bases) == 1, f"indexed groups use different source bases: {accesses}"
  expected = tuple(range(0, words * lanes * 4, 4))
  all_offsets = [offset for _, _, vector, _ in accesses for offset in vector]
  assert tuple(sorted(all_offsets)) == expected, (all_offsets, expected)

  # The base advances by exactly the span partitioned by the indexed groups.
  # Thus the same active predicate lane cannot be reread by the next loop
  # iteration, and every active lane in the tile has one corresponding load.
  base = next(iter(bases))
  increments = []
  for line in path_instructions:
    match = re.fullmatch(rf"add\s+{base},\s*{base},\s*#(\d+)", line.strip())
    if match:
      increments.append(int(match[1]))
  assert increments == [words * vl_bytes], (base, increments, words * vl_bytes)


def _assert_direct_source_step(path_instructions, source_loads, *, vl_bytes, words):
  first = re.search(r"\[(x\d+),\s*(x\d+),\s*lsl\s*#2\]", source_loads[0])
  assert first, f"expected an element-indexed source loop: {source_loads[0]}"
  index = first[2]
  increments = []
  for line in path_instructions:
    match = re.fullmatch(rf"add\s+{index},\s*{index},\s*#(\d+)", line.strip())
    if match:
      increments.append(int(match[1]))
  expected = words * (vl_bytes // 4)
  assert increments == [expected], (index, increments, expected)


def test_source_memory_load_accounting_accepts_byte_load_spelling_only_for_input_alias():
  code = """
  ld1w { z0.s }, p0/z, [x20, x8, lsl #2]
  add x11, x20, x9
  ld1b { z1.b }, p2/z, [x11, x10]
  ld1b { z2.b }, p2/z, [x11, x12]
  ld1b { z3.b }, p2/z, [x11, x13]
  ld1b { z4.b }, p3/z, [x4, x6]
  """
  reads = _source_memory_loads(code)
  assert [line.split()[0] for line in reads] == ["ld1w", "ld1b", "ld1b", "ld1b"]
  assert_full_sve_memory_words("ptrue p0.s\nptrue p2.b\n" + "\n".join(reads), kind="ld1", words=4)


def test_source_memory_group_requires_contiguous_vector_byte_offsets():
  lines = [
    "ld1w { z0.s }, p0/z, [x20, x8, lsl #2]",
    "lsl x9, x8, #2",
    "add x11, x20, x9",
    "cntb x10",
    "and x10, x10, #0xfffffffffffffffc",
    "ld1b { z1.b }, p1/z, [x11, x10]",
    "cntb x12",
    "lsl x12, x12, #1",
    "and x12, x12, #0xfffffffffffffffc",
    "ld1b { z2.b }, p2/z, [x11, x12]",
    "mov x13, x12",
    "cntb x12",
    "and x12, x12, #0xfffffffffffffffc",
    "add x10, x12, x13",
    "ld1b { z3.b }, p3/z, [x11, x10]",
  ]
  _assert_contiguous_source_groups(lines, [lines[0], lines[5], lines[9], lines[14]], words=4)
  non_contiguous = [line for line in lines if line != "add x10, x12, x13"]
  with pytest.raises(AssertionError, match="non-contiguous input group"):
    _assert_contiguous_source_groups(non_contiguous, [lines[0], lines[5], lines[9], lines[14]], words=4)


def test_byte_load_second_group_requires_disjoint_complete_a_coverage():
  lines = [
    "ptrue p0.b",
    "cntb x10",
    "lsl x9, x8, #2",
    "ld1w { z0.s }, p0/z, [x20, x8, lsl #2]",
    "add x11, x20, x9",
    "ld1b { z1.b }, p0/z, [x11, x10]",
  ]
  _assert_contiguous_source_groups(lines, [lines[3], lines[5]], words=2)
  assert_full_sve_memory_words("\n".join(lines), kind="ld1", words=2)

  # A same-sized load from a different pointer is not the second A group.
  wrong_base = lines.copy()
  wrong_base[4] = "add x11, x30, x9"
  with pytest.raises(AssertionError, match="not rooted at A"):
    _assert_contiguous_source_groups(wrong_base, [wrong_base[3], wrong_base[5]], words=2)

  # Re-reading the first byte range is overlap, not complete group coverage.
  overlap = lines.copy()
  overlap[5:6] = ["mov x12, xzr", "ld1b { z1.b }, p0/z, [x11, x12]"]
  with pytest.raises(AssertionError, match="non-contiguous input group"):
    _assert_contiguous_source_groups(overlap, [overlap[3], overlap[6]], words=2)

  # Omitting the second physical access is rejected; LD1W instruction count
  # alone cannot stand in for a two-group conversion.
  with pytest.raises(AssertionError):
    _assert_contiguous_source_groups(lines, [lines[3]], words=2)


def test_byte_group_rejects_partial_predicate_even_when_offsets_are_complete():
  body = "whilelo p0.s, xzr, x8\nld1w { z0.s }, p0/z, [x20]\nptrue p1.b\nld1b { z1.b }, p1/z, [x20, x9]"
  with pytest.raises(AssertionError):
    assert_full_sve_memory_words(body, kind="ld1", words=2)


def test_source_memory_alias_tracking_kills_redefined_pointer_registers():
  code = """
  ld1w { z0.s }, p0/z, [x20]
  add x11, x20, x9
  ld1w { z1.s }, p0/z, [x11]
  mov x11, x19
  ld1w { z2.s }, p0/z, [x11]
  """
  accesses = _path_source_loads(code.splitlines(), include_byte_loads=True)
  assert [line for _, _, line in accesses] == [
    "ld1w { z0.s }, p0/z, [x20]",
    "ld1w { z1.s }, p0/z, [x11]",
  ]


def test_indexed_source_group_checks_exact_lane_offsets_and_iteration_advance():
  function = """\
.Lpreheader:
  index z0.s, #0, #1
  mov z1.d, z0.d
  mov z2.d, z0.d
  lsl z16.s, z0.s, #2
  incw z0.s, all, mul #3
  incw z1.s
  incw z2.s, all, mul #2
  lsl z20.s, z1.s, #2
  lsl z21.s, z2.s, #2
  lsl z22.s, z0.s, #2
.Lheader: // =>This Inner Loop Header: Depth=1
  ld1w { z0.s }, p0/z, [x28, z16.s, sxtw]
  ld1w { z0.s }, p0/z, [x28, z20.s, sxtw]
  ld1w { z0.s }, p0/z, [x28, z21.s, sxtw]
  ld1w { z0.s }, p0/z, [x28, z22.s, sxtw]
  add x28, x28, #64
  b .Lheader
"""
  header = re.search(r"^\.Lheader:", function, re.MULTILINE)
  assert header
  loads = [line.strip() for line in function.splitlines() if line.strip().startswith("ld1w")]
  _assert_indexed_source_groups(function, header, function.splitlines(), loads, vl_bytes=16, words=4)
  non_contiguous = function.replace("lsl z21.s, z2.s, #2", "lsl z21.s, z1.s, #2")
  header = re.search(r"^\.Lheader:", non_contiguous, re.MULTILINE)
  with pytest.raises(AssertionError):
    _assert_indexed_source_groups(non_contiguous, header, non_contiguous.splitlines(), loads, vl_bytes=16, words=4)
  clobbered = function.replace(
    "  ld1w { z0.s }, p0/z, [x28, z21.s, sxtw]",
    "  mov z21.d, z19.d\n  ld1w { z0.s }, p0/z, [x28, z21.s, sxtw]",
  )
  header = re.search(r"^\.Lheader:", clobbered, re.MULTILINE)
  with pytest.raises(AssertionError, match="no reaching INDEX/INCV"):
    _assert_indexed_source_groups(clobbered, header, clobbered.splitlines(), loads, vl_bytes=16, words=4)


def _iteration_input_reads(assembly, blocks, header, *, include_byte_loads):
  """Maximum source-vector load count along one CFG iteration."""
  loop_header = header.group("label").removeprefix(".L")
  instructions, labels, members = [], {}, set()
  for block in blocks:
    labels[block.group("label")] = len(instructions)
    member = block is header or f"Header={loop_header} " in block.group("comment")
    for line in block.group("body").splitlines():
      line = line.split("//", 1)[0].strip()
      if not line or line.startswith("."):
        continue
      index = len(instructions)
      instructions.append((line.split(None, 1), line))
      if member:
        members.add(index)
  first = labels[header.group("label")]
  best, best_path, pending, maximum, maximum_path = {first: 0}, {first: ()}, [first], 0, ()
  updates = 0
  while pending:
    index = pending.pop()
    (operation, *arguments), _instruction = instructions[index]
    path = best_path[index] + (index,)
    path_instructions = [instructions[path_index][1] for path_index in path]
    count = len(_path_source_loads(path_instructions, include_byte_loads=include_byte_loads))
    target = arguments[0].split(",")[-1].strip() if arguments else ""
    if operation == "b":
      successors = (labels.get(target),)
    elif operation.startswith("b.") or operation in {"cbz", "cbnz", "tbz", "tbnz"}:
      successors = (labels.get(target), index + 1)
    elif operation == "ret":
      successors = ()
    else:
      successors = (index + 1,)
    for successor in successors:
      if successor is None or successor not in members or successor == first:
        if count > maximum:
          maximum, maximum_path = count, path
      elif count > best.get(successor, -1):
        best[successor] = count
        best_path[successor] = path
        pending.append(successor)
        updates += 1
        assert updates <= len(instructions) ** 2, "unexpected nested input-reading cycle"
  path_instructions = [instructions[index][1] for index in maximum_path]
  path_loads = [line for _, _, line in _path_source_loads(path_instructions, include_byte_loads=include_byte_loads)]
  return maximum, path_loads, path_instructions


def _assembly_functions(assembly):
  starts = list(re.finditer(r"^\s*\.type\s+[^,\n]+,\s*@function\b", assembly, re.MULTILINE))
  for index, start in enumerate(starts):
    end = re.search(r"^\s*\.size\b", assembly[start.start() :], re.MULTILINE)
    stop = start.start() + end.end() if end else starts[index + 1].start() if index + 1 < len(starts) else len(assembly)
    yield assembly[start.start() : stop]


@pytest.mark.parametrize("vl,svl", ((16, 64), (32, 16), (64, 32)))
@pytest.mark.parametrize("shared", (False, True))
@pytest.mark.parametrize("masked", (False, True))
def test_captured_cast_real_arm_cross_keeps_rounding_without_extra_input_reads(tmp_path, vl, svl, shared, masked):
  artifact = CPUBackend().lower(
    _rounding_template(shared, masked).ir,
    (),
    target="SVE2",
    options={"_vector_state": VectorState(vl, svl), "cpu_math_chunk_pow": 0},
  )
  assembly = _cross(tmp_path, artifact.source, _rounding_wrapper, vl=vl, svl=svl)
  # Inspect the actual converting hot loop, not the wrapper (which can call
  # __kernel__) or the allocation/copy path of the conservative alias guard.
  # Its two-word group has two F32 words when shared, or four physical F32
  # words when the converting store selects F16 as the iteration dtype.
  converted_loops = []
  for function in _assembly_functions(assembly):
    blocks = list(
      re.finditer(
        r"^(?P<label>\.LBB\d+_\d+):(?P<comment>[^\n]*)\n(?P<body>.*?)"
        r"(?=^\.LBB\d+_\d+:|^\.Lfunc_end\d+:|^\s*\.size\b)",
        function,
        re.MULTILINE | re.DOTALL,
      )
    )
    for header in blocks:
      if "Inner Loop Header: Depth=1" not in header.group("comment"):
        continue
      loop_header = header.group("label").removeprefix(".L")
      # Restrict to one function: LLVM reuses .LBB labels across functions.
      members = [
        block.group("body") for block in blocks if block is header or f"Header={loop_header} " in block.group("comment")
      ]
      code = "\n".join(members)
      if not re.search(r"^\s*fcvt\s+z\d+\.h", code, re.MULTILINE):
        continue
      input_reads, source_loads, path_instructions = _iteration_input_reads(
        function, blocks, header, include_byte_loads=True
      )
      converted_loops.append((code, input_reads, source_loads, path_instructions, function, header))
  assert converted_loops
  expected_reads = 2 if shared else 4
  hot_loops = [loop for loop in converted_loops if loop[1] >= expected_reads]
  assert len(hot_loops) == (1 if masked else 2), [loop[1] for loop in converted_loops]
  assert all(loop[1] == expected_reads for loop in hot_loops), [loop[1] for loop in hot_loops]

  for code, _input_reads, source_loads, path_instructions, function, header in hot_loops:
    # Snapshot and original-storage arms are mutually exclusive; inspect each
    # conversion loop independently and never merge their static instructions.
    if any(re.search(r"\[x\d+,\s*z\d+\.s,\s*sxtw\]", line) for line in source_loads):
      _assert_indexed_source_groups(
        function, header, path_instructions, source_loads, vl_bytes=vl, words=expected_reads
      )
    else:
      _assert_contiguous_source_groups(path_instructions, source_loads, words=expected_reads)
      _assert_direct_source_step(path_instructions, source_loads, vl_bytes=vl, words=expected_reads)
      access_predicates = {match[1] for line in source_loads if (match := re.search(r"\},\s*(p\d+)/z,", line))}
      function_before = function[: header.start()]
      reaching_defs = {}
      predicate_writer = re.compile(r"\s*(\w+)\s+(p\d+)(?:\.([bhsd]))?\b(.*)")
      for line in function_before.splitlines():
        match = predicate_writer.match(line)
        if match and match[2] in access_predicates:
          reaching_defs[match[2]] = line.strip()
      access_body = "\n".join(
        list(reaching_defs.values())
        + [line for line in path_instructions if predicate_writer.match(line) or line in source_loads]
      )
      assert_full_sve_memory_words(access_body, kind="ld1", words=expected_reads)
    assert re.search(r"^\s*fcvt\s+z\d+\.h", code, re.MULTILINE)
    assert re.search(r"^\s*fcvt\s+z\d+\.s", code, re.MULTILINE)
    assert not re.search(r"^\s*smstart\b", code, re.MULTILINE)
    if not masked:
      assert not re.search(r"^\s*(?:bl|blr)\b", code, re.MULTILINE)
    else:
      assert not re.search(r"^\s*bl\s+\S*(?:load_convert|store_convert|convert)\S*", code, re.MULTILINE)


def _values(size):
  values = (1 + (np.arange(size) % 19 + 0.5) / 1024).astype("float32")
  values[1::2] *= -1
  values[::17] = 0.0
  values[1::17] = -0.0
  return values


def _arguments(size, layout, masked, *, count=3):
  stride = {"separate": 1, "overlap": 1, "shifted": 1, "negative": -2}[layout]
  backing = np.full(max(16, size * abs(stride) + 16), 19.25, dtype="float32")
  a = backing[4 : 4 + size * abs(stride) : abs(stride)]
  if stride < 0:
    a = a[::-1]
  a[:] = _values(size)
  valid = (np.arange(size) % 4 != 2).astype("int8")
  if layout in ("overlap", "shifted"):
    c_backing = backing.view("float16")
    start = 8 + int(layout == "shifted")
    c = c_backing[start : start + size]
  else:
    c_backing = np.full(size * abs(stride) + 8, -29.5, dtype="float16")
    c = c_backing[4 : 4 + size * abs(stride) : abs(stride)]
    if stride < 0:
      c = c[::-1]
  d, e = np.full(size, -17.5, dtype="float32"), np.full(size, -23.5, dtype="float32")
  # Simulate byte overlap on a distinct backing copy. The oracle is computed
  # before invoking the kernel and never reads its potentially damaged A.
  expected_backing, expected_c_backing = backing.copy(), c_backing.copy()
  if np.shares_memory(a, c):
    expected_c_backing = expected_backing.view("float16")
  expected_a = np.ndarray(
    a.shape, dtype=a.dtype, buffer=expected_backing, offset=a.ctypes.data - backing.ctypes.data, strides=a.strides
  )
  expected_c = np.ndarray(
    c.shape, dtype=c.dtype, buffer=expected_c_backing, offset=c.ctypes.data - c_backing.ctypes.data, strides=c.strides
  )
  captured = np.where(valid != 0, a.copy(), np.float32(-1.5)) if masked else a.copy()
  rounded = captured.astype("float16")
  expected_c[:] = rounded
  expected_d = rounded.astype("float32")
  expected_e = expected_a.copy()
  return (
    (a, valid, c, d, e, count, True),
    (backing, c_backing),
    (expected_backing, expected_c_backing, expected_d, expected_e),
  )


@pytest.mark.parametrize("layout", ("separate", "overlap", "shifted", "negative"))
@pytest.mark.parametrize("masked", (False, True))
def test_byte_overlap_capture_oracle_is_independent(layout, masked):
  args, storages, expected = _arguments(7, layout, masked)
  a, valid, _, _, _, _, _ = args
  np.testing.assert_array_equal(a, _values(7))
  narrowed = np.where(valid != 0, a, -1.5) if masked else a
  np.testing.assert_array_equal(expected[2], narrowed.astype("float16").astype("float32"))
  if layout in ("overlap", "shifted"):
    assert not np.array_equal(expected[0], storages[0])
    assert not np.array_equal(expected[3], a)
  else:
    np.testing.assert_array_equal(expected[3], a)


@pytest.fixture(scope="module")
def native_cast_captures(tmp_path_factory):
  directory = tmp_path_factory.mktemp("cast-captures")
  compiler = make_compiler(
    target="Native", jobs=4, output="quiet", cache_dir=directory / "cache", build_dir=directory / "build"
  )
  kernels = {}
  with vecops.collect_compile_requests() as collector:
    for kind in KINDS:
      for masked in (False, True):
        kernel = T.compile(capture_template(kind, masked), compiler=compiler)
        kernel.operator(
          vecops.TensorMeta((129,), strides=(1,), dtype="float32"),
          vecops.TensorMeta((129,), strides=(1,), dtype="int8"),
          vecops.TensorMeta((129,), strides=(1,), dtype="float16", access=vecops.TensorAccess.output),
          vecops.TensorMeta((129,), strides=(1,), dtype="float32", access=vecops.TensorAccess.output),
          vecops.TensorMeta((129,), strides=(1,), dtype="float32", access=vecops.TensorAccess.output),
          3,
          True,
          **kernel._core.specialization_values,
        )
        kernels[kind, masked] = kernel
  assert len(collector.requests) == len(kernels) == 8
  vecops.compile_batch(collector.requests, parallelism=4, execution_parallelism=1)
  return kernels


@pytest.mark.parametrize("kind", KINDS)
@pytest.mark.parametrize("masked", (False, True))
def test_cast_capture_native_alias_effects_helper_and_loop(native_cast_captures, kind, masked):
  for size in (1, 7, 16, 17, 31, 32, 33, 129):
    for layout in ("separate", "overlap", "shifted", "negative"):
      arguments, actual_storage, expected = _arguments(size, layout, masked)
      native_cast_captures[kind, masked](*arguments)
      context = f"kind={kind}, masked={masked}, size={size}, layout={layout}"
      for actual, wanted in zip((*actual_storage, arguments[3], arguments[4]), expected, strict=True):
        np.testing.assert_array_equal(
          actual.view(f"u{actual.itemsize}"), wanted.view(f"u{wanted.itemsize}"), err_msg=context
        )
