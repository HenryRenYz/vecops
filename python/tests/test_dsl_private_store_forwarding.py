# SPDX-FileCopyrightText: 2026 HenryRenYz and vecops contributors
# SPDX-License-Identifier: MIT
"""Eliminate private round trips without discarding conversion or read history."""

from dataclasses import replace

import numpy as np
import pytest
import vecops.language as T
from dsl_test_utils import make_compiler
from vecops.language._compiler import ir as I
from vecops.language._compiler.ir.traversal import nodes
from vecops.language._compiler.passes.kernel.forwarding import forward_private_stores
from vecops.language._compiler.passes.manager import PassContext


def example():
  shape = (I.literal(17),)
  temporary = I.View("temporary", "float32", shape, (I.literal(1),), I.literal(0), owned=True)
  source, output = replace(temporary, storage="A", owned=False), replace(temporary, storage="C", owned=False)
  original = I.Expr(I.TileType("float32", shape), "load", data=source)
  value = I.Expr(original.type, "add", (original, I.literal(1)))
  read = I.Expr(original.type, "load", data=temporary)
  producer = I.Store(temporary, value)
  consumer = I.Store(output, I.Expr(original.type, "mul", (read, I.literal(2))))
  return temporary, original, producer, read, consumer


def run(statements, **options):
  ir = I.KernelIR(I.Block(statements), {}, {})
  result = forward_private_stores(ir, PassContext("kernel-cleanup", options=options))
  return ir, result.changed


def test_forwarding_retains_exact_producer_read_identity():
  temporary, original, producer, read, consumer = example()
  ir, changed = run([I.Allocate(temporary), producer, I.BindTile(read), consumer])
  assert changed
  assert producer not in ir.body.statements
  assert read not in list(nodes(ir.body))
  assert ir.body.statements[-1].value.args[0] is producer.value
  assert any(value is original for value in nodes(ir.body))


@pytest.mark.parametrize(
  "boundary",
  (
    "public",
    "masked-write",
    "masked-read",
    "masked-output",
    "narrowing",
    "partial",
    "strided",
    "early-capture",
    "second-consumer",
    "second-write",
    "fence",
    "control",
    "metadata",
    "opaque",
    "duplicate-statement",
    "broadcast",
  ),
)
def test_forwarding_requires_a_complete_private_identity_round_trip(boundary):
  temporary, _original, producer, read, consumer = example()
  prefix, middle, suffix = [], [], []
  if boundary == "public":
    temporary = replace(temporary, owned=False)
    producer = replace(producer, destination=temporary)
  elif boundary == "masked-write":
    producer = replace(producer, mask=I.literal(True))
  elif boundary == "masked-read":
    read = replace(read, args=(I.literal(True), I.literal(0)))
    consumer = replace(consumer, value=replace(consumer.value, args=(read, I.literal(2))))
  elif boundary == "masked-output":
    consumer = replace(consumer, mask=I.literal(True))
  elif boundary == "narrowing":
    producer = replace(producer, value=replace(producer.value, type=I.TileType("float64", temporary.shape)))
  elif boundary == "partial":
    producer = replace(producer, destination=replace(temporary, shape=(I.literal(16),)))
  elif boundary == "strided":
    producer = replace(producer, destination=replace(temporary, strides=(I.literal(2),)))
  elif boundary == "early-capture":
    prefix = [I.BindTile(read)]
  elif boundary == "second-consumer":
    suffix = [replace(consumer, destination=replace(consumer.destination, storage="D"))]
  elif boundary == "second-write":
    suffix = [producer]
  elif boundary == "fence":
    middle = [I.Control('asm volatile("" ::: "memory")')]
  elif boundary == "control":
    consumer = I.If(I.literal(True), I.Block([consumer]), I.Block())
  elif boundary == "metadata":
    suffix = [I.Assign("metadata", I.Expr(I.ScalarType("int64"), "raw", data=(read,)))]
  elif boundary == "opaque":

    class Hidden:
      def __getattribute__(self, name):
        raise AssertionError("opaque ownership must not run custom access")

    suffix = [I.Assign("metadata", I.Expr(I.ScalarType("int64"), "raw", data=Hidden()))]
  elif boundary == "duplicate-statement":
    suffix = [I.If(I.literal(True), I.Block([consumer]), I.Block())]
  elif boundary == "broadcast":
    consumer = replace(consumer, destination=replace(consumer.destination, shape=(I.literal(2), *temporary.shape)))
  ir, changed = run([I.Allocate(temporary), *prefix, producer, *middle, consumer, *suffix])
  assert not changed
  assert producer in ir.body.statements


def test_forwarding_can_be_disabled_and_validates_its_option():
  temporary, _, producer, _, consumer = example()
  assert not run([I.Allocate(temporary), producer, consumer], private_store_forwarding=False)[1]
  with pytest.raises(TypeError, match="private_store_forwarding"):
    run([I.Allocate(temporary), producer, consumer], private_store_forwarding=1)


@pytest.mark.parametrize(
  "where", ("container-cycle", "node-cycle", "materialization", "binding", "repeated-allocation", "raw-code")
)
def test_forwarding_rejects_cycles_and_nonlocal_storage_owners(where):
  temporary, _, producer, _, consumer = example()
  ir = I.KernelIR(I.Block([I.Allocate(temporary), producer, consumer]), {}, {})
  if where == "container-cycle":
    cyclic = []
    cyclic.append(cyclic)
    ir.body.statements.append(I.Assign("metadata", I.Expr(I.ScalarType("int64"), "raw", data=cyclic)))
  elif where == "node-cycle":
    cyclic = I.Expr(I.ScalarType("int64"), "identity")
    object.__setattr__(cyclic, "args", (cyclic,))
    ir.body.statements.append(I.Assign("metadata", cyclic))
  elif where == "materialization":
    ir.materializations[123] = temporary
  elif where == "binding":
    ir.bindings["escaped"] = I.Expr(I.TileType("float32", temporary.shape), "load", data=temporary)
  elif where == "raw-code":
    ir.body.statements.append(I.Control("consume(temporary.data());"))
  else:
    ir.body.statements.append(ir.body.statements[0])
  assert not forward_private_stores(ir, PassContext("kernel-cleanup")).changed
  assert producer in ir.body.statements


@pytest.mark.parametrize("width", (1, 17, 257))
def test_native_forwarding_keeps_overlapping_inputs_historical(tmp_path, width):
  N = T.dynamic(lower=0, upper=512)

  @T.template
  def program(A: T.Tensor((N, width), dtype="float32"), C: T.Tensor((N, width), dtype="float32", access="output")):
    for row in T.serial(N, unroll=1):
      temporary = T.alloc((width,), dtype="float32")
      T.store(temporary, T.load(A[row, :]) + 1)
      T.store(C[row, :], T.load(temporary) * 2)

  compiler = make_compiler(cache_dir=tmp_path / "cache", build_dir=tmp_path / "build", jobs=2)
  compiled = T.compile(program, compiler=compiler)
  assert not any(isinstance(node, I.Allocate) for node in nodes(compiled.kernel_ir.body))
  # One row isolates the semantics of this round trip from inter-row writes.
  # Shifted views force the substituted A read to acquire real alias protection.
  for rows in (0, 1):
    for shift in (-1, 0, 1):
      for poison in (90, 195):
        owner = np.empty((width + 34) * 4, dtype=np.uint8)
        owner.fill(poison)
        source = np.ndarray((rows, width), dtype=np.float32, buffer=owner, offset=17 * 4, strides=(width * 4, 4))
        output = np.ndarray(
          (rows, width), dtype=np.float32, buffer=owner, offset=(17 + shift) * 4, strides=(width * 4, 4)
        )
        source[:] = np.linspace(-1, 1, width, dtype=np.float32)
        expected = (source.copy() + 1) * 2
        before = owner.copy()
        compiled(source, output)
        np.testing.assert_array_equal(output, expected)
        mask = np.zeros(owner.size, dtype=bool)
        start = (17 + shift) * 4
        mask[start : start + output.nbytes] = True
        np.testing.assert_array_equal(owner[~mask], before[~mask])
