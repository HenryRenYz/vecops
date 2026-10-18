# SPDX-FileCopyrightText: 2026 HenryRenYz and vecops contributors
# SPDX-License-Identifier: MIT
"""A fused converting read must not also execute its unused source read."""

import re

import numpy as np
import pytest
import vecops.language as T
from dsl_test_utils import make_compiler
from vecops.language._compiler import ir as I
from vecops.language._compiler.backends.cpu.backend import CPUBackend
from vecops.language._compiler.backends.cpu.cpp.emitter import CPUEmitter
from vecops.language._compiler.ir.traversal import nodes, walk_unique
from vecops.language._compiler.lowering.tile_to_vector import lower_region


def converting_template(masked=False):
  count = T.dynamic(lower=0)
  stride = T.dynamic()

  @T.template
  def kernel(A: T.Tensor((count,), (stride,), "int8"),
             Valid: T.Tensor((count,), dtype="int8"),
             C: T.Tensor((count,), dtype="uint32", access="output")):
    if masked:
      original = T.load(A, mask=T.load(Valid) != 0, other=-7)
    else:
      original = T.load(A)
    widened = T.cast(original, "int32")
    T.store(C, T.bitcast(widened, "uint32") + T.cast(3, "uint32"))

  return kernel


@pytest.mark.parametrize("masked", [False, True])
@pytest.mark.parametrize("full", [False, True])
@pytest.mark.parametrize("shared", [False, True])
def test_converting_schedule_has_no_separate_unused_byte_read(masked, full, shared):
  artifact = CPUBackend().lower(converting_template(masked).ir, (), target="AVX512")
  store = next(s for s in nodes(artifact.kernel.body) if isinstance(s, I.Store) and s.destination.storage == "C")
  cast = next(e for e in walk_unique(store.value) if e.op == "cast" and e.args[0].op == "load")
  roots = (cast, cast.args[0]) if shared else (cast,)
  region = lower_region(cast.type.shape, "int32", roots, manager=artifact.lowered.manager)
  emitter = CPUEmitter(artifact.backend_ir)
  tag = emitter.state_tag("int32", True, 0)
  emitter.full_vector = full
  emitter.emit_operations(region, (tag,), (("0",),), ("16",), full=full)
  # Inspect the closed vector schedule independently of the kernel's required
  # byte-preserving alias snapshots. Each A read here already produces i32.
  reads = [line for line in emitter.lines if "load_convert(" in line and "A.data()" in line]
  assert len(reads) == 1
  if shared:
    # A second consumer needs the original byte register. Keep one byte load
    # and convert that register rather than rereading it at the cast.
    assert re.search(r"load_convert\(vecops::vec::ScalableTag<std::int8_t, -2>\{\}", reads[0])
    assert any("vecops::vec::convert(" in line for line in emitter.lines)
  else:
    assert re.search(r"load_convert\(vecops::vec::ScalableTag<std::int32_t, 0>\{\}", reads[0])


@pytest.mark.parametrize("masked", [False, True])
def test_native_converting_read_strides_masks_empty_and_tails(masked, tmp_path):
  kernel = T.compile(converting_template(masked), compiler=make_compiler(
      jobs=2, cache_dir=tmp_path / "cache", build_dir=tmp_path / "build"))
  for count in [0, 1, 15, 16, 17, 31, 32, 33, 257]:
    for stride in [1, 3, -2]:
      storage = (np.arange(max(1, count * abs(stride)), dtype=np.int64) % 256 - 128).astype(np.int8)
      source = storage[::stride][:count]
      valid = (np.arange(count) % 3 != 0).astype(np.int8)
      output = np.full(count, 123, dtype=np.uint32)
      kernel(source, valid, output)
      values = np.where(valid != 0, source, -7) if masked else source
      expected = values.astype(np.int32).view(np.uint32) + np.uint32(3)
      np.testing.assert_array_equal(output, expected)
