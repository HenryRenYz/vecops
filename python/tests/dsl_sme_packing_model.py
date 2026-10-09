# SPDX-FileCopyrightText: 2026 HenryRenYz and vecops contributors
# SPDX-License-Identifier: MIT
"""Test-only bit-move model; no matrix ISA or production interpreter emulation.

ZA words follow the published load/write/read/store slice contract. The model
checks the DSL recipe independently of emitted C++ and of the native packer.
Real target tests are still required for instruction spelling and performance.
"""

import numpy as np
from vecops.language._compiler import ir as I
from vecops.language._compiler.dtypes import ELEMENT_BYTES
from vecops.language._debug.interpreter import Interpreter


class PackingModel:
  def __init__(self, program, *, dtype, svl, source, destination, spatial, reduction, pitch, capacity):
    self.program, self.svl = program, svl
    self.memory = {"sme_packing_input": source.view(np.uint8), "sme_packing_output": destination.view(np.uint8)}
    self.reference = Interpreter(I.KernelIR(I.Block([]), {}, {}), {})
    self.reference.environment.update(
      {
        "sme_packing.m": spatial,
        "sme_packing.k": reduction,
        "sme_packing.pitch": pitch,
        "sme_packing.capacity": capacity,
      }
    )
    scalar = self.reference.value

    def value(expression):
      if isinstance(expression, I.Expr) and expression.op == "streaming_lanes":
        return svl // ELEMENT_BYTES[expression.data]
      return scalar(expression)

    self.reference.value = value
    self.word_dtype = np.dtype("uint64" if ELEMENT_BYTES[dtype] == 8 else "uint32")
    self.width = svl // self.word_dtype.itemsize
    # Poison unused state: zero padding must come from explicit masked moves,
    # not from assuming ZA was initially zero or a previous panel was absent.
    self.za = np.full(
      (self.word_dtype.itemsize, self.width, self.width), np.iinfo(self.word_dtype).max, dtype=self.word_dtype
    )
    self.vectors = {}
    self.masks = {}
    self.trace = []

  def scalar(self, value):
    return int(self.reference.value(value))

  def memory_slice(self, view):
    count = self.scalar(view.shape[0]) if view.shape else 1
    offset = self.scalar(view.offset) * ELEMENT_BYTES[view.dtype]
    end = offset + count * ELEMENT_BYTES[view.dtype]
    root = self.memory[view.storage]
    assert 0 <= offset <= end <= root.size, (view.storage, offset, end, root.size)
    return root[offset:end]

  def run(self):
    self.block(self.program)

  def block(self, block):
    for statement in block.statements:
      if isinstance(statement, I.Assign):
        assert statement.declare
        self.reference.environment[statement.target] = self.reference.value(statement.value)
      elif isinstance(statement, I.For):
        start, stop, step = map(self.scalar, (statement.start, statement.stop, statement.step))
        saved = dict(self.reference.environment)
        for index in range(start, stop, step):
          self.reference.environment = {**saved, statement.target: index}
          self.block(statement.body)
        self.reference.environment = saved
      elif isinstance(statement, I.If):
        saved = dict(self.reference.environment)
        self.block(statement.body if self.scalar(statement.condition) else statement.otherwise)
        self.reference.environment = saved
      elif isinstance(statement, I.Prefetch):
        self.memory_slice(statement.view)  # Even hints must form valid addresses.
      elif isinstance(statement, I.ArchitectureCall):
        self.instruction(statement)
      else:
        raise TypeError(type(statement))

  def instruction(self, operation):
    kind = operation.operation
    vector = operation.vector
    index = self.scalar(operation.slice_index) if operation.slice_index is not None else None
    if index is not None:
      bytes_ = ELEMENT_BYTES[operation.operands[0]] if kind == "sme.load_hor" else self.word_dtype.itemsize
      assert 0 <= index < self.svl // bytes_
    self.trace.append((kind, operation.tile, index))
    if kind == "sme.mask":
      count = self.scalar(vector.extent)
      assert 0 <= count <= self.svl // ELEMENT_BYTES[vector.dtype]
      self.masks[vector.name] = count
    elif kind in {"sme.load_hor", "sme.store_ver"}:
      memory = self.memory_slice(operation.view)
      if kind == "sme.load_hor":
        storage_word = np.dtype(operation.operands[0])
        assert memory.size <= self.svl
        assert memory.size % storage_word.itemsize == 0
        # Tile t's row r is physical byte row r*sizeof(element)+t.
        # Reinterpret that same row through this model's packed-word view.
        physical = index * storage_word.itemsize + operation.tile
        tile, row = physical % self.word_dtype.itemsize, physical // self.word_dtype.itemsize
        self.za[tile, row, :] = 0
        self.za[tile, row, :].view(np.uint8)[: memory.size] = memory
      else:
        assert operation.operands[0] == self.word_dtype.name
        count = memory.size // self.word_dtype.itemsize
        assert count <= self.width
        memory[: count * self.word_dtype.itemsize] = self.za[operation.tile, :count, index].copy().view(np.uint8)
    elif kind == "sme.load_bits":
      memory = self.memory_slice(operation.view)
      assert memory.size <= self.svl
      words = np.zeros(self.width, dtype=self.word_dtype)
      words.view(np.uint8)[: memory.size] = memory
      self.vectors[vector.name] = words
    elif kind in {"sme.write_hor", "sme.read_ver"}:
      assert vector.dtype == self.word_dtype.name
      count = self.scalar(vector.extent)
      assert 0 <= count <= self.width
      if operation.operands:
        assert self.masks[operation.operands[0].name] == count
      if kind == "sme.write_hor":
        self.za[operation.tile, index, :count] = self.vectors[vector.name][:count]
      else:
        words = np.zeros(self.width, dtype=self.word_dtype)
        words[:count] = self.za[operation.tile, :count, index]
        self.vectors[vector.name] = words
    elif kind == "sme.store_bits":
      memory = self.memory_slice(operation.view)
      assert memory.size <= self.svl
      memory[:] = self.vectors[vector.name].view(np.uint8)[: memory.size]
    else:
      raise AssertionError(kind)
