# SPDX-FileCopyrightText: 2026 HenryRenYz and vecops contributors
# SPDX-License-Identifier: MIT
"""Compilation-local compact keys for typed expressions and shape DAGs.

Shape nodes are numbered bottom-up: hashing a shared diamond never expands it
as a tree. Opaque payloads retain object identity; memory/reference identity is
not inferred from repr(). Floating literal keys preserve their exact bits.
"""

import struct

from .. import ir as I


class ExpressionKeys:
  def __init__(self):
    self.types = {}
    self.metadata = {}
    self.numbers = {}
    self.opaque = {}

  def data(self, value):
    if isinstance(value, float):
      return float, struct.pack("!d", value)
    if value is None or isinstance(value, (str, bytes, int, bool)):
      return type(value), value
    if isinstance(value, tuple):
      return tuple(self.data(item) for item in value)
    self.opaque[id(value)] = value
    return type(value), id(value)

  def type(self, value):
    if value in self.types:
      return self.types[value]
    if isinstance(value, I.ScalarType):
      key = type(value), value.dtype, value.weak
    else:
      tag = value.tag if isinstance(value, I.VectorType) else None
      key = type(value), value.dtype, value.mask, tuple(self.shape(s) for s in value.shape), tag
    self.types[value] = key
    return key

  def shape(self, value):
    if value in self.metadata:
      return self.metadata[value]
    # Named/memory/state leaves are not equated by spelling. That would turn
    # two lexical reads or distinct reference versions into the same shape.
    if value.op in {"load", "packed_load", "local", "frozen", "state_ref"}:
      key = "identity", id(value)
    else:
      key = self.type(value.type), value.op, tuple(self.shape(a) for a in value.args), self.data(value.data)
    number = self.numbers.setdefault(key, len(self.numbers))
    self.metadata[value] = number
    return number

  def expression(self, value):
    return self.type(value.type), value.op, tuple(map(id, value.args)), self.data(value.data)
