# SPDX-FileCopyrightText: 2026 HenryRenYz and vecops contributors
# SPDX-License-Identifier: MIT
"""Observe physical state consumers while lowering the real store programs."""

from contextlib import contextmanager

from vecops.language._compiler import ir as I
from vecops.language._compiler.backends.cpu.cpp.control import ControlEmitter
from vecops.language._compiler.backends.cpu.cpp.memory import MemoryEmitter
from vecops.language._compiler.backends.cpu.cpp.state import StateEmitter
from vecops.language._compiler.ir.traversal import nodes


@contextmanager
def state_consumers():
  """Retain actual Store/Group, state and physical-tag identities for assertions."""
  records = []
  original_state = StateEmitter.state_vector
  original_resize = StateEmitter.closed_fixed_register_resize
  original_store = MemoryEmitter.store
  original_group = ControlEmitter.elementwise_group

  def scope(original):
    def emit(self, subject, *args, **kwargs):
      previous = getattr(self, "_test_consumer_subject", None)
      self._test_consumer_subject = subject
      try:
        return original(self, subject, *args, **kwargs)
      finally:
        self._test_consumer_subject = previous

    return emit

  def resize(self, *args, **kwargs):
    result = original_resize(self, *args, **kwargs)
    if result is not None:
      self._test_closed_resize_result = result
    return result

  def state(self, value, tag, root_shape, coords, active, axis):
    plan = self.prepared.states[value.data.name]
    subject = getattr(self, "_test_consumer_subject", None)
    stores = subject.stores if type(subject) is I.ElementwiseGroup else (subject,) if type(subject) is I.Store else ()
    connected = any(
      type(node) is I.Expr and node.op == "state_ref" and node.data is value.data
      for store in stores
      for node in nodes(store.value)
    )
    own = self.coordinates(value.type.shape, root_shape, coords)
    previous = getattr(self, "_test_closed_resize_result", None)
    self._test_closed_resize_result = None
    start = len(self.lines)
    try:
      result = original_state(self, value, tag, root_shape, coords, active, axis)
      resized = self._test_closed_resize_result
    finally:
      self._test_closed_resize_result = previous
    records.append(
      {
        "subject": subject,
        "value": value,
        "state": plan.state,
        "plan": plan,
        "connected_to_store": connected,
        "requested_geometry": self.tag_geometry.get(tag),
        "requested_dtype": self.tag_dtypes.get(tag),
        "native_bits": self.environment.codegen_target.vector_bits,
        "own_coords": own,
        "active": str(active),
        "axis": axis,
        "result": result,
        "resize_result": resized,
        "emitted": "\n".join(self.lines[start:]),
      }
    )
    return result

  StateEmitter.state_vector = state
  StateEmitter.closed_fixed_register_resize = resize
  MemoryEmitter.store = scope(original_store)
  ControlEmitter.elementwise_group = scope(original_group)
  try:
    yield records
  finally:
    StateEmitter.state_vector = original_state
    StateEmitter.closed_fixed_register_resize = original_resize
    MemoryEmitter.store = original_store
    ControlEmitter.elementwise_group = original_group


def assert_closed_store_consumers(prepared, source, records, *, efficient=False, resized=False):
  """Require the real prepared Fixed representation and its store-connected read."""
  closed = [record for record in records if record["connected_to_store"] and record["plan"].closed_fixed]
  assert closed
  exact = []
  resized_reads = []
  for record in closed:
    plan, value = record["plan"], record["value"]
    assert prepared.states[value.data.name] is plan
    assert plan.state is record["state"]
    # Rewritten historical reads can carry a distinct TileState descriptor;
    # emission resolves the same prepared state by its stable binding name.
    assert type(value.data) is I.TileState
    assert plan.state.name == value.data.name
    assert type(plan.state.type) is I.VectorType
    assert len(plan.pieces) == 1
    piece = plan.pieces[0]
    assert not piece.scalable and piece.coordinates == (0,)
    assert plan.state.type.tag.N == piece.lanes
    assert record["requested_dtype"] == value.type.dtype
    if (
      record["requested_geometry"] == (False, piece.lanes)
      and record["own_coords"] == ("0",)
      and record["active"] == str(piece.lanes)
      and record["axis"] == 0
    ):
      assert record["result"] == f"{value.data.name}_r0"
      exact.append(record)
    if record["resize_result"] is not None:
      assert plan.state.type.dtype == value.type.dtype
      assert record["result"] == record["resize_result"]
      scalable, width = record["requested_geometry"]
      from vecops.language._compiler.dtypes import ELEMENT_BYTES

      lanes = (record["native_bits"] // (8 * ELEMENT_BYTES[value.type.dtype])) * (1 << width) if scalable else width
      assert lanes > piece.lanes
      assert "vecops::vec::concat(" in record["emitted"] or "vecops::vec::bitcast(" in record["emitted"]
      assert f"< {piece.lanes} ?" in record["emitted"]
      assert "static_assert(std::is_same_v" in record["emitted"]
      resized_reads.append(record)
  assert "vecops::vec::store_convert(" in source
  if efficient:
    assert exact or resized_reads
  if resized:
    assert resized_reads
