# SPDX-FileCopyrightText: 2026 HenryRenYz and vecops contributors
# SPDX-License-Identifier: MIT
"""CPU prefetch hints and explicitly owned double-buffer storage."""

from .. import ir as I


class PipelineFrontend:
  def pipeline_operation(self, operation, args, kwargs, node):
    if operation == "prefetch":
      if len(args) != 1 or not isinstance(args[0], I.View):
        self.error(node, "prefetch expects a memory view")
      level, intent = kwargs.get("level", "L1"), kwargs.get("intent", "read")
      temporal = I.constant(self.value(kwargs.get("temporal", True), node))
      if level not in {"L1", "L2"} or intent not in {"read", "write"} or type(temporal) is not bool:
        self.error(node, "invalid prefetch cache level/intent/temporal hint")
      self.block.statements.append(I.Prefetch(args[0], level, intent, temporal))
      return None
    if operation == "double_buffer":
      if not 1 <= len(args) <= 2:
        self.error(node, "double_buffer expects a shape and optional dtype")
      # The first dimension selects the two scalar-storage banks. Views retain
      # ordinary access/alias semantics, with no array of hardware vectors.
      shape = (I.literal(2), *self.shape(args[0], node))
      return self.call_allocation([shape, *args[1:]], kwargs, node)
    if operation == "pipelined":
      stages = self.unroll_count(kwargs.get("stages", I.literal(2)), node)
      distance = self.unroll_count(kwargs.get("distance", I.literal(1)), node)
      if stages != 2:
        self.error(node, "CPU software pipelines currently support two stages")
      buffering = I.constant(self.value(kwargs.get("buffering", False), node))
      if type(buffering) is not bool or buffering and distance != 1:
        self.error(node, "pipeline buffering requires a compile-time bool and distance=1")
      return (
        operation,
        args,
        (stages, distance, self.unroll_count(kwargs.get("unroll", I.literal(2)), node), buffering),
      )
    return NotImplemented
