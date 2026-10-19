# SPDX-FileCopyrightText: 2026 HenryRenYz and vecops contributors
# SPDX-License-Identifier: MIT
"""CPU compilation driver: frontend -> kernel passes -> vector lowering -> C++."""

from dataclasses import replace
from itertools import islice

from ....types import TensorType
from ... import ir as I
from ...environment import freeze
from ...frontend.parser import lower
from ...ir.traversal import nodes
from ...lowering.tile_to_vector import lower_kernel
from ...operations import Trait
from ...passes.manager import registered_passes
from ...pipeline import cpu_memory_pipeline, cpu_vector_pipeline, kernel_pipeline, load_builtin_passes
from ..base import LoweredArtifact
from .cpp.emitter import generate
from .planning import prepare


def _compound_vector_owner(kernel, catalog):
  """Bound the control/storage scan used for AVX-512 tail profitability.

  Large private-helper owners and hardware regions benefit from independently
  masked words: grouped dynamic tails otherwise keep several words live across
  fallback paths. Large register-layout owners with narrow floating arithmetic
  also benefit: their arithmetic widens within a native word. Ordinary maps,
  copies and conversion-only packers retain grouped tails.
  This changes only the default; explicit tail options retain precedence.
  """
  structure = tuple(islice(nodes(kernel.body, skip=(I.Expr, I.View)), 65))
  if any(isinstance(node, I.HardwareRegion) for node in structure) or (
    len(structure) > 64
    and any(isinstance(node, (I.HardwareRegion, I.ExitRegion)) for node in nodes(kernel.body, skip=(I.Expr, I.View)))
  ):
    return True
  if len(structure) <= 64:
    return False
  permutation, narrow_arithmetic = False, False
  for node in nodes(kernel.body, skip=(I.View,)):
    if not isinstance(node, I.Expr):
      continue
    permutation |= catalog.has(node, Trait.REGISTER_TRANSFORM)
    narrow_arithmetic |= (
      type(node.type) is I.TileType
      and node.type.dtype in {"float16", "bfloat16"}
      and catalog.get(node).family == "arithmetic"
      and node.op != "identity"
    )
    if permutation and narrow_arithmetic:
      return True
  return False


class CPUBackend:
  def lower(self, template, out_idx, *, pipeline=None, options=None, target="generic"):
    # Freeze all stages before executing any user pass. Plugins cannot change
    # the vector or memory pipeline of an already-started compilation.
    options = dict(options or {})
    options.setdefault("cpu_ordered_loop_unroll", True)
    options.setdefault("cpu_state_vector_reuse", True)
    environment = freeze(options, target=target)
    options["_compiler_environment"] = environment
    load_builtin_passes()
    registry = registered_passes()
    kernel_manager = kernel_pipeline(pipeline, options=options, _registry_snapshot=registry).with_options(options)
    vector_manager = cpu_vector_pipeline(options=options, _registry_snapshot=registry)
    memory_manager = cpu_memory_pipeline(options=options, _registry_snapshot=registry)
    annotations = dict(template.annotations)
    parameters = tuple(template.signature.parameters.values())
    for index in out_idx:
      name = parameters[index].name
      annotation = annotations.get(name)
      if isinstance(annotation, TensorType) and annotation.access == "input":
        annotations[name] = replace(annotation, access="output")
    frontend = lower(replace(template, annotations=annotations), catalog=environment.operations)
    session = options.get("_debug_session")
    if session is not None:
      session.stage("frontend", frontend)
    if not options.get("debug_prints", True):
      from ... import ir as I
      from ...ir.traversal import children

      def strip(block):
        block.statements = [s for s in block.statements if not isinstance(s, I.Print)]
        for statement in block.statements:
          for child in children(statement):
            strip(child)

      strip(frontend.body)
    kernel = kernel_manager.run(frontend)
    if session is not None:
      session.stage("kernel", kernel)
    target_info = environment.codegen_target
    if target_info.architecture == "x86" and target_info.isa == "avx512":
      vector_manager = vector_manager.with_options(
        {"_cpu_compound_owner": _compound_vector_owner(kernel, environment.operations)}
      )
    vectors = lower_kernel(kernel, manager=vector_manager)
    prepared = prepare(vectors, memory_manager=memory_manager)
    source, emitted_regions = generate(prepared)
    source = f"// compiler environment: {environment.fingerprint}\n" + source
    vectors.regions.extend(emitted_regions)
    if session is not None:
      session.stage("vector", vectors)
      session.stage("backend", prepared)
    return LoweredArtifact(kernel, vectors, source, prepared)
