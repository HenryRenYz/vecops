# SPDX-FileCopyrightText: 2026 HenryRenYz and vecops contributors
# SPDX-License-Identifier: MIT
"""CPU compilation driver: frontend -> kernel passes -> vector lowering -> C++."""

from dataclasses import replace

from ....types import TensorType
from ...environment import freeze
from ...frontend.parser import lower
from ...lowering.tile_to_vector import lower_kernel
from ...passes.manager import registered_passes
from ...pipeline import cpu_memory_pipeline, cpu_vector_pipeline, kernel_pipeline, load_builtin_passes
from ..base import LoweredArtifact
from .cpp.emitter import generate
from .planning import prepare


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
    vectors = lower_kernel(kernel, manager=vector_manager)
    prepared = prepare(vectors, memory_manager=memory_manager)
    source, emitted_regions = generate(prepared)
    source = f"// compiler environment: {environment.fingerprint}\n" + source
    vectors.regions.extend(emitted_regions)
    if session is not None:
      session.stage("vector", vectors)
      session.stage("backend", prepared)
    return LoweredArtifact(kernel, vectors, source, prepared)
