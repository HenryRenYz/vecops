"""CPU compilation driver: frontend -> kernel passes -> vector lowering -> C++."""

from dataclasses import replace
from ....types import TensorType
from ...frontend.parser import lower
from ...pipeline import kernel_pipeline, cpu_vector_pipeline, cpu_memory_pipeline
from ...lowering.tile_to_vector import lower_kernel
from ..base import LoweredArtifact
from .cpp.emitter import generate
from .planning import prepare
from ...environment import freeze


class CPUBackend:
  def lower(self, template, out_idx, *, pipeline=None, options=None, target="generic"):
    # Freeze all stages before executing any user pass. Plugins cannot change
    # the vector or memory pipeline of an already-started compilation.
    options = dict(options or {})
    environment = freeze(options, target=target)
    options["_compiler_environment"] = environment
    kernel_manager = kernel_pipeline(pipeline, options=options).with_options(options)
    vector_manager = cpu_vector_pipeline(options=options)
    memory_manager = cpu_memory_pipeline(options=options)
    annotations = dict(template.annotations)
    parameters = tuple(template.signature.parameters.values())
    for index in out_idx:
      name = parameters[index].name
      annotation = annotations.get(name)
      if isinstance(annotation, TensorType) and annotation.access == "input":
        annotations[name] = replace(annotation, access="output")
    kernel = kernel_manager.run(lower(replace(template, annotations=annotations), catalog=environment.operations))
    vectors = lower_kernel(kernel, manager=vector_manager)
    prepared = prepare(vectors, memory_manager=memory_manager)
    source, emitted_regions = generate(prepared)
    source = f"// compiler environment: {environment.fingerprint}\n" + source
    vectors.regions.extend(emitted_regions)
    return LoweredArtifact(kernel, vectors, source)
