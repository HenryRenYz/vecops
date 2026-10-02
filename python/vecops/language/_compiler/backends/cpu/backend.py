"""CPU compilation driver: frontend -> kernel passes -> vector lowering -> C++."""
from dataclasses import replace
from ....types import TensorType
from ...frontend.parser import lower
from ...pipeline import kernel_pipeline, cpu_vector_pipeline
from ...lowering.tile_to_vector import lower_kernel
from ..base import LoweredArtifact
from .cpp.emitter import generate
from .planning import prepare


class CPUBackend:
  def lower(self, template, out_idx, *, pipeline=None, options=None):
    # Freeze both stages before executing any user pass. Kernel plugins may
    # register vector plugins, but must not alter this compilation mid-flight.
    kernel_manager = kernel_pipeline(pipeline, options=options)
    vector_manager = cpu_vector_pipeline(options=options)
    annotations = dict(template.annotations)
    parameters = tuple(template.signature.parameters.values())
    for index in out_idx:
      name = parameters[index].name
      annotation = annotations.get(name)
      if isinstance(annotation, TensorType) and annotation.access == "input":
        annotations[name] = replace(annotation, access="output")
    kernel = kernel_manager.run(lower(replace(template, annotations=annotations)))
    vectors = lower_kernel(kernel, manager=vector_manager)
    prepared = prepare(vectors)
    source, emitted_regions = generate(prepared)
    vectors.regions.extend(emitted_regions)
    return LoweredArtifact(kernel, vectors, source)
