# SPDX-FileCopyrightText: 2026 HenryRenYz and vecops contributors
# SPDX-License-Identifier: MIT
"""Graph facts feed generic resource compilation without fixed memory tiers."""
from vecops import graph,planner


class Tensor:
  pass


def test_storage_traffic_counts_payload_not_metadata_views():
  capture=graph.GraphCapture('traffic')
  a,b=Tensor(),Tensor()
  metadata=graph.TensorMetadata(shape=(16,),strides=(1,),dtype='uint8',device='cpu',element_size=1,logical_bytes=16)
  with capture:
    capture.tensor(a,metadata,storage_identity=('input',1),storage_bytes=16,external=True)
    capture.tensor(b,metadata,storage_identity=('result',1),storage_bytes=16)
    capture.record_operator('copy',inputs=[(a,graph.AccessKind.read,'source')],outputs=[b])
    capture.record_operator('aten.size',inputs=[(b,graph.AccessKind.read,'metadata')],outputs=[])
  execution=capture.finish()
  storage=execution.tensors[capture.tensor_id(a)].storage_id
  assert planner.storage_traffic(execution)[storage]==16
  assert planner.graph_digest(execution)==execution.content_digest()
  requirements=planner.requirements_from_graph(execution)
  assert len(requirements)==2
  assert next(req for req in requirements if req.id==storage).ownership=='borrowed'


def test_compiler_public_api_contains_generic_policy_and_native_data():
  assert {'GraphMemoryCompiler','BufferRequirement','ProgramCall','PreparedProgram'}<=set(planner.__all__)
  assert not {'MemoryTier','MachineProfile','PlanSession','BufferSession'}&set(planner.__all__)
