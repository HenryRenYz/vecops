# SPDX-FileCopyrightText: 2026 HenryRenYz and vecops contributors
# SPDX-License-Identifier: MIT
"""Framework-neutral graph facts and resource-backed memory compilation."""
from collections import defaultdict
from vecops.graph import AccessKind, ExecutionGraph
from .compiler import GraphMemoryCompiler, requirements_from_graph, _event_weights, _metadata_only_event


def graph_digest(graph):
  return graph.content_digest()


def storage_traffic(graph):
  """Loop-weighted payload bytes, with each storage counted once per event."""
  result=defaultdict(float)
  weights=_event_weights(graph)
  for event in graph.events:
    if _metadata_only_event(graph,event):
      continue
    touched={}
    for use in event.uses:
      if use.access in {AccessKind.alias,AccessKind.allocate}:
        continue
      tensor=graph.tensors[use.tensor_id]
      size=tensor.metadata.logical_bytes
      if not isinstance(size,int):
        size=graph.storages[tensor.storage_id].bytes
      if isinstance(size,int):
        touched[tensor.storage_id]=max(touched.get(tensor.storage_id,0),size)
    for storage,size in touched.items():
      result[storage]+=size*weights[event.index]
  return dict(result)


def plan_memory(graph,queries,view,budgets,*,allowed_pools=(),consumers=(),owned_outputs=False,
                native_module=None,participant=None,caches=()):
  """Compile tensor storage using the actual runtime's public resource queries."""
  requirements=requirements_from_graph(graph,allowed_pools=allowed_pools,
      consumers=consumers,owned_outputs=owned_outputs)
  return GraphMemoryCompiler(queries,view,budgets,native_module=native_module,
      participant=participant,caches=caches).compile(requirements)
