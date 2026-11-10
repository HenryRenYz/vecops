# SPDX-FileCopyrightText: 2026 HenryRenYz and vecops contributors
# SPDX-License-Identifier: MIT
"""Cold collective-to-kernel-plan adapter; graph replay contains no Python."""
from dataclasses import dataclass
import os
from types import SimpleNamespace


@dataclass
class _Lease:
  native: object
  argument_paths: tuple[str, ...]
  consumer: object = None
  route_planner: object = None
  kind: str = 'all_gather'
  input_bytes: int = 0
  output_paths: tuple[str,...] = ()
  input_paths: tuple[str,...] = ()
  routing: object = None
  consumer_contract: object = None

  @property
  def placement_paths(self):
    return self.argument_paths if self.route_planner is not None else ()

  def configure_tensor_placements(self,endpoints):
    if self.route_planner is not None:
      self.routing=self.route_planner.configure(self.native,kind=self.kind,
          inputs=[endpoints.get(path) for path in self.input_paths],
          outputs=[endpoints.get(path) for path in self.output_paths],input_bytes=self.input_bytes,
          consumer=self.consumer_contract,previous=self.routing)

  @property
  def capacity_provider_id(self):
    return self.native.capacity_provider_id

  def provision_capacity(self):
    """Collectively provision this provider after all local plans are registered.

    The native adapter drains queued work and releases the GIL for the cold
    stopped/remap handshake. Return is any-rank root growth, identical on peers.
    """
    return self.native.provision_capacity()

  def preparation_consensus(self,local_success):
    """Cold all-rank AND; false does not poison a valid preparation group."""
    return self.native.preparation_consensus(local_success)

  @property
  def buffer_requirements(self):
    return self.consumer[0].buffer_requirements if self.consumer is not None else []

  def activate_buffers(self, bindings, owner):
    if self.consumer is not None:
      self.consumer[0].activate_buffers(bindings, owner)
    elif bindings:
      raise ValueError('collective without a nested consumer has no kernel buffers')

  def suspend_buffers(self):
    if self.consumer is not None:
      self.consumer[0].suspend_buffers()

  @property
  def abi_address(self):
    return self.native.abi_address


def install(module):
  from vecops import _schema_bridge

  def prepare(function, arguments, describe, *, describe_only=False):
    if os.environ.get('VECOPS_GRAPH_RESIDENT_COMM', '1') == '0':
      return None
    import torch
    from vecops._dtype import normalize_dtype
    borrowed = function._schema.name == 'vecops_comm::consume'
    identity = arguments[0]
    outputs = [arguments[1]] if borrowed else arguments[1]
    inputs = [arguments[2]] if borrowed else arguments[2]

    def meta(ref):
      m = describe(ref)
      if str(m.device) != 'cpu':
        raise ValueError('resident communication requires captured CPU metadata')
      return torch.empty_strided(m.shape, m.strides, device='meta',
                                 dtype=getattr(torch, m.dtype.removeprefix('torch.')))

    out_meta, in_meta = [meta(x) for x in outputs], [meta(x) for x in inputs]
    # Opaque-byte complex collectives remain legal via the ordinary native op.
    if any(t.is_complex() for t in out_meta+in_meta):
      return None
    types = [int(normalize_dtype(str(t.dtype).removeprefix('torch.'))) for t in out_meta+in_meta]
    nested = None
    if borrowed:
      name, addresses, specs = module.resident_consumer_info(identity)
      namespace, operator = name.split('::', 1)
      operation = getattr(getattr(torch.ops, namespace), operator).default

      def consumer_meta(ref):
        if ref is addresses:
          return SimpleNamespace(shape=tuple(addresses.shape), strides=tuple(addresses.stride()),
                                 dtype=str(addresses.dtype), device='cpu')
        return describe(ref)

      nested = _schema_bridge.prepare_replay_plan(operation,
          (addresses, outputs[0], *specs), consumer_meta, describe_only=describe_only)
      if nested is None:
        return None
    native = module.resident_plan(identity, out_meta, in_meta, types,
                                  nested[0].abi_address if nested else 0)
    if native is None:
      return None
    paths = ('1', '2') if borrowed else tuple(
        f'{argument}:{i}' for argument, values in ((1, outputs), (2, inputs))
        for i in range(len(values)))
    from ._template import routing_for,describe as describe_collective
    planner=routing_for(function._schema.name,identity)
    kind='borrowed-consumer' if borrowed else None
    if planner is not None and not borrowed:
      kind=describe_collective(function._schema.name,identity).get('kind')
      if kind not in {'all_gather','all_gather_into_tensor','all_gatherv'}:planner=None
    output_paths=('1',) if borrowed else tuple(f'1:{i}' for i in range(len(outputs)))
    input_paths=('2',) if borrowed else tuple(f'2:{i}' for i in range(len(inputs)))
    lease = _Lease(native, paths, nested,planner,kind or 'all_gather',
        in_meta[0].numel()*in_meta[0].element_size() if len(in_meta)==1 else 0,output_paths,input_paths)
    if borrowed:
      from ._routing import consumer_contract
      lease.consumer_contract=consumer_contract(name,in_meta[0],out_meta[0],specs)
    # A borrowed consumer may read-modify-write its owned destination. Keep
    # its declared access so graph planning cannot treat prior contents as dead.
    output_access = [nested[1][1]] if borrowed else [2]*len(outputs)
    return lease, output_access+[1]*len(inputs), [0]*len(types), []

  for name in ('vecops_comm::run', 'vecops_comm::consume'):
    _schema_bridge.register_replay_plan_provider(name, prepare)
