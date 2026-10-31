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

  @property
  def abi_address(self):
    return self.native.abi_address


def install(module):
  from vecops import _schema_bridge

  def prepare(function, arguments, describe):
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
          (addresses, outputs[0], *specs), consumer_meta)
      if nested is None:
        return None
    native = module.resident_plan(identity, out_meta, in_meta, types,
                                  nested[0].abi_address if nested else 0)
    if native is None:
      return None
    paths = ('1', '2') if borrowed else tuple(
        f'{argument}:{i}' for argument, values in ((1, outputs), (2, inputs))
        for i in range(len(values)))
    lease = _Lease(native, paths, nested)
    return lease, [2]*len(outputs)+[1]*len(inputs), [0]*len(types), []

  for name in ('vecops_comm::run', 'vecops_comm::consume'):
    _schema_bridge.register_replay_plan_provider(name, prepare)
