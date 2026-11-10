# SPDX-FileCopyrightText: 2026 HenryRenYz and vecops contributors
# SPDX-License-Identifier: MIT
"""Address-free preparation requests; executable handles stay process local."""
from dataclasses import replace
import json
import platform

from . import _C
from ._identity import digest
from ._operator import _declarations


def _definition_key(definition):
  if hasattr(definition,'_discovery_key'):
    return definition._discovery_key
  public = definition if hasattr(definition,'_native') else None
  if public is not None:definition=public._native()
  def bound(value):
    return ('symbol', value.symbol) if value.is_symbol else ('fixed', value.fixed_value)
  def dimension(value, expression=None):
    if value.kind == _C.DimensionKind.expression:
      if expression is None or not hasattr(expression,'canonical'):
        # Raw expert declarations do not expose their expression tree.
        raise ValueError('opaque dimension expression')
      return ('expression',expression.canonical())
    return (str(value.kind), value.symbol, value.fixed_value,
            bound(value.alignment), bound(value.lower_bound), bound(value.upper_bound))
  parameters = []
  for index,p in enumerate(definition.inputs):
    if isinstance(p, _C.TensorDef):
      original = None if public is None else public.parameters[index]
      def dimensions(values, attribute):
        originals = [None]*len(values) if original is None else getattr(original,attribute)
        return tuple(dimension(x,expression) for x,expression in zip(values,originals))
      parameters.append(('tensor', p.name, dimensions(p.shape,'shape'),
          dimensions(p.strides,'strides'), p.dtype.symbol,
          None if p.dtype.dtype is None else int(p.dtype.dtype), p.dtype.anonymous,
          str(p.access), p.optional, p.device_type, p.device_index))
    else:
      parameters.append(('scalar', p.name, int(p.dtype),
                         None if p.default is None else (int(p.default.dtype), p.default.value)))
  return (definition.name, parameters, sorted((k, str(v)) for k, v in definition.values.items()),
          definition.runtime_parallelism)


def operator_key(operator):
  declaration = _declarations.get(operator._handle)
  if declaration is None:
    return None
  try:
    definition = _definition_key(declaration.kernel_def)
  except ValueError:
    return None
  compiler = declaration.compiler
  return digest(('prepare-operator-v1', platform.node(), str(declaration.source), json.dumps(definition),
      declaration._build_identity, declaration._source_identity, declaration._recipe_id,
      str(compiler.cache_dir), str(compiler.cache_mode)))


def _encode_argument(value):
  if value is None:
    return None
  if isinstance(value, _C.TensorMeta):
    return ('tensor', int(value.dtype), tuple(value.sizes), tuple(value.strides), value.writable)
  if isinstance(value, _C.Scalar):
    return ('scalar', int(value.dtype), value.value)
  raise ValueError('preparation exchange only accepts storage-free arguments')


def _decode_argument(value, signature):
  if value is None:
    return None
  if value[0] == 'tensor':
    _, dtype, shape, strides, writable = value
    flags = 3 if 'inout' in signature[4].lower() else 2 if writable else 1
    return _C.TensorMeta(_C.DType(dtype), shape, strides, flags, signature[5], signature[6])
  _, dtype, payload = value
  dtype = _C.DType(dtype)
  from ._precompile import _SIGNED_DTYPES, _UNSIGNED_DTYPES
  return (_C.Scalar.signed_integer(payload, dtype) if dtype in _SIGNED_DTYPES else
          _C.Scalar.unsigned_integer(payload, dtype) if dtype in _UNSIGNED_DTYPES else
          _C.Scalar.floating(payload, dtype))


def exchange_requests(requests, rank, exchange, execution_parallelism):
  """Return this process's union slice and shared-owner ranks.

  ``exchange`` must return one JSON-compatible record per rank in rank order.
  Uneven shapes are transferred only to a process with the exact same declared
  operator, build policy, cache namespace and execution parallelism. Unknown
  declarations are retained on their originating process. No tensor payload,
  native pointer, pickle of an Operator, or executable plan is exchanged.
  """
  local = {}
  records = []
  private = []
  for request in requests:
    handle = request.operator._handle
    if handle not in local:
      local[handle] = (operator_key(request.operator), request)
    key, _ = local[handle]
    if key is None:
      private.append(request)
      continue
    signature = request.key[1]
    values = [(name, ('dtype', int(value)) if isinstance(value, _C.DType) else ('int', value))
              for name, value in request.call.values.items()]
    arguments = tuple(_encode_argument(x) for x in request.call.arguments.values)
    records.append(dict(operator=key, signature=signature, values=values, arguments=arguments,
                        key=digest((key, signature, values))))
  shared = {key: request for key, request in local.values() if key is not None}
  gathered = exchange(dict(version=1, parallelism=execution_parallelism,
                          operators=list(shared), requests=records, private=bool(private)))
  if not gathered or not 0 <= rank < len(gathered):
    raise ValueError('invalid preparation exchange rank order')
  owners = {}
  for r, payload in enumerate(gathered):
    if payload['version'] != 1:
      raise ValueError('incompatible preparation exchange version')
    for key in payload['operators']:
      owners.setdefault((payload['parallelism'], key), r)
  selected = list(private)
  active = {r for r, payload in enumerate(gathered) if payload['private']}
  seen = set()
  for payload in gathered:
    for record in payload['requests']:
      owner = owners[(payload['parallelism'], record['operator'])]
      active.add(owner)
      token = (payload['parallelism'], record['key'])
      if owner != rank or token in seen:
        continue
      seen.add(token)
      prototype = shared[record['operator']]
      values = {k: _C.DType(v) if kind == 'dtype' else v for k, (kind, v) in record['values']}
      signature = tuple(tuple(x) for x in record['signature'])
      call = _C.KernelCall([_decode_argument(x, s) for x, s in zip(record['arguments'], signature)], values)
      selected.append(replace(prototype, call=call, key=(prototype.operator._handle, record['key'])))
  local_complete = all(owners[(execution_parallelism, key)] == rank for key in shared)
  return tuple(selected), tuple(sorted(active)), local_complete
