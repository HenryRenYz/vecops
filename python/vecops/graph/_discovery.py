# SPDX-FileCopyrightText: 2026 HenryRenYz and vecops contributors
# SPDX-License-Identifier: MIT
"""Discover graph lowering requests during the existing FakeTensor trace.

The observer shares the real replay selector. It creates artifact requests,
never graph buffers or executable bindings. Real initializer/steady capture
still proves storage ownership and aliases after the shared build frontier.
"""

import os
from contextlib import contextmanager
from types import SimpleNamespace


@contextmanager
def _compiler_bookkeeping():
  """Keep factory elaboration out of an enclosing semantic memory trace."""
  from torch.utils._python_dispatch import _disable_current_modes

  from .capture import _active_capture
  token = _active_capture.set(None)
  try:
    with _disable_current_modes():
      yield
  finally:
    _active_capture.reset(token)


def discover_pointwise_requests(function, *args, borrow_outputs=False, include_grid_seed=False, **kwargs):
  """Add a named region's pointwise artifacts to model-level collection.

  Real-tensor traces keep their existing behavior. Fake semantic memory
  traces also collect requests; compiler bookkeeping is hidden from their IR.
  Retaining real intermediate tensors here would inflate large-model memory;
  the storage-free discovery path is enabled only by precompile's fake trace.
  """
  from vecops import _precompile

  collector = _precompile._active_collector
  if (
    collector is None or not collector.discover_graph_artifacts or os.environ.get("VECOPS_GRAPH_POINTWISE", "1") == "0"
  ):
    return function(*args, **kwargs)

  import torch
  from torch.utils._python_dispatch import TorchDispatchMode
  from torch.utils._pytree import tree_flatten, tree_map

  from vecops import TensorMeta
  from vecops._arm_vector_state import get_vector_state

  from . import _int_scalar
  from ._pointwise_plan import select_kernel

  parallelism = int(os.environ.get("VECOPS_NUM_THREADS", torch.get_num_threads()))
  vector_token = get_vector_state().env_token
  tensors, candidates = {}, []

  def ref(value):
    if not isinstance(value, torch.Tensor):
      return value
    if id(value) not in tensors:
      tensors[id(value)] = SimpleNamespace(
        id=id(value),
        tensor=value,
        metadata=SimpleNamespace(
          shape=tuple(value.shape), strides=tuple(value.stride()), dtype=str(value.dtype), device=value.device.type
        ),
      )
    return tensors[id(value)]

  class Observe(TorchDispatchMode):
    def __torch_dispatch__(self, operation, types, args=(), kwargs=None):
      result = operation(*args, **(kwargs or {}))
      if isinstance(result, torch.Tensor):
        values = []
        for index, parameter in enumerate(operation._schema.arguments):
          if index < len(args):
            values.append(args[index])
          elif parameter.name in (kwargs or {}):
            values.append(kwargs[parameter.name])
          elif parameter.has_default_value():
            values.append(parameter.default_value)
          else:
            return result
        # Keep metadata only while observing. Factory elaboration is performed
        # after leaving TorchDispatchMode, so compiler bookkeeping is invisible.
        candidates.append((operation._schema, tree_map(ref, tuple(values)), ref(result)))
      return result

  with Observe():
    result = function(*args, **kwargs)
  escapes = {x.untyped_storage()._cdata for x in tree_flatten(result)[0] if isinstance(x, torch.Tensor)}

  def metadata(value):
    return value.metadata

  def tensor_meta(value):
    m = metadata(value)
    return TensorMeta(m.shape or (1,), strides=m.strides or (1,), dtype=m.dtype.removeprefix("torch."))

  with _compiler_bookkeeping():
    for schema, arguments, output in candidates:
      inplace = schema.name in {"aten::add_", "aten::sub_", "aten::mul_", "aten::div_", "aten::copy_", "aten::fill_"}
      if not inplace and not borrow_outputs and output.tensor.untyped_storage()._cdata in escapes:
        continue
      selected = select_kernel(schema, arguments, output, metadata, parallelism, vector_token)
      if selected is not None:
        compiled, a, b, target, alpha, _ = selected
        compiled(tensor_meta(a), tensor_meta(b), tensor_meta(target), alpha)
        collector.graph_kernels[id(compiled)] = compiled

    if include_grid_seed and os.environ.get("VECOPS_GRAPH_INTEGER_SCALARS", "1") != "0":
      compiled = _int_scalar.kernel((1,), (1,), (1,), (1,), parallelism, vector_token)
      seed = TensorMeta((1,), dtype="int64")
      compiled(seed, seed, seed, 0)
      collector.graph_kernels[id(compiled)] = compiled
  return result


def discover_recorded_requests(capture, *, borrow_outputs=False, include_grid_seed=False):
  """Select artifacts from an already recorded tape, without another observer."""
  from vecops import _precompile, TensorMeta
  collector = _precompile._active_collector
  if collector is None or not collector.discover_graph_artifacts:
    raise ValueError('recorded request discovery requires an active collector')
  import torch
  from vecops._arm_vector_state import get_vector_state
  from ._replay import _Ref
  from ._pointwise_plan import select_kernel
  from . import _int_scalar
  tape,graph = capture._torch_replay_tape,capture._torch_execution_graph
  workers=int(os.environ.get('VECOPS_NUM_THREADS',torch.get_num_threads()))
  vector=get_vector_state().env_token
  escaping={graph.tensors[ref.id].storage_id for ref in tape.output_leaves if isinstance(ref,_Ref)}
  def metadata(ref):return graph.tensors[ref.id].metadata
  def tensor_meta(ref):
    m=metadata(ref)
    return TensorMeta(m.shape or (1,),strides=m.strides or (1,),dtype=m.dtype.removeprefix('torch.'))
  with _compiler_bookkeeping():
    for instruction in tape.instructions:
      output=instruction.result
      if not isinstance(output,_Ref):continue
      schema=instruction.function._schema
      inplace=schema.name in {'aten::add_','aten::sub_','aten::mul_','aten::div_','aten::copy_','aten::fill_'}
      if not inplace and not borrow_outputs and graph.tensors[output.id].storage_id in escaping:continue
      selected=select_kernel(schema,instruction.arguments,output,metadata,workers,vector)
      if selected is not None:
        compiled,a,b,target,alpha,_=selected
        compiled(tensor_meta(a),tensor_meta(b),tensor_meta(target),alpha)
        collector.graph_kernels[id(compiled)]=compiled
    if include_grid_seed and os.environ.get('VECOPS_GRAPH_INTEGER_SCALARS','1')!='0':
      compiled=_int_scalar.kernel((1,),(1,),(1,),(1,),workers,vector)
      seed=TensorMeta((1,),dtype='int64');compiled(seed,seed,seed,0)
      collector.graph_kernels[id(compiled)]=compiled
