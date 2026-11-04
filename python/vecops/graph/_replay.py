# SPDX-FileCopyrightText: 2026 HenryRenYz and vecops contributors
# SPDX-License-Identifier: MIT
"""Torch lowering for the single GraphCapture IR and native CPU interpreter.

Only capture/lowering and the submission boundary run Python. The executable
contains dispatcher operators, never a callback to a model or Python kernel.
"""
from __future__ import annotations

import contextlib
import hashlib
import os
import re
from pathlib import Path
import shlex
import shutil
import subprocess
import threading
import weakref
from dataclasses import dataclass, replace
from typing import Any

_loader_lock = threading.Lock()
_executor_class = None
_executor_namespace = None


def _load_executor():
  global _executor_class, _executor_namespace
  with _loader_lock:
    if _executor_class is not None:
      return _executor_class
    import torch
    from torch.utils.cpp_extension import include_paths, library_paths
    from vecops._cache_lock import file_lock
    from vecops._compiler import default_cache_dir, _sdk, require_native_build_api

    require_native_build_api()
    sdk_root, _ = _sdk()
    abi_headers = {name:(sdk_root / 'include/vecops/runtime' / name).read_bytes()
                   for name in ('KernelPlanAbi.h', 'CallAbi.h')}

    source = Path(__file__).with_suffix('.cpp')
    compiler = shlex.split(os.environ.get('VECOPS_GRAPH_REPLAY_CXX') or
                          os.environ.get('CXX') or
                          os.environ.get('VECOPS_TORCH_BRIDGE_CXX') or 'g++')
    resolved = shutil.which(compiler[0])
    if resolved is None:
      raise RuntimeError(f'graph replay compiler was not found: {compiler[0]}')
    compiler[0] = resolved
    includes = include_paths('cpu'); libraries = library_paths('cpu')
    version = subprocess.check_output(compiler + ['--version'], text=True)
    frozen_source = source.read_bytes()
    # Newer Torch headers use C++20 concepts and three-way comparisons.
    cxx_standard = 'c++20'
    identity = (frozen_source, abi_headers, torch.__version__, torch.compiled_with_cxx11_abi(),
                includes, libraries, compiler, version, cxx_standard)
    key = hashlib.sha256(repr(identity).encode()).hexdigest()[:24]
    namespace = 'vecops_graph_replay_' + key
    root = Path(os.environ.get('VECOPS_GRAPH_REPLAY_CACHE',
                               default_cache_dir() / 'graph-replay')).expanduser() / key
    root.mkdir(parents=True, exist_ok=True)
    library = root / 'executor.so'
    with file_lock(root / 'build.lock'):
      if not library.exists():
        temporary = root / ('executor-' + str(os.getpid()) + '.so')
        frozen_path = root / 'executor.cpp'
        frozen_path.write_bytes(frozen_source)
        frozen_includes = root / 'include'
        header_root = frozen_includes / 'vecops/runtime'
        header_root.mkdir(parents=True, exist_ok=True)
        for name, contents in abi_headers.items():
          (header_root / name).write_bytes(contents)
        command = compiler + ['-std=' + cxx_standard, '-O2', '-shared', '-fPIC',
          f'-D_GLIBCXX_USE_CXX11_ABI={int(torch.compiled_with_cxx11_abi())}',
          '-DVECOPS_GRAPH_REPLAY_NAMESPACE=' + namespace,
          '-I' + str(frozen_includes), *['-I' + str(p) for p in includes], str(frozen_path), '-o', str(temporary),
          *['-L' + str(p) for p in libraries],
          *['-Wl,-rpath,' + str(p) for p in libraries], '-ltorch', '-ltorch_cpu', '-lc10']
        # No -fopenmp, libgomp, SDK or Python extension linkage: use only this
        # already-loaded Torch build. In particular never add a second OMP runtime.
        try:
          subprocess.run(command, check=True, stdout=subprocess.PIPE,
                         stderr=subprocess.STDOUT, text=True)
          temporary.replace(library)
        except subprocess.CalledProcessError as error:
          raise RuntimeError('graph replay extension build failed:\n' + error.stdout) from error
        finally:
          temporary.unlink(missing_ok=True)
        (root / 'build-command.txt').write_text(shlex.join(command) + '\n')
    torch.ops.load_library(str(library))
    _executor_class = getattr(getattr(torch.classes, namespace), 'Executor')
    _executor_namespace = namespace
    return _executor_class


def _check_context():
  import torch
  from torch.utils._python_dispatch import _get_current_dispatch_mode
  if _get_current_dispatch_mode() is not None:
    raise RuntimeError('graph replay does not support an active TorchDispatchMode')
  if torch.overrides._get_current_function_mode() is not None:
    raise RuntimeError('graph replay does not support an active TorchFunctionMode')
  if torch.is_autocast_enabled('cpu'):
    raise RuntimeError('graph replay does not support CPU autocast contexts')
  forbidden = ('PythonDispatcher', 'PreDispatch', 'Functionalize',
               'FuncTorchDynamicLayerFrontMode', 'FuncTorchDynamicLayerBackMode',
               'FuncTorchBatched', 'FuncTorchVmapMode', 'FuncTorchGradWrapper')
  included = torch._C._dispatch_tls_local_include_set()
  if any(included.has(getattr(torch._C.DispatchKey, key)) for key in forbidden):
    raise RuntimeError('graph replay rejects Python/functional/pre-dispatch transformation contexts')
  if torch.compiler.is_compiling():
    raise RuntimeError('graph replay cannot be captured by torch.compile')


def _check_tensor(value):
  import torch
  if type(value) not in (torch.Tensor, torch.nn.Parameter):
    raise TypeError('graph replay rejects Tensor subclasses, FakeTensors and tensor modes')
  if value.device.type != 'cpu' or value.layout != torch.strided:
    raise ValueError('graph replay supports only dense-strided CPU tensors')
  if value.is_quantized or value.is_nested or value.is_conj() or value.is_neg():
    raise ValueError('graph replay does not support quantized, nested or lazy conjugate/negative tensors')
  if any(name is not None for name in getattr(value, 'names', ())):
    raise ValueError('graph replay does not support named tensors')


@contextlib.contextmanager
def _reject_tensor_data_escapes():
  import torch
  from torch.overrides import TorchFunctionMode
  class Guard(TorchFunctionMode):
    def __torch_function__(self, function, types, args=(), kwargs=None):
      if function in (torch.Tensor.numpy, torch.Tensor.tolist, torch.Tensor.item,
                      torch.Tensor.__array__, torch.Tensor.storage,
                      torch.Tensor.untyped_storage, torch.Tensor.data_ptr,
                      torch.Tensor._typed_storage, torch._C._is_alias_of):
        raise ValueError('unsupported tensor data escape from the captured program: ' + function.__name__)
      return function(*args, **(kwargs or {}))
  # This private C helper bypasses TorchFunctionMode. Detect this one explicit
  # unsupported storage-identity escape; do not attempt a general Python sandbox.
  import sys
  previous=sys.getprofile()
  if previous is not None and not callable(previous):
    raise RuntimeError('capture with an active C profiler is unsupported; capture first, then profile replay')
  def profile(frame,event,argument):
    if event=='c_call' and argument is torch._C._is_alias_of:
      raise ValueError('unsupported tensor data escape: storage identity observation')
    if previous is not None:previous(frame,event,argument)
  with Guard():
    sys.setprofile(profile)
    try:
      yield
    finally:
      sys.setprofile(previous)


@dataclass
class _StorageOrigin:
  weak: Any
  begin: int
  end: int


@dataclass(frozen=True)
class _Ref:
  id: str


@dataclass
class _Instruction:
  event_id: str
  function: Any
  arguments: tuple[Any, ...]
  result: Any
  matmul_precision: str


class _Tape:
  def __init__(self, capture, args, kwargs):
    from torch.utils._pytree import tree_flatten
    self.capture = weakref.proxy(capture)
    self.instructions = []
    self.tensors = {}
    self.produced = set()
    self.produced_storage_origins = {}
    self.tensor_spans = {}
    self.input_leaves, self.input_spec = tree_flatten((args, kwargs))
    self.input_refs = self.encode(self.input_leaves)
    self.output_leaves = None
    self.output_spec = None
    self.module_owner = None
    self.native_operators = {}
    from vecops.planner.runtime import current_session
    self.session_owner = current_session()
    self.session_scopes = None if self.session_owner is None else list(self.session_owner._scope)
    self.storage_bindings = {}
    self.arena_leases = []

  def encode(self, value, *, produced=False):
    import torch
    if isinstance(value, torch.Tensor):
      _check_tensor(value)
      identity = self.capture.tensor_id(value)
      # Keep only byte metadata for produced values, never their Tensor owners.
      # This also records derived aliases after their eager objects have died.
      with torch._C.DisableTorchFunction():
        start = value.data_ptr()
        extent = 0 if not value.numel() else 1 + sum(
            (size - 1) * stride for size, stride in zip(value.shape, value.stride()))
        self.tensor_spans[identity] = (start, start + extent * value.element_size())
      if produced:
        self.produced.add(identity)
      elif identity not in self.produced:
        if identity not in self.tensors:
          storage = value.untyped_storage()
          low, high = storage.data_ptr(), storage.data_ptr() + storage.nbytes()
          for key, origin in tuple(self.produced_storage_origins.items()):
            if origin.weak.expired():
              del self.produced_storage_origins[key]
            elif high > origin.begin and low < origin.end:
              raise ValueError('unrecorded storage alias of a produced tensor; do not disable dispatcher recording')
        # Only inputs and values entering from outside the dispatcher program
        # need a strong storage owner. Produced temporaries follow eager life:
        # GraphCapture's weakrefs/live-storage counters prevent identity reuse.
        self.tensors[identity] = value
      return _Ref(identity)
    if isinstance(value, torch.Generator):
      raise ValueError('explicit Generator objects are unsupported; use the default CPU generator')
    if isinstance(value, tuple):
      return tuple(self.encode(x, produced=produced) for x in value)
    if isinstance(value, list):
      return [self.encode(x, produced=produced) for x in value]
    if isinstance(value, dict):
      if not all(isinstance(x, str) for x in value):
        raise TypeError('graph replay dictionary keys must be strings')
      return {k: self.encode(v, produced=produced) for k, v in value.items()}
    if value is None or type(value) in (int, float, bool, str) or isinstance(
        value, (torch.dtype, torch.device, torch.layout, torch.memory_format)):
      return value
    raise TypeError(f'unsupported graph replay value type {type(value).__name__}')

  @staticmethod
  def validate_operator(function, args, kwargs):
    import torch
    schema = function._schema
    if torch.is_autocast_enabled('cpu'):
      raise ValueError('graph capture does not support CPU autocast contexts')
    tags = set(function.tags)
    denied = {getattr(torch.Tag, name) for name in ('dynamic_output_shape',
              'data_dependent_output', 'inplace_view') if hasattr(torch.Tag, name)}
    static_integer_index = schema.name in {'aten::index', 'aten::_unsafe_index'} and len(args) > 1 and all(
        x is None or isinstance(x, torch.Tensor) and x.dtype in (torch.int32, torch.int64)
        for x in args[1])
    inference_detach = schema.name == 'aten::detach_' and args and not args[0].requires_grad
    if (tags & denied and not static_integer_index and not inference_detach) or schema.name in {'aten::resize_', 'aten::resize_as_', 'aten::set_',
                                       'aten::record_stream', 'aten::_local_scalar_dense'}:
      raise ValueError(f'unsupported data-dependent, metadata-mutating or asynchronous operator: {schema}')
    # Seeded ATen CPU random ops are intentionally allowed: the native kernel
    # advances the ordinary default Generator on every replay.
    if any(str(argument.type).startswith('Generator') for argument in schema.arguments):
      # None is checked by encode; an explicit Generator is rejected there.
      pass

  def record(self, function, args, kwargs, result, event):
    import torch
    schema = function._schema
    values = []
    for index, argument in enumerate(schema.arguments):
      if index < len(args):
        values.append(args[index])
      elif argument.name in kwargs:
        values.append(kwargs[argument.name])
      elif argument.has_default_value():
        values.append(argument.default_value)
      else:
        raise ValueError(f'missing schema argument {argument.name}')
      if argument.name == 'dtype' and values[-1] is None and isinstance(result, torch.Tensor):
        # Native factories/reductions must not consult a later ambient default
        # dtype: freeze the actual captured result dtype, including like ops.
        values[-1] = result.dtype
    self.instructions.append(_Instruction(event.id, function,
        self.encode(tuple(values)), self.encode(result, produced=True), torch.get_float32_matmul_precision()))
    # Weak StorageImpl references distinguish a live hidden alias from allocator
    # address reuse without retaining intermediate Tensor payloads. In particular
    # An explicitly hidden alias can outlive its original Tensor object.
    from torch.multiprocessing.reductions import StorageWeakRef
    from .torch import tensor_leaves
    for tensor in tensor_leaves(result):
      identity = self.capture.tensor_id(tensor)
      storage = self.capture.storages[self.capture.tensors[identity].storage_id]
      if storage.allocation_event == event.id:
        physical = tensor.untyped_storage()
        self.produced_storage_origins[physical._cdata] = _StorageOrigin(
            StorageWeakRef(physical), physical.data_ptr(), physical.data_ptr() + physical.nbytes())



def capture_torch(capture, function, args, kwargs, *, preserve_rng_state):
  """Execute once to record the same IR and an optional executable payload."""
  import torch
  import vecops
  from .torch import record_torch, register_inputs, register_module_state, mark_graph_outputs, tensor_leaves
  if (capture.events or capture._finished or capture._active_token is not None
      or getattr(capture, '_torch_replay_failed', False)):
    raise RuntimeError('executable capture requires a fresh inactive GraphCapture')
  _check_context()
  if vecops.is_precompiling():
    raise RuntimeError('executable capture requires actual eager kernels, not precompile collection')
  if not isinstance(preserve_rng_state, bool):
    raise TypeError('preserve_rng_state must be bool')
  # Capture may fork the default RNG and temporarily change Torch's global
  # math policy. Use the same native gate as replay, including direct C++
  # submissions. Metadata-only/precompile GraphCapture never enters this path.
  _load_executor()
  lease = getattr(getattr(torch.classes, _executor_namespace), 'CaptureLease')()
  try:
    with capture:
      register_inputs(capture, (args, kwargs))
      tape = _Tape(capture, args, kwargs)
      owner = function if hasattr(function, 'named_parameters') else getattr(function, '__self__', None)
      if owner is not None and hasattr(owner, 'named_parameters'):
        register_module_state(capture, owner)
        tape.module_owner = owner
      capture._torch_replay_tape = tape
      # A real capture executes once and may mutate inputs/state. By default
      # only the CPU RNG state is restored, including when capture raises.
      rng = torch.random.fork_rng(devices=[], enabled=preserve_rng_state)
      with rng, torch.inference_mode(), record_torch(capture), _reject_tensor_data_escapes():
        result = function(*args, **kwargs)
      from .torch import register_tensor
      for tensor in tensor_leaves(result):
        try:
          capture.tensor_id(tensor)
        except KeyError:
          register_tensor(capture, tensor, kind='external', external=True)
      mark_graph_outputs(capture, result)
      from torch.utils._pytree import tree_flatten
      leaves, tape.output_spec = tree_flatten(result)
      tape.output_leaves = tape.encode(leaves)
      # Freeze each TensorImpl/storage handle, not its contents. In-place
      # updates remain visible, while a PlanSession closing and restoring a
      # parameter's .data cannot redirect this graph to another allocation.
      tape.storage_bindings = {key:tensor.detach() for key,tensor in tape.tensors.items()}
      from ._replay_arena import capture_leases
      tape.arena_leases = capture_leases(tape.session_owner)
    capture._torch_execution_graph = capture.finish()
    return result
  except BaseException:
    tape = getattr(capture, '_torch_replay_tape', None)
    if tape is not None and tape.session_owner is not None and not tape.session_owner._closed:
      tape.session_owner._scope[:] = tape.session_scopes
    capture._torch_replay_tape = None
    capture._torch_replay_failed = True
    raise
  finally:
    # Do not rely on traceback/cyclic-GC lifetime to release this global gate.
    lease.release()


def _qualified(function):
  schema = function._schema
  return schema.name + ('.' + schema.overload_name if schema.overload_name else '')


def _native_owners(functions, explicit):
  """Require actual C++ registration evidence, not just an op namespace."""
  import torch
  from vecops import _schema_bridge
  owners = list(explicit.values())
  registrations = getattr(_schema_bridge, '_registrations', {})
  for function in functions:
    name = _qualified(function);schema = function._schema
    table = torch._C._dispatch_dump_table(name)
    cpu = next((line for line in table.splitlines() if line.startswith('CPU:')), '')
    if not cpu or not any(suffix in cpu for suffix in ('.cpp:', '.cc:', '.cxx:')):
      raise ValueError(f'operator has no proven C++ CPU implementation: {name}: {cpu}')
    if schema.name.startswith('aten::'):
      if not any(marker in cpu for marker in ('/aten/', '/ATen/', '\\aten\\', '\\ATen\\')):
        raise ValueError(f'ATen CPU registration is not a packaged native kernel: {name}: {cpu}')
      continue
    key = tuple(schema.name.split('::', 1))
    registration = registrations.get(key)
    if registration is not None and registration.materialized and registration.library_path:
      owners.append(registration)
    elif name not in explicit:
      raise ValueError(f'custom operator requires an explicit native registration owner: {name}')
  # ctypes/CDLL modules, compiler/operator state and registrations remain alive.
  modules = getattr(_schema_bridge, '_loaded_bridge_modules', {})
  owners.extend(modules.values() if hasattr(modules, 'values') else modules)
  return owners


def compile_torch(capture, *, memory_plan=None, system=None, domain=None,
                  large_pages=True, native_operators=None, borrow_readonly_state=False,
                  native_pointwise=False, scratch_allocator=None, borrow_outputs=False, native_donation=False):
  """Compile transactionally and release failed-attempt arena ownership.

  Inactive compiler/allocator frame locals are cleared on error, so retaining
  the exception does not retain temporary arenas. Error text and traceback
  file/line structure remain available; caller frames have not yet unwound.
  """
  tape = getattr(capture, '_torch_replay_tape', None)
  saved = None if tape is None else (dict(tape.tensors), dict(tape.storage_bindings),
                                    list(tape.arena_leases), tape.session_owner)
  try:
    return _compile_torch_impl(capture, memory_plan=memory_plan, system=system,
        domain=domain, large_pages=large_pages, native_operators=native_operators,
        borrow_readonly_state=borrow_readonly_state, native_pointwise=native_pointwise,
        scratch_allocator=scratch_allocator, borrow_outputs=borrow_outputs, native_donation=native_donation)
  except BaseException as error:
    if saved is not None:
      tape.tensors, tape.storage_bindings, tape.arena_leases, tape.session_owner = saved
    import traceback
    traceback.clear_frames(error.__traceback__)
    if hasattr(error, 'add_note'):
      error.add_note('Graph compile temporaries released; inactive compiler frame locals cleared, traceback locations preserved.')
    raise


def _compile_torch_impl(capture, *, memory_plan=None, system=None, domain=None,
                        large_pages=True, native_operators=None, borrow_readonly_state=False,
                        native_pointwise=False, scratch_allocator=None, borrow_outputs=False, native_donation=False):
  import torch
  tape = getattr(capture, '_torch_replay_tape', None)
  if tape is None or getattr(capture, '_torch_replay_failed', False):
    raise RuntimeError('call GraphCapture.capture_torch successfully before compile_torch')
  _check_context()
  if not isinstance(borrow_readonly_state, bool):
    raise TypeError('borrow_readonly_state must be bool')
  if not isinstance(native_donation, bool):
    raise TypeError('native_donation must be bool')
  if native_donation and not native_pointwise:
    raise ValueError('native_donation requires native_pointwise')
  if not isinstance(native_pointwise, bool):
    raise TypeError('native_pointwise must be bool')
  if not isinstance(borrow_outputs, bool):
    raise TypeError('borrow_outputs must be bool')
  if borrow_outputs and not native_pointwise:
    raise ValueError('borrow_outputs requires native_pointwise lowering')
  if scratch_allocator is not None and not callable(scratch_allocator):
    raise TypeError('scratch_allocator must be callable')
  explicit = dict(native_operators or {})
  explicit.update(tape.native_operators)
  owners = _native_owners((i.function for i in tape.instructions), explicit)
  executor_class = _load_executor()
  graph = torch._C.Graph();values = {};bound = [];examples = [];input_ids = [];bound_ids = set();bound_order = []
  for leaf, ref in zip(tape.input_leaves, tape.input_refs):
    if isinstance(ref, _Ref):
      item = graph.addInput('input_' + str(len(examples)))
      item.setType(torch._C.TensorType.get());values[ref.id] = item
      examples.append(leaf);input_ids.append(ref.id)
  produced = set()
  def refs(value):
    if isinstance(value, _Ref):
      yield value.id
    elif isinstance(value, (list, tuple)):
      for item in value:
        yield from refs(item)
    elif isinstance(value, dict):
      for item in value.values():
        yield from refs(item)
  for instruction in tape.instructions:
    for identity in refs(instruction.arguments):
      if identity not in values and identity not in produced:
        item = graph.addInput('bound_' + str(len(bound)));item.setType(torch._C.TensorType.get())
        values[identity] = item;bound.append(tape.storage_bindings[identity]);examples.append(tape.storage_bindings[identity])
        bound_ids.add(identity);bound_order.append(identity)
    produced.update(refs(instruction.result))
  for identity in refs(tape.output_leaves):
    if identity not in values and identity not in produced:
      item = graph.addInput('bound_' + str(len(bound)));item.setType(torch._C.TensorType.get())
      values[identity] = item;bound.append(tape.storage_bindings[identity]);examples.append(tape.storage_bindings[identity])
      bound_ids.add(identity);bound_order.append(identity)
  from ._replay_arena import isolate_bindings
  isolated_bindings, arena_owners, arena_leases, isolated_ids, arena_rows = isolate_bindings(
      {identity:tape.storage_bindings[identity] for identity in bound_order},
      examples[:len(input_ids)], tape.arena_leases,
      borrow_readonly_state=borrow_readonly_state,
      tensor_spans=tape.tensor_spans,
      written_ids={use.tensor_id for event in capture._torch_execution_graph.events
                   for use in event.uses if use.access.value in {'write', 'read_write'}})
  bound = [isolated_bindings[identity] for identity in bound_order]
  examples = examples[:len(input_ids)] + bound
  owners.extend(arena_owners)
  isolated_storages = {capture._torch_execution_graph.tensors[x].storage_id for x in isolated_ids}
  isolated_aliases = {identity for identity,tensor in capture._torch_execution_graph.tensors.items()
                      if tensor.storage_id in isolated_storages}
  planned, plan_owners, unplanned = _plan_views(capture, memory_plan, system, domain, large_pages, tape)
  owners.extend(plan_owners)
  for identity, tensor in planned.items():
    item = graph.addInput('planned_' + identity);item.setType(torch._C.TensorType.get())
    values[identity] = item;bound.append(tensor);examples.append(tensor)
  planned_storages = {capture._torch_execution_graph.tensors[x].storage_id for x in planned}
  planned_aliases = {identity for identity,tensor in capture._torch_execution_graph.tensors.items()
                     if tensor.storage_id in planned_storages}
  # Lower statically guarded ATen intermediates into recordable C++ grids.
  # Escaping outputs keep the original functional-allocation contract.
  execution = capture._torch_execution_graph
  pointwise = {}
  pointwise_inplace = {}
  donations = {}
  replacements = {}
  native_rewrites = {}
  prepared_cache = {}
  opaque_tensors = set()
  tensor_spans = tape.tensor_spans
  static_factories = set()
  scratch_owner = None
  scratch_required_bytes = 0
  output_owner = None
  retained_output_count = 0
  if native_pointwise and os.environ.get('VECOPS_GRAPH_POINTWISE', '1') != '0':
    from . import _pointwise, _reduction, _int_scalar
    from vecops import language as T
    from vecops._arm_vector_state import get_vector_state
    pointwise_vector_token = get_vector_state().env_token
    def metadata(ref):
      return execution.tensors[ref.id].metadata
    for ordinal, instruction in enumerate(tape.instructions):
      result_ref = instruction.result
      if not isinstance(result_ref, _Ref):
        continue
      tensor = execution.tensors[result_ref.id]
      inplace = instruction.function._schema.name in {'aten::add_', 'aten::sub_', 'aten::mul_', 'aten::div_', 'aten::copy_', 'aten::fill_'}
      if (not inplace and not borrow_outputs and
          any(execution.tensors[x].escapes for x in execution.storages[tensor.storage_id].tensors)):
        continue
      reduction_input = (_reduction.select(instruction.function._schema, instruction.arguments, result_ref, metadata)
          if os.environ.get('VECOPS_GRAPH_SHORT_REDUCTION', '1') != '0' else None)
      integer = (_int_scalar.select(instruction.function._schema, instruction.arguments, result_ref, metadata)
          if os.environ.get('VECOPS_GRAPH_INTEGER_SCALARS', '1') != '0' else None)
      selected = ((31, integer[0], integer[0], integer[1]) if integer is not None else
          (30, reduction_input, reduction_input, 0.) if reduction_input is not None else
          _pointwise.select(instruction.function._schema, instruction.arguments, result_ref, metadata))
      if selected is None:
        continue
      code, a, b, alpha = selected
      output_ref = instruction.arguments[0] if inplace else result_ref
      am, bm, om = metadata(a), metadata(b), metadata(output_ref)
      # SIMD inner vectors require contiguous/zero-stride inner operands.
      # Other layouts retain ATen's general implementation.
      if any(m.device != 'cpu' or (m.shape and m.shape[-1] != 1 and m.strides[-1] != 1)
             for m in (am, bm, om)):
        continue
      contiguous = _pointwise.contiguous_layout(om.shape, om.strides)
      strided_copy = (instruction.function._schema.name == 'aten::copy_'
                      and _pointwise.nonoverlapping_layout(om.shape, om.strides))
      if contiguous or strided_copy:
        if code == 31:
          compiled = _int_scalar.kernel(am.shape, am.strides, om.shape, om.strides,
              int(os.environ.get('VECOPS_NUM_THREADS', torch.get_num_threads())), pointwise_vector_token)
        elif code == 30:
          compiled = _reduction.kernel(am.shape, am.strides, am.dtype.removeprefix('torch.'),
              om.shape, om.strides, om.dtype.removeprefix('torch.'),
              int(os.environ.get('VECOPS_NUM_THREADS', torch.get_num_threads())), pointwise_vector_token)
        else:
          compiled = _pointwise.kernel(code, am.shape or (1,), am.strides or (1,), am.dtype.removeprefix('torch.'),
            bm.shape or (1,), bm.strides or (1,), bm.dtype.removeprefix('torch.'), om.shape or (1,), om.strides or (1,),
            om.dtype.removeprefix('torch.'), int(os.environ.get('VECOPS_NUM_THREADS', torch.get_num_threads())),
            pointwise_vector_token)
        pointwise[ordinal] = (compiled, a, b, output_ref, alpha)
        if inplace:
          pointwise_inplace[ordinal] = (result_ref, output_ref)
    if pointwise:
      T.compile_pending()
      import vecops
      def tensor_meta(ref):
        m = metadata(ref)
        return vecops.TensorMeta(m.shape or (1,), strides=m.strides or (1,),
                                 dtype=m.dtype.removeprefix('torch.'))
      pending = {id(compiled):compiled for compiled, *_ in pointwise.values()
                 if not getattr(compiled, '_graph_artifact_prepared', False)}
      if pending:
        with vecops.collect_compile_requests() as collector:
          for compiled, a, b, result_ref, alpha in pointwise.values():
            if id(compiled) in pending:
              compiled(tensor_meta(a), tensor_meta(b), tensor_meta(result_ref), alpha)
        vecops.compile_batch(collector.requests,
            execution_parallelism=int(os.environ.get('VECOPS_NUM_THREADS', torch.get_num_threads())))
        for compiled in pending.values():
          compiled._graph_artifact_prepared = True
      owners.extend(_native_owners((x[0].explicit_torch_binding()[0] for x in pointwise.values()), {}))
    for instruction in tape.instructions:
      if instruction.function._schema.name not in {'aten::empty', 'aten::empty_strided', 'aten::empty_like'}:
        continue
      if not isinstance(instruction.result, _Ref) or instruction.result.id in planned:
        continue
      identity = instruction.result.id
      tensor = execution.tensors[identity]
      storage = execution.storages[tensor.storage_id]
      if (tensor.metadata.device == 'cpu' and not storage.external and not storage.persistent
          and (borrow_outputs or not any(execution.tensors[x].escapes for x in storage.tensors))):
        static_factories.add(identity)
    if native_donation:
      from ._donation import plan_reuse
      from vecops import _schema_bridge
      native_contracts, native_outputs = {}, {}
      for ordinal, instruction in enumerate(tape.instructions):
        if ordinal in pointwise:
          continue
        prepared = _schema_bridge.prepare_replay_plan(instruction.function, instruction.arguments,
            lambda ref: execution.tensors[ref.id].metadata)
        prepared_cache[ordinal] = prepared
        if prepared is None:
          if not instruction.function._schema.name.startswith('aten::'):
            opaque_tensors.update(refs(instruction.arguments))
            opaque_tensors.update(refs(instruction.result))
          continue
        lease, flags, _, _ = prepared
        # Only SDK source kernels export this contract. Communication/custom
        # providers keep their independent completion and ownership semantics.
        if (not hasattr(lease, 'inplace_contracts') or
            instruction.function._schema.name in _schema_bridge._replay_plan_providers):
          opaque_tensors.update(refs(instruction.arguments))
          opaque_tensors.update(refs(instruction.result))
          continue
        paths = getattr(lease, 'argument_paths', None)
        # Source ABI indices describe call values. Nested/custom paths need a
        # separate ownership proof; never interpret them as schema positions.
        if (len(flags) > len(instruction.arguments)
            or any(isinstance(value, (list, tuple, dict))
                   for value in instruction.arguments[:len(flags)])
            or any(refs(instruction.arguments[len(flags):]))
            or (paths is not None and tuple(paths) != tuple(map(str, range(len(flags)))))):
          opaque_tensors.update(refs(instruction.arguments))
          opaque_tensors.update(refs(instruction.result))
          continue
        contracts = lease.inplace_contracts
        if any(i >= len(flags) or o >= len(flags) or flags[i] != 1 or flags[o] != 2
               for i,o in contracts):
          raise ValueError('invalid native inplace input/output access contract')
        native_contracts[ordinal] = contracts
        native_outputs[ordinal] = [value for value,flag in zip(instruction.arguments,flags)
                                   if flag == 2 and isinstance(value, _Ref)]
      eligible = {ordinal:value for ordinal,value in pointwise.items()
                  if ordinal not in pointwise_inplace
                  and tape.instructions[ordinal].function._schema.name in {
                      'aten::add', 'aten::sub', 'aten::mul', 'aten::div', 'aten::relu', 'aten::neg', 'aten::pow'}}
      execution, donations, replacements, native_rewrites, tensor_spans = plan_reuse(
          execution, tape.instructions, eligible, pointwise_inplace,
          set(planned) | set(input_ids) | bound_ids | opaque_tensors, tensor_spans,
          native_contracts, native_outputs, static_factories)
      pointwise.update(eligible)
      pointwise_inplace.update(donations)
      static_factories.difference_update(replacements)
    scratch_ids = static_factories | {ref.id for ordinal,(_,_,_,ref,_) in pointwise.items()
                                      if ordinal not in pointwise_inplace}
    if scratch_ids:
      from ._replay_scratch import allocate
      retained = {identity for identity in scratch_ids if any(execution.tensors[x].escapes
          for x in execution.storages[execution.tensors[identity].storage_id].tensors)}
      scratch = {}
      transient = scratch_ids - retained
      # Retained results coexist with transient work and may feed many later
      # replays. Bind them before reusable temporaries consume common capacity.
      if retained:
        outputs, output_owner, _ = allocate(execution, sorted(retained), allow_escaping=True)
        owners.append(output_owner)
        scratch.update(outputs)
        retained_output_count = len(retained)
      if transient:
        intermediates, scratch_owner, scratch_required_bytes = allocate(
            execution, sorted(transient), allocator=scratch_allocator)
        owners.append(scratch_owner)
        scratch.update(intermediates)
      for identity, owned in scratch.items():
        item = graph.addInput('scratch_' + identity);item.setType(torch._C.TensorType.get())
        values[identity] = item;bound.append(owned);examples.append(owned)
      planned.update({identity:scratch[identity] for identity in static_factories})
      planned_aliases.update(identity for identity,tensor in execution.tensors.items()
                             if tensor.storage_id in {execution.tensors[x].storage_id for x in static_factories})
  def donated_identity(identity):
    # A donated target can be a view, whose JIT Value is constructed by the
    # static-view pass or ordinary instruction lowering below. Resolve uses
    # lazily rather than requiring every view to exist before those passes.
    while identity in replacements:
      identity = replacements[identity].id
    return identity
  def value(item, expected=None):
    if isinstance(item, _Ref):
      return values[donated_identity(item.id)]
    if isinstance(expected, torch._C.TensorType) and type(item) in (int, float, bool):
      # TorchDispatch may present a Python scalar to a Tensor schema argument;
      # reproduce the dispatcher's wrapped-number conversion in native IR.
      node = graph.create(_executor_namespace + '::wrapped_scalar', [graph.insertConstant(item)], 1)
      graph.appendNode(node);node.output().setType(torch._C.TensorType.get())
      return node.output()
    if isinstance(item, list):
      element = expected.getElementType() if isinstance(expected, torch._C.ListType) else None
      node = graph.create('prim::ListConstruct', [value(x, element) for x in item], 1)
      if expected is None:
        raise ValueError('a list argument requires an unambiguous schema list type')
      graph.appendNode(node);node.output().setType(expected);return node.output()
    if isinstance(item, tuple):
      node = graph.create('prim::TupleConstruct', [value(x) for x in item], 1)
      graph.appendNode(node);node.output().setType(torch._C.TupleType([v.type()for v in node.inputs()]))
      return node.output()
    if isinstance(item, dict):
      raise ValueError('dispatcher dictionary arguments are not supported')
    if isinstance(item, torch.memory_format):
      item = {torch.contiguous_format:0, torch.preserve_format:1,
              torch.channels_last:2, torch.channels_last_3d:3}[item]
    return graph.insertConstant(item)
  def assign(encoded, result):
    if isinstance(encoded, _Ref):
      values[encoded.id] = result
    elif isinstance(encoded, (tuple, list)):
      kind = 'prim::TupleUnpack' if isinstance(encoded, tuple) else 'prim::ListUnpack'
      node = graph.create(kind, [result], len(encoded));graph.appendNode(node)
      for item, output in zip(encoded, node.outputs()):
        output.setType(torch._C.TensorType.get() if isinstance(item, _Ref) else torch._C.AnyType.get())
        assign(item, output)
  # Static views of already-bound storage do not depend on data computation.
  # Construct immutable bound aliases once; input aliases are constructed once
  # at submission, before all grids, and follow rebound input storage addresses.
  alias_nodes = set()
  static_views = {'aten::view', 'aten::_unsafe_view', 'aten::reshape', 'aten::reshape_as',
                  'aten::select', 'aten::slice', 'aten::narrow', 'aten::unsqueeze', 'aten::squeeze',
                  'aten::expand', 'aten::transpose', 'aten::permute', 'aten::detach',
                  'aten::alias', 'aten::broadcast_to', 'aten::unflatten',
                  'aten::movedim', 'aten::flatten', 'aten::split',
                  'aten::split_with_sizes', 'aten::unbind'}
  anchors = {}
  for identity, item in values.items():
    if item.node().kind() == 'prim::Param':
      anchors.setdefault(execution.tensors[identity].storage_id, []).append(identity)
  bound_tensors = dict(zip(input_ids + bound_order, examples[:len(input_ids)+len(bound_order)]))
  bound_tensors.update(planned)
  for ordinal, (_, _, _, ref, _) in pointwise.items():
    if ordinal not in pointwise_inplace:
      bound_tensors[ref.id] = examples[list(graph.inputs()).index(values[ref.id])]
  for ordinal, instruction in enumerate(tape.instructions):
    name = instruction.function._schema.name
    if name not in static_views:
      continue
    # narrow.Tensor reads its start from tensor data. Only the integer
    # overload has geometry fixed by the captured, guarded call contract.
    if name == 'aten::narrow' and any(type(x) is not int for x in instruction.arguments[1:]):
      continue
    if isinstance(instruction.result, _Ref):
      outputs = (instruction.result,)
    elif (name in {'aten::split', 'aten::split_with_sizes', 'aten::unbind'}
          and isinstance(instruction.result, (tuple, list))
          and all(isinstance(item, _Ref) for item in instruction.result)):
      outputs = instruction.result
    else:
      continue
    views = []
    for ref in outputs:
      identity = ref.id
      tensor = execution.tensors[identity]
      low, high = tensor_spans[identity]
      root = next((candidate for candidate in anchors.get(tensor.storage_id, ())
          if tensor.metadata.dtype == execution.tensors[candidate].metadata.dtype
          and tensor_spans[candidate][0] <= low
          and high <= tensor_spans[candidate][1]), None)
      if root is None:
        break  # E.g. flatten of a noncontiguous Tensor really allocates.
      meta = tensor.metadata
      # Relocation can compact old arena offsets. Preserve actual relative
      # byte addresses, and validate every result before replacing a list op.
      delta = tensor_spans[identity][0] - tensor_spans[root][0]
      if delta % meta.element_size:
        break
      offset = delta // meta.element_size
      if root not in input_ids:
        base = bound_tensors[root]
        first = base.storage_offset() + offset
        extent = 0 if any(x == 0 for x in meta.shape) else 1 + sum(
            (size - 1) * stride for size, stride in zip(meta.shape, meta.strides))
        if first < 0 or (first + extent) * meta.element_size > base.untyped_storage().nbytes():
          break
      views.append((identity, root, meta, offset))
    if len(views) != len(outputs):
      continue
    for identity, root, meta, offset in views:
      if root in input_ids:
        def integer_list(items):
          if items:
            return graph.insertConstant(list(items))
          # Python [] loses its element type at the IValue boundary. Fold a
          # separate, pure typed-list graph; never optimize the native tape
          # here, where mutable/void calls must retain their ordering.
          literal = torch._C.Graph()
          empty = literal.create('prim::ListConstruct', [], 1)
          empty.output().setType(torch._C.ListType.ofInts())
          literal.appendNode(empty);literal.registerOutput(empty.output())
          torch._C._jit_pass_constant_propagation(literal)
          node = graph.createClone(next(literal.nodes()), lambda value: value)
          graph.appendNode(node)
          return node.output()
        node = graph.create(_executor_namespace + '::static_view',
            [values[root], integer_list(meta.shape), integer_list(meta.strides),
             graph.insertConstant(offset)], 1)
        graph.appendNode(node);node.output().setType(torch._C.TensorType.get())
        values[identity] = node.output()
      else:
        base = bound_tensors[root]
        alias = base.as_strided(meta.shape, meta.strides, base.storage_offset() + offset)
        item = graph.addInput('alias_' + identity);item.setType(torch._C.TensorType.get())
        values[identity] = item;bound.append(alias);examples.append(alias)
    alias_nodes.add(ordinal)
  prepared_nodes = 0
  static_noops = 0
  for ordinal, instruction in enumerate(tape.instructions):
    if ordinal in native_rewrites:
      instruction = replace(instruction, arguments=native_rewrites[ordinal])
    if ordinal in alias_nodes:
      continue
    # Torch's composite to/contiguous entry can be recorded even when it
    # returned the exact input TensorImpl. Static input metadata guarantees
    # that this remains an identity, so it must not break a resident island.
    if (native_pointwise and instruction.function._schema.name in {'aten::to', 'aten::contiguous'}
        and isinstance(instruction.result, _Ref) and instruction.arguments
        and isinstance(instruction.arguments[0], _Ref)
        and instruction.result.id == instruction.arguments[0].id):
      static_noops += 1
      continue
    pointwise_output = None
    if ordinal in pointwise:
      compiled, a, b, pointwise_output, alpha = pointwise[ordinal]
      if ordinal in pointwise_inplace:
        original_result, target = pointwise_inplace[ordinal]
        values[original_result.id] = value(target)
      function, specs = compiled.explicit_torch_binding()
      instruction = _Instruction(instruction.event_id, function,
          (a, b, pointwise_output, alpha, *specs), None, instruction.matmul_precision)
    schema = instruction.function._schema
    if schema.name in {'aten::storage_offset', 'aten::as_strided'} and any(
        x in planned_aliases or x in isolated_aliases for x in refs(instruction.arguments)):
      # A relative view inherits the relocated self offset. Permit only a
      # captured byte range already retained by that self: compaction need not
      # retain other rows merely because they shared the original arena owner.
      relative = False
      if (schema.name == 'aten::as_strided' and len(instruction.arguments) == 4
          and instruction.arguments[3] is None
          and isinstance(instruction.arguments[0], _Ref)
          and isinstance(instruction.result, _Ref)):
        source_id = instruction.arguments[0].id
        result_id = instruction.result.id
        source = capture._torch_execution_graph.tensors[source_id]
        output = capture._torch_execution_graph.tensors[result_id]
        low, high = tensor_spans[source_id]
        start, end = tensor_spans[result_id]
        same_storage = source.storage_id == output.storage_id
        same_dtype = source.metadata.dtype == output.metadata.dtype
        # Empty views access no byte; their implicit offset is inherited too.
        empty = any(size == 0 for size in output.metadata.shape)
        relative = same_storage and same_dtype and (
            empty or (start == low and low <= end <= high))
      if not relative:
        raise ValueError('explicit storage-offset operations on planned arena tensors '
                         f'are unsupported (node {ordinal}: {schema})')
    result_ids = list(refs(instruction.result))
    if schema.name in {'aten::empty', 'aten::empty_like', 'aten::empty_strided'} and result_ids and all(identity in planned or identity in replacements for identity in result_ids):
      continue
    arguments = []
    for argument, item in zip(schema.arguments, instruction.arguments):
      expected = argument.type
      if isinstance(expected, torch._C.OptionalType):
        expected = expected.getElementType()
      arguments.append(value(item, expected))
    node = graph.create(schema.name, arguments, len(schema.returns));graph.appendNode(node)
    node.s_('vecops_matmul_precision', instruction.matmul_precision)
    for output, ret in zip(node.outputs(), schema.returns):
      output.setType(ret.type)
    # JIT nodes resolve overloads from their exact input types. Fail rather than
    # accept an ambiguous or different schema from the recorded dispatcher op.
    resolved = node.schema()
    if resolved != str(schema):
      raise ValueError(f'cannot bind captured overload exactly: {schema}; resolved {resolved}')
    from vecops import _schema_bridge
    def describe_tensor(ref):
      if not isinstance(ref, _Ref):
        raise ValueError('recordable source kernel requires an explicit tensor argument')
      m = capture._torch_execution_graph.tensors[ref.id].metadata
      # SDK tensor patterns have positive rank. A captured scalar Tensor is
      # proven to span one element, so its native broadcast operand is a [1]
      # logical view of the same pointer. No runtime reshape/metadata dispatch
      # is necessary: PlanBinding patches data_ptr only after graph guards.
      if pointwise_output is not None and not m.shape:
        return replace(m, shape=(1,), strides=(1,))
      return m
    prepared = (prepared_cache[ordinal] if ordinal in prepared_cache else
                _schema_bridge.prepare_replay_plan(instruction.function, instruction.arguments, describe_tensor))
    if prepared is not None:
      lease, flags, dtypes, outputs = prepared
      owners.append(lease)
      node.i_('vecops_plan_address', lease.abi_address)
      # IRParser infers empty-list attributes through node.output(), which is
      # invalid for out-style operators with zero returns. Use explicit strings.
      node.s_('vecops_plan_flags', ','.join(map(str, flags)))
      node.s_('vecops_plan_dtypes', ','.join(map(str, dtypes)))
      node.s_('vecops_plan_outputs', ','.join(map(str, outputs)))
      if (paths := getattr(lease, 'argument_paths', None)) is not None:
        node.s_('vecops_plan_argument_paths', ','.join(paths))
      prepared_nodes += 1
    if len(schema.returns) == 1:
      assign(instruction.result, node.output())
    elif len(schema.returns) > 1:
      for item, output in zip(instruction.result, node.outputs()):
        assign(item, output)
  outputs = [value(x) for x in tape.output_leaves if isinstance(x, _Ref)]
  node = graph.create('prim::TupleConstruct', outputs, 1);graph.appendNode(node)
  node.output().setType(torch._C.TupleType([x.type() for x in outputs]));graph.registerOutput(node.output())
  torch._C._jit_pass_lint(graph)
  # Graph's textual printer writes an empty assignment for a zero-return op;
  # IRParser expects the same instruction without the empty left-hand side.
  ir = re.sub(r'^( +) = ', r'\1', str(graph), flags=re.MULTILINE)
  if any(name in ir for name in ('prim::PythonOp', 'prim::CallFunction', 'prim::CallMethod')):
    raise ValueError('lowered graph contains an indirect or Python call')
  executor = executor_class(ir, examples)
  result = TorchReplay(executor, tape, bound, owners, ir, capture._torch_execution_graph,
                       len(planned), memory_plan, unplanned)
  result._retained_output_owner = output_owner
  result._ownership_proof = native_donation
  result._protected_storages = {execution.tensors[x].storage_id for x in opaque_tensors}
  result.allocator_policy.update(
      recorded_cpp_plan_nodes=prepared_nodes,
      native_pointwise_nodes=len(pointwise),
      hoisted_static_views=len(alias_nodes),
      eliminated_static_noops=static_noops,
      native_island_sizes=list(executor.native_island_sizes()),
      pointwise_scratch_bytes=sum(execution.tensors[ref.id].metadata.logical_bytes
          for ordinal,(_,_,_,ref,_) in pointwise.items() if ordinal not in pointwise_inplace),
      native_pointwise_inplace_nodes=len(pointwise_inplace),
      native_pointwise_donations=len(donations),
      native_kernel_donations=len(native_rewrites),
      static_factory_count=len(static_factories),
      static_scratch_arena_bytes=scratch_owner.numel() if scratch_owner is not None else 0,
      static_scratch_required_bytes=scratch_required_bytes,
      custom_scratch_allocator=scratch_allocator is not None,
      buffer_placement={kind:owner.placement_stats() for kind,owner in
          (('transient', scratch_owner), ('retained', output_owner)) if hasattr(owner, 'placement_stats')},
      borrowed_outputs=borrow_outputs,
      retained_output_tensor_count=retained_output_count,
      retained_output_arena_bytes=output_owner.numel() if output_owner is not None else 0,
      captured_arena_copy_bytes=sum(row['private_bytes'] for row in arena_rows),
      captured_arena_borrowed_state_bytes=sum(row.get('borrowed_state_bytes', 0) for row in arena_rows),
      immutable_planner_state_borrowing=borrow_readonly_state,
      captured_arena_regions=arena_rows,
      captured_arena_weights=(
          'pinned dedicated immutable planner state (must not be externally mutated); other arena bindings are compile-time private snapshots'
          if any(row.get('borrowed_state_bytes', 0) for row in arena_rows) else
          'compile-time private snapshot; recapture to update' if arena_rows else 'none'))
  if static_factories:
    result.allocator_policy['factory_storage'] = (
        'private static scratch and separate borrowed output arena' if borrow_outputs else
        'private static scratch; escaping factories use ATen')
  tape.storage_bindings.update(isolated_bindings)
  tape.tensors.update(isolated_bindings)
  tape.arena_leases = arena_leases
  tape.session_owner = None
  tape.tensors = {identity:tensor for identity,tensor in tape.tensors.items()
                  if identity in bound_ids or identity in input_ids}
  tape.storage_bindings = {identity:tensor for identity,tensor in tape.storage_bindings.items()
                          if identity in bound_ids or identity in input_ids}
  return result


def _plan_views(capture, plan, system, domain, large_pages, tape):
  if plan is None:
    return {}, [], []
  from vecops.planner import PlanSession
  from vecops.planner.validate import validate_plan
  from vecops.planner.ir import MemoryTier
  execution = capture._torch_execution_graph
  validate_plan(execution, plan)
  factory_ids = {ref.id for i in tape.instructions
                 if i.function._schema.name in {'aten::empty', 'aten::empty_like', 'aten::empty_strided'}
                 for ref in ([i.result] if isinstance(i.result, _Ref) else [])}
  selected = set()
  unplanned = []
  for storage_id, placement in plan.placements.items():
    if placement.offset is None:
      continue
    storage = execution.storages[storage_id]
    candidates = factory_ids.intersection(storage.tensors)
    if not candidates or storage.external or storage.persistent or any(
        execution.tensors[t].escapes for t in storage.tensors):
      if placement.tier == MemoryTier.fast:
        raise ValueError(f'replay plan only supports internal nonescaping fast factory storage: {storage_id}')
      # Ordinary result-producing ATen operations still use their own native
      # allocator. They are explicitly reported rather than claimed as planned.
      unplanned.append(storage_id)
      continue
    selected.update(candidates)
  # Allocate a private arena lease once. Closing the temporary PlanSession
  # restores its context; retained owners/views keep these allocations alive
  # independently of all future planner sessions.
  with PlanSession(execution, plan, system=system, domain=domain,
                   enable_factory_replay=True, large_pages=large_pages) as session:
    views = {identity:session.tensor_view(identity) for identity in selected}
    owners = [x for x in (session.fast_owner, session.slow_owner) if x is not None]
  return views, owners, unplanned


class TorchReplay:
  """CPU inference executable compiled by :class:`GraphCapture`.

  Weights/buffers are the Tensor objects observed during capture: changing
  their contents is visible; replacing a module attribute requires recapture.
  Outputs preserve aliases and storage lifetime. Outputs aliased to input/state
  or a private captured planner arena are borrowed and may change on later
  replay; clone them to retain values. By default functional outputs are
  allocated by ATen. Explicit borrow_outputs also reuses supported outputs.
  Newly planned arenas are private, and escaping outputs cannot use their slots.
  """
  def __init__(self, executor, tape, bound, owners, ir, execution, planned_count, plan, unplanned):
    from vecops.planner.buffers import current_session
    self._buffer_session = current_session()
    self._executor = executor
    self._input_spec = tape.input_spec
    self._input_refs = tape.input_refs
    self._tensor_only_inputs = all(isinstance(x, _Ref) for x in self._input_refs)
    self._output_leaves = tape.output_leaves
    self._output_spec = tape.output_spec
    self._bound = tuple(bound)
    # Cold, non-owning evidence for alias-closure checks when linking outputs
    # into another executable. Distinct StorageImpls may overlap physically.
    import torch
    def span(value):
      if not isinstance(value, torch.Tensor):
        return None
      storage = value.untyped_storage()
      return storage.data_ptr(), storage.nbytes()
    self._input_storage_spans = tuple(map(span, tape.input_leaves))
    self._bound_storage_spans = tuple(map(span, self._bound))
    self._owners = tuple(owners) + (tape.module_owner,)
    self.ir = ir
    self.graph = execution
    self.memory_plan = plan
    self.planned_factory_count = planned_count
    self.allocator_policy = {'factory_storage':'private plan arenas' if plan else 'captured/ATen',
                             'functional_aten':'native ATen allocator',
                             'unapplied_slow_plan_storages':unplanned}
    self._lock = threading.Lock()

  def fixed_output_values(self, values):
    """Cold descriptors for fixed, privately owned output storage only."""
    from ._output_transfer import fixed_values
    return fixed_values(self, values)

  def prepare_output_transfer(self, values, consumer, *, input_offset, expected_replays):
    """Prepare readonly-cache promotion and native copies into free fast space.

    The caller asserts invocation-scoped immutability across all other graphs.
    Consumer write/opaque effects and unused caches are excluded. Existing
    placement stays fixed; source allocations retain canonical contents.
    """
    from ._output_transfer import prepare
    return prepare(self, values, consumer, input_offset=input_offset, expected_replays=expected_replays)

  def run_tensors(self, tensors):
    """Replay already flattened, all-Tensor inputs in capture leaf order.

    This avoids Python pytree rebuilding and duplicate Tensor metadata queries.
    Exact Python Tensor types are still required; the ordinary native executor
    checks every input/state's metadata, keys, names and aliases before running
    anything. Context/ownership/reentrancy/output contracts match __call__.
    """
    import torch
    from torch.utils._pytree import tree_unflatten
    _check_context()
    if self._buffer_session is not None:
      from vecops.planner.buffers import current_session
      if (current_session() is not self._buffer_session or
          threading.get_ident() != self._buffer_session.thread_id):
        raise RuntimeError('graph replay requires its prepared BufferSession on its execution thread')
    if not self._lock.acquire(blocking=False):
      raise RuntimeError('graph replay is not reentrant or concurrently callable')
    try:
      if not self._tensor_only_inputs:
        raise ValueError('flat Tensor replay requires all captured input leaves to be tensors')
      tensors = tuple(tensors)
      if len(tensors) != len(self._input_refs):
        raise ValueError('graph replay flat input count changed')
      if any(type(x) not in (torch.Tensor, torch.nn.Parameter) for x in tensors):
        raise TypeError('graph replay rejects Tensor subclasses, FakeTensors and tensor modes')
      result = iter(self._executor.run([*tensors, *self._bound]))
      outputs = [next(result) if isinstance(x, _Ref) else x for x in self._output_leaves]
      return tree_unflatten(outputs, self._output_spec)
    finally:
      self._lock.release()

  def __call__(self, *args, **kwargs):
    import torch
    from torch.utils._pytree import tree_flatten, tree_unflatten
    _check_context()
    if self._buffer_session is not None:
      from vecops.planner.buffers import current_session
      if (current_session() is not self._buffer_session or
          threading.get_ident() != self._buffer_session.thread_id):
        raise RuntimeError('graph replay requires its prepared BufferSession on its execution thread')
    if not self._lock.acquire(blocking=False):
      raise RuntimeError('graph replay is not reentrant or concurrently callable')
    try:
      leaves, spec = tree_flatten((args, kwargs))
      if spec != self._input_spec:
        raise ValueError('graph replay input container structure changed')
      tensors = []
      for expected, actual in zip(self._input_refs, leaves):
        if isinstance(expected, _Ref):
          _check_tensor(actual);tensors.append(actual)
        elif type(actual) is not type(expected) or actual != expected:
          raise ValueError('graph replay static non-Tensor input changed')
      result = iter(self._executor.run(tensors + list(self._bound)))
      outputs = [next(result) if isinstance(x, _Ref) else x for x in self._output_leaves]
      return tree_unflatten(outputs, self._output_spec)
    finally:
      self._lock.release()


__all__ = ['TorchReplay']
