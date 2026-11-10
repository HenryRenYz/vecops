# SPDX-FileCopyrightText: 2026 HenryRenYz and vecops contributors
# SPDX-License-Identifier: MIT
"""Thin Python ownership adapters for the C++ memory resource runtime.

Resource discovery, capacity admission, physical allocation and mapping live in
C++. Graph policies consume ``native`` value objects and the runtime queries;
this module neither recognizes CPU hierarchy names nor ranks memory tiers.
"""
from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from hashlib import sha256

from .. import _C

native = _C.memory


def _contract_value(item):
  """Encode public native value fields without addresses, epochs or owners."""
  if item is None or type(item) in (str, bool, int, float):
    return item
  if isinstance(item, dict):
    return {key: _contract_value(value) for key, value in sorted(item.items())}
  if isinstance(item, (tuple, list)):
    return [_contract_value(value) for value in item]
  if isinstance(getattr(item, 'name', None), str):
    return [type(item).__name__, item.name]
  fields = {name: getattr(item, name) for name in dir(item) if not name.startswith('_')}
  return {name: _contract_value(value) for name, value in sorted(fields.items()) if not callable(value)}


def value(type_name: str, **fields):
  """Construct a native value; byte quantities and offsets remain integers."""
  result = getattr(native, type_name)()
  for name, item in fields.items():
    setattr(result, name, item)
  return result


class Runtime:
  """Own one native runtime; construction from a certified runtime is explicit.

  Multi-process launchers pass the same private ``capacity_directory`` in native
  ``LinuxRuntimeOptions`` to coordinate physical roots on this host. Policies and
  hardware templates contain no rank-local address or hostname.
  """

  def __init__(self, runtime):
    if not isinstance(runtime, native.MemoryRuntime):
      raise TypeError('Runtime requires a native MemoryRuntime')
    self.native = runtime
    self._configuration = {}

  @classmethod
  def discover(cls, hardware_file: str | Path | None = None, *, options=None,
               profile=None):
    result = cls(native.discover_linux_runtime(
      None if hardware_file is None else str(hardware_file),
      native.LinuxCpuProfile.generic_cpu if profile is None else profile,
      native.LinuxRuntimeOptions() if options is None else options))
    result._configuration = {'hardware': None if hardware_file is None else sha256(Path(hardware_file).read_bytes()).hexdigest()}
    return result

  @classmethod
  def open(cls, hardware_file: str | Path, *, policy_file=None, calibration_file=None, options=None):
    result = cls(native.open_runtime(str(hardware_file),
      None if policy_file is None else str(policy_file),
      None if calibration_file is None else str(calibration_file),
      native.LinuxRuntimeOptions() if options is None else options))
    result._configuration = {name: None if path is None else sha256(Path(path).read_bytes()).hexdigest()
      for name, path in [('hardware', hardware_file), ('policy', policy_file), ('calibration', calibration_file)]}
    return result

  def contract(self):
    """Stable cache identity for resources, process grants and configured limits.

    Current free bytes, ledger epochs, participant IDs and virtual addresses do
    not describe program semantics and are excluded. Admission still uses a
    fresh snapshot when a cached graph is instantiated.
    """
    view = self.view()
    return {'graph': _contract_value(view.graph),
      'executable_agents': list(view.executable_agents),
      'allocatable_pools': list(view.allocatable_pools),
      'accessible_pools': list(view.accessible_pools),
      'execution_slots': dict(view.execution_slots),
      'limits': sorted((group.capacity_group, group.hard_limit_bytes) for group in self.snapshot(view).groups),
      'configuration': dict(self._configuration)}

  def __getattr__(self, name):
    return getattr(self.native, name)

  def view(self, scope=None):
    return self.native.view(self.native.current_scope() if scope is None else scope)

  def snapshot(self, view=None, *, caches=()):
    return self.native.snapshot(self.view() if view is None else view, list(caches))

  def create_arena(self, domain: str, *, view=None, allowed_pools: Sequence[str] = (),
                   hard_limit: int | float = 1.0, initial: int | float = 0,
                   grow_by_bytes: int = 16 << 20, retain_idle_bytes: int = 64 << 20,
                   page_bytes: int = 0, remappable: bool = True, allow_growth: bool = True):
    options = value('ArenaOptions', domain=domain, allowed_pools=list(allowed_pools),
      hard_limit=value('Limit', bytes_or_capacity_fraction=hard_limit),
      initial=value('Preallocate', bytes_or_limit_fraction=initial),
      grow_by_bytes=grow_by_bytes, retain_idle_bytes=retain_idle_bytes, page_bytes=page_bytes,
      backing=native.BackingMode.shareable_remappable if remappable else native.BackingMode.private_memory,
      allow_growth=allow_growth)
    return self.native.create_arena(self.view() if view is None else view, options)

  def import_tensor(self, tensor, *, symbol: str, pool: str | None = None, read_only=False, view=None):
    """Import a CPU storage owner without inferring placement from its address.

    The descriptor uses the real accessible byte span, including positive stride
    gaps. Backend physical-identity checks still decide whether operands alias.
    """
    if tensor.device.type != 'cpu' or tensor.layout.__str__() != 'torch.strided':
      raise ValueError('host runtime requires a strided CPU tensor')
    if any(stride < 0 for stride in tensor.stride()):
      raise ValueError('negative tensor strides are unsupported')
    if tensor.numel() == 0:
      raise ValueError('empty tensors need no imported memory range')
    elements = 1 + sum((size-1)*stride for size, stride in zip(tensor.shape, tensor.stride()))
    address = value('VirtualAddress', address_space='host', value=tensor.data_ptr())
    external = native.ExternalMemoryView(symbol, address, elements*tensor.element_size(),
      tensor.untyped_storage(), pool, read_only)
    return self.native.import_memory(self.view() if view is None else view, external)


def tensor_view(buffer, *, shape, dtype, byte_offset: int = 0, strides=None):
  """Create a Torch view whose Storage retains the native buffer owner.

  PreparedAddresses buffers may represent reserved VA before materialization.
  The program binds physical storage before the returned tensor is accessed and
  retains the corresponding MemoryBinding through all uses/escaped outputs.
  """
  import math
  import torch
  if byte_offset < 0:
    raise ValueError('negative buffer offset')
  shape = tuple(int(size) for size in shape)
  if any(size < 0 for size in shape):
    raise ValueError('negative tensor size')
  element_bytes = torch.empty(0, dtype=dtype, device='cpu').element_size()
  if byte_offset % element_bytes:
    raise ValueError('buffer offset is not element aligned')
  if strides is None:
    count = math.prod(shape)
  else:
    strides = tuple(int(stride) for stride in strides)
    if len(strides) != len(shape) or any(stride < 0 for stride in strides):
      raise ValueError('strides must be nonnegative and match tensor rank')
    count = 0 if 0 in shape else 1 + sum((size-1)*stride for size, stride in zip(shape,strides))
  if count == 0:
    return torch.empty_strided(shape, strides, dtype=dtype, device='cpu') if strides is not None else torch.empty(shape,dtype=dtype, device='cpu')
  result = torch.frombuffer(buffer, dtype=dtype, count=count, offset=byte_offset)
  return result.reshape(shape) if strides is None else result.as_strided(shape,strides)


__all__ = ['Runtime', 'native', 'tensor_view', 'value']

# The active application owns the runtime lifetime; graph preparation and eager
# allocation resolve the same context rather than creating independent ledgers.
import contextvars
import os
import threading
from contextlib import contextmanager

_active_runtime = contextvars.ContextVar('vecops_memory_runtime', default=None)
_default_lock = threading.Lock()
_default_by_pid = {}


def current_runtime():
  return _active_runtime.get()


def default_runtime():
  pid = os.getpid()
  with _default_lock:
    if pid not in _default_by_pid:
      runtime = Runtime(native.default_workspace_runtime())
      runtime._configuration = {name: None if not os.environ.get(key) else sha256(Path(os.environ[key]).read_bytes()).hexdigest()
        for name, key in [('hardware', 'VECOPS_MEMORY_HARDWARE'), ('policy', 'VECOPS_MEMORY_POLICY'),
                          ('calibration', 'VECOPS_MEMORY_CALIBRATION')]}
      _default_by_pid[pid] = runtime
    return _default_by_pid[pid]


@contextmanager
def use_runtime(runtime):
  """Select one authority for graph, pools and ordinary native eager calls.

  This synchronous scope is thread-affine and must close in LIFO order; do not
  suspend an asyncio task while the scope is active. Cached fallback workspaces
  retire at each boundary, while explicit program bindings retain their owners.
  """
  if isinstance(runtime, native.MemoryRuntime):
    runtime = Runtime(runtime)
  if not isinstance(runtime, Runtime):
    raise TypeError('use_runtime requires a Runtime or native MemoryRuntime')
  scope = native._WorkspaceRuntimeScope(runtime.native)
  token = _active_runtime.set(runtime)
  try:
    yield runtime
  finally:
    try:
      scope.close()
    finally:
      _active_runtime.reset(token)


__all__ += ['current_runtime', 'default_runtime', 'use_runtime']


class Pool:
  """Eager tensor factory backed by one native arena.

  Dropping the final Tensor storage owner returns its page-aligned allocation to
  the C++ arena. Exported native aliases and in-flight completion tokens retain
  the physical block independently. ``trim`` returns whole idle growth chunks.
  """
  def __init__(self, arena):
    if not isinstance(arena, native.MemoryArena):
      raise TypeError('Pool requires a native MemoryArena')
    self.native = arena

  @classmethod
  def create(cls, runtime, domain, **options):
    runtime = runtime if isinstance(runtime, Runtime) else Runtime(runtime)
    return cls(runtime.create_arena(domain, **options))

  @property
  def page_bytes(self):
    return self.native.page_bytes

  def snapshot(self):
    return self.native.snapshot()

  def allocate(self, bytes, *, preferred_pool=None, allow_domain_fallback=True):
    return self.native.allocate(value('AllocateRequest', bytes=bytes, preferred_pool=preferred_pool,
      allow_domain_fallback=allow_domain_fallback))

  def empty(self, shape, *, dtype=None, preferred_pool=None, allow_domain_fallback=True,
            strides=None, requires_grad=False):
    import math
    import torch
    dtype = torch.get_default_dtype() if dtype is None else dtype
    shape = (shape,) if isinstance(shape, int) else tuple(shape)
    if any(isinstance(size, bool) or not isinstance(size, int) or size < 0 for size in shape):
      raise ValueError('tensor dimensions must be nonnegative integers')
    if strides is not None:
      strides = tuple(strides)
      if len(strides) != len(shape) or any(isinstance(stride, bool) or not isinstance(stride,int) or stride < 0 for stride in strides):
        raise ValueError('strides must be nonnegative integers matching tensor rank')
      count = 0 if 0 in shape else 1 + sum((size-1)*stride for size,stride in zip(shape,strides))
    else:
      count = math.prod(shape)
    element_bytes = torch.empty(0,dtype=dtype, device='cpu').element_size()
    if count == 0:
      result = torch.empty_strided(shape,strides,dtype=dtype, device='cpu') if strides is not None else torch.empty(shape,dtype=dtype, device='cpu')
    else:
      if count * element_bytes >= 1 << 64:
        raise OverflowError('tensor size exceeds uint64 byte range')
      allocation = self.allocate(count*element_bytes,preferred_pool=preferred_pool,
        allow_domain_fallback=allow_domain_fallback)
      result = tensor_view(allocation,shape=shape,dtype=dtype,strides=strides)
    return result.requires_grad_(requires_grad)

  def trim(self, retain_idle_bytes=0):
    return self.native.trim(retain_idle_bytes)


__all__ += ['Pool']
