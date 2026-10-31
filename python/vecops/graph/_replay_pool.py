# SPDX-FileCopyrightText: 2026 HenryRenYz and vecops contributors
# SPDX-License-Identifier: MIT
"""Capture native-pool activation through the provider's actual C ABI DSO.

Resolution and ctypes are capture-only. The tape contains native helper ops;
replay calls these function addresses in C++, with exception-safe region exits.
"""
from __future__ import annotations

import ctypes
from contextlib import contextmanager


class _PoolPrefix(ctypes.Structure):
  _fields_ = [('struct_size', ctypes.c_uint32), ('abi_major', ctypes.c_uint16),
              ('abi_minor', ctypes.c_uint16), ('identity', ctypes.c_uint64),
              ('context', ctypes.c_void_p), ('max_parallelism', ctypes.c_void_p),
              ('in_parallel_region', ctypes.c_void_p), ('parallel_for', ctypes.c_void_p)]


class _DlInfo(ctypes.Structure):
  _fields_ = [('filename', ctypes.c_char_p), ('base', ctypes.c_void_p),
              ('symbol', ctypes.c_char_p), ('symbol_address', ctypes.c_void_p)]


def _resolve_provider(threads):
  from vecops import _C, _schema_bridge
  address = int(_C.native_thread_pool_address(threads))
  if not address:
    raise ValueError('native-pool replay requires a nonnull native provider')
  selected = getattr(_schema_bridge, '_native_thread_pool', None)
  selected = getattr(selected, 'value', selected)
  if selected and selected != address:
    raise ValueError('native-pool replay provider differs from the registered kernel provider')
  pool = _PoolPrefix.from_address(address)
  if pool.struct_size < ctypes.sizeof(_PoolPrefix) or pool.abi_major != 1 or not pool.parallel_for:
    raise ValueError('native-pool replay requires the V1 provider ABI')
  loader = ctypes.CDLL(None)
  dladdr = loader.dladdr
  dladdr.argtypes = [ctypes.c_void_p, ctypes.POINTER(_DlInfo)]
  dladdr.restype = ctypes.c_int
  info = _DlInfo()
  if not dladdr(pool.parallel_for, ctypes.byref(info)) or not info.filename:
    raise ValueError('cannot prove the native-pool provider DSO lifetime')
  library = ctypes.CDLL(info.filename.decode())
  try:
    factory = library.vecops_native_thread_pool_v1
    begin = library.vecops_native_thread_pool_begin_active
    end = library.vecops_native_thread_pool_end_active
  except AttributeError as error:
    raise ValueError('native-pool provider does not export replayable activation C ABI') from error
  factory.argtypes = [ctypes.c_uint32]; factory.restype = ctypes.c_void_p
  if factory(threads) != address:
    raise ValueError('native-pool activation DSO does not own the actual selected provider')
  return library, int(ctypes.cast(begin, ctypes.c_void_p).value), int(ctypes.cast(end, ctypes.c_void_p).value)


@contextmanager
def capture_pool_region(capture, threads):
  import torch
  from . import _replay
  owner, begin, end = _resolve_provider(threads)
  _replay._load_executor()
  namespace = _replay._executor_namespace
  operators = getattr(torch.ops, namespace)
  tape = capture._torch_replay_tape
  for name in ('pool_begin', 'pool_end'):
    tape.native_operators[namespace + '::' + name] = owner
  operators.pool_begin(begin, end, threads)
  try:
    yield
  finally:
    operators.pool_end(end)
