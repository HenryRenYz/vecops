# SPDX-FileCopyrightText: 2026 HenryRenYz and vecops contributors
# SPDX-License-Identifier: MIT
"""Cold, opt-in shared Torch/OpenMP resident provider; no native workers.

Requires a compiler that can emit OpenMP for Torch's actual runtime. The
packaged-GOMP function-table bridge remains available without this extension.
The CDLL owner is retained for the interpreter lifetime, like Torch bridges.
"""
import ctypes
import hashlib
import os
from pathlib import Path
import shlex
import subprocess
from threading import Lock

_lock = Lock()
_owner = None
_address = None


def address():
  global _owner, _address
  with _lock:
    if _address is not None:
      return _address
    import torch
    from . import _schema_bridge as bridge
    from ._compiler import Compiler, _sdk, default_cache_dir
    from ._cache_lock import file_lock
    _, _, includes, libraries, parallel, _, torch_identity = bridge._torch_bridge_inputs()
    policy = bridge._torch_bridge_build_policy(Compiler()._native_config(), libraries, parallel)
    if not policy.compile_flags:
      raise RuntimeError('Torch resident teams require a direct OpenMP compiler adapter; '
                         'use GNU for packaged libgomp or Clang for libomp')
    source = Path(__file__).with_name('_TorchPool.cpp')
    sdk, _ = _sdk()
    compiler = str(policy.toolchain.cxx_compiler)
    environment = {**os.environ, **dict(policy.toolchain.environment)}
    cxxflags = shlex.split(environment.get('CXXFLAGS', ''))
    ldflags = shlex.split(environment.get('LDFLAGS', ''))
    version = subprocess.check_output([compiler, '--version'], env=environment)
    search_environment = tuple((key, environment.get(key, '')) for key in
        ('CPATH', 'CPLUS_INCLUDE_PATH', 'C_INCLUDE_PATH', 'LIBRARY_PATH'))
    runtimes = bridge._mapped_openmp_runtime_paths()
    if len(runtimes) != 1:
      raise RuntimeError('Torch resident teams require exactly one loaded OpenMP runtime')
    from ._identity import file_identity
    runtime_identity = tuple(file_identity(path) for path in sorted(runtimes))
    identity = (source.read_bytes(), (sdk/'include/vecops/runtime/CallAbi.h').read_bytes(),
                torch_identity, file_identity(compiler), version, policy.compile_flags,
                tuple(cxxflags), tuple(ldflags), search_environment, runtime_identity)
    key = hashlib.sha256(repr(identity).encode()).hexdigest()[:24]
    root = default_cache_dir()/'torch-resident-pool'/key
    root.mkdir(parents=True, exist_ok=True)
    library = root/'provider.so'
    with file_lock(root/'build.lock'):
      if not library.exists():
        temporary = root/f'provider-{os.getpid()}.so'
        command = [compiler, '-std=c++20', '-O2', '-shared', '-fPIC', '-pthread',
                   *cxxflags, *policy.compile_flags, f'-D_GLIBCXX_USE_CXX11_ABI={int(torch.compiled_with_cxx11_abi())}',
                   '-I'+str(sdk/'include'), *['-I'+str(p) for p in includes],
                   str(source), '-o', str(temporary), *['-L'+str(p) for p in libraries],
                   *['-Wl,-rpath,'+str(p) for p in libraries], *ldflags, '-ltorch_cpu', '-lc10']
        (root/'build-command.txt').write_text(shlex.join(command)+'\n')
        try:
          result = subprocess.run(command, capture_output=True, text=True, env=environment)
          if result.returncode:
            raise RuntimeError('Torch resident provider build failed:\n'+result.stdout+result.stderr)
          temporary.replace(library)
        finally:
          temporary.unlink(missing_ok=True)
    _owner = ctypes.CDLL(str(library))
    if bridge._mapped_openmp_runtime_paths() != runtimes:
      raise RuntimeError('Torch resident provider loaded a different OpenMP runtime; use a matching compiler')
    query = _owner.vecops_torch_resident_pool_v1
    query.restype = ctypes.c_void_p
    _address = ctypes.c_void_p(query())
    return _address
