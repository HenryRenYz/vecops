# SPDX-FileCopyrightText: 2026 HenryRenYz and vecops contributors
# SPDX-License-Identifier: MIT
"""Version-pinned optional Torch bridge; core itself has no Torch dependency."""
import functools
import hashlib
import importlib.util
import os
import platform
from pathlib import Path
import shlex
import shutil
import subprocess


@functools.cache
def load():
  import torch
  from torch.utils.cpp_extension import include_paths, library_paths
  from vecops._compiler import _sdk, default_cache_dir
  from vecops._cache_lock import file_lock
  sdk, _ = _sdk()
  checkout = Path(__file__).resolve().parents[3]
  if (checkout/'include/vecops/comm/Communication.h').is_file():
    sdk = checkout
  source = Path(__file__).with_name('_torch.cpp')
  core = sdk / 'src/comm/Communication.cpp'
  if not core.exists():
    core = Path(__file__).with_name('_core.cpp')
  header = sdk / 'include/vecops/comm/Communication.h'
  copy_source = sdk / 'src/comm/Copy.cpp'
  if not copy_source.exists():
    copy_source = Path(__file__).with_name('_copy.cpp')
  copy_dispatch = sdk / 'src/comm/CopyDispatch.cpp'
  if not copy_dispatch.exists():
    copy_dispatch = Path(__file__).with_name('_copy_dispatch.cpp')
  arch = platform.machine().lower()
  variants = [('portable', ['-DCPU_CAPABILITY_GENERIC=1'])]
  if arch in {'x86_64','amd64'}:
    variants += [('avx2',['-mavx2']),
                 ('avx512',['-mavx512f','-mavx512bw','-mavx512vl','-mavx512dq','-mavx512cd'])]
  elif arch in {'aarch64','arm64'}:
    variants += [('sve',['-march=armv8-a+sve'])]
  headers = tuple((str(p.relative_to(sdk/'include')), p.read_bytes())
                  for p in sorted((sdk/'include').rglob('*')) if p.is_file())
  compiler = shlex.split(os.environ.get('VECOPS_COMM_CXX') or os.environ.get('CXX') or 'g++')
  compiler[0] = shutil.which(compiler[0]) or compiler[0]
  compile_flags = shlex.split(os.environ.get('VECOPS_COMM_CXXFLAGS', os.environ.get('CXXFLAGS', '')))
  link_flags = shlex.split(os.environ.get('VECOPS_COMM_LDFLAGS', os.environ.get('LDFLAGS', '')))
  copy_text, dispatch_text = copy_source.read_bytes(), copy_dispatch.read_bytes()
  identity = (source.read_bytes(), core.read_bytes(), header.read_bytes(),
              (sdk/'include/vecops/runtime/CallAbi.h').read_bytes(),
              torch.__version__, torch.compiled_with_cxx11_abi(), compiler,
              subprocess.check_output(compiler + ['--version']),
              copy_text, dispatch_text, variants, headers, compile_flags, link_flags)
  key = hashlib.sha256(repr(identity).encode()).hexdigest()[:24]
  directory = default_cache_dir() / 'communication' / key
  directory.mkdir(parents=True, exist_ok=True)
  library = directory / '_vecops_comm.so'
  with file_lock(directory / 'build.lock'):
    if not library.exists():
      temporary = directory / f'comm-{os.getpid()}.so'
      frozen = directory/'include/vecops'
      (frozen/'comm').mkdir(parents=True, exist_ok=True)
      (frozen/'runtime').mkdir(parents=True, exist_ok=True)
      (directory/'torch.cpp').write_bytes(identity[0])
      (directory/'core.cpp').write_bytes(identity[1])
      (directory/'copy.cpp').write_bytes(copy_text)
      (directory/'copy_dispatch.cpp').write_bytes(dispatch_text)
      for relative, contents in headers:
        target=directory/'include'/relative
        target.parent.mkdir(parents=True,exist_ok=True)
        target.write_bytes(contents)
      (frozen/'comm/Communication.h').write_bytes(identity[2])
      (frozen/'runtime/CallAbi.h').write_bytes(identity[3])
      import sysconfig
      command = compiler + compile_flags + ['-std=c++20', '-O3', '-shared', '-fPIC', '-pthread',
          '-DTORCH_EXTENSION_NAME=_vecops_comm',
          f'-D_GLIBCXX_USE_CXX11_ABI={int(torch.compiled_with_cxx11_abi())}',
          '-I'+str(directory/'include'), '-I'+str(directory/'include/vecops'), '-I'+sysconfig.get_path('include'),
          *['-I'+p for p in include_paths('cpu')], str(directory/'torch.cpp'), str(directory/'core.cpp'), str(directory/'copy_dispatch.cpp'), '-o', str(temporary),
          *['-L'+p for p in library_paths('cpu')],
          *['-Wl,-rpath,'+p for p in library_paths('cpu')],
          '-ltorch_python', '-ltorch_cpu', '-ltorch', '-lc10', '-lrt', *link_flags]
      try:
        for variant, flags in variants:
          copy_command=compiler+compile_flags+['-std=c++20','-O3','-fPIC','-Wno-ignored-attributes',
              '-fvisibility=hidden','-fvisibility-inlines-hidden',
              f'-D_GLIBCXX_USE_CXX11_ABI={int(torch.compiled_with_cxx11_abi())}',
              '-DVECOPS_COMM_COPY_VARIANT='+variant,*flags,
              '-I'+str(directory/'include'),'-I'+str(directory/'include/vecops'),
              '-c',str(directory/'copy.cpp'),'-o',str(directory/f'copy-{variant}.o')]
          (directory/f'copy-{variant}-build-command.txt').write_text(shlex.join(copy_command)+'\n')
          subprocess.run(copy_command,check=True,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True)
          command.append(str(directory/f'copy-{variant}.o'))
          if variant!='portable':
            command.append('-DVECOPS_COMM_COPY_'+variant.upper()+'=1')
        (directory/'build-command.txt').write_text(shlex.join(command)+'\n')
        subprocess.run(command, check=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        temporary.replace(library)
      except subprocess.CalledProcessError as error:
        raise RuntimeError('vecops communication build failed:\n'+error.stdout) from error
      finally:
        temporary.unlink(missing_ok=True)
      (directory/'build-command.txt').write_text(shlex.join(command)+'\n')
  spec = importlib.util.spec_from_file_location('_vecops_comm', library)
  module = importlib.util.module_from_spec(spec)
  spec.loader.exec_module(module)
  module.build_id = key
  from . import _resident
  _resident.install(module)
  return module
