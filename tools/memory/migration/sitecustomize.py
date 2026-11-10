# SPDX-FileCopyrightText: 2026 HenryRenYz and vecops contributors
# SPDX-License-Identifier: MIT
"""Process-local source/native selection for isolated target qualification.

Put this directory on PYTHONPATH with both paths below set. The installed
environment and its editable package routing remain unchanged. Child workers
inherit exactly the same source and extension selection.
"""
import importlib.abc
import importlib.machinery
import importlib.util
import os
from pathlib import Path
import sys

if _trace_seconds := os.environ.get('VECOPS_PREPARE_TRACEBACK_SECONDS'):
  import faulthandler
  faulthandler.enable()
  faulthandler.dump_traceback_later(int(_trace_seconds), repeat=True)

_source = os.environ.get('VECOPS_SOURCE_ROOT')
_extension = os.environ.get('VECOPS_NATIVE_EXTENSION')
_af3_source = os.environ.get('AF3_SOURCE_ROOT')
if bool(_source) != bool(_extension):
  raise RuntimeError('isolated vecops loading requires source and native extension together')
if _source:
  _package = Path(_source).resolve(strict=True) / 'python/vecops'
  _binary = Path(_extension).resolve(strict=True)
  _af3_package = None if not _af3_source else Path(_af3_source).resolve(strict=True) / 'src/alphafold3'
  if not (_package / '__init__.py').is_file() or not _binary.is_file():
    raise RuntimeError('isolated vecops source or extension is missing')

  class _QualifiedSource(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
      if fullname == 'vecops':
        return importlib.util.spec_from_file_location(fullname, _package / '__init__.py',
            submodule_search_locations=[str(_package)])
      if fullname == 'vecops._C':
        return importlib.util.spec_from_file_location(fullname, _binary)
      if fullname.startswith('vecops.'):
        result = importlib.machinery.PathFinder.find_spec(fullname, path)
        if result is None:
          raise ModuleNotFoundError(f'{fullname} is absent from the selected vecops source')
        return result
      if _af3_package is not None:
        if fullname == 'alphafold3':
          return importlib.util.spec_from_file_location(fullname, _af3_package / '__init__.py',
              submodule_search_locations=[str(_af3_package)])
        if fullname.startswith('alphafold3.'):
          result = importlib.machinery.PathFinder.find_spec(fullname, path)
          if result is None:
            raise ModuleNotFoundError(f'{fullname} is absent from the selected AF3 source/dependencies')
          return result
      return None

  sys.meta_path.insert(0, _QualifiedSource())
