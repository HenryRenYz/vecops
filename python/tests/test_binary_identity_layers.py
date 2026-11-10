# SPDX-FileCopyrightText: 2026 HenryRenYz and vecops contributors
# SPDX-License-Identifier: MIT
"""Runtime-only edits refresh frontend caches without expiring native code."""
from pathlib import Path
from types import SimpleNamespace
import os

import pytest


@pytest.fixture
def identity_tree(tmp_path,monkeypatch):
  from vecops import _identity,_compiler
  package=tmp_path/'package';sdk=tmp_path/'sdk'
  for path in (package/'graph',package/'planner',sdk/'include',sdk/'src',sdk/'cmake'):
    path.mkdir(parents=True)
  (sdk/'CMakeLists.txt').write_text('project(vecops)\n')
  (sdk/'include/contract.h').write_text('// ABI one\n')
  (package/'_schema_bridge.py').write_text('# generator one\n')
  for name in ('graph/_replay.py','graph/_metadata.py','graph/_pointwise_recipe.py','planner/runtime.py'):
    (package/name).write_text('# runtime one\n')
  monkeypatch.setattr(_identity,'__file__',str(package/'_identity.py'))
  monkeypatch.setattr(_compiler,'_sdk',lambda:(sdk,SimpleNamespace(name='source_tree')))
  return _identity,package,sdk


@pytest.mark.parametrize('name',['graph/_replay.py','graph/_metadata.py','graph/_pointwise_recipe.py','planner/runtime.py'])
def test_runtime_edit_preserves_native_but_invalidates_frontend(identity_tree,name):
  identity,package,_=identity_tree
  native=identity._implementation_identity(binary=True)
  frontend=identity._implementation_identity()
  path=package/name;before=path.stat()
  path.write_text('# runtime two\n')
  os.utime(path,ns=(before.st_atime_ns,before.st_mtime_ns))
  assert identity._implementation_identity(binary=True)==native
  assert identity._implementation_identity()!=frontend


@pytest.mark.parametrize('kind',['generator','header','new_runtime_module','removed_runtime_module'])
def test_native_dependencies_and_inventory_still_refresh(identity_tree,kind):
  identity,package,sdk=identity_tree
  native=identity._implementation_identity(binary=True)
  if kind=='generator':(package/'_schema_bridge.py').write_text('# changed generator\n')
  elif kind=='header':(sdk/'include/contract.h').write_text('// ABI two\n')
  elif kind=='new_runtime_module':(package/'graph/unclassified.py').write_text('# new code\n')
  else:(package/'graph/_metadata.py').unlink()
  assert identity._implementation_identity(binary=True)!=native
