# SPDX-FileCopyrightText: 2026 HenryRenYz and vecops contributors
# SPDX-License-Identifier: MIT
"""Archive ownership across actual wheel/editable/CMake path layouts; no Torch."""
import builtins
import importlib.util
from pathlib import Path

import pytest


def load_helper():
  source = Path(__file__).resolve().parents[1]/'vecops/distributed/_build.py'
  spec = importlib.util.spec_from_file_location('_communication_archive_lookup_test', source)
  module = importlib.util.module_from_spec(spec)
  spec.loader.exec_module(module)
  return module._resolve_memory_library


def file(path, data=b'!<arch>\n'):
  path.parent.mkdir(parents=True, exist_ok=True)
  path.write_bytes(data)
  return path


def test_archive_helper_imports_without_torch_or_vecops(monkeypatch):
  original = builtins.__import__
  def guarded(name, *args, **kwargs):
    if name.split('.',1)[0] in {'torch','vecops'}:
      raise AssertionError('pure archive lookup imported '+name)
    return original(name,*args,**kwargs)
  monkeypatch.setattr(builtins,'__import__',guarded)
  assert callable(load_helper())


def test_installed_wheel_uses_its_sdk_archive(tmp_path):
  package = tmp_path/'site-packages/vecops'
  native = file(package/'_C.so')
  archive = file(package/'_sdk/lib/libvecops_memory.a')
  assert load_helper()(package/'_sdk',native) == archive.resolve()


def test_editable_checkout_defaults_to_active_extension_sdk(tmp_path):
  checkout = tmp_path/'source'
  file(checkout/'include/vecops/comm/Communication.h')
  package = tmp_path/'site-packages/vecops'
  native = file(package/'_C.so')
  archive = file(package/'_sdk/lib/libvecops_memory.a')
  assert load_helper()(checkout,native) == archive.resolve()


def test_selected_sdk_archive_precedes_active_extension_sdk(tmp_path):
  sdk = tmp_path/'selected-sdk'
  expected = file(sdk/'lib/libvecops_memory.a',b'selected SDK')
  native = file(tmp_path/'site/vecops/_C.so')
  file(native.parent/'_sdk/lib/libvecops_memory.a',b'installed SDK')
  assert load_helper()(sdk,native) == expected.resolve()


def test_source_cmake_build_falls_back_to_its_own_archive(tmp_path):
  native = file(tmp_path/'build/python/bindings/_C.so')
  archive = file(tmp_path/'build/src/libvecops_memory.a')
  assert load_helper()(tmp_path/'checkout',native) == archive.resolve()


def test_extension_sdk_precedes_cmake_build_fallback(tmp_path):
  native = file(tmp_path/'build/python/bindings/_C.so')
  expected = file(native.parent/'_sdk/lib/libvecops_memory.a')
  file(tmp_path/'build/src/libvecops_memory.a',b'another archive')
  assert load_helper()(tmp_path/'checkout',native) == expected.resolve()


def test_extension_symlink_keeps_its_installed_sdk_adjacency(tmp_path):
  built = file(tmp_path/'build/python/bindings/_C.so')
  file(tmp_path/'build/src/libvecops_memory.a',b'build archive')
  native = tmp_path/'site-packages/vecops/_C.so'
  native.parent.mkdir(parents=True)
  native.symlink_to(built)
  expected = file(native.parent/'_sdk/lib/libvecops_memory.a',b'installed archive')
  assert load_helper()(tmp_path/'checkout',native) == expected.resolve()


def test_arbitrary_ancestors_are_not_treated_as_source_builds(tmp_path):
  native = file(tmp_path/'site-packages/vecops/_C.so')
  # The old unconditional parents[2]/src probe could select this unrelated file.
  file(tmp_path/'src/libvecops_memory.a',b'unrelated ancestor archive')
  with pytest.raises(RuntimeError,match='requires libvecops_memory.a'):
    load_helper()(tmp_path/'checkout',native)


def test_explicit_archive_overrides_both_default_locations(tmp_path):
  sdk = tmp_path/'sdk';file(sdk/'lib/libvecops_memory.a')
  native = file(tmp_path/'site/vecops/_C.so');file(native.parent/'_sdk/lib/libvecops_memory.a')
  expected = file(tmp_path/'explicit/libvecops_memory.a')
  assert load_helper()(sdk,native,str(expected)) == expected.resolve()


@pytest.mark.parametrize('kind',['missing','directory','broken-symlink','empty'])
def test_invalid_explicit_archive_never_falls_back(tmp_path,kind):
  sdk = tmp_path/'sdk';file(sdk/'lib/libvecops_memory.a')
  native = file(tmp_path/'site/vecops/_C.so');file(native.parent/'_sdk/lib/libvecops_memory.a')
  override = tmp_path/'override'
  if kind == 'directory':override.mkdir()
  elif kind == 'broken-symlink':override.symlink_to(tmp_path/'absent')
  value = '' if kind == 'empty' else str(override)
  with pytest.raises(RuntimeError,match='VECOPS_MEMORY_LIBRARY'):
    load_helper()(sdk,native,value)


def test_archive_symlink_resolves_to_exact_selected_file(tmp_path):
  expected = file(tmp_path/'archive/runtime.a')
  sdk = tmp_path/'sdk';(sdk/'lib').mkdir(parents=True)
  (sdk/'lib/libvecops_memory.a').symlink_to(expected)
  native = file(tmp_path/'site/vecops/_C.so')
  assert load_helper()(sdk,native) == expected.resolve()
