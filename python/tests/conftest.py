# SPDX-FileCopyrightText: 2026 HenryRenYz and vecops contributors
# SPDX-License-Identifier: MIT
"""Private, explicit cross-process capacity namespaces for transport tests."""
import pytest


@pytest.fixture(autouse=True)
def distributed_capacity_namespace(request, monkeypatch):
  if request.path.name.startswith('test_distributed'):
    tmp_path=request.getfixturevalue('tmp_path')
    directory=tmp_path/'capacity-authority'
    directory.mkdir(mode=0o700)
    monkeypatch.setenv('VECOPS_MEMORY_CAPACITY_DIRECTORY',str(directory))
