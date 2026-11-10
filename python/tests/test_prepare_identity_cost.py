# SPDX-FileCopyrightText: 2026 HenryRenYz and vecops contributors
# SPDX-License-Identifier: MIT
"""Compact identities retain byte invalidation; cost hints normalize Ninja paths."""
import json
import os

import pytest
from vecops._build_session import BuildSession
from vecops._identity import tree_identity
from vecops._schema_bridge import _torch_content_identity
from vecops._build_cost import observe_batch


def test_compact_content_identity_rejects_same_size_same_mtime_edit(tmp_path):
  source=tmp_path/'kernel.h';source.write_bytes(b'AAAA');stat=source.stat()
  with pytest.raises(RuntimeError,match='build inputs changed'):
    with BuildSession(parallelism=1,reuse_sdk=False,concurrent=False):
      first=_torch_content_identity((tree_identity(tmp_path),))
      assert len(repr(first))<100
      assert first==_torch_content_identity((tree_identity(tmp_path),))
      source.write_bytes(b'BBBB');os.utime(source,ns=(stat.st_atime_ns,stat.st_mtime_ns))
  with BuildSession(parallelism=1,reuse_sdk=False,concurrent=False):
    assert _torch_content_identity((tree_identity(tmp_path),))!=first
  # A new header also changes the directory inventory.
  (tmp_path/'new.h').write_bytes(b'new')
  assert _torch_content_identity((tree_identity(tmp_path),))!=first


@pytest.mark.parametrize('absolute_command_output',[False,True])
@pytest.mark.parametrize('absolute_ninja_output',[False,True])
def test_cost_history_matches_relative_and_absolute_objects(tmp_path,absolute_command_output,absolute_ninja_output):
  build=tmp_path/'build';build.mkdir();source=tmp_path/'adapter.cpp'
  kernel=tmp_path/'kernel.cpp';kernel.write_text('kernel')
  source.write_text('#include "kernel.cpp"\n')
  obj='obj/adapter.o';absolute=str(build/obj)
  (build/'compile_commands.json').write_text(json.dumps([dict(directory=str(build),file=str(source),
      output=absolute if absolute_command_output else obj)]))
  (build/'.ninja_log').write_text('# ninja log v5\n0\t12345\t0\t'+
      (absolute if absolute_ninja_output else obj)+'\thash\n')
  assert observe_batch(build/'vecops-build.log')=={str(kernel):12.345}
