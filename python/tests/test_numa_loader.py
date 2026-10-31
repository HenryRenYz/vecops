# SPDX-FileCopyrightText: 2026 HenryRenYz and vecops contributors
# SPDX-License-Identifier: MIT
"""Runtime provenance must distinguish a byte-identical audit copy from a foreign DSO."""
import hashlib

from vecops._numa_loader import is_loader_copy


def test_loader_copy_requires_source_association_content_and_private_inode(tmp_path, monkeypatch):
  original = tmp_path / 'libgomp.so.1'
  original.write_bytes(b'runtime bytes')
  cache = tmp_path / 'cache'
  digest = hashlib.sha256(original.read_bytes()).hexdigest()
  copied = cache / 'v1' / 'host' / '0,1' / ('identity-' + digest) / 'root' / str(original)[1:]
  copied.parent.mkdir(parents=True)
  copied.write_bytes(original.read_bytes())
  copied.chmod(0o500)
  monkeypatch.setenv('VECOPS_NUMA_PLACEMENT', '0,1')
  monkeypatch.setenv('VECOPS_NUMA_CACHE_DIR', str(cache))
  assert is_loader_copy(original, copied)
  copied.chmod(0o700)
  copied.write_bytes(b'foreign bytes')
  copied.chmod(0o500)
  assert not is_loader_copy(original, copied)
  copied.unlink()
  copied.hardlink_to(original)
  assert not is_loader_copy(original, copied)
  assert not is_loader_copy(tmp_path / 'different.so', copied)
  monkeypatch.delenv('VECOPS_NUMA_PLACEMENT')
  assert not is_loader_copy(original, copied)
