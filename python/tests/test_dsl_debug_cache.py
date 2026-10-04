"""Source/header/config/backend invalidation, cache-only and Python wrappers."""

import numpy as np
import pytest
import vecops
import vecops.language as T


def test_watched_hash_detects_same_size_same_mtime_edits(tmp_path):
  import os

  from vecops._identity import file_identity, tree_identity

  path = tmp_path / "header.h"
  path.write_text("first")
  timestamp = path.stat()
  first = file_identity(path)
  first_tree = tree_identity(tmp_path)
  path.write_text("other")
  os.utime(path, ns=(timestamp.st_atime_ns, timestamp.st_mtime_ns))
  assert first != file_identity(path)
  assert first_tree != tree_identity(tmp_path)
  (tmp_path / "shadow.h").write_text("new header")
  assert len(tree_identity(tmp_path)) == 2


def test_source_headers_jit_wrapper_and_build_flags(tmp_path):
  header = tmp_path / "value.h"
  header.write_text("constexpr float value = 1;\n")
  source = tmp_path / "kernel.cpp"
  source.write_text("""#include "vecops/Kernel.h"
#include "value.h"
void __kernel__(vecops::tensor::TensorLike auto A, vecops::tensor::TensorLike auto C) { C.data()[0] = A.data()[0] + value; }
""")
  compiler = vecops.Compiler(
    cc="gcc",
    cxx="g++",
    target="Scalar",
    output="quiet",
    jobs=2,
    cache_dir=tmp_path / "cache",
    build_dir=tmp_path / "build",
  )

  @T.jit(out_idx=-1, compiler=compiler)
  def factory():
    @T.template(source=source)
    def kernel(A: T.Tensor((1,), dtype=T.float32), C: T.Tensor((1,), dtype=T.float32)):
      pass

    return kernel

  first = factory()
  assert factory() is first
  compiler.cflags = ("-ffp-contract=off",)
  changed_policy = factory()
  assert changed_policy is not first
  compiler.cflags = ()
  first = factory()
  a = np.array([4], dtype=np.float32)
  np.testing.assert_array_equal(first(a), [5])
  header.write_text("constexpr float value = 3;\n")
  second = factory()
  assert second is not first
  assert second.get_cache_info()["key"] != first.get_cache_info()["key"]
  np.testing.assert_array_equal(second(a), [7])
  # Prepared objects are immutable snapshots: no hot file reads/rebuilds.
  np.testing.assert_array_equal(first(a), [5])

  @T.jit
  def forwarding():
    selected = factory()

    def wrap(A, C=None):
      return selected(A) if C is None else selected(A, C)

    return wrap

  wrapper = forwarding()
  assert wrapper.debug.children
  header.write_text("constexpr float value = 5;\n")
  assert forwarding() is not wrapper
  different = vecops.Compiler(
    cc="gcc",
    cxx="g++",
    target="Scalar",
    output="quiet",
    cflags=("-ffp-contract=off",),
    cache_dir=tmp_path / "cache",
    build_dir=tmp_path / "other",
  )
  third = T.compile(second.template, out_idx=-1, compiler=different)
  assert third.get_cache_info()["build_identity"] != second.get_cache_info()["build_identity"]
  reader = vecops.Compiler(
    target="Scalar",
    cache_mode="cache-only",
    cc="missing",
    cxx="missing",
    cache_dir=tmp_path / "cache",
    build_dir=tmp_path / "must-not-create",
  )
  # Header changed since second was prepared, so the cache-only reader must
  # miss rather than load its old source-key artifact.
  cached = T.compile(second.template, out_idx=-1, compiler=reader)
  with pytest.raises(ImportError, match="cache-only|cached"):
    cached(a)
  assert not reader.build_dir.exists()


def test_callback_identity_is_stable_after_execution():
  from vecops._identity import callable_identity

  def callback():
    return "interned strings", 42

  before = callable_identity(callback)
  callback()
  assert callable_identity(callback) == before


def test_forked_reader_does_not_consume_parent_notifications(tmp_path):
  import multiprocessing

  from vecops._identity import file_identity

  if "fork" not in multiprocessing.get_all_start_methods():
    pytest.skip("POSIX fork only")
  source = tmp_path / "fork.h"
  source.write_text("before")
  before = file_identity(source)
  context = multiprocessing.get_context("fork")
  parent, child = context.Pipe()

  def reader():
    source.write_text("after!")
    child.send(file_identity(source))
    child.close()

  process = context.Process(target=reader)
  process.start()
  try:
    assert parent.poll(10)
    received = parent.recv()
    process.join(10)
    assert process.exitcode == 0
    assert received != before
    assert file_identity(source) == received
  finally:
    if process.is_alive():
      process.terminate()
      process.join()
    parent.close()
    child.close()


def test_cache_only_missing_attestation_never_creates_dirs(tmp_path):
  compiler = vecops.Compiler(cache_mode="cache-only", cache_dir=tmp_path / "missing", cc="missing", cxx="missing")
  from vecops._identity import compiler_identity

  with pytest.raises(ImportError, match="cache-only"):
    compiler_identity(compiler)
  assert not compiler.cache_dir.exists()
