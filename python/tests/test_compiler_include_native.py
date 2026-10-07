"""Actual compiler/cache behavior with an implicit current-directory include."""

import os

import numpy as np
import vecops.language as T
from dsl_test_utils import make_compiler
from vecops._identity import compiler_identity


def test_native_include_cwd_is_frozen_and_outputs_do_not_poison_identity(tmp_path, monkeypatch):
  project = tmp_path / "project"
  source_directory = project / "source"
  search_directory = project / "include"
  other_directory = tmp_path / "other"
  for path in (source_directory, search_directory, other_directory):
    path.mkdir(parents=True)
  header = project / "value.h"
  header.write_text("constexpr float included_value = 1;\n")
  # Angle includes must not use the source file's directory. Keeping a
  # same-named header there catches an incorrect dependency/shadow scan.
  (source_directory / "value.h").write_text("constexpr float included_value = 51;\n")
  (other_directory / "value.h").write_text("constexpr float included_value = 91;\n")
  source = source_directory / "kernel.cpp"
  source.write_text(
    '#include "vecops/Kernel.h"\n'
    "#include <value.h>\n"
    "void __kernel__(vecops::tensor::TensorLike auto C) { C.data()[0] = included_value; }\n"
  )
  monkeypatch.chdir(project)
  compiler = make_compiler(
    jobs=2,
    output="quiet",
    build_dir=tmp_path / "build",
    cache_dir=tmp_path / "cache",
    environment={
      "CPATH": str(search_directory) + os.pathsep,
      "CPLUS_INCLUDE_PATH": "",
      "C_INCLUDE_PATH": "",
    },
  )

  @T.jit(compiler=compiler)
  def factory():
    @T.template(source=source)
    def kernel(C: T.Tensor((1,), dtype=T.float32, access="output")):
      pass

    return kernel

  before = compiler_identity(compiler)
  first = factory()
  monkeypatch.chdir(other_directory)
  output = np.empty((1,), dtype=np.float32)
  first(output)
  np.testing.assert_array_equal(output, [1])
  assert before == compiler_identity(compiler)

  # A build/output-like file under the implicit cwd is not a toolchain or
  # source dependency. Include closure/shadow resolution still is.
  (project / "unrelated-build-output.log").write_text("completed\n")
  assert factory() is first
  assert before == compiler_identity(compiler)

  header.write_text("constexpr float included_value = 3;\n")
  changed = factory()
  assert changed is not first
  assert changed.get_cache_info()["key"] != first.get_cache_info()["key"]
  changed(output)
  np.testing.assert_array_equal(output, [3])
  first(output)
  np.testing.assert_array_equal(output, [1])
  assert before == compiler_identity(compiler)
