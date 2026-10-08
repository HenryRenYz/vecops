# SPDX-FileCopyrightText: 2026 HenryRenYz and vecops contributors
# SPDX-License-Identifier: MIT
"""JIT must preserve the compile configuration, including worker cost policy."""

import importlib

import vecops.language as T


def test_jit_forwards_compile_options_and_snapshots_option_mapping(monkeypatch):
  module = importlib.import_module("vecops.language.jit")
  calls = []
  compiler, pipeline, debug = object(), ("stream-row-maps",), object()
  options = {"cpu_grid_regions": "adjacent"}

  def compile_template(template, **kwargs):
    calls.append((template, kwargs))
    return object()

  monkeypatch.setattr(module, "compile", compile_template)

  @T.jit(compiler=compiler, pipeline=pipeline, pass_options=options, debug=debug, out_idx=[-1])
  def factory(N):
    @T.template
    def kernel(A: T.Tensor((N,), dtype="float32"), C: T.Tensor((N,), dtype="float32")):
      T.store(C, T.load(A))

    return kernel

  options["cpu_grid_regions"] = "individual"
  first = factory(17)
  assert factory(17) is first
  assert len(calls) == 1
  assert calls[0][1] == {
    "out_idx": [-1],
    "compiler": compiler,
    "pipeline": pipeline,
    "pass_options": {"cpu_grid_regions": "adjacent"},
    "debug": debug,
  }
  factory.cache_clear()
  assert factory(17) is not first
  assert len(calls) == 2


def test_jit_compile_options_do_not_compile_python_forwarders(monkeypatch):
  module = importlib.import_module("vecops.language.jit")

  def unexpected_compile(*args, **kwargs):
    raise AssertionError("ordinary Python forwarders are not templates")

  monkeypatch.setattr(module, "compile", unexpected_compile)

  @T.jit(pass_options={"cpu_grid_regions": "adjacent"}, pipeline=("stream-row-maps",))
  def factory(N):
    def forward(value):
      return value + int(N)

    return forward

  first = factory(3)
  assert first(5) == 8
  assert factory(3) is first
