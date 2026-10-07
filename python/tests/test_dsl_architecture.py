"""Package boundaries and compilation/emission separation."""

import ast
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from pathlib import Path

import vecops.language as T
from vecops.language._compiler import ir as I
from vecops.language._compiler.backends.cpu.cpp.emitter import generate
from vecops.language._compiler.backends.cpu.planning import prepare
from vecops.language._compiler.frontend.parser import lower
from vecops.language._compiler.ir.traversal import rewrite_expressions
from vecops.language._compiler.lowering.tile_to_vector import lower_kernel
from vecops.language._compiler.pipeline import cpu_vector_pipeline, load_builtin_passes, optimize

ROOT = Path(T.__file__).parent


def test_public_facades_and_private_layout():
  import pickle

  from vecops.language._compiler.passes.manager import PassManager
  from vecops.language.compile import CompiledKernel, compile, compile_pending

  assert (CompiledKernel, compile, compile_pending) == (T.CompiledKernel, T.compile, T.compile_pending)
  assert T.passes.PassManager is PassManager
  for value in (
    CompiledKernel,
    compile,
    PassManager,
    T.passes.PassContext,
    T.passes.PassResult,
    T.passes.register_pass,
  ):
    assert pickle.loads(pickle.dumps(value)) is value
  assert all(
    not (ROOT / name).exists()
    for name in (
      "_frontend.py",
      "_ir.py",
      "_vector_ir.py",
      "_vector_lowering.py",
      "_codegen.py",
      "_optimize.py",
      "_cpu_vector.py",
    )
  )


def imports(path):
  tree = ast.parse(path.read_text())
  return [(node.module or "") for node in ast.walk(tree) if isinstance(node, ast.ImportFrom)]


def test_ir_is_independent_and_cpp_cannot_execute_transforms():
  for path in (ROOT / "_compiler/ir").glob("*.py"):
    assert not any(
      any(part in module.split(".") for part in ("frontend", "backends", "_runtime", "passes", "lowering"))
      for module in imports(path)
    ), path
  for path in (ROOT / "_compiler/backends/cpu/cpp").glob("*.py"):
    assert not any(
      any(part in module.split(".") for part in ("frontend", "lowering", "passes", "pipeline"))
      for module in imports(path)
    ), path
    tree = ast.parse(path.read_text())
    assert not any(
      isinstance(node, ast.Call)
      and isinstance(node.func, ast.Attribute)
      and node.func.attr in {"run", "lower_region", "lower_axis", "lower_stores", "lower_reduction"}
      for node in ast.walk(tree)
    ), path


def test_builtin_loading_is_idempotent_and_order_stable():
  before = T.passes.registered_passes()
  snapshot = T.passes.PassManager()
  with ThreadPoolExecutor(max_workers=4) as pool:
    list(pool.map(lambda _: load_builtin_passes(), range(24)))
  assert T.passes.registered_passes() == before
  assert tuple(snapshot.specs) == T.passes.PassManager().specs
  assert [p.name for p in snapshot.specs] == [
    "normalize-reduction-layout",
    "stream-row-maps",
    "decompose-row-maps",
    "fold-value-transforms",
    "canonicalize-memory",
    "expression-cse",
    "fuse-dot-epilogue",
    "materialize-dependent-dots",
    "orient-matmul",
    "normalize-simd-access",
    "normalize-iteration-regions",
    "plan-computation-reuse",
    "fuse-reductions",
    "fuse-reduction-elementwise",
    "cpu-expand-dot-recipes",
    "fuse-elementwise",
    "fuse-axis-folds",
    "fuse-loop-regions",
    "fuse-loop-consumers",
    "unroll-serial",
    "refresh-computation-reuse",
    "plan-software-pipeline",
  ]


def test_emission_runs_no_passes_or_lowering(monkeypatch):
  @T.template
  def kernel(
    A: T.Tensor((4, 17), dtype=T.float32),
    C: T.Tensor((17,), dtype=T.float32, access="output"),
    D: T.Tensor((17,), dtype=T.float32, access="output"),
  ):
    y = T.reduce_add(T.load(A), axis=0, unroll=3)
    T.store(C, y)
    T.store(D, T.load(C) + 1)

  ir = optimize(lower(kernel.ir))
  vectors = lower_kernel(ir, manager=cpu_vector_pipeline())
  prepared = prepare(vectors)
  primary_regions = tuple(vectors.regions)

  def forbidden(*args, **kwargs):
    raise AssertionError("compiler work leaked into source emission")

  monkeypatch.setattr(T.passes.PassManager, "run", forbidden)
  from vecops.language._compiler.lowering import tile_to_vector

  for name in ("lower_region", "lower_stores", "lower_axis", "lower_reduction"):
    monkeypatch.setattr(tile_to_vector, name, forbidden)
  first, extra = generate(prepared)
  second, extra_again = generate(prepared)
  assert first == second
  assert "alias" in first and "reduction_axis" in first
  assert extra and len(extra) == len(extra_again)
  assert tuple(vectors.regions) == primary_regions


def test_ir_rewriting_does_not_depend_on_dataclass_module_names():
  extent = I.literal(17)
  view = I.View("A", "float32", (extent,), (I.literal(1),), I.literal(0))
  value = I.Expr(I.TileType("float32", view.shape), "load", data=view)
  ir = I.KernelIR(I.Block([I.Store(view, value)]), {}, {})
  rewrite_expressions(ir, lambda e: replace(e, data=19) if e is extent else e)
  assert I.constant(ir.body.statements[0].destination.shape[0]) == 19
  assert I.VectorType("float32", (extent,)) == I.VectorType("float32", (extent,))


def test_backend_contract_has_no_cpu_lowering_requirement():
  modules = imports(ROOT / "_compiler/backends/base.py")
  assert not any("cpu" in module or "vector" in module or "cpp" in module for module in modules)


def test_both_pipeline_snapshots_precede_user_pass_execution():
  from vecops.language._compiler.backends.cpu.backend import CPUBackend

  P = T.passes

  @T.template
  def kernel(A: T.Tensor((17,), dtype=T.float32), C: T.Tensor((17,), dtype=T.float32, access="output")):
    T.store(C, T.load(A) + 1)

  vector_name = "unit-late-vector-registration"
  kernel_name = "unit-register-vector-during-kernel-pass"

  @P.register_pass(kernel_name, before="canonicalize-memory")
  def register_late(ir, context):
    if not any(spec.name == vector_name for spec in P.registered_passes(stage="vector")):

      @P.register_pass(vector_name, stage="vector")
      def late(region, context):
        return region

    return ir

  try:
    first = CPUBackend().lower(kernel.ir, ())
    assert all(vector_name not in region.pass_trace for region in first.lowered.regions)
    second = CPUBackend().lower(kernel.ir, ())
    assert all(vector_name in region.pass_trace for region in second.lowered.regions)
    assert first.source == second.source
  finally:
    P.unregister_pass(kernel_name)
    if any(spec.name == vector_name for spec in P.registered_passes(stage="vector")):
      P.unregister_pass(vector_name)


def test_memory_pipeline_is_frozen_before_user_passes():
  from vecops.language._compiler.backends.cpu.backend import CPUBackend

  P = T.passes

  @T.template
  def kernel(A: T.Tensor((16, 16), dtype=T.float32), C: T.Tensor((16, 16), dtype=T.float32, access="output")):
    T.transpose_copy(A, C)

  name, registrar = "unit-late-memory", "unit-register-memory-during-kernel-pass"
  calls = []

  @P.register_pass(registrar, before="canonicalize-memory")
  def register_late(ir, context):
    if not any(spec.name == name for spec in P.registered_passes(stage="cpu-memory")):

      @P.register_pass(name, stage="cpu-memory", after="cpu-plan-transpose")
      def late(prepared, context):
        calls.append(len(prepared.transposes))
        return prepared

    return ir

  try:
    CPUBackend().lower(kernel.ir, ())
    assert calls == []
    CPUBackend().lower(kernel.ir, ())
    assert calls == [1]
  finally:
    P.unregister_pass(registrar)
    if any(spec.name == name for spec in P.registered_passes(stage="cpu-memory")):
      P.unregister_pass(name)
