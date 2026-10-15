# SPDX-FileCopyrightText: 2026 HenryRenYz and vecops contributors
# SPDX-License-Identifier: MIT
"""Pass registration, explicit Vector IR, unroll grouping and fusion contracts."""
import numpy as np
import pytest
import vecops.language as T
from dsl_test_utils import make_compiler
from vecops.language._compiler import ir as I
from vecops.language._compiler.ir import vector as V
from vecops.language._compiler.lowering.tile_to_vector import lower_region


@pytest.fixture(scope="module")
def compiler(tmp_path_factory):
  root = tmp_path_factory.mktemp("dsl-passes")
  return make_compiler(target="Native", jobs=4,
                       build_dir=root / "build", cache_dir=root / "cache")


def test_pass_registry_snapshot_order_and_analyses():
  P = T.passes
  @P.register_pass("unit-first", stage="unit", before="unit-second")
  def first(ir, context):
    context.analysis("value", lambda ir: 17, ir)
    ir.append("first")
    return P.PassResult(ir, changed=False)
  @P.register_pass("unit-second", stage="unit")
  def second(ir, context):
    assert context.analyses["value"] == 17
    ir.append("second")
    return ir
  try:
    manager = P.PassManager(stage="unit")
    with pytest.raises(ValueError, match="already registered"):
      P.register_pass("unit-first", stage="unit")(first)
    P.unregister_pass("unit-first")
    P.unregister_pass("unit-second")
    assert manager.run([]) == ["first", "second"]
  finally:
    for name in ("unit-first", "unit-second"):
      if any(p.name == name for p in P.registered_passes()):
        P.unregister_pass(name)
  @P.register_pass("unit-a", stage="cycle", after="unit-b")
  def a(ir, context): return ir
  @P.register_pass("unit-b", stage="cycle", after="unit-a")
  def b(ir, context): return ir
  try:
    with pytest.raises(ValueError, match="cycle"):
      P.PassManager(stage="cycle")
  finally:
    P.unregister_pass("unit-a"); P.unregister_pass("unit-b")


def test_registered_builtin_stages_and_operation_major_vector_ir():
  names = [p.name for p in T.passes.registered_passes(stage="kernel")]
  assert {"fuse-reductions", "fuse-reduction-elementwise", "fuse-elementwise", "unroll-serial"} <= set(names)
  view = I.View("A", "float16", (I.literal(257),), (I.literal(1),), I.literal(0))
  x = I.Expr(I.TileType("float32", view.shape), "load", data=view)
  expression = I.binary("add", I.binary("mul", x, x), I.literal(1))
  region = lower_region(view.shape, "float32", (expression,), unroll=7)
  assert [op.expression.op for op in region.operations] == ["load", "mul", "add"]
  assert [(c.operation, c.iteration) for c in region.copies] == [(op, i) for op in range(3) for i in range(7)]
  assert [(g.count, g.power) for g in region.groupings[2]] == [(4, 2), (2, 1), (1, 0)]
  assert region.pass_trace == [
    "cpu-plan-full-chunks",
    "cpu-choose-vector-chunk",
    "unroll-vector-operations",
    "coalesce-vector-groups",
    "legalize-vector-types",
    "cpu-legalize-math",
    "cpu-plan-native-word-tails",
    "verify-vector",
  ]
  assert all(isinstance(op.type, I.VectorType) for op in region.operations)


def test_max_power_groups_and_oversized_virtual_vector_split():
  assert [(g.count, g.power) for g in V.group_vectors(7, 1, 2)] == [(2, 2), (2, 2), (2, 2), (1, 1)]
  assert [(g.count, g.power) for g in V.group_vectors(7, 2, 2)] == [(1, 2)] * 7
  groups = V.group_vectors(2, 9, 2)
  assert [(g.power, g.repeat) for g in groups] == [(2, 128)] * 2
  typ = I.VectorType("float32", (I.literal(257),), tag=I.VectorTag(pow=2, basis_dtype="float32"))
  view = I.View("A", "float32", typ.shape, (I.literal(1),), I.literal(0))
  region = lower_region(typ.shape, "float32", (I.Expr(typ, "load", data=view),), unroll=7)
  assert region.base_power == 2
  assert [g.power for g in region.groupings[2]] == [4, 3, 2]


def test_compile_registered_plugin_and_explicit_pipeline(compiler):
  @T.passes.register_pass("unit-kernel-plugin", before="fuse-reductions")
  def plugin(ir, context):
    ir.diagnostics.append("plugin ran")
    return T.passes.PassResult(ir, changed=False)
  try:
    @T.template
    def kernel(A: T.Tensor((3,), dtype=T.float32), C: T.Tensor((3,), dtype=T.float32)):
      T.store(C, T.max(T.min(T.load(A), 3), 1))
    compiled = T.compile(kernel, out_idx=[-1], compiler=compiler)
    assert "plugin ran" in compiled.kernel_ir.diagnostics
    trace = compiled.kernel_ir.pass_trace
    assert trace.index("unit-kernel-plugin") < trace.index("fuse-reductions")
    np.testing.assert_array_equal(compiled(np.array([0, 2, 4], dtype=np.float32)), [1, 2, 3])
    plain = T.compile(kernel, out_idx=[-1], compiler=compiler, pipeline=[])
    assert plain.kernel_ir.pass_trace == []
    np.testing.assert_array_equal(plain(np.array([0, 2, 4], dtype=np.float32)), [1, 2, 3])
  finally:
    T.passes.unregister_pass("unit-kernel-plugin")


@pytest.mark.parametrize("unroll", [1, 2, 7])
def test_reduction_unroll_min_max_and_fused_group(compiler, unroll):
  N = T.dynamic(lower=0)
  @T.jit(out_idx=[-1], compiler=compiler)
  def factory(U):
    @T.template
    def kernel(A: T.Tensor((N,), dtype=T.float32), C: T.Tensor((3,), dtype=T.float32)):
      x = T.load(A)
      add = T.reduce_add(x, unroll=U)
      minimum = T.reduce_min(x, unroll=U)
      maximum = T.reduce_max(x, unroll=U)
      C[0] = add
      C[1] = minimum
      C[2] = maximum
    return kernel
  kernel = factory(unroll)
  for width in (1, 17, 113, 257):
    a = np.linspace(-3, 5, width, dtype=np.float32)
    np.testing.assert_allclose(kernel(a), [a.sum(), a.min(), a.max()], rtol=2e-6, atol=1e-5)
  assert "fuse-reductions" in kernel.kernel_ir.pass_trace
  assert all(region.unroll == unroll for region in kernel.vector_ir.regions)


@pytest.mark.parametrize("unroll", [1, 2, 7])
def test_axis_reduction_unroll(compiler, unroll):
  @T.jit(out_idx=[-1], compiler=compiler)
  def factory(U):
    @T.template
    def kernel(A: T.Tensor((19, 37), dtype=T.float32), C: T.Tensor((19,), dtype=T.float32)):
      T.store(C, T.reduce_add(T.load(A) * 2, axis=1, unroll=U))
    return kernel
  kernel = factory(unroll)
  a = np.arange(19 * 37, dtype=np.float32).reshape(19, 37)
  np.testing.assert_array_equal(kernel(a), (a * 2).sum(1))


def test_reduce_names_and_unroll_diagnostics(compiler):
  assert not hasattr(T, "sum")
  @T.template
  def invalid(A: T.Tensor((3,), dtype=T.float32)):
    T.max(T.load(A))
  with pytest.raises(TypeError, match="two elementwise operands"):
    T.compile(invalid, compiler=compiler)
  @T.template
  def bad(A: T.Tensor((3,), dtype=T.float32)):
    T.reduce_add(T.load(A), unroll=0)
  with pytest.raises(TypeError, match="positive compile-time integer"):
    T.compile(bad, compiler=compiler)


@pytest.mark.parametrize("step", [1, -1])
def test_serial_operation_major_unroll_tail_and_alias_fallback(compiler, step):
  @T.jit(out_idx=[-1], compiler=compiler)
  def factory(S):
    @T.template
    def kernel(A: T.Tensor((17,), dtype=T.float32), C: T.Tensor((17,), dtype=T.float32)):
      if S > 0:
        for i in T.serial(17, unroll=7):
          x = T.load(A[i])
          C[i] = x * 2 + 1
      else:
        for i in T.serial(16, -1, -1, unroll=7):
          x = T.load(A[i])
          C[i] = x * 2 + 1
    return kernel
  kernel = factory(step)
  a = np.arange(17, dtype=np.float32)
  np.testing.assert_array_equal(kernel(a), a * 2 + 1)
  loop = next(s for s in kernel.kernel_ir.body.statements if isinstance(s, I.UnrolledFor))
  assert len(loop.copies.statements) == len(loop.loop.body.statements) * 7
  assert "operation-major serial unroll" in kernel.generated_source
  base = np.arange(18, dtype=np.float32)
  expected = base.copy()
  for i in (range(17) if step > 0 else range(16, -1, -1)):
    expected[i + 1] = expected[i] * 2 + 1
  kernel(base[:-1], base[1:])
  np.testing.assert_array_equal(base, expected)


def test_elementwise_fusion_and_sequential_alias_effects(compiler):
  @T.jit(out_idx=[-2, -1], compiler=compiler)
  def factory():
    @T.template
    def kernel(A: T.Tensor((113,), dtype=T.float32), B: T.Tensor((113,), dtype=T.float32),
               C: T.Tensor((113,), dtype=T.float32)):
      T.store(B, T.load(A) * 2)
      T.store(C, T.load(A) + 1)
    return kernel
  kernel = factory()
  assert any(isinstance(s, I.ElementwiseGroup) for s in kernel.kernel_ir.body.statements)
  a = np.arange(113, dtype=np.float32)
  b, c = kernel(a)
  np.testing.assert_array_equal(b, a * 2)
  np.testing.assert_array_equal(c, a + 1)
  original = a.copy()
  result = np.empty_like(a)
  kernel(a, a, result)
  np.testing.assert_array_equal(result, original * 2 + 1)


def test_reduction_elementwise_fusion_inplace_and_overlap(compiler):
  @T.jit(out_idx=[-2, -1], compiler=compiler)
  def factory():
    @T.template
    def kernel(A: T.Tensor((113,), dtype=T.float32), B: T.Tensor((113,), dtype=T.float32),
               C: T.Tensor((1,), dtype=T.float32)):
      total = T.reduce_add(T.load(A), unroll=7)
      T.store(B, T.load(A) * 2)
      C[0] = total
    return kernel
  kernel = factory()
  assert any(isinstance(s, I.ReductionGroup) and s.stores for s in kernel.kernel_ir.body.statements)
  base = np.arange(114, dtype=np.float32)
  expected = base[:-1].copy()
  total = np.empty(1, dtype=np.float32)
  kernel(base[:-1], base[1:], total)
  np.testing.assert_array_equal(base[1:], expected * 2)
  np.testing.assert_allclose(total, [expected.sum()])


def test_expanding_load_narrowing_store_vector_groups(compiler):
  @T.jit(out_idx=[-2, -1], compiler=compiler)
  def factory():
    @T.template
    def kernel(A: T.Tensor((113,), dtype=T.float16), B: T.Tensor((113,), dtype=T.float16),
               C: T.Tensor((1,), dtype=T.float32)):
      x = T.load(A, dtype=T.float32)
      total = T.reduce_add(x * 2, unroll=7)
      T.store(B, x * 2)
      C[0] = total
    return kernel
  kernel = factory()
  a = np.arange(113, dtype=np.float16)
  b, c = kernel(a)
  np.testing.assert_array_equal(b, a * 2)
  np.testing.assert_allclose(c, [(a.astype(np.float32) * 2).sum()])
  plan = next(r for r in kernel.vector_ir.regions if r.unroll == 7)
  assert [g.count for g in plan.groupings[2]] == [4, 2, 1]
  assert "ScalableTag<float, 2>" in kernel.generated_source
  assert "load_convert" in kernel.generated_source and "store_convert" in kernel.generated_source


def test_cpu_policies_are_independent_of_explicit_vector_contracts():
  shape = (I.literal(257),)
  view = I.View("A", "float32", shape, (I.literal(1),), I.literal(0))
  def expression(typ):
    return I.Expr(typ, "exp_neg", (I.Expr(typ, "load", data=view),), "strict")
  cpu = T.passes.PassManager(stage="vector", options={"backend": "cpu"})
  auto = lower_region(shape, "float32", (expression(I.TileType("float32", shape)),), manager=cpu)
  assert auto.base_power == 1 and auto.unroll == 2 and auto.full_chunks
  assert auto.operations[-1].native_math_words
  explicit = lower_region(shape, "float32", (expression(I.VectorType("float32", shape)),), manager=cpu)
  assert explicit.base_power == 0 and explicit.unroll == 2
  disabled = T.passes.PassManager(stage="vector", options={"backend": "cpu", "cpu_full_chunks": False,
                       "cpu_math_chunk_pow": 0, "cpu_wordwise_math": False})
  plain = lower_region(shape, "float32", (expression(I.TileType("float32", shape)),), manager=disabled)
  assert plain.base_power == 0 and not plain.full_chunks
  assert not any(op.native_math_words for op in plain.operations)
  from vecops.language._compiler.lowering.tile_to_vector import lower_axis
  axis = lower_axis(expression(I.TileType("float32", shape)), "float32", 0, "add", 2, manager=cpu)
  assert axis.base_power == 0 and all(g.count == 1 for g in axis.groupings[0])
  from vecops.language._compiler.lowering.tile_to_vector import lower_reduction
  load = I.Expr(I.TileType("float32", shape), "load", data=view)
  reduction = I.Expr(I.ScalarType("float32"), "reduce_max", (load,), 2)
  assert lower_reduction((reduction,), manager=cpu).base_power == 1
  narrow = T.passes.PassManager(stage="vector", options={"backend": "cpu", "cpu_reduction_chunk_pow": 0})
  assert lower_reduction((reduction,), manager=narrow).base_power == 0
  for option in ({"cpu_full_chunks": 1}, {"cpu_math_chunk_pow": -1}, {"cpu_wordwise_math": 1}):
    with pytest.raises(ValueError, match="must be"):
      lower_region(shape, "float32", (expression(I.TileType("float32", shape)),),
                   manager=T.passes.PassManager(stage="vector", options={"backend": "cpu", **option}))


@pytest.mark.parametrize("dtype", [T.float32, T.float64])
def test_cpu_full_chunks_math_and_empty_tails(compiler, dtype):
  N = T.dynamic(lower=0)
  @T.jit(out_idx=[-2, -1], compiler=compiler)
  def factory(D):
    @T.template
    def kernel(A: T.Tensor((N,), dtype=D), C: T.Tensor((N,), dtype=D), R: T.Tensor((1,), dtype=D)):
      x = T.load(A)
      e = T.exp_neg(x)
      total = T.reduce_add(e)
      T.store(C, e)
      R[0] = total
    return kernel
  kernel = factory(dtype)
  np_dtype = np.float32 if dtype == T.float32 else np.float64
  for width in (0, 1, 15, 16, 17, 31, 32, 33, 63, 64, 65, 127, 128, 129, 257):
    a = np.linspace(-5, -0.01, width, dtype=np_dtype)
    c, r = kernel(a)
    expected = np.exp(a)
    np.testing.assert_allclose(c, expected, rtol=2e-6, atol=1e-7)
    np.testing.assert_allclose(r, [expected.sum()], rtol=2e-6, atol=1e-7)
  assert "full vector groups" in kernel.generated_source
  assert "masked vector tail" in kernel.generated_source
  plan = kernel.vector_ir.regions[0]
  assert plan.initializations and plan.effects and plan.finalizations


def test_cse_keeps_shared_edges_through_parent_operations(compiler):
  @T.template
  def kernel(A: T.Tensor((35,), dtype=T.float32), B: T.Tensor((35,), dtype=T.float32),
             C: T.Tensor((35,), dtype=T.float32)):
    x = T.load(A)
    first = x * x
    second = x * x
    T.store(B, first + 1)
    T.store(C, second + 2)
  compiled = T.compile(kernel, out_idx=[-2, -1], compiler=compiler)
  a = np.arange(35, dtype=np.float32)
  b, c = compiled(a)
  np.testing.assert_array_equal(b, a * a + 1)
  np.testing.assert_array_equal(c, a * a + 2)
  fused = next(region for region in compiled.vector_ir.regions if len(region.roots) == 2)
  assert [op.expression.op for op in fused.operations].count("mul") == 1


def test_reductions_are_scoped_and_scalar_control_remains_lazy(compiler):
  import re
  N = T.dynamic(lower=1)
  @T.jit(out_idx=[-1], compiler=compiler)
  def factory():
    @T.template
    def kernel(A: T.Tensor((2, N), dtype=T.float32), C: T.Tensor((6,), dtype=T.float32),
               *, flag: bool = False):
      C[0] = T.reduce_add(T.load(A[0, :])) if flag else T.reduce_add(T.load(A[1, :]))
      C[1] = T.where(flag, T.reduce_min(T.load(A[0, :])), T.reduce_max(T.load(A[1, :])))
      positive = flag and T.reduce_min(T.load(A[0, :])) > 0
      negative = flag or T.reduce_max(T.load(A[1, :])) < 0
      C[2] = T.where(positive, 1.0, 0.0)
      C[3] = T.where(negative, 1.0, 0.0)
      i = 0
      while i < 2 and T.reduce_add(T.load(A[i, :])) > 0:
        i += 1
        if i == 1:
          continue
      C[4] = i
      C[5] = T.load(A[0, 0], mask=flag, other=T.reduce_min(T.load(A[1, :])))
    return kernel
  kernel = factory()
  for width in (1, 17, 65):
    a = np.stack([np.arange(1, width + 1), -np.arange(1, width + 1)]).astype(np.float32)
    for flag in (False, True):
      expected = [a[0].sum() if flag else a[1].sum(), a[0].min() if flag else a[1].max(),
                  int(flag), 1, 1, a[0, 0] if flag else a[1].min()]
      np.testing.assert_allclose(kernel(a, flag=flag), expected, atol=1e-5, rtol=2e-6)
    a[1] = a[0]
    assert kernel(a)[4] == 2  # condition reevaluates after continue; final arm short-circuits
  source = kernel.generated_source
  assert "/* reduction region: no capture boundary */" in source
  assert "while (true)" in source
  assert "dsl_c_scalar_select_" in source
  # Shared target helpers may use a forced-inline constexpr initializer;
  # reduction/control-flow lowering itself must not introduce an IIFE.
  assert not re.search(r"\[[^\]\n]*\]\(\)", source.split("void __kernel__(", 1)[1])
  assert all(tail.lstrip().startswith("VECOPS_INLINE_LAMBDA")
             for tail in re.split(r"\[[^\]\n]*\]\(\)", source)[1:])


def test_expression_only_reduction_closures_are_forced_inline(compiler):
  import re
  @T.template
  def kernel(A: T.Tensor((3,), dtype=T.float32), C: T.Tensor((1,), dtype=T.float32)):
    count = 0
    for i in T.serial(T.cast(T.reduce_add(T.load(A)), T.int64), unroll=1):
      count += 1
      if i == 1:
        continue
    C[0] = count
  compiled = T.compile(kernel, out_idx=[-1], compiler=compiler)
  np.testing.assert_array_equal(compiled(np.ones(3, dtype=np.float32)), [3])
  source = compiled.generated_source
  assert "([&]() VECOPS_INLINE_LAMBDA" in source
  assert all(tail.lstrip().startswith("VECOPS_INLINE_LAMBDA")
             for tail in re.split(r"\[[^\]\n]*\]\(\)", source)[1:])
