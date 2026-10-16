# SPDX-FileCopyrightText: 2026 HenryRenYz and vecops contributors
# SPDX-License-Identifier: MIT
"""Conditional producer facts preserve lazy storage and historical reads."""

from dataclasses import dataclass, replace

import pytest
import vecops.language as T
from vecops.language._compiler import ir as I
from vecops.language._compiler.environment import freeze
from vecops.language._compiler.ir.traversal import nodes
from vecops.language._compiler.operations import OpDef, snapshot, use_catalog
from vecops.language._compiler.operations.registry import OperationCatalog
from vecops.language._compiler.passes.kernel.cleanup import simplify_ir
from vecops.language._compiler.passes.kernel.facts import propagate_value_facts
from vecops.language._compiler.passes.manager import PassContext


def local(name="enabled", dtype="bool"):
  return I.Expr(I.ScalarType(dtype), "local", data=name)


def values():
  shape = (I.literal(8),)
  source = I.View("A", "float32", shape, (I.literal(1),), I.literal(0), writable=False)
  cache = replace(source, storage="cache", owned=True, writable=True)
  output = replace(source, storage="C", writable=True)
  producer = I.Expr(I.TileType("float32", shape), "load", data=source)
  return producer, cache, output


def cached(producer, cache, guard=None):
  return I.Expr(producer.type, "cached_value", (producer,), (cache, local() if guard is None else guard))


def run(statements, *, cleanup=True, catalog=None, iterations=2):
  kernel = I.KernelIR(I.Block(statements), {}, {})
  environment = freeze()
  if catalog is not None:
    environment = replace(environment, operations=catalog)
  context = PassContext("kernel-cleanup", {"_compiler_environment": environment})
  with use_catalog(environment.operations):
    for _ in range(iterations):
      propagate_value_facts(kernel, context)
      if cleanup:
        simplify_ir(kernel, context)
  return kernel


def setup(producer, cache, guard=None):
  return I.If(local() if guard is None else guard, I.Block([I.Allocate(cache), I.Store(cache, producer)]), I.Block([]))


def test_false_cache_consumers_and_bindings_select_original_historical_producer():
  producer, cache, output = values()
  wrapper = cached(producer, cache)
  kernel = run([I.Assign("enabled", I.literal(False)), setup(producer, cache),
                I.BindTile(wrapper), I.Store(output, wrapper)])
  assert not any(isinstance(node, I.Allocate) for node in nodes(kernel.body))
  assert kernel.body.statements[-2].value is producer
  assert kernel.body.statements[-1].value is producer
  assert next(I.loads_unique(kernel.body.statements[-1].value)) is producer


@pytest.mark.parametrize("iterations", [1, 4])
def test_owner_removal_waits_until_next_round_proves_no_cache_descriptor_remains(iterations):
  producer, cache, output = values()
  wrapper = cached(producer, cache)
  kernel = run([I.Assign("enabled", I.literal(False)), setup(producer, cache), I.Store(output, wrapper)],
               iterations=iterations)
  assert kernel.body.statements[-1].value is producer
  allocations = [node for node in nodes(kernel.body) if isinstance(node, I.Allocate)]
  assert bool(allocations) is (iterations == 1)


@pytest.mark.parametrize("operation", ["cast", "value_transform", "reduce_add"])
def test_nested_value_consumers_share_rewrite_and_preserve_read_leaf(operation):
  producer, cache, _output = values()
  wrapper = cached(producer, cache)
  typ = I.ScalarType("float32") if operation == "reduce_add" else producer.type
  data = ("transpose", (0,)) if operation == "value_transform" else 2 if operation == "reduce_add" else None
  parent = I.Expr(typ, operation, (wrapper,), data)
  kernel = run([I.Assign("enabled", I.literal(False)), setup(producer, cache),
                I.Assign("first", parent), I.Assign("second", parent)])
  first, second = kernel.body.statements[-2:]
  assert first.value is second.value
  assert first.value is not parent
  assert first.value.args[0] is producer


@pytest.mark.parametrize("known", [True, None])
def test_true_or_unknown_guard_preserves_wrapper_and_owner(known):
  producer, cache, output = values()
  wrapper = cached(producer, cache)
  prefix = [] if known is None else [I.Assign("enabled", I.literal(known))]
  kernel = run([*prefix, setup(producer, cache), I.Store(output, wrapper)])
  assert kernel.body.statements[-1].value is wrapper
  assert any(isinstance(node, I.Allocate) and node.view is cache for node in nodes(kernel.body))


def test_shared_wrapper_observes_writes_at_each_program_point():
  producer, cache, output = values()
  wrapper = cached(producer, cache)
  kernel = run([I.Assign("enabled", I.literal(False)), setup(producer, cache), I.Store(output, wrapper),
                I.Assign("enabled", I.literal(True), declare=False), setup(producer, cache), I.Store(output, wrapper)])
  stores = [node for node in kernel.body.statements if isinstance(node, I.Store)]
  assert stores[0].value is producer
  assert stores[1].value is wrapper
  # Mutable selectors retain both guarded owners; facts cannot delete a
  # declaration while another use of this shared wrapper remains unresolved.
  assert len([node for node in kernel.body.statements if isinstance(node, I.If)]) == 2


def test_branch_fact_does_not_escape_into_shared_wrapper():
  producer, cache, output = values()
  wrapper = cached(producer, cache)
  branch = I.If(local("choice"), I.Block([I.Assign("enabled", I.literal(False)), I.Store(output, wrapper)]), I.Block([]))
  kernel = run([I.Allocate(cache), branch, I.Store(output, wrapper)])
  assert branch.body.statements[-1].value is producer
  assert kernel.body.statements[-1].value is wrapper


def test_loop_writes_kill_incoming_selector_fact():
  producer, cache, output = values()
  wrapper = cached(producer, cache)
  loop = I.For("i", I.literal(0), I.literal(4), I.literal(1),
               I.Block([I.Store(output, wrapper), I.Assign("enabled", I.literal(True), declare=False)]))
  kernel = run([I.Assign("enabled", I.literal(False)), I.Allocate(cache), loop, I.Store(output, wrapper)])
  assert loop.body.statements[0].value is wrapper
  assert kernel.body.statements[-1].value is wrapper


def test_false_selection_never_drops_trapping_or_state_producer():
  producer, cache, output = values()
  trap = I.Expr(I.ScalarType("int64"), "floordiv", (I.literal(1), I.literal(0)))
  producer = I.Expr(producer.type, "add", (producer, trap))
  wrapper = cached(producer, cache)
  kernel = run([I.Assign("enabled", I.literal(False)), setup(producer, cache), I.Store(output, wrapper)])
  assert kernel.body.statements[-1].value is producer
  assert trap in tuple(nodes(kernel.body.statements[-1].value))
  state = I.TileState("state", producer.type)
  reference = I.Expr(producer.type, "state_ref", data=state)
  state_wrapper = cached(reference, cache)
  kernel = run([I.Assign("enabled", I.literal(False)), I.Store(output, state_wrapper)])
  assert kernel.body.statements[-1].value is reference


@pytest.mark.parametrize("guard_kind", ["trap", "read", "malformed_literal", "wrong_type"])
def test_unproven_or_non_total_selector_is_not_discarded(guard_kind):
  producer, cache, output = values()
  if guard_kind == "trap":
    trap = I.Expr(I.ScalarType("int64"), "floordiv", (I.literal(1), I.literal(0)))
    guard = I.Expr(I.ScalarType("bool"), "eq", (trap, I.literal(0)))
  elif guard_kind == "read":
    guard = I.Expr(I.ScalarType("bool"), "load", data=replace(cache, shape=(), strides=()))
  elif guard_kind == "malformed_literal":
    guard = I.Expr(I.ScalarType("bool"), "literal", (producer,), False)
  else:
    guard = I.literal(0.0)
  wrapper = cached(producer, cache, guard)
  kernel = run([I.Store(output, wrapper)], cleanup=False)
  assert kernel.body.statements[-1].value is wrapper
  assert wrapper.data[1] is guard


def test_unknown_guard_subclass_cannot_hide_owned_read_behind_false_literal():
  @dataclass(frozen=True, eq=False)
  class OwnedGuard(I.Expr):
    hidden: object = None

  producer, cache, output = values()
  guard = OwnedGuard(I.ScalarType("bool"), "literal", data=False, hidden=producer)
  wrapper = cached(producer, cache, guard)
  kernel = run([I.Store(output, wrapper)], cleanup=False)
  assert kernel.body.statements[-1].value is wrapper
  assert wrapper.data[1] is guard
  assert guard.hidden is producer
  indirect = cached(producer, cache)
  kernel = run([I.Assign("enabled", guard), setup(producer, cache), I.Store(output, indirect)])
  assert kernel.body.statements[-1].value is indirect
  assert any(isinstance(node, I.Allocate) for node in nodes(kernel.body))


def test_blocked_cache_selector_does_not_disable_unrelated_boolean_facts():
  producer, cache, output = values()
  wrapper = cached(producer, cache)
  read = I.Expr(producer.type, "gather", (wrapper,), producer.data)
  unrelated = I.If(local("other"), I.Block([I.Store(output, producer)]), I.Block([]))
  kernel = run([I.Assign("enabled", I.literal(False)), setup(producer, cache), I.Store(output, read),
                I.Assign("other", I.literal(False)), unrelated])
  branches = [node for node in kernel.body.statements if isinstance(node, I.If)]
  assert len(branches) == 1
  assert branches[0].condition.data == "enabled"


@pytest.mark.parametrize("predicate", ["local", "not", "and", "alias"])
def test_unrewritable_owned_metadata_preserves_whole_allocation_predicate(predicate):
  producer, cache, output = values()
  wrapper = cached(producer, cache)
  # This extension holds its value in owned metadata. The facts pass must
  # preserve the selector's control structure instead of rewriting that data.
  base = snapshot()
  definitions = dict(base.definitions)
  definitions["test.metadata_cache"] = OpDef("test.metadata_cache", "test", effects="none")
  catalog = OperationCatalog(definitions, "cached-metadata-barrier")
  use = I.Expr(producer.type, "test.metadata_cache", data=wrapper)
  prefix = [I.Assign("enabled", I.literal(False))]
  guard = local()
  if predicate == "not":
    guard = I.Expr(I.ScalarType("bool"), "not", (guard,))
  elif predicate == "and":
    guard = I.Expr(I.ScalarType("bool"), "and", (guard, local("choice")))
  elif predicate == "alias":
    prefix.append(I.Assign("alias", guard))
    guard = local("alias")
  kernel = run([*prefix, setup(producer, cache, guard), I.Store(output, use)], catalog=catalog)
  assert any(isinstance(node, I.If) for node in kernel.body.statements)
  assert kernel.body.statements[-1].value is use
  assert any(isinstance(node, I.Allocate) and node.view is cache for node in nodes(kernel.body))


def test_cached_value_under_historical_read_is_not_cloned():
  producer, cache, output = values()
  wrapper = cached(producer, cache)
  read = I.Expr(producer.type, "gather", (wrapper,), producer.data)
  kernel = run([I.Assign("enabled", I.literal(False)), setup(producer, cache), I.Store(output, read)])
  assert kernel.body.statements[-1].value is read
  assert read.args[0] is wrapper
  assert any(isinstance(node, I.Allocate) and node.view is cache for node in nodes(kernel.body))


@pytest.mark.parametrize("container", ["dataclass", "custom", "dict_subclass", "dict"])
def test_unknown_owned_container_preserves_prior_allocation_without_inspecting_plugin(container):
  producer, cache, output = values()
  wrapper = cached(producer, cache)

  @dataclass
  class Descriptor:
    hidden: object

  class Custom:
    def __init__(self, hidden):
      self.hidden = hidden

    def __iter__(self):
      pytest.fail("facts must not iterate an unknown metadata container")

  class Dictionary(dict):
    def items(self):
      pytest.fail("facts must not execute a container subclass's items")

  data = (Descriptor(wrapper) if container == "dataclass" else Custom(wrapper) if container == "custom"
          else Dictionary(hidden=wrapper) if container == "dict_subclass" else {"hidden": wrapper})
  base = snapshot()
  definitions = dict(base.definitions)
  definitions["test.hidden_cache"] = OpDef("test.hidden_cache", "test", effects="none")
  catalog = OperationCatalog(definitions, "cached-unknown-container-barrier")
  use = I.Expr(producer.type, "test.hidden_cache", data=data)
  kernel = run([I.Assign("enabled", I.literal(False)), setup(producer, cache), I.Store(output, use)], catalog=catalog)
  assert kernel.body.statements[-1].value is use
  assert any(isinstance(node, I.If) for node in kernel.body.statements)
  assert any(isinstance(node, I.Allocate) and node.view is cache for node in nodes(kernel.body))


@pytest.mark.parametrize("metadata_kind", ["unknown_operation", "invalid_raw"])
def test_pure_parent_with_opaque_metadata_cannot_delete_owned_cache(metadata_kind):
  producer, cache, output = values()
  wrapper = cached(producer, cache)
  metadata = (I.Expr(I.ScalarType("int64"), "test.unknown") if metadata_kind == "unknown_operation"
              else I.Expr(I.ScalarType("int64"), "raw", data="unmodeled + expression"))
  base = snapshot()
  definitions = dict(base.definitions)
  definitions["test.parent"] = OpDef("test.parent", "test", effects="none")
  catalog = OperationCatalog(definitions, "cached-opaque-parent-barrier")
  use = I.Expr(producer.type, "test.parent", (wrapper,), metadata)
  kernel = run([I.Assign("enabled", I.literal(False)), setup(producer, cache), I.Store(output, use)], catalog=catalog)
  assert kernel.body.statements[-1].value is use
  assert any(isinstance(node, I.Allocate) and node.view is cache for node in nodes(kernel.body))


@pytest.mark.parametrize("kind", ["expression", "descriptor", "guard"])
def test_unknown_ir_subclass_does_not_execute_custom_field_access(kind):
  producer, cache, output = values()
  wrapper = cached(producer, cache)

  @dataclass(frozen=True, eq=False)
  class UnknownExpr(I.Expr):
    hidden: object = None

    def __getattribute__(self, name):
      if name in {"op", "hidden"}:
        pytest.fail("facts must not inspect unknown Expr ownership")
      return super().__getattribute__(name)

  @dataclass(frozen=True)
  class UnknownDescriptor(I.Node):
    hidden: object

    def __getattribute__(self, name):
      if name == "hidden":
        pytest.fail("facts must not inspect unknown Node ownership")
      return super().__getattribute__(name)

  metadata = (UnknownExpr(I.ScalarType("bool"), "literal", data=False, hidden=producer) if kind == "guard"
              else UnknownExpr(producer.type, "test.unknown", hidden=wrapper) if kind == "expression"
              else UnknownDescriptor(wrapper))
  base = snapshot()
  definitions = dict(base.definitions)
  definitions["test.parent"] = OpDef("test.parent", "test", effects="none")
  catalog = OperationCatalog(definitions, "cached-custom-access-barrier")
  use = cached(producer, cache, metadata) if kind == "guard" else I.Expr(producer.type, "test.parent", data=metadata)
  # The source pass, including its prior-definition and control scans, must
  # treat extension-owned fields as opaque rather than executing accessors.
  kernel = run([I.Assign("enabled", I.literal(False)), setup(producer, cache), I.Store(output, use)],
               cleanup=False, catalog=catalog)
  assert kernel.body.statements[-1].value is use
  assert any(isinstance(node, I.If) for node in kernel.body.statements)
  assert any(isinstance(node, I.Allocate) for node in kernel.body.statements[-2].body.statements)


def test_opaque_historical_binding_kills_prior_selector_and_scalar_facts():
  producer, cache, output = values()
  wrapper = cached(producer, cache)
  unknown = I.Expr(producer.type, "test.unknown_effect")
  kernel = run([I.Assign("enabled", I.literal(False)), I.Allocate(cache), I.BindTile(unknown),
                I.Assign("after", local()), I.Store(output, wrapper)], cleanup=False)
  assert kernel.body.statements[-2].value.op == "local"
  assert kernel.body.statements[-1].value is wrapper
  assert kernel.body.statements[2].value is unknown


@pytest.mark.parametrize("barrier", ["binding", "store_address", "control", "unknown_statement", "loop"])
@pytest.mark.parametrize("iterations", [1, 4])
def test_cache_owner_survives_false_fact_lost_before_consumer(barrier, iterations):
  producer, cache, output = values()
  wrapper = cached(producer, cache)
  unknown = I.Expr(producer.type, "test.unknown_effect")

  @dataclass
  class UnknownStatement(I.Node):
    pass

  boundary = (
    I.BindTile(unknown) if barrier == "binding" else
    I.Store(replace(output, offset=I.Expr(I.ScalarType("int64"), "test.unknown_address")), producer)
    if barrier == "store_address" else I.Control("continue") if barrier == "control" else
    UnknownStatement() if barrier == "unknown_statement" else
    I.For("i", I.literal(0), I.literal(2), I.literal(1), I.Block([I.BindTile(unknown)]))
  )
  kernel = run([I.Assign("enabled", I.literal(False)), setup(producer, cache), boundary, I.Store(output, wrapper)],
               iterations=iterations)
  assert kernel.body.statements[-1].value is wrapper
  assert any(isinstance(node, I.Allocate) and node.view is cache for node in nodes(kernel.body))
  assert any(isinstance(node, I.If) and node.condition.op == "local" for node in kernel.body.statements)


def test_public_bounded_conversion_dot_has_no_dangling_cache_owner():
  M, N, K = T.dynamic(32, 32, 128), T.dynamic(32, 32, 128), T.dynamic()

  @T.template
  def product(A: T.Tensor((M, K), dtype="float16"), B: T.Tensor((N, K), dtype="float16"),
              C: T.Tensor((M, N), dtype="float32", access="output")):
    for row in T.serial(0, M, 32, unroll=1):
      for column in T.serial(0, N, 32, unroll=1):
        end_m, end_n = T.min(row + 32, M), T.min(column + 32, N)
        a = T.cast(T.load(A[row:end_m, :]), "bfloat16")
        b = T.transpose(T.cast(T.load(B[column:end_n, :]), "bfloat16"))
        T.store(C[row:end_m, column:end_n], T.dot(a, b, acc_dtype="float32"))

  lowered = T.lower(product, target="Native")
  allocated = {node.view.storage for node in nodes(lowered.kernel_ir.body) if isinstance(node, I.Allocate)}
  for node in nodes(lowered.kernel_ir.body):
    if isinstance(node, I.Expr) and node.op == "cached_value":
      assert node.data[0].storage in allocated
