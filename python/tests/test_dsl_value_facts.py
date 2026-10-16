# SPDX-FileCopyrightText: 2026 HenryRenYz and vecops contributors
# SPDX-License-Identifier: MIT
"""Program-point facts preserve loop semantics, traps and historical reads."""

from dataclasses import dataclass, replace

import pytest
import vecops.language as T
from vecops.language._compiler import ir as I
from vecops.language._compiler.analysis.facts import IntegerFacts, ScalarAliases, ValueFacts, induction_facts
from vecops.language._compiler.environment import freeze
from vecops.language._compiler.operations import OpDef, snapshot, use_catalog
from vecops.language._compiler.operations.registry import OperationCatalog
from vecops.language._compiler.passes.kernel.facts import propagate_value_facts
from vecops.language._compiler.passes.manager import PassContext


def local(name, dtype="int64"):
  return I.Expr(I.ScalarType(dtype), "local", data=name)


def binary(op, left, right, dtype=None):
  return I.Expr(I.ScalarType(dtype or ("bool" if op in {"eq", "lt", "ge"} else left.type.dtype)), op, (left, right))


def run(statements, *, catalog=None):
  kernel = I.KernelIR(I.Block(statements), {}, {})
  environment = freeze()
  if catalog is not None:
    environment = replace(environment, operations=catalog)
  context = PassContext("kernel-cleanup", {"_compiler_environment": environment})
  with use_catalog(environment.operations):
    result = propagate_value_facts(kernel, context)
  return kernel.body.statements, result, context


def loop(body, start=0, stop=64, step=32):
  return I.For("i", I.literal(start), I.literal(stop), I.literal(step), I.Block(body))


def test_same_dag_is_evaluated_at_each_program_point_and_converges():
  expression = binary("add", local("x"), I.literal(1))
  statements, result, context = run([
    I.Assign("x", I.literal(2)), I.Assign("first", expression),
    I.Assign("x", I.literal(9), declare=False), I.Assign("second", expression),
  ])
  assert result.changed
  assert [I.constant(statements[i].value) for i in (1, 3)] == [3, 10]
  with use_catalog(context.environment.operations):
    assert not propagate_value_facts(result.ir, context).changed


def test_fact_queries_include_missing_bindings_and_the_reference_namespace():
  facts = ValueFacts()
  for operation in ("local", "frozen", "raw", "symbol"):
    facts.infer(I.Expr(I.ScalarType("int64"), operation, data="x"))
  assert facts.queried == {(operation, "x") for operation in ("local", "frozen", "raw", "symbol")}


def test_unrelated_new_ssa_declaration_preserves_shared_dag_memo():
  expression = binary("add", local("x"), I.literal(1))
  statements, _, _ = run([
    I.Assign("x", I.literal(3)), I.Assign("first", expression),
    I.Assign("unrelated", I.literal(17)), I.Assign("second", expression),
  ])
  assert I.constant(statements[1].value) == 4
  assert statements[1].value is statements[3].value


@pytest.mark.parametrize("reference", ["local", "frozen"])
def test_new_fact_invalidates_cached_missing_binding_and_failed_overflow_proof(reference):
  read = I.Expr(I.ScalarType("int64"), "load", data=I.View("A", "int64", (), (), I.literal(0)))
  expression = binary("add", I.Expr(I.ScalarType("int64"), reference, data="x"), I.literal(1))
  statements, _, _ = run([
    I.Assign("x", read), I.Assign("first", expression),
    I.Assign("x", I.literal(3), declare=False), I.Assign("second", expression),
  ])
  assert statements[1].value is expression
  assert I.constant(statements[3].value) == 4


@pytest.mark.parametrize("reference", ["local", "frozen"])
def test_branch_forget_invalidates_both_mutable_reference_namespaces(reference):
  expression = binary("add", I.Expr(I.ScalarType("int64"), reference, data="x"), I.literal(1))
  branch = I.If(local("flag", "bool"), I.Block([I.Assign("x", I.literal(9), declare=False)]), I.Block([]))
  statements, _, _ = run([
    I.Assign("x", I.literal(3)), I.Assign("first", expression), branch, I.Assign("second", expression),
  ])
  assert I.constant(statements[1].value) == 4
  assert I.constant(statements[3].value) is None


def test_boolean_facts_and_branch_writes_are_lexical():
  flag = local("flag", "bool")
  branch = I.If(flag, I.Block([I.Assign("x", I.literal(7), declare=False)]), I.Block([]))
  statements, _, _ = run([
    I.Assign("flag", I.literal(True)), I.Assign("x", I.literal(3)), branch,
    I.Assign("after", local("x")), I.Assign("still", flag),
  ])
  assert I.constant(statements[2].condition) is True
  assert statements[3].value.op == "local"
  assert I.constant(statements[4].value) is True


@pytest.mark.parametrize("start,stop,step,lower,upper,divisor,residue", [
  (0, 64, 32, 0, 32, 32, 0), (3, 20, 4, 3, 19, 4, 3),
  (20, 3, -4, 4, 20, 4, 0), (7, 8, 32, 7, 7, 32, 7),
])
def test_induction_bounds_include_all_and_only_congruent_reachable_integers(start, stop, step, lower, upper, divisor, residue):
  fact = induction_facts(loop([], start, stop, step), ValueFacts())
  assert fact == IntegerFacts(lower, upper, divisor, residue)
  assert all(fact.lower <= i <= fact.upper and i % divisor == residue for i in range(start, stop, step))


@pytest.mark.parametrize("start,stop,step", [
  (0, 0, 1), (5, 0, 2), (0, 5, -2), (0, 5, 0),
  ((1 << 63) - 2, (1 << 63) - 1, 2), (-(1 << 63) + 1, -(1 << 63), -2),
])
def test_empty_zero_step_and_exit_increment_overflow_have_no_induction_fact(start, stop, step):
  assert induction_facts(loop([], start, stop, step), ValueFacts()) is None


@pytest.mark.parametrize("start,stop,step", [(0, 64, 32), (32, -1, -32)])
def test_loop_congruence_and_full_panel_width_fold_locally(start, stop, step):
  index = local("i")
  end = binary("min", binary("add", index, I.literal(32)), I.literal(64))
  width = binary("sub", end, index)
  statements, _, _ = run([
    loop([I.Assign("width", width), I.Assign("aligned", binary("mod", index, I.literal(32)))], start, stop, step),
    I.Assign("outside", binary("mod", index, I.literal(32))),
  ])
  body = statements[0].body.statements
  assert I.constant(body[0].value) == 32
  assert I.constant(body[1].value) == 0
  assert statements[1].value.op == "mod"


@pytest.mark.parametrize("mutation", ["stop", "step", "i", "carried"])
def test_loop_carried_writes_and_mutable_headers_are_not_frozen(mutation):
  target = local(mutation)
  observed = binary("mod", local("i"), I.literal(32))
  body = [I.Assign("before", observed), I.Assign(mutation, binary("add", target, I.literal(1)), declare=False)]
  current = loop(body)
  if mutation in {"stop", "step"}:
    current = replace(current, **{mutation: target})
  statements, _, _ = run([I.Assign(mutation, I.literal(32)), current, I.Assign("after", target)])
  if mutation != "carried":
    assert statements[1].body.statements[0].value.op == "mod"
  assert statements[1].body.statements[1].value.op == "add"
  assert statements[2].value.op == "local"


def test_while_condition_is_not_specialized_from_initial_carried_value():
  condition = binary("lt", local("x"), I.literal(4))
  statements, _, _ = run([
    I.Assign("x", I.literal(0)),
    I.While(condition, I.Block([I.Assign("x", binary("add", local("x"), I.literal(1)), declare=False)])),
    I.Assign("after", local("x")),
  ])
  assert statements[1].condition is condition
  assert statements[2].value.op == "local"


@pytest.mark.parametrize("dtype,constant", [("int8", 127), ("uint8", 255), ("int64", (1 << 63) - 1)])
def test_cancellation_cannot_hide_intermediate_integer_overflow(dtype, constant):
  x = local("x", dtype)
  plus = binary("add", x, I.literal(1, dtype))
  expression = binary("sub", plus, x)
  statements, _, _ = run([I.Assign("x", I.literal(constant, dtype)), I.Assign("result", expression)])
  assert statements[1].value.op == "sub"
  assert statements[1].value.args[0].op == "add"


@pytest.mark.parametrize("dtype", ["int8", "int64"])
def test_safe_narrow_integer_cancellation_preserves_type(dtype):
  x = local("x", dtype)
  expression = binary("sub", binary("add", x, I.literal(1, dtype)), x)
  statements, _, _ = run([I.Assign("x", I.literal(3, dtype)), I.Assign("result", expression)])
  assert I.constant(statements[1].value) == 1
  assert statements[1].value.type == expression.type


@pytest.mark.parametrize("dtype", ["int8", "uint8", "int32", "uint32", "int64", "bool"])
@pytest.mark.parametrize("operation", ["local", "frozen"])
def test_unknown_auto_local_has_no_nominal_type_range(dtype, operation):
  facts = ValueFacts()
  value = I.Expr(I.ScalarType(dtype), operation, data="unknown")
  assert facts.infer(value) is None
  assert (operation, "unknown") in facts.queried
  assert facts.infer(I.Expr(value.type, "raw", data="abi_parameter")) is not None


@pytest.mark.parametrize("operation", ["ge", "le", "eq", "ne", "add", "sub", "mul"])
def test_unsigned_and_mixed_scalar_expressions_keep_original_operands(operation):
  unsigned = local("u", "uint64")
  expression = I.binary(operation, unsigned, I.literal(-1))
  assert ValueFacts().infer(expression) is None
  statements, _, _ = run([I.Assign("u", I.literal(0, "uint64")), I.Assign("result", expression)])
  assert statements[1].value is expression
  assert statements[1].value.args[0] is unsigned


@pytest.mark.parametrize("reference", ["local", "frozen"])
def test_promoted_local_does_not_inherit_narrow_range_and_later_binding_invalidates_unknown(reference):
  value = I.binary("add", I.Expr(I.ScalarType("int8"), "raw", data="x"), I.literal(1))
  local_value = I.Expr(I.ScalarType("int8"), reference, data="y")
  comparison = I.binary("le", local_value, I.literal(127))
  statements, _, _ = run([
    I.Assign("y", value), I.Assign("before", comparison),
    I.Assign("y", I.literal(3, "int8"), declare=False), I.Assign("after", comparison),
  ])
  assert statements[1].value is comparison, "auto y executes in int64 and can equal 128"
  assert I.constant(statements[3].value) is True, "a newly established binding must invalidate the missing proof"


def test_public_scalar_frontend_comparisons_retain_existing_cpp_promotions():
  from vecops.language._compiler.frontend.parser import lower

  @T.template
  def unsigned(u: T.uint64, C: T.Tensor((1,), dtype="int64", access="output")):
    if u >= -1:
      T.store(C[0], 1)
    else:
      T.store(C[0], 0)

  @T.template
  def promoted(x: T.int8, C: T.Tensor((1,), dtype="int64", access="output")):
    y = x + 1
    if y <= 127:
      T.store(C[0], 1)
    else:
      T.store(C[0], 0)

  for template in (unsigned, promoted):
    environment = freeze()
    program = lower(template.ir, catalog=environment.operations)
    branch = program.body.statements[-1]
    original = branch.condition
    context = PassContext("kernel-cleanup", {"_compiler_environment": environment})
    with use_catalog(environment.operations):
      propagate_value_facts(program, context)
    assert program.body.statements[-1].condition is original


@pytest.mark.parametrize("payload", ["load", "broadcast_dim"])
def test_owned_data_read_or_trap_cannot_be_canceled(payload):
  definitions = dict(snapshot().definitions)
  definitions["owned_metadata"] = OpDef("owned_metadata", "metadata", may_trap=False, speculatable=True)
  catalog = OperationCatalog(definitions, "facts-owned-data")
  data = (I.Expr(I.ScalarType("int64"), "load", data=I.View("A", "int64", (), (), I.literal(0)))
          if payload == "load" else I.Expr(I.ScalarType("int64"), "broadcast_dim", (I.literal(2), I.literal(3))))
  value = I.Expr(I.ScalarType("int64"), "owned_metadata", data=data)
  expression = I.binary("sub", value, value)
  statements, _, _ = run([I.Assign("result", expression)], catalog=catalog)
  assert statements[0].value is expression


@pytest.mark.parametrize("op", ["floordiv", "mod", "broadcast_dim", "unknown"])
def test_constant_range_does_not_remove_traps_or_unknown_effects(op):
  dangerous = I.Expr(I.ScalarType("int64"), op, (I.literal(1), I.literal(0)))
  expression = binary("min", dangerous, I.literal(-(1 << 63)))
  statements, _, _ = run([I.Assign("result", expression)])
  assert statements[0].value is expression


def test_historical_load_and_state_identity_survive_changing_index_facts():
  index = local("x")
  view = I.View("A", "int64", (index,), (I.literal(1),), index)
  read = I.Expr(I.ScalarType("int64"), "load", (index,), view)
  state = I.Expr(I.ScalarType("int64"), "state_ref", data="s")
  tile = I.Expr(I.TileType("int64", (index,)), "fill", (index,))
  statements, _, _ = run([
    I.Assign("x", I.literal(1)), I.BindTile(read), I.BindTile(tile),
    I.Assign("x", I.literal(2), declare=False), I.Assign("read", read), I.Assign("state", state),
  ])
  assert statements[1].value is statements[4].value is read
  assert statements[2].value is tile
  assert read.args[0] is view.offset is index
  assert statements[5].value is state


@pytest.mark.parametrize("unknown", [False, True])
def test_unknown_control_and_expression_kill_facts_across_iterations(unknown):
  @dataclass(frozen=True)
  class Opaque(I.Node):
    pass

  barrier = I.Assign("opaque", I.Expr(I.ScalarType("int64"), "opaque_mutation")) if unknown else Opaque()
  modulo = binary("mod", local("i"), I.literal(32))
  statements, _, _ = run([
    I.Assign("x", I.literal(7)),
    loop([I.Assign("before", modulo), barrier, I.Assign("after", local("x"))]),
    I.Assign("outside", local("x")),
  ])
  assert statements[1].body.statements[0].value.op == "mod"
  assert statements[1].body.statements[2].value.op == "local"
  assert statements[2].value.op == "local"


@pytest.mark.parametrize("kind", ["condition", "header", "assert", "sibling", "raw"])
def test_opaque_evaluation_never_leaves_stale_sibling_or_continuation_facts(kind):
  effect = I.Expr(I.ScalarType("int64"), "opaque_mutation")
  if kind == "condition":
    statement = I.If(effect, I.Block([I.Assign("inside", local("x"))]), I.Block([]))
  elif kind == "header":
    statement = replace(loop([I.Assign("inside", local("x"))]), start=effect)
  elif kind == "assert":
    statement = I.Assert(effect, "opaque predicate")
  else:
    if kind == "raw":
      effect = I.Expr(I.ScalarType("int64"), "raw", data="mutate_x()")
    statement = I.Assign("result", binary("add", effect, local("x")))
  statements, _, _ = run([I.Assign("x", I.literal(7)), statement, I.Assign("outside", local("x"))])
  assert statements[2].value.op == "local"
  if kind in {"condition", "header"}:
    assert statements[1].body.statements[0].value.op == "local"
  elif kind in {"sibling", "raw"}:
    assert statements[1].value.args[1].op == "local"


@pytest.mark.parametrize("metadata", ["address", "shape"])
def test_opaque_owned_metadata_kills_facts_without_cloning_the_read(metadata):
  effect = I.Expr(I.ScalarType("int64"), "opaque_mutation")
  view = I.View("A", "int64", (effect if metadata == "shape" else I.literal(1),),
                (I.literal(1),), effect if metadata == "address" else I.literal(0))
  read = I.Expr(I.ScalarType("int64"), "load", data=view)
  statements, _, _ = run([I.Assign("x", I.literal(7)), I.Assign("read", read), I.Assign("after", local("x"))])
  assert statements[1].value is read
  assert statements[2].value.op == "local"


def test_opaque_store_address_cannot_specialize_its_value():
  effect = I.Expr(I.ScalarType("int64"), "opaque_mutation")
  destination = I.View("C", "int64", (), (), effect)
  store = I.Store(destination, local("x"))
  statements, _, _ = run([I.Assign("x", I.literal(7)), store, I.Assign("after", local("x"))])
  assert statements[1] is store
  assert statements[2].value.op == "local"


def test_immutable_loop_local_definition_preserves_its_safe_relation():
  end = binary("min", binary("add", local("i"), I.literal(32)), I.literal(64))
  statements, _, _ = run([loop([
    I.Assign("end", end), I.Assign("width", binary("sub", local("end"), local("i"))),
  ])])
  assert I.constant(statements[0].body.statements[1].value) == 32


def test_alias_versions_kill_transitive_recomputation_and_keep_missing_queries():
  aliases = ScalarAliases()
  assert aliases.lookup(local("end")) is None
  assert ("local", "end") in aliases.queried
  aliases.write({"i"})
  end = binary("add", local("i"), I.literal(1))
  aliases.define("end", end, aliases.dependencies(end))
  capture = binary("add", local("end"), I.literal(2))
  aliases.define("capture", capture, aliases.dependencies(capture))
  child = aliases.copy()
  assert child.lookup(local("end")) is end
  changed = child.write({"i"})
  assert changed == {(op, name) for op in ("local", "frozen") for name in ("end", "capture")}
  assert child.lookup(local("capture")) is None
  assert aliases.lookup(local("end")) is end


@pytest.mark.parametrize("where", ["straight", "branch", "while"])
def test_dependency_write_never_recomputes_an_immutable_snapshot(where):
  observed = binary("sub", local("end"), local("x"))
  update = I.Assign("x", binary("add", local("i"), I.literal(2)), declare=False)
  if where == "branch":
    update = I.If(local("flag", "bool"), I.Block([update]), I.Block([]))
  elif where == "while":
    update = I.While(local("flag", "bool"), I.Block([update]))
  statements, _, _ = run([loop([
    I.Assign("x", local("i")), I.Assign("end", binary("add", local("x"), I.literal(1))),
    I.Assign("before", observed), update, I.Assign("after", observed),
  ])])
  body = statements[0].body.statements
  assert I.constant(body[2].value) == 1
  assert I.constant(body[4].value) is None


def test_single_declaration_does_not_mean_immutable_when_reassigned():
  statements, _, _ = run([loop([
    I.Assign("end", binary("add", local("i"), I.literal(1))),
    I.Assign("before", binary("sub", local("end"), local("i"))),
    I.Assign("end", binary("add", local("i"), I.literal(2)), declare=False),
  ])])
  assert I.constant(statements[0].body.statements[1].value) is None


def test_branch_local_definition_does_not_escape_its_declaration_scope():
  statements, _, _ = run([loop([
    I.If(local("flag", "bool"), I.Block([I.Assign("end", binary("add", local("i"), I.literal(1)))]), I.Block([])),
    I.Assign("after", binary("sub", local("end"), local("i"))),
  ])])
  assert I.constant(statements[0].body.statements[1].value) is None


def test_aliases_do_not_cross_loop_carried_or_explicit_induction_writes():
  statements, _, _ = run([loop([
    I.Assign("end", binary("add", local("i"), I.literal(1))),
    I.Assign("before", binary("sub", local("end"), local("i"))),
    I.Assign("i", binary("add", local("i"), I.literal(1)), declare=False),
  ])])
  assert I.constant(statements[0].body.statements[1].value) is None


@pytest.mark.parametrize("dtype", ["int8", "int32", "uint64", "float32"])
def test_symbolic_alias_domain_excludes_narrow_unsigned_and_float_values(dtype):
  index = local("i", dtype)
  expression = binary("add", index, I.literal(1, dtype))
  statements, _, _ = run([loop([
    I.Assign("end", expression), I.Assign("width", binary("sub", local("end", dtype), index)),
  ])])
  assert I.constant(statements[0].body.statements[1].value) is None


def test_aliases_preserve_historical_read_and_owned_metadata_identity():
  index = local("end")
  view = I.View("A", "int64", (index,), (I.literal(1),), index)
  read = I.Expr(I.ScalarType("int64"), "load", data=view)
  statements, _, _ = run([loop([
    I.Assign("end", binary("add", local("i"), I.literal(1))), I.BindTile(read), I.Assign("copy", read),
  ])])
  body = statements[0].body.statements
  assert body[1].value is body[2].value is read
  assert read.data is view and view.offset is index


def test_owned_statement_barrier_kills_scalar_and_symbolic_facts():
  definitions = dict(snapshot().definitions)
  definitions["wrapper"] = OpDef("wrapper", "metadata", may_trap=False, speculatable=True)
  catalog = OperationCatalog(definitions, "alias-owned-statement")
  effect = I.Expr(I.ScalarType("int64"), "wrapper", data=I.Block([I.Assign("x", I.literal(9), declare=False)]))
  statements, _, _ = run([
    I.Assign("x", I.literal(3)), I.Assign("opaque", effect), I.Assign("after", local("x")),
  ], catalog=catalog)
  assert statements[1].value is effect
  assert statements[2].value.op == "local"


def test_alias_expansion_budget_counts_shared_nodes_instead_of_tree_paths():
  value = local("i")
  for _ in range(62):
    value = binary("add", value, value)
  assert ScalarAliases.bounded(value)
  value = binary("add", value, value)
  assert ScalarAliases.bounded(value)
  assert not ScalarAliases.bounded(binary("add", value, value))


def test_alias_dependency_budget_bounds_long_single_node_copy_chains(monkeypatch):
  monkeypatch.setattr(ScalarAliases, "max_dependencies", 4)
  definitions = {}
  define = ScalarAliases.define

  def record(self, name, value, dependencies):
    definitions[name] = (value, dependencies)
    return define(self, name, value, dependencies)

  monkeypatch.setattr(ScalarAliases, "define", record)
  assignments = []
  previous = local("i")
  for index in range(24):
    name = f"copy_{index}"
    assignments.append(I.Assign(name, previous))
    previous = local(name)
  statements, _, _ = run([loop(assignments)])
  assert definitions and len(definitions) < len(assignments)
  assert max(len(dependencies) for _, dependencies in definitions.values()) == 4
  assert all(value.op == "local" for value, _ in definitions.values())
  assert len(statements[0].body.statements) == len(assignments)


def test_alias_rewrites_reach_a_stable_fixed_point():
  statements, result, context = run([loop([
    I.Assign("end", binary("add", local("i"), I.literal(32))),
    I.Assign("width", binary("sub", local("end"), local("i"))),
  ])])
  assert I.constant(statements[0].body.statements[1].value) == 32
  with use_catalog(context.environment.operations):
    assert not propagate_value_facts(result.ir, context).changed


def test_new_unqueried_alias_preserves_existing_shared_dag_memo():
  observed = binary("sub", local("end"), local("i"))
  statements, _, _ = run([loop([
    I.Assign("end", binary("add", local("i"), I.literal(1))), I.Assign("first", observed),
    I.Assign("unrelated", binary("add", local("i"), I.literal(2))), I.Assign("second", observed),
  ])])
  body = statements[0].body.statements
  assert I.constant(body[1].value) == 1
  assert body[1].value is body[3].value


@pytest.mark.parametrize("limit,expected", [(2, None), (3, 1)])
def test_alias_node_budget_falls_back_without_inventing_the_relation(monkeypatch, limit, expected):
  monkeypatch.setattr(ScalarAliases, "max_nodes", limit)
  statements, _, _ = run([loop([
    I.Assign("end", binary("add", local("i"), I.literal(1))),
    I.Assign("width", binary("sub", local("end"), local("i"))),
  ])])
  assert I.constant(statements[0].body.statements[1].value) == expected


def test_chained_aliases_retain_all_transitive_dependency_versions():
  observed = binary("sub", local("last"), local("x"))
  statements, _, _ = run([loop([
    I.Assign("x", local("i")), I.Assign("first", binary("add", local("x"), I.literal(1))),
    I.Assign("last", binary("add", local("first"), I.literal(2))), I.Assign("before", observed),
    I.Assign("x", binary("add", local("i"), I.literal(4)), declare=False), I.Assign("after", observed),
  ])])
  body = statements[0].body.statements
  assert I.constant(body[3].value) == 3
  assert I.constant(body[5].value) is None


@pytest.mark.parametrize("start,stop,step,expected", [(32, -1, -32, 32), (0, 0, 32, None), (64, 0, 32, None)])
def test_aliases_use_only_proven_reachable_loop_domains(start, stop, step, expected):
  statements, _, _ = run([loop([
    I.Assign("end", binary("add", local("i"), I.literal(32))),
    I.Assign("width", binary("sub", local("end"), local("i"))),
  ], start, stop, step)])
  assert I.constant(statements[0].body.statements[1].value) == expected


@pytest.mark.parametrize("operation", ["floordiv", "mod", "streaming_lanes"])
def test_alias_definitions_cannot_hide_traps_or_hardware_queries(operation):
  args = () if operation == "streaming_lanes" else (I.literal(1), I.literal(0))
  value = I.Expr(I.ScalarType("int64"), operation, args, "float32" if not args else None)
  statements, _, _ = run([I.Assign("end", value), I.Assign("after", binary("sub", local("end"), local("end")))])
  assert statements[0].value is value
  assert I.constant(statements[1].value) is None


@pytest.mark.parametrize("operation", ["raw", "symbol"])
def test_aliases_do_not_recompute_unvalidated_raw_geometry(monkeypatch, operation):
  definitions = []
  define = ScalarAliases.define

  def record(self, name, value, dependencies):
    definitions.append(name)
    return define(self, name, value, dependencies)

  monkeypatch.setattr(ScalarAliases, "define", record)
  geometry = I.Expr(I.ScalarType("int64"), operation, data="runtime_lanes")
  statements, _, _ = run([I.Assign("captured", geometry), I.Assign("use", local("captured"))])
  assert "captured" not in definitions
  assert statements[0].value is geometry
  assert statements[1].value.op == "local"


def test_unrolled_wrapper_does_not_invent_counter_ranges_or_export_writes():
  modulo = binary("mod", local("i"), I.literal(32))
  wrapped = I.UnrolledFor(loop([I.Assign("tail", modulo)]), I.Block([I.Assign("x", I.literal(9))]))
  statements, _, _ = run([I.Assign("x", I.literal(3)), wrapped, I.Assign("after", local("x"))])
  assert statements[1].loop.body.statements[0].value is modulo
  assert statements[2].value.op == "local"


def test_uint64_bound_uses_no_signed_counter_fact():
  current = replace(loop([]), stop=I.literal(64, "uint64"))
  assert induction_facts(current, ValueFacts()) is None


def test_custom_catalog_remains_the_authority_for_purity():
  definitions = dict(snapshot().definitions)
  definitions["sub"] = replace(definitions["sub"], effects="unknown")
  catalog = OperationCatalog(definitions, "facts-unknown-sub")
  expression = binary("sub", local("x"), local("x"))
  statements, _, _ = run([I.Assign("x", I.literal(3)), I.Assign("result", expression)], catalog=catalog)
  assert statements[1].value is expression


def test_custom_pure_but_trapping_operation_cannot_be_canceled():
  definitions = dict(snapshot().definitions)
  definitions["checked"] = OpDef("checked", "metadata", effects="none", may_trap=True)
  catalog = OperationCatalog(definitions, "facts-trapping")
  checked = I.Expr(I.ScalarType("int64"), "checked", (local("x"),))
  expression = binary("sub", checked, checked)
  statements, _, _ = run([I.Assign("x", I.literal(3)), I.Assign("result", expression)], catalog=catalog)
  assert statements[1].value.op == "sub"
  assert statements[1].value.args[0].op == "checked"


def test_default_recipe_cleanup_consumes_loop_local_facts_in_source_only_lowering():
  from vecops.language._compiler.backends.cpu.backend import CPUBackend
  from vecops.language._compiler.ir.traversal import nodes
  from vecops.language._compiler.pipeline import kernel_pipeline

  manager = kernel_pipeline()
  cleanup = next(step for step in manager.steps if getattr(step, "name", None) == "recipe-cleanup")
  names = [step.spec.name for step in cleanup.steps]
  assert names.index("propagate-value-facts") < names.index("simplify-ir")

  @T.template
  def panels(C: T.Tensor((2,), dtype="int64", access="output")):
    for begin in T.serial(0, 64, 32, unroll=1):
      if T.min(begin + 32, 64) - begin == 32 and begin % 32 == 0:
        T.store(C[begin // 32], 1)
      else:
        T.store(C[begin // 32], 0)

  artifact = CPUBackend().lower(panels.ir, (), target="generic")
  assert not any(isinstance(node, I.If) for node in nodes(artifact.kernel.body)), artifact.kernel.body
  stores = [node for node in nodes(artifact.kernel.body) if isinstance(node, I.Store)]
  assert stores and all(I.constant(store.value) == 1 for store in stores)


def test_default_recipe_cleanup_consumes_captured_full_span_equality():
  from vecops.language._compiler.backends.cpu.backend import CPUBackend
  from vecops.language._compiler.ir.traversal import nodes

  @T.template
  def panels(C: T.Tensor((2,), dtype="int64", access="output")):
    for begin in T.serial(0, 64, 32, unroll=1):
      end = T.min(begin + 32, 64)
      span = end - begin
      if span == 32:
        T.store(C[begin // 32], 1)
      else:
        T.store(C[begin // 32], -99)

  artifact = CPUBackend().lower(panels.ir, (), target="generic")
  owned = list(nodes(artifact.kernel.body))
  assert not any(isinstance(node, I.If) for node in owned)
  stores = [node for node in owned if isinstance(node, I.Store)]
  assert stores and all(I.constant(store.value) == 1 for store in stores)
