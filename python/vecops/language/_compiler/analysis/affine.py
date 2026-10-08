# SPDX-FileCopyrightText: 2026 HenryRenYz and vecops contributors
# SPDX-License-Identifier: MIT
"""Shared integer affine decomposition and conservative loop intervals.

Preserve cancellation before taking intervals (i + width - i is width).
Unknown nonlinear expressions remain unproved, never optimistic bounds.
"""

from dataclasses import fields, replace

from .. import ir as I


def positive_extent(value, catalog, names=frozenset()):
  """Strictly positive metadata, including products of native extents.

  Unknown parameters and subtraction/division remain unproved. As with all
  shape arithmetic, callers must separately validate representable extents.
  """
  from ..operations import Trait

  constant = I.constant(value)
  return (
    type(constant) is int
    and constant > 0
    or catalog.has(value, Trait.POSITIVE_EXTENT)
    or value.op in {"local", "frozen"}
    and value.data in names
    or value.op == "mul"
    and all(positive_extent(v, catalog, names) for v in value.args)
  )


def expand_definitions(value, definitions, seen=frozenset()):
  if value.op in {"local", "frozen"} and value.data in definitions and value.data not in seen:
    return expand_definitions(definitions[value.data], definitions, seen | {value.data})
  return replace(value, args=tuple(expand_definitions(a, definitions, seen) for a in value.args))


def affine(value, variables, definitions, seen=frozenset()):
  """Return integer variable coefficients and an invariant remainder."""
  if value.op in {"local", "frozen"} and value.data in definitions and value.data not in seen:
    return affine(definitions[value.data], variables, definitions, seen | {value.data})
  if value.op == "local" and value.data in variables:
    return {value.data: 1}, I.literal(0)
  if value.op in {"add", "sub"}:
    left, right = (affine(a, variables, definitions, seen) for a in value.args)
    if left is None or right is None:
      return None
    sign = 1 if value.op == "add" else -1
    coefficients = dict(left[0])
    for name, factor in right[0].items():
      coefficients[name] = coefficients.get(name, 0) + sign * factor
    return {k: v for k, v in coefficients.items() if v}, I.binary(value.op, left[1], right[1])
  if value.op == "mul":
    for constant, operand in (value.args, value.args[::-1]):
      factor = I.constant(constant)
      if type(factor) is int:
        result = affine(operand, variables, definitions, seen)
        if result is not None:
          return {k: v * factor for k, v in result[0].items()}, I.binary("mul", result[1], constant)
  if affine_contains(value, variables, definitions, seen):
    return None
  return {}, expand_definitions(value, definitions, seen)


def affine_contains(value, variables, definitions, seen=frozenset()):
  if value.op in {"local", "frozen"}:
    if value.data in variables:
      return True
    if value.data in definitions and value.data not in seen:
      return affine_contains(definitions[value.data], variables, definitions, seen | {value.data})
  if any(affine_contains(a, variables, definitions, seen) for a in value.args):
    return True
  from ..ir.traversal import nodes

  return any(isinstance(v, I.Expr) and affine_contains(v, variables, definitions, seen) for v in nodes(value.data))


def interval(value, bounds):
  """Inclusive bounds in loop-invariant metadata, or None if unproved."""
  value = canonical_integer(simplify_ordered(value, bounds))
  # Push monotone affine arithmetic through min/max before bounding. This
  # retains relationships such as i + min(width, M - i) <= M; separately
  # bounding the two operands would lose that cancellation at clipped tails.
  if value.op in {"add", "sub"}:
    for index, operand in enumerate(value.args):
      if operand.op not in {"min", "max"}:
        continue
      operation = operand.op
      if value.op == "sub" and index == 1:
        operation = "max" if operation == "min" else "min"
      alternatives = []
      for arm in operand.args:
        arguments = list(value.args)
        arguments[index] = arm
        alternatives.append(interval(I.binary(value.op, *arguments), bounds))
      if any(v is None for v in alternatives):
        return None
      return tuple(I.binary(operation, alternatives[0][i], alternatives[1][i]) for i in (0, 1))
  if value.op == "mul":
    for literal, operand in (value.args, value.args[::-1]):
      factor = I.constant(literal)
      if type(factor) is not int or operand.op not in {"min", "max"}:
        continue
      operation = operand.op if factor >= 0 else "max" if operand.op == "min" else "min"
      alternatives = [interval(I.binary("mul", literal, arm), bounds) for arm in operand.args]
      if any(v is None for v in alternatives):
        return None
      return tuple(I.binary(operation, alternatives[0][i], alternatives[1][i]) for i in (0, 1))
  # Eliminate innermost induction variables first. Their endpoints may depend
  # on enclosing indices; substituting all endpoints at once is not sound
  # interval arithmetic for correlated nested domains.
  for name in reversed(bounds):
    linear = affine(value, {name}, {})
    if linear is None or name not in linear[0]:
      continue
    coefficient = linear[0][name]
    rest = {key: pair for key, pair in bounds.items() if key != name}
    endpoints = bounds[name] if coefficient >= 0 else bounds[name][::-1]
    lo = interval(I.binary("add", linear[1], I.binary("mul", I.literal(coefficient), endpoints[0])), rest)
    hi = interval(I.binary("add", linear[1], I.binary("mul", I.literal(coefficient), endpoints[1])), rest)
    return None if lo is None or hi is None else (lo[0], hi[1])
  linear = affine(value, bounds, {})
  if linear is not None:
    coefficients, lower = linear
    upper = lower
    for name, factor in coefficients.items():
      lo, hi = bounds[name]
      if factor < 0:
        lo, hi = hi, lo
      lower = I.binary("add", lower, I.binary("mul", I.literal(factor), lo))
      upper = I.binary("add", upper, I.binary("mul", I.literal(factor), hi))
    return lower, upper
  operands = [interval(a, bounds) for a in value.args]
  if any(v is None for v in operands):
    return None
  if value.op in {"min", "max"}:
    return tuple(I.binary(value.op, operands[0][i], operands[1][i]) for i in (0, 1))
  if value.op in {"add", "sub"}:
    a, b = operands
    return (I.binary(value.op, a[0], b[value.op == "sub"]), I.binary(value.op, a[1], b[value.op != "sub"]))
  if value.op == "floordiv" and type(I.constant(value.args[1])) is int and I.constant(value.args[1]) > 0:
    return tuple(I.binary("floordiv", v, value.args[1]) for v in operands[0])
  if value.op == "mul":
    for index, literal in enumerate(value.args):
      factor = I.constant(literal)
      if type(factor) is int:
        endpoints = operands[1 - index] if factor >= 0 else operands[1 - index][::-1]
        return tuple(I.binary("mul", literal, endpoint) for endpoint in endpoints)
  return None


def simplify_ordered(value, bounds):
  """Remove metadata clamps already implied by an induction domain.

  Do this before separating a subtraction into independent intervals. The
  exact clipped origin max(0, min(i, extent)) is i when 0 <= i < extent;
  forgetting that equality loses the correlation with the tail length.
  Unknown/nonlinear orders stay untouched.
  """
  value = replace(value, args=tuple(simplify_ordered(a, bounds) for a in value.args))
  if value.op not in {"min", "max"} or not bounds:
    return value
  endpoints = []
  for upper in (False, True):
    endpoint = I.binary("sub", *value.args)
    for name in reversed(bounds):
      linear = affine(endpoint, {name}, {})
      if linear is None:
        break
      factor = linear[0].get(name, 0)
      endpoint = canonical_integer(
        I.binary("add", linear[1], I.binary("mul", I.literal(factor), bounds[name][upper != (factor < 0)]))
      )
    endpoints.append(I.constant(endpoint))
  lower, upper = endpoints
  if type(upper) is int and upper <= 0:
    return value.args[value.op == "max"]
  if type(lower) is int and lower >= 0:
    return value.args[value.op == "min"]
  return value


def canonical_integer(value):
  """Normalize pure integer metadata sums without touching floating DAGs."""
  if not isinstance(value.type, I.ScalarType) or not value.type.dtype.startswith("int"):
    return value
  arguments = tuple(canonical_integer(a) for a in value.args)
  value = replace(value, args=arguments)
  if value.op == "sub" and value.args[0].op in {"min", "max"}:
    first, second = value.args
    for index, operand in enumerate(first.args):
      if I.same_expr(operand, second):
        return I.binary(first.op, I.literal(0), canonical_integer(I.binary("sub", first.args[1 - index], second)))
  if value.op not in {"add", "sub", "mul"}:
    return value
  constants, terms = 0, {}

  def key(expression):
    if expression.op in {"raw", "symbol", "local", "frozen"}:
      return expression.op, expression.data, expression.type.dtype
    if expression.op in {"literal", "min", "max", "streaming_lanes"}:
      return expression.op, expression.data, expression.type.dtype, tuple(key(a) for a in expression.args)
    return id(expression)

  def collect(expression, factor):
    nonlocal constants
    constant = I.constant(expression)
    if type(constant) is int:
      constants += factor * constant
    elif expression.op in {"add", "sub"}:
      collect(expression.args[0], factor)
      collect(expression.args[1], factor if expression.op == "add" else -factor)
    elif expression.op == "mul" and any(type(I.constant(a)) is int for a in expression.args):
      literal, operand = expression.args if type(I.constant(expression.args[0])) is int else expression.args[::-1]
      collect(operand, factor * I.constant(literal))
    else:
      identity = key(expression)
      previous, coefficient = terms.get(identity, (expression, 0))
      terms[identity] = previous, coefficient + factor

  collect(value, 1)
  result = I.literal(constants)
  # Distributing several independent clamps builds a Cartesian product of
  # arms, exponential in nested slicing depth. Retain exact arithmetic when
  # more than one clamp survives cancellation; one-clamp affine rewrites
  # still preserve the origin/tail relationships needed by interval proofs.
  distribute = sum(bool(factor) and expression.op in {"min", "max"} for expression, factor in terms.values()) <= 1
  for expression, factor in sorted(terms.values(), key=lambda term: term[0].op in {"min", "max"}):
    if factor:
      if distribute and expression.op in {"min", "max"}:
        operation = expression.op if factor > 0 else "max" if expression.op == "min" else "min"
        # Keep the affine terms inside each clamp arm before bounding.
        # (-1 - i) + min(i + width, N) is min(width - 1, N - i - 1),
        # not the sum of two independent ranges.
        result = I.binary(
          operation,
          *(
            canonical_integer(I.binary("add", result, I.binary("mul", I.literal(factor), arm)))
            for arm in expression.args
          ),
        )
      else:
        result = I.binary("add", result, I.binary("mul", I.literal(factor), expression))
  return result


def simplify_loop_metadata(block, bounds=None):
  """Simplify integer addresses in a validated immutable structured region.

  This is an induction-domain optimization, not an operator-name pattern.
  It retains SSA references, execution order, payload types and native length
  queries. No metadata definition is expanded into repeated runtime queries.
  Scopes have separate DAG memoization because an expression can be shared
  between loops with different induction domains.
  """
  bounds = {} if bounds is None else bounds
  memo = {}

  def visit(value):
    if isinstance(value, I.For):
      start, stop, step = (visit(v) for v in (value.start, value.stop, value.step))
      increment = I.constant(step)
      # Validated nonconstant steps are positive native extents. Conservative
      # endpoints are sufficient here; preflight retains exact trip intervals.
      endpoints = (
        (I.binary("add", stop, I.literal(1)), start)
        if increment is not None and increment < 0
        else (start, I.binary("sub", stop, I.literal(1)))
      )
      return replace(
        value,
        start=start,
        stop=stop,
        step=step,
        body=simplify_loop_metadata(value.body, {**bounds, value.target: endpoints}),
      )
    if isinstance(value, I.Block):
      return I.Block([visit(s) for s in value.statements])
    if isinstance(value, tuple):
      return tuple(visit(v) for v in value)
    if isinstance(value, I.Node):
      if id(value) not in memo:
        changed = replace(value, **{field.name: visit(getattr(value, field.name)) for field in fields(value)})
        memo[id(value)] = (
          canonical_integer(simplify_ordered(changed, bounds)) if isinstance(changed, I.Expr) else changed
        )
      return memo[id(value)]
    return value

  return visit(block)


def bind_instruction_metadata(block, *, prefix="instruction_metadata"):
  """Share pure metadata in lexical SSA instead of printing expression trees.

  Only use after validating an immutable, no-throw instruction program.
  Definitions never escape a branch/loop scope and loop-dependent addresses
  are evaluated inside that loop. View provenance/payload values are retained;
  only the integer metadata consumed by terminal printing is rewritten.
  """
  counter = 0
  identities, table, dependencies = {}, {}, {}

  def identity(value):
    if id(value) not in identities:
      operands = tuple(identity(v) for v in value.args)
      key = (value.type, value.op, operands, repr(value.data))
      index = table.setdefault(key, len(table))
      dependencies[index] = (frozenset((value.data,)) if value.op in {"local", "frozen"} else frozenset()).union(
        *(dependencies[v] for v in operands)
      )
      identities[id(value)] = index
    return identities[id(value)]

  def scope(block, inherited):
    known, output = dict(inherited), []

    def scalar(value, *, bind=True):
      nonlocal counter
      key = identity(value)
      if key in known:
        return known[key]
      changed = replace(value, args=tuple(scalar(v) for v in value.args))
      if bind and value.args:
        counter += 1
        name = f"{prefix}_{counter}"
        output.append(I.Assign(name, changed))
        changed = I.Expr(value.type, "local", data=name)
      known[key] = changed
      return changed

    def metadata(value):
      if isinstance(value, I.Expr):
        return scalar(value)
      if isinstance(value, I.View):
        return replace(
          value,
          offset=scalar(value.offset),
          shape=tuple(map(scalar, value.shape)),
          strides=tuple(map(scalar, value.strides)),
        )
      if isinstance(value, I.SMEVector):
        return replace(value, extent=scalar(value.extent))
      if isinstance(value, tuple):
        return tuple(map(metadata, value))
      return value

    for statement in block.statements:
      if isinstance(statement, I.For):
        start, stop, step = map(scalar, (statement.start, statement.stop, statement.step))
        available = {key: value for key, value in known.items() if statement.target not in dependencies[key]}
        changed = replace(statement, start=start, stop=stop, step=step, body=scope(statement.body, available))
      elif isinstance(statement, I.If):
        condition = scalar(statement.condition)
        changed = replace(
          statement, condition=condition, body=scope(statement.body, known), otherwise=scope(statement.otherwise, known)
        )
      elif isinstance(statement, I.Assign):
        changed = replace(statement, value=scalar(statement.value, bind=False))
        known[identity(statement.value)] = I.Expr(statement.value.type, "local", data=statement.target)
      elif isinstance(statement, (I.ArchitectureCall, I.Prefetch)):
        changed = replace(
          statement, **{field.name: metadata(getattr(statement, field.name)) for field in fields(statement)}
        )
      else:
        raise TypeError("metadata binding requires a validated architecture instruction program")
      output.append(changed)
    return I.Block(output)

  return scope(block, {})
