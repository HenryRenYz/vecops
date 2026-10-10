# SPDX-FileCopyrightText: 2026 HenryRenYz and vecops contributors
# SPDX-License-Identifier: MIT
"""Shared integer affine decomposition and conservative loop intervals.

Preserve cancellation before taking intervals (i + width - i is width).
Unknown nonlinear expressions remain unproved, never optimistic bounds.
"""

from dataclasses import fields, replace

from .. import ir as I


def metadata_reference(value):
  """Induction names and immutable ABI metadata use distinct namespaces."""
  if value.op == "local":
    return value.data
  if value.op in {"raw", "symbol"} and isinstance(value.data, str):
    return value.op, value.data
  return None


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
  reference = metadata_reference(value)
  if reference is not None and reference in variables:
    return {reference: 1}, I.literal(0)
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
  reference = metadata_reference(value)
  if reference is not None and reference in variables:
    return True
  if value.op in {"local", "frozen"}:
    if value.data in variables:
      return True
    if value.data in definitions and value.data not in seen:
      return affine_contains(definitions[value.data], variables, definitions, seen | {value.data})
  if any(affine_contains(a, variables, definitions, seen) for a in value.args):
    return True
  from ..ir.traversal import nodes

  return any(isinstance(v, I.Expr) and affine_contains(v, variables, definitions, seen) for v in nodes(value.data))


def interval(value, bounds, *, _ordered=True):
  """Inclusive bounds in loop-invariant metadata, or None if unproved."""
  value = canonical_integer(simplify_ordered(value, bounds) if _ordered else value)

  def bound(operand, ranges):
    return interval(operand, ranges, _ordered=_ordered)

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
        alternatives.append(bound(I.binary(value.op, *arguments), bounds))
      if any(v is None for v in alternatives):
        return None
      return tuple(I.binary(operation, alternatives[0][i], alternatives[1][i]) for i in (0, 1))
  if value.op == "mul":
    for literal, operand in (value.args, value.args[::-1]):
      factor = I.constant(literal)
      if type(factor) is not int or operand.op not in {"min", "max"}:
        continue
      operation = operand.op if factor >= 0 else "max" if operand.op == "min" else "min"
      alternatives = [bound(I.binary("mul", literal, arm), bounds) for arm in operand.args]
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
    lo = bound(I.binary("add", linear[1], I.binary("mul", I.literal(coefficient), endpoints[0])), rest)
    hi = bound(I.binary("add", linear[1], I.binary("mul", I.literal(coefficient), endpoints[1])), rest)
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
  operands = [bound(a, bounds) for a in value.args]
  if any(v is None for v in operands):
    return None
  if value.op in {"min", "max"}:
    return tuple(I.binary(value.op, operands[0][i], operands[1][i]) for i in (0, 1))
  if value.op in {"add", "sub"}:
    a, b = operands
    return (I.binary(value.op, a[0], b[value.op == "sub"]), I.binary(value.op, a[1], b[value.op != "sub"]))
  if value.op == "floordiv" and type(I.constant(value.args[1])) is int and I.constant(value.args[1]) > 0:
    return tuple(I.binary("floordiv", v, value.args[1]) for v in operands[0])
  if value.op == "mod" and type(I.constant(value.args[1])) is int and I.constant(value.args[1]) > 0:
    # DSL floor division/remainder is Euclidean, including signed operands.
    return I.literal(0), I.literal(I.constant(value.args[1]) - 1)
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
  arguments = tuple(simplify_ordered(a, bounds) for a in value.args)
  if any(a is not b for a, b in zip(arguments, value.args)):
    value = replace(value, args=arguments)
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
    constant = I.constant(endpoint)
    if constant is None:
      # Affine substitution can end at a bounded nonlinear remainder, e.g.
      # -(K % chunk). Use the same interval rules without recursively asking
      # this clamp-ordering routine to prove its own premise.
      span = interval(endpoint, bounds, _ordered=False)
      constant = None if span is None else I.constant(span[upper])
    endpoints.append(constant)
  lower, upper = endpoints
  if type(upper) is int and upper <= 0:
    return value.args[value.op == "max"]
  if type(lower) is int and lower >= 0:
    return value.args[value.op == "min"]
  return value


def canonical_integer(value, *, _memo=None):
  """Normalize pure integer metadata sums without touching floating DAGs."""
  memo = {} if _memo is None else _memo
  identity = id(value)
  if identity in memo:
    return memo[identity][1]
  result = _canonical_integer(value, memo)
  # Retain the original too: synthesized temporary Exprs can otherwise lose
  # their last reference and have an id reused during this same traversal.
  memo[identity] = value, result
  memo[id(result)] = result, result
  return result


def _canonical_integer(value, memo):
  if not isinstance(value.type, I.ScalarType) or not (value.type.dtype.startswith("int") or value.type.dtype == "bool"):
    return value
  arguments = tuple(canonical_integer(a, _memo=memo) for a in value.args)
  if any(a is not b for a, b in zip(arguments, value.args)):
    value = replace(value, args=arguments)
  if len(arguments) == 2 and all(a.op == "literal" for a in arguments):
    from ..operations import current

    if current().get(value).fold is not None:
      try:
        # Reuse the operation registry's folds/promotion, including comparisons
        # and quotient/remainder after a captured length became a literal.
        value = I.binary(value.op, *arguments)
      except ZeroDivisionError:
        # A dead branch is not permission to evaluate its undefined arithmetic.
        pass
  if value.type.dtype == "bool":
    return value
  if value.op == "sub" and value.args[0].op in {"min", "max"}:
    first, second = value.args
    for index, operand in enumerate(first.args):
      if I.same_expr(operand, second):
        return I.binary(
          first.op, I.literal(0), canonical_integer(I.binary("sub", first.args[1 - index], second), _memo=memo)
        )
  if value.op not in {"add", "sub", "mul"}:
    return value
  constants, terms = 0, {}

  def key(expression):
    from ..operations import current

    if expression.op in {"raw", "symbol", "local", "frozen"}:
      return expression.op, expression.data, expression.type.dtype
    if expression.op in {"literal", "min", "max", "streaming_lanes"}:
      return expression.op, expression.data, expression.type.dtype, tuple(key(a) for a in expression.args)
    if expression.data is None and current().get(expression).effects == "none":
      # Equal pure metadata products/quotients may arrive through distinct
      # view edges. Identity-only terms lose cancellation of row*pitch and
      # retain physical-address clamps even when logical bounds are proved.
      # Memory/state reads and opaque operations deliberately stay distinct.
      return expression.op, expression.type.dtype, tuple(key(a) for a in expression.args)
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
  for expression, factor in tuple(terms.values()):
    if expression.op != "floordiv" or not factor:
      continue
    numerator, denominator = expression.args
    divisor = I.constant(denominator)
    if type(divisor) is not int or divisor <= 0 or factor % divisor:
      continue
    own_key, numerator_key = key(expression), key(numerator)
    multiplier = factor // divisor
    if numerator_key not in terms or terms[numerator_key][1] != -multiplier:
      continue
    # d * floor(n/d) - n == -(n % d). Retaining that correlation makes
    # full-chunk bounds provable rather than bounding n and n/d separately.
    terms[own_key] = expression, 0
    terms[numerator_key] = numerator, 0
    remainder = I.binary("mod", numerator, denominator)
    remainder_key = key(remainder)
    previous, coefficient = terms.get(remainder_key, (remainder, 0))
    terms[remainder_key] = previous, coefficient - multiplier
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
            canonical_integer(I.binary("add", result, I.binary("mul", I.literal(factor), arm)), _memo=memo)
            for arm in expression.args
          ),
        )
      else:
        result = I.binary("add", result, I.binary("mul", I.literal(factor), expression))
  return result


def branch_bounds(condition, bounds, *, taken=True):
  """Refine enclosing integer induction ranges on one branch edge.

  Only unit-coefficient affine comparisons are used. A bound may refer to
  outer indices, never its own or an inner index; this preserves the acyclic
  substitution order used by interval(). Unknown/disjunctive facts remain
  unproved, and facts from one arm cannot escape to its sibling or successor.
  """
  if condition.op == "not":
    return branch_bounds(condition.args[0], bounds, taken=not taken)
  if condition.op == ("and" if taken else "or"):
    for operand in condition.args:
      bounds = branch_bounds(operand, bounds, taken=taken)
    return bounds
  operation = condition.op
  if operation not in {"lt", "le", "gt", "ge", "eq", "ne"}:
    return bounds
  if not taken:
    operation = {"lt": "ge", "le": "gt", "gt": "le", "ge": "lt", "eq": "ne", "ne": "eq"}[operation]
  if operation == "ne" or any(
    not isinstance(v.type, I.ScalarType) or not v.type.dtype.startswith("int") for v in condition.args
  ):
    return bounds
  # ABI metadata is invariant outside every induction domain. Place its
  # namespace before loop indices even when first encountered in an inner
  # branch, so no metadata endpoint can depend on an inner induction value.
  result = {name: pair for name, pair in bounds.items() if isinstance(name, tuple)}
  from ..ir.traversal import walk_unique

  for argument in condition.args:
    for value in walk_unique(argument):
      reference = metadata_reference(value)
      if isinstance(reference, tuple) and value.type.dtype == "int64":
        result.setdefault(reference, (I.literal(-(1 << 63)), I.literal((1 << 63) - 1)))
  result.update({name: pair for name, pair in bounds.items() if not isinstance(name, tuple)})
  names = tuple(result)
  difference = canonical_integer(I.binary("sub", *condition.args))
  for position, name in enumerate(names):
    linear = affine(difference, {name}, {})
    if linear is None or linear[0].get(name) not in (-1, 1):
      continue
    coefficient, remainder = linear[0][name], linear[1]
    if affine_contains(remainder, set(names[position:]), {}):
      continue
    endpoint = canonical_integer(I.binary("mul", I.literal(-coefficient), remainder))
    endpoint_value = I.constant(endpoint)
    if type(endpoint_value) is int and not -(1 << 63) < endpoint_value < (1 << 63) - 1:
      # Do not synthesize an unrepresentable signed endpoint/strict offset.
      # Keeping the original range remains conservative on extreme inputs.
      continue
    selected = operation
    if coefficient < 0:
      selected = {"lt": "gt", "le": "ge", "gt": "lt", "ge": "le", "eq": "eq"}[selected]
    lower, upper = result[name]
    if selected in {"gt", "ge", "eq"}:
      candidate = I.binary("add", endpoint, I.literal(int(selected == "gt")))
      lower = canonical_integer(simplify_ordered(I.binary("max", lower, candidate), result))
    if selected in {"lt", "le", "eq"}:
      candidate = I.binary("sub", endpoint, I.literal(int(selected == "lt")))
      upper = canonical_integer(simplify_ordered(I.binary("min", upper, candidate), result))
    result[name] = lower, upper
  return result


def simplify_loop_metadata(block, bounds=None, *, streaming_bytes=None, _constants=None, _equalities=()):
  """Simplify integer addresses in a validated immutable structured region.

  This is an induction-domain optimization, not an operator-name pattern.
  It retains SSA references, execution order and payload types. An explicitly
  captured SVL may specialize native lengths; unknown/cross-target geometry
  retains native queries. Only literal SSA definitions are propagated, never
  expanded runtime queries. Branch ranges and exact integer equality facts
  stay lexical. A full clipped extent proved equal to a native width need not
  reconstruct a tail mask at each instruction in that branch.
  Scopes have separate DAG memoization because an expression can be shared
  between loops with different induction domains.
  """
  bounds = {} if bounds is None else bounds
  if streaming_bytes is not None and (
    type(streaming_bytes) is not int or not 16 <= streaming_bytes <= 256 or streaming_bytes & (streaming_bytes - 1)
  ):
    raise ValueError("captured streaming length must be a power of two between 16 and 256 bytes")
  constants = {} if _constants is None else dict(_constants)
  equalities = tuple(_equalities)
  memo = {}

  def nested(body, ranges, *, shadow=None, facts=()):
    inherited = {name: value for name, value in constants.items() if name != shadow}
    known = tuple(pair for pair in equalities if shadow is None or not affine_contains(pair[0], {shadow}, {}))
    return simplify_loop_metadata(
      body, ranges, streaming_bytes=streaming_bytes, _constants=inherited, _equalities=(*known, *facts)
    )

  def edge_equalities(condition, taken=True):
    from ..ir.traversal import walk_unique
    from ..operations import current

    if condition.op == "not":
      return edge_equalities(condition.args[0], not taken)
    if condition.op == ("and" if taken else "or"):
      return tuple(pair for arg in condition.args for pair in edge_equalities(arg, taken))
    if condition.op != ("eq" if taken else "ne"):
      return ()
    for expression, literal in (condition.args, condition.args[::-1]):
      if (
        type(I.constant(literal)) is int
        and isinstance(expression.type, I.ScalarType)
        and expression.type.dtype.startswith("int")
        and isinstance(literal.type, I.ScalarType)
        and literal.type.dtype.startswith("int")
        and all(
          isinstance(value.type, I.ScalarType)
          and value.type.dtype in {"bool", "int32", "int64"}
          and current().get(value).effects == "none"
          for value in walk_unique(expression)
        )
      ):
        return ((expression, I.literal(I.constant(literal), expression.type.dtype)),)
    return ()

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
        body=nested(value.body, {**bounds, value.target: endpoints}, shadow=value.target),
      )
    if isinstance(value, I.If):
      condition = visit(value.condition)
      return replace(
        value,
        condition=condition,
        body=nested(value.body, branch_bounds(condition, bounds), facts=edge_equalities(condition)),
        otherwise=nested(
          value.otherwise, branch_bounds(condition, bounds, taken=False), facts=edge_equalities(condition, False)
        ),
      )
    if isinstance(value, I.Assign):
      changed = replace(value, value=visit(value.value))
      constants.pop(value.target, None)
      if value.declare and type(I.constant(changed.value)) is int:
        constants[value.target] = changed.value
      return changed
    if isinstance(value, I.Block):
      return I.Block([visit(s) for s in value.statements])
    if isinstance(value, tuple):
      return tuple(visit(v) for v in value)
    if isinstance(value, I.Node):
      if id(value) not in memo:
        changed = replace(value, **{field.name: visit(getattr(value, field.name)) for field in fields(value)})
        if isinstance(changed, I.Expr):
          if changed.op in {"local", "frozen"} and changed.data in constants:
            changed = constants[changed.data]
          elif changed.op == "streaming_lanes" and streaming_bytes is not None:
            from ..dtypes import ELEMENT_BYTES

            changed = I.literal(streaming_bytes // ELEMENT_BYTES[changed.data])
        if isinstance(changed, I.Expr):
          changed = canonical_integer(simplify_ordered(changed, bounds))
          if isinstance(changed.type, I.ScalarType) and changed.type.dtype.startswith("int"):
            changed = next((literal for expression, literal in equalities if I.same_expr(changed, expression)), changed)
        memo[id(value)] = changed
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
      if isinstance(value, (I.SMEVector, I.SMEMask)):
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
        # A literal is already an SSA value. Replacing it with a local name
        # hides native-width facts from primitive lowering (e.g. a full mask)
        # without sharing any work. Preserve its original type/weakness.
        known[identity(statement.value)] = (
          changed.value
          if changed.value.op == "literal"
          else I.Expr(statement.value.type, "local", data=statement.target)
        )
      elif isinstance(statement, (I.ArchitectureCall, I.Prefetch)):
        changed = replace(
          statement, **{field.name: metadata(getattr(statement, field.name)) for field in fields(statement)}
        )
      else:
        raise TypeError("metadata binding requires a validated architecture instruction program")
      output.append(changed)
    return I.Block(output)

  return scope(block, {})
