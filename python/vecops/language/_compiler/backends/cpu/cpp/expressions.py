"""C++ emission: expressions."""
from __future__ import annotations
import math
from ..capabilities import CPP_TYPES

class ExpressionsEmitter:
  def scalar_has_statements(self, value):
    """Whether a scalar expression needs structured statements, not just text."""
    if value.op.startswith("reduce_"):
      return True
    if value.op == "load":
      view = value.data
      if any(self.scalar_has_statements(v) for v in (view.offset, *view.shape, *view.strides)):
        return True
    return any(self.scalar_has_statements(a) for a in value.args)


  def scalar_inline(self, value):
    """Preserve repeated evaluation in a C++ for-condition expression.

    Most consumers emit scalar statements directly. A for condition must run
    again after every increment (including continue); its exceptional closure
    is always forced inline through CoreDefs.h.
    """
    if not self.scalar_has_statements(value):
      return self.scalar(value)
    previous_lines, previous_indent = self.lines, self.indent
    self.lines, self.indent = [], previous_indent + 1
    expression = self.scalar(value)
    self.line(f"return {expression};")
    body = "\n".join(self.lines)
    self.lines, self.indent = previous_lines, previous_indent
    return "([&]() VECOPS_INLINE_LAMBDA {\n" + body + "\n" + "  " * self.indent + "}())"


  def scalar_select(self, condition, when_true, when_false, dtype):
    """Lower lazy scalar arms in their own scopes without capturing a closure."""
    predicate = self.scalar(condition)
    result = self.unique("scalar_select")
    self.line(f"{CPP_TYPES[dtype]} {result}{{}};")
    previous_materialized = dict(self.reduction_materialized)
    self.opening(f"if ({predicate})")
    self.line(f"{result} = {when_true()};")
    self.close()
    self.reduction_materialized = dict(previous_materialized)
    self.opening("else")
    self.line(f"{result} = {when_false()};")
    self.close()
    # A cache computed in only one arm is not unconditionally available.
    self.reduction_materialized = previous_materialized
    return result


  def scalar(self, value):
    if value.op == "literal":
      if isinstance(value.data, bool):
        return "true" if value.data else "false"
      if isinstance(value.data, float):
        if math.isinf(value.data):
          return ("-" if value.data < 0 else "") + f"std::numeric_limits<{CPP_TYPES[value.type.dtype]}>::infinity()"
        if math.isnan(value.data):
          return f"std::numeric_limits<{CPP_TYPES[value.type.dtype]}>::quiet_NaN()"
        return repr(value.data)
      return f"std::int64_t{{{value.data}}}"
    if value.op in {"local", "raw", "frozen"}:
      return value.data
    if value.op == "symbol":
      if value.data not in self.ir.bindings:
        raise TypeError(f"dynamic symbol {value.data!r} has no direct tensor shape/stride binding")
      return self.scalar(self.ir.bindings[value.data])
    if value.op == "lanes":
      dtype, power = value.data
      base = f"vecops::vec::size(vecops::vec::ScalableTag<{CPP_TYPES[dtype]}, 0>{{}})"
      if not -62 <= power <= 62:
        raise TypeError("virtual vector extent exceeds the signed 64-bit extent domain")
      return f"vecops_dsl::lanes({base}, {power})"
    if value.op.startswith("reduce_"):
      return self.reduction(value)
    if value.op == "load":
      view = value.data
      def loaded():
        return f"static_cast<{CPP_TYPES[value.type.dtype]}>({self.pointer(view)}[0])"
      if value.args:
        condition, other = value.args
        address_values = (view.offset, *view.shape, *view.strides)
        if any(self.scalar_has_statements(v) for v in (*address_values, other)):
          return self.scalar_select(condition, loaded, lambda: self.scalar(other), value.type.dtype)
        predicate = self.scalar(condition)
        return f"({predicate} ? {loaded()} : {self.scalar(other)})"
      return loaded()
    if value.op == "where" and any(self.scalar_has_statements(a) for a in value.args[1:]):
      condition, a, b = value.args
      return self.scalar_select(condition, lambda: self.scalar(a), lambda: self.scalar(b), value.type.dtype)
    if value.op in {"and", "or"} and self.scalar_has_statements(value.args[1]):
      left, right = value.args
      selected = lambda: f"static_cast<bool>({self.scalar(right)})"
      return (self.scalar_select(left, selected, lambda: "false", value.type.dtype) if value.op == "and"
              else self.scalar_select(left, lambda: "true", selected, value.type.dtype))
    arguments = [self.scalar(a) for a in value.args]
    operators = {"add": "+", "sub": "-", "mul": "*", "div": "/", "lt": "<", "le": "<=", "gt": ">", "ge": ">=", "eq": "==", "ne": "!=", "and": "&&", "or": "||", "mask_xor": "!=", "bit_and": "&", "bit_or": "|", "bit_xor": "^", "shl": "<<", "shr": ">>"}
    if value.op in operators:
      expr = f"({arguments[0]} {operators[value.op]} {arguments[1]})"
      if value.op == "div" and value.type.dtype.startswith("float"):
        expr = f"(static_cast<{CPP_TYPES[value.type.dtype]}>({arguments[0]}) / {arguments[1]})"
      return expr
    if value.op in {"min", "max"}:
      return f"std::{value.op}<{CPP_TYPES[value.type.dtype]}>({', '.join(arguments)})"
    if value.op == "broadcast_dim":
      return f"vecops_dsl::broadcast_dim({', '.join(arguments)})"
    if value.op in {"floordiv", "mod", "ceildiv"}:
      cpp = CPP_TYPES[value.type.dtype]
      return f"static_cast<{cpp}>(vecops_dsl::{value.op}({', '.join(f'static_cast<{cpp}>({a})' for a in arguments)}))"
    if value.op == "cast":
      return f"static_cast<{CPP_TYPES[value.type.dtype]}>({arguments[0]})"
    if value.op in {"neg", "not", "identity", "bit_not"}:
      operator = {"neg": "-", "not": "!", "identity": "+", "bit_not": "~"}[value.op]
      return f"({operator}({arguments[0]}))"
    if value.op == "where":
      return f"({arguments[0]} ? {arguments[1]} : {arguments[2]})"
    if value.op == "fma":
      return f"std::fma({', '.join(arguments)})"
    if value.op in {"exp", "exp2", "exp_neg", "exp2_neg", "log", "log2", "sqrt", "abs"}:
      return f"std::{value.op.removesuffix('_neg')}({arguments[0]})"
    if value.op in {"rsqrt", "rcp"}:
      return f"(static_cast<{CPP_TYPES[value.type.dtype]}>(1) / {'std::sqrt(' + arguments[0] + ')' if value.op == 'rsqrt' else arguments[0]})"
    raise TypeError(f"no scalar lowering for {value.op}")
