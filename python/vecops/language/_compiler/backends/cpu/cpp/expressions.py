"""C++ emission: expressions."""

from __future__ import annotations

import math

from ....operations import Trait
from ..capabilities import CPP_TYPES


class ExpressionsEmitter:
  def scalar_has_statements(self, value):
    """Whether a scalar expression needs structured statements, not just text."""
    if (
      self.environment.operations.has(value, Trait.SCALAR_REDUCTION | Trait.MASK_REDUCTION) or value.op == "element_at"
    ):
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
    if value.op == "worker_capacity":
      return "dsl_workspace.parallelism()"
    if value.op == "view_disjoint":
      return f"!({self.overlap(*value.data)})"
    if value.op == "view_nonoverlap":
      view = value.data
      shape = ", ".join(self.scalar(n) for n in view.shape)
      strides = ", ".join(self.scalar(s) for s in view.strides)
      return f"vecops_dsl::grid_layout_safe<{len(view.shape)}>({{{shape}}}, {{{strides}}})"
    if self.environment.operations.get(value).family == "atomic":
      return self.atomic_scalar(value)
    if self.environment.operations.has(value, Trait.MASK_REDUCTION):
      return self.mask_query(value)
    if value.op == "element_at":
      return self.element_at(value)
    if value.op == "packing_size":
      from .matmul import FORMATS

      dtype, format_ = value.data
      return f"vecops_dsl::pack_size<{CPP_TYPES[dtype]}, {FORMATS[format_]}>({', '.join(self.scalar(v) for v in value.args)})"
    if value.op == "streaming_lanes":
      return f"vecops::vec::sme::streaming_lanes<{CPP_TYPES[value.data]}>()"
    if value.op == "literal":
      if isinstance(value.data, bool):
        return "true" if value.data else "false"
      if isinstance(value.data, float):
        if math.isinf(value.data):
          return ("-" if value.data < 0 else "") + f"std::numeric_limits<{CPP_TYPES[value.type.dtype]}>::infinity()"
        if math.isnan(value.data):
          return f"std::numeric_limits<{CPP_TYPES[value.type.dtype]}>::quiet_NaN()"
        text = repr(value.data)
        # C++ unsuffixed literals are double even when the IR literal is
        # explicitly F32. Make strong scalar computation execute in the
        # promised type; weak shaped fills retain their existing spelling.
        if not value.type.weak:
          return f"static_cast<{CPP_TYPES[value.type.dtype]}>({text})"
        return text
      return f"std::int64_t{{{value.data}}}"
    if value.op == "raw":
      # Exact entry-descriptor reads only. Do not pattern-match arbitrary raw
      # C++ or hoist a local/state expression that can change in a loop.
      return self.parameter_metadata.get(value.data, value.data)
    if value.op in {"local", "frozen"}:
      return value.data
    if value.op == "symbol":
      if value.data not in self.ir.bindings:
        raise TypeError(f"dynamic symbol {value.data!r} has no direct tensor shape/stride binding")
      return self.scalar(self.ir.bindings[value.data])
    if value.op == "lanes":
      dtype, power = value.data
      base = self.vector_size(f"vecops::vec::ScalableTag<{CPP_TYPES[dtype]}, 0>")
      if not -62 <= power <= 62:
        raise TypeError("virtual vector extent exceeds the signed 64-bit extent domain")
      return f"vecops_dsl::lanes({base}, {power})"
    if self.environment.operations.has(value, Trait.SCALAR_REDUCTION):
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
      return (
        self.scalar_select(left, selected, lambda: "false", value.type.dtype)
        if value.op == "and"
        else self.scalar_select(left, lambda: "true", selected, value.type.dtype)
      )
    arguments = [self.scalar(a) for a in value.args]
    if getattr(self, "hardware_scope", None) is not None:
      # Instruction-region validation proves these strictly positive divisors.
      # The ordinary floor/remainder helpers may throw or outline; metadata
      # here is nonnegative (clipped view sizes), so native integer arithmetic
      # has the same result and needs no call within the manual owner.
      if value.op in {"floordiv", "mod"}:
        operator = "/" if value.op == "floordiv" else "%"
        quotient = f"({arguments[0]} {operator} {arguments[1]})"
        # General signed metadata retains Python floor/remainder semantics.
        if value.op == "floordiv":
          return f"({quotient} - (({arguments[0]}) % ({arguments[1]}) < 0))"
        return f"({quotient} + ({quotient} < 0 ? ({arguments[1]}) : 0))"
      if value.op in {"min", "max"} and self.hardware_scope[0] == "sme":
        return f"vecops::vec::sme::{value.op}_value<{CPP_TYPES[value.type.dtype]}>({', '.join(arguments)})"
    implementation = self.environment.implementations.get(value.op)
    if self.environment.operations.has(value, Trait.ELEMENTWISE) and value.op != "cast":
      # Small-float scalar operators have competing conversion overloads.
      # Computation uses float, matching the IR scalar promotion; explicit
      # casts/stores are the rounding boundaries, not accidental C++ overloads.
      arguments = [
        f"static_cast<float>({text})" if arg.type.dtype in {"float16", "bfloat16"} else text
        for arg, text in zip(value.args, arguments)
      ]
    if implementation.scalar_kind == "infix":
      expr = f"({arguments[0]} {implementation.scalar_spelling} {arguments[1]})"
      if value.op == "div" and value.type.dtype.startswith("float"):
        expr = f"(static_cast<{CPP_TYPES[value.type.dtype]}>({arguments[0]}) / {arguments[1]})"
      return expr
    if implementation.scalar_kind == "std_binary":
      return f"std::{value.op}<{CPP_TYPES[value.type.dtype]}>({', '.join(arguments)})"
    if value.op == "broadcast_dim":
      if getattr(self, "matrix_validated", False):
        return f"std::max<std::int64_t>({', '.join(arguments)})"
      return f"vecops_dsl::broadcast_dim({', '.join(arguments)})"
    if implementation.lane_fallback:
      cpp = CPP_TYPES[value.type.dtype]
      return f"static_cast<{cpp}>(vecops_dsl::{value.op}({', '.join(f'static_cast<{cpp}>({a})' for a in arguments)}))"
    if value.op == "cast":
      return f"static_cast<{CPP_TYPES[value.type.dtype]}>({arguments[0]})"
    if implementation.scalar_kind == "prefix":
      operator = implementation.scalar_spelling
      return f"({operator}({arguments[0]}))"
    if value.op == "where":
      return f"({arguments[0]} ? {arguments[1]} : {arguments[2]})"
    if value.op == "fma":
      return f"std::fma({', '.join(arguments)})"
    if implementation.scalar_kind == "std_math":
      return f"std::{implementation.scalar_spelling}({arguments[0]})"
    if implementation.scalar_kind == "reciprocal":
      if value.type.dtype in {"float32", "float64"}:
        # Row statistics are scalar, but their math contract is the same as
        # vector math. Broadcasting into one native word lets each backend
        # select its reciprocal estimate/refinement tier (as C++ LayerNorm
        # does) rather than silently routing every tier through scalar libm.
        # A single get extracts the result; no closure or register array is
        # introduced, including the sizeless SVE representation.
        cpp = CPP_TYPES[value.type.dtype]
        tag = f"vecops::vec::ScalableTag<{cpp}, 0>"
        tier = value.data or "strict"
        return (
          f"vecops::vec::get({tag}{{}}, vecops::vec::{value.op}({tag}{{}}, "
          f"vecops::vec::fill({tag}{{}}, static_cast<{cpp}>({arguments[0]})), "
          f"vecops::vec::opt::math::{tier}), 0)"
        )
      return f"(static_cast<{CPP_TYPES[value.type.dtype]}>(1) / {'std::sqrt(' + arguments[0] + ')' if value.op == 'rsqrt' else arguments[0]})"
    raise TypeError(f"no scalar lowering for {value.op}")
