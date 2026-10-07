"""Python print/f-string semantics, frozen output targets and source locations."""

import ast
import builtins
import re
import sys

from ..._debug.options import PrintTarget
from ..._debug.session import current_session
from .. import ir as I


class PrintFrontend:
  def joined_string(self, node):
    format_, arguments = "", []
    for part in node.values:
      if isinstance(part, ast.Constant):
        format_ += part.value.replace("{", "{{").replace("}", "}}")
        continue
      value = self.expression(part.value)
      specification = ""
      if part.format_spec is not None:
        if any(not isinstance(v, ast.Constant) for v in part.format_spec.values):
          self.error(part, "print format specifications must be compile-time strings")
        specification = "".join(v.value for v in part.format_spec.values)
        precision = re.search(r"\.(\d+)", specification)
        if precision and int(precision[1]) > 64:
          self.error(part, "kernel print precision must be at most 64 for allocation-free formatting")
        if "L" in specification or "?" in specification:
          self.error(part, "kernel print does not support allocating locale/debug string formats")
      if part.conversion not in (-1, 115):
        self.error(part, "kernel f-strings currently support default conversion and !s, not !r/!a")
      format_ += "{:" + specification + "}" if specification else "{}"
      arguments.append(value)
    return I.FormattedText(format_, tuple(arguments))

  def print_call(self, operation, args, kwargs, node):
    from ..._debug.options import stdout

    if set(kwargs) - {"sep", "end", "file", "flush", "buffer_size", "truncation"}:
      self.error(node, "print accepts sep, end, file, flush, buffer_size and truncation keyword arguments")

    def text(value, default):
      if value is None:
        return default
      if not isinstance(value, str):
        self.error(node, "print sep/end must be compile-time strings or None")
      return value

    separator, end = text(kwargs.get("sep"), " "), text(kwargs.get("end"), "\n")
    buffer_size = kwargs.get("buffer_size", 1024)
    if isinstance(buffer_size, I.Expr) and buffer_size.op == "literal":
      buffer_size = buffer_size.data
    truncation = kwargs.get("truncation", " ... (trunc)")
    if type(buffer_size) is not int or not 0 < buffer_size <= 1048576:
      self.error(node, "print buffer_size must be a compile-time integer in [1, 1048576]")
    if not isinstance(truncation, str) or len(truncation.encode("utf-8")) > buffer_size:
      self.error(node, "print truncation must be a compile-time string fitting buffer_size (UTF-8 bytes)")
    destination = kwargs.get("file")
    if destination is None:
      destination = stdout
    if not isinstance(destination, PrintTarget):
      self.error(node, "kernel print file must be T.stdout/T.stderr or None, not a Python file object")
    flush = self.value(kwargs.get("flush", False), node)
    if not isinstance(flush.type, I.ScalarType) or flush.type.dtype != "bool":
      self.error(node, "print flush must be boolean")

    def constant(value):
      if isinstance(value, I.FormattedText):
        return value.format.format(*(constant(v) for v in value.arguments))
      if isinstance(value, tuple):
        return tuple(constant(v) for v in value)
      if isinstance(value, I.Expr):
        if value.op != "literal":
          self.error(node, "static_print arguments must be compile-time values")
        return value.data
      if isinstance(value, I.View):
        self.error(node, "static_print arguments must be compile-time values, not memory views")
      return value

    if operation == "static_print":
      values = tuple(constant(v) for v in args)
      rendered = separator.join(str(v) for v in values) + end
      encoded = rendered.encode("utf-8")
      capacity = buffer_size - len(truncation.encode("utf-8"))
      if len(encoded) > capacity:
        rendered = encoded[:capacity].decode("utf-8", errors="ignore") + truncation
      session = current_session.get()
      if session is not None:
        session.messages.append({"kind": "static_print", "text": rendered})
      builtins.print(
        rendered, end="", file=sys.stderr if destination.name == "stderr" else sys.stdout, flush=constant(flush)
      )
      return

    def validate(value):
      if isinstance(value, tuple):
        for item in value:
          validate(item)
      elif isinstance(value, I.FormattedText):
        for item in value.arguments:
          validate(item)
      elif not isinstance(value, (I.Expr, I.View, str, type(None))):
        self.error(node, "print supports DSL values and immutable text/tuples")

    for value in args:
      validate(value)

    def readable(value):
      # Memory effects must be visible to dependence/ownership analyses too,
      # rather than introducing hidden loads only in the final C++ printer.
      if isinstance(value, I.View):
        return I.Expr(I.TileType(value.dtype, value.shape), "load", data=value)
      if isinstance(value, tuple):
        return tuple(readable(v) for v in value)
      if isinstance(value, I.FormattedText):
        return I.FormattedText(value.format, tuple(readable(v) for v in value.arguments))
      return value

    self.block.statements.append(
      I.Print(
        tuple(readable(v) for v in args),
        separator,
        end,
        destination.name,
        flush,
        (self.body.filename, self.body.first_line + node.lineno - 1, node.col_offset),
        buffer_size,
        truncation,
      )
    )
    return
