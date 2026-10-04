"""Print diagnostics through C++20 format_to and a synchronized stream sink."""

import json

from .... import ir as I


class PrintEmitter:
  def emit_print(self, statement):
    options = self.vector_ir.manager.options
    if not options.get("debug_prints", True):
      return
    record = self.unique("print")
    self.line(f"// print from {json.dumps(statement.origin[0])}:{statement.origin[1]}")

    def literal(text):
      # Octal escapes preserve embedded NUL/control bytes in C++ literals.
      encoded = text.encode("utf-8")
      quoted = '"' + "".join(chr(b) if 32 <= b < 127 and b not in (34, 92) else f"\\{b:03o}" for b in encoded) + '"'
      return f"std::string_view{{{quoted}, {len(encoded)}}}"

    self.line(f"vecops::execution::PrintRecord<{statement.buffer_size}> {record}({literal(statement.truncation)});")

    def append_text(text):
      if text:
        self.line(f'{record}.append("{{}}", {literal(text)});')

    def scalar(value):
      expression = self.scalar(value)
      if value.type.dtype == "bool":
        return f'(({expression}) ? "True" : "False")'
      if value.type.dtype.startswith(("float", "bfloat")):
        return f"static_cast<double>({expression})"
      return expression

    def argument(value, format_="{}"):
      if isinstance(value, I.FormattedText):
        # Each field is formatted directly into the same private record.
        import string

        cursor = 0
        for text, name, specification, conversion in string.Formatter().parse(value.format):
          append_text(text)
          if name is not None:
            if name not in ("", str(cursor)) or conversion is not None:
              raise TypeError("unsupported kernel f-string field")
            argument(value.arguments[cursor], "{:" + specification + "}" if specification else "{}")
            cursor += 1
        return
      if isinstance(value, tuple):
        append_text("(")
        for index, item in enumerate(value):
          if index:
            append_text(", ")
          argument(item)
        append_text(",)" if len(value) == 1 else ")")
        return
      if isinstance(value, I.View):
        value = I.Expr(I.TileType(value.dtype, value.shape), "load", data=value)
      if value is None or isinstance(value, str):
        self.line(f"{record}.append({json.dumps(format_)}, {literal('None' if value is None else value)});")
      elif isinstance(value.type, I.ScalarType):
        self.line(f"{record}.append({json.dumps(format_)}, {scalar(value)});")
      else:
        is_mask = value.type.mask
        # GetOp has a public mask-lane overload. Avoid converting predicates
        # through uint8 vectors (unnecessary work and problematic SVE subwords).
        append_text(f"{'bool' if is_mask else value.type.dtype}[")
        count = self.unique("print_count")
        index = self.unique("print_index")
        shape = [self.scalar(n) for n in value.type.shape]
        self.line(f"const auto {count} = vecops_dsl::checked_shape({{{', '.join(shape)}}});")
        limit = options.get("max_print_elements", 16)
        self.opening(f"for (vecops::nint_t {index}=0; {index}<std::min<vecops::nint_t>({count},{limit}); ++{index})")
        self.line(f'if ({index}) {record}.append(", ");')
        remainder, coordinates = index, []
        for extent in reversed(shape):
          coordinates.append(f"(({remainder}) % ({extent}))")
          remainder = f"(({remainder}) / ({extent}))"
        points = tuple(I.Expr(I.ScalarType("int64"), "raw", data=c) for c in reversed(coordinates))
        element = I.Expr(I.ScalarType("bool" if is_mask else value.type.dtype), "element_at", (value, *points))
        self.chunk_cache = {}
        self.line(f"{record}.append({json.dumps(format_)}, {scalar(element)});")
        self.close()
        self.line(f'if ({count}>{limit}) {record}.append(", ... ({{}} elements)", {count});')
        append_text("]")

    for index, value in enumerate(statement.arguments):
      if index:
        append_text(statement.separator)
      argument(value)
    append_text(statement.end)
    self.line(
      f"{record}.emit({'true' if statement.destination == 'stderr' else 'false'}, {self.scalar(statement.flush)});"
    )
