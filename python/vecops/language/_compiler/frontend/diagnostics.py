"""Source-located frontend diagnostics."""


class DSLError(TypeError):
  """An unsupported or ill-typed kernel operation, with its Python location."""
