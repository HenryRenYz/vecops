# SPDX-FileCopyrightText: 2026 HenryRenYz and vecops contributors
# SPDX-License-Identifier: MIT
"""Source-located frontend diagnostics."""


class DSLError(TypeError):
  """An unsupported or ill-typed kernel operation, with its Python location."""
