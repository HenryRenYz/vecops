# SPDX-FileCopyrightText: 2026 HenryRenYz and vecops contributors
# SPDX-License-Identifier: MIT
"""Source-level structured control helpers; no backend representation choices."""

import ast
import copy


def assigned_names(nodes):
  return {
    node.id
    for statement in nodes
    for node in ast.walk(statement)
    if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Store)
  }


def helper_returns(nodes, result, error):
  """Normalize acyclic early returns by threading their continuation.

  No runtime closures or duplicated *execution*: following statements are
  placed only in paths which have not returned. Loop returns need a separate
  exit region and are diagnosed instead of escaping the enclosing kernel.
  """

  def has_return(node):
    return any(isinstance(item, ast.Return) for item in ast.walk(node))

  def transform(statements):
    output = []
    for index, statement in enumerate(statements):
      rest = statements[index + 1 :]
      if isinstance(statement, ast.Return):
        if statement.value is None and result is not None:
          error(statement, "value-returning T.func helpers must return a value on every path")
        output.append(
          ast.copy_location(
            ast.Assign([ast.Name(result, ast.Store())], statement.value) if result is not None else ast.Pass(),
            statement,
          )
        )
        return output
      if isinstance(statement, (ast.For, ast.While)) and has_return(statement):
        error(statement, "T.func return inside a loop requires an exit region; move the return after the loop")
      if isinstance(statement, ast.If) and has_return(statement):
        branch = copy.copy(statement)
        branch.body = transform(statement.body + rest)
        branch.orelse = transform(statement.orelse + rest)
        output.append(branch)
        return output
      output.append(statement)
    return output

  return transform(list(nodes))
