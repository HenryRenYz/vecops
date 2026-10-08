# SPDX-FileCopyrightText: 2026 HenryRenYz and vecops contributors
# SPDX-License-Identifier: MIT
"""Semantic operation catalog; independent of frontend, passes and backends."""

from . import builtin as _builtin  # noqa: F401 -- register built-in contracts once
from .registry import OpDef, Trait, current, register_operation, snapshot, unregister_operation, use_catalog


def has(operation, trait, catalog=None):
  return (catalog or current()).has(operation, trait)


__all__ = ["OpDef", "Trait", "current", "has", "register_operation", "snapshot", "unregister_operation", "use_catalog"]
