# SPDX-FileCopyrightText: 2026 HenryRenYz and vecops contributors
# SPDX-License-Identifier: MIT
"""Generic C++ memory resources, arenas, mappings and Python ownership helpers.

Machine-specific discovery facts are backend descriptors. The public planning
surface uses agents, memory pools, capacity groups, paths and explicit owners.
"""
from ._resource import (
  Pool,
  Runtime,
  current_runtime,
  default_runtime,
  native,
  tensor_view,
  use_runtime,
)

__all__ = [
  'Pool', 'Runtime', 'current_runtime', 'default_runtime', 'native', 'tensor_view', 'use_runtime',
]
