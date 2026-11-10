# SPDX-FileCopyrightText: 2026 HenryRenYz and vecops contributors
# SPDX-License-Identifier: MIT
"""Unified storage requirements for tensors and native operator workspace."""
from .compiler import BufferRequirement, native_workspace_requirements
__all__=['BufferRequirement','native_workspace_requirements']
