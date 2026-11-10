# SPDX-FileCopyrightText: 2026 HenryRenYz and vecops contributors
# SPDX-License-Identifier: MIT
"""Topology-independent tensor/workspace placement and offset compilation."""
from .compiler import (BufferRequirement, BufferBinding, InitializationCopy, ProgramCall,
    CompiledMemoryProgram, GraphMemoryCompiler, PreparedProgram, requirements_from_graph,
    native_workspace_requirements, activate_native_buffers, reserve_recipe_latest, reserve_budget_latest, call_time_domains, ScopedNativeLease)
from .core import graph_digest,plan_memory,storage_traffic
from .validate import validate_plan,plan_summary
from .audit import audit_plan
from .profile import resource_snapshot
__all__=['BufferRequirement','BufferBinding','InitializationCopy','ProgramCall','CompiledMemoryProgram',
  'GraphMemoryCompiler','PreparedProgram','requirements_from_graph','native_workspace_requirements',
  'activate_native_buffers','reserve_recipe_latest','reserve_budget_latest','call_time_domains','ScopedNativeLease',
  'graph_digest','plan_memory','storage_traffic',
  'validate_plan','plan_summary','audit_plan','resource_snapshot']
