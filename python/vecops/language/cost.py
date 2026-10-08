# SPDX-FileCopyrightText: 2026 HenryRenYz and vecops contributors
# SPDX-License-Identifier: MIT
"""Immutable cost profiles for compiler tuning, independent of DSL semantics.

Pass profiles through T.compile(..., pass_options={"cost_profiles": (profile,)})
or register a named profile before compilation. Cycle values are estimates;
registering a cost profile never enables an ISA or changes numerical semantics.
"""

from ._compiler.cost import CostProfile, CostQuery, CostRule, OperationCost, register_profile, unregister_profile

__all__ = ["CostProfile", "CostQuery", "CostRule", "OperationCost", "register_profile", "unregister_profile"]
