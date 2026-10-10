# SPDX-FileCopyrightText: 2026 HenryRenYz and vecops contributors
# SPDX-License-Identifier: MIT
"""SME target costs; algorithm traversal belongs to source DSL recipes."""

from ....cost.model import CostProfile

SME = CostProfile(
  "sme",
  version="7",
  # Constant-pitch F32-to-F16 transpose selection is separate from legality.
  # Real target features and the captured process SVL remain authoritative.
  policies=(("sme_const_half_transpose", 1),),
)
