# SPDX-FileCopyrightText: 2026 HenryRenYz and vecops contributors
# SPDX-License-Identifier: MIT
"""SME costs, bounded native-A packing and prepacked traversal policies.

The cycle/byte rates below are conservative seeds, not hardware measurements.
They cost the actual spatial gather direction and padded panel, rather than
assuming that K-contiguous A is also contiguous for an outer-product leaf.
"""

from dataclasses import dataclass

from ....cost.model import CostProfile

_NINT_MAX = (1 << 63) - 1


@dataclass(frozen=True)
class NativeAPackingPolicy:
  minimum_work: int = 128 * 1024
  minimum_spatial_pairs: int = 128
  minimum_reuse_groups: int = 2
  maximum_padding_ratio: int = 4
  gather_cycles: int = 4
  contiguous_cycles: int = 1
  packing_bytes_per_cycle: int = 32
  packing_setup_cycles: int = 64

  def __post_init__(self):
    for name in self.__dataclass_fields__:
      value = getattr(self, name)
      if type(value) is not int or not 0 <= value <= (1 << 20):
        raise ValueError(f"sme_native_a_pack_{name} must be an integer in [0,1048576]")
    if (
      not self.minimum_work
      or not self.minimum_spatial_pairs
      or self.minimum_reuse_groups < 2
      or not self.maximum_padding_ratio
      or self.gather_cycles <= self.contiguous_cycles
      or not self.packing_bytes_per_cycle
    ):
      raise ValueError("SME native-A packing policy needs positive budgets and a gather penalty")

  @classmethod
  def from_costs(cls, costs):
    defaults = cls()
    return cls(
      **{name: costs.policy(f"sme_native_a_pack_{name}", getattr(defaults, name)) for name in cls.__dataclass_fields__}
    )

  def profitable(self, m, n, k, *, panel, lanes, k_group, element_bytes):
    """Evaluate one actual leaf/NC panel, never speculative full-N reuse.

    GEMV and high-padding skinny leaves deliberately keep their existing raw
    route. Overflowing physical allocations are rejected without constructing
    a buffer. The emitter uses the same quotient-based comparison in C++.
    """
    if min(m, n) <= 1 or k <= 0 or min(panel, lanes, k_group, element_bytes) <= 0:
      return False
    padded_m = (m + panel - 1) // panel * panel
    padded_k = (k + k_group - 1) // k_group * k_group
    if max(m * n, m * k, n * k) > _NINT_MAX or padded_m * padded_k > _NINT_MAX // element_bytes:
      return False
    logical = m * k
    physical = padded_m * padded_k
    # Raw/conditional-A F64 retains the shared eight-ZA schedule: 4x2
    # wins first, then 2x4 covers 2L <= M < 4L and N >= 4L. Credit only
    # actual N-block reuse, not two copies for one 2x4 family invocation.
    expanded_columns = element_bytes == 8 and 2 * lanes <= m < 4 * lanes and n >= 4 * lanes
    column_span = (4 if m <= lanes or expanded_columns else 1 if n <= lanes else 2) * lanes
    reuse = (n + column_span - 1) // column_span
    if (
      m * n < self.minimum_spatial_pairs
      or m * n < (self.minimum_work + k - 1) // k
      or reuse < self.minimum_reuse_groups
      or physical > logical * self.maximum_padding_ratio
    ):
      return False
    groups = (m + lanes - 1) // lanes * ((k + k_group - 1) // k_group)
    saving_per_reuse = groups * (self.gather_cycles - self.contiguous_cycles) * self.packing_bytes_per_cycle
    traffic = (logical + physical) * element_bytes + self.packing_setup_cycles * self.packing_bytes_per_cycle
    return reuse - 1 >= (traffic + saving_per_reuse - 1) // saving_per_reuse


@dataclass(frozen=True)
class PackedABTraversalPolicy:
  """Conservative whole-K candidate for short, wide prepacked products.

  These are tuning budgets, not instruction legality or a fixed hardware VL.
  A whole two-by-two ZA micro-panel must fit the declared private-cache share.
  Keeping K live avoids partial-C traffic and per-cache-panel owner entry. The
  caller separately proves operand format, dtype and automatic-block selection.
  """

  maximum_row_panels: int = 8
  minimum_column_panels: int = 8
  minimum_k_groups: int = 256
  # The retained full-K packed panels are streams, not a separately allocated
  # packing workspace. Budget the complete 2x2 micro-panel against private L2;
  # an arbitrary half-cache reserve rejects the measured 512 KiB panel on a
  # 768 KiB L2. Targets may retain a stricter share through their cost profile.
  private_cache_divisor: int = 1

  def __post_init__(self):
    for name in self.__dataclass_fields__:
      value = getattr(self, name)
      if type(value) is not int or not 1 <= value <= (1 << 20):
        raise ValueError(f"sme_packed_ab_{name} must be an integer in [1,1048576]")

  @classmethod
  def from_costs(cls, costs):
    defaults = cls()
    return cls(**{name: costs.policy(f"sme_packed_ab_{name}", getattr(defaults, name)) for name in vars(defaults)})

  def shape_candidate(self, m, n, k, *, lanes, k_group):
    """A cheap geometry gate, evaluated before querying CPU cache metadata."""
    if min(m, n, k, lanes, k_group) <= 0 or max(m, n, k) > _NINT_MAX:
      return False
    return (
      m <= self.maximum_row_panels * lanes
      and n // lanes >= self.minimum_column_panels
      and k // k_group >= self.minimum_k_groups
    )

  def profitable(self, m, n, k, *, lanes, k_group, element_bytes, l2_bytes):
    """Mirror the generated quotient checks, including padded final K."""
    if element_bytes <= 0 or l2_bytes <= 0 or not self.shape_candidate(m, n, k, lanes=lanes, k_group=k_group):
      return False
    groups = k // k_group + int(k % k_group != 0)
    budget = l2_bytes // self.private_cache_divisor // (4 * lanes * element_bytes)
    return groups <= budget // k_group


SME = CostProfile(
  "sme",
  version="6",
  # The complete packed-AB geometry cohort admits this narrow default. The
  # emitter still checks formats, automatic blocking, geometry and actual L2
  # residency before selecting it. A processor profile can disable it without
  # changing frontend semantics or the packed ABI.
  # The complete transpose catalog also validates this bounded, constant-pitch
  # F32-to-F16 schedule. Instruction legality still comes from the real target,
  # captured SVL, simple load and signed offsets, never from a cost profile.
  policies=(("sme_packed_ab_cache_bypass", 1), ("sme_const_half_transpose", 1))
  + tuple((f"sme_native_a_pack_{name}", value) for name, value in vars(NativeAPackingPolicy()).items())
  + tuple((f"sme_packed_ab_{name}", value) for name, value in vars(PackedABTraversalPolicy()).items()),
)
