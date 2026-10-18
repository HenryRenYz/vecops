# SPDX-FileCopyrightText: 2026 HenryRenYz and vecops contributors
# SPDX-License-Identifier: MIT
"""Complete SIMD prefixes must never consume a partial contraction group."""

import re

import pytest
import vecops.language as T
from vecops.language._compiler.analysis.projection import complete_grouped_prefix
from vecops.language._compiler.backends.cpu.backend import CPUBackend


def test_prefix_proof_for_all_small_extents_and_scalable_width_multiples():
  for group in (2, 4):
    for k in range(1025):
      for multiple in (1, 2, 4, 8, 16):
        if complete_grouped_prefix(k, group, multiple):
          for lanes in range(multiple, 257, multiple):
            full_outputs = ((k + group - 1) // group) // lanes * lanes
            assert full_outputs * group <= k
  assert not complete_grouped_prefix(None, 4, 16)
  assert not complete_grouped_prefix(-1, 4, 16)
  assert not complete_grouped_prefix(65, 4, 0)


@pytest.mark.parametrize("k,masked", [(255, True), (256, False), (257, False)])
def test_partial_source_group_is_masked_only_when_it_can_end_a_full_chunk(k, masked):
  @T.template
  def kernel(
    A: T.Tensor((k,), dtype="uint8"),
    B: T.Tensor((k,), dtype="uint8"),
    C: T.Tensor(((k + 3) // 4,), dtype="uint32", access="output"),
  ):
    T.store(C, T.widening_dot(T.load(A), T.load(B), dtype="uint32"))

  source = CPUBackend().lower(kernel.ir, (), target="AVX512").source
  full = [
    section.split("/* masked vector tail */", 1)[0]
    for section in re.split(r"/\* full vector groups \*/ \{", source)[1:]
  ]
  contractions = [section for section in full if "vecops::vec::widening_dot" in section]
  assert contractions
  # Runtime singleton alternatives may emit a FixedTag<1> scalar projection;
  # its own partial group still needs masking. Check the actual SIMD tags.
  assert all(
    bool(re.search(r"dot_input.*= vecops::vec::blend\(vecops::vec::ViewAs<std::uint8_t, \w+group_tag", section))
    is masked
    for section in contractions
  )
  # A nonempty partial source group still requires tail sanitization.
  if k % 4:
    assert "dot_input" in source
