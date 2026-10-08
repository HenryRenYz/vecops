# SPDX-FileCopyrightText: 2026 HenryRenYz and vecops contributors
# SPDX-License-Identifier: MIT
"""Native compiler row hints do not change the explicit vector schedule."""

from dataclasses import replace

import pytest
from test_dsl_transpose_vla_selection import CONVERSIONS, lower, output_store, transpose_template
from vecops.language._compiler.ir.vector import VectorRegion


@pytest.mark.parametrize("source,output,compute", CONVERSIONS)
def test_tiny_sve_conversion_rolls_hot_alias_and_snapshot_rows(source, output, compute):
  artifact = lower(transpose_template(source, output, compute))
  region = artifact.lowered.plans[id(output_store(artifact))]
  assert region.outer_unroll == 1
  assert region.unroll == 1
  assert region.base_power == 0
  assert tuple(region.groupings) == (0,)
  snapshots = tuple(artifact.backend_ir.snapshots.values())
  assert snapshots and all(snapshot.outer_unroll == 1 for snapshot in snapshots)
  assert all(snapshot.unroll == 2 for snapshot in snapshots)
  for snapshot in snapshots:
    assert replace(snapshot, effects=()).outer_unroll == 1  # Alias destination rebinding.
  assert "#pragma clang loop unroll(disable)" in artifact.source
  assert "#pragma GCC unroll 1" in artifact.source
  assert "vecops::vec::load_convert" in artifact.source
  assert "vecops::vec::store_convert" in artifact.source
  assert "dsl_c_alias_" in artifact.source
  assert "dsl_c_snapshot_" in artifact.source


@pytest.mark.parametrize(
  "kwargs,lowering",
  (
    ({}, {"target": "AVX512"}),
    ({}, {"target": "Scalar"}),
    ({}, {"vl": 16, "svl": 64}),
    ({}, {"vl": None, "svl": 64}),
    ({"implementation": "gather"}, {}),
    ({"implementation": "vector"}, {}),
    ({"implementation": "sme"}, {}),
    ({"source": "float64", "shape": (63, 129)}, {}),
    ({"shape": (17, 17)}, {}),
    ({"expression": "affine"}, {}),
    ({"source": "float32", "output": "float32"}, {}),
  ),
)
def test_row_hint_does_not_override_other_algorithms_shapes_or_targets(kwargs, lowering):
  artifact = lower(transpose_template(**kwargs), **lowering)
  assert all(region.outer_unroll is None for region in artifact.lowered.regions if isinstance(region, VectorRegion))
  assert all(region.outer_unroll is None for region in artifact.backend_ir.snapshots.values())
  assert "#pragma clang loop unroll(disable)" not in artifact.source


@pytest.mark.parametrize("shape", ((2, 7), (7, 13), (15, 16), (16, 15)))
def test_small_non_square_conversion_keeps_independent_rows_rolled(shape):
  artifact = lower(transpose_template(shape=shape, arbitrary_strides=True), vl=32, svl=16)
  assert artifact.lowered.plans[id(output_store(artifact))].outer_unroll == 1
  assert "#pragma clang loop unroll(disable)" in artifact.source
