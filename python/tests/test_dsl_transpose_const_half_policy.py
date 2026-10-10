# SPDX-FileCopyrightText: 2026 HenryRenYz and vecops contributors
# SPDX-License-Identifier: MIT
"""Default policy selection is distinct from transpose instruction legality."""

import pytest
from dsl_test_utils import assert_no_vec_mask_arrays
from test_dsl_transpose_const_half_full_input import OPTION, OUTPUT_OPTION, template
from vecops._arm_vector_state import VectorState
from vecops.language._compiler.backends.cpu.backend import CPUBackend
from vecops.language._compiler.cost.model import CostProfile


def artifact(*, vl=64, svl=64, target="SME", kernel=None, **options):
  return CPUBackend().lower(
    (template() if kernel is None else kernel).ir,
    (),
    target=target,
    options={"_vector_state": VectorState(vl, svl), **options},
  )


def lower(**options):
  return artifact(**options).source


def selected(**options):
  return tuple(
    any(getattr(plan, field) is not None for plan in artifact(**options).backend_ir.transposes.values())
    for field in ("sme_const_half_full_input_rows", "sme_const_half_output_columns")
  )


@pytest.mark.parametrize("vl,svl", ((16, 16), (64, 16), (16, 32), (16, 64), (64, 64), (16, 128), (64, 256)))
def test_default_matches_the_exact_explicit_combined_schedule(vl, svl):
  default = lower(vl=vl, svl=svl)
  assert default == lower(vl=vl, svl=svl, **{OPTION: True, OUTPUT_OPTION: True})
  assert selected(vl=vl, svl=svl) == (svl <= 64, svl <= 64)
  assert_no_vec_mask_arrays(default)


def test_profile_can_disable_both_and_explicit_options_still_override_it():
  profile = CostProfile("disable-const-half-transpose", policies=(("sme_const_half_transpose", 0),))
  options = {"cost_profiles": (profile,)}
  disabled = lower(**options)
  assert selected(**options) == (False, False)
  assert disabled == lower(**options, **{OPTION: False, OUTPUT_OPTION: False})
  overridden, automatic = lower(**options, **{OPTION: True, OUTPUT_OPTION: True}), lower()
  # Different immutable cost snapshots must have different cache fingerprints.
  # The single diagnostic identity comment is not a computation difference.
  prefix = "// compiler environment: "
  assert overridden.startswith(prefix) and automatic.startswith(prefix)
  assert overridden.partition("\n")[0] != automatic.partition("\n")[0]
  assert overridden.partition("\n")[2] == automatic.partition("\n")[2]


def test_output_disable_also_disables_implicit_input_but_overrides_are_independent():
  assert lower(**{OUTPUT_OPTION: False}) == lower(**{OPTION: False, OUTPUT_OPTION: False})
  assert selected(**{OPTION: False}) == (False, True)
  assert selected(**{OUTPUT_OPTION: False, OPTION: True}) == (True, False)


@pytest.mark.parametrize("value", (True, 2, 0.5, 1.0))
def test_default_policy_requires_an_exact_binary_integer(value):
  profile = CostProfile("invalid-const-half-transpose", policies=(("sme_const_half_transpose", value),))
  with pytest.raises(ValueError, match="policy must be 0 or 1"):
    lower(cost_profiles=(profile,))


def test_negative_policy_is_rejected_by_the_profile_before_lowering():
  with pytest.raises(ValueError, match="finite and nonnegative"):
    CostProfile("negative-const-half-transpose", policies=(("sme_const_half_transpose", -1),))


@pytest.mark.parametrize("option", (OPTION, OUTPUT_OPTION))
@pytest.mark.parametrize("value", (None, 0, 1, "true"))
def test_explicit_options_remain_strict_booleans(option, value):
  with pytest.raises(TypeError, match=option + " must be bool"):
    lower(**{option: value})


@pytest.mark.parametrize(
  "arguments",
  (
    {"svl": None},
    {"svl": 128},
    {"target": "AVX512", "cpu_cost_target": "SME"},
    {"kernel": template(types=("float64", "float16", "float64"))},
    {"kernel": template(source_pitch=None, output_pitch=None)},
    {"kernel": template(transformed=True)},
  ),
)
def test_profile_never_supplies_isa_geometry_dtype_or_expression_legality(arguments):
  automatic = lower(**arguments)
  assert selected(**arguments) == (False, False)
  assert automatic == lower(**arguments, **{OPTION: False, OUTPUT_OPTION: False})
