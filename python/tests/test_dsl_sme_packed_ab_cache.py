# SPDX-FileCopyrightText: 2026 HenryRenYz and vecops contributors
# SPDX-License-Identifier: MIT
"""Whole-problem prepacked traversal: bounded, profile-selected and SVL-relative."""

import re
import subprocess

import numpy as np
import pytest
import vecops
import vecops.language as T
from dsl_test_utils import assert_no_vec_mask_arrays, cpp_compile_command, make_compiler
from test_dsl_matmul_sve_optimized import _cross
from test_dsl_sme_packed_ab_col import _OPTIONS, _count, _geometry, _output, _pack_template, _template, _wrapper
from test_dsl_sve_packed_mmla import _sme_available
from vecops._arm_vector_state import VectorState
from vecops.language._compiler.backends.cpu.backend import CPUBackend
from vecops.language._compiler.backends.cpu.cost.sme import PackedABTraversalPolicy
from vecops.language._compiler.cost.model import CostProfile
from vecops.language._compiler.environment import freeze

_OPTION = "cpu_sme_packed_ab_cache_bypass"
_MARKER = "/* SME packed-AB whole-problem geometry */"


def _lower(*, enabled=None, vl=64, svl=64, cost_profiles=None, **kwargs):
  options = {**_OPTIONS, "_vector_state": VectorState(vl, svl)}
  if enabled is not None:
    options[_OPTION] = enabled
  if cost_profiles is not None:
    options["cost_profiles"] = cost_profiles
  return CPUBackend().lower(_template(**kwargs).ir, (), target="SME", options=options).source


@pytest.mark.parametrize("svl", (16, 32, 64, 128, 256))
def test_policy_geometry_and_cache_budget_follow_streaming_length(svl):
  policy = PackedABTraversalPolicy()
  lanes = svl // 4
  shape = (8 * lanes, 8 * lanes, 513)
  exact_budget = 514 * 4 * lanes * 2 * policy.private_cache_divisor
  assert policy.profitable(*shape, lanes=lanes, k_group=2, element_bytes=2, l2_bytes=exact_budget)
  assert not policy.profitable(*shape, lanes=lanes, k_group=2, element_bytes=2, l2_bytes=exact_budget - 1)
  for m, n, k in ((0, shape[1], 513), (8 * lanes + 1, shape[1], 513), (lanes, 8 * lanes - 1, 513), (lanes, 512, 511)):
    assert not policy.profitable(m, n, k, lanes=lanes, k_group=2, element_bytes=2, l2_bytes=1 << 30)
  assert not policy.profitable(128, 4096, 2**63 - 1, lanes=lanes, k_group=2, element_bytes=2, l2_bytes=2**63 - 1)


def test_policy_overlays_are_central_immutable_and_validated():
  model = freeze(
    {"cost_profiles": (CostProfile("packed-ab-test", policies=(("sme_packed_ab_maximum_row_panels", 4),)),)},
    target="SME",
  )
  policy = PackedABTraversalPolicy.from_costs(model.costs)
  assert policy.maximum_row_panels == 4 and PackedABTraversalPolicy().maximum_row_panels == 8
  with pytest.raises(AttributeError):
    policy.maximum_row_panels = 5
  for value in (True, 0, -1, 2.5, 2**21):
    with pytest.raises(ValueError, match="integer"):
      PackedABTraversalPolicy(private_cache_divisor=value)


def test_private_l2_budget_admits_measured_streaming_panel_without_unbounded_bypass():
  policy = PackedABTraversalPolicy()
  assert policy.private_cache_divisor == 1
  geometry = {"lanes": 16, "k_group": 2, "element_bytes": 2, "l2_bytes": 768 * 1024}
  assert policy.profitable(128, 4096, 4096, **geometry)
  assert not PackedABTraversalPolicy(private_cache_divisor=2).profitable(128, 4096, 4096, **geometry)
  # Exact residency boundary includes padding the last BF16 K pair.
  assert policy.profitable(128, 4096, 6144, **geometry)
  assert not policy.profitable(128, 4096, 6145, **geometry)
  assert not policy.profitable(129, 4096, 4096, **geometry)
  assert not policy.profitable(128, 4096, 8192, **geometry)


@pytest.mark.parametrize("shape", (None, (128, 4096, 4096), (16, 32, 5)))
@pytest.mark.parametrize("level", ("dot", "matmul", "blocked"))
def test_default_and_enabled_source_are_byte_identical(shape, level):
  assert _lower(shape=shape, level=level) == _lower(shape=shape, level=level, enabled=True)


def test_target_profile_controls_default_but_explicit_option_takes_precedence():
  profile = CostProfile("disable-packed-ab", policies=(("sme_packed_ab_cache_bypass", 0),))
  kwargs = {"shape": (128, 4096, 4096), "level": "matmul", "cost_profiles": (profile,)}
  assert _MARKER in _lower(shape=kwargs["shape"], level=kwargs["level"])
  assert _MARKER not in _lower(**kwargs)
  assert _lower(**kwargs) == _lower(**kwargs, enabled=False)
  assert _MARKER in _lower(**kwargs, enabled=True)


@pytest.mark.parametrize("value", (True, 2, 0.5, 1.0))
def test_target_profile_default_is_a_validated_binary_policy(value):
  profile = CostProfile("bad-packed-ab", policies=(("sme_packed_ab_cache_bypass", value),))
  with pytest.raises(ValueError, match="policy must be 0 or 1"):
    _lower(level="matmul", cost_profiles=(profile,))


@pytest.mark.parametrize(
  "kwargs",
  (
    {"level": "dot"},
    {"level": "blocked"},
    {"level": "matmul", "dtype": "float16"},
    {"level": "matmul", "implementation": "vector"},
    {"level": "matmul", "shape": (16, 32, 5)},
    {"level": "matmul", "shape": (129, 4096, 4096)},
  ),
)
def test_ineligible_lowering_retains_exact_old_source(kwargs):
  assert _lower(**kwargs, enabled=True) == _lower(**kwargs, enabled=False)


@pytest.mark.parametrize("value", (1, "auto", None, (), []))
def test_option_is_a_strict_boolean(value):
  with pytest.raises(ValueError, match="must be bool"):
    CPUBackend().lower(_template().ir, (), target="SME", options={_OPTION: value})


@pytest.mark.parametrize("vl,svl", ((16, 64), (64, 16), (64, 64), (64, 128)))
def test_runtime_gate_precedes_owner_and_retains_fused_fallback(vl, svl):
  source = _lower(enabled=True, level="matmul", vl=vl, svl=svl)
  assert _MARKER in source
  guard = source.index(_MARKER)
  owner = source.index(".with_resources(vecops::execution::sme::Region{}", guard)
  assert guard < source.index("cache_info().l2_bytes", guard) < owner
  assert "complete padded two-by-two micro-panel" in source
  assert "=vecops_dsl::matmul_blocks" in source
  assert "epilogue exactly once" in source and "prologue exactly once" in source
  assert "sme::mopa" in source and "sme::read_hor" in source
  assert_no_vec_mask_arrays(source)


def test_actual_emitted_quotient_predicate_matches_python_under_ubsan(tmp_path):
  source = _lower(enabled=True, level="matmul", vl=None, svl=None)
  start = source.index("const vecops::nint_t dsl_c_sme_packed_ab_lanes_")
  end = source.index("/* complete padded two-by-two micro-panel stays in private L2 */", start)
  snippet = source[start : end + len("/* complete padded two-by-two micro-panel stays in private L2 */")] + "\n}"
  lanes_name = re.search(r"const vecops::nint_t (\w+)=", snippet)[1]
  selected = re.search(r"bool (\w+)=false;", snippet)[1]
  condition = re.search(r"if \(\(([^)]+)\)>0.*&& \(([^)]+)\)/\w+>=8 && \(([^)]+)\)/2>=256", snippet)[0]
  dimensions = re.search(r"if \(\(([^)]+)\)>0.*&& \(([^)]+)\)/\w+>=8 && \(([^)]+)\)/2>=256", condition).groups()
  for expression, variable in zip(dimensions, ("m", "n", "k")):
    snippet = snippet.replace(expression, variable)
  snippet = snippet.replace("vecops::vec::sme::streaming_lanes<float>()", "lanes")
  snippet = snippet.replace("vecops::platform::cache_info().l2_bytes", "l2")
  assert lanes_name in snippet
  cpp, executable = tmp_path / "packed_cost.cpp", tmp_path / "packed_cost"
  cpp.write_text(
    "#include <iostream>\nnamespace vecops { using nint_t=long long; }\n"
    f"int main() {{ long long m,n,k,lanes,l2; while(std::cin>>m>>n>>k>>lanes>>l2) {{ {snippet}\n"
    f"std::cout<<{selected}<<'\\n'; }} }}\n",
    encoding="utf-8",
  )
  command = cpp_compile_command(
    cpp, executable, flags=("-std=c++20", "-O2", "-fsanitize=undefined", "-fno-sanitize-recover=all")
  )
  compiled = subprocess.run(command, capture_output=True, text=True, check=False)
  assert compiled.returncode == 0, compiled.stderr[-4000:]
  rng = np.random.default_rng(822)
  cases = [
    (int(m), int(n), int(k), lanes, l2)
    for m, n, k in rng.integers(0, 8192, size=(128, 3))
    for lanes in (4, 16, 64)
    for l2 in (1 << 17, 1 << 20, 2**63 - 1)
  ]
  cases += [(128, 4096, k, 16, 2**63 - 1) for k in (511, 512, 513, 2**63 - 1)]
  result = subprocess.run(
    [str(executable)],
    input="\n".join(" ".join(map(str, case)) for case in cases),
    capture_output=True,
    text=True,
    check=False,
  )
  assert result.returncode == 0, result.stderr
  expected = [
    int(PackedABTraversalPolicy().profitable(m, n, k, lanes=lanes, k_group=2, element_bytes=2, l2_bytes=l2))
    for m, n, k, lanes, l2 in cases
  ]
  assert list(map(int, result.stdout.split())) == expected


@pytest.mark.parametrize("shape,vl,svl", (((128, 4096, 4096), 64, 64), (None, 16, 64), (None, 64, 16)))
def test_real_cross_prepacked_whole_problem_retains_mopa_and_named_vectors(tmp_path, shape, vl, svl):
  source = _lower(enabled=True, shape=shape, level="matmul", vl=vl, svl=svl)
  assembly = _cross(tmp_path, source, lambda value: _wrapper(value, shape=shape), vl=vl, svl=svl)
  assert re.search(r"^\s*bfmopa\b", assembly, re.MULTILINE)
  assert _MARKER in source
  assert_no_vec_mask_arrays(source)


@pytest.mark.skipif(not _sme_available(), reason="actual ARM SME required")
def test_native_prepacked_cache_bypass_full_tail_fusion_alias_and_fallback(tmp_path):
  """Independent exact dyadic oracle, including zero-K and boundary rejects."""
  pytest.importorskip("ml_dtypes")
  lanes, panel, group = _geometry()
  shape = (2 * lanes, 8 * lanes + 1, 513)
  compiler = make_compiler(target="Native", jobs=4, cache_dir=tmp_path / "cache", build_dir=tmp_path / "build")
  pack = T.compile(_pack_template(), compiler=compiler, pass_options=_OPTIONS)
  kernels = {
    "control": T.compile(_template(level="matmul"), compiler=compiler, pass_options={**_OPTIONS, _OPTION: False}),
    "dynamic": T.compile(_template(level="matmul"), compiler=compiler, pass_options={**_OPTIONS, _OPTION: True}),
    "constant": T.compile(
      _template(shape=shape, level="matmul"), compiler=compiler, pass_options={**_OPTIONS, _OPTION: True}
    ),
  }
  m, n, k = shape
  psa, psb = _count(m, k, panel, group), _count(n, k, panel, group)
  with vecops.collect_compile_requests() as collector:
    pack.operator(
      vecops.TensorMeta((m, k), dtype="float32"),
      vecops.TensorMeta((k, n), dtype="bfloat16"),
      vecops.TensorMeta((psa,), dtype="bfloat16"),
      vecops.TensorMeta((psb,), dtype="bfloat16"),
      **pack._core.specialization_values,
    )
    for kernel in kernels.values():
      kernel.operator(
        vecops.TensorMeta((m, k), dtype="float32"),
        vecops.TensorMeta((psa,), dtype="bfloat16"),
        vecops.TensorMeta((psb,), dtype="bfloat16"),
        vecops.TensorMeta((m, n), dtype="float32"),
        vecops.TensorMeta((m, n), dtype="float32"),
        vecops.TensorMeta((1, n), dtype="float32"),
        **kernel._core.specialization_values,
      )
  assert len(collector.requests) == 4
  vecops.compile_batch(collector.requests, parallelism=4, execution_parallelism=1)
  assert _MARKER in kernels["constant"].generated_source
  rng = np.random.default_rng(629)
  shapes = (shape, (8 * lanes + 1, n, k), (lanes, 8 * lanes - 1, k), (m, n, 511), (m, n, 0), (0, n, k), (m, 0, k))
  calls = 0
  for m, n, k in shapes:
    for negative in (False, True):
      a = rng.integers(-4, 5, size=(m, k)).astype("float32") / np.float32(16)
      b = (rng.integers(-3, 4, size=(k, n)) / 8).astype("bfloat16")
      if negative:
        a, b = a[:, ::-1], b[::-1]
      ap = np.full(_count(m, k, panel, group), 7, dtype="bfloat16")
      bp = np.full(_count(n, k, panel, group), 7, dtype="bfloat16")
      pack.operator.native(a, b, ap, bp, **pack._core.specialization_values)
      transformed = (a.copy() * np.float32(0.5) + np.float32(0.125)).astype("bfloat16").astype("float32")
      product = transformed @ b.astype("float32")
      for prior_mode in ("disjoint", "alias", "broadcast"):
        prior = _output(1 if prior_mode == "broadcast" else m, n, negative=negative)
        prior[:] = rng.integers(-2, 3, size=prior.shape) / 8
        bias = np.full((1, n), np.float32(0.125))
        expected = np.maximum(((product + prior.copy()) + bias) * np.float32(0.5) + np.float32(0.125), 0)
        for label, kernel in kernels.items():
          if label == "constant" and (m, n, k) != shape:
            continue
          d = prior.copy() if prior_mode != "alias" else _output(m, n, negative=negative)
          d[:] = prior
          c = d if prior_mode == "alias" else _output(m, n, negative=negative)
          physical = c.base.copy()
          kernel.operator.native(a, ap, bp, d, c, bias, **kernel._core.specialization_values)
          np.testing.assert_array_equal(c, expected, err_msg=f"{label}/{(m, n, k)}/{negative}/{prior_mode}")
          active = np.zeros(c.base.size, dtype=bool)
          active[[i * (2 * n + 3) + 2 * j for i in range(m) for j in range(n)]] = True
          np.testing.assert_array_equal(c.base[~active], physical[~active])
          calls += 1
  assert calls == 90
