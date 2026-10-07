"""Native masked fragments preserve grouped reduction banks and full schedules."""

from __future__ import annotations

import platform
import re

import numpy as np
import pytest
import vecops
import vecops.language as T
from dsl_test_utils import assert_no_vec_mask_arrays, make_compiler
from vecops._arm_vector_state import VectorState
from vecops.language._compiler.backends.cpu.backend import CPUBackend
from vecops.language._compiler.backends.cpu.cost import CPUTarget
from vecops.language._compiler.ir.vector import group_vectors


def reduction_template(
  source="float32", compute="float32", *, fma=False, unroll=2, exponential=False, extent=None, output=None
):
  N, AS, CS = T.dynamic(lower=0) if extent is None else extent, T.dynamic(), T.dynamic()
  output = compute if output is None else output
  positive_inf, negative_inf = float("inf"), float("-inf")

  @T.template
  def kernel(
    A: T.Tensor((N,), (AS,), source),
    Mask: T.Tensor((N,), dtype="float32"),
    C: T.Tensor((N,), (CS,), output, access="output"),
    Stats: T.Tensor((3,), dtype=compute, access="output"),
    branch: bool = True,
  ):
    x = T.load(A, dtype=compute)
    valid = T.load(Mask) > 0
    if fma:
      y = T.fma(x, T.cast(1.25, compute), T.cast(-0.125, compute))
    elif exponential:
      y = T.exp(x, accuracy="strict")
    else:
      y = x
    total = T.reduce_add(T.where(valid, y, T.cast(0, compute)), unroll=unroll)
    if exponential:
      minimum, maximum = T.cast(0, compute), T.cast(0, compute)
    else:
      minimum = T.reduce_min(T.where(valid, y, T.cast(positive_inf, compute)), unroll=unroll)
      maximum = T.reduce_max(T.where(valid, y, T.cast(negative_inf, compute)), unroll=unroll)
    Stats[0], Stats[1], Stats[2] = total, minimum, maximum
    if branch:
      T.store(C, T.where(valid, y, T.cast(0, compute)) + total)
    else:
      T.store(C, T.where(valid, y, T.cast(0, compute)) - total)

  return kernel


def lower(template, *, target="SVE", enabled=True, cost=None, power=1, vl=64, svl=32):
  options = {
    "_vector_state": VectorState(vl, svl),
    "cpu_native_word_tails": enabled,
    "cpu_math_chunk_pow": power,
    "cpu_reduction_chunk_pow": power,
  }
  if cost is not None:
    options["cpu_cost_target"] = cost
  return CPUBackend().lower(template.ir, (), target=target, options=options)


def full_blocks(source):
  lines = source.splitlines()
  for index, line in enumerate(lines):
    if "/* full vector groups */" not in line:
      continue
    balance, block = line.count("{") - line.count("}"), [line]
    for child in lines[index + 1 :]:
      block.append(child)
      balance += child.count("{") - child.count("}")
      if not balance:
        break
    assert not balance
    # Tail insertion can renumber later SSA locals. Preserve each local's
    # semantic stem and all repeated uses while alpha-renaming its suffix.
    names = {}

    def alpha_name(match, names=names):
      name = match.group()
      if name not in names:
        names[name] = len(names)
      return f"{match[1]}_{names[name]}"

    yield re.sub(r"\b(dsl_[A-Za-z0-9_]*?)_\d+\b", alpha_name, "\n".join(block))


@pytest.mark.parametrize(
  "source,compute",
  (
    ("float32", "float32"),
    ("float64", "float64"),
    ("float16", "float32"),
    ("float16", "float64"),
    ("float64", "float32"),
  ),
)
@pytest.mark.parametrize("power", (0, 2))
def test_native_tail_fragments_keep_full_groups_and_bank_lane_tree(source, compute, power):
  template = reduction_template(source, compute, exponential=True)
  ordinary = lower(template, enabled=False, power=power)
  artifact = lower(template, power=power)
  regions = [region for region in artifact.lowered.regions if region.native_tails]
  assert regions
  assert tuple(full_blocks(ordinary.source)) == tuple(full_blocks(artifact.source))
  for region in regions:
    assert region.unroll == 2
    assert "cpu-plan-native-word-tails" in region.pass_trace
    for variant, fragments in region.native_tails.items():
      assert all(fragment.power == -region.rebind_headroom for fragment in fragments)
      assert tuple(fragment.first for fragment in fragments) == tuple(range(len(fragments)))
      groups = region.groupings[variant] if variant is not None else None
      if groups is not None:
        for parent, group in enumerate(groups):
          paths = [fragment.path for fragment in fragments if fragment.parent == parent]
          assert len(paths) == 1 << (group.power + region.rebind_headroom)
          assert tuple(paths) == tuple(sorted(paths))
  assert "/* native-word masked remainder */" in artifact.source
  assert "vecops::vec::concat(" in artifact.source
  assert "vecops::vec::mwhilelt(" in artifact.source
  assert "float{0}" in artifact.source or "double{0}" in artifact.source
  assert_no_vec_mask_arrays(artifact.source)


@pytest.mark.parametrize("target,selected", (("SVE", True), ("SME", True), ("AVX512", False), ("Scalar", False)))
def test_native_tail_selection_uses_codegen_isa_not_cost_override(target, selected):
  cost = CPUTarget("x86", "avx512", vector_bits=512) if selected else CPUTarget("arm", "sme", vector_bits=512)
  artifact = lower(reduction_template(), target=target, cost=cost)
  assert any(region.native_tails for region in artifact.lowered.regions) is selected
  assert ("native-word masked remainder" in artifact.source) is selected


@pytest.mark.parametrize("vl,svl", ((16, 64), (32, 16), (64, 32)))
def test_native_tail_grouping_is_independent_of_ordinary_and_streaming_lengths(vl, svl):
  artifact = lower(reduction_template("float64", "float32"), target="SME", vl=vl, svl=svl)
  regions = [region for region in artifact.lowered.regions if region.native_tails]
  assert regions
  assert any(region.rebind_headroom == 1 for region in regions)
  assert all(
    fragment.power == -region.rebind_headroom
    for region in regions
    for parts in region.native_tails.values()
    for fragment in parts
  )
  assert "native-word masked remainder" in artifact.source
  assert_no_vec_mask_arrays(artifact.source)


def test_native_tail_keeps_min_max_identity_and_explicit_fma():
  artifact = lower(reduction_template(fma=True))
  assert "std::numeric_limits<float>::infinity()" in artifact.source
  assert "-std::numeric_limits<float>::infinity()" in artifact.source
  assert "vecops::vec::fmadd(" in artifact.source
  assert_no_vec_mask_arrays(artifact.source)


@pytest.mark.parametrize("unroll", (1, 2, 7, 17))
def test_native_tail_fragments_cover_each_original_bank_once(unroll):
  artifact = lower(reduction_template(unroll=unroll))
  for region in artifact.lowered.regions:
    for variant, fragments in region.native_tails.items():
      base_power = region.base_power if variant is not None else 0
      groups = region.groupings[variant] if variant is not None else group_vectors(region.unroll, 0, 0)
      intervals = []
      for parent, group in enumerate(groups):
        parts = [fragment for fragment in fragments if fragment.parent == parent]
        expected = 1 << (group.power + region.rebind_headroom)
        assert len(parts) == expected
        assert len({fragment.path for fragment in parts}) == expected
        intervals.extend(fragment.first for fragment in parts)
      assert tuple(intervals) == tuple(range(region.unroll << (base_power + region.rebind_headroom)))


@pytest.mark.parametrize("compute,output,power", (("float16", "float64", -2), ("float32", "float64", -1)))
def test_native_tail_legalizes_wider_store_memory_without_changing_full_groups(compute, output, power):
  template = reduction_template(compute, compute, output=output)
  ordinary, artifact = lower(template, enabled=False), lower(template)
  assert tuple(full_blocks(ordinary.source)) == tuple(full_blocks(artifact.source))
  fragments = [
    fragment
    for region in artifact.lowered.regions
    if any(effect.store is not None and effect.store.destination.storage == "C" for effect in region.effects)
    for group in region.native_tails.values()
    for fragment in group
  ]
  assert fragments
  assert all(fragment.power == power for fragment in fragments)
  assert_no_vec_mask_arrays(artifact.source)


def test_explicit_simd_state_retains_its_existing_extent_contract():
  @T.template
  def kernel(A: T.Tensor((37,), dtype="float32"), C: T.Tensor((37,), dtype="float32", access="output")):
    for i in T.simd(37, vector=T.Vector("float32", pow=2)):
      T.scatter(C, i, T.gather(A, i) * 2 + 1, unique=True)

  artifact = lower(kernel)
  assert not any(region.native_tails for region in artifact.lowered.regions)
  assert "native-word masked remainder" not in artifact.source
  assert_no_vec_mask_arrays(artifact.source)


@pytest.mark.parametrize("invalid", (0, 1, "yes", None))
def test_native_tail_option_requires_bool(invalid):
  with pytest.raises(ValueError, match="cpu_native_word_tails must be bool"):
    lower(reduction_template(), enabled=invalid)


NATIVE_CASES = (
  ("float32", "float32", "float32", False),
  ("float64", "float64", "float64", False),
  ("float16", "float32", "float32", False),
  ("float16", "float64", "float64", False),
  ("float64", "float32", "float32", True),
  ("float32", "float64", "float64", True),
  ("float32", "float32", "float64", False),
  ("float16", "float16", "float64", False),
  ("float64", "float32", "float64", True),
)


@pytest.fixture(scope="module")
def native_tail_pairs(tmp_path_factory):
  root = tmp_path_factory.mktemp("native-word-tails")
  compiler = make_compiler(jobs=4, cache_dir=root / "cache", build_dir=root / "build")
  pairs = {}
  with vecops.collect_compile_requests() as collector:
    for source, compute, output, fma in NATIVE_CASES:
      pair = []
      for enabled in (False, True):
        kernel = T.compile(
          reduction_template(source, compute, output=output, fma=fma),
          compiler=compiler,
          pass_options={"cpu_native_word_tails": enabled},
        )
        kernel.operator(
          vecops.TensorMeta((129,), strides=(1,), dtype=source),
          vecops.TensorMeta((129,), dtype="float32"),
          vecops.TensorMeta((129,), strides=(1,), dtype=output),
          vecops.TensorMeta((3,), dtype=compute),
          True,
          **kernel._core.specialization_values,
        )
        pair.append(kernel)
      pairs[source, compute, output, fma] = pair
  vecops.compile_batch(collector.requests, parallelism=4, execution_parallelism=1)
  return pairs


def assert_same_float_bits(actual, expected, *, context=""):
  assert actual.dtype == expected.dtype
  np.testing.assert_array_equal(np.isnan(actual), np.isnan(expected), err_msg=context)
  valid = ~np.isnan(expected)
  unsigned = np.dtype(f"uint{actual.dtype.itemsize * 8}")
  np.testing.assert_array_equal(actual[valid].view(unsigned), expected[valid].view(unsigned), err_msg=context)


@pytest.mark.parametrize("source,compute,output,fma", NATIVE_CASES)
def test_native_fragments_preserve_bits_specials_masks_branch_and_alias(
  native_tail_pairs, source, compute, output, fma
):
  pair = native_tail_pairs[source, compute, output, fma]
  for width in (0, 1, 7, 15, 16, 17, 31, 32, 33, 63, 64, 65, 127, 128, 129, 257, 1000):
    for layout in ("dense", "negative", "strided", "inplace", "shifted"):
      if layout in {"inplace", "shifted"} and source != output:
        continue
      original = np.resize(np.array((1, -1, 2**-12, -0.0, 0.0, 3, -2, 0.125), dtype=source), width)
      mask = np.ones(width, dtype="float32")
      mask[::3] = 0
      specials = (False, True, "zero") if width in {1, 17, 33} else (False, True)
      for special in specials:
        values = original.copy()
        if special == "zero":
          values = np.resize(np.array((-0.0, 0.0), dtype=source), width)
        elif special and width:
          values[0] = np.nan
          if width > 3:
            values[1], values[3] = np.inf, -np.inf
          if width > 2:
            values[2] = np.nan  # Active NaN, unlike the deliberately masked lane 0.
        for branch in (False, True):
          outputs = []
          for kernel in pair:
            storage = np.full(width * 2 + 3, -17, dtype=source)
            a = storage[1 : width + 1]
            if layout == "strided":
              a = storage[1 : 2 * width + 1 : 2]
            elif layout == "negative":
              a = a[::-1]
            a[...] = values
            c = np.empty(width * 2 + 3, dtype=output)[1 : width + 1]
            if layout == "strided":
              c = np.empty(width * 2 + 3, dtype=output)[1 : 2 * width + 1 : 2]
            elif layout == "negative":
              c = c[::-1]
            elif layout == "inplace":
              c = a
            elif layout == "shifted":
              c = storage[2 : width + 2]
            stats = np.empty(3, dtype=compute)
            kernel.operator.native(a, mask, c, stats, branch, **kernel._core.specialization_values)
            outputs.append((c.copy(), stats.copy()))
          for component, (actual, expected) in enumerate(zip(outputs[1], outputs[0])):
            context = (
              f"source={source}, compute={compute}, output={output}, fma={fma}, "
              f"width={width}, layout={layout}, special={special}, branch={branch}, component={component}"
            )
            assert_same_float_bits(actual, expected, context=context)
          if not width:
            np.testing.assert_array_equal(outputs[1][1], (0, np.inf, -np.inf))
  assert_no_vec_mask_arrays(pair[1].generated_source)
  assert not re.search(r"std::(?:array|vector)\s*<\s*vecops::vec::(?:Vec|Mask)", pair[1].generated_source)
  if platform.machine().lower() in {"aarch64", "arm64"}:
    assert "native-word masked remainder" in pair[1].generated_source
    assert "native-word masked remainder" not in pair[0].generated_source
