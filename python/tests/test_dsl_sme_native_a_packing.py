# SPDX-FileCopyrightText: 2026 HenryRenYz and vecops contributors
# SPDX-License-Identifier: MIT
"""NC-local native A auto packing; strict semantics and pre-K dispatch."""

import re
import subprocess

import numpy as np
import pytest
import vecops.language as T
from dsl_test_utils import assert_no_vec_mask_arrays, cpp_compile_command, make_compiler
from test_dsl_matmul_sve_optimized import _cross, _strided_output
from test_dsl_sve_packed_mmla import _body, _sme_available
from vecops._arm_vector_state import VectorState
from vecops.language._compiler.backends.cpu.backend import CPUBackend
from vecops.language._compiler.backends.cpu.cost.sme import NativeAPackingPolicy
from vecops.language._compiler.cost.model import CostProfile
from vecops.language._compiler.environment import freeze


def _template(*, shape=None, library=True, packing="auto", bm=None, bn=None, bk=None, row_a=False, dtype="bfloat16"):
  M, N, K = tuple(T.Const(v) for v in shape) if shape else tuple(T.Dynamic(name=n, lower=0) for n in ("M", "N", "K"))
  A0, C0, C1, D0, D1 = (T.dynamic() for _ in range(5))
  if row_a:
    A0 = K
  acc_dtype = "float64" if dtype == "float64" else "float32"

  @T.template
  def native_a(
    A: T.Tensor((M, K), (A0, 1), dtype),
    B: T.Tensor((K, N), (1, K), dtype),
    C: T.Tensor((M, N), (C0, C1), acc_dtype, access="output"),
    D: T.Tensor((M, N), (D0, D1), acc_dtype),
    Bias: T.Tensor((1, N), dtype=acc_dtype),
  ):
    product = (
      T.library.matmul((T.load(A), T.load(B)), C, acc=T.load(D), packing=packing, swap=False, BM=bm, BN=bn, BK=bk)
      if library
      else T.dot((T.load(A), T.load(B)), C, acc=T.load(D))
    )
    T.store(C, T.max((product + T.load(Bias)) * 0.5 + 0.125, 0))

  return native_a


def _lower(*, vl=64, svl=64, **kwargs):
  result = CPUBackend().lower(
    _template(**kwargs).ir,
    (),
    target="SME",
    options={"cpu_matmul_pipeline": "off", "_vector_state": VectorState(vl, svl)},
  )
  return result


@pytest.mark.parametrize(
  "shape,expected",
  (
    ((128, 256, 768), True),
    ((1, 1152, 896), False),
    ((2, 1024, 768), False),
    ((16, 32, 768), False),
    ((128, 256, 1), False),
    ((8, 256, 768), True),
    ((128, 32, 768), False),
    ((0, 256, 768), False),
    ((128, 256, 0), False),
  ),
)
def test_native_a_cost_uses_padding_work_and_actual_nc_reuse(shape, expected):
  assert NativeAPackingPolicy().profitable(*shape, panel=32, lanes=16, k_group=2, element_bytes=2) is expected


def test_native_a_cost_rejects_overflow_and_invalid_policies():
  policy = NativeAPackingPolicy()
  for shape in ((2**63 - 1, 256, 768), (128, 2**63 - 1, 768), (128, 256, 2**63 - 1)):
    assert not policy.profitable(*shape, panel=32, lanes=16, k_group=2, element_bytes=2)
  with pytest.raises(ValueError, match="integer"):
    NativeAPackingPolicy(gather_cycles=4.0)
  with pytest.raises(ValueError, match="gather penalty"):
    NativeAPackingPolicy(gather_cycles=1)


def test_native_a_cost_overlay_is_central_and_immutable():
  model = freeze(
    {"cost_profiles": (CostProfile("native-a-test", policies=(("sme_native_a_pack_gather_cycles", 2),)),)}, target="SME"
  )
  policy = NativeAPackingPolicy.from_costs(model.costs)
  assert policy.gather_cycles == 2
  assert NativeAPackingPolicy().gather_cycles == 4
  with pytest.raises(AttributeError):
    policy.gather_cycles = 5


@pytest.mark.parametrize(
  "dtype,panel,lanes,k_group,element_bytes", (("bfloat16", 32, 16, 2, 2), ("float64", 16, 8, 1, 8))
)
def test_emitted_cost_predicate_matches_python_with_ubsan(tmp_path, dtype, panel, lanes, k_group, element_bytes):
  # Exercise the actual generated cost arithmetic, without pretending an x86
  # process executes SME. Only its public geometry queries become constants.
  source = _lower(bm=64, bn=256, bk=65, dtype=dtype).source
  match = re.search(r"bool (dsl_c_sme_native_a_pack_profitable_\d+)=false;", source)
  assert match
  enabled = match[1]
  end = source.index(f"if ({enabled}) /* pack A only", match.start())
  snippet = source[match.start() : end]
  dimensions = re.findall(r"const __int128 dsl_c_sme_pack_cost_([mnk])_\d+=\(([^\n]+)\);", snippet)
  assert [axis for axis, _ in dimensions] == ["m", "n", "k"]
  stride = re.search(r"&& \((\w+)\)!=0", snippet)[1]
  # Slicing/broadcast preservation can make an extent a C++ expression,
  # rather than one generated identifier. Bind its actual scalar inputs to
  # the corresponding test extent without replacing or reimplementing the
  # generated expression or any of the cost arithmetic under UBSan.
  bindings = {}
  for axis, expression in dimensions:
    identifiers = set(re.findall(r"\bdsl_c_\w+\b", expression))
    assert identifiers, expression
    assert not identifiers & bindings.keys(), "cost dimensions unexpectedly share scalar inputs"
    assert stride not in identifiers
    bindings.update((name, axis) for name in identifiers)
  declarations = "\n".join(f"const auto {name}={value};" for name, value in sorted(bindings.items()))
  cpp_type, acc_type = ("double", "double") if dtype == "float64" else ("vecops::bfloat16_t", "float")
  snippet = snippet.replace(f"vecops_dsl::pack_panel<{cpp_type},2>()", str(panel))
  snippet = snippet.replace(f"vecops::vec::sme::streaming_lanes<{acc_type}>()", str(lanes))
  cpp, executable = tmp_path / "packing_cost.cpp", tmp_path / "packing_cost"
  cpp.write_text(
    "#include <cstdint>\n#include <iostream>\n#include <limits>\nnamespace vecops { using nint_t=long long; }\n"
    f"int main() {{ long long m,n,k,as; while(std::cin>>m>>n>>k>>as) {{ {declarations}\n"
    f"const auto {stride}=as;\n{snippet}\nstd::cout<<{enabled}<<'\\n'; }} }}\n",
    encoding="utf-8",
  )
  command = cpp_compile_command(
    cpp, executable, flags=("-std=c++20", "-O2", "-fsanitize=undefined", "-fno-sanitize-recover=all")
  )
  compiled = subprocess.run(command, capture_output=True, text=True, check=False)
  assert compiled.returncode == 0, compiled.stderr[-4000:]
  rng = np.random.default_rng(222)
  shapes = [(int(m), int(n), int(k)) for m, n, k in rng.integers(1, 2048, size=(128, 3))]
  shapes += [(2**63 - 1, 256, 768), (128, 2**63 - 1, 768), (128, 256, 2**63 - 1), (0, 256, 257), (1, 256, 257)]
  shapes += [(16, 32, 4096), (24, 32, 4096), (24, 64, 4096), (24, 96, 4096), (32, 32, 4096), (32, 48, 4096)]
  cases = [(*shape, stride) for shape in shapes for stride in (0, 1, 17, -17, -(2**63))]
  result = subprocess.run(
    [str(executable)],
    input="\n".join(" ".join(map(str, c)) for c in cases),
    capture_output=True,
    text=True,
    check=False,
  )
  assert result.returncode == 0, result.stderr
  policy = NativeAPackingPolicy()
  expected = [
    int(
      stride not in (0, 1)
      and policy.profitable(m, n, k, panel=panel, lanes=lanes, k_group=k_group, element_bytes=element_bytes)
    )
    for m, n, k, stride in cases
  ]
  assert list(map(int, result.stdout.split())) == expected


@pytest.mark.parametrize(
  "m,n,expected", ((16, 32, False), (24, 32, False), (24, 64, False), (24, 96, True), (32, 32, False), (32, 48, True))
)
def test_native_a_cost_uses_actual_fp64_expanded_column_family(m, n, expected):
  assert NativeAPackingPolicy().profitable(m, n, 4096, panel=16, lanes=8, k_group=1, element_bytes=8) is expected


@pytest.mark.parametrize("shape", ((1, 1152, 896), (2, 4, 257), (16, 32, 65)))
def test_known_small_native_a_retains_no_extra_scratch_or_dispatch(shape):
  source = _lower(shape=shape).source
  assert "sme_native_a_pack_profitable" not in source
  assert "native A packing dispatch outside K" not in source


@pytest.mark.parametrize("packing", ("none", "B", "A", "AB"))
def test_explicit_packing_requests_do_not_use_native_a_cost_policy(packing):
  source = _lower(packing=packing).source
  assert "sme_native_a_pack_profitable" not in source


def test_dot_does_not_auto_pack_and_dynamic_library_dispatches_before_k():
  assert "sme_native_a_pack_profitable" not in _lower(library=False).source
  source = _lower().source
  assert "NC-local gather versus padded pack cost" in source
  assert "native A packing dispatch outside K" in source
  # The only dynamic choice is before K; neither variant tests it in K.
  for marker in re.finditer(r"/\* native A packing dispatch outside K \*/", source):
    first = _body(source[marker.start() :], "/* native A packing dispatch outside K */")
    assert "sme_native_a_pack_profitable" not in first
    assert "full K groups" in first and "sme::mopa" in first
  assert "raw A: no scratch load or allocation" in source
  assert_no_vec_mask_arrays(source)


def test_cache_native_a_cost_counts_nc_not_original_full_n():
  source = _lower(bm=64, bn=128, bk=65).source
  costs = [line for line in source.splitlines() if re.search(r"const __int128 dsl_c_sme_pack_cost_n_\d+=", line)]
  assert costs and all("panel_n" in line for line in costs)
  assert "epilogue exactly once" in source and "prologue exactly once" in source


def test_optional_packed_a_never_bypasses_decision_via_ordinary_mmla():
  source = _lower(bm=64, bn=256, bk=65).source
  assert "native A packing dispatch outside K" in source
  assert "ordinary-SVE PackedDot MMLA" not in source
  # Explicit, unconditional packed buffers retain the ordinary-SVE ladder.
  assert "ordinary-SVE PackedDot MMLA" in _lower(packing="AB", bm=64, bn=256, bk=65).source


def test_exact_native_a_decision_folds_to_one_packed_path():
  source = _lower(shape=(64, 256, 129), row_a=True).source
  assert "pack A: input transform" in source
  assert "sme_native_a_pack_profitable" not in source
  assert "native A packing dispatch outside K" not in source


def _wrapper(source, *, dtype="bfloat16"):
  input_type, acc_type = ("double", "double") if dtype == "float64" else ("vecops::bfloat16_t", "float")
  definitions = (
    '#include "vecops/Kernel.h"',
    "namespace vecops::spec { inline constexpr auto Parallelism = vecops::meta::cint<1>; }",
    source,
    'extern "C" void native_a(void* ap,void* bp,void* cp,void* dp,void* xp,vecops::nint_t m,vecops::nint_t n,vecops::nint_t k,vecops::nint_t as) {',
    "auto M=vecops::meta::Any{m},N=vecops::meta::Any{n},K=vecops::meta::Any{k};",
    f"auto A=vecops::tensor::make_tensor(static_cast<{input_type}*>(ap),vecops::tensor::make_shape(M,K),vecops::tensor::make_strides(vecops::meta::Any{{as}},vecops::meta::cint<1>));",
    f"auto B=vecops::tensor::make_tensor(static_cast<{input_type}*>(bp),vecops::tensor::make_shape(K,N),vecops::tensor::make_strides(vecops::meta::cint<1>,K));",
    f"auto C=vecops::tensor::make_tensor(static_cast<{acc_type}*>(cp),vecops::tensor::make_shape(M,N),vecops::tensor::make_strides(N,vecops::meta::cint<1>));",
    f"auto D=vecops::tensor::make_tensor(static_cast<{acc_type}*>(dp),vecops::tensor::make_shape(M,N),vecops::tensor::make_strides(N,vecops::meta::cint<1>));",
    f"auto Bias=vecops::tensor::make_tensor(static_cast<{acc_type}*>(xp),vecops::tensor::make_shape(vecops::meta::cint<1>,N),vecops::tensor::make_strides(N,vecops::meta::cint<1>));",
    "__kernel__(A,B,C,D,Bias);}",
  )
  return "\n".join(definitions)


@pytest.mark.parametrize("vl,svl", ((16, 16), (32, 32), (64, 64), (16, 64), (64, 16)))
def test_native_a_arm_cross_keeps_two_specialized_k_paths(tmp_path, vl, svl):
  source = _lower(vl=vl, svl=svl, bm=64, bn=256, bk=65).source
  assembly = _cross(tmp_path, source, _wrapper, vl=vl, svl=svl)
  assert re.search(r"^\s*bfmopa\b", assembly, re.MULTILINE)
  assert re.search(r"^\s*ld1w[^\n]*sxtw", assembly, re.MULTILINE)
  # Non-extending halfword register loads are contiguous; indexed halfword
  # gathers extend into .s lanes. LLVM may print LD1H rather than LD1B.
  assert re.search(r"^\s*ld1h[^\n]*\{\s*z\d+\.h", assembly, re.MULTILINE)
  assert not re.search(r"^\s*bl[^\n]*sme_pack_cost", assembly, re.MULTILINE)


def test_native_a_arm_cross_fp64_cost_keeps_shared_expanded_family(tmp_path):
  source = _lower(dtype="float64", bm=64, bn=256, bk=65).source
  assembly = _cross(tmp_path, source, lambda source: _wrapper(source, dtype="float64"), vl=16, svl=64)
  assert re.search(r"^\s*fmopa\s+za7\.d", assembly, re.MULTILINE)
  assert "expanded ZA64 catalog" in source
  assert "native A packing dispatch outside K" in source
  assert_no_vec_mask_arrays(source)


@pytest.mark.skipif(not _sme_available(), reason="actual ARM SME required")
@pytest.mark.parametrize("blocked", (False, True))
def test_native_a_arm_native_profit_and_fallback_alias_strides(blocked, tmp_path):
  pytest.importorskip("ml_dtypes")
  template = _template(bm=64 if blocked else None, bn=256 if blocked else None, bk=65 if blocked else None)
  compiler = make_compiler(target="Native", jobs=4, cache_dir=tmp_path / "cache", build_dir=tmp_path / "build")
  kernel = T.compile(template, compiler=compiler)
  rng = np.random.default_rng(22)
  for m, n, k in ((1, 16, 5), (2, 5, 7), (8, 256, 129), (64, 256, 129), (128, 256, 257)):
    a = (rng.integers(-3, 4, size=(m, k)) / 8).astype("bfloat16")
    backing = (rng.integers(-3, 4, size=(n, k)) / 8).astype("bfloat16")
    b = backing.T
    bias = rng.integers(-2, 3, size=(1, n)).astype(np.float32)
    for negative_a in (False, True):
      av = a[::-1] if negative_a else a
      for layout in ("row", "column", "negative"):
        c = _strided_output(m, n, np.float32, layout)
        c[:] = rng.integers(-2, 3, size=(m, n))
        for broadcast in (False, True):
          d = np.broadcast_to(c[:1], (m, n)) if broadcast else c
          expected = np.maximum((av.astype(np.float32) @ b.astype(np.float32) + d.copy() + bias) * 0.5 + 0.125, 0)
          kernel.operator.native(av, b, c, d, bias, **kernel._core.specialization_values)
          np.testing.assert_array_equal(
            c, expected, err_msg=f"{blocked=}, {(m, n, k)}, {negative_a=}, {layout=}, {broadcast=}"
          )
