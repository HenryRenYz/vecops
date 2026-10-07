"""Actual FA64 cross assembly for the isolated conditional packing candidate."""

import re

import pytest
from dsl_test_utils import assert_no_vec_mask_arrays
from test_dsl_matmul_sve_optimized import _cross
from test_dsl_sme_conditional_pack_buffers import _source
from test_dsl_sme_conditional_packing import library_template
from test_dsl_sme_family_owners import _owners
from vecops._arm_vector_state import VectorState
from vecops.language._compiler.backends.cpu.backend import CPUBackend


def _wrapper(source):
  lines = [
    '#include "vecops/Kernel.h"',
    "namespace vecops::spec { inline constexpr auto Parallelism = vecops::meta::cint<1>; }",
    source,
    (
      'extern "C" void conditional_pack(void* ap,void* bp,void* cp,void* dp,'
      "vecops::nint_t m,vecops::nint_t n,vecops::nint_t k,vecops::nint_t as,vecops::nint_t ak,"
      "vecops::nint_t bk,vecops::nint_t bs,vecops::nint_t cs,vecops::nint_t cn,"
      "vecops::nint_t ds,vecops::nint_t dn) {"
    ),
  ]
  for name, pointer, shape, strides in (
    ("A", "ap", ("m", "k"), ("as", "ak")),
    ("B", "bp", ("k", "n"), ("bk", "bs")),
    ("C", "cp", ("m", "n"), ("cs", "cn")),
    ("D", "dp", ("m", "n"), ("ds", "dn")),
  ):
    dimensions = ",".join(f"vecops::meta::Any{{{s}}}" for s in shape)
    steps = ",".join(f"vecops::meta::Any{{{s}}}" for s in strides)
    lines.append(
      f"auto {name}=vecops::tensor::make_tensor(static_cast<float*>({pointer}),"
      f"vecops::tensor::make_shape({dimensions}),vecops::tensor::make_strides({steps}));"
    )
  return "\n".join((*lines, "__kernel__(A,B,C,D);", "}"))


@pytest.mark.parametrize("vl,svl", ((16, 64), (64, 16), (64, 64)))
@pytest.mark.parametrize("dtype", ("bfloat16", "float16"))
def test_conditional_fp32_pack_cross_uses_za_without_escaping_calls(tmp_path, vl, svl, dtype):
  source = (
    CPUBackend()
    .lower(library_template(dtype).ir, (), target="SME", options={"_vector_state": VectorState(vl, svl)})
    .source
  )
  assert_no_vec_mask_arrays(source)
  assert "FP32 DAG K-contiguous SME ZA packing" in source
  assembly = _cross(tmp_path, source, _wrapper, vl=vl, svl=svl)
  owners = _owners(assembly, vl=vl, svl=svl)
  pack_owners = [body for body in owners if not re.search(r"^\s*\w*mopa\b", body, re.MULTILINE)]
  assert pack_owners
  for body in pack_owners:
    assert re.search(r"^\s*ld1w\s+.*za", body, re.MULTILINE)
    assert re.search(r"^\s*(?:mov|mova)\s+z\d+\.s,.*za", body, re.MULTILINE)
    assert re.search(r"^\s*bfcvt\b" if dtype == "bfloat16" else r"^\s*fcvt\b", body, re.MULTILINE)
    assert not re.search(r"^\s*(?:bl|blr)\b", body, re.MULTILINE)
    assert len(re.findall(r"^\s*smstart\b", body, re.MULTILINE)) == 1
    assert len(re.findall(r"^\s*smstop\b", body, re.MULTILINE)) == 1
    assert not re.search(r"^\s*(?:ldr|str)\s+z\d+[^\n]*\[sp", body, re.MULTILINE)


def _pack_wrapper(dtype, side):
  cpp = "vecops::bfloat16_t" if dtype == "bfloat16" else "vecops::float16_t"

  def wrapper(source):
    shape, strides = (("s", "k"), ("ss", "ks")) if side == "A" else (("k", "s"), ("ks", "ss"))
    layout = (
      "vecops::tensor::make_shape(" + ",".join(f"vecops::meta::Any{{{x}}}" for x in shape) + "),"
      "vecops::tensor::make_strides(" + ",".join(f"vecops::meta::Any{{{x}}}" for x in strides) + ")"
    )
    declarations = []
    for name, pointer, element in (("X", "xp", "float"), ("Cached", "cp", cpp), ("History", "hp", "float")):
      declarations.append(f"auto {name}=vecops::tensor::make_tensor(static_cast<{element}*>({pointer}),{layout});")
    declarations.extend(
      (
        (
          f"auto P=vecops::tensor::make_tensor(static_cast<{cpp}*>(pp),"
          "vecops::tensor::make_shape(vecops::meta::Any{ps}),vecops::tensor::make_strides(vecops::meta::cint<1>));"
        ),
        (
          "auto Probe=vecops::tensor::make_tensor(static_cast<std::int32_t*>(probe),"
          "vecops::tensor::make_shape(vecops::meta::cint<1>),vecops::tensor::make_strides(vecops::meta::cint<1>));"
        ),
      )
    )
    return (
      source + '\nextern "C" void guarded_pack(void* xp,void* cp,void* hp,void* pp,void* probe,'
      "vecops::nint_t s,vecops::nint_t k,vecops::nint_t ss,vecops::nint_t ks,vecops::nint_t ps,"
      "bool use_cache,bool use_history){\n"
      + "\n".join(declarations)
      + "\n__kernel__(X,Cached,History,P,Probe,use_cache,use_history);\n}"
    )

  return wrapper


@pytest.mark.parametrize("vl,svl", ((16, 64), (64, 16)))
@pytest.mark.parametrize("side", ("A", "B"))
@pytest.mark.parametrize("dtype", ("bfloat16", "float16"))
def test_runtime_guard_buffers_cross_retains_za_and_real_vector_fallback(tmp_path, vl, svl, side, dtype):
  source = _source(dtype, side)
  assembly = _cross(tmp_path, source, _pack_wrapper(dtype, side), vl=vl, svl=svl)
  owners = _owners(assembly, vl=vl, svl=svl)
  assert len(owners) == 1
  body = owners[0]
  assert re.search(r"^\s*ld1w\s+.*za", body, re.MULTILINE)
  assert re.search(r"^\s*(?:mov|mova)\s+z\d+\.s,.*za", body, re.MULTILINE)
  assert re.search(r"^\s*bfcvt\b" if dtype == "bfloat16" else r"^\s*fcvt\b", body, re.MULTILINE)
  assert not re.search(r"^\s*(?:bl|blr)\b", body, re.MULTILINE)
  assert len(re.findall(r"^\s*smstart\b", body, re.MULTILINE)) == 1
  assert len(re.findall(r"^\s*smstop\b", body, re.MULTILINE)) == 1
  assert not re.search(r"^\s*(?:ldr|str)\s+z\d+[^\n]*\[sp", body, re.MULTILINE)
  # The ordinary arm remains in this same object for cache/history/strided
  # conditions. Actual runtime Probe values, not this assembly assertion,
  # prove which arm was selected in the Native bitwise test.
  assert re.search(r"^\s*ld1w\s+\{\s*z\d+\.s\s*\},", assembly, re.MULTILINE)
