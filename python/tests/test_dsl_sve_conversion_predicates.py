"""Complete converting-memory leaves retain logical Tag validity domains.

Genuine ARM objects distinguish ptrue full words from fixed/subword prefixes.
The same-machine old/new native comparison accepts an explicit retained old
SDK include root; it never weakens NaN payload or alias assertions.
"""

import ctypes
import hashlib
import json
import os
import re
import subprocess
from pathlib import Path

import numpy as np
import pytest
from dsl_test_utils import arm_cross_cxx_include_flags, cpp_compile_command
from test_dsl_arm_vector_state_native import (
  PR_SME_SET_VL,
  PR_SVE_SET_VL,
  arm_vector_control,  # noqa: F401 - Shared safe state fixture.
)
from test_dsl_ordered_f16_pair_store import _body
from vecops._arm_vector_state import PR_SME_GET_VL, PR_SVE_GET_VL

CPP = r"""
#include "vecops/vec/Vec.h"
#include "vecops/vec/ConversionMemory.h"
namespace V = vecops::vec;
using F16 = vecops::float16_t;
using BF16 = vecops::bfloat16_t;
extern "C" void full_bf16_word(const BF16* a, float* c) {
  const auto t=V::ScalableTag<float,0>{}; V::store(t,c,V::load_convert(t,a));
}
extern "C" void full_f16_pair(const F16* a, F16* c) {
  const auto t=V::ScalableTag<float,1>{}; V::store_convert(t,c,V::load_convert(t,a));
}
extern "C" void full_bf16_pair(const BF16* a, BF16* c) {
  const auto t=V::ScalableTag<float,1>{}; V::store_convert(t,c,V::load_convert(t,a));
}
extern "C" void full_f16_four(const F16* a, F16* c) {
  const auto t=V::ScalableTag<float,2>{}; V::store_convert(t,c,V::load_convert(t,a));
}
extern "C" void full_i16_pair(const std::int16_t* a, std::int8_t* c) {
  const auto t=V::ScalableTag<std::int16_t,1>{}; V::store_convert(t,c,V::load(t,a));
}
extern "C" void full_f32_to_f16_pair(const float* a, F16* c) {
  const auto t=V::ScalableTag<float,1>{}; V::store_convert(t,c,V::load(t,a));
}
extern "C" void full_f32_to_bf16_pair(const float* a, BF16* c) {
  const auto t=V::ScalableTag<float,1>{}; V::store_convert(t,c,V::load(t,a));
}
extern "C" void f16_word_prefix(const F16* a, F16* c) {
  const auto t=V::ScalableTag<float,0>{}; V::store_convert(t,c,V::load_convert(t,a));
}
extern "C" void f16_subword(const F16* a, F16* c) {
  const auto t=V::ScalableTag<float,-1>{}; V::store_convert(t,c,V::load_convert(t,a));
}
extern "C" void f16_fixed_one(const F16* a, F16* c) {
  const auto t=V::FixedTag<float,1>{}; V::store_convert(t,c,V::load_convert(t,a));
}
extern "C" void f16_first(const F16* a, F16* c, vecops::nint_t count) {
  const auto t=V::ScalableTag<float,1>{};
  auto x=V::load_convert(t,a,V::opt::first(count),V::opt::merge(-17.25f));
  V::store_convert(t,c,x,V::opt::first(count));
}
extern "C" void f16_negative_stride(const F16* a, F16* c) {
  const auto t=V::ScalableTag<float,1>{};
  V::store_convert(t,c,V::load_convert(t,a,V::opt::strided(-1)),V::opt::strided(-1));
}
extern "C" vecops::nint_t f32_lanes() { return V::size(V::ScalableTag<float,0>{}); }
extern "C" vecops::nint_t i16_lanes() { return V::size(V::ScalableTag<std::int16_t,0>{}); }
#if defined(__ARM_FEATURE_SME) && !defined(HAS_FIXED_SVE_BITS)
extern "C" __arm_locally_streaming void streaming_f16_pair(const F16* a, F16* c) {
  const auto t=V::ScalableTag<float,1>{}; V::store_convert(t,c,V::load_convert(t,a));
}
extern "C" __arm_locally_streaming vecops::nint_t streaming_f32_lanes() {
  return V::size(V::ScalableTag<float,0>{});
}
#endif
"""


def _instruction_counts(body):
  instructions = [line.split("//", 1)[0].strip() for line in body.splitlines()]
  return {
    name: sum(bool(re.match(pattern, instruction)) for instruction in instructions)
    for name, pattern in {
      "length_query": r"cnt[bhwd]\b",
      "prefix": r"while(?:lo|lt)\b",
      "full": r"ptrue\b",
      "load_half": r"ld1h\b",
      "store_half": r"st1h\b",
      "load_byte": r"ld1b\b",
      "store_byte": r"st1b\b",
      "call": r"bl\b",
      "streaming_start": r"smstart\b",
      "streaming_stop": r"smstop\b",
    }.items()
  }


def _assert_full_half_memory_words(body, *, words, bits):
  """Accept full-width byte spelling only with a matching full predicate.

  BiSheng can represent the second FP16 memory word as bytes. That is not a
  gather, extra load, partial prefix or lane permutation: each raw word must
  retain one complete load/store and the original ordered conversions.
  """
  accesses = re.findall(r"\b(ld1|st1)([bh])\s+\{\s*z\d+\.([bh])\s*\},\s*(p\d+)(?:/z)?", body)
  assert sum(kind == "ld1" for kind, *_ in accesses) == words, body
  assert sum(kind == "st1" for kind, *_ in accesses) == words, body
  assert not re.search(r"\b(?:whilelo|whilelt|bl|smstart|smstop)\b", body), body
  for _kind, element, register, predicate in accesses:
    assert element == register, body
    full = re.findall(rf"(?m)^\s*ptrue\s+{predicate}\.{element}([^\n]*)", body)
    assert len(full) == 1, (predicate, full, body)
    suffix = full[0].split("//", 1)[0].strip()
    # An explicit VL suffix is complete only for the declared fixed physical
    # word size. Scalable byte access must use current-domain all lanes.
    expected = "" if bits == "scalable" else f", vl{int(bits) // (8 if element == 'b' else 16)}"
    assert suffix in {"", expected}, (element, suffix, bits, body)
    writes = re.findall(rf"(?m)^\s*(\w+)\s+{predicate}\.[bhsd]\b", body)
    assert writes and all(instruction == "ptrue" for instruction in writes), (predicate, writes, body)
  assert len(re.findall(r"\buunpklo\b", body)) == words
  assert len(re.findall(r"\buunpkhi\b", body)) == words
  assert len(re.findall(r"\bfcvt\s+z\d+\.s\b", body)) == words * 2
  assert len(re.findall(r"\bfcvt\s+z\d+\.h\b", body)) == words * 2
  assert len(re.findall(r"\buzp1\s+z\d+\.h\b", body)) == words


@pytest.mark.parametrize("bits", ("scalable", "128", "256", "512"))
@pytest.mark.parametrize("vl,svl", ((16, 64), (32, 16), (64, 32)))
def test_real_cross_unmasked_conversion_uses_tag_domain_predicate(tmp_path, bits, vl, svl):
  clang, sysroot = (os.environ.get(key) for key in ("VECOPS_TEST_ARM_CROSS_CLANG", "VECOPS_TEST_ARM_CROSS_SYSROOT"))
  if not clang or not sysroot:
    pytest.skip("set VECOPS_TEST_ARM_CROSS_CLANG/SYSROOT for genuine ARM cross objects")
  root = Path(__file__).resolve().parents[2]
  source, assembly, obj = tmp_path / "conversion.cpp", tmp_path / "conversion.s", tmp_path / "conversion.o"
  source.write_text(CPP)
  flags = [
    clang,
    "--target=aarch64-linux-gnu",
    f"--sysroot={sysroot}",
    "-march=armv9-a+sve2+bf16+sme+sme-fa64",
    f"-msve-vector-bits={bits}",
    "-std=c++20",
    "-O3",
    "-ffp-contract=off",
    "-DNDEBUG",
    f"-DVECOPS_KERNEL_SVE_BYTES={vl}",
    f"-DVECOPS_KERNEL_SME_BYTES={svl}",
    f"-DVECOPS_TARGET_FIXED_STREAMING_SVE_BITS={svl * 8}",
    "-I",
    str(root / "include"),
    "-I",
    str(root / "include/vecops"),
    *arm_cross_cxx_include_flags(sysroot),
  ]
  commands = []
  for output, action in ((assembly, "-S"), (obj, "-c")):
    command = [*flags, action, str(source), "-o", str(output)]
    commands.append(command)
    result = subprocess.run(command, capture_output=True, text=True, check=False)
    output.with_suffix(output.suffix + ".diagnostics").write_text(result.stdout + result.stderr)
    assert result.returncode == 0, result.stderr[-8000:]
  source.with_suffix(".commands.json").write_text(json.dumps(commands, indent=2))
  elf = obj.read_bytes()
  assert elf[:4] == b"\x7fELF" and int.from_bytes(elf[18:20], "little") == 183
  text = assembly.read_text()
  for name in ("full_bf16_word", "full_f16_pair", "full_bf16_pair", "full_i16_pair"):
    body = _body(text, name)
    assert "whilelo" not in body and "whilelt" not in body, (name, body)
    assert not re.search(r"\b(?:smstart|smstop|bl)\b", body)
    assert re.search(r"\b(?:ld1h|ld1sh|ld1w)\b", body)
    if bits == "scalable" and name in {"full_bf16_word", "full_f16_pair"}:
      assert not re.search(r"\bcnt[bhwd]\b", body), (name, body)
  four = _body(text, "full_f16_four")
  _assert_full_half_memory_words(four, words=2, bits=bits)
  # Subword memory is not full merely because its compute word is full.
  if bits == "scalable":
    for name in ("f16_word_prefix", "f16_subword"):
      assert re.search(r"\bwhile(?:lo|lt)\b", _body(text, name)), name
  fixed = _body(text, "f16_fixed_one")
  assert re.search(r"ptrue\s+p\d+\.h,\s*vl1\b|while(?:lo|lt)", fixed), fixed
  assert re.search(r"\bwhile(?:lo|lt)\b", _body(text, "f16_first"))
  if bits == "scalable":
    streaming = _body(text, "streaming_f16_pair")
    assert "whilelo" not in streaming and "whilelt" not in streaming
    assert len(re.findall(r"\bld1h\b", streaming)) == 1
    assert len(re.findall(r"\bst1h\b", streaming)) == 1
    assert "smstart" in streaming and "smstop" in streaming
  baseline = os.environ.get("VECOPS_TEST_SVE_PREDICATE_BASELINE_INCLUDE")
  if baseline:
    baseline = Path(baseline).resolve()
    assert (baseline / "vecops/vec/details/sve/ConversionMemory.h").is_file()
    old_commands = []
    for output, action in ((assembly, "-S"), (obj, "-c")):
      old_output = output.with_name("old-" + output.name)
      old_flags = [
        str(baseline / Path(value).relative_to(root / "include")) if value.startswith(str(root / "include")) else value
        for value in flags
      ]
      command = [*old_flags, action, str(source), "-o", str(old_output)]
      old_commands.append(command)
      result = subprocess.run(command, capture_output=True, text=True, check=False)
      old_output.with_suffix(old_output.suffix + ".diagnostics").write_text(result.stdout + result.stderr)
      assert result.returncode == 0, result.stderr[-8000:]
    source.with_suffix(".old-commands.json").write_text(json.dumps(old_commands, indent=2))
    old_text = assembly.with_name("old-" + assembly.name).read_text()
    symbols = (
      "full_bf16_word",
      "full_f16_pair",
      "full_bf16_pair",
      "full_f16_four",
      "f16_word_prefix",
      "f16_subword",
      "f16_fixed_one",
      "f16_first",
      "f16_negative_stride",
    )
    if bits == "scalable":
      symbols += ("streaming_f16_pair",)
    counts = {
      symbol: {"old": _instruction_counts(_body(old_text, symbol)), "new": _instruction_counts(_body(text, symbol))}
      for symbol in symbols
    }
    source.with_suffix(".old-new.json").write_text(json.dumps(counts, indent=2))
    # The conservative controls retain their original instruction geometry.
    # In particular Fixed1/subword must not acquire extra opaque CNTB queries.
    for symbol in ("f16_subword", "f16_fixed_one", "f16_first", "f16_negative_stride"):
      assert counts[symbol]["old"] == counts[symbol]["new"], (symbol, counts[symbol])
    # Here only the memory Tag is a subword: compute F32 P0 is legitimately
    # complete. Its queries can disappear while memory prefixes must remain.
    word_prefix = counts["f16_word_prefix"]
    assert word_prefix["new"]["length_query"] <= word_prefix["old"]["length_query"]
    assert {key: value for key, value in word_prefix["old"].items() if key != "length_query"} == {
      key: value for key, value in word_prefix["new"].items() if key != "length_query"
    }, word_prefix


def test_header_only_changes_unmasked_contiguous_predicates():
  root = Path(__file__).resolve().parents[2]
  header = (root / "include/vecops/vec/details/sve/ConversionMemory.h").read_text()
  masked = header.split("VECOPS_ALWAYS_INLINE Vec<ToTag> sve_load_convert_word(\n", 1)[1].split(
    "VECOPS_ALWAYS_INLINE Vec<ToTag> sve_load_convert_word_unmasked(\n", 1
  )[0]
  assert "const auto valid = sve_prefix_predicate<To>(size(to));" in masked
  assert "sve_convert_mask(FromTag{}, to, active)" in masked
  unmasked = header.split("VECOPS_ALWAYS_INLINE Vec<ToTag> sve_load_convert_word_unmasked(\n", 1)[1].split(
    "sve_widen_ordered_f16_half", 1
  )[0]
  assert "sve_conversion_word_predicate(to)" in unmasked
  assert "sve_conversion_word_predicate(FromTag{})" in unmasked
  assert "num_words(to) == 1 && num_words(FromTag{}) == 1" in unmasked
  helper = header.split("svbool_t sve_conversion_word_predicate(Tag tag)", 1)[1].split("\n}\n", 1)[0]
  assert "static_assert(num_words(tag) == 1);" in helper
  assert "if constexpr (sve_word_is_always_full<Tag>())" in helper
  assert "return sve_valid_word_predicate<0>(tag);" in helper
  assert "return sve_prefix_predicate<ElementOf<Tag>>(size(tag));" in helper


@pytest.fixture(scope="module")
def native_old_new(tmp_path_factory):
  baseline = os.environ.get("VECOPS_TEST_SVE_PREDICATE_BASELINE_INCLUDE")
  if not baseline:
    pytest.skip("set VECOPS_TEST_SVE_PREDICATE_BASELINE_INCLUDE to a retained old SDK include root")
  baseline = Path(baseline).resolve()
  root = Path(__file__).resolve().parents[2]
  header = "vecops/vec/details/sve/ConversionMemory.h"
  assert (baseline / header).is_file()
  directory = tmp_path_factory.mktemp("sve-conversion-predicates-native")
  source = directory / "conversion.cpp"
  source.write_text(CPP)
  result, manifest = {}, {}
  native_arch = os.environ.get("VECOPS_TEST_SVE_PREDICATE_NATIVE_ARCH", "native")
  for mode, include in (("old", baseline), ("new", root / "include")):
    library = directory / (mode + ".so")
    command = cpp_compile_command(
      source,
      library,
      flags=(
        "-std=c++20",
        "-O3",
        "-ffp-contract=off",
        f"-march={native_arch}",
        "-shared",
        "-fPIC",
        "-Wno-ignored-attributes",
        "-I",
        str(include),
        "-I",
        str(include / "vecops"),
      ),
    )
    build = subprocess.run(command, capture_output=True, text=True, check=False)
    (directory / (mode + ".build.log")).write_text(build.stdout + build.stderr)
    assert build.returncode == 0, build.stderr[-8000:]
    result[mode] = ctypes.CDLL(str(library))
    for symbol in ("f32_lanes", "i16_lanes"):
      getattr(result[mode], symbol).restype = ctypes.c_int64
    if hasattr(result[mode], "streaming_f32_lanes"):
      result[mode].streaming_f32_lanes.restype = ctypes.c_int64
    manifest[mode] = {
      "command": command,
      "native_arch": native_arch,
      "header": str(include / header),
      "header_sha256": hashlib.sha256((include / header).read_bytes()).hexdigest(),
      "library": str(library),
      "library_sha256": hashlib.sha256(library.read_bytes()).hexdigest(),
    }
  (directory / "manifest.json").write_text(json.dumps(manifest, indent=2))
  return result


def compare_libraries(libraries):
  for symbol, ratio, storage in (
    ("full_f16_pair", 2, "float16"),
    ("full_f16_four", 4, "float16"),
    ("f16_word_prefix", 1, "float16"),
    ("f16_subword", 0.5, "float16"),
    ("f16_fixed_one", None, "float16"),
    ("f16_first", 2, "float16"),
    ("f16_negative_stride", 2, "float16"),
    ("full_bf16_pair", 2, "bfloat16"),
  ):
    widths = [int(lib.f32_lanes() * ratio) if ratio is not None else 1 for lib in libraries.values()]
    assert widths[0] == widths[1]
    width = widths[0]
    patterns = np.array((0, 0x8000, 0x3C00, 0x7C00, 0xFC00, 0x7E25, 0x7C19), dtype=np.uint16)
    if storage == "bfloat16":
      patterns = np.array((0, 0x8000, 0x3F80, 0x7F80, 0xFF80, 0x7FC5, 0x7F99), dtype=np.uint16)
    for alias in (False, True):
      for count in (0, 1, width - 1, width) if symbol == "f16_first" else (width,):
        outputs = []
        for lib in libraries.values():
          source = np.resize(patterns, width).copy()
          backing = np.full(width + 4, 0xA125, dtype=np.uint16)
          output = backing[2:-2]
          if alias:
            output[:] = source
            source = output
          original = output.copy()
          ap, cp = source.ctypes.data, output.ctypes.data
          if symbol == "f16_negative_stride":
            ap += (width - 1) * 2
            cp += (width - 1) * 2
          arguments = [ctypes.c_void_p(ap), ctypes.c_void_p(cp)]
          if symbol == "f16_first":
            arguments.append(ctypes.c_int64(count))
          getattr(lib, symbol)(*arguments)
          assert np.all(backing[:2] == 0xA125) and np.all(backing[-2:] == 0xA125)
          if symbol == "f16_first":
            np.testing.assert_array_equal(output[count:], original[count:])
          outputs.append(output.copy())
        np.testing.assert_array_equal(outputs[0], outputs[1], err_msg=f"{symbol},count={count},alias={alias}")
  # Real F32 narrowing, rather than only a half round trip, checks rounding
  # ties, adjacent representables, underflow, overflow and distinct NaN bits.
  ties = np.array((1 + 2**-11, -1 - 2**-11, 1 + 2**-8, -1 - 2**-8, 2**-25, -(2**-25), 65520), dtype=np.float32)
  nearby = np.concatenate((ties, np.nextafter(ties, np.float32(0)), np.nextafter(ties, np.float32(np.inf))))
  specials = np.array((0, 0x80000000, 0x7F800000, 0xFF800000, 0x7FC00019, 0x7F800125), dtype=np.uint32).view(np.float32)
  values = np.resize(np.concatenate((nearby, specials)), libraries["old"].f32_lanes() * 2)
  for symbol in ("full_f32_to_f16_pair", "full_f32_to_bf16_pair"):
    outputs = []
    for lib in libraries.values():
      backing = np.full(values.size + 4, 0xA125, dtype=np.uint16)
      output = backing[2:-2]
      getattr(lib, symbol)(ctypes.c_void_p(values.ctypes.data), ctypes.c_void_p(output.ctypes.data))
      assert np.all(backing[:2] == 0xA125) and np.all(backing[-2:] == 0xA125)
      outputs.append(output.copy())
    np.testing.assert_array_equal(outputs[0], outputs[1], err_msg=symbol)
  integers = np.resize(
    np.array((-32768, -129, -128, -1, 0, 127, 128, 32767), dtype=np.int16), libraries["old"].i16_lanes() * 2
  )
  for lib in libraries.values():
    backing = np.full(integers.size + 4, 51, dtype=np.int8)
    output = backing[2:-2]
    lib.full_i16_pair(ctypes.c_void_p(integers.ctypes.data), ctypes.c_void_p(output.ctypes.data))
    np.testing.assert_array_equal(output, np.clip(integers, -128, 127).astype(np.int8))
    assert np.all(backing[:2] == 51) and np.all(backing[-2:] == 51)


def test_native_conversion_predicate_same_machine_special_bits_alias_and_prefix(native_old_new):
  compare_libraries(native_old_new)


def test_native_conversion_predicate_unequal_vector_domains(native_old_new, request):
  control = request.getfixturevalue("arm_vector_control")
  pairs = [(vl, svl) for vl in control.supported(PR_SVE_SET_VL) for svl in control.supported(PR_SME_SET_VL)]
  unequal = [pair for pair in pairs if pair[0] != pair[1]]
  if not unequal:
    pytest.skip("the processor does not expose distinct legal VL/SVL pairs")
  selected = list(dict.fromkeys([pair for pair in ((16, 64), (64, 16)) if pair in unequal] + [unequal[0], unequal[-1]]))
  domains = []
  try:
    for pair in selected:
      control.select(pair)
      ordinary = [lib.f32_lanes() for lib in native_old_new.values()]
      assert ordinary == [pair[0] // 4, pair[0] // 4]
      compare_libraries(native_old_new)
      widths = [lib.streaming_f32_lanes() for lib in native_old_new.values()]
      assert widths == [pair[1] // 4, pair[1] // 4]
      domains.append(
        {"vl_svl_bytes": pair, "ordinary_f32_lanes_old_new": ordinary, "streaming_f32_lanes_old_new": widths}
      )
      width = widths[0] * 2
      patterns = np.array((0, 0x8000, 0x3C00, 0x7C00, 0xFC00, 0x7E25, 0x7C19), dtype=np.uint16)
      values = np.resize(patterns, width)
      outputs = []
      for lib in native_old_new.values():
        backing = np.full(width + 4, 0xA125, dtype=np.uint16)
        output = backing[2:-2]
        lib.streaming_f16_pair(ctypes.c_void_p(values.ctypes.data), ctypes.c_void_p(output.ctypes.data))
        assert np.all(backing[:2] == 0xA125) and np.all(backing[-2:] == 0xA125)
        outputs.append(output.copy())
        control.assert_pair(pair)
      np.testing.assert_array_equal(outputs[0], outputs[1])
      control.assert_pair(pair)
  finally:
    control.restore()
    output = Path(native_old_new["new"]._name).parent / "vector-domain-proof.json"
    output.write_text(
      json.dumps(
        {
          "selected_pairs": selected,
          "executed_domains": domains,
          "restored_sve_raw": control.raw(PR_SVE_GET_VL),
          "restored_sme_raw": control.raw(PR_SME_GET_VL),
          "original_sve_raw": control.original_sve,
          "original_sme_raw": control.original_sme,
        },
        indent=2,
      )
    )
