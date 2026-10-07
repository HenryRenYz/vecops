"""Architecture-target test-helper regressions."""

from types import SimpleNamespace

import dsl_test_utils
import pytest


def test_x86_target_is_not_available_on_arm_even_for_native_target(monkeypatch):
  monkeypatch.setenv("VECOPS_TEST_TARGET", "Native")
  monkeypatch.setattr(dsl_test_utils.os, "uname", lambda: SimpleNamespace(machine="aarch64"))
  assert not dsl_test_utils.x86_target_available_for_test()


def test_x86_native_target_is_available_on_x86(monkeypatch):
  monkeypatch.setenv("VECOPS_TEST_TARGET", "Native")
  monkeypatch.setattr(dsl_test_utils.os, "uname", lambda: SimpleNamespace(machine="x86_64"))
  assert dsl_test_utils.x86_target_available_for_test()


def test_explicit_non_x86_target_is_not_available_for_x86_tests(monkeypatch):
  monkeypatch.setenv("VECOPS_TEST_TARGET", "NativeFixedStreamingSVE")
  monkeypatch.setattr(dsl_test_utils.os, "uname", lambda: SimpleNamespace(machine="x86_64"))
  assert not dsl_test_utils.x86_target_available_for_test()


def test_amx_tile_gate_requires_matching_machine_and_target(monkeypatch):
  monkeypatch.setenv("VECOPS_TEST_TARGET", "Native")
  monkeypatch.setattr(dsl_test_utils.os, "uname", lambda: SimpleNamespace(machine="aarch64"))
  monkeypatch.setattr(dsl_test_utils.Path, "read_text", lambda *_args, **_kwargs: "amx_tile")
  assert not dsl_test_utils.amx_tile_available_for_test()

  monkeypatch.setattr(dsl_test_utils.os, "uname", lambda: SimpleNamespace(machine="x86_64"))
  monkeypatch.setenv("VECOPS_TEST_TARGET", "NativeFixedStreamingSVE")
  assert not dsl_test_utils.amx_tile_available_for_test()

  monkeypatch.setenv("VECOPS_TEST_TARGET", "Native")
  assert dsl_test_utils.amx_tile_available_for_test()
  assert not dsl_test_utils.amx_available_for_test()


def test_cpp_compile_command_honors_compiler_and_split_flags(monkeypatch):
  monkeypatch.setenv("CXX", "clang++")
  monkeypatch.setenv("CXXFLAGS", "-isystem /toolchain/include -DNAME='two words'")
  monkeypatch.setenv("LDFLAGS", "-L /toolchain/lib -Wl,--as-needed")
  command = dsl_test_utils.cpp_compile_command("input.cpp", "output", flags=("-O3",))
  assert command == [
    "clang++",
    "-isystem",
    "/toolchain/include",
    "-DNAME=two words",
    "-O3",
    "input.cpp",
    "-L",
    "/toolchain/lib",
    "-Wl,--as-needed",
    "-o",
    "output",
  ]


def test_metadata_assertion_resolves_exact_expression_not_tensor_prefix():
  source = "const auto dsl_c_parameter_A_data_7 = A.data();\nconst auto dsl_c_parameter_AB_data_9 = AB.data();"
  assert dsl_test_utils.parameter_snapshot_name(source, "A.data()") == "dsl_c_parameter_A_data_7"
  assert dsl_test_utils.parameter_snapshot_name(source, "AB.data()") == "dsl_c_parameter_AB_data_9"
  with pytest.raises(AssertionError, match="one entry snapshot"):
    dsl_test_utils.parameter_snapshot_name(source, "C.data()")
  with pytest.raises(AssertionError, match="read once"):
    dsl_test_utils.parameter_snapshot_name(source + "\nA.data();", "A.data()")


def test_arm_cross_headers_use_explicit_private_pair_without_system_fallback(monkeypatch, tmp_path):
  generic = tmp_path / "gcc14/include/c++"
  target = generic / "aarch64-conda-linux-gnu"
  target.mkdir(parents=True)
  (generic / "backward").mkdir()
  monkeypatch.setenv("VECOPS_TEST_ARM_CROSS_CXX_INCLUDE", str(generic))
  monkeypatch.setenv("VECOPS_TEST_ARM_CROSS_CXX_TARGET_INCLUDE", str(target))
  assert dsl_test_utils.arm_cross_cxx_include_flags("/unused-sysroot") == [
    "-nostdinc++",
    "-isystem",
    str(generic),
    "-isystem",
    str(target),
    "-isystem",
    str(generic / "backward"),
  ]


def test_arm_cross_headers_preserve_default_gcc13_sysroot(monkeypatch, tmp_path):
  monkeypatch.delenv("VECOPS_TEST_ARM_CROSS_CXX_INCLUDE", raising=False)
  monkeypatch.delenv("VECOPS_TEST_ARM_CROSS_CXX_TARGET_INCLUDE", raising=False)
  generic = tmp_path / "include/c++/13"
  target = generic / "aarch64-linux-gnu"
  target.mkdir(parents=True)
  assert dsl_test_utils.arm_cross_cxx_include_flags(tmp_path) == [
    "-nostdinc++",
    "-isystem",
    str(generic),
    "-isystem",
    str(target),
  ]


@pytest.mark.parametrize("field", ("VECOPS_TEST_ARM_CROSS_CXX_INCLUDE", "VECOPS_TEST_ARM_CROSS_CXX_TARGET_INCLUDE"))
def test_arm_cross_headers_reject_partial_override(monkeypatch, field):
  monkeypatch.delenv("VECOPS_TEST_ARM_CROSS_CXX_INCLUDE", raising=False)
  monkeypatch.delenv("VECOPS_TEST_ARM_CROSS_CXX_TARGET_INCLUDE", raising=False)
  monkeypatch.setenv(field, "/private/headers")
  with pytest.raises(ValueError, match="both generic and target"):
    dsl_test_utils.arm_cross_cxx_include_flags("/")


def test_arm_cross_headers_reject_missing_directories(monkeypatch, tmp_path):
  monkeypatch.delenv("VECOPS_TEST_ARM_CROSS_CXX_INCLUDE", raising=False)
  monkeypatch.delenv("VECOPS_TEST_ARM_CROSS_CXX_TARGET_INCLUDE", raising=False)
  with pytest.raises(ValueError, match="directory is missing"):
    dsl_test_utils.arm_cross_cxx_include_flags(tmp_path)


def test_full_sve_word_memory_accepts_equivalent_byte_and_word_spelling():
  body = """
    ptrue p0.h
    ld1h { z0.h }, p0/z, [x0]
    ptrue p0.b
    ld1b { z1.b }, p0/z, [x0, x8]
    ptrue p1.s
    st1w { z0.s }, p1, [x1]
    st1b { z1.b }, p0, [x1, x8]
  """
  dsl_test_utils.assert_full_sve_memory_words(body, kind="ld1", words=2)
  dsl_test_utils.assert_full_sve_memory_words(body, kind="st1", words=2)


@pytest.mark.parametrize(
  "predicate",
  (
    "ptrue p0.h, vl8",  # A fixed prefix is not a full scalable word.
    "ptrue p0.s",  # Coarser predicate leaves halfword lanes inactive.
    "whilelo p0.h, xzr, x8",  # A bounded active prefix.
    "ptrue p0.h\npfalse p0.b",  # Clobbered after a full definition.
    "ptrue p0.h\nldr p0, [sp]",  # A predicate reload is not a proven full word.
  ),
)
def test_full_sve_word_memory_rejects_partial_or_clobbered_predicates(predicate):
  with pytest.raises(AssertionError):
    dsl_test_utils.assert_full_sve_memory_words(f"{predicate}\nld1h {{ z0.h }}, p0/z, [x0]", kind="ld1", words=1)


def test_full_sve_word_memory_checks_reaching_definition_not_later_overwrite():
  body = "ptrue p0.h\nld1h { z0.h }, p0/z, [x0]\nptrue p0.s, vl2"
  dsl_test_utils.assert_full_sve_memory_words(body, kind="ld1", words=1)


def test_full_sve_word_memory_accepts_only_exact_fixed_width_prefix():
  body = "ptrue p0.h, vl8\nld1h { z0.h }, p0/z, [x0]"
  dsl_test_utils.assert_full_sve_memory_words(body, kind="ld1", words=1, bits="128")
  with pytest.raises(AssertionError):
    dsl_test_utils.assert_full_sve_memory_words(body, kind="ld1", words=1, bits="256")
  with pytest.raises(AssertionError):
    dsl_test_utils.assert_full_sve_memory_words(body, kind="ld1", words=2, bits="128")
