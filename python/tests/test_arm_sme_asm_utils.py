"""Negative controls for the manual Streaming/ZA assembly gate."""

import pytest
from arm_sme_asm_utils import assert_sme_owners


def _function(name, body):
  return f".type {name},@function\n{name}:\n{body}\n.size {name},0\n"


def _single(body="zero {za}"):
  return _function("with_streaming_za_probe", f"smstart\n{body}\nsmstop\nret")


def _bridge(*, entry=None, body=None):
  entry = entry or "smstart sm\nbl streaming_za_body_probe\nsmstop sm\nret"
  body = body or "addvl sp, sp, #-2\nsmstart za\nzero {za}\nsmstop za\naddvl sp, sp, #2\nret"
  return _function("with_streaming_za_probe", entry) + _function("streaming_za_body_probe", body)


def test_single_owner_and_same_domain_spills_are_checked():
  assert_sme_owners(_single("str p0, [sp, #1, mul vl]\nzero {za}"), 1, equal_domains=True)
  with pytest.raises(AssertionError):
    assert_sme_owners(_single("bl external\nzero {za}"), 1, equal_domains=True)


def test_private_bridge_and_loop_cfg_are_checked():
  body = "addvl sp, sp, #-2\nsmstart za\n.Lloop:\nzero {za}\ncbnz x0, .Lloop\nsmstop za\naddvl sp, sp, #2\nret"
  assert_sme_owners(_bridge(body=body), 1, equal_domains=False)


@pytest.mark.parametrize("bridge", (False, True))
def test_mutually_exclusive_close_blocks_preserve_one_exit_per_path(bridge):
  operand = " za" if bridge else ""
  body = f"smstart{operand}\ncbz x0, .Lzero\nzero {{za}}\nsmstop{operand}\nret\n.Lzero:\nsmstop{operand}\nret"
  assembly = _bridge(body=body) if bridge else _function("with_streaming_za_probe", body)
  assert_sme_owners(assembly, 1, equal_domains=not bridge)
  broken = body.replace(f".Lzero:\nsmstop{operand}\nret", ".Lzero:\nret")
  assembly = _bridge(body=broken) if bridge else _function("with_streaming_za_probe", broken)
  with pytest.raises(AssertionError, match="unbalanced return"):
    assert_sme_owners(assembly, 1, equal_domains=not bridge)


@pytest.mark.parametrize(
  "entry",
  (
    "smstart sm\nblr x0\nsmstop sm\nret",
    "smstart sm\nbl external\nsmstop sm\nret",
    "smstart sm\nbl streaming_za_body_probe\nbl streaming_za_body_probe\nsmstop sm\nret",
    "addvl sp, sp, #-1\nsmstart sm\nbl streaming_za_body_probe\nsmstop sm\nret",
    "smstart sm\nstr p0, [sp]\nbl streaming_za_body_probe\nsmstop sm\nret",
    "smstart sm\ncbz x0, .Lexit\nbl streaming_za_body_probe\n.Lexit:\nsmstop sm\nret",
    "smstart sm\nbl streaming_za_body_probe\nret",
  ),
)
def test_bridge_rejects_bad_outer_frame_call_and_return(entry):
  with pytest.raises(AssertionError):
    assert_sme_owners(_bridge(entry=entry), 1, equal_domains=False)


@pytest.mark.parametrize(
  "body",
  (
    "smstart za\nbl external\nsmstop za\nret",
    "smstart za\nbr x0\nsmstop za\nret",
    "smstart\nzero {za}\nsmstop\nret",
    "smstart za\ncbz x0, .Lexit\nsmstop za\n.Lexit:\nret",
    "zero {za}\nsmstart za\nsmstop za\nret",
    "smstart za\nb external\nsmstop za\nret",
  ),
)
def test_bridge_rejects_outlined_compute_mode_changes_and_escape(body):
  with pytest.raises(AssertionError):
    assert_sme_owners(_bridge(body=body), 1, equal_domains=False)


def test_bridge_contract_cannot_be_substituted_for_equal_single_owner():
  with pytest.raises(AssertionError):
    assert_sme_owners(_bridge(), 1, equal_domains=True)
  with pytest.raises(AssertionError):
    assert_sme_owners(_single(), 1, equal_domains=False)


def test_unbound_compute_body_is_not_ignored():
  unused = _function("streaming_za_body_unused", "smstart za\nzero {za}\nsmstop za\nret")
  with pytest.raises(AssertionError):
    assert_sme_owners(_bridge() + unused, 1, equal_domains=False)


def test_unreachable_call_or_duplicate_boundary_is_not_ignored():
  for after in ("bl external", "smstop za"):
    body = f"smstart za\nzero {{za}}\nsmstop za\nret\n{after}"
    with pytest.raises(AssertionError):
      assert_sme_owners(_bridge(body=body), 1, equal_domains=False)
