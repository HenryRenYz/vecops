# SPDX-FileCopyrightText: 2026 HenryRenYz and vecops contributors
# SPDX-License-Identifier: MIT
"""Incomplete writes cannot inherit another implementation's correct output."""

import importlib.util
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

MODULE = Path(__file__).resolve().parents[2] / "benchmarks/dsl_matmul_artifact_ab.py"
spec = importlib.util.spec_from_file_location("artifact_validation_test", MODULE)
ab = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ab)


def test_shared_prior_restores_mutability_without_changing_pointer_or_layout():
  owner = np.arange(23, dtype=np.float32).reshape(1, 1, 23)
  view = np.lib.stride_tricks.as_strided(owner, shape=(4, 1, 23), strides=(0, 92, 4), writeable=False)
  restored = ab.mutable_prior_view(view)
  assert restored.flags.writeable
  assert restored.ctypes.data == view.ctypes.data
  assert restored.shape == view.shape and restored.strides == view.strides
  assert not view.flags.writeable
  np.testing.assert_array_equal(restored, view)


def test_genuinely_readonly_prior_owner_is_not_made_writable():
  owner = np.arange(23, dtype=np.float32).reshape(1, 1, 23)
  owner.flags.writeable = False
  view = np.lib.stride_tricks.as_strided(owner, shape=(4, 1, 23), strides=(0, 92, 4), writeable=False)
  with pytest.raises(ValueError, match="writable same-address"):
    ab.mutable_prior_view(view)


def test_noop_compute_cannot_inherit_a_preceding_correct_output():
  output = np.zeros((2, 3), dtype=np.float32)
  with pytest.raises(AssertionError, match="initial poison"):
    ab.validate_complete_writes(lambda: None, (output,), label="missing DSL")


def test_partial_compute_write_is_detected_even_when_oracle_is_zero():
  output = np.zeros((2, 3), dtype=np.float32)

  def incomplete():
    output[0] = 0

  with pytest.raises(AssertionError, match="initial poison"):
    ab.validate_complete_writes(incomplete, (output,), label="partial DSL")


def test_strided_compute_preserves_broadcast_prior_and_external_canary():
  backing = np.full((5, 7), -111, dtype=np.float64)
  output = backing[::2, ::-2]
  prior = np.arange(output.shape[1], dtype=np.float64)[None, :]
  saved = prior.copy()
  before = backing.copy()
  mask = np.ones(backing.shape, dtype=bool)
  mask[::2, ::-2] = False
  ab.validate_complete_writes(lambda: output.__setitem__(slice(None), 3 + prior), (output,), label="seeded")
  np.testing.assert_array_equal(output, np.broadcast_to(3 + prior, output.shape))
  np.testing.assert_array_equal(prior, saved)
  np.testing.assert_array_equal(backing[mask], before[mask])


def test_packer_cannot_inherit_reference_padding(monkeypatch):
  values = [np.empty(1, dtype=np.int8) for _ in range(13)]
  values[11] = np.zeros(8, dtype=np.uint8)
  values[12] = np.zeros(8, dtype=np.uint8)
  monkeypatch.setattr(ab, "arguments", lambda values, repeat: None)

  def correct(_):
    values[11][:] = np.arange(8)

  def partial(_):
    values[11][:4] = np.arange(4)

  executables = {"cpp": SimpleNamespace(invoke=correct), "dsl": SimpleNamespace(invoke=partial)}
  with pytest.raises(AssertionError, match="initial poison"):
    ab.validate_packing_abi(("cpp", "dsl"), executables.__getitem__, values, (True, False))


def test_complete_packer_abi_includes_padding_and_ignores_inactive_buffer(monkeypatch):
  values = [np.empty(1, dtype=np.int8) for _ in range(13)]
  values[11] = np.empty(8, dtype=np.uint8)
  values[12] = np.full(8, 123, dtype=np.uint8)
  monkeypatch.setattr(ab, "arguments", lambda values, repeat: None)

  def pack(_):
    values[11][:4] = np.arange(4)
    values[11][4:] = 0

  ab.validate_packing_abi(("cpp", "dsl"), lambda _: SimpleNamespace(invoke=pack), values, (True, False))
  np.testing.assert_array_equal(values[12], 123)


def test_wrong_but_fully_written_padding_is_rejected(monkeypatch):
  values = [np.empty(1, dtype=np.int8) for _ in range(13)]
  values[11] = np.empty(8, dtype=np.uint8)
  monkeypatch.setattr(ab, "arguments", lambda values, repeat: None)
  executables = {
    "cpp": SimpleNamespace(invoke=lambda _: values[11].fill(0)),
    "dsl": SimpleNamespace(invoke=lambda _: values[11].fill(1)),
  }
  with pytest.raises(AssertionError, match="including padding"):
    ab.validate_packing_abi(("cpp", "dsl"), executables.__getitem__, values, (True, False))


def test_compensated_packing_sidecar_cannot_inherit_cpp(monkeypatch):
  values = [np.zeros(1, dtype=np.int32) for _ in range(13)]
  values[3] = np.zeros((1, 7), dtype=np.int32)
  monkeypatch.setattr(ab, "arguments", lambda values, repeat: None)

  def cpp(_):
    values[12][:] = 1
    values[3][:] = 2

  def incomplete(_):
    values[12][:] = 1
    values[3][:, :-1] = 2

  executables = {"cpp": SimpleNamespace(invoke=cpp), "dsl": SimpleNamespace(invoke=incomplete)}
  with pytest.raises(AssertionError, match="initial poison"):
    ab.validate_packing_abi(("cpp", "dsl"), executables.__getitem__, values, (False, True), compensated=True)


def test_shared_b_sidecar_only_poison_valid_physical_columns(monkeypatch):
  values = [np.zeros(1, dtype=np.int32) for _ in range(13)]
  backing = np.full(9, 99, dtype=np.int32)
  values[3] = np.lib.stride_tricks.as_strided(backing[:7], shape=(4, 1, 7), strides=(0, 0, 4))
  monkeypatch.setattr(ab, "arguments", lambda values, repeat: None)

  def pack(_):
    values[12][:] = 1
    values[3][0, 0, :] = 3

  ab.validate_packing_abi(
    ("cpp", "dsl"), lambda _: SimpleNamespace(invoke=pack), values, (False, True), compensated=True
  )
  np.testing.assert_array_equal(backing[7:], 99)
  np.testing.assert_array_equal(values[3], 3)


def test_ordinary_packing_does_not_poison_prior(monkeypatch):
  values = [np.zeros(1, dtype=np.int32) for _ in range(13)]
  values[3] = np.arange(7, dtype=np.int32)[None, :]
  saved = values[3].copy()
  monkeypatch.setattr(ab, "arguments", lambda values, repeat: None)
  ab.validate_packing_abi(
    ("cpp", "dsl"), lambda _: SimpleNamespace(invoke=lambda _: values[12].fill(1)), values, (False, True)
  )
  np.testing.assert_array_equal(values[3], saved)


def test_stage_availability_packing_ignores_missing_compute_roles():
  paths = {"dsl": None, "cpp": None, "pack_dsl": "dsl-packer", "pack_cpp": "cpp-packer"}
  assert ab.stage_availability(paths, paths, packing_only=True, packing_expected=True, reference_only=True) == {
    "status": "available"
  }
  assert ab.stage_availability(paths, paths, packing_only=False, packing_expected=True, reference_only=True)[
    "missing_roles"
  ] == ["candidate.cpp", "candidate.dsl"]


def test_compute_requires_complete_independent_packing_abi():
  paths = {"dsl": "dsl-compute", "cpp": "cpp-compute", "pack_dsl": None, "pack_cpp": "cpp-packer"}
  result = ab.stage_availability(paths, paths, packing_only=False, packing_expected=True, reference_only=True)
  assert result == {"status": "not_run", "reason": "missing_artifacts", "missing_roles": ["candidate.pack_dsl"]}
  assert ab.stage_availability(paths, paths, packing_only=False, packing_expected=False, reference_only=True) == {
    "status": "available"
  }


def test_missing_packer_is_not_not_applicable_and_ab_requires_baseline_packer():
  assert ab.stage_availability({}, {}, packing_only=True, packing_expected=True)["status"] == "not_run"
  assert ab.stage_availability({}, {}, packing_only=True, packing_expected=False) == {"status": "not_applicable"}
  new = {"dsl": "new", "cpp": "cpp", "pack_dsl": "new-pack", "pack_cpp": "cpp-pack"}
  old = {"dsl": "old", "pack_dsl": None}
  assert ab.stage_availability(old, new, packing_only=False, packing_expected=True)["missing_roles"] == [
    "baseline.pack_dsl"
  ]
