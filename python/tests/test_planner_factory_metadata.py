# SPDX-FileCopyrightText: 2026 HenryRenYz and vecops contributors
# SPDX-License-Identifier: MIT
"""Replay metadata caching must not grant permission to another allocation."""
import gc
import weakref

import pytest

torch = pytest.importorskip("torch")
from vecops.planner import runtime


def session_for(prototype, *, strict=True, dynamic=False):
  """Exercise the real request/cursor path against one captured factory slot."""
  from collections import Counter
  session = object.__new__(runtime.PlanSession)
  session.enable_factory_replay = True
  session.enable_dynamic_factories = dynamic
  session.strict_factory_replay = strict
  session._scope = ["diffusion"]
  session._loop_epochs = [1]
  session._factory_cursors = {}
  session._factory_cursor_keys = [[]]
  session._factory_metadata_cache = {}
  session._factory_metadata_cache_hits = 0
  session._factory_metadata_cache_misses = 0
  session._planned_factory_allocations = 0
  session._factory_allocated_bytes = 0
  session._factory_fallbacks = 0
  session._factory_fallback_reasons = Counter()
  session._factory_fallback_samples = []
  session._dynamic_pool = None
  shape = tuple(prototype.shape)
  strides = runtime.PlanSession._factory_strides("aten.empty_like.default", (prototype,), {}, shape)
  source = ("model.py", "block")
  key = (tuple(session._scope), *source, "aten.empty_like.default", shape, str(prototype.dtype))
  candidate = runtime._FactoryCandidate("temporary", shape, strides, str(prototype.dtype), prototype.numel() * prototype.element_size())
  session._factory_sites = {key: [candidate]}
  session._known_factory_keys = {key}
  session._factory_signatures = {key[:4]}
  session._views = {"temporary": torch.empty_strided(shape, strides, dtype=prototype.dtype)}
  return session, source


def request(session, source, prototype, **kwargs):
  return session._factory_tensor("aten.empty_like.default", (prototype,), kwargs, source=source)


def next_epoch(session):
  session._loop_epochs[0] += 1


@pytest.mark.parametrize("prototype", [
    torch.empty(3, 5), torch.empty(3, 5).t(), torch.empty(6, 5)[::2],
    torch.empty(0, 5), torch.empty(()), torch.empty(2, 3, 4, 5).to(memory_format=torch.channels_last),
])
def test_keyword_free_requests_preserve_dense_nondense_and_empty_layouts(prototype):
  session, source = session_for(prototype)
  first = request(session, source, prototype)
  assert first.stride() == session._views["temporary"].stride()
  next_epoch(session)
  assert request(session, source, prototype.clone(memory_format=torch.preserve_format)) is first
  # Cache keys must never own the prototype or its storage.
  for key in session._factory_metadata_cache:
    assert all(not isinstance(value, torch.Tensor) for value in key)


@pytest.mark.parametrize("change", ["dtype", "shape", "stride", "device"])
def test_same_site_changed_metadata_is_not_an_authorized_replay(change):
  value = torch.empty(3, 5)
  session, source = session_for(value)
  request(session, source, value)
  next_epoch(session)
  changed = {
      "dtype": lambda: value.double(), "shape": lambda: value[:2],
      "stride": lambda: torch.empty(5, 3).t(),
      "device": lambda: torch.empty(3, 5, device="meta"),
  }[change]()
  if change == "device":
    assert request(session, source, changed) is None
    assert session._factory_fallback_reasons["non_cpu"] == 1
  else:
    with pytest.raises(RuntimeError, match="metadata"):
      request(session, source, changed)


@pytest.mark.parametrize("kwargs", [
    {"requires_grad": True}, {"pin_memory": True}, {"layout": torch.sparse_coo},
    {"memory_format": torch.channels_last}, {"unrecognized": []},
])
def test_keyword_overrides_retain_strict_validation(kwargs):
  value = torch.empty(2, 3, 4, 5)
  session, source = session_for(value)
  request(session, source, value)
  next_epoch(session)
  with pytest.raises(RuntimeError, match="metadata_mismatch"):
    request(session, source, value, **kwargs)


def test_explicit_contiguous_override_uses_original_stride_validation():
  value = torch.empty(3, 5).t()
  session, source = session_for(value)
  request(session, source, value)
  next_epoch(session)
  with pytest.raises(RuntimeError, match="metadata_mismatch"):
    request(session, source, value, memory_format=torch.contiguous_format)


def test_cursor_exhaustion_epoch_reset_and_retained_dynamic_slot(monkeypatch):
  value = torch.empty(3, 5)
  session, source = session_for(value, dynamic=True)
  key = next(iter(session._factory_sites))
  session._factory_sites[key].insert(0, None)
  fallback = torch.zeros_like(value)
  monkeypatch.setattr(session, "_dynamic_factory_tensor", lambda *args: fallback)
  for _ in range(2):
    assert request(session, source, value) is fallback
    assert request(session, source, value) is session._views["temporary"]
    with pytest.raises(RuntimeError, match="epoch_exhausted"):
      request(session, source, value)
    next_epoch(session)
  assert session._planned_factory_allocations == 2


def test_nonstrict_metadata_mismatch_still_calls_dynamic_fallback(monkeypatch):
  value = torch.empty(3, 5)
  session, source = session_for(value, strict=False, dynamic=True)
  request(session, source, value)
  next_epoch(session)
  fallback = torch.empty(5, 3).t()
  monkeypatch.setattr(session, "_dynamic_factory_tensor", lambda *args: fallback)
  assert request(session, source, fallback) is fallback
  assert session._factory_fallback_reasons["metadata_mismatch"] == 1


def test_metadata_cache_does_not_keep_prototype_alive():
  value = torch.empty(3, 5)
  session, source = session_for(value)
  request(session, source, value)
  reference = weakref.ref(value)
  del value
  gc.collect()
  assert reference() is None


def test_valid_explicit_overrides_do_not_reuse_default_metadata_cache():
  value = torch.empty(3, 5)
  session, source = session_for(value)
  expected = request(session, source, value)
  next_epoch(session)
  actual = request(session, source, value, dtype=value.dtype, device="cpu",
                   layout=torch.strided, requires_grad=False, pin_memory=False,
                   memory_format=torch.preserve_format)
  assert actual is expected
  assert session._factory_metadata_cache_hits == 0


def test_tensor_subclasses_keep_general_validation():
  value = torch.nn.Parameter(torch.empty(3, 5))
  session, source = session_for(value)
  assert request(session, source, value) is session._views["temporary"]
  assert session._factory_metadata_cache == {}


def test_metadata_cache_is_bounded_for_changing_shapes():
  session, source = session_for(torch.empty(3, 5), strict=False)
  for size in range(300):
    request(session, source, torch.empty(size, 5))
  assert len(session._factory_metadata_cache) == 256


@pytest.mark.parametrize("layout", ["sparse_coo", "sparse_csr", "mkldnn"])
def test_unplanned_nonstrided_empty_like_keeps_torch_fallback(monkeypatch, layout):
  seed = torch.zeros(3, 5)
  if layout == "sparse_coo":
    prototype = seed.to_sparse()
  elif layout == "sparse_csr":
    prototype = seed.to_sparse_csr()
  else:
    if not torch.backends.mkldnn.is_available():
      pytest.skip("oneDNN unavailable")
    prototype = seed.to_mkldnn()
  assert type(prototype) is torch.Tensor
  assert prototype.layout != torch.strided
  session, _ = session_for(seed)
  monkeypatch.setattr(runtime, "current_session", lambda: session)
  try:
    expected = torch.empty_like(prototype)
  except NotImplementedError:
    # Some Torch builds do not implement empty_like for MKLDNN. Preserve
    # that backend error after the planner declines the unplanned request.
    with pytest.raises(NotImplementedError, match="empty_strided"):
      runtime.empty_like(prototype)
  else:
    result = runtime.empty_like(prototype)
    assert result.shape == expected.shape
    assert result.layout == expected.layout
    assert result.dtype == expected.dtype
  assert session._factory_metadata_cache == {}
  assert session._factory_fallback_reasons["unplanned_site"] == 1
