"""Versioned profile registry and immutable snapshots, without host probing."""

import hashlib
import threading

from .model import CostProfile

_registry = {}
_lock = threading.RLock()
_fingerprint = None


def register_profile(profile):
  global _fingerprint
  if not isinstance(profile, CostProfile) or not profile.name:
    raise ValueError("profile registration requires a named CostProfile")
  with _lock:
    if profile.name in _registry:
      raise ValueError(f"cost profile {profile.name!r} is already registered")
    _registry[profile.name] = profile
    _fingerprint = None


def unregister_profile(name):
  global _fingerprint
  with _lock:
    del _registry[name]
    _fingerprint = None


def snapshot():
  with _lock:
    return dict(_registry)


def fingerprint():
  global _fingerprint
  with _lock:
    if _fingerprint is None:
      _fingerprint = hashlib.sha256(repr(sorted(_registry.items())).encode()).hexdigest()
    return _fingerprint
