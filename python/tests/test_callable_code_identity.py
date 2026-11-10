# SPDX-FileCopyrightText: 2026 HenryRenYz and vecops contributors
# SPDX-License-Identifier: MIT
"""Immutable code reuse must preserve fresh callback and source identities."""

import gc
import multiprocessing
import threading
import weakref
from concurrent.futures import ThreadPoolExecutor
from types import CodeType, FunctionType

import pytest
import vecops._identity as identity


def fresh_code(filename="<code-identity>"):
  return compile("lambda x: (x + 2, (3, b'text'), lambda y: y * 4)", filename, "eval").co_consts[0]


def test_code_digest_matches_reference_without_repeating_code_walk(monkeypatch):
  code = fresh_code()
  expected = identity.digest(identity._code_identity(code))
  original = identity._code_identity
  calls = []

  def observe(value):
    calls.append(value)
    return original(value)

  monkeypatch.setattr(identity, "_code_identity", observe)
  assert identity._code_digest(code) == expected
  count = len(calls)
  assert count > 0
  assert identity._code_digest(code) == expected
  assert len(calls) == count


def test_equal_code_with_different_source_locations_has_distinct_identity():
  code = fresh_code("first.py")
  moved = code.replace(co_filename="second.py")
  assert code == moved
  assert identity._code_digest(code) != identity._code_digest(moved)
  assert identity._code_digest(moved) == identity.digest(identity._code_identity(moved))


def test_nonstandard_mutable_code_constants_are_read_again():
  state = [1]
  code = fresh_code().replace(co_consts=(state,))
  before = identity._code_digest(code)
  state.append(2)
  assert identity._code_digest(code) != before
  assert identity._code_digest(code) == identity.digest(identity._code_identity(code))


def test_nonstandard_mutable_source_name_is_read_again():
  class MutableName(str):
    label = "first"

    def __repr__(self):
      return self.label

  name = MutableName("source.py")
  code = fresh_code().replace(co_filename=name)
  before = identity._code_digest(code)
  name.label = "second"
  assert identity._code_digest(code) != before
  assert identity._code_digest(code) == identity.digest(identity._code_identity(code))


@pytest.mark.parametrize("nested", [False, True])
def test_tuple_subclass_constants_with_mutable_iteration_are_read_again(nested):
  class MutableTuple(tuple):
    value = 1

    def __iter__(self):
      return iter((self.value,))

  values = MutableTuple((1,))
  code = fresh_code().replace(co_consts=(values,) if nested else values)
  before = identity._code_digest(code)
  values.value = 2
  assert identity._code_digest(code) != before
  assert identity._code_digest(code) == identity.digest(identity._code_identity(code))


def test_code_cache_does_not_retain_unloaded_code():
  code = fresh_code()
  reference = weakref.ref(code)
  token = id(code)
  identity._code_digest(code)
  assert token in identity._code_digests
  del code
  gc.collect()
  assert reference() is None
  assert token not in identity._code_digests


def test_callable_globals_closures_defaults_and_code_are_fresh():
  scope = {"state": [1]}
  module = compile("lambda x=1, *, y=2: (state, x, y)", __file__, "eval")
  code = next(value for value in module.co_consts if isinstance(value, CodeType))
  callback = FunctionType(code, scope, argdefs=(1,))
  callback.__kwdefaults__ = {"y": 2}
  previous = identity.callable_identity(callback)
  for mutate in [lambda: scope["state"].append(2),
                 lambda: setattr(callback, "__defaults__", (3,)),
                 lambda: callback.__kwdefaults__.update(y=4),
                 lambda: setattr(callback, "__code__", callback.__code__.replace(co_firstlineno=100))]:
    mutate()
    current = identity.callable_identity(callback)
    assert current != previous
    previous = current

  state = [1]

  def closed():
    return state

  previous = identity.callable_identity(closed)
  state.append(2)
  assert identity.callable_identity(closed) != previous


def test_referenced_callback_code_replacement_is_observed():
  scope = {}
  helper = FunctionType(compile("lambda: 1", __file__, "eval").co_consts[0], scope)
  scope["helper"] = helper
  callback = FunctionType(compile("lambda: helper()", __file__, "eval").co_consts[0], scope)
  previous = identity.callable_identity(callback)
  helper.__code__ = helper.__code__.replace(co_consts=(None, 2))
  assert identity.callable_identity(callback) != previous


def test_source_bytes_remain_fresh_with_cached_code(tmp_path):
  source = tmp_path / "callback.py"
  source.write_text("def callback(): return 1\n")
  callback = FunctionType(compile(source.read_text(), str(source), "exec").co_consts[0], {})
  previous = identity.callable_identity(callback)
  source.write_text("def callback(): return 2\n")
  assert identity.callable_identity(callback) != previous


def test_build_session_reuses_location_but_reads_live_callable_state(tmp_path, monkeypatch):
  from vecops import BuildSession
  source=tmp_path/'callback.py'
  source.write_text('def callback(x=1): return state + x\n')
  scope={'state':1}
  exec(compile(source.read_text(),str(source),'exec'),scope)
  callback=scope['callback']
  with BuildSession(parallelism=1):
    previous=identity.callable_identity(callback)
    with monkeypatch.context() as guard:
      guard.setattr(identity.inspect,'getsourcefile',lambda _:
          (_ for _ in ()).throw(AssertionError('repeated source resolution')))
      assert identity.callable_identity(callback)==previous
      scope['state']=2
      assert identity.callable_identity(callback)!=previous
    source.write_text('def callback(x=1): return state - x\n')
    with pytest.raises(RuntimeError,match='build inputs changed'):
      from vecops._build_session import current_build_session
      current_build_session().checkpoint()
    source.write_text('def callback(x=1): return state + x\n')


def test_concurrent_code_lookups_match_reference():
  code = fresh_code()
  expected = identity.digest(identity._code_identity(code))
  with ThreadPoolExecutor(max_workers=8) as pool:
    assert list(pool.map(lambda _: identity._code_digest(code), range(100))) == [expected] * 100


@pytest.mark.skipif("fork" not in multiprocessing.get_all_start_methods(), reason="POSIX fork")
def test_fork_resets_code_cache_lock_owned_by_another_thread():
  code = fresh_code()
  expected = identity._code_digest(code)
  held, release = threading.Event(), threading.Event()
  context = multiprocessing.get_context("fork")
  reader, writer = context.Pipe(duplex=False)

  def hold():
    with identity._code_cache_lock:
      held.set()
      assert release.wait(10)

  def child():
    writer.send(identity._code_digest(code))
    writer.close()

  with ThreadPoolExecutor(max_workers=1) as pool:
    future = pool.submit(hold)
    assert held.wait(5)
    process = context.Process(target=child)
    process.start()
    try:
      assert reader.poll(5), "child inherited a locked code-cache mutex"
      assert reader.recv() == expected
      process.join(5)
      assert process.exitcode == 0
    finally:
      release.set()
      future.result()
      if process.is_alive():
        process.terminate()
      process.join(5)
      reader.close()
      writer.close()
