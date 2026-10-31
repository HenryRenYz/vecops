# SPDX-FileCopyrightText: 2026 HenryRenYz and vecops contributors
# SPDX-License-Identifier: MIT
"""Executable GraphCapture must never replay a Python model/operator loop."""
import gc
import weakref

import pytest

torch = pytest.importorskip('torch')
from vecops import graph, planner


def _compile(function, *args, **kwargs):
  capture = graph.GraphCapture('native-replay-test')
  expected = capture.capture_torch(function, *args, **kwargs)
  return capture, capture.compile_torch(), expected


def test_native_interpreter_mixed_aten_views_copies_and_static_guards():
  value = torch.arange(24.).reshape(4, 6).t()
  def model(x, factor=2):
    y = x.to(torch.bfloat16).contiguous().to(torch.float32)
    y.mul_(factor)
    return {'full':y, 'view':y[:, 1:3], 'static':'ok'}
  capture, replay, expected = _compile(model, value, factor=2)
  actual = replay(value + 1, factor=2)
  reference = model(value + 1, factor=2)
  torch.testing.assert_close(actual['full'], reference['full'], rtol=0, atol=0)
  assert actual['full'].untyped_storage()._cdata == actual['view'].untyped_storage()._cdata
  assert replay.graph is capture._torch_execution_graph
  assert not any(x in replay.ir for x in ('PythonOp', 'CallFunction', 'CallMethod'))
  with pytest.raises((RuntimeError, ValueError), match='metadata|structure'):
    replay(torch.zeros(4, 6), factor=2)
  with pytest.raises(ValueError, match='non-Tensor'):
    replay(value, factor=3)
  with pytest.raises(RuntimeError, match='metadata'):
    replay(value.to(torch.float64), factor=2)
  torch.testing.assert_close(expected['view'], expected['full'][:, 1:3])


def test_default_cpu_rng_is_preserved_by_capture_and_advanced_by_replay():
  torch.manual_seed(1923)
  value = torch.ones(17)
  before = torch.get_rng_state().clone()
  capture, replay, expected = _compile(lambda x:x + torch.randn_like(x), value)
  assert torch.equal(before, torch.get_rng_state())
  first = replay(value); second = replay(value)
  torch.testing.assert_close(first, expected, rtol=0, atol=0)
  assert not torch.equal(first, second)
  torch.manual_seed(1923)
  torch.testing.assert_close(replay(value), expected, rtol=0, atol=0)


def test_failed_capture_restores_context_rng_and_rejects_explicit_generator():
  def fail(x):
    torch.randn_like(x)
    return x.nonzero()
  capture = graph.GraphCapture('failure');before = torch.get_rng_state().clone()
  with pytest.raises(ValueError, match='unsupported'):
    capture.capture_torch(fail, torch.ones(4))
  assert graph.current_capture() is None
  assert torch.equal(before, torch.get_rng_state())
  with pytest.raises(RuntimeError, match='successfully'):
    capture.compile_torch()
  generator = torch.Generator().manual_seed(13);state=generator.get_state().clone()
  with pytest.raises(ValueError, match='Generator'):
    graph.GraphCapture('generator').capture_torch(
        lambda x:torch.randn(x.shape, generator=generator), torch.ones(4))
  assert torch.equal(state, generator.get_state())


def test_alias_guards_cover_storage_identity_and_distinct_frombuffer_overlap():
  def model(x, y):
    x.add_(1)
    return y + 2
  backing = torch.arange(12.)
  _, replay, _ = _compile(model, backing[:8], backing[4:])
  new = torch.arange(12.)
  torch.testing.assert_close(replay(new[:8], new[4:]), new[4:] + 2)
  with pytest.raises(RuntimeError, match='metadata|alias'):
    replay(torch.ones(8), torch.ones(8))
  # Two separately created StorageImpls can share/overlap a Python buffer.
  raw = bytearray(64)
  a = torch.frombuffer(raw, dtype=torch.float32, count=8, offset=0)
  b = torch.frombuffer(raw, dtype=torch.float32, count=8, offset=16)
  _, overlap, _ = _compile(model, a, b)
  other = bytearray(64)
  aa = torch.frombuffer(other, dtype=torch.float32, count=8, offset=0)
  bb = torch.frombuffer(other, dtype=torch.float32, count=8, offset=16)
  overlap(aa, bb)
  with pytest.raises(RuntimeError, match='alias'):
    overlap(torch.ones(8), torch.ones(8))
  _, independent, _ = _compile(model, torch.ones(8), torch.ones(8))
  with pytest.raises(RuntimeError, match='alias'):
    independent(aa, bb)


def test_python_dispatch_modes_subclasses_and_python_custom_ops_are_rejected():
  from torch.utils._python_dispatch import TorchDispatchMode
  class Mode(TorchDispatchMode):
    def __torch_dispatch__(self, function, types, args=(), kwargs=None):
      return function(*args, **(kwargs or {}))
  _, replay, _ = _compile(lambda x:x + 1, torch.ones(4))
  with Mode(), pytest.raises(RuntimeError, match='TorchDispatchMode'):
    replay(torch.ones(4))
  class Subclass(torch.Tensor):
    pass
  with pytest.raises(TypeError, match='subclasses'):
    replay(torch.ones(4).as_subclass(Subclass))
  library = torch.library.Library('vecops_replay_python_test', 'DEF')
  library.define('python_cpu(Tensor x) -> Tensor')
  library.impl('python_cpu', lambda x:x + 1, 'CPU')
  capture = graph.GraphCapture('python-custom')
  capture.capture_torch(torch.ops.vecops_replay_python_test.python_cpu.default, torch.ones(4))
  with pytest.raises(ValueError, match='proven C\+\+ CPU'):
    capture.compile_torch(native_operators={'vecops_replay_python_test::python_cpu':library})


def test_weights_are_live_storage_bindings_not_folded_tensor_constants():
  class Model(torch.nn.Module):
    def __init__(self):
      super().__init__();self.weight=torch.nn.Parameter(torch.ones(4))
    def forward(self, x):
      return x + self.weight * 2
  model = Model();_, replay, _ = _compile(model, torch.ones(4))
  model.forward = lambda *args:(_ for _ in ()).throw(AssertionError('Python model called'))
  with torch.no_grad():model.weight.add_(3)
  torch.testing.assert_close(replay(torch.ones(4)), torch.full((4,), 9.))
  del model;gc.collect()
  torch.testing.assert_close(replay(torch.ones(4)), torch.full((4,), 9.))


def test_compile_releases_intermediates_and_exceptions_release_replay_lock():
  references=[]
  def model(x, index):
    middle = x + 2;references.append(weakref.ref(middle))
    return middle[index]
  capture, replay, _ = _compile(model, torch.ones(8), torch.tensor([0, 1]))
  gc.collect();assert references[0]() is None
  with pytest.raises((RuntimeError, IndexError)):
    replay(torch.ones(8), torch.tensor([90, 91]))
  torch.testing.assert_close(replay(torch.ones(8), torch.tensor([0, 1])), torch.full((2,), 3.))
  again = capture.compile_torch()
  torch.testing.assert_close(again(torch.ones(8), torch.tensor([0, 1])), torch.full((2,), 3.))


def test_native_vecops_out_side_effects_and_bridge_owner_survive(tmp_path, monkeypatch):
  import vecops
  import vecops.language as T
  from dsl_test_utils import make_compiler
  torch.set_num_threads(2)
  compiler = make_compiler(target='Native', jobs=2,
      cache_dir=tmp_path/'kernel', build_dir=tmp_path/'build')
  @T.jit(out_idx=-1, compiler=compiler)
  def factory():
    @T.template
    def kernel(A:T.Tensor((8,), dtype='float32'), C:T.Tensor((8,), dtype='float32')):
      for i in T.grid(8, parallel='require'):
        T.store(C[i], T.load(A[i]) * 3)
    return kernel
  kernel=factory()
  kernel(torch.ones(8)) # Materialize actual C++ Torch registration before capture.
  def model(x, output):
    kernel(x, output)
    return x + 1 # Native out side effect must remain despite unused op return.
  output=torch.empty(8);capture, replay, _=_compile(model, torch.ones(8), output)
  def forbidden(*args, **kwargs):raise AssertionError('Python bridge called during replay')
  monkeypatch.setattr(kernel._core, 'torch', forbidden)
  data=torch.arange(8.);out=torch.empty(8)
  replay(data, out)
  torch.testing.assert_close(out, data * 3)
  assert 'vecops_language::' in replay.ir
  del kernel, compiler;gc.collect()
  replay(data + 1, out)
  torch.testing.assert_close(out, (data + 1) * 3)
  schema=next(i.function._schema for i in capture._torch_replay_tape.instructions
              if i.function._schema.name.startswith('vecops_language::'))
  namespace, opname=schema.name.split('::', 1)
  calls=[]
  def python_replacement(*args):
    calls.append(True)
    raise AssertionError('dispatcher replacement entered Python')
  replacement=torch.library.Library(namespace, 'IMPL')
  try:
    replacement.impl(opname, python_replacement, 'CPU')
    try:
      replay(data, out)
    except RuntimeError as error:
      assert 'invalidated' in str(error)
    else:
      torch.testing.assert_close(out, data * 3)
    assert not calls
  finally:
    replacement._destroy()


@pytest.mark.parametrize('shape', [(), (0,), (2, 0, 3), (2, 3, 4, 5)])
def test_scalar_empty_and_memory_format_shapes(shape):
  value = torch.ones(shape)
  if len(shape) == 4:
    value = value.contiguous(memory_format=torch.channels_last)
  _, replay, expected = _compile(lambda x:x.clone(memory_format=torch.preserve_format) + 2, value)
  torch.testing.assert_close(replay(value), expected, rtol=0, atol=0)


def test_overlap_component_also_guards_relative_displacements():
  raw=bytearray(96)
  def t(offset):return torch.frombuffer(raw, dtype=torch.float32, count=8, offset=offset)
  _, replay, _ = _compile(lambda a,b,c:a + b + c, t(0), t(16), t(40))
  # Still one connected overlap component, but its pairwise geometry changed.
  with pytest.raises(RuntimeError, match='byte-overlap'):
    replay(t(0), t(16), t(24))


class _Owner(bytearray):
  pass


class _System:
  def __init__(self):self.allocations=[]
  def current_cpu_domain(self):return 0
  def allocate(self, size, **kwargs):
    owner=_Owner(size);self.allocations.append(weakref.ref(owner));return owner


def _machine():
  return planner.MachineProfile(fast_capacity=4096, slow_capacity=1 << 20,
      fast_bandwidth_mib_s=100, slow_bandwidth_mib_s=10,
      transfer_bandwidth_mib_s=100, alignment=64)


def _factory_plan(execution):
  return planner.plan_memory(execution, _machine(), config=planner.PlannerConfig(
      minimum_fast_bytes=1, allow_input_fast=False, allow_persistent_fast=False,
      only_explicit_factory_temporaries=True, require_replayable_temporaries=True))


def _factory_model(value):
  temporary=planner.empty_like(value)
  temporary.copy_(value + 1)
  return temporary * 3


def test_same_capture_plan_reuses_private_factory_arena_and_preserves_output_values():
  value=torch.arange(8.);capture=graph.GraphCapture('planned')
  capture.capture_torch(_factory_model, value)
  plan=_factory_plan(capture._torch_execution_graph);system=_System()
  replay=capture.compile_torch(memory_plan=plan, system=system, large_pages=False)
  assert replay.planned_factory_count == 1
  assert replay.allocator_policy['functional_aten']=='native ATen allocator'
  result=replay(value);saved=result.clone()
  torch.testing.assert_close(replay(value + 4), (value + 5) * 3)
  torch.testing.assert_close(result, saved)
  assert any(reference() is not None for reference in system.allocations)
  assert planner.current_session() is None


def test_capture_real_factory_replay_session_lease_survives_close():
  import vecops
  value=torch.arange(8.)
  original=vecops.precompile(_factory_model, value, capture_graph=True).graph
  plan=_factory_plan(original);system=_System()
  with planner.PlanSession(original, plan, system=system, large_pages=False,
      enable_factory_replay=True, strict_factory_replay=True) as session:
    capture=graph.GraphCapture('active-session')
    expected=capture.capture_torch(_factory_model, value)
    replay=capture.compile_torch()
    assert session.stats()['planned_factory_allocations'] == 1
    torch.testing.assert_close(replay(value), expected)
    assert replay.allocator_policy['captured_arena_copy_bytes'] > 0
  assert any(reference() is not None for reference in system.allocations)
  torch.testing.assert_close(replay(value), expected)
  torch.testing.assert_close(replay(value + 2), (value + 3) * 3)
  # A later planner session must allocate its own arena while graph holds its lease.
  with planner.PlanSession(original, plan, system=system, large_pages=False,
      enable_factory_replay=True) as other:
    _factory_model(torch.ones(8) * 99)
  torch.testing.assert_close(replay(value), expected)


def test_bound_output_is_explicitly_borrowed_and_can_be_cloned():
  buffer=torch.empty(8)
  _, replay, _=_compile(lambda x:buffer.copy_(x + 1), torch.ones(8))
  first=replay(torch.ones(8));saved=first.clone()
  second=replay(torch.ones(8) * 4)
  assert first.data_ptr()==second.data_ptr()
  torch.testing.assert_close(first, torch.full((8,), 5.))
  torch.testing.assert_close(saved, torch.full((8,), 2.))


def test_real_tensor_factory_detach_and_constant_binding_can_recompile():
  capture, replay, expected=_compile(lambda x:x + torch.tensor([1., 2., 3.]), torch.ones(3))
  torch.testing.assert_close(replay(torch.ones(3)), expected)
  again=capture.compile_torch()
  torch.testing.assert_close(again(torch.ones(3)), expected)
  with pytest.raises(ValueError, match='unsupported'):
    graph.GraphCapture('resize').capture_torch(lambda x:x.resize_(9), torch.ones(3))


def test_empty_storage_inside_overlap_component_does_not_split_component():
  raw=bytearray(96)
  a=torch.frombuffer(raw, dtype=torch.uint8, count=80, offset=0)
  b=torch.frombuffer(raw, dtype=torch.uint8, count=16, offset=60)
  storage=torch._C._construct_storage_from_data_pointer(a.data_ptr()+50, torch.device('cpu'), 0)
  empty=torch.empty(0, dtype=torch.uint8).set_(storage)
  _, replay, _=_compile(lambda x,e,y:x.sum() + y.sum() + e.sum(), a, empty, b)
  other=bytearray(96)
  aa=torch.frombuffer(other, dtype=torch.uint8, count=80, offset=0)
  bb=torch.frombuffer(other, dtype=torch.uint8, count=16, offset=60)
  ee_storage=torch._C._construct_storage_from_data_pointer(aa.data_ptr()+50, torch.device('cpu'), 0)
  ee=torch.empty(0, dtype=torch.uint8).set_(ee_storage)
  replay(aa, ee, bb)
  with pytest.raises(RuntimeError, match='byte-overlap'):
    replay(aa, ee, torch.ones(16, dtype=torch.uint8))


def test_planned_factory_view_chain_rejects_absolute_storage_offset():
  def model(value):
    temporary=planner.empty_like(value)
    temporary.copy_(value)
    alias=temporary.view(-1)
    return torch.as_strided(alias, value.shape, value.stride(), 0) + 1
  capture=graph.GraphCapture('planned-offset')
  capture.capture_torch(model, torch.ones(8))
  plan=_factory_plan(capture._torch_execution_graph)
  with pytest.raises(ValueError, match='storage-offset'):
    capture.compile_torch(memory_plan=plan, system=_System(), large_pages=False)


@pytest.mark.parametrize('dtype', [torch.float16, torch.bfloat16, torch.float32, torch.float64])
def test_wrapped_python_float_preserves_scalar_tensor_promotion(dtype):
  value=torch.tensor(2., dtype=dtype)
  _, replay, expected=_compile(lambda x:(x / 16.0 * 0.25)[None] * torch.ones(8, dtype=dtype), value)
  actual=replay(value)
  assert actual.dtype==expected.dtype
  torch.testing.assert_close(actual, expected, rtol=0, atol=0)


def test_each_native_node_preserves_eager_matmul_precision_and_restores_caller():
  old=torch.get_float32_matmul_precision()
  try:
    torch.set_float32_matmul_precision('medium')
    x=torch.randn(37,256);weight=torch.randn(256,384)
    def model(x):
      try:
        torch.set_float32_matmul_precision('highest')
        exact=torch.einsum('mk,kn->mn',x,weight)
      finally:torch.set_float32_matmul_precision('medium')
      return exact,torch.einsum('mk,kn->mn',x,weight)
    capture,replay,expected=_compile(model,x)
    torch.set_float32_matmul_precision('high')
    result=replay(x)
    assert torch.get_float32_matmul_precision()=='high'
    for a,b in zip(result,expected):assert torch.equal(a,b)
    assert 'vecops_matmul_precision="highest"' in replay.ir
    with torch.autocast('cpu',dtype=torch.bfloat16):
      with pytest.raises(RuntimeError,match='autocast'):replay(x)
    # An internal autocast context is normalized by Torch before this recorder:
    # explicit cast operators and the BF16 matrix product enter the tape.
    inner,inner_replay,inner_expected=_compile(lambda x:_autocast_mm(x,weight),x)
    assert torch.equal(inner_replay(x),inner_expected)
  finally:torch.set_float32_matmul_precision(old)


def _autocast_mm(x,weight):
  with torch.autocast('cpu',dtype=torch.bfloat16):return x @ weight


def test_matmul_policy_restores_caller_after_native_exception():
  old=torch.get_float32_matmul_precision()
  try:
    torch.set_float32_matmul_precision('medium')
    table=torch.arange(8.)
    def model(index):
      try:
        torch.set_float32_matmul_precision('highest')
        return table[index]
      finally:torch.set_float32_matmul_precision('medium')
    _,replay,_=_compile(model,torch.tensor([1,2]))
    torch.set_float32_matmul_precision('high')
    with pytest.raises(RuntimeError,match='index'):
      replay(torch.tensor([99,2]))
    assert torch.get_float32_matmul_precision()=='high'
    assert torch.equal(replay(torch.tensor([1,2])),torch.tensor([1.,2.]))
  finally:torch.set_float32_matmul_precision(old)


def test_factory_default_dtype_is_frozen_per_observed_tensor_result():
  old=torch.get_default_dtype()
  try:
    torch.set_default_dtype(torch.float32)
    def model(x):
      try:
        torch.set_default_dtype(torch.float64)
        return torch.ones(7)+x,torch.empty(3).fill_(4),torch.randn(5),torch.ones_like(x)
      finally:torch.set_default_dtype(torch.float32)
    torch.manual_seed(77)
    capture,replay,expected=_compile(model,torch.ones(7))
    torch.set_default_dtype(torch.float16)
    actual=replay(torch.ones(7,dtype=torch.float32))
    for a,b in zip(actual,expected):assert torch.equal(a,b) and a.dtype==b.dtype
    assert torch.get_default_dtype()==torch.float16
  finally:torch.set_default_dtype(old)


def test_produced_capture_temporaries_are_released_before_compile():
  references=[]
  def model(x):
    y=x+1;references.append(weakref.ref(y))
    for i in range(200):y=y+1
    return y*2
  value=torch.ones(7)
  capture=graph.GraphCapture('weak-temporary-capture')
  expected=capture.capture_torch(model,value)
  gc.collect()
  assert references[0]() is None
  assert len(capture._torch_replay_tape.tensors)==1
  replay=capture.compile_torch()
  assert torch.equal(replay(value),expected)


@pytest.mark.parametrize('drop_original',[False,True])
def test_unrecorded_alias_of_produced_storage_is_rejected(drop_original):
  def model(x):
    value=x+1
    with torch._C._DisableTorchDispatch():hidden=value.detach()
    if drop_original:del value
    return hidden*2
  with pytest.raises(ValueError,match='unrecorded storage alias'):
    graph.GraphCapture('hidden-data-alias').capture_torch(model,torch.ones(8))
  assert graph.current_capture() is None


def test_hidden_python_dispatcher_tls_is_rejected_even_at_cpp_entry():
  _,replay,_=_compile(lambda x:x+1,torch.ones(8))
  with torch._C._EnablePythonDispatcher():
    with pytest.raises(RuntimeError,match='dispatch'):
      replay(torch.ones(8))
    with pytest.raises(RuntimeError,match='dispatch'):
      replay._executor.run([torch.ones(8)])
    with pytest.raises(RuntimeError,match='dispatch'):
      graph.GraphCapture('hidden-tls').capture_torch(lambda x:x+1,torch.ones(8))


def test_cpp_entry_rejects_wrapper_keys_before_virtual_metadata_or_python():
  _,replay,_=_compile(lambda x:x+1,torch.ones(8))
  class Wrapper(torch.Tensor):
    @staticmethod
    def __new__(cls):
      return torch.Tensor._make_wrapper_subclass(cls,(8,),strides=(1,),dtype=torch.float32,device='cpu')
    @classmethod
    def __torch_dispatch__(cls,*args,**kwargs):
      raise AssertionError('C++ rejection must not enter Python subclass dispatch')
  wrapper=Wrapper()
  with pytest.raises(RuntimeError,match='unsupported.*dispatch key'):
    replay._executor.run([wrapper])


def test_legacy_registration_module_owner_list_is_supported(monkeypatch):
  import vecops._schema_bridge as bridge
  owner=object()
  monkeypatch.setattr(bridge,'_loaded_bridge_modules',[owner])
  _,replay,expected=_compile(lambda x:x+1,torch.ones(8))
  assert owner in replay._owners
  assert torch.equal(replay(torch.ones(8)),expected)


def test_ordinary_data_property_is_dispatcher_visible_and_keeps_dynamic_alias():
  def model(x):
    value=x+1;hidden=value.data;del value
    return hidden*2
  capture,replay,expected=_compile(model,torch.ones(8))
  assert any(i.function._schema.name=='aten::detach' for i in capture._torch_replay_tape.instructions)
  assert torch.equal(replay(torch.ones(8)*7),torch.ones(8)*16)


@pytest.mark.parametrize('method',['numpy','tolist','item'])
def test_python_tensor_data_escapes_are_rejected_before_becoming_constants(method):
  def model(x):
    value=x.sum()
    if method=='numpy':return torch.from_numpy(value.numpy().copy())
    if method=='tolist':return value.tolist()
    return value.item()
  with pytest.raises(ValueError,match='unsupported.*data escape'):
    graph.GraphCapture('data-escape').capture_torch(model,torch.ones(8))
  assert graph.current_capture() is None


def test_distinct_storage_pointer_alias_of_produced_tensor_is_rejected():
  import ctypes
  def model(x):
    value=x+1
    raw=(ctypes.c_float*8).from_address(value.data_ptr());raw._tensor_owner=value
    hidden=torch.frombuffer(raw,dtype=torch.float32)
    return hidden*2
  with pytest.raises(ValueError,match='tensor data escape.*data_ptr'):
    graph.GraphCapture('cross-storage-hidden-alias').capture_torch(model,torch.ones(8))


def test_capture_and_private_arena_do_not_need_cyclic_gc_to_release():
  was_enabled=gc.isenabled();gc.disable()
  try:
    value=torch.arange(8.)
    capture=graph.GraphCapture('no-capture-cycle')
    capture.capture_torch(_factory_model,value)
    reference=weakref.ref(capture)
    system=_System();plan=_factory_plan(capture._torch_execution_graph)
    replay=capture.compile_torch(memory_plan=plan,system=system,large_pages=False)
    del capture
    assert reference() is None, 'mode classes, weak identity callbacks and tape must not retain capture'
    assert any(owner() is not None for owner in system.allocations)
    del replay
    assert all(owner() is None for owner in system.allocations), 'private arenas must release without gc.collect()'
  finally:
    if was_enabled:gc.enable()


def test_compile_error_releases_arena_even_while_traceback_is_retained_and_can_retry():
  def model(x):
    temporary=planner.empty_like(x);temporary.copy_(x)
    return temporary.view(-1).as_strided((4,),(1,),0)*2
  value=torch.ones(8)
  capture=graph.GraphCapture('compile-rollback')
  expected=capture.capture_torch(model,value)
  original={key:tensor.data_ptr()for key,tensor in capture._torch_replay_tape.storage_bindings.items()}
  system=_System();plan=_factory_plan(capture._torch_execution_graph)
  was_enabled=gc.isenabled();gc.disable()
  try:
    with pytest.raises(ValueError,match='storage-offset') as failure:
      capture.compile_torch(memory_plan=plan,system=system,large_pages=False)
    assert failure.value.__traceback__ is not None
    assert all(owner()is None for owner in system.allocations)
    assert {key:tensor.data_ptr()for key,tensor in capture._torch_replay_tape.storage_bindings.items()}==original
    replay=capture.compile_torch()
    assert torch.equal(replay(value),expected)
  finally:
    if was_enabled:gc.enable()


def test_tensor_build_without_named_tensor_property(monkeypatch):
  # Some Torch builds omit the Python named-tensor property entirely. Ordinary
  # dense tensors must still pass replay's boundary checks on those builds.
  def unavailable(self):
    raise AttributeError('named tensor support is unavailable')
  monkeypatch.setattr(torch.Tensor, 'names', property(unavailable))
  _, replay, _ = _compile(lambda x: x + 1, torch.arange(7.))
  torch.testing.assert_close(replay(torch.arange(7.) + 3), torch.arange(7.) + 4)
