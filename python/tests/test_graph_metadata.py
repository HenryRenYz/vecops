# SPDX-FileCopyrightText: 2026 HenryRenYz and vecops contributors
# SPDX-License-Identifier: MIT
"""Executable construction must not evaluate real inputs or mutate RNG/state."""
import torch
import pytest
from torch._subclasses.fake_tensor import is_fake

from vecops.graph import GraphCapture
from vecops.graph._metadata import bind_metadata, metadata_mode


def test_metadata_mutation_and_rng_are_deferred_until_replay():
  x = torch.arange(6.).reshape(2,3)
  original = x.clone()
  rng = torch.random.get_rng_state().clone()
  def fn(a):
    a.add_(2)
    return a + torch.randn_like(a)
  capture = GraphCapture('metadata-mutating-random')
  with metadata_mode() as mode:
    result = capture.capture_torch_metadata(fn, mode.from_tensor(x))
    assert is_fake(result)
  assert torch.equal(x, original)
  assert torch.equal(torch.random.get_rng_state(), rng)
  bind_metadata(capture,(x,))
  executable = capture.compile_torch()
  assert torch.equal(x, original)
  assert torch.equal(torch.random.get_rng_state(), rng)
  expected_input = original.clone()
  expected = fn(expected_input)
  torch.random.set_rng_state(rng)
  torch.testing.assert_close(executable(x), expected,rtol=0,atol=0)
  torch.testing.assert_close(x,expected_input,rtol=0,atol=0)


def test_metadata_initializer_outputs_bind_steady_without_execution():
  x = torch.arange(6.).reshape(2,3)
  cold = GraphCapture('metadata-init')
  steady = GraphCapture('metadata-steady')
  with metadata_mode() as mode:
    fake = mode.from_tensor(x)
    fake_values = cold.capture_torch_metadata(lambda a: (a*3).transpose(0,1), fake)
    steady.capture_torch_metadata(lambda a,v:a+v.transpose(0,1), fake, fake_values)
  with torch.device('meta'):
    bind_metadata(cold,(x,))
  initial = cold.compile_torch(native_pointwise=True,borrow_outputs=True)
  values = initial.metadata_outputs
  with torch.device('meta'):
    bind_metadata(steady,(x,values))
  executable = steady.compile_torch(native_pointwise=True,borrow_outputs=True)
  assert values.device.type == 'cpu'
  actual_values = initial(x)
  assert actual_values.stride()==values.stride()
  assert actual_values.untyped_storage().nbytes()==values.untyped_storage().nbytes()
  torch.testing.assert_close(executable(x,actual_values),x*4,rtol=0,atol=0)


def test_cpu_metadata_prototypes_preserve_alias_offsets_under_meta_default():
  from vecops.graph._metadata import prototypes
  source = torch.arange(24.).reshape(4,6)
  with metadata_mode() as mode:
    fake = mode.from_tensor(source)
    aliases = (fake, fake[1:,::2])
  with torch.device('meta'):
    whole, sliced = prototypes(aliases)
  assert whole.device.type == sliced.device.type == 'cpu'
  assert whole.untyped_storage()._cdata == sliced.untyped_storage()._cdata
  assert sliced.storage_offset() == 6 and sliced.stride() == (6,2)
  whole.fill_(7)
  assert torch.equal(sliced,whole[1:,::2])


def test_metadata_rejects_data_dependent_branch():
  with metadata_mode() as mode:
    with pytest.raises((ValueError, RuntimeError),match='escape|data-dependent'):
      GraphCapture().capture_torch_metadata(lambda a:a+int(a.sum()),mode.from_tensor(torch.ones(2)))


def test_metadata_rejects_changed_input_alias_contract():
  source=torch.arange(8.)
  capture=GraphCapture('metadata-alias')
  with metadata_mode() as mode:
    capture.capture_torch_metadata(lambda a,b:a+b,mode.from_tensor(source[:4]),mode.from_tensor(source[4:]))
  with pytest.raises(ValueError,match='alias contract'):
    bind_metadata(capture,(torch.ones(4),torch.ones(4)))


def test_metadata_disables_real_cpu_fallback():
  calls=[]
  library=torch.library.Library('vecops_metadata_no_fallback','DEF')
  library.define('run(Tensor a) -> Tensor')
  def cpu(a):
    calls.append(1)
    return a+1
  library.impl('run',cpu,'CPU')
  with metadata_mode() as mode:
    with pytest.raises(RuntimeError,match='unsupported|no_fallback'):
      torch.ops.vecops_metadata_no_fallback.run(mode.from_tensor(torch.ones(2)))
  assert calls==[]
  import vecops
  with pytest.raises(RuntimeError,match='unsupported|no_fallback'):
    vecops.precompile(lambda x:torch.ops.vecops_metadata_no_fallback.run(x),torch.ones(2))
  assert calls==[]


def test_metadata_template_restores_current_state_without_capture_or_execution():
  import json
  from vecops.graph._template import export_capture,restore_metadata_capture
  class Owner(torch.nn.Module):
    def __init__(self,value):
      super().__init__();self.register_buffer('weight',torch.full((2,3),value))
    def forward(self,x):return (x+self.weight).transpose(0,1)
  owner=Owner(2.)
  x=torch.ones(2,3)
  with torch.inference_mode():
    cold=GraphCapture('metadata-template')
    with metadata_mode() as mode:cold.capture_torch_metadata(owner,mode.from_tensor(x))
    bind_metadata(cold,(x,))
    payload=json.loads(json.dumps(export_capture(cold,owner,(x,))))
    other=Owner(7.);current=torch.full_like(x,4.)
    before=current.clone();rng=torch.random.get_rng_state().clone()
    restored=GraphCapture('metadata-template-restored')
    with metadata_mode() as mode:
      result=restore_metadata_capture(restored,payload,other,(mode.from_tensor(current),))
      assert is_fake(result)and result.shape==(3,2)
    bind_metadata(restored,(current,))
    executable=restored.compile_torch(borrow_readonly_state=True,native_pointwise=False)
    assert torch.equal(current,before)and torch.equal(rng,torch.random.get_rng_state())
    torch.testing.assert_close(executable(current),other(current),rtol=0,atol=0)


@pytest.mark.parametrize('overlapping',[False,True])
def test_metadata_template_guards_physical_state_overlap(overlapping):
  from vecops.graph._template import export_capture,restore_metadata_capture
  class Owner(torch.nn.Module):
    def __init__(self,overlap):
      super().__init__()
      raw=bytearray(48)
      self.register_buffer('a',torch.frombuffer(raw,dtype=torch.float32,count=8))
      self.register_buffer('b',torch.frombuffer(raw if overlap else bytearray(48),
                                              dtype=torch.float32,count=8,offset=16))
    def forward(self,x):return x+self.a+self.b
  source=Owner(overlapping);x=torch.ones(8)
  capture=GraphCapture('state-overlap')
  with metadata_mode() as mode:capture.capture_torch_metadata(source,mode.from_tensor(x))
  bind_metadata(capture,(x,))
  payload=export_capture(capture,source,(x,))
  with metadata_mode() as mode:
    restore_metadata_capture(GraphCapture('same-overlap'),payload,Owner(overlapping),(mode.from_tensor(x),))
    with pytest.raises(ValueError,match='physical alias'):
      restore_metadata_capture(GraphCapture('changed-overlap'),payload,Owner(not overlapping),(mode.from_tensor(x),))
