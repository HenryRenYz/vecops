# SPDX-FileCopyrightText: 2026 HenryRenYz and vecops contributors
# SPDX-License-Identifier: MIT
"""Author-audited C++ alias contracts plus independent graph donation proof."""
import pytest
import torch
import vecops.language as T
from vecops.graph import GraphCapture


@pytest.fixture(scope='module')
def native_call(tmp_path_factory):
  source = tmp_path_factory.mktemp('native-donation')/'kernel.cpp'
  source.write_text('''
// SPDX-FileCopyrightText: 2026 HenryRenYz and vecops contributors
// SPDX-License-Identifier: MIT
#define VECOPS_RECORDABLE_KERNEL 1
#include "vecops/Kernel.h"
#include "vecops/execution/KernelPlan.h"
using namespace vecops;
constexpr nint_t P = spec::Parallelism;
void __kernel__(execution::KernelBuilder& b, auto x, auto y) {
  b.allow_inplace(x, y);
  b.grid<P>("double", 0, x.size(0), 1, execution::kernel_bind(execution::read(x), execution::write(y)),
    [](auto, auto work, auto x, auto y) { y(work.begin()) = x(work.begin()) * static_cast<float>(spec::Factor); });
}
''')
  @T.jit(out_idx=[1])
  def factory():
    N = T.dynamic(lower=0)
    Factor = T.Const(2)
    @T.template(source=str(source), recipe_id='native-donation-proof-v1')
    def kernel(x: T.Tensor((N,), (1,), T.float32), y: T.Tensor((N,), (1,), T.float32)):
      Factor
    return kernel
  torch.set_num_threads(4)
  return factory().explicit_torch_function()


@pytest.mark.parametrize('metadata_op', ['reshape_as','view_as','expand_as'])
@pytest.mark.parametrize('view_input', [False, True])
@pytest.mark.parametrize('keep_old', [False, True])
def test_cpp_inplace_permission_requires_dead_internal_input(native_call, monkeypatch, keep_old, view_input, metadata_op):
  monkeypatch.setenv('VECOPS_THREAD_POOL', 'native')
  def function(x):
    a = x + 1
    if view_input:
      a = a.view(1, -1).reshape_as(x)
    output = torch.empty_like(a)
    native_call(a, output)
    shaped = getattr(output,metadata_op)(a)
    return shaped + (a.reshape_as(shaped) if keep_old else 3)
  with torch.inference_mode():
    x = torch.arange(37.)
    capture = GraphCapture('cpp-inplace-contract')
    capture.capture_torch(function, x)
    program = capture.compile_torch(native_pointwise=True, native_donation=True, borrow_outputs=True)
    assert program.allocator_policy['native_kernel_donations'] == (0 if keep_old else 1)
    for repeat in range(4):
      value = x + repeat
      saved = value.clone()
      torch.testing.assert_close(program(value), function(value), rtol=0, atol=0)
      torch.testing.assert_close(value, saved, rtol=0, atol=0)


@pytest.mark.parametrize('view_input', [False, True])
def test_external_input_is_not_donated(native_call, monkeypatch, view_input):
  monkeypatch.setenv('VECOPS_THREAD_POOL', 'native')
  def function(x):
    if view_input:
      x = x.view(1, -1).reshape_as(x)
    output = torch.empty_like(x)
    native_call(x, output)
    return output
  with torch.inference_mode():
    x = torch.arange(17.)
    capture = GraphCapture('cpp-external-no-donation')
    capture.capture_torch(function, x)
    program = capture.compile_torch(native_pointwise=True, native_donation=True, borrow_outputs=True)
    assert program.allocator_policy['native_kernel_donations'] == 0
    torch.testing.assert_close(program(x), x * 2, rtol=0, atol=0)
    torch.testing.assert_close(x, torch.arange(17.), rtol=0, atol=0)


@pytest.fixture(scope='module')
def opaque_call(tmp_path_factory):
  source = tmp_path_factory.mktemp('opaque-donation')/'kernel.cpp'
  source.write_text('''
// SPDX-FileCopyrightText: 2026 HenryRenYz and vecops contributors
// SPDX-License-Identifier: MIT
#include "vecops/Kernel.h"
void __kernel__(auto x, auto y) { y(0) = x(0); }
''')
  @T.jit(out_idx=[1])
  def factory():
    @T.template(source=str(source), recipe_id='opaque-donation-boundary-v1')
    def kernel(x: T.Tensor((17,), (1,), T.float32), y: T.Tensor((1,), (1,), T.float32)):
      pass
    return kernel
  return factory().explicit_torch_function()


def test_opaque_or_peer_provider_uses_prevent_donation(opaque_call, monkeypatch):
  monkeypatch.setenv('VECOPS_THREAD_POOL', 'native')
  def function(x):
    a = x + 1
    out = torch.empty(1)
    opaque_call(a, out)
    return a.relu().sum()
  with torch.inference_mode():
    x = torch.arange(17.) - 8
    capture = GraphCapture('opaque-no-donation')
    capture.capture_torch(function, x)
    program = capture.compile_torch(native_pointwise=True, native_donation=True)
    assert program.allocator_policy['native_pointwise_donations'] == 0
    torch.testing.assert_close(program(x), function(x), rtol=0, atol=0)


def test_native_donation_resolves_an_unhoisted_view_at_its_use(native_call, monkeypatch):
  monkeypatch.setenv('VECOPS_THREAD_POOL','native')
  def function(x):
    internal = x + 1
    # This view is deliberately absent from the static-view hoisting list.
    view = torch.ops.aten._reshape_alias.default(internal,[37],[1])
    output = torch.empty_like(view)
    native_call(view,output)
    return output
  with torch.inference_mode():
    x = torch.arange(37.)
    capture = GraphCapture('cpp-unhoisted-view-donation')
    capture.capture_torch(function,x)
    program = capture.compile_torch(native_pointwise=True,native_donation=True,borrow_outputs=True)
    assert program.allocator_policy['native_kernel_donations']==1
    for repeat in range(3):
      value = x + repeat
      torch.testing.assert_close(program(value),function(value),rtol=0,atol=0)


def test_predefinition_output_template_stays_conservative(native_call, monkeypatch):
  monkeypatch.setenv('VECOPS_THREAD_POOL','native')
  def function(x):
    output = torch.empty_like(x)
    internal = (x+1).view_as(output)
    native_call(internal,output)
    return output
  with torch.inference_mode():
    x = torch.arange(37.)
    capture = GraphCapture('cpp-predefinition-output-template')
    capture.capture_torch(function,x)
    program = capture.compile_torch(native_pointwise=True,native_donation=True,borrow_outputs=True)
    assert program.allocator_policy['native_kernel_donations']==0
    torch.testing.assert_close(program(x),function(x),rtol=0,atol=0)


@pytest.fixture(scope='module')
def native_two_input_call(tmp_path_factory):
  source = tmp_path_factory.mktemp('native-other-alias')/'kernel.cpp'
  source.write_text('''
// SPDX-FileCopyrightText: 2026 HenryRenYz and vecops contributors
// SPDX-License-Identifier: MIT
#define VECOPS_RECORDABLE_KERNEL 1
#include "vecops/Kernel.h"
#include "vecops/execution/KernelPlan.h"
using namespace vecops;
constexpr nint_t P = spec::Parallelism;
void __kernel__(execution::KernelBuilder& b, auto x, auto other, auto y) {
  b.allow_inplace(x, y);
  b.grid<P>("double", 0, x.size(0), 1,
    execution::kernel_bind(execution::read(x), execution::write(y)),
    [](auto, auto work, auto x, auto y) { y(work.begin()) = 2.0f*x(work.begin()); });
  b.grid<P>("other_read_after_output", 0, y.size(0), 1,
    execution::kernel_bind(execution::read(other), execution::read_write(y)),
    [](auto, auto work, auto other, auto y) { y(work.begin()) += other(work.begin()); });
}
''')
  @T.jit(out_idx=[2])
  def factory():
    N = T.dynamic(lower=0)
    @T.template(source=str(source), recipe_id='native-other-alias-proof-v1')
    def kernel(x: T.Tensor((N,), (1,), T.float32),
               other: T.Tensor((N,), (1,), T.float32),
               y: T.Tensor((N,), (1,), T.float32)):
      pass
    return kernel
  torch.set_num_threads(4)
  return factory().explicit_torch_function()


@pytest.mark.parametrize('other_alias', ['identical', 'view', 'separate'])
def test_native_donation_excludes_every_other_argument_alias(native_two_input_call, monkeypatch, other_alias):
  monkeypatch.setenv('VECOPS_THREAD_POOL','native')
  def function(x):
    internal = x + 1
    other = internal if other_alias == 'identical' else (
        internal.view_as(internal) if other_alias == 'view' else x + 5)
    output = torch.empty_like(internal)
    native_two_input_call(internal,other,output)
    return output
  with torch.inference_mode():
    x = torch.arange(37.)
    capture = GraphCapture('cpp-other-argument-alias')
    capture.capture_torch(function,x)
    program = capture.compile_torch(native_pointwise=True,native_donation=True,borrow_outputs=True)
    assert program.allocator_policy['native_kernel_donations'] == (1 if other_alias == 'separate' else 0)
    for repeat in range(3):
      value = x + repeat
      saved = value.clone()
      torch.testing.assert_close(program(value),function(value),rtol=0,atol=0)
      torch.testing.assert_close(value,saved,rtol=0,atol=0)
