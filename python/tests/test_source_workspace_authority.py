# SPDX-FileCopyrightText: 2026 HenryRenYz and vecops contributors
# SPDX-License-Identifier: MIT
"""Generated non-recordable kernels still use the embedding resource authority."""
import gc
import pytest
import torch
import vecops
import vecops.language as T
from test_graph_replay import _resource_runtime, _committed


@pytest.fixture(scope='module',params=['context','owner'])
def source_workspace_call(request,tmp_path_factory):
  root=tmp_path_factory.mktemp('source-workspace-authority')
  path=root/'kernel.cpp'
  if request.param=='context':
    signature='execution::WorkspaceContext& workspace, auto x, auto y'
    body='auto slot=workspace.request("temp", {.bytes=x.size(0)*nint_t(sizeof(float))}); auto* temp=static_cast<float*>(slot.replica());'
  else:
    signature='auto x, auto y'
    body='kernel::Workspace owner(x.size(0)*nint_t(sizeof(float))); auto view=owner.view(); auto* temp=view.template allocate<float>(x.size(0));'
  path.write_text('''
// SPDX-FileCopyrightText: 2026 HenryRenYz and vecops contributors
// SPDX-License-Identifier: MIT
#include "vecops/Kernel.h"
#include "vecops/execution/WorkspaceContext.h"
using namespace vecops;
void __kernel__('''+signature+''') {
  '''+body+'''
  for(nint_t i=0;i<x.size(0);++i) temp[i]=x.data()[i]+1;
  for(nint_t i=0;i<x.size(0);++i) y.data()[i]=temp[i]*2;
}
''')
  @T.jit(out_idx=[1])
  def factory():
    N=T.dynamic(lower=1)
    @T.template(source=str(path),recipe_id='source-workspace-authority-'+request.param)
    def kernel(x:T.Tensor((N,),(1,),T.float32),y:T.Tensor((N,),(1,),T.float32)):
      pass
    return kernel
  return factory().explicit_torch_function()


def test_dynamic_and_owning_source_workspaces_respect_one_budget_and_release(source_workspace_call):
  from vecops.memory import Runtime,Pool,use_runtime
  runtime=_resource_runtime()
  arena=Runtime(runtime).create_arena('eager',allowed_pools=['uma'],hard_limit=1<<20,
      grow_by_bytes=4096,retain_idle_bytes=0,page_bytes=4096)
  held=Pool(arena).empty((32,),dtype=torch.float32).fill_(19)
  with use_runtime(runtime),torch.inference_mode():
    x,y=torch.arange(37.),torch.empty(37)
    for _ in range(2):
      source_workspace_call(x,y)
      torch.testing.assert_close(y,(x+1)*2,rtol=0,atol=0)
      assert _committed(runtime)==4096
    x,y=torch.ones((1<<20)//4+1),torch.full(((1<<20)//4+1,),-17.)
    with pytest.raises(RuntimeError,match='capacity|workspace|allocation|fit|budget'):
      source_workspace_call(x,y)
    torch.testing.assert_close(y,torch.full_like(y,-17),rtol=0,atol=0)
    assert _committed(runtime)==4096
    torch.testing.assert_close(held,torch.full_like(held,19),rtol=0,atol=0)
  del held;gc.collect();arena.trim(0)
  assert _committed(runtime)==0
