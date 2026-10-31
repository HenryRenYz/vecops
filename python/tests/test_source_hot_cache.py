# SPDX-FileCopyrightText: 2026 HenryRenYz and vecops contributors
# SPDX-License-Identifier: MIT
"""Repeating source signatures and workspace-context replacement."""
import pytest
import torch
import vecops
import vecops.language as T


@pytest.fixture(scope='module')
def source_call(tmp_path_factory):
  path = tmp_path_factory.mktemp('source-hot-cache')/'kernel.cpp'
  path.write_text('''
// SPDX-FileCopyrightText: 2026 HenryRenYz and vecops contributors
// SPDX-License-Identifier: MIT
#define VECOPS_RECORDABLE_KERNEL 1
#include "vecops/Kernel.h"
#include "vecops/execution/KernelPlan.h"
using namespace vecops;
using namespace vecops::tensor;
constexpr nint_t P = spec::Parallelism;
void __kernel__(execution::KernelBuilder& b, auto x, auto y) {
  auto temp = b.tensor<float32_t>("temp", make_shape(x.size(0)));
  b.grid<P>("first", nint_t{0}, x.size(0), cint<1>, execution::kernel_bind(x, temp),
      [](execution::TaskContext<P>, const auto& work, auto x, auto temp) {
        auto i = work.begin(); temp.data()[i] = x.data()[i] + 1.f;
      });
  b.grid<P>("second", nint_t{0}, x.size(0), cint<1>, execution::kernel_bind(temp, y),
      [](execution::TaskContext<P>, const auto& work, auto temp, auto y) {
        auto i = work.begin(); y.data()[i] = temp.data()[i] * 2.f;
      });
}
''')
  @T.jit(out_idx=[1])
  def factory():
    N = T.dynamic(lower=1)
    @T.template(source=str(path), recipe_id='source-hot-cache-context-test')
    def kernel(x: T.Tensor((N,), (1,), T.float32),
               y: T.Tensor((N,), (1,), T.float32)):
      pass
    return kernel
  torch.set_num_threads(4)
  return factory().explicit_torch_function()


def test_more_than_eight_repeating_metadata_signatures_keep_prepared_plans(source_call):
  arrays = [(torch.arange(float(n)), torch.empty(n)) for n in range(17, 31)]
  with torch.inference_mode():
    for x,y in arrays:
      source_call(x,y)
    vecops._C.reset_bridge_hot_cache_stats()
    for x,y in arrays:
      source_call(x,y)
      torch.testing.assert_close(y,(x+1)*2,rtol=0,atol=0)
    stats = vecops._C.bridge_hot_cache_stats()
    if stats['enabled'] and stats['capacity'] >= len(arrays):
      assert stats['hits'] == len(arrays)
      assert stats['prepares'] == 0 and stats['evictions'] == 0


def test_identical_metadata_in_a_new_workspace_reprepares_instead_of_failing(source_call):
  system = vecops.memory.System()
  x, y = torch.arange(37.), torch.empty(37)
  with torch.inference_mode():
    for offset in range(2):
      with system.workspace_session(fast_capacity=0, slow_capacity=4096):
        source_call(x+offset,y)
        torch.testing.assert_close(y,(x+offset+1)*2,rtol=0,atol=0)
