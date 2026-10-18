# SPDX-FileCopyrightText: 2026 HenryRenYz and vecops contributors
# SPDX-License-Identifier: MIT
"""Inline matrix dispatch preserves native and SIMD tails, seeds and aliases."""

import pytest
import vecops.language as T
from test_dsl_outline_vector_fallback import (
  contains_disjoint_guard,
  nodes,
  numeric_template,
  template,
)
from vecops.language._compiler import ir as I
from vecops.language._compiler.backends.cpu.backend import CPUBackend


def test_inline_dispatch_keeps_architectural_owners_inside_the_kernel():
  artifact = CPUBackend().lower(
    template().ir, (), target="AVX512", options={"cpu_outline_vector_fallback": False},
  )
  walked = list(nodes(artifact.kernel.body))
  assert any(isinstance(node, I.HardwareRegion) and node.kind == "amx" for node in walked)
  calls = [node for node in walked if isinstance(node, I.ExitRegion) and node.outlined]
  # The alias-preserving SIMD dispatcher has its own boundary. Native owners
  # and the proven-disjoint branch remain in the caller.
  assert calls
  assert all(not contains_disjoint_guard(call.body) for call in calls)
  assert all(not any(isinstance(node, I.HardwareRegion) for node in nodes(call.body)) for call in calls)
  assert not any(isinstance(node, I.ExitRegion) and node.outlined for call in calls for node in nodes(call.body))


@pytest.fixture(scope="module")
def native_inline_dispatch(tmp_path_factory):
  import vecops
  from dsl_test_utils import amx_available_for_test, make_compiler

  if not amx_available_for_test():
    pytest.skip("requires an AMX target to exercise native matrix dispatch")
  root = tmp_path_factory.mktemp("inline-matrix-dispatch")
  compiled = T.compile(
    numeric_template(),
    compiler=make_compiler(jobs=2, cache_dir=root / "cache", build_dir=root / "build"),
    pass_options={"cpu_outline_vector_fallback": False},
  )
  arguments = (
    vecops.TensorMeta((33, 65), dtype="bfloat16"),
    vecops.TensorMeta((65, 35), dtype="bfloat16"),
    vecops.TensorMeta((1, 35), dtype="float32"),
    vecops.TensorMeta((33, 35), dtype="float32", access=vecops.TensorAccess.output),
  )
  with vecops.collect_compile_requests() as collector:
    compiled.operator(*arguments, **compiled._core.specialization_values)
  assert len(collector.requests) == 1
  vecops.compile_batch(collector.requests, parallelism=2, execution_parallelism=1)
  return compiled


@pytest.mark.parametrize(
  "m,n,k,alias",
  [
    (1, 8, 0, "prior"),
    (1, 8, 1, "prior"),
    (1, 8, 65, "prior"),
    (8, 1, 65, "prior"),
    (1, 8, 65, "A"),
    (1, 8, 65, "B"),
    (32, 32, 17, "prior"),
    (33, 35, 65, "prior"),
    (33, 35, 65, "A"),
    (33, 35, 65, "B"),
  ],
)
def test_native_inline_dispatch_preserves_tail_seed_epilogue_and_alias(native_inline_dispatch, m, n, k, alias):
  from test_dsl_outline_vector_fallback import (
    test_native_outlined_fallback_preserves_tail_seed_epilogue_and_alias as check_numeric,
  )

  # Retain the independent FP64 oracle, mutable aliased Prior, A/B shared
  # storage and untouched-byte canaries for the explicitly inline boundary.
  check_numeric(native_inline_dispatch, m, n, k, alias)
