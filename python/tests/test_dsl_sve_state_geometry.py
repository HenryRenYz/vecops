# SPDX-FileCopyrightText: 2026 HenryRenYz and vecops contributors
# SPDX-License-Identifier: MIT
"""Typed native words retain geometry through snapshots and register transforms."""

import vecops.language as T
from vecops._arm_vector_state import VectorState
from vecops.language._compiler import ir as I
from vecops.language._compiler.backends.cpu.backend import CPUBackend


@T.template
def alias_kernel(A: T.Tensor((128,), dtype="bfloat16"), B: T.Tensor((128,), dtype="bfloat16"),
                 O: T.Tensor((64,), dtype="float32", access="output"), choose: T.int64):
  count = T.lanes(T.Vector("bfloat16"))
  a = T.load(A[:count], vector=T.Vector("bfloat16"))
  b = T.load(B[:count], vector=T.Vector("bfloat16"))
  joined = T.sve.join_k_groups(a, b)
  snapshot = joined
  if choose != 0:
    snapshot = T.sve.join_k_groups(b, a)
  accumulator = T.full(T.Vector("float32"), 0)
  value = T.sve.mmla(T.sve.broadcast_segment(snapshot, 0), joined, accumulator)
  T.store(O[:T.lanes(T.Vector("float32"))], value)


def test_complete_sve_word_aliases_do_not_split_into_half_registers():
  artifact = CPUBackend().lower(alias_kernel.ir, (), target="SME", options={"_vector_state": VectorState(64, 64)})
  complete = [p for p in artifact.backend_ir.states.values()
              if isinstance(p.state.type, I.VectorType) and p.state.type.dtype == "bfloat16" and
              I.same_expr(p.state.type.shape[0], I.Expr(I.ScalarType("int64"), "lanes", data=("bfloat16", 0)))]
  assert len(complete) >= 3
  assert all(len(p.pieces) == 1 and p.pieces[0].scalable and p.pieces[0].power == 0 for p in complete)


@T.template
def partial_kernel(O: T.Tensor((T.dynamic(lower=0),), dtype="uint32", access="output")):
  count = T.min(O.shape[0], T.lanes(T.Vector("uint32")))
  total = T.full(T.Vector("uint32"), 17, active=count)
  for ki in T.serial(2):
    signed = T.as_vector(T.bitcast(total, "int32"), T.Vector("int32"))
    total = T.as_vector(T.bitcast(signed, "uint32"), T.Vector("uint32"))
  T.store(O[:count], total)


def test_dynamic_active_word_transform_keeps_existing_register_piece():
  artifact = CPUBackend().lower(partial_kernel.ir, (), target="SME", options={"_vector_state": VectorState(64, 64)})
  unsigned = [p for p in artifact.backend_ir.states.values()
              if isinstance(p.state.type, I.VectorType) and p.state.type.dtype == "uint32" and
              p.state.type.tag.basis_dtype == "uint32"]
  assert unsigned
  assert all(len(p.pieces) == 1 and p.pieces[0].scalable and p.pieces[0].power == 0 for p in unsigned)
  assert "state_next" not in artifact.source
