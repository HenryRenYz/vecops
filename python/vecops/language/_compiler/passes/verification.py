# SPDX-FileCopyrightText: 2026 HenryRenYz and vecops contributors
# SPDX-License-Identifier: MIT
"""Named verifier invocations at representation boundaries."""

from ..backends.cpu.verification import verify_prepared
from ..ir.verification import verify_kernel, verify_vector
from .manager import PassResult, register_pass


@register_pass("verify-kernel", stage="kernel-cleanup")
def kernel(ir, context):
  if context.options.get("verify_ir", True):
    verify_kernel(ir)
  return PassResult(ir, changed=False, preserved=tuple(context.analyses))


@register_pass("verify-vector", stage="vector", after=("cpu-legalize-math", "cpu-plan-native-word-tails"))
def vector(ir, context):
  if context.options.get("verify_ir", True):
    verify_vector(ir)
  return PassResult(ir, changed=False, preserved=tuple(context.analyses))


@register_pass("verify-cpu-plan", stage="cpu-memory", after="cpu-plan-private-captures")
def prepared(ir, context):
  if context.options.get("verify_ir", True):
    verify_prepared(ir)
  return PassResult(ir, changed=False, preserved=tuple(context.analyses))
