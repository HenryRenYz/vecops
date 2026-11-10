# SPDX-FileCopyrightText: 2026 HenryRenYz and vecops contributors
# SPDX-License-Identifier: MIT
"""Acquire immutable graph and current budget views from runtime authority."""

def resource_snapshot(runtime,scope=None,*,caches=()):
  runtime=getattr(runtime,'native',runtime)
  view=runtime.view(scope if scope is not None else runtime.current_scope())
  return view,runtime.snapshot(view,list(caches))
