# SPDX-FileCopyrightText: 2026 HenryRenYz and vecops contributors
# SPDX-License-Identifier: MIT
"""Bound external storage retains its explicit Torch/native physical owner.

Graph-owned buffers are prepared by resource programs. They are never copied
out of a separate capture-time arena, so offsets and alias guards refer to the
same canonical storage used by the program.
"""


def isolate_bindings(bindings,inputs,leases,*,borrow_readonly_state=False,
                     tensor_spans=None,written_ids=()):
  if leases:
    raise ValueError('capture-time allocator arenas are unsupported; bind a resource program')
  return dict(bindings),[],[],set(),[]
