# SPDX-FileCopyrightText: 2026 HenryRenYz and vecops contributors
# SPDX-License-Identifier: MIT
"""Benchmark allocation uses the transport's selected resource-runtime pool."""
def pool(*,base_pages=False):
  from vecops import memory
  from vecops.distributed import _payload_runtime
  runtime,identifier,_page=_payload_runtime()
  factory=memory.Pool.create(runtime,identifier,allowed_pools=[identifier],
    page_bytes=4096 if base_pages else 0,retain_idle_bytes=0)
  return factory,identifier,runtime
