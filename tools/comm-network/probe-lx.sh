#!/usr/bin/env bash
# SPDX-FileCopyrightText: 2026 HenryRenYz and vecops contributors
# SPDX-License-Identifier: MIT
set -euo pipefail
# Executed by lx-run, which sources bashrc and activates af3 on compute.
task_root=$HOME/vecops-neo/build/multinode-probe-20261104
mkdir -p "$task_root"
printf 'HOST=%s RANK=%s HOME=%s\n' "$(hostname)" "${LX_WORLD_RANK:-${PMIX_RANK:-unknown}}" "$HOME"
command -v clang++
command -v mpicc
mpicc --showme:version
python - <<'PY'
import torch,torch.distributed as dist
print({'torch':torch.__version__,'mpi_available':dist.is_mpi_available(),'gloo_available':dist.is_gloo_available()},flush=True)
PY
if command -v ucx_info >/dev/null; then ucx_info -v; ucx_info -d; fi
for task_device in /sys/class/infiniband/*; do
  test -d "$task_device" || continue
  printf 'RDMA_DEVICE=%s NUMA=' "$task_device"
  cat "$task_device/device/numa_node"
  for task_rate in "$task_device"/ports/*/rate; do cat "$task_rate"; done
done
# Each host compiles its own executable; no shared built-marker race.
task_binary="$task_root/mpi_proxy_probe-$(hostname)"
clang++ -std=c++20 -O3 -pthread $(mpicc --showme:compile) \
  "$task_root/mpi_proxy_probe.cpp" $(mpicc --showme:link) -o "$task_binary"
sha256sum "$task_root/mpi_proxy_probe.cpp" "$task_binary"
ldd "$task_binary"
numactl --membind=0 "$task_binary"
