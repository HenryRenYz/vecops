SPDX-FileCopyrightText: 2026 HenryRenYz and vecops contributors
SPDX-License-Identifier: MIT

Two-node MPI and bounded SPSC dispatch probe

This is a standalone baseline tool, not a production vecops network backend.
It requires exactly 2 MPI ranks, one rank per host, and exactly 2 allowed CPUs
per rank. Payload MPI calls use the same physical CPU in both modes. The other
CPU deliberately busy-polls, so the result does not demonstrate compute overlap,
CPU sleep behavior, or cross-process agent overhead. There is no Torch MPI
speed comparison. AllGather/AllToAll include their library-local work.

From the vecops-neo checkout on the local machine:

  ssh 920f-lx 'bash -l -s' <<'REMOTE'
  source ~/.bashrc
  mkdir -p "$HOME/vecops-neo/build/multinode-probe-20261104"
  REMOTE
  rsync -az tools/comm-network/mpi_proxy_probe.cpp 920f-lx:vecops-neo/build/multinode-probe-20261104/
  ~/.local/bin/lx-run submit --mode mpi --nodes 2 --ranks-per-node 1 \
    --cpus-per-rank 2 --cpu-list '0-1' --mem-per-rank 2G --exclusive \
    --walltime 900 --cwd '~/vecops-neo' --env UCX_MAX_RNDV_RAILS=1 \
    --env UCX_LOG_LEVEL=info --script tools/comm-network/probe-lx.sh

Read the user's 920f-lx cluster guide first. Check queue/allocations/quota before
submitting. The login CPU is not a build/performance target. The helper sources
bashrc and activates af3 on compute; the script compiles with explicit clang++
and the installed MPI wrapper's include/link flags. Per-host binaries avoid a
shared build-marker race. Do not edit/replace the source while a job uses it.
Use a new task directory for concurrent campaigns.

After successful completion (replace JOB_ID and destination):

  ~/.local/bin/lx-run status JOB_ID
  ~/.local/bin/lx-run fetch JOB_ID build/comm-network-20261104/JOB_ID
  cp tools/comm-network/{mpi_proxy_probe.cpp,probe-lx.sh} build/comm-network-20261104/JOB_ID/
  python3 tools/comm-network/analyze_probe.py build/comm-network-20261104/JOB_ID \
    --output build/comm-network-20261104/JOB_ID/summary.json

Keep the source snapshot used by that job. The analyzer requires exitcode=0,
two distinct hosts, matching source hashes, matched payload MPI CPU, exactly
210 blocks, five complete rounds per operation/size/mode, and valid payloads.
Each block uses max(8,min(4096,2GiB/bytes)) iterations after three warmup calls.
Metrics are five-block medians of per-call block averages, not per-call p99.
For two ranks, remote_payload_GBps counts N remote bytes per direction per rank;
it excludes local self contributions and does not add the two directions.
Explicit 200Gbps port line rate is 25GB/s before overhead. Rail limits alone
are not a NIC-byte-counter proof; retain actual endpoint/device/linkage logs.

Earlier jobs2022774/2022825 are exploratory/intermediate, with changed payload
CPU or too-short 1B iteration counts. Do not mix their small-message results
into the corrected campaign. The large-message data were still useful for
finding that MPI collective performance can differ from SendRecv performance.

The design and complete limitations are in:
  docs/MultinodeCommunicationDesign-2026-11-04.html
