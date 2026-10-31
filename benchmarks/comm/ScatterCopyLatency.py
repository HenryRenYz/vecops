# SPDX-FileCopyrightText: 2026 HenryRenYz and vecops contributors
# SPDX-License-Identifier: MIT
"""920f-4 protocol-free AllToAll-shaped scatter, eight simultaneous ranks.

Each process privately allocates disjoint destination slices on all eight HBM
nodes. There is no vecops communicator, shared mapping, or transport protocol.
The same vec backend/native pool and 18 MiB/72-task/36-lane shape remain.
Multiprocessing admission barriers and full-output validation are not timed.
--shared-buffer instead borrows the exact input/destination addresses of a
prepared collective from an instrument_protocol.py --snapshot-addresses build.
Its copy timing still bypasses the communication protocol; compare plain copies
and copies immediately preceded by an untimed full collective.
"""
import argparse
import ctypes
import json
import os
import resource
import mmap
import subprocess
import sys
import time
from pathlib import Path

os.environ.setdefault('OMP_WAIT_POLICY', 'PASSIVE')
os.environ.setdefault('GOMP_SPINCOUNT', '0')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')


class FileBarrier:
    def __init__(self, options):
        self.file = open(options.output/'admission', 'r+b')
        self.mapping = mmap.mmap(self.file.fileno(), 4096)
        self.state = (ctypes.c_uint64 * 2).from_buffer(self.mapping)
        self.epoch = 0
        self.library = ctypes.CDLL(options.library)
        self.library.scatter_admit.argtypes = [ctypes.POINTER(ctypes.c_uint64), ctypes.c_uint64, ctypes.c_uint]

    def wait(self, timeout=180):
        self.epoch += 1
        code = self.library.scatter_admit(self.state, self.epoch * 8, timeout)
        if code:
            raise RuntimeError(('admission barrier', code, self.epoch))


def placement_snapshot(options, rank, phase, ranges, library):
    """Read every payload page's actual node; query-only, never migrate/fault."""
    import ctypes.util
    numa = ctypes.CDLL(ctypes.util.find_library('numa'), use_errno=True)
    numa.numa_move_pages.argtypes = [ctypes.c_int, ctypes.c_ulong,
        ctypes.POINTER(ctypes.c_void_p), ctypes.POINTER(ctypes.c_int),
        ctypes.POINTER(ctypes.c_int), ctypes.c_int]
    page = os.sysconf('SC_PAGESIZE')

    def query(address, size):
        addresses = list(range(address//page*page, (address+size-1)//page*page+page, page))
        counts = {}
        for i in range(0, len(addresses), 4096):
            part = addresses[i:i+4096]
            pointers = (ctypes.c_void_p * len(part))(*part)
            status = (ctypes.c_int * len(part))()
            result = numa.numa_move_pages(0, len(part), pointers, None, status, 0)
            if result < 0:
                return dict(query_errno=ctypes.get_errno())
            for value in status:
                counts[str(value)] = counts.get(str(value), 0) + 1
        return counts

    entries = [dict(name=name, address=address, bytes=size,
                    nodes=query(address, size)) for name,address,size in ranges]
    maps = Path('/proc/self/numa_maps').read_text()
    Path(options.output, f'numa-{rank}-{phase}.txt').write_text(maps)
    vmas = [line.split(None, 5) for line in Path('/proc/self/maps').read_text().splitlines()]
    dispatch = getattr(library, '_ZN6vecops4comm6detail12copy_backendEv')
    dispatch.restype = ctypes.c_void_p
    copy_function = ctypes.c_void_p.from_address(dispatch()).value
    functions = []
    for name, address in [('scatter_wrapper', ctypes.cast(library.scatter_copy, ctypes.c_void_p).value),
                          ('vec_copy_bytes', copy_function)]:
        vma = next(v for v in vmas if int(v[0].split('-')[0],16)<=address<int(v[0].split('-')[1],16))
        functions.append(dict(name=name, address=address, mapping=vma,
                              resident_code_page=query(address, 1)))
    Path(options.output, f'placement-{rank}-{phase}.json').write_text(json.dumps(dict(
        rank=rank, cpu_numa=options.socket*8+rank, payloads=entries, functions=functions,
        loader_environment={k:v for k,v in os.environ.items()
                            if k.startswith(('LD_AUDIT','VECOPS_NUMA'))}), indent=2)+'\n')


def worker(rank, options, barrier):
    node = options.socket * 8 + rank
    first = node * 38 + 2
    os.sched_setaffinity(0, set(range(first, first + 36)))
    os.environ.update(VECOPS_THREAD_POOL='native', VECOPS_NUM_THREADS='36',
                      VECOPS_THREAD_POOL_CPU_LIST=f'{first}-{first+35}',
                      VECOPS_THREAD_POOL_WAKE_POLICY='generation',
                      VECOPS_THREAD_POOL_SPIN_COUNT='1024')
    import torch
    import vecops
    torch.set_num_threads(36)
    torch.set_num_interop_threads(1)
    system = vecops.memory.System()
    owners = []
    peer_elements = 1179648  # BF16 2.25 MiB per destination.

    def allocate(elements, memory_node):
        owner = system.allocate(elements * 2, os_numa_id=memory_node,
                                fallback=None, large_pages=False)
        owners.append(owner)
        return torch.frombuffer(owner, dtype=torch.bfloat16, count=elements)

    outputs = [] if options.shared_buffer else [
        allocate(peer_elements, 16 + options.socket * 8 + p).fill_(-1) for p in range(8)]
    sources = {tier: allocate(peer_elements * 8, node + (16 if tier == 'hbm' else 0))
               for tier in ('hbm', 'ddr')}
    for source in sources.values():
        for peer in range(8):
            source[peer * peer_elements:(peer + 1) * peer_elements].fill_(rank * 16 + peer)
    address = vecops._C.native_thread_pool_address(36)
    os.sched_setaffinity(0, {first})
    group = None
    if options.shared_buffer:
        import torch.distributed as td
        import vecops.distributed as vd
        vd.register_backend()
        td.init_process_group('vecops', init_method='file://'+str(options.output/'store'),
                              rank=rank, world_size=8)
        group = td.group.WORLD
        result = allocate(peer_elements * 8, 16 + node).reshape(8, peer_elements)
        outputs = list(result.unbind(0))
        inputs = {tier: list(t.reshape(8, peer_elements).unbind(0))
                  for tier, t in sources.items()}
        handle = vd.backend().prepare('all_to_all', outputs, inputs['hbm'], [], [], 0)
        torch.ops.vecops_comm.run.default(handle, outputs, inputs['hbm'])
        vd.seal()
        diagnostic = ctypes.CDLL(vd.load_communication().__file__)
        diagnostic.comm_diag_snapshot.argtypes = [ctypes.POINTER(ctypes.c_void_p),
            ctypes.POINTER(ctypes.c_void_p), ctypes.POINTER(ctypes.c_size_t), ctypes.c_size_t]
        diagnostic.comm_diag_snapshot.restype = ctypes.c_size_t
    library = ctypes.CDLL(options.library)
    library.scatter_copy.argtypes = [ctypes.c_void_p, ctypes.c_uint, ctypes.c_void_p,
        ctypes.POINTER(ctypes.c_void_p), ctypes.c_size_t, ctypes.c_size_t,
        ctypes.c_size_t, ctypes.POINTER(ctypes.c_double),
        ctypes.POINTER(ctypes.c_double), ctypes.POINTER(ctypes.c_double)]
    library.scatter_copy_pmu.argtypes = library.scatter_copy.argtypes + [ctypes.POINTER(ctypes.c_double)]
    destinations = (ctypes.c_void_p * 8)(*[t.data_ptr() for t in outputs])
    records = []
    # Repeat both tiers in reverse order to expose allocation/time dependence.
    with torch.inference_mode():
        modes = ('copy-only', 'after-collective') if options.shared_buffer else ('copy-only',)
        cases = [(tier, mode) for tier in ('hbm', 'ddr', 'ddr', 'hbm') for mode in modes]
        for tier, mode in cases:
            if options.shared_buffer:
                torch.ops.vecops_comm.run.default(handle, outputs, inputs[tier])
                raw_sources, raw_sizes = (ctypes.c_void_p * 8)(), (ctypes.c_size_t * 8)()
                count = diagnostic.comm_diag_snapshot(raw_sources, destinations, raw_sizes, 8)
                assert count == 8 and all(x == peer_elements * 2 for x in raw_sizes)
                assert all(p == sources[tier].data_ptr()+i*peer_elements*2
                           for i,p in enumerate(raw_sources))
            ranges = [(tier, sources[tier].data_ptr(), peer_elements*8*2)] + [
                (f'destination-{peer}', address, peer_elements*2)
                for peer,address in enumerate(destinations)]
            if options.placement:
                placement_snapshot(options, rank, f'{len(records)}-before', ranges, library)
                barrier.wait(timeout=180)
            samples = []
            for iteration in range(options.repeats + 10):
                barrier.wait(timeout=180)
                if mode == 'after-collective':
                    torch.ops.vecops_comm.run.default(handle, outputs, inputs[tier])
                wall = ctypes.c_double()
                lane_wall, lane_cpu = (ctypes.c_double * 36)(), (ctypes.c_double * 36)()
                lane_pmu = (ctypes.c_double * 288)()
                before = resource.getrusage(resource.RUSAGE_SELF)
                begin_ns = time.monotonic_ns()
                args = (address, 36, sources[tier].data_ptr(), destinations, 8,
                        peer_elements * 2, 256 << 10, ctypes.byref(wall), lane_wall, lane_cpu)
                status = library.scatter_copy_pmu(*args, lane_pmu) if options.pmu else library.scatter_copy(*args)
                end_ns = time.monotonic_ns()
                after = resource.getrusage(resource.RUSAGE_SELF)
                if status:
                    raise RuntimeError(('scatter_copy', rank, status))
                if iteration >= 10:
                    critical = max(range(36), key=lambda lane: lane_wall[lane])
                    sample = dict(wall_ns=wall.value, begin_ns=begin_ns, end_ns=end_ns,
                        critical_lane=critical, critical_wall_ns=lane_wall[critical],
                        critical_cpu_ns=lane_cpu[critical],
                        minor_faults=after.ru_minflt-before.ru_minflt,
                        major_faults=after.ru_majflt-before.ru_majflt,
                        lane_wall_ns=list(lane_wall), lane_cpu_ns=list(lane_cpu))
                    if options.pmu:
                        sample['critical_pmu'] = list(lane_pmu)[8*critical:8*critical+4]
                        sample['critical_rusage'] = list(lane_pmu)[8*critical+4:8*critical+8]
                    samples.append(sample)
            for peer in range(8):
                output = torch.frombuffer((ctypes.c_uint16 * peer_elements).from_address(
                    destinations[peer]), dtype=torch.bfloat16)
                assert bool((output == rank * 16 + peer).all()), (rank, tier, peer)
            records.append(dict(tier=tier, mode=mode, exact=True, samples=samples))
            if options.placement:
                placement_snapshot(options, rank, f'{len(records)-1}-after', ranges, library)
                barrier.wait(timeout=180)
    Path(options.output, f'rank-{rank}.json').write_text(json.dumps(dict(
        rank=rank, socket=options.socket, numa=node,
        shared_buffer=options.shared_buffer, records=records), indent=2)+'\n')
    if group is not None:
        barrier.wait(timeout=180)
        td.destroy_process_group()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--library', required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--socket', type=int, choices=(0, 1), required=True)
    parser.add_argument('--repeats', type=int, default=100)
    parser.add_argument('--shared-buffer', action='store_true')
    parser.add_argument('--placement', action='store_true', help='query actual data/code pages outside clocks')
    parser.add_argument('--exec-workers', action='store_true', help='bind before exec, required for LD_AUDIT')
    parser.add_argument('--pmu', action='store_true', help='diagnostic user cycles/instructions; reads outside callback clocks')
    parser.add_argument('--rank', type=int, choices=range(8), help=argparse.SUPPRESS)
    options = parser.parse_args()
    if options.repeats < 1:
        parser.error('positive repeats required')
    if options.rank is not None:
        worker(options.rank, options, FileBarrier(options))
        sys.exit(0)
    options.output.mkdir(parents=True, exist_ok=False)
    if options.exec_workers:
        (options.output/'admission').write_bytes(bytes(4096))
        children = []
        for rank in range(8):
            first = (options.socket*8+rank)*38+2
            command = ['taskset','-c',f'{first}-{first+35}',sys.executable,
                       str(Path(__file__).resolve()),*sys.argv[1:],'--rank',str(rank)]
            children.append(subprocess.Popen(command))
        codes = [child.wait() for child in children]
        if any(codes):
            raise RuntimeError(('worker exit codes', codes))
        sys.exit(0)
    import torch.multiprocessing as mp
    barrier = mp.get_context('spawn').Barrier(8)
    mp.spawn(worker, args=(options, barrier), nprocs=8, join=True)
