<!-- SPDX-FileCopyrightText: 2026 HenryRenYz and vecops contributors -->
<!-- SPDX-License-Identifier: MIT -->
# NUMA-group dynamic-library loader

`libvecops_numa_loader.so` is an optional Linux/glibc `LD_AUDIT` module. Members
of a configured NUMA group share an independent file-cache inode for each DSO;
different groups get separate files. It does not change page policies, force
physical placement, run a separate prefault pass, modify executable mappings, or enter
kernel/replay dispatch. It depends only on glibc (libc and the runtime loader), not Torch/Python/hwloc/OpenMP.

## Build and install

```bash
cmake -S . -B build-numa-loader \
  -DCMAKE_C_COMPILER="$(command -v gcc)" \
  -DCMAKE_CXX_COMPILER="$(command -v g++)" \
  -DVECOPS_BUILD_NUMA_LOADER=ON -DCMAKE_INSTALL_LIBDIR=lib -DCMAKE_BUILD_TYPE=Release
cmake --build build-numa-loader --target vecops_numa_loader -j2
cmake --install build-numa-loader --prefix ~/.local --component numa-loader
```

Use explicit `clang`/`clang++` paths on ARM. Other SDK options can remain enabled;
to configure just portable dependencies, additionally disable
`VECOPS_BUILD_COMPILER`, `VECOPS_ENABLE_OPENMP`, and `VECOPS_BUILD_MEMORY`.
The default loader option is OFF. Python wheel builds with the option enabled
also install the DSO into `vecops/_sdk/lib` (component `python`).

## Environment, before exec

| Variable | Meaning |
|---|---|
| `LD_BIND_NOW` | Set `1` on old AArch64 audit ABI v1 (including 920f-4 glibc 2.34); eager symbol binding bypasses its broken PLT trampoline |
| `LD_AUDIT` | Absolute loader DSO path, or glibc's colon-separated audit list |
| `VECOPS_NUMA_PLACEMENT` | Semicolon-separated node sets, e.g. `0-3;4-7;8-11;12-15` |
| `VECOPS_NUMA_GROUP` | Optional exact node-set selection from placement; does not bind CPUs |
| `VECOPS_NUMA_INCLUDE`, `VECOPS_NUMA_EXCLUDE` | POSIX ERE search on original canonical basename |
| `VECOPS_NUMA_INCLUDE_PATH`, `VECOPS_NUMA_EXCLUDE_PATH` | POSIX ERE search on original resolved absolute path |
| `VECOPS_NUMA_CACHE_DIR` | Absolute private cache root; otherwise absolute XDG_CACHE_HOME + `vecops/numa-loader`, then HOME + `.cache/vecops/numa-loader` |
| `VECOPS_NUMA_VERBOSE` | `1` logs selection, source/copy association, hash, hit/publish, file/text bytes and preparation milliseconds; default warnings only |

Without an explicit group, **all CPUs in the initial affinity** must belong to
one valid group. A later Python `sched_setaffinity` cannot change this decision;
bind before exec, or exec again after establishing rank binding. Fork alone
inherits existing objects. Changes to environment do not retarget loaded code.
Empty/invalid/uncovered selection uses original files. Node IDs are OS NUMA IDs;
online memory-only nodes are invalid CPU group members. Entire malformed groups
are skipped, exact duplicates merged, and all partially overlapping groups
invalidated, independent of order. Parsing is bounded (4096 node IDs, 128 unique
groups, 262144 CPU IDs); out-of-range configuration falls back.

Exclusion wins. If any include is set, at least one include must match. Unset
includes select all eligible DSOs; an empty include matches nothing. An empty
exclude excludes nothing. Any invalid regex disables redirection with one warning.
Bootstrap libc/ld-linux and the audit module are always left at their originals.

Example for verified standalone kernels and Torch's compute DSO:

```bash
numactl --cpunodebind=4 env \
  LD_AUDIT="$HOME/.local/lib/libvecops_numa_loader.so" \
  LD_BIND_NOW=1 \
  VECOPS_NUMA_PLACEMENT='0-3;4-7;8-11;12-15' \
  VECOPS_NUMA_INCLUDE='^(kernel\.so|executor\.so|libvecops.*\.so|libtorch_cpu\.so)$' \
  python inference.py
```

AF3's direct profile uses a broader selection and excludes interpreter,
compiler, and CUDA plugin runtimes. Its worker binds then execs once before
creating console threads or importing Torch. Preparation loads kernels and
executors; formal inference only replays them. No `prepare_code_locality` call
is needed; that older manual COW API has been removed.

Old AArch64 glibc audit ABI v1 does not preserve the x8 indirect-result register
on lazy PLT calls, **even without function audit callbacks**. This is upstream
[glibc BZ#26643](https://sourceware.org/pipermail/glibc-cvs/2022q1/076541.html).
For these builds the loader refuses activation unless `LD_BIND_NOW` is nonempty;
set it to `1` before exec. A no-redirection/invalid configuration returns audit
version zero, disabling the auditor itself. AF3's 920f profile sets eager binding.
The regression suite exercises a real PLT call returning a 24-byte struct with
and without eager binding. Eager binding adds preparation cost, not graph work.

## Cache and failure boundaries

Layout is `ROOT/v1/HOST/GROUP/SOURCE-VERSION/root/absolute/source/path`.
The host key hashes machine-id and hostname; the normalized group is stable
across ordering/duplicates. A source version includes device/inode, size,
mtime/ctime with nanoseconds, and full SHA-256. Copy and source hashes must agree;
the source must remain unchanged during copying. Different versions publish
new files, preserving existing mapped versions. A stable association is retained
in the audit namespace for loaded object paths and their original directories.

A 30-second bounded `flock` serializes publication for each version. Copying is
performed synchronously by the requesting process, using its affinity/policy;
there is no copy daemon or fork helper. Same-directory `mkostemp`, full digest
verification, fsync and atomic rename prevent loading partial copies. Cache hits
are also byte-verified before initial use. Private directories are checked for
ownership/mode and symlink substitution; copied files are independent, single-link
read/execute files. The root must allow executable mappings; no `/tmp` default
or shared cache is used. Failures fall back to the loader's original candidate,
never `NULL`. Warnings are deduplicated per reason class.

Integrity hashing and copying necessarily read/write complete file contents,
including non-executable sections; these pages can remain in the reclaimable
file cache even if they never become resident executable mappings.
Copies incur file-cache/storage costs. File bytes, executable segment bytes,
RSS and shared PSS are different measures. Existing cache pages are not moved
on reuse. There is no automatic eviction; remove inactive cache versions only
when no process needs to reopen them. A bad LD_AUDIT deployment path itself is
outside the module's recoverable-error boundary.

## Path compatibility

The loader does not invent SONAME search rules. Only actual path candidates
provided by glibc are redirected. For a copied referring object, expanded
`$ORIGIN` search candidates are translated to the original absolute tree before
filtering/copying; this also works when the dependency itself is excluded.
Regression coverage includes ordinary absolute paths, symlinks to the same
canonical standalone image, and repeated/multithreaded dlopen identity. This
is qualification of those patterns, not arbitrary path virtualization. ELF executables, non-ELF files and relative slash-containing DT_NEEDED
patterns are left alone. libc/loader bootstrap dependencies are excluded.

`dladdr` may report a copy path or retain glibc's original/search-candidate
spelling; it is not a stable promise of the original pathname. A library that finds
adjacent data/plugins using its own path needs an exclusion rule: arbitrary
resource accesses cannot be intercepted by LD_AUDIT. This release does not
claim transparent resource relocation. The regression fixture verifies such
libraries keep their original adjacent resource when excluded; `$ORIGIN`
DT_NEEDED dependencies are separately covered. Mixing different hard-link
source names, or symlink paths that change the intended origin/resource tree,
is outside this release's qualified set: exclude every alias of such an image
before exec. Extra packaging hard links that are never used as alternative
runtime names do not by themselves prevent copying. AF3 acceptance inventories
check that each loaded source image uses one canonical runtime pathname.
Future path-sensitive libraries
must be qualified or excluded before enabling a broad include policy.

Torch GOMP verification accounts for the mapped cache path without weakening
its identity guard: prove the recorded source path and SHA-256, retain the
actual mapped object's NOLOAD handle, then require every GOMP/OMP symbol to
resolve to that one object, using the VMA containing each function address
when glibc retains an original `dladdr` name. Native pool ownership checks continue to pin and
verify the real provider DSO.

## Validation

```bash
python tools/numa-loader/test_loader.py -v
# Select an already-built module instead of compiling with gcc:
VECOPS_TEST_NUMA_LOADER=/absolute/libvecops_numa_loader.so \
  python tools/numa-loader/test_loader.py -v
```

The stdlib-only suite covers group/regex failures, concurrent first publication,
source updates, independent inodes/content, cache integrity and permissions,
ORIGIN dependencies, adjacent-resource exclusion and threaded/repeated dlopen.
Multi-node grouping tests need two CPU NUMA domains. Fixtures use `~/tmp`.
Model-level acceptance must additionally compare matched no-profiler timings,
complete numerical outputs, physical page samples/RSS/PSS and communications.
File isolation alone cannot promise the old per-rank COW placement or speed.
