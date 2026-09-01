# BiSheng clang 19.1.7 SVE SimplifyCFG crash — minimal reproducer

`clang++ -march=armv8-a+sve -O3` (BiSheng Enterprise 5.1.0.2, LLVM 19.1.7
derivative) aborts with `fatal error: error in backend: Invalid size request
on a scalable vector` whenever a function contains, on two branches of a
conditional, a store of a scalable-vector *constant* (e.g. `zeros`) to
different pointers with a common successor.

Reproduce (any of):

    opt -passes='default<O3>' simplifycfg-sink-scalable-store.ll -o /dev/null
    opt -passes='simplifycfg<bonus-inst-threshold=1;keep-loops;speculate-blocks;simplify-cond-branch;sink-common-insts>' repro.bc

Analysis (bisection via IR variants, 2026-08-30):

- The fatal is raised from `llvm::TypeSize::operator unsigned long()` inside
  the late SimplifyCFG run that carries `sink-common-insts`
  (`SinkCommonCodeFromPredecessors` path): sinking the common stores queries
  alias information, which sizes the scalable-vector store and hits the
  missing `isScalable` guard that upstream LLVM uses for this failure class
  (cf. llvm/llvm-project#88576 fixed by #88590 for the SelectionDAG sibling).
- Necessary-shape probes: one store per branch suffices; `poison` values do
  not trigger, `zeroinitializer` does; `noalias`/`align 16`/compile barriers
  between stores do not work around it; the `-mllvm` flag family
  (`simplifycfg-merge-cond-stores`, `hoist-cond-stores`, `phi-node-folding`,
  jump-threading, speculation-depth) does not disable it — it is a pass
  parameter, not a cl::opt.
- Triggered in vecops by `tests/kernel/Tile2DGemmTest.cpp`
  (`tile2d_gemm_scalable_probe`: K==0 exit vs loop exit each storing zeroed
  SVE accumulators); the same TU crashes on the pre-refactor HEAD, and open
  clang 12 compiles it fine.

Please report to BiSheng with this directory attached.
