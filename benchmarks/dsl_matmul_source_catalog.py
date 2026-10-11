# SPDX-FileCopyrightText: 2026 HenryRenYz and vecops contributors
# SPDX-License-Identifier: MIT
"""Execute every concrete ARM source registration, with auditable coverage.

The six registration cohorts are distinct from shape/vocabulary indexes. Every
source row is expanded across its declared metadata and packing modes; source
labels are retained even when the same executable is shared. Offline packing is
validated outside timing, online packing is inside both native repeat loops.
The C++ reference uses the actual required family/tile policy and the public
rank-three Matmul rather than a per-batch rewrite. The DSL rank-three wrapper
records serial versus shared-B folding; it is not a native T.grid claim.
"raw" describes the supplied input layout, not a ban on the public C++
operator's internal automatic packing. In particular, unbounded Dynamic SME
inputs may be packed by the C++ reference inside the timed call. Native T.dot
prepares instruction-required layouts; T.library.matmul additionally selects
cache phases and optional dense caches. Raw dot versus public Matmul ratios
therefore need separate algorithm/path interpretation.

Use --list to audit, --cohort to schedule whole cohorts, or --chunk-size and
--chunk-index for bounded build graphs. No default FLOP/shape cap applies.
"""

import argparse
import csv
import hashlib
import json
import os
import sys
from collections import Counter
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np
import torch
import vecops
import vecops.language as T
from dsl_benchmark_utils import add_toolchain_arguments, default_cpu, is_arm, packed_size, packing_geometry
from dsl_matmul_catalog import ROOT, load_cases
from dsl_matmul_fusion_catalog import (
  INPUTS,
  OUTPUTS,
  Scenario,
  arguments,
  metadata,
  validate_pipeline_contract,
)
from dsl_timing import isolate_compilation, reserve_cpu
from dsl_transpose import aligned_empty, measure

COHORTS = {
  "base": "MatmulBaseExpanded",
  "packed": "MatmulPackedExpanded",
  "family": "MatmulFamilyExpanded",
  "policy": "MatmulPolicyExpanded",
  "fusion": "MatmulFusionExpanded",
  "orientation": "MatmulOrientation",
}
PACKING = ("raw", "packeda", "packedb", "packedab", "onlinepackeda", "onlinepackedb", "onlinepackedab")
FAMILIES = ("Automatic", "SmallVector", "PackedDot", "RuntimeQuantInt8")
POLICIES = ("Automatic", "ExactCover", "FourRegions")


def normalize_case(case):
  """Fill only the plain registration defaults; never guess a family selector."""
  runtime_quant = case.catalog == "MatmulFamilyExpanded" and "register_runtime_quant_pair<" in case.declaration
  # Identity is the source registrar's name for an already-native input;
  # the shared runner's Convert pipeline is also an identity for that dtype.
  ia = None if case.input_pipeline_a == "Identity" else case.input_pipeline_a
  ib = None if case.input_pipeline_b == "Identity" else case.input_pipeline_b
  return replace(
    case,
    memory_a=case.memory_a or ("fp32" if runtime_quant else case.native_a),
    memory_b=case.memory_b or case.native_b,
    memory_c=case.memory_c or ("fp32" if runtime_quant else case.accumulator),
    memory_c_input=case.memory_c_input or case.memory_c or case.accumulator,
    input_pipeline_a=ia or ("RuntimePerRowAsymmetricQuantize" if runtime_quant else "Convert"),
    input_pipeline_b=ib or "Convert",
    output_pipeline=case.output_pipeline or ("RuntimePerRowColumnDequantize" if runtime_quant else "Convert"),
  )


def reference_selectors(case):
  family, policy = "Automatic", "Automatic"
  config = case.configuration or ""
  if case.catalog == "MatmulFamilyExpanded":
    if "SmallVector" in config or config == "Small":
      family = "SmallVector"
    elif "PackedDot" in config:
      family = "PackedDot"
    elif "RuntimeQuant" in config:
      family = "RuntimeQuantInt8"
    elif config not in ("Automatic", "family_selection::Automatic", "historical_auto"):
      raise ValueError(f"unresolved C++ family selector: {config!r}; {case.declaration}")
  elif case.catalog == "MatmulPolicyExpanded":
    policy = config.rsplit("::", 1)[-1]
    if policy not in POLICIES:
      raise ValueError(f"unresolved C++ policy selector: {config!r}")
  return family, policy


@dataclass(frozen=True)
class SourceScenario(Scenario):
  packing: str = "raw"

  @property
  def pack_a(self):
    return self.packing.endswith(("packeda", "packedab"))

  @property
  def pack_b(self):
    return self.packing.endswith(("packedb", "packedab"))

  @property
  def online(self):
    return self.packing.startswith("online")

  @property
  def reference(self):
    return reference_selectors(self.case)

  @property
  def key(self):
    return f"{super().key}/{self.packing}/{self.reference}/SVL={self.case.svl_bytes}"

  @property
  def compilation_key(self):
    return super().compilation_key, self.packing, self.reference

  @property
  def execution_id(self):
    return hashlib.sha256(self.key.encode()).hexdigest()[:24]

  def validate(self):
    Scenario(replace(self.case, input_modes=("raw",)), self.mode, self.level, self.batch_strategy).validate()
    if self.packing not in PACKING or self.packing not in self.case.input_modes:
      raise ValueError(f"packing mode does not belong to this source registration: {self.packing}")
    if self.batched and (self.pack_a or self.pack_b and not self.shared_b):
      raise ValueError("rank-three packing is registered for shared B only")
    _ = self.reference


def source_templates(scenario, *, blocks=(None, None, None)):
  """Generate the exact registration; optional blocks are diagnostic DSL tuning.

  The C++ reference and offline packers are never changed by this override.
  Omitted blocks preserve the ordinary catalog's automatic cache policy.
  """
  if (
    not isinstance(blocks, tuple)
    or len(blocks) != 3
    or any(value is not None and (type(value) is not int or value <= 0) for value in blocks)
  ):
    raise ValueError("blocks must be a tuple of three positive integers or None")
  if scenario.level != "matmul" and any(value is not None for value in blocks):
    raise ValueError("cache blocks apply to library matmul, not dot")
  BM, BN, BK = blocks
  case = scenario.case
  M, N, K = (
    (case.m, case.n, case.k)
    if scenario.mode == "Const"
    else tuple(T.Dynamic(name=name, lower=1) for name in ("M", "N", "K"))
  )
  Batch = scenario.batch if scenario.mode == "Const" else T.Dynamic(name="Batch", lower=1)
  PSA, PSB = T.Dynamic(name="PSA", lower=1), T.Dynamic(name="PSB", lower=1)
  ad, bd, cd, dd, na, nb, acc = scenario.types
  at, bt, ct = scenario.orientation
  astride, bstride, cstride = (1, M) if at else (K, 1), (1, K) if bt else (N, 1), (1, M) if ct else (N, 1)
  DM = 1 if scenario.broadcast_prior else M
  if scenario.batched:
    AT = T.Tensor((Batch, M, K), (0 if scenario.shared_a else M * K, *astride), ad)
    BT = T.Tensor((Batch, K, N), (0 if scenario.shared_b else K * N, *bstride), bd)
    CT = T.Tensor((Batch, M, N), (M * N, *cstride), cd, access="output")
    DT = T.Tensor((Batch, DM, N), (0 if scenario.shared_prior else DM * N, N, 1), dd)
    RT = T.Tensor((Batch, M, 1), (0 if scenario.shared_a else M, 1, 1), "float32")
  else:
    AT, BT, CT = (
      T.Tensor((M, K), astride, ad),
      T.Tensor((K, N), bstride, bd),
      T.Tensor((M, N), cstride, cd, access="output"),
    )
    DT, RT = T.Tensor((DM, N), dtype=dd), T.Tensor((M, 1), dtype="float32")
  APT, BPT = T.Tensor((PSA,), dtype=na, access="inout"), T.Tensor((PSB,), dtype=nb, access="inout")
  pa, pb, online = scenario.pack_a, scenario.pack_b, scenario.online
  batched, flatten, flat_prior = scenario.batched, scenario.flatten_batch, scenario.shared_prior
  level, swap, prior = scenario.level, T.Const(scenario.swap), scenario.has_prior
  format_ = packing_geometry(na)[0]
  ia, ib, out = case.input_pipeline_a, case.input_pipeline_b, case.output_pipeline
  qa, qb = ia in INPUTS[1:4], ib in INPUTS[1:4]
  float_epilogue = out in (
    "Dequantize",
    "ReluRequantize",
    "RequantizeU8Zp7",
    "AsymmetricDequantize",
    "BiasPerColumnDequantize",
    "BiasPerColumnRequantizeU8",
    "RuntimePerRowColumnDequantize",
  )
  relu_epilogue = out in ("Relu", "BiasRelu", "AccumulateReluScale", "ReluRequantize")
  silu_epilogue = out in ("Silu", "BiasSilu")
  scale_epilogue = out in ("Scale", "AccumulateReluScale")
  clamp_epilogue = out in ("Clamp", "BiasClamp")
  dequant_epilogue = out in ("Dequantize", "ReluRequantize")
  column_epilogue = out in ("BiasPerColumnDequantize", "BiasPerColumnRequantizeU8")
  azp = 3 if ia != "Quantize4" and na == "uint8" else 0
  bzp = 5 if ib == "AsymmetricQuantize4Zp3Zp5" else 0

  @T.func
  def input_a(A, RQ, Z):
    if ia == "RuntimePerRowAsymmetricQuantize":
      return T.cast(T.load(A, dtype="float32") * T.load(RQ) + T.load(Z, dtype="float32"), na)
    elif qa:
      return T.cast(T.load(A, dtype="float32") * 4 + azp, na)
    return T.load(A, dtype=na)

  @T.func
  def input_b(B):
    if qb:
      return T.cast(T.load(B, dtype="float32") * 4 + bzp, nb)
    return T.load(B, dtype=nb)

  @T.func
  def packing(A, B, RQ, Z, AP, BP):
    if pa:
      T.pack(input_a(A, RQ, Z), AP, side="A", format=format_, dtype=na)
    if pb:
      if batched:
        T.pack(input_b(B[0]), BP, side="B", format=format_, dtype=nb)
      else:
        T.pack(input_b(B), BP, side="B", format=format_, dtype=nb)

  @T.func
  def compute(A, B, C, D, S, SC, SQ, RQ, RD, W, Z, AP, BP):
    a = T.packed(AP, (A.shape[0], A.shape[1]), side="A", format=format_) if pa else input_a(A, RQ, Z)
    b = T.packed(BP, (B.shape[0], B.shape[1]), side="B", format=format_) if pb else input_b(B)
    initial = T.load(D, dtype=acc) if prior else None
    if level == "dot":
      result = T.dot((a, b), C, acc=initial, acc_dtype=acc)
    else:
      result = T.library.matmul((a, b), C, acc=initial, acc_dtype=acc, swap=swap, BM=BM, BN=BN, BK=BK)
    if float_epilogue:
      result = T.cast(result, "float32")
    if relu_epilogue:
      result = T.max(result, 0)
    if out == "Sigmoid":
      result = 1 / (1 + T.exp(-result))
    elif silu_epilogue:
      result = result * (1 / (1 + T.exp(-result)))
    elif scale_epilogue:
      result = result * 0.5
    elif out == "DynamicScale":
      result = result * T.load(S)
    elif out == "PerColumnScale":
      result = result * T.load(SC)
    elif clamp_epilogue:
      result = T.min(T.max(result, -0.25), 0.25)
    elif dequant_epilogue:
      result = result * 0.125
    elif out == "RequantizeU8Zp7":
      result = result * 0.125 + 7
    elif out == "AsymmetricDequantize":
      result = result * 0.0625
    elif column_epilogue:
      result = result * T.load(SQ)
      if out == "BiasPerColumnRequantizeU8":
        result = result + T.load(Z, dtype="float32")
    elif out == "RuntimePerRowColumnDequantize":
      result = result * T.load(RD) * T.load(W)
    T.store(C, result)

  def template(only_pack):
    @T.template
    def dsl(
      A: AT,
      B: BT,
      C: CT,
      D: DT,
      S: T.Tensor((1, 1), dtype=acc),
      SC: T.Tensor((1, N), dtype=acc),
      SQ: T.Tensor((1, N), dtype="float32"),
      RQ: RT,
      RD: RT,
      W: T.Tensor((1, N), dtype="float32"),
      Z: T.Tensor((1, 1), dtype="int32"),
      AP: APT,
      BP: BPT,
      repeat: int,
    ):
      for _iteration in T.serial(repeat, unroll=1):
        if only_pack or online:
          packing(A, B, RQ, Z, AP, BP)
        if not only_pack:
          if flatten:
            a = T.reshape(A, (A.shape[0] * A.shape[1], A.shape[2]))
            c = T.reshape(C, (C.shape[0] * C.shape[1], C.shape[2]))
            rq = T.reshape(RQ, (RQ.shape[0] * RQ.shape[1], 1))
            rd = T.reshape(RD, (RD.shape[0] * RD.shape[1], 1))
            d = D[0] if flat_prior else T.reshape(D, (D.shape[0] * D.shape[1], D.shape[2]))
            compute(a, B[0], c, d, S, SC, SQ, rq, rd, W, Z, AP, BP)
          elif batched:
            for batch in T.serial(Batch, unroll=1):
              compute(A[batch], B[batch], C[batch], D[batch], S, SC, SQ, RQ[batch], RD[batch], W, Z, AP, BP)
          else:
            compute(A, B, C, D, S, SC, SQ, RQ, RD, W, Z, AP, BP)
        T.compiler_fence()

    NativeA, NativeB = T.DType(getattr(vecops, na)), T.DType(getattr(vecops, nb))
    InputPipelineA, InputPipelineB = T.Const(INPUTS.index(ia)), T.Const(INPUTS.index(ib))
    OutputPipeline = T.Const(OUTPUTS.index(out))
    HasPrior, BroadcastPrior, SwapAB = T.Const(prior), T.Const(scenario.broadcast_prior), T.Const(scenario.swap)
    PackA, PackB, OnlinePack, OnlyPack, SharedB = tuple(
      T.Const(value) for value in (pa, pb, online, only_pack, scenario.shared_b)
    )
    family, policy = scenario.reference
    ReferenceFamily, ReferencePolicy = T.Const(FAMILIES.index(family)), T.Const(POLICIES.index(policy))

    @T.template(source=ROOT / "benchmarks/dsl_matmul_source_reference.cpp")
    def cpp(
      A: AT,
      B: BT,
      C: CT,
      D: DT,
      S: T.Tensor((1, 1), dtype=acc),
      SC: T.Tensor((1, N), dtype=acc),
      SQ: T.Tensor((1, N), dtype="float32"),
      RQ: RT,
      RD: RT,
      W: T.Tensor((1, N), dtype="float32"),
      Z: T.Tensor((1, 1), dtype="int32"),
      AP: APT,
      BP: BPT,
      repeat: int,
    ):
      NativeA, NativeB, InputPipelineA, InputPipelineB, OutputPipeline, HasPrior, BroadcastPrior, SwapAB  # noqa: B018
      PackA, PackB, OnlinePack, OnlyPack, SharedB, ReferenceFamily, ReferencePolicy  # noqa: B018

    return dsl, cpp

  return template(False), template(True) if pa or pb else None


def source_metadata(scenario):
  _, _, _, _, na, nb, _ = scenario.types
  case = scenario.case
  ap = packed_size(case.m, case.k, na) if scenario.pack_a else 1
  bp = packed_size(case.n, case.k, nb) if scenario.pack_b else 1
  return (
    *metadata(scenario),
    vecops.TensorMeta((ap,), dtype=getattr(vecops, na)),
    vecops.TensorMeta((bp,), dtype=getattr(vecops, nb)),
  )


def source_arguments(scenario):
  values, expected = arguments(scenario)
  _, _, _, _, na, nb, _ = scenario.types
  case = scenario.case
  ap = aligned_empty((packed_size(case.m, case.k, na) if scenario.pack_a else 1,), na)
  bp = aligned_empty((packed_size(case.n, case.k, nb) if scenario.pack_b else 1,), nb)
  return (*values, ap, bp), expected


def scenarios(cases, *, modes=("Const", "Dynamic"), levels=("matmul",), batch_strategy="auto"):
  """Preserve every concrete registration; unresolved schemas are explicit errors."""
  for case in cases:
    if case.catalog not in COHORTS.values():
      continue
    normalized = normalize_case(case)
    for mode in normalized.metadata:
      if mode not in modes:
        continue
      for packing in normalized.input_modes:
        for level in levels:
          scenario = SourceScenario(normalized, mode, level, batch_strategy, packing)
          scenario.validate()
          yield scenario


def audit_document(cases, executions):
  source_hashes = {
    source: hashlib.sha256((ROOT / source).read_bytes()).hexdigest() for source in {case.source for case in cases}
  }
  return {
    "source_hashes": source_hashes,
    "runner_hashes": {
      name: hashlib.sha256((ROOT / "benchmarks" / name).read_bytes()).hexdigest()
      for name in (
        "dsl_matmul_source_catalog.py",
        "dsl_matmul_source_reference.cpp",
        "dsl_matmul_fusion_catalog.py",
        "dsl_matmul_fusion_reference.cpp",
        "dsl_matmul_atom.h",
        "dsl_matmul_catalog.py",
      )
    },
    "source_rows": dict(Counter(case.catalog for case in cases)),
    "execution_rows": dict(Counter(s.case.catalog for s in executions)),
    "registration_rows": sum(case.catalog in COHORTS.values() for case in cases),
    "executions": [
      dict(
        s.case.as_dict(),
        execution_id=s.execution_id,
        meta=s.mode,
        level=s.level,
        packing=s.packing,
        reference_family=s.reference[0],
        reference_policy=s.reference[1],
        batch_strategy="flatten" if s.flatten_batch else "serial",
      )
      for s in executions
    ],
    "declaration_indexes": [case.as_dict() for case in cases if case.catalog not in COHORTS.values()],
  }


def result_metadata(scenario):
  return {
    "execution_id": scenario.execution_id,
    "catalog": scenario.case.catalog,
    "case": scenario.case.name,
    "m": scenario.case.m,
    "n": scenario.case.n,
    "k": scenario.case.k,
    "batch": scenario.batch,
    "broadcast": scenario.case.broadcast,
    "meta": scenario.mode,
    "level": scenario.level,
    "packing": scenario.packing,
    "reference_family": scenario.reference[0],
    "reference_policy": scenario.reference[1],
    "batch_strategy": "flatten" if scenario.flatten_batch else "serial",
  }


def verify_manifests(paths, expected, identity):
  """Check exact execution IDs, not row totals or an approximate shape set."""
  expected = {scenario.execution_id for scenario in expected}
  if not expected:
    raise ValueError("cannot verify an empty execution selection; zero executions is not a pass")
  records, duplicates, stale = {}, [], []
  for path in paths:
    document = json.loads(Path(path).read_text())
    if any(document.get(key) != identity[key] for key in ("source_hashes", "runner_hashes")) or document.get(
      "pass_options", {}
    ) != identity.get("pass_options", {}):
      stale.append(str(path))
    selected = set(document["selected_execution_ids"])
    for key, record in document["results"].items():
      if key not in selected or record.get("execution_id", key) != key:
        raise ValueError(f"manifest result does not belong to its selected execution: {path}: {key}")
      if key in records:
        duplicates.append(key)
      records[key] = record
  missing = sorted(expected - records.keys())
  unexpected = sorted(records.keys() - expected)
  failed = sorted(key for key in expected & records.keys() if records[key].get("status") != "passed")
  result = {
    "expected": len(expected),
    "recorded": len(records),
    "passed": len(expected) - len(missing) - len(failed),
    "missing": missing,
    "unexpected": unexpected,
    "failed": failed,
    "duplicates": duplicates,
    "stale": stale,
  }
  result["complete"] = not any((missing, unexpected, failed, duplicates, stale))
  return result


def select_scenarios(executions, *, cohorts, modes, ranks, native=None, shapes=None, packing=None):
  """Reject unmatched requested filters, including partial matches, before audit.

  A typo must not turn an intended performance run into a successful empty list
  or silently discard one of several explicitly requested shapes/dtypes/modes.
  Default cohort/metadata/rank axes may legitimately be absent from a subset.
  """
  selected = [
    s
    for s in executions
    if s.case.catalog in {COHORTS[name] for name in cohorts}
    and s.mode in modes
    and (3 if s.batched else 2) in ranks
    and (not native or s.case.native_a in native)
    and (not shapes or f"{s.case.m}x{s.case.n}x{s.case.k}" in shapes)
    and (not packing or s.packing in packing)
  ]
  for label, requested, actual in (
    ("native dtype", native, {s.case.native_a for s in selected}),
    ("shape", shapes, {f"{s.case.m}x{s.case.n}x{s.case.k}" for s in selected}),
    ("packing", packing, {s.packing for s in selected}),
  ):
    missing = set(requested or ()) - actual
    if missing:
      raise ValueError(f"requested {label} filters have no matching source executions: {', '.join(sorted(missing))}")
  if not selected:
    raise ValueError("no source executions selected; zero executions is not a pass")
  return selected


def validate_kernel_pair(dsl, cpp, output, expected, *, tolerance):
  """Retain the existing numerical contract and identify the failing oracle.

  A reference failure stops before invoking DSL: reporting that result as a
  DSL mismatch would be misleading. Successful integer outputs still require
  exact DSL/reference equality; floating-point tolerances are unchanged.
  """
  cpp()
  reference = output.copy()
  try:
    np.testing.assert_allclose(reference, expected, rtol=tolerance, atol=tolerance)
  except AssertionError as error:
    raise AssertionError(f"C++ reference vs independent oracle:\n{error}") from error
  dsl()
  try:
    if output.dtype.kind in "iu":
      np.testing.assert_array_equal(output, reference)
    else:
      np.testing.assert_allclose(output, reference, rtol=tolerance, atol=tolerance)
  except AssertionError as error:
    raise AssertionError(f"DSL vs C++ reference:\n{error}") from error


def main(argv=None):
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--cohort", nargs="+", choices=tuple(COHORTS), default=list(COHORTS))
  parser.add_argument("--mode", nargs="+", choices=("Const", "Dynamic"), default=["Const", "Dynamic"])
  parser.add_argument("--level", nargs="+", choices=("dot", "matmul"), default=["matmul"])
  parser.add_argument("--native", nargs="+")
  parser.add_argument("--pipeline", nargs="+", choices=OUTPUTS)
  parser.add_argument("--shape", nargs="+")
  parser.add_argument("--packing", nargs="+", choices=PACKING)
  parser.add_argument("--batch-strategy", choices=("auto", "serial"), default="auto")
  parser.add_argument("--rank", nargs="+", choices=(2, 3), type=int, default=[2, 3])
  parser.add_argument("--chunk-size", type=int, default=32)
  parser.add_argument(
    "--chunk-index", type=int, help="Execute one explicit chunk; other rows remain unexecuted in the manifest"
  )
  parser.add_argument("--svl-bytes", type=int, help="Audit geometry; execution requires the actual current SVL")
  parser.add_argument("--list", action="store_true")
  parser.add_argument(
    "--verify",
    nargs="+",
    type=Path,
    help="Audit disjoint cohort/chunk manifests; nonzero if any execution is missing, failed or stale",
  )
  parser.add_argument("--manifest", type=Path)
  parser.add_argument("--output", type=Path)
  parser.add_argument("--build-dir", type=Path, default=ROOT / "build-dsl-matmul-source-catalog")
  parser.add_argument("--rounds", type=int, default=7)
  parser.add_argument("--jobs", type=int, default=8)
  parser.add_argument("--cpu", type=int, default=default_cpu())
  parser.add_argument("--fusion", choices=("auto", "always", "never"), default="auto")
  parser.add_argument("--schedule-pipeline", choices=("auto", "off", "prefetch", "staged"), default="auto")
  add_toolchain_arguments(parser)
  args = parser.parse_args(argv)
  if args.chunk_size < 1 or args.chunk_index is not None and args.chunk_index < 0:
    raise ValueError("chunk size must be positive and index nonnegative")
  validate_pipeline_contract()
  cases = load_cases(svl_bytes=args.svl_bytes)
  all_scenarios = tuple(scenarios(cases, levels=args.level, batch_strategy=args.batch_strategy))
  selected = select_scenarios(
    all_scenarios,
    cohorts=args.cohort,
    modes=args.mode,
    ranks=args.rank,
    native=args.native,
    shapes=args.shape,
    packing=args.packing,
  )
  if args.pipeline:
    selected = tuple(s for s in selected if s.case.output_pipeline in args.pipeline)
    missing = set(args.pipeline) - {s.case.output_pipeline for s in selected}
    if missing:
      raise ValueError(f"requested pipelines have no matching source executions: {sorted(missing)}")
  chunks = [selected[start : start + args.chunk_size] for start in range(0, len(selected), args.chunk_size)]
  if args.chunk_index is not None and args.chunk_index >= len(chunks):
    raise ValueError(f"chunk index exceeds {len(chunks)} selected chunks")
  chosen = chunks if args.chunk_index is None else [chunks[args.chunk_index]]
  options = {"cpu_matmul_fusion": args.fusion, "cpu_matmul_pipeline": args.schedule_pipeline}
  audit = audit_document(cases, all_scenarios)
  audit.update(
    pass_options=options,
    selected=len(selected),
    chunks=len(chunks),
    chunk_index=args.chunk_index,
    selected_execution_ids=[s.execution_id for chunk in chosen for s in chunk],
    requested_execution_ids=[s.execution_id for s in selected],
    results={},
    artifact_sources={},
    executed=False,
    completed=0,
    failures=0,
  )
  if args.verify:
    result = verify_manifests(args.verify, selected, audit)
    print(json.dumps(result, indent=2))
    return 0 if result["complete"] else 1

  def save():
    if args.manifest:
      args.manifest.parent.mkdir(parents=True, exist_ok=True)
      args.manifest.write_text(json.dumps(audit, indent=2) + "\n")

  save()
  if args.list:
    print(json.dumps(audit, indent=2))
    return 0
  if not args.output or not args.manifest:
    raise ValueError("execution requires --output and --manifest to retain complete coverage evidence")
  if args.svl_bytes is not None:
    from vecops._arm_vector_state import get_vector_state

    if get_vector_state().sme_bytes != args.svl_bytes:
      raise ValueError("audit SVL differs from the execution thread's actual SVL")
  if not is_arm() and any(s.types[4] not in ("bfloat16", "int8", "uint8") for s in selected):
    raise RuntimeError("ARM source registrations require ARM; an x86 smoke must explicitly filter --native")
  print(f"{len(selected)} source executions in {len(chunks)} chunks; executing {sum(map(len, chosen))}", flush=True)
  isolate_compilation(args.cpu)
  torch.set_num_threads(1)
  torch.set_num_interop_threads(1)
  compiler = vecops.Compiler(
    target=args.target,
    cc=args.cc,
    cxx=args.cxx,
    jobs=args.jobs,
    cache_dir=args.build_dir / "cache",
    build_dir=args.build_dir / "build",
  )
  args.output.parent.mkdir(parents=True, exist_ok=True)
  failures = 0
  audit["executed"] = True
  save()
  with args.output.open("w") as stream:
    writer = csv.DictWriter(
      stream,
      fieldnames=(
        "execution_id",
        "catalog",
        "case",
        "m",
        "n",
        "k",
        "batch",
        "broadcast",
        "meta",
        "level",
        "packing",
        "reference_family",
        "reference_policy",
        "batch_strategy",
        "dsl_us",
        "cpp_us",
        "dsl_over_cpp",
        "status",
        "error",
      ),
    )
    writer.writeheader()
    for chunk in chosen:
      cache, prepared = {}, []
      try:
        with vecops.collect_compile_requests() as collector:
          for scenario in chunk:
            key = scenario.compilation_key
            if key not in cache:
              pair, packing = source_templates(scenario)
              pairs = [
                tuple(
                  T.compile(t, compiler=compiler, pass_options=options if i == 0 else None) for i, t in enumerate(pair)
                )
              ]
              if packing:
                pairs.append(
                  tuple(
                    T.compile(t, compiler=compiler, pass_options=options if i == 0 else None)
                    for i, t in enumerate(packing)
                  )
                )
              cache[key] = pairs
            pairs = cache[key]
            # Do not collect just the first Dynamic observation: every call
            # must still undergo binding/relationship and alias validation.
            for pair in pairs:
              for kernel in pair:
                kernel.operator(*source_metadata(scenario), 1, **kernel._core.specialization_values)
            prepared.append((scenario, pairs))
            audit["artifact_sources"][scenario.execution_id] = {
              ("pack_" if index else "") + role: str(kernel.source)
              for index, pair in enumerate(pairs)
              for role, kernel in zip(("dsl", "cpp"), pair)
            }
        vecops.compile_batch(collector.requests, parallelism=args.jobs, execution_parallelism=1)
      except Exception as error:  # noqa: BLE001 -- retain unexpected compiler failures in the complete ledger
        for scenario in chunk:
          row = {**result_metadata(scenario), "status": "compile_failed", "error": str(error)}
          writer.writerow(row)
          audit["results"][scenario.execution_id] = row
        stream.flush()
        failures += len(chunk)
        audit.update(completed=len(audit["results"]), failures=failures)
        save()
        continue
      with reserve_cpu(args.cpu):
        original_affinity = os.sched_getaffinity(0)
        os.sched_setaffinity(0, {args.cpu})
        try:
          for scenario, pairs in prepared:
            row = result_metadata(scenario)
            try:
              values, expected = source_arguments(scenario)
              if len(pairs) == 2:
                for i, packer in enumerate(pairs[1]):
                  packer.operator.native(*values, 1, **packer._core.specialization_values)
                  packed = tuple(values[index].copy() for index in (11, 12))
                  if i == 0:
                    expected_packed = packed
                  else:
                    for actual, reference, active in zip(
                      packed, expected_packed, (scenario.pack_a, scenario.pack_b), strict=True
                    ):
                      if active:
                        np.testing.assert_array_equal(actual, reference)
              c = values[2]
              pair = pairs[0]

              def calls(repeat, values=values, pair=pair):
                return tuple(
                  lambda kernel=kernel: kernel.operator.native(*values, repeat, **kernel._core.specialization_values)
                  for kernel in pair
                )

              dsl, cpp = calls(1)
              tolerance = 2e-2 if c.dtype.itemsize <= 2 else 2e-10 if c.dtype.itemsize == 8 else 8e-4
              validate_kernel_pair(dsl, cpp, c, expected, tolerance=tolerance)
              repeat = max(
                2, min(4000, 20_000_000 // (scenario.batch * scenario.case.m * scenario.case.n * scenario.case.k))
              )
              td, tc = measure(calls(repeat), repeat, args.rounds)
              row.update(dsl_us=td, cpp_us=tc, dsl_over_cpp=td / tc, status="passed", error="")
            except Exception as error:  # noqa: BLE001 -- persist failure instead of dropping a source registration
              row.update(status="validation_failed", error=str(error))
              failures += 1
            writer.writerow(row)
            stream.flush()
            audit["results"][scenario.execution_id] = row
            audit.update(completed=len(audit["results"]), failures=failures)
            save()
            print(f"{scenario.execution_id} {scenario.key}: {row['status']}", flush=True)
        finally:
          os.sched_setaffinity(0, original_affinity)
  audit["completed"] = len(audit["results"])
  audit["failures"] = failures
  save()
  return 1 if failures else 0


if __name__ == "__main__":
  sys.exit(main())
