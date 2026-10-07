"""Paired Fusion/Orientation catalogs with the real public C++ Matmul.

All selected metadata are audited before compilation. Unsupported cases are
errors, never silently dropped. Batched registrations use the public rank-3
Matmul, including its shared-operand strategies. Asymmetric compensation is precomputed once and
passed as the same accumulator to both implementations, as on ARM in the
original C++ benchmark. Native repeat loops exclude Python/JIT from timing.

T.dot/library.matmul themselves currently accept rank two. The batched DSL
is a user-level wrapper: serial by default for independent inputs, optionally
zero-copy batch*M folding for contiguous shared-B inputs. CSV records this
algorithm choice; it does not claim native rank-three or T.grid scheduling.
"""

import argparse
import csv
import json
import os
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import vecops
import vecops.language as T
from dsl_benchmark_utils import add_toolchain_arguments, default_cpu, is_arm
from dsl_matmul_catalog import CatalogCase, load_cases
from dsl_timing import isolate_compilation, reserve_cpu
from dsl_transpose import aligned_empty, measure, numpy_dtype

ROOT = Path(__file__).resolve().parents[1]
DTYPES = {
  "bf16": "bfloat16",
  "fp16": "float16",
  "fp32": "float32",
  "fp64": "float64",
  "int8": "int8",
  "uint8": "uint8",
  "int32": "int32",
}
INPUTS = (
  "Convert",
  "Quantize4",
  "AsymmetricQuantize4Zp3",
  "AsymmetricQuantize4Zp3Zp5",
  "PrequantizedAsymmetricZp3",
  "RuntimePerRowAsymmetricQuantize",
)
OUTPUTS = (
  "Convert",
  "Accumulate",
  "Relu",
  "Sigmoid",
  "Silu",
  "Scale",
  "DynamicScale",
  "PerColumnScale",
  "Clamp",
  "AccumulateReluScale",
  "Dequantize",
  "ReluRequantize",
  "RequantizeU8Zp7",
  "Bias",
  "BiasRelu",
  "BiasClamp",
  "BiasSilu",
  "BiasPerColumnDequantize",
  "BiasPerColumnRequantizeU8",
  "RuntimePerRowColumnDequantize",
  "AsymmetricDequantize",
)
BIAS = frozenset(name for name in OUTPUTS if name.startswith("Bias"))
ACCUMULATE = frozenset(("Accumulate", "AccumulateReluScale"))
ASYMMETRIC = frozenset(INPUTS[2:])


def validate_pipeline_contract():
  """An enum addition/reordering must update this runner, not alter semantics silently."""
  original = (ROOT / "benchmarks/ops/MatmulScenarioBenchCommon.h").read_text()
  reference = (ROOT / "benchmarks/dsl_matmul_fusion_reference.cpp").read_text()
  for expected, source_name, reference_name in (
    (INPUTS, "InputPipeline", "Input"),
    (OUTPUTS, "OutputPipeline", "Output"),
  ):
    for text, name in ((original, source_name), (reference, reference_name)):
      body = re.search(r"enum class " + name + r"\s*\{([^}]+)\}", text).group(1)
      actual = tuple(value.strip() for value in body.split(",") if value.strip())
      if actual != expected:
        raise RuntimeError(f"{name} changed: update the paired pipeline implementation")


@dataclass(frozen=True)
class Scenario:
  case: CatalogCase
  mode: str
  level: str
  batch_strategy: str = "auto"

  @property
  def batched(self):
    return self.case.batch is not None

  @property
  def batch(self):
    return int(self.case.batch) if self.batched else 1

  @property
  def shared_a(self):
    return self.batched and self.case.broadcast == "A"

  @property
  def shared_b(self):
    return self.batched and self.case.broadcast == "B"

  @property
  def shared_prior(self):
    return self.case.output_pipeline in BIAS or self.asymmetric and self.broadcast_prior and self.shared_b

  @property
  def flatten_batch(self):
    # Equivalent to public Matmul's shared-B batching fold, not a weaker
    # rank-two native reference. Only contiguous A/C admit this zero-copy view.
    at, _, ct = self.orientation
    return self.batch_strategy == "auto" and self.shared_b and not at and not ct

  @property
  def orientation(self):
    if self.case.catalog != "MatmulOrientation":
      return False, True, False  # C++ Fusion uses [N,K] storage for B.
    values = dict(re.findall(r"([ABC])=([RT])", self.case.orientation or ""))
    if set(values) != {"A", "B", "C"}:
      raise ValueError(f"unresolved orientation: {self.case.orientation}")
    return tuple(values[name] == "T" for name in "ABC")

  @property
  def swap(self):
    return "EnableSwapAB=false" not in (self.case.configuration or "")

  @property
  def asymmetric(self):
    return self.case.input_pipeline_a in ASYMMETRIC or self.case.input_pipeline_b in ASYMMETRIC

  @property
  def broadcast_prior(self):
    return (
      self.case.output_pipeline in BIAS or self.asymmetric and self.case.input_pipeline_a != "AsymmetricQuantize4Zp3Zp5"
    )

  @property
  def has_prior(self):
    return self.asymmetric or self.case.output_pipeline in BIAS | ACCUMULATE

  @property
  def types(self):
    case = self.case
    prior = case.accumulator if self.asymmetric else case.memory_c_input
    return tuple(
      DTYPES[name]
      for name in (case.memory_a, case.memory_b, case.memory_c, prior, case.native_a, case.native_b, case.accumulator)
    )

  @property
  def key(self):
    case = self.case
    return "/".join(
      map(
        str,
        (
          case.catalog,
          case.line,
          case.name,
          case.m,
          case.n,
          case.k,
          self.mode,
          self.level,
          self.batch,
          self.case.broadcast,
          self.flatten_batch,
          *self.types,
          case.input_pipeline_a,
          case.input_pipeline_b,
          case.output_pipeline,
          self.orientation,
          self.swap,
        ),
      )
    )

  @property
  def compilation_key(self):
    case = self.case
    return (
      self.mode,
      self.level,
      self.batched,
      self.case.broadcast,
      self.flatten_batch,
      self.types,
      case.input_pipeline_a,
      case.input_pipeline_b,
      case.output_pipeline,
      self.orientation,
      self.swap,
      (self.batch, case.m, case.n, case.k) if self.mode == "Const" else None,
    )

  def validate(self):
    case = self.case
    if self.batched and (self.batch <= 0 or case.broadcast not in ("A", "B", "independent")):
      raise ValueError("unresolved batch extent or operand broadcast mode")
    if not all(isinstance(extent, int) and extent > 0 for extent in (case.m, case.n, case.k)):
      raise ValueError("unresolved or nonpositive shape")
    if case.input_modes not in ((), ("raw",)):
      raise ValueError(f"packing mode is not implemented by this raw runner: {case.input_modes}")
    if case.input_pipeline_a not in INPUTS or case.input_pipeline_b not in INPUTS:
      raise ValueError("unknown input transform")
    if case.output_pipeline not in OUTPUTS:
      raise ValueError("unknown output transform")
    if case.input_pipeline_b == "RuntimePerRowAsymmetricQuantize":
      raise ValueError("runtime per-row quantization is defined for A only")
    _ = self.types, self.orientation


def templates(scenario):
  case = scenario.case
  M, N, K = (case.m, case.n, case.k) if scenario.mode == "Const" else tuple(T.dynamic(lower=1) for _ in range(3))
  ad, bd, cd, dd, na, nb, acc = scenario.types
  at, bt, ct = scenario.orientation
  astride = (1, M) if at else (K, 1)
  bstride = (1, K) if bt else (N, 1)
  cstride = (1, M) if ct else (N, 1)
  DM = 1 if scenario.broadcast_prior else M
  batched = scenario.batched
  flatten_batch, flat_prior = scenario.flatten_batch, scenario.shared_prior
  Batch = scenario.batch if scenario.mode == "Const" else T.dynamic(lower=1)
  if batched:
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
  input_a, input_b, output = case.input_pipeline_a, case.input_pipeline_b, case.output_pipeline
  prior, swap, level = scenario.has_prior, T.Const(scenario.swap), scenario.level
  quant_a, quant_b = input_a in INPUTS[1:4], input_b in INPUTS[1:4]
  azp = 3 if input_a != "Quantize4" and na == "uint8" else 0
  bzp = 5 if input_b == "AsymmetricQuantize4Zp3Zp5" else 0
  float_epilogue = output in (
    "Dequantize",
    "ReluRequantize",
    "RequantizeU8Zp7",
    "AsymmetricDequantize",
    "BiasPerColumnDequantize",
    "BiasPerColumnRequantizeU8",
    "RuntimePerRowColumnDequantize",
  )
  relu_epilogue = output in ("Relu", "BiasRelu", "AccumulateReluScale", "ReluRequantize")
  silu_epilogue = output in ("Silu", "BiasSilu")
  scale_epilogue = output in ("Scale", "AccumulateReluScale")
  clamp_epilogue = output in ("Clamp", "BiasClamp")
  dequant_epilogue = output in ("Dequantize", "ReluRequantize")
  column_quant_epilogue = output in ("BiasPerColumnDequantize", "BiasPerColumnRequantizeU8")

  @T.func
  def compute(
    A: T.Tensor((M, K), astride, ad),
    B: T.Tensor((K, N), bstride, bd),
    C: T.Tensor((M, N), cstride, cd, access="output"),
    D: T.Tensor((DM, N), (N, 1), dd),
    S: T.Tensor((1, 1), dtype=acc),
    SC: T.Tensor((1, N), dtype=acc),
    SQ: T.Tensor((1, N), dtype="float32"),
    RQ: T.Tensor((M, 1), dtype="float32"),
    RD: T.Tensor((M, 1), dtype="float32"),
    W: T.Tensor((1, N), dtype="float32"),
    Z: T.Tensor((1, 1), dtype="int32"),
    repeat: int,
  ):
    for _iteration in T.serial(repeat, unroll=1):
      if input_a == "RuntimePerRowAsymmetricQuantize":
        a = T.cast(T.load(A, dtype="float32") * T.load(RQ) + T.load(Z, dtype="float32"), na)
      elif quant_a:
        a = T.cast(T.load(A, dtype="float32") * 4 + azp, na)
      else:
        a = T.load(A, dtype=na)
      if quant_b:
        b = T.cast(T.load(B, dtype="float32") * 4 + bzp, nb)
      else:
        b = T.load(B, dtype=nb)
      initial = T.load(D, dtype=acc) if prior else None
      if level == "dot":
        result = T.dot((a, b), C, acc=initial, acc_dtype=acc)
      else:
        result = T.library.matmul((a, b), C, acc=initial, acc_dtype=acc, swap=swap)
      if float_epilogue:
        result = T.cast(result, "float32")
      if relu_epilogue:
        result = T.max(result, 0)
      if output == "Sigmoid":
        result = 1 / (1 + T.exp(-result))
      elif silu_epilogue:
        result = result * (1 / (1 + T.exp(-result)))
      elif scale_epilogue:
        result = result * 0.5
      elif output == "DynamicScale":
        result = result * T.load(S)
      elif output == "PerColumnScale":
        result = result * T.load(SC)
      elif clamp_epilogue:
        result = T.min(T.max(result, -0.25), 0.25)
      elif dequant_epilogue:
        result = result * 0.125
      elif output == "RequantizeU8Zp7":
        result = result * 0.125 + 7
      elif output == "AsymmetricDequantize":
        result = result * 0.0625
      elif column_quant_epilogue:
        result = result * T.load(SQ)
        if output == "BiasPerColumnRequantizeU8":
          result = result + T.load(Z, dtype="float32")
      elif output == "RuntimePerRowColumnDequantize":
        result = result * T.load(RD) * T.load(W)
      T.store(C, result)
      T.compiler_fence()

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
    repeat: int,
  ):
    for _iteration in T.serial(repeat, unroll=1):
      if flatten_batch:
        a = T.reshape(A, (A.shape[0] * A.shape[1], A.shape[2]))
        c = T.reshape(C, (C.shape[0] * C.shape[1], C.shape[2]))
        rq = T.reshape(RQ, (RQ.shape[0] * RQ.shape[1], 1))
        rd = T.reshape(RD, (RD.shape[0] * RD.shape[1], 1))
        if flat_prior:
          d = D[0]
        else:
          d = T.reshape(D, (D.shape[0] * D.shape[1], D.shape[2]))
        compute(a, B[0], c, d, S, SC, SQ, rq, rd, W, Z, 1)
      elif batched:
        for batch in T.serial(Batch, unroll=1):
          compute(A[batch], B[batch], C[batch], D[batch], S, SC, SQ, RQ[batch], RD[batch], W, Z, 1)
      else:
        compute(A, B, C, D, S, SC, SQ, RQ, RD, W, Z, 1)

  NativeA, NativeB = T.DType(getattr(vecops, na)), T.DType(getattr(vecops, nb))
  InputPipelineA, InputPipelineB = T.Const(INPUTS.index(input_a)), T.Const(INPUTS.index(input_b))
  OutputPipeline = T.Const(OUTPUTS.index(output))
  HasPrior, BroadcastPrior, SwapAB = T.Const(prior), T.Const(scenario.broadcast_prior), T.Const(scenario.swap)

  @T.template(source=ROOT / "benchmarks/dsl_matmul_fusion_reference.cpp")
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
    repeat: int,
  ):
    NativeA, NativeB, InputPipelineA, InputPipelineB, OutputPipeline, HasPrior, BroadcastPrior, SwapAB  # noqa: B018

  return dsl, cpp


def matrix(shape, dtype, transposed=False):
  storage_shape = (*shape[:-2], *shape[-2:][::-1]) if transposed else shape
  result = aligned_empty(storage_shape, dtype)
  return result.swapaxes(-1, -2) if transposed else result


def padded_parameter(shape, dtype):
  """C++ lane-local transforms may use unmasked loads; provide the same tail padding."""
  storage = aligned_empty((int(np.prod(shape)) + 64,), dtype)
  storage.fill(0)
  return np.ndarray(shape, dtype=storage.dtype, buffer=storage)


def transformed_input(value, native, pipeline, side, row_quant, zero):
  result = value.astype(numpy_dtype(native))
  if pipeline == "RuntimePerRowAsymmetricQuantize":
    result = value.astype(np.float32) * row_quant + zero
  elif pipeline in INPUTS[1:4]:
    zp = (
      (3 if side == "A" else 5)
      if pipeline == "AsymmetricQuantize4Zp3Zp5"
      else (3 if pipeline == "AsymmetricQuantize4Zp3" and side == "A" and native == "uint8" else 0)
    )
    result = value.astype(np.float32) * 4 + zp
  return result.astype(numpy_dtype(native))


def arguments(scenario):
  """Build deterministic source-equivalent data and untimed asymmetric correction."""
  case = scenario.case
  m, n, k = case.m, case.n, case.k
  ad, bd, cd, dd, na, nb, acc = scenario.types
  at, bt, ct = scenario.orientation
  batch = scenario.batch
  ab, bb = (1 if scenario.shared_a else batch), (1 if scenario.shared_b else batch)
  db = 1 if scenario.shared_prior else batch
  ashape, bshape, cshape = (m, k), (k, n), (m, n)
  dshape, rshape = (1 if scenario.broadcast_prior else m, n), (m, 1)
  if scenario.batched:
    ashape, bshape, cshape = (ab, *ashape), (bb, *bshape), (batch, *cshape)
    dshape, rshape = (db, *dshape), (ab, *rshape)
  a, b, c = matrix(ashape, ad, at), matrix(bshape, bd, bt), matrix(cshape, cd, ct)
  d = matrix(dshape, dd)
  s, sc = padded_parameter((1, 1), acc), padded_parameter((1, n), acc)
  sq, rq = padded_parameter((1, n), "float32"), padded_parameter(rshape, "float32")
  rd, w = padded_parameter(rshape, "float32"), padded_parameter((1, n), "float32")
  z = padded_parameter((1, 1), "int32")
  s.fill(0.75)
  sc[:] = 0.5 + (np.arange(n) % 7) * 0.0625
  sq[:] = 0.015625 + (np.arange(n) % 7) * 0.00390625
  rq[...] = (4 * (2 ** (np.arange(ab * m if scenario.batched else m) % 3))).reshape(rshape)
  rd[:] = 1 / rq
  w[:] = 0.03125 + (np.arange(n) % 5) * 0.0078125
  z.fill(7)

  def fill(value, native, pipeline, seed, side):
    # Fusion B's physical [N,K] order matches the original benchmark.
    indices = (
      np.arange(value.size).reshape((*value.shape[:-2], *value.shape[-2:][::-1])).swapaxes(-1, -2)
      if side == "B"
      else np.arange(value.size).reshape(value.shape)
    )
    if pipeline in INPUTS[1:4]:
      low = 0 if native == "uint8" else -3
      result = (low + (indices * 5 + seed) % 7) * 0.25
    elif value.dtype.kind in "iu":
      result = (0 if value.dtype.kind == "u" else -3) + (indices * 5 + seed) % 7
    else:
      result = ((indices * 7 + seed) % 17 - 8) / 16
    value[:] = result

  fill(a, na, case.input_pipeline_a, 3, "A")
  fill(b, nb, case.input_pipeline_b, 11, "B")
  fill(d, acc, "Convert", 5, "A")
  if case.catalog == "MatmulOrientation":
    for value, seed in ((a, 3), (b, 11)):
      rows, columns = np.indices(value.shape)
      value[:] = ((rows * 13 + columns * 7 + seed) % 29 - 14) / 31
  if case.input_pipeline_a == "RuntimePerRowAsymmetricQuantize":
    a[:] = ((np.arange(a.size).reshape(a.shape) % 7) - 3) * rd
  aa = transformed_input(a, na, case.input_pipeline_a, "A", rq, z).astype(numpy_dtype(acc))
  bb = transformed_input(b, nb, case.input_pipeline_b, "B", rq, z).astype(numpy_dtype(acc))
  if scenario.asymmetric:
    zp = 7 if case.input_pipeline_a == "RuntimePerRowAsymmetricQuantize" else 3
    compensation = -zp * bb.sum(axis=-2, dtype=np.int32)[..., None, :]
    if case.input_pipeline_a == "AsymmetricQuantize4Zp3Zp5":
      compensation = compensation - 5 * aa.sum(axis=-1, dtype=np.int32)[..., :, None] + k * 15
    d[:] = compensation
  result = aa @ bb
  if scenario.has_prior:
    result = result + d.astype(numpy_dtype(acc))
  out = case.output_pipeline
  if out in ("Relu", "BiasRelu", "AccumulateReluScale", "ReluRequantize"):
    result = np.maximum(result, 0)
  if out == "Sigmoid":
    result = 1 / (1 + np.exp(-result))
  elif out in ("Silu", "BiasSilu"):
    result = result / (1 + np.exp(-result))
  elif out in ("Scale", "AccumulateReluScale"):
    result = result * 0.5
  elif out == "DynamicScale":
    result = result * s
  elif out == "PerColumnScale":
    result = result * sc
  elif out in ("Clamp", "BiasClamp"):
    result = np.clip(result, -0.25, 0.25)
  elif out in ("Dequantize", "ReluRequantize"):
    result = result * 0.125
  elif out == "RequantizeU8Zp7":
    result = result * 0.125 + 7
  elif out == "AsymmetricDequantize":
    result = result * 0.0625
  elif out in ("BiasPerColumnDequantize", "BiasPerColumnRequantizeU8"):
    result = result * sq
    if out == "BiasPerColumnRequantizeU8":
      result = result + z
  elif out == "RuntimePerRowColumnDequantize":
    result = result * rd * w
  if c.dtype.kind in "iu":
    bounds = np.iinfo(c.dtype)
    result = np.clip(result, bounds.min, bounds.max)
  if scenario.batched:
    a, b, d, rq, rd = (
      np.lib.stride_tricks.as_strided(
        value, shape=(batch, *value.shape[-2:]), strides=(0, *value.strides[-2:]), writeable=False
      )
      if shared
      else value
      for value, shared in (
        (a, scenario.shared_a),
        (b, scenario.shared_b),
        (d, scenario.shared_prior),
        (rq, scenario.shared_a),
        (rd, scenario.shared_a),
      )
    )
  return (a, b, c, d, s, sc, sq, rq, rd, w, z), result.astype(c.dtype)


def metadata(scenario):
  case = scenario.case
  m, n, k = case.m, case.n, case.k
  ad, bd, cd, dd, _, _, acc = scenario.types
  at, bt, ct = scenario.orientation

  def tensor(shape, dtype, transpose=False, shared=False):
    strides = (1, shape[-2]) if transpose else (shape[-1], 1)
    if len(shape) == 3:
      strides = (0 if shared else shape[-2] * shape[-1], *strides)
    return vecops.TensorMeta(shape, strides=strides, dtype=getattr(vecops, dtype))

  prefix = (scenario.batch,) if scenario.batched else ()
  return (
    tensor((*prefix, m, k), ad, at, scenario.shared_a),
    tensor((*prefix, k, n), bd, bt, scenario.shared_b),
    tensor((*prefix, m, n), cd, ct),
    tensor((*prefix, 1 if scenario.broadcast_prior else m, n), dd, shared=scenario.shared_prior),
    tensor((1, 1), acc),
    tensor((1, n), acc),
    tensor((1, n), "float32"),
    tensor((*prefix, m, 1), "float32", shared=scenario.shared_a),
    tensor((*prefix, m, 1), "float32", shared=scenario.shared_a),
    tensor((1, n), "float32"),
    tensor((1, 1), "int32"),
  )


def main():
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--build-dir", type=Path, default=ROOT / "build-dsl-matmul-fusion-catalog")
  parser.add_argument("--output", type=Path)
  parser.add_argument("--manifest", type=Path)
  parser.add_argument("--catalog", nargs="+", choices=("fusion", "orientation"), default=["fusion", "orientation"])
  parser.add_argument("--rank", nargs="+", type=int, choices=(2, 3), default=[2, 3])
  parser.add_argument("--shape", nargs="+")
  parser.add_argument("--pipeline", nargs="+", choices=OUTPUTS)
  parser.add_argument("--native", nargs="+", choices=tuple(DTYPES))
  parser.add_argument("--mode", nargs="+", choices=("Const", "Dynamic"), default=["Const", "Dynamic"])
  parser.add_argument("--level", nargs="+", choices=("dot", "matmul"), default=["matmul"])
  parser.add_argument("--limit", type=int, help="Explicit smoke subset, printed with the selected/full counts")
  parser.add_argument("--list", action="store_true")
  parser.add_argument("--rounds", type=int, default=7)
  parser.add_argument("--jobs", type=int, default=8)
  parser.add_argument("--cpu", type=int, default=default_cpu())
  parser.add_argument("--fusion", choices=("auto", "always", "never"), default="auto")
  parser.add_argument("--schedule-pipeline", choices=("auto", "off", "prefetch", "staged"), default="auto")
  parser.add_argument("--batch-strategy", choices=("auto", "serial"), default="auto")
  add_toolchain_arguments(parser)
  args = parser.parse_args()
  validate_pipeline_contract()
  selected = []
  rank2, batched = 0, 0
  for case in load_cases():
    catalog = (
      "fusion"
      if case.catalog == "MatmulFusionExpanded"
      else "orientation"
      if case.catalog == "MatmulOrientation"
      else None
    )
    if catalog not in args.catalog:
      continue
    if case.batch is not None:
      batched += 1
      if 3 not in args.rank:
        continue
    else:
      rank2 += 1
      if 2 not in args.rank:
        continue
    if args.shape and f"{case.m}x{case.n}x{case.k}" not in args.shape:
      continue
    if args.pipeline and case.output_pipeline not in args.pipeline:
      continue
    if args.native and case.native_a not in args.native:
      continue
    for mode in args.mode:
      if mode not in case.metadata:
        continue
      for level in args.level:
        scenario = Scenario(case, mode, level, args.batch_strategy)
        scenario.validate()
        selected.append(scenario)
  full = len(selected)
  if args.limit is not None:
    selected = selected[: args.limit]
  audit = {
    "ranks": args.rank,
    "source_rank2_rows": rank2,
    "source_rank3_rows": batched,
    "selected_before_limit": full,
    "selected": len(selected),
    "cases": [
      dict(scenario.case.as_dict(), meta=scenario.mode, level=scenario.level, execution_key=scenario.key)
      for scenario in selected
    ],
  }
  if args.manifest:
    args.manifest.write_text(json.dumps(audit, indent=2) + "\n")
  if args.list:
    print(json.dumps(audit, indent=2))
    return
  if not selected:
    raise ValueError("no cases selected")
  if not is_arm() and any(s.types[4] not in ("bfloat16", "int8", "uint8") for s in selected):
    raise RuntimeError(
      "ARM native fp16/fp32/fp64 catalog requires the ARM target; filter --native for an x86 smoke run"
    )
  print(
    f"{len(selected)}/{full} selected configurations; source rows: {rank2} rank-2 + {batched} rank-3",
    flush=True,
  )
  isolate_compilation(args.cpu)
  args.build_dir.mkdir(parents=True, exist_ok=True)
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
  options = {"cpu_matmul_fusion": args.fusion, "cpu_matmul_pipeline": args.schedule_pipeline}
  cache, cases = {}, []
  with vecops.collect_compile_requests() as collector:
    for scenario in selected:
      key = scenario.compilation_key
      if key not in cache:
        dsl, cpp = templates(scenario)
        pair = T.compile(dsl, compiler=compiler, pass_options=options), T.compile(cpp, compiler=compiler)
        for kernel in pair:
          kernel.operator(*metadata(scenario), 1, **kernel._core.specialization_values)
        cache[key] = pair
      cases.append((scenario, cache[key]))
  print(f"Compiling {len(collector)} artifacts in one batch", flush=True)
  vecops.compile_batch(collector.requests, parallelism=args.jobs, execution_parallelism=1)
  with reserve_cpu(args.cpu):
    os.sched_setaffinity(0, {args.cpu})
    stream = args.output.open("w") if args.output else __import__("sys").stdout
    writer = csv.writer(stream)
    writer.writerow(
      (
        "case",
        "m",
        "n",
        "k",
        "batch",
        "broadcast",
        "batch_strategy",
        "meta",
        "level",
        "memory_a",
        "memory_b",
        "memory_c",
        "native_a",
        "native_b",
        "acc",
        "input_a",
        "input_b",
        "output",
        "orientation",
        "swap",
        "dsl_us",
        "cpp_us",
        "dsl_over_cpp",
      )
    )
    for scenario, pair in cases:
      values, expected = arguments(scenario)
      c = values[2]

      def calls(repeat, values=values, pair=pair):
        return tuple(
          lambda kernel=kernel: kernel.operator.native(*values, repeat, **kernel._core.specialization_values)
          for kernel in pair
        )

      dsl, cpp = calls(1)
      cpp()
      reference = c.copy()
      tolerance = 2e-2 if c.dtype.itemsize <= 2 else 2e-10 if c.dtype.itemsize == 8 else 8e-4
      np.testing.assert_allclose(reference, expected, rtol=tolerance, atol=tolerance)
      dsl()
      if c.dtype.kind in "iu":
        np.testing.assert_array_equal(c, reference)
      else:
        np.testing.assert_allclose(c, reference, rtol=tolerance, atol=tolerance)
      case = scenario.case
      repeat = max(2, min(4000, 20_000_000 // (scenario.batch * case.m * case.n * case.k)))
      td, tc = measure(calls(repeat), repeat, args.rounds)
      writer.writerow(
        (
          case.name,
          case.m,
          case.n,
          case.k,
          scenario.batch,
          case.broadcast or "none",
          "flatten" if scenario.flatten_batch else "serial",
          scenario.mode,
          scenario.level,
          case.memory_a,
          case.memory_b,
          case.memory_c,
          case.native_a,
          case.native_b,
          case.accumulator,
          case.input_pipeline_a,
          case.input_pipeline_b,
          case.output_pipeline,
          case.orientation or "A=R,B=T,C=R",
          scenario.swap,
          td,
          tc,
          td / tc,
        )
      )
      stream.flush()
      if args.output:
        print(f"{scenario.key}: DSL/C++ {td / tc:.3f}", flush=True)
    if args.output:
      stream.close()


if __name__ == "__main__":
  main()
