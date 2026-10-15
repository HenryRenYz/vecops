# SPDX-FileCopyrightText: 2026 HenryRenYz and vecops contributors
# SPDX-License-Identifier: MIT
"""Exact supplemental registration ledger; separate from public Matmul cohorts.

Native registration counts include forced implementation experiments. Prepared
plans and shared-weight lifecycles are not substituted with stateless matmul.
The shard-parity metadata of WeightReuse is intentional and preserved.
"""

import argparse
import hashlib
import json
import re
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DTYPES = {"bfloat16_t": "bfloat16", "float16_t": "float16", "float32_t": "float32",
          "float64_t": "float64", "int8_t": "int8", "uint8_t": "uint8", "int32_t": "int32"}
SELECTORS = {"Automatic": 0, "Vector": 1, "SMEPostprocess": 2, "SMEFP32ToFP64": 3,
             "SMEStagedTransform": 4, "SMEStagedFP16ToFP32": 5}


def atom_types(atom):
  if "BF16" in atom:
    return "bfloat16", "bfloat16"
  if "F16" in atom:
    return "float16", "float16"
  if "F64" in atom:
    return "float64", "float64"
  if "F32" in atom:
    return "float32", "float32"
  suffix = atom.rsplit("_", 1)[-1].lower()
  return tuple("uint8" if token == "u8" else "int8" for token in (suffix[:2], suffix[2:]))


def _row(architecture, suite, **values):
  row = {"architecture": architecture, "suite": suite, **values}
  row["registration_id"] = hashlib.sha256(json.dumps(row, sort_keys=True).encode()).hexdigest()[:24]
  return row


def _pack_cases(root, architecture, features):
  common = (root / "benchmarks/ops/MatmulPackBenchCommon.h").read_text()
  source = (root / "benchmarks/ops/MatmulPackBench.cpp").read_text()
  # This boundary is the top-level x86/ARM split, not a nested feature gate.
  x86, arm = source.split("#else\n\n#include \"MatmulPackBenchCommon.h\"", 1)
  source = x86 if architecture == "x86" else arm
  prefix = "AMX" if architecture == "x86" else "SME"
  names = dict(re.findall(r'VECOPS_PACK_CASE_NAME\((\w+),\s*"([^"]+)"\)', common))
  groups = dict(re.findall(r'struct (\w+) \{ static constexpr const char\* name = "([^"]+)";', common))
  tables = {}
  for table in ("RepresentativeCases", "SecondaryCases"):
    body = re.search(r'using ' + table + r' = CaseList<(.*?)>;', common, re.DOTALL)[1]
    tables[table] = [(groups[g], names[n], int(s), int(k), a != "false", b != "false")
                     for g, n, s, k, a, b in re.findall(
                       r'PackCase<(\w+),\s*(\w+),\s*(\d+),\s*(\d+)(?:,\s*(true|false),\s*(true|false))?\s*>', body)]
  atoms = [f"{prefix}_BF16F32", f"{prefix}_F16F32"]
  if architecture == "arm":
    atoms.append("SME_F32F32")
  atoms += [f"{prefix}_I8I32_{s}" for s in ("s8s8", "s8u8", "u8s8", "u8u8")]
  if architecture == "arm" and "sme_f64" in features:
    atoms.append("SME_F64F64")
  rows = []
  for atom in atoms:
    for group, name, spatial, k, pack_a, pack_b in tables["RepresentativeCases" if "BF16" in atom else "SecondaryCases"]:
      for side, enabled in (("A", pack_a), ("B", pack_b)):
        if not enabled:
          continue
        native = atom_types(atom)[side == "B"]
        for metadata in ("Const", "Dynamic"):
          rows.append(_row(architecture, "MatmulPack", category="plain", group=group, name=name,
            atom=atom, native_a=atom_types(atom)[0], native_b=atom_types(atom)[1],
            spatial=spatial, k=k, side=side, metadata=metadata, memory=native, native=native,
            pipeline="identity", selector="Automatic", dsl_equivalent=True,
            source="benchmarks/ops/MatmulPackBenchCommon.h", isa=prefix))
  traits = {}
  for name, body in re.findall(r'struct FusedPipelineTraits<FusedPipeline::(\w+)> \{(.*?)\n\};', source, re.DOTALL):
    memory = DTYPES[re.search(r'using Memory = (\w+);', body)[1]]
    native = DTYPES[re.search(r'using T = (\w+);', body)[1]]
    transform = "transform = true" in body
    pipeline = "multiply4" if transform and native in {"int8", "uint8"} else "add1" if transform else "identity"
    if native in {"int8", "uint8"}:
      atom = f"{prefix}_I8I32_{'s8s8' if native == 'int8' else 'u8u8'}"
    else:
      atom = f"{prefix}_{ {'bfloat16':'BF16F32','float16':'F16F32','float32':'F32F32','float64':'F64F64'}[native]}"
    traits[name] = memory, native, pipeline, atom
  for pipeline_name, side, spatial, k in re.findall(
      r'register_fused_case<\s*FusedPipeline::(\w+),\s*::vecops::matmul::Operand::([AB]),\s*(\d+),\s*(\d+)\s*>', source):
    if pipeline_name == "FP32ToFP64" and "sme_f64" not in features:
      continue
    memory, native, pipeline, atom = traits[pipeline_name]
    selectors = ["Vector"]
    if architecture == "arm":
      selectors.append("SMEFP32ToFP64" if pipeline_name == "FP32ToFP64" else
                       "SMEStagedTransform" if pipeline != "identity" else
                       "SMEStagedFP16ToFP32" if pipeline_name == "FP16ToFP32" else "SMEPostprocess")
    for metadata in ("Const", "Dynamic"):
      for selector in selectors:
        rows.append(_row(architecture, "MatmulPack", category="fused", name=pipeline_name,
          atom=atom, native_a=atom_types(atom)[0], native_b=atom_types(atom)[1],
          spatial=int(spatial), k=int(k), side=side, metadata=metadata, memory=memory, native=native,
          pipeline=pipeline, selector=selector, dsl_equivalent=True,
          source="benchmarks/ops/MatmulPackBench.cpp", isa=prefix))
  if architecture == "x86":
    for input_kind, spatial, k in re.findall(r'register_all_compensation_modes<\s*CompensationInput::(\w+),\s*(\d+),\s*(\d+)\s*>', source):
      for metadata in ("Const", "Dynamic"):
        for mode in ("PackOnly", "SeparateSum", "Fused"):
          rows.append(_row(architecture, "MatmulPack", category="compensation", name=input_kind,
            atom="AMX_I8I32_u8s8", native_a="uint8", native_b="int8", spatial=int(spatial), k=int(k),
            side="B", metadata=metadata, memory="int8" if input_kind == "DirectS8" else "float32",
            native="int8", pipeline="identity" if input_kind == "DirectS8" else "multiply4",
            selector="Vector", original_dispatch="Automatic AMX packing resolves to Vector", compensation_mode=mode, zero_point=3, dsl_equivalent=True,
            dsl_schedule="pack_only" if mode=="PackOnly" else "pack_then_unsigned_modular_column_reduce",
            cpp_schedule="pack_only" if mode=="PackOnly" else "pack_then_scalar_column_sum" if mode=="SeparateSum" else "fused_pack_and_column_sum",
            comparison_status="harness_ready_native_validation_pending",
            source="benchmarks/ops/MatmulPackBench.cpp", isa=prefix))
  return rows


def _stateful(architecture):
  return [_row(architecture, "StatefulMatmul", shape=[m, n, k], name=mode, metadata="Dynamic",
    memory_a="bfloat16", memory_b="bfloat16", memory_c="float32", selector="Automatic",
    isa="AMX" if architecture == "x86" else "SME", dsl_equivalent=mode == "Legacy",
    comparison_status="unsupported_public_api" if mode == "Prepared" else "harness_ready_native_validation_pending",
    unsupported_reason="Prepared binds a persistent C++ operation plan outside timing; DSL workspace only reuses storage capacity and has no equivalent prepared-plan API" if mode == "Prepared" else None,
    source="benchmarks/ops/StatefulMatmulBench.cpp")
    for m, n, k in ((1,16,32),(16,16,32),(16,128,128),(64,64,128),(256,256,256)) for mode in ("Legacy", "Prepared")]


def _weight_reuse(architecture, features):
  prefix = "AMX" if architecture == "x86" else "SME"
  lifecycle = [(f"{prefix}_BF16F32", "bfloat16", "bfloat16", "float32", "Convert", "BiasRelu"),
               (f"{prefix}_BF16F32", "float32", "float32", "float32", "Convert", "Relu"),
               (f"{prefix}_I8I32_s8u8", "float32", "float32", "float32", "Quantize4", "Dequantize")]
  cases = [("decode",b,1,256,256,r) for b in (1,2,4,8,16) for r in (1,2,4,8,16)]
  cases += [("small",b,1,64,128,r) for b in (1,2,4,8,16) for r in (1,4,16)]
  cases += [("mlp_projection",8,1,1024,4096,r) for r in (1,2,4,8)]
  cases += [("qwen_o_proj",8,1,3584,3584,r) for r in (1,2,4)]
  rows = []
  def add(config, case, strategy, metadata, cohort):
    atom, ma, mb, mc, inp, out = config
    name, batch, m, n, k, reuse = case
    rows.append(_row(architecture, "MatmulWeightReuse", name=name, batch=batch, shape=[m,n,k],
      calls_per_pack=reuse, broadcast="B", strategy=strategy, metadata=metadata, atom=atom,
      memory_a=ma, memory_b=mb, memory_c=mc, memory_c_input="float32" if mc in {"bfloat16", "float16"} else mc,
      input_pipeline_a=inp, input_pipeline_b=inp if cohort=="lifecycle" else "Convert", output_pipeline=out,
      native_a=atom_types(atom)[0], native_b=atom_types(atom)[1], selector="Automatic", cohort=cohort,
      dsl_equivalent=True, comparison_status="harness_ready_native_validation_pending",
      lifecycle="one online B pack then calls_per_pack shared-B batched invocations" if strategy=="pack_once_then_reuse" else "calls_per_pack shared-B raw batched invocations",
      source="benchmarks/ops/MatmulWeightReuseBench.cpp", isa=prefix))
  for config in lifecycle:
    for case in cases:
      for strategy, metadata in (("raw","Dynamic"),("pack_once_then_reuse","Const")):
        add(config,case,strategy,metadata,"lifecycle")
  configs = [(f"{prefix}_BF16F32", "bfloat16", "bfloat16", "bfloat16", "Convert", "BiasRelu")]
  if architecture == "arm" or "amx_fp16" in features:
    configs.append((f"{prefix}_F16F32", "float16", "float16", "float16", "Convert", "BiasRelu"))
  if architecture == "arm":
    configs.append(("SME_F32F32", "float32", "float32", "float32", "Convert", "BiasRelu"))
    if "sme_f64" in features:
      configs.append(("SME_F64F64", "float64", "float64", "float64", "Convert", "BiasRelu"))
  for suffix in ("s8s8","s8u8","u8s8","u8u8"):
    atom=f"{prefix}_I8I32_{suffix}"
    ma,mb=atom_types(atom)
    configs.append((atom,ma,mb,"int32","Convert","Convert"))
  configs += [(f"{prefix}_I8I32_s8s8","float32","int8","float32","Quantize4","Dequantize"),
              (f"{prefix}_I8I32_u8s8","float32","int8","float32","AsymmetricQuantize4Zp3","AsymmetricDequantize")]
  probes = [("decode",1,1,256,256,1),("decode",1,1,256,256,16),
            ("decode",8,1,256,256,1),("decode",8,1,256,256,16),
            ("small",8,1,64,128,1),("small",8,1,64,128,16),
            ("mlp_projection",8,1,1024,4096,4),("qwen_o_proj",8,1,3584,3584,2)]
  for config in configs:
    for part,case in enumerate(probes):
      for strategy in ("raw","pack_once_then_reuse"):
        add(config,case,strategy,"Dynamic" if part%2==0 else "Const","dtype_probe")
  return rows


def supplemental_cases(architecture, features=(), root=ROOT):
  if architecture not in {"x86", "arm"}:
    raise ValueError("architecture must be x86 or arm")
  root = Path(root)
  features = frozenset(features)
  rows = _pack_cases(root,architecture,features)+_stateful(architecture)+_weight_reuse(architecture,features)
  if architecture == "arm":
    for native in ("bfloat16","int8","uint8"):
      for m,n in ((2,2),(2,4),(4,2),(2,8),(8,2),(4,4)):
        for k in (64,65,256,257,512,513,1024,1025):
          for metadata in ("Const","Dynamic"):
            for engine in ("mopa","mmla_current_pack"):
              rows.append(_row(architecture,"MatmulMMLAExperiment",shape=[m,n,k],native=native,
                metadata=metadata,engine=engine,selector="Automatic" if engine=="mopa" else "DirectSVE",
                build_option="VECOPS_ENABLE_SME_MMLA_EXPERIMENT_BENCHMARK",
                isa="SVE_BF16+SVE_I8MM+SME",input="packed_ab",
                dsl_equivalent=False,unsupported_reason="Optional experiment has no paired supplemental compute runner; engine:mopa calls public Automatic, not forced MOPA" if engine=="mopa" else "Direct-ISA experiment; existing DSL packed-dot leaf can express it, but requires a separately labeled experiment runner rather than public Automatic comparison",
                source="benchmarks/ops/MatmulMMLAExperimentBench.cpp"))
  return tuple(rows)


def manifest(architecture, features=(), root=ROOT):
  rows=supplemental_cases(architecture,features,root)
  paths={row['source'] for row in rows}|{"benchmarks/ops/MatmulScenarioBenchCommon.h"}
  return {"architecture":architecture,"features":list(features),"source_hashes":{
    path:hashlib.sha256((Path(root)/path).read_bytes()).hexdigest() for path in sorted(paths)},
    "counts":dict(Counter(row['suite'] for row in rows)),
    "equivalent_packing_count":sum(row['suite']=='MatmulPack' and row['dsl_equivalent'] for row in rows),
    "registrations":rows,"notes":["Counts are original registrations, not declarations, shard count, aggregates, or duplicate pipeline-only executables.",
    "WeightReuse lifecycle mode and metadata share shard parity: raw=Dynamic, pack_once=Const; dtype probe metadata is probe-part parity.",
    "A selector-labeled forced packing implementation is not a public Automatic Matmul comparison."]}


def main():
  parser=argparse.ArgumentParser(description=__doc__)
  parser.add_argument('--architecture',choices=('x86','arm'),required=True)
  parser.add_argument('--features',nargs='*',choices=('amx_fp16','sme_f64'),default=[])
  parser.add_argument('--output',type=Path,required=True)
  args=parser.parse_args()
  result=manifest(args.architecture,args.features)
  args.output.parent.mkdir(parents=True,exist_ok=True)
  args.output.write_text(json.dumps(result,indent=2)+'\n')
  print(result['counts'], 'equivalent packing',result['equivalent_packing_count'])


if __name__=='__main__':
  main()
