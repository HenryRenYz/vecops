#!/usr/bin/env bash
# Rebuild, run, merge, and summarize one x86 + 920f-4 benchmark iteration.

set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source_root="$(cd "${script_dir}/../../.." && pwd)"
run_id="${1:-$(date +%Y%m%dT%H%M%S)}"
filter="${VECOPS_BENCH_FILTER:-.*}"
x86_filter="${VECOPS_X86_FILTER:-${filter}}"
arm_vecops_filter="${VECOPS_ARM_VECOPS_FILTER:-${filter}}"
# Full coverage is the default. A reduced peer filter is diagnostic-only and
# must be requested explicitly; otherwise it creates misleading vecops-only
# shapes in the completed matrix.
arm_peer_filter="${VECOPS_ARM_PEER_FILTER:-${filter}}"
x86_build="${VECOPS_X86_BUILD:-${source_root}/cmake-build-matmul-otherlibs-x86}"
remote_host="${VECOPS_ARM_HOST:-920f-4}"
remote_root="${VECOPS_ARM_ROOT:-vecops-neo-matmul-otherlibs}"
remote_build="${VECOPS_ARM_BUILD:-cmake-build-matmul-otherlibs-arm}"
x86_cpu="${VECOPS_X86_CPU:-4}"
arm_cpu="${VECOPS_ARM_CPU:-4}"
disable_aslr="${VECOPS_DISABLE_ASLR:-0}"
output_root="${VECOPS_CYCLE_OUTPUT:-${script_dir}/results/raw/cycles/${run_id}}"
baseline="${VECOPS_BENCH_BASELINE:-${script_dir}/results/benchmark-results-current-measured-20260904.csv}"

vecops_targets=(
  MatmulOtherlibsVecopsGeneralBench-Native
  MatmulOtherlibsVecopsLLMBench-Native
  MatmulOtherlibsVecopsAF3ProjectionBench-Native
  MatmulOtherlibsVecopsAF3AttentionQKLargeBench-Native
  MatmulOtherlibsVecopsAF3AttentionShallowBench-Native
  MatmulOtherlibsVecopsAF3AttentionPVBench-Native
  MatmulOtherlibsVecopsAF3TriangleBench-Native
  MatmulOtherlibsVecopsBiasBench-Native
  MatmulOtherlibsVecopsBiasReluBench-Native
  MatmulOtherlibsVecopsBiasSiluBench-Native
)
x86_targets=(
  "${vecops_targets[@]}"
  MatmulOtherlibsOpenBLASBench-Native
  MatmulOtherlibsOneDNNBench-Native
  MatmulOtherlibsLIBXSMMBench-Native
)
arm_targets=(
  "${vecops_targets[@]}"
  MatmulOtherlibsOpenBLASBench-Native
  MatmulOtherlibsOneDNNBench-Native
  MatmulOtherlibsLIBXSMMBench-Native
  MatmulOtherlibsACLBench-Native
  MatmulOtherlibsKUPLBench-Native
)

mkdir -p "${output_root}/x86" "${output_root}/arm" "${output_root}/analysis"
x86_csv="${output_root}/x86/local-x86_${run_id}.csv"
arm_csv="${output_root}/arm/920f-4_${run_id}.csv"
aslr_args=()
if [[ "${disable_aslr}" == "1" ]]; then
  aslr_args+=(--disable-aslr)
fi

if [[ -f "${x86_csv}" ]]; then
  echo "[x86] reusing completed ${x86_csv}"
else
  echo "[x86] building ${x86_build}"
  cmake --build "${x86_build}" --target "${x86_targets[@]}" \
    -j "${VECOPS_BUILD_JOBS:-4}"
  python3 "${script_dir}/run_suite.py" \
    --build-dir "${x86_build}" --output-dir "${output_root}/x86" \
    --host local-x86 --cpu "${x86_cpu}" --run-id "${run_id}" \
    --filter "${x86_filter}" --providers vecops openblas onednn libxsmm \
    "${aslr_args[@]}" \
    --compiler "gcc" --backend-isa "AVX-512+AMX-BF16"
fi

if [[ -f "${arm_csv}" ]]; then
  echo "[arm] reusing completed ${arm_csv}"
else
  echo "[arm] syncing ${source_root} to ${remote_host}:~/${remote_root}"
  rsync -az \
    --exclude='cmake-build-*' --exclude='build/' --exclude='.cache/' \
    --exclude='__pycache__/' --exclude='.pytest_cache/' \
    "${source_root}/" "${remote_host}:~/${remote_root}/"

  # OpenSSH joins its command arguments into one remote-shell string. Build
  # that string with %q so regex metacharacters remain one positional arg.
  printf -v remote_command 'bash -s -- %q %q %q %q %q %q %q' \
    "${remote_root}" "${remote_build}" "${arm_cpu}" "${run_id}" \
    "${arm_vecops_filter}" "${arm_peer_filter}" "${disable_aslr}"
  for target in "${arm_targets[@]}"; do
    printf -v quoted_target '%q' "${target}"
    remote_command+=" ${quoted_target}"
  done
  ssh "${remote_host}" "${remote_command}" <<'EOF'
source ~/.bashrc
set -euo pipefail
remote_root="$1"; shift
remote_build="$1"; shift
arm_cpu="$1"; shift
run_id="$1"; shift
vecops_filter="$1"; shift
peer_filter="$1"; shift
disable_aslr="$1"; shift
aslr_args=()
if [[ "${disable_aslr}" == "1" ]]; then
  aslr_args+=(--disable-aslr)
fi
cd "${HOME}/${remote_root}"
libomp_path="$(clang++ --print-file-name=libomp.so)"
if [[ "${libomp_path}" == */* ]]; then
  export LD_LIBRARY_PATH="$(dirname "${libomp_path}"):${LD_LIBRARY_PATH:-}"
fi
cmake -S . -B "${remote_build}" \
  -DCMAKE_C_COMPILER="$(which clang)" \
  -DCMAKE_CXX_COMPILER="$(which clang++)"
cmake --build "${remote_build}" --target "$@" -j "${VECOPS_BUILD_JOBS:-2}"
python3 benchmarks/ops/matmul-otherlibs/run_suite.py \
  --build-dir "${remote_build}" \
  --output-dir "benchmarks/ops/matmul-otherlibs/results/raw/cycles/${run_id}/arm-vecops" \
  --host 920f-4 --cpu "${arm_cpu}" --run-id "${run_id}-vecops" \
  --filter "${vecops_filter}" --providers vecops --resume \
  "${aslr_args[@]}" \
  --compiler "BiSheng clang" --backend-isa "SVE/SVE2/SME; VL=SVL=512b"
python3 benchmarks/ops/matmul-otherlibs/run_suite.py \
  --build-dir "${remote_build}" \
  --output-dir "benchmarks/ops/matmul-otherlibs/results/raw/cycles/${run_id}/arm-peers" \
  --host 920f-4 --cpu "${arm_cpu}" --run-id "${run_id}-peers" \
  --filter "${peer_filter}" \
  --providers openblas onednn libxsmm acl kupl --resume \
  "${aslr_args[@]}" \
  --compiler "BiSheng clang" --backend-isa "SVE/SVE2/SME; VL=SVL=512b"
python3 benchmarks/ops/matmul-otherlibs/combine_csv.py \
  "benchmarks/ops/matmul-otherlibs/results/raw/cycles/${run_id}/arm-vecops/920f-4_${run_id}-vecops.csv" \
  "benchmarks/ops/matmul-otherlibs/results/raw/cycles/${run_id}/arm-peers/920f-4_${run_id}-peers.csv" \
  --output "benchmarks/ops/matmul-otherlibs/results/raw/cycles/${run_id}/arm/920f-4_${run_id}.csv"
EOF

  rsync -az \
    "${remote_host}:~/${remote_root}/benchmarks/ops/matmul-otherlibs/results/raw/cycles/${run_id}/arm/" \
    "${output_root}/arm/"
fi

combined="${output_root}/combined.csv"
python3 "${script_dir}/combine_csv.py" "${x86_csv}" "${arm_csv}" \
  --output "${combined}"
complete="${output_root}/benchmark-results-complete.csv"
python3 "${script_dir}/complete_results.py" "${combined}" \
  --catalog "${combined}" --output "${complete}"
python3 "${script_dir}/validate_results.py" "${complete}"
python3 "${script_dir}/combine_csv.py" "${complete}" \
  --common-cases-across-hosts \
  --output "${output_root}/cross-arch-common.csv"
python3 "${script_dir}/summarize_vecops.py" "${complete}" \
  --output "${output_root}/vecops-performance-summary.csv"
analysis_args=(
  --current "${complete}"
  --output-dir "${output_root}/analysis"
)
if [[ -f "${baseline}" ]]; then
  # The current complete run already contains every peer. The old snapshot is
  # regression-only; importing its prepared rows as fallback peers would mix
  # measurement phases in the new comparison tables.
  analysis_args+=(--baseline "${baseline}")
fi
python3 "${script_dir}/analyze_cycle.py" "${analysis_args[@]}"

echo "cycle results: ${output_root}"
