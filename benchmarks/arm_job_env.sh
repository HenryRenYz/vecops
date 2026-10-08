#!/usr/bin/env bash
# SPDX-FileCopyrightText: 2026 HenryRenYz and vecops contributors
# SPDX-License-Identifier: MIT
# Source this from an already initialized af3 shell inside an exclusive ARM job.
# The helper materializes only the pinned C++ headers under a private /tmp root;
# durable runtime DSOs are hash-checked and preloaded before future Python runs.

vecops_arm_job_env_init() {
  local task_script_dir task_helper task_probe task_python task_prefix task_input
  local task_clang task_clangxx task_bundle task_bundle_sha task_libgcc task_libgcc_sha
  local task_libstdcxx task_libstdcxx_sha task_envfile
  task_script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
  task_helper="${task_script_dir}/arm_job_env_helper.py"
  task_probe="${task_script_dir}/arm_toolchain_probe.cpp"
  task_python="$(command -v python || true)"
  task_prefix="${CONDA_PREFIX:-}"
  if [[ -z "${task_python}" || -z "${task_prefix}" || "$(basename -- "${task_prefix}")" != af3* ]]; then
    echo "Activate the existing af3 environment before sourcing arm_job_env.sh." >&2
    return 2
  fi
  if [[ "$(uname -m)" != aarch64 || -z "${CCS_JOB_ID:-}" || ! -f "${CCS_ALLOC_FILE:-}" ]]; then
    echo "Source this job environment inside an allocated ARM compute job, not on the login node." >&2
    return 2
  fi
  case "$(hostname -s)" in login*) echo "Refusing to configure toolchain on a login node." >&2; return 2 ;; esac

  task_clang="${VECOPS_PRIVATE_CLANG:-$(command -v clang || true)}"
  task_clangxx="${VECOPS_PRIVATE_CLANGXX:-$(command -v clang++ || true)}"
  task_bundle="${VECOPS_GCC14_BUNDLE:-}"
  task_bundle_sha="${VECOPS_GCC14_BUNDLE_SHA256:-}"
  task_libgcc="${VECOPS_DURABLE_LIBGCC:-}"
  task_libgcc_sha="${VECOPS_DURABLE_LIBGCC_SHA256:-}"
  task_libstdcxx="${VECOPS_DURABLE_LIBSTDCXX:-}"
  task_libstdcxx_sha="${VECOPS_DURABLE_LIBSTDCXX_SHA256:-}"
  if [[ -z "${task_clang}" || -z "${task_clangxx}" || -z "${task_bundle}" || -z "${task_bundle_sha}" \
      || -z "${task_libgcc}" || -z "${task_libgcc_sha}" || -z "${task_libstdcxx}" || -z "${task_libstdcxx_sha}" ]]; then
    cat >&2 <<'EOF'
Set these project/job-local variables before sourcing:
  VECOPS_GCC14_BUNDLE and VECOPS_GCC14_BUNDLE_SHA256
  VECOPS_DURABLE_LIBGCC and VECOPS_DURABLE_LIBGCC_SHA256
  VECOPS_DURABLE_LIBSTDCXX and VECOPS_DURABLE_LIBSTDCXX_SHA256
Optionally set VECOPS_PRIVATE_CLANG and VECOPS_PRIVATE_CLANGXX; otherwise active clang is used.
This does not modify af3, shell startup files, or global compiler configuration.
EOF
    return 2
  fi
  for task_input in "${task_helper}" "${task_script_dir}/arm_job_env.sh" "${task_probe}"; do
    if [[ ! -f "${task_input}" || -L "${task_input}" || ! -r "${task_input}" ]]; then
      echo "Missing, unreadable, or symlinked project toolchain input: ${task_input}" >&2
      return 2
    fi
  done
  task_envfile="$("${task_python}" "${task_helper}" prepare \
    --python "${task_python}" --conda-prefix "${task_prefix}" \
    --clang "${task_clang}" --clangxx "${task_clangxx}" \
    --bundle "${task_bundle}" --bundle-sha256 "${task_bundle_sha}" \
    --libgcc "${task_libgcc}" --libgcc-sha256 "${task_libgcc_sha}" \
    --libstdcxx "${task_libstdcxx}" --libstdcxx-sha256 "${task_libstdcxx_sha}" \
    --shell-path "${task_script_dir}/arm_job_env.sh" --probe-path "${task_probe}")" || return $?
  if [[ ! -f "${task_envfile}" || -L "${task_envfile}" ]]; then
    echo "Toolchain helper did not create a regular private job-env file." >&2
    return 1
  fi
  # LD_PRELOAD is established now, before this shell can import NumPy/Torch/vecops.
  # Future child Python processes inherit the same runtime identities.
  source "${task_envfile}"
  export VECOPS_ARM_JOB_ENV_MANIFEST="${VECOPS_ARM_PRIVATE_STDLIB_ROOT}/job-env-manifest.json"
  echo "[arm-job-env] active clang=${CXX}; private headers=${VECOPS_ARM_PRIVATE_GCC_INSTALL}" >&2
  echo "[arm-job-env] preloaded libgcc/libstdc++ from durable runtime paths before Python imports." >&2
}

vecops_arm_job_env_init "$@"
task_init_status=$?
unset -f vecops_arm_job_env_init
return "${task_init_status}" 2>/dev/null || exit "${task_init_status}"
