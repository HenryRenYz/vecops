#!/usr/bin/env bash
# Prepare a project-private GCC 14 standard library for the existing clang.
# Execute inside an exclusive lx-run job, never on the login node. No af3
# activation/configuration, bashrc, or other shared environment is modified.
set -euo pipefail
umask 077

task_script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
task_project_root="$(cd -- "${task_script_dir}/.." && pwd -P)"
task_prefix="${task_project_root}/build-toolchain-gcc14"
task_prefix_explicit=false
task_envfile=""
task_package_dir=""
task_scratch_root=""
while (($#)); do
  case "$1" in
    --prefix) task_prefix="${2:?--prefix needs a path}"; task_prefix_explicit=true; shift 2 ;;
    --envfile) task_envfile="${2:?--envfile needs a path}"; shift 2 ;;
    --offline-package-dir) task_package_dir="${2:?--offline-package-dir needs a path}"; shift 2 ;;
    --scratch-root) task_scratch_root="${2:?--scratch-root needs a path}"; shift 2 ;;
    *) echo "Unknown argument: $1" >&2; exit 2 ;;
  esac
done

if [[ "$(uname -m)" != aarch64 || -z "${CCS_JOB_ID:-}" || ! -f "${CCS_ALLOC_FILE:-}" ]]; then
  echo "Run arm_toolchain_setup.sh inside an exclusive ARM Donau/lx-run compute job." >&2
  exit 2
fi
case "$(hostname -s)" in
  login*) echo "Refusing toolchain installation on a login node." >&2; exit 2 ;;
esac
# The account's bashrc supplies its proxy/conda functions. Preserve nounset
# compatibility with existing interactive configuration; do not rewrite it.
task_job_id="${CCS_JOB_ID}"
task_allocation="${CCS_ALLOC_FILE}"
set +eu
source "${HOME}/.bashrc"
task_initialization_status=$?
set -eu
if ((task_initialization_status)); then
  echo "bashrc initialization failed; refusing to guess compiler/conda/proxy settings." >&2
  exit "${task_initialization_status}"
fi
export CCS_JOB_ID="${task_job_id}" CCS_ALLOC_FILE="${task_allocation}"
task_storage_base="${task_project_root}"
if [[ -n "${task_scratch_root}" ]]; then
  case "${task_scratch_root}" in
    /tmp/vecops-* | /dev/shm/vecops-*) ;;
    *) echo "Scratch storage must be an explicitly selected private /tmp/vecops-* or /dev/shm/vecops-* directory." >&2; exit 2 ;;
  esac
  if [[ "$(dirname -- "${task_scratch_root}")" != /tmp \
      && "$(dirname -- "${task_scratch_root}")" != /dev/shm ]]; then
    echo "Scratch root must be a directly named task directory, not a nested or broad directory." >&2
    exit 2
  fi
  if [[ ! -d "${task_scratch_root}" || -L "${task_scratch_root}" \
      || "$(realpath -- "${task_scratch_root}")" != "${task_scratch_root}" \
      || "$(stat -c %u -- "${task_scratch_root}")" != "$(id -u)" \
      || "$(stat -c %a -- "${task_scratch_root}")" != 700 ]]; then
    echo "Scratch root must be canonical, non-symlink, owned by this UID, and mode 0700 (use mktemp -d)." >&2
    exit 2
  fi
  task_storage_base="${task_scratch_root}"
  if [[ "${task_prefix_explicit}" == false ]]; then
    task_prefix="${task_storage_base}/build-toolchain-gcc14"
  fi
fi
task_prefix="$(realpath -m -- "${task_prefix}")"
case "${task_prefix}" in
  "${task_storage_base}"/build-toolchain-*) ;;
  *) echo "The private prefix must be a build-toolchain-* directory inside ${task_storage_base}." >&2; exit 2 ;;
esac
if [[ "${task_prefix}" == "${CONDA_PREFIX:-}" ]]; then
  echo "Refusing to change the active conda environment." >&2
  exit 2
fi
task_envfile="${task_envfile:-${task_prefix}/job-env.sh}"
task_envfile="$(realpath -m -- "${task_envfile}")"
case "${task_envfile}" in
  "${task_storage_base}"/build-*) ;;
  *) echo "The generated job-only envfile must remain in the selected private build directories." >&2; exit 2 ;;
esac
task_cc="$(command -v clang)"
task_cxx="$(command -v clang++)"
echo "[toolchain] clang=${task_cxx}; private prefix=${task_prefix}; job=${CCS_JOB_ID}"
"${task_cxx}" --version
if [[ -n "${https_proxy:-${HTTPS_PROXY:-}}" ]]; then
  echo "[toolchain] Existing bashrc HTTPS proxy is set (value intentionally not logged)."
else
  echo "[toolchain] No HTTPS proxy is set; cached packages may work, network failures will not alter proxy settings."
fi

task_conda_operation=create
[[ -f "${task_prefix}/conda-meta/history" ]] && task_conda_operation=install
task_conda_options=()
[[ "${task_conda_operation}" != create ]] || task_conda_options+=("--no-default-packages")
# Online solving also writes package/repodata caches. Keep every mode under
# the selected private storage, not conda's shared af3/default package cache.
export CONDA_PKGS_DIRS="${task_storage_base}/build-toolchain-gcc14-cache"
if [[ -n "${task_package_dir}" ]]; then
  # An offline bundle is prepared on the local host when the compute node
  # cannot reach the owner's login-node proxy. Never reconfigure that proxy.
  task_package_dir="$(realpath -m -- "${task_package_dir}")"
  case "${task_package_dir}" in
    "${task_project_root}"/build-toolchain-*) ;;
    *) echo "The offline bundle must remain in this project's build-toolchain-* directories." >&2; exit 2 ;;
  esac
  if [[ ! -f "${task_package_dir}/SHA256SUMS" ]]; then
    echo "The offline bundle needs a SHA256SUMS manifest." >&2
    exit 2
  fi
  (cd -- "${task_package_dir}" && sha256sum --check SHA256SUMS)
  task_packages=()
  while IFS= read -r -d '' task_package; do
    task_packages+=("${task_package}")
  done < <(find "${task_package_dir}" -maxdepth 1 -type f \
    \( -name '*.conda' -o -name '*.tar.bz2' \) -print0)
  if ((${#task_packages[@]} == 0)); then
    echo "The offline bundle contains no conda package archives." >&2
    exit 2
  fi
  # Explicit archives and --offline prevent solver metadata/network access.
  # Keep the extracted cache project-private too; do not fill af3's cache.
  conda "${task_conda_operation}" --yes --offline "${task_conda_options[@]}" \
    --prefix "${task_prefix}" "${task_packages[@]}"
elif ! conda "${task_conda_operation}" --yes --prefix "${task_prefix}" \
    "${task_conda_options[@]}" \
    --override-channels --channel conda-forge \
    'libstdcxx-devel_linux-aarch64=14' 'libgcc-devel_linux-aarch64=14' \
    'libstdcxx>=14' 'libgcc>=14'; then
  echo "[toolchain] conda-forge failed; checking whether defaults provides the requested GCC 14 implementation."
  if ! conda search --override-channels --channel defaults 'libstdcxx-devel_linux-aarch64=14'; then
    echo "[toolchain] No usable defaults fallback; retaining failed artifacts/logs without changing proxy or af3." >&2
    exit 1
  fi
  task_conda_operation=create
  [[ -f "${task_prefix}/conda-meta/history" ]] && task_conda_operation=install
  task_conda_options=()
  [[ "${task_conda_operation}" != create ]] || task_conda_options+=("--no-default-packages")
  conda "${task_conda_operation}" --yes --prefix "${task_prefix}" \
    "${task_conda_options[@]}" \
    --override-channels --channel defaults \
    'libstdcxx-devel_linux-aarch64=14' 'libgcc-devel_linux-aarch64=14' \
    'libstdcxx>=14' 'libgcc>=14'
fi

# Install only headers, compiler runtime/start files and shared runtimes: the
# GCC executable, binutils and a second glibc sysroot are unnecessary. Do not
# activate this prefix: clang remains CC/CXX and uses the host's C library.
task_gcc_install=""
while IFS= read -r task_candidate; do
  if [[ -f "${task_candidate}/libgcc.a" ]]; then
    task_gcc_install="${task_candidate}"
  fi
done < <(find "${task_prefix}/lib/gcc" -mindepth 2 -maxdepth 2 -type d | sort -V)
if [[ -z "${task_gcc_install}" || "$(basename -- "${task_gcc_install}")" != 14* ]]; then
  echo "Cannot find a real GCC 14 install with libgcc.a below ${task_prefix}/lib/gcc." >&2
  exit 1
fi
task_triple="$(basename -- "$(dirname -- "${task_gcc_install}")")"
task_cxx_include="${task_gcc_install}/include/c++"
if [[ ! -f "${task_cxx_include}/format" || ! -d "${task_cxx_include}/${task_triple}" ]]; then
  echo "Cannot find GCC 14 <format> and target-specific C++ headers." >&2
  exit 1
fi
# Conda keeps C++ headers under the GCC installation rather than the system
# distribution's include/c++/<version> layout. Select all three roots explicitly
# so a system GCC 10 bits/ header cannot accidentally enter the private build.
task_header_flags=("-nostdinc++" "-isystem" "${task_cxx_include}" \
  "-isystem" "${task_cxx_include}/${task_triple}" \
  "-isystem" "${task_cxx_include}/backward")
task_library_dirs=()
for task_dir in "${task_prefix}/lib" "${task_prefix}/${task_triple}/lib" \
    "${task_prefix}/${task_triple}/lib64" "${task_gcc_install}"; do
  if [[ -f "${task_dir}/libstdc++.so.6" || -f "${task_dir}/libgcc_s.so.1" ]]; then
    task_library_dirs+=("${task_dir}")
  fi
done
if ((${#task_library_dirs[@]} == 0)); then
  echo "The private installation has no loadable libstdc++/libgcc runtime directories." >&2
  exit 1
fi
task_ld_path="$(IFS=:; echo "${task_library_dirs[*]}")"
export LD_LIBRARY_PATH="${task_ld_path}${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
task_probe_dir="${task_prefix}/probe"
mkdir -p -- "${task_probe_dir}"
task_link_flags=()
for task_dir in "${task_library_dirs[@]}"; do
  task_link_flags+=("-L${task_dir}" "-Wl,-rpath,${task_dir}")
done
task_selected_flags=()
task_probe_passed=false
for task_selector in "--gcc-install-dir=${task_gcc_install}" "--gcc-toolchain=${task_prefix}"; do
  echo "[toolchain] Probing clang with ${task_selector} (syntax, link, execution)."
  task_flags=("${task_selector}" "${task_header_flags[@]}" "${task_link_flags[@]}")
  if "${task_cxx}" "${task_flags[@]}" -std=c++20 -pthread \
      -I"${task_project_root}/include" -fsyntax-only "${task_script_dir}/arm_toolchain_probe.cpp" \
      >"${task_probe_dir}/syntax.log" 2>&1 \
    && "${task_cxx}" "${task_flags[@]}" -std=c++20 -O2 -pthread \
      -I"${task_project_root}/include" "${task_script_dir}/arm_toolchain_probe.cpp" \
      -o "${task_probe_dir}/stdlib-probe" >"${task_probe_dir}/link.log" 2>&1 \
    && "${task_probe_dir}/stdlib-probe"; then
    task_selected_flags=("${task_flags[@]}")
    task_probe_passed=true
    break
  fi
  echo "[toolchain] Probe failed; full diagnostics follow."
  [[ ! -f "${task_probe_dir}/syntax.log" ]] || sed -n '1,$p' "${task_probe_dir}/syntax.log"
  [[ ! -f "${task_probe_dir}/link.log" ]] || sed -n '1,$p' "${task_probe_dir}/link.log"
done
if [[ "${task_probe_passed}" != true ]]; then
  echo "Private GCC 14 was installed but clang cannot use it correctly; no usable envfile produced." >&2
  exit 1
fi
printf -v task_private_cxxflags '%q ' "${task_selected_flags[@]}"
task_private_cxxflags="${task_private_cxxflags% }"
task_private_cxxflags="${CXXFLAGS:+${CXXFLAGS} }${task_private_cxxflags}"
mkdir -p -- "$(dirname -- "${task_envfile}")"
{
  echo '# Generated for this project only. Source in a scheduled job, never from global bashrc.'
  printf 'export CC=%q\n' "${task_cc}"
  printf 'export CXX=%q\n' "${task_cxx}"
  # af3 may retain an older GNU bridge override. Private GCC14 headers are
  # selected for this Clang toolchain, not for that unrelated GCC executable.
  printf 'export VECOPS_TORCH_BRIDGE_CC=%q\n' "${task_cc}"
  printf 'export VECOPS_TORCH_BRIDGE_CXX=%q\n' "${task_cxx}"
  printf 'export CXXFLAGS=%q\n' "${task_private_cxxflags}"
  printf 'export LD_LIBRARY_PATH=%q\n' "${LD_LIBRARY_PATH}"
} >"${task_envfile}"
echo "[toolchain] PASS: real GCC install=${task_gcc_install}"
echo "[toolchain] Job-only envfile=${task_envfile}; af3 and global configuration were not modified."
echo "[toolchain] Source this file before Python/CMake so private headers and runtime remain consistent."
