#!/usr/bin/env bash
# SPDX-FileCopyrightText: 2026 HenryRenYz and vecops contributors
# SPDX-License-Identifier: MIT
set -euo pipefail

# Install only the Clang 19 UBSan static runtimes into this project's private
# Clang resource directory, then verify normal execution and overflow trapping.
# No compiler install, environment, or system directory is modified.

project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P)"
toolchain_root="$project_root/build-toolchain-compiler-rt19"
package="$toolchain_root/compiler-rt_linux-aarch64-19.1.7-hfefdfc9_1.conda"
resource_dir="$toolchain_root/clang19-resource"
probe_source="$project_root/benchmarks/arm_sanitizer_ubsan_probe.cpp"
expected_package_sha="bdd4a6072775543febaba398f8e40d776793a259ea4ca3961475c34ae187a631"
original_resource_dir=""
stage_dir=""
work_dir=""

fail() {
  printf 'arm-sanitizer-setup: %s\n' "$*" >&2
  exit 1
}

cleanup() {
  if [[ -n "$stage_dir" && -d "$stage_dir" ]]; then
    rm -rf -- "$stage_dir"
  fi
  if [[ -n "$work_dir" && -d "$work_dir" ]]; then
    rm -rf -- "$work_dir"
  fi
}
trap cleanup EXIT

[[ "$(uname -m)" == "aarch64" ]] || fail "must run on an AArch64 compute node"
[[ -f "$package" ]] || fail "compiler-rt package missing: $package"
[[ -f "$probe_source" ]] || fail "probe source missing: $probe_source"
[[ -d "$toolchain_root" && ! -L "$toolchain_root" ]] || fail "toolchain root must be a real project directory"

clang="${CC:-$(command -v clang || true)}"
clangxx="${CXX:-$(command -v clang++ || true)}"
[[ -n "$clang" && -x "$clang" ]] || fail "set CC to the intended clang executable"
[[ -n "$clangxx" && -x "$clangxx" ]] || fail "set CXX to the intended clang++ executable"
[[ "$(basename "$clang")" == clang ]] || fail "CC must explicitly name clang: $clang"
[[ "$(basename "$clangxx")" == clang++ ]] || fail "CXX must explicitly name clang++: $clangxx"

clang_version="$($clang --version | head -n 1)"
clang_target="$($clang -print-target-triple)"
[[ "$clang_version" == *"clang version 19.1.7"* ]] || fail "expected Clang 19.1.7, got: $clang_version"
[[ "$clang_target" == aarch64-unknown-linux-gnu ]] || fail "unexpected compiler target: $clang_target"
original_resource_dir="$($clang --print-resource-dir)"
[[ -d "$original_resource_dir/include" ]] || fail "Clang builtin headers missing: $original_resource_dir/include"

actual_package_sha="$(sha256sum "$package" | awk '{print $1}')"
[[ "$actual_package_sha" == "$expected_package_sha" ]] || fail "compiler-rt package SHA mismatch: $actual_package_sha"

if [[ -e "$resource_dir" || -L "$resource_dir" ]]; then
  [[ -d "$resource_dir" && ! -L "$resource_dir" ]] || fail "private resource path is not a real directory"
  [[ -L "$resource_dir/include" && "$(readlink -f "$resource_dir/include")" == "$(readlink -f "$original_resource_dir/include")" ]] \
    || fail "existing private resource has unexpected include target"
  [[ -f "$resource_dir/install-manifest.txt" ]] || fail "existing private resource has no install manifest"
  grep -Fx "package_sha256=$expected_package_sha" "$resource_dir/install-manifest.txt" >/dev/null \
    || fail "existing private resource was installed from a different package"
  grep -Fx "original_resource_dir=$original_resource_dir" "$resource_dir/install-manifest.txt" >/dev/null \
    || fail "existing private resource has a different original resource directory"
  [[ "$(sha256sum "$resource_dir/lib/linux/libclang_rt.ubsan_standalone-aarch64.a" | awk '{print $1}')" == \
    "8ae940a0fa2557d2dec7a462a9a3d4bfef7f040a256c3ddc9f3f4c096951f7f9" ]] \
    || fail "existing standalone archive hash differs"
  [[ "$(sha256sum "$resource_dir/lib/linux/libclang_rt.ubsan_standalone_cxx-aarch64.a" | awk '{print $1}')" == \
    "138f96702bd554da5ede243424461357d038dec94d7410d1c91bbe191f5fcd92" ]] \
    || fail "existing C++ archive hash differs"
  for archive in "$resource_dir"/lib/linux/libclang_rt.ubsan_standalone*.a; do
    ar t "$archive" >/dev/null || fail "invalid installed archive: $archive"
  done
  printf 'reusing_verified_private_resource_dir: %s\n' "$resource_dir"
else
  stage_dir="$(mktemp -d "$toolchain_root/.clang19-resource.XXXXXX")"
  mkdir -p "$stage_dir/lib/linux"
  ln -s "$original_resource_dir/include" "$stage_dir/include"

  python3 - "$package" "$stage_dir/lib/linux" <<'PY'
import io
import os
import sys
import tarfile
import zipfile

import zstandard

package, destination = sys.argv[1:]
expected = {
    "lib/clang/19/lib/linux/libclang_rt.ubsan_standalone-aarch64.a": 6924182,
    "lib/clang/19/lib/linux/libclang_rt.ubsan_standalone_cxx-aarch64.a": 151282,
}
found = set()
with zipfile.ZipFile(package) as conda:
    members = [name for name in conda.namelist() if name.startswith("pkg-") and name.endswith(".tar.zst")]
    if len(members) != 1:
        raise SystemExit(f"expected one conda payload, found {members!r}")
    payload = io.BytesIO(conda.read(members[0]))
    reader = zstandard.ZstdDecompressor().stream_reader(payload)
    with tarfile.open(fileobj=reader, mode="r|") as archive:
        for member in archive:
            if member.name not in expected:
                continue
            if not member.isfile() or member.size != expected[member.name]:
                raise SystemExit(f"unexpected runtime archive metadata: {member.name} {member.size}")
            source = archive.extractfile(member)
            if source is None:
                raise SystemExit(f"unable to read {member.name}")
            target = os.path.join(destination, os.path.basename(member.name))
            with open(target, "xb") as output:
                output.write(source.read())
            os.chmod(target, 0o644)
            found.add(member.name)
if found != set(expected):
    raise SystemExit(f"runtime archives missing: {sorted(set(expected) - found)}")
PY

  for archive in \
    "$stage_dir/lib/linux/libclang_rt.ubsan_standalone-aarch64.a" \
    "$stage_dir/lib/linux/libclang_rt.ubsan_standalone_cxx-aarch64.a"; do
    ar t "$archive" >/dev/null || fail "invalid static archive: $archive"
  done

{
  printf 'package=%s\n' "$package"
  printf 'package_sha256=%s\n' "$actual_package_sha"
  printf 'clang=%s\n' "$clang"
  printf 'clang_version=%s\n' "$clang_version"
  printf 'target=%s\n' "$clang_target"
  printf 'original_resource_dir=%s\n' "$original_resource_dir"
  printf 'private_resource_dir=%s\n' "$resource_dir"
  printf 'ubsan_standalone_aarch64_sha256=%s\n' "$(sha256sum "$stage_dir/lib/linux/libclang_rt.ubsan_standalone-aarch64.a" | awk '{print $1}')"
  printf 'ubsan_standalone_cxx_aarch64_sha256=%s\n' "$(sha256sum "$stage_dir/lib/linux/libclang_rt.ubsan_standalone_cxx-aarch64.a" | awk '{print $1}')"
} > "$stage_dir/install-manifest.txt"

  # Same-filesystem, no-replace publication. Never write under the Clang install.
  mv -T -- "$stage_dir" "$resource_dir" || fail "could not publish private resource directory"
  stage_dir=""
fi

work_root="${TMPDIR:-$toolchain_root}"
work_dir="$(mktemp -d "$work_root/clang19-ubsan-probe.XXXXXX")"
probe="$work_dir/ubsan_probe"
plan="$work_dir/clang-link-plan.txt"
compile_log="$work_dir/compile.log"
overflow_log="$work_dir/overflow.log"
ulimit -c 0

printf 'clang: %s\n%s\ntarget: %s\n' "$clang" "$clang_version" "$clang_target"
printf 'original_resource_dir: %s\nprivate_resource_dir: %s\n' "$original_resource_dir" "$resource_dir"
printf 'package_sha256: %s\n' "$actual_package_sha"
grep -vE '^[0-9a-f]{64}  ' "$resource_dir/install-manifest.txt"
printf 'ubsan_standalone_aarch64_sha256=%s\n' "$(sha256sum "$resource_dir/lib/linux/libclang_rt.ubsan_standalone-aarch64.a" | awk '{print $1}')"
printf 'ubsan_standalone_cxx_aarch64_sha256=%s\n' "$(sha256sum "$resource_dir/lib/linux/libclang_rt.ubsan_standalone_cxx-aarch64.a" | awk '{print $1}')"

"$clangxx" --target="$clang_target" -resource-dir="$resource_dir" \
  -nostdinc++ -fsanitize=undefined -fno-sanitize-recover=undefined \
  -### "$probe_source" -o "$probe" 2> "$plan"
grep -F "$resource_dir/lib/linux/libclang_rt.ubsan_standalone-aarch64.a" "$plan" >/dev/null \
  || fail "Clang driver did not select the private UBSan runtime archive"
grep -F "$resource_dir/lib/linux/libclang_rt.ubsan_standalone_cxx-aarch64.a" "$plan" >/dev/null \
  || fail "Clang++ driver did not select the private C++ UBSan runtime archive"

printf 'link_plan_sha256: %s\n' "$(sha256sum "$plan" | awk '{print $1}')"
printf 'link_plan_runtime_paths:\n'
grep -oE "$resource_dir/lib/linux/libclang_rt[^ ]+" "$plan" | sort -u

"$clangxx" --target="$clang_target" -resource-dir="$resource_dir" \
  -nostdinc++ -fsanitize=undefined -fno-sanitize-recover=undefined \
  -O1 -g "$probe_source" -o "$probe" 2> "$compile_log" \
  || { cat "$compile_log" >&2; fail "UBSan probe failed to link with clang++"; }

"$probe"
set +e
"$probe" --overflow > "$overflow_log" 2>&1
overflow_status=$?
set -e
[[ "$overflow_status" -ne 0 ]] || fail "signed-overflow probe unexpectedly returned success"
grep -i 'signed integer overflow' "$overflow_log" >/dev/null \
  || { cat "$overflow_log" >&2; fail "signed-overflow diagnostic was not emitted"; }
printf 'expected_overflow_status: %s\n' "$overflow_status"
grep -i 'signed integer overflow' "$overflow_log"
printf 'probe_source_sha256: %s\n' "$(sha256sum "$probe_source" | awk '{print $1}')"
printf 'probe_executable_sha256: %s\n' "$(sha256sum "$probe" | awk '{print $1}')"
printf 'UBSAN_CLANG_RESOURCE_DIR=%s\n' "$resource_dir"
