<!-- SPDX-FileCopyrightText: 2026 HenryRenYz and vecops contributors -->
<!-- SPDX-License-Identifier: MIT -->
# Calibration data and executable overlays

`920f-lx.toml`, `a800.toml` and `h800.toml` retain historical measurements and the
broader design model, including compute/triad/fallback fields. They are reference
data, not files to pass unchanged to the production runtime. They do not qualify
the current executor as calibrated.

The production `memory-calibration/0.4` parser accepts access `read`/`write`
curves, copy transfer curves or explicit `operation="copy"` cases, and resource
`write_cost.copy`. Nonempty `[methods]`, triad, `extra_write`, unknown fields and
unbound profiles are rejected rather than ignored. Use only supported fields in
an application overlay; loading hardware does not require calibration.

See [the executable-schema example and control/provenance rules](../../MemoryResourceMigration.md#optional-calibration-overlays)
and [the Python runtime interface](../../Python.md). CPU core counts and opaque
implementation controls must describe the actual executor. Missing or mismatched
controls and unsampled ranges retain reference estimates; a historical value is
not promoted to calibrated quality by moving it into a new file.
