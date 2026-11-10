SPDX-FileCopyrightText: 2026 HenryRenYz and vecops contributors
SPDX-License-Identifier: MIT

Open index.html for the resource-model design, figures and embedded TOML.

Current implementation and authoritative interfaces:
- ../../include/vecops/memory/Model.h: C++ values and allocation/mapping contracts.
- ../../include/vecops/memory/ResourceModel.h: resource, placement and cost queries.
- ../../include/vecops/memory/LinuxRuntime.h: the Linux CPU runtime and options.
- ../Memory.md and ../Python.md: supported C++/Python usage and ownership.
- ../MemoryResourceMigration.md: implemented backend and calibration boundaries.

api/MemoryModel.h is a design-field excerpt, not a second normative header.
api/GraphPlanning.h is a C++ concept protocol; the production graph planner is
Python and uses the C++ runtime through its public API. GraphExamples.cpp is a
concept illustration, not an installed C++ planner example. The mapping is:
- BufferRequirement -> vecops.planner.BufferRequirement.
- GraphMemoryCompiler.compile -> compile(requirements).
- GraphMemoryCompiler.compile_model -> compile_model(programs, calls,
  serialized=..., scoped_requirements=...).
- Model preparation -> GraphMemoryCompiler.prepare_model(...), which returns an
  address-only PreparedProgram with pending capacity admission.
- Backing activation/release -> PreparedProgram.admit/materialize/close.
- Communication planning -> vecops.distributed.compiler.CommunicationCompiler,
  with separate candidates/estimate/select methods.
ModelExecutor, ModelExecutionInstance, LayoutProvider, DistributedLayoutLinker
and ReuseProof are conceptual responsibilities, not delivered C++ interfaces.

configs/*.toml describe portable hardware resources and reference rates.
Bandwidth is GiB/s; unknown latency/startup values are omitted. Groups and IDs
are arbitrary names, without reserved CPU hierarchy labels. HardwareTemplate
performs generic expansion; backend discovery binds physical objects. Linux CPU
allocation/mapping and optional SDMA execution are implemented. A800/H800
configs support topology/cost modelling; they do not supply a GPU runtime backend.

calibration/*.toml are historical reference data containing the broader design
models, not directly loadable production overlays. calibration/README.md lists
the strict supported subset and links an executable-schema example. Historical
measurements are not reclassified as measurements of the current executor.
policies/*.toml configure arena budgets and preallocation separately from hardware.
examples/offset-audit.txt is a pointer-free two-owner mapping illustration.
references/*.toml and manifest.json preserve historical inputs and source hashes.
The retired Python reference parsers/checkers are not production dependencies;
the former generator/rebuild commands are intentionally no longer advertised.
