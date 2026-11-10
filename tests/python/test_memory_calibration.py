# SPDX-FileCopyrightText: 2026 HenryRenYz and vecops contributors
# SPDX-License-Identifier: MIT
"""Strict optional calibration through the public runtime factory, without payload allocation."""

import importlib
import os

import pytest


@pytest.fixture(scope="module")
def native():
    name = os.environ.get("VECOPS_MEMORY_TEST_MODULE", "vecops._C.memory")
    return importlib.import_module(name)


def _hardware(native):
    groups = len(native.discover_linux_runtime().physical_graph.agents)
    assert groups
    return f"""schema='memory-model/0.3'
name='public runtime calibration test'
[units]
bandwidth='GiB/s'
[groups]
coherent_region={{count='runtime'}}
[agents]
compute={{per='coherent_region',count={groups},bind='cpu-group',capabilities=['host_load_store']}}
[pools]
storage={{per='compute',capacity='discover',bind='local-ddr',service='controller'}}
[resources]
controller={{per='compute',read=1000,write=1000,sharing='read_write'}}
[paths]
read_path={{agent='compute',pool='storage',where='any',bandwidth={{read=80,write=80}},mapping='direct'}}
"""


OVERLAY = """schema='memory-calibration/0.4'
name='synthetic contract test; not hardware qualification'
[units]
bandwidth='GiB/s'
[models]
'access:read_path'='reference_stream'
[profiles.reference_stream]
read={cpu_cores=[16],bandwidth=[40]}
provenance={quality='reference',source='synthetic regression',implementation='test-workload'}
"""


def _open(native, tmp_path, *, controls):
    hardware = _hardware(native)
    hardware_path, calibration_path = tmp_path / "hardware.toml", tmp_path / "calibration.toml"
    hardware_path.write_text(hardware)
    calibration_path.write_text(OVERLAY)
    options = native.LinuxRuntimeOptions()
    if controls:
        expanded = native.expand_hardware_template(hardware, {"coherent_region": 1})
        actual = native.CalibrationControl()
        actual.cpu_cores = 16
        actual.implementation = "test-workload"
        options.calibration_controls = {a.id: actual for a in expanded.graph.agents}
    return native.open_runtime(str(hardware_path), calibration_file=str(calibration_path), options=options)


def _query(native, runtime):
    view = runtime.view(runtime.current_scope())
    query = native.AccessQuery()
    query.agent = view.executable_agents[0]
    segment = native.PlacedSegment()
    segment.offset_bytes = 0
    segment.size_bytes = 4096
    segment.pool = view.allocatable_pools[0]
    endpoint = native.TransferEndpoint()
    endpoint.address_space = next(p.address_space for p in view.graph.pools if p.id == segment.pool)
    endpoint.segments = [segment]
    query.memory = endpoint
    query.access = native.AccessMode.read
    query.payload_bytes = 4096
    context = native.CostContext()
    context.view_epoch = view.epoch
    context.execution_slots = {query.agent: 16}  # authorization is not the curve's actual control
    return view, query, context


def test_explicit_controls_and_detached_performance_database(native, tmp_path):
    runtime = _open(native, tmp_path, controls=True)
    view, query, context = _query(native, runtime)
    cost = runtime.costs.estimate_access(view, query, context)
    assert cost.total_us == pytest.approx(4096 * 1e6 / (40 * 2**30))
    assert cost.quality == native.CostQuality.reference
    assert not cost.missing_facts
    database = runtime.performance()
    assert database["bindings"]
    assert database["profiles"]["reference_stream"]["read"]["cpu_cores"] == [16]
    database["profiles"]["reference_stream"]["read"]["bandwidth"][0] = 1
    assert runtime.performance()["profiles"]["reference_stream"]["read"]["bandwidth"] == [40]
    assert runtime.physical_graph.access[0].bandwidth["read"] == 80


def test_slots_do_not_supply_missing_implementation_or_thread_control(native, tmp_path):
    runtime = _open(native, tmp_path, controls=False)
    view, query, context = _query(native, runtime)
    assert runtime.costs.estimate_access(view, query, context).total_us == pytest.approx(4096 * 1e6 / (80 * 2**30))


def test_unsupported_semantics_fail_instead_of_partial_overlay(native, tmp_path):
    hardware = tmp_path / "hardware.toml"
    calibration = tmp_path / "calibration.toml"
    hardware.write_text(_hardware(native))
    calibration.write_text(OVERLAY + "\n[methods]\nread={profile='reference_stream',extra_write='256 KiB'}\n")
    with pytest.raises(RuntimeError, match="calibration.unsupported"):
        native.open_runtime(str(hardware), calibration_file=str(calibration))


def test_sdma_shared_channel_is_explicit_optional_control(native):
    options = native.LinuxSdmaOptions()
    assert options.shared_channel == 0
    options.shared_channel = None
    assert options.shared_channel is None
    options.shared_channel = 7
    assert options.shared_channel == 7
    with pytest.raises((TypeError, OverflowError)):
        options.shared_channel = -1
