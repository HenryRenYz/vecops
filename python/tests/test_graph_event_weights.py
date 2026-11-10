# SPDX-FileCopyrightText: 2026 HenryRenYz and vecops contributors
# SPDX-License-Identifier: MIT
"""Public sampled-loop weights preserve exact IDs, roles and nested counts."""
from dataclasses import replace
from fractions import Fraction
from types import SimpleNamespace

import pytest

from vecops import graph


def test_public_weights_read_live_capture_and_frozen_roundtrip_without_mutation():
  capture = graph.GraphCapture('opaque recurrence')
  with capture:
    outside = capture.record_marker('operation', 'outside')
    loop = capture.declare_loop('uma/cowos', 7)
    with loop.iteration(0, role='prologue'):
      first = capture.record_marker('operation', 'first')
    with loop.iteration(1, role='steady_sample'):
      steady = capture.record_marker('operation', 'steady')
    loop.close()
    live = graph.event_execution_weights(capture)
  execution = capture.finish()
  before = execution.dumps()
  weights = graph.event_execution_weights(execution)
  assert weights[first.id] == 1 and weights[steady.id] == 6
  assert weights[outside.id] == 1
  assert all(type(weight) is Fraction for weight in weights.values())
  assert all(weights[key] == value for key, value in live.items())
  assert graph.event_execution_weights(execution.freeze()) == weights
  assert graph.event_execution_weights(graph.ExecutionGraph.loads(before)) == weights
  assert execution.dumps() == before


def test_multiple_steady_samples_split_remaining_iterations_exactly():
  capture = graph.GraphCapture('multi-sample')
  observed = []
  with capture:
    loop = capture.declare_loop('sampled', 9)
    for index, role in enumerate(('prologue', 'steady_sample', 'steady_sample', 'epilogue')):
      with loop.iteration(index, role=role):
        observed.append(capture.record_marker('operation', role))
    loop.close()
  weights = graph.event_execution_weights(capture.finish())
  assert [weights[event.id] for event in observed] == [1, Fraction(7, 2), Fraction(7, 2), 1]
  assert sum(weights[event.id] for event in observed) == 9


def test_nested_sample_products_represent_all_thirty_five_logical_calls():
  capture = graph.GraphCapture('nested')
  observed = []
  with capture:
    outer = capture.declare_loop('same opaque name', 7)
    for i in range(2):
      with outer.iteration(i, role='prologue' if i == 0 else 'steady_sample'):
        # Identical names do not merge independent loop occurrences.
        inner = capture.declare_loop('same opaque name', 5)
        for j in range(2):
          with inner.iteration(j, role='prologue' if j == 0 else 'steady_sample'):
            observed.append(capture.record_marker('operation', f'{i}:{j}'))
        inner.close()
    outer.close()
  weights = graph.event_execution_weights(capture.finish())
  assert [weights[event.id] for event in observed] == [1, 4, 6, 24]
  assert sum(weights[event.id] for event in observed) == 35


def test_symbolic_and_overobserved_loops_do_not_invent_execution_counts():
  capture = graph.GraphCapture('unknown')
  observed = []
  with capture:
    for count in ('runtime count', 0, 1):
      loop = capture.declare_loop('loop', count)
      for index in range(2):
        # The legacy unlabelled first sample is an explicit cold iteration.
        with loop.iteration(index):
          observed.append(capture.record_marker('operation', str(count)))
      loop.close()
  weights = graph.event_execution_weights(capture.finish())
  assert all(weights[event.id] == 1 for event in observed)


def test_sparse_event_indexes_and_planner_adapter_use_same_id_weights():
  from vecops.planner.compiler import _event_weights
  capture = graph.GraphCapture('sparse')
  with capture:
    loop = capture.declare_loop('loop', 8)
    for index in range(3):
      with loop.iteration(index):
        capture.record_marker('operation', str(index))
    loop.close()
  execution = capture.finish()
  duck = SimpleNamespace(events=[replace(event, index=event.index * 17) for event in execution.events],
                         loops=execution.loops)
  weights = graph.event_execution_weights(duck)
  assert weights == graph.event_execution_weights(execution)
  assert _event_weights(duck) == {event.index: weights[event.id] for event in duck.events}


def test_open_loop_cannot_publish_a_partial_execution_weight_summary():
  capture = graph.GraphCapture('open')
  with capture:
    loop = capture.declare_loop('loop', 7)
    with loop.iteration(0):
      capture.record_marker('operation', 'first')
    with pytest.raises(ValueError, match='closed loops'):
      graph.event_execution_weights(capture)
    loop.close()


@pytest.mark.parametrize('failure', ['unknown', 'reversed'])
def test_invalid_sample_event_range_is_rejected(failure):
  capture = graph.GraphCapture('invalid')
  with capture:
    loop = capture.declare_loop('loop', 7)
    with loop.iteration(0):
      capture.record_marker('operation', 'first')
    loop.close()
  execution = capture.finish()
  sample = execution.loops[0].samples[0]
  bad = replace(sample, end_event='missing') if failure == 'unknown' else replace(
      sample, begin_event=sample.end_event, end_event=sample.begin_event)
  duck = SimpleNamespace(events=execution.events, loops=[replace(execution.loops[0], samples=[bad])])
  with pytest.raises(ValueError, match='sample event ranges'):
    graph.event_execution_weights(duck)
