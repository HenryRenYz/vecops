# SPDX-FileCopyrightText: 2026 HenryRenYz and vecops contributors
# SPDX-License-Identifier: MIT
"""Exact execution multiplicities for closed, statically sampled graph loops."""
from bisect import bisect_left, bisect_right
from fractions import Fraction


def event_execution_weights(graph) -> dict[str, Fraction]:
  """Return sampled-loop execution weights keyed by stable event ID.

  ``graph`` may be an ExecutionGraph, immutable snapshot, or an active
  GraphCapture whose loops have already closed. Only its events and loops are
  read. Prologue/epilogue samples and the first observed sample have weight one;
  the remaining declared iterations are distributed equally among other
  samples, with a minimum weight of one. Nested sample factors multiply.

  Symbolic/nonpositive trip counts and loops without samples contribute no
  expansion: their observed events retain weight one. This is a static traffic
  summary, not an executable loop or a proof that sampled iterations have
  identical effects. Fractions preserve exact multiplicities until the caller
  chooses its rounding policy. Open loops or invalid event ranges are errors.
  """
  events = tuple(graph.events)
  indexes = {event.id: event.index for event in events}
  if len(indexes) != len(events) or len(set(indexes.values())) != len(events):
    raise ValueError('execution weights require unique event IDs and indexes')
  ordered = sorted((event.index, event.id) for event in events)
  positions = [index for index, _identity in ordered]
  weights = dict.fromkeys(indexes, Fraction(1))
  for loop in graph.loops:
    if loop.end_event is None:
      raise ValueError('execution weights require closed loops')
    if loop.end_event not in indexes:
      raise ValueError('execution weights reference an unknown loop end event')
    samples = tuple(loop.samples)
    ranges = []
    for sample in samples:
      if sample.begin_event not in indexes or sample.end_event not in indexes:
        raise ValueError('execution weights require closed, known sample event ranges')
      begin, end = indexes[sample.begin_event], indexes[sample.end_event]
      if end < begin:
        raise ValueError('execution weights require ordered sample event ranges')
      ranges.append((bisect_left(positions, begin), bisect_right(positions, end)))
    if not isinstance(loop.trip_count, int) or loop.trip_count <= 0 or not samples:
      continue
    explicit = sum(sample.role in {'prologue', 'epilogue'} or index == 0
                   for index, sample in enumerate(samples))
    steady = len(samples) - explicit
    factor = max(Fraction(1), Fraction(max(0, loop.trip_count - explicit), max(1, steady)))
    for index, (sample, (begin, end)) in enumerate(zip(samples, ranges)):
      if index == 0 or sample.role in {'prologue', 'epilogue'}:
        continue
      for _position, identity in ordered[begin:end]:
        weights[identity] *= factor
  return weights
