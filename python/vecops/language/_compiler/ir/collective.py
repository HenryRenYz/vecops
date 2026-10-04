"""Resident execution and scalar collectives, distinct from logical grids."""

from dataclasses import dataclass

from .types import Node


@dataclass(frozen=True)
class Team(Node):
  name: str
  workers: object
  body: object


@dataclass(frozen=True)
class Barrier(Node):
  team: str


@dataclass(frozen=True)
class TeamReduce(Node):
  team: str
  target: str
  value: object
  operation: str
