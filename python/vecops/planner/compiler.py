# SPDX-FileCopyrightText: 2026 HenryRenYz and vecops contributors
# SPDX-License-Identifier: MIT
"""Resource-graph memory compilation with integer offsets and deferred backing.

Graph policy lives here; topology, placement patterns, mapping validation and
costs come exclusively from the public C++ memory model. Tensor temporaries and
kernel workspaces use the same BufferRequirement. No hardware hierarchy or
fixed number of memory tiers is part of this compiler.
"""
from __future__ import annotations

from collections import Counter, defaultdict
from bisect import bisect_left
from dataclasses import dataclass, field, replace
from hashlib import sha256
import importlib
import itertools
import json
import math
import re
from types import MappingProxyType
import threading
import time
from typing import Any, Mapping, Sequence


class _ColdPhysicalHeadroom:
    """Primitive, prepare-local observations; no native snapshot/error ownership.

    New typed loss requires the complete, same-epoch accounting fingerprint.
    Proved margins survive unrelated changes, but not changed accounting in
    their pool/overlapping-constraint component. Incomplete components also
    depend on epoch. Reserve proofs retain their stricter complete accounting.
    """
    def __init__(self):
        self.accounting = None
        self.losses = {}
        self._components = {}

    @staticmethod
    def fingerprint(snapshot):
        # Physical samples/pool allocatable values are deliberately not part of
        # this identity: their oscillation is the specific retry observation.
        groups={g.capacity_group:g for g in snapshot.groups}
        return (snapshot.epoch,
            tuple(sorted((g.capacity_group,g.hard_limit_bytes,g.committed_bytes,
                g.live_bytes,g.idle_available_bytes,g.idle_reserved_bytes,
                g.retired_bytes,g.unbacked_reserved_bytes,g.largest_cached_block_bytes,
                g.shared_remaining_bytes) for g in snapshot.groups)),
            tuple(sorted((p.pool,p.capacity_group,p.largest_cached_block_bytes,
                tuple(p.shared_constraints),
                ('group_physical_bound',) if p.allocatable_bytes ==
                    groups[p.capacity_group].idle_available_bytes +
                    _ColdPhysicalHeadroom.fresh(groups[p.capacity_group])
                else ('independent_pool_bound',p.allocatable_bytes))
                for p in snapshot.pools)),
            tuple(sorted((c.id,tuple(c.capacity_groups),c.remaining_bytes)
                for c in snapshot.constraints)))

    @staticmethod
    def fresh(group):
        # system_available_bytes is already net of pending. Pending appears
        # only in the independent hard-limit test, not another physical debit.
        hard=max(0,group.hard_limit_bytes-group.committed_bytes-group.unbacked_reserved_bytes)
        if group.shared_remaining_bytes is not None:
            hard=min(hard,group.shared_remaining_bytes)
        return min(hard,group.system_available_bytes)

    @staticmethod
    def untruncated(snapshot, identity):
        groups={g.capacity_group:g for g in snapshot.groups}
        pools=[p for p in snapshot.pools if p.capacity_group==identity]
        if identity not in groups or not pools:
            return False
        group=groups[identity]
        available=group.idle_available_bytes+_ColdPhysicalHeadroom.fresh(group)
        return all(p.allocatable_bytes==available for p in pools)

    @staticmethod
    def _component_keys(signature):
        """Primitive dependency closures; unknown members keep epoch protection."""
        epoch,groups,pools,constraints=signature
        group_rows={row[0]:row for row in groups}
        constraint_rows={row[0]:row for row in constraints}
        links={name:set() for name in group_rows}
        missing_constraints={}
        for _identity,members,_remaining in constraints:
            for name in members:
                links.setdefault(name,set()).update(members)
        for _pool,group,_cached,identities,_bound in pools:
            for identity in identities:
                if identity not in constraint_rows:
                    missing_constraints.setdefault(group,set()).add(identity)
                    continue
                members=constraint_rows[identity][1]
                for name in (group,)+members:
                    links.setdefault(name,set()).update((group,)+members)
        result={}
        for name in group_rows:
            if name in result:
                continue
            members,pending=set(),[name]
            while pending:
                member=pending.pop()
                if member not in members:
                    members.add(member)
                    pending.extend(links[member]-members)
            selected_pools=tuple(row for row in pools if row[1] in members)
            referenced={identity for row in selected_pools for identity in row[3]}
            selected_constraints=tuple(row for row in constraints
                if row[0] in referenced or members.intersection(row[1]))
            absent_groups=tuple(sorted(members-group_rows.keys()))
            absent_constraints=tuple(sorted({identity for member in members
                for identity in missing_constraints.get(member,())}))
            key=(tuple(group_rows[g] for g in sorted(members) if g in group_rows),
                 selected_pools,selected_constraints,absent_groups,absent_constraints,
                 epoch if absent_groups or absent_constraints else None)
            result.update((g,key) for g in members if g in group_rows)
        return result

    def _sync(self, snapshot, signature=None):
        signature=self.fingerprint(snapshot) if signature is None else signature
        components=self._component_keys(signature)
        for identity in tuple(self.losses):
            if identity not in components or self._components.get(identity)!=components[identity]:
                del self.losses[identity]
        self._components=components
        # Keep the full fingerprint for reserve proof/diagnostic compatibility;
        # only retention of an already proved loss uses a narrower component.
        self.accounting=signature

    def observe(self, shortfalls, planned, current):
        old,new=self.fingerprint(planned),self.fingerprint(current)
        self._sync(current,new)
        # Endpoint equality within a component cannot rule out peer-grant ABA.
        # Learning a new loss therefore still requires the full original key.
        if old != new:
            return False
        previous=dict(self.losses)
        groups={g.capacity_group:g for g in planned.groups}
        physical_rejection=False
        for item in shortfalls:
            if (type(item)is not tuple or len(item)!=4
                    or item[0]!='capacity_group' or type(item[1])is not str
                    or any(type(n)is not int or not 0<=n<1<<64 for n in item[2:])):
                continue
            _,identity,requested,available=item
            if identity not in groups or not available < requested:
                continue
            if not self.untruncated(planned,identity) or not self.untruncated(current,identity):
                continue
            # Excludes cache-shape rejection at unchanged fresh allowance and
            # every pool/constraint-only failure. Never infer a reserve error.
            if available >= self.fresh(groups[identity]):
                continue
            physical_rejection=True
            loss=self.fresh(groups[identity])-available
            self.losses[identity]=max(self.losses.get(identity,0),loss)
        if physical_rejection:
            # One slow compile can observe physical loss in several groups even
            # when only its currently fullest group produced a typed shortfall.
            # Same accounting excludes peer pending; only unconstrained physical
            # allowances qualify, and idle bytes never enter the learned loss.
            current_groups={g.capacity_group:g for g in current.groups}
            for identity,group in groups.items():
                if not self.untruncated(planned,identity) or not self.untruncated(current,identity):
                    continue
                loss=self.fresh(group)-self.fresh(current_groups[identity])
                if loss>0:
                    self.losses[identity]=max(self.losses.get(identity,0),loss)
        return self.losses != previous

    def observe_reserve(self, error, planned, current):
        """Learn only a known rejected reserve with restored complete accounting.

        Reserve validates the last snapshot before taking its local grant, and
        the shared physical-free rejection occurs before a shared grant exists.
        Local-grant rollback may advance epoch while restoring every balance.
        No unrelated changed quota, cache, peer balance or constraint is waived.
        """
        fingerprint, identity, requested, available, allowances = error.evidence
        before, after = self.fingerprint(planned), self.fingerprint(current)
        self._sync(current,after)
        if (before[1:] != fingerprint[1:] or fingerprint[1:] != after[1:]
                or not before[0] <= fingerprint[0] < after[0]):
            return False
        fresh = {name: (count, untruncated) for name,count,untruncated in allowances}
        if identity not in fresh:
            return False
        _, untruncated = fresh[identity]
        planned_groups = {g.capacity_group:g for g in planned.groups}
        count = self.fresh(planned_groups[identity])
        if (not untruncated or not self.untruncated(planned, identity)
                or not self.untruncated(current, identity)
                or not 0 < requested <= count or not available < requested):
            return False
        previous = dict(self.losses)
        # The strict three-snapshot proof above is unchanged. Synchronization
        # preserves only unrelated, previously proved component margins.
        self.losses[identity] = max(self.losses.get(identity, 0), count-available)
        return self.losses != previous

    def apply(self, snapshot, groups):
        self.unadjusted_groups=dict(groups)
        self.adjusted_groups_id=id(groups)
        self._sync(snapshot)
        for identity,loss in self.losses.items():
            if identity in groups:
                total,idle=groups[identity]
                groups[identity]=(idle+max(0,total-idle-loss),idle)


class _PlanningCapacityError(MemoryError):
    """A known placement/estimate capacity rejection, before backing acquisition.

    Shortfalls retain only scalar limits, never native owners or traceback frames.
    """
    def __init__(self, message, *, shortfalls=()):
        super().__init__(message)
        self.shortfalls = tuple(shortfalls)




def _physical_free_rejection(message):
    """Recognize the native authority's known unaccepted sampled-free rejection."""
    return re.fullmatch(
        r'shared_capacity\.physical_free: physical free sample minus pending reservations is exhausted: '
        r'group=.+, requested_bytes=[0-9]+, sampled_free_bytes=[0-9]+, pending_bytes=[0-9]+',
        message, re.DOTALL) is not None


class _ReservePhysicalCapacityError(RuntimeError):
    """An unaccepted native reserve rejection with primitive last-snapshot evidence."""
    def __init__(self, message, evidence):
        super().__init__(message)
        self.evidence = evidence


def _reserve_physical_evidence(message, snapshot):
    match = re.fullmatch(
        r'shared_capacity\.physical_free: physical free sample minus pending reservations is exhausted: '
        r'group=(.+), requested_bytes=([0-9]+), sampled_free_bytes=([0-9]+), pending_bytes=([0-9]+)',
        message, re.DOTALL)
    if match is None:
        return None
    identity = match[1]
    try:
        requested, sample, pending = (int(n) for n in match.groups()[1:])
    except ValueError:
        return None
    if not 0 < requested < 1 << 64 or not 0 <= sample < 1 << 64 or not 0 <= pending < 1 << 64:
        return None
    available = max(0, sample-pending)  # sample is raw here; public snapshot is already net.
    if available >= requested:
        return None
    try:
        fingerprint = _ColdPhysicalHeadroom.fingerprint(snapshot)
        allowances = tuple((g.capacity_group, _ColdPhysicalHeadroom.fresh(g),
            _ColdPhysicalHeadroom.untruncated(snapshot, g.capacity_group)) for g in snapshot.groups)
    except (AttributeError, TypeError, ValueError, OverflowError):
        return None  # Unsupported provider shape: preserve its original error.
    return fingerprint, identity, requested, available, allowances


def reserve_recipe_latest(runtime, view, recipe, caches=()):
    """Reserve against a fresh snapshot, retrying only rejected stale epochs.

    Other processes can change a shared capacity authority between snapshot and
    reserve. This admission policy retries that race at most 64 times; it never
    retries capacity, permission, recipe-validation or backing errors. The
    native explicit-epoch reserve contract remains unchanged.

    A known unaccepted shared physical-free rejection carries primitive facts
    from the last actual reserve snapshot. Model preparation may use that
    evidence for a later plan; this helper does not repeat a capacity failure.
    """
    caches = list(caches)
    stale_errors = {
        'invalid_argument: stale budget snapshot',
        'shared_capacity.stale: stale shared capacity snapshot',
    }
    for attempt in range(64):
        snapshot = runtime.snapshot(view, caches)
        rejected = None
        try:
            return runtime.reserve(view, recipe, snapshot.epoch, caches)
        except RuntimeError as error:
            message = str(error)
            evidence = _reserve_physical_evidence(message, snapshot)
            if evidence is not None:
                rejected = (message, evidence)
            else:
                if message not in stale_errors or attempt == 63:
                    raise
                time.sleep(min(attempt + 1, 8) * 0.0001)
        if rejected is not None:
            # Leave the handler first: no original exception/native snapshot is
            # attached to the evidence or retained as an exception context.
            del snapshot
            raise _ReservePhysicalCapacityError(*rejected)


def reserve_budget_latest(runtime, view, byte_count, pool_order, *, native_module=None, attempts=64):
    """Reserve pending framework capacity across an explicit ordered pool domain.

    No backing or virtual address is acquired, and the recipe does not control
    actual malloc placement. Each fresh snapshot is packed greedily in caller
    order, charging whole pool pages, shared groups only once, and every
    overlapping constraint. Cached managed pages are never malloc headroom.
    A conservative packing failure does not prove that no other packing exists.
    Only known, unaccepted stale/capacity rejections permit a bounded fresh try;
    permission, validation and uncertain-commit failures propagate unchanged.
    Return the owning lease and the exact (possibly page-rounded) recipe.
    """
    _integer(byte_count, 'byte_count')
    if type(attempts) is not int or not 1 <= attempts <= 64:
        raise ValueError('attempts must be in [1,64]')
    if isinstance(pool_order, (str, bytes)):
        raise ValueError('pool_order must contain explicit pool IDs')
    pool_order = tuple(pool_order)
    if (not pool_order or any(not isinstance(p, str) or not p for p in pool_order)
            or len(set(pool_order)) != len(pool_order)):
        raise ValueError('pool_order must contain distinct nonempty pool IDs')
    pools = {pool.id: pool for pool in view.graph.pools}
    for name in pool_order:
        if name not in pools or name not in view.allocatable_pools:
            raise ValueError('budget pool is outside the allocatable resource view: ' + name)
        _integer(pools[name].capacity_bytes, 'pool capacity')
    participant = runtime.participant
    if not isinstance(participant, str) or not participant:
        raise ValueError('runtime participant must be a nonempty identity')
    m = native_module or importlib.import_module('vecops.memory').native
    rejected = {
        'invalid_argument: stale budget snapshot',
        'shared_capacity.stale: stale shared capacity snapshot',
        'invalid_argument: pool capacity exhausted',
        'invalid_argument: hard capacity limit exceeded',
        'invalid_argument: physical free capacity exceeded',
        'invalid_argument: shared domain capacity limit exceeded',
        'shared_capacity.exhausted: shared group hard capacity exhausted',
        'shared_capacity.exhausted: overlapping shared capacity constraint exhausted',
        'shared_capacity.physical_free: physical free sample minus pending reservations is exhausted',
    }
    for attempt in range(attempts):
        snapshot = runtime.snapshot(view, [])
        groups = {group.capacity_group: group for group in snapshot.groups}
        available = {pool.pool: pool for pool in snapshot.pools}
        constraints, membership = {}, defaultdict(list)
        for cut in snapshot.constraints:
            if not cut.id or cut.id in constraints or not cut.capacity_groups or len(set(cut.capacity_groups)) != len(cut.capacity_groups):
                raise ValueError('invalid budget constraint inventory')
            constraints[cut.id] = _integer(cut.remaining_bytes, 'constraint remaining')
            for group in cut.capacity_groups:
                membership[group].append(cut.id)
        remaining, group_free, backings, charged = byte_count, {}, [], 0
        for name in pool_order:
            pool = pools[name]
            if pool.capacity_group not in groups or name not in available:
                raise ValueError('budget snapshot omits a selected pool/group')
            budget, allowance = groups[pool.capacity_group], available[name]
            if (allowance.capacity_group != pool.capacity_group
                    or set(allowance.shared_constraints) != set(membership[pool.capacity_group])):
                raise ValueError('budget snapshot pool/constraint identity mismatch')
            if not pool.page_sizes_bytes:
                raise ValueError('budget pool has no concrete page quantum')
            quanta = [pool.granularity_bytes or 1, *pool.page_sizes_bytes]
            if any(_integer(q, 'page quantum', positive=True) & (q - 1) for q in quanta):
                raise ValueError('budget page quantum must be a power of two')
            quantum = max(quanta)
            if pool.capacity_group not in group_free:
                # system_available already deducts outstanding pending grants.
                # Unlike budget_available(), never add managed idle backing.
                hard = _integer(budget.hard_limit_bytes, 'group hard limit')
                used = _integer(_integer(budget.committed_bytes, 'group committed')
                    + _integer(budget.unbacked_reserved_bytes, 'group pending'), 'group accepted charge')
                cap = min(max(0, hard - used), _integer(budget.system_available_bytes, 'group physical free'))
                if budget.shared_remaining_bytes is not None:
                    cap = min(cap, _integer(budget.shared_remaining_bytes, 'shared remaining'))
                group_free[pool.capacity_group] = cap
            cap = min(group_free[pool.capacity_group], pool.capacity_bytes,
                      _integer(allowance.allocatable_bytes, 'pool allocatable'),
                      *(constraints[key] for key in membership[pool.capacity_group]))
            count = min(cap // quantum * quantum, _align(remaining, quantum))
            if not count:
                continue
            charged = _integer(charged + count, 'rounded budget charge')
            backings.append(_object(m, 'BackingRequest', id='pending-budget:' + str(len(backings)),
                owner=participant, pool=name, bytes=count, page_bytes=quantum,
                mode=m.BackingMode.private_memory, export_to=[]))
            group_free[pool.capacity_group] -= count
            for key in membership[pool.capacity_group]:
                constraints[key] -= count
            remaining = max(0, remaining - count)
            if not remaining:
                break
        if remaining:
            raise MemoryError('ordered pending budget does not fit selected pools/constraints: '
                              f'requested={byte_count}, unplaced={remaining}')
        recipe = _object(m, 'MemoryRecipe', participant=participant, layout_generation=1,
            backings=backings, addresses=_object(m, 'AddressLayout', regions=[], slots=[]), mappings=[])
        try:
            return runtime.reserve(view, recipe, snapshot.epoch, []), recipe
        except RuntimeError as error:
            message = str(error)
            if (message not in rejected and not _physical_free_rejection(message)) or attempt + 1 == attempts:
                raise
            time.sleep(min(attempt + 1, 8) * 0.0001)


def _object(module, name, **fields):
    result = getattr(module, name)()
    for key, value in fields.items():
        setattr(result, key, value)
    return result


def _align(value: int, quantum: int) -> int:
    return (value + quantum - 1) // quantum * quantum


def _integer(value, name, *, positive=False):
    if isinstance(value, bool) or not isinstance(value, int) or value < int(positive) or value >= 1 << 64:
        raise ValueError(f"{name} must be an exact {'positive ' if positive else ''}uint64 integer")
    return value


def _overlap(left, right):
    return any(a < d and c < b for a, b in left for c, d in right)


def _normal_lifetimes(lifetimes):
    """Union adjacent/overlapping intervals without changing their live set."""
    result = []
    for begin, end in sorted(lifetimes):
        if result and begin <= result[-1][1]:
            result[-1] = (result[-1][0], max(end, result[-1][1]))
        else:
            result.append((begin, end))
    return tuple(result)


def _plain_enum(value):
    return getattr(value, 'value', value)


def _sharing_identity(contract):
    return None if contract is None else (contract.id, tuple(contract.participants),
        tuple(contract.producers), tuple(contract.consumers), contract.protocol)


class _LifetimeIndex:
    """Treap of live intervals; query work follows overlaps, not graph length."""
    __slots__=('begin','end','offset','size','order','priority','maximum','left','right')
    _orders=itertools.count(1)

    def __init__(self,begin,end,offset,size):
        self.begin,self.end,self.offset,self.size=begin,end,offset,size
        self.order=next(self._orders)
        # Deterministic mixed priorities avoid an ordered-event linked list.
        mixed=(self.order+0x9e3779b97f4a7c15)&((1<<64)-1)
        mixed=((mixed^(mixed>>30))*0xbf58476d1ce4e5b9)&((1<<64)-1)
        mixed=((mixed^(mixed>>27))*0x94d049bb133111eb)&((1<<64)-1)
        self.priority=mixed^(mixed>>31)
        self.maximum=end;self.left=self.right=None

    def update(self):
        self.maximum=max(self.end,self.left.maximum if self.left else 0,
                         self.right.maximum if self.right else 0)
        return self

    @staticmethod
    def insert(root,node):
        if root is None:return node
        if (node.begin,node.order)<(root.begin,root.order):
            root.left=_LifetimeIndex.insert(root.left,node)
            if root.left.priority<root.priority:
                child=root.left;root.left=child.right;child.right=root.update();root=child
        else:
            root.right=_LifetimeIndex.insert(root.right,node)
            if root.right.priority<root.priority:
                child=root.right;root.right=child.left;child.left=root.update();root=child
        return root.update()

    @staticmethod
    def overlaps(root,begin,end,result):
        if root is None or root.maximum<=begin:return
        _LifetimeIndex.overlaps(root.left,begin,end,result)
        if root.begin<end:
            if root.end>begin:result.add((root.offset,root.offset+root.size))
            _LifetimeIndex.overlaps(root.right,begin,end,result)


class _RangeUnion:
    """Disjoint VA ranges; retained lifetimes need one spatial union, not N visits."""
    def __init__(self):
        self.ranges=[]

    def add(self, low, high):
        left=bisect_left(self.ranges,(low,))
        if left and self.ranges[left-1][1]>=low:
            left-=1
        right=left
        while right<len(self.ranges) and self.ranges[right][0]<=high:
            low=min(low,self.ranges[right][0])
            high=max(high,self.ranges[right][1])
            right+=1
        self.ranges[left:right]=[(low,high)]


@dataclass(frozen=True)
class BufferRequirement:
    """One storage identity, including every aliased tensor or workspace view.

    Lifetimes are half-open. External borrowed buffers have no allocated slot.
    Runtime-owned outputs and explicitly redirected state do have slots; their
    lifetime must include all consumers beyond the producing subgraph.
    independent_mapping/exported ranges receive exclusive page envelopes.
    """
    id: str
    bytes: int
    alignment: int = 64  # minimum absolute buffer-address alignment, not just an offset
    lifetimes: tuple[tuple[int, int], ...] = ((0, 1),)
    allowed_pools: tuple[str, ...] = ()
    consumers: tuple[str, ...] = ()
    read_bytes: int = 0
    write_bytes: int = 0
    placement: Any = None  # native PlacementPlan; optional exact user policy
    boundary: str = "internal"
    external_role: str | None = None  # input/output/state
    ownership: str = "runtime_owned"
    may_redirect: bool = False
    read_only: bool = False
    independent_mapping: bool = False
    sharing_contract: Any = None  # native SharingContract
    export_to: tuple[str, ...] = ()
    backing: Any = None  # native BackingMode; None selects a supported local mode
    retained: bool = False  # provider state survives calls until program destruction
    traffic_assumed: bool = False  # one-pass payload fallback, never a hardware measurement

    @property
    def allocated(self):
        return self.boundary == 'internal' or self.ownership == 'runtime_owned' or self.may_redirect

    def validate(self):
        if not isinstance(self.id, str) or not self.id:
            raise ValueError("buffer identity cannot be empty")
        if not isinstance(self.retained, bool) or not isinstance(self.traffic_assumed,bool):
            raise ValueError('retained and traffic_assumed must be boolean')
        _integer(self.bytes, 'buffer bytes', positive=True)
        _integer(self.alignment, 'buffer alignment', positive=True)
        if self.alignment & (self.alignment - 1):
            raise ValueError('buffer alignment must be a power of two')
        if self.boundary not in {'internal', 'external'}:
            raise ValueError('buffer boundary must be internal or external')
        if self.ownership not in {'runtime_owned', 'borrowed'}:
            raise ValueError('unknown buffer ownership')
        if self.boundary == 'external' and self.external_role not in {'input', 'output', 'state'}:
            raise ValueError('external buffer requires input/output/state role')
        if self.boundary == 'internal' and self.external_role is not None:
            raise ValueError('internal buffer cannot have an external role')
        if self.may_redirect and (self.boundary != 'external' or self.external_role != 'state'):
            raise ValueError('only explicit state buffers may be redirected')
        if self.may_redirect and not self.read_only:
            raise ValueError('mutable state redirection needs a commit contract, which was not provided')
        if self.may_redirect and self.placement is None:
            raise ValueError('state redirection requires an explicit placement until its migration is costed')
        if not self.lifetimes:
            raise ValueError('buffer needs at least one lifetime')
        for begin, end in self.lifetimes:
            _integer(begin, 'lifetime begin')
            _integer(end, 'lifetime end', positive=True)
            if begin >= end:
                raise ValueError('lifetimes must be nonempty half-open intervals')
        for field_name in ('allowed_pools', 'consumers', 'export_to'):
            values = getattr(self, field_name)
            if len(set(values)) != len(values):
                raise ValueError(f'duplicate {field_name}')
        _integer(self.read_bytes, 'read traffic')
        _integer(self.write_bytes, 'write traffic')
        if self.read_only and self.write_bytes:
            raise ValueError('read-only buffer has write accesses')


@dataclass(frozen=True)
class BufferBinding:
    buffer: str
    slot: str | None
    external: str | None
    view_offset_bytes: int
    size_bytes: int
    alignment: int
    independent_mapping: bool
    placement_key: str | None


@dataclass(frozen=True)
class InitializationCopy:
    external: str
    buffer: str
    bytes: int


@dataclass(frozen=True)
class ProgramCall:
    id: str
    program: str
    bindings: Mapping[str, str] = field(default_factory=dict)
    repeat_count: int = 1


def call_time_domains(programs, calls, *, serialized):
    """Return the compiler's offset-only time domains for explicit call order.

    Sequential identical-binding repetitions occupy one liveness domain, with
    external/carry ranges conservatively live throughout that domain. Without
    an ordering contract every call occupies the same conservative domain.
    Frontends may use these domains to describe explicitly activated native
    leases; the compiler does not infer when a lease initializes or stops.
    """
    if type(serialized) is not bool:
        raise ValueError('serialized must be an explicit boolean execution contract')
    if len({call.id for call in calls}) != len(calls):
        raise ValueError('model call identities must be unique')
    result={}
    cursor=0
    for call in calls:
        _integer(call.repeat_count,'repeat count',positive=True)
        if call.repeat_count>10000:
            raise ValueError('expand repeated model calls within the explicit planning work budget')
        duration=max((end for requirement in programs[call.program]
                      for _begin,end in requirement.lifetimes),default=1)
        _integer(duration,'call time domain',positive=True)
        end=cursor+duration if serialized else 1
        _integer(end,'model time domain',positive=True)
        result[call.id]=(cursor,end)
        if serialized:
            cursor=end
    return MappingProxyType(result)


@dataclass(frozen=True)
class CompiledMemoryProgram:
    """Immutable compiler product; memory returns a fresh native recipe copy."""
    _memory_toml: str
    bindings: Mapping[str, BufferBinding]
    requirements: tuple[BufferRequirement, ...]
    initializations: tuple[InitializationCopy, ...]
    peak_bytes_by_capacity_group: Mapping[str, int]
    resource_signature: str
    budget_epoch: int
    used_conservative_fallback: bool
    _native: Any = field(repr=False, compare=False)
    call_bindings: Mapping[str, Mapping[str, str]] = field(default_factory=dict)
    call_sequence: tuple[str, ...] = ()

    @property
    def memory(self):
        return self._native.recipe_from_toml(self._memory_toml)

    def dumps(self):
        return self._memory_toml

    def prepare(self, runtime, view, *, caches=()):
        return PreparedProgram(self, getattr(runtime, 'native', runtime), view, tuple(caches))


class GraphMemoryCompiler:
    """Greedy placement with exact slab packing/admission for every candidate.

    Unknown hardware costs remain unknown. If no measured candidate exists,
    explicit pool order supplies a deterministic conservative placement.
    Capacities are checked by unique capacity group and all shared constraints.
    """
    def __init__(self, queries, view, budgets, *, native_module=None, participant=None, caches=(),
                 preferred_alignment_bytes=None):
        if preferred_alignment_bytes is not None:
            _integer(preferred_alignment_bytes, 'preferred alignment', positive=True)
            if preferred_alignment_bytes & (preferred_alignment_bytes - 1):
                raise ValueError('preferred alignment must be a power of two')
        self.preferred_alignment_bytes = preferred_alignment_bytes
        self.queries = getattr(queries, 'native', queries)
        self.view, self.budgets = view, budgets
        self.native = native_module or importlib.import_module('vecops.memory').native
        self.participant = participant or getattr(self.queries, 'participant', None)
        if not self.participant:
            raise ValueError('a recipe participant must be explicit')
        self.caches = tuple(caches)
        self.costs = getattr(self.queries, 'costs', self.queries)
        self.pools = {pool.id: pool for pool in view.graph.pools}
        self.pool_order = tuple(pool.id for pool in view.graph.pools if pool.id in view.allocatable_pools)
        self._reset_queries()

    def _reset_queries(self):
        # Only exact cold queries are reused, and only within one compile.
        # Mutable admission/free capacity is always checked against the ledger.
        self._placement_cache={}
        self._capability_cache={}
        self._access_cache={}
        self._score_cache={}
        self._endpoint_keys={}
        self._endpoints={}
        self._score_context=None
        self._executable_agents=frozenset(self.view.executable_agents)
        self._pool_members=frozenset(self.pool_order)
        self._score_context_key=(self.view.epoch,
            tuple(sorted(self.view.execution_slots.items())),False)
        # Snapshot properties cross the binding boundary. Freeze their scalar
        # values once per compile attempt; prepare_model refreshes self.budgets
        # before recompiling after a rejected/stale admission.
        groups={item.capacity_group:item for item in self.budgets.groups}
        self._budget_groups={key:(self.native.budget_available(item),item.idle_available_bytes)
                             for key,item in groups.items()}
        self._budget_pools={item.pool:item.allocatable_bytes for item in self.budgets.pools}
        self._budget_constraints=tuple((tuple(item.capacity_groups),item.remaining_bytes)
                                       for item in self.budgets.constraints)
        policy=getattr(self,'_cold_physical_headroom',None)
        if policy is not None:
            policy.apply(self.budgets,self._budget_groups)
        self.query_counts=Counter()

    def _placement(self, plan, size, mode=None):
        mode = mode or self.native.BackingMode.shareable_remappable
        key=(plan.address_space,plan.stripe_bytes,tuple(sorted(plan.weights.items())),
             tuple(plan.cycle),plan.phase,size,str(mode))
        if key in self._placement_cache:
            self.query_counts['placement_hits']+=1
            return self._placement_cache[key]
        if hasattr(self.queries, 'mapping_capabilities'):
            pools = tuple(dict.fromkeys(plan.cycle or plan.weights))
            capability = self._mapping_capabilities(pools, mode)
            if not capability.reserve_without_backing:
                raise ValueError('backend cannot prepare this placement without physical backing')
            if mode == self.native.BackingMode.shareable_remappable and not (capability.shareable and capability.remappable):
                raise ValueError('backend cannot provide shareable/remappable backing for this placement')
            self.query_counts['placement_queries']+=1
            resolved=self.queries.resolve_placement(self.view, plan, size, mode,
                [self.native.AccessMode.read, self.native.AccessMode.write])
        else:
            # Fully offline ResourceModel has the smaller pure-model contract.
            self.query_counts['placement_queries']+=1
            resolved=self.queries.resolve_placement(self.view,plan,size)
        self._placement_cache[key]=resolved
        return resolved

    def _mapping_capabilities(self, pools, mode):
        key = tuple(pools), str(mode)
        if key not in self._capability_cache:
            self.query_counts['capability_queries'] += 1
            self._capability_cache[key] = self.queries.mapping_capabilities(self.view, list(pools), mode)
        return self._capability_cache[key]

    def _backing_mode(self, requirement, plan):
        """Choose storage through public capabilities, before resolving placement.

        A movable root has one active VA mapping. Ordinary tensor views within
        that mapping remain valid; exported, striped or independently remapped
        ranges instead need the FD-style mapping contract. Explicit modes are
        authoritative, including an explicit shareable request with no exports.
        """
        modes = self.native.BackingMode
        if requirement.backing is not None:
            mode = requirement.backing
        else:
            mode = modes.shareable_remappable
            pools = tuple(dict.fromkeys(plan.cycle or plan.weights))
            if (len(pools) == 1 and not requirement.export_to
                    and requirement.sharing_contract is None and not requirement.independent_mapping
                    and hasattr(self.queries, 'mapping_capabilities')):
                capability = self._mapping_capabilities(pools, modes.private_movable)
                if (capability.reserve_without_backing and capability.remappable
                        and capability.stable_address_rebind):
                    mode = modes.private_movable
        if mode == modes.private_movable and (
                requirement.export_to or requirement.sharing_contract is not None
                or requirement.independent_mapping
                or len(set(plan.cycle or plan.weights)) != 1):
            raise ValueError('private_movable requires one pool and no shared or independently mapped range')
        return mode

    def _score(self, requirement, resolved):
        m = self.native
        agents = requirement.consumers
        if not agents or not (requirement.read_bytes or requirement.write_bytes):
            return True, 0.0
        unknown, total = False, 0.0
        # Retain the resolved owner alongside its key, so id reuse cannot join
        # different endpoints. Endpoint/AccessQuery objects are needed only on
        # an exact access-cache miss, not for each cache hit.
        entry=self._endpoint_keys.get(id(resolved))
        if entry is None:
            endpoint_key=(resolved.address_space,tuple((s.offset_bytes,s.size_bytes,s.pool)
                                                       for s in resolved.segments))
            self._endpoint_keys[id(resolved)]=(resolved,endpoint_key)
        else:
            endpoint_key=entry[1]
        context_key=self._score_context_key
        score_key=(endpoint_key,tuple(agents),requirement.bytes,
                   requirement.read_bytes,requirement.write_bytes,context_key)
        cached=self._score_cache.get(score_key)
        if cached is not None:
            # Every access of a previously successful score is an exact hit.
            self.query_counts['access_hits']+=len(agents)*(bool(requirement.read_bytes)
                                                         +bool(requirement.write_bytes))
            return cached
        for agent in agents:
            if agent not in self._executable_agents:
                raise ValueError(f'buffer consumer is not executable: {agent}')
            for mode, amount in ((m.AccessMode.read, requirement.read_bytes),
                                 (m.AccessMode.write, requirement.write_bytes)):
                if not amount:
                    continue
                payload = min(amount, requirement.bytes)
                key=(agent,str(mode),payload,endpoint_key,context_key)
                if key not in self._access_cache:
                    endpoint=self._endpoints.get(endpoint_key)
                    if endpoint is None:
                        endpoint=_object(m,'TransferEndpoint',address_space=resolved.address_space,
                                         segments=resolved.segments)
                        self._endpoints[endpoint_key]=endpoint
                    if self._score_context is None:
                        self._score_context=_object(m,'CostContext',view_epoch=context_key[0],active=[],
                            execution_slots=dict(context_key[1]),include_preparation=context_key[2])
                    context=self._score_context
                    query=_object(m,'AccessQuery',agent=agent,memory=endpoint,access=mode,
                                  payload_bytes=payload)
                    self.query_counts['access_queries']+=1
                    try:
                        estimate=self.costs.estimate_access(self.view,query,context)
                    except RuntimeError as error:
                        # Keep text, never a traceback that retains compiler roots.
                        self._access_cache[key]=(False,str(error),0.0)
                    else:
                        # Binding properties may copy vectors. Freeze one exact
                        # query's value, not the native estimate owner, on miss.
                        cost_unknown=bool(estimate.quality == m.CostQuality.incomplete
                                          or estimate.missing_facts)
                        self._access_cache[key]=(True,cost_unknown,estimate.total_us)
                else:
                    self.query_counts['access_hits']+=1
                ok,cost_unknown,cost_us=self._access_cache[key]
                if not ok:
                    raise ValueError(f'no usable access cost for {agent}: {cost_unknown}')
                unknown = unknown or cost_unknown
                total += cost_us * (amount / payload)
        result=unknown,total
        self._score_cache[score_key]=result
        return result

    def _candidates(self, requirement):
        m = self.native
        allowed = requirement.allowed_pools or self.pool_order
        if any(pool not in self._pool_members for pool in allowed):
            raise ValueError(f'placement contains an unavailable pool for {requirement.id}')
        if requirement.placement is not None:
            mode = self._backing_mode(requirement, requirement.placement)
            resolved = self._placement(requirement.placement, requirement.bytes, mode)
            if any(segment.pool not in allowed for segment in resolved.segments):
                raise ValueError('explicit placement violates allowed pools')
            yield requirement.placement, resolved, self._score(requirement, resolved)[0], mode
            return
        candidates = []
        for order, pool_id in enumerate(allowed):
            pool = self.pools[pool_id]
            quantum = max([pool.granularity_bytes or 1, *pool.page_sizes_bytes])
            plan = _object(m, 'PlacementPlan', address_space=pool.address_space, stripe_bytes=quantum,
                           weights={pool_id: 1}, cycle=[], phase=0)
            mode = self._backing_mode(requirement, plan)
            resolved = self._placement(plan, requirement.bytes, mode)
            try:
                unknown, cost = self._score(requirement, resolved)
            except ValueError:
                continue
            candidates.append(((unknown, cost, order), plan, resolved, mode))
        for (unknown, _cost, _order), plan, resolved, mode in sorted(candidates, key=lambda x: x[0]):
            yield plan, resolved, unknown, mode

    def _key(self, requirement, resolved, backing):
        contract = requirement.sharing_contract.id if requirement.sharing_contract is not None else None
        return (resolved.key, resolved.address_space, resolved.page_bytes, contract,
                str(backing), tuple(sorted(requirement.export_to)), requirement.independent_mapping)

    def _append(self,slabs,requirement,plan,resolved,unknown,backing,*,prefer_page_alignment=False):
        """Trial immutable packing; optional page alignment never changes minima."""
        key=self._key(requirement,resolved,backing)
        previous=slabs.get(key)
        slab=(dict(previous) if previous is not None else
              dict(plan=plan,resolved=resolved,entries=[],placed=[],bytes=0,
                   permanent=_RangeUnion(),spatial=_RangeUnion(),
                   backing=backing,
                   export_to=tuple(sorted(requirement.export_to)),contract=requirement.sharing_contract,
                   unknown=unknown,lifetime_index=None))
        q=resolved.page_bytes
        period=plan.stripe_bytes*(len(plan.cycle) if plan.cycle else sum(plan.weights.values()))
        uniform=len({segment.pool for segment in resolved.segments})==1 and len(plan.cycle)<=1 and len(plan.weights)<=1
        alignment=max(requirement.alignment,q if requirement.independent_mapping or requirement.export_to else 1)
        preferred=max(q,self.preferred_alignment_bytes or q)
        page_preference=(prefer_page_alignment and requirement.allocated
                         and requirement.bytes>=1<<20 and alignment<preferred)
        if page_preference:
            alignment=max(alignment,preferred)
        if not uniform:alignment=math.lcm(alignment,period)
        reserved=_align(requirement.bytes,q) if requirement.independent_mapping or requirement.export_to else requirement.bytes
        occupied=set(slab['spatial' if requirement.retained else 'permanent'].ranges)
        if not requirement.retained:
            for begin,end in requirement.lifetimes:
                _LifetimeIndex.overlaps(slab['lifetime_index'],begin,end,occupied)
        # A retained buffer conflicts with every earlier VA range. Other
        # buffers conflict with the retained union plus overlapping transients.
        # Pool/cost ordering is unchanged: retention never gets priority over
        # a higher-traffic transient competing for the same scarce memory.
        offset=0
        for low,high in sorted(occupied):
            offset=_align(offset,alignment)
            if offset+reserved<=low:break
            offset=max(offset,high)
        offset=_align(offset,alignment)
        if page_preference and _align(offset+reserved,q)>=1<<64:
            raise _PlanningCapacityError('page-aligned slab exceeds the uint64 address extent')
        _integer(offset+reserved,'slab extent',positive=True)
        slab['bytes']=max(slab['bytes'],_align(offset+reserved,q))
        slab['placement']=self._placement(plan,slab['bytes'],slab['backing'])
        slab['_pending']=requirement,offset,reserved
        candidate=dict(slabs);candidate[key]=slab
        return candidate,key

    @staticmethod
    def _commit(slabs,key):
        slab=slabs[key]
        requirement,offset,reserved=slab.pop('_pending')
        slab['entries'].append(requirement)
        slab['placed'].append((requirement,offset,reserved))
        slab['spatial'].add(offset,offset+reserved)
        if requirement.retained:
            slab['permanent'].add(offset,offset+reserved)
        else:
            for begin,end in requirement.lifetimes:
                slab['lifetime_index']=_LifetimeIndex.insert(slab['lifetime_index'],
                    _LifetimeIndex(begin,end,offset,reserved))

    def _demands(self, slabs):
        by_pool = Counter()
        for slab in slabs.values():
            for segment in slab['placement'].segments:
                by_pool[segment.pool] += segment.size_bytes
        by_group = Counter()
        for pool, count in by_pool.items():
            by_group[self.pools[pool].capacity_group] += count
        return by_pool, by_group

    def _fits(self, slabs):
        by_pool, by_group = self._demands(slabs)
        groups,pools=self._budget_groups,self._budget_pools
        for group, count in by_group.items():
            if group not in groups or count > groups[group][0]:
                return False
        for pool, count in by_pool.items():
            if pool not in pools or count > pools[pool]:
                return False
        for members,remaining in self._budget_constraints:
            fresh = sum(max(0, by_group[group] - groups[group][1])
                        for group in members if group in by_group)
            if fresh > remaining:
                return False
        return True

    def compile(self, requirements: Sequence[BufferRequirement]):
        """Prefer page starts for large buffers, with bounded compact fallback.

        Alignment remains a minimum, not an exact offset/placement guarantee.
        Retry the complete cold layout only for a planning capacity rejection;
        provider, permission and validation failures propagate unchanged.
        """
        requirements=tuple(requirements)
        try:
            return self._compile(requirements,prefer_page_alignment=True)
        except _PlanningCapacityError:
            if not any(r.allocated and r.bytes>=1<<20 for r in requirements):
                raise  # No eligible preference could have changed this layout.
            previous_counts=self.query_counts.copy()
        # Leave the handler before compiling again: the preferred traceback
        # otherwise retains its full failed layout during compact preparation.
        try:
            return self._compile(requirements,prefer_page_alignment=False)
        finally:
            self.query_counts.update(previous_counts)
            self.query_counts['packing_retries']+=1

    def _compile(self,requirements,*,prefer_page_alignment):
        self._reset_queries()
        m = self.native
        if len({r.id for r in requirements}) != len(requirements):
            raise ValueError('buffer identities must be unique')
        sharing = {}
        for requirement in requirements:
            requirement.validate()
            if not self._executable_agents.issuperset(requirement.consumers):
                raise ValueError('buffer consumer is outside the executable resource view')
            contract = requirement.sharing_contract
            if contract is not None:
                identity = _sharing_identity(contract)
                if contract.id in sharing and sharing[contract.id] != identity:
                    raise ValueError('conflicting sharing contract identity')
                sharing[contract.id] = identity
        finish = max((end for requirement in requirements for _begin,end in requirement.lifetimes), default=1)
        requirements = tuple(replace(requirement,lifetimes=((0,finish),)) if requirement.retained else
                             replace(requirement,lifetimes=_normal_lifetimes(requirement.lifetimes))
                             for requirement in requirements)
        choices = []
        slabs = {}
        allocated = [requirement for requirement in requirements if requirement.allocated]
        # Long-lived/high-traffic buffers choose first; every trial computes the
        # actual padded physical demand rather than summing logical tensor bytes.
        allocated.sort(key=lambda r: (not (r.boundary == 'external'),
                                     -(r.read_bytes + r.write_bytes), -r.bytes, r.id))
        for requirement in allocated:
            for plan, resolved, unknown, backing in self._candidates(requirement):
                candidate,key = self._append(slabs,requirement,plan,resolved,unknown,backing,
                                             prefer_page_alignment=prefer_page_alignment)
                fits=self._fits(candidate)
                if not fits and prefer_page_alignment and requirement.bytes>=1<<20:
                    # Keep this cost-ordered placement before trying another pool.
                    candidate,key=self._append(slabs,requirement,plan,resolved,unknown,backing)
                    fits=self._fits(candidate)
                if fits:
                    self._commit(candidate,key)
                    slabs=candidate
                    choices.append((requirement,plan,resolved,unknown))
                    break
            else:
                raise _PlanningCapacityError(f'no feasible placement for {requirement.id} in the planning budget')
        regions, slots, mappings, backings = [], [], [], []
        contracts = {}
        bindings = {}
        initializations = []
        # Separate modes/export rights are distinct roots, even in the same pool.
        roots, root_offsets = {}, Counter()
        for index, slab in enumerate(slabs.values()):
            slot_id = f'slab-{index}'
            resolved = slab['placement']
            base_alignment=max(resolved.page_bytes,
                *(requirement.alignment for requirement in slab['entries']))
            if (self.preferred_alignment_bytes is not None
                    and any(requirement.allocated and requirement.bytes>=1<<20
                            for requirement in slab['entries'])):
                # VA alignment is independent of compact offset fallback and
                # does not round up the slab's physical backing demand.
                base_alignment=max(base_alignment,self.preferred_alignment_bytes)
            regions.append(_object(m, 'VirtualRegion', id=slot_id, address_space=resolved.address_space,
                                   bytes=slab['bytes'], page_bytes=resolved.page_bytes,
                                   base_alignment_bytes=base_alignment))
            contract = slab['contract']
            if contract is not None:
                value = (tuple(contract.participants), tuple(contract.producers), tuple(contract.consumers), contract.protocol)
                if contract.id in contracts and contracts[contract.id][0] != value:
                    raise ValueError('conflicting sharing contract identity')
                contracts[contract.id] = value, contract
            slots.append(_object(m, 'AddressSlot', id=slot_id, region=slot_id, offset_bytes=0,
                                 size_bytes=slab['bytes'], placement=resolved,
                                 sharing_contract=None if contract is None else contract.id))
            capability = (self._mapping_capabilities(
                tuple(dict.fromkeys(segment.pool for segment in resolved.segments)), slab['backing'])
                if hasattr(self.queries, 'mapping_capabilities') else None)
            for segment_index, segment in enumerate(resolved.segments):
                key = segment.pool, str(slab['backing']), slab['export_to'], resolved.page_bytes
                if capability is not None and (
                        not capability.partial_backing_map or not capability.multiple_mappings):
                    key += (index, segment_index)
                if key not in roots:
                    root_id = f'root-{len(roots)}'
                    roots[key] = _object(m, 'BackingRequest', id=root_id, owner=self.participant, pool=segment.pool,
                                         bytes=0, page_bytes=resolved.page_bytes, mode=slab['backing'],
                                         export_to=list(slab['export_to']))
                backing = roots[key]
                offset = root_offsets[key]
                mappings.append(_object(m, 'PageMapping', region=slot_id, virtual_offset_bytes=segment.offset_bytes,
                                        backing=backing.id, backing_offset_bytes=offset,
                                        size_bytes=segment.size_bytes,
                                        access=[m.AccessMode.read, m.AccessMode.write]))
                root_offsets[key] += segment.size_bytes
                backing.bytes = root_offsets[key]
            for requirement, offset, _reserved in slab['placed']:
                bindings[requirement.id] = BufferBinding(requirement.id, slot_id, None, offset, requirement.bytes,
                    requirement.alignment, requirement.independent_mapping, resolved.key)
                if requirement.may_redirect:
                    initializations.append(InitializationCopy(requirement.id, requirement.id, requirement.bytes))
        for requirement in requirements:
            if not requirement.allocated:
                bindings[requirement.id] = BufferBinding(requirement.id, None, requirement.id, 0,
                    requirement.bytes, requirement.alignment, requirement.independent_mapping, None)
        recipe = _object(m, 'MemoryRecipe', participant=self.participant, layout_generation=1,
                         sharing_contracts=[value[1] for value in contracts.values()],
                         addresses=_object(m, 'AddressLayout', regions=regions, slots=slots),
                         backings=list(roots.values()), mappings=mappings)
        self.queries.validate_recipe(self.view, recipe)
        if hasattr(self.queries, 'snapshot'):
            estimate = self.queries.estimate(self.view, recipe, list(self.caches))
            if not estimate.feasible:
                details = ', '.join(f'{group}: new={count/(1<<30):.6f} GiB'
                    for group,count in sorted(estimate.new_backing_bytes_by_capacity_group.items()))
                shortfalls = tuple((getattr(item.kind, 'name', str(item.kind).rsplit('.', 1)[-1]),
                    item.id, item.requested_bytes, item.available_bytes)
                    for item in getattr(estimate, 'shortfalls', ()))
                limits = '; '.join(f'{kind} {identity}: requested={requested/(1<<30):.6f} GiB, '
                    f'available={available/(1<<30):.6f} GiB'
                    for kind,identity,requested,available in shortfalls)
                raise _PlanningCapacityError('runtime rejected the final physical backing estimate'
                    + (f' ({details})' if details else '') + (f'; limits: {limits}' if limits else ''),
                    shortfalls=shortfalls)
        by_pool, by_group = self._demands(slabs)
        signature = sha256(json.dumps({
            'pools': [(p.id, p.capacity_group, p.capacity_bytes, p.address_space, p.granularity_bytes,
                       p.page_sizes_bytes) for p in self.view.graph.pools],
            'agents': list(self.view.executable_agents), 'allocatable': list(self.view.allocatable_pools),
            'slots': self.view.execution_slots,
        }, sort_keys=True).encode()).hexdigest()
        return CompiledMemoryProgram(m.recipe_to_toml(recipe), MappingProxyType(bindings), requirements,
            tuple(initializations), MappingProxyType(dict(by_group)), signature, self.budgets.epoch,
            any(choice[3] for choice in choices), m)

    def compile_model(self, programs: Mapping[str, Sequence[BufferRequirement]], calls: Sequence[ProgramCall],
                      *, serialized: bool, scoped_requirements: Sequence[BufferRequirement] = ()):
        """Link explicit call bindings before packing, including retained outputs.

        Serialized is an execution contract, not a guess from list order. A
        repeated serialized call has identical bindings and no interleaved call;
        its strictly internal lifetimes can be folded onto one iteration.
        External ranges cover that entire iteration, including the backedge:
        tail-written outputs cannot alias temporaries before next iteration's
        first read. Provider-retained buffers still cover the whole Program.
        Internal means no accesses escape the call; loop-carried state must be
        external or retained. Changing bindings require explicit calls.
        scoped_requirements have explicit model-global time offsets. A frontend
        may use them only with a real activation/stop contract; local native
        retention alone does not prove that two prepared states may alias.
        """
        domains=call_time_domains(programs,calls,serialized=serialized)
        combined={requirement.id:requirement for requirement in scoped_requirements}
        if len(combined)!=len(scoped_requirements):
            raise ValueError('scoped buffer identities must be unique')
        scoped_ids=set(combined)
        call_bindings={}
        for call in calls:
            local = tuple(programs[call.program])
            if len({requirement.id for requirement in local}) != len(local):
                raise ValueError('program contains duplicate local buffer identities')
            duration = max((end for requirement in local for _begin, end in requirement.lifetimes), default=1)
            cursor=domains[call.id][0]
            names = {requirement.id for requirement in local}
            if not set(call.bindings) <= names:
                raise ValueError('call binding names an unknown local buffer')
            call_bindings[call.id] = {}
            for requirement in local:
                identity = call.bindings.get(requirement.id, f'{call.id}/{requirement.id}')
                if identity in scoped_ids:
                    raise ValueError('scoped buffer collides with a call binding')
                call_bindings[call.id][requirement.id] = identity
                if serialized:
                    local_live = (((0,duration),) if call.repeat_count > 1 and requirement.boundary == 'external'
                                  else requirement.lifetimes)
                    lifetimes = tuple((cursor + begin, cursor + end) for begin,end in local_live)
                else:
                    # With no inter-call ordering contract, even different
                    # local phases may execute concurrently in different calls.
                    lifetimes = ((0,1),)
                item = replace(requirement, id=identity, lifetimes=lifetimes,
                    read_bytes=requirement.read_bytes*call.repeat_count,
                    write_bytes=requirement.write_bytes*call.repeat_count)
                if identity not in combined:
                    combined[identity] = item
                    continue
                old = combined[identity]
                if old.bytes != item.bytes:
                    raise ValueError('linked storage capacity differs between calls')
                # One local output becomes another call's borrowed input. Keep
                # storage alive across the gap, not just at producer/consumer uses.
                owning = old if old.allocated else item
                if old.allocated and item.allocated and old.boundary == item.boundary == 'internal':
                    raise ValueError('two independent internal buffers cannot silently become one model storage')
                begin = min(a for a, _ in old.lifetimes + item.lifetimes)
                end = max(b for _, b in old.lifetimes + item.lifetimes)
                allowed = (tuple(pool for pool in old.allowed_pools if pool in item.allowed_pools)
                           if old.allowed_pools and item.allowed_pools else old.allowed_pools or item.allowed_pools)
                if old.allowed_pools and item.allowed_pools and not allowed:
                    raise ValueError('linked storage has incompatible pool constraints')
                def pattern(plan):
                    return None if plan is None else (plan.address_space, plan.stripe_bytes,
                        tuple(sorted(plan.weights.items())), tuple(plan.cycle), plan.phase)
                if old.placement is not None and item.placement is not None and pattern(old.placement) != pattern(item.placement):
                    raise ValueError('linked storage has incompatible placement patterns')
                if old.sharing_contract is not None and item.sharing_contract is not None and \
                        _sharing_identity(old.sharing_contract) != _sharing_identity(item.sharing_contract):
                    raise ValueError('linked storage has incompatible sharing contracts')
                if old.backing is not None and item.backing is not None and old.backing != item.backing:
                    raise ValueError('linked storage has incompatible backing modes')
                combined[identity] = replace(owning, lifetimes=((begin, end),),
                    alignment=max(old.alignment, item.alignment), retained=old.retained or item.retained,
                    traffic_assumed=old.traffic_assumed or item.traffic_assumed,
                    allowed_pools=allowed, placement=old.placement or item.placement,
                    sharing_contract=old.sharing_contract or item.sharing_contract,
                    independent_mapping=old.independent_mapping or item.independent_mapping,
                    export_to=tuple(dict.fromkeys(old.export_to + item.export_to)),
                    backing=old.backing if old.backing is not None else item.backing,
                    consumers=tuple(dict.fromkeys(old.consumers + item.consumers)),
                    read_bytes=old.read_bytes + item.read_bytes, write_bytes=old.write_bytes + item.write_bytes)
        program = self.compile(tuple(combined.values()))
        return replace(program, call_bindings=MappingProxyType({key: MappingProxyType(value)
                       for key, value in call_bindings.items()}), call_sequence=tuple(call.id for call in calls))


    def prepare_model(self, programs, calls, *, serialized, admission_attempts=16,
                      configure_resources=None, probe_resources=None,
                      finalize_resources=None, preparation_consensus=None, **options):
        """Close cold resource capacity, then admit an offset-only model plan.

        ``finalize_resources(prepared)`` inspects proposed placements and may
        provision external resources; True requests a fresh budget and replan.
        Distributed callers supply ``preparation_consensus(local_ok)``: a cold
        all-participant AND with stopped submissions. Planning, resource changes
        and admission outcomes then take the same retry branch on every rank.
        Accepted grants stay held while unsuccessful peers replan against the
        remaining budget. Optional local `configure_resources(prepared)` runs
        before collective resource callbacks. A `probe_resources(prepared)`
        may close a no-growth resource contract, but must never allocate
        backing. When it reports growth on any participant, withdraw grants
        at a barrier before `finalize_resources` provisions the new backing.
        A finalizer without a probe is treated as potentially allocating.
        Growth invalidates all proposals. Failure closes local proposals and
        grants. No graph backing is acquired.

        A typed fresh-capacity rejection may leave observed physical sampling
        headroom on a later attempt of this call. Learning requires identical
        accounting/epoch and untruncated pool allowances; idle cache remains
        reusable. A known unaccepted reserve can rebase a rollback epoch only
        when compile, last-reserve and current nonphysical accounting agree.
        The private guard never changes native admission limits and is
        discarded when its related pool/constraint component changes or this
        call exits. Unrelated accounting changes retain proved margins; new
        typed learning still requires the full same-epoch fingerprint. Compiler
        instances retain their existing non-reentrant preparation contract.
        """
        if type(admission_attempts) is not int or not 1 <= admission_attempts <= 64:
            raise ValueError('admission_attempts must be in [1,64]')
        for name,callback in (('configure_resources',configure_resources),
                              ('probe_resources',probe_resources),('finalize_resources',finalize_resources),
                              ('preparation_consensus',preparation_consensus)):
            if callback is not None and not callable(callback):
                raise TypeError(name+' must be callable')
        if probe_resources is not None and finalize_resources is None:
            raise ValueError('a resource probe requires a growth finalizer')
        def agree(local):
            result=local if preparation_consensus is None else preparation_consensus(local)
            if type(result) is not bool or (result and not local):
                raise ValueError('preparation_consensus must return an all-participant bool AND')
            return result
        programs = {name: tuple(requirements) for name, requirements in programs.items()}
        calls = tuple(calls)
        options = dict(options)
        if 'scoped_requirements' in options:
            options['scoped_requirements'] = tuple(options['scoped_requirements'])
        capacity_errors = {
            'invalid_argument: pool capacity exhausted',
            'invalid_argument: hard capacity limit exceeded',
            'invalid_argument: physical free capacity exceeded',
            'invalid_argument: shared domain capacity limit exceeded',
            'shared_capacity.exhausted: shared group hard capacity exhausted',
            'shared_capacity.exhausted: overlapping shared capacity constraint exhausted',
            'shared_capacity.physical_free: physical free sample minus pending reservations is exhausted',
        }
        def retryable(error):
            return isinstance(error,_PlanningCapacityError) or (
                isinstance(error,RuntimeError) and (str(error) in capacity_errors
                    or _physical_free_rejection(str(error))))
        def available(snapshot):
            return (tuple(sorted((pool.pool, pool.allocatable_bytes) for pool in snapshot.pools)),
                    tuple(sorted((constraint.id, constraint.remaining_bytes) for constraint in snapshot.constraints)))
        _no_policy=object()
        _previous_policy=getattr(self,'_cold_physical_headroom',_no_policy)
        _cold_policy=_ColdPhysicalHeadroom()
        self._cold_physical_headroom=_cold_policy
        prepared=None
        held=False
        try:
            for attempt in range(admission_attempts):
                planned_budget=available(self.budgets)
                error=None
                if prepared is None:
                    try:
                        prepared=self.compile_model(programs,calls,serialized=serialized,**options).prepare(
                            self.queries,self.view,caches=self.caches)
                    except Exception as failure:
                        error=failure
                if agree(error is None):
                    changed=False
                    if configure_resources is not None:
                        try:
                            configure_resources(prepared)
                        except Exception as failure:
                            error=failure
                        if not agree(error is None):
                            raise error if error is not None else RuntimeError('peer cold resource configuration failed')
                    needs_growth=finalize_resources is not None
                    if probe_resources is not None:
                        try:
                            needs_growth=probe_resources(prepared)
                            if type(needs_growth) is not bool:
                                raise TypeError('probe_resources must return bool')
                        except Exception as failure:
                            error=failure
                        if not agree(error is None):
                            raise error if error is not None else RuntimeError('peer cold resource probe failed')
                        needs_growth=not agree(not needs_growth)
                    if needs_growth:
                        # Keep successful grants while peers replan. Once every
                        # proposal is ready, withdraw those grants and synchronize
                        # before a collective finalizer can provision its roots.
                        # The offset plans survive; no rank compiles in this gap.
                        prepared._withdraw_admission()
                        held=False
                        agree(True)
                        try:
                            changed=finalize_resources(prepared)
                            if type(changed) is not bool:
                                raise TypeError('finalize_resources must return bool')
                        except Exception as failure:
                            error=failure
                        if not agree(error is None):
                            raise error if error is not None else RuntimeError('peer cold resource finalization failed')
                        changed=not agree(not changed)
                        # A positive probe required synchronized withdrawal.
                        # Even if the provider no longer needs backing, the
                        # proposals' admission budgets must be refreshed.
                        changed=changed or probe_resources is not None
                    if changed:
                        prepared.close();prepared=None
                        if attempt+1==admission_attempts:
                            raise RuntimeError('external resource capacity did not stabilize before admission')
                        self.budgets=self.queries.snapshot(self.view,list(self.caches))
                        continue
                    if not held:
                        try:
                            prepared.admit()
                            held=True
                        except Exception as failure:
                            error=failure
                    if agree(error is None):
                        return prepared
                # Accepted grants remain visible to unsuccessful peers. Releasing
                # every winner here would let all ranks choose the same scarce
                # pool again even when a jointly feasible spill placement exists.
                if prepared is not None and not held:
                    prepared.close();prepared=None
                if not agree(error is None or retryable(error)):
                    raise error if error is not None else RuntimeError('peer model preparation failed')
                current=self.queries.snapshot(self.view,list(self.caches))
                headroom_changed=False
                if isinstance(error,_PlanningCapacityError):
                    headroom_changed=self._cold_physical_headroom.observe(error.shortfalls,self.budgets,current)
                elif isinstance(error,_ReservePhysicalCapacityError):
                    headroom_changed=self._cold_physical_headroom.observe_reserve(error,self.budgets,current)
                changed=not agree(available(current)==planned_budget and not headroom_changed)
                if not changed or attempt+1==admission_attempts:
                    raise error if error is not None else RuntimeError('peer model capacity admission failed')
                self.budgets=current
        except BaseException:
            if prepared is not None:
                prepared.close()
            raise
        finally:
            if id(getattr(self,'_budget_groups',None)) == getattr(_cold_policy,'adjusted_groups_id',None):
                self._budget_groups=_cold_policy.unadjusted_groups
            if _previous_policy is _no_policy:
                del self._cold_physical_headroom
            else:
                self._cold_physical_headroom=_previous_policy


class PreparedProgram:
    """Address-only preparation followed by an explicit backing transaction.

    Tensor views can be built before materialization but cannot be read/written
    until materialize succeeds. A tensor Storage pins its VA owner and actual
    physical binding, so escaped aliases do not depend on this Python owner.
    """
    _invocations = itertools.count(1)

    def __init__(self, program, runtime, view, caches=()):
        self.program, self.runtime, self.view, self.caches = program, runtime, view, caches
        self.recipe = program.memory
        self._placements={slot.id:slot.placement for slot in self.recipe.addresses.slots}
        self.addresses = runtime.prepare_addresses(view, self.recipe.addresses)
        self.backings = self.binding = None
        self._lease = None
        self.external = {}
        self._lock = threading.RLock()
        import weakref
        self._views = {}
        self._pins = weakref.WeakSet()
        self._closed = False

    def _withdraw_admission(self):
        """Cancel only a pending grant while retaining the offset plan and VA.

        Used at the compiler's stopped, all-proposal finalization barrier.
        Materialized/externally bound programs must never enter this phase.
        """
        with self._lock:
            if self._closed or self.binding is not None or self.backings is not None:
                raise RuntimeError('only an unmaterialized prepared program can withdraw admission')
            self._lease=None

    def admit(self):
        """Hold pending physical capacity without allocating or mapping pages."""
        with self._lock:
            if self._closed:
                raise RuntimeError('prepared program is closed')
            if self.binding is not None:
                raise RuntimeError('prepared program is already materialized')
            if self._lease is None:
                self._lease = reserve_recipe_latest(self.runtime, self.view, self.recipe, self.caches)
        return self

    def materialize(self, *, external=None):
        with self._lock:
            if self._closed:
                raise RuntimeError('prepared program is closed')
            if self.binding is not None:
                raise RuntimeError('prepared program is already materialized')
            inputs = dict(external or {})
            missing = {binding.external for binding in self.program.bindings.values()
                       if binding.external is not None} - inputs.keys()
            missing.update(copy.external for copy in self.program.initializations if copy.external not in inputs)
            if missing:
                raise ValueError(f'missing external memory bindings: {sorted(missing)}')
            self.admit()
            try:
                backing = self.runtime.acquire_backing(self.view, self.recipe, self._lease)
                binding = self.runtime.bind(self.view, self.recipe, self.addresses, backing)
            finally:
                # An acquisition attempt may consume only part of a lease.
                # Drop any remainder on both success and failure.
                self._lease = None
            self.backings, self.binding, self.external = backing, binding, inputs
            for pin in self._pins:
                pin.binding = binding
            try:
                for copy in self.program.initializations:
                    source = inputs[copy.external]
                    if not hasattr(source, 'untyped_storage'):
                        raise TypeError('state initialization currently requires an owning Torch tensor')
                    import torch
                    source_bytes = torch.empty(0, dtype=torch.uint8, device='cpu').set_(source.untyped_storage(), 0,
                        (copy.bytes,), (1,))
                    self.tensor(copy.buffer, (copy.bytes,), torch.uint8).copy_(source_bytes)
            except BaseException:
                proof = self.runtime.quiesce(binding, [])
                self.runtime.unbind(binding, proof)
                self.backings = self.binding = None
                for pin in self._pins:
                    pin.binding = None
                self.external = {}
                raise
        return self

    def tensor(self, buffer, shape, dtype, strides=None, *, byte_offset=0):
        with self._lock:
            import torch
            if self._closed:
                raise RuntimeError('prepared program is closed')
            binding = self.program.bindings[buffer]
            if binding.external is not None:
                raise ValueError('borrowed external buffers retain their caller-provided tensor views')
            shape = tuple(_integer(value,'tensor extent') for value in shape)
            if any(value < 0 for value in shape):
                raise ValueError('negative tensor extent')
            itemsize = dtype.itemsize
            strides = tuple(strides) if strides is not None else None
            if strides is not None and (len(strides) != len(shape) or any(value < 0 for value in strides)):
                raise ValueError('invalid tensor strides')
            extent = math.prod(shape) if strides is None else (0 if 0 in shape else 1 + sum((d - 1) * s for d, s in zip(shape, strides)))
            if byte_offset < 0 or byte_offset % itemsize or byte_offset + extent * itemsize > binding.size_bytes:
                raise ValueError('tensor view exceeds its logical buffer')
            # One StorageImpl per logical buffer preserves its aliases while allowing
            # distinct buffers to overlap physically at disjoint proven lifetimes.
            key = buffer
            if key not in self._views:
                owner = self.addresses.slot_buffer(binding.slot)
                view = memoryview(owner)
                raw = torch.frombuffer(view, dtype=torch.uint8, count=binding.size_bytes,
                                       offset=binding.view_offset_bytes)
                import weakref
                # frombuffer's StorageImpl retains the exact memoryview, including
                # all later tensor aliases. Its finalizer owns a separate future
                # binding pin; it never captures this instance and creates no cycle.
                class Pin:
                    pass
                pin = Pin()
                pin.binding = self.binding
                self._pins.add(pin)
                weakref.finalize(view, lambda native_owner=pin: None)
                self._views[key] = raw
            canonical = self._views[key]
            raw = canonical.narrow(0,0,binding.size_bytes//itemsize*itemsize).view(dtype)
            offset = byte_offset // itemsize
            if strides is None:
                return raw.narrow(0, offset, extent).reshape(shape)
            return raw.as_strided(shape, strides, offset)

    def placement(self,buffer,*,byte_offset=0,size_bytes=None):
        """Describe an actually materialized logical subrange by offsets/pools.

        This does not sample arbitrary pointers. Use planned_placement to inspect
        a proposal before physical backing; it does not certify resident pages.
        """
        with self._lock:
            if self._closed or self.binding is None:
                raise RuntimeError('placement query requires a live materialized program')
            return self.planned_placement(buffer,byte_offset=byte_offset,size_bytes=size_bytes)

    def planned_placement(self,buffer,*,byte_offset=0,size_bytes=None):
        """Inspect an offset-only placement proposal without backing or admission.

        Returned endpoints describe the recipe, not observed physical residency.
        Cold communication planning may use them before reserving graph capacity.
        Borrowed external storage has no placement certified by this recipe.
        """
        with self._lock:
            if self._closed:
                raise RuntimeError('planned placement query requires a live program')
            binding=self.program.bindings[buffer]
            if binding.slot is None:
                raise ValueError('borrowed external placement is not described by this program')
            _integer(byte_offset,'placement subrange offset')
            size_bytes=binding.size_bytes-byte_offset if size_bytes is None else size_bytes
            _integer(size_bytes,'placement subrange size',positive=True)
            if byte_offset+size_bytes>binding.size_bytes:
                raise ValueError('placement subrange exceeds its logical buffer')
            low=binding.view_offset_bytes+byte_offset
            high=low+size_bytes
            placement=self._placements[binding.slot]
            m=self.program._native
            segments=[]
            for segment in placement.segments:
                a=max(low,segment.offset_bytes);b=min(high,segment.offset_bytes+segment.size_bytes)
                if a<b:
                    segments.append(_object(m,'PlacedSegment',offset_bytes=a-low,size_bytes=b-a,pool=segment.pool))
            return _object(m,'TransferEndpoint',address_space=placement.address_space,segments=segments)

    def record_use(self, token):
        with self._lock:
            if self.binding is None:
                raise RuntimeError('program is not materialized')
            self.binding.record_use(token)

    def completed(self, *, succeeded=True):
        with self._lock:
            if self.binding is None:
                raise RuntimeError('program is not materialized')
            m = self.program._native
            stamp = _object(m, 'InvocationStamp', id=f'graph-{next(self._invocations)}',
                            binding_generation=self.binding.generation)
            status = _object(m, 'EventStatus', state=m.EventState.succeeded if succeeded else m.EventState.failed,
                             accesses_stopped=True)
            return self.runtime.events.complete(stamp, status)

    def close(self):
        with self._lock:
            if self._closed:
                return
            self._views.clear()
            if self.binding is not None:
                if not self._pins:
                    proof = self.runtime.quiesce(self.binding, [])
                    self.runtime.unbind(self.binding, proof)
            self.binding = self.backings = self._lease = None
            self.external = {}
            self._closed = True


def _metadata_only_event(graph,event):
    if 'state_summary_metadata_only' in event.attributes:
        return event.attributes['state_summary_metadata_only']
    inputs={graph.tensors[use.tensor_id].storage_id for use in event.uses}
    outputs={graph.tensors[identity].storage_id for identity in event.outputs}
    modes={_plain_enum(use.access) for use in event.uses}
    return (not modes.intersection({'write','read_write','allocate'}) and bool(outputs) and outputs<=inputs
            or event.name.startswith(('prim.','aten.size','aten.sym_size','aten.stride',
                                      'aten.empty.','aten.empty_like.','aten.empty_strided.')))


def _event_weights(graph):
    from vecops.graph import event_execution_weights
    weights=event_execution_weights(graph)
    return {event.index:weights[event.id] for event in graph.events}


def requirements_from_graph(graph, *, allowed_pools=(), consumers=(), owned_outputs=False):
    """Normalize captured tensors to external/internal storage requirements.

    TensorKind is frontend provenance only. All tensor views of one storage
    share one requirement; operator workspaces can be appended without another
    allocation category. State remains borrowed until an explicit redirect is
    supplied by graph policy.
    """
    graph.validate()
    outputs = set(graph.outputs)
    result = []
    weights=_event_weights(graph)
    traffic=defaultdict(lambda: {'read':0,'write':0})
    # Visit each event once. Per-storage rescans make metadata preparation
    # quadratic in large captured graphs; aliases still count only the maximum
    # logical payload of each access mode within the event.
    for event in graph.events:
        if _metadata_only_event(graph,event):
            continue
        touched=defaultdict(lambda: {'read':0,'write':0})
        for use in event.uses:
            tensor=graph.tensors[use.tensor_id]
            mode=_plain_enum(use.access)
            logical=tensor.metadata.logical_bytes
            size=logical if isinstance(logical,int) else graph.storages[tensor.storage_id].bytes
            if mode in {'read','read_write'}:
                touched[tensor.storage_id]['read']=max(touched[tensor.storage_id]['read'],size)
            if mode in {'write','read_write'}:
                touched[tensor.storage_id]['write']=max(touched[tensor.storage_id]['write'],size)
        for identity,amounts in touched.items():
            for mode,amount in amounts.items():
                traffic[identity][mode]+=math.ceil(amount*weights[event.index])
    for storage_id, storage in graph.storages.items():
        if storage.bytes == 0:
            continue
        tensors = [graph.tensors[identity] for identity in storage.tensors]
        kinds = {_plain_enum(tensor.kind) for tensor in tensors}
        escaping = bool(outputs.intersection(storage.tensors)) or any(tensor.escapes for tensor in tensors)
        external = storage.external or storage.persistent or escaping
        role = ('state' if storage.persistent or kinds & {'parameter', 'buffer'} else
                'output' if escaping else 'input') if external else None
        read_bytes=traffic[storage_id]['read']
        write_bytes=traffic[storage_id]['write']
        result.append(BufferRequirement(storage_id, storage.bytes, lifetimes=((storage.lifetime_start,
            storage.lifetime_end + 1),), allowed_pools=tuple(allowed_pools), consumers=tuple(consumers),
            read_bytes=read_bytes, write_bytes=write_bytes, boundary='external' if external else 'internal',
            external_role=role, ownership='runtime_owned' if not external or (owned_outputs and role == 'output')
            else 'borrowed', read_only=all(tensor.read_only for tensor in tensors),
            retained=any(tensor.annotations.get('retained',False) for tensor in tensors)))
    return tuple(result)


def _workspace_descriptors(lease):
    descriptors=tuple(lease.buffer_requirements)
    names=set()
    for descriptor in descriptors:
        name=descriptor['name']
        if not isinstance(name,str) or not name:
            raise ValueError('native workspace identity must be a nonempty string')
        if name in names:
            raise ValueError(f'duplicate native workspace identity: {name}')
        names.add(name)
    return descriptors


def native_workspace_requirements(lease, *, prefix, lifetime=(0, 1), allowed_pools=(), consumers=(),
                                  execution_count=1):
    """Describe native workspaces without constructing their physical storage.

    Replicas share one logical requirement whose stride is provider-defined.
    Without an explicit graph subinterval proof all native local lifetimes are
    conservatively covered by the enclosing call's half-open lifetime.
    ``execution_count`` weights per-call traffic only; storage size, replica
    count and lifetime do not grow with the number of synchronous calls. Pass
    the complete logical count, including any enclosing recurrence, once.
    """
    _integer(execution_count, 'native execution count', positive=True)
    result = []
    for descriptor in _workspace_descriptors(lease):
        size = _integer(descriptor['bytes'], 'workspace bytes')
        replicas = _integer(descriptor['replicas'], 'workspace replicas', positive=True)
        stride = _integer(descriptor['replica_stride'], 'replica stride')
        alignment = _integer(descriptor['alignment'], 'workspace alignment', positive=True)
        if replicas > 1 and (stride < size or stride % alignment):
            raise ValueError('workspace replica stride violates size/alignment')
        if descriptor['domain'] not in {'global', 'worker_local'}:
            raise ValueError('unknown native workspace domain')
        total = (replicas - 1) * stride + size
        traffic = descriptor.get('traffic_bytes', 0)
        if not math.isfinite(traffic) or traffic < 0:
            raise ValueError('invalid native workspace traffic')
        read_bytes = _integer((math.ceil(traffic) if traffic else total) * execution_count,
                              'native workspace read traffic')
        result.append(BufferRequirement(f'{prefix}/{descriptor["name"]}', max(1, total), alignment,
            lifetimes=(tuple(lifetime),), allowed_pools=tuple(allowed_pools), consumers=tuple(consumers),
            read_bytes=read_bytes, retained=descriptor.get('retained',True),
            traffic_assumed=not bool(traffic)))
    return tuple(result)


class ScopedNativeLease:
    """An explicit activation-to-stop lifetime for a prepared native instance.

    Native ``retained`` means state survives calls while its instance is active.
    This adapter may lower it to a bounded Program range only because activate
    delays construction/initialization, and suspend destroys the state before
    another range reuses its pages. Callers must order both lifecycle operations
    around every use; a mere list of observed call times is insufficient.
    """
    def __init__(self,lease,*,prefix,lifetime,allowed_pools=(),consumers=(),execution_count=1):
        begin,end=lifetime
        _integer(begin,'activation begin')
        _integer(end,'activation end',positive=True)
        if begin>=end:
            raise ValueError('native activation lifetime must be nonempty')
        if not hasattr(lease,'suspend_buffers'):
            raise ValueError('scoped native lease requires an explicit stop operation')
        _integer(execution_count, 'native execution count', positive=True)
        self.lease,self.prefix,self.lifetime=lease,prefix,(begin,end)
        self.allowed_pools,self.consumers=tuple(allowed_pools),tuple(consumers)
        self.execution_count=execution_count
        self.active=False

    def requirements(self):
        return tuple(replace(requirement,retained=False) for requirement in native_workspace_requirements(
            self.lease,prefix=self.prefix,lifetime=self.lifetime,
            allowed_pools=self.allowed_pools,consumers=self.consumers,
            execution_count=self.execution_count))

    def activate(self,prepared):
        if self.active:
            return
        activate_native_buffers(self.lease,prepared,prefix=self.prefix)
        self.active=True

    def suspend(self):
        if self.active:
            self.lease.suspend_buffers()
            self.active=False


def activate_native_buffers(lease,prepared,*,prefix,bindings=None):
    """Activate under the same lock as materialization and explicit close."""
    with prepared._lock:
        return _activate_native_buffers_locked(lease,prepared,prefix=prefix,bindings=bindings)


def _activate_native_buffers_locked(lease, prepared, *, prefix, bindings=None):
    """Activate a described kernel with compiler-chosen root ownership."""
    if prepared.binding is None:
        raise RuntimeError('materialize the program before activating native buffers')
    names = bindings or {}
    mapping = {}
    for descriptor in _workspace_descriptors(lease):
        name = descriptor['name']
        identity = names.get(name, f'{prefix}/{name}')
        binding = prepared.program.bindings[identity]
        if binding.slot is None:
            raise ValueError('native workspace must have runtime-owned storage')
        address = prepared.addresses.slot(binding.slot).value + binding.view_offset_bytes
        total = (descriptor['replicas'] - 1) * descriptor['replica_stride'] + descriptor['bytes']
        if total > binding.size_bytes or address % descriptor['alignment']:
            raise ValueError('native workspace binding violates its descriptor')
        mapping[name] = dict(address=address, bytes=total, replica_stride=descriptor['replica_stride'])
    class Pin:
        pass
    pin = Pin()
    pin.binding, pin.addresses = prepared.binding, prepared.addresses
    prepared._pins.add(pin)
    lease.activate_buffers(mapping, pin)
    return lease


def framework_budget_recipe(requirements,pool,*,participant,native_module=None):
    """Create a backing-only pending-budget recipe, never physical placement.

    The caller selects the pool whose capacity is reserved for framework malloc.
    Real OS pages may differ. Keeping this lease while OS free samples already
    include those mallocs deliberately yields a conservative double reservation.
    """
    m=native_module or importlib.import_module('vecops.memory').native
    if not getattr(pool,'id',None) or not pool.page_sizes_bytes:
        raise ValueError('framework budget requires an explicit concrete MemoryPool')
    requirements=tuple(requirements)
    finish=max((end for req in requirements for _begin,end in req.lifetimes),default=1)
    storages={}
    for requirement in requirements:
        requirement.validate()
        begin=0 if requirement.retained else min(begin for begin,_end in requirement.lifetimes)
        end=finish if requirement.retained else max(end for _begin,end in requirement.lifetimes)
        if requirement.id in storages:
            size,old_begin,old_end=storages[requirement.id]
            if size!=requirement.bytes:
                raise ValueError(f'linked framework storage {requirement.id} changes capacity: '
                                 f'{size} versus {requirement.bytes} bytes')
            begin,end=min(begin,old_begin),max(end,old_end)
        storages[requirement.id]=(requirement.bytes,begin,end)
    changes=Counter()
    for size,begin,end in storages.values():
        changes[begin]+=size
        changes[end]-=size
    current=peak=0
    for event in sorted(changes):
        current+=changes[event]
        peak=max(peak,current)
    backings=[]
    if peak:
        quantum=max([pool.granularity_bytes or 1,*pool.page_sizes_bytes])
        backings=[_object(m,'BackingRequest',id='framework-budget',owner=participant,pool=pool.id,
            bytes=_align(peak,quantum),page_bytes=quantum,mode=m.BackingMode.private_memory,export_to=[])]
    return _object(m,'MemoryRecipe',participant=participant,layout_generation=1,backings=backings,
                   addresses=_object(m,'AddressLayout',regions=[],slots=[]),mappings=[])
