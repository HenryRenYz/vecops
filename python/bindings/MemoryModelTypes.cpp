// SPDX-FileCopyrightText: 2026 HenryRenYz and vecops contributors
// SPDX-License-Identifier: MIT
#include "MemoryModelBindings.h"
#include "vecops/memory/HardwareTemplate.h"
#include "vecops/memory/ResourceModel.h"
#include <pybind11/stl.h>
#include <stdexcept>

namespace py = pybind11;
namespace {
using namespace vecops::memory::model;
template <class T>
T unwrap(Result<T> result) {
  if (!result) {
    const auto& error = std::get<Error>(result.state);
    throw std::runtime_error(error.code + ": " + error.message);
  }
  return std::move(result).value();
}
void unwrap(Result<void> result) {
  if (!result)
    throw std::runtime_error(result.error->code + ": " + result.error->message);
}
// std::variant<string,uint64,bool> would otherwise permit bool -> integer.
// Preserve the exact fact type used by hardware requirement validation.
HardwareFacts facts_from_python(const py::dict& values) {
  HardwareFacts result;
  for (const auto& [key, value] : values) {
    if (!py::isinstance<py::str>(key))
      throw py::type_error("hardware fact names must be strings");
    const auto name = py::cast<std::string>(key);
    if (py::isinstance<py::bool_>(value))
      result.emplace(name, py::cast<bool>(value));
    else if (py::isinstance<py::int_>(value))
      result.emplace(name, py::cast<std::uint64_t>(value));
    else if (py::isinstance<py::str>(value))
      result.emplace(name, py::cast<std::string>(value));
    else
      throw py::type_error("hardware facts must be strings, uint64 integers, or booleans");
  }
  return result;
}
py::dict facts_to_python(const HardwareFacts& values) {
  py::dict result;
  for (const auto& [key, value] : values)
    std::visit([&](const auto& item) { result[py::str(key)] = py::cast(item); }, value);
  return result;
}
} // namespace

// Direct uint64 fields retain Python's integer precision. Vector/map properties
// use pybind11's STL value conversion: assign an updated container back rather
// than mutating a temporary list/dict returned by a property getter.
void bind_memory_model_types(py::module_& m) {
  using namespace vecops::memory::model;
  py::enum_<AccessMode>(m, "AccessMode")
    .value("read", AccessMode::read)
    .value("write", AccessMode::write);
  py::enum_<MappingMode>(m, "MappingMode")
    .value("unknown", MappingMode::unknown)
    .value("none", MappingMode::none)
    .value("direct", MappingMode::direct)
    .value("requires_mapping", MappingMode::requires_mapping);
  py::enum_<SharingMode>(m, "SharingMode")
    .value("shared", SharingMode::shared)
    .value("read_write", SharingMode::read_write);
  py::enum_<DemandKind>(m, "DemandKind")
    .value("read", DemandKind::read)
    .value("write", DemandKind::write)
    .value("traffic", DemandKind::traffic)
    .value("occupancy", DemandKind::occupancy);
  py::enum_<ResourceKind>(m, "ResourceKind")
    .value("bandwidth", ResourceKind::bandwidth)
    .value("execution_slots", ResourceKind::execution_slots);
  py::enum_<ModelTargetKind>(m, "ModelTargetKind")
    .value("access", ModelTargetKind::access)
    .value("method", ModelTargetKind::method)
    .value("resource", ModelTargetKind::resource);
  py::enum_<BackingMode>(m, "BackingMode")
    .value("private_memory", BackingMode::private_memory)
    .value("shareable_remappable", BackingMode::shareable_remappable)
    .value("private_movable", BackingMode::private_movable);
  py::enum_<CostQuality>(m, "CostQuality")
    .value("calibrated", CostQuality::calibrated)
    .value("interpolated", CostQuality::interpolated)
    .value("reference", CostQuality::reference)
    .value("incomplete", CostQuality::incomplete);
  py::enum_<MemoryRefKind>(m, "MemoryRefKind")
    .value("slot", MemoryRefKind::slot)
    .value("external", MemoryRefKind::external);
  py::enum_<MemoryOpcode>(m, "MemoryOpcode")
    .value("copy", MemoryOpcode::copy)
    .value("fill", MemoryOpcode::fill)
    .value("register_range", MemoryOpcode::register_range)
    .value("unregister_range", MemoryOpcode::unregister_range)
    .value("fence", MemoryOpcode::fence);
  py::enum_<EventState>(m, "EventState")
    .value("pending", EventState::pending)
    .value("succeeded", EventState::succeeded)
    .value("failed", EventState::failed)
    .value("cancelled", EventState::cancelled);
  py::enum_<ProgressMode>(m, "ProgressMode")
    .value("unknown", ProgressMode::unknown)
    .value("caller_driven", ProgressMode::caller_driven)
    .value("cooperative", ProgressMode::cooperative)
    .value("dedicated_agent", ProgressMode::dedicated_agent)
    .value("autonomous", ProgressMode::autonomous);

#define MEMORY_FIELD(Type, Name) .def_readwrite(#Name, &Type::Name)
  py::class_<CapacityGroup>(m, "CapacityGroup")
    .def(py::init<>())
    MEMORY_FIELD(CapacityGroup, id)
    MEMORY_FIELD(CapacityGroup, capacity_bytes);
  py::class_<AllocationDomain>(m, "AllocationDomain")
    .def(py::init<>())
    MEMORY_FIELD(AllocationDomain, id)
    MEMORY_FIELD(AllocationDomain, rule)
    MEMORY_FIELD(AllocationDomain, members);
  py::class_<Agent>(m, "Agent")
    .def(py::init<>())
    MEMORY_FIELD(Agent, id)
    MEMORY_FIELD(Agent, capabilities)
    MEMORY_FIELD(Agent, address_space);
  py::class_<MemoryPool>(m, "MemoryPool")
    .def(py::init<>())
    MEMORY_FIELD(MemoryPool, id)
    MEMORY_FIELD(MemoryPool, capacity_bytes)
    MEMORY_FIELD(MemoryPool, address_space)
    MEMORY_FIELD(MemoryPool, granularity_bytes)
    MEMORY_FIELD(MemoryPool, page_sizes_bytes)
    MEMORY_FIELD(MemoryPool, read_service)
    MEMORY_FIELD(MemoryPool, write_service)
    MEMORY_FIELD(MemoryPool, capacity_group);
  py::class_<AccessConnection>(m, "AccessConnection")
    .def(py::init<>())
    MEMORY_FIELD(AccessConnection, id)
    MEMORY_FIELD(AccessConnection, rule)
    MEMORY_FIELD(AccessConnection, agent)
    MEMORY_FIELD(AccessConnection, pool)
    MEMORY_FIELD(AccessConnection, access)
    MEMORY_FIELD(AccessConnection, bandwidth)
    MEMORY_FIELD(AccessConnection, latency_ns)
    MEMORY_FIELD(AccessConnection, mapping)
    MEMORY_FIELD(AccessConnection, prerequisites)
    MEMORY_FIELD(AccessConnection, read_service)
    MEMORY_FIELD(AccessConnection, write_service)
    MEMORY_FIELD(AccessConnection, read_fabric)
    MEMORY_FIELD(AccessConnection, write_fabric);
  py::class_<Resource>(m, "Resource")
    .def(py::init<>())
    MEMORY_FIELD(Resource, id)
    MEMORY_FIELD(Resource, rule)
    MEMORY_FIELD(Resource, bandwidth)
    MEMORY_FIELD(Resource, read)
    MEMORY_FIELD(Resource, write)
    MEMORY_FIELD(Resource, sharing)
    MEMORY_FIELD(Resource, kind)
    MEMORY_FIELD(Resource, slots);
  py::class_<ResourceDemand>(m, "ResourceDemand")
    .def(py::init<>())
    MEMORY_FIELD(ResourceDemand, resource)
    MEMORY_FIELD(ResourceDemand, service)
    MEMORY_FIELD(ResourceDemand, amount)
    MEMORY_FIELD(ResourceDemand, per_invocation_bytes);
  py::class_<Route>(m, "Route")
    .def(py::init<>())
    MEMORY_FIELD(Route, id)
    MEMORY_FIELD(Route, agent)
    MEMORY_FIELD(Route, controller)
    MEMORY_FIELD(Route, source)
    MEMORY_FIELD(Route, target)
    MEMORY_FIELD(Route, demands);
  py::class_<Method>(m, "Method")
    .def(py::init<>())
    MEMORY_FIELD(Method, id)
    MEMORY_FIELD(Method, request_executor)
    MEMORY_FIELD(Method, bandwidth)
    MEMORY_FIELD(Method, startup_us)
    MEMORY_FIELD(Method, registration_bits)
    MEMORY_FIELD(Method, routes);
  py::class_<ResourceGraph>(m, "ResourceGraph")
    .def(py::init<>())
    MEMORY_FIELD(ResourceGraph, agents)
    MEMORY_FIELD(ResourceGraph, pools)
    MEMORY_FIELD(ResourceGraph, access)
    MEMORY_FIELD(ResourceGraph, resources)
    MEMORY_FIELD(ResourceGraph, methods)
    MEMORY_FIELD(ResourceGraph, domains)
    MEMORY_FIELD(ResourceGraph, capacity_groups);
  py::class_<ModelBinding>(m, "ModelBinding")
    .def(py::init<>())
    MEMORY_FIELD(ModelBinding, target_kind)
    MEMORY_FIELD(ModelBinding, target)
    MEMORY_FIELD(ModelBinding, profile);
  py::class_<CalibrationCase>(m, "CalibrationCase")
    .def(py::init<>())
    MEMORY_FIELD(CalibrationCase, method)
    MEMORY_FIELD(CalibrationCase, route)
    MEMORY_FIELD(CalibrationCase, executor)
    MEMORY_FIELD(CalibrationCase, operation)
    MEMORY_FIELD(CalibrationCase, profile)
    MEMORY_FIELD(CalibrationCase, sample);
  py::class_<PlacementPlan>(m, "PlacementPlan")
    .def(py::init<>())
    MEMORY_FIELD(PlacementPlan, address_space)
    MEMORY_FIELD(PlacementPlan, stripe_bytes)
    MEMORY_FIELD(PlacementPlan, weights)
    MEMORY_FIELD(PlacementPlan, cycle)
    MEMORY_FIELD(PlacementPlan, phase);
  py::class_<PlacedSegment>(m, "PlacedSegment")
    .def(py::init<>())
    MEMORY_FIELD(PlacedSegment, offset_bytes)
    MEMORY_FIELD(PlacedSegment, size_bytes)
    MEMORY_FIELD(PlacedSegment, pool);
  py::class_<Activity>(m, "Activity")
    .def(py::init<>())
    MEMORY_FIELD(Activity, id)
    MEMORY_FIELD(Activity, method)
    MEMORY_FIELD(Activity, route)
    MEMORY_FIELD(Activity, payload_bytes)
    MEMORY_FIELD(Activity, dependencies)
    MEMORY_FIELD(Activity, demands)
    MEMORY_FIELD(Activity, fixed_us)
    MEMORY_FIELD(Activity, executor);
  py::class_<CostPlan>(m, "CostPlan")
    .def(py::init<>())
    MEMORY_FIELD(CostPlan, activities);
  py::class_<PoolBudget>(m, "PoolBudget")
    .def(py::init<>())
    MEMORY_FIELD(PoolBudget, capacity_group)
    MEMORY_FIELD(PoolBudget, system_available_bytes)
    MEMORY_FIELD(PoolBudget, hard_limit_bytes)
    MEMORY_FIELD(PoolBudget, committed_bytes)
    MEMORY_FIELD(PoolBudget, live_bytes)
    MEMORY_FIELD(PoolBudget, idle_available_bytes)
    MEMORY_FIELD(PoolBudget, idle_reserved_bytes)
    MEMORY_FIELD(PoolBudget, retired_bytes)
    MEMORY_FIELD(PoolBudget, unbacked_reserved_bytes)
    MEMORY_FIELD(PoolBudget, largest_cached_block_bytes)
    MEMORY_FIELD(PoolBudget, shared_remaining_bytes);
  py::class_<BudgetConstraint>(m, "BudgetConstraint")
    .def(py::init<>())
    MEMORY_FIELD(BudgetConstraint, id)
    MEMORY_FIELD(BudgetConstraint, capacity_groups)
    MEMORY_FIELD(BudgetConstraint, remaining_bytes);
  py::class_<PoolAvailability>(m, "PoolAvailability")
    .def(py::init<>())
    MEMORY_FIELD(PoolAvailability, pool)
    MEMORY_FIELD(PoolAvailability, capacity_group)
    MEMORY_FIELD(PoolAvailability, allocatable_bytes)
    MEMORY_FIELD(PoolAvailability, largest_cached_block_bytes)
    MEMORY_FIELD(PoolAvailability, shared_constraints);
  py::class_<BudgetSnapshot>(m, "BudgetSnapshot")
    .def(py::init<>())
    MEMORY_FIELD(BudgetSnapshot, epoch)
    MEMORY_FIELD(BudgetSnapshot, groups)
    MEMORY_FIELD(BudgetSnapshot, constraints)
    MEMORY_FIELD(BudgetSnapshot, pools);
  py::class_<RuntimeState>(m, "RuntimeState")
    .def(py::init<>())
    MEMORY_FIELD(RuntimeState, budgets)
    MEMORY_FIELD(RuntimeState, active)
    MEMORY_FIELD(RuntimeState, registrations);
  py::class_<ResourceView>(m, "ResourceView")
    .def(py::init<>())
    MEMORY_FIELD(ResourceView, graph)
    MEMORY_FIELD(ResourceView, executable_agents)
    MEMORY_FIELD(ResourceView, allocatable_pools)
    MEMORY_FIELD(ResourceView, accessible_pools)
    MEMORY_FIELD(ResourceView, execution_slots)
    MEMORY_FIELD(ResourceView, epoch);
  py::class_<AllocationInfo>(m, "AllocationInfo")
    .def(py::init<>())
    MEMORY_FIELD(AllocationInfo, id)
    MEMORY_FIELD(AllocationInfo, address_space)
    MEMORY_FIELD(AllocationInfo, requested_bytes)
    MEMORY_FIELD(AllocationInfo, mapped_bytes)
    MEMORY_FIELD(AllocationInfo, page_bytes)
    MEMORY_FIELD(AllocationInfo, actual_page_sizes_bytes)
    MEMORY_FIELD(AllocationInfo, segments)
    MEMORY_FIELD(AllocationInfo, shareable)
    MEMORY_FIELD(AllocationInfo, remappable)
    MEMORY_FIELD(AllocationInfo, generation)
    MEMORY_FIELD(AllocationInfo, mode);
  py::class_<ResolvedPlacement>(m, "ResolvedPlacement")
    .def(py::init<>())
    MEMORY_FIELD(ResolvedPlacement, key)
    MEMORY_FIELD(ResolvedPlacement, address_space)
    MEMORY_FIELD(ResolvedPlacement, page_bytes)
    MEMORY_FIELD(ResolvedPlacement, segments);
  py::class_<VirtualRegion>(m, "VirtualRegion")
    .def(py::init<>())
    MEMORY_FIELD(VirtualRegion, id)
    MEMORY_FIELD(VirtualRegion, address_space)
    MEMORY_FIELD(VirtualRegion, bytes)
    MEMORY_FIELD(VirtualRegion, page_bytes)
    MEMORY_FIELD(VirtualRegion, base_alignment_bytes);
  py::class_<SharingContract>(m, "SharingContract")
    .def(py::init<>())
    MEMORY_FIELD(SharingContract, id)
    MEMORY_FIELD(SharingContract, participants)
    MEMORY_FIELD(SharingContract, producers)
    MEMORY_FIELD(SharingContract, consumers)
    MEMORY_FIELD(SharingContract, protocol);
  py::class_<AddressSlot>(m, "AddressSlot")
    .def(py::init<>())
    MEMORY_FIELD(AddressSlot, id)
    MEMORY_FIELD(AddressSlot, region)
    MEMORY_FIELD(AddressSlot, offset_bytes)
    MEMORY_FIELD(AddressSlot, size_bytes)
    MEMORY_FIELD(AddressSlot, placement)
    MEMORY_FIELD(AddressSlot, sharing_contract);
  py::class_<AddressLayout>(m, "AddressLayout")
    .def(py::init<>())
    MEMORY_FIELD(AddressLayout, regions)
    MEMORY_FIELD(AddressLayout, slots);
  py::class_<BackingRequest>(m, "BackingRequest")
    .def(py::init<>())
    MEMORY_FIELD(BackingRequest, id)
    MEMORY_FIELD(BackingRequest, owner)
    MEMORY_FIELD(BackingRequest, pool)
    MEMORY_FIELD(BackingRequest, bytes)
    MEMORY_FIELD(BackingRequest, page_bytes)
    MEMORY_FIELD(BackingRequest, mode)
    MEMORY_FIELD(BackingRequest, export_to);
  py::class_<PageMapping>(m, "PageMapping")
    .def(py::init<>())
    MEMORY_FIELD(PageMapping, region)
    MEMORY_FIELD(PageMapping, virtual_offset_bytes)
    MEMORY_FIELD(PageMapping, backing)
    MEMORY_FIELD(PageMapping, backing_offset_bytes)
    MEMORY_FIELD(PageMapping, size_bytes)
    MEMORY_FIELD(PageMapping, access);
  py::class_<MemoryRecipe>(m, "MemoryRecipe")
    .def(py::init<>())
    MEMORY_FIELD(MemoryRecipe, participant)
    MEMORY_FIELD(MemoryRecipe, layout_id)
    MEMORY_FIELD(MemoryRecipe, layout_generation)
    MEMORY_FIELD(MemoryRecipe, sharing_contracts)
    MEMORY_FIELD(MemoryRecipe, addresses)
    MEMORY_FIELD(MemoryRecipe, backings)
    MEMORY_FIELD(MemoryRecipe, mappings);
  py::class_<ParticipantLayout>(m, "ParticipantLayout")
    .def(py::init<>())
    MEMORY_FIELD(ParticipantLayout, participant)
    MEMORY_FIELD(ParticipantLayout, addresses)
    MEMORY_FIELD(ParticipantLayout, mappings);
  py::class_<DistributedLayout>(m, "DistributedLayout")
    .def(py::init<>())
    MEMORY_FIELD(DistributedLayout, id)
    MEMORY_FIELD(DistributedLayout, generation)
    MEMORY_FIELD(DistributedLayout, sharing_contracts)
    MEMORY_FIELD(DistributedLayout, backings)
    MEMORY_FIELD(DistributedLayout, participants);
  py::class_<SharedRange>(m, "SharedRange")
    .def(py::init<>())
    MEMORY_FIELD(SharedRange, backing)
    MEMORY_FIELD(SharedRange, offset_bytes)
    MEMORY_FIELD(SharedRange, size_bytes)
    MEMORY_FIELD(SharedRange, sharing_contract)
    MEMORY_FIELD(SharedRange, layout_generation)
    MEMORY_FIELD(SharedRange, data_epoch);
  py::class_<TransferEndpoint>(m, "TransferEndpoint")
    .def(py::init<>())
    MEMORY_FIELD(TransferEndpoint, address_space)
    MEMORY_FIELD(TransferEndpoint, segments);
  py::class_<AccessQuery>(m, "AccessQuery")
    .def(py::init<>())
    MEMORY_FIELD(AccessQuery, agent)
    MEMORY_FIELD(AccessQuery, memory)
    MEMORY_FIELD(AccessQuery, access)
    MEMORY_FIELD(AccessQuery, payload_bytes)
    MEMORY_FIELD(AccessQuery, method);
  py::class_<TransferQuery>(m, "TransferQuery")
    .def(py::init<>())
    MEMORY_FIELD(TransferQuery, source)
    MEMORY_FIELD(TransferQuery, target)
    MEMORY_FIELD(TransferQuery, method)
    MEMORY_FIELD(TransferQuery, executor)
    MEMORY_FIELD(TransferQuery, payload_bytes)
    MEMORY_FIELD(TransferQuery, satisfied_prerequisites);
  py::class_<TransferCandidate>(m, "TransferCandidate")
    .def(py::init<>())
    MEMORY_FIELD(TransferCandidate, method)
    MEMORY_FIELD(TransferCandidate, routes)
    MEMORY_FIELD(TransferCandidate, prerequisites);
  py::class_<CostContext>(m, "CostContext")
    .def(py::init<>())
    MEMORY_FIELD(CostContext, view_epoch)
    MEMORY_FIELD(CostContext, active)
    MEMORY_FIELD(CostContext, execution_slots)
    MEMORY_FIELD(CostContext, include_preparation);
  py::class_<CostEstimate>(m, "CostEstimate")
    .def(py::init<>())
    MEMORY_FIELD(CostEstimate, preparation_us)
    MEMORY_FIELD(CostEstimate, service_us)
    MEMORY_FIELD(CostEstimate, completion_us)
    MEMORY_FIELD(CostEstimate, total_us)
    MEMORY_FIELD(CostEstimate, lower_us)
    MEMORY_FIELD(CostEstimate, upper_us)
    MEMORY_FIELD(CostEstimate, quality)
    MEMORY_FIELD(CostEstimate, demands)
    MEMORY_FIELD(CostEstimate, missing_facts);
  py::class_<ByteRangeRef>(m, "ByteRangeRef")
    .def(py::init<>())
    MEMORY_FIELD(ByteRangeRef, kind)
    MEMORY_FIELD(ByteRangeRef, memory)
    MEMORY_FIELD(ByteRangeRef, offset_bytes)
    MEMORY_FIELD(ByteRangeRef, size_bytes);
  py::class_<MemoryCommand>(m, "MemoryCommand")
    .def(py::init<>())
    MEMORY_FIELD(MemoryCommand, id)
    MEMORY_FIELD(MemoryCommand, opcode)
    MEMORY_FIELD(MemoryCommand, source)
    MEMORY_FIELD(MemoryCommand, target)
    MEMORY_FIELD(MemoryCommand, fill_byte)
    MEMORY_FIELD(MemoryCommand, method)
    MEMORY_FIELD(MemoryCommand, route)
    MEMORY_FIELD(MemoryCommand, queue)
    MEMORY_FIELD(MemoryCommand, dependencies)
    MEMORY_FIELD(MemoryCommand, wait_ports);
  py::class_<MemoryCommandPlan>(m, "MemoryCommandPlan")
    .def(py::init<>())
    MEMORY_FIELD(MemoryCommandPlan, commands);
  py::class_<InvocationStamp>(m, "InvocationStamp")
    .def(py::init<>())
    MEMORY_FIELD(InvocationStamp, id)
    MEMORY_FIELD(InvocationStamp, binding_generation);
  py::class_<ResolvedMemorySpan>(m, "ResolvedMemorySpan")
    .def(py::init<>())
    MEMORY_FIELD(ResolvedMemorySpan, offset_bytes)
    MEMORY_FIELD(ResolvedMemorySpan, size_bytes)
    MEMORY_FIELD(ResolvedMemorySpan, physical_identity)
    MEMORY_FIELD(ResolvedMemorySpan, physical_offset_bytes)
    MEMORY_FIELD(ResolvedMemorySpan, access)
    MEMORY_FIELD(ResolvedMemorySpan, pool);
  py::class_<MemoryDescriptor>(m, "MemoryDescriptor")
    .def(py::init<>())
    MEMORY_FIELD(MemoryDescriptor, kind)
    MEMORY_FIELD(MemoryDescriptor, symbol)
    MEMORY_FIELD(MemoryDescriptor, address_space)
    MEMORY_FIELD(MemoryDescriptor, size_bytes)
    MEMORY_FIELD(MemoryDescriptor, generation)
    MEMORY_FIELD(MemoryDescriptor, spans);
  py::class_<OperandEnvironment>(m, "OperandEnvironment")
    .def(py::init<>())
    MEMORY_FIELD(OperandEnvironment, invocation)
    MEMORY_FIELD(OperandEnvironment, memories);
  py::class_<Error>(m, "Error")
    .def(py::init<>())
    MEMORY_FIELD(Error, code)
    MEMORY_FIELD(Error, message);
  py::class_<Scope>(m, "Scope")
    .def(py::init<>())
    MEMORY_FIELD(Scope, executable_agents)
    MEMORY_FIELD(Scope, allocatable_pools)
    MEMORY_FIELD(Scope, accessible_pools)
    MEMORY_FIELD(Scope, execution_slots);
  py::class_<BackendDescriptor>(m, "BackendDescriptor")
    .def(py::init<>())
    MEMORY_FIELD(BackendDescriptor, object)
    MEMORY_FIELD(BackendDescriptor, backend)
    MEMORY_FIELD(BackendDescriptor, labels);
  py::class_<Limit>(m, "Limit")
    .def(py::init<>())
    MEMORY_FIELD(Limit, bytes_or_capacity_fraction);
  py::class_<Preallocate>(m, "Preallocate")
    .def(py::init<>())
    MEMORY_FIELD(Preallocate, bytes_or_limit_fraction);
  py::class_<ArenaOptions>(m, "ArenaOptions")
    .def(py::init<>())
    MEMORY_FIELD(ArenaOptions, domain)
    MEMORY_FIELD(ArenaOptions, allowed_pools)
    MEMORY_FIELD(ArenaOptions, hard_limit)
    MEMORY_FIELD(ArenaOptions, initial)
    MEMORY_FIELD(ArenaOptions, grow_by_bytes)
    MEMORY_FIELD(ArenaOptions, retain_idle_bytes)
    MEMORY_FIELD(ArenaOptions, page_bytes)
    MEMORY_FIELD(ArenaOptions, backing)
    MEMORY_FIELD(ArenaOptions, allow_growth);
  py::class_<AllocateRequest>(m, "AllocateRequest")
    .def(py::init<>())
    MEMORY_FIELD(AllocateRequest, bytes)
    MEMORY_FIELD(AllocateRequest, preferred_pool)
    MEMORY_FIELD(AllocateRequest, allow_domain_fallback);
  py::class_<CompletionToken>(m, "CompletionToken")
    .def(py::init<>())
    MEMORY_FIELD(CompletionToken, id)
    MEMORY_FIELD(CompletionToken, invocation)
    MEMORY_FIELD(CompletionToken, binding_generation);
  py::class_<QuiescenceToken>(m, "QuiescenceToken")
    .def(py::init<>())
    MEMORY_FIELD(QuiescenceToken, id);
  py::class_<EventStatus>(m, "EventStatus")
    .def(py::init<>())
    MEMORY_FIELD(EventStatus, state)
    MEMORY_FIELD(EventStatus, accesses_stopped)
    MEMORY_FIELD(EventStatus, error);
  py::class_<ProgressCapabilities>(m, "ProgressCapabilities")
    .def(py::init<>())
    MEMORY_FIELD(ProgressCapabilities, mode)
    MEMORY_FIELD(ProgressCapabilities, progress_agent)
    MEMORY_FIELD(ProgressCapabilities, max_inflight)
    MEMORY_FIELD(ProgressCapabilities, chunk_completion)
    MEMORY_FIELD(ProgressCapabilities, concurrent_invocations);
  py::class_<EventBinding>(m, "EventBinding")
    .def(py::init<>())
    MEMORY_FIELD(EventBinding, port)
    MEMORY_FIELD(EventBinding, event);
  py::class_<Submission>(m, "Submission")
    .def(py::init<>())
    MEMORY_FIELD(Submission, accepted_commands)
    MEMORY_FIELD(Submission, submission_error)
    MEMORY_FIELD(Submission, source_reusable)
    MEMORY_FIELD(Submission, target_visible)
    MEMORY_FIELD(Submission, finished);
  py::class_<PhysicalAllocationRecord>(m, "PhysicalAllocationRecord")
    .def(py::init<>())
    MEMORY_FIELD(PhysicalAllocationRecord, identity)
    MEMORY_FIELD(PhysicalAllocationRecord, owner)
    MEMORY_FIELD(PhysicalAllocationRecord, capacity_group)
    MEMORY_FIELD(PhysicalAllocationRecord, committed_bytes)
    MEMORY_FIELD(PhysicalAllocationRecord, placement)
    MEMORY_FIELD(PhysicalAllocationRecord, charged_to_this_runtime);
  py::class_<ResidencyObservation>(m, "ResidencyObservation")
    .def(py::init<>())
    MEMORY_FIELD(ResidencyObservation, physical_identity)
    MEMORY_FIELD(ResidencyObservation, resident_bytes)
    MEMORY_FIELD(ResidencyObservation, observed_placement)
    MEMORY_FIELD(ResidencyObservation, complete)
    MEMORY_FIELD(ResidencyObservation, sample_epoch);
  py::class_<MemoryObservation>(m, "MemoryObservation")
    .def(py::init<>())
    MEMORY_FIELD(MemoryObservation, allocations)
    MEMORY_FIELD(MemoryObservation, residency)
    MEMORY_FIELD(MemoryObservation, unmanaged_committed_bytes);
  py::class_<MapRequest>(m, "MapRequest")
    .def(py::init<>())
    MEMORY_FIELD(MapRequest, backing_offset_bytes)
    MEMORY_FIELD(MapRequest, target_offset_bytes)
    MEMORY_FIELD(MapRequest, size_bytes);
  py::class_<MappingCapabilities>(m, "MappingCapabilities")
    .def(py::init<>())
    MEMORY_FIELD(MappingCapabilities, page_bytes)
    MEMORY_FIELD(MappingCapabilities, reserve_without_backing)
    MEMORY_FIELD(MappingCapabilities, shareable)
    MEMORY_FIELD(MappingCapabilities, remappable)
    MEMORY_FIELD(MappingCapabilities, stable_address_rebind)
    MEMORY_FIELD(MappingCapabilities, partial_backing_map)
    MEMORY_FIELD(MappingCapabilities, max_backing_bytes)
    MEMORY_FIELD(MappingCapabilities, max_region_bytes)
    MEMORY_FIELD(MappingCapabilities, max_mapping_bytes)
    MEMORY_FIELD(MappingCapabilities, multiple_mappings);
  py::enum_<CapacityLimitKind>(m, "CapacityLimitKind")
    .value("capacity_group", CapacityLimitKind::capacity_group)
    .value("pool", CapacityLimitKind::pool)
    .value("constraint", CapacityLimitKind::constraint);
  py::class_<CapacityShortfall>(m, "CapacityShortfall")
    .def(py::init<>())
    MEMORY_FIELD(CapacityShortfall, kind)
    MEMORY_FIELD(CapacityShortfall, id)
    MEMORY_FIELD(CapacityShortfall, requested_bytes)
    MEMORY_FIELD(CapacityShortfall, available_bytes);
  py::class_<PlanEstimate>(m, "PlanEstimate")
    .def(py::init<>())
    MEMORY_FIELD(PlanEstimate, peak_bytes_by_capacity_group)
    MEMORY_FIELD(PlanEstimate, cached_reuse_bytes_by_capacity_group)
    MEMORY_FIELD(PlanEstimate, new_backing_bytes_by_capacity_group)
    MEMORY_FIELD(PlanEstimate, feasible)
    MEMORY_FIELD(PlanEstimate, shortfalls);
  py::class_<HardwareExpansion>(m, "HardwareExpansion")
    .def(py::init<>())
    MEMORY_FIELD(HardwareExpansion, name)
    MEMORY_FIELD(HardwareExpansion, graph)
    MEMORY_FIELD(HardwareExpansion, binding_requests)
    .def_property("requirements", [] (const HardwareExpansion& value) { return facts_to_python(value.requirements); },
                  [] (HardwareExpansion& value, const py::dict& facts) { value.requirements = facts_from_python(facts); });
  py::class_<HardwareBinding>(m, "HardwareBinding")
    .def(py::init<>())
    MEMORY_FIELD(HardwareBinding, capacity_bytes)
    MEMORY_FIELD(HardwareBinding, address_space)
    MEMORY_FIELD(HardwareBinding, granularity_bytes)
    MEMORY_FIELD(HardwareBinding, page_sizes_bytes)
    MEMORY_FIELD(HardwareBinding, capabilities)
    MEMORY_FIELD(HardwareBinding, capacity_group)
    MEMORY_FIELD(HardwareBinding, capacity_group_bytes);
  py::class_<AccessDiscovery>(m, "AccessDiscovery")
    .def(py::init<>())
    MEMORY_FIELD(AccessDiscovery, mapping)
    MEMORY_FIELD(AccessDiscovery, prerequisites);
  py::class_<HardwareInventory>(m, "HardwareInventory")
    .def(py::init<>())
    .def_property("facts", [] (const HardwareInventory& value) { return facts_to_python(value.facts); },
                  [] (HardwareInventory& value, const py::dict& facts) { value.facts = facts_from_python(facts); })
    MEMORY_FIELD(HardwareInventory, bindings)
    MEMORY_FIELD(HardwareInventory, access);
#undef MEMORY_FIELD
  // Register the cost interface once so MemoryRuntime::costs() can return its
  // existing model by reference without any second Python model implementation.
  py::class_<MemoryCostModel>(m, "MemoryCostModel")
    .def("candidates", [](const MemoryCostModel& self, const ResourceView& view,
                           const TransferEndpoint& source, const TransferEndpoint& target) {
      return unwrap(self.candidates(view, source, target));
    }, py::arg("view"), py::arg("source"), py::arg("target"))
    .def("estimate", [](const MemoryCostModel& self, const ResourceView& view,
                         const TransferQuery& query, const CostContext& context) {
      return unwrap(self.estimate(view, query, context));
    }, py::arg("view"), py::arg("query"), py::arg("context"))
    .def("estimate_access", [](const MemoryCostModel& self, const ResourceView& view,
                                const AccessQuery& query, const CostContext& context) {
      return unwrap(self.estimate_access(view, query, context));
    }, py::arg("view"), py::arg("query"), py::arg("context"))
    .def("estimate_schedule", [](const MemoryCostModel& self, const ResourceView& view,
                                  const CostPlan& plan, const CostContext& context) {
      return unwrap(self.estimate_schedule(view, plan, context));
    }, py::arg("view"), py::arg("plan"), py::arg("context"));
  py::class_<ResourceModel, MemoryCostModel>(m, "ResourceModel")
    .def(py::init<ResourceGraph>(), py::arg("graph"))
    // Return a value: exposing the const C++ graph through read/write-bound
    // children would let Python mutate an immutable model without validation.
    .def_property_readonly("graph", [](const ResourceModel& self) { return self.graph(); })
    .def("view", [](const ResourceModel& self, const Scope& scope, std::uint64_t epoch) {
      return unwrap(self.view(scope, epoch));
    }, py::arg("scope"), py::arg("epoch") = 0)
    .def("resolve_placement", [](const ResourceModel& self, const ResourceView& view,
                                  const PlacementPlan& plan, std::uint64_t bytes) {
      return unwrap(self.resolve_placement(view, plan, bytes));
    }, py::arg("view"), py::arg("plan"), py::arg("bytes"))
    .def("compatible_overlap", [](const ResourceModel& self, const ResolvedPlacement& left,
                                   std::uint64_t left_offset, const ResolvedPlacement& right,
                                   std::uint64_t right_offset, std::uint64_t bytes) {
      return unwrap(self.compatible_overlap(left, left_offset, right, right_offset, bytes));
    }, py::arg("left"), py::arg("left_offset"), py::arg("right"), py::arg("right_offset"), py::arg("bytes"))
    .def("validate_recipe", [](const ResourceModel& self, const ResourceView& view, const MemoryRecipe& recipe) {
      unwrap(self.validate_recipe(view, recipe));
    }, py::arg("view"), py::arg("recipe"))
    .def("describe_operands", [](const ResourceModel& self, const ResourceView& view,
                                  const MemoryRecipe& recipe, const std::vector<MemoryDescriptor>& external) {
      return unwrap(self.describe_operands(view, recipe, external));
    }, py::arg("view"), py::arg("recipe"), py::arg("external") = std::vector<MemoryDescriptor>{})
    .def("validate_commands", [](const ResourceModel& self, const ResourceView& view,
                                  const MemoryCommandPlan& plan, const OperandEnvironment& environment) {
      unwrap(self.validate_commands(view, plan, environment));
    }, py::arg("view"), py::arg("plan"), py::arg("environment"))
    .def("cost_plan", [](const ResourceModel& self, const ResourceView& view,
                          const MemoryCommandPlan& plan, const OperandEnvironment& environment) {
      return unwrap(self.cost_plan(view, plan, environment));
    }, py::arg("view"), py::arg("plan"), py::arg("environment"));

  m.def("page_charge", &page_charge, py::arg("bytes"), py::arg("quantum"));
  m.def("budget_available", [](const PoolBudget& budget) { return unwrap(budget_available(budget)); },
        py::arg("budget"));
  m.def("parse_memory_size", [](const std::string& text) { return unwrap(parse_memory_size(text)); }, py::arg("text"));
  m.def("expand_hardware_template", [](const std::string& text, const RuntimeCounts& counts) {
    return unwrap(expand_hardware_template(text, counts));
  }, py::arg("toml"), py::arg("counts") = RuntimeCounts{});
  m.def("bind_hardware_inventory", [](const HardwareExpansion& expansion, const HardwareInventory& inventory) {
    return unwrap(bind_hardware_inventory(expansion, inventory));
  }, py::arg("expansion"), py::arg("inventory"));
  m.def("compile_hardware_template", [](const std::string& text, const RuntimeCounts& counts,
                                         const HardwareInventory& inventory) {
    return unwrap(compile_hardware_template(text, counts, inventory));
  }, py::arg("toml"), py::arg("counts"), py::arg("inventory"));
  m.def("recipe_to_toml", [](const MemoryRecipe& value) { return unwrap(recipe_to_toml(value)); }, py::arg("recipe"));
  m.def("recipe_from_toml", [](const std::string& text) { return unwrap(recipe_from_toml(text)); }, py::arg("toml"));
  m.def("layout_to_toml", [](const DistributedLayout& value) { return unwrap(layout_to_toml(value)); }, py::arg("layout"));
  m.def("layout_from_toml", [](const std::string& text) { return unwrap(layout_from_toml(text)); }, py::arg("toml"));
}
