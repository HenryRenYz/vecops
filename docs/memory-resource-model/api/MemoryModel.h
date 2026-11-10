// SPDX-FileCopyrightText: 2026 HenryRenYz and vecops contributors
// SPDX-License-Identifier: MIT
#pragma once
// Design-field excerpt, not a separate API specification or installed header.
// The authoritative implementation is include/vecops/memory/Model.h; see also
// ResourceModel.h and LinuxRuntime.h. This excerpt supports the design document.
// Bandwidth is GiB/s; byte fields are exact integers.
#include <any>
#include <cstdint>
#include <map>
#include <memory>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <variant>
#include <utility>
#include <vector>

namespace vecops::memory::model {
// BEGIN DOCUMENTED TYPES
enum class AccessMode { read, write };
enum class MappingMode { unknown, none, direct, requires_mapping };
enum class SharingMode { shared, read_write };
enum class DemandKind { read, write, traffic, occupancy };
enum class ResourceKind { bandwidth, execution_slots };
enum class ModelTargetKind { access, method, resource };
using BindingRequests = std::map<std::string, std::string>;
using BackendBindings = std::map<std::string, std::any>;
struct CapacityGroup {
    std::string id;
    std::optional<std::uint64_t> capacity_bytes;
};
struct AllocationDomain {
    std::string id;
    std::string rule;
    std::vector<std::string> members;
};

struct Agent {
    std::string id;
    std::vector<std::string> capabilities;
    std::optional<std::string> address_space;
};

struct MemoryPool {
    std::string id;
    std::optional<std::uint64_t> capacity_bytes;
    std::optional<std::string> address_space;
    std::optional<std::uint64_t> granularity_bytes;
    std::vector<std::uint64_t> page_sizes_bytes;
    std::optional<std::string> read_service;
    std::optional<std::string> write_service;
    std::string capacity_group;
};

struct AccessConnection {
    std::string id;
    std::string rule;
    std::string agent;
    std::string pool;
    std::vector<AccessMode> access;
    std::optional<std::map<std::string, double>> bandwidth;
    std::optional<double> latency_ns;
    MappingMode mapping;
    std::vector<std::string> prerequisites;
    std::optional<std::string> read_service;
    std::optional<std::string> write_service;
    std::vector<std::string> read_fabric;
    std::vector<std::string> write_fabric;
};

struct Resource {
    std::string id;
    std::string rule;
    std::optional<double> bandwidth;
    std::optional<double> read;
    std::optional<double> write;
    SharingMode sharing;
    ResourceKind kind;
    std::optional<std::uint64_t> slots;
};

struct ResourceDemand {
    std::string resource;
    DemandKind service;
    double amount; // traffic: GiB work/GiB payload; occupancy: fixed slot count
    std::uint64_t per_invocation_bytes; // bandwidth demands only, separate from occupancy
};

struct Route {
    std::string id;
    std::optional<std::string> agent;
    std::optional<std::string> controller;
    std::string source;
    std::string target;
    std::vector<ResourceDemand> demands;
};

struct Method {
    std::string id;
    bool request_executor;
    std::optional<double> bandwidth;
    std::optional<double> startup_us;
    std::optional<std::uint64_t> registration_bits;
    std::vector<Route> routes;
};

struct ResourceGraph {
    std::vector<Agent> agents;
    std::vector<MemoryPool> pools;
    std::vector<AccessConnection> access;
    std::vector<Resource> resources;
    std::vector<Method> methods;
    std::vector<AllocationDomain> domains;
    std::vector<CapacityGroup> capacity_groups;
};

struct ModelBinding {
    ModelTargetKind target_kind;
    std::string target;
    std::string profile;
};

struct CalibrationCase {
    std::string method;
    std::string route;
    std::optional<std::string> executor;
    std::string operation;
    std::string profile;
    std::uint64_t sample;
};

struct PerformanceDatabase {
    std::vector<ModelBinding> bindings;
    std::map<std::string, std::map<std::string, std::any>> profiles;
    std::map<std::string, std::map<std::string, std::any>> method_models;
    std::vector<CalibrationCase> cases; // topology predicates already resolved
};
// END DOCUMENTED TYPES

// BEGIN PLANNING TYPES
struct PlacementPlan {
    std::string address_space;
    std::uint64_t stripe_bytes;
    std::map<std::string, std::uint64_t> weights;
    std::vector<std::string> cycle;
    std::uint64_t phase;
};
struct PlacedSegment {
    std::uint64_t offset_bytes;
    std::uint64_t size_bytes;
    std::string pool;
};
struct Activity {
    std::string id;
    std::string method;
    std::optional<std::string> route;
    std::uint64_t payload_bytes;
    std::vector<std::string> dependencies;
    std::vector<ResourceDemand> demands;
    double fixed_us; // non-bandwidth service: compute, registration, synchronization
    std::optional<std::string> executor;
};
struct CostPlan {
    std::vector<Activity> activities;
};
struct PoolBudget {
    std::string capacity_group;
    std::uint64_t system_available_bytes;
    std::uint64_t hard_limit_bytes;
    std::uint64_t committed_bytes;
    std::uint64_t live_bytes;
    std::uint64_t idle_available_bytes;
    std::uint64_t idle_reserved_bytes;
    std::uint64_t retired_bytes;
    std::uint64_t unbacked_reserved_bytes;
    std::uint64_t largest_cached_block_bytes;
    // Local ownership counters; shared_remaining covers all participants.
    std::optional<std::uint64_t> shared_remaining_bytes = std::nullopt;
};
struct BudgetConstraint {
    std::string id;
    std::vector<std::string> capacity_groups;
    std::uint64_t remaining_bytes;
};
struct PoolAvailability {
    std::string pool;
    std::string capacity_group;
    std::uint64_t allocatable_bytes;
    std::uint64_t largest_cached_block_bytes;
    std::vector<std::string> shared_constraints;
};
struct BudgetSnapshot {
    std::uint64_t epoch;
    std::vector<PoolBudget> groups;
    std::vector<BudgetConstraint> constraints;
    std::vector<PoolAvailability> pools;
};
struct RuntimeState {
    BudgetSnapshot budgets;
    std::vector<Activity> active;
    std::map<std::string, std::string> registrations;
};
struct ResourceView {
    ResourceGraph graph;
    std::vector<std::string> executable_agents;
    std::vector<std::string> allocatable_pools;
    std::vector<std::string> accessible_pools; // may include import/protocol-only pools
    std::map<std::string, std::uint64_t> execution_slots;
    std::uint64_t epoch;
};
enum class BackingMode { private_memory, shareable_remappable };
struct AllocationInfo {
    std::string id;
    std::string address_space;
    std::uint64_t requested_bytes;
    std::uint64_t mapped_bytes;
    std::uint64_t page_bytes;
    std::vector<std::uint64_t> actual_page_sizes_bytes;
    std::vector<PlacedSegment> segments;
    bool shareable;
    bool remappable;
    std::uint64_t generation;
};
// A key identifies a page pattern and its mapping constraints, excluding length.
struct ResolvedPlacement {
    std::string key;
    std::string address_space;
    std::uint64_t page_bytes;
    std::vector<PlacedSegment> segments;
};
struct VirtualRegion {
    std::string id;
    std::string address_space;
    std::uint64_t bytes;
    std::uint64_t page_bytes;
    // Independent minimum absolute VA base alignment; 0 defaults to page_bytes.
    // Does not change mapping quantum or physical capacity demand.
    std::uint64_t base_alignment_bytes = 0;
};
struct SharingContract {
    std::string id;
    std::vector<std::string> participants;
    std::vector<std::string> producers;
    std::vector<std::string> consumers;
    std::string protocol; // publish/acquire/retire semantics, not a collective name
};
struct AddressSlot {
    std::string id;
    std::string region;
    std::uint64_t offset_bytes;
    std::uint64_t size_bytes;
    ResolvedPlacement placement;
    std::optional<std::string> sharing_contract;
};
struct AddressLayout {
    std::vector<VirtualRegion> regions;
    std::vector<AddressSlot> slots;
};
struct BackingRequest {
    std::string id; // globally unique within the agreed layout generation
    std::string owner;
    std::string pool;
    std::uint64_t bytes;
    std::uint64_t page_bytes;
    BackingMode mode;
    std::vector<std::string> export_to;
};
struct PageMapping {
    std::string region;
    std::uint64_t virtual_offset_bytes;
    std::string backing;
    std::uint64_t backing_offset_bytes;
    std::uint64_t size_bytes;
    std::vector<AccessMode> access;
};
// Contains no tensor/operator/lifetime concepts. Slots are opaque byte ranges.
struct MemoryRecipe {
    std::string participant;
    std::optional<std::string> layout_id;
    std::uint64_t layout_generation;
    std::vector<SharingContract> sharing_contracts;
    AddressLayout addresses;
    std::vector<BackingRequest> backings;
    std::vector<PageMapping> mappings;
};
struct ParticipantLayout {
    std::string participant;
    AddressLayout addresses;
    std::vector<PageMapping> mappings;
};
struct DistributedLayout {
    std::string id;
    std::uint64_t generation;
    std::vector<SharingContract> sharing_contracts;
    std::vector<BackingRequest> backings;
    std::vector<ParticipantLayout> participants;
};
struct SharedRange {
    std::string backing;
    std::uint64_t offset_bytes;
    std::uint64_t size_bytes;
    std::string sharing_contract;
    std::uint64_t layout_generation;
    std::uint64_t data_epoch;
};
struct TransferEndpoint {
    std::string address_space;
    std::vector<PlacedSegment> segments;
};
struct AccessQuery {
    std::string agent;
    TransferEndpoint memory;
    AccessMode access;
    std::uint64_t payload_bytes;
    std::optional<std::string> method;
};
struct TransferQuery {
    TransferEndpoint source;
    TransferEndpoint target;
    std::string method;
    std::optional<std::string> executor;
    std::uint64_t payload_bytes;
    std::vector<std::string> satisfied_prerequisites;
};
struct TransferCandidate {
    std::string method;
    std::vector<std::string> routes;
    std::vector<std::string> prerequisites;
};
struct CostContext {
    std::uint64_t view_epoch;
    std::vector<Activity> active;
    std::map<std::string, std::uint64_t> execution_slots;
    bool include_preparation;
};
enum class CostQuality { calibrated, interpolated, reference, incomplete };
struct CostEstimate {
    double preparation_us;
    double service_us;
    double completion_us;
    double total_us;
    std::optional<double> lower_us;
    std::optional<double> upper_us;
    CostQuality quality;
    std::vector<ResourceDemand> demands;
    std::vector<std::string> missing_facts;
};
enum class MemoryRefKind { slot, external };
struct ByteRangeRef {
    MemoryRefKind kind;
    std::string memory; // plan symbol in the selected namespace, never an imported-object ID
    std::uint64_t offset_bytes;
    std::uint64_t size_bytes;
};
enum class MemoryOpcode { copy, fill, register_range, unregister_range, fence };
struct MemoryCommand {
    std::string id;
    MemoryOpcode opcode;
    std::optional<ByteRangeRef> source;
    std::optional<ByteRangeRef> target;
    std::uint64_t fill_byte;
    std::string method;
    std::optional<std::string> route;
    std::string queue; // symbolic queue resolved by backend; includes executor binding
    std::vector<std::string> dependencies;
    std::vector<std::string> wait_ports; // symbolic event ports, bound per invocation
};
struct MemoryCommandPlan { std::vector<MemoryCommand> commands; };
struct InvocationStamp {
    std::string id;
    std::uint64_t binding_generation;
};
struct ResolvedMemorySpan {
    std::uint64_t offset_bytes; // relative to the bound symbol
    std::uint64_t size_bytes;
    std::optional<std::string> physical_identity; // canonical backing identity, not a VA
    std::uint64_t physical_offset_bytes;
    std::vector<AccessMode> access;
    std::optional<std::string> pool;
};
struct MemoryDescriptor {
    MemoryRefKind kind;
    std::string symbol;
    std::string address_space;
    std::uint64_t size_bytes;
    std::uint64_t generation;
    std::vector<ResolvedMemorySpan> spans; // sorted, complete, nonoverlapping coverage
};
struct OperandEnvironment {
    std::optional<InvocationStamp> invocation; // null: offline metadata, not submit authority
    std::vector<MemoryDescriptor> memories;
};
// END PLANNING TYPES

// BEGIN API TYPES
struct Error { std::string code; std::string message; };
template<class T> struct Result {
    std::variant<T, Error> state;
    explicit operator bool() const { return std::holds_alternative<T>(state); }
    T& value() & { return std::get<T>(state); }
    T value() && { return std::get<T>(std::move(state)); }
};
template<> struct Result<void> {
    std::optional<Error> error;
    explicit operator bool() const { return !error; }
};

struct Scope {
    // IDs selected by discovery after affinity / cpuset.mems / device grants.
    std::vector<std::string> executable_agents;
    std::vector<std::string> allocatable_pools;
    std::vector<std::string> accessible_pools; // may include import/protocol-only pools
    std::map<std::string, std::uint64_t> execution_slots;
};
struct BackendDescriptor {
    std::string object;
    std::string backend;
    std::map<std::string, std::string> labels; // diagnostics only: OS NUMA, CPU set…
};
struct Limit {
    // Exactly one alternative. Fraction base: unique capacity in the arena view.
    std::variant<std::uint64_t, double> bytes_or_capacity_fraction;
};
struct Preallocate {
    // Fraction base: the resolved hard limit; not machine total capacity.
    std::variant<std::uint64_t, double> bytes_or_limit_fraction;
};
struct ArenaOptions {
    std::string domain; // AllocationDomain ID, or Pool ID for a singleton arena
    std::vector<std::string> allowed_pools; // empty = all permitted domain members
    Limit hard_limit;
    Preallocate initial;
    std::uint64_t grow_by_bytes;
    std::uint64_t retain_idle_bytes;
    std::uint64_t page_bytes; // fixed Q; 0 = use resolved pool quantum
    BackingMode backing;
    bool allow_growth;
};
struct AllocateRequest {
    std::uint64_t bytes;
    std::optional<std::string> preferred_pool;
    bool allow_domain_fallback;
};
struct CompletionToken {
    std::string id;
    std::string invocation;
    std::uint64_t binding_generation;
};
struct QuiescenceToken { std::string id; }; // runtime validates fence provenance

enum class EventState { pending, succeeded, failed, cancelled };
struct EventStatus {
    EventState state;
    bool accesses_stopped; // failure/cancellation alone never authorizes reuse
    std::optional<Error> error;
};
struct ExternalEvent {
    std::string backend; // registered adapter, e.g. CUDA stream / MPI request / CPU task
    std::any handle; // process-local; never serialized
    std::shared_ptr<void> owner;
};
struct InvocationContext;
class EventService {
public:
    virtual ~EventService() = default;
    virtual Result<CompletionToken> import_event(
        const ExternalEvent&, const InvocationContext&) = 0;
    virtual Result<EventStatus> query(const CompletionToken&) const = 0;
    virtual Result<EventStatus> wait(const CompletionToken&) = 0;
    virtual Result<CompletionToken> join(
        std::span<const CompletionToken>, const InvocationContext&) = 0;
    virtual Result<void> request_cancel(const CompletionToken&) = 0;
    // Release after the last consumer; all events must be terminal and stopped.
    virtual Result<void> release(std::span<const CompletionToken>) = 0;
};
enum class ProgressMode { unknown, caller_driven, cooperative, dedicated_agent, autonomous };
struct ProgressCapabilities {
    ProgressMode mode;
    std::optional<std::string> progress_agent;
    std::uint64_t max_inflight; // 0 = unknown, never unlimited by inference
    bool chunk_completion;
    bool concurrent_invocations;
};
struct EventBinding { std::string port; CompletionToken event; };
class ImportedMemory;
struct ExternalBinding {
    std::string symbol; // invocation-local plan symbol, e.g. input; not view().id
    std::shared_ptr<ImportedMemory> memory;
};
struct InvocationContext {
    InvocationStamp stamp; // unique replay and current binding generation
    std::vector<EventBinding> events;
    std::vector<ExternalBinding> external; // no runtime-global symbol registry
};
struct Submission {
    std::vector<std::string> accepted_commands;
    std::optional<Error> submission_error; // partial acceptance is not rolled back
    CompletionToken source_reusable;
    CompletionToken target_visible;
    CompletionToken finished; // runtime retains resources until accesses have stopped
};
struct PhysicalAllocationRecord {
    std::string identity; // backend-certified physical identity, deduplicates imported aliases
    std::string owner;
    std::string capacity_group;
    std::uint64_t committed_bytes;
    std::vector<PlacedSegment> placement;
    bool charged_to_this_runtime; // false for references to another owner's backing
};
struct ResidencyObservation {
    std::string physical_identity;
    std::optional<std::uint64_t> resident_bytes; // unknown is not zero
    std::vector<PlacedSegment> observed_placement;
    bool complete;
    std::uint64_t sample_epoch;
};
struct MemoryObservation {
    std::vector<PhysicalAllocationRecord> allocations;
    std::vector<ResidencyObservation> residency;
    std::optional<std::uint64_t> unmanaged_committed_bytes;
};

class SharedBacking;
class VirtualReservation;
class MappedRange;
class PlanLease;
class PreparedAddresses;
class BackingSet;
class MemoryBinding;
class LayoutAgreement;
class Allocation {
public:
    Allocation(Allocation&&) noexcept;
    Allocation& operator=(Allocation&&) noexcept;
    ~Allocation();
    Allocation(const Allocation&) = delete;
    Allocation& operator=(const Allocation&) = delete;
    const AllocationInfo& info() const;
    // A successful pointer belongs to address_space and is aligned to page_bytes.
    Result<void*> data() const;
    Result<std::shared_ptr<SharedBacking>> export_backing() const;
private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};
class SharedBacking {
public:
    virtual ~SharedBacking() = default;
    virtual const AllocationInfo& info() const = 0;
    // Backend-specific export handle; lifetime owned by SharedBacking.
    virtual Result<std::any> export_handle() const = 0;
};
class VirtualReservation {
public:
    virtual ~VirtualReservation() = default;
    virtual std::uint64_t size_bytes() const = 0;
    virtual std::uint64_t page_bytes() const = 0;
};
class MappedRange {
public:
    virtual ~MappedRange() = default;
    virtual Result<void*> data() const = 0;
    virtual std::uint64_t generation() const = 0;
};
struct MapRequest {
    std::uint64_t backing_offset_bytes;
    std::uint64_t target_offset_bytes;
    std::uint64_t size_bytes;
};
class PlanLease {
public:
    virtual ~PlanLease() = default; // cancel unused grants; does not own mappings
};
struct VirtualAddress {
    std::string address_space;
    std::uint64_t value; // a reserved address, not permission to dereference
};
class PreparedAddresses {
public:
    virtual ~PreparedAddresses() = default;
    virtual Result<VirtualAddress> slot(const std::string& id) const = 0;
};
class LayoutAgreement {
public:
    virtual ~LayoutAgreement() = default;
    virtual const std::string& digest() const = 0;
    virtual std::uint64_t generation() const = 0;
};
class BackingSet {
public:
    virtual ~BackingSet() = default;
    virtual std::span<const AllocationInfo> owned_allocations() const = 0;
    virtual std::vector<std::string> accessible_backings() const = 0;
};
class MemoryBinding {
public:
    virtual ~MemoryBinding() = default;
    virtual Result<void*> slot(const std::string& id) const = 0;
    virtual std::uint64_t generation() const = 0;
    virtual Result<void> record_use(CompletionToken) = 0;
};
struct MappingCapabilities {
    std::uint64_t page_bytes;
    bool reserve_without_backing;
    bool shareable;
    bool remappable;
    bool stable_address_rebind;
    bool partial_backing_map; // otherwise lower backing into independent map chunks
    std::optional<std::uint64_t> max_backing_bytes;
    std::optional<std::uint64_t> max_region_bytes;
    std::optional<std::uint64_t> max_mapping_bytes;
};
struct ExternalMemoryView {
    std::string id; // import label, never a plan symbol or physical identity
    VirtualAddress address;
    std::uint64_t bytes;
    std::optional<std::string> known_pool;
    std::shared_ptr<void> owner; // lifetime token; no ownership transfer implied
    bool read_only;
    std::optional<PhysicalAllocationRecord> accounting;
};
class ImportedMemory {
public:
    virtual ~ImportedMemory() = default;
    virtual const ExternalMemoryView& view() const = 0;
    virtual Result<TransferEndpoint> endpoint() const = 0;
    virtual Result<MemoryDescriptor> describe(const std::string& symbol) const = 0;
    virtual MappingCapabilities mapping_capabilities() const = 0;
};
class MemoryCostModel {
public:
    virtual ~MemoryCostModel() = default;
    virtual Result<std::vector<TransferCandidate>> candidates(
        const ResourceView&, const TransferEndpoint&, const TransferEndpoint&) const = 0;
    virtual Result<CostEstimate> estimate(
        const ResourceView&, const TransferQuery&, const CostContext&) const = 0;
    virtual Result<CostEstimate> estimate_access(
        const ResourceView&, const AccessQuery&, const CostContext&) const = 0;
    virtual Result<CostEstimate> estimate_schedule(
        const ResourceView&, const CostPlan&, const CostContext&) const = 0;
};
class MemoryArena {
public:
    virtual ~MemoryArena() = default;
    virtual const std::string& id() const = 0;
    virtual BudgetSnapshot snapshot() const = 0;
    virtual std::uint64_t page_bytes() const = 0;
    virtual Result<Allocation> allocate(const AllocateRequest&) = 0;
    virtual Result<void> record_use(Allocation&, CompletionToken) = 0;
    virtual Result<void> release(Allocation&&, CompletionToken last_use) = 0;
    virtual Result<std::uint64_t> trim(std::uint64_t retain_idle_bytes) = 0;
};
struct PlanEstimate {
    std::map<std::string, std::uint64_t> peak_bytes_by_capacity_group;
    std::map<std::string, std::uint64_t> cached_reuse_bytes_by_capacity_group;
    std::map<std::string, std::uint64_t> new_backing_bytes_by_capacity_group;
    bool feasible;
};
class MemoryQueries {
public:
    virtual ~MemoryQueries() = default;
    virtual BudgetSnapshot snapshot(
        const ResourceView&, std::span<MemoryArena* const> caches = {}) const = 0;
    virtual Result<void> validate_recipe(const ResourceView&, const MemoryRecipe&) const = 0;
    virtual Result<PlanEstimate> estimate(
        const ResourceView&, const MemoryRecipe&,
        std::span<MemoryArena* const> caches = {}) const = 0;
    virtual const MemoryCostModel& costs() const = 0;
    virtual Result<ProgressCapabilities> progress(
        const ResourceView&, const std::string& method,
        const std::optional<std::string>& route) const = 0;
    virtual Result<OperandEnvironment> describe_operands(
        const ResourceView&, const MemoryRecipe&,
        std::span<const MemoryDescriptor> external) const = 0;
    virtual Result<void> validate_commands(
        const ResourceView&, const MemoryCommandPlan&, const OperandEnvironment&) const = 0;
    virtual Result<CostPlan> cost_plan(
        const ResourceView&, const MemoryCommandPlan&, const OperandEnvironment&) const = 0;
    virtual Result<MappingCapabilities> mapping_capabilities(
        const ResourceView&, std::span<const std::string> pools) const = 0;
    virtual Result<ResolvedPlacement> resolve_placement(
        const ResourceView&, const PlacementPlan&, std::uint64_t bytes,
        BackingMode, std::span<const AccessMode>) const = 0;
    virtual Result<bool> compatible_overlap(
        const ResolvedPlacement&, std::uint64_t left_offset,
        const ResolvedPlacement&, std::uint64_t right_offset,
        std::uint64_t bytes) const = 0;
};
class MemoryRuntime : public MemoryQueries {
public:
    virtual ~MemoryRuntime() = default;
    virtual const std::string& participant() const = 0;
    // Launch-local opaque coordinator identity; nullopt: independent accounting.
    // Compare within a host; excluded from serialized placement/cache contracts.
    virtual std::optional<std::string> capacity_authority() const = 0;
    virtual const ResourceGraph& physical_graph() const = 0;
    virtual const PerformanceDatabase& performance() const = 0;
    virtual EventService& events() = 0;
    virtual Result<MemoryObservation> observe(const ResourceView&) const = 0;
    // Actual page-placement sample; unknown pages remain gaps. No fault/migration.
    virtual Result<ResidencyObservation> observe_external(const ResourceView&, const ExternalMemoryView&) const = 0;
    virtual Result<Scope> current_scope() const = 0;
    // Refresh physical free capacity between materialization transactions.
    virtual Result<void> refresh_available() = 0;
    virtual Result<ResourceView> view(const Scope&) const = 0;
    virtual std::vector<BackendDescriptor> describe_bindings() const = 0;
    virtual std::vector<std::string> arena_ids() const = 0;
    virtual Result<std::shared_ptr<MemoryArena>> arena(const std::string& id) = 0;
    virtual Result<std::shared_ptr<MemoryArena>> create_arena(
        const ResourceView&, const ArenaOptions&) = 0;
    virtual Result<Allocation> allocate(
        const ResourceView&, const BackingRequest&) = 0;
    virtual Result<std::unique_ptr<PlanLease>> reserve(
        const ResourceView&, const MemoryRecipe&, std::uint64_t budget_epoch,
        std::span<MemoryArena* const> caches = {}) = 0;
    virtual Result<std::shared_ptr<ImportedMemory>> import_memory(
        const ResourceView&, const ExternalMemoryView&) = 0;
    virtual Result<std::shared_ptr<PreparedAddresses>> prepare_addresses(
        const ResourceView&, const AddressLayout&) = 0;
    virtual Result<std::unique_ptr<LayoutAgreement>> agree_layout(
        const DistributedLayout&) = 0;
    virtual Result<std::shared_ptr<BackingSet>> exchange_backing(
        const LayoutAgreement&, std::shared_ptr<BackingSet> owned) = 0;
    virtual Result<std::shared_ptr<BackingSet>> acquire_backing(
        const ResourceView&, const MemoryRecipe&, PlanLease&) = 0;
    virtual Result<std::unique_ptr<MemoryBinding>> bind(
        const ResourceView&, const MemoryRecipe&,
        std::shared_ptr<PreparedAddresses>, std::shared_ptr<BackingSet>,
        const LayoutAgreement* agreement = nullptr) = 0;
    virtual Result<QuiescenceToken> quiesce(
        MemoryBinding&, std::span<const CompletionToken>) = 0;
    virtual Result<void> unbind(MemoryBinding&, const QuiescenceToken&) = 0;
    virtual Result<CompletionToken> begin_shared(const SharedRange&) = 0;
    virtual Result<CompletionToken> publish_shared(
        const SharedRange&, CompletionToken producer_ready) = 0;
    virtual Result<CompletionToken> acquire_shared(const SharedRange&) = 0;
    // Completion means ALL required participants retired, not merely local free.
    virtual Result<CompletionToken> retire_shared(
        const SharedRange&, std::span<const CompletionToken> local_last_uses) = 0;
    virtual Result<QuiescenceToken> quiesce(
        MappedRange&, std::span<const CompletionToken>) = 0;
    virtual Result<std::unique_ptr<VirtualReservation>> reserve_address(
        const std::string& address_space, std::uint64_t bytes,
        std::uint64_t page_bytes) = 0;
    virtual Result<std::unique_ptr<MappedRange>> map(
        VirtualReservation&, std::shared_ptr<SharedBacking>, const MapRequest&) = 0;
    virtual Result<void> rebind(
        MappedRange&, std::shared_ptr<SharedBacking>, const MapRequest&,
        const QuiescenceToken&) = 0;
    // Resolve fresh operands from binding + invocation imports; snapshots are not authority.
    virtual Result<OperandEnvironment> resolve_operands(
        const ResourceView&, MemoryBinding&, const InvocationContext&) const = 0;
    // Result error: nothing accepted. Partial success returns Submission + error.
    // Kernel calls belong to the graph executor, not this byte-level runtime.
    virtual Result<Submission> submit(
        const ResourceView&, const MemoryCommandPlan&, MemoryBinding&,
        const InvocationContext&) = 0;
};
// The runtime policy installs shared domain caps before any arena/direct allocation.
Result<std::unique_ptr<MemoryRuntime>> open_runtime(
    const std::string& hardware_toml,
    const std::optional<std::string>& runtime_policy_toml = std::nullopt,
    const std::optional<std::string>& calibration_toml = std::nullopt);
// Serialization contains integer offsets and symbolic IDs, never runtime handles.
Result<std::string> recipe_to_toml(const MemoryRecipe&);
Result<MemoryRecipe> recipe_from_toml(std::string_view);
Result<std::string> layout_to_toml(const DistributedLayout&);
Result<DistributedLayout> layout_from_toml(std::string_view);
// END API TYPES
} // namespace vecops::memory::model
