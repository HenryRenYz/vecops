// SPDX-FileCopyrightText: 2026 HenryRenYz and vecops contributors
// SPDX-License-Identifier: MIT
#pragma once
// C++ concept protocol for the upper graph layer, not an installed C++ API.
// Production planning is Python: python/vecops/planner/compiler.py. The memory
// infrastructure alone is C++; its public types come from vecops/memory/Model.h.
// Concept -> production mapping (see ../README.txt and index.html):
// BufferRequirement -> planner.BufferRequirement (lifetimes/retention/placement).
// GraphMemoryCompiler.compile -> compile(requirements).
// GraphMemoryCompiler.compile_model -> compile_model(programs, calls,
//     serialized=..., scoped_requirements=...).
// Model preparation -> GraphMemoryCompiler.prepare_model(...): address-only
//     PreparedProgram plus pending admission; no physical backing acquired.
// Activation/backing -> PreparedProgram.admit/materialize/close.
// CommunicationSynthesizer -> distributed.compiler.CommunicationCompiler;
//     independent candidates/estimate/select, not an argument to compile().
// ModelExecutor, ModelExecutionInstance, LayoutProvider, DistributedLayoutLinker
// and ReuseProof below express design responsibilities, not delivered classes.
// GraphExamples.cpp demonstrates this concept protocol only. It is not an SDK
// integration example. The dependency points only toward MemoryModel.h.
#include "MemoryModel.h"
namespace vecops::graph::planning {
namespace memory = vecops::memory::model;
// BEGIN GRAPH TYPES
enum class BufferBoundary { external, internal };
enum class ExternalRole { input, output, state };
enum class Ownership { borrowed, runtime_owned };
enum class Access { read, write, read_write, metadata_only };
enum class FallbackPolicy { reject, recompile, conservative_dynamic };
struct ShapeBound {
    std::string symbol;
    std::uint64_t minimum;
    std::uint64_t maximum;
};
struct ActivationContract {
    std::vector<ShapeBound> shapes;
    std::map<std::string, std::string> required_state_versions; // storage -> version
    std::vector<std::string> required_donations;
    std::map<std::string, std::string> control_guards; // registered guard -> expected result
    FallbackPolicy fallback;
    bool reentrant; // false: reject overlapping replay of the same bound instance
};
struct LayoutDescriptor {
    std::string id;
    std::string provider;
    std::string version;
    std::string opaque_key; // provider-owned blocked/packed/quantized layout description
    std::uint64_t storage_bytes; // queried physical size, includes padding
    std::uint64_t alignment;
    bool initialize_padding;
    std::vector<std::string> auxiliary_buffers;
};
struct ExternalContract {
    std::string buffer;
    ExternalRole role;
    Ownership ownership;
    bool may_redirect;
    bool may_donate;
    bool preserve_after_run;
    std::string version;
    std::optional<memory::TransferEndpoint> initial_location; // source for migration costing
    std::string ready_port;
};
struct BufferRequirement {
    std::string id; // physical storage identity, shared by all tensor views
    BufferBoundary boundary;
    std::uint64_t bytes;
    std::uint64_t alignment;
    std::vector<std::string> consumers; // Agent IDs
    std::vector<std::string> allowed_pools;
    memory::BackingMode backing;
};
struct TensorView {
    std::string id;
    std::string storage;
    std::uint64_t byte_offset;
    std::uint64_t span_bytes;
    std::vector<std::int64_t> shape;
    std::vector<std::int64_t> strides;
    std::string dtype;
    std::optional<std::string> backend_layout; // absent: ordinary strided representation
};
struct ActualArgument {
    TensorView view; // actual dtype/shape/strides/storage/offset for this invocation
    std::optional<LayoutDescriptor> layout; // actual provider descriptor for opaque layout
};
struct BufferUse {
    std::string buffer;
    std::string activity;
    Access access;
    std::uint64_t offset_bytes;
    std::uint64_t size_bytes;
};
struct AliasRule {
    std::string input;
    std::string output;
    std::string kernel_contract; // audited overlap/layout/read-before-write rules
    bool required;
};
struct KernelVariant {
    std::string id;
    std::string agent;
    std::vector<BufferRequirement> internal_buffers; // tensor temps AND workspace
    std::vector<AliasRule> alias_rules;
    double compute_us;
    std::vector<memory::ResourceDemand> demands;
    std::map<std::string, std::string> operand_layouts; // operand -> LayoutDescriptor ID
    std::vector<std::string> readiness_events; // rank/chunk-specific readiness
    memory::CostQuality cost_quality;
};
struct OperatorContract {
    std::string id;
    std::vector<KernelVariant> variants;
    std::vector<std::string> dependencies;
    bool opaque_effects;
};
struct ExecutionGraph {
    std::string fingerprint; // shape/layout/control-flow/kernel contract version
    std::vector<BufferRequirement> buffers;
    std::vector<ExternalContract> external;
    std::vector<TensorView> views;
    std::vector<BufferUse> uses;
    std::vector<OperatorContract> operators;
    std::vector<LayoutDescriptor> layouts;
    ActivationContract activation;
};
struct DataPiece {
    std::string storage;
    std::string version;
    std::uint64_t offset_bytes;
    std::uint64_t size_bytes;
};
struct Receiver {
    std::string agent;
    std::string output;
    std::vector<DataPiece> pieces;
    bool read_only_consumer;
};
enum class ReductionOrder { fixed, reassociation_allowed }; // fixed: source-order left fold
struct ReductionContract {
    std::string provider;
    std::string operation; // sum/max/custom, not inferred from byte length
    std::string version;
    std::string element_type;
    std::uint64_t element_count;
    bool associative;
    bool commutative;
    ReductionOrder order; // floating-point reassociation requires explicit permission
};
struct CommunicationRequest {
    std::string id;
    std::string operation; // broadcast/allgather/alltoall/reduction semantics
    std::vector<DataPiece> sources;
    std::vector<Receiver> receivers;
    bool deterministic_order;
    std::optional<ReductionContract> reduction;
};
struct KernelInvocation {
    std::string id;
    std::string provider;
    std::string version;
    KernelVariant implementation;
    std::map<std::string, DataPiece> inputs;
    std::map<std::string, DataPiece> outputs;
    std::vector<std::string> input_order; // operand names, explicit for noncommutative/fixed folds
    std::optional<ReductionContract> reduction;
};
struct GraphStep {
    std::string id;
    std::optional<std::string> kernel; // KernelInvocation ID; executed by graph executor
    std::optional<std::string> memory_command; // exactly one kernel/command
    std::vector<std::string> dependencies;
};
struct GraphExecutionPlan {
    std::vector<KernelInvocation> kernels;
    std::vector<GraphStep> steps;
    memory::MemoryCommandPlan memory_commands;
    memory::CostPlan cost;
};
struct TransferInvocation {
    std::string id;
    DataPiece source;
    DataPiece target;
    std::string method;
    std::optional<std::string> route;
    std::string executor;
};
struct FragmentStep {
    std::string id;
    std::optional<std::string> kernel;
    std::optional<std::string> transfer; // exactly one kernel/transfer
    std::vector<std::string> dependencies;
};
struct GraphFragment {
    std::vector<BufferRequirement> internal_buffers; // partials, pack and transport staging
    std::vector<BufferUse> uses;
    std::vector<KernelInvocation> kernels;
    std::vector<TransferInvocation> transfers; // logical storage, lowered after placement
    std::vector<FragmentStep> steps;
    std::vector<std::string> entry_ports;
    std::vector<std::string> completion_ports;
};
struct CommunicationPlan {
    std::string id;
    GraphFragment fragment; // sole executable source; transport/cost are derived
    std::map<std::string, std::uint64_t> traffic_bytes_by_resource;
    memory::CostEstimate cost;
    std::vector<memory::SharedRange> shared_ranges;
};
struct BufferBinding {
    std::string buffer;
    std::optional<std::string> slot;
    std::optional<std::string> external; // exactly one of slot/external
    std::uint64_t view_offset_bytes; // tensor-view offset; allocation base stays aligned
    std::uint64_t size_bytes;
    std::uint64_t alignment; // suballocation, not backing/map granularity
    bool independent_mapping; // true requires an exclusive page-aligned envelope
};
struct ReuseProof {
    std::string earlier_buffer;
    std::string later_buffer;
    std::vector<std::string> retire_events;
    std::vector<std::string> later_first_use_events;
    std::string placement_key;
    std::optional<std::string> sharing_contract;
    std::uint64_t shared_data_epoch;
};
struct CompiledMemoryProgram {
    ExecutionGraph rewritten;
    GraphExecutionPlan initialization; // relocation/packing outside replay
    GraphExecutionPlan execution;
    memory::MemoryRecipe memory;
    std::vector<BufferBinding> bindings;
    std::vector<CommunicationPlan> communications;
    std::vector<ReuseProof> reuse_proofs;
    std::string resource_signature;
    std::uint64_t budget_epoch;
};
struct CallBinding {
    std::string local_buffer;
    std::string model_storage; // global identity; external to subgraph may be internal here
    std::uint64_t offset_bytes;
    std::uint64_t size_bytes;
};
struct ProgramCall {
    std::string id;
    std::string graph;
    std::vector<std::string> dependencies;
    std::vector<CallBinding> bindings;
    std::uint64_t repeat_count; // exact positive count, sequential, identical bindings
    std::vector<std::string> retained_storage; // liveness only; no implicit feedback
};
struct ModelExecutionPlan {
    std::string fingerprint;
    std::map<std::string, ExecutionGraph> graphs;
    std::vector<ProgramCall> calls; // includes graph-external native calls as graph nodes
    std::vector<BufferRequirement> buffers;
    std::vector<ExternalContract> external; // boundary of the entire model invocation
    ActivationContract activation;
};
struct SearchBudget {
    std::uint64_t max_candidates;
    std::uint64_t max_planning_us;
    FallbackPolicy on_exhaustion; // return a proven feasible baseline or explicit failure
};
struct ModelMemoryProgram {
    std::map<std::string, CompiledMemoryProgram> programs; // call ID -> specialization
    ModelExecutionPlan schedule;
    std::vector<ReuseProof> cross_call_reuse;
    std::map<std::string, std::uint64_t> unique_peak_bytes_by_capacity_group;
    std::map<std::string, std::uint64_t> retained_bytes_by_capacity_group;
    memory::MemoryRecipe memory; // linked physical arenas and offset bindings
    bool used_conservative_fallback;
};
// END GRAPH TYPES

// BEGIN GRAPH API
class CommunicationSynthesizer {
public:
    virtual ~CommunicationSynthesizer() = default;
    virtual memory::Result<std::vector<CommunicationPlan>> candidates(
        const CommunicationRequest&, const memory::ResourceView&,
        const memory::MemoryCostModel&, const memory::CostContext&) const = 0;
};
class GraphMemoryCompiler {
public:
    virtual ~GraphMemoryCompiler() = default;
    virtual memory::Result<CompiledMemoryProgram> compile(
        const ExecutionGraph&, const memory::ResourceView&,
        const memory::BudgetSnapshot&, const memory::MemoryQueries&,
        const CommunicationSynthesizer&) const = 0;
    virtual memory::Result<ModelMemoryProgram> compile_model(
        const ModelExecutionPlan&, const memory::ResourceView&,
        const memory::BudgetSnapshot&, const memory::MemoryQueries&,
        const CommunicationSynthesizer&, const SearchBudget&) const = 0;
};
class LayoutProvider {
public:
    virtual ~LayoutProvider() = default;
    virtual memory::Result<LayoutDescriptor> describe(
        const TensorView&, const std::string& opaque_key) const = 0;
    virtual memory::Result<OperatorContract> conversion(
        const LayoutDescriptor&, const LayoutDescriptor&) const = 0;
};
struct InvocationInputs {
    std::map<std::string, std::shared_ptr<memory::ImportedMemory>> buffers; // model storage symbols
    std::map<std::string, ActualArgument> arguments; // model ABI tensor names
    std::map<std::string, std::uint64_t> shape_values;
    std::map<std::string, std::string> state_versions;
    std::vector<std::string> donated_buffers; // current owner's explicit authorization
};
class ModelExecutionInstance {
public:
    virtual ~ModelExecutionInstance() = default;
    virtual std::uint64_t binding_generation() const = 0;
    // Revalidate shape/layout/state/donation for every run; retain all accepted work.
    // Result error means none accepted; partial execution fails through its event.
    // Build the byte runtime's InvocationContext.external from inputs.buffers once.
    virtual memory::Result<memory::CompletionToken> run(
        const InvocationInputs&, const memory::InvocationStamp&,
        std::span<const memory::EventBinding> ready_events = {}) = 0;
};
class ModelExecutor {
public:
    virtual ~ModelExecutor() = default;
    // Runtime-facing upper layer, separate from the read-only GraphMemoryCompiler.
    virtual memory::Result<std::shared_ptr<ModelExecutionInstance>> activate(
        const ModelMemoryProgram&, const InvocationInputs&,
        memory::MemoryRuntime&, const memory::ResourceView&,
        std::shared_ptr<memory::PreparedAddresses>) const = 0;
};
class DistributedLayoutLinker {
public:
    virtual ~DistributedLayoutLinker() = default;
    virtual memory::Result<memory::DistributedLayout> link(
        std::span<const CompiledMemoryProgram>) const = 0;
};
// END GRAPH API
} // namespace vecops::graph::planning
