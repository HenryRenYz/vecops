// SPDX-FileCopyrightText: 2026 HenryRenYz and vecops contributors
// SPDX-License-Identifier: MIT
#include "GraphPlanning.h"
using namespace vecops::graph::planning;

// BEGIN GRAPH EXAMPLE
struct PreparedProgram {
    CompiledMemoryProgram program;
    std::shared_ptr<memory::PreparedAddresses> addresses;
};
PreparedProgram prepare_graph(
    const GraphMemoryCompiler& compiler, const CommunicationSynthesizer& communication,
    const ExecutionGraph& graph, memory::MemoryRuntime& runtime,
    const memory::ResourceView& view) {
    auto snapshot = runtime.snapshot(view);
    auto program = compiler.compile(graph, view, snapshot, runtime, communication).value();
    auto addresses = runtime.prepare_addresses(view, program.memory.addresses).value();
    // Bind executable argument addresses/guards now; do not read unmapped storage.
    return {std::move(program), std::move(addresses)};
}
struct PreparedModel {
    ModelMemoryProgram program;
    std::shared_ptr<memory::PreparedAddresses> addresses;
};
PreparedModel prepare_model(
    const GraphMemoryCompiler& compiler, const CommunicationSynthesizer& communication,
    const ModelExecutionPlan& model, memory::MemoryRuntime& runtime,
    const memory::ResourceView& view) {
    auto snapshot = runtime.snapshot(view);
    SearchBudget search{32, 1000000, FallbackPolicy::reject};
    auto program = compiler.compile_model(model, view, snapshot, runtime,
                                           communication, search).value();
    // Evoformer/diffusion/confidence AND native workspace share one physical plan.
    // ProgramCall bindings link subgraph external buffers to model storage identities.
    auto addresses = runtime.prepare_addresses(view, program.memory.addresses).value();
    return {std::move(program), std::move(addresses)};
}
std::shared_ptr<ModelExecutionInstance> activate_model(
    const ModelExecutor& executor, const PreparedModel& prepared,
    const InvocationInputs& inputs, memory::MemoryRuntime& runtime,
    const memory::ResourceView& view) {
    // Guard actual inputs and state, admit capacity, acquire/map physical backing.
    return executor.activate(prepared.program, inputs, runtime, view,
                             prepared.addresses).value();
}
memory::CompletionToken run_model(
    ModelExecutionInstance& instance, const InvocationInputs& actual_inputs,
    const std::string& invocation_id, std::span<const memory::EventBinding> ready) {
    // actual_inputs contains imported storage handles AND actual view/layout metadata.
    memory::InvocationStamp stamp{invocation_id, instance.binding_generation()};
    return instance.run(actual_inputs, stamp, ready).value();
}
// END GRAPH EXAMPLE
