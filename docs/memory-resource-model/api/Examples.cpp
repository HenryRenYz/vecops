// SPDX-FileCopyrightText: 2026 HenryRenYz and vecops contributors
// SPDX-License-Identifier: MIT
#include "MemoryModel.h"
#include <stdexcept>
using namespace vecops::memory::model;

// BEGIN EAGER EXAMPLE
void eager_step(MemoryArena& arena, CompletionToken finished,
                const std::string& local_pool) {
    auto allocation = arena.allocate({
        .bytes = 20ULL << 20,
        .preferred_pool = local_pool,
        .allow_domain_fallback = true,
    }).value();
    void* pointer = allocation.data().value(); // always aligned to arena.page_bytes()
    const auto& actual = allocation.info().segments; // actual pools after fallback
    // Run tensor work using pointer; finished covers its last asynchronous use.
    (void)pointer;
    (void)actual;
    auto released = arena.release(std::move(allocation), finished);
    if (!released) throw std::runtime_error(released.error->message);
}
// END EAGER EXAMPLE

// BEGIN PLANNED EXAMPLE
// Stage 1: no physical allocation; slot() returns a reserved virtual address only.
std::shared_ptr<PreparedAddresses> prepare_memory(
    MemoryRuntime& runtime, const ResourceView& view, const MemoryRecipe& recipe) {
    return runtime.prepare_addresses(view, recipe.addresses).value();
}
// Stage 2: may run much later, using the same offline offsets and reserved VAs.
std::unique_ptr<MemoryBinding> activate_memory(
    MemoryRuntime& runtime, const ResourceView& view, const MemoryRecipe& recipe,
    std::shared_ptr<PreparedAddresses> addresses, MemoryArena& cache) {
    MemoryArena* caches[] = {&cache};
    auto budget = runtime.snapshot(view, caches);
    auto estimate = runtime.estimate(view, recipe, caches).value();
    if (!estimate.feasible) throw std::runtime_error("placement no longer fits");
    auto lease = runtime.reserve(view, recipe, budget.epoch, caches).value();
    auto backing = runtime.acquire_backing(view, recipe, *lease).value();
    return runtime.bind(view, recipe, std::move(addresses), std::move(backing)).value();
}
// END PLANNED EXAMPLE

// BEGIN REMAP EXAMPLE
std::unique_ptr<MappedRange> alias_pages(
    MemoryRuntime& runtime, Allocation& allocation, VirtualReservation& target) {
    auto backing = allocation.export_backing().value(); // requires remappable backing
    const auto q = allocation.info().page_bytes;
    return runtime.map(target, std::move(backing), {
        .backing_offset_bytes = 0,
        .target_offset_bytes = 0,
        .size_bytes = q,
    }).value();
}
// END REMAP EXAMPLE

// BEGIN COST EXAMPLE
CostEstimate estimate_transfer(MemoryRuntime& runtime, const ResourceView& view,
                               const TransferQuery& query, const CostContext& context) {
    auto candidates = runtime.costs().candidates(view, query.source, query.target).value();
    if (candidates.empty()) throw std::runtime_error("no legal transfer method");
    // query.method/executor select a candidate. Endpoint page layout is explicit.
    return runtime.costs().estimate(view, query, context).value();
}
Submission copy_slice(MemoryRuntime& runtime, const ResourceView& view,
                      MemoryBinding& binding, const InvocationContext& invocation) {
    MemoryCommand command{};
    command.id = "copy-input";
    command.opcode = MemoryOpcode::copy;
    command.source = ByteRangeRef{MemoryRefKind::external, "input", 4096, 65536};
    command.target = ByteRangeRef{MemoryRefKind::slot, "staging", 0, 65536};
    command.method = "selected-copy-method";
    command.queue = "selected-copy-queue";
    command.wait_ports = {"input-ready"};
    // Invocation supplies an input -> ImportedMemory binding, ready event and stamp.
    auto operands = runtime.resolve_operands(view, binding, invocation).value();
    auto valid = runtime.validate_commands(view, MemoryCommandPlan{{command}}, operands);
    if (!valid) throw std::runtime_error(valid.error->message);
    // submit re-resolves under retained leases; this query is not an execution permit.
    // The runtime resolves both operands, validates routes and retains their owners.
    return runtime.submit(view, MemoryCommandPlan{{command}}, binding, invocation).value();
}
InvocationContext bind_input(InvocationStamp stamp, CompletionToken ready,
                             std::shared_ptr<ImportedMemory> imported_input) {
    // Another instance may bind the same plan symbol to a different imported object.
    return {std::move(stamp), {{"input-ready", std::move(ready)}},
            {{"input", std::move(imported_input)}}};
}
// END COST EXAMPLE

// BEGIN SHARED EXAMPLE
std::unique_ptr<MemoryBinding> activate_shared_memory(
    MemoryRuntime& runtime, const ResourceView& view, const MemoryRecipe& local,
    const DistributedLayout& manifest, std::shared_ptr<PreparedAddresses> addresses) {
    auto agreement = runtime.agree_layout(manifest).value(); // digest, offsets, owners
    auto budget = runtime.snapshot(view);
    auto lease = runtime.reserve(view, local, budget.epoch).value();
    auto owned = runtime.acquire_backing(view, local, *lease).value(); // local owner only
    auto accessible = runtime.exchange_backing(*agreement, std::move(owned)).value();
    return runtime.bind(view, local, std::move(addresses), std::move(accessible),
                        agreement.get()).value(); // retains agreed generation
}
// END SHARED EXAMPLE
