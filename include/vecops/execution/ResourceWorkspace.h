// SPDX-FileCopyrightText: 2026 HenryRenYz and vecops contributors
// SPDX-License-Identifier: MIT
#pragma once
#include "vecops/execution/WorkspaceArena.h"

namespace vecops::memory::model {
class MemoryRuntime;
}
namespace vecops::execution {
/** Default host runtime shared by ordinary native eager calls in this module.
 * Environment hardware/policy/capacity settings are consumed on first use.
 * Applications can supply their own runtime instead of creating another ledger.
 */
std::shared_ptr<memory::model::MemoryRuntime> default_workspace_runtime();
/** Adapt a resource runtime to native workspace binding. Pool IDs are opaque.
 * Empty pools select grant pools directly readable/writable by every executable
 * host_load_store agent in the view.
 * Every allocation passes fresh admission; there is no heap fallback.
 */
std::shared_ptr<WorkspaceArenaProvider> resource_workspace_arena_provider(std::shared_ptr<memory::model::MemoryRuntime>,
                                                                          std::vector<std::string> pools = {});
/** Thread-affine scope primitive. Return the previous provider for restoration.
 * The caller must restore scopes in LIFO order on the same thread.
 */
std::shared_ptr<WorkspaceArenaProvider> exchange_workspace_arena_provider(std::shared_ptr<WorkspaceArenaProvider>);
} // namespace vecops::execution
