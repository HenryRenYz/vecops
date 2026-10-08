// SPDX-FileCopyrightText: 2026 HenryRenYz and vecops contributors
// SPDX-License-Identifier: MIT
#include "WorkspaceContextFixture.h"

#include <cstddef>
#include <cstdint>

#include "vecops/runtime/KernelAbi.h"

namespace {

using vecops::test::WorkspaceContextFixtureBytes;
using vecops::test::WorkspaceContextObservation;

int32_t workspace(const VecopsCall* call, std::uint64_t* size, VecopsError*) {
  if (call == nullptr || size == nullptr || call->context == nullptr || call->context->user_data == nullptr)
    return VECOPS_STATUS_INVALID_ARGUMENT;
  auto& observation = *static_cast<WorkspaceContextObservation*>(call->context->user_data);
  ++observation.workspace_query_calls;
  observation.query_context = call->context;
  *size = WorkspaceContextFixtureBytes;
  return VECOPS_STATUS_OK;
}

int32_t run(const VecopsCall* call, VecopsError*) {
  if (call == nullptr || call->context == nullptr || call->context->user_data == nullptr ||
      call->workspace == nullptr || call->workspace_size < WorkspaceContextFixtureBytes)
    return VECOPS_STATUS_INVALID_ARGUMENT;
  auto& observation = *static_cast<WorkspaceContextObservation*>(call->context->user_data);
  ++observation.run_calls;
  observation.run_context = call->context;
  observation.run_workspace = call->workspace;
  observation.run_workspace_size = call->workspace_size;
  observation.requested_threads = call->context->requested_threads;
  observation.stream = call->context->stream;
  observation.context_flags = call->context->flags;
  static_cast<std::byte*>(call->workspace)[0] = std::byte{0x5a};
  return VECOPS_STATUS_OK;
}

const VecopsKernelDescriptorV1 descriptor{
  sizeof(VecopsKernelDescriptorV1),
  VECOPS_KERNEL_ABI_VERSION_MAJOR,
  VECOPS_KERNEL_ABI_VERSION_MINOR,
  "test::workspace-context",
  "workspace-context-v1",
  0,
  sizeof(VecopsParameterDescriptor),
  nullptr,
  workspace,
  run,
  0,
};

} // namespace

extern "C" VECOPS_RUNTIME_EXPORT const VecopsKernelDescriptorV1* vecops_kernel_query_v1() {
  return &descriptor;
}
