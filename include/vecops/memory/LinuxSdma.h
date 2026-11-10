// SPDX-FileCopyrightText: 2026 HenryRenYz and vecops contributors
// SPDX-License-Identifier: MIT
#pragma once
#include "vecops/memory/LinuxTransfer.h"
#include <functional>

namespace vecops::memory::model {
struct LinuxSdmaOptions {
  std::string device = "/dev/sdma0";
  std::string library = "libsdma_dk.so";
  std::uint64_t chunk_bytes = 16 * 1024 * 1024;
  std::uint64_t page_bytes = 4096;
  // SDK shared-channel index; nullopt requests a driver-exclusive channel.
  std::optional<std::uint32_t> shared_channel = 0;
};
/** Driver boundary for official sdma-dk and deterministic fault injection.
 * Each instance owns one SDK channel handle and permits one in-flight task.
 * Shared channels use a request-specific completion receipt; other clients
 * may share the physical queue without sharing this driver's task ownership.
 * submit errors do NOT prove nonacceptance. completed() may certify stopped
 * only after every potentially accepted descriptor was retired by the SDK.
 */
class LinuxSdmaDriver {
public:
  virtual ~LinuxSdmaDriver() = default;
  virtual Result<std::uint64_t> pin(void*, std::uint32_t bytes) = 0;
  virtual Result<void> unpin(std::uint64_t cookie) = 0;
  virtual Result<void> submit(void* target, const void* source, std::uint32_t bytes) = 0;
  virtual Result<bool> completed() = 0;
};
using LinuxSdmaDriverFactory = std::function<Result<std::unique_ptr<LinuxSdmaDriver>>() >;
/** Optional official SDK adapter. An unavailable SDK/device is an error, never
 * a CPU copy. Registrations are page-aligned, <= uint32 SDK pin limits; large
 * ranges are split. The executor reports caller-driven, one task in flight.
 *
 * Runtime submission errors with unknown device quiescence return failed with
 * accesses_stopped=false. Such event owners, SDK/channel and page registrations
 * are quarantined until a later query certifies completion; permanent hardware
 * failure retains them until process exit rather than blocking runtime teardown
 * or allowing unsafe memory reuse. Keep the runtime/token to retry query().
 */
Result<std::shared_ptr<LinuxTransferExecutor>> make_linux_sdma_executor(
  LinuxSdmaOptions = {}, LinuxSdmaDriverFactory = {});
bool linux_sdma_sdk_compiled() noexcept;
} // namespace vecops::memory::model
