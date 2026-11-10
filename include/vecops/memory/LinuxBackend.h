// SPDX-FileCopyrightText: 2026 HenryRenYz and vecops contributors
// SPDX-License-Identifier: MIT
#pragma once
#include "vecops/memory/Model.h"

namespace vecops::memory::model {
/** Backend binding only. Generic planning never interprets OS NUMA identifiers. */
struct LinuxPoolBinding {
  std::string pool;
  std::vector<unsigned> numa_nodes; // empty: kernel default; nonempty: strict MPOL_BIND
  std::string address_space = "host";
};
/** Serializable same-host export metadata. Opening an export certifies the inode;
 * no virtual pointer or process-local fd is part of the exported representation.
 * The exporter must retain its backing until all intended imports complete.
 */
struct LinuxBackingHandle {
  std::string name;
  std::string physical_identity;
  AllocationInfo allocation;
  std::uint64_t backing_offset_bytes = 0;
};

/** Linux page store and VA operations underlying MemoryRuntime.
 * Every mapping retains both backing and reservation. Destruction restores a
 * PROT_NONE reservation instead of releasing its address for accidental reuse.
 * A replacement returning an error preserves the previous mapping. Failure to
 * recover an unexpectedly lost placeholder/rollback is fail-stop, never a
 * MAP_FIXED overwrite of an unowned address. private_movable roots are eagerly
 * materialized at acquire and moved only whole, under proven quiescence.
 */
class LinuxPageStore {
public:
  explicit LinuxPageStore(std::vector<LinuxPoolBinding> bindings = {});
  /** Probe Linux DONTUNMAP/NORESERVE/DONTFORK support using one temporary base
   * page. Unsupported kernels return false; no caller backing is allocated. */
  static bool private_movable_supported();
  Result<std::shared_ptr<SharedBacking>> allocate(const BackingRequest&, std::shared_ptr<void> lifetime = {}) const;
  Result<std::shared_ptr<SharedBacking>> import_backing(const LinuxBackingHandle&) const;
  /** A slice retains its parent and this optional per-slice lifetime ticket.
   * Mappings/descendant slices therefore keep arena reuse tickets and the root
   * physical-capacity charge alive until their final local reference ends.
   * Export metadata alone is not ownership; retain the exporting backing until
   * the shared-layout protocol certifies that remote users have retired.
   */
  Result<std::shared_ptr<SharedBacking>> slice(std::shared_ptr<SharedBacking>, std::uint64_t offset,
                                               std::uint64_t bytes, const std::string& id,
                                               std::shared_ptr<void> lifetime = {}) const;
  /** Reserve the same byte/page geometry with an independent absolute VA base
   * minimum. Zero uses page_bytes; a nonzero minimum must be a power of two.
   * Existing large-reservation THP alignment preference is retained.
   */
  Result<std::unique_ptr<VirtualReservation>> reserve(std::uint64_t bytes, std::uint64_t page_bytes,
                                                      std::uint64_t base_alignment_bytes = 0) const;
  Result<std::unique_ptr<MappedRange>> map(VirtualReservation&, std::shared_ptr<SharedBacking>, const MapRequest&,
                                           std::span<const AccessMode> = {}) const;
  /** The caller must hold the runtime's proven quiescence lease over range. */
  Result<void> replace_quiescent(MappedRange&, std::shared_ptr<SharedBacking>, const MapRequest&,
                                 std::span<const AccessMode> = {}) const;
  static Result<std::string> physical_identity(const SharedBacking&);
  static Result<std::uint64_t> physical_offset(const SharedBacking&);
  /** Reserved address only; this does not grant permission to dereference. */
  static Result<void*> address(VirtualReservation&);
  static Result<MemoryDescriptor> describe_external(const ExternalMemoryView&, const std::string& symbol,
                                                    std::uint64_t generation);
  /** Query page residency in bounded batches. Unfaulted pages, unavailable OS
   * permission and nodes without one unambiguous accessible pool remain unknown.
   */
  Result<ResidencyObservation> observe_external(const ExternalMemoryView&,
                                                 std::span<const std::string> accessible_pools) const;

private:
  std::vector<LinuxPoolBinding> bindings_;
};
} // namespace vecops::memory::model
