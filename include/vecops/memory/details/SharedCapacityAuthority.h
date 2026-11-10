// SPDX-FileCopyrightText: 2026 HenryRenYz and vecops contributors
// SPDX-License-Identifier: MIT
#pragma once
#include "vecops/memory/Model.h"
#include <filesystem>
#include <functional>
#include <mutex>

namespace vecops::memory::model::details {
struct SharedCapacityGroup {
  std::string id;
  std::uint64_t capacity_bytes;
  std::uint64_t hard_limit_bytes;
};
enum class SharedCapacityOwnerState { alive, owner_dead, unknown };
struct SharedCapacityOwner {
  std::string participant;
  std::uint64_t pid;
  std::uint64_t start_time_ticks;
  SharedCapacityOwnerState state;
};
struct SharedCapacityGrantReceipt {
  std::string id;
  SharedCapacityOwner owner;
  std::map<std::string, std::uint64_t> remaining;
};
struct SharedCapacityRootReceipt {
  std::string id;
  SharedCapacityOwner owner;
  std::string group;
  std::uint64_t bytes;
  std::string physical_identity;
};
struct SharedCapacityGroupSnapshot {
  std::string id;
  std::uint64_t capacity_bytes;
  std::uint64_t hard_limit_bytes;
  std::uint64_t committed_bytes;
  std::uint64_t pending_bytes;
  std::uint64_t hard_remaining_bytes;
};
struct SharedCapacityConstraintSnapshot {
  std::string id;
  std::vector<std::string> capacity_groups;
  std::uint64_t hard_limit_bytes;
  std::uint64_t remaining_bytes;
};
struct SharedCapacitySnapshot {
  std::uint64_t epoch;
  std::vector<SharedCapacityGroupSnapshot> groups;
  std::vector<SharedCapacityConstraintSnapshot> constraints;
  std::vector<SharedCapacityGrantReceipt> grants;
  std::vector<SharedCapacityRootReceipt> roots;
};
struct SharedCapacityState;
class SharedCapacityCharge;
/** Exclusive ownership of one participant's unmaterialized global reservation.
 * A root commit moves bytes from pending to committed atomically. Destruction
 * refunds only unused bytes; committed roots have independent charge ownership.
 * Open a new authority in a forked child instead of using inherited handles.
 */
class SharedCapacityGrant {
public:
  ~SharedCapacityGrant();
  SharedCapacityGrant(const SharedCapacityGrant&) = delete;
  SharedCapacityGrant& operator=(const SharedCapacityGrant&) = delete;
  const std::string& id() const noexcept;
  Result<std::shared_ptr<SharedCapacityCharge>> commit(const std::string& group, std::uint64_t bytes,
                                                        const std::string& canonical_physical_identity);

private:
  friend struct SharedCapacityState;
  SharedCapacityGrant(std::shared_ptr<SharedCapacityState>, std::string);
  std::shared_ptr<SharedCapacityState> state_;
  std::string id_;
  std::mutex mutex_;
  bool active_ = false;
};
/** One root physical backing's global charge. Share this pointer with every
 * local alias/slice; only the final release refunds committed capacity. Exported
 * references in other processes require the owner's explicit retire protocol;
 * their import must not create a second charge.
 */
class SharedCapacityCharge {
public:
  ~SharedCapacityCharge();
  SharedCapacityCharge(const SharedCapacityCharge&) = delete;
  SharedCapacityCharge& operator=(const SharedCapacityCharge&) = delete;
  const SharedCapacityRootReceipt& receipt() const noexcept;

private:
  friend struct SharedCapacityState;
  SharedCapacityCharge(std::shared_ptr<SharedCapacityState>, SharedCapacityRootReceipt);
  std::shared_ptr<SharedCapacityState> state_;
  SharedCapacityRootReceipt receipt_;
  bool active_ = false;
};
/** Same-host authority for infrequent root backing reserve/commit/release only.
 * The caller provides a job/runtime namespace directory. It must be owned by
 * the current UID with mode 0700. An immutable lock inode serializes metadata
 * transactions; capacity.toml uses atomic rename and mode 0600. Configuration
 * contains no changing free-memory samples. Hosts/boots cannot share a ledger.
 *
 * Physical-free samples already exclude all materialized memory. Admission
 * deducts outstanding pending grants again, and checks all overlapping hard
 * constraints in the same transaction. Uncommitted materialization may therefore
 * be conservatively counted twice until its commit.
 *
 * Dead/unknown owners are reported but never reclaimed automatically: another
 * participant may still access their exported backing. Metadata/IO failures
 * preserve conservative ownership. A post-rename durability error reports
 * commit_uncertain; callers must inspect receipts rather than retry allocation.
 */
class SharedCapacityAuthority {
public:
  static Result<std::shared_ptr<SharedCapacityAuthority>>
  open(const std::filesystem::path& directory, std::vector<SharedCapacityGroup>,
       std::vector<BudgetConstraint>, std::string participant);
  Result<SharedCapacitySnapshot> snapshot() const;
  Result<std::unique_ptr<SharedCapacityGrant>>
  reserve(const std::map<std::string, std::uint64_t>& bytes, std::uint64_t expected_epoch,
          const std::map<std::string, std::uint64_t>& physical_free_samples);
  /** Automatic root growth samples physical free capacity inside the admission
   * transaction. This prevents contention on unrelated roots from invalidating
   * a sample/epoch pair. The sampler must not reenter this authority or acquire
   * another runtime's admission lock. Errors leave metadata unchanged; neither
   * allocation nor durability errors are retried. Explicit plans use reserve()
   * and retain their caller-supplied epoch check.
   */
  Result<std::unique_ptr<SharedCapacityGrant>> reserve_current(
      const std::map<std::string, std::uint64_t>& bytes,
      const std::function<Result<std::map<std::string, std::uint64_t>>()>& physical_free_sampler);
  const std::string& participant() const noexcept;
  /** Host/boot and certified lock-inode identity; identical for participants
   * using the same coordinator through different filesystem path aliases. */
  const std::string& namespace_id() const noexcept;
  /** First failed automatic refund, if any. Destructors never throw or silently
   * discard the associated persisted receipts. */
  std::optional<Error> cleanup_error() const;

private:
  explicit SharedCapacityAuthority(std::shared_ptr<SharedCapacityState>);
  std::shared_ptr<SharedCapacityState> state_;
};
} // namespace vecops::memory::model::details
