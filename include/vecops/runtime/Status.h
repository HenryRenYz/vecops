// SPDX-FileCopyrightText: 2026 HenryRenYz and vecops contributors
// SPDX-License-Identifier: MIT
/**
 * @file Status.h
 * @brief Value-based C++ status and result transport for runtime operations.
 *
 * These types own diagnostic strings and do not use exceptions to represent
 * operation failure. Copying or moving them has normal value semantics;
 * immutable instances can be read concurrently.
 */
#ifndef VECOPS_RUNTIME_STATUS_H
#define VECOPS_RUNTIME_STATUS_H

#include <optional>
#include <string>
#include <utility>

#include "vecops/runtime/CallAbi.h"

namespace vecops::runtime {

/** @brief C++ status code mirroring the stable ABI status enumeration. */
enum class StatusCode : int {
  /** @brief Successful operation. */
  Ok = VECOPS_STATUS_OK,
  /** @brief Caller supplied invalid metadata or values. */
  InvalidArgument = VECOPS_STATUS_INVALID_ARGUMENT,
  /** @brief Candidate specialization cannot serve the request. */
  NotApplicable = VECOPS_STATUS_NOT_APPLICABLE,
  /** @brief Requested module, recipe, or specialization was absent. */
  NotFound = VECOPS_STATUS_NOT_FOUND,
  /** @brief C++ runtime and artifact ABI contracts differ. */
  AbiMismatch = VECOPS_STATUS_ABI_MISMATCH,
  /** @brief Dynamic library load or symbol lookup failed. */
  LoadError = VECOPS_STATUS_LOAD_ERROR,
  /** @brief Kernel execution reported failure. */
  ExecutionError = VECOPS_STATUS_EXECUTION_ERROR,
  /** @brief Runtime implementation encountered an unexpected failure. */
  InternalError = VECOPS_STATUS_INTERNAL_ERROR,
};

/**
 * @brief Owned outcome of an operation without a value payload.
 *
 * A default-constructed status is success.  The message is explanatory only;
 * callers should branch on `code()`.  Accessors do not mutate the object and
 * are safe for concurrent reads.
 */
class Status {
public:
  /** @brief Construct successful status. */
  Status() = default;
  /**
   * @brief Construct a status with an owned diagnostic.
   * @param code Result category; `Ok` is permitted but normally use `success()`.
   * @param message Human-readable diagnostic copied or moved into this status.
   */
  Status(StatusCode code, std::string message)
    : code_(code)
    , message_(std::move(message)) {
  }

  /** @brief Return a successful status value. */
  static Status success() {
    return {};
  }
  /** @brief Test whether this status is successful. */
  [[nodiscard]] bool ok() const {
    return code_ == StatusCode::Ok;
  }
  /** @brief Return the machine-readable result category. */
  [[nodiscard]] StatusCode code() const {
    return code_;
  }
  /**
   * @brief Return the owned diagnostic string.
   * @return Reference valid while this `Status` remains alive.
   */
  [[nodiscard]] const std::string& message() const {
    return message_;
  }

private:
  StatusCode code_ = StatusCode::Ok;
  std::string message_;
};

/**
 * @brief Either an owned value of `T` or a non-success `Status`.
 *
 * `value()` has the precondition `ok()`.  Constructing from success status is
 * converted to `InternalError` because a successful `Result<T>` must contain a
 * value.  The wrapper itself is thread-safe only for concurrent immutable use.
 *
 * @tparam T Value type held on success.
 */
template <typename T>
class Result {
public:
  /** @brief Construct a successful result by taking ownership of @p value. */
  Result(T value)
    : value_(std::move(value)) {
  }
  /**
   * @brief Construct a failed result from @p status.
   * @param status Non-success status; accidental success becomes `InternalError`.
   */
  Result(Status status)
    : status_(status.ok() ? Status(StatusCode::InternalError, "a value-less Result cannot contain success")
                          : std::move(status)) {
  }

  /** @brief Test whether a value is present. */
  [[nodiscard]] bool ok() const {
    return value_.has_value();
  }
  /** @brief Boolean shorthand for `ok()`. */
  [[nodiscard]] explicit operator bool() const {
    return ok();
  }
  /**
   * @brief Return failure status or a shared immutable success status.
   * @return Reference valid for the program lifetime on success, otherwise for
   * this result's lifetime.
   */
  [[nodiscard]] const Status& status() const {
    static const Status success_status;
    return ok() ? success_status : status_;
  }
  /** @brief Return mutable value; requires `ok()`. */
  [[nodiscard]] T& value() & {
    return *value_;
  }
  /** @brief Return immutable value; requires `ok()`. */
  [[nodiscard]] const T& value() const& {
    return *value_;
  }
  /** @brief Move the value out; requires `ok()`. */
  [[nodiscard]] T&& value() && {
    return std::move(*value_);
  }

private:
  std::optional<T> value_;
  Status status_;
};

/**
 * @brief Result specialization for operations that return no value.
 *
 * Default construction succeeds.  `status()` returns an owned status reference
 * whose lifetime is that of this result.
 */
template <>
class Result<void> {
public:
  /** @brief Construct successful void result. */
  Result() = default;
  /** @brief Construct from an owned status value. */
  Result(Status status)
    : status_(std::move(status)) {
  }

  /** @brief Test whether the operation succeeded. */
  [[nodiscard]] bool ok() const {
    return status_.ok();
  }
  /** @brief Boolean shorthand for `ok()`. */
  [[nodiscard]] explicit operator bool() const {
    return ok();
  }
  /** @brief Return the owned status reference. */
  [[nodiscard]] const Status& status() const {
    return status_;
  }

private:
  Status status_;
};

} // namespace vecops::runtime

#endif // VECOPS_RUNTIME_STATUS_H
