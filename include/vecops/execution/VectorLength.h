// Copyright (c) vecops contributors.

#ifndef VECOPS_EXECUTION_VECTOR_LENGTH_H
#define VECOPS_EXECUTION_VECTOR_LENGTH_H

#include <cerrno>
#include <cstdint>
#include <exception>
#include <stdexcept>
#include <system_error>

#if defined(__linux__) && defined(__aarch64__)
#  include <sys/auxv.h>
#  include <sys/prctl.h>
#endif

/**
 * @file vecops/execution/VectorLength.h
 * @brief Propagate Linux AArch64 per-thread SVE/SME lengths into worker pools.
 *
 * SVE VL and SME SVL are independent thread properties. Persistent workers
 * need not share the dispatching thread's lengths. Capture their expected
 * values before launching a region and install ScopedArmVectorLengths at its
 * worker entry, before constructing vectors or entering streaming mode.
 * No live vector, predicate, or ZA value may cross a length change. The guard
 * restores the worker's original lengths and inheritance flags on exit.
 */
namespace vecops::execution {

/** Zero denotes a vector extension unavailable to this thread/OS. */
struct ArmVectorLengths {
  std::uint32_t sve_bytes = 0;
  std::uint32_t sme_bytes = 0;

  friend constexpr bool operator==(ArmVectorLengths, ArmVectorLengths) = default;
};

namespace details {

#if defined(__linux__) && defined(__aarch64__)
// Keep compatibility with libc headers predating Linux's SME prctl API.
inline constexpr int sve_set_vl = 50;
inline constexpr int sve_get_vl = 51;
inline constexpr int sme_set_vl = 63;
inline constexpr int sme_get_vl = 64;
inline constexpr unsigned long vector_length_mask = 0xffff;
inline constexpr unsigned long vector_length_inherit = 1UL << 17;

[[nodiscard]] inline unsigned supported_vector_extensions() noexcept {
  // Linux's process-wide HWCAP contract is safe across persistent workers.
  // libc reads the saved auxv rather than probing each dispatch with prctl.
  static const unsigned extensions =
    ((::getauxval(AT_HWCAP) & (1UL << 22)) != 0 ? 1U : 0U) | ((::getauxval(AT_HWCAP2) & (1UL << 23)) != 0 ? 2U : 0U);
  return extensions;
}

[[nodiscard]] inline std::uint32_t read_sve_bytes() noexcept {
  // Raw instruction encodings also work in the backend-neutral runtime TU,
  // which deliberately need not be compiled with -march=...+sve/+sme.
  // Volatile asm prevents LLVM from hoisting/CSE'ing CNTB across SMSTART.
  register unsigned long bytes __asm__("x0");
  __asm__ __volatile__(".inst 0x0420e3e0 // cntb x0" : "=r"(bytes) : : "memory");
  return static_cast<std::uint32_t>(bytes);
}

[[nodiscard]] inline std::uint32_t read_sme_bytes() noexcept {
  register unsigned long bytes __asm__("x0");
  __asm__ __volatile__(".inst 0x04bf5820 // rdsvl x0,#1" : "=r"(bytes) : : "memory");
  return static_cast<std::uint32_t>(bytes);
}

[[nodiscard]] inline unsigned long query_vector_config(int operation) {
  const int result = ::prctl(operation, 0UL, 0UL, 0UL, 0UL);
  if (result >= 0)
    return static_cast<unsigned long>(result);
  const int error = errno;
  if (error == EINVAL || error == ENOSYS)
    return 0;
  throw std::system_error(error, std::generic_category(), "query AArch64 vector length");
}

[[nodiscard]] inline unsigned long set_vector_config(int operation, unsigned long value) {
  const int result = ::prctl(operation, value, 0UL, 0UL, 0UL);
  if (result < 0)
    throw std::system_error(errno, std::generic_category(), "set AArch64 worker vector length");
  return static_cast<unsigned long>(result);
}

inline void restore_vector_config(int operation, unsigned long value) noexcept {
  // Restoration failure must not leave a pool worker silently misconfigured.
  const int result = ::prctl(operation, value, 0UL, 0UL, 0UL);
  if (result < 0 || (static_cast<unsigned long>(result) & vector_length_mask) != (value & vector_length_mask))
    std::terminate();
}
#endif

} // namespace details

/** Read current VL/SVL outside streaming mode; unsupported extensions are zero. */
[[nodiscard]] inline ArmVectorLengths current_thread_vector_lengths() noexcept {
#if defined(__linux__) && defined(__aarch64__)
  const auto supported = details::supported_vector_extensions();
  return {(supported & 1U) != 0 ? details::read_sve_bytes() : 0, (supported & 2U) != 0 ? details::read_sme_bytes() : 0};
#else
  return {};
#endif
}

/**
 * Cheap architectural check before kernel/worker entry (SM must be zero).
 * Auxv support bits guard every architectural read, so requesting an
 * unavailable extension reports a mismatch instead of SIGILL. Unlike svcntb's
 * readnone LLVM intrinsic, these volatile reads cannot cross SM boundaries.
 */
[[nodiscard]] inline bool current_thread_vector_lengths_match(ArmVectorLengths expected) noexcept {
  const auto valid = [](std::uint32_t bytes) { return bytes == 0 || (bytes >= 16 && bytes <= 256 && bytes % 16 == 0); };
  if (!valid(expected.sve_bytes) || !valid(expected.sme_bytes))
    return false;
#if defined(__linux__) && defined(__aarch64__)
  const auto supported = details::supported_vector_extensions();
  if ((expected.sve_bytes != 0 && (supported & 1U) == 0) || (expected.sme_bytes != 0 && (supported & 2U) == 0))
    return false;
  return (expected.sve_bytes == 0 || details::read_sve_bytes() == expected.sve_bytes) &&
         (expected.sme_bytes == 0 || details::read_sme_bytes() == expected.sme_bytes);
#else
  return expected == ArmVectorLengths{};
#endif
}

/** Reject invoking a fixed-geometry kernel after the caller changes VL/SVL. */
inline void require_current_thread_vector_lengths(ArmVectorLengths expected) {
  if (!current_thread_vector_lengths_match(expected))
    throw std::runtime_error("kernel SVE VL/SME SVL differs from the calling thread; obtain the kernel again "
                             "through T.jit or T.compile for the current vector lengths");
}

/**
 * Temporarily match the dispatching thread's independent VL/SVL pair.
 *
 * Unsupported requested lengths and OS rounding are errors, not permission
 * to execute with a different geometry. An unavailable extension (zero) is
 * left unchanged. Non-Linux/non-AArch64 builds accept only the zero pair.
 */
class ScopedArmVectorLengths {
public:
  explicit ScopedArmVectorLengths(ArmVectorLengths expected) {
#if defined(__linux__) && defined(__aarch64__)
    if (current_thread_vector_lengths_match(expected))
      return;
    sve_config_ = details::query_vector_config(details::sve_get_vl);
    sme_config_ = details::query_vector_config(details::sme_get_vl);
    try {
      install(details::sve_set_vl, expected.sve_bytes, sve_config_, sve_changed_);
      install(details::sme_set_vl, expected.sme_bytes, sme_config_, sme_changed_);
    } catch (...) {
      restore();
      throw;
    }
#else
    if (expected.sve_bytes != 0 || expected.sme_bytes != 0)
      throw std::invalid_argument("AArch64 vector lengths are unavailable on this platform");
#endif
  }

  ScopedArmVectorLengths(const ScopedArmVectorLengths&) = delete;
  ScopedArmVectorLengths& operator=(const ScopedArmVectorLengths&) = delete;
  ~ScopedArmVectorLengths() noexcept {
#if defined(__linux__) && defined(__aarch64__)
    restore();
#endif
  }

private:
#if defined(__linux__) && defined(__aarch64__)
  static void install(int operation, std::uint32_t expected, unsigned long original, bool& changed) {
    if (expected == 0 || expected == (original & details::vector_length_mask))
      return;
    if (expected < 16 || expected > 256 || expected % 16 != 0)
      throw std::invalid_argument("AArch64 vector length must be a multiple of 16 bytes in [16,256]");
    if ((original & details::vector_length_mask) == 0)
      throw std::runtime_error("worker lacks the caller's AArch64 vector extension");
    const auto selected = details::set_vector_config(operation, expected | (original & details::vector_length_inherit));
    changed = true;
    if ((selected & details::vector_length_mask) != expected)
      throw std::runtime_error("OS did not select the exact caller AArch64 vector length");
  }

  void restore() noexcept {
    if (sme_changed_) {
      details::restore_vector_config(details::sme_set_vl, sme_config_);
      sme_changed_ = false;
    }
    if (sve_changed_) {
      details::restore_vector_config(details::sve_set_vl, sve_config_);
      sve_changed_ = false;
    }
  }

  unsigned long sve_config_ = 0;
  unsigned long sme_config_ = 0;
  bool sve_changed_ = false;
  bool sme_changed_ = false;
#endif
};

} // namespace vecops::execution

#endif // VECOPS_EXECUTION_VECTOR_LENGTH_H
