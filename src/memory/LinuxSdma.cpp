// SPDX-FileCopyrightText: 2026 HenryRenYz and vecops contributors
// SPDX-License-Identifier: MIT
#include "vecops/memory/LinuxSdma.h"
#include "details/RuntimeInternal.h"
#include <atomic>
#include <cerrno>
#include <dlfcn.h>
#include <fcntl.h>
#include <unistd.h>
#ifdef VECOPS_RUNTIME_HAVE_SDMA
#include <mdk_sdma.h>
#endif

namespace vecops::memory::model {
namespace {
using namespace details;
class Operation final : public LinuxTransferOperation, public std::enable_shared_from_this<Operation> {
  std::shared_ptr<Operation> inflight_;
  std::unique_ptr<LinuxSdmaDriver> driver_;
  std::shared_ptr<void> owner_;
  std::vector<U> pins_;
  mutable std::mutex mutex_;
  U pinned_bytes_ = 0;
  unsigned char* target_;
  const unsigned char* source_;
  U bytes_, chunk_, offset_ = 0, submitted_ = 0;
  bool pending_ = false;
  std::optional<Error> error_;
  EventStatus status_{EventState::pending, false, {}};
  void remember(Error e) { if (!error_) error_ = std::move(e); }
  void release_pins() {
    for (const auto cookie : pins_) {
      auto r = driver_->unpin(cookie);
      if (!r) remember(*r.error);
    }
    pins_.clear();
    pinned_bytes_ = 0;
    owner_.reset();
  }
  void pin_range(const void* pointer, U bytes, U page) {
    const U address = reinterpret_cast<std::uintptr_t>(pointer);
    require(bytes <= std::numeric_limits<std::uintptr_t>::max() - address, "SDMA address range overflow");
    U first = address & ~(page - 1);
    const U end = add(address, bytes);
    require(end <= std::numeric_limits<U>::max() - (page - 1), "SDMA pin envelope overflow");
    const U last = (end + page - 1) & ~(page - 1);
    const U limit = U(UINT32_MAX) & ~(page - 1);
    while (first < last) {
      const U n = std::min(limit, last - first);
      pins_.push_back(take(driver_->pin(reinterpret_cast<void*>(first), static_cast<std::uint32_t>(n))));
      pinned_bytes_ = add(pinned_bytes_, n);
      first += n;
    }
  }
public:
  Operation(std::unique_ptr<LinuxSdmaDriver> driver, void* target, const void* source, U bytes,
            const LinuxSdmaOptions& o, std::shared_ptr<void> owner)
    : driver_(std::move(driver)), owner_(std::move(owner)), target_(static_cast<unsigned char*>(target)),
      source_(static_cast<const unsigned char*>(source)), bytes_(bytes), chunk_(o.chunk_bytes) {
    try {
      if (bytes) { pin_range(source, bytes, o.page_bytes); pin_range(target, bytes, o.page_bytes); }
    } catch (...) { release_pins(); throw; }
  }
  ~Operation() override {
    // Runtime never destroys an operation while pending: its event quarantine
    // retains it. Driver-level users must obey the same ownership contract.
    if (!pending_) release_pins();
  }
  void stop_submitting(Error error) override {
    std::lock_guard lock(mutex_);
    remember(std::move(error));
  }
  LinuxTransferStatistics statistics() const override {
    std::lock_guard lock(mutex_);
    return {bytes_, pinned_bytes_, pins_.size()};
  }
  Result<EventStatus> query() override {
    std::lock_guard lock(mutex_);
    if (stopped(status_)) return {status_};
    if (pending_) {
      auto r = driver_->completed();
      if (!r) remember(std::get<Error>(r.state));
      else if (r.value()) { pending_ = false; offset_ += submitted_; }
      if (pending_) {
        status_ = {error_ ? EventState::failed : EventState::pending, false, error_};
        return {status_};
      }
      std::atomic_thread_fence(std::memory_order_acquire);
    }
    if (error_ || offset_ == bytes_) {
      release_pins();
      inflight_.reset();
      status_ = {error_ ? EventState::failed : EventState::succeeded, true, error_};
      return {status_};
    }
    submitted_ = std::min(chunk_, bytes_ - offset_);
    // Set before SDK entry: a nonzero return may follow partial enqueue.
    pending_ = true;
    inflight_ = shared_from_this();
    std::atomic_thread_fence(std::memory_order_release);
    auto r = driver_->submit(target_ + offset_, source_ + offset_, static_cast<std::uint32_t>(submitted_));
    if (!r) remember(*r.error);
    status_ = {error_ ? EventState::failed : EventState::pending, false, error_};
    return {status_};
  }
};
class Executor final : public LinuxTransferExecutor {
  LinuxSdmaOptions options_;
  LinuxSdmaDriverFactory factory_;
public:
  Executor(LinuxSdmaOptions o, LinuxSdmaDriverFactory f) : options_(std::move(o)), factory_(std::move(f)) {}
  ProgressCapabilities progress() const override { return {ProgressMode::caller_driven, {}, 1, false, true}; }
  Result<std::shared_ptr<LinuxTransferOperation>> copy(void* target, const void* source, U bytes,
                                                       std::shared_ptr<void> owner) override {
    return protect([&]() -> std::shared_ptr<LinuxTransferOperation> {
      require(owner != nullptr, "SDMA operands require a lifetime owner");
      require(!bytes || (target && source), "null SDMA operand");
      const U t = reinterpret_cast<std::uintptr_t>(target), s = reinterpret_cast<std::uintptr_t>(source);
      require(bytes <= std::numeric_limits<std::uintptr_t>::max() - t &&
              bytes <= std::numeric_limits<std::uintptr_t>::max() - s, "SDMA address overflow");
      require(!bytes || t + bytes <= s || s + bytes <= t, "overlapping SDMA copy");
      return std::make_shared<Operation>(take(factory_()), target, source, bytes, options_, std::move(owner));
    });
  }
};
#ifdef VECOPS_RUNTIME_HAVE_SDMA
class SdkDriver final : public LinuxSdmaDriver {
  void* library_ = nullptr;
  int fd_ = -1;
  void* channel_ = nullptr;
  std::uint32_t pasid_ = 0;
  sdma_sqe_task_t task_{};
  sdma_request_t request_{};
  bool shared_ = false;
  bool receipt_ = false;
#define SDK(name) decltype(&::name) name##_ = nullptr
  SDK(sdma_alloc_chn); SDK(sdma_free_chn); SDK(sdma_copy_data);
  SDK(sdma_init_chn); SDK(sdma_deinit_chn); SDK(sdma_icopy_data); SDK(sdma_iquery_chn);
  SDK(sdma_query_chn); SDK(sdma_wait_chn); SDK(sdma_get_process_id);
  SDK(sdma_pin_umem); SDK(sdma_unpin_umem);
#undef SDK
  static void check(int rc, const char* name) {
    if (rc) throw Failure{Error{"sdma_error", std::string(name) + ": SDK code " + std::to_string(rc)}};
  }
  void cleanup() {
    if (channel_) {
      if (shared_) sdma_deinit_chn_(channel_);
      else sdma_free_chn_(channel_);
    }
    if (fd_ >= 0) ::close(fd_);
    if (library_) ::dlclose(library_);
    channel_ = library_ = nullptr; fd_ = -1;
  }
public:
  explicit SdkDriver(const LinuxSdmaOptions& options) {
    try {
      library_ = ::dlopen(options.library.c_str(), RTLD_NOW | RTLD_LOCAL);
      if (!library_) throw Failure{Error{"sdma_unavailable", ::dlerror()}};
#define LOAD(name) name##_ = reinterpret_cast<decltype(name##_)>(::dlsym(library_, #name)); require(name##_ != nullptr, "missing SDMA SDK symbol: " #name)
      LOAD(sdma_alloc_chn); LOAD(sdma_free_chn); LOAD(sdma_copy_data); LOAD(sdma_query_chn);
      LOAD(sdma_init_chn); LOAD(sdma_deinit_chn); LOAD(sdma_icopy_data); LOAD(sdma_iquery_chn);
      LOAD(sdma_wait_chn); LOAD(sdma_get_process_id); LOAD(sdma_pin_umem); LOAD(sdma_unpin_umem);
#undef LOAD
      fd_ = ::open(options.device.c_str(), O_RDWR | O_CLOEXEC);
      if (fd_ < 0) throw Failure{Error{"sdma_unavailable", "cannot open SDMA device: " + options.device}};
      check(sdma_get_process_id_(fd_, &pasid_), "sdma_get_process_id");
      shared_ = options.shared_channel.has_value();
      channel_ = shared_ ? sdma_init_chn_(fd_, *options.shared_channel) : sdma_alloc_chn_(fd_);
      require(channel_ != nullptr,
              shared_ ? "SDMA shared channel initialization failed" : "SDMA exclusive channel allocation failed");
    } catch (...) { cleanup(); throw; }
  }
  ~SdkDriver() override { cleanup(); }
  Result<U> pin(void* address, std::uint32_t bytes) override {
    return protect([&] { U cookie{}; check(sdma_pin_umem_(fd_, address, bytes, &cookie), "sdma_pin_umem"); return cookie; });
  }
  Result<void> unpin(U cookie) override {
    return protect([&] { check(sdma_unpin_umem_(fd_, cookie), "sdma_unpin_umem"); });
  }
  Result<void> submit(void* target, const void* source, std::uint32_t bytes) override {
    return protect([&] {
      task_ = {}; task_.src_addr = reinterpret_cast<std::uintptr_t>(source);
      task_.dst_addr = reinterpret_cast<std::uintptr_t>(target);
      task_.src_process_id = task_.dst_process_id = pasid_; task_.length = bytes;
      if (shared_) {
        request_ = {};
        receipt_ = false;
        const int rc = sdma_icopy_data_(channel_, &task_, 1, &request_);
        // The SDK may return failure after accepting a descriptor. Preserve a
        // populated receipt even then; an empty request is never stop proof.
        receipt_ = request_.req_cnt == 1;
        check(rc, "sdma_icopy_data");
        require(receipt_, "SDMA shared submission returned no completion receipt");
      } else {
        check(sdma_copy_data_(channel_, &task_, 1), "sdma_copy_data");
      }
    });
  }
  Result<bool> completed() override {
    return protect([&] {
      if (shared_) {
        require(receipt_, "SDMA shared submission has no certified request receipt");
        const int rc = sdma_iquery_chn_(channel_, &request_);
        if (rc == SDMA_TASK_UNFINISH || rc == SDMA_RNDCNT_ERR) return false;
        check(rc, "sdma_iquery_chn");
        return true;
      }
      const int rc = sdma_query_chn_(channel_, 1);
      if (rc == SDMA_TASK_UNFINISH) return false;
      check(rc, "sdma_query_chn");
      check(sdma_wait_chn_(channel_, 1), "sdma_wait_chn");
      return true;
    });
  }
};
#endif
} // namespace
bool linux_sdma_sdk_compiled() noexcept {
#ifdef VECOPS_RUNTIME_HAVE_SDMA
  return true;
#else
  return false;
#endif
}
Result<std::shared_ptr<LinuxTransferExecutor>> make_linux_sdma_executor(LinuxSdmaOptions o,
                                                                       LinuxSdmaDriverFactory factory) {
  return details::protect([&]() -> std::shared_ptr<LinuxTransferExecutor> {
    details::require(details::power2(o.page_bytes) && o.page_bytes <= (U(1) << 31), "invalid SDMA pin page size");
    details::require(o.page_bytes == static_cast<U>(::sysconf(_SC_PAGESIZE)), "SDMA pin quantum must match Linux base page size");
    details::require(!o.shared_channel || *o.shared_channel <= INT32_MAX, "invalid SDMA shared channel index");
    details::require(o.chunk_bytes > 0 && o.chunk_bytes <= UINT32_MAX, "invalid SDMA chunk size");
    if (!factory) {
#ifdef VECOPS_RUNTIME_HAVE_SDMA
      factory = [o]() -> Result<std::unique_ptr<LinuxSdmaDriver>> {
        return details::protect([&]() -> std::unique_ptr<LinuxSdmaDriver> { return std::make_unique<SdkDriver>(o); });
      };
#else
      throw details::Failure{Error{"sdma_unavailable", "runtime built without official mdk_sdma.h"}};
#endif
    }
    return std::make_shared<Executor>(std::move(o), std::move(factory));
  });
}
} // namespace vecops::memory::model
