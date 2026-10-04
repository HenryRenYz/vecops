/** @file Print.h
 * @brief Type-safe record formatting and synchronized kernel diagnostics.
 *
 * Each record is formatted privately using C++20 format_to, then transferred
 * to one C stream under its stream lock. The lock belongs to the shared C
 * runtime, not a mutex duplicated in each generated kernel DSO. Atomicity is
 * per record within this process, not across processes or other descriptors.
 * No ordering or computation-memory synchronization contract is provided to
 * callers. flush=true flushes the selected stream before releasing its lock.
 */
#pragma once

#include <cstdio>
#include <format>
#include <iterator>
#include <stdexcept>
#include <string_view>
#include <utility>

#include "vecops/CoreDefs.h"
#include "vecops/execution/details/PrintFormat.h"

namespace vecops::execution {

/** An invocation-private, allocation-free record, bounded in UTF-8 bytes.
 * The marker is reserved even when the record is not truncated. No terminator
 * byte is needed: sinks receive an explicit byte count. This byte buffer is
 * not an array of SIMD values. Formatting continues into a discard iterator
 * after overflow, so formatting side effects/errors retain normal semantics.
 */
template <std::size_t Capacity = 1024>
class PrintRecord {
  static_assert(Capacity > 0);
  char bytes_[Capacity];
  std::size_t size_ = 0;
  std::string_view marker_;
  bool truncated_ = false;
  bool emitted_ = false;

  struct Iterator {
    using difference_type = std::ptrdiff_t;
    using iterator_category = std::output_iterator_tag;
    PrintRecord* record;
    Iterator& operator*() {
      return *this;
    }
    Iterator& operator++() {
      return *this;
    }
    Iterator operator++(int) {
      return *this;
    }
    Iterator& operator=(char c) {
      if (record->size_ < Capacity - record->marker_.size())
        record->bytes_[record->size_++] = c;
      else
        record->truncated_ = true;
      return *this;
    }
  };

public:
  explicit PrintRecord(std::string_view marker = " ... (trunc)")
    : marker_(marker) {
    if (marker.size() > Capacity)
      throw std::invalid_argument("print truncation marker exceeds buffer capacity");
  }

  template <class... Args>
  VECOPS_ALWAYS_INLINE void
  append(std::format_string<decltype(details::print_argument(std::declval<Args>()))...> format, Args&&... args) {
    std::format_to(Iterator{this}, format, details::print_argument(std::forward<Args>(args))...);
  }

  /** Commit once, including the caller's end text, as one stream write. */
  void emit(bool stderr_target = false, bool flush = false) {
    if (emitted_)
      throw std::logic_error("print record was already emitted");
    emitted_ = true;
    if (truncated_) {
      // Do not leave a partial UTF-8 codepoint immediately before the marker.
      auto end = size_;
      while (end && (static_cast<unsigned char>(bytes_[end - 1]) & 0xc0) == 0x80)
        --end;
      if (end) {
        const auto lead = static_cast<unsigned char>(bytes_[end - 1]);
        const auto width = lead < 0x80 ? 1 : lead < 0xe0 ? 2 : lead < 0xf0 ? 3 : 4;
        if (size_ - end + 1 < width)
          size_ = end - 1;
      }
      for (auto c : marker_)
        bytes_[size_++] = c;
    }
    auto* stream = stderr_target ? stderr : stdout;
#if defined(_WIN32)
    _lock_file(stream);
#else
    ::flockfile(stream);
#endif
    const auto count = std::fwrite(bytes_, 1, size_, stream);
    const auto error = count != size_;
    const auto flush_error = flush && std::fflush(stream) != 0;
#if defined(_WIN32)
    _unlock_file(stream);
#else
    ::funlockfile(stream);
#endif
    if (error || flush_error)
      throw std::runtime_error("kernel diagnostic output failed");
  }
};

} // namespace vecops::execution
