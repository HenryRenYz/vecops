/** Private ARM standard-library capability probe; compiled with clang. */
#include <atomic>
#include <charconv>
#include <format>
#include <string_view>
#include <thread>

#include "vecops/execution/Print.h"

int main() {
  char formatted[64]{};
  const auto end = std::format_to(formatted, "{}:{:.3f}", 42, 1.25);
  if (std::string_view(formatted, static_cast<std::size_t>(end - formatted)) != "42:1.250")
    return 1;
  char converted[64]{};
  const auto result = std::to_chars(converted, converted + sizeof(converted), 1.25);
  if (result.ec != std::errc{} || std::string_view(converted, result.ptr - converted) != "1.25")
    return 2;
  std::atomic<int> ready{0};
  std::thread worker([&] {
    ready.store(1, std::memory_order_release);
    ready.notify_one();
  });
  ready.wait(0, std::memory_order_acquire);
  worker.join();
  if (ready.load(std::memory_order_acquire) != 1)
    return 3;
  vecops::execution::PrintRecord<128> record;
  record.append("Private libstdc++ probe passed: format={} chars={} atomic={}\n", 42, 1.25, 1);
  record.emit(false, true);
  return 0;
}
