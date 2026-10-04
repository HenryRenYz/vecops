// Fixed-buffer formatting must not allocate, including floating-point fields.
#include <atomic>
#include <cstdlib>
#include <new>
#include <limits>
#include "vecops/execution/Print.h"

std::atomic<unsigned> allocations{0};
void* operator new(std::size_t size) {
  allocations.fetch_add(1);
  if (auto* result = std::malloc(size))
    return result;
  throw std::bad_alloc();
}
void* operator new[](std::size_t size) {
  return ::operator new(size);
}
void operator delete(void* value) noexcept {
  std::free(value);
}
void operator delete[](void* value) noexcept {
  std::free(value);
}
void operator delete(void* value, std::size_t) noexcept {
  std::free(value);
}
void operator delete[](void* value, std::size_t) noexcept {
  std::free(value);
}

int main() {
  const auto before = allocations.load();
  vecops::execution::PrintRecord<> numeric;
  numeric.append("{} {:.3f} {}\n", 42, 1.25, std::string_view{"text"});
  numeric.emit(false, true);
  vecops::execution::PrintRecord<16> utf8("[cut]");
  utf8.append("{}", std::string_view{"中文中文"});
  utf8.emit(false, true);
  vecops::execution::PrintRecord<5> empty_marker("");
  empty_marker.append("{}", "abcdef");
  empty_marker.emit(false, true);
  vecops::execution::PrintRecord<3> marker_only("END");
  marker_only.append("{}", "x");
  marker_only.emit(false, true);
  vecops::execution::PrintRecord<8> embedded("");
  embedded.append("{}", std::string_view{"a\0b", 3});
  embedded.emit(false, true);
  vecops::execution::PrintRecord<2048> extreme("");
  extreme.append("\n{:.3f}|{:+010.2f}|{:#.4g}|{:.0f}\n", 1e300, 0.5, 10.0, std::numeric_limits<double>::max());
  extreme.emit(false, true);
  if (allocations.load() != before)
    return 1;
  // The explicit standard-format oracle may allocate; it runs after the
  // allocation-free record checks and never participates in the print path.
  for (double value : {0.0, -0.0, 1e-300, 1e300, 12.343, 10.0, 0.00001, std::numeric_limits<double>::infinity(),
                       -std::numeric_limits<double>::infinity(), std::numeric_limits<double>::quiet_NaN()}) {
#define CHECK_FORMAT(spec)                                                                          \
  if (std::format(spec, value) != std::format(spec, vecops::execution::details::PrintFloat{value})) \
    return 2;
    CHECK_FORMAT("{}")
    CHECK_FORMAT("{:.2f}")
    CHECK_FORMAT("{:010.3E}")
    CHECK_FORMAT("{:#.4g}")
    CHECK_FORMAT("{:*^+20.3G}")
    CHECK_FORMAT("{:#a}")
    CHECK_FORMAT("{:#.6}")
#undef CHECK_FORMAT
  }
  return 0;
}
