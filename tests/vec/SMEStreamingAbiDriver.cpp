// Baseline-ISA launcher. No VL-dependent value crosses a prctl transition.
#include <array>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <sys/prctl.h>
#include "vecops/execution/VectorLength.h"

int vecops_stride_streaming_main();
extern "C" int vecops_streaming_abi_canary(int (*)(), std::uint64_t*);

int main(int argc, char** argv) {
  if (argc != 3) return 2;
  const auto vl = static_cast<std::uint32_t>(std::strtoul(argv[1], nullptr, 10));
  const auto svl = static_cast<std::uint32_t>(std::strtoul(argv[2], nullptr, 10));
  if (vl < 16 || vl > 256 || svl < 16 || svl > 256 ||
      (vl & (vl - 1)) || (svl & (svl - 1))) return 3;
  const int before_vl = ::prctl(51, 0UL, 0UL, 0UL, 0UL);
  const int before_svl = ::prctl(64, 0UL, 0UL, 0UL, 0UL);
  int result = 0;
  {
    const vecops::execution::ArmVectorLengths requested{vl, svl};
    const vecops::execution::ScopedArmVectorLengths guard(requested);
    if (vecops::execution::current_thread_vector_lengths() != requested) return 4;
    // Ordinary scalar metadata records lower64-bit callee-saved ABI values.
    std::array<std::uint64_t, 8> actual{};
    const int numerical = vecops_streaming_abi_canary(vecops_stride_streaming_main, actual.data());
    std::printf("queried_vl=%u queried_svl=%u numerical_exit=%d\n", vl, svl, numerical);
    if (numerical) result = numerical;
    for (std::size_t index = 0; index < actual.size(); ++index) {
      const std::uint64_t expected = (index + 8) * UINT64_C(0x0101010101010101);
      std::printf("d%zu expected=%016llx actual=%016llx\n", index + 8,
                  static_cast<unsigned long long>(expected),
                  static_cast<unsigned long long>(actual[index]));
      if (expected != actual[index]) result = 71;
    }
  }
  const int after_vl = ::prctl(51, 0UL, 0UL, 0UL, 0UL);
  const int after_svl = ::prctl(64, 0UL, 0UL, 0UL, 0UL);
  std::printf("before_vl=%d before_svl=%d after_vl=%d after_svl=%d abi_exit=%d\n",
              before_vl, before_svl, after_vl, after_svl, result);
  if (before_vl != after_vl || before_svl != after_svl) return 72;
  // Diagnostic I/O is strictly outside SM; bind loaded libraries to this PID.
  std::FILE* maps = std::fopen("/proc/self/maps", "r");
  if (!maps) return 73;
  char line[1024];
  while (std::fgets(line, sizeof(line), maps)) std::fputs(line, stdout);
  if (std::fclose(maps) != 0) return 74;
  return result;
}
