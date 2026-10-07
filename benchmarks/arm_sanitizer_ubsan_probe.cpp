// Minimal C++ source used to validate a task-private Clang UBSan runtime.
// It intentionally avoids C++ standard-library headers and APIs: the probe
// tests compiler-rt selection, not the separate libstdc++ toolchain setup.

extern "C" int puts(const char*);

__attribute__((noinline)) int add(int lhs, int rhs) {
  volatile int left = lhs;
  volatile int right = rhs;
  return left + right;
}

int main(int argc, char**) {
  if (argc == 1) {
    const int result = add(40, 2);
    if (result != 42) return 2;
    puts("ubsan legal arithmetic: 42");
    return 0;
  }

  volatile int maximum = 2147483647;
  return add(maximum, 1);
}
