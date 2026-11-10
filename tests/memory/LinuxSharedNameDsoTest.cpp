// SPDX-FileCopyrightText: 2026 HenryRenYz and vecops contributors
// SPDX-License-Identifier: MIT
// Two ordinary RTLD_LOCAL consumers must coexist without allocation-name collisions.
#include <dlfcn.h>
#include <sys/stat.h>
#include <unistd.h>
#include <cerrno>
#include <iostream>
#include <set>
#include <stdexcept>
#include <string>

namespace {
void require(bool value, const std::string& message) {
  if (!value) throw std::runtime_error(message);
}
struct Consumer {
  void* module = nullptr;
  int (*allocate)(unsigned) = nullptr;
  unsigned (*count)() = nullptr;
  const char* (*error)() = nullptr;
  const char* (*name)(unsigned) = nullptr;
  void (*drain)() = nullptr;
  template <class T> T symbol(const char* id) {
    auto* value = ::dlsym(module, id);
    require(value != nullptr, std::string("missing fixture symbol: ") + id);
    return reinterpret_cast<T>(value);
  }
  explicit Consumer(const char* file) {
    module = ::dlopen(file, RTLD_NOW | RTLD_LOCAL);
    if (!module) throw std::runtime_error(::dlerror());
    try {
      allocate = symbol<decltype(allocate)>("shared_name_allocate");
      count = symbol<decltype(count)>("shared_name_count");
      error = symbol<decltype(error)>("shared_name_error");
      name = symbol<decltype(name)>("shared_name_at");
      drain = symbol<decltype(drain)>("shared_name_drain");
    } catch (...) { ::dlclose(module); throw; }
  }
  ~Consumer() {
    // A failed allocation can leave earlier successful fixture roots alive.
    if (drain) drain();
    if (module) ::dlclose(module);
  }
  Consumer(const Consumer&) = delete;
  Consumer& operator=(const Consumer&) = delete;
};
void collect(Consumer& consumer, std::set<std::string>& owned) {
  for (unsigned i = 0; i < consumer.count(); ++i) {
    const auto* exported = consumer.name(i);
    require(exported != nullptr, consumer.error());
    const std::string name(exported);
    require(!name.empty() && name.front() == '/' && name.find('/', 1) == std::string::npos,
            "invalid exported POSIX SHM name");
    require(owned.insert("/dev/shm" + name).second, "duplicate live publication name");
    struct stat metadata {};
    require(::lstat(("/dev/shm" + name).c_str(), &metadata) == 0 && S_ISREG(metadata.st_mode) &&
              metadata.st_uid == ::getuid() && metadata.st_size == ::sysconf(_SC_PAGESIZE) &&
              (metadata.st_mode & 0777) == 0600,
            "unexpected owned backing metadata");
  }
}
}
int main(int argc, char** argv) {
  std::set<std::string> owned;
  bool passed = false;
  try {
    require(argc == 3, "usage: LinuxSharedNameDsoTest consumerA.so consumerB.so");
    Consumer a(argv[1]), b(argv[2]);
    const int first = a.allocate(110);
    collect(a, owned);
    require(first == 0 && a.count() == 110, std::string("first consumer: ") + a.error());
    const int second = b.allocate(1);
    collect(b, owned);
    require(second == 0 && b.count() == 1, std::string("second consumer: ") + b.error());
    require(owned.size() == 111, "missing live backing");
    b.drain(); a.drain();
    require(a.count() == 0 && b.count() == 0, "fixture did not drain");
    passed = true;
  } catch (const std::exception& error) { std::cerr << error.what() << '\n'; }
  // No unlink/reaper: exact names must vanish through the ordinary owner lifecycle.
  for (const auto& name : owned) {
    struct stat metadata {};
    errno = 0;
    if (::lstat(name.c_str(), &metadata) == 0 || errno != ENOENT) {
      std::cerr << "owned publication not released: " << name << '\n';
      passed = false;
    }
  }
  if (passed) std::cout << "111 distinct single-page roots across two RTLD_LOCAL SDK consumers; all released\n";
  return passed ? 0 : 1;
}
