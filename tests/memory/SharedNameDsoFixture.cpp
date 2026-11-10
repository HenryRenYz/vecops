// SPDX-FileCopyrightText: 2026 HenryRenYz and vecops contributors
// SPDX-License-Identifier: MIT
// Independently linked SDK consumer; intentionally no symbol-binding overrides.
#include "vecops/memory/LinuxBackend.h"
#include <any>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>
#include <unistd.h>

namespace {
namespace mm = vecops::memory::model;
mm::LinuxPageStore store{{{"fixture-memory", {}}}};
std::vector<std::shared_ptr<mm::SharedBacking>> roots;
std::string error;
std::string name;
template <class T> T take(mm::Result<T> result) {
  if (!result) throw std::runtime_error(std::get<mm::Error>(result.state).message);
  return std::move(result).value();
}
}
extern "C" int shared_name_allocate(unsigned count) noexcept {
  try {
    error.clear();
    const auto page = static_cast<std::uint64_t>(::sysconf(_SC_PAGESIZE));
    for (unsigned i = 0; i < count; ++i)
      roots.push_back(take(store.allocate({"root-" + std::to_string(roots.size()), "fixture", "fixture-memory",
                                          page, page, mm::BackingMode::shareable_remappable, {}})));
    return 0;
  } catch (const std::exception& e) { error = e.what(); return 1; }
  catch (...) { error = "unknown C++ exception"; return 2; }
}
extern "C" unsigned shared_name_count() noexcept { return static_cast<unsigned>(roots.size()); }
extern "C" const char* shared_name_error() noexcept { return error.c_str(); }
extern "C" const char* shared_name_at(unsigned index) noexcept {
  try {
    name = std::any_cast<mm::LinuxBackingHandle>(take(roots.at(index)->export_handle())).name;
    return name.c_str();
  } catch (const std::exception& e) { error = e.what(); return nullptr; }
  catch (...) { error = "unknown C++ exception"; return nullptr; }
}
extern "C" void shared_name_drain() noexcept { roots.clear(); }
