// SPDX-FileCopyrightText: 2026 HenryRenYz and vecops contributors
// SPDX-License-Identifier: MIT
/** @file content_watch_reference.cpp
 * @brief dnotify descriptor budgets remain safe under reduced process limits.
 */
#include "ContentWatch.h"

#include <fstream>

int main(int argc, char** argv) {
  if (argc != 2)
    return 1;
  struct rlimit original {};
  if (::getrlimit(RLIMIT_NOFILE, &original))
    return 2;
  auto low = original;
  low.rlim_cur = std::min<rlim_t>(64, original.rlim_cur);
  if (::setrlimit(RLIMIT_NOFILE, &low))
    return 3;
  const auto root = std::filesystem::path(argv[1]);
  std::size_t accepted = 0;
  for (int i = 0; i < 200; ++i) {
    auto directory = root / std::to_string(i);
    std::filesystem::create_directory(directory);
    accepted += content_watch.add(directory);
  }
  if (!accepted)
    return 77; // dnotify unavailable or its reserved signals already owned.
  if (accepted > low.rlim_cur / 8)
    return 4;
  std::ofstream(root / "diagnostic.txt") << "FD budget preserved\n";
  if (!std::filesystem::is_regular_file(root / "diagnostic.txt"))
    return 5;
  // Shrinking a live observer closes its excess FDs and invalidates the cache.
  auto smaller = low;
  smaller.rlim_cur = std::min<rlim_t>(32, low.rlim_cur);
  const auto epoch = content_watch.epoch();
  if (::setrlimit(RLIMIT_NOFILE, &smaller))
    return 6;
  if (!content_watch.add(root / "0") || content_watch.epoch() <= epoch)
    return 7;
  return 0;
}
