//
// Created by renyz on 2026/5/9.
//
#include <gtest/gtest.h>
#include <vector>
#include <numeric>
#include <memory>
#include "vecops/array/Array.h"
#include "vecops/array/Gemm.h"

using namespace vecops;
using namespace vecops::array;

// ============================================================================
// Main Function
// ============================================================================

int main(int argc, char** argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
