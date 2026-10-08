// SPDX-FileCopyrightText: 2026 HenryRenYz and vecops contributors
// SPDX-License-Identifier: MIT
#pragma once

#include <cstdint>
#include <span>

namespace vecops::bench::inst {

enum class MeasureMode {
  Control,
  Latency,
  Throughput,
};

enum class InstructionCategory {
  Control,
  Arithmetic,
  Shift,
  Bitwise,
  Blend,
  Memory,
};

enum class MemoryPattern {
  None,
  Hot,
  Stream,
  PointerChase,
  StoreForward,
};

enum class WorkingSetLevel {
  None,
  L1,
  L2,
  L3,
  BeyondLastCache,
};

enum class MemoryConversionKind {
  None,
  SignExtendLoad,
  ZeroExtendLoad,
  TruncateStore,
  SignedSaturatingStore,
  UnsignedSaturatingStore,
};

using Kernel = void (*)(uint64_t loops);
using MemoryKernel = void (*)(uint64_t loops, void* memory, uint64_t memory_bytes);

struct InstructionCase {
  const char* name;
  const char* isa;
  const char* description;
  MeasureMode mode;
  Kernel kernel;
  uint32_t sequences_per_loop;
  uint32_t instructions_per_sequence;
  uint32_t dependency_chains;
  uint32_t vector_bits;
  InstructionCategory category = InstructionCategory::Arithmetic;
  MemoryKernel memory_kernel = nullptr;
  MemoryPattern memory_pattern = MemoryPattern::None;
  WorkingSetLevel working_set = WorkingSetLevel::None;
  uint32_t bytes_per_sequence = 0;
  const char* baseline_name = "empty_loop";
  MemoryConversionKind conversion_kind = MemoryConversionKind::None;
  uint32_t source_element_bits = 0;
  uint32_t destination_element_bits = 0;
  uint32_t elements_per_sequence = 0;
  uint64_t working_set_bytes_cap = 0;
};

std::span<const InstructionCase> arch_instruction_cases();

constexpr const char* mode_name(MeasureMode mode) {
  switch (mode) {
    case MeasureMode::Control: return "control";
    case MeasureMode::Latency: return "latency";
    case MeasureMode::Throughput: return "throughput";
  }
  return "unknown";
}

constexpr const char* category_name(InstructionCategory category) {
  switch (category) {
    case InstructionCategory::Control: return "control";
    case InstructionCategory::Arithmetic: return "arithmetic";
    case InstructionCategory::Shift: return "shift";
    case InstructionCategory::Bitwise: return "bitwise";
    case InstructionCategory::Blend: return "blend";
    case InstructionCategory::Memory: return "memory";
  }
  return "unknown";
}

constexpr const char* memory_pattern_name(MemoryPattern pattern) {
  switch (pattern) {
    case MemoryPattern::None: return "none";
    case MemoryPattern::Hot: return "hot";
    case MemoryPattern::Stream: return "stream";
    case MemoryPattern::PointerChase: return "pointer_chase";
    case MemoryPattern::StoreForward: return "store_forward";
  }
  return "unknown";
}

constexpr const char* working_set_level_name(WorkingSetLevel level) {
  switch (level) {
    case WorkingSetLevel::None: return "none";
    case WorkingSetLevel::L1: return "l1";
    case WorkingSetLevel::L2: return "l2";
    case WorkingSetLevel::L3: return "l3";
    case WorkingSetLevel::BeyondLastCache: return "beyond_last_reported_cache";
  }
  return "unknown";
}

constexpr const char* memory_conversion_kind_name(MemoryConversionKind kind) {
  switch (kind) {
    case MemoryConversionKind::None: return "none";
    case MemoryConversionKind::SignExtendLoad: return "sign_extend_load";
    case MemoryConversionKind::ZeroExtendLoad: return "zero_extend_load";
    case MemoryConversionKind::TruncateStore: return "truncate_store";
    case MemoryConversionKind::SignedSaturatingStore:
      return "signed_saturating_store";
    case MemoryConversionKind::UnsignedSaturatingStore:
      return "unsigned_saturating_store";
  }
  return "unknown";
}

} // namespace vecops::bench::inst

#define VECOPS_INST_REPEAT_2(x) x x
#define VECOPS_INST_REPEAT_4(x) VECOPS_INST_REPEAT_2(x) VECOPS_INST_REPEAT_2(x)
#define VECOPS_INST_REPEAT_8(x) VECOPS_INST_REPEAT_4(x) VECOPS_INST_REPEAT_4(x)
#define VECOPS_INST_REPEAT_16(x) VECOPS_INST_REPEAT_8(x) VECOPS_INST_REPEAT_8(x)
#define VECOPS_INST_REPEAT_32(x) VECOPS_INST_REPEAT_16(x) VECOPS_INST_REPEAT_16(x)
