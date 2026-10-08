// SPDX-FileCopyrightText: 2026 HenryRenYz and vecops contributors
// SPDX-License-Identifier: MIT
#include "InstructionBench.h"

#include <cstdint>
#include <span>

#if !defined(__x86_64__) && !defined(__i386__)
#error "InstructionBenchX86.cpp requires an x86 target"
#endif

namespace vecops::bench::inst {
namespace {

#if defined(__GNUC__) || defined(__clang__)
#define VECOPS_INST_NOINLINE_USED __attribute__((noinline, used))
#else
#define VECOPS_INST_NOINLINE_USED
#endif

#define X86_DEFINE_KERNEL(name, setup, body, ...)                            \
  extern "C" VECOPS_INST_NOINLINE_USED void name(uint64_t loops) {          \
    if (loops == 0) return;                                                  \
    __asm__ volatile(                                                        \
        setup                                                               \
        ".p2align 6\n\t"                                                    \
        "1:\n\t"                                                          \
        body                                                                \
        "dec %[loops]\n\t"                                                \
        "jnz 1b\n\t"                                                      \
        : [loops] "+r"(loops)                                              \
        :                                                                   \
        : "cc" __VA_OPT__(,) __VA_ARGS__);                                 \
  }

extern "C" VECOPS_INST_NOINLINE_USED void x86_control_empty_loop(uint64_t loops) {
  if (loops == 0) return;
  __asm__ volatile(
      ".p2align 6\n\t"
      "1:\n\t"
      "dec %[loops]\n\t"
      "jnz 1b\n\t"
      : [loops] "+r"(loops)
      :
      : "cc");
}

X86_DEFINE_KERNEL(
    x86_control_nop_32,
    "",
    VECOPS_INST_REPEAT_32("nop\n\t"))

#if defined(__AVX512F__)

#define X86_ZMM_SETUP                                                        \
  "vpxord %%zmm0, %%zmm0, %%zmm0\n\t"                                         \
  "vpxord %%zmm1, %%zmm1, %%zmm1\n\t"                                         \
  "vpxord %%zmm2, %%zmm2, %%zmm2\n\t"                                         \
  "vpxord %%zmm3, %%zmm3, %%zmm3\n\t"                                         \
  "vpxord %%zmm4, %%zmm4, %%zmm4\n\t"                                         \
  "vpxord %%zmm5, %%zmm5, %%zmm5\n\t"                                         \
  "vpxord %%zmm6, %%zmm6, %%zmm6\n\t"                                         \
  "vpxord %%zmm7, %%zmm7, %%zmm7\n\t"                                         \
  "vpxord %%zmm16, %%zmm16, %%zmm16\n\t"                                      \
  "mov $7, %%eax\n\t"                                                       \
  "vmovd %%eax, %%xmm17\n\t"                                               \
  "vpbroadcastw %%xmm17, %%zmm17\n\t"                                      \
  "vmovd %%eax, %%xmm18\n\t"                                               \
  "vpbroadcastd %%xmm18, %%zmm18\n\t"                                      \
  "vmovd %%eax, %%xmm19\n\t"                                               \
  "vpbroadcastq %%xmm19, %%zmm19\n\t"                                      \
  "mov $0xaaaa, %%eax\n\t"                                                 \
  "kmovw %%eax, %%k1\n\t"                                                  \
  "mov $0xaa, %%eax\n\t"                                                   \
  "kmovb %%eax, %%k2\n\t"                                                  \
  "mov $0xaaaaaaaa, %%eax\n\t"                                             \
  "kmovd %%eax, %%k3\n\t"                                                  \
  "movabs $0xaaaaaaaaaaaaaaaa, %%rax\n\t"                                   \
  "kmovq %%rax, %%k4\n\t"

#define X86_ZMM_CLOBBERS                                                     \
  "zmm0", "zmm1", "zmm2", "zmm3", "zmm4", "zmm5", "zmm6", "zmm7", \
      "zmm16", "zmm17", "zmm18", "zmm19", "k1", "k2", "k3", "k4", "rax"

#define X86_ZMM_BINARY_GROUP(op)                                             \
  op " %%zmm16, %%zmm0, %%zmm0\n\t"                                           \
  op " %%zmm16, %%zmm1, %%zmm1\n\t"                                           \
  op " %%zmm16, %%zmm2, %%zmm2\n\t"                                           \
  op " %%zmm16, %%zmm3, %%zmm3\n\t"                                           \
  op " %%zmm16, %%zmm4, %%zmm4\n\t"                                           \
  op " %%zmm16, %%zmm5, %%zmm5\n\t"                                           \
  op " %%zmm16, %%zmm6, %%zmm6\n\t"                                           \
  op " %%zmm16, %%zmm7, %%zmm7\n\t"

#define X86_ZMM_FMA_GROUP                                                    \
  "vfmadd231ps %%zmm17, %%zmm16, %%zmm0\n\t"                                \
  "vfmadd231ps %%zmm17, %%zmm16, %%zmm1\n\t"                                \
  "vfmadd231ps %%zmm17, %%zmm16, %%zmm2\n\t"                                \
  "vfmadd231ps %%zmm17, %%zmm16, %%zmm3\n\t"                                \
  "vfmadd231ps %%zmm17, %%zmm16, %%zmm4\n\t"                                \
  "vfmadd231ps %%zmm17, %%zmm16, %%zmm5\n\t"                                \
  "vfmadd231ps %%zmm17, %%zmm16, %%zmm6\n\t"                                \
  "vfmadd231ps %%zmm17, %%zmm16, %%zmm7\n\t"

#define X86_ZMM_SHUFFLE_GROUP                                                \
  "vshufi32x4 $0, %%zmm16, %%zmm0, %%zmm0\n\t"                              \
  "vshufi32x4 $0, %%zmm16, %%zmm1, %%zmm1\n\t"                              \
  "vshufi32x4 $0, %%zmm16, %%zmm2, %%zmm2\n\t"                              \
  "vshufi32x4 $0, %%zmm16, %%zmm3, %%zmm3\n\t"                              \
  "vshufi32x4 $0, %%zmm16, %%zmm4, %%zmm4\n\t"                              \
  "vshufi32x4 $0, %%zmm16, %%zmm5, %%zmm5\n\t"                              \
  "vshufi32x4 $0, %%zmm16, %%zmm6, %%zmm6\n\t"                              \
  "vshufi32x4 $0, %%zmm16, %%zmm7, %%zmm7\n\t"

#define X86_ZMM_IMMEDIATE_GROUP(op)                                          \
  op " $7, %%zmm0, %%zmm0\n\t"                                             \
  op " $7, %%zmm1, %%zmm1\n\t"                                             \
  op " $7, %%zmm2, %%zmm2\n\t"                                             \
  op " $7, %%zmm3, %%zmm3\n\t"                                             \
  op " $7, %%zmm4, %%zmm4\n\t"                                             \
  op " $7, %%zmm5, %%zmm5\n\t"                                             \
  op " $7, %%zmm6, %%zmm6\n\t"                                             \
  op " $7, %%zmm7, %%zmm7\n\t"

#define X86_ZMM_COUNT_GROUP(op, count)                                       \
  op " %%" count ", %%zmm0, %%zmm0\n\t"                                    \
  op " %%" count ", %%zmm1, %%zmm1\n\t"                                    \
  op " %%" count ", %%zmm2, %%zmm2\n\t"                                    \
  op " %%" count ", %%zmm3, %%zmm3\n\t"                                    \
  op " %%" count ", %%zmm4, %%zmm4\n\t"                                    \
  op " %%" count ", %%zmm5, %%zmm5\n\t"                                    \
  op " %%" count ", %%zmm6, %%zmm6\n\t"                                    \
  op " %%" count ", %%zmm7, %%zmm7\n\t"

#define X86_ZMM_BLEND_GROUP(op, mask)                                        \
  op " %%zmm16, %%zmm0, %%zmm0%{%%" mask "%}\n\t"                          \
  op " %%zmm16, %%zmm1, %%zmm1%{%%" mask "%}\n\t"                          \
  op " %%zmm16, %%zmm2, %%zmm2%{%%" mask "%}\n\t"                          \
  op " %%zmm16, %%zmm3, %%zmm3%{%%" mask "%}\n\t"                          \
  op " %%zmm16, %%zmm4, %%zmm4%{%%" mask "%}\n\t"                          \
  op " %%zmm16, %%zmm5, %%zmm5%{%%" mask "%}\n\t"                          \
  op " %%zmm16, %%zmm6, %%zmm6%{%%" mask "%}\n\t"                          \
  op " %%zmm16, %%zmm7, %%zmm7%{%%" mask "%}\n\t"

#define X86_ZMM_NOT_GROUP                                                   \
  "vpternlogd $0x55, %%zmm0, %%zmm0, %%zmm0\n\t"                            \
  "vpternlogd $0x55, %%zmm1, %%zmm1, %%zmm1\n\t"                            \
  "vpternlogd $0x55, %%zmm2, %%zmm2, %%zmm2\n\t"                            \
  "vpternlogd $0x55, %%zmm3, %%zmm3, %%zmm3\n\t"                            \
  "vpternlogd $0x55, %%zmm4, %%zmm4, %%zmm4\n\t"                            \
  "vpternlogd $0x55, %%zmm5, %%zmm5, %%zmm5\n\t"                            \
  "vpternlogd $0x55, %%zmm6, %%zmm6, %%zmm6\n\t"                            \
  "vpternlogd $0x55, %%zmm7, %%zmm7, %%zmm7\n\t"

#define DEFINE_ZMM_LAT_THR(name, instruction, group)                         \
  X86_DEFINE_KERNEL(                                                         \
      name##_latency, X86_ZMM_SETUP,                                         \
      VECOPS_INST_REPEAT_32(instruction), X86_ZMM_CLOBBERS)                  \
  X86_DEFINE_KERNEL(                                                         \
      name##_throughput, X86_ZMM_SETUP,                                      \
      VECOPS_INST_REPEAT_4(group), X86_ZMM_CLOBBERS)

DEFINE_ZMM_LAT_THR(
    x86_vaddps_zmm, "vaddps %%zmm16, %%zmm0, %%zmm0\n\t", X86_ZMM_BINARY_GROUP("vaddps"))
DEFINE_ZMM_LAT_THR(
    x86_vmulps_zmm, "vmulps %%zmm16, %%zmm0, %%zmm0\n\t", X86_ZMM_BINARY_GROUP("vmulps"))
DEFINE_ZMM_LAT_THR(
    x86_vfmadd231ps_zmm,
    "vfmadd231ps %%zmm17, %%zmm16, %%zmm0\n\t",
    X86_ZMM_FMA_GROUP)
DEFINE_ZMM_LAT_THR(
    x86_vpaddd_zmm, "vpaddd %%zmm16, %%zmm0, %%zmm0\n\t", X86_ZMM_BINARY_GROUP("vpaddd"))
DEFINE_ZMM_LAT_THR(
    x86_vpmulld_zmm, "vpmulld %%zmm16, %%zmm0, %%zmm0\n\t", X86_ZMM_BINARY_GROUP("vpmulld"))
DEFINE_ZMM_LAT_THR(
    x86_vpandd_zmm, "vpandd %%zmm16, %%zmm0, %%zmm0\n\t", X86_ZMM_BINARY_GROUP("vpandd"))
DEFINE_ZMM_LAT_THR(
    x86_vpord_zmm, "vpord %%zmm16, %%zmm0, %%zmm0\n\t", X86_ZMM_BINARY_GROUP("vpord"))
DEFINE_ZMM_LAT_THR(
    x86_vpxord_zmm, "vpxord %%zmm16, %%zmm0, %%zmm0\n\t", X86_ZMM_BINARY_GROUP("vpxord"))
DEFINE_ZMM_LAT_THR(
    x86_vpandnd_zmm, "vpandnd %%zmm16, %%zmm0, %%zmm0\n\t", X86_ZMM_BINARY_GROUP("vpandnd"))
DEFINE_ZMM_LAT_THR(
    x86_vpternlogd_not_zmm,
    "vpternlogd $0x55, %%zmm0, %%zmm0, %%zmm0\n\t",
    X86_ZMM_NOT_GROUP)
DEFINE_ZMM_LAT_THR(
    x86_vpermd_zmm, "vpermd %%zmm16, %%zmm0, %%zmm0\n\t", X86_ZMM_BINARY_GROUP("vpermd"))
DEFINE_ZMM_LAT_THR(
    x86_vshufi32x4_zmm,
    "vshufi32x4 $0, %%zmm16, %%zmm0, %%zmm0\n\t",
    X86_ZMM_SHUFFLE_GROUP)

#define DEFINE_ZMM_IMMEDIATE_SHIFT(name, op)                                 \
  DEFINE_ZMM_LAT_THR(name, op " $7, %%zmm0, %%zmm0\n\t", X86_ZMM_IMMEDIATE_GROUP(op))
#define DEFINE_ZMM_COUNT_SHIFT(name, op, count)                              \
  DEFINE_ZMM_LAT_THR(name, op " %%" count ", %%zmm0, %%zmm0\n\t",          \
                     X86_ZMM_COUNT_GROUP(op, count))

DEFINE_ZMM_IMMEDIATE_SHIFT(x86_vpsllw_zmm_imm7, "vpsllw")
DEFINE_ZMM_IMMEDIATE_SHIFT(x86_vpslld_zmm_imm7, "vpslld")
DEFINE_ZMM_IMMEDIATE_SHIFT(x86_vpsllq_zmm_imm7, "vpsllq")
DEFINE_ZMM_IMMEDIATE_SHIFT(x86_vpsrlw_zmm_imm7, "vpsrlw")
DEFINE_ZMM_IMMEDIATE_SHIFT(x86_vpsrld_zmm_imm7, "vpsrld")
DEFINE_ZMM_IMMEDIATE_SHIFT(x86_vpsrlq_zmm_imm7, "vpsrlq")
DEFINE_ZMM_IMMEDIATE_SHIFT(x86_vpsraw_zmm_imm7, "vpsraw")
DEFINE_ZMM_IMMEDIATE_SHIFT(x86_vpsrad_zmm_imm7, "vpsrad")
DEFINE_ZMM_IMMEDIATE_SHIFT(x86_vpsraq_zmm_imm7, "vpsraq")

DEFINE_ZMM_COUNT_SHIFT(x86_vpsllw_zmm_xmm, "vpsllw", "xmm19")
DEFINE_ZMM_COUNT_SHIFT(x86_vpslld_zmm_xmm, "vpslld", "xmm19")
DEFINE_ZMM_COUNT_SHIFT(x86_vpsllq_zmm_xmm, "vpsllq", "xmm19")
DEFINE_ZMM_COUNT_SHIFT(x86_vpsrlw_zmm_xmm, "vpsrlw", "xmm19")
DEFINE_ZMM_COUNT_SHIFT(x86_vpsrld_zmm_xmm, "vpsrld", "xmm19")
DEFINE_ZMM_COUNT_SHIFT(x86_vpsrlq_zmm_xmm, "vpsrlq", "xmm19")
DEFINE_ZMM_COUNT_SHIFT(x86_vpsraw_zmm_xmm, "vpsraw", "xmm19")
DEFINE_ZMM_COUNT_SHIFT(x86_vpsrad_zmm_xmm, "vpsrad", "xmm19")
DEFINE_ZMM_COUNT_SHIFT(x86_vpsraq_zmm_xmm, "vpsraq", "xmm19")

DEFINE_ZMM_COUNT_SHIFT(x86_vpsllvw_zmm, "vpsllvw", "zmm17")
DEFINE_ZMM_COUNT_SHIFT(x86_vpsllvd_zmm, "vpsllvd", "zmm18")
DEFINE_ZMM_COUNT_SHIFT(x86_vpsllvq_zmm, "vpsllvq", "zmm19")
DEFINE_ZMM_COUNT_SHIFT(x86_vpsrlvw_zmm, "vpsrlvw", "zmm17")
DEFINE_ZMM_COUNT_SHIFT(x86_vpsrlvd_zmm, "vpsrlvd", "zmm18")
DEFINE_ZMM_COUNT_SHIFT(x86_vpsrlvq_zmm, "vpsrlvq", "zmm19")
DEFINE_ZMM_COUNT_SHIFT(x86_vpsravw_zmm, "vpsravw", "zmm17")
DEFINE_ZMM_COUNT_SHIFT(x86_vpsravd_zmm, "vpsravd", "zmm18")
DEFINE_ZMM_COUNT_SHIFT(x86_vpsravq_zmm, "vpsravq", "zmm19")

DEFINE_ZMM_LAT_THR(
    x86_vpblendmd_zmm,
    "vpblendmd %%zmm16, %%zmm0, %%zmm0%{%%k1%}\n\t",
    X86_ZMM_BLEND_GROUP("vpblendmd", "k1"))
DEFINE_ZMM_LAT_THR(
    x86_vpblendmq_zmm,
    "vpblendmq %%zmm16, %%zmm0, %%zmm0%{%%k2%}\n\t",
    X86_ZMM_BLEND_GROUP("vpblendmq", "k2"))
DEFINE_ZMM_LAT_THR(
    x86_vpblendmw_zmm,
    "vpblendmw %%zmm16, %%zmm0, %%zmm0%{%%k3%}\n\t",
    X86_ZMM_BLEND_GROUP("vpblendmw", "k3"))
DEFINE_ZMM_LAT_THR(
    x86_vpblendmb_zmm,
    "vpblendmb %%zmm16, %%zmm0, %%zmm0%{%%k4%}\n\t",
    X86_ZMM_BLEND_GROUP("vpblendmb", "k4"))

#define X86_UNARY_8(op, src, dst)                                            \
  op " " src ", %%" dst "0\n\t"                                           \
  op " " src ", %%" dst "1\n\t"                                           \
  op " " src ", %%" dst "2\n\t"                                           \
  op " " src ", %%" dst "3\n\t"                                           \
  op " " src ", %%" dst "4\n\t"                                           \
  op " " src ", %%" dst "5\n\t"                                           \
  op " " src ", %%" dst "6\n\t"                                           \
  op " " src ", %%" dst "7\n\t"

X86_DEFINE_KERNEL(
    x86_vcvtph2ps_ymm_to_zmm_throughput,
    X86_ZMM_SETUP,
    VECOPS_INST_REPEAT_4(X86_UNARY_8("vcvtph2ps", "%%ymm16", "zmm")),
    X86_ZMM_CLOBBERS)
X86_DEFINE_KERNEL(
    x86_vcvtps2ph_zmm_to_ymm_throughput,
    X86_ZMM_SETUP,
    VECOPS_INST_REPEAT_4(X86_UNARY_8("vcvtps2ph $0,", "%%zmm16", "ymm")),
    X86_ZMM_CLOBBERS)
X86_DEFINE_KERNEL(
    x86_vcvtps2pd_ymm_to_zmm_throughput,
    X86_ZMM_SETUP,
    VECOPS_INST_REPEAT_4(X86_UNARY_8("vcvtps2pd", "%%ymm16", "zmm")),
    X86_ZMM_CLOBBERS)
X86_DEFINE_KERNEL(
    x86_vcvtpd2ps_zmm_to_ymm_throughput,
    X86_ZMM_SETUP,
    VECOPS_INST_REPEAT_4(X86_UNARY_8("vcvtpd2ps", "%%zmm16", "ymm")),
    X86_ZMM_CLOBBERS)
X86_DEFINE_KERNEL(
    x86_vpmovsxwd_ymm_to_zmm_throughput,
    X86_ZMM_SETUP,
    VECOPS_INST_REPEAT_4(X86_UNARY_8("vpmovsxwd", "%%ymm16", "zmm")),
    X86_ZMM_CLOBBERS)
X86_DEFINE_KERNEL(
    x86_vpmovdw_zmm_to_ymm_throughput,
    X86_ZMM_SETUP,
    VECOPS_INST_REPEAT_4(X86_UNARY_8("vpmovdw", "%%zmm16", "ymm")),
    X86_ZMM_CLOBBERS)

X86_DEFINE_KERNEL(
    x86_vcvtph2ps_vcvtps2ph_latency,
    X86_ZMM_SETUP,
    VECOPS_INST_REPEAT_16(
        "vcvtph2ps %%ymm0, %%zmm0\n\t"
        "vcvtps2ph $0, %%zmm0, %%ymm0\n\t"),
    X86_ZMM_CLOBBERS)
X86_DEFINE_KERNEL(
    x86_vcvtps2pd_vcvtpd2ps_latency,
    X86_ZMM_SETUP,
    VECOPS_INST_REPEAT_16(
        "vcvtps2pd %%ymm0, %%zmm0\n\t"
        "vcvtpd2ps %%zmm0, %%ymm0\n\t"),
    X86_ZMM_CLOBBERS)
X86_DEFINE_KERNEL(
    x86_vpmovsxwd_vpmovdw_latency,
    X86_ZMM_SETUP,
    VECOPS_INST_REPEAT_16(
        "vpmovsxwd %%ymm0, %%zmm0\n\t"
        "vpmovdw %%zmm0, %%ymm0\n\t"),
    X86_ZMM_CLOBBERS)

#define X86_DEFINE_MEMORY_KERNEL(name, setup, body, ...)                     \
  extern "C" VECOPS_INST_NOINLINE_USED void name(                           \
      uint64_t loops, void* memory, uint64_t memory_bytes) {                 \
    if (loops == 0 || memory_bytes == 0) return;                             \
    auto* base = static_cast<uint8_t*>(memory);                              \
    auto* ptr = base;                                                        \
    auto* end = base + memory_bytes;                                         \
    __asm__ volatile(                                                        \
        setup                                                               \
        ".p2align 6\n\t"                                                    \
        "1:\n\t"                                                            \
        body                                                                \
        "dec %[loops]\n\t"                                                  \
        "jnz 1b\n\t"                                                        \
        : [loops] "+r"(loops), [ptr] "+&r"(ptr)                             \
        : [base] "r"(base), [end] "r"(end)                                 \
        : "cc", "memory" __VA_OPT__(,) __VA_ARGS__);                        \
  }

#define X86_ZMM_STREAM_ADVANCE                                               \
  "add $512, %[ptr]\n\t"                                                    \
  "cmp %[end], %[ptr]\n\t"                                                  \
  "cmovae %[base], %[ptr]\n\t"

X86_DEFINE_MEMORY_KERNEL(
    x86_zmm_stream_control,
    "",
    X86_ZMM_STREAM_ADVANCE)
X86_DEFINE_MEMORY_KERNEL(
    x86_zmm_stream_load,
    "",
    "vmovdqa64 0(%[ptr]), %%zmm0\n\t"
    "vmovdqa64 64(%[ptr]), %%zmm1\n\t"
    "vmovdqa64 128(%[ptr]), %%zmm2\n\t"
    "vmovdqa64 192(%[ptr]), %%zmm3\n\t"
    "vmovdqa64 256(%[ptr]), %%zmm4\n\t"
    "vmovdqa64 320(%[ptr]), %%zmm5\n\t"
    "vmovdqa64 384(%[ptr]), %%zmm6\n\t"
    "vmovdqa64 448(%[ptr]), %%zmm7\n\t"
    X86_ZMM_STREAM_ADVANCE,
    "zmm0", "zmm1", "zmm2", "zmm3", "zmm4", "zmm5", "zmm6", "zmm7")
X86_DEFINE_MEMORY_KERNEL(
    x86_zmm_stream_store,
    "vpxord %%zmm16, %%zmm16, %%zmm16\n\t",
    "vmovdqa64 %%zmm16, 0(%[ptr])\n\t"
    "vmovdqa64 %%zmm16, 64(%[ptr])\n\t"
    "vmovdqa64 %%zmm16, 128(%[ptr])\n\t"
    "vmovdqa64 %%zmm16, 192(%[ptr])\n\t"
    "vmovdqa64 %%zmm16, 256(%[ptr])\n\t"
    "vmovdqa64 %%zmm16, 320(%[ptr])\n\t"
    "vmovdqa64 %%zmm16, 384(%[ptr])\n\t"
    "vmovdqa64 %%zmm16, 448(%[ptr])\n\t"
    X86_ZMM_STREAM_ADVANCE,
    "zmm16")
X86_DEFINE_MEMORY_KERNEL(
    x86_zmm_pointer_chase,
    "",
    "vmovdqa64 (%[ptr]), %%zmm0\n\t"
    "vmovq %%xmm0, %[ptr]\n\t",
    "zmm0")
X86_DEFINE_MEMORY_KERNEL(
    x86_zmm_store_forward,
    "vpxord %%zmm0, %%zmm0, %%zmm0\n\t",
    "vmovdqa64 %%zmm0, (%[ptr])\n\t"
    "vmovdqa64 (%[ptr]), %%zmm0\n\t",
    "zmm0")
X86_DEFINE_MEMORY_KERNEL(
    x86_zmm_crossline_load,
    "add $32, %[ptr]\n\t",
    "vmovdqu64 0(%[ptr]), %%zmm0\n\t"
    "vmovdqu64 64(%[ptr]), %%zmm1\n\t"
    "vmovdqu64 128(%[ptr]), %%zmm2\n\t"
    "vmovdqu64 192(%[ptr]), %%zmm3\n\t"
    "vmovdqu64 256(%[ptr]), %%zmm4\n\t"
    "vmovdqu64 320(%[ptr]), %%zmm5\n\t"
    "vmovdqu64 384(%[ptr]), %%zmm6\n\t"
    "vmovdqu64 448(%[ptr]), %%zmm7\n\t",
    "zmm0", "zmm1", "zmm2", "zmm3", "zmm4", "zmm5", "zmm6", "zmm7")
X86_DEFINE_MEMORY_KERNEL(
    x86_zmm_crossline_store,
    "add $32, %[ptr]\n\t"
    "vpxord %%zmm16, %%zmm16, %%zmm16\n\t",
    "vmovdqu64 %%zmm16, 0(%[ptr])\n\t"
    "vmovdqu64 %%zmm16, 64(%[ptr])\n\t"
    "vmovdqu64 %%zmm16, 128(%[ptr])\n\t"
    "vmovdqu64 %%zmm16, 192(%[ptr])\n\t"
    "vmovdqu64 %%zmm16, 256(%[ptr])\n\t"
    "vmovdqu64 %%zmm16, 320(%[ptr])\n\t"
    "vmovdqu64 %%zmm16, 384(%[ptr])\n\t"
    "vmovdqu64 %%zmm16, 448(%[ptr])\n\t",
    "zmm16")
X86_DEFINE_MEMORY_KERNEL(
    x86_zmm_mask_load,
    "mov $0xaaaa, %%eax\n\t"
    "kmovw %%eax, %%k1\n\t",
    "vmovdqu32 0(%[ptr]), %%zmm0%{%%k1%}%{z%}\n\t"
    "vmovdqu32 64(%[ptr]), %%zmm1%{%%k1%}%{z%}\n\t"
    "vmovdqu32 128(%[ptr]), %%zmm2%{%%k1%}%{z%}\n\t"
    "vmovdqu32 192(%[ptr]), %%zmm3%{%%k1%}%{z%}\n\t"
    "vmovdqu32 256(%[ptr]), %%zmm4%{%%k1%}%{z%}\n\t"
    "vmovdqu32 320(%[ptr]), %%zmm5%{%%k1%}%{z%}\n\t"
    "vmovdqu32 384(%[ptr]), %%zmm6%{%%k1%}%{z%}\n\t"
    "vmovdqu32 448(%[ptr]), %%zmm7%{%%k1%}%{z%}\n\t",
    "rax", "k1", "zmm0", "zmm1", "zmm2", "zmm3",
    "zmm4", "zmm5", "zmm6", "zmm7")
X86_DEFINE_MEMORY_KERNEL(
    x86_zmm_mask_store,
    "mov $0xaaaa, %%eax\n\t"
    "kmovw %%eax, %%k1\n\t"
    "vpxord %%zmm16, %%zmm16, %%zmm16\n\t",
    "vmovdqu32 %%zmm16, 0(%[ptr])%{%%k1%}\n\t"
    "vmovdqu32 %%zmm16, 64(%[ptr])%{%%k1%}\n\t"
    "vmovdqu32 %%zmm16, 128(%[ptr])%{%%k1%}\n\t"
    "vmovdqu32 %%zmm16, 192(%[ptr])%{%%k1%}\n\t"
    "vmovdqu32 %%zmm16, 256(%[ptr])%{%%k1%}\n\t"
    "vmovdqu32 %%zmm16, 320(%[ptr])%{%%k1%}\n\t"
    "vmovdqu32 %%zmm16, 384(%[ptr])%{%%k1%}\n\t"
    "vmovdqu32 %%zmm16, 448(%[ptr])%{%%k1%}\n\t",
    "rax", "k1", "zmm16")

#define X86_CONV_ADVANCE(bytes)                                              \
  "add $" #bytes ", %[ptr]\n\t"                                             \
  "cmp %[end], %[ptr]\n\t"                                                  \
  "cmovae %[base], %[ptr]\n\t"
#define X86_DEFINE_ZMM_CONV_STREAM_CONTROL(name, stride)                     \
  X86_DEFINE_MEMORY_KERNEL(name, "", X86_CONV_ADVANCE(stride))
#define X86_DEFINE_ZMM_CONV_LOAD(name, op, o1, o2, o3, o4, o5, o6, o7, stride) \
  X86_DEFINE_MEMORY_KERNEL(                                                  \
      name, "",                                                             \
      op " 0(%[ptr]), %%zmm0\n\t"                                          \
      op " " #o1 "(%[ptr]), %%zmm1\n\t"                                    \
      op " " #o2 "(%[ptr]), %%zmm2\n\t"                                    \
      op " " #o3 "(%[ptr]), %%zmm3\n\t"                                    \
      op " " #o4 "(%[ptr]), %%zmm4\n\t"                                    \
      op " " #o5 "(%[ptr]), %%zmm5\n\t"                                    \
      op " " #o6 "(%[ptr]), %%zmm6\n\t"                                    \
      op " " #o7 "(%[ptr]), %%zmm7\n\t"                                    \
      X86_CONV_ADVANCE(stride),                                              \
      "zmm0", "zmm1", "zmm2", "zmm3", "zmm4", "zmm5", "zmm6", "zmm7")
#define X86_DEFINE_ZMM_CONV_STORE(name, op, o1, o2, o3, o4, o5, o6, o7, stride) \
  X86_DEFINE_MEMORY_KERNEL(                                                  \
      name,                                                                 \
      "movabs $0x80000100ffff007f, %%rax\n\t"                               \
      "vpbroadcastq %%rax, %%zmm16\n\t",                                    \
      op " %%zmm16, 0(%[ptr])\n\t"                                         \
      op " %%zmm16, " #o1 "(%[ptr])\n\t"                                   \
      op " %%zmm16, " #o2 "(%[ptr])\n\t"                                   \
      op " %%zmm16, " #o3 "(%[ptr])\n\t"                                   \
      op " %%zmm16, " #o4 "(%[ptr])\n\t"                                   \
      op " %%zmm16, " #o5 "(%[ptr])\n\t"                                   \
      op " %%zmm16, " #o6 "(%[ptr])\n\t"                                   \
      op " %%zmm16, " #o7 "(%[ptr])\n\t"                                   \
      X86_CONV_ADVANCE(stride), "rax", "zmm16")

X86_DEFINE_ZMM_CONV_STREAM_CONTROL(x86_zmm_conv_control_8b, 64)
X86_DEFINE_ZMM_CONV_STREAM_CONTROL(x86_zmm_conv_control_16b, 128)
X86_DEFINE_ZMM_CONV_STREAM_CONTROL(x86_zmm_conv_control_32b, 256)

#define X86_ZMM_CONV_LOAD_8(name, op)                                       \
  X86_DEFINE_ZMM_CONV_LOAD(name, op, 8, 16, 24, 32, 40, 48, 56, 64)
#define X86_ZMM_CONV_LOAD_16(name, op)                                      \
  X86_DEFINE_ZMM_CONV_LOAD(name, op, 16, 32, 48, 64, 80, 96, 112, 128)
#define X86_ZMM_CONV_LOAD_32(name, op)                                      \
  X86_DEFINE_ZMM_CONV_LOAD(name, op, 32, 64, 96, 128, 160, 192, 224, 256)
#define X86_ZMM_CONV_STORE_8(name, op)                                      \
  X86_DEFINE_ZMM_CONV_STORE(name, op, 8, 16, 24, 32, 40, 48, 56, 64)
#define X86_ZMM_CONV_STORE_16(name, op)                                     \
  X86_DEFINE_ZMM_CONV_STORE(name, op, 16, 32, 48, 64, 80, 96, 112, 128)
#define X86_ZMM_CONV_STORE_32(name, op)                                     \
  X86_DEFINE_ZMM_CONV_STORE(name, op, 32, 64, 96, 128, 160, 192, 224, 256)

X86_ZMM_CONV_LOAD_32(x86_zmm_vpmovsxbw_stream, "vpmovsxbw")
X86_ZMM_CONV_LOAD_32(x86_zmm_vpmovzxbw_stream, "vpmovzxbw")
X86_ZMM_CONV_LOAD_16(x86_zmm_vpmovsxbd_stream, "vpmovsxbd")
X86_ZMM_CONV_LOAD_16(x86_zmm_vpmovzxbd_stream, "vpmovzxbd")
X86_ZMM_CONV_LOAD_8(x86_zmm_vpmovsxbq_stream, "vpmovsxbq")
X86_ZMM_CONV_LOAD_8(x86_zmm_vpmovzxbq_stream, "vpmovzxbq")
X86_ZMM_CONV_LOAD_32(x86_zmm_vpmovsxwd_stream, "vpmovsxwd")
X86_ZMM_CONV_LOAD_32(x86_zmm_vpmovzxwd_stream, "vpmovzxwd")
X86_ZMM_CONV_LOAD_16(x86_zmm_vpmovsxwq_stream, "vpmovsxwq")
X86_ZMM_CONV_LOAD_16(x86_zmm_vpmovzxwq_stream, "vpmovzxwq")
X86_ZMM_CONV_LOAD_32(x86_zmm_vpmovsxdq_stream, "vpmovsxdq")
X86_ZMM_CONV_LOAD_32(x86_zmm_vpmovzxdq_stream, "vpmovzxdq")

#define X86_ZMM_DEFINE_STORE_FAMILY(prefix, wb, db, dw, qb, qw, qd)          \
  X86_ZMM_CONV_STORE_32(prefix##_wb_stream, wb)                              \
  X86_ZMM_CONV_STORE_16(prefix##_db_stream, db)                              \
  X86_ZMM_CONV_STORE_32(prefix##_dw_stream, dw)                              \
  X86_ZMM_CONV_STORE_8(prefix##_qb_stream, qb)                               \
  X86_ZMM_CONV_STORE_16(prefix##_qw_stream, qw)                              \
  X86_ZMM_CONV_STORE_32(prefix##_qd_stream, qd)
X86_ZMM_DEFINE_STORE_FAMILY(
    x86_zmm_vpmov, "vpmovwb", "vpmovdb", "vpmovdw",
    "vpmovqb", "vpmovqw", "vpmovqd")
X86_ZMM_DEFINE_STORE_FAMILY(
    x86_zmm_vpmovs, "vpmovswb", "vpmovsdb", "vpmovsdw",
    "vpmovsqb", "vpmovsqw", "vpmovsqd")
X86_ZMM_DEFINE_STORE_FAMILY(
    x86_zmm_vpmovus, "vpmovuswb", "vpmovusdb", "vpmovusdw",
    "vpmovusqb", "vpmovusqw", "vpmovusqd")

#define X86_DEFINE_ZMM_INDEX_CHASE(name, op, normalize, signed_anchor)       \
  extern "C" VECOPS_INST_NOINLINE_USED void name(                           \
      uint64_t loops, void* memory, uint64_t memory_bytes) {                 \
    if (loops == 0 || memory_bytes == 0) return;                             \
    auto* base = static_cast<uint8_t*>(memory);                              \
    auto* ptr = base;                                                        \
    auto* anchor = base + (signed_anchor);                                   \
    __asm__ volatile(                                                        \
        ".p2align 6\n\t"                                                    \
        "1:\n\t"                                                            \
        op " (%[ptr]), %%zmm0\n\t"                                         \
        normalize                                                           \
        "shl $6, %%rax\n\t"                                                 \
        "lea (%[anchor], %%rax), %[ptr]\n\t"                                \
        "dec %[loops]\n\t"                                                  \
        "jnz 1b\n\t"                                                        \
        : [loops] "+r"(loops), [ptr] "+&r"(ptr)                             \
        : [anchor] "r"(anchor)                                              \
        : "cc", "memory", "rax", "zmm0");                                  \
  }
#define X86_EXTRACT_S16 "vmovd %%xmm0, %%eax\n\tmovswq %%ax, %%rax\n\t"
#define X86_EXTRACT_U16 "vmovd %%xmm0, %%eax\n\tmovzwl %%ax, %%eax\n\t"
#define X86_EXTRACT_S32 "vmovd %%xmm0, %%eax\n\tmovslq %%eax, %%rax\n\t"
#define X86_EXTRACT_U32 "vmovd %%xmm0, %%eax\n\t"
#define X86_EXTRACT_64 "vmovq %%xmm0, %%rax\n\t"
#define X86_ZMM_INDEX_PAIR(symbol_s, op_s, symbol_u, op_u, extract_s, extract_u) \
  X86_DEFINE_ZMM_INDEX_CHASE(symbol_s, op_s, extract_s, 4096)                \
  X86_DEFINE_ZMM_INDEX_CHASE(symbol_u, op_u, extract_u, 0)
X86_ZMM_INDEX_PAIR(
    x86_zmm_vpmovsxbw_index, "vpmovsxbw", x86_zmm_vpmovzxbw_index,
    "vpmovzxbw", X86_EXTRACT_S16, X86_EXTRACT_U16)
X86_ZMM_INDEX_PAIR(
    x86_zmm_vpmovsxbd_index, "vpmovsxbd", x86_zmm_vpmovzxbd_index,
    "vpmovzxbd", X86_EXTRACT_S32, X86_EXTRACT_U32)
X86_ZMM_INDEX_PAIR(
    x86_zmm_vpmovsxbq_index, "vpmovsxbq", x86_zmm_vpmovzxbq_index,
    "vpmovzxbq", X86_EXTRACT_64, X86_EXTRACT_64)
X86_ZMM_INDEX_PAIR(
    x86_zmm_vpmovsxwd_index, "vpmovsxwd", x86_zmm_vpmovzxwd_index,
    "vpmovzxwd", X86_EXTRACT_S32, X86_EXTRACT_U32)
X86_ZMM_INDEX_PAIR(
    x86_zmm_vpmovsxwq_index, "vpmovsxwq", x86_zmm_vpmovzxwq_index,
    "vpmovzxwq", X86_EXTRACT_64, X86_EXTRACT_64)
X86_ZMM_INDEX_PAIR(
    x86_zmm_vpmovsxdq_index, "vpmovsxdq", x86_zmm_vpmovzxdq_index,
    "vpmovzxdq", X86_EXTRACT_64, X86_EXTRACT_64)

#define X86_DEFINE_SCALAR_INDEX_CONTROL(name, load_op, signed_anchor)        \
  extern "C" VECOPS_INST_NOINLINE_USED void name(                           \
      uint64_t loops, void* memory, uint64_t memory_bytes) {                 \
    if (loops == 0 || memory_bytes == 0) return;                             \
    auto* base = static_cast<uint8_t*>(memory);                              \
    auto* ptr = base;                                                        \
    auto* anchor = base + (signed_anchor);                                   \
    __asm__ volatile(                                                        \
        ".p2align 6\n\t"                                                    \
        "1:\n\t"                                                            \
        load_op                                                             \
        "shl $6, %%rax\n\t"                                                 \
        "lea (%[anchor], %%rax), %[ptr]\n\t"                                \
        "dec %[loops]\n\t"                                                  \
        "jnz 1b\n\t"                                                        \
        : [loops] "+r"(loops), [ptr] "+&r"(ptr)                             \
        : [anchor] "r"(anchor)                                              \
        : "cc", "memory", "rax");                                          \
  }
X86_DEFINE_SCALAR_INDEX_CONTROL(
    x86_zmm_index_control_s8, "movsbq (%[ptr]), %%rax\n\t", 4096)
X86_DEFINE_SCALAR_INDEX_CONTROL(
    x86_zmm_index_control_u8, "movzbl (%[ptr]), %%eax\n\t", 0)
X86_DEFINE_SCALAR_INDEX_CONTROL(
    x86_zmm_index_control_s16, "movswq (%[ptr]), %%rax\n\t", 4096)
X86_DEFINE_SCALAR_INDEX_CONTROL(
    x86_zmm_index_control_u16, "movzwl (%[ptr]), %%eax\n\t", 0)
X86_DEFINE_SCALAR_INDEX_CONTROL(
    x86_zmm_index_control_s32, "movslq (%[ptr]), %%rax\n\t", 4096)
X86_DEFINE_SCALAR_INDEX_CONTROL(
    x86_zmm_index_control_u32, "movl (%[ptr]), %%eax\n\t", 0)

#define X86_DEFINE_ZMM_FORWARD(name, store_op, load_op)                      \
  X86_DEFINE_MEMORY_KERNEL(                                                  \
      name,                                                                 \
      "movabs $0x80000100ffff007f, %%rax\n\t"                               \
      "vpbroadcastq %%rax, %%zmm0\n\t",                                    \
      store_op " %%zmm0, (%[ptr])\n\t"                                     \
      load_op " (%[ptr]), %%zmm0\n\t", "rax", "zmm0")
#define X86_ZMM_FORWARD_FAMILY(prefix, wb, db, dw, qb, qw, qd,               \
                               l8w, l8d, l16d, l8q, l16q, l32q)              \
  X86_DEFINE_ZMM_FORWARD(prefix##_wb_forward, wb, l8w)                       \
  X86_DEFINE_ZMM_FORWARD(prefix##_db_forward, db, l8d)                       \
  X86_DEFINE_ZMM_FORWARD(prefix##_dw_forward, dw, l16d)                      \
  X86_DEFINE_ZMM_FORWARD(prefix##_qb_forward, qb, l8q)                       \
  X86_DEFINE_ZMM_FORWARD(prefix##_qw_forward, qw, l16q)                      \
  X86_DEFINE_ZMM_FORWARD(prefix##_qd_forward, qd, l32q)
X86_ZMM_FORWARD_FAMILY(
    x86_zmm_vpmov, "vpmovwb", "vpmovdb", "vpmovdw", "vpmovqb", "vpmovqw",
    "vpmovqd", "vpmovzxbw", "vpmovzxbd", "vpmovzxwd", "vpmovzxbq",
    "vpmovzxwq", "vpmovzxdq")
X86_ZMM_FORWARD_FAMILY(
    x86_zmm_vpmovs, "vpmovswb", "vpmovsdb", "vpmovsdw", "vpmovsqb",
    "vpmovsqw", "vpmovsqd", "vpmovsxbw", "vpmovsxbd", "vpmovsxwd",
    "vpmovsxbq", "vpmovsxwq", "vpmovsxdq")
X86_ZMM_FORWARD_FAMILY(
    x86_zmm_vpmovus, "vpmovuswb", "vpmovusdb", "vpmovusdw", "vpmovusqb",
    "vpmovusqw", "vpmovusqd", "vpmovzxbw", "vpmovzxbd", "vpmovzxwd",
    "vpmovzxbq", "vpmovzxwq", "vpmovzxdq")

const InstructionCase kCases[] = {
    {"empty_loop", "x86", "Loop-control baseline", MeasureMode::Control,
     x86_control_empty_loop, 32, 0, 0, 512},
    {"nop_32", "x86", "Thirty-two one-byte NOPs", MeasureMode::Control,
     x86_control_nop_32, 32, 1, 0, 512},
#define X86_LAT_THR_CASES(label, isa, symbol)                                \
    {label, isa, "Single true-dependency chain", MeasureMode::Latency,       \
     symbol##_latency, 32, 1, 1, 512},                                      \
    {label, isa, "Eight independent destination chains",                    \
     MeasureMode::Throughput, symbol##_throughput, 32, 1, 8, 512}
#define X86_CATEGORY_CASES(label, isa, symbol, category_value)               \
    {label, isa, "Single true-dependency chain", MeasureMode::Latency,       \
     symbol##_latency, 32, 1, 1, 512, category_value},                      \
    {label, isa, "Eight independent destination chains",                    \
     MeasureMode::Throughput, symbol##_throughput, 32, 1, 8, 512, category_value}
    X86_LAT_THR_CASES("vaddps_zmm_zmm_zmm", "AVX512F", x86_vaddps_zmm),
    X86_LAT_THR_CASES("vmulps_zmm_zmm_zmm", "AVX512F", x86_vmulps_zmm),
    X86_LAT_THR_CASES("vfmadd231ps_zmm_zmm_zmm", "AVX512F+FMA", x86_vfmadd231ps_zmm),
    X86_LAT_THR_CASES("vpaddd_zmm_zmm_zmm", "AVX512F", x86_vpaddd_zmm),
    X86_LAT_THR_CASES("vpmulld_zmm_zmm_zmm", "AVX512F", x86_vpmulld_zmm),
    X86_CATEGORY_CASES("vpandd_zmm_zmm_zmm", "AVX512F", x86_vpandd_zmm,
                       InstructionCategory::Bitwise),
    X86_CATEGORY_CASES("vpord_zmm_zmm_zmm", "AVX512F", x86_vpord_zmm,
                       InstructionCategory::Bitwise),
    X86_CATEGORY_CASES("vpxord_zmm_zmm_zmm", "AVX512F", x86_vpxord_zmm,
                       InstructionCategory::Bitwise),
    X86_CATEGORY_CASES("vpandnd_zmm_zmm_zmm", "AVX512F", x86_vpandnd_zmm,
                       InstructionCategory::Bitwise),
    X86_CATEGORY_CASES("vpternlogd_zmm_not", "AVX512F",
                       x86_vpternlogd_not_zmm, InstructionCategory::Bitwise),
    X86_LAT_THR_CASES("vpermd_zmm_zmm_zmm", "AVX512F", x86_vpermd_zmm),
    X86_LAT_THR_CASES("vshufi32x4_zmm_zmm_imm0", "AVX512F", x86_vshufi32x4_zmm),
#define X86_SHIFT_CASE(label, symbol)                                        \
    X86_CATEGORY_CASES(label, "AVX512F", symbol, InstructionCategory::Shift)
    X86_SHIFT_CASE("vpsllw_zmm_zmm_imm7", x86_vpsllw_zmm_imm7),
    X86_SHIFT_CASE("vpslld_zmm_zmm_imm7", x86_vpslld_zmm_imm7),
    X86_SHIFT_CASE("vpsllq_zmm_zmm_imm7", x86_vpsllq_zmm_imm7),
    X86_SHIFT_CASE("vpsrlw_zmm_zmm_imm7", x86_vpsrlw_zmm_imm7),
    X86_SHIFT_CASE("vpsrld_zmm_zmm_imm7", x86_vpsrld_zmm_imm7),
    X86_SHIFT_CASE("vpsrlq_zmm_zmm_imm7", x86_vpsrlq_zmm_imm7),
    X86_SHIFT_CASE("vpsraw_zmm_zmm_imm7", x86_vpsraw_zmm_imm7),
    X86_SHIFT_CASE("vpsrad_zmm_zmm_imm7", x86_vpsrad_zmm_imm7),
    X86_SHIFT_CASE("vpsraq_zmm_zmm_imm7", x86_vpsraq_zmm_imm7),
    X86_SHIFT_CASE("vpsllw_zmm_zmm_xmm", x86_vpsllw_zmm_xmm),
    X86_SHIFT_CASE("vpslld_zmm_zmm_xmm", x86_vpslld_zmm_xmm),
    X86_SHIFT_CASE("vpsllq_zmm_zmm_xmm", x86_vpsllq_zmm_xmm),
    X86_SHIFT_CASE("vpsrlw_zmm_zmm_xmm", x86_vpsrlw_zmm_xmm),
    X86_SHIFT_CASE("vpsrld_zmm_zmm_xmm", x86_vpsrld_zmm_xmm),
    X86_SHIFT_CASE("vpsrlq_zmm_zmm_xmm", x86_vpsrlq_zmm_xmm),
    X86_SHIFT_CASE("vpsraw_zmm_zmm_xmm", x86_vpsraw_zmm_xmm),
    X86_SHIFT_CASE("vpsrad_zmm_zmm_xmm", x86_vpsrad_zmm_xmm),
    X86_SHIFT_CASE("vpsraq_zmm_zmm_xmm", x86_vpsraq_zmm_xmm),
    X86_SHIFT_CASE("vpsllvw_zmm_zmm_zmm", x86_vpsllvw_zmm),
    X86_SHIFT_CASE("vpsllvd_zmm_zmm_zmm", x86_vpsllvd_zmm),
    X86_SHIFT_CASE("vpsllvq_zmm_zmm_zmm", x86_vpsllvq_zmm),
    X86_SHIFT_CASE("vpsrlvw_zmm_zmm_zmm", x86_vpsrlvw_zmm),
    X86_SHIFT_CASE("vpsrlvd_zmm_zmm_zmm", x86_vpsrlvd_zmm),
    X86_SHIFT_CASE("vpsrlvq_zmm_zmm_zmm", x86_vpsrlvq_zmm),
    X86_SHIFT_CASE("vpsravw_zmm_zmm_zmm", x86_vpsravw_zmm),
    X86_SHIFT_CASE("vpsravd_zmm_zmm_zmm", x86_vpsravd_zmm),
    X86_SHIFT_CASE("vpsravq_zmm_zmm_zmm", x86_vpsravq_zmm),
    X86_CATEGORY_CASES("vpblendmd_zmm_k_zmm_zmm", "AVX512F",
                       x86_vpblendmd_zmm, InstructionCategory::Blend),
    X86_CATEGORY_CASES("vpblendmq_zmm_k_zmm_zmm", "AVX512F",
                       x86_vpblendmq_zmm, InstructionCategory::Blend),
    X86_CATEGORY_CASES("vpblendmw_zmm_k_zmm_zmm", "AVX512BW",
                       x86_vpblendmw_zmm, InstructionCategory::Blend),
    X86_CATEGORY_CASES("vpblendmb_zmm_k_zmm_zmm", "AVX512BW",
                       x86_vpblendmb_zmm, InstructionCategory::Blend),
    {"vcvtph2ps_ymm_to_zmm", "AVX512F+F16C", "Widen packed fp16 to fp32",
     MeasureMode::Throughput, x86_vcvtph2ps_ymm_to_zmm_throughput, 32, 1, 8, 512},
    {"vcvtps2ph_zmm_to_ymm_imm0", "AVX512F+F16C", "Narrow packed fp32 to fp16",
     MeasureMode::Throughput, x86_vcvtps2ph_zmm_to_ymm_throughput, 32, 1, 8, 512},
    {"vcvtps2pd_ymm_to_zmm", "AVX512F", "Widen packed fp32 to fp64",
     MeasureMode::Throughput, x86_vcvtps2pd_ymm_to_zmm_throughput, 32, 1, 8, 512},
    {"vcvtpd2ps_zmm_to_ymm", "AVX512F", "Narrow packed fp64 to fp32",
     MeasureMode::Throughput, x86_vcvtpd2ps_zmm_to_ymm_throughput, 32, 1, 8, 512},
    {"vpmovsxwd_ymm_to_zmm", "AVX512F", "Sign-extend packed i16 to i32",
     MeasureMode::Throughput, x86_vpmovsxwd_ymm_to_zmm_throughput, 32, 1, 8, 512},
    {"vpmovdw_zmm_to_ymm", "AVX512F", "Truncate packed i32 to i16",
     MeasureMode::Throughput, x86_vpmovdw_zmm_to_ymm_throughput, 32, 1, 8, 512},
    {"vcvtph2ps_ymm_to_zmm+vcvtps2ph_zmm_to_ymm", "AVX512F+F16C",
     "Closed fp16/fp32 dependency sequence", MeasureMode::Latency,
     x86_vcvtph2ps_vcvtps2ph_latency, 16, 2, 1, 512},
    {"vcvtps2pd_ymm_to_zmm+vcvtpd2ps_zmm_to_ymm", "AVX512F",
     "Closed fp32/fp64 dependency sequence", MeasureMode::Latency,
     x86_vcvtps2pd_vcvtpd2ps_latency, 16, 2, 1, 512},
    {"vpmovsxwd_ymm_to_zmm+vpmovdw_zmm_to_ymm", "AVX512F",
     "Closed sign-extend/truncate dependency sequence", MeasureMode::Latency,
     x86_vpmovsxwd_vpmovdw_latency, 16, 2, 1, 512},
#define X86_MEMORY_CASE(label, description, mode_value, memory_symbol, seq, inst, chains, \
                        pattern_value, level_value, bytes_value, baseline_value)          \
    {label, "AVX512F", description, mode_value, nullptr, seq, inst, chains, 512,           \
     InstructionCategory::Memory, memory_symbol, pattern_value, level_value,               \
     bytes_value, baseline_value}
#define X86_MEMORY_LEVEL(level_label, level_value)                             \
    X86_MEMORY_CASE("stream_control_" level_label, "Address-generation baseline",          \
                    MeasureMode::Control, x86_zmm_stream_control, 8, 0, 0,                  \
                    MemoryPattern::Stream, level_value, 0, "empty_loop"),                   \
    X86_MEMORY_CASE("vmovdqa64_stream_load_" level_label, "Sequential aligned vector load", \
                    MeasureMode::Throughput, x86_zmm_stream_load, 8, 1, 8,                 \
                    MemoryPattern::Stream, level_value, 64, "stream_control_" level_label), \
    X86_MEMORY_CASE("vmovdqa64_stream_store_" level_label, "Sequential aligned vector store",\
                    MeasureMode::Throughput, x86_zmm_stream_store, 8, 1, 8,                \
                    MemoryPattern::Stream, level_value, 64, "stream_control_" level_label),\
    X86_MEMORY_CASE("vmovdqa64+vmovq_pointer_chase_" level_label,                           \
                    "Dependent vector-load cache-latency sequence", MeasureMode::Latency,  \
                    x86_zmm_pointer_chase, 1, 2, 1, MemoryPattern::PointerChase,            \
                    level_value, 64, "empty_loop")
    X86_MEMORY_LEVEL("l1", WorkingSetLevel::L1),
    X86_MEMORY_LEVEL("l2", WorkingSetLevel::L2),
    X86_MEMORY_LEVEL("l3", WorkingSetLevel::L3),
    X86_MEMORY_LEVEL("beyond_last_reported_cache", WorkingSetLevel::BeyondLastCache),
    X86_MEMORY_CASE("vmovdqa64_store+load_forward", "L1 store-to-load forwarding round trip",
                    MeasureMode::Latency, x86_zmm_store_forward, 1, 2, 1,
                    MemoryPattern::StoreForward, WorkingSetLevel::L1, 64, "empty_loop"),
    X86_MEMORY_CASE("vmovdqu64_crossline_load", "Cross-cache-line unaligned load",
                    MeasureMode::Throughput, x86_zmm_crossline_load, 8, 1, 8,
                    MemoryPattern::Hot, WorkingSetLevel::L1, 64, "empty_loop"),
    X86_MEMORY_CASE("vmovdqu64_crossline_store", "Cross-cache-line unaligned store",
                    MeasureMode::Throughput, x86_zmm_crossline_store, 8, 1, 8,
                    MemoryPattern::Hot, WorkingSetLevel::L1, 64, "empty_loop"),
    X86_MEMORY_CASE("vmovdqu32_mask50_load", "Half-mask contiguous load",
                    MeasureMode::Throughput, x86_zmm_mask_load, 8, 1, 8,
                    MemoryPattern::Hot, WorkingSetLevel::L1, 32, "empty_loop"),
    X86_MEMORY_CASE("vmovdqu32_mask50_store", "Half-mask contiguous store",
                    MeasureMode::Throughput, x86_zmm_mask_store, 8, 1, 8,
                    MemoryPattern::Hot, WorkingSetLevel::L1, 32, "empty_loop"),
#define X86_ZMM_CONV_CASE(label, isa_value, description, mode_value, symbol, seq, inst, \
                          chains, pattern, level, bytes, baseline, kind, src, dst, elems, cap) \
    {label, isa_value, description, mode_value, nullptr, seq, inst, chains, 512,         \
     InstructionCategory::Memory, symbol, pattern, level, bytes, baseline, kind, src,    \
     dst, elems, cap}
#define X86_ZMM_CONV_CONTROLS(level_label, level_value)                       \
    X86_ZMM_CONV_CASE("conv_stream_control_8b_" level_label, "AVX512F",       \
                      "8-byte conversion-stream address baseline", MeasureMode::Control,\
                      x86_zmm_conv_control_8b, 8, 0, 0, MemoryPattern::Stream, level_value,\
                      0, "empty_loop", MemoryConversionKind::None, 0, 0, 0, 0),           \
    X86_ZMM_CONV_CASE("conv_stream_control_16b_" level_label, "AVX512F",      \
                      "16-byte conversion-stream address baseline", MeasureMode::Control,\
                      x86_zmm_conv_control_16b, 8, 0, 0, MemoryPattern::Stream, level_value,\
                      0, "empty_loop", MemoryConversionKind::None, 0, 0, 0, 0),           \
    X86_ZMM_CONV_CASE("conv_stream_control_32b_" level_label, "AVX512F",      \
                      "32-byte conversion-stream address baseline", MeasureMode::Control,\
                      x86_zmm_conv_control_32b, 8, 0, 0, MemoryPattern::Stream, level_value,\
                      0, "empty_loop", MemoryConversionKind::None, 0, 0, 0, 0)
    X86_ZMM_CONV_CONTROLS("l1", WorkingSetLevel::L1),
    X86_ZMM_CONV_CONTROLS("l2", WorkingSetLevel::L2),
    X86_ZMM_CONV_CONTROLS("l3", WorkingSetLevel::L3),
    X86_ZMM_CONV_CONTROLS(
        "beyond_last_reported_cache", WorkingSetLevel::BeyondLastCache),
#define X86_ZMM_CONV_STREAM_LEVEL(label, symbol, kind, src, dst, bytes, elems, bytes_tag, \
                                  level_label, level_value)                              \
    X86_ZMM_CONV_CASE(label "_stream_" level_label, "AVX512F",                         \
                      "Sequential integer converting memory instruction",              \
                      MeasureMode::Throughput, symbol, 8, 1, 8, MemoryPattern::Stream,   \
                      level_value, bytes, "conv_stream_control_" bytes_tag "_" level_label,\
                      kind, src, dst, elems, 0)
#define X86_ZMM_CONV_STREAM_LEVELS(label, symbol, kind, src, dst, bytes, elems, bytes_tag) \
    X86_ZMM_CONV_STREAM_LEVEL(label, symbol, kind, src, dst, bytes, elems, bytes_tag,     \
                              "l1", WorkingSetLevel::L1),                                 \
    X86_ZMM_CONV_STREAM_LEVEL(label, symbol, kind, src, dst, bytes, elems, bytes_tag,     \
                              "l2", WorkingSetLevel::L2),                                 \
    X86_ZMM_CONV_STREAM_LEVEL(label, symbol, kind, src, dst, bytes, elems, bytes_tag,     \
                              "l3", WorkingSetLevel::L3),                                 \
    X86_ZMM_CONV_STREAM_LEVEL(label, symbol, kind, src, dst, bytes, elems, bytes_tag,     \
                              "beyond_last_reported_cache",                              \
                              WorkingSetLevel::BeyondLastCache)
#define X86_ZMM_EXTEND_LEVELS(label_s, label_u, symbol_s, symbol_u, src, dst, bytes, elems, bytes_tag) \
    X86_ZMM_CONV_STREAM_LEVELS(label_s, symbol_s, MemoryConversionKind::SignExtendLoad,    \
                               src, dst, bytes, elems, bytes_tag),                          \
    X86_ZMM_CONV_STREAM_LEVELS(label_u, symbol_u, MemoryConversionKind::ZeroExtendLoad,    \
                               src, dst, bytes, elems, bytes_tag)
    X86_ZMM_EXTEND_LEVELS(
        "vpmovsxbw", "vpmovzxbw", x86_zmm_vpmovsxbw_stream,
        x86_zmm_vpmovzxbw_stream,
        8, 16, 32, 32, "32b"),
    X86_ZMM_EXTEND_LEVELS(
        "vpmovsxbd", "vpmovzxbd", x86_zmm_vpmovsxbd_stream,
        x86_zmm_vpmovzxbd_stream,
        8, 32, 16, 16, "16b"),
    X86_ZMM_EXTEND_LEVELS(
        "vpmovsxbq", "vpmovzxbq", x86_zmm_vpmovsxbq_stream,
        x86_zmm_vpmovzxbq_stream,
        8, 64, 8, 8, "8b"),
    X86_ZMM_EXTEND_LEVELS(
        "vpmovsxwd", "vpmovzxwd", x86_zmm_vpmovsxwd_stream,
        x86_zmm_vpmovzxwd_stream,
        16, 32, 32, 16, "32b"),
    X86_ZMM_EXTEND_LEVELS(
        "vpmovsxwq", "vpmovzxwq", x86_zmm_vpmovsxwq_stream,
        x86_zmm_vpmovzxwq_stream,
        16, 64, 16, 8, "16b"),
    X86_ZMM_EXTEND_LEVELS(
        "vpmovsxdq", "vpmovzxdq", x86_zmm_vpmovsxdq_stream,
        x86_zmm_vpmovzxdq_stream,
        32, 64, 32, 8, "32b"),
#define X86_ZMM_STORE_LEVELS(label, symbol, kind, src, dst, bytes, elems, bytes_tag) \
    X86_ZMM_CONV_STREAM_LEVELS(label, symbol, kind, src, dst, bytes, elems, bytes_tag)
#define X86_ZMM_STORE_FAMILY_LEVELS(prefix, kind)                             \
    X86_ZMM_STORE_LEVELS(#prefix "wb", x86_zmm_##prefix##_wb_stream, kind, 16, 8, 32, 32, "32b"),\
    X86_ZMM_STORE_LEVELS(#prefix "db", x86_zmm_##prefix##_db_stream, kind, 32, 8, 16, 16, "16b"),\
    X86_ZMM_STORE_LEVELS(#prefix "dw", x86_zmm_##prefix##_dw_stream, kind, 32, 16, 32, 16, "32b"),\
    X86_ZMM_STORE_LEVELS(#prefix "qb", x86_zmm_##prefix##_qb_stream, kind, 64, 8, 8, 8, "8b"),\
    X86_ZMM_STORE_LEVELS(#prefix "qw", x86_zmm_##prefix##_qw_stream, kind, 64, 16, 16, 8, "16b"),\
    X86_ZMM_STORE_LEVELS(#prefix "qd", x86_zmm_##prefix##_qd_stream, kind, 64, 32, 32, 8, "32b")
    X86_ZMM_STORE_FAMILY_LEVELS(vpmov, MemoryConversionKind::TruncateStore),
    X86_ZMM_STORE_FAMILY_LEVELS(
        vpmovs, MemoryConversionKind::SignedSaturatingStore),
    X86_ZMM_STORE_FAMILY_LEVELS(
        vpmovus, MemoryConversionKind::UnsignedSaturatingStore),
#define X86_ZMM_INDEX_CONTROL(label, symbol, kind, src)                       \
    X86_ZMM_CONV_CASE(label, "AVX512F", "Matched scalar extending-load index-chase control",\
                      MeasureMode::Control, symbol, 1, 0, 1,                  \
                      MemoryPattern::PointerChase, WorkingSetLevel::L1, 0,    \
                      "empty_loop", kind, src, src, 0, 8192)
    X86_ZMM_INDEX_CONTROL(
        "conv_index_control_s8", x86_zmm_index_control_s8,
        MemoryConversionKind::SignExtendLoad, 8),
    X86_ZMM_INDEX_CONTROL(
        "conv_index_control_u8", x86_zmm_index_control_u8,
        MemoryConversionKind::ZeroExtendLoad, 8),
    X86_ZMM_INDEX_CONTROL(
        "conv_index_control_s16", x86_zmm_index_control_s16,
        MemoryConversionKind::SignExtendLoad, 16),
    X86_ZMM_INDEX_CONTROL(
        "conv_index_control_u16", x86_zmm_index_control_u16,
        MemoryConversionKind::ZeroExtendLoad, 16),
    X86_ZMM_INDEX_CONTROL(
        "conv_index_control_s32", x86_zmm_index_control_s32,
        MemoryConversionKind::SignExtendLoad, 32),
    X86_ZMM_INDEX_CONTROL(
        "conv_index_control_u32", x86_zmm_index_control_u32,
        MemoryConversionKind::ZeroExtendLoad, 32),
#define X86_ZMM_EXTEND_LAT(label, symbol, kind, src, dst, elems, baseline)    \
    X86_ZMM_CONV_CASE(label "_index_chase_l1", "AVX512F",                   \
                      "L1 dependent extending-load index chase", MeasureMode::Latency,\
                      symbol, 1, 4, 1, MemoryPattern::PointerChase,           \
                      WorkingSetLevel::L1, src / 8 * elems, baseline, kind,   \
                      src, dst, elems, 8192)
    X86_ZMM_EXTEND_LAT(
        "vpmovsxbw", x86_zmm_vpmovsxbw_index, MemoryConversionKind::SignExtendLoad,
        8, 16, 32, "conv_index_control_s8"),
    X86_ZMM_EXTEND_LAT(
        "vpmovzxbw", x86_zmm_vpmovzxbw_index, MemoryConversionKind::ZeroExtendLoad,
        8, 16, 32, "conv_index_control_u8"),
    X86_ZMM_EXTEND_LAT(
        "vpmovsxbd", x86_zmm_vpmovsxbd_index, MemoryConversionKind::SignExtendLoad,
        8, 32, 16, "conv_index_control_s8"),
    X86_ZMM_EXTEND_LAT(
        "vpmovzxbd", x86_zmm_vpmovzxbd_index, MemoryConversionKind::ZeroExtendLoad,
        8, 32, 16, "conv_index_control_u8"),
    X86_ZMM_EXTEND_LAT(
        "vpmovsxbq", x86_zmm_vpmovsxbq_index, MemoryConversionKind::SignExtendLoad,
        8, 64, 8, "conv_index_control_s8"),
    X86_ZMM_EXTEND_LAT(
        "vpmovzxbq", x86_zmm_vpmovzxbq_index, MemoryConversionKind::ZeroExtendLoad,
        8, 64, 8, "conv_index_control_u8"),
    X86_ZMM_EXTEND_LAT(
        "vpmovsxwd", x86_zmm_vpmovsxwd_index, MemoryConversionKind::SignExtendLoad,
        16, 32, 16, "conv_index_control_s16"),
    X86_ZMM_EXTEND_LAT(
        "vpmovzxwd", x86_zmm_vpmovzxwd_index, MemoryConversionKind::ZeroExtendLoad,
        16, 32, 16, "conv_index_control_u16"),
    X86_ZMM_EXTEND_LAT(
        "vpmovsxwq", x86_zmm_vpmovsxwq_index, MemoryConversionKind::SignExtendLoad,
        16, 64, 8, "conv_index_control_s16"),
    X86_ZMM_EXTEND_LAT(
        "vpmovzxwq", x86_zmm_vpmovzxwq_index, MemoryConversionKind::ZeroExtendLoad,
        16, 64, 8, "conv_index_control_u16"),
    X86_ZMM_EXTEND_LAT(
        "vpmovsxdq", x86_zmm_vpmovsxdq_index, MemoryConversionKind::SignExtendLoad,
        32, 64, 8, "conv_index_control_s32"),
    X86_ZMM_EXTEND_LAT(
        "vpmovzxdq", x86_zmm_vpmovzxdq_index, MemoryConversionKind::ZeroExtendLoad,
        32, 64, 8, "conv_index_control_u32"),
#define X86_ZMM_STORE_LAT(label, symbol, kind, src, dst, bytes, elems)        \
    X86_ZMM_CONV_CASE(label "_store_forward_l1", "AVX512F",                 \
                      "L1 dependent narrow-store plus extending-load forwarding round trip",\
                      MeasureMode::Latency, symbol, 1, 2, 1,                  \
                      MemoryPattern::StoreForward, WorkingSetLevel::L1, bytes,\
                      "empty_loop", kind, src, dst, elems, 0)
#define X86_ZMM_STORE_FAMILY_LAT(prefix, kind)                               \
    X86_ZMM_STORE_LAT(#prefix "wb", x86_zmm_##prefix##_wb_forward, kind, 16, 8, 32, 32),\
    X86_ZMM_STORE_LAT(#prefix "db", x86_zmm_##prefix##_db_forward, kind, 32, 8, 16, 16),\
    X86_ZMM_STORE_LAT(#prefix "dw", x86_zmm_##prefix##_dw_forward, kind, 32, 16, 32, 16),\
    X86_ZMM_STORE_LAT(#prefix "qb", x86_zmm_##prefix##_qb_forward, kind, 64, 8, 8, 8),\
    X86_ZMM_STORE_LAT(#prefix "qw", x86_zmm_##prefix##_qw_forward, kind, 64, 16, 16, 8),\
    X86_ZMM_STORE_LAT(#prefix "qd", x86_zmm_##prefix##_qd_forward, kind, 64, 32, 32, 8)
    X86_ZMM_STORE_FAMILY_LAT(vpmov, MemoryConversionKind::TruncateStore),
    X86_ZMM_STORE_FAMILY_LAT(vpmovs, MemoryConversionKind::SignedSaturatingStore),
    X86_ZMM_STORE_FAMILY_LAT(
        vpmovus, MemoryConversionKind::UnsignedSaturatingStore),
};

#elif defined(__AVX2__)

#define X86_YMM_SETUP                                                        \
  "vpxor %%ymm0, %%ymm0, %%ymm0\n\t"                                           \
  "vpxor %%ymm1, %%ymm1, %%ymm1\n\t"                                           \
  "vpxor %%ymm2, %%ymm2, %%ymm2\n\t"                                           \
  "vpxor %%ymm3, %%ymm3, %%ymm3\n\t"                                           \
  "vpxor %%ymm4, %%ymm4, %%ymm4\n\t"                                           \
  "vpxor %%ymm5, %%ymm5, %%ymm5\n\t"                                           \
  "vpxor %%ymm6, %%ymm6, %%ymm6\n\t"                                           \
  "vpxor %%ymm7, %%ymm7, %%ymm7\n\t"                                           \
  "vpcmpeqd %%ymm10, %%ymm10, %%ymm10\n\t"                                     \
  "vpxor %%ymm14, %%ymm14, %%ymm14\n\t"                                         \
  "vpxor %%ymm15, %%ymm15, %%ymm15\n\t"                                         \
  "mov $7, %%eax\n\t"                                                       \
  "vmovd %%eax, %%xmm11\n\t"                                               \
  "vpbroadcastw %%xmm11, %%ymm11\n\t"                                      \
  "vmovd %%eax, %%xmm12\n\t"                                               \
  "vpbroadcastd %%xmm12, %%ymm12\n\t"                                      \
  "vmovd %%eax, %%xmm13\n\t"                                               \
  "vpbroadcastq %%xmm13, %%ymm13\n\t"
#define X86_YMM_CLOBBERS                                                     \
  "ymm0", "ymm1", "ymm2", "ymm3", "ymm4", "ymm5", "ymm6", "ymm7", \
      "ymm10", "ymm11", "ymm12", "ymm13", "ymm14", "ymm15", "rax"
#define X86_YMM_BINARY_GROUP(op)                                             \
  op " %%ymm14, %%ymm0, %%ymm0\n\t"                                           \
  op " %%ymm14, %%ymm1, %%ymm1\n\t"                                           \
  op " %%ymm14, %%ymm2, %%ymm2\n\t"                                           \
  op " %%ymm14, %%ymm3, %%ymm3\n\t"                                           \
  op " %%ymm14, %%ymm4, %%ymm4\n\t"                                           \
  op " %%ymm14, %%ymm5, %%ymm5\n\t"                                           \
  op " %%ymm14, %%ymm6, %%ymm6\n\t"                                           \
  op " %%ymm14, %%ymm7, %%ymm7\n\t"
#define DEFINE_YMM_LAT_THR(name, op)                                         \
  X86_DEFINE_KERNEL(name##_latency, X86_YMM_SETUP,                           \
      VECOPS_INST_REPEAT_32(op " %%ymm14, %%ymm0, %%ymm0\n\t"),               \
      X86_YMM_CLOBBERS)                                                      \
  X86_DEFINE_KERNEL(name##_throughput, X86_YMM_SETUP,                        \
      VECOPS_INST_REPEAT_4(X86_YMM_BINARY_GROUP(op)), X86_YMM_CLOBBERS)

#define X86_YMM_NOT_GROUP                                                    \
  "vpxor %%ymm10, %%ymm0, %%ymm0\n\t"                                        \
  "vpxor %%ymm10, %%ymm1, %%ymm1\n\t"                                        \
  "vpxor %%ymm10, %%ymm2, %%ymm2\n\t"                                        \
  "vpxor %%ymm10, %%ymm3, %%ymm3\n\t"                                        \
  "vpxor %%ymm10, %%ymm4, %%ymm4\n\t"                                        \
  "vpxor %%ymm10, %%ymm5, %%ymm5\n\t"                                        \
  "vpxor %%ymm10, %%ymm6, %%ymm6\n\t"                                        \
  "vpxor %%ymm10, %%ymm7, %%ymm7\n\t"

#define DEFINE_YMM_NOT(name)                                                 \
  X86_DEFINE_KERNEL(name##_latency, X86_YMM_SETUP,                           \
      VECOPS_INST_REPEAT_32("vpxor %%ymm10, %%ymm0, %%ymm0\n\t"),            \
      X86_YMM_CLOBBERS)                                                      \
  X86_DEFINE_KERNEL(name##_throughput, X86_YMM_SETUP,                        \
      VECOPS_INST_REPEAT_4(X86_YMM_NOT_GROUP), X86_YMM_CLOBBERS)

#define X86_YMM_IMMEDIATE_GROUP(op)                                          \
  op " $7, %%ymm0, %%ymm0\n\t"                                             \
  op " $7, %%ymm1, %%ymm1\n\t"                                             \
  op " $7, %%ymm2, %%ymm2\n\t"                                             \
  op " $7, %%ymm3, %%ymm3\n\t"                                             \
  op " $7, %%ymm4, %%ymm4\n\t"                                             \
  op " $7, %%ymm5, %%ymm5\n\t"                                             \
  op " $7, %%ymm6, %%ymm6\n\t"                                             \
  op " $7, %%ymm7, %%ymm7\n\t"
#define X86_YMM_COUNT_GROUP(op, count)                                       \
  op " %%" count ", %%ymm0, %%ymm0\n\t"                                    \
  op " %%" count ", %%ymm1, %%ymm1\n\t"                                    \
  op " %%" count ", %%ymm2, %%ymm2\n\t"                                    \
  op " %%" count ", %%ymm3, %%ymm3\n\t"                                    \
  op " %%" count ", %%ymm4, %%ymm4\n\t"                                    \
  op " %%" count ", %%ymm5, %%ymm5\n\t"                                    \
  op " %%" count ", %%ymm6, %%ymm6\n\t"                                    \
  op " %%" count ", %%ymm7, %%ymm7\n\t"
#define DEFINE_YMM_IMMEDIATE_SHIFT(name, op)                                 \
  X86_DEFINE_KERNEL(name##_latency, X86_YMM_SETUP,                          \
      VECOPS_INST_REPEAT_32(op " $7, %%ymm0, %%ymm0\n\t"), X86_YMM_CLOBBERS)\
  X86_DEFINE_KERNEL(name##_throughput, X86_YMM_SETUP,                       \
      VECOPS_INST_REPEAT_4(X86_YMM_IMMEDIATE_GROUP(op)), X86_YMM_CLOBBERS)
#define DEFINE_YMM_COUNT_SHIFT(name, op, count)                              \
  X86_DEFINE_KERNEL(name##_latency, X86_YMM_SETUP,                          \
      VECOPS_INST_REPEAT_32(op " %%" count ", %%ymm0, %%ymm0\n\t"),         \
      X86_YMM_CLOBBERS)                                                      \
  X86_DEFINE_KERNEL(name##_throughput, X86_YMM_SETUP,                       \
      VECOPS_INST_REPEAT_4(X86_YMM_COUNT_GROUP(op, count)), X86_YMM_CLOBBERS)

DEFINE_YMM_LAT_THR(x86_vaddps_ymm, "vaddps")
DEFINE_YMM_LAT_THR(x86_vmulps_ymm, "vmulps")
DEFINE_YMM_LAT_THR(x86_vpaddd_ymm, "vpaddd")
DEFINE_YMM_LAT_THR(x86_vpmulld_ymm, "vpmulld")
DEFINE_YMM_LAT_THR(x86_vpand_ymm, "vpand")
DEFINE_YMM_LAT_THR(x86_vpor_ymm, "vpor")
DEFINE_YMM_LAT_THR(x86_vpxor_ymm, "vpxor")
DEFINE_YMM_LAT_THR(x86_vpandn_ymm, "vpandn")
DEFINE_YMM_NOT(x86_vpxor_not_ymm)
DEFINE_YMM_LAT_THR(x86_vpermps_ymm, "vpermps")

DEFINE_YMM_IMMEDIATE_SHIFT(x86_vpsllw_ymm_imm7, "vpsllw")
DEFINE_YMM_IMMEDIATE_SHIFT(x86_vpslld_ymm_imm7, "vpslld")
DEFINE_YMM_IMMEDIATE_SHIFT(x86_vpsllq_ymm_imm7, "vpsllq")
DEFINE_YMM_IMMEDIATE_SHIFT(x86_vpsrlw_ymm_imm7, "vpsrlw")
DEFINE_YMM_IMMEDIATE_SHIFT(x86_vpsrld_ymm_imm7, "vpsrld")
DEFINE_YMM_IMMEDIATE_SHIFT(x86_vpsrlq_ymm_imm7, "vpsrlq")
DEFINE_YMM_IMMEDIATE_SHIFT(x86_vpsraw_ymm_imm7, "vpsraw")
DEFINE_YMM_IMMEDIATE_SHIFT(x86_vpsrad_ymm_imm7, "vpsrad")

DEFINE_YMM_COUNT_SHIFT(x86_vpsllw_ymm_xmm, "vpsllw", "xmm13")
DEFINE_YMM_COUNT_SHIFT(x86_vpslld_ymm_xmm, "vpslld", "xmm13")
DEFINE_YMM_COUNT_SHIFT(x86_vpsllq_ymm_xmm, "vpsllq", "xmm13")
DEFINE_YMM_COUNT_SHIFT(x86_vpsrlw_ymm_xmm, "vpsrlw", "xmm13")
DEFINE_YMM_COUNT_SHIFT(x86_vpsrld_ymm_xmm, "vpsrld", "xmm13")
DEFINE_YMM_COUNT_SHIFT(x86_vpsrlq_ymm_xmm, "vpsrlq", "xmm13")
DEFINE_YMM_COUNT_SHIFT(x86_vpsraw_ymm_xmm, "vpsraw", "xmm13")
DEFINE_YMM_COUNT_SHIFT(x86_vpsrad_ymm_xmm, "vpsrad", "xmm13")

DEFINE_YMM_COUNT_SHIFT(x86_vpsllvd_ymm, "vpsllvd", "ymm12")
DEFINE_YMM_COUNT_SHIFT(x86_vpsllvq_ymm, "vpsllvq", "ymm13")
DEFINE_YMM_COUNT_SHIFT(x86_vpsrlvd_ymm, "vpsrlvd", "ymm12")
DEFINE_YMM_COUNT_SHIFT(x86_vpsrlvq_ymm, "vpsrlvq", "ymm13")
DEFINE_YMM_COUNT_SHIFT(x86_vpsravd_ymm, "vpsravd", "ymm12")

#define X86_YMM_BLEND_GROUP(op)                                              \
  op " %%ymm13, %%ymm14, %%ymm0, %%ymm0\n\t"                               \
  op " %%ymm13, %%ymm14, %%ymm1, %%ymm1\n\t"                               \
  op " %%ymm13, %%ymm14, %%ymm2, %%ymm2\n\t"                               \
  op " %%ymm13, %%ymm14, %%ymm3, %%ymm3\n\t"                               \
  op " %%ymm13, %%ymm14, %%ymm4, %%ymm4\n\t"                               \
  op " %%ymm13, %%ymm14, %%ymm5, %%ymm5\n\t"                               \
  op " %%ymm13, %%ymm14, %%ymm6, %%ymm6\n\t"                               \
  op " %%ymm13, %%ymm14, %%ymm7, %%ymm7\n\t"
#define DEFINE_YMM_BLEND(name, op)                                           \
  X86_DEFINE_KERNEL(name##_latency, X86_YMM_SETUP,                          \
      VECOPS_INST_REPEAT_32(op " %%ymm13, %%ymm14, %%ymm0, %%ymm0\n\t"),    \
      X86_YMM_CLOBBERS)                                                      \
  X86_DEFINE_KERNEL(name##_throughput, X86_YMM_SETUP,                       \
      VECOPS_INST_REPEAT_4(X86_YMM_BLEND_GROUP(op)), X86_YMM_CLOBBERS)

DEFINE_YMM_BLEND(x86_vblendvps_ymm, "vblendvps")
DEFINE_YMM_BLEND(x86_vblendvpd_ymm, "vblendvpd")
DEFINE_YMM_BLEND(x86_vpblendvb_ymm, "vpblendvb")

#define X86_DEFINE_MEMORY_KERNEL(name, setup, body, ...)                     \
  extern "C" VECOPS_INST_NOINLINE_USED void name(                           \
      uint64_t loops, void* memory, uint64_t memory_bytes) {                 \
    if (loops == 0 || memory_bytes == 0) return;                             \
    auto* base = static_cast<uint8_t*>(memory);                              \
    auto* ptr = base;                                                        \
    auto* end = base + memory_bytes;                                         \
    __asm__ volatile(                                                        \
        setup                                                               \
        ".p2align 6\n\t"                                                    \
        "1:\n\t"                                                            \
        body                                                                \
        "dec %[loops]\n\t"                                                  \
        "jnz 1b\n\t"                                                        \
        : [loops] "+r"(loops), [ptr] "+&r"(ptr)                             \
        : [base] "r"(base), [end] "r"(end)                                 \
        : "cc", "memory" __VA_OPT__(,) __VA_ARGS__);                        \
  }
#define X86_YMM_STREAM_ADVANCE                                               \
  "add $256, %[ptr]\n\t"                                                    \
  "cmp %[end], %[ptr]\n\t"                                                  \
  "cmovae %[base], %[ptr]\n\t"

X86_DEFINE_MEMORY_KERNEL(x86_ymm_stream_control, "", X86_YMM_STREAM_ADVANCE)
X86_DEFINE_MEMORY_KERNEL(
    x86_ymm_stream_load,
    "",
    "vmovdqa 0(%[ptr]), %%ymm0\n\t"
    "vmovdqa 32(%[ptr]), %%ymm1\n\t"
    "vmovdqa 64(%[ptr]), %%ymm2\n\t"
    "vmovdqa 96(%[ptr]), %%ymm3\n\t"
    "vmovdqa 128(%[ptr]), %%ymm4\n\t"
    "vmovdqa 160(%[ptr]), %%ymm5\n\t"
    "vmovdqa 192(%[ptr]), %%ymm6\n\t"
    "vmovdqa 224(%[ptr]), %%ymm7\n\t"
    X86_YMM_STREAM_ADVANCE,
    "ymm0", "ymm1", "ymm2", "ymm3", "ymm4", "ymm5", "ymm6", "ymm7")
X86_DEFINE_MEMORY_KERNEL(
    x86_ymm_stream_store,
    "vpxor %%ymm14, %%ymm14, %%ymm14\n\t",
    "vmovdqa %%ymm14, 0(%[ptr])\n\t"
    "vmovdqa %%ymm14, 32(%[ptr])\n\t"
    "vmovdqa %%ymm14, 64(%[ptr])\n\t"
    "vmovdqa %%ymm14, 96(%[ptr])\n\t"
    "vmovdqa %%ymm14, 128(%[ptr])\n\t"
    "vmovdqa %%ymm14, 160(%[ptr])\n\t"
    "vmovdqa %%ymm14, 192(%[ptr])\n\t"
    "vmovdqa %%ymm14, 224(%[ptr])\n\t"
    X86_YMM_STREAM_ADVANCE,
    "ymm14")
X86_DEFINE_MEMORY_KERNEL(
    x86_ymm_pointer_chase,
    "",
    "vmovdqa (%[ptr]), %%ymm0\n\t"
    "vmovq %%xmm0, %[ptr]\n\t",
    "ymm0")
X86_DEFINE_MEMORY_KERNEL(
    x86_ymm_store_forward,
    "vpxor %%ymm0, %%ymm0, %%ymm0\n\t",
    "vmovdqa %%ymm0, (%[ptr])\n\t"
    "vmovdqa (%[ptr]), %%ymm0\n\t",
    "ymm0")
X86_DEFINE_MEMORY_KERNEL(
    x86_ymm_crossline_load,
    "add $48, %[ptr]\n\t",
    "vmovdqu 0(%[ptr]), %%ymm0\n\t"
    "vmovdqu 64(%[ptr]), %%ymm1\n\t"
    "vmovdqu 128(%[ptr]), %%ymm2\n\t"
    "vmovdqu 192(%[ptr]), %%ymm3\n\t"
    "vmovdqu 256(%[ptr]), %%ymm4\n\t"
    "vmovdqu 320(%[ptr]), %%ymm5\n\t"
    "vmovdqu 384(%[ptr]), %%ymm6\n\t"
    "vmovdqu 448(%[ptr]), %%ymm7\n\t",
    "ymm0", "ymm1", "ymm2", "ymm3", "ymm4", "ymm5", "ymm6", "ymm7")
X86_DEFINE_MEMORY_KERNEL(
    x86_ymm_crossline_store,
    "add $48, %[ptr]\n\t"
    "vpxor %%ymm14, %%ymm14, %%ymm14\n\t",
    "vmovdqu %%ymm14, 0(%[ptr])\n\t"
    "vmovdqu %%ymm14, 64(%[ptr])\n\t"
    "vmovdqu %%ymm14, 128(%[ptr])\n\t"
    "vmovdqu %%ymm14, 192(%[ptr])\n\t"
    "vmovdqu %%ymm14, 256(%[ptr])\n\t"
    "vmovdqu %%ymm14, 320(%[ptr])\n\t"
    "vmovdqu %%ymm14, 384(%[ptr])\n\t"
    "vmovdqu %%ymm14, 448(%[ptr])\n\t",
    "ymm14")
X86_DEFINE_MEMORY_KERNEL(
    x86_ymm_mask_load,
    "vpcmpeqd %%ymm13, %%ymm13, %%ymm13\n\t"
    "vpsllq $32, %%ymm13, %%ymm13\n\t",
    "vmaskmovps (%[ptr]), %%ymm13, %%ymm0\n\t"
    "vmaskmovps 32(%[ptr]), %%ymm13, %%ymm1\n\t"
    "vmaskmovps 64(%[ptr]), %%ymm13, %%ymm2\n\t"
    "vmaskmovps 96(%[ptr]), %%ymm13, %%ymm3\n\t"
    "vmaskmovps 128(%[ptr]), %%ymm13, %%ymm4\n\t"
    "vmaskmovps 160(%[ptr]), %%ymm13, %%ymm5\n\t"
    "vmaskmovps 192(%[ptr]), %%ymm13, %%ymm6\n\t"
    "vmaskmovps 224(%[ptr]), %%ymm13, %%ymm7\n\t",
    "ymm0", "ymm1", "ymm2", "ymm3", "ymm4", "ymm5", "ymm6", "ymm7", "ymm13")
X86_DEFINE_MEMORY_KERNEL(
    x86_ymm_mask_store,
    "vpcmpeqd %%ymm13, %%ymm13, %%ymm13\n\t"
    "vpsllq $32, %%ymm13, %%ymm13\n\t"
    "vpxor %%ymm14, %%ymm14, %%ymm14\n\t",
    "vmaskmovps %%ymm14, %%ymm13, 0(%[ptr])\n\t"
    "vmaskmovps %%ymm14, %%ymm13, 32(%[ptr])\n\t"
    "vmaskmovps %%ymm14, %%ymm13, 64(%[ptr])\n\t"
    "vmaskmovps %%ymm14, %%ymm13, 96(%[ptr])\n\t"
    "vmaskmovps %%ymm14, %%ymm13, 128(%[ptr])\n\t"
    "vmaskmovps %%ymm14, %%ymm13, 160(%[ptr])\n\t"
    "vmaskmovps %%ymm14, %%ymm13, 192(%[ptr])\n\t"
    "vmaskmovps %%ymm14, %%ymm13, 224(%[ptr])\n\t",
    "ymm13", "ymm14")

#define X86_YMM_CONV_ADVANCE(bytes)                                         \
  "add $" #bytes ", %[ptr]\n\t"                                             \
  "cmp %[end], %[ptr]\n\t"                                                  \
  "cmovae %[base], %[ptr]\n\t"
#define X86_DEFINE_YMM_CONV_STREAM_CONTROL(name, stride)                    \
  X86_DEFINE_MEMORY_KERNEL(name, "", X86_YMM_CONV_ADVANCE(stride))
#define X86_DEFINE_YMM_CONV_LOAD(name, op, o1, o2, o3, o4, o5, o6, o7, stride) \
  X86_DEFINE_MEMORY_KERNEL(                                                  \
      name, "",                                                             \
      op " 0(%[ptr]), %%ymm0\n\t"                                          \
      op " " #o1 "(%[ptr]), %%ymm1\n\t"                                    \
      op " " #o2 "(%[ptr]), %%ymm2\n\t"                                    \
      op " " #o3 "(%[ptr]), %%ymm3\n\t"                                    \
      op " " #o4 "(%[ptr]), %%ymm4\n\t"                                    \
      op " " #o5 "(%[ptr]), %%ymm5\n\t"                                    \
      op " " #o6 "(%[ptr]), %%ymm6\n\t"                                    \
      op " " #o7 "(%[ptr]), %%ymm7\n\t"                                    \
      X86_YMM_CONV_ADVANCE(stride),                                         \
      "ymm0", "ymm1", "ymm2", "ymm3", "ymm4", "ymm5", "ymm6", "ymm7")
X86_DEFINE_YMM_CONV_STREAM_CONTROL(x86_ymm_conv_control_4b, 32)
X86_DEFINE_YMM_CONV_STREAM_CONTROL(x86_ymm_conv_control_8b, 64)
X86_DEFINE_YMM_CONV_STREAM_CONTROL(x86_ymm_conv_control_16b, 128)
#define X86_YMM_CONV_LOAD_4(name, op)                                       \
  X86_DEFINE_YMM_CONV_LOAD(name, op, 4, 8, 12, 16, 20, 24, 28, 32)
#define X86_YMM_CONV_LOAD_8(name, op)                                       \
  X86_DEFINE_YMM_CONV_LOAD(name, op, 8, 16, 24, 32, 40, 48, 56, 64)
#define X86_YMM_CONV_LOAD_16(name, op)                                      \
  X86_DEFINE_YMM_CONV_LOAD(name, op, 16, 32, 48, 64, 80, 96, 112, 128)
X86_YMM_CONV_LOAD_16(x86_ymm_vpmovsxbw_stream, "vpmovsxbw")
X86_YMM_CONV_LOAD_16(x86_ymm_vpmovzxbw_stream, "vpmovzxbw")
X86_YMM_CONV_LOAD_8(x86_ymm_vpmovsxbd_stream, "vpmovsxbd")
X86_YMM_CONV_LOAD_8(x86_ymm_vpmovzxbd_stream, "vpmovzxbd")
X86_YMM_CONV_LOAD_4(x86_ymm_vpmovsxbq_stream, "vpmovsxbq")
X86_YMM_CONV_LOAD_4(x86_ymm_vpmovzxbq_stream, "vpmovzxbq")
X86_YMM_CONV_LOAD_16(x86_ymm_vpmovsxwd_stream, "vpmovsxwd")
X86_YMM_CONV_LOAD_16(x86_ymm_vpmovzxwd_stream, "vpmovzxwd")
X86_YMM_CONV_LOAD_8(x86_ymm_vpmovsxwq_stream, "vpmovsxwq")
X86_YMM_CONV_LOAD_8(x86_ymm_vpmovzxwq_stream, "vpmovzxwq")
X86_YMM_CONV_LOAD_16(x86_ymm_vpmovsxdq_stream, "vpmovsxdq")
X86_YMM_CONV_LOAD_16(x86_ymm_vpmovzxdq_stream, "vpmovzxdq")

#define X86_DEFINE_YMM_INDEX_CHASE(name, op, normalize, signed_anchor)       \
  extern "C" VECOPS_INST_NOINLINE_USED void name(                           \
      uint64_t loops, void* memory, uint64_t memory_bytes) {                 \
    if (loops == 0 || memory_bytes == 0) return;                             \
    auto* base = static_cast<uint8_t*>(memory);                              \
    auto* ptr = base;                                                        \
    auto* anchor = base + (signed_anchor);                                   \
    __asm__ volatile(                                                        \
        ".p2align 6\n\t"                                                    \
        "1:\n\t"                                                            \
        op " (%[ptr]), %%ymm0\n\t"                                         \
        normalize                                                           \
        "shl $6, %%rax\n\t"                                                 \
        "lea (%[anchor], %%rax), %[ptr]\n\t"                                \
        "dec %[loops]\n\t"                                                  \
        "jnz 1b\n\t"                                                        \
        : [loops] "+r"(loops), [ptr] "+&r"(ptr)                             \
        : [anchor] "r"(anchor)                                              \
        : "cc", "memory", "rax", "ymm0");                                  \
  }
#define X86_YMM_EXTRACT_S16 "vmovd %%xmm0, %%eax\n\tmovswq %%ax, %%rax\n\t"
#define X86_YMM_EXTRACT_U16 "vmovd %%xmm0, %%eax\n\tmovzwl %%ax, %%eax\n\t"
#define X86_YMM_EXTRACT_S32 "vmovd %%xmm0, %%eax\n\tmovslq %%eax, %%rax\n\t"
#define X86_YMM_EXTRACT_U32 "vmovd %%xmm0, %%eax\n\t"
#define X86_YMM_EXTRACT_64 "vmovq %%xmm0, %%rax\n\t"
#define X86_YMM_INDEX_PAIR(symbol_s, op_s, symbol_u, op_u, extract_s, extract_u) \
  X86_DEFINE_YMM_INDEX_CHASE(symbol_s, op_s, extract_s, 4096)                \
  X86_DEFINE_YMM_INDEX_CHASE(symbol_u, op_u, extract_u, 0)
X86_YMM_INDEX_PAIR(
    x86_ymm_vpmovsxbw_index, "vpmovsxbw", x86_ymm_vpmovzxbw_index,
    "vpmovzxbw", X86_YMM_EXTRACT_S16, X86_YMM_EXTRACT_U16)
X86_YMM_INDEX_PAIR(
    x86_ymm_vpmovsxbd_index, "vpmovsxbd", x86_ymm_vpmovzxbd_index,
    "vpmovzxbd", X86_YMM_EXTRACT_S32, X86_YMM_EXTRACT_U32)
X86_YMM_INDEX_PAIR(
    x86_ymm_vpmovsxbq_index, "vpmovsxbq", x86_ymm_vpmovzxbq_index,
    "vpmovzxbq", X86_YMM_EXTRACT_64, X86_YMM_EXTRACT_64)
X86_YMM_INDEX_PAIR(
    x86_ymm_vpmovsxwd_index, "vpmovsxwd", x86_ymm_vpmovzxwd_index,
    "vpmovzxwd", X86_YMM_EXTRACT_S32, X86_YMM_EXTRACT_U32)
X86_YMM_INDEX_PAIR(
    x86_ymm_vpmovsxwq_index, "vpmovsxwq", x86_ymm_vpmovzxwq_index,
    "vpmovzxwq", X86_YMM_EXTRACT_64, X86_YMM_EXTRACT_64)
X86_YMM_INDEX_PAIR(
    x86_ymm_vpmovsxdq_index, "vpmovsxdq", x86_ymm_vpmovzxdq_index,
    "vpmovzxdq", X86_YMM_EXTRACT_64, X86_YMM_EXTRACT_64)

#define X86_DEFINE_YMM_SCALAR_INDEX_CONTROL(name, load_op, signed_anchor)    \
  extern "C" VECOPS_INST_NOINLINE_USED void name(                           \
      uint64_t loops, void* memory, uint64_t memory_bytes) {                 \
    if (loops == 0 || memory_bytes == 0) return;                             \
    auto* base = static_cast<uint8_t*>(memory);                              \
    auto* ptr = base;                                                        \
    auto* anchor = base + (signed_anchor);                                   \
    __asm__ volatile(                                                        \
        ".p2align 6\n\t"                                                    \
        "1:\n\t"                                                            \
        load_op                                                             \
        "shl $6, %%rax\n\t"                                                 \
        "lea (%[anchor], %%rax), %[ptr]\n\t"                                \
        "dec %[loops]\n\t"                                                  \
        "jnz 1b\n\t"                                                        \
        : [loops] "+r"(loops), [ptr] "+&r"(ptr)                             \
        : [anchor] "r"(anchor)                                              \
        : "cc", "memory", "rax");                                          \
  }
X86_DEFINE_YMM_SCALAR_INDEX_CONTROL(
    x86_ymm_index_control_s8, "movsbq (%[ptr]), %%rax\n\t", 4096)
X86_DEFINE_YMM_SCALAR_INDEX_CONTROL(
    x86_ymm_index_control_u8, "movzbl (%[ptr]), %%eax\n\t", 0)
X86_DEFINE_YMM_SCALAR_INDEX_CONTROL(
    x86_ymm_index_control_s16, "movswq (%[ptr]), %%rax\n\t", 4096)
X86_DEFINE_YMM_SCALAR_INDEX_CONTROL(
    x86_ymm_index_control_u16, "movzwl (%[ptr]), %%eax\n\t", 0)
X86_DEFINE_YMM_SCALAR_INDEX_CONTROL(
    x86_ymm_index_control_s32, "movslq (%[ptr]), %%rax\n\t", 4096)
X86_DEFINE_YMM_SCALAR_INDEX_CONTROL(
    x86_ymm_index_control_u32, "movl (%[ptr]), %%eax\n\t", 0)

const InstructionCase kCases[] = {
    {"empty_loop", "x86", "Loop-control baseline", MeasureMode::Control,
     x86_control_empty_loop, 32, 0, 0, 256},
    {"nop_32", "x86", "Thirty-two one-byte NOPs", MeasureMode::Control,
     x86_control_nop_32, 32, 1, 0, 256},
#define X86_YMM_CASES(label, symbol)                                         \
    {label, "AVX2", "Single true-dependency chain", MeasureMode::Latency,   \
     symbol##_latency, 32, 1, 1, 256},                                      \
    {label, "AVX2", "Eight independent destination chains",                \
     MeasureMode::Throughput, symbol##_throughput, 32, 1, 8, 256}
#define X86_YMM_CATEGORY_CASES(label, symbol, category_value)                \
    {label, "AVX2", "Single true-dependency chain", MeasureMode::Latency,    \
     symbol##_latency, 32, 1, 1, 256, category_value},                      \
    {label, "AVX2", "Eight independent destination chains",                 \
     MeasureMode::Throughput, symbol##_throughput, 32, 1, 8, 256, category_value}
    X86_YMM_CASES("vaddps_ymm_ymm_ymm", x86_vaddps_ymm),
    X86_YMM_CASES("vmulps_ymm_ymm_ymm", x86_vmulps_ymm),
    X86_YMM_CASES("vpaddd_ymm_ymm_ymm", x86_vpaddd_ymm),
    X86_YMM_CASES("vpmulld_ymm_ymm_ymm", x86_vpmulld_ymm),
    X86_YMM_CATEGORY_CASES("vpand_ymm_ymm_ymm", x86_vpand_ymm,
                           InstructionCategory::Bitwise),
    X86_YMM_CATEGORY_CASES("vpor_ymm_ymm_ymm", x86_vpor_ymm,
                           InstructionCategory::Bitwise),
    X86_YMM_CATEGORY_CASES("vpxor_ymm_ymm_ymm", x86_vpxor_ymm,
                           InstructionCategory::Bitwise),
    X86_YMM_CATEGORY_CASES("vpandn_ymm_ymm_ymm", x86_vpandn_ymm,
                           InstructionCategory::Bitwise),
    X86_YMM_CATEGORY_CASES("vpxor_ymm_allones_not", x86_vpxor_not_ymm,
                           InstructionCategory::Bitwise),
    X86_YMM_CASES("vpermps_ymm_ymm_ymm", x86_vpermps_ymm),
#define X86_YMM_SHIFT_CASE(label, symbol)                                    \
    X86_YMM_CATEGORY_CASES(label, symbol, InstructionCategory::Shift)
    X86_YMM_SHIFT_CASE("vpsllw_ymm_ymm_imm7", x86_vpsllw_ymm_imm7),
    X86_YMM_SHIFT_CASE("vpslld_ymm_ymm_imm7", x86_vpslld_ymm_imm7),
    X86_YMM_SHIFT_CASE("vpsllq_ymm_ymm_imm7", x86_vpsllq_ymm_imm7),
    X86_YMM_SHIFT_CASE("vpsrlw_ymm_ymm_imm7", x86_vpsrlw_ymm_imm7),
    X86_YMM_SHIFT_CASE("vpsrld_ymm_ymm_imm7", x86_vpsrld_ymm_imm7),
    X86_YMM_SHIFT_CASE("vpsrlq_ymm_ymm_imm7", x86_vpsrlq_ymm_imm7),
    X86_YMM_SHIFT_CASE("vpsraw_ymm_ymm_imm7", x86_vpsraw_ymm_imm7),
    X86_YMM_SHIFT_CASE("vpsrad_ymm_ymm_imm7", x86_vpsrad_ymm_imm7),
    X86_YMM_SHIFT_CASE("vpsllw_ymm_ymm_xmm", x86_vpsllw_ymm_xmm),
    X86_YMM_SHIFT_CASE("vpslld_ymm_ymm_xmm", x86_vpslld_ymm_xmm),
    X86_YMM_SHIFT_CASE("vpsllq_ymm_ymm_xmm", x86_vpsllq_ymm_xmm),
    X86_YMM_SHIFT_CASE("vpsrlw_ymm_ymm_xmm", x86_vpsrlw_ymm_xmm),
    X86_YMM_SHIFT_CASE("vpsrld_ymm_ymm_xmm", x86_vpsrld_ymm_xmm),
    X86_YMM_SHIFT_CASE("vpsrlq_ymm_ymm_xmm", x86_vpsrlq_ymm_xmm),
    X86_YMM_SHIFT_CASE("vpsraw_ymm_ymm_xmm", x86_vpsraw_ymm_xmm),
    X86_YMM_SHIFT_CASE("vpsrad_ymm_ymm_xmm", x86_vpsrad_ymm_xmm),
    X86_YMM_SHIFT_CASE("vpsllvd_ymm_ymm_ymm", x86_vpsllvd_ymm),
    X86_YMM_SHIFT_CASE("vpsllvq_ymm_ymm_ymm", x86_vpsllvq_ymm),
    X86_YMM_SHIFT_CASE("vpsrlvd_ymm_ymm_ymm", x86_vpsrlvd_ymm),
    X86_YMM_SHIFT_CASE("vpsrlvq_ymm_ymm_ymm", x86_vpsrlvq_ymm),
    X86_YMM_SHIFT_CASE("vpsravd_ymm_ymm_ymm", x86_vpsravd_ymm),
    X86_YMM_CATEGORY_CASES("vblendvps_ymm_ymm_ymm_ymm", x86_vblendvps_ymm,
                           InstructionCategory::Blend),
    X86_YMM_CATEGORY_CASES("vblendvpd_ymm_ymm_ymm_ymm", x86_vblendvpd_ymm,
                           InstructionCategory::Blend),
    X86_YMM_CATEGORY_CASES("vpblendvb_ymm_ymm_ymm_ymm", x86_vpblendvb_ymm,
                           InstructionCategory::Blend),
#define X86_YMM_MEMORY_CASE(label, description, mode_value, memory_symbol, seq, inst, chains,\
                            pattern_value, level_value, bytes_value, baseline_value)         \
    {label, "AVX2", description, mode_value, nullptr, seq, inst, chains, 256,                \
     InstructionCategory::Memory, memory_symbol, pattern_value, level_value,                 \
     bytes_value, baseline_value}
#define X86_YMM_MEMORY_LEVEL(level_label, level_value)                        \
    X86_YMM_MEMORY_CASE("stream_control_" level_label, "Address-generation baseline",       \
                        MeasureMode::Control, x86_ymm_stream_control, 8, 0, 0,               \
                        MemoryPattern::Stream, level_value, 0, "empty_loop"),                \
    X86_YMM_MEMORY_CASE("vmovdqa_stream_load_" level_label, "Sequential aligned vector load",\
                        MeasureMode::Throughput, x86_ymm_stream_load, 8, 1, 8,               \
                        MemoryPattern::Stream, level_value, 32,                              \
                        "stream_control_" level_label),                                      \
    X86_YMM_MEMORY_CASE("vmovdqa_stream_store_" level_label,                                 \
                        "Sequential aligned vector store", MeasureMode::Throughput,          \
                        x86_ymm_stream_store, 8, 1, 8, MemoryPattern::Stream, level_value, 32,\
                        "stream_control_" level_label),                                      \
    X86_YMM_MEMORY_CASE("vmovdqa+vmovq_pointer_chase_" level_label,                          \
                        "Dependent vector-load cache-latency sequence", MeasureMode::Latency,\
                        x86_ymm_pointer_chase, 1, 2, 1, MemoryPattern::PointerChase,          \
                        level_value, 32, "empty_loop")
    X86_YMM_MEMORY_LEVEL("l1", WorkingSetLevel::L1),
    X86_YMM_MEMORY_LEVEL("l2", WorkingSetLevel::L2),
    X86_YMM_MEMORY_LEVEL("l3", WorkingSetLevel::L3),
    X86_YMM_MEMORY_LEVEL("beyond_last_reported_cache", WorkingSetLevel::BeyondLastCache),
    X86_YMM_MEMORY_CASE("vmovdqa_store+load_forward",
                        "L1 store-to-load forwarding round trip", MeasureMode::Latency,
                        x86_ymm_store_forward, 1, 2, 1, MemoryPattern::StoreForward,
                        WorkingSetLevel::L1, 32, "empty_loop"),
    X86_YMM_MEMORY_CASE("vmovdqu_crossline_load", "Cross-cache-line unaligned load",
                        MeasureMode::Throughput, x86_ymm_crossline_load, 8, 1, 8,
                        MemoryPattern::Hot, WorkingSetLevel::L1, 32, "empty_loop"),
    X86_YMM_MEMORY_CASE("vmovdqu_crossline_store", "Cross-cache-line unaligned store",
                        MeasureMode::Throughput, x86_ymm_crossline_store, 8, 1, 8,
                        MemoryPattern::Hot, WorkingSetLevel::L1, 32, "empty_loop"),
    X86_YMM_MEMORY_CASE("vmaskmovps_mask50_load", "Half-mask contiguous load",
                        MeasureMode::Throughput, x86_ymm_mask_load, 8, 1, 8,
                        MemoryPattern::Hot, WorkingSetLevel::L1, 16, "empty_loop"),
    X86_YMM_MEMORY_CASE("vmaskmovps_mask50_store", "Half-mask contiguous store",
                        MeasureMode::Throughput, x86_ymm_mask_store, 8, 1, 8,
                        MemoryPattern::Hot, WorkingSetLevel::L1, 16, "empty_loop"),
#define X86_YMM_CONV_CASE(label, description, mode_value, symbol, seq, inst, chains, \
                          pattern, level, bytes, baseline, kind, src, dst, elems, cap) \
    {label, "AVX2", description, mode_value, nullptr, seq, inst, chains, 256,          \
     InstructionCategory::Memory, symbol, pattern, level, bytes, baseline, kind, src,  \
     dst, elems, cap}
#define X86_YMM_CONV_CONTROLS(level_label, level_value)                       \
    X86_YMM_CONV_CASE("conv_stream_control_4b_" level_label,                  \
                      "4-byte conversion-stream address baseline", MeasureMode::Control,\
                      x86_ymm_conv_control_4b, 8, 0, 0, MemoryPattern::Stream, level_value,\
                      0, "empty_loop", MemoryConversionKind::None, 0, 0, 0, 0),           \
    X86_YMM_CONV_CASE("conv_stream_control_8b_" level_label,                  \
                      "8-byte conversion-stream address baseline", MeasureMode::Control,\
                      x86_ymm_conv_control_8b, 8, 0, 0, MemoryPattern::Stream, level_value,\
                      0, "empty_loop", MemoryConversionKind::None, 0, 0, 0, 0),           \
    X86_YMM_CONV_CASE("conv_stream_control_16b_" level_label,                 \
                      "16-byte conversion-stream address baseline", MeasureMode::Control,\
                      x86_ymm_conv_control_16b, 8, 0, 0, MemoryPattern::Stream, level_value,\
                      0, "empty_loop", MemoryConversionKind::None, 0, 0, 0, 0)
    X86_YMM_CONV_CONTROLS("l1", WorkingSetLevel::L1),
    X86_YMM_CONV_CONTROLS("l2", WorkingSetLevel::L2),
    X86_YMM_CONV_CONTROLS("l3", WorkingSetLevel::L3),
    X86_YMM_CONV_CONTROLS(
        "beyond_last_reported_cache", WorkingSetLevel::BeyondLastCache),
#define X86_YMM_CONV_STREAM_LEVEL(label, symbol, kind, src, dst, bytes, elems, bytes_tag, \
                                  level_label, level_value)                              \
    X86_YMM_CONV_CASE(label "_stream_" level_label,                                 \
                      "Sequential integer extending memory load", MeasureMode::Throughput,\
                      symbol, 8, 1, 8, MemoryPattern::Stream, level_value, bytes,          \
                      "conv_stream_control_" bytes_tag "_" level_label, kind, src, dst,  \
                      elems, 0)
#define X86_YMM_CONV_STREAM_LEVELS(label, symbol, kind, src, dst, bytes, elems, bytes_tag) \
    X86_YMM_CONV_STREAM_LEVEL(label, symbol, kind, src, dst, bytes, elems, bytes_tag,     \
                              "l1", WorkingSetLevel::L1),                                 \
    X86_YMM_CONV_STREAM_LEVEL(label, symbol, kind, src, dst, bytes, elems, bytes_tag,     \
                              "l2", WorkingSetLevel::L2),                                 \
    X86_YMM_CONV_STREAM_LEVEL(label, symbol, kind, src, dst, bytes, elems, bytes_tag,     \
                              "l3", WorkingSetLevel::L3),                                 \
    X86_YMM_CONV_STREAM_LEVEL(label, symbol, kind, src, dst, bytes, elems, bytes_tag,     \
                              "beyond_last_reported_cache",                              \
                              WorkingSetLevel::BeyondLastCache)
#define X86_YMM_EXTEND_LEVELS(label_s, label_u, symbol_s, symbol_u, src, dst, bytes, elems, bytes_tag) \
    X86_YMM_CONV_STREAM_LEVELS(label_s, symbol_s, MemoryConversionKind::SignExtendLoad,    \
                               src, dst, bytes, elems, bytes_tag),                          \
    X86_YMM_CONV_STREAM_LEVELS(label_u, symbol_u, MemoryConversionKind::ZeroExtendLoad,    \
                               src, dst, bytes, elems, bytes_tag)
    X86_YMM_EXTEND_LEVELS(
        "vpmovsxbw", "vpmovzxbw", x86_ymm_vpmovsxbw_stream,
        x86_ymm_vpmovzxbw_stream, 8, 16, 16, 16, "16b"),
    X86_YMM_EXTEND_LEVELS(
        "vpmovsxbd", "vpmovzxbd", x86_ymm_vpmovsxbd_stream,
        x86_ymm_vpmovzxbd_stream, 8, 32, 8, 8, "8b"),
    X86_YMM_EXTEND_LEVELS(
        "vpmovsxbq", "vpmovzxbq", x86_ymm_vpmovsxbq_stream,
        x86_ymm_vpmovzxbq_stream, 8, 64, 4, 4, "4b"),
    X86_YMM_EXTEND_LEVELS(
        "vpmovsxwd", "vpmovzxwd", x86_ymm_vpmovsxwd_stream,
        x86_ymm_vpmovzxwd_stream, 16, 32, 16, 8, "16b"),
    X86_YMM_EXTEND_LEVELS(
        "vpmovsxwq", "vpmovzxwq", x86_ymm_vpmovsxwq_stream,
        x86_ymm_vpmovzxwq_stream, 16, 64, 8, 4, "8b"),
    X86_YMM_EXTEND_LEVELS(
        "vpmovsxdq", "vpmovzxdq", x86_ymm_vpmovsxdq_stream,
        x86_ymm_vpmovzxdq_stream, 32, 64, 16, 4, "16b"),
#define X86_YMM_INDEX_CONTROL(label, symbol, kind, src)                       \
    X86_YMM_CONV_CASE(label, "Matched scalar extending-load index-chase control",\
                      MeasureMode::Control, symbol, 1, 0, 1,                  \
                      MemoryPattern::PointerChase, WorkingSetLevel::L1, 0,    \
                      "empty_loop", kind, src, src, 0, 8192)
    X86_YMM_INDEX_CONTROL(
        "conv_index_control_s8", x86_ymm_index_control_s8,
        MemoryConversionKind::SignExtendLoad, 8),
    X86_YMM_INDEX_CONTROL(
        "conv_index_control_u8", x86_ymm_index_control_u8,
        MemoryConversionKind::ZeroExtendLoad, 8),
    X86_YMM_INDEX_CONTROL(
        "conv_index_control_s16", x86_ymm_index_control_s16,
        MemoryConversionKind::SignExtendLoad, 16),
    X86_YMM_INDEX_CONTROL(
        "conv_index_control_u16", x86_ymm_index_control_u16,
        MemoryConversionKind::ZeroExtendLoad, 16),
    X86_YMM_INDEX_CONTROL(
        "conv_index_control_s32", x86_ymm_index_control_s32,
        MemoryConversionKind::SignExtendLoad, 32),
    X86_YMM_INDEX_CONTROL(
        "conv_index_control_u32", x86_ymm_index_control_u32,
        MemoryConversionKind::ZeroExtendLoad, 32),
#define X86_YMM_EXTEND_LAT(label, symbol, kind, src, dst, elems, baseline)    \
    X86_YMM_CONV_CASE(label "_index_chase_l1",                              \
                      "L1 dependent extending-load index chase", MeasureMode::Latency,\
                      symbol, 1, 4, 1, MemoryPattern::PointerChase,           \
                      WorkingSetLevel::L1, src / 8 * elems, baseline, kind,   \
                      src, dst, elems, 8192)
    X86_YMM_EXTEND_LAT(
        "vpmovsxbw", x86_ymm_vpmovsxbw_index, MemoryConversionKind::SignExtendLoad,
        8, 16, 16, "conv_index_control_s8"),
    X86_YMM_EXTEND_LAT(
        "vpmovzxbw", x86_ymm_vpmovzxbw_index, MemoryConversionKind::ZeroExtendLoad,
        8, 16, 16, "conv_index_control_u8"),
    X86_YMM_EXTEND_LAT(
        "vpmovsxbd", x86_ymm_vpmovsxbd_index, MemoryConversionKind::SignExtendLoad,
        8, 32, 8, "conv_index_control_s8"),
    X86_YMM_EXTEND_LAT(
        "vpmovzxbd", x86_ymm_vpmovzxbd_index, MemoryConversionKind::ZeroExtendLoad,
        8, 32, 8, "conv_index_control_u8"),
    X86_YMM_EXTEND_LAT(
        "vpmovsxbq", x86_ymm_vpmovsxbq_index, MemoryConversionKind::SignExtendLoad,
        8, 64, 4, "conv_index_control_s8"),
    X86_YMM_EXTEND_LAT(
        "vpmovzxbq", x86_ymm_vpmovzxbq_index, MemoryConversionKind::ZeroExtendLoad,
        8, 64, 4, "conv_index_control_u8"),
    X86_YMM_EXTEND_LAT(
        "vpmovsxwd", x86_ymm_vpmovsxwd_index, MemoryConversionKind::SignExtendLoad,
        16, 32, 8, "conv_index_control_s16"),
    X86_YMM_EXTEND_LAT(
        "vpmovzxwd", x86_ymm_vpmovzxwd_index, MemoryConversionKind::ZeroExtendLoad,
        16, 32, 8, "conv_index_control_u16"),
    X86_YMM_EXTEND_LAT(
        "vpmovsxwq", x86_ymm_vpmovsxwq_index, MemoryConversionKind::SignExtendLoad,
        16, 64, 4, "conv_index_control_s16"),
    X86_YMM_EXTEND_LAT(
        "vpmovzxwq", x86_ymm_vpmovzxwq_index, MemoryConversionKind::ZeroExtendLoad,
        16, 64, 4, "conv_index_control_u16"),
    X86_YMM_EXTEND_LAT(
        "vpmovsxdq", x86_ymm_vpmovsxdq_index, MemoryConversionKind::SignExtendLoad,
        32, 64, 4, "conv_index_control_s32"),
    X86_YMM_EXTEND_LAT(
        "vpmovzxdq", x86_ymm_vpmovzxdq_index, MemoryConversionKind::ZeroExtendLoad,
        32, 64, 4, "conv_index_control_u32"),
};

#else
#error "InstructionBench-Native requires at least AVX2 on x86"
#endif

} // namespace

std::span<const InstructionCase> arch_instruction_cases() {
  return kCases;
}

} // namespace vecops::bench::inst
