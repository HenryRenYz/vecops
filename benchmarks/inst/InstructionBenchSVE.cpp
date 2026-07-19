#include "InstructionBench.h"

#include <arm_sve.h>

#include <cstdint>
#include <span>

#if !defined(__ARM_FEATURE_SVE)
#error "InstructionBenchSVE.cpp requires an SVE-enabled target"
#endif

namespace vecops::bench::inst {
namespace {

#if defined(__GNUC__) || defined(__clang__)
#define VECOPS_INST_NOINLINE_USED __attribute__((noinline, used))
#else
#define VECOPS_INST_NOINLINE_USED
#endif

#define SVE_DEFINE_KERNEL(name, setup, body, ...)                            \
  extern "C" VECOPS_INST_NOINLINE_USED void name(uint64_t loops) {          \
    if (loops == 0) return;                                                  \
    __asm__ volatile(                                                        \
        setup                                                               \
        ".p2align 6\n\t"                                                    \
        "1:\n\t"                                                          \
        body                                                                \
        "subs %[loops], %[loops], #1\n\t"                                  \
        "b.ne 1b\n\t"                                                     \
        : [loops] "+r"(loops)                                              \
        :                                                                   \
        : "cc" __VA_OPT__(,) __VA_ARGS__);                                 \
  }

extern "C" VECOPS_INST_NOINLINE_USED void sve_control_empty_loop(uint64_t loops) {
  if (loops == 0) return;
  __asm__ volatile(
      ".p2align 6\n\t"
      "1:\n\t"
      "subs %[loops], %[loops], #1\n\t"
      "b.ne 1b\n\t"
      : [loops] "+r"(loops)
      :
      : "cc");
}

SVE_DEFINE_KERNEL(
    sve_control_nop_32,
    "",
    VECOPS_INST_REPEAT_32("nop\n\t"))

#define SVE_SETUP                                                            \
  "ptrue p0.b\n\t"                                                         \
  "cntb x9\n\t"                                                            \
  "lsr x9, x9, #1\n\t"                                                     \
  "whilelo p1.b, xzr, x9\n\t"                                              \
  "eor z0.d, z0.d, z0.d\n\t"                                              \
  "eor z1.d, z1.d, z1.d\n\t"                                              \
  "eor z2.d, z2.d, z2.d\n\t"                                              \
  "eor z3.d, z3.d, z3.d\n\t"                                              \
  "eor z4.d, z4.d, z4.d\n\t"                                              \
  "eor z5.d, z5.d, z5.d\n\t"                                              \
  "eor z6.d, z6.d, z6.d\n\t"                                              \
  "eor z7.d, z7.d, z7.d\n\t"                                              \
  "eor z16.d, z16.d, z16.d\n\t"                                           \
  "eor z17.d, z17.d, z17.d\n\t"                                           \
  "dup z18.b, #7\n\t"                                                      \
  "dup z19.h, #7\n\t"                                                      \
  "dup z20.s, #7\n\t"                                                      \
  "dup z21.d, #7\n\t"

#define SVE_CLOBBERS                                                         \
  "x9", "p0", "p1", "z0", "z1", "z2", "z3", "z4", "z5", "z6", \
      "z7", "z16", "z17", "z18", "z19", "z20", "z21"

#define SVE_BINARY_GROUP(op, type)                                           \
  op " z0." type ", z0." type ", z16." type "\n\t"                     \
  op " z1." type ", z1." type ", z16." type "\n\t"                     \
  op " z2." type ", z2." type ", z16." type "\n\t"                     \
  op " z3." type ", z3." type ", z16." type "\n\t"                     \
  op " z4." type ", z4." type ", z16." type "\n\t"                     \
  op " z5." type ", z5." type ", z16." type "\n\t"                     \
  op " z6." type ", z6." type ", z16." type "\n\t"                     \
  op " z7." type ", z7." type ", z16." type "\n\t"

#define SVE_PRED_BINARY_GROUP(op, type)                                      \
  op " z0." type ", p0/m, z0." type ", z16." type "\n\t"               \
  op " z1." type ", p0/m, z1." type ", z16." type "\n\t"               \
  op " z2." type ", p0/m, z2." type ", z16." type "\n\t"               \
  op " z3." type ", p0/m, z3." type ", z16." type "\n\t"               \
  op " z4." type ", p0/m, z4." type ", z16." type "\n\t"               \
  op " z5." type ", p0/m, z5." type ", z16." type "\n\t"               \
  op " z6." type ", p0/m, z6." type ", z16." type "\n\t"               \
  op " z7." type ", p0/m, z7." type ", z16." type "\n\t"

#define SVE_FMLA_GROUP                                                       \
  "fmla z0.s, p0/m, z16.s, z17.s\n\t"                                      \
  "fmla z1.s, p0/m, z16.s, z17.s\n\t"                                      \
  "fmla z2.s, p0/m, z16.s, z17.s\n\t"                                      \
  "fmla z3.s, p0/m, z16.s, z17.s\n\t"                                      \
  "fmla z4.s, p0/m, z16.s, z17.s\n\t"                                      \
  "fmla z5.s, p0/m, z16.s, z17.s\n\t"                                      \
  "fmla z6.s, p0/m, z16.s, z17.s\n\t"                                      \
  "fmla z7.s, p0/m, z16.s, z17.s\n\t"

#define SVE_REARRANGE_GROUP(op, type)                                       \
  op " z0." type ", z0." type ", z16." type "\n\t"                     \
  op " z1." type ", z1." type ", z16." type "\n\t"                     \
  op " z2." type ", z2." type ", z16." type "\n\t"                     \
  op " z3." type ", z3." type ", z16." type "\n\t"                     \
  op " z4." type ", z4." type ", z16." type "\n\t"                     \
  op " z5." type ", z5." type ", z16." type "\n\t"                     \
  op " z6." type ", z6." type ", z16." type "\n\t"                     \
  op " z7." type ", z7." type ", z16." type "\n\t"

#define SVE_TBL_GROUP                                                        \
  "tbl z0.b, {z0.b}, z16.b\n\t"                                           \
  "tbl z1.b, {z1.b}, z16.b\n\t"                                           \
  "tbl z2.b, {z2.b}, z16.b\n\t"                                           \
  "tbl z3.b, {z3.b}, z16.b\n\t"                                           \
  "tbl z4.b, {z4.b}, z16.b\n\t"                                           \
  "tbl z5.b, {z5.b}, z16.b\n\t"                                           \
  "tbl z6.b, {z6.b}, z16.b\n\t"                                           \
  "tbl z7.b, {z7.b}, z16.b\n\t"

#define SVE_CONVERT_GROUP(op, dst_type, src_type)                            \
  op " z0." dst_type ", p0/m, z16." src_type "\n\t"                       \
  op " z1." dst_type ", p0/m, z16." src_type "\n\t"                       \
  op " z2." dst_type ", p0/m, z16." src_type "\n\t"                       \
  op " z3." dst_type ", p0/m, z16." src_type "\n\t"                       \
  op " z4." dst_type ", p0/m, z16." src_type "\n\t"                       \
  op " z5." dst_type ", p0/m, z16." src_type "\n\t"                       \
  op " z6." dst_type ", p0/m, z16." src_type "\n\t"                       \
  op " z7." dst_type ", p0/m, z16." src_type "\n\t"

#define SVE_SHIFT_IMMEDIATE_GROUP(op, type)                                  \
  op " z0." type ", p0/m, z0." type ", #7\n\t"                            \
  op " z1." type ", p0/m, z1." type ", #7\n\t"                            \
  op " z2." type ", p0/m, z2." type ", #7\n\t"                            \
  op " z3." type ", p0/m, z3." type ", #7\n\t"                            \
  op " z4." type ", p0/m, z4." type ", #7\n\t"                            \
  op " z5." type ", p0/m, z5." type ", #7\n\t"                            \
  op " z6." type ", p0/m, z6." type ", #7\n\t"                            \
  op " z7." type ", p0/m, z7." type ", #7\n\t"

#define SVE_SHIFT_VECTOR_GROUP(op, type, count)                              \
  op " z0." type ", p0/m, z0." type ", " count "." type "\n\t"           \
  op " z1." type ", p0/m, z1." type ", " count "." type "\n\t"           \
  op " z2." type ", p0/m, z2." type ", " count "." type "\n\t"           \
  op " z3." type ", p0/m, z3." type ", " count "." type "\n\t"           \
  op " z4." type ", p0/m, z4." type ", " count "." type "\n\t"           \
  op " z5." type ", p0/m, z5." type ", " count "." type "\n\t"           \
  op " z6." type ", p0/m, z6." type ", " count "." type "\n\t"           \
  op " z7." type ", p0/m, z7." type ", " count "." type "\n\t"

#define SVE_SEL_GROUP(type)                                                  \
  "sel z0." type ", p1, z16." type ", z0." type "\n\t"                   \
  "sel z1." type ", p1, z16." type ", z1." type "\n\t"                   \
  "sel z2." type ", p1, z16." type ", z2." type "\n\t"                   \
  "sel z3." type ", p1, z16." type ", z3." type "\n\t"                   \
  "sel z4." type ", p1, z16." type ", z4." type "\n\t"                   \
  "sel z5." type ", p1, z16." type ", z5." type "\n\t"                   \
  "sel z6." type ", p1, z16." type ", z6." type "\n\t"                   \
  "sel z7." type ", p1, z16." type ", z7." type "\n\t"

#define SVE_PRED_UNARY_GROUP(op, type)                                       \
  op " z0." type ", p0/m, z0." type "\n\t"                                \
  op " z1." type ", p0/m, z1." type "\n\t"                                \
  op " z2." type ", p0/m, z2." type "\n\t"                                \
  op " z3." type ", p0/m, z3." type "\n\t"                                \
  op " z4." type ", p0/m, z4." type "\n\t"                                \
  op " z5." type ", p0/m, z5." type "\n\t"                                \
  op " z6." type ", p0/m, z6." type "\n\t"                                \
  op " z7." type ", p0/m, z7." type "\n\t"

#define DEFINE_SVE_LAT_THR(name, latency_instruction, throughput_group)      \
  SVE_DEFINE_KERNEL(                                                         \
      name##_latency, SVE_SETUP,                                             \
      VECOPS_INST_REPEAT_32(latency_instruction), SVE_CLOBBERS)              \
  SVE_DEFINE_KERNEL(                                                         \
      name##_throughput, SVE_SETUP,                                          \
      VECOPS_INST_REPEAT_4(throughput_group), SVE_CLOBBERS)

DEFINE_SVE_LAT_THR(
    sve_fadd_z_s_p0m,
    "fadd z0.s, p0/m, z0.s, z16.s\n\t",
    SVE_PRED_BINARY_GROUP("fadd", "s"))
DEFINE_SVE_LAT_THR(
    sve_fmul_z_s_p0m,
    "fmul z0.s, p0/m, z0.s, z16.s\n\t",
    SVE_PRED_BINARY_GROUP("fmul", "s"))
DEFINE_SVE_LAT_THR(
    sve_fmla_z_s_p0m,
    "fmla z0.s, p0/m, z16.s, z17.s\n\t",
    SVE_FMLA_GROUP)
DEFINE_SVE_LAT_THR(
    sve_add_z_s,
    "add z0.s, z0.s, z16.s\n\t",
    SVE_BINARY_GROUP("add", "s"))
DEFINE_SVE_LAT_THR(
    sve_mul_z_s,
    "mul z0.s, p0/m, z0.s, z16.s\n\t",
    SVE_PRED_BINARY_GROUP("mul", "s"))
DEFINE_SVE_LAT_THR(
    sve_and_z_d,
    "and z0.d, z0.d, z16.d\n\t",
    SVE_BINARY_GROUP("and", "d"))
DEFINE_SVE_LAT_THR(
    sve_orr_z_d,
    "orr z0.d, z0.d, z16.d\n\t",
    SVE_BINARY_GROUP("orr", "d"))
DEFINE_SVE_LAT_THR(
    sve_eor_z_d,
    "eor z0.d, z0.d, z16.d\n\t",
    SVE_BINARY_GROUP("eor", "d"))
DEFINE_SVE_LAT_THR(
    sve_bic_z_d,
    "bic z0.d, z0.d, z16.d\n\t",
    SVE_BINARY_GROUP("bic", "d"))
DEFINE_SVE_LAT_THR(
    sve_not_z_d,
    "not z0.d, p0/m, z0.d\n\t",
    SVE_PRED_UNARY_GROUP("not", "d"))

#define DEFINE_SVE_IMMEDIATE_SHIFT(name, op, type)                           \
  DEFINE_SVE_LAT_THR(name,                                                   \
      op " z0." type ", p0/m, z0." type ", #7\n\t",                       \
      SVE_SHIFT_IMMEDIATE_GROUP(op, type))
#define DEFINE_SVE_VECTOR_SHIFT(name, op, type, count)                       \
  DEFINE_SVE_LAT_THR(name,                                                   \
      op " z0." type ", p0/m, z0." type ", " count "." type "\n\t",      \
      SVE_SHIFT_VECTOR_GROUP(op, type, count))

DEFINE_SVE_IMMEDIATE_SHIFT(sve_lsl_z_b_imm7, "lsl", "b")
DEFINE_SVE_IMMEDIATE_SHIFT(sve_lsl_z_h_imm7, "lsl", "h")
DEFINE_SVE_IMMEDIATE_SHIFT(sve_lsl_z_s_imm7, "lsl", "s")
DEFINE_SVE_IMMEDIATE_SHIFT(sve_lsl_z_d_imm7, "lsl", "d")
DEFINE_SVE_IMMEDIATE_SHIFT(sve_lsr_z_b_imm7, "lsr", "b")
DEFINE_SVE_IMMEDIATE_SHIFT(sve_lsr_z_h_imm7, "lsr", "h")
DEFINE_SVE_IMMEDIATE_SHIFT(sve_lsr_z_s_imm7, "lsr", "s")
DEFINE_SVE_IMMEDIATE_SHIFT(sve_lsr_z_d_imm7, "lsr", "d")
DEFINE_SVE_IMMEDIATE_SHIFT(sve_asr_z_b_imm7, "asr", "b")
DEFINE_SVE_IMMEDIATE_SHIFT(sve_asr_z_h_imm7, "asr", "h")
DEFINE_SVE_IMMEDIATE_SHIFT(sve_asr_z_s_imm7, "asr", "s")
DEFINE_SVE_IMMEDIATE_SHIFT(sve_asr_z_d_imm7, "asr", "d")

DEFINE_SVE_VECTOR_SHIFT(sve_lsl_z_b_z_b, "lsl", "b", "z18")
DEFINE_SVE_VECTOR_SHIFT(sve_lsl_z_h_z_h, "lsl", "h", "z19")
DEFINE_SVE_VECTOR_SHIFT(sve_lsl_z_s_z_s, "lsl", "s", "z20")
DEFINE_SVE_VECTOR_SHIFT(sve_lsl_z_d_z_d, "lsl", "d", "z21")
DEFINE_SVE_VECTOR_SHIFT(sve_lsr_z_b_z_b, "lsr", "b", "z18")
DEFINE_SVE_VECTOR_SHIFT(sve_lsr_z_h_z_h, "lsr", "h", "z19")
DEFINE_SVE_VECTOR_SHIFT(sve_lsr_z_s_z_s, "lsr", "s", "z20")
DEFINE_SVE_VECTOR_SHIFT(sve_lsr_z_d_z_d, "lsr", "d", "z21")
DEFINE_SVE_VECTOR_SHIFT(sve_asr_z_b_z_b, "asr", "b", "z18")
DEFINE_SVE_VECTOR_SHIFT(sve_asr_z_h_z_h, "asr", "h", "z19")
DEFINE_SVE_VECTOR_SHIFT(sve_asr_z_s_z_s, "asr", "s", "z20")
DEFINE_SVE_VECTOR_SHIFT(sve_asr_z_d_z_d, "asr", "d", "z21")

#define DEFINE_SVE_SEL(name, type)                                           \
  DEFINE_SVE_LAT_THR(name,                                                   \
      "sel z0." type ", p1, z16." type ", z0." type "\n\t",               \
      SVE_SEL_GROUP(type))
DEFINE_SVE_SEL(sve_sel_z_b, "b")
DEFINE_SVE_SEL(sve_sel_z_h, "h")
DEFINE_SVE_SEL(sve_sel_z_s, "s")
DEFINE_SVE_SEL(sve_sel_z_d, "d")
DEFINE_SVE_LAT_THR(
    sve_zip1_z_s,
    "zip1 z0.s, z0.s, z16.s\n\t",
    SVE_REARRANGE_GROUP("zip1", "s"))
DEFINE_SVE_LAT_THR(
    sve_zip2_z_s,
    "zip2 z0.s, z0.s, z16.s\n\t",
    SVE_REARRANGE_GROUP("zip2", "s"))
DEFINE_SVE_LAT_THR(
    sve_uzp1_z_s,
    "uzp1 z0.s, z0.s, z16.s\n\t",
    SVE_REARRANGE_GROUP("uzp1", "s"))
DEFINE_SVE_LAT_THR(
    sve_uzp2_z_s,
    "uzp2 z0.s, z0.s, z16.s\n\t",
    SVE_REARRANGE_GROUP("uzp2", "s"))
DEFINE_SVE_LAT_THR(
    sve_trn1_z_s,
    "trn1 z0.s, z0.s, z16.s\n\t",
    SVE_REARRANGE_GROUP("trn1", "s"))
DEFINE_SVE_LAT_THR(
    sve_trn2_z_s,
    "trn2 z0.s, z0.s, z16.s\n\t",
    SVE_REARRANGE_GROUP("trn2", "s"))
DEFINE_SVE_LAT_THR(
    sve_tbl_z_b,
    "tbl z0.b, {z0.b}, z16.b\n\t",
    SVE_TBL_GROUP)

SVE_DEFINE_KERNEL(
    sve_scvtf_z_s_p0m_throughput,
    SVE_SETUP,
    VECOPS_INST_REPEAT_4(SVE_CONVERT_GROUP("scvtf", "s", "s")),
    SVE_CLOBBERS)
SVE_DEFINE_KERNEL(
    sve_fcvtzs_z_s_p0m_throughput,
    SVE_SETUP,
    VECOPS_INST_REPEAT_4(SVE_CONVERT_GROUP("fcvtzs", "s", "s")),
    SVE_CLOBBERS)
SVE_DEFINE_KERNEL(
    sve_fcvt_z_s_from_z_h_throughput,
    SVE_SETUP,
    VECOPS_INST_REPEAT_4(SVE_CONVERT_GROUP("fcvt", "s", "h")),
    SVE_CLOBBERS)
SVE_DEFINE_KERNEL(
    sve_fcvt_z_h_from_z_s_throughput,
    SVE_SETUP,
    VECOPS_INST_REPEAT_4(SVE_CONVERT_GROUP("fcvt", "h", "s")),
    SVE_CLOBBERS)
SVE_DEFINE_KERNEL(
    sve_scvtf_fcvtzs_latency,
    SVE_SETUP,
    VECOPS_INST_REPEAT_16(
        "scvtf z0.s, p0/m, z0.s\n\t"
        "fcvtzs z0.s, p0/m, z0.s\n\t"),
    SVE_CLOBBERS)
SVE_DEFINE_KERNEL(
    sve_fcvt_hs_fcvt_sh_latency,
    SVE_SETUP,
    VECOPS_INST_REPEAT_16(
        "fcvt z0.s, p0/m, z0.h\n\t"
        "fcvt z0.h, p0/m, z0.s\n\t"),
    SVE_CLOBBERS)

#if defined(__ARM_FEATURE_SVE2)
SVE_DEFINE_KERNEL(
    sve_fcvtlt_z_s_from_z_h_throughput,
    SVE_SETUP,
    VECOPS_INST_REPEAT_4(SVE_CONVERT_GROUP("fcvtlt", "s", "h")),
    SVE_CLOBBERS)
#endif

#if defined(__ARM_FEATURE_BF16_VECTOR_ARITHMETIC)
SVE_DEFINE_KERNEL(
    sve_bfcvt_z_h_from_z_s_throughput,
    SVE_SETUP,
    VECOPS_INST_REPEAT_4(SVE_CONVERT_GROUP("bfcvt", "h", "s")),
    SVE_CLOBBERS)
#if defined(__ARM_FEATURE_SVE2)
SVE_DEFINE_KERNEL(
    sve_bfcvtnt_z_h_from_z_s_throughput,
    SVE_SETUP,
    VECOPS_INST_REPEAT_4(SVE_CONVERT_GROUP("bfcvtnt", "h", "s")),
    SVE_CLOBBERS)
#endif
#endif

#define SVE_DEFINE_MEMORY_KERNEL(name, setup, body, ...)                     \
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
        "subs %[loops], %[loops], #1\n\t"                                   \
        "b.ne 1b\n\t"                                                       \
        : [loops] "+r"(loops), [ptr] "+&r"(ptr)                             \
        : [base] "r"(base), [end] "r"(end)                                 \
        : "cc", "memory" __VA_OPT__(,) __VA_ARGS__);                        \
  }
#define SVE_STREAM_ADVANCE                                                   \
  "addvl %[ptr], %[ptr], #8\n\t"                                           \
  "cmp %[ptr], %[end]\n\t"                                                 \
  "csel %[ptr], %[base], %[ptr], hs\n\t"
#define SVE_MEMORY_PTRUE "ptrue p0.d\n\t"
#define SVE_MEMORY_HALF_PREDICATE                                            \
  "cntb x9\n\t"                                                            \
  "lsr x9, x9, #1\n\t"                                                     \
  "whilelo p1.b, xzr, x9\n\t"

SVE_DEFINE_MEMORY_KERNEL(sve_stream_control, "", SVE_STREAM_ADVANCE)
SVE_DEFINE_MEMORY_KERNEL(
    sve_stream_load,
    SVE_MEMORY_PTRUE,
    "ld1d {z0.d}, p0/z, [%[ptr], #0, mul vl]\n\t"
    "ld1d {z1.d}, p0/z, [%[ptr], #1, mul vl]\n\t"
    "ld1d {z2.d}, p0/z, [%[ptr], #2, mul vl]\n\t"
    "ld1d {z3.d}, p0/z, [%[ptr], #3, mul vl]\n\t"
    "ld1d {z4.d}, p0/z, [%[ptr], #4, mul vl]\n\t"
    "ld1d {z5.d}, p0/z, [%[ptr], #5, mul vl]\n\t"
    "ld1d {z6.d}, p0/z, [%[ptr], #6, mul vl]\n\t"
    "ld1d {z7.d}, p0/z, [%[ptr], #7, mul vl]\n\t"
    SVE_STREAM_ADVANCE,
    "p0", "z0", "z1", "z2", "z3", "z4", "z5", "z6", "z7")
SVE_DEFINE_MEMORY_KERNEL(
    sve_stream_store,
    SVE_MEMORY_PTRUE
    "eor z16.d, z16.d, z16.d\n\t",
    "st1d {z16.d}, p0, [%[ptr], #0, mul vl]\n\t"
    "st1d {z16.d}, p0, [%[ptr], #1, mul vl]\n\t"
    "st1d {z16.d}, p0, [%[ptr], #2, mul vl]\n\t"
    "st1d {z16.d}, p0, [%[ptr], #3, mul vl]\n\t"
    "st1d {z16.d}, p0, [%[ptr], #4, mul vl]\n\t"
    "st1d {z16.d}, p0, [%[ptr], #5, mul vl]\n\t"
    "st1d {z16.d}, p0, [%[ptr], #6, mul vl]\n\t"
    "st1d {z16.d}, p0, [%[ptr], #7, mul vl]\n\t"
    SVE_STREAM_ADVANCE,
    "p0", "z16")
SVE_DEFINE_MEMORY_KERNEL(
    sve_pointer_chase,
    SVE_MEMORY_PTRUE,
    "ld1d {z0.d}, p0/z, [%[ptr]]\n\t"
    "lastb %[ptr], p0, z0.d\n\t",
    "p0", "z0")
SVE_DEFINE_MEMORY_KERNEL(
    sve_store_forward,
    SVE_MEMORY_PTRUE
    "eor z0.d, z0.d, z0.d\n\t",
    "st1d {z0.d}, p0, [%[ptr]]\n\t"
    "ld1d {z0.d}, p0/z, [%[ptr]]\n\t",
    "p0", "z0")
SVE_DEFINE_MEMORY_KERNEL(
    sve_crossline_load,
    SVE_MEMORY_PTRUE
    "add %[ptr], %[ptr], #32\n\t",
    "ld1d {z0.d}, p0/z, [%[ptr], #0, mul vl]\n\t"
    "ld1d {z1.d}, p0/z, [%[ptr], #1, mul vl]\n\t"
    "ld1d {z2.d}, p0/z, [%[ptr], #2, mul vl]\n\t"
    "ld1d {z3.d}, p0/z, [%[ptr], #3, mul vl]\n\t"
    "ld1d {z4.d}, p0/z, [%[ptr], #4, mul vl]\n\t"
    "ld1d {z5.d}, p0/z, [%[ptr], #5, mul vl]\n\t"
    "ld1d {z6.d}, p0/z, [%[ptr], #6, mul vl]\n\t"
    "ld1d {z7.d}, p0/z, [%[ptr], #7, mul vl]\n\t",
    "p0", "z0", "z1", "z2", "z3", "z4", "z5", "z6", "z7")
SVE_DEFINE_MEMORY_KERNEL(
    sve_crossline_store,
    SVE_MEMORY_PTRUE
    "add %[ptr], %[ptr], #32\n\t"
    "eor z16.d, z16.d, z16.d\n\t",
    "st1d {z16.d}, p0, [%[ptr], #0, mul vl]\n\t"
    "st1d {z16.d}, p0, [%[ptr], #1, mul vl]\n\t"
    "st1d {z16.d}, p0, [%[ptr], #2, mul vl]\n\t"
    "st1d {z16.d}, p0, [%[ptr], #3, mul vl]\n\t"
    "st1d {z16.d}, p0, [%[ptr], #4, mul vl]\n\t"
    "st1d {z16.d}, p0, [%[ptr], #5, mul vl]\n\t"
    "st1d {z16.d}, p0, [%[ptr], #6, mul vl]\n\t"
    "st1d {z16.d}, p0, [%[ptr], #7, mul vl]\n\t",
    "p0", "z16")
SVE_DEFINE_MEMORY_KERNEL(
    sve_mask_load,
    SVE_MEMORY_HALF_PREDICATE,
    "ld1w {z0.s}, p1/z, [%[ptr], #0, mul vl]\n\t"
    "ld1w {z1.s}, p1/z, [%[ptr], #1, mul vl]\n\t"
    "ld1w {z2.s}, p1/z, [%[ptr], #2, mul vl]\n\t"
    "ld1w {z3.s}, p1/z, [%[ptr], #3, mul vl]\n\t"
    "ld1w {z4.s}, p1/z, [%[ptr], #4, mul vl]\n\t"
    "ld1w {z5.s}, p1/z, [%[ptr], #5, mul vl]\n\t"
    "ld1w {z6.s}, p1/z, [%[ptr], #6, mul vl]\n\t"
    "ld1w {z7.s}, p1/z, [%[ptr], #7, mul vl]\n\t",
    "x9", "p1", "z0", "z1", "z2", "z3", "z4", "z5", "z6", "z7")
SVE_DEFINE_MEMORY_KERNEL(
    sve_mask_store,
    SVE_MEMORY_HALF_PREDICATE
    "eor z16.d, z16.d, z16.d\n\t",
    "st1w {z16.s}, p1, [%[ptr], #0, mul vl]\n\t"
    "st1w {z16.s}, p1, [%[ptr], #1, mul vl]\n\t"
    "st1w {z16.s}, p1, [%[ptr], #2, mul vl]\n\t"
    "st1w {z16.s}, p1, [%[ptr], #3, mul vl]\n\t"
    "st1w {z16.s}, p1, [%[ptr], #4, mul vl]\n\t"
    "st1w {z16.s}, p1, [%[ptr], #5, mul vl]\n\t"
    "st1w {z16.s}, p1, [%[ptr], #6, mul vl]\n\t"
    "st1w {z16.s}, p1, [%[ptr], #7, mul vl]\n\t",
    "x9", "p1", "z16")

#define SVE_CONV_OFFSET_SETUP(index_shift, advance_shift)                   \
  "cntb x9\n\t"                                                            \
  "lsr x9, x9, #" #index_shift "\n\t"                                     \
  "add x10, x9, x9\n\t"                                                    \
  "add x11, x10, x9\n\t"                                                   \
  "add x12, x11, x9\n\t"                                                   \
  "add x13, x12, x9\n\t"                                                   \
  "add x14, x13, x9\n\t"                                                   \
  "add x15, x14, x9\n\t"                                                   \
  "lsl x16, x9, #" #advance_shift "\n\t"
#define SVE_CONV_ADVANCE                                                    \
  "add %[ptr], %[ptr], x16\n\t"                                            \
  "cmp %[ptr], %[end]\n\t"                                                 \
  "csel %[ptr], %[base], %[ptr], hs\n\t"
#define SVE_DEFINE_CONV_CONTROL(name, shift)                                \
  SVE_DEFINE_MEMORY_KERNEL(                                                  \
      name, SVE_CONV_OFFSET_SETUP(shift, 3), SVE_CONV_ADVANCE, "x9", "x10",\
      "x11", "x12", "x13", "x14", "x15", "x16")
#define SVE_DEFINE_CONV_LOAD(name, op, type, index_shift, advance_shift, address_suffix) \
  SVE_DEFINE_MEMORY_KERNEL(                                                  \
      name,                                                                 \
      "ptrue p0." type "\n\t"                                              \
      SVE_CONV_OFFSET_SETUP(index_shift, advance_shift),                     \
      op " z0." type ", p0/z, [%[ptr]]\n\t"                               \
      op " z1." type ", p0/z, [%[ptr], x9" address_suffix "]\n\t"         \
      op " z2." type ", p0/z, [%[ptr], x10" address_suffix "]\n\t"        \
      op " z3." type ", p0/z, [%[ptr], x11" address_suffix "]\n\t"        \
      op " z4." type ", p0/z, [%[ptr], x12" address_suffix "]\n\t"        \
      op " z5." type ", p0/z, [%[ptr], x13" address_suffix "]\n\t"        \
      op " z6." type ", p0/z, [%[ptr], x14" address_suffix "]\n\t"        \
      op " z7." type ", p0/z, [%[ptr], x15" address_suffix "]\n\t"        \
      SVE_CONV_ADVANCE,                                                      \
      "x9", "x10", "x11", "x12", "x13", "x14", "x15", "x16", "p0",\
      "z0", "z1", "z2", "z3", "z4", "z5", "z6", "z7")
#define SVE_DEFINE_CONV_STORE(name, op, type, index_shift, advance_shift, address_suffix) \
  SVE_DEFINE_MEMORY_KERNEL(                                                  \
      name,                                                                 \
      "ptrue p0." type "\n\t"                                              \
      "dup z16.d, #127\n\t"                                                \
      SVE_CONV_OFFSET_SETUP(index_shift, advance_shift),                     \
      op " z16." type ", p0, [%[ptr]]\n\t"                                \
      op " z16." type ", p0, [%[ptr], x9" address_suffix "]\n\t"          \
      op " z16." type ", p0, [%[ptr], x10" address_suffix "]\n\t"         \
      op " z16." type ", p0, [%[ptr], x11" address_suffix "]\n\t"         \
      op " z16." type ", p0, [%[ptr], x12" address_suffix "]\n\t"         \
      op " z16." type ", p0, [%[ptr], x13" address_suffix "]\n\t"         \
      op " z16." type ", p0, [%[ptr], x14" address_suffix "]\n\t"         \
      op " z16." type ", p0, [%[ptr], x15" address_suffix "]\n\t"         \
      SVE_CONV_ADVANCE,                                                      \
      "x9", "x10", "x11", "x12", "x13", "x14", "x15", "x16", "p0",\
      "z16")

SVE_DEFINE_CONV_CONTROL(sve_conv_control_vl_div_8, 3)
SVE_DEFINE_CONV_CONTROL(sve_conv_control_vl_div_4, 2)
SVE_DEFINE_CONV_CONTROL(sve_conv_control_vl_div_2, 1)
SVE_DEFINE_CONV_LOAD(sve_ld1sb_h_stream, "ld1sb", "h", 1, 3, "")
SVE_DEFINE_CONV_LOAD(sve_ld1b_h_stream, "ld1b", "h", 1, 3, "")
SVE_DEFINE_CONV_LOAD(sve_ld1sb_s_stream, "ld1sb", "s", 2, 3, "")
SVE_DEFINE_CONV_LOAD(sve_ld1b_s_stream, "ld1b", "s", 2, 3, "")
SVE_DEFINE_CONV_LOAD(sve_ld1sb_d_stream, "ld1sb", "d", 3, 3, "")
SVE_DEFINE_CONV_LOAD(sve_ld1b_d_stream, "ld1b", "d", 3, 3, "")
SVE_DEFINE_CONV_LOAD(sve_ld1sh_s_stream, "ld1sh", "s", 2, 4, ", lsl #1")
SVE_DEFINE_CONV_LOAD(sve_ld1h_s_stream, "ld1h", "s", 2, 4, ", lsl #1")
SVE_DEFINE_CONV_LOAD(sve_ld1sh_d_stream, "ld1sh", "d", 3, 4, ", lsl #1")
SVE_DEFINE_CONV_LOAD(sve_ld1h_d_stream, "ld1h", "d", 3, 4, ", lsl #1")
SVE_DEFINE_CONV_LOAD(sve_ld1sw_d_stream, "ld1sw", "d", 3, 5, ", lsl #2")
SVE_DEFINE_CONV_LOAD(sve_ld1w_d_stream, "ld1w", "d", 3, 5, ", lsl #2")
SVE_DEFINE_CONV_STORE(sve_st1b_h_stream, "st1b", "h", 1, 3, "")
SVE_DEFINE_CONV_STORE(sve_st1b_s_stream, "st1b", "s", 2, 3, "")
SVE_DEFINE_CONV_STORE(sve_st1h_s_stream, "st1h", "s", 2, 4, ", lsl #1")
SVE_DEFINE_CONV_STORE(sve_st1b_d_stream, "st1b", "d", 3, 3, "")
SVE_DEFINE_CONV_STORE(sve_st1h_d_stream, "st1h", "d", 3, 4, ", lsl #1")
SVE_DEFINE_CONV_STORE(sve_st1w_d_stream, "st1w", "d", 3, 5, ", lsl #2")

#define SVE_DEFINE_INDEX_CHASE(name, op, type, last_reg, normalize, signed_anchor) \
  extern "C" VECOPS_INST_NOINLINE_USED void name(                           \
      uint64_t loops, void* memory, uint64_t memory_bytes) {                 \
    if (loops == 0 || memory_bytes == 0) return;                             \
    auto* base = static_cast<uint8_t*>(memory);                              \
    auto* ptr = base;                                                        \
    auto* anchor = base + (signed_anchor);                                   \
    __asm__ volatile(                                                        \
        "ptrue p0." type "\n\t"                                             \
        ".p2align 6\n\t"                                                    \
        "1:\n\t"                                                            \
        op " z0." type ", p0/z, [%[ptr]]\n\t"                              \
        "lastb " last_reg ", p0, z0." type "\n\t"                          \
        normalize                                                           \
        "lsl x9, x9, #6\n\t"                                                \
        "add %[ptr], %[anchor], x9\n\t"                                     \
        "subs %[loops], %[loops], #1\n\t"                                   \
        "b.ne 1b\n\t"                                                       \
        : [loops] "+r"(loops), [ptr] "+&r"(ptr)                             \
        : [anchor] "r"(anchor)                                              \
        : "cc", "memory", "x9", "p0", "z0");                              \
  }
#define SVE_INDEX_PAIR(symbol_s, op_s, symbol_u, op_u, type, last_reg, normalize_s) \
  SVE_DEFINE_INDEX_CHASE(symbol_s, op_s, type, last_reg, normalize_s, 4096)  \
  SVE_DEFINE_INDEX_CHASE(symbol_u, op_u, type, last_reg, "", 0)
SVE_INDEX_PAIR(
    sve_ld1sb_h_index, "ld1sb", sve_ld1b_h_index, "ld1b", "h", "w9",
    "sxth x9, w9\n\t")
SVE_INDEX_PAIR(
    sve_ld1sb_s_index, "ld1sb", sve_ld1b_s_index, "ld1b", "s", "w9",
    "sxtw x9, w9\n\t")
SVE_INDEX_PAIR(
    sve_ld1sb_d_index, "ld1sb", sve_ld1b_d_index, "ld1b", "d", "x9", "")
SVE_INDEX_PAIR(
    sve_ld1sh_s_index, "ld1sh", sve_ld1h_s_index, "ld1h", "s", "w9",
    "sxtw x9, w9\n\t")
SVE_INDEX_PAIR(
    sve_ld1sh_d_index, "ld1sh", sve_ld1h_d_index, "ld1h", "d", "x9", "")
SVE_INDEX_PAIR(
    sve_ld1sw_d_index, "ld1sw", sve_ld1w_d_index, "ld1w", "d", "x9", "")

#define SVE_DEFINE_SCALAR_INDEX_CONTROL(name, load_op, signed_anchor)        \
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
        "lsl x9, x9, #6\n\t"                                                \
        "add %[ptr], %[anchor], x9\n\t"                                     \
        "subs %[loops], %[loops], #1\n\t"                                   \
        "b.ne 1b\n\t"                                                       \
        : [loops] "+r"(loops), [ptr] "+&r"(ptr)                             \
        : [anchor] "r"(anchor)                                              \
        : "cc", "memory", "x9");                                           \
  }
SVE_DEFINE_SCALAR_INDEX_CONTROL(
    sve_index_control_s8, "ldrsb x9, [%[ptr]]\n\t", 4096)
SVE_DEFINE_SCALAR_INDEX_CONTROL(
    sve_index_control_u8, "ldrb w9, [%[ptr]]\n\t", 0)
SVE_DEFINE_SCALAR_INDEX_CONTROL(
    sve_index_control_s16, "ldrsh x9, [%[ptr]]\n\t", 4096)
SVE_DEFINE_SCALAR_INDEX_CONTROL(
    sve_index_control_u16, "ldrh w9, [%[ptr]]\n\t", 0)
SVE_DEFINE_SCALAR_INDEX_CONTROL(
    sve_index_control_s32, "ldrsw x9, [%[ptr]]\n\t", 4096)
SVE_DEFINE_SCALAR_INDEX_CONTROL(
    sve_index_control_u32, "ldr w9, [%[ptr]]\n\t", 0)

#define SVE_DEFINE_STORE_FORWARD(name, store_op, load_op, type)              \
  SVE_DEFINE_MEMORY_KERNEL(                                                  \
      name, "ptrue p0." type "\n\tdup z0.d, #127\n\t",                    \
      store_op " z0." type ", p0, [%[ptr]]\n\t"                            \
      load_op " z0." type ", p0/z, [%[ptr]]\n\t", "p0", "z0")
SVE_DEFINE_STORE_FORWARD(sve_st1b_h_forward, "st1b", "ld1b", "h")
SVE_DEFINE_STORE_FORWARD(sve_st1b_s_forward, "st1b", "ld1b", "s")
SVE_DEFINE_STORE_FORWARD(sve_st1h_s_forward, "st1h", "ld1h", "s")
SVE_DEFINE_STORE_FORWARD(sve_st1b_d_forward, "st1b", "ld1b", "d")
SVE_DEFINE_STORE_FORWARD(sve_st1h_d_forward, "st1h", "ld1h", "d")
SVE_DEFINE_STORE_FORWARD(sve_st1w_d_forward, "st1w", "ld1w", "d")

} // namespace

std::span<const InstructionCase> arch_instruction_cases() {
  const uint32_t vector_bits = static_cast<uint32_t>(svcntb() * 8);
  static const InstructionCase cases[] = {
      {"empty_loop", "SVE", "Loop-control baseline", MeasureMode::Control,
       sve_control_empty_loop, 32, 0, 0, vector_bits},
      {"nop_32", "SVE", "Thirty-two architectural NOPs", MeasureMode::Control,
       sve_control_nop_32, 32, 1, 0, vector_bits},
#define SVE_LAT_THR_CASES(label, isa, symbol)                                \
      {label, isa, "Single true-dependency chain", MeasureMode::Latency,     \
       symbol##_latency, 32, 1, 1, vector_bits},                            \
      {label, isa, "Eight independent destination chains",                  \
       MeasureMode::Throughput, symbol##_throughput, 32, 1, 8, vector_bits}
#define SVE_CATEGORY_CASES(label, isa, symbol, category_value)               \
      {label, isa, "Single true-dependency chain", MeasureMode::Latency,     \
       symbol##_latency, 32, 1, 1, vector_bits, category_value},            \
      {label, isa, "Eight independent destination chains",                  \
       MeasureMode::Throughput, symbol##_throughput, 32, 1, 8, vector_bits, \
       category_value}
      SVE_LAT_THR_CASES("fadd_z_s_p0m_z_s", "SVE", sve_fadd_z_s_p0m),
      SVE_LAT_THR_CASES("fmul_z_s_p0m_z_s", "SVE", sve_fmul_z_s_p0m),
      SVE_LAT_THR_CASES("fmla_z_s_p0m_z_s_z_s", "SVE", sve_fmla_z_s_p0m),
      SVE_LAT_THR_CASES("add_z_s_z_s_z_s", "SVE", sve_add_z_s),
      SVE_LAT_THR_CASES("mul_z_s_z_s_z_s", "SVE", sve_mul_z_s),
      SVE_CATEGORY_CASES("and_z_d_z_d_z_d", "SVE", sve_and_z_d,
                         InstructionCategory::Bitwise),
      SVE_CATEGORY_CASES("orr_z_d_z_d_z_d", "SVE", sve_orr_z_d,
                         InstructionCategory::Bitwise),
      SVE_CATEGORY_CASES("eor_z_d_z_d_z_d", "SVE", sve_eor_z_d,
                         InstructionCategory::Bitwise),
      SVE_CATEGORY_CASES("bic_z_d_z_d_z_d", "SVE", sve_bic_z_d,
                         InstructionCategory::Bitwise),
      SVE_CATEGORY_CASES("not_z_d_p0m_z_d", "SVE", sve_not_z_d,
                         InstructionCategory::Bitwise),
#define SVE_SHIFT_CASE(label, symbol)                                        \
      SVE_CATEGORY_CASES(label, "SVE", symbol, InstructionCategory::Shift)
      SVE_SHIFT_CASE("lsl_z_b_p0m_z_b_imm7", sve_lsl_z_b_imm7),
      SVE_SHIFT_CASE("lsl_z_h_p0m_z_h_imm7", sve_lsl_z_h_imm7),
      SVE_SHIFT_CASE("lsl_z_s_p0m_z_s_imm7", sve_lsl_z_s_imm7),
      SVE_SHIFT_CASE("lsl_z_d_p0m_z_d_imm7", sve_lsl_z_d_imm7),
      SVE_SHIFT_CASE("lsr_z_b_p0m_z_b_imm7", sve_lsr_z_b_imm7),
      SVE_SHIFT_CASE("lsr_z_h_p0m_z_h_imm7", sve_lsr_z_h_imm7),
      SVE_SHIFT_CASE("lsr_z_s_p0m_z_s_imm7", sve_lsr_z_s_imm7),
      SVE_SHIFT_CASE("lsr_z_d_p0m_z_d_imm7", sve_lsr_z_d_imm7),
      SVE_SHIFT_CASE("asr_z_b_p0m_z_b_imm7", sve_asr_z_b_imm7),
      SVE_SHIFT_CASE("asr_z_h_p0m_z_h_imm7", sve_asr_z_h_imm7),
      SVE_SHIFT_CASE("asr_z_s_p0m_z_s_imm7", sve_asr_z_s_imm7),
      SVE_SHIFT_CASE("asr_z_d_p0m_z_d_imm7", sve_asr_z_d_imm7),
      SVE_SHIFT_CASE("lsl_z_b_p0m_z_b_z_b", sve_lsl_z_b_z_b),
      SVE_SHIFT_CASE("lsl_z_h_p0m_z_h_z_h", sve_lsl_z_h_z_h),
      SVE_SHIFT_CASE("lsl_z_s_p0m_z_s_z_s", sve_lsl_z_s_z_s),
      SVE_SHIFT_CASE("lsl_z_d_p0m_z_d_z_d", sve_lsl_z_d_z_d),
      SVE_SHIFT_CASE("lsr_z_b_p0m_z_b_z_b", sve_lsr_z_b_z_b),
      SVE_SHIFT_CASE("lsr_z_h_p0m_z_h_z_h", sve_lsr_z_h_z_h),
      SVE_SHIFT_CASE("lsr_z_s_p0m_z_s_z_s", sve_lsr_z_s_z_s),
      SVE_SHIFT_CASE("lsr_z_d_p0m_z_d_z_d", sve_lsr_z_d_z_d),
      SVE_SHIFT_CASE("asr_z_b_p0m_z_b_z_b", sve_asr_z_b_z_b),
      SVE_SHIFT_CASE("asr_z_h_p0m_z_h_z_h", sve_asr_z_h_z_h),
      SVE_SHIFT_CASE("asr_z_s_p0m_z_s_z_s", sve_asr_z_s_z_s),
      SVE_SHIFT_CASE("asr_z_d_p0m_z_d_z_d", sve_asr_z_d_z_d),
      SVE_CATEGORY_CASES("sel_z_b_p1_z_b_z_b", "SVE", sve_sel_z_b,
                         InstructionCategory::Blend),
      SVE_CATEGORY_CASES("sel_z_h_p1_z_h_z_h", "SVE", sve_sel_z_h,
                         InstructionCategory::Blend),
      SVE_CATEGORY_CASES("sel_z_s_p1_z_s_z_s", "SVE", sve_sel_z_s,
                         InstructionCategory::Blend),
      SVE_CATEGORY_CASES("sel_z_d_p1_z_d_z_d", "SVE", sve_sel_z_d,
                         InstructionCategory::Blend),
      SVE_LAT_THR_CASES("zip1_z_s_z_s_z_s", "SVE", sve_zip1_z_s),
      SVE_LAT_THR_CASES("zip2_z_s_z_s_z_s", "SVE", sve_zip2_z_s),
      SVE_LAT_THR_CASES("uzp1_z_s_z_s_z_s", "SVE", sve_uzp1_z_s),
      SVE_LAT_THR_CASES("uzp2_z_s_z_s_z_s", "SVE", sve_uzp2_z_s),
      SVE_LAT_THR_CASES("trn1_z_s_z_s_z_s", "SVE", sve_trn1_z_s),
      SVE_LAT_THR_CASES("trn2_z_s_z_s_z_s", "SVE", sve_trn2_z_s),
      SVE_LAT_THR_CASES("tbl_z_b_z_b_z_b", "SVE", sve_tbl_z_b),
      {"scvtf_z_s_p0m_z_s", "SVE", "Signed i32 to fp32 conversion",
       MeasureMode::Throughput, sve_scvtf_z_s_p0m_throughput, 32, 1, 8, vector_bits},
      {"fcvtzs_z_s_p0m_z_s", "SVE", "Truncating fp32 to signed i32 conversion",
       MeasureMode::Throughput, sve_fcvtzs_z_s_p0m_throughput, 32, 1, 8, vector_bits},
      {"fcvt_z_s_p0m_z_h", "SVE", "Convert even fp16 elements to fp32",
       MeasureMode::Throughput, sve_fcvt_z_s_from_z_h_throughput, 32, 1, 8, vector_bits},
      {"fcvt_z_h_p0m_z_s", "SVE", "Narrow fp32 elements to fp16",
       MeasureMode::Throughput, sve_fcvt_z_h_from_z_s_throughput, 32, 1, 8, vector_bits},
      {"scvtf_z_s+fcvtzs_z_s", "SVE", "Closed signed-i32/fp32 dependency sequence",
       MeasureMode::Latency, sve_scvtf_fcvtzs_latency, 16, 2, 1, vector_bits},
      {"fcvt_z_s_from_z_h+fcvt_z_h_from_z_s", "SVE",
       "Closed fp16/fp32 dependency sequence", MeasureMode::Latency,
       sve_fcvt_hs_fcvt_sh_latency, 16, 2, 1, vector_bits},
#if defined(__ARM_FEATURE_SVE2)
      {"fcvtlt_z_s_p0m_z_h", "SVE2", "Convert odd fp16 elements to fp32",
       MeasureMode::Throughput, sve_fcvtlt_z_s_from_z_h_throughput, 32, 1, 8, vector_bits},
#endif
#if defined(__ARM_FEATURE_BF16_VECTOR_ARITHMETIC)
      {"bfcvt_z_h_p0m_z_s", "SVE+BF16", "Narrow fp32 to bottom bf16 elements",
       MeasureMode::Throughput, sve_bfcvt_z_h_from_z_s_throughput, 32, 1, 8, vector_bits},
#if defined(__ARM_FEATURE_SVE2)
      {"bfcvtnt_z_h_p0m_z_s", "SVE2+BF16", "Narrow fp32 to top bf16 elements",
       MeasureMode::Throughput, sve_bfcvtnt_z_h_from_z_s_throughput, 32, 1, 8, vector_bits},
#endif
#endif
#define SVE_MEMORY_CASE(label, description, mode_value, memory_symbol, seq, inst, chains, \
                        pattern_value, level_value, bytes_value, baseline_value)          \
      {label, "SVE", description, mode_value, nullptr, seq, inst, chains, vector_bits,    \
       InstructionCategory::Memory, memory_symbol, pattern_value, level_value,             \
       bytes_value, baseline_value}
#define SVE_MEMORY_LEVEL(level_label, level_value)                             \
      SVE_MEMORY_CASE("stream_control_" level_label, "Address-generation baseline",       \
                      MeasureMode::Control, sve_stream_control, 8, 0, 0,                   \
                      MemoryPattern::Stream, level_value, 0, "empty_loop"),                \
      SVE_MEMORY_CASE("ld1d_stream_load_" level_label, "Sequential aligned vector load",   \
                      MeasureMode::Throughput, sve_stream_load, 8, 1, 8,                   \
                      MemoryPattern::Stream, level_value, vector_bits / 8,                 \
                      "stream_control_" level_label),                                      \
      SVE_MEMORY_CASE("st1d_stream_store_" level_label, "Sequential aligned vector store", \
                      MeasureMode::Throughput, sve_stream_store, 8, 1, 8,                  \
                      MemoryPattern::Stream, level_value, vector_bits / 8,                 \
                      "stream_control_" level_label),                                      \
      SVE_MEMORY_CASE("ld1d+lastb_pointer_chase_" level_label,                              \
                      "Dependent vector-load cache-latency sequence", MeasureMode::Latency,\
                      sve_pointer_chase, 1, 2, 1, MemoryPattern::PointerChase,              \
                      level_value, vector_bits / 8, "empty_loop")
      SVE_MEMORY_LEVEL("l1", WorkingSetLevel::L1),
      SVE_MEMORY_LEVEL("l2", WorkingSetLevel::L2),
      SVE_MEMORY_LEVEL("l3", WorkingSetLevel::L3),
      SVE_MEMORY_LEVEL("beyond_last_reported_cache", WorkingSetLevel::BeyondLastCache),
      SVE_MEMORY_CASE("st1d+ld1d_store_forward",
                      "L1 store-to-load forwarding round trip", MeasureMode::Latency,
                      sve_store_forward, 1, 2, 1, MemoryPattern::StoreForward,
                      WorkingSetLevel::L1, vector_bits / 8, "empty_loop"),
      SVE_MEMORY_CASE("ld1d_crossline_load", "Cross-cache-line unaligned load",
                      MeasureMode::Throughput, sve_crossline_load, 8, 1, 8,
                      MemoryPattern::Hot, WorkingSetLevel::L1, vector_bits / 8,
                      "empty_loop"),
      SVE_MEMORY_CASE("st1d_crossline_store", "Cross-cache-line unaligned store",
                      MeasureMode::Throughput, sve_crossline_store, 8, 1, 8,
                      MemoryPattern::Hot, WorkingSetLevel::L1, vector_bits / 8,
                      "empty_loop"),
      SVE_MEMORY_CASE("ld1w_p1half_load", "Half-predicate contiguous load",
                      MeasureMode::Throughput, sve_mask_load, 8, 1, 8,
                      MemoryPattern::Hot, WorkingSetLevel::L1, vector_bits / 16,
                      "empty_loop"),
      SVE_MEMORY_CASE("st1w_p1half_store", "Half-predicate contiguous store",
                      MeasureMode::Throughput, sve_mask_store, 8, 1, 8,
                      MemoryPattern::Hot, WorkingSetLevel::L1, vector_bits / 16,
                      "empty_loop"),
#define SVE_CONV_CASE(label, description, mode_value, symbol, seq, inst, chains, pattern, \
                      level, bytes, baseline, kind, src, dst, elems, cap)                  \
      {label, "SVE", description, mode_value, nullptr, seq, inst, chains, vector_bits,    \
       InstructionCategory::Memory, symbol, pattern, level, bytes, baseline, kind, src,    \
       dst, elems, cap}
#define SVE_CONV_CONTROLS(level_label, level_value)                           \
      SVE_CONV_CASE("conv_stream_control_vl_div_8_" level_label,             \
                    "VL/8 conversion-stream address baseline", MeasureMode::Control,\
                    sve_conv_control_vl_div_8, 8, 0, 0, MemoryPattern::Stream, level_value,\
                    0, "empty_loop", MemoryConversionKind::None, 0, 0, 0, 0),             \
      SVE_CONV_CASE("conv_stream_control_vl_div_4_" level_label,             \
                    "VL/4 conversion-stream address baseline", MeasureMode::Control,\
                    sve_conv_control_vl_div_4, 8, 0, 0, MemoryPattern::Stream, level_value,\
                    0, "empty_loop", MemoryConversionKind::None, 0, 0, 0, 0),             \
      SVE_CONV_CASE("conv_stream_control_vl_div_2_" level_label,             \
                    "VL/2 conversion-stream address baseline", MeasureMode::Control,\
                    sve_conv_control_vl_div_2, 8, 0, 0, MemoryPattern::Stream, level_value,\
                    0, "empty_loop", MemoryConversionKind::None, 0, 0, 0, 0)
      SVE_CONV_CONTROLS("l1", WorkingSetLevel::L1),
      SVE_CONV_CONTROLS("l2", WorkingSetLevel::L2),
      SVE_CONV_CONTROLS("l3", WorkingSetLevel::L3),
      SVE_CONV_CONTROLS(
          "beyond_last_reported_cache", WorkingSetLevel::BeyondLastCache),
#define SVE_CONV_STREAM_LEVEL(label, description, symbol, kind, src, dst, bytes, elems, \
                              ratio_tag, level_label, level_value)                       \
      SVE_CONV_CASE(label "_stream_" level_label, description, MeasureMode::Throughput, \
                    symbol, 8, 1, 8, MemoryPattern::Stream, level_value, bytes,           \
                    "conv_stream_control_" ratio_tag "_" level_label, kind, src, dst,    \
                    elems, 0)
#define SVE_CONV_STREAM_LEVELS(label, description, symbol, kind, src, dst, bytes, elems, ratio_tag) \
      SVE_CONV_STREAM_LEVEL(label, description, symbol, kind, src, dst, bytes, elems, ratio_tag,\
                            "l1", WorkingSetLevel::L1),                                  \
      SVE_CONV_STREAM_LEVEL(label, description, symbol, kind, src, dst, bytes, elems, ratio_tag,\
                            "l2", WorkingSetLevel::L2),                                  \
      SVE_CONV_STREAM_LEVEL(label, description, symbol, kind, src, dst, bytes, elems, ratio_tag,\
                            "l3", WorkingSetLevel::L3),                                  \
      SVE_CONV_STREAM_LEVEL(label, description, symbol, kind, src, dst, bytes, elems, ratio_tag,\
                            "beyond_last_reported_cache",                               \
                            WorkingSetLevel::BeyondLastCache)
#define SVE_EXTEND_LEVELS(label_s, label_u, symbol_s, symbol_u, src, dst, bytes, elems, ratio_tag) \
      SVE_CONV_STREAM_LEVELS(label_s, "Sequential SVE sign-extending load", symbol_s,     \
                             MemoryConversionKind::SignExtendLoad, src, dst, bytes, elems,\
                             ratio_tag),                                                  \
      SVE_CONV_STREAM_LEVELS(label_u, "Sequential SVE zero-extending load (ACLE svld1u*)",\
                             symbol_u, MemoryConversionKind::ZeroExtendLoad, src, dst,    \
                             bytes, elems, ratio_tag)
      SVE_EXTEND_LEVELS(
          "ld1sb_z_h", "ld1b_z_h", sve_ld1sb_h_stream, sve_ld1b_h_stream,
          8, 16, vector_bits / 16, vector_bits / 16, "vl_div_2"),
      SVE_EXTEND_LEVELS(
          "ld1sb_z_s", "ld1b_z_s", sve_ld1sb_s_stream, sve_ld1b_s_stream,
          8, 32, vector_bits / 32, vector_bits / 32, "vl_div_4"),
      SVE_EXTEND_LEVELS(
          "ld1sb_z_d", "ld1b_z_d", sve_ld1sb_d_stream, sve_ld1b_d_stream,
          8, 64, vector_bits / 64, vector_bits / 64, "vl_div_8"),
      SVE_EXTEND_LEVELS(
          "ld1sh_z_s", "ld1h_z_s", sve_ld1sh_s_stream, sve_ld1h_s_stream,
          16, 32, vector_bits / 16, vector_bits / 32, "vl_div_2"),
      SVE_EXTEND_LEVELS(
          "ld1sh_z_d", "ld1h_z_d", sve_ld1sh_d_stream, sve_ld1h_d_stream,
          16, 64, vector_bits / 32, vector_bits / 64, "vl_div_4"),
      SVE_EXTEND_LEVELS(
          "ld1sw_z_d", "ld1w_z_d", sve_ld1sw_d_stream, sve_ld1w_d_stream,
          32, 64, vector_bits / 16, vector_bits / 64, "vl_div_2"),
#define SVE_STORE_LEVELS(label, symbol, src, dst, bytes, elems, ratio_tag)    \
      SVE_CONV_STREAM_LEVELS(label, "Sequential SVE truncating narrow store", symbol,    \
                             MemoryConversionKind::TruncateStore, src, dst, bytes, elems,\
                             ratio_tag)
      SVE_STORE_LEVELS(
          "st1b_z_h", sve_st1b_h_stream, 16, 8, vector_bits / 16,
          vector_bits / 16, "vl_div_2"),
      SVE_STORE_LEVELS(
          "st1b_z_s", sve_st1b_s_stream, 32, 8, vector_bits / 32,
          vector_bits / 32, "vl_div_4"),
      SVE_STORE_LEVELS(
          "st1h_z_s", sve_st1h_s_stream, 32, 16, vector_bits / 16,
          vector_bits / 32, "vl_div_2"),
      SVE_STORE_LEVELS(
          "st1b_z_d", sve_st1b_d_stream, 64, 8, vector_bits / 64,
          vector_bits / 64, "vl_div_8"),
      SVE_STORE_LEVELS(
          "st1h_z_d", sve_st1h_d_stream, 64, 16, vector_bits / 32,
          vector_bits / 64, "vl_div_4"),
      SVE_STORE_LEVELS(
          "st1w_z_d", sve_st1w_d_stream, 64, 32, vector_bits / 16,
          vector_bits / 64, "vl_div_2"),
#define SVE_INDEX_CONTROL(label, symbol, kind, src)                           \
      SVE_CONV_CASE(label, "Matched scalar extending-load index-chase control",\
                    MeasureMode::Control, symbol, 1, 0, 1,                   \
                    MemoryPattern::PointerChase, WorkingSetLevel::L1, 0,     \
                    "empty_loop", kind, src, src, 0, 8192)
      SVE_INDEX_CONTROL(
          "conv_index_control_s8", sve_index_control_s8,
          MemoryConversionKind::SignExtendLoad, 8),
      SVE_INDEX_CONTROL(
          "conv_index_control_u8", sve_index_control_u8,
          MemoryConversionKind::ZeroExtendLoad, 8),
      SVE_INDEX_CONTROL(
          "conv_index_control_s16", sve_index_control_s16,
          MemoryConversionKind::SignExtendLoad, 16),
      SVE_INDEX_CONTROL(
          "conv_index_control_u16", sve_index_control_u16,
          MemoryConversionKind::ZeroExtendLoad, 16),
      SVE_INDEX_CONTROL(
          "conv_index_control_s32", sve_index_control_s32,
          MemoryConversionKind::SignExtendLoad, 32),
      SVE_INDEX_CONTROL(
          "conv_index_control_u32", sve_index_control_u32,
          MemoryConversionKind::ZeroExtendLoad, 32),
#define SVE_EXTEND_LAT(label, symbol, kind, src, dst, bytes, elems, baseline) \
      SVE_CONV_CASE(label "_index_chase_l1", "L1 dependent extending-load index chase",\
                    MeasureMode::Latency, symbol, 1, 5, 1, MemoryPattern::PointerChase,   \
                    WorkingSetLevel::L1, bytes, baseline, kind, src, dst, elems, 8192)
      SVE_EXTEND_LAT(
          "ld1sb_z_h", sve_ld1sb_h_index, MemoryConversionKind::SignExtendLoad,
          8, 16, vector_bits / 16, vector_bits / 16, "conv_index_control_s8"),
      SVE_EXTEND_LAT(
          "ld1b_z_h", sve_ld1b_h_index, MemoryConversionKind::ZeroExtendLoad,
          8, 16, vector_bits / 16, vector_bits / 16, "conv_index_control_u8"),
      SVE_EXTEND_LAT(
          "ld1sb_z_s", sve_ld1sb_s_index, MemoryConversionKind::SignExtendLoad,
          8, 32, vector_bits / 32, vector_bits / 32, "conv_index_control_s8"),
      SVE_EXTEND_LAT(
          "ld1b_z_s", sve_ld1b_s_index, MemoryConversionKind::ZeroExtendLoad,
          8, 32, vector_bits / 32, vector_bits / 32, "conv_index_control_u8"),
      SVE_EXTEND_LAT(
          "ld1sb_z_d", sve_ld1sb_d_index, MemoryConversionKind::SignExtendLoad,
          8, 64, vector_bits / 64, vector_bits / 64, "conv_index_control_s8"),
      SVE_EXTEND_LAT(
          "ld1b_z_d", sve_ld1b_d_index, MemoryConversionKind::ZeroExtendLoad,
          8, 64, vector_bits / 64, vector_bits / 64, "conv_index_control_u8"),
      SVE_EXTEND_LAT(
          "ld1sh_z_s", sve_ld1sh_s_index, MemoryConversionKind::SignExtendLoad,
          16, 32, vector_bits / 16, vector_bits / 32, "conv_index_control_s16"),
      SVE_EXTEND_LAT(
          "ld1h_z_s", sve_ld1h_s_index, MemoryConversionKind::ZeroExtendLoad,
          16, 32, vector_bits / 16, vector_bits / 32, "conv_index_control_u16"),
      SVE_EXTEND_LAT(
          "ld1sh_z_d", sve_ld1sh_d_index, MemoryConversionKind::SignExtendLoad,
          16, 64, vector_bits / 32, vector_bits / 64, "conv_index_control_s16"),
      SVE_EXTEND_LAT(
          "ld1h_z_d", sve_ld1h_d_index, MemoryConversionKind::ZeroExtendLoad,
          16, 64, vector_bits / 32, vector_bits / 64, "conv_index_control_u16"),
      SVE_EXTEND_LAT(
          "ld1sw_z_d", sve_ld1sw_d_index, MemoryConversionKind::SignExtendLoad,
          32, 64, vector_bits / 16, vector_bits / 64, "conv_index_control_s32"),
      SVE_EXTEND_LAT(
          "ld1w_z_d", sve_ld1w_d_index, MemoryConversionKind::ZeroExtendLoad,
          32, 64, vector_bits / 16, vector_bits / 64, "conv_index_control_u32"),
#define SVE_STORE_LAT(label, symbol, src, dst, bytes, elems)                  \
      SVE_CONV_CASE(label "_store_forward_l1",                              \
                    "L1 dependent truncating-store plus extending-load forwarding round trip",\
                    MeasureMode::Latency, symbol, 1, 2, 1, MemoryPattern::StoreForward,   \
                    WorkingSetLevel::L1, bytes, "empty_loop",                       \
                    MemoryConversionKind::TruncateStore, src, dst, elems, 0)
      SVE_STORE_LAT(
          "st1b_z_h", sve_st1b_h_forward, 16, 8, vector_bits / 16,
          vector_bits / 16),
      SVE_STORE_LAT(
          "st1b_z_s", sve_st1b_s_forward, 32, 8, vector_bits / 32,
          vector_bits / 32),
      SVE_STORE_LAT(
          "st1h_z_s", sve_st1h_s_forward, 32, 16, vector_bits / 16,
          vector_bits / 32),
      SVE_STORE_LAT(
          "st1b_z_d", sve_st1b_d_forward, 64, 8, vector_bits / 64,
          vector_bits / 64),
      SVE_STORE_LAT(
          "st1h_z_d", sve_st1h_d_forward, 64, 16, vector_bits / 32,
          vector_bits / 64),
      SVE_STORE_LAT(
          "st1w_z_d", sve_st1w_d_forward, 64, 32, vector_bits / 16,
          vector_bits / 64),
  };
  return cases;
}

} // namespace vecops::bench::inst
