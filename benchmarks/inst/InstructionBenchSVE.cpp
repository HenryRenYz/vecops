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
  };
  return cases;
}

} // namespace vecops::bench::inst
