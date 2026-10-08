// SPDX-FileCopyrightText: 2026 HenryRenYz and vecops contributors
// SPDX-License-Identifier: MIT
//
// Cross-library BF16 GEMM workload catalog.
//

#pragma once

#include <array>
#include <cstdint>
#include <string_view>

#include "vecops/CoreTypes.h"

namespace vecops::bench::matmul_otherlibs {

enum class Model : std::uint8_t { General, LLM, AF3 };
enum class Operation : std::uint8_t {
  Gemm,
  GemmAdd,
  Bias,
  BiasRelu,
  BiasSilu,
};
enum class Orientation : std::uint8_t { RowContiguous, ColumnStrided };

inline constexpr std::uint32_t op_bit(Operation op) {
  return 1u << static_cast<unsigned>(op);
}

inline constexpr std::uint32_t GemmOps =
    op_bit(Operation::Gemm) | op_bit(Operation::GemmAdd);
inline constexpr std::uint32_t ProjectionOps =
    GemmOps | op_bit(Operation::Bias) | op_bit(Operation::BiasRelu);
inline constexpr std::uint32_t MlpOps =
    ProjectionOps | op_bit(Operation::BiasSilu);

struct Case {
  std::string_view id;
  Model model;
  std::string_view subgraph;
  // Unified semantic notation: Y[N,M] = X[N,K] * W[M,K]^T.
  nint_t m;
  nint_t n;
  nint_t k;
  nint_t batch;
  nint_t num_tokens;
  nint_t chunk_size;
  nint_t call_multiplicity;
  std::uint32_t operations;
  Orientation orientation = Orientation::RowContiguous;
};

// General and LLM cases are selected directly from MatmulBenchCommon.h.
inline constexpr std::array CoreCases{
    Case{"G01_square", Model::General, "square", 256, 256, 256, 1, 0, 0, 1, GemmOps},
    Case{"G02_ragged", Model::General, "ragged", 1025, 127, 769, 1, 0, 0, 1, GemmOps},
    Case{"G03_conv1x1", Model::General, "conv1x1", 64, 3136, 576, 1, 0, 0, 1, GemmOps},
    Case{"G04_rank_reduce", Model::General, "rank_reduce", 64, 1024, 4096, 1, 0, 0, 1, GemmOps},
    Case{"G05_rank_expand", Model::General, "rank_expand", 1024, 64, 4096, 1, 0, 0, 1, GemmOps},
    Case{"L01_qkv_decode", Model::LLM, "qkv", 4096, 1, 4096, 1, 1, 0, 1, ProjectionOps},
    Case{"L02_qkv_batch4", Model::LLM, "qkv", 4096, 4, 4096, 1, 4, 0, 1, ProjectionOps},
    Case{"L03_qkv_prefill", Model::LLM, "qkv", 4096, 128, 4096, 1, 128, 0, 1, ProjectionOps},
    Case{"L04_mlp_up_decode", Model::LLM, "mlp_up", 11008, 1, 4096, 1, 1, 0, 1, MlpOps},
    Case{"L05_mlp_up_prefill", Model::LLM, "mlp_up", 11008, 128, 4096, 1, 128, 0, 1, MlpOps},
    Case{"L06_mlp_down_decode", Model::LLM, "mlp_down", 4096, 1, 11008, 1, 1, 0, 1, ProjectionOps},
    Case{"L07_mlp_down_prefill", Model::LLM, "mlp_down", 4096, 128, 11008, 1, 128, 0, 1, ProjectionOps},
    Case{"L08_attention", Model::LLM, "attention", 128, 128, 128, 1, 128, 0, 1, GemmOps},
};

// AlphaFold 3 v3.0.2 GridSelfAttention: pair channel 128, 4 heads,
// head dimension 32. C is the inference chunk size selected by AF3.
inline constexpr std::array AF3Cases{
    Case{"A01_gsa_proj_r128", Model::AF3, "gsa_projection", 128, 128 * 128, 128, 1, 128, 128, 5, ProjectionOps},
    Case{"A02_gsa_proj_r512", Model::AF3, "gsa_projection", 128, 128 * 512, 128, 1, 512, 128, 20, ProjectionOps},
    Case{"A03_gsa_proj_r1536", Model::AF3, "gsa_projection", 128, 128 * 1536, 128, 1, 1536, 128, 60, ProjectionOps},
    Case{"A04_gsa_proj_r2048", Model::AF3, "gsa_projection", 128, 32 * 2048, 128, 1, 2048, 32, 320, ProjectionOps},
    Case{"A05_pair_bias_r128", Model::AF3, "pair_bias_projection", 4, 128 * 128, 128, 1, 128, 0, 1, GemmOps},
    Case{"A06_pair_bias_r512", Model::AF3, "pair_bias_projection", 4, 512 * 512, 128, 1, 512, 0, 1, GemmOps},
    Case{"A07_pair_bias_r1536", Model::AF3, "pair_bias_projection", 4, 1536 * 1536, 128, 1, 1536, 0, 1, GemmOps},
    Case{"A08_gsa_qk_r128", Model::AF3, "gsa_qk", 128, 128, 32, 4, 128, 128, 128, GemmOps},
    Case{"A09_gsa_qk_r512", Model::AF3, "gsa_qk", 512, 512, 32, 4, 512, 128, 512, GemmOps},
    Case{"A10_gsa_qk_r1536", Model::AF3, "gsa_qk", 1536, 1536, 32, 4, 1536, 128, 1536, GemmOps},
    Case{"A11_gsa_qk_r2048", Model::AF3, "gsa_qk", 2048, 2048, 32, 4, 2048, 32, 2048, GemmOps},
    Case{"A12_gsa_pv_r128", Model::AF3, "gsa_pv", 32, 128, 128, 4, 128, 128, 128, GemmOps},
    Case{"A13_gsa_pv_r512", Model::AF3, "gsa_pv", 32, 512, 512, 4, 512, 128, 512, GemmOps},
    Case{"A14_gsa_pv_r1536", Model::AF3, "gsa_pv", 32, 1536, 1536, 4, 1536, 128, 1536, GemmOps},
    Case{"A15_gsa_pv_r2048", Model::AF3, "gsa_pv", 32, 2048, 2048, 4, 2048, 32, 2048, GemmOps},
    Case{"A16_tri_mul_r128", Model::AF3, "triangle_multiplication", 128, 128, 128, 128, 128, 0, 2, GemmOps},
    Case{"A17_tri_mul_r512", Model::AF3, "triangle_multiplication", 512, 512, 512, 128, 512, 0, 2, GemmOps},
    Case{"A18_tri_mul_r1536_ch1", Model::AF3, "triangle_multiplication", 1536, 1536, 1536, 1, 1536, 0, 256, GemmOps},
};

inline constexpr const char* model_name(Model model) {
  switch (model) {
    case Model::General: return "general";
    case Model::LLM: return "llm";
    case Model::AF3: return "af3";
  }
  return "unknown";
}

inline constexpr const char* operation_name(Operation op) {
  switch (op) {
    case Operation::Gemm: return "gemm";
    case Operation::GemmAdd: return "gemm_add";
    case Operation::Bias: return "bias";
    case Operation::BiasRelu: return "bias_relu";
    case Operation::BiasSilu: return "bias_silu";
  }
  return "unknown";
}

inline constexpr const char* orientation_name(Orientation orientation) {
  return orientation == Orientation::RowContiguous
             ? "row_contiguous"
             : "column_strided";
}

} // namespace vecops::bench::matmul_otherlibs
