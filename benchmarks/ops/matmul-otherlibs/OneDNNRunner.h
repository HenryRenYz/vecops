// SPDX-FileCopyrightText: 2026 HenryRenYz and vecops contributors
// SPDX-License-Identifier: MIT
// Shared oneDNN execution adapter for the raw-E2E suite and phase probes.
// The public weights view is [batch, M, K]; oneDNN consumes its logical
// transpose [batch, K, M] through an opaque implementation-selected layout.

#pragma once

#include "ProviderCommon.h"

#include <oneapi/dnnl/dnnl.hpp>

#include <unordered_map>

namespace vecops::bench::matmul_otherlibs {

class OneDNNRunner {
public:
  OneDNNRunner(const Case& c, Buffers& buffers, Operation op)
      : c_(c), buffers_(buffers), op_(op),
        engine_(dnnl::engine::kind::cpu, 0), stream_(engine_) {
    using dt = dnnl::memory::data_type;
    const bool rank2 = c.batch == 1;
    const dnnl::memory::dims src_dims = rank2
        ? dnnl::memory::dims{c.n, c.k}
        : dnnl::memory::dims{c.batch, c.n, c.k};
    const dnnl::memory::dims weights_dims = rank2
        ? dnnl::memory::dims{c.k, c.m}
        : dnnl::memory::dims{c.batch, c.k, c.m};
    const dnnl::memory::dims dst_dims = rank2
        ? dnnl::memory::dims{c.n, c.m}
        : dnnl::memory::dims{c.batch, c.n, c.m};
    const dnnl::memory::dims src_strides = rank2
        ? dnnl::memory::dims{c.k, 1}
        : dnnl::memory::dims{c.n * c.k, c.k, 1};
    // Physical W is [batch,M,K], exposed logically as [batch,K,M].
    const dnnl::memory::dims weights_strides = rank2
        ? dnnl::memory::dims{1, c.k}
        : dnnl::memory::dims{c.m * c.k, 1, c.k};
    const dnnl::memory::dims dst_strides = rank2
        ? dnnl::memory::dims{c.m, 1}
        : dnnl::memory::dims{c.n * c.m, c.m, 1};
    const dnnl::memory::desc src_md(src_dims, dt::bf16, src_strides);
    const dnnl::memory::desc user_weights_md(
        weights_dims, dt::bf16, weights_strides);
    const dnnl::memory::desc weights_md(
        weights_dims, dt::bf16, dnnl::memory::format_tag::any);
    const dnnl::memory::desc dst_md(dst_dims, dt::f32, dst_strides);

    dnnl::primitive_attr attr;
    attr.set_scratchpad_mode(dnnl::scratchpad_mode::user);
    if (op == Operation::GemmAdd) {
      dnnl::post_ops post_ops;
      post_ops.append_sum(1.0f);
      attr.set_post_ops(post_ops);
    }

    const bool has_bias = op == Operation::Bias ||
                          op == Operation::BiasRelu ||
                          op == Operation::BiasSilu;
    if (has_bias) {
      const dnnl::memory::dims bias_dims = rank2
          ? dnnl::memory::dims{1, c.m}
          : dnnl::memory::dims{1, 1, c.m};
      const dnnl::memory::dims bias_strides = rank2
          ? dnnl::memory::dims{c.m, 1}
          : dnnl::memory::dims{c.m, c.m, 1};
      const dnnl::memory::desc bias_md(bias_dims, dt::f32, bias_strides);
      pd_ = dnnl::matmul::primitive_desc(
          engine_, src_md, weights_md, bias_md, dst_md, attr);
      bias_mem_ = dnnl::memory(bias_md, engine_, buffers.bias.data());
    } else {
      pd_ = dnnl::matmul::primitive_desc(
          engine_, src_md, weights_md, dst_md, attr);
    }
    primitive_ = dnnl::matmul(pd_);
    src_mem_ = dnnl::memory(src_md, engine_, buffers.x.data());
    user_weights_mem_ = dnnl::memory(
        user_weights_md, engine_, buffers.weight.data());
    weights_mem_ = dnnl::memory(pd_.weights_desc(), engine_);
    dst_mem_ = dnnl::memory(dst_md, engine_, buffers.output.data());
    if (pd_.scratchpad_desc().get_size() != 0) {
      scratchpad_mem_ = dnnl::memory(pd_.scratchpad_desc(), engine_);
    }
  }

  // Performs the data-dependent blocked-weight transform selected by oneDNN.
  // It is intentionally exposed so phase probes can separate it from compute.
  void prepare_weights() {
    dnnl::reorder(user_weights_mem_, weights_mem_).execute(
        stream_, user_weights_mem_, weights_mem_);
    stream_.wait();
  }

  // Executes using the last prepared opaque weights.  Call prepare_weights()
  // first after constructing the runner or changing the public weight buffer.
  void execute_prepared() {
    std::unordered_map<int, dnnl::memory> args{
        {DNNL_ARG_SRC, src_mem_},
        {DNNL_ARG_WEIGHTS, weights_mem_},
        {DNNL_ARG_DST, dst_mem_},
    };
    if (op_ == Operation::Bias || op_ == Operation::BiasRelu ||
        op_ == Operation::BiasSilu) {
      args.emplace(DNNL_ARG_BIAS, bias_mem_);
    }
    if (pd_.scratchpad_desc().get_size() != 0) {
      args.emplace(DNNL_ARG_SCRATCHPAD, scratchpad_mem_);
    }
    primitive_.execute(stream_, args);
    stream_.wait();
    if (op_ == Operation::BiasRelu) {
      for (float& value : buffers_.output) value = std::max(value, 0.0f);
    } else if (op_ == Operation::BiasSilu) {
      for (float& value : buffers_.output) {
        value = value / (1.0f + std::exp(-value));
      }
    }
  }

  // Raw E2E means public row-major weights are transformed for each call.
  void operator()() {
    prepare_weights();
    execute_prepared();
  }

  const char* implementation() const { return pd_.impl_info_str(); }

  /// Whether oneDNN selected a physical weights layout distinct from the
  /// caller's strided public view.  A reorder primitive is still submitted
  /// for a matching descriptor, but it is a no-op rather than weight packing.
  bool weights_reorder_required() const {
    return user_weights_mem_.get_desc() != weights_mem_.get_desc();
  }

  std::size_t prepared_weight_bytes() const {
    return weights_mem_.get_desc().get_size();
  }

private:
  const Case& c_;
  Buffers& buffers_;
  Operation op_;
  dnnl::engine engine_;
  dnnl::stream stream_;
  dnnl::matmul::primitive_desc pd_;
  dnnl::matmul primitive_;
  dnnl::memory src_mem_;
  dnnl::memory user_weights_mem_;
  dnnl::memory weights_mem_;
  dnnl::memory dst_mem_;
  dnnl::memory bias_mem_;
  dnnl::memory scratchpad_mem_;
};

} // namespace vecops::bench::matmul_otherlibs
