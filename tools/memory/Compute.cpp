// SPDX-FileCopyrightText: 2026 HenryRenYz and vecops contributors
// SPDX-License-Identifier: MIT
// Shared global tensors, disjoint output-row ownership, native vecops operators.
#include "Bench.h"
#include "vecops/ops/Matmul.h"
#include "vecops/ops/LayerNorm.h"
#include <algorithm>
#include <atomic>
#include <cmath>
#include <condition_variable>
#include <exception>
#include <functional>
#include <iomanip>
#include <iostream>
#include <limits>
#include <memory>
#include <mutex>
#include <numeric>
#include <set>
#include <thread>
#include <sys/resource.h>
#include <sched.h>
#if defined(ARCH_X86_FAMILY)
#include <sys/syscall.h>
#include <unistd.h>
#endif

namespace memory_bench {
namespace {
using namespace vecops;
using namespace vecops::tensor;
#if defined(HAS_AMX_BF16)
using GemmAtom = matmul::AMX_BF16F32;
constexpr const char* gemm_backend = "AMX_BF16F32";
#elif defined(HAS_SME)
using GemmAtom = matmul::SME_BF16F32;
constexpr const char* gemm_backend = "SME_BF16F32";
#endif
float value(std::size_t i, std::uint64_t salt) {
  std::uint64_t x = i + salt;
  x = (x ^ (x >> 30)) * 0xbf58476d1ce4e5b9ULL;
  x = (x ^ (x >> 27)) * 0x94d049bb133111ebULL;
  return (static_cast<int>((x ^ (x >> 31)) % 17) - 8) * 0.125f;
}
std::size_t add_size(std::size_t a, std::size_t b) {
  if (a > std::numeric_limits<std::size_t>::max() - b) throw std::runtime_error("Compute size overflow");
  return a + b;
}
struct Gate {
  explicit Gate(std::size_t n) : count(n), left(n) {}
  void wait() {
    std::unique_lock lock(mutex); auto old = generation;
    if (--left == 0) { left = count; ++generation; lock.unlock(); cv.notify_all(); }
    else cv.wait(lock, [&] { return generation != old; });
  }
  std::size_t count, left, generation = 0;
  std::mutex mutex; std::condition_variable cv;
};
struct Worker {
  int cpu;
  std::size_t begin, end;
  std::unique_ptr<Buffer> scratch;
  std::function<void()> call;
  std::exception_ptr error;
  std::uint64_t start = 0, finish = 0, minor = 0, major = 0, switches = 0;
  int end_cpu = -1;
};
class Team {
 public:
  Team(const std::vector<int>& cpus, std::size_t rows, const std::function<void(Worker&)>& setup)
      : gate(cpus.size() + 1) {
    for (std::size_t i = 0; i < cpus.size(); ++i) {
      auto w = std::make_unique<Worker>(); w->cpu = cpus[i];
      // Quotient/remainder-free bounds also cover unequal row counts exactly.
      w->begin = rows * i / cpus.size(); w->end = rows * (i + 1) / cpus.size();
      workers.push_back(std::move(w));
    }
    try {
      for (auto& w : workers) threads.emplace_back([&, p=w.get()] {
        while (!launched.load(std::memory_order_acquire)) relax();
        if (cancel.load()) return;
        try { pin(p->cpu); setup(*p); } catch (...) { p->error = std::current_exception(); }
        gate.wait();
        for (;;) {
          gate.wait(); if (stopping) break;
          try {
            rusage before{}, after{}; completion_fence();
            getrusage(RUSAGE_THREAD, &before);
            while (now_ns() < start) relax();
            p->start = now_ns();
            for (std::uint64_t i = 0; i < repeats; ++i) p->call();
            completion_fence(); p->finish = now_ns(); p->end_cpu = sched_getcpu();
            getrusage(RUSAGE_THREAD, &after);
            p->minor = after.ru_minflt - before.ru_minflt;
            p->major = after.ru_majflt - before.ru_majflt;
            p->switches = after.ru_nvcsw - before.ru_nvcsw + after.ru_nivcsw - before.ru_nivcsw;
          } catch (...) { p->error = std::current_exception(); }
          gate.wait();
        }
      });
    } catch (...) {
      cancel = true; launched.store(true, std::memory_order_release);
      for (auto& t : threads) t.join();
      throw;
    }
    launched.store(true, std::memory_order_release); gate.wait();
    try { check(); } catch (...) { stop(); throw; }
  }
  ~Team() { stop(); }
  void check() const { for (auto& w : workers) if (w->error) std::rethrow_exception(w->error); }
  void phase(std::uint64_t n) {
    repeats = n; start = now_ns() + std::max<std::uint64_t>(5000000, workers.size()*100000);
    gate.wait(); gate.wait(); check();
  }
  void stop() {
    if (stopping) return;
    stopping = true; gate.wait(); for (auto& t : threads) t.join();
  }
  std::vector<std::unique_ptr<Worker>> workers;
 private:
  Gate gate; std::vector<std::thread> threads;
  std::atomic<bool> launched{false}, cancel{false}; bool stopping = false;
  std::uint64_t repeats = 0, start = 0;
};
std::vector<int> pool(const std::vector<int>& explicit_nodes, int fallback, const Options& o, const Topology& t) {
  auto nodes = explicit_nodes.empty() ? std::vector<int>{fallback} : explicit_nodes;
  std::sort(nodes.begin(), nodes.end()); nodes.erase(std::unique(nodes.begin(), nodes.end()), nodes.end());
  if (nodes.size() > 1 && !o.interleave) throw std::runtime_error("Multiple compute memory nodes require --interleave");
  for (int n : nodes) if (!t.node(n).memory_allowed) throw std::runtime_error("Compute memory node excluded by cpuset");
  return nodes;
}
void sum_pages(std::map<int,std::size_t>& dst, const Buffer& b) {
  for (auto [n,count] : b.page_counts()) dst[n] += count;
}
} // namespace

void run_compute(const Options& o, const Topology& t, Reporter& reporter) {
  const bool gemm = o.operations.front() == "gemm";
  const auto rows = o.shape[0], width = o.shape[1], reduction = gemm ? o.shape[2] : width;
  const auto output_elements = checked_mul(rows, width);
  const auto a_elements = checked_mul(rows, reduction), b_elements = gemm ? checked_mul(width, reduction) : 0;
  const auto element_bytes = gemm ? 2UL : 4UL;
  const auto a_bytes = checked_mul(a_elements, element_bytes), b_bytes = checked_mul(b_elements, element_bytes);
  const auto c_bytes = checked_mul(output_elements, element_bytes);
  for (auto dim : o.shape) if (dim > static_cast<std::size_t>(std::numeric_limits<nint_t>::max())) throw std::runtime_error("Shape exceeds operator index range");
  const auto param_bytes = !gemm && o.ln_affine ? checked_mul(width, sizeof(float)) : 0;
  int config = 0;
  for (const auto& cpu_spec : split(o.cpu, ';')) {
    const auto mapping = groups(o, t, cpu_list(cpu_spec));
    if (mapping.size() != 1) throw std::runtime_error("compute requires one execution group; use explicit input/output pools");
    const auto& g = mapping.front();
    if (rows < g.cpus.size()) throw std::runtime_error("Every compute worker must own at least one output row");
    const auto src = pool(o.input_nodes, g.route.src, o, t), dst = pool(o.output_nodes, g.route.dst, o, t);
    const int scratch_node = o.workspace_node < 0 ? g.exec : o.workspace_node;
    if (!t.node(scratch_node).memory_allowed) throw std::runtime_error("Workspace node excluded by cpuset");
    Row base; base.kind = "compute"; base.operation = o.operations.front(); base.kernel = "vecops-operator";
    base.row_end = rows; base.config = ++config; base.threads = g.cpus.size(); base.cpus = join_cpus(g.cpus);
    base.src = g.route.src; base.dst = g.route.dst; base.exec = g.exec; base.group = 0; base.scope = "total";
    for (auto d : o.shape) base.shape += (base.shape.empty() ? "" : ",") + std::to_string(d);
    base.dtype = gemm ? "bf16,bf16->bf16;acc=fp32" : "fp32->fp32";
    base.compute_backend = gemm ? "unavailable" : std::string(vector_backend()) + ":LayerNorm";
#if defined(HAS_AMX_BF16) || defined(HAS_SME)
    if (gemm) base.compute_backend = std::string(gemm_backend) + ":" + std::string(ops::Matmul<ops::MatmulConfig<GemmAtom>>::kernel_family_name());
#else
    if (gemm) throw std::runtime_error("BF16 GEMM requires a native AMX or SME target");
#endif
    base.access = "operator-native"; base.load_policy = base.store_policy = "operator-native";
    base.source_nodes = src; base.destination_nodes = dst;
    base.workspace_nodes = std::to_string(scratch_node);
    base.tensor_input_bytes = add_size(add_size(a_bytes, b_bytes), checked_mul(param_bytes, 2));
    base.tensor_output_bytes = c_bytes; base.working_bytes = add_size(base.tensor_input_bytes, c_bytes);
    base.vector_bytes = native_vector_bytes(); base.page_policy = o.pages;
    std::cout << "\n  Compute " << base.operation << " shape=" << base.shape << " workers=" << base.cpus
              << " input=[" << join_cpus(src) << "] output=[" << join_cpus(dst)
              << "] scratch=" << scratch_node << " backend=" << base.compute_backend << '\n';
    std::cout.flush();
    if (o.dry_run) continue;
    const auto setup_start = now_ns();
    // One admission ledger includes all global operands and every private workspace.
    std::map<int,std::size_t> demand; std::mutex admission;
    auto reserve = [&](std::size_t bytes, const std::vector<int>& nodes) {
      std::lock_guard lock(admission);
      const auto page = o.pages == "huge" ? 2*1024*1024UL : t.page_bytes;
      const auto count = add_size(bytes, page-1)/page;
      for (std::size_t i=0;i<nodes.size();++i) {
        auto amount = checked_mul(count/nodes.size()+(i<count%nodes.size())+(nodes.size()>1),page);
        auto& total = demand[nodes[i]]; total = add_size(total,amount);
        const auto free = o.pages == "huge" ? t.node(nodes[i]).huge_free_bytes : t.node(nodes[i]).free_bytes;
        if (total > free*0.9) throw std::runtime_error("Compute operands and workspace exceed 90% of page pool on node " + std::to_string(nodes[i]));
      }
    };
    auto allocate = [&](std::size_t bytes, const std::vector<int>& nodes) {
      reserve(bytes,nodes); return std::make_unique<Buffer>(bytes,nodes,o.pages,t.page_bytes);
    };
    auto a = allocate(a_bytes,src), c = allocate(c_bytes,dst);
    std::unique_ptr<Buffer> b, gamma, beta;
    if (gemm) b = allocate(b_bytes,src);
    if (param_bytes) { gamma = allocate(param_bytes,src); beta = allocate(param_bytes,src); }
    if (gemm) {
      for (std::size_t i=0;i<a_elements;++i) static_cast<vecops::bfloat16_t*>(a->data)[i] = vecops::bfloat16_t{value(i,o.seed)};
      for (std::size_t i=0;i<b_elements;++i) static_cast<vecops::bfloat16_t*>(b->data)[i] = vecops::bfloat16_t{value(i,o.seed+12345)};
      for (std::size_t i=0;i<output_elements;++i) static_cast<vecops::bfloat16_t*>(c->data)[i] = vecops::bfloat16_t{std::numeric_limits<float>::quiet_NaN()};
    } else {
      for (std::size_t i=0;i<a_elements;++i) static_cast<float*>(a->data)[i] = value(i,o.seed);
      std::fill_n(static_cast<float*>(c->data),output_elements,std::numeric_limits<float>::quiet_NaN());
      if (gamma) for (std::size_t i=0;i<width;++i) {
        static_cast<float*>(gamma->data)[i] = 1.0f + value(i,42)*0.25f;
        static_cast<float*>(beta->data)[i] = value(i,43)*0.125f;
      }
    }
    auto verify_pages = [&] {
      for (auto* p : {a.get(),b.get(),gamma.get(),beta.get(),c.get()}) if (p) p->verify();
    };
    verify_pages(); base.page_policy = a->page_policy;
    auto make_workspace = [&](Worker& w, std::size_t bytes) {
      if (bytes) { w.scratch = allocate(bytes,{scratch_node}); std::fill_n(static_cast<char*>(w.scratch->data),bytes,0); w.scratch->verify(); }
      return kernel::WorkspaceView{w.scratch ? w.scratch->data : nullptr, static_cast<nint_t>(bytes)};
    };
    Team team(g.cpus,rows,[&](Worker& w) {
      const auto count = static_cast<nint_t>(w.end-w.begin);
      if (gemm) {
#if defined(HAS_AMX_BF16) || defined(HAS_SME)
#if defined(ARCH_X86_FAMILY)
        if (syscall(SYS_arch_prctl,0x1023,18) != 0) throw std::runtime_error("AMX tile-state permission denied");
#endif
        auto at = make_tensor(static_cast<vecops::bfloat16_t*>(a->data)+w.begin*reduction,make_shape(meta::Any{count},meta::Any{static_cast<nint_t>(reduction)}));
        auto bt = make_tensor(static_cast<vecops::bfloat16_t*>(b->data),make_shape(meta::Any{static_cast<nint_t>(width)},meta::Any{static_cast<nint_t>(reduction)}));
        auto ct = make_tensor(static_cast<vecops::bfloat16_t*>(c->data)+w.begin*width,make_shape(meta::Any{count},meta::Any{static_cast<nint_t>(width)}));
        auto op = ops::matmul(ops::MatmulConfig<GemmAtom>{});
        const auto bytes = op.required_workspace(count,static_cast<nint_t>(width),static_cast<nint_t>(reduction),at,bt,ct);
        auto workspace = make_workspace(w,bytes);
        auto prepared = op.prepare(workspace,count,static_cast<nint_t>(width),static_cast<nint_t>(reduction),at,bt,ct);
        w.call = [prepared=std::move(prepared)] { prepared(); };
#endif
      } else {
        auto xt = make_tensor(static_cast<float*>(a->data)+w.begin*width,make_shape(meta::Any{count},meta::Any{static_cast<nint_t>(width)}));
        auto yt = make_tensor(static_cast<float*>(c->data)+w.begin*width,make_shape(meta::Any{count},meta::Any{static_cast<nint_t>(width)}));
        auto op = ops::layer_norm(ops::LayerNormConfig<float>{.eps=static_cast<float>(o.ln_eps)});
        if (gamma) {
          auto gt = make_tensor(static_cast<float*>(gamma->data),make_shape(meta::Any{static_cast<nint_t>(width)}));
          auto bt = make_tensor(static_cast<float*>(beta->data),make_shape(meta::Any{static_cast<nint_t>(width)}));
          auto workspace = make_workspace(w,op.required_workspace(xt,gt,bt,yt));
          w.call = [op,xt,gt,bt,yt,workspace]() mutable { op(workspace,xt,gt,bt,yt); };
        } else {
          auto workspace = make_workspace(w,op.required_workspace(xt,yt));
          w.call = [op,xt,yt,workspace]() mutable { op(workspace,xt,yt); };
        }
      }
    });
    for (auto* p : {a.get(),b.get(),gamma.get(),beta.get()}) if (p) sum_pages(base.source_pages,*p);
    sum_pages(base.destination_pages,*c);
    std::map<int,std::size_t> workspace_pages;
    for (auto& w : team.workers) if (w->scratch) { base.workspace_bytes += w->scratch->bytes; sum_pages(workspace_pages,*w->scratch); }
    base.workspace_pages = page_counts_text(workspace_pages);
    base.placement = "all-pages-verified";
    auto verify = [&] {
      verify_pages(); for (auto& w : team.workers) if (w->scratch) w->scratch->verify();
      if (gemm) {
        auto* ap=static_cast<vecops::bfloat16_t*>(a->data); auto* bp=static_cast<vecops::bfloat16_t*>(b->data); auto* cp=static_cast<vecops::bfloat16_t*>(c->data);
        for (std::size_t i=0;i<a_elements;++i) if (float(ap[i])!=value(i,o.seed)) throw std::runtime_error("GEMM A changed");
        for (std::size_t i=0;i<b_elements;++i) if (float(bp[i])!=value(i,o.seed+12345)) throw std::runtime_error("GEMM B changed");
        for (std::size_t i=0;i<output_elements;++i) if (!std::isfinite(float(cp[i]))) throw std::runtime_error("GEMM incomplete/nonfinite output");
        auto check_cell = [&](std::size_t index) {
          const auto row=index/width, col=index%width; double sum=0;
          for (std::size_t k=0;k<reduction;++k) sum += double(float(ap[row*reduction+k]))*float(bp[col*reduction+k]);
          const float expected = float(vecops::bfloat16_t{static_cast<float>(sum)}), actual=float(cp[index]);
          if (actual!=expected) throw std::runtime_error("GEMM BF16 reference mismatch at " + std::to_string(index) + ": " + std::to_string(actual) + " vs " + std::to_string(expected));
        };
        if (double(output_elements)*reduction <= 8000000) for (std::size_t i=0;i<output_elements;++i) check_cell(i);
        else {
          for (auto& w : team.workers) for (auto row : {w->begin,w->end-1}) { check_cell(row*width); check_cell(row*width+width-1); }
          for (std::size_t i=0;i<1024;++i) check_cell((i*0x9e3779b97f4a7c15ULL+o.seed)%output_elements);
        }
      } else {
        auto* xp=static_cast<float*>(a->data); auto* yp=static_cast<float*>(c->data);
        for (std::size_t row=0;row<rows;++row) {
          double mean=0,var=0;
          for (std::size_t col=0;col<width;++col) { auto i=row*width+col; if (xp[i]!=value(i,o.seed)) throw std::runtime_error("LN input changed"); mean+=xp[i]; }
          mean/=width;
          for (std::size_t col=0;col<width;++col) { double d=xp[row*width+col]-mean; var+=d*d; }
          const auto scale=1/std::sqrt(var/width+static_cast<float>(o.ln_eps));
          for (std::size_t col=0;col<width;++col) {
            double expected=(xp[row*width+col]-mean)*scale;
            if (gamma) {
              if (static_cast<float*>(gamma->data)[col]!=1.0f+value(col,42)*0.25f || static_cast<float*>(beta->data)[col]!=value(col,43)*0.125f) throw std::runtime_error("LN affine input changed");
              expected=expected*static_cast<float*>(gamma->data)[col]+static_cast<float*>(beta->data)[col];
            }
            auto actual=yp[row*width+col];
            if (!std::isfinite(actual) || std::abs(actual-expected)>5e-4+2e-4*std::abs(expected)) throw std::runtime_error("LN scalar reference mismatch");
          }
        }
      }
    };
    team.phase(1); verify();
    base.verification = gemm ? (double(output_elements)*reduction<=8000000 ? "full-fp64-reference" : "1024+row-boundaries-fp64-reference;all-finite;all-inputs") : "full-fp64-reference;all-inputs";
    auto elapsed = [&] {
      std::uint64_t begin=UINT64_MAX,end=0;
      for (auto& w : team.workers) { begin=std::min(begin,w->start); end=std::max(end,w->finish); }
      return (end-begin)/1e9;
    };
    double per_call=elapsed(); base.calibration_seconds=per_call;
    auto repeat_count = [](double seconds,double cost) { return static_cast<std::uint64_t>(std::clamp(std::ceil(seconds/std::max(cost,1e-9)),1.0,1000000.0)); };
    // A cold first call can greatly overestimate steady cost. Accumulate actual
    // timed warmup work, recalibrating until the requested duration is reached.
    while (base.warmup_actual_seconds < o.warmup) {
      auto count=repeat_count(o.warmup-base.warmup_actual_seconds,per_call);
      team.phase(count);
      const auto duration=elapsed();
      base.warmup_actual_seconds+=duration; per_call=duration/count;
    }
    const auto repeats=repeat_count(o.seconds,per_call);
    base.setup_seconds=(now_ns()-setup_start)/1e9;
    std::vector<Row> retained;
    for (int sample=1;sample<=o.samples;++sample) for (int attempt=0;attempt<=o.retries;++attempt) {
      team.phase(repeats);
      Row total=base; total.sample=sample; total.attempt=attempt; total.iterations=repeats;
      total.start_ns=UINT64_MAX; std::uint64_t latest=0,earliest=UINT64_MAX;
      std::vector<Row> workers;
      for (auto& w : team.workers) {
        Row r=base; r.scope="thread:cpu"+std::to_string(w->cpu); r.cpus=std::to_string(w->cpu); r.threads=1;
        r.sample=sample; r.attempt=attempt; r.iterations=repeats; r.row_begin=w->begin; r.row_end=w->end;
        r.start_ns=w->start; r.end_ns=w->finish; r.seconds=(w->finish-w->start)/1e9;
        r.minor_faults=w->minor; r.major_faults=w->major; r.switches=w->switches;
        // Global tensor/page inventories appear only on total rows, never duplicated per worker.
        r.source_pages.clear(); r.destination_pages.clear(); r.workspace_pages.clear();
        r.tensor_input_bytes=r.tensor_output_bytes=r.working_bytes=0;
        r.workspace_bytes=w->scratch ? w->scratch->bytes : 0;
        if (w->end_cpu!=w->cpu || w->switches) { r.status="interrupted"; r.detail="Worker switched/migrated"; }
        if (w->minor || w->major) { r.status="page-fault"; r.detail="Faults in compute interval"; }
        if (r.status!="ok") { total.status=r.status; total.detail=r.detail; }
        total.start_ns=std::min(total.start_ns,w->start); total.end_ns=std::max(total.end_ns,w->finish);
        latest=std::max(latest,w->start); earliest=std::min(earliest,w->finish);
        total.minor_faults+=w->minor; total.major_faults+=w->major; total.switches+=w->switches;
        workers.push_back(std::move(r));
      }
      total.start_skew_ns=latest-total.start_ns; total.end_skew_ns=total.end_ns-earliest;
      total.common_ns=earliest>latest ? earliest-latest : 0;
      total.seconds=(total.end_ns-total.start_ns)/1e9;
      // End spread is real operator imbalance, not an admission failure: all lanes
      // perform identical repeat counts and global throughput includes the slowest.
      if (total.start_skew_ns>o.max_skew*o.seconds*1e9) { total.status="skew"; total.detail="Compute start skew exceeded threshold"; }
      total.operation_seconds=total.seconds/repeats;
      total.gelements=double(output_elements)*repeats/total.seconds/1e9;
      total.gflops=gemm ? 2*double(rows)*width*reduction*repeats/total.seconds/1e9 : 0;
      total.logical_gbs=double(add_size(base.tensor_input_bytes,base.tensor_output_bytes))*repeats/total.seconds/1e9;
      if (total.status!="ok") for (auto& r : workers) if (r.status=="ok") { r.status="retry"; r.detail="Another worker invalidated whole attempt"; }
      retained.insert(retained.end(),workers.begin(),workers.end()); retained.push_back(total);
      if (total.status=="ok" || attempt==o.retries) break;
    }
    try { verify(); } catch (...) {
      for (auto& r : retained) { r.status="verification-failed"; reporter.add(r); } throw;
    }
    std::vector<Row> totals;
    for (const auto& r : retained) { reporter.add(r); if (r.scope=="total") totals.push_back(r); }
    reporter.summary(totals,"Compute "+base.operation+" "+base.shape);
  }
}
} // namespace memory_bench
