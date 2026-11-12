// SPDX-FileCopyrightText: 2026 HenryRenYz and vecops contributors
// SPDX-License-Identifier: MIT
// A static CPU entry point into Torch's existing C++ interpreter. It deliberately
// does not use GraphExecutor's profiling, fusion, specialization or DCE passes:
// every data operation and mutable/void native call must survive unchanged.
// Pure views of guarded inputs may move into the submission prologue.
#include <ATen/core/Tensor.h>
#include <ATen/ScalarOps.h>
#include <ATen/Context.h>
#include <ATen/record_function.h>
#include <ATen/autocast_mode.h>
#include <ATen/core/dispatch/Dispatcher.h>
#include <c10/core/InferenceMode.h>
#include <c10/core/impl/LocalDispatchKeySet.h>
#include <torch/custom_class.h>
#include <torch/library.h>
#include <torch/csrc/jit/ir/ir.h>
#include <torch/csrc/jit/ir/irparser.h>
#include <torch/csrc/jit/runtime/interpreter.h>
#include <torch/csrc/jit/runtime/operator.h>
#include <atomic>
#include <algorithm>
#include <chrono>
#include <limits>
#include <mutex>
#include <sstream>
#include <unordered_map>
#include <unordered_set>
#include <functional>
#include "vecops/runtime/KernelPlanAbi.h"

namespace {
// Torch's float32 matmul precision is process-global, not TLS. Graphs serialize
// their own execution; concurrent external mutation of Torch global math policy
// is unsupported, just as it is for eager contexts that change this flag.
std::mutex replay_math_mutex;
class CaptureLease : public torch::CustomClassHolder {
 public:
  CaptureLease() : lock_(replay_math_mutex, std::try_to_lock) {
    TORCH_CHECK(lock_.owns_lock(), "concurrent graph capture/execution is unsupported");
  }
  void release() {
    if (lock_.owns_lock()) lock_.unlock();
  }
 private:
  std::unique_lock<std::mutex> lock_;
};
struct MathPrecision;
thread_local MathPrecision* active_math_precision = nullptr;
struct MathPrecision {
  at::Float32MatmulPrecision old, current;
  MathPrecision() : old(at::globalContext().float32MatmulPrecision()), current(old) {
    TORCH_CHECK(active_math_precision == nullptr, "nested graph math context is unsupported");
    active_math_precision = this;
  }
  void set(at::Float32MatmulPrecision requested) {
    if (current != requested) {
      at::globalContext().setFloat32MatmulPrecision(
          requested == at::Float32MatmulPrecision::HIGHEST ? "highest" :
          requested == at::Float32MatmulPrecision::HIGH ? "high" : "medium");
      current = requested;
    }
  }
  ~MathPrecision() {
    active_math_precision = nullptr;
    set(old);
  }
};

// Pool activation is an ordinary recorded native side effect. Keep the exact
// provider's C ABI functions, not a callback into the Python region wrapper.
// Successful enters are unwound on interpreter errors as well as normal exits.
using PoolBegin = void (*)(uint32_t);
using PoolEnd = void (*)();
struct PoolRegions;
thread_local PoolRegions* active_pool_regions = nullptr;
struct PoolRegions {
  std::vector<PoolEnd>& exits;
  explicit PoolRegions(std::vector<PoolEnd>& storage) : exits(storage) {
    TORCH_CHECK(active_pool_regions == nullptr && exits.empty(), "nested graph pool context is unsupported");
    active_pool_regions = this;
  }
  void enter(PoolBegin begin, PoolEnd end, uint32_t threads) {
    TORCH_INTERNAL_ASSERT(exits.size() < exits.capacity());
    begin(threads);
    exits.push_back(end);
  }
  void leave(PoolEnd end) {
    TORCH_CHECK(!exits.empty() && exits.back() == end, "unbalanced graph native-pool region");
    exits.pop_back();
    end();
  }
  ~PoolRegions() {
    // Native pool C ABI end functions are nonthrowing. The provider DSO stays
    // strongly owned by the executable until this scope has finished.
    while (!exits.empty()) {
      const auto end = exits.back(); exits.pop_back(); end();
    }
    active_pool_regions = nullptr;
  }
};

void check_tensor_keys(const at::Tensor& t, size_t index) {
  TORCH_CHECK(t.defined(), "graph input ", index, " must be defined");
  const c10::DispatchKeySet allowed({c10::DispatchKey::CPU,
      c10::DispatchKey::ADInplaceOrView, c10::DispatchKey::AutogradCPU,
      c10::DispatchKey::AutocastCPU});
  // Inspect native keys before any potentially virtual metadata access on a
  // wrapper TensorImpl; rejecting it must not invoke Python to ask its sizes.
  for (auto key : t.key_set()) {
    TORCH_CHECK(allowed.has(key), "graph input ", index,
                " has unsupported Python/functional/batched dispatch key ", key);
  }
}

struct Guard {
  std::vector<int64_t> sizes, strides;
  int64_t offset;
  c10::ScalarType dtype;
  bool requires_grad;
  size_t storage_bytes;
  explicit Guard(const at::Tensor& t)
      : sizes(t.sizes().vec()), strides(t.strides().vec()),
        offset(t.storage_offset()), dtype(t.scalar_type()),
        requires_grad(t.requires_grad()), storage_bytes(t.storage().nbytes()) {}
  void check(const at::Tensor& t, size_t index) const {
    check_tensor_keys(t, index);
    TORCH_CHECK(t.defined() && t.device().is_cpu() && t.layout() == c10::kStrided &&
                    !t.is_quantized() && !t.is_nested() && !t.is_conj() && !t.is_neg(),
                "graph input ", index, " must be a plain dense-strided CPU tensor");
    TORCH_CHECK(t.sizes().equals(sizes) && t.strides().equals(strides) &&
                    t.storage_offset() == offset && t.scalar_type() == dtype &&
                    t.requires_grad() == requires_grad && t.storage().nbytes() == storage_bytes,
                "static metadata guard failed for graph input ", index);
  }
};

std::vector<int64_t> aliases(const std::vector<at::Tensor>& tensors) {
  std::unordered_map<const c10::StorageImpl*, int64_t> groups;
  std::vector<int64_t> result;
  for (const auto& t : tensors) {
    auto [it, inserted] = groups.emplace(t.storage().unsafeGetStorageImpl(), groups.size());
    result.push_back(it->second);
  }
  return result;
}

// Distinct StorageImpls can still overlap (from_blob/frombuffer). Record their
// connected byte-interval components and relative offsets in O(n log n), not
// just StorageImpl equality. Component IDs follow input order, so moving an
// independent allocation to another address remains legal.
std::vector<std::vector<uint64_t>> byte_aliases(const std::vector<at::Tensor>& tensors) {
  struct Span { uintptr_t begin, end; size_t index; };
  std::unordered_map<const c10::StorageImpl*, size_t> unique;
  std::vector<size_t> tensor_storage;
  std::vector<Span> spans;
  for (const auto& t : tensors) {
    auto* storage = t.storage().unsafeGetStorageImpl();
    auto [it, inserted] = unique.emplace(storage, spans.size());
    tensor_storage.push_back(it->second);
    if (inserted) {
      auto begin = reinterpret_cast<uintptr_t>(storage->data());
      auto bytes = storage->nbytes();
      TORCH_CHECK(bytes <= std::numeric_limits<uintptr_t>::max() - begin, "storage address overflow");
      spans.push_back({begin, begin + bytes, spans.size()});
    }
  }
  auto ordered = spans;
  std::sort(ordered.begin(), ordered.end(), [](const Span& a, const Span& b) {
    return a.begin < b.begin || (a.begin == b.begin && a.index < b.index);
  });
  std::vector<size_t> component(spans.size());
  std::vector<uintptr_t> origin(spans.size());
  uintptr_t end = 0, start = 0; size_t current = 0; bool first = true;
  for (const auto& span : ordered) {
    if (span.begin == span.end) {
      // Empty spans are independent but must not truncate a live nonempty
      // component that happens to contain their address.
      component[span.index] = span.index; origin[span.index] = span.begin;
      continue;
    }
    if (first || span.begin >= end) {
      current = span.index; start = span.begin; end = span.end; first = false;
    } else {
      end = std::max(end, span.end);
    }
    component[span.index] = current; origin[span.index] = start;
  }
  std::unordered_map<size_t, size_t> canonical;
  std::vector<std::vector<uint64_t>> result;
  for (auto index : tensor_storage) {
    auto [it, inserted] = canonical.emplace(component[index], canonical.size());
    result.push_back({it->second, spans[index].begin - origin[index],
                      spans[index].end - spans[index].begin});
  }
  return result;
}

// A retained C ABI plan replaces the native wrapper, metadata conversion and
// source entry. Its runtime metadata is fixed by this graph's static program;
// only logical data addresses/scalar payloads are patched from the JIT stack.
std::vector<int64_t> plan_indices(const std::string& text) {
  std::vector<int64_t> values;
  size_t begin = 0;
  while (begin < text.size()) {
    const auto end = text.find(',', begin);
    const auto token = text.substr(begin, end - begin);
    size_t consumed = 0;
    values.push_back(std::stoll(token, &consumed));
    TORCH_CHECK(consumed == token.size(), "invalid native plan index table");
    if (end == std::string::npos) break;
    begin = end + 1;
  }
  return values;
}

class PlanBinding {
 public:
  PlanBinding(int64_t address, std::vector<int64_t> flags, std::vector<int64_t> dtypes,
              std::vector<int64_t> outputs, size_t arguments)
      : flags_(std::move(flags)), dtypes_(std::move(dtypes)), outputs_(std::move(outputs)), arguments_(arguments) {
    TORCH_CHECK(address, "null native kernel plan");
    const auto* source = reinterpret_cast<const VecopsPreparedKernelV1*>(static_cast<uintptr_t>(address));
    TORCH_CHECK(source->struct_size >= offsetof(VecopsPreparedKernelV1, inplace_count) && source->abi_major == 1 &&
        source->retain && source->destroy && source->bind_prevalidated && source->replay &&
        source->execution_context, "invalid retained native kernel plan");
    TORCH_CHECK(flags_.size() == dtypes_.size() && flags_.size() <= arguments_, "invalid native plan argument table");
    for (auto index : outputs_) TORCH_CHECK(index >= 0 && static_cast<size_t>(index) < flags_.size(), "invalid plan output");
    values_.resize(flags_.size()); results_.resize(outputs_.size());
    std::memcpy(&abi_, source, std::min<std::size_t>(source->struct_size, sizeof(abi_)));
    abi_.retain(abi_.context);
    for (size_t i = 0; i < values_.size(); ++i) {
      values_[i].struct_size = sizeof(VecopsValue);
    }
  }
  PlanBinding(const PlanBinding&) = delete;
  ~PlanBinding() { abi_.destroy(abi_.context); }
  const VecopsPreparedKernelV1& abi() const { return abi_; }
  size_t arguments() const { return arguments_; }
  void bind(const torch::jit::Stack& stack, size_t base) {
    RECORD_FUNCTION("vecops::plan_bind", c10::ArrayRef<c10::IValue>());
    TORCH_CHECK(stack.size() >= base + arguments_, "native plan stack underflow");
    for (size_t i = 0; i < values_.size(); ++i) {
      const auto& value = stack[base + i]; auto& item = values_[i];
      if (flags_[i]) {
        if (value.isNone()) { item.kind = VECOPS_VALUE_NONE; continue; }
        const auto& tensor = value.toTensor();
        item.kind = VECOPS_VALUE_TENSOR;
        // Tensor meta/layout was lowered at compile time. data_ptr includes
        // the current storage offset, so the symbolic view starts at zero.
        item.value.tensor = {};
        item.value.tensor.struct_size = sizeof(VecopsTensorView);
        item.value.tensor.data = const_cast<void*>(tensor.const_data_ptr());
      } else {
        item.kind = VECOPS_VALUE_SCALAR;
        item.value.scalar = {};
        item.value.scalar.struct_size = sizeof(VecopsScalar);
        item.value.scalar.dtype = static_cast<uint32_t>(dtypes_[i]);
        if (dtypes_[i] == VECOPS_DTYPE_BOOL) item.value.scalar.value.u64 = value.toBool();
        else if (dtypes_[i] >= VECOPS_DTYPE_F16) item.value.scalar.value.f64 = value.toDouble();
        else if (dtypes_[i] == VECOPS_DTYPE_U8 || dtypes_[i] == VECOPS_DTYPE_U16 ||
                 dtypes_[i] == VECOPS_DTYPE_U32 || dtypes_[i] == VECOPS_DTYPE_U64)
          item.value.scalar.value.u64 = static_cast<uint64_t>(value.toInt());
        else item.value.scalar.value.i64 = value.toInt();
      }
    }
    VecopsCall call{sizeof(VecopsCall), static_cast<uint32_t>(values_.size()), values_.data(), nullptr, 0,
                     abi_.execution_context};
    char message[1024]{}; VecopsError error{sizeof(VecopsError), 0, message, sizeof(message), 0};
    TORCH_CHECK(abi_.bind_prevalidated(abi_.context, &call, &error) == VECOPS_STATUS_OK,
                "native plan binding failed: ", message);
  }
  void call(torch::jit::Stack& stack) {
    RECORD_FUNCTION("vecops::kernel_plan", c10::ArrayRef<c10::IValue>());
    TORCH_CHECK(stack.size() >= arguments_, "native plan stack underflow");
    const auto base = stack.size() - arguments_;
    bind(stack, base);
    char message[1024]{}; VecopsError error{sizeof(VecopsError), 0, message, sizeof(message), 0};
    TORCH_CHECK(abi_.replay(abi_.context, &error) == VECOPS_STATUS_OK, "native plan execution failed: ", message);
    for (size_t i = 0; i < outputs_.size(); ++i) results_[i] = stack[base + outputs_[i]];
    stack.resize(base);
    for (auto& output : results_) stack.push_back(std::move(output));
  }
 private:
  VecopsPreparedKernelV1 abi_{};
  std::vector<int64_t> flags_, dtypes_, outputs_;
  size_t arguments_;
  std::vector<VecopsValue> values_;
  std::vector<c10::IValue> results_;
};

// Consecutive out-style kernels have no interpreter-produced intermediate
// values. Bind them once and let one physical team advance the entire island.
class PlanSequenceBinding {
 public:
  struct Timing {
    using Clock = std::chrono::steady_clock;
    Clock::time_point start;
    double elapsed_us = 0;
  };
  using Markers = std::pair<std::shared_ptr<Timing>, std::shared_ptr<Timing>>;
  explicit PlanSequenceBinding(std::vector<std::shared_ptr<PlanBinding>> plans,
                               std::vector<Markers> markers)
      : plans_(std::move(plans)), markers_(std::move(markers)) {
    for (const auto& plan : plans_) { arguments_ += plan->arguments(); workers_ = std::max(workers_, plan->abi().workers); }
  }
  void call(torch::jit::Stack& stack) {
    RECORD_FUNCTION("vecops::native_island", c10::ArrayRef<c10::IValue>());
    TORCH_CHECK(stack.size() >= arguments_, "native sequence stack underflow");
    const auto base = stack.size() - arguments_;
    size_t offset = base, entered = 0;
    char message[1024]{}; VecopsError error{sizeof(VecopsError), 0, message, sizeof(message), 0};
    try {
      for (const auto& plan : plans_) {
        plan->bind(stack, offset); offset += plan->arguments();
        const auto& abi = plan->abi();
        TORCH_CHECK(abi.enter(abi.context, &error) == VECOPS_STATUS_OK, "native sequence enter failed: ", message);
        ++entered;
      }
      const auto* pool = plans_.front()->abi().thread_pool;
      RECORD_FUNCTION("vecops::resident_team", c10::ArrayRef<c10::IValue>());
      TORCH_CHECK(pool->team_executor->run(pool->context, workers_, this,
          [](void* opaque, const VecopsTeamContextV1* team) {
            auto& self = *static_cast<PlanSequenceBinding*>(opaque);
            auto run = [&](const auto& plan) {
              char message[1024]{}; VecopsError error{sizeof(VecopsError), 0, message, sizeof(message), 0};
              const auto& abi = plan->abi();
              TORCH_CHECK(abi.run_lane(abi.context, team, &error) == VECOPS_STATUS_OK,
                          "native sequence worker failed: ", message);
            };
            if (self.markers_.empty()) {
              for (const auto& plan : self.plans_) run(plan);
              return;
            }
            auto sync = [&] {
              if (team->worker_count == 1) return;
              char message[1024]{}; VecopsError error{sizeof(VecopsError), 0, message, sizeof(message), 0};
              TORCH_CHECK(team->barrier(team->context, &error) == VECOPS_STATUS_OK,
                          "shared-page calibration barrier failed: ", message);
            };
            for (size_t i = 0; i < self.plans_.size(); ++i) {
              const auto& [begin,end] = self.markers_[i];
              if (begin) {
                sync();
                if (team->worker_id == 0) begin->start = Timing::Clock::now();
                sync();
              }
              run(self.plans_[i]);
              if (end) {
                sync();
                if (team->worker_id == 0)
                  end->elapsed_us = std::chrono::duration<double,std::micro>(Timing::Clock::now()-end->start).count();
              }
            }
          }, &error) == VECOPS_STATUS_OK, "native sequence execution failed: ", message);
    } catch (...) {
      while (entered) { const auto& abi = plans_[--entered]->abi(); abi.leave(abi.context, 0); }
      throw;
    }
    while (entered) { const auto& abi = plans_[--entered]->abi(); abi.leave(abi.context, 1); }
    stack.resize(base);
  }
 private:
  std::vector<std::shared_ptr<PlanBinding>> plans_;
  std::vector<Markers> markers_;
  size_t arguments_ = 0;
  uint32_t workers_ = 1;
};

// A graph-local JIT name binds an immutable, token-checked CPU KernelFunction.
// Dispatcher registration replacement must not redirect replay into Python.
// The SafeKernelFunction either retains the original native callable or rejects
// a registration invalidated by Torch. Owners are torn down after Code/Graph.
struct BoundOperators {
  std::vector<c10::FunctionSchema> schemas;
  ~BoundOperators() {
    for (const auto& schema : schemas) torch::jit::deregisterOperator(schema);
  }
};

// These views depend only on guarded input metadata and literal sizes/strides.
// Keep their ordinary per-submission binding, but construct them before any
// compute so they cannot split a resident team between two source kernels.
// No tensor data operation, allocation or mutation is moved by this pass.
int64_t move_input_views_to_prologue(const std::shared_ptr<torch::jit::Graph>& graph) {
  using Node = torch::jit::Node;
  std::vector<Node*> ordered(graph->nodes().begin(), graph->nodes().end());
  std::unordered_set<torch::jit::Value*> ready(graph->inputs().begin(), graph->inputs().end());
  std::unordered_set<Node*> selected;
  std::function<bool(torch::jit::Value*, std::unordered_set<Node*>&)> literals =
      [&](torch::jit::Value* value, std::unordered_set<Node*>& dependencies) {
        auto* node = value->node();
        if (node->kind() != torch::jit::prim::Constant && node->kind() != torch::jit::prim::ListConstruct)
          return false;
        for (auto* input : node->inputs())
          if (!literals(input, dependencies)) return false;
        dependencies.insert(node);
        return true;
      };
  int64_t count = 0;
  for (auto* node : ordered) {
    const std::string name = node->kind().toQualString();
    const bool relative_view = name.size() >= 13 && name.compare(name.size() - 13, 13, "::static_view") == 0;
    if ((name != "aten::as_strided" && !relative_view) || node->inputs().size() != 4 ||
        node->outputs().size() != 1 || !ready.count(node->input(0))) continue;
    std::unordered_set<Node*> dependencies;
    bool fixed = true;
    for (size_t i = 1; i < node->inputs().size(); ++i)
      fixed &= literals(node->input(i), dependencies);
    if (!fixed) continue;
    selected.insert(dependencies.begin(), dependencies.end());
    selected.insert(node);
    ready.insert(node->output());
    ++count;
  }
  if (!ordered.empty()) {
    auto* anchor = ordered.front();
    for (auto* node : ordered) if (selected.count(node)) {
      if (node == anchor) anchor = node->next();
      else node->moveBefore(anchor);
    }
  }
  return count;
}

class Executor : public torch::CustomClassHolder {
 public:
  Executor(std::string ir, std::vector<at::Tensor> examples) {
    graph_ = std::make_shared<torch::jit::Graph>();
    torch::jit::parseIR(ir, graph_.get());
    TORCH_CHECK(graph_->inputs().size() == examples.size(), "graph input count mismatch");
    for (const auto& t : examples) {
      check_tensor_keys(t, guards_.size());
      TORCH_CHECK(t.defined() && t.device().is_cpu() && t.layout() == c10::kStrided,
                  "only CPU strided examples are supported");
      guards_.emplace_back(t);
      guards_.back().check(t, guards_.size() - 1);
    }
    aliases_ = aliases(examples);
    byte_aliases_ = byte_aliases(examples);
    prologue_views_ = move_input_views_to_prologue(graph_);
    // No blocks/control flow or indirect/Python calls are accepted, even if a
    // caller manually bypasses the Python lowering validation.
    std::vector<torch::jit::Node*> nodes(graph_->nodes().begin(), graph_->nodes().end());
    std::unordered_map<std::string, std::string> bound_names;
    std::unordered_map<std::string, std::shared_ptr<PlanBinding>> bound_plans;
    std::unordered_map<torch::jit::Node*, std::shared_ptr<PlanBinding>> recorded_nodes;
    const auto timing_begin = c10::Symbol::attr("vecops_shared_measure_begin");
    const auto timing_end = c10::Symbol::attr("vecops_shared_measure_end");
    static std::atomic<uint64_t> sequence{0};
    const std::string prefix = "vecops_bound_" +
        std::to_string(reinterpret_cast<uintptr_t>(this)) + "_" +
        std::to_string(sequence.fetch_add(1, std::memory_order_relaxed)) + "::op_";
    for (auto* node : nodes) {
      TORCH_CHECK(node->blocks().empty(), "replay IR cannot contain control-flow blocks");
      const std::string name = node->kind().toQualString();
      TORCH_CHECK(name.rfind("prim::", 0) != 0 || name == "prim::Constant" ||
                      name == "prim::ListConstruct" || name == "prim::TupleConstruct" ||
                      name == "prim::TupleUnpack" || name == "prim::ListUnpack" ||
                      name == "prim::NumToTensor",
                  "unsupported interpreter primitive: ", name);
      if (name.rfind("prim::", 0) == 0) continue;
      const auto original = node->schema();
      if (original.name().size() >= 12 &&
          original.name().compare(original.name().size() - 12, 12, "::pool_begin") == 0)
        ++pool_region_capacity_;
      auto signature = c10::toString(original);
      const auto precision_symbol = c10::Symbol::attr("vecops_matmul_precision");
      const auto precision = node->hasAttribute(precision_symbol)
          ? node->s(precision_symbol) : "";
      TORCH_CHECK(precision.empty() || precision == "highest" || precision == "high" || precision == "medium",
                  "invalid captured matmul precision: ", precision);
      const auto precision_code = precision.empty() ? -1 : static_cast<int>(
          precision == "highest" ? at::Float32MatmulPrecision::HIGHEST :
          precision == "high" ? at::Float32MatmulPrecision::HIGH : at::Float32MatmulPrecision::MEDIUM);
      const auto plan_symbol = c10::Symbol::attr("vecops_plan_address");
      const auto plan_address = node->hasAttribute(plan_symbol) ? node->i(plan_symbol) : 0;
      std::vector<torch::jit::Value*> binding_inputs(node->inputs().begin(),node->inputs().end());
      const auto paths_symbol=c10::Symbol::attr("vecops_plan_argument_paths");
      const bool flattened=plan_address && node->hasAttribute(paths_symbol);
      if(flattened) {
        binding_inputs.clear();const auto text=node->s(paths_symbol);size_t begin=0;
        while(begin<text.size()) {
          const auto end=text.find(',',begin);
          const auto token=text.substr(begin,end==std::string::npos?end:end-begin);
          const auto split=token.find(':');const auto argument=std::stoull(token.substr(0,split));
          TORCH_CHECK(argument<node->inputs().size(),"prepared effect argument path out of range");
          auto* value=node->input(argument);
          if(split!=std::string::npos) {
            TORCH_CHECK(value->node()->kind()==torch::jit::prim::ListConstruct,"prepared effect requires a static TensorList");
            const auto element=std::stoull(token.substr(split+1));
            TORCH_CHECK(element<value->node()->inputs().size(),"prepared effect list path out of range");
            value=value->node()->input(element);
          }
          TORCH_CHECK(value->type()->isSubtypeOf(c10::TensorType::get()),"prepared effect binding is not a Tensor");
          binding_inputs.push_back(value);if(end==std::string::npos)break;begin=end+1;
        }
      }
      const auto key = signature + "|precision=" + precision + "|plan=" + std::to_string(plan_address);
      auto found = bound_names.find(key);
      std::string bound_name;
      if (found == bound_names.end()) {
        bound_name = prefix + std::to_string(bound_names.size());
        auto handle = c10::Dispatcher::singleton().findSchemaOrThrow(
            original.name().c_str(), original.overload_name().c_str());
        auto kernel = handle.getComputedKernelForDispatchKey(c10::DispatchKey::CPU);
        const auto& debug = kernel.debug();
        TORCH_CHECK(debug.find(".cpp:") != std::string::npos ||
                        debug.find(".cc:") != std::string::npos ||
                        debug.find(".cxx:") != std::string::npos,
                    "CPU kernel has no native C++ registration evidence: ", debug);
        auto bound = original.cloneWithName(bound_name, original.overload_name());
        if(flattened) {
          std::vector<c10::Argument> flat_arguments;
          for(size_t i=0;i<binding_inputs.size();++i)flat_arguments.emplace_back("arg"+std::to_string(i),binding_inputs[i]->type());
          bound=c10::FunctionSchema(bound_name,"",std::move(flat_arguments),original.returns());
        }
        const auto profile_name = original.name() +
            (original.overload_name().empty() ? "" : "." + original.overload_name());
        const auto arity = binding_inputs.size();
        std::shared_ptr<PlanBinding> plan;
        if (plan_address) {
          auto outputs = plan_indices(node->s(c10::Symbol::attr("vecops_plan_outputs")));
          TORCH_CHECK(outputs.size() == original.returns().size(), "native plan return count differs from its schema");
          plan = std::make_shared<PlanBinding>(plan_address, plan_indices(node->s(c10::Symbol::attr("vecops_plan_flags"))),
              plan_indices(node->s(c10::Symbol::attr("vecops_plan_dtypes"))), std::move(outputs), arity);
          bound_plans.emplace(bound_name, plan);
        }
        torch::jit::registerOperator(torch::jit::Operator(
            bound.name(), bound.overload_name(), bound.arguments(), bound.returns(),
            [kernel = std::move(kernel), handle = std::move(handle), signature, precision_code, plan,
             profile_name, arity](torch::jit::Stack& stack) {
              RECORD_FUNCTION(profile_name, c10::ArrayRef<c10::IValue>(stack).slice(stack.size() - arity));
              try {
                if (precision_code >= 0) {
                  TORCH_INTERNAL_ASSERT(active_math_precision != nullptr);
                  active_math_precision->set(static_cast<at::Float32MatmulPrecision>(precision_code));
                }
                if (plan) plan->call(stack);
                else kernel.callBoxed(handle, c10::DispatchKeySet(c10::DispatchKey::CPU), &stack);
              } catch (c10::Error& error) {
                error.add_context("vecops replay operator: " + signature);
                throw;
              }
            }, flattened?c10::AliasAnalysisKind::CONSERVATIVE:c10::AliasAnalysisKind::FROM_SCHEMA));
        bindings_.schemas.push_back(std::move(bound));
        bound_names.emplace(key, bound_name);
      } else {
        bound_name = found->second;
      }
      auto* replacement = graph_->create(c10::Symbol::fromQualString(bound_name),
                                         binding_inputs, node->outputs().size());
      replacement->insertBefore(node);
      for (auto marker : {timing_begin,timing_end})
        if (node->hasAttribute(marker)) replacement->i_(marker,node->i(marker));
      if (auto plan = bound_plans.find(bound_name); plan != bound_plans.end()) recorded_nodes.emplace(replacement, plan->second);
      for (size_t i = 0; i < node->outputs().size(); ++i) {
        replacement->output(i)->copyMetadata(node->output(i));
        node->output(i)->replaceAllUsesWith(replacement->output(i));
      }
      node->destroy();
    }
    // Flattening only removes list packaging, never a mutable/void effect.
    // Eliminate now-unused list constructors before finding resident islands.
    nodes.assign(graph_->nodes().begin(),graph_->nodes().end());
    for(auto it=nodes.rbegin();it!=nodes.rend();++it)
      if((*it)->kind()==torch::jit::prim::ListConstruct && !(*it)->hasUses())(*it)->destroy();
    // Pure Constant nodes may separate source calls. They can be evaluated
    // before the island; every other interpreter operation ends the island.
    std::vector<torch::jit::Node*> island;
    auto flush_island = [&] {
      if (!island.empty()) native_island_sizes_.push_back(island.size());
      if (island.size() < 2) { island.clear(); return; }
      std::vector<std::shared_ptr<PlanBinding>> plans;
      std::vector<torch::jit::Value*> inputs;
      std::vector<c10::Argument> arguments;
      std::vector<PlanSequenceBinding::Markers> markers;
      std::unordered_map<int64_t,std::shared_ptr<PlanSequenceBinding::Timing>> pending;
      for (auto* node : island) {
        plans.push_back(recorded_nodes.at(node));
        PlanSequenceBinding::Markers pair;
        if (node->hasAttribute(timing_begin)) {
          auto id = node->i(timing_begin);
          pair.first = std::make_shared<PlanSequenceBinding::Timing>();
          TORCH_CHECK(pending.emplace(id,pair.first).second,"duplicate shared-page measurement");
        }
        if (node->hasAttribute(timing_end)) {
          auto id = node->i(timing_end);
          auto found = pending.find(id);
          TORCH_CHECK(found!=pending.end(),"shared-page measurement crosses a native island");
          pair.second = found->second;
          pending.erase(found);
          shared_timings_.emplace(id,pair.second);
        }
        if (!markers.empty() || pair.first || pair.second) {
          if (markers.empty()) markers.resize(plans.size()-1);
          markers.push_back(std::move(pair));
        }
        for (auto* input : node->inputs()) {
          arguments.emplace_back("arg" + std::to_string(inputs.size()), input->type());
          inputs.push_back(input);
        }
      }
      const auto name = prefix + "island_" + std::to_string(bindings_.schemas.size());
      TORCH_CHECK(pending.empty(),"shared-page measurement crosses a native island");
      auto sequence = std::make_shared<PlanSequenceBinding>(std::move(plans),std::move(markers));
      c10::FunctionSchema schema(name, "", arguments, {});
      torch::jit::registerOperator(torch::jit::Operator(name, "", schema.arguments(), schema.returns(),
          [sequence](torch::jit::Stack& stack) { sequence->call(stack); }, c10::AliasAnalysisKind::CONSERVATIVE));
      bindings_.schemas.push_back(std::move(schema));
      auto* replacement = graph_->create(c10::Symbol::fromQualString(name), inputs, 0);
      replacement->insertBefore(island.back());
      for (auto* node : island) node->destroy();
      ++native_islands_;
      island.clear();
    };
    nodes.assign(graph_->nodes().begin(), graph_->nodes().end());
    for (auto* node : nodes) {
      if (node->kind() == torch::jit::prim::Constant) continue;
      auto found = recorded_nodes.find(node);
      bool eligible = found != recorded_nodes.end() && node->outputs().empty();
      if (eligible) {
        const auto& abi = found->second->abi();
        const auto* pool = abi.thread_pool;
        eligible = pool && pool->struct_size >= sizeof(VecopsThreadPoolV1) && pool->team_executor &&
            pool->team_executor->struct_size >= sizeof(VecopsTeamExecutorV1) &&
            pool->team_executor->abi_major == 1 && pool->team_executor->run &&
            abi.enter && abi.run_lane && abi.leave;
        if (eligible && !island.empty() && recorded_nodes.at(island.front())->abi().thread_pool != pool) flush_island();
      }
      if (!eligible) flush_island();
      else island.push_back(node);
    }
    flush_island();
    code_ = torch::jit::Code(graph_, "vecops_static_replay");
    pool_exits_.reserve(pool_region_capacity_);
  }

  int64_t native_islands() const { return native_islands_; }
  int64_t prologue_views() const { return prologue_views_; }
  std::vector<int64_t> native_island_sizes() const { return native_island_sizes_; }
  std::vector<std::tuple<int64_t,double>> shared_page_samples() const {
    std::vector<std::tuple<int64_t,double>> samples;
    for (const auto& [id,timing] : shared_timings_) samples.emplace_back(id,timing->elapsed_us);
    return samples;
  }
  std::vector<at::Tensor> run(std::vector<at::Tensor> tensors) {
    RECORD_FUNCTION("vecops::graph_replay", c10::ArrayRef<c10::IValue>());
    TORCH_CHECK(!running_.test_and_set(std::memory_order_acquire),
                "graph replay is not reentrant or concurrently callable");
    struct Release {
      std::atomic_flag& flag;
      ~Release() { flag.clear(std::memory_order_release); }
    } release{running_};
    const auto keys = c10::impl::tls_local_dispatch_key_set().included_;
    for (auto forbidden : {c10::DispatchKey::Python, c10::DispatchKey::PythonTLSSnapshot,
         c10::DispatchKey::PythonDispatcher, c10::DispatchKey::PreDispatch,
         c10::DispatchKey::Functionalize, c10::DispatchKey::FuncTorchDynamicLayerFrontMode,
         c10::DispatchKey::FuncTorchDynamicLayerBackMode, c10::DispatchKey::FuncTorchBatched,
         c10::DispatchKey::FuncTorchVmapMode, c10::DispatchKey::FuncTorchGradWrapper}) {
      TORCH_CHECK(!keys.has(forbidden),
                  "graph replay rejects an active Python/functional dispatch context: ", forbidden);
    }
    TORCH_CHECK(tensors.size() == guards_.size(), "graph replay input count mismatch");
    for (size_t i = 0; i < tensors.size(); ++i) guards_[i].check(tensors[i], i);
    TORCH_CHECK(aliases(tensors) == aliases_, "graph input storage-alias guard failed");
    TORCH_CHECK(byte_aliases(tensors) == byte_aliases_, "graph input byte-overlap alias guard failed");
    TORCH_CHECK(!at::autocast::is_autocast_enabled(at::kCPU),
                "graph replay rejects CPU autocast contexts");
    std::unique_lock<std::mutex> math_lock(replay_math_mutex, std::try_to_lock);
    TORCH_CHECK(math_lock.owns_lock(), "concurrent graph execution is unsupported");
    MathPrecision math_precision;
    PoolRegions pool_regions(pool_exits_);
    c10::InferenceMode inference;
    torch::jit::Stack stack;
    stack.reserve(tensors.size());
    for (auto& t : tensors) stack.emplace_back(std::move(t));
    torch::jit::InterpreterState(code_).run(stack);
    TORCH_CHECK(stack.size() == 1 && stack[0].isTuple(), "invalid replay return stack");
    std::vector<at::Tensor> result;
    for (const auto& value : stack[0].toTupleRef().elements()) result.push_back(value.toTensor());
    return result;
  }

 private:
  BoundOperators bindings_;
  std::shared_ptr<torch::jit::Graph> graph_;
  torch::jit::Code code_;
  std::vector<Guard> guards_;
  std::vector<int64_t> aliases_;
  std::vector<std::vector<uint64_t>> byte_aliases_;
  std::atomic_flag running_ = ATOMIC_FLAG_INIT;
  size_t pool_region_capacity_ = 0;
  int64_t native_islands_ = 0;
  std::vector<int64_t> native_island_sizes_;
  std::unordered_map<int64_t,std::shared_ptr<PlanSequenceBinding::Timing>> shared_timings_;
  int64_t prologue_views_ = 0;
  std::vector<PoolEnd> pool_exits_;
};
} // namespace

#define VECOPS_DEFINE_GRAPH_LIBRARY(NS) TORCH_LIBRARY(NS, m)
VECOPS_DEFINE_GRAPH_LIBRARY(VECOPS_GRAPH_REPLAY_NAMESPACE) {
  m.def("static_view(Tensor(a) input, int[] sizes, int[] strides, int relative_offset) -> Tensor(a)",
        [](const at::Tensor& input, at::IntArrayRef sizes, at::IntArrayRef strides, int64_t relative_offset) {
    return input.as_strided(sizes, strides, input.storage_offset() + relative_offset);
  });
  // NumToTensor creates an ordinary zero-dimensional tensor, which can promote
  // a scalar float32 input to float64. Torch's Tensor-schema Python number
  // conversion instead uses this wrapped-number representation.
  m.def("wrapped_scalar(Scalar value) -> Tensor", [](const c10::Scalar& value) {
    return at::native::wrapped_scalar_tensor(value);
  });
  m.def("pool_begin(int begin_address, int end_address, int threads) -> ()",
        [](int64_t begin_address, int64_t end_address, int64_t threads) {
    TORCH_CHECK(begin_address && end_address && threads > 0 &&
                    static_cast<uint64_t>(threads) <= std::numeric_limits<uint32_t>::max(),
                "invalid native-pool activation binding");
    const auto begin = reinterpret_cast<PoolBegin>(static_cast<uintptr_t>(begin_address));
    const auto end = reinterpret_cast<PoolEnd>(static_cast<uintptr_t>(end_address));
    if (active_pool_regions) active_pool_regions->enter(begin, end, static_cast<uint32_t>(threads));
    else begin(static_cast<uint32_t>(threads));
  });
  m.def("pool_end(int end_address) -> ()", [](int64_t end_address) {
    TORCH_CHECK(end_address, "invalid native-pool exit binding");
    const auto end = reinterpret_cast<PoolEnd>(static_cast<uintptr_t>(end_address));
    if (active_pool_regions) active_pool_regions->leave(end);
    else end();
  });
  m.class_<Executor>("Executor")
      .def(torch::init<std::string, std::vector<at::Tensor>>())
      .def("run", &Executor::run)
      .def("native_islands", &Executor::native_islands)
      .def("prologue_views", &Executor::prologue_views)
      .def("shared_page_samples", &Executor::shared_page_samples)
      .def("native_island_sizes", &Executor::native_island_sizes);
  m.class_<CaptureLease>("CaptureLease")
      .def(torch::init<>())
      .def("release", &CaptureLease::release);
}
