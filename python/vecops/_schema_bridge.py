# SPDX-FileCopyrightText: 2026 HenryRenYz and vecops contributors
# SPDX-License-Identifier: MIT
"""Generate and load stable Torch bridges from a native :class:`KernelDef`.

The generated extension contains only Torch-to-``CallAbi`` adaptation and a
process-local operator handle. Kernel selection, caching, and JIT compilation
remain in vecops C++ runtime code, so a call never uses a Python dispatch
callback. The module-level template is intentionally kept outside generation
functions to make its ABI surface reviewable as ordinary source text.
"""

from __future__ import annotations

import ast
import ctypes
import fnmatch
import hashlib
import keyword
import os
import re
import shlex
import shutil
import tempfile
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from string import Template
from threading import Lock
from typing import Any

from . import _C as native
from ._dtype import dtype_name as _dtype_name


@dataclass
class _BridgeRegistration:
  library: str
  name: str
  operator: native.Operator
  kernel_def: native.KernelDef
  compiler: Any
  build_directory: str | os.PathLike[str] | None
  verbose: bool
  extra_cflags: tuple[str, ...]
  return_outputs: bool
  spec_order: tuple[str, ...] | None = None
  gomp_runtime_path: Path | None = None
  wrapper: Any = None
  materialized: bool = False
  library_path: Path | None = None
  ready: list[bool] = field(default_factory=lambda: [False])
  torch_op: list[Any] = field(default_factory=lambda: [None])


_registrations: dict[tuple[str, str], _BridgeRegistration] = {}
_registration_lock = Lock()
_materialize_lock = Lock()
_loaded_bridge_modules: dict[str, tuple[Any, Any]] = {}
_verified_gomp_handles: dict[str, tuple[Any, dict[str, int]]] = {}

# Replaced by `vecops.workspace_observer` at import; returns a ctypes
# c_void_p to the active VecopsAllocationObserverV1, or NULL.
_workspace_observer_factory: Any = lambda: ctypes.c_void_p(None)
_native_thread_pool: Any = None


def _workspace_observer_pointer() -> Any:
  return _workspace_observer_factory()


def prepare_replay_plan(function, arguments, describe_tensor):
  """Prepare grids from captured metadata, without executing the operator.

  Produced intermediates need no capture-time storage owner: the graph supplies
  metadata and replay binds their actual addresses before entering any grid.
  """
  from ._dtype import normalize_dtype
  key = tuple(function._schema.name.split('::', 1))
  registration = _registrations.get(key)
  if registration is None or not registration.materialized:
    return None
  owner = _loaded_bridge_modules[str(registration.library_path.resolve())][0]
  query = owner.vecops_torch_bridge_pool_v1
  query.argtypes = [ctypes.c_char_p, ctypes.c_char_p]
  query.restype = ctypes.c_void_p
  pool = query(registration.library.encode(), registration.name.encode())
  if not pool:
    raise RuntimeError('registered Torch bridge has no execution provider')
  records, flags, dtypes, outputs = [], [], [], []
  parameters = registration.kernel_def.inputs
  for index, (parameter, value) in enumerate(zip(parameters, arguments)):
    if isinstance(parameter, native.TensorDef):
      access = parameter.access
      flag = 3 if access == native.TensorAccess.inout else 2 if access == native.TensorAccess.output else 1
      flags.append(flag); dtypes.append(0)
      if parameter.output:
        outputs.append(index)
      if value is None:
        records.append(None)
      else:
        metadata = describe_tensor(value)
        records.append(dict(kind='tensor', data=0, dtype=normalize_dtype(metadata.dtype.removeprefix('torch.')),
                            sizes=metadata.shape, strides=metadata.strides, flags=flag))
    else:
      records.append(dict(kind='scalar', dtype=parameter.dtype, value=value))
      flags.append(0); dtypes.append(int(parameter.dtype))
  values = {}
  for name, value in zip(_ordered_values(registration.kernel_def, registration.spec_order), arguments[len(parameters):]):
    if value is not None:
      values[name] = int(value) if registration.kernel_def.values[name] == native.ConstInt else normalize_dtype(value)
  prepared = native.prepare_operator_plan(registration.operator._handle, records, values, pool)
  if prepared is None:
    return None
  return prepared, flags, dtypes, outputs if registration.return_outputs else []


def _thread_pool_pointer(torch: Any, qualified_name: str) -> ctypes.c_void_p:
  """Resolve the opt-in process-local vecops pool once per interpreter."""
  global _native_thread_pool
  mode = os.environ.get("VECOPS_THREAD_POOL", "torch").strip().lower()
  if mode in ("", "torch"):
    return ctypes.c_void_p(None)
  if mode != "native":
    raise RuntimeError(f"VECOPS_THREAD_POOL must be 'torch' or 'native', got {mode!r}")
  configured_ops = os.environ.get("VECOPS_NATIVE_THREAD_POOL_OPS")
  if configured_ops:
    patterns = [item.strip() for item in configured_ops.split(",") if item.strip()]
    if not any(fnmatch.fnmatchcase(qualified_name, pattern) for pattern in patterns):
      return ctypes.c_void_p(None)
  if _native_thread_pool is None:
    configured = os.environ.get("VECOPS_NUM_THREADS")
    threads = int(configured) if configured else int(torch.get_num_threads())
    if threads < 1:
      raise RuntimeError("VECOPS_NUM_THREADS must be a positive integer")
    _native_thread_pool = ctypes.c_void_p(native.native_thread_pool_address(threads))
  return _native_thread_pool


_identifier = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


def _require_identifier(value: str, what: str) -> None:
  """Reject names that cannot safely become generated C++ identifiers."""
  if not _identifier.fullmatch(value) or keyword.iskeyword(value):
    raise ValueError(f"{what} must be a C++ identifier, got {value!r}")


_dtype_abi = {
  "bool": "VECOPS_DTYPE_BOOL",
  "int8": "VECOPS_DTYPE_I8",
  "uint8": "VECOPS_DTYPE_U8",
  "int16": "VECOPS_DTYPE_I16",
  "uint16": "VECOPS_DTYPE_U16",
  "int32": "VECOPS_DTYPE_I32",
  "uint32": "VECOPS_DTYPE_U32",
  "int64": "VECOPS_DTYPE_I64",
  "uint64": "VECOPS_DTYPE_U64",
  "float16": "VECOPS_DTYPE_F16",
  "bfloat16": "VECOPS_DTYPE_BF16",
  "float32": "VECOPS_DTYPE_F32",
  "float64": "VECOPS_DTYPE_F64",
}


def _scalar_cpp(dtype: native.DType) -> str:
  name = _dtype_name(dtype)
  if name == "bool":
    return "bool"
  if name.startswith("float") or name == "bfloat16":
    return "double"
  return "int64_t"


def _scalar_schema(dtype: native.DType) -> str:
  name = _dtype_name(dtype)
  if name == "bool":
    return "bool"
  if name.startswith("float") or name == "bfloat16":
    return "float"
  return "int"


def _schema_default(value: Any) -> str:
  if isinstance(value, bool):
    return "True" if value else "False"
  return repr(value)


def _ordered_values(definition: native.KernelDef, spec_order: Sequence[str] | None) -> list[str]:
  """Return spec names in declaration order, validating completeness."""
  if spec_order is None:
    return list(definition.values)
  declared = dict(definition.values)
  if set(spec_order) != set(declared) or len(spec_order) != len(declared):
    raise ValueError(f"spec_order must be a permutation of the kernel definition values: {sorted(declared)}")
  return list(spec_order)


def _torch_schema(
  name: str,
  definition: native.KernelDef,
  *,
  return_outputs: bool = False,
  spec_order: Sequence[str] | None = None,
) -> str:
  """Lower a KernelDef to a mutable out-style Torch dispatcher schema.

  ``spec_order`` controls the positional order of the trailing
  ``__spec_*`` arguments; the native KernelDef stores values in a
  name-ordered map, so the declaration order must be threaded explicitly
  to keep positional spec passing aligned with the kernel definition.
  """
  arguments: list[str] = []
  outputs: list[str] = []
  alias = ord("a")
  parameters = definition.inputs
  for index, parameter in enumerate(parameters):
    _require_identifier(parameter.name, "kernel parameter name")
    if isinstance(parameter, native.TensorDef):
      if parameter.output and parameter.optional:
        raise ValueError("Torch bridge output tensors must be present and caller-allocated")
      if parameter.output:
        tensor_type = f"Tensor({chr(alias)}!)"
        # The public wrapper returns the caller-owned output argument.  The
        # internal Torch op only needs a Tensor result so older Inductor
        # custom-op wrappers do not reject a void return.  Keep the result
        # unannotated: AOT functionalization cannot represent a custom-op
        # output that aliases a mutable argument, while mutation tracking is
        # already carried by the argument's (a!) annotation.
        outputs.append("Tensor")
        alias += 1
      else:
        tensor_type = "Tensor"
      if parameter.optional:
        tensor_type += "?"
      arguments.append(f"{tensor_type} {parameter.name}")
    else:
      item = f"{_scalar_schema(parameter.dtype)} {parameter.name}"
      if parameter.default is not None:
        item += f"={_schema_default(parameter.default.value)}"
      arguments.append(item)
  for symbol in _ordered_values(definition, spec_order):
    symbol_type = definition.values[symbol]
    _require_identifier(symbol, "specialization value name")
    torch_type = "int?" if symbol_type == native.ConstInt else "ScalarType?"
    arguments.append(f"{torch_type} __spec_{symbol}=None")
  returns = "()"
  if return_outputs:
    if len(outputs) == 1:
      returns = outputs[0]
    elif outputs:
      returns = f"({', '.join(outputs)})"
  return f"{name}({', '.join(arguments)}) -> {returns}"


_DYNAMIC_TORCH_BRIDGE_TEMPLATE = Template(r"""// Generated by vecops; do not edit.
#include <ATen/ATen.h>
#include <ATen/Parallel.h>
#include <c10/util/ParallelGuard.h>
#include <torch/library.h>
#include <algorithm>
#include <array>
#include <atomic>
#include <cstdint>
#include <cstring>
#include <exception>
#include <memory>
#include <mutex>
#include <optional>
#include <string>
#include <tuple>
#include <unordered_map>
#include <vector>
#include "vecops/runtime/OperatorBridgeAbi.h"
namespace {
using Invoke = decltype(&vecops_operator_bridge_invoke_v1);
using GompParallel = void (*)(void (*)(void*), void*, unsigned int, unsigned int);
using OmpInt = int (*)();
struct State;
struct Work {
  State* state;
  uint32_t count;
  void* context;
  VecopsParallelTaskFn body;
  std::atomic_flag failed = ATOMIC_FLAG_INIT;
  std::exception_ptr exception;
};
struct State {
  uint64_t handle;
  Invoke invoke;
  std::string schema;
  std::vector<std::string> specs;
  const VecopsThreadPoolV1* pool;
  const VecopsAllocationObserverV1* observer;
  GompParallel gomp;
  OmpInt get_max, get_num, get_id, is_nested;
  VecopsThreadPoolV1 torch_pool;
  State(uint64_t h, Invoke i, std::string q, std::vector<std::string> s, const VecopsThreadPoolV1* p,
        const VecopsAllocationObserverV1* o, GompParallel g, OmpInt mx, OmpInt n, OmpInt id, OmpInt ip);
};
uint32_t max_threads(void* p) {
  auto* s = static_cast<State*>(p);
  return static_cast<uint32_t>(std::max(1, s->gomp ? s->get_max() : at::get_num_threads()));
}
uint32_t in_parallel(void* p) {
  auto* s = static_cast<State*>(p);
  return s->gomp ? (s->is_nested() != 0) : at::in_parallel_region();
}
void work_gomp(void* p) noexcept {
  auto& w = *static_cast<Work*>(p);
  auto* s = w.state;
  const int nt = s->get_num(), tid = s->get_id();
  const int64_t active = std::min<int64_t>(nt, w.count);
  if (tid >= active)
    return;
  const int64_t chunk = (w.count + active - 1) / active;
  const int64_t begin = tid * chunk, end = std::min<int64_t>(w.count, begin + chunk);
  try {
    at::internal::ThreadIdGuard id_guard(tid);
    c10::ParallelGuard parallel_guard(true);
    for (int64_t i = begin; i < end; ++i)
      w.body(w.context, static_cast<uint32_t>(i), w.count);
  } catch (...) {
    if (!w.failed.test_and_set(std::memory_order_relaxed))
      w.exception = std::current_exception();
  }
}
int32_t parallel_for(void* p, uint32_t count, void* context, VecopsParallelTaskFn body, VecopsError* error) {
  auto* s = static_cast<State*>(p);
  if (!body || !count)
    return VECOPS_STATUS_OK;
  try {
    if (s->gomp)
      at::internal::lazy_init_num_threads();
    if (count <= 1 || in_parallel(p) || max_threads(p) <= 1) {
      at::internal::ThreadIdGuard id_guard(0);
      c10::ParallelGuard parallel_guard(true);
      for (uint32_t i = 0; i < count; ++i)
        body(context, i, count);
    } else if (s->gomp) {
      Work w{s, count, context, body};
      s->gomp(&work_gomp, &w, 0, 0);
      if (w.exception)
        std::rethrow_exception(w.exception);
    } else {
      at::parallel_for(0, count, 1, [=](int64_t b, int64_t e) {
        for (auto i = b; i < e; ++i)
          body(context, static_cast<uint32_t>(i), count);
      });
    }
    return VECOPS_STATUS_OK;
  } catch (const std::exception& e) {
    if (error && error->struct_size >= sizeof(VecopsError)) {
      error->code = VECOPS_STATUS_EXECUTION_ERROR;
      size_t n = std::strlen(e.what());
      error->message_required = n + 1;
      if (error->message && error->message_capacity) {
        n = std::min(n, error->message_capacity - 1);
        std::memcpy(error->message, e.what(), n);
        error->message[n] = 0;
      }
    }
    return VECOPS_STATUS_EXECUTION_ERROR;
  } catch (...) {
    return VECOPS_STATUS_EXECUTION_ERROR;
  }
}
State::State(uint64_t h, Invoke i, std::string q, std::vector<std::string> s, const VecopsThreadPoolV1* p,
             const VecopsAllocationObserverV1* o, GompParallel g, OmpInt mx, OmpInt n, OmpInt id, OmpInt ip)
  : handle(h)
  , invoke(i)
  , schema(std::move(q))
  , specs(std::move(s))
  , pool(p)
  , observer(o)
  , gomp(g)
  , get_max(mx)
  , get_num(n)
  , get_id(id)
  , is_nested(ip)
  , torch_pool{sizeof(VecopsThreadPoolV1),
               VECOPS_THREAD_POOL_ABI_MAJOR,
               VECOPS_THREAD_POOL_ABI_MINOR,
               UINT64_C(0x746f726368),
               this,
               &max_threads,
               &in_parallel,
               &parallel_for,
               nullptr,
               nullptr,
               0} {
}
uint32_t dtype_to_abi(c10::ScalarType d) {
  switch (d) {
  case c10::ScalarType::Bool:
    return VECOPS_DTYPE_BOOL;
  case c10::ScalarType::Char:
    return VECOPS_DTYPE_I8;
  case c10::ScalarType::Byte:
    return VECOPS_DTYPE_U8;
  case c10::ScalarType::Short:
    return VECOPS_DTYPE_I16;
  case c10::ScalarType::UInt16:
    return VECOPS_DTYPE_U16;
  case c10::ScalarType::Int:
    return VECOPS_DTYPE_I32;
  case c10::ScalarType::UInt32:
    return VECOPS_DTYPE_U32;
  case c10::ScalarType::Long:
    return VECOPS_DTYPE_I64;
  case c10::ScalarType::UInt64:
    return VECOPS_DTYPE_U64;
  case c10::ScalarType::Half:
    return VECOPS_DTYPE_F16;
  case c10::ScalarType::BFloat16:
    return VECOPS_DTYPE_BF16;
  case c10::ScalarType::Float:
    return VECOPS_DTYPE_F32;
  case c10::ScalarType::Double:
    return VECOPS_DTYPE_F64;
  default:
    TORCH_CHECK(false, "unsupported Torch dtype ", d);
  }
}
VecopsValue tensor_value(const at::Tensor* t, uint64_t flags) {
  VecopsValue v{};
  v.struct_size = sizeof(v);
  if (!t) {
    v.kind = VECOPS_VALUE_NONE;
    return v;
  }
  TORCH_CHECK(t->defined() && t->device().is_cpu() && t->layout() == c10::Layout::Strided,
              "vecops bridge requires defined strided CPU tensors");
  v.kind = VECOPS_VALUE_TENSOR;
  v.value.tensor = VecopsTensorView{sizeof(VecopsTensorView),
                                    (flags & VECOPS_TENSOR_WRITE) ? const_cast<at::Tensor*>(t)->mutable_data_ptr()
                                                                  : const_cast<void*>(t->const_data_ptr()),
                                    0,
                                    VECOPS_DEVICE_CPU,
                                    t->device().index(),
                                    dtype_to_abi(t->scalar_type()),
                                    static_cast<uint32_t>(t->dim()),
                                    t->sizes().data(),
                                    t->strides().data(),
                                    flags};
  return v;
}
VecopsValue scalar_value(int64_t x, uint32_t d) {
  VecopsValue v{};
  v.struct_size = sizeof(v);
  v.kind = VECOPS_VALUE_SCALAR;
  v.value.scalar.struct_size = sizeof(VecopsScalar);
  v.value.scalar.dtype = d;
  v.value.scalar.value.i64 = x;
  return v;
}
VecopsValue uscalar_value(uint64_t x, uint32_t d) {
  VecopsValue v{};
  v.struct_size = sizeof(v);
  v.kind = VECOPS_VALUE_SCALAR;
  v.value.scalar.struct_size = sizeof(VecopsScalar);
  v.value.scalar.dtype = d;
  v.value.scalar.value.u64 = x;
  return v;
}
VecopsValue fscalar_value(double x, uint32_t d) {
  VecopsValue v{};
  v.struct_size = sizeof(v);
  v.kind = VECOPS_VALUE_SCALAR;
  v.value.scalar.struct_size = sizeof(VecopsScalar);
  v.value.scalar.dtype = d;
  v.value.scalar.value.f64 = x;
  return v;
}
VecopsSpecializationArgument int_spec(const char* n, int64_t x) {
  VecopsSpecializationArgument v{};
  v.struct_size = sizeof(v);
  v.kind = VECOPS_SPECIALIZATION_CONST_INT;
  v.name = n;
  v.value.integer = x;
  return v;
}
VecopsSpecializationArgument dtype_spec(const char* n, uint32_t x) {
  VecopsSpecializationArgument v{};
  v.struct_size = sizeof(v);
  v.kind = VECOPS_SPECIALIZATION_DTYPE;
  v.name = n;
  v.value.dtype = x;
  return v;
}
$typed_functors
std::mutex registry_mutex;
std::unordered_map<std::string, std::shared_ptr<State>> states;
std::vector<std::unique_ptr<torch::Library>> libraries;
} // namespace
extern "C" VECOPS_RUNTIME_EXPORT int32_t vecops_torch_bridge_register_v2(
  const char* ns, const char* name, const char* schema, uint64_t handle, Invoke invoke, const char* const* specs,
  uint32_t count, const VecopsThreadPoolV1* pool, const VecopsAllocationObserverV1* observer, void* gp, void* mx,
  void* nt, void* id, void* ip, char* err, size_t cap) {
  try {
    TORCH_CHECK(ns && name && schema && invoke, "invalid bridge registration arguments");
    std::string key = std::string(ns) + "::" + name;
    std::lock_guard<std::mutex> l(registry_mutex);
    const bool any_gomp = gp || mx || nt || id || ip;
    TORCH_CHECK(!any_gomp || (gp && mx && nt && id && ip), "incomplete packaged GOMP function table");
    std::vector<std::string> names;
    for (uint32_t i = 0; i < count; ++i) {
      TORCH_CHECK(specs && specs[i], "missing specialization name");
      names.emplace_back(specs[i]);
    }
    const auto parallel = reinterpret_cast<GompParallel>(gp);
    const auto get_max = reinterpret_cast<OmpInt>(mx);
    const auto get_num = reinterpret_cast<OmpInt>(nt);
    const auto get_id = reinterpret_cast<OmpInt>(id);
    const auto is_nested = reinterpret_cast<OmpInt>(ip);
    auto it = states.find(key);
    if (it != states.end()) {
      const auto& old = *it->second;
      TORCH_CHECK(old.handle == handle && old.invoke == invoke && old.schema == schema && old.specs == names &&
                    old.pool == pool && old.observer == observer && old.gomp == parallel && old.get_max == get_max &&
                    old.get_num == get_num && old.get_id == get_id && old.is_nested == is_nested,
                  "registration key already owns a different operator handle or bridge context");
      return VECOPS_STATUS_OK;
    }
    auto s = std::make_shared<State>(handle, invoke, std::string(schema), std::move(names), pool, observer, parallel,
                                     get_max, get_num, get_id, is_nested);
    auto d = std::make_unique<torch::Library>(torch::Library::Kind::FRAGMENT, ns, std::nullopt, __FILE__, __LINE__);
    d->def(schema);
    auto c =
      std::make_unique<torch::Library>(torch::Library::Kind::IMPL, ns, c10::DispatchKey::CPU, __FILE__, __LINE__);
    c->impl(name, CpuFunctor{s});
    auto m =
      std::make_unique<torch::Library>(torch::Library::Kind::IMPL, ns, c10::DispatchKey::Meta, __FILE__, __LINE__);
    m->impl(name, MetaFunctor{});
    states.emplace(key, std::move(s));
    libraries.emplace_back(std::move(d));
    libraries.emplace_back(std::move(c));
    libraries.emplace_back(std::move(m));
    return VECOPS_STATUS_OK;
  } catch (const std::exception& e) {
    if (err && cap) {
      size_t n = std::min(std::strlen(e.what()), cap - 1);
      std::memcpy(err, e.what(), n);
      err[n] = 0;
    }
    return VECOPS_STATUS_EXECUTION_ERROR;
  } catch (...) {
    return VECOPS_STATUS_EXECUTION_ERROR;
  }
}
extern "C" VECOPS_RUNTIME_EXPORT const VecopsThreadPoolV1* vecops_torch_bridge_pool_v1(
    const char* ns, const char* name) {
  if (!ns || !name) return nullptr;
  std::lock_guard<std::mutex> lock(registry_mutex);
  auto found = states.find(std::string(ns) + "::" + name);
  if (found == states.end()) return nullptr;
  auto& state = *found->second;
  return state.pool ? state.pool : &state.torch_pool;
}
""")


def _generate_torch_source(
  library: str,
  name: str,
  definition: native.KernelDef,
  *,
  return_outputs: bool = False,
  spec_order: Sequence[str] | None = None,
  parallel_backend: str = "aten",
) -> str:
  """Generate the complete bridge translation unit without writing it to disk."""
  parameters = definition.inputs
  cpp_parameters: list[str] = []
  call_append: list[str] = []
  for index, parameter in enumerate(parameters):
    local = f"p{index}"
    if isinstance(parameter, native.TensorDef):
      access = parameter.access
      flags = (
        "VECOPS_TENSOR_READ | VECOPS_TENSOR_WRITE"
        if access == native.TensorAccess.inout
        else "VECOPS_TENSOR_WRITE"
        if access == native.TensorAccess.output
        else "VECOPS_TENSOR_READ"
      )
      if parameter.optional:
        cpp_parameters.append(f"std::optional<at::Tensor> {local}")
        call_append.append(f"  values[{index}] = tensor_value({local}.has_value() ? &*{local} : nullptr, {flags});")
      else:
        tensor_type = "at::Tensor&" if parameter.output else "const at::Tensor&"
        cpp_parameters.append(f"{tensor_type} {local}")
        call_append.append(f"  values[{index}] = tensor_value(&{local}, {flags});")
    else:
      cpp_parameters.append(f"{_scalar_cpp(parameter.dtype)} {local}")
      dtype = _dtype_abi[_dtype_name(parameter.dtype)]
      kind = _dtype_name(parameter.dtype)
      if kind == "bool" or kind.startswith("uint"):
        storage = f"uscalar_value({local}, {dtype})"
      elif kind.startswith("int"):
        storage = f"scalar_value({local}, {dtype})"
      else:
        storage = f"fscalar_value({local}, {dtype})"
      call_append.append(f"  values[{index}] = {storage};")

  spec_append: list[str] = []
  for index, symbol in enumerate(_ordered_values(definition, spec_order)):
    symbol_type = definition.values[symbol]
    local = f"s{index}"
    if symbol_type == native.ConstInt:
      cpp_parameters.append(f"std::optional<int64_t> {local}")
      spec_append.append(f"  if ({local}) specs[spec_count++] = int_spec(state->specs[{index}].c_str(), *{local});")
    else:
      cpp_parameters.append(f"std::optional<c10::ScalarType> {local}")
      spec_append.append(
        f"  if ({local}) specs[spec_count++] = dtype_spec(state->specs[{index}].c_str(), dtype_to_abi(*{local}));"
      )

  output_locals = [
    f"p{index}"
    for index, parameter in enumerate(parameters)
    if isinstance(parameter, native.TensorDef) and parameter.output
  ]
  cpp_return_type = "void"
  return_statement = ""
  if return_outputs and len(output_locals) == 1:
    cpp_return_type = "at::Tensor"
    return_statement = f"  return {output_locals[0]};"
  elif return_outputs and output_locals:
    cpp_return_type = "std::tuple<" + ", ".join("at::Tensor" for _ in output_locals) + ">"
    return_statement = f"  return std::make_tuple({', '.join(output_locals)});"

  if parallel_backend not in {"aten", "gomp"}:
    raise ValueError(f"unsupported Torch bridge parallel backend {parallel_backend!r}")

  cpu_body = "\n".join(
    [
      "    std::array<VecopsValue, $parameter_count> values{};".replace("$parameter_count", str(len(parameters))),
      *call_append,
      f"    std::array<VecopsSpecializationArgument, {len(definition.values)}> specs{{}};",
      "    uint32_t spec_count=0;",
      *spec_append,
      "    VecopsExecutionContext ctx{}; ctx.struct_size=sizeof(ctx);",
      "    ctx.thread_pool=state->pool ? state->pool : &state->torch_pool;",
      "    ctx.requested_threads=ctx.thread_pool->max_parallelism(ctx.thread_pool->context); ctx.allocation_observer=state->observer;",
      "    VecopsCall call{sizeof(VecopsCall),static_cast<uint32_t>(values.size()),values.data(),nullptr,0,&ctx};",
      "    std::array<char,1024> msg{}; VecopsError error{sizeof(VecopsError),0,msg.data(),msg.size(),0};",
      "    auto status=state->invoke(state->handle,&call,spec_count,specs.data(),&error);",
      "    TORCH_CHECK(status==VECOPS_STATUS_OK,msg.data());",
      return_statement,
    ]
  )
  typed_functors = f"""
struct CpuFunctor {{ std::shared_ptr<State> state;
  {cpp_return_type} operator()({", ".join(cpp_parameters)}) const {{
{cpu_body}
  }}
}};
struct MetaFunctor {{ {cpp_return_type} operator()({", ".join(cpp_parameters)}) const {{
{return_statement}
  }} }};
"""
  return _DYNAMIC_TORCH_BRIDGE_TEMPLATE.substitute(typed_functors=typed_functors)


def _public_wrapper(
  library: str,
  name: str,
  definition: native.KernelDef,
  operator: native.Operator,
  compiler: Any,
  registration: _BridgeRegistration,
  ready: list[bool],
  torch_op_holder: list[Any],
  spec_order: Sequence[str] | None = None,
):
  """Wrap raw ``torch.ops`` positional specializations in the public call form.

  ``spec_order`` is the kernel-definition declaration order of the spec
  values; it defines the positional order of the trailing spec arguments
  of both the generated Torch op schema and this wrapper.
  """
  from . import _precompile as _precompile_mod

  parameters = list(definition.inputs)
  symbols = _ordered_values(definition, spec_order)
  output_indices = [
    index for index, parameter in enumerate(parameters) if isinstance(parameter, native.TensorDef) and parameter.output
  ]
  n_parameters = len(parameters)
  n_specs = len(symbols)
  spec_names = frozenset(symbols)
  single_output_index = output_indices[0] if len(output_indices) == 1 else -1

  def invoke(*args, **kwargs):
    # Hot inference path: bridge ready and no collector or graph capture
    # active. Two calling conventions avoid the per-call import, dict
    # build, string formatting and output generator of the general path:
    # inputs only, or inputs followed by every spec value positionally in
    # declaration order.
    if (
      ready[0]
      and _precompile_mod._active_collector is None
      and not kwargs
      and len(args) in (n_parameters, n_parameters + n_specs)
    ):
      torch_op_holder[0](*args)
      if single_output_index >= 0:
        return args[single_output_index]
      return tuple(args[index] for index in output_indices)

    if (
      ready[0]
      and _precompile_mod._active_collector is None
      and len(args) == n_parameters
      and all(keyword in spec_names for keyword in kwargs)
    ):
      torch_op_holder[0](*args, *(kwargs.get(name, None) for name in symbols))
      if single_output_index >= 0:
        return args[single_output_index]
      return tuple(args[index] for index in output_indices)

    positional_specs = None
    if len(args) > len(parameters):
      if len(args) != n_parameters + n_specs:
        raise TypeError(
          f"{library}.{name} expected {n_parameters} kernel arguments "
          f"with at most {n_specs} trailing specialization values"
        )
      args, positional_specs = args[:n_parameters], list(args[n_parameters:])
    ordered = list(args)
    for parameter in parameters[len(ordered) :]:
      if parameter.name in kwargs:
        ordered.append(kwargs.pop(parameter.name))
      elif isinstance(parameter, native.TensorDef) and parameter.optional:
        ordered.append(None)
      elif isinstance(parameter, native.ValueDef) and parameter.default is not None:
        ordered.append(parameter.default.value)
      else:
        raise TypeError(f"missing required argument: {parameter.name}")
    if positional_specs is None:
      spec_values = [kwargs.pop(symbol, None) for symbol in symbols]
    else:
      if any(symbol in kwargs for symbol in symbols):
        raise TypeError("specialization values were passed both positionally and by keyword")
      spec_values = positional_specs
    if kwargs:
      unexpected = next(iter(kwargs))
      raise TypeError(f"unexpected keyword argument: {unexpected}")
    outputs = tuple(ordered[index] for index in output_indices)
    collected = _precompile_mod.maybe_collect_call(
      operator,
      definition,
      ordered,
      dict(zip(symbols, spec_values)),
      name=f"{library}::{name}",
    )
    if not collected:
      if not ready[0]:
        if compiler.cache_mode == native.ArtifactCacheMode.cache_only:
          _load_cached_only_torch_bridges((registration,))
        else:
          compile_pending_torch_bridges()
      torch_op = torch_op_holder[0]
      torch_op(*ordered, *spec_values)
    if len(outputs) == 1:
      return outputs[0]
    return outputs

  invoke.__name__ = name
  invoke.__qualname__ = f"vecops._schema_bridge.{library}.{name}"
  invoke.__doc__ = f"Generated mutable vecops wrapper for {library}::{name}."
  invoke._vecops_operator = operator
  invoke._vecops_torch_op_holder = torch_op_holder
  invoke._vecops_ready = ready
  return invoke


def _bridge_include_root() -> Path:
  include_root = Path(__file__).resolve().parents[2] / "include"
  configured_root = os.environ.get("VECOPS_SOURCE_ROOT")
  if configured_root:
    include_root = Path(configured_root).expanduser().resolve() / "include"
  if (include_root / "vecops" / "runtime" / "OperatorBridgeAbi.h").is_file():
    return include_root
  packaged = Path(__file__).resolve().parent / "_sdk" / "include"
  if (packaged / "vecops" / "runtime" / "OperatorBridgeAbi.h").is_file():
    return packaged
  raise RuntimeError("vecops framework bridge headers were not found")


def _torch_bridge_inputs():
  """Return the content identity inputs shared by build and cache lookup."""
  import torch
  from torch.utils.cpp_extension import include_paths, library_paths

  include_root = _bridge_include_root()
  torch_includes = tuple(Path(path).resolve() for path in include_paths("cpu"))
  torch_libraries = tuple(Path(path).resolve() for path in library_paths("cpu"))
  torch_parallel_info = torch.__config__.parallel_info()
  native_library = Path(native.__file__).resolve()
  torch_identity = (
    torch.__version__,
    torch.compiled_with_cxx11_abi(),
    tuple(map(str, torch_includes)),
    tuple(map(str, torch_libraries)),
    torch_parallel_info,
  )
  from ._identity import file_identity, tree_identity

  torch_identity += (
    tree_identity(include_root),
    tuple(tree_identity(path) for path in torch_includes),
    tuple(tree_identity(path, {".so", ".a", ".dylib"}) for path in torch_libraries),
    file_identity(native_library),
  )
  return torch, include_root, torch_includes, torch_libraries, torch_parallel_info, native_library, torch_identity


def _bridge_cache_base(registration: _BridgeRegistration) -> Path:
  if registration.build_directory is not None:
    return Path(registration.build_directory).expanduser().resolve()
  from ._compiler import default_cache_dir

  override = os.environ.get("VECOPS_TORCH_BRIDGE_DIR")
  return Path(override).expanduser().resolve() if override else default_cache_dir() / "torch-bridges-v5"


def _bridge_signature_identity(registration: _BridgeRegistration) -> str:
  return _torch_schema(
    "vecops_signature",
    registration.kernel_def,
    return_outputs=registration.return_outputs,
    spec_order=registration.spec_order,
  )


def _bridge_artifact_key(
  source: str,
  torch_identity: Any,
  compiler_attestation: Any,
  extra_cflags: Sequence[str],
  signature_identity: str = "",
) -> str:
  bridge_override = (os.environ.get("VECOPS_TORCH_BRIDGE_CC"), os.environ.get("VECOPS_TORCH_BRIDGE_CXX"))
  payload = repr(
    (
      "torch-bridge-v7",
      source,
      signature_identity,
      torch_identity,
      compiler_attestation,
      tuple(extra_cflags),
      bridge_override,
    )
  )
  return hashlib.sha256(payload.encode()).hexdigest()[:24]


def _load_cached_only_torch_bridges(registrations: Sequence[_BridgeRegistration]) -> None:
  """Load existing bridges without creating directories or discovering tools."""
  with _materialize_lock:
    pending = [registration for registration in registrations if not registration.materialized]
    if not pending:
      return
    torch, _, _, _torch_libraries, _torch_parallel_info, native_library, torch_identity = _torch_bridge_inputs()
    from ._identity import compiler_identity, file_identity

    for registration in pending:
      source = _generate_torch_source(
        registration.library,
        registration.name,
        registration.kernel_def,
        return_outputs=registration.return_outputs,
        spec_order=registration.spec_order,
      )
      signature_identity = _bridge_signature_identity(registration)
      build_identity = compiler_identity(registration.compiler)
      base = _bridge_cache_base(registration)
      key = _bridge_artifact_key(source, torch_identity, build_identity, registration.extra_cflags, signature_identity)
      directory = base / key if registration.build_directory is None else base
      output_name = f"vecops_bridge_{key}"
      manifest = directory / f"{output_name}.identity"
      try:
        manifest_text = manifest.read_text()
        identity = ast.literal_eval(manifest_text)
      except (OSError, SyntaxError, ValueError):
        identity = None
        manifest_text = ""
      if (
        not isinstance(identity, tuple)
        or len(identity) != 4
        or identity[0] != source
        or identity[1] != torch_identity
        or identity[2] != signature_identity
        or not isinstance(identity[3], tuple)
        or len(identity[3]) < 5
        or identity[3][0] != "torch-bridge-v7"
        or identity[3][1] != build_identity
        or identity[3][2] != tuple(registration.extra_cflags)
      ):
        raise ImportError(
          f"cache-only Torch bridge {registration.library}::{registration.name} is not found; "
          "build it once in read-write mode"
        )
      runtime_path = identity[3][3]
      runtime_identity = identity[3][4]
      if runtime_path is not None:
        from ._identity import file_identity

        path = Path(runtime_path)
        if file_identity(path) != runtime_identity:
          raise ImportError("cache-only Torch bridge packaged OpenMP runtime identity changed")
        _verify_gomp_runtime(path)
        registration.gomp_runtime_path = path
      cached = _cached_bridge(directory, output_name, manifest_text)
      if cached is None:
        raise ImportError(
          f"cache-only Torch bridge {registration.library}::{registration.name} artifact is missing; "
          "build it once in read-write mode"
        )
      _load_bridge(registration, cached, native_library, torch)


def _cached_bridge(directory: Path, output_name: str, identity: str) -> Path | None:
  if not directory.is_dir():
    return None
  manifest = directory / f"{output_name}.identity"
  if not manifest.is_file() or manifest.read_text() != identity:
    return None
  candidates = [
    path
    for path in directory.iterdir()
    if path.is_file() and path.suffix in {".so", ".dylib", ".dll"} and output_name in path.name
  ]
  return candidates[0] if len(candidates) == 1 else None


def _toolchain_key(config: native.KernelCompilerConfig) -> tuple[Any, ...]:
  toolchain = config.toolchain
  return (
    str(config.sdk.path),
    config.sdk.layout,
    str(toolchain.cmake_program),
    str(toolchain.c_compiler),
    str(toolchain.cxx_compiler),
    str(toolchain.toolchain_file),
    toolchain.generator,
    toolchain.build_type,
    tuple(sorted(toolchain.environment.items())),
  )


def _copy_toolchain(toolchain: native.ToolchainSpec) -> native.ToolchainSpec:
  copied = native.ToolchainSpec()
  copied.cmake_program = toolchain.cmake_program
  copied.c_compiler = toolchain.c_compiler
  copied.cxx_compiler = toolchain.cxx_compiler
  copied.toolchain_file = toolchain.toolchain_file
  copied.generator = toolchain.generator
  copied.build_type = toolchain.build_type
  copied.parallel_jobs = toolchain.parallel_jobs
  copied.environment = toolchain.environment
  copied.output_mode = toolchain.output_mode
  copied.observer = toolchain.observer
  return copied


def _gnu_compatible_cxxflags(flags: str) -> str:
  """Remove Clang's GCC-install selector while preserving ordinary flags.

  An explicitly requested GNU bridge compiler may be used even when the kernel
  compiler is Clang. In that case the active CXXFLAGS can contain Clang-only
  ``--gcc-install-dir``/``--gcc-toolchain`` selectors alongside essential private libstdc++
  include, library, and rpath flags. Filter only the incompatible selector;
  do not discard the rest of the user's toolchain configuration.
  """
  arguments = shlex.split(flags)
  result: list[str] = []
  skip_value = False
  for argument in arguments:
    if skip_value:
      skip_value = False
      continue
    if argument in {"--gcc-install-dir", "--gcc-toolchain"}:
      skip_value = True
    elif argument.startswith(("--gcc-install-dir=", "--gcc-toolchain=")):
      continue
    else:
      result.append(argument)
  if skip_value:
    raise ValueError("CXXFLAGS has --gcc-install-dir without a path")
  return shlex.join(result)


def _reject_conflicting_openmp_flags(toolchain: native.ToolchainSpec, extra_flags: Sequence[str] = ()) -> None:
  """A direct GOMP adapter must not implicitly compile/link another OpenMP runtime."""
  environment = dict(toolchain.environment)
  all_flags = [*extra_flags]
  for variable in ("CXXFLAGS", "LDFLAGS"):
    value = environment.get(variable, os.environ.get(variable, ""))
    all_flags.extend(shlex.split(value))
  conflicting = [
    flag
    for flag in all_flags
    if (flag.startswith("-fopenmp") and flag != "-fopenmp-simd") or flag in {"-qopenmp", "-openmp"}
  ]
  if conflicting:
    raise RuntimeError(
      "Torch's Clang/GOMP adapter calls the packaged runtime directly and cannot use compiler OpenMP flags "
      f"({', '.join(conflicting)}); remove them from bridge CXXFLAGS/LDFLAGS or explicit bridge flags"
    )


def _torch_openmp_runtime(torch_libraries: tuple[Path, ...], torch_parallel_info: str) -> str | None:
  if "ATen parallel backend: OpenMP" not in torch_parallel_info:
    return None
  detected = {
    "gomp" if runtime == "gomp" else "omp"
    for directory in torch_libraries
    for path in directory.iterdir()
    if path.is_file() and (runtime := _openmp_runtime_kind(path.name)) is not None
  }
  if len(detected) != 1:
    detail = ", ".join(sorted(detected)) or "none"
    raise RuntimeError(
      "Torch reports its OpenMP backend, but its packaged OpenMP runtime "
      f"could not be identified unambiguously (found: {detail})"
    )
  return detected.pop()


@dataclass(frozen=True)
class _TorchBridgeBuildPolicy:
  toolchain: native.ToolchainSpec
  compile_flags: tuple[str, ...] = ()
  link_libraries: tuple[str, ...] = ()
  parallel_backend: str = "aten"
  runtime_path: Path | None = None
  runtime_identity: tuple[str, str] | None = None
  link_options: tuple[str, ...] = ()


def _torch_gomp_library(torch_libraries: tuple[Path, ...]) -> Path:
  candidates = {
    path.resolve()
    for directory in torch_libraries
    for path in directory.iterdir()
    if _is_openmp_runtime_basename(path.name, "gomp") and path.is_file()
  }
  if len(candidates) != 1:
    found = ", ".join(map(str, sorted(candidates))) or "none"
    raise RuntimeError(
      "Torch's GNU OpenMP backend requires exactly one packaged libgomp runtime; "
      f"found: {found}. Install a Torch build with an unambiguous packaged runtime."
    )
  return next(iter(candidates))


def _is_openmp_runtime_basename(name: str, family: str | None = None) -> bool:
  return _openmp_runtime_kind(name) in ({family} if family is not None else {"gomp", "omp", "iomp5"})


def _openmp_runtime_kind(name: str) -> str | None:
  # A wheel may add a hash before the shared-library suffix, while platform
  # sonames may add numeric version components after it. Reject archives and
  # merely similar prefixes instead of treating any matching filename as a
  # candidate runtime.
  for family in ("gomp", "omp", "iomp5"):
    if re.fullmatch(rf"lib{family}(?:-[a-z0-9_]+)?(?:\.so(?:\.\d+)*|\.dylib)", name.lower()):
      return family
  return None


def _openmp_runtime_path_from_maps_line(line: str) -> Path | None:
  fields = line.split(maxsplit=5)
  if len(fields) != 6 or not fields[5].startswith("/"):
    return None
  candidate = Path(fields[5].removesuffix(" (deleted)"))
  # PyTorch wheels hash bundled DSOs into names such as
  # ``libgomp-a49a47f9.so.1.0.0``. Match only a library basename followed by
  # a conventional soname/hash delimiter; don't accept similarly prefixed
  # application libraries (for example ``libgomp_helper.so``).
  if not _is_openmp_runtime_basename(candidate.name):
    return None
  return candidate.resolve()


def _mapped_openmp_runtime_paths() -> set[Path]:
  maps = Path("/proc/self/maps")
  if not maps.is_file():
    raise RuntimeError("cannot verify the loaded Torch OpenMP runtime: /proc/self/maps is unavailable")
  return {
    path for line in maps.read_text().splitlines() if (path := _openmp_runtime_path_from_maps_line(line)) is not None
  }


def _dladdr_library(address: int) -> Path:
  class DlInfo(ctypes.Structure):
    _fields_ = [
      ("dli_fname", ctypes.c_char_p),
      ("dli_fbase", ctypes.c_void_p),
      ("dli_sname", ctypes.c_char_p),
      ("dli_saddr", ctypes.c_void_p),
    ]

  try:
    dladdr = ctypes.CDLL(None).dladdr
  except AttributeError as exc:
    raise RuntimeError("cannot verify OpenMP symbol provenance: dladdr is unavailable") from exc
  dladdr.argtypes = [ctypes.c_void_p, ctypes.POINTER(DlInfo)]
  dladdr.restype = ctypes.c_int
  info = DlInfo()
  if not dladdr(ctypes.c_void_p(address), ctypes.byref(info)) or not info.dli_fname:
    raise RuntimeError("cannot determine the shared library providing a Torch GOMP symbol")
  logical = Path(os.fsdecode(info.dli_fname)).resolve()
  # glibc can retain the requested/original l_name after audit redirection.
  # The VMA containing the FUNCTION ADDRESS proves which inode was mapped.
  if os.environ.get('VECOPS_NUMA_PLACEMENT'):
    from ._numa_loader import is_loader_copy

    for line in Path('/proc/self/maps').read_text().splitlines():
      fields = line.split(maxsplit=5)
      if len(fields) != 6 or 'x' not in fields[1] or not fields[5].startswith('/'):
        continue
      start, end = (int(value, 16) for value in fields[0].split('-'))
      if start <= address < end:
        mapped = Path(fields[5]).resolve()
        if mapped == logical or is_loader_copy(logical, mapped):
          return mapped
        raise RuntimeError(f'OpenMP symbol mapping {mapped} differs from dladdr provenance {logical}')
  return logical


def _verify_gomp_runtime(path: Path) -> tuple[str, str]:
  """Require the already-loaded packaged GOMP ABI and record its exact file identity."""
  if os.name != "posix" or not hasattr(os, "RTLD_NOLOAD"):
    raise RuntimeError("Clang's direct Torch GOMP adapter is currently supported only on ELF/Linux")
  path = path.resolve()
  loaded_runtimes = _mapped_openmp_runtime_paths()
  mapped_path = path
  if path not in loaded_runtimes:
    from ._numa_loader import is_loader_copy

    aliases = [candidate for candidate in loaded_runtimes if is_loader_copy(path, candidate)]
    if len(aliases) == 1:
      mapped_path = aliases[0]
  if mapped_path not in loaded_runtimes:
    found = ", ".join(map(str, sorted(loaded_runtimes))) or "none"
    raise RuntimeError(f"Torch's packaged GNU runtime {path} is not loaded (found OpenMP runtimes: {found})")
  try:
    handle = ctypes.CDLL(str(mapped_path), mode=os.RTLD_LOCAL | os.RTLD_NOW | os.RTLD_NOLOAD)
  except OSError as exc:
    raise RuntimeError(
      f"Torch's packaged OpenMP runtime {path} is not already loaded; refusing to load a second runtime"
    ) from exc
  required = ("GOMP_parallel", "omp_get_max_threads", "omp_get_num_threads", "omp_get_thread_num", "omp_in_parallel")
  missing = [symbol for symbol in required if not hasattr(handle, symbol)]
  if missing:
    raise RuntimeError(
      f"Torch's packaged OpenMP runtime {path} lacks required GNU/standard OpenMP symbols: " + ", ".join(missing)
    )
  addresses = {symbol: ctypes.cast(getattr(handle, symbol), ctypes.c_void_p).value for symbol in required}
  foreign = {symbol: _dladdr_library(address) for symbol, address in addresses.items() if address}
  if len(foreign) != len(required) or any(symbol_path != mapped_path for symbol_path in foreign.values()):
    detail = ", ".join(f"{symbol}={symbol_path}" for symbol, symbol_path in sorted(foreign.items())) or "no symbols"
    raise RuntimeError(f"Torch GOMP symbols do not all resolve to packaged runtime {path}: {detail}")
  import torch

  configured_threads = int(torch.get_num_threads())
  runtime_threads = int(handle.omp_get_max_threads())
  if runtime_threads != configured_threads:
    raise RuntimeError(
      f"Torch reports {configured_threads} intra-op threads but packaged libgomp reports {runtime_threads}; "
      "the OpenMP runtime binding is inconsistent"
    )
  _verified_gomp_handles[str(path)] = (handle, {name: int(address) for name, address in addresses.items()})
  from ._identity import file_identity

  return file_identity(path)


def _torch_bridge_build_policy(
  config: native.KernelCompilerConfig,
  torch_libraries: tuple[Path, ...],
  torch_parallel_info: str,
) -> _TorchBridgeBuildPolicy:
  """Select a bridge adapter whose ABI matches Torch's packaged OpenMP runtime."""
  toolchain = _copy_toolchain(config.toolchain)
  runtime = _torch_openmp_runtime(torch_libraries, torch_parallel_info)
  if runtime is None:
    return _TorchBridgeBuildPolicy(toolchain)

  compiler_name = Path(toolchain.cxx_compiler).name.lower()
  compiler_is_clang = "clang" in compiler_name
  compiler_is_gnu = not compiler_is_clang and ("g++" in compiler_name or "gcc" in compiler_name)
  explicit_cc = os.environ.get("VECOPS_TORCH_BRIDGE_CC")
  explicit_cxx = os.environ.get("VECOPS_TORCH_BRIDGE_CXX")
  if bool(explicit_cc) != bool(explicit_cxx):
    raise RuntimeError("set both VECOPS_TORCH_BRIDGE_CC and VECOPS_TORCH_BRIDGE_CXX for an explicit bridge override")
  if explicit_cc and explicit_cxx:
    bridge_cc, bridge_cxx = shutil.which(explicit_cc), shutil.which(explicit_cxx)
    if bridge_cc is None or bridge_cxx is None:
      raise RuntimeError("the explicitly configured Torch bridge compiler was not found")
    toolchain.c_compiler, toolchain.cxx_compiler = Path(bridge_cc).absolute(), Path(bridge_cxx).absolute()
    compiler_name = Path(toolchain.cxx_compiler).name.lower()
    compiler_is_clang = "clang" in compiler_name
    compiler_is_gnu = not compiler_is_clang and ("g++" in compiler_name or "gcc" in compiler_name)
    if runtime == "gomp" and compiler_is_gnu:
      environment = dict(toolchain.environment)
      flags = environment.get("CXXFLAGS", os.environ.get("CXXFLAGS", ""))
      environment["CXXFLAGS"] = _gnu_compatible_cxxflags(flags)
      toolchain.environment = environment

  if runtime == "gomp" and compiler_is_clang:
    _reject_conflicting_openmp_flags(toolchain)
    runtime_path = _torch_gomp_library(torch_libraries)
    runtime_identity = _verify_gomp_runtime(runtime_path)
    return _TorchBridgeBuildPolicy(
      toolchain,
      link_libraries=(str(runtime_path),),
      parallel_backend="gomp",
      runtime_path=runtime_path,
      runtime_identity=runtime_identity,
      link_options=("-Wl,--no-as-needed",),
    )
  if runtime == "gomp" and compiler_is_gnu:
    return _TorchBridgeBuildPolicy(toolchain, compile_flags=("-fopenmp",))
  if runtime == "gomp":
    raise RuntimeError(
      "Torch packages GNU OpenMP, but the selected bridge compiler is neither Clang nor GNU; "
      "select Clang for the direct packaged-GOMP adapter or explicitly configure a compatible bridge compiler"
    )
  if runtime == "omp" and compiler_is_clang:
    return _TorchBridgeBuildPolicy(toolchain, compile_flags=("-fopenmp=libomp",))
  if runtime == "omp" and compiler_is_gnu:
    raise RuntimeError(
      "Torch packages LLVM/Intel OpenMP, but the selected bridge compiler is GNU; "
      "select Clang or explicitly configure a compatible bridge compiler"
    )
  raise RuntimeError(f"Torch OpenMP runtime {runtime!r} is unsupported by compiler {toolchain.cxx_compiler}")


def _load_bridge(
  registration: _BridgeRegistration,
  library_path: Path,
  native_library: Path,
  torch: Any,
) -> None:
  path_key = str(library_path.resolve())
  loaded = _loaded_bridge_modules.get(path_key)
  if loaded is None:
    torch.ops.load_library(path_key)
    module = ctypes.CDLL(path_key)
    native_module = ctypes.CDLL(str(native_library))
    _loaded_bridge_modules[path_key] = (module, native_module)
  else:
    module, native_module = loaded
  invoke = ctypes.cast(native_module.vecops_operator_bridge_invoke_v1, ctypes.c_void_p)
  gomp_addresses = [ctypes.c_void_p()] * 5
  if registration.gomp_runtime_path is not None:
    runtime = _verified_gomp_handles.get(str(registration.gomp_runtime_path.resolve()))
    if runtime is None:
      raise RuntimeError("Torch GOMP bridge has no verified packaged-runtime function table")
    _, symbols = runtime
    gomp_addresses = [
      ctypes.c_void_p(symbols[name])
      for name in (
        "GOMP_parallel",
        "omp_get_max_threads",
        "omp_get_num_threads",
        "omp_get_thread_num",
        "omp_in_parallel",
      )
    ]
  spec_names = _ordered_values(registration.kernel_def, registration.spec_order)
  encoded_names = [name.encode() for name in spec_names]
  name_array = (ctypes.c_char_p * len(encoded_names))(*encoded_names) if encoded_names else None
  schema = _torch_schema(
    registration.name,
    registration.kernel_def,
    return_outputs=registration.return_outputs,
    spec_order=registration.spec_order,
  ).encode()
  register = module.vecops_torch_bridge_register_v2
  register.argtypes = [
    ctypes.c_char_p,
    ctypes.c_char_p,
    ctypes.c_char_p,
    ctypes.c_uint64,
    ctypes.c_void_p,
    ctypes.POINTER(ctypes.c_char_p),
    ctypes.c_uint32,
    ctypes.c_void_p,
    ctypes.c_void_p,
    *([ctypes.c_void_p] * 5),
    ctypes.c_char_p,
    ctypes.c_size_t,
  ]
  register.restype = ctypes.c_int32
  error = ctypes.create_string_buffer(2048)
  status = register(
    registration.library.encode(),
    registration.name.encode(),
    schema,
    registration.operator._handle,
    invoke,
    name_array,
    len(encoded_names),
    _thread_pool_pointer(torch, f"{registration.library}::{registration.name}"),
    _workspace_observer_pointer(),
    *gomp_addresses,
    error,
    len(error),
  )
  if status != 0:
    raise RuntimeError(
      f"Torch bridge registration failed for {registration.library}::{registration.name}: {error.value.decode(errors='replace')}"
    )
  registration.library_path = library_path
  registration.materialized = True
  packet = getattr(getattr(torch.ops, registration.library), registration.name)
  # Resolve the OverloadPacket to its concrete OpOverload once: packet
  # __call__ re-runs python overload resolution (jit::
  # _get_operation_for_overload_or_packet) on every invocation, which
  # dominates the custom-op boundary cost on slow hosts.
  registration.torch_op[0] = getattr(packet, "default", packet)
  registration.ready[0] = True


def _compile_pending_torch_bridges_impl(
  *,
  parallelism: int | None = None,
  registrations: Sequence[_BridgeRegistration] | None = None,
) -> int:
  """Build every pending Torch bridge through shared CMake batches."""
  if parallelism is not None and (isinstance(parallelism, bool) or parallelism < 1):
    raise ValueError("parallelism must be a positive integer")
  with _materialize_lock:
    with _registration_lock:
      selected = registrations if registrations is not None else _registrations.values()
      pending = [item for item in selected if not item.materialized]
    if not pending:
      return 0

    torch, include_root, torch_includes, torch_libraries, torch_parallel_info, native_library, torch_identity = (
      _torch_bridge_inputs()
    )
    from ._identity import compiler_identity, file_identity, tree_identity

    groups: dict[tuple[Any, ...], list[tuple[_BridgeRegistration, Any]]] = {}
    for registration in pending:
      config = registration.compiler._native_config()
      if parallelism is not None:
        toolchain = config.toolchain
        toolchain.parallel_jobs = parallelism
        config.toolchain = toolchain
      groups.setdefault(_toolchain_key(config), []).append((registration, config))

    built_count = 0
    for group in groups.values():
      first_registration, first_config = group[0]
      bridge_policy = _torch_bridge_build_policy(first_config, torch_libraries, torch_parallel_info)
      bridge_toolchain = bridge_policy.toolchain
      batch = native.KernelBuildBatchRequest()
      attempt: Path | None = None
      tasks = []
      task_registrations = []
      publications = []
      validators = []
      resolved: dict[tuple[str, str], Path] = {}
      planned_artifacts: set[str] = set()
      artifact_aliases: dict[str, list[tuple[str, str]]] = {}

      for index, (registration, config) in enumerate(group):
        registration.gomp_runtime_path = bridge_policy.runtime_path
        if bridge_policy.parallel_backend == "gomp":
          _reject_conflicting_openmp_flags(bridge_toolchain, registration.extra_cflags)
        source = _generate_torch_source(
          registration.library,
          registration.name,
          registration.kernel_def,
          return_outputs=registration.return_outputs,
          spec_order=registration.spec_order,
        )
        signature_identity = _bridge_signature_identity(registration)
        build_attestation = compiler_identity(registration.compiler)
        artifact_key = _bridge_artifact_key(
          source, torch_identity, build_attestation, registration.extra_cflags, signature_identity
        )
        output_name = f"vecops_bridge_{artifact_key}"
        artifact_directory = (
          Path(registration.build_directory).expanduser().resolve()
          if registration.build_directory is not None
          else _bridge_cache_base(registration) / artifact_key
        )
        runtime_path = str(bridge_policy.runtime_path) if bridge_policy.runtime_path is not None else None
        bridge_identity = (
          "torch-bridge-v7",
          build_attestation,
          tuple(registration.extra_cflags),
          runtime_path,
          bridge_policy.runtime_identity,
          bridge_policy.parallel_backend,
        )
        identity = repr((source, torch_identity, signature_identity, bridge_identity))
        cached = _cached_bridge(artifact_directory, output_name, identity)
        if cached is not None:
          resolved[(registration.library, registration.name)] = cached
          continue
        if artifact_key in planned_artifacts:
          # The backend chooses the final filename (including a possible
          # platform library prefix). Resolve aliases after publication.
          artifact_aliases.setdefault(output_name, []).append((registration.library, registration.name))
          continue
        planned_artifacts.add(artifact_key)

        if attempt is None:
          first_registration.compiler.build_dir.mkdir(parents=True, exist_ok=True)
          attempt = Path(
            tempfile.mkdtemp(
              prefix="torch-bridge-batch-",
              dir=first_registration.compiler.build_dir,
            )
          )
          batch.generated_source_directory = attempt / "project"
          batch.build_directory = attempt / "build"

        input_directory = attempt / "inputs" / str(index)
        input_directory.mkdir(parents=True, exist_ok=True)
        source_path = input_directory / "bridge.cpp"
        source_path.write_text(source)
        request = native.KernelBuildRequest()
        request.target_name = output_name
        request.output_name = output_name
        request.sdk = config.sdk
        request.toolchain = bridge_toolchain
        request.link_runtime = False
        request.sources = [source_path]
        request.include_directories = [include_root, *torch_includes]
        request.compile_definitions = [f"_GLIBCXX_USE_CXX11_ABI={int(torch.compiled_with_cxx11_abi())}"]
        request.compile_options = [
          "-O3",
          "-std=c++20",
          *bridge_policy.compile_flags,
          *registration.extra_cflags,
        ]
        request.link_directories = list(torch_libraries)
        request.link_libraries = ["torch", "torch_cpu", "c10", *bridge_policy.link_libraries]
        request.link_options = [
          *bridge_policy.link_options,
          *bridge_policy.compile_flags,
          *(f"-Wl,-rpath,{path}" for path in torch_libraries),
        ]
        request.generated_source_directory = batch.generated_source_directory / "tasks" / str(index)
        request.build_directory = batch.build_directory / "tasks" / str(index)
        # Never let a partial/racing build overwrite the public cache artifact.
        request.artifact_directory = attempt / "artifacts" / str(index)
        tasks.append(request)
        task_registrations.append(registration)
        publications.append((artifact_directory, output_name, identity))

        def validate(
          registration=registration,
          expected=bridge_identity,
          toolchain=bridge_toolchain,
          policy=bridge_policy,
        ):
          return expected == (
            "torch-bridge-v7",
            compiler_identity(registration.compiler),
            tuple(registration.extra_cflags),
            str(policy.runtime_path) if policy.runtime_path is not None else None,
            policy.runtime_identity,
            policy.parallel_backend,
          )

        validators.append(validate)

      if tasks:
        batch.tasks = tasks
        result = first_registration.compiler.build_backend.compile_batch(batch)
        if not result.success:
          detail = result.error
          if result.configure is not None and result.configure.output:
            detail += f"\n[configure log: {result.configure.log_path}]\n" + result.configure.output[-6000:]
          if result.build is not None and result.build.output:
            detail += f"\n[build log: {result.build.log_path}]\n" + result.build.output[-6000:]
          raise RuntimeError(f"Torch bridge batch compilation failed: {detail}")
        invalid_policies = [index for index, validate in enumerate(validators) if not validate()]
        current_torch_inputs = (
          tree_identity(include_root),
          tuple(tree_identity(path) for path in torch_includes),
          tuple(tree_identity(path, {".so", ".a", ".dylib"}) for path in torch_libraries),
          file_identity(native_library),
        )
        changed_torch_inputs = [
          name
          for name, old, new in zip(
            ("vecops bridge headers", "Torch headers", "Torch libraries", "vecops native library"),
            torch_identity[-4:],
            current_torch_inputs,
          )
          if old != new
        ]
        if invalid_policies or changed_torch_inputs:
          detail = []
          if invalid_policies:
            detail.append(f"changed bridge policy/toolchain for tasks {invalid_policies}")
          if changed_torch_inputs:
            detail.append("changed inputs: " + ", ".join(changed_torch_inputs))
          raise RuntimeError(
            "Torch bridge build inputs changed during compilation (" + "; ".join(detail) + "); retry with stable inputs"
          )
        for registration, task_result, publication in zip(task_registrations, result.tasks, publications):
          if not task_result.success or task_result.kernel_library is None:
            raise RuntimeError(
              f"Torch bridge {registration.library}::{registration.name} "
              f"did not produce a shared library: {task_result.error}"
            )
          artifact_directory, output_name, identity = publication
          artifact_directory.mkdir(parents=True, exist_ok=True)
          library = Path(task_result.kernel_library)
          destination = artifact_directory / library.name
          for target, content in ((destination, None), (artifact_directory / f"{output_name}.identity", identity)):
            descriptor, temporary = tempfile.mkstemp(prefix="bridge-", dir=artifact_directory)
            try:
              os.close(descriptor)
              if content is None:
                shutil.copy2(library, temporary)
              else:
                Path(temporary).write_text(content)
              os.replace(temporary, target)
            finally:
              Path(temporary).unlink(missing_ok=True)
          resolved[(registration.library, registration.name)] = destination
          for alias in artifact_aliases.get(output_name, ()):
            resolved[alias] = destination
          built_count += 1

      for registration, _ in group:
        path = resolved[(registration.library, registration.name)]
        _load_bridge(registration, path, native_library, torch)
    return built_count


def compile_pending_torch_bridges(*, parallelism: int | None = None) -> int:
  """Build pending bridges while coalescing cache writers across processes."""
  if parallelism is not None and (isinstance(parallelism, bool) or parallelism < 1):
    raise ValueError("parallelism must be a positive integer")
  with _registration_lock:
    pending = [item for item in _registrations.values() if not item.materialized]
  if not pending:
    return 0
  cache_only = [item for item in pending if item.compiler.cache_mode == native.ArtifactCacheMode.cache_only]
  buildable = [item for item in pending if item.compiler.cache_mode != native.ArtifactCacheMode.cache_only]
  if cache_only:
    _load_cached_only_torch_bridges(cache_only)
  if not buildable:
    return 0
  from ._compiler import require_native_build_api

  require_native_build_api()
  try:
    import fcntl
  except ImportError:
    return _compile_pending_torch_bridges_impl(parallelism=parallelism, registrations=buildable)
  from ._compiler import default_cache_dir

  override = os.environ.get("VECOPS_TORCH_BRIDGE_DIR")
  lock_root = Path(override).expanduser().resolve() if override else default_cache_dir() / "torch-bridges-v5"
  lock_root.mkdir(parents=True, exist_ok=True)
  with (lock_root / ".compile.lock").open("a+b") as lock:
    fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
    try:
      return _compile_pending_torch_bridges_impl(parallelism=parallelism, registrations=buildable)
    finally:
      fcntl.flock(lock.fileno(), fcntl.LOCK_UN)


def register_torch_operator(
  library: str,
  name: str,
  operator: native.Operator,
  kernel_def: native.KernelDef,
  *,
  compiler: Any,
  build_directory: str | os.PathLike[str] | None = None,
  verbose: bool = False,
  extra_cflags: tuple[str, ...] = (),
  return_outputs: bool = False,
  spec_order: Sequence[str] | None = None,
):
  """Declare one mutable Torch bridge and return its public wrapper.

  Registration is idempotent per ``(library, name)`` in this Python process.
  Registration queues a content-addressed build task. Pending tasks are
  materialized together by :func:`compile_pending_torch_bridges`, or lazily by
  the first real wrapper invocation.
  """
  _require_identifier(library, "Torch library name")
  _require_identifier(name, "Torch operator name")
  key = (library, name)
  with _registration_lock:
    if key in _registrations:
      return _registrations[key].wrapper
    registration = _BridgeRegistration(
      library=library,
      name=name,
      operator=operator,
      kernel_def=kernel_def,
      compiler=compiler,
      build_directory=build_directory,
      verbose=verbose,
      extra_cflags=tuple(extra_cflags),
      return_outputs=return_outputs,
      spec_order=tuple(spec_order) if spec_order is not None else None,
    )
    registration.wrapper = _public_wrapper(
      library,
      name,
      kernel_def,
      operator,
      compiler,
      registration,
      registration.ready,
      registration.torch_op,
      spec_order=registration.spec_order,
    )
    _registrations[key] = registration
    return registration.wrapper


__all__ = ["compile_pending_torch_bridges", "register_torch_operator"]
