// SPDX-FileCopyrightText: 2026 HenryRenYz and vecops contributors
// SPDX-License-Identifier: MIT
#include "MemoryRuntimeBindings.h"
#include "vecops/memory/LinuxRuntime.h"
#include "vecops/memory/LinuxSdma.h"
#include "vecops/memory/HostEvents.h"
#include "vecops/memory/RuntimePolicy.h"
#include "vecops/memory/RuntimeBridge.h"
#include <pybind11/stl.h>
#include <algorithm>
#include <limits>
#include <set>
#include <tuple>

namespace py = pybind11;
namespace {
using namespace vecops::memory::model;
py::object calibration_value(const std::any& value) {
  if (const auto* v=std::any_cast<std::string>(&value)) return py::cast(*v);
  if (const auto* v=std::any_cast<std::int64_t>(&value)) return py::cast(*v);
  if (const auto* v=std::any_cast<double>(&value)) return py::cast(*v);
  if (const auto* v=std::any_cast<bool>(&value)) return py::cast(*v);
  if (const auto* v=std::any_cast<std::vector<std::any>>(&value)) {
    py::list out; for(const auto& item:*v) out.append(calibration_value(item)); return out;
  }
  if (const auto* v=std::any_cast<std::map<std::string,std::any>>(&value)) {
    py::dict out; for(const auto& [key,item]:*v) out[py::str(key)]=calibration_value(item); return out;
  }
  throw std::runtime_error("unsupported calibration database value");
}
py::dict performance_copy(const PerformanceDatabase& database) {
  py::dict out,profiles; py::list bindings,cases;
  for(const auto& [name,value]:database.profiles) profiles[py::str(name)]=calibration_value(value);
  for(const auto& b:database.bindings) {
    py::dict item;
    item["target_kind"]=py::cast(b.target_kind); item["target"]=b.target; item["profile"]=b.profile;
    bindings.append(std::move(item));
  }
  for(const auto& c:database.cases) {
    py::dict item;
    item["method"]=c.method; item["route"]=c.route; item["executor"]=py::cast(c.executor);
    item["operation"]=c.operation; item["profile"]=c.profile; item["sample"]=c.sample;
    cases.append(std::move(item));
  }
  out["bindings"]=bindings; out["profiles"]=profiles; out["cases"]=cases;
  out["method_models"]=py::dict(); // parser rejects unsupported computation models
  return out;
}
template <class T>
T unwrap(Result<T> r) {
  if (!r) {
    const auto& e = std::get<Error>(r.state);
    throw std::runtime_error(e.code + ": " + e.message);
  }
  return std::move(r).value();
}
void unwrap(Result<void> r) {
  if (!r)
    throw std::runtime_error(r.error->code + ": " + r.error->message);
}
template <class T>
void destroy_native(T* pointer) {
  // Last-owner destruction can drain asynchronous work. Python producer threads
  // must be able to run while the native event service waits for quiescence.
  if (Py_IsInitialized() && PyGILState_Check()) {
    py::gil_scoped_release release;
    delete pointer;
  } else
    delete pointer;
}
template <class T>
std::shared_ptr<T> share(std::unique_ptr<T> p) {
  return {p.release(), &destroy_native<T>};
}
template <class T>
std::shared_ptr<T> retain(std::shared_ptr<T> p) {
  auto* raw = p.get();
  return {raw, [owner = std::move(p)](T*) mutable {
            if (Py_IsInitialized() && PyGILState_Check()) {
              py::gil_scoped_release release;
              owner.reset();
            } else
              owner.reset();
          }};
}
struct PythonOwner {
  PyObject* object;
  explicit PythonOwner(py::handle h)
    : object(h.ptr()) {
    Py_INCREF(object);
  }
  ~PythonOwner() {
    if (Py_IsInitialized()) {
      py::gil_scoped_acquire gil;
      Py_DECREF(object);
    }
  }
};
std::shared_ptr<PythonOwner> owner(py::handle h) {
  return std::make_shared<PythonOwner>(h);
}
Result<EventStatus> call_status(const std::shared_ptr<PythonOwner>& fn) {
  py::gil_scoped_acquire gil;
  try {
    return {py::reinterpret_borrow<py::object>(fn->object)().cast<EventStatus>()};
  } catch (const py::error_already_set& e) {
    return {Error{"python.event", e.what()}};
  } catch (const py::cast_error& e) {
    return {Error{"python.event_type", e.what()}};
  }
}
std::vector<CompletionToken> unique_tokens(const std::vector<CompletionToken>& tokens) {
  std::set<std::tuple<std::string, std::string, std::uint64_t>> seen;
  std::vector<CompletionToken> result;
  for (const auto& t : tokens)
    if (seen.emplace(t.id, t.invocation, t.binding_generation).second)
      result.push_back(t);
  return result;
}
struct Prepared {
  std::shared_ptr<PreparedAddresses> native;
  AddressLayout layout;
};
struct AddressBuffer {
  std::shared_ptr<Prepared> owner;
  VirtualAddress address;
  std::uint64_t bytes;
  bool read_only;
};
struct Binding {
  std::shared_ptr<MemoryBinding> native;
  std::shared_ptr<Prepared> prepared;
};
py::buffer_info buffer_info(void* address, std::uint64_t bytes, bool read_only = false) {
  if (bytes > static_cast<std::uint64_t>(std::numeric_limits<py::ssize_t>::max()))
    throw py::value_error("buffer length exceeds Python index range");
  return {address, 1,        py::format_descriptor<unsigned char>::format(), 1, {static_cast<py::ssize_t>(bytes)},
          {1},     read_only};
}
std::vector<MemoryArena*> cache_pointers(const std::vector<std::shared_ptr<MemoryArena>>& caches) {
  std::vector<MemoryArena*> result;
  for (const auto& a : caches) {
    if (!a)
      throw py::value_error("null arena cache");
    result.push_back(a.get());
  }
  return result;
}
MemoryBinding& native(Binding& b) {
  if (!b.native)
    throw py::value_error("closed memory binding");
  return *b.native;
}
} // namespace
void bind_memory_runtime(py::module_& m) {
  using namespace vecops::memory::model;
  py::class_<VirtualAddress>(m, "VirtualAddress")
    .def(py::init<>())
    .def_readwrite("address_space", &VirtualAddress::address_space)
    .def_readwrite("value", &VirtualAddress::value);
  py::class_<LinuxPoolBinding>(m, "LinuxPoolBinding")
    .def(py::init<>())
    .def_readwrite("pool", &LinuxPoolBinding::pool)
    .def_readwrite("numa_nodes", &LinuxPoolBinding::numa_nodes)
    .def_readwrite("address_space", &LinuxPoolBinding::address_space);
  py::class_<LinuxDiscovery>(m, "LinuxDiscovery")
    .def(py::init<>())
    .def_readwrite("graph", &LinuxDiscovery::graph)
    .def_readwrite("scope", &LinuxDiscovery::scope)
    .def_readwrite("pool_bindings", &LinuxDiscovery::pool_bindings)
    .def_readwrite("descriptors", &LinuxDiscovery::descriptors)
    .def_readwrite("pool_available_bytes", &LinuxDiscovery::pool_available_bytes)
    .def_readwrite("inventory", &LinuxDiscovery::inventory);
  py::enum_<LinuxCpuProfile>(m, "LinuxCpuProfile")
    .value("generic_cpu", LinuxCpuProfile::generic_cpu)
    .value("lx2_flat", LinuxCpuProfile::lx2_flat);
  py::class_<LinuxTransferStatistics>(m, "LinuxTransferStatistics")
    .def_readonly("payload_bytes", &LinuxTransferStatistics::payload_bytes)
    .def_readonly("pinned_bytes", &LinuxTransferStatistics::pinned_bytes)
    .def_readonly("registrations", &LinuxTransferStatistics::registrations);
  py::class_<LinuxTransferDiagnostic>(m, "LinuxTransferDiagnostic")
    .def_readonly("token", &LinuxTransferDiagnostic::token)
    .def_readonly("status", &LinuxTransferDiagnostic::status)
    .def_readonly("retained", &LinuxTransferDiagnostic::retained);
  py::class_<LinuxTransferExecutor, std::shared_ptr<LinuxTransferExecutor>>(m, "LinuxTransferExecutor");
  py::class_<LinuxTransferQueue>(m, "LinuxTransferQueue")
    .def(py::init<>())
    .def_readwrite("agent", &LinuxTransferQueue::agent)
    .def_readwrite("controller", &LinuxTransferQueue::controller)
    .def_readwrite("methods", &LinuxTransferQueue::methods)
    .def_readwrite("executor", &LinuxTransferQueue::executor);
  py::class_<LinuxSdmaOptions>(m, "LinuxSdmaOptions")
    .def(py::init<>())
    .def_readwrite("device", &LinuxSdmaOptions::device)
    .def_readwrite("library", &LinuxSdmaOptions::library)
    .def_readwrite("shared_channel", &LinuxSdmaOptions::shared_channel)
    .def_readwrite("chunk_bytes", &LinuxSdmaOptions::chunk_bytes)
    .def_readwrite("page_bytes", &LinuxSdmaOptions::page_bytes);
  m.def("linux_sdma_sdk_compiled", &linux_sdma_sdk_compiled);
  m.def("make_linux_sdma_executor", [](LinuxSdmaOptions o) { return unwrap(make_linux_sdma_executor(std::move(o))); });
  m.def("poll_linux_transfer_quarantine", [] { py::gil_scoped_release release;
    return unwrap(poll_linux_transfer_quarantine()); });
  py::class_<CalibrationControl>(m, "CalibrationControl")
    .def(py::init<>())
    .def_readwrite("cpu_cores", &CalibrationControl::cpu_cores)
    .def_readwrite("implementation", &CalibrationControl::implementation);
  py::class_<LinuxRuntimeOptions>(m, "LinuxRuntimeOptions")
    .def(py::init<>())
    .def_readwrite("participant", &LinuxRuntimeOptions::participant)
    .def_readwrite("group_hard_limits", &LinuxRuntimeOptions::group_hard_limits)
    .def_readwrite("shared_limits", &LinuxRuntimeOptions::shared_limits)
    .def_readwrite("cpu_queues", &LinuxRuntimeOptions::cpu_queues)
    .def_readwrite("cpu_copy_methods", &LinuxRuntimeOptions::cpu_copy_methods)
    .def_readwrite("transfer_queues", &LinuxRuntimeOptions::transfer_queues)
    .def_readwrite("transfer_wait_ms", &LinuxRuntimeOptions::transfer_wait_ms)
    .def_readwrite("runtime_counts", &LinuxRuntimeOptions::runtime_counts)
    .def_readwrite("calibration_controls", &LinuxRuntimeOptions::calibration_controls)
    .def_readwrite("capacity_directory", &LinuxRuntimeOptions::capacity_directory);

  py::class_<SharedBacking, std::shared_ptr<SharedBacking>>(m, "SharedBacking")
    .def_property_readonly("info", [](const SharedBacking& b) { return b.info(); })
    .def_property_readonly("physical_identity",
                           [](const SharedBacking& b) { return unwrap(LinuxPageStore::physical_identity(b)); })
    .def_property_readonly("physical_offset",
                           [](const SharedBacking& b) { return unwrap(LinuxPageStore::physical_offset(b)); });
  py::class_<Allocation, std::shared_ptr<Allocation>>(m, "Allocation", py::buffer_protocol())
    .def_property_readonly("info", [](const Allocation& a) { return a.info(); })
    .def_property_readonly("address",
                           [](const Allocation& a) { return reinterpret_cast<std::uintptr_t>(unwrap(a.data())); })
    .def("export_backing", [](const Allocation& a) { return retain(unwrap(a.export_backing())); })
    .def_buffer([](Allocation& a) { return buffer_info(unwrap(a.data()), a.info().requested_bytes); });
  py::class_<MemoryArena, std::shared_ptr<MemoryArena>>(m, "MemoryArena")
    .def_property_readonly("id", &MemoryArena::id)
    .def_property_readonly("page_bytes", &MemoryArena::page_bytes)
    .def("snapshot", &MemoryArena::snapshot)
    .def("allocate",
         [](MemoryArena& a, AllocateRequest r) {
           py::gil_scoped_release release;
           return share(std::make_unique<Allocation>(unwrap(a.allocate(r))));
         })
    .def("record_use",
         [](MemoryArena& a, Allocation& allocation, CompletionToken t) {
           py::gil_scoped_release release;
           unwrap(a.record_use(allocation, std::move(t)));
         })
    .def("release",
         [](MemoryArena& a, Allocation& allocation, CompletionToken t) {
           py::gil_scoped_release release;
           unwrap(a.release(std::move(allocation), std::move(t)));
         })
    .def(
      "trim",
      [](MemoryArena& a, std::uint64_t bytes) {
        py::gil_scoped_release release;
        return unwrap(a.trim(bytes));
      },
      py::arg("retain_idle_bytes") = 0);
  py::class_<PlanLease, std::shared_ptr<PlanLease>>(m, "PlanLease");
  py::class_<LayoutAgreement, std::shared_ptr<LayoutAgreement>>(m, "LayoutAgreement")
    .def_property_readonly("digest", &LayoutAgreement::digest)
    .def_property_readonly("generation", &LayoutAgreement::generation);
  py::class_<BackingSet, std::shared_ptr<BackingSet>>(m, "BackingSet")
    .def_property_readonly("owned_allocations",
                           [](const BackingSet& b) {
                             auto span = b.owned_allocations();
                             return std::vector<AllocationInfo>(span.begin(), span.end());
                           })
    .def_property_readonly("accessible_backings", &BackingSet::accessible_backings);
  py::class_<VirtualReservation, std::shared_ptr<VirtualReservation>>(m, "VirtualReservation")
    .def_property_readonly("size_bytes", &VirtualReservation::size_bytes)
    .def_property_readonly("page_bytes", &VirtualReservation::page_bytes);
  py::class_<MappedRange, std::shared_ptr<MappedRange>>(m, "MappedRange")
    .def_property_readonly("address",
                           [](const MappedRange& r) { return reinterpret_cast<std::uintptr_t>(unwrap(r.data())); })
    .def_property_readonly("generation", &MappedRange::generation);
  py::class_<AddressBuffer, std::shared_ptr<AddressBuffer>>(m, "AddressBuffer", py::buffer_protocol())
    .def_property_readonly("address", [](const AddressBuffer& b) { return b.address.value; })
    .def_property_readonly("bytes", [](const AddressBuffer& b) { return b.bytes; })
    .def_buffer(
      [](AddressBuffer& b) { return buffer_info(reinterpret_cast<void*>(b.address.value), b.bytes, b.read_only); });
  py::class_<Prepared, std::shared_ptr<Prepared>>(m, "PreparedAddresses")
    .def("slot", [](const Prepared& p, const std::string& id) { return unwrap(p.native->slot(id)); })
    .def(
      "slot_buffer",
      [](std::shared_ptr<Prepared> p, const std::string& id, bool read_only) {
        auto slot =
          std::find_if(p->layout.slots.begin(), p->layout.slots.end(), [&](const auto& s) { return s.id == id; });
        if (slot == p->layout.slots.end())
          throw py::value_error("unknown prepared slot");
        return std::make_shared<AddressBuffer>(
          AddressBuffer{p, unwrap(p->native->slot(id)), slot->size_bytes, read_only});
      },
      py::arg("id"), py::arg("read_only") = false);
  py::class_<Binding, std::shared_ptr<Binding>>(m, "MemoryBinding")
    .def_property_readonly("generation", [](Binding& b) { return native(b).generation(); })
    .def("slot",
         [](Binding& b, const std::string& id) { return reinterpret_cast<std::uintptr_t>(unwrap(native(b).slot(id))); })
    .def("record_use", [](Binding& b, CompletionToken t) {
      py::gil_scoped_release release;
      unwrap(native(b).record_use(std::move(t)));
    });
  py::class_<ExternalMemoryView>(m, "ExternalMemoryView")
    .def(py::init([](std::string id, VirtualAddress address, std::uint64_t bytes, py::object lifetime,
                     std::optional<std::string> pool, bool read_only) {
           if (lifetime.is_none())
             throw py::value_error("external memory requires an owner");
           return ExternalMemoryView{std::move(id),   std::move(address), bytes, std::move(pool),
                                     owner(lifetime), read_only,          {}};
         }),
         py::arg("id"), py::arg("address"), py::arg("bytes"), py::arg("owner"), py::arg("known_pool") = std::nullopt,
         py::arg("read_only") = false)
    .def_readonly("id", &ExternalMemoryView::id)
    .def_readonly("address", &ExternalMemoryView::address)
    .def_readonly("bytes", &ExternalMemoryView::bytes)
    .def_readonly("known_pool", &ExternalMemoryView::known_pool)
    .def_readonly("read_only", &ExternalMemoryView::read_only);
  py::class_<ImportedMemory, std::shared_ptr<ImportedMemory>>(m, "ImportedMemory")
    .def_property_readonly("view", [](const ImportedMemory& a) { return a.view(); })
    .def("endpoint", [](const ImportedMemory& a) { return unwrap(a.endpoint()); })
    .def("describe", [](const ImportedMemory& a, const std::string& symbol) { return unwrap(a.describe(symbol)); });
  py::class_<ExternalBinding>(m, "ExternalBinding")
    .def(py::init<>())
    .def_readwrite("symbol", &ExternalBinding::symbol)
    .def_readwrite("memory", &ExternalBinding::memory);
  py::class_<InvocationContext>(m, "InvocationContext")
    .def(py::init<>())
    .def_readwrite("stamp", &InvocationContext::stamp)
    .def_readwrite("events", &InvocationContext::events)
    .def_readwrite("external", &InvocationContext::external);
  py::class_<EventService, std::shared_ptr<EventService>>(m, "EventService")
    .def("query",
         [](EventService& e, CompletionToken t) {
           py::gil_scoped_release release;
           return unwrap(e.query(t));
         })
    .def("wait",
         [](EventService& e, CompletionToken t) {
           py::gil_scoped_release release;
           return unwrap(e.wait(t));
         })
    .def("join",
         [](EventService& e, std::vector<CompletionToken> t, InvocationContext ctx) {
           auto unique = unique_tokens(t);
           py::gil_scoped_release release;
           return unwrap(e.join(unique, ctx));
         })
    .def("request_cancel",
         [](EventService& e, CompletionToken t) {
           py::gil_scoped_release release;
           unwrap(e.request_cancel(t));
         })
    .def("release",
         [](EventService& e, std::vector<CompletionToken> t) {
           auto unique = unique_tokens(t);
           py::gil_scoped_release release;
           unwrap(e.release(unique));
         })
    .def("complete",
         [](EventService& e, InvocationStamp stamp, EventStatus status) {
           auto* host = dynamic_cast<HostEventService*>(&e);
           if (!host)
             throw py::value_error("event service cannot mint host completions");
           return unwrap(host->complete(stamp, status));
         })
    .def(
      "import_host_event",
      [](EventService& e, py::function query, py::function wait, py::object cancel, py::object lifetime,
         InvocationContext ctx) {
        auto q = owner(query), w = owner(wait);
        HostEventCallbacks callbacks;
        callbacks.query = [q] { return call_status(q); };
        callbacks.wait = [w] { return call_status(w); };
        if (!cancel.is_none()) {
          if (!PyCallable_Check(cancel.ptr()))
            throw py::type_error("cancel must be callable or None");
          auto c = owner(cancel);
          callbacks.cancel = [c]() -> Result<void> {
            py::gil_scoped_acquire gil;
            try {
              py::reinterpret_borrow<py::object>(c->object)();
              return {};
            } catch (const py::error_already_set& error) {
              return {Error{"python.cancel", error.what()}};
            }
          };
        }
        ExternalEvent event{"host-callback", std::move(callbacks), owner(lifetime)};
        py::gil_scoped_release release;
        return unwrap(e.import_event(event, ctx));
      },
      py::arg("query"), py::arg("wait"), py::arg("cancel"), py::arg("owner"), py::arg("context"));

  py::class_<MemoryRuntime, std::shared_ptr<MemoryRuntime>>(m, "MemoryRuntime")
    .def("export_bridge", [](std::shared_ptr<MemoryRuntime> runtime) {
      namespace bridge = vecops::memory::bridge;
      auto exported = bridge::export_runtime(std::move(runtime));
      py::capsule capsule(exported.get(), bridge::capsule_name, [](void* pointer) {
        bridge::ReleaseDescriptor{}(static_cast<bridge::Descriptor*>(pointer));
      });
      exported.release();
      return capsule;
    })
    .def_property_readonly("participant", &MemoryRuntime::participant)
    .def_property_readonly("physical_graph", [](const MemoryRuntime& r) { return r.physical_graph(); })
    .def("performance", [](const MemoryRuntime& r) { return performance_copy(r.performance()); })
    .def_property_readonly("costs", &MemoryRuntime::costs, py::return_value_policy::reference_internal)
    .def_property_readonly("events",
                           [](std::shared_ptr<MemoryRuntime> r) {
                             auto* events = &r->events();
                             return retain(std::shared_ptr<EventService>(std::move(r), events));
                           })
    .def("current_scope", [](MemoryRuntime& r) { return unwrap(r.current_scope()); })
    .def("refresh_available",
         [](MemoryRuntime& r) {
           py::gil_scoped_release release;
           unwrap(r.refresh_available());
         })
    .def("view", [](MemoryRuntime& r, const Scope& s) { return unwrap(r.view(s)); })
    .def("describe_bindings", &MemoryRuntime::describe_bindings)
    .def("capacity_authority", &MemoryRuntime::capacity_authority)
    .def("observe", [](MemoryRuntime& r, const ResourceView& v) { return unwrap(r.observe(v)); })
    .def("observe_external", [](MemoryRuntime& r, const ResourceView& v, const ExternalMemoryView& memory) {
      py::gil_scoped_release release;
      return unwrap(r.observe_external(v,memory));
    })
    .def("arena_ids", &MemoryRuntime::arena_ids)
    .def("arena", [](MemoryRuntime& r, const std::string& id) { return retain(unwrap(r.arena(id))); })
    .def("create_arena",
         [](MemoryRuntime& r, ResourceView v, ArenaOptions options) {
           py::gil_scoped_release release;
           return retain(unwrap(r.create_arena(v, options)));
         })
    .def("allocate",
         [](MemoryRuntime& r, ResourceView v, BackingRequest request) {
           py::gil_scoped_release release;
           return share(std::make_unique<Allocation>(unwrap(r.allocate(v, request))));
         })
    .def(
      "snapshot",
      [](MemoryRuntime& r, ResourceView v, const std::vector<std::shared_ptr<MemoryArena>>& caches) {
        auto ptrs = cache_pointers(caches);
        return r.snapshot(v, ptrs);
      },
      py::arg("view"), py::arg("caches") = std::vector<std::shared_ptr<MemoryArena>>{})
    .def("validate_recipe", [](MemoryRuntime& r, const ResourceView& v,
                               const MemoryRecipe& recipe) { unwrap(r.validate_recipe(v, recipe)); })
    .def(
      "estimate",
      [](MemoryRuntime& r, const ResourceView& v, const MemoryRecipe& recipe,
         const std::vector<std::shared_ptr<MemoryArena>>& caches) {
        auto ptrs = cache_pointers(caches);
        return unwrap(r.estimate(v, recipe, ptrs));
      },
      py::arg("view"), py::arg("recipe"), py::arg("caches") = std::vector<std::shared_ptr<MemoryArena>>{})
    .def(
      "progress",
      [](MemoryRuntime& r, const ResourceView& v, const std::string& method, const std::optional<std::string>& route) {
        return unwrap(r.progress(v, method, route));
      },
      py::arg("view"), py::arg("method"), py::arg("route") = std::nullopt)
    .def("mapping_capabilities",
         [](MemoryRuntime& r, const ResourceView& v, const std::vector<std::string>& pools, BackingMode mode) {
           return unwrap(r.mapping_capabilities(v, pools, mode));
         }, py::arg("view"), py::arg("pools"), py::arg("mode") = BackingMode::shareable_remappable)
    .def(
      "resolve_placement",
      [](MemoryRuntime& r, const ResourceView& v, const PlacementPlan& p, std::uint64_t bytes, BackingMode mode,
         const std::vector<AccessMode>& access) { return unwrap(r.resolve_placement(v, p, bytes, mode, access)); },
      py::arg("view"), py::arg("plan"), py::arg("bytes"), py::arg("mode"), py::arg("access"))
    .def("compatible_overlap",
         [](MemoryRuntime& r, const ResolvedPlacement& a, std::uint64_t ao, const ResolvedPlacement& b,
            std::uint64_t bo, std::uint64_t bytes) { return unwrap(r.compatible_overlap(a, ao, b, bo, bytes)); })
    .def(
      "describe_operands",
      [](MemoryRuntime& r, const ResourceView& v, const MemoryRecipe& recipe,
         const std::vector<MemoryDescriptor>& external) { return unwrap(r.describe_operands(v, recipe, external)); },
      py::arg("view"), py::arg("recipe"), py::arg("external") = std::vector<MemoryDescriptor>{})
    .def("validate_commands", [](MemoryRuntime& r, const ResourceView& v, const MemoryCommandPlan& p,
                                 const OperandEnvironment& e) { unwrap(r.validate_commands(v, p, e)); })
    .def("cost_plan", [](MemoryRuntime& r, const ResourceView& v, const MemoryCommandPlan& p,
                         const OperandEnvironment& e) { return unwrap(r.cost_plan(v, p, e)); })
    .def(
      "reserve",
      [](MemoryRuntime& r, ResourceView v, MemoryRecipe recipe, std::uint64_t epoch,
         const std::vector<std::shared_ptr<MemoryArena>>& caches) {
        auto ptrs = cache_pointers(caches);
        py::gil_scoped_release release;
        return share(unwrap(r.reserve(v, recipe, epoch, ptrs)));
      },
      py::arg("view"), py::arg("recipe"), py::arg("budget_epoch"),
      py::arg("caches") = std::vector<std::shared_ptr<MemoryArena>>{})
    .def("acquire_backing",
         [](MemoryRuntime& r, ResourceView v, MemoryRecipe recipe, PlanLease& lease) {
           py::gil_scoped_release release;
           return retain(unwrap(r.acquire_backing(v, recipe, lease)));
         })
    .def("prepare_addresses",
         [](MemoryRuntime& r, ResourceView v, AddressLayout layout) {
           auto p = std::make_shared<Prepared>();
           p->layout = layout;
           {
             py::gil_scoped_release release;
             p->native = unwrap(r.prepare_addresses(v, layout));
           }
           return p;
         })
    .def("agree_layout",
         [](MemoryRuntime& r, DistributedLayout layout) {
           py::gil_scoped_release release;
           return share(unwrap(r.agree_layout(layout)));
         })
    .def("exchange_backing",
         [](MemoryRuntime& r, LayoutAgreement& agreement, std::shared_ptr<BackingSet> b) {
           py::gil_scoped_release release;
           return retain(unwrap(r.exchange_backing(agreement, std::move(b))));
         })
    .def(
      "bind",
      [](MemoryRuntime& r, ResourceView v, MemoryRecipe recipe, std::shared_ptr<Prepared> p,
         std::shared_ptr<BackingSet> b, std::shared_ptr<LayoutAgreement> agreement) {
        auto binding = std::make_shared<Binding>();
        binding->prepared = p;
        {
          py::gil_scoped_release release;
          binding->native = share(unwrap(r.bind(v, recipe, p->native, std::move(b), agreement.get())));
        }
        return binding;
      },
      py::arg("view"), py::arg("recipe"), py::arg("addresses"), py::arg("backing"), py::arg("agreement") = nullptr)
    .def("quiesce",
         [](MemoryRuntime& r, Binding& b, std::vector<CompletionToken> tokens) {
           py::gil_scoped_release release;
           return unwrap(r.quiesce(native(b), tokens));
         })
    .def("unbind",
         [](MemoryRuntime& r, Binding& b, QuiescenceToken proof) {
           py::gil_scoped_release release;
           unwrap(r.unbind(native(b), proof));
         })
    .def("import_memory",
         [](MemoryRuntime& r, ResourceView v, ExternalMemoryView external) {
           py::gil_scoped_release release;
           return retain(unwrap(r.import_memory(v, external)));
         })
    .def("resolve_operands", [](MemoryRuntime& r, const ResourceView& v, Binding& b,
                                const InvocationContext& ctx) { return unwrap(r.resolve_operands(v, native(b), ctx)); })
    .def("submit",
         [](MemoryRuntime& r, ResourceView v, MemoryCommandPlan plan, Binding& b, InvocationContext ctx) {
           py::gil_scoped_release release;
           return unwrap(r.submit(v, plan, native(b), ctx));
         })
    .def("begin_shared",
         [](MemoryRuntime& r, SharedRange range) {
           py::gil_scoped_release release;
           return unwrap(r.begin_shared(range));
         })
    .def("publish_shared",
         [](MemoryRuntime& r, SharedRange range, CompletionToken token) {
           py::gil_scoped_release release;
           return unwrap(r.publish_shared(range, std::move(token)));
         })
    .def("acquire_shared",
         [](MemoryRuntime& r, SharedRange range) {
           py::gil_scoped_release release;
           return unwrap(r.acquire_shared(range));
         })
    .def("retire_shared",
         [](MemoryRuntime& r, SharedRange range, std::vector<CompletionToken> tokens) {
           py::gil_scoped_release release;
           return unwrap(r.retire_shared(range, tokens));
         })
    .def("reserve_address",
         [](MemoryRuntime& r, std::string space, std::uint64_t bytes, std::uint64_t quantum) {
           py::gil_scoped_release release;
           return share(unwrap(r.reserve_address(space, bytes, quantum)));
         })
    .def("map",
         [](MemoryRuntime& r, VirtualReservation& address, std::shared_ptr<SharedBacking> backing, MapRequest request) {
           py::gil_scoped_release release;
           return share(unwrap(r.map(address, std::move(backing), request)));
         })
    .def("quiesce_mapping",
         [](MemoryRuntime& r, MappedRange& mapping, std::vector<CompletionToken> tokens) {
           py::gil_scoped_release release;
           return unwrap(r.quiesce(mapping, tokens));
         })
    .def("rebind", [](MemoryRuntime& r, MappedRange& mapping, std::shared_ptr<SharedBacking> backing,
                      MapRequest request, QuiescenceToken proof) {
      py::gil_scoped_release release;
      unwrap(r.rebind(mapping, std::move(backing), request, proof));
    });
  m.def(
    "make_linux_runtime",
    [](LinuxDiscovery d, LinuxRuntimeOptions options) {
      py::gil_scoped_release release;
      return share(unwrap(make_linux_runtime(std::move(d), std::move(options))));
    },
    py::arg("discovery"), py::arg("options") = LinuxRuntimeOptions{});
  m.def(
    "discover_linux_runtime",
    [](std::optional<std::string> hardware, LinuxCpuProfile profile, LinuxRuntimeOptions options) {
      py::gil_scoped_release release;
      return share(unwrap(discover_linux_runtime(hardware, profile, std::move(options))));
    },
    py::arg("hardware_file") = std::nullopt, py::arg("profile") = LinuxCpuProfile::generic_cpu,
    py::arg("options") = LinuxRuntimeOptions{});
  m.def(
    "open_runtime",
    [](std::string hardware, std::optional<std::string> policy, std::optional<std::string> calibration,
       LinuxRuntimeOptions options) {
      py::gil_scoped_release release;
      return share(unwrap(open_linux_runtime(hardware, policy, calibration, std::move(options))));
    },
    py::arg("hardware_file"), py::arg("policy_file") = std::nullopt, py::arg("calibration_file") = std::nullopt,
    py::arg("options") = LinuxRuntimeOptions{});
}
