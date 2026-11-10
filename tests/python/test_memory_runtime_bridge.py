# SPDX-FileCopyrightText: 2026 HenryRenYz and vecops contributors
# SPDX-License-Identifier: MIT
"""Runtime ownership crosses independent extension registries through capsules.

The small consumer deliberately uses a different pybind internals namespace in
this test only. Production extensions keep their own normal pybind settings.
The real c10d adapter is also exercised, including a two-process collective.
"""
import gc
import importlib.util
import os
from pathlib import Path
import shlex
import shutil
import subprocess
import sysconfig
import uuid

import pytest


PRODUCER = r'''
#include <pybind11/pybind11.h>
#include "vecops/memory/RuntimeBridge.h"
#include "vecops/memory/LinuxRuntime.h"
namespace py=pybind11;
namespace m=vecops::memory::model;
namespace b=vecops::memory::bridge;
std::weak_ptr<m::MemoryRuntime> latest;
struct Owner {
  std::shared_ptr<m::MemoryRuntime> runtime;
  Owner(const std::string& participant,const std::string& directory) {
    m::LinuxRuntimeOptions options;options.participant=participant;
    if(!directory.empty())options.capacity_directory=directory;
    auto result=m::discover_linux_runtime(std::nullopt,m::LinuxCpuProfile::generic_cpu,options);
    if(!result)throw std::runtime_error(std::get<m::Error>(result.state).message);
    runtime=std::shared_ptr<m::MemoryRuntime>(std::move(result).value());latest=runtime;
  }
  py::capsule capsule(const std::string& invalid) {
    auto value=b::export_runtime(runtime);
    const char* name=b::capsule_name;
    if(invalid=="name")name="unrelated.Runtime.v1";
    else if(invalid=="version")value->bridge_version++;
    else if(invalid=="interface")value->runtime_interface_revision++;
    else if(invalid=="type")value->runtime_type++;
    else if(invalid=="size")value->struct_size--;
    else if(invalid=="abi")value->standard_library_abi="different-standard-library-ABI";
    else if(!invalid.empty())throw std::invalid_argument("unknown test mutation");
    py::capsule result(value.get(),name,[](void* p){b::ReleaseDescriptor{}(static_cast<b::Descriptor*>(p));});
    value.release();return result;
  }
  std::string pool()const {
    auto scope=runtime->current_scope();
    auto view=runtime->view(std::move(scope).value());
    return std::move(view).value().allocatable_pools.at(0);
  }
};
PYBIND11_MODULE(_bridge_producer,m) {
 m.attr("internals_id")=PYBIND11_INTERNALS_ID;
 py::class_<Owner>(m,"Owner").def(py::init<const std::string&,const std::string&>(),
    py::arg("participant")="bridge-test",py::arg("directory")="")
   .def("export_bridge",&Owner::capsule,py::arg("invalid")="")
   .def_property_readonly("pool",&Owner::pool);
 m.def("alive",[]{return !latest.expired();});
}
'''

CONSUMER = r'''
#include <pybind11/pybind11.h>
#include "vecops/memory/RuntimeBridge.h"
namespace py=pybind11;
namespace b=vecops::memory::bridge;
struct Holder {
 std::shared_ptr<vecops::memory::model::MemoryRuntime> runtime;
 explicit Holder(py::handle value) {
   if(!PyCapsule_CheckExact(value.ptr()) || !PyCapsule_IsValid(value.ptr(),b::capsule_name))
     throw py::type_error("expected a named MemoryRuntime bridge capsule");
   runtime=b::acquire(*static_cast<b::Descriptor*>(PyCapsule_GetPointer(value.ptr(),b::capsule_name)));
 }
};
PYBIND11_MODULE(_bridge_consumer,m) {
 m.attr("internals_id")=PYBIND11_INTERNALS_ID;
 py::class_<Holder>(m,"Holder").def(py::init<py::handle>())
   .def_property_readonly("participant",[](const Holder& value){return value.runtime->participant();});
}
'''


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope='module')
def bridge_modules():
    import pybind11
    import torch
    from torch.utils.cpp_extension import include_paths
    from vecops import _C
    root = Path(__file__).resolve().parents[2]
    directory = root/'build/memory-migration/runtime-bridge-tests'/uuid.uuid4().hex
    directory.mkdir(parents=True)
    archive = Path(os.environ.get('VECOPS_MEMORY_LIBRARY',root/'build/python-kernel-plan/src/libvecops_memory.a'))
    if not archive.is_file():
        pytest.fail('set VECOPS_MEMORY_LIBRARY to the qualified memory archive')
    compiler = shlex.split(os.environ.get('CXX','g++'))
    compiler[0] = shutil.which(compiler[0]) or compiler[0]
    extension = sysconfig.get_config_var('EXT_SUFFIX')
    flags = ['-std=c++20','-O0','-shared','-fPIC','-pthread',
             f'-D_GLIBCXX_USE_CXX11_ABI={int(torch.compiled_with_cxx11_abi())}',
             '-I'+str(root/'include'),'-I'+sysconfig.get_path('include')]
    paths = {}
    for name,source,includes,extra in (
        ('_bridge_producer',PRODUCER,[pybind11.get_include()],[]),
        ('_bridge_consumer',CONSUMER,include_paths('cpu'),['-DPYBIND11_INTERNALS_VERSION=12'])):
        cpp = directory/(name+'.cpp'); cpp.write_text(source)
        library = directory/(name+extension)
        command = [*compiler,*flags,*['-I'+path for path in includes],*extra,str(cpp),
                   str(archive),'-ldl','-lrt','-o',str(library)]
        (directory/(name+'-command.txt')).write_text(shlex.join(command)+'\n')
        result = subprocess.run(command,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True)
        (directory/(name+'-build.log')).write_text(result.stdout)
        assert result.returncode == 0, result.stdout
        paths[name] = library
    producer = load_module('_bridge_producer',paths['_bridge_producer'])
    consumer = load_module('_bridge_consumer',paths['_bridge_consumer'])
    assert producer.internals_id != consumer.internals_id
    return producer,consumer,paths,directory


def test_independent_pybind_registries_retain_runtime_after_exporter_and_capsule_die(bridge_modules):
    producer,consumer,*_ = bridge_modules
    owner = producer.Owner()
    capsule = owner.export_bridge()
    holder = consumer.Holder(capsule)
    del owner,capsule
    gc.collect()
    assert producer.alive() and holder.participant == 'bridge-test'
    del holder
    gc.collect()
    assert not producer.alive()


@pytest.mark.parametrize('invalid',['name','version','interface','type','size','abi'])
def test_wrong_capsule_type_revision_or_standard_library_abi_is_rejected(bridge_modules,invalid):
    producer,consumer,*_ = bridge_modules
    owner = producer.Owner()
    with pytest.raises((TypeError,ValueError),match='MemoryRuntime'):
        consumer.Holder(owner.export_bridge(invalid))
    del owner
    gc.collect()
    assert not producer.alive()


def test_runtime_bridge_rejects_python_objects_and_integer_addresses(bridge_modules):
    producer,consumer,*_ = bridge_modules
    owner = producer.Owner()
    for value in (owner,42,None,object()):
        with pytest.raises(TypeError,match='named MemoryRuntime bridge capsule'):
            consumer.Holder(value)


def test_public_runtime_export_has_the_same_cross_registry_contract(bridge_modules):
    from vecops.memory import native
    _,consumer,*_ = bridge_modules
    options = native.LinuxRuntimeOptions();options.participant='public-export'
    runtime = native.discover_linux_runtime(options=options)
    capsule = runtime.export_bridge()
    holder = consumer.Holder(capsule)
    del runtime,capsule
    gc.collect()
    assert holder.participant == 'public-export'


def test_real_torch_backend_retains_runtime_and_executes_collective(bridge_modules):
    import torch
    from vecops import _C
    from vecops.distributed._build import load
    producer,_,_,_ = bridge_modules
    owner = producer.Owner()
    capsule = owner.export_bridge()
    extension = load()
    backend = extension.Backend(torch.distributed.HashStore(),0,1,5000,1,
        int(_C.native_thread_pool_address(1)),capsule,owner.pool,[0],4096,4096,4096,
        False,[],[],True,False)
    del owner,capsule
    gc.collect()
    assert producer.alive()
    source = torch.arange(8.,device='cpu');target=torch.empty_like(source)
    backend.gatherv(target,source,[8]).wait()
    torch.testing.assert_close(target,source,rtol=0,atol=0)
    backend.shutdown()
    del backend
    gc.collect()
    assert not producer.alive()


def _worker(rank, rendezvous, producer_path, authority):
    import torch
    from vecops import _C
    from vecops.distributed._build import load
    producer = load_module('_bridge_producer',Path(producer_path))
    owner = producer.Owner(f'bridge-rank-{rank}',authority)
    backend = load().Backend(torch.distributed.FileStore(rendezvous,2),rank,2,10000,1,
        int(_C.native_thread_pool_address(1)),owner.export_bridge(),owner.pool,[0,1],
        4096,4096,4096,False,[],[],True,False)
    del owner
    gc.collect()
    assert producer.alive()
    source = torch.arange(4.,device='cpu')+rank*4
    target = torch.empty(8,device='cpu')
    backend.gatherv(target,source,[4,4]).wait()
    torch.testing.assert_close(target,torch.arange(8.,device='cpu'),rtol=0,atol=0)
    backend.shutdown()
    del backend
    gc.collect()
    assert not producer.alive()


def test_two_process_torch_backend_uses_owned_runtime_bridge(bridge_modules):
    import torch.multiprocessing as mp
    from vecops.distributed._build import load
    _,_,paths,directory = bridge_modules
    load()  # Build once before spawning; each interpreter loads its own module.
    authority = directory/'authority';authority.mkdir(mode=0o700)
    mp.spawn(_worker,args=(str(directory/'store'),str(paths['_bridge_producer']),str(authority)),nprocs=2,join=True)
