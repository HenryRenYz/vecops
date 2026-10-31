# SPDX-FileCopyrightText: 2026 HenryRenYz and vecops contributors
# SPDX-License-Identifier: MIT
"""Standalone integration tests: python tools/numa-loader/test_loader.py [-v].

Only a C compiler and Python stdlib are needed. Fixtures stay under ~/tmp,
including on shared ARM hosts. An installed module can be selected with
VECOPS_TEST_NUMA_LOADER; otherwise compile the production source with gcc.
"""
import concurrent.futures
import hashlib
import os
from pathlib import Path
import subprocess
import tempfile
import unittest


class LoaderTests(unittest.TestCase):
  @classmethod
  def setUpClass(cls):
    home_tmp = Path.home() / 'tmp'
    home_tmp.mkdir(exist_ok=True)
    cls.tmp = tempfile.TemporaryDirectory(prefix='vecops-numa-loader-test-', dir=home_tmp)
    cls.root = Path(cls.tmp.name)
    cls.cc = os.environ.get('CC', 'gcc')
    cls.module = Path(os.environ.get('VECOPS_TEST_NUMA_LOADER', cls.root / 'libvecops_numa_loader.so'))
    if 'VECOPS_TEST_NUMA_LOADER' not in os.environ:
      subprocess.run([cls.cc, '-O2', '-shared', '-fPIC', '-Wall', '-Wextra', '-Werror',
                      '-fvisibility=hidden', str(Path(__file__).with_name('NumaLoader.c')),
                      '-o', str(cls.module)], check=True)
    cls.nodes = [int(p.name[4:]) for p in Path('/sys/devices/system/node').glob('node[0-9]*')
                 if (p / 'cpulist').read_text().strip()]
    cls.nodes.sort()
    if not cls.nodes:
      raise unittest.SkipTest('no CPU NUMA topology')
    cls.node = str(cls.nodes[0])
    cls.lib = cls.root / 'libprobe.so'
    cls.build_library(cls.lib, 'int value(void) { return 42; }')
    src = cls.root / 'probe.c'
    src.write_text(r'''
#define _GNU_SOURCE
#include <dlfcn.h>
#include <stdio.h>
#include <pthread.h>
#include <stdlib.h>
#include <string.h>
static const char *path;
static void *worker(void *unused) {
  (void)unused;
  for (int i=0;i<50;i++) { void *h=dlopen(path,RTLD_NOW|RTLD_LOCAL); if(!h)abort();
    int (*f)(void)=dlsym(h,"value"); if(!f||f()!=42)abort();dlclose(h); }
  return NULL;
}
int main(int argc,char **argv) {
  path=argv[1];void *h=dlopen(path,RTLD_NOW|RTLD_LOCAL);
  if(!h){fprintf(stderr,"%s\n",dlerror());return 2;}
  int (*f)(void)=dlsym(h,"value");if(!f)return 3;
  Dl_info info;if(!dladdr((void*)f,&info))return 4;
  printf("%s\n%d\n",info.dli_fname,f());fflush(stdout);
  void *h2=dlopen(path,RTLD_NOW|RTLD_LOCAL);if(h2!=h)return 5;
  if(argc>3 && !strcmp(argv[2],"replace")) {
    char saved[4096];snprintf(saved,sizeof(saved),"%s",info.dli_fname);
    dlclose(h2);dlclose(h);
    if(rename(argv[3],path))return 6;
    void *pinned=dlopen(saved,RTLD_NOW|RTLD_LOCAL);
    if(!pinned)return 7;
    int (*old_value)(void)=dlsym(pinned,"value");
    if(!old_value||old_value()!=42)return 8;
    dlclose(pinned);return 0;
  }
  if(argc>2 && !strcmp(argv[2],"threads")) { pthread_t t[8];for(int i=0;i<8;i++)pthread_create(&t[i],NULL,worker,NULL);
    for(int i=0;i<8;i++)pthread_join(t[i],NULL); }
  dlclose(h2);dlclose(h);return 0;
}''')
    cls.probe = cls.root / 'probe'
    subprocess.run([cls.cc, str(src), '-o', str(cls.probe), '-ldl', '-pthread'], check=True)

    # A real PLT call returning 24 bytes exercises AArch64's x8 ABI. dlsym
    # calls alone do not traverse the broken old-glibc audit trampoline.
    cls.build_library(cls.root / 'libprobe_abi.so',
        'struct triple { unsigned long a,b,c; }; '
        'struct triple make_triple(unsigned long x) { return (struct triple){x,x+1,x+2}; }')
    src = cls.root / 'abi.c'
    src.write_text("""
#include <stdio.h>
struct triple { unsigned long a,b,c; };
extern struct triple make_triple(unsigned long);
int main(void) {struct triple v=make_triple(42);printf("%lu %lu %lu\\n",v.a,v.b,v.c);
return v.a!=42||v.b!=43||v.c!=44;}
""")
    cls.abi_probe = cls.root / 'abi'
    subprocess.run([cls.cc, str(src), '-L'+str(cls.root), '-lprobe_abi',
                    '-Wl,-rpath,'+str(cls.root), '-o', str(cls.abi_probe)], check=True)

  @classmethod
  def tearDownClass(cls):
    cls.tmp.cleanup()

  @classmethod
  def build_library(cls, path, source, extra=()):
    path.parent.mkdir(parents=True, exist_ok=True)
    src = path.with_suffix('.c')
    src.write_text(source)
    subprocess.run([cls.cc, '-shared', '-fPIC', str(src), '-o', str(path), *extra], check=True)

  def setUp(self):
    self.cache = self.root / self.id().split('.')[-1]

  def run_probe(self, extra=None, lib=None, threads=False, affinity=None, replace=None):
    env = {k: v for k, v in os.environ.items()
           if not k.startswith('VECOPS_NUMA_') and k != 'LD_AUDIT'}
    env.update(LD_AUDIT=str(self.module), LD_BIND_NOW="1", VECOPS_NUMA_PLACEMENT=self.node,
               VECOPS_NUMA_GROUP=self.node, VECOPS_NUMA_CACHE_DIR=str(self.cache),
               VECOPS_NUMA_INCLUDE='^libprobe', VECOPS_NUMA_VERBOSE='1')
    for k, v in (extra or {}).items():
      if v is None:
        env.pop(k, None)
      else:
        env[k] = v
    cmd = [str(self.probe), str(lib or self.lib), *(['threads'] if threads else [])]
    if replace is not None:
      cmd += ['replace', str(replace)]
    if affinity is not None:
      cmd = ['taskset', '-c', affinity, *cmd]
    p = subprocess.run(cmd, env=env, text=True, capture_output=True, timeout=90)
    self.assertEqual(p.returncode, 0, p.stderr)
    lines = p.stdout.splitlines()
    self.assertEqual(lines[1], '42', p.stderr)
    return Path(lines[0]), p.stderr

  def assert_copy(self, path, source=None):
    source = source or self.lib
    self.assertNotEqual(path, source)
    a, b = source.stat(), path.stat()
    self.assertNotEqual((a.st_dev, a.st_ino), (b.st_dev, b.st_ino))
    digest = hashlib.sha256(source.read_bytes()).hexdigest()
    self.assertEqual(digest, hashlib.sha256(path.read_bytes()).hexdigest())
    self.assertTrue(path.relative_to(self.cache).parts[3].endswith('-'+digest))

  def test_struct_return_plt_and_old_audit_fallback(self):
    for eager in (False, True):
      env = {key: value for key, value in os.environ.items()
             if not key.startswith('VECOPS_NUMA_') and key not in ('LD_AUDIT', 'LD_BIND_NOW')}
      env.update(LD_AUDIT=str(self.module), VECOPS_NUMA_PLACEMENT=self.node,
                 VECOPS_NUMA_GROUP=self.node, VECOPS_NUMA_CACHE_DIR=str(self.cache),
                 VECOPS_NUMA_INCLUDE='^libprobe_abi')
      if eager:
        env['LD_BIND_NOW'] = '1'
      p = subprocess.run([str(self.abi_probe)], env=env, text=True, capture_output=True, timeout=90)
      self.assertEqual(p.returncode, 0, p.stderr)
      self.assertEqual(p.stdout.strip(), '42 43 44')

  def test_copy_and_cache_hit(self):
    path, _ = self.run_probe()
    self.assert_copy(path)
    other, log = self.run_probe()
    self.assertEqual(path, other)
    self.assertIn('hit source=', log)

  def test_group_normalization(self):
    path, _ = self.run_probe({'VECOPS_NUMA_PLACEMENT': f' ; {self.node}, {self.node}; {self.node}; '})
    self.assert_copy(path)
    other, _ = self.run_probe()
    self.assertEqual(path, other)

  def test_bad_groups(self):
    for placement in ('', ';;', '9-2', '99999999999999999999999999', '0-9999999999999999', f'{self.node},4095', f'{self.node},', '-1', '0x1'):
      with self.subTest(placement=placement):
        path, _ = self.run_probe({'VECOPS_NUMA_PLACEMENT': placement})
        self.assertEqual(path, self.lib)

  def test_invalid_neighbor_group_keeps_valid(self):
    path, _ = self.run_probe({'VECOPS_NUMA_PLACEMENT': f'{self.node},4095;{self.node}'})
    self.assert_copy(path)

  def test_bad_explicit_group(self):
    for value in ('', '4095', 'a', f'{self.node},4095'):
      path, _ = self.run_probe({'VECOPS_NUMA_GROUP': value})
      self.assertEqual(path, self.lib)

  def test_filters(self):
    cases = [({'VECOPS_NUMA_INCLUDE': ''}, False),
             ({'VECOPS_NUMA_EXCLUDE': ''}, True),
             ({'VECOPS_NUMA_EXCLUDE': 'probe'}, False),
             ({'VECOPS_NUMA_INCLUDE': 'other'}, False),
             ({'VECOPS_NUMA_INCLUDE': 'other', 'VECOPS_NUMA_INCLUDE_PATH': str(self.root)}, True),
             ({'VECOPS_NUMA_EXCLUDE_PATH': str(self.root)}, False),
             ({'VECOPS_NUMA_INCLUDE': None}, True),
             ({'VECOPS_NUMA_INCLUDE': '[bad'}, False),
             ({'VECOPS_NUMA_EXCLUDE': '[bad'}, False)]
    for extra, copied in cases:
      with self.subTest(extra=extra):
        path, _ = self.run_probe(extra)
        self.assertEqual(path != self.lib, copied)

  def test_concurrent_first_load(self):
    with concurrent.futures.ThreadPoolExecutor(8) as pool:
      paths = list(pool.map(lambda _: self.run_probe()[0], range(16)))
    self.assertEqual(len(set(paths)), 1)
    self.assert_copy(paths[0])
    self.assertFalse(list(self.cache.rglob('.copy-*')))

  def test_multithreaded_dlopen_identity(self):
    path, _ = self.run_probe(threads=True)
    self.assert_copy(path)

  def test_alias_identity(self):
    alias = self.root / 'alias.so'
    alias.symlink_to(self.lib)
    path, _ = self.run_probe(lib=alias)
    self.assert_copy(path)

  def test_source_update(self):
    lib = self.root / 'updated' / 'libprobe.so'
    self.build_library(lib, 'int value(void) { return 42; }')
    path, _ = self.run_probe(lib=lib)
    before = path.read_bytes()
    lib.unlink()
    self.build_library(lib, 'int value(void) { return 42; } int added(void) { return 7; }')
    other, _ = self.run_probe(lib=lib)
    self.assertNotEqual(path, other)
    self.assertEqual(path.read_bytes(), before)
    self.assert_copy(other, lib)

  def test_reopen_old_cache_image_keeps_version_after_source_replace(self):
    original = self.root / 'pinned' / 'libprobe.so'
    replacement = self.root / 'pinned-new' / 'libprobe.so'
    self.build_library(original, 'int value(void) { return 42; }')
    self.build_library(replacement, 'int value(void) { return 84; } int new_symbol(void) { return 84; }')
    before = original.read_bytes()
    path, _ = self.run_probe(lib=original, replace=replacement)
    self.assertNotEqual(path, original)
    self.assertEqual(path.read_bytes(), before)
    self.assertNotEqual(original.read_bytes(), before)

  def test_cache_corruption(self):
    path, _ = self.run_probe()
    path.chmod(0o700)
    path.write_bytes(b'broken')
    other, _ = self.run_probe()
    self.assertEqual(path, other)
    self.assert_copy(other)

  def test_cache_failures(self):
    for path in ('relative', str(self.lib), '/proc/vecops-nonexistent'):
      actual, _ = self.run_probe({'VECOPS_NUMA_CACHE_DIR': path})
      self.assertEqual(actual, self.lib)
    self.cache.mkdir(mode=0o755)
    actual, _ = self.run_probe()
    self.assertEqual(actual, self.lib)

  def test_cache_symlink_rejected(self):
    target = self.root / 'other-private-dir'
    target.mkdir(mode=0o700)
    self.cache.symlink_to(target, target_is_directory=True)
    path, _ = self.run_probe()
    self.assertEqual(path, self.lib)

  def test_origin_dependencies(self):
    dep = self.root / 'origin' / 'deps' / 'libdep.so'
    self.build_library(dep, 'int dep(void) { return 42; }')
    lib = dep.parent.parent / 'libprobe_origin.so'
    self.build_library(lib, 'extern int dep(void); int value(void) { return dep(); }',
                       ('-L'+str(dep.parent), '-ldep', '-Wl,-rpath,$ORIGIN/deps'))
    # Both selected and excluded dependencies must resolve the ORIGINAL tree.
    for include in ('^libprobe', '^(libprobe|libdep)'):
      path, _ = self.run_probe({'VECOPS_NUMA_INCLUDE': include}, lib=lib)
      self.assert_copy(path, lib)

  def test_transitive_origin_rpath(self):
    deps = self.root / 'transitive' / 'deps'
    self.build_library(deps / 'libchild.so', 'int child(void) { return 42; }')
    self.build_library(deps / 'libdep.so',
                       'extern int child(void); int dep(void) { return child(); }',
                       ('-L'+str(deps), '-lchild'))
    lib = deps.parent / 'libprobe_transitive.so'
    self.build_library(lib, 'extern int dep(void); int value(void) { return dep(); }',
                       ('-L'+str(deps), '-ldep', '-Wl,--disable-new-dtags,-rpath,$ORIGIN/deps'))
    for include in ('^libprobe', '^(libprobe|libdep|libchild)'):
      path, _ = self.run_probe({'VECOPS_NUMA_INCLUDE': include}, lib=lib)
      self.assert_copy(path, lib)

  def test_adjacent_resource_exclusion(self):
    lib = self.root / 'resource' / 'libprobe_resource.so'
    self.build_library(lib, r'''
#define _GNU_SOURCE
#include <dlfcn.h>
#include <stdio.h>
#include <string.h>
int value(void) { Dl_info i;dladdr((void*)value,&i); char p[4096];strcpy(p,i.dli_fname);
  strcpy(strrchr(p,'/')+1,"data.txt");FILE *f=fopen(p,"r");int v=0;
  if(f){fscanf(f,"%d",&v);fclose(f);}return v; }
''', ('-ldl',))
    (lib.parent / 'data.txt').write_text('42')
    path, _ = self.run_probe({'VECOPS_NUMA_EXCLUDE': '^libprobe_resource'}, lib=lib)
    self.assertEqual(path, lib)

  def test_affinity_selection(self):
    selected = None
    for node in self.nodes:
      cpus = (Path('/sys/devices/system/node') / f'node{node}' / 'cpulist').read_text().strip()
      available = [c for c in sorted(os.sched_getaffinity(0)) if any(
          int(r.split('-')[0]) <= c <= int(r.split('-')[-1]) for r in cpus.split(','))]
      if available:
        selected = node, available[0]
        break
    self.assertIsNotNone(selected)
    node, first = selected
    path, log = self.run_probe({'VECOPS_NUMA_GROUP': None, 'VECOPS_NUMA_PLACEMENT': str(node)},
                              affinity=str(first))
    self.assert_copy(path)
    self.assertIn('selection=initial-affinity', log)

  def test_other_groups_and_overlap(self):
    if len(self.nodes) < 2:
      self.skipTest('requires two CPU NUMA nodes; also run on ARM')
    a, b = map(str, self.nodes[:2])
    p, _ = self.run_probe({'VECOPS_NUMA_PLACEMENT': f'{a};{b}'})
    q, _ = self.run_probe({'VECOPS_NUMA_PLACEMENT': f'{a};{b}', 'VECOPS_NUMA_GROUP': b})
    self.assert_copy(p); self.assert_copy(q)
    self.assertNotEqual(p.stat().st_ino, q.stat().st_ino)
    path, _ = self.run_probe({'VECOPS_NUMA_PLACEMENT': f'{a},{b};{a}'})
    self.assertEqual(path, self.lib)
    shared, _ = self.run_probe({'VECOPS_NUMA_PLACEMENT': f'{a},{b}', 'VECOPS_NUMA_GROUP': f'{b},{a}'})
    self.assert_copy(shared)
    path, _ = self.run_probe({'VECOPS_NUMA_PLACEMENT': f'{b}'})
    self.assertEqual(path, self.lib)
    path, _ = self.run_probe({'VECOPS_NUMA_PLACEMENT': f'{a};{b}', 'VECOPS_NUMA_GROUP': None})
    self.assertEqual(path, self.lib)


if __name__ == '__main__':
  unittest.main()
