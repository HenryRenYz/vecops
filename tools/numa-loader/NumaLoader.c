// SPDX-FileCopyrightText: 2026 HenryRenYz and vecops contributors
// SPDX-License-Identifier: MIT
/* Optional glibc audit module. Only object search/open callbacks are exported;
 * no symbol bindings, page policy changes, dlopen, or replay interception.
 * All returned strings live until process exit (the loader retains them).
 */
#define _GNU_SOURCE
#include <link.h>
#include <elf.h>
#include <sched.h>
#include <regex.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <stdint.h>
#include <limits.h>
#include <ctype.h>
#include <errno.h>
#include <fcntl.h>
#include <unistd.h>
#include <dirent.h>
#include <sys/stat.h>
#include <sys/statvfs.h>
#include <sys/file.h>
#include <sys/mman.h>
#include <time.h>
#include "Sha256.h"

#define EXPORT __attribute__((visibility("default")))
#define MAX_NODES 4096
#define MAX_GROUPS 128
#define MAX_CPUS 262144
#define WORDS (MAX_NODES / 64)
typedef struct {
  uint64_t w[WORDS];
} Nodes;
typedef struct {
  Nodes nodes;
  int valid;
} Group;
typedef struct {
  regex_t rx;
  int set, empty;
} Filter;
typedef struct Object {
  char original[PATH_MAX], copy[PATH_MAX], tree[PATH_MAX];
  struct stat identity;
  struct Object* next;
} Object;
static Group groups[MAX_GROUPS];
static unsigned ngroups;
static Nodes online, cpu_nodes;
static Filter filters[4]; /* include basename, exclude basename, include path, exclude path */
static char root[PATH_MAX], group_name[MAX_NODES * 5], host[128];
static int enabled, verbose;
static unsigned warnings;
static Object* objects;

static void warn_once(unsigned bit, const char* reason) {
  if (!(warnings & bit)) {
    warnings |= bit;
    dprintf(2, "[vecops-numa-loader] warning: %s; using original where affected\n", reason);
  }
}
static int has(const Nodes* s, unsigned n) {
  return (s->w[n / 64] >> (n % 64)) & 1;
}
static void put(Nodes* s, unsigned n) {
  s->w[n / 64] |= UINT64_C(1) << (n % 64);
}
static int equal(const Nodes* a, const Nodes* b) {
  return !memcmp(a, b, sizeof(*a));
}
static int intersects(const Nodes* a, const Nodes* b) {
  for (int i = 0; i < WORDS; i++)
    if (a->w[i] & b->w[i])
      return 1;
  return 0;
}
static int subset(const Nodes* a, const Nodes* b) {
  for (int i = 0; i < WORDS; i++)
    if (a->w[i] & ~b->w[i])
      return 0;
  return 1;
}
static int nonempty(const Nodes* s) {
  for (int i = 0; i < WORDS; i++)
    if (s->w[i])
      return 1;
  return 0;
}
static void space(const char** p) {
  while (isspace((unsigned char)**p))
    ++*p;
}
static int number(const char** p, unsigned bound, unsigned* v) {
  if (!isdigit((unsigned char)**p))
    return 0;
  unsigned x = 0;
  do {
    unsigned d = *(*p)++ - '0';
    if (x > (bound - 1 - d) / 10)
      return 0;
    x = x * 10 + d;
  } while (isdigit((unsigned char)**p));
  if (x >= bound)
    return 0;
  *v = x;
  return 1;
}
/* Bounded before iterating: neither huge intervals nor integers allocate memory. */
static int parse_nodes(const char* p, Nodes* s) {
  memset(s, 0, sizeof(*s));
  space(&p);
  if (!*p)
    return 0;
  while (*p) {
    unsigned a, b;
    if (!number(&p, MAX_NODES, &a))
      return 0;
    space(&p);
    b = a;
    if (*p == '-') {
      ++p;
      space(&p);
      if (!number(&p, MAX_NODES, &b) || b < a)
        return 0;
      space(&p);
    }
    for (unsigned i = a; i <= b; i++)
      put(s, i);
    if (!*p)
      return 1;
    if (*p++ != ',')
      return 0;
    space(&p);
    if (!*p)
      return 0;
  }
  return 0;
}
static int read_text(const char* path, char* out, size_t n) {
  int fd = open(path, O_RDONLY | O_CLOEXEC);
  if (fd < 0)
    return 0;
  ssize_t r = read(fd, out, n - 1);
  close(fd);
  if (r < 0 || (size_t)r == n - 1)
    return 0;
  out[r] = 0;
  return 1;
}
static int node_cpus(unsigned node, cpu_set_t* aff, size_t bytes, int* hit) {
  char path[128], text[32768];
  snprintf(path, sizeof(path), "/sys/devices/system/node/node%u/cpulist", node);
  if (!read_text(path, text, sizeof(text)))
    return 0;
  const char* p = text;
  space(&p);
  int any = 0;
  while (*p) {
    unsigned a, b;
    if (!number(&p, MAX_CPUS, &a))
      return 0;
    space(&p);
    b = a;
    if (*p == '-') {
      ++p;
      if (!number(&p, MAX_CPUS, &b) || b < a)
        return 0;
      space(&p);
    }
    any = 1;
    if (aff)
      for (unsigned c = a; c <= b; c++)
        if (CPU_ISSET_S(c, bytes, aff))
          *hit = 1;
    if (!*p)
      break;
    if (*p++ != ',')
      return 0;
    space(&p);
  }
  return any;
}
static int choose_group(void) {
  char text[32768];
  if (!read_text("/sys/devices/system/node/online", text, sizeof(text)) || !parse_nodes(text, &online))
    return -1;
  for (unsigned n = 0; n < MAX_NODES; n++)
    if (has(&online, n) && node_cpus(n, NULL, 0, NULL))
      put(&cpu_nodes, n);
  const char* placement = getenv("VECOPS_NUMA_PLACEMENT");
  if (!placement || !*placement)
    return -1;
  if (strlen(placement) > 65536) {
    warn_once(1, "placement too long");
    return -1;
  }
  char* copy = strdup(placement);
  if (!copy)
    return -1;
  char* p = copy;
  while (p) {
    char* next = strchr(p, ';');
    if (next)
      *next++ = 0;
    const char* start = p;
    space(&start);
    if (*start) {
      Nodes set;
      if (!parse_nodes(start, &set) || !subset(&set, &cpu_nodes))
        warn_once(1, "invalid or non-CPU NUMA group");
      else {
        int duplicate = 0;
        for (unsigned i = 0; i < ngroups; i++)
          if (equal(&groups[i].nodes, &set))
            duplicate = 1;
        if (!duplicate) {
          if (ngroups == MAX_GROUPS) {
            free(copy);
            warn_once(1, "too many NUMA groups");
            return -1;
          }
          groups[ngroups++] = (Group){.nodes = set, .valid = 1};
        }
      }
    }
    p = next;
  }
  free(copy);
  /* Compare all pairs, including already-conflicting groups; no order priority. */
  for (unsigned i = 0; i < ngroups; i++)
    for (unsigned j = i + 1; j < ngroups; j++)
      if (intersects(&groups[i].nodes, &groups[j].nodes)) {
        groups[i].valid = groups[j].valid = 0;
        warn_once(2, "overlapping NUMA groups");
      }
  const char* explicit_group = getenv("VECOPS_NUMA_GROUP");
  Nodes requested = {0};
  if (explicit_group) {
    if (!parse_nodes(explicit_group, &requested)) {
      warn_once(4, "invalid explicit group");
      return -1;
    }
    for (unsigned i = 0; i < ngroups; i++)
      if (groups[i].valid && equal(&requested, &groups[i].nodes))
        return (int)i;
    warn_once(4, "explicit group is not a valid placement group");
    return -1;
  }
  size_t bytes = CPU_ALLOC_SIZE(MAX_CPUS);
  cpu_set_t* aff = CPU_ALLOC(MAX_CPUS);
  if (!aff)
    return -1;
  CPU_ZERO_S(bytes, aff);
  if (sched_getaffinity(0, bytes, aff)) {
    CPU_FREE(aff);
    return -1;
  }
  unsigned counted = 0;
  for (unsigned n = 0; n < MAX_NODES; n++)
    if (has(&cpu_nodes, n)) {
      int hit = 0;
      node_cpus(n, aff, bytes, &hit);
      if (hit)
        put(&requested, n);
      /* Check CPUs missing from readable topology as well as cross-group masks. */
      if (hit) {
        char path[128], buf[32768];
        snprintf(path, sizeof(path), "/sys/devices/system/node/node%u/cpulist", n);
        if (read_text(path, buf, sizeof(buf))) {
          const char* q = buf;
          space(&q);
          while (*q) {
            unsigned a, b;
            if (!number(&q, MAX_CPUS, &a))
              break;
            b = a;
            if (*q == '-') {
              ++q;
              if (!number(&q, MAX_CPUS, &b))
                break;
            }
            for (unsigned c = a; c <= b; c++)
              counted += !!CPU_ISSET_S(c, bytes, aff);
            space(&q);
            if (*q != ',')
              break;
            ++q;
          }
        }
      }
    }
  unsigned allowed = CPU_COUNT_S(bytes, aff);
  CPU_FREE(aff);
  if (counted != allowed || !nonempty(&requested))
    return -1;
  for (unsigned i = 0; i < ngroups; i++)
    if (groups[i].valid && subset(&requested, &groups[i].nodes))
      return (int)i;
  return -1;
}
static int setup_filters(void) {
  const char* names[] = {"VECOPS_NUMA_INCLUDE", "VECOPS_NUMA_EXCLUDE", "VECOPS_NUMA_INCLUDE_PATH",
                         "VECOPS_NUMA_EXCLUDE_PATH"};
  for (int i = 0; i < 4; i++) {
    const char* value = getenv(names[i]);
    if (!value)
      continue;
    filters[i].set = 1;
    filters[i].empty = !*value;
    if (*value && regcomp(&filters[i].rx, value, REG_EXTENDED | REG_NOSUB)) {
      warn_once(8, "invalid POSIX regular expression; redirection disabled");
      return 0;
    }
  }
  return 1;
}
static int match(int i, const char* s) {
  return filters[i].set && !filters[i].empty && !regexec(&filters[i].rx, s, 0, NULL, 0);
}
static int selected(const char* path) {
  const char* base = strrchr(path, '/');
  base = base ? base + 1 : path;
  if (match(1, base) || match(3, path))
    return 0;
  return !(filters[0].set || filters[2].set) || match(0, base) || match(2, path);
}
static int private_dir(const char* path) {
  if (mkdir(path, 0700) && errno != EEXIST)
    return 0;
  struct stat s;
  return !lstat(path, &s) && S_ISDIR(s.st_mode) && s.st_uid == geteuid() && !(s.st_mode & 0077);
}
/* Root ancestors may be the user's existing home. Every loader-owned descendant
 * must be a real, user-private directory; no symlinks to shared source files. */
static int root_dirs(char* path) {
  for (char* p = path + 1; *p; p++)
    if (*p == '/') {
      *p = 0;
      if (mkdir(path, 0700) && errno != EEXIST) {
        *p = '/';
        return 0;
      }
      *p = '/';
    }
  return private_dir(path);
}
static int descendants(const char* path) {
  size_t r = strlen(root);
  if (strncmp(root, path, r) || path[r] != '/')
    return 0;
  char buf[PATH_MAX];
  if (strlen(path) >= sizeof(buf))
    return 0;
  strcpy(buf, path);
  for (char* p = buf + r + 1;; p++)
    if (*p == '/' || !*p) {
      char saved = *p;
      *p = 0;
      int ok = private_dir(buf);
      *p = saved;
      if (!ok)
        return 0;
      if (!saved)
        break;
    }
  return 1;
}
static int setup_root(void) {
  char path[PATH_MAX], resolved[PATH_MAX];
  const char* v = getenv("VECOPS_NUMA_CACHE_DIR");
  int n;
  if (v)
    n = snprintf(path, sizeof(path), "%s", v);
  else if ((v = getenv("XDG_CACHE_HOME")) && v[0] == '/')
    n = snprintf(path, sizeof(path), "%s/vecops/numa-loader", v);
  else if ((v = getenv("HOME")) && v[0] == '/')
    n = snprintf(path, sizeof(path), "%s/.cache/vecops/numa-loader", v);
  else
    return 0;
  if (n < 0 || (size_t)n >= sizeof(path) || path[0] != '/' || !root_dirs(path) || !realpath(path, resolved))
    return 0;
  strcpy(root, resolved);
  struct statvfs fs;
  if (statvfs(root, &fs) || (fs.f_flag & ST_NOEXEC))
    return 0;
  /* machine-id alone can be cloned; include the kernel hostname as well. */
  char id[256] = "", name[256] = "";
  read_text("/etc/machine-id", id, sizeof(id));
  gethostname(name, sizeof(name) - 1);
  Sha256 sha;
  sha_init(&sha);
  sha_add(&sha, id, strlen(id));
  sha_add(&sha, name, strlen(name));
  sha_end(&sha, host);
  return 1;
}
EXPORT unsigned int la_version(unsigned int version) {
  if (version == 0)
    return 0;
  verbose = getenv("VECOPS_NUMA_VERBOSE") && !strcmp(getenv("VECOPS_NUMA_VERBOSE"), "1");
  int chosen = choose_group();
  if (chosen < 0)
    return 0;
#if defined(__aarch64__) && LAV_CURRENT < 2
  /* glibc BZ#26643: old AArch64 audit trampolines corrupt the indirect
   * result register x8, even with no PLT callbacks. Eager binding bypasses
   * that trampoline. Never leave an unsafe auditor active on fallback. */
  const char* bind_now = getenv("LD_BIND_NOW");
  if (!bind_now || !*bind_now) {
    warn_once(256, "AArch64 audit ABI v1 needs LD_BIND_NOW=1 (glibc BZ#26643); auditor disabled");
    return 0;
  }
#endif
  size_t at = 0;
  for (unsigned i = 0; i < MAX_NODES; i++)
    if (has(&groups[chosen].nodes, i))
      at += (size_t)snprintf(group_name + at, sizeof(group_name) - at, "%s%u", at ? "," : "", i);
  if (!setup_filters())
    return 0;
  if (!setup_root()) {
    warn_once(16, "cache root is unavailable, non-private, or non-executable");
    return 0;
  }
  enabled = 1;
  if (verbose)
    dprintf(2, "[vecops-numa-loader] group=%s selection=%s cache=%s\n", group_name,
            getenv("VECOPS_NUMA_GROUP") ? "explicit" : "initial-affinity", root);
  return LAV_CURRENT;
}
static int same_version(const struct stat* a, const struct stat* b) {
  return a->st_dev == b->st_dev && a->st_ino == b->st_ino && a->st_size == b->st_size &&
         a->st_mtim.tv_sec == b->st_mtim.tv_sec && a->st_mtim.tv_nsec == b->st_mtim.tv_nsec &&
         a->st_ctim.tv_sec == b->st_ctim.tv_sec && a->st_ctim.tv_nsec == b->st_ctim.tv_nsec;
}
static int file_hash(int fd, char out[65]) {
  unsigned char buf[65536];
  off_t at = 0;
  Sha256 s;
  sha_init(&s);
  for (;;) {
    ssize_t n = pread(fd, buf, sizeof(buf), at);
    if (n < 0) {
      if (errno == EINTR)
        continue;
      return 0;
    }
    if (!n)
      break;
    sha_add(&s, buf, (size_t)n);
    at += n;
  }
  sha_end(&s, out);
  return 1;
}
/* Parse only a native ELF DSO. PT_INTERP marks a PIE executable, not a library.
 * Relative slash-containing DT_NEEDED bypasses the loader's search paths and
 * cannot safely be relocated. $ORIGIN search candidates are handled below.
 */
static int elf_candidate(int fd, off_t size, uint64_t* text_bytes) {
  ElfW(Ehdr) h;
  if (pread(fd, &h, sizeof(h), 0) != sizeof(h) || memcmp(h.e_ident, ELFMAG, SELFMAG) ||
      h.e_ident[EI_CLASS] != (sizeof(void*) == 8 ? ELFCLASS64 : ELFCLASS32) || h.e_ident[EI_DATA] != ELFDATA2LSB ||
      h.e_type != ET_DYN || h.e_phentsize != sizeof(ElfW(Phdr)) || h.e_phnum > 4096)
    return 0;
  if (h.e_phoff > (uint64_t)size || (uint64_t)h.e_phnum * sizeof(ElfW(Phdr)) > (uint64_t)size - h.e_phoff)
    return 0;
  ElfW(Phdr)* p = malloc((size_t)h.e_phnum * sizeof(*p));
  if (!p)
    return 0;
  if (pread(fd, p, (size_t)h.e_phnum * sizeof(*p), h.e_phoff) != (ssize_t)((size_t)h.e_phnum * sizeof(*p))) {
    free(p);
    return 0;
  }
  uint64_t dynoff = 0, dynsize = 0, straddr = 0, strsize = 0, stroff = 0;
  int ok = 1;
  for (unsigned i = 0; i < h.e_phnum; i++) {
    if (p[i].p_type == PT_INTERP)
      ok = 0;
    if (p[i].p_type == PT_LOAD && (p[i].p_flags & PF_X))
      *text_bytes += p[i].p_filesz;
    if (p[i].p_type == PT_DYNAMIC) {
      dynoff = p[i].p_offset;
      dynsize = p[i].p_filesz;
    }
  }
  if (!ok || !dynsize || dynsize > 16 * 1024 * 1024 || dynoff > (uint64_t)size || dynsize > (uint64_t)size - dynoff) {
    free(p);
    return 0;
  }
  ElfW(Dyn)* d = malloc(dynsize);
  if (!d) {
    free(p);
    return 0;
  }
  if (pread(fd, d, dynsize, dynoff) != (ssize_t)dynsize) {
    free(d);
    free(p);
    return 0;
  }
  size_t count = dynsize / sizeof(*d);
  for (size_t i = 0; i < count && d[i].d_tag != DT_NULL; i++) {
    if (d[i].d_tag == DT_STRTAB)
      straddr = d[i].d_un.d_ptr;
    if (d[i].d_tag == DT_STRSZ)
      strsize = d[i].d_un.d_val;
  }
  for (unsigned i = 0; i < h.e_phnum; i++)
    if (p[i].p_type == PT_LOAD && straddr >= p[i].p_vaddr && straddr - p[i].p_vaddr < p[i].p_filesz)
      stroff = p[i].p_offset + (straddr - p[i].p_vaddr);
  free(p);
  if (!stroff || strsize > 64 * 1024 * 1024 || stroff > (uint64_t)size || strsize > (uint64_t)size - stroff) {
    free(d);
    return 0;
  }
  char* str = malloc(strsize);
  if (!str) {
    free(d);
    return 0;
  }
  if (pread(fd, str, strsize, stroff) != (ssize_t)strsize)
    ok = 0;
  for (size_t i = 0; ok && i < count && d[i].d_tag != DT_NULL; i++)
    if (d[i].d_tag == DT_NEEDED) {
      uint64_t off = d[i].d_un.d_val;
      if (off >= strsize || !memchr(str + off, 0, strsize - off)) {
        ok = 0;
        break;
      }
      if (str[off] != '/' && strchr(str + off, '/'))
        ok = 0;
      /* These early runtime objects remain in their original namespace/path. */
    }
  free(str);
  free(d);
  return ok;
}
static uint64_t nanos(void) {
  struct timespec t;
  clock_gettime(CLOCK_MONOTONIC, &t);
  return (uint64_t)t.tv_sec * 1000000000 + t.tv_nsec;
}
static int bounded_lock(int fd) {
  uint64_t until = nanos() + UINT64_C(30000000000);
  while (flock(fd, LOCK_EX | LOCK_NB)) {
    if (errno != EWOULDBLOCK && errno != EINTR)
      return 0;
    if (nanos() >= until)
      return 0;
    struct timespec t = {0, 1000000};
    nanosleep(&t, NULL);
  }
  return 1;
}
static int copy_bytes(int in, int out) {
  char buf[65536];
  off_t at = 0;
  for (;;) {
    ssize_t n = pread(in, buf, sizeof(buf), at);
    if (n < 0) {
      if (errno == EINTR)
        continue;
      return 0;
    }
    if (!n)
      return 1;
    ssize_t done = 0;
    while (done < n) {
      ssize_t w = write(out, buf + done, (size_t)(n - done));
      if (w < 0 && errno == EINTR)
        continue;
      if (w <= 0)
        return 0;
      done += w;
    }
    at += n;
  }
}
static int valid_copy(const char* path, const struct stat* source, const char* digest) {
  int fd = open(path, O_RDONLY | O_NOFOLLOW | O_CLOEXEC);
  if (fd < 0)
    return 0;
  struct stat st;
  char sum[65];
  int ok = !fstat(fd, &st) && S_ISREG(st.st_mode) && st.st_uid == geteuid() && !(st.st_mode & 0022) &&
           st.st_nlink == 1 && st.st_size == source->st_size &&
           !(st.st_dev == source->st_dev && st.st_ino == source->st_ino) && file_hash(fd, sum) && !strcmp(sum, digest);
  close(fd);
  return ok;
}
static Object* prepare(const char* path) {
  int source = open(path, O_RDONLY | O_CLOEXEC | O_NOFOLLOW);
  if (source < 0)
    return NULL;
  struct stat before, after;
  uint64_t exec_bytes = 0;
  if (fstat(source, &before) || !S_ISREG(before.st_mode)) {
    close(source);
    return NULL;
  }
  for (Object* o = objects; o; o = o->next)
    if (!strcmp(o->original, path) && same_version(&o->identity, &before)) {
      close(source);
      return o;
    }
  if (!elf_candidate(source, before.st_size, &exec_bytes)) {
    close(source);
    return NULL;
  }
  uint64_t begin = nanos();
  char digest[65];
  if (!file_hash(source, digest) || fstat(source, &after) || !same_version(&before, &after)) {
    close(source);
    warn_once(32, "source changed while hashing");
    return NULL;
  }
  Object* o = calloc(1, sizeof(*o));
  if (!o) {
    close(source);
    return NULL;
  }
  strcpy(o->original, path);
  o->identity = before;
  char version[PATH_MAX], dir[PATH_MAX], lockpath[PATH_MAX], temp[PATH_MAX];
  int n = snprintf(version, sizeof(version), "%s/v1/%s/%s/%jx-%jx-%jx-%jx.%jx-%jx.%jx-%s", root, host, group_name,
                   (uintmax_t)before.st_dev, (uintmax_t)before.st_ino, (uintmax_t)before.st_size,
                   (uintmax_t)before.st_mtim.tv_sec, (uintmax_t)before.st_mtim.tv_nsec,
                   (uintmax_t)before.st_ctim.tv_sec, (uintmax_t)before.st_ctim.tv_nsec, digest);
  if (n < 0 || (size_t)n >= sizeof(version) || !descendants(version))
    goto fail;
  n = snprintf(o->tree, sizeof(o->tree), "%s/root", version);
  if (n < 0 || (size_t)n >= sizeof(o->tree))
    goto fail;
  n = snprintf(o->copy, sizeof(o->copy), "%s%s", o->tree, path);
  if (n < 0 || (size_t)n >= sizeof(o->copy))
    goto fail;
  strcpy(dir, o->copy);
  *strrchr(dir, '/') = 0;
  if (!descendants(dir))
    goto fail;
  n = snprintf(lockpath, sizeof(lockpath), "%s/.lock", version);
  if (n < 0 || (size_t)n >= sizeof(lockpath))
    goto fail;
  int lock = open(lockpath, O_CREAT | O_RDWR | O_CLOEXEC | O_NOFOLLOW, 0600);
  struct stat ls;
  if (lock < 0)
    goto fail;
  if (fstat(lock, &ls) || !S_ISREG(ls.st_mode) || ls.st_uid != geteuid() || ls.st_nlink != 1 || (ls.st_mode & 0077) ||
      !bounded_lock(lock)) {
    close(lock);
    warn_once(64, "cache lock unavailable or timed out");
    goto fail;
  }
  int hit = valid_copy(o->copy, &before, digest), ok = hit;
  if (!hit) {
    n = snprintf(temp, sizeof(temp), "%s/.copy-%ld-XXXXXX", dir, (long)getpid());
    int out = (n >= 0 && (size_t)n < sizeof(temp)) ? mkostemp(temp, O_CLOEXEC) : -1;
    if (out >= 0) {
      char copied[65];
      struct stat named;
      ok = copy_bytes(source, out) && !fchmod(out, 0500) && file_hash(out, copied) && !strcmp(copied, digest) &&
           !fstat(source, &after) && same_version(&before, &after) && !stat(path, &named) &&
           same_version(&before, &named) && !fsync(out);
      if (close(out))
        ok = 0;
      if (ok)
        ok = !rename(temp, o->copy);
      if (!ok)
        unlink(temp);
    }
  }
  close(lock);
  if (!ok)
    goto fail;
  /* The published cache is complete, immutable by this implementation, and on
   * an executable mount. A changed source is not redirected to an old image. */
  struct stat named_final;
  if (fstat(source, &after) || !same_version(&before, &after) || stat(path, &named_final) ||
      !same_version(&before, &named_final))
    goto fail;
  close(source);
  o->next = objects;
  objects = o;
  if (verbose)
    dprintf(2,
            "[vecops-numa-loader] %s source=%s copy=%s sha256=%s file_bytes=%jd executable_bytes=%ju elapsed_ms=%.3f\n",
            hit ? "hit" : "publish", path, o->copy, digest, (intmax_t)before.st_size, (uintmax_t)exec_bytes,
            (nanos() - begin) / 1e6);
  return o;
fail:
  close(source);
  free(o);
  warn_once(128, "cache preparation failed (permissions, space, integrity, or path length)");
  return NULL;
}
static char* stable_or_original(const char* path, const char* original) {
  char* p = strdup(path);
  return p ? p : (char*)original;
}
static int bootstrap(const char* base) {
  return !strcmp(base, "libvecops_numa_loader.so") || !strcmp(base, "libc.so.6") || !strncmp(base, "ld-linux", 8) ||
         !strncmp(base, "ld-musl", 7) || !strcmp(base, "linux-vdso.so.1");
}
EXPORT char* la_objsearch(const char* name, uintptr_t* cookie, unsigned int flag) {
  if (!enabled || !strchr(name, '/'))
    return (char*)name;
  /* A direct reopen of an immutable cache image pins THAT version, even if
   * its original source has since been replaced. Only dependency-search
   * candidates (RPATH/RUNPATH etc.) need ORIGIN translation below. */
  size_t r = strlen(root);
  if (flag == LA_SER_ORIG && !strncmp(name, root, r) && name[r] == '/')
    return (char*)name;
  char original[PATH_MAX], translated[PATH_MAX];
  const char* candidate = name;
  int translated_path = 0;
  /* The referencing object's ORIGIN is now its cache directory. Translate
   * candidates back to the original absolute tree BEFORE resolving symlinks,
   * even if an excluded dependency has no cache copy. Never implement our own
   * SONAME/RPATH search. glibc still chooses the search order/candidates. */
  (void)cookie;
  /* DT_RPATH is inherited: a child's search can use an ancestor's ORIGIN.
   * The search cookie need not own the cache prefix in that candidate. */
  for (Object* owner = objects; owner; owner = owner->next) {
    size_t len = strlen(owner->tree);
    if (!strncmp(name, owner->tree, len) && name[len] == '/') {
      if (strlen(name + len) >= sizeof(translated))
        return (char*)name;
      strcpy(translated, name + len);
      candidate = translated;
      translated_path = 1;
      break;
    }
  }
  if (!translated_path && !strncmp(name, root, r) && name[r] == '/')
    return (char*)name;
  if (!realpath(candidate, original)) {
    /* Return a stable translated search candidate even if it does not exist;
     * otherwise glibc could search the wrong cached directory. */
    return translated_path ? stable_or_original(translated, name) : (char*)name;
  }
  const char* base = strrchr(original, '/');
  base = base ? base + 1 : original;
  if (bootstrap(base) || !selected(original))
    return translated_path ? stable_or_original(original, name) : (char*)name;
  Object* o = prepare(original);
  if (o)
    return o->copy;
  return translated_path ? stable_or_original(original, name) : (char*)name;
}
EXPORT unsigned int la_objopen(struct link_map* map, Lmid_t lmid, uintptr_t* cookie) {
  (void)lmid;
  *cookie = 0;
  for (Object* o = objects; o; o = o->next)
    if (!strcmp(o->copy, map->l_name)) {
      *cookie = (uintptr_t)o;
      break;
    }
  return 0; /* No per-function/PLT auditing. */
}
