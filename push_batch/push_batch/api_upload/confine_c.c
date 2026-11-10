/* libconfine.c — LD_PRELOAD-based deny-list enforcer for the aidata sandbox.
 *
 * Hooks the libc path-taking entrypoints (open / openat / stat / opendir /
 * unlink / rename / chmod / execve / ...) and returns EACCES when the target
 * path falls under any prefix in the colon-separated AIDATA_CONFINE_DENY
 * environment variable.
 *
 * It ALSO hooks getaddrinfo() and returns EAI_NONAME for any host whose name
 * matches (suffix) an entry in the comma-separated AIDATA_CONFINE_NET_DENY
 * variable — a name-resolution wall so the agent's tools cannot reach the
 * public skill store. This is LD_PRELOAD-only (no mount-ns equivalent) and is
 * a best-effort layer: direct-IP connects, statically-linked binaries, or raw
 * syscalls bypass it (see README §Network deny / What it does NOT cover).
 *
 * It ALSO fakes the wall clock when AIDATA_FAKE_NOW (unix epoch seconds) is
 * set: time() / gettimeofday() / clock_gettime(CLOCK_REALTIME) are shifted by
 * a fixed offset so the agent's `date` matches the mock's MOCK_SERVER_NOW
 * instead of the container's real time. Offset-based (CLOCK_MONOTONIC left
 * alone) so timers/sleeps/timeouts are unaffected. See README §Time faking.
 *
 * Loaded by aidata's shell MCP via:
 *
 *   LD_PRELOAD=/workspace/.aidata/confine/libconfine.so
 *   AIDATA_CONFINE_DENY=/workspace/.aidata:/var/lib/mock-server:...
 *   AIDATA_CONFINE_NET_DENY=skillhub.cn,skillhub-....myqcloud.com
 *
 * Two design decisions worth noting:
 *
 *   1) Path canonicalisation is intentionally limited. We do NOT call
 *      realpath() — it would re-enter our own stat() hook and risk
 *      recursive lock-stepping. Instead we strip the most common bypass
 *      prefix (`/proc/{pid,self}/root/...`) explicitly and rely on the
 *      kernel's own canonicalisation for the actual syscall (i.e. an
 *      attacker crafting `/foo/../workspace/.aidata/x` would still hit
 *      our hooks because libc passes the literal string down — we only
 *      have to defeat the *string* obfuscations that don't go through
 *      `..`-resolution at the libc layer). If that turns out to be too
 *      lax in practice, swap is_denied() for a guarded realpath() with a
 *      thread-local re-entry flag.
 *
 *   2) execve() hook re-injects LD_PRELOAD + AIDATA_CONFINE_DENY into envp.
 *      This closes the `unset LD_PRELOAD; cat /...` bypass: even when bash
 *      strips the variable before exec, our hook re-adds it so the child
 *      process still loads the .so. The .so path itself is recorded at
 *      constructor time via dladdr() — no env-var feedback loop.
 *
 * Everything is fail-CLOSED inside hooks: dlsym failure during init aborts
 * the process. The "no .so → refuse" half of fail-closed is enforced one
 * layer up in shell_server.py (Python checks `os.path.isfile(SO)` before
 * subprocess.run()).
 */

#define _GNU_SOURCE
#include <ctype.h>          /* tolower */
#include <dirent.h>
#include <dlfcn.h>
#include <errno.h>
#include <fcntl.h>
#include <limits.h>
#include <netdb.h>          /* getaddrinfo, struct addrinfo, EAI_* */
#include <sched.h>          /* unshare, CLONE_NEWNS */
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mount.h>      /* mount, MS_* */
#include <sys/stat.h>
#include <sys/time.h>       /* gettimeofday, struct timeval */
#include <sys/types.h>
#include <time.h>           /* time, clock_gettime, struct timespec, CLOCK_* */
#include <unistd.h>


/* ------------------------------------------------------------------------ */
/* State                                                                     */
/* ------------------------------------------------------------------------ */

#define MAX_DENY 64

static char  *g_deny[MAX_DENY];
static size_t g_deny_n[MAX_DENY];
static int    g_deny_count = 0;
static char   g_so_path[PATH_MAX] = "";

/* Time faking: an integer-second offset added to CLOCK_REALTIME /
 * gettimeofday() / time() so the agent's shell sees the mock's
 * MOCK_SERVER_NOW instead of the container wall clock. Activated only when
 * AIDATA_FAKE_NOW (unix epoch seconds) is set; otherwise every wrapper is a
 * pass-through and behaviour is byte-identical to the no-faketime build.
 *
 * OFFSET-based, not frozen: offset = desired_now - real_now is captured once
 * at constructor time, so the faked clock keeps advancing at the real rate
 * from the mock anchor. We deliberately leave CLOCK_MONOTONIC untouched —
 * sleeps, subprocess timeouts, and any CLOCK_REALTIME-deadline polling loop
 * keep working, which a fully frozen REALTIME clock would break (classic
 * libfaketime hang). Over a single rollout the drift vs MockServer's frozen
 * clock.py value is seconds, never a day boundary — so "today's date" stays
 * consistent, which is the whole point. See README §Time faking. */
static int    g_fake_active = 0;
static time_t g_fake_off_sec = 0;
static char   g_fake_now_str[32] = "";

/* Network deny: lowercased host suffixes from AIDATA_CONFINE_NET_DENY. A query
 * host matches if it equals a pattern or ends with "." + pattern (so
 * "skillhub.cn" covers "api.skillhub.cn" but not "evilskillhub.cn"). */
#define MAX_NET_DENY 64
static char  *g_net_deny[MAX_NET_DENY];
static size_t g_net_deny_n[MAX_NET_DENY];
static int    g_net_deny_count = 0;

/* Per-thread re-entry guard: when our own hook calls back into the libc
 * functions we wrap (e.g. fprintf doing fstat on stderr), we must not
 * recursively re-check or we lock-step ourselves into an infinite loop.
 */
static __thread int in_confine = 0;

/* Initialization guard: on glibc 2.33+ systems with libselinux (or similar
 * libraries), their constructors may execute BEFORE ours during dl-init. They
 * call fopen/open/stat etc. which land in our hook wrappers while r_* pointers
 * are still NULL. We cannot simply dlsym on the fly (it may re-enter hooks).
 * Instead, we bypass all deny checks and call the real libc via a late-resolved
 * function pointer: if r_FOO is NULL, use dlsym(RTLD_NEXT, "FOO") inline.
 *
 * g_initialized is set to 1 at the end of confine_init(). Before that, the
 * early-call helpers below handle the fallback.
 */
static volatile int g_initialized = 0;


/* ------------------------------------------------------------------------ */
/* Real symbols                                                              */
/* ------------------------------------------------------------------------ */

/* All function pointers are volatile to prevent the compiler from assuming
 * they are already initialized when another library's constructor calls into
 * our hooks (e.g. libselinux's init_lib → fopen before confine_init runs on
 * some dl-init orderings). Without volatile, -O2 can eliminate the NULL
 * guard or reorder the load relative to the store in confine_init, causing
 * a NULL-pointer call (SIGSEGV) during process startup.
 */

static int   (* volatile r_open)     (const char *, int, ...);
static int   (* volatile r_open64)   (const char *, int, ...);
static int   (* volatile r_openat)   (int, const char *, int, ...);
static int   (* volatile r_openat64) (int, const char *, int, ...);
static int   (* volatile r_creat)    (const char *, mode_t);
static FILE *(* volatile r_fopen)    (const char *, const char *);
static FILE *(* volatile r_fopen64)  (const char *, const char *);

/* glibc fortify alias: when a TU is compiled with -D_FORTIFY_SOURCE>=2
 * and the call site uses 2-arg open() (no O_CREAT), gcc rewrites the
 * call to ``__open_2(path, flags)``. Programs like grep ship that way,
 * so without these hooks ``grep -r`` opens deny-listed files anyway —
 * the public ``open`` symbol is never reached. Same story for openat /
 * the 64-bit variants.
 */
static int   (* volatile r_x_open_2)    (const char *, int);
static int   (* volatile r_x_open64_2)  (const char *, int);
static int   (* volatile r_x_openat_2)  (int, const char *, int);
static int   (* volatile r_x_openat64_2)(int, const char *, int);

static int   (* volatile r_stat)     (const char *, struct stat *);
static int   (* volatile r_lstat)    (const char *, struct stat *);
static int   (* volatile r_xstat)    (int, const char *, struct stat *);
static int   (* volatile r_lxstat)   (int, const char *, struct stat *);
static int   (* volatile r_fxstatat) (int, int, const char *, struct stat *, int);
static int   (* volatile r_fstatat)  (int, const char *, struct stat *, int);
static int   (* volatile r_access)   (const char *, int);
static int   (* volatile r_faccessat)(int, const char *, int, int);
static DIR  *(* volatile r_opendir)  (const char *);

static int   (* volatile r_unlink)   (const char *);
static int   (* volatile r_unlinkat) (int, const char *, int);
static int   (* volatile r_rename)   (const char *, const char *);
static int   (* volatile r_renameat) (int, const char *, int, const char *);
static int   (* volatile r_truncate) (const char *, off_t);
static int   (* volatile r_chmod)    (const char *, mode_t);
static int   (* volatile r_chown)    (const char *, uid_t, gid_t);

/* Exec family: ``execve`` is the kernel-facing entry, but glibc exposes a
 * pile of variants (``execvp``, ``execv``, ``execvpe``, ``execl*`` …). The
 * variants do NOT all bottom out at the public ``execve`` symbol — some
 * issue the syscall directly, or use an internal alias — so a hook that
 * only intercepts ``execve`` misses ``env -u LD_PRELOAD cat ...`` style
 * bypasses where ``env`` calls ``execvp`` from inside its own libc copy.
 */
static int   (* volatile r_execve)   (const char *, char *const[], char *const[]);
static int   (* volatile r_execv)    (const char *, char *const[]);
static int   (* volatile r_execvp)   (const char *, char *const[]);
static int   (* volatile r_execvpe)  (const char *, char *const[], char *const[]);

/* Network resolution entry point (the one curl / wget / python / node use). */
static int   (* volatile r_getaddrinfo)(const char *, const char *,
                              const struct addrinfo *, struct addrinfo **);

/* Time sources — see g_fake_* above. Wrapped unconditionally (so every
 * call routes through us) but only adjusted when g_fake_active. */
static time_t (*r_time)         (time_t *);
static int    (*r_gettimeofday) (struct timeval *, void *);
static int    (*r_clock_gettime)(clockid_t, struct timespec *);


/* ------------------------------------------------------------------------ */
/* Path matching                                                             */
/* ------------------------------------------------------------------------ */

/* Strip "/proc/<pid>/root/..." or "/proc/self/root/..." in place so the
 * deny-prefix match sees the underlying path. Anything else is left
 * untouched (the kernel handles `..` and symlinks itself for the actual
 * syscall, and our intent is to block the libc-level string).
 */
static void strip_proc_root(char *p)
{
    if (strncmp(p, "/proc/", 6) != 0) return;
    char *slash = strchr(p + 6, '/');
    if (!slash) return;
    if (strncmp(slash, "/root/", 6) == 0) {
        memmove(p, slash + 5, strlen(slash + 5) + 1);
    } else if (strcmp(slash, "/root") == 0) {
        p[0] = '/'; p[1] = '\0';
    }
}

static int is_denied_raw(const char *raw_path)
{
    if (!raw_path || g_deny_count == 0) return 0;

    char p[PATH_MAX];
    size_t rl = strnlen(raw_path, PATH_MAX - 1);
    memcpy(p, raw_path, rl);
    p[rl] = '\0';
    strip_proc_root(p);

    for (int i = 0; i < g_deny_count; i++) {
        size_t n = g_deny_n[i];
        if (strncmp(p, g_deny[i], n) == 0 &&
            (p[n] == '\0' || p[n] == '/')) {
            return 1;
        }
    }
    return 0;
}

static int is_denied(const char *raw_path)
{
    if (!g_initialized) return 0;
    if (in_confine) return 0;
    in_confine = 1;
    int r = is_denied_raw(raw_path);
    in_confine = 0;
    return r;
}

/* Resolve a (dirfd, relpath) pair into an absolute path string for prefix
 * matching. This is the fix for the grep -r bypass: when a caller does
 * ``openat(parent_dirfd, ".aidata", ...)`` we'd see the literal string
 * ``.aidata`` and miss our ``/workspace/.aidata`` prefix entirely; here
 * we read /proc/self/fd/<dirfd> back to the parent's absolute path and
 * concatenate.
 *
 * Returns 1 (denied) / 0 (allowed). Path-too-long or unreadable /proc
 * entry => fall back to the raw-string check (same as ``is_denied``),
 * which keeps unrelated dirfds (sockets, pipes) from spuriously failing.
 */
static int is_denied_at(int dirfd, const char *path)
{
    if (!g_initialized) return 0;
    if (in_confine) return 0;
    if (!path) return 0;

    /* Absolute path or AT_FDCWD: delegate to the simple string matcher. */
    if (path[0] == '/' || dirfd == AT_FDCWD) {
        in_confine = 1;
        int r = is_denied_raw(path);
        in_confine = 0;
        return r;
    }

    in_confine = 1;
    char fd_link[64];
    snprintf(fd_link, sizeof(fd_link), "/proc/self/fd/%d", dirfd);
    char dir_abs[PATH_MAX];
    ssize_t n = readlink(fd_link, dir_abs, sizeof(dir_abs) - 1);
    int r = 0;
    if (n > 0) {
        dir_abs[n] = '\0';
        char joined[PATH_MAX];
        /* snprintf truncates safely on overflow; truncated paths can't
         * accidentally match a deny prefix that's longer than what fit,
         * so the fail-safe direction is "don't match" here.
         */
        if (snprintf(joined, sizeof(joined), "%s/%s", dir_abs, path)
                < (int)sizeof(joined)) {
            r = is_denied_raw(joined);
        }
    } else {
        /* dirfd is something we can't readlink (a socket, anonymous pipe,
         * /memfd handle, ...). Fall back to the raw-string check — if
         * the relative path itself happens to literally contain a deny
         * prefix (e.g. "/workspace/.aidata/foo"), still block.
         */
        r = is_denied_raw(path);
    }
    in_confine = 0;
    return r;
}


/* ------------------------------------------------------------------------ */
/* Host matching (network deny)                                              */
/* ------------------------------------------------------------------------ */

/* Case-insensitive suffix match of ``node`` against the AIDATA_CONFINE_NET_DENY
 * patterns. Returns 1 (denied) / 0 (allowed). Pure string work — no hooked
 * libc calls — so it needs no re-entry guard. NULL / numeric-IP nodes never
 * match (an IP literal isn't a denied *name*; blocking by IP is out of scope —
 * see README). */
static int host_is_denied(const char *node)
{
    if (!node || g_net_deny_count == 0) return 0;

    char h[256];
    size_t nl = strnlen(node, sizeof(h) - 1);
    for (size_t i = 0; i < nl; i++) h[i] = (char)tolower((unsigned char)node[i]);
    h[nl] = '\0';
    /* tolerate the FQDN trailing-dot form ("skillhub.cn.") */
    while (nl > 0 && h[nl - 1] == '.') h[--nl] = '\0';

    for (int i = 0; i < g_net_deny_count; i++) {
        size_t pl = g_net_deny_n[i];
        if (pl == 0 || pl > nl) continue;
        if (nl == pl) {
            if (memcmp(h, g_net_deny[i], pl) == 0) return 1;
        } else if (h[nl - pl - 1] == '.' &&
                   memcmp(h + (nl - pl), g_net_deny[i], pl) == 0) {
            return 1;
        }
    }
    return 0;
}


/* ------------------------------------------------------------------------ */
/* Constructor                                                               */
/* ------------------------------------------------------------------------ */

static void parse_net_deny(const char *raw)
{
    if (!raw || !*raw) return;
    char *s = strdup(raw);
    if (!s) return;

    char *save = NULL;
    /* accept comma- or colon-separated host lists */
    for (char *tok = strtok_r(s, ",:", &save);
         tok && g_net_deny_count < MAX_NET_DENY;
         tok = strtok_r(NULL, ",:", &save)) {
        if (!*tok) continue;
        /* lowercase + strip surrounding dots/space so suffix match is clean */
        while (*tok == '.' || *tok == ' ') tok++;
        size_t n = strlen(tok);
        while (n > 0 && (tok[n - 1] == '.' || tok[n - 1] == ' ')) tok[--n] = '\0';
        if (n == 0) continue;
        char *low = strdup(tok);
        if (!low) continue;
        for (size_t i = 0; i < n; i++) low[i] = (char)tolower((unsigned char)low[i]);
        g_net_deny[g_net_deny_count] = low;
        g_net_deny_n[g_net_deny_count] = n;
        g_net_deny_count++;
    }
    free(s);
}

static void parse_deny(const char *raw)
{
    if (!raw || !*raw) return;
    char *s = strdup(raw);
    if (!s) return;

    char *save = NULL;
    for (char *tok = strtok_r(s, ":", &save);
         tok && g_deny_count < MAX_DENY;
         tok = strtok_r(NULL, ":", &save)) {
        if (!*tok) continue;
        size_t n = strlen(tok);
        /* drop trailing slash so prefix match still works on "/foo" vs "/foo/" */
        while (n > 1 && tok[n - 1] == '/') tok[--n] = '\0';
        g_deny[g_deny_count] = strdup(tok);
        g_deny_n[g_deny_count] = n;
        g_deny_count++;
    }
    free(s);
}

__attribute__((constructor))
static void confine_init(void)
{
    r_open      = dlsym(RTLD_NEXT, "open");
    r_open64    = dlsym(RTLD_NEXT, "open64");
    r_openat    = dlsym(RTLD_NEXT, "openat");
    r_openat64  = dlsym(RTLD_NEXT, "openat64");
    r_creat     = dlsym(RTLD_NEXT, "creat");
    r_fopen     = dlsym(RTLD_NEXT, "fopen");
    r_fopen64   = dlsym(RTLD_NEXT, "fopen64");
    r_x_open_2     = dlsym(RTLD_NEXT, "__open_2");
    r_x_open64_2   = dlsym(RTLD_NEXT, "__open64_2");
    r_x_openat_2   = dlsym(RTLD_NEXT, "__openat_2");
    r_x_openat64_2 = dlsym(RTLD_NEXT, "__openat64_2");

    r_stat      = dlsym(RTLD_NEXT, "stat");
    r_lstat     = dlsym(RTLD_NEXT, "lstat");
    r_xstat     = dlsym(RTLD_NEXT, "__xstat");
    r_lxstat    = dlsym(RTLD_NEXT, "__lxstat");
    r_fxstatat  = dlsym(RTLD_NEXT, "__fxstatat");
    r_fstatat   = dlsym(RTLD_NEXT, "fstatat");
    r_access    = dlsym(RTLD_NEXT, "access");
    r_faccessat = dlsym(RTLD_NEXT, "faccessat");
    r_opendir   = dlsym(RTLD_NEXT, "opendir");

    r_unlink    = dlsym(RTLD_NEXT, "unlink");
    r_unlinkat  = dlsym(RTLD_NEXT, "unlinkat");
    r_rename    = dlsym(RTLD_NEXT, "rename");
    r_renameat  = dlsym(RTLD_NEXT, "renameat");
    r_truncate  = dlsym(RTLD_NEXT, "truncate");
    r_chmod     = dlsym(RTLD_NEXT, "chmod");
    r_chown     = dlsym(RTLD_NEXT, "chown");

    r_execve    = dlsym(RTLD_NEXT, "execve");
    r_execv     = dlsym(RTLD_NEXT, "execv");
    r_execvp    = dlsym(RTLD_NEXT, "execvp");
    r_execvpe   = dlsym(RTLD_NEXT, "execvpe");

    r_getaddrinfo = dlsym(RTLD_NEXT, "getaddrinfo");

    r_time          = dlsym(RTLD_NEXT, "time");
    r_gettimeofday  = dlsym(RTLD_NEXT, "gettimeofday");
    r_clock_gettime = dlsym(RTLD_NEXT, "clock_gettime");

    /* Fail-CLOSED: every hook below dereferences ``r_*`` directly when no
     * fallback is documented, so a missing symbol would segfault on the
     * next call. Abort early with a precise error instead. The list is
     * exactly the set of pointers that have NO ``r_FOO ? r_FOO(...) :
     * r_FALLBACK(...)`` ternary in the wrappers below — keep them in sync.
     *
     * NOTE: __xstat / __lxstat / __fxstatat are OPTIONAL — glibc >= 2.33
     * removed these internal versioned-stat symbols in favour of the POSIX
     * names (stat / lstat / fstatat). The wrappers below use ternary
     * fallbacks, so missing symbols are safe. Similarly fstatat may be
     * absent on very old glibc — the wrapper falls back to __fxstatat.
     */
    struct { const char *name; void *fn; } required[] = {
        {"open",       (void *)r_open},
        {"openat",     (void *)r_openat},
        {"fopen",      (void *)r_fopen},
        {"stat",       (void *)r_stat},
        {"lstat",      (void *)r_lstat},
        {"access",     (void *)r_access},
        {"faccessat",  (void *)r_faccessat},
        {"opendir",    (void *)r_opendir},
        {"unlink",     (void *)r_unlink},
        {"unlinkat",   (void *)r_unlinkat},
        {"rename",     (void *)r_rename},
        {"renameat",   (void *)r_renameat},
        {"truncate",   (void *)r_truncate},
        {"chmod",      (void *)r_chmod},
        {"chown",      (void *)r_chown},
        {"execve",     (void *)r_execve},
        {"getaddrinfo",(void *)r_getaddrinfo},
        {"time",         (void *)r_time},
        {"gettimeofday", (void *)r_gettimeofday},
        {"clock_gettime",(void *)r_clock_gettime},
    };
    for (size_t i = 0; i < sizeof(required) / sizeof(required[0]); i++) {
        if (!required[i].fn) {
            fprintf(stderr, "[confine] FATAL: dlsym(%s) failed\n", required[i].name);
            _exit(99);
        }
    }
    /* At least one of fstatat / __fxstatat must be available for the
     * dirfd-relative stat hook to function. */
    if (!r_fstatat && !r_fxstatat) {
        fprintf(stderr, "[confine] FATAL: neither fstatat nor __fxstatat resolved\n");
        _exit(99);
    }

    parse_deny(getenv("AIDATA_CONFINE_DENY"));
    parse_net_deny(getenv("AIDATA_CONFINE_NET_DENY"));

    /* Record our own .so path for execve re-injection. dladdr() against
     * any function we own lets us recover the absolute load path, so we
     * don't need to trust LD_PRELOAD's value (an attacker may have
     * unset/munged it before the bash that spawned us got to exec).
     */
    Dl_info info;
    if (dladdr((void *)confine_init, &info) && info.dli_fname) {
        size_t n = strnlen(info.dli_fname, PATH_MAX - 1);
        memcpy(g_so_path, info.dli_fname, n);
        g_so_path[n] = '\0';
    }

    /* Defence-in-depth: mount-namespace isolation.
     *
     * The LD_PRELOAD hooks above cover every libc-mediated path-taking
     * call we know of, but glibc's own internal traversal helpers
     * (`__opendir`, `__open`, `__readdir64`) bypass dlsym(RTLD_NEXT, ...)
     * because libc reaches them through internal aliases, not the public
     * symbols. ``grep -r`` exhibits exactly this — its fts walker uses
     * those internal symbols and sails through the deny prefix.
     *
     * The fix-once-and-for-all answer is to put the bash subprocess in
     * its own mount namespace and bind a fresh empty tmpfs over each
     * deny prefix. After that, the kernel itself returns ENOENT/empty
     * dirs for those paths to *every* caller — grep, awk, statically
     * linked Go binaries, ctypes-syscall, the lot. The LD_PRELOAD hooks
     * stay as a belt-and-suspenders for cases that find a way out.
     *
     * Requires CAP_SYS_ADMIN (gongfeng containers run as root and have
     * it; vanilla unprivileged containers won't). If unshare/mount fail,
     * we silently degrade to "LD_PRELOAD only" — the agent can still get
     * around grep-style hooks but every other vector is locked.
     */
    /* AIDATA_CONFINE_DEBUG=1 turns on stderr logs for the namespace path,
     * useful when investigating "grep -r still reads /workspace/.aidata"
     * style leaks during bring-up.
     */
    int dbg = !!getenv("AIDATA_CONFINE_DEBUG");

    /* Faketime init: anchor the REALTIME clock to AIDATA_FAKE_NOW (unix epoch
     * seconds, computed host-side from the case's MOCK_SERVER_NOW). Capture
     * the offset ONCE here so the wrappers below are a single add — see the
     * g_fake_* comment for why this is offset-based, not frozen. */
    {
        const char *fn = getenv("AIDATA_FAKE_NOW");
        if (fn && *fn) {
            char *end = NULL;
            long long want = strtoll(fn, &end, 10);
            if (end != fn && want > 0) {
                time_t real = r_time(NULL);
                g_fake_off_sec = (time_t)want - real;
                g_fake_active = 1;
                size_t fl = strnlen(fn, sizeof(g_fake_now_str) - 1);
                memcpy(g_fake_now_str, fn, fl);
                g_fake_now_str[fl] = '\0';
                if (dbg) fprintf(stderr,
                    "[confine] faketime active: AIDATA_FAKE_NOW=%lld, offset=%lds\n",
                    want, (long)g_fake_off_sec);
            } else if (dbg) {
                fprintf(stderr,
                    "[confine] faketime: ignoring unparseable AIDATA_FAKE_NOW=%s\n", fn);
            }
        }
    }

    if (g_deny_count > 0) {
        if (unshare(CLONE_NEWNS) != 0) {
            if (dbg) fprintf(stderr,
                "[confine] unshare(CLONE_NEWNS) failed: %s — falling back "
                "to LD_PRELOAD-only mode (grep -r style internal-symbol "
                "bypass remains possible)\n", strerror(errno));
        } else {
            /* mark / as private so our mounts inside this namespace don't
             * propagate back out to siblings.
             */
            if (mount("none", "/", NULL, MS_REC | MS_PRIVATE, NULL) != 0 && dbg) {
                fprintf(stderr,
                    "[confine] mount(/, MS_PRIVATE) failed: %s\n", strerror(errno));
            }
            int mounted = 0;
            for (int i = 0; i < g_deny_count; i++) {
                /* tmpfs ro mount: empty + non-writable + non-exec.
                 * MS_NODEV/NOEXEC are belt-and-suspenders against accidentally
                 * stashing a setuid binary or device node inside.
                 */
                if (mount("none", g_deny[i], "tmpfs",
                          MS_RDONLY | MS_NODEV | MS_NOEXEC, NULL) == 0) {
                    mounted++;
                } else if (dbg) {
                    fprintf(stderr,
                        "[confine] mount(tmpfs, %s) failed: %s\n",
                        g_deny[i], strerror(errno));
                }
            }
            if (dbg) fprintf(stderr,
                "[confine] mounted %d/%d tmpfs deny prefixes\n",
                mounted, g_deny_count);
        }
    }

    g_initialized = 1;
}


/* ------------------------------------------------------------------------ */
/* open() family                                                             */
/* ------------------------------------------------------------------------ */

/* Helper: lazily resolve a symbol if the pointer is still NULL (i.e. our
 * constructor has not run yet but another library's constructor already calls
 * into our hook). Thread-safety: harmless TOCTOU — worst case we dlsym twice
 * and both get the same address. */
#define RESOLVE_IF_NULL(ptr, name) \
    do { if (!(ptr)) (ptr) = dlsym(RTLD_NEXT, (name)); } while(0)

int open(const char *path, int flags, ...)
{
    if (is_denied(path)) { errno = EACCES; return -1; }
    RESOLVE_IF_NULL(r_open, "open");
    mode_t mode = 0;
    if (flags & O_CREAT) {
        va_list ap; va_start(ap, flags); mode = va_arg(ap, mode_t); va_end(ap);
    }
    return r_open(path, flags, mode);
}

int open64(const char *path, int flags, ...)
{
    if (is_denied(path)) { errno = EACCES; return -1; }
    RESOLVE_IF_NULL(r_open64, "open64");
    RESOLVE_IF_NULL(r_open, "open");
    mode_t mode = 0;
    if (flags & O_CREAT) {
        va_list ap; va_start(ap, flags); mode = va_arg(ap, mode_t); va_end(ap);
    }
    return r_open64 ? r_open64(path, flags, mode) : r_open(path, flags, mode);
}

int openat(int dirfd, const char *path, int flags, ...)
{
    if (is_denied_at(dirfd, path)) { errno = EACCES; return -1; }
    RESOLVE_IF_NULL(r_openat, "openat");
    mode_t mode = 0;
    if (flags & O_CREAT) {
        va_list ap; va_start(ap, flags); mode = va_arg(ap, mode_t); va_end(ap);
    }
    return r_openat(dirfd, path, flags, mode);
}

int openat64(int dirfd, const char *path, int flags, ...)
{
    if (is_denied_at(dirfd, path)) { errno = EACCES; return -1; }
    RESOLVE_IF_NULL(r_openat64, "openat64");
    RESOLVE_IF_NULL(r_openat, "openat");
    mode_t mode = 0;
    if (flags & O_CREAT) {
        va_list ap; va_start(ap, flags); mode = va_arg(ap, mode_t); va_end(ap);
    }
    return r_openat64 ? r_openat64(dirfd, path, flags, mode)
                      : r_openat(dirfd, path, flags, mode);
}

int creat(const char *path, mode_t mode)
{
    if (is_denied(path)) { errno = EACCES; return -1; }
    RESOLVE_IF_NULL(r_creat, "creat");
    RESOLVE_IF_NULL(r_open, "open");
    return r_creat ? r_creat(path, mode) : r_open(path, O_CREAT | O_WRONLY | O_TRUNC, mode);
}

FILE *fopen(const char *path, const char *mode)
{
    if (is_denied(path)) { errno = EACCES; return NULL; }
    RESOLVE_IF_NULL(r_fopen, "fopen");
    return r_fopen(path, mode);
}

FILE *fopen64(const char *path, const char *mode)
{
    if (is_denied(path)) { errno = EACCES; return NULL; }
    RESOLVE_IF_NULL(r_fopen64, "fopen64");
    RESOLVE_IF_NULL(r_fopen, "fopen");
    return r_fopen64 ? r_fopen64(path, mode) : r_fopen(path, mode);
}

/* fortify open() wrappers — see comments next to the dlsym block above */

int __open_2(const char *path, int flags)
{
    if (is_denied(path)) { errno = EACCES; return -1; }
    RESOLVE_IF_NULL(r_x_open_2, "__open_2");
    RESOLVE_IF_NULL(r_open, "open");
    if (r_x_open_2) return r_x_open_2(path, flags);
    return r_open(path, flags);
}

int __open64_2(const char *path, int flags)
{
    if (is_denied(path)) { errno = EACCES; return -1; }
    RESOLVE_IF_NULL(r_x_open64_2, "__open64_2");
    RESOLVE_IF_NULL(r_open64, "open64");
    RESOLVE_IF_NULL(r_open, "open");
    if (r_x_open64_2) return r_x_open64_2(path, flags);
    return r_open64 ? r_open64(path, flags) : r_open(path, flags);
}

int __openat_2(int dirfd, const char *path, int flags)
{
    if (is_denied_at(dirfd, path)) { errno = EACCES; return -1; }
    RESOLVE_IF_NULL(r_x_openat_2, "__openat_2");
    RESOLVE_IF_NULL(r_openat, "openat");
    if (r_x_openat_2) return r_x_openat_2(dirfd, path, flags);
    return r_openat(dirfd, path, flags);
}

int __openat64_2(int dirfd, const char *path, int flags)
{
    if (is_denied_at(dirfd, path)) { errno = EACCES; return -1; }
    RESOLVE_IF_NULL(r_x_openat64_2, "__openat64_2");
    RESOLVE_IF_NULL(r_openat64, "openat64");
    RESOLVE_IF_NULL(r_openat, "openat");
    if (r_x_openat64_2) return r_x_openat64_2(dirfd, path, flags);
    return r_openat64 ? r_openat64(dirfd, path, flags) : r_openat(dirfd, path, flags);
}


/* ------------------------------------------------------------------------ */
/* stat() / access() / opendir()                                             */
/* ------------------------------------------------------------------------ */

int stat(const char *path, struct stat *buf)
{
    if (is_denied(path)) { errno = EACCES; return -1; }
    RESOLVE_IF_NULL(r_stat, "stat");
    return r_stat(path, buf);
}

int lstat(const char *path, struct stat *buf)
{
    if (is_denied(path)) { errno = EACCES; return -1; }
    RESOLVE_IF_NULL(r_lstat, "lstat");
    return r_lstat(path, buf);
}

int __xstat(int ver, const char *path, struct stat *buf)
{
    if (is_denied(path)) { errno = EACCES; return -1; }
    RESOLVE_IF_NULL(r_xstat, "__xstat");
    RESOLVE_IF_NULL(r_stat, "stat");
    if (r_xstat) return r_xstat(ver, path, buf);
    return r_stat(path, buf);
}

int __lxstat(int ver, const char *path, struct stat *buf)
{
    if (is_denied(path)) { errno = EACCES; return -1; }
    RESOLVE_IF_NULL(r_lxstat, "__lxstat");
    RESOLVE_IF_NULL(r_lstat, "lstat");
    if (r_lxstat) return r_lxstat(ver, path, buf);
    return r_lstat(path, buf);
}

int __fxstatat(int ver, int dirfd, const char *path, struct stat *buf, int flag)
{
    if (is_denied_at(dirfd, path)) { errno = EACCES; return -1; }
    RESOLVE_IF_NULL(r_fxstatat, "__fxstatat");
    RESOLVE_IF_NULL(r_fstatat, "fstatat");
    if (r_fxstatat) return r_fxstatat(ver, dirfd, path, buf, flag);
    return r_fstatat(dirfd, path, buf, flag);
}

int fstatat(int dirfd, const char *path, struct stat *buf, int flag)
{
    if (is_denied_at(dirfd, path)) { errno = EACCES; return -1; }
    RESOLVE_IF_NULL(r_fstatat, "fstatat");
    RESOLVE_IF_NULL(r_fxstatat, "__fxstatat");
    if (r_fstatat) return r_fstatat(dirfd, path, buf, flag);
    return r_fxstatat(1, dirfd, path, buf, flag);
}

int access(const char *path, int mode)
{
    if (is_denied(path)) { errno = EACCES; return -1; }
    RESOLVE_IF_NULL(r_access, "access");
    return r_access(path, mode);
}

int faccessat(int dirfd, const char *path, int mode, int flag)
{
    if (is_denied_at(dirfd, path)) { errno = EACCES; return -1; }
    RESOLVE_IF_NULL(r_faccessat, "faccessat");
    return r_faccessat(dirfd, path, mode, flag);
}

DIR *opendir(const char *path)
{
    if (is_denied(path)) { errno = EACCES; return NULL; }
    RESOLVE_IF_NULL(r_opendir, "opendir");
    return r_opendir(path);
}


/* ------------------------------------------------------------------------ */
/* mutating ops                                                              */
/* ------------------------------------------------------------------------ */

int unlink(const char *path)
{
    if (is_denied(path)) { errno = EACCES; return -1; }
    return r_unlink(path);
}

int unlinkat(int dirfd, const char *path, int flag)
{
    if (is_denied_at(dirfd, path)) { errno = EACCES; return -1; }
    return r_unlinkat(dirfd, path, flag);
}

int rename(const char *from, const char *to)
{
    if (is_denied(from) || is_denied(to)) { errno = EACCES; return -1; }
    return r_rename(from, to);
}

int renameat(int from_fd, const char *from, int to_fd, const char *to)
{
    if (is_denied_at(from_fd, from) || is_denied_at(to_fd, to)) {
        errno = EACCES; return -1;
    }
    return r_renameat(from_fd, from, to_fd, to);
}

int truncate(const char *path, off_t length)
{
    if (is_denied(path)) { errno = EACCES; return -1; }
    return r_truncate(path, length);
}

int chmod(const char *path, mode_t mode)
{
    if (is_denied(path)) { errno = EACCES; return -1; }
    return r_chmod(path, mode);
}

int chown(const char *path, uid_t uid, gid_t gid)
{
    if (is_denied(path)) { errno = EACCES; return -1; }
    return r_chown(path, uid, gid);
}


/* ------------------------------------------------------------------------ */
/* network: getaddrinfo name-resolution deny                                 */
/* ------------------------------------------------------------------------ */

/* Block name resolution for AIDATA_CONFINE_NET_DENY hosts. Returning
 * EAI_NONAME mimics a non-existent domain, so callers (curl / requests / ...)
 * fail their connect with the usual "could not resolve host" rather than a
 * surprising error class. Allowed hosts fall through to the real resolver,
 * which reads /etc/{hosts,resolv.conf,nsswitch.conf} (none deny-listed) as
 * normal. */
int getaddrinfo(const char *node, const char *service,
                const struct addrinfo *hints, struct addrinfo **res)
{
    if (host_is_denied(node)) return EAI_NONAME;
    return r_getaddrinfo(node, service, hints, res);
}


/* ------------------------------------------------------------------------ */
/* time faking: shift CLOCK_REALTIME / gettimeofday / time() by g_fake_off_sec */
/* ------------------------------------------------------------------------ */
/*
 * Only the REALTIME family is shifted. CLOCK_MONOTONIC (and the *_RAW /
 * BOOTTIME / PROCESS/THREAD CPU clocks) pass through untouched so timers,
 * sleeps, and subprocess timeouts keep measuring true elapsed time. When
 * g_fake_active==0 these are exact pass-throughs.
 */

time_t time(time_t *tloc)
{
    time_t v = r_time(NULL);
    if (g_fake_active) v += g_fake_off_sec;
    if (tloc) *tloc = v;
    return v;
}

int gettimeofday(struct timeval *tv, void *tz)
{
    int rc = r_gettimeofday(tv, tz);
    /* tv is __nonnull per glibc; r_gettimeofday already wrote through it. */
    if (rc == 0 && g_fake_active) tv->tv_sec += g_fake_off_sec;
    return rc;
}

int clock_gettime(clockid_t clk, struct timespec *ts)
{
    int rc = r_clock_gettime(clk, ts);
    /* ts is __nonnull per glibc; r_clock_gettime already wrote through it. */
    if (rc == 0 && g_fake_active &&
        (clk == CLOCK_REALTIME
#ifdef CLOCK_REALTIME_COARSE
         || clk == CLOCK_REALTIME_COARSE
#endif
        )) {
        ts->tv_sec += g_fake_off_sec;
    }
    return rc;
}


/* ------------------------------------------------------------------------ */
/* execve: re-inject LD_PRELOAD + AIDATA_CONFINE_DENY + AIDATA_CONFINE_NET_DENY*/
/* ------------------------------------------------------------------------ */

/* Returns a freshly-malloc'd envp array (NULL-terminated) where we have:
 *   1. dropped any LD_PRELOAD / AIDATA_CONFINE_DENY entries the caller set
 *      (they may have been munged: `LD_PRELOAD=`, `LD_PRELOAD=/dev/null`)
 *   2. re-added our canonical .so path + the deny list from our own env
 *
 * If allocation fails we return NULL and the caller should fall back to
 * the original envp — broken confine is better than a crashed bash.
 */
static char **rebuild_envp(char *const envp[], int exempt)
{
    int n = 0;
    while (envp && envp[n]) n++;

    char **out = (char **)malloc(sizeof(char *) * (n + 6));
    if (!out) return NULL;

    int j = 0;
    for (int i = 0; i < n; i++) {
        const char *e = envp[i];
        if (strncmp(e, "LD_PRELOAD=",             11) == 0) continue;
        if (strncmp(e, "AIDATA_CONFINE_DENY=",     20) == 0) continue;
        if (strncmp(e, "AIDATA_CONFINE_NET_DENY=", 24) == 0) continue;
        if (strncmp(e, "AIDATA_FAKE_NOW=",         16) == 0) continue;
        out[j] = strdup(e);
        if (!out[j]) goto oom;
        j++;
    }

    /* Exempt targets (e.g. the LibreOffice launcher chain) run UNCONFINED:
     * return the stripped envp WITHOUT re-adding LD_PRELOAD / confine vars.
     * Our libc hooks segfault soffice during its multithreaded startup. */
    if (exempt) { out[j] = NULL; return out; }

    if (g_so_path[0]) {
        char buf[PATH_MAX + 16];
        snprintf(buf, sizeof(buf), "LD_PRELOAD=%s", g_so_path);
        out[j] = strdup(buf);
        if (!out[j]) goto oom;
        j++;
    }

    const char *deny = getenv("AIDATA_CONFINE_DENY");
    if (deny && *deny) {
        size_t dlen = strlen(deny) + 32;
        char *buf = (char *)malloc(dlen);
        if (!buf) goto oom;
        snprintf(buf, dlen, "AIDATA_CONFINE_DENY=%s", deny);
        out[j++] = buf;
    }

    const char *net_deny = getenv("AIDATA_CONFINE_NET_DENY");
    if (net_deny && *net_deny) {
        size_t dlen = strlen(net_deny) + 32;
        char *buf = (char *)malloc(dlen);
        if (!buf) goto oom;
        snprintf(buf, dlen, "AIDATA_CONFINE_NET_DENY=%s", net_deny);
        out[j++] = buf;
    }

    /* Re-add the faketime anchor so a child that did `env -i` / unset it
     * still observes the same shifted clock as its parent. */
    if (g_fake_active && g_fake_now_str[0]) {
        char buf[64];
        snprintf(buf, sizeof(buf), "AIDATA_FAKE_NOW=%s", g_fake_now_str);
        out[j] = strdup(buf);
        if (!out[j]) goto oom;
        j++;
    }

    out[j] = NULL;
    return out;

oom:
    for (int k = 0; k < j; k++) free(out[k]);
    free(out);
    return NULL;
}

/* Allowlist: LibreOffice's launcher chain (soffice -> oosplash -> soffice.bin)
 * deterministically SIGSEGVs under our LD_PRELOAD libc interposition during its
 * multithreaded startup. Exec these UNCONFINED — they only ever process the
 * agent's own files under the writable workspace, so skipping the deny-list for
 * them does not widen the attack surface in a meaningful way. */
static int is_exec_exempt(const char *file, char *const argv[])
{
    const char *cands[2]; int nc = 0;
    if (file) cands[nc++] = file;
    if (argv && argv[0]) cands[nc++] = argv[0];
    for (int i = 0; i < nc; i++) {
        const char *b = strrchr(cands[i], '/');
        b = b ? b + 1 : cands[i];
        if (strcmp(b, "soffice") == 0 || strcmp(b, "soffice.bin") == 0 ||
            strcmp(b, "oosplash") == 0)
            return 1;
    }
    return 0;
}

static void confine_strip_env(void)
{
    unsetenv("LD_PRELOAD");
    unsetenv("AIDATA_CONFINE_DENY");
    unsetenv("AIDATA_CONFINE_NET_DENY");
}

int execve(const char *file, char *const argv[], char *const envp[])
{
    char **new_envp = rebuild_envp(envp, is_exec_exempt(file, argv));
    int rc = r_execve(file, argv, new_envp ? new_envp : envp);
    /* execve only returns on failure */
    if (new_envp) {
        for (char **p = new_envp; *p; p++) free(*p);
        free(new_envp);
    }
    return rc;
}

/* execv / execvp: these read process-wide ``environ`` (no envp parameter).
 * If the caller previously did ``unsetenv("LD_PRELOAD")`` (which is what
 * ``env -u LD_PRELOAD`` does internally) ``environ`` no longer carries the
 * variable. We re-setenv it before forwarding so the new image inherits
 * the .so. ``setenv`` is glibc-thread-safe; we don't worry about the brief
 * window inside this hook.
 */
static void confine_reinject_env(void)
{
    if (g_so_path[0])
        setenv("LD_PRELOAD", g_so_path, /*overwrite=*/1);
    if (g_deny_count > 0) {
        size_t total = 1;
        for (int i = 0; i < g_deny_count; i++) total += g_deny_n[i] + 1;
        char *buf = (char *)malloc(total);
        if (buf) {
            char *p = buf;
            for (int i = 0; i < g_deny_count; i++) {
                if (i > 0) *p++ = ':';
                memcpy(p, g_deny[i], g_deny_n[i]);
                p += g_deny_n[i];
            }
            *p = '\0';
            setenv("AIDATA_CONFINE_DENY", buf, /*overwrite=*/1);
            free(buf);
        }
    }
    /* Same for the network deny list, rebuilt from g_net_deny[] (immune to a
     * prior ``unsetenv``/``env -u AIDATA_CONFINE_NET_DENY`` by the caller). */
    if (g_net_deny_count > 0) {
        size_t total = 1;
        for (int i = 0; i < g_net_deny_count; i++) total += g_net_deny_n[i] + 1;
        char *buf = (char *)malloc(total);
        if (buf) {
            char *p = buf;
            for (int i = 0; i < g_net_deny_count; i++) {
                if (i > 0) *p++ = ',';
                memcpy(p, g_net_deny[i], g_net_deny_n[i]);
                p += g_net_deny_n[i];
            }
            *p = '\0';
            setenv("AIDATA_CONFINE_NET_DENY", buf, /*overwrite=*/1);
            free(buf);
        }
    }
    /* Same for the faketime anchor (immune to a prior unsetenv / env -u). */
    if (g_fake_active && g_fake_now_str[0])
        setenv("AIDATA_FAKE_NOW", g_fake_now_str, /*overwrite=*/1);
}

int execv(const char *file, char *const argv[])
{
    if (is_exec_exempt(file, argv)) confine_strip_env(); else confine_reinject_env();
    if (r_execv) return r_execv(file, argv);
    /* Fallback: pretend execvp through execve. environ is the global one. */
    extern char **environ;
    return r_execve(file, argv, environ);
}

int execvp(const char *file, char *const argv[])
{
    if (is_exec_exempt(file, argv)) confine_strip_env(); else confine_reinject_env();
    if (r_execvp) return r_execvp(file, argv);
    extern char **environ;
    return r_execve(file, argv, environ);
}

int execvpe(const char *file, char *const argv[], char *const envp[])
{
    char **new_envp = rebuild_envp(envp, is_exec_exempt(file, argv));
    int rc;
    if (r_execvpe) {
        rc = r_execvpe(file, argv, new_envp ? new_envp : envp);
    } else {
        rc = r_execve(file, argv, new_envp ? new_envp : envp);
    }
    if (new_envp) {
        for (char **p = new_envp; *p; p++) free(*p);
        free(new_envp);
    }
    return rc;
}
