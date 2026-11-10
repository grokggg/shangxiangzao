#!/bin/bash
# In-sandbox compile of libconfine.so. Called once per task by
# engine/task.py (right after upload_dir(CONFINE_DIR_HOST, CONFINE_DIR_IN))
# so the .so picks up the runtime image's exact glibc + arch every time.
#
# Why compile-on-launch instead of shipping a prebuilt .so:
#   * Zero image-bake step. Any glibc-based image (aio-agent-rc /
#     aidata-aio-agent:2.x / work_mcp_mirror / your fork) just works.
#   * No multi-arch matrix in the aidata repo (x86_64 / aarch64 / ...).
#   * Auditors see source on every PR — no opaque blob to diff.
#
# Cost: ~150 ms gcc invocation per task. Negligible vs gongfeng cold-start
# (~40 s). The compiled .so is cached in /workspace/.aidata/confine/ for
# the lifetime of the sandbox; engine/task.py only re-runs compile if the
# precompiled artifact is missing or older than the source.

set -euo pipefail

DIR="$(cd "$(dirname "$0")" && pwd)"
SRC="$DIR/libconfine.c"
SO="$DIR/libconfine.so"

if [ ! -r "$SRC" ]; then
    echo "[confine] FATAL: $SRC missing — engine upload likely failed" >&2
    exit 99
fi

# Pick the first available compiler. aio-agent-rc has gcc, aidata-aio-agent:2.x
# also has gcc, almost every glibc image has either. clang is a fallback for
# minimal images that prefer it.
CC=""
for candidate in cc gcc clang; do
    if command -v "$candidate" >/dev/null 2>&1; then CC="$candidate"; break; fi
done
if [ -z "$CC" ]; then
    echo "[confine] FATAL: no C compiler found (cc/gcc/clang). Either bake one into the image or ship a prebuilt .so." >&2
    exit 99
fi

# -O2          — small + fast, no debug overhead in the hot path
# -fPIC -shared — required for LD_PRELOAD .so
# -ldl         — dlsym()
# -Wl,-soname  — keeps `ldd` / readelf identifying the library cleanly
# -Wl,-z,now   — BIND_NOW: resolve all PLT entries at load time. Without this,
#                lazy binding inside our hooks can segfault when another library's
#                constructor (e.g. libselinux init_lib) calls our fopen/open hook
#                before the dynamic linker has finished resolving all symbols.
# -Wall -Werror=implicit-function-declaration — catch typos at build time;
#                                               but DON'T turn on -Werror
#                                               wholesale (different glibc
#                                               headers warn differently).
"$CC" \
    -O2 -fPIC -shared \
    -Wall -Werror=implicit-function-declaration \
    -Wl,-soname,libconfine.so \
    -Wl,-z,now \
    -o "$SO" "$SRC" \
    -ldl

# sanity-check dlopenability — silent success means the .so loads cleanly.
# Failing to *build* the smoke binary used to be silently skipped (`|| true`
# + ``[ -x ... ]`` guard) which let a broken toolchain hide a broken .so;
# now both build and run errors fail the script with exit 99 so the engine's
# ``[aidata] confine kit setup failed`` warning fires for the right reason.
if ! "$CC" -x c -o /tmp/_confine_smoke - >/dev/null 2>&1 <<'EOF'
#include <dlfcn.h>
#incl