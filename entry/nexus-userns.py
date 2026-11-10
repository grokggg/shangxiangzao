#!/usr/bin/python3
"""Map ordinary container users into an isolated user namespace before bwrap."""
import ctypes
import os
import pathlib
import signal
import sys


def main():
    ready_read, ready_write = os.pipe()
    release_read, release_write = os.pipe()
    pid = os.fork()
    if pid == 0:
        os.close(ready_read)
        os.close(release_write)
        libc = ctypes.CDLL(None, use_errno=True)
        if libc.unshare(0x10000000) != 0:  # CLONE_NEWUSER
            os._exit(1)
        os.write(ready_write, b'1')
        os.close(ready_write)
        if os.read(release_read, 1) != b'1':
            os._exit(1)
        os.close(release_read)
        os.execv(sys.argv[1], sys.argv[1:])
    os.close(ready_write)
    os.close(release_read)
    try:
        if os.read(ready_read, 1) != b'1':
            raise RuntimeError('cannot create Nexus user namespace')
        # The trusted parent writes the maps while it still has capabilities in
        # the container namespace. The child cannot modify these maps afterwards.
        for name in ('uid_map', 'gid_map'):
            pathlib.Path(f'/proc/{pid}/{name}').write_text('0 0 65536\n')
        os.write(release_write, b'1')
    except BaseException:
        os.kill(pid, signal.SIGKILL)
        os.waitpid(pid, 0)
        raise
    finally:
        os.close(ready_read)
        os.close(release_write)
    _, status = os.waitpid(pid, 0)
    return os.waitstatus_to_exitcode(status)


if __name__ == '__main__':
    sys.exit(main())
