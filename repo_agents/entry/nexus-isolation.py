#!/usr/bin/python3
"""Start Nexus in the container's mount namespace, without bwrap."""
import os
import pathlib


def runtime_args(port=32000, python_version='3.10'):
    return ['/nix/runtime/nix-env/bin/bash',
            '/nix/runtime/nexus/deployment/runtime/start.sh', '--host', '0.0.0.0',
            '--port', str(port), '--python-version', python_version]


def main():
    port = int(os.environ.get('NEXUS_RUNTIME_PORT', '32000'))
    if not 1 <= port <= 65535:
        raise ValueError('invalid NEXUS_RUNTIME_PORT')
    # Do not access Wiki storage at startup. The shared mount namespace lets
    # Nexus observe volumes attached after snapshot restore, just like sandboxd.
    pathlib.Path('/data/workspace').mkdir(parents=True, exist_ok=True)
    args = runtime_args(port, os.environ.get('NEXUS_PYTHON_VERSION', '3.10'))
    os.execv(args[0], args)


if __name__ == '__main__':
    main()
