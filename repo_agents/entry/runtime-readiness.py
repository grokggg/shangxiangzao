#!/usr/bin/python3
"""Require live service health, never a snapshot marker."""
import argparse
import json
import os
import socket
import sys
import time
import urllib.error
import urllib.request


def probe(host, port):
    try:
        with socket.create_connection((host, port), timeout=1):
            return True
    except OSError:
        return False


class NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None


def probe_health(port, path, expected_status):
    # Always contact this container directly, without inherited HTTP proxies.
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}), NoRedirect())
    try:
        with opener.open(f'http://127.0.0.1:{port}{path}', timeout=1) as response:
            if response.status != 200:
                return False
            data = response.read(65537)
            if len(data) > 65536:
                return False
            body = json.loads(data)
            return isinstance(body, dict) and body.get('status') == expected_status
    except (OSError, ValueError, urllib.error.URLError):
        return False


def check_ready(port_probe=probe, health_probe=probe_health, *, dependencies_only=False):
    ports = [int(os.environ.get('NEXUS_ENVD_PORT', '49983')),
             int(os.environ.get('SANDBOXD_PORT', '20001')),
             int(os.environ.get('NEXUS_RUNTIME_PORT', '32000'))]
    if any(not 1 <= port <= 65535 for port in ports) or len(set(ports)) != 3:
        raise ValueError('service ports must be distinct and between 1 and 65535')
    results = [port_probe('127.0.0.1', ports[0]),
               health_probe(ports[2], '/health', 'ok')]
    # sandboxd invokes this mode from its own HTTP handler. Probing that handler
    # again would recurse; the caller's HTTP response proves sandboxd is serving.
    if not dependencies_only:
        results.append(health_probe(ports[1], '/api/v1/runtime/health', 'READY'))
    return all(results)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--wait', action='store_true')
    parser.add_argument('--supervisor-pid', type=int)
    parser.add_argument('--dependencies-only', action='store_true')
    args = parser.parse_args()
    timeout = float(os.environ.get('RUNTIME_START_TIMEOUT_SECONDS', '120')) if args.wait else 0
    deadline = time.monotonic() + timeout
    while True:
        if args.supervisor_pid:
            try:
                os.kill(args.supervisor_pid, 0)
            except ProcessLookupError:
                print('Supervisor exited before all service ports were listening', file=sys.stderr)
                return 1
        if check_ready(dependencies_only=args.dependencies_only):
            return 0
        if time.monotonic() >= deadline:
            print('envd listener, sandboxd READY and Nexus health ok are required', file=sys.stderr)
            return 1
        time.sleep(0.5)


if __name__ == '__main__':
    sys.exit(main())
