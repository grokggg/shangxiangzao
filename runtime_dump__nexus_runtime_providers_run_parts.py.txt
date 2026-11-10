"""Reusable run parts for runtime providers."""

# /// script
# dependencies = []
#
# [project]
# name = "runtime_provider_run_parts"
# authors = [
#     { name = "tarschen" },
# ]
#
# [tool.nexus]
# run_parts = { gongfeng = "nexus.runtime.providers.run_parts", docker = "nexus.runtime.providers.run_parts" }
# ///

from __future__ import annotations

from typing import Any

from nexus.runner.compose import run_part


@run_part('gongfeng')
def gongfeng_run_part(params: dict[str, Any]) -> dict[str, Any]:
    if params:
        raise ValueError(f'Run part gongfeng does not accept parameters: {sorted(params)}')
    return {
        'runtime_provider_name': 'gongfeng',
        'runtime_provider_config': {
            'api_base_url': '${oc.env:GONGFENG_RUNTIME_BASE_URL,http://hyrl-sandbox.prod.woa.com:8052}',
            'token': '${oc.env:GONGFENG_RUNTIME_TOKEN,""}',
        },
    }


@run_part('docker')
def docker_run_part(params: dict[str, Any]) -> dict[str, Any]:
    if params:
        raise ValueError(f'Run part docker does not accept parameters: {sorted(params)}')
    return {
        'runtime_provider_name': 'docker',
        'runtime_provider_config': {},
    }
