"""Built-in direct scaffold policies."""

from __future__ import annotations

from nexus.gateway.policies.generic import GENERIC_POLICY
from nexus.gateway.policies.hycode import HYCODE_POLICY
from nexus.gateway.policies.nexus import NEXUS_POLICY
from nexus.gateway.scaffold_policy import register_scaffold_policy


register_scaffold_policy(NEXUS_POLICY)
register_scaffold_policy(GENERIC_POLICY)
register_scaffold_policy(HYCODE_POLICY)

__all__ = ['GENERIC_POLICY', 'HYCODE_POLICY', 'NEXUS_POLICY']
