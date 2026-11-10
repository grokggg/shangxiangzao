"""Private scalar facts and rebuildable cursors for direct Trajectory assembly."""

from __future__ import annotations

from collections.abc import Hashable
from dataclasses import dataclass, field
from typing import Any

from nexus.gateway.trajectory import MessageId, NodeId, SegmentBoundary, Trajectory, TrajectoryNode


@dataclass(frozen=True)
class PendingBoundary:
    boundary: SegmentBoundary
    claims: tuple[tuple[str, Hashable], ...] = ()


@dataclass(frozen=True)
class RuntimeFactOccurrence:
    sequence: int
    fact: Any


@dataclass(frozen=True)
class NodeRuntimeFacts:
    identity: Hashable
    cursor_sequence: int | None = None
    tools_identity: Hashable = None
    context_key: Hashable | None = None
    boundary_claims: tuple[tuple[str, Hashable], ...] = ()
    occurrences: tuple[RuntimeFactOccurrence, ...] = ()

    def facts_of_type(self, fact_type: type) -> list[tuple[int, Any]]:
        return [
            (occurrence.sequence, occurrence.fact)
            for occurrence in self.occurrences
            if isinstance(occurrence.fact, fact_type)
        ]


@dataclass
class RuntimeFactsLedger:
    by_node: dict[NodeId, NodeRuntimeFacts] = field(default_factory=dict)


@dataclass
class RuntimeLookupCache:
    last_node: NodeId | None = None
    last_node_by_context: dict[Hashable, NodeId] = field(default_factory=dict)
    last_llm_by_context: dict[Hashable, NodeId] = field(default_factory=dict)
    llm_by_message_id: dict[MessageId, NodeId] = field(default_factory=dict)

    def clear(self) -> None:
        self.last_node = None
        self.last_node_by_context.clear()
        self.last_llm_by_context.clear()
        self.llm_by_message_id.clear()

    def observe(self, node_id: NodeId, node: TrajectoryNode, facts: NodeRuntimeFacts) -> None:
        self.last_node = node_id
        if facts.context_key is not None:
            self.last_node_by_context[facts.context_key] = node_id
            if node.generated_by == 'llm':
                self.last_llm_by_context[facts.context_key] = node_id
        if node.generated_by == 'llm':
            self.llm_by_message_id[node.message_id] = node_id

    def rebuild(self, trajectory: Trajectory, facts: RuntimeFactsLedger) -> None:
        self.clear()
        cursor_order = sorted(
            (
                (runtime_facts.cursor_sequence, node_id)
                for node_id, runtime_facts in facts.by_node.items()
                if runtime_facts.cursor_sequence is not None
            )
        )
        for _, node_id in cursor_order:
            self.observe(node_id, trajectory.nodes[node_id], facts.by_node[node_id])


@dataclass
class RuntimeState:
    facts: RuntimeFactsLedger = field(default_factory=RuntimeFactsLedger)
    lookup_cache: RuntimeLookupCache = field(default_factory=RuntimeLookupCache)
    next_node_id: int = 0
    next_message_id: int = 0
    next_request_sequence: int = 0
    next_cursor_sequence: int = 0
    next_fact_sequence: int = 0
    pending_boundary: PendingBoundary | None = None

    def rebuild_lookup_cache(self, trajectory: Trajectory) -> None:
        self.lookup_cache.rebuild(trajectory, self.facts)
