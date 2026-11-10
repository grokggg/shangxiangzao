"""Freeze direct Gateway runtime facts into the public Trajectory contract."""

from __future__ import annotations

import copy

from collections.abc import Hashable
from dataclasses import dataclass, field
from typing import Any

from nexus.gateway.runtime_state import NodeRuntimeFacts, RuntimeState
from nexus.gateway.scaffold_policy import (
    BoundAnchor,
    ChildPromptCandidate,
    DeclaredChainClaim,
    DeliveryReceipt,
    NotificationDelivery,
    ParentClaim,
    ResponsePolicy,
    RetrievalIntent,
    SidechainMembership,
    SpawnBinding,
    SpawnIntent,
)
from nexus.gateway.trajectory import (
    MAIN_CHAIN_ID,
    Chain,
    JoinLink,
    NodeId,
    SpawnLink,
    Trajectory,
)


JoinCandidate = tuple[str | None, str, NodeId, int]


@dataclass
class _CorrelationAnchor:
    spawn_pairs: list[tuple[str, NodeId]]
    retrievals: list[tuple[str, NodeId, int]]
    receipts: list[tuple[str, bool, NodeId, int]]
    notifications: list[tuple[NodeId, int]]
    declared_parents: list[str] = field(default_factory=list)
    parent_agents: list[str] = field(default_factory=list)
    structural_parents: list[str] = field(default_factory=list)
    join_candidates: list[JoinCandidate] = field(default_factory=list)
    publish_unjoined: bool = False
    evidence: dict[str, Any] = field(default_factory=dict)

    @property
    def spawn_ids(self) -> list[str]:
        return list(dict.fromkeys(call_id for call_id, _ in self.spawn_pairs))

    @property
    def retrieval_ids(self) -> set[str]:
        return {call_id for call_id, _, _ in self.retrievals}

    def reduce_delivery(self, trajectory: Trajectory, *, publish: bool = False) -> list[JoinCandidate]:
        candidates, delivery_evidence = _direct_reduce_delivery_facts(
            trajectory, self.receipts, self.notifications, self.spawn_ids, self.retrieval_ids
        )
        self.publish_unjoined = self.publish_unjoined or bool(candidates)
        if not publish:
            return candidates
        self.join_candidates = candidates
        _direct_set_join_summary(self.evidence, self.join_candidates, self.publish_unjoined, trajectory)
        self.evidence.update(delivery_evidence)
        if delivery_evidence and not self.join_candidates:
            self.evidence.pop('joined', None)
        return candidates

    def scope_to(self, parent: str, ownership: dict[NodeId, str]) -> bool:
        spawn_pairs = [pair for pair in self.spawn_pairs if ownership[pair[1]] == parent]
        retrievals = [item for item in self.retrievals if ownership[item[1]] == parent]
        receipts = [item for item in self.receipts if ownership[item[2]] == parent]
        notifications = [item for item in self.notifications if ownership[item[0]] == parent]
        spawn_mismatch = bool(self.spawn_pairs) and not spawn_pairs
        if spawn_pairs:
            _direct_set_spawn_summary(self.evidence, spawn_pairs)
        self.spawn_pairs, self.retrievals = spawn_pairs, retrievals
        self.receipts, self.notifications = receipts, notifications
        return spawn_mismatch

    def has_facts(self) -> bool:
        return any(vars(self).values())


def freeze_trajectory(
    live_trajectory: Trajectory,
    runtime_state: RuntimeState,
    response_policy: ResponsePolicy,
    *,
    session_id: str | None = None,
    state: str | None = None,
    session_timeline: list[dict[str, Any]] | None = None,
    latest_messages: list[dict[str, Any]] | None = None,
    termination_reason: str | None = None,
    metrics: dict[str, Any] | None = None,
    extra: dict[str, Any] | None = None,
    copy_inputs: bool = True,
) -> Trajectory:
    """Resolve a consistent Trajectory/runtime-state pair without changing node IDs.

    The only topology read here is the public ``roots``/``parent``/``children``
    graph on ``Trajectory`` itself. Runtime facts supply scalar interpretation.
    Active snapshots use the default copy; session finalization passes
    ``copy_inputs=False`` and returns the live trajectory directly.
    """
    if copy_inputs:
        trajectory, snapshot_state = copy.deepcopy((live_trajectory, runtime_state))
    else:
        trajectory, snapshot_state = live_trajectory, runtime_state
    _validate_freeze_inputs(trajectory, snapshot_state)
    facts = snapshot_state.facts
    _apply_direct_selection(trajectory, facts, response_policy)
    _resolve_direct_chains(trajectory, facts)
    unanchored = _attach_direct_session_events(trajectory, copy.deepcopy(session_timeline or []))
    trajectory.session_id = session_id if session_id is not None else trajectory.session_id
    trajectory.state = state if state is not None else trajectory.state
    trajectory.latest_messages = copy.deepcopy(latest_messages)
    trajectory.timeline = unanchored
    trajectory.termination_reason = termination_reason
    trajectory.metrics = copy.deepcopy(metrics or {})
    trajectory.extra = copy.deepcopy(extra or {})
    Trajectory.from_dump(trajectory.dump())
    return trajectory


def _apply_direct_selection(
    trajectory: Trajectory,
    facts: Any,
    response_policy: ResponsePolicy,
) -> None:
    sibling_groups = [trajectory.roots, *(node.children for node in trajectory.nodes.values())]
    for siblings in sibling_groups:
        llm_ids = [node_id for node_id in siblings if trajectory.nodes[node_id].generated_by == 'llm']
        if len(llm_ids) <= 1:
            continue
        streams: dict[Any, list[NodeId]] = {}
        for node_id in llm_ids:
            streams.setdefault(facts.by_node[node_id].context_key, []).append(node_id)
        selected: set[NodeId] = set()
        for stream in streams.values():
            if response_policy is ResponsePolicy.KEEP_ALL:
                selected.update(stream)
            elif response_policy is ResponsePolicy.PREFER_EXTENDED_ELSE_LAST:
                selected.update([node_id for node_id in stream if trajectory.nodes[node_id].children] or stream[-1:])
            else:
                raise ValueError(f'Unsupported response policy: {response_policy!r}')
        for node_id in llm_ids:
            trajectory.nodes[node_id].selection = 'selected' if node_id in selected else 'rejected_sibling'


def _resolve_direct_chains(trajectory: Trajectory, facts: Any) -> None:
    main = Chain(chain_id=MAIN_CHAIN_ID, kind='main')
    chains: dict[str, Chain] = {MAIN_CHAIN_ID: main}
    ownership: dict[NodeId, str] = {}
    branches: dict[str, list[tuple[NodeId, NodeId]]] = {}
    membership_cache: dict[NodeId, tuple[SidechainMembership, ...] | None] = {}
    contains_main_llm_cache: dict[NodeId, bool] = {}
    subtree_has_llm_cache: dict[NodeId, bool] = {}
    declared_parent_cache: dict[NodeId, str | None] = {}
    anonymous_chains: set[str] = set()

    def branch_membership_identity(membership: SidechainMembership) -> tuple[Any, ...]:
        if membership.chain_id:
            return ('agent', membership.chain_id)
        return ('anonymous', membership.kind, membership.detector)

    def record_attributes(chain: Chain, membership: SidechainMembership) -> None:
        stable = chain.evidence.setdefault('attributes', {})
        candidates = chain.evidence.setdefault('attribute_candidates', {})
        for key, value in membership.attributes:
            if key in candidates:
                if value not in candidates[key]:
                    candidates[key].append(copy.deepcopy(value))
            elif key not in stable:
                stable[key] = copy.deepcopy(value)
            elif stable[key] != value:
                candidates[key] = [stable.pop(key), copy.deepcopy(value)]
        if not stable:
            chain.evidence.pop('attributes', None)
        if not candidates:
            chain.evidence.pop('attribute_candidates', None)

    def contains_main_llm(root_id: NodeId) -> bool:
        stack = [(root_id, False)]
        while stack:
            node_id, expanded = stack.pop()
            if node_id in contains_main_llm_cache:
                continue
            node = trajectory.nodes[node_id]
            if not expanded:
                stack.append((node_id, True))
                stack.extend((child_id, False) for child_id in reversed(node.children))
                continue
            node_facts: NodeRuntimeFacts = facts.by_node[node_id]
            subtree_has_llm_cache[node_id] = node.generated_by == 'llm' or any(
                subtree_has_llm_cache[child_id] for child_id in node.children
            )
            contains_main_llm_cache[node_id] = (
                node.generated_by == 'llm' and not node_facts.facts_of_type(SidechainMembership)
            ) or any(contains_main_llm_cache[child_id] for child_id in node.children)
        return contains_main_llm_cache[root_id]

    def is_ambiguous_sidechain_branch(root_id: NodeId) -> bool:
        return not contains_main_llm(root_id) and subtree_has_llm_cache[root_id] and branch_memberships(root_id) is None

    def branch_memberships(root_id: NodeId) -> tuple[SidechainMembership, ...] | None:
        if root_id in membership_cache:
            return membership_cache[root_id]
        if contains_main_llm(root_id):
            membership_cache[root_id] = None
            return None
        memberships: list[SidechainMembership] = []
        has_llm = False
        stack = [root_id]
        while stack:
            node_id = stack.pop()
            node = trajectory.nodes[node_id]
            if node.generated_by == 'llm':
                has_llm = True
                node_facts: NodeRuntimeFacts = facts.by_node[node_id]
                occurrences = node_facts.facts_of_type(SidechainMembership)
                if not occurrences:
                    membership_cache[root_id] = None
                    return None
                memberships.extend(membership for _, membership in occurrences)
                continue
            stack.extend(reversed(node.children))
        identities = {branch_membership_identity(membership) for membership in memberships}
        membership_cache[root_id] = tuple(memberships) if has_llm and len(identities) == 1 else None
        return membership_cache[root_id]

    def branch_declared_parent(root_id: NodeId, membership: SidechainMembership) -> str | None:
        if root_id in declared_parent_cache:
            return declared_parent_cache[root_id]
        stack = [root_id]
        while stack:
            node_id = stack.pop()
            node = trajectory.nodes[node_id]
            if node.generated_by == 'llm':
                node_facts: NodeRuntimeFacts = facts.by_node[node_id]
                declared_parent_cache[root_id] = next(
                    (
                        claim.parent_chain_id
                        for _, claim in node_facts.facts_of_type(DeclaredChainClaim)
                        if claim.chain_id == membership.chain_id and isinstance(claim.parent_chain_id, str)
                    ),
                    None,
                )
                return declared_parent_cache[root_id]
            stack.extend(reversed(node.children))
        declared_parent_cache[root_id] = None
        return declared_parent_cache[root_id]

    def register(
        memberships: tuple[SidechainMembership, ...],
        root_id: NodeId,
        parent_chain: str | None,
        declared_parent: str | None,
    ) -> str:
        membership = memberships[0]
        agent_id = membership.chain_id or None
        chain_id = agent_id or f'chain@{root_id}'
        resolved_parent = declared_parent if declared_parent is not None else parent_chain
        existing = chains.get(chain_id)
        if existing is not None:
            if declared_parent is None and parent_chain is None:
                resolved_parent = existing.parent_chain_id
            if (
                existing.kind == membership.kind
                and existing.parent_chain_id == resolved_parent
                and existing.detector == membership.detector
            ):
                for item in memberships:
                    record_attributes(existing, item)
                return chain_id
            if agent_id is None:
                raise ValueError(f'anonymous chain id collision: {chain_id!r}')
            chain_id = f'{agent_id}@{root_id}'
            if chain_id in chains:
                raise ValueError(f'sidechain id collision: {chain_id!r}')
        evidence: dict[str, Any] = {}
        physical_parent = trajectory.nodes[root_id].parent
        if physical_parent is not None:
            evidence['anchor_node'] = physical_parent
        if agent_id is not None and chain_id != agent_id:
            evidence['collides_with'] = agent_id
        chains[chain_id] = Chain(
            chain_id=chain_id,
            kind=membership.kind,
            parent_chain_id=resolved_parent,
            detector=membership.detector,
            evidence=evidence,
        )
        if agent_id is None:
            anonymous_chains.add(chain_id)
        for item in memberships:
            record_attributes(chains[chain_id], item)
        return chain_id

    for root_id in trajectory.roots:
        memberships = branch_memberships(root_id)
        root_chain = (
            register(memberships, root_id, None, branch_declared_parent(root_id, memberships[0]))
            if memberships is not None
            else MAIN_CHAIN_ID
        )
        correlation_root = root_id if memberships is None and is_ambiguous_sidechain_branch(root_id) else None
        stack = [(root_id, root_chain, True, correlation_root)]
        while stack:
            node_id, chain_id, is_entry, correlation_root = stack.pop()
            entry_anchor = node_id if is_entry and chain_id != MAIN_CHAIN_ID else None
            if not is_entry:
                node_memberships = branch_memberships(node_id)
                if node_memberships is None:
                    if correlation_root is None and is_ambiguous_sidechain_branch(node_id):
                        correlation_root = node_id
                else:
                    membership = node_memberships[0]
                    current_chain = chains[chain_id]
                    agent_id = membership.chain_id or None
                    same_chain = chain_id != MAIN_CHAIN_ID and (
                        current_chain.kind == membership.kind
                        and current_chain.detector == membership.detector
                        and (
                            (agent_id is None and chain_id in anonymous_chains)
                            or agent_id == chain_id
                            or current_chain.evidence.get('collides_with') == agent_id
                        )
                    )
                    if same_chain:
                        for item in node_memberships:
                            record_attributes(current_chain, item)
                    else:
                        declared_parent = branch_declared_parent(node_id, membership)
                        nested_chain = register(node_memberships, node_id, chain_id, declared_parent)
                        is_entry = nested_chain != chain_id
                        chain_id = nested_chain
                        if is_entry:
                            entry_anchor = correlation_root if correlation_root is not None else node_id
                            correlation_root = None

            ownership[node_id] = chain_id
            if is_entry:
                chains[chain_id].roots.append(node_id)
                if chain_id != MAIN_CHAIN_ID:
                    anchor_root = entry_anchor if entry_anchor is not None else node_id
                    branches.setdefault(chain_id, []).append((anchor_root, node_id))
            for child_id in reversed(trajectory.nodes[node_id].children):
                stack.append((child_id, chain_id, False, correlation_root))

    for node_id, chain_id in ownership.items():
        if trajectory.nodes[node_id].generated_by != 'llm':
            continue
        for _, membership in facts.by_node[node_id].facts_of_type(SidechainMembership):
            record_attributes(chains[chain_id], membership)

    trajectory.chains = chains
    trajectory._invalidate_caches()
    for chain_id, roots in branches.items():
        _resolve_direct_chain_links(trajectory, facts, ownership, chain_id, roots)
    _record_direct_unresolved_correlations(trajectory, facts, ownership, chains)


def _resolve_direct_chain_links(
    trajectory: Trajectory,
    facts: Any,
    ownership: dict[NodeId, str],
    chain_id: str,
    branches: list[tuple[NodeId, NodeId]],
) -> None:
    chain = trajectory.chains[chain_id]
    correlation_agent_id = chain.evidence.get('collides_with')
    if not isinstance(correlation_agent_id, str):
        correlation_agent_id = chain_id
    anchors: list[_CorrelationAnchor] = []
    for anchor_root_id, member_root_id in branches:
        outside = _direct_nodes_outside_branch(trajectory, anchor_root_id)
        leaf_paths = _direct_chain_leaf_paths(trajectory, member_root_id, ownership, chain_id)
        if not leaf_paths:
            continue
        for member_path in leaf_paths:
            leaf_path = _direct_ancestor_path(trajectory, anchor_root_id, member_path[-1])
            prompts = _direct_prompt_candidates(facts, leaf_path)
            anchor = _direct_correlation_anchor(
                facts,
                correlation_agent_id,
                prompts[0].value if prompts else None,
                outside,
            )
            declared_parents, parent_agents = _direct_parent_claims(facts, leaf_path)
            anchor.declared_parents = declared_parents
            anchor.parent_agents = parent_agents
            if len(declared_parents) == 1:
                anchor.evidence['parent_chain_id'] = declared_parents[0]
            if len(parent_agents) == 1:
                anchor.evidence['parent_agent_id'] = parent_agents[0]
            structural_parent = trajectory.nodes[anchor_root_id].parent
            if structural_parent is not None and ownership[structural_parent] != chain_id:
                anchor.structural_parents = [ownership[structural_parent]]
                anchor.evidence.update(_direct_fork_evidence(trajectory, structural_parent))
            if anchor.has_facts():
                anchors.append(anchor)

    if not anchors:
        return

    spawn_ids, spawn_pairs = _direct_spawn_resolution(anchors)
    join_candidates = list(
        dict.fromkeys(candidate for anchor in anchors for candidate in anchor.reduce_delivery(trajectory))
    )
    declared_parents = {parent for anchor in anchors for parent in anchor.declared_parents}
    parent_agents = {parent for anchor in anchors for parent in anchor.parent_agents}
    local_parent_agents = parent_agents.intersection(trajectory.chains) - {chain_id}
    spawn_parents = {ownership[node_id] for _, node_id in spawn_pairs if ownership[node_id] != chain_id}
    join_parents = {ownership[node_id] for _, _, node_id, _ in join_candidates if ownership[node_id] != chain_id}
    structural_parents = {parent for anchor in anchors for parent in anchor.structural_parents if parent != chain_id}
    external_parent_agents = parent_agents - set(trajectory.chains)
    parent_rungs = (
        ('declared_parent', declared_parents),
        ('local_parent_agent', local_parent_agents),
        ('spawn_node_chain', spawn_parents),
        ('join_node_chain', join_parents),
        ('structural_parent', structural_parents),
        ('external_parent_agent', external_parent_agents),
    )
    parent_source, parent_candidates = next(
        ((source, candidates) for source, candidates in parent_rungs if candidates),
        (None, set()),
    )
    if parent_source is None:
        chain.parent_chain_id = None
    elif len(parent_candidates) == 1:
        chain.parent_chain_id = next(iter(parent_candidates))
    else:
        chain.parent_chain_id = None
        chain.evidence['parent_chain_candidates'] = sorted(parent_candidates)

    local_parent = chain.parent_chain_id if chain.parent_chain_id in trajectory.chains else None
    if local_parent is not None:
        mismatches = [anchor.scope_to(local_parent, ownership) for anchor in anchors]
        if any(mismatches):
            chain.evidence['spawn_parent_mismatch'] = True

    for anchor in anchors:
        anchor.reduce_delivery(trajectory, publish=True)

    public_anchors: list[dict[str, Any]] = []
    for anchor in anchors:
        public = copy.deepcopy(anchor.evidence)
        if public and public not in public_anchors:
            public_anchors.append(public)
    if len(public_anchors) == 1:
        chain.evidence['correlation'] = public_anchors[0]
    elif public_anchors:
        chain.evidence['correlation_candidates'] = public_anchors

    spawn_ids, spawn_pairs = _direct_spawn_resolution(anchors)
    if spawn_ids:
        spawn_nodes = list(dict.fromkeys(node_id for _, node_id in spawn_pairs))
        generated_spawn_nodes = [node_id for node_id in spawn_nodes if trajectory.nodes[node_id].generated_by == 'llm']
        if generated_spawn_nodes:
            spawn_nodes = generated_spawn_nodes
        chain.spawn = SpawnLink(
            node=spawn_nodes[0] if len(spawn_nodes) == 1 else None,
            tool_use_id=spawn_ids[0] if len(spawn_ids) == 1 else None,
            candidates=[] if len(spawn_ids) == 1 else spawn_ids,
        )
    else:
        chain.spawn = None

    join_candidates = list(
        dict.fromkeys(
            candidate
            for anchor in anchors
            for candidate in anchor.join_candidates
            if candidate[1] != 'inline' or not spawn_ids or candidate[0] in spawn_ids
        )
    )
    ambiguous_join_parent = len(parent_candidates) > 1 and parent_candidates == join_parents
    if join_candidates and not ambiguous_join_parent:
        latest = _direct_latest_join_candidates(join_candidates, trajectory)
        nodes = list(dict.fromkeys(node_id for _, _, node_id, _ in latest))
        join_ids = {join_id for join_id, _, _, _ in latest if isinstance(join_id, str)}
        kinds = {kind for _, kind, _, _ in latest}
        chain.join = JoinLink(
            node=nodes[0] if len(nodes) == 1 else None,
            tool_use_id=next(iter(join_ids)) if len(join_ids) == 1 else None,
            kind=next(iter(kinds)) if len(kinds) == 1 else None,
        )
    elif not ambiguous_join_parent and any(anchor.evidence.get('joined') is True for anchor in anchors):
        join_ids = {
            join_id for anchor in anchors if isinstance((join_id := anchor.evidence.get('join_tool_use_id')), str)
        }
        kinds = {kind for anchor in anchors if isinstance((kind := anchor.evidence.get('join_kind')), str)}
        chain.join = JoinLink(
            tool_use_id=next(iter(join_ids)) if len(join_ids) == 1 else None,
            kind=next(iter(kinds)) if len(kinds) == 1 else None,
        )
    else:
        chain.join = None


def _direct_correlation_anchor(
    facts: Any,
    correlation_agent_id: str,
    target: str | None,
    outside: list[NodeId],
) -> _CorrelationAnchor:
    spawn_intents: list[tuple[str, NodeId]] = []
    exact: list[tuple[str, NodeId]] = []
    prefix: list[tuple[str, NodeId]] = []
    retrievals: list[tuple[str, NodeId, int]] = []
    bound_spawn_ids: set[str] = set()
    receipts: list[tuple[str, bool, NodeId, int]] = []
    notifications: list[tuple[NodeId, int]] = []
    for node_id in outside:
        node_facts: NodeRuntimeFacts = facts.by_node[node_id]
        for _, intent in node_facts.facts_of_type(SpawnIntent):
            pair = (intent.tool_use_id, node_id)
            spawn_intents.append(pair)
            if target is not None and intent.child_prompt == target:
                exact.append(pair)
            elif target is not None and (
                target.startswith(intent.child_prompt) or intent.child_prompt.startswith(target)
            ):
                prefix.append(pair)
        retrievals.extend(
            (intent.tool_use_id, node_id, sequence)
            for sequence, intent in node_facts.facts_of_type(RetrievalIntent)
            if intent.agent_id == correlation_agent_id
        )
        bound_spawn_ids.update(
            binding.tool_use_id
            for _, binding in node_facts.facts_of_type(SpawnBinding)
            if binding.agent_id == correlation_agent_id
        )
        receipts.extend(
            (receipt.tool_use_id, receipt.delivered, node_id, sequence)
            for sequence, receipt in node_facts.facts_of_type(DeliveryReceipt)
        )
        notifications.extend(
            (node_id, sequence)
            for sequence, delivery in node_facts.facts_of_type(NotificationDelivery)
            if delivery.agent_id == correlation_agent_id
        )
    bound_spawn_pairs = [pair for pair in spawn_intents if pair[0] in bound_spawn_ids]
    spawn_pairs = bound_spawn_pairs or exact or prefix
    spawn_ids = list(dict.fromkeys(call_id for call_id, _ in spawn_pairs))
    retrieval_ids = {call_id for call_id, _, _ in retrievals}
    relevant_ids = set(spawn_ids).union(retrieval_ids)
    anchor = _CorrelationAnchor(
        spawn_pairs=spawn_pairs,
        retrievals=retrievals,
        receipts=[receipt for receipt in receipts if receipt[0] in relevant_ids],
        notifications=notifications,
        publish_unjoined=bool(spawn_pairs),
    )
    _direct_set_spawn_summary(anchor.evidence, spawn_pairs)
    return anchor


def _record_direct_unresolved_correlations(
    trajectory: Trajectory,
    facts: Any,
    ownership: dict[NodeId, str],
    chains: dict[str, Chain],
) -> None:
    resolved_spawn_ids = {
        tool_use_id
        for chain in chains.values()
        if chain.spawn is not None
        for tool_use_id in (
            [chain.spawn.tool_use_id] if chain.spawn.tool_use_id is not None else chain.spawn.candidates
        )
    }
    unresolved_nodes: dict[str, list[NodeId]] = {}
    binding_agents: dict[str, set[str]] = {}
    for node_id, node_facts in facts.by_node.items():
        for _, intent in node_facts.facts_of_type(SpawnIntent):
            if intent.tool_use_id not in resolved_spawn_ids:
                unresolved_nodes.setdefault(intent.tool_use_id, []).append(node_id)
    for node_facts in facts.by_node.values():
        for _, binding in node_facts.facts_of_type(SpawnBinding):
            if binding.tool_use_id in unresolved_nodes:
                binding_agents.setdefault(binding.tool_use_id, set()).add(binding.agent_id)
    for tool_use_id, candidates in unresolved_nodes.items():
        candidates = list(dict.fromkeys(candidates))
        generated = [node_id for node_id in candidates if trajectory.nodes[node_id].generated_by == 'llm']
        anchors = generated or candidates
        item: dict[str, Any] = {'kind': 'spawn', 'tool_use_id': tool_use_id}
        if len(anchors) == 1:
            item['node'] = anchors[0]
        else:
            item['node_candidates'] = anchors
        agents = sorted(binding_agents.get(tool_use_id, ()))
        if len(agents) == 1:
            item['agent_id'] = agents[0]
        elif agents:
            item['agent_id_candidates'] = agents
            item['reason'] = 'conflicting_spawn_bindings'
        for chain_id in dict.fromkeys(ownership[node_id] for node_id in anchors):
            evidence = chains[chain_id].evidence.setdefault('unresolved_correlations', [])
            evidence.append(copy.deepcopy(item))


def _direct_prompt_candidates(facts: Any, path: list[NodeId]) -> list[ChildPromptCandidate]:
    candidates = [
        (prompt.order, path_index, sequence, prompt)
        for path_index, node_id in enumerate(path)
        for sequence, prompt in facts.by_node[node_id].facts_of_type(ChildPromptCandidate)
    ]
    candidates.sort(key=lambda item: item[:3])
    return [item[3] for item in candidates]


def _direct_parent_claims(facts: Any, path: list[NodeId]) -> tuple[list[str], list[str]]:
    owner_facts = None
    for node_id in reversed(path):
        node_facts: NodeRuntimeFacts = facts.by_node[node_id]
        if node_facts.facts_of_type(SidechainMembership):
            owner_facts = node_facts
            break
    if owner_facts is None:
        return [], []
    declared = [
        claim.parent_chain_id
        for _, claim in owner_facts.facts_of_type(DeclaredChainClaim)
        if isinstance(claim.parent_chain_id, str)
    ]
    memberships = owner_facts.facts_of_type(SidechainMembership)
    membership_parents = [
        membership.parent_chain_id for _, membership in memberships if isinstance(membership.parent_chain_id, str)
    ]
    membership_context = owner_facts.context_key
    raw_parent_agents = (
        [
            claim.parent_chain_id
            for node_facts in facts.by_node.values()
            if node_facts.context_key == membership_context and node_facts.facts_of_type(SidechainMembership)
            for _, claim in node_facts.facts_of_type(ParentClaim)
        ]
        if membership_context is not None
        else []
    )
    unique_parent_agents = list(dict.fromkeys(raw_parent_agents))
    parents = unique_parent_agents if len(unique_parent_agents) == 1 else []
    if not declared and not raw_parent_agents:
        parents.extend(membership_parents)
    return list(dict.fromkeys(declared)), list(dict.fromkeys(parents))


def _direct_fork_evidence(trajectory: Trajectory, parent_id: NodeId) -> dict[str, int]:
    path: list[NodeId] = []
    current: NodeId | None = parent_id
    while current is not None:
        path.append(current)
        current = trajectory.nodes[current].parent
    path.reverse()
    segment_index = 0
    seen_llm = False
    for node_id in path:
        node = trajectory.nodes[node_id]
        if node.generated_by != 'llm':
            continue
        if seen_llm and node.segment_boundary is not None and node.segment_boundary.starts_new_segment:
            segment_index += 1
        seen_llm = True
    return {'fork_node_index': len(path) - 1, 'fork_segment_index': segment_index}


def _direct_set_spawn_summary(anchor: dict[str, Any], spawn_pairs: list[tuple[str, NodeId]]) -> None:
    anchor.pop('spawn_tool_use_id', None)
    anchor.pop('spawn_candidates', None)
    spawn_ids = list(dict.fromkeys(call_id for call_id, _ in spawn_pairs))
    if len(spawn_ids) == 1:
        anchor['spawn_tool_use_id'] = spawn_ids[0]
    elif spawn_ids:
        anchor['spawn_candidates'] = spawn_ids


def _direct_reduce_delivery_facts(
    trajectory: Trajectory,
    raw_receipts: list[tuple[str, bool, NodeId, int]] | tuple[tuple[str, bool, NodeId, int], ...],
    raw_notifications: list[tuple[NodeId, int]] | tuple[tuple[NodeId, int], ...],
    spawn_ids: list[str],
    retrieval_ids: set[str],
) -> tuple[list[JoinCandidate], dict[str, Any]]:
    relevant_ids = set(spawn_ids).union(retrieval_ids)
    latest_receipts: dict[str, list[tuple[str, bool, NodeId, int]]] = {}
    for receipt in raw_receipts:
        call_id, _, node_id, sequence = receipt
        if call_id not in relevant_ids:
            continue
        current = latest_receipts.get(call_id, [])
        message_id = trajectory.nodes[node_id].message_id
        current_message_id = trajectory.nodes[current[0][2]].message_id if current else None
        if current_message_id is None or message_id > current_message_id:
            latest_receipts[call_id] = [receipt]
            continue
        if message_id != current_message_id:
            continue
        same_node = next((index for index, item in enumerate(current) if item[2] == node_id), None)
        if same_node is None:
            current.append(receipt)
        elif sequence > current[same_node][3]:
            current[same_node] = receipt
    ambiguous_receipts = sorted(
        (
            receipt
            for receipts_at_latest_message in latest_receipts.values()
            if len({delivered for _, delivered, _, _ in receipts_at_latest_message}) > 1
            for receipt in receipts_at_latest_message
        ),
        key=lambda receipt: (
            trajectory.nodes[receipt[2]].message_id,
            receipt[2],
            receipt[3],
            receipt[0],
            receipt[1],
        ),
    )
    confirmed_receipts = [
        receipt
        for receipts_at_latest_message in latest_receipts.values()
        if len({delivered for _, delivered, _, _ in receipts_at_latest_message}) == 1
        for receipt in receipts_at_latest_message
    ]
    join_candidates = [
        (call_id, 'inline' if call_id in spawn_ids else 'poll', node_id, sequence)
        for call_id, delivered, node_id, sequence in confirmed_receipts
        if delivered
    ]
    join_candidates.extend((None, 'notify', node_id, sequence) for node_id, sequence in raw_notifications)
    delivery_evidence: dict[str, Any] = {}
    if ambiguous_receipts:
        delivery_evidence['delivery_ambiguous'] = True
        delivery_evidence['delivery_candidates'] = [
            {
                'node': node_id,
                'tool_use_id': call_id,
                'delivered': delivered,
                'message_id': trajectory.nodes[node_id].message_id,
            }
            for call_id, delivered, node_id, _ in ambiguous_receipts
        ]
    return join_candidates, delivery_evidence


def _direct_set_join_summary(
    anchor: dict[str, Any],
    candidates: list[JoinCandidate],
    publish_unjoined: bool,
    trajectory: Trajectory,
) -> None:
    anchor.pop('join_tool_use_id', None)
    anchor.pop('join_kind', None)
    if not candidates:
        if publish_unjoined:
            anchor['joined'] = False
        else:
            anchor.pop('joined', None)
        return
    latest = _direct_latest_join_candidates(candidates, trajectory)
    join_ids = {join_id for join_id, _, _, _ in latest if isinstance(join_id, str)}
    kinds = {kind for _, kind, _, _ in latest}
    anchor['joined'] = True
    if len(join_ids) == 1:
        anchor['join_tool_use_id'] = next(iter(join_ids))
    if len(kinds) == 1:
        anchor['join_kind'] = next(iter(kinds))


def _direct_spawn_resolution(anchors: list[_CorrelationAnchor]) -> tuple[list[str], list[tuple[str, NodeId]]]:
    spawn_groups = [anchor.spawn_ids for anchor in anchors]
    spawn_groups = [group for group in spawn_groups if group]
    all_spawn_ids = list(dict.fromkeys(spawn_id for group in spawn_groups for spawn_id in group))
    common = set(spawn_groups[0]).intersection(*spawn_groups[1:]) if spawn_groups else set()
    spawn_ids = [spawn_id for spawn_id in all_spawn_ids if not common or spawn_id in common]
    spawn_pairs = list(dict.fromkeys(pair for anchor in anchors for pair in anchor.spawn_pairs if pair[0] in spawn_ids))
    return spawn_ids, spawn_pairs


def _direct_latest_join_candidates(candidates: list[JoinCandidate], trajectory: Trajectory) -> list[JoinCandidate]:
    latest_by_delivery: dict[tuple[str | None, str, NodeId], JoinCandidate] = {}
    for candidate in candidates:
        join_id, kind, node_id, sequence = candidate
        key = (join_id, kind, node_id)
        previous = latest_by_delivery.get(key)
        if previous is None or sequence >= previous[3]:
            latest_by_delivery[key] = candidate
    deduplicated = list(latest_by_delivery.values())
    latest_message_id = max(trajectory.nodes[node_id].message_id for _, _, node_id, _ in deduplicated)
    return [candidate for candidate in deduplicated if trajectory.nodes[candidate[2]].message_id == latest_message_id]


def _direct_nodes_outside_branch(trajectory: Trajectory, branch_root_id: NodeId) -> list[NodeId]:
    result: list[NodeId] = []
    stack = list(trajectory.roots)
    while stack:
        node_id = stack.pop()
        if node_id == branch_root_id:
            continue
        result.append(node_id)
        stack.extend(trajectory.nodes[node_id].children)
    result.sort(key=lambda node_id: trajectory.nodes[node_id].message_id)
    return result


def _direct_ancestor_path(trajectory: Trajectory, ancestor_id: NodeId, descendant_id: NodeId) -> list[NodeId]:
    reverse_path: list[NodeId] = []
    current: NodeId | None = descendant_id
    while current is not None:
        reverse_path.append(current)
        if current == ancestor_id:
            return list(reversed(reverse_path))
        current = trajectory.nodes[current].parent
    raise ValueError(f'node {ancestor_id} is not an ancestor of node {descendant_id}')


def _direct_chain_leaf_paths(
    trajectory: Trajectory,
    root_id: NodeId,
    ownership: dict[NodeId, str],
    chain_id: str,
) -> list[list[NodeId]]:
    if trajectory.nodes[root_id].selection == 'rejected_sibling':
        return []
    paths: list[list[NodeId]] = []
    stack = [(root_id, [root_id])]
    while stack:
        node_id, path = stack.pop()
        children = [
            child_id
            for child_id in trajectory.nodes[node_id].children
            if ownership[child_id] == chain_id and trajectory.nodes[child_id].selection != 'rejected_sibling'
        ]
        if not children:
            paths.append(path)
            continue
        stack.extend((child_id, [*path, child_id]) for child_id in children)
    return paths


def _attach_direct_session_events(
    trajectory: Trajectory,
    session_timeline: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    by_times: dict[tuple[Any, Any], list[NodeId]] = {}
    for node in trajectory.iter_nodes():
        if node.generated_by != 'llm':
            continue
        node.timeline = [event for event in node.timeline if event.get('source') != 'session']
        for event in node.timeline:
            if event.get('event_type') == 'llm_request':
                by_times.setdefault((event.get('start_time'), event.get('end_time')), []).append(node.node_id)
    unanchored: list[dict[str, Any]] = []
    for event in session_timeline:
        candidates = by_times.get((event.get('start'), event.get('end')), [])
        if len(candidates) == 1:
            trajectory.nodes[candidates[0]].timeline.append({'source': 'session', **event})
        else:
            unanchored.append({'source': 'session', **event})
    return unanchored


def _validate_freeze_inputs(trajectory: Trajectory, runtime_state: RuntimeState) -> None:
    """Validate the physical forest and runtime facts consumed by the resolver."""
    seen: set[NodeId] = set()
    stack = [(root_id, None) for root_id in reversed(trajectory.roots)]
    while stack:
        node_id, expected_parent = stack.pop()
        if node_id not in trajectory.nodes:
            raise ValueError(f'trajectory topology references missing node {node_id}')
        if node_id in seen:
            raise ValueError(f'trajectory topology reaches node {node_id} more than once')
        node = trajectory.nodes[node_id]
        if node.node_id != node_id:
            raise ValueError(f'trajectory node key {node_id} does not match embedded node id {node.node_id}')
        if node.parent != expected_parent:
            raise ValueError(
                f'trajectory node {node_id} parent {node.parent!r} does not match physical parent {expected_parent!r}'
            )
        seen.add(node_id)
        stack.extend((child_id, node_id) for child_id in reversed(node.children))
    unreachable = set(trajectory.nodes) - seen
    if unreachable:
        raise ValueError(f'trajectory nodes {sorted(unreachable)} are not reachable from trajectory roots')
    fact_nodes = set(runtime_state.facts.by_node)
    if fact_nodes != seen:
        missing = sorted(seen - fact_nodes)
        extra = sorted(fact_nodes - seen)
        raise ValueError(f'runtime facts do not match trajectory nodes: missing={missing}, extra={extra}')

    for facts in runtime_state.facts.by_node.values():
        for _, node_id in facts.boundary_claims:
            _validate_node_reference(node_id, seen, 'runtime boundary claim')
        for occurrence in facts.occurrences:
            anchor = getattr(occurrence.fact, 'anchor', None)
            if isinstance(anchor, BoundAnchor):
                _validate_node_reference(anchor.node_id, seen, 'runtime fact anchor')


def _validate_node_reference(node_id: Hashable, nodes: set[NodeId], label: str) -> None:
    if node_id not in nodes:
        raise ValueError(f'{label} {node_id} does not reference a trajectory node')
