"""Direct event-fact policy for Nexus-native scaffolds."""

from dataclasses import replace

from nexus.gateway.scaffold_policy import (
    BoundaryFact,
    DeclaredChainClaim,
    GenericScaffoldPolicy,
    InputMessage,
    ObservationEvent,
    OutputMessage,
    RequestEvent,
    ResponseEvent,
    SidechainMembership,
)


class NexusScaffoldPolicy(GenericScaffoldPolicy):
    """Declared-chain attribution plus explicit/marker compaction boundaries."""

    def __init__(self):
        super().__init__(policy_id='nexus')

    def inspect_event(self, event):
        facts = super().inspect_event(event)
        request = event.request if isinstance(event, ResponseEvent) else event
        if isinstance(request, ObservationEvent):
            messages = request.parent_messages or ()
            meta = request.request_meta or {}
        elif isinstance(request, RequestEvent):
            messages = request.messages
            meta = request.request_meta or {}
        else:
            return facts

        declared = meta.get('declared_chain')
        if not isinstance(declared, dict) or declared.get('chain_id') is None:
            declared_claim = None
            membership = None
            chain_id = 'main'
        else:
            chain_id = str(declared['chain_id'])
            parent_chain_id = None if declared.get('parent_chain_id') is None else str(declared['parent_chain_id'])
            kind = str(declared.get('kind') or 'sidechain')
            membership_attributes = []
            if parent_chain_id is not None:
                membership_attributes.append(('parent_chain_id', parent_chain_id))
            if isinstance(declared.get('spawn_order'), int):
                membership_attributes.append(('spawn_order', declared['spawn_order']))
            declared_claim = DeclaredChainClaim(chain_id, kind, parent_chain_id)
            membership = (
                None
                if kind == 'main'
                else SidechainMembership(
                    chain_id=chain_id,
                    kind=kind,
                    parent_chain_id=parent_chain_id,
                    detector='declared_chain',
                    attributes=tuple(membership_attributes),
                )
            )

        if isinstance(event, RequestEvent) and messages:
            boundary_facts = self._boundary_facts(messages, meta)
        elif isinstance(event, ResponseEvent) and not messages:
            boundary_facts = self._empty_request_boundary_facts(meta)
        else:
            boundary_facts = ()
        timeline = dict(facts.timeline)
        if timeline:
            timeline['is_sidechain'] = membership is not None
        return replace(
            facts,
            context_key=('nexus', chain_id, membership is not None),
            membership=membership,
            declared_claim=declared_claim,
            parent_claim=None,
            boundary_facts=boundary_facts,
            timeline=tuple(timeline.items()),
        )

    @staticmethod
    def _boundary_facts(messages, meta):
        facts = []
        invocation_kind = meta.get('invocation_kind')
        if invocation_kind in ('summary_inference', 'compaction_resume'):
            facts.append(
                BoundaryFact(
                    'nexus_summary_inference'
                    if invocation_kind == 'summary_inference'
                    else 'nexus_compaction_resume_explicit',
                    InputMessage(len(messages) - 1),
                    invocation_kind,
                    'compaction',
                    from_request_context=True,
                    at_request_edge=True,
                )
            )
        compacted_index = next(
            (index for index in range(len(messages) - 1, -1, -1) if messages[index].get('_is_compacted_summary')),
            None,
        )
        if compacted_index is not None:
            facts.append(
                BoundaryFact(
                    'nexus_compaction_resume',
                    InputMessage(compacted_index),
                    'compaction_resume',
                    'compaction',
                    at_request_edge=compacted_index == len(messages) - 1,
                )
            )
        return tuple(facts)

    @staticmethod
    def _empty_request_boundary_facts(meta):
        invocation_kind = meta.get('invocation_kind')
        if invocation_kind not in ('summary_inference', 'compaction_resume'):
            return ()
        return (
            BoundaryFact(
                'nexus_summary_inference'
                if invocation_kind == 'summary_inference'
                else 'nexus_compaction_resume_explicit',
                OutputMessage(),
                invocation_kind,
                'compaction',
                from_request_context=True,
                at_request_edge=True,
            ),
        )


NEXUS_POLICY = NexusScaffoldPolicy()
